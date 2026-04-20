"""Main pipeline: generate images, score with ImageReward + PickScore, save results."""
import os
os.environ["HF_HOME"] = "/ocean/projects/cis250290p/tzhou6/hf_cache/"

import argparse
import json
import time
import torch
from pathlib import Path
from tqdm import tqdm

PROJECT_DIR = "/ocean/projects/cis250290p/tzhou6"
IR_CACHE = f"{PROJECT_DIR}/hf_cache/ImageReward"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts_file", type=str, default="data/prompts.json")
    p.add_argument("--num_prompts", type=int, default=500)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--output_dir", type=str, default="outputs/run_001")
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_inference_steps", type=int, default=1)
    return p.parse_args()


def load_prompts(path, num_prompts, start_idx):
    with open(path) as f:
        data = json.load(f)
    prompts = data["prompts"][start_idx : start_idx + num_prompts]
    print(f"Loaded {len(prompts)} prompts (start_idx={start_idx}) from {path}")
    return prompts


def load_generator(device="cuda"):
    from diffusers import AutoPipelineForText2Image
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to(device)
    return pipe


def load_image_reward(device="cuda"):
    import ImageReward as RM
    model = RM.load("ImageReward-v1.0", device=device, download_root=IR_CACHE)
    return model


def load_pickscore(device="cuda"):
    from transformers import AutoProcessor, AutoModel
    processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to(device)
    return model, processor


def score_pickscore_batch(prompt, images, ps_model, ps_processor, device="cuda"):
    image_inputs = ps_processor(images=images, return_tensors="pt", padding=True).to(device)
    text_inputs = ps_processor(
        text=prompt, return_tensors="pt", padding=True, truncation=True, max_length=77
    ).to(device)
    with torch.no_grad():
        img_embs = ps_model.get_image_features(**image_inputs)
        img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)
        txt_embs = ps_model.get_text_features(**text_inputs)
        txt_embs = txt_embs / txt_embs.norm(dim=-1, keepdim=True)
        scores = (ps_model.logit_scale.exp() * (txt_embs @ img_embs.T))[0]
    return scores.cpu().tolist()


def load_self_clip(device="cuda"):
    """CLIP-L: same model weights SDXL uses for text conditioning."""
    from transformers import CLIPModel, CLIPProcessor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval().to(device)
    return model, processor


def score_self_clip_batch(prompt, images, clip_model, clip_processor, device="cuda"):
    """Self-CLIP score: cosine similarity in CLIP-L embedding space (same space SDXL is conditioned on)."""
    image_inputs = clip_processor(images=images, return_tensors="pt").to(device)
    text_inputs = clip_processor(
        text=prompt, return_tensors="pt", padding=True, truncation=True, max_length=77
    ).to(device)
    with torch.no_grad():
        img_embs = clip_model.get_image_features(**image_inputs)
        img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)
        txt_embs = clip_model.get_text_features(**text_inputs)
        txt_embs = txt_embs / txt_embs.norm(dim=-1, keepdim=True)
        scores = (txt_embs @ img_embs.T)[0]  # raw cosine similarity
    return scores.cpu().tolist()


def main():
    args = parse_args()

    # --- Setup output dir ---
    out_dir = Path(args.output_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"

    # --- Load prompts ---
    prompts = load_prompts(args.prompts_file, args.num_prompts, args.start_idx)
    if not prompts:
        print("No prompts to process.")
        return

    # --- Load models ---
    print("Loading models...")
    t0 = time.time()
    pipe = load_generator()
    ir_model = load_image_reward()
    ps_model, ps_processor = load_pickscore()
    clip_model, clip_processor = load_self_clip()
    print(f"Models loaded in {time.time() - t0:.1f}s")
    print(f"GPU mem: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")

    # --- Main loop ---
    total_gen_time = 0.0
    total_score_time = 0.0
    wall_start = time.time()

    with open(results_path, "a") as fout:
        for i, prompt in enumerate(tqdm(prompts, desc="Processing")):
            prompt_idx = args.start_idx + i
            try:
                # Generate K images; capture final latents via callback for latent_norm self-score
                final_latents = {}
                def capture_latents(p, step, timestep, kwargs):
                    final_latents["z"] = kwargs["latents"].detach()
                    return kwargs

                t_gen = time.time()
                generators = [
                    torch.Generator(device="cuda").manual_seed(args.seed + prompt_idx * args.K + k)
                    for k in range(args.K)
                ]
                images = pipe(
                    prompt=[prompt] * args.K,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=0.0,
                    height=512,
                    width=512,
                    generator=generators,
                    callback_on_step_end=capture_latents,
                    callback_on_step_end_tensor_inputs=["latents"],
                ).images
                gen_time = time.time() - t_gen
                total_gen_time += gen_time

                # Latent norm: L2 norm per candidate (truly generator-internal "self" signal)
                latent_norms = final_latents["z"].flatten(1).norm(dim=1).cpu().tolist()

                # Score with ImageReward (all K)
                t_score = time.time()
                ir_rankings, ir_rewards = ir_model.inference_rank(prompt, images)
                ir_scores = [float(r) for r in ir_rewards]

                # Score with PickScore (all K, batch)
                ps_scores = score_pickscore_batch(prompt, images, ps_model, ps_processor)

                # Self-CLIP score (CLIP-L, same weights SDXL uses for text conditioning)
                self_clip_scores = score_self_clip_batch(prompt, images, clip_model, clip_processor)

                score_time = time.time() - t_score
                total_score_time += score_time

                # Top-1 by ImageReward
                top1_idx = ir_rankings.index(1)

                # Save images
                image_paths = []
                for k, img in enumerate(images):
                    fname = f"images/{prompt_idx:04d}_k{k}.png"
                    img.save(out_dir / fname)
                    image_paths.append(fname)

                # Write result
                record = {
                    "prompt_idx": prompt_idx,
                    "prompt": prompt,
                    "K": args.K,
                    "seed": args.seed,
                    "image_reward_scores": ir_scores,
                    "image_reward_rankings": ir_rankings,
                    "pickscore_scores": ps_scores,
                    "self_clip_scores": self_clip_scores,
                    "latent_norms": latent_norms,
                    "top1_idx": top1_idx,
                    "top1_image_reward": ir_scores[top1_idx],
                    "top1_pickscore": ps_scores[top1_idx],
                    "top1_self_clip": self_clip_scores[top1_idx],
                    "top1_latent_norm": latent_norms[top1_idx],
                    "image_paths": image_paths,
                    "generation_time_sec": round(gen_time, 3),
                    "scoring_time_sec": round(score_time, 3),
                }
                fout.write(json.dumps(record) + "\n")
                fout.flush()

            except Exception as e:
                # Log error and continue
                error_record = {
                    "prompt_idx": prompt_idx,
                    "prompt": prompt,
                    "error": str(e),
                }
                fout.write(json.dumps(error_record) + "\n")
                fout.flush()
                print(f"\nError at prompt {prompt_idx}: {e}")

            finally:
                # Free CPU memory
                if "images" in dir():
                    del images

            # Print stats every 20 prompts
            if (i + 1) % 20 == 0:
                elapsed = time.time() - wall_start
                rate = (i + 1) / elapsed
                eta = (len(prompts) - i - 1) / rate
                print(
                    f"\n[{i+1}/{len(prompts)}] "
                    f"gen={total_gen_time:.1f}s score={total_score_time:.1f}s "
                    f"rate={rate:.2f} prompts/s ETA={eta/60:.1f}min"
                )

    # --- Save metadata ---
    wall_total = time.time() - wall_start
    metadata = {
        "generator": "stabilityai/sdxl-turbo",
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": 0.0,
        "resolution": 512,
        "K": args.K,
        "num_prompts": len(prompts),
        "start_idx": args.start_idx,
        "prompts_file": args.prompts_file,
        "selection_scorer": "ImageReward-v1.0",
        "eval_scorer": "PickScore_v1",
        "self_scorers": ["self_clip (openai/clip-vit-large-patch14)", "latent_norm"],
        "base_seed": args.seed,
        "gpu": torch.cuda.get_device_name(0),
        "total_wall_clock_sec": round(wall_total, 1),
        "total_generation_sec": round(total_gen_time, 1),
        "total_scoring_sec": round(total_score_time, 1),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone. Wall time: {wall_total/60:.1f}min")
    print(f"  Generation: {total_gen_time:.1f}s")
    print(f"  Scoring: {total_score_time:.1f}s")
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
