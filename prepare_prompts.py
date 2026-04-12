"""Extract unique prompts from Pick-a-Pic and save locally."""
import os
os.environ["HF_HOME"] = "/ocean/projects/cis250290p/tzhou6/hf_cache/"

import argparse
import json
from datasets import load_dataset
from transformers import CLIPTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/prompts.json")
    parser.add_argument("--max_prompts", type=int, default=5000)
    parser.add_argument("--max_tokens", type=int, default=35,
                        help="Filter prompts longer than this (ImageReward limit=35)")
    args = parser.parse_args()

    print("Loading PickaPic-rankings...")
    ds = load_dataset("yuvalkirstain/PickaPic-rankings", split="train")
    print(f"Total rows: {len(ds)}")

    unique = sorted(set(p.strip() for p in ds["prompt"] if p and p.strip()))
    print(f"Unique non-empty prompts: {len(unique)}")

    # Filter by token length
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    filtered = [p for p in unique if len(tokenizer.encode(p)) <= args.max_tokens]
    print(f"After filtering (<= {args.max_tokens} tokens): {len(filtered)} ({len(filtered)/len(unique)*100:.1f}%)")

    prompts = filtered[: args.max_prompts]
    print(f"Keeping first {len(prompts)} prompts")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out = {
        "prompts": prompts,
        "source": "yuvalkirstain/PickaPic-rankings",
        "total_unique": len(unique),
        "after_token_filter": len(filtered),
        "max_tokens": args.max_tokens,
        "saved": len(prompts),
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
