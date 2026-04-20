# Selective Generation with Conformal Abstention

## 0. PSC Bridges-2 Quick Start

All experiments run on **PSC Bridges-2**. A single H100 GPU is sufficient -- the full pipeline (500 prompts, K=4) takes about 6 minutes.

### 0.1 Request an interactive GPU node

```bash
salloc -J Interact --mem=120GB --gres=gpu:h100-80:1 --partition=GPU-shared --ntasks-per-node=4 --time=03:00:00
```

This gives you 1x H100 80GB for 3 hours. Once allocated, you'll be dropped into a shell on the GPU node.

### 0.2 Development environment

Two options for connecting to the GPU node:

1. **PSC OnDemand web portal** -- Access VS Code in browser via the Bridges-2 OnDemand interface (recommended for getting started quickly).
2. **Local VS Code + SSH** -- Use VS Code Remote-SSH extension to connect to Bridges-2, then SSH again to the allocated GPU node from the terminal.

**Important: keep your allocation alive.** If you connect via VS Code or any remote session, the GPU node may be reclaimed if it detects no activity. Run this in a terminal on the GPU node to prevent that:

```bash
while true; do sleep 300; echo "keepalive"; done
```

---

## 1. Environment Setup

Platform: PSC Bridges-2, RHEL 8.10, glibc 2.28, NVIDIA H100 80GB

### 1.1 Set your project directory

Each team member should set their own project directory. Replace the placeholder below:

```bash
# Set this to YOUR project directory on /ocean/ (not /jet/home/)
export PROJECT_DIR="/ocean/projects/<your_allocation>/<your_username>"
```

### 1.2 Create conda environment

Run `setup_env.sh` (you may need to edit paths inside) or follow the steps below.
For a full list of installed packages, see `requirements.txt` (pip) and `conda-env-export.txt` (conda).

```bash
ENV_PREFIX="${PROJECT_DIR}/envs/gen-eval"

# Create env (Python 3.10)
conda create -p "${ENV_PREFIX}" python=3.10 -y

# Activate
conda activate "${ENV_PREFIX}"

# PyTorch (cu124 wheels bundle CUDA runtime, no need for module load)
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124

# Core dependencies (transformers MUST be >=4.38 and <=4.44)
#   - >=4.38: required by SDXL-Turbo via diffusers
#   - <=4.44: ImageReward uses BLIP, which breaks on transformers >=4.45
pip install transformers==4.44.2 diffusers==0.30.3 accelerate==0.34.2 \
            datasets safetensors pillow tqdm

# ImageReward (install last; fairscale may need compilation)
module load cuda/12.4.0
pip install image-reward

# ImageReward has an undeclared dependency on OpenAI CLIP
pip install git+https://github.com/openai/CLIP.git

# Verify
python -c "import torch, diffusers, transformers, ImageReward, datasets; print('OK')"
```

### 1.3 Activate (for subsequent sessions)

```bash
conda activate "${PROJECT_DIR}/envs/gen-eval"
export HF_HOME="${PROJECT_DIR}/hf_cache/"
```

---

## 2. Storage Paths (IMPORTANT)

All data, models, and outputs MUST be stored under `/ocean/`, NOT under `/jet/home/`.
Home directory (`/jet/home/`) has strict quota limits and will run out of space.

| What | Path |
|------|------|
| Conda env | `${PROJECT_DIR}/envs/gen-eval` |
| HuggingFace cache | `${PROJECT_DIR}/hf_cache/` |
| ImageReward model | `${PROJECT_DIR}/hf_cache/ImageReward/` |
| Experiment outputs | `outputs/` (within this repo) |

Set the HF cache env var before running any script:
```bash
export HF_HOME="${PROJECT_DIR}/hf_cache/"
```

When using ImageReward in code, you MUST pass `download_root` explicitly:
```python
import ImageReward as RM
model = RM.load("ImageReward-v1.0", device="cuda",
                download_root=os.path.join(os.environ["PROJECT_DIR"], "hf_cache/ImageReward"))
```

Otherwise it defaults to `~/.cache/ImageReward/` which is on `/jet/home/` and will fill up your quota.

---

## 3. Code Files

| File | Purpose |
|------|---------|
| `setup_env.sh` | One-shot environment creation script. Installs all dependencies in correct order. Edit the paths at the top to match your `PROJECT_DIR`. |
| `prepare_prompts.py` | Extracts unique prompts from Pick-a-Pic dataset (`yuvalkirstain/PickaPic-rankings`), filters by token length (default: <=35 tokens, the ImageReward limit), and saves to local JSON. Only needs to run once. |
| `generate_and_score.py` | Main pipeline. Loads SDXL-Turbo + ImageReward + PickScore onto one GPU (~12GB VRAM). For each prompt: generates K images, scores all K with both scorers, selects top-1 by ImageReward, saves images + scores to JSONL. Supports resume via `--start_idx`. Edit `PROJECT_DIR` and `IR_CACHE` at the top to match your paths. |
| `analyze_results.py` | Post-hoc analysis. Computes SelQual and AccRate at multiple PickScore thresholds (P10 through P90). Outputs score distributions, LaTeX table row, and `analysis.json`. |

---

## 4. Usage

The pipeline has three steps. `data/prompts.json` is already committed, so **if you only need to run experiments on the existing prompts, skip straight to Step 2**.

**Main entry point: `generate_and_score.py`** -- this is the script you'll run most often.

### Step 1: Prepare prompts (run once, already done)

```bash
python prepare_prompts.py --output data/prompts.json --max_prompts 5000 --max_tokens 35
```

| Arg | Default | Description |
|-----|---------|-------------|
| `--output` | `data/prompts.json` | Output file path |
| `--max_prompts` | 5000 | Keep at most N prompts |
| `--max_tokens` | 35 | Filter out prompts longer than this (ImageReward limit) |

### Step 2: Generate and score

```bash
python generate_and_score.py \
  --prompts_file data/prompts.json \
  --num_prompts 500 \
  --K 4 \
  --output_dir outputs/run_001 \
  --seed 42 \
  --num_inference_steps 1
```

| Arg | Default | Description |
|-----|---------|-------------|
| `--prompts_file` | `data/prompts.json` | Path to prepared prompts |
| `--num_prompts` | 500 | How many prompts to process |
| `--K` | 4 | Candidates per prompt |
| `--output_dir` | `outputs/run_001` | Where to save results |
| `--seed` | 42 | Base random seed (deterministic generation) |
| `--num_inference_steps` | 1 | SDXL-Turbo denoising steps (1-4, 1 is fastest, 4 shows no quality gain) |
| `--start_idx` | 0 | Resume from this prompt index (for crash recovery) |

### Step 3: Analyze results

```bash
python analyze_results.py --results_dir outputs/run_001
```

### Output format: `results.jsonl`

Each line is one prompt (Person 2: this is your primary input):

```json
{
  "prompt_idx": 0,
  "prompt": "a cat wizard",
  "K": 4,
  "seed": 42,
  "image_reward_scores": [0.58, 0.27, -1.41, -2.03],
  "image_reward_rankings": [1, 2, 3, 4],
  "pickscore_scores": [21.3, 20.8, 19.2, 18.9],
  "self_clip_scores": [0.318, 0.256, 0.277, 0.241],
  "latent_norms": [105.1, 107.2, 79.3, 75.9],
  "top1_idx": 0,
  "top1_image_reward": 0.58,
  "top1_pickscore": 21.3,
  "top1_self_clip": 0.318,
  "top1_latent_norm": 105.1,
  "image_paths": ["images/0000_k0.png", ...],
  "generation_time_sec": 0.8,
  "scoring_time_sec": 1.2
}
```

Key fields for Person 2:
- `image_reward_scores`: all K scores from the selection scorer -- use these for threshold/conformal calibration
- `pickscore_scores`: all K scores from the evaluation scorer -- use these to compute SelQual
- `self_clip_scores`: all K cosine similarities in the CLIP-L embedding space (same encoder SDXL uses for conditioning) -- a **generator-internal** heuristic, useful as a self-score baseline for conformal prediction
- `latent_norms`: all K L2 norms of the final denoised latents -- a **purely internal** signal, no external model involved
- `top1_idx`: which candidate was selected by ImageReward (baseline)

---

## 5. Project Layout

```
when-to-generate-conformal-abstention/
  # --- Code (in git) ---
  generate_and_score.py     # Main entry point: generate images + score
  prepare_prompts.py        # Data preparation (already run, see data/)
  analyze_results.py        # Result analysis
  setup_env.sh              # Environment setup script
  requirements.txt          # pip freeze of the working environment
  conda-env-export.txt      # conda environment export
  CLAUDE.md                 # Claude Code config
  README.md                 # This file

  # --- Data (in git) ---
  data/
    prompts.json            # 5000 prompts, <=35 tokens each

  # --- Documentation (in git) ---
  project.md                # Course project guidelines
  proposal.tex              # Project proposal
  genai-project-role-assignment.md  # Role assignments

  # --- Experiment results (JSON in git, images NOT in git) ---
  outputs/
    run_001/                # Baseline: 500 prompts, K=4, 1 inference step
      results.jsonl         #   Per-prompt scores (in git)
      metadata.json         #   Run config and timing (in git)
      analysis.json         #   SelQual at various thresholds (in git)
      images/               #   2000 PNG files, ~721MB (NOT in git)
    run_steps4/             # Comparison: 500 prompts, K=4, 4 inference steps
      results.jsonl
      metadata.json
      analysis.json
      images/               #   (NOT in git)

  # --- Ignored ---
  logs/                     # Slurm job logs
```

### Models used

| Role | Model | Params | VRAM |
|------|-------|--------|------|
| Generator | `stabilityai/sdxl-turbo` | ~2.6B | ~6.6GB (fp16) |
| Selection scorer | `ImageReward-v1.0` (THUDM) | ~1.1B | ~1.7GB |
| Evaluation scorer | `yuvalkirstain/PickScore_v1` (CLIP ViT-H) | ~1B | ~3.7GB |
| **Total** | | | **~12GB** |

All three models load simultaneously on a single H100 (80GB).

---

## 6. What Has Been Done (Role 1)

### 6.1 Completed work

1. **Environment**: Set up conda env with pinned dependency versions. Key constraint: `transformers` must be `>=4.38, <=4.44` — outside this window either SDXL-Turbo or ImageReward breaks.
2. **Data preparation**: Extracted 8269 unique prompts from Pick-a-Pic, filtered to 7222 prompts with <=35 tokens (ImageReward's max input length), saved 5000 to `data/prompts.json`.
3. **Pipeline**: Built end-to-end `generate_and_score.py` — generates K candidates per prompt, scores all K with both ImageReward and PickScore, saves everything to JSONL.
4. **Baseline experiments**: Ran two experiments (1-step and 4-step SDXL-Turbo). Conclusion: 1 step and 4 steps produce nearly identical quality (PickScore mean difference < 0.01), so 1 step is sufficient.

### 6.2 Baseline results (Top-1, AccRate = 1.0)

The Top-1 baseline always returns the best candidate by ImageReward, never abstains. This is the baseline Person 2's conformal method should improve upon.

**run_001** (1 inference step, 500 prompts, K=4):

| Scorer | Subset | Mean | P25 | P50 | P75 |
|--------|--------|------|-----|-----|-----|
| ImageReward | All 2000 candidates | 0.925 | 0.399 | 1.137 | 1.686 |
| ImageReward | Top-1 (500) | 1.241 | 0.881 | 1.449 | 1.813 |
| PickScore | All 2000 candidates | 22.288 | 21.410 | 22.356 | 23.155 |
| PickScore | Top-1 (500) | 22.461 | 21.698 | 22.497 | 23.265 |

Top-1 selection by ImageReward lifts the mean IR score from 0.925 to 1.241, confirming that selecting among K=4 candidates improves quality.

### 6.3 How the quality threshold is defined

To compute **SelQual** (selective quality), we need to define what counts as a "good" image. We use PickScore (the evaluation scorer) with a threshold `q`:

```
Y(x, I) = 1  if  PickScore(x, I) >= q    ("good")
Y(x, I) = 0  otherwise                    ("bad")
```

The threshold `q` is set as a **percentile of the PickScore distribution across all 2000 candidates** (not just top-1). This gives a relative definition of quality:

| Threshold | q value (PickScore) | SelQual (Top-1 baseline) | Interpretation |
|-----------|---------------------|--------------------------|----------------|
| P25 | 21.41 | 0.798 | "Better than bottom 25%" |
| **P50** | **22.36** | **0.560** | **"Better than median" (primary)** |
| P75 | 23.16 | 0.280 | "Top 25% quality" |

**P50 is the primary threshold** — it asks "of the images we return, how many are better than the median candidate?" The Top-1 baseline achieves SelQual=0.560 at P50, meaning 56% of returned images are above median quality. Person 2's conformal method should improve this by abstaining on low-quality prompts (at the cost of lower AccRate).

We report all thresholds (P10 through P90) in `analysis.json` so you can choose whichever is most appropriate.

### 6.4 Timing and compute

| | run_001 (1 step) | run_steps4 (4 steps) |
|--|------------------|----------------------|
| Wall time | 363s (6.0 min) | 437s (7.3 min) |
| Generation | 85s | 150s |
| Scoring | 71s | 72s |
| GPU | H100 80GB | H100 80GB |
| GPU hours | ~0.1h | ~0.12h |
