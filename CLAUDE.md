# CLAUDE.md

## Project Overview

**Course**: 10-x23 Generative AI (Spring 2026)
**Title**: Knowing When to Generate: Selective Generation with Conformal Abstention for Human Preference Alignment
**Repo owner**: Tianyang Zhou (tzhou6) — Role 1 / Person 1

We build a selective text-to-image generation framework: generate K candidate images per prompt, score them with a human-preference reward model (PickScore), and accept/reject via a conformally calibrated threshold. The system abstains when no candidate is good enough, providing a distribution-free quality guarantee on accepted outputs.

## Current Milestone

**Midway Executive Summary** — due Monday, April 13 2026 at 11:59pm (3-4 pages).
Requires: baseline results, skeleton tables/plots, approach description in near-final form.

## My Role (Role 1 / Person 1): Environment and Baseline Pipeline Owner

### Deliverables for Midway Report
1. **Environment setup** — Conda or Apptainer with all dependencies (diffusers, transformers, Pick-a-Pic dataset, PickScore, ImageReward)
2. **End-to-end generation/scoring pipeline** — given a prompt, generate K images with SDXL (or SDXL-Turbo), score each with PickScore, return the top-1
3. **Standardized I/O format** — consistent data format for prompts, generated images, scores, so Person 2 can plug in threshold/conformal logic
4. **Top-1 baseline result** at non-trivial dataset scale — always accept the best-scoring candidate, report SelQual and AccRate

### Not Responsible For
- Running all later ablations or threshold/conformal experiments (Person 2)
- Writing the midway report draft or aggregating results (Person 3)

## Architecture

```
Prompt (from Pick-a-Pic)
  -> SDXL generator (K candidates)
  -> PickScore (selection scorer, ranks candidates)
  -> Top-1 selection (baseline) / Threshold + Conformal (Person 2)
  -> Accept or ABSTAIN
  -> ImageReward (external evaluation scorer, measures final quality)
```

## Key Models
- **Generator**: SDXL or SDXL-Turbo (pretrained, ~2.6B params)
- **Selection scorer**: PickScore (CLIP-H finetuned on Pick-a-Pic)
- **Evaluation scorer**: ImageReward (separate model to avoid circular evaluation)

## Evaluation Metrics
- **SelQual** = fraction of accepted images that are good (primary)
- **AccRate** = fraction of prompts where system returns an image

## Dataset
- **Pick-a-Pic** — text prompts with human preference annotations
- Prompts are used as input; human annotations validate quality signal

## Compute Environment
- Platform: PSC Bridges-2 (or similar HPC), Linux
- Track GPU hours, hardware specs, and cost for the Thought-Experiment on Compute section

## Conventions
- Record runtime, hardware, and resource usage for every experiment
- Use standardized output format so results can be handed off to Person 2 and Person 3
