"""Analyze pipeline results: compute SelQual/AccRate at various thresholds."""
import argparse
import json
import statistics
from pathlib import Path


def load_results(path):
    records = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "error" not in r:
                records.append(r)
    return records


def compute_quantiles(values, percentiles):
    s = sorted(values)
    return {p: s[int(len(s) * p / 100)] for p in percentiles}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="outputs/run_001")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    records = load_results(results_dir / "results.jsonl")
    print(f"Loaded {len(records)} records from {results_dir}")

    # Load metadata
    with open(results_dir / "metadata.json") as f:
        meta = json.load(f)

    # --- Collect scores ---
    # All K candidates
    all_ir = [s for r in records for s in r["image_reward_scores"]]
    all_ps = [s for r in records for s in r["pickscore_scores"]]
    # Top-1 only
    top1_ir = [r["top1_image_reward"] for r in records]
    top1_ps = [r["top1_pickscore"] for r in records]

    n_prompts = len(records)
    K = meta["K"]

    # --- Score distributions ---
    pcts = [10, 25, 50, 75, 90]
    all_ps_q = compute_quantiles(all_ps, pcts)
    all_ir_q = compute_quantiles(all_ir, pcts)
    top1_ps_q = compute_quantiles(top1_ps, pcts)
    top1_ir_q = compute_quantiles(top1_ir, pcts)

    print(f"\n{'='*60}")
    print(f"Run config: K={K}, prompts={n_prompts}, steps={meta['num_inference_steps']}")
    print(f"GPU: {meta['gpu']}, wall={meta['total_wall_clock_sec']:.0f}s "
          f"(gen={meta['total_generation_sec']:.0f}s, score={meta['total_scoring_sec']:.0f}s)")
    print(f"{'='*60}")

    print(f"\n--- Score Distributions ---")
    print(f"{'':>20} {'Mean':>8} {'P10':>8} {'P25':>8} {'P50':>8} {'P75':>8} {'P90':>8}")
    print(f"{'IR (all '+str(len(all_ir))+')':>20} {statistics.mean(all_ir):>8.3f} "
          + " ".join(f"{all_ir_q[p]:>8.3f}" for p in pcts))
    print(f"{'IR (top-1 '+str(n_prompts)+')':>20} {statistics.mean(top1_ir):>8.3f} "
          + " ".join(f"{top1_ir_q[p]:>8.3f}" for p in pcts))
    print(f"{'PS (all '+str(len(all_ps))+')':>20} {statistics.mean(all_ps):>8.3f} "
          + " ".join(f"{all_ps_q[p]:>8.3f}" for p in pcts))
    print(f"{'PS (top-1 '+str(n_prompts)+')':>20} {statistics.mean(top1_ps):>8.3f} "
          + " ".join(f"{top1_ps_q[p]:>8.3f}" for p in pcts))

    # --- SelQual at various PickScore thresholds ---
    # Thresholds based on ALL candidates distribution
    print(f"\n--- SelQual (Top-1 Baseline, AccRate=1.0) ---")
    print(f"  Eval scorer: PickScore | Selection scorer: ImageReward")
    print(f"  Threshold q = percentile of ALL {len(all_ps)} candidates' PickScore")
    print(f"  {'Threshold':>12} {'q value':>10} {'SelQual':>10} {'Good/Total':>12}")

    threshold_results = []
    for pct in [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90]:
        q = all_ps_q.get(pct) or compute_quantiles(all_ps, [pct])[pct]
        good = sum(1 for s in top1_ps if s >= q)
        selqual = good / n_prompts
        print(f"  {'P'+str(pct):>12} {q:>10.3f} {selqual:>10.3f} {good:>5}/{n_prompts}")
        threshold_results.append({
            "threshold_percentile": pct,
            "threshold_value": round(q, 4),
            "selqual": round(selqual, 4),
            "good": good,
            "total": n_prompts,
            "accrate": 1.0,
        })

    # --- Also compute SelQual using ImageReward as eval (cross-check) ---
    print(f"\n--- SelQual cross-check (using ImageReward > 0 as 'good') ---")
    ir_good = sum(1 for s in top1_ir if s > 0)
    print(f"  IR > 0: {ir_good}/{n_prompts} = {ir_good/n_prompts:.3f}")

    # --- LaTeX table row ---
    q50 = all_ps_q[50]
    good50 = sum(1 for s in top1_ps if s >= q50)
    selqual50 = good50 / n_prompts

    print(f"\n--- LaTeX table row (q=P50) ---")
    print(f"Top-1 & {K} & 1.000 & {selqual50:.3f} & {statistics.mean(top1_ir):.3f} & {statistics.mean(top1_ps):.3f} \\\\")

    # --- Save analysis ---
    analysis = {
        "n_prompts": n_prompts,
        "K": K,
        "all_candidates": len(all_ps),
        "score_distributions": {
            "image_reward_all": {"mean": round(statistics.mean(all_ir), 4), **{f"P{p}": round(all_ir_q[p], 4) for p in pcts}},
            "image_reward_top1": {"mean": round(statistics.mean(top1_ir), 4), **{f"P{p}": round(top1_ir_q[p], 4) for p in pcts}},
            "pickscore_all": {"mean": round(statistics.mean(all_ps), 4), **{f"P{p}": round(all_ps_q[p], 4) for p in pcts}},
            "pickscore_top1": {"mean": round(statistics.mean(top1_ps), 4), **{f"P{p}": round(top1_ps_q[p], 4) for p in pcts}},
        },
        "selqual_by_threshold": threshold_results,
        "timing": {
            "total_wall_clock_sec": meta["total_wall_clock_sec"],
            "total_generation_sec": meta["total_generation_sec"],
            "total_scoring_sec": meta["total_scoring_sec"],
        },
    }
    out_path = results_dir / "analysis.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved analysis to {out_path}")


if __name__ == "__main__":
    main()
