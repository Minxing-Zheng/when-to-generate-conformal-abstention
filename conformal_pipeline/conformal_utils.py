from __future__ import annotations

import json
import math
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import beta as scipy_beta


HF_BASE_URL = "https://huggingface.co/datasets/tyzhou42/when-to-generate/resolve/main"

DEFAULT_DOWNLOADS = {
    "prompts": ("data/prompts.json", "prompts.json"),
    "metadata": ("outputs/run_k10/metadata.json", "run_k10/metadata.json"),
    "results": ("outputs/run_k10/results.jsonl", "run_k10/results.jsonl"),
}

OPTIONAL_DOWNLOADS = {
    "images": ("outputs/run_k10/images.tar.gz", "run_k10/images.tar.gz"),
}

SCORE_KEY_MAP = {
    "image_reward": "image_reward_scores",
    "pickapick": "pickscore_scores",
    "pickscore": "pickscore_scores",
    "clip": "self_clip_scores",
    "self_clip": "self_clip_scores",
    "latent_norm": "latent_norms",
}

# --- score direction for ranking ("higher" means larger is better) ---
SCORE_PREFERENCE = {
    "image_reward": "higher",
    "pickapick": "higher",
    "pickscore": "higher",
    "clip": "higher",
    "self_clip": "higher",
    "latent_norm": "higher",  # set to "lower" if you want smaller latent norm to be preferred
}



def _format_bytes(num_bytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def _download_file(
    url: str,
    destination: Path,
    overwrite: bool = False,
    show_progress: bool = True,
    chunk_size: int = 1024 * 1024,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        if show_progress:
            print(f"[skip] {destination.name} already exists at {destination}")
        return destination

    with urllib.request.urlopen(url) as response:
        total_bytes = response.headers.get("Content-Length")
        total_bytes = int(total_bytes) if total_bytes is not None else None
        downloaded_bytes = 0
        start_time = time.time()
        last_update = 0.0

        if show_progress:
            if total_bytes is None:
                print(f"[download] {destination.name} -> {destination}")
            else:
                print(
                    f"[download] {destination.name} "
                    f"({_format_bytes(total_bytes)}) -> {destination}"
                )

        with destination.open("wb") as handle:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded_bytes += len(chunk)

                if not show_progress:
                    continue

                now = time.time()
                if downloaded_bytes == total_bytes or now - last_update >= 0.2:
                    elapsed = max(now - start_time, 1e-9)
                    speed = downloaded_bytes / elapsed
                    if total_bytes:
                        progress = downloaded_bytes / total_bytes
                        filled = int(progress * 24)
                        bar = "#" * filled + "-" * (24 - filled)
                        message = (
                            f"\r    [{bar}] {progress:6.1%} "
                            f"{_format_bytes(downloaded_bytes)} / {_format_bytes(total_bytes)} "
                            f"at {_format_bytes(speed)}/s"
                        )
                    else:
                        message = (
                            f"\r    {_format_bytes(downloaded_bytes)} "
                            f"downloaded at {_format_bytes(speed)}/s"
                        )
                    sys.stdout.write(message)
                    sys.stdout.flush()
                    last_update = now

        if show_progress:
            elapsed = max(time.time() - start_time, 1e-9)
            avg_speed = downloaded_bytes / elapsed
            sys.stdout.write(
                f"\r    done: {_format_bytes(downloaded_bytes)} "
                f"in {elapsed:.1f}s at {_format_bytes(avg_speed)}/s\n"
            )
            sys.stdout.flush()
    return destination


def download_when_to_generate(
    root_dir: str | Path,
    include_images: bool = False,
    overwrite: bool = False,
    show_progress: bool = True,
) -> dict[str, Path]:
    root_dir = Path(root_dir)
    downloaded = {}

    for key, (remote_path, local_path) in DEFAULT_DOWNLOADS.items():
        downloaded[key] = _download_file(
            f"{HF_BASE_URL}/{remote_path}",
            root_dir / local_path,
            overwrite=overwrite,
            show_progress=show_progress,
        )

    if include_images:
        remote_path, local_path = OPTIONAL_DOWNLOADS["images"]
        downloaded["images"] = _download_file(
            f"{HF_BASE_URL}/{remote_path}",
            root_dir / local_path,
            overwrite=overwrite,
            show_progress=show_progress,
        )

    return downloaded


def extract_tar_gz(
    archive_path: str | Path,
    destination_dir: str | Path | None = None,
    overwrite: bool = False,
    show_progress: bool = True,
) -> Path:
    archive_path = Path(archive_path)
    if destination_dir is None:
        destination_dir = archive_path.parent
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()
        total_members = len(members)
        start_time = time.time()

        if show_progress:
            print(f"[extract] {archive_path.name} -> {destination_dir}")

        for idx, member in enumerate(members, start=1):
            target_path = destination_dir / member.name
            if target_path.exists() and not overwrite:
                continue
            tar.extract(member, path=destination_dir)

            if show_progress and (idx == total_members or idx % 250 == 0):
                elapsed = max(time.time() - start_time, 1e-9)
                rate = idx / elapsed
                progress = idx / max(total_members, 1)
                filled = int(progress * 24)
                bar = "#" * filled + "-" * (24 - filled)
                sys.stdout.write(
                    f"\r    [{bar}] {progress:6.1%} "
                    f"{idx}/{total_members} members at {rate:.1f} files/s"
                )
                sys.stdout.flush()

        if show_progress:
            elapsed = max(time.time() - start_time, 1e-9)
            sys.stdout.write(
                f"\r    done: extracted {total_members} members in {elapsed:.1f}s\n"
            )
            sys.stdout.flush()

    return destination_dir


def load_results(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open() as handle:
            return [json.loads(line) for line in handle if line.strip()]

    if path.suffix == ".json":
        try:
            with path.open() as handle:
                data = json.load(handle)
        except json.JSONDecodeError:
            # Some exported runs are newline-delimited JSON even with a .json extension.
            with path.open() as handle:
                return [json.loads(line) for line in handle if line.strip()]
    else:
        with path.open() as handle:
            data = json.load(handle)

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
        return data["results"]
    raise ValueError(f"Unsupported results format in {path}")


def _canonical_score_name(name: str) -> str:
    key = str(name).strip().lower()
    if key not in SCORE_KEY_MAP:
        raise KeyError(f"Unknown score '{name}'. Available: {sorted(SCORE_KEY_MAP)}")
    return key


def _aligned_scores(raw_scores: np.ndarray, preference: str) -> np.ndarray:
    if preference == "higher":
        return raw_scores
    if preference == "lower":
        return -raw_scores
    raise ValueError(f"preference must be 'higher' or 'lower', got: {preference}")


def _selected_features(
    aligned_scores: np.ndarray,
    selected_idx: int,
    temperature: float = 1.0,
    minmax_fixed_epsilon: float = 0.10,
) -> dict:
    score_selected = float(aligned_scores[selected_idx])
    score_min = float(aligned_scores.min())
    score_max = float(aligned_scores.max())
    score_mean = float(aligned_scores.mean())
    score_std = float(aligned_scores.std())
    score_range = float(score_max - score_min)
    score_sorted = np.sort(aligned_scores)
    second_best = float(score_sorted[-2]) if len(score_sorted) >= 2 else math.nan
    margin = float(score_selected - second_best) if len(score_sorted) >= 2 else math.nan
    minmax = (score_selected - score_min) / (score_range + 1e-8) if score_range > 0 else 1.0
    minmax_fixed_noise = (
        (score_selected - score_min) / (score_range + float(minmax_fixed_epsilon))
        if score_range > 0
        else 1.0
    )
    zscore = (score_selected - score_mean) / (score_std + 1e-8) if score_std > 0 else 0.0

    shifted = aligned_scores - np.max(aligned_scores)
    probs = np.exp(shifted / temperature)
    probs = probs / probs.sum()
    softmax_score = float(probs[selected_idx])

    return {
        "selected_score_raw": score_selected,
        "selected_score_minmax": float(minmax),
        "selected_score_minmax_fixed_noise": float(minmax_fixed_noise),
        "selected_score_zscore": float(zscore),
        "selected_score_margin": float(margin) if not math.isnan(margin) else math.nan,
        "selected_score_softmax": softmax_score,
        "score_min": score_min,
        "score_max": score_max,
        "score_mean": score_mean,
        "score_std": score_std,
        "score_range": score_range,
        "score_second_best": second_best,
    }


def build_prompt_table_for_baseline(
    records: list[dict],
    run_dir: Path,
    baseline_score: str,
    ground_truth_score: str,
    label_top_k: int = 1,
    softmax_temperature: float = 1.0,
    minmax_fixed_epsilon: float = 0.10,
) -> pd.DataFrame:
    baseline_name = _canonical_score_name(baseline_score)
    gt_name = _canonical_score_name(ground_truth_score)

    baseline_key = SCORE_KEY_MAP[baseline_name]
    gt_key = SCORE_KEY_MAP[gt_name]
    baseline_pref = SCORE_PREFERENCE[baseline_name]
    gt_pref = SCORE_PREFERENCE[gt_name]

    rows = []
    for record in records:
        baseline_raw = np.asarray(record[baseline_key], dtype=float)
        gt_raw = np.asarray(record[gt_key], dtype=float)
        k_value = int(record["K"])
        top_k = max(1, min(int(label_top_k), k_value))

        if len(baseline_raw) != k_value or len(gt_raw) != k_value:
            raise ValueError(
                f"Prompt {record.get('prompt_idx')} has inconsistent score lengths: "
                f"K={k_value}, baseline={len(baseline_raw)}, gt={len(gt_raw)}"
            )

        baseline_aligned = _aligned_scores(baseline_raw, baseline_pref)
        gt_aligned = _aligned_scores(gt_raw, gt_pref)

        selected_idx = int(np.argmax(baseline_aligned))
        top_gt_indices = set(np.argsort(-gt_aligned)[:top_k].tolist())
        selected_label = int(selected_idx in top_gt_indices)
        gt_best_idx = int(np.argmax(gt_aligned))

        selected_features = _selected_features(
            baseline_aligned,
            selected_idx=selected_idx,
            temperature=softmax_temperature,
            minmax_fixed_epsilon=minmax_fixed_epsilon,
        )

        rows.append(
            {
                "prompt_idx": int(record["prompt_idx"]),
                "prompt": record["prompt"],
                "K": k_value,
                "baseline_score": baseline_name,
                "ground_truth_score": gt_name,
                "selected_idx": selected_idx,
                "ground_truth_best_idx": gt_best_idx,
                "top_ground_truth_indices": sorted(top_gt_indices),
                "selected_label": selected_label,
                "label_top_k": top_k,
                "selected_ground_truth_value": float(gt_aligned[selected_idx]),
                "ground_truth_best_value": float(gt_aligned[gt_best_idx]),
                "ground_truth_gap_to_best": float(gt_aligned[gt_best_idx] - gt_aligned[selected_idx]),
                "baseline_scores_raw": baseline_raw.tolist(),
                "baseline_scores_aligned": baseline_aligned.tolist(),
                "ground_truth_scores_raw": gt_raw.tolist(),
                "ground_truth_scores_aligned": gt_aligned.tolist(),
                "image_paths": [str(run_dir / p) for p in record["image_paths"]],
                "generation_time_sec": float(record.get("generation_time_sec", math.nan)),
                "scoring_time_sec": float(record.get("scoring_time_sec", math.nan)),
                **selected_features,
            }
        )

    return pd.DataFrame(rows).sort_values("prompt_idx").reset_index(drop=True)


def build_candidate_table(
    records: Iterable[dict],
    run_dir: str | Path,
    pickscore_top_k_label: int = 1,
) -> pd.DataFrame:
    rows = []
    run_dir = Path(run_dir)

    for record in records:
        reward_scores = np.asarray(record["image_reward_scores"], dtype=float)
        pick_scores = np.asarray(record["pickscore_scores"], dtype=float)
        k_value = int(record["K"])
        top_k = max(1, min(int(pickscore_top_k_label), k_value))
        reward_top_idx = int(np.argmax(reward_scores))
        pick_top_idx = int(np.argmax(pick_scores))
        top_pick_indices = set(np.argsort(-pick_scores)[:top_k].tolist())

        for candidate_idx, (reward_score, pick_score, image_path) in enumerate(
            zip(reward_scores, pick_scores, record["image_paths"])
        ):
            rows.append(
                {
                    "prompt_idx": int(record["prompt_idx"]),
                    "prompt": record["prompt"],
                    "candidate_idx": candidate_idx,
                    "reward_score": float(reward_score),
                    "pickscore": float(pick_score),
                    "is_reward_top": int(candidate_idx == reward_top_idx),
                    "is_pick_top": int(candidate_idx == pick_top_idx),
                    "Y_ik": int(candidate_idx in top_pick_indices),
                    "label_top_k": top_k,
                    "image_path": str(run_dir / image_path),
                }
            )

    return pd.DataFrame(rows).sort_values(["prompt_idx", "candidate_idx"]).reset_index(drop=True)


def build_prompt_table(
    records: Iterable[dict],
    run_dir: str | Path,
    pickscore_top_k_label: int = 1,
    reward_softmax_temperature: float = 1.0,
) -> pd.DataFrame:
    rows = []
    run_dir = Path(run_dir)

    for record in records:
        reward_scores = np.asarray(record["image_reward_scores"], dtype=float)
        pick_scores = np.asarray(record["pickscore_scores"], dtype=float)
        k_value = int(record["K"])
        top_k = max(1, min(int(pickscore_top_k_label), k_value))
        reward_top_idx = int(np.argmax(reward_scores))
        pick_top_idx = int(np.argmax(pick_scores))
        top_pick_indices = set(np.argsort(-pick_scores)[:top_k].tolist())
        reward_sorted = np.sort(reward_scores)
        reward_selected = float(reward_scores[reward_top_idx])
        reward_min = float(reward_scores.min())
        reward_max = float(reward_scores.max())
        reward_mean = float(reward_scores.mean())
        reward_std = float(reward_scores.std())
        reward_second_best = float(reward_sorted[-2]) if len(reward_sorted) >= 2 else math.nan
        reward_margin = (
            float(reward_selected - reward_second_best) if len(reward_sorted) >= 2 else math.nan
        )
        reward_range = reward_max - reward_min
        selected_reward_minmax = (
            (reward_selected - reward_min) / (reward_range + 1e-3) if reward_range > 0 else 1.0
        )
        selected_reward_zscore = (
            (reward_selected - reward_mean) / (reward_std + 1e-8) if reward_std > 0 else 0.0
        )
        shifted_rewards = reward_scores - np.max(reward_scores)
        reward_probs = np.exp(shifted_rewards / reward_softmax_temperature)
        reward_probs = reward_probs / reward_probs.sum()
        selected_reward_softmax = float(reward_probs[reward_top_idx])
        rows.append(
            {
                "prompt_idx": int(record["prompt_idx"]),
                "prompt": record["prompt"],
                "K": int(record["K"]),
                "selected_idx": reward_top_idx,
                "selected_reward": reward_selected,
                "selected_reward_raw": reward_selected,
                "selected_reward_minmax": float(selected_reward_minmax),
                "selected_reward_zscore": float(selected_reward_zscore),
                "selected_reward_margin": float(reward_margin) if not math.isnan(reward_margin) else math.nan,
                "selected_reward_softmax": selected_reward_softmax,
                "selected_pickscore": float(pick_scores[reward_top_idx]),
                "best_pick_idx": pick_top_idx,
                "top_pick_indices": sorted(top_pick_indices),
                "best_pickscore": float(pick_scores[pick_top_idx]),
                "best_pick_reward": float(reward_scores[pick_top_idx]),
                "selected_label": int(reward_top_idx in top_pick_indices),
                "label": int(reward_top_idx in top_pick_indices),
                "label_top_k": top_k,
                "pickscore_gap_to_best": float(pick_scores[pick_top_idx] - pick_scores[reward_top_idx]),
                "reward_margin": reward_margin,
                "reward_min": reward_min,
                "reward_max": reward_max,
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "reward_range": reward_range,
                "reward_second_best": reward_second_best,
                "generation_time_sec": float(record.get("generation_time_sec", math.nan)),
                "scoring_time_sec": float(record.get("scoring_time_sec", math.nan)),
                "reward_scores": reward_scores.tolist(),
                "pickscore_scores": pick_scores.tolist(),
                "Y_i_vector": [int(idx in top_pick_indices) for idx in range(k_value)],
                "image_paths": [str(run_dir / image_path) for image_path in record["image_paths"]],
            }
        )

    return pd.DataFrame(rows).sort_values("prompt_idx").reset_index(drop=True)


def ensure_prompt_table_columns(prompt_df: pd.DataFrame) -> pd.DataFrame:
    prompt_df = prompt_df.copy()

    if "label_top_k" not in prompt_df.columns:
        prompt_df["label_top_k"] = 1

    if "top_pick_indices" not in prompt_df.columns and "pickscore_scores" in prompt_df.columns:
        prompt_df["top_pick_indices"] = [
            sorted(
                set(
                    np.argsort(-np.asarray(pick_scores, dtype=float))[
                        : max(1, min(int(top_k), len(pick_scores)))
                    ].tolist()
                )
            )
            for pick_scores, top_k in zip(prompt_df["pickscore_scores"], prompt_df["label_top_k"])
        ]

    if "selected_label" not in prompt_df.columns:
        if {"selected_idx", "top_pick_indices"}.issubset(prompt_df.columns):
            prompt_df["selected_label"] = [
                int(int(selected_idx) in set(top_pick_indices))
                for selected_idx, top_pick_indices in zip(
                    prompt_df["selected_idx"], prompt_df["top_pick_indices"]
                )
            ]
        elif "label" in prompt_df.columns:
            prompt_df["selected_label"] = prompt_df["label"].astype(int)
        elif {"selected_idx", "best_pick_idx"}.issubset(prompt_df.columns):
            prompt_df["selected_label"] = (
                prompt_df["selected_idx"].astype(int) == prompt_df["best_pick_idx"].astype(int)
            ).astype(int)
        else:
            raise KeyError("Cannot infer selected_label from prompt_df")

    if "label" not in prompt_df.columns:
        prompt_df["label"] = prompt_df["selected_label"].astype(int)

    if "Y_i_vector" not in prompt_df.columns:
        if {"K", "top_pick_indices"}.issubset(prompt_df.columns):
            prompt_df["Y_i_vector"] = [
                [int(idx in set(top_pick_indices)) for idx in range(int(k_value))]
                for k_value, top_pick_indices in zip(prompt_df["K"], prompt_df["top_pick_indices"])
            ]
        elif {"K", "best_pick_idx"}.issubset(prompt_df.columns):
            prompt_df["Y_i_vector"] = [
                [int(idx == int(best_pick_idx)) for idx in range(int(k_value))]
                for k_value, best_pick_idx in zip(prompt_df["K"], prompt_df["best_pick_idx"])
            ]
        else:
            raise KeyError("Cannot infer Y_i_vector from prompt_df")

    if "selected_reward_raw" not in prompt_df.columns and "selected_reward" in prompt_df.columns:
        prompt_df["selected_reward_raw"] = prompt_df["selected_reward"].astype(float)

    if "reward_min" not in prompt_df.columns and "reward_scores" in prompt_df.columns:
        prompt_df["reward_min"] = [
            float(np.min(np.asarray(reward_scores, dtype=float))) for reward_scores in prompt_df["reward_scores"]
        ]
    if "reward_max" not in prompt_df.columns and "reward_scores" in prompt_df.columns:
        prompt_df["reward_max"] = [
            float(np.max(np.asarray(reward_scores, dtype=float))) for reward_scores in prompt_df["reward_scores"]
        ]
    if "reward_mean" not in prompt_df.columns and "reward_scores" in prompt_df.columns:
        prompt_df["reward_mean"] = [
            float(np.mean(np.asarray(reward_scores, dtype=float))) for reward_scores in prompt_df["reward_scores"]
        ]
    if "reward_std" not in prompt_df.columns and "reward_scores" in prompt_df.columns:
        prompt_df["reward_std"] = [
            float(np.std(np.asarray(reward_scores, dtype=float))) for reward_scores in prompt_df["reward_scores"]
        ]
    if "reward_range" not in prompt_df.columns and {"reward_min", "reward_max"}.issubset(prompt_df.columns):
        prompt_df["reward_range"] = prompt_df["reward_max"] - prompt_df["reward_min"]
    if "reward_second_best" not in prompt_df.columns and "reward_scores" in prompt_df.columns:
        prompt_df["reward_second_best"] = [
            float(np.sort(np.asarray(reward_scores, dtype=float))[-2]) if len(reward_scores) >= 2 else math.nan
            for reward_scores in prompt_df["reward_scores"]
        ]
    if "selected_reward_minmax" not in prompt_df.columns and {
        "selected_reward_raw",
        "reward_min",
        "reward_range",
    }.issubset(prompt_df.columns):
        prompt_df["selected_reward_minmax"] = [
            (selected_reward - reward_min) / (reward_range + 1e-8) if reward_range > 0 else 1.0
            for selected_reward, reward_min, reward_range in zip(
                prompt_df["selected_reward_raw"], prompt_df["reward_min"], prompt_df["reward_range"]
            )
        ]
    if "selected_reward_zscore" not in prompt_df.columns and {
        "selected_reward_raw",
        "reward_mean",
        "reward_std",
    }.issubset(prompt_df.columns):
        prompt_df["selected_reward_zscore"] = [
            (selected_reward - reward_mean) / (reward_std + 1e-8) if reward_std > 0 else 0.0
            for selected_reward, reward_mean, reward_std in zip(
                prompt_df["selected_reward_raw"], prompt_df["reward_mean"], prompt_df["reward_std"]
            )
        ]
    if "selected_reward_margin" not in prompt_df.columns and {
        "selected_reward_raw",
        "reward_second_best",
    }.issubset(prompt_df.columns):
        prompt_df["selected_reward_margin"] = [
            float(selected_reward - reward_second_best) if not math.isnan(reward_second_best) else math.nan
            for selected_reward, reward_second_best in zip(
                prompt_df["selected_reward_raw"], prompt_df["reward_second_best"]
            )
        ]
    if "selected_reward_softmax" not in prompt_df.columns and {
        "reward_scores",
        "selected_idx",
    }.issubset(prompt_df.columns):
        prompt_df["selected_reward_softmax"] = [
            float(
                (
                    np.exp(reward_scores - np.max(reward_scores))
                    / np.exp(reward_scores - np.max(reward_scores)).sum()
                )[int(selected_idx)]
            )
            for reward_scores, selected_idx in zip(
                [
                    np.asarray(reward_scores, dtype=float)
                    for reward_scores in prompt_df["reward_scores"]
                ],
                prompt_df["selected_idx"],
            )
        ]

    return prompt_df


def ensure_candidate_table_columns(candidate_df: pd.DataFrame) -> pd.DataFrame:
    candidate_df = candidate_df.copy()

    if "label_top_k" not in candidate_df.columns:
        candidate_df["label_top_k"] = 1

    if "Y_ik" not in candidate_df.columns:
        if "is_pick_top" in candidate_df.columns:
            candidate_df["Y_ik"] = candidate_df["is_pick_top"].astype(int)
        else:
            raise KeyError("Cannot infer Y_ik from candidate_df")

    return candidate_df


def summarize_prompt_table(prompt_df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "num_prompts": int(len(prompt_df)),
            "K": int(prompt_df["K"].iloc[0]),
            "reward_top_matches_pick_top": float(prompt_df["selected_label"].mean()),
            "mean_selected_reward": float(prompt_df["selected_reward"].mean()),
            "mean_selected_pickscore": float(prompt_df["selected_pickscore"].mean()),
            "mean_best_pickscore": float(prompt_df["best_pickscore"].mean()),
            "mean_pickscore_gap_to_best": float(prompt_df["pickscore_gap_to_best"].mean()),
            "mean_reward_margin": float(prompt_df["reward_margin"].mean()),
        }
    )


def calibration_test_split(
    prompt_df: pd.DataFrame,
    calibration_fraction: float = 0.5,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1")

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(prompt_df))
    calibration_size = int(round(len(prompt_df) * calibration_fraction))
    calibration_idx = np.sort(permutation[:calibration_size])
    test_idx = np.sort(permutation[calibration_size:])

    return (
        prompt_df.iloc[calibration_idx].reset_index(drop=True),
        prompt_df.iloc[test_idx].reset_index(drop=True),
    )


def build_selective_calibration_table(
    prompt_df: pd.DataFrame,
    score_column: str = "selected_reward",
) -> pd.DataFrame:
    prompt_df = ensure_prompt_table_columns(prompt_df)
    if score_column not in prompt_df.columns:
        available_score_columns = [
            column
            for column in [
                "selected_reward",
                "selected_reward_raw",
                "selected_reward_minmax",
                "selected_reward_zscore",
                "selected_reward_margin",
                "selected_reward_softmax",
            ]
            if column in prompt_df.columns
        ]
        raise KeyError(
            f"score_column '{score_column}' not found. "
            f"Available score columns: {available_score_columns}"
        )
    calibration_df = prompt_df.copy()
    calibration_df["S_i"] = calibration_df[score_column].astype(float)
    calibration_df["Z_i"] = calibration_df["selected_label"].astype(int)
    calibration_df["error_i"] = 1 - calibration_df["Z_i"]
    calibration_df["score_column"] = score_column
    return calibration_df


def _kl_divergence_bernoulli(p: float, q: float) -> float:
    p = float(np.clip(p, 0.0, 1.0))
    q = float(np.clip(q, 0.0, 1.0))

    if q <= 0.0:
        return 0.0 if p <= 0.0 else math.inf
    if q >= 1.0:
        return 0.0 if p >= 1.0 else math.inf

    if p == 0.0:
        return -math.log(max(1.0 - q, 1e-15))
    if p == 1.0:
        return -math.log(max(q, 1e-15))
    return p * math.log(max(p / q, 1e-15)) + (1.0 - p) * math.log(
        max((1.0 - p) / (1.0 - q), 1e-15)
    )


def _ucb_risk_hoeffding(empirical_risk: float, accepted_count: int, grid_size: int, delta: float) -> float:
    return empirical_risk + math.sqrt(math.log(grid_size / delta) / (2 * accepted_count))


def _ucb_risk_clopper_pearson(
    bad_accepts: int,
    accepted_count: int,
    grid_size: int,
    delta: float,
) -> float:
    if accepted_count == 0:
        return math.inf
    if bad_accepts >= accepted_count:
        return 1.0

    per_threshold_delta = float(delta) / float(grid_size)
    confidence = 1.0 - per_threshold_delta
    # One-sided exact binomial upper bound.
    return float(scipy_beta.ppf(confidence, bad_accepts + 1, accepted_count - bad_accepts))


def _ucb_risk_kl_binomial(
    empirical_risk: float,
    accepted_count: int,
    grid_size: int,
    delta: float,
    tol: float = 1e-10,
    max_iter: int = 80,
) -> float:
    if accepted_count == 0:
        return math.inf

    if empirical_risk >= 1.0:
        return 1.0

    rhs = math.log(grid_size / delta)
    low = float(np.clip(empirical_risk, 0.0, 1.0))
    high = 1.0

    # If even q=1 satisfies inequality (rare), return 1.
    if accepted_count * _kl_divergence_bernoulli(low, high) <= rhs:
        return 1.0

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        lhs = accepted_count * _kl_divergence_bernoulli(empirical_risk, mid)
        if lhs <= rhs:
            low = mid
        else:
            high = mid
        if high - low <= tol:
            break

    return float(np.clip(low, empirical_risk, 1.0))


def selective_risk_sweep(
    scores: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    delta: float,
    thresholds: np.ndarray | None = None,
    bound_type: str = "hoeffding",
) -> pd.DataFrame:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    bound_type = str(bound_type).lower()
    supported_bounds = {"hoeffding", "clopper_pearson", "kl_binomial"}
    if bound_type not in supported_bounds:
        raise ValueError(f"Unsupported bound_type '{bound_type}'. Choose from: {sorted(supported_bounds)}")

    if thresholds is None:
        thresholds = np.sort(np.unique(scores))[::-1]
    else:
        thresholds = np.asarray(thresholds, dtype=float)
        thresholds = np.sort(np.unique(thresholds))[::-1]
    grid_size = max(len(thresholds), 1)
    rows = []

    for threshold in thresholds:
        accepted = scores >= threshold
        accepted_count = int(accepted.sum())  # A(tau)
        if accepted_count == 0:
            bad_accepts = 0
            empirical_risk = math.nan
            ucb_risk = math.inf
            selective_accuracy = math.nan
        else:
            accepted_labels = labels[accepted]
            bad_accepts = int((1 - accepted_labels).sum())  # E(tau)
            empirical_risk = bad_accepts / accepted_count
            if bound_type == "hoeffding":
                ucb_risk = _ucb_risk_hoeffding(
                    empirical_risk=empirical_risk,
                    accepted_count=accepted_count,
                    grid_size=grid_size,
                    delta=delta,
                )
            elif bound_type == "clopper_pearson":
                ucb_risk = _ucb_risk_clopper_pearson(
                    bad_accepts=bad_accepts,
                    accepted_count=accepted_count,
                    grid_size=grid_size,
                    delta=delta,
                )
            else:
                ucb_risk = _ucb_risk_kl_binomial(
                    empirical_risk=empirical_risk,
                    accepted_count=accepted_count,
                    grid_size=grid_size,
                    delta=delta,
                )
            selective_accuracy = float(accepted_labels.mean())

        rows.append(
            {
                "threshold": float(threshold),
                "A_tau": accepted_count,
                "E_tau": bad_accepts,
                "empirical_risk": float(empirical_risk) if not math.isnan(empirical_risk) else math.nan,
                "ucb_risk": float(ucb_risk) if math.isfinite(ucb_risk) else math.inf,
                "selective_accuracy": (
                    float(selective_accuracy) if not math.isnan(selective_accuracy) else math.nan
                ),
                "acceptance_rate": accepted_count / len(scores),
                "alpha": float(alpha),
                "delta": float(delta),
                "bound_type": bound_type,
            }
        )

    return pd.DataFrame(rows)


def calibrate_selective_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    delta: float = 0.1,
    thresholds: np.ndarray | None = None,
    bound_type: str = "hoeffding",
) -> dict:
    sweep_df = selective_risk_sweep(
        scores,
        labels,
        alpha=alpha,
        delta=delta,
        thresholds=thresholds,
        bound_type=bound_type,
    )
    valid_rows = sweep_df.loc[(sweep_df["A_tau"] > 0) & (sweep_df["ucb_risk"] <= alpha)].copy()

    if valid_rows.empty:
        chosen_threshold = math.inf
        chosen_reason = (
            f"No threshold satisfied the UCB risk target on calibration (bound_type={bound_type})."
        )
    else:
        chosen_threshold = float(valid_rows["threshold"].min())
        chosen_reason = (
            "Selected the most permissive threshold whose UCB risk stayed below alpha "
            f"(bound_type={bound_type})."
        )

    metrics = evaluate_threshold(scores, labels, chosen_threshold)
    return {
        "threshold": chosen_threshold,
        "alpha": float(alpha),
        "delta": float(delta),
        "bound_type": str(bound_type).lower(),
        "reason": chosen_reason,
        "sweep": sweep_df,
        "calibration_metrics": metrics,
    }


def evaluate_selective_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    accepted = scores >= threshold
    accepted_count = int(accepted.sum())
    total = int(len(scores))

    if accepted_count == 0:
        return {
            "threshold": float(threshold),
            "accepted": 0,
            "total": total,
            "acceptance_rate": 0.0,
            "selective_accuracy": math.nan,
            "selective_risk": math.nan,
        }

    accepted_labels = labels[accepted]
    selective_accuracy = float(accepted_labels.mean())
    return {
        "threshold": float(threshold),
        "accepted": accepted_count,
        "total": total,
        "acceptance_rate": accepted_count / total,
        "selective_accuracy": selective_accuracy,
        "selective_risk": 1.0 - selective_accuracy,
    }


def run_selective_alpha_sweep(
    calibration_df: pd.DataFrame,
    test_df: pd.DataFrame,
    alpha_values: Iterable[float],
    delta: float = 0.1,
    bound_type: str = "hoeffding",
) -> pd.DataFrame:
    rows = []
    calibration_table = build_selective_calibration_table(calibration_df)
    test_table = build_selective_calibration_table(test_df)
    calibration_scores = calibration_table["S_i"].to_numpy()
    calibration_labels = calibration_table["Z_i"].to_numpy()
    test_scores = test_table["S_i"].to_numpy()
    test_labels = test_table["Z_i"].to_numpy()

    for alpha in alpha_values:
        calibrated = calibrate_selective_threshold(
            calibration_scores,
            calibration_labels,
            alpha=float(alpha),
            delta=delta,
            bound_type=bound_type,
        )
        test_metrics = evaluate_selective_threshold(test_scores, test_labels, calibrated["threshold"])
        rows.append(
            {
                "alpha": float(alpha),
                "delta": float(delta),
                "bound_type": str(bound_type).lower(),
                "threshold": float(calibrated["threshold"]),
                "calibration_acceptance_rate": calibrated["calibration_metrics"]["acceptance_rate"],
                "calibration_selective_accuracy": calibrated["calibration_metrics"]["selective_accuracy"],
                "test_acceptance_rate": test_metrics["acceptance_rate"],
                "test_selective_accuracy": test_metrics["selective_accuracy"],
                "test_selective_risk": test_metrics["selective_risk"],
                "accepted_test_prompts": int(test_metrics["accepted"]),
            }
        )

    return pd.DataFrame(rows)


def threshold_sweep(
    scores: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    delta: float,
    bound_type: str = "hoeffding",
) -> pd.DataFrame:
    return selective_risk_sweep(
        scores=scores,
        labels=labels,
        alpha=alpha,
        delta=delta,
        bound_type=bound_type,
    )


def calibrate_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    delta: float = 0.1,
    bound_type: str = "hoeffding",
) -> dict:
    return calibrate_selective_threshold(
        scores=scores,
        labels=labels,
        alpha=alpha,
        delta=delta,
        bound_type=bound_type,
    )


def evaluate_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    return evaluate_selective_threshold(scores=scores, labels=labels, threshold=threshold)


def run_alpha_sweep(
    calibration_df: pd.DataFrame,
    test_df: pd.DataFrame,
    alpha_values: Iterable[float],
    delta: float = 0.1,
    bound_type: str = "hoeffding",
) -> pd.DataFrame:
    return run_selective_alpha_sweep(
        calibration_df=calibration_df,
        test_df=test_df,
        alpha_values=alpha_values,
        delta=delta,
        bound_type=bound_type,
    )


def sample_disagreements(
    prompt_df: pd.DataFrame,
    num_examples: int = 3,
    sort_by: str = "pickscore_gap_to_best",
) -> pd.DataFrame:
    subset = prompt_df.loc[prompt_df["selected_label"] == 0].copy()
    if sort_by in subset.columns:
        subset = subset.sort_values(sort_by, ascending=False)
    return subset.head(num_examples).reset_index(drop=True)


def plot_prompt_images(prompt_row: pd.Series, figsize_scale: float = 3.0):
    from pathlib import Path as _Path

    import matplotlib.pyplot as plt
    from PIL import Image

    reward_top_idx = int(prompt_row["selected_idx"])
    pick_top_idx = int(prompt_row["best_pick_idx"])
    image_paths = prompt_row["image_paths"]
    reward_scores = prompt_row["reward_scores"]
    pickscore_scores = prompt_row["pickscore_scores"]
    num_images = len(image_paths)

    fig, axes = plt.subplots(1, num_images, figsize=(figsize_scale * num_images, figsize_scale))
    if num_images == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        image = Image.open(_Path(image_paths[idx]))
        ax.imshow(image)

        border_color = "black"
        if idx == reward_top_idx and idx == pick_top_idx:
            border_color = "gold"
        elif idx == reward_top_idx:
            border_color = "tab:red"
        elif idx == pick_top_idx:
            border_color = "tab:green"

        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(4 if border_color != "black" else 1.0)

        ax.set_title(
            f"k={idx}\nR={reward_scores[idx]:.2f}\nP={pickscore_scores[idx]:.2f}",
            fontsize=10,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(prompt_row["prompt"], fontsize=12)
    fig.tight_layout()
    return fig
