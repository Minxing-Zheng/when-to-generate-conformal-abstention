"""Upload a single experiment run to a HuggingFace dataset repo.

Packages images into a single tar.gz (much faster than uploading 10000+ small files),
then uploads tar.gz + results.jsonl + metadata.json to the specified repo.

Example:
    python upload_to_hf.py --repo_id tyzhou42/when-to-generate --run_dir outputs/run_k10_3000
"""
import os
import argparse
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

load_dotenv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True,
                        help="HuggingFace repo id, e.g. 'tyzhou42/when-to-generate'")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Local run directory (e.g. outputs/run_k10_3000)")
    parser.add_argument("--include_readme", action="store_true", default=False,
                        help="Also upload data/prompts.json and README.md")
    parser.add_argument("--private", action="store_true", default=False)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not found. Set it in .env.")

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"{run_dir} not found")

    results_file = run_dir / "results.jsonl"
    metadata_file = run_dir / "metadata.json"
    images_dir = run_dir / "images"
    tar_file = run_dir / "images.tar.gz"

    for f in [results_file, metadata_file]:
        if not f.exists():
            raise FileNotFoundError(f"{f} not found")

    # --- 1. Package images into tar.gz ---
    if images_dir.exists() and not tar_file.exists():
        print(f"Packaging {images_dir} into tar.gz...")
        subprocess.run(
            ["tar", "czf", "images.tar.gz", "images/"],
            cwd=run_dir, check=True,
        )
        print(f"Created {tar_file} ({tar_file.stat().st_size / 1024**3:.2f} GB)")
    elif tar_file.exists():
        print(f"Using existing {tar_file}")
    else:
        print(f"Warning: no images directory found at {images_dir}")

    # --- 2. Setup HF API ---
    api = HfApi(token=token)
    create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private,
                token=token, exist_ok=True)
    print(f"Repo: https://huggingface.co/datasets/{args.repo_id}")

    # Files to upload, stored at the same relative path in the HF repo
    uploads = [
        (results_file, str(results_file)),
        (metadata_file, str(metadata_file)),
    ]
    if tar_file.exists():
        uploads.append((tar_file, str(tar_file)))

    if args.include_readme:
        for extra in ["README.md", "data/prompts.json"]:
            if Path(extra).exists():
                uploads.append((Path(extra), extra))

    # --- 3. Upload ---
    for local, remote in uploads:
        size_mb = local.stat().st_size / 1024**2
        print(f"Uploading {local} ({size_mb:.1f} MB) -> {remote}...")
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=remote,
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=f"Upload {remote}",
        )
        print(f"  Done.")

    print(f"\nAll uploaded. View at: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
