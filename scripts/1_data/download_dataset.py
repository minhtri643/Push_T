"""Pipeline 1 (data): download the ONLY training data source, https://huggingface.co/datasets/lerobot/pusht.

    python scripts/1_data/download_dataset.py            # -> data/lerobot_pusht/ (about 7.7 MB)

206 human demonstrations, 25,650 frames (96x96 AV1 video), pusher position and actions. The dataset has no
block pose; scripts/1_data/fit_block_pose.py recovers it from the frames afterwards. Hugging Face downloads
are sometimes reset on Windows (WinError 10054), so the download is retried.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from pusht_wm_dit.paths import chdir_root  # noqa: E402
chdir_root()

import argparse  # noqa: E402
import time  # noqa: E402

REPO_ID = "lerobot/pusht"
OUT = "data/lerobot_pusht"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=OUT)
    p.add_argument("--retries", type=int, default=20)
    args = p.parse_args()
    from huggingface_hub import snapshot_download
    for attempt in range(1, args.retries + 1):
        try:
            path = snapshot_download(REPO_ID, repo_type="dataset", local_dir=args.out)
            break
        except Exception as e:  # connection resets are common; the download resumes where it stopped
            print(f"attempt {attempt}/{args.retries} failed: {e!r}")
            if attempt == args.retries:
                raise
            time.sleep(min(30, 2 * attempt))
    need = ["meta/info.json", "data/chunk-000/file-000.parquet", "videos/observation.image/chunk-000/file-000.mp4"]
    missing = [f for f in need if not os.path.exists(os.path.join(path, f))]
    if missing:
        sys.exit(f"download incomplete, missing: {missing}")
    print(f"lerobot/pusht ready in {path}")


if __name__ == "__main__":
    main()
