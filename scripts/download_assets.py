"""Download the trained models and processed data from the GitHub Release of this repository.

    python scripts/download_assets.py                    # final system only (~90 MB): enough for the demo and the benchmark
    python scripts/download_assets.py --what vision      # + vision World Models (stage E5)
    python scripts/download_assets.py --what history     # + models of every earlier stage (E0-E4 round 0)
    python scripts/download_assets.py --what data        # + lerobot/pusht with recovered block pose (needed to retrain)
    python scripts/download_assets.py --what all

Every file is checked against the SHA-256 in scripts/release/assets.json before it is unpacked into the repository
(checkpoints/..., data/...). The raw lerobot/pusht dataset itself comes from Hugging Face:
python scripts/1_data/download_dataset.py.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pusht_wm_dit.paths import chdir_root  # noqa: E402
chdir_root()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
import urllib.request  # noqa: E402
import zipfile  # noqa: E402

MANIFEST = "scripts/release/assets.json"
WHAT = {"final": ["pusht_checkpoints_final.zip"], "vision": ["pusht_checkpoints_vision.zip"],
        "history": ["pusht_checkpoints_history.zip"], "data": ["pusht_data_processed.zip"],
        "legacy": ["pusht_data_legacy.zip"], "videos": ["final_policy.mp4", "baseline_vs_final.mp4"]}
CACHE = os.path.join("data", "downloads")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(url, dst, size, retries=5):
    for attempt in range(1, retries + 1):
        try:
            tmp = dst + ".part"
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                done = 0
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
                    done += len(b)
                    print(f"\r  {os.path.basename(dst)}: {done / 1e6:7.1f} / {size / 1e6:.1f} MB", end="", flush=True)
            os.replace(tmp, dst)
            print()
            return
        except Exception as e:
            print(f"\n  attempt {attempt}/{retries} failed: {e!r}")
            if attempt == retries:
                raise
            time.sleep(3 * attempt)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--what", nargs="*", default=["final"], choices=list(WHAT) + ["all"])
    args = p.parse_args()
    m = json.load(open(MANIFEST, encoding="utf-8"))
    base = f"https://github.com/{m['repository']}/releases/download/{m['tag']}"
    assets = {a["name"]: a for a in m["assets"]}
    names = [n for w in (list(WHAT) if "all" in args.what else args.what) for n in WHAT[w]]
    os.makedirs(CACHE, exist_ok=True)
    for name in names:
        a = assets[name]
        local = os.path.join(CACHE, name)
        if not (os.path.exists(local) and sha256(local) == a["sha256"]):
            print(f"downloading {name} ({a['bytes'] / 1e6:.0f} MB): {a['description']}")
            fetch(f"{base}/{name}", local, a["bytes"])
        if sha256(local) != a["sha256"]:
            sys.exit(f"SHA-256 mismatch for {name}: delete {local} and try again")
        if name.endswith(".zip"):
            with zipfile.ZipFile(local) as z:
                z.extractall(".")
            print(f"  unpacked {name} ({a['files']} files), SHA-256 ok")
        else:
            os.makedirs(os.path.join("results", "videos"), exist_ok=True)
            shutil.copyfile(local, os.path.join("results", "videos", name))
            print(f"  saved results/videos/{name}, SHA-256 ok")
    print("done")


if __name__ == "__main__":
    main()
