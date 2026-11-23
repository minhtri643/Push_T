"""Maintainer tool: build the GitHub Release assets (checkpoints, processed data, videos) + their manifest.

    python scripts/release/package_release.py --out <folder outside the repo>

Writes one zip per asset group (each < 2 GiB, the GitHub limit per release file) and
scripts/release/assets.json (file name, size, SHA-256, content), which scripts/download_assets.py reads.

World Model checkpoints are published WITHOUT the optimizer state (132 MB -> 46 MB); nothing in the code reads it.
The SHA-256 of the original local file is recorded inside the published checkpoint ("stripped_from_sha256").
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from pusht_wm_dit.paths import chdir_root  # noqa: E402
chdir_root()

import argparse  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import zipfile  # noqa: E402

import torch  # noqa: E402

GROUPS = {  # asset -> (description, list of repo-relative paths or globs)
    "pusht_checkpoints_final.zip": ("final system (stage E4): world model, dynamics, DiT, inverse dynamics",
                                    ["checkpoints/final/*.pt"]),
    "pusht_checkpoints_vision.zip": ("vision World Models (stage E5): JEPA, V-JEPA 2, DINOv2, legacy DINOv2",
                                     ["checkpoints/E5_vision/*.pt", "checkpoints/E5_vision/legacy_dinov2_zarr_sim/*.pt"]),
    "pusht_checkpoints_history.zip": ("models of stages E0-E4 round 0 (for the evolution table)",
                                      ["checkpoints/E0_baseline/*.pt", "checkpoints/E1_augfix/*.pt", "checkpoints/E3_*/*.pt",
                                       "checkpoints/E4_hf_round0/*.pt"]),
    "pusht_data_processed.zip": ("lerobot/pusht with the block pose recovered from the images + decoded frames",
                                 ["data/pusht_hf.zarr/**", "data/lerobot_pusht/frames.npy"]),
    "pusht_data_legacy.zip": ("stages E1-E3 only: Diffusion Policy zarr + simulator interaction data",
                              ["data/legacy/**"]),
}
VIDEOS = ["results/videos/final_policy.mp4", "results/videos/baseline_vs_final.mp4"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def files_of(patterns):
    out = []
    for pat in patterns:
        out += [p for p in glob.glob(pat, recursive=True) if os.path.isfile(p)]
    return sorted(set(out))


def stripped_copy(path, tmp):
    """World Model checkpoints: drop the optimizer state, keep the weights and metadata."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not (isinstance(ck, dict) and "optimizer_state_dict" in ck):
        return path
    ck = {k: v for k, v in ck.items() if k != "optimizer_state_dict"}
    ck["stripped_from_sha256"] = sha256(path)
    out = os.path.join(tmp, hashlib.md5(path.encode()).hexdigest() + ".pt")
    torch.save(ck, out)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="folder for the release files (outside the repository)")
    p.add_argument("--tag", default="v1.0")
    p.add_argument("--repo", default="OWNER/REPO", help="GitHub <owner>/<repository> that hosts the release")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    manifest = {"repository": args.repo, "tag": args.tag, "assets": []}
    with tempfile.TemporaryDirectory() as tmp:
        for name, (desc, pats) in GROUPS.items():
            files = files_of(pats)
            if not files:
                print(f"skip {name}: no files")
                continue
            dst = os.path.join(args.out, name)
            with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
                for f in files:
                    src = stripped_copy(f, tmp) if f.startswith("checkpoints/") else f
                    z.write(src, f.replace("\\", "/"))
            size = os.path.getsize(dst)
            assert size < 2 * 1024 ** 3, f"{name} is {size / 1e9:.2f} GB, over the 2 GiB GitHub limit"
            manifest["assets"].append({"name": name, "description": desc, "bytes": size, "sha256": sha256(dst), "files": len(files)})
            print(f"{name}: {len(files)} files, {size / 1e6:.0f} MB")
    for v in VIDEOS:
        if os.path.exists(v):
            dst = os.path.join(args.out, os.path.basename(v))
            shutil.copyfile(v, dst)
            manifest["assets"].append({"name": os.path.basename(v), "description": "video", "bytes": os.path.getsize(dst), "sha256": sha256(dst), "files": 1})
    with open(os.path.join(args.out, "SHA256SUMS.txt"), "w") as f:
        f.writelines(f"{a['sha256']}  {a['name']}\n" for a in manifest["assets"])
    with open("scripts/release/assets.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {len(manifest['assets'])} assets to {args.out} and scripts/release/assets.json")


if __name__ == "__main__":
    main()
