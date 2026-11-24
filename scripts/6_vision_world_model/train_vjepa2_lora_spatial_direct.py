"""Sub-stage E5e: V-JEPA 2 + LoRA (r=8) + SpatialSoftmax (K=16) + Direct Action Diffusion (No Inverse Dynamics).

Usage:
    python scripts/6_vision_world_model/train_vjepa2_lora_spatial_direct.py [--epochs 15] [--eval]
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from pusht_wm_dit.paths import chdir_root  # noqa: E402
chdir_root()

import argparse
from experiments.E5_vision_world_models.code_E5e_vjepa2_lora_spatial_direct.train import main as train_main
from experiments.E5_vision_world_models.code_E5e_vjepa2_lora_spatial_direct.eval import main as eval_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-only", action="store_true", help="only run closed-loop evaluation")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--smoke", action="store_true")
    args, unknown = parser.parse_known_args()

    if not args.eval_only:
        sys.argv = [sys.argv[0]] + (["--smoke"] if args.smoke else ["--epochs", str(args.epochs), "--batch-size", str(args.batch_size)])
        train_main()
    else:
        sys.argv = [sys.argv[0]]
        eval_main()
