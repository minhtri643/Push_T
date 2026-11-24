"""Entry point for training Stage E6a:
End-to-End ResNet-18 + Spatial-Softmax + DiT Action Diffusion Policy.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from pusht_wm_dit.paths import chdir_root  # noqa: E402
chdir_root()

from experiments.E6_end2end_vision_diffusion.code_E6a_resnet_spatial_direct.train import main

if __name__ == "__main__":
    main()
