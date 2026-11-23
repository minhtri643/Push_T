"""Smoke test: does a fresh clone + `pip install -r requirements.txt` + `python scripts/download_assets.py` run?

    python tests/smoke_test.py          # CPU is enough, about 1-2 minutes

Checks: every module imports; every checkpoint path of the final system in configs/default.yaml exists; the final
policy (dit_wm) and the DiT + inverse-dynamics policy load and run a short closed-loop episode; the video drawing
works. Used by .github/workflows/smoke_test.yml on Linux, Windows and macOS.
"""

import glob
import importlib
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from pusht_wm_dit.paths import chdir_root, load_config  # noqa: E402

chdir_root()


def check(name, fn):
    try:
        fn()
        print(f"ok    {name}")
        return True
    except Exception as e:  # report every failure, not only the first
        print(f"FAIL  {name}: {e!r}")
        return False


def imports():
    for f in glob.glob("pusht_wm_dit/**/*.py", recursive=True):
        importlib.import_module(f[:-3].replace(os.sep, ".").replace("/", ".").replace(".__init__", ""))


def final_checkpoints():
    cfg = load_config()
    missing = [cfg[k]["save_path"] for k in ("world_model", "dynamics", "dit_planner", "inverse_dynamics")
               if not os.path.exists(cfg[k]["save_path"])]
    assert not missing, f"missing {missing}: run python scripts/download_assets.py"


def episode(kind):
    def run():
        import torch
        from pusht_wm_dit.envs.pusht_sim import PushTSim
        from pusht_wm_dit.policy import build_policy, run_episode
        torch.manual_seed(0)
        cfg = load_config()
        pol = build_policy(cfg, "cpu", kind=kind)
        sim = PushTSim(env_name=cfg["env"]["env_name"], render_size=tuple(cfg["env"]["render_size"]), max_steps=16)
        r = run_episode(sim, pol, seed=42, max_steps=16, exec_horizon=cfg["evaluation"]["exec_horizon"])
        sim.close()
        assert r["steps"] == 16 and 0.0 <= r["max_coverage"] <= 1.0, r
    return run


def video_drawing():
    spec = importlib.util.spec_from_file_location("make_video", os.path.join("scripts", "5_evaluation", "make_video.py"))
    mv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mv)
    from pusht_wm_dit.envs.pusht_sim import PushTSim
    sim = PushTSim(max_steps=10)
    obs, _ = sim.reset(seed=42)
    frame = mv.draw(obs, 240, title="smoke test", cov=obs["coverage"], peak=obs["coverage"])
    sim.close()
    assert frame.shape[1] == 240 and frame.shape[2] == 3


if __name__ == "__main__":
    results = [check("import every module", imports), check("final checkpoints present", final_checkpoints),
               check("final system (dit_wm) runs 16 steps on CPU", episode("dit_wm")),
               check("DiT + inverse dynamics (dit_id) runs 16 steps on CPU", episode("dit_id")),
               check("video drawing", video_drawing)]
    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)
