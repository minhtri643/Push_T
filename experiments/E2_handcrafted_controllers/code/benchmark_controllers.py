"""Stage E2: re-measurement of the hand-written controllers on the standard paired benchmark.

Same 95 starts, same per-start random seeds, same 300-step / coverage > 0.95 protocol as
scripts/5_evaluation/benchmark.py, so every row can be paired with every other run in results/benchmark/.
The learned models are those of stage E1 (checkpoints/E1_augfix), as during the original E2 work.

    python experiments/E2_handcrafted_controllers/code/benchmark_controllers.py            # all configurations
    python experiments/E2_handcrafted_controllers/code/benchmark_controllers.py --configs guard tactical_best

Configurations (results/benchmark/E2_<name>.json). The plain DiT + inverse dynamics row is E1_augfix__dit_id.
    guard               DiT + ID, pusher targets filtered by the non-penetration guard
    lookahead           DiT + ID, optimal stopping with the old latent World Model (TRUNCATE / LOCK)
    guard_lookahead     both (the default of the old demo)
    tactical_default    3-phase ORIENT -> ALIGN -> PUSH controller, guard + lookahead on, locks at 88% / settle 80%
    tactical_no_lock    same without the locks
    tactical_best       no locks, no lookahead, guard on (best configuration of the original E2 review)
    tactical_no_guard   no locks, no lookahead, guard off
The tactical controller reads the true block state from the simulator (privileged information).
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from pusht_wm_dit.paths import chdir_root  # noqa: E402
chdir_root()

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from pusht_wm_dit.envs.pusht_sim import PushTSim  # noqa: E402
from pusht_wm_dit.paths import load_config  # noqa: E402
from pusht_wm_dit.policy import (GOAL_XY, SUCCESS_THRESHOLD, angle_error_to_goal, build_policy,  # noqa: E402
                                 postprocess_action, resolve_device, summarize)
from pusht_wm_dit.utils.normalizer import normalize_pos  # noqa: E402
from controllers.tactical_controller import TacticalHierarchicalController  # noqa: E402
from controllers.wm_lookahead import WorldModelLookaheadOptimizer, compute_safe_retreat  # noqa: E402

CKPT = "checkpoints/E1_augfix"
NEVER = 1.01   # a coverage threshold that is never reached = lock disabled
CONFIGS = {
    "guard": {"tactical": False, "guard": True, "lookahead": False},
    "lookahead": {"tactical": False, "guard": False, "lookahead": True},
    "guard_lookahead": {"tactical": False, "guard": True, "lookahead": True},
    "tactical_default": {"tactical": True, "guard": True, "lookahead": True, "lock": 0.88, "settle": 0.80},
    "tactical_no_lock": {"tactical": True, "guard": True, "lookahead": True, "lock": NEVER, "settle": NEVER},
    "tactical_best": {"tactical": True, "guard": True, "lookahead": False, "lock": NEVER, "settle": NEVER},
    "tactical_no_guard": {"tactical": True, "guard": False, "lookahead": False, "lock": NEVER, "settle": NEVER},
}


def load_starts():
    path = os.path.join("scripts", "5_evaluation", "benchmark.py")
    spec = importlib.util.spec_from_file_location("benchmark", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.starts()


def episode(sim, policy, cfg, init, max_steps, exec_horizon, tactical=None, optimizer=None):
    sim.reset(seed=init["seed"])
    obs = sim.set_state(init["init_state"][:2], init["init_state"][2:5]) if "init_state" in init else sim.reset(seed=init["seed"])[0]
    init_obs, max_cov, step, end = obs, obs["coverage"], 0, "MAX_STEPS"
    done = obs["coverage"] > SUCCESS_THRESHOLD
    if done:
        end = "SUCCESS"
    if tactical is not None:
        tactical.reset()
    body = lambda: sim.env.unwrapped.block  # noqa: E731
    while not done and step < max_steps:
        if tactical is not None:
            act_world, _, info = tactical.get_action(obs, sim, policy, exec_horizon=exec_horizon)
            queue = [postprocess_action(normalize_pos(act_world), obs)]
            if info.get("phase") == "LOCKED":
                end = "LOCKED"
        else:
            actions, _ = policy.plan(obs)
            queue = list(actions[:exec_horizon])
            if optimizer is not None:
                res = optimizer.evaluate_rollout(policy.wm, obs, np.asarray(queue), policy.device)
                if res["decision"] == "LOCK":   # stop pushing: retreat one step and end the episode
                    retreat = compute_safe_retreat(np.asarray(obs["agent_pos_px"]), np.asarray(obs["block_pose"][:2]), retreat_dist=16.0)
                    obs, *_ = sim.step(normalize_pos(retreat), normalized_action=True)
                    step += 1
                    max_cov = max(max_cov, obs["coverage"])
                    end = "WM_LOCK"
                    break
                if res["decision"] == "TRUNCATE":
                    queue = list(res["actions_to_execute"])
            queue = [("raw", a) for a in queue]
        for item in queue:
            act = item if tactical is not None else postprocess_action(item[1], obs, block_body=body(), use_collision_guard=cfg["guard"])
            obs, _, terminated, truncated, _ = sim.step(act, normalized_action=True)
            if hasattr(policy, "observe"):
                policy.observe(obs)
            step += 1
            max_cov = max(max_cov, obs["coverage"])
            if terminated or obs["coverage"] > SUCCESS_THRESHOLD:
                done, end = True, "SUCCESS"
            elif truncated or step >= max_steps:
                done, end = True, "MAX_STEPS"
            if done:
                break
        if end == "LOCKED":
            break

    def goal_dist(o):
        return float(math.hypot(o["block_pose"][0] - GOAL_XY[0], o["block_pose"][1] - GOAL_XY[1]))
    return {"seed": init["seed"], "steps": step, "end_reason": end,
            "init_block_pose": [float(v) for v in init_obs["block_pose"]], "init_agent_pos": [float(v) for v in init_obs["agent_pos_px"]],
            "final_block_pose": [float(v) for v in obs["block_pose"]], "init_coverage": float(init_obs["coverage"]),
            "final_coverage": float(obs["coverage"]), "max_coverage": float(max_cov), "success": bool(max_cov > SUCCESS_THRESHOLD),
            "init_goal_dist": goal_dist(init_obs), "final_goal_dist": goal_dist(obs),
            "init_angle_err_deg": math.degrees(angle_error_to_goal(float(init_obs["block_pose"][2]))),
            "final_angle_err_deg": math.degrees(angle_error_to_goal(float(obs["block_pose"][2]))), "mean_fde": 0.0, "max_fde": 0.0}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", nargs="*", default=list(CONFIGS), choices=list(CONFIGS))
    p.add_argument("--max-steps", type=int, default=None)
    args = p.parse_args()
    cfg_all = load_config()
    device = resolve_device(cfg_all)
    max_steps = args.max_steps or cfg_all["env"]["max_steps"]
    exec_h = cfg_all["evaluation"]["exec_horizon"]
    starts = load_starts()
    for name in args.configs:
        c = CONFIGS[name]
        policy = build_policy(cfg_all, device, ckpt_dir=CKPT, kind="dit_id")
        tactical = optimizer = None
        if c["tactical"]:
            tactical = TacticalHierarchicalController(target_lock_cov=c["lock"], peak_settle_cov=c["settle"], enable_collision_guard=c["guard"],
                                                      enable_wm_lookahead=c["lookahead"], enable_mppi=False, device=device)
        elif c["lookahead"]:
            optimizer = WorldModelLookaheadOptimizer()
        sim = PushTSim(env_name=cfg_all["env"]["env_name"], render_size=tuple(cfg_all["env"]["render_size"]), max_steps=max_steps)
        rows, t0 = [], time.perf_counter()
        for k, (suite, sname, init) in enumerate(starts):
            torch.manual_seed(10_000 + k)            # same seeds as scripts/5_evaluation/benchmark.py
            np.random.seed(10_000 + k)
            r = episode(sim, policy, c, init, max_steps, exec_h, tactical, optimizer)
            r.update({"suite": suite, "name": sname})
            rows.append(r)
        sim.close()
        label = f"E2_{name}"
        out = {"label": label, "stage": "E2", "policy": "tactical" if c["tactical"] else "dit_id", "ckpt_dir": CKPT,
               "e2_config": c, "max_steps": max_steps, "seconds": time.perf_counter() - t0,
               "summary": {s: summarize([r for r in rows if r["suite"] == s]) for s in ("seeds50", "hard20", "random25")},
               "overall": summarize(rows), "episodes": rows}
        with open(os.path.join("results", "benchmark", f"{label}.json"), "w") as f:
            json.dump(out, f, indent=1)
        o = out["overall"]
        print(f"{label}: coverage {o['mean_max_coverage'] * 100:.1f}% | success {o['success_rate'] * 100:.0f}% | "
              f"<50% {o['rate_max_cov_lt_0.50'] * 100:.0f}% | {out['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
