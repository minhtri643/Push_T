"""E7 closed-loop evaluation. Runs with the main .venv and writes ONLY inside experiments/E7_self_supervised_jepa_dit.

the original E7 code (E7/code, commit ebe99bd) and this repository's library / checkpoints are only read.

Modes
  bench95  The E7 JEPA + DiT on the 95-start benchmark of E0-E6: same starts, same per-start torch seed and the same
          episode loop (pusht_wm_dit.policy.run_episode, including its 40 px/step limit) as
          scripts/5_evaluation/benchmark.py. --seed-offset 0/1/2 gives three diffusion-noise draws.
  orig50  the E7 model in the same loop on the original 50 seeds (100000..100049), generator seeded with the env seed
          exactly like the original src/eval_sim.py. This checks the adapter against the original evaluator.
  ours50  Policies of this repository (E4 final, E5 vision) on the original 50 seeds; checkpoints read-only.

    python experiments/E7_self_supervised_jepa_dit/benchmark_e7.py bench95 --seed-offset 0
    python experiments/E7_self_supervised_jepa_dit/benchmark_e7.py orig50
    python experiments/E7_self_supervised_jepa_dit/benchmark_e7.py ours50 --label E4_final__dit_wm --seed-offset 0

Ablations and sweeps of the original run (E7 modes; the defaults are its main configuration, label without a tag):
    --policy-run dit_random_init --wm-random-init --tag A1_random_init     A1: DiT trained on random-init latents
    --policy-run dit_no_proprio --tag A2_no_proprio                          A2: DiT without the pusher position
    --n-action-steps 4|6|12 --tag k4|k6|k12                                  targets executed per plan (main: 8)
    --n-infer 4|8 --tag ddim4|ddim8                                          DDIM steps (main: 16)
The episode loop executes exactly n_action_steps targets per plan, as the original evaluator does.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

E7 = Path(__file__).resolve().parent
ROOT = E7.parents[1]
# The original E7 packages (src, scripts) first, so that its `scripts` package wins over this repository's scripts/ folder.
sys.path[:0] = [str(E7 / "_deps"), str(E7 / "code"), str(ROOT)]

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.dit_policy import path_to_velocity, velocity_to_targets  # noqa: E402  (original E7 code)
from src.eval_sim import PushTAgent, load_models  # noqa: E402  (original E7 code)

from pusht_wm_dit.envs.pusht_sim import PushTSim  # noqa: E402
from pusht_wm_dit.policy import MAX_STEP_PX, build_policy, resolve_device, run_episode, summarize  # noqa: E402
from pusht_wm_dit.utils.normalizer import normalize_pos  # noqa: E402


def _load_e0e6_benchmark():
    """scripts/5_evaluation/benchmark.py (read-only import: the 95 starts)."""
    spec = importlib.util.spec_from_file_location("e0e6_benchmark", ROOT / "scripts" / "5_evaluation" / "benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # its chdir_root() sets cwd = repository root
    return mod


E7_RUNS = E7 / "run" / "outputs" / "runs"
OUT = E7 / "results"
ORIG_SEEDS = [100000 + i for i in range(50)]


class JepaDitAdapter:
    """The original PushTAgent.act() split into the plan()/observe() interface of run_episode.

    Same computation as PushTAgent.act(): frames = (previous frame, current frame), the first frame repeated
    twice, DDIM sampling of a 20-point pusher path, its first n_action_steps points executed as targets.
    Also counts how often run_episode's 40 px/step limit changes one of the original targets."""

    def __init__(self, agent: PushTAgent, seed_mode: str):
        self.a = agent
        self.seed_mode = seed_mode  # "env" (= the original evaluator) or "torch" (= drawn from the benchmark's torch seed)
        self.env_seed = None
        self.limited = self.steps = 0

    def reset(self, obs):
        seed = self.env_seed if self.seed_mode == "env" else int(torch.randint(0, 2**31 - 1, (1,)).item())
        self.a.reset(seed)
        self._targets, self._i = None, 0

    def _count_limit(self, obs):
        if self._targets is not None and self._i < len(self._targets):
            self.steps += 1
            if np.linalg.norm(self._targets[self._i] - np.asarray(obs["agent_pos_px"], np.float64)) > MAX_STEP_PX:
                self.limited += 1

    def observe(self, obs):
        self.a.frames.append(obs["pixels"])
        self._i += 1
        self._count_limit(obs)

    @torch.no_grad()
    def plan(self, obs):
        a = self.a
        pix = obs["pixels"]
        if not a.frames:
            a.frames.append(pix)
            a.frames.append(pix)
        fr = torch.from_numpy(np.stack(a.frames)[None]).to(a.device)
        s_t = a.wm.encode_state(fr)
        pos = torch.from_numpy(np.asarray(obs["agent_pos_px"], dtype=np.float32))[None].to(a.device)
        path = a.policy.sample(s_t, pos, a.n_infer, generator=a.gen)
        v = path_to_velocity(path, pos, a.dt)
        targets = velocity_to_targets(v, pos, a.dt)[0, : a.n_action_steps].clamp(0, a.ws)
        a.last_path = path[0].cpu().numpy()
        self._targets, self._i = targets.cpu().numpy().astype(np.float64), 0
        self._count_limit(obs)
        return np.asarray(normalize_pos(self._targets), dtype=np.float32), None


def jepa_dit_policy(device, seed_mode, policy_run="dit_main", wm_random_init=False, n_action_steps=8, n_infer=16):
    # strict fp32 like the GTX 1650 of the original run (no TF32); see run_e7.ps1. Policies of E0-E6 keep PyTorch's defaults.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    # wm_random_init: the original load_models rebuilds the random encoder with set_seed(123), as export_latents did
    wm, pol, cfg = load_models("wm_main", policy_run, device, out_root=str(E7_RUNS), wm_random_init=wm_random_init)
    agent = PushTAgent(wm, pol, cfg, device, n_action_steps=n_action_steps, n_infer=n_infer)  # 8 / 16 = the original main
    return JepaDitAdapter(agent, seed_mode), cfg


def write(out_dir: Path, label: str, meta: dict, rows: list, suites: tuple, t0: float):
    out = {"label": label, "stage": "E7", **meta, "seconds": time.perf_counter() - t0,
           "summary": {s: summarize([r for r in rows if r["suite"] == s]) for s in suites if any(r["suite"] == s for r in rows)},
           "overall": summarize(rows), "episodes": rows}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{label}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    o = out["overall"]
    print(f"{label}: success {o['success_rate'] * 100:.0f}% | coverage {o['mean_max_coverage'] * 100:.1f}% | "
          f"{out['seconds']:.0f}s -> {out_dir / (label + '.json')}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["bench95", "orig50", "ours50"])
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--label", default=None, help="ours50: label of an existing results/benchmark run (policy, ckpt dir and vision backend are read from it)")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--no-rate-limit", action="store_true",
                   help="E7 modes: skip run_episode's 40 px/step limit (the original evaluator has none). Replaced in "
                        "memory for this process only; pusht_wm_dit/policy.py is not changed.")
    p.add_argument("--policy-run", default="dit_main", help="E7 modes: DiT run in E7/run/outputs/runs")
    p.add_argument("--wm-random-init", action="store_true", help="E7 modes: random-init encoder (ablation A1)")
    p.add_argument("--n-action-steps", type=int, default=8, help="E7 modes: targets executed per plan")
    p.add_argument("--n-infer", type=int, default=16, help="E7 modes: DDIM steps")
    p.add_argument("--tag", default="", help="E7 modes: label suffix for a non-main configuration")
    args = p.parse_args()
    model_opts = (args.policy_run, args.wm_random_init, args.n_action_steps, args.n_infer) != ("dit_main", False, 8, 16)
    if args.mode == "ours50" and (args.no_rate_limit or model_opts or args.tag):
        raise SystemExit("--no-rate-limit and the E7 options are only for the E7 model")
    if args.mode != "ours50" and model_opts and not args.tag:
        raise SystemExit("a non-main E7 configuration needs --tag (else it would overwrite the main results)")
    if args.no_rate_limit:
        import pusht_wm_dit.policy as _pol
        _pol.postprocess_action = lambda act_norm, obs, **kw: np.asarray(act_norm, dtype=np.float32)

    bench = _load_e0e6_benchmark()
    import yaml
    cfg_repo = yaml.safe_load(open(ROOT / "configs" / "default.yaml", encoding="utf-8"))
    device = resolve_device(cfg_repo)
    sim = PushTSim(env_name=cfg_repo["env"]["env_name"], render_size=tuple(cfg_repo["env"]["render_size"]), max_steps=args.max_steps)
    noise = ("__nolimit" if args.no_rate_limit else "") + (f"__noise{args.seed_offset}" if args.seed_offset else "")
    tag = f"__{args.tag}" if args.tag else ""
    rows, t0 = [], time.perf_counter()
    kopts = dict(policy_run=args.policy_run, wm_random_init=args.wm_random_init,
                 n_action_steps=args.n_action_steps, n_infer=args.n_infer)
    kmeta = {"policy_run": args.policy_run, "wm_random_init": args.wm_random_init,
             "n_action_steps": args.n_action_steps, "n_infer": args.n_infer}

    if args.mode == "bench95":
        pol, _ = jepa_dit_policy(torch.device(device), "torch", **kopts)
        for k, (suite, name, init) in enumerate(bench.starts()):
            torch.manual_seed(10_000 + k + 1000 * args.seed_offset)  # = scripts/5_evaluation/benchmark.py
            np.random.seed(10_000 + k + 1000 * args.seed_offset)
            r = run_episode(sim, pol, max_steps=args.max_steps, exec_horizon=args.n_action_steps, **init)
            r.pop("coverage_history")
            r.update({"suite": suite, "name": name})
            rows.append(r)
        meta = {"policy": "jepa_dit", "ckpt_dir": f"experiments/E7_self_supervised_jepa_dit/run/outputs/runs/"
                f"{{{'random-init encoder' if args.wm_random_init else 'wm_main'},{args.policy_run}}}", **kmeta,
                "diffusion_noise_offset": args.seed_offset, "max_steps": args.max_steps,
                "rate_limit_40px": not args.no_rate_limit,
                "rate_limit_steps": pol.limited, "executed_steps": pol.steps}  # with the limit off: targets it would have changed
        write(ROOT / "results" / "benchmark", f"E7_jepa_dit{tag}{noise}", meta, rows, ("seeds50", "hard20", "random25"), t0)

    elif args.mode == "orig50":
        # offset 0: generator seed = env seed, exactly as the original src/eval_sim.py; offsets 1, 2: env seed + 1000 * offset
        pol, _ = jepa_dit_policy(torch.device(device), "env", **kopts)
        for s in ORIG_SEEDS:
            pol.env_seed = s + 1000 * args.seed_offset
            r = run_episode(sim, pol, seed=s, max_steps=args.max_steps, exec_horizon=args.n_action_steps)
            r.pop("coverage_history")
            r.update({"suite": "orig50", "name": f"seed {s}"})
            rows.append(r)
        meta = {"policy": "jepa_dit", "generator_seed": "env seed + 1000 * offset (offset 0 = src/eval_sim.py)", **kmeta,
                "diffusion_noise_offset": args.seed_offset, "max_steps": args.max_steps,
                "rate_limit_40px": not args.no_rate_limit,
                "rate_limit_steps": pol.limited, "executed_steps": pol.steps}  # with the limit off: targets it would have changed
        write(OUT / "seeds100000", f"E7_jepa_dit__harness{tag}{noise}", meta, rows, ("orig50",), t0)

    else:
        ref = json.loads((ROOT / "results" / "benchmark" / f"{args.label}.json").read_text(encoding="utf-8"))
        ckpt_dir = ref.get("ckpt_dir")
        ckpt_dir = None if ckpt_dir in (None, "checkpoints/final") else ckpt_dir
        policy = build_policy(cfg_repo, device, ckpt_dir=ckpt_dir, kind=ref["policy"], vision_backend=ref.get("vision_backend"))
        for i, s in enumerate(ORIG_SEEDS):
            torch.manual_seed(10_000 + i + 1000 * args.seed_offset)
            np.random.seed(10_000 + i + 1000 * args.seed_offset)
            r = run_episode(sim, policy, seed=s, max_steps=args.max_steps, exec_horizon=cfg_repo["evaluation"]["exec_horizon"])
            r.pop("coverage_history")
            r.update({"suite": "orig50", "name": f"seed {s}"})
            rows.append(r)
        meta = {"policy": ref["policy"], "ckpt_dir": ref.get("ckpt_dir"), "vision_backend": ref.get("vision_backend"),
                "source_label": args.label, "diffusion_noise_offset": args.seed_offset, "max_steps": args.max_steps}
        write(OUT / "seeds100000", f"{args.label}{noise}", meta, rows, ("orig50",), t0)
    sim.close()


if __name__ == "__main__":
    main()
