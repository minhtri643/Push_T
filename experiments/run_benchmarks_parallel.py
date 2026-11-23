"""Run several benchmark jobs at the same time (each benchmark uses ~0.3-1 GB of GPU memory and is mostly
CPU-bound: physics simulation, one episode at a time). Results are unchanged by running in parallel: every
episode uses its own fixed seeds; the job `E4_final__dit_wm__rerun_2026-11-23-parallel` re-runs a stored result
to verify this (compare with E4_final__dit_wm.json, 95/95 starts must be identical).

    python experiments/run_benchmarks_parallel.py --workers 4
"""

import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
B = [sys.executable, "scripts/5_evaluation/benchmark.py"]
VISION = [("E5b_jepa", "jepa_wm"), ("E5c_dinov2_frozen", "dinov2_hf_wm"), ("E5d_vjepa2_frozen", "vjepa2_wm")]
JOBS = []  # (label, args); longest first so the workers finish together
for o in (1, 2):
    JOBS.append((f"E5d_vjepa2_frozen__dit_id_vision__noise{o}", ["--policy", "dit_id_vision", "--vision-backend", "vjepa2_wm", "--seed-offset", str(o)]))
for o in (1, 2):
    for name, backend in VISION[:2]:
        JOBS.append((f"{name}__dit_id_vision__noise{o}", ["--policy", "dit_id_vision", "--vision-backend", backend, "--seed-offset", str(o)]))
JOBS += [("E4_final__dit_wm__noise2", ["--policy", "dit_wm", "--seed-offset", "2"]),
         ("E4_final__dit_id__noise2", ["--policy", "dit_id", "--seed-offset", "2"]),
         ("E4_final__dit_wm__rerun_2026-11-23-parallel", ["--policy", "dit_wm"])]


def run(job):
    label, args = job
    if os.path.exists(os.path.join(ROOT, "results", "benchmark", f"{label}.json")):
        return f"skip {label} (result exists)"
    t0 = time.perf_counter()
    p = subprocess.run(B + args + ["--label", label], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = [l for l in (p.stdout + p.stderr).splitlines() if not re.search(r"Warning|warn|frozen successfully", l)]
    with open(os.path.join(ROOT, "results", "benchmark", "logs", f"{label}.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    last = next((l for l in reversed(out) if "coverage" in l), out[-1] if out else "")
    return f"{'ok' if p.returncode == 0 else 'FAILED'} {time.perf_counter() - t0:5.0f}s  {last}"


if __name__ == "__main__":
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    print(f"{len(JOBS)} jobs, {workers} at a time ({time.strftime('%H:%M:%S')})", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        for msg in ex.map(run, JOBS):
            print(msg, flush=True)
    print(f"ALL DONE {time.strftime('%H:%M:%S')}", flush=True)
