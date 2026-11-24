"""E7 analysis: reproduction of the original results and comparison with E0-E6 (reads only; writes E7/results/summary.*).

    python experiments/E7_self_supervised_jepa_dit/analyze_e7.py

Statistics (same as scripts/5_evaluation/summarize_results.py and the E4/E5 READMEs)
    success        max coverage > 0.95 within 300 steps; 95% Wilson interval
    success@0.90   max coverage > 0.90 (the level where the human demonstrations stop, the second metric of the original run)
    DP score       mean of min(max coverage / 0.95, 1)
    McNemar        exact, success paired by start, one noise draw against another run
    pooled sign    per start, number of successes over the three noise draws, compared between two policies;
                   exact two-sided sign test over the starts where the counts differ

Two variants of the original policy in this repository's episode loop:
    "40 px limit"  run_episode exactly as for E0-E6 (the commanded target moves at most 40 px per step)
    "no limit"     the limit switched off, as in the original evaluator (its policy was trained and evaluated without it)
"""

import csv
import json
import math
from pathlib import Path

import numpy as np

E7 = Path(__file__).resolve().parent
ROOT = E7.parents[1]
BENCH = ROOT / "results" / "benchmark"
RES = E7 / "results"
DRAWS = ("", "__noise1", "__noise2")


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def binom_two_sided(a, b):
    n, k = a + b, min(a, b)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def load_json(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def episodes(d):
    return np.array([e["max_coverage"] for e in d["episodes"]])


def draws(folder, label):
    out = []
    for suf in DRAWS:
        d = load_json(folder / f"{label}{suf}.json")
        if d is not None:
            out.append(episodes(d))
    return out


def row_stats(mcs):
    s = [m > 0.95 for m in mcs]
    rates = [float(x.mean()) for x in s]
    return {"n": len(mcs[0]), "draws": rates, "mean": float(np.mean(rates)),
            "sd": float(np.std(rates, ddof=1)) if len(rates) > 1 else None,
            "ci_first": wilson(int(s[0].sum()), len(mcs[0])), "s090": float(np.mean([(m > 0.90).mean() for m in mcs])),
            "coverage": float(np.mean([m.mean() for m in mcs])), "dp": float(np.mean([np.minimum(m / 0.95, 1).mean() for m in mcs]))}


def pooled_sign(a_mcs, b_mcs):
    ca = np.sum([m > 0.95 for m in a_mcs], axis=0)
    cb = np.sum([m > 0.95 for m in b_mcs], axis=0)
    better, worse = int((ca > cb).sum()), int((ca < cb).sum())
    return better, worse, binom_two_sided(better, worse)


def mcnemar(a, b):
    sa, sb = a > 0.95, b > 0.95
    oa, ob = int((sa & ~sb).sum()), int((~sa & sb).sum())
    return oa, ob, binom_two_sided(oa, ob)


def pct(x):
    return f"{x * 100:.0f}%"


def fmt_row(name, st):
    dr = " / ".join(pct(x) for x in st["draws"])
    msd = f"{st['mean'] * 100:.1f} ± {st['sd'] * 100:.1f}" if st["sd"] is not None else f"{st['mean'] * 100:.1f}"
    lo, hi = st["ci_first"]
    return f"| {name} | {dr} | {msd} | {pct(lo)}–{pct(hi)} | {pct(st['s090'])} | {st['coverage'] * 100:.1f}% | {st['dp']:.3f} |"


HEAD = ("| Policy | Success per noise draw | Mean ± sd (%) | 95% CI (draw 0) | Success@0.90 | Mean max coverage | DP score |\n"
        "|---|---|---|---|---|---|---|")


def orig_csv(path):
    if not Path(path).exists():
        return None
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    return {int(r["seed"]): (r["success"] == "True", float(r["max_coverage"])) for r in rows}


def paired_table(variants, others, folder_of_other, lines, store):
    """variants: [(name, mcs)] of E7; others: [(name, label)]; pooled sign + McNemar on draw 0 per variant."""
    head = "| vs | " + " | ".join(f"{vn}: E7 better / worse (pooled), p | McNemar draw 0, p" for vn, _ in variants) + " |"
    lines += [head, "|---|" + "---|---|" * len(variants)]
    for name, lab in others:
        mc = draws(folder_of_other, lab)
        if not mc:
            continue
        cells = []
        for vn, e7 in variants:
            if len(mc) > 1 and len(e7) > 1:
                b, w, p = pooled_sign(e7, mc)
                pooled = f"{b} / {w}, p = {p:.4f}"
            else:
                b = w = p = None
                pooled = "-"
            oa, ob, pm = mcnemar(e7[0], mc[0])
            cells += [pooled, f"{oa} / {ob}, p = {pm:.4f}"]
            store.setdefault(lab, {})[vn] = {"pooled": [b, w, p], "mcnemar0": [oa, ob, pm]}
        lines.append(f"| {name} | " + " | ".join(cells) + " |")


def offline(base, wm="wm_main"):
    out = {}
    log = base / f"outputs/runs/{wm}/log.csv"
    if log.exists():
        rows = list(csv.DictReader(open(log, encoding="utf-8")))
        out["epochs"] = len(rows)
        out["val_l_pred"] = min(float(r["val_l_pred"]) for r in rows)
        out["std_s"], out["erank_s"] = float(rows[-1]["std_s"]), float(rows[-1]["erank_s"])
    probe = base / f"outputs/runs/{wm}/figures/probe_table.csv"
    if probe.exists():
        r = next(csv.DictReader(open(probe, encoding="utf-8")))
        out.update({"r2_pos": float(r["r2_block_pos"]), "r2_ang": float(r["r2_block_angle"]),
                    "pos_err": float(r["block_pos_err_px"]), "ang_err": float(r["block_angle_err_deg"])})
    fde = base / "outputs/runs/dit_main/fde_obj/fde_summary.json"
    if fde.exists() and wm == "wm_main":
        d = json.loads(fde.read_text(encoding="utf-8"))["DiT"]
        out.update({"fde_pos": d["FDE_pos"]["mean"], "fde_th": d["FDE_theta_deg"]["mean"], "fde_obj": d["FDE_obj"]["mean"],
                    "minfde_obj5": d["minFDE_obj@5"]})
    return out


# the original configurations: (tag in E7/results, name, DiT run, eval folder in its outputs, n_action_steps, DDIM steps)
CONFIGS = [("", "Main (k = 8, DDIM 16)", "dit_main", "main"),
           ("A1_random_init", "A1: random-init encoder", "dit_random_init", "a1_k8_s16"),
           ("A2_no_proprio", "A2: images only (no pusher position)", "dit_no_proprio", "a2_k8_s16"),
           ("k4", "n_action_steps = 4", "dit_main", "sweep_k4_s16"),
           ("k6", "n_action_steps = 6", "dit_main", "sweep_k6_s16"),
           ("k12", "n_action_steps = 12", "dit_main", "sweep_k12_s16"),
           ("ddim4", "DDIM 4 steps", "dit_main", "a4_k8_s4"),
           ("ddim8", "DDIM 8 steps", "dit_main", "a4_k8_s8")]


def orig_eval(base, run, folder):
    d = load_json(base / f"outputs/runs/{run}/eval/{folder}/eval_summary.json")
    eps = orig_csv(base / f"outputs/runs/{run}/eval/{folder}/eval_episodes.csv")
    return d, eps


def fmt_orig(d):
    if d is None:
        return "not re-run"
    med = (d.get("time_to_success_s") or {}).get("median")
    return (f"{pct(d['success_rate'])} / {pct(d['success_090_rate'])} / {d['mean_max_coverage']:.3f} / "
            f"{'-' if med is None else f'{med:.1f} s'} / {d['mean_infer_ms']:.0f} ms")


def live_log(path):
    if not Path(path).exists():
        return None
    rows = list(csv.DictReader(open(path, encoding="utf-8")))[-20:]  # the log is appended; the last 20 rows = latest run
    ok = [r["success"] == "True" for r in rows]
    return {"n": len(rows), "success": int(sum(ok)), "mean_max_cov": float(np.mean([float(r["max_coverage"]) for r in rows])),
            "seeds": f"{rows[0]['seed']}-{rows[-1]['seed']}" if rows else ""}


def section_ablations(lines, summary):
    orig, rerun = E7 / "original_results", E7 / "run"
    out = {"original_evaluator": {}, "benchmark95": {}}

    # 4a. the original evaluator, its 50 seeds: original vs re-run, plus the adapter check
    lines += ["## 4. Ablations and sweeps of the original run", "",
              "### 4a. The original evaluator on the original 50 seeds (one noise draw, noise = env seed): the original run vs the E7 re-run", "",
              "Cells: success@0.95 / success@0.90 / mean max coverage / median time to success / inference time per plan.", "",
              "| Configuration | Original run (GTX 1650) | E7 re-run (RTX 3050) | Same seed outcome original vs E7, McNemar p | "
              "Adapter check: this repository's loop = the original evaluator (E7 weights) |",
              "|---|---|---|---|---|"]
    for tag, name, run, folder in CONFIGS:
        dk, ek = orig_eval(orig, run, folder)
        dr, er = orig_eval(rerun, run, folder)
        pair = "-"
        if ek and er:
            S = sorted(ek)
            a = np.array([ek[s][0] for s in S]); b = np.array([er[s][0] for s in S])
            x, y = int((a & ~b).sum()), int((~a & b).sum())
            pair = f"{int((a == b).sum())}/50 same; original only {x}, E7 only {y}, p = {binom_two_sided(x, y):.2f}"
        adapter = "-"
        h = load_json(RES / f"seeds100000/E7_jepa_dit__harness{'__' + tag if tag else ''}__nolimit.json")
        if h and er:
            seeds = [int(e["name"].split()[1]) for e in h["episodes"]]
            m = episodes(h)
            same = sum(er[sd][0] == (mm > 0.95) for sd, mm in zip(seeds, m))
            ident = sum(abs(er[sd][1] - mm) < 1e-6 for sd, mm in zip(seeds, m))
            adapter = f"{same}/50 same success, identical coverage on {ident}/50"
        lines.append(f"| {name} | {fmt_orig(dk)} | {fmt_orig(dr)} | {pair} | {adapter} |")
        out["original_evaluator"][tag or "main"] = {"original": dk, "e7": dr, "seed_pairs": pair, "adapter": adapter}
    if (rerun / "outputs/runs/dit_main/eval/sweep_k6_s16").exists():
        lines += ["", "E7 inference times for k = 6 and k = 12 are not valid: those two runs shared the GPU with other "
                      "evaluations. Success and coverage are not affected (the episode loop does not depend on timing)."]
    lk, le =live_log(orig / "outputs/demo/live_log.csv"), live_log(rerun / "outputs/demo/live_log.csv")
    if lk or le:
        f = lambda d: "not re-run" if d is None else f"{d['success']}/{d['n']} ({d['success'] / d['n'] * 100:.0f}%), mean max coverage {d['mean_max_cov']:.3f}"
        lines += ["", f"Live demo on 20 fresh seeds ({(lk or le)['seeds']}, DDIM 8, scripts/demo_live.py): original run {f(lk)}; E7 re-run {f(le)}."]
        out["live_demo"] = {"original": lk, "e7": le}

    # 4b. offline: probe baselines and FDE of the ablations
    lines += ["", "### 4b. Offline metrics: probe baselines and FDE of the ablations (original run / E7 re-run)", "",
              "| Probe on S_t (20 val episodes) | R² block position | R² block angle | position error (px) | angle error (°) |",
              "|---|---|---|---|---|"]
    pk = {r["method"]: r for r in csv.DictReader(open(orig / "outputs/runs/wm_main/figures/probe_table.csv", encoding="utf-8"))}
    pr = {r["method"]: r for r in csv.DictReader(open(rerun / "outputs/runs/wm_main/figures/probe_table.csv", encoding="utf-8"))}
    for meth in pk:
        if meth in pr:
            c = [f"{float(pk[meth][k]):{f}} / {float(pr[meth][k]):{f}}" for k, f in
                 (("r2_block_pos", ".3f"), ("r2_block_angle", ".3f"), ("block_pos_err_px", ".1f"), ("block_angle_err_deg", ".1f"))]
            lines.append(f"| {meth} | " + " | ".join(c) + " |")
    lines += ["", "| Policy, FDE of the block after 2 s (375 val samples) | FDE position (px) | FDE angle (°) | FDE_obj | minFDE_obj@5 |",
              "|---|---|---|---|---|"]
    fde = {}
    for run, name in (("dit_main", "Main (JEPA + DiT)"), ("dit_no_proprio", "A2: no pusher position"), ("dit_random_init", "A1: random-init encoder")):
        dk = load_json(orig / f"outputs/runs/{run}/fde_obj/fde_summary.json")
        dr = load_json(rerun / f"outputs/runs/{run}/fde_obj/fde_summary.json")
        rows_ = [("DiT", name)] + ([("constant velocity", "Constant velocity (baseline)"), ("stand still", "Stand still (baseline)")] if run == "dit_main" else [])
        for key, nm in rows_:
            def cell(m, f="{:.1f}"):
                a = (dk or {}).get(key, {}); b = (dr or {}).get(key, {})
                va = a.get(m, {}).get("mean") if isinstance(a.get(m), dict) else a.get(m)
                vb = b.get(m, {}).get("mean") if isinstance(b.get(m), dict) else b.get(m)
                return f"{'-' if va is None else f.format(va)} / {'-' if vb is None else f.format(vb)}"
            lines.append(f"| {nm} | {cell('FDE_pos')} | {cell('FDE_theta_deg')} | {cell('FDE_obj')} | {cell('minFDE_obj@5')} |")
        fde[run] = {"original": dk, "e7": dr}
    out["fde"] = fde

    # 4c. the 95-start benchmark, three noise draws, no 40 px limit
    main_mc = draws(BENCH, "E7_jepa_dit__nolimit")
    if not any(draws(BENCH, f"E7_jepa_dit__{t}__nolimit") for t, _, _, _ in CONFIGS if t):
        summary["ablations"] = out
        return  # the ablations / sweeps were not run on the 95 starts
    lines += ["", "### 4c. The 95-start benchmark of E0-E6 (three noise draws, no 40 px limit, E7 re-run weights)", "",
              HEAD.replace("| Policy |", "| Configuration |")]
    rows95 = []
    for tag, name, run, folder in CONFIGS:
        mc = draws(BENCH, f"E7_jepa_dit{'__' + tag if tag else ''}__nolimit")
        if not mc:
            continue
        st = row_stats(mc)
        out["benchmark95"][tag or "main"] = st
        rows95.append(fmt_row(name, st))
        if tag and len(mc) == 3 and len(main_mc) == 3:
            b, w, p = pooled_sign(mc, main_mc)
            out["benchmark95"][tag]["vs_main_pooled"] = [b, w, p]
    lines += rows95
    vs = [(name, out["benchmark95"][tag]["vs_main_pooled"]) for tag, name, _, _ in CONFIGS
          if tag and tag in out["benchmark95"] and "vs_main_pooled" in out["benchmark95"][tag]]
    if vs:
        lines += ["", "Pooled sign test against the main configuration (per start, successes over the three draws):", "",
                  "| Configuration | better / worse than main | p |", "|---|---|---|"]
        lines += [f"| {n} | {b} / {w} | {p:.4f} |" for n, (b, w, p) in vs]
    summary["ablations"] = out


def main():
    lines = ["# E7 results", ""]
    summary = {}

    # ---------------------------------------------------------------- 1. reproduction on the original 50 seeds
    orig = orig_csv(E7 / "original_results/outputs/runs/dit_main/eval/main/eval_episodes.csv")
    rerun = orig_csv(E7 / "run/outputs/runs/dit_main/eval/main/eval_episodes.csv")
    lines += ["## 1. Reproduction of the original result (the original 50 seeds 100000-100049, DDIM 16, 8 steps executed, noise = env seed)", "",
              "| Run | Success@0.95 | 95% CI | Success@0.90 | Mean max coverage | Same success as the original evaluator (E7 re-training) |",
              "|---|---|---|---|---|---|"]
    rep = {}
    for name, d in (("Original run (GTX 1650)", orig), ("E7 re-training, the original evaluator (src/eval_sim.py)", rerun)):
        if d:
            s = np.array([v[0] for v in d.values()]); m = np.array([v[1] for v in d.values()])
            lo, hi = wilson(int(s.sum()), len(s))
            lines.append(f"| {name} | {pct(s.mean())} ({int(s.sum())}/{len(s)}) | {pct(lo)}–{pct(hi)} | {pct((m > 0.90).mean())} | {m.mean() * 100:.1f}% | - |")
            rep[name] = {"success": float(s.mean()), "s090": float((m > 0.90).mean()), "coverage": float(m.mean())}
    notes = []
    for name, lab in (("E7 re-training, this repository's loop, no limit (adapter check)", "E7_jepa_dit__harness__nolimit"),
                      ("E7 re-training, this repository's loop, 40 px limit", "E7_jepa_dit__harness")):
        h = load_json(RES / f"seeds100000/{lab}.json")
        if not h:
            continue
        m = episodes(h); s = m > 0.95
        lo, hi = wilson(int(s.sum()), len(s))
        agree = "-"
        if rerun:
            seeds = [int(e["name"].split()[1]) for e in h["episodes"]]
            same = sum(rerun[sd][0] == (mm > 0.95) for sd, mm in zip(seeds, m))
            ident = sum(abs(rerun[sd][1] - mm) < 1e-6 for sd, mm in zip(seeds, m))
            agree = f"{same}/50 (identical coverage on {ident}/50)"
            rep[lab] = {"success": float(s.mean()), "same_success": int(same), "identical_coverage": int(ident)}
        lines.append(f"| {name} | {pct(s.mean())} ({int(s.sum())}/{len(s)}) | {pct(lo)}–{pct(hi)} | {pct((m > 0.90).mean())} | {m.mean() * 100:.1f}% | {agree} |")
        if h.get("rate_limit_40px", True):
            notes.append(f"With the 40 px limit, {h['rate_limit_steps']} of {h['executed_steps']} executed targets were changed "
                         f"({h['rate_limit_steps'] / max(h['executed_steps'], 1) * 100:.1f}%).")
    lines += [""] + notes
    summary["reproduction"] = rep
    lines.append("")

    # ---------------------------------------------------------------- 1b. offline metrics
    off = {"Original run": offline(E7 / "original_results"), "E7 fp32 (used)": offline(E7 / "run"),
           "E7 with TF32 (not used)": offline(E7 / "run", "wm_main_tf32")}
    keys = [("epochs", "WM epochs", "{:.0f}"), ("val_l_pred", "best val latent loss", "{:.4f}"), ("std_s", "std of S_t", "{:.3f}"),
            ("erank_s", "effective rank of S_t", "{:.1f}"), ("r2_pos", "probe R² block position", "{:.3f}"),
            ("r2_ang", "probe R² block angle", "{:.3f}"), ("pos_err", "probe position error (px)", "{:.1f}"),
            ("ang_err", "probe angle error (°)", "{:.1f}"), ("fde_pos", "DiT FDE position at 2 s (px)", "{:.1f}"),
            ("fde_th", "DiT FDE angle at 2 s (°)", "{:.1f}"), ("fde_obj", "DiT FDE_obj", "{:.1f}"), ("minfde_obj5", "DiT minFDE_obj@5", "{:.1f}")]
    lines += ["### Offline metrics (20 validation episodes)", "", "| Metric | " + " | ".join(off) + " |", "|---|" + "---|" * len(off)]
    for k, name, f in keys:
        lines.append(f"| {name} | " + " | ".join(f.format(v[k]) if k in v else "-" for v in off.values()) + " |")
    summary["offline"] = off
    lines.append("")

    # ---------------------------------------------------------------- 2. the 95-start benchmark of E0-E6
    refs = [("E0 baseline (`dit_id`)", "E0_baseline__dit_id"),
            ("E3 action diffusion (DiT → actions, frozen WM encoder)", "E3_action_diffusion_baseline"),
            ("E4 final `dit_wm` *(true block pose)*", "E4_final__dit_wm"),
            ("E4 `dit_id` *(true block pose)*", "E4_final__dit_id"),
            ("E5b JEPA `dit_id_vision`", "E5b_jepa__dit_id_vision"),
            ("E5c DINOv2 `dit_id_vision`", "E5c_dinov2_frozen__dit_id_vision"),
            ("E5d V-JEPA 2 `dit_id_vision`", "E5d_vjepa2_frozen__dit_id_vision")]
    e7_variants = [("40 px limit", draws(BENCH, "E7_jepa_dit")),
                   ("no limit", draws(BENCH, "E7_jepa_dit__nolimit"))]
    e7_variants = [(n, v) for n, v in e7_variants if v]
    lines += ["## 2. The 95-start benchmark of E0-E6 (paired starts, 300 steps)", "", HEAD]
    b95 = {}
    for vn, mc in e7_variants:
        st = row_stats(mc); b95[f"E7 ({vn})"] = st
        lines.append(fmt_row(f"**E7 JEPA + DiT, {vn}** (images + pusher only)", st))
    for name, lab in refs:
        mc = draws(BENCH, lab)
        if mc:
            b95[lab] = row_stats(mc)
            lines.append(fmt_row(name, b95[lab]))
    if e7_variants:
        lines += ["", "Paired tests against E7 (per start; pooled = three noise draws):", ""]
        tests = {}
        paired_table(e7_variants, refs, BENCH, lines, tests)
        b95["paired_vs_e7"] = tests
        if len(e7_variants) == 2:
            b, w, p = pooled_sign(e7_variants[1][1], e7_variants[0][1])
            lines += ["", f"E7 no limit vs E7 40 px limit (pooled): {b} better / {w} worse, p = {p:.4f}."]
        for vn, lab in (("40 px limit", "E7_jepa_dit"), ("no limit", "E7_jepa_dit__nolimit")):
            d0 = load_json(BENCH / f"{lab}.json")
            if d0:
                lines.append(f"E7 {vn}, per suite (draw 0), success / mean max coverage: " +
                             ", ".join(f"{k} {pct(v['success_rate'])} / {v['mean_max_coverage'] * 100:.0f}%" for k, v in d0["summary"].items()) +
                             f"; targets beyond 40 px: {d0['rate_limit_steps']} of {d0['executed_steps']} "
                             f"({d0['rate_limit_steps'] / max(d0['executed_steps'], 1) * 100:.1f}%).")
    a, b = draws(BENCH, "E4_final__dit_wm"), draws(BENCH, "E4_final__dit_id")
    if len(a) == 3 and len(b) == 3:
        bb, ww, pp = pooled_sign(a, b)
        lines.append(f"Check of the pooled test: E4 `dit_wm` vs `dit_id` = {bb} / {ww}, p = {pp:.4f} (published: p = 0.0001).")
    summary["benchmark95"] = b95
    lines.append("")

    # ---------------------------------------------------------------- 3. the original 50 seeds, every policy
    lines += ["## 3. The original 50 seeds (100000-100049), every policy in the same episode loop", "", HEAD]
    k50 = {}
    orig_variants = [("40 px limit", draws(RES / "seeds100000", "E7_jepa_dit__harness")),
                   ("no limit", draws(RES / "seeds100000", "E7_jepa_dit__harness__nolimit"))]
    orig_variants = [(n, v) for n, v in orig_variants if v]
    for vn, mc in orig_variants:
        k50[f"E7 ({vn})"] = row_stats(mc)
        lines.append(fmt_row(f"**E7 JEPA + DiT, {vn}**", k50[f"E7 ({vn})"]))
    ours = [("E4 final `dit_wm` *(true block pose)*", "E4_final__dit_wm"), ("E4 `dit_id` *(true block pose)*", "E4_final__dit_id"),
            ("E5b JEPA `dit_id_vision`", "E5b_jepa__dit_id_vision"), ("E5c DINOv2 `dit_id_vision`", "E5c_dinov2_frozen__dit_id_vision"),
            ("E5d V-JEPA 2 `dit_id_vision`", "E5d_vjepa2_frozen__dit_id_vision")]
    for name, lab in ours:
        mc = draws(RES / "seeds100000", lab)
        if mc:
            k50[lab] = row_stats(mc)
            lines.append(fmt_row(name, k50[lab]))
    if orig_variants:
        lines += ["", "Paired tests against E7 on the same 50 seeds:", ""]
        tests = {}
        paired_table(orig_variants, ours, RES / "seeds100000", lines, tests)
        k50["paired_vs_e7"] = tests
    summary["seeds100000"] = k50
    lines.append("")

    # ---------------------------------------------------------------- 4. Ablations and sweeps of the original run
    section_ablations(lines, summary)

    (RES / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RES / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
