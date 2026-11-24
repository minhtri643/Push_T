"""Pipeline 5 (evaluation): one table with every benchmark run, from the raw per-episode JSON files.

    python scripts/5_evaluation/summarize_results.py

Reads results/benchmark/*.json (one file per run, 95 paired starts each) and results/benchmark/run_registry.json
(description, stage, training data, date of each configuration), and writes
    results/summary/benchmark_table.md      English table
    results/summary/benchmark_table_vi.md   Vietnamese table
    results/summary/runs.csv                one row per run (registry + metrics)
    results/summary/summary.json            the same numbers, machine-readable (README, figures)

Statistics
    success          coverage > 0.95 at any step of the 300; 95% Wilson interval over the 95 starts
    noise repeats    runs __noise1/__noise2 repeat a configuration with other diffusion noise (same models,
                     same starts): mean +- sd of the success rate over the repeats
    DP score         mean over starts of min(max coverage / 0.95, 1), the score of the Diffusion Policy paper
    vs final         exact McNemar test on success, paired by start, against E4_final__dit_wm (noise 0)
    reruns           __rerun_<date> files repeat a run exactly (same noise) to check determinism
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from pusht_wm_dit.paths import chdir_root  # noqa: E402
chdir_root()

import csv  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402

import numpy as np  # noqa: E402

BENCH, OUT = "results/benchmark", "results/summary"
FINAL = "E4_final__dit_wm"
SUITES = ("seeds50", "hard20", "random25")
STAGE_DIRS = {"E0": "experiments/E0_baseline", "E1": "experiments/E1_protocol_and_augmentation_fix",
              "E2": "experiments/E2_handcrafted_controllers", "E3": "experiments/E3_dynamics_world_model",
              "E4": "experiments/E4_hf_only_final", "E5": "experiments/E5_vision_world_models",
              "E6": "experiments/E6_end2end_vision_diffusion", "E7": "experiments/E7_self_supervised_jepa_dit"}


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar(a, b):
    """Exact two-sided McNemar on paired booleans: (only a, only b, p)."""
    oa, ob = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n, k = oa + ob, min(oa, ob)
    return oa, ob, 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def split_label(label):
    m = re.match(r"^(.*?)(?:__noise(\d+))?(?:__rerun_(.+))?$", label)
    return m.group(1), int(m.group(2) or 0), m.group(3)


def load():
    reg = json.load(open(os.path.join(BENCH, "run_registry.json"), encoding="utf-8"))
    runs = {}
    for f in sorted(glob.glob(os.path.join(BENCH, "*.json"))):
        if f.endswith("run_registry.json"):
            continue
        d = json.load(open(f, encoding="utf-8"))
        runs[d["label"]] = d
    return reg, runs


def metrics(d):
    mc = np.array([e["max_coverage"] for e in d["episodes"]])
    s = mc > 0.95
    suites = np.array([e["suite"] for e in d["episodes"]])
    return {"n": int(len(s)), "success": float(s.mean()), "ci": wilson(int(s.sum()), len(s)), "coverage": float(mc.mean()),
            "dp_score": float(np.minimum(mc / 0.95, 1).mean()), "severe": float((mc < 0.5).mean()),
            "suites": {k: float(s[suites == k].mean()) for k in SUITES}, "_succ": s}


DATA_VI = {"zarr": "zarr", "zarr (+ simulator data for the dynamics)": "zarr (+ dữ liệu mô phỏng cho động lực học)",
           "lerobot/pusht only": "chỉ lerobot/pusht", "zarr + simulator frames": "zarr + ảnh mô phỏng",
           "lerobot/pusht only (+ frozen pretrained DINOv2)": "chỉ lerobot/pusht (+ DINOv2 tiền huấn luyện, đóng băng)",
           "lerobot/pusht only (+ frozen pretrained V-JEPA 2)": "chỉ lerobot/pusht (+ V-JEPA 2 tiền huấn luyện, đóng băng)"}


def data_text(s, lang):
    return DATA_VI.get(s, s) if lang == "vi" else s


README_ROWS = [  # (label, what it uses at run time EN, VI)
    ("E4_final__dit_wm", "image + pusher position + block pose from the simulator", "ảnh + vị trí đầu đẩy + tư thế khối từ mô phỏng"),
    ("E4_final__dit_id", "same, without the World Model choosing the actions", "như trên, không có World Model chọn hành động"),
    ("E5d_vjepa2_frozen__dit_id_vision", "image + pusher position only (block pose read from the image)", "chỉ ảnh + vị trí đầu đẩy (tư thế khối đọc từ ảnh)"),
    ("E3_action_diffusion_baseline", "image + pusher position (Diffusion-Policy-style baseline, zarr data)", "ảnh + vị trí đầu đẩy (baseline kiểu Diffusion Policy, dữ liệu zarr)"),
    ("E7_jepa_dit", "image + pusher position, fully self-supervised JEPA encoder", "ảnh + vị trí đầu đẩy, bộ mã hoá JEPA hoàn toàn tự giám sát"),
    ("E6a_resnet_spatial_direct", "image + pusher position, end-to-end ResNet-18, actions diffused directly", "ảnh + vị trí đầu đẩy, ResNet-18 học đầu-cuối, khuếch tán trực tiếp hành động"),
    ("E0_baseline__dit_id", "the original version of this project", "phiên bản đầu tiên của dự án"),
]


def readme_blocks(rows, reg, pct):
    """Writes the result tables between <!-- RESULTS:START --> and <!-- RESULTS:END --> in README.md / README_vi.md."""
    by = {r["label"]: r for r in rows}

    def succ(r, lang):
        s = f"{pct(r['success'])} [{pct(r['success_ci_low'])}–{pct(r['success_ci_high'])}]"
        if r["noise_repeats"] > 1:
            m, sd, n = r["success_mean_over_noise"] * 100, r["success_sd_over_noise"] * 100, r["noise_repeats"]
            s += f"; {m:.0f} ± {sd:.0f}% over {n} noise draws" if lang == "en" else f"; {m:.0f} ± {sd:.0f}% qua {n} lần lặp nhiễu"
        return s

    for lang, path in (("en", "README.md"), ("vi", "README_vi.md")):
        if not os.path.exists(path):
            continue
        H = {"en": ("System", "Uses at run time", "Success [95% CI]", "Mean max coverage", "DP score", "Stage", "Best run of the stage",
                    "Training data", "Success", "Evolution of the project (best run of each stage)",
                    "95 paired starts, 300 steps, success = coverage > 0.95. All runs: [results/summary/benchmark_table.md](results/summary/benchmark_table.md)."),
             "vi": ("Hệ thống", "Dùng lúc chạy", "Thành công [KTC 95%]", "Độ phủ TB", "Điểm DP", "Giai đoạn", "Lần chạy tốt nhất của giai đoạn",
                    "Dữ liệu học", "Thành công", "Tiến hoá của dự án (lần chạy tốt nhất mỗi giai đoạn)",
                    "95 điểm xuất phát ghép cặp, 300 bước, thành công = độ phủ > 0,95. Mọi lần chạy: [results/summary/benchmark_table_vi.md](results/summary/benchmark_table_vi.md).")}[lang]
        L = [H[10], "", f"| {H[0]} | {H[1]} | {H[2]} | {H[3]} | {H[4]} |", "|---|---|---|---|---|"]
        for lab, en, vi in README_ROWS:
            if lab in by:
                r = by[lab]
                name = f"**`{lab}`**" if lab == FINAL else f"`{lab}`"
                L.append(f"| {name} | {en if lang == 'en' else vi} | {succ(r, lang)} | {r['mean_max_coverage'] * 100:.1f}% | {r['dp_score']:.3f} |")
        L += ["", f"**{H[9]}**", "", f"| {H[5]} | {H[6]} | {H[7]} | {H[8]} | {H[3]} |", "|---|---|---|---|---|"]
        for st in sorted({r["stage"] for r in rows}):
            best = max((r for r in rows if r["stage"] == st), key=lambda r: (r["success"], r["mean_max_coverage"]))
            name = reg["stages"].get(st, {}).get(lang, st)
            L.append(f"| {st} — {name} | `{best['label']}` | {data_text(best['train_data'], lang)} | {pct(best['success'])} | {best['mean_max_coverage'] * 100:.1f}% |")
        text = open(path, encoding="utf-8").read()
        a, b = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
        if a in text and b in text:
            text = text[:text.index(a) + len(a)] + "\n" + "\n".join(L) + "\n" + text[text.index(b):]
            open(path, "w", encoding="utf-8").write(text)


def main():
    reg, runs = load()
    groups = {}
    for lab, d in runs.items():
        base, noise, rerun = split_label(lab)
        g = groups.setdefault(base, {"noise": {}, "rerun": {}})
        if rerun:
            g["rerun"][rerun] = d
        else:
            g["noise"][noise] = d
    final_s = metrics(groups[FINAL]["noise"][0])["_succ"] if FINAL in groups else None
    rows, determinism = [], []
    order = list(reg["runs"]) + sorted(set(groups) - set(reg["runs"]))
    for base in order:
        if base not in groups or 0 not in groups[base]["noise"]:
            continue
        g, info = groups[base], reg["runs"].get(base, {})
        m0 = metrics(g["noise"][0])
        reps = [metrics(g["noise"][k])["success"] for k in sorted(g["noise"])]
        row = {"label": base, "stage": re.match(r"E\d", base).group(0) if re.match(r"E\d", base) else base.split("_")[0],
               "date": info.get("date", ""), "train_data": info.get("train_data", ""),
               "description_en": info.get("en", ""), "description_vi": info.get("vi", ""), "policy": g["noise"][0].get("policy"),
               "ckpt_dir": g["noise"][0].get("ckpt_dir"), "vision_backend": g["noise"][0].get("vision_backend") or "",
               "n_starts": m0["n"], "success": m0["success"], "success_ci_low": m0["ci"][0], "success_ci_high": m0["ci"][1],
               "noise_repeats": len(reps), "success_mean_over_noise": float(np.mean(reps)),
               "success_sd_over_noise": float(np.std(reps, ddof=1)) if len(reps) > 1 else float("nan"),
               "mean_max_coverage": m0["coverage"], "dp_score": m0["dp_score"], "severe_lt50": m0["severe"],
               **{f"success_{k}": v for k, v in m0["suites"].items()}}
        if final_s is not None and base != FINAL:
            row["vs_final_only_this"], row["vs_final_only_final"], row["vs_final_p"] = mcnemar(m0["_succ"], final_s)
        rows.append(row)
        for tag, d in g["rerun"].items():
            a = [e["max_coverage"] for e in g["noise"][0]["episodes"]]
            b = [e["max_coverage"] for e in d["episodes"]]
            determinism.append({"label": base, "rerun": tag, "identical_starts": int(sum(abs(x - y) < 1e-12 for x, y in zip(a, b))), "n": len(a)})

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "runs.csv"), "w", newline="", encoding="utf-8") as f:
        keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    json.dump({"final": FINAL, "rows": rows, "determinism": determinism, "stages": reg["stages"]},
              open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8"), indent=1)

    pct = lambda x: "-" if x != x else f"{x * 100:.0f}%"  # noqa: E731
    for lang in ("en", "vi"):
        T = {"en": {"title": "Closed-loop benchmark: every run", "cfg": "Configuration", "data": "Training data",
                    "succ": "Success [95% CI]", "noise": "Success over noise repeats", "cov": "Mean max coverage", "dp": "DP score",
                    "suite": "seeds50 / hard20 / random25", "vs": f"vs {FINAL} (only this / only final, p)",
                    "note": "95 paired starts (50 gym-pusht seeds, 20 hard placements, 25 random placements), 300 steps, success = coverage > 0.95. "
                            "Noise repeats: same models and starts, other diffusion noise (mean ± sd over n runs). "
                            "McNemar: exact test on success, paired by start.",
                    "det": "Determinism check (exact re-runs)", "det_row": "identical starts"},
             "vi": {"title": "Benchmark vòng kín: mọi lần chạy", "cfg": "Cấu hình", "data": "Dữ liệu học",
                    "succ": "Thành công [KTC 95%]", "noise": "Thành công qua các lần lặp nhiễu", "cov": "Độ phủ TB", "dp": "Điểm DP",
                    "suite": "seeds50 / hard20 / random25", "vs": f"so với {FINAL} (chỉ cấu hình này / chỉ bản cuối, p)",
                    "note": "95 điểm xuất phát ghép cặp (50 seed gym-pusht, 20 vị trí khó, 25 vị trí ngẫu nhiên), 300 bước, thành công = độ phủ > 0,95. "
                            "Lặp nhiễu: cùng mô hình, cùng điểm xuất phát, nhiễu khuếch tán khác (TB ± độ lệch qua n lần). "
                            "McNemar: kiểm định chính xác trên thành công, ghép theo điểm xuất phát.",
                    "det": "Kiểm tra tính tất định (chạy lại y hệt)", "det_row": "điểm xuất phát trùng khớp"}}[lang]
        lines = [f"# {T['title']}", "", T["note"], ""]
        stage = None
        for r in rows:
            if r["stage"] != stage:
                stage = r["stage"]
                st = reg["stages"].get(stage, {}).get(lang, stage)
                lines += ["", f"## {stage} — {st}", "",
                          f"| {T['cfg']} | {T['data']} | {T['succ']} | {T['noise']} | {T['cov']} | {T['dp']} | {T['suite']} | {T['vs']} |",
                          "|---|---|---|---|---|---|---|---|"]
            noise = f"{r['success_mean_over_noise'] * 100:.1f} ± {r['success_sd_over_noise'] * 100:.1f} (n={r['noise_repeats']})" if r["noise_repeats"] > 1 else "-"
            vs = f"{r['vs_final_only_this']} / {r['vs_final_only_final']}, p={r['vs_final_p']:.3f}" if "vs_final_p" in r else "—"
            name = f"**`{r['label']}`**" if r["label"] == FINAL else f"`{r['label']}`"
            desc = r[f"description_{lang}"]
            lines.append(f"| {name}<br>{desc} | {data_text(r['train_data'], lang)} | {pct(r['success'])} [{pct(r['success_ci_low'])}–{pct(r['success_ci_high'])}] | "
                         f"{noise} | {r['mean_max_coverage'] * 100:.1f}% | {r['dp_score']:.3f} | "
                         f"{' / '.join(pct(r['success_' + k]) for k in SUITES)} | {vs} |")
        if determinism:
            lines += ["", f"## {T['det']}", ""] + [f"- `{d['label']}` rerun {d['rerun']}: {d['identical_starts']}/{d['n']} {T['det_row']}" for d in determinism]
        name = "benchmark_table.md" if lang == "en" else "benchmark_table_vi.md"
        open(os.path.join(OUT, name), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    # one generated results table per stage folder (experiments/<stage>/results.md), bilingual
    for stage, folder in STAGE_DIRS.items():
        sr = [r for r in rows if r["stage"] == stage]
        if not sr or not os.path.isdir(folder):
            continue
        st = reg["stages"].get(stage, {})
        lines = [f"# {stage} — results / kết quả", "",
                 f"*{st.get('en', '')} / {st.get('vi', '')}.* Generated by `scripts/5_evaluation/summarize_results.py` from "
                 "`results/benchmark/*.json` — do not edit by hand. / Sinh tự động, không sửa tay.", "",
                 "95 paired starts, 300 steps, success = coverage > 0.95; CI = 95% Wilson interval; "
                 f"McNemar vs the final system `{FINAL}`. / 95 điểm xuất phát ghép cặp, 300 bước, thành công = độ phủ > 0,95.", "",
                 "| Run / Lần chạy | Training data / Dữ liệu học | Success [95% CI] | Noise repeats (mean ± sd) | Mean max coverage | DP score | seeds50 / hard20 / random25 | vs final (only this / only final, p) |",
                 "|---|---|---|---|---|---|---|---|"]
        for r in sr:
            noise = f"{r['success_mean_over_noise'] * 100:.1f} ± {r['success_sd_over_noise'] * 100:.1f} (n={r['noise_repeats']})" if r["noise_repeats"] > 1 else "-"
            vs = f"{r['vs_final_only_this']} / {r['vs_final_only_final']}, p={r['vs_final_p']:.3f}" if "vs_final_p" in r else "— (final)"
            lines.append(f"| `{r['label']}`<br>{r['description_en']}<br>*{r['description_vi']}* | {r['train_data']} | "
                         f"{pct(r['success'])} [{pct(r['success_ci_low'])}–{pct(r['success_ci_high'])}] | {noise} | "
                         f"{r['mean_max_coverage'] * 100:.1f}% | {r['dp_score']:.3f} | {' / '.join(pct(r['success_' + k]) for k in SUITES)} | {vs} |")
        det = [d for d in determinism if d["label"].startswith(stage)]
        if det:
            lines += ["", "Determinism / tính tất định: " + "; ".join(f"`{d['label']}` rerun {d['rerun']}: {d['identical_starts']}/{d['n']} identical starts" for d in det)]
        open(os.path.join(folder, "results.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    readme_blocks(rows, reg, pct)
    print(f"{len(rows)} configurations, {len(runs)} runs -> {OUT}/benchmark_table.md, benchmark_table_vi.md, runs.csv, summary.json, "
          "experiments/*/results.md, README result blocks")
    for r in rows:
        print(f"  {r['label']:<42} {pct(r['success']):>4}  cov {r['mean_max_coverage'] * 100:5.1f}%  repeats {r['noise_repeats']}")


if __name__ == "__main__":
    main()
