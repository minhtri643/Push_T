# E2 — Hand-written controllers (archived) / Bộ điều khiển viết tay (đã lưu trữ)

[English](#english) · [Tiếng Việt](#tiếng-việt) · Numbers / số liệu: [results.md](results.md)

## English

**Date:** 2026-11-18/19. **Models:** E1 (`checkpoints/E1_augfix`). Kept for the ablation section of the report; **not** part of the final system.

| File (`code/`) | What it does |
|---|---|
| `controllers/tactical_controller.py` | 3-phase ORIENT → ALIGN → PUSH controller written by hand; hands over to the DiT for docking. Reads the true block state from the physics engine. |
| `controllers/collision_guard.py` | Non-penetration filter: re-routes pusher targets that would cut through the T-block. |
| `controllers/wm_lookahead.py` | Optimal stopping with the *old* latent World Model (truncate / lock before overshoot). |
| `controllers/wm_mppi_planner.py` | MPPI planning with the old latent World Model plus geometric affordances. |
| `benchmark_controllers.py` | **Re-measurement on the standard paired benchmark** (2026-11-23), 7 configurations → `results/benchmark/E2_*.json`. |
| `benchmark_tactical.py`, `play_test_suite.py` | The original scripts. Caveats: single `_set_state` (up to 90 px misplacement), success counted at 78%. |
| `legacy_interactive_demo.py` | The previous demo with all of the above as toggles. |

**What the re-measurement shows** (95 paired starts, 2026-11-23; complete table: [results.md](results.md)):

| Configuration | Success | vs E1 DiT + ID alone (41%) |
|---|---|---|
| + collision guard | 27% | worse |
| + old-WM lookahead | 19% | worse |
| + guard + lookahead (old demo default) | 13% | worse |
| tactical, locks at 88% / 80% | 0% | the locks stop every episode before 95% |
| tactical, guard + lookahead, **no locks** | **58%** | **better**, McNemar p = 0.033 |
| tactical, guard, no lookahead, no locks | 53% | p = 0.14 |
| tactical, no guard, no lookahead, no locks | 38% | — |

- **Hand-written rules can help:** without its locks, the tactical controller improves the E1 models from 41% to 53–58%.
- **But it uses the true block state of the simulator and hand-written geometry.** With the **same E1 models**, the learned dynamics World Model of E3 reaches **84%** (p < 0.001 against the best tactical configuration). That is why the controllers were archived.

**Why the guard, the lookahead and the locks hurt:**
- **The old latent World Model barely reacts to the action.** Counterfactual sensitivity is about 5%: 4.8% when re-evaluated on HF data in `results/world_model_eval/E4_final/metrics.json`. Its 16-step prediction (23.7 px) is hardly better than "the block stays still" (25.6 px). So lookahead and MPPI decisions built on it are close to noise, and its LOCK rule stops episodes before 95%.
- **The guard hurts the learned policy.** It blocks 7.7% of the human demonstrators' own pushing actions, while real penetration by the learned policy happens in about 0.04% of steps. The tactical controller needs the guard because its hand-written waypoints cut through the block.
- **The tactical controller uses privileged information.** It is the most robust option for far / inverted starts, but it is a hand-written rule on simulator state, not something the models learned.

**First measurement (2026-11-19 review, 95 starts; per-start files were not kept):**

| Configuration | Coverage | Success | < 50% |
|---|---|---|---|
| DiT + inverse dynamics, nothing else | 79.7% | 45% | 14 |
| + collision guard | 67.3% | 26% | 29 |
| + old-WM lookahead | 78.3% | 20% | 14 |
| + guard + lookahead (old demo default) | 66.9% | 14% | 29 |
| tactical, default (locks at 88%) | 85.2% | 0% | 5 |
| tactical, no locks, no lookahead, guard on | 90.3% | 47% | 5 |
| tactical, no locks, no lookahead, guard off | 74.9% | 40% | 20 |

The re-measurement with `benchmark_controllers.py` uses exactly the seeds and protocol of every other run, so it can be paired with them.
- Rows that exist in both measurements agree within 1–2 points (guard 27 vs 26%, lookahead 19 vs 20%, both 13 vs 14%, tactical default 0 vs 0%, no guard 38 vs 40%).
- The "no locks" rows differ more (58% vs 36%, 53% vs 47%). The first review's "lock off" row most likely still had the peak-settle lock (80%), which ends episodes early; the re-measured configuration disables both locks. The re-measured definitions are written in `code/benchmark_controllers.py`.

**Lesson.** Hand-written rules on privileged state can patch part of a weak policy (+17 points). A learned World Model that predicts the effect of the actions does much more (+43 points with the same models) and needs no hand-written geometry. Rules built on a World Model that ignores the actions (lookahead, locks) only hurt.

```
python experiments/E2_handcrafted_controllers/code/benchmark_controllers.py
```

## Tiếng Việt

**Ngày:** 18–19/11/2026. **Mô hình:** E1. Giữ lại cho phần thí nghiệm loại trừ (ablation) của báo cáo, **không** thuộc hệ thống cuối.

**Đo lại ngày 23/11 trên 95 điểm xuất phát ghép cặp** (bảng đầy đủ: [results.md](results.md)):
- Bộ chặn xuyên thấu, nhìn trước và luật khoá **làm hại**: 27%, 19%, 13% và 0%, so với 41% của E1.
- **Bộ điều khiển 3 nhịp khi bỏ luật khoá thì có giúp:** 41% → 58% (p = 0,033). Nhưng nó dùng trạng thái thật của mô phỏng và hình học viết tay.
- **Với cùng mô hình E1**, World Model động lực học học được ở E3 đạt **84%** (p < 0,001 so với cấu hình tactical tốt nhất). Vì vậy các bộ điều khiển được lưu trữ.

**Vì sao bộ chặn, nhìn trước và luật khoá làm hại:**
- **World Model ẩn cũ gần như không phản ứng với hành động.** Độ nhạy khoảng 5%. Dự đoán 16 bước (23,7 px) gần như không tốt hơn "khối đứng yên" (25,6 px). Vì vậy quyết định nhìn trước / MPPI dựng trên nó gần như ngẫu nhiên, và luật khoá của nó dừng ván trước khi đạt 95%.
- **Bộ chặn xuyên thấu làm hại chính sách học được.** Nó chặn 7,7% hành động đẩy của người điều khiển thật, trong khi xuyên thấu thật chỉ xảy ra ở khoảng 0,04% số bước. Bộ điều khiển 3 nhịp cần bộ chặn vì các điểm đi viết tay của nó cắt xuyên khối.
- **Bộ điều khiển 3 nhịp dùng thông tin đặc quyền.** Nó vững nhất với vị trí xa / lật ngược, nhưng là luật viết tay trên trạng thái mô phỏng, không phải thứ mô hình học được.

**So với lần đo đầu:** các cấu hình có ở cả hai lần đo khớp trong phạm vi 1–2 điểm. Riêng cấu hình "không khoá" lệch nhiều hơn (58% so với 36%), có lẽ vì lần đầu vẫn giữ luật khoá 80%; lần đo lại tắt cả hai luật khoá.

**Bài học.** Luật viết tay trên thông tin đặc quyền vá được một phần policy yếu (+17 điểm). World Model học được, biết dự đoán tác động của hành động, làm tốt hơn nhiều (+43 điểm với cùng mô hình) và không cần hình học viết tay. Luật dựng trên một World Model không để ý tới hành động (nhìn trước, luật khoá) chỉ làm hại.
