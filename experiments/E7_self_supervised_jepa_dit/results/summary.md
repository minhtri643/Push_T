# E7 results

## 1. Reproduction of the original result (the original 50 seeds 100000-100049, DDIM 16, 8 steps executed, noise = env seed)

| Run | Success@0.95 | 95% CI | Success@0.90 | Mean max coverage | Same success as the original evaluator (E7 re-training) |
|---|---|---|---|---|---|
| Original run (GTX 1650) | 26% (13/50) | 16%–40% | 50% | 77.6% | - |
| E7 re-training, the original evaluator (src/eval_sim.py) | 20% (10/50) | 11%–33% | 48% | 72.8% | - |
| E7 re-training, this repository's loop, no limit (adapter check) | 20% (10/50) | 11%–33% | 48% | 72.8% | 50/50 (identical coverage on 48/50) |
| E7 re-training, this repository's loop, 40 px limit | 24% (12/50) | 14%–37% | 52% | 70.1% | 32/50 (identical coverage on 2/50) |

With the 40 px limit, 1371 of 13542 executed targets were changed (10.1%).

### Offline metrics (20 validation episodes)

| Metric | Original run | E7 fp32 (used) | E7 with TF32 (not used) |
|---|---|---|---|
| WM epochs | 36 | 40 | 40 |
| best val latent loss | 0.1407 | 0.1446 | 0.1371 |
| std of S_t | 0.840 | 0.870 | 0.834 |
| effective rank of S_t | 137.6 | 140.5 | 138.4 |
| probe R² block position | 0.904 | 0.881 | 0.916 |
| probe R² block angle | 0.595 | 0.558 | 0.656 |
| probe position error (px) | 20.8 | 23.3 | 19.5 |
| probe angle error (°) | 18.5 | 20.0 | 16.7 |
| DiT FDE position at 2 s (px) | 22.6 | 21.3 | - |
| DiT FDE angle at 2 s (°) | 13.9 | 13.4 | - |
| DiT FDE_obj | 37.2 | 35.4 | - |
| DiT minFDE_obj@5 | 21.2 | 19.9 | - |

## 2. The 95-start benchmark of E0-E6 (paired starts, 300 steps)

| Policy | Success per noise draw | Mean ± sd (%) | 95% CI (draw 0) | Success@0.90 | Mean max coverage | DP score |
|---|---|---|---|---|---|---|
| **E7 JEPA + DiT, 40 px limit** (images + pusher only) | 32% / 26% / 32% | 29.8 ± 3.0 | 23%–41% | 52% | 68.7% | 0.720 |
| **E7 JEPA + DiT, no limit** (images + pusher only) | 24% / 35% / 33% | 30.5 ± 5.6 | 17%–34% | 53% | 69.2% | 0.726 |
| E0 baseline (`dit_id`) | 2% | 2.1 | 1%–7% | 4% | 34.6% | 0.364 |
| E3 action diffusion (DiT → actions, frozen WM encoder) | 32% | 31.6 | 23%–41% | 80% | 88.3% | 0.926 |
| E4 final `dit_wm` *(true block pose)* | 76% / 69% / 66% | 70.5 ± 4.8 | 66%–83% | 96% | 93.2% | 0.973 |
| E4 `dit_id` *(true block pose)* | 59% / 48% / 57% | 54.7 ± 5.6 | 49%–68% | 84% | 86.7% | 0.907 |
| E5b JEPA `dit_id_vision` | 45% / 43% / 45% | 44.6 ± 1.2 | 36%–55% | 67% | 74.0% | 0.775 |
| E5c DINOv2 `dit_id_vision` | 49% / 53% / 56% | 52.6 ± 3.2 | 40%–59% | 79% | 85.1% | 0.890 |
| E5d V-JEPA 2 `dit_id_vision` | 54% / 56% / 52% | 53.7 ± 2.1 | 44%–63% | 81% | 86.4% | 0.903 |

Paired tests against E7 (per start; pooled = three noise draws):

| vs | 40 px limit: E7 better / worse (pooled), p | McNemar draw 0, p | no limit: E7 better / worse (pooled), p | McNemar draw 0, p |
|---|---|---|---|---|
| E0 baseline (`dit_id`) | - | 30 / 2, p = 0.0000 | - | 22 / 1, p = 0.0000 |
| E3 action diffusion (DiT → actions, frozen WM encoder) | - | 16 / 16, p = 1.0000 | - | 14 / 21, p = 0.3105 |
| E4 final `dit_wm` *(true block pose)* | 10 / 73, p = 0.0000 | 7 / 49, p = 0.0000 | 7 / 72, p = 0.0000 | 5 / 54, p = 0.0000 |
| E4 `dit_id` *(true block pose)* | 16 / 57, p = 0.0000 | 11 / 37, p = 0.0002 | 16 / 53, p = 0.0000 | 10 / 43, p = 0.0000 |
| E5b JEPA `dit_id_vision` | 22 / 44, p = 0.0092 | 15 / 28, p = 0.0660 | 18 / 45, p = 0.0009 | 11 / 31, p = 0.0029 |
| E5c DINOv2 `dit_id_vision` | 14 / 50, p = 0.0000 | 17 / 34, p = 0.0241 | 15 / 50, p = 0.0000 | 5 / 29, p = 0.0000 |
| E5d V-JEPA 2 `dit_id_vision` | 17 / 52, p = 0.0000 | 13 / 34, p = 0.0031 | 17 / 51, p = 0.0000 | 11 / 39, p = 0.0001 |

E7 no limit vs E7 40 px limit (pooled): 25 better / 27 worse, p = 0.8899.
E7 40 px limit, per suite (draw 0), success / mean max coverage: seeds50 28% / 73%, hard20 30% / 74%, random25 40% / 68%; targets beyond 40 px: 3009 of 25757 (11.7%).
E7 no limit, per suite (draw 0), success / mean max coverage: seeds50 26% / 68%, hard20 15% / 72%, random25 28% / 63%; targets beyond 40 px: 2505 of 26042 (9.6%).
Check of the pooled test: E4 `dit_wm` vs `dit_id` = 53 / 20, p = 0.0001 (published: p = 0.0001).

## 3. The original 50 seeds (100000-100049), every policy in the same episode loop

| Policy | Success per noise draw | Mean ± sd (%) | 95% CI (draw 0) | Success@0.90 | Mean max coverage | DP score |
|---|---|---|---|---|---|---|
| **E7 JEPA + DiT, 40 px limit** | 24% / 40% / 16% | 26.7 ± 12.2 | 14%–37% | 49% | 70.7% | 0.741 |
| **E7 JEPA + DiT, no limit** | 20% / 24% / 28% | 24.0 ± 4.0 | 11%–33% | 48% | 70.9% | 0.744 |
| E4 final `dit_wm` *(true block pose)* | 70% / 64% / 64% | 66.0 ± 3.5 | 56%–81% | 96% | 94.2% | 0.985 |
| E4 `dit_id` *(true block pose)* | 54% / 52% / 56% | 54.0 ± 2.0 | 40%–67% | 86% | 88.8% | 0.929 |
| E5b JEPA `dit_id_vision` | 48% / 44% / 40% | 44.0 ± 4.0 | 35%–61% | 73% | 79.8% | 0.835 |
| E5c DINOv2 `dit_id_vision` | 52% / 48% / 56% | 52.0 ± 4.0 | 39%–65% | 85% | 90.2% | 0.944 |
| E5d V-JEPA 2 `dit_id_vision` | 42% / 50% / 52% | 48.0 ± 5.3 | 29%–56% | 81% | 86.5% | 0.906 |

Paired tests against E7 on the same 50 seeds:

| vs | 40 px limit: E7 better / worse (pooled), p | McNemar draw 0, p | no limit: E7 better / worse (pooled), p | McNemar draw 0, p |
|---|---|---|---|---|
| E4 final `dit_wm` *(true block pose)* | 2 / 37, p = 0.0000 | 3 / 26, p = 0.0000 | 1 / 33, p = 0.0000 | 5 / 30, p = 0.0000 |
| E4 `dit_id` *(true block pose)* | 9 / 26, p = 0.0060 | 7 / 22, p = 0.0081 | 7 / 28, p = 0.0005 | 4 / 21, p = 0.0009 |
| E5b JEPA `dit_id_vision` | 13 / 25, p = 0.0730 | 8 / 20, p = 0.0357 | 9 / 28, p = 0.0026 | 4 / 18, p = 0.0043 |
| E5c DINOv2 `dit_id_vision` | 9 / 27, p = 0.0039 | 6 / 20, p = 0.0094 | 10 / 33, p = 0.0006 | 6 / 22, p = 0.0037 |
| E5d V-JEPA 2 `dit_id_vision` | 10 / 30, p = 0.0022 | 6 / 15, p = 0.0784 | 7 / 28, p = 0.0005 | 6 / 17, p = 0.0347 |

## 4. Ablations and sweeps of the original run

### 4a. The original evaluator on the original 50 seeds (one noise draw, noise = env seed): the original run vs the E7 re-run

Cells: success@0.95 / success@0.90 / mean max coverage / median time to success / inference time per plan.

| Configuration | Original run (GTX 1650) | E7 re-run (RTX 3050) | Same seed outcome original vs E7, McNemar p | Adapter check: this repository's loop = the original evaluator (E7 weights) |
|---|---|---|---|---|
| Main (k = 8, DDIM 16) | 26% / 50% / 0.776 / 16.3 s / 117 ms | 20% / 48% / 0.728 / 21.4 s / 74 ms | 33/50 same; original only 10, E7 only 7, p = 0.63 | 50/50 same success, identical coverage on 48/50 |
| A1: random-init encoder | 0% / 0% / 0.269 / - / 101 ms | not re-run | - | - |
| A2: images only (no pusher position) | 32% / 52% / 0.657 / 19.3 s / 111 ms | not re-run | - | - |
| n_action_steps = 4 | 20% / 36% / 0.652 / 22.0 s / 112 ms | 36% / 48% / 0.682 / 17.9 s / 81 ms | 28/50 same; original only 7, E7 only 15, p = 0.13 | - |
| n_action_steps = 6 | 32% / 52% / 0.744 / 18.1 s / 108 ms | 20% / 38% / 0.636 / 24.4 s / 108 ms | 28/50 same; original only 14, E7 only 8, p = 0.29 | - |
| n_action_steps = 12 | 18% / 54% / 0.780 / 23.3 s / 108 ms | 32% / 54% / 0.719 / 20.1 s / 148 ms | 35/50 same; original only 4, E7 only 11, p = 0.12 | - |
| DDIM 4 steps | 16% / 32% / 0.655 / 16.9 s / 35 ms | 14% / 36% / 0.626 / 18.4 s / 25 ms | 39/50 same; original only 6, E7 only 5, p = 1.00 | - |
| DDIM 8 steps | 22% / 58% / 0.799 / 26.1 s / 63 ms | 38% / 58% / 0.725 / 18.4 s / 42 ms | 32/50 same; original only 5, E7 only 13, p = 0.10 | - |

E7 inference times for k = 6 and k = 12 are not valid: those two runs shared the GPU with other evaluations. Success and coverage are not affected (the episode loop does not depend on timing).

Live demo on 20 fresh seeds (1000-1019, DDIM 8, scripts/demo_live.py): original run 5/20 (25%), mean max coverage 0.724; E7 re-run not re-run.

### 4b. Offline metrics: probe baselines and FDE of the ablations (original run / E7 re-run)

| Probe on S_t (20 val episodes) | R² block position | R² block angle | position error (px) | angle error (°) |
|---|---|---|---|---|
| JEPA S_t | 0.904 / 0.881 | 0.595 / 0.558 | 20.8 / 23.3 | 18.5 / 20.0 |
| random-init S_t | 0.687 / 0.687 | -0.046 / -0.046 | 40.4 / 40.4 | 40.5 / 40.5 |
| pixel PCA-256 | 0.821 / 0.821 | -0.124 / -0.124 | 28.1 / 28.1 | 42.8 / 42.8 |

| Policy, FDE of the block after 2 s (375 val samples) | FDE position (px) | FDE angle (°) | FDE_obj | minFDE_obj@5 |
|---|---|---|---|---|
| Main (JEPA + DiT) | 22.6 / 21.3 | 13.9 / 13.4 | 37.2 / 35.4 | 21.2 / 19.9 |
| Constant velocity (baseline) | 27.1 / 27.1 | 16.7 / 16.7 | 44.6 / 44.6 | - / - |
| Stand still (baseline) | 31.7 / 31.7 | 18.9 / 18.9 | 51.5 / 51.5 | - / - |
| A2: no pusher position | 24.2 / - | 14.8 / - | 39.8 / - | 22.8 / - |
| A1: random-init encoder | 34.2 / - | 21.5 / - | 56.7 / - | 30.3 / - |
