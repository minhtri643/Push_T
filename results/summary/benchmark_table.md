# Closed-loop benchmark: every run

95 paired starts (50 gym-pusht seeds, 20 hard placements, 25 random placements), 300 steps, success = coverage > 0.95. Noise repeats: same models and starts, other diffusion noise (mean ± sd over n runs). McNemar: exact test on success, paired by start.


## E0 — Original baseline

| Configuration | Training data | Success [95% CI] | Success over noise repeats | Mean max coverage | DP score | seeds50 / hard20 / random25 | vs E4_final__dit_wm (only this / only final, p) |
|---|---|---|---|---|---|---|---|
| `E0_baseline__dit_id`<br>original code and models (trained before 2026-11-16), DiT + inverse dynamics | zarr | 2% [1%–7%] | - | 34.6% | 0.364 | 0% / 5% / 4% | 0 / 70, p=0.000 |

## E1 — Evaluation protocol + augmentation fix

| Configuration | Training data | Success [95% CI] | Success over noise repeats | Mean max coverage | DP score | seeds50 / hard20 / random25 | vs E4_final__dit_wm (only this / only final, p) |
|---|---|---|---|---|---|---|---|
| `E1_augfix__dit_id`<br>retrained with the corrected anti-diagonal reflection augmentation | zarr | 41% [32%–51%] | - | 77.3% | 0.809 | 46% / 45% / 28% | 7 / 40, p=0.000 |

## E2 — Hand-written controllers (archived)

| Configuration | Training data | Success [95% CI] | Success over noise repeats | Mean max coverage | DP score | seeds50 / hard20 / random25 | vs E4_final__dit_wm (only this / only final, p) |
|---|---|---|---|---|---|---|---|
| `E2_guard`<br>E1 models + non-penetration guard | zarr | 27% [19%–37%] | - | 67.0% | 0.702 | 30% / 20% / 28% | 10 / 56, p=0.000 |
| `E2_lookahead`<br>E1 models + optimal stopping with the old latent World Model | zarr | 19% [12%–28%] | - | 76.0% | 0.798 | 26% / 15% / 8% | 2 / 56, p=0.000 |
| `E2_guard_lookahead`<br>guard + lookahead (old demo default) | zarr | 13% [7%–21%] | - | 66.4% | 0.697 | 16% / 5% / 12% | 4 / 64, p=0.000 |
| `E2_tactical_default`<br>3-phase tactical controller, guard + lookahead, locks at 88% / settle 80% | zarr | 0% [0%–4%] | - | 86.8% | 0.914 | 0% / 0% / 0% | 0 / 72, p=0.000 |
| `E2_tactical_no_lock`<br>tactical, guard + lookahead, no locks | zarr | 58% [48%–67%] | - | 91.6% | 0.957 | 64% / 45% / 56% | 12 / 29, p=0.012 |
| `E2_tactical_best`<br>tactical, guard, no lookahead, no locks | zarr | 53% [43%–62%] | - | 91.7% | 0.958 | 60% / 45% / 44% | 12 / 34, p=0.002 |
| `E2_tactical_no_guard`<br>tactical, no guard, no lookahead, no locks | zarr | 38% [29%–48%] | - | 74.6% | 0.781 | 44% / 30% / 32% | 9 / 45, p=0.000 |

## E3 — Action-conditioned dynamics World Model (dit_wm)

| Configuration | Training data | Success [95% CI] | Success over noise repeats | Mean max coverage | DP score | seeds50 / hard20 / random25 | vs E4_final__dit_wm (only this / only final, p) |
|---|---|---|---|---|---|---|---|
| `E3_augfix_ckpt__dit_wm`<br>E1 models + actions chosen by the new dynamics World Model (64 candidates) | zarr (+ simulator data for the dynamics) | 84% [76%–90%] | 81.1 ± 4.5 (n=2) | 94.5% | 0.985 | 82% / 80% / 92% | 19 / 11, p=0.200 |
| `E3_dit100ep_rmse_id__dit_id`<br>DiT retrained 100 epochs + EMA; inverse dynamics chosen by offline RMSE | zarr | 3% [1%–9%] | - | 50.7% | 0.533 | 2% / 5% / 4% | 2 / 71, p=0.000 |
| `E3_dit100ep_rmse_id__dit_wm`<br>same, actions chosen by the dynamics World Model | zarr (+ simulator data for the dynamics) | 40% [31%–50%] | - | 83.1% | 0.870 | 42% / 40% / 36% | 15 / 49, p=0.000 |
| `E3_dit100ep_closedloop_id25__dit_id`<br>inverse dynamics chosen by closed-loop snapshot selection (<= 25 epochs) | zarr | 51% [41%–60%] | - | 78.1% | 0.816 | 52% / 55% / 44% | 13 / 37, p=0.001 |
| `E3_dit100ep_closedloop_id25__dit_wm`<br>same, actions chosen by the dynamics World Model | zarr (+ simulator data for the dynamics) | 72% [62%–80%] | - | 90.1% | 0.942 | 76% / 60% / 72% | 19 / 23, p=0.644 |
| `E3_dit100ep_closedloop_id60__dit_id`<br>inverse dynamics chosen by closed-loop snapshot selection (<= 60 epochs) | zarr | 64% [54%–73%] | - | 86.0% | 0.899 | 76% / 55% / 48% | 14 / 25, p=0.108 |
| `E3_dit100ep_closedloop_id60__dit_wm`<br>same, actions chosen by the dynamics World Model | zarr (+ simulator data for the dynamics) | 74% [64%–81%] | 71.6 ± 3.0 (n=2) | 92.4% | 0.965 | 80% / 60% / 72% | 16 / 18, p=0.864 |
| `E3_dit100ep_history2__dit_id`<br>2-frame observation history | zarr | 56% [46%–65%] | - | 84.5% | 0.884 | 64% / 30% / 60% | 9 / 28, p=0.003 |
| `E3_dit100ep_history2__dit_wm`<br>same, actions chosen by the dynamics World Model | zarr (+ simulator data for the dynamics) | 78% [69%–85%] | - | 92.0% | 0.962 | 72% / 85% / 84% | 18 / 16, p=0.864 |
| `E3_dit100ep_with_augfix_id__dit_id`<br>100-epoch DiT with the E1 inverse dynamics | zarr | 58% [48%–67%] | - | 81.6% | 0.853 | 66% / 60% / 40% | 13 / 30, p=0.014 |
| `E3_dit100ep_with_augfix_id__dit_wm`<br>same, actions chosen by the dynamics World Model | zarr (+ simulator data for the dynamics) | 75% [65%–82%] | - | 92.0% | 0.962 | 78% / 60% / 80% | 17 / 18, p=1.000 |
| `E3_action_diffusion_baseline`<br>Diffusion-Policy-style baseline: diffuses 16 actions from the image latent | zarr | 32% [23%–41%] | - | 88.3% | 0.926 | 34% / 25% / 32% | 6 / 48, p=0.000 |

## E4 — Hugging Face data only (FINAL)

| Configuration | Training data | Success [95% CI] | Success over noise repeats | Mean max coverage | DP score | seeds50 / hard20 / random25 | vs E4_final__dit_wm (only this / only final, p) |
|---|---|---|---|---|---|---|---|
| `E4_hf_round0__dit_id`<br>HF-only retrain, block pose fitted before the image-offset fix | lerobot/pusht only | 58% [48%–67%] | - | 88.2% | 0.923 | 58% / 80% / 40% | 12 / 29, p=0.012 |
| `E4_hf_round0__dit_wm`<br>same, actions chosen by the dynamics World Model | lerobot/pusht only | 56% [46%–65%] | - | 90.1% | 0.942 | 54% / 70% / 48% | 12 / 31, p=0.005 |
| `E4_final__dit_id`<br>FINAL models, DiT + inverse dynamics | lerobot/pusht only | 59% [49%–68%] | 54.7 ± 5.6 (n=3) | 87.4% | 0.914 | 62% / 60% / 52% | 11 / 27, p=0.014 |
| **`E4_final__dit_wm`**<br>FINAL system: actions chosen by the dynamics World Model (demo default) | lerobot/pusht only | 76% [66%–83%] | 70.5 ± 4.8 (n=3) | 94.6% | 0.988 | 78% / 80% / 68% | — |

## E5 — Vision-only extension

| Configuration | Training data | Success [95% CI] | Success over noise repeats | Mean max coverage | DP score | seeds50 / hard20 / random25 | vs E4_final__dit_wm (only this / only final, p) |
|---|---|---|---|---|---|---|---|
| `E5a_dinov2_zarr_sim__dit_id_vision`<br>vision only: block pose read by a probe on frozen DINOv2 features (E1 planner) | zarr + simulator frames | 35% [26%–45%] | - | 76.8% | 0.804 | 30% / 30% / 48% | 10 / 49, p=0.000 |
| `E5a_dinov2_zarr_sim__dit_vwm`<br>same, actions chosen by the vision World Model | zarr + simulator frames | 8% [4%–16%] | - | 84.8% | 0.892 | 12% / 0% / 8% | 4 / 68, p=0.000 |
| `E5b_jepa__dit_id_vision`<br>vision only: self-trained JEPA encoder + pose probe | lerobot/pusht only | 45% [36%–55%] | 44.6 ± 1.2 (n=3) | 73.0% | 0.764 | 48% / 50% / 36% | 9 / 38, p=0.000 |
| `E5b_jepa__dit_vwm`<br>same, actions chosen by the vision World Model | lerobot/pusht only | 20% [13%–29%] | - | 76.7% | 0.805 | 26% / 20% / 8% | 5 / 58, p=0.000 |
| `E5c_dinov2_frozen__dit_id_vision`<br>vision only: frozen DINOv2 ViT-S + pose probe | lerobot/pusht only (+ frozen pretrained DINOv2) | 49% [40%–59%] | 52.6 ± 3.2 (n=3) | 83.9% | 0.877 | 56% / 30% / 52% | 11 / 36, p=0.000 |
| `E5c_dinov2_frozen__dit_vwm`<br>same, actions chosen by the vision World Model | lerobot/pusht only (+ frozen pretrained DINOv2) | 13% [7%–21%] | - | 72.0% | 0.756 | 10% / 30% / 4% | 2 / 62, p=0.000 |
| `E5d_vjepa2_frozen__dit_id_vision`<br>vision only: frozen V-JEPA 2 ViT-L + pose probe | lerobot/pusht only (+ frozen pretrained V-JEPA 2) | 54% [44%–63%] | 53.7 ± 2.1 (n=3) | 85.4% | 0.894 | 56% / 55% / 48% | 15 / 36, p=0.005 |
| `E5d_vjepa2_frozen__dit_vwm`<br>same, actions chosen by the vision World Model | lerobot/pusht only (+ frozen pretrained V-JEPA 2) | 11% [6%–18%] | - | 73.9% | 0.776 | 12% / 5% / 12% | 2 / 64, p=0.000 |

## Determinism check (exact re-runs)

- `E4_final__dit_id` rerun 2026-11-23: 95/95 identical starts
- `E4_final__dit_wm` rerun 2026-11-23-parallel: 95/95 identical starts
- `E4_final__dit_wm` rerun 2026-11-23: 95/95 identical starts
