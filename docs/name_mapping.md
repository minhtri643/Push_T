# Name mapping (reorganisation 2026-11-23)

Benchmark labels before → after. `episodes sha256` is the hash of the per-episode results, identical before and after (only the label/metadata fields changed).

| Old label | New label | Checkpoint folder | Vision backend | episodes sha256 |
|---|---|---|---|---|
| `v2_dit_id` | `E1_augfix__dit_id` | `checkpoints/E1_augfix` | - | `333b5ea4f79f` |
| `v2_dit_wm` | `E3_augfix_ckpt__dit_wm` | `checkpoints/E1_augfix` | - | `334e118616e1` |
| `v2_dit_wm_seed1` | `E3_augfix_ckpt__dit_wm__noise1` | `checkpoints/E1_augfix` | - | `a0753b2053fd` |
| `v4_dit_id` | `E3_dit100ep_rmse_id__dit_id` | `checkpoints/E3_dit100ep_rmse_id` | - | `2a35564b3fcf` |
| `v4_dit_wm` | `E3_dit100ep_rmse_id__dit_wm` | `checkpoints/E3_dit100ep_rmse_id` | - | `722860ae0846` |
| `v4b_dit_id` | `E3_dit100ep_closedloop_id25__dit_id` | `checkpoints/E3_dit100ep_closedloop_id25` | - | `3d778a57b1af` |
| `v4b_dit_wm` | `E3_dit100ep_closedloop_id25__dit_wm` | `checkpoints/E3_dit100ep_closedloop_id25` | - | `cbdb637abae1` |
| `v4c_dit_id` | `E3_dit100ep_closedloop_id60__dit_id` | `checkpoints/E3_dit100ep_closedloop_id60` | - | `90fdc11ef3bc` |
| `v4c_dit_wm` | `E3_dit100ep_closedloop_id60__dit_wm` | `checkpoints/E3_dit100ep_closedloop_id60` | - | `f3fb35c83449` |
| `v4c_dit_wm_seed1` | `E3_dit100ep_closedloop_id60__dit_wm__noise1` | `checkpoints/E3_dit100ep_closedloop_id60` | - | `307f1122ffca` |
| `v4_h2_dit_id` | `E3_dit100ep_history2__dit_id` | `checkpoints/E3_dit100ep_history2` | - | `4c568d8e25f8` |
| `v4_h2_dit_wm` | `E3_dit100ep_history2__dit_wm` | `checkpoints/E3_dit100ep_history2` | - | `0183b142f0a9` |
| `mix_v4dit_v2id_dit_id` | `E3_dit100ep_with_augfix_id__dit_id` | `checkpoints/E3_dit100ep_with_augfix_id` | - | `7b6632dfb3f1` |
| `mix_v4dit_v2id_dit_wm` | `E3_dit100ep_with_augfix_id__dit_wm` | `checkpoints/E3_dit100ep_with_augfix_id` | - | `142c859c3de9` |
| `action_diffusion` | `E3_action_diffusion_baseline` | `checkpoints/E3_action_diffusion` | - | `e4474a4e0977` |
| `hf0_dit_id` | `E4_hf_round0__dit_id` | `checkpoints/E4_hf_round0` | - | `6346a1cb4cc1` |
| `hf0_dit_wm` | `E4_hf_round0__dit_wm` | `checkpoints/E4_hf_round0` | - | `d3cc57595b91` |
| `hf_dit_id` | `E4_final__dit_id` | `checkpoints/final` | - | `d8c2fed05c93` |
| `hf_dit_wm` | `E4_final__dit_wm` | `checkpoints/final` | - | `7deb33269d40` |
| `dit_id` | `E4_final__dit_id__rerun_2026-11-23` | `checkpoints/final` | - | `5e3da36e741b` |
| `vision_dit_id` | `E5a_dinov2_zarr_sim__dit_id_vision` | `checkpoints/E1_augfix` | vision_wm | `debfc9daf551` |
| `vision_dit_vwm` | `E5a_dinov2_zarr_sim__dit_vwm` | `checkpoints/E1_augfix` | vision_wm | `54389c5ad068` |
| `jepa_dit_id_vision` | `E5b_jepa__dit_id_vision` | `checkpoints/final` | jepa_wm | `db1b438d33fc` |
| `jepa_dit_vwm` | `E5b_jepa__dit_vwm` | `checkpoints/final` | jepa_wm | `b8e1496563ff` |
| `dinov2hf_dit_id_vision` | `E5c_dinov2_frozen__dit_id_vision` | `checkpoints/final` | dinov2_hf_wm | `3a2464c7ceb7` |
| `dinov2hf_dit_vwm` | `E5c_dinov2_frozen__dit_vwm` | `checkpoints/final` | dinov2_hf_wm | `d2aefbb0924f` |
| `vjepa2_dit_id_vision` | `E5d_vjepa2_frozen__dit_id_vision` | `checkpoints/final` | vjepa2_wm | `811818a967c7` |
| `vjepa2_dit_vwm` | `E5d_vjepa2_frozen__dit_vwm` | `checkpoints/final` | vjepa2_wm | `af1555d8905f` |
| `diag_v2` | `E3_diag_augfix` | `checkpoints/E1_augfix` | - | `7f0d7a9675dc` |
| `diag_v4` | `E3_diag_dit100ep_rmse_id` | `checkpoints/E3_dit100ep_rmse_id` | - | `04a46ac58f0d` |
| `diag_mix_v4dit_v2id` | `E3_diag_dit100ep_with_augfix_id` | `checkpoints/E3_dit100ep_with_augfix_id` | - | `8d57d74617ef` |
| `diag_mix_v2dit_v4id` | `E3_diag_augfix_dit_with_rmse_id` | `checkpoints/E3_augfix_dit_with_rmse_id` | - | `6f140e204987` |

Checkpoint folders before → after: `pusht_wm_dit/checkpoints/` (root files) → `checkpoints/final/`; `v2` → `E1_augfix`; `v4` → `E3_dit100ep_rmse_id`; `v4b` → `E3_dit100ep_closedloop_id25`; `v4c` → `E3_dit100ep_closedloop_id60`; `v4_h2` → `E3_dit100ep_history2`; `mix_v4dit_v2id` → `E3_dit100ep_with_augfix_id`; `mix_v2dit_v4id` → `E3_augfix_dit_with_rmse_id`; `backups/hf_round0` → `E4_hf_round0`; `backups/baseline_v1/checkpoints` → `E0_baseline`. Full file-by-file log: `docs/provenance/reorg_log_2026-11-23.json`.
