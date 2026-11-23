# E3 — Action-conditioned dynamics World Model / World Model động lực học theo hành động

[English](#english) · [Tiếng Việt](#tiếng-việt) · Numbers / số liệu: [results.md](results.md)

## English

**Date:** 2026-11-19/20. **Training data:** Diffusion Policy zarr; the dynamics model also used simulator interactions (`code/collect_sim_interactions.py`). Config of that time: `config_zarr_era.yaml`.

**What changed:**
1. **`PushDynamics`** (`pusht_wm_dit/models/dynamics.py`): the World Model part that the course asks for.
   - It predicts the change of the block pose (Δx, Δy, Δθ) from the current state and the robot's actions.
   - It is an MLP in the block's own frame, rolled out open-loop for K = 16 steps.
   - It is trained with the course's displacement error f(Δx, Δy, Δθ) accumulated over the 16 steps (px / deg, Huber), with moving windows weighted ×3.
2. **`dit_wm`: the World Model chooses the actions.**
   - Proposals: 64 candidate action sequences around the inverse-dynamics proposal, made of temporally smooth perturbations of scale 0, 0.02, 0.04 and 0.08.
   - Each candidate is imagined with `PushDynamics`.
   - The one whose imagined block trajectory best follows the DiT path is executed.
3. **DiT retrained** for 100 epochs with EMA, and three ways of choosing the inverse-dynamics snapshot:
   - offline RMSE (`rmse_id`);
   - closed-loop success on development seeds 90000+, never used by the benchmark (`closedloop_id25`, `closedloop_id60`);
   - plus a 2-frame observation history (`history2`) and mixes of old and new parts.
4. **Baseline in the style of Diffusion Policy:** diffuse 16 actions directly from the image latent (`action_diffusion`).

**Results** (complete table: [results.md](results.md)):
- **Dynamics accuracy** (measured 2026-11-19 on zarr validation episodes): 16-step error 0.7 px / 0.5°, versus 25.6 px for "the block stays still" and 22.7 px for the old latent World Model. Counterfactual sensitivity 99%, versus 5%.
  - That metrics file was later overwritten by the E4 evaluation.
  - The same design trained on HF data gives 1.06 px / 0.63° and 96.5% (`results/world_model_eval/E4_final/metrics.json`).
  - A dynamics model trained on demonstrations only is almost as accurate as one with extra simulator data: the **design** mattered more than the extra data.
- **The World Model as action selector is the biggest single gain:** E1 models + `dit_wm` = **84%**, the highest success measured in the project (zarr data). A second noise draw gives 78%.
- **Offline metrics mislead.**
  - The inverse dynamics has about 25 px RMSE at every snapshot, but its closed-loop success ranges from 0 to 65%.
  - Choosing the snapshot by RMSE gave **3%**; choosing it in closed loop gave 64%.
  - Training the inverse dynamics on DiT-sampled paths kept the offline error but dropped closed-loop success from 46% to 10% (training-time selection runs).
- A 2-frame observation did not help.
- The action-diffusion baseline reaches high coverage (88%) but rarely docks (32% success).

```
python scripts/5_evaluation/benchmark.py --policy dit_wm --ckpt-dir checkpoints/E1_augfix --label E3_augfix_ckpt__dit_wm
python scripts/5_evaluation/benchmark.py --policy dit_wm --ckpt-dir checkpoints/E3_dit100ep_closedloop_id60 --label E3_dit100ep_closedloop_id60__dit_wm
python scripts/5_evaluation/benchmark.py --policy action_diffusion --ckpt-dir checkpoints/E3_action_diffusion --label E3_action_diffusion_baseline
```

**Files:**
- Checkpoints: `checkpoints/E1_augfix/` (+ dynamics), `checkpoints/E3_*/`, `checkpoints/E3_dynamics_variants/` (demo-only and demo + simulator dynamics).
- Logs: `results/training_logs/E3/`.
- Partial diagnostic runs: `results/benchmark/diagnostics/E3_diag_*.json`.

## Tiếng Việt

**Ngày:** 19–20/11/2026. **Dữ liệu học:** zarr; mô hình động lực học dùng thêm dữ liệu tương tác sinh từ mô phỏng.

**Đã thay đổi:**
1. **`PushDynamics`:** phần World Model mà đề bài yêu cầu.
   - Dự đoán độ thay đổi tư thế khối (Δx, Δy, Δθ) từ trạng thái hiện tại và hành động.
   - MLP trong hệ toạ độ của khối, lăn 16 bước liền không cần quan sát.
   - Hàm mất mát là sai số dịch chuyển f(Δx, Δy, Δθ) của đề bài, cộng dồn qua 16 bước.
2. **`dit_wm`: World Model chọn hành động.**
   - Tạo 64 chuỗi hành động ứng viên quanh đề xuất của inverse dynamics.
   - Tưởng tượng từng chuỗi bằng `PushDynamics`.
   - Chọn chuỗi có quỹ đạo khối bám đường đi của DiT nhất.
3. **Học lại DiT** 100 epoch + EMA, thử 3 cách chọn snapshot inverse dynamics, quan sát 2 khung ảnh, và các tổ hợp cũ / mới.
4. **Baseline kiểu Diffusion Policy:** khuếch tán trực tiếp 16 hành động.

**Kết quả** (bảng đầy đủ: [results.md](results.md)):
- **World Model chọn hành động là bước cải thiện lớn nhất:** mô hình E1 + `dit_wm` đạt **84%**, cao nhất dự án (dữ liệu zarr); lần lặp nhiễu thứ hai đạt 78%.
- **Chỉ số offline đánh lừa.** RMSE của ID khoảng 25 px ở mọi snapshot, nhưng thành công vòng kín dao động 0–65%. Chọn theo RMSE cho **3%**; chọn bằng vòng kín cho 64%.
- **Thiết kế quan trọng hơn lượng dữ liệu:** động lực học chỉ học từ demo gần bằng bản có thêm dữ liệu mô phỏng.
- 2 khung ảnh không giúp.
- Baseline khuếch tán hành động có độ phủ cao (88%) nhưng ít khi về đích (32%).
