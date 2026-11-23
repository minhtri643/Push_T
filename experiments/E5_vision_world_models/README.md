# E5 — Vision-only extension / Nhánh mở rộng chỉ dùng ảnh

[English](#english) · [Tiếng Việt](#tiếng-việt) · Numbers / số liệu: [results.md](results.md), [`results/vision_world_models/comparison.md`](../../results/vision_world_models/comparison.md)

## English

**Goal.** Remove the privileged block pose of E4: the block pose comes from the image, and a **vision World Model** predicts the next image features from the current ones and the actions (DINO-WM / V-JEPA 2-AC style). This is a research extension; it does **not** beat the final system E4.

**Two policies:**
- `dit_id_vision`: the block pose is read from the image by a pose probe on the encoder's features; then DiT + inverse dynamics.
- `dit_vwm`: same, and the actions are chosen by imagining the candidates with the vision World Model.

| Sub-stage | Encoder | Trained on | Code |
|---|---|---|---|
| E5a (2026-11-20, legacy) | frozen DINOv2 ViT-S | zarr + simulator frames | `code_legacy_dinov2/` |
| E5b (2026-11-20) | **self-trained** JEPA ConvEncoder (0.41M params) | lerobot/pusht only | `scripts/6_vision_world_model/train_jepa_wm.py` |
| E5c (2026-11-22) | frozen DINOv2 ViT-S (21M) | lerobot/pusht only | `scripts/6_vision_world_model/train_frozen_encoder_wm.py --encoder dinov2` |
| E5d (2026-11-22) | frozen **V-JEPA 2** ViT-L (304M) | lerobot/pusht only | `… --encoder vjepa2` |

In E5c and E5d the pretrained encoder is frozen ("standing on the shoulders of giants"). Only the action-conditioned predictor (about 3.5M parameters) and the pose probe are trained.

**Vision World Models, 1,915 held-out windows (HF validation episodes):**

| | JEPA (E5b) | DINOv2 (E5c) | V-JEPA 2 (E5d) |
|---|---|---|---|
| Pose read-out on HF frames (px / °) | 11.1 / 14.0 | 5.4 / 2.9 | **5.3 / 2.7** |
| Pose read-out at the 95 benchmark starts, simulator frames (px) | 33.3 | **12.5** | 17.2 |
| Error on the change of position at 16 steps (px) | 11.7 | **10.3** | 11.5 |
| FDE at 16 steps | 0.206 | **0.138** | 0.166 |
| Counterfactual sensitivity | **77%** | 66% | 63% |

For reference: "block stays still" 25.6 px (FDE 0.301); state dynamics of E4 1.1 px (FDE 0.013).

**Closed loop** (95 starts; `dit_id_vision` repeated with three diffusion-noise draws; details in [results.md](results.md)):

| Encoder | `dit_id_vision`: 3 draws (mean) | `dit_vwm` |
|---|---|---|
| E5a DINOv2 legacy | 35% | 8% |
| E5b JEPA | 45 / 43 / 45% (44.6%) | 20% |
| E5c DINOv2 frozen | 49 / 53 / 56% (52.6%) | 13% |
| E5d V-JEPA 2 frozen | 54 / 56 / 52% (53.7%) | 11% |

- **V-JEPA 2 and DINOv2 are equal** in closed loop (53.7% vs 52.6%, per-start sign test over the three draws p = 0.70). Both are ahead of JEPA, but not significantly (p = 0.13 and 0.38).
- **Reading the pose from the image costs nothing for DiT + inverse dynamics.** V-JEPA 2 `dit_id_vision` 53.7% vs E4 `dit_id` with the true pose 54.7% (p = 0.69). The remaining gap to the final system comes from the World Model choosing the actions, which needs a precise state.

**Lessons:**
1. **Pretrained features help mainly to read the pose.** The angle error falls from 14° to about 3°, which raises `dit_id_vision` from 45% to 54%.
2. **Bigger is not better here.** V-JEPA 2 has 15× the parameters of DINOv2 but does not beat it, and runs twice as slowly (415 s vs 210 s for 95 episodes). It is used sub-optimally: one frame repeated as a 2-frame clip, 96 px upscaled to 256.
3. **Choosing actions with the vision World Model hurts for every encoder** (11–20% vs 45–54%, p < 0.001). An error of about 10 px is too coarse to rank candidates that differ by 5–20 px, and the selection exploits the model's errors. The order of `dit_vwm` follows the counterfactual sensitivity (77 > 66 > 63%), but three points are only a hypothesis.
4. **Domain gap.** The probes are 2–3× less accurate on simulator frames than on dataset frames, and the closed loop runs in the simulator.

## Tiếng Việt

**Mục tiêu.** Bỏ thông tin đặc quyền của E4: tư thế khối đọc từ ảnh, và **World Model ảnh** dự đoán đặc trưng ảnh tiếp theo từ đặc trưng hiện tại cùng hành động. Đây là nhánh nghiên cứu mở rộng, **chưa vượt** hệ thống cuối E4.

**Bài học:**
1. **Đặc trưng tiền huấn luyện chủ yếu giúp đọc tư thế.** Sai số góc giảm từ 14° xuống khoảng 3°, nhờ đó `dit_id_vision` tăng từ 45% lên 54%.
2. **Lớn hơn không tốt hơn ở đây.** V-JEPA 2 có số tham số gấp 15 lần DINOv2 nhưng không vượt nó, lại chạy chậm gấp đôi.
3. **Dùng World Model ảnh để chọn hành động làm kết quả tệ đi ở mọi bộ mã hoá** (p < 0,001). Sai số khoảng 10 px quá thô để xếp hạng các ứng viên chỉ khác nhau 5–20 px.
4. **Khác biệt miền ảnh:** đầu dò kém hơn 2–3 lần trên ảnh mô phỏng so với ảnh của bộ dữ liệu.
