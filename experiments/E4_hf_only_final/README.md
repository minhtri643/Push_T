# E4 — Hugging Face data only: the FINAL system / Chỉ dữ liệu Hugging Face: HỆ THỐNG CUỐI

[English](#english) · [Tiếng Việt](#tiếng-việt) · Numbers / số liệu: [results.md](results.md)

## English

**Date:** 2026-11-20. **Training data:** only [lerobot/pusht](https://huggingface.co/datasets/lerobot/pusht). No zarr, no simulator-generated data, no pretrained weights. The simulator is used only to measure.

**Why.** One public, citable data source that matches the course brief.

**The data.** 206 human demonstrations, 25,650 frames (96×96, AV1 video), pusher position and actions. The pusher position and actions are bit-identical to the zarr. The dataset has **no block pose**. Its `next.reward` is clip(coverage(t+1) / 0.95), computed from the true pose.

**Block pose recovered from the images** (`scripts/1_data/fit_block_pose.py`, `pusht_wm_dit/data/block_pose_fit.py`):
- **Method:** render-and-compare. A T-block is rendered at a candidate pose and compared with the frame; the pose is refined until they match.
- **Accuracy on simulator frames with known pose:** 0.57 px / 0.18° (clean), 0.58 px / 0.17° after AV1 compression like the dataset.
- **Image-to-world offset** of (−0.65, −0.75) px, estimated from the dataset's own reward (pygame half-pixel convention).
  - Mean error between the reward and the coverage of the fitted pose: 0.0111 before the correction, 0.0059 after (frames with reward > 0.2).
  - After the correction, 97.0% of the frames have coverage of the fitted pose within 0.02 of the reward.
- **Stillness check:** when the block cannot move, the fitted pose varies by 0.04 px between frames.
- Output: `data/pusht_hf.zarr`. Reports: `results/pose_recovery/`.

**Retraining.** The whole pipeline was retrained on HF data with the scripts of E3. Two rounds:
1. **Round 0** (`checkpoints/E4_hf_round0`), before the offset correction.
2. **Final** (`checkpoints/final`).

**Results** (complete: [results.md](results.md)):
- **Final system `E4_final__dit_wm`: 76% success, 94.6% mean max coverage, DP score 0.988** on the reference run. Over three diffusion-noise draws (same models, same starts): 76%, 69%, 66%, mean **71 ± 5%**. On the 50 gym-pusht seeds alone: 78%, 70%, 66% (mean 71%).
- **The World Model choosing the actions is significant:** 71% versus 55% for DiT + inverse dynamics, averaged over the three noise draws; per-start sign test p = 0.0001. Per draw: 76 vs 59 (p = 0.014), 69 vs 48 (p = 0.003), 66 vs 57 (p = 0.22).
- **~1 px of label offset matters:** round 0 reached only 56%, p = 0.005 against the final.
- **HF-only costs nothing measurable** against the zarr models of E3: 76% versus 74% (`closedloop_id60`, p = 0.86) and 84% (E1 models + dynamics, p = 0.20).
- **Dynamics World Model on HF validation episodes**, 16 steps:

  | Model | Error | Counterfactual sensitivity |
  |---|---|---|
  | `PushDynamics` | 1.06 px / 0.63° | 96.5% |
  | old latent World Model | 23.7 px | 4.8% |
  | "block stays still" | 25.6 px | — |

  Source: `results/world_model_eval/E4_final/`.
- **Determinism:** exact re-runs on 2026-11-23 reproduce 95/95 starts, after the repository was reorganised.

**Limitations (say them in the report):**
- At run time `dit_wm` uses the **true block pose from the simulator** to pin step 0 of the DiT path and as the state of the dynamics model. This is privileged information; E5 removes it.
- The goal is fixed.
- One training seed; 95 starts give a ±8–9 point interval.

```
python scripts/5_evaluation/benchmark.py --policy dit_wm --label E4_final__dit_wm
python scripts/5_evaluation/benchmark.py --policy dit_id --label E4_final__dit_id
python scripts/5_evaluation/benchmark.py --policy dit_wm --ckpt-dir checkpoints/E4_hf_round0 --label E4_hf_round0__dit_wm
```

Training logs: `results/training_logs/E4_final/`, `results/training_logs/E4_hf_round0/`.

## Tiếng Việt

**Ngày:** 20/11/2026. **Dữ liệu học:** chỉ lerobot/pusht. Không zarr, không dữ liệu sinh từ mô phỏng, không trọng số tiền huấn luyện. Bộ mô phỏng chỉ dùng để đo.

**Dữ liệu.** 206 lần trình diễn của người, 25.650 khung. Bộ dữ liệu **không có tư thế khối T**.

**Khôi phục tư thế từ ảnh:**
- **Cách làm:** render-and-compare, tức vẽ khối T ở một tư thế thử rồi so với ảnh, chỉnh dần tới khi khớp.
- **Độ chính xác trên ảnh mô phỏng biết trước tư thế:** 0,58 px / 0,17° sau nén AV1.
- **Độ lệch ảnh → thế giới** (−0,65, −0,75) px, ước lượng từ chính reward của dữ liệu. Sau khi sửa, 97,0% số khung có độ phủ khớp reward trong phạm vi 0,02.

**Kết quả:**
- **Hệ thống cuối đạt 76% thành công, độ phủ TB 94,6%** ở lần chạy tham chiếu; qua ba lần lặp nhiễu khuếch tán: 76%, 69%, 66%, trung bình **71 ± 5%**.
- World Model chọn hành động nâng 55% → 71% (trung bình 3 lần lặp nhiễu; p = 0,0001).
- Lệch nhãn khoảng 1 px làm tụt còn 56% (vòng 0).
- Chỉ dùng HF không làm giảm kết quả có ý nghĩa so với zarr (p = 0,86 và 0,20).
- Chạy lại y hệt sau khi sắp xếp repo cho đúng 95/95 ván.

**Hạn chế:**
- Lúc chạy, `dit_wm` dùng **tư thế khối thật từ mô phỏng**, tức thông tin đặc quyền; E5 bỏ thông tin này.
- Đích cố định.
- Mới có 1 seed huấn luyện.
