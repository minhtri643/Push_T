# E1 — Evaluation protocol + augmentation fix / Sửa giao thức đánh giá + augmentation

[English](#english) · [Tiếng Việt](#tiếng-việt) · Numbers / số liệu: [results.md](results.md)

## English

**Date:** 2026-11-17. **Training data:** Diffusion Policy zarr.

**What changed:**
1. **One evaluation module** (`pusht_wm_dit/policy.py`) used by every script. The standard protocol is 300 steps and success = coverage > 0.95; heuristics are opt-in flags.
2. **Exact placement:** `PushTSim.set_state` calls `_set_state` twice, so the block lands exactly where asked.
3. **Coverage** is taken from gym-pusht (intersection / goal area), not IoU.
4. **Augmentation fixed.** The reflection is the anti-diagonal one through the goal centre: (x, y) → (−y, −x), θ → π/2 − θ, image transposed and flipped on both axes.
   - Verified against renders: mean pixel difference 0.48, versus 7.07 for the old version.
   - The latent cache carries a version number, so stale caches are rebuilt.
5. World Model, DiT and inverse dynamics were retrained with the fixed augmentation.

**Results.**
- Same 50-episode protocol as E0: coverage 36.7% → **80.6%**, success 2% → **34%**; hard cases 0/20 → 9/20; random placements 0/25 → 10/25.
- **Hand-written heuristics hurt** the retrained model (coverage / success):

  | Heuristic | Coverage / success |
  |---|---|
  | none | 80.6 / 34 |
  | stance | 55.9 / 24 |
  | lock | 67.9 / 0 |
  | both | 47.3 / 0 |

- Standard 95-start benchmark: `E1_augfix__dit_id`, see [results.md](results.md).

**Lesson.** Before improving a model, make the measurement right: about half of the apparent weakness of E0 came from the evaluation and from inconsistent training labels.

**Reproduce:**

```
python scripts/5_evaluation/benchmark.py --policy dit_id --ckpt-dir checkpoints/E1_augfix --label E1_augfix__dit_id
python experiments/E1_protocol_and_augmentation_fix/code/evaluate_loop_old_protocol.py --ckpt-dir checkpoints/E1_augfix   # old 50-episode protocol
```

**Files:**
- `checkpoints/E1_augfix/`.
- Training logs: `results/training_logs/E1_augfix/`.
- Old-protocol results: `results/legacy_protocol/E1_augfix/`.

## Tiếng Việt

**Ngày:** 17/11/2026. **Dữ liệu học:** zarr của Diffusion Policy.

**Đã thay đổi:**
1. **Một module đánh giá duy nhất** (`policy.py`). Giao thức chuẩn: 300 bước, thành công khi độ phủ > 0,95; heuristic chỉ bật khi được yêu cầu.
2. **Đặt vị trí chính xác:** gọi `_set_state` hai lần.
3. **Độ phủ** lấy từ gym-pusht (giao / diện tích đích), không dùng IoU.
4. **Sửa augmentation:** phản chiếu qua đường chéo phụ đi qua tâm đích.
   - Đã kiểm chứng với ảnh render: sai khác điểm ảnh 0,48, so với 7,07 của bản cũ.
   - Cache latent có số phiên bản để tự tạo lại khi cũ.
5. Học lại World Model, DiT và inverse dynamics với augmentation đã sửa.

**Kết quả.**
- Giao thức 50 ván: độ phủ 36,7% → **80,6%**, thành công 2% → **34%**; ca khó 0/20 → 9/20; vị trí ngẫu nhiên 0/25 → 10/25.
- **Heuristic viết tay làm giảm kết quả** (bảng trên).
- Benchmark chuẩn 95 điểm xuất phát: xem [results.md](results.md).

**Bài học.** Phải đo đúng trước khi cải tiến mô hình: khoảng một nửa điểm yếu của E0 đến từ cách đánh giá và nhãn học không khớp.
