# E0 — Original baseline / Bản gốc

[English](#english) · [Tiếng Việt](#tiếng-việt) · Numbers / số liệu: [results.md](results.md)

## English

**What it is.** The first version of the project (before 2026-11-16):
- a World Model encoder (ResNet-18 with GroupNorm, frame → latent `s_t`) with a latent transition head;
- a DiT that diffuses a 16-step path of the T-block;
- an inverse-dynamics controller that turns the path into robot actions.

All models were trained on the Diffusion Policy zarr (`pusht_cchi_v7_replay.zarr`).

**Problems found when it was reviewed (fixed in E1):**
1. **Placement bug.** `set_state` placed the block with a single `_set_state` call. pymunk rotates the body about its centre of gravity, 45 px from the T origin, so the block could be misplaced by up to 90 px.
2. **Wrong coverage.** Coverage was computed as IoU instead of gym-pusht's intersection / goal area.
3. **Augmentation bug.** The reflection augmentation flipped the image and the labels inconsistently, so half of the DiT training targets did not match their images.

**Results.**
- Original evaluation protocol (50 episodes): mean max coverage 36.7%, success 2%; 0/20 hard cases, 0/25 random placements. See `results/legacy_protocol/E0_baseline/`.
- Standard paired protocol (95 starts), re-measured on 2026-11-23 with the original checkpoints: see [results.md](results.md).

**Reproduce:**

```
python scripts/5_evaluation/benchmark.py --policy dit_id --ckpt-dir checkpoints/E0_baseline --label E0_baseline__dit_id
```

**Files:**
- `code_snapshot/`: the original code, for reference only; it is not run any more.
- `checkpoints/E0_baseline/`.
- `results/legacy_protocol/E0_baseline/` (`from_original_run/` = numbers produced by the original code itself).

## Tiếng Việt

**Đây là gì.** Phiên bản đầu tiên của dự án (trước 16/11/2026):
- bộ mã hoá World Model (ResNet-18 + GroupNorm, ảnh → latent `s_t`) kèm đầu chuyển trạng thái ẩn;
- DiT sinh đường đi 16 bước của khối T;
- inverse dynamics biến đường đi thành hành động.

Tất cả học trên zarr của Diffusion Policy.

**Lỗi phát hiện khi rà soát (sửa ở E1):**
1. **Lỗi đặt vị trí.** `set_state` đặt khối bằng một lần `_set_state`. pymunk xoay quanh trọng tâm, cách gốc chữ T 45 px, nên khối có thể lệch tới 90 px.
2. **Sai độ phủ.** Độ phủ tính bằng IoU thay vì giao / diện tích đích của gym-pusht.
3. **Lỗi augmentation.** Phép phản chiếu lật ảnh và nhãn không khớp nhau, nên một nửa nhãn DiT không khớp ảnh.

**Kết quả.**
- Giao thức cũ (50 ván): độ phủ TB 36,7%, thành công 2%; 0/20 ca khó, 0/25 vị trí ngẫu nhiên.
- Giao thức chuẩn 95 điểm xuất phát, đo lại ngày 23/11/2026 với checkpoint gốc: xem [results.md](results.md).
