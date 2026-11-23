# Benchmark vòng kín: mọi lần chạy

95 điểm xuất phát ghép cặp (50 seed gym-pusht, 20 vị trí khó, 25 vị trí ngẫu nhiên), 300 bước, thành công = độ phủ > 0,95. Lặp nhiễu: cùng mô hình, cùng điểm xuất phát, nhiễu khuếch tán khác (TB ± độ lệch qua n lần). McNemar: kiểm định chính xác trên thành công, ghép theo điểm xuất phát.


## E0 — Bản gốc

| Cấu hình | Dữ liệu học | Thành công [KTC 95%] | Thành công qua các lần lặp nhiễu | Độ phủ TB | Điểm DP | seeds50 / hard20 / random25 | so với E4_final__dit_wm (chỉ cấu hình này / chỉ bản cuối, p) |
|---|---|---|---|---|---|---|---|
| `E0_baseline__dit_id`<br>mã và mô hình ban đầu (học trước 16/11), DiT + inverse dynamics | zarr | 2% [1%–7%] | - | 34.6% | 0.364 | 0% / 5% / 4% | 0 / 70, p=0.000 |

## E1 — Sửa giao thức đánh giá + augmentation

| Cấu hình | Dữ liệu học | Thành công [KTC 95%] | Thành công qua các lần lặp nhiễu | Độ phủ TB | Điểm DP | seeds50 / hard20 / random25 | so với E4_final__dit_wm (chỉ cấu hình này / chỉ bản cuối, p) |
|---|---|---|---|---|---|---|---|
| `E1_augfix__dit_id`<br>học lại với phép phản chiếu chéo đã sửa | zarr | 41% [32%–51%] | - | 77.3% | 0.809 | 46% / 45% / 28% | 7 / 40, p=0.000 |

## E2 — Bộ điều khiển viết tay (đã lưu trữ)

| Cấu hình | Dữ liệu học | Thành công [KTC 95%] | Thành công qua các lần lặp nhiễu | Độ phủ TB | Điểm DP | seeds50 / hard20 / random25 | so với E4_final__dit_wm (chỉ cấu hình này / chỉ bản cuối, p) |
|---|---|---|---|---|---|---|---|
| `E2_guard`<br>mô hình E1 + chặn xuyên thấu | zarr | 27% [19%–37%] | - | 67.0% | 0.702 | 30% / 20% / 28% | 10 / 56, p=0.000 |
| `E2_lookahead`<br>mô hình E1 + dừng tối ưu bằng World Model ẩn cũ | zarr | 19% [12%–28%] | - | 76.0% | 0.798 | 26% / 15% / 8% | 2 / 56, p=0.000 |
| `E2_guard_lookahead`<br>chặn xuyên thấu + nhìn trước (mặc định của demo cũ) | zarr | 13% [7%–21%] | - | 66.4% | 0.697 | 16% / 5% / 12% | 4 / 64, p=0.000 |
| `E2_tactical_default`<br>bộ điều khiển 3 nhịp, chặn + nhìn trước, khoá ở 88% / 80% | zarr | 0% [0%–4%] | - | 86.8% | 0.914 | 0% / 0% / 0% | 0 / 72, p=0.000 |
| `E2_tactical_no_lock`<br>tactical, chặn + nhìn trước, không khoá | zarr | 58% [48%–67%] | - | 91.6% | 0.957 | 64% / 45% / 56% | 12 / 29, p=0.012 |
| `E2_tactical_best`<br>tactical, chặn, không nhìn trước, không khoá | zarr | 53% [43%–62%] | - | 91.7% | 0.958 | 60% / 45% / 44% | 12 / 34, p=0.002 |
| `E2_tactical_no_guard`<br>tactical, không chặn, không nhìn trước, không khoá | zarr | 38% [29%–48%] | - | 74.6% | 0.781 | 44% / 30% / 32% | 9 / 45, p=0.000 |

## E3 — World Model động lực học theo hành động (dit_wm)

| Cấu hình | Dữ liệu học | Thành công [KTC 95%] | Thành công qua các lần lặp nhiễu | Độ phủ TB | Điểm DP | seeds50 / hard20 / random25 | so với E4_final__dit_wm (chỉ cấu hình này / chỉ bản cuối, p) |
|---|---|---|---|---|---|---|---|
| `E3_augfix_ckpt__dit_wm`<br>mô hình E1 + World Model động lực học mới chọn hành động (64 ứng viên) | zarr (+ dữ liệu mô phỏng cho động lực học) | 84% [76%–90%] | 81.1 ± 4.5 (n=2) | 94.5% | 0.985 | 82% / 80% / 92% | 19 / 11, p=0.200 |
| `E3_dit100ep_rmse_id__dit_id`<br>DiT học lại 100 epoch + EMA; ID chọn theo RMSE offline | zarr | 3% [1%–9%] | - | 50.7% | 0.533 | 2% / 5% / 4% | 2 / 71, p=0.000 |
| `E3_dit100ep_rmse_id__dit_wm`<br>như trên, World Model chọn hành động | zarr (+ dữ liệu mô phỏng cho động lực học) | 40% [31%–50%] | - | 83.1% | 0.870 | 42% / 40% / 36% | 15 / 49, p=0.000 |
| `E3_dit100ep_closedloop_id25__dit_id`<br>ID chọn bằng vòng kín (<= 25 epoch) | zarr | 51% [41%–60%] | - | 78.1% | 0.816 | 52% / 55% / 44% | 13 / 37, p=0.001 |
| `E3_dit100ep_closedloop_id25__dit_wm`<br>như trên, World Model chọn hành động | zarr (+ dữ liệu mô phỏng cho động lực học) | 72% [62%–80%] | - | 90.1% | 0.942 | 76% / 60% / 72% | 19 / 23, p=0.644 |
| `E3_dit100ep_closedloop_id60__dit_id`<br>ID chọn bằng vòng kín (<= 60 epoch) | zarr | 64% [54%–73%] | - | 86.0% | 0.899 | 76% / 55% / 48% | 14 / 25, p=0.108 |
| `E3_dit100ep_closedloop_id60__dit_wm`<br>như trên, World Model chọn hành động | zarr (+ dữ liệu mô phỏng cho động lực học) | 74% [64%–81%] | 71.6 ± 3.0 (n=2) | 92.4% | 0.965 | 80% / 60% / 72% | 16 / 18, p=0.864 |
| `E3_dit100ep_history2__dit_id`<br>2 khung ảnh làm quan sát | zarr | 56% [46%–65%] | - | 84.5% | 0.884 | 64% / 30% / 60% | 9 / 28, p=0.003 |
| `E3_dit100ep_history2__dit_wm`<br>như trên, World Model chọn hành động | zarr (+ dữ liệu mô phỏng cho động lực học) | 78% [69%–85%] | - | 92.0% | 0.962 | 72% / 85% / 84% | 18 / 16, p=0.864 |
| `E3_dit100ep_with_augfix_id__dit_id`<br>DiT 100 epoch + ID của E1 | zarr | 58% [48%–67%] | - | 81.6% | 0.853 | 66% / 60% / 40% | 13 / 30, p=0.014 |
| `E3_dit100ep_with_augfix_id__dit_wm`<br>như trên, World Model chọn hành động | zarr (+ dữ liệu mô phỏng cho động lực học) | 75% [65%–82%] | - | 92.0% | 0.962 | 78% / 60% / 80% | 17 / 18, p=1.000 |
| `E3_action_diffusion_baseline`<br>baseline kiểu Diffusion Policy: khuếch tán trực tiếp 16 hành động từ latent ảnh | zarr | 32% [23%–41%] | - | 88.3% | 0.926 | 34% / 25% / 32% | 6 / 48, p=0.000 |

## E4 — Chỉ dữ liệu Hugging Face (BẢN CUỐI)

| Cấu hình | Dữ liệu học | Thành công [KTC 95%] | Thành công qua các lần lặp nhiễu | Độ phủ TB | Điểm DP | seeds50 / hard20 / random25 | so với E4_final__dit_wm (chỉ cấu hình này / chỉ bản cuối, p) |
|---|---|---|---|---|---|---|---|
| `E4_hf_round0__dit_id`<br>học lại chỉ trên HF, tư thế khớp trước khi sửa độ lệch ảnh | chỉ lerobot/pusht | 58% [48%–67%] | - | 88.2% | 0.923 | 58% / 80% / 40% | 12 / 29, p=0.012 |
| `E4_hf_round0__dit_wm`<br>như trên, World Model chọn hành động | chỉ lerobot/pusht | 56% [46%–65%] | - | 90.1% | 0.942 | 54% / 70% / 48% | 12 / 31, p=0.005 |
| `E4_final__dit_id`<br>mô hình CUỐI, DiT + inverse dynamics | chỉ lerobot/pusht | 59% [49%–68%] | 54.7 ± 5.6 (n=3) | 87.4% | 0.914 | 62% / 60% / 52% | 11 / 27, p=0.014 |
| **`E4_final__dit_wm`**<br>HỆ THỐNG CUỐI: World Model động lực học chọn hành động (mặc định của demo) | chỉ lerobot/pusht | 76% [66%–83%] | 70.5 ± 4.8 (n=3) | 94.6% | 0.988 | 78% / 80% / 68% | — |

## E5 — Nhánh mở rộng chỉ dùng ảnh

| Cấu hình | Dữ liệu học | Thành công [KTC 95%] | Thành công qua các lần lặp nhiễu | Độ phủ TB | Điểm DP | seeds50 / hard20 / random25 | so với E4_final__dit_wm (chỉ cấu hình này / chỉ bản cuối, p) |
|---|---|---|---|---|---|---|---|
| `E5a_dinov2_zarr_sim__dit_id_vision`<br>chỉ ảnh: tư thế đọc bằng đầu dò trên DINOv2 đóng băng (planner E1) | zarr + ảnh mô phỏng | 35% [26%–45%] | - | 76.8% | 0.804 | 30% / 30% / 48% | 10 / 49, p=0.000 |
| `E5a_dinov2_zarr_sim__dit_vwm`<br>như trên, World Model ảnh chọn hành động | zarr + ảnh mô phỏng | 8% [4%–16%] | - | 84.8% | 0.892 | 12% / 0% / 8% | 4 / 68, p=0.000 |
| `E5b_jepa__dit_id_vision`<br>chỉ ảnh: bộ mã hoá JEPA tự học + đầu dò tư thế | chỉ lerobot/pusht | 45% [36%–55%] | 44.6 ± 1.2 (n=3) | 73.0% | 0.764 | 48% / 50% / 36% | 9 / 38, p=0.000 |
| `E5b_jepa__dit_vwm`<br>như trên, World Model ảnh chọn hành động | chỉ lerobot/pusht | 20% [13%–29%] | - | 76.7% | 0.805 | 26% / 20% / 8% | 5 / 58, p=0.000 |
| `E5c_dinov2_frozen__dit_id_vision`<br>chỉ ảnh: DINOv2 ViT-S đóng băng + đầu dò tư thế | chỉ lerobot/pusht (+ DINOv2 tiền huấn luyện, đóng băng) | 49% [40%–59%] | 52.6 ± 3.2 (n=3) | 83.9% | 0.877 | 56% / 30% / 52% | 11 / 36, p=0.000 |
| `E5c_dinov2_frozen__dit_vwm`<br>như trên, World Model ảnh chọn hành động | chỉ lerobot/pusht (+ DINOv2 tiền huấn luyện, đóng băng) | 13% [7%–21%] | - | 72.0% | 0.756 | 10% / 30% / 4% | 2 / 62, p=0.000 |
| `E5d_vjepa2_frozen__dit_id_vision`<br>chỉ ảnh: V-JEPA 2 ViT-L đóng băng + đầu dò tư thế | chỉ lerobot/pusht (+ V-JEPA 2 tiền huấn luyện, đóng băng) | 54% [44%–63%] | 53.7 ± 2.1 (n=3) | 85.4% | 0.894 | 56% / 55% / 48% | 15 / 36, p=0.005 |
| `E5d_vjepa2_frozen__dit_vwm`<br>như trên, World Model ảnh chọn hành động | chỉ lerobot/pusht (+ V-JEPA 2 tiền huấn luyện, đóng băng) | 11% [6%–18%] | - | 73.9% | 0.776 | 12% / 5% / 12% | 2 / 64, p=0.000 |

## Kiểm tra tính tất định (chạy lại y hệt)

- `E4_final__dit_id` rerun 2026-11-23: 95/95 điểm xuất phát trùng khớp
- `E4_final__dit_wm` rerun 2026-11-23-parallel: 95/95 điểm xuất phát trùng khớp
- `E4_final__dit_wm` rerun 2026-11-23: 95/95 điểm xuất phát trùng khớp
