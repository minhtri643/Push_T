# So sánh bộ mã hoá cho World Model ảnh (2026-11-22)

Dữ liệu huấn luyện: chỉ lerobot/pusht. Bộ mô phỏng chỉ dùng để đo. Bộ mã hoá V-JEPA 2 / DINOv2 đóng băng; chỉ bộ dự đoán và đầu dò được huấn luyện (scripts/13). JEPA tự học: scripts/12.

## Đọc tư thế (đầu dò) — vị trí TB px / góc TB °
| Bộ mã hoá | HF val | sim sạch | sim AV1 | 95 điểm xuất phát | seeds50 | hard20 | random25 |
| JEPA tự học (0.41M, HF) | 11.1 / 14.0 | 40.6 / 71.7 | 40.6 / 77.2 | 33.3 / 63.7 | 36.0 / 63.3 | 25.4 / 54.9 | 34.5 / 71.8 |
| DINOv2 ViT-S đóng băng (21M) | 5.4 / 2.9 | 18.4 / 15.3 | 18.2 / 16.4 | 12.5 / 10.8 | 10.0 / 8.5 | 18.8 / 20.2 | 12.7 / 7.8 |
| V-JEPA 2 ViT-L đóng băng (304M) | 5.3 / 2.7 | 24.0 / 24.4 | 20.8 / 20.4 | 17.2 / 20.6 | 13.9 / 15.7 | 22.6 / 28.6 | 19.3 / 24.2 |

## Tưởng tượng trên 1,915 cửa sổ kiểm tra (20 tập HF giữ lại)
| Mô hình | vị trí px k=4/8/12/16 | sai số Δ vị trí px k=4/8/12/16 | góc k=16 | FDE k=16 | xáo trộn hành động k=16 | độ nhạy phản thực tế |
| WM JEPA tự học (0.41M, HF) | 11.6 / 11.6 / 11.4 / 11.0 | 6.3 / 9.2 / 10.7 / 11.7 | 11.3° | 0.206 | 31.4 px | 77% |
|   sàn đầu dò (JEPA tự học (0.41M, HF)) | 10.2 / 9.3 / 8.3 / 7.4 | 6.5 / 9.0 / 10.0 / 10.4 | 7.4° | 0.135 | | |
| WM DINOv2 ViT-S đóng băng (21M) | 6.7 / 8.2 / 9.5 / 10.5 | 5.2 / 7.4 / 9.1 / 10.3 | 7.3° | 0.138 | 25.8 px | 66% |
|   sàn đầu dò (DINOv2 ViT-S đóng băng (21M)) | 5.1 / 4.8 / 4.6 / 4.3 | 5.5 / 6.2 / 6.3 / 6.5 | 2.1° | 0.043 | | |
| WM V-JEPA 2 ViT-L đóng băng (304M) | 7.2 / 9.0 / 10.5 / 11.8 | 5.5 / 8.0 / 9.9 / 11.5 | 8.7° | 0.166 | 25.3 px | 63% |
|   sàn đầu dò (V-JEPA 2 ViT-L đóng băng (304M)) | 5.0 / 4.8 / 4.6 / 4.4 | 5.3 / 5.8 / 6.0 / 6.3 | 1.8° | 0.039 | | |
| Đoán đứng yên | 6.1 / 12.3 / 18.8 / 25.6 | 6.1 / 12.3 / 18.8 / 25.6 | 15.0° | 0.301 | | |
| Động lực học trạng thái (cận trên) | 0.6 / 0.8 / 0.9 / 1.1 | 0.6 / 0.8 / 0.9 / 1.1 | 0.7° | 0.013 | | |

## Vòng kín 95 điểm xuất phát (300 bước, thành công > 95%)
| Nhãn | thành công [CI 95%] | độ phủ TB | DP-score | seeds50 | hard20 | random25 |
| jepa_dit_id_vision | 45% [36–55] | 73.0% | 0.764 | 48% | 50% | 36% |
| jepa_dit_vwm | 20% [13–29] | 76.7% | 0.805 | 26% | 20% | 8% |
| dinov2hf_dit_id_vision | 49% [40–59] | 83.9% | 0.877 | 56% | 30% | 52% |
| dinov2hf_dit_vwm | 13% [7–21] | 72.0% | 0.756 | 10% | 30% | 4% |
| vjepa2_dit_id_vision | 54% [44–63] | 85.4% | 0.894 | 56% | 55% | 48% |
| vjepa2_dit_vwm | 11% [6–18] | 73.9% | 0.776 | 12% | 5% | 12% |
| vision_dit_id | 35% [26–45] | 76.8% | 0.804 | 30% | 30% | 48% |
| vision_dit_vwm | 8% [4–16] | 84.8% | 0.892 | 12% | 0% | 8% |
| hf_dit_id | 59% [49–68] | 87.4% | 0.914 | 62% | 60% | 52% |
| hf_dit_wm | 76% [66–83] | 94.6% | 0.988 | 78% | 80% | 68% |

## McNemar (thành công, cặp cùng điểm xuất phát)
vjepa2_dit_id_vision vs jepa_dit_id_vision: 54% vs 45% | chỉ A 26, chỉ B 18 | p=0.291
dinov2hf_dit_id_vision vs jepa_dit_id_vision: 49% vs 45% | chỉ A 24, chỉ B 20 | p=0.652
vjepa2_dit_id_vision vs dinov2hf_dit_id_vision: 54% vs 49% | chỉ A 20, chỉ B 16 | p=0.618
vjepa2_dit_vwm vs jepa_dit_vwm: 11% vs 20% | chỉ A 7, chỉ B 16 | p=0.093
dinov2hf_dit_vwm vs jepa_dit_vwm: 13% vs 20% | chỉ A 11, chỉ B 18 | p=0.265
vjepa2_dit_vwm vs vjepa2_dit_id_vision: 11% vs 54% | chỉ A 6, chỉ B 47 | p=0.000
dinov2hf_dit_vwm vs dinov2hf_dit_id_vision: 13% vs 49% | chỉ A 5, chỉ B 40 | p=0.000
vjepa2_dit_id_vision vs hf_dit_id: 54% vs 59% | chỉ A 16, chỉ B 21 | p=0.511
vjepa2_dit_vwm vs hf_dit_wm: 11% vs 76% | chỉ A 2, chỉ B 64 | p=0.000
