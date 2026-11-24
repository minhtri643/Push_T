# PushT: JEPA world model + DiT diffusion policy

Học một **latent world model kiểu JEPA** từ chuỗi ảnh PushT, đóng băng nó, rồi huấn luyện một **Diffusion Transformer (DiT)** sinh quỹ đạo 2 giây của pusher trên không gian latent đó. Chạy hoàn toàn trên GTX 1650 (fp32).

| Thành công (seed 100001) | Thất bại (seed 100000) |
|---|---|
| ![success](media/demo_success.gif) | ![failure](media/demo_failure.gif) |

Hai tập là hai video đầu tiên của bộ đánh giá chính (50 seed), không chọn tập đẹp nhất. Video demo trên 20 seed mới nằm trong `media/live/`.

## Highlights

- **JEPA học được thông tin vật lý.** Probe tuyến tính đọc vị trí khối T từ S_t đạt R² = 0,904 (random-init 0,687, PCA pixel 0,821). Không collapse: std_s = 0,836, effective rank 138,4.
- **DiT thắng baseline offline.** FDE của khối T sau 2 s: 37,2 (DiT) so với 44,6 (ngoại suy vận tốc) và 51,5 (đứng yên).
- **Closed-loop trong sim: 26% success@0,95, 50% success@0,90** trên 50 seed; 25% trên 20 seed mới. Thấp hơn nhiều so với Diffusion Policy end-to-end (65,4%), xem phần Hạn chế.
- **Ablation A1 là bằng chứng mạnh nhất:** thay encoder JEPA bằng encoder random-init thì success về 0%.

## Kiến trúc

![architecture](report/figures/fig01_architecture.png)

Hai giai đoạn:

1. **JEPA world model** (2,17 M tham số). Hai khung ảnh liên tiếp cho ra vector trạng thái S_t (256 chiều). Predictor có điều kiện action dự đoán *biểu diễn ẩn* của 8 khung tương lai, không dự đoán pixel. Target do encoder EMA (stop-gradient) tạo ra, chống collapse bằng loss variance và covariance kiểu VICReg.
2. **DiT policy** (1,36 M tham số). S_t (đã freeze) cộng vị trí pusher làm điều kiện, DiT khử nhiễu quỹ đạo 20 điểm (2 giây). Suy luận bằng DDIM, quỹ đạo đổi sang vận tốc rồi thực thi 8 bước trước khi dự đoán lại.

## Dữ liệu và giao thức đánh giá

- Dataset `lerobot/pusht`: 206 tập, 25 650 khung, 10 fps. Chia theo tập: 186 train / 20 val (seed 42).
- Demo của người dừng ở coverage khoảng 0,90 (chỉ 7/206 tập vượt 0,95). Vì vậy báo cáo ba thước đo: **success@0,95** (chuẩn gym-pusht), **success@0,90** (mức demo người) và **mean max coverage** (thước đo của paper Diffusion Policy).
- Closed-loop: 50 seed bắt đầu từ 100000, tối đa 300 bước. Demo live dùng seed từ 1000 trở đi, tách khỏi bộ đánh giá.
- Mốc tham chiếu: replay demo trên 20 tập val đạt 0% success@0,95, 50% success@0,90, mean max coverage 0,898. Agent ngẫu nhiên: 0/10, coverage 0,070. `lerobot/diffusion_pusht` (DP end-to-end, khoảng 0,3 B tham số): 65,4% và 0,955 trên 500 tập.

## Kết quả

### World model (tập val)

| Chỉ số | JEPA | Random-init | PCA pixel |
|---|---|---|---|
| R² vị trí khối T | **0,904** | 0,687 | 0,821 |
| R² cosθ / sinθ (trung bình) | **0,595** | -0,046 | -0,124 |
| Sai số vị trí (px) | **20,8** | 40,4 | 28,1 |
| Sai số góc (độ) | **18,5** | 40,5 | 42,8 |

![pca](report/figures/fig05a_pca_scatter.png)

Loss dự đoán trong latent thấp hơn baseline "copy" từ k = 3 trở đi và khoảng cách tăng theo k, nhưng **thua copy ở k = 1, 2** (hai khung liền kề gần như trùng nhau). Xem `fig03a_pred_error_vs_k.png`.

### Policy offline: FDE khối T sau 2 s (375 mẫu val, đo bằng sim)

![fde](report/figures/fig07_fde.png)

| Cấu hình | FDE_pos (px) | FDE_θ (độ) | FDE_obj | minFDE_obj@5 |
|---|---|---|---|---|
| **Main (JEPA + DiT)** | **22,6** | **14,0** | **37,2** | 21,2 |
| A2: không dùng proprio | 24,2 | 14,8 | 39,8 | 22,8 |
| A1: encoder random-init | 34,2 | 21,5 | 56,7 | 30,3 |
| Ngoại suy vận tốc | 27,1 | 16,7 | 44,6 | - |
| Đứng yên | 31,7 | 18,9 | 51,5 | - |

`FDE_obj = FDE_pos + 60 · |Δθ|` (rad), góc đổi thành cung tròn theo nửa chiều dài thanh ngang của chữ T.

### Closed-loop trong sim (50 seed, 300 bước)

![ablation](report/figures/fig08_ablation.png)

| Cấu hình | Success@0,95 | Success@0,90 | Mean max cov. | Median time-to-success | Latency/lần suy luận |
|---|---|---|---|---|---|
| **Main** (k = 8, DDIM 16) | 26% | 50% | 0,776 | 16,3 s | 117 ms |
| A1: encoder random-init | 0% | 0% | 0,269 | - | 101 ms |
| A2: chỉ ảnh (không proprio) | 32% | 52% | 0,657 | 19,3 s | 111 ms |
| n_action_steps = 4 | 20% | 36% | 0,652 | 22,0 s | 112 ms |
| n_action_steps = 6 | 32% | 52% | 0,744 | 18,2 s | 108 ms |
| n_action_steps = 12 | 18% | 54% | 0,780 | 23,3 s | 108 ms |
| DDIM 4 bước | 16% | 32% | 0,655 | 16,9 s | 35 ms |
| DDIM 8 bước | 22% | 58% | 0,799 | 26,1 s | 63 ms |

Với n = 50, khoảng tin cậy 95% của 26% khoảng 16% đến 40%. Các cấu hình k = 6, 8, 12 và DDIM 8, 16 **không khác nhau có ý nghĩa thống kê**. Cấu hình chính được giữ là k = 8 (mặc định của Diffusion Policy), không chọn theo nhiễu. Các điểm rút ra được:

- Encoder JEPA là thành phần cốt lõi (A1: 0%).
- Bỏ proprio (A2) không làm giảm success (32% so với 26%, khác biệt không có ý nghĩa với 50 seed) nhưng coverage trung bình giảm (0,657 so với 0,776), nên chưa kết luận được proprio có ích hay không.
- DDIM 8 bước cho kết quả tương đương 16 bước với nửa độ trễ, nên demo live mặc định dùng 8 bước.
- Thời gian hoàn thành chỉ tính trên các tập thành công và bị cắt ở 30 s, nên không dùng để so sánh tốc độ giữa các cấu hình.

### Demo live trên 20 seed mới (1000 đến 1019, DDIM 8)

**5/20 thành công (25%), mean max coverage 0,724.** Khớp với bộ 50 seed nên kết quả không bị chọn lọc. Chi tiết 20 seed: 5 thành công; 4 gần xong (coverage 0,92 đến 0,95); 3 ở khoảng 0,88 đến 0,89; 8 thất bại hẳn (coverage dưới 0,75, thấp nhất 0,12). Nhật ký từng seed: `outputs/demo/live_log.csv`.

![sequence](report/figures/fig09_success_sequence.png)

*Chuỗi 6 khung của một tập thành công: ở khung (b) khối T bị đẩy ra xa đích (coverage 0,000), sau đó policy đưa nó về và đạt 0,953 ở 16,1 s. Dấu thập đỏ: mục tiêu action hiện tại; đường đỏ: quỹ đạo DiT dự đoán; đoạn hồng: phần đã thực thi.*

## Hạn chế (nói thẳng)

- **Thấp hơn nhiều so với Diffusion Policy end-to-end** (26% so với 65,4%). Nguyên nhân chưa được kiểm chứng. Các giả thuyết, chưa chứng minh: (1) demo dừng ở coverage khoảng 0,90 nên đoạn căn chỉnh cuối ít được học; (2) encoder đóng băng, nhỏ (2,17 M) và chỉ thấy 206 tập; (3) S_t rất nhạy với dịch ảnh (cosine giữa ảnh gốc và ảnh dịch chỉ 0,415), nên augmentation latent có thể làm nhiễu điều kiện.
- **Ở mức đơn vị vật lý, WM không thắng rõ rệt baseline "không động lực"** (probe trên S_t): tại k = 8 là 26,1 px / 22,7° so với 26,6 px / 20,9°, và cả hai bị chặn bởi sàn của probe tuyến tính (khoảng 22 px, 20°). Bằng chứng WM học động lực nằm ở loss trong latent (thắng copy từ k = 3), không nằm ở mức vật lý qua probe tuyến tính.
- WM train 36/40 epoch (dừng tay khi lr đã giảm còn khoảng 3% đỉnh). Một lần train, một seed, chưa có báo cáo mean ± std qua nhiều seed.
- Chưa làm A5 (BC-MSE) và A6 (horizon), chưa thử V-JEPA 2.1 pretrained.
- Chưa có thử nghiệm trên robot thật.

## Cài đặt

```bash
conda create -n pusht python=3.10 -y
conda activate pusht
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -c "import torch, gym_pusht, pymunk; print(torch.cuda.is_available(), pymunk.version)"
```

`pymunk` phải là bản 6.x (`pip install "pymunk<7"` nếu cài nhầm 7.x).

## Tái tạo kết quả

wm_main được huấn luyện ở fp32. Huấn luyện lại sẽ cho kết quả gần nhưng không trùng từng chữ số vì GPU không tất định.

Checkpoint (`*.pt`) và latents (`*.npy`) không được commit, nên cần chạy lần lượt các bước sau (khoảng 2 giờ trên GTX 1650):

```bash
# Bước 1: dữ liệu
python -m scripts.download_data
python -m scripts.inspect_data
python -m scripts.build_cache --config configs/default.yaml

# Bước 2: kiểm tra simulator
python -m scripts.sim_smoke_test --config configs/default.yaml

# Bước 3: JEPA world model (khoảng 1,5 giờ), đánh giá, xuất latents
python -m scripts.train_world_model --config configs/default.yaml --run_name wm_main
python -m scripts.eval_world_model --run_name wm_main
python -m scripts.export_latents --run_name wm_main

# Bước 4: DiT policy (khoảng 30 phút)
python -m scripts.train_policy --config configs/default.yaml --run_name dit_main
python -m scripts.viz_policy --run_name dit_main

# Bước 5: đánh giá
python -m scripts.eval_offline_fde --policy_run dit_main --latents data/cache/latents_wm_main.npy
python -m src.eval_sim --config configs/default.yaml --wm_run wm_main --policy_run dit_main
python -m scripts.make_figures
```

## Demo live (cho người xem)

```bash
python -m scripts.demo_live --n_episodes 3          # thử nhanh
python -m scripts.demo_live --continuous            # chạy mãi trên seed mới, bấm q hoặc Esc để thoát
python -m scripts.demo_live --no_window --n_episodes 20 --save_dir media/live   # lưu video từng seed
```

Cửa sổ hiện sim cùng quỹ đạo dự đoán. Terminal in kết quả mỗi seed và tỷ lệ thành công cộng dồn, thất bại cũng được in. Mọi seed được ghi nối vào `outputs/demo/live_log.csv`.

## Cấu trúc repo

```
configs/default.yaml     # mọi siêu tham số
src/
  utils.py               # config, seed, chuẩn hóa tọa độ, logger
  dataset.py             # cache, split theo tập, WMClipDataset, LatentPathDataset
  sim_env.py             # wrapper gym-pusht, đặt trạng thái trực tiếp lên pymunk
  world_model.py         # FrameEncoder, ContextAggregator, LatentDynamics, JEPA
  dit_policy.py          # DiT adaLN-Zero, DDPM/DDIM, đổi quỹ đạo sang vận tốc
  eval_sim.py            # đánh giá closed-loop
scripts/                 # train, eval, export, make_figures, demo_live
report/figures/          # hình dùng trong báo cáo
media/                   # GIF và video demo
outputs/runs/<run>/      # log.csv, config, hình của từng lần chạy (checkpoint bị .gitignore)
```

## Tài liệu tham khảo và nguồn

- Dataset: [`lerobot/pusht`](https://huggingface.co/datasets/lerobot/pusht)
- Diffusion Policy: [project page](https://diffusion-policy.cs.columbia.edu), [source](https://github.com/real-stanford/diffusion_policy)
- V-JEPA 2: [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)
- Simulator: gym-pusht; thư viện LeRobot cho mốc tham chiếu 65,4%.
- Chi et al. 2023 (Diffusion Policy); Peebles & Xie 2023 (DiT); Ho et al. 2020 (DDPM); Song et al. 2021 (DDIM); Bardes et al. 2022 (VICReg); Assran et al. 2023 (I-JEPA); Bardes et al. 2024 (V-JEPA); Assran et al. 2025 (V-JEPA 2).

## Tác giả

Nhóm thực hiện đồ án (môn Lập trình ứng dụng trong AI).
