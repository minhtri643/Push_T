# Push-T: World Model + Diffusion Transformer — tài liệu bối cảnh đầy đủ (dành cho AI)

> **Ảnh chụp trạng thái dự án ngày 2026-11-23, sau khi sắp xếp lại repo để đưa lên GitHub.** Tài liệu này dành cho một AI **không truy cập được mã nguồn** nhưng cần hiểu dự án như người đã đọc toàn bộ code, cấu hình, log và kết quả.
> - **Nhãn mới:** mọi lần chạy mang nhãn `E<giai đoạn>_<mô hình>__<policy>`. Giai đoạn: E0 bản gốc, E1 sửa giao thức + augmentation, E2 bộ điều khiển viết tay, E3 World Model động lực học (zarr), E4 chỉ dữ liệu HF (**bản cuối**), E5 nhánh chỉ dùng ảnh.
> - Bảng tên cũ → mới: `docs/name_mapping.md`. Lịch sử từng giai đoạn: `experiments/E*/README.md`. Mọi bảng số liệu sinh tự động: `results/summary/`.
> - Mọi con số đều đo bằng cách chạy chính code của dự án. Chỗ nào chưa đo sẽ ghi rõ "chưa đo".
> - Số thập phân dùng dấu chấm. Đường dẫn tính từ thư mục gốc dự án (`Final_Project_PushT/`).

---

## 0. Lưu ý cho AI đọc tài liệu

- **Chủ dự án** là học viên thạc sĩ người Việt; đây là đồ án cuối kỳ môn World Model.
  - Luôn trả lời bằng **tiếng Việt**.
  - Người dùng muốn **sự thật đã đo được**, không tô vẽ, không phóng đại.
- **Ràng buộc cứng** (người dùng quyết định ngày 2026-11-20):
  - Mọi mô hình **chỉ học từ dataset Hugging Face [`lerobot/pusht`](https://huggingface.co/datasets/lerobot/pusht)**.
  - Không dùng zarr của Diffusion Policy, không dùng dữ liệu sinh từ mô phỏng, không dùng trọng số học sẵn (như DINOv2) để **học**.
  - Mô phỏng chỉ được dùng để **đánh giá, đo lường và chọn checkpoint**, và phải nói rõ điều đó.
- **Luôn tách hai loại kết quả:**
  - (a) policy dùng **tư thế khối T thật lấy từ mô phỏng** khi chạy (thông tin đặc quyền, "dạng trạng thái");
  - (b) policy **chỉ dùng ảnh** (cộng vị trí đầu đẩy, tức proprioception).
- **Không tuyên bố "vượt SOTA".** Cách nói đúng nằm ở mục 10.
- **Bạn không chạy được code.** Nếu viết code giúp người dùng, hãy bám đúng API ở mục 15.
- **Máy chỉ có GPU laptop 4 GB.** Đừng đề xuất chạy hai job GPU cùng lúc: bộ nhớ bị tràn sang RAM, chậm 5–10 lần.

---

## 1. Tóm tắt 1 phút

- **Bài toán:** Push-T. Một đầu đẩy tròn đẩy khối chữ T vào ô đích cố định (256, 256, 45°). Mô phỏng `gym_pusht/PushT-v0`.
- **Dữ liệu:** 206 tập do người điều khiển (25,650 khung, 10 Hz) từ `lerobot/pusht`.
  - Dataset **không có tư thế khối T**, nên tư thế được khôi phục từ từng ảnh bằng **render-and-compare**: vẽ khung hình bằng một bộ vẽ khả vi rồi khớp với ảnh thật.
  - Sai số khoảng **0.6 px** (thế giới 512 px).
- **Hệ thống:**
  1. Bộ mã hoá ảnh ResNet-18 (GroupNorm) tạo `s_t`.
  2. **DiT** sinh đường đi 16 bước của khối T.
  3. **Inverse dynamics** đề xuất 8 hành động.
  4. **World Model động lực học** (MLP trong hệ toạ độ gắn với khối T) tưởng tượng 64 chuỗi hành động ứng viên, rồi chọn chuỗi có quỹ đạo khối T dự đoán bám đường đi của DiT nhất. Chế độ này tên là `dit_wm`.
- **World Model động lực học:** sau **16 bước tưởng tượng** sai **1.1 px / 0.63°**. Mốc "đoán khối đứng yên" sai 25.6 px. **Độ nhạy với hành động 96%** (lý tưởng 100%).
- **Vòng kín** (95 điểm xuất phát, 300 bước, thành công khi độ phủ > 95%):
  - `dit_wm`: độ phủ TB **94.6%**, thành công **76%** [KTC 95%: 66–83] ở lần chạy tham chiếu; **71 ± 5%** qua 3 lần lặp nhiễu. Dùng tư thế thật.
  - Bỏ bước World Model chọn hành động (`dit_id`): 59% (55 ± 6% qua 3 lần). Gộp 3 lần: p = 0.0001; riêng lần đầu McNemar p = 0.014.
- **Chỉ dùng ảnh:** JEPA tự học **45%**; DINOv2 đóng băng 49%; V-JEPA 2 đóng băng **54%** (khác biệt giữa ba bộ chưa có ý nghĩa thống kê). Dùng World Model ảnh để chọn hành động làm hại (11–20%).
- **So với bài báo:**
  - `dit_wm` **ngang** Diffusion Policy dạng trạng thái (LeRobot keypoints 71%, 500 ván); không khác biệt có ý nghĩa thống kê.
  - Bản chỉ dùng ảnh **kém rõ** Diffusion Policy dạng ảnh (LeRobot 63.8%).

---

## 2. Đề bài và cách hệ thống đáp ứng

Đề bài của Thầy:
> *"Xây dựng World Model để học quy luật tương tác giữa robot và vật thể. Từ trạng thái hiện tại và hành động của robot, mô hình dự đoán sự thay đổi vị trí vật thể."*

- **Dataset Thầy chỉ định:** `lerobot/pusht`.
- **Quy trình Thầy vẽ:** Load data → Setup simulation → Train WM → Train Diffusion → Evaluate, rồi quay lại Train WM.
- **Ký hiệu trên bảng:** `FDE = f(Δx, Δy, Δθ)`; `frame → WM → s_t → DiT → path → action`.

| Yêu cầu | Hiện thực | File chính |
|---|---|---|
| Load data | Tải `lerobot/pusht`, giải mã video AV1, khôi phục tư thế khối T từ ảnh | `data/lerobot_pusht.py`, `data/block_pose_fit.py`, `scripts/1_data/fit_block_pose.py` |
| Setup simulation | Wrapper `PushTSim` quanh `gym_pusht/PushT-v0`, có hàm đặt trạng thái chính xác | `envs/pusht_sim.py` |
| Train WM (phần 1) | Bộ mã hoá ResNet-18 GN: ảnh + vị trí đầu đẩy → `s_t` ∈ R^128 | `models/world_model.py`, `scripts/3_world_model/train_encoder.py` |
| Train WM (phần 2) — **đúng câu đề bài** | `PushDynamics`: (trạng thái, hành động) → **Δ(x, y, θ) của khối T** và Δ vị trí đầu đẩy | `models/dynamics.py`, `scripts/3_world_model/train_dynamics.py` |
| Train WM (ảnh, hướng nghiên cứu) | JEPA: bộ mã hoá CNN tự học + bộ dự đoán đặc trưng ảnh tương lai theo hành động | `models/jepa_encoder.py`, `models/latent_world_model.py`, `scripts/6_vision_world_model/train_jepa_wm.py` |
| Train Diffusion | DiT sinh đường đi khối T; inverse dynamics ra hành động | `models/dit_planner.py`, `models/inverse_dynamics.py`, `scripts/4_diffusion/train_dit_planner.py` |
| Evaluate | Đánh giá WM offline và phép thử phản thực tế; benchmark vòng kín | `scripts/3_world_model/evaluate_world_model.py`, `scripts/5_evaluation/benchmark.py` |
| FDE | `sqrt(w_xy·(Δx²+Δy²) + w_θ·Δθ²)` tại bước cuối, toạ độ chuẩn hoá và radian; Δθ tính bằng atan2 từ sin/cos | `utils/metrics.py` |

**FDE được dùng ở ba chỗ:**
- loss DiT (MSE + λ·FDE²);
- loss mô hình động lực học (dạng px/độ, cộng dồn cả 16 bước);
- hàm chi phí khi `dit_wm` chọn hành động.

---

## 3. Môi trường máy và cách làm việc

- **Hệ thống:** Windows 11, PowerShell. Repo được chuẩn bị để đưa lên GitHub (công khai). Tại thời điểm viết, máy **chưa cài git**.
  - Bản sao lưu toàn bộ trước khi sắp xếp lại: `D:\Thac_Si\AI_Programming\World_Model\Final_Project_PushT_snapshot_2026-11-23` (ngoài repo).
  - Thư mục `backups/` cũ đã được chuyển hết vào `checkpoints/`, `experiments/`, `results/`.
- **Python:** 3.11.9 trong `.venv`.
  - Venv không có pip; cài gói bằng `uv pip install --python .venv\Scripts\python.exe <gói>`.
  - Phiên bản ghim đầy đủ trong `requirements.txt` và `pyproject.toml`.
- **Phiên bản:** torch 2.6.0+cu124, gymnasium 1.4.0, gym-pusht 0.1.8, pymunk 6.11.1, numpy 2.4.6, zarr 3.1.6, shapely 2.2.0, transformers 4.57.6 (chỉ cho V-JEPA 2), huggingface-hub 0.36.2.
- **Đường dẫn:** mọi đường dẫn tương đối tính từ gốc repo. Script gọi `pusht_wm_dit.paths.chdir_root()` nên chạy được từ thư mục nào cũng được.
- **GPU:** NVIDIA RTX 3050 Laptop 4 GB.
- **Video:** ffmpeg đi kèm `imageio-ffmpeg`, dùng bộ giải mã libaom-av1.
- **Tải từ HF:** đôi khi báo `WinError 10054`; chạy lại là được (snapshot_download tải tiếp).

---

## 4. Môi trường Push-T: các sự thật kỹ thuật

- **Không gian và đích:**
  - Không gian làm việc 512×512 px thế giới. Ảnh quan sát 96×96 RGB (vẽ ở 512 rồi thu nhỏ).
  - **Đích cố định** (256, 256, π/4) trong toàn bộ dữ liệu và đánh giá.
- **Hình khối T** (hệ toạ độ thân khối; giống `add_tee` của gym-pusht, scale 30):
  - Thanh ngang 120×30 với y ∈ [0, 30]; thân dọc 30×90 với y ∈ [30, 120].
  - Diện tích 6300 px². Trọng tâm hình học ở (0, 40.7).
- **Lỗi đặt trạng thái (đã sửa):**
  - `_set_state` của gym-pusht gán vị trí **trước** góc. Pymunk xoay quanh tâm khối lượng, cách gốc khối T khoảng 45 px.
  - Vì vậy gọi một lần có thể đặt lệch khối tới **90 px**.
  - `PushTSim.set_state(exact=True)` gọi hai lần để đặt đúng (x, y, θ).
- **Hành động:** là **điểm đích của đầu đẩy** (px thế giới). Bộ điều khiển PD của gym-pusht kéo đầu đẩy tới đó. Tần số điều khiển 10 Hz.
- **Độ phủ** = diện tích(khối ∩ đích) / diện tích(đích). Đây **không phải IoU**.
  - Reward của gym = clip(độ phủ/0.95, 0, 1).
  - **Thành công** khi độ phủ > 0.95; ván dừng ngay khi đạt.
- **Chuẩn hoá:**
  - Vị trí: (p − 256)/256, nằm trong [−1, 1].
  - Góc: radian.
  - Tư thế 4 chiều: `pose4 = [x_n, y_n, sin θ, cos θ]`.
- **Hậu xử lý hành động** (`policy.postprocess_action`, áp dụng cho **mọi** policy):
  - Điểm đích cách đầu đẩy hiện tại tối đa **40 px/bước**; cắt về [0, 512].
  - Hai heuristic cũ ("stance", "collision guard") **tắt** theo mặc định.
- **Số bước:** đánh giá tối đa **300 bước** (`env.max_steps`); giao diện demo cho chạy tới 1000 bước (`demo.max_steps`).

---

## 5. Dữ liệu

### 5.1 `lerobot/pusht`
- **Định dạng:** LeRobot v3.0, 7.7 MB. Gồm:
  - `data/chunk-000/file-000.parquet`
  - `meta/episodes/…parquet`
  - `meta/info.json`
  - **một** video AV1 chung cho mọi tập: `videos/observation.image/chunk-000/file-000.mp4`
- **Quy mô:** 206 tập, 25,650 khung, 10 Hz.
- **Các cột chính:**
  - `observation.state`: (x, y) của đầu đẩy, px thế giới.
  - `action`: (x, y) điểm đích mà người điều khiển chỉ chuột tới.
  - `next.reward`.
  - `next.success`: **luôn = 0** ở cả 206 tập, vì bản ghi dừng ngay trước khi vượt ngưỡng 95%. Không dùng cột này.
  - Các cột chỉ số: `episode_index`, `frame_index`, `index`.
- **Không có** tư thế hay keypoint của khối T.
- **Quan hệ với zarr của Diffusion Policy** (`pusht_cchi_v7_replay.zarr`):
  - Cùng 206 tập: `episode_ends` trùng nhau, vị trí đầu đẩy và hành động **giống hệt từng bit**.
  - Ảnh HF bị nén AV1 có mất mát: lệch trung bình 0.69/255 so với ảnh zarr.
  - Zarr có sẵn tư thế khối T; nó từng được dùng trước ngày 2026-11-20 và **nay bị cấm dùng để học**.
- **Tính đồng bộ ảnh – trạng thái:** đầu đẩy nhìn thấy trong khung t cách `observation.state(t)` 1.7 px, còn cách trạng thái t±1 khoảng 14 px. Vậy ảnh và trạng thái khớp nhau.
- **Ý nghĩa `next.reward(t)`:** bằng `clip(độ phủ(tư thế(t+1)) / 0.95, 0, 1)`, tính từ tư thế **thật**. Đây là giả thuyết khớp nhất trong 4 giả thuyết đã thử (mục 5.3).

### 5.2 Khôi phục tư thế khối T bằng render-and-compare
Code ở `data/block_pose_fit.py` và `scripts/1_data/fit_block_pose.py hf`.

1. **Bộ vẽ khả vi** tái tạo cách gym-pusht vẽ (theo `pymunk_override`):
   - nền trắng, T đích màu LightGreen, tường LightGray;
   - đầu đẩy RoyalBlue bán kính 15, lõi sáng bán kính 11;
   - khối T vẽ bằng hai đa giác (thanh ngang trước, thân sau), mỗi đa giác tô màu LightSlateGray sáng hơn (×1.2) và có viền 2 px màu LightSlateGray;
   - pixel (hàng i, cột j) lấy mẫu điểm thế giới ((j+0.5)·512/96, (i+0.5)·512/96);
   - cạnh được làm mềm bằng sigmoid có độ rộng `s`.
2. **Ẩn số:** vị trí đầu đẩy đã biết, chỉ cần tìm (x, y, θ).
3. **Khởi tạo:** tâm của các pixel có màu khối T, với 8 góc khởi đầu cách nhau 45°.
4. **Tối ưu:**
   - Adam, lr 2.0 px cho vị trí và 0.04 rad cho góc, giảm dần kiểu cosine; 120 vòng.
   - Độ mờ đi từ s0 = 10 xuống `s_final` trong 60% số vòng đầu (từ mờ đến nét).
   - Loss quang học robust kiểu Charbonnier (ε 0.02) trên **vùng cắt 44×44 px** quanh khối, bỏ 2 px viền ảnh.
   - Chọn điểm khởi đầu có loss thấp nhất. Có đo "biên độ mơ hồ": 0 khung có biên < 0.002.
5. **Hiệu chỉnh màu và độ mờ:** 7 màu và độ mờ được học **cùng lúc với tư thế** trên 256 khung ngẫu nhiên (2 vòng × 300 bước), vì giải mã AV1 làm lệch màu. Kết quả trên HF:

   | Thông số | Giá trị |
   |---|---|
   | Độ mờ `s` | 0.41 px thế giới |
   | Nền | 255, 255, 255 |
   | Đích | 143, 236, 144 |
   | Tường | 212, 203, 214 |
   | Đầu đẩy / lõi | 82, 109, 184 / 82, 123, 248 |
   | Viền khối / ruột khối | 117, 135, 149 / 143, 162, 181 |

6. **Độ lệch toạ độ ảnh → thế giới:**
   - Tìm bằng lưới thô đến mịn (bước 0.2 / 0.1 / 0.05 / 0.025 px) một phép tịnh tiến cố định.
   - Mục tiêu: cực tiểu |clip(độ phủ(tư thế(t+1) + lệch)/0.95) − `next.reward(t)`| trên các khung có reward > 0.2.
   - Kết quả **(−0.65, −0.75) px**; MAE trên các khung đó giảm từ 0.0111 xuống 0.0059.
   - Nguyên nhân: pygame vẽ toạ độ p vào ô pixel [p, p+1) (lệch nửa pixel), cộng thêm độ lệch màu của codec.
   - Kiểm tra độc lập trên ảnh mô phỏng biết trước tư thế cũng phát hiện một độ lệch hệ thống cỡ (0.25, 0.40) px (sai số sau khi bỏ độ lệch: 0.37 px). Độ lệch nhỏ hơn có lẽ vì bộ nén libaom khác libsvtav1 của LeRobot.
7. **Thời gian:** 936 s (khoảng 16 phút) cho 25,650 khung trên RTX 3050.

### 5.3 Kiểm chứng tư thế khôi phục

| Kiểm định | Kết quả |
|---|---|
| 2,000 ảnh mô phỏng biết trước tư thế, nén AV1 (libaom crf 30) | Vị trí TB 0.58 px (trung vị 0.59, p99 0.96, lớn nhất 1.69); góc 0.17° (lớn nhất 0.57°); 0 khung sai > 5 px |
| Cùng tập ảnh trên, không nén | 0.57 px / 0.18° |
| Phép thử reward trên 25,650 khung HF, **sau** khi sửa độ lệch | MAE 0.0036, trung vị 0.0012, p95 0.015; **97.0%** số khung lệch < 0.02 |
| Phép thử reward, **trước** khi sửa độ lệch | MAE 0.0067; 92.3% số khung lệch < 0.02 |
| Sai số tư thế suy ra từ phép thử reward | khoảng 0.6 px (khoảng 0.1 pixel ảnh) |
| Rung giữa các khung khi đầu đẩy ở xa > 30 px (9,834 cặp) | 0.04 px / 0.02° (phép thử yếu) |
| So với tư thế chính xác trong zarr (chỉ đọc để đo, không dùng để học) | 0.55 px / 0.24° |

- **Ảnh mô phỏng biết trước tư thế:** 20 tập có nhiều va chạm, tổng 2,000 khung, sinh bởi bộ điều khiển chuyên gia viết tay. Khung có tiếp xúc sai 0.71 px.
- **Chọn giả thuyết reward:** cách tính `coverage(t+1)/0.95` khớp nhất. Các cách khác cho MAE lớn hơn: `coverage(t)` 0.022, `coverage(t)/0.95` 0.011, `coverage(t+1)` 0.015.
- **So với zarr:** người dùng **chưa quyết định** có trích con số 0.55 px vào báo cáo hay không, vì phép này đụng tới file zarr.

### 5.4 Dữ liệu huấn luyện sau khôi phục, cách chia tập, augmentation
- **File `data/pusht_hf.zarr`:**
  - `data/img` (25650, 96, 96, 3) uint8;
  - `data/state` (25650, 5) = [agent_x, agent_y, block_x, block_y, θ mod 2π];
  - `data/action` (25650, 2);
  - `data/reward`, `data/tpose_fit_loss`;
  - `meta/episode_ends` (206 giá trị, chỉ số kết thúc không bao gồm).
  - Cùng cấu trúc với zarr cũ, nên các script huấn luyện dùng lại được mà không phải sửa.
- **Chia theo tập:** `RandomState(42).permutation(206)`. 20 tập đầu là **val**, 186 tập còn lại là **train**: 23,415 bước train, 2,235 bước val. Mọi script dùng cùng cách chia này.
- **Augmentation** (chỉ áp cho train): **phản xạ qua đường chéo phụ đi qua tâm đích**.
  - Toạ độ chuẩn hoá: (x, y) → (−y, −x). Góc: θ → π/2 − θ. Ảnh: chuyển vị rồi lật cả hai trục.
  - Phép này biến T đích thành chính nó, và vật lý đẩy đối xứng gương.
  - Đã kiểm chứng trên ảnh vẽ: chênh 0.48, so với 7.07 của augmentation sai cũ.
  - Sau augmentation, train có **46,830 mẫu**.

---

## 6. Kiến trúc chi tiết

```
quan sát: ảnh 96×96 + vị trí đầu đẩy  (+ tư thế khối T thật nếu chạy ở chế độ đặc quyền)
        │
        ▼
[WM phần 1] ResNet-18 GroupNorm + MLP hợp nhất ──► s_t ∈ R^128
        │
        ▼
[DiT] đường đi 16 bước của khối T (DDIM 10 bước, ghim bước 0 = tư thế hiện tại, chọn 1 trong 8)
        │
        ▼
[Inverse dynamics] (s_t, đường đi, vị trí đầu đẩy) ──► 8 hành động đề xuất
        │
        ▼  (chỉ ở chế độ dit_wm)
[WM phần 2] PushDynamics: tưởng tượng 64 chuỗi hành động quanh đề xuất,
            giữ chuỗi có quỹ đạo khối T dự đoán khớp đường đi DiT nhất
        │
        ▼
thực thi 8 bước (mỗi bước tối đa 40 px) ──► quan sát lại ──► lặp
```

### 6.1 World Model phần 1: bộ mã hoá (`models/world_model.py`, class `WorldModel`)
- **Kiến trúc:**
  - Ảnh [0, 1] đi qua **ResNet-18 tự định nghĩa**, BatchNorm thay bằng **GroupNorm(16)**, kích hoạt GELU, global average pooling → 512 chiều.
  - Vị trí đầu đẩy đi qua MLP 2→64→64.
  - Hợp nhất: 576→256 (LayerNorm, GELU)→128 (LayerNorm) cho ra **`s_t` 128 chiều**.
- **Hai đầu giải mã để neo `s_t` vào vật lý:**
  - tư thế 128→128→4 `[x, y, sin, cos]`;
  - vị trí đầu đẩy 128→64→2.
- **"WM cũ":** có thêm một MLP chuyển trạng thái ẩn `s_{t+1} = s_t + MLP(s_t, a_t)`.
  - Đo được là **gần như mù hành động** (độ nhạy 5%), nên **không dùng để lập kế hoạch**. Vẫn giữ trong code để đối chứng.
- **Loss:** MSE(tư thế) + MSE(đầu đẩy) + 0.5·MSE(chuyển trạng thái, đích dừng gradient).
- **Huấn luyện:** AdamW (lr 1e-3, weight decay 1e-4), cosine, batch 64, AMP, **5 epoch**. Checkpoint tốt nhất ở epoch 5 (val loss 0.00067). Mất khoảng 11 phút, kể cả bước lưu cache.
- **Kích thước:** 11.52 triệu tham số, trong đó ResNet 11.18 triệu.
- **Sau huấn luyện:**
  - Đóng băng.
  - Tính sẵn `s_t` cho toàn bộ train vào `pusht_wm_dit/data/precomputed_latents_hf_aug.pt` (46,830×128). Cache có metadata kiểm tra phiên bản và `zarr_path`.
- **Đọc tư thế từ ảnh qua đầu giải mã:** sai khoảng **3.5 px / 2.2°** trên khung val.
  - Chưa đo trên ảnh khởi đầu của benchmark.
  - Chưa dùng làm nguồn tư thế cho vòng kín chỉ dùng ảnh (xem mục 12).

### 6.2 World Model phần 2: động lực học (`models/dynamics.py`, class `PushDynamics`)
Đây là phần trực tiếp đáp ứng câu đề bài: từ trạng thái và hành động, dự đoán thay đổi vị trí vật.

- **Trạng thái:**
  - khối (x, y, θ) chuẩn hoá;
  - đầu đẩy (x, y);
  - vận tốc đầu đẩy = độ dời trong một bước (`agent_t − agent_{t−1}`).
- **Hành động:** điểm đích của đầu đẩy (chuẩn hoá).
- **12 đặc trưng đầu vào**, phần lớn đặt **trong hệ toạ độ gắn với khối T**:
  - bx, by, sin θ, cos θ (vẫn giữ vị trí tuyệt đối để học được ảnh hưởng của tường);
  - vị trí đầu đẩy so với khối (×4);
  - hành động so với khối (×4);
  - hành động so với đầu đẩy (×4);
  - vận tốc đầu đẩy (×10).
- **Mạng:** MLP 12→256 (SiLU) → [256, LayerNorm, SiLU]×2 → 5. Tổng **137k tham số**.
  - **Lớp cuối khởi tạo bằng 0**, nên mô hình chưa học tương đương mốc "đứng yên".
- **Đầu ra:**
  - Δx, Δy của khối trong hệ khối (×0.05 ≈ 12.8 px);
  - Δθ (×0.1 rad);
  - Δ đầu đẩy trong hệ khối (×0.2).
  - Các giá trị được quay về hệ thế giới rồi cộng dồn. `rollout` tưởng tượng K bước vòng hở.
- **Loss:**
  - Tưởng tượng **16 bước vòng hở** từ trạng thái thật.
  - Mỗi bước: Huber(δ = 2) của (sai vị trí px + sai góc độ + 0.3·sai đầu đẩy px).
  - Cửa sổ có khối di chuyển (> 2 px hoặc > 1° sau 16 bước) nhân trọng số 3; phần vượt quá cuối tập bị che (mask).
- **Huấn luyện:**
  - AdamW (lr 2e-3, weight decay 1e-4), OneCycle (pct 0.1), batch 1024, 60 epoch, cắt gradient 1.
  - Có augmentation phản xạ; chỉ dữ liệu demo HF.
  - Chọn theo điểm val (TB sai px + TB sai độ): tốt nhất ở epoch 58. Mất khoảng 4 phút.
- **Bài học từ giai đoạn zarr:** bản chỉ học từ demo cho kết quả gần như bằng bản học từ demo + mô phỏng. Thiết kế (hệ toạ độ khối, dự đoán độ thay đổi, khởi tạo 0, loss nhiều bước) quan trọng hơn lượng dữ liệu.

### 6.3 DiT lập kế hoạch đường đi (`models/dit_planner.py`, class `DiTTrajectoryPlanner`)
- **Kiến trúc:** transformer 1D theo kiểu DiT (Peebles & Xie).
  - 16 token, mỗi token là một `pose4` → Linear(4→256), cộng embedding vị trí học được.
  - 4 lớp, dim 256, 4 head, MLP ×4.
  - Điều kiện qua **AdaLN-Zero**: `c = MLP([s_t, embedding_thời_gian])`.
  - Mạng dự đoán **x0 trực tiếp** (không dự đoán ε). Tổng **5.17 triệu tham số**.
- **Lịch nhiễu:** cosine (`squaredcos_cap_v2`), 100 bước khuếch tán.
- **Đường đi mục tiêu:** tư thế khối ở t, t+1, …, t+15; phần tử thứ 0 là tư thế hiện tại.
- **Loss:** MSE(x̂0, đường đi) + λ_fde·FDE² với λ = 1; FDE tính ở bước cuối.
- **Huấn luyện:**
  - 100 epoch, batch 128, AdamW (lr 2e-4, weight decay 1e-4), OneCycle (pct 0.05), cắt gradient 1.
  - **EMA 0.999** (có khởi động); bản EMA được dùng khi chạy.
  - Cứ 10 epoch, **chọn checkpoint theo FDE của đường đi lấy mẫu** trên 20 tập val. Tốt nhất ở epoch 60 (val FDE 0.134, ADE 0.080, đơn vị chuẩn hoá). Mất khoảng 16 phút.
- **Lấy mẫu:**
  - DDIM 10 bước, η = 0.
  - **Inpainting:** ở mỗi bước, phần tử 0 được ghim bằng tư thế hiện tại (đã cộng nhiễu tương ứng) và x̂0[0] bị ghi đè.
  - Cắt về [−1, 1].
  - **Chọn 1 trong 8 mẫu** bằng hàm điểm **viết tay** hướng về đích: `3·|xy cuối − đích|² + 2·(1 − cos Δθ_đích) + 0.05·Σ|Δxy|²` (càng thấp càng tốt).
- **Điều kiện:** chỉ dùng `s_t` của một khung. Đã thử điều kiện hai khung (`v4_h2`), không cải thiện có ý nghĩa.

### 6.4 Inverse dynamics (`models/inverse_dynamics.py`, class `InverseDynamicsController`)
- **Đầu vào:**
  - `s_t` (128);
  - đường đi 16×4 làm phẳng → MLP 64→256→256;
  - vị trí đầu đẩy;
  - (vị trí đầu đẩy − xy của bước 0 trên đường đi).
- **Mạng:** in_proj → 2 khối ResMLP (256) → head → **8×2 hành động** (tanh, chuẩn hoá). Tổng 519k tham số.
- **Dữ liệu học:** đường đi **thật** cộng nhiễu N(0, 0.02), MSE so với hành động của demo.
  - Đã thử học trên đường đi do DiT sinh ra: sai số offline như nhau nhưng thành công vòng kín tụt từ 46% xuống 10%.
- **Huấn luyện:** AdamW (lr 3e-4, weight decay 1e-4), cosine, batch 128, 60 epoch. Lưu snapshot mỗi 5 epoch.
- **Chọn snapshot bằng mô phỏng vòng kín** ở chế độ `dit_id` trên **20 seed val 90000–90019**; các seed này không trùng benchmark.
  - Snapshot được chọn: epoch 60 (thành công 65%, độ phủ 85.5%).
  - RMSE offline trên đường đi DiT **khoảng 26 px ở mọi snapshot**, trong khi thành công vòng kín dao động 0–65%. Sai số offline **không dự báo được** chất lượng vòng kín.

### 6.5 `dit_wm`: World Model chọn hành động (`policy.py`, `WMDiTPolicy._choose_with_world_model`)
1. Đề xuất = 8 hành động của inverse dynamics.
2. Sinh **64 ứng viên**:
   - ứng viên 0 = đề xuất;
   - các ứng viên khác = đề xuất + nhiễu Gauss mượt theo thời gian `g_t = 0.7·g_{t−1} + 0.71·ε`, nhân hệ số {0, 0.02, 0.04, 0.08} (mỗi mức 16 ứng viên; 0.02 ≈ 5 px), rồi cắt về [−1, 1].
   - Lưu ý: 16 ứng viên ở mức 0 trùng với đề xuất, nên thực chất chỉ có **49 ứng viên khác nhau**.
3. `PushDynamics` tưởng tượng 8 bước cho từng ứng viên, **xuất phát từ trạng thái thật của mô phỏng** (khối, đầu đẩy, vận tốc).
4. Chi phí = trung bình qua 8 bước của (sai vị trí px)² + (sai góc độ)² so với `path[1..8]` của DiT. Chọn ứng viên có chi phí nhỏ nhất.
5. Thực thi 8 bước rồi lập lại kế hoạch.

### 6.6 World Model ảnh kiểu JEPA (`models/jepa_encoder.py`, `models/latent_world_model.py`, `scripts/6_vision_world_model/train_jepa_wm.py`)
Mô hình này **không bao giờ thấy tư thế khối T** khi học.

- **`ConvEncoder`** (0.41 triệu tham số):
  - Đầu vào: RGB − 0.5, cộng 2 kênh toạ độ, tổng 5 kênh.
  - Các khối Conv + GroupNorm(8) + GELU: 4×4 stride 2 (→48), 3×3, 4×4 stride 2 (→24), 3×3, 4×4 stride 2 (→12), 3×3, rồi 1×1 về 64 kênh.
  - Đầu ra **lưới 12×12 = 144 token × 64**. Mỗi token được layer-norm không tham số.
  - Bản xử lý ở độ phân giải đầy đủ 96×96 chậm khoảng 5 lần trên GPU 4 GB, nên đã giảm độ phân giải ngay ở lớp đầu.
- **`LatentPredictor`** (3.34 triệu tham số):
  - Đầu vào: token của z_{t−S} và z_t chiếu lên 256 chiều, cộng một token điều kiện từ MLP([đầu đẩy_{t−S}, đầu đẩy_t, S hành động, S hành động trừ đầu đẩy_t]).
  - Transformer pre-norm: depth 4, 4 head.
  - Đầu ra theo kiểu cộng dư: `z_{t+S} = z_t + head(h)`; thêm một đầu dự đoán vị trí đầu đẩy. Các đầu ra khởi tạo bằng 0.
  - **S = 4 bước (0.4 s)**; khi học cuộn **K = 4 lần** (tương đương 16 bước điều khiển).
- **Loss:**
  - MSE giữa đặc trưng dự đoán và token của **bộ mã hoá EMA** (dừng gradient), qua 4 bước cuộn;
  - + 100·MSE(vị trí đầu đẩy);
  - + 10·MSE(**đầu inverse dynamics**: từ z_{t−S}, z_t đoán S hành động ở giữa);
  - + 1.0·VICReg phương sai (ngưỡng độ lệch chuẩn γ = 0.5, tính theo từng kênh của từng token trên batch);
  - + 0.04·VICReg hiệp phương sai.
  - Công thức theo hướng PLDM (Sobal và cộng sự, 2025).
- **Huấn luyện:**
  - EMA momentum tăng kiểu cosine 0.99 → 0.999. AdamW (lr 3e-4, weight decay 1e-4), batch 64, **20 epoch** (156 s/epoch, khoảng 53 phút), có phản xạ.
  - Giữ epoch cuối, vì loss dự đoán trên val không so sánh được giữa các epoch (đích dịch chuyển theo EMA).
  - Cửa sổ train 20,439, val 1,915.
- **Không sụp đổ biểu diễn:** độ lệch chuẩn token khoảng 0.61 (train) / 0.53 (val).
- **`PoseProbe`** (chỉ để đo, 2.63 triệu tham số):
  - Linear(64→32) + GELU cho từng token, làm phẳng 144·32, rồi 512 → 512 → 4.
  - Học 20 epoch trên đặc trưng đã đóng băng, nhãn là tư thế khôi phục.
- **Cách dùng ở vòng kín chỉ dùng ảnh:** đầu dò đọc tư thế từ ảnh thay cho tư thế thật.

### 6.7 World Model ảnh cũ dùng DINOv2 (đã bỏ khỏi huấn luyện; số liệu giữ để đối chứng)
- Đặc trưng patch DINOv2 đóng băng, lưới 8×8×384 (kiểu DINO-WM), cùng `LatentPredictor`.
- Học trên zarr cộng khung hình mô phỏng (95,895 khung).
- Sai khoảng 10 px ở bước 16; độ nhạy 80%; đầu dò đọc tư thế khoảng 4 px ở **mọi** tư thế.
- Checkpoint: `checkpoints/E5_vision/legacy_dinov2_zarr_sim/vision_wm.pt` (giai đoạn E5a).

### 6.8 Các policy và thông tin mỗi policy dùng khi chạy

| `--policy` | Tư thế khối T khi chạy | Cách chọn hành động | Ghi chú |
|---|---|---|---|
| `dit_id` | **thật, từ mô phỏng** (để ghim bước 0) | inverse dynamics | `s_t` vẫn lấy từ ảnh |
| `dit_wm` (mặc định) | **thật, từ mô phỏng** | `PushDynamics` chọn trong 64 ứng viên | kết quả chính |
| `dit_id_vision` | **đọc từ ảnh** qua đầu dò trên bộ mã hoá ảnh (`--vision-backend`: jepa_wm mặc định, vjepa2_wm, dinov2_hf_wm, vision_wm) | inverse dynamics | không dùng tư thế từ mô phỏng |
| `dit_vwm` | đọc từ ảnh | World Model ảnh chọn trong 64 ứng viên (2 bước dự đoán × 4 = 8 hành động; so với `path[4]`, `path[8]`) | có tuỳ chọn `vision_wm.gate_coverage`, hiện **không đặt** (luôn dùng World Model ảnh) |
| `action_diffusion` | không dùng (chỉ `s_t` từ ảnh + đầu đẩy) | cùng kiến trúc DiT với `traj_dim = 2`, sinh 16 hành động, chạy 8 | chỉ có bản học từ zarr: `--ckpt-dir checkpoints/E3_action_diffusion`; chưa học lại trên HF |

**Vòng kín** (`policy.run_episode`):
1. `plan(obs)`.
2. Thực thi tối đa 8 hành động qua `postprocess_action`.
3. Dừng khi độ phủ > 0.95 hoặc đủ 300 bước.
4. Ghi lại max/final coverage, số bước, FDE từng bước so với đường đi, góc và khoảng cách tới đích.

---

## 7. Giao thức đánh giá

### 7.1 World Model offline (`scripts/3_world_model/evaluate_world_model.py` → `results/world_model_eval/E4_final/metrics.json`)
- **Dữ liệu:** 20 tập val, **2,235 cửa sổ** (1,846 cửa sổ có khối di chuyển), tầm 16 bước. Mỗi mô hình tưởng tượng vòng hở từ trạng thái thật.
- **Các biến thể đối chứng:**
  - mốc "đứng yên";
  - "WM cũ" (chuyển trạng thái ẩn rồi giải mã);
  - `PushDynamics` xuất phát từ tư thế thật;
  - `PushDynamics` xuất phát từ tư thế đọc ra từ ảnh (đầu giải mã của 6.1);
  - **hành động xáo trộn** (hoán vị chuỗi hành động giữa các cửa sổ). Nếu sai số không tăng, mô hình đang phớt lờ hành động.
- **Phép thử phản thực tế:**
  - Lấy 150 trạng thái val có chuyển động; chạy ba chuỗi hành động 16 bước trong gym-pusht từ đúng trạng thái đó: (1) theo người làm mẫu, (2) đứng yên, (3) chuỗi làm mẫu dịch 25 px theo hướng ngẫu nhiên.
  - **Độ nhạy** = trung vị của |hiệu hai dự đoán| / |hiệu hai kết quả thật|, chỉ xét các cặp có hiệu thật > 5 px. Lý tưởng là 100%.
  - Kèm theo trung vị sai vị trí cuối khi dùng hành động lạ.

### 7.2 World Model ảnh JEPA (`results/vision_world_models/E5b_jepa/metrics.json`)
- **Dữ liệu:** 1,915 cửa sổ val (1,232 có di chuyển > 5 px); các mốc 4/8/12/16 bước. Tư thế đọc bằng đầu dò.
- **Các dòng so sánh:**
  - "sàn" = đầu dò đọc trên khung thật của tương lai;
  - mốc đứng yên;
  - hành động xáo trộn;
  - cận trên = mô hình động lực học trạng thái;
  - phép thử phản thực tế như 7.1.

### 7.3 Benchmark vòng kín (`scripts/5_evaluation/benchmark.py` → `results/benchmark/<nhãn>.json`, `summary.md`)
- **95 điểm xuất phát cố định:**
  - `seeds50`: gym-pusht reset với seed 42 + 101·i, i = 0..49 (phân bố ngẫu nhiên mặc định của gym-pusht).
  - `hard20`: 20 vị trí chọn tay — Center-Bottom/Right/Top/Left, Q1 normal angle, Q1 inverted angle, Q1 near top wall, Q2 standard, Q2 opposite angle, Q2 near left wall, Q3 standard, Q3 extreme angle, Q3 near corner, Q4 standard, Q4 rotated 90, Q4 near right wall, Pusher opposite side, Pusher behind goal, Inverted at center, Tilted 180 at center.
  - `random25`: `RandomState(2026)`, khối x, y ∈ U(110, 400), θ ∈ U(−π, π), đầu đẩy ∈ U(90, 420).
- **Ghép cặp:** mọi policy dùng **cùng** seed torch/numpy cho từng điểm xuất phát (10000 + k, cộng 1000·`--seed-offset`), nên kết quả so sánh được theo từng cặp.
- **Quy tắc:** 300 bước; thành công khi max độ phủ > 0.95; không heuristic.
- **Chỉ số:** độ phủ TB (trung bình của max độ phủ), tỉ lệ thành công, tỉ lệ ≥ 80%, tỉ lệ < 50% ("thất bại nặng"), trung vị số bước tới thành công, kết quả theo từng nhóm.
- **So sánh cặp với một nhãn tham chiếu:** đếm số điểm xuất phát mà max độ phủ hơn/kém quá 0.10, rồi dùng sign test hai phía.
- **Thời gian:** khoảng 3–4 phút cho một lần 95 ván (`dit_wm` 214 s; `dit_vwm` 589 s).
- **Nhiễu khuếch tán:** cùng một mô hình, đổi nhiễu (`--seed-offset 1`) thì E3_augfix_ckpt__dit_wm đổi từ 84% sang 78%, v4c từ 74% sang 69%.

### 7.4 Chỉ số tính thêm để so với bài báo (2026-11-21, tính lại từ các file JSON)
- **Điểm kiểu Diffusion Policy** = trung bình qua các ván của min(max độ phủ/0.95, 1).
- **Độ phủ cuối ván.**
- **Khoảng tin cậy Wilson 95%** cho tỉ lệ thành công.
- **Kiểm định McNemar chính xác** trên thành công theo cặp.

---

## 8. Kết quả

### 8.1 Khôi phục tư thế
Xem mục 5.3. Tóm lại: khoảng 0.6 px, 97% số khung khớp reward trong khoảng 0.02.

### 8.2 World Model động lực học (`results/world_model_eval/E4_final/metrics.json`)

| Mô hình (2,235 cửa sổ val) | Sai vị trí (px) k=1 / 2 / 4 / 8 / 12 / 16 | Sai góc k=16 | Độ nhạy phản thực tế | Sai khi gặp hành động lạ |
|---|---|---|---|---|
| Đoán đứng yên | 1.65 / 3.22 / 6.37 / 12.70 / 19.08 / 25.62 | 15.0° | — | — |
| WM cũ (chuyển trạng thái ẩn) | 4.36 / 5.60 / 8.66 / 14.59 / 19.57 / 23.67 | 10.5° | **5%** | 20.8 px |
| WM cũ, hành động xáo trộn | 6.20 / 8.84 / 12.20 / 16.52 / 20.75 / 25.44 | 11.1° | | |
| **PushDynamics (chỉ demo HF)** | **0.54 / 0.55 / 0.65 / 0.79 / 0.92 / 1.06** | **0.63°** | **96%** | **1.0 px** |
| PushDynamics, xuất phát từ tư thế đọc ra từ ảnh | 3.48 / 3.43 / 3.42 / 3.54 / 3.78 / 4.14 | 2.18° | | |
| PushDynamics, hành động xáo trộn | 2.25 / 4.32 / 8.86 / 15.71 / 21.91 / 28.35 | 16.9° | | |

- **Chỉ xét cửa sổ có chuyển động, k = 16:** PushDynamics 1.19 px / 0.69°; đứng yên 31.6 px / 18.5°.
- **Sai số ở bước 1 (0.54 px) phần lớn là nhiễu của chính nhãn**, vì tư thế khôi phục có nhiễu khoảng 0.6 px.
- **Bản giai đoạn zarr** (nhãn chính xác): k = 16 là 0.7 px / 0.5°, độ nhạy 99%.

### 8.3 Vòng kín: các dòng chính (95 điểm xuất phát)

| Nhãn | Tư thế khi chạy | Dữ liệu học | Độ phủ TB | Thành công [KTC 95%] | ≥ 80% | < 50% | Điểm kiểu DP (95 / seeds50) | Độ phủ cuối |
|---|---|---|---|---|---|---|---|---|
| **E4_final__dit_wm** | thật (đặc quyền) | HF | **94.6%** | **76% [66–83]** | 99% | 1% | 0.988 / 0.980 | 0.905 |
| E4_final__dit_id | thật | HF | 87.4% | 59% [49–68] | 89% | 8% | 0.914 / 0.952 | 0.781 |
| E5b_jepa__dit_id_vision | **từ ảnh** | HF | 73.0% | 45% [36–55] | 75% | 25% | 0.764 / 0.774 | 0.630 |
| E5b_jepa__dit_vwm | từ ảnh | HF | 76.7% | 20% [13–29] | 79% | 18% | 0.805 / 0.808 | 0.670 |
| E4_hf_round0__dit_wm (chưa sửa độ lệch) | thật | HF | 90.1% | 56% [46–65] | 95% | 4% | 0.942 / 0.957 | — |
| E3_augfix_ckpt__dit_wm *(đối chứng)* | thật | zarr (+ mô phỏng) | 94.5% | 84% [76–90] | 99% | 1% | 0.985 / 0.975 | 0.936 |
| E3_dit100ep_closedloop_id60__dit_wm *(đối chứng, cùng script)* | thật | zarr (+ mô phỏng) | 92.4% | 74% [64–81] | 97% | 3% | 0.965 / 0.979 | — |
| action_diffusion *(đối chứng)* | không dùng (chỉ `s_t`) | zarr | 88.3% | 32% [23–41] | 94% | 5% | 0.926 / 0.960 | 0.787 |
| E5a_dinov2_zarr_sim__dit_id_vision *(DINOv2, cũ)* | từ ảnh | zarr + mô phỏng + DINOv2 | 76.8% | 35% [26–45] | 74% | 20% | 0.804 / 0.831 | 0.594 |

- **E4_final__dit_wm theo nhóm** (độ phủ / thành công): seeds50 94% / 78%; hard20 95% / 80%; random25 96% / 68%. Trung vị 169 bước tới thành công.
- **Điểm kiểu DP rất dễ dãi:** `action_diffusion` đạt 0.960 trên seeds50 mà chỉ thành công 34%. Tỉ lệ thành công mới là thước đo phân biệt được các policy.

### 8.4 Kiểm định cặp (McNemar chính xác trên thành công, 95 cặp)

| So sánh | Thành công | Chỉ A thắng / chỉ B thắng | p | Kết luận |
|---|---|---|---|---|
| E4_final__dit_wm vs E4_final__dit_id | 76 vs 59 | 27 / 11 | **0.014** | World Model chọn hành động có ích thật |
| E4_final__dit_wm vs action_diffusion | 76 vs 32 | 48 / 6 | < 0.001 | |
| E4_final__dit_wm vs E4_hf_round0__dit_wm | 76 vs 56 | 31 / 12 | **0.005** | sửa độ lệch nhãn khoảng 1 px có ích cho `dit_wm` |
| E4_final__dit_id vs E4_hf_round0__dit_id | 59 vs 58 | 24 / 23 | 1.0 | sửa độ lệch **không** ảnh hưởng `dit_id` |
| E4_final__dit_wm vs E3_dit100ep_closedloop_id60__dit_wm | 76 vs 74 | 18 / 16 | 0.86 | chỉ dùng HF ≈ zarr khi cùng script |
| E4_final__dit_wm vs E3_augfix_ckpt__dit_wm | 76 vs 84 | 11 / 19 | 0.20 | chưa đủ bằng chứng v2 tốt hơn |
| E3_augfix_ckpt__dit_wm vs E3_augfix_ckpt__dit_wm__noise1 (cùng mô hình, khác nhiễu) | 84 vs 78 | 17 / 11 | 0.35 | **28/95 ván đổi kết quả chỉ vì nhiễu lấy mẫu** |
| E3_augfix_ckpt__dit_wm vs E1_augfix__dit_id | 84 vs 41 | 44 / 3 | < 0.001 | |
| E5b_jepa__dit_id_vision vs E4_final__dit_id | 45 vs 59 | 12 / 25 | 0.047 | cái giá của việc chỉ dùng ảnh |
| E5b_jepa__dit_vwm vs E5b_jepa__dit_id_vision | 20 vs 45 | 8 / 32 | < 0.001 | World Model ảnh chọn hành động **làm hại** |
| E5b_jepa__dit_id_vision vs E5a_dinov2_zarr_sim__dit_id_vision (DINOv2) | 45 vs 35 | 28 / 18 | 0.18 | không khác biệt có ý nghĩa |

### 8.5 World Model ảnh JEPA

| 1,915 cửa sổ val | Sai vị trí (px) bước 4 / 8 / 12 / 16 | Sai góc bước 16 |
|---|---|---|
| Đoán đứng yên | 6.1 / 12.3 / 18.8 / 25.6 | 15.0° |
| **JEPA tưởng tượng** (tư thế đọc bằng đầu dò) | **11.6 / 11.6 / 11.4 / 11.0** | **11.3°** |
| JEPA, hành động xáo trộn | 20.1 / 24.6 / 28.0 / 31.4 | 34.6° |
| Đầu dò trên khung thật ("sàn") | 10.2 / 9.3 / 8.3 / 7.4 | 7.4° |
| Động lực học trạng thái (cận trên) | 0.6 / 0.8 / 0.9 / 1.1 | 0.7° |

- **Độ nhạy phản thực tế 77%;** sai vị trí cuối khi gặp hành động lạ 10.5 px. Mô hình **thật sự dùng hành động**.
- **Bộ mã hoá tổng quát hoá kém** (đây là nút thắt):
  - Khung val HF: đầu dò sai TB 11.1 px / 14.0° (trung vị khoảng 4 px).
  - Ảnh mô phỏng của các tập chuyên gia: 40.6 px / 71.7° (không nén), 40.6 px / 77.2° (AV1).
  - **Ảnh khởi đầu của benchmark: khoảng 30–36 px.**
  - Ba kiểu đầu dò đều sai > 27 px trên ảnh khởi đầu mô phỏng: MLP như trong script; MLP + dropout 0.2 + weight decay 0.05; đầu dò attention-pooling có kênh toạ độ. Vậy vấn đề nằm ở **bộ mã hoá**, không phải ở đầu dò.
  - Đối chiếu: DINOv2 (cũ) đọc khoảng 4 px ở mọi tư thế.
- **Diễn giải đúng (có sắc thái):**
  - Diffusion Policy học ResNet-18 **từ đầu** trên đúng 206 tập này (huấn luyện chung với policy, có cắt ảnh ngẫu nhiên) và vẫn đạt khoảng 64% thành công.
  - DynaMo tự giám sát trên đúng 206 tập và ngang ImageNet.
  - Vậy vấn đề là **công thức JEPA của dự án** (không có augmentation dịch chuyển, học tách rời policy), **không phải** "206 tập là quá ít".
  - README cũ từng có câu quá tay theo hướng ngược lại; README mới (2026-11-23) đã bỏ câu đó.
- **Vì sao `dit_vwm` tệ:** World Model ảnh tưởng tượng sai khoảng 11 px, trong khi chặng cuối (độ phủ > 95%) cần độ chính xác cỡ 1–2 px.
  - Thời DINOv2, đã thử một "cổng" chuyển sang inverse dynamics khi gần đích, dò trên seed phát triển 90000+.
  - Mọi mức cổng (không cổng, 0.5, 0.7, 0.85) đều cho thành công 20–23%, nên không có cải thiện.

### 8.6 Bảng benchmark đầy đủ (`results/summary/benchmark_table_vi.md`; nhóm: độ phủ / thành công)

| Nhãn | Nguồn | Độ phủ TB | Thành công | ≥ 80% | < 50% | Trung vị số bước | seeds50 | hard20 | random25 |
|---|---|---|---|---|---|---|---|---|---|
| E4_final__dit_wm | HF, hiện tại | 94.6 | 76 | 99 | 1 | 169 | 94/78 | 95/80 | 96/68 |
| E4_final__dit_id | HF | 87.4 | 59 | 89 | 8 | 148 | 91/62 | 90/60 | 78/52 |
| E4_hf_round0__dit_wm | HF vòng 0 (chưa sửa độ lệch) | 90.1 | 56 | 95 | 4 | 196 | 91/54 | 95/70 | 83/48 |
| E4_hf_round0__dit_id | HF vòng 0 | 88.2 | 58 | 92 | 6 | 158 | 88/58 | 94/80 | 83/40 |
| E5b_jepa__dit_id_vision | HF + JEPA, chỉ ảnh | 73.0 | 45 | 75 | 25 | 164 | 74/48 | 81/50 | 65/36 |
| E5b_jepa__dit_vwm | HF + JEPA, chỉ ảnh | 76.7 | 20 | 79 | 18 | 187 | 77/26 | 75/20 | 77/8 |
| E3_augfix_ckpt__dit_wm | zarr v2 (DiT/ID ngày 11-17) + động lực học giai đoạn zarr | 94.5 | 84 | 99 | 1 | 187 | 94/82 | 95/80 | 96/92 |
| E3_augfix_ckpt__dit_wm__noise1 | như trên, khác nhiễu | 94.1 | 78 | 98 | 2 | 195 | 96/86 | 96/75 | 90/64 |
| E1_augfix__dit_id | zarr v2 | 77.3 | 41 | 74 | 18 | 195 | 86/46 | 79/45 | 58/28 |
| E3_dit100ep_rmse_id__dit_wm | zarr v4: DiT 100 epoch, ID chọn theo RMSE offline | 83.1 | 40 | 86 | 12 | 168 | 89/42 | 72/40 | 81/36 |
| E3_dit100ep_rmse_id__dit_id | zarr v4 | 50.7 | 3 | 27 | 51 | 207 | 55/2 | 52/5 | 42/4 |
| E3_dit100ep_closedloop_id25__dit_wm | zarr v4b: DiT 100 epoch, ID chọn bằng vòng kín tối đa 25 epoch | 90.1 | 72 | 94 | 5 | 194 | 93/76 | 81/60 | 91/72 |
| E3_dit100ep_closedloop_id25__dit_id | zarr v4b | 78.1 | 51 | 74 | 17 | 177 | 81/52 | 72/55 | 78/44 |
| E3_dit100ep_closedloop_id60__dit_wm | zarr v4c: ID chọn bằng vòng kín tối đa 60 epoch (trùng script hiện tại) | 92.4 | 74 | 97 | 3 | 190 | 94/80 | 86/60 | 95/72 |
| E3_dit100ep_closedloop_id60__dit_wm__noise1 | như trên, khác nhiễu | 90.7 | 69 | 95 | 4 | 178 | 94/62 | 81/80 | 91/76 |
| E3_dit100ep_closedloop_id60__dit_id | zarr v4c | 86.0 | 64 | 86 | 13 | 165 | 91/76 | 85/55 | 76/48 |
| E3_dit100ep_history2__dit_wm | zarr, điều kiện 2 khung | 92.0 | 78 | 96 | 4 | 172 | 94/72 | 87/85 | 92/84 |
| E3_dit100ep_history2__dit_id | zarr, điều kiện 2 khung | 84.5 | 56 | 84 | 8 | 170 | 91/64 | 69/30 | 85/60 |
| E3_dit100ep_with_augfix_id__dit_wm | DiT v4 + ID v2 | 92.0 | 75 | 97 | 3 | 186 | 95/78 | 84/60 | 91/80 |
| E3_dit100ep_with_augfix_id__dit_id | DiT v4 + ID v2 | 81.6 | 58 | 81 | 16 | 175 | 86/66 | 82/60 | 72/40 |
| action_diffusion | zarr, khuếch tán hành động | 88.3 | 32 | 94 | 5 | 201 | 92/34 | 81/25 | 87/32 |
| E5a_dinov2_zarr_sim__dit_id_vision | zarr + mô phỏng + DINOv2 | 76.8 | 35 | 74 | 20 | 179 | 79/30 | 76/30 | 73/48 |
| E5a_dinov2_zarr_sim__dit_vwm | zarr + mô phỏng + DINOv2 | 84.8 | 8 | 80 | 5 | 201 | 86/12 | 84/0 | 82/8 |

Các dòng E1, E3, E5a (tên cũ `v2_*`, `v4*`, `mix_*`, `vision_*`, `action_diffusion`) thuộc giai đoạn trước khi chuyển sang HF. Chúng **chỉ là đối chứng**. Bảng mới nhất, đủ mọi lần chạy, sinh tự động: `results/summary/benchmark_table_vi.md`.

### 8.7 Đo lại ngày 2026-11-23: E0, E2 và lặp nhiễu khuếch tán
- **E0 trên giao thức chuẩn** (`E0_baseline__dit_id`, checkpoint gốc): độ phủ 34.6%, thành công 2% [1–7], khớp giao thức cũ (36.7% / 2%).
- **E2 đo lại** (`experiments/E2_handcrafted_controllers/code/benchmark_controllers.py`, mô hình E1, cùng seed):

| Cấu hình | Độ phủ | Thành công | Lần đo đầu (11-19) |
|---|---|---|---|
| + chặn xuyên thấu | 67.0% | 27% | 67.3% / 26% |
| + nhìn trước (WM cũ) | 76.0% | 19% | 78.3% / 20% |
| + chặn + nhìn trước (mặc định demo cũ) | 66.4% | 13% | 66.9% / 14% |
| tactical, khoá 88% / 80% | 86.8% | 0% | 85.2% / 0% |
| tactical, chặn + nhìn trước, **không khoá** | 91.6% | **58%** | 88.6% / 36% (có lẽ vẫn còn khoá 80%) |
| tactical, chặn, không nhìn trước, không khoá | 91.7% | 53% | 90.3% / 47% |
| tactical, không chặn, không nhìn trước, không khoá | 74.6% | 38% | 74.9% / 40% |

  - Tactical không khoá so với E1 `dit_id` (41%): p = 0.033. Luật viết tay **có giúp**, nhưng dùng trạng thái đặc quyền.
  - E3 `dit_wm` với cùng mô hình E1 (84%) so với tactical tốt nhất: p < 0.001.
- **Lặp nhiễu khuếch tán** (`__noise1`, `__noise2`: cùng mô hình, cùng điểm xuất phát, `--seed-offset` 1 và 2):

| Cấu hình | Lần 0 / 1 / 2 | Trung bình ± độ lệch |
|---|---|---|
| `E4_final__dit_wm` (bản cuối) | 76 / 69 / 66% | **71 ± 5%** |
| `E4_final__dit_id` | 59 / 48 / 57% | 55 ± 6% |
| `E5b_jepa__dit_id_vision` | 45 / 43 / 45% | 44.6% |
| `E5c_dinov2_frozen__dit_id_vision` | 49 / 53 / 56% | 52.6% |
| `E5d_vjepa2_frozen__dit_id_vision` | 54 / 56 / 52% | 53.7% |

  - `dit_wm` so với `dit_id`, gộp 3 lần (kiểm định dấu theo điểm xuất phát): 70.5% so với 54.7%, p = 0.0001. Từng lần: p = 0.014 / 0.003 / 0.22.
  - V-JEPA 2 so với DINOv2: p = 0.70 (ngang nhau). V-JEPA 2 chỉ dùng ảnh so với E4 `dit_id` dùng tư thế thật: 53.7% so với 54.7%, p = 0.69.
  - Bản cuối trên 50 seed gym-pusht: 78 / 70 / 66% (trung bình 71%), bằng LeRobot keypoints 71%.
- **Tính tất định:** chạy lại `E4_final__dit_wm` (tuần tự và song song 4 tiến trình) cho cùng quỹ đạo ở 95/95 ván: tư thế cuối chênh 0, số bước và thành công không đổi. Độ phủ chỉ lệch ở chữ số nhị phân cuối (≤ 4.4e-16).
- **Video README** (`scripts/5_evaluation/make_video.py`) phát lại đúng các ván benchmark; độ phủ cao nhất của mỗi ván phát lại khớp file JSON.
- **Chạy trên CPU (cài sạch, torch 2.6.0+cpu):** `dit_wm` đạt 72% thành công / 90.3% độ phủ, 376 s cho 95 ván trên i5-11400H.
  - Từng ván khác GPU vì bộ sinh số ngẫu nhiên của torch trên CPU khác CUDA (cùng seed, khác nhiễu), nên tương đương một lần lặp nhiễu mới.
  - So với GPU: chỉ GPU thắng 19 ván, chỉ CPU thắng 15 ván (không khác biệt có ý nghĩa).

### 8.8 World Model ảnh trên bộ mã hoá tiền huấn luyện đóng băng (E5c, E5d; 2026-11-21/22)
- **Cách làm** (`scripts/6_vision_world_model/train_frozen_encoder_wm.py`, `pusht_wm_dit/models/frozen_encoders.py`):
  - Bộ mã hoá **đóng băng**: DINOv2 ViT-S/14 (21M) hoặc V-JEPA 2 ViT-L/16 (304M, `facebook/vjepa2-vitl-fpc64-256`, nạp bằng `transformers.VJEPA2Model`, chỉ giữ encoder).
  - Ảnh 96 px phóng lên 224 / 256 px (cả khung, không cắt); 16×16 token gộp trung bình thành 8×8 = 64 token. V-JEPA 2 nhận một khung lặp lại thành clip 2 khung.
  - Đặc trưng của 25,650 khung + bản phản chiếu được lưu cache float16 (`data/cache/<enc>_feats_hf.npy`; V-JEPA 2 6.7 GB, trích khoảng 15 phút).
  - **Chỉ học** đầu dò tư thế (khoảng 1.3M) và `LatentPredictor` (khoảng 3.5M), cùng công thức và cùng cửa sổ val với JEPA. Dữ liệu học vẫn chỉ HF.
- **Kết quả** (1,915 cửa sổ val; `results/vision_world_models/comparison.md`):

| | JEPA tự học (E5b) | DINOv2 đóng băng (E5c) | V-JEPA 2 đóng băng (E5d) |
|---|---|---|---|
| Đầu dò trên khung HF (px / °) | 11.1 / 14.0 | 5.4 / 2.9 | 5.3 / 2.7 |
| Đầu dò ở 95 điểm xuất phát (ảnh mô phỏng, px / °) | 33.3 / 63.7 | 12.5 / 10.8 | 17.2 / 20.6 |
| Sai số độ thay đổi vị trí, bước 4 / 8 / 12 / 16 (px) | 6.3 / 9.2 / 10.7 / 11.7 | 5.2 / 7.4 / 9.1 / 10.3 | 5.5 / 8.0 / 9.9 / 11.5 |
| FDE bước 16 | 0.206 | 0.138 | 0.166 |
| Độ nhạy phản thực tế | 77% | 66% | 63% |
| `dit_id_vision` (vòng kín; 3 lần lặp nhiễu, TB) | 45 / 43 / 45% (44.6%) | 49 / 53 / 56% (52.6%) | 54 / 56 / 52% (53.7%) |
| `dit_vwm` (vòng kín) | 20% | 13% | 11% |

- Mốc so sánh: đoán đứng yên 25.6 px (FDE 0.301); `PushDynamics` trên trạng thái 1.1 px (FDE 0.013).
- **Giữa các bộ mã hoá** (`dit_id_vision`, gộp 3 lần): V-JEPA 2 so với DINOv2 p = 0.70; so với JEPA p = 0.13: **chưa khác biệt có ý nghĩa**. V-JEPA 2 (53.7%) so với `E4_final__dit_id` (54.7%, dùng tư thế thật): p = 0.69.
- `dit_vwm` so với `dit_id_vision`: p < 0.001 cho cả DINOv2 và V-JEPA 2. World Model ảnh chọn hành động làm hại; thứ tự `dit_vwm` trùng thứ tự độ nhạy, nhưng mới có 3 điểm nên chỉ là giả thuyết.
- V-JEPA 2 chạy chậm gấp đôi DINOv2 (415 s so với 210 s cho 95 ván) và được dùng chưa đúng sở trường (một khung lặp thành clip, ảnh 96 px phóng lên).

---

## 9. Lịch sử phát triển và bài học

| Ngày | Giai đoạn (tên cũ) | Thay đổi | Kết quả |
|---|---|---|---|
| ≤ 11-16 | **E0** (baseline_v1, code ban đầu) | WM ẩn + DiT + ID trên zarr | 50 ván: độ phủ 36.7%, thành công 2%. Đo lại 95 ván ngày 11-23: 34.6% / 2% |
| 11-17 | **E1** (v2, sửa lỗi) | Thống nhất giao thức đánh giá (`policy.py`); đặt trạng thái gọi hai lần; độ phủ lấy từ env thay vì IoU; sửa augmentation (bản cũ làm một nửa nhãn DiT không khớp ảnh) | 50 ván: 80.6% / 34%. Heuristic làm hại: stance 55.9/24, lock 67.9/0, cả hai 47.3/0 |
| 11-18/19 | **E2** — bộ điều khiển viết tay của người dùng | Chiến thuật 3 nhịp, chặn xuyên thấu, nhìn trước / MPPI trên WM cũ | Tốt nhất 90.3/47; mặc định của demo 66.9/14 (kém nhất); chỉ DiT 79.7/45. Đã lưu trữ vào `experiments/` (lý do: WM cũ mù hành động) |
| 11-19/20 | **E3** (v4, "6 bước") | `PushDynamics` mới; `dit_wm`; học lại DiT 100 epoch + EMA; chọn ID bằng vòng kín; baseline `action_diffusion`; giao diện mới | **E3_augfix_ckpt__dit_wm 94.5/84** (cao nhất từng đo); E3_dit100ep_closedloop_id60__dit_wm 92.4/74; E3_dit100ep_rmse_id__dit_id 3% (ID chọn theo RMSE offline); action_diffusion 88.3/32 |
| 11-20 | **E5a** — World Model ảnh DINOv2 (kiểu DINO-WM) | DINOv2 đóng băng; zarr + khung hình mô phỏng | khoảng 10 px ở bước 16, độ nhạy 80%; E5a_dinov2_zarr_sim__dit_id_vision 35%, E5a_dinov2_zarr_sim__dit_vwm 8% |
| 11-20 | **E4 — chỉ dùng HF (BẢN CUỐI)** | Người dùng quyết định; khôi phục tư thế từ ảnh; học lại toàn bộ | vòng 0 (chưa sửa độ lệch): E4_hf_round0__dit_wm 56%; sau khi sửa: **E4_final__dit_wm 76%**, E4_final__dit_id 59% |
| 11-20 | **E5b** — JEPA | Bộ mã hoá tự học | E5b_jepa__dit_id_vision 45%, E5b_jepa__dit_vwm 20% |
| 11-21 | So sánh với bài báo | — | mục 10 |
| 11-22 | **E5c / E5d** — bộ mã hoá tiền huấn luyện đóng băng | DINOv2 ViT-S và V-JEPA 2 ViT-L đóng băng; chỉ học bộ dự đoán + đầu dò trên HF | `dit_id_vision` 49% / 54%; `dit_vwm` 13% / 11% (mục 8.8) |
| 11-23 | Sắp xếp lại repo | Cấu trúc mới, đổi nhãn, sửa lỗi hard link, đo lại E0 / E2 trên giao thức chuẩn, lặp nhiễu cho các cấu hình chính | mục 8.7; chạy lại bản cuối khớp 95/95 ván |

**Bài học đã đo được:**
1. **Sai số offline đánh lừa.**
   - RMSE của ID khoảng 26 px ở mọi snapshot, nhưng thành công vòng kín dao động 0–65%. Chọn ID theo RMSE offline cho 3% thành công.
   - Vì vậy phải chọn bằng vòng kín trên seed riêng.
2. **Học ID trên đường đi do DiT sinh** giữ nguyên sai số offline nhưng thành công tụt từ 46% xuống 10%.
3. **World Model mù hành động còn tệ hơn không có.** Nhìn trước / MPPI dựng trên nó làm giảm thành công.
4. **Chặng cuối nhạy tới từng pixel.** Độ lệch nhãn khoảng 1 px làm `dit_wm` từ 76% còn 56%.
5. **Heuristic viết tay** (khoá, tư thế đứng, chặn xuyên thấu) đều làm giảm thành công dưới giao thức chặt.
6. **Nhiễu khuếch tán lớn.** Cùng một mô hình dao động ±6 điểm, và 28/95 ván đổi kết quả.
7. **Thiết kế mô hình động lực học quan trọng hơn lượng dữ liệu** (demo-only ≈ demo + mô phỏng).

---

## 10. So sánh với các công trình đã công bố (đã kiểm chứng nguồn ngày 2026-11-21)

### 10.1 Cùng bài toán Push-T chuẩn
Các bài dưới đây cùng 206 tập người điều khiển, đưa khối T vào ô đích, tối đa 300 bước.

| Phương pháp | Nguồn | Quan sát | Kết quả công bố |
|---|---|---|---|
| Diffusion Policy-C / -T | Chi et al., RSS 2023 | trạng thái (keypoint) | điểm 0.95 / 0.91 (C), 0.95 / 0.79 (T) — checkpoint tốt nhất / TB 10 checkpoint cuối; 3 seed huấn luyện × 50 khởi tạo |
| IBC / BET / LSTM-GMM | cùng bài | trạng thái | 0.90/0.84, 0.79/0.70, 0.67/0.61 |
| Diffusion Policy-C / -T | cùng bài | ảnh | 0.91/0.84, 0.78/0.66 (IBC 0.75/0.64, LSTM-GMM 0.69/0.54) |
| LeRobot `diffusion_pusht_keypoints` | model card HF | trạng thái | **71.0%** thành công; overlap 0.97 (500 ván) |
| LeRobot `diffusion_pusht` | model card HF | ảnh | **63.8%** thành công; overlap 0.959 (500 ván); code gốc: 64.2% / 0.957 |
| DynaMo (Cui et al., NeurIPS 2024) | bộ mã hoá tự giám sát trong miền, **đúng 206 tập** | ảnh | độ phủ cuối 0.66 (VQ-BeT) / 0.73 (Diffusion Policy) |
| (các bộ mã hoá khác đo trong bài DynaMo) | | ảnh | ImageNet 0.41 / 0.73; MoCo v3 0.57 / 0.67; R3M 0.49 / 0.63; BYOL 0.23 / 0.40; MAE 0.07 / 0.06; khởi tạo ngẫu nhiên 0.07 / 0.04 (100 ván) |

- Điểm của Diffusion Policy = trung bình của max reward, tức min(độ phủ/0.95, 1) — cùng định nghĩa với "điểm kiểu DP" ở 7.4.
- LeRobot ghi "overlap ratio" nhưng không rõ đã chia cho 0.95 hay chưa.
- DP dạng ảnh dùng ResNet-18 **không học sẵn**, GroupNorm, có cắt ảnh ngẫu nhiên.

**Đặt kết quả dự án cạnh các số trên:**
- **Dạng trạng thái:** `E4_final__dit_wm` có điểm kiểu DP 0.980 (seeds50) và thành công 78% [65–87] trên 50 ván. So với LeRobot keypoints 71% [khoảng 67–75], **ngang, chưa vượt có ý nghĩa**:
  - khoảng tin cậy chồng nhau;
  - dự án chỉ có 1 seed huấn luyện;
  - dự án dùng thêm tính toán lúc chạy (64 ứng viên + World Model);
  - seed khởi tạo khác nhau, dù cùng kiểu phân bố (gym-pusht được port từ môi trường của Diffusion Policy).
- **Chỉ dùng ảnh:** JEPA 45–48% so với 63.8% là **kém rõ**.
  - Về độ phủ cuối, 0.63–0.65 ngang DynaMo + VQ-BeT (0.66) và thấp hơn DynaMo + DP (0.73). Các bài dùng ảnh 224 px và số ván khác, nên chỉ tham khảo.
  - Baseline `action_diffusion` của dự án (cũng chỉ dùng ảnh, học từ zarr): 32%.

### 10.2 Các bài World Model (khác nhiệm vụ, không so số trực tiếp)

| Bài | Push-T của họ | Kết quả |
|---|---|---|
| DINO-WM (Zhou, Pan, LeCun, Pinto; ICML 2025) | đi tới một trạng thái đích **lấy từ dữ liệu** trong **25 bước**; **18,500** quỹ đạo | 90% (MPC); IRIS 32%, DreamerV3 30%, TD-MPC2 0%. Ablation bộ mã hoá: ResNet 20%, R3M 42%, DINO CLS 44%, DINO patch 90% |
| Terver et al. (Meta FAIR; preprint 2025/2026) | đích lấy từ dữ liệu | 70.2% so với DINO-WM 66.0% (đánh giá lại); chỉ dùng bộ mã hoá học sẵn đóng băng; DINO tốt hơn V-JEPA |
| LeWorldModel (Maes, …, LeCun, Balestriero; NeurIPS 2026; arXiv 2603.19312) | Push-T kiểu DINO-WM | World Model JEPA nhỏ (khoảng 15M tham số) học từ đầu; ở cùng ngân sách FLOPs khoảng 90% so với DINO-WM khoảng 13% (đọc từ biểu đồ) |

- Khớp với kết quả E5 của dự án: bộ mã hoá đóng băng lớn hơn (V-JEPA 2, 304M) **không** tốt hơn DINOv2 (21M) trên Push-T.

- World Model động lực học của dự án làm việc trên **trạng thái ít chiều**, dễ hơn nhiều so với dự đoán từ ảnh. Không đặt con số 1.1 px cạnh DINO-WM.

### 10.3 Ý tưởng của dự án trùng với bài nào (cần trích dẫn)

| Thành phần | Công trình gần nhất |
|---|---|
| DiT sinh đường đi, ràng buộc bằng inpainting | Diffuser (Janner et al., ICML 2022); kiến trúc DiT/AdaLN-Zero (Peebles & Xie, ICCV 2023); DDIM (Song et al., ICLR 2021) |
| Sinh chuỗi trạng thái rồi dùng inverse dynamics ra hành động (`dit_id`) | Decision Diffuser (Ajay et al., ICLR 2023); UniPi (Du et al., NeurIPS 2023, dạng video) |
| Lấy nhiều ứng viên, World Model nhiều bước chấm điểm, chọn tốt nhất (`dit_wm`) | **GPC (Qi, Yin, Du, Yang; RA-L 2026; arXiv 2502.00622) — gần nhất**: policy khuếch tán đóng băng + World Model theo hành động, xếp hạng ứng viên (GPC-RANK) trên Push-T. D-MPC (Zhou et al., DeepMind, TMLR 2025); BID (Liu et al., ICLR 2025, chọn mẫu lúc chạy trên chính Push-T) |
| Bộ mã hoá tự giám sát + bộ dự đoán ẩn + đầu IDM trên 206 tập Push-T | **DynaMo (NeurIPS 2024) — gần nhất**; PLDM (Sobal et al., 2025); I-JEPA / V-JEPA 2 (Assran et al.); VICReg (Bardes et al., ICLR 2022) |
| World Model trên đặc trưng ảnh để lập kế hoạch | DINO-WM (ICML 2025) |
| Khôi phục tư thế bằng render-and-compare | DeepIM (Li et al., ECCV 2018); CosyPose (Labbé et al., ECCV 2020); Soft Rasterizer (Liu et al., ICCV 2019) |

**Phần riêng của dự án** (chưa thấy bài nào làm đúng như vậy, nhưng chưa tra hết nên **không gọi là "đầu tiên"**):
- dùng chính reward của dataset HF để hiệu chỉnh độ lệch ảnh → thế giới và kiểm chứng tư thế khôi phục;
- quy trình đánh giá theo cặp, có kiểm định thống kê.

### 10.4 Cách nói trung thực trong báo cáo

**Được nói:**
- "Với vị trí khối từ mô phỏng, hệ thống đạt 76% ở lần chạy tham chiếu và 71 ± 5% qua ba lần lặp nhiễu (95 ván); trên 50 seed gym-pusht trung bình 71%, **tương đương** Diffusion Policy dạng trạng thái (LeRobot 71%)."
- "World Model chọn hành động nâng thành công từ 55% lên 71% (trung bình ba lần lặp nhiễu, p = 0.0001)."
- "Chỉ dùng dữ liệu HF không làm giảm hiệu năng so với zarr (p = 0.86)."

**Không được nói:**
- "Vượt Diffusion Policy / SOTA."
- "Hơn DINO-WM."
- "Bản chỉ dùng ảnh đạt mức các bài báo."
- "Kiến trúc DiT + ID + WM là ý tưởng mới."

---

## 11. Hạn chế và các vấn đề đã biết

**Hạn chế về phương pháp:**
- `dit_wm` (76%) dùng **tư thế khối T thật từ mô phỏng** để ghim bước 0 của DiT và làm trạng thái cho mô hình động lực học. Đó là thông tin đặc quyền.
- Bản chỉ dùng ảnh đạt 45% (JEPA tự học) đến 54% (V-JEPA 2 đóng băng). Bộ mã hoá JEPA đọc tư thế lạ sai khoảng 30 px; DINOv2 / V-JEPA 2 đóng băng khoảng 12–17 px.
- Dùng World Model **ảnh** để chọn hành động (`dit_vwm`) làm hại với mọi bộ mã hoá đã thử (11–20%).
- **Đích cố định** (256, 256, 45°) ở mọi tập, nên mô hình không đẩy được tới đích khác. Bước chọn 1 trong 8 của DiT dùng hàm điểm viết tay chứa vị trí đích.
- **Thống kê:** 1 seed huấn luyện; 95 ván cho khoảng tin cậy rộng ±8–9 điểm; nhiễu lấy mẫu lớn.
- **Mô phỏng được dùng để chọn snapshot ID** (vòng kín trên seed 90000–90019). Không phải dữ liệu học, nhưng là chọn mô hình bằng mô phỏng.
- **Kiểm định tư thế trên ảnh mô phỏng** nén bằng libaom, trong khi LeRobot dùng libsvtav1. Mức nén tương tự nhưng không giống hệt.
- **Các heuristic nhỏ vẫn còn:**
  - giới hạn 40 px/bước cho mọi policy;
  - 16/64 ứng viên của `dit_wm` trùng với đề xuất.
- **"WM cũ"** (chuyển trạng thái ẩn) vẫn nằm trong `WorldModel` nhưng không dùng, vì nó mù hành động.

**Trạng thái file sau khi sắp xếp lại (2026-11-23):**
- **Sự cố hard link (đã sửa).** `world_model_best.pt` ở thư mục checkpoint gốc từng là hard link với 7 thư mục giai đoạn zarr, nên lần huấn luyện HF ngày 11-20 đã ghi đè World Model của các thư mục đó.
  - Kết quả benchmark của chúng (chạy ngày 11-19) vẫn đúng.
  - World Model gốc (sha `894ca3740d15`) đã được khôi phục thành file độc lập trong `checkpoints/E1_augfix` và `checkpoints/E3_*`.
- **`PRESENTATION_SLIDES.md` lỗi thời** (trình bày guard / lookahead như đóng góp): chuyển vào `docs/archive/`, không đưa lên git.
- `results/pose_recovery/validation_summary_before_offset_fix.json` ghi số **trước** khi sửa độ lệch (92.3%); số đúng nằm trong `hf_report.json` (97.0%).
- `--policy action_diffusion` chạy được với `--ckpt-dir checkpoints/E3_action_diffusion` (bản học từ zarr). Chưa có bản học trên HF.
- Dữ liệu cũ (zarr gốc, dữ liệu mô phỏng) nằm ở `data/legacy/`, không đưa vào git. Đặc trưng DINOv2 cũ (4.6 GB, tạo lại được) đã bị xoá theo duyệt của người dùng.
- Mã của các giai đoạn cũ nằm trong `experiments/E*/code*`. Mã E0 chỉ để tham khảo (`code_snapshot/`).

---

## 12. Hướng phát triển đề xuất (chưa làm; tất cả đều giữ ràng buộc chỉ dùng HF)

1. **Hệ lai chỉ dùng ảnh (đã thảo luận, chưa làm, dự kiến E6):** tư thế đọc từ ảnh bằng đầu dò trên DINOv2 / V-JEPA 2, chọn hành động bằng `PushDynamics` thay cho World Model ảnh. Thí nghiệm này tách nguyên nhân của khoảng cách 54% → 76%: do đọc tư thế hay do World Model ảnh.
2. **V-JEPA 2 với clip 2 khung thật** thay vì nhân đôi một khung.
3. **JEPA:** thêm augmentation dịch chuyển / cắt ảnh (như DrQ và Diffusion Policy).
4. **Vòng kín bằng đầu giải mã tư thế của ResNet-18** (`wm.decode(s_t)`, khoảng 3.5 px trên khung val): **chưa đo**.
5. **Chạy 500 ván** để khoảng tin cậy còn khoảng ±4 điểm; học lại với vài seed huấn luyện.
6. **Học lại `action_diffusion` trên HF** để có baseline kiểu Diffusion Policy chỉ dùng HF.

---

## 13. Cấu trúc thư mục (sau khi sắp xếp lại, 2026-11-23)

```
<repo>/
├── README.md, README_vi.md       README cho người (tiếng Anh / tiếng Việt); bảng kết quả sinh tự động
├── LICENSE (MIT), requirements.txt, pyproject.toml, .gitignore
├── .github/workflows/smoke_test.yml   cài từ đầu + chạy thử trên Linux / Windows / macOS
├── demo.py                       demo tương tác (pygame)
├── configs/default.yaml          cấu hình toàn hệ thống (mọi đường dẫn tính từ gốc repo)
├── pusht_wm_dit/                 THƯ VIỆN
│   ├── paths.py                  ROOT, chdir_root(), load_config()
│   ├── data/                     lerobot_pusht.py (parquet + giải mã AV1), block_pose_fit.py (render-and-compare),
│   │                             pusht_dataset.py, latent_arrays.py, dynamics_data.py, precompute_latents.py
│   ├── envs/pusht_sim.py         PushTSim, VideoRecorder, PushTExpertPolicy (chỉ dùng để đo)
│   ├── models/                   world_model.py (WM phần 1), dynamics.py (WM phần 2), dit_planner.py, inverse_dynamics.py,
│   │                             jepa_encoder.py, latent_world_model.py, frozen_encoders.py (V-JEPA 2 / DINOv2), dinov2_encoder.py
│   ├── policy.py                 nạp mô hình, các policy, vòng kín, summarize
│   ├── training.py, utils/
├── scripts/                      PIPELINE theo thứ tự đề bài
│   ├── download_assets.py        tải checkpoint / dữ liệu đã xử lý từ GitHub Releases, kiểm tra SHA-256
│   ├── 1_data/                   download_dataset.py, fit_block_pose.py (hf | synthetic), inspect_dataset.py
│   ├── 2_simulation/             check_simulator.py
│   ├── 3_world_model/            train_encoder.py, train_dynamics.py, evaluate_world_model.py
│   ├── 4_diffusion/              train_dit_planner.py (DiT + ID, chọn ID bằng vòng kín), train_action_diffusion.py
│   ├── 5_evaluation/             benchmark.py, summarize_results.py, make_video.py
│   ├── 6_vision_world_model/     train_jepa_wm.py (E5b), train_frozen_encoder_wm.py (E5c/E5d; --encoder jepa --eval-only cho E5b)
│   └── release/                  package_release.py, assets.json (manifest của Release)
├── experiments/                  LỊCH SỬ: README.md tổng quan + E0_baseline/ … E5_vision_world_models/
│   │                             (mỗi thư mục: README.md song ngữ + results.md sinh tự động + mã cũ của giai đoạn đó)
│   └── run_phase3_reruns.ps1     đo lại E0 / E2 + lặp nhiễu
├── results/                      MỌI SỐ LIỆU (vào git)
│   ├── benchmark/                <nhãn>.json (95 ván mỗi file), run_registry.json, logs/, diagnostics/
│   ├── summary/                  benchmark_table.md, benchmark_table_vi.md, runs.csv, summary.json (sinh tự động)
│   ├── world_model_eval/         E4_final/ (metrics.json, curves.png, trajectories.png), E2_old_latent_wm/
│   ├── pose_recovery/            hf_report.json, synthetic_report.json, hf_tpose(_raw).npz, synthetic_frames.npz, log
│   ├── vision_world_models/      E5a_dinov2_zarr_sim/, E5b_jepa/, E5c_dinov2_frozen/, E5d_vjepa2_frozen/, comparison.md
│   ├── training_logs/            E1_augfix/, E3/, E4_hf_round0/, E4_final/
│   ├── legacy_protocol/          E0_baseline/, E1_augfix/ (giao thức cũ 50 ván)
│   ├── E2_handcrafted_controllers/  play_test_*
│   └── videos/                   MP4 (cũng có trên Releases)
├── docs/                         media/ (GIF), name_mapping.md, provenance/ (nhật ký sắp xếp lại), README_FOR_AI.md
├── checkpoints/                  KHÔNG vào git (GitHub Releases)
│   ├── final/                    E4: world_model_best, dynamics_best, dit_planner_best, inverse_dynamics_best
│   ├── E5_vision/                jepa_wm, vjepa2_wm, dinov2_hf_wm; legacy_dinov2_zarr_sim/ (vision_wm, vision_wm_v2)
│   ├── E0_baseline/, E1_augfix/, E3_dit100ep_*/, E3_augfix_dit_with_rmse_id/, E3_action_diffusion/,
│   │   E3_dynamics_variants/, E4_hf_round0/
└── data/                         KHÔNG vào git
    ├── lerobot_pusht/            bản tải từ HF (+ frames.npy đã giải mã)
    ├── pusht_hf.zarr             dữ liệu HF + tư thế khôi phục (đầu vào của mọi script học)
    ├── cache/                    precomputed_latents_hf_aug.pt, vjepa2_feats_hf.npy, dinov2_feats_hf.npy (tạo lại được)
    └── legacy/                   zarr gốc + dữ liệu tương tác sinh từ mô phỏng (chỉ E1–E3, E5a)
```

**Checkpoint bản cuối** (`checkpoints/final/`, đều học từ HF):

| File | Dung lượng | Nội dung |
|---|---|---|
| `world_model_best.pt` | 132 MB (bản phát hành 46 MB, bỏ trạng thái optimizer) | WM phần 1 |
| `dynamics_best.pt` | 0.5 MB | WM phần 2 |
| `dit_planner_best.pt` | 39.5 MB | bản EMA + bản thô |
| `inverse_dynamics_best.pt` | 2 MB | |

---

## 14. Cách chạy lại (tại gốc repo; script chạy được từ mọi thư mục)

```bash
pip install -r requirements.txt
python scripts/download_assets.py                         # bản cuối (~90 MB); --what vision | history | data | legacy | all
python demo.py                                            # demo
python scripts/5_evaluation/benchmark.py --policy dit_wm --label E4_final__dit_wm   # dit_id | dit_wm | dit_id_vision | dit_vwm | action_diffusion
python scripts/5_evaluation/benchmark.py --policy dit_wm --ckpt-dir checkpoints/E3_dit100ep_closedloop_id60 --label E3_dit100ep_closedloop_id60__dit_wm
python scripts/5_evaluation/benchmark.py --policy dit_id_vision --vision-backend vjepa2_wm --label E5d_vjepa2_frozen__dit_id_vision
python scripts/5_evaluation/summarize_results.py          # mọi bảng + bảng trong README
python scripts/5_evaluation/make_video.py grid            # GIF bản cuối (phát lại đúng các ván benchmark)
python tests/smoke_test.py                                # kiểm tra nhanh trên CPU
# Học lại từ đầu (TUẦN TỰ, không chạy song song trên GPU 4 GB)
python scripts/1_data/download_dataset.py
python scripts/1_data/fit_block_pose.py hf                # ~16 phút
python scripts/3_world_model/train_encoder.py             # ~11 phút
python scripts/3_world_model/train_dynamics.py            # ~4 phút
python scripts/4_diffusion/train_dit_planner.py           # ~27 phút
python scripts/3_world_model/evaluate_world_model.py
python scripts/6_vision_world_model/train_jepa_wm.py      # ~53 phút
python scripts/6_vision_world_model/train_frozen_encoder_wm.py --encoder vjepa2   # ~47 phút (trích đặc trưng ~15 phút)
python experiments/E2_handcrafted_controllers/code/benchmark_controllers.py      # đo lại E2
```

**Giao diện** (`demo.py`, pygame, chữ tiếng Việt):
- **Thao tác:**
  - chuột trái kéo khối T; cuộn chuột hoặc Q/E để xoay; chuột phải đặt đầu đẩy;
  - Space chạy/tạm dừng; R đặt lại; N vị trí ngẫu nhiên;
  - **M** bật/tắt "World Model chọn hành động"; **V** bật/tắt "chỉ dùng ảnh" (tự tắt M, vì `dit_vwm` kém hơn); **G** hiện dự đoán; Esc thoát.
- **Hiển thị:**
  - bóng T mờ = nơi World Model tưởng tượng khối sẽ tới sau các hành động sắp chạy;
  - bảng bên cạnh cho thấy dự đoán đó lệch bao nhiêu so với thực tế.
- Mặc định `action_mode: wm` (đúng cấu hình `E4_final__dit_wm`); tối đa 1000 bước (benchmark dùng 300).

---

## 15. API tham chiếu nhanh (để viết code tương thích)

```python
# Môi trường
from pusht_wm_dit.envs.pusht_sim import PushTSim
sim = PushTSim(env_name="gym_pusht/PushT-v0", render_size=(96, 96), max_steps=300)
obs, info = sim.reset(seed=42)
obs = sim.set_state(agent_xy=[ax, ay], block_xytheta=[bx, by, th])      # đặt chính xác (gọi _set_state 2 lần)
obs, reward, terminated, truncated, info = sim.step(action, normalized_action=True)   # action trong [-1, 1]
# obs: pixels (96,96,3) uint8 | agent_pos (2,) chuẩn hoá | agent_pos_px (2,) | block_pose (3,) [x,y,θ] px/rad
#      block_pose_4d (4,) | coverage float | step | prev_pixels | prev_agent_pos | agent_vel (2,) chuẩn hoá/bước

# Policy và vòng kín
from pusht_wm_dit.paths import chdir_root, load_config; chdir_root(); cfg = load_config()   # configs/default.yaml
from pusht_wm_dit.policy import build_policy, run_episode, summarize
pol = build_policy(cfg, "cuda", ckpt_dir=None, kind="dit_wm", vision_backend=None)   # kind: dit_id | dit_wm | dit_id_vision | dit_vwm | action_diffusion; ckpt_dir: checkpoints/<giai đoạn>; vision_backend: jepa_wm | vjepa2_wm | dinov2_hf_wm | vision_wm
actions, path = pol.plan(obs)        # actions (8,2) trong [-1,1]; path (16,4) pose4 chuẩn hoá (path[0] = hiện tại)
res = run_episode(sim, pol, seed=42, init_state=None, max_steps=300, exec_horizon=8)
# res: seed, steps, end_reason, init/final_block_pose, init_agent_pos, init/final/max_coverage, success,
#      init/final_goal_dist, init/final_angle_err_deg, mean_fde, max_fde, coverage_history

# Mô hình
wm.encode(frame_bchw_0to1, agent_norm)  -> s_t (B,128);   wm.decode(s_t) -> (pose4 (B,4), agent (B,2))
dit.sample(s_t, num_inference_steps=10, current_pose_4d=pose4 (B,4), best_of_n=8) -> path (B,16,4)
id_ctrl(s_t, path, agent_pos=agent_norm) -> actions (B,8,2)
dyn.rollout(block (B,3) [x_n,y_n,θ], agent (B,2), agent_vel (B,2), actions (B,K,2)) -> (blocks (B,K,3), agents (B,K,2))
from pusht_wm_dit.models.latent_world_model import load_vision_wm   # -> {"norm","predictor","probe","encoder"}
predictor.rollout(z_prev, z_cur, ag_prev, ag_cur, actions (B, K*4, 2)) -> (z (B,K,144,64), agents (B,K,2))
```

- **File kết quả benchmark** (`results/benchmark/<nhãn>.json`): `{label, stage, policy, ckpt_dir, vision_backend, diffusion_noise_offset, max_steps, seconds, summary: {seeds50, hard20, random25}, overall, episodes: [...]}`; các file đổi tên ngày 2026-11-23 có thêm `original_label`, `original_ckpt_dir`. Mô tả / dữ liệu học / ngày của mỗi cấu hình: `results/benchmark/run_registry.json`.
- Mỗi phần tử trong `episodes` có thêm trường `suite` và `name`.
- `summarize` trả về: `mean_max_coverage`, `std_max_coverage`, `mean_final_coverage`, `success_rate`, `rate_max_cov_ge_0.80`, `rate_max_cov_lt_0.50`, `median_steps_to_success`, `mean_fde`, `mean_final_goal_dist`, `mean_steps`.

---

## 16. Thuật ngữ

- **Độ phủ (coverage):** diện tích (khối ∩ đích) / diện tích đích. Thành công khi > 0.95.
- **Độ phủ TB:** trung bình qua các ván của max độ phủ trong ván.
- **Thất bại nặng:** max độ phủ < 50%.
- **Đặc quyền (privileged):** policy đọc tư thế khối T thật từ mô phỏng.
- **Chỉ dùng ảnh:** tư thế đọc từ ảnh; robot chỉ biết vị trí đầu đẩy của mình.
- **FDE:** sai số ở bước cuối f(Δx, Δy, Δθ).
- **Độ nhạy phản thực tế:** mức độ dự đoán phản ứng đúng với thay đổi của hành động (100% = phản ứng đúng như thật).
- **Đoán đứng yên:** mốc cho rằng khối không di chuyển.
- **Vòng 0 (`E4_hf_round0__*`, tên cũ `hf0_*`):** lần học từ HF đầu tiên, trước khi sửa độ lệch ảnh → thế giới.
- **Giai đoạn E0–E5:** xem đầu tài liệu và `experiments/README.md`. Hậu tố `__noiseN` = cùng mô hình, cùng điểm xuất phát, nhiễu khuếch tán khác; `__rerun_<ngày>` = chạy lại y hệt để kiểm tra tính tất định.
- **Điểm kiểu DP:** min(max độ phủ/0.95, 1), lấy trung bình qua các ván (thước đo của bài Diffusion Policy).

---

## 17. Nguồn tham khảo đã kiểm chứng

- Dataset: https://huggingface.co/datasets/lerobot/pusht
- Diffusion Policy (Chi et al., RSS 2023): https://ar5iv.labs.arxiv.org/html/2303.04137
- LeRobot model cards: https://huggingface.co/lerobot/diffusion_pusht · https://huggingface.co/lerobot/diffusion_pusht_keypoints
- DynaMo (NeurIPS 2024): https://arxiv.org/html/2409.12192v2
- DINO-WM: https://arxiv.org/html/2411.04983v2
- Terver et al., JEPA world models: https://arxiv.org/html/2512.24497v2
- D-MPC: https://arxiv.org/abs/2410.05364
- BID (ICLR 2025): https://arxiv.org/abs/2408.17355
- VQ-BeT (ICML 2024): https://arxiv.org/abs/2403.03181
- GPC (generative predictive control): https://arxiv.org/abs/2502.00622
- LeWorldModel: https://arxiv.org/abs/2603.19312
- V-JEPA 2 (Hugging Face `facebook/vjepa2-vitl-fpc64-256`): https://huggingface.co/facebook/vjepa2-vitl-fpc64-256
- Các bài kinh điển trích theo tên và hội nghị: Diffuser, Decision Diffuser, UniPi, DiT, DDIM, VICReg, I-JEPA / V-JEPA 2, PLDM, DrQ, DeepIM, CosyPose, Soft Rasterizer.
