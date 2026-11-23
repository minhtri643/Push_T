# Bảng ánh xạ sắp xếp lại dự án — bản để duyệt (Giai đoạn 0)

- **Ngày:** 2026-11-23.
- **Trạng thái:** chưa có file nào trong dự án bị đổi tên, di chuyển hay xoá.
- **Bản sao lưu đầy đủ:** `D:\Thac_Si\AI_Programming\World_Model\Final_Project_PushT_snapshot_2026-11-23`
  - 1.230 file, 4,95 GB, không lỗi.
  - Chỉ bỏ qua 4 file đặc trưng tạo lại được: `vjepa2_feats_hf.npy`, `dinov2_feats_hf.npy`, `dino_feats_demo.npy`, `dino_feats_sim.npy`.
- **Cách duyệt:**
  - Đánh dấu hoặc ghi chú vào **mục K** (quyết định) và **mục I** (danh sách xoá).
  - Các mục khác chỉ cần đọc lướt; nếu không có ý kiến, mình làm đúng như bảng.

---

## A. Phát hiện trong lúc kiểm kê

1. **Lỗi toàn vẹn checkpoint (quan trọng).**
   - `pusht_wm_dit/checkpoints/world_model_best.pt` là **hard link** với 7 thư mục giai đoạn zarr: `v2`, `v4`, `v4b`, `v4c`, `v4_h2`, `mix_v4dit_v2id`, `mix_v2dit_v4id`.
   - Ngày 20/11, huấn luyện lại trên HF đã ghi đè file gốc. Vì cùng dữ liệu trên đĩa, cả 7 thư mục đó **giờ chứa World Model của HF** (sha `11a66f19367e`, val_loss 0.00067), không phải World Model gốc của giai đoạn zarr.
   - World Model gốc (sha `894ca3740d15`, val_loss 0.00115) **vẫn còn** ở `backups/before_hf/` và `backups/before_v4/`.
   - Mọi kết quả benchmark v2/v4 đều chạy ngày 19/11, **trước** khi bị ghi đè, nên số liệu đã lưu vẫn đúng.
   - **Hành động:** tách hard link và khôi phục `894ca…` vào các thư mục E1/E3. Bắt buộc làm trước khi chạy lại bất kỳ cấu hình E1/E3 nào.
2. **`results/benchmark/dit_id.json`** được tạo hôm nay lúc 16:08 (chạy benchmark mặc định).
   - Kết quả trùng `hf_dit_id` (20/11) ở **95/95 điểm xuất phát**. Đây là bằng chứng benchmark tất định: chạy lại cho ra đúng kết quả cũ.
   - **Hành động:** giữ, đổi tên thành `E4_final__dit_id__rerun_2026-11-23`.
3. **`requirements.txt` không rỗng** (436 byte; trước đây mình đọc nhầm "0 KB").
   - Nhưng file chưa ghim phiên bản và mâu thuẫn với môi trường thật: ghi `numpy<2.3` trong khi đang dùng 2.4.6, còn `zarr` đang là 3.1.6.
   - Thiếu `transformers` và `huggingface-hub`.
4. **Các log trước đây tưởng "0 KB" thực ra có nội dung:** mỗi file một dòng kết quả, 165–434 byte. **Giữ lại.**
5. **Video trùng:** `pusht_wm_dit/eval_videos/*` giống hệt `backups/baseline_v1/eval_videos/*` và `backups/before_v4/.../eval_videos/*`.
6. **Checkpoint chiếm 2,4 GB nhưng chỉ có 1,0 GB khác nhau**: nhiều file trùng nhau theo mã băm.
7. **World Model lưu kèm trạng thái optimizer** (132 MB). Bản chỉ có trọng số khoảng 46 MB. Chỉ `03_train_wm.py` ghi trạng thái này; không script nào đọc nó.

---

## B. Cây thư mục đích

```
<repo>/
├── README.md                 tiếng Anh: GIF kết quả ở đầu, mô tả, kết quả, Quickstart (MỚI)
├── README_vi.md              bản tiếng Việt đầy đủ (MỚI)
├── LICENSE                   (MỚI, xem K5)
├── pyproject.toml            ghim phiên bản theo môi trường đang chạy được (MỚI)
├── requirements.txt          viết lại, ghim phiên bản
├── .gitignore                bỏ checkpoints/, data/, .venv/, cache (MỚI)
├── .github/workflows/smoke_test.yml   chạy thử trên Linux / Windows / macOS (MỚI)
├── demo.py                   demo tương tác (← pusht_wm_dit/scripts/interactive_demo.py)
├── configs/
│   └── default.yaml          (← pusht_wm_dit/configs/default_config.yaml)
├── pusht_wm_dit/             THƯ VIỆN: chỉ chứa mã import được
│   ├── paths.py              đường dẫn tính từ gốc repo, chạy được từ mọi thư mục (MỚI)
│   ├── data/  envs/  models/  utils/
│   ├── policy.py
│   └── training.py
├── scripts/                  PIPELINE, theo đúng thứ tự đề bài
│   ├── download_assets.py    tải checkpoint/dữ liệu từ GitHub Releases + Hugging Face (MỚI)
│   ├── 1_data/               tải lerobot/pusht, khớp tư thế khối T, xem dữ liệu
│   ├── 2_simulation/         kiểm tra bộ mô phỏng
│   ├── 3_world_model/        bộ mã hoá, động lực học, đánh giá World Model
│   ├── 4_diffusion/          DiT + inverse dynamics, baseline action diffusion
│   ├── 5_evaluation/         benchmark, tạo video, bảng tổng hợp
│   └── 6_vision_world_model/ JEPA tự học, V-JEPA 2 / DINOv2 đóng băng
├── experiments/              LỊCH SỬ E0 → E5: README song ngữ cho mỗi giai đoạn + mã cũ
│   ├── README.md             bảng tiến hoá E0 → E5
│   ├── E0_baseline/
│   ├── E1_protocol_and_augmentation_fix/
│   ├── E2_handcrafted_controllers/
│   ├── E3_dynamics_world_model/
│   ├── E4_hf_only_final/                   ← BẢN TỐT NHẤT
│   └── E5_vision_world_models/
├── results/                  mọi số liệu (vào git)
│   ├── benchmark/            <giai đoạn>__<mô hình>__<policy>.json, bảng tổng, log
│   ├── world_model_eval/  pose_recovery/  vision_world_models/
│   ├── training_logs/        log huấn luyện theo giai đoạn
│   └── legacy_protocol/      E0, E1 đo theo giao thức cũ (50 ván)
├── docs/
│   ├── media/                GIF, video (GIF vào git; MP4 lên Releases)
│   ├── figures/              biểu đồ
│   ├── README_FOR_AI.md
│   └── name_mapping.md       tên/nhãn cũ → mới, để truy vết
├── checkpoints/              KHÔNG vào git → GitHub Releases
└── data/                     KHÔNG vào git → tải về / Releases / tạo lại
    ├── lerobot_pusht/  pusht_hf.zarr
    ├── cache/                file đặc trưng / latent tạo lại được
    └── legacy/               zarr gốc + dữ liệu sinh từ mô phỏng (giai đoạn cũ)
```

- Tên gói `pusht_wm_dit` **giữ nguyên**, để không làm hỏng việc nạp các checkpoint cũ.
- Tên file checkpoint giữ nguyên (`world_model_best.pt`, …), vì `--ckpt-dir` tìm file theo tên.

---

## C. Các giai đoạn (theo đúng lịch sử bạn đã đi qua)

| Giai đoạn | Ngày | Nội dung | Dữ liệu học | Kết quả chính đã có |
|---|---|---|---|---|
| **E0** Bản gốc | ≤ 16/11 | WM ẩn + DiT + ID, mã ban đầu | zarr | giao thức cũ 50 ván: 36,7% / 2% |
| **E1** Sửa giao thức + augmentation | 17/11 | đánh giá thống nhất, sửa đặt trạng thái, sửa phản chiếu | zarr | `v2_dit_id` 77,3% / 41% (95 ván) |
| **E2** Bộ điều khiển viết tay | 18–19/11 | tactical / guard / lookahead / MPPI trên WM cũ (mù hành động) | zarr | tốt nhất 90,3% / 47%; chỉ có số tổng → **chạy lại** |
| **E3** World Model động lực học (`dit_wm`) | 19/11 | `PushDynamics`, chọn hành động bằng WM; học lại DiT 100 epoch; cách chọn ID; 2 khung ảnh; baseline action diffusion | zarr + mô phỏng | `v2_dit_wm` 84% (cao nhất, dữ liệu zarr); `v4c_dit_wm` 74%; action diffusion 32% |
| **E4** Chỉ dữ liệu HF — **BẢN CUỐI** | 20/11 | khớp tư thế từ ảnh, sửa độ lệch, học lại toàn bộ | **chỉ HF** | vòng 0: 56%; **`hf_dit_wm` 76% / 94,6%** |
| **E5** Chỉ dùng ảnh (nhánh mở rộng) | 20–22/11 | DINOv2 cũ, JEPA tự học, DINOv2 / V-JEPA 2 đóng băng | E5a zarr + mô phỏng; E5b–d chỉ HF | 35 / 45 / 49 / 54% (`dit_id_vision`) |

E5 xếp sau E4 vì nó là **nhánh mở rộng chưa vượt bản cuối**. Bản tốt nhất vẫn là E4, đúng cấu hình demo đang chạy mặc định.

---

## D. Mã nguồn

### D1. Thư viện `pusht_wm_dit/` (giữ chỗ, chỉ sửa nhỏ)

| Hiện tại | Mới | Hành động |
|---|---|---|
| `pusht_wm_dit/__init__.py`, `policy.py`, `training.py` | giữ chỗ | sửa đường dẫn; đổi chỗ import guard của E2 |
| — | `pusht_wm_dit/paths.py` | **mới** |
| `pusht_wm_dit/configs/default_config.yaml` | `configs/default.yaml` | chuyển + sửa đường dẫn checkpoint/dữ liệu |
| `data/lerobot_pusht.py`, `pusht_dataset.py`, `latent_arrays.py`, `dynamics_data.py`, `precompute_latents.py` | giữ | sửa đường dẫn |
| `data/tpose_fit.py` | `data/block_pose_fit.py` | đổi tên cho rõ nghĩa |
| `envs/pusht_sim.py` | giữ | — |
| `models/world_model.py`, `dynamics.py`, `dit_planner.py`, `inverse_dynamics.py`, `jepa_encoder.py`, `latent_world_model.py`, `frozen_encoders.py` | giữ | — |
| `models/vision_encoder.py` | `models/dinov2_encoder.py` | đổi tên cho rõ nghĩa |
| `utils/metrics.py`, `utils/normalizer.py` | giữ | — |

### D2. Script pipeline (`pusht_wm_dit/scripts/` → `scripts/`)

| Hiện tại | Mới |
|---|---|
| (chỉ có lệnh một dòng trong README) | `scripts/1_data/download_dataset.py` (**mới**) |
| `11_fit_tpose.py` | `scripts/1_data/fit_block_pose.py` |
| `01_inspect_data.py` | `scripts/1_data/inspect_dataset.py` |
| `02_test_sim.py` | `scripts/2_simulation/check_simulator.py` |
| `03_train_wm.py` | `scripts/3_world_model/train_encoder.py` |
| `03b_train_dynamics.py` | `scripts/3_world_model/train_dynamics.py` |
| `06_evaluate_world_model.py` | `scripts/3_world_model/evaluate_world_model.py` |
| `04_train_dit.py` | `scripts/4_diffusion/train_dit_planner.py` |
| `04b_train_action_diffusion.py` | `scripts/4_diffusion/train_action_diffusion.py` |
| `07_benchmark.py` | `scripts/5_evaluation/benchmark.py` |
| — | `scripts/5_evaluation/make_video.py` (**mới**) |
| script phân tích trong thư mục tạm (bảng so sánh, kiểm định cặp, DP-score) | `scripts/5_evaluation/summarize_results.py` (**mới**, gộp lại) |
| `12_train_jepa_wm.py` | `scripts/6_vision_world_model/train_jepa_wm.py` |
| `13_train_frozen_wm.py` | `scripts/6_vision_world_model/train_frozen_encoder_wm.py` |
| `interactive_demo.py` | `demo.py` (gốc repo) |
| — | `scripts/download_assets.py` (**mới**) |

### D3. Mã của các giai đoạn cũ (→ `experiments/`)

| Hiện tại | Mới |
|---|---|
| `backups/baseline_v1/code/` | `experiments/E0_baseline/code_snapshot/` (chỉ để tham khảo) |
| `scripts/05_evaluate_loop.py` (giao thức cũ 50 ván, có công tắc heuristic) | `experiments/E1_protocol_and_augmentation_fix/code/evaluate_loop_old_protocol.py` |
| `pusht_wm_dit/experiments/controllers/`, `benchmark_tactical.py`, `play_test_suite.py`, `legacy_interactive_demo.py` | `experiments/E2_handcrafted_controllers/code/` |
| `pusht_wm_dit/experiments/README.md` | nội dung đưa vào `experiments/E2_handcrafted_controllers/README.md` |
| `scripts/collect_sim_interactions.py` | `experiments/E3_dynamics_world_model/code/collect_sim_interactions.py` |
| `backups/before_hf/default_config.yaml` (config giai đoạn zarr) | `experiments/E3_dynamics_world_model/config_zarr_era.yaml` |
| `scripts/08_extract_dino_features.py`, `09_train_vision_wm.py`, `10_tune_vision_gate.py` | `experiments/E5_vision_world_models/code_legacy_dinov2/` |
| `backups/before_v4/` (ảnh chụp mã cuối giai đoạn E2) | **xem K2** |

---

## E. Checkpoint (→ `checkpoints/`, không vào git, đưa lên GitHub Releases)

| Hiện tại | Mới | Ghi chú |
|---|---|---|
| `pusht_wm_dit/checkpoints/{world_model,dynamics,dit_planner,inverse_dynamics}_best.pt` | `checkpoints/final/` | **E4, bản cuối** (sha 11a66…, cf9f…, bb7f…, c471…) |
| `pusht_wm_dit/checkpoints/jepa_wm.pt`, `vjepa2_wm.pt`, `dinov2_hf_wm.pt` | `checkpoints/E5_vision/` | E5b–d |
| `backups/before_hf/checkpoints/vision_wm.pt`, `vision_wm_v2.pt` | `checkpoints/E5_vision/legacy_dinov2_zarr_sim/` | E5a |
| `backups/baseline_v1/checkpoints/*` | `checkpoints/E0_baseline/` | E0 |
| `pusht_wm_dit/checkpoints/v2/` | `checkpoints/E1_augfix/` | **khôi phục WM 894ca…**; dynamics trong thư mục này dùng cho lần chạy E3 `v2_dit_wm` |
| `…/v4/` | `checkpoints/E3_dit100ep_rmse_id/` | DiT 100 epoch, ID chọn theo RMSE offline; **khôi phục WM** |
| `…/v4b/` | `checkpoints/E3_dit100ep_closedloop_id25/` | ID chọn bằng vòng kín, tối đa 25 epoch; **khôi phục WM** |
| `…/v4c/` | `checkpoints/E3_dit100ep_closedloop_id60/` | ID chọn bằng vòng kín, tối đa 60 epoch; **khôi phục WM** |
| `…/v4_h2/` | `checkpoints/E3_dit100ep_history2/` | 2 khung ảnh làm quan sát; **khôi phục WM** |
| `…/mix_v4dit_v2id/` | `checkpoints/E3_dit100ep_with_augfix_id/` | **khôi phục WM** |
| `…/mix_v2dit_v4id/` | `checkpoints/E3_augfix_dit_with_rmse_id/` | chỉ có chạy chẩn đoán; **khôi phục WM** |
| `backups/before_hf/checkpoints/action_diffusion_best.pt` | `checkpoints/E3_action_diffusion/` | baseline giai đoạn zarr |
| `backups/before_hf/checkpoints/dynamics_demo.pt` | `checkpoints/E3_dynamics_variants/dynamics_demo_only.pt` | biến thể chỉ học từ demo |
| `backups/hf_round0/*.pt` (trừ action_diffusion) | `checkpoints/E4_hf_round0/` | E4 vòng 0 (chưa sửa độ lệch) |
| các bản trùng mã băm trong `backups/` | — | xoá, xem I4 |

**Gói Releases dự kiến:**
- `final` (để chạy demo/benchmark): khoảng 90 MB nếu bỏ trạng thái optimizer, 180 MB nếu giữ.
- `E5_vision`: khoảng 130 MB.
- `history` (E0–E4 vòng 0): khoảng 0,9 GB nếu bỏ trạng thái optimizer, 1,6 GB nếu giữ (K3); có thể tách theo giai đoạn.
- `processed_data`: `pusht_hf.zarr` + `frames.npy`, khoảng 0,7 GB.
- `legacy_data`: zarr gốc + dữ liệu mô phỏng, khoảng 1,9 GB.

Mỗi gói kèm mã SHA256.

---

## F. Dữ liệu (→ `data/`, không vào git)

| Hiện tại | Mới | Nguồn khi người khác tải về |
|---|---|---|
| `pusht_wm_dit/data/lerobot_pusht/` (video, parquet) | `data/lerobot_pusht/` | Hugging Face (`download_dataset.py`) |
| `…/lerobot_pusht/frames.npy` (khung đã giải mã, 0,7 GB) | `data/lerobot_pusht/frames.npy` | Releases (đảm bảo tái lập chính xác) |
| `pusht_wm_dit/data/pusht_hf.zarr` (tư thế khớp từ ảnh) | `data/pusht_hf.zarr` | Releases |
| `precomputed_latents_hf_aug.pt` | `data/cache/` | tự tạo lại |
| `vjepa2_feats_hf.npy`, `dinov2_feats_hf.npy` (+ `.json`) | `data/cache/` | tự tạo lại (chỉ cần cho E5c–d) |
| `pusht_cchi_v7_replay.zarr` | `data/legacy/` | Releases (`legacy_data`) |
| `sim_interactions.npz`, `sim_interactions_img.npz`, `sim_interactions_img.npz.frames.npy` | `data/legacy/` | Releases (`legacy_data`) |
| `dino_feats_demo.npy`, `dino_feats_sim.npy`, `precomputed_latents_aug.pt`, `backups/baseline_v1/data/precomputed_latents*.pt` | — | **xoá** (I2, I3), tạo lại được |

---

## G. Kết quả (→ `results/`, vào git)

### G1. Đổi nhãn benchmark

- Chỉ đổi `label` và điền `ckpt_dir` còn trống.
- Thêm trường `original_label` để truy vết.
- **Không đụng** vào số liệu từng ván.

| Nhãn cũ | Nhãn mới |
|---|---|
| `v2_dit_id` | `E1_augfix__dit_id` |
| `v2_dit_wm`, `v2_dit_wm_seed1` | `E3_augfix_ckpt__dit_wm`, `…__noise1` |
| `v4_dit_id`, `v4_dit_wm` | `E3_dit100ep_rmse_id__dit_id`, `…__dit_wm` |
| `v4b_dit_id`, `v4b_dit_wm` | `E3_dit100ep_closedloop_id25__dit_id`, `…__dit_wm` |
| `v4c_dit_id`, `v4c_dit_wm`, `v4c_dit_wm_seed1` | `E3_dit100ep_closedloop_id60__dit_id`, `…__dit_wm`, `…__dit_wm__noise1` |
| `v4_h2_dit_id`, `v4_h2_dit_wm` | `E3_dit100ep_history2__dit_id`, `…__dit_wm` |
| `mix_v4dit_v2id_dit_id`, `…_dit_wm` | `E3_dit100ep_with_augfix_id__dit_id`, `…__dit_wm` |
| `action_diffusion` | `E3_action_diffusion_baseline` |
| `hf0_dit_id`, `hf0_dit_wm` | `E4_hf_round0__dit_id`, `…__dit_wm` |
| `hf_dit_id`, **`hf_dit_wm`** | `E4_final__dit_id`, **`E4_final__dit_wm`** |
| `dit_id` (hôm nay) | `E4_final__dit_id__rerun_2026-11-23` |
| `vision_dit_id`, `vision_dit_vwm` | `E5a_dinov2_zarr_sim__dit_id_vision`, `…__dit_vwm` |
| `jepa_dit_id_vision`, `jepa_dit_vwm` | `E5b_jepa__dit_id_vision`, `…__dit_vwm` |
| `dinov2hf_dit_id_vision`, `…_dit_vwm` | `E5c_dinov2_frozen__dit_id_vision`, `…__dit_vwm` |
| `vjepa2_dit_id_vision`, `…_dit_vwm` | `E5d_vjepa2_frozen__dit_id_vision`, `…__dit_vwm` |
| `diagnostics/diag_*.json` | `diagnostics/E3_diag_*.json` |

### G2. Các kết quả khác

| Hiện tại | Mới |
|---|---|
| `results/baseline_v1/` | `results/legacy_protocol/E0_baseline/` |
| `results/v2_aug_fix/` (trừ `log_train_*`) | `results/legacy_protocol/E1_augfix/` |
| `results/v2_aug_fix/log_train_*.txt` | `results/training_logs/E1_augfix/` |
| `results/v4/log_train_*.txt` | `results/training_logs/E3/` |
| `results/v4/log_bench_*.txt`, `results/hf_pipeline/07_*.log`, `results/vision_wm/log_bench_*` | `results/benchmark/logs/` |
| `results/hf_pipeline/0[3-6]*.log`, `12_jepa_wm.log` | `results/training_logs/E4_final/` |
| `backups/hf_round0/*.log` | `results/training_logs/E4_hf_round0/` |
| `results/world_model_eval_metrics.json`, `wm_multistep_error_curves.png`, `wm_prediction_visual_trajectories.png` (19/11, WM ẩn cũ mù hành động) | `results/world_model_eval/E2_old_latent_wm/` |
| `results/wm_eval/` | `results/world_model_eval/E4_final/` |
| `results/play_test_scenarios.png`, `play_test_summary.json` | `results/E2_handcrafted_controllers/` |
| `results/tpose_fit/` | `results/pose_recovery/` |
| `results/tpose_fit/validation_summary.json` (số **trước** khi sửa độ lệch) | `results/pose_recovery/validation_summary_before_offset_fix.json` |
| `results/vision_wm/`, `vision_wm_v2/` | `results/vision_world_models/E5a_dinov2_zarr_sim/` |
| `results/jepa_wm/` | `results/vision_world_models/E5b_jepa/` |
| `results/dinov2_hf_wm/` | `results/vision_world_models/E5c_dinov2_frozen/` |
| `results/vjepa2_wm/` | `results/vision_world_models/E5d_vjepa2_frozen/` |
| `results/vision_encoders_comparison.md` | `results/vision_world_models/comparison.md` |
| `backups/baseline_v1/eval_videos/*.mp4` | `results/legacy_protocol/E0_baseline/videos/` |
| `backups/baseline_v1/eval_videos/*.json` | `results/legacy_protocol/E0_baseline/` (so trùng trước khi chuyển) |

---

## H. Tài liệu

| Hiện tại | Mới |
|---|---|
| `README.md` | viết lại bằng tiếng Anh (giữ phần nội dung còn đúng) |
| — | `README_vi.md` |
| `README_FOR_AI.md` | `docs/README_FOR_AI.md`, cập nhật theo cấu trúc mới và kết quả E5 |
| `PRESENTATION_SLIDES.md` | **xem K1** |
| `REORG_MAPPING.md` (file này) | `docs/provenance/reorg_mapping_2026-11-23.md` khi xong |

---

## I. Danh sách XOÁ — cần bạn duyệt

- Mọi file dưới đây đều còn trong bản sao lưu ngoài dự án.
- Gạch dòng nào bạn **không** muốn xoá.

| # | File | Dung lượng | Lý do | Tạo lại được? |
|---|---|---|---|---|
| I1 | mọi thư mục `__pycache__/` | nhỏ | Python tự tạo | có, tự động |
| I2 | `pusht_wm_dit/data/dino_feats_demo.npy`, `dino_feats_sim.npy` | 4,6 GB | đặc trưng DINOv2 của nhánh E5a cũ | có (mã E5a + `data/legacy`) |
| I3 | `pusht_wm_dit/data/precomputed_latents_aug.pt`, `backups/baseline_v1/data/precomputed_latents*.pt` | 57 MB | cache latent giai đoạn zarr / E0 | có |
| I4 | checkpoint trùng mã băm: `backups/before_hf/checkpoints/{world_model,dit_planner,inverse_dynamics,dynamics}_best.pt`, `dynamics_demo_sim.pt`; `backups/before_v4/pusht_wm_dit/checkpoints/*`; `backups/hf_round0/action_diffusion_best.pt` | ~0,4 GB | sau khi khôi phục vào `checkpoints/E1…`/`E3…` thì chỉ còn là bản trùng | — (đã có bản giống hệt) |
| I5 | `pusht_wm_dit/eval_videos/*`, `backups/before_v4/pusht_wm_dit/eval_videos/*` | 0,2 MB | trùng `backups/baseline_v1/eval_videos` | — |
| I6 | `pusht_wm_dit/data_inspection.png` | 0,1 MB | ảnh từ dữ liệu zarr; sẽ tạo lại từ dữ liệu HF | có |
| I7 | `results/benchmark/summary.md`, `summary_log.txt` | nhỏ | bảng cũ; sẽ sinh lại bằng script | có |
| I8 | thư mục `backups/` (khi đã rỗng sau các bước chuyển) | — | mọi thứ đã chuyển hoặc là bản trùng | — |
| I9 | `backups/before_v4/` phần mã nguồn | 0,3 MB | **tuỳ K2** | — |
| I10 | `PRESENTATION_SLIDES.md` | 15 KB | **tuỳ K1** | — |

**Không xoá:**
- `vjepa2_feats_hf.npy`, `dinov2_feats_hf.npy` (9 GB): chuyển vào `data/cache/` và giữ ở máy cho tiện chạy lại; không lên git.
- Mọi log, JSON, hình.

---

## J. Thay đổi nội dung mã (không chỉ di chuyển)

1. `pusht_wm_dit/paths.py`: mọi script nạp config và đường dẫn tính từ gốc repo, nên **chạy được từ bất kỳ thư mục nào**.
2. `configs/default.yaml`: đường dẫn checkpoint → `checkpoints/final/…`, dữ liệu → `data/…`.
3. Tách hard link và khôi phục World Model gốc vào các thư mục E1/E3.
4. `policy.py`: import guard của E2 từ chỗ mới. Demo: bật "Chỉ dùng ảnh" thì **tự tắt** chọn hành động bằng World Model, để không rơi vào cấu hình tệ nhất (20%).
5. Đường dẫn trong `train_frozen_encoder_wm.py` (nó nạp `benchmark.py`), docstring và dòng hướng dẫn chạy của mọi script.
6. `requirements.txt` + `pyproject.toml` ghim theo môi trường đang chạy được (torch 2.6.0, gym-pusht 0.1.8, numpy 2.4.6, zarr 3.1.6, transformers 4.57.6, …).
7. JSON benchmark: đổi nhãn + điền `ckpt_dir` + `original_label`. Số liệu từng ván giữ nguyên.
8. Kiểm tra chạy trên CPU (`resolve_device`).
9. **Kiểm tra hồi quy sau bước này:** benchmark `E4_final__dit_wm` phải ra lại **đúng 95/95 ván** như `hf_dit_wm`.

---

## K. Cần bạn quyết định

| # | Câu hỏi | Đề xuất của mình |
|---|---|---|
| K1 | `PRESENTATION_SLIDES.md` (slide của bạn, ngày 19/11): trình bày guard / lookahead như đóng góp, trong khi đo đạc cho thấy chúng làm hại | **Không đưa lên git**: chuyển vào `docs/archive/` và đưa vào `.gitignore` (giữ ở máy). Hoặc bạn chọn xoá. |
| K2 | `backups/before_v4/` (ảnh chụp mã cuối giai đoạn E2) | **Xoá** khỏi dự án: mã E1/E2 đã nằm trong mã hiện tại + `experiments/E2…/code`; bản đầy đủ vẫn còn trong bản sao lưu ngoài. |
| K3 | Checkpoint đưa lên Releases có **bỏ trạng thái optimizer** không (World Model 132 MB → khoảng 46 MB; không script nào cần nó) | **Có** (bản gốc vẫn giữ ở máy); ghi lại mã băm gốc để truy vết. |
| K4 | Bộ mã hoá ảnh cho chế độ "Chỉ dùng ảnh" trong demo | **Giữ JEPA**: không cần tải thêm. V-JEPA 2 tốt hơn (54% so với 45%, chưa có ý nghĩa thống kê) nhưng phải tải 1,2 GB trọng số và chạy chậm gấp đôi; vẫn chọn được bằng tham số. |
| K5 | Giấy phép cho repo công khai | **MIT** |
