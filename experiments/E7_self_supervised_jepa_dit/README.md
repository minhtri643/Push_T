# E7 — Self-supervised JEPA World Model + DiT / World Model JEPA tự giám sát + DiT

[English](#english) · [Tiếng Việt](#tiếng-việt) · Numbers / số liệu: [results/summary.md](results/summary.md), [results.md](results.md) · Written report of the branch / báo cáo của nhánh: [report/BAO_CAO_DAY_DU.md](report/BAO_CAO_DAY_DU.md)

## English

**What this stage is.** A separate research branch of the project, developed in parallel with E0–E6 and merged on 2026-11-24.
- `code/`: the original source code of the branch (`src/`, `scripts/`, `configs/`, commit `ebe99bd` of its development repository), unchanged.
- `original_results/`: the original run of the branch on a GTX 1650: logs, tables, figures, GIFs (MP4 videos left out).
- `report/`: the written report of the branch (Vietnamese) with its figures.
- Everything else in this folder re-runs and evaluates the branch here. Outside this folder, E7 only adds its 95-start runs to `results/benchmark/` (`E7_jepa_dit*.json`, listed in `run_registry.json`), so that it appears in the summary tables and the main README.

**What is different from E0–E6.** E7 is the only fully self-supervised, pixel-only system of the project. No block pose is used anywhere, neither in training nor at run time.

| | E5 vision branch (`dit_id_vision`) | E7 |
|---|---|---|
| Run-time input | image + pusher position | image + pusher position |
| Block pose | read from the image by a probe trained on pose labels (fitted from the dataset images with the known T shape) | not used |
| Encoder | E4 ResNet-18 (trained with pose labels) for the DiT, plus JEPA / DINOv2 / V-JEPA 2 for the pose | self-supervised JEPA World Model, 2.17 M parameters, then frozen (details below) |
| Planner | DiT generates the block path; inverse dynamics turns it into actions | DiT (1.36 M parameters) generates 20 pusher targets (2 s) directly; the first 8 are executed |
| World Model at run time | encoder (+ the E4 dynamics model in `dit_wm`) | encoder only; the predictor is not used |

E7's JEPA World Model takes 2 frames and produces S_t (256 dimensions). An action-conditioned predictor forecasts the latents of the next 8 frames. The target encoder is an EMA copy, and a VICReg-style loss prevents collapse.

**Same data, same split.**
- Training data: lerobot/pusht only (images, pusher position, actions).
- Diffusion Policy zarr: used only for evaluation (probe targets, FDE), as in the original code. Here it is read from `data/legacy/`.
- Episode split: identical to E0–E6, `np.random.RandomState(42).permutation(206)` with the same 20 validation episodes (checked).

**How it was run here** (2026-11-24, RTX 3050 Laptop 4 GB, main `.venv`):

| Step (original pipeline) | Time |
|---|---|
| data cache, simulator check (matches the original run: demo replay 0% / mean max coverage 0.898) | 2 min |
| JEPA World Model, 40 epochs | 56 min |
| World Model evaluation + latent export | 3 min |
| DiT policy, 300 epochs | 14 min |
| offline FDE + 50-seed evaluation | 3 min |

Differences from the original run:
1. **GPU:** RTX 3050 here, GTX 1650 in the original run.
2. **Strict fp32.**
   - On Ampere GPUs, PyTorch allows TF32 in cuDNN convolutions by default. The GTX 1650 has no TF32, so the original run is plain fp32.
   - With TF32 on, the code's own export check fails: the latents depend on the batch size, with a difference of 7.7e-3, above its 1e-3 limit. With TF32 off the difference is 2.1e-5.
   - `run_e7.ps1` therefore switches TF32 off for every step. A first World Model trained with TF32 is kept as `run/outputs/runs/wm_main_tf32` (not used).
3. **Library versions:**
   - Main `.venv`: torch 2.6.0, numpy 2.4, zarr 3.1. Original run: torch 2.5.1, numpy 2.2, zarr 2.18.
   - Packages missing from the main `.venv` are installed into `_deps/` at the original locked versions: pandas 2.3.3, scikit-learn 1.7.2, av 17.1.0, joblib, threadpoolctl, pytz.
4. **World Model epochs:** 40, as in the config. The original run was stopped by hand after 36.

**Evaluation in this repository's protocol.** `benchmark_e7.py` wraps the branch's `PushTAgent` in the episode loop of E0–E6 (`pusht_wm_dit.policy.run_episode`). It uses the same 95 starts, the same per-start torch seed and three diffusion-noise draws.
- That loop limits the commanded target to 40 px per step. The limit changes about 10% of the branch's targets, and its policy never saw it. Both variants are therefore reported: "40 px limit", the same loop as E0–E6, and "no limit", the original protocol.
- **Adapter check:** without the limit, the adapter gives the same success as the original evaluator on **50/50** of the original seeds, with identical coverage on 48/50.

### Results

**1. Reproduction** (original 50 seeds 100000–100049, DDIM 16, 8 steps executed):

| Run | Success@0.95 | 95% CI | Success@0.90 | Mean max coverage |
|---|---|---|---|---|
| Original run (GTX 1650) | 26% (13/50) | 16–40% | 50% | 77.6% |
| E7 re-training, original evaluator | **20% (10/50)** | 11–33% | 48% | 72.8% |
| E7 re-training, three noise draws | 20 / 24 / 28% → **24 ± 4%** | | 48% | 70.9% |

| Offline metric (20 validation episodes) | Original run | E7 re-run |
|---|---|---|
| best validation latent loss (copy baseline ≈ 0.16) | 0.141 | 0.145 |
| std / effective rank of S_t (no collapse) | 0.84 / 138 | 0.87 / 141 |
| linear probe R², block position / angle | 0.904 / 0.595 | 0.881 / 0.558 |
| linear probe error, position / angle | 20.8 px / 18.5° | 23.3 px / 20.0° |
| DiT FDE of the block at 2 s, position / angle / FDE_obj | 22.6 px / 13.9° / 37.2 | 21.3 px / 13.4° / 35.4 |
| inference time per path | 117 ms (GTX 1650) | 74 ms (RTX 3050) |

The original result reproduces within noise. Its single draw gave 26%; the re-training gives 24 ± 4% over three draws. The offline metrics are close: the probe errors are 6–12% higher, the FDE is 5% lower.

**2. The 95-start benchmark of E0–E6** (paired starts, 300 steps, three noise draws):

| Policy | Input | Success per draw | Mean ± sd | Success@0.90 | Coverage | E7 (no limit) vs it, pooled sign test |
|---|---|---|---|---|---|---|
| **E7, no limit** | image + pusher | 24 / 35 / 33% | **30.5 ± 5.6%** | 53% | 69.2% | – |
| **E7, 40 px limit** | image + pusher | 32 / 26 / 32% | **29.8 ± 3.0%** | 52% | 68.7% | 25 / 27, p = 0.89 |
| E3 action diffusion | image + pusher (encoder trained with pose labels) | 32% | 31.6% | 80% | 88.3% | McNemar 14 / 21, p = 0.31 |
| E5b JEPA `dit_id_vision` | image + pusher | 45 / 43 / 45% | 44.6 ± 1.2% | 67% | 74.0% | 18 / 45, p = 0.0009 |
| E5c DINOv2 `dit_id_vision` | image + pusher | 49 / 53 / 56% | 52.6 ± 3.2% | 79% | 85.1% | 15 / 50, p < 0.0001 |
| E5d V-JEPA 2 `dit_id_vision` | image + pusher | 54 / 56 / 52% | 53.7 ± 2.1% | 81% | 86.4% | 17 / 51, p < 0.0001 |
| E4 `dit_id` | true block pose | 59 / 48 / 57% | 54.7 ± 5.6% | 84% | 86.7% | 16 / 53, p < 0.0001 |
| E4 final `dit_wm` | true block pose | 76 / 69 / 66% | 70.5 ± 4.8% | 96% | 93.2% | 7 / 72, p < 0.0001 |

**3. The original 50 seeds, every policy in the same loop** (three noise draws):

| Policy | Mean ± sd | E7 (no limit) vs it, pooled sign test |
|---|---|---|
| **E7, no limit / 40 px limit** | **24.0 ± 4.0% / 26.7 ± 12.2%** | – |
| E5b JEPA `dit_id_vision` | 44.0 ± 4.0% | 9 / 28, p = 0.0026 |
| E5c DINOv2 `dit_id_vision` | 52.0 ± 4.0% | 10 / 33, p = 0.0006 |
| E5d V-JEPA 2 `dit_id_vision` | 48.0 ± 5.3% | 7 / 28, p = 0.0005 |
| E4 `dit_id` (true pose) | 54.0 ± 2.0% | 7 / 28, p = 0.0005 |
| E4 final `dit_wm` (true pose) | 66.0 ± 3.5% | 1 / 33, p < 0.0001 |

Both seed sets give the same ranking. All tables, McNemar tests per draw and per-suite numbers are in [results/summary.md](results/summary.md).

**4. The original sweeps, re-run with the original evaluator** (original 50 seeds, one noise draw; E7 re-run weights; success@0.95 / success@0.90):

| Configuration | Original run | E7 re-run | Seeds with the same outcome, McNemar p |
|---|---|---|---|
| Main (k = 8, DDIM 16) | 26% / 50% | 20% / 48% | 33/50, p = 0.63 |
| n_action_steps = 4 | 20% / 36% | 36% / 48% | 28/50, p = 0.13 |
| n_action_steps = 6 | 32% / 52% | 20% / 38% | 28/50, p = 0.29 |
| n_action_steps = 12 | 18% / 54% | 32% / 54% | 35/50, p = 0.12 |
| DDIM 4 steps | 16% / 32% | 14% / 36% | 39/50, p = 1.00 |
| DDIM 8 steps | 22% / 58% | 38% / 58% | 32/50, p = 0.10 |

The probe baselines also match the original numbers to three digits, because they do not depend on the training run: R² of the block position 0.687 for the random-init encoder and 0.821 for pixel PCA.

### Findings

1. **The original result is reproducible.** The original run reported 26% on its seeds; the re-training gives 24 ± 4% there and 30% on the 95 starts.
2. **With the same run-time inputs (image + pusher), the E5 vision branch is clearly ahead:**
   - 45–54% versus 30% on the 95 starts. The pooled p is ≤ 0.01 for every encoder in both variants, and < 0.0001 for DINOv2 and V-JEPA 2.
   - 44–52% versus 24% on the original seeds.
3. **The gap is not a pure encoder comparison.**
   - E5 uses more supervision: block-pose labels train its encoder and its probes. E7 uses none.
   - The gap therefore measures two things together: "pose supervision + block path + inverse dynamics" against "pure self-supervision + direct pusher-path diffusion".
4. **The two direct policies end at the same success.**
   - In both, the DiT outputs pusher targets directly: E7 (self-supervised JEPA) and E3 `action_diffusion` (encoder trained with pose labels). E7 reaches 30%, E3 reaches 32% (McNemar p = 0.31–1.0).
   - Both are below every DiT + inverse dynamics policy. With image input, splitting the problem into a block path and inverse dynamics adds 13–23 points in this project (E5b–E5d versus E3 and E7).
   - Direct diffusion can work much better with far more training: lerobot/diffusion_pusht reaches 65.4% (the number quoted in the branch's report). So the split helps here but is not indispensable.
5. **E7 fails differently from E3.** E3 brings the block close but rarely docks: 80% reach coverage 0.90 but only 32% succeed. E7 fails completely more often: 53% reach coverage 0.90.
6. **The 40 px/step limit of the E0–E6 loop does not change E7's success** (pooled p = 0.89), although it alters about 10% of its targets.
7. **The differences between the k / DDIM settings are noise.**
   - In the re-run, the best and worst settings swap places with the original run: k = 6 drops from 32% to 20%, while k = 4 rises from 20% to 36%.
   - No setting differs between the two runs with p < 0.05, as the branch's report already stated for 50 seeds.

### Reproduce

From the repository root, after the normal setup (main `.venv`, `data/lerobot_pusht`, `data/legacy/pusht_cchi_v7_replay.zarr` from the `pusht_data_legacy.zip` release asset):

```powershell
uv pip install --python .venv\Scripts\python.exe --target experiments\E7_self_supervised_jepa_dit\_deps --no-deps `
    pandas==2.3.3 scikit-learn==1.7.2 av==17.1.0 joblib==1.6.0 threadpoolctl==3.8.0 pytz==2026.6
$r = "experiments\E7_self_supervised_jepa_dit\run_e7.ps1"          # original pipeline steps, cwd = run/, TF32 off
powershell -File $r scripts.build_cache --config configs/default.yaml
powershell -File $r scripts.train_world_model --config configs/default.yaml --run_name wm_main
powershell -File $r scripts.eval_world_model --run_name wm_main
powershell -File $r scripts.export_latents --run_name wm_main
powershell -File $r scripts.train_policy --config configs/default.yaml --run_name dit_main
powershell -File $r scripts.eval_offline_fde --policy_run dit_main --latents data/cache/latents_wm_main.npy
powershell -File $r src.eval_sim --config configs/default.yaml --wm_run wm_main --policy_run dit_main
powershell -File $r scripts.sweep_eval --name a4 --configs 8:4 8:8                 # DDIM 4 / 8 steps
powershell -File $r scripts.sweep_eval --name sweep --configs 4:16 6:16 12:16       # n_action_steps 4 / 6 / 12
# ablations A1 / A2 of the original run (not re-run here): see code/run_ablations.ps1
# this repository's protocol (offset 0/1/2 = three noise draws; --no-rate-limit = original protocol)
python experiments\E7_self_supervised_jepa_dit\benchmark_e7.py bench95 --seed-offset 0 [--no-rate-limit]
python experiments\E7_self_supervised_jepa_dit\benchmark_e7.py orig50 --seed-offset 0 [--no-rate-limit]
python experiments\E7_self_supervised_jepa_dit\benchmark_e7.py ours50 --label E4_final__dit_wm --seed-offset 0
python experiments\E7_self_supervised_jepa_dit\analyze_e7.py
```

On Linux or macOS, replace `run_e7.ps1` with:

```bash
cd experiments/E7_self_supervised_jepa_dit/run
PYTHONPATH=../_deps:../code python -c "import runpy,sys,torch; torch.backends.cudnn.allow_tf32=False; torch.backends.cuda.matmul.allow_tf32=False; m=sys.argv[1]; sys.argv=[m]+sys.argv[2:]; runpy.run_module(m, run_name='__main__', alter_sys=True)" scripts.train_world_model --run_name wm_main
```

The checkpoints of E7 are not in the GitHub release; they are regenerated in about 80 minutes. The data cache and the checkpoints are git-ignored by `.gitignore` in this folder.

### Files

```
code/              original source code of the branch, commit ebe99bd, unchanged (src/, scripts/, configs/, requirements)
original_results/  original run (GTX 1650): outputs/ (all tables, incl. A1 / A2 / sweeps), media/*.gif,
                   report/figures/ (fig01 architecture ... fig09 success sequence, figS1-S2)
report/            written report of the branch (BAO_CAO_DAY_DU.md, Vietnamese) + its figures (fig00 is new, the rest
                   are the same files as original_results/report/figures/)
run/               working folder of the re-run: configs/default.yaml (original config, only the 3 data paths changed),
                   outputs/runs/{wm_main, dit_main, wm_main_tf32}, dit_main/eval/{main, a4_*, sweep_*}, logs/,
                   data/cache (git-ignored)
benchmark_e7.py    the branch's agent in this repository's episode loop (95 starts, original 50 seeds) + E4/E5 on those seeds
analyze_e7.py      reproduction and comparison tables -> results/summary.md, results/summary.json
results/           seeds100000/ (original 50 seeds 100000-100049; per-episode JSON, same format as results/benchmark), logs/
                   (the 95-start runs are in the repository's results/benchmark/E7_jepa_dit*.json)
results.md         generated by scripts/5_evaluation/summarize_results.py, like every stage
run_e7.ps1         runs one step of the original pipeline inside this folder (cwd = run/, PYTHONPATH = _deps + code, TF32 off)
_deps/             pandas, scikit-learn, av at the original locked versions (git-ignored)
```

**Not done.**
- Ablations A1 (random-init encoder) and A2 (no proprio) were not re-trained; the original numbers are in `original_results/`. The k / DDIM sweeps were re-run on the original 50 seeds only, not on the 95 starts.

## Tiếng Việt

**Giai đoạn này là gì.** Một nhánh nghiên cứu riêng của dự án, phát triển song song với E0–E6 và được gộp vào ngày 24/11/2026.
- `code/`: mã nguồn gốc của nhánh (`src/`, `scripts/`, `configs/`, commit `ebe99bd` của kho phát triển nhánh), **giữ nguyên**.
- `original_results/`: lần chạy gốc của nhánh trên GTX 1650 (bỏ các video MP4).
- `report/`: báo cáo viết của nhánh (tiếng Việt) cùng các hình.
- Ngoài thư mục này, E7 chỉ thêm các lần chạy trên 95 điểm xuất phát vào `results/benchmark/` (`E7_jepa_dit*.json`, khai báo trong `run_registry.json`) để có mặt trong bảng tổng và README chính.

**Khác gì so với E0–E6.** E7 là hệ thống duy nhất của dự án **hoàn toàn tự giám sát, chỉ dùng ảnh**. Không dùng tư thế khối ở bất kỳ đâu, kể cả lúc huấn luyện.
- **World Model JEPA (2,17 triệu tham số)** học biểu diễn S_t từ 2 khung ảnh, dự đoán latent của 8 khung tới theo hành động, rồi được đóng băng.
- **DiT (1,36 triệu tham số)** nhận S_t và vị trí pusher, sinh trực tiếp 20 vị trí đích của pusher (2 giây), chạy 8 bước đầu.
- Khác với nhánh ảnh E5: E5 đọc tư thế khối bằng đầu dò học từ nhãn tư thế (khớp từ ảnh), rồi dùng DiT + inverse dynamics.

**Dữ liệu và cách chia giống hệt E0–E6.** Chỉ huấn luyện bằng lerobot/pusht. Zarr của Diffusion Policy chỉ dùng để đánh giá, như mã nguồn gốc. Cùng 20 tập val (đã kiểm tra).

**Chạy lại ở đây** (RTX 3050, khoảng 80 phút tổng) có các khác biệt so với lần chạy gốc:
1. **Tắt TF32 để tính fp32 thuần** như GTX 1650 của lần chạy gốc. Nếu để TF32 bật (mặc định của PyTorch trên RTX 3050), chính phép kiểm tra trong mã nguồn sẽ báo lỗi.
2. **Phiên bản thư viện mới hơn.** Các gói còn thiếu được cài riêng vào `_deps/`, đúng phiên bản đã khóa trong mã nguồn gốc.
3. **World Model chạy đủ 40 epoch** theo cấu hình. Lần chạy gốc dừng tay ở 36.

**Kết quả chính:**

| | Đầu vào | 95 điểm xuất phát (3 lần bốc nhiễu) | 50 seed gốc |
|---|---|---|---|
| Lần chạy gốc | ảnh + pusher | – | 26% (1 lần) |
| **E7 chạy lại (không giới hạn / giới hạn 40 px)** | ảnh + pusher | **30,5 ± 5,6% / 29,8 ± 3,0%** | **24,0 ± 4,0% / 26,7 ± 12,2%** |
| E3 action diffusion | ảnh + pusher | 31,6% | – |
| E5b JEPA `dit_id_vision` | ảnh + pusher | 44,6 ± 1,2% (p = 0,0009) | 44,0 ± 4,0% (p = 0,003) |
| E5c DINOv2 `dit_id_vision` | ảnh + pusher | 52,6 ± 3,2% (p < 0,0001) | 52,0 ± 4,0% (p = 0,0006) |
| E5d V-JEPA 2 `dit_id_vision` | ảnh + pusher | 53,7 ± 2,1% (p < 0,0001) | 48,0 ± 5,3% (p = 0,0005) |
| E4 cuối `dit_wm` | tư thế thật | 70,5 ± 4,8% | 66,0 ± 3,5% |

Giá trị p là kiểm định dấu theo từng điểm xuất phát, gộp 3 lần bốc nhiễu, so với E7 không giới hạn.

**Bài học:**
1. **Kết quả gốc tái tạo được.** Lần chạy gốc đạt 26%; chạy lại cho 24 ± 4% trên cùng 50 seed. Các số đo offline gần như nhau: sai số đầu dò cao hơn 6–12%, FDE thấp hơn 5%.

   Lớp nối đưa mô hình của nhánh vào vòng benchmark của dự án và đã được kiểm chứng: khi tắt giới hạn 40 px, nó cho cùng kết quả với bộ đánh giá gốc ở **50/50 seed**.
2. **Cùng đầu vào (ảnh + vị trí pusher), nhánh E5 hơn rõ:** 45–54% so với 30%, có ý nghĩa thống kê ở cả hai bộ seed.
3. **Đây không phải so sánh encoder thuần túy.** E5 có thêm giám sát bằng nhãn tư thế, và dùng inverse dynamics. E7 tự giám sát hoàn toàn.
4. **Hai policy sinh hành động trực tiếp đều dừng ở khoảng 30%:** E7 và E3 (p = 0,31–1,0). Chúng dùng encoder rất khác nhau, nhưng đều thấp hơn mọi policy DiT + inverse dynamics.

   Với đầu vào là ảnh, tách bài toán thành quỹ đạo khối và inverse dynamics giúp thêm 13–23 điểm trong dự án này (E5b–E5d so với E3 và E7). Nhưng cách tách này không phải bắt buộc: lerobot/diffusion_pusht sinh hành động trực tiếp vẫn đạt 65,4% (số trích trong báo cáo của nhánh) khi được huấn luyện nhiều hơn rất nhiều.
5. **Giới hạn 40 px/bước không làm đổi kết quả của E7** (p = 0,89), dù nó sửa khoảng 10% số bước.
6. **Chênh lệch giữa các cấu hình k / DDIM là nhiễu.** Chạy lại bằng bộ đánh giá gốc trên 50 seed gốc, thứ hạng bị đảo:
   - k = 6 từ 32% xuống 20%, còn k = 4 từ 20% lên 36%;
   - DDIM 4: 16% → 14%; DDIM 8: 22% → 38%; k = 12: 18% → 32%.

   Không cấu hình nào khác nhau giữa hai lần chạy với p < 0,05, đúng như báo cáo của nhánh đã nêu. Hai baseline của đầu dò (encoder ngẫu nhiên, PCA trên pixel) trùng số gốc tới 3 chữ số.

**Chưa làm:**
- Chưa huấn luyện lại ablation A1 (encoder ngẫu nhiên) và A2 (chỉ ảnh); số liệu gốc có trong `original_results/`. Các lần quét k / DDIM chỉ chạy lại trên 50 seed gốc, chưa chạy trên 95 điểm xuất phát.
