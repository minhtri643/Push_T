# E6 — End-to-end vision diffusion policy (no inverse dynamics) / Chính sách khuếch tán ảnh đầu-cuối (không inverse dynamics)

[English](#english) · [Tiếng Việt](#tiếng-việt) · Numbers / số liệu: [results.md](results.md)

## English

**Goal.** Can a policy go **straight from pixels to actions**, trained end to end on the 206 demonstrations, without a block pose, a block-path planner or inverse dynamics? This is the design of Diffusion Policy (Chi et al., 2023). E6 tests it with the data and the benchmark of this project.

**E6a: model** (`code_E6a_resnet_spatial_direct/`, 2026-11-23/24):
- **Visual backbone:** ResNet-18 with GroupNorm, keeping a 6 × 6 feature map.
- **Spatial softmax:** 16 keypoints, i.e. 32 image coordinates per frame.
- **Observation:** 2 frames (t−1, t) + 2 pusher positions → a 68-dimensional condition.
- **Policy:** a DiT diffuses 16 future pusher targets directly; the first 8 are executed, then it re-plans.
- **Training:** end to end (the diffusion loss also trains the ResNet), reflection augmentation, `data/pusht_hf.zarr` (lerobot/pusht only), the same episode split as E0–E5, 30 epochs (about 9 min per epoch, 4.5 h on an RTX 3050).

**Evaluation.** `eval.py` uses the 95 starts of the benchmark (identical lists, checked) and the same per-start torch seed. Two differences from `scripts/5_evaluation/benchmark.py`:
- the actions are sent without the 40 px/step limit;
- the episode stops 10 steps after the first success, which does not change the success.

`export_benchmark_json.py` writes the result as `results/benchmark/E6a_resnet_spatial_direct.json`, so E6a is in every summary table.

**Results** (95 starts, one diffusion-noise draw):

| | Success [95% CI] | Mean max coverage | Coverage ≥ 0.80 | Severe failure (< 0.50) | seeds50 / hard20 / random25 |
|---|---|---|---|---|---|
| **E6a** | **17% (16/95) [11%–26%]** | 45.3% | 34% | 56% | 16% / 15% / 20% |

Paired with other policies on the same starts (exact McNemar test, first noise draw):

| E6a vs | Its success | Only E6a / only the other | p |
|---|---|---|---|
| E0 original baseline | 2% | 16 / 2 | 0.001 |
| E7 JEPA + DiT, no limit (also direct, also no limit) | 24% | 12 / 19 | 0.28 |
| E7 JEPA + DiT, 40 px limit | 32% | 10 / 24 | 0.024 |
| E3 action diffusion (direct actions, encoder trained with pose labels) | 32% | 14 / 28 | 0.044 |
| E5d V-JEPA 2 `dit_id_vision` (block path + inverse dynamics) | 54% | 7 / 42 | < 0.0001 |
| E4 final `dit_wm` | 76% | 6 / 62 | < 0.0001 |

**Training curve.** The validation action RMSE fell from 53.2 px (epoch 1) to a plateau of about 36.5 px (epochs 13–29). For reference, on the same validation episodes, repeating the last action gives 38.6 px and holding the position gives 45.5 px.
- The last epoch shows 28.8 px, but only because it was measured on the **full** validation set, while epochs 1–29 used 2 validation batches. The numbers are not comparable, and the "best" checkpoint is the last one for that reason, not because it improved.

**Lessons:**
1. **Direct pixel-to-action diffusion works, but weakly, at this scale.** E6a succeeds on 17% of the starts, far above the original baseline (2%), but below every policy that plans the block path and uses inverse dynamics (45–76%, p < 0.0001).
2. **All direct policies of the project end in the same range.** E6a (17%), E7 (24–32%) and E3 action diffusion (32%) predict actions directly from the image. E6a is below E3 and E7-with-limit (p = 0.04 and 0.02), and not distinguishable from E7 without the limit (p = 0.28).
3. **Coverage is the main weakness.** 56% of the episodes stay below coverage 0.50 (E3: 5%, E7: 23–27%): E6a often does not bring the block near the goal at all.
4. **Hypotheses, not tested.** Longer training (the default of `train.py` is 80 epochs), an EMA of the weights or more data might help. LeRobot's Diffusion Policy reaches 63.8% from images with a much larger model and far more training (500 episodes, other starts; see the main README). None of this was measured here.

**Reproduce** (GPU; checkpoints are not in the release):

```bash
python scripts/7_end2end_vision_diffusion/train_resnet_spatial_direct.py --epochs 30     # -> checkpoints/E6_vision/
python experiments/E6_end2end_vision_diffusion/code_E6a_resnet_spatial_direct/eval.py     # -> results/vision_world_models/E6a_resnet_spatial_direct/
python experiments/E6_end2end_vision_diffusion/export_benchmark_json.py                  # -> results/benchmark/E6a_resnet_spatial_direct.json
python scripts/5_evaluation/summarize_results.py
```

## Tiếng Việt

**Mục tiêu.** Một chính sách có thể đi **thẳng từ ảnh ra hành động** không, học đầu-cuối trên 206 demo, không cần tư thế khối, không cần DiT lập quỹ đạo khối hay inverse dynamics? Đây là thiết kế của Diffusion Policy (Chi và cộng sự, 2023). E6 kiểm tra thiết kế này với dữ liệu và benchmark của dự án.

**Mô hình E6a:**
- ResNet-18 GroupNorm, giữ bản đồ đặc trưng 6 × 6, cộng spatial softmax 16 điểm (32 tọa độ mỗi khung).
- Đầu vào: 2 khung ảnh và 2 vị trí đầu đẩy, thành vector điều kiện 68 chiều.
- DiT khuếch tán trực tiếp 16 vị trí đích của đầu đẩy, chạy 8 bước đầu rồi lập kế hoạch lại.
- Học đầu-cuối (loss khuếch tán huấn luyện luôn ResNet), augmentation phản xạ, chỉ dữ liệu lerobot/pusht, cùng cách chia tập với E0–E5, 30 epoch (khoảng 4,5 giờ trên RTX 3050).

**Đánh giá.** Đúng 95 điểm xuất phát và seed của benchmark (đã kiểm tra). Khác biệt: không có giới hạn 40 px/bước, và tập dừng 10 bước sau lần thành công đầu (không đổi kết quả thành công).

**Kết quả:** **17% thành công (16/95)**, độ phủ tối đa trung bình 45,3%, 56% số tập có độ phủ dưới 0,50.

**Bài học:**
1. **Khuếch tán trực tiếp từ ảnh ra hành động chạy được nhưng còn yếu ở quy mô này.** E6a hơn hẳn bản gốc E0 (17% so với 2%, p = 0,001), nhưng thua mọi policy có DiT quỹ đạo khối + inverse dynamics (45–76%, p < 0,0001).
2. **Các policy sinh hành động trực tiếp của dự án đều nằm cùng một vùng:** E6a 17%, E7 24–32%, E3 action diffusion 32%. E6a thấp hơn E3 và E7 có giới hạn (p = 0,04 và 0,02), không phân biệt được với E7 không giới hạn (p = 0,28).
3. **Điểm yếu chính là không đưa được khối tới gần đích:** 56% số tập có độ phủ dưới 0,50 (E3: 5%, E7: 23–27%).
4. **Sai số hành động trên tập val** giảm từ 53,2 px xuống khoảng 36,5 px (epoch 13–29). Để so sánh: lặp lại hành động cuối cho 38,6 px, đứng yên cho 45,5 px. Epoch 30 ghi 28,8 px chỉ vì được đo trên toàn bộ tập val, còn các epoch trước đo trên 2 batch, nên không so sánh được.
5. **Giả thuyết, chưa kiểm chứng:** học lâu hơn (mặc định của `train.py` là 80 epoch), dùng EMA trọng số hoặc thêm dữ liệu có thể giúp. Chưa có thí nghiệm nào ở đây đo điều đó.
