# PLAN PHASE C — Vòng cải tiến cuối (lập 2026-11-16, chốt TRƯỚC khi có số)

Đọc kèm: `docs/BACKUP_PLAN.md` (tiêu chí F1–F5, chẩn đoán D1–D6, các hướng A/B/C/D/E),
`docs/HANDOFF.md` (mục "B4a §10", "CL2N", "P4 + Stage 1 + end-to-end"), `docs/PLAN_STAGE2_ONESHOT.md`.

> Mọi luật quyết định trong file này được ghi **trước khi chạy** Bước 1. Sửa luật sau khi thấy
> số = chọn theo kết quả; nếu buộc phải sửa, ghi lý do + ngày vào cuối file, không xoá bản cũ.

---

## 0. Vì sao có vòng này

Đối chiếu kết quả 2026-11-15 với BACKUP_PLAN §1:

| Mã | Kết quả | Kết luận |
|---|---|---|
| F1 | P1 `h_in` top-1 0.036 (R1a), không khác ImageNet + CL2N 0.105 ở mức lớp | **Đúng F1 trên P1** |
| F2 | P4: nhóm S tốt; H tự nhiên 0.38–0.50; H chọn 0.23 | **F2 trên P4** |
| F3 | P1 AUROC 0.78 (đạt); P4 AUROC cos 0.68–0.70 (sát 0.70), lề 0.88 | Một phần, chủ yếu do τ không chuyển miền |
| F4 | P3 chưa đo | — |

Chẩn đoán đã có:
- **D1 (top-k)**: lẫn lộn. S19 top-3 0.999 / top-1 0.001; S3 top-3 0.85; R1b `flow_direction` hạng 2
  ở 99.7% crop P4 → một phần là lỗi *quyết định*. S4, S10, S18 top-3 = 0 → một phần là lỗi *biểu diễn*.
- **D2 (lệch đầu vào)**: **chưa từng sửa**. `utils/dataset.py` dùng `Resize((imgsz, imgsz))` → bóp méo
  tỉ lệ (template rộng/cao 0.38–3.58); template sạch vs crop nhiễu. A4 được khuyến nghị làm trước R1
  nhưng đã bị bỏ qua.
- **Nghi phạm chính**: khác **phong cách vẽ** giữa ảnh mẫu ISO và crop DigitizePID. Bằng chứng gián tiếp:
  cùng encoder, P4 (crop cùng phong cách với template) tốt hơn P1 rất nhiều. Chưa phải kết quả đo
  trực tiếp → thí nghiệm 1e dưới đây kiểm.

Mốc thời gian (BACKUP_PLAN §5, **giả định hạn nộp đầu 1/2027 — user chưa xác nhận**):

| Mốc | Ngày | Việc |
|---|---|---|
| Xong Bước 1 | ~2026-11-20 | Có bảng thăm dò + quyết định Bước 2 |
| Hạn chót **bắt đầu** Bước 2 | 2026-11-25 | Sau ngày này không bắt đầu việc phải train lại |
| Xong Bước 2 | ~2026-12-10 | Số Colab đầy đủ cho phiên bản mới |
| **Đóng băng thí nghiệm** | 2026-12-15 | Dù Bước 2 ra sao; sau đó chỉ viết |

---

## 1. Bước 1 — Thăm dò rẻ (không train lại; CPU cục bộ; ~2–3 ngày)

### 1.0 Dữ liệu và code chung

- **Code** (vào repo, để tái lập trong luận văn):
  - `src/utils/input_norm.py`: chuẩn hoá đầu vào A4 (dưới), dùng lại nguyên ở Bước 2.
  - `scripts/oneshot_probe.py`: cắt crop từ COCO DigitizePID theo đúng cách suy luận (box + 10%,
    `utils.crop_box.cut`), nhúng qua 8 view + `aggregate=mean`, chấm. Một bảng chung cho mọi dòng.
  - Test cho `input_norm` và phần chấm của probe. Không sửa `configs/config.yaml`, không thêm
    dependency (DINOv2 qua `timm` có sẵn; chỉ tải trọng số như ImageNet).
- **Hai tập crop** (cố định seed 0, ghi danh sách crop ra CSV):
  1. **Tập quyết định — "P1-val"** (chỉ dùng tập này cho luật ở 1.2): ảnh `val_sel` (50 ảnh, đã dùng
     chọn τ — không phải test). Lớp **biết** = 10 lớp R1a đã train có ảnh mẫu ISO: S1, S17, S21, S24
     (nhóm S) + S6, S8, S14, S20, S22, S23 (VARIANT, chấm theo ảnh mẫu ISO tương ứng như P2). Lớp
     **lạ** = 7 lớp OOV R1a đã train: S7, S16, S26, S28, S29, S31, S32. Tối đa 150 crop/lớp.
     Với encoder **không train trên DigitizePID** (ImageNet, DINOv2) mọi lớp này đều "chưa thấy" →
     cùng bản chất P1 nhưng không đụng ảnh/lớp nào của P1.
  2. **Tập tham khảo — mẫu P1** đã dùng cho CL2N (150 crop/lớp × 6 `h_in` + 4 `h_out`, + 30 crop/lớp
     × 19 lớp R1a đã thấy cho tâm). **Chỉ báo cáo, không dùng để quyết định** (P1 là tập test).
- **Tâm CL2N**: TB embedding mọi view của crop **train** các lớp R1a đã thấy (như notebook 10d), tính
  lại cho từng encoder và từng kiểu tiền xử lý. Ghi chú: tập P1-val dùng ảnh `val_sel`, tâm chỉ từ
  ảnh train → không có ảnh nào dùng chung; tâm có chứa lớp của P1-val (thống kê không nhãn) — ghi
  rõ trong luận văn.

### 1.1 Các dòng thăm dò

| # | Dòng | Encoder | Đổi gì | Vai trò |
|---|---|---|---|---|
| 1a | Mốc | ImageNet + CL2N; R1a | Không đổi | Mốc so sánh (phải tái lập số đã có trên mẫu P1: 0.131 / 0.044) |
| 1b | **A4 lúc suy luận** | ImageNet + CL2N (chính); R1a (tham khảo) | Chuẩn hoá cả crop lẫn ảnh mẫu (dưới) | Ứng viên Bước 2A |
| 1c | **A1 DINOv2** | `timm` ViT-S/14 DINOv2 (kiểm tên model khi làm), thô và + CL2N; có / không A4 | Đổi encoder, không train | Ứng viên Bước 2B; dù sao cũng là một dòng baseline |
| 1d | **B2 không học** | Không có encoder: HOG (bản numpy HOG-lite đã dùng ở phân tích S9) trên ảnh đã A4 | So cosine HOG crop ↔ HOG ảnh mẫu | Baseline "phương pháp đơn giản nhất"; **không** kích hoạt Bước 2 |
| 1e | **Ảnh mẫu cùng phong cách** | R1a; ImageNet + CL2N | Thư mục = 1 crop DigitizePID / lớp thay cho ảnh mẫu ISO | Kiểm chẩn đoán "khác phong cách"; **không** kích hoạt Bước 2 |

**A4 (1b, 1c, 1d)** — `utils/input_norm.py`, áp **giống hệt** cho crop và ảnh mẫu, trước transform hiện có:
1. Xám → nhị phân Otsu (nét = đen, nền = trắng).
2. Cắt sát phần mực của **thành phần liên thông chứa tâm** (cộng các thành phần chạm vào khung tâm
   50%), để đường ống / chữ ở rìa crop không kéo dài khung.
3. Letterbox: đệm trắng thành hình vuông, giữ tỉ lệ, lề 10% (khớp `CROP_MARGIN_FRAC`).
4. Không làm mảnh / làm dày nét ở vòng này (BACKUP_PLAN A4 bước 4: tuỳ chọn, để sau).

**1e — thiết kế chi tiết** (trên mẫu P1, chỉ chẩn đoán):
- Thư mục = 1 crop mỗi lớp DigitizePID lấy từ ảnh **train**, cho 28 lớp (32 trừ 4 lớp `h_out`, vì
  `h_out` là "lạ"). Crop truy vấn = mẫu P1; bỏ mọi truy vấn cùng ảnh với crop làm mẫu.
- Lặp 5 lần chọn crop mẫu (seed 0–4), báo TB ± std. Ngẫu nhiên top-1 = 1/28 ≈ 0.036 (khác 1/42 của
  P1 với ảnh mẫu ISO — báo kèm mức ngẫu nhiên, không so số tuyệt đối giữa hai thư mục).
- Đọc: top-1 `h_in` ≥ 0.5 → khác phong cách là nút thắt chính (P1 hiện tại 0.036). Gần ngẫu nhiên
  → R1a không tổng quát sang lớp mới **kể cả cùng phong cách** → vấn đề nằm ở biểu diễn, không chỉ
  ở ảnh mẫu. Đây cũng là con số gần với protocol OSSR-PID nhất (cùng phong cách), nhưng trên lớp
  chưa thấy — vẫn ghi rõ khác protocol.

**Chỉ số mỗi dòng** (cả hai tập): macro top-1 / top-3 của lớp biết, từng lớp; AUROC cos và lề (biết vs
lạ); tỉ lệ hub (tên dự đoán nhiều nhất). Cùng định nghĩa như `oneshot_eval` P1.

### 1.2 Luật quyết định Bước 1 → Bước 2 (chốt 2026-11-16)

Chỉ xét **1b** (A4 trên ImageNet + CL2N) và **1c** (DINOv2, bản tốt hơn trong thô / + CL2N, có / không
A4 — chọn trên P1-val), so với mốc **ImageNet + CL2N không A4** trên **tập P1-val**:

- **ĐẠT** nếu cả hai: (i) top-1 tăng ở **≥ 6/10 lớp biết** (lớp mà cả hai bản đều 0 tính là *không*
  tăng); và (ii) macro top-1 tăng **≥ +0.05**.
- **Hoặc ĐẠT** nếu AUROC cos (hoặc lề) tăng **≥ +0.05** mà macro top-1 không giảm quá 0.01.
- 1d, 1e và mọi số trên mẫu P1 **không** tham gia quyết định (chỉ báo cáo).

Kết quả:
- 1b ĐẠT → **Bước 2A**. 1c ĐẠT → **Bước 2B**. Cả hai → 2B trước (không train), rồi 2A nếu còn kịp
  hạn 2026-11-25. Không dòng nào ĐẠT → **bỏ Bước 2**, sang Bước 3.

## 1.3 Kết quả Bước 1 — 2026-11-16 (chạy cục bộ CPU, `runs/phase_c/probe/`, `report.txt`)

`python scripts/oneshot_probe.py --out runs/phase_c/probe` (~1.5 giờ, Mac phải giữ thức bằng `caffeinate`).
4,760 crop (P1-val 2,550; mẫu P1 2,070; ảnh mẫu 1e 5 × 28). Kiểm: prototype dựng lại khớp file eval dùng
(ImageNet sai khác 0, R1a 1.6e-6); 1a tái lập đúng mẫu P1 (ImageNet + CL2N 0.131, R1a 0.044).

| dòng (encoder / đầu vào / CL2N) | P1-val top-1 | top-3 | AUROC cos | AUROC lề | mẫu P1 top-1 | top-3 | AUROC |
|---|---|---|---|---|---|---|---|
| imagenet / thô / thô | 0.158 | 0.389 | 0.623 | 0.541 | 0.010 | 0.119 | 0.477 |
| **imagenet / thô / CL2N (mốc)** | **0.175** | 0.360 | 0.427 | 0.354 | 0.131 | 0.407 | 0.444 |
| imagenet / A4 / thô | 0.119 | 0.215 | 0.746 | 0.682 | 0.079 | 0.219 | 0.767 |
| **1b imagenet / A4 / CL2N** | 0.157 | 0.362 | 0.456 | 0.535 | 0.367 | 0.600 | 0.740 |
| lo (R1a) / thô / thô | 0.600* | 0.700 | 0.847 | 0.672 | 0.044 | 0.384 | 0.776 |
| lo / A4 / CL2N | 0.278* | 0.439 | 0.696 | 0.657 | 0.370 | 0.622 | 0.787 |
| dinov2 / thô / thô | 0.373 | 0.582 | 0.386 | 0.576 | 0.311 | 0.470 | 0.346 |
| dinov2 / A4 / thô | 0.433 | 0.552 | 0.625 | 0.688 | 0.204 | 0.332 | 0.250 |
| **1c dinov2 / A4 / CL2N** | **0.477** | 0.637 | 0.692 | 0.561 | **0.579** | **0.829** | 0.351 |
| 1d HOG / thô | 0.062 | 0.268 | 0.477 | 0.544 | 0.150 | 0.327 | 0.653 |
| 1d HOG / A4 | 0.053 | 0.180 | 0.474 | 0.562 | 0.020 | 0.214 | 0.575 |

\* R1a ĐÃ train trên mọi lớp P1-val → với R1a, P1-val là lớp đã thấy (không dùng để quyết định).

**Quyết định theo luật §1.2 (mốc imagenet / thô / CL2N trên P1-val):**
- **1b** imagenet / A4 / CL2N: top-1 tăng 3/10 lớp, Δ macro top-1 −0.018; AUROC lề +0.181 nhưng top-1 giảm
  0.018 > 0.01 → **KHÔNG ĐẠT** (cả hai nhánh). → Không làm 2A (train lại R1a với A4).
- **1c** DINOv2 (bản tốt nhất trên P1-val: A4 + CL2N): tăng 6/10 lớp (S1 0.65→0.99, S21 0.53→0.99, S24
  0→1.0, S20, S23, S17), Δ macro top-1 **+0.302**, Δ AUROC +0.265 → **ĐẠT** → **Bước 2B**.

**Tham khảo (mẫu P1 = test, không dùng quyết định):**
- DINOv2 / A4 / CL2N: top-1 0.579 / top-3 0.829 (R1a 0.044 / 0.384; ImageNet + CL2N 0.131 / 0.407). Từng lớp
  `h_in`: S10 0.96, S18 0.90, S5 0.79, S3 0.41, S19 0.41, S4 0.00. Nhưng AUROC 0.351 **< 0.5** (lớp lạ có cos cao
  hơn lớp biết) → cổng "Other" ngược chiều trên P1 với DINOv2; trên P1-val AUROC 0.692.
- A4 giúp mạnh trên P1 cho mọi encoder (ImageNet + CL2N 0.131 → 0.367; R1a 0.044 → 0.370 dù R1a train không
  có A4) nhưng hại trên P1-val với ImageNet (S1 0.65 → 0.14). Luật đã chốt không cho dùng số P1 để mở 2A.
- HOG (không học) ≈ ngẫu nhiên trên P1-val → baseline "đơn giản nhất" yếu, đủ làm mốc dưới.

**1e — ảnh mẫu cùng phong cách** (mẫu P1, thư mục = 1 crop DigitizePID / lớp, 28 lớp, 5 lần rút; ngẫu nhiên 0.036):

| encoder / đầu vào / CL2N | top-1 (TB ± std) | AUROC | cùng encoder, ảnh mẫu ISO |
|---|---|---|---|
| imagenet / thô / CL2N | 0.562 ± 0.050 | 0.729 | 0.131 |
| lo (R1a) / thô / thô | **0.781 ± 0.065** | 0.757 | 0.044 |
| dinov2 / A4 / CL2N | 0.815 ± 0.022 | 0.818 | 0.579 |

→ **Chẩn đoán "khác phong cách" được xác nhận trực tiếp**: cùng encoder, cùng crop truy vấn, chỉ đổi ảnh mẫu
từ ISO sang một crop DigitizePID, R1a nhảy 0.044 → 0.781. Train DigitizePID giúp so khớp lớp chưa thấy
**cùng phong cách** (R1a 0.78 > ImageNet 0.56), nhưng không vượt được khác phong cách. Đây là con số gần
protocol OSSR-PID nhất (cùng phong cách) nhưng trên lớp chưa thấy — vẫn ghi rõ khác protocol.

---

## 2. Bước 2 — Đưa phần thắng vào pipeline (chỉ khi Bước 1 ĐẠT; bắt đầu ≤ 2026-11-25)

### 2A — A4 + train lại encoder (~1 tuần: code 2–3 ngày + 2–3 phiên Colab)

- Khóa mới `++fgc.input_norm=ink_letterbox` (mặc định tắt, đọc bằng `.get`): áp **cùng một hàm**
  `utils.input_norm` ở dataset train + eval (`utils/dataset.py`), ảnh mẫu (`pid-iso-prototypes`),
  view truy vấn (`oneshot_infer.make_views`), `fgc_infer`. Checkpoint ghi `input_norm`; nạp encoder
  với tiền xử lý khác lúc train → lỗi (như guard encoder ↔ prototype hiện có).
- Train lại **R1a** trước (cùng epoch 3, cùng `AUG`, cùng H, cùng seed) → `__global_lo_a4__.pt`; R1b
  (`__global_a4__.pt`) nếu còn phiên Colab.
- Chạy lại notebook: §7 (prototype), §9 (τ trên `val_sel`, P1, P2), §10d–10e (CL2N), §11 (P4), §12
  (end-to-end) với cặp encoder mới; τ chọn lại theo đúng quy tắc 9a.

### 2B — DINOv2 làm encoder (không train, ~2–3 ngày)

- Bọc DINOv2 thành checkpoint dùng được bởi `load_arcface_encoder` (tương tự
  `scripts/make_imagenet_encoder.py`: `fine_classes=[]`, mọi lớp `excluded`), có A4 nếu 1c chọn A4.
- Chạy §9 τ (`groups_ckpt` = R1a), P1; §10 CL2N; §11 P4 với encoder mới.

**CODE ✅ 2026-11-16** (chưa chạy Colab):
- `oneshot.input_norm` (`++`, null = tắt; `utils.input_norm.INPUT_NORMS`): áp A4 cho ảnh mẫu (`pid-iso-prototypes`, ghi
  `meta.input_norm`) và mọi crop truy vấn (`OneShotClassifier.embed_crop` → mọi protocol eval, tâm CL2N, `fgc --infer`).
  Prototype / tâm / lượt chạy lệch `input_norm` → lỗi (`check_encoder_matches_prototypes`, `load_oneshot_classifier`).
- `utils.model.backbone_kwargs`: backbone ViT (`vit_*`, `deit*`, `eva*`) dựng với `img_size=fgc.imgsz`; CNN không đổi.
- `scripts/make_imagenet_encoder.py` dựng luôn checkpoint DINOv2 với `fgc.backbone=vit_small_patch14_dinov2.lvd142m
  fgc.imgsz=224 fgc.arcface.embed_dim=384`; checkpoint ghi `pretrained_only: True`, `imagenet_only: False`.
- Kiểm cục bộ: prototype A4 dựng bằng pipeline thật **trùng tuyệt đối** prototype của probe (sai khác 0); embedding của
  `OneShotClassifier` trùng probe (< 1e-6); thiếu `input_norm` lúc chạy → bị chặn. 800 test pass.
- Notebook §13 (RUNBOOK §5h): 13a encoder + prototype, 13b tâm → τ (`val_sel`) → P1 đủ + bảng cạnh R1a / ImageNet + CL2N,
  13c–13d P4 cho DINOv2 + A4 + CL2N **và hai dòng CL2N của §4** (ImageNet + CL2N, R1a + CL2N), 13e (tuỳ chọn) end-to-end.

### 2B — Kết quả Colab 2026-11-17 (§13; áp luật "Báo cáo Bước 2" đã chốt)

τ DINOv2 + A4 + CL2N trên `val_sel`: A 0.1979, B 0.4032 (J 0.470), δ 0.0146. Prototype: mean cos 0.634, max 0.970.

**P1 đủ (17,379 crop):**

| | R1a | ImageNet + CL2N | DINOv2 + A4 + CL2N |
|---|---|---|---|
| `h_in` macro top-1 / top-3 | 0.036 / 0.366 | 0.105 / 0.383 | **0.576 / 0.827** |
| AUROC cos / lề | 0.777 / 0.705 | 0.438 / 0.363 | **0.360 / 0.348** |
| OSCR | 0.030 | 0.054 | 0.205 |
| nhận + đúng tại τ_A | 0.037 | 0.097 | 0.514 |

Từng lớp top-1 (R1a → DINOv2): S3 0.07 → 0.41, S10 0 → 0.98, S5 0.15 → 0.77, S4 0 → 0.005, S18 0 → 0.88, S19 0.001 → 0.41.
Tăng ở 6/6 lớp so với R1a (đổi dấu chính xác p = 2/64 = 0.031, mức nhỏ nhất đạt được); 5/6 so với ImageNet + CL2N
(S4 giảm). **Theo luật: KHÔNG được gọi là "cải thiện" P1** — top-1 tăng ở ≥ 4/6 lớp nhưng AUROC giảm (0.78 → 0.36, dưới
0.5: lớp lạ có cos cao hơn lớp biết). Ghi: "gọi tên lớp chưa thấy tốt hơn hẳn; tách lạ kém hơn hẳn".
Mẫu P1 cục bộ (Bước 1) dự báo đúng: 0.579 / 0.829 / AUROC 0.351.

**P4 (suy giảm, nhóm theo R1a; bootstrap theo 12 tờ, 1000 lần):**

| chênh | giá trị | 95% CI |
|---|---|---|
| H_tu_nhien top-1 DINOv2 − R1a | +0.262 | [+0.197, +0.305] |
| H_tu_nhien top-1 DINOv2 − R1b | +0.147 | [+0.094, +0.205] |
| H_chon top-1 DINOv2 − R1a | +0.314 | [+0.268, +0.351] |
| V top-1 DINOv2 − R1a | +0.265 | [+0.230, +0.308] |
| V top-1 DINOv2 − R1b | −0.006 | [−0.027, +0.014] |
| S top-1 DINOv2 − R1a | **−0.146** | [−0.187, −0.119] |
| AUROC cos DINOv2 − R1a | +0.031 | [−0.013, +0.065] |
| AUROC lề DINOv2 − R1a | **−0.175** | [−0.210, −0.137] |

DINOv2 (suy giảm): S 0.847, V 0.436, H_chon 0.547, H_tu_nhien 0.646; AUROC cos 0.728, lề 0.703, OSCR 0.599. → P4 **cải
thiện** gọi tên ở V / H_chon / H_tu_nhien (CI không chứa 0); S kém hơn (flanged_connection 0.59 so với 1.00); tách lạ
bằng cos "không khác", bằng lề kém hơn. DINOv2 không mắc lỗi `flow_direction` → `pipeline` của R1b (0.996).
`other_isa` → vessel_spherical 96%, `other_generic` → tank 89%.
CL2N cho R1a / ImageNet trên P4: R1a + CL2N ≈ R1a (AUROC +0.010, H_tu_nhien −0.026); ImageNet + CL2N H_chon +0.165 nhưng
AUROC cos 0.369 (ngược chiều).

**End-to-end (suy giảm, detection `agnostic_merge`):** DINOv2 tại τ: P 0.429 / R 0.434 (R1a tại τ 0.377 / 0.381); theo nhóm
S 0.878, V 0.349, H_chon 0.371, H_tu_nhien 0.431 (R1a 0.876 / 0.128 / 0.129 / 0.209) nhưng `other_*` ≈ 0 (τ = 0.198 quá
thấp, không loại gì). τ + δ: P 0.442 / R 0.447. R1a τ + δ vẫn có P / R gộp cao nhất (0.619 / 0.625) nhờ flow_direction +
other_isa, với V / H ≈ 0 → không so P / R gộp, so theo nhóm.

**Đọc:** DINOv2 + A4 + CL2N là encoder **gọi tên** tốt nhất trên mọi đề thi có lớp chưa train (P1, P4 H, end-to-end H),
nhưng **cổng Other** của nó không dùng được (P1 AUROC < 0.5; e2e không loại gì). R1a vẫn là encoder **tách lạ** tốt nhất.
Đúng mô hình đánh đổi đã thấy ở P1 (Kornblith et al. 2021).

### Báo cáo Bước 2 (chốt trước)

- **Luôn báo cả bản cũ lẫn bản mới**, cạnh nhau, như ablation. Không thay bản cũ bằng bản mới chỉ
  vì bản mới tốt hơn trên P1 / P4 (đó là test).
- Chỉ nói "cải thiện" khi: P1 đủ tập — top-1 tăng ở ≥ 4/6 lớp `h_in` **và** AUROC không giảm; P4 —
  khoảng tin cậy bootstrap theo 12 tờ của chênh lệch không chứa 0. Ngược lại ghi "không khác".

## 2C. Hướng cải thiện tiếp — GHI TẠM 2026-11-17, CHƯA CHỐT, CHƯA CHẠY

Sau 2B: DINOv2 + A4 + CL2N gọi tên tốt nhất (P1 top-1 0.576, vững ở mức lớp), R1a tách lạ tốt nhất (P1 AUROC 0.777).
Chỗ yếu còn lại: cổng "Other" của DINOv2 (P1 AUROC 0.360), S4 / `flanged_connection`, khoảng cách phong cách (1e: một
crop thật thay ảnh mẫu ISO → 0.78–0.82). Trước khi chạy hướng nào phải chốt luật quyết định ở đây (như §1.2), không chọn
trên P1 / P4.

| # | Hướng | Nguồn | Chi phí | Kiểm trên | Ghi chú |
|---|---|---|---|---|---|
| C1 | **Ghép hai encoder**: R1a làm cổng (loại nếu cos_R1a < τ_A của R1a, chọn ở 9a), DINOv2 đặt tên | Fort, Ren & Lakshminarayanan 2021 (đặc trưng trong miền tách lạ gần tốt hơn); bài 2026 về DINOv2 nhận nhầm lạ gần | Vài giờ, từ CSV / cache có sẵn | Tính thẳng P1, P4 từ CSV | Luật ghép cố định, không chỉnh; ghi rõ là ý **sau khi xem test**. P1-val không kiểm công bằng cổng R1a (R1a đã train các lớp OOV của P1-val) |
| C2 | **Mahalanobis** thay cos: hiệp phương sai chung từ crop train lớp đã thấy, shrinkage; biến thể Relative Mahalanobis | Simple CNAPS (Bateni CVPR 2020, arXiv 1912.03432); Ren et al. 2021 (arXiv 2106.09022) | Vài giờ, cache probe | P1-val | Nối tiếp CL2N (CL2N = chỉ trừ trung bình) |
| C3 | **DINOv2 `reg4`** và **ViT-B/14** | Darcet et al. ICLR 2024 (arXiv 2309.16588): token nhiễu ở vùng nền | ~0.5–1 ngày nhúng lại probe | P1-val | Chỉ đổi tên model |
| C4 | **So khớp theo patch** (token patch DINOv2, láng giềng gần nhất crop → ảnh mẫu) | DN4 (Li CVPR 2019) | ~1 ngày code | P1-val | Bền với ống / chữ dính crop |
| C5 | **Few-shot với vài crop thật** (bộ nhớ đệm kiểu Tip-Adapter) | Tip-Adapter (ECCV 2022, arXiv 2207.09519) | Số đã có (1e) | — | Đổi đóng góp #1 thành few-shot; đưa vào mở rộng / thảo luận |

**Không làm vòng này**: fine-tune DINOv2; học biến đổi riêng prototype / ảnh (CoPA, NeurIPS 2024) — thiếu cặp ảnh mẫu ↔
crop thật, không kịp đóng băng. kNN-OOD (Sun et al. ICML 2022) không hợp: "lạ" của ta = không có ảnh mẫu, không phải
khác dữ liệu train.

**Đề xuất thứ tự** (chờ user): C1 + C2 trước (rẻ), rồi C3; C4 chỉ nếu còn trước 2026-11-25.

### 2C.0 Cơ sở của từng hướng tối ưu (user 2026-11-17: mọi cách tối ưu phải có cơ sở / khuyến nghị từ paper)

Mức cơ sở: **mạnh** = paper đo đúng loại thay đổi này trên bài toán gần (đặc trưng đóng băng, ít mẫu, khác miền);
**vừa** = paper đo trên bài toán khác nhưng cùng cơ chế; **yếu** = chỉ thực hành chuẩn / chẩn đoán nội bộ.

| Hướng | Cơ sở (paper → điều paper khẳng định) | Mức |
|---|---|---|
| CL2N (đã dùng) | SimpleShot (Wang et al. 2019, arXiv 1911.04623): trừ trung bình rồi L2 cải thiện nearest-centroid với đặc trưng pretrained | mạnh |
| DINOv2 đóng băng (đã dùng) | DINOv2 (Oquab et al., TMLR 2024, arXiv 2304.07193): đặc trưng dùng được "across image distributions and tasks without finetuning"; Bảng 6: ImageNet-Sketch ViT-S 41.2, ImageNet-R 53.7 (miền nét vẽ / tranh vẽ) | mạnh |
| A4 (đã dùng, trong biến thể thắng) | Chẩn đoán D2 nội bộ (BACKUP_PLAN: `Resize` bóp méo tỉ lệ 0.38–3.58, template sạch vs crop nhiễu); nhị phân hoá + chuẩn kích thước là thực hành chuẩn của nhận dạng ký hiệu / tài liệu. Chưa có paper đối chứng trực tiếp; bằng chứng là số P1-val (DINOv2 + CL2N 0.333 → 0.477 khi thêm A4) | yếu (paper) / có số đo nội bộ |
| C1 ghép cổng R1a | Fort, Ren & Lakshminarayanan (NeurIPS 2021): đặc trưng fine-tune trong miền tách near-OOD tốt hơn; Vaze et al. (ICLR 2022): phân loại đóng tốt ↔ open-set tốt | vừa |
| C2 Mahalanobis | Simple CNAPS (Bateni CVPR 2020, arXiv 1912.03432); Relative Mahalanobis (Ren et al. 2021, arXiv 2106.09022) | vừa (đã thử: KHÔNG ĐẠT) |
| C3a ViT-B/14 | DINOv2 Bảng 4 / 6 / 9: S → B tăng mọi đánh giá đặc trưng đóng băng — kNN ImageNet 79.0 → 82.1, **ImageNet-Sketch 41.2 → 50.6**, ImageNet-R 53.7 → 63.3, nhận dạng mức thể hiện (instance retrieval) tăng | **mạnh** |
| C3b registers (`reg4`) | Darcet et al. (ICLR 2024, arXiv 2309.16588): bỏ token nhiễu ở vùng nền, lợi chủ yếu cho đặc trưng **dày**; phân loại toàn ảnh gần như không đổi (DINOv2 linear 84.3 → 84.8) | yếu cho đặc trưng toàn ảnh (vừa nếu làm C4 theo patch) |
| C4 so khớp theo patch | DN4 (Li et al. CVPR 2019): image-to-class bằng descriptor cục bộ + kNN tốt hơn so khớp vector toàn ảnh khi ít mẫu | vừa |
| C5 few-shot crop thật | Tip-Adapter (Zhang et al. ECCV 2022, arXiv 2207.09519): cache key-value từ vài mẫu, không train; số 1e nội bộ | mạnh (đổi bài toán) |

Quy ước: hướng mức **yếu** không được mở thêm vòng chạy riêng chỉ vì nó; chỉ đi kèm hướng có cơ sở mạnh / vừa, hoặc báo như
ablation. C3b đang chạy cùng C3a (cùng script, chi phí nhỏ) — kết quả của nó chỉ được báo như ablation dù có ĐẠT.

### 2C.1 Luật cho C1 + C2 — CHỐT 2026-11-17, TRƯỚC KHI CHẠY (user duyệt "C1 và C2")

**C1 — ghép hai encoder (không có tham số tự do, không chọn gì):**
- Luật ghép cố định: crop bị gọi "Other" nếu `best_cos` của **R1a** < τ_A của R1a (0.4699, chọn ở 9a trên `val_sel`);
  không thì nhãn = top-1 của **DINOv2 + A4 + CL2N**. Điểm tách lạ của cả hệ = `best_cos` của R1a; đúng / sai tên = DINOv2.
- Tính từ CSV có sẵn (cùng crop: P1 khớp theo `path`, P4 cùng thứ tự `split` / `ann_id`): P1 đủ, P4 suy giảm và sạch.
  End-to-end **không** tính được cục bộ (cần `detections_oneshot` của hai encoder trên Drive) → để sau nếu C1 đáng làm.
- Đây là ý nảy ra **sau khi xem P1 / P4** → luôn báo như "kết hợp hậu kiểm", cạnh R1a và DINOv2 riêng lẻ; không thay số chính.
- Chỉ được nói "C1 tốt hơn DINOv2 một mình" về tách lạ khi: P1 — AUROC và OSCR tăng, khoảng tin cậy **theo lớp** của
  chênh lệch không chứa 0; P4 — khoảng tin cậy theo 12 tờ không chứa 0. Báo kèm nhận + đúng tại τ_A (cái giá của cổng).

**C2 — Mahalanobis thay cos (tập quyết định = P1-val, cache `runs/phase_c/probe/`):**
- Không gian: vector CL2N đã gộp view (như pipeline). Hiệp phương sai chung Σ = **Ledoit–Wolf** (shrinkage tự động, không
  tham số) trên vector CL2N của 570 crop train lớp R1a đã thấy (đúng tập crop của tâm CL2N). P = Σ⁻¹, W = Σ^(−1/2).
- Hai biến thể: **(a) cos đã làm trắng** cos(W·q, W·p); **(b) Mahalanobis** d_c = (q − p_c)ᵀ P (q − p_c), nhãn = argmin d_c,
  điểm tách lạ = **Relative Mahalanobis** (Ren et al. 2021): −(min_c d_c − d_0), d_0 = khoảng cách tới trung bình chung.
- Encoder: **DINOv2 / A4** (chính — encoder đang dùng) và ImageNet / thô (tham khảo). R1a không xét (P1-val là lớp R1a đã
  thấy). Mỗi encoder chọn biến thể (a) hoặc (b) có macro top-1 P1-val cao hơn.
- Luật = §1.2, mốc = cos + CL2N **cùng encoder** trên P1-val: ĐẠT nếu top-1 tăng ở ≥ 6/10 lớp biết và macro top-1 tăng ≥ 0.05;
  hoặc AUROC (điểm chính hoặc lề) tăng ≥ 0.05 mà macro top-1 giảm ≤ 0.01.
- **DINOv2 / A4 ĐẠT → đưa C2 vào pipeline** (file "làm trắng" như file tâm CL2N, chạy lại §13 trên Colab, bắt đầu ≤ 2026-11-25).
  Chỉ ImageNet ĐẠT → chỉ báo cáo. Mẫu P1 cục bộ chỉ báo tham khảo, không quyết định.

### 2C.2 Kết quả C1 + C2 — 2026-11-17 (`scripts/phase_c_c1c2.py` → `runs/phase_c/c1c2/results.json`)

**C1 (R1a làm cổng, DINOv2 đặt tên; hậu kiểm):**

| | đúng, không cổng | nhận + đúng | lạ bị loại | AUROC | OSCR |
|---|---|---|---|---|---|
| P1 — R1a | 0.037 | 0.037 | 0.467 | 0.777 | 0.030 |
| P1 — DINOv2 | 0.575 | 0.514 | 0.017 | 0.360 | 0.205 |
| P1 — **C1** | 0.575 | **0.574** | **0.467** | **0.777** | **0.427** |
| P4 suy giảm — DINOv2 | 0.790 | **0.786** | 0.022 | 0.728 | 0.599 |
| P4 suy giảm — C1 | 0.790 | 0.753 | 0.001 | 0.697 | 0.601 |
| P4 sạch — DINOv2 | 0.792 | 0.790 | 0.015 | 0.807 | 0.651 |
| P4 sạch — C1 | 0.792 | 0.752 | 0.000 | 0.716 | 0.597 |

C1 − DINOv2: P1 AUROC +0.417 (theo lớp [+0.093, +0.724]), OSCR +0.221 (theo lớp [−0.001, +0.460]), nhận + đúng +0.061
(theo lớp [+0.008, +0.112]). P4 suy giảm: AUROC −0.031 [−0.065, +0.013], OSCR +0.001 [−0.020, +0.032], nhận + đúng −0.033
[−0.053, −0.015] (nhóm S 0.760 so với 0.840); P4 sạch: AUROC −0.091, OSCR −0.054 (đều không chứa 0).
**Theo luật §2C.1: KHÔNG được nói "C1 tốt hơn"** — P1: AUROC đạt nhưng OSCR chạm 0 ở mức lớp; P4: kém hơn. → Cổng R1a có ích
với ký hiệu lạ kiểu DigitizePID (P1), vô ích với ký hiệu lạ kiểu ISO (P4: τ không chuyển miền, cắt 5.6% crop biết). Không đưa
vào pipeline; ghi vào luận văn như phân tích "cổng Other phụ thuộc miền".

**C2 (Mahalanobis, quyết định trên P1-val):** **KHÔNG ĐẠT** cả hai encoder.

| P1-val | cos + CL2N | cos đã làm trắng | Mahalanobis (RMD) |
|---|---|---|---|
| DINOv2 / A4: top-1 / AUROC | 0.477 / 0.692 | 0.098 / 0.556 | 0.052 / 0.572 |
| ImageNet / thô: top-1 / AUROC | 0.175 / 0.427 | 0.061 / 0.574 | 0.061 / 0.600 |

DINOv2 chọn "làm trắng": 2/10 lớp tăng, Δtop-1 −0.379. Có khả năng Σ ước lượng trên crop DigitizePID làm trắng khuếch đại đúng
các hướng ảnh mẫu ISO khác crop. (Tham khảo, không quyết định: trên mẫu P1, DINOv2 làm trắng 0.616 so với 0.579.)

**User 2026-11-17: "tạm thời không cần quá lo deadline làm ảnh hưởng quá trình tối ưu"** → mốc 2026-11-25 / 2026-12-15 ở §0
là mốc mềm; vẫn giữ nguyên tắc chốt luật trước khi chạy.

### 2C.3 Luật C3 — CHỐT 2026-11-17, TRƯỚC KHI CHẠY

- Ứng viên (timm, không train, A4 / CL2N như Bước 1): `vit_small_patch14_reg4_dinov2.lvd142m` (S + registers),
  `vit_base_patch14_dinov2.lvd142m` (B), `vit_base_patch14_reg4_dinov2.lvd142m` (B + registers). Chạy bằng
  `scripts/oneshot_probe.py` (cùng crop, cùng tâm), cache `runs/phase_c/probe/`.
- Mỗi ứng viên chọn biến thể tốt nhất trong {thô, A4} × {thô, CL2N} theo macro top-1 **P1-val** (như 1c).
- Mốc = **DINOv2-S / A4 / CL2N** (encoder hiện tại) trên P1-val. Luật §1.2: ĐẠT nếu top-1 tăng ở ≥ 6/10 lớp và macro top-1
  tăng ≥ 0.05; hoặc AUROC (cos hoặc lề) tăng ≥ 0.05 mà macro top-1 giảm ≤ 0.01.
- Nhiều ứng viên ĐẠT → lấy ứng viên có macro top-1 P1-val cao nhất. ĐẠT → thay encoder trong §13 (chỉ đổi `DINO`: backbone,
  `embed_dim` 384 / 768) và chạy lại §13 trên Colab, báo cạnh DINOv2-S. Mẫu P1 chỉ tham khảo.

### 2C.3b Kết quả C3 — 2026-11-17 (`scripts/oneshot_probe.py --score-only`, `runs/phase_c/probe/report.txt`, log `run_c3.log`)

**KHÔNG ỨNG VIÊN NÀO ĐẠT → giữ DINOv2-S / A4 / CL2N; §13 không đổi; H2 dùng DINOv2-S.**

| P1-val (quyết định) | top-1 | top-3 | AUROC | lề | hub | mẫu P1 top-1 / top-3 / AUROC (chỉ tham khảo) |
|---|---|---|---|---|---|---|
| **mốc** dinov2 (S) / A4 / CL2N | **0.477** | 0.637 | 0.692 | 0.561 | 0.158 | 0.579 / 0.829 / 0.351 |
| dinov2reg (S+reg) / A4 / thô ← tốt nhất, ablation | 0.429 | 0.647 | 0.797 | 0.628 | 0.200 | 0.362 / 0.824 / 0.373 |
| dinov2reg / A4 / CL2N | 0.374 | 0.675 | 0.822 | 0.523 | 0.120 | 0.496 / 0.871 / 0.414 |
| dinov2b (B) / A4 / thô ← tốt nhất | 0.457 | 0.516 | 0.413 | 0.598 | 0.231 | 0.447 / 0.690 / 0.405 |
| dinov2b / A4 / CL2N | 0.409 | 0.539 | 0.639 | 0.610 | 0.127 | 0.566 / 0.966 / 0.290 |
| dinov2breg (B+reg) / A4 / thô ← tốt nhất | 0.421 | 0.543 | 0.759 | 0.611 | 0.224 | 0.240 / 0.659 / 0.554 |
| dinov2breg / A4 / CL2N | 0.365 | 0.617 | 0.837 | 0.686 | 0.151 | 0.543 / 0.976 / 0.541 |

Áp luật (biến thể tốt nhất theo top-1 P1-val, so mốc): dinov2reg 4/10 lớp, Δtop-1 −0.048, ΔAUROC +0.105 → KHÔNG ĐẠT (nhánh
AUROC cần top-1 giảm ≤ 0.01); dinov2b 3/10, −0.020, +0.037 → KHÔNG ĐẠT; dinov2breg 2/10, −0.055, +0.067 → KHÔNG ĐẠT. Kiểm cả 12
biến thể: không biến thể nào qua nhánh nào (biến thể có ΔAUROC ≥ 0.05 đều giảm top-1 ≥ 0.048).

**Đọc**:
1. Encoder lớn hơn **không** gọi tên tốt hơn trên P1-val; điều DINOv2 Bảng 9 báo (ViT-S → ViT-B tăng ~9 điểm trên
   ImageNet-Sketch / -R) không chuyển sang ảnh mẫu ISO → crop P&ID. Khoảng cách phong cách ở đây không phải loại "ảnh tự nhiên
   → tranh vẽ" mà các bảng đó đo.
2. Registers và ViT-B làm **tách lạ tốt hơn** (AUROC P1-val tới 0.84 vs 0.69) nhưng phải trả bằng top-1 → không qua luật.
   Ghi như ablation: nếu sau này ưu tiên cổng Other, `dinov2breg / A4 / CL2N` là ứng viên — **phải chốt luật mới trước**.
3. Mẫu P1 (test) cho ViT-B + CL2N top-3 0.97–0.98 (S 0.83) nhưng top-1 không hơn và AUROC P1 không ổn định (0.29 / 0.54); P1-val
   và P1 xếp hạng biến thể khác nhau (P1-val có 4 lớp VARIANT). Không dùng để quyết định (§1.2); chỉ ghi để không giấu số.
4. 1e (ảnh mẫu = crop thật): mọi ứng viên 0.81–0.82, ngang DINOv2-S → khi hết khác phong cách, encoder không còn là nút thắt
   → củng cố lý do làm H1 / H2 (§2D) thay vì đổi encoder tiếp.

**Cổng "Other" lấy đi bao nhiêu** (đo 2026-11-17 từ CSV, số crop, không gộp nhóm):

| | đúng không cổng | nhận + đúng tại τ_A | lạ bị loại tại τ_A |
|---|---|---|---|
| P1, R1a | 0.037 | 0.037 | 0.467 |
| P1, DINOv2 + A4 + CL2N | 0.575 | 0.514 | 0.017 |
| P4 suy giảm, R1b | 0.217 | 0.216 | 0.005 |
| P4 suy giảm, R1a | 0.723 | 0.669 | 0.001 |
| P4 suy giảm, DINOv2 + A4 + CL2N | 0.790 | 0.786 | 0.022 |

Với DINOv2 trên P1, cổng chặn 13.1% crop gọi sai tên nhưng cũng chặn 10.7% crop gọi đúng → gần như không phân biệt.

## 2D. Hướng tiếp cận thay thế — GHI 2026-11-17 (user: "không chỉ tối ưu encoder")

**Vì sao cần**: C1–C4 đều chỉnh bên trong cùng khung "1 ảnh mẫu ISO → encoder đóng băng → cosine". Nút thắt đo được là
**khác phong cách vẽ** giữa ảnh mẫu ISO và bản vẽ (1e: chỉ đổi ảnh mẫu ISO → 1 crop thật, R1a 0.04 → 0.78; DINOv2 + A4 +
CL2N 0.58 → 0.82). Đổi encoder chỉ thu hẹp một phần khoảng cách; các hướng dưới tấn công thẳng chỗ tạo ra nó.

| # | Hướng | Cơ sở | Mức | Chi phí / rủi ro | Trạng thái |
|---|---|---|---|---|---|
| H1 | **Ảnh mẫu cùng phong cách bản vẽ** (legend của chính bản vẽ, hoặc vài crop thật) thay / bổ sung ảnh mẫu ISO | Hain et al., ECML PKDD 2025 ("Legend-Informed Symbol Recognition…", doi 10.1007/978-3-662-72243-5_23: prototype = ký hiệu legend, encoder train tương phản trên augment của legend, loại nếu không khớp; tóm tắt báo > 80% qua nhiều phong cách — chưa đọc bản đủ); Tip-Adapter (ECCV 2022); số 1e nội bộ | mạnh | Thấp (đã có số 1e). Hạn chế: DigitizePID / OPEN100 không có trang legend → mô phỏng bằng 1–k crop / lớp | Đưa vào luận văn như phương án thay thế đã đo; tuỳ chọn: đường k-shot (k = 1, 3, 5) từ cache probe |
| H2 | **Ảnh mẫu ISO "giống bản vẽ"** trước khi nhúng: mỗi ảnh mẫu → nhiều biến thể (đoạn ống ở cổng nối thật = **B9**, độ dày nét ±, suy giảm scan), prototype = trung bình | Domain randomization (Tobin et al., IROS 2017); B9 đã thiết kế (PLAN_STAGE2 "B9"); Hain 2025 dùng augment của mẫu làm view dương | vừa | Thấp–vừa, không train: code B9 (cổng từ `utils.iso_templates`, `ScanDegrade`) ~1–2 ngày | **Làm sau C3** — luật §2D.1 |
| H3 | **Train encoder căn hai miền** ảnh mẫu ↔ crop (loss căn miền kiểu zero-shot SBIR) | Domain-Smoothing Network (IJCAI 2021); khảo sát sketch (arXiv 2001.02600) | vừa | Cao: thiếu cặp (R1a chỉ ~10 lớp có ảnh mẫu ISO); lấy cặp từ OPEN100 sẽ dính test P4; 1–2 tuần | Chưa làm; chỉ khi user chấp nhận chi phí |
| H4 | Fine-tune DINOv2 trên DigitizePID | — (Kornblith et al. 2021 và chính R1a: train trên lớp đã thấy làm mất chuyển giao) | ngược | — | **Không làm** |
| H5 | Mô hình đa phương thức (VLM) đọc bản vẽ | arXiv 2607.19568 (P&ID → process graph bằng MLLM) | vừa | Rất cao, lệch phạm vi | **Không làm**; nhắc ở hướng tương lai |

### 2D.1 Luật H2 — CHỐT 2026-11-17, TRƯỚC KHI CODE / CHẠY

- Thiết kế = B9 (PLAN_STAGE2_ONESHOT "B9") + hai loại biến thể thêm: mỗi ảnh mẫu → `1 + K` view: bản sạch; K bản "bẩn",
  mỗi bản rút ngẫu nhiên (seed cố định): đoạn ống ở **cổng nối thật** (cạnh / vị trí từ `utils.iso_templates`, dày = nét đo
  được, dài ngẫu nhiên tới mép khung); độ dày nét ±1 px (giãn / co hình thái); `ScanDegrade(p=1)` dải hẹp. Template không có
  cổng: chỉ view sạch + độ dày + suy giảm, ghi log. K = 8 (cố định trước, không dò).
- Tiền xử lý / encoder như biến thể đang dùng (A4 áp **sau** khi sinh biến thể, giống crop); encoder = DINOv2-S / A4 / CL2N,
  hoặc ViT-B nếu C3 ĐẠT (chốt theo kết quả C3, trước khi chạy H2).
- 🧑 Duyệt contact sheet (mỗi ảnh mẫu: sạch + 3 bản bẩn) trước khi chấm, như B1 / B9.
- Quyết định trên **P1-val**, mốc = cùng encoder với 1 view sạch; luật §1.2 (≥ 6/10 lớp tăng và Δ macro top-1 ≥ +0.05; hoặc
  ΔAUROC ≥ +0.05 mà top-1 giảm ≤ 0.01). ĐẠT → đưa vào pipeline (`oneshot.template_views` mở rộng, như thiết kế B9) và chạy
  lại §13 (P1, P4, e2e). Báo cả P1 lẫn P4; nếu chỉ P4 tăng → có thể chỉ khớp cách bộ render vẽ ống, ghi rõ (cảnh báo B9).
- **Encoder chốt sau C3 (2026-11-17)**: C3 KHÔNG ĐẠT → H2 dùng **DINOv2-S / A4 / CL2N**; mốc = `dinov2/a4/cl2n` của probe (0.477).

**Code H2 — 2026-11-17** (đã chấm: §2D.2) (`src/utils/template_variants.py`, `scripts/h2_template_variants.py`,
`tests/test_template_variants.py`). Hằng số chốt trước khi chấm (không dò):
- K = 8, seed 0 (seed riêng từng ảnh mẫu = seed ⊕ crc32(tên), không phụ thuộc thứ tự).
- Lề = `CROP_MARGIN_FRAC` (10%) của hộp mực mỗi phía (= lề crop thật). Mỗi cổng có ống với xác suất 0.8 (≥ 1 cổng); trục ống =
  vị trí cổng SVG theo chiều ngang cạnh, ánh xạ qua hộp mực PNG; ống bắt đầu ở mực đầu tiên gặp khi đi từ cạnh vào (không gặp →
  vị trí cổng SVG); ống chạy ra quá hộp mực 0.3–1.0 × lề; dày = nét đo (`scan_degrade.stroke_width`).
- Nét ±1 px (kernel 2×2; không làm mảnh nét ≤ 3 px). `ScanDegrade(p=1)` dải hẹp: morph 0, blur 0–0.3, pixelate 0–0.3,
  lighting 0–0.1, nhiễu 0–6, JPEG 70–90.
- **Bỏ "lệch tâm"** của B9: A4 cắt lại theo mực nên dịch tâm không còn tác dụng.
- SVG chọn theo REG#; nhiều bản vẽ (X8075) → bản có tỉ lệ khung gần PNG nhất. `pipeline` (405) không có SVG → chỉ nét + suy giảm.
- Prototype = trung bình 9 embedding (sạch + 8 bẩn, mỗi cái qua A4) rồi chuẩn hoá; CL2N như cũ (tâm từ crop, không đổi).
  Kiểm khi chấm: embedding của view sạch phải trùng prototype probe (< 1e-4).
- Contact sheet: `runs/phase_c/h2/contact_sheet.png` (sạch | 3 bẩn | bẩn 1 sau A4) + `ports.csv`. user đã duyệt 2026-11-17.
  Lệnh chấm (chỉ sau khi duyệt): `.venv/bin/python scripts/h2_template_variants.py --out runs/phase_c/h2 --score --approved`.

### 2D.2 Kết quả H2 — 2026-11-17 (user duyệt contact sheet; `runs/phase_c/h2/report.txt`, `results.json`, `protos_h2.pt`)

**KHÔNG ĐẠT → không đưa vào pipeline; §13 không đổi.** Kiểm: view sạch trùng prototype probe (|diff| 3e-8).

| | P1-val top-1 | top-3 | AUROC | lề | hub | mẫu P1 top-1 / top-3 / AUROC (chỉ tham khảo) |
|---|---|---|---|---|---|---|
| mốc dinov2 / A4 / CL2N (1 view sạch) | **0.477** | 0.637 | 0.692 | 0.561 | 0.158 | 0.579 / 0.829 / 0.351 |
| H2 (sạch + 8 bẩn) / A4 / CL2N | 0.421 | 0.639 | 0.651 | 0.540 | 0.204 | 0.649 / 0.837 / 0.306 |
| mốc / A4 / thô | 0.433 | 0.552 | 0.625 | 0.688 | 0.307 | 0.204 / 0.332 / 0.250 |
| H2 / A4 / thô | 0.358 | 0.523 | 0.501 | 0.632 | 0.210 | 0.258 / 0.401 / 0.269 |

Luật §1.2: lớp tăng 3/10 (S1, S21, S22 — S1, S21 đã ≈ 1.0), Δtop-1 −0.055, ΔAUROC −0.022 → KHÔNG ĐẠT.
Từng lớp (CL2N): S24 `flow_direction` 1.000 → 0.727, S23 `piping_heated_cooled_insulated` 0.853 → 0.693, S17 `open_disc`
0.220 → 0.127, S20 `reducer` 0.513 → 0.433; S22 `hx_coil` 0.193 → 0.233; S6, S8, S14 vẫn 0.

**Đọc**:
1. Thêm ống / nét / suy giảm làm prototype "giống bản vẽ" hơn theo mắt nhưng không gần crop DigitizePID hơn trong không gian
   DINOv2: lớp mất nhiều nhất là các ký hiệu nhỏ / dạng đường (mũi tên, ống bọc) — ống thêm vào đổi hẳn tỉ lệ hình sau A4.
2. Mẫu P1 (test) top-1 tăng 0.579 → 0.649 nhưng không dùng để quyết định (§1.2); P1-val và P1 lại xếp hạng ngược nhau như ở C3
   → chênh lệch ở mức này nằm trong độ dao động giữa tập lớp, không phải tín hiệu.
3. K, xác suất ống, độ dài… đã chốt trước, **không dò lại** (dò trên P1-val rồi báo P1 là dò trên tập quyết định nhiều lần).
4. Cùng với C3: đổi encoder hay làm "bẩn" ảnh mẫu ISO đều không qua; chỉ ảnh mẫu là **crop thật** (1e: 0.82) vượt được khác
   phong cách → H1 là hướng còn lại có số; khung luận văn nghiêng về E2 + E3 (§3) với DINOv2 + A4 + CL2N là cải tiến đã đạt.


### 2D.3 Đường k-shot H1 — THIẾT KẾ CHỐT 2026-11-17, TRƯỚC KHI CHẠY (user chọn "Đường k-shot của H1")

Phân tích mô tả (không đưa gì vào pipeline, không có luật ĐẠT / KHÔNG ĐẠT): ảnh mẫu là crop thật thì cần bao nhiêu crop / lớp.
- Thư mục = 28 lớp (32 trừ `h_out`), mỗi lớp k crop DigitizePID từ ảnh **train**; prototype = trung bình k embedding crop (mỗi
  crop: các view `query_views` gộp mean), chuẩn hoá. k ∈ {1, 3, 5}, lồng nhau (k = 1 ⊂ 3 ⊂ 5, rút
  `Random("kshot/<seed>/<lớp>").sample(…, 5)`), 5 lần rút (seed 0–4).
- Truy vấn = mẫu P1 của probe (như 1e): `h_in` = biết (6 lớp, chấm top-1 / top-3 macro trên 28 ứng viên), `h_out` = lạ (AUROC
  cos); bỏ mọi crop truy vấn cùng tờ với 5 ảnh mẫu của lần rút đó (cùng tập truy vấn cho mọi k).
- Encoder × biến thể: R1a (`lo`), ImageNet, DINOv2-S × {thô, A4} × {thô, CL2N}; tâm CL2N như probe. Báo trung bình ± độ lệch
  chuẩn qua 5 lần rút; cạnh đó số P1 với ảnh mẫu ISO (k = 0, 42 ứng viên — không cùng tập ứng viên, chỉ làm mốc).
- Hạn chế ghi sẵn: crop ảnh mẫu và truy vấn cùng bộ DigitizePID (cùng phong cách, khác tờ) → đây là cận trên kiểu "legend của
  chính bộ bản vẽ" (Hain et al. 2025), không phải ảnh mẫu từ nguồn khác.
- Script `scripts/h1_kshot.py` → `runs/phase_c/h1/` (cache embedding crop ảnh mẫu; truy vấn + tâm lấy từ cache probe).

**Kết quả H1 — 2026-11-17** (`runs/phase_c/h1/report.txt`, `results.json`; 1144 truy vấn / lần rút, 6 lớp `h_in` biết, 4 lớp
`h_out` lạ; trung bình ± độ lệch chuẩn qua 5 lần rút):

| encoder / biến thể | ảnh mẫu ISO top-1 (42 ứng viên, tham khảo) | k = 1 top-1 | k = 3 | k = 5 | top-3 k = 5 | AUROC k = 1 → 5 |
|---|---|---|---|---|---|---|
| R1a / thô / thô | 0.044 | 0.776 ± 0.081 | 0.892 ± 0.059 | **0.903** ± 0.022 | 1.000 | 0.768 → 0.785 |
| ImageNet / A4 / thô | 0.079 | 0.623 ± 0.075 | 0.739 ± 0.085 | 0.791 ± 0.026 | 0.970 | 0.841 → **0.883** |
| ImageNet / thô / CL2N | 0.131 | 0.524 ± 0.070 | 0.651 ± 0.035 | 0.688 ± 0.014 | 0.949 | 0.721 → 0.736 |
| DINOv2-S / A4 / thô | 0.204 | **0.788** ± 0.035 | **0.892** ± 0.035 | 0.899 ± 0.016 | 0.997 | 0.785 → 0.868 |
| DINOv2-S / A4 / CL2N (encoder 2B) | 0.579 | 0.780 ± 0.037 | 0.884 ± 0.041 | 0.899 ± 0.026 | 0.999 | 0.790 → 0.856 |

(Bảng đủ 12 dòng trong `report.txt`.)

**Đọc**:
1. Chỉ đổi nguồn ảnh mẫu (ISO → crop cùng phong cách) là bước nhảy lớn nhất của cả Phase C: R1a 0.04 → 0.78 (k = 1) → 0.90
   (k = 5); DINOv2-S 0.58 → 0.78 → 0.90. Phần lớn lợi ích đến ở k = 3; từ 3 lên 5 thêm ≤ 0.05 và độ lệch giữa các lần rút giảm
   (R1a ± 0.08 → ± 0.02).
2. Khi ảnh mẫu cùng phong cách, **R1a ngang DINOv2** (0.90 vs 0.90; top-3 R1a 1.000) dù chưa từng thấy 6 lớp này → encoder
   train trên DigitizePID **có** chuyển giao sang lớp mới; cái nó không vượt được là khác phong cách ISO ↔ bản vẽ. Củng cố
   chẩn đoán 1e, C3, H2.
3. Cổng Other khi ảnh mẫu là crop: AUROC 0.78–0.88, mọi encoder > 0.5 (khác hẳn DINOv2 + ảnh mẫu ISO trên P1: 0.36). AUROC tăng
   theo k ở DINOv2 / ImageNet + A4, gần như đứng yên ở R1a.
4. CL2N gần như không đổi gì khi ảnh mẫu là crop (tâm và ảnh mẫu cùng miền) — CL2N chữa lệch miền ảnh mẫu ↔ crop, đúng như kỳ vọng.
5. Hạn chế: chỉ 6 lớp biết + 4 lớp lạ; ảnh mẫu và truy vấn cùng bộ DigitizePID (khác tờ) → cận trên kiểu "legend của chính bộ
   bản vẽ" (Hain et al. 2025), không phải chuyển giao sang bộ bản vẽ khác. Không có số P4 tương ứng (OPEN100 không có crop
   train riêng để làm ảnh mẫu mà không đụng test).

---

## 2E. Hướng có huấn luyện (fine-tune) — GHI TẠM 2026-11-18, CHƯA CHỐT, CHƯA CHẠY (user: "còn nhiều thời gian, tra thêm, như finetune")

**Vì sao xem lại**: H4 ("fine-tune DINOv2 trên DigitizePID", §2D) bị gạt vì sợ mất chuyển giao. H1 (§2D.3) cho thấy R1a — đã train
trên DigitizePID — vẫn chuyển giao sang 6 lớp chưa thấy (0.90 với 5 crop / lớp, ngang DINOv2). Cái không chuyển là **phong cách
ISO ↔ bản vẽ**. Vậy fine-tune chỉ đáng làm nếu nó **học ánh xạ ảnh mẫu ISO → crop bản vẽ**, không phải học thêm lớp DigitizePID.

**Ràng buộc dữ liệu (quyết định độ khả thi)**: cặp thật (ảnh mẫu ISO, crop DigitizePID) chỉ có ở 10 lớp R1a đã train có ảnh mẫu
ISO — đúng 10 lớp "biết" của P1-val (tập quyết định). Train trên cả 10 thì mất tập quyết định. Cặp nhân tạo: bộ render OPEN100-ISO
(V7 / V8) sinh được không giới hạn cặp (ảnh mẫu, ký hiệu trên tờ có ống / chữ / suy giảm) cho mọi lớp ISO — nhưng là phong cách
**của bộ render**, và P4 cũng do bộ render này vẽ (H2 đã thấy "bẩn kiểu render" không kéo ảnh mẫu về phía DigitizePID).

| # | Hướng | Cơ sở | Ý tưởng cho đề tài | Chi phí / rủi ro | Đánh giá |
|---|---|---|---|---|---|
| F0 | **Thử rẻ: căn chỉnh tuyến tính** trên đặc trưng DINOv2 đóng băng (học W để embedding ảnh mẫu ISO → trung tâm crop cùng lớp; Procrustes / ridge) | LP-FT (Kumar et al. ICLR 2022: probe tuyến tính trước, ít làm méo đặc trưng); CORAL (Sun et al. AAAI 2016) | Chạy trên cache probe, CPU vài phút; chia 10 lớp cặp thật thành 2 nửa (train 5 / đo 5, đổi chiều) + đo P1 tham khảo | Rất thấp. Chỉ 5 lớp để học W → dễ khớp quá | **Làm trước**: nếu ánh xạ tuyến tính học trên 5 lớp chuyển được sang 5 lớp kia thì F1 đáng làm; nếu không, F1 khó thành |
| F1 | **Train encoder "ảnh mẫu → crop"** (meta-task kiểu VPE / prototypical: prototype = ảnh mẫu, truy vấn = crop; hoặc tương phản có ảnh mẫu làm neo) | **VPE** (Kim et al. CVPR 2019 — đúng dạng bài toán: one-shot từ ảnh nguyên mẫu (biển báo, logo) sang ảnh thật, test trên lớp chưa thấy chỉ với ảnh nguyên mẫu); Hain et al. ECML PKDD 2025 (legend); ZS-SBIR; chống méo: LP-FT, WiSE-FT (Wortsman et al. CVPR 2022: nội suy trọng số gốc / fine-tune, +4–6 điểm dưới dịch chuyển phân phối) | Backbone DINOv2-S (LoRA / vài block cuối) hoặc R1a; cặp = cặp thật (theo nửa lớp như F0) + cặp render cho lớp ISO **không** thuộc P1 (`h_in`, `h_out`) và nhóm H của P4; test P1 = chỉ ảnh mẫu ISO như cũ → vẫn giữ "thêm ký hiệu = thêm 1 ảnh" | Cao: code ~1 tuần + Colab; ít lớp cặp thật; P4 S / V bị "dễ" nếu train bằng bộ render → chỉ báo P4 nhóm H | **Ứng viên chính** nếu F0 cho tín hiệu |
| F2 | **Tự giám sát thích nghi miền** (tiếp tục huấn luyện DINO trên ảnh không nhãn của cả hai miền: ảnh mẫu ISO + augment, crop DigitizePID lớp đã thấy, crop render) | STARTUP (Phoo & Hariharan ICLR 2021: tự huấn luyện trên dữ liệu không nhãn miền đích, ~+2.9 điểm 1-shot trên BSCD-FSL); PMF (Hu et al. CVPR 2022: tiền huấn luyện ngoài → meta-train → fine-tune) | Không cần cặp → tránh được ràng buộc dữ liệu; loại mọi crop thuộc lớp holdout / tờ test | Vừa–cao; lợi ích paper báo nhỏ; không có tín hiệu ghép cặp nên không chắc kéo hai miền lại | Phụ; chỉ nếu F1 vướng dữ liệu |
| F3 | **Fine-tune lúc suy luận trên tập hỗ trợ** (42 ảnh mẫu ISO + augment) | PMF giai đoạn 3; BSCD-FSL (Guo et al. ECCV 2020: dịch chuyển miền lớn → fine-tune đơn giản thắng meta-learning, nhưng với ảnh hỗ trợ **cùng miền đích**); Sketch3T (Sain et al. CVPR 2022: test-time training từ 1 sketch) | Rẻ, giữ tính mở rộng | Ảnh hỗ trợ là miền ISO → chỉ học tách ảnh mẫu với nhau, không học cầu sang bản vẽ; Han et al. CVPR 2024 (FSOD với DINOv2): fine-tune vài lớp DINOv2 không giúp | **Thấp**, không đề xuất |
| F4 | Fine-tune DINOv2 trên DigitizePID (kiểu R1a) + WiSE-FT | Kumar 2022, Wortsman 2022; H1 (R1a chuyển giao trong cùng phong cách) | Cải thiện chế độ H1 (vài crop làm ảnh mẫu) và cổng Other | Vừa; không đụng nút thắt ISO ↔ bản vẽ | Chỉ nếu luận văn chọn H1 làm chế độ chính |

Liên quan (dẫn trong tổng quan, không phải hướng thử): Jamieson, Elyan, Moreno-García, "Few-Shot Symbol Detection in Engineering
Drawings", Applied Artificial Intelligence 38(1), 2024 (doi 10.1080/08839514.2024.2406712) — few-shot với < 10, có khi 1, mẫu
**cùng bản vẽ** → cùng tinh thần H1; Su, Zhu, Gong, "Open Logo Detection Challenge", BMVC 2018 — dán logo mẫu vào ngữ cảnh làm dữ liệu
nhân tạo, và chỉ ra lệch ngoại hình mẫu ↔ ngữ cảnh làm giảm khái quát (giống kết quả H2).

**Đề xuất thứ tự**: F0 (vài giờ, CPU) → nếu đạt luật sẽ chốt trước → F1 trên Colab. F2 / F4 dự phòng. F3 không làm. Mọi hướng
có train phải chốt trước: tập quyết định (vì 10 lớp P1-val là toàn bộ cặp thật), lớp nào bị loại khỏi train, cách báo P4.

### 2E.1 Luật F0 — CHỐT 2026-11-18, TRƯỚC KHI CODE / CHẠY (user: "Hãy làm F0")

**Không nhúng lại gì.** Mọi thứ lấy từ cache Bước 1 `runs/phase_c/probe/emb_dinov2_a4.pt` (DINOv2-S, A4, 8 view / crop, 42 ảnh
mẫu). Không gian = CL2N như mốc: tâm `c` = trung bình view của crop "center" (lớp đã thấy, tờ **train**, 30 crop / lớp). Truy
vấn không đổi; **chỉ ảnh mẫu được biến đổi**. Mốc = dinov2 / A4 / CL2N (P1-val top-1 0.477, AUROC cos 0.692).

- `x_j = normalize(t_j − c)`: ảnh mẫu j trong không gian CL2N (j = 42 ảnh mẫu).
- `y_k = normalize(trung bình q)` với `q` = truy vấn CL2N (như `rank`) của 30 crop center lớp k: đích của lớp k. Cặp thật chỉ có
  ở 10 lớp P1VAL_KNOWN (ảnh mẫu khác nhau từng lớp); crop đích lấy từ tờ **train**, không chạm tờ `val_sel`.

| dòng | Cách | Cơ sở | Dùng nhãn? |
|---|---|---|---|
| **F0-a** | Tâm riêng từng miền: ảnh mẫu trừ `μ_T` (trung bình 42 ảnh mẫu thô) thay vì `c`; truy vấn vẫn trừ `c` | Khoảng cách phương thức (Liang et al., "Mind the Gap", NeurIPS 2022: dời embedding một miền theo hiệu hai trung bình); AdaBN (Li et al. 2016: thống kê riêng từng miền) | Không (chỉ thống kê) |
| **F0-b** | Hồi quy ridge phần dư: `x' = normalize(x + b + (x − x̄)A)`; `b` = trung bình `y_k − x_k`, `A` ridge trên phần dư đã trừ tâm (dạng đối ngẫu, nhân tuyến tính). λ ∈ {0.01, 0.1, 1, 10, 100, ∞ (A = 0)} chọn **trong** tập lớp train bằng leave-one-class-out lồng (tiêu chí: cos trung bình giữa `x'` dự đoán và `y`) — không nhìn P1-val | LP-FT (Kumar et al. ICLR 2022: tầng tuyến tính trên đặc trưng đóng băng, gần đặc trưng gốc); ridge kéo về đồng nhất = không biến đổi khi thiếu dữ liệu | Có (cặp thật) |

**Đánh giá F0-b = leave-one-class-out (LOCO) trên 10 lớp**: lần k học trên 9 lớp còn lại, biến đổi **cả 42** ảnh mẫu (9 lớp
train bị kéo về crop của chúng → cạnh tranh mạnh hơn, như khi triển khai), chấm crop P1-val của lớp k. Top-1 lớp k lấy từ lần k →
macro top-1 / số lớp tăng so được trực tiếp với mốc. AUROC = trung bình 10 lần (biết = lớp k, lạ = 7 lớp OOV), mốc tính **đúng cùng
cách** (trung bình 10 AUROC "lớp k vs 7 OOV" với ảnh mẫu không biến đổi). F0-a không học nên chấm thẳng P1-val như mọi dòng cũ.

**Luật quyết định**: §1.2 nguyên văn (≥ 6/10 lớp tăng và Δ macro top-1 ≥ +0.05; hoặc ΔAUROC cos/lề ≥ +0.05 mà top-1 không giảm quá
0.01), từng dòng F0-a, F0-b so với mốc.
- **F0-b ĐẠT** → ánh xạ học từ cặp thật chuyển được sang lớp chưa thấy → đề xuất **F1** (viết luật F1 riêng, chờ user duyệt).
- **F0-a ĐẠT** → ứng viên không cần train cho pipeline (như 2B: cần P1 đủ + P4 trên Colab, chờ user duyệt).
- **Không dòng nào ĐẠT** → F1 **không** là bước tiếp theo được đề xuất (ánh xạ tuyến tính đã không chuyển; F1 phi tuyến với ~10 lớp
  cặp thật khó hơn, không loại hẳn); củng cố khung E2 + E3 (§3).

**Chỉ báo cáo, không quyết định**: (i) F0-b chỉ dịch `b` (λ = ∞); (ii) F0-b học trên 5 lớp (5 cách chia ngẫu nhiên 5 / 5 cố định
`Random("f0/split/<s>")`, s = 0..4, mỗi cách 2 chiều; top-1 lớp = trung bình các lần nó bị giữ ra) → độ nhạy theo số lớp train;
(iii) mẫu P1 (h_in biết, h_out lạ) với F0-a và F0-b học trên **cả 10** lớp — đây là lớp holdout thật, nhưng P1 không tham gia
quyết định; (iv) λ được chọn mỗi lần.

Code: `scripts/f0_linear_align.py` (chỉ đọc cache, ghi `runs/phase_c/f0/`), hàm tính toán trong `src/utils/linear_align.py` + test.

### 2E.2 Kết quả F0 — 2026-11-18 (`runs/phase_c/f0/report.txt`, `results.json`; CPU vài giây)

Kiểm: mốc tái lập đúng probe (P1-val 0.477 / AUROC 0.692; mẫu P1 0.579), sai khác 0. Mỗi lớp có 30 crop đích từ tờ train.

| dòng (P1-val) | top-1 | top-3 | AUROC cos | AUROC lề | lớp tăng | Luật §1.2 |
|---|---|---|---|---|---|---|
| mốc dinov2 / A4 / CL2N | 0.477 | 0.637 | 0.692 | 0.561 | — | — |
| **F0-a** tâm riêng miền | 0.474 | 0.635 | **0.619** (−0.073) | 0.620 (+0.058) | 2/10 | **ĐẠT** (nhánh AUROC, nhờ **lề**) |
| mốc theo cách LOCO | 0.477 | 0.637 | 0.692 | 0.561 | — | — |
| **F0-b** ridge, LOCO | 0.416 | 0.579 | 0.845 (+0.153) | 0.661 | 1/10 | **KHÔNG ĐẠT** (top-1 −0.061) |
| F0-b chỉ dịch `b` (báo cáo) | 0.477 | 0.637 | 0.813 (+0.121) | 0.627 | — | không quyết định |
| F0-b 5 lớp train (báo cáo) | 0.462 | — | — | — | — | không quyết định |

Từng lớp: S6, S8 = 0 và S14 ≈ 0.01 ở **mọi** dòng. F0-a: S17 0.22 → 0.48, S20 0.51 → 0.67, nhưng S24 1.00 → 0.81, S21 0.99 → 0.85,
S23 0.85 → 0.75. F0-b: S20 0.51 → **0.00** (lần đó λ lồng chọn 0.01, khớp quá); 9 lần còn lại chọn λ = 1.

Mẫu P1 (lớp holdout thật, **không quyết định**):

| dòng | top-1 | top-3 | AUROC cos | AUROC lề |
|---|---|---|---|---|
| mốc | 0.579 | 0.829 | 0.351 | 0.345 |
| F0-a | 0.717 | 0.933 | 0.441 | 0.334 |
| F0-b học trên cả 10 lớp (λ = 1) | 0.561 | 0.913 | 0.841 | 0.435 |

**Đọc kết quả** (theo luật đã chốt, không sửa luật):
- **F0-a ĐẠT theo chữ của luật** nhưng chỉ nhờ AUROC **lề**; AUROC **cos giảm 0.073**, top-1 không đổi. Cổng "Other" mặc định của
  pipeline dùng cos (`best_cos < τ`; lề chỉ là tùy chọn `++eval.oneshot.margin_delta`) → ở cấu hình mặc định F0-a làm cổng **kém đi**.
  Theo §2E.1, F0-a thành ứng viên cho pipeline (P1 đủ + P4 trên Colab) **chờ user duyệt**, báo kèm cảnh báo này.
- **F0-b KHÔNG ĐẠT** → theo §2E.1, F1 **không** là bước tiếp theo được đề xuất.
- Quan sát (không quyết định, xem sau khi có số → mọi bước tiếp phải chốt luật mới và đo trên dữ liệu chưa dùng cho F0): cặp thật
  mang tín hiệu cho **tách lạ** (AUROC cos +0.12 đến +0.15 trên P1-val; mẫu P1 0.351 → 0.841 — đúng điểm yếu của 2B "P1 AUROC
  0.360"), **không** cho gọi tên: S6 / S8 / S14 không nhúc nhích dưới mọi phép biến đổi tuyến tính → nhầm lớp không phải lệch tuyến tính.

### 2E.3 Luật kiểm lại F0 trên Colab (P1 đủ + P4) — CHỐT 2026-11-18, TRƯỚC KHI CODE / CHẠY (user chọn "Kiểm lại F0-a và F0-b chỉ dịch")

**Hai dòng**, đều trên DINOv2-S + A4 + CL2N của 2B (encoder, 42 prototype, tâm `center_dinov2_a4.pt` của 13b — **không dựng lại**);
chỉ ảnh mẫu đổi, truy vấn không đổi:
- `f0a`: `++oneshot.template_align=domain_center` — ảnh mẫu = normalize(t − μ_T), μ_T = trung bình 42 ảnh mẫu thô (đúng §2E.1).
- `shift`: `++oneshot.template_align=pair_shift ++oneshot.template_pairs=<file>` — ảnh mẫu = normalize(normalize(t − c) + b),
  b = TB(y_k − x_k) trên **cả 10** lớp cặp thật (P1VAL_KNOWN, 30 crop tờ train / lớp). File cặp (dựng cục bộ từ cache probe) lưu
  embedding **thô** (ảnh mẫu + mọi view crop) → b tính lúc nạp với **chính tâm Colab** c; ảnh mẫu trong file phải khớp prototype
  Colab (sai khác tuyệt đối ≤ 1e-3, code chặn). Không có λ, không chọn gì.

**Chạy** (notebook §14): τ trên `val_sel` (luật A / B, δ, như 13b) → P1 đủ → P4 (suy giảm = số chính). Không chọn gì trên P1 / P4.
**Mốc** = lượt 2B (`oneshot_p1_crops_dinov2.csv`, `oneshot_p4_crops_dinov2.csv`, đã có cục bộ). So sánh + bootstrap chạy cục bộ
(`scripts/f0_colab_compare.py`, đọc CSV từng crop, như C1).

**Luật** (mỗi dòng so với mốc):
- **P1 đủ — "cải thiện tách lạ"** nếu ΔAUROC **cos** ≥ +0.05 **và** Δ macro top-1 `h_in` ≥ −0.01 (cos vì cổng mặc định dùng cos).
  "Cải thiện gọi tên" theo luật "Báo cáo Bước 2" (≥ 4/6 lớp `h_in` tăng top-1 và AUROC cos không giảm). Kèm CI bootstrap theo ảnh
  (1000 lần) của ΔAUROC, chỉ báo cáo.
- **P4** (suy giảm; bootstrap theo 12 tờ, 1000 lần): chênh top-1 từng nhóm S / V / H_chon / H_tu_nhien và AUROC cos / lề:
  "tốt hơn" / "kém hơn" khi CI 95% nằm hẳn trên / dưới 0, ngược lại "không khác".
- **Thay 2B làm cấu hình chính** chỉ khi: P1 "cải thiện tách lạ" **và** P4 AUROC cos không "kém hơn" **và** không nhóm P4 nào "kém
  hơn" về top-1. Cả hai dòng cùng đạt → chọn `f0a` (không cần nhãn, là dòng đã ĐẠT luật F0). Không đạt → ghi ablation, 2B giữ nguyên.
- End-to-end: không làm trong vòng này.

**Giới hạn ghi trước**: mẫu P1 của Bước 1 (một phần P1 đủ) đã được xem ở F0 → P1 đủ không hoàn toàn mới; P4 chưa dùng cho F0.
`shift` học từ crop DigitizePID → trên P4 (phong cách bộ render) là phép dịch "khác miền"; báo đúng như vậy.

**Code ✅ 2026-11-18** (chưa chạy Colab):
- `oneshot.template_align` (`++`, null = tắt, cần `oneshot.center`): `domain_center` | `pair_shift` (+ `++oneshot.template_pairs`).
  Hàm ở `utils.linear_align` (`align_templates`, `pair_shift`, `save_pairs` / `load_pairs`); nạp ở `load_oneshot_classifier` nên
  mọi protocol eval và `fgc --infer` đều dùng được. `metrics.json` ghi `template_align` (kind, lớp cặp, `shift_norm`, sai khác ảnh mẫu).
- File cặp: `scripts/f0_linear_align.py` ghi `runs/phase_c/f0/pairs_dinov2_a4.pt` (10 ảnh mẫu thô + 30 × 8 view thô / lớp, các ann id).
- Kiểm cục bộ: classifier thật (`__dinov2__.pt`, prototype A4, tâm của probe) cho prototype **trùng tuyệt đối** F0 cục bộ ở cả hai
  dòng (sai khác 0; `shift_norm` 0.618); `f0_colab_compare.py` chạy trên CSV 2B tái lập đúng số 2B (P1 0.576 / AUROC 0.360; P4 S / V
  / H_chon / H_tu_nhien 0.847 / 0.436 / 0.547 / 0.646, AUROC cos 0.728, lề 0.703).
- Notebook §14 (RUNBOOK §5i): 14a τ + P1, 14b P4, 14c bảng.

**Dự báo ghi trước khi có số Colab** (mẫu P1 cục bộ, cùng tâm probe): `f0a` top-1 0.717 / AUROC cos 0.441; `shift` (b học trên
cả 10 lớp) top-1 0.623 / top-3 0.914 / AUROC cos 0.660 / lề 0.380 (mốc 0.579 / 0.829 / 0.351 / 0.345). Nếu P1 đủ giống mẫu:
`shift` đạt "cải thiện tách lạ", `f0a` cũng đạt về cos (+0.09) dù trên P1-val cos của nó giảm — hai tập ngược chiều nhau.

## 3. Bước 3 — Đóng băng và viết (2026-12-15, dù Bước 2 ra sao)

Khung luận văn theo kết quả (BACKUP_PLAN nhóm E):
- Bước 2 cải thiện thật → giữ khung chính, A4 / DINOv2 là đóng góp phụ, mọi số cũ thành ablation.
- Không cải thiện → **E2 + E3**: nghiên cứu thực nghiệm "one-shot từ ảnh mẫu chuyển giao tới đâu":
  được khi cùng phong cách (P4), không vượt khác phong cách (P1, kèm 1e làm bằng chứng trực tiếp);
  cổng "Other" chỉ dùng được trong cùng miền; khẳng định thu hẹp về các nhóm đã đo.

---

## 4. Làm song song (không phụ thuộc Bước 1, không đổi kết luận chính)

- **Điểm 4 (`pipeline`)**: quyết dựa trên **P2** (bằng chứng có từ 2026-11-13: S24 → pipeline). Nếu bỏ,
  chạy lại P2 trước; P4 khi đó chỉ là ablation **hậu kiểm** (đã xem P4), ghi rõ.
- **P4 thêm 2 dòng CL2N** (ImageNet + CL2N, R1a + CL2N): `++oneshot.center=encoders/center_<tag>.pt`
  ở §11; tâm đã có trên Drive.
- **Báo cáo theo nhóm / macro theo lớp**; P/R end-to-end gộp chỉ kèm phân rã theo nhóm.

## 5. Không làm trong vòng này (và lý do)

| Hướng | Lý do |
|---|---|
| A3 train theo episode | Episode chéo phong cách cần lớp train có ảnh mẫu ISO; R1a chỉ còn 4 lớp như vậy |
| D1 thêm dữ liệu ASU | License chưa rõ, map lớp tốn công, F4 chưa đo |
| B1 graph (OSSR-PID), B3 primitive + luật, A2 contrastive | Đắt, không kịp trước đóng băng |
| B9 prototype có stub (d) | Vẫn hoãn; không nhắm vào nghi phạm chính |
| Hiệu chỉnh τ trên OPEN100 | Cả 12 tờ đã là test và đã xem → ghi là giới hạn |

## 6. Checklist

- [x] 1.0 `utils/input_norm.py` + test; `scripts/oneshot_probe.py` + test; CSV danh sách crop P1-val (2026-11-16)
- [x] 1a mốc tái lập (mẫu P1: ImageNet + CL2N 0.131, R1a 0.044 — khớp)
- [x] 1b A4 suy luận — KHÔNG ĐẠT
- [x] 1c DINOv2 (thô / + CL2N, có / không A4) — ĐẠT (A4 + CL2N)
- [x] 1d HOG + A4
- [x] 1e ảnh mẫu cùng phong cách (5 seed) — xác nhận chẩn đoán
- [x] Áp luật 1.2, ghi quyết định + bảng (§1.3, HANDOFF) — → Bước 2B
- [x] Bước 2B code (2026-11-16): `oneshot.input_norm`, ViT backbone, checkpoint DINOv2, notebook §13
- [x] Bước 2B chạy Colab §13 (2026-11-17): P1 đủ, P4, e2e — kết quả §2B; 2A không làm
- [x] C1 / C2 (2026-11-17) — C1 không được gọi là tốt hơn (P4 kém), C2 KHÔNG ĐẠT (§2C.2)
- [x] C3 (2026-11-17) — KHÔNG ĐẠT, giữ DINOv2-S (§2C.3b)
- [x] H2 ảnh mẫu ISO "giống bản vẽ" (2026-11-17): user duyệt contact sheet → KHÔNG ĐẠT (§2D.2)
- [x] H1 đường k-shot (2026-11-17): k = 1 / 3 / 5 với crop thật → top-1 0.78 / 0.89 / 0.90 (§2D.3)
- [x] F0 căn chỉnh tuyến tính (2026-11-18): F0-a ĐẠT nhờ AUROC lề (cos giảm), F0-b KHÔNG ĐẠT (§2E.2); chờ user chọn bước tiếp
- [x] Luật + code kiểm lại F0 trên Colab (2026-11-18, §2E.3): `oneshot.template_align`, file cặp, notebook §14, script so sánh
- [ ] Chạy Colab §14 (user) → `scripts/f0_colab_compare.py` → áp luật §2E.3
- [ ] Song song: Điểm 4 trên P2 (chờ user); P4 + CL2N → đã gộp vào §13c (chạy cùng 2B)
- [ ] Đóng băng 2026-12-15

## 7. Nhật ký sửa luật

- **2026-11-17, trước khi có số C3** (lần chạy C3 đang nhúng `dinov2reg`, chưa chấm gì): sau yêu cầu của user "tối ưu phải
  có cơ sở từ paper" (§2C.0), ứng viên được **đưa vào pipeline** ở §2C.3 chỉ còn ViT-B (`dinov2b`, `dinov2breg` — cơ sở
  mạnh, DINOv2 Bảng 4 / 6 / 9). `dinov2reg` (ViT-S + registers; cơ sở yếu cho đặc trưng toàn ảnh, Darcet et al.) chỉ báo
  như ablation, kể cả khi ĐẠT. Ngưỡng của luật §1.2 không đổi.
