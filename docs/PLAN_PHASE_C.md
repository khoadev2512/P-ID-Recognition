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
   và P1 xếp hạng biến thể khác nhau (P1-val có 6 lớp VARIANT — sửa 2026-11-18, trước ghi nhầm "4"; xem §2F). Không dùng để
   quyết định (§1.2); chỉ ghi để không giấu số.
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

**14a Colab — 2026-11-18, P1 đủ (CHƯA QUYẾT: thiếu P4)** (`runs/phase_c/f0_colab/report.txt`). Kiểm: `max_template_diff` 6.8e-7,
`shift_norm` 0.621 (cục bộ 0.618), cùng 17,379 crop với 2B. τ luật A: f0a 0.310, shift 0.387 (2B 0.198).

| P1 đủ | top-1 | top-3 | AUROC cos (Δ, CI theo ảnh) | AUROC lề | lớp tăng | tách lạ | gọi tên |
|---|---|---|---|---|---|---|---|
| 2B | 0.576 | 0.827 | 0.360 | 0.348 | — | — | — |
| f0a | 0.696 | 0.927 | 0.443 (+0.083, [+0.078, +0.088]) | 0.354 | 4/6 | đạt | đạt |
| shift | 0.624 | 0.905 | 0.659 (+0.299, [+0.291, +0.308]) | 0.400 | 4/6 | đạt | đạt |

Khớp dự báo ghi trước (mẫu P1: f0a 0.717 / 0.441, shift 0.623 / 0.660). Lớp giảm ở cả hai: S18 (0.875 → 0.614 / 0.484), S10 ≈. f0a
AUROC vẫn < 0.5. CI theo ảnh hẹp vì 17k crop; chỉ 6 lớp `h_in` → không suy ra gì theo lớp.

### 2E.4 Kết quả F0 Colab đủ (P1 + P4) — 2026-11-18, áp luật §2E.3 (`runs/phase_c/f0_colab/report.txt`, `results.json`)

P4 suy giảm (số chính), chênh so với 2B, CI 95% bootstrap theo 12 tờ:

| P4 suy giảm | 2B | f0a | Δ f0a [CI] | shift | Δ shift [CI] |
|---|---|---|---|---|---|
| S top-1 | 0.847 | 0.662 | **−0.185 [−0.237, −0.138] kém hơn** | 0.833 | **−0.014 [−0.024, −0.002] kém hơn** |
| V top-1 | 0.436 | 0.533 | +0.097 [+0.080, +0.121] tốt hơn | 0.579 | +0.143 [+0.130, +0.162] tốt hơn |
| H_chon top-1 | 0.547 | 0.557 | +0.010 [−0.020, +0.044] không khác | 0.521 | −0.026 [−0.052, +0.007] không khác |
| H_tu_nhien top-1 | 0.646 | 0.628 | −0.018 [−0.045, +0.010] không khác | 0.616 | **−0.030 [−0.058, −0.006] kém hơn** |
| AUROC cos | 0.728 | 0.776 | +0.048 [+0.013, +0.084] tốt hơn | 0.848 | +0.119 [+0.079, +0.165] tốt hơn |
| AUROC lề | 0.703 | 0.483 | −0.219 kém hơn | 0.645 | −0.057 kém hơn |

**Quyết định theo luật: KHÔNG dòng nào thay 2B** — f0a: nhóm S kém hơn (−0.185); shift: nhóm S (−0.014) và H_tu_nhien (−0.030)
kém hơn. 2B giữ nguyên; f0a / shift ghi là **ablation**.

**Đọc** (không đổi quyết định):
- `shift` là **đánh đổi**: tách lạ tăng lớn và nhất quán (P1 AUROC 0.36 → 0.66, P4 0.73 → 0.85, cả hai CI không chứa 0), V tăng
  (+0.14), đổi lấy mất nhỏ nhưng có ý nghĩa ở S (−0.014) và H_tu_nhien (−0.030). Phép dịch học từ DigitizePID vẫn giúp tách lạ trên
  P4 (phong cách bộ render) — trái với lo ngại "khác miền" ghi trước.
- `f0a`: gọi tên P1 tốt nhất (0.70) nhưng phá nhóm S trên P4 (flanged_connection / flow_direction / valve_gate là lớp ISO "dễ") và
  AUROC lề sụp.
- Mọi ý tưởng tiếp theo trên P1 / P4 (ví dụ: cổng Other theo `shift` + tên theo 2B, kiểu C1) đều là **hậu kiểm** — hai tập đã xem;
  chỉ ghi là hướng mở, không gọi là cải thiện.

## 2F. Phân tích lỗi S6 / S8 / S14 — 2026-11-18 (mô tả, không quyết định; user chọn "việc 1")

`scripts/variant_contact_sheet.py` → `runs/phase_c/variants/variant_sheet.png`, `stats.json`. DINOv2-S / A4 / CL2N, crop từ tờ
**train** (30 crop "center" / lớp của probe; không dùng thêm P1-val). Mỗi hàng: ảnh mẫu ánh xạ | 5 crop | 3 ảnh mẫu hay bị đoán nhất.

| lớp | trạng thái `symbol_map` | top-1 (30 crop train) | hạng TV ảnh mẫu đúng / 42 | đoán nhiều nhất |
|---|---|---|---|---|
| S6 → valve_butterfly | VARIANT | 0.00 | 11 | check_valve_swing 37%, flow_direction 37% |
| S8 → check_valve_general | VARIANT | 0.00 | 12 | reducer 57%, valve_control 27% |
| S14 → valve_control | VARIANT (user xác nhận 2026-10-25) | 0.00 | 13 | valve_gate 77% |
| S22 → hx_coil | VARIANT | 0.23 | 3 | hose 33%, hx_coil 23% |
| S20 → reducer | VARIANT | 0.47 | 2 | reducer 47%, tank 33% |
| S23 → piping_heated_cooled_insulated | VARIANT | 0.90 | 1 | đúng 90% |
| S17 → open_disc | ISO (ghi chú: vẽ gắn trên cặp mặt bích) | 0.07 | 3 | motor_electric 53%, flanged_connection 40% |
| S1 → valve_gate (đối chứng) | ISO | 0.97 | 1 | đúng 97% |

**Đọc (về hình, không về nghĩa — nghĩa do user xác nhận):**
- S6 / S8 / S14 đều là **VARIANT**: `symbol_map` đã ghi DigitizePID vẽ theo chuẩn khác ISO (S6 hình chữ Z + chấm; S8 một tam giác +
  vạch; S14 van + bộ truyền động màng). Ảnh mẫu ISO tương ứng là **hình khác hẳn** → không phải lỗi encoder; khớp theo ảnh mẫu ISO
  không thể đúng dù encoder nào. Các đoán sai giải thích được bằng hình: S8 → tam giác của reducer, S14 → bow-tie của valve_gate,
  S6 → chấm tròn của check_valve_swing.
- **Ký hiệu ghép**: S14 (van + bộ truyền động) và S17 (đĩa trên cặp mặt bích) — ảnh mẫu chỉ có một phần của hình trên bản vẽ.
- **Hệ quả phương pháp** (sau khi có số, không đổi quyết định cũ): P1-val (tập quyết định) có **6/10 lớp VARIANT**, còn P1 test
  (`h_in`) **toàn ISO**. Tách P1-val theo trạng thái, mốc dinov2 / A4 / CL2N: ISO(4) 0.800, VARIANT(6) 0.261; mọi encoder không train
  trên DigitizePID có VARIANT ≤ 0.33. Luật "≥ 6/10 lớp tăng" vì thế gần như đòi tăng ở lớp VARIANT. Nhưng VARIANT **không** giải
  thích hết độ lệch P1-val ↔ P1: ngay phần ISO(4) cũng không dự báo P1 (H2: ISO 0.80 → 0.71 trong khi mẫu P1 0.58 → 0.65), vì 4 lớp
  ISO của P1-val (cổng, đĩa, mặt bích, mũi tên) khác loại 6 lớp van / đĩa của P1. Ghi vào luận văn như giới hạn của tập quyết định.

**User trả lời 2026-11-18** (ảnh `runs/phase_c/variants/q1_s6_s8.png`, `q2_s8_reducer.png`, `q3_composite.png`):
1. S6 = van bướm, S8 = van một chiều: **ánh xạ nghĩa đúng**. "Chuẩn khác" = **dạng hình khác** (thù hình khác), không phải hình
   gần giống; chỉ **nghĩa** là như vậy → ghi chú nghĩa, còn khi nhận dạng theo ảnh mẫu ISO thì nên đánh là **Other** (user đề xuất).
2. S8 (tam giác + vạch) **không** nhầm được với reducer trên bản vẽ thật → đoán "reducer" của model là lỗi thuần hình dạng.
3. **511 (flanged_connection) ghép C0098 (open_disc) = S17** trên DigitizePID.
(S14: bộ ký hiệu ISO của dự án không có bộ truyền động → không ghép được.)
4. (lượt sau) S20, S22, S23 **cũng khác dạng hình** → mọi lớp VARIANT = "cùng nghĩa, khác dạng hình". S18 / S19 trên DigitizePID
   = **511 ghép 2044 / 2045** (đúng như ghi chú `symbol_map`).

**Train trên dạng VARIANT có giúp nhận dạng ISO cùng nghĩa không? — KHÔNG** (P4 suy giảm, CSV cục bộ, mô tả; nhóm V = 4 ký
hiệu ISO mà R1a học qua bản vẽ VARIANT). ImageNet (điểm xuất phát của R1a) → R1a: valve_butterfly 0.458 → **0.000**,
check_valve_general 0.362 → **0.052**, reducer 0 → 0, valve_control 0.404 → 0.632; nhóm V 0.306 → 0.171 trong khi nhóm H tăng
(H_tu_nhien 0.208 → 0.385, H_chon 0.009 → 0.233) và nhóm S (học đúng dạng ISO) 0.99. R1a đoán butterfly → valve_gate /
check_valve_globe, check_valve_general → valve_control: encoder học **hình**, không học **nghĩa**. Chỉ 4 lớp → mô tả, không kiểm định.
Hệ quả: nhãn VARIANT không đổi gì khi train (trừ loại `h_extra` khỏi R1a — thận trọng, vô hại, giữ nguyên); nó chỉ quyết định
**chấm thế nào**.

**Đổi được câu trả lời của model không? — minh hoạ 2026-11-18** (crop tờ train, DINOv2 / A4 / CL2N; minh hoạ, không quyết định;
truy vấn có thể cùng tờ với ảnh mẫu thêm → lạc quan; số đo đúng cách là H1 §2D.3). Nhãn VARIANT chỉ đổi **cách chấm**; câu trả lời
đổi qua **thư mục ảnh mẫu**, **τ**, hoặc **train encoder**:
- Thêm 1 crop vẽ theo chuẩn bản vẽ cho mỗi lớp VARIANT (mang nghĩa ISO): S6 0 → 0.83, S8 0 → 1.00, S14 0 → 1.00, S20 0.48 → 1.00,
  S22 0.24 → 0.97, S23 0.90 → 1.00. **Nhưng** S1 van cổng 0.97 → **0.03**: 23/29 crop sang ảnh S14 (van + bộ truyền động), 5 sang
  ảnh S8 → ảnh mẫu **cùng phong cách bản vẽ** hút cả lớp ISO lân cận (phong cách thắng hình). Trộn ảnh mẫu ISO với ảnh mẫu chuẩn bản
  vẽ là nguy hiểm; thêm theo chuẩn bản vẽ thì nên thêm **cho cả bộ** (= H1: 0.78 / 0.90 với 1 / 5 crop).
- τ không cứu được: cos tốt nhất của câu trả lời SAI ở S6 (trung vị 0.333, max 0.624) chồng lên câu trả lời ĐÚNG ở S1 (trung vị
  0.424, min 0.255) → ngưỡng nào chặn S6 cũng chặn S1.

**Đề xuất (CHƯA CHỐT, chờ user)** — sửa luật cho các bước MỚI, ghi §7 khi chốt: bỏ lớp VARIANT khỏi chỉ số quyết định (như P1 test
đã làm); báo riêng một dòng VARIANT: % gọi đúng nghĩa và % bị trả về Other (cả hai chấp nhận được, gọi sai nghĩa mới là lỗi). Không
chấm lại quyết định cũ; P1 test, P4, luật §2E.3 không bị ảnh hưởng. Ảnh mẫu ghép S17 / S18 / S19 (511 + C0098 / 2044 / 2045): để
sau kết quả F0-Colab, cần luật riêng (đụng lớp test `h_in`), ảnh ghép không ghi vào `iso_symbols_clean` / `vocab.csv`.

### 2F.1 Thư mục ảnh mẫu trộn phong cách — THIẾT KẾ CHỐT 2026-11-18, TRƯỚC KHI CODE / CHẠY (user: "Hãy làm (b) đúng cách")

**Mô tả, không quyết định** (không đổi cấu hình, không chọn gì). Câu hỏi: "thêm ký hiệu = thêm 1 ảnh" còn đúng không khi ảnh thêm
vào **khác phong cách** với các ảnh mẫu sẵn có? Bản thử sơ bộ (§2F, crop train, 1 lần rút) thấy VARIANT tăng mạnh nhưng S1 sụp.

- **Không nhúng lại**: cache `runs/phase_c/probe/emb_dinov2_a4.pt` (DINOv2-S, A4, 8 view); CL2N với tâm của probe (crop train).
- **Ảnh theo chuẩn bản vẽ** = ảnh mẫu 1e của probe: lần rút d ∈ {0..4} (`ex<d>`), 1 crop / lớp, tờ **train**, 28 lớp (mọi lớp trừ
  `h_out`). Prototype = TB 8 view (như 1e). Tên = **nghĩa ISO** của lớp (`groups[k].template`); lớp không có nghĩa ISO (OOV, UNSURE)
  giữ tên lớp. 7 lớp OOV của P1-val (S7, S16, S26, S28, S29, S31, S32) **bị loại** khỏi mọi thư mục để vẫn là "lạ" → còn 21 ảnh.
- **Truy vấn** = crop P1-val (tờ `val_sel`, không trùng tờ nào với ảnh mẫu — đã kiểm). Biết = 10 lớp (4 ISO: S1, S17, S21, S24;
  6 VARIANT: S6, S8, S14, S20, S22, S23); lạ = 7 lớp OOV. Đúng khi tên top-1 = nghĩa ISO của lớp.
- **Bốn thư mục**:
  - **D0** 42 ảnh mẫu ISO (mốc).
  - **D1** 42 ISO + 6 ảnh chuẩn bản vẽ cho 6 lớp VARIANT (= bản sơ bộ, làm đúng cách).
  - **D2** 42 ISO + 21 ảnh chuẩn bản vẽ (mọi lớp đều có ảnh chuẩn bản vẽ, cạnh ISO).
  - **D3** chỉ 21 ảnh chuẩn bản vẽ (chế độ "legend").
- **Chỉ số** (D1–D3: TB ± độ lệch chuẩn 5 lần rút): macro top-1 ISO(4), VARIANT(6), cả 10; top-1 từng lớp + tên bị đoán nhiều nhất;
  AUROC cos (10 lớp biết vs 7 lạ). Không có τ (τ phải chọn lại cho từng thư mục → ngoài phạm vi).
- **Dự đoán ghi trước** (từ bản sơ bộ): D1 VARIANT ≫ D0 nhưng ISO(4) < D0 (S1 bị ảnh S14 / S8 hút); D2 khôi phục ISO(4) vì S1 có ảnh
  cùng phong cách; D3 ≈ D2 (cùng phong cách, như 1e 0.82).
- Code: `scripts/mixed_directory.py` → `runs/phase_c/mixed/` (`report.txt`, `results.json`); hàm dựng thư mục có test.

### 2F.2 Kết quả — 2026-11-18 (CPU vài giây; D0 tái lập đúng probe: 0.477 / AUROC 0.692)

| thư mục | # ảnh mẫu | ISO (4) | VARIANT (6) | cả 10 | AUROC cos |
|---|---|---|---|---|---|
| D0 chỉ 42 ISO | 42 | 0.800 | 0.261 | 0.477 | 0.692 |
| D1 ISO + 6 ảnh bản vẽ (chỉ lớp VARIANT) | 48 | **0.568 ± 0.068** | 0.957 ± 0.004 | 0.801 ± 0.029 | 0.840 ± 0.014 |
| D2 ISO + 21 ảnh bản vẽ (mọi lớp) | 63 | 0.956 ± 0.036 | 0.940 ± 0.010 | 0.947 ± 0.009 | 0.933 ± 0.008 |
| D3 chỉ 21 ảnh bản vẽ | 21 | 0.956 ± 0.036 | 0.961 ± 0.019 | 0.959 ± 0.012 | 0.936 ± 0.009 |

Từng lớp (D0 → D1 → D2 → D3): S1 0.99 → **0.17** (bị đoán valve_control = ảnh S14) → 0.90 → 0.90; S17 0.22 → 0.14 → 0.92 → 0.92;
S6 0 → 0.95 → 0.89 → 0.89; S8 0 → 1.00; S14 0.01 → 1.00; S22 0.19 → 0.81 / 0.78 / 0.88.

**Đọc** (khớp dự đoán ghi trước):
- Thêm ảnh khác phong cách cho **một phần** lớp (D1) kéo các lớp đó lên gần 1 nhưng **hút** lớp lân cận chỉ có ảnh ISO (S1 0.99 → 0.17,
  ISO(4) 0.80 → 0.57). Cái hại là **phủ không đều phong cách**, không phải bản thân việc trộn: D2 (mọi lớp đều có ảnh bản vẽ, ISO vẫn
  nằm cạnh) ≈ D3 (chỉ ảnh bản vẽ) — khi đã đủ ảnh cùng phong cách, 42 ảnh ISO **vô hại và vô dụng**.
- Ảnh cùng phong cách còn làm **tách lạ** tốt hẳn (AUROC 0.69 → 0.93), dù 7 lớp lạ không có ảnh nào.
- Hệ quả cho tuyên bố "thêm ký hiệu = thêm 1 ảnh": đúng **khi mọi ảnh mẫu cùng phong cách với bản vẽ**; thêm lẻ một ảnh khác phong
  cách vào thư mục ISO có thể làm hỏng lớp khác. Khuyến nghị triển khai: với bộ bản vẽ theo chuẩn khác, lấy ảnh mẫu cho **cả bộ** từ
  legend / chính bản vẽ (D3), không vá từng lớp.
- Giới hạn: ảnh bản vẽ lấy cùng bộ DigitizePID (tờ train) → cận trên kiểu "legend cùng bộ"; 10 lớp biết; mô tả, không kiểm định.

## 2G. Điểm 4 — bỏ ★ `pipeline` khỏi thư mục ảnh mẫu — CHỐT 2026-11-19, TRƯỚC KHI CODE / CHẠY (user: "nó nên chỉ là nét nối")

**Quyết định (chuyên gia P&ID, lý do ngữ nghĩa, KHÔNG dựa trên số):** `pipeline` (vocab 405) là nét nối giữa các ký hiệu, không phải
một ký hiệu cần nhận dạng → hệ thống đề xuất dùng **41 ảnh mẫu**: `++oneshot.exclude_prototypes=[pipeline]` (cơ chế có sẵn, có test).
Không sửa `iso_symbols_clean`, `vocab.csv`, `config.yaml`; không dựng lại prototype / tâm (tâm CL2N tính từ crop, mỗi ảnh mẫu
chuẩn hoá riêng → bỏ một ảnh mẫu không đổi cosine của 41 ảnh còn lại). Đây là quyết định phạm vi từ vựng, không phải một cách tối ưu.

**Đã biết trước khi khoá (ghi để minh bạch):**
- P2 của R1b (2026-11-13/14): S24 (`flow_direction`) → `pipeline` 180/180 crop — nguồn gốc của Điểm 4.
- Hôm nay, trước khi khoá, đã đếm trong CSV của 2B: P1 có 2 crop top-1 = `pipeline`; P4 có 2216 (933 là `flanged_connection`
  suy giảm). Không lớp nào trong P1 / P2 / P4 nhận `pipeline` làm đáp án.

**Hệ quả lý thuyết (kiểm bằng code):** chỉ crop có `pipeline` trong top-2 đổi kết quả (top-1 / best_cos / second_cos lùi một bậc,
tính đúng được từ top-3 của CSV 42). Vì không lớp nào có đáp án `pipeline`, **top-1 của mọi lớp không thể giảm**; AUROC có thể
đổi cả hai chiều (cả crop biết lẫn crop lạ từng khớp `pipeline` đều giảm best_cos).

**Đo (2B: DINOv2-S + A4 + CL2N, encoder / prototype / tâm của §13, chỉ thêm `exclude_prototypes`):**
1. **P2** (val, `image_subset=test`, nhóm theo R1b `__global__.pt`): 2B 42 và 2B 41. 2B **chưa từng chạy P2** → đây là bằng chứng
   sạch của Điểm 4. Báo nhóm S / V top-1 macro, top-3, AUROC cos / lề; CI 95 % bootstrap theo ảnh trong lớp.
2. **τ** (val_sel, nhóm R1a, luật A / B, δ) chọn lại cho 41 — hợp lệ vì là val.
3. **P1 đủ, P4**: 41 so với 2B 42 — **HẬU KIỂM** (P1 / P4 của 2B đã xem, và đã đếm số crop `pipeline` ở trên). Báo cạnh số 42,
   gắn nhãn hậu kiểm; P4 dùng cùng bootstrap theo tờ như §2E.3.
4. **Kiểm nhất quán (chặn):** top-1 của từng crop trong CSV 41 phải trùng kết quả tính từ CSV 42 (bỏ `pipeline` khỏi top-3),
   best_cos / second_cos lệch ≤ 1e-4; `n_prototypes` = 41 trong mọi metrics. Lệch → dừng, báo user, không ghi số.

**Báo cáo:** số 42 đã báo ở §2B giữ nguyên làm số ghi trước. Luận văn mô tả hệ thống đề xuất = 41 ảnh mẫu, kèm P2 (sạch) và
P1 / P4 41 (hậu kiểm, ghi rõ). Nếu P2 cho AUROC cos hoặc lề "kém hơn" (CI nằm hẳn dưới 0) → báo user; quyết định ngữ nghĩa
không tự đảo, nhưng phải ghi là đánh đổi.

**Code:** notebook §15 (τ 41 → P2 42 / 41 → P1 41 → P4 41, chép CSV vào `runs/nopipe_csv/`); `scripts/nopipe_compare.py`
(kiểm nhất quán + bảng, đọc CSV, CPU) → `runs/phase_c/nopipe/`.

**Mô phỏng HẬU KIỂM từ CSV 2B có sẵn — 2026-11-19** (user: "hãy tính thử"; `drop_template`, top-1 / AUROC chính xác, top-3 cận
dưới; `runs/phase_c/nopipe_sim/results.json`): tác động **nhỏ**.
- P1 đủ: không đổi gì (0.576 / AUROC 0.360).
- P4 suy giảm: S 0.847 → 0.863 (+0.016, CI theo tờ [+0.004, +0.040]); V / H_chon / H_tu_nhien không đổi; AUROC cos 0.728 → 0.726
  (−0.002, kém hơn rất nhẹ), lề 0.703 → 0.708. Sạch: S +0.006, cùng chiều.
- Lớp duy nhất đổi: `flanged_connection` 0.592 → 0.639. Trong 933 crop từng bị gọi `pipeline`, chỉ 195 thành đúng; còn lại sang
  flow_direction 374, tank 326 → điểm yếu mặt bích là của encoder, `pipeline` chỉ là nơi lỗi rơi vào trước.

## 2H. A — So khớp theo patch DINOv2 (tầng giữa) — LUẬT CHỐT 2026-11-19, TRƯỚC KHI CODE / CHẠY (user: "Hãy thử A")

**Cơ sở** (tra 2026-11-19, user yêu cầu): không paper P&ID nào đo đúng thiết lập "ảnh mẫu theo chuẩn ISO → bản vẽ khác phong cách"
(OSSR-PID 2021, Gupta et al. 2024, Hain et al. 2025, mask-aware template matching EC3 2026 đều lấy ảnh mẫu từ chính bộ bản vẽ /
legend của nó). Kỹ thuật gần nhất cho đúng loại lệch miền (ảnh mẫu sạch / tổng hợp ↔ ảnh thật): **FoundPose** (Örnek et al.,
ECCV 2024, arXiv 2311.18809) — kNN trên descriptor patch DINOv2 **tầng giữa** khớp được ảnh thật với ảnh mẫu tổng hợp, không train;
**DN4** (Li et al., CVPR 2019) — so khớp ảnh-với-lớp bằng descriptor cục bộ tốt hơn vector toàn ảnh khi ít mẫu; nhận dạng
watermark one-shot "1 hình vẽ / lớp → ảnh chụp" (Shen et al. 2019, arXiv 1908.10254) — đặc trưng tầng giữa. Mức: **vừa–mạnh**.
(= hướng C4 cũ, nay có cơ sở trực tiếp hơn.)

**Thiết lập (cố định trước):**
- Encoder DINOv2-S/14 đóng băng (`vit_small_patch14_dinov2.lvd142m`, 224 px, 16 × 16 = 256 patch), A4 cho crop và ảnh mẫu (như 2B).
  Token patch lấy bằng `forward_intermediates(..., norm=True)` (LayerNorm cuối, như `get_intermediate_layers` của DINOv2).
- **Patch nền bỏ đi**: chỉ giữ patch có ≥ 1 pixel mực (xám < 128) — mặt nạ mực tính trên view ở độ phân giải gốc (sau A4), gộp
  max về lưới 16 × 16 (thu về 224 trước sẽ làm nhạt nét 1 px của ảnh mẫu lớn). View trắng trơn → giữ mọi patch.
- Ảnh mẫu: 1 view (như `template_views`), **41 ảnh mẫu** (§2G). Truy vấn: 8 view của `query_views` (2 tỉ lệ × 4 góc).
- Điểm một (view, ảnh mẫu) = Chamfer đối xứng trên cosine patch: ½ (TB_q max_t cos + TB_t max_q cos). Điểm lớp = TB qua 8 view.
  Top-1 / top-3 / best / second / AUROC (điểm và lề) như probe (`summarize`).
- **4 biến thể** (2 × 2): tầng {9 (≈ 3/4 độ sâu, như FoundPose 18/24), 12 (cuối)} × tâm patch {không, có — trừ TB descriptor patch
  mực của 570 crop "center" (crop train 19 lớp R1a, cùng tập tâm CL2N) rồi L2}. Biến thể dự thi = **top-1 P1-val cao nhất**
  (hoà → thứ tự liệt kê), như cách 1c chọn bản trong §1.2.
- Chỉ P1-val (quyết định). Không chạy mẫu P1 / 1e (tiết kiệm; không tham gia quyết định).

**Luật (§1.2, ngưỡng không đổi):** so với **mốc 2B toàn ảnh** (DINOv2-S / A4 / CL2N, cùng 41 ảnh mẫu, tính lại trong cùng script
từ cache probe) trên P1-val: ĐẠT nếu (i) top-1 tăng ở ≥ 6/10 lớp biết **và** macro top-1 +0.05; **hoặc** (ii) AUROC (điểm hoặc lề)
+0.05 mà macro top-1 giảm ≤ 0.01.
- ĐẠT → đưa vào pipeline + Colab (τ, P1 đủ, P4), với luật chấp nhận viết riêng TRƯỚC khi chạy (dạng §2E.3).
- KHÔNG ĐẠT → ghi ablation (cả 4 biến thể), dừng hướng A.

**Code:** `scripts/patch_probe.py` (+ test các hàm thuần) → `runs/phase_c/patch/` (`report.txt`, `results.json`, cache điểm).

### 2H.1 Kết quả A — 2026-11-19 (MPS ~10 phút; mốc 41 ảnh mẫu = 0.477 / 0.692, trùng bản 42) → **KHÔNG ĐẠT**

| P1-val | top-1 | top-3 | AUROC điểm | AUROC lề | hub |
|---|---|---|---|---|---|
| **mốc 2B toàn ảnh / CL2N** | **0.477** | 0.643 | **0.692** | 0.565 | 0.158 |
| patch tầng 9 / thô | 0.353 | 0.486 | 0.175 | 0.474 | 0.196 |
| patch tầng 9 / trừ tâm | 0.297 | 0.444 | 0.122 | 0.465 | 0.133 |
| patch tầng 12 / thô | 0.415 | 0.571 | 0.312 | 0.603 | 0.211 |
| patch tầng 12 / trừ tâm ← dự thi | 0.450 | 0.641 | 0.320 | **0.644** | 0.196 |

Áp luật: 4/10 lớp tăng, Δtop-1 −0.027, ΔAUROC lề +0.078 → nhánh AUROC cần top-1 giảm ≤ 0.01 → **KHÔNG ĐẠT**. Ghi ablation; dừng A.

**Đọc:**
- Tầng giữa (9) **kém hơn** tầng cuối ở mọi chỉ số — ngược khuyến nghị FoundPose; với ảnh mẫu ISO → crop P&ID, đặc trưng ngữ nghĩa
  muộn chuyển giao tốt hơn đặc trưng hình học tầng giữa.
- Từng lớp: S20 0.51 → 0.72 (tăng), nhưng S23 0.85 → 0.48, S17 0.22 → 0.09. S6 / S8 / S14 vẫn ≈ 0: VARIANT khác hình dạng (§2F),
  so khớp cục bộ không cứu được — đúng chẩn đoán "khác phong cách / khác hình", không phải "vector toàn ảnh mất chi tiết".
- AUROC theo điểm < 0.5 ở mọi biến thể: crop lạ có điểm Chamfer **cao hơn** crop biết (giả thuyết, chưa kiểm: crop nhiều nét
  khớp cục bộ với nhiều ảnh mẫu) → điểm Chamfer thô không dùng làm cổng "Other" được; chỉ lề có ích.

## 2I. Đặc trưng tầng giữa CNN ImageNet (Gan et al. 2023) — LUẬT CHỐT 2026-11-19, TRƯỚC KHI CODE / CHẠY (user: "Hãy thực hiện phần đáng thử")

**Nguồn gốc:** user đưa đề xuất "ghép GeM layer3 với embedding ArcFace theo α" (trên R1b `__global_ep3__.pt`), dẫn Gan et al. 2023.
Đã kiểm 2026-11-19: (a) paper có thật nhưng **không** ghép tầng giữa với tầng cuối theo α — mỗi tầng dùng riêng → bỏ bản α
(không có cơ sở paper, thêm hằng số phải chỉnh); (b) R1b không còn là encoder chính (2B = DINOv2); (c) cụm vẫn còn trong 2B
(CL2N: pump_general–compressor_vacuum 0.936, globe–butterfly 0.898, check_valve_globe–swing 0.938), nhưng cos giữa ảnh mẫu chỉ
là chỉ báo (B4) → quyết định bằng P1-val. Chỉ làm **bản đúng paper**.

**Cơ sở:** Gan, Li, Togo, Maeda, Ogawa, Haseyama, *Zero-Shot Traffic Sign Recognition Based on Midlevel Feature Matching*, Sensors
23(23):9607, 2023 (doi 10.3390/s23239607). Ảnh mẫu chuẩn → ảnh thật, không train; CNN ImageNet không fine-tune; tầng giữa hơn hẳn
tầng cuối (top-1 GTSRB ResNet-50 0.521 vs 0.148; Sapporo DenseNet-121 0.817 vs 0.169). Mức: **vừa** (miền biển báo, không báo
phương sai). Lưu ý trước khi chạy: §2H đã thấy tầng giữa DINOv2 kém tầng cuối (ViT ≠ CNN nên chưa loại được).

**Thiết lập (cố định trước):**
- Backbone `timm`, trọng số ImageNet, không fine-tune: **ResNet-50** `resnet50.tv_in1k` và **DenseNet-121** `densenet121.tv_in1k`
  (hai backbone của paper), **ResNet-34** `resnet34.a1_in1k` (= trọng số mốc ImageNet 1a của ta). Không làm EfficientNet-B0 (tầng
  giữa 320×7×7 của paper kém hai cái kia ở cả hai tập).
- **Tầng giữa** = đầu ra stride 16 ở 224 px: ResNet `layer3` (1024×14×14; ResNet-34 256×14×14), DenseNet `denseblock3`
  (1024×14×14) — đúng kích thước Bảng 1 của paper (đã kiểm hình dạng, chưa xem kết quả).
- Tiền xử lý: A4 cho crop và ảnh mẫu (như 2B); resize thẳng 224 × 224 (paper, không cắt giữa); chuẩn hoá mean/std ImageNet.
  41 ảnh mẫu (§2G), 1 view ảnh mẫu, 8 view truy vấn (`query_views`); điểm lớp = TB qua 8 view.
- Đặc trưng = **cả tensor tầng giữa trải phẳng** (giữ vị trí, không pool). Hai cách chấm:
  **E** = −‖f_q − f_t‖² trên tensor thô (đúng Eq. 3 của paper); **C** = cosine giữa hai tensor trải phẳng (bỏ ảnh hưởng độ lớn).
- **6 biến thể dự thi** = 3 backbone × {E, C}. Dự thi = **top-1 P1-val cao nhất** (hoà → thứ tự ResNet-50, DenseNet-121,
  ResNet-34; E trước C).
- Dòng **tham khảo, không dự thi**: tầng cuối (stride 32, 7 × 7, trải phẳng) cùng 3 backbone × {E, C} — kiểm lại "giữa > cuối"
  của paper trên dữ liệu ta (mô tả).
- Chỉ P1-val (quyết định). Không chạy mẫu P1 / P4.

**Luật (§1.2, ngưỡng không đổi):** so với **mốc 2B toàn ảnh** (DINOv2-S / A4 / CL2N, 41 ảnh mẫu, tính lại từ cache probe) trên
P1-val: ĐẠT nếu (i) top-1 tăng ở ≥ 6/10 lớp biết **và** macro top-1 +0.05; **hoặc** (ii) AUROC (điểm hoặc lề) +0.05 mà macro
top-1 giảm ≤ 0.01.
- ĐẠT → luật chấp nhận riêng (dạng §2E.3) TRƯỚC khi đưa vào pipeline + Colab.
- KHÔNG ĐẠT → ghi ablation (6 biến thể + dòng tham khảo), dừng hướng "tầng giữa".

**Code:** `scripts/midlevel_probe.py` (+ test hàm thuần) → `runs/phase_c/midlevel/` (`report.txt`, `results.json`, cache điểm).

### 2I.1 Kết quả — 2026-11-19 (MPS ~5 phút; mốc 41 ảnh mẫu = 0.477 / 0.692; hình dạng đúng Bảng 1 của paper) → **KHÔNG ĐẠT**

| P1-val | top-1 | top-3 | AUROC điểm | AUROC lề | hub |
|---|---|---|---|---|---|
| **mốc 2B toàn ảnh / CL2N** | **0.477** | **0.643** | 0.692 | 0.565 | **0.158** |
| ResNet-50 giữa / E · C | 0.112 · 0.216 | 0.268 · 0.383 | 0.845 · 0.827 | 0.516 · 0.569 | 0.964 · 0.780 |
| DenseNet-121 giữa / E · C | 0.101 · 0.117 | 0.300 · 0.297 | 0.751 · 0.735 | 0.257 · 0.251 | 0.997 · 0.989 |
| ResNet-34 giữa / E · C ← dự thi (C) | 0.111 · 0.251 | 0.494 · 0.503 | 0.825 · 0.805 | 0.475 · 0.383 | 0.884 · 0.715 |
| *tham khảo* ResNet-50 cuối / E · C | 0.182 · 0.336 | 0.288 · 0.526 | 0.427 · 0.366 | 0.340 · 0.528 | 0.492 · 0.333 |
| *tham khảo* DenseNet-121 cuối / E · C | 0.245 · 0.247 | 0.423 · 0.460 | 0.322 · 0.532 | 0.700 · 0.527 | 0.329 · 0.313 |
| *tham khảo* ResNet-34 cuối / E · C | 0.196 · 0.336 | 0.276 · 0.434 | 0.259 · 0.550 | 0.311 · 0.641 | 0.735 · 0.167 |

Áp luật (ResNet-34 giữa / C vs mốc): 3/10 lớp tăng, Δtop-1 **−0.225**, ΔAUROC điểm +0.113 → nhánh AUROC cần top-1 giảm ≤ 0.01
→ **KHÔNG ĐẠT**. Ghi ablation; dừng hướng "tầng giữa" (cả ViT §2H lẫn CNN §2I).

**Đọc:**
- **Sụp về một ảnh mẫu**: hub 0.72–0.997 — gần như mọi crop ra `flanged_connection` (ảnh mẫu ít nét). So khớp cứng theo vị trí
  trên bản đồ 14 × 14 phạt mọi lệch nét giữa phong cách ISO và DigitizePID, nên ảnh mẫu "trống" nhất thắng. Biển báo của paper
  là ảnh thật đã cắt sát, cùng bố cục với ảnh mẫu; crop P&ID thì không.
- **Tầng giữa kém tầng cuối** về top-1 ở cả 3 backbone (ngược paper; khớp §2H với DINOv2).
- AUROC điểm cao (0.75–0.85) **không dùng được**: top-1 đã sụp, và cao một phần vì crop biết bị kéo về một ảnh mẫu (chưa kiểm).
- Từng lớp (ResNet-34 giữa / C): S22 0.19 → 0.55, S14 0.01 → 0.16, nhưng S1 0.99 → 0.01, S20 0.51 → 0, S23 0.85 → 0.19;
  S6 / S8 vẫn 0.

## 2J. Ba phần còn sót của đề xuất "4 hướng" — LUẬT CHỐT 2026-11-19, TRƯỚC KHI CODE / CHẠY (user: "Hãy chạy thử §2J")

**Nguồn gốc:** user đưa 4 hướng (trừ trung bình từng miền, so khớp patch có mực, `reg4` + ghép tầng, đồng bộ hình thái ảnh
mẫu). Đối chiếu 2026-11-19: hướng 1 ≈ F0-a (§2E), hướng 2 ≈ A (§2H), `reg4` = C3 (§2C.3), letterbox + lề = A4, dày nét / ống /
ScanDegrade = H2 (§2D) — đều đã đo, KHÔNG ĐẠT hoặc đã dùng. Còn sót 3 phần chưa đo, gộp vào **một** vòng duy nhất, chỉ là ablation.

**Cơ sở (mức):**
- J-a Chamfer **một chiều**: chiều crop → ảnh mẫu = DN4 (Li et al. CVPR 2019: mỗi descriptor truy vấn tìm láng giềng trong lớp) và
  AnomalyDINO (Damm et al. WACV 2025: mỗi patch ảnh thử tìm patch gần nhất trong bộ nhớ mẫu); chiều ảnh mẫu → crop = đề xuất của
  user (bỏ qua patch ống thừa trên crop), không có paper riêng. Mức **vừa** / **yếu**.
- J-b Gộp **trung bình các patch có mực** (thay CLS): thực hành che nền, không có paper đo trên bài này (GeM của Radenović et al.
  TPAMI 2019 cho đặc trưng sau ReLU của CNN, không hợp token ViT có giá trị âm → không dùng GeM). Mức **yếu**.
- J-c **Ghép CLS 4 tầng cuối**: DINO (Caron et al. ICCV 2021) / DINOv2 dùng cho đánh giá *linear* (có train), không cho k-NN.
  Mức **yếu**; §2H / §2I đã thấy tầng giữa kém tầng cuối.
→ Theo quy ước §2C.0 các hướng yếu chỉ báo như ablation; dù ĐẠT cũng phải có luật chấp nhận riêng trước khi vào pipeline.

**Thiết lập (cố định trước):** DINOv2-S/14 đóng băng, A4, 224 px, **41 ảnh mẫu** (§2G), 1 view ảnh mẫu, 8 view truy vấn; token lấy
bằng `forward_intermediates(..., norm=True)` như §2H; mặt nạ mực và tâm patch đúng như §2H. **6 biến thể**:
- `oneway/q2t`, `oneway/t2q`: tầng 12, trừ tâm patch (cấu hình dự thi của A), điểm = TB qua patch mực của **một** phía về max cos
  sang phía kia; TB qua 8 view.
- `inkmean/raw`, `inkmean/cl2n`: vector = TB token patch tầng 12 trên patch mực, L2 từng view; gộp view và CL2N đúng như `rank`
  của probe (tâm = TB view của 570 crop "center").
- `cat4/raw`, `cat4/cl2n`: vector = ghép CLS (sau LayerNorm cuối) của tầng 9–12 (1536-d), L2 từng view; gộp / CL2N như trên.
- Dự thi = **top-1 P1-val cao nhất** (hoà → thứ tự liệt kê trên).
- **Kiểm tái lập trước khi chấm**: (i) CLS tầng 12 của ảnh mẫu (L2) trùng prototype cache probe (sai khác tối đa < 5e-3, ghi số);
  (ii) ½ (q2t + t2q) tái lập điểm `L12/center` của A (`runs/phase_c/patch/scores.pt`, < 1e-3). Lệch → dừng, không chấm.
- Chỉ P1-val.

**Luật (§1.2, ngưỡng không đổi)** so với mốc 2B toàn ảnh (DINOv2-S / A4 / CL2N, 41 ảnh mẫu, 0.477 / 0.692): ĐẠT nếu (i) top-1
tăng ở ≥ 6/10 lớp biết **và** macro top-1 +0.05; **hoặc** (ii) AUROC (điểm hoặc lề) +0.05 mà macro top-1 giảm ≤ 0.01.
- ĐẠT → chỉ báo; muốn đưa vào pipeline phải chốt luật chấp nhận riêng (dạng §2E.3) và user duyệt.
- KHÔNG ĐẠT → ghi ablation, **đóng danh sách tối ưu encoder / cách đọc đặc trưng**, chuyển Colab §15 + Bước 3.

**Code:** `scripts/probe_2j.py` (+ test hàm thuần) → `runs/phase_c/j/` (`report.txt`, `results.json`, cache điểm).

### 2J.1 Kết quả — 2026-11-19 (MPS ~5 phút) → **KHÔNG ĐẠT**; đóng danh sách tối ưu cách đọc đặc trưng

Kiểm tái lập qua: CLS-12 của ảnh mẫu vs prototype probe 1.2e-6; ½ (q2t + t2q) vs điểm A `L12/center` 1.2e-7.

| P1-val | top-1 | top-3 | AUROC điểm | AUROC lề | hub |
|---|---|---|---|---|---|
| **mốc 2B toàn ảnh / CL2N** | **0.477** | 0.643 | **0.692** | 0.565 | 0.158 |
| oneway/q2t (crop → ảnh mẫu) | 0.440 | 0.607 | 0.441 | **0.623** | 0.205 |
| oneway/t2q (ảnh mẫu → crop, đề xuất) | 0.423 | 0.641 | 0.217 | 0.526 | 0.174 |
| inkmean/raw | 0.417 | 0.589 | 0.450 | 0.575 | 0.205 |
| inkmean/cl2n | 0.445 | 0.623 | 0.521 | 0.598 | 0.169 |
| cat4/raw | 0.415 | 0.555 | 0.621 | 0.610 | 0.329 |
| cat4/cl2n ← dự thi | 0.471 | **0.651** | 0.673 | 0.567 | **0.155** |

Áp luật (cat4/cl2n vs mốc): 1/10 lớp tăng, Δtop-1 −0.006, ΔAUROC +0.002 → **KHÔNG ĐẠT**. Ghi ablation.

**Đọc:**
- `cat4/cl2n` gần như trùng mốc (từng lớp lệch ≤ 0.06; S22 0.19 → 0.24, S17 0.22 → 0.16): thêm CLS tầng 9–11 không mang thông
  tin mới cho bài này — khớp §2H / §2I (tầng giữa không giúp).
- Chiều **ảnh mẫu → crop** (đề xuất "bỏ qua ống thừa") có AUROC điểm **0.217** — tệ nhất mọi dòng: crop nhiều nét (kể cả crop lạ)
  luôn chứa chỗ khớp cho patch của ảnh mẫu → đúng cảnh báo trước khi chạy. Chiều crop → ảnh mẫu (DN4) tốt hơn về top-1 / lề,
  nhưng vẫn dưới mốc; ½ hai chiều = A (0.450).
- Gộp patch có mực (inkmean) kém CLS: CLS của DINOv2 đã bỏ qua nền đủ tốt ở ảnh A4 (nền trắng đồng nhất).
- S6 / S8 / S14 vẫn 0 / 0 / 0.007 ở dự thi — như mọi dòng từ Bước 1.

## 2K. Mục tiêu mới: tách ảnh mẫu ISO giống nhau — bước 1: V-ISO + chẩn đoán + R2 (448 px) — LUẬT CHỐT 2026-11-19, TRƯỚC KHI CODE / CHẠY (user: "đồng ý thứ tự này, bắt đầu với 1")

**Vì sao đổi mục tiêu** (user 2026-11-19, sau phân tích lỗi): lỗi còn lại chia ba loại — (1) khác phong cách, cùng hình: đã giải
phần lớn (A4 + DINOv2); (2) khác dạng hình (VARIANT): chỉ legend / ví dụ thật giải được (§2F.2); (3) **ảnh mẫu ISO giống nhau, chỉ khác
chi tiết nhỏ** (butterfly ↔ globe ↔ swing…): chưa giải, nằm trong phạm vi bài toán ISO. **Động cơ lấy từ phân tích lỗi P4 (tập test)**
— phải ghi rõ trong luận văn: P4 suy giảm top-1 0.790, top-3 0.935, 69% lỗi có đáp án ở hạng 2–3, 30% lỗi rơi vào ảnh mẫu có cos ≥ 0.85
với ảnh mẫu đúng (valve_globe → valve_butterfly 489 crop, check_valve_globe → swing 181, butterfly → swing 166…). Vì đã nhìn P4, **mọi
lựa chọn từ đây dùng tập quyết định riêng V-ISO**; P4 chỉ dùng (a) kiểm V-ISO có đại diện không (cổng dưới, không chọn phương pháp),
(b) xác nhận một lần với luật viết trước. Thứ tự đã duyệt: bước 1 (rẻ, không train) → bước 2 fine-tune "giữ ảnh mẫu ra" (luật riêng).

**Lớp "sát nhau" (chốt trước, chỉ từ 41 ảnh mẫu, DINOv2-S / A4 / CL2N 224 px, cos ≥ 0.85): 19 cặp, 17 lớp** — blind_disc,
breather_valve, check_valve_general, check_valve_globe, check_valve_swing, compressor_centrifugal, compressor_vacuum,
interchangeable_disc, interchangeable_disc_open, open_disc, pump_centrifugal, pump_diaphragm, pump_general, valve_butterfly,
valve_globe, valve_needle, valve_three_way. 24 lớp còn lại = "không sát".

**V-ISO (tập quyết định mới, không dùng gì của P4 / OPEN100):** truy vấn sinh từ chính 41 ảnh mẫu bằng bộ sinh của H2
(`utils.template_variants.dirty_view`: lề 10%, ống ở cổng thật, nét ±1 px, `ScanDegrade` dải hẹp), **seed 2026** (H2 dùng seed 0),
**30 truy vấn / ảnh mẫu** = 1,230. Đi qua đúng đường của crop: A4 → 8 view `query_views` → DINOv2 → CL2N (tâm probe) → cos với 41
prototype (1 view sạch). Chỉ số: macro top-1 trên 17 lớp sát (**chính**), trên cả 41, trên 24 lớp không sát; tỉ lệ từng cặp nhầm.

**Cổng "V-ISO có đại diện cho P4 không" (đo trước khi so gì, ở 224 px, dạng đủ):**
- G1: Spearman ρ giữa top-1 từng lớp của V-ISO và của P4 suy giảm (CSV 2B, lớp có trong cả hai) **≥ 0.5**;
- G2: tỉ lệ valve_globe → valve_butterfly trên V-ISO **≥ 0.20** (P4 suy giảm 0.72).
- Trượt → V-ISO không đại diện: chỉ báo chẩn đoán (mô tả), **không** quyết định R2 trên V-ISO; dừng, bàn với user.

**Chẩn đoán (mô tả, không quyết định)** — cùng 30 truy vấn / ảnh mẫu, tắt / bật từng thành phần của bộ sinh: (a) **sạch** (chính ảnh
mẫu, chỉ khác 8 view truy vấn: 2 tỉ lệ × 4 góc xoay); (b) chỉ ống nối; (c) chỉ nét ± + suy giảm; (d) đủ (= V-ISO). Thêm (e) dạng đủ
nhưng **1 view** truy vấn (không xoay / phóng). Mỗi dạng ở 224 và 448 px. Đọc: lỗi đến từ cách gộp view, từ ống nối, từ suy giảm, hay
từ độ phân giải.

**R2 — tăng độ phân giải** (cơ sở: DINOv2, Oquab et al. TMLR 2024, train giai đoạn cuối ở 518 px; chi tiết nhỏ chiếm ít patch ở 224):
`vit_small_patch14_dinov2.lvd142m` với `img_size=448` (32 × 32 patch, nội suy positional embedding của timm), **mọi thứ ở 448**: ảnh
mẫu, tâm CL2N (570 crop center dựng lại), truy vấn. Một ứng viên duy nhất (không dò 336 / 518).
**Luật R2 (chỉ áp nếu cổng qua)**, so với 224 px:
- (i) V-ISO dạng đủ: macro top-1 17 lớp sát **+0.05**; **và**
- (ii) V-ISO cả 41 lớp: macro top-1 không giảm quá 0.01; **và**
- (iii) chặn hại phong cách: P1-val (crop DigitizePID, 2,550) macro top-1 không giảm quá 0.01 (báo kèm AUROC).
- ĐẠT → viết luật chấp nhận Colab (P4 + P1, dạng §2E.3) TRƯỚC khi chạy, chờ user duyệt. KHÔNG ĐẠT → ghi ablation; bước 2 (fine-tune)
  thiết kế theo kết quả chẩn đoán.

**Code:** `scripts/viso_probe.py` (+ test); thêm cờ tắt / bật từng bước vào `dirty_view` (mặc định giữ nguyên hành vi H2, có test).
→ `runs/phase_c/viso/` (`report.txt`, `results.json`, cache embedding).

### 2K.1 Kết quả bước 1 — 2026-11-19 (MPS ~30 phút) → **cổng TRƯỢT**; R2 KHÔNG ĐẠT (chặn (iii))

Kiểm: đường 224 tái lập prototype probe (1.2e-6) và P1-val 0.477; `dirty_view` mặc định trùng code H2 cũ 0 / 378 ảnh khác.

**Cổng:** Spearman ρ (top-1 từng lớp V-ISO vs P4 suy giảm, 30 lớp) **0.152** (cần ≥ 0.5) → **TRƯỢT**; globe → butterfly 0.80 (P4
0.72) qua. V-ISO tái hiện globe → butterfly và pump_general (0.23 vs P4 0.25) nhưng **không** tái hiện butterfly (V-ISO 1.00 vs P4
0.60), valve_needle (1.00 vs 0.22), check_valve_globe (1.00 vs 0.72); interchangeable_disc ngược chiều (0.07 vs 0.75). → P4 có nguồn
lỗi mà bộ sinh H2 không có. Theo luật: không quyết định R2 trên V-ISO; phần dưới chỉ là mô tả.

| V-ISO macro top-1 (17 sát / 24 khác / cả 41) | 224 px | 448 px |
|---|---|---|
| (a) sạch (chính ảnh mẫu, 8 view) | 0.941 / 1.000 / 0.976 | 0.882 / 1.000 / 0.951 |
| (b) chỉ ống nối | 0.835 / 0.967 / 0.912 | 0.941 / 0.990 / 0.970 |
| (c) chỉ nét ± + suy giảm | 0.939 / 0.986 / 0.967 | 0.910 / 0.982 / 0.952 |
| (d) đủ (= V-ISO) | 0.839 / 0.960 / 0.910 | 0.914 / 0.963 / 0.942 |
| (e) đủ, **1 view** truy vấn | 0.941 / 0.994 / 0.972 | **1.000 / 1.000 / 1.000** |
| P1-val (crop DigitizePID) top-1 / AUROC | 0.477 / 0.692 | **0.423 / 0.448** |

**R2:** dù cổng trượt, chặn (iii) **độc lập với V-ISO** đã hỏng: P1-val −0.054 (> 0.01), AUROC 0.692 → 0.448 → R2 **KHÔNG ĐẠT** bất kể
V-ISO. Ghi ablation.

**Đọc (mô tả; V-ISO không đại diện P4 nên chỉ nói về bộ sinh của ta):**
- **Gộp 8 view bằng trung bình làm mất chi tiết**: 1 view 0.941 vs 8 view 0.839 (17 sát, 224); ngay chính ảnh mẫu (a) cũng sai
  interchangeable_disc → _open. Lưu ý: truy vấn V-ISO đều đứng thẳng; crop thật có thể xoay, đó là lý do có 4 góc → không suy ra
  "1 view tốt hơn" cho crop thật. Hướng mở (cần luật riêng + tập quyết định đại diện): gộp `vote` (mặc định config, OSSR-PID) hoặc
  max theo góc thay vì trung bình embedding.
- **Ống nối là nguồn nhầm lớn nhất ở 224** (b 0.835 vs c 0.939); 448 gỡ được (0.941).
- **448 giúp chi tiết nhưng hại chuyển phong cách**: V-ISO tăng, P1-val giảm và AUROC sụp → đặc trưng độ phân giải cao bám pixel hơn,
  khoảng cách ISO ↔ DigitizePID rộng ra. Đánh đổi, không phải cải tiến.
- Hệ quả cho bước 2 (fine-tune): cần một tập quyết định **đại diện cho P4** trước. Ứng viên: render tờ mới bằng chính bộ render
  OPEN100-ISO (`render_iso`, V5–V8) từ khung xương khác trong PID2Graph.zip (`Dataset PID` 500 / `Synthetic` 250, chưa dùng làm test)
  → "P4-val". Chờ user.

### 2K.2 P4-val — tập quyết định đại diện P4 — THIẾT KẾ CHỐT 2026-11-19, TRƯỚC KHI CODE / RENDER (user chọn "A")

**Mục đích:** tập quyết định cho mục tiêu "tách ảnh mẫu ISO giống nhau" (bước 2 fine-tune, và mọi hướng như đổi cách gộp view).
Cùng **bộ render + ảnh mẫu vector** với P4 (OPEN100-ISO), khác **khung xương** → không đụng 12 tờ test.

**Nguồn (đã kiểm 2026-11-19):** `PID2Graph.zip` → `Complete/Dataset PID` (500 tờ, png + graphml; nhiều van: TB 56.5 / tờ, không bơm /
bồn) và `Complete/PID2Graph Synthetic` (250 tờ, jpg + graphml; có bơm 204 tờ, bồn 242 tờ, ít van). Cùng định dạng GraphML với OPEN100
(cùng 6 khoá, cùng tập nhãn). **0 file trùng md5** với 12 tờ OPEN100 cục bộ. Hai bộ này là topology tổng hợp — PLAN_ISO_RENDER V9 cấm
dùng **làm test thay OPEN100**; ở đây chỉ làm **tập quyết định**, đúng vai trò cho phép.

**Chuẩn bị (cố định trước):**
- Rút ngẫu nhiên `Random("p4val/2026")`: **20 tờ `Dataset PID` + 20 tờ `Synthetic`** = 40 tờ, đánh số lại **100–139** (bảng ánh xạ lưu lại).
- **Đưa về thang OPEN100**: mọi toạ độ × f, f = 26.3 / trung vị cạnh dài của nút `valve` trên **toàn bộ tờ của bộ đó** (OPEN100: 26.3
  px; đo sơ bộ 60 tờ: `Dataset PID` 83.0 → f ≈ 0.32, `Synthetic` 60.0 → f ≈ 0.44; dùng trị đo trên toàn bộ). Nút `connector` /
  `crossing` (điểm đánh dấu) giữ nguyên kích thước quanh tâm mới. PNG nguồn = ảnh trắng đúng kích thước sau khi co (bộ render chỉ đọc
  kích thước). Hệ quả đã biết: ở `Synthetic`, bơm / bồn nhỏ hơn P4 (tỉ lệ cỡ giữa các nhãn của bộ đó khác OPEN100) → ghi hạn chế.
- **Render đúng config hiện tại** (`configs/render_iso.yaml`, như P4, không chỉnh): V6 `make_variants.py` **2 biến thể / tờ**,
  `base_seed` mặc định của config, lọc G1 như cũ (**sửa 2026-11-19, trước khi chấm**: bản đầu ghi `--base-seed 2026`, nhưng V7 / V8
  vẽ lại ảnh sạch từ seed của config để kiểm, nên seed phải đến từ config; seed = hash(base, **số hiệu tờ**, biến thể, bước) → đánh số
  tờ 100–139 cho seed khác hẳn P4 (tờ 0–11) mà không sửa config); V7 `make_degraded.py`; V8 `pid-render-iso` `images=degraded` (P4 dùng ảnh suy giảm làm số chính).
  Tờ không giữ được biến thể nào → bỏ, ghi số. Mọi file ghi vào thư mục mới `data/iso_p4val/` (không đụng `data/canonical`,
  `data/iso_open100`, `iso_symbols_clean`, `vocab.csv`).
- 🧑 **Thử trước 2 tờ mỗi bộ**, ảnh cho user duyệt (như G1) rồi mới render đủ 40 tờ.

**Chấm:** crop theo hộp COCO đúng cách P4 (`utils.crop_box.cut`), 2B (DINOv2-S / A4 / 8 view gộp trung bình / CL2N tâm probe / **41**
ảnh mẫu) — đường của `viso_probe` (đã tái lập probe 1.2e-6). Biết = nhãn có trong 41 ảnh mẫu; `other_*` = lạ.

**Cổng đại diện (như §2K, ngưỡng không đổi):** G1 Spearman ρ top-1 từng lớp P4-val vs P4 suy giảm (lớp có ≥ 20 crop ở P4-val) **≥ 0.5**;
G2 globe → butterfly **≥ 0.20**; thêm G0: ≥ 15 lớp chung. Qua → P4-val là tập quyết định của mục tiêu này (luật bước 2 viết sau, trước
khi chạy). Trượt → dừng mục tiêu, ghi giới hạn (phương án C).

**Ràng buộc cho bước 2 (ghi trước):** nếu fine-tune dùng ảnh render thì **không** được dùng tờ P4-val để train (chia lại theo tờ trước
khi train); bản FT-chính chỉ train trên ảnh mẫu + augment nên P4-val giữ nguyên vai trò quyết định.

**Code:** `scripts/p4val_prepare.py` (rút tờ, co toạ độ, PNG trắng, bảng ánh xạ; hàm co có test) + `scripts/p4val_probe.py` (crop, chấm,
cổng) → `data/iso_p4val/`, `runs/phase_c/p4val/`.

### 2K.3 Kết quả P4-val — 2026-11-19 (user duyệt bản thử 4 tờ; render ~2 phút, chấm MPS vài phút) → **cổng QUA**

**Dựng:** 40 tờ nguồn (f `Dataset PID` 0.3169, `Synthetic` 0.4916 — trung vị van toàn bộ 83.0 / 53.5 px); V6 giữ **73 / 80** biến thể:
tờ 102, 111, 114 trượt G1 cả 60 lần thử (bỏ, theo luật), tờ 135 chỉ 1 biến thể khác nhau → **37 tờ, 73 ảnh suy giảm, 4,832 ký
hiệu** (2,378 biết / 2,454 lạ). `scripts/p4val_probe.py` → `runs/phase_c/p4val/` (`report.txt`, `results.json`,
`p4val_crops_dinov2.csv` có cột `template` = dạng SVG được vẽ).

**Cổng:** G0 18 lớp chung (≥ 15) · G1 Spearman ρ **0.815** (≥ 0.5) · G2 globe → butterfly **0.812** (≥ 0.20) → **QUA**. P4-val là tập
quyết định cho mục tiêu "tách ảnh mẫu giống nhau" (gọi tên). 2B trên P4-val: macro top-1 0.742, top-3 0.960 (lớp ≥ 20 crop).
**Không** dùng P4-val cho cổng "Other": AUROC cos 0.451 (P4 0.728) — thành phần `other_*` khác P4 (`other_generic` 1,213 hộp lớn từ
`Synthetic`), cổng chỉ kiểm gọi tên.

**Phát hiện (mô tả):**
- **valve_butterfly có hai dạng hình trong chính BK-2020** (taxonomy: X8075 form A / form B; ảnh mẫu PNG chỉ có một): crop vẽ form A
  (X8075_1) đúng **1.00** (61 crop), form B (X8075_2) đúng **0.06** (70 crop). Lỗi butterfly trên P4-val là lỗi "một lớp, hai dạng,
  một ảnh mẫu" — cùng bản chất loại 2 (VARIANT) nhưng **bên trong chuẩn ISO**. Trong 41 lớp chỉ butterfly có hai dạng (taxonomy).
- **flanged_connection** vẽ kèm đĩa (2044 / 2045 / C0096 / C0097 — luật "đĩa giữa cặp mặt bích" của bộ render) đúng 0.16–0.32, vẽ
  riêng (511) đúng 0.72 → lỗi ký hiệu ghép (như S17), không phải chi tiết nhỏ.
- Cặp nhầm hàng đầu khớp P4: globe → butterfly 121, valve_general → three_way 98, flanged → flow_direction 87, check_valve_general →
  breather_valve 58, needle → three_way 55, butterfly → swing 48.

**Bước tiếp — chờ user, mỗi hướng phải chốt luật trước, quyết định trên P4-val (gọi tên), P1-val làm chặn hại phong cách:**
(i) thêm dạng hình thứ hai của butterfly (form B) làm ảnh mẫu; (ii) đổi cách gộp 8 view (`vote` / max theo góc) — chẩn đoán V-ISO;
(iii) bước 2 fine-tune "giữ ảnh mẫu ra" như đã duyệt.

### 2K.4 Luật (i) dạng B của butterfly + (ii) cách gộp view — CHỐT 2026-11-19, TRƯỚC KHI CODE / CHẠY (user: "Làm (i) và (ii) trước")

**Chung:** encoder 2B (DINOv2-S 224 / A4 / CL2N tâm probe), không train. **Tập quyết định = P4-val** (chỉ gọi tên; 37 tờ, chỉ số chính =
macro top-1 trên **18 lớp có ≥ 20 crop**, như §2K.3; mốc 0.742). **Chặn hại phong cách = P1-val** (macro top-1 không giảm quá 0.01; báo
kèm AUROC). Khoảng tin cậy 95%: bootstrap **theo tờ** P4-val (B = 1000, seed 0), chênh lệch tính trên cùng mẫu rút. P4 / P1 không chạm.

**(i) Ảnh mẫu dạng B của valve_butterfly** — cơ sở: chính BK-2020 / taxonomy của dự án ghi X8075 có hai dạng (form A / form B) cho cùng
một lớp; "mỗi dạng hình một ảnh mẫu" giữ nguyên nguyên tắc "thêm ký hiệu = thêm ảnh". Không phải chọn từ dữ liệu: dạng B là dạng chuẩn
duy nhất còn thiếu (taxonomy: chỉ butterfly có hai dạng trong 41 lớp).
- Ảnh: `data/raw/template_iso_svg/X8075_2.svg` render bằng đúng hàm `render` của `scripts/extract_iso_symbols_svg.py` (600 dpi, lề 6 px)
  — cách đã tạo `iso_symbols_clean`. Kiểm nhất quán (đã đo trước khi chấm): render lại form A (X8075_1.svg) cùng cách → cùng kích thước
  255 × 136 với ảnh mẫu hiện có, IoU mực 0.925. Lưu ở `runs/phase_c/k4/` (**không** ghi vào `iso_symbols_clean`).
- Thư mục = 41 + 1 (tên `valve_butterfly@B`); đoán `valve_butterfly@B` = `valve_butterfly`. Điểm lớp = max điểm các ảnh mẫu của lớp.
- **ĐẠT** khi cả bốn: butterfly top-1 P4-val **+0.10**; không lớp nào khác trong 18 lớp giảm quá **0.05** (chống "hút lớp khác" kiểu D1);
  macro top-1 18 lớp không giảm; P1-val không giảm quá 0.01.

**(ii) Cách gộp 8 view truy vấn** (2 tỉ lệ × 4 góc), thay trung bình embedding (mốc). Hai ứng viên:
- `vote`: mỗi view chọn ảnh mẫu gần nhất (sau CL2N), lớp nhiều phiếu nhất thắng, hoà → tổng cos lớn hơn — đúng `classify_views(strategy=
  "vote")` của pipeline, không cổng; cơ sở OSSR-PID (Paliwal et al. 2021, kỹ thuật #3) — đang là mặc định `oneshot.aggregate` của config.
- `max`: điểm (crop, ảnh mẫu) = max theo view của cos (sau CL2N); cơ sở TI-pooling (Laptev et al., CVPR 2016: lấy max qua các bản biến đổi
  để bất biến mà không trộn đặc trưng).
- Dự thi = ứng viên macro top-1 P4-val cao hơn (hoà → `vote`). **ĐẠT** khi: Δ macro top-1 18 lớp **≥ +0.03** và CI 95% theo tờ **không chứa
  0**; P1-val không giảm quá 0.01.

**Kết hợp:** (i) và (ii) quyết định **riêng** so với mốc. Cả hai ĐẠT → báo thêm dòng kết hợp (mô tả). Mỗi cái ĐẠT → viết luật chấp nhận
Colab (P4 + P1, dạng §2E.3) trước khi chạy, chờ user. KHÔNG ĐẠT → ablation. Phần lỗi còn lại → bước 2 (fine-tune).

**Code:** `scripts/k4_probe.py` (+ test: gộp vote / max, quy tên, bootstrap) → `runs/phase_c/k4/`.

**Code pipeline (2026-11-19, sau §2K.5):** `utils.oneshot` — `class_of` / `FORM_SEP="@"`; `PrototypeDirectory.classes`,
`class_cosines` (mỗi lớp = max các dạng); `topk` / `match` / `match_views` quyết định theo **lớp** (thư mục không có `@` → y như cũ);
`match_views(strategy="max")`; `without(lớp)` bỏ mọi dạng của lớp. `oneshot_infer` — `second_cos` theo lớp, hỗ trợ `max`, chặn
`aggregate` lạ. `oneshot_eval.rank_names` — `max`, theo lớp; kiểm "lớp có ảnh mẫu" theo `classes`. `iso_prototypes` —
`++oneshot.extra_forms_dir` (file `<lớp>@<dạng>.png`, lớp phải có trong vocab; ghi `meta.extra_forms`). Ảnh dạng B:
`scripts/make_iso_forms.py` (kiểm form A render lại vs `X8075.png` IoU ≥ 0.9: 0.925) → `data/raw/iso_symbols_forms/` (thêm un-ignore
vào `.gitignore` như `iso_symbols_clean`). **Kiểm đầu–cuối**: `run_pipeline.py prototypes` với `__dinov2__.pt` + dạng B → 42 hàng
trùng probe (0), dạng B lệch 9e-7; `rank_names` trên P4-val tái lập k4: 41/mean 0.7422, 41/max 0.7998, 41+B/mean 0.7698, 41+B/max
0.8274 (trùng); `vote` lệch 0.0002 (phá hoà khác, không dự thi).

### 2K.6 Luật chấp nhận Colab cho (i) + (ii) — SOẠN 2026-11-19, **CHỐT 2026-11-20 (user duyệt), TRƯỚC KHI CODE / CHẠY**

**Mốc = 2B với 41 ảnh mẫu, gộp trung bình** (hệ thống đề xuất sau §2G; CSV của notebook §15 — **§15 phải chạy trước**). Bốn dòng, cùng
encoder `__dinov2__.pt`, A4, tâm `center_dinov2_a4.pt` (không dựng lại), `++oneshot.exclude_prototypes=[pipeline]`:

| dòng | prototype | `oneshot.aggregate` | vai trò |
|---|---|---|---|
| `base41` | `iso_prototypes_dinov2_a4.pt` (42 hàng) | mean | mốc (= §15 `dinov2_nopipe`) |
| `formB` | `iso_prototypes_dinov2_a4_formB.pt` (43 hàng: + `valve_butterfly@B`) | mean | (i) |
| `max` | 42 hàng | **max** | (ii) |
| `formB_max` | 43 hàng | max | (i) + (ii) |

- File 43 hàng dựng trên Colab bằng `prototypes ... ++oneshot.extra_forms_dir=$REPO_DATA/iso_symbols_forms` (ảnh dạng B đi qua git).
  **Chặn:** 42 hàng đầu phải trùng file 42 (sai khác ≤ 1e-5), `meta.extra_forms == ['valve_butterfly@B']`.
- Mỗi dòng: **τ chọn lại trên val_sel** (luật A / B, δ, nhóm R1a — như 13b; phân bố điểm `max` khác mean) → **P2** (val, `test`, nhóm
  R1b) → **P1 đủ** → **P4** (suy giảm = số chính, sạch tham khảo). Không chọn gì trên P1 / P2 / P4.
- CSV từng crop chép về; so sánh + bootstrap **cục bộ** (`scripts/k4_colab_compare.py`, dùng lại hàm của `f0_colab_compare` /
  `nopipe_compare`): P4 theo **12 tờ**, P1 theo ảnh, P2 theo ảnh trong lớp; B = 1000; "tốt hơn" / "kém hơn" = CI 95% nằm hẳn trên /
  dưới 0, ngược lại "không khác".
- **Kiểm nhất quán (chặn):** cùng tập crop với mốc ở mọi protocol; `formB` chỉ được đổi top-1 ở crop mà `valve_butterfly` (dạng nào)
  vào top-3 của nó hoặc của mốc — lệch khác → dừng, báo user.

**Luật (mỗi dòng so với `base41`):**
- **(ii) `max` được nhận vào cấu hình đề xuất** khi đủ cả:
  1. P4 suy giảm: **không nhóm nào** (S / V / H_chon / H_tu_nhien) "kém hơn" về top-1, **và ít nhất một nhóm "tốt hơn"**; AUROC cos
     P4 không "kém hơn".
  2. P2: nhóm S và V top-1 không "kém hơn"; AUROC cos P2 không "kém hơn".
  3. P1 đủ: Δ macro top-1 `h_in` ≥ −0.01 và ΔAUROC cos ≥ −0.01 (chặn hại phong cách / lớp chưa thấy).
- **(i) `formB` được nhận** khi: không nhóm P4 / P2 nào "kém hơn" về top-1, AUROC cos P4 / P2 không "kém hơn", P1 Δ top-1 ≥ −0.01.
  (Không đòi "tốt hơn": đây là hoàn thiện thư mục theo chuẩn, mức tăng ở butterfly là có sẵn do bộ render dùng cùng SVG.)
- **Kết hợp:** cả (i) và (ii) được nhận **và** `formB_max` thoả luật (ii) so với `base41` → cấu hình đề xuất = `formB_max`. Chỉ một cái
  được nhận → cấu hình đề xuất = cái đó. Không cái nào → 2B 41 giữ nguyên, (i) / (ii) ghi ablation.
- End-to-end (§12 / 13e): không làm vòng này; nếu cấu hình đề xuất đổi, chạy e2e sau với luật riêng.

**Rủi ro ghi trước (từ P4-val, §2K.5):** `max` làm `flanged_connection` 0.61 → 0.38 trên P4-val; mặt bích thuộc **nhóm S** của P4 và là
lớp đông nhất ở đó → nhóm S có thể "kém hơn" → luật (ii).1 có thể trượt dù tổng thể tăng. Nếu trượt đúng vì S: ghi đánh đổi, **không**
sửa luật sau khi xem số (mọi ý "max trừ mặt bích" là hậu kiểm, cần tập quyết định mới).

**Dự báo ghi trước** (P4-val, 18 lớp, cùng renderer): `max` +0.058 gọi tên (valve_general, check_valve_general, interchangeable_disc,
globe tăng; flanged, valve_control, needle giảm); `formB` chỉ đổi butterfly (0.50 → 0.99), V của P4 tăng; `formB_max` ≈ +0.085.
P1-val: `max` top-1 +0.003, AUROC +0.063; `formB` +0.009.

**Giới hạn / ghi rõ khi báo cáo:** động cơ của cả mục §2K đến từ **phân tích lỗi P4** (P4 đã xem chi tiết từng lớp, chưa dùng để chọn);
lựa chọn làm trên P4-val; P4 ở đây là **xác nhận**, P2 là bằng chứng sạch nhất (2B chưa từng nhìn P2 trước §15). Mức tăng của `formB`
trên P4 có sẵn theo cách dựng dữ liệu. Số 2B (42, mean) đã báo ở §2B giữ nguyên làm số ghi trước; dòng mới báo cạnh.

**Code sau khi duyệt:** notebook §16 (16a dựng 43 hàng + kiểm; 16b τ → P2 → P1 cho 3 dòng mới; 16c P4; 16d bảng nhanh; CSV vào
`runs/k4_csv/`), test notebook, RUNBOOK §5k; `scripts/k4_colab_compare.py` (+ test, thử khô trên CSV giả).

**Code ✅ 2026-11-20** (chưa chạy Colab): notebook §16 (cell-30a–30e; test `test_section_16_…`; §15 / §13 tách bằng `_is_16`), RUNBOOK
§5k, `scripts/k4_colab_compare.py` (+ 5 test: luật (i) / (ii), cấu hình cuối, kiểm nhất quán dạng B). Thử khô trên CSV giả (dựng từ CSV
2B): đọc đủ 12 file, kiểm nhất quán chặn đúng khi `formB_max` không bắt nguồn từ `max`, bảng + quyết định in ra.

### 2K.5 Kết quả (i) + (ii) — 2026-11-19 (CPU / MPS vài giây trên cache) → **(i) ĐẠT, (ii) ĐẠT (`max`)**

Kiểm: mốc 41/mean tái lập P4-val 0.742, P1-val 0.477 / AUROC 0.692. Form A render lại vs ảnh mẫu hiện có: cos CL2N **0.991**; form B vs
ảnh mẫu: 0.650 (khác hình thật).

| P4-val (18 lớp) / P1-val | P4-val macro | P1-val top-1 | P1-val AUROC | butterfly | globe |
|---|---|---|---|---|---|
| **41 / mean (mốc 2B)** | 0.742 | 0.477 | 0.692 | 0.50 | 0.19 |
| 41 / vote | 0.720 | 0.467 | 0.751 | 0.52 | 0.28 |
| 41 / **max** | **0.800** | 0.479 | **0.755** | 0.49 | 0.36 |
| **41+B** / mean | 0.770 | 0.485 | 0.693 | **0.99** | 0.19 |
| 41+B / vote | 0.749 | 0.472 | 0.752 | 0.99 | 0.28 |
| 41+B / max (kết hợp, mô tả) | **0.827** | 0.480 | 0.755 | 0.98 | 0.36 |

- **(i) ĐẠT**: butterfly +0.496 (≥ +0.10); không lớp nào khác giảm (tệ nhất 0.000); macro +0.028; P1-val +0.009.
- **(ii) dự thi `max` ĐẠT**: macro +0.058 (≥ +0.03), CI 95% theo tờ [+0.040, +0.077]; P1-val +0.003; P1-val AUROC +0.063 (báo kèm).
  `vote` (mặc định config, OSSR-PID) **kém** mean trên P4-val (−0.022).
- **Cảnh báo — đánh đổi theo lớp của `max` (luật (ii) không đo, ghi để không giấu):** tăng mạnh valve_general 0.27 → 0.91,
  check_valve_general 0.64 → 0.97, interchangeable_disc 0.61 → 1.00, globe 0.19 → 0.36; **giảm** flanged_connection 0.61 → **0.38**,
  valve_control 0.72 → 0.51, valve_needle 0.51 → 0.32. flanged_connection thuộc nhóm **S** của P4 → luật xác nhận Colab phải kiểm theo
  nhóm (như §2E.3), nhóm S có thể tụt.
- Đọc: lấy view tốt nhất (max) giữ chi tiết mà trung bình làm nhoè (đúng chẩn đoán V-ISO), nhưng ký hiệu đơn giản / đối xứng (mặt bích =
  hai vạch) dễ "khớp" một view xoay của ảnh mẫu khác.

**Việc tiếp (chờ user):** cả hai cần code pipeline trước Colab: (a) thư mục ảnh mẫu cho phép **nhiều dạng / lớp** (`tên@dạng` → lớp; hiện
`PrototypeDirectory` cấm trùng tên); (b) `oneshot.aggregate=max` (hiện chỉ `vote` / `mean`); (c) **τ chọn lại trên val_sel** cho cấu hình
mới (phân bố điểm `max` khác mean). Rồi luật chấp nhận Colab (P4 + P1 + P2, theo nhóm, CI theo tờ) viết trước khi chạy.

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
- [x] Chạy Colab §14 (2026-11-18) → áp luật §2E.3: KHÔNG dòng nào thay 2B (f0a: S −0.185; shift: S −0.014, H_tu_nhien −0.030); ablation (§2E.4)
- [x] Phân tích lỗi S6 / S8 / S14 + trả lời của user (2026-11-18, §2F); cách chấm VARIANT: không đổi (không bước nào còn dùng P1-val)
- [x] Thư mục trộn phong cách D0–D3 (2026-11-18, §2F.1–2F.2): phủ không đều phong cách làm hỏng lớp ISO; cùng phong cách cả bộ → 0.96
- [x] P4 + CL2N → đã gộp vào §13c (chạy cùng 2B)
- [x] Điểm 4 quyết (user 2026-11-19): bỏ `pipeline` (nét nối) → 41 ảnh mẫu; luật đo §2G
- [x] Điểm 4 code (2026-11-19): notebook §15, `scripts/nopipe_compare.py` + test (RUNBOOK §5j); 831 test pass
- [ ] Chạy Colab §15 → áp §2G
- [x] Tra paper cross-standard + A so khớp patch DINOv2 (2026-11-19, §2H): KHÔNG ĐẠT (top-1 −0.027, lề +0.078); ablation
- [x] Tầng giữa CNN ImageNet, Gan et al. 2023 (2026-11-19, §2I): KHÔNG ĐẠT (top-1 0.251 vs 0.477, sụp hub); ablation
- [x] §2J ba phần còn sót (Chamfer một chiều, TB patch mực, ghép CLS 4 tầng) (2026-11-19): KHÔNG ĐẠT (tốt nhất cat4/cl2n
  0.471, ≈ mốc); ablation → đóng danh sách tối ưu cách đọc đặc trưng
- [x] §2K bước 1 V-ISO + chẩn đoán + R2 448 px (2026-11-19): cổng V-ISO ↔ P4 TRƯỢT (ρ 0.15); R2 KHÔNG ĐẠT (P1-val −0.054,
  AUROC 0.45); chẩn đoán: gộp trung bình 8 view và ống nối làm mất chi tiết
- [x] P4-val (§2K.2–2K.3, 2026-11-19): 37 tờ / 73 ảnh / 4,832 ký hiệu; cổng đại diện QUA (ρ 0.815); butterfly form B 0.06 vs form A 1.00
- [x] §2K.4–2K.5 (2026-11-19): (i) ảnh mẫu dạng B butterfly ĐẠT (+0.50 butterfly); (ii) gộp `max` ĐẠT (+0.058, CI [+0.040, +0.077]);
  cảnh báo flanged_connection 0.61 → 0.38 với `max`
- [x] Code pipeline (2026-11-19, user duyệt "bước 1"): nhiều dạng / lớp (`tên@dạng`, chấm theo lớp), `oneshot.aggregate=max`,
  `++oneshot.extra_forms_dir`, `scripts/make_iso_forms.py` → `data/raw/iso_symbols_forms/valve_butterfly@B.png` (git theo dõi);
  tái lập k4_probe trên P4-val (mean / max khớp 4 chữ số); 876 test
- [x] Luật chấp nhận Colab §2K.6 (user duyệt 2026-11-20) + notebook §16 + `scripts/k4_colab_compare.py` (thử khô trên CSV giả) +
  RUNBOOK §5k
- [ ] Chạy Colab §15 rồi §16 → `k4_colab_compare.py` → áp §2K.6 (nhận / không nhận (i), (ii) vào cấu hình chính)
- [ ] Đóng băng 2026-12-15

## 7. Nhật ký sửa luật

- **2026-11-17, trước khi có số C3** (lần chạy C3 đang nhúng `dinov2reg`, chưa chấm gì): sau yêu cầu của user "tối ưu phải
  có cơ sở từ paper" (§2C.0), ứng viên được **đưa vào pipeline** ở §2C.3 chỉ còn ViT-B (`dinov2b`, `dinov2breg` — cơ sở
  mạnh, DINOv2 Bảng 4 / 6 / 9). `dinov2reg` (ViT-S + registers; cơ sở yếu cho đặc trưng toàn ảnh, Darcet et al.) chỉ báo
  như ablation, kể cả khi ĐẠT. Ngưỡng của luật §1.2 không đổi.
