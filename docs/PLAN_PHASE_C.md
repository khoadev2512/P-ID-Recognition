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

### Báo cáo Bước 2 (chốt trước)

- **Luôn báo cả bản cũ lẫn bản mới**, cạnh nhau, như ablation. Không thay bản cũ bằng bản mới chỉ
  vì bản mới tốt hơn trên P1 / P4 (đó là test).
- Chỉ nói "cải thiện" khi: P1 đủ tập — top-1 tăng ở ≥ 4/6 lớp `h_in` **và** AUROC không giảm; P4 —
  khoảng tin cậy bootstrap theo 12 tờ của chênh lệch không chứa 0. Ngược lại ghi "không khác".

---

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
- [ ] Bước 2B chạy Colab §13 (P1 đủ, P4, e2e tuỳ chọn); 2A không làm
- [ ] Song song: Điểm 4 trên P2 (chờ user); P4 + CL2N → đã gộp vào §13c (chạy cùng 2B)
- [ ] Đóng băng 2026-12-15

## 7. Nhật ký sửa luật

(trống — mọi thay đổi luật sau 2026-11-16 ghi ở đây kèm lý do)
