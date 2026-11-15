# HANDOFF — Trạng thái dự án & hướng đi (cập nhật 2026-11-12)

File này để **tiếp tục ở session sau** mà không phải làm lại những gì đã thử. Đọc file này
+ `docs/RENDER_RULES.md` + `~/.claude/projects/.../memory/*.md` là nắm toàn bộ.

> **VIỆC TIẾP THEO → `docs/PLAN_STAGE2_ONESHOT.md`** (lập 2026-10-14). User chốt: **code
> hết trước (Phase A: B0, B1, B2, B6a, B6b, B7-code, B4a), rồi mới chạy từng GĐ trên Colab
> (Phase B: R1 train → R2 prototypes → R3 fine-tune nếu cần → R4 eval + chốt số).**
> **Protocol chính = leave-classes-out (P1)**: giữ một nhóm class H ra khỏi train encoder và
> test one-shot + "Other" trên chính chúng — test trên class encoder đã thấy KHÔNG chứng minh
> được one-shot. Danh sách H phải chọn ở B5, TRƯỚC khi train.
> **Nếu hướng chính thất bại → `docs/BACKUP_PLAN.md`**: định nghĩa thất bại (F1–F5), chẩn đoán
> trước khi đổi hướng, các hướng thay thế đã xếp hạng, cây quyết định + mốc thời gian.
>
> **CẬP NHẬT 2026-11-09 — xem "Quyết định 2026-11-09" đầu `docs/PLAN_STAGE2_ONESHOT.md`**, thay
> các chỗ mâu thuẫn trong file này: khung luận văn = phương án B (lõi closed-set DigitizePID +
> so với YOLO một-stage; ISO và "Other" là phần chuyển giao / mở rộng); **template ISO chỉ dùng
> lúc suy luận**; GĐ3 (fine-tune ISO) **bỏ**, thay bằng prototype có stub (B9, cách d — hoãn sang
sau R4, 2026-11-10); R1b chạy
> trước R1a; S/H trên lớp ISO tính tự động từ `symbol_map.csv` + checkpoint.
>
> **CẬP NHẬT 2026-11-12 — xem Mục 6d**: R1b đã chạy tới epoch ≥3 rồi Colab ngắt (checkpoint
> duy nhất = epoch 3); R2 đã chạy trên checkpoint đó. Separation đạt ngưỡng mean nhưng KHÔNG đạt
> ngưỡng max: hai cụm template gộp nhau (van hình nơ có/không vạch đứng; hình tròn bơm/bình/máy
> nén). Nguyên nhân cụm van đã xác nhận: đặc điểm không cộng dồn (chấm lấn át vạch). 10 lớp S tách
> tốt (S–S max 0.389). Template thiếu lề 10% so với crop (code). Đã tách val → `val_sel`/`test`.
> **TRỌNG SỐ LUẬN VĂN ĐỔI 2026-11-12: 70/20/10 → 40/40/20** (lõi C / chuyển sang ISO P4 / mở rộng
> P1 + "Other") vì đề thi C đã bão hoà. Đường găng chuyển sang P4; R1a bắt buộc. Xem PLAN.
> **P2 thu nhỏ XONG 2026-11-12** (`val_sel`, epoch 3): S 7/10 lớp đúng, V gần 0, lề template không
> bản nào thắng. Số liệu + 4 vấn đề chờ user: PLAN "Vấn đề mở 2026-11-12" điểm 1, 4, 5, 6.

---

## 0. TÓM TẮT 1 PHÚT

Thesis: **Detection + Fine-Grained Classification symbol P&ID theo chuẩn ISO 10628-2**.
Pipeline 2 tầng: Stage 1 = YOLOv8 detect coarse (SAHI+WBF), Stage 2 = ResNet-34 + **ArcFace**
fine-grained + **"Other"/open-set gate**.

**HƯỚNG CHỐT CUỐI CÙNG (2026-10-09): one-shot ArcFace, BỎ render synthetic.**
- Train ArcFace trên **DigitizePID** (data thật, nhiều mẫu) → học embedding space.
- ~~**Fine-tune với ISO template** (39 symbol library).~~ Bỏ 2026-11-09: template ISO chỉ dùng
  lúc suy luận (PLAN B7/B9).
- Nhận symbol ISO MỚI (không có trong train) bằng **cosine distance tới ISO prototype** → one-shot.
- Symbol ngoài vocab → **"Other"** (cosine < ngưỡng). ĐÂY LÀ ĐÓNG GÓP CHÍNH.

**ĐÃ BỎ**: toàn bộ hướng render synthetic P&ID (2 cách đều thất bại — xem Mục 4).

**TRẠNG THÁI 2026-10-10**: GĐ2 (one-shot core: prototype directory + cosine match +
voting + "Other" gate) ĐÃ CODE XONG. User ĐÃ CHỐT: **1 encoder TOÀN CỤC**
(`fgc.global_encoder=true`) — đã code xuyên 3 stage. Chưa chạy pytest (máy thiếu
uv/torch). Việc tiếp theo: **train GĐ1 trên Colab** với global encoder.

**TRẠNG THÁI 2026-11-12**: R1b chạy dở (epoch 3 là checkpoint duy nhất), R2 xong trên
checkpoint đó. Việc tiếp theo và kết quả: **Mục 6d**.

---

## 1. ĐÓNG GÓP THESIS (đã chốt với user: cả 3 trục, nhấn mạnh #2)

> **2026-11-12 — trọng số đổi lần nữa: 40/40/20** (lõi closed-set / chuyển sang ISO / mở rộng),
> vì DigitizePID đã bão hoà (R1b val_acc 0.9999). Chi tiết + hệ quả: PLAN "Quyết định 2026-11-12
> — trọng số".
>
> **2026-11-09 — trọng số đổi (phương án B)**: lõi luận văn là **closed-set fine-grained** trên
> DigitizePID 32 lớp, hai-stage so với YOLO một-stage (khớp tên đề tài). Ba trục dưới đây chuyển
> thành phần "chuyển sang ISO" (#3) và "mở rộng" (#1, #2). So với OSSR-PID: kế thừa khung
> one-shot; khác ở cổng "Other", thêm ký hiệu không train lại (giao thức leave-classes-out), và
> bộ đánh giá ISO cả tờ (OPEN100-ISO). Pixel vs graph là đơn giản hoá, không phải đóng góp.

1. **Pixel-based one-shot** (ResNet+ArcFace trên ảnh) — đơn giản hơn OSSR-PID (dùng graph/DGCNN).
2. **"Other"/open-set rejection** ← ĐIỂM CHÍNH, MỚI NHẤT. Cosine threshold reject symbol
   ngoài vocab. CHƯA paper P&ID nào làm (OSSR-PID closed-set; conclusion ghi zero-shot=future work).
3. **Gắn ISO 10628-2** — framework nhận symbol theo chuẩn quốc tế, mở rộng bằng template
   (thêm symbol = thêm 1 template, KHÔNG train lại).

**Khác các paper**:
- vs **OSSR-PID** (arXiv 2109.03849): pixel vs graph; CÓ "Other"; gắn ISO (họ dùng symbol tự chế).
- vs **DigitizePID** (2109.03794): one-shot + open-set vs fully-supervised.
- vs **SynthPID** (2604.16513) / **PID2Graph**: họ làm data-gen/graph, ta làm recognition method.

**Trung thực**: luận văn thạc sĩ KHÔNG cần breakthrough. Đóng góp = áp dụng + cải tiến + "Other",
đo đạc trung thực, so OSSR-PID (cùng dataset DigitizePID). "Other" là trục mạnh/mới nhất.

---

## 2. CODE ĐÃ CÓ (đừng viết lại)

Pipeline gốc (src/pipeline, src/utils) — 2 package flat, Hydra config `configs/config.yaml`.

**ArcFace + "Other" gate ĐÃ CODE XONG (từ các session trước):**
- `src/utils/model.py`: `ArcMarginProduct` (angular margin head), `ArcFaceModel` (backbone→
  embedding→class centers), `build_model` rẽ nhánh theo `fgc.metric_learning`.
- `src/pipeline/fgc_train.py`: train ArcFace per fgc_group, lưu `class_centers` vào checkpoint.
  Có `_maybe_resume` (fgc.resume_from). Pass labels vào head khi train (margin).
- `src/pipeline/fgc_infer.py`: "Other" gate — cosine tới nearest center < `fgc.route.other_min_cosine`
  → `OTHER_FINE_ID (-2)`. `OTHER_CLASS_NAME="__other__"`. Softmax quyết nhãn, cosine gác cổng Other.
- `src/utils/classmap.py`: `OTHER_FINE_ID=-2`, `UNREFINED_FINE_ID=-1`.
- `src/pipeline/evaluation.py`: eval FGC + `other_gate.false_reject_rate` (ArcFace).
- `src/pipeline/build_classes.py`: ĐỌC CỜ `fgc` từ COCO category (ISO) nếu có, fallback
  "coarse >1 fine" (DigitizePID). QUAN TRỌNG: với vocab ISO chi tiết, "fine>1" over-trigger
  (pump 4 fine nhưng khác hình rõ) → phải dùng cờ fgc thủ công.
- `src/pipeline/detect_infer.py`: **WBF bug ĐÃ FIX** — gộp mọi tile thành 1 WBF model (trước
  chia score cho ~54 tile → score 0.02; sau giữ 0.9). Xem test_detect_infer_wbf.py.
- `configs/config.yaml`: `fgc.metric_learning=true` (default), `fgc.arcface.{m,s,embed_dim}`,
  `fgc.route.other_min_cosine=0.35` (PLACEHOLDER, chưa calibrate), `detector.resume/resume_dir`,
  `fgc.resume_from`. `fgc.groups=[valve,blind_disc,heat_exchanger,instrument]`.

**GĐ2 ONE-SHOT CORE ĐÃ CODE XONG (session 2026-10-10):**
- `src/utils/oneshot.py` — LÕI one-shot, KHÔNG phụ thuộc pipeline/:
  - `ViewSpec` (scales × orientations) + `make_views`/`refit_box`/`rotate_quarter`.
    QUAN TRỌNG: `scale` là HỆ SỐ BOX-FIT, không phải resolution (model input cố định
    128px nên multi-scale theo pixel là no-op). s>1 pad trắng quanh symbol (bbox lỏng,
    giống YOLO crop), s<1 crop vào (bbox chặt). Đây mới đúng domain gap giữa ISO
    template (bbox vector sát) và detector crop.
  - `embed_images` (L2-norm sẵn) + `aggregate_views` (mean rồi RE-NORMALIZE — không
    renorm thì prototype multi-view có norm nhỏ hơn single-view → không dùng chung
    ngưỡng cosine được).
  - `PrototypeDirectory` {fine_name → unit embedding} + cột `coarse` song song +
    `meta` (provenance). `match()`, `topk()`, `match_views()`, `subset_by_coarse()`,
    `save()/load()` (có `format_version`).
  - `build_prototypes()` — views → embed → mean → unit vector, giữ thứ tự template.
- `src/pipeline/iso_prototypes.py` — Hydra stage `pid-iso-prototypes`: đọc vocab.csv +
  iso_symbols_clean → đẩy qua encoder ArcFace ĐÓNG BĂNG → lưu prototype directory.
  Log "prototype separation" (mean/max cosine giữa các prototype + cặp dễ lẫn nhất) —
  check rẻ nhất xem encoder GĐ1 có transfer sang ISO hay không, KHÔNG cần label ISO.
- `src/utils/model.py`: `load_arcface_encoder(cfg, ckpt, device)` — dựng lại
  ArcFaceModel từ checkpoint pid-fgc-train, eval mode. num_fine đọc từ `head.weight`
  (không lấy từ config → không lệch). Báo lỗi rõ nếu ckpt là linear-head
  (metric_learning=false → không có embedding space) hoặc lệch embed_dim.
- `src/utils/classmap.py`: `OTHER_CLASS_NAME="__other__"` CHUYỂN về đây (cạnh
  OTHER_FINE_ID) vì cả 2 gate dùng; fgc_infer re-export để call site cũ không vỡ.
- `configs/config.yaml`: section `oneshot.*` (symbols_dir, vocab_csv, encoder_ckpt,
  prototypes, fgc_groups_only, coarse_filter, background, template_views, query_views,
  aggregate, other_min_cosine).
- Tests: `tests/test_oneshot.py` (~45 test, dùng `_FakeEncoder` — projection tuyến tính
  cố định của pixel → test TOÀN BỘ logic one-shot KHÔNG cần train, KHÔNG cần timm),
  `tests/test_iso_prototypes.py` (parse vocab + build end-to-end trên 39 symbol ISO
  THẬT với ArcFace chưa train → kiểm wiring, không kiểm chất lượng).

**QUYẾT ĐỊNH THIẾT KẾ GĐ2 (đã cân nhắc, đừng bàn lại):**
1. **Prototype = 1 view (upright) mặc định.** Template ISO là artwork chuẩn, đã đúng
   hướng → one-shot trung thực. Mở rộng `template_views.scales` chỉ để làm ablation.
   Augment mạnh (rot ±20°, shear) là việc của GĐ3 (fine-tune), KHÔNG nhét vào prototype.
2. **Query views VOTE, không average.** Symbol xoay 90° chỉ khớp template ở 1 trong 4
   orientation; mean embedding bị 3 view vô nghĩa kéo lệch, majority thì sống.
3. **Chỉ view vượt ngưỡng mới được bỏ phiếu** (khác OSSR-PID một chút, có lý do):
   với symbol bất đối xứng, 3 orientation sai là nearest-neighbour nhiễu và có thể
   thắng 2-1 ngược lại view đúng. Gate đã là phát biểu "đây là match thật" nên dùng
   luôn làm điều kiện được bỏ phiếu → không cần ngưỡng thứ hai, và cosine thắng luôn
   >= ngưỡng. Không view nào vượt → "Other" (n_votes=0, báo cosine cao nhất để calibrate).
4. **`oneshot.other_min_cosine` TÁCH RIÊNG khỏi `fgc.route.other_min_cosine`.** Một cái
   đo khoảng cách tới class center ĐÃ TRAIN, cái kia tới prototype 1-template — 2 phân
   bố cosine khác nhau, KHÔNG dùng chung số. Cả hai vẫn là PLACEHOLDER 0.35.
5. **`coarse_filter`** để ghép đúng encoder per-group với họ ISO tương ứng
   (valve.pt ↔ coarse valve+check_valve = 11 prototype), vì không gian train để tách
   valve không nói gì về vessel.

**GLOBAL ENCODER ĐÃ CODE (session 2026-10-10, theo quyết định Mục 6b):**
`utils/classmap.py` (GLOBAL_GROUP sentinel + fine_of_group), `pipeline/fgc_train.py`
(`training_groups()`), `pipeline/extract_crops.py` (`_wanted_anns()`),
`pipeline/fgc_infer.py` (chặn + hướng dẫn), `configs/config.yaml`
(`fgc.global_encoder: false` — GĐ1 phải override `=true`).

**Resume (Colab ~6h sessions) ĐÃ CODE**: detector.resume (YOLO last.pt), fgc.resume_from.

**Tests**: ~109 pass trước session này + ~90 test mới (test_oneshot, test_iso_prototypes,
test_global_encoder)
CHƯA CHẠY ĐƯỢC (xem Mục 7: máy session 2026-10-10 không có uv/torch) — phần logic
torch-free đã verify tay trên data thật, phần torch cần `uv run pytest` để xác nhận.
test_arcface, test_resume, test_detect_infer_wbf, test_convert_yolo_to_coco,
test_extract_iso_symbols, test_render_pid(_replace), test_visualize_detections.

**Scripts** (scripts/):
- `convert_yolo_to_coco.py` — DigitizePID YOLO → COCO (8 coarse, Symbol_N). DÙNG CHO HƯỚNG CŨ.
- `extract_iso_symbols.py` — cắt 39 ISO symbol từ PDF (vector bbox + override tay + loại legend).
- `extract_bg_patches.py` — [HƯỚNG RENDER ĐÃ BỎ] cắt background patch.
- `render_pid.py`, `render_pid_replace.py` — [HƯỚNG RENDER ĐÃ BỎ] 2 renderer, đều không ổn.
- `visualize_detections.py` — vẽ box + legend + lọc score (--score-thr, --fill, --show-text).

**Entrypoints** (pyproject [project.scripts]): thêm `pid-iso-prototypes`.

---

## 3. DATA

- **`data/raw/digitize_pid_yolo/`** (1.3G) — DigitizePID: ảnh JPG 7168×4561 + YOLO label,
  400 train/100 val, 32 class (0-indexed, cân bằng ~1700/class). **DÙNG ĐỂ TRAIN** (hướng one-shot).
  Nguồn HF `hamzas/digitize-pid-yolo`.
- **`data/raw/iso_symbols_clean/`** (39 PNG; **2026-11-12: 42 PNG `<reg_number>.png` + `symbols.csv`**,
  khớp 42 dòng vocab.csv, không lớp nào thiếu file) — ISO symbol library (cắt từ ISO_10628-2 PDF).
  User đã crop tay ~20 cái lỗi → ~36/39 đẹp. **DÙNG LÀM TEMPLATE REFERENCE** (one-shot).
  THEO GIT (un-ignored trong .gitignore).
- **`data/raw/iso_symbols/vocab.csv`** — 39 symbol: reg_number, coarse, fine_name (tên nghĩa:
  valve_gate...), pid_group, fgc_group (yes/no THỦ CÔNG), desc. 12 coarse, 5 fgc_group.
- `data/raw/digitize_pid_npy/` (30M) — bản .npy gốc (có graph/lines/words), giữ cho graph sau.
- `data/canonical/` — DigitizePID đã convert COCO (8 coarse, Symbol_N). HƯỚNG CŨ.
- `data/canonical_iso/` — [RENDER ĐÃ BỎ] ĐÃ XÓA (kiểm 2026-10-14).
- **Kiểm 2026-10-14**: `data/canonical/` = DigitizePID gốc (convert_yolo_to_coco, KHÔNG phải
  bản render): 500 ảnh 7168×4561, 59,498 ann, 32 category `Symbol_N` với supercategory =
  8 coarse, không có cờ `fgc` → build_classes dùng fallback "coarse >1 fine" (đúng).
  `data/derived/` chỉ có `yolo_coarse/`. `data/raw/real/` và `synthetic/` trống.
  `bg_patches/` (28M) là đồ của hướng render đã bỏ, xóa được.
- Ảnh/data KHÔNG theo git (.gitignore `data/*`), trừ iso_symbols_clean + vocab.
- **Split DigitizePID (kiểm 2026-11-12)**: `data/canonical/digitizepid_splits.json` = {id ảnh →
  "train"/"val"}, chia THEO ẢNH 400/100, không rò rỉ (kiểm theo tiền tố id trong tên crop
  `<id ảnh>_<thứ tự>.png`). KHÔNG có test → tách 100 ảnh val thành `val_sel` 50 / `test` 50,
  seed 20261112, file `data/canonical/digitizepid_val_test.json` (đã lưu Drive, **CẦN commit**).
  Mọi lớp ≥142 crop ở mỗi nửa (Symbol_21 ~550). Crop toàn cục: 47,658 train / 11,840 val.
- **Colab**: data lên Google Drive, code qua git. Fix I/O: copy tiles/crops Drive→/content
  local trước train (Drive đọc 27k file chậm → cháy quota). Xem notebooks/train_stage1_colab.ipynb.

**Bảng coarse map (DigitizePID 32 class → ISO)** — `docs/symbol_mapping/symbol_map.csv`:
valve c0-15, blind_disc c16-18, reducer c19, flange c20, heat_exchanger c21-22,
flow_direction c23, safety_valve c24, instrument c25-31.
→ ISO coarse: reducer/flange/flow_direction → piping; instrument → agitator (ISO không có bubble).

---

## 4. NHỮNG GÌ ĐÃ THỬ VÀ THẤT BẠI (ĐỪNG LÀM LẠI)

### 4a. Render synthetic — Cách A (patch + random placement) → BỎ
`render_pid.py`: cắt background patch từ DigitizePID + đặt symbol ISO random + L-pipe + tag +
degrade. **Vấn đề**: (1) symbol đè lên legend/text của background (đã fix bằng _region_is_clear
nhưng vẫn không tự nhiên); (2) topology GIẢ (random, không giống P&ID thật). 400 ảnh, 6038 ann.

### 4b. Render synthetic — Cách B (replace in-place, SynthPID-style) → BỎ
`render_pid_replace.py`: giữ ảnh DigitizePID thật, xóa symbol cũ (tô trắng bbox) + dán ISO cùng
loại. 500 ảnh, 59498 ann. **Vấn đề user chỉ ra**: symbol ISO và bbox DigitizePID KHÔNG ăn khớp
tỷ lệ — có symbol ghép tốt, có symbol méo/lệch/để lại ô trắng quanh. Không đạt chất lượng.

### 4c. Kết luận render: GHÉP SYMBOL TỔNG HỢP KHÓ ĐẠT CHẤT LƯỢNG CAO.
SynthPID phải viết cả paper CVPR mới làm tốt (topology-preserving, cần seed real). Với deadline
~3 tháng, render tốt là bẫy thời gian. → BỎ render, dùng one-shot (OSSR-PID chứng minh không cần).

### 4d. Các quyết định data khác đã chốt (đừng bàn lại):
- DigitizePID là MỚ HỖN HỢP CHUẨN (ISO + ISA + tool khác như edrawmax). Không thuần ISO.
- Class-agnostic Stage 1: GIỮ 8-coarse (P=R=0.996 trên DigitizePID val), agnostic để dành
  ablation sau. Xem memory stage1-agnostic-decision.
- ISO không có instrument bubble (thuộc ISA-5.1) → dùng agitator thay.
- Stage 1 train DigitizePID đạt mAP50=0.995 (synthetic dễ + 8 coarse tách biệt rõ).

### 4e. Mở lại nhánh render dạng vẽ lại vector trên topology OPEN100, có cổng G1 — G1 ĐẠT 2026-11-02
Khác 4a/4b: không dán ảnh ký hiệu lên nền có sẵn, mà vẽ lại cả tờ PID2Graph OPEN100 (giữ nguyên
topology) bằng template vector ISO 10628-2 (BK-2020) có khai báo cổng; ống dừng đúng tại cổng.
G1 (tờ 3): 4 tiêu chí tự động + user duyệt ảnh đều đạt. Chi tiết, số liệu 12 tờ và các bước tiếp theo
(V5b bình + agitator → V6–V8): `docs/PLAN_ISO_RENDER_PID2GRAPH.md`. Không đụng Stage 1/2.

---

## 5. PAPER THAM CHIẾU CHÍNH (đã đọc kỹ)

**OSSR-PID** (arXiv 2109.03849, Paliwal/Sharma/Vig, TCS Research) — QUAN TRỌNG NHẤT.
PDF đã lưu: `~/.claude/projects/.../tool-results/webfetch-1791527185350-ojh5rs.pdf`.
- One-shot symbol recognition trên CHÍNH DigitizePID (Dataset-P&ID), 1 template/class.
- Method: image trace (Potrace) → path sampling → graph → **DGCNN + ArcFace** → cosine match.
- Table I: DGCNN+ArcFace acc **85.78%** vs cross-entropy **77.07%** (ArcFace +8.7%). DGCNN+
  ResNet34+ArcFace **85.98%** (graph+pixel gần bằng pixel → KHÔNG cần graph phức tạp).
- Table III: train one-shot → test P&ID THẬT (12 real sheets), F1 0.90-0.99. ← kịch bản của ta.
- Conclusion: vẫn cần train khi thêm class mới; zero-shot = future work (← chỗ "Other" của ta mới).

**3 kỹ thuật MƯỢN từ OSSR-PID (áp dụng Stage 2 pixel)**:
1. One-shot: ISO template → embedding reference → nhận bằng cosine distance.
2. **Per-subpart augment**: affine RIÊNG từng vùng symbol (không augment cả ảnh) cho symbol
   giống nhau. rotation -20..20°, scale 0.9-1.1, shear -0.05..0.05.
3. **Multi-scale + voting infer**: crop mỗi vùng 2 scale (min width 300 & 600px) × 4 orientation
   = 8 embeddings, cosine với directory, majority vote.

Paper khác: SynthPID (2604.16513, data-gen topology-preserving, cần seed real), PID2Graph
(2411.13929, graph extraction), Few-Shot Symbol Detection 2024 (tandfonline, paywall chưa đọc).

---

## 6. PLAN TRIỂN KHAI ONE-SHOT (5 giai đoạn) — BƯỚC TIẾP THEO

User chốt: **train DigitizePID + fine-tune ISO**. Làm từng giai đoạn, verify từng cái.

### GĐ1 — Train embedding base (DigitizePID) [CẦN CHẠY — code đã đủ]
- `pid-prep-crops fgc.global_encoder=true` (BẮT BUỘC chạy lại, xem Mục 6b) rồi
  `pid-fgc-train fgc.global_encoder=true fgc.metric_learning=true` → `__global__.pt`
  (1 head, 32 class DigitizePID).
- CẦN THÊM (chưa code): per-subpart augment (OSSR-PID kỹ thuật #2) vào dataset.py —
  affine RIÊNG từng vùng, rotation -20..20°, scale 0.9-1.1, shear -0.05..0.05.
  Hiện `fgc.aug` chỉ có hflip/rotate/jitter áp lên cả crop.
- Chạy Colab (DigitizePID 7168px nặng, cần GPU + tiling + fix I/O).

### GĐ2 — ISO template → prototype (CỐT LÕI one-shot) ✅ CODE XONG 2026-10-10
- `pid-iso-prototypes oneshot.encoder_ckpt=runs/<run>/fgc_checkpoints/<group>.pt`
- 39 ISO symbol → 39 vector prototype, lưu `data/derived/iso_prototypes.pt`.
- Chi tiết + 5 quyết định thiết kế: xem Mục 2.
- CÒN LẠI của GĐ2: chạy thật sau khi có checkpoint GĐ1, đọc log "prototype separation"
  (nếu max off-diagonal cosine ~1.0 → 2 symbol ISO không tách được, one-shot giữa chúng
  là tung xu; đó là tín hiệu encoder GĐ1 chưa transfer).

### GĐ3 — Fine-tune với ISO template — ❌ BỎ 2026-11-09
- Trái với "template ISO chỉ dùng lúc suy luận"; thay bằng prototype có stub (PLAN B9), không train.

### GĐ4 — Inference one-shot + "Other" [lõi đã có, CẦN WIRE vào fgc_infer]
- Đã có: `PrototypeDirectory.match_views(...)` (multi-scale + voting + gate) và
  `subset_by_coarse(...)` để giới hạn theo coarse class Stage 1 đã đoán.
- CẦN LÀM: thêm nhánh one-shot vào `fgc_infer._classify_crop` — thay softmax-argmax
  trên head đã train bằng cosine tới ISO prototype. Dự kiến cờ `fgc.oneshot=true` +
  đọc `oneshot.prototypes`; `_safe_fine_id` cần map ISO fine_name → id (hiện map qua
  classmap của DigitizePID, VOCAB KHÁC NHAU → đây là chỗ cần quyết).
- Calibrate `oneshot.other_min_cosine` trên val.

### GĐ5 — Eval [cần mở rộng evaluation.py]
- Accuracy nhận ISO, "Other" precision/recall (cần tập symbol out-of-vocab để test),
  cross-domain (train DigitizePID → nhận ISO), so OSSR-PID.

**ĐÃ CHỌN GĐ2 và làm xong (2026-10-10).** Thứ tự còn lại: GĐ1 (train Colab) → chạy GĐ2
thật → GĐ4 (wire fgc_infer) → GĐ3 (fine-tune ISO, chỉ làm nếu GĐ2 cho separation kém)
→ GĐ5 (eval).

---

## 6b. ĐÃ CHỐT: 1 ENCODER TOÀN CỤC (`fgc.global_encoder=true`) — 2026-10-10

**Lý do** (đừng bàn lại): per-group head học các embedding space RIÊNG BIỆT, cosine
giữa 2 space khác nhau KHÔNG so sánh được → prototype directory duy nhất và "Other"
toàn vocab đều đòi mọi class ở CÙNG MỘT space (như OSSR-PID). Thêm nữa, vocab ISO
(12 coarse) ≠ coarse DigitizePID (8): các họ ISO pump/vessel/compressor/motor/piping/
fitting KHÔNG có encoder per-group nào tương ứng.

**Cờ `fgc.global_encoder` xuyên 3 stage (phải nhất quán cả 3):**
| Stage | khi `global_encoder=true` |
|---|---|
| `pid-prep-crops` | cắt crop cho MỌI symbol, không chỉ class thuộc fgc_group. Class không có crop = thiếu hẳn khỏi embedding space. **Lưu ý: DigitizePID phải chạy LẠI prep-crops**, vì lần trước chỉ cắt 28/32 class (thiếu reducer/flange/flow_direction/safety_valve). |
| `pid-fgc-train` | train 1 head duy nhất, size = toàn bộ fine vocab → checkpoint `__global__.pt` |
| `pid-fgc-infer` | ~~TỪ CHỐI (NotImplementedError)~~ → 2026-11-13: closed-set với `__global__.pt` (B6c); `++fgc.oneshot=true` = one-shot với prototype ISO (B6a, `pipeline/oneshot_infer.py`) |

**Code**: `utils/classmap.py` thêm sentinel `GLOBAL_GROUP="__global__"`, và
`fine_of_group(GLOBAL_GROUP)` trả về TOÀN BỘ fine vocab → nhờ vậy dataset, train loop
và cách đặt tên checkpoint đều đi đường global mà không cần code path thứ hai.
`fgc_train.training_groups()` quyết định train gì. `extract_crops._wanted_anns()` quyết
định cắt gì. Tests: `tests/test_global_encoder.py`.

**Lệnh GĐ1 (Colab)**:
```
pid-prep-crops  fgc.global_encoder=true
pid-fgc-train   fgc.global_encoder=true fgc.metric_learning=true
pid-iso-prototypes oneshot.encoder_ckpt=runs/<run>/fgc_checkpoints/__global__.pt
```

**CÒN MỞ — vocab nào là vocab cuối của output?** GĐ4 trả về ISO fine_name, nhưng
`classmap`/`classes.yaml` hiện là vocab DigitizePID (Symbol_N). Cần quyết: sinh
`classes.yaml` từ vocab ISO, hay để prototype directory tự làm vocab của Stage 2.
Đây là chỗ `fgc_infer._safe_fine_id` sẽ vướng khi wire GĐ4.

**CÒN MỞ — `evaluation.py` lặp theo `classmap.fgc_groups`** nên cũng chưa hiểu global
encoder; sẽ phải mở rộng ở GĐ5 (hiện fgc_infer chặn trước nên user gặp lỗi rõ ràng,
không phải lỗi mơ hồ).

---

## 6c. ĐÃ CHỐT: DANH SÁCH H (2026-11-11, trước mọi lượt train Stage 2)

Chốt **trước khi có bất kỳ kết quả nào**; không sửa sau khi đã thấy số (PLAN, mục "Protocol").
Nguồn: trang kiến trúc "Framework hai stage" (https://claude.ai/artifact/BtJxRzz8PdFjXVhciZiCo6),
mục "Chốt danh sách H". `check_b5.py`: 0 lỗi, 0 cảnh báo.

**Chia A — bộ chính** (ghi trong `configs/config.yaml` `oneshot.holdout`, dùng cho R1a):

| Vai trò | Lớp | Lý do / hàng xóm còn trong train |
|---|---|---|
| `h_in` | S3 valve_globe | S1 gate |
| `h_in` | S10 valve_needle | S1 gate, S6 butterfly |
| `h_in` | S5 valve_ball | S1 gate, S14 control |
| `h_in` | S4 valve_general | S1 gate, S8 check_valve, S14 control |
| `h_in` | S18 interchangeable_disc | S17 open_disc |
| `h_in` | S19 interchangeable_disc_open | S17 open_disc |
| `h_extra` | S2, S13 (theo S5); S12 (theo S4) | VARIANT cùng ký hiệu ISO: bỏ khỏi train, không thi |
| `h_out` | S9 valve_diaphragm, S15 valve_rotary_generic (khó: trông như van) | S7, S16 vẫn train |
| `h_out` | S27 isa_bubble_ro, S30 isa_square_sta (dễ) | S26, S28, S29, S31, S32 vẫn train |

- R1a còn 19/32 lớp: S1, S6, S7, S8, S11, S14, S16, S17, S20–S26, S28, S29, S31, S32.
- Tiêu chí: giữ hàng xóm khó, không rút cạn họ (S21, S24 không chọn được; blind disc rút tối đa
  2), chặn rò rỉ VARIANT (`h_extra`), `h_out` có cả khó lẫn dễ, đủ mẫu (mỗi lớp ~1,300–1,500
  train, ~300–380 val).
- Trên OPEN100-ISO, cả 6 lớp `h_in` đều có mẫu → nhóm "H chọn" của P4 (R1a) có 7,958 instance
  (sạch + suy giảm).
- UNSURE (S11, S25) vẫn train (không thuộc H), chỉ bị loại khỏi mọi tập test.
- `h_extra`: notebook Colab dựng R1a với `fgc.exclude_classes = h_in + h_out + h_extra` và cell
  finalize kiểm cả `h_extra` (2026-11-11). `check_b5.py` chưa kiểm luật VARIANT (B5b); chia A đã
  kiểm tay.

**Chia B — bộ dự phòng, đăng ký trước** (chỉ chạy khi còn quota Colab; nếu chạy, báo cáo trung
bình ± độ lệch trên hai bộ):

| Vai trò | Lớp |
|---|---|
| `h_in` | S1 valve_gate, S10 valve_needle, S5 valve_ball, S17 open_disc, S18 interchangeable_disc (giữ S3, S4 trong train) |
| `h_extra` | S2, S13 (theo S5) |
| `h_out` | S7 valve_manual, S16 valve_rotary_generic_nc (khó); S26 isa_bubble_field, S32 isa_box_lg (dễ) |

- `h_in` 5 lớp (dưới mức 6–8 của plan: blind disc rút tối đa 2, S3/S4 cố ý giữ lại để đổi vai
  với chia A). `check_b5.py` sẽ cảnh báo số lượng, không phải lỗi.
- Chạy chia B = đổi `oneshot.holdout` sang bộ này cho một lượt R1a riêng (run_dir riêng), rồi
  trả lại chia A. Danh sách cố định từ hôm nay, không chọn lại sau khi thấy số chia A.

---

## 6d. KẾT QUẢ R1b (dở) + R2 TRÊN EPOCH 3 — 2026-11-11/12

### R1b — closed-set bão hoà ngay từ epoch 1
```
epoch 1  train_loss=1.5901 train_acc=0.8576 val_loss=0.0015 val_acc=0.9997  (best, saved)
epoch 2  train_loss=0.0457 train_acc=0.9947 val_loss=0.0048 val_acc=0.9997
epoch 3  train_loss=0.0449 train_acc=0.9949 val_loss=0.0004 val_acc=0.9999  (best, saved)
```
Colab ngắt sau đó. `runs/r1b/2026-11-11_13-17-16/fgc_checkpoints/__global__.pt` = **epoch 3**.
- **Không phải lỗi**: đã loại rò rỉ (split theo ảnh, giao nhau 0). DigitizePID tổng hợp + cân bằng →
  closed-set quá dễ. Đề thi C chỉ còn là mốc ("closed-set đã bão hoà"), không phải bằng chứng.
- **Hệ quả cho code**: chọn checkpoint theo `val_acc` đã chết từ epoch 3 (không thể vượt 0.9999).
  ✅ **Đã quyết + sửa 2026-11-13** (PLAN "Quyết định 2026-11-13 — chọn checkpoint"): R1a và R1b so
  ở **cùng epoch 3**; không resume R1b; `fgc_train` ghi `.last.pt` mỗi epoch và `.ep<N>.pt` theo
  `++fgc.save_epochs`; R1a train 5 epoch.
- Đã copy sang `runs/encoders/__global__.pt` và `__global_ep3__.pt` (bản giữ lại, vì
  `__global__.pt` sẽ bị ghi đè sau resume).
- ~~**Bẫy notebook**: `finalize()` định nghĩa trong ô 5b~~ → ✅ 2026-11-13: định nghĩa ở cell 4b
  riêng, lấy checkpoint epoch `CMP_EPOCH` của run.

### R2 — prototype separation (epoch 3, 1 view, imgsz 128, chưa (d))
`mean off-diagonal cos=0.245, max=0.989 (pump_general vs vessel_spherical)`
- 861 cặp: >0.95: 7 · >0.9: 12 · >0.85: 17 · >0.8: 22.
- Trung bình trong nhóm thấp: valve 0.282 (36 cặp), vessel 0.241, heat_exchanger 0.212,
  blind_disc 0.256, piping 0.286 → phần lớn không gian dùng được.
- **B4: mean ĐẠT (< 0.6), max KHÔNG ĐẠT (> 0.8).** Lỗi dồn vào HAI CỤM:

**Cụm 1 — hình tròn (toàn H tự nhiên):** pump_general, vessel_spherical, compressor_vacuum,
pump_diaphragm (đôi một 0.959–0.989), viewing_glass (~0.88 với cả 4); thêm pump_centrifugal —
compressor_centrifugal 0.939. Encoder thấy "hình tròn", bỏ qua đường bên trong.
Rủi ro thêm: bong bóng ISA (h_out, phải ra "Other") cũng là hình tròn → dễ khớp nhầm cụm này.

**Cụm 2 — van hình nơ (có lớp S: valve_globe ← S3):** valve_globe — valve_butterfly **0.981**; check_valve_globe —
check_valve_swing 0.921; check_valve_swing với globe/butterfly 0.90–0.91. Mỗi cặp khác nhau đúng
ở **vạch đứng mảnh xuyên chấm tròn**.

### Nhóm S/V/H của các cặp sát nhau (R1b, đối chiếu `symbol_map.csv`; S/V/H = 10/6/26)
| Cặp | cos | Nhóm |
|---|---|---|
| valve_globe — valve_butterfly | 0.981 | S — V (S6 = VARIANT) |
| check_valve_globe — check_valve_swing | 0.921 | H — H (cả hai có mẫu trên OPEN100-ISO) |
| check_valve_general — breather_valve | 0.918 | V — H |
| valve_globe / butterfly — check_valve_swing | 0.90–0.91 | S/V — H |
| valve_general — check_valve_general | 0.857 | S — V |
| cụm tròn (bơm, máy nén, bình cầu, kính quan sát) | 0.88–0.989 | H — H |

**Không có cặp S–S nào sát: S–S gần nhất 0.389** (valve_general — valve_needle). Lỗi dồn vào V/H,
tức phần "chuyển sang ISO" và "mở rộng", không phải lõi. Theo B4 (max ≥ 0.9 → xem PNG): đã xem,
các cặp KHÔNG giống nhau với mắt người → giới hạn chuyển giao của encoder.

### Chẩn đoán cụm van — ĐÃ XÁC NHẬN 2026-11-12: đặc điểm không cộng dồn
| Cặp | khác nhau ở | cos |
|---|---|---|
| valve_gate — valve_general | vạch (không chấm) | **0.133** |
| valve_globe — valve_general | chấm | 0.262 |
| valve_globe — valve_gate | chấm + vạch | 0.122 |
| valve_globe — valve_butterfly | vạch, khi ĐÃ có chấm | **0.981** |
- DigitizePID có "nơ + vạch" (S1 gate) và "nơ + chấm" (S3 globe) nhưng không lớp nào có cả hai →
  encoder học vạch và chấm riêng lẻ, không học rằng vạch vẫn có nghĩa khi có chấm. Cặp
  check_valve_globe/swing cùng khuôn (khác nhau ở vạch, cả hai có chấm).
- Phát hiện cho luận văn: encoder chỉ chuyển được đặc điểm từng mang tính phân biệt trong dữ liệu
  train, và các đặc điểm đó KHÔNG cộng dồn thành tổ hợp chưa thấy.
- ĐÃ LOẠI: tiền xử lý (ở 128 px chi tiết còn rõ); giả thuyết "vạch = ống" (S1/S4 tách 0.133 →
  encoder có nhìn vạch); rò rỉ; mất cân bằng (`balanced_sampler: true` + dữ liệu cân bằng).
- (d) không giải quyết cụm van. Cụm tròn: nhiều khả năng cùng cơ chế, chưa kiểm riêng.
- `build_eval_transform` = `Resize((128,128))` ép vuông, không giữ tỉ lệ — áp như nhau cho crop và
  template nên không gây lệch.

### Lề template ≠ lề crop (code, 2026-11-12)
- `iso_prototypes.py` KHÔNG thêm lề: template nhúng sát bbox vector (ký hiệu chiếm 100% khung),
  crop từ B1c = box + 10% mỗi bên (~83%). Query view 1.25 còn nới thêm.
- `template_views.scales=[1.2]` (= +10% mỗi bên theo `refit_box`, khớp `CROP_MARGIN_FRAC`):
  separation mean 0.240, max 0.990 (valve_globe vs valve_butterfly) — gần như y hệt scale 1.0.
  → lề KHÔNG đổi separation; chọn 1.0 hay 1.2 phải dựa vào P2 thu nhỏ trên `val_sel`.
- File: `runs/encoders/iso_prototypes.pt` (scale 1.0), `iso_prototypes_m12.pt` (scale 1.2).
- Nếu 1.2 tốt hơn: đổi mặc định = quyết định mới, sửa quyết định thiết kế GĐ2 số 1 (Mục 2).

### Khung luận văn — ĐÃ QUYẾT 2026-11-12: 40/40/20
Đề thi C bão hoà (Stage 2 0.9999, Stage 1 coarse mAP50 0.995) → user đổi trọng số từ 70/20/10 sang
**40 lõi C / 40 chuyển sang ISO (P4, P2) / 20 mở rộng (P1, "Other")**. C vẫn báo cáo, nói rõ đã bão
hoà, thêm mAP50:95. Còn mở: có chấm C trên ảnh DigitizePID suy giảm không. Hệ quả: đường găng =
B6a/B6b/B6d (P4); R1a bắt buộc; chia B nên chạy nếu còn quota; phân tích cụm (Mục này) thành một
mục chính của luận văn; B4a nên làm.

### Ánh xạ S6 — ĐÃ CÓ, KHÔNG CẦN DUYỆT LẠI
S6 = **VARIANT** của valve_butterfly trong `symbol_map.csv` (user xếp từ trước, xác nhận lại
2026-11-12 sau khi xem crop). Bản trước của mục này ghi "cần duyệt" là do AI chưa đọc
`symbol_map.csv`. Chia A và cách đếm S/V/H không đổi.
Bài học: tiêu chí ISO vs VARIANT nên gồm cả "các chi tiết phân biệt lớp ISO này với lớp ISO lân
cận có mặt trong hình DigitizePID" — dùng khi rà lại các lớp khác nếu cần.

### P2 thu nhỏ — XONG 2026-11-12 (chi tiết: PLAN "Vấn đề mở 2026-11-12")
Script `p2_mini.py` (đặt ở `/content/`, chạy với `--max-per-class 50`; CSV ở `runs/p2_mini_q50/`).
- Nhóm S: **7/10 lớp đúng** (scale 1.0). Sai: S3 → valve_butterfly (cụm van), S24 → `pipeline`
  (★ đường thẳng hút crop có ống), S21 → agitator (glyph khác template, own_cos 0.29).
- Nhóm V: **8/9 lớp VARIANT sai** (chỉ S22 hx_coil đúng). ArcFace train S2/S13 tách khỏi S5, S12
  tách khỏi S4 → đẩy VARIANT ra xa ★ của chính nó. Có lỗi tự tin tới 0.93 → τ không chặn được.
- Lề template: macro S 0.70 (1.0) vs 0.67 (1.2); 1.2 sửa một phần S3 nhưng làm hỏng S10.
- Mỗi lớp ≈ một điểm dữ liệu (50 crop cùng một dự đoán) → báo cáo THEO LỚP; không cần chạy đủ crop.

### R1a (chia A) + R2 — XONG 2026-11-13
```
holding out 13 classes: S2 S3 S4 S5 S9 S10 S12 S13 S15 S18 S19 S27 S30
19 fine classes, 29360 train crops, 7329 val crops
epoch 1  train_loss=1.7550 train_acc=0.8336 val_loss=0.0000 val_acc=0.9999
epoch 2  train_loss=0.0423 train_acc=0.9939 val_loss=0.0004 val_acc=0.9999
epoch 3  train_loss=0.0066 train_acc=0.9989 val_loss=0.0000 val_acc=1.0000   -> .ep3.pt (CMP_EPOCH)
epoch 4  train_loss=0.0216 train_acc=0.9974 val_loss=0.0000 val_acc=1.0000
epoch 5  train_loss=0.0182 train_acc=0.9970 val_loss=0.0000 val_acc=1.0000   -> .ep5.pt
```
- Run `runs/r1a/2026-11-13_11-05-11/`. Finalize: 19 lớp, **epoch 3**, không rò rỉ →
  `runs/encoders/__global_lo__.pt`. (Finalize lấy `__global__.pt` của epoch 3, trùng nội dung với
  `.ep3.pt` vì best `val_acc` rơi đúng epoch 3; notebook đã sửa để ưu tiên `.ep<N>.pt`.)
- Bão hoà giống R1b → quy tắc epoch cố định là đúng chỗ.
- R2, 1 view, scale 1.0:

| Encoder | mean off-diag cos | max | cặp sát nhất |
|---|---|---|---|
| R1b `__global__.pt` (32 lớp) | 0.245 | 0.989 | pump_general – vessel_spherical |
| R1a `__global_lo__.pt` (19 lớp) | 0.315 | 0.957 | valve_globe – check_valve_globe |

- Hai encoder là hai không gian khác nhau: so **thứ hạng, z-score trong từng ma trận, láng giềng gần
  nhất (NN)**, không so trị số tuyệt đối. Separation chỉ là chỉ báo; bằng chứng P1 là B6b-1.

**So ma trận đầy đủ (2026-11-13, 861 cặp, prototype 1 view scale 1.0):**

| | mean | p90 | p99 | max | >0.85 | >0.9 | >0.95 |
|---|---|---|---|---|---|---|---|
| R1b | 0.245 | 0.415 | 0.919 | 0.989 | 17 | 12 | 7 |
| R1a | 0.315 | 0.560 | 0.840 | 0.957 | 6 | 3 | 2 |

1. **Cấu trúc phụ thuộc mạnh vào tập train**: Spearman giữa cos các cặp của R1b và R1a chỉ 0.458;
   NN đổi ở 30/42 template. Hệ quả: "max separation" một số là chỉ báo không ổn định → báo phân
   bố + NN theo nhóm; τ phải chọn riêng cho từng encoder (đã đúng theo plan).
2. **Cụm tròn của R1b vỡ một phần ở R1a**: pump_diaphragm–vessel_spherical 0.965 → 0.359,
   viewing_glass rời cụm (NN → silencer 0.639); nhưng cụm bơm/máy nén khác hình thành (pump_general–
   compressor_vacuum 0.868, pump_centrifugal–hx_general 0.884). Giả thuyết (CHƯA kiểm): R1b train
   S27 `isa_bubble_ro` (vòng tròn trơn), R1a không → ở R1b mọi template có vòng tròn lớn bị kéo về
   hướng của S27. Kiểm được bằng class center trong checkpoint (template nào gần center S27 nhất).
3. **Cụm van globe ổn định ở cả hai encoder**: globe–butterfly 0.981 → 0.951 (hạng 3 → 2),
   globe–check_valve_globe 0.837 → 0.957. Lỗi "đặc điểm không cộng dồn" không do tập train cụ thể
   → kết quả vững cho luận văn.
4. **Template H chọn mất tách biệt khi lớp của nó bị giữ ra**: NN cos trung bình 0.607 → 0.792
   (z +2.35 → +2.88), ngược với S (z 1.88 → 1.87, không đổi) và V (0.832 → 0.677). Cụ thể:
   valve_general ↔ valve_needle 0.389 → 0.790 (cả hai `h_in`, NN của nhau); interchangeable_disc ↔
   interchangeable_disc_open 0.726 (NN của nhau, cả hai `h_in`), open_disc 0.717; valve_ball NN →
   pump_general 0.761. Đọc: encoder chỉ tách được template bằng đặc điểm nó đã học từ lớp train.

**Class center (ArcFace) ↔ template — CƠ CHẾ của cụm (2026-11-13).** cos(template, tâm lớp train):
- **Giả thuyết S27 BỊ BÁC**: tâm S27 xa mọi template (max 0.201, motor_electric). Cụm tròn của R1b
  là do **S15** (`valve_rotary_generic_other_std`, OOV, `h_out`): pump_general, pump_diaphragm,
  compressor_vacuum, vessel_spherical, viewing_glass đều có tâm gần nhất là S15 (0.56–0.66).
- **Cụm = các template chung một tâm lớp gần nhất.** R1b: S15 hút 5 template tròn; S3 hút globe,
  butterfly, check_valve_globe/swing. R1a (không có S15, S3): S25 (`emergency_relief_valve`,
  UNSURE) hút 9 template bơm/hx; S6 (butterfly VARIANT) hút 8 template van. Cụm van tồn tại ở cả hai
  vì luôn còn một lớp van "bow-tie" trong train để hút chúng.
- **Tâm lớp → template gần nhất tái hiện P2 thu nhỏ của R1b 19/19 lớp**: đúng 7 lớp S + S22, sai
  đúng S3 → butterfly, S24 → pipeline, S21 → agitator_general, S8 → reducer, S23 → hx_straight_tube
  (cùng tên sai như P2 thu nhỏ). Lý do: crop DigitizePID gom chặt quanh tâm lớp (val_acc ≈ 1). →
  **Proxy gần như miễn phí cho P2 trên lớp đã train** (không cần crop/GPU): dùng để sàng lề template,
  ★ `pipeline`, S21 trước khi chạy P2 thật. KHÔNG dùng được cho P1 (lớp giữ ra không có tâm).
- ★ `pipeline` gần tâm S24 hơn chính ★ flow_direction (0.604 > 0.546) ở R1b → giải thích vấn đề 4;
  ở R1a đảo lại (flow_direction 0.507, pipeline không trong top 3). S21 ở R1a: ★ flanged_connection
  hạng 1 nhưng cos chỉ 0.194.
- OOV đã train ở R1b nằm gần template: S15 → compressor_vacuum 0.664, S29 → compressor_centrifugal
  0.605, S26 → pump_gear 0.489 → rủi ro cổng "Other" ở P2/P4 với R1b.
- Script: scratchpad `sep_compare.py` + đoạn class-center (chưa vào `scripts/`).

### Eval Colab 2026-11-13 — C trên `test`, chọn τ, P1, P2 (notebook §8–§9, `aggregate=mean`)
CSV từng crop: `runs/oneshot_{p1,p2,tau}_crops.csv` (local, đã tải về; tau = encoder full).

**C (R1b, nửa `test`, 6021 crop)**: top-1 0.9998, top-3 1.0, macro-F1 0.9998, nhảy họ 0 — bão hoà.

**τ (9a, `val_sel`, lớp nhóm S encoder đã train)**:

| Encoder | lớp / crop | A (TPR95) | B (Youden) | B bỏ cặp > 0.95 |
|---|---|---|---|---|
| R1a `lo` | S1, S17, S21, S24 / 1094 | 0.470 | 0.430 (J 0.673) | 0.430 (không cặp nào) |
| R1b `full` | 10 lớp S / 2138 | 0.429 | 0.521 (J 0.471) | 0.521 (bỏ S3–butterfly 0.981) |

Chưa ghi vào `config.yaml` (cần user duyệt).

**P1 (R1a, crop train + val: 10470 h_in / 6909 h_out)**: h_in macro top-1 **0.036** (ngẫu nhiên
1/42 = 0.024), top-3 0.365; AUROC 0.777; OSCR 0.030. So cặp cùng lớp, R1b (đã thấy, P2 `test`) →
R1a (chưa thấy):

| Lớp | ★ | R1b top-1 (cos) | R1a top-1 / top-3 (cos) | R1a hay ra |
|---|---|---|---|---|
| S3 | valve_globe | 0.00 (0.974) | 0.07 / 0.85 (0.931) | valve_butterfly |
| S4 | valve_general | 1.00 (0.961) | 0.00 / 0.00 (0.702) | valve_control |
| S5 | valve_ball | 1.00 (0.529) | 0.15 / 0.34 (0.652) | check_valve_swing |
| S10 | valve_needle | 1.00 (0.681) | 0.00 / 0.00 (0.641) | valve_control |
| S18 | interchangeable_disc | 1.00 (0.904) | 0.00 / 0.00 (0.527) | open_disc |
| S19 | interchangeable_disc_open | 1.00 (0.844) | 0.00 / 1.00 (0.532) | open_disc |

→ **5/5 lớp đúng khi đã thấy thành sai khi chưa thấy**; S3 sai ở cả hai. One-shot qua template ISO
cho lớp DigitizePID chưa thấy ≈ ngẫu nhiên. Thành công của P2 trên lớp S là do encoder đã học lớp
đó (template rơi gần tâm lớp), không phải khả năng so khớp one-shot. h_out (τ_A = 0.470):

| Lớp | cos trung vị | bị loại tại τ_A | hay ra |
|---|---|---|---|
| S9 `valve_diaphragm_other_std` | 0.676 | 0% | pump_diaphragm |
| S15 `valve_rotary_generic_other_std` | 0.563 | 2% | compressor_vacuum |
| S27 `isa_bubble_ro` | 0.464 | 91% | pump_gear |
| S30 `isa_square_sta` | 0.376 | 98% | reducer |

"Other" tại τ_A: P 0.997, R 0.467, F1 0.636 (tại τ_B 0.430: R 0.211) — thực chất 2/4 lớp bị loại.

**P2 (R1b, nửa `test`, 5665 crop)**: S macro top-1 0.70 (top-3 0.914) — sai đúng S3 → butterfly,
S21 → agitator_general, S24 → pipeline, **y hệt P2 thu nhỏ và proxy tâm lớp**; V 1/6 (chỉ S22);
V_of_S 0/3 (S2, S12, S13; cos ~0.26 → ArcFace đẩy chúng xa ★, vấn đề mở 6); OOV_seen: S15 →
pump_general cos 0.952. AUROC 0.682, OSCR 0.372. UNSURE loại: S11 168, S25 188 crop.

**Lưu ý cho P4**: prototype (`iso_symbols_clean`) và ảnh OPEN100-ISO cùng nguồn vector BK-2020 → nhóm
H tự nhiên của P4 là so khớp gần-trùng (cùng hình vẽ, khác render/suy giảm), KHÁC P1 (khác phong cách
vẽ). P1 thất bại không suy ra P4 thất bại; P4 tốt cũng không chứng minh chuyển giao sang phong cách
lạ. Phải nói rõ khi viết.

**τ đã ghi vào `config.yaml` (user duyệt 2026-11-14)**: `oneshot.other_min_cosine: 0.4291` = R1b,
quy tắc A; bảng đủ 4 giá trị (R1a/R1b × A/B) nằm trong comment ngay trên khóa. R1a hoặc quy tắc B
truyền qua CLI. τ chọn với `aggregate=mean` — nếu dùng `vote` (mặc định config) phải chọn lại.

**S9 — vì sao khớp pump_diaphragm (2026-11-14, trang so sánh
https://claude.ai/artifact/Qxu6bm5yNvpvTdDGx46PsP)**: tính lại cục bộ 12 crop S9 (khớp CSV đến 3
chữ số). Với R1a, 11/12 crop có tâm lớp gần nhất là **S25** (`emergency_relief_valve`, tròn có X;
cos 0.36–0.60, mọi tâm khác < 0.12 trừ S7); tâm S25 → pump_diaphragm 0.516 > hx_general 0.494 >
pump_centrifugal 0.486 > pump_general 0.350 — **trùng đúng thứ tự top-4 của crop S9**, lề top1−top2
trung vị chỉ 0.007. Crop gần S7 (`valve_manual_other_std`) hơn thì ra valve_control. Kết luận:
pump_diaphragm thắng vì là ★ gần tâm S25 nhất, không vì nét riêng của S9 — cùng cơ chế "đi theo tâm
lớp đã học gần nhất" như h_in. R1b (đã train S9): ra hx_general cos ~0.47 → **lọt τ_A 0.4291 (loại
0%) nhưng bị τ_B 0.5210 loại 97%**. Trên cả OOV_seen của P2: τ_A loại 27.8%, τ_B 47.2%, còn S+V
nhận-và-đúng gần như không đổi (0.444 vs 0.435). Không đổi mặc định theo số này (là chỉnh τ trên
tập đánh giá); ghi làm quan sát cho chương thảo luận.
User trả lời 2026-11-14 (`docs/i_chi_u_chuy_n_m_n_l_p_s9_van_m_ng.md`): **chốt** S9 giữ OOV / `h_out`,
không gán ★ ISO nào; **chốt** giữ τ_A làm mặc định. **Còn mở**: mục 1 + câu 1 của tài liệu mô tả hình
ngược với ảnh gốc (kiểm 8 crop ngẫu nhiên mỗi lớp từ COCO: S9 = hai tam giác đối đỉnh + cung tròn một
phía; S25 = vòng tròn có X; S7 = hai tam giác + móc chữ L; ★ X8087 = hai tam giác + gạch chéo, không
vòm) → lập luận "S9 là tròn có X" cần user xác nhận lại trước khi đưa vào luận văn.
**Câu 1 đo trực tiếp (2026-11-14, 40 crop/lớp, 19 lớp R1a đã thấy, cắt từ COCO)**: lớp đã thấy gần S9
nhất — HOG-lite (pixel, không học): tâm S1 0.751 > S11 0.742 > S6 0.723 > S25 0.702, từng crop hay
ra S7 (14/40); ImageNet (B4a): tâm S7 0.959 > S1 > S14 > S11, từng crop S7 25/40, S25 7/40; **R1a: S25
0.898, kế S7 0.394, từng crop S25 40/40**. → Theo hình (pixel, ImageNet) S9 gần các lớp van hai tam
giác (S7, S1, S11) nhất; **lực hút về S25 do train DigitizePID tạo ra**, không phải hình học sẵn có.
Khớp Kornblith et al. NeurIPS 2021 (loss tách lớp tốt hơn → feature cố định chuyển giao kém hơn,
co biến thiên trong lớp). Dùng làm luận điểm cho P1 âm, và là dự đoán cho B4a §10 (ImageNet có thể
không kém R1a trên P1).
**Lề top1−top2 (dạng điểm 2 của PLAN B8), thử nhanh 2026-11-14 trên R1b**: δ chọn trên `val_sel` (tau
CSV) theo quy tắc TPR95 → δ = 0.0321 (không nhìn P2 khi chọn). Trên P2: cổng "cos < τ_A **hoặc**
lề < δ" loại OOV_seen 47.9% (chỉ τ_A: 27.8%), **S9 82%, S15 100%** (τ_A: 0% cả hai; S15 cos 0.952);
cái giá: S bị loại 5.8% → 10.9%, V 15.9% → 34.7%. Lạ "gần" (cùng họ van: S9, S15) không loại được bằng
cos nhưng lộ ra ở lề nhỏ (crop nằm giữa nhiều ★ gần nhau); lạ "xa" (S27, S30, ISA) đã bị τ loại ở P1.
**Code ✅ 2026-11-14 (user duyệt)**: `margins` / `margin_rule_a` / `margin_auroc` / `gated_other_prf`
(`utils/oneshot_metrics.py`), lượt τ ghi `margin_rule_a.delta`, P1/P2 ghi `margin_auroc`, notebook in dòng
`rule_a + lề` (RUNBOOK §5e). Kiểm notebook trên CSV P2 thật: Other R 0.278 → 0.478, P 0.649 → 0.610,
F1 0.389 → 0.536, nhận+đúng 0.4437 → 0.4434. AUROC của lề một mình **không** hơn cos (P1 0.705 vs
0.777; P2 0.686 vs 0.682): hai điểm bắt hai loại lạ khác nhau (xa = cos thấp, gần = lề nhỏ) nên chỉ
cổng HOẶC mới có lợi. Cần chạy lại 9a trên Colab để có δ của R1a, rồi 9b.

**Câu trả lời user cho câu 2, 3 — đánh giá sau khi đọc tài liệu ngoài (2026-11-14, chỉ ghi ở đây)**:
- Câu 2 — giữ kết luận (S9 OOV, không ★). Bỏ: "X8087 có mũ vòm" (★ trong repo không có vòm), "S9 gần
  pump_diaphragm/hx_general về pixel" (dựa trên mô tả hình bị tráo; HOG nói ngược lại), mã "2107" ISO
  14617-8 (chưa xác minh; 14617-8 đã bị thay bằng 14617-2:2025, bản trả phí). Thêm: S9 là lạ **gần**
  (cùng họ van) — h_out chia gần (S9, S15: loại 0–2% ở P1) / xa (S27, S30 ISA: 91–98%), khớp khung
  near/far semantic shift của Vaze et al. ICLR 2022 (arXiv 2110.06207).
- Câu 3 — giữ τ_A. Bỏ: "cos 0.676 → 0.473 chứng tỏ R1b tốt hơn". Sửa: "τ_B loại 97% S9" phải nêu kèm
  cái giá (S+V được nhận 0.909 → 0.651). Thêm: Dhamija et al. NeurIPS 2018 (arXiv 1811.04110, đặc
  trưng lạ chồng lên biết → ngưỡng độ tương đồng không đủ; nguồn của OSCR) → động cơ cho dạng điểm 2.
- Khung cho P1 âm: Kornblith et al. NeurIPS 2021 (arXiv 2010.16402: loss tách lớp tốt hơn → feature
  cố định chuyển giao kém, co biến thiên trong lớp); đối trọng Galanti et al. ICLR 2022 (arXiv
  2112.15121: neural collapse giúp chuyển giao khi nguồn đa dạng — DigitizePID chỉ 32 lớp). Không dùng so sánh
cos 0.676 (R1a) → 0.473 (R1b) như bằng chứng "R1b tốt hơn": hai encoder khác thang cos, R1b đã thấy S9.

### B6 code — 2026-11-14 (user: làm xong B6 trước khi báo cáo sơ bộ với thầy)
- **B6d ✅**: export kiểm tham số ảnh suy giảm (sinh lại từ seed + config hiện tại; lệch → lỗi, không
  export); kiểm chỉ-đọc trên `data/iso_open100` hiện có: **432/432 khớp**. Mũi tên dẫn + cờ ghi vào COCO
  là vùng bỏ qua (`iscrowd: 1`, category `__ignore__` cuối danh sách, `ignore_kind`); build_classes /
  build_manifest / coco_to_yolo / extract_crops bỏ qua; DigitizePID ra giống hệt từng byte. **Đã
  export lại `data/iso_open100` 2026-11-14** (user duyệt; 1 phút 42 giây): 864 ảnh, 93,800 symbol +
  12,096 vùng bỏ qua (leader_arrow 5,184, flag 6,912; 2–27 vùng/ảnh, cả 864 ảnh). Đối chiếu với bản
  cũ: images, 45 category đầu, 93,800 symbol (id, box, nhãn) giống hệt; chỉ thêm category 46
  `__ignore__` và các vùng bỏ qua ở cuối; classes.yaml / manifest.csv / splits.json không đổi.
  `scripts/check_export.py` khớp. **Còn**: zip lại `canonical` lên Drive cho Colab (cell 11a đếm
  93,800 annotation không iscrowd).
- **B6e ✅**: `++detector.agnostic_merge=true` (mặc định tắt): `agnostic_nms` mỗi tile + WBF không phân
  nhãn, nhãn theo box thành phần điểm cao nhất. Bỏ `data.label_level=agnostic` (dùng `single_cls`
  nếu cần detector một lớp).
- **B6b-2 P4 mức crop ✅**: `++eval.oneshot.protocol=p4` (+ `++eval.oneshot.max_per_class`, `seed`);
  nhóm tự động `assign_iso_groups` khớp bảng PLAN (R1b S9/V4/H tự nhiên 17; R1a S3/V4/H chọn 6/H tự
  nhiên 17); không bao giờ gộp sạch/suy giảm; theo tờ mean ± std; Other: dương = other_iso,
  other_isa / other_generic riêng. Notebook §11 (11a giải nén OPEN100-ISO từ Drive:
  `<DRIVE_ROOT>/data/iso_open100_canonical.zip` ~1.9 GB, tạo bằng
  `cd data/iso_open100 && zip -r -0 iso_open100_canonical.zip canonical`).
  **Số sơ bộ cục bộ (CPU, 30 crop/lớp/view, `aggregate=mean`, suy giảm)**: S top-1 R1b 0.53 / R1a 1.00
  (chỉ 3 lớp) / ImageNet 0.80; H tự nhiên R1b 0.50 / R1a 0.37 / ImageNet 0.20; H chọn R1a 0.24 /
  ImageNet 0.02; AUROC Other (cos) 0.56 / 0.65 / 0.61, (lề) 0.70 / 0.72 / 0.43. Tại τ_A không loại
  other_iso nào (cos other_iso ~0.76 vs vocab ~0.79 với R1b); τ_A HOẶC lề bắt ~57% other_iso nhưng
  nhận+đúng 0.50 → 0.40. S của R1b bị kéo bởi flanged_connection → agitator_general, flow_direction →
  pipeline, valve_globe → butterfly (**cùng lỗi P2: vấn đề mở 4, 5, cụm van**), valve_needle → gate.
  Trái với kỳ vọng "P4 gần-trùng nên dễ": cổng Other yếu trên OPEN100 cho cả ba encoder.
- **B6b-2 phần detection / end-to-end / box sai ✅ (2026-11-14)** — chi tiết PLAN B6b. Khóa mới (chỉ
  `++`): `eval.class_agnostic`, `eval.oneshot.margin_delta`, `detector.classes_yaml`, end-to-end =
  `eval.mode=det_plus_fgc ++fgc.oneshot=true`. Notebook §12 (RUNBOOK §5g); user phải điền `DET_CKPT`.
  Chưa chạy detector thật (không có `best.pt` trên máy) — chỉ thử đường code với box giả.
  Mặc định đã chọn (user có thể đổi): Other đúng cho cả `other_iso` / `other_isa` / `other_generic` (báo
  riêng từng dòng); Other trên nền = FP; end-to-end OPEN100 dùng detection `agnostic_merge`, DigitizePID
  dùng gộp theo nhãn.
- **Phase A (code) XONG 2026-11-14**: 766 test pass. Còn lại là chạy (Colab §9 lại 9a, §10, §11, §12)
  + export lại OPEN100-ISO (chờ duyệt).

### B4a §10 trên Colab — 2026-11-15 (P1, cùng 17,379 crop với 9b; `aggregate=mean`)
Tính lại từ `runs/oneshot_p1_crops_imagenet.csv`: khớp log từng số.

| P1 (`h_in` 6 lớp / `h_out` 4 lớp) | ImageNet thô | R1a | ngẫu nhiên |
|---|---|---|---|
| macro top-1 / top-3 | 0.008 / 0.116 | 0.036 / 0.366 | 0.024 / 0.071 |
| AUROC cos / lề | 0.490 / 0.434 | 0.777 / 0.705 | 0.5 |
| OSCR | 0.006 | 0.030 | |

- 10a: prototype ImageNet mean cos 0.474, max 0.978 (check_valve_globe–swing), 36 cặp > 0.8.
- 10b: τ ImageNet (lớp S1/S17/S21/S24 của R1a) A 0.6840, B 0.7717 (J 0.108, R1a 0.673), δ 0.0022.
- **ImageNet thô dưới mức ngẫu nhiên vì HUB**: 58.9% mọi crop (biết + lạ) → `flow_direction`. Feature sau
  average-pool ≥ 0 (min đúng 0.0) → mọi cos cao (5–95%: 0.66–0.84, lề trung vị 0.025); prototype gần hướng
  trung bình của crop DigitizePID thắng. Không phải `flow_direction` gần trung bình *prototype* (gần nhất
  là valve_ball 0.834) → hub đến từ lệch miền crop ↔ template. Dự đoán ghi 2026-11-14 "ImageNet có thể
  không kém R1a trên P1" → **SAI với bản thô**, nhưng ĐÚNG về top-1/top-3 sau CL2N (dưới).
- **Cổng lề δ không chuyển sang lớp chưa thấy (giới hạn, ghi vào luận văn)**: δ của R1a = 0.1301 chọn trên
  4 lớp R1a ĐÃ TRAIN (lề lớn). Trên P1 chỉ 1.2% crop `h_in` có lề ≥ δ (trung vị 0.063; `h_out` 0.3%) → cổng
  "τ_A hoặc δ" loại gần hết: nhận+đúng 0.000, Other R 0.997, P 0.400 (= tỉ lệ crop lạ). Cổng cos τ_A =
  0.4699 thì tốt: nhận 99.9% `h_in`, loại 46.7% lạ, P 0.997. → δ chỉ dùng cho lớp đã thấy (P2: vẫn có
  ích, xem trên); P1 chỉ báo cổng τ.

### CL2N (căn giữa) — đo 2026-11-15, CODE ✅ (user: "nếu đổi đáng kể thì thêm code")
SimpleShot (Wang et al. 2019, arXiv 1911.04623): trừ vector trung bình rồi mới L2. Đo cục bộ trên mẫu P1
1,500 crop (150/lớp, seed 0) + 570 crop train của 19 lớp R1a đã thấy (30/lớp) làm tâm; mẫu thô tái lập
Colab (top-1 0.010, hub 59%). Script: scratchpad `cl2n/` (không trong repo).

| P1 mẫu | top-1 | top-3 | AUROC cos | AUROC lề | hub |
|---|---|---|---|---|---|
| ImageNet thô | 0.010 | 0.119 | 0.477 | 0.422 | flow_direction 59% |
| **ImageNet + CL2N** (tâm = TB crop train lớp đã thấy) | **0.131** | **0.407** | 0.444 | 0.337 | valve_three_way 13% |
| ImageNet, tâm = TB 42 prototype | 0.001 | 0.123 | 0.471 | 0.426 | flow_direction 78% |
| R1a thô | 0.044 | 0.384 | 0.776 | 0.698 | valve_control 20% |
| R1a + CL2N | 0.040 | 0.382 | 0.803 | 0.655 | valve_control 20% |

(Dòng CL2N tính qua đúng đường code mới: tâm = TB mọi view, căn từng view rồi gộp `mean`; bản tạm căn sau
khi gộp ra 0.133 / 0.410 / 0.433 — cùng kết luận.)
- **Đọc**: ImageNet + CL2N **hơn R1a về top-1 (0.13 vs 0.04) và top-3 (0.41 vs 0.38)** trên lớp chưa thấy,
  nhưng **kém xa về tách lạ** (AUROC 0.44 vs 0.78). → train DigitizePID đổi khả năng gọi đúng tên lớp mới
  lấy khả năng nhận ra "lạ" (khớp Kornblith et al. 2021). Đây là đối chứng phải dùng cho P1, không phải
  ImageNet thô (thô = baseline bị làm yếu bởi hub).
- Tâm phải từ miền crop (DigitizePID), không từ template. CL2N gần như không đổi R1a (top-1), +0.03 AUROC.
- Code: `utils.oneshot.center_embeddings` / `PrototypeDirectory.centered` / `save_center` / `load_center`;
  `OneShotClassifier.set_center`; khóa `++oneshot.center=<file>` (mọi đường one-shot: eval p1/p2/tau/p4 và
  `fgc.oneshot=true`; sha256 encoder phải khớp file tâm); `++eval.oneshot.protocol=center` ghi
  `oneshot_center.pt` (crop `crops_fgc/train` của lớp đã train theo `groups_ckpt`, LUÔN trừ mọi lớp
  `oneshot.holdout`; `++eval.oneshot.max_per_class`). metrics.json ghi `center`. τ phải chọn lại với tâm.
  Notebook 10d (tâm + τ + P1 cho ImageNet và R1a) / 10e (bảng 4 cột). RUNBOOK §5f. 777 test pass.
- **Chờ Colab 10d–10e** (số đủ 17,379 crop + τ chọn lại). P4 với CL2N: chưa nối vào §11 (tâm có sẵn
  `encoders/center_<tag>.pt`, thêm `++oneshot.center=` là chạy được) — quyết sau khi có 10e.

### Chờ user quyết (PLAN "Vấn đề mở 2026-11-12")
- Điểm 4: bỏ ★ `pipeline` (và template thuần đường kẻ) khỏi thư mục prototype?
- Điểm 5: S21 giữ ISO hay đổi VARIANT? (xem crop cạnh template mặt bích, như đã làm với S6)
- Điểm 6: có train bằng nhãn theo nghĩa ISO (gộp S2, S13 → S5; S12 → S4) không? Chưa làm.
- Có chấm C trên ảnh DigitizePID suy giảm không; có làm P3 (ASU) không.

### Việc tiếp theo (thứ tự)
1. Commit HANDOFF + PLAN mới, hai file split (`digitizepid_splits.json`, `digitizepid_val_test.json`)
   và `p2_mini.py` (đề xuất `scripts/`).
2. Thử prototype trung bình hai scale: `'oneshot.template_views.scales=[1.0,1.2]'` →
   `iso_prototypes_m10_12.pt`; chạy lại `p2_mini.py` với 3 file, `--max-per-class 50`. Chốt lề.
3. User quyết điểm 4, 5 ở trên (đổi `symbol_map.csv` / thư mục prototype nếu cần) → chạy lại R2 +
   P2 thu nhỏ để có số nền mới.
4. ✅ B5b (2026-11-13): `check_b5.py` kiểm luật VARIANT, h_extra, không rút cạn họ; chia A 0 lỗi.
5. ✅ Quyết 2026-11-13: R1b dùng epoch 3, không resume; R1a so ở cùng epoch 3 (`fgc_train` +
   notebook đã sửa).
6. ✅ R1a (chia A) xong 2026-11-13: `__global_lo__.pt` (epoch 3), `iso_prototypes_lo.pt`.
7. Code đường găng P4: B6a, B6b (P2/P4, AUROC/OSCR, đường quét τ, chọn τ trên `val_sel`), B6d.
   ✅ B6c (2026-11-13): `evaluate eval.mode=fgc_only` cho `__global__.pt`, YOLO fine chấm bằng
   `det_only detector.label_level=fine`; lọc `++eval.image_subset=test` ✅ (B6b-1).
   ✅ B6a (2026-11-13): `++fgc.oneshot=true` → `derived/detections_oneshot/` (điểm chưa cắt ngưỡng
   cho B6b).
   ✅ B6b-1 (2026-11-13): `eval.mode=oneshot` p1/p2/tau mức crop; notebook §9 → **chạy trên Colab**
   (RUNBOOK §5e). Còn: B6b-2 (P4) + B6d.
8. ✅ B4a code (2026-11-14, user chốt làm): `scripts/make_imagenet_encoder.py`, `++eval.oneshot.groups_ckpt`,
   notebook §10 (RUNBOOK §5f). Separation cục bộ: ImageNet mean 0.474 / max 0.978 (lo 0.315, full
   0.245) → train DigitizePID kéo prototype xa nhau. ✅ §10 chạy Colab 2026-11-15 (ImageNet thô bị hub,
   xem "B4a §10 trên Colab"); ✅ CL2N code. **Chạy 10d–10e trên Colab** — ImageNet + CL2N vs R1a là đối
   chứng quyết định cho kết quả P1 âm.
9. Chưa đụng `test` cho tới khi chốt checkpoint, τ, lề template, (d).

Đã xong: tách val; S6 = VARIANT (có sẵn); loại giả thuyết "vạch = ống"; R2; P2 thu nhỏ.

### Dự đoán đã ghi trước khi có số (để đối chiếu)
- val_acc 0.95–0.99 → thực tế 0.9999 (dễ hơn dự đoán).
- mean separation 0.5–0.7 → thực tế 0.245 (tốt hơn nhiều).
- cặp gần nhất trong họ van > 0.85 → đúng 0.981, nhưng cặp gần nhất toàn cục là hình tròn (sai).

**Dự đoán cho P1 / R1a (ghi 2026-11-13, trước khi train R1a):**
- Lớp `h_in` mà R1b đã sai trên `val_sel` (S3 → valve_butterfly) vẫn sai ở R1a: không mang thông
  tin về khả năng nhận lớp chưa thấy. P1 đọc theo **từng lớp, so cặp R1b ↔ R1a**; bằng chứng là
  các lớp đúng ở R1b mà sai ở R1a. Không đổi H vì kết quả này (H đã chốt).
- `h_out` S27 (`isa_bubble_ro`, hình tròn) khớp nhầm cụm template tròn (pump/vessel/compressor)
  với cosine cao → cổng "Other" khó chặn. S30 (`isa_square_sta`, hình vuông) ít rủi ro hơn.
- `h_extra` không thi, nhưng loại S2/S13/S12 khỏi train có thể làm S5/S4 (`h_in`) khớp ★ ISO dễ
  hơn ở R1a so với R1b (không còn lớp anh em bị ArcFace đẩy ra xa — vấn đề mở 6).

**Dự đoán bổ sung từ ma trận R1a (ghi 2026-11-13, trước B6b-1):**
- S4 (valve_general) và S10 (valve_needle) nhầm sang nhau.
- S18 và S19 nhầm sang nhau và/hoặc sang open_disc.
- S3 → check_valve_globe hoặc valve_butterfly (sai ở cả R1b lẫn R1a).
- S5 (valve_ball) có nguy cơ khớp template thân tròn (pump_general).
- P1 trên các lớp `h_in` thấp hơn P2 trên các lớp S tương ứng ở R1b (cùng ★) — đó là cái giá của
  "chưa thấy", là số chính của P1.
**Đối chiếu dự đoán P1 (2026-11-13, sau Colab):**
- P1 `h_in` thấp hơn P2 cùng lớp → **ĐÚNG**, mạnh hơn dự kiến (5/5 lớp từ 1.0 về ~0).
- S3 sai ở cả hai → **ĐÚNG** (→ valve_butterfly).
- S18, S19 nhầm sang nhau / open_disc → **ĐÚNG** (cả hai → open_disc).
- S4 ↔ S10 nhầm sang nhau → **SAI chi tiết**: cả hai cùng ra valve_control (cùng sụp, nhưng sang ★ thứ ba).
- S5 → template thân tròn → **SAI** (→ check_valve_swing).
- S27 khớp cụm tròn với cos cao → **SAI** (cos 0.464, 91% bị loại). Bản sửa "S15 rủi ro nhất" → đúng
  một phần (S15 → compressor_vacuum, 2% bị loại), nhưng **S9 còn tệ hơn** (cos 0.676, 0% bị loại).
- **Sửa đổi (2026-11-13, sau phân tích class center, vẫn trước P1)**: rủi ro "Other" lớn nhất trong
  `h_out` là **S15** (ở R1b tâm S15 ≈ template tròn 0.66), không phải S27 (tâm S27 xa mọi template).
  Dự đoán S27 ở trên giữ nguyên để đối chiếu.

### Stage 1 — ĐÃ CHỐT trong PLAN (Quyết định 2026-11-09 mục 10)
Giữ detector 8 lớp coarse + `agnostic_merge` lúc suy luận (B6e); detector 1 lớp là ablation (giả
thuyết OLN). Bản trước của mục này ghi "chưa chốt" là do AI chưa đọc PLAN. Vẫn đo recall Stage 1
theo S/V/H trên OPEN100-ISO.

---

## 7. MÔI TRƯỜNG / LƯU Ý VẬN HÀNH

- `uv run pytest` / `uv run ruff check src/ tests/ scripts/` — phải xanh trước khi commit.
- **CẢNH BÁO session 2026-10-10 (vẫn đúng 2026-10-14)**: máy KHÔNG có `uv` và KHÔNG có torch/timm
  (chỉ system python3.9 + numpy/PIL/yaml/pytest) → code GĐ2 CHƯA chạy pytest/ruff.
  Đã verify tay phần torch-free (ViewSpec/refit_box/rotate_quarter/make_views/
  read_vocab/load_templates trên 39 symbol ISO thật) + transcribe thuật toán voting
  sang numpy để kiểm semantics (tally, tie-break, gate admission). PHẦN TORCH
  (embed/PrototypeDirectory/save-load/pipeline end-to-end) CẦN `uv run pytest` xác nhận.
- Repo ĐÃ là git (user tự quản lý branch/commit). ĐỪNG `git init`/`rm -rf .git` (đã có lần nhầm).
- Colab: ảnh/checkpoint trên Drive; code qua git; copy tiles/crops → /content local trước train.
- `fgc.route.other_min_cosine=0.35` là PLACEHOLDER — PHẢI calibrate trên val thật.
  **2026-11-12**: `oneshot.other_min_cosine` chọn trên `val_sel` theo quy tắc A (τ@TPR95) và B
  (bỏ-prototype + crop OOV còn train làm "lạ"); KHÔNG chọn trên `test` hay OPEN100-ISO.
- Kiểm tra chỉ đọc file (split, separation, xem template) chạy được trên runtime CPU; đổi
  runtime = reset phiên → chạy lại mục 1–3 notebook (crop có sẵn zip trên Drive, chỉ giải nén).
- Ngôn ngữ làm việc: tiếng Việt. User là tác giả luận văn, có chuyên môn P&ID (nhận diện symbol
  giỏi hơn AI — luôn để user duyệt symbol, đừng tự đoán valve loại nào).