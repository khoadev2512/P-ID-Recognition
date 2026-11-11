# HANDOFF — Trạng thái dự án & hướng đi (cập nhật 2026-10-10)

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

---

## 1. ĐÓNG GÓP THESIS (đã chốt với user: cả 3 trục, nhấn mạnh #2)

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
- **`data/raw/iso_symbols_clean/`** (39 PNG) — ISO symbol library (cắt từ ISO_10628-2 PDF).
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
| `pid-fgc-infer` | TỪ CHỐI (raise NotImplementedError có hướng dẫn) — stage này load 1 ckpt/group và route theo coarse; đối ứng của nó là one-shot GĐ4 |

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
- Ngôn ngữ làm việc: tiếng Việt. User là tác giả luận văn, có chuyên môn P&ID (nhận diện symbol
  giỏi hơn AI — luôn để user duyệt symbol, đừng tự đoán valve loại nào).
