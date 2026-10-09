# HANDOFF — Trạng thái dự án & hướng đi (cập nhật 2026-10-09)

File này để **tiếp tục ở session sau** mà không phải làm lại những gì đã thử. Đọc file này
+ `docs/RENDER_RULES.md` + `~/.claude/projects/.../memory/*.md` là nắm toàn bộ.

---

## 0. TÓM TẮT 1 PHÚT

Thesis: **Detection + Fine-Grained Classification symbol P&ID theo chuẩn ISO 10628-2**.
Pipeline 2 tầng: Stage 1 = YOLOv8 detect coarse (SAHI+WBF), Stage 2 = ResNet-34 + **ArcFace**
fine-grained + **"Other"/open-set gate**.

**HƯỚNG CHỐT CUỐI CÙNG (2026-10-09): one-shot ArcFace, BỎ render synthetic.**
- Train ArcFace trên **DigitizePID** (data thật, nhiều mẫu) → học embedding space.
- **Fine-tune với ISO template** (39 symbol library).
- Nhận symbol ISO MỚI (không có trong train) bằng **cosine distance tới ISO prototype** → one-shot.
- Symbol ngoài vocab → **"Other"** (cosine < ngưỡng). ĐÂY LÀ ĐÓNG GÓP CHÍNH.

**ĐÃ BỎ**: toàn bộ hướng render synthetic P&ID (2 cách đều thất bại — xem Mục 4).

---

## 1. ĐÓNG GÓP THESIS (đã chốt với user: cả 3 trục, nhấn mạnh #2)

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

**Resume (Colab ~6h sessions) ĐÃ CODE**: detector.resume (YOLO last.pt), fgc.resume_from.

**Tests**: ~109 pass, ruff sạch. test_arcface, test_resume, test_detect_infer_wbf,
test_convert_yolo_to_coco, test_extract_iso_symbols, test_render_pid(_replace),
test_visualize_detections.

**Scripts** (scripts/):
- `convert_yolo_to_coco.py` — DigitizePID YOLO → COCO (8 coarse, Symbol_N). DÙNG CHO HƯỚNG CŨ.
- `extract_iso_symbols.py` — cắt 39 ISO symbol từ PDF (vector bbox + override tay + loại legend).
- `extract_bg_patches.py` — [HƯỚNG RENDER ĐÃ BỎ] cắt background patch.
- `render_pid.py`, `render_pid_replace.py` — [HƯỚNG RENDER ĐÃ BỎ] 2 renderer, đều không ổn.
- `visualize_detections.py` — vẽ box + legend + lọc score (--score-thr, --fill, --show-text).

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
- `data/canonical_iso/` — [RENDER ĐÃ BỎ] 500 ảnh replace, có thể XÓA.
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

### GĐ1 — Train embedding base (DigitizePID) [code gần xong, cần chạy]
- `fgc --train data_root=<dpid> fgc.metric_learning=true` trên DigitizePID crop (32 class).
- CẦN THÊM: per-subpart augment (OSSR-PID kỹ thuật #2) vào dataset.py/fgc_train.
- Chạy Colab (DigitizePID 7168px nặng, cần GPU + tiling + fix I/O).

### GĐ2 — ISO template → prototype (CỐT LÕI one-shot) [CẦN CODE, file mới]
- Đưa 39 ISO symbol (iso_symbols_clean) qua model GĐ1 → 39 vector prototype.
- Lưu "ISO prototype directory" {fine_name → embedding}.
- Đây là bước biến "closed-set DigitizePID" thành "one-shot ISO".

### GĐ3 — Fine-tune với ISO template [CẦN CODE]
- Fine-tune model GĐ1 với 39 ISO template (augment mạnh vì ít mẫu).
- Giúp ISO prototype tách biệt hơn trong embedding space.

### GĐ4 — Inference one-shot + "Other" [gate đã có, cần sửa fgc_infer]
- Crop query → embedding → cosine với 39 ISO prototype → ISO fine_name gần nhất.
- Cosine < ngưỡng → "Other" (gate đã có, calibrate other_min_cosine trên val).
- Thêm multi-scale + voting (OSSR-PID #3).

### GĐ5 — Eval [cần mở rộng evaluation.py]
- Accuracy nhận ISO, "Other" precision/recall (cần tập symbol out-of-vocab để test),
  cross-domain (train DigitizePID → nhận ISO), so OSSR-PID.

**User ĐANG PHÂN VÂN bắt đầu GĐ nào** (câu hỏi cuối chưa trả lời): GĐ2 (code one-shot core,
test logic model giả) HAY GĐ1 (train base Colab trước) HAY viết plan+notebook đầy đủ.
→ Gợi ý: GĐ2 trước (code one-shot core, test bằng model nhỏ), train thật sau.

---

## 7. MÔI TRƯỜNG / LƯU Ý VẬN HÀNH

- `uv run pytest` / `uv run ruff check src/ tests/ scripts/` — phải xanh trước khi commit.
- Repo CHƯA phải git (user tự quản lý git). ĐỪNG `git init`/`rm -rf .git` (đã có lần nhầm).
- Colab: ảnh/checkpoint trên Drive; code qua git; copy tiles/crops → /content local trước train.
- `fgc.route.other_min_cosine=0.35` là PLACEHOLDER — PHẢI calibrate trên val thật.
- Ngôn ngữ làm việc: tiếng Việt. User là tác giả luận văn, có chuyên môn P&ID (nhận diện symbol
  giỏi hơn AI — luôn để user duyệt symbol, đừng tự đoán valve loại nào).
