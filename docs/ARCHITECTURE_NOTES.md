# P&ID Recognition — Architecture Notes

> Bản tóm tắt kiến trúc để giữ context khi chuyển máy / mở lại project.
> Ghi lần đầu: 2026-09-23. Đọc-hiểu source, chưa sửa code.

## Mục tiêu

Base source cho luận văn *Symbol Detection and Fine-Grained Classification in P&ID*.
Hướng tiếp cận: **detection-then-fine-grained-classification** (phát hiện thô → phân loại tinh).

**Trạng thái:** code viết end-to-end nhưng CHƯA chạy trên dữ liệu thật —
`data/canonical/` vẫn là placeholder rỗng. Mới test qua unit test + fixture tổng hợp.

## Kiến trúc 2 tầng tách rời

| Tầng | Model | Nhiệm vụ |
|------|-------|----------|
| **1. Detect** | YOLOv8 (Ultralytics) | Localization + phân loại **thô** (coarse) trên tile SAHI, gộp bằng WBF |
| **2. FGC** | ResNet-34 (timm) | Phân loại **tinh** trong nhóm, trên crop từng ký hiệu |
| **Eval** | — | Metric detection (mAP@50:95) + metric FGC (confusion trong nhóm, top-k, macro-F1 lớp hiếm) |

Hai tầng cố tình **tách rời** để đo đóng góp của FGC như một *ablation* (báo cáo §4.2.1).
Lý do cần FGC: nhiều van có **cùng silhouette bow-tie** → detector không phân biệt nổi,
cần head riêng zoom vào crop.

## File quan trọng nhất: `classes.yaml`

Định nghĩa: vocabulary tinh, super-category thô, mapping fine→coarse, và nhóm nào cần
route sang FGC (`fgc_groups`). Xem mẫu ở `configs/classes.example.yaml`.
`src/utils/classmap.py` load + validate chặt chẽ (kiểm tra fgc_groups/parents/rare hợp lệ).

## Luồng dữ liệu — 3 tầng (`src/utils/paths.py`)

```
data/raw/         tier 0 — nguồn bất biến (synthetic/, real/)
data/canonical/   tier 1 — source of truth (images/, annotations.coco.json, classes.yaml, manifest.csv)
data/derived/     tier 2 — tái tạo được (yolo_coarse/, yolo_fine/, tiles/, crops_fgc/, detections/)
```

## Pipeline tuần tự (`src/run_pipeline.py`)

**Data prep** (thứ tự phụ thuộc): `classes` → `manifest` → `yolo` → `tiles` → `crops`
- `build_classes.py` — sinh classes.yaml từ COCO categories (coarse sort abc, rare = bottom-decile theo số instance)
- `build_manifest.py` — 1 row/ảnh; **real → ép split=test (held-out)**, synthetic → train/val theo ratio
- `coco_to_yolo.py` — COCO → YOLO center-xywh chuẩn hoá [0,1], xuất full-image labels + data.yaml
- `tiling.py` — SAHI slicing; `_axis_starts()` kéo tile cuối sát biên (fix rớt dải mép so với repo gốc)
- `extract_crops.py` — cắt crop cho FGC (chỉ fgc_groups), +10% margin, layout ImageFolder

**Stage 1:** `detect_train` (wrapper YOLO) → `detect_infer` (SAHI slice → YOLO → WBF fuse)
- Output: `derived/detections/<split>/<id>.txt`, format `class cx cy w h score`

**Stage 2:** `fgc_train` (1 checkpoint ResNet/nhóm) → `fgc_infer` (routing)
- Routing: chỉ tinh-phân-loại detection thuộc fgc_group có score ≥ `fgc.route.min_score`
- Sentinel id `-1` cho detection không refine được
- Output: `derived/detections_fgc/<split>/<id>.txt`, thêm cột `fine_class fine_score`

**Eval:** `evaluation.py` — metric detection + FGC, xuất `metrics.json` (+ confusion PNG).
Ablation qua `eval.mode` = `det_only | det_plus_fgc`. FGC eval dùng GT crop (độc lập detector).

## Điểm thiết kế đáng chú ý

- Mỗi stage = subclass `BasePipeline` (`src/pipeline/base.py`), nhận sẵn `DictConfig` đã compose.
  Không stage nào tự hardcode path (đi qua `DataPaths`).
- Config **phẳng, một file** `configs/config.yaml` (không dùng Hydra config-group).
  Override CLI: `detector.imgsz=1280 fgc.loss=ce`.
- **Hai cách chạy tương đương**: `run_pipeline.py <cmd>` (Compose API) hoặc console script
  `pid-*` (mỗi stage một `@hydra.main`, cài qua `uv sync`).
- **Chống rò rỉ**: ảnh `real/` luôn split=test, không lẫn training → phục vụ Phase 3
  (so sánh synthetic vs real held-out).
- **Long-tail**: focal loss (`fgc.loss=focal`), balanced sampler, `rare_fine` bottom-decile,
  macro-F1 chỉ trên lớp hiếm. Class weight = nghịch đảo tần suất chuẩn hoá về mean 1.
- **Không xoay/flip** (`degrees:0`, `hflip:0`) vì ký hiệu P&ID có ý nghĩa hướng.
- Layout 2 package top-level `pipeline/` + `utils/` (mirror repo `PID_Symbol_Detection`),
  không có wrapper package → tránh cài chung env với package khác cũng publish `pipeline`/`utils`.

## Phases (chiến lược dữ liệu, báo cáo §4.2.4)

Không phải config file riêng — chỉ là config mặc định + vài field set tay:
- **phase1** — synthetic baseline (coarse). Default đã khớp.
- **phase2** — synthesis coarse+fine + degradation aug (`fgc.balanced_sampler=true`, đã default).
- **phase3** — real-world held-out eval: `pid-eval eval.split=test eval.mode=det_plus_fgc`,
  so với cùng lệnh trên synthetic test split.

## Utils chính (`src/utils/`)

- `types.py` — `BBox` (center-xywh, frozen), `Detection`, `SymbolInstance`
- `bbox_utils.py` — chuyển đổi YOLO↔xyxy, clip, crop, đọc/ghi YOLO label
- `model.py` — build ResNet-34 (timm) + head/nhóm, FocalLoss, class weight nghịch đảo tần suất.
  **ArcFace** (`metric_learning=true`): `ArcMarginProduct` (angular-margin head) + `ArcFaceModel`
  (backbone→embedding→class centers)

## ArcFace + "Other" reject gate (thêm 2026-09-23)

Bật qua `fgc.metric_learning=true` (mặc định `false` → giữ nguyên baseline CE/focal).

- **Train** (`fgc_train.py`): head thành ArcFace — embedding L2-norm, mỗi fine class là 1
  class-center trên mặt cầu, margin góc `m` siết cùng-lớp lại. Margin chỉ áp lúc train
  (truyền label vào forward). Checkpoint lưu thêm `class_centers`.
- **Infer** (`fgc_infer.py`): **nhãn vẫn softmax-argmax** (giữ nguyên format 7 cột). Thêm
  **cổng "Other"**: cosine tới center gần nhất `< fgc.route.other_min_cosine` → symbol
  ngoài vocab → ghi `fine_id = -2` (`OTHER_FINE_ID`) thay vì ép vào lớp gần nhất.
- **Sentinel** (`classmap.py`): `-1` = un-refined (score thấp), `-2` = Other (embedding xa).
- **Eval** (`evaluation.py`): với ArcFace, đo thêm `other_gate.false_reject_rate` trên GT crop
  (đều in-vocab → cắt ngưỡng ở đây = false reject) để calibrate `other_min_cosine`.
- **Config**: `fgc.arcface.{m,s,embed_dim}`, `fgc.route.other_min_cosine` (0.35 là placeholder,
  CHƯA tune — phải calibrate trên val thật).
- **Tests**: `tests/test_arcface.py` (head/model/checkpoint/reject gate).
- **Giới hạn**: chưa train thật (chưa có data); softmax vẫn quyết nhãn nên cổng Other dựa
  cosine — chỉ đáng tin khi ArcFace đã train hội tụ.
- `dataset.py` — `_GroupCropDataset` (label local theo nhóm) + WeightedRandomSampler
- `detection_metrics.py` — IoU, AP/mAP all-points interpolation (VOC/COCO), per-class AP
- `fgc_metrics.py` — within-group confusion + accuracy, top-k, macro-F1 lớp hiếm
- `cli.py` — resolve `configs/` dir cho `@hydra.main`

## Resume train ngắt-quãng (thêm 2026-09-24)

Cho lịch train nhiều phiên ngắn (vd Colab ~6h/phiên). Xem `docs/COLAB_RUNBOOK.md`.

- **Detector** (`detect_train.py`): `detector.resume=true` + `detector.resume_dir=<run cũ>`
  → Ultralytics resume THẬT: nạp `last.pt` + optimizer/LR/epoch, train tiếp tới `epochs`
  gốc (KHÔNG truyền lại epochs/aug — đọc từ run cũ). Dùng `last.pt`, không phải `best.pt`.
- **FGC** (`fgc_train.py`): `fgc.resume_from=<fgc_checkpoints dir>` → mỗi group nạp
  model+optimizer+epoch từ `<group>.pt`, train tiếp; group thiếu ckpt thì train mới.
  `_maybe_resume` trả `(start_epoch, best_val_acc)` để không ghi đè ckpt tốt hơn.
- Cả hai mặc định tắt (`resume: false`, `resume_from: null`) → hành vi cũ không đổi.
- Tests: `tests/test_resume.py` (path resolution, validate, checkpoint round-trip).
- **Điều kiện Colab**: checkpoint + data phải nằm trên Drive (Colab xoá `/content`).
