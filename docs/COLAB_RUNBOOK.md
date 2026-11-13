# Colab Runbook — chạy & train ngắt-quãng (split 6h sessions)

Runbook để chạy pipeline trên Google Colab, train được nhiều phiên ngắn (vd 6 tiếng
rồi tiếp) nhờ **resume thật** (đã thêm vào code): detector dùng Ultralytics resume,
FGC dùng `fgc.resume_from`.

> Điều kiện sống còn: **checkpoint + data phải nằm trên Google Drive**, không phải
> `/content` (Colab xoá `/content` khi hết session).

## 0. Bật GPU + mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```
Runtime → Change runtime type → **GPU** (T4 là đủ).

## 1. Lấy code + cài deps

```python
!git clone <repo-url> /content/pid      # hoặc copy từ Drive
%cd /content/pid
!pip install -q uv
!uv sync                                 # torch/timm/ultralytics...; dùng CUDA wheel sẵn của Colab
```

## 2. Trỏ data + runs sang Drive (bền qua các phiên)

```python
import os
os.environ["PID_DATA_ROOT"] = "/content/drive/MyDrive/pid/data"
RUNS = "/content/drive/MyDrive/pid/runs"     # mọi lệnh train dưới đây thêm run_dir=$RUNS
```

## 3. Data prep (một lần)

```python
!uv run python src/run_pipeline.py prep --step all \
    data_root=/content/drive/MyDrive/pid/data
```

## 4. Train DETECTOR — resume thật

**Phiên 1** (chạy tới khi Colab ngắt; YOLO tự lưu `last.pt` mỗi epoch vào run dir):
```python
!uv run python src/run_pipeline.py detect --train \
    detector.epochs=100 \
    run_dir=/content/drive/MyDrive/pid/runs
# -> tạo run dir mới, vd runs/2026-01-01_10-00-00/
```

**Phiên 2+** (session mới sau khi ngắt) — nối lại ĐÚNG chỗ (optimizer + LR + epoch):
```python
!uv run python src/run_pipeline.py detect --train \
    detector.resume=true \
    detector.resume_dir=/content/drive/MyDrive/pid/runs/2026-01-01_10-00-00
```
- `resume=true` → nạp `last.pt` + optimizer/LR/epoch, train tiếp **tới `epochs` gốc**.
- **Không** truyền lại `epochs/imgsz/aug` — Ultralytics đọc lại từ run cũ.
- Lặp lại phiên 2 với **cùng** `resume_dir` cho tới khi đủ epoch.

## 5. Train FGC (ArcFace) — resume qua `resume_from`

**Phiên 1:**
```python
!uv run python src/run_pipeline.py fgc --train \
    fgc.metric_learning=true fgc.epochs=60 \
    run_dir=/content/drive/MyDrive/pid/runs
# -> checkpoints vào runs/<ts>/fgc_checkpoints/<group>.pt
```

**Phiên 2+:**
```python
!uv run python src/run_pipeline.py fgc --train \
    fgc.metric_learning=true fgc.epochs=60 \
    fgc.resume_from=/content/drive/MyDrive/pid/runs/<ts-phiên-1>/fgc_checkpoints \
    run_dir=/content/drive/MyDrive/pid/runs
```
- Mỗi group có `<group>.pt` trong `resume_from` → nạp model+optimizer+epoch, train tiếp.
- Group nào chưa có checkpoint → train từ đầu.
- Checkpoint MỚI ghi vào run dir mới của phiên này → phiên sau trỏ `resume_from` vào đó.

## 5b. Stage 2 one-shot — encoder toàn cục (`notebooks/train_stage2_colab.ipynb`)

Đây là cấu hình dùng cho luận văn (docs/PLAN_STAGE2_ONESHOT.md, Phase B). Notebook làm theo thứ tự:

| Cell | Việc | Kết quả |
|---|---|---|
| 3 | Cắt crop **toàn cục** (`fgc.global_encoder=true`) ở local, zip lên Drive | `data/derived/crops_fgc_ctx.zip`, có assert đủ 32 class và `_boxes.csv` ở mọi lớp |
| 4 | Đọc danh sách H từ `oneshot.holdout` trong config (điền ở B5); đặt `AUG` | Danh sách trống hoặc tên sai thì báo lỗi |
| 6 | **R1b** (chạy trước, 2026-11-09): train đủ 32 class | `runs/encoders/__global__.pt` |
| 5 | **R1a**: train leave-out, `fgc.exclude_classes=<h_in + h_out>` | `runs/encoders/__global_lo__.pt` |
| 7 | **R2**: `run_pipeline.py prototypes` cho từng encoder + log separation | `runs/encoders/iso_prototypes{_lo,}.pt` |

Lệnh tương đương (chạy tay):
```python
!python src/run_pipeline.py prep --step crops data_root=$LOCAL fgc.global_encoder=true
!python src/run_pipeline.py fgc --train data_root=$LOCAL run_dir=$RUNS/r1a \
    fgc.global_encoder=true fgc.metric_learning=true fgc.epochs=60 "fgc.exclude_classes=[...]" \
    ++fgc.aug.degrade.p=0.5 ++fgc.aug.box_jitter.p=0.5
!python src/run_pipeline.py prototypes oneshot.encoder_ckpt=$RUNS/encoders/__global_lo__.pt \
    oneshot.prototypes=$RUNS/encoders/iso_prototypes_lo.pt \
    oneshot.symbols_dir=/content/pid/data/raw/iso_symbols_clean \
    oneshot.vocab_csv=/content/pid/data/raw/iso_symbols/vocab.csv
```

Cạm bẫy:
- **Không dùng `derived/crops_fgc/` hay `crops_fgc.zip` cũ** (28 class, cắt theo nhóm). Cũng không
  cắt chồng vào thư mục cũ, vì tên file của hai chế độ đánh số khác nhau và sẽ ra crop trùng.
- **Không dùng `crops_fgc_global.zip`** (crop cắt trước 2026-11-11, không có `_boxes.csv`). Crop mới
  lưu vùng ngữ cảnh + vị trí box (`utils/crop_box.py`); box jitter cần nó và dừng với lỗi nếu thiếu.
- **R1a và R1b dùng cùng `AUG`** (suy giảm kiểu scan + lệch box), kể cả khi resume. Resume thiếu
  override thì phần còn lại của run train không có augment mà không báo gì.
- **Template ISO nằm trong git** (`data/raw/iso_symbols_clean/`), không nằm trên Drive. Vì vậy
  bước prototypes phải trỏ `oneshot.symbols_dir` và `oneshot.vocab_csv` vào repo đã clone.
- **Resume R1a phải dùng cùng danh sách H.** Nếu khác, code dừng: head cùng kích thước nhưng
  cột sẽ mang nghĩa của class khác.
- **Checkpoint (2026-11-13)**: `__global__.last.pt` mỗi epoch (resume tiếp từ đó, mất tối đa một
  epoch); `__global__.ep<N>.pt` ở các epoch trong `++fgc.save_epochs`; `__global__.pt` = best
  `val_acc`, **không dùng để so** (val_acc bão hoà từ epoch 1). R1a/R1b so ở epoch 3 (`CMP_EPOCH`).
- **`evaluate` chưa dùng được với encoder toàn cục.** Nó lặp theo `fgc_groups` nên bỏ qua im lặng.
  Phần đánh giá one-shot là B6b.

## 5c. Chạy R1b — từng bước (2026-11-11)

R1b = encoder đủ 32 lớp, phần lõi luận văn; chạy **trước** R1a. Notebook:
`notebooks/train_stage2_colab.ipynb`.

**0. Trước khi mở Colab (máy local)**
- Code mới phải có trên GitHub: Colab `git clone` nhánh `feat/prepare-for-colab-stage-2`. Kiểm
  `git log origin/feat/prepare-for-colab-stage-2 --oneline -1` trùng commit local mới nhất.
- Drive có `MyDrive/pid/data/canonical/` (hoặc `MyDrive/PID_Recognition/...`) với `images/`,
  `annotations.coco.json`, `classes.yaml`, `manifest.csv` của DigitizePID — cùng chỗ đã dùng
  cho Stage 1.

**1. Runtime**: Runtime → Change runtime type → **T4 GPU**.

**2. Chạy theo thứ tự mục** (mục 6 nằm sau mục 5 trong file):

| Mục | Cell | Việc | Kiểm |
|---|---|---|---|
| 1 | GPU + Drive | mount Drive, dò `DRIVE_ROOT` | in ra `DRIVE_ROOT` và số ảnh canonical (500) |
| 2 | Code + deps | **sửa `REPO_URL`** thành repo của bạn; clone + cài deps | dòng `git log` cuối là commit mới nhất |
| 3 | Crop toàn cục | phiên đầu: copy ảnh, cắt crop định dạng mới, zip `crops_fgc_ctx.zip` lên Drive; phiên sau: chỉ giải nén | `train` và `val` đều 32 lớp, có `_boxes.csv`; ~47.7k crop train, ~11.8k val |
| 4 | Danh sách H + `AUG` + `LOADER` | đọc `oneshot.holdout` (R1b không dùng H nhưng cần `AUG`, `LOADER`) | in H_in 6, H_out 4, h_extra 3; `AUG`, `LOADER` |
| **6** | **R1b train** | (2026-11-12: 60 epoch; nay `EPOCHS`=5 + `"$SAVE"`) batch 128, `$AUG $LOADER` | log `[fgc/__global__] 32 fine classes`; loss giảm |
| 6a | Resume R1b | chỉ khi Colab ngắt giữa chừng; tự lấy run mới nhất, cùng `AUG` | in "resume từ ..." |
| 6b | Finalize | kiểm checkpoint đủ 32 lớp, không lớp nào bị loại → copy `runs/encoders/__global__.pt` | in số lớp, epoch, `val_acc` |
| 7 | R2 (tuỳ chọn) | prototype ISO cho `__global__.pt` (bỏ qua `_lo` chưa có) | dòng `prototype separation: mean …, max …` |
| 8 | Eval C (2026-11-13) | `evaluate eval.mode=fgc_only` trên crop val, không cần detector | in top-1, top-3, macro-F1, tỉ lệ nhảy họ, 5 lớp F1 thấp nhất; `runs/eval_c/<ts>/metrics.json` |

**3. Cái gì là bình thường**
- Phiên đầu, mục 3 mất thêm thời gian copy ~1.2 GB ảnh và cắt ~59.5k crop. Zip crop khoảng 1 GB
  (crop giờ lưu cả vùng ngữ cảnh, ~16 KB/crop đo trên máy local).
- Augment (subpart, suy giảm, lệch box) chạy trên CPU; `LOADER` dùng 2 worker. Thời gian mỗi
  epoch **chưa đo trên Colab** (local: ~1.2 ms nạp một crop) — ghi lại thời gian epoch đầu để
  ước lượng có kịp 60 epoch trong một phiên không.
- `val_acc` closed-set 32 lớp nên cao (plan B3: > 0.95). Thấp hơn nhiều là dấu hiệu lỗi dữ liệu.
- Checkpoint: xem §5b (`.last.pt`, `.ep<N>.pt`; 2026-11-13).

**4. Không làm**
- Không xem kết quả từng lớp rồi sửa `oneshot.holdout`: H đã chốt (HANDOFF §6c).
- Không đổi `AUG` / `LOADER` giữa train và resume (notebook đã dùng chung một chuỗi).
- Không dùng zip crop cũ (`crops_fgc.zip`, `crops_fgc_global.zip`).

**5. Gửi lại sau khi chạy**: `val_acc` cuối + epoch tốt nhất, thời gian một epoch, đường dẫn
`runs/encoders/__global__.pt`, dòng `prototype separation` nếu đã chạy mục 7, dòng tóm tắt của
mục 8 + `metrics.json` + `confusion___global__.png`, và lỗi nếu có.

## 5d. Chạy R1a — từng bước (2026-11-13)

R1b **đã xong** (2026-11-12, epoch 3 → `runs/encoders/__global__.pt`). R1a = encoder không thấy
`h_in + h_out + h_extra` (13 lớp), cho protocol P1. Quy tắc checkpoint chốt trước: **so ở epoch 3**
(PLAN "Quyết định 2026-11-13 — chọn checkpoint").

**Trước khi mở Colab**: push nhánh (notebook cần `holdout_classes` 3 danh sách + `fgc.save_epochs`).

| Mục | Cell | Việc | Kiểm |
|---|---|---|---|
| 1–3 | GPU, code, crop | như §5c (crop: chỉ giải nén zip) | dòng `git log` là commit mới nhất; 32 lớp ở cả hai split |
| 4 | H + `AUG` + epoch | đọc `oneshot.holdout`; đặt `CMP_EPOCH=3`, `EPOCHS=5`, `SAVE` | H_in 6, H_out 4, h_extra 3; "R1a train trên 19/32 class" |
| 4b | `finalize` | chỉ định nghĩa hàm | không in gì |
| **5** | **R1a train** | 5 epoch, `$AUG $LOADER "$SAVE"`, `fgc.exclude_classes=$EXCLUDE` | log `holding out 13 classes`, `19 fine classes`; `epoch 3 -> saved …ep3.pt` |
| 5a | Resume R1a | chỉ khi Colab ngắt; tự lấy phiên mới nhất, train tiếp từ `.last.pt` | in "resume từ …"; log `resuming from …last.pt` |
| 5b | Finalize | lấy `__global__.ep3.pt`, kiểm không rò rỉ → `runs/encoders/__global_lo__.pt` | in 19 class, **epoch 3** |
| 7 | R2 | prototype cho cả `__global_lo__.pt` và `__global__.pt` | hai dòng `prototype separation` |

**Không làm**: không đổi `CMP_EPOCH` sau khi thấy số; không lấy `__global__.pt` (best `val_acc`) của
R1a; không xem số P1 rồi sửa H.

**Gửi lại**: log 5 epoch (loss, `val_acc`, thời gian/epoch), dòng in của 5b, hai dòng `prototype
separation` của mục 7, lỗi nếu có.

## 5e. Eval one-shot mức crop — notebook §8 + §9 (2026-11-13)

**Trước khi mở Colab**: commit + push code B6b-1 **và** `configs/splits/digitizepid_val_test.json`
(notebook đọc file này từ repo đã clone).

| Mục | Cell | Việc | Kiểm / gửi lại |
|---|---|---|---|
| 1–4 | như §5d | mount, clone, giải nén crop, đọc H | 32 lớp ở cả hai split |
| 7 | R2 | chỉ nếu `encoders/iso_prototypes*.pt` chưa có | — |
| 8 | C | `fgc_only ++eval.image_subset=test` | top-1, macro-F1, 5 lớp yếu nhất |
| **9a** | **chọn τ** | cả `lo` và `full`, chỉ `val_sel`, lớp nhóm S | in τ quy tắc A / B / B bỏ cặp sinh đôi, số crop, lớp dùng |
| 9b | P1 | `__global_lo__.pt`, `h_in` vs `h_out` | macro top-1/top-3 `h_in`, AUROC, OSCR, từng lớp, "Other" tại ba τ |
| 9c | P2 | `__global__.pt`, nửa `test` | S / V / V_of_S / OOV_seen, AUROC, OSCR, "Other" tại ba τ |

**Không làm**: không chạy 9b/9c trước 9a; không đổi τ sau khi thấy số 9b/9c; không chạy lại 9c nhiều lần
(nửa `test`). **Gửi lại**: toàn bộ in ra của 8, 9a, 9b, 9c + ba file `oneshot_*_crops.csv` (để phân tích
lại không cần GPU) + hai `oneshot_*_sweep.png`.

## 6. Infer + eval (sau khi train xong)

```python
BEST=/content/drive/MyDrive/pid/runs/<ts>/train/weights/best.pt
FGC=/content/drive/MyDrive/pid/runs/<ts>/fgc_checkpoints

!uv run python src/run_pipeline.py detect --infer detector.ckpt=$BEST
!uv run python src/run_pipeline.py fgc    --infer fgc.metric_learning=true fgc.ckpt_dir=$FGC
!uv run python src/run_pipeline.py evaluate eval.mode=det_plus_fgc
```

Encoder toàn cục (closed-set, protocol C; PLAN B6c, 2026-11-13):

```python
ENC=/content/drive/MyDrive/pid/runs/encoders            # chứa __global__.pt (mục 6b notebook)
GLOBAL="fgc.global_encoder=true fgc.metric_learning=true fgc.ckpt_dir=$ENC"

# Chỉ Stage 2, trên crop GT (không cần detector):
!uv run python src/run_pipeline.py evaluate eval.mode=fgc_only eval.split=val $GLOBAL
# End-to-end, mAP 32 lớp fine (detector coarse → __global__.pt):
!uv run python src/run_pipeline.py detect --infer detector.ckpt=$BEST eval.split=val
!uv run python src/run_pipeline.py fgc --infer eval.split=val $GLOBAL   # ++fgc.route.by_coarse=true: ablation
!uv run python src/run_pipeline.py evaluate eval.mode=det_plus_fgc eval.split=val $GLOBAL
# Baseline YOLO một-stage 32 lớp fine:
!uv run python src/run_pipeline.py detect --infer detector.ckpt=$BEST_FINE detector.label_level=fine eval.split=val
!uv run python src/run_pipeline.py evaluate eval.mode=det_only detector.label_level=fine eval.split=val
```

Hai lượt `detect --infer` ghi cùng `derived/detections/<split>/`: chấm xong detector này rồi mới
chạy detector kia. `eval.split=val` vì DigitizePID không có split `test`.

## Lưu ý / cạm bẫy

- **Chia epoch theo thời lượng phiên**: đặt `epochs` tổng lớn, cứ để mỗi phiên chạy tới
  khi ngắt — resume lo phần còn lại. YOLO lưu `last.pt` mỗi epoch nên mất tối đa 1 epoch.
- **`resume_dir` (detector) vs `resume_from` (FGC)** khác nhau: detector trỏ vào **run dir**
  (chứa `train/weights/last.pt`); FGC trỏ vào **thư mục `fgc_checkpoints`**.
- **best.pt vs last.pt**: resume detector dùng `last.pt` (đang dở); infer/eval dùng `best.pt`.
- Nếu `uv sync` kéo torch CPU: cài torch CUDA trước (xem README), rồi `uv sync` lại.
- Cơ chế resume này đã có **unit test** (`tests/test_resume.py`); logic ArcFace ở
  `tests/test_arcface.py`.
