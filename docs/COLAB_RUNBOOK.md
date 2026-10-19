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
| 3 | Cắt crop **toàn cục** (`fgc.global_encoder=true`) ở local, zip lên Drive | `data/derived/crops_fgc_global.zip`, có assert đủ 32 class |
| 4 | Đọc danh sách H từ `oneshot.holdout` trong config (điền ở B5) | Danh sách trống hoặc tên sai thì báo lỗi |
| 5 | **R1a**: train leave-out, `fgc.exclude_classes=<h_in + h_out>` | `runs/encoders/__global_lo__.pt` |
| 6 | **R1b**: train đủ 32 class | `runs/encoders/__global__.pt` |
| 7 | **R2**: `run_pipeline.py prototypes` cho từng encoder + log separation | `runs/encoders/iso_prototypes{_lo,}.pt` |

Lệnh tương đương (chạy tay):
```python
!python src/run_pipeline.py prep --step crops data_root=$LOCAL fgc.global_encoder=true
!python src/run_pipeline.py fgc --train data_root=$LOCAL run_dir=$RUNS/r1a \
    fgc.global_encoder=true fgc.metric_learning=true fgc.epochs=60 "fgc.exclude_classes=[...]"
!python src/run_pipeline.py prototypes oneshot.encoder_ckpt=$RUNS/encoders/__global_lo__.pt \
    oneshot.prototypes=$RUNS/encoders/iso_prototypes_lo.pt \
    oneshot.symbols_dir=/content/pid/data/raw/iso_symbols_clean \
    oneshot.vocab_csv=/content/pid/data/raw/iso_symbols/vocab.csv
```

Cạm bẫy:
- **Không dùng `derived/crops_fgc/` hay `crops_fgc.zip` cũ** (28 class, cắt theo nhóm). Cũng không
  cắt chồng vào thư mục cũ, vì tên file của hai chế độ đánh số khác nhau và sẽ ra crop trùng.
- **Template ISO nằm trong git** (`data/raw/iso_symbols_clean/`), không nằm trên Drive. Vì vậy
  bước prototypes phải trỏ `oneshot.symbols_dir` và `oneshot.vocab_csv` vào repo đã clone.
- **Resume R1a phải dùng cùng danh sách H.** Nếu khác, code dừng: head cùng kích thước nhưng
  cột sẽ mang nghĩa của class khác.
- **Checkpoint chỉ ghi khi `val_acc` tăng.** Phiên resume nào không tăng `val_acc` thì không ghi
  checkpoint mới, nên phiên sau resume lại từ epoch tốt nhất cũ và mất các epoch đã chạy. Vì vậy
  nên chạy hết 60 epoch trong một phiên nếu kịp: một epoch khoảng 59k crop.
- **`evaluate` chưa dùng được với encoder toàn cục.** Nó lặp theo `fgc_groups` nên bỏ qua im lặng.
  Phần đánh giá one-shot là B6b.

## 6. Infer + eval (sau khi train xong)

```python
BEST=/content/drive/MyDrive/pid/runs/<ts>/train/weights/best.pt
FGC=/content/drive/MyDrive/pid/runs/<ts>/fgc_checkpoints

!uv run python src/run_pipeline.py detect --infer detector.ckpt=$BEST
!uv run python src/run_pipeline.py fgc    --infer fgc.metric_learning=true fgc.ckpt_dir=$FGC
!uv run python src/run_pipeline.py evaluate eval.mode=det_plus_fgc
```

## Lưu ý / cạm bẫy

- **Chia epoch theo thời lượng phiên**: đặt `epochs` tổng lớn, cứ để mỗi phiên chạy tới
  khi ngắt — resume lo phần còn lại. YOLO lưu `last.pt` mỗi epoch nên mất tối đa 1 epoch.
- **`resume_dir` (detector) vs `resume_from` (FGC)** khác nhau: detector trỏ vào **run dir**
  (chứa `train/weights/last.pt`); FGC trỏ vào **thư mục `fgc_checkpoints`**.
- **best.pt vs last.pt**: resume detector dùng `last.pt` (đang dở); infer/eval dùng `best.pt`.
- Nếu `uv sync` kéo torch CPU: cài torch CUDA trước (xem README), rồi `uv sync` lại.
- Cơ chế resume này đã có **unit test** (`tests/test_resume.py`); logic ArcFace ở
  `tests/test_arcface.py`.
