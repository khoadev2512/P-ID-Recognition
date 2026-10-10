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
