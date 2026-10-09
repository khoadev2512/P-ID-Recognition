# Push code lên GitHub + chạy train trên Colab

Luồng: **code qua git, data qua Google Drive.** Data 1.3GB KHÔNG đi qua git
(`.gitignore` đã chặn `data/`). Colab ghép hai nguồn: `git clone` code + mount Drive data.

## Phần A — Máy này: push code lên GitHub (1 lần)

```bash
cd /home/khoanv/Study/P-ID-Recognition-main
git init
git add -A
git status          # KIỂM TRA: không được thấy data/, runs/, .venv, *.pdf
git commit -m "P&ID detection+FGC pipeline: ArcFace, resume, DigitizePID converter"
```

Tạo repo rỗng trên GitHub (github.com → New repository, KHÔNG init README), rồi:

```bash
git remote add origin https://github.com/<USERNAME>/<REPO>.git
git branch -M main
git push -u origin main
```

**Cái gì được push** (nhỏ, gọn): `src/`, `configs/`, `scripts/`, `tests/`, `notebooks/`,
`docs/`, `pyproject.toml`, `uv.lock`, `README.md`, và `models/base_yolo_models/yolov8n.pt`
(6.5MB, init YOLO).
**Cái gì KHÔNG push**: `data/` (1.3GB), `runs/`, `.venv/`, các PDF chuẩn, ảnh scratch.

## Phần B — Upload data lên Google Drive (1 lần)

Chỉ cần upload thư mục **`data/canonical/`** (không cần `data/raw/` — Colab chỉ đọc
canonical). Đưa lên Drive tại:

```
MyDrive/pid/data/canonical/
    images/                    (500 ảnh .jpg, ~1.3GB)
    annotations.coco.json
    digitizepid_splits.json
    classes.yaml               (nếu đã chạy prep; nếu chưa, Colab tự sinh)
    manifest.csv               (nếu đã có; nếu chưa, Colab tự sinh)
```

**Mẹo upload nhanh** (500 file ảnh lên Drive qua web chậm): nén lại rồi upload 1 file, giải nén trên Colab:
```bash
cd /home/khoanv/Study/P-ID-Recognition-main/data
zip -r canonical.zip canonical    # 1 file ~1.3GB, upload nhanh hơn 500 file rời
```
Trên Colab: `!unzip -q /content/drive/MyDrive/pid/canonical.zip -d /content/drive/MyDrive/pid/data/`

## Phần C — Colab: pull + prep + train

1. Mở **`notebooks/train_stage1_colab.ipynb`** trên Colab (hoặc upload file này lên colab.research.google.com).
2. Sửa 2 chỗ trong notebook:
   - Cell 3: `REPO_URL` = link GitHub repo của bạn.
   - Cell 2: `DRIVE_ROOT` nếu bạn đặt data ở path khác.
3. Runtime → Change runtime type → **GPU**.
4. Chạy tuần tự các cell:
   - Cell 5 (prep tiles) chạy **1 lần** — kết quả lưu lên Drive, session sau tự skip.
   - Cell 6 train; nếu Colab ngắt → Cell 7 resume (sửa `RESUME_DIR`).

## Cập nhật code sau này

Sửa code ở máy → `git add -A && git commit && git push` → trên Colab chạy lại Cell 3
(`git pull`). Data trên Drive không đụng tới.

## Lưu ý

- **Checkpoint sống trên Drive** (`run_dir=$RUNS_DIR` = `MyDrive/pid/runs`) → không mất khi Colab ngắt.
- **Tiles ~1.5GB** sinh trên Drive ở `data/derived/tiles/`. Nếu Drive gần đầy, cân nhắc
  dọn hoặc dùng Colab local disk (`/content`) cho tiles — nhưng khi đó phải prep lại mỗi session.
- **Split gốc DigitizePID được giữ nguyên** nhờ `digitizepid_splits.json` (đừng quên upload file này).
