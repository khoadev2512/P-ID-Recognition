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

**Lề top1−top2 (2026-11-14)**: 9a in thêm `δ` (TPR95 trên lề, cùng `val_sel`); 9b/9c/10c in thêm dòng
`rule_a + lề` và AUROC của lề. Lượt 9a cũ không có δ → chạy lại 9a (rẻ, chỉ `val_sel`) rồi 9b, 9c.

**Không làm**: không chạy 9b/9c trước 9a; không đổi τ sau khi thấy số 9b/9c; không chạy lại 9c nhiều lần
(nửa `test`). **Gửi lại**: toàn bộ in ra của 8, 9a, 9b, 9c + ba file `oneshot_*_crops.csv` (để phân tích
lại không cần GPU) + hai `oneshot_*_sweep.png`.

## 5f. Baseline chỉ-ImageNet (B4a) — notebook §10 (2026-11-14)

**Trước khi mở Colab**: commit + push `scripts/make_imagenet_encoder.py`, `oneshot_eval.py`, notebook.
Cần `encoders/__global_lo__.pt` trên Drive (10b lấy nhóm S của R1a). Nên chạy 9a–9b trong cùng phiên
để 10c in "Other" tại τ (không có thì chỉ in gợi ý, AUROC/OSCR vẫn đủ).

| Mục | Việc | Kiểm / gửi lại |
|---|---|---|
| 1–4 | mount, clone, giải nén crop, đọc H | như §5d |
| 10a | tạo `__imagenet__.pt` (không train, ~1 phút) + `iso_prototypes_imagenet.pt` | separation mean/max cạnh lo/full |
| 10b | τ trên `val_sel`, `++eval.oneshot.groups_ckpt=$ENC_DIR/__global_lo__.pt` | lớp dùng = S1/S17/S21/S24 như R1a (notebook assert) |
| 10c | P1 với encoder ImageNet | AUROC, OSCR, `h_in` top-1/top-3, từng lớp — in cạnh R1a |
| 10d | CL2N (2026-11-15): tâm (`protocol=center`, crop train, 200/lớp) cho ImageNet và R1a → `encoders/center_<tag>.pt`; τ lại (`val_sel`) + P1 lại với `++oneshot.center` | lớp τ = như 9a (assert); "bỏ lớp giữ ra" = [] với groups R1a |
| 10e | bảng 4 cột ImageNet / +CL2N / R1a / +CL2N, τ chọn lại, từng lớp | cùng `n_crops` (assert); hub ImageNet thô ~0.59 |

10a–10c đã chạy 2026-11-15: chỉ cần mục 1–4 rồi 10d → 10e (10d đọc `eval_tau_lo` của 9a; 10e dùng
`at_chosen_taus` của 9a: chạy phần def đầu cell 9a, không cần chạy lại τ). Code mới: push trước khi mở Colab.

**Gửi lại**: in ra 10a–10e + `eval_p1_imagenet/*/oneshot_p1_crops.csv`, `eval_p1_imagenet_cl2n/*/oneshot_p1_crops.csv`
(+ CSV τ nếu tiện).

## 5g. P4 + Stage 1 trên OPEN100-ISO — notebook §11 + §12 (2026-11-14)

**Trước khi mở Colab**:
1. Export lại OPEN100-ISO trên máy để có vùng bỏ qua: **đã làm 2026-11-14**.
2. Zip: `cd data/iso_open100 && zip -r -0 iso_open100_canonical.zip canonical` (~1.9 GB) → đặt ở
   `<DRIVE_ROOT>/data/iso_open100_canonical.zip`.
3. Commit + push code B6.
4. Cần trên Drive: ba encoder + prototype (`encoders/`), τ/δ của 9a (và 10b cho ImageNet), và
   `best.pt` của detector coarse → điền `DET_CKPT` ở cell 12a (cell in mọi `best.pt` tìm thấy và dừng nếu sai).

| Mục | Việc | Kiểm / gửi lại |
|---|---|---|
| 11a ▶▶ | giải nén OPEN100-ISO vào `/content/iso_open100` (`LOCAL_ISO_ROOT`) | 864 ảnh, 93,800 symbol (không đếm `iscrowd`) |
| 11b | P4 mức crop: R1b, R1a, ImageNet (ImageNet nhóm như R1a) | — |
| 11c | bảng cạnh nhau (suy giảm trước, sạch sau) + tại τ/δ của 9a | toàn bộ in ra + `eval_p4_*/*/oneshot_p4_crops.csv` |
| 12a | detector trên OPEN100-ISO, gộp theo nhãn và `agnostic_merge`; chấm không phân lớp, recall theo nhóm (R1b, R1a) | bảng recall theo nhóm, số dự đoán rơi vào vùng bỏ qua |
| 12b | end-to-end P4 (R1b, R1a) trên detection `agnostic_merge`, tại τ, rồi τ + δ | khối `oneshot_e2e` |
| 12c | detector + one-shot trên DigitizePID `val`, chấm `val_sel` → tỉ lệ box sai bị loại | `wrong_boxes` |

Detection lưu trên Drive (`runs/stage1_iso/`, `runs/stage1_dpid/`) — phiên sau dùng lại, không chạy lại
detector. 864 tờ × mỗi lượt detector / encoder: lâu, nên chạy §12 trong phiên riêng.

## 5h. PHASE C Bước 2B — DINOv2 + A4 + CL2N — notebook §13 (2026-11-16)

**Trước khi mở Colab**: commit + push (`src/utils/input_norm.py`, `utils/model.py`, `oneshot_infer.py`, `iso_prototypes.py`,
`oneshot_eval.py`, `scripts/make_imagenet_encoder.py`, `scripts/oneshot_probe.py`, notebook). Cần trên Drive: `encoders/__global_lo__.pt`,
`center_imagenet.pt`, `center_lo.pt` (10d), lượt τ `eval_tau_lo`, `eval_tau_imagenet_cl2n`, `eval_tau_lo_cl2n`, và zip OPEN100-ISO (11a).
Không train: phiên chủ yếu là nhúng crop (P1 ~17k crop, P4 ~94k crop × 3 lượt). DINOv2 tải trọng số HF (~90 MB) lần đầu.

| Mục | Việc | Kiểm / gửi lại |
|---|---|---|
| 1–4, def 9a | mount, clone, crop, H; phần def đầu cell 9a (`at_chosen_taus`) | như §5d |
| 13a | `__dinov2__.pt` (không train) + `iso_prototypes_dinov2_a4.pt` | assert `input_norm`, `imgsz=224`, 42 prototype; separation |
| 13b | tâm (`protocol=center`, nhóm R1a) → τ trên `val_sel` → P1 đủ, bảng cạnh R1a / ImageNet + CL2N | lớp τ = như 9a (assert); cùng `n_crops` |
| 11a | giải nén OPEN100-ISO (nếu phiên mới) | 864 ảnh, 93,800 ký hiệu |
| 13c–13d | P4: DINOv2 + A4 + CL2N, ImageNet + CL2N, R1a + CL2N; bảng 5 cột cạnh R1a / ImageNet | cùng số crop (assert) |
| 13e | (tuỳ chọn) end-to-end DINOv2 — cần cell 12a trong phiên (`DET_CKPT`), rồi chạy lại 12b (bảng) | khối `oneshot_e2e` |

**Gửi lại**: toàn bộ in ra của 13a–13e + `eval_p1_dinov2/*/oneshot_p1_crops.csv`, `eval_p4_{dinov2,imagenet_cl2n,lo_cl2n}/*/oneshot_p4_crops.csv`.

## 5i. PHASE C §2E.3 — kiểm lại F0 (căn chỉnh ảnh mẫu) — notebook §14 (2026-11-18)

**Trước khi mở Colab**:
- Commit + push: `src/utils/linear_align.py`, `src/pipeline/oneshot_infer.py`, `src/pipeline/oneshot_eval.py`, notebook,
  `scripts/f0_linear_align.py`, `scripts/f0_colab_compare.py`, các test.
- **Tải lên Drive** `encoders/pairs_dinov2_a4.pt`, chép từ máy `runs/phase_c/f0/pairs_dinov2_a4.pt` (3.7 MB; `runs/` không vào git).
- Trên Drive phải còn từ §13: `__dinov2__.pt`, `iso_prototypes_dinov2_a4.pt`, `center_dinov2_a4.pt`, `__global_lo__.pt`, các lượt
  `eval_{tau,p1,p4}_dinov2`.

Không train, không dựng lại encoder / prototype / tâm. Chi phí ≈ 2 × (τ + P1 + P4) của 13b–13c.

| Mục | Việc | Kiểm / gửi lại |
|---|---|---|
| 1–4, def 9a, 11a | mount, clone, crop, H; OPEN100-ISO | như §5h |
| 14a | `f0a` và `shift`: τ trên `val_sel` → P1 đủ | `template_align.kind` đúng (assert); cùng `n_crops` với 2B; in `shift_norm`, `max_template_diff` |
| 14b | P4 cho hai dòng | `template_align.kind` đúng (assert) |
| 14c | bảng nhanh cạnh 2B | — |

Nếu 14a dừng với "template_pairs templates differ from the prototypes", prototype trên Drive đã được dựng khác (lệch > 1e-3) so với
file cặp. Khi đó **không** nới ngưỡng: gửi lại thông báo lỗi.

**Gửi lại**: in ra của 14a–14c + 4 file trong `runs/f0_csv/` (`oneshot_{p1,p4}_crops_dinov2_{f0a,shift}.csv`). Đặt 4 file vào `runs/`
trên máy rồi chạy `python scripts/f0_colab_compare.py --out runs/phase_c/f0_colab` (luật §2E.3, bootstrap).

## 5j. PHASE C §2G — Điểm 4: bỏ `pipeline` (41 ảnh mẫu) — notebook §15 (2026-11-19)

**Trước khi mở Colab**:
- Commit + push: notebook, `scripts/nopipe_compare.py`, `tests/test_nopipe_compare.py`, `tests/test_colab_notebook.py`.
- Trên Drive `encoders/` phải còn: `__dinov2__.pt`, `iso_prototypes_dinov2_a4.pt`, `center_dinov2_a4.pt` (§13), `__global__.pt`
  (R1b, nhóm P2), `__global_lo__.pt` (R1a, nhóm τ / P4); và các lượt `eval_{tau,p1,p4}_dinov2` của §13 (bảng 15c đọc).

Không train, không dựng lại gì; chỉ thêm `'++oneshot.exclude_prototypes=[pipeline]'` (có nháy đơn để shell không hiểu `[...]` là glob).
Chi phí ≈ τ + 2 × P2 + P1 + P4 của 2B.

| Mục | Việc | Kiểm / gửi lại |
|---|---|---|
| 1–4, 11a | mount, clone, crop; OPEN100-ISO | như §5h |
| 15a | τ 41 (val_sel) → P2 42 và 41 (val `test`, nhóm R1b) → P1 đủ 41 | `n_prototypes` 41 / 42 (assert); cùng `n_crops` |
| 15b | P4 41 | `n_prototypes` 41 (assert) |
| 15c | bảng nhanh 42 / 41 (* = hậu kiểm) | — |

**Gửi lại**: in ra của 15a–15c + 4 file trong `runs/nopipe_csv/` (`oneshot_p2_crops_dinov2.csv`,
`oneshot_{p1,p2,p4}_crops_dinov2_nopipe.csv`). Đặt vào `runs/` trên máy rồi chạy
`python scripts/nopipe_compare.py --out runs/phase_c/nopipe` (luật §2G: kiểm nhất quán trước, rồi bootstrap).

## 5k. PHASE C §2K.6 — dạng B của butterfly + gộp view `max` — notebook §16 (2026-11-20)

**Trước khi mở Colab**:
- **Chạy §15 trước** (mốc = 2B 41 ảnh mẫu, `dinov2_nopipe`); 16a dừng nếu thiếu.
- Commit + push: notebook, `.gitignore` (un-ignore `data/raw/iso_symbols_forms/`), **`data/raw/iso_symbols_forms/valve_butterfly@B.png`**,
  `src/utils/oneshot.py`, `src/pipeline/{oneshot_infer,oneshot_eval,iso_prototypes}.py`, `scripts/{make_iso_forms,k4_colab_compare}.py`
  và các test. Ảnh dạng B đi qua git (như `iso_symbols_clean`); 16a dừng nếu thiếu.
- Trên Drive `encoders/`: như §5j (`__dinov2__.pt`, `iso_prototypes_dinov2_a4.pt`, `center_dinov2_a4.pt`, `__global__.pt`,
  `__global_lo__.pt`).

Không train, không dựng lại encoder / tâm; chỉ dựng thêm một file prototype 43 hàng (`iso_prototypes_dinov2_a4_formB.pt`, vài giây).
Chi phí ≈ 3 × (τ + P2 + P1 + P4) của 2B.

| Mục | Việc | Kiểm / gửi lại |
|---|---|---|
| 1–4, 11a | mount, clone, crop; OPEN100-ISO | như §5h |
| 16a | dựng 43 prototype (`++oneshot.extra_forms_dir`) | 42 hàng đầu trùng file 13a (≤ 1e-5), `extra_forms` = `['valve_butterfly@B']` (assert) |
| 16b | mỗi dòng `formB` / `max` / `formB_max`: τ (val_sel) → P2 (val `test`, nhóm R1b) → P1 đủ | `n_prototypes` 42 / 41 / 42 và `aggregate` mean / max / max (assert); P1 cùng `n_crops` với mốc |
| 16c | P4 cho 3 dòng | như 16b (assert) |
| 16d | bảng nhanh cạnh `base41` | — |

**Gửi lại**: in ra của 16a–16d + 9 file trong `runs/k4_csv/` (`oneshot_{p1,p2,p4}_crops_dinov2_{formB,max,formB_max}.csv`) và 3 file
mốc của §15 trong `runs/nopipe_csv/` (`oneshot_{p1,p2,p4}_crops_dinov2_nopipe.csv`). Đặt cả 12 vào `runs/` trên máy rồi chạy
`python scripts/k4_colab_compare.py --out runs/phase_c/k4_colab` (luật §2K.6: kiểm nhất quán trước, rồi bootstrap và quyết định).

## 5l. PHASE C §2L — FT-triển khai (DINOv2 fine-tune trên 42 ảnh mẫu) + end-to-end — notebook §17 (2026-11-22)

**Trước khi mở Colab**:
- Encoder đã train **cục bộ** (MPS): `python scripts/ft_templates.py --out runs/phase_c/ft --deploy --device mps` →
  `runs/phase_c/ft/__dinov2_ft_all_a0.5__.pt` (~87 MB) + `.sha256`. **Tải file `.pt` lên `encoders/` trên Drive.** 17a so sha256 với
  `FT_SHA` trong notebook (đã điền) → dừng nếu khác.
- **§16 đã chạy** (mốc `formB`: `eval_{tau,p2,p1,p4}_dinov2_formB`, `iso_prototypes_dinov2_a4_formB.pt`); 17a dừng nếu thiếu.
- Commit + push: notebook, `scripts/{ft_templates,ft_colab_compare}.py` và các test (pipeline không đổi).
- Trên Drive `encoders/`: `__dinov2__.pt`, `iso_prototypes_dinov2_a4_formB.pt`, `center_dinov2_a4.pt`, `__global__.pt`, `__global_lo__.pt`,
  `__dinov2_ft_all_a0.5__.pt`.

Không train trên Colab. Dựng 43 prototype + tâm CL2N với encoder FT (vài phút), rồi τ + P2 + P1 + P4 cho một dòng (`ft`), rồi e2e cho
hai dòng (`formB`, `ft`: mỗi dòng một lượt `fgc --infer` GPU + một `evaluate`; Stage 2 lưu ở `stage1_iso/detections_oneshot_dinov2_*`,
phiên sau không chạy lại; khoá của `ft` gồm 8 ký tự đầu của sha256).

| Mục | Việc | Kiểm / gửi lại |
|---|---|---|
| 1–4, 11a | mount, clone, crop; OPEN100-ISO | như §5h |
| 12a + def 9a | detection `agnostic_merge` đã lưu; `chosen_taus` | chỉ cần cho 17d |
| 17a | kiểm sha256 + meta; 43 prototype + tâm CL2N bằng encoder FT | tên prototype trùng file `formB`; tâm cùng `n_crops` / lớp với tâm §13 (assert) |
| 17b | τ (val_sel, R1a) → P2 (val `test`, R1b) → P1 đủ | `n_prototypes` 42, mean, có tâm, A4; cùng crop / lớp với `formB` (assert) |
| 17c | P4 | như 17b (assert) |
| 17d | e2e `formB` và `ft`, τ quy tắc A của từng dòng | τ trong metrics = τ đã chọn (assert) |
| 17e | bảng nhanh `formB` / `ft` + e2e | — |

**Gửi lại**: in ra của 17a–17e + 5 file trong `runs/ft_csv/` (`oneshot_{p1,p2,p4}_crops_dinov2_ft.csv`, `e2e_dinov2_{formB,ft}.json`).
Đặt vào `runs/` trên máy (CSV `formB` của §16 đã có ở đó) rồi chạy `python scripts/ft_colab_compare.py --out runs/phase_c/ft_colab`
(luật (ii) §2K.6 + chặn (b) của §2L; e2e chỉ mô tả).

## 5m. PHASE C §2M — OPEN100-ISO v2 (sửa lỗi render + khung bản vẽ): `formB` và `ft` — notebook §18 (2026-11-23)

**Trước khi mở Colab**:
- Upload `data/iso_open100_v2/iso_open100_v2_canonical.zip` (2.7 GB, đã tạo ở máy; tạo lại:
  `cd data/iso_open100_v2 && zip -r -0 iso_open100_v2_canonical.zip canonical`) thành `<DRIVE_ROOT>/data/iso_open100_v2_canonical.zip`.
  18a kiểm 864 ảnh, 93,794 annotation ký hiệu + 12,096 vùng bỏ qua, mỗi ảnh có `drawing_box`.
- §16 và §17 đã chạy (mốc v1 + τ của `formB` / `ft`; không chọn lại τ). Không cần file mới trong `encoders/`.
- Commit + push: renderer (`src/utils/{page_frame,page_svg,iso_layout,iso_mapping,iso_variants}.py`), `src/pipeline/render_iso.py`,
  `configs/render_iso{,_v2}.yaml`, notebook, test.

Thứ tự trong phiên: 1–4, 11a (v1, cho 12a), 12a (DET_CKPT, `restore`; detection v1 đã lưu), def đầu 9a, 16a–16b (biến; lượt τ / P2 /
P1 đã có thì **chỉ cần chạy phần gán biến** — hoặc chạy lại, kết quả như cũ), 17a–17b (như vậy), 17d (biến `E2E17`), rồi 18a–18e.

| Mục | Việc | Kiểm / gửi lại |
|---|---|---|
| 18a | giải nén v2 vào `/content/iso_open100_v2` | số ảnh / annotation / `drawing_box` (assert) |
| 18b | Stage 1 `agnostic_merge` trên v2 (lưu `runs/stage1_iso_v2/`), chấm không phân lớp | in dòng `detection` |
| 18c | P4 `formB`, `ft` trên v2 | 42 prototype, mean, có tâm, A4 (assert) |
| 18d | e2e `formB`, `ft` trên v2, τ quy tắc A của từng dòng | τ trong metrics = τ đã chọn (assert) |
| 18e | bảng v1 \| v2 + zip detection v2 | — |

**Gửi lại**: in ra của 18a–18e + các file trong `runs/v2_csv/` (`oneshot_p4v2_crops_dinov2_{formB,ft}.csv`,
`e2e_v2_dinov2_{formB,ft}.json`, `detections_v2.zip`). Đặt vào `runs/v2_csv/` trên máy: phần đếm dự đoán rơi vào khung bản vẽ
(`drawing_box`) làm ở máy. **Từ 2026-11-24:** 19c đếm luôn trên Colab (cả detector cũ lẫn mới) → chạy §18 rồi §19 trong cùng phiên,
`detections_v2.zip` không cần tải.

## 5n. PHASE C §2N — Stage 1 một lớp học trên trang ISO (S1-ISO) — notebook mới `train_stage1_iso_colab.ipynb` (2026-11-23)

**Trước khi mở Colab** (máy local, đã tạo sẵn): upload `data/s1iso_tiles.zip` (~6.7 GB) → `<DRIVE_ROOT>/data/s1iso_tiles.zip` và
`data/iso_p4val_v2/iso_p4val_v2_canonical.zip` → `<DRIVE_ROOT>/data/iso_p4val_v2_canonical.zip`. Commit + push (notebook,
`configs/classes_symbol.yaml`, `scripts/s1iso_*.py`, test). Trong notebook **sửa `DET_CKPT_OLD`** (= `DET_CKPT` của notebook Stage 2
mục 12a). **GPU: L4 khuyến nghị** (một phiên); T4 được (~20 phút / epoch, resume mục 6 qua 2 phiên).

| Mục | Việc | Kiểm / gửi lại |
|---|---|---|
| 1–3 | Drive, code, giải nén tile | 50,248 train / 6,924 val (assert) |
| 4 | detector cũ | 8 lớp coarse (assert), in sha256 |
| 5 / 6 | train 20 epoch / resume | — |
| 7 | chốt `encoders/det_s1iso.pt` | đủ 20 epoch, 1 lớp `symbol` (assert) |
| 8 | P4-val v2: cũ, mới (`agnostic_merge`), chấm không phân lớp | 73 ảnh / 4,832 ký hiệu / `drawing_box` (assert) |
| 9 | bảng cũ \| mới | — |

**Gửi lại**: in ra của 7–9 + `runs/s1iso_csv/det_p4v2_{old,new}.json` → đặt vào `runs/s1iso_csv/` trên máy, chạy
`python scripts/s1iso_compare.py --out runs/phase_c/s1iso` (luật 6 §2N). NHẬN → notebook Stage 2 §19 (test v2 + e2e) sẽ được soạn.
**Kết quả 2026-11-24: NHẬN** (`det_s1iso.pt`, sha `57c4c4c61430`; xem PLAN §2N) → §5o.

## 5o. PHASE C §2N luật 7 — Stage 1 mới trên OPEN100-ISO v2, e2e `formB` / `ft` — notebook Stage 2 §19 (2026-11-24)

**Trước khi mở Colab**: `encoders/det_s1iso.pt` đã có trên Drive (mục 7 notebook S1-ISO). Commit + push: notebook,
`scripts/frame_predictions.py`, test. Như §5m: zip v2 trên Drive. **GPU** (suy luận detector + DINOv2).

Thứ tự trong phiên: như §5m (1–4, 11a, 12a, def đầu 9a, 16a–16b, 17a–17b, 17d), **18a–18e** (detector cũ trên v2: mốc của §19; đã lưu
Drive thì không chạy lại), rồi **19a–19c**. Một phiên GPU đủ: mỗi detector chạy một lần trên 864 ảnh, đã lưu thì phiên sau bỏ qua.

| Mục | Việc | Kiểm / gửi lại |
|---|---|---|
| 19a | detector mới `agnostic_merge` trên v2 (lưu `runs/stage1_iso_v2/detections_agnostic_s1iso`), chấm không phân lớp | sha256 + 1 lớp (assert); in dòng `detection` |
| 19b | e2e `formB`, `ft` trên box mới, τ quy tắc A của từng dòng | τ trong metrics = τ đã chọn (assert) |
| 19c | bảng cũ \| mới (đã thấy / chưa thấy, e2e), đếm dự đoán rơi vào khung | — |

**Gửi lại**: in ra của 18a–18e và 19a–19c + thư mục `runs/s1iso_test_csv/` (`det_test_v2_{old,new}.json`,
`e2e_v2_s1iso_dinov2_{formB,ft}.json`, `frame_v2.json`) và `runs/v2_csv/` (trừ `detections_v2.zip`) → đặt cùng tên dưới `runs/` trên máy.

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
