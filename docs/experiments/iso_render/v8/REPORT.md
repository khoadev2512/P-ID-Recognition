# V8 — Xuất dataset + nối vào pipeline (báo cáo, 2026-11-06)

Commit: xem `git log` (`feat: RENDER_ISO V8 dataset export`). 561 test (548 cũ + 13 mới), ruff sạch trên file đã đụng.
Quyết định V6, V7 giữ nguyên. Không sửa module pipeline/utils có sẵn, `configs/config.yaml`, `data/canonical`, `data/raw`.

## 1. Đã làm gì

| File | Vai trò |
|---|---|
| `src/utils/iso_export.py` | hàm thuần: danh mục COCO, luật nhãn, hộp nút, đọc/ghi GraphML, chia theo gốc, liên kết file |
| `src/pipeline/render_iso.py` | stage `RenderIsoPipeline(BasePipeline)` + `main()` |
| `pyproject.toml` | **một dòng**: `pid-render-iso = "pipeline.render_iso:main"` (không `uv sync`/`uv lock`; chạy bằng `python -m pipeline.render_iso`) |
| `configs/render_iso.yaml` | khối `render_iso.export` |
| `scripts/check_export.py` | thống kê + hình bằng chứng |
| `tests/test_iso_export.py` | 13 test |

Lệnh xuất (đọc `data/iso_open100/variants/` và `degraded/` có sẵn, **không đếm cứng**):

```
PYTHONPATH=src PID_DATA_ROOT=data/iso_open100 .venv/bin/python -m pipeline.render_iso            # = pid-render-iso
... render_iso.export.images=degraded                      # clean | degraded | both (mặc định both)
... render_iso.export.split.mode=by_source render_iso.export.split.test_fraction=0.5
... render_iso.export.link=symlink                         # hardlink (mặc định) | symlink | copy
```
Ghi đè kiểu Hydra, nhưng dùng API `compose` nên không tạo thư mục `outputs/`/`runs/`. Gốc dữ liệu từ `$PID_DATA_ROOT` (mặc định `data/iso_open100`); stage **từ chối** gốc là `data/` hoặc gốc có `raw/`.

## 2. Định dạng (đã đọc code pipeline trước khi viết)

Các ràng buộc từ code đọc:
- `DataPaths.find_image(id)` tìm `canonical/images/<id>.(jpg|jpeg|png|bmp|tiff)` **phẳng**, id = phần tên file sau `/` bỏ đuôi (`coco_to_yolo`, `evaluation`, `extract_crops`, `build_manifest` đều tách `file_name` bằng `/`).
- `build_manifest` bắt `file_name` có tiền tố `synthetic/` hoặc `real/` (`real` luôn là `test`); với `synthetic`, lấy split từ `canonical/digitizepid_splits.json` nếu có, nếu không thì chia ngẫu nhiên train/val.
- `build_classes` suy `classes.yaml` từ `categories` (`name`, `supercategory`, cờ `fgc`); `classmap` đọc `classes.yaml` của **gốc đang dùng**.

Kết quả (`data/iso_open100/`):
```
canonical/images/<stem>.png|jpg     hardlink tới variants/ (sạch, .png) hoặc degraded/ (.jpg)
canonical/annotations.coco.json     images (+ sheet, variant, degraded), annotations (+ node, ann_id, added), categories (+ fgc)
canonical/classes.yaml, manifest.csv   do chính BuildClassesPipeline / BuildManifestPipeline sinh
canonical/digitizepid_splits.json   {stem: split}: để chạy lại pid-prep-manifest ra đúng split này
graphml/<stem>.graphml   splits.json   export_summary.json
```
`stem` = `iso<sheet>_v<NNN>` (sạch) hoặc `iso<sheet>_v<NNN>_deg` (suy giảm); `file_name` = `synthetic/<stem>.<ext>`.

## 3. Danh mục (categories)

- 42 lớp của `vocab.csv` (đúng thứ tự file, `supercategory` = `coarse`, `fgc` = `fgc_group == yes`) + `other_iso`, `other_isa`, `other_generic` (`supercategory = other`, `fgc = false`) = 45. Id ổn định bất kể ảnh nào có mặt.
- Nhãn = trường `label` V5 đã ghi (đã là `fine_name` trong vocab hoặc `other_*`); nhãn lạ → lỗi, không đoán. `vessel_dished_ends` → `other_iso` (đã có từ V5): vì thế `tank` có 0 mẫu.

**Mọi loại `label: None` / không có trong vocab** (quyết định, đổi bằng `drop_labels`):

| Loại | Quyết định | Lý do |
|---|---|---|
| cờ inlet/outlet (`kind: flag`, `label: None`) | **không đưa vào COCO**, vẫn là nút GraphML (nhãn `flag`, `category` rỗng) | không có trong vocab, không phải ký hiệu ISO; detector không nên bị phạt vì nó |
| mũi tên dẫn (`leader_arrow`) | **không đưa vào COCO**, nút GraphML nhãn `leader_arrow` | không có trong vocab; V5 tách riêng đúng vì "kẹp vào lớp `arrow`" là quyết định riêng (chờ user) |
| mũi tên chảy (`flow_direction`, mã 241) | **giữ** | có trong vocab, và DigitizePID cũng có lớp này (28.6% mẫu) |
| hộp trơn `other_generic` | **giữ** | ký hiệu vẽ thật nhưng không có mẫu ISO |
| đĩa + cặp mặt bích | 2 hộp COCO (`<nút>` và `<nút>/flanges`) cùng một nút GraphML (hộp nút = hợp hai hộp) | |
| agitator (`added: true`) | trong COCO (cờ `added`), **không có nút GraphML** | PID2Graph không có nút này |

Cái mà Stage 1 / evaluation mong: `classes.yaml` + `manifest.csv` + COCO của gốc đang dùng; `coco_to_yolo` lấy `supercategory` làm lớp thô nên `other` là một lớp thô riêng. Mũi tên dẫn và cờ không có hộp GT: nếu detector bắt chúng sẽ tính là dương giả; ghi để user biết.

## 4. GraphML

Từ GraphML OPEN100 gốc, **cùng nút (cùng thứ tự id) và cùng cạnh (cùng `edge_label`)**; thêm `orig_label`, `category`, `drawn` và (đồ thị) `image`, `affine`. Khoá `label`, `xmin`, `xmax`, `ymin`, `ymax` vẫn như gốc nên `utils.pid2graph.load_graphml` đọc được (test).
- `label` = danh mục COCO; nút ký hiệu không vào COCO mang loại của nó (`flag`, `leader_arrow`); connector/crossing/background giữ nhãn gốc.
- Toạ độ = **toạ độ đã vẽ**: nút có annotation lấy hợp các hộp annotation (bong bóng: hộp của **vòng tròn đã dời**, nên tâm = tâm hộp COCO); nút còn lại lấy hộp V4 (với ảnh suy giảm: hộp V7 đã dời bằng affine). Agitator không vào hộp nút của bình.
- Ảnh sạch: trang vẽ lại từ seed, **so từng hộp với JSON V6** (lệch > 0.02 px → báo lỗi "không phải thứ config hiện tại vẽ"); giúp phát hiện khi V6 được sinh lại với config khác.

## 5. Chia tập

`splits.json`: `{mode, seed, sheets: {sheet: split}, images: {stem: split}}`; mọi biến thể (sạch + suy giảm) của một gốc cùng split. Hai chế độ:
- `all_test` (**mặc định**): cả 12 gốc vào `test` — đúng mục đích chính của plan (tập test đầu–cuối). Lý do mặc định: pipeline chỉ cần `manifest.csv` có cột `split`; mọi stage đánh giá đọc `test`; không cần train/val trên tập này.
- `by_source`: xáo trộn các gốc theo seed, `round(test_fraction · n)` gốc vào `test`, phần còn lại `train` (luôn ≥ 1 gốc mỗi bên khi có ≥ 2 gốc). Có test (rời nhau theo gốc, khớp `manifest.csv`).

| split | gốc | ảnh |
|---|---|---|
| test | 0–11 (12) | 826 |

(`per_sheet.csv`: số ảnh sạch/suy giảm, số ký hiệu trung bình mỗi tờ.) `by_source` chỉ chạy trong test (2 gốc), không xuất bộ đầy đủ.

## 6. Số liệu (`summary.txt`, `per_class.csv`, `per_sheet.csv`)

- **826 ảnh** (413 sạch + 413 suy giảm), **89,698 annotation**, 11,300 không đưa vào COCO (cờ + mũi tên dẫn, 2 bản).
- **Xuất lại 2026-11-06** sau khi V6 đổi sang lọc trùng theo nội dung: **856 ảnh** (428 + 428), **92,304
  annotation**, 11,880 không đưa vào COCO, 107.8 ký hiệu/ảnh, `other_*` 38.1%, agitator 356, `column` 216,
  `valve_three_way` 78, vẫn 30/42 lớp vocab có mẫu (`summary.txt`, `per_class.csv`, `per_sheet.csv` đã ghi
  đè). `data/canonical` (504 file) và `data/raw` + `configs/config.yaml` (365 mục) trùng checksum với
  `*_checksums_after.txt`. Các số khác trong report này là của lần xuất đầu.
- **Xuất lại lần 2** (cùng ngày, sau khi V5 nối khe): **864 ảnh** (432 + 432), **93,800 annotation**,
  12,096 không đưa vào COCO, 108.6 ký hiệu/ảnh, `other_*` 38.1%, agitator 380, vẫn 30/42 lớp vocab có mẫu.
  Checksum `data/canonical`, `data/raw`, `configs/config.yaml` vẫn trùng.
- **Xuất lại lần 3** (2026-11-08, ảnh có chữ V5c): vẫn 864 ảnh, **93,800 annotation** (chữ không vào COCO),
  checksum vẫn trùng.
- Ký hiệu/ảnh: trung bình 108.6, nhỏ nhất 45, lớn nhất 201 (theo tờ: tờ 8 ~45, tờ 10 ~193).
- **`other_*` = 38.2%** của annotation: `other_isa` 21,478 (23.9%), `other_iso` 9,688 (10.8%), `other_generic` 3,088 (3.4%). `flow_direction` 25,672 (28.6%) và `flanged_connection` 7,928 (8.8%) cũng chi phối; mỗi lớp van 1,180–1,420 (1.3–1.6%).
- Có mẫu: 30/42 lớp vocab. **Không có mẫu**: `pump_diaphragm`, `compressor_centrifugal`, `hx_general`, `hx_straight_tube`, `hx_coil`, `tank` (→ `other_iso`), `vessel_spherical`, `motor_electric`, `pipeline`, `hose`, `piping_heated_cooled_insulated`, `open_disc` (không xuất hiện ở 413 biến thể; chưa tìm nguyên nhân). Ít mẫu: `valve_three_way` 80, `agitator_*` 110–130 (360 tổng), `column` 216.
- `rare_fine` (do `build_classes` tính) = 12 lớp, toàn lớp 0 mẫu.
- Hình: `class_hist.png`.

## 7. Dung lượng, thời gian

- **Ảnh vào `canonical/images/` bằng hardlink: 0 byte thêm** (826/826 `st_nlink ≥ 2`, `-links 1` = 0 file). Hardlink vì pipeline `exists()` + PIL/cv2 đọc bình thường, không phụ thuộc đường dẫn tương đối như symlink. Hệ quả: nếu V6 được sinh lại thì `canonical/images` **giữ bản cũ** đến khi chạy lại xuất (xuất xoá và tạo lại `canonical/images`, `graphml/`).
- Thêm: COCO 19.5 MB, GraphML 128 MB (826 file × ~155 KB), manifest 260 KB. Cả `data/iso_open100/` = 2.0 GB, trong đó `degraded/` 1.75 GB, `variants/` 130 MB.
- Thời gian xuất: 8.5 s (4 tiến trình), gồm vẽ lại 413 trang để lấy hộp nút và đối chiếu.

## 8. Kiểm tra với pipeline (`pipeline_stages.txt`, `pipeline_loaded.txt`)

**Chạy** (`PID_DATA_ROOT=data/iso_open100`, `hydra.run.dir=data/iso_open100/_hydra`, `oneshot.symbols_dir=data/raw/iso_symbols_clean`, `oneshot.vocab_csv=data/raw/iso_symbols/vocab.csv`):
- `pipeline.build_classes` (cũng được chính stage xuất gọi) → `classes.yaml` 13 lớp thô / 45 lớp mịn, `fgc_groups` = agitator, blind_disc, check_valve, heat_exchanger, valve.
- `pipeline.build_manifest` → 826 dòng, tất cả `test`, `source = synthetic`.
- Chạy lại hai stage trên bằng lệnh ngoài: `classes.yaml`, `manifest.csv`, `annotations.coco.json` **giống từng byte** (sha256).
- `pipeline.coco_to_yolo` (nhãn thô) → 826 file nhãn + `data.yaml` (13 lớp) trong `data/iso_open100/derived/yolo_coarse/`.

**Chỉ nạp / dựng đối tượng** (không chạy): `DataPaths.find_image` cho cả 826 id, `load_classmap`, `evaluation._stem`, dựng `DetectInferPipeline` và `FGCInferPipeline` với config gốc + 2 override (thấy `classmap`, `manifest.csv`). **Không chạy**: `detect_infer`, `fgc_infer`, `evaluation`, `iso_prototypes`, `detect_train`, `fgc_train`, `tiling`, `extract_crops`.

**Không đụng Stage 1/2** (`checksum_verdict.txt`): `data/canonical` 504 file sha256 trước = sau; `data/raw/iso_symbols`, `iso_symbols_clean`, `pid2graph/open100` + `configs/config.yaml` 365 mục trước = sau; thư mục `runs/` gốc không có thêm gì.

## 9. Ảnh (cục bộ; `*.png` trong `docs/experiments/` bị gitignore)

- `docs/experiments/iso_render/v8/overlay_coco.png` — hộp COCO trên một ảnh sạch và bản suy giảm của lò phản ứng tờ 9 (`iso09_v012`, `_deg`). Đã xem: hộp đỏ agitator ôm trục và cánh, hộp xanh khít bình và hai bong bóng; bản suy giảm (xoay ~1°) hộp theo cong vênh đúng.
- `docs/experiments/iso_render/v8/overlay_graphml.png` — nút GraphML (đỏ = đã vẽ, xám = connector/crossing) và cạnh (xanh, nối tâm–tâm) trên ảnh suy giảm. Đã xem: hộp nút bình và cặp tag trùng với ảnh, cạnh nối tâm các nút (đường chéo dài là nối tâm, không phải đường ống), không có nút/hộp cho agitator.
- `docs/experiments/iso_render/v8/class_hist.png` — số mẫu theo lớp (log). Đã xem.
- CSV/TXT: `per_class.csv`, `per_sheet.csv`, `summary.txt`, `pipeline_stages.txt`, `pipeline_loaded.txt`, `checksum_verdict.txt`, `canonical_checksums_{before,after}.txt`, `raw_checksums_{before,after}.txt`.

## 10. Hạn chế và câu hỏi cho user

- Một detector Stage 1 train trên DigitizePID có lớp thô khác (`Symbol_N` → `valve`, `instrument`...): `classes.yaml` của gốc này có **13 lớp thô khác**, nên `detect_infer`/`evaluation` trên gốc này cần detector train trên chính nó (hoặc một lớp ánh xạ — chưa làm). Đây là điểm cần quyết cho đánh giá Stage 1.
- Cờ inlet/outlet và mũi tên dẫn bị bỏ khỏi COCO (mục 3) — đồng ý?
- Mặc định `images: both` + `all_test`: đánh giá đầy đủ chạy trên 826 ảnh ~30 MP. Muốn tập suy giảm thôi thì `images=degraded` (413 ảnh).
- Hardlink giữ bản cũ nếu V6 sinh lại mà không xuất lại; có thể đổi `link=symlink`.
- `by_source` chỉ có `train`/`test` (không `val`); thêm `val` nếu train Stage 1.
- GraphML 128 MB; nếu thừa có thể chỉ ghi bản sạch.
