# P&ID Synthetic Renderer — Rules & Plan

Renderer sinh ảnh P&ID synthetic + annotation COCO từ library symbol ISO, output thẳng
`data/canonical/` để pipeline (detect→FGC→eval) chạy nguyên. Thay DigitizePID (không
chuẩn) bằng data ISO chuẩn 100%. Xem [[iso-render-pivot]] trong memory.

**Input**: `data/raw/iso_symbols_clean/*.png` (39 symbol) + `vocab.csv` (coarse/fine/fgc).
**Output**: `data/canonical/{images/, annotations.coco.json, digitizepid_splits.json}`.

## Bộ quy tắc render (7 nhóm)

### 0. Background — cắt từ DigitizePID (vùng trống thật)
- **KHÔNG nền trắng trơn.** Cắt pool patch từ 400 ảnh DigitizePID tại vùng KHÔNG có symbol
  (dùng YOLO labels để tránh — symbol chỉ chiếm ~2% diện tích, 98% trống).
- **Lọc theo ink% = 1-10%**: patch có line/text/border + nền scan thật (giống P&ID) nhưng
  không rối. Bỏ patch quá trắng (<1%, như canvas trơn) và quá đậm (>10%, nền rối).
- `scripts/extract_bg_patches.py` sinh pool `data/raw/bg_patches/*.png` (1 lần).
- Renderer: mỗi ảnh lấy 1 patch ngẫu nhiên làm nền. Đặt symbol ưu tiên vùng ít-ink của patch
  (tránh đè text đậm → symbol vẫn rõ, nhãn đúng).

### 1. Layout — đặt symbol
- **Random placement** (không grid cứng), chống chồng lấp: kiểm bbox trước khi đặt, bỏ nếu đè.
- N symbol/ảnh ngẫu nhiên (vd 15-40). Canvas = kích thước background patch.
- Margin biên: không đặt symbol sát mép. Tránh đè vùng text đậm của background.

### 2. Chọn symbol — cân bằng
- Random từ 39 symbol, **cân bằng theo coarse** (không để valve 8-fine lấn át).
- **Tăng tần suất family FGC** (valve/disc/HX/check/agitator) → đủ mẫu mỗi fine class cho Stage 2.

### 3. Biến đổi symbol (augment hình học)
- **Scale** ngẫu nhiên (0.7-1.5x) — symbol P&ID thật nhiều kích cỡ.
- **KHÔNG xoay/lật** — symbol P&ID có hướng ý nghĩa (van xoay = van khác). degrees=0, flip=0.
- Giữ connection stub (để nối pipe).

### 4. Pipe/line — nối symbol
- **v1: L-shaped pipe** (ngang rồi dọc) nối 2 symbol qua connection point. Đơn giản, giống
  thật hơn line thẳng. KHÔNG cần routing tránh né hoàn hảo (bẫy thời gian — xem cảnh báo).
- **v2 (sau, nếu kịp)**: pipe tránh symbol + nhánh rẽ.
- **KHÔNG annotate pipe** — pipeline chỉ detect symbol. Pipe chỉ là context thị giác.

### 5. Text tag
- Thêm tag cạnh symbol (vd "V-101", "P-205") — P&ID thật luôn có. Buộc detector học bỏ qua text.
- **KHÔNG annotate text**.

### 6. Degradation — ĐẦY ĐỦ (quan trọng cho generalize sang P&ID thật)
Symbol ISO sạch tinh → phải degrade cho giống scan thật (thu hẹp domain gap):
- Gaussian noise + salt-pepper
- Blur nhẹ (scan mờ)
- Pixelate / downscale-upscale (độ phân giải thấp)
- Nét không đều (độ đậm thay đổi)

### 7. Annotation — tự sinh
- Mỗi symbol đặt → bbox + category(fine_name) tự sinh.
- Xuất COCO: categories từ vocab (name=fine, supercategory=coarse). Prefix `synthetic/`.
- Split train/val (ghi `digitizepid_splits.json` để build_manifest giữ nguyên).

## Plan triển khai (từng bước verify)

1. `scripts/render_pid.py`: random placement + bbox + xuất COCO (chưa pipe/degrade).
   → render 1 ảnh, nhìn bbox đúng.
2. Thêm L-shaped pipe nối symbol. → nhìn ảnh có pipe.
3. Thêm text tag. → nhìn tag.
4. Thêm degradation (noise/blur/pixelate). → nhìn giống scan.
5. Xuất `data/canonical/` + split. → chạy `pid-prep-classes` ra đúng vocab ISO.
6. Render ~300-500 ảnh → prep→train. → pipeline chạy thông trên data ISO.

## Nguyên tắc phạm vi (nhắc lại)

Renderer ĐỦ DÙNG, đừng cầu toàn. Pipe/degrade làm tốt vừa phải. Đóng góp chính của
thesis là detection+FGC+ArcFace, KHÔNG phải data-gen. Pipe routing hoàn hảo là bẫy thời
gian (SynthPID viết cả paper CVPR cho nó) — v1 L-shape là đủ.

## Config render (đề xuất, tune sau)
- canvas: 2560x1920 | symbols/img: 15-40 | scale: 0.7-1.5
- noise σ: 5-15 | blur: 0-1.5 | pixelate: 0.5-1.0x
- n_images: 400 | split train/val: 0.8/0.2
