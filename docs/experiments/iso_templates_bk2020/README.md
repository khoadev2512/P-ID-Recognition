# Template ISO: tạo lại từ vector BK-2020-04 (2026-10-18)

## Nguồn

- `data/raw/iso_symbols_clean/` lúc đầu được cắt từ `ISO_10628-2_2012_Symbols_Sheet_*.pdf`.
  Metadata của các PDF này ghi: tác giả **Steven Baltakatei Sandoval**, bộ bản vẽ
  `BK-2020-04-PID-1`, license **CC BY-SA 4.0**. PID2Graph (arXiv 2411.13929, ref [30])
  dùng cùng nguồn; bài đó ghi license là CC BY, nhưng file ghi CC BY-SA. Luận văn phải
  ghi nguồn. Nếu công bố lại bộ template thì phải giữ cùng license.
- Repo gốc: https://gitlab.com/baltakatei/BK-2020-04, file `DWG/BK-2020-04-PID-1-SHT{1..7}.svg`.
  Trong đó mỗi symbol là một group có nhãn `REG#:<id>_symbol`, có đủ 39/39 symbol của
  `vocab.csv`. Vì cùng tác giả nên bộ này **không** cho thêm template thứ hai với phong
  cách khác.

## Đã làm: tạo lại cả bộ bằng `scripts/extract_iso_symbols_svg.py`

Các bước:
1. Tách từng symbol bằng vector: xoá mọi phần tử SVG không thuộc group của symbol đó,
   rồi render 600 dpi.
2. Bỏ đoạn ống nối. Một đoạn được coi là ống nối khi nó là một đoạn thẳng dài 5 đơn vị
   **và** nằm ngoài khung bao của thân symbol.
   - Thanh dọc giữa thân van (X8074, X8075, X8078) và phần bên trong 2033 cũng dài 5
     nhưng nằm trong thân, nên được giữ.
   - Symbol chỉ gồm toàn đoạn dài 5 (511, mặt bích) giữ nguyên.
3. Chuyển nét đứt thành các đoạn ngắn, vì MuPDF bỏ qua `stroke-dasharray`. Chỉ X8100
   (cột) có nét đứt: xem `x8100_dashes.png`.
4. Riêng 241 (mũi tên chiều dòng) chép nguyên bản cũ, vì đầu mũi tên vẽ bằng SVG marker
   mà MuPDF không render.

So sánh `compare_final.png`, mỗi symbol bên trái là bản cũ, bên phải là bản mới:

| | Bản cũ (cắt từ PDF) | Bản mới (vector) |
|---|---|---|
| X8071 ball valve | Lỗi: symbol tí hon + vạch lạc | Đúng |
| Nét | Răng cưa ở một số symbol (2101, X8075) | Sạch |
| Đoạn ống nối | Không đồng nhất: 301, 2063, 2511, C0082… có ống; valve thì không | Bỏ hết, trừ 511 |
| Độ dày nét / cạnh symbol (valve) | 0.025–0.037, X8071 là 0.013 | **0.024, đồng nhất** |

Độ dày nét không còn dao động do lỗi cắt ảnh. Template ISO vẫn mảnh hơn crop DigitizePID
(0.041–0.051), tức là cùng chiều với giả định của `SubpartWarp` (B1). Con số mini-P1 ở B1
được đo trên bộ template cũ.

`with_stubs/` và `compare_with_stubs.png` là bản render còn giữ đoạn ống nối. Đây là
ablation tiềm năng: crop DigitizePID có ống dính liền thân symbol. Tạo bản này bằng
`--keep-stubs`.

## Chạy lại

```bash
git clone --depth 1 https://gitlab.com/baltakatei/BK-2020-04.git /tmp/bk2020
uv run python scripts/extract_iso_symbols_svg.py --svg-dir /tmp/bk2020/DWG
#   --keep-stubs                 giữ đoạn ống nối
#   --out <dir>                  ghi ra thư mục khác (mặc định data/raw/iso_symbols_clean)
```
