# Crop tay các symbol ISO lỗi nặng

Auto-extract (`extract_iso_symbols.py`) cắt sạch ~29/39 symbol. Số còn lại — chủ yếu
valve nhỏ + symbol dính số hàng/kẻ ô — cần **crop tay** cho sạch.

## Ảnh để crop

**`docs/symbol_mapping/crop_guide/<REG>_p<page>.png`** — ảnh phóng to 400 DPI, mỗi file
một vùng chứa symbol + REG# (để định vị). Crop phần SYMBOL (bỏ text REG#/DESC, bỏ số
hàng, bỏ kẻ ô). Hoặc dùng ảnh full `iso_pages/page_N.png`.

## Lưu kết quả

Crop xong → lưu đè vào `data/raw/iso_symbols_clean/<REG>.png`
(vd valve_gate → `data/raw/iso_symbols_clean/X8074.png`). Nền trắng, ôm sát symbol
(giữ connection stub ngắn 2 bên — đó là phần của ký hiệu ISO, ĐỪNG cắt mất).

## MỨC 1 — Nặng nhất, PHẢI crop (7 symbol, đều page 6)

| REG# | fine_name | guide |
|---|---|---|
| X8074 | valve_gate | `crop_guide/X8074_p6.png` |
| X8068 | valve_globe | `crop_guide/X8068_p6.png` |
| 2101 | valve_general | `crop_guide/2101_p6.png` |
| 2102 | valve_angle | `crop_guide/2102_p6.png` |
| 2103 | valve_three_way | `crop_guide/2103_p6.png` |
| X8075 | valve_butterfly | `crop_guide/X8075_p6.png` |
| 405 | pipeline | `crop_guide/405_p6.png` |

## MỨC 2 — Vừa, nên crop (3 symbol, dính số hàng)

| REG# | fine_name | guide |
|---|---|---|
| 2301 | pump_general | `crop_guide/2301_p4.png` |
| 2672 | agitator_general | `crop_guide/2672_p7.png` |
| X8079 | hx_general | `crop_guide/X8079_p2.png` |

## KHÔNG cần crop (đã nhận đúng loại, nét mép mảnh → render che được)

2063, 2302, 2322, 2511, 2514, 8091, C0082, X2124, X8071, X8095, X8179

## Sau khi crop xong

Báo mình → mình verify lại lưới toàn bộ + chuyển sang viết renderer.
