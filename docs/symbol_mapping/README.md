# Symbol Mapping — nhận diện & đặt tên fine cho 32 DigitizePID symbol

Nơi tập trung việc **nhận diện từng symbol** (Symbol_1..32) → **tên ISO/ISA thật**, để Stage 2
(FGC) có label có nghĩa thay vì `Symbol_N` vô danh.

## Files

- **`symbol_map.csv`** — bảng theo dõi chính. Mỗi dòng 1 symbol. Cột:
  | cột | nghĩa |
  |---|---|
  | `symbol_n` | Symbol_N theo paper/Fig.3 (1-indexed) |
  | `class_id` | id trong YOLO label (0-indexed = symbol_n − 1) |
  | `coarse` | nhóm coarse (8 nhóm đã chốt) |
  | `fgc_group` | yes = thuộc nhóm cần FGC (valve/blind_disc/heat_exchanger/instrument) |
  | `status` | `DONE` = đã nhận diện, `TODO` = chưa |
  | `fine_name` | **tên fine sẽ dùng làm label** (điền vào đây) |
  | `reg_number` | REG# ISO 10628-2 (nếu có) |
  | `desc_iso` | DESC chính thức theo ISO/ISA |
  | `notes` | mô tả hình / gợi ý |

- **`thumbnails/symbol_NN.png`** — ảnh crop thật của từng symbol (10 mẫu/strip) để đối chiếu.
  Sinh từ `data/raw/digitize_pid_yolo`. Tái tạo bằng script ở cuối README này.

## Trạng thái hiện tại

| fgc_group | symbols | status |
|---|---|---|
| **valve** | 1–16 | **TODO** (16 loại bow-tie, cần đối chiếu ISO Group 21) |
| **blind_disc** | 17,18,19 | DONE (open/blind/interchangeable disc) |
| **heat_exchanger** | 22,23 | DONE (insulation / HX coil) |
| **instrument** | 26–32 | TODO (bubble có text — map ISA-5.1: GLR/RO/SDL/DDL/STA/ZLO/LG) |
| reducer/flange/flow/safety | 20,21,24,25 | DONE (coarse=fine, không FGC) |

→ Còn **valve (16)** và **instrument (7)** cần điền `fine_name`.

## Cách dùng

1. Mở `thumbnails/symbol_NN.png` của symbol cần nhận diện.
2. Đối chiếu với sheet ISO 10628-2 (Group 21 Valve / Group 22 Check valve) hoặc ISA-5.1
   (instrument), xác định loại.
3. Điền `fine_name` (snake_case, vd `gate_valve`), `reg_number`, `desc_iso` vào `symbol_map.csv`,
   đổi `status` = `DONE`.
4. Khi đủ (hoặc đủ phần muốn), chạy converter để áp tên vào `classes.yaml`:
   → sửa `scripts/convert_yolo_to_coco.py` dùng `fine_name` từ CSV (xem TODO dưới),
   hoặc báo mình áp giúp.

**Lưu ý trung thực**: DigitizePID là synthetic — vài "loại valve" có thể KHÔNG ứng 1-1 với
valve ISO có tên. Symbol nào không chắc → giữ tên mô tả (`valve_variant_N`) hoặc để TODO,
đừng ép tên ISO sai.

## Bước tiếp (chưa làm)

- [ ] Điền `fine_name` cho valve 1–16 + instrument 26–32.
- [ ] Cho `convert_yolo_to_coco.py` đọc `fine_name` từ CSV này (thay vì hardcode `Symbol_N`),
      để `name` trong COCO = tên thật → `classes.yaml` + crops_fgc folder mang tên có nghĩa.
- [ ] Re-prep crops với tên mới.

## Tái tạo thumbnails

```bash
uv run python - <<'PY'
from pathlib import Path
from PIL import Image
root = Path("data/raw/digitize_pid_yolo"); out = Path("docs/symbol_mapping/thumbnails")
MARGIN=0.25; N=10; CELL=140
crops={c:[] for c in range(32)}
for lab in sorted((root/"labels"/"train").glob("*.txt"), key=lambda p:int(p.stem)):
    need=[c for c in range(32) if len(crops[c])<N]
    if not need: break
    rows=[l.split() for l in lab.read_text().splitlines() if l.strip()]
    if not any(int(r[0]) in need for r in rows): continue
    im=Image.open(root/"images"/"train"/f"{lab.stem}.jpg"); W,H=im.size
    for r in rows:
        c=int(r[0])
        if c in need and len(crops[c])<N:
            cx,cy,w,h=map(float,r[1:5]); bw,bh=w*W,h*H
            x1=max(0,int(cx*W-bw*(0.5+MARGIN))); y1=max(0,int(cy*H-bh*(0.5+MARGIN)))
            x2=min(W,int(cx*W+bw*(0.5+MARGIN))); y2=min(H,int(cy*H+bh*(0.5+MARGIN)))
            crops[c].append(im.crop((x1,y1,x2,y2)).resize((CELL,CELL)))
for c in range(32):
    if not crops[c]: continue
    strip=Image.new("RGB",(CELL*len(crops[c]),CELL),"white")
    for i,im in enumerate(crops[c]): strip.paste(im,(i*CELL,0))
    strip.save(out/f"symbol_{c+1:02d}.png")
PY
```
