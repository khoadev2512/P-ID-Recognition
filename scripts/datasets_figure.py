"""Explanatory figure: the data sources, what each encoder is trained on, and the four Stage-2
evaluation sets (P1-val, P1, P2, P4) with real example crops. Descriptive only, no scores.

    python scripts/datasets_figure.py --out runs/phase_c/variants/datasets_overview.png
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import oneshot_probe as probe  # noqa: E402
from variant_contact_sheet import _font  # noqa: E402

from pipeline.iso_prototypes import load_templates, read_vocab  # noqa: E402

W = 1900
CROP = 104
GREEN, RED, BLUE, GREY = (30, 130, 60), (200, 50, 50), (40, 90, 170), (120, 120, 120)
LIGHT = (246, 246, 246)
F_T, F_H, F, F_S = _font(26), _font(19), _font(15), _font(13)

# (class, caption) per card; captions are the ISO meaning or the role, from symbol_map / groups
P1VAL = (
    [
        ("Symbol_1", "S1 → valve_gate"),
        ("Symbol_24", "S24 → flow_dir."),
        ("Symbol_6", "S6 (VARIANT)"),
    ],
    [("Symbol_7", "S7"), ("Symbol_26", "S26")],
)
P1 = (
    [("Symbol_3", "S3 → valve_globe"), ("Symbol_18", "S18"), ("Symbol_4", "S4")],
    [("Symbol_9", "S9"), ("Symbol_15", "S15")],
)
P2 = (
    [("Symbol_21", "S21 (nhóm S)"), ("Symbol_10", "S10 (nhóm S)"), ("Symbol_8", "S8 (nhóm V)")],
    [("Symbol_16", "S16 (OOV_seen)"), ("Symbol_29", "S29 (OOV_seen)")],
)
P4 = (
    [
        ("valve_gate", "valve_gate (S)"),
        ("valve_butterfly", "butterfly (V)"),
        ("pump_centrifugal", "bơm (H tự nhiên)"),
    ],
    [("other_isa", "other_isa"), ("other_iso", "other_iso")],
)


def box(d, xy, outline, fill=LIGHT, width=2):
    d.rounded_rectangle(xy, radius=10, fill=fill, outline=outline, width=width)


def thumb(arr_or_img, size=CROP, frame=None) -> Image.Image:
    im = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(arr_or_img)
    im = im.convert("RGB")
    im.thumbnail((size - 8, size - 8))
    out = Image.new("RGB", (size, size), (232, 232, 232))
    out.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    if frame:
        ImageDraw.Draw(out).rectangle([0, 0, size - 1, size - 1], outline=frame, width=3)
    return out


def sheet(path: Path, w: int, h: int) -> Image.Image:
    im = Image.open(path).convert("RGB")
    im.thumbnail((w, h))
    return im


def text(d, xy, lines, font=F, fill="black", step=20):
    x, y = xy
    for t in lines:
        d.text((x, y), t, font=font, fill=fill)
        y += step
    return y


def digitize_crops(world, pick, want: list[str], where) -> dict[str, np.ndarray]:
    """One crop per class from DigitizePID sheets accepted by `where(stem)` (seeded)."""
    coco = world["coco"]
    cat = {c["id"]: c["name"] for c in coco["categories"]}
    stem = {i["id"]: Path(i["file_name"]).stem for i in coco["images"]}
    out = []
    for name in want:
        anns = sorted(
            (
                a
                for a in coco["annotations"]
                if not a.get("iscrowd")
                and cat[a["category_id"]] == name
                and where(stem[a["image_id"]])
            ),
            key=lambda a: a["id"],
        )
        a = random.Random(f"fig/{name}/{pick}").choice(anns)
        out.append({"class": name, "ann": a, "stem": stem[a["image_id"]]})
    px = probe.cut_crops(out)
    return {c["class"]: px[c["ann"]["id"]] for c in out}


def open100_crops(want: list[str]) -> tuple[dict[str, np.ndarray], Path]:
    root = ROOT / "data/iso_open100/canonical"
    coco = json.loads((root / "annotations.coco.json").read_text())
    img = next(i for i in coco["images"] if i["sheet"] == 3 and i["degraded"])
    cat = {c["name"]: c["id"] for c in coco["categories"]}
    path = root / "images" / Path(img["file_name"]).name
    pixels = np.array(Image.open(path).convert("RGB"))
    out = {}
    for name in want:
        anns = [
            a
            for a in coco["annotations"]
            if a["image_id"] == img["id"] and a["category_id"] == cat[name]
        ]
        if not anns:  # fall back to any degraded sheet
            ids = {i["id"] for i in coco["images"] if i["degraded"]}
            anns = [
                a
                for a in coco["annotations"]
                if a["image_id"] in ids and a["category_id"] == cat[name]
            ]
            a = random.Random(f"fig/{name}").choice(sorted(anns, key=lambda a: a["id"]))
            im = next(i for i in coco["images"] if i["id"] == a["image_id"])
            px = np.array(Image.open(root / "images" / Path(im["file_name"]).name).convert("RGB"))
        else:
            a, px = random.Random(f"fig/{name}").choice(sorted(anns, key=lambda a: a["id"])), pixels
        x, y, w, h = a["bbox"]
        out[name] = np.ascontiguousarray(probe.cut(px, (x, y, x + w, y + h)))
    return out, path


def card(canvas, d, x, y, w, h, spec):
    title, color, src, known, unknown, n_line, question, seen, role = spec
    box(d, (x, y, x + w, y + h), color, width=3)
    d.text((x + 14, y + 10), title, font=F_T, fill=color)
    yy = text(d, (x + 14, y + 46), src, font=F, step=19)
    yy += 6
    d.text((x + 14, yy), "BIẾT (có ảnh mẫu ISO → phải gọi đúng tên)", font=F_S, fill=GREEN)
    yy += 18
    for j, (im, cap) in enumerate(known):
        cx = x + 14 + j * (CROP + 10)
        canvas.paste(thumb(im, frame=GREEN), (cx, yy))
        d.text((cx, yy + CROP + 2), cap, font=F_S, fill="black")
    yy += CROP + 24
    d.text((x + 14, yy), "LẠ (không có ảnh mẫu → phải ra “Other”)", font=F_S, fill=RED)
    yy += 18
    for j, (im, cap) in enumerate(unknown):
        cx = x + 14 + j * (CROP + 10)
        canvas.paste(thumb(im, frame=RED), (cx, yy))
        d.text((cx, yy + CROP + 2), cap, font=F_S, fill="black")
    yy += CROP + 26
    yy = text(d, (x + 14, yy), n_line, font=F, step=19) + 6
    d.text((x + 14, yy), "Câu hỏi:", font=F_H, fill=color)
    yy = text(d, (x + 14, yy + 26), question, font=F, step=19) + 8
    d.text((x + 14, yy), "Lớp BIẾT model đã từng train chưa?", font=F_H, fill="black")
    yy += 26
    for name, ok in seen:
        d.text((x + 24, yy), f"{name}:", font=F, fill="black")
        d.text((x + 120, yy), ok, font=F, fill=GREEN if ok.startswith("CHƯA") else GREY)
        yy += 20
    d.rounded_rectangle((x + 10, y + h - 44, x + w - 10, y + h - 10), radius=8, fill=color)
    d.text((x + 20, y + h - 37), role, font=F_H, fill="white")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    cfg = OmegaConf.load(ROOT / "configs/config.yaml")
    cfg.data_root = str(ROOT / "data")
    world = probe.load_world(cfg)
    split, val_sel = world["split"], world["val_sel"]
    images, _ = load_templates(
        ROOT / "data/raw/iso_symbols_clean", read_vocab(ROOT / "data/raw/iso_symbols/vocab.csv")
    )

    def classes(spec):
        return [c for c, _ in spec[0] + spec[1]]

    def is_val_sel(s):
        return s in val_sel

    def is_test(s):
        return split.get(s) == "val" and s not in val_sel

    def anywhere(s):
        return True

    crops = {
        "p1val": digitize_crops(world, "p1val", classes(P1VAL), is_val_sel),
        "p1": digitize_crops(world, "p1", classes(P1), anywhere),
        "p2": digitize_crops(world, "p2", classes(P2), is_test),
    }
    crops["p4"], p4_sheet = open100_crops(classes(P4))
    dpid_sheet = (
        ROOT / "data/canonical/images" / f"{sorted(s for s in split if split[s] == 'train')[3]}.jpg"
    )

    H = 1450
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    d.text(
        (20, 14),
        "Dữ liệu, model và 4 tập đánh giá của Stage 2 (nhận dạng theo ảnh mẫu ISO)",
        font=F_T,
        fill="black",
    )

    # ---- band 1: sources -------------------------------------------------------------------
    y0 = 60
    d.text((20, y0), "1. NGUỒN DỮ LIỆU", font=F_H, fill=BLUE)
    y0 += 30
    bw = (W - 80) // 3
    # DigitizePID
    x = 20
    box(d, (x, y0, x + bw, y0 + 330), BLUE)
    d.text((x + 12, y0 + 8), "DigitizePID — bản vẽ THẬT", font=F_H, fill=BLUE)
    text(
        d,
        (x + 12, y0 + 36),
        [
            "500 tờ, 32 lớp ký hiệu (Symbol_1 … Symbol_32),",
            "phong cách riêng của bộ này (không theo ISO).",
        ],
        step=19,
    )
    canvas.paste(sheet(dpid_sheet, 250, 220), (x + 12, y0 + 82))
    bx, by = x + 280, y0 + 90
    parts = [
        ("train 400 tờ", 400, (90, 140, 210)),
        ("val_sel 50", 50, (240, 170, 60)),
        ("test 50", 50, (200, 80, 80)),
    ]
    total = 500
    for i, (lab, n, col) in enumerate(parts):
        d.rectangle((bx, by + i * 54, bx + int(220 * n / total) + 20, by + i * 54 + 30), fill=col)
        d.text((bx + int(220 * n / total) + 30, by + i * 54 + 6), lab, font=F, fill="black")
    text(
        d,
        (x + 280, by + 166),
        ["Chia theo TỜ: không tờ nào", "nằm ở hai phần cùng lúc."],
        font=F_S,
        step=17,
    )
    # OPEN100-ISO
    x = 40 + bw
    box(d, (x, y0, x + bw, y0 + 330), BLUE)
    d.text((x + 12, y0 + 8), "OPEN100-ISO — bản vẽ TỔNG HỢP theo ISO", font=F_H, fill=BLUE)
    text(
        d,
        (x + 12, y0 + 36),
        [
            "12 tờ bố cục P&ID (OPEN100) × 72 biến thể (sạch / suy giảm);",
            "ký hiệu vẽ từ CÙNG nguồn vector với ảnh mẫu ISO.",
        ],
        step=19,
    )
    full = Image.open(p4_sheet).convert("RGB")
    fw, fh = full.size
    zoom = full.crop((int(fw * 0.30), int(fh * 0.30), int(fw * 0.62), int(fh * 0.62)))
    zoom.thumbnail((bw - 30, 228))
    canvas.paste(zoom, (x + 12, y0 + 86))
    d.text((x + 12, y0 + 316), "(phóng to một vùng của tờ)", font=F_S, fill=GREY)
    # templates
    x = 60 + 2 * bw
    box(d, (x, y0, x + bw, y0 + 330), BLUE)
    d.text((x + 12, y0 + 8), "Ảnh mẫu ISO 10628-2 — “từ điển”", font=F_H, fill=BLUE)
    text(
        d,
        (x + 12, y0 + 36),
        [
            "41 ảnh (bỏ `pipeline`), MỖI ký hiệu ĐÚNG 1 ẢNH.",
            "Chỉ dùng lúc nhận dạng; không dùng để train.",
        ],
        step=19,
    )
    names = [
        "valve_gate",
        "valve_globe",
        "valve_butterfly",
        "check_valve_swing",
        "pump_centrifugal",
        "flow_direction",
        "reducer",
        "tank",
        "flanged_connection",
        "silencer",
    ]
    for i, n in enumerate(names):
        cx, cy = x + 12 + (i % 5) * 112, y0 + 86 + (i // 5) * 122
        canvas.paste(thumb(images[n], 104), (cx, cy))
        d.text((cx, cy + 104), n[:16], font=F_S, fill="black")

    # ---- band 2: models --------------------------------------------------------------------
    y1 = y0 + 350
    d.text(
        (20, y1),
        "2. MODEL HỌC TỪ ĐÂU (encoder của Stage 2; Stage 1 YOLO train trên tờ train DigitizePID)",
        font=F_H,
        fill=BLUE,
    )
    y1 += 30
    models = [
        (
            "DINOv2-S — cấu hình 2B (ĐANG DÙNG)",
            GREEN,
            [
                "Học sẵn (tự giám sát) trên ~142 triệu ẢNH TỰ NHIÊN. Đóng băng.",
                "KHÔNG train trên bất kỳ bản vẽ P&ID nào.",
                "Lấy từ DigitizePID đúng 2 thứ, không cần nhãn tên:",
                " • tâm CL2N = TB vector của 570 crop tờ train;  • ngưỡng τ chọn trên val_sel.",
            ],
        ),
        (
            "R1b — ResNet-34 + ArcFace (cũ)",
            GREY,
            [
                "Train có nhãn trên tờ TRAIN DigitizePID, CẢ 32 lớp.",
                "→ đã thấy mọi lớp của P1-val, P1, P2 (ở tờ khác).",
                "Dùng làm mốc so sánh.",
            ],
        ),
        (
            "R1a — ResNet-34 + ArcFace (cũ, ‘giữ lớp ra’)",
            GREY,
            [
                "Train như R1b nhưng chỉ 19 lớp: GIỮ RA 13 lớp",
                "(6 h_in + 4 h_out + 3 h_extra) → các lớp này",
                "R1a CHƯA TỪNG THẤY — dùng cho P1.",
            ],
        ),
    ]
    for i, (t, col, lines) in enumerate(models):
        x = 20 + i * (bw + 20)
        box(d, (x, y1, x + bw, y1 + 128), col, width=3)
        d.text((x + 12, y1 + 8), t, font=F_H, fill=col)
        text(d, (x + 12, y1 + 36), lines, step=21)
    y1 += 140
    box(d, (20, y1, W - 20, y1 + 50), BLUE, fill=(236, 243, 252))
    d.text(
        (34, y1 + 13),
        "Nhận dạng = so vector của crop với 41 vector ảnh mẫu (cosine) → chọn ảnh mẫu gần nhất;"
        " cos thấp hơn τ → “Other”.  KHÔNG có bước học nào.",
        font=F,
        fill="black",
    )

    # ---- band 3: the four sets ---------------------------------------------------------------
    y2 = y1 + 70
    d.text((20, y2), "3. BỐN TẬP ĐÁNH GIÁ", font=F_H, fill=BLUE)
    y2 += 30
    cw = (W - 100) // 4
    ch = 690

    def pick(key, spec):
        return [(crops[key][c], cap) for c, cap in spec[0]], [
            (crops[key][c], cap) for c, cap in spec[1]
        ]

    specs = [
        (
            "P1-val",
            (230, 140, 30),
            [
                "DigitizePID, 50 tờ val_sel",
                "10 lớp biết (4 vẽ đúng ISO + 6 VARIANT)",
                "+ 7 lớp lạ; 150 crop / lớp",
            ],
            *pick("p1val", P1VAL),
            ["2,550 crop"],
            ["Đổi cách làm X có tốt hơn không?", "(chỉ dùng để CHỌN, không báo là kết quả)"],
            [("DINOv2", "CHƯA"), ("R1a", "đã thấy"), ("R1b", "đã thấy")],
            "TẬP QUYẾT ĐỊNH (Phase C)",
        ),
        (
            "P1 — giữ lớp ra",
            (150, 60, 160),
            [
                "DigitizePID, mọi tờ",
                "6 lớp biết h_in (S3, S4, S5, S10, S18, S19)",
                "+ 4 lớp lạ h_out (S9, S15, S27, S30)",
            ],
            *pick("p1", P1),
            ["10,470 crop biết + 6,909 crop lạ"],
            ["Nhận được ký hiệu MÀ ENCODER CHƯA", "TỪNG THẤY chỉ nhờ 1 ảnh mẫu không?"],
            [("DINOv2", "CHƯA"), ("R1a", "CHƯA (đã giữ ra)"), ("R1b", "đã thấy")],
            "TEST: one-shot thật",
        ),
        (
            "P2 — nhóm của R1b",
            (60, 120, 120),
            [
                "DigitizePID, 50 tờ test",
                "S: 10 lớp vẽ đúng ISO; V: 6 VARIANT + 3 V_of_S",
                "OOV_seen: 11 lớp không ảnh mẫu",
            ],
            *pick("p2", P2),
            ["2,161 S + 1,560 V + 1,944 lạ = 5,665 crop"],
            ["Trên chính bộ bản vẽ đã train (R1b),", "ảnh mẫu ISO gọi tên và chặn lạ ra sao?"],
            [("DINOv2", "CHƯA"), ("R1a", "10/19 lớp đã thấy"), ("R1b", "đã thấy")],
            "TEST: trong miền DigitizePID",
        ),
        (
            "P4 — OPEN100-ISO",
            (40, 90, 170),
            [
                "Bộ bản vẽ KHÁC hẳn (12 tờ tổng hợp)",
                "S 3 / V 4 / H chọn 6 / H tự nhiên 17 lớp",
                "+ lạ: other_iso / other_isa / other_generic",
            ],
            *pick("p4", P4),
            ["46,900 crop × (sạch, suy giảm)"],
            [
                "Đem sang bản vẽ vẽ ĐÚNG chuẩn ISO",
                "(khác bộ train) thì gọi tên có đúng không?",
                "H chọn = nghĩa ISO của 6 lớp h_in;",
                "H tự nhiên = ký hiệu ISO không có lớp DigitizePID.",
            ],
            [("DINOv2", "CHƯA"), ("R1a", "CHƯA thấy bộ vẽ này"), ("R1b", "CHƯA thấy bộ vẽ này")],
            "TEST: chuyển sang ISO (40%)",
        ),
    ]
    for i, spec in enumerate(specs):
        card(canvas, d, 20 + i * (cw + 20), y2, cw, ch, spec)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(args.out, canvas.size)


if __name__ == "__main__":
    main()
