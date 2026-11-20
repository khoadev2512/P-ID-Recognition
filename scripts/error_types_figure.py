"""Explanatory figure: the two kinds of error behind the ISO-template -> drawing gap (PLAN §2F).

Type 1 "different style, same shape" (ISO classes S1, S21, S24) vs type 2 "different shape" (VARIANT
classes S6, S8, S14). Each row: the mapped ISO template | 3 DigitizePID crops from TRAIN sheets
(the probe's "center" crops, as variant_contact_sheet.py; P1-val unused) | the template DINOv2-S /
A4 / CL2N (41 templates, §2G) names most often for the 30 crops, with top-1 on them.
Descriptive only.

    python scripts/error_types_figure.py --out runs/phase_c/variants/two_error_types.png
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import oneshot_probe as probe  # noqa: E402
from variant_contact_sheet import _font  # noqa: E402

from pipeline.iso_prototypes import load_templates, read_vocab  # noqa: E402

DROPPED = "pipeline"
PANELS = [
    (
        "LOẠI 1 — KHÁC PHONG CÁCH, CÙNG HÌNH (lớp ISO)",
        "Bản vẽ vẽ ĐÚNG hình ISO, chỉ khác cách vẽ (nét, kích thước, ống nối, nhiễu scan).",
        ["Symbol_1", "Symbol_21", "Symbol_24"],
    ),
    (
        "LOẠI 2 — KHÁC DẠNG HÌNH (lớp VARIANT)",
        "Cùng NGHĨA nhưng bản vẽ dùng hình khác hẳn ảnh mẫu ISO (chuyên gia xác nhận).",
        ["Symbol_6", "Symbol_8", "Symbol_14"],
    ),
]
N_CROPS = 3
CELL, LABEL_W, PAD = 170, 300, 10
OK, BAD, GREY = (30, 130, 60), (200, 50, 50), (238, 238, 238)


def cell(img: Image.Image, frame=None) -> Image.Image:
    im = img.convert("RGB")
    im.thumbnail((CELL - 16, CELL - 34))
    out = Image.new("RGB", (CELL, CELL), GREY)
    out.paste(im, ((CELL - im.width) // 2, 6 + (CELL - 34 - im.height) // 2))
    if frame:
        ImageDraw.Draw(out).rectangle([0, 0, CELL - 1, CELL - 1], outline=frame, width=4)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--probe", type=Path, default=ROOT / "runs/phase_c/probe")
    args = ap.parse_args(argv)
    cfg = OmegaConf.load(ROOT / "configs/config.yaml")
    cfg.data_root = str(ROOT / "data")
    world = probe.load_world(cfg)
    crops = probe.pick_crops(world)
    groups = world["groups"]
    cache = torch.load(args.probe / "emb_dinov2_a4.pt", weights_only=False)
    keep = [i for i, n in enumerate(cache["names"]) if n != DROPPED]
    names = [cache["names"][i] for i in keep]
    emb = cache["emb"]
    centers = [c for c in crops if c["set"] == "p1" and c["role"] == "center"]
    center = torch.cat([emb[c["ann"]["id"]] for c in centers]).mean(0)
    p = F.normalize(cache["protos"][keep] - center, dim=-1)
    images, _ = load_templates(
        ROOT / "data/raw/iso_symbols_clean", read_vocab(ROOT / "data/raw/iso_symbols/vocab.csv")
    )

    rows = {}
    for _, _, classes in PANELS:
        for cls in classes:
            cs = [c for c in centers if c["class"] == cls]
            q = torch.stack(
                [
                    F.normalize(F.normalize(emb[c["ann"]["id"]] - center, dim=-1).mean(0), dim=0)
                    for c in cs
                ]
            )
            top1 = [names[i] for i in (q @ p.T).argmax(1).tolist()]
            want = groups[cls].template
            guess, k = Counter(top1).most_common(1)[0]
            rows[cls] = (cs[:N_CROPS], want, guess, k / len(cs), top1.count(want) / len(cs))
    pixels = probe.cut_crops([c for v in rows.values() for c in v[0]])

    font_h, font, small = _font(20), _font(15), _font(13)
    cols = 1 + N_CROPS + 1
    width = LABEL_W + cols * CELL + 2 * PAD
    heights = [70 + len(cl) * (CELL + PAD) for _, _, cl in PANELS]
    canvas = Image.new("RGB", (width, 40 + sum(heights) + 150), "white")
    d = ImageDraw.Draw(canvas)
    head = ["Ảnh mẫu ISO"] + [f"Crop bản vẽ {i + 1}" for i in range(N_CROPS)] + ["Model đoán"]
    y = 8
    for c, h in enumerate(head):
        d.text((PAD + LABEL_W + c * CELL + 8, y), h, fill="black", font=font)
    y = 40
    for title, sub, classes in PANELS:
        color = OK if title.startswith("LOẠI 1") else BAD
        d.rectangle([PAD, y, width - PAD, y + 56], fill=(245, 245, 245), outline=color, width=2)
        d.text((PAD + 10, y + 6), title, fill=color, font=font_h)
        d.text((PAD + 10, y + 32), sub, fill="black", font=font)
        y += 66
        for cls in classes:
            cs, want, guess, share, acc = rows[cls]
            d.text((PAD + 6, y + 10), cls.replace("Symbol_", "S"), fill="black", font=font_h)
            d.text((PAD + 6, y + 40), f"nghĩa: {want}", fill="black", font=font)
            d.text((PAD + 6, y + 66), f"đúng {acc:.0%} trên {30} crop train", fill=color, font=font)
            x = PAD + LABEL_W
            canvas.paste(cell(images[want], OK), (x, y))
            for j, c in enumerate(cs):
                canvas.paste(cell(Image.fromarray(pixels[c["ann"]["id"]])), (x + (1 + j) * CELL, y))
            gx = x + (1 + N_CROPS) * CELL
            canvas.paste(cell(images[guess], OK if guess == want else BAD), (gx, y))
            d.text((gx + 6, y + CELL - 26), f"{guess[:18]} {share:.0%}", fill="black", font=small)
            y += CELL + PAD
        y += 4
    notes = [
        "Loại 1: hình giống nhau → A4 (cắt theo mực, nhị phân) + DINOv2 đã xử lý phần lớn"
        " (P1-val ISO 0.80; S1 / S21 / S24 ≈ 1.0).",
        "Loại 2: hình khác hẳn → không phép đổi pixel / vector nào nối được"
        " (H2, F0, A, §2I, §2J: S6 / S8 vẫn 0). Chỉ ảnh mẫu theo quy ước",
        "của chính bản vẽ (legend / 1 crop mỗi ký hiệu) mới đúng:"
        " D3 → S6 0 → 0.89, S8 0 → 1.00, S14 0.01 → 1.00 (PLAN §2F.2).",
        "Khung xanh = ảnh mẫu đúng; khung đỏ = model đoán sai."
        " Số trên crop train (tờ train, không phải tập quyết định).",
    ]
    for i, t in enumerate(notes):
        d.text((PAD + 6, y + 8 + i * 26), t, fill="black", font=font)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(args.out, canvas.size)
    for cls, (_, want, guess, share, acc) in rows.items():
        print(f"{cls:10s} {want:28s} top1 {acc:.2f}  guess {guess} {share:.0%}")


if __name__ == "__main__":
    main()
