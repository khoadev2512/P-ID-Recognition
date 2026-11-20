"""Montage of raw crops (10% margin, like pid-prep-crops) for a set of classes."""
import json, random, sys
from pathlib import Path
from PIL import Image, ImageDraw

root = Path(sys.argv[1]); out = Path(sys.argv[2]); classes = sys.argv[3].split(",")
n = int(sys.argv[4]) if len(sys.argv) > 4 else 6
coco = json.load(open(root / "annotations.coco.json"))
cat = {c["id"]: c["name"] for c in coco["categories"]}
imgs = {i["id"]: i for i in coco["images"]}
rng = random.Random(0)
by = {}
for a in coco["annotations"]:
    by.setdefault(cat[a["category_id"]], []).append(a)
cell = 140
sheet = Image.new("RGB", (cell * n + 90, cell * len(classes)), "white")
d = ImageDraw.Draw(sheet)
cache = {}
for r, cname in enumerate(classes):
    d.text((4, r * cell + cell // 2), cname, fill="red")
    for c, a in enumerate(rng.sample(by[cname], n)):
        fn = Path(imgs[a["image_id"]]["file_name"]).name
        if fn not in cache:
            cache.clear(); cache[fn] = Image.open(root / "images" / fn).convert("L")
        x, y, w, h = a["bbox"]; mx, my = 0.1 * w, 0.1 * h
        crop = cache[fn].crop((int(x - mx), int(y - my), int(x + w + mx), int(y + h + my)))
        crop.thumbnail((cell - 8, cell - 8)) if max(crop.size) > cell - 8 else None
        s = (cell - 8) / max(crop.size); crop = crop.resize((int(crop.width * s), int(crop.height * s)), Image.NEAREST)
        sheet.paste(crop.convert("RGB"), (90 + c * cell + 4, r * cell + 4))
sheet.save(out)
print(out)
