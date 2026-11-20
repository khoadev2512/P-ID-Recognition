"""Review sheet: real crops (blue frame) + augmented versions from utils.augment.SubpartWarp."""
import pickle, sys
from PIL import Image, ImageDraw
from utils.augment import SubpartWarp
data = pickle.load(open(sys.argv[1], "rb"))["val"]; out = sys.argv[2]; classes = sys.argv[3].split(",")
aug = SubpartWarp(p=1.0, seed=42)
cell, nv, per = 120, 4, 2
sheet = Image.new("RGB", (90 + cell * (nv + 1), cell * len(classes) * per), "white"); d = ImageDraw.Draw(sheet)
row = 0
for c in classes:
    for x in data[c][3:3 + per]:
        src = Image.fromarray(x).convert("RGB")
        tiles = [src] + [aug(src) for _ in range(nv)]
        d.text((4, row * cell + cell // 2), c, fill="red")
        for j, t in enumerate(tiles):
            s = (cell - 10) / max(t.size); t = t.resize((int(t.width * s), int(t.height * s)), Image.BILINEAR)
            sheet.paste(t, (90 + j * cell + 4, row * cell + 4))
        d.rectangle([92, row * cell + 2, 90 + cell - 4, row * cell + cell - 4], outline="blue")
        row += 1
sheet.save(out); print(out)
