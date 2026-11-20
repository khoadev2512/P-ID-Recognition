"""Cache grayscale valve crops (10% margin, as pid-prep-crops) per class and split."""
import json, pickle, random, sys
from pathlib import Path
import cv2, pandas as pd
root = Path(sys.argv[1]); out = Path(sys.argv[2])
N = {"train": 250, "val": 100}
coco = json.load(open(root / "annotations.coco.json"))
cat = {c["id"]: c["name"] for c in coco["categories"]}
split = dict(zip(*[pd.read_csv(root / "manifest.csv", dtype={"id": str})[k] for k in ("id", "split")]))
by_img = {}
for a in coco["annotations"]:
    n = cat[a["category_id"]]
    if int(n.split("_")[1]) <= 16: by_img.setdefault(a["image_id"], []).append(a)
imgs = {i["id"]: i for i in coco["images"]}
data = {"train": {}, "val": {}}
order = sorted(by_img); random.Random(0).shuffle(order)
for iid in order:
    fn = Path(imgs[iid]["file_name"]).name; sp = split.get(Path(fn).stem)
    if sp not in data: continue
    need = [a for a in by_img[iid] if len(data[sp].get(cat[a["category_id"]], [])) < N[sp]]
    if not need: continue
    g = cv2.imread(str(root / "images" / fn), cv2.IMREAD_GRAYSCALE)
    for a in need:
        x, y, w, h = a["bbox"]; mx, my = .1 * w, .1 * h
        c = g[max(0, int(y - my)):int(y + h + my), max(0, int(x - mx)):int(x + w + mx)].copy()
        data[sp].setdefault(cat[a["category_id"]], []).append(c)
    if all(len(data[s].get(f"Symbol_{k}", [])) >= N[s] for s in data for k in range(1, 17)): break
pickle.dump(data, open(out, "wb"))
print({s: {k: len(v) for k, v in sorted(d.items(), key=lambda kv: int(kv[0].split('_')[1]))} for s, d in data.items()})
