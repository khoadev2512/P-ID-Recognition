import json, random, sys
from pathlib import Path
import cv2, numpy as np
root = Path(sys.argv[1])
coco = json.load(open(root / "annotations.coco.json"))
cat = {c["id"]: c["name"] for c in coco["categories"]}
imgs = {i["id"]: i for i in coco["images"]}
rng = random.Random(1)
anns = [a for a in coco["annotations"] if int(cat[a["category_id"]].split("_")[1]) <= 32]
by_img = {}
for a in anns: by_img.setdefault(a["image_id"], []).append(a)
stats = {}
for iid in rng.sample(sorted(by_img), 12):
    g = cv2.imread(str(root / "images" / Path(imgs[iid]["file_name"]).name), cv2.IMREAD_GRAYSCALE)
    for a in by_img[iid]:
        x, y, w, h = a["bbox"]; mx, my = .1*w, .1*h
        c = g[max(0,int(y-my)):int(y+h+my), max(0,int(x-mx)):int(x+w+mx)]
        _, ink = cv2.threshold(c, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        areas = sorted(st[1:, cv2.CC_STAT_AREA], reverse=True)
        tot = sum(areas) or 1
        sig = [s for s in areas if s >= 0.03 * tot]   # >=3% of the ink
        stats.setdefault(cat[a["category_id"]], []).append(len(sig))
for k in sorted(stats, key=lambda s: int(s.split("_")[1])):
    v = np.array(stats[k]); print(f"{k:10s} n={len(v):3d}  >=2 parts: {np.mean(v>=2):.0%}  median parts={int(np.median(v))}")
