import json, math, random, sys
from pathlib import Path
import cv2, numpy as np
from PIL import Image, ImageDraw

def _kmeans(pts, k, rng, iters=10):
    c = pts[rng.sample(range(len(pts)), k)].astype(np.float32)
    for _ in range(iters):
        lab = np.argmin(((pts[:, None, :] - c[None]) ** 2).sum(-1), 1)
        for j in range(k):
            if (lab == j).any(): c[j] = pts[lab == j].mean(0)
    return lab

def smooth_warp(dark, rng, mag=0.06, grid=3):
    H, W = dark.shape
    nrng = np.random.default_rng(rng.randrange(1 << 30))
    dx = nrng.uniform(-mag, mag, (grid, grid)).astype(np.float32) * W
    dy = nrng.uniform(-mag, mag, (grid, grid)).astype(np.float32) * H
    dx = cv2.resize(dx, (W, H), interpolation=cv2.INTER_CUBIC); dy = cv2.resize(dy, (W, H), interpolation=cv2.INTER_CUBIC)
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    return cv2.remap(dark, xs + dx, ys + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def aug(gray, rng, mode, rot=20, sc=(0.9, 1.1), sh=0.05, kmax=4, min_frac=0.03):
    if mode == "warp": rot, sc, sh = 8, (0.92, 1.08), 0.05
    g = gray.astype(np.float32)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if ink.sum() == 0: return gray
    bg = float(np.median(gray[ink == 0]))
    ink_d = cv2.dilate(ink, np.ones((3, 3), np.uint8)) > 0
    dark = np.clip(bg - g, 0, None) * ink_d
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    tot = st[1:, cv2.CC_STAT_AREA].sum()
    parts = []
    lab_d = cv2.dilate(lab.astype(np.float32), np.ones((3,3),np.uint8), ).astype(np.int32)  # rough owner map for halo
    for i in range(1, n):
        m = lab_d == i
        if st[i, cv2.CC_STAT_AREA] < min_frac * tot: parts.append((m, False)); continue
        if mode == "cluster" and st[i, cv2.CC_STAT_AREA] >= 0.25 * tot:
            ys, xs = np.nonzero(m); pts = np.stack([xs, ys], 1).astype(np.float32)
            k = rng.randint(2, kmax); l = _kmeans(pts, k, rng)
            for j in range(k):
                mm = np.zeros_like(m); mm[ys[l == j], xs[l == j]] = True; parts.append((mm, True))
        else:
            parts.append((m, True))
    movable = [p for p in parts if p[1]]
    H, W = gray.shape
    def M_for(m):
        ys, xs = np.nonzero(m); cx, cy = xs.mean(), ys.mean()
        a = math.radians(rng.uniform(-rot, rot)); s = rng.uniform(*sc); shx = rng.uniform(-sh, sh)
        A = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]) @ np.array([[1, shx], [0, 1]]) * s
        t = np.array([cx, cy]) - A @ np.array([cx, cy])
        return np.hstack([A, t[:, None]]).astype(np.float32)
    if mode == "warp":
        out = np.zeros_like(dark)
        for m, mv in parts:
            layer = dark * m
            if mv and len(movable) > 1: layer = cv2.warpAffine(layer, M_for(m), (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
            out = np.maximum(out, layer)
        out = smooth_warp(out, rng)
        r = rng.random()
        if r < 0.25: out = cv2.dilate(out, np.ones((2, 2), np.uint8))
        elif r < 0.5: out = cv2.erode(out, np.ones((2, 2), np.uint8))
    elif len(movable) <= 1:   # whole-crop fallback
        M = M_for(np.ones_like(ink, bool))
        out = cv2.warpAffine(dark, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    else:
        out = np.zeros_like(dark)
        for m, mv in parts:
            layer = dark * m
            if mv: layer = cv2.warpAffine(layer, M_for(m), (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
            out = np.maximum(out, layer)
    base = g.copy(); base[ink_d] = bg
    return np.clip(np.minimum(base, bg - out), 0, 255).astype(np.uint8)

root = Path(sys.argv[1]); outp = Path(sys.argv[2]); mode = sys.argv[3]; classes = sys.argv[4].split(",")
coco = json.load(open(root / "annotations.coco.json"))
cat = {c["id"]: c["name"] for c in coco["categories"]}; imgs = {i["id"]: i for i in coco["images"]}
by = {}
for a in coco["annotations"]: by.setdefault(cat[a["category_id"]], []).append(a)
rng = random.Random(0); cell = 120; nv = 5
sheet = Image.new("RGB", (90 + cell * (nv + 1) , cell * len(classes) * 2), "white"); d = ImageDraw.Draw(sheet)
row = 0
for cname in classes:
    for a in random.Random(cname).sample(by[cname], 2):
        g = cv2.imread(str(root / "images" / Path(imgs[a["image_id"]]["file_name"]).name), cv2.IMREAD_GRAYSCALE)
        x, y, w, h = a["bbox"]; mx, my = .1*w, .1*h
        c = g[max(0,int(y-my)):int(y+h+my), max(0,int(x-mx)):int(x+w+mx)]
        tiles = [c] + [aug(c, rng, mode) for _ in range(nv)]
        d.text((4, row*cell + cell//2), cname + ("" if row % 2 == 0 else ""), fill="red")
        for j, t in enumerate(tiles):
            im = Image.fromarray(t); s = (cell-10)/max(im.size); im = im.resize((int(im.width*s), int(im.height*s)), Image.BILINEAR)
            sheet.paste(im.convert("RGB"), (90 + j*cell + 4, row*cell + 4))
            if j == 0: d.rectangle([90+2, row*cell+2, 90+cell-4, row*cell+cell-4], outline="blue")
        row += 1
sheet.save(outp); print(outp)
