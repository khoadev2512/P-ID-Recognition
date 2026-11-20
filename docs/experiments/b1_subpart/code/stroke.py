"""Relative stroke width = stroke px / ink-bbox long side, ISO templates vs DigitizePID crops vs augmented."""
import pickle, random, sys
import cv2, numpy as np
sys.path.insert(0, sys.argv[2]); from augproto import aug
def rel_stroke(gray):
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dt = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    ridge = (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8))) & (dt > 0)
    ys, xs = np.nonzero(ink)
    return float(2 * np.median(dt[ridge]) / max(np.ptp(ys) + 1, np.ptp(xs) + 1))
iso = {"Symbol_1": "X8074", "Symbol_3": "X8068", "Symbol_4": "2101", "Symbol_10": "X8076"}
data = pickle.load(open(sys.argv[1], "rb"))["train"]
rng = random.Random(0)
print(f"{'class':10s} {'ISO':>6s} {'crop':>6s} {'opt1 p5-p95':>14s} {'opt3 p5-p95':>14s}")
for c, t in iso.items():
    tg = cv2.imread(f"{sys.argv[3]}/{t}.png", cv2.IMREAD_GRAYSCALE)
    crops = data[c][:100]
    base = np.median([rel_stroke(x) for x in crops])
    o1 = [rel_stroke(aug(x, rng, "cc")) for x in crops]
    o3 = [rel_stroke(aug(x, rng, "warp")) for x in crops]
    p = lambda v: f"{np.percentile(v,5):.3f}-{np.percentile(v,95):.3f}"
    print(f"{c:10s} {rel_stroke(tg):6.3f} {base:6.3f} {p(o1):>14s} {p(o3):>14s}")
