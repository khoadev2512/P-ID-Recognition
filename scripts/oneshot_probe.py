"""PLAN_PHASE_C Bước 1: cheap probes, no retraining, CPU (docs/PLAN_PHASE_C.md §1).

One table for every probe, on two fixed crop sets cut from the DigitizePID COCO exactly as at
inference time (box + 10%, utils.crop_box.cut), embedded through oneshot.query_views and
aggregated by `mean` (the eval default):

  p1val  DECISION set (§1.2): val_sel images; known = 10 classes R1a trained on that have an ISO
         template (S1 S17 S21 S24 + VARIANT S6 S8 S14 S20 S22 S23, graded against their template
         as in P2), unknown = 7 OOV classes R1a trained on. <= 150 crops / class. For encoders
         that never trained on DigitizePID every class here is unseen, like P1, and no P1 image
         or class is touched.
  p1     REFERENCE only: the P1 sample of the CL2N check (150 crops / class of h_in + h_out from
         every image) plus 30 train crops / class of the 19 R1a classes (the CL2N center).

Rows: encoder (imagenet = scripts/make_imagenet_encoder.py, lo = R1a, dinov2 = timm ViT-S/14
DINOv2, frozen) x input (raw | a4 = utils.input_norm.ink_letterbox on crops AND templates) x
CL2N (off | on; center = mean unit embedding of every view of the 570 seen train crops). Plus
1d (HOG on a single view, raw and a4, no learning) and 1e (directory = one DigitizePID crop per
class from train images, 28 classes = 32 minus h_out, 5 draws, on the p1 sample).

    python scripts/oneshot_probe.py --out runs/phase_c/probe            # embed + score
    python scripts/oneshot_probe.py --out runs/phase_c/probe --score-only

Embeddings are cached per (encoder, input) in --out, so scoring can be redone. Reads only
data/canonical, data/raw/iso_symbols*, runs/*.pt; writes only --out.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.iso_prototypes import load_templates, read_vocab  # noqa: E402
from utils.crop_box import cut  # noqa: E402
from utils.input_norm import ink_letterbox  # noqa: E402
from utils.oneshot import (  # noqa: E402
    ViewSpec,
    build_prototypes,
    embed_images,
    holdout_classes,
    make_views,
)
from utils.oneshot_metrics import (  # noqa: E402
    assign_groups,
    auroc,
    read_symbol_map,
    symbol_map_path,
)
from utils.oneshot_metrics import read_vocab as read_vocab_rows  # noqa: E402

logger = logging.getLogger("oneshot_probe")

P1VAL_KNOWN = [
    "Symbol_1",
    "Symbol_17",
    "Symbol_21",
    "Symbol_24",
    "Symbol_6",
    "Symbol_8",
    "Symbol_14",
    "Symbol_20",
    "Symbol_22",
    "Symbol_23",
]
P1VAL_UNKNOWN = [
    "Symbol_7",
    "Symbol_16",
    "Symbol_26",
    "Symbol_28",
    "Symbol_29",
    "Symbol_31",
    "Symbol_32",
]
P1VAL_PER_CLASS = 150
P1_TEST_PER_CLASS, P1_SEEN_PER_CLASS = 150, 30
EXEMPLAR_SEEDS = (0, 1, 2, 3, 4)
DINOV2 = "vit_small_patch14_dinov2.lvd142m"
# frozen timm encoders by probe name: 1c (dinov2) and PLAN_PHASE_C C3 (registers, ViT-B)
TIMM_ENCODERS = {
    "dinov2": DINOV2,
    "dinov2reg": "vit_small_patch14_reg4_dinov2.lvd142m",
    "dinov2b": "vit_base_patch14_dinov2.lvd142m",
    "dinov2breg": "vit_base_patch14_reg4_dinov2.lvd142m",
}
C3_ENCODERS = ("dinov2reg", "dinov2b", "dinov2breg")
DINOV2_IMGSZ = 224
HOG_SIZE, HOG_CELL, HOG_BINS = 64, 8, 9
ENCODERS = ("imagenet", "lo", "dinov2")
INPUTS = ("raw", "a4")

# --------------------------------------------------------------------------- #
# crop sets
# --------------------------------------------------------------------------- #


def load_world(cfg) -> dict:
    coco = json.loads((ROOT / "data/canonical/annotations.coco.json").read_text())
    split = json.loads((ROOT / "data/canonical/digitizepid_splits.json").read_text())
    subsets = json.loads((ROOT / "configs/splits/digitizepid_val_test.json").read_text())
    lo = torch.load(ROOT / "runs/__global_lo__.pt", map_location="cpu", weights_only=False)
    holdout = holdout_classes(cfg)
    groups = assign_groups(
        read_symbol_map(symbol_map_path(None)),
        read_vocab_rows(ROOT / "data/raw/iso_symbols/vocab.csv"),
        list(lo["fine_classes"]),
        list(lo.get("excluded_classes", [])),
        holdout,
    )
    return {
        "coco": coco,
        "split": split,
        "val_sel": {str(s) for s in subsets["val_sel"]},
        "seen": sorted(lo["fine_classes"]),
        "holdout": holdout,
        "groups": groups,
    }


def pick_crops(world: dict) -> list[dict]:
    """Every crop the probe uses, one dict each: set, role, class, ann (COCO annotation), stem.

    The p1 part replays the CL2N check's draw exactly (one Random(0) over the classes in
    annotation order), so 1a reproduces its numbers; the rest draws per class."""
    coco, split = world["coco"], world["split"]
    cat = {c["id"]: c["name"] for c in coco["categories"]}
    stem = {i["id"]: Path(i["file_name"]).stem for i in coco["images"]}
    tested = sorted(world["holdout"].tested)
    seen = world["seen"]
    out: list[dict] = []

    by: dict[str, list] = defaultdict(list)
    for a in coco["annotations"]:
        if a.get("iscrowd"):
            continue
        name = cat[a["category_id"]]
        if name in tested or (name in seen and split.get(stem[a["image_id"]]) == "train"):
            by[name].append(a)
    rng = random.Random(0)
    for name, anns in by.items():
        n = P1_TEST_PER_CLASS if name in tested else P1_SEEN_PER_CLASS
        for a in rng.sample(anns, min(n, len(anns))):
            role = "test" if name in tested else "center"
            out.append({"set": "p1", "role": role, "class": name, "ann": a})

    wanted = set(P1VAL_KNOWN) | set(P1VAL_UNKNOWN)
    by = defaultdict(list)
    for a in coco["annotations"]:
        name = cat[a["category_id"]]
        if not a.get("iscrowd") and name in wanted and stem[a["image_id"]] in world["val_sel"]:
            by[name].append(a)
    for name in sorted(by):
        anns = sorted(by[name], key=lambda a: a["id"])
        draw = random.Random(f"p1val/0/{name}").sample(anns, min(P1VAL_PER_CLASS, len(anns)))
        for a in sorted(draw, key=lambda a: a["id"]):
            role = "known" if name in P1VAL_KNOWN else "unknown"
            out.append({"set": "p1val", "role": role, "class": name, "ann": a})

    h_out = set(world["holdout"].h_out)
    by = defaultdict(list)
    for a in coco["annotations"]:
        name = cat[a["category_id"]]
        if not a.get("iscrowd") and name not in h_out and split.get(stem[a["image_id"]]) == "train":
            by[name].append(a)
    for seed in EXEMPLAR_SEEDS:
        for name in sorted(by):
            anns = sorted(by[name], key=lambda a: a["id"])
            a = random.Random(f"ex/{seed}/{name}").choice(anns)
            out.append({"set": f"ex{seed}", "role": "exemplar", "class": name, "ann": a})
    for c in out:
        c["stem"] = stem[c["ann"]["image_id"]]
    return out


def cut_crops(crops: list[dict]) -> dict[int, np.ndarray]:
    """ann id -> RGB crop, each sheet read once."""
    by_image: dict[int, list[dict]] = defaultdict(list)
    for c in crops:
        by_image[c["ann"]["image_id"]].append(c)
    out: dict[int, np.ndarray] = {}
    for i, (_, cs) in enumerate(sorted(by_image.items()), 1):
        stem = cs[0]["stem"]
        pixels = np.array(Image.open(ROOT / "data/canonical/images" / f"{stem}.jpg").convert("RGB"))
        for c in cs:
            x, y, w, h = c["ann"]["bbox"]
            out[c["ann"]["id"]] = np.ascontiguousarray(cut(pixels, (x, y, x + w, y + h)))
        if i % 50 == 0:
            logger.info("cut %d / %d sheets", i, len(by_image))
    return out


# --------------------------------------------------------------------------- #
# encoders
# --------------------------------------------------------------------------- #


class TimmEncoder:
    """A frozen timm model as an utils.oneshot.Encoder (pooled pre-logits)."""

    def __init__(self, name: str, imgsz: int, device) -> None:
        import timm

        self.model = timm.create_model(name, pretrained=True, num_classes=0, img_size=imgsz)
        self.model.eval().to(device)

    @torch.inference_mode()
    def embed(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)


def load_encoder(name: str, cfg, device):
    """(encoder, transform) for one probe encoder."""
    from torchvision import transforms

    from utils.dataset import _IMAGENET_MEAN, _IMAGENET_STD, build_eval_transform
    from utils.model import load_arcface_encoder

    if name in TIMM_ENCODERS:
        tf = transforms.Compose(
            [
                transforms.Resize((DINOV2_IMGSZ, DINOV2_IMGSZ)),
                transforms.ToTensor(),
                transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ]
        )
        return TimmEncoder(TIMM_ENCODERS[name], DINOV2_IMGSZ, device), tf
    ckpt = ROOT / "runs" / {"imagenet": "__imagenet__.pt", "lo": "__global_lo__.pt"}[name]
    encoder, _ = load_arcface_encoder(cfg, ckpt, device)
    encoder.eval()
    return encoder, build_eval_transform(cfg)


def norm_fn(kind: str):
    return ink_letterbox if kind == "a4" else (lambda img: img)


@torch.inference_mode()
def embed_all(encoder, transform, crops, pixels, prep, spec, background, device, chunk=32):
    """ann id -> (V, D) unit view embeddings."""
    ids = sorted({c["ann"]["id"] for c in crops})
    out: dict[int, torch.Tensor] = {}
    for start in range(0, len(ids), chunk):
        part = ids[start : start + chunk]
        views = []
        for i in part:
            views += make_views(prep(Image.fromarray(pixels[i])), spec, background)
        emb = embed_images(encoder, views, transform, device, batch_size=256)
        for k, i in enumerate(part):
            out[i] = emb[k * spec.n_views : (k + 1) * spec.n_views].clone()
        if (start // chunk) % 20 == 0:
            logger.info("  embedded %d / %d crops", start + len(part), len(ids))
    return out


# --------------------------------------------------------------------------- #
# scoring (pure functions, tested in tests/test_oneshot_probe.py)
# --------------------------------------------------------------------------- #


def rank(views: torch.Tensor, protos: torch.Tensor, center: torch.Tensor | None, k: int = 3):
    """views (n, V, D) unit, protos (N, D) unit -> (top-k indices, top-k cosines), the
    oneshot_eval path with aggregate=mean: CL2N centers every view and every prototype, then
    the views are averaged and renormalized."""
    if center is not None:
        views = F.normalize(views - center, dim=-1)
        protos = F.normalize(protos - center, dim=-1)
    q = F.normalize(views.mean(dim=1), dim=-1)
    cos = q @ protos.T
    top = cos.topk(min(k, protos.shape[0]), dim=1)
    return top.indices, top.values


def summarize(rows: list[dict]) -> dict:
    """rows: class, known (bool), expected, top (names, best first), best, second.
    Macro top-1 / top-3 over known classes, per class, AUROC of best cos and of the margin
    (known higher), hub (most frequent top-1 over every row)."""
    known = [r for r in rows if r["known"]]
    unknown = [r for r in rows if not r["known"]]
    per = {}
    for name in sorted({r["class"] for r in known}):
        rs = [r for r in known if r["class"] == name]
        per[name] = {
            "n": len(rs),
            "expected": rs[0]["expected"],
            "top1": float(np.mean([r["top"][0] == r["expected"] for r in rs])),
            "top3": float(np.mean([r["expected"] in r["top"][:3] for r in rs])),
            "most_predicted": Counter(r["top"][0] for r in rs).most_common(1)[0][0],
        }
    hub = Counter(r["top"][0] for r in rows).most_common(1)[0] if rows else (None, 0)
    return {
        "macro_top1": float(np.mean([p["top1"] for p in per.values()])) if per else None,
        "macro_top3": float(np.mean([p["top3"] for p in per.values()])) if per else None,
        "auroc_cos": auroc([r["best"] for r in known], [r["best"] for r in unknown]),
        "auroc_margin": auroc(
            [r["best"] - r["second"] for r in known], [r["best"] - r["second"] for r in unknown]
        ),
        "hub": hub[0],
        "hub_share": hub[1] / len(rows) if rows else None,
        "n_known": len(known),
        "n_unknown": len(unknown),
        "per_class": per,
    }


def decide(base: dict, cand: dict, min_classes: int = 6, min_gain: float = 0.05) -> dict:
    """PLAN_PHASE_C §1.2 on the p1val summaries: PASS if top-1 rises on >= min_classes known
    classes (both 0 = not a rise) AND macro top-1 by >= min_gain; or if AUROC (cos or margin)
    rises by >= min_gain while macro top-1 drops by at most 0.01."""
    up = [c for c, p in cand["per_class"].items() if p["top1"] > base["per_class"][c]["top1"]]
    d_top1 = cand["macro_top1"] - base["macro_top1"]
    d_auc = max(cand["auroc_cos"] - base["auroc_cos"], cand["auroc_margin"] - base["auroc_margin"])
    by_top1 = len(up) >= min_classes and d_top1 >= min_gain
    by_auroc = d_auc >= min_gain and d_top1 >= -0.01
    return {
        "classes_up": up,
        "n_up": len(up),
        "d_macro_top1": d_top1,
        "d_auroc": d_auc,
        "pass_top1": by_top1,
        "pass_auroc": by_auroc,
        "pass": by_top1 or by_auroc,
    }


def hog_lite(gray: np.ndarray) -> np.ndarray:
    """HOG without block normalization on a HOG_SIZE square (the S9 analysis' version):
    Sobel gradients, HOG_CELL cells, HOG_BINS unsigned orientation bins, L2-normalized."""
    import cv2

    g = cv2.resize(gray, (HOG_SIZE, HOG_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    ang = (np.degrees(np.arctan2(gy, gx)) % 180.0) / (180.0 / HOG_BINS)
    b = np.minimum(ang.astype(int), HOG_BINS - 1)
    n = HOG_SIZE // HOG_CELL
    hist = np.zeros((n, n, HOG_BINS), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            sl = (slice(i * HOG_CELL, (i + 1) * HOG_CELL), slice(j * HOG_CELL, (j + 1) * HOG_CELL))
            hist[i, j] = np.bincount(b[sl].ravel(), mag[sl].ravel(), HOG_BINS)
    v = hist.ravel()
    return v / (np.linalg.norm(v) + 1e-8)


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #


def rows_for(crops, emb, protos, names, center, expected_of, known_role):
    idx, val = rank(torch.stack([emb[c["ann"]["id"]] for c in crops]), protos, center)
    rows = []
    for c, ii, vv in zip(crops, idx.tolist(), val.tolist(), strict=True):
        rows.append(
            {
                "class": c["class"],
                "known": c["role"] == known_role,
                "expected": expected_of(c["class"]),
                "top": [names[i] for i in ii],
                "best": vv[0],
                "second": vv[1],
                "stem": c["stem"],
            }
        )
    return rows


def score_config(world, crops, emb, protos, names, exemplar_emb):
    groups = world["groups"]
    by_set = defaultdict(list)
    for c in crops:
        by_set[(c["set"], c["role"])].append(c)
    seen_views = torch.cat([emb[c["ann"]["id"]] for c in by_set[("p1", "center")]])
    center = seen_views.mean(dim=0)
    template = lambda name: groups[name].template or ""  # noqa: E731
    out = {}
    for cl2n in (False, True):
        cen = center if cl2n else None
        res = {}
        p1val = by_set[("p1val", "known")] + by_set[("p1val", "unknown")]
        res["p1val"] = summarize(rows_for(p1val, emb, protos, names, cen, template, "known"))
        h_in = set(world["holdout"].h_in)
        p1 = [
            dict(c, role="known" if c["class"] in h_in else "unknown")
            for c in by_set[("p1", "test")]
        ]
        res["p1"] = summarize(rows_for(p1, emb, protos, names, cen, template, "known"))
        # 1e: one DigitizePID crop per class as the directory, five draws
        draws = []
        for seed in EXEMPLAR_SEEDS:
            ex = by_set[(f"ex{seed}", "exemplar")]
            ex_names = [c["class"] for c in ex]
            ex_protos = F.normalize(
                torch.stack([exemplar_emb[c["ann"]["id"]].mean(0) for c in ex]), dim=-1
            )
            used = {c["stem"] for c in ex}
            q = [c for c in p1 if c["stem"] not in used]
            draws.append(
                summarize(rows_for(q, emb, ex_protos, ex_names, cen, lambda n: n, "known"))
            )
        res["same_style"] = {
            "draws": draws,
            "macro_top1_mean": float(np.mean([d["macro_top1"] for d in draws])),
            "macro_top1_std": float(np.std([d["macro_top1"] for d in draws], ddof=1)),
            "auroc_cos_mean": float(np.mean([d["auroc_cos"] for d in draws])),
            "n_classes": len(by_set[("ex0", "exemplar")]),
        }
        out["cl2n" if cl2n else "raw"] = res
    return out


def score_hog(world, crops, pixels, templates):
    groups = world["groups"]
    names = list(templates)
    out = {}
    for kind in INPUTS:
        prep = norm_fn(kind)

        def feat(img, prep=prep):
            return hog_lite(np.asarray(prep(img).convert("L")))

        protos = torch.tensor(np.stack([feat(templates[n]) for n in names]))
        res = {}
        for set_name in ("p1val", "p1"):
            cs = [c for c in crops if c["set"] == set_name and c["role"] != "center"]
            if set_name == "p1":
                h_in = set(world["holdout"].h_in)
                cs = [dict(c, role="known" if c["class"] in h_in else "unknown") for c in cs]
            views = torch.tensor(
                np.stack([feat(Image.fromarray(pixels[c["ann"]["id"]])) for c in cs])
            )
            emb = {c["ann"]["id"]: views[k : k + 1] for k, c in enumerate(cs)}
            res[set_name] = summarize(
                rows_for(cs, emb, protos, names, None, lambda n: groups[n].template or "", "known")
            )
        out[kind] = res
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--encoders", default=",".join(ENCODERS))
    ap.add_argument("--inputs", default=",".join(INPUTS))
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(ROOT / "configs/config.yaml")
    cfg.data_root = str(ROOT / "data")
    world = load_world(cfg)
    crops = pick_crops(world)
    with (args.out / "crops.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "role", "class", "ann_id", "image"])
        for c in crops:
            w.writerow([c["set"], c["role"], c["class"], c["ann"]["id"], c["stem"]])
    logger.info("%d crops (%s)", len(crops), dict(Counter(c["set"] for c in crops)))

    rows = read_vocab(ROOT / "data/raw/iso_symbols/vocab.csv")
    templates, coarse_of = load_templates(ROOT / "data/raw/iso_symbols_clean", rows)
    names = list(templates)
    spec_q = ViewSpec.from_cfg(cfg.oneshot.query_views)
    spec_t = ViewSpec.from_cfg(cfg.oneshot.template_views)
    bg = tuple(int(c) for c in cfg.oneshot.background)
    device = torch.device(args.device)

    pixels = None if args.score_only else cut_crops(crops)
    results = {}
    for enc_name in args.encoders.split(","):
        for kind in args.inputs.split(","):
            cache = args.out / f"emb_{enc_name}_{kind}.pt"
            if not args.score_only or not cache.exists():
                if pixels is None:
                    pixels = cut_crops(crops)
                logger.info("embedding %s / %s", enc_name, kind)
                encoder, tf = load_encoder(enc_name, cfg, device)
                prep = norm_fn(kind)
                with torch.inference_mode():
                    directory = build_prototypes(
                        encoder,
                        {n: prep(t) for n, t in templates.items()},
                        tf,
                        spec_t,
                        coarse_of,
                        device,
                        bg,
                    )
                stored = {
                    ("imagenet", "raw"): "iso_prototypes_imagenet.pt",
                    ("lo", "raw"): "iso_prototypes_lo.pt",
                }.get((enc_name, kind))
                if stored:  # the rebuilt directory must be the one the eval used
                    ref = torch.load(ROOT / "runs" / stored, map_location="cpu", weights_only=False)
                    diff = float((ref["embeddings"].float() - directory.embeddings).abs().max())
                    logger.info("prototypes vs runs/%s: max |diff| %.2e", stored, diff)
                    assert diff < 1e-4, diff
                emb = embed_all(encoder, tf, crops, pixels, prep, spec_q, bg, device)
                torch.save(
                    {"emb": emb, "protos": directory.embeddings, "names": directory.names}, cache
                )
            data = torch.load(cache, weights_only=False)
            assert data["names"] == names
            results[f"{enc_name}/{kind}"] = score_config(
                world, crops, data["emb"], data["protos"], names, data["emb"]
            )
            logger.info("scored %s / %s", enc_name, kind)
    if pixels is not None or not (args.out / "hog.json").exists():
        pixels = pixels if pixels is not None else cut_crops(crops)
        hog = score_hog(world, crops, pixels, templates)
        (args.out / "hog.json").write_text(json.dumps(hog, indent=1))
    results["hog"] = json.loads((args.out / "hog.json").read_text())
    (args.out / "results.json").write_text(json.dumps(results, indent=1))
    report(results, args.out)


def _fmt(v):
    return "     -" if v is None else f"{v:6.3f}"


def report(results: dict, out: Path) -> None:
    lines = []
    head = (
        f"{'dòng':28s} {'p1val top1':>10s} {'top3':>6s} {'AUROC':>6s} {'lề':>6s} {'hub':>6s}"
        f"   | {'p1 top1':>8s} {'top3':>6s} {'AUROC':>6s}"
    )
    lines.append(head)
    for key, res in results.items():
        if key == "hog":
            variants = {f"hog/{k}": v for k, v in res.items()}
        else:
            variants = {f"{key}/{c}": v for c, v in res.items()}
        for name, r in variants.items():
            a, b = r["p1val"], r["p1"]
            lines.append(
                f"{name:28s} {_fmt(a['macro_top1']):>10s} {_fmt(a['macro_top3'])} "
                f"{_fmt(a['auroc_cos'])} {_fmt(a['auroc_margin'])} {_fmt(a['hub_share'])}"
                f"   | {_fmt(b['macro_top1']):>8s} {_fmt(b['macro_top3'])} {_fmt(b['auroc_cos'])}"
            )
    lines.append("")
    lines.append("1e ảnh mẫu cùng phong cách (mẫu P1, 28 lớp, ngẫu nhiên top-1 = 1/28 = 0.036):")
    for key, res in results.items():
        if key == "hog":
            continue
        for c, r in res.items():
            s = r["same_style"]
            lines.append(
                f"  {key}/{c:5s}  top1 {s['macro_top1_mean']:.3f} ± {s['macro_top1_std']:.3f}"
                f"   AUROC {s['auroc_cos_mean']:.3f}"
                f"   (P1 với ảnh mẫu ISO: {r['p1']['macro_top1']:.3f})"
            )
    if "imagenet/raw" in results:
        base = results["imagenet/raw"]["cl2n"]["p1val"]
        lines.append("")
        lines.append("Luật §1.2 trên p1val, mốc = imagenet/raw/cl2n (chỉ 1b và 1c quyết định):")
        cands = []
        if "imagenet/a4" in results:
            cands.append(("1b", "imagenet/a4/cl2n", results["imagenet/a4"]["cl2n"]["p1val"]))
        dino = [
            (f"{k}/{c}", r["p1val"])
            for k, res in results.items()
            if k.startswith("dinov2/")
            for c, r in res.items()
        ]
        if dino:  # 1c: the best DINOv2 variant on p1val macro top-1 (§1.2)
            name, r = max(dino, key=lambda x: x[1]["macro_top1"])
            cands.append(("1c", name, r))
        for tag, name, r in cands:
            d = decide(base, r)
            lines.append(
                f"  {tag} {name:22s} lớp tăng {d['n_up']}/10  Δtop1 {d['d_macro_top1']:+.3f}"
                f"  ΔAUROC {d['d_auroc']:+.3f}  -> {'ĐẠT' if d['pass'] else 'KHÔNG ĐẠT'}"
            )
    c3 = [k for k in C3_ENCODERS if any(key.startswith(f"{k}/") for key in results)]
    if c3 and "dinov2/a4" in results:
        base = results["dinov2/a4"]["cl2n"]["p1val"]
        lines.append("")
        lines.append("C3 (PLAN_PHASE_C §2C.3) trên p1val, mốc = dinov2/a4/cl2n:")
        for k in c3:
            variants = [
                (f"{key}/{c}", r["p1val"])
                for key, res in results.items()
                if key.startswith(f"{k}/")
                for c, r in res.items()
            ]
            name, r = max(variants, key=lambda x: x[1]["macro_top1"])
            d = decide(base, r)
            lines.append(
                f"  {name:24s} top1 {r['macro_top1']:.3f} lớp tăng {d['n_up']}/10  "
                f"Δtop1 {d['d_macro_top1']:+.3f}  ΔAUROC {d['d_auroc']:+.3f}  "
                f"-> {'ĐẠT' if d['pass'] else 'KHÔNG ĐẠT'}"
            )
    text = "\n".join(lines)
    (out / "report.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
