"""PLAN_PHASE_C §2L: fine-tune DINOv2-S on ISO templates only, with similar-template clusters held
out, and test whether the skill "see the small detail" carries over to template pairs never trained.

Two folds (held out: K2; K3 + K4 + K5). Each fold trains on every other template of the `formB`
directory (41 classes, 42 images: + valve_butterfly@B), drawn through H2's generator
(utils.template_variants: the clean template with H2's share 1 / (1 + K_VIEWS), else dirty_view)
then A4 and one random query view (2 scales x 4 turns), seed 2027. ArcFace; LP-FT: 300 steps on
the class centres alone, then 1,500 steps with the last 2 blocks + final LayerNorm open; WiSE-FT
alpha = 0.5 (alpha = 1.0 reported, never chosen).

Scoring = the 2B path of k4_probe on P4-val (A4, 8 views, mean, CL2N; centre rebuilt with each
encoder on the same 570 centre crops; 43 prototypes rebuilt), P1-val (style guard) and V-ISO
clean (collapse guard (a)). The baseline is the frozen encoder through the SAME code, and must
reproduce k4's `41+B/mean` (P4-val 0.770, P1-val 0.485) before anything is compared.

    python scripts/ft_templates.py --out runs/phase_c/ft [--device mps] [--workers 6]
    python scripts/ft_templates.py --out runs/phase_c/ft --deploy [--device mps]

--deploy = FT-triển khai (§2L): the same recipe on all 42 template images (nothing held out, fold
"all"), alpha = 0.5 only -> `__dinov2_ft_all_a0.5__.pt` + its sha256, scored once on the same
sets as a description (every class trained: optimistic, decides nothing).

Writes only --out: per-model embeddings, encoder checkpoints in the `__dinov2__.pt` format
(scripts/make_imagenet_encoder.py; readable by the pipeline as oneshot.encoder_ckpt), results.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import oneshot_probe as probe  # noqa: E402
from h2_template_variants import load_svg_templates, vocab_templates  # noqa: E402
from k4_probe import _auroc as auroc_known  # noqa: E402
from k4_probe import boot_delta, class_scores, confidence, macro_top1  # noqa: E402

from utils.crop_box import cut  # noqa: E402
from utils.oneshot import ViewSpec, embed_images, make_views  # noqa: E402
from utils.template_variants import K_VIEWS, axes_of, dirty_view, template_seed  # noqa: E402

logger = logging.getLogger("ft")

DROPPED = "pipeline"  # §2G
FORM_B = ("valve_butterfly", "X8075_2", ROOT / "data/raw/iso_symbols_forms/valve_butterfly@B.png")
CLUSTERS = {  # §2L: connected components of CL2N cos >= 0.85 between the 41 templates (§2K)
    "K1": [
        "compressor_centrifugal",
        "compressor_vacuum",
        "pump_centrifugal",
        "pump_diaphragm",
        "pump_general",
    ],
    "K2": ["check_valve_globe", "check_valve_swing", "valve_butterfly", "valve_globe"],
    "K3": ["valve_needle", "valve_three_way"],
    "K4": ["breather_valve", "check_valve_general"],
    "K5": ["blind_disc", "interchangeable_disc", "interchangeable_disc_open", "open_disc"],
}
CLOSE_COS = 0.85
FOLDS = {"k2": ("K2",), "k345": ("K3", "K4", "K5")}
DEPLOY = "all"  # FT-triển khai: every template trained
SEED = 2027
P_CLEAN = 1 / (1 + K_VIEWS)  # H2: template_views = [clean] + K_VIEWS dirty
BATCH, LP_STEPS, FT_STEPS = 64, 300, 1500
LR_HEAD, LR_BACKBONE, WEIGHT_DECAY = 1e-3, 1e-5, 0.05
N_OPEN = 2  # last transformer blocks opened (+ the final LayerNorm)
ALPHAS = (0.5, 1.0)  # WiSE-FT; 0.5 decides, 1.0 is the ablation
ALPHA_MAIN = 0.5
MIN_CROPS = 20
GAIN, CLASS_DROP, P1_DROP, VISO_DROP = 0.05, 0.10, 0.01, 0.10
BASE_CHECK = {"p4val": 0.7698, "p1val": 0.4853}  # k4_probe 41+B/mean (results.json), 4 digits
DINO_CFG = {"backbone": probe.DINOV2, "imgsz": 224, "embed_dim": 384}


# --------------------------------------------------------------------------- #
# pure functions (tests/test_ft_templates.py)
# --------------------------------------------------------------------------- #


def components(names: list[str], sim: torch.Tensor, thr: float = CLOSE_COS) -> list[list[str]]:
    """Connected components (size >= 2) of the graph `sim[i, j] >= thr`, each sorted."""
    adj = defaultdict(set)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if float(sim[i, j]) >= thr:
                adj[i].add(j)
                adj[j].add(i)
    seen, out = set(), []
    for i in sorted(adj):
        if i in seen:
            continue
        stack, comp = [i], set()
        while stack:
            k = stack.pop()
            if k not in comp:
                comp.add(k)
                stack += list(adj[k])
        seen |= comp
        out.append(sorted(names[k] for k in comp))
    return sorted(out)


def held_out(fold: str) -> list[str]:
    return sorted(c for k in FOLDS[fold] for c in CLUSTERS[k])


def train_items(items: list[tuple], fold: str | None) -> list[tuple]:
    """Items whose first field is a class, minus the fold's held-out classes (None = keep all)."""
    out = set(held_out(fold)) if fold else set()
    return [it for it in items if it[0] not in out]


def ft_meta(fold: str, alpha: float, classes: list[str]) -> dict:
    """What a fine-tuned checkpoint records about itself (fold DEPLOY = nothing held out)."""
    return {
        "plan": "PLAN_PHASE_C §2L",
        "fold": fold,
        "held_out": [] if fold == DEPLOY else held_out(fold),
        "alpha": alpha,
        "seed": SEED,
        "recipe": {
            "batch": BATCH,
            "lp_steps": LP_STEPS,
            "ft_steps": FT_STEPS,
            "lr_head": LR_HEAD,
            "lr_backbone": LR_BACKBONE,
            "weight_decay": WEIGHT_DECAY,
            "n_open": N_OPEN,
            "p_clean": P_CLEAN,
        },
        "train_classes": classes,
    }


def sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wise(theta0: dict, theta1: dict, alpha: float) -> dict:
    """WiSE-FT: (1 - alpha) * theta0 + alpha * theta1 per key (non-float tensors: theta1)."""
    if theta0.keys() != theta1.keys():
        raise ValueError("WiSE-FT needs two state dicts of the same model")
    return {
        k: (1 - alpha) * theta0[k] + alpha * v if v.is_floating_point() else v.clone()
        for k, v in theta1.items()
    }


def assemble(frames: dict[str, pd.DataFrame], measured: dict[str, list[str]]) -> pd.DataFrame:
    """One frame: for each fold, its rows of the classes it held out (LOCO, as F0-b)."""
    return pd.concat(
        [frames[f][frames[f]["class"].isin(cls)] for f, cls in measured.items()]
    ).sort_values("ann_id", kind="stable")


def decide(
    d_macro: float,
    ci: tuple[float, float],
    d_per_class: dict[str, float],
    p1: dict[str, dict[str, float]],
    viso: dict[str, dict[str, float]],
) -> dict:
    """§2L rule FT-chính. p1[fold] = {d_top1, d_auroc}; viso[fold] = {class: d_top1}."""
    checks = {
        "gain": d_macro >= GAIN,
        "ci_above_0": ci[0] > 0,
        "no_class_drop": all(v >= -CLASS_DROP for v in d_per_class.values()),
        "p1val_top1": all(x["d_top1"] >= -P1_DROP for x in p1.values()),
        "p1val_auroc": all(x["d_auroc"] >= -P1_DROP for x in p1.values()),
        "viso_clean": all(v >= -VISO_DROP for f in viso.values() for v in f.values()),
    }
    return {"checks": checks, "pass": all(checks.values())}


def sample(item: tuple[int, Image.Image, list], i: int, fold: str, spec_q, bg, a4) -> Image.Image:
    """Training image number `i` of a fold from one (class, template, pipe axes) item: the clean
    template (share P_CLEAN) or one dirty_view, then A4, then one random query view. The same
    (fold, i, item) gives the same picture whatever the worker."""
    _, img, axes = item
    rng = random.Random(template_seed(SEED, f"{fold}/{i}"))
    pic = img.convert("RGB") if rng.random() < P_CLEAN else dirty_view(img, axes, rng)
    one = ViewSpec((rng.choice(spec_q.scales),), (rng.choice(spec_q.orientations),))
    return make_views(a4(pic), one, bg)[0]


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #


class Stream(torch.utils.data.Dataset):
    """steps x BATCH samples; sample i picks a class uniformly, then one of its images."""

    def __init__(self, items, n, fold, spec_q, bg, tf) -> None:
        self.by_class = defaultdict(list)
        for k, it in enumerate(items):
            self.by_class[it[0]].append(k)
        self.classes = sorted(self.by_class)
        self.items, self.n, self.fold = items, n, fold
        self.spec_q, self.bg, self.tf = spec_q, bg, tf

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        rng = random.Random(template_seed(SEED, f"{self.fold}/pick/{i}"))
        c = rng.randrange(len(self.classes))
        item = self.items[rng.choice(self.by_class[self.classes[c]])]
        pic = sample(item, i, self.fold, self.spec_q, self.bg, probe.norm_fn("a4"))
        return self.tf(pic), c


def build_model(n_cls: int, device):
    from utils.model import ArcFaceModel

    return ArcFaceModel(dino_cfg(), num_fine=n_cls).to(device)


def dino_cfg():
    return OmegaConf.create(
        {
            "fgc": {
                "backbone": DINO_CFG["backbone"],
                "pretrained": True,
                "imgsz": DINO_CFG["imgsz"],
                "arcface": {"s": 30.0, "m": 0.5, "embed_dim": DINO_CFG["embed_dim"]},
            }
        }
    )


def train(items, fold, tf, spec_q, bg, device, workers) -> tuple[dict, dict, list[str]]:
    """-> (theta0, theta_ft) backbone state dicts, training class names."""
    torch.manual_seed(SEED)
    data = Stream(items, (LP_STEPS + FT_STEPS) * BATCH, fold, spec_q, bg, tf)
    model = build_model(len(data.classes), device)
    theta0 = copy.deepcopy(model.backbone.state_dict())
    bb = model.backbone
    for p in bb.parameters():
        p.requires_grad_(False)
    opened = [*bb.blocks[-N_OPEN:].parameters(), *bb.norm.parameters()]
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=BATCH,
        num_workers=workers,
        persistent_workers=workers > 0,
        shuffle=False,
    )
    opt = torch.optim.AdamW(model.head.parameters(), lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    model.train()
    t0, run = time.time(), []
    for step, (x, y) in enumerate(loader):
        if step == LP_STEPS:  # LP-FT: open the last blocks once the centres are fitted
            for p in opened:
                p.requires_grad_(True)
            opt = torch.optim.AdamW(
                [
                    {"params": model.head.parameters(), "lr": LR_HEAD},
                    {"params": opened, "lr": LR_BACKBONE},
                ],
                weight_decay=WEIGHT_DECAY,
            )
        x, y = x.to(device), y.to(device)
        loss = F.cross_entropy(model(x, y), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        run.append(loss.item())
        if (step + 1) % 100 == 0:
            logger.info(
                "%s step %d / %d  loss %.3f  (%.0fs)",
                fold,
                step + 1,
                len(loader),
                np.mean(run[-100:]),
                time.time() - t0,
            )
    theta1 = {k: v.detach().cpu().clone() for k, v in bb.state_dict().items()}
    theta0 = {k: v.cpu() for k, v in theta0.items()}
    return theta0, theta1, data.classes


def checkpoint(theta: dict, meta: dict) -> dict:
    """The `__dinov2__.pt` layout (no head: the one-shot path never reads class centres)."""
    from utils.classmap import load_classmap
    from utils.paths import DataPaths

    model = build_model(0, "cpu")
    model.backbone.load_state_dict(theta)
    classes = load_classmap(DataPaths(root=ROOT / "data").classes_yaml).fine_classes
    return {
        "model": model.state_dict(),
        "epoch": None,
        "val_acc": None,
        "best_val_acc": None,
        "fine_classes": [],
        "excluded_classes": list(classes),  # trained on ISO templates only: no DigitizePID class
        "class_centers": model.class_centers().cpu(),
        "pretrained_only": False,
        "imagenet_only": False,
        "pretrained_weights": f"timm/{DINO_CFG['backbone']}",
        "fine_tuned": meta,
    }


def transform():
    """The probe / pipeline eval transform of the 224 px DINOv2 (viso_probe.encoder_at)."""
    from torchvision import transforms

    from utils.dataset import _IMAGENET_MEAN, _IMAGENET_STD

    size = DINO_CFG["imgsz"]
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


class Backbone:
    """A timm backbone as a utils.oneshot.Encoder."""

    def __init__(self, theta: dict | None, device) -> None:
        self.model = build_model(0, device).backbone
        if theta is not None:
            self.model.load_state_dict(theta)
        self.model.eval()

    @torch.inference_mode()
    def embed(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)


# --------------------------------------------------------------------------- #
# scoring sets
# --------------------------------------------------------------------------- #


def load_sets(cfg) -> dict:
    """Every image a model is scored on (PIL), with labels; read once."""
    world = probe.load_world(cfg)
    crops = probe.pick_crops(world)
    groups = world["groups"]
    p1val = [c for c in crops if c["set"] == "p1val"]
    centers = [c for c in crops if c["set"] == "p1" and c["role"] == "center"]
    px = probe.cut_crops(p1val + centers)
    images, _, chosen = vocab_templates()
    names = [n for n in images if n != DROPPED]
    svg = load_svg_templates(
        ROOT / "data/raw/template_iso_svg", ROOT / "docs/symbol_mapping/iso_taxonomy.csv", "all"
    )
    cls_b, svg_b, png_b = FORM_B
    templates = [(n, images[n], axes_of(chosen[n][0])) for n in names]
    templates.append((cls_b, Image.open(png_b).convert("RGB"), axes_of(svg[svg_b])))

    root = ROOT / "data/iso_p4val"
    coco = json.loads((root / "canonical/annotations.coco.json").read_text())
    cat = {c["id"]: c["name"] for c in coco["categories"]}
    by_image = defaultdict(list)
    for a in coco["annotations"]:
        if not a.get("iscrowd") and cat[a["category_id"]] in names:
            by_image[a["image_id"]].append(a)
    info = {i["id"]: i for i in coco["images"]}
    p4_imgs, p4_rows = [], []
    for img_id, group in sorted(by_image.items()):
        im = info[img_id]
        arr = np.array(
            Image.open(root / "canonical/images" / Path(im["file_name"]).name).convert("RGB")
        )
        for a in group:
            x, y, w, h = a["bbox"]
            p4_imgs.append(Image.fromarray(np.ascontiguousarray(cut(arr, (x, y, x + w, y + h)))))
            p4_rows.append(
                {"sheet": im["sheet"], "ann_id": a["ann_id"], "class": cat[a["category_id"]]}
            )
    return {
        "names": names,
        "templates": templates,
        "center": [Image.fromarray(px[c["ann"]["id"]]) for c in centers],
        "p1val": [Image.fromarray(px[c["ann"]["id"]]) for c in p1val],
        "p1val_true": [groups[c["class"]].template or "" for c in p1val],
        "p1val_class": [c["class"] for c in p1val],
        "p1val_known": np.array([c["role"] == "known" for c in p1val]),
        "p4val": p4_imgs,
        "p4val_rows": pd.DataFrame(p4_rows),
        "viso": [images[n].convert("RGB") for n in names],  # V-ISO clean (§2K): the template
    }


def embed_sets(encoder, sets, tf, spec_q, spec_t, bg, device) -> dict:
    from viso_probe import embed_views

    a4 = probe.norm_fn("a4")
    t_imgs = [make_views(a4(img), spec_t, bg)[0] for _, img, _ in sets["templates"]]
    out = {"protos": embed_images(encoder, t_imgs, tf, device)}
    for key in ("center", "p1val", "viso", "p4val"):
        out[key] = embed_views(encoder, tf, sets[key], spec_q, bg, a4, device)
        logger.info("  embedded %s (%d)", key, len(sets[key]))
    out["center"] = out["center"].flatten(0, 1).mean(0)
    return out


def score(e: dict, sets: dict) -> dict:
    """2B path (k4_probe `41+B/mean`) on one model's embeddings."""
    names = sets["names"]
    owner = list(range(len(names))) + [names.index(FORM_B[0])]
    args = (e["protos"], e["center"], "mean", names, owner)
    s = class_scores(e["p4val"], *args)
    df = sets["p4val_rows"].copy()
    df["top1"] = [names[i] for i in s.argmax(1).tolist()]
    sp = class_scores(e["p1val"], *args)
    pred = [names[i] for i in sp.argmax(1).tolist()]
    known = sets["p1val_known"]
    hit = pd.Series([t == p for t, p in zip(sets["p1val_true"], pred, strict=True)])
    p1 = hit[known].groupby(np.array(sets["p1val_class"])[known]).mean()
    sv = class_scores(e["viso"], *args)
    viso = {n: float(names[i] == n) for n, i in zip(names, sv.argmax(1).tolist(), strict=True)}
    return {
        "p4val": df,
        "p1val_macro": float(p1.mean()),
        "p1val_auroc": auroc_known(confidence(e["p1val"], e["protos"], e["center"], "mean"), known),
        "viso_clean": viso,
    }


def per_class(df: pd.DataFrame, classes) -> dict[str, float]:
    per = (df.top1 == df["class"]).groupby(df["class"]).mean()
    return {c: float(per[c]) for c in classes}


# --------------------------------------------------------------------------- #


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--deploy", action="store_true", help="FT-triển khai: train on all 42")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cfg = OmegaConf.load(ROOT / "configs/config.yaml")
    cfg.data_root = str(ROOT / "data")
    spec_q = ViewSpec.from_cfg(cfg.oneshot.query_views)
    spec_t = ViewSpec.from_cfg(cfg.oneshot.template_views)
    bg = tuple(int(c) for c in cfg.oneshot.background)
    tf = transform()
    sets = load_sets(cfg)
    names = sets["names"]
    logger.info(
        "sets: %d templates, P4-val %d crops, P1-val %d",
        len(sets["templates"]),
        len(sets["p4val"]),
        len(sets["p1val"]),
    )

    def embedded(tag: str, theta: dict | None) -> dict:
        f = args.out / f"emb_{tag}.pt"
        if f.exists():
            return torch.load(f, weights_only=False)
        logger.info("embedding %s", tag)
        e = embed_sets(Backbone(theta, device), sets, tf, spec_q, spec_t, bg, device)
        torch.save(e, f)
        return e

    base_e = embedded("base", None)
    # clusters must be the ones §2L locked (from the baseline's own prototypes)
    p = F.normalize(base_e["protos"][: len(names)] - base_e["center"], dim=-1)
    found = components(names, p @ p.T)
    if found != sorted(sorted(v) for v in CLUSTERS.values()):
        raise SystemExit(f"clusters differ from §2L: {found}")
    base = score(base_e, sets)
    n = base["p4val"]["class"].value_counts()
    main_cls = sorted(c for c in n.index if n[c] >= MIN_CROPS)
    got = {"p4val": macro_top1(base["p4val"], main_cls), "p1val": base["p1val_macro"]}
    logger.info(
        "baseline: P4-val %.4f  P1-val %.4f (k4: %s)", got["p4val"], got["p1val"], BASE_CHECK
    )
    for k, v in BASE_CHECK.items():
        if abs(got[k] - v) > 2e-3:
            raise SystemExit(f"baseline {k} {got[k]:.4f} does not reproduce k4 ({v})")
    measured = {f: [c for c in held_out(f) if c in main_cls] for f in FOLDS}
    logger.info("measured held-out classes: %s", measured)
    if sum(len(v) for v in measured.values()) != 9:
        raise SystemExit(f"§2L measures 9 held-out classes, here {measured}")

    items = [(c, img, axes) for c, img, axes in sets["templates"]]
    if args.deploy:
        deploy(items, sets, base, main_cls, embedded, tf, spec_q, bg, device, args)
        return
    res = {"main_classes": main_cls, "measured": measured, "baseline": {}, "models": {}}
    scored = {}
    for fold in FOLDS:
        thetas = None
        for alpha in ALPHAS:
            tag = f"{fold}_a{alpha:g}"
            if not (args.out / f"emb_{tag}.pt").exists() and thetas is None:
                ck = args.out / f"train_{fold}.pt"
                if ck.exists():
                    thetas = torch.load(ck, weights_only=False)
                else:
                    t0, t1, cls = train(
                        train_items(items, fold), fold, tf, spec_q, bg, device, args.workers
                    )
                    thetas = {"theta0": t0, "theta1": t1, "classes": cls}
                    torch.save(thetas, ck)
            theta = None
            if thetas is not None:
                theta = wise(thetas["theta0"], thetas["theta1"], alpha)
                meta = ft_meta(fold, alpha, thetas["classes"])
                torch.save(checkpoint(theta, meta), args.out / f"__dinov2_ft_{tag}__.pt")
            scored[tag] = score(embedded(tag, theta), sets)

    def summary(s):
        return {
            "p4val_macro": macro_top1(s["p4val"], main_cls),
            "p4val_per_class": per_class(s["p4val"], main_cls),
            "p1val_macro": s["p1val_macro"],
            "p1val_auroc": s["p1val_auroc"],
            "viso_clean": s["viso_clean"],
        }

    res["baseline"] = summary(base)
    res["models"] = {t: summary(s) for t, s in scored.items()}
    cls9 = sorted(c for v in measured.values() for c in v)
    for alpha in ALPHAS:
        frames = {f: scored[f"{f}_a{alpha:g}"]["p4val"] for f in FOLDS}
        cand = assemble(frames, measured)
        ref = assemble({f: base["p4val"] for f in FOLDS}, measured)
        d_macro = macro_top1(cand, cls9) - macro_top1(ref, cls9)
        ci = boot_delta(ref, cand, cls9)
        b, c = per_class(ref, cls9), per_class(cand, cls9)
        p1 = {
            f: {
                "d_top1": scored[f"{f}_a{alpha:g}"]["p1val_macro"] - base["p1val_macro"],
                "d_auroc": scored[f"{f}_a{alpha:g}"]["p1val_auroc"] - base["p1val_auroc"],
            }
            for f in FOLDS
        }
        viso = {
            f: {
                k: scored[f"{f}_a{alpha:g}"]["viso_clean"][k] - base["viso_clean"][k]
                for k in held_out(f)
            }
            for f in FOLDS
        }
        d = decide(d_macro, ci, {k: c[k] - b[k] for k in cls9}, p1, viso)
        pairs = {}
        for a_, b_ in (
            ("valve_globe", "valve_butterfly"),
            ("valve_needle", "valve_three_way"),
            ("check_valve_general", "breather_valve"),
            ("valve_butterfly", "check_valve_swing"),
        ):
            sel_b, sel_c = ref[ref["class"] == a_], cand[cand["class"] == a_]
            pairs[f"{a_} -> {b_}"] = [
                float((sel_b.top1 == b_).mean()),
                float((sel_c.top1 == b_).mean()),
            ]
        res[f"alpha_{alpha:g}"] = {
            "macro_base": macro_top1(ref, cls9),
            "macro_ft": macro_top1(cand, cls9),
            "d_macro": d_macro,
            "ci95": list(ci),
            "per_class": {k: [b[k], c[k]] for k in cls9},
            "p1val": p1,
            "viso_clean": viso,
            "pairs": pairs,
            "decision": d,
        }
    res["decision"] = res[f"alpha_{ALPHA_MAIN:g}"]["decision"]
    (args.out / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    report(res, args.out)


def deploy(items, sets, base, main_cls, embedded, tf, spec_q, bg, device, args) -> None:
    """FT-triển khai: train on every template, WiSE alpha = ALPHA_MAIN, checkpoint + sha256, and a
    descriptive score (all classes trained: optimistic, decides nothing)."""
    ck = args.out / f"train_{DEPLOY}.pt"
    if ck.exists():
        thetas = torch.load(ck, weights_only=False)
    else:
        t0, t1, cls = train(train_items(items, None), DEPLOY, tf, spec_q, bg, device, args.workers)
        thetas = {"theta0": t0, "theta1": t1, "classes": cls}
        torch.save(thetas, ck)
    if len(thetas["classes"]) != len(sets["names"]):
        raise SystemExit(f"deploy must train all {len(sets['names'])} classes")
    tag = f"{DEPLOY}_a{ALPHA_MAIN:g}"
    theta = wise(thetas["theta0"], thetas["theta1"], ALPHA_MAIN)
    out_ck = args.out / f"__dinov2_ft_{tag}__.pt"
    torch.save(checkpoint(theta, ft_meta(DEPLOY, ALPHA_MAIN, thetas["classes"])), out_ck)
    digest = sha256(out_ck)
    (args.out / f"{out_ck.name}.sha256").write_text(f"{digest}  {out_ck.name}\n")
    s = score(embedded(tag, theta), sets)
    res = {
        "checkpoint": out_ck.name,
        "sha256": digest,
        "train_classes": thetas["classes"],
        "baseline": {
            "p4val_macro": macro_top1(base["p4val"], main_cls),
            "p4val_per_class": per_class(base["p4val"], main_cls),
            "p1val_macro": base["p1val_macro"],
            "p1val_auroc": base["p1val_auroc"],
            "viso_clean": sum(base["viso_clean"].values()),
        },
        "deploy": {
            "p4val_macro": macro_top1(s["p4val"], main_cls),
            "p4val_per_class": per_class(s["p4val"], main_cls),
            "p1val_macro": s["p1val_macro"],
            "p1val_auroc": s["p1val_auroc"],
            "viso_clean": sum(s["viso_clean"].values()),
            "viso_lost": sorted(k for k, v in s["viso_clean"].items() if v < 1),
        },
    }
    (args.out / "deploy.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    b, d = res["baseline"], res["deploy"]
    lines = [
        "§2L FT-triển khai (42 ảnh mẫu, α = 0.5) — MÔ TẢ, lạc quan (mọi lớp đã train)",
        f"checkpoint {out_ck.name}  sha256 {digest}",
        f"P4-val 18 lớp: {b['p4val_macro']:.3f} -> {d['p4val_macro']:.3f}",
        f"P1-val top-1 {b['p1val_macro']:.3f} -> {d['p1val_macro']:.3f};"
        f" AUROC {b['p1val_auroc']:.3f} -> {d['p1val_auroc']:.3f}",
        f"V-ISO sạch: {b['viso_clean']:.0f} -> {d['viso_clean']:.0f} / {len(sets['names'])}"
        f" (văng: {d['viso_lost'] or 'không'})",
    ]
    lines += [
        f"  {k:28s} {b['p4val_per_class'][k]:.2f} -> {d['p4val_per_class'][k]:.2f}"
        for k in main_cls
    ]
    text = "\n".join(lines)
    (args.out / "deploy_report.txt").write_text(text + "\n")
    print(text)


def report(r: dict, out: Path) -> None:
    lines = [
        "§2L FT-chính: fine-tune DINOv2-S trên ảnh mẫu, giữ cụm ra (K2 | K3+K4+K5), chấm P4-val",
        f"mốc formB (đường 2B): P4-val {r['baseline']['p4val_macro']:.3f},"
        f" P1-val {r['baseline']['p1val_macro']:.3f} / AUROC {r['baseline']['p1val_auroc']:.3f}",
        "",
    ]
    for alpha in ALPHAS:
        x = r[f"alpha_{alpha:g}"]
        d = x["decision"]
        tag = "QUYẾT ĐỊNH" if alpha == ALPHA_MAIN else "ablation, không chọn"
        lines += [
            f"===== WiSE α = {alpha:g} ({tag}) -> {'ĐẠT' if d['pass'] else 'KHÔNG ĐẠT'}",
            f"9 lớp giữ ra: {x['macro_base']:.3f} -> {x['macro_ft']:.3f}  Δ {x['d_macro']:+.3f}"
            f"  CI95 theo tờ [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]  (cần ≥ +{GAIN}, CI > 0)",
        ]
        lines += [f"  {k:28s} {v[0]:.2f} -> {v[1]:.2f}" for k, v in x["per_class"].items()]
        for f, p in x["p1val"].items():
            lines.append(
                f"P1-val ({f}): Δ top-1 {p['d_top1']:+.3f}, Δ AUROC {p['d_auroc']:+.3f}"
                f" (mỗi cái ≥ −{P1_DROP})"
            )
        for f, v in x["viso_clean"].items():
            bad = [k for k, dv in v.items() if dv < -VISO_DROP]
            lines.append(f"V-ISO sạch ({f}): lớp giữ ra bị văng: {bad or 'không'}")
        lines.append(
            "Cặp nhầm (mốc -> FT): "
            + "; ".join(f"{k} {a:.2f} -> {b:.2f}" for k, (a, b) in x["pairs"].items())
        )
        lines.append(
            "Điều kiện: " + ", ".join(f"{k} {'✓' if v else '✗'}" for k, v in d["checks"].items())
        )
        lines.append("")
    lines.append("Lớp đã train (lạc quan, mô tả) — P4-val macro 18 lớp / P1-val từng model:")
    for t, m in r["models"].items():
        lines.append(
            f"  {t:12s} P4-val {m['p4val_macro']:.3f}  P1-val {m['p1val_macro']:.3f}"
            f"  AUROC {m['p1val_auroc']:.3f}"
        )
    text = "\n".join(lines)
    (out / "report.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
