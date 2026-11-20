"""PLAN_PHASE_C §2K.2: score 2B on P4-val and check the "P4-val is representative of P4" gate.

P4-val = the OPEN100-ISO renderer on 37 other PID2Graph skeletons (data/iso_p4val, degraded
images, made by p4val_prepare.py + make_variants.py + make_degraded.py + pipeline.render_iso).
Crops are cut from the COCO boxes as P4's (utils.crop_box.cut), then 2B exactly as viso_probe:
A4 -> 8 query views (mean) -> DINOv2-S 224 -> CL2N (probe center) -> 41 one-view prototypes.
Known = a class among the 41 templates; other_* = unknown. Writes a per-crop CSV (P4 columns plus
`template`, the drawn SVG, from the V7 json) for later decisions.

    python scripts/p4val_probe.py --out runs/phase_c/p4val [--device mps]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import oneshot_probe as probe  # noqa: E402
from viso_probe import embed_views, encoder_at, spearman  # noqa: E402

from utils.crop_box import cut  # noqa: E402
from utils.oneshot import ViewSpec  # noqa: E402

logger = logging.getLogger("p4val")

DROPPED = "pipeline"  # §2G
MIN_CROPS, MIN_COMMON = 20, 15
GATE_RHO, GATE_GB = 0.5, 0.20


def gate(val_per: dict, val_n: dict, p4_per: dict, globe_to_butterfly: float) -> dict:
    """§2K.2: G0 >= 15 common classes (>= 20 P4-val crops), G1 Spearman >= 0.5, G2 >= 0.20."""
    common = sorted(c for c in val_per if val_n.get(c, 0) >= MIN_CROPS and c in p4_per)
    rho = (
        spearman([val_per[c] for c in common], [p4_per[c] for c in common])
        if len(common) >= 3
        else float("nan")
    )
    ok = len(common) >= MIN_COMMON and rho >= GATE_RHO and globe_to_butterfly >= GATE_GB
    return {
        "n_common": len(common),
        "common": common,
        "rho": rho,
        "globe_to_butterfly": globe_to_butterfly,
        "pass": bool(ok),
    }


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score of a known > score of an unknown), ties half."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    s = np.concatenate([pos, neg])
    r = pd.Series(s).rank().to_numpy()
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=ROOT / "data/iso_p4val")
    ap.add_argument("--probe", type=Path, default=ROOT / "runs/phase_c/probe")
    ap.add_argument("--p4", type=Path, default=ROOT / "runs/oneshot_p4_crops_dinov2.csv")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    cfg = OmegaConf.load(ROOT / "configs/config.yaml")
    cfg.data_root = str(ROOT / "data")
    world = probe.load_world(cfg)
    centers = [c for c in probe.pick_crops(world) if c["set"] == "p1" and c["role"] == "center"]
    spec_q = ViewSpec.from_cfg(cfg.oneshot.query_views)
    bg = tuple(int(c) for c in cfg.oneshot.background)
    a4 = probe.norm_fn("a4")
    cache = torch.load(args.probe / "emb_dinov2_a4.pt", weights_only=False)
    keep = [i for i, n in enumerate(cache["names"]) if n != DROPPED]
    names = [cache["names"][i] for i in keep]
    protos = cache["protos"][keep]
    center = torch.cat([cache["emb"][c["ann"]["id"]] for c in centers]).mean(0)

    coco = json.loads((args.root / "canonical/annotations.coco.json").read_text())
    cat = {c["id"]: c["name"] for c in coco["categories"]}
    images = {i["id"]: i for i in coco["images"]}
    drawn = {}  # (sheet, variant, node) -> template drawn, from the V7 json
    for jf in (args.root / "degraded").glob("sheet*/v*.json"):
        doc = json.loads(jf.read_text())
        for a in doc["annotations"]:
            drawn[(int(doc["sheet"]), int(doc["index"]), a["node"])] = a.get("template", "")
    anns = [a for a in coco["annotations"] if not a.get("iscrowd")]
    by_image = defaultdict(list)
    for a in anns:
        by_image[a["image_id"]].append(a)

    store = args.out / "emb.pt"
    if store.exists():
        st = torch.load(store, weights_only=False)
    else:
        encoder, tf = encoder_at(224, device)
        embs, rows = [], []
        for k, (img_id, group) in enumerate(sorted(by_image.items()), 1):
            info = images[img_id]
            px = np.array(
                Image.open(args.root / "canonical/images" / Path(info["file_name"]).name).convert(
                    "RGB"
                )
            )
            crops = []
            for a in group:
                x, y, w, h = a["bbox"]
                crops.append(Image.fromarray(np.ascontiguousarray(cut(px, (x, y, x + w, y + h)))))
                rows.append(
                    {
                        "image": Path(info["file_name"]).stem,
                        "sheet": info["sheet"],
                        "variant": info["variant"],
                        "ann_id": a["ann_id"],
                        "class": cat[a["category_id"]],
                        "template": drawn.get((info["sheet"], info["variant"], a["node"]), ""),
                    }
                )
            embs.append(embed_views(encoder, tf, crops, spec_q, bg, a4, device))
            if k % 10 == 0:
                logger.info("embedded %d / %d images", k, len(by_image))
        st = {"names": names, "rows": rows, "emb": torch.cat(embs)}
        torch.save(st, store)
    assert st["names"] == names

    idx, val = probe.rank(st["emb"], protos, center)
    df = pd.DataFrame(st["rows"])
    for j in range(3):
        df[f"top{j + 1}"] = [names[i] for i in idx[:, j].tolist()]
        df[f"cos{j + 1}"] = val[:, j].numpy()
    df["known"] = df["class"].isin(names)
    df.to_csv(args.out / "p4val_crops_dinov2.csv", index=False)

    kn = df[df.known]
    per = (kn.top1 == kn["class"]).groupby(kn["class"]).mean().to_dict()
    n = kn["class"].value_counts().to_dict()
    gb = kn[kn["class"] == "valve_globe"]
    gb_rate = float((gb.top1 == "valve_butterfly").mean()) if len(gb) else float("nan")
    p4 = pd.read_csv(args.p4)
    p4 = p4[(p4.split == "degraded") & p4.expected.isin(names)]
    p4_per = (p4.top1 == p4.expected).groupby(p4.expected).mean().to_dict()
    g = gate(per, n, p4_per, gb_rate)
    margin = (df.cos1 - df.cos2).to_numpy()
    res = {
        "n_images": int(df.image.nunique()),
        "n_sheets": int(df.sheet.nunique()),
        "n_known": int(len(kn)),
        "n_unknown": int((~df.known).sum()),
        "macro_top1": float(np.mean([per[c] for c in per if n[c] >= MIN_CROPS])),
        "macro_top3": float(
            np.mean(
                [
                    ((kn[kn["class"] == c][["top1", "top2", "top3"]].values == c).any(1)).mean()
                    for c in per
                    if n[c] >= MIN_CROPS
                ]
            )
        ),
        "auroc_cos": auroc(df.cos1[df.known].to_numpy(), df.cos1[~df.known].to_numpy()),
        "auroc_margin": auroc(margin[df.known.to_numpy()], margin[~df.known.to_numpy()]),
        "per_class": {c: {"top1": per[c], "n": n[c], "p4": p4_per.get(c)} for c in sorted(per)},
        "gate": g,
        "butterfly_by_template": {
            t: {"n": int(len(x)), "top1": float((x.top1 == "valve_butterfly").mean())}
            for t, x in kn[kn["class"] == "valve_butterfly"].groupby("template")
        },
        "top_confusions": (
            (kn[kn.top1 != kn["class"]]["class"] + " -> " + kn[kn.top1 != kn["class"]].top1)
            .value_counts()
            .head(15)
            .to_dict()
        ),
    }
    (args.out / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    report(res, args.out)


def report(r: dict, out: Path) -> None:
    g = r["gate"]
    lines = [
        f"P4-val (§2K.2): {r['n_sheets']} tờ, {r['n_images']} ảnh suy giảm,"
        f" {r['n_known']} crop biết, {r['n_unknown']} crop lạ; 2B, 41 ảnh mẫu",
        f"macro top-1 {r['macro_top1']:.3f}, top-3 {r['macro_top3']:.3f} (lớp ≥ {MIN_CROPS} crop);"
        f" AUROC cos {r['auroc_cos']:.3f}, lề {r['auroc_margin']:.3f}",
        "",
        f"Cổng: G0 {g['n_common']} lớp chung (cần ≥ {MIN_COMMON}); G1 Spearman ρ {g['rho']:.3f}"
        f" (cần ≥ {GATE_RHO}); G2 globe → butterfly {g['globe_to_butterfly']:.3f} (cần ≥ {GATE_GB})"
        f" -> {'QUA' if g['pass'] else 'TRƯỢT'}",
        "",
        f"{'lớp':28s} {'n':>5s} {'P4-val':>7s} {'P4':>6s}",
    ]
    for c, s in sorted(r["per_class"].items(), key=lambda kv: -kv[1]["n"]):
        p4 = "—" if s["p4"] is None else f"{s['p4']:.2f}"
        lines.append(f"{c:28s} {s['n']:5d} {s['top1']:7.2f} {p4:>6s}")
    lines += ["", "valve_butterfly theo dạng hình được vẽ (template):"]
    for t, s in r["butterfly_by_template"].items():
        lines.append(f"  {t or '?':12s} n {s['n']:4d}  top-1 {s['top1']:.2f}")
    lines += ["", "Cặp nhầm nhiều nhất:"]
    lines += [f"  {k:52s} {v}" for k, v in r["top_confusions"].items()]
    text = "\n".join(lines)
    (out / "report.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
