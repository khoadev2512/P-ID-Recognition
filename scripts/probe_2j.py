"""PLAN_PHASE_C §2J: the three untested left-overs of the "4 directions" proposal, on P1-val.

DINOv2-S/14 frozen, A4, 41 templates (§2G), 1 template view, 8 query views; tokens from
`forward_intermediates(..., norm=True)`; ink mask and patch center exactly as §2H (patch_probe).

    oneway/q2t, oneway/t2q   layer 12, patch-centered; ONE side of the §2H Chamfer
    inkmean/raw, /cl2n       mean of the layer-12 ink-patch tokens, unit per view, probe `rank`
    cat4/raw, /cl2n          CLS of blocks 9-12 concatenated (1536-d), unit per view, probe `rank`

Reproduction gates before scoring: layer-12 CLS of the templates == probe prototypes (< 5e-3);
(q2t + t2q) / 2 == the §2H L12/center scores (< 1e-3). The best by P1-val macro top-1 goes
against the 2B global baseline under the §1.2 rule (oneshot_probe.decide).

    python scripts/probe_2j.py --out runs/phase_c/j [--device mps]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import oneshot_probe as probe  # noqa: E402
from patch_probe import ink_grid, prep_desc, rows_from_scores  # noqa: E402

from pipeline.iso_prototypes import load_templates, read_vocab  # noqa: E402
from utils.oneshot import ViewSpec, make_views  # noqa: E402

logger = logging.getLogger("probe_2j")

DROPPED = "pipeline"  # §2G
BLOCKS = (9, 10, 11, 12)  # 1-based
VARIANTS = [  # tie -> this order
    "oneway/q2t",
    "oneway/t2q",
    "inkmean/raw",
    "inkmean/cl2n",
    "cat4/raw",
    "cat4/cl2n",
]
TOL_PROTO, TOL_CHAMFER = 5e-3, 1e-3


# --------------------------------------------------------------------------- #
# pure functions (tests/test_probe_2j.py)
# --------------------------------------------------------------------------- #


def directional(
    q: torch.Tensor, qmask: torch.Tensor, t: torch.Tensor, tmask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """q (V, P, D), qmask (V, P); t (T, P', D), tmask (T, P') unit descriptors -> (q2t, t2q),
    each (V, T): mean over one side's ink patches of the best cosine on the other side's ink
    patches. (q2t + t2q) / 2 is patch_probe.chamfer."""
    sim = torch.einsum("vqd,tpd->vtqp", q, t)
    neg = torch.finfo(sim.dtype).min
    qm, tm = qmask[:, None, :, None], tmask[None, :, None, :]
    q_best = sim.masked_fill(~tm, neg).amax(dim=3)  # (V, T, Q)
    t_best = sim.masked_fill(~qm, neg).amax(dim=2)  # (V, T, P)
    qw, tw = qmask[:, None, :].float(), tmask[None, :, :].float()
    return (q_best * qw).sum(-1) / qw.sum(-1), (t_best * tw).sum(-1) / tw.sum(-1)


def masked_mean(tok: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """tok (n, P, D), mask (n, P) -> (n, D) unit mean of the masked tokens."""
    w = mask.float()[..., None]
    return F.normalize((tok * w).sum(1) / w.sum(1), dim=-1)


def concat_cls(cls: torch.Tensor) -> torch.Tensor:
    """cls (n, B, D) per-block CLS -> (n, B*D) unit concatenation."""
    return F.normalize(cls.flatten(1), dim=-1)


def pick_best(summaries: dict[str, dict]) -> str:
    return max(VARIANTS, key=lambda k: (summaries[k]["macro_top1"], -VARIANTS.index(k)))


# --------------------------------------------------------------------------- #
# embedding
# --------------------------------------------------------------------------- #


@torch.inference_mode()
def features(model, tf, images: list[Image.Image], device) -> dict[str, torch.Tensor]:
    """-> tok (n, 256, D) layer-12 patch tokens, cls (n, 4, D) CLS of blocks 9-12, mask (n, 256)."""
    x = torch.stack([tf(im.convert("RGB")) for im in images]).to(device)
    out = model.forward_intermediates(
        x,
        indices=[b - 1 for b in BLOCKS],
        return_prefix_tokens=True,
        norm=True,
        output_fmt="NLC",
        intermediates_only=True,
    )
    return {
        "tok": out[-1][0].float().cpu(),
        "cls": torch.stack([o[1][:, 0] for o in out], dim=1).float().cpu(),
        "mask": torch.stack([ink_grid(im) for im in images]),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--probe", type=Path, default=ROOT / "runs/phase_c/probe")
    ap.add_argument("--patch", type=Path, default=ROOT / "runs/phase_c/patch/scores.pt")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    cfg = OmegaConf.load(ROOT / "configs/config.yaml")
    cfg.data_root = str(ROOT / "data")
    world = probe.load_world(cfg)
    crops = probe.pick_crops(world)
    groups = world["groups"]
    expected_of = lambda n: groups[n].template or ""  # noqa: E731
    p1val = [c for c in crops if c["set"] == "p1val"]
    centers = [c for c in crops if c["set"] == "p1" and c["role"] == "center"]
    spec_q = ViewSpec.from_cfg(cfg.oneshot.query_views)
    spec_t = ViewSpec.from_cfg(cfg.oneshot.template_views)
    bg = tuple(int(c) for c in cfg.oneshot.background)
    a4 = probe.norm_fn("a4")
    n_v = spec_q.n_views

    cache = torch.load(args.probe / "emb_dinov2_a4.pt", weights_only=False)
    keep = [i for i, n in enumerate(cache["names"]) if n != DROPPED]
    names = [cache["names"][i] for i in keep]
    emb = cache["emb"]
    gcenter = torch.cat([emb[c["ann"]["id"]] for c in centers]).mean(0)
    base = probe.summarize(
        probe.rows_for(p1val, emb, cache["protos"][keep], names, gcenter, expected_of, "known")
    )
    logger.info("baseline 2B (41): top-1 %.3f AUROC %.3f", base["macro_top1"], base["auroc_cos"])
    ids = [c["ann"]["id"] for c in p1val]

    store = args.out / "scores.pt"
    if store.exists():
        st = torch.load(store, weights_only=False)
    else:
        encoder, tf = probe.load_encoder("dinov2", cfg, device)
        model = encoder.model
        rows = read_vocab(ROOT / "data/raw/iso_symbols/vocab.csv")
        templates, _ = load_templates(ROOT / "data/raw/iso_symbols_clean", rows)
        t_imgs = [make_views(a4(templates[n]), spec_t, bg)[0] for n in names]
        tf_t = features(model, tf, t_imgs, device)
        proto_diff = float(
            (F.normalize(tf_t["cls"][:, -1], dim=-1) - cache["protos"][keep]).abs().max()
        )
        logger.info("gate (i): template CLS-12 vs probe prototypes max |diff| %.2e", proto_diff)
        if proto_diff >= TOL_PROTO:
            raise SystemExit(f"gate (i) failed: {proto_diff:.2e} >= {TOL_PROTO}")
        pixels = probe.cut_crops(p1val + centers)

        def views_of(c):
            return make_views(a4(Image.fromarray(pixels[c["ann"]["id"]])), spec_q, bg)

        # pass 1: center crops -> patch center (as §2H) and global view vectors (for CL2N)
        g_emb = {"inkmean": {}, "cat4": {}}
        psum, count = 0.0, 0
        for k in range(0, len(centers), 16):
            part = centers[k : k + 16]
            f = features(model, tf, [v for c in part for v in views_of(c)], device)
            psum = psum + f["tok"][f["mask"]].sum(0)
            count += int(f["mask"].sum())
            im, c4 = masked_mean(f["tok"], f["mask"]), concat_cls(f["cls"])
            for j, c in enumerate(part):
                g_emb["inkmean"][c["ann"]["id"]] = im[j * n_v : (j + 1) * n_v]
                g_emb["cat4"][c["ann"]["id"]] = c4[j * n_v : (j + 1) * n_v]
        pcenter = psum / count
        logger.info("patch center from %d ink patches of %d crops", count, len(centers))

        # pass 2: P1-val
        t_desc = prep_desc(tf_t["tok"], pcenter).to(device)
        t_mask = tf_t["mask"].to(device)
        q2t = torch.empty(len(p1val), len(names))
        t2q = torch.empty(len(p1val), len(names))
        for k in range(0, len(p1val), 8):
            part = p1val[k : k + 8]
            f = features(model, tf, [v for c in part for v in views_of(c)], device)
            q = prep_desc(f["tok"], pcenter).to(device)
            m = f["mask"].to(device)
            im, c4 = masked_mean(f["tok"], f["mask"]), concat_cls(f["cls"])
            for j, c in enumerate(part):
                sl = slice(j * n_v, (j + 1) * n_v)
                a, b = directional(q[sl], m[sl], t_desc, t_mask)
                q2t[k + j], t2q[k + j] = a.mean(0).cpu(), b.mean(0).cpu()
                g_emb["inkmean"][c["ann"]["id"]] = im[sl]
                g_emb["cat4"][c["ann"]["id"]] = c4[sl]
            if (k // 8) % 25 == 0:
                logger.info("scored %d / %d P1-val crops", k + len(part), len(p1val))
        st = {
            "names": names,
            "ann_ids": ids,
            "proto_diff": proto_diff,
            "q2t": q2t,
            "t2q": t2q,
            "g_emb": g_emb,
            "g_protos": {
                "inkmean": masked_mean(tf_t["tok"], tf_t["mask"]),
                "cat4": concat_cls(tf_t["cls"]),
            },
        }
        torch.save(st, store)
    assert st["names"] == names and st["ann_ids"] == ids

    # gate (ii): the two halves rebuild the §2H symmetric Chamfer
    patch = torch.load(args.patch, weights_only=False)
    assert patch["names"] == names and patch["ann_ids"] == ids
    chamfer_diff = float(
        (0.5 * (st["q2t"] + st["t2q"]) - patch["scores"]["L12/center"]).abs().max()
    )
    logger.info("gate (ii): (q2t + t2q) / 2 vs §2H L12/center max |diff| %.2e", chamfer_diff)
    if chamfer_diff >= TOL_CHAMFER:
        raise SystemExit(f"gate (ii) failed: {chamfer_diff:.2e} >= {TOL_CHAMFER}")

    summ = {
        "oneway/q2t": probe.summarize(rows_from_scores(p1val, st["q2t"], names, expected_of)),
        "oneway/t2q": probe.summarize(rows_from_scores(p1val, st["t2q"], names, expected_of)),
    }
    for kind in ("inkmean", "cat4"):
        e = st["g_emb"][kind]
        center = torch.cat([e[c["ann"]["id"]] for c in centers]).mean(0)
        for tag, cen in (("raw", None), ("cl2n", center)):
            summ[f"{kind}/{tag}"] = probe.summarize(
                probe.rows_for(p1val, e, st["g_protos"][kind], names, cen, expected_of, "known")
            )
    best = pick_best(summ)
    results = {
        "baseline_2b_41": base,
        "gates": {"proto_diff": st["proto_diff"], "chamfer_diff": chamfer_diff},
        "variants": {k: summ[k] for k in VARIANTS},
        "best": best,
        "decision": probe.decide(base, summ[best]),
    }
    (args.out / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    report(results, args.out)


def report(r: dict, out: Path) -> None:
    f = probe._fmt
    g = r["gates"]
    lines = [
        "§2J: ba phần còn sót (DINOv2-S / A4, 41 ảnh mẫu, P1-val)",
        f"kiểm tái lập: CLS-12 vs prototype {g['proto_diff']:.1e};"
        f" ½(q2t + t2q) vs A L12/center {g['chamfer_diff']:.1e}",
        "",
        f"{'dòng':26s} {'top-1':>6s} {'top-3':>6s} {'AUROC':>6s} {'lề':>6s} {'hub':>6s}",
    ]
    rows = [("mốc 2B toàn ảnh (CL2N)", r["baseline_2b_41"])]
    rows += [(k + (" ← dự thi" if k == r["best"] else ""), s) for k, s in r["variants"].items()]
    for label, s in rows:
        lines.append(
            f"{label:26s} {f(s['macro_top1'])} {f(s['macro_top3'])} {f(s['auroc_cos'])}"
            f" {f(s['auroc_margin'])} {f(s['hub_share'])}"
        )
    d = r["decision"]
    lines += [
        "",
        f"Luật §1.2 ({r['best']} vs mốc): {d['n_up']}/10 lớp tăng, Δtop-1 {d['d_macro_top1']:+.3f},"
        f" ΔAUROC {d['d_auroc']:+.3f} -> {'ĐẠT' if d['pass'] else 'KHÔNG ĐẠT'}",
        "",
        "Từng lớp top-1 (mốc -> dự thi) [đoán nhiều nhất của dự thi]:",
    ]
    b, c = r["baseline_2b_41"]["per_class"], r["variants"][r["best"]]["per_class"]
    for k in b:
        lines.append(
            f"  {k:10s} {b[k]['top1']:.3f} -> {c[k]['top1']:.3f}  [{c[k]['most_predicted']}]"
        )
    text = "\n".join(lines)
    (out / "report.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
