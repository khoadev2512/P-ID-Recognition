"""PLAN_PHASE_C §2F.1: a template directory that mixes styles (ISO templates + drawing-style crops).

Descriptive, no decision. Reads only the Bước 1 probe cache (DINOv2-S / A4, CL2N with the probe's
center); drawing-style exemplars = the probe's 1e exemplars (`ex<d>`, one train crop per class, five
draws), named by the class' ISO meaning; queries = the P1-val crops (val_sel sheets). Directories:

    D0  the 42 ISO templates
    D1  D0 + exemplars of the 6 VARIANT classes
    D2  D0 + exemplars of every class (except h_out and the 7 P1-val unknowns)
    D3  those exemplars alone ("legend" mode)

    python scripts/mixed_directory.py --out runs/phase_c/mixed
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import oneshot_probe as probe  # noqa: E402

ISO_KNOWN = ["Symbol_1", "Symbol_17", "Symbol_21", "Symbol_24"]
VARIANT_KNOWN = ["Symbol_6", "Symbol_8", "Symbol_14", "Symbol_20", "Symbol_22", "Symbol_23"]
DRAWS = (0, 1, 2, 3, 4)
DIRECTORIES = ("D0", "D1", "D2", "D3")


def build_directory(
    kind: str,
    iso: torch.Tensor,
    iso_names: list[str],
    exemplars: dict[str, torch.Tensor],
    meaning: dict[str, str],
    variant: list[str],
) -> tuple[torch.Tensor, list[str]]:
    """(N, D) raw prototypes and their names for directory `kind`. `exemplars`: class -> raw
    embedding of its drawing-style crop (already restricted to the allowed classes); `meaning`:
    class -> ISO template name (or the class name when it has none)."""
    if kind == "D0":
        return iso, list(iso_names)
    if kind == "D1":
        keep = [k for k in variant if k in exemplars]
    elif kind in ("D2", "D3"):
        keep = sorted(exemplars)
    else:
        raise ValueError(f"unknown directory {kind!r}")
    ex = torch.stack([exemplars[k] for k in keep])
    names = [meaning[k] for k in keep]
    if kind == "D3":
        return ex, names
    return torch.cat([iso, ex]), list(iso_names) + names


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--probe", type=Path, default=ROOT / "runs/phase_c/probe")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(ROOT / "configs/config.yaml")
    cfg.data_root = str(ROOT / "data")
    world = probe.load_world(cfg)
    crops = probe.pick_crops(world)
    cache = torch.load(args.probe / "emb_dinov2_a4.pt", weights_only=False)
    emb, iso, iso_names = cache["emb"], cache["protos"], cache["names"]
    groups = world["groups"]
    center = torch.cat(
        [emb[c["ann"]["id"]] for c in crops if c["set"] == "p1" and c["role"] == "center"]
    ).mean(0)
    meaning = lambda k: groups[k].template or k  # noqa: E731
    unknown_cls = set(probe.P1VAL_UNKNOWN)
    p1val = [c for c in crops if c["set"] == "p1val"]
    val_stems = {c["stem"] for c in p1val}
    assert set(probe.P1VAL_KNOWN) == set(ISO_KNOWN + VARIANT_KNOWN)

    results: dict = {}
    for kind in DIRECTORIES:
        draws = []
        for d in DRAWS if kind != "D0" else (None,):
            ex = {}
            if d is not None:
                for c in crops:
                    if c["set"] == f"ex{d}" and c["class"] not in unknown_cls:
                        assert c["stem"] not in val_stems
                        ex[c["class"]] = F.normalize(emb[c["ann"]["id"]].mean(0), dim=-1)
            mean_of = {k: meaning(k) for k in ex}
            protos, names = build_directory(kind, iso, iso_names, ex, mean_of, VARIANT_KNOWN)
            s = probe.summarize(probe.rows_for(p1val, emb, protos, names, center, meaning, "known"))
            pc = s["per_class"]
            draws.append(
                {
                    "n_protos": len(names),
                    "iso": float(np.mean([pc[k]["top1"] for k in ISO_KNOWN])),
                    "variant": float(np.mean([pc[k]["top1"] for k in VARIANT_KNOWN])),
                    "all": s["macro_top1"],
                    "auroc_cos": s["auroc_cos"],
                    "per_class": {k: pc[k]["top1"] for k in ISO_KNOWN + VARIANT_KNOWN},
                    "most_predicted": {
                        k: pc[k]["most_predicted"] for k in ISO_KNOWN + VARIANT_KNOWN
                    },
                }
            )
        agg = {
            key: {
                "mean": float(np.mean([x[key] for x in draws])),
                "std": float(np.std([x[key] for x in draws], ddof=1)) if len(draws) > 1 else 0.0,
            }
            for key in ("iso", "variant", "all", "auroc_cos")
        }
        agg["per_class"] = {
            k: float(np.mean([x["per_class"][k] for x in draws])) for k in draws[0]["per_class"]
        }
        agg["most_predicted"] = {
            k: Counter(x["most_predicted"][k] for x in draws).most_common(1)[0][0]
            for k in draws[0]["most_predicted"]
        }
        agg["n_protos"] = draws[0]["n_protos"]
        results[kind] = {"summary": agg, "draws": draws}
    (args.out / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    report(results, args.out)


def report(r: dict, out: Path) -> None:
    label = {
        "D0": "D0 chỉ ISO",
        "D1": "D1 ISO + 6 VARIANT",
        "D2": "D2 ISO + 21 bản vẽ",
        "D3": "D3 chỉ 21 bản vẽ",
    }
    pm = lambda x: f"{x['mean']:.3f} ± {x['std']:.3f}"  # noqa: E731
    lines = [
        "Thư mục trộn phong cách (PLAN_PHASE_C §2F.1), DINOv2-S / A4 / CL2N,"
        " truy vấn P1-val, 5 lần rút",
        "",
        f"{'thư mục':22s} {'#proto':>6s} {'ISO(4)':>15s} {'VARIANT(6)':>15s} {'cả 10':>15s}"
        f" {'AUROC cos':>15s}",
    ]
    for k in DIRECTORIES:
        s = r[k]["summary"]
        lines.append(
            f"{label[k]:22s} {s['n_protos']:6d} {pm(s['iso']):>15s} {pm(s['variant']):>15s}"
            f" {pm(s['all']):>15s} {pm(s['auroc_cos']):>15s}"
        )
    lines += ["", "Từng lớp top-1 (TB 5 lần rút) [tên bị đoán nhiều nhất]:"]
    for cls in ISO_KNOWN + VARIANT_KNOWN:
        tag = "ISO" if cls in ISO_KNOWN else "VAR"
        sums = [r[k]["summary"] for k in DIRECTORIES]
        cells = "  ".join(
            f"{s['per_class'][cls]:.2f} [{s['most_predicted'][cls][:16]}]" for s in sums
        )
        lines.append(f"  {cls:10s} {tag}  {cells}")
    text = "\n".join(lines)
    (out / "report.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
