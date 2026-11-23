"""Draw one OPEN100-ISO sheet with the end-to-end output (Stage 1 + one-shot Stage 2) coloured by
the SAME matching as the evaluator (pipeline.detection_eval.e2e_block): IoU >= 0.5 and the right
label = correct; a box on a GT symbol with the wrong label (or "Other" on a vocabulary symbol) =
wrong name; a box on no GT symbol = extra box; a GT symbol no prediction covers = missed.
Predictions inside ignore regions are not drawn (they count as neither TP nor FP).

Input: the detections_oneshot files of notebook 17d (Drive
`runs/stage1_iso/detections_oneshot_dinov2_<tag>/<image_id>.txt`), at the tau of the run.

    python scripts/draw_e2e_sheet.py --det-dir runs/e2e_ft_det --image iso05_v000_deg \\
        --tau 0.1923 --out docs/figures/e2e_iso05.png [--check runs/e2e_dinov2_ft.json]

--check: classify every image of --det-dir and compare TP / FP of the degraded view with that
metrics copy (must match exactly, or the picture does not show what was scored).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline import detection_eval as de  # noqa: E402
from pipeline.oneshot_infer import UNREFINED_LABEL, read_oneshot_detections  # noqa: E402
from utils import bbox_utils, detection_metrics  # noqa: E402
from utils.classmap import OTHER_CLASS_NAME  # noqa: E402

DATA = ROOT / "data/iso_open100"
SCORE_THR = 0.25  # configs/config.yaml eval.score_thr
KINDS = ("correct", "wrong_name", "extra_box", "missed")


def classify(
    image_id: str,
    preds: list[de.E2EPred],
    gts: list[de.GtBox],
    ignores: list[de.IgnoreRegion],
    expected: dict[str, str | None],
    tau: float,
) -> dict:
    """Per prediction and per GT box of one image, its kind (KINDS) — e2e_block's matching."""
    scored = [g for g in gts if expected.get(g.name) is not None]
    answers_exp = [expected[g.name] for g in scored]
    regions = [(r.image_id, r.xyxy) for r in ignores] + [
        (g.image_id, g.xyxy) for g in gts if expected.get(g.name) is None
    ]
    gate = de.reject([p.best_cos for p in preds], [p.second_cos for p in preds], tau)
    answers = [OTHER_CLASS_NAME if r else p.best_name for p, r in zip(preds, gate, strict=True)]
    label_id = {a: k for k, a in enumerate(sorted(set(answers) | set(answers_exp)))}
    lab_preds = [
        (p.image_id, label_id[a], p.score, p.xyxy) for p, a in zip(preds, answers, strict=True)
    ]
    lab_gts = [(g.image_id, label_id[e], g.xyxy) for g, e in zip(scored, answers_exp, strict=True)]
    hits = detection_metrics.ignore_hits(lab_preds, regions)
    match = detection_metrics.greedy_match(lab_preds, lab_gts, de.E2E_IOU, hits)
    iou = (
        detection_metrics.iou_matrix([p.xyxy for p in preds], [g.xyxy for g in scored])
        if preds and scored
        else np.zeros((len(preds), len(scored)))
    )
    pred_kind, pred_truth = [], []
    for i in range(len(preds)):
        best = int(iou[i].argmax()) if len(scored) else -1
        truth = answers_exp[best] if best >= 0 and iou[i, best] >= de.E2E_IOU else None
        if match.tp[i]:
            kind = "correct"
        elif match.pred_ignored[i]:
            kind = "ignored"
        elif truth is not None:
            kind = "wrong_name"
        else:
            kind = "extra_box"
        pred_kind.append(kind)
        pred_truth.append(truth)
    live = np.array([k != "ignored" for k in pred_kind])
    covered = (iou[live] >= de.E2E_IOU).any(axis=0) if live.any() else np.zeros(len(scored), bool)
    gt_missed = ~match.gt_hit & ~covered
    # per GT symbol, a partition (one symbol may carry several boxes: right one + a duplicate)
    gt_kind = np.where(match.gt_hit, "correct", np.where(covered, "wrong_name", "missed"))
    return {
        "gt_kind": gt_kind.tolist(),
        "answers": answers,
        "pred_kind": pred_kind,
        "pred_truth": pred_truth,
        "scored": scored,
        "gt_missed": gt_missed,
        "tp": int(match.tp.sum()),
        "fp": int(match.fp.sum()),
    }


def load(det_dir: Path, image_ids: list[str]):
    coco = json.loads((DATA / "canonical/annotations.coco.json").read_text())
    info = {de._stem(i["file_name"]): i for i in coco["images"]}
    id_map = {i["id"]: de._stem(i["file_name"]) for i in coco["images"]}
    gts, ignores = de.load_boxes(coco, set(image_ids), id_map)
    cfg = OmegaConf.create({"oneshot": {"vocab_csv": str(ROOT / "data/raw/iso_symbols/vocab.csv")}})
    expected = de.gt_groups(cfg, {g.name for g in gts}, require_vocab=True).expected
    preds: dict[str, list[de.E2EPred]] = {}
    for image_id in image_ids:
        w, h = info[image_id]["width"], info[image_id]["height"]
        preds[image_id] = [
            de.E2EPred(
                image_id,
                r.best_name,
                r.score,
                r.best_cos,
                r.second_cos,
                bbox_utils.yolo_to_xyxy(r.box, w, h),
            )
            for r in read_oneshot_detections(det_dir / f"{image_id}.txt")
            if r.score >= SCORE_THR and r.label != UNREFINED_LABEL and np.isfinite(r.best_cos)
        ]
    return info, gts, ignores, expected, preds


def check(det_dir: Path, tau: float, metrics: Path) -> None:
    coco = json.loads((DATA / "canonical/annotations.coco.json").read_text())
    ids = [
        de._stem(i["file_name"])
        for i in coco["images"]
        if i.get("degraded") and (det_dir / f"{de._stem(i['file_name'])}.txt").exists()
    ]
    _, gts, ignores, expected, preds = load(det_dir, ids)
    tp = fp = 0
    for image_id in ids:
        c = classify(
            image_id,
            preds[image_id],
            [g for g in gts if g.image_id == image_id],
            [r for r in ignores if r.image_id == image_id],
            expected,
            tau,
        )
        tp, fp = tp + c["tp"], fp + c["fp"]
    ref = json.loads(metrics.read_text())["degraded"]
    print(f"{len(ids)} degraded images: TP {tp} FP {fp}  |  metrics: TP {ref['tp']} FP {ref['fp']}")
    if (tp, fp) != (ref["tp"], ref["fp"]):
        raise SystemExit("does not reproduce the evaluator: the picture would not match the score")


def draw(det_dir: Path, image_id: str, tau: float, out: Path, note: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from PIL import Image

    info, gts, ignores, expected, preds = load(det_dir, [image_id])
    c = classify(image_id, preds[image_id], gts, ignores, expected, tau)
    img = Image.open(DATA / "canonical/images" / Path(info[image_id]["file_name"]).name).convert(
        "RGB"
    )
    style = {  # colour + line style (red / green sit at the CVD floor: dash pattern too)
        "correct": ("#008300", "-", 0.55),
        "wrong_name": ("#e34948", "-", 0.9),
        "extra_box": ("#eda100", ":", 0.9),
        "missed": ("#6250d6", "--", 0.8),
    }
    name = {
        "correct": "ký hiệu nhận đúng (box + tên)",
        "wrong_name": "ký hiệu sai tên",
        "extra_box": "box thừa (không có ký hiệu)",
        "missed": "ký hiệu bỏ sót (Stage 1 không tìm thấy)",
    }
    boxes = [
        (k, p.xyxy, a, t)
        for p, k, a, t in zip(
            preds[image_id], c["pred_kind"], c["answers"], c["pred_truth"], strict=True
        )
        if k != "ignored"
    ] + [
        ("missed", g.xyxy, None, None)
        for g, m in zip(c["scored"], c["gt_missed"], strict=True)
        if m
    ]
    count = {k: c["gt_kind"].count(k) for k in ("correct", "wrong_name", "missed")}
    count["extra_box"] = c["pred_kind"].count("extra_box")
    n_gt = len(c["scored"])

    def panel(ax, crop, labels, lw_scale):
        x0, y0, x1, y1 = crop
        ax.imshow(img.crop(crop), extent=(x0, x1, y1, y0))
        for k, (bx0, by0, bx1, by1), ans, truth in boxes:
            if bx1 < x0 or bx0 > x1 or by1 < y0 or by0 > y1:
                continue
            col, ls, lw = style[k]
            ax.add_patch(
                Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False, ec=col, ls=ls,
                          lw=lw * lw_scale)
            )  # fmt: skip
            if labels and k in ("correct", "wrong_name"):
                show = {OTHER_CLASS_NAME: "Other"}
                ans, truth = show.get(ans, ans), show.get(truth, truth)
                txt = ans if k == "correct" else f"{ans}\n(đúng: {truth})"
                tag = {"fc": "white", "ec": col, "lw": 0.8, "pad": 1.2, "alpha": 0.9}
                if bx0 >= x0 and by0 - 60 >= y0 and bx0 < x1 - 200:  # label fits inside the crop
                    ax.text(bx0, by0 - 6, txt, fontsize=6, color="#0b0b0b", va="bottom",
                            bbox=tag, clip_on=True)  # fmt: skip
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_xticks([])
        ax.set_yticks([])

    legend = [
        Line2D([], [], color=style[k][0], ls=style[k][1], lw=2.2, label=f"{name[k]}: {count[k]}")
        for k in KINDS
    ]
    other = [expected[g.name] == OTHER_CLASS_NAME for g in c["scored"]]
    split = {
        (o, k): sum(1 for oo, kk in zip(other, c["gt_kind"], strict=True) if (oo, kk) == (o, k))
        for o in (False, True)
        for k in ("correct", "wrong_name", "missed")
    }
    title = (
        f"OPEN100-ISO, tờ {image_id} (bản suy giảm){note} — Stage 1 + Stage 2 fine-tune,"
        f" τ = {tau:.4f}\n"
        f"{n_gt} ký hiệu: {count['correct']} nhận đúng, {count['wrong_name']} sai tên, "
        f"{count['missed']} bỏ sót; thêm {count['extra_box']} box thừa\n"
        f"• {other.count(False)} ký hiệu thuộc từ vựng ISO: {split[False, 'correct']} đúng, "
        f"{split[False, 'wrong_name']} sai tên, {split[False, 'missed']} bỏ sót   "
        f"• {other.count(True)} ký hiệu lạ (bong bóng ISA…, đáp án đúng = Other): "
        f"{split[True, 'correct']} gọi Other, {split[True, 'wrong_name']} bị gán tên, "
        f"{split[True, 'missed']} bỏ sót"
    )
    w, h = img.size
    fig, ax = plt.subplots(figsize=(12, 12 * h / w + 1.2), dpi=200)
    panel(ax, (0, 0, w, h), labels=False, lw_scale=1.0)
    zoom = densest_window(c["scored"], w, h)
    ax.add_patch(Rectangle(zoom[:2], zoom[2] - zoom[0], zoom[3] - zoom[1], fill=False,
                           ec="#0b0b0b", lw=1.2, ls="-"))  # fmt: skip
    ax.text(zoom[0], zoom[1] - 30, "phóng to ở hình sau", fontsize=8, va="bottom")
    ax.set_title(title, fontsize=10, loc="left")
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=4,
              frameon=False, fontsize=8.5)  # fmt: skip
    fig.tight_layout()
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    zw, zh = zoom[2] - zoom[0], zoom[3] - zoom[1]
    fig, ax = plt.subplots(figsize=(12, 12 * zh / zw + 0.8), dpi=200)
    panel(ax, zoom, labels=True, lw_scale=1.3)
    ax.set_title(f"Phóng to một vùng của tờ {image_id}: tên dự đoán trên mỗi box", fontsize=10,
                 loc="left")  # fmt: skip
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=4,
              frameon=False, fontsize=8.5)  # fmt: skip
    fig.tight_layout()
    zout = out.with_name(out.stem + "_zoom" + out.suffix)
    fig.savefig(zout, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(title.replace("\n", " | "))
    print("->", out, zout)


def densest_window(gts: list[de.GtBox], w: int, h: int, frac: float = 0.22, steps: int = 24):
    """The window (frac of the sheet, sheet aspect) holding the most GT symbol centres, on a
    grid of positions; the first one on a tie."""
    ww, wh = w * frac, h * frac
    cx = np.array([(g.xyxy[0] + g.xyxy[2]) / 2 for g in gts])
    cy = np.array([(g.xyxy[1] + g.xyxy[3]) / 2 for g in gts])
    best, out = -1, (0, 0, ww, wh)
    for x in np.linspace(0, w - ww, steps):
        for y in np.linspace(0, h - wh, steps):
            n = int(((cx >= x) & (cx < x + ww) & (cy >= y) & (cy < y + wh)).sum())
            if n > best:
                best, out = n, (int(x), int(y), int(x + ww), int(y + wh))
    # shrink to the symbols inside (plus a margin for the labels), so no empty band is shown
    x0, y0, x1, y1 = out
    inside = [g.xyxy for g in gts if x0 <= (g.xyxy[0] + g.xyxy[2]) / 2 < x1
              and y0 <= (g.xyxy[1] + g.xyxy[3]) / 2 < y1]  # fmt: skip
    if inside:
        pad = 0.04 * max(w, h)
        out = (
            int(max(x0, min(b[0] for b in inside) - pad)),
            int(max(y0, min(b[1] for b in inside) - pad)),
            int(min(x1, max(b[2] for b in inside) + pad)),
            int(min(y1, max(b[3] for b in inside) + pad)),
        )
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--det-dir", type=Path, required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check", type=Path, default=None)
    ap.add_argument("--note", default="", help="appended to the sheet name, e.g. how it was chosen")
    args = ap.parse_args(argv)
    if args.check is not None:
        check(args.det_dir, args.tau, args.check)
    draw(args.det_dir, args.image, args.tau, args.out, args.note)


if __name__ == "__main__":
    main()
