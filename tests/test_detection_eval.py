"""Detection reports beyond mAP (PLAN B6b-2, B6d.2): ignore regions, eval.class_agnostic with
recall per group, OPEN100-ISO views, the one-shot end-to-end score and the wrong-box rate.

Tiny worlds written under tmp_path: an ISO-like export (vocab templates gate / ball / pump,
other_iso, other_isa, a leader-arrow ignore region; 2 sheets x clean / degraded) and a
DigitizePID-like one (Symbol_N names, no `degraded`). The encoder checkpoint is a bare dict with
`fine_classes` / `excluded_classes`: grouping reads nothing else.
"""

from __future__ import annotations

import json
import random

import pytest
import yaml
from omegaconf import OmegaConf

from pipeline.evaluation import EvaluationPipeline
from utils import detection_metrics
from utils.classmap import IGNORE_CATEGORY
from utils.detection_metrics import greedy_match, ignore_hits, match_report, mean_ap

# ------------------------------------------------------------------ #
# utils.detection_metrics: the vectorised path is the old one, plus ignore regions
# ------------------------------------------------------------------ #


def _random_boxes(rng, n, images, classes, scores=True):
    out = []
    for _ in range(n):
        x, y = rng.uniform(0, 80), rng.uniform(0, 80)
        w, h = rng.uniform(2, 20), rng.uniform(2, 20)
        image, cls = rng.choice(images), rng.choice(classes)
        if scores:
            out.append((image, cls, round(rng.uniform(0, 1), 1), (x, y, x + w, y + h)))
        else:
            out.append((image, cls, (x, y, x + w, y + h)))
    return out


@pytest.mark.parametrize("seed", range(5))
def test_match_report_reproduces_mean_ap_exactly_without_ignore_regions(seed):
    rng = random.Random(seed)
    gts = _random_boxes(rng, 60, ["a", "b", "c"], [0, 1, 2], scores=False)
    # predictions near the GT boxes (some matching) plus noise; scores rounded -> many ties
    preds = [
        (i, c, round(rng.uniform(0, 1), 1), tuple(v + rng.uniform(-2, 2) for v in b))
        for i, c, b in gts
        if rng.random() < 0.7
    ] + _random_boxes(rng, 40, ["a", "b", "c", "d"], [0, 1, 2, 3])
    thrs = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    want = mean_ap(preds, gts, thrs)
    got, match, _ = match_report(preds, gts, thrs)
    assert got["map_50"] == want["map_50"] and got["map_50_95"] == want["map_50_95"]
    assert got["per_class_ap_50"] == want["per_class_ap_50"]
    assert got["tp"] == int(match.gt_hit.sum()) and got["tp"] + got["fp"] == len(preds)


def test_iou_matrix_is_the_scalar_iou():
    rng = random.Random(0)
    a = [b for _, _, b in _random_boxes(rng, 15, ["a"], [0], scores=False)]
    b = [b for _, _, b in _random_boxes(rng, 12, ["a"], [0], scores=False)]
    m = detection_metrics.iou_matrix(a, b)
    assert all(m[i, j] == detection_metrics.iou(a[i], b[j]) for i in range(15) for j in range(12))


def test_a_prediction_in_an_ignore_region_is_neither_tp_nor_fp():
    gts = [("a", 0, (0.0, 0.0, 10.0, 10.0))]
    tp = ("a", 0, 0.9, (0.0, 0.0, 10.0, 10.0))
    fp = ("a", 0, 0.8, (50.0, 50.0, 60.0, 60.0))
    region = [("a", (100.0, 100.0, 200.0, 200.0))]
    inside = ("a", 0, 0.95, (110.0, 110.0, 120.0, 120.0))  # IoU 0.01, fully inside
    base, _, _ = match_report([tp, fp], gts, [0.5], region)
    with_it, match, hits = match_report([tp, fp, inside], gts, [0.5], region)
    assert (with_it["precision"], with_it["recall"]) == (base["precision"], base["recall"])
    assert with_it["map_50"] == base["map_50"]
    assert match.pred_ignored.tolist() == [False, False, True] and hits.tolist() == [-1, -1, 0]
    # without the region the same prediction is an FP (and the best-scored one: AP drops)
    without, _, _ = match_report([tp, fp, inside], gts, [0.5])
    assert without["fp"] == 2 and without["map_50"] < base["map_50"]


def test_ignore_match_rule_iou_or_mostly_inside():
    region = [("a", (0.0, 0.0, 10.0, 10.0))]

    def hit(box):
        return int(ignore_hits([("a", 0, 1.0, box)], region)[0])

    assert hit((0.0, 0.0, 10.0, 10.0)) == 0  # IoU 1
    assert hit((5.0, 0.0, 14.0, 10.0)) == 0  # 5/9 of the prediction inside
    assert hit((6.0, 0.0, 16.0, 10.0)) == -1  # 4/10 inside, IoU 0.25
    assert hit((-100.0, -100.0, 10.0, 10.0)) == -1  # a big box around the region
    assert int(ignore_hits([("b", 0, 1.0, (0.0, 0.0, 10.0, 10.0))], region)[0]) == -1


def test_a_true_positive_inside_an_ignore_region_stays_a_true_positive():
    gts = [("a", 0, (0.0, 0.0, 10.0, 10.0))]
    region = [("a", (0.0, 0.0, 30.0, 30.0))]  # a leader arrow over the symbol
    preds = [("a", 0, 0.9, (0.0, 0.0, 10.0, 10.0)), ("a", 0, 0.8, (0.0, 0.0, 10.0, 10.0))]
    m = greedy_match(preds, gts, 0.5, ignore_hits(preds, region))
    assert m.tp.tolist() == [True, False] and m.pred_ignored.tolist() == [False, True]


def test_right_box_wrong_class_is_tp_when_agnostic_and_fp_when_not():
    gts = [("a", 0, (0.0, 0.0, 10.0, 10.0))]
    preds = [("a", 3, 0.9, (0.0, 0.0, 10.0, 10.0))]
    by_class, _, _ = match_report(preds, gts, [0.5])
    agnostic, _, _ = match_report([(i, 0, s, b) for i, _, s, b in preds], gts, [0.5])
    assert (by_class["tp"], by_class["fp"], by_class["map_50"]) == (0, 1, 0.0)
    assert (agnostic["tp"], agnostic["fp"], agnostic["map_50"]) == (1, 0, 1.0)


# ------------------------------------------------------------------ #
# worlds
# ------------------------------------------------------------------ #

_W = _H = 1000
_VOCAB = """reg_number,coarse,fine_name,pid_group,fgc_group,desc
A1,valve,gate,valve,yes,
A2,valve,ball,valve,yes,
A5,pump,pump,equipment,no,
"""
_SYMBOL_MAP = """symbol_n,class_id,coarse,fgc_group,status,fine_name,reg_number
1,0,valve,yes,ISO,gate,A1
2,1,valve,yes,ISO,ball,A2
6,5,instrument,yes,OOV,bubble_a,
8,7,instrument,yes,UNSURE,odd,
"""
# R1a-like: Symbol_2 (ball) held out -> gate S, ball H_chon, pump H_tu_nhien
_FINE, _EXCLUDED = ["Symbol_1", "Symbol_6", "Symbol_8"], ["Symbol_2"]
# ISO-like sheet: one box per category (x, y, w, h) + a leader-arrow ignore region
_ISO_BOXES = {
    "gate": (100, 100, 50, 50),
    "ball": (300, 100, 50, 50),
    "pump": (500, 100, 50, 50),
    "other_iso": (700, 100, 50, 50),
    "other_isa": (100, 500, 50, 50),
}
_REGION = (500, 500, 100, 100)
_ISO_GROUP = {"gate": "S", "ball": "H_chon", "pump": "H_tu_nhien",
              "other_iso": "other_iso", "other_isa": "other_isa"}  # fmt: skip


def _xyxy(b):
    x, y, w, h = b
    return (x, y, x + w, y + h)


def _yolo(cid, xyxy, score):
    x1, y1, x2, y2 = xyxy
    cx, cy, w, h = (x1 + x2) / 2 / _W, (y1 + y2) / 2 / _H, (x2 - x1) / _W, (y2 - y1) / _H
    return f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {score:.6f}"


def _write_root(tmp_path, coco, ids, split="test"):
    data = tmp_path / "data"
    (data / "canonical").mkdir(parents=True, exist_ok=True)
    cats = [c for c in coco["categories"] if c["name"] != IGNORE_CATEGORY]
    classes = {
        "coarse": sorted({c["supercategory"] for c in cats}),
        "fine": {c["name"]: c["supercategory"] for c in cats},
        "fgc_groups": [],
    }
    (data / "canonical" / "classes.yaml").write_text(yaml.safe_dump(classes, sort_keys=False))
    (data / "canonical" / "annotations.coco.json").write_text(json.dumps(coco))
    (data / "canonical" / "manifest.csv").write_text(
        "id,split\n" + "".join(f"{i},{split}\n" for i in ids)
    )
    (tmp_path / "vocab.csv").write_text(_VOCAB)
    (tmp_path / "symbol_map.csv").write_text(_SYMBOL_MAP)
    return data


def _iso_coco(ignore=True, tags=True):
    names = list(_ISO_BOXES)
    cats = [{"id": i + 1, "name": n, "supercategory": "valve" if n in ("gate", "ball") else "x"}
            for i, n in enumerate(names)]  # fmt: skip
    if ignore:
        cats.append({"id": 99, "name": IGNORE_CATEGORY, "supercategory": IGNORE_CATEGORY})
    images, anns = [], []
    for sheet in (0, 1):
        for degraded in (False, True):
            stem = f"iso{sheet:02d}_v000" + ("_deg" if degraded else "")
            image_id = len(images) + 1
            im = {"id": image_id, "file_name": f"synthetic/{stem}.png", "width": _W, "height": _H}
            if tags:
                im.update(sheet=sheet, variant=0, degraded=degraded)
            images.append(im)
            for i, n in enumerate(names):
                anns.append({"id": len(anns) + 1, "image_id": image_id, "category_id": i + 1,
                             "bbox": list(_ISO_BOXES[n]), "iscrowd": 0})  # fmt: skip
            if ignore:
                anns.append({"id": len(anns) + 1, "image_id": image_id, "category_id": 99,
                             "bbox": list(_REGION), "iscrowd": 1,
                             "ignore_kind": "leader_arrow"})  # fmt: skip
    return {"images": images, "annotations": anns, "categories": cats}


def _stems(coco):
    return [im["file_name"].split("/")[1].rsplit(".", 1)[0] for im in coco["images"]]


def _cfg(tmp_path, data, mode="det_only", ckpt=True, **over):
    enc = tmp_path / "enc.pt"
    if ckpt:
        torch = pytest.importorskip("torch")
        torch.save({"fine_classes": _FINE, "excluded_classes": _EXCLUDED}, enc)
    cfg = OmegaConf.create(
        {
            "data_root": str(data),
            "run_dir": str(tmp_path / "runs"),
            "eval": {"mode": mode, "iou_thrs": [0.5, 0.75], "score_thr": 0.25,
                     "per_class_ap": True, "topk": [1, 3], "split": "test"},
            "fgc": {"ckpt_dir": None},
            "detector": {"label_level": "coarse"},
            "oneshot": {"vocab_csv": str(tmp_path / "vocab.csv"),
                        "symbol_map": str(tmp_path / "symbol_map.csv"),
                        "encoder_ckpt": str(enc) if ckpt else None,
                        "other_min_cosine": 0.5},
        }
    )  # fmt: skip
    return OmegaConf.merge(cfg, OmegaConf.create(over))


def _stage1(data, stems, lines, split="test"):
    out = data / "derived" / "detections" / split
    out.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        (out / f"{stem}.txt").write_text("\n".join(lines) + "\n")


def _iso_stage1(tmp_path, extra=(), ignore=True, tags=True):
    """Every sheet: exact boxes on gate (class 0), ball (class 1: another class), other_iso;
    one box inside the leader-arrow region, one far from everything."""
    coco = _iso_coco(ignore=ignore, tags=tags)
    data = _write_root(tmp_path, coco, _stems(coco))
    lines = [
        _yolo(0, _xyxy(_ISO_BOXES["gate"]), 0.9),
        _yolo(1, _xyxy(_ISO_BOXES["ball"]), 0.8),
        _yolo(0, _xyxy(_ISO_BOXES["other_iso"]), 0.7),
        _yolo(0, (520, 520, 550, 550), 0.95),  # inside the ignore region
        _yolo(0, (900, 900, 940, 940), 0.6),  # nothing there
        *extra,
    ]
    _stage1(data, _stems(coco), lines)
    return data


# ------------------------------------------------------------------ #
# EvaluationPipeline: DigitizePID unchanged
# ------------------------------------------------------------------ #


def _digitizepid_world(tmp_path, ignore=False):
    rng = random.Random(3)
    cats = [{"id": 1, "name": "Symbol_1", "supercategory": "valve"},
            {"id": 2, "name": "Symbol_6", "supercategory": "instrument"}]  # fmt: skip
    images = [{"id": k, "file_name": f"synthetic/{k}.jpg", "width": _W, "height": _H}
              for k in (10, 11, 12)]  # fmt: skip
    anns, lines = [], {}
    for im in images:
        lines[str(im["id"])] = []
        for _ in range(8):
            x, y = rng.uniform(0, 900), rng.uniform(0, 900)
            w, h = rng.uniform(20, 80), rng.uniform(20, 80)
            cat = rng.choice([1, 2])
            anns.append({"id": len(anns) + 1, "image_id": im["id"], "category_id": cat,
                         "bbox": [x, y, w, h], "iscrowd": 0})  # fmt: skip
            if rng.random() < 0.8:
                j = [rng.uniform(-6, 6) for _ in range(4)]
                box = (x + j[0], y + j[1], x + w + j[2], y + h + j[3])
                # classes.yaml coarse order is [instrument, valve]: Symbol_1 (valve) is id 1
                cid = 2 - cat if rng.random() < 0.8 else cat - 1
                lines[str(im["id"])].append(_yolo(cid, box, round(rng.uniform(0.2, 1), 1)))
        lines[str(im["id"])].append(_yolo(0, (950, 950, 990, 990), 0.5))
    coco = {"images": images, "annotations": anns, "categories": cats}
    if ignore:
        coco["categories"].append({"id": 99, "name": IGNORE_CATEGORY, "supercategory": "x"})
        coco["annotations"].append({"id": 999, "image_id": 10, "category_id": 99, "iscrowd": 1,
                                    "bbox": [0, 0, 5, 5], "ignore_kind": "flag"})  # fmt: skip
    data = _write_root(tmp_path, coco, ["10", "11", "12"])
    for stem, ls in lines.items():
        _stage1(data, [stem], ls)
    return data, coco


def test_digitizepid_without_switches_takes_the_old_path_unchanged(tmp_path):
    data, coco = _digitizepid_world(tmp_path)
    pipe = EvaluationPipeline(_cfg(tmp_path, data, ckpt=False))
    got = pipe.compute_detection_metrics()
    assert set(got) == {"map_50", "map_50_95", "per_class_ap_50"}  # no extended block
    image_id_to_wh, coco_ids = pipe._image_sizes(coco)
    ids = {"10", "11", "12"}
    preds = pipe._load_preds(ids, image_id_to_wh)
    want = mean_ap(
        preds, pipe._load_gt(coco, ids, image_id_to_wh, coco_ids, "det_only"), [0.5, 0.75]
    )
    assert got == want and 0 < got["map_50"] < 1


def test_an_ignore_region_nothing_falls_in_changes_no_number(tmp_path):
    data, _ = _digitizepid_world(tmp_path / "a")
    plain = EvaluationPipeline(_cfg(tmp_path / "a", data, ckpt=False)).compute_detection_metrics()
    data, _ = _digitizepid_world(tmp_path / "b", ignore=True)
    ext = EvaluationPipeline(_cfg(tmp_path / "b", data, ckpt=False)).compute_detection_metrics()
    assert ext["class_agnostic"] is False and ext["ignore"]["n_regions"] == 1
    assert ext["ignore"]["n_predictions"] == 0
    for key in ("map_50", "map_50_95", "per_class_ap_50"):
        assert ext[key] == plain[key]


def test_class_agnostic_on_and_off_agree_when_there_is_one_class(tmp_path):
    data, coco = _digitizepid_world(tmp_path)
    for ann in coco["annotations"]:
        ann["category_id"] = 1
    (data / "canonical" / "annotations.coco.json").write_text(json.dumps(coco))
    for f in (data / "derived" / "detections" / "test").glob("*.txt"):
        f.write_text("".join("1" + line[1:] + "\n" for line in f.read_text().split("\n") if line))
    off = EvaluationPipeline(_cfg(tmp_path, data, ckpt=False)).compute_detection_metrics()
    on = EvaluationPipeline(
        _cfg(tmp_path, data, ckpt=False, eval={"class_agnostic": True})
    ).compute_detection_metrics()
    assert on["class_agnostic"] is True
    assert (on["map_50"], on["map_50_95"]) == (off["map_50"], off["map_50_95"])


def test_class_agnostic_scores_the_box_not_the_class(tmp_path):
    data, coco = _digitizepid_world(tmp_path)
    off = EvaluationPipeline(_cfg(tmp_path, data, ckpt=False)).compute_detection_metrics()
    on = EvaluationPipeline(
        _cfg(tmp_path, data, ckpt=False, eval={"class_agnostic": True})
    ).compute_detection_metrics()
    assert on["map_50"] > off["map_50"]  # 20% of the boxes carry the other class
    assert on["groups"]["note"] and "recall_by_group" not in on  # no checkpoint: no group
    assert sum(r["n"] for r in on["recall_by_class"].values()) == on["n_gt"] == 24


# ------------------------------------------------------------------ #
# EvaluationPipeline: OPEN100-ISO class-agnostic
# ------------------------------------------------------------------ #


def test_iso_class_agnostic_ignore_regions_groups_and_views(tmp_path):
    data = _iso_stage1(tmp_path)
    cfg = _cfg(tmp_path, data, eval={"class_agnostic": True})
    EvaluationPipeline(cfg).run()
    det = json.loads((tmp_path / "runs" / "metrics.json").read_text())["detection"]
    assert det["primary"] == "degraded" and "map_50" not in det  # never pooled
    for view in ("degraded", "clean"):
        b = det[view]
        assert b["n_images"] == 2 and b["n_gt"] == 10  # the ignore regions are not GT
        assert (b["tp"], b["fp"]) == (6, 2) and b["recall"] == pytest.approx(0.6)
        assert b["precision"] == pytest.approx(0.75)  # the box in the region is not an FP
        arrows = {"leader_arrow": 2}
        assert b["ignore"] == {"n_regions": 2, "regions_by_kind": arrows,
                               "n_predictions": 2, "predictions_by_kind": arrows}  # fmt: skip
        groups = b["recall_by_group"]
        assert {g: r["recall"] for g, r in groups.items()} == {
            "S": 1.0, "H_chon": 1.0, "H_tu_nhien": 0.0, "other_iso": 1.0, "other_isa": 0.0
        }  # fmt: skip
        # group hits add up to the total, every GT box has a group
        assert sum(r["tp"] for r in groups.values()) == b["tp"]
        assert sum(r["n"] for r in groups.values()) == b["n_gt"]
        assert {n: r["group"] for n, r in b["recall_by_class"].items()} == _ISO_GROUP
        assert set(b["per_sheet"]) == {"iso00", "iso01"}
        assert (
            b["per_sheet"]["iso00"]["n_gt"] == 5
            and "recall_by_class" not in b["per_sheet"]["iso00"]
        )
        ms = b["sheet_mean_std"]
        assert ms["recall"] == {"mean": pytest.approx(0.6), "std": 0.0, "n": 2}
        assert ms["recall_by_group"]["S"]["recall"]["mean"] == 1.0
    assert det["groups"]["checkpoint"].endswith("enc.pt")


def test_iso_without_views_tags_is_one_block_and_without_ignore_the_arrow_box_is_an_fp(tmp_path):
    data = _iso_stage1(tmp_path, ignore=False, tags=False)
    det = EvaluationPipeline(
        _cfg(tmp_path, data, eval={"class_agnostic": True})
    ).compute_detection_metrics()
    assert "degraded" not in det and "ignore" not in det
    assert (det["tp"], det["fp"]) == (12, 8)


def test_class_agnostic_groups_follow_eval_oneshot_groups_ckpt(tmp_path):
    torch = pytest.importorskip("torch")
    data = _iso_stage1(tmp_path)
    torch.save({"fine_classes": ["Symbol_1", "Symbol_2"], "excluded_classes": []},
               tmp_path / "full.pt")  # fmt: skip
    over = {"class_agnostic": True, "oneshot": {"groups_ckpt": str(tmp_path / "full.pt")}}
    cfg = _cfg(tmp_path, data, eval=over)
    det = EvaluationPipeline(cfg).compute_detection_metrics()
    assert set(det["degraded"]["recall_by_group"]) == {"S", "H_tu_nhien", "other_iso", "other_isa"}
    assert det["groups"]["of_name"]["ball"] == "S"


def test_partially_tagged_images_are_refused(tmp_path):
    data = _iso_stage1(tmp_path)
    coco = json.loads((data / "canonical" / "annotations.coco.json").read_text())
    del coco["images"][0]["degraded"]
    (data / "canonical" / "annotations.coco.json").write_text(json.dumps(coco))
    with pytest.raises(ValueError, match="pool|degraded"):
        EvaluationPipeline(_cfg(tmp_path, data, ckpt=False)).compute_detection_metrics()


# ------------------------------------------------------------------ #
# End to end (det_plus_fgc ++fgc.oneshot=true) + wrong boxes
# ------------------------------------------------------------------ #


def _oneshot_rows(rows, data, stems, split="test"):
    pytest.importorskip("torch")
    from pipeline.oneshot_infer import OneShotDetection, write_oneshot_detections
    from utils.types import BBox

    out = data / "derived" / "detections_oneshot" / split
    out.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        dets = []
        for label, name, xyxy, score, cos, second in rows:
            x1, y1, x2, y2 = xyxy
            box = BBox(
                x=(x1 + x2) / 2 / _W, y=(y1 + y2) / 2 / _H, w=(x2 - x1) / _W, h=(y2 - y1) / _H
            )
            dets.append(OneShotDetection(label, name, box, score, cos, second, 0))
        write_oneshot_detections(out / f"{stem}.txt", dets)


_E2E_ROWS = [
    # label (as inference gated it; re-gated at eval), best_name, box, det score, cos, 2nd cos
    ("gate", "gate", _xyxy(_ISO_BOXES["gate"]), 0.9, 0.9, 0.5),  # TP
    ("gate", "gate", _xyxy(_ISO_BOXES["ball"]), 0.9, 0.8, 0.5),  # wrong name
    ("pump", "pump", _xyxy(_ISO_BOXES["pump"]), 0.9, 0.3, 0.1),  # rejected at 0.5
    ("ball", "ball", _xyxy(_ISO_BOXES["other_iso"]), 0.9, 0.2, 0.1),  # Other on other_iso: TP
    ("gate", "gate", (520, 520, 550, 550), 0.9, 0.9, 0.2),  # in the leader-arrow region
    ("gate", "gate", (900, 900, 940, 940), 0.9, 0.4, 0.1),  # wrong box, rejected at tau
    ("gate", "gate", (850, 300, 890, 340), 0.9, 0.9, 0.88),  # wrong box, only the margin rejects
    ("__unrefined__", "__unrefined__", (300, 300, 340, 340), 0.9, float("nan"), float("nan")),
    ("gate", "gate", (600, 900, 640, 940), 0.1, 0.9, 0.1),  # below eval.score_thr
]


def _e2e(tmp_path, tags=True, **over):
    coco = _iso_coco(tags=tags)
    data = _write_root(tmp_path, coco, _stems(coco))
    _oneshot_rows(_E2E_ROWS, data, _stems(coco))
    cfg = _cfg(tmp_path, data, mode="det_plus_fgc", fgc={"ckpt_dir": None, "oneshot": True})
    cfg = OmegaConf.merge(cfg, OmegaConf.create(over))
    EvaluationPipeline(cfg).run()
    return json.loads((tmp_path / "runs" / "metrics.json").read_text())


def test_e2e_tp_needs_the_box_and_the_iso_label(tmp_path):
    m = _e2e(tmp_path)
    assert "detection" not in m and "fgc" not in m
    o = m["oneshot_e2e"]
    assert o["tau"] == 0.5 and o["margin_delta"] is None and o["n_unrefined_dropped"] == 4
    for view in ("degraded", "clean"):
        b = o[view]
        assert (b["n_gt"], b["n_predictions"]) == (10, 14)
        assert (b["tp"], b["fp"], b["n_ignored_predictions"]) == (4, 8, 2)
        assert (b["fp_box"], b["fp_label"]) == (4, 4)
        assert b["recall"] == pytest.approx(0.4) and b["box_recall"] == pytest.approx(0.8)
        g = b["by_group"]
        assert {k: r["recall"] for k, r in g.items()} == {
            "S": 1.0, "H_chon": 0.0, "H_tu_nhien": 0.0, "other_iso": 1.0, "other_isa": 0.0
        }  # fmt: skip
        assert g["H_chon"]["box_matched"] == {"right": 0, "rejected": 0, "wrong_name": 2}
        assert g["H_tu_nhien"]["box_matched"] == {"right": 0, "rejected": 2, "wrong_name": 0}
        assert g["H_tu_nhien"]["box_recall"] == 1.0 and g["other_isa"]["box_recall"] == 0.0
        assert sum(r["tp"] for r in g.values()) == b["tp"]
        prec = b["precision_by_predicted_group"]
        assert prec["Other"] == {"n": 6, "tp": 2, "precision": pytest.approx(1 / 3)}
        # named gate: the gate box (right), the ball box and the margin-only wrong box (wrong)
        assert prec["S"] == {"n": 6, "tp": 2, "precision": pytest.approx(1 / 3)}
        assert set(b["per_sheet"]) == {"iso00", "iso01"}
        assert b["sheet_mean_std"]["by_group"]["S"]["recall"]["mean"] == 1.0
        assert b["by_class"]["ball"]["expected"] == "ball"
        assert b["by_class"]["other_iso"]["expected"] == "__other__"


def test_e2e_wrong_box_rate_at_tau_and_at_tau_or_margin(tmp_path):
    b = _e2e(tmp_path)["oneshot_e2e"]["degraded"]
    w = b["wrong_boxes"]
    # three boxes with IoU < 0.3 to every GT box per sheet: the far one (cos 0.4: rejected), the
    # margin one (cos 0.9) and the one on the leader arrow (cos 0.9, inside an ignore region)
    assert (w["n"], w["in_ignore_region"]) == (6, 2)
    assert w["rejected_tau"] == pytest.approx(1 / 3) and "rejected_tau_or_margin" not in w
    assert w["frac_of_predictions"] == pytest.approx(6 / 14)
    t = b["true_vocab_boxes"]  # gate, ball and pump boxes: pump (cos 0.3) is rejected
    assert t["n"] == 6 and t["rejected_tau"] == pytest.approx(1 / 3)
    w = _e2e(tmp_path, eval={"oneshot": {"margin_delta": 0.05}})["oneshot_e2e"]["degraded"]
    assert w["wrong_boxes"]["rejected_tau_or_margin"] == pytest.approx(2 / 3)
    assert w["wrong_boxes"]["rejected_tau"] == pytest.approx(1 / 3)


def test_e2e_tau_is_applied_at_eval_time(tmp_path):
    """Same files, tau 0.1: pump is named (right), other_iso is named ball (wrong)."""
    b = _e2e(tmp_path, oneshot={"other_min_cosine": 0.1})["oneshot_e2e"]["clean"]
    g = b["by_group"]
    assert g["H_tu_nhien"]["recall"] == 1.0 and g["other_iso"]["recall"] == 0.0
    assert b["wrong_boxes"]["rejected_tau"] == 0.0


def test_e2e_on_digitizepid_reads_symbol_map_and_the_image_subset(tmp_path):
    """DigitizePID GT (Symbol_N): the template of an ISO class, Other for an OOV class, no test
    for an UNSURE one; ++eval.image_subset=val_sel keeps the val_sel images only."""
    cats = [{"id": 1, "name": "Symbol_1", "supercategory": "valve"},
            {"id": 2, "name": "Symbol_6", "supercategory": "instrument"},
            {"id": 3, "name": "Symbol_8", "supercategory": "instrument"}]  # fmt: skip
    images = [{"id": k, "file_name": f"synthetic/{k}.jpg", "width": _W, "height": _H}
              for k in (10, 12)]  # fmt: skip
    anns = [{"id": 10 * im["id"] + c, "image_id": im["id"], "category_id": c,
             "bbox": [100 + 200 * c, 100, 50, 50], "iscrowd": 0}
            for im in images for c in (1, 2, 3)]  # fmt: skip
    data = _write_root(tmp_path, {"images": images, "annotations": anns, "categories": cats},
                       ["10", "12"], split="val")  # fmt: skip
    rows = [
        ("gate", "gate", (300, 100, 350, 150), 0.9, 0.9, 0.1),  # Symbol_1 named gate: TP
        ("gate", "gate", (500, 100, 550, 150), 0.9, 0.2, 0.1),  # Symbol_6 (OOV) -> Other: TP
        ("gate", "gate", (700, 100, 750, 150), 0.9, 0.9, 0.1),  # UNSURE: no test, ignored
        ("gate", "gate", (900, 900, 940, 940), 0.9, 0.3, 0.1),  # wrong box
    ]
    _oneshot_rows(rows, data, ["10", "12"], split="val")
    (tmp_path / "splits.json").write_text(json.dumps({"val_sel": ["10"], "test": ["12"]}))
    cfg = _cfg(tmp_path, data, mode="det_plus_fgc", fgc={"ckpt_dir": None, "oneshot": True},
               eval={"split": "val", "image_subset": "val_sel",
                     "splits_file": str(tmp_path / "splits.json")})  # fmt: skip
    pipe = EvaluationPipeline(cfg)
    assert pipe.validate() is True
    pipe.run()
    m = json.loads((tmp_path / "runs" / "metrics.json").read_text())
    o = m["oneshot_e2e"]
    assert m["image_subset"] == "val_sel" and "degraded" not in o
    assert (o["n_images"], o["n_gt"], o["n_gt_no_answer"]) == (1, 2, 1)
    assert (o["tp"], o["fp"], o["n_ignored_predictions"]) == (2, 1, 1)
    assert o["by_group"]["S"]["recall"] == 1.0 and o["by_group"]["OOV_seen"]["recall"] == 1.0
    assert o["wrong_boxes"]["n"] == 1 and o["wrong_boxes"]["rejected_tau"] == 1.0
    with pytest.raises(ValueError, match="crop-level"):
        EvaluationPipeline(OmegaConf.merge(cfg, {"fgc": {"oneshot": False}})).validate()


def test_e2e_validate_points_at_the_oneshot_inference(tmp_path):
    coco = _iso_coco()
    data = _write_root(tmp_path, coco, _stems(coco))
    cfg = _cfg(tmp_path, data, mode="det_plus_fgc", fgc={"ckpt_dir": None, "oneshot": True})
    with pytest.raises(FileNotFoundError, match=r"detections_oneshot.*fgc.oneshot=true"):
        EvaluationPipeline(cfg).validate()


# ------------------------------------------------------------------ #
# detect_infer: a detector trained on another dataset
# ------------------------------------------------------------------ #


def test_detect_infer_reads_the_label_set_from_detector_classes_yaml(tmp_path):
    pytest.importorskip("ultralytics")
    pytest.importorskip("sahi")
    from pipeline.detect_infer import DetectInferPipeline, _write_detections_txt
    from utils.types import BBox

    class _Yolo:
        def __init__(self, names):
            self.names = dict(enumerate(names))

    coco = _iso_coco()
    data = _write_root(tmp_path, coco, _stems(coco))  # ISO classes.yaml: coarse [valve, x]
    other = tmp_path / "digitizepid_classes.yaml"
    other.write_text("coarse: [flange, valve, pump]\nfine: {a: flange, b: valve, c: pump}\n"
                     "fgc_groups: []\n")  # fmt: skip
    cfg = _cfg(tmp_path, data, ckpt=False)
    with pytest.raises(ValueError, match="detector.classes_yaml"):
        DetectInferPipeline(cfg)._check_model_names(_Yolo(["flange", "valve", "pump"]))
    pipe = DetectInferPipeline(OmegaConf.merge(cfg, {"detector": {"classes_yaml": str(other)}}))
    pipe._check_model_names(_Yolo(["flange", "valve", "pump"]))
    det = pipe._detection(BBox.from_xyxy(0, 0, 10, 10), 2, 0.9)
    assert det.coarse_class == "pump"
    out = tmp_path / "a.txt"
    _write_detections_txt(out, [det], pipe.detector_classmap, 100, 100)
    assert out.read_text().split()[0] == "2"
    assert pipe.classmap.coarse_classes == ["valve", "x"]  # the data root's own is untouched
