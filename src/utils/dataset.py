"""Crop dataset + balanced sampling for the FGC stage.

Reads derived/crops_fgc/{split}/<fine_class>/ — a FLAT layout: every fine class
across every fgc_group lives as a sibling folder directly under crops_fgc/<split>/,
not nested by group. torchvision's ImageFolder auto-discovers every subdirectory as a
class, so it can't be pointed at that tree and restricted to one group's fine classes
cleanly (it would mix classes across groups, or choke on subfolders belonging to other
groups). Instead, `_GroupCropDataset` below explicitly filters to `group`'s fine
classes via classmap.fine_of_group(group) and assigns GROUP-LOCAL label indices.

Rare-class balancing (§2.2.3): a WeightedRandomSampler that can build a batch with
roughly one patch per fine class regardless of natural frequency — impractical at
full-image scale, which is the point of the separable stage (§4.2.3).

One classifier head PER fgc_group is the default (a valve-family head, a bubble-family
head, ...), since fine classes only compete within their group.

Crops are stored with extra context and a `_boxes.csv` per class folder (see
utils.crop_box): every split cuts the classification view (box + 10% margin, as at
inference) out of the stored crop; the train split may cut it around a jittered box
instead (`fgc.aug.box_jitter`, off unless p > 0). Crops without a `_boxes.csv` (cut before
that format) are used as they are.
"""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

from utils.augment import SubpartWarp
from utils.classmap import load_classmap, training_fine_classes
from utils.crop_box import BoxJitter, StoredCrop, read_boxes, view
from utils.paths import DataPaths
from utils.scan_degrade import ScanDegrade

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

# eval.splits_file default, resolved against the repo root (not data_root): the file is
# tracked in git (PLAN 2026-11-12, val split into val_sel / test).
DEFAULT_SPLITS_FILE = "configs/splits/digitizepid_val_test.json"
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Standard ImageNet normalization stats — required to feed a timm backbone that was
# pretrained on ImageNet (cfg.fgc.pretrained=true uses those pretrained weights).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def crop_image_id(name: str) -> str:
    """Source image id of a stored crop: `<image_id>_<n>.png` (pipeline.extract_crops)."""
    return Path(name).stem.rsplit("_", 1)[0]


def scan_crops(
    root: Path, fine_classes: list[str], image_ids: set[str] | None = None
) -> tuple[list[tuple[Path, int]], dict[Path, StoredCrop]]:
    """(crop path, index into `fine_classes`) for every crop under root/<class>/, plus where
    each crop with a `_boxes.csv` row sits. `image_ids` keeps only crops cut from those
    images (None keeps all)."""
    samples: list[tuple[Path, int]] = []
    stored: dict[Path, StoredCrop] = {}
    for local_idx, fine_class in enumerate(fine_classes):
        class_dir = root / fine_class
        if not class_dir.is_dir():
            continue
        boxes = read_boxes(class_dir)
        for p in sorted(class_dir.iterdir()):
            if p.suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            if image_ids is not None and crop_image_id(p.name) not in image_ids:
                continue
            samples.append((p, local_idx))
            if p.name in boxes:
                stored[p] = boxes[p.name]
    return samples, stored


def image_subset_ids(cfg: DictConfig, split: str) -> set[str] | None:
    """Image ids of cfg.eval.image_subset (all | val_sel | test), None for `all`.

    The subsets partition DigitizePID's `val` images (eval.splits_file), so anything but
    `all` is only meaningful on split=val: on another split it would silently keep nothing.
    """
    subset = str(cfg.eval.get("image_subset", "all"))
    if subset == "all":
        return None
    path = Path(str(cfg.eval.get("splits_file", DEFAULT_SPLITS_FILE)))
    if not path.is_absolute():
        path = _REPO_ROOT / path
    with path.open() as f:
        subsets = {k: {str(i) for i in v} for k, v in json.load(f).items()}
    if subset not in subsets:
        raise ValueError(
            f"eval.image_subset={subset!r} is not 'all' or a key of {path}: {sorted(subsets)}"
        )
    if split != "val":
        raise ValueError(
            f"eval.image_subset={subset!r} partitions the val images, not split={split!r}"
        )
    return subsets[subset]


class _GroupCropDataset(Dataset):
    """Flat crops_fgc/<split>/<fine_class>/*.png restricted to one fgc_group.

    Labels are GROUP-LOCAL indices (0..len(fine_classes)-1, in `fine_classes` order —
    i.e. classmap.fine_of_group(group) order) rather than classmap's global fine ids,
    since each group trains its own classifier head sized to only its own fine
    classes.
    """

    def __init__(
        self,
        root: Path,
        fine_classes: list[str],
        transform,
        jitter: BoxJitter | None = None,
        image_ids: set[str] | None = None,
    ) -> None:
        self.fine_classes = fine_classes
        # stored: crop -> where it sits (utils.crop_box)
        self.samples, self.stored = scan_crops(root, fine_classes, image_ids)
        if jitter is not None and len(self.stored) < len(self.samples):
            missing = len(self.samples) - len(self.stored)
            raise ValueError(
                f"fgc.aug.box_jitter needs the stored context of every crop, but {missing} "
                f"crop(s) under {root} have no _boxes.csv entry: re-run pid-prep-crops"
            )
        self.transform = transform
        self.jitter = jitter

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        where = self.stored.get(path)
        if where is not None:
            # The classification view; at train time around a jittered box.
            img = view(img, where, self.jitter(where.box) if self.jitter else None)
        return self.transform(img), label


def _build_transform(cfg: DictConfig, split: str) -> transforms.Compose:
    imgsz = int(cfg.fgc.imgsz)
    ops: list = []
    if split == "train" and float(cfg.fgc.aug.subpart.p) > 0:
        # Per-subpart augment runs on the native crop, BEFORE Resize: it needs a clean
        # ink mask, which downscaling would blur (see utils.augment).
        ops.append(SubpartWarp.from_cfg(cfg.fgc.aug.subpart))
    degrade = ScanDegrade.from_cfg(cfg.fgc.aug.get("degrade")) if split == "train" else None
    if degrade is not None:
        # Scan-like degradation (off unless fgc.aug.degrade.p > 0), also on the native crop:
        # its limits follow the stroke width measured there (see utils.scan_degrade).
        ops.append(degrade)
    ops.append(transforms.Resize((imgsz, imgsz)))
    if split == "train":
        aug = cfg.fgc.aug
        ops += [
            # Orientation matters for P&ID symbols (an arrow/valve mirrored is a
            # different symbol) — hflip defaults to 0 in configs/config.yaml's fgc: section.
            transforms.RandomHorizontalFlip(p=float(aug.hflip)),
            transforms.RandomRotation(degrees=float(aug.rotate)),
            transforms.ColorJitter(brightness=float(aug.jitter), contrast=float(aug.jitter)),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]
    return transforms.Compose(ops)


def build_eval_transform(cfg: DictConfig) -> transforms.Compose:
    """Resize+normalize transform with NO augmentation — identical to what
    build_dataset uses for split in {val, test}. Exposed so pipeline.fgc_infer can apply the
    exact same preprocessing to detector crops at inference time (a train/inference
    preprocessing mismatch would silently hurt accuracy).
    """
    return _build_transform(cfg, split="val")


def excluded_classes(cfg: DictConfig) -> list[str]:
    """cfg.fgc.exclude_classes as a plain list; absent or null means none held out."""
    return list(cfg.fgc.get("exclude_classes", None) or [])


def build_dataset(
    cfg: DictConfig,
    split: str,
    group: str,
    fine_classes: list[str] | None = None,
    image_ids: set[str] | None = None,
) -> Dataset:
    """ImageFolder-equivalent restricted to `group`'s TRAINING fine classes, with
    GROUP-LOCAL label indices (0..n-1, in classmap.training_fine_classes order) -- NOT
    global fine ids, since each group gets its own classifier head sized to only its
    own fine classes. Held-out classes (cfg.fgc.exclude_classes) are absent from every
    split here: this dataset feeds a head that has no column for them.

    `fine_classes` overrides that list with a trained checkpoint's own column order
    (evaluation of `__global__.pt`), so labels follow the head that is evaluated, not
    whatever fgc.exclude_classes the eval command happens to carry. `image_ids` keeps only
    crops cut from those images (eval.image_subset, see `image_subset_ids`).
    """
    paths = DataPaths(root=Path(str(cfg.data_root)))
    if fine_classes is None:
        classmap = load_classmap(paths.classes_yaml)
        fine_classes = training_fine_classes(classmap, group, excluded_classes(cfg))
    root = paths.crops_fgc / split
    transform = _build_transform(cfg, split)
    jitter = BoxJitter.from_cfg(cfg.fgc.aug.get("box_jitter")) if split == "train" else None
    return _GroupCropDataset(root, fine_classes, transform, jitter, image_ids)


def build_balanced_sampler(dataset: Dataset, cfg: DictConfig) -> WeightedRandomSampler:
    """Inverse-class-frequency weights per sample, so a batch sees roughly one
    example per fine class regardless of natural frequency (long-tail handling,
    report §2.2.2/§2.2.3).
    """
    # cfg is accepted for interface symmetry with build_dataset/build_model; no
    # sampler-specific knobs beyond cfg.fgc.balanced_sampler, which the caller (train.py)
    # already checks before deciding whether to build a sampler at all.
    counts: dict[int, int] = {}
    for _, label in dataset.samples:
        counts[label] = counts.get(label, 0) + 1
    weights = [1.0 / counts[label] for _, label in dataset.samples]
    return WeightedRandomSampler(weights=weights, num_samples=len(dataset), replacement=True)
