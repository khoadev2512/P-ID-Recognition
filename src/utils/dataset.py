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
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

from utils.classmap import load_classmap
from utils.paths import DataPaths

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

# Standard ImageNet normalization stats — required to feed a timm backbone that was
# pretrained on ImageNet (cfg.fgc.pretrained=true uses those pretrained weights).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class _GroupCropDataset(Dataset):
    """Flat crops_fgc/<split>/<fine_class>/*.png restricted to one fgc_group.

    Labels are GROUP-LOCAL indices (0..len(fine_classes)-1, in `fine_classes` order —
    i.e. classmap.fine_of_group(group) order) rather than classmap's global fine ids,
    since each group trains its own classifier head sized to only its own fine
    classes.
    """

    def __init__(self, root: Path, fine_classes: list[str], transform) -> None:
        self.fine_classes = fine_classes
        self.samples: list[tuple[Path, int]] = []
        for local_idx, fine_class in enumerate(fine_classes):
            class_dir = root / fine_class
            if not class_dir.is_dir():
                continue
            for p in sorted(class_dir.iterdir()):
                if p.suffix.lower() in _IMAGE_SUFFIXES:
                    self.samples.append((p, local_idx))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def _build_transform(cfg: DictConfig, split: str) -> transforms.Compose:
    imgsz = int(cfg.fgc.imgsz)
    ops: list = [transforms.Resize((imgsz, imgsz))]
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


def build_dataset(cfg: DictConfig, split: str, group: str) -> Dataset:
    """ImageFolder-equivalent restricted to `group`'s fine classes, with GROUP-LOCAL
    label indices (0..len(fine_of_group(group))-1, in classmap.fine_of_group(group)
    order) -- NOT global fine ids, since each group gets its own classifier head sized
    to only its own fine classes.
    """
    paths = DataPaths(root=Path(str(cfg.data_root)))
    classmap = load_classmap(paths.classes_yaml)
    fine_classes = classmap.fine_of_group(group)
    root = paths.crops_fgc / split
    transform = _build_transform(cfg, split)
    return _GroupCropDataset(root, fine_classes, transform)


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
