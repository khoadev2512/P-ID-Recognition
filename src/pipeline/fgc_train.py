"""Train the fine-grained classifier(s) on symbol crops.

Trains one head per fgc_group over derived/crops_fgc/train, validates on
crops_fgc/val. Owns the loop (optimizer, scheduler, checkpoint) — kept minimal and
config-driven for the ablations in §4.2 / roadmap Phase 3.

Loop shape mirrors PID_Symbol_Detection's `utils.few_shot_trainer.FewShotTrainer`
(per-epoch train/val pass, running loss/accuracy tracking, save-best-by-val-metric) —
same shape, different loss/model: plain supervised CE/focal classification instead of
triplet-loss few-shot embedding.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from pipeline.base import BasePipeline
from utils.cli import CONFIG_DIR
from utils.dataset import build_balanced_sampler, build_dataset
from utils.model import build_loss, build_model

logger = logging.getLogger(__name__)


class FGCTrainPipeline(BasePipeline):
    """Trains one ResNet-34 (or configured backbone) head per fgc_group."""

    def validate(self) -> bool:
        train_dir = self.paths.crops_fgc / "train"
        if not train_dir.is_dir() or not any(train_dir.iterdir()):
            raise FileNotFoundError(
                f"Expected FGC training crops at {train_dir} (run pid-prep-crops first)."
            )
        return True

    def run(self) -> None:
        self.validate()
        device = torch.device(
            "cuda" if str(self.cfg.device) == "cuda" and torch.cuda.is_available() else "cpu"
        )
        ckpt_root = self.output_dir / "fgc_checkpoints"
        ckpt_root.mkdir(parents=True, exist_ok=True)

        # classmap.fgc_groups (not cfg.fgc.groups) is the validated source of truth —
        # cfg.fgc.groups is just the human-authored expectation of the same list.
        for group in self.classmap.fgc_groups:
            self._train_group(group, device, ckpt_root)

    def _train_group(self, group: str, device: torch.device, ckpt_root: Path) -> None:
        cfg = self.cfg
        fine_classes = self.classmap.fine_of_group(group)
        if not fine_classes:
            logger.warning("fgc_group=%s has no fine classes in classes.yaml; skipping.", group)
            return

        train_ds = build_dataset(cfg, "train", group)
        val_ds = build_dataset(cfg, "val", group)
        if len(train_ds) == 0:
            logger.warning(
                "fgc_group=%s has no training crops under %s; skipping.",
                group,
                self.paths.crops_fgc / "train",
            )
            return

        sampler = build_balanced_sampler(train_ds, cfg) if cfg.fgc.balanced_sampler else None
        train_loader = DataLoader(
            train_ds, batch_size=int(cfg.fgc.batch), sampler=sampler, shuffle=(sampler is None)
        )
        val_loader = DataLoader(val_ds, batch_size=int(cfg.fgc.batch), shuffle=False)

        model = build_model(cfg, group, len(fine_classes)).to(device)

        class_counts = [0] * len(fine_classes)
        for _, label in train_ds.samples:
            class_counts[label] += 1

        criterion = build_loss(cfg, class_counts).to(device)
        # No LR is configured anywhere for fgc (configs/config.yaml's fgc: section has no lr
        # field) -- 1e-3 is a standard Adam default and a sane choice for fine-tuning a
        # pretrained backbone at this crop scale; document it here since it's not
        # config-driven.
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        best_val_acc = -1.0
        ckpt_path = ckpt_root / f"{group}.pt"
        epochs = int(cfg.fgc.epochs)

        logger.info(
            "[fgc/%s] %d fine classes, %d train crops, %d val crops",
            group,
            len(fine_classes),
            len(train_ds),
            len(val_ds),
        )

        for epoch in range(epochs):
            train_loss, train_acc = self._train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = self._eval_one_epoch(model, val_loader, criterion, device)
            logger.info(
                "[fgc/%s] epoch %d/%d train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
                group,
                epoch + 1,
                epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "val_acc": val_acc,
                        "fine_classes": fine_classes,
                    },
                    ckpt_path,
                )
                logger.info("[fgc/%s] new best val_acc=%.4f -> saved %s", group, val_acc, ckpt_path)

        logger.info(
            "[fgc/%s] training finished, best val_acc=%.4f (%s)", group, best_val_acc, ckpt_path
        )

    @staticmethod
    def _train_one_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / total if total else 0.0
        accuracy = correct / total if total else 0.0
        return avg_loss, accuracy

    @staticmethod
    @torch.no_grad()
    def _eval_one_epoch(model, loader, criterion, device) -> tuple[float, float]:
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / total if total else 0.0
        accuracy = correct / total if total else 0.0
        return avg_loss, accuracy


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    FGCTrainPipeline(cfg).run()


if __name__ == "__main__":
    main()
