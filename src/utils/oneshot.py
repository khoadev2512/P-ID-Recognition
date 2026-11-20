"""One-shot ISO symbol recognition: template views, prototype directory, cosine match.

This is GĐ2/GĐ4 of the one-shot plan in docs/HANDOFF.md §6 — the piece that turns a
closed-set classifier trained on DigitizePID into a recognizer for the ISO 10628-2
vocabulary without ever training on ISO data:

    ISO template PNG --(frozen ArcFace encoder)--> prototype vector      [support side]
    detector crop    --(same frozen encoder)-----> query vector          [query side]
    label = argmax_name cos(query, prototype[name])                      [one-shot]
    cos < other_min_cosine  =>  "Other" (out-of-vocabulary)              [open set]

Why it works at all: ArcFace trains the backbone so that *angle* carries class
identity (see utils.model), which makes cosine-to-a-single-reference-vector a usable
decision rule even for a class the head never had a center for. OSSR-PID (arXiv
2109.03849, Table I) measures the same effect: ArcFace 85.78% vs. cross-entropy 77.07%
on exactly this one-template-per-class protocol.

Three design points worth keeping straight:

1. ADDING A SYMBOL IS ADDING A ROW. A PrototypeDirectory is just (names, L2-normalized
   embedding matrix); a new ISO symbol = one more template through the encoder, no
   retraining. That is the ISO-extensibility contribution (HANDOFF §1.3).

2. VIEWS DIFFER BY SIDE. The support side (templates) is canonical artwork, so its
   default is a single view; the query side is a detector crop that can be rotated and
   whose box is looser or tighter than the template's, so it gets multi-scale ×
   orientation views that VOTE (OSSR-PID technique #3). `scale` here is a box-fit
   factor, not a resolution: s>1 pads background around the symbol (a loose detector
   box), s<1 center-crops it (a tight one). Resolution alone would be a no-op — the
   model input is a fixed cfg.fgc.imgsz square.

3. PREPROCESSING MUST MATCH TRAINING. Callers pass the transform from
   utils.dataset.build_eval_transform, the same resize+normalize the encoder saw in
   training; a mismatch there silently costs accuracy and is invisible in the metrics.

The "Other" gate is deliberately the SAME shape as the one in pipeline.fgc_infer
(cosine below a threshold -> OTHER_CLASS_NAME) but is a SEPARATE threshold: there the
reference vectors are trained class centers, here they are single-template prototypes,
and those two cosine distributions are not interchangeable. Both numbers are
placeholders until calibrated on a val split (HANDOFF §7).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple, Protocol

import torch
import torch.nn.functional as F
from PIL import Image

from utils.classmap import OTHER_CLASS_NAME

# Bumped when the saved prototype payload changes shape, so an old file fails loudly
# instead of being silently misread.
PROTOTYPE_FORMAT_VERSION = 1

# Same, for the saved CL2N center (save_center / load_center).
CENTER_FORMAT_VERSION = 1

# P&ID artwork is black-on-white, so padding a looser box means adding white.
WHITE = (255, 255, 255)

# A class drawn in more than one form (ISO 10628-2 / BK-2020 X8075: butterfly valve form A and
# form B) gets one prototype per form, named `<fine_name>@<form>` next to the plain
# `<fine_name>` (PLAN_PHASE_C §2K.4 (i)). Every match is decided per CLASS: a class scores the
# best of its forms, and what is reported is the class name. A directory with no `@` name
# behaves exactly as before (class == prototype).
FORM_SEP = "@"
AGGREGATES = ("vote", "mean", "max")


def class_of(name: str) -> str:
    """The class a prototype name belongs to: `valve_butterfly@B` -> `valve_butterfly`."""
    return name.split(FORM_SEP, 1)[0]


class Encoder(Protocol):
    """What the one-shot path needs from a model: an embedding, not logits.

    utils.model.ArcFaceModel satisfies this. Declaring it as a Protocol (rather than
    typing against ArcFaceModel) keeps this module free of a hard dependency on the
    classifier head and lets tests substitute a trivial stand-in encoder — the logic
    here is testable without training anything.
    """

    def embed(self, images: torch.Tensor) -> torch.Tensor: ...


# ------------------------------------------------------------------ #
# Views (support and query side)
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class ViewSpec:
    """A deterministic set of views: every (scale, orientation) combination.

    scales       box-fit factors (see module docstring): 1.0 = the box as given,
                 >1 pads background around it, <1 center-crops into it.
    orientations multiples of 90° applied after scaling. Exact (lossless) rotations,
                 so no interpolation blur — and it is the query side's answer to
                 symbols drawn rotated in a real sheet.

    View order is scale-major, then orientation — fixed so a saved prototype and a
    query built from the same spec stay comparable.
    """

    scales: tuple[float, ...] = (1.0,)
    orientations: tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        if not self.scales:
            raise ValueError("ViewSpec.scales must not be empty")
        if not self.orientations:
            raise ValueError("ViewSpec.orientations must not be empty")
        for s in self.scales:
            if s <= 0:
                raise ValueError(f"ViewSpec.scales must be positive, got {s}")
        for o in self.orientations:
            if o % 90 != 0:
                raise ValueError(
                    f"ViewSpec.orientations must be multiples of 90 (lossless), got {o}"
                )

    @property
    def n_views(self) -> int:
        return len(self.scales) * len(self.orientations)

    @classmethod
    def from_cfg(cls, node) -> ViewSpec:
        """Build from a cfg node with `scales` / `orientations` lists (configs/config.yaml
        oneshot.template_views / oneshot.query_views)."""
        return cls(
            scales=tuple(float(s) for s in node.scales),
            orientations=tuple(int(o) for o in node.orientations),
        )

    def as_dict(self) -> dict:
        """Plain-dict form, stored in a directory's meta so a saved prototype records
        the views it was built from."""
        return {"scales": list(self.scales), "orientations": list(self.orientations)}


def refit_box(img: Image.Image, scale: float, background: tuple[int, int, int] = WHITE):
    """Simulate a detector box that is `scale`x the given one, keeping aspect ratio.

    scale > 1 -> pad each side by (scale-1)/2 of that dimension (a loose box: the
    symbol sits smaller inside more background). scale < 1 -> center-crop by the same
    fraction (a tight box: the symbol's outer edges are clipped). This is the real
    support/query gap in this project — an ISO template is cropped to its exact vector
    bbox, a YOLO box is not — so it is the axis worth being invariant to.
    """
    if scale == 1.0:
        return img.copy()
    w, h = img.size
    dx = int(round(abs(scale - 1.0) / 2.0 * w))
    dy = int(round(abs(scale - 1.0) / 2.0 * h))
    if scale > 1.0:
        canvas = Image.new("RGB", (w + 2 * dx, h + 2 * dy), background)
        canvas.paste(img.convert("RGB"), (dx, dy))
        return canvas
    # Tight box: never crop away everything — keep at least a 1px window.
    left, top = min(dx, (w - 1) // 2), min(dy, (h - 1) // 2)
    return img.crop((left, top, w - left, h - top))


def rotate_quarter(img: Image.Image, degrees: int):
    """Lossless rotation by a multiple of 90° (PIL transpose, no interpolation)."""
    turns = (degrees // 90) % 4
    if turns == 0:
        return img.copy()
    op = {
        1: Image.Transpose.ROTATE_90,
        2: Image.Transpose.ROTATE_180,
        3: Image.Transpose.ROTATE_270,
    }[turns]
    return img.transpose(op)


def make_views(
    img: Image.Image, spec: ViewSpec, background: tuple[int, int, int] = WHITE
) -> list[Image.Image]:
    """Expand one image into spec.n_views images, scale-major then orientation."""
    views: list[Image.Image] = []
    for scale in spec.scales:
        refit = refit_box(img, scale, background)
        for degrees in spec.orientations:
            views.append(rotate_quarter(refit, degrees))
    return views


# ------------------------------------------------------------------ #
# Embedding
# ------------------------------------------------------------------ #


@torch.no_grad()
def embed_images(
    encoder: Encoder,
    images: list[Image.Image],
    transform,
    device: torch.device | str = "cpu",
    batch_size: int = 64,
) -> torch.Tensor:
    """L2-normalized embeddings, (len(images), embed_dim), on the CPU.

    `transform` must be the SAME preprocessing the encoder trained under
    (utils.dataset.build_eval_transform) — see module docstring point 3. Normalizing
    here means every downstream cosine is a plain dot product and callers can't forget
    to normalize.
    """
    if not images:
        raise ValueError("embed_images got no images")
    out: list[torch.Tensor] = []
    for start in range(0, len(images), batch_size):
        batch = torch.stack(
            [transform(img.convert("RGB")) for img in images[start : start + batch_size]]
        ).to(device)
        out.append(F.normalize(encoder.embed(batch)).cpu())
    return torch.cat(out, dim=0)


def aggregate_views(view_embeddings: torch.Tensor) -> torch.Tensor:
    """Mean of (already normalized) view embeddings, renormalized -> (embed_dim,).

    The mean of unit vectors is not a unit vector, so it is renormalized; without that
    an averaged multi-view prototype would have a systematically smaller norm than a
    single-view one and the two would not be comparable under a shared threshold.
    """
    if view_embeddings.ndim != 2:
        raise ValueError(
            f"expected (V, D) view embeddings, got shape {tuple(view_embeddings.shape)}"
        )
    return F.normalize(view_embeddings.mean(dim=0), dim=0)


# ------------------------------------------------------------------ #
# Prototype directory + matching
# ------------------------------------------------------------------ #


@dataclass(frozen=True)
class Match:
    """One matching decision. `name` is OTHER_CLASS_NAME exactly when is_other."""

    name: str
    cosine: float
    is_other: bool = False
    n_votes: int = 1  # views backing this name (1 for single-view matching)


@dataclass
class PrototypeDirectory:
    """{fine_name -> unit embedding} plus the ISO bookkeeping needed to use it.

    `coarse` (parallel to `names`) lets GĐ4 restrict matching to the coarse class
    Stage 1 already predicted — matching a valve crop against only the valve
    prototypes is both faster and strictly more accurate than matching it against all
    of them, and it keeps the two stages composed the way the rest of the pipeline is.
    """

    names: list[str]
    embeddings: torch.Tensor  # (N, D), L2-normalized rows
    coarse: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.embeddings.ndim != 2:
            raise ValueError(f"embeddings must be (N, D), got shape {tuple(self.embeddings.shape)}")
        if len(self.names) != self.embeddings.shape[0]:
            raise ValueError(
                f"{len(self.names)} names vs. {self.embeddings.shape[0]} embedding rows"
            )
        if self.coarse and len(self.coarse) != len(self.names):
            raise ValueError(f"{len(self.coarse)} coarse entries vs. {len(self.names)} names")
        duplicates = {n for n in self.names if self.names.count(n) > 1}
        if duplicates:
            raise ValueError(f"duplicate prototype names: {sorted(duplicates)}")

    def __len__(self) -> int:
        return len(self.names)

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])

    @property
    def classes(self) -> list[str]:
        """Class names in first-appearance order (== `names` when no prototype has a form)."""
        return list(dict.fromkeys(class_of(n) for n in self.names))

    def _owner(self, device=None) -> torch.Tensor:
        """(N,) index into `classes` of every prototype."""
        index = {c: i for i, c in enumerate(self.classes)}
        return torch.tensor([index[class_of(n)] for n in self.names], device=device)

    def class_cosines(self, embedding: torch.Tensor) -> torch.Tensor:
        """cos to every CLASS (the best of its forms). (D,) -> (C,); (V, D) -> (V, C)."""
        cos = self.cosines(embedding)
        if len(self.classes) == len(self.names):
            return cos
        owner = self._owner(cos.device).expand_as(cos)
        out = torch.full(
            (*cos.shape[:-1], len(self.classes)), -torch.inf, device=cos.device, dtype=cos.dtype
        )
        return out.scatter_reduce(-1, owner, cos, reduce="amax")

    def subset_by_coarse(self, coarse_classes: set[str] | list[str]) -> PrototypeDirectory:
        """A directory holding only prototypes whose coarse class is in `coarse_classes`.

        Raises if this directory carries no coarse column (nothing to filter on) — a
        silently unfiltered result would look like a working restriction while actually
        matching against the whole vocabulary.
        """
        if not self.coarse:
            raise ValueError("this directory has no `coarse` column to subset by")
        wanted = set(coarse_classes)
        keep = [i for i, c in enumerate(self.coarse) if c in wanted]
        if not keep:
            raise ValueError(
                f"no prototypes with coarse in {sorted(wanted)}; "
                f"available: {sorted(set(self.coarse))}"
            )
        return PrototypeDirectory(
            names=[self.names[i] for i in keep],
            embeddings=self.embeddings[keep].clone(),
            coarse=[self.coarse[i] for i in keep],
            meta={**self.meta, "subset_of": sorted(wanted)},
        )

    def without(self, names: list[str] | set[str]) -> PrototypeDirectory:
        """A directory with the named prototypes removed (e.g. line-only templates that
        would absorb any crop containing a pipe). An unknown name is an error: a typo
        would otherwise leave the prototype in while looking like an exclusion."""
        drop = set(names)
        unknown = drop - set(self.names) - set(self.classes)
        if unknown:
            raise ValueError(f"unknown prototype name(s) {sorted(unknown)}; have {self.names}")
        # A class name drops every form of it; a `name@form` drops that form only.
        keep = [i for i, n in enumerate(self.names) if n not in drop and class_of(n) not in drop]
        if not keep:
            raise ValueError("excluding these prototypes would leave the directory empty")
        return PrototypeDirectory(
            names=[self.names[i] for i in keep],
            embeddings=self.embeddings[keep].clone(),
            coarse=[self.coarse[i] for i in keep] if self.coarse else [],
            meta={**self.meta, "excluded": sorted(drop)},
        )

    def centered(self, center: torch.Tensor) -> PrototypeDirectory:
        """The directory in CL2N space (center_embeddings): every row minus `center`,
        renormalized. Exact for a single template view; with several template views it
        centers their (renormalized) mean, an approximation."""
        return PrototypeDirectory(
            names=list(self.names),
            embeddings=center_embeddings(self.embeddings, center),
            coarse=list(self.coarse),
            meta={**self.meta, "centered": True},
        )

    def cosines(self, embedding: torch.Tensor) -> torch.Tensor:
        """cos to every prototype. (D,) -> (N,); (V, D) -> (V, N). Inputs are
        renormalized defensively so a caller passing raw embeddings still gets cosines."""
        query = F.normalize(embedding, dim=-1)
        # Prototypes are stored on the CPU (that is where embed_images returns and where
        # a loaded directory lands); move them to the query instead of the reverse, so a
        # caller holding a CUDA embedding is not a device-mismatch crash.
        protos = self.embeddings.to(device=query.device, dtype=query.dtype)
        return query @ protos.T if query.ndim == 2 else protos @ query

    def topk(self, embedding: torch.Tensor, k: int = 3) -> list[tuple[str, float]]:
        """(name, cosine) for the k nearest prototypes to a single (D,) embedding —
        the one-shot analogue of top-k accuracy, and useful for inspecting confusions."""
        cos = self.class_cosines(embedding)
        classes = self.classes
        k = min(k, len(classes))
        values, idx = torch.topk(cos, k)
        return [(classes[int(i)], float(v)) for v, i in zip(values, idx, strict=True)]

    def match(self, embedding: torch.Tensor, min_cosine: float | None = None) -> Match:
        """Nearest prototype to a single (D,) embedding, gated by `min_cosine`.

        min_cosine=None disables the gate (closed-set one-shot: always name the
        nearest). Otherwise a best cosine below it is reported as OTHER_CLASS_NAME,
        carrying the cosine that triggered the rejection so it can be logged/calibrated.
        """
        cos = self.class_cosines(embedding)
        best = int(torch.argmax(cos))
        best_cos = float(cos[best])
        if min_cosine is not None and best_cos < min_cosine:
            return Match(OTHER_CLASS_NAME, best_cos, is_other=True)
        return Match(self.classes[best], best_cos)

    def match_views(
        self,
        view_embeddings: torch.Tensor,
        min_cosine: float | None = None,
        strategy: str = "vote",
    ) -> Match:
        """Match a (V, D) stack of query views (OSSR-PID technique #3).

        strategy="vote" (default): each view names its own nearest prototype and the
        most frequent name wins; ties break on summed cosine (a 2-2 split goes to the
        stronger pair), and the reported cosine is the mean over the WINNING views only.
        Voting — not averaging — is what makes the orientation views work: a symbol
        drawn rotated 90° matches its upright template in exactly one of the four
        orientation views, and a majority can survive that whereas a mean embedding
        would be dragged toward the three uninformative ones.

        A view only gets a vote if its own best cosine clears `min_cosine`. That
        matters for precisely the case the orientation views exist for: for an
        asymmetric symbol only the correctly-oriented view is confident, and the other
        three are arbitrary nearest-neighbour noise that could otherwise out-vote it
        2-to-1. The gate is already the project's statement of "this is a real match",
        so reusing it as the vote's admission test needs no second threshold — and it
        means the winning cosine is >= min_cosine by construction, so there is no
        separate rejection step afterwards. When NO view clears the gate the crop is
        out-of-vocabulary: Other, with n_votes=0 and the best cosine any view achieved
        (the number to calibrate the threshold on). min_cosine=None admits every view
        (closed-set: pure majority).

        strategy="mean": collapse the views into one prototype-style vector first
        (aggregate_views) and match that, gate included. Cheaper, and the right choice
        when the views are mild scale jitter only — it is the ablation counterpart to
        voting, and the one to use if a future orientation-invariant encoder makes the
        per-view disagreement disappear.

        strategy="max": a class scores its best cosine over the views (TI-pooling, Laptev et
        al. CVPR 2016: max over transformed copies, invariance without blending them); the
        best class wins, gated on that cosine (so a tau sweep over it is exact). n_votes is
        the number of views whose own nearest class it is. PLAN_PHASE_C §2K.4 (ii).

        Every strategy decides per class (`class_cosines`): a class drawn in several forms
        scores the best of them, and the reported name is the class.
        """
        if view_embeddings.ndim != 2:
            raise ValueError(
                f"expected (V, D) view embeddings, got shape {tuple(view_embeddings.shape)}"
            )
        if strategy == "mean":
            return self.match(aggregate_views(view_embeddings), min_cosine)
        if strategy not in AGGREGATES:
            raise ValueError(f"unknown strategy={strategy!r}; expected one of {AGGREGATES}")

        cos = self.class_cosines(view_embeddings)  # (V, C)
        if strategy == "max":
            per_class = cos.max(dim=0).values
            best = int(torch.argmax(per_class))
            best_cos = float(per_class[best])
            if min_cosine is not None and best_cos < min_cosine:
                return Match(OTHER_CLASS_NAME, best_cos, is_other=True, n_votes=0)
            n_best = int((cos.argmax(dim=1) == best).sum())
            return Match(self.classes[best], best_cos, n_votes=n_best)

        best_cos, best_idx = cos.max(dim=1)

        voters = range(len(best_idx))
        if min_cosine is not None:
            voters = [v for v in voters if float(best_cos[v]) >= min_cosine]
            if not voters:
                # Every view's best match is below the gate -> out-of-vocabulary. Report
                # the strongest cosine any view reached, which is the quantity to
                # calibrate oneshot.other_min_cosine against.
                return Match(OTHER_CLASS_NAME, float(best_cos.max()), is_other=True, n_votes=0)

        per_name: dict[int, list[float]] = defaultdict(list)
        for v in voters:
            per_name[int(best_idx[v])].append(float(best_cos[v]))
        # Most votes; tie -> larger summed cosine.
        proto_idx, winning_cosines = max(per_name.items(), key=lambda kv: (len(kv[1]), sum(kv[1])))
        mean_cos = sum(winning_cosines) / len(winning_cosines)
        return Match(self.classes[proto_idx], mean_cos, n_votes=len(winning_cosines))

    # -------------------------------------------------------------- #
    # Persistence
    # -------------------------------------------------------------- #

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": PROTOTYPE_FORMAT_VERSION,
                "names": self.names,
                "coarse": self.coarse,
                "embeddings": self.embeddings,
                "meta": self.meta,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> PrototypeDirectory:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No prototype directory at {path} (build it with pid-iso-prototypes)."
            )
        payload = torch.load(path, map_location=map_location, weights_only=False)
        version = int(payload.get("format_version", 0))
        if version != PROTOTYPE_FORMAT_VERSION:
            raise ValueError(
                f"{path}: prototype format_version={version}, this code expects "
                f"{PROTOTYPE_FORMAT_VERSION}; rebuild it with pid-iso-prototypes."
            )
        return cls(
            names=list(payload["names"]),
            embeddings=payload["embeddings"],
            coarse=list(payload.get("coarse", [])),
            meta=dict(payload.get("meta", {})),
        )


# --------------------------------------------------------------------------- #
# CL2N centering (SimpleShot, Wang et al. 2019, arXiv 1911.04623)
# --------------------------------------------------------------------------- #
# Pooled ResNet features are post-ReLU, so every embedding lies in the positive orthant and
# all cosines are high; the prototype nearest the mean query direction then wins most crops
# (a hub: the ImageNet-only encoder sent 59% of the P1 crops to flow_direction). Subtracting
# the mean embedding of SEEN-class training crops before the L2 normalization removes that
# shared direction. The center comes from DigitizePID train crops of classes the protocol
# never tests (pipeline.oneshot_eval protocol `center`), so it carries no label of a tested
# class. The mean of the prototypes is NOT a substitute: it is the template domain's mean,
# not the query domain's (measured 2026-11-15: it made the hub worse).


def center_embeddings(embeddings: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """(x - center) renormalized along the last dim; (D,) or (N, D) -> same shape."""
    if center.ndim != 1 or embeddings.shape[-1] != center.shape[0]:
        raise ValueError(
            f"center of shape {tuple(center.shape)} does not fit embeddings of shape "
            f"{tuple(embeddings.shape)}"
        )
    return F.normalize(embeddings - center.to(embeddings), dim=-1)


def save_center(path: str | Path, center: torch.Tensor, meta: dict) -> Path:
    """Write a CL2N center; `meta` must carry `encoder_sha256` (checked by load_center's
    caller against the encoder actually loaded)."""
    if "encoder_sha256" not in meta:
        raise ValueError("center meta needs encoder_sha256")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"format_version": CENTER_FORMAT_VERSION, "center": center.detach().cpu(), "meta": meta},
        path,
    )
    return path


def load_center(path: str | Path) -> tuple[torch.Tensor, dict]:
    """(center (D,), meta) from save_center."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No center file at {path} (build it with eval.mode=oneshot "
            "++eval.oneshot.protocol=center)"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    version = int(payload.get("format_version", 0))
    if version != CENTER_FORMAT_VERSION:
        raise ValueError(
            f"{path}: center format_version={version}, this code expects {CENTER_FORMAT_VERSION}"
        )
    return payload["center"].float(), dict(payload["meta"])


def build_prototypes(
    encoder: Encoder,
    templates: dict[str, Image.Image],
    transform,
    spec: ViewSpec | None = None,
    coarse_of: dict[str, str] | None = None,
    device: torch.device | str = "cpu",
    background: tuple[int, int, int] = WHITE,
    meta: dict | None = None,
) -> PrototypeDirectory:
    """One prototype per template: views -> embeddings -> mean -> unit vector.

    `templates` maps fine_name -> the single template image for that class (one-shot:
    exactly one image each). Iteration follows `templates` insertion order, so the row
    order of the resulting directory is reproducible from the vocab file's row order.
    """
    if not templates:
        raise ValueError("build_prototypes got no templates")
    spec = spec or ViewSpec()
    names = list(templates.keys())
    vectors = [
        aggregate_views(
            embed_images(encoder, make_views(templates[name], spec, background), transform, device)
        )
        for name in names
    ]
    return PrototypeDirectory(
        names=names,
        embeddings=torch.stack(vectors),
        coarse=[coarse_of[n] for n in names] if coarse_of else [],
        meta={**(meta or {}), "template_views": spec.as_dict(), "n_prototypes": len(names)},
    )


# ------------------------------------------------------------------ #
# Protocol P1: the held-out class lists
# ------------------------------------------------------------------ #


class Holdout(NamedTuple):
    """The B5 lists. `h_in` + `h_out` are the P1 test classes (`tested`); `h_extra` are
    VARIANT classes of an `h_in` symbol, left out of training but never tested (`excluded`
    is what fgc.exclude_classes gets for R1a)."""

    h_in: list[str]
    h_out: list[str]
    h_extra: list[str]

    @property
    def tested(self) -> list[str]:
        return self.h_in + self.h_out

    @property
    def excluded(self) -> list[str]:
        return self.h_in + self.h_out + self.h_extra


def holdout_classes(cfg) -> Holdout:
    """(h_in, h_out, h_extra) from cfg.oneshot.holdout — the unseen-class lists of P1.

    The single source for both the leave-out training run (fgc.exclude_classes is
    `excluded` = h_in + h_out + h_extra) and the P1 evaluation (`tested` = h_in + h_out),
    so the two can never disagree about which classes the encoder did not see. A class in
    two lists is an error: it cannot be an ISO symbol to recognize and an out-of-vocabulary
    symbol to reject at once, and an h_extra class is never tested. Names are checked
    against classes.yaml where they are used (utils.classmap.training_fine_classes), and
    the VARIANT rule for h_extra in scripts/check_b5.py, since both need more than cfg.
    """
    node = cfg.oneshot.get("holdout", None) or {}
    lists = {k: list(node.get(k, None) or []) for k in ("h_in", "h_out", "h_extra")}
    for name, lst in lists.items():
        dupes = sorted({c for c in lst if lst.count(c) > 1})
        if dupes:
            raise ValueError(f"oneshot.holdout.{name} lists {dupes} more than once")
    names = list(lists)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            both = sorted(set(lists[a]) & set(lists[b]))
            if both:
                raise ValueError(f"oneshot.holdout: {both} in both {a} and {b}")
    return Holdout(**lists)
