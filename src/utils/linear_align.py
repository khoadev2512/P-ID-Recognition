"""Linear template -> crop alignment on frozen embeddings (PLAN_PHASE_C §2E.1, F0).

Only the template (prototype) side moves; queries stay as they are. Two maps:

- `domain_center`: each domain centered on its own mean (Liang et al., NeurIPS 2022, "Mind the
  Gap"; AdaBN): templates minus the template mean, unit. No labels.
- `ResidualRidge`: x' = normalize(x + b + (x - x_bar) A), fitted on (template, crop-class mean)
  pairs; b = mean residual, A = ridge on the centered residuals, solved in dual form (n pairs
  << D). lam = inf keeps only the shift b. Ridge pulls A toward 0, i.e. toward "no change".

Pipeline side (`oneshot.template_align`, PLAN_PHASE_C §2E.3): `align_templates` gives the
prototype rows in CL2N space for `domain_center` or `pair_shift`; the shift b of `pair_shift` is
computed at load time from a pairs file of RAW embeddings (`save_pairs`), so it follows whatever
CL2N center the run uses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

LAMBDAS = (0.01, 0.1, 1.0, 10.0, 100.0, math.inf)


def domain_center(templates: torch.Tensor) -> torch.Tensor:
    """(N, D) template embeddings -> unit embeddings centered on their own mean."""
    return F.normalize(templates - templates.mean(dim=0), dim=-1)


@dataclass
class ResidualRidge:
    lam: float
    b: torch.Tensor  # (D,)
    x_bar: torch.Tensor  # (D,)
    xc: torch.Tensor  # (n, D) centered training templates
    alpha: torch.Tensor | None  # (n, D) dual coefficients; None when lam = inf

    @classmethod
    def fit(cls, x: torch.Tensor, y: torch.Tensor, lam: float) -> ResidualRidge:
        """x (n, D) templates, y (n, D) targets, both unit; n >= 1."""
        if x.shape != y.shape or x.ndim != 2 or len(x) == 0:
            raise ValueError(f"x {tuple(x.shape)} and y {tuple(y.shape)} must be equal (n, D)")
        x, y = x.double(), y.double()
        r = y - x
        b = r.mean(dim=0)
        x_bar = x.mean(dim=0)
        xc = x - x_bar
        alpha = None
        if not math.isinf(lam):
            gram = xc @ xc.T + lam * torch.eye(len(x), dtype=x.dtype)
            alpha = torch.linalg.solve(gram, r - b)
        return cls(lam, b, x_bar, xc, alpha)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        z = x.double()
        out = z + self.b
        if self.alpha is not None:
            out = out + ((z - self.x_bar) @ self.xc.T) @ self.alpha
        return F.normalize(out, dim=-1).to(x.dtype)


def loo_fit_score(x: torch.Tensor, y: torch.Tensor, lam: float) -> float:
    """Mean cos(map fitted without pair j applied to x_j, y_j) over the n pairs (n >= 2)."""
    n = len(x)
    if n < 2:
        raise ValueError("leave-one-out needs at least 2 pairs")
    cos = []
    for j in range(n):
        keep = [i for i in range(n) if i != j]
        m = ResidualRidge.fit(x[keep], y[keep], lam)
        cos.append(float((m(x[j : j + 1]).double() @ y[j].double()).item()))
    return sum(cos) / n


def choose_lambda(x: torch.Tensor, y: torch.Tensor, grid=LAMBDAS) -> float:
    """The grid value with the best leave-one-pair-out fit; ties go to the larger lam (less
    change)."""
    scores = [(loo_fit_score(x, y, lam), lam) for lam in grid]
    best = max(s for s, _ in scores)
    return max(lam for s, lam in scores if s >= best - 1e-12)


# --------------------------------------------------------------------------- #
# pipeline: oneshot.template_align
# --------------------------------------------------------------------------- #

TEMPLATE_ALIGNS = ("domain_center", "pair_shift")
PAIRS_FORMAT_VERSION = 1
# Largest |difference| allowed between a pairs-file template and the run's prototype row (the
# pairs come from a CPU embedding, the prototypes may come from a GPU one).
PAIR_TEMPLATE_TOL = 1e-3


def _cl2n(x: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    return F.normalize(x - center, dim=-1)


def save_pairs(
    path: str | Path,
    names: list[str],
    templates: torch.Tensor,
    crop_views: list[torch.Tensor],
    meta: dict,
) -> Path:
    """(template, crops) pairs with RAW embeddings: names[k] (prototype name), templates (n, D),
    crop_views[k] (m_k, V, D) = the query views of class k's crops."""
    if len(names) != len(templates) or len(names) != len(crop_views) or not names:
        raise ValueError("names, templates and crop_views must have the same non-zero length")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": PAIRS_FORMAT_VERSION,
            "names": list(names),
            "templates": templates.detach().cpu().float(),
            "crop_views": [v.detach().cpu().float() for v in crop_views],
            "meta": dict(meta),
        },
        path,
    )
    return path


def load_pairs(path: str | Path) -> dict:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    version = payload.get("format_version")
    if version != PAIRS_FORMAT_VERSION:
        raise ValueError(f"{path}: pairs format_version={version}, expected {PAIRS_FORMAT_VERSION}")
    return payload


def pair_targets(crop_views: list[torch.Tensor], center: torch.Tensor) -> torch.Tensor:
    """(n, D) unit class targets y_k: the mean of the class' CL2N queries (each query = its
    views centered, unit, averaged, unit; the oneshot aggregate=mean path)."""
    out = []
    for views in crop_views:
        q = F.normalize(_cl2n(views, center).mean(dim=1), dim=-1)  # (m, D)
        out.append(F.normalize(q.mean(dim=0), dim=-1))
    return torch.stack(out)


def pair_shift(templates: torch.Tensor, crop_views: list[torch.Tensor], center: torch.Tensor):
    """b = mean over pairs of (y_k - x_k), x_k = the CL2N template (ResidualRidge, lam = inf)."""
    x = _cl2n(templates, center)
    return ResidualRidge.fit(x, pair_targets(crop_views, center), math.inf).b.to(templates.dtype)


def align_templates(
    raw: torch.Tensor, center: torch.Tensor, kind: str, shift: torch.Tensor | None = None
) -> torch.Tensor:
    """(N, D) raw prototype rows -> rows in CL2N space for `kind` (queries stay x - center)."""
    if kind == "domain_center":
        return domain_center(raw)
    if kind == "pair_shift":
        if shift is None:
            raise ValueError("pair_shift needs the shift b (pair_shift())")
        return F.normalize(_cl2n(raw, center) + shift.to(raw), dim=-1)
    raise ValueError(f"unknown template_align {kind!r}; expected one of {TEMPLATE_ALIGNS}")
