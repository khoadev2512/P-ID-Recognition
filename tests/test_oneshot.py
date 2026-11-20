"""Unit tests for the one-shot core (utils.oneshot): views, prototypes, cosine match,
"Other" gate and view voting.

The whole point of this module is that the one-shot LOGIC is testable without training
anything and without timm: every test runs against `_FakeEncoder`, a fixed linear
projection of downsampled pixels. It is a deterministic embedding — not a good one, but
the matching rules (argmax cosine, majority vote, tie-break, reject threshold,
persistence) are independent of embedding quality, and that is exactly what is being
pinned down here. The quality question belongs to the eval stage on real crops.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402

from utils.classmap import OTHER_CLASS_NAME  # noqa: E402
from utils.oneshot import (  # noqa: E402
    PROTOTYPE_FORMAT_VERSION,
    Match,
    PrototypeDirectory,
    ViewSpec,
    aggregate_views,
    build_prototypes,
    embed_images,
    make_views,
    refit_box,
    rotate_quarter,
)

_DIM = 16
_SIDE = 8


class _FakeEncoder(torch.nn.Module):
    """Deterministic stand-in for ArcFaceModel: fixed random projection of pixels.

    Satisfies the utils.oneshot.Encoder protocol (`embed` only). Seeded so two test
    runs agree, and linear so "same image in -> same vector out" holds exactly, which
    is what lets a test assert an exact-template match has cosine 1.0.
    """

    def __init__(self, dim: int = _DIM, side: int = _SIDE) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.register_buffer("w", torch.randn(3 * side * side, dim, generator=generator))

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        return images.flatten(1) @ self.w


def _transform(side: int = _SIDE):
    """Minimal resize+tensor transform, standing in for build_eval_transform."""
    return transforms.Compose([transforms.Resize((side, side)), transforms.ToTensor()])


def _solid(color: tuple[int, int, int], size: tuple[int, int] = (20, 12)) -> Image.Image:
    return Image.new("RGB", size, color)


def _glyph(seed: int, size: tuple[int, int] = (20, 12)) -> Image.Image:
    """A deterministic pseudo-random image — stands in for a distinct symbol drawing."""
    generator = torch.Generator().manual_seed(seed)
    pixels = (torch.rand(size[1], size[0], 3, generator=generator) * 255).to(torch.uint8)
    return Image.fromarray(pixels.numpy(), mode="RGB")


# ------------------------------------------------------------------ #
# ViewSpec
# ------------------------------------------------------------------ #


def test_viewspec_defaults_are_a_single_canonical_view():
    spec = ViewSpec()
    assert spec.scales == (1.0,) and spec.orientations == (0,)
    assert spec.n_views == 1


def test_viewspec_n_views_is_the_cross_product():
    assert ViewSpec(scales=(1.0, 1.25), orientations=(0, 90, 180, 270)).n_views == 8


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scales": ()},
        {"orientations": ()},
        {"scales": (0.0,)},
        {"scales": (-1.0,)},
        # 45° would need interpolation; only lossless quarter turns are allowed.
        {"orientations": (45,)},
    ],
)
def test_viewspec_rejects_invalid_specs(kwargs):
    with pytest.raises(ValueError):
        ViewSpec(**kwargs)


def test_viewspec_from_cfg_and_round_trip_through_as_dict():
    from omegaconf import OmegaConf

    node = OmegaConf.create({"scales": [1.0, 1.5], "orientations": [0, 180]})
    spec = ViewSpec.from_cfg(node)
    assert spec.scales == (1.0, 1.5) and spec.orientations == (0, 180)
    assert spec.as_dict() == {"scales": [1.0, 1.5], "orientations": [0, 180]}


# ------------------------------------------------------------------ #
# Views
# ------------------------------------------------------------------ #


def test_refit_box_scale_one_is_an_unchanged_copy():
    img = _glyph(1)
    out = refit_box(img, 1.0)
    assert out.size == img.size
    assert list(out.getdata()) == list(img.getdata())


def test_refit_box_above_one_pads_with_background_keeping_aspect():
    """A loose detector box = the same symbol with background around it. Padding is
    proportional per axis, so the aspect ratio of the content is preserved."""
    img = _solid((0, 0, 0), size=(20, 10))
    out = refit_box(img, 1.5, background=(255, 255, 255))
    # Pad = (1.5-1)/2 = 0.25 of each axis, per side: dx = 0.25*20 = 5,
    # dy = 0.25*10 = 2.5 -> 2 (Python rounds .5 to even), so 30x14.
    assert out.size == (30, 14)
    assert out.getpixel((0, 0)) == (255, 255, 255)  # corner is background
    assert out.getpixel((15, 7)) == (0, 0, 0)  # centre is still the symbol


def test_refit_box_below_one_crops_inward():
    img = _solid((0, 0, 0), size=(20, 10))
    out = refit_box(img, 0.5)
    assert out.size == (20 - 2 * 5, 10 - 2 * 2)


def test_refit_box_tight_crop_never_empties_the_image():
    """A pathologically small scale must still yield a usable (non-zero) image rather
    than a zero-size crop that would blow up in the transform."""
    out = refit_box(_solid((0, 0, 0), size=(5, 5)), 0.01)
    assert out.size[0] >= 1 and out.size[1] >= 1


def test_rotate_quarter_is_lossless_and_periodic():
    img = _glyph(2)
    assert list(rotate_quarter(img, 0).getdata()) == list(img.getdata())
    assert list(rotate_quarter(img, 360).getdata()) == list(img.getdata())
    # 90° on a non-square image swaps the side lengths.
    assert rotate_quarter(img, 90).size == (img.size[1], img.size[0])
    # Four quarter turns return the original exactly (no interpolation drift).
    four = img
    for _ in range(4):
        four = rotate_quarter(four, 90)
    assert list(four.getdata()) == list(img.getdata())


def test_make_views_count_and_scale_major_order():
    img = _glyph(3)
    spec = ViewSpec(scales=(1.0, 1.5), orientations=(0, 90))
    views = make_views(img, spec)
    assert len(views) == spec.n_views == 4
    # Scale-major: the first two views are scale 1.0 (orientations 0, 90), the last two
    # are scale 1.5 -- so views[0] is the untouched image and views[2] is the padded one.
    assert views[0].size == img.size
    assert views[2].size[0] > img.size[0]


# ------------------------------------------------------------------ #
# Embedding
# ------------------------------------------------------------------ #


def test_embed_images_returns_unit_rows():
    embeddings = embed_images(_FakeEncoder(), [_glyph(i) for i in range(5)], _transform())
    assert embeddings.shape == (5, _DIM)
    assert torch.allclose(embeddings.norm(dim=1), torch.ones(5), atol=1e-5)


def test_embed_images_batches_without_changing_the_result():
    """Batching is an implementation detail; a smaller batch_size must not change
    anything (it would silently if normalization were done per batch over the batch)."""
    encoder, images = _FakeEncoder(), [_glyph(i) for i in range(7)]
    one_shot = embed_images(encoder, images, _transform(), batch_size=64)
    chunked = embed_images(encoder, images, _transform(), batch_size=2)
    assert torch.allclose(one_shot, chunked, atol=1e-6)


def test_embed_images_rejects_an_empty_list():
    with pytest.raises(ValueError):
        embed_images(_FakeEncoder(), [], _transform())


def test_aggregate_views_renormalizes_the_mean():
    """The mean of unit vectors is shorter than a unit vector; without renormalization a
    multi-view prototype and a single-view one could not share a cosine threshold."""
    views = torch.nn.functional.normalize(torch.randn(4, _DIM), dim=1)
    out = aggregate_views(views)
    assert out.shape == (_DIM,)
    assert pytest.approx(1.0, abs=1e-5) == float(out.norm())


def test_aggregate_views_of_one_view_is_that_view():
    view = torch.nn.functional.normalize(torch.randn(1, _DIM), dim=1)
    assert torch.allclose(aggregate_views(view), view[0], atol=1e-6)


def test_aggregate_views_rejects_a_1d_input():
    with pytest.raises(ValueError):
        aggregate_views(torch.randn(_DIM))


# ------------------------------------------------------------------ #
# PrototypeDirectory: construction + validation
# ------------------------------------------------------------------ #


def _directory(n: int = 4, dim: int = _DIM, coarse: list[str] | None = None):
    generator = torch.Generator().manual_seed(7)
    embeddings = torch.nn.functional.normalize(torch.randn(n, dim, generator=generator), dim=1)
    return PrototypeDirectory(
        names=[f"sym_{i}" for i in range(n)],
        embeddings=embeddings,
        coarse=coarse or [],
    )


def test_directory_len_and_dim():
    directory = _directory(n=5)
    assert len(directory) == 5
    assert directory.dim == _DIM


def test_directory_rejects_name_embedding_count_mismatch():
    with pytest.raises(ValueError, match="embedding rows"):
        PrototypeDirectory(names=["a", "b"], embeddings=torch.randn(3, _DIM))


def test_directory_rejects_non_2d_embeddings():
    with pytest.raises(ValueError, match=r"\(N, D\)"):
        PrototypeDirectory(names=["a"], embeddings=torch.randn(_DIM))


def test_directory_rejects_duplicate_names():
    """Two rows with one name would collapse into a single addressable prototype and
    silently make one of the two ISO symbols unrecognizable."""
    with pytest.raises(ValueError, match="duplicate"):
        PrototypeDirectory(names=["a", "a"], embeddings=torch.randn(2, _DIM))


def test_directory_rejects_coarse_length_mismatch():
    with pytest.raises(ValueError, match="coarse"):
        PrototypeDirectory(names=["a", "b"], embeddings=torch.randn(2, _DIM), coarse=["valve"])


# ------------------------------------------------------------------ #
# PrototypeDirectory: matching
# ------------------------------------------------------------------ #


def test_cosines_shapes_for_single_and_multi_view_queries():
    directory = _directory(n=4)
    assert directory.cosines(torch.randn(_DIM)).shape == (4,)
    assert directory.cosines(torch.randn(3, _DIM)).shape == (3, 4)


def test_cosine_to_own_prototype_is_one():
    directory = _directory(n=4)
    for i, name in enumerate(directory.names):
        match = directory.match(directory.embeddings[i])
        assert match.name == name
        assert pytest.approx(1.0, abs=1e-5) == match.cosine
        assert not match.is_other


def test_match_renormalizes_an_unnormalized_query():
    """Cosine is scale-invariant, so a caller handing over a raw (unnormalized)
    embedding must get the same answer as one who normalized first."""
    directory = _directory(n=4)
    scaled = directory.embeddings[2] * 17.0
    assert directory.match(scaled).name == "sym_2"
    assert pytest.approx(1.0, abs=1e-5) == directory.match(scaled).cosine


def test_match_with_no_threshold_never_rejects():
    """Closed-set one-shot: min_cosine=None always names the nearest prototype, even
    for a query that matches nothing well."""
    directory = _directory(n=4)
    match = directory.match(torch.randn(_DIM), min_cosine=None)
    assert match.name in directory.names
    assert not match.is_other


def test_other_gate_rejects_a_far_query():
    """An embedding below the threshold is reported as Other, carrying the cosine that
    caused the rejection (so it can be logged and the threshold calibrated)."""
    directory = _directory(n=4)
    # A threshold above 1.0 can never be met -> everything is out-of-vocabulary.
    match = directory.match(directory.embeddings[0], min_cosine=1.5)
    assert match.is_other
    assert match.name == OTHER_CLASS_NAME
    assert pytest.approx(1.0, abs=1e-5) == match.cosine


def test_other_gate_accepts_a_query_exactly_at_the_threshold():
    """The gate is `cos < min_cosine`, so a cosine EQUAL to the threshold is accepted --
    pinned down because an off-by-one-sided comparison here shifts every reported
    false-reject rate."""
    directory = _directory(n=2)
    best = float(directory.cosines(directory.embeddings[0]).max())
    assert not directory.match(directory.embeddings[0], min_cosine=best).is_other


def test_topk_is_ordered_and_clipped_to_the_directory_size():
    directory = _directory(n=4)
    top = directory.topk(directory.embeddings[1], k=3)
    assert [name for name, _ in top][0] == "sym_1"
    assert [cos for _, cos in top] == sorted((cos for _, cos in top), reverse=True)
    # k larger than the vocabulary returns the whole vocabulary, not an error.
    assert len(directory.topk(directory.embeddings[0], k=99)) == 4


# ------------------------------------------------------------------ #
# PrototypeDirectory: view voting
# ------------------------------------------------------------------ #


def test_match_views_vote_follows_the_majority_not_the_mean():
    """Three views point at sym_0, one at sym_1 -> sym_0 wins with 3 votes, and the
    reported cosine averages only the WINNING views."""
    directory = _directory(n=4)
    views = torch.stack(
        [
            directory.embeddings[0],
            directory.embeddings[0],
            directory.embeddings[0],
            directory.embeddings[1],
        ]
    )
    match = directory.match_views(views, strategy="vote")
    assert match.name == "sym_0"
    assert match.n_votes == 3
    assert pytest.approx(1.0, abs=1e-5) == match.cosine


def test_match_views_tie_breaks_on_summed_cosine():
    """A 2-2 split goes to the pair with the stronger total similarity: here two exact
    hits on sym_1 vs. two weakened (blended) hits on sym_0."""
    directory = _directory(n=4)
    weak = torch.nn.functional.normalize(
        directory.embeddings[0] + 0.9 * directory.embeddings[2], dim=0
    )
    views = torch.stack([weak, weak, directory.embeddings[1], directory.embeddings[1]])
    match = directory.match_views(views, strategy="vote")
    assert match.name == "sym_1"
    assert match.n_votes == 2


def test_match_views_rejects_when_no_view_clears_the_gate():
    """No view admitted -> Other, n_votes=0, and the reported cosine is the best any
    view reached (the number oneshot.other_min_cosine gets calibrated against)."""
    directory = _directory(n=4)
    views = directory.embeddings[:1].repeat(3, 1)
    match = directory.match_views(views, min_cosine=1.5, strategy="vote")
    assert match.is_other and match.name == OTHER_CLASS_NAME
    assert match.n_votes == 0
    assert pytest.approx(1.0, abs=1e-5) == match.cosine


def test_only_views_clearing_the_gate_get_a_vote():
    """One confident view must beat a bloc of uninformative ones.

    Two views sit exactly on sym_1 (cosine 1.0); two agree with each other on sym_0 but
    only weakly. With a gate between those two confidence levels the weak pair is not
    admitted at all, so sym_1 wins 2-0 instead of tying 2-2. Without the admission rule
    a rotated asymmetric symbol would routinely lose to its own noise views.
    """
    directory = _directory(n=4)
    weak = torch.nn.functional.normalize(directory.embeddings[0] + directory.embeddings[2], dim=0)
    weak_cos = float(directory.cosines(weak).max())
    assert weak_cos < 0.95, "fixture must keep the decoy views genuinely unconfident"

    views = torch.stack([weak, weak, directory.embeddings[1], directory.embeddings[1]])
    gated = directory.match_views(views, min_cosine=0.95, strategy="vote")
    assert gated.name == "sym_1" and gated.n_votes == 2

    # With the gate off it is a 2-2 tie resolved by summed cosine -- still sym_1 here,
    # but on votes alone, which is the behaviour the gate is protecting against.
    ungated = directory.match_views(views, min_cosine=None, strategy="vote")
    assert ungated.n_votes == 2


def test_match_views_mean_strategy_matches_the_aggregated_vector():
    directory = _directory(n=4)
    views = torch.nn.functional.normalize(torch.randn(5, _DIM), dim=1)
    by_mean = directory.match_views(views, strategy="mean")
    direct = directory.match(aggregate_views(views))
    assert by_mean.name == direct.name
    assert pytest.approx(direct.cosine, abs=1e-6) == by_mean.cosine


def test_match_views_rejects_unknown_strategy_and_bad_shape():
    directory = _directory(n=4)
    with pytest.raises(ValueError, match="strategy"):
        directory.match_views(torch.randn(2, _DIM), strategy="median")
    with pytest.raises(ValueError, match=r"\(V, D\)"):
        directory.match_views(torch.randn(_DIM))


def test_voting_recovers_a_rotated_query_that_a_single_view_misses():
    """The reason the orientation views exist (OSSR-PID technique #3).

    Prototypes are built from upright templates only; the query is the SAME symbol
    drawn rotated 90°. The upright view of that query is not a match, but one of the
    four orientation views un-rotates it back onto its template, and the vote picks it
    up. Asserted end-to-end through the encoder, not on synthetic embeddings.
    """
    encoder, transform = _FakeEncoder(), _transform()
    templates = {f"sym_{i}": _glyph(100 + i, size=(16, 16)) for i in range(5)}
    directory = build_prototypes(encoder, templates, transform, spec=ViewSpec())

    target = "sym_2"
    rotated_query = rotate_quarter(templates[target], 90)

    upright_only = directory.match(
        embed_images(encoder, [rotated_query], transform)[0], min_cosine=0.99
    )
    assert upright_only.is_other, "a rotated query should NOT match upright-only at 0.99"

    # Four orientation views: exactly one un-rotates the query back onto its template
    # (cosine 1.0, the only one admitted by the 0.99 gate), so the vote is decisive.
    query_views = make_views(rotated_query, ViewSpec(orientations=(0, 90, 180, 270)))
    voted = directory.match_views(
        embed_images(encoder, query_views, transform), min_cosine=0.99, strategy="vote"
    )
    assert voted.name == target and not voted.is_other
    assert voted.n_votes == 1


# ------------------------------------------------------------------ #
# PrototypeDirectory: coarse subsetting
# ------------------------------------------------------------------ #


def test_subset_by_coarse_keeps_only_the_requested_classes():
    directory = _directory(n=4, coarse=["valve", "valve", "pump", "vessel"])
    subset = directory.subset_by_coarse({"valve"})
    assert subset.names == ["sym_0", "sym_1"]
    assert subset.coarse == ["valve", "valve"]
    assert torch.allclose(subset.embeddings, directory.embeddings[:2])
    assert subset.meta["subset_of"] == ["valve"]


def test_subset_by_coarse_restricts_what_can_be_matched():
    """The point of subsetting: a crop whose coarse class Stage 1 already decided can
    no longer be labelled as a symbol from another coarse family."""
    directory = _directory(n=4, coarse=["valve", "valve", "pump", "vessel"])
    subset = directory.subset_by_coarse({"pump", "vessel"})
    assert subset.match(directory.embeddings[0]).name in {"sym_2", "sym_3"}


def test_subset_by_coarse_without_a_coarse_column_is_an_error():
    """Silently returning everything would look like a working restriction."""
    with pytest.raises(ValueError, match="no `coarse` column"):
        _directory(n=3).subset_by_coarse({"valve"})


def test_subset_by_coarse_with_no_match_names_what_was_available():
    with pytest.raises(ValueError, match="available"):
        _directory(n=2, coarse=["valve", "pump"]).subset_by_coarse({"nope"})


# ------------------------------------------------------------------ #
# Persistence
# ------------------------------------------------------------------ #


def test_save_load_round_trip(tmp_path):
    directory = _directory(n=3, coarse=["valve", "pump", "valve"])
    directory.meta["encoder_ckpt"] = "runs/fake/valve.pt"
    path = directory.save(tmp_path / "nested" / "iso_prototypes.pt")
    assert path.exists(), "save must create missing parent directories"

    loaded = PrototypeDirectory.load(path)
    assert loaded.names == directory.names
    assert loaded.coarse == directory.coarse
    assert loaded.meta["encoder_ckpt"] == "runs/fake/valve.pt"
    assert torch.allclose(loaded.embeddings, directory.embeddings)


def test_load_missing_file_points_at_the_builder(tmp_path):
    with pytest.raises(FileNotFoundError, match="pid-iso-prototypes"):
        PrototypeDirectory.load(tmp_path / "absent.pt")


def test_load_rejects_a_stale_format_version(tmp_path):
    """A format change must fail loudly rather than be misread as the current layout."""
    path = tmp_path / "old.pt"
    torch.save(
        {
            "format_version": PROTOTYPE_FORMAT_VERSION + 1,
            "names": ["a"],
            "coarse": [],
            "embeddings": torch.randn(1, _DIM),
            "meta": {},
        },
        path,
    )
    with pytest.raises(ValueError, match="format_version"):
        PrototypeDirectory.load(path)


# ------------------------------------------------------------------ #
# build_prototypes
# ------------------------------------------------------------------ #


def test_build_prototypes_preserves_template_order_and_unit_norm():
    templates = {f"sym_{i}": _glyph(200 + i) for i in range(4)}
    directory = build_prototypes(_FakeEncoder(), templates, _transform())
    assert directory.names == list(templates.keys())
    assert directory.embeddings.shape == (4, _DIM)
    assert torch.allclose(directory.embeddings.norm(dim=1), torch.ones(4), atol=1e-5)


def test_build_prototypes_records_views_and_carries_coarse_and_meta():
    templates = {"a": _glyph(1), "b": _glyph(2)}
    spec = ViewSpec(scales=(1.0, 1.2), orientations=(0,))
    directory = build_prototypes(
        _FakeEncoder(),
        templates,
        _transform(),
        spec=spec,
        coarse_of={"a": "valve", "b": "pump"},
        meta={"encoder_ckpt": "x.pt"},
    )
    assert directory.coarse == ["valve", "pump"]
    assert directory.meta["template_views"] == {"scales": [1.0, 1.2], "orientations": [0]}
    assert directory.meta["n_prototypes"] == 2
    assert directory.meta["encoder_ckpt"] == "x.pt"


def test_build_prototypes_single_view_equals_a_direct_embedding():
    """With the default one-view spec a prototype is exactly the template's embedding --
    i.e. the honest one-shot baseline, with no averaging smuggled in."""
    encoder, transform = _FakeEncoder(), _transform()
    template = _glyph(42)
    directory = build_prototypes(encoder, {"a": template}, transform)
    assert torch.allclose(
        directory.embeddings[0], embed_images(encoder, [template], transform)[0], atol=1e-6
    )


def test_build_prototypes_rejects_an_empty_template_set():
    with pytest.raises(ValueError):
        build_prototypes(_FakeEncoder(), {}, _transform())


def test_match_result_is_an_immutable_record():
    """Match is frozen so a decision can be passed around (logged, aggregated) without
    a later stage mutating what the gate actually decided."""
    import dataclasses

    match = Match("valve_gate", 0.81)
    assert match.name == "valve_gate" and not match.is_other and match.n_votes == 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        match.cosine = 0.1  # type: ignore[misc]


def test_center_embeddings_subtracts_and_renormalizes():
    from utils.oneshot import center_embeddings

    x = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    c = torch.tensor([1.0, 0.5])
    out = center_embeddings(x, c)
    assert torch.allclose(out, torch.tensor([[0.0, 1.0], [0.0, -1.0]]))
    assert torch.allclose(center_embeddings(x[0], c), out[0])
    with pytest.raises(ValueError, match="does not fit"):
        center_embeddings(x, torch.zeros(3))


def test_centered_directory_keeps_names_and_marks_meta():
    from utils.oneshot import PrototypeDirectory

    d = PrototypeDirectory(["a", "b"], torch.eye(2), coarse=["v", "v"], meta={"x": 1})
    c = d.centered(torch.tensor([0.5, 0.5]))
    assert c.names == ["a", "b"] and c.coarse == ["v", "v"]
    assert c.meta == {"x": 1, "centered": True} and "centered" not in d.meta
    assert torch.allclose(c.embeddings.norm(dim=1), torch.ones(2))
    assert float(c.embeddings[0] @ c.embeddings[1]) == pytest.approx(-1.0)  # was 0 before


def test_center_file_round_trip_and_needs_the_encoder_hash(tmp_path):
    from utils.oneshot import load_center, save_center

    with pytest.raises(ValueError, match="encoder_sha256"):
        save_center(tmp_path / "c.pt", torch.ones(4), {})
    path = save_center(tmp_path / "c.pt", torch.arange(4.0), {"encoder_sha256": "ab"})
    center, meta = load_center(path)
    assert torch.equal(center, torch.arange(4.0)) and meta == {"encoder_sha256": "ab"}
    with pytest.raises(FileNotFoundError, match="protocol=center"):
        load_center(tmp_path / "missing.pt")


# ------------------------------------------------------------------ #
# Several forms of one class (`name@form`) and aggregate=max (PLAN_PHASE_C §2K.4)
# ------------------------------------------------------------------ #

_E = torch.eye(4)


def _forms_directory():
    # b@B is a second form of b; c only has one form
    return PrototypeDirectory(names=["a", "b", "b@B", "c"], embeddings=_E.clone())


def test_class_of_and_classes_keep_first_appearance_order():
    from utils.oneshot import class_of

    assert class_of("valve_butterfly@B") == "valve_butterfly" and class_of("x") == "x"
    assert _forms_directory().classes == ["a", "b", "c"]
    assert _directory(n=3).classes == ["sym_0", "sym_1", "sym_2"]


def test_class_cosines_take_the_best_form_and_are_plain_cosines_without_forms():
    d = _forms_directory()
    q = torch.nn.functional.normalize(torch.tensor([0.1, 0.2, 0.9, 0.0]), dim=0)
    cc = d.class_cosines(q)
    assert cc.shape == (3,) and cc[1] == pytest.approx(float(q[2]))  # b scores its form B
    plain = _directory(n=4)
    v = torch.randn(3, _DIM)
    assert torch.equal(plain.class_cosines(v), plain.cosines(v))


def test_a_query_like_the_second_form_is_named_the_class():
    d = _forms_directory()
    assert d.match(_E[2]).name == "b"
    assert [n for n, _ in d.topk(_E[2], 3)][0] == "b"
    assert len({n for n, _ in d.topk(_E[2], 3)}) == 3  # one entry per class
    for strategy in ("vote", "mean", "max"):
        assert d.match_views(_E[[2, 2]], None, strategy).name == "b"


def test_votes_for_two_forms_of_one_class_count_together():
    d = _forms_directory()
    views = torch.nn.functional.normalize(_E[[1, 2, 0]] + 0.01, dim=1)  # b, b@B, a
    m = d.match_views(views, None, "vote")
    assert m.name == "b" and m.n_votes == 2


def test_max_takes_the_best_view_and_gates_on_it():
    d = PrototypeDirectory(names=["a", "b"], embeddings=_E[:2].clone())
    # two views close to a, one exactly b: the mean leans to a, the best single view is b
    views = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.1, 0, 0], [1.0, 0.1, 0, 0], [0.0, 1.0, 0, 0]]), dim=1
    )
    assert d.match_views(views, None, "mean").name == "a"  # the mean leans to a
    m = d.match_views(views, None, "max")
    assert m.name == "b" and m.cosine == pytest.approx(1.0) and m.n_votes == 1
    assert d.match_views(views, 1.01, "max").is_other
    with pytest.raises(ValueError, match="strategy"):
        d.match_views(views, None, "median")


def test_without_a_class_drops_all_its_forms_and_a_form_drops_only_itself():
    d = _forms_directory()
    assert d.without(["b"]).names == ["a", "c"]
    assert d.without(["b@B"]).names == ["a", "b", "c"]
    with pytest.raises(ValueError, match="unknown"):
        d.without(["zz"])
