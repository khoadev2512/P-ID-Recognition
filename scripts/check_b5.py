"""Check the B5 decisions before any training: symbol_map.csv statuses + oneshot.holdout.

docs/PLAN_STAGE2_ONESHOT.md B5. The held-out lists define protocol P1 and must be fixed
before R1, so the rules they have to satisfy are checked here, where a mistake costs a
re-edit, instead of after a Colab run, where it costs the run.

symbol_map.csv `status` per row:
    ISO     maps to exactly one ISO 10628-2 symbol; `reg_number` set (one-shot accuracy)
    OOV     not an ISO 10628-2 symbol ("Other" positives)
    VARIANT same MEANING as an ISO symbol (`reg_number` set) but drawn in another
            convention the template does not depict (S2: ball valve with an X in the
            circle vs ISO X8071's plain circle) — kept out of every P1 test set; usable
            only for a separate drawing-convention analysis
    UNSURE  resembles one but not certainly the same — kept out of every test set

ERROR (exit 1) when:
    - a status is not ISO/OOV/UNSURE (TODO/DONE left over)
    - an ISO/VARIANT row has no reg_number, or one whose template is not in vocab.csv /
      on disk
    - an h_in class is not ISO, an h_out class is not OOV (so VARIANT/UNSURE are in
      neither)
    - an h_in class has a VARIANT with the same reg_number that is not in h_extra (the
      encoder would still learn the held-out symbol through its variant: S5 valve_ball
      needs S2, S13), or an h_extra class is not such a VARIANT
    - holding H out empties a coarse family from training
WARN when:
    - an ISO row's fine_name differs from vocab.csv's (the prototype is named by vocab)
    - h_in outside 6-8 or h_out outside 2-4 classes, h_in from a single family
    - two ISO rows share one reg_number (two DigitizePID classes, one ISO symbol; a
      VARIANT sharing an ISO row's reg_number is the expected case, not a warning)

Run:
    uv run python scripts/check_b5.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.classmap import load_classmap  # noqa: E402
from utils.oneshot import holdout_classes  # noqa: E402

STATUSES = ("ISO", "OOV", "VARIANT", "UNSURE")


def check(
    symbol_map: Path, vocab_csv: Path, symbols_dir: Path, classes_yaml: Path, config: Path
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warns: list[str] = []

    with symbol_map.open(newline="", encoding="utf-8") as f:
        rows = {f"Symbol_{r['symbol_n'].strip()}": r for r in csv.DictReader(f)}
    with vocab_csv.open(newline="", encoding="utf-8") as f:
        vocab = {r["reg_number"].strip(): r for r in csv.DictReader(f)}
    classmap = load_classmap(classes_yaml)
    h_in, h_out, h_extra = holdout_classes(OmegaConf.load(config))

    status = {name: (r.get("status") or "").strip().upper() for name, r in rows.items()}
    for name, st in status.items():
        if st not in STATUSES:
            errors.append(f"{name}: status {st or '<empty>'!r} — must be one of {STATUSES}")

    iso_regs: dict[str, str] = {}
    for name, r in rows.items():
        if status[name] not in ("ISO", "VARIANT"):
            continue
        reg = (r.get("reg_number") or "").strip()
        if status[name] == "ISO":
            iso_regs[name] = reg
        if not reg:
            errors.append(f"{name}: {status[name]} but reg_number is empty")
        elif reg not in vocab:
            errors.append(
                f"{name}: reg_number {reg} not in {vocab_csv.name} — add the row, then "
                f"re-run scripts/extract_iso_symbols_svg.py so the template exists"
            )
        elif not (symbols_dir / f"{reg}.png").exists():
            errors.append(f"{name}: {reg} in vocab but no {symbols_dir / (reg + '.png')}")
        elif status[name] == "ISO":
            want = vocab[reg].get("fine_name", "").strip()
            have = (r.get("fine_name") or "").strip()
            if have != want:
                warns.append(f"{name}: fine_name {have!r} != vocab.csv's {want!r} for {reg}")
    for reg, n in Counter(r for r in iso_regs.values() if r).items():
        if n > 1:
            who = sorted(k for k, v in iso_regs.items() if v == reg)
            warns.append(f"{who} all map to {reg} — one ISO symbol for {n} DigitizePID classes")

    unknown = sorted(set(h_in + h_out + h_extra) - set(classmap.fine_classes))
    if unknown:
        errors.append(f"oneshot.holdout names unknown classes {unknown}")
    for name in h_in:
        if name in status and status[name] != "ISO":
            errors.append(f"h_in {name} has status {status[name]!r}, needs ISO")
    for name in h_out:
        if name in status and status[name] != "OOV":
            errors.append(f"h_out {name} has status {status[name]!r}, needs OOV")

    # VARIANT rule (PLAN B5b): an h_in symbol is unseen only if its variants are too.
    reg_of = {name: (r.get("reg_number") or "").strip() for name, r in rows.items()}
    variants: dict[str, list[str]] = {}
    for name, st in status.items():
        if st == "VARIANT" and reg_of[name]:
            variants.setdefault(reg_of[name], []).append(name)
    h_in_regs = {reg_of[n] for n in h_in if status.get(n) == "ISO" and reg_of.get(n)}
    for name in h_in:
        missing = sorted(set(variants.get(reg_of.get(name, ""), [])) - set(h_extra))
        if status.get(name) == "ISO" and missing:
            errors.append(
                f"h_in {name} ({reg_of[name]}) has VARIANT {missing} with the same "
                f"reg_number still in training — add them to oneshot.holdout.h_extra"
            )
    for name in h_extra:
        if name in status and status[name] != "VARIANT":
            errors.append(f"h_extra {name} has status {status[name]!r}, needs VARIANT")
        elif name in status and reg_of[name] not in h_in_regs:
            errors.append(
                f"h_extra {name} ({reg_of[name] or 'no reg_number'}) is not a VARIANT of "
                f"an h_in class — it would leave training for nothing"
            )

    held = set(h_in + h_out + h_extra)
    families: dict[str, list[str]] = {}
    for fine, coarse in classmap.fine_to_coarse.items():
        families.setdefault(coarse, []).append(fine)
    for coarse, members in families.items():
        if members and set(members) <= held:
            errors.append(f"coarse family {coarse!r} has every class held out: {members}")

    if not 6 <= len(h_in) <= 8:
        warns.append(f"h_in has {len(h_in)} classes (plan: 6-8)")
    if not 2 <= len(h_out) <= 4:
        warns.append(f"h_out has {len(h_out)} classes (plan: 2-4)")
    fams_in = {classmap.fine_to_coarse[c] for c in h_in if c in classmap.fine_to_coarse}
    if h_in and len(fams_in) < 2:
        warns.append(f"h_in comes from one family only ({sorted(fams_in)}) — plan: spread it")
    return errors, warns


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--symbol-map", type=Path, default=root / "docs/symbol_mapping/symbol_map.csv")
    ap.add_argument("--vocab", type=Path, default=root / "data/raw/iso_symbols/vocab.csv")
    ap.add_argument("--symbols-dir", type=Path, default=root / "data/raw/iso_symbols_clean")
    ap.add_argument("--classes", type=Path, default=root / "data/canonical/classes.yaml")
    ap.add_argument("--config", type=Path, default=root / "configs/config.yaml")
    args = ap.parse_args()

    errors, warns = check(args.symbol_map, args.vocab, args.symbols_dir, args.classes, args.config)
    for w in warns:
        print("WARN ", w)
    for e in errors:
        print("ERROR", e)
    print(f"{len(errors)} error(s), {len(warns)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
