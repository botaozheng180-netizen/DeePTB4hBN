"""
Create reproducible DeePTB train/validation dataset trees.

Commands
--------
Random 80/20 split:
    python preprocessing/split_train_val.py CONVERTED_DATA SPLIT_DATA

Fixed validation fraction and seed:
    python preprocessing/split_train_val.py CONVERTED_DATA SPLIT_DATA \
        --val-fraction 0.2 --seed 42

Explicit validation sets:
    python preprocessing/split_train_val.py CONVERTED_DATA SPLIT_DATA \
        --val-sets set.000007 set.000013 set.000017

Read validation sets from a text file:
    python preprocessing/split_train_val.py CONVERTED_DATA SPLIT_DATA \
        --val-list validation_sets.txt

Inspect without writing:
    python preprocessing/split_train_val.py CONVERTED_DATA SPLIT_DATA \
        --val-fraction 0.2 --seed 42 --dry-run

The source dataset is never modified. The output contains train/ and val/
directories plus split_manifest.json and split_manifest.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
from pathlib import Path

SCRIPT_NAME = "split_train_val.py"
SCHEMA_VERSION = 1
SET_RE = re.compile(r"^set\.(\d+)$")
REQUIRED_FILES = ("info.json", "eigenvalues.npy", "kpoints.npy", "xdat.traj")


def natural_set_key(path: Path) -> int:
    match = SET_RE.match(path.name)
    if match is None:
        raise ValueError(f"Invalid set directory name: {path.name}")
    return int(match.group(1))


def normalize_set_name(token: str) -> str:
    token = token.strip()
    if not token:
        raise ValueError("Empty set identifier")
    if token.startswith("set."):
        token = token[4:]
    if not token.isdigit():
        raise ValueError(
            f"Invalid set identifier {token!r}; use e.g. 7, 000007, or set.000007"
        )
    return f"set.{int(token):06d}"


def discover_sets(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Input dataset does not exist: {root}")

    sets = []
    invalid = []
    for path in root.iterdir():
        if not path.is_dir() or SET_RE.match(path.name) is None:
            continue
        missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
        if missing:
            invalid.append((path.name, missing))
        else:
            sets.append(path)

    if invalid:
        details = "\n".join(
            f"  {name}: missing {', '.join(missing)}" for name, missing in invalid
        )
        raise ValueError(f"Incomplete converted set directories found:\n{details}")

    sets.sort(key=natural_set_key)
    if len(sets) < 2:
        raise ValueError(f"Need at least two valid set.* directories in {root}")
    return sets


def read_val_list(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Validation list does not exist: {path}")
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.append(normalize_set_name(line))
    if not names:
        raise ValueError(f"No validation sets found in {path}")
    return names


def choose_validation_sets(
    set_names: list[str],
    *,
    explicit: list[str] | None,
    val_list: Path | None,
    val_count: int | None,
    val_fraction: float | None,
    seed: int,
) -> tuple[list[str], dict]:
    available = set(set_names)

    if explicit is not None:
        requested = [normalize_set_name(x) for x in explicit]
        method = "explicit"
        selector = {"val_sets": requested}
    elif val_list is not None:
        requested = read_val_list(val_list)
        method = "list_file"
        selector = {"val_list": str(val_list), "val_sets": requested}
    else:
        n_total = len(set_names)
        if val_count is not None:
            n_val = val_count
            method = "random_count"
            selector = {"val_count": n_val, "seed": seed}
        else:
            fraction = 0.2 if val_fraction is None else val_fraction
            n_val = int(round(n_total * fraction))
            n_val = max(1, min(n_total - 1, n_val))
            method = "random_fraction"
            selector = {
                "val_fraction": fraction,
                "resolved_val_count": n_val,
                "seed": seed,
            }

        if not 1 <= n_val < n_total:
            raise ValueError(
                f"Validation count must be between 1 and {n_total - 1}; got {n_val}"
            )
        rng = random.Random(seed)
        requested = sorted(rng.sample(set_names, n_val), key=lambda x: int(x[4:]))

    duplicates = sorted({x for x in requested if requested.count(x) > 1})
    if duplicates:
        raise ValueError(
            "Validation selection contains duplicates: " + ", ".join(duplicates)
        )

    missing = sorted(set(requested) - available)
    if missing:
        raise ValueError(
            "Requested validation sets were not found: " + ", ".join(missing)
        )

    if not requested or len(requested) >= len(set_names):
        raise ValueError("Validation selection must leave at least one training set")

    requested_set = set(requested)
    val_names = [name for name in set_names if name in requested_set]
    return val_names, {"method": method, **selector}


def path_is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def prepare_output_dirs(
    output_root: Path,
    *,
    overwrite: bool,
) -> tuple[Path, Path]:
    train_dir = output_root / "train"
    val_dir = output_root / "val"

    existing = [p for p in (train_dir, val_dir) if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Output train/ or val/ directory already exists. "
            "Use --overwrite to replace the split."
        )

    if overwrite:
        for path in (train_dir, val_dir):
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)

    output_root.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir()
    val_dir.mkdir()
    return train_dir, val_dir


def materialize_set(src: Path, dst: Path, mode: str) -> None:
    if mode == "copy":
        shutil.copytree(src, dst, copy_function=shutil.copy2)
    elif mode == "symlink":
        relative_target = os.path.relpath(src.resolve(), start=dst.parent.resolve())
        dst.symlink_to(relative_target, target_is_directory=True)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def write_manifests(
    output_root: Path,
    *,
    input_root: Path,
    mode: str,
    selection: dict,
    train_names: list[str],
    val_names: list[str],
) -> None:
    records = [
        {"set": name, "split": "train", "source": str(input_root / name)}
        for name in train_names
    ] + [
        {"set": name, "split": "val", "source": str(input_root / name)}
        for name in val_names
    ]
    records.sort(key=lambda row: int(row["set"][4:]))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "script": SCRIPT_NAME,
        "input_root": str(input_root),
        "materialization_mode": mode,
        "selection": selection,
        "n_total": len(records),
        "n_train": len(train_names),
        "n_val": len(val_names),
        "train_sets": train_names,
        "val_sets": val_names,
    }

    (output_root / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    with (output_root / "split_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=["set", "split", "source"])
        writer.writeheader()
        writer.writerows(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create reproducible train/validation trees from converted set.* data."
    )
    parser.add_argument(
        "input_root",
        type=Path,
        help="Converted dataset containing set.* directories.",
    )
    parser.add_argument(
        "output_root",
        type=Path,
        help="Output root; train/ and val/ are created here.",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--val-sets",
        nargs="+",
        metavar="SET",
        help="Explicit validation set identifiers, e.g. 7 13 set.000017.",
    )
    group.add_argument(
        "--val-list",
        type=Path,
        help="Text file containing one validation set identifier per line.",
    )
    group.add_argument(
        "--val-count",
        type=int,
        help="Randomly select exactly this many validation sets.",
    )
    group.add_argument(
        "--val-fraction",
        type=float,
        help="Random validation fraction. Default when no selector is given: 0.2.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for fraction/count splitting. Default: 42.",
    )
    parser.add_argument(
        "--mode",
        choices=("copy", "symlink"),
        default="copy",
        help="How to create split set directories. Default: copy.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved split without writing files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output train/ and val/ directories.",
    )
    parser.add_argument(
        "--allow-nested-output",
        action="store_true",
        help=(
            "Allow OUTPUT_ROOT inside INPUT_ROOT. Not recommended because "
            "recursive tools may see duplicate sets."
        ),
    )

    args = parser.parse_args()

    if args.val_count is not None and args.val_count < 1:
        parser.error("--val-count must be >= 1")
    if args.val_fraction is not None and not (0.0 < args.val_fraction < 1.0):
        parser.error("--val-fraction must be between 0 and 1")

    return args


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()

    if input_root == output_root:
        raise ValueError("INPUT_ROOT and OUTPUT_ROOT must be different")
    if path_is_within(output_root, input_root) and not args.allow_nested_output:
        raise ValueError(
            "OUTPUT_ROOT is inside INPUT_ROOT. Use a sibling output directory, "
            "or pass --allow-nested-output if this is intentional."
        )

    set_dirs = discover_sets(input_root)
    set_names = [p.name for p in set_dirs]

    val_names, selection = choose_validation_sets(
        set_names,
        explicit=args.val_sets,
        val_list=args.val_list,
        val_count=args.val_count,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    val_set = set(val_names)
    train_names = [name for name in set_names if name not in val_set]

    print(f"Input:  {input_root}")
    print(f"Output: {output_root}")
    print(
        f"Split:  {len(train_names)} train / {len(val_names)} val "
        f"({len(set_names)} total)"
    )
    print(f"Method: {selection['method']}")
    if "seed" in selection:
        print(f"Seed:   {selection['seed']}")
    print("Validation sets:")
    print("  " + " ".join(val_names))

    if args.dry_run:
        print("\nDry run only; no files were written.")
        return

    train_dir, val_dir = prepare_output_dirs(
        output_root,
        overwrite=args.overwrite,
    )

    by_name = {p.name: p for p in set_dirs}
    for name in train_names:
        materialize_set(by_name[name], train_dir / name, args.mode)
    for name in val_names:
        materialize_set(by_name[name], val_dir / name, args.mode)

    write_manifests(
        output_root,
        input_root=input_root,
        mode=args.mode,
        selection=selection,
        train_names=train_names,
        val_names=val_names,
    )

    print(f"\nWrote: {train_dir}")
    print(f"Wrote: {val_dir}")
    print(f"Wrote: {output_root / 'split_manifest.json'}")
    print(f"Wrote: {output_root / 'split_manifest.csv'}")


if __name__ == "__main__":
    main()
