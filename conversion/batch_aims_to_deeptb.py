"""
Batch-convert a directory tree of FHI-aims hBN calculations into one-frame
DeePTB datasets using ``aims_to_deeptb.py``.

Usage
-----
Recommended adaptive conversion::

    python conversion/batch_aims_to_deeptb.py INPUT_ROOT OUTPUT_ROOT \
        --band-policy adaptive-factor --band-factor 2.0

Resume an interrupted conversion without re-running completed sets::

    python conversion/batch_aims_to_deeptb.py INPUT_ROOT OUTPUT_ROOT \
        --band-policy adaptive-factor --band-factor 2.0 --resume

Override the default converter with another custom converter::

    python conversion/batch_aims_to_deeptb.py INPUT_ROOT OUTPUT_ROOT \
        --converter conversion/aims_to_deeptb.py \
        --band-policy adaptive-factor --band-factor 2.0

Output layout
-------------
Each successful source calculation is written to a separate directory::

    OUTPUT_ROOT/set.000000/
    OUTPUT_ROOT/set.000001/
    ...

``manifest.json`` and ``manifest.csv`` record the source-to-set mapping,
selected band policy, band count, atom count, composition, and failures.

Each set contains one structure/frame.  This allows different supercell sizes
and adaptive band counts to coexist in one dataset collection.  Cases containing
``band2*.out`` are skipped by default so that a second spin channel is not
silently discarded; use ``--include-spin2`` only when that behavior is intended.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def discover_cases(root: Path, geometry: str, band_glob: str) -> List[Path]:
    cases = []
    for geom in root.rglob(geometry):
        case = geom.parent
        if any(case.glob(band_glob)):
            cases.append(case)
    return sorted(cases, key=lambda p: str(p.relative_to(root)))


def load_report(path: Path) -> Dict:
    with path.open() as f:
        return json.load(f)


def write_manifests(records: List[Dict], out_root: Path) -> None:
    json_path = out_root / "manifest.json"
    csv_path = out_root / "manifest.csv"

    with json_path.open("w") as f:
        json.dump(records, f, indent=2)

    fields = [
        "set_id",
        "status",
        "source_case",
        "source_relative",
        "output_dir",
        "natoms",
        "element_counts",
        "inferred_core_bands",
        "contiguous_noncore_full_count",
        "last_band_with_occupation",
        "band_policy",
        "band_factor",
        "available_noncore_bands",
        "factor_target_count",
        "frontier_span_noncore",
        "requested_count_before_available_cap",
        "requested_noncore_bands",
        "selected_first_band",
        "selected_last_band",
        "selected_band_count",
        "nkpoints",
        "spin2_detected",
        "error",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            row = {k: rec.get(k) for k in fields}
            if isinstance(row.get("element_counts"), dict):
                row["element_counts"] = json.dumps(
                    row["element_counts"], sort_keys=True
                )
            writer.writerow(row)


def build_parser():
    p = argparse.ArgumentParser(
        description=(
            "Recursively convert FHI-aims calculations into separate "
            "DeePTB sets using an explicit non-core band-selection policy."
        )
    )
    p.add_argument("input_root", type=Path)
    p.add_argument("output_root", type=Path)
    p.add_argument(
        "--converter",
        type=Path,
        default=Path(__file__).resolve().with_name("aims_to_deeptb.py"),
        help="Path to aims_to_deeptb.py",
    )
    p.add_argument("--geometry", default="geometry.in")
    p.add_argument("--band-glob", default="band1*.out")
    p.add_argument(
        "--band-policy",
        choices=["adaptive-factor", "all-noncore", "fixed-noncore"],
        default="adaptive-factor",
        help=(
            "Band-selection policy passed to the converter. "
            "Default: adaptive-factor."
        ),
    )
    p.add_argument(
        "--band-factor",
        type=float,
        default=2.0,
        metavar="F",
        help=(
            "Factor F for --band-policy adaptive-factor (default: 2.0)."
        ),
    )
    p.add_argument(
        "--noncore-bands",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Required only for --band-policy fixed-noncore: retain exactly "
            "N non-core bands per case."
        ),
    )
    p.add_argument(
        "--include-spin2",
        action="store_true",
        help=(
            "Allow cases containing band2*.out to be converted using only "
            "the selected --band-glob. Default is to skip them so a second "
            "spin channel is never silently ignored."
        ),
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite already populated set directories.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "If a set directory already contains conversion_report.json, "
            "reuse that result instead of rerunning the converter."
        ),
    )
    p.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First numerical set ID (default: 0).",
    )
    return p


def main():
    args = build_parser().parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    converter = args.converter.resolve()

    if args.band_policy == "adaptive-factor":
        if args.band_factor < 1.0:
            raise SystemExit("--band-factor must be >= 1.0.")
        if args.noncore_bands is not None:
            raise SystemExit(
                "--noncore-bands is only valid with --band-policy fixed-noncore."
            )
    elif args.band_policy == "all-noncore":
        if args.noncore_bands is not None:
            raise SystemExit(
                "--noncore-bands is only valid with --band-policy fixed-noncore."
            )
    elif args.band_policy == "fixed-noncore":
        if args.noncore_bands is None or args.noncore_bands <= 0:
            raise SystemExit(
                "--band-policy fixed-noncore requires --noncore-bands N > 0."
            )

    if not input_root.is_dir():
        raise SystemExit(f"Input root not found: {input_root}")
    if not converter.is_file():
        raise SystemExit(f"Converter not found: {converter}")

    output_root.mkdir(parents=True, exist_ok=True)

    cases = discover_cases(input_root, args.geometry, args.band_glob)
    print(f"Discovered {len(cases)} candidate calculation folders.")
    print(f"Band policy: {args.band_policy}")
    if args.band_policy == "adaptive-factor":
        print(f"Band factor: {args.band_factor}")
    elif args.band_policy == "fixed-noncore":
        print(f"Fixed non-core bands: {args.noncore_bands}")

    records: List[Dict] = []
    ok = skipped_spin = failed = reused = 0

    for ordinal, case in enumerate(cases, start=args.start_index):
        set_id = f"set.{ordinal:06d}"
        out_dir = output_root / set_id
        rel = str(case.relative_to(input_root))
        spin2 = sorted(p.name for p in case.glob("band2*.out"))

        print(f"[{ordinal - args.start_index + 1}/{len(cases)}] {rel}")

        if spin2 and not args.include_spin2:
            print("  SKIP: band2*.out detected (possible second spin channel)")
            records.append({
                "set_id": set_id,
                "status": "skipped_spin2",
                "source_case": str(case),
                "source_relative": rel,
                "output_dir": str(out_dir),
                "spin2_detected": True,
                "error": "band2*.out detected; spin handling intentionally deferred",
            })
            skipped_spin += 1
            continue

        report_path = out_dir / "conversion_report.json"
        if args.resume and report_path.exists():
            report = load_report(report_path)
            selection = report.get("selection", {})
            report_policy = selection.get("band_policy_effective")

            policy_matches = report_policy == args.band_policy
            if args.band_policy == "adaptive-factor":
                report_factor = selection.get("band_factor")
                policy_matches = (
                    policy_matches
                    and report_factor is not None
                    and abs(float(report_factor) - float(args.band_factor)) < 1e-12
                )
            elif args.band_policy == "fixed-noncore":
                report_n = selection.get("requested_noncore_bands")
                policy_matches = (
                    policy_matches
                    and report_n is not None
                    and int(report_n) == int(args.noncore_bands)
                )

            if not policy_matches:
                raise SystemExit(
                    f"Refusing --resume for {out_dir}: existing conversion_report.json "
                    f"uses policy={report_policy!r}, factor="
                    f"{selection.get('band_factor')!r}, requested_noncore="
                    f"{selection.get('requested_noncore_bands')!r}; requested "
                    f"policy={args.band_policy!r}, factor={args.band_factor!r}, "
                    f"noncore_bands={args.noncore_bands!r}. Use a fresh output root "
                    "or --overwrite without --resume."
                )

            status = "reused"
            reused += 1
        else:
            cmd = [
                sys.executable,
                str(converter),
                str(case),
                "-o", str(out_dir),
                "--geometry", args.geometry,
                "--band-glob", args.band_glob,
                "--band-policy", args.band_policy,
            ]

            if args.band_policy == "adaptive-factor":
                cmd.extend(["--band-factor", str(args.band_factor)])
            elif args.band_policy == "fixed-noncore":
                cmd.extend(["--noncore-bands", str(args.noncore_bands)])

            if args.overwrite:
                cmd.append("--overwrite")

            proc = subprocess.run(
                cmd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

            if proc.returncode != 0:
                print("  FAIL")
                # Print the last few lines, which normally contain the useful error.
                lines = proc.stdout.rstrip().splitlines()
                for line in lines[-8:]:
                    print("   ", line)
                records.append({
                    "set_id": set_id,
                    "status": "failed",
                    "source_case": str(case),
                    "source_relative": rel,
                    "output_dir": str(out_dir),
                    "spin2_detected": bool(spin2),
                    "error": "\n".join(lines[-20:]),
                })
                failed += 1
                write_manifests(records, output_root)
                continue

            if not report_path.exists():
                records.append({
                    "set_id": set_id,
                    "status": "failed",
                    "source_case": str(case),
                    "source_relative": rel,
                    "output_dir": str(out_dir),
                    "spin2_detected": bool(spin2),
                    "error": "converter returned success but report is missing",
                })
                failed += 1
                write_manifests(records, output_root)
                continue

            report = load_report(report_path)
            status = "ok"
            ok += 1

        frontier = report.get("frontier_diagnostics", {})
        core = report.get("core_inference", {})
        selection = report.get("selection", {})
        selected_range = selection.get(
            "selected_original_band_range_1based_inclusive", [None, None]
        )
        kshape = report.get("output_kpoints_shape", [None, None])

        rec = {
            "set_id": set_id,
            "status": status,
            "source_case": str(case),
            "source_relative": rel,
            "output_dir": str(out_dir),
            "natoms": report.get("natoms"),
            "element_counts": report.get("element_counts"),
            "inferred_core_bands": core.get("inferred_core_band_count"),
            "contiguous_noncore_full_count": frontier.get(
                "contiguous_noncore_fully_occupied_band_count"
            ),
            "last_band_with_occupation": frontier.get(
                "last_band_with_any_occupation_1based"
            ),
            "band_policy": selection.get("band_policy_effective"),
            "band_factor": selection.get("band_factor"),
            "available_noncore_bands": selection.get("available_noncore_bands"),
            "factor_target_count": selection.get("factor_target_count"),
            "frontier_span_noncore": selection.get("frontier_span_noncore"),
            "requested_count_before_available_cap": selection.get(
                "requested_count_before_available_cap"
            ),
            "requested_noncore_bands": selection.get("requested_noncore_bands"),
            "selected_first_band": selected_range[0],
            "selected_last_band": selected_range[1],
            "selected_band_count": selection.get("selected_band_count"),
            "nkpoints": kshape[0],
            "spin2_detected": bool(
                report.get("possible_second_spin_channel_files")
            ),
            "error": None,
        }
        records.append(rec)
        policy_text = rec.get("band_policy")
        if rec.get("band_factor") is not None:
            policy_text = f"{policy_text}({rec['band_factor']:g})"
        print(
            f"  {status.upper()}: atoms={rec['natoms']}, "
            f"bands={rec['selected_band_count']}, nk={rec['nkpoints']}, "
            f"policy={policy_text}"
        )
        write_manifests(records, output_root)

    write_manifests(records, output_root)
    print("\nBatch conversion complete")
    print(f"  policy:        {args.band_policy}")
    if args.band_policy == "adaptive-factor":
        print(f"  factor:        {args.band_factor}")
    elif args.band_policy == "fixed-noncore":
        print(f"  noncore bands: {args.noncore_bands}")
    print(f"  converted:     {ok}")
    print(f"  reused:        {reused}")
    print(f"  skipped spin:  {skipped_spin}")
    print(f"  failed:        {failed}")
    print(f"  manifest:      {output_root / 'manifest.json'}")
    print(f"  CSV manifest:  {output_root / 'manifest.csv'}")


if __name__ == "__main__":
    main()
