"""Shared sampling selection for conversion and downstream dataset tools.

This module is imported, not executed. From the repository root run:
    python conversion/aims_to_deeptb.py CASE -o MESH_SET --sampling mesh
    python conversion/aims_to_deeptb.py CASE -o PATH_SET --sampling path
    python preprocessing/split_train_val.py DATA SPLIT --sampling mesh

Legacy reports without sampling metadata are treated as path datasets. New
reports are validated; contradictory or unknown metadata is never guessed.
"""
import json
from pathlib import Path

SOURCE_BY_SAMPLING = {"mesh": "postscf", "path": "band"}


def add_conversion_sampling(parser):
    parser.add_argument("--sampling", choices=tuple(SOURCE_BY_SAMPLING),
                        help="mesh: Final_KS_eigenvalues.dat; path: band1*.out (default)")
    parser.add_argument("--eigenvalue-source", choices=("band", "postscf"), default=None,
                        help="Compatibility alias: band=path, postscf=mesh")


def resolve_conversion_sampling(args, parser):
    source = SOURCE_BY_SAMPLING.get(args.sampling)
    if source and args.eigenvalue_source and source != args.eigenvalue_source:
        parser.error("--sampling conflicts with --eigenvalue-source")
    args.eigenvalue_source = source or args.eigenvalue_source or "band"
    args.sampling = "mesh" if args.eigenvalue_source == "postscf" else "path"


def dataset_sampling(set_dir):
    report_path = Path(set_dir) / "conversion_report.json"
    if not report_path.exists():
        return "path"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    sampling = report.get("kpoint_sampling")
    source = report.get("eigenvalue_source")
    if source is not None and source not in SOURCE_BY_SAMPLING.values():
        raise ValueError(f"{report_path}: unknown eigenvalue source {source!r}")
    inferred = "mesh" if source == "postscf" else "path"
    sampling = sampling or inferred
    if sampling not in SOURCE_BY_SAMPLING:
        raise ValueError(f"{report_path}: unknown sampling {sampling!r}")
    if source is not None and SOURCE_BY_SAMPLING[sampling] != source:
        raise ValueError(f"{report_path}: inconsistent sampling/source metadata")
    return sampling


def add_sampling_filter(parser):
    parser.add_argument("--sampling", choices=("mesh", "path", "all"), default="all",
                        help="Select datasets by conversion metadata (default: all; legacy sets are path)")


def filter_sets(sets, sampling):
    selected = [p for p in sets if dataset_sampling(p) == sampling or sampling == "all"]
    if not selected:
        raise ValueError(f"No datasets match --sampling {sampling}")
    return selected


def filter_cases(cases, sampling):
    selected = [(split, p) for split, p in cases
                if dataset_sampling(p) == sampling or sampling == "all"]
    if not selected:
        raise ValueError(f"No datasets match --sampling {sampling}")
    return selected


def require_path(set_dir):
    if dataset_sampling(set_dir) != "path":
        raise ValueError(f"{set_dir}: band-path plotting requires a separate path dataset; "
                         "use visualize_band_regions.py --mode selected for mesh QA")
