"""
Detect and apply DeePTB energy-window weighting metadata.

Commands
--------
Dry run:
    python preprocessing/weight_band_regions.py DATA --recursive

Generate weighted metadata and activate it:
    python preprocessing/weight_band_regions.py DATA --recursive --apply

Restore equal-weight metadata:
    python preprocessing/weight_band_regions.py DATA --recursive --undo

Reactivate an existing weighted configuration:
    python preprocessing/weight_band_regions.py DATA --recursive --activate-weighted

Inspect active metadata state:
    python preprocessing/weight_band_regions.py DATA --recursive --status

Only info.json is read by DeePTB. The script keeps two canonical copies beside it:
    info.unweighted.json
    info.weighted.json

Detector details, parameters, checks and file hashes are stored in band_regions.json.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

SCRIPT_NAME = "weight_band_regions.py"
SCHEMA_VERSION = 1
DETECTOR_VERSION = "1.0"
ACTIVE_INFO = "info.json"
UNWEIGHTED_INFO = "info.unweighted.json"
WEIGHTED_INFO = "info.weighted.json"
REGION_META = "band_regions.json"
SUMMARY_NAME = "band_region_summary.csv"


@dataclass
class DetectionResult:
    status: str
    method: str
    confidence: str
    baseline_energy_eV: float
    max_energy_eV: float
    split_energy_eV: Optional[float]
    split_offset_eV: Optional[float]
    region_left_eV: Optional[float]
    region_right_eV: Optional[float]
    gap_width_eV: Optional[float]
    sparse_width_eV: Optional[float]
    sparse_mean_crossings: Optional[float]
    sparse_min_crossings: Optional[int]
    sparse_score: Optional[float]
    n_kpoints: int
    n_bands: int
    cmax: int
    notes: list[str]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def copy_atomic(src: Path, dst: Path) -> None:
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def active_state(set_dir: Path) -> str:
    active = set_dir / ACTIVE_INFO
    unweighted = set_dir / UNWEIGHTED_INFO
    weighted = set_dir / WEIGHTED_INFO
    if not active.exists():
        return "missing"
    ha = sha256_file(active)
    hu = sha256_file(unweighted)
    hw = sha256_file(weighted)
    if hu is not None and ha == hu:
        return "unweighted"
    if hw is not None and ha == hw:
        return "weighted"
    return "unknown"


def load_eigenvalues(path: Path) -> np.ndarray:
    eig = np.load(path)
    if eig.ndim == 3:
        if eig.shape[0] != 1:
            raise ValueError(f"{path}: expected one frame, got shape {eig.shape}")
        eig = eig[0]
    elif eig.ndim != 2:
        raise ValueError(f"{path}: expected (1,nk,nb) or (nk,nb), got {eig.shape}")
    eig = np.asarray(eig, dtype=float)
    if not np.isfinite(eig).all():
        raise ValueError(f"{path}: eigenvalues contain NaN/inf")
    if eig.shape[0] < 1 or eig.shape[1] < 2:
        raise ValueError(f"{path}: need at least 1 k-point and 2 bands")
    return eig


def contiguous_true_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def detect_strict_gap(
    eig: np.ndarray,
    *,
    min_gap: float,
    min_offset: float,
    max_offset: float,
    min_bands_below: int,
) -> Optional[dict]:
    e0 = float(eig.min())
    band_min = eig.min(axis=0)
    band_max = eig.max(axis=0)
    candidates = []

    for n in range(eig.shape[1] - 1):
        lower = float(band_max[n])
        upper = float(band_min[n + 1])
        gap = upper - lower
        midpoint = 0.5 * (lower + upper)
        if gap < min_gap:
            continue
        if lower - e0 < min_offset:
            continue
        if midpoint - e0 > max_offset:
            continue
        if n + 1 < min_bands_below:
            continue
        candidates.append(
            {
                "lower": lower,
                "upper": upper,
                "gap": gap,
                "midpoint": midpoint,
                "band_below": n,
                "band_above": n + 1,
            }
        )

    if not candidates:
        return None
    candidates.sort(key=lambda x: (x["midpoint"], -x["gap"]))
    return candidates[0]


def crossing_count_scan(
    eig: np.ndarray,
    *,
    min_offset: float,
    max_offset: float,
    grid_step: float,
    cmax: int,
    min_width: float,
) -> Optional[dict]:
    e0 = float(eig.min())
    lo = e0 + min_offset
    hi = min(e0 + max_offset, float(eig.max()))
    if hi <= lo:
        return None

    npts = max(2, int(math.ceil((hi - lo) / grid_step)) + 1)
    grid = np.linspace(lo, hi, npts)
    band_min = eig.min(axis=0)
    band_max = eig.max(axis=0)
    counts = np.sum(
        (band_min[None, :] <= grid[:, None])
        & (band_max[None, :] >= grid[:, None]),
        axis=1,
    )

    candidates = []
    for i0, i1 in contiguous_true_regions(counts <= cmax):
        left = float(grid[i0])
        right = float(grid[i1])
        width = right - left
        if width < min_width:
            continue
        local = counts[i0 : i1 + 1]
        mean_count = float(np.mean(local))
        min_count = int(np.min(local))
        score = width / (1.0 + mean_count)

        plateaus = contiguous_true_regions(local == min_count)
        _, p0, p1 = max((b - a + 1, a, b) for a, b in plateaus)
        split = 0.5 * (float(grid[i0 + p0]) + float(grid[i0 + p1]))

        candidates.append(
            {
                "left": left,
                "right": right,
                "width": width,
                "mean_count": mean_count,
                "min_count": min_count,
                "score": score,
                "split": split,
            }
        )

    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x["score"], x["split"]))
    return candidates[0]


def detect_region_split(
    eig: np.ndarray,
    *,
    min_gap: float,
    min_offset: float,
    max_offset: float,
    min_bands_below: int,
    grid_step: float,
    cmax: int,
    sparse_min_width: float,
) -> DetectionResult:
    e0 = float(eig.min())
    emax = float(eig.max())
    nk, nb = eig.shape
    notes: list[str] = []

    strict = detect_strict_gap(
        eig,
        min_gap=min_gap,
        min_offset=min_offset,
        max_offset=max_offset,
        min_bands_below=min_bands_below,
    )
    if strict is not None:
        notes.append(
            f"Strict empty gap between retained band indices "
            f"{strict['band_below']} and {strict['band_above']}."
        )
        return DetectionResult(
            status="detected",
            method="strict_gap",
            confidence="high",
            baseline_energy_eV=e0,
            max_energy_eV=emax,
            split_energy_eV=float(strict["midpoint"]),
            split_offset_eV=float(strict["midpoint"] - e0),
            region_left_eV=float(strict["lower"]),
            region_right_eV=float(strict["upper"]),
            gap_width_eV=float(strict["gap"]),
            sparse_width_eV=None,
            sparse_mean_crossings=None,
            sparse_min_crossings=None,
            sparse_score=None,
            n_kpoints=nk,
            n_bands=nb,
            cmax=cmax,
            notes=notes,
        )

    notes.append("No strict gap passed; using sparse-corridor fallback.")
    sparse = crossing_count_scan(
        eig,
        min_offset=min_offset,
        max_offset=max_offset,
        grid_step=grid_step,
        cmax=cmax,
        min_width=sparse_min_width,
    )
    if sparse is not None:
        notes.append(
            f"Low-density region {sparse['left']:.3f} to {sparse['right']:.3f} eV; "
            f"mean crossing count {sparse['mean_count']:.2f}."
        )
        return DetectionResult(
            status="detected",
            method="sparse_corridor",
            confidence="medium",
            baseline_energy_eV=e0,
            max_energy_eV=emax,
            split_energy_eV=float(sparse["split"]),
            split_offset_eV=float(sparse["split"] - e0),
            region_left_eV=float(sparse["left"]),
            region_right_eV=float(sparse["right"]),
            gap_width_eV=None,
            sparse_width_eV=float(sparse["width"]),
            sparse_mean_crossings=float(sparse["mean_count"]),
            sparse_min_crossings=int(sparse["min_count"]),
            sparse_score=float(sparse["score"]),
            n_kpoints=nk,
            n_bands=nb,
            cmax=cmax,
            notes=notes,
        )

    notes.append("No reliable split found; uniform weighting retained.")
    return DetectionResult(
        status="not_detected",
        method="none",
        confidence="low",
        baseline_energy_eV=e0,
        max_energy_eV=emax,
        split_energy_eV=None,
        split_offset_eV=None,
        region_left_eV=None,
        region_right_eV=None,
        gap_width_eV=None,
        sparse_width_eV=None,
        sparse_mean_crossings=None,
        sparse_min_crossings=None,
        sparse_score=None,
        n_kpoints=nk,
        n_bands=nb,
        cmax=cmax,
        notes=notes,
    )


def find_set_dirs(root: Path, recursive: bool) -> list[Path]:
    if (root / "eigenvalues.npy").exists() and (root / ACTIVE_INFO).exists():
        return [root]
    candidates = root.rglob("set.*") if recursive else root.glob("set.*")
    found = [
        p
        for p in candidates
        if p.is_dir()
        and (p / "eigenvalues.npy").exists()
        and (p / ACTIVE_INFO).exists()
    ]
    found.sort()
    if not found:
        raise FileNotFoundError(f"No converted set.* directories found below {root}")
    return found


def energy_window(info: dict) -> tuple[object, object]:
    bandinfo = info.get("bandinfo", {})
    if not isinstance(bandinfo, dict):
        return None, None
    return bandinfo.get("emin"), bandinfo.get("emax")


def validate_weighted_metadata(
    result: DetectionResult,
    emin: float,
    emax: float,
    emax_margin: float,
    tol: float,
) -> dict:
    if result.status != "detected":
        raise ValueError("Cannot validate weighting metadata without a detected split")

    expected_offset = float(result.split_energy_eV - result.baseline_energy_eV)
    aligned_max = float(result.max_energy_eV - result.baseline_energy_eV)
    checks = {
        "split_offset_error_eV": abs(float(result.split_offset_eV) - expected_offset),
        "emin_split_error_eV": abs(float(emin) - expected_offset),
        "emax_margin_error_eV": abs((float(emax) - aligned_max) - emax_margin),
        "emin_positive": bool(emin > 0),
        "emax_above_emin": bool(emax > emin),
        "emax_covers_selected_spectrum": bool(emax > aligned_max),
    }
    checks["passed"] = bool(
        checks["split_offset_error_eV"] <= tol
        and checks["emin_split_error_eV"] <= tol
        and checks["emax_margin_error_eV"] <= tol
        and checks["emin_positive"]
        and checks["emax_above_emin"]
        and checks["emax_covers_selected_spectrum"]
    )
    if not checks["passed"]:
        raise ValueError(f"Weighting metadata consistency check failed: {checks}")
    return checks


def ensure_unweighted_baseline(set_dir: Path, *, force: bool) -> Path:
    active = set_dir / ACTIVE_INFO
    baseline = set_dir / UNWEIGHTED_INFO
    legacy_backup = set_dir / "info.json.before_band_weighting"

    if baseline.exists():
        return baseline

    source = legacy_backup if legacy_backup.exists() else active
    info = load_json(source)
    emin, emax = energy_window(info)
    if (emin is not None or emax is not None) and not force:
        raise RuntimeError(
            f"{set_dir.name}: cannot establish an unweighted baseline because "
            f"{source.name} already contains a non-null energy window. "
            "Restore the equal-weight info.json first or use --force-switch only "
            "if this file is intentionally your baseline."
        )
    copy_atomic(source, baseline)
    return baseline


def update_region_active_mode(set_dir: Path, mode: str) -> None:
    path = set_dir / REGION_META
    if not path.exists():
        return
    data = load_json(path)
    data["active_mode"] = mode
    data.setdefault("files", {})["active_info_sha256"] = sha256_file(set_dir / ACTIVE_INFO)
    write_json_atomic(path, data)


def apply_one(set_dir: Path, args: argparse.Namespace) -> dict:
    eig = load_eigenvalues(set_dir / "eigenvalues.npy")
    nb = eig.shape[1]
    cmax = (
        min(4, max(2, int(math.ceil(0.02 * nb))))
        if args.sparse_max_crossings is None
        else args.sparse_max_crossings
    )
    min_bands_below = 4 if args.min_bands_below is None else args.min_bands_below

    result = detect_region_split(
        eig,
        min_gap=args.min_gap,
        min_offset=args.min_offset,
        max_offset=args.max_offset,
        min_bands_below=min_bands_below,
        grid_step=args.grid_step,
        cmax=cmax,
        sparse_min_width=args.sparse_min_width,
    )

    record = {
        "schema_version": SCHEMA_VERSION,
        "script": SCRIPT_NAME,
        "detector_version": DETECTOR_VERSION,
        "set": set_dir.name,
        "status": result.status,
        "method": result.method,
        "confidence": result.confidence,
        "spectrum": {
            "n_kpoints": result.n_kpoints,
            "n_bands": result.n_bands,
            "baseline_energy_eV": result.baseline_energy_eV,
            "max_energy_eV": result.max_energy_eV,
        },
        "region": {
            "split_energy_eV": result.split_energy_eV,
            "split_offset_eV": result.split_offset_eV,
            "candidate_left_eV": result.region_left_eV,
            "candidate_right_eV": result.region_right_eV,
            "gap_width_eV": result.gap_width_eV,
            "sparse_width_eV": result.sparse_width_eV,
            "sparse_mean_crossings": result.sparse_mean_crossings,
            "sparse_min_crossings": result.sparse_min_crossings,
            "sparse_score": result.sparse_score,
        },
        "detector_parameters": {
            "min_gap_eV": args.min_gap,
            "min_offset_eV": args.min_offset,
            "max_offset_eV": args.max_offset,
            "min_bands_below": min_bands_below,
            "grid_step_eV": args.grid_step,
            "sparse_max_crossings": cmax,
            "sparse_min_width_eV": args.sparse_min_width,
            "emax_margin_eV": args.emax_margin,
        },
        "recommended_eout_weight": args.recommended_eout_weight,
        "notes": result.notes,
        "active_mode": active_state(set_dir),
    }

    if result.status != "detected":
        if args.apply:
            write_json_atomic(set_dir / REGION_META, record)
        return record

    emin = float(result.split_offset_eV)
    emax = float(
        (result.max_energy_eV - result.baseline_energy_eV) + args.emax_margin
    )
    checks = validate_weighted_metadata(
        result, emin, emax, args.emax_margin, args.consistency_tol
    )
    record["deeptb_window"] = {"emin": emin, "emax": emax}
    record["consistency"] = checks

    if not args.apply:
        return record

    current = active_state(set_dir)
    if current == "unknown" and (set_dir / UNWEIGHTED_INFO).exists() and not args.force_switch:
        raise RuntimeError(
            f"{set_dir.name}: active info.json matches neither canonical metadata file. "
            "Refusing to overwrite it without --force-switch."
        )

    baseline_path = ensure_unweighted_baseline(set_dir, force=args.force_switch)
    base_info = load_json(baseline_path)
    weighted_info = copy.deepcopy(base_info)
    if "bandinfo" not in weighted_info or not isinstance(weighted_info["bandinfo"], dict):
        weighted_info["bandinfo"] = {}
    weighted_info["bandinfo"]["emin"] = emin
    weighted_info["bandinfo"]["emax"] = emax

    weighted_path = set_dir / WEIGHTED_INFO
    write_json_atomic(weighted_path, weighted_info)

    # Re-read and verify the canonical weighted file before activation.
    stored = load_json(weighted_path)
    stored_emin, stored_emax = energy_window(stored)
    if stored_emin is None or stored_emax is None:
        raise RuntimeError(f"{set_dir.name}: weighted metadata lost its energy window")
    if abs(float(stored_emin) - emin) > args.consistency_tol:
        raise RuntimeError(f"{set_dir.name}: stored emin failed consistency check")
    if abs(float(stored_emax) - emax) > args.consistency_tol:
        raise RuntimeError(f"{set_dir.name}: stored emax failed consistency check")

    copy_atomic(weighted_path, set_dir / ACTIVE_INFO)
    if sha256_file(set_dir / ACTIVE_INFO) != sha256_file(weighted_path):
        raise RuntimeError(f"{set_dir.name}: active weighted metadata hash mismatch")

    record["active_mode"] = "weighted"
    record["files"] = {
        "active": ACTIVE_INFO,
        "unweighted": UNWEIGHTED_INFO,
        "weighted": WEIGHTED_INFO,
        "unweighted_info_sha256": sha256_file(baseline_path),
        "weighted_info_sha256": sha256_file(weighted_path),
        "active_info_sha256": sha256_file(set_dir / ACTIVE_INFO),
    }
    write_json_atomic(set_dir / REGION_META, record)
    return record


def switch_one(set_dir: Path, mode: str, *, force: bool) -> dict:
    target_name = UNWEIGHTED_INFO if mode == "unweighted" else WEIGHTED_INFO
    target = set_dir / target_name

    # Backward-compatible recovery from the backup name used by the earlier script.
    if mode == "unweighted" and not target.exists():
        legacy = set_dir / "info.json.before_band_weighting"
        if legacy.exists():
            info = load_json(legacy)
            emin, emax = energy_window(info)
            if (emin is not None or emax is not None) and not force:
                raise RuntimeError(
                    f"{set_dir.name}: legacy backup contains a non-null energy window; "
                    "refusing to adopt it without --force-switch."
                )
            copy_atomic(legacy, target)

    if not target.exists():
        raise FileNotFoundError(f"{set_dir.name}: missing {target_name}")

    current = active_state(set_dir)
    if current == "unknown" and not force:
        raise RuntimeError(
            f"{set_dir.name}: active info.json matches neither canonical copy; "
            "use --force-switch to replace it."
        )

    copy_atomic(target, set_dir / ACTIVE_INFO)
    if sha256_file(set_dir / ACTIVE_INFO) != sha256_file(target):
        raise RuntimeError(f"{set_dir.name}: metadata switch hash mismatch")
    update_region_active_mode(set_dir, mode)
    return {
        "set": set_dir.name,
        "status": "switched",
        "active_mode": mode,
        "method": None,
        "confidence": None,
    }


def status_one(set_dir: Path) -> dict:
    state = active_state(set_dir)
    return {
        "set": set_dir.name,
        "status": "ok",
        "active_mode": state,
        "has_unweighted": (set_dir / UNWEIGHTED_INFO).exists(),
        "has_weighted": (set_dir / WEIGHTED_INFO).exists(),
        "has_regions": (set_dir / REGION_META).exists(),
        "active_sha256": sha256_file(set_dir / ACTIVE_INFO),
        "unweighted_sha256": sha256_file(set_dir / UNWEIGHTED_INFO),
        "weighted_sha256": sha256_file(set_dir / WEIGHTED_INFO),
    }


def print_detection(record: dict) -> None:
    print(f"\n{record['set']}\n{'-' * len(record['set'])}")
    if record.get("status") == "error":
        print(record["error"])
        return
    if record.get("status") == "switched":
        print(f"active metadata: {record['active_mode']}")
        return

    spectrum = record.get("spectrum", {})
    if spectrum:
        print(
            f"bands={spectrum['n_bands']}  kpoints={spectrum['n_kpoints']}  "
            f"E0={spectrum['baseline_energy_eV']:.4f} eV"
        )
    print(
        f"status={record.get('status')}  method={record.get('method')}  "
        f"confidence={record.get('confidence')}"
    )
    region = record.get("region", {})
    if region.get("split_energy_eV") is not None:
        print(
            f"split={region['split_energy_eV']:.4f} eV  "
            f"offset={region['split_offset_eV']:.4f} eV"
        )
        if region.get("sparse_width_eV") is not None:
            print(
                f"low-density width={region['sparse_width_eV']:.4f} eV  "
                f"mean crossings={region['sparse_mean_crossings']:.3f}"
            )
        if region.get("gap_width_eV") is not None:
            print(f"strict gap width={region['gap_width_eV']:.4f} eV")
    window = record.get("deeptb_window")
    if window:
        print(f"DeePTB window: emin={window['emin']:.4f}, emax={window['emax']:.4f} eV")
    print(f"active metadata: {record.get('active_mode', 'unknown')}")


def print_status(records: list[dict]) -> None:
    counts: dict[str, int] = {}
    for rec in records:
        state = rec.get("active_mode", "unknown")
        counts[state] = counts.get(state, 0) + 1
    print(f"Processed {len(records)} set(s)")
    for key in ("weighted", "unweighted", "unknown", "missing"):
        if key in counts:
            print(f"  {key:10s}: {counts[key]}")


def write_summary(root: Path, records: list[dict]) -> Path:
    out = root / SUMMARY_NAME
    fields = [
        "set", "status", "method", "confidence", "active_mode",
        "baseline_energy_eV", "split_energy_eV", "split_offset_eV",
        "candidate_left_eV", "candidate_right_eV", "gap_width_eV",
        "sparse_width_eV", "sparse_mean_crossings", "sparse_min_crossings",
        "emin", "emax", "consistency_passed",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            spectrum = rec.get("spectrum", {})
            region = rec.get("region", {})
            window = rec.get("deeptb_window", {}) or {}
            checks = rec.get("consistency", {}) or {}
            writer.writerow(
                {
                    "set": rec.get("set"),
                    "status": rec.get("status"),
                    "method": rec.get("method"),
                    "confidence": rec.get("confidence"),
                    "active_mode": rec.get("active_mode"),
                    "baseline_energy_eV": spectrum.get("baseline_energy_eV"),
                    "split_energy_eV": region.get("split_energy_eV"),
                    "split_offset_eV": region.get("split_offset_eV"),
                    "candidate_left_eV": region.get("candidate_left_eV"),
                    "candidate_right_eV": region.get("candidate_right_eV"),
                    "gap_width_eV": region.get("gap_width_eV"),
                    "sparse_width_eV": region.get("sparse_width_eV"),
                    "sparse_mean_crossings": region.get("sparse_mean_crossings"),
                    "sparse_min_crossings": region.get("sparse_min_crossings"),
                    "emin": window.get("emin"),
                    "emax": window.get("emax"),
                    "consistency_passed": checks.get("passed"),
                }
            )
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect and switch DeePTB band-region weighting metadata.")
    p.add_argument("path", type=Path, help="One set.* directory or a dataset root.")
    p.add_argument("--recursive", action="store_true", help="Search recursively for set.* directories.")

    actions = p.add_mutually_exclusive_group()
    actions.add_argument("--apply", action="store_true", help="Generate and activate weighted info.json metadata.")
    actions.add_argument("--undo", action="store_true", help="Restore info.unweighted.json as active info.json.")
    actions.add_argument("--activate-weighted", action="store_true", help="Restore info.weighted.json as active info.json.")
    actions.add_argument("--status", action="store_true", help="Report the active metadata state without modifying files.")

    p.add_argument("--force-switch", action="store_true", help="Allow replacement of an unknown active info.json state.")
    p.add_argument("--min-gap", type=float, default=5.0)
    p.add_argument("--min-offset", type=float, default=5.0)
    p.add_argument("--max-offset", type=float, default=15.0)
    p.add_argument("--min-bands-below", type=int, default=None)
    p.add_argument("--grid-step", type=float, default=0.05)
    p.add_argument("--sparse-max-crossings", type=int, default=None)
    p.add_argument("--sparse-min-width", type=float, default=1.0)
    p.add_argument("--emax-margin", type=float, default=1.0)
    p.add_argument("--consistency-tol", type=float, default=1e-8)
    p.add_argument(
        "--recommended-eout-weight",
        type=float,
        default=0.5,
        help="Recorded for provenance only; set eout_weight separately in the training input.",
    )
    args = p.parse_args()

    if args.min_gap <= 0:
        p.error("--min-gap must be > 0")
    if args.min_offset < 0:
        p.error("--min-offset must be >= 0")
    if args.max_offset <= args.min_offset:
        p.error("--max-offset must be greater than --min-offset")
    if args.grid_step <= 0:
        p.error("--grid-step must be > 0")
    if args.sparse_min_width <= 0:
        p.error("--sparse-min-width must be > 0")
    if args.sparse_max_crossings is not None and args.sparse_max_crossings < 0:
        p.error("--sparse-max-crossings must be >= 0")
    if args.min_bands_below is not None and args.min_bands_below < 1:
        p.error("--min-bands-below must be >= 1")
    if args.emax_margin <= 0:
        p.error("--emax-margin must be > 0")
    if args.consistency_tol <= 0:
        p.error("--consistency-tol must be > 0")
    if not 0 <= args.recommended_eout_weight <= 1:
        p.error("--recommended-eout-weight must be between 0 and 1")
    return args


def main() -> None:
    args = parse_args()
    root = args.path.resolve()
    set_dirs = find_set_dirs(root, args.recursive)
    records = []

    if args.status:
        records = [status_one(set_dir) for set_dir in set_dirs]
        print_status(records)
        return

    for set_dir in set_dirs:
        try:
            if args.undo:
                rec = switch_one(set_dir, "unweighted", force=args.force_switch)
            elif args.activate_weighted:
                rec = switch_one(set_dir, "weighted", force=args.force_switch)
            else:
                rec = apply_one(set_dir, args)
        except Exception as exc:
            rec = {
                "set": set_dir.name,
                "status": "error",
                "method": None,
                "confidence": None,
                "active_mode": active_state(set_dir),
                "error": f"{type(exc).__name__}: {exc}",
            }
        print_detection(rec)
        if rec.get("status") == "error":
            print(f"  - {rec['error']}")
        records.append(rec)

    if args.apply and len(set_dirs) > 1:
        summary = write_summary(root, records)
        print(f"\nWrote summary: {summary}")

    errors = sum(r.get("status") == "error" for r in records)
    detected = sum(r.get("status") == "detected" for r in records)
    switched = sum(r.get("status") == "switched" for r in records)
    if args.undo or args.activate_weighted:
        print(f"\nProcessed {len(records)} set(s): {switched} switched, {errors} error(s).")
    else:
        print(f"\nProcessed {len(records)} set(s): {detected} detected, {errors} error(s).")
        if not args.apply:
            print("Dry run only. Use --apply to generate and activate weighted metadata.")


if __name__ == "__main__":
    main()
