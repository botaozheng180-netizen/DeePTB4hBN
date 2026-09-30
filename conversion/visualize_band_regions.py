"""
Visual QA for weight_band_regions.py.

Commands
--------
Selected converted spectrum:
    python visualization/visualize_band_regions.py DATA --mode selected

Full non-core FHI-aims spectrum:
    python visualization/visualize_band_regions.py DATA --mode full

Inspect active instead of canonical weighted metadata:
    python visualization/visualize_band_regions.py DATA --mode selected --window-source active

The plot uses dashed red lines for emin/emax and light orange or gray shading for
the detected low-density region. Batch runs also produce contact sheets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

ACTIVE_INFO = "info.json"
UNWEIGHTED_INFO = "info.unweighted.json"
WEIGHTED_INFO = "info.weighted.json"
REGION_META = "band_regions.json"

CORE_BANDS_PER_ATOM = {"H": 0, "B": 1, "C": 1, "N": 1, "O": 1}


def _f(x: str) -> float:
    return float(x.replace("D", "E").replace("d", "e"))


def natural_key(value: str | Path):
    s = Path(value).name
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


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
    if not active.exists():
        return "missing"
    ha = sha256_file(active)
    hu = sha256_file(set_dir / UNWEIGHTED_INFO)
    hw = sha256_file(set_dir / WEIGHTED_INFO)
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
    return eig


def load_kpoints(path: Path) -> np.ndarray:
    kp = np.asarray(np.load(path), dtype=float)
    if kp.ndim == 3 and kp.shape[0] == 1:
        kp = kp[0]
    if kp.ndim != 2 or kp.shape[1] != 3:
        raise ValueError(f"{path}: expected (nk,3), got {kp.shape}")
    return kp


def parse_numeric_band_file(path: Path):
    rows = []
    with path.open("r", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            toks = s.split()
            try:
                vals = [_f(t) for t in toks]
            except ValueError:
                continue
            if len(vals) < 6 or (len(vals) - 4) % 2 != 0:
                continue
            rows.append(vals)

    if not rows:
        raise ValueError(f"{path}: no standard numeric FHI-aims band rows found")
    nbands_set = {(len(v) - 4) // 2 for v in rows}
    if len(nbands_set) != 1:
        raise ValueError(f"{path}: inconsistent band counts")
    nbands = nbands_set.pop()

    kpoints, occs, eigs = [], [], []
    for vals in rows:
        kpoints.append(vals[1:4])
        pairs = np.asarray(vals[4:], dtype=float).reshape(nbands, 2)
        occs.append(pairs[:, 0])
        eigs.append(pairs[:, 1])
    return np.asarray(kpoints), np.asarray(occs), np.asarray(eigs)


def combine_aims_segments(files: list[Path], atol: float = 1e-8):
    all_k, all_occ, all_eig = [], [], []
    last_k = None
    nbands = None
    for path in sorted(files, key=natural_key):
        kp, occ, eig = parse_numeric_band_file(path)
        if nbands is None:
            nbands = eig.shape[1]
        elif eig.shape[1] != nbands:
            raise ValueError(f"Band-count mismatch in {path.name}")
        start = 0
        if last_k is not None and np.allclose(last_k, kp[0], atol=atol, rtol=0):
            start = 1
        if start < len(kp):
            all_k.append(kp[start:])
            all_occ.append(occ[start:])
            all_eig.append(eig[start:])
            last_k = kp[-1].copy()
    if not all_k:
        raise ValueError("No FHI-aims k-points remain after joining band segments")
    return np.concatenate(all_k), np.concatenate(all_occ), np.concatenate(all_eig)


def infer_core_band_count(symbols: list[str]) -> int:
    unsupported = sorted(set(symbols) - set(CORE_BANDS_PER_ATOM))
    if unsupported:
        raise ValueError("Cannot infer 1s core stripping for: " + ", ".join(unsupported))
    return int(sum(CORE_BANDS_PER_ATOM[s] for s in symbols))


def source_case_from_report(set_dir: Path) -> Optional[Path]:
    path = set_dir / "conversion_report.json"
    if not path.exists():
        return None
    source = load_json(path).get("source_case")
    return Path(source) if source else None


def load_full_noncore_reference(aims_dir: Path, atoms, expected_kpoints: np.ndarray, k_tol: float):
    band1 = sorted(aims_dir.glob("band1*.out"), key=natural_key)
    band2 = sorted(aims_dir.glob("band2*.out"), key=natural_key)
    if not band1:
        raise FileNotFoundError(f"No band1*.out in {aims_dir}")
    if band2:
        raise RuntimeError("This QA script expects the current single-channel band workflow")

    kp, _occ, eig_all = combine_aims_segments(band1)
    if kp.shape != expected_kpoints.shape:
        raise ValueError(f"Raw k-point shape {kp.shape} != converted {expected_kpoints.shape}")
    max_kdiff = float(np.max(np.abs(kp - expected_kpoints)))
    if max_kdiff > k_tol:
        raise ValueError(f"Raw/converted kpoints differ: max |dk|={max_kdiff:.3e}")

    n_core = infer_core_band_count(list(atoms.get_chemical_symbols()))
    if n_core >= eig_all.shape[1]:
        raise ValueError("Inferred core count exceeds available DFT bands")
    return np.asarray(eig_all[:, n_core:], dtype=float), n_core, eig_all.shape[1]


def cumulative_k_distance(kpoints_frac: np.ndarray, atoms) -> np.ndarray:
    reciprocal = 2.0 * np.pi * np.asarray(atoms.cell.reciprocal())
    k_cart = np.asarray(kpoints_frac) @ reciprocal
    dk = np.linalg.norm(np.diff(k_cart, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(dk)))


def find_hbn_high_symmetry_points(kpoints, xlist, tolerance=2.0e-3):
    targets = [
        ("M", np.array([0.5, 0.0, 0.0])),
        (r"$\Gamma$", np.array([0.0, 0.0, 0.0])),
        ("K", np.array([1.0 / 3.0, 1.0 / 3.0, 0.0])),
        ("M", np.array([0.5, 0.5, 0.0])),
    ]
    found = []
    for label, target in targets:
        dist = np.linalg.norm(kpoints - target, axis=1)
        idx = int(np.argmin(dist))
        if float(dist[idx]) > tolerance:
            return []
        found.append((idx, float(xlist[idx]), label))
    found.sort(key=lambda item: item[0])
    if len({item[0] for item in found}) != len(found):
        return []
    return found


def energy_window_from_info(path: Path) -> tuple[Optional[float], Optional[float]]:
    if not path.exists():
        return None, None
    info = load_json(path)
    bandinfo = info.get("bandinfo", {})
    if not isinstance(bandinfo, dict):
        return None, None
    emin = bandinfo.get("emin")
    emax = bandinfo.get("emax")
    return (
        float(emin) if emin is not None else None,
        float(emax) if emax is not None else None,
    )


def load_region_metadata(set_dir: Path, selected_eig: np.ndarray, window_source: str) -> dict:
    region_path = set_dir / REGION_META
    region = load_json(region_path) if region_path.exists() else {}

    if window_source == "weighted" and (set_dir / WEIGHTED_INFO).exists():
        info_path = set_dir / WEIGHTED_INFO
    else:
        info_path = set_dir / ACTIVE_INFO

    emin_rel, emax_rel = energy_window_from_info(info_path)

    # New schema first; legacy fields remain supported for existing QA output.
    spectrum_meta = region.get("spectrum", {}) if isinstance(region.get("spectrum", {}), dict) else {}
    region_meta = region.get("region", {}) if isinstance(region.get("region", {}), dict) else {}
    baseline = spectrum_meta.get("baseline_energy_eV", region.get("baseline_energy_eV"))
    if baseline is None:
        baseline = float(np.min(selected_eig))
    baseline = float(baseline)

    split_energy = region_meta.get("split_energy_eV", region.get("split_energy_eV"))
    split_offset = region_meta.get("split_offset_eV", region.get("split_offset_eV"))
    left = region_meta.get("candidate_left_eV")
    right = region_meta.get("candidate_right_eV")

    if left is None:
        if region.get("sparse_left_eV") is not None:
            left = float(region["sparse_left_eV"])
        elif region.get("lower_edge_offset_eV") is not None:
            left = baseline + float(region["lower_edge_offset_eV"])
    if right is None:
        if region.get("sparse_right_eV") is not None:
            right = float(region["sparse_right_eV"])
        elif left is not None and region.get("sparse_width_eV") is not None:
            right = float(left) + float(region["sparse_width_eV"])
        elif left is not None and region.get("gap_width_eV") is not None:
            right = float(left) + float(region["gap_width_eV"])

    method = str(region.get("method", "unknown"))
    confidence = str(region.get("confidence", "unknown"))
    active_mode = active_state(set_dir)

    if emin_rel is None and isinstance(region.get("deeptb_window"), dict):
        proposed = region["deeptb_window"]
        if proposed.get("emin") is not None and proposed.get("emax") is not None:
            emin_rel = float(proposed["emin"])
            emax_rel = float(proposed["emax"])

    if emin_rel is None and isinstance(region.get("deeptb_energy_window"), dict):
        proposed = region["deeptb_energy_window"]
        if proposed.get("emin") is not None and proposed.get("emax") is not None:
            emin_rel = float(proposed["emin"])
            emax_rel = float(proposed["emax"])

    emin_abs = baseline + emin_rel if emin_rel is not None else None
    emax_abs = baseline + emax_rel if emax_rel is not None else None
    split_mismatch = None
    if split_energy is not None and emin_abs is not None:
        split_mismatch = abs(float(split_energy) - float(emin_abs))

    return {
        "baseline": baseline,
        "emin_rel": emin_rel,
        "emax_rel": emax_rel,
        "emin_abs": emin_abs,
        "emax_abs": emax_abs,
        "window_file": info_path.name,
        "method": method,
        "confidence": confidence,
        "active_mode": active_mode,
        "corridor_left": float(left) if left is not None else None,
        "corridor_right": float(right) if right is not None else None,
        "split_energy_eV": float(split_energy) if split_energy is not None else None,
        "split_offset_eV": float(split_offset) if split_offset is not None else None,
        "sparse_width_eV": region_meta.get("sparse_width_eV", region.get("sparse_width_eV")),
        "sparse_mean_crossings": region_meta.get("sparse_mean_crossings", region.get("sparse_mean_crossings")),
        "gap_width_eV": region_meta.get("gap_width_eV", region.get("gap_width_eV")),
        "split_mismatch_eV": split_mismatch,
    }


def plot_one_set(
    set_dir: Path,
    output_path: Path,
    *,
    mode: str,
    window_source: str,
    shade_color: str,
    k_tol: float,
    aims_dir_override: Optional[Path],
    dpi: int,
    linewidth: float,
    band_alpha: float,
) -> dict:
    selected_path = set_dir / "eigenvalues.npy"
    kpoints_path = set_dir / "kpoints.npy"
    structure_path = set_dir / "xdat.traj"
    for path in (selected_path, kpoints_path, structure_path, set_dir / ACTIVE_INFO):
        if not path.exists():
            raise FileNotFoundError(path)

    selected = load_eigenvalues(selected_path)
    kpoints = load_kpoints(kpoints_path)
    try:
        from ase.io import read
    except ImportError as exc:
        raise ImportError("ASE is required to read xdat.traj") from exc
    atoms = read(str(structure_path))

    if selected.shape[0] != kpoints.shape[0]:
        raise ValueError("Eigenvalue and k-point counts do not match")

    metadata = load_region_metadata(set_dir, selected, window_source)

    n_core = None
    n_all = None
    aims_dir = None
    if mode == "selected":
        spectrum = selected
    else:
        aims_dir = aims_dir_override or source_case_from_report(set_dir)
        if aims_dir is None:
            raise FileNotFoundError("Full mode needs conversion_report.json -> source_case or --aims-dir")
        aims_dir = aims_dir.resolve()
        if not aims_dir.is_dir():
            raise FileNotFoundError(aims_dir)
        spectrum, n_core, n_all = load_full_noncore_reference(aims_dir, atoms, kpoints, k_tol)
        ncheck = min(selected.shape[1], spectrum.shape[1])
        if ncheck:
            dmax = float(np.max(np.abs(selected[:, :ncheck] - spectrum[:, :ncheck])))
            if dmax > 2e-5:
                raise ValueError(f"Converted target is not the expected non-core prefix; max |dE|={dmax:.3e} eV")

    x = cumulative_k_distance(kpoints, atoms)
    hs = find_hbn_high_symmetry_points(kpoints, x)
    fig, ax = plt.subplots(figsize=(6.2, 5.0))

    c_left = metadata["corridor_left"]
    c_right = metadata["corridor_right"]
    shade = "#f4b860" if shade_color == "orange" else "#d9d9d9"
    if c_left is not None and c_right is not None and c_right > c_left:
        ax.axhspan(c_left, c_right, color=shade, alpha=0.28, zorder=0)

    for ib in range(spectrum.shape[1]):
        ax.plot(x, spectrum[:, ib], color="black", lw=linewidth, alpha=band_alpha, zorder=2)

    line_color = "#c62828"
    emin_abs = metadata["emin_abs"]
    emax_abs = metadata["emax_abs"]
    if emin_abs is not None:
        ax.axhline(emin_abs, color=line_color, ls="--", lw=1.2, zorder=3)
    if emax_abs is not None:
        ax.axhline(emax_abs, color=line_color, ls="--", lw=1.2, zorder=3)

    if hs:
        for _idx, xpos, _label in hs:
            ax.axvline(xpos, color="0.82", lw=0.7, zorder=1)
        ax.set_xticks([item[1] for item in hs])
        ax.set_xticklabels([item[2] for item in hs])
    else:
        ax.set_xlabel("k-path distance")

    ax.set_ylabel("Energy (eV)")
    ax.set_xlim(float(x[0]), float(x[-1]))

    y_candidates = [float(np.min(spectrum)), float(np.max(spectrum))]
    for value in (emin_abs, emax_abs, c_left, c_right):
        if value is not None:
            y_candidates.append(float(value))
    ymin, ymax = min(y_candidates), max(y_candidates)
    pad = max(0.8, 0.03 * max(ymax - ymin, 1.0))
    ax.set_ylim(ymin - pad, ymax + pad)

    title_mode = "selected target" if mode == "selected" else "full non-core"
    ax.set_title(
        f"{set_dir.name} | {title_mode} | {metadata['method']} ({metadata['confidence']}) | active: {metadata['active_mode']}",
        fontsize=9.4,
    )

    annotation = [f"E0 = {metadata['baseline']:.3f} eV"]
    if metadata["split_energy_eV"] is not None:
        annotation.append(f"split = {metadata['split_energy_eV']:.3f} eV")
    if metadata["split_offset_eV"] is not None:
        annotation.append(f"offset = {metadata['split_offset_eV']:.3f} eV")
    if metadata["sparse_width_eV"] is not None:
        annotation.append(f"region width = {float(metadata['sparse_width_eV']):.3f} eV")
    if metadata["sparse_mean_crossings"] is not None:
        annotation.append(f"mean crossings = {float(metadata['sparse_mean_crossings']):.2f}")
    if metadata["gap_width_eV"] is not None:
        annotation.append(f"gap width = {float(metadata['gap_width_eV']):.3f} eV")
    annotation.append(f"window: {metadata['window_file']}")
    if metadata["split_mismatch_eV"] is not None and metadata["split_mismatch_eV"] > 1e-6:
        annotation.append(f"WARNING split/emin mismatch = {metadata['split_mismatch_eV']:.2e} eV")

    ax.text(
        0.015,
        0.985,
        "\n".join(annotation),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.75", alpha=0.88),
        zorder=5,
    )

    xr = float(x[-1])
    if emin_abs is not None:
        ax.annotate(
            f"emin {metadata['emin_rel']:.3f}",
            xy=(xr, emin_abs),
            xytext=(-2, 3),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=7.2,
            color=line_color,
        )
    if emax_abs is not None:
        ax.annotate(
            f"emax {metadata['emax_rel']:.3f}",
            xy=(xr, emax_abs),
            xytext=(-2, -3),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=7.2,
            color=line_color,
        )

    handles = [Line2D([0], [0], color="black", lw=1.0, label="FHI-aims bands")]
    if c_left is not None and c_right is not None:
        handles.append(Patch(facecolor=shade, edgecolor="none", alpha=0.35, label="Low-density region"))
    if emin_abs is not None or emax_abs is not None:
        handles.append(Line2D([0], [0], color=line_color, ls="--", lw=1.2, label="emin / emax"))
    ax.legend(handles=handles, loc="lower right", fontsize=7.3, framealpha=0.9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return {
        "set": set_dir.name,
        "set_dir": str(set_dir),
        "mode": mode,
        "status": "ok",
        "method": metadata["method"],
        "confidence": metadata["confidence"],
        "active_mode": metadata["active_mode"],
        "window_file": metadata["window_file"],
        "baseline_energy_eV": metadata["baseline"],
        "emin_relative_eV": metadata["emin_rel"],
        "emax_relative_eV": metadata["emax_rel"],
        "emin_plot_eV": emin_abs,
        "emax_plot_eV": emax_abs,
        "corridor_left_eV": c_left,
        "corridor_right_eV": c_right,
        "split_mismatch_eV": metadata["split_mismatch_eV"],
        "n_selected_bands": int(selected.shape[1]),
        "n_plotted_bands": int(spectrum.shape[1]),
        "n_core_stripped": n_core,
        "n_all_electron_bands": n_all,
        "aims_dir": str(aims_dir) if aims_dir is not None else "",
        "plot": str(output_path),
    }


def make_contact_sheets(image_paths: list[Path], output_dir: Path, *, mode: str, cols: int, rows: int, dpi: int) -> list[Path]:
    if not image_paths:
        return []
    per_page = cols * rows
    pages = []
    for page_idx, start in enumerate(range(0, len(image_paths), per_page), start=1):
        batch = image_paths[start : start + per_page]
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.1, rows * 3.3))
        axes = np.asarray(axes, dtype=object).reshape(-1)
        for ax, path in zip(axes, batch):
            ax.imshow(plt.imread(path))
            ax.axis("off")
        for ax in axes[len(batch) :]:
            ax.axis("off")
        fig.suptitle(f"Band-region QA | {mode} | page {page_idx}", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        out = output_dir / f"band_region_contact_{mode}_{page_idx:02d}.png"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        pages.append(out)
    return pages


def write_summary(output_dir: Path, records: list[dict]) -> Path:
    path = output_dir / "band_region_visualization_summary.csv"
    fields = [
        "set", "set_dir", "mode", "status", "method", "confidence", "active_mode",
        "window_file", "baseline_energy_eV", "emin_relative_eV", "emax_relative_eV",
        "emin_plot_eV", "emax_plot_eV", "corridor_left_eV", "corridor_right_eV",
        "split_mismatch_eV", "n_selected_bands", "n_plotted_bands",
        "n_core_stripped", "n_all_electron_bands", "aims_dir", "plot", "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            writer.writerow({key: rec.get(key, "") for key in fields})
    return path


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
    found.sort(key=natural_key)
    if not found:
        raise FileNotFoundError(f"No converted set.* directories found below {root}")
    return found


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visual QA for DeePTB band-region weighting metadata.")
    p.add_argument("path", type=Path, help="One set.* directory or a dataset root.")
    p.add_argument("--mode", choices=("selected", "full"), default="selected")
    p.add_argument(
        "--window-source",
        choices=("weighted", "active"),
        default="weighted",
        help="Use canonical info.weighted.json when available, or the active info.json. Default: weighted.",
    )
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--aims-dir", type=Path, default=None, help="FHI-aims directory override for a single-set full-mode run.")
    p.add_argument("--shade-color", choices=("orange", "gray"), default="orange")
    p.add_argument("--k-tol", type=float, default=2e-6)
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--linewidth", type=float, default=0.55)
    p.add_argument("--band-alpha", type=float, default=0.72)
    p.add_argument("--contact-sheet", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--contact-cols", type=int, default=4)
    p.add_argument("--contact-rows", type=int, default=4)
    p.add_argument("--contact-dpi", type=int, default=130)
    args = p.parse_args()
    if args.k_tol <= 0:
        p.error("--k-tol must be > 0")
    if args.dpi < 50:
        p.error("--dpi must be >= 50")
    if args.linewidth <= 0:
        p.error("--linewidth must be > 0")
    if not 0 < args.band_alpha <= 1:
        p.error("--band-alpha must be in (0,1]")
    if args.contact_cols < 1 or args.contact_rows < 1:
        p.error("contact-sheet rows/columns must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    root = args.path.resolve()
    set_dirs = find_set_dirs(root, args.recursive)
    if args.aims_dir is not None and len(set_dirs) != 1:
        raise ValueError("--aims-dir is only supported for a single-set run")

    output_dir = (
        args.output.resolve()
        if args.output is not None
        else root / f"band_region_visualization_{args.mode}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    images = []
    for idx, set_dir in enumerate(set_dirs, start=1):
        print(f"[{idx:>3}/{len(set_dirs)}] {set_dir.name}")
        out = output_dir / f"{set_dir.name}_{args.mode}.png"
        try:
            rec = plot_one_set(
                set_dir,
                out,
                mode=args.mode,
                window_source=args.window_source,
                shade_color=args.shade_color,
                k_tol=args.k_tol,
                aims_dir_override=args.aims_dir,
                dpi=args.dpi,
                linewidth=args.linewidth,
                band_alpha=args.band_alpha,
            )
        except Exception as exc:
            print(f"    ERROR: {type(exc).__name__}: {exc}")
            rec = {
                "set": set_dir.name,
                "set_dir": str(set_dir),
                "mode": args.mode,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        else:
            images.append(out)
            print(
                f"    {rec['method']} ({rec['confidence']}), "
                f"active={rec['active_mode']}, "
                f"window={rec['window_file']}"
            )
        records.append(rec)

    summary = write_summary(output_dir, records)
    print(f"\nWrote summary: {summary}")

    if args.contact_sheet and len(images) > 1:
        pages = make_contact_sheets(
            images,
            output_dir,
            mode=args.mode,
            cols=args.contact_cols,
            rows=args.contact_rows,
            dpi=args.contact_dpi,
        )
        print(f"Wrote {len(pages)} contact sheet(s) to: {output_dir}")

    n_ok = sum(r.get("status") == "ok" for r in records)
    n_err = sum(r.get("status") == "error" for r in records)
    print(f"Processed {len(records)} set(s): {n_ok} plotted, {n_err} error(s).")


if __name__ == "__main__":
    main()
