"""
Plot and compare FHI-aims DFT bands with a DeePTB checkpoint on the same
k-point path for one converted structure.

Usage
-----
Compare only the supervised training window::

    python scripts/band_plot.py \
        --checkpoint RUN/checkpoint/nnsk.epN.pth \
        --set DATA/set.000000 \
        --output RESULTS/band_plot \
        --mode supervised

Compare all available non-core DFT bands::

    python scripts/band_plot.py \
        --checkpoint RUN/checkpoint/nnsk.epN.pth \
        --set DATA/set.000000 \
        --output RESULTS/band_plot_full \
        --mode full

In ``full`` mode the original FHI-aims directory is read from
``set/conversion_report.json`` (``source_case``) unless ``--aims-dir`` is
provided explicitly.

Modes
-----
``supervised`` uses the converted ``eigenvalues.npy`` target that entered the
training loss. ``full`` reloads the original FHI-aims ``band1*.out`` files,
strips the inferred H/B/C/N/O 1s core bands, and compares
``min(N_DFT_noncore, N_DeePTB)`` bands.  Full mode reports supervised and
above-supervision errors separately when both regions are available.

By default, the DFT and DeePTB spectra are independently shifted by their
minimum energy before metrics are calculated.  The high-symmetry labeling is
specialized for the hBN M-Gamma-K-M path used by this project.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
from ase.io import read

from dptb.nn.build import build_model
from dptb.postprocess.bandstructure.band import Band


# One frozen 1s spatial core orbital per B/C/N/O atom; H contributes none.
# Keep this chemistry-specific rule consistent with aims_to_deeptb.py.
CORE_BANDS_PER_ATOM: Dict[str, int] = {
    "H": 0,
    "B": 1,
    "C": 1,
    "N": 1,
    "O": 1,
}


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_eigenvalues_npy(path: Path) -> np.ndarray:
    """Load (1,nk,nb) or (nk,nb) eigenvalue arrays as (nk,nb)."""
    eig = np.load(path)
    if eig.ndim == 3:
        if eig.shape[0] != 1:
            raise ValueError(
                f"Expected one-frame eigenvalues (1,nk,nb), got {eig.shape}"
            )
        eig = eig[0]
    elif eig.ndim != 2:
        raise ValueError(
            f"Expected eigenvalues shape (1,nk,nb) or (nk,nb), got {eig.shape}"
        )
    return np.asarray(eig, dtype=float)


def _f(x: str) -> float:
    return float(x.replace("D", "E").replace("d", "e"))


def natural_key(path: str | Path):
    s = Path(path).name
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def parse_numeric_band_file(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse standard FHI-aims band*.out rows: ik kx ky kz occ E occ E ..."""
    rows = []
    with path.open("r", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            toks = stripped.split()
            try:
                vals = [_f(t) for t in toks]
            except ValueError:
                continue
            if len(vals) < 6 or (len(vals) - 4) % 2 != 0:
                continue
            rows.append(vals)

    if not rows:
        raise ValueError(f"{path}: no standard numeric FHI-aims band rows found")

    nbands_set = {(len(vals) - 4) // 2 for vals in rows}
    if len(nbands_set) != 1:
        raise ValueError(
            f"{path}: inconsistent band counts across rows: {sorted(nbands_set)}"
        )
    nbands = nbands_set.pop()

    kpoints, occupations, eigenvalues = [], [], []
    for vals in rows:
        kpoints.append(vals[1:4])
        pairs = np.asarray(vals[4:], dtype=float).reshape(nbands, 2)
        occupations.append(pairs[:, 0])
        eigenvalues.append(pairs[:, 1])

    return (
        np.asarray(kpoints, dtype=float),
        np.asarray(occupations, dtype=float),
        np.asarray(eigenvalues, dtype=float),
    )


def combine_aims_segments(files: List[Path], atol: float = 1e-8):
    """Natural-order and concatenate FHI-aims band path segments, deduplicating joins."""
    all_k, all_occ, all_eig = [], [], []
    last_k = None
    nbands = None

    for path in sorted(files, key=natural_key):
        kp, occ, eig = parse_numeric_band_file(path)
        if nbands is None:
            nbands = eig.shape[1]
        elif eig.shape[1] != nbands:
            raise ValueError(
                f"Band-count mismatch: {path.name} has {eig.shape[1]}, expected {nbands}"
            )

        start = 0
        if last_k is not None and len(kp) and np.allclose(last_k, kp[0], atol=atol, rtol=0):
            start = 1

        if start < len(kp):
            all_k.append(kp[start:])
            all_occ.append(occ[start:])
            all_eig.append(eig[start:])
            last_k = kp[-1].copy()

    if not all_k:
        raise ValueError("No FHI-aims k-points remain after combining segments")

    return (
        np.concatenate(all_k, axis=0),
        np.concatenate(all_occ, axis=0),
        np.concatenate(all_eig, axis=0),
    )


def infer_core_band_count(symbols: List[str]) -> int:
    unsupported = sorted(set(symbols) - set(CORE_BANDS_PER_ATOM))
    if unsupported:
        raise ValueError(
            "Cannot safely infer 1s core-band count for elements: "
            + ", ".join(unsupported)
        )
    return int(sum(CORE_BANDS_PER_ATOM[s] for s in symbols))


def source_case_from_report(set_dir: Path) -> Path | None:
    report = set_dir / "conversion_report.json"
    if not report.exists():
        return None
    try:
        data = json.loads(report.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Could not read {report}: {exc}") from exc
    source = data.get("source_case")
    if not source:
        return None
    return Path(source)


def load_full_noncore_aims_reference(
    aims_dir: Path,
    atoms,
    expected_kpoints: np.ndarray,
    k_tol: float = 2e-6,
) -> tuple[np.ndarray, int, int]:
    """
    Load all available spin-1 FHI-aims bands and strip inferred 1s core bands.

    Returns
    -------
    eigenvalues_noncore : (nk, n_noncore)
    n_core              : number of stripped core bands
    n_all_dft           : all-electron band count before stripping
    """
    if not aims_dir.exists():
        raise FileNotFoundError(f"FHI-aims source directory not found: {aims_dir}")

    band1 = sorted(aims_dir.glob("band1*.out"), key=natural_key)
    band2 = sorted(aims_dir.glob("band2*.out"), key=natural_key)

    if not band1:
        raise FileNotFoundError(f"No band1*.out files found in {aims_dir}")
    if band2:
        raise RuntimeError(
            f"Found {len(band2)} band2*.out files in {aims_dir}. "
            "band_plot expects the spin-unpolarized/single-channel "
            "workflow used for the present DeePTB training; inspect spin handling first."
        )

    kp_raw, _occ_raw, eig_raw = combine_aims_segments(band1)

    if kp_raw.shape != expected_kpoints.shape:
        raise ValueError(
            "Raw FHI-aims and converted k-point shapes differ: "
            f"{kp_raw.shape} vs {expected_kpoints.shape}"
        )
    max_kdiff = float(np.max(np.abs(kp_raw - expected_kpoints)))
    if max_kdiff > k_tol:
        raise ValueError(
            "Raw FHI-aims k-points do not match converted kpoints.npy; "
            f"max |dk| = {max_kdiff:.3e} > {k_tol:.3e}"
        )

    symbols = list(atoms.get_chemical_symbols())
    n_core = infer_core_band_count(symbols)
    n_all = int(eig_raw.shape[1])
    if n_core >= n_all:
        raise ValueError(
            f"Inferred {n_core} core bands but raw DFT contains only {n_all} bands"
        )

    return np.asarray(eig_raw[:, n_core:], dtype=float), n_core, n_all


def cumulative_k_distance(kpoints_frac: np.ndarray, atoms) -> np.ndarray:
    reciprocal = 2.0 * np.pi * np.asarray(atoms.cell.reciprocal())
    k_cart = np.asarray(kpoints_frac) @ reciprocal
    dk = np.linalg.norm(np.diff(k_cart, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(dk)))


def find_hbn_high_symmetry_points(
    kpoints: np.ndarray,
    xlist: np.ndarray,
    tolerance: float = 2.0e-3,
):
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
        dmin = float(dist[idx])
        if dmin > tolerance:
            raise ValueError(
                f"Could not identify {label}: nearest k-point is {kpoints[idx]} "
                f"at distance {dmin:.6g}"
            )
        found.append(
            dict(
                label=label,
                index=idx,
                distance=dmin,
                kpoint=kpoints[idx].copy(),
                x=float(xlist[idx]),
            )
        )
    found.sort(key=lambda item: item["index"])
    indices = [p["index"] for p in found]
    if len(set(indices)) != len(indices):
        raise ValueError(f"Duplicate high-symmetry indices detected: {indices}")
    return found


def choose_band_window(n_available: int, first_band: int | None, last_band: int | None):
    """Convert 1-based inclusive CLI band numbers to a Python slice."""
    first = 1 if first_band is None else first_band
    last = n_available if last_band is None else last_band
    if first < 1:
        raise ValueError("--first-band must be >= 1")
    if last < first:
        raise ValueError("--last-band must be >= --first-band")
    if last > n_available:
        raise ValueError(
            f"Requested band {last}, but only {n_available} comparable bands exist "
            f"in mode={CURRENT_MODE_FOR_ERROR}."
        )
    return first - 1, last


# Only used to make choose_band_window error messages more informative.
CURRENT_MODE_FOR_ERROR = "unknown"


def metrics(pred: np.ndarray, ref: np.ndarray):
    if pred.size == 0 or ref.size == 0:
        return None
    residual = pred - ref
    mse = float(np.mean(residual ** 2))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(residual))),
        "max_abs": float(np.max(np.abs(residual))),
    }


def write_metrics(f, title: str, m):
    f.write(f"\n{title}:\n")
    if m is None:
        f.write("  not available\n")
        return
    f.write(f"  MSE:      {m['mse']:.10f} eV^2\n")
    f.write(f"  RMSE:     {m['rmse']:.10f} eV\n")
    f.write(f"  MAE:      {m['mae']:.10f} eV\n")
    f.write(f"  Max |dE|: {m['max_abs']:.10f} eV\n")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Overlay FHI-aims and DeePTB bands using either the supervised "
            "training window or all available non-core DFT bands."
        )
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--set", dest="set_dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--mode",
        choices=("supervised", "full"),
        default="supervised",
        help=(
            "supervised: reproduce the converted training target; "
            "full: compare all available non-core FHI-aims bands against "
            "DeePTB up to min(N_DFT_noncore, N_DeePTB). Default: supervised"
        ),
    )
    p.add_argument(
        "--aims-dir",
        default=None,
        help=(
            "Original FHI-aims calculation directory for --mode full. If omitted, "
            "read source_case from set/conversion_report.json."
        ),
    )
    p.add_argument("--r-max", type=float, default=5.0)
    p.add_argument("--oer-max", type=float, default=4.0)
    p.add_argument(
        "--align",
        choices=("loss", "none"),
        default="loss",
        help="loss: independently subtract compared-spectrum minima; none: raw energies",
    )
    p.add_argument("--first-band", type=int, default=None)
    p.add_argument("--last-band", type=int, default=None)
    p.add_argument("--emin", type=float, default=None)
    p.add_argument("--emax", type=float, default=None)
    p.add_argument("--ref-stride", type=int, default=4)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--title", default=None)
    return p.parse_args()


def main():
    global CURRENT_MODE_FOR_ERROR
    args = parse_args()
    CURRENT_MODE_FOR_ERROR = args.mode

    checkpoint = Path(args.checkpoint)
    set_dir = Path(args.set_dir)
    output_dir = Path(args.output)

    structure_path = set_dir / "xdat.traj"
    kpoints_path = set_dir / "kpoints.npy"
    supervised_path = set_dir / "eigenvalues.npy"

    for path in (checkpoint, structure_path, kpoints_path, supervised_path):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
    if args.ref_stride < 1:
        raise ValueError("--ref-stride must be >= 1")

    output_dir.mkdir(parents=True, exist_ok=True)

    atoms = read(str(structure_path))
    kpoints = np.asarray(np.load(kpoints_path), dtype=float)
    ref_supervised_raw = load_eigenvalues_npy(supervised_path)
    n_supervised = int(ref_supervised_raw.shape[1])

    if kpoints.ndim != 2 or kpoints.shape[1] != 3:
        raise ValueError(f"Expected kpoints.npy shape (nk,3), got {kpoints.shape}")
    if ref_supervised_raw.shape[0] != kpoints.shape[0]:
        raise ValueError(
            f"Supervised DFT has {ref_supervised_raw.shape[0]} k rows; "
            f"kpoints.npy has {kpoints.shape[0]}"
        )

    aims_dir = None
    n_core_stripped = None
    n_dft_all_electron = None

    if args.mode == "supervised":
        ref_raw = ref_supervised_raw.copy()
        reference_description = "converted supervised DFT target"
    else:
        if args.aims_dir is not None:
            aims_dir = Path(args.aims_dir)
        else:
            aims_dir = source_case_from_report(set_dir)
            if aims_dir is None:
                raise FileNotFoundError(
                    "--mode full needs the original FHI-aims data. Supply --aims-dir, "
                    "or ensure set/conversion_report.json contains source_case."
                )

        ref_raw, n_core_stripped, n_dft_all_electron = load_full_noncore_aims_reference(
            aims_dir=aims_dir,
            atoms=atoms,
            expected_kpoints=kpoints,
        )
        reference_description = "all available FHI-aims non-core bands"

        # Sanity check: the converter target should be the prefix of the full
        # non-core DFT spectrum, apart from tiny text/NumPy roundoff.
        ncheck = min(n_supervised, ref_raw.shape[1])
        if ncheck:
            max_prefix_diff = float(
                np.max(np.abs(ref_supervised_raw[:, :ncheck] - ref_raw[:, :ncheck]))
            )
            if max_prefix_diff > 2e-5:
                raise ValueError(
                    "The supervised eigenvalues.npy is not the expected prefix of the "
                    "full non-core FHI-aims spectrum. max |dE| = "
                    f"{max_prefix_diff:.3e} eV. Check source_case/core stripping."
                )

    xlist = cumulative_k_distance(kpoints, atoms)
    hs_points = find_hbn_high_symmetry_points(kpoints, xlist)
    hs_x = np.asarray([p["x"] for p in hs_points], dtype=float)
    hs_labels = [p["label"] for p in hs_points]

    print(f"Loading DeePTB model: {checkpoint}")
    model = build_model(checkpoint=str(checkpoint))
    print(f"Model device: {model.device}")

    kpath_kwargs = {
        "kline_type": "array",
        "kpath": kpoints,
        "xlist": xlist,
        "high_sym_kpoints": hs_x,
        "labels": hs_labels,
    }
    atomic_data_options = {
        "r_max": args.r_max,
        "oer_max": args.oer_max,
        "pbc": True,
    }

    bcal = Band(
        model=model,
        use_gui=False,
        results_path=str(output_dir),
        device=model.device,
    )
    eigenstatus = bcal.get_bands(
        data=str(structure_path),
        kpath_kwargs=kpath_kwargs,
        AtomicData_options=atomic_data_options,
    )
    pred_raw = to_numpy(eigenstatus["eigenvalues"])
    if pred_raw.ndim == 3:
        if pred_raw.shape[0] != 1:
            raise ValueError(f"Unexpected DeePTB eigenvalue shape {pred_raw.shape}")
        pred_raw = pred_raw[0]
    if pred_raw.ndim != 2:
        raise ValueError(f"Expected DeePTB eigenvalues (nk,nb), got {pred_raw.shape}")
    if pred_raw.shape[0] != kpoints.shape[0]:
        raise ValueError(
            f"Predicted k rows {pred_raw.shape[0]} != converted k rows {kpoints.shape[0]}"
        )

    # Common practice for eigenvalue-by-index comparison: use the lower of the
    # two available band counts.  In full mode the DFT count is already non-core.
    n_dft_reference = int(ref_raw.shape[1])
    n_dptb = int(pred_raw.shape[1])
    n_compare = min(n_dft_reference, n_dptb)

    ref_compare = ref_raw[:, :n_compare].copy()
    pred_compare = pred_raw[:, :n_compare].copy()

    if args.align == "loss":
        ref_compare -= np.min(ref_compare)
        pred_compare -= np.min(pred_compare)
        ylabel = r"$E-E_{\min}$ (eV)"
    else:
        ylabel = "Energy (eV)"

    m_common = metrics(pred_compare, ref_compare)

    # Always calculate the training-window metric on the same alignment used
    # for the selected plot.  In full mode this lets us compare trained vs unseen
    # spectral regions in one report.
    n_sup_common = min(n_supervised, n_compare)
    m_supervised = metrics(
        pred_compare[:, :n_sup_common],
        ref_compare[:, :n_sup_common],
    )
    if n_compare > n_supervised:
        m_extrap = metrics(
            pred_compare[:, n_supervised:n_compare],
            ref_compare[:, n_supervised:n_compare],
        )
    else:
        m_extrap = None

    b0, b1 = choose_band_window(n_compare, args.first_band, args.last_band)
    ref_plot = ref_compare[:, b0:b1]
    pred_plot = pred_compare[:, b0:b1]
    m_plot = metrics(pred_plot, ref_plot)

    # Save both raw/model outputs and the exact aligned arrays used for comparison.
    np.save(output_dir / "deeptb_eigenvalues_raw.npy", pred_raw)
    np.save(output_dir / f"dft_eigenvalues_{args.mode}_raw.npy", ref_raw)
    np.save(output_dir / f"deeptb_eigenvalues_{args.mode}_compared.npy", pred_compare)
    np.save(output_dir / f"dft_eigenvalues_{args.mode}_compared.npy", ref_compare)
    np.save(output_dir / "kpoints.npy", kpoints)
    np.save(output_dir / "xlist.npy", xlist)

    # ------------------------------------------------------------------
    # Plot: same bottom-to-top color sequence for DFT and DeePTB band j.
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    cmap = plt.get_cmap("turbo")
    stride = args.ref_stride

    for j in range(ref_plot.shape[1]):
        global_band_idx = b0 + j
        color = cmap(global_band_idx / (n_compare - 1)) if n_compare > 1 else cmap(0.5)

        ax.plot(
            xlist,
            pred_plot[:, j],
            color=color,
            linewidth=0.8,
            alpha=0.95,
        )
        ax.plot(
            xlist[::stride],
            ref_plot[::stride, j],
            linestyle="None",
            marker="o",
            color=color,
            markersize=2.4,
            alpha=0.75,
        )

    for x in hs_x[1:-1]:
        ax.axvline(x, linewidth=0.6, linestyle="--", alpha=0.7, color="gray")

    ax.set_xticks(hs_x)
    ax.set_xticklabels(hs_labels)
    ax.set_xlim(xlist[0], xlist[-1])
    ax.set_ylabel(ylabel)
    ax.set_xlabel("k-path")

    if args.emin is not None or args.emax is not None:
        current_ymin, current_ymax = ax.get_ylim()
        ymin = current_ymin if args.emin is None else args.emin
        ymax = current_ymax if args.emax is None else args.emax
        if ymin >= ymax:
            raise ValueError("Need emin < emax")
        ax.set_ylim(ymin, ymax)

    if args.title is not None:
        ax.set_title(args.title)
    else:
        scope_label = "supervised" if args.mode == "supervised" else "full non-core"
        ax.set_title(
            f"{set_dir.name}: DFT vs DeePTB ({scope_label})\n"
            f"bands {b0 + 1}-{b1}, RMSE = {m_plot['rmse']:.3f} eV"
        )

    legend_handles = [
        Line2D([0], [0], color="black", lw=1.2, label="DeePTB"),
        Line2D(
            [0], [0], color="black", marker="o", linestyle="None",
            markersize=4, label="DFT (FHI-aims)"
        ),
    ]
    ax.legend(handles=legend_handles, loc="best")
    ax.tick_params(direction="in")
    fig.tight_layout()

    png_path = output_dir / f"band_compare_{args.mode}.png"
    pdf_path = output_dir / f"band_compare_{args.mode}.pdf"
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    report_path = output_dir / f"comparison_report_{args.mode}.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("DFT vs DeePTB band comparison — band_plot\n")
        f.write("=" * 56 + "\n")
        f.write(f"mode: {args.mode}\n")
        f.write(f"checkpoint: {checkpoint}\n")
        f.write(f"set_dir: {set_dir}\n")
        f.write(f"structure: {structure_path}\n")
        f.write(f"reference: {reference_description}\n")
        if aims_dir is not None:
            f.write(f"FHI-aims source: {aims_dir}\n")
        if n_core_stripped is not None:
            f.write(f"DFT all-electron bands available: {n_dft_all_electron}\n")
            f.write(f"1s core bands stripped for plot: {n_core_stripped}\n")
        f.write(f"nkpoints: {kpoints.shape[0]}\n")
        f.write(f"supervised DFT band count: {n_supervised}\n")
        f.write(f"DFT bands available in selected mode: {n_dft_reference}\n")
        f.write(f"DeePTB raw bands: {n_dptb}\n")
        f.write(f"common upper band = min(DFT, DeePTB): {n_compare}\n")
        f.write(f"plotted bands: {b0 + 1}-{b1}\n")
        f.write(f"alignment: {args.align}\n")
        if args.mode == "full":
            f.write(f"supervised cutoff within non-core indexing: band {n_supervised}\n")
            f.write(
                f"bands above training cutoff available for comparison: "
                f"{max(0, n_compare - n_supervised)}\n"
            )

        f.write("\nHigh-symmetry points:\n")
        for point in hs_points:
            f.write(
                f"  {point['label']:8s} index={point['index']:3d} "
                f"k={point['kpoint']} x={point['x']:.8f} "
                f"target_distance={point['distance']:.3e}\n"
            )

        write_metrics(f, "Common-window metrics", m_common)
        write_metrics(f, f"Supervised-region metrics (bands 1-{n_sup_common})", m_supervised)
        if args.mode == "full":
            write_metrics(
                f,
                f"Above-supervision extrapolation metrics (bands {n_supervised + 1}-{n_compare})",
                m_extrap,
            )
        write_metrics(f, f"Plotted-window metrics (bands {b0 + 1}-{b1})", m_plot)

    print("\nHigh-symmetry path detected:")
    for point in hs_points:
        print(
            f"  index {point['index']:3d}: {point['label']:8s} "
            f"k = {point['kpoint']}"
        )

    print("\nComparison:")
    print(f"  Mode:                         {args.mode}")
    print(f"  k-points:                     {kpoints.shape[0]}")
    print(f"  Supervised DFT bands:         {n_supervised}")
    if n_core_stripped is not None:
        print(f"  DFT all-electron bands:       {n_dft_all_electron}")
        print(f"  DFT 1s core bands stripped:   {n_core_stripped}")
    print(f"  DFT bands in chosen mode:     {n_dft_reference}")
    print(f"  DeePTB raw bands:             {n_dptb}")
    print(f"  Common bands = min(...):      {n_compare}")
    print(f"  Plotted bands:                {b0 + 1}-{b1}")
    if args.mode == "full":
        print(f"  Above-supervision bands:      {max(0, n_compare - n_supervised)}")
    print(f"  Alignment:                    {args.align}")
    print(f"  Common-window RMSE:           {m_common['rmse']:.6f} eV")
    print(f"  Supervised-region RMSE:       {m_supervised['rmse']:.6f} eV")
    if m_extrap is not None:
        print(f"  Above-supervision RMSE:       {m_extrap['rmse']:.6f} eV")
    print(f"\nSaved plot:   {png_path}")
    print(f"Saved PDF:    {pdf_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
