"""
Evaluate DeePTB error band-by-band for one hBN structure, optionally comparing
two checkpoints and including non-core DFT bands above the supervised cutoff.

Usage
-----
Evaluate one checkpoint over the full available non-core DFT spectrum::

    python evaluation/evaluate_band_error.py \
        --checkpoint RUN/checkpoint/nnsk.epN.pth \
        --set DATA/set.000000 \
        --output RESULTS/band_error_epN \
        --mode full

Compare two checkpoints on the same structure::

    python evaluation/evaluate_band_error.py \
        --checkpoint RUN_A/checkpoint/nnsk.epN.pth \
        --checkpoint2 RUN_B/checkpoint/nnsk.epM.pth \
        --set DATA/set.000000 \
        --output RESULTS/band_error_compare \
        --mode full

Modes
-----
``full`` reloads the original FHI-aims ``band1*.out`` files, strips inferred
H/B/C/N/O 1s core bands, and evaluates the common non-core spectrum.  The
number of columns in ``set/eigenvalues.npy`` defines the supervised cutoff.
``supervised`` evaluates only that converted training target.

With ``--align loss`` (default), the DFT and each prediction are shifted using
the minimum energy of their supervised windows; the same fixed shifts are then
applied to unsupervised bands.  Full mode requires ``--aims-dir`` or a
``source_case`` entry in ``conversion_report.json``.  Cases with ``band2*.out``
are rejected rather than silently assuming a spin treatment.

The script writes per-band errors, supervised/unsupervised aggregate metrics,
RMSE-versus-band plots, and (for two checkpoints) a delta-RMSE comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.io import read

from dptb.nn.build import build_model
from dptb.postprocess.bandstructure.band import Band


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


def safe_label(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return text.strip("_") or "checkpoint"


def infer_checkpoint_label(path: Path) -> str:
    m = re.search(r"ep(\d+)", path.name, flags=re.IGNORECASE)
    if m:
        return f"ep{m.group(1)}"
    return path.stem


def load_eigenvalues_npy(path: Path) -> np.ndarray:
    eig = np.load(path)
    if eig.ndim == 3:
        if eig.shape[0] != 1:
            raise ValueError(f"Expected one frame, got eigenvalues shape {eig.shape}")
        eig = eig[0]
    elif eig.ndim != 2:
        raise ValueError(
            f"Expected eigenvalues.npy shape (1,nk,nb) or (nk,nb), got {eig.shape}"
        )
    return np.asarray(eig, dtype=float)


def _f(x: str) -> float:
    return float(x.replace("D", "E").replace("d", "e"))


def natural_key(path: str | Path):
    s = Path(path).name
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def parse_numeric_band_file(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse FHI-aims rows: ik kx ky kz occ E occ E ..."""
    rows = []
    with path.open("r", errors="replace") as fh:
        for line in fh:
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
        raise ValueError("No FHI-aims k-points remain after joining band segments")

    return (
        np.concatenate(all_k, axis=0),
        np.concatenate(all_occ, axis=0),
        np.concatenate(all_eig, axis=0),
    )


def infer_core_band_count(symbols: List[str]) -> int:
    unsupported = sorted(set(symbols) - set(CORE_BANDS_PER_ATOM))
    if unsupported:
        raise ValueError(
            "Cannot infer core stripping for elements: " + ", ".join(unsupported)
        )
    return int(sum(CORE_BANDS_PER_ATOM[s] for s in symbols))


def source_case_from_report(set_dir: Path) -> Path | None:
    report = set_dir / "conversion_report.json"
    if not report.exists():
        return None
    data = json.loads(report.read_text(encoding="utf-8"))
    source = data.get("source_case")
    return Path(source) if source else None


def load_full_noncore_reference(
    aims_dir: Path,
    atoms,
    expected_kpoints: np.ndarray,
    k_tol: float,
):
    band1 = sorted(aims_dir.glob("band1*.out"), key=natural_key)
    band2 = sorted(aims_dir.glob("band2*.out"), key=natural_key)

    if not band1:
        raise FileNotFoundError(f"No band1*.out files found in {aims_dir}")
    if band2:
        raise RuntimeError(
            f"Found {len(band2)} band2*.out files in {aims_dir}. "
            "This evaluator assumes a single-channel workflow."
        )

    kp, _occ, eig_all = combine_aims_segments(band1)
    if kp.shape != expected_kpoints.shape:
        raise ValueError(
            f"Raw FHI-aims k-point shape {kp.shape} != converted {expected_kpoints.shape}"
        )
    max_kdiff = float(np.max(np.abs(kp - expected_kpoints)))
    if max_kdiff > k_tol:
        raise ValueError(
            f"Raw/converted k-points differ: max |dk|={max_kdiff:.3e} > {k_tol:.3e}"
        )

    n_core = infer_core_band_count(list(atoms.get_chemical_symbols()))
    n_all = int(eig_all.shape[1])
    if n_core >= n_all:
        raise ValueError(f"Inferred {n_core} core bands but DFT contains only {n_all}")

    return np.asarray(eig_all[:, n_core:], dtype=float), n_core, n_all


def predict_checkpoint(
    checkpoint: Path,
    structure_path: Path,
    kpoints: np.ndarray,
    output_dir: Path,
    r_max: float,
    oer_max: float,
) -> np.ndarray:
    work = output_dir / f"_work_{safe_label(infer_checkpoint_label(checkpoint))}"
    work.mkdir(parents=True, exist_ok=True)

    model = build_model(checkpoint=str(checkpoint))
    print(f"Loaded {checkpoint} on {model.device}")

    # x coordinates are irrelevant to eigenvalues; a simple monotonic list is
    # sufficient because we are evaluating exactly the supplied k-point array.
    xlist = np.arange(kpoints.shape[0], dtype=float)
    kpath_kwargs = {
        "kline_type": "array",
        "kpath": kpoints,
        "xlist": xlist,
        "high_sym_kpoints": None,
        "labels": None,
    }
    atomic_data_options = {
        "r_max": r_max,
        "oer_max": oer_max,
        "pbc": True,
    }

    bcal = Band(
        model=model,
        use_gui=False,
        results_path=str(work),
        device=model.device,
    )
    status = bcal.get_bands(
        data=str(structure_path),
        kpath_kwargs=kpath_kwargs,
        AtomicData_options=atomic_data_options,
    )

    pred = to_numpy(status["eigenvalues"])
    if pred.ndim == 3:
        if pred.shape[0] != 1:
            raise ValueError(f"Unexpected DeePTB eigenvalue shape {pred.shape}")
        pred = pred[0]
    if pred.ndim != 2:
        raise ValueError(f"Expected DeePTB eigenvalues (nk,nb), got {pred.shape}")
    if pred.shape[0] != kpoints.shape[0]:
        raise ValueError(
            f"DeePTB k-point count {pred.shape[0]} != DFT {kpoints.shape[0]}"
        )
    return np.asarray(pred, dtype=float)


def align_spectra(
    pred: np.ndarray,
    ref: np.ndarray,
    n_supervised: int,
    method: str,
):
    """
    Return aligned copies plus the shifts that were subtracted.

    loss:
        Use the minimum of the supervised region for each spectrum.  This is
        the energy gauge optimized by the DeePTB EigLoss, and the
        same shifts are then applied to the unsupervised bands.
    full-min:
        Use each complete common spectrum's minimum.
    none:
        No alignment.
    """
    pred_a = pred.copy()
    ref_a = ref.copy()

    if method == "none":
        return pred_a, ref_a, 0.0, 0.0

    if method == "loss":
        n = min(n_supervised, pred.shape[1], ref.shape[1])
        if n < 1:
            raise ValueError("No supervised bands available for loss alignment")
        pred_shift = float(np.min(pred[:, :n]))
        ref_shift = float(np.min(ref[:, :n]))
    elif method == "full-min":
        pred_shift = float(np.min(pred))
        ref_shift = float(np.min(ref))
    else:
        raise ValueError(f"Unknown alignment method: {method}")

    pred_a -= pred_shift
    ref_a -= ref_shift
    return pred_a, ref_a, pred_shift, ref_shift


def aggregate_metrics(residual: np.ndarray):
    if residual.size == 0:
        return None
    mse = float(np.mean(residual ** 2))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "max_abs": float(np.max(np.abs(residual))),
    }


def band_metrics(pred: np.ndarray, ref: np.ndarray):
    residual = pred - ref
    mse = np.mean(residual ** 2, axis=0)
    return {
        "rmse": np.sqrt(mse),
        "mae": np.mean(np.abs(residual), axis=0),
        "bias": np.mean(residual, axis=0),
        "max_abs": np.max(np.abs(residual), axis=0),
        "ref_mean": np.mean(ref, axis=0),
        "pred_mean": np.mean(pred, axis=0),
    }


def region_metrics(pred: np.ndarray, ref: np.ndarray, n_supervised: int):
    n = pred.shape[1]
    ns = min(n_supervised, n)
    return {
        "full": aggregate_metrics(pred - ref),
        "supervised": aggregate_metrics(pred[:, :ns] - ref[:, :ns]),
        "unsupervised": (
            aggregate_metrics(pred[:, ns:] - ref[:, ns:]) if n > ns else None
        ),
        "n_supervised_common": ns,
        "n_unsupervised_common": max(0, n - ns),
    }


def write_csv(
    path: Path,
    ref: np.ndarray,
    checkpoint_data: List[dict],
    n_supervised: int,
    n_common: int,
):
    fields = [
        "band_index",
        "region",
        "dft_mean_aligned_eV",
        "dft_min_aligned_eV",
        "dft_max_aligned_eV",
    ]
    for item in checkpoint_data:
        label = item["safe_label"]
        fields.extend([
            f"{label}_pred_mean_aligned_eV",
            f"{label}_rmse_eV",
            f"{label}_mae_eV",
            f"{label}_bias_eV",
            f"{label}_max_abs_eV",
        ])
    if len(checkpoint_data) == 2:
        a = checkpoint_data[0]["safe_label"]
        b = checkpoint_data[1]["safe_label"]
        fields.extend([
            f"delta_rmse_{b}_minus_{a}_eV",
            f"delta_abs_bias_{b}_minus_{a}_eV",
        ])

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for j in range(n_common):
            row = {
                "band_index": j + 1,
                "region": "supervised" if j < n_supervised else "unsupervised",
                "dft_mean_aligned_eV": float(np.mean(ref[:, j])),
                "dft_min_aligned_eV": float(np.min(ref[:, j])),
                "dft_max_aligned_eV": float(np.max(ref[:, j])),
            }
            for item in checkpoint_data:
                label = item["safe_label"]
                bm = item["band_metrics"]
                row.update({
                    f"{label}_pred_mean_aligned_eV": float(bm["pred_mean"][j]),
                    f"{label}_rmse_eV": float(bm["rmse"][j]),
                    f"{label}_mae_eV": float(bm["mae"][j]),
                    f"{label}_bias_eV": float(bm["bias"][j]),
                    f"{label}_max_abs_eV": float(bm["max_abs"][j]),
                })
            if len(checkpoint_data) == 2:
                m0 = checkpoint_data[0]["band_metrics"]
                m1 = checkpoint_data[1]["band_metrics"]
                a = checkpoint_data[0]["safe_label"]
                b = checkpoint_data[1]["safe_label"]
                row[f"delta_rmse_{b}_minus_{a}_eV"] = float(
                    m1["rmse"][j] - m0["rmse"][j]
                )
                row[f"delta_abs_bias_{b}_minus_{a}_eV"] = float(
                    abs(m1["bias"][j]) - abs(m0["bias"][j])
                )
            writer.writerow(row)


def plot_rmse(
    output_dir: Path,
    checkpoint_data: List[dict],
    n_supervised: int,
    n_common: int,
    title: str,
    dpi: int,
):
    bands = np.arange(1, n_common + 1)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))

    for item in checkpoint_data:
        ax.plot(
            bands,
            item["band_metrics"]["rmse"],
            linewidth=1.5,
            label=item["label"],
        )

    if 0 < n_supervised < n_common:
        # Boundary lies between bands n_supervised and n_supervised+1.
        ax.axvline(
            n_supervised + 0.5,
            linestyle="--",
            linewidth=1.0,
            label=f"supervised cutoff ({n_supervised})",
        )

    ax.set_xlabel("Non-core band index")
    ax.set_ylabel("Band RMSE over k-path (eV)")
    ax.set_title(title)
    ax.legend()
    ax.tick_params(direction="in")
    fig.tight_layout()

    png = output_dir / "rmse_vs_band_index.png"
    pdf = output_dir / "rmse_vs_band_index.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def plot_delta_rmse(
    output_dir: Path,
    checkpoint_data: List[dict],
    n_supervised: int,
    n_common: int,
    dpi: int,
):
    if len(checkpoint_data) != 2:
        return None, None

    a, b = checkpoint_data
    bands = np.arange(1, n_common + 1)
    delta = b["band_metrics"]["rmse"] - a["band_metrics"]["rmse"]

    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    ax.plot(bands, delta, linewidth=1.5)
    ax.axhline(0.0, linestyle="--", linewidth=0.9)
    if 0 < n_supervised < n_common:
        ax.axvline(n_supervised + 0.5, linestyle="--", linewidth=1.0)

    ax.set_xlabel("Non-core band index")
    ax.set_ylabel(f"Delta RMSE: {b['label']} - {a['label']} (eV)")
    ax.set_title(
        "Change in band-resolved error\n"
        "negative = checkpoint 2 improved"
    )
    ax.tick_params(direction="in")
    fig.tight_layout()

    stem = f"delta_rmse_{b['safe_label']}_minus_{a['safe_label']}"
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def metric_line(name: str, m):
    if m is None:
        return f"{name}: not available"
    return (
        f"{name}: RMSE={m['rmse']:.6f} eV, "
        f"MAE={m['mae']:.6f} eV, bias={m['bias']:.6f} eV, "
        f"max|dE|={m['max_abs']:.6f} eV"
    )


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Evaluate DeePTB error versus non-core band index for one structure, "
            "optionally comparing two checkpoints."
        )
    )
    p.add_argument("--checkpoint", required=True, type=Path,
                   help="First checkpoint (e.g. nnsk.ep30.pth)")
    p.add_argument("--checkpoint2", type=Path, default=None,
                   help="Optional second checkpoint for epoch-to-epoch comparison")
    p.add_argument("--label", default=None,
                   help="Display label for checkpoint 1; inferred if omitted")
    p.add_argument("--label2", default=None,
                   help="Display label for checkpoint 2; inferred if omitted")
    p.add_argument("--set", dest="set_dir", required=True, type=Path,
                   help="Converted set.* directory")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--mode", choices=("full", "supervised"), default="full")
    p.add_argument("--aims-dir", type=Path, default=None,
                   help="Original FHI-aims case; otherwise read source_case from conversion_report.json")
    p.add_argument("--align", choices=("loss", "full-min", "none"), default="loss")
    p.add_argument("--supervised-count", type=int, default=None,
                   help="Override supervised band count; default = columns in set/eigenvalues.npy")
    p.add_argument("--r-max", type=float, default=5.0)
    p.add_argument("--oer-max", type=float, default=4.0)
    p.add_argument("--k-tol", type=float, default=2e-6)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    structure_path = args.set_dir / "xdat.traj"
    kpoints_path = args.set_dir / "kpoints.npy"
    supervised_path = args.set_dir / "eigenvalues.npy"

    for path in (args.checkpoint, structure_path, kpoints_path, supervised_path):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
    if args.checkpoint2 is not None and not args.checkpoint2.exists():
        raise FileNotFoundError(f"Second checkpoint not found: {args.checkpoint2}")

    atoms = read(str(structure_path))
    kpoints = np.asarray(np.load(kpoints_path), dtype=float)
    supervised_ref = load_eigenvalues_npy(supervised_path)

    if kpoints.ndim != 2 or kpoints.shape[1] != 3:
        raise ValueError(f"Expected kpoints.npy (nk,3), got {kpoints.shape}")
    if supervised_ref.shape[0] != kpoints.shape[0]:
        raise ValueError("k-point count mismatch between kpoints.npy and eigenvalues.npy")

    n_supervised = (
        int(args.supervised_count)
        if args.supervised_count is not None
        else int(supervised_ref.shape[1])
    )
    if n_supervised < 1:
        raise ValueError("supervised count must be >= 1")

    aims_dir = None
    n_core = None
    n_dft_all = None

    if args.mode == "supervised":
        ref_raw = supervised_ref.copy()
    else:
        aims_dir = args.aims_dir or source_case_from_report(args.set_dir)
        if aims_dir is None:
            raise FileNotFoundError(
                "Full mode needs --aims-dir or source_case in conversion_report.json"
            )
        ref_raw, n_core, n_dft_all = load_full_noncore_reference(
            aims_dir=aims_dir,
            atoms=atoms,
            expected_kpoints=kpoints,
            k_tol=args.k_tol,
        )

        # Verify the converted supervised target really is the prefix of the
        # full non-core spectrum used here.
        ncheck = min(supervised_ref.shape[1], ref_raw.shape[1])
        max_prefix_diff = float(
            np.max(np.abs(supervised_ref[:, :ncheck] - ref_raw[:, :ncheck]))
        )
        if max_prefix_diff > 2e-5:
            raise ValueError(
                "Converted eigenvalues.npy is not the expected prefix of raw non-core DFT: "
                f"max |dE|={max_prefix_diff:.3e} eV"
            )

    checkpoints = [args.checkpoint]
    labels = [args.label or infer_checkpoint_label(args.checkpoint)]
    if args.checkpoint2 is not None:
        checkpoints.append(args.checkpoint2)
        labels.append(args.label2 or infer_checkpoint_label(args.checkpoint2))

    pred_raw_list = []
    for checkpoint in checkpoints:
        pred = predict_checkpoint(
            checkpoint=checkpoint,
            structure_path=structure_path,
            kpoints=kpoints,
            output_dir=args.output,
            r_max=args.r_max,
            oer_max=args.oer_max,
        )
        pred_raw_list.append(pred)

    # To compare checkpoints on identical bands, use ONE common upper bound
    # across DFT and every supplied checkpoint.
    n_common = min([ref_raw.shape[1]] + [p.shape[1] for p in pred_raw_list])
    if n_common < 1:
        raise RuntimeError("No common bands available")

    ref_common_raw = ref_raw[:, :n_common].copy()

    # Reference alignment is checkpoint-independent, but each prediction gets
    # its own DeePTB shift, exactly as EigLoss would do.
    checkpoint_data = []
    aligned_ref = None
    ref_shift_record = None

    for checkpoint, label, pred_raw in zip(checkpoints, labels, pred_raw_list):
        pred_common_raw = pred_raw[:, :n_common].copy()
        pred_a, ref_a, pred_shift, ref_shift = align_spectra(
            pred_common_raw,
            ref_common_raw,
            n_supervised=n_supervised,
            method=args.align,
        )
        if aligned_ref is None:
            aligned_ref = ref_a
            ref_shift_record = ref_shift
        elif not np.allclose(aligned_ref, ref_a, atol=1e-10, rtol=0):
            raise RuntimeError("Reference alignment unexpectedly changed between checkpoints")

        bm = band_metrics(pred_a, ref_a)
        rm = region_metrics(pred_a, ref_a, n_supervised)
        item = {
            "checkpoint": checkpoint,
            "label": label,
            "safe_label": safe_label(label),
            "pred_raw": pred_raw,
            "pred_aligned": pred_a,
            "pred_shift": pred_shift,
            "band_metrics": bm,
            "region_metrics": rm,
        }
        checkpoint_data.append(item)

        np.save(args.output / f"predicted_eigenvalues_{item['safe_label']}.npy", pred_raw)
        np.save(args.output / f"aligned_predicted_eigenvalues_{item['safe_label']}.npy", pred_a)

    np.save(args.output / "reference_noncore_eigenvalues.npy", ref_raw)
    np.save(args.output / "aligned_reference_noncore_eigenvalues.npy", aligned_ref)
    np.save(args.output / "kpoints.npy", kpoints)

    csv_path = args.output / "band_errors.csv"
    write_csv(
        csv_path,
        ref=aligned_ref,
        checkpoint_data=checkpoint_data,
        n_supervised=n_supervised,
        n_common=n_common,
    )

    title = (
        f"{args.set_dir.name}: DeePTB error versus non-core band index"
        if len(checkpoint_data) == 1
        else f"{args.set_dir.name}: band-error comparison between checkpoints"
    )
    rmse_png, rmse_pdf = plot_rmse(
        output_dir=args.output,
        checkpoint_data=checkpoint_data,
        n_supervised=n_supervised,
        n_common=n_common,
        title=title,
        dpi=args.dpi,
    )
    delta_png, delta_pdf = plot_delta_rmse(
        output_dir=args.output,
        checkpoint_data=checkpoint_data,
        n_supervised=n_supervised,
        n_common=n_common,
        dpi=args.dpi,
    )

    report_path = args.output / "evaluation_report.txt"
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("Band-index-resolved DeePTB evaluation\n")
        fh.write("=" * 72 + "\n")
        fh.write(f"set_dir: {args.set_dir}\n")
        fh.write(f"structure: {structure_path}\n")
        fh.write(f"mode: {args.mode}\n")
        fh.write(f"alignment: {args.align}\n")
        fh.write(f"kpoints: {kpoints.shape[0]}\n")
        fh.write(f"supervised band count: {n_supervised}\n")
        fh.write(f"DFT bands in selected mode: {ref_raw.shape[1]}\n")
        if n_dft_all is not None:
            fh.write(f"DFT all-electron bands available: {n_dft_all}\n")
            fh.write(f"1s core bands stripped: {n_core}\n")
            fh.write(f"FHI-aims source: {aims_dir}\n")
        for item in checkpoint_data:
            fh.write(
                f"DeePTB raw bands ({item['label']}): {item['pred_raw'].shape[1]}\n"
            )
        fh.write(f"common compared bands: {n_common}\n")
        fh.write(
            f"supervised/unsupervised boundary: {n_supervised} | {n_supervised + 1}\n"
        )
        fh.write(f"common unsupervised bands: {max(0, n_common - n_supervised)}\n")
        fh.write(f"DFT alignment shift subtracted: {ref_shift_record:.10f} eV\n\n")

        for item in checkpoint_data:
            rm = item["region_metrics"]
            fh.write(f"Checkpoint: {item['label']}\n")
            fh.write(f"  path: {item['checkpoint']}\n")
            fh.write(f"  DeePTB alignment shift subtracted: {item['pred_shift']:.10f} eV\n")
            fh.write("  " + metric_line("full common", rm["full"]) + "\n")
            fh.write("  " + metric_line("supervised", rm["supervised"]) + "\n")
            fh.write("  " + metric_line("unsupervised", rm["unsupervised"]) + "\n\n")

        if len(checkpoint_data) == 2:
            a, b = checkpoint_data
            delta = b["band_metrics"]["rmse"] - a["band_metrics"]["rmse"]
            ns = min(n_supervised, n_common)
            fh.write(f"Checkpoint comparison: {b['label']} - {a['label']}\n")
            fh.write("  Delta RMSE < 0 means checkpoint 2 improved.\n")
            fh.write(f"  Mean delta RMSE, all common bands: {np.mean(delta):.10f} eV\n")
            fh.write(f"  Mean delta RMSE, supervised: {np.mean(delta[:ns]):.10f} eV\n")
            if n_common > ns:
                fh.write(
                    f"  Mean delta RMSE, unsupervised: {np.mean(delta[ns:]):.10f} eV\n"
                )
            improved = int(np.sum(delta < 0))
            worsened = int(np.sum(delta > 0))
            unchanged = int(np.sum(np.isclose(delta, 0.0, atol=1e-12)))
            fh.write(
                f"  Bands improved/worsened/unchanged: "
                f"{improved}/{worsened}/{unchanged}\n"
            )

    print("\n" + "=" * 72)
    print("Band-index-resolved DeePTB evaluation")
    print("=" * 72)
    print(f"Structure:                 {args.set_dir.name}")
    print(f"Mode:                      {args.mode}")
    print(f"k-points:                  {kpoints.shape[0]}")
    print(f"Supervised cutoff:         {n_supervised} | {n_supervised + 1}")
    if n_dft_all is not None:
        print(f"DFT all-electron bands:    {n_dft_all}")
        print(f"1s core bands stripped:    {n_core}")
    print(f"DFT non-core bands:        {ref_raw.shape[1]}")
    print(f"Common bands:              {n_common}")
    print(f"Unsupervised common bands: {max(0, n_common - n_supervised)}")

    for item in checkpoint_data:
        rm = item["region_metrics"]
        print(f"\n{item['label']}: {item['checkpoint']}")
        print(f"  full RMSE:         {rm['full']['rmse']:.6f} eV")
        print(f"  supervised RMSE:   {rm['supervised']['rmse']:.6f} eV")
        if rm["unsupervised"] is not None:
            print(f"  unsupervised RMSE: {rm['unsupervised']['rmse']:.6f} eV")

    print(f"\nSaved band CSV:   {csv_path}")
    print(f"Saved RMSE plot:  {rmse_png}")
    if delta_png is not None:
        print(f"Saved delta plot: {delta_png}")
    print(f"Saved report:     {report_path}")


if __name__ == "__main__":
    main()
