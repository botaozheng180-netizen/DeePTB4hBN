"""
Batch-evaluate one DeePTB checkpoint against converted train and/or validation
sets for the hBN-defect workflow.

Usage
-----
Evaluate both splits and generate summary plots::

    python evaluation/evaluate_model.py \
        --checkpoint RUN/checkpoint/nnsk.epN.pth \
        --train DATA/train \
        --val DATA/val \
        --output RESULTS/eval_epN \
        --make-plots

At least one of ``--train`` or ``--val`` is required.

What it evaluates
-----------------
For every ``set.*`` directory, the script evaluates the checkpoint at the exact
stored DFT k-points and compares the common prefix of ``eigenvalues.npy`` and
the DeePTB prediction.  This is the *supervised converted target*; it does not
reload higher FHI-aims bands outside the training window.

The DFT and DeePTB spectra are independently shifted by their minimum energy,
matching the energy-offset treatment used by the training loss.  Metrics are
reported for the full stored target, quartiles, individual bands, normalized
band position, and an occupation-frontier window when conversion metadata are
available.

Outputs include ``case_metrics.csv``, ``band_metrics.csv``,
``band_summary_by_split.csv``, ``normalized_band_summary.csv``,
``split_summary.csv``, ``evaluation_report.txt``, and optional diagnostic plots.
Use ``evaluate_band_error.py`` when a single structure needs full non-core
DFT coverage above the supervised cutoff or direct checkpoint-to-checkpoint
band-error comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from ase.io import read
from dptb.nn.build import build_model
from dptb.postprocess.bandstructure.band import Band


REQUIRED_FILES = ("xdat.traj", "kpoints.npy", "eigenvalues.npy")
DEFECT_RE = re.compile(
    r"cb(?P<cb>\d+)cn(?P<cn>\d+)ob(?P<ob>\d+)on(?P<on>\d+)"
    r"vb(?P<vb>\d+)vn(?P<vn>\d+)",
    re.IGNORECASE,
)


def natural_key(text: str):
    return [int(x) if x.isdigit() else x.lower()
            for x in re.split(r"(\d+)", text)]


def ensure_2d_eigenvalues(arr: np.ndarray, name: str) -> np.ndarray:
    """Normalize common eigenvalue layouts to (nk, nb)."""
    a = np.asarray(arr)
    if a.ndim == 3:
        if a.shape[0] != 1:
            raise ValueError(
                f"{name}: expected one frame in axis 0, got shape {a.shape}"
            )
        a = a[0]
    if a.ndim != 2:
        raise ValueError(f"{name}: expected (nk, nb) or (1, nk, nb), got {a.shape}")
    return np.asarray(a, dtype=float)


def safe_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            obj = json.load(fh)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def nested_get(obj: Optional[Dict[str, Any]], *keys: str):
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def selected_first_original_band(report: Optional[Dict[str, Any]]) -> Optional[int]:
    """Return selected first original FHI-aims band, 1-based, when available."""
    if not report:
        return None

    rng = nested_get(report, "selection", "selected_original_band_range_1based_inclusive")
    if isinstance(rng, (list, tuple)) and len(rng) >= 1:
        try:
            return int(rng[0])
        except Exception:
            pass

    rng = report.get("selected_original_band_range_1based_inclusive")
    if isinstance(rng, (list, tuple)) and len(rng) >= 1:
        try:
            return int(rng[0])
        except Exception:
            pass

    for key in ("selected_first_band", "selected_first_band_1based"):
        if key in report:
            try:
                return int(report[key])
            except Exception:
                pass
    return None


def infer_frontier_band_1based(
    set_dir: Path,
    n_compare: int,
) -> Tuple[Optional[int], str]:
    """
    Infer last occupied/fractionally occupied converted band, 1-based.

    Returns (band, source_string). If metadata are insufficient, (None, "...").
    """
    reports: List[Tuple[str, Dict[str, Any]]] = []
    for name in ("conversion_report.json", "info.json"):
        obj = safe_json(set_dir / name)
        if obj:
            reports.append((name, obj))

    for name, rep in reports:
        # Preferred: already expressed as non-core/converted valence count.
        val_count = nested_get(
            rep, "electronic_structure", "valence_band_count_after_inferred_core"
        )
        if val_count is not None:
            try:
                frontier = int(val_count)
                if 1 <= frontier <= n_compare:
                    return frontier, f"{name}:electronic_structure.valence_band_count_after_inferred_core"
            except Exception:
                pass

        # Convert original FHI-aims band number to the selected window.
        last_occ = nested_get(rep, "electronic_structure", "last_occupied_band_1based")
        first_sel = selected_first_original_band(rep)
        if last_occ is not None and first_sel is not None:
            try:
                frontier = int(last_occ) - int(first_sel) + 1
                if 1 <= frontier <= n_compare:
                    return frontier, f"{name}:last_occupied-selected_first+1"
            except Exception:
                pass

        # Batch-manifest-like fallback if these fields were copied into info.json.
        last_any_occ = rep.get("last_band_with_occupation")
        if last_any_occ is not None and first_sel is not None:
            try:
                frontier = int(last_any_occ) - int(first_sel) + 1
                if 1 <= frontier <= n_compare:
                    return frontier, f"{name}:last_band_with_occupation-selected_first+1"
            except Exception:
                pass

        # This is a weaker fallback: contiguous full non-core block. It misses
        # fractionally occupied frontier bands, so only use it when nothing above exists.
        n_full = rep.get("contiguous_noncore_full_count")
        if n_full is not None:
            try:
                frontier = int(n_full)
                if 1 <= frontier <= n_compare:
                    return frontier, f"{name}:contiguous_noncore_full_count"
            except Exception:
                pass

    return None, "metadata unavailable"


def extract_source_metadata(set_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "source_case": "",
        "source_relative": "",
        "case_name": "",
    }

    for name in ("conversion_report.json", "info.json"):
        rep = safe_json(set_dir / name)
        if not rep:
            continue
        for key in ("source_case", "source_relative"):
            if not out[key] and rep.get(key):
                out[key] = str(rep[key])

    source_for_name = out["source_relative"] or out["source_case"]
    if source_for_name:
        out["case_name"] = Path(source_for_name).name
    return out


def parse_defect_family(case_name: str) -> Dict[str, Any]:
    counts = {k: None for k in ("cb", "cn", "ob", "on", "vb", "vn")}
    result: Dict[str, Any] = {
        **counts,
        "defect_count": "",
        "defect_family": "",
    }

    if not case_name:
        return result

    if "pure" in case_name.lower() or "pristine" in case_name.lower():
        result.update({k: 0 for k in counts})
        result["defect_count"] = 0
        result["defect_family"] = "pristine"
        return result

    m = DEFECT_RE.search(case_name)
    if not m:
        return result

    parsed = {k: int(v) for k, v in m.groupdict().items()}
    result.update(parsed)
    result["defect_count"] = sum(parsed.values())

    pieces = []
    for key in ("cb", "cn", "ob", "on", "vb", "vn"):
        n = parsed[key]
        if n <= 0:
            continue
        label = key.upper()
        pieces.append(label if n == 1 else f"{n}{label}")
    result["defect_family"] = "+".join(pieces) if pieces else "pristine"
    return result


def metric_dict(ref: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    diff = np.asarray(pred, dtype=float) - np.asarray(ref, dtype=float)
    if diff.size == 0:
        return {
            "mse": math.nan,
            "rmse": math.nan,
            "mae": math.nan,
            "max_abs": math.nan,
            "bias": math.nan,
        }
    return {
        "mse": float(np.mean(diff ** 2)),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "mae": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "bias": float(np.mean(diff)),
    }


def band_window_metrics(
    ref: np.ndarray,
    pred: np.ndarray,
    indices: Sequence[int],
) -> Dict[str, float]:
    idx = np.asarray(list(indices), dtype=int)
    if idx.size == 0:
        return metric_dict(np.array([]), np.array([]))
    return metric_dict(ref[:, idx], pred[:, idx])


def quartile_indices(nbands: int) -> List[np.ndarray]:
    return [np.asarray(x, dtype=int) for x in np.array_split(np.arange(nbands), 4)]


def atomic_composition(structure_path: Path) -> Tuple[int, Dict[str, int]]:
    atoms = read(str(structure_path))
    counts: Dict[str, int] = defaultdict(int)
    for s in atoms.get_chemical_symbols():
        counts[s] += 1
    return len(atoms), dict(counts)


def calculate_xlist(structure_path: Path, kpoints: np.ndarray) -> np.ndarray:
    """Accumulated reciprocal-space path distance, matching band_plot logic."""
    atoms = read(str(structure_path))
    reciprocal = 2.0 * np.pi * np.asarray(atoms.cell.reciprocal())
    kcart = np.asarray(kpoints, dtype=float) @ reciprocal
    x = np.zeros(len(kcart), dtype=float)
    if len(kcart) > 1:
        x[1:] = np.cumsum(np.linalg.norm(np.diff(kcart, axis=0), axis=1))
    return x


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]):
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def finite_mean(values: Iterable[Any]) -> float:
    vals = []
    for x in values:
        try:
            f = float(x)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f):
            vals.append(f)
    return float(np.mean(vals)) if vals else math.nan


def finite_median(values: Iterable[Any]) -> float:
    vals = []
    for x in values:
        try:
            f = float(x)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f):
            vals.append(f)
    return float(np.median(vals)) if vals else math.nan


def discover_cases(roots: List[Tuple[str, Path]], case_glob: str):
    cases = []
    for split, root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"{split} root does not exist: {root}")
        dirs = [p for p in root.glob(case_glob) if p.is_dir()]
        dirs.sort(key=lambda p: natural_key(p.name))
        for p in dirs:
            cases.append((split, p))
    return cases


def prepare_output_dir(output_dir: Path, overwrite: bool):
    if output_dir.exists():
        contents = list(output_dir.iterdir())
        if contents and not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}\n"
                f"Use a new versioned directory (recommended) or pass --overwrite."
            )
        if contents and overwrite:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def evaluate_one_case(
    model,
    split: str,
    set_dir: Path,
    work_root: Path,
    r_max: float,
    oer_max: float,
    frontier_half_width: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:

    for req in REQUIRED_FILES:
        if not (set_dir / req).is_file():
            raise FileNotFoundError(f"{set_dir}: missing required file {req}")

    structure_path = set_dir / "xdat.traj"
    kpoints = np.asarray(np.load(set_dir / "kpoints.npy"), dtype=float)
    dft = ensure_2d_eigenvalues(
        np.load(set_dir / "eigenvalues.npy"), f"{set_dir}/eigenvalues.npy"
    )

    if kpoints.ndim != 2 or kpoints.shape[1] != 3:
        raise ValueError(f"{set_dir}: kpoints.npy must have shape (nk,3), got {kpoints.shape}")
    if dft.shape[0] != kpoints.shape[0]:
        raise ValueError(
            f"{set_dir}: DFT nk={dft.shape[0]} does not match kpoints nk={kpoints.shape[0]}"
        )

    xlist = calculate_xlist(structure_path, kpoints)

    # Each case gets an isolated DeePTB results directory because Band writes
    # bandstructure.npy into results_path.
    case_work = work_root / split / set_dir.name
    case_work.mkdir(parents=True, exist_ok=True)

    kpath_kwargs = {
        "kline_type": "array",
        "kpath": kpoints,
        "xlist": xlist,
    }
    atomic_data_options = {
        "r_max": float(r_max),
        "oer_max": float(oer_max),
        "pbc": True,
    }

    bcal = Band(
        model=model,
        use_gui=False,
        results_path=str(case_work),
        device=model.device,
    )
    bcal.get_bands(
        data=str(structure_path),
        kpath_kwargs=kpath_kwargs,
        AtomicData_options=atomic_data_options,
    )

    band_file = case_work / "bandstructure.npy"
    if not band_file.is_file():
        raise FileNotFoundError(
            f"DeePTB finished but {band_file} was not written."
        )

    band_obj = np.load(band_file, allow_pickle=True)
    if band_obj.shape != () or band_obj.dtype != object:
        raise ValueError(f"Unexpected DeePTB bandstructure.npy format at {band_file}")
    band_dict = band_obj.item()
    if "eigenvalues" not in band_dict:
        raise KeyError(f"{band_file} does not contain key 'eigenvalues'")

    pred_raw = ensure_2d_eigenvalues(
        np.asarray(band_dict["eigenvalues"]),
        f"{band_file}:eigenvalues",
    )

    if pred_raw.shape[0] != kpoints.shape[0]:
        raise ValueError(
            f"{set_dir}: DeePTB nk={pred_raw.shape[0]} does not match DFT nk={kpoints.shape[0]}"
        )

    n_dft = dft.shape[1]
    n_pred = pred_raw.shape[1]
    n_compare = min(n_dft, n_pred)
    if n_compare < 1:
        raise ValueError(f"{set_dir}: zero comparable bands")

    ref = np.asarray(dft[:, :n_compare], dtype=float)
    pred = np.asarray(pred_raw[:, :n_compare], dtype=float)

    # Match the DeePTB EigLoss energy-offset alignment.
    ref_aligned = ref - np.min(ref)
    pred_aligned = pred - np.min(pred)

    full = metric_dict(ref_aligned, pred_aligned)

    qidx = quartile_indices(n_compare)
    qmetrics = [band_window_metrics(ref_aligned, pred_aligned, q) for q in qidx]

    frontier_band, frontier_source = infer_frontier_band_1based(set_dir, n_compare)
    frontier_metrics = {
        "mse": math.nan,
        "rmse": math.nan,
        "mae": math.nan,
        "max_abs": math.nan,
        "bias": math.nan,
    }
    frontier_first = ""
    frontier_last = ""

    if frontier_band is not None:
        # Inclusive converted-band window [frontier-hw, frontier+hw], 1-based.
        first_1b = max(1, frontier_band - frontier_half_width)
        last_1b = min(n_compare, frontier_band + frontier_half_width)
        frontier_first = first_1b
        frontier_last = last_1b
        idx = np.arange(first_1b - 1, last_1b, dtype=int)
        frontier_metrics = band_window_metrics(ref_aligned, pred_aligned, idx)

    natoms, composition = atomic_composition(structure_path)
    source = extract_source_metadata(set_dir)
    defect = parse_defect_family(source["case_name"])

    case_row: Dict[str, Any] = {
        "split": split,
        "set_id": set_dir.name,
        "set_dir": str(set_dir),
        "source_case": source["source_case"],
        "source_relative": source["source_relative"],
        "case_name": source["case_name"],
        "natoms": natoms,
        "B": composition.get("B", 0),
        "C": composition.get("C", 0),
        "N": composition.get("N", 0),
        "O": composition.get("O", 0),
        "cb": defect["cb"],
        "cn": defect["cn"],
        "ob": defect["ob"],
        "on": defect["on"],
        "vb": defect["vb"],
        "vn": defect["vn"],
        "defect_count": defect["defect_count"],
        "defect_family": defect["defect_family"],
        "nkpoints": int(kpoints.shape[0]),
        "dft_target_bands": int(n_dft),
        "deeptb_raw_bands": int(n_pred),
        "comparable_bands": int(n_compare),
        "alignment": "independent_global_min",
        "full_mse_eV2": full["mse"],
        "full_rmse_eV": full["rmse"],
        "full_mae_eV": full["mae"],
        "full_max_abs_eV": full["max_abs"],
        "full_bias_eV": full["bias"],
        "q1_first_band": int(qidx[0][0] + 1),
        "q1_last_band": int(qidx[0][-1] + 1),
        "q1_rmse_eV": qmetrics[0]["rmse"],
        "q2_first_band": int(qidx[1][0] + 1),
        "q2_last_band": int(qidx[1][-1] + 1),
        "q2_rmse_eV": qmetrics[1]["rmse"],
        "q3_first_band": int(qidx[2][0] + 1),
        "q3_last_band": int(qidx[2][-1] + 1),
        "q3_rmse_eV": qmetrics[2]["rmse"],
        "q4_first_band": int(qidx[3][0] + 1),
        "q4_last_band": int(qidx[3][-1] + 1),
        "q4_rmse_eV": qmetrics[3]["rmse"],
        "frontier_band_1based": frontier_band if frontier_band is not None else "",
        "frontier_source": frontier_source,
        "frontier_first_band": frontier_first,
        "frontier_last_band": frontier_last,
        "frontier_rmse_eV": frontier_metrics["rmse"],
        "frontier_mae_eV": frontier_metrics["mae"],
        "frontier_max_abs_eV": frontier_metrics["max_abs"],
    }

    band_rows: List[Dict[str, Any]] = []
    diff = pred_aligned - ref_aligned
    for j in range(n_compare):
        band_diff = diff[:, j]
        band_rows.append({
            "split": split,
            "set_id": set_dir.name,
            "case_name": source["case_name"],
            "defect_family": defect["defect_family"],
            "band_index_1based": j + 1,
            "relative_band_position": (j / (n_compare - 1)) if n_compare > 1 else 0.0,
            "band_rmse_eV": float(np.sqrt(np.mean(band_diff ** 2))),
            "band_mae_eV": float(np.mean(np.abs(band_diff))),
            "band_max_abs_eV": float(np.max(np.abs(band_diff))),
            "band_bias_eV": float(np.mean(band_diff)),
            "dft_band_mean_aligned_eV": float(np.mean(ref_aligned[:, j])),
            "deeptb_band_mean_aligned_eV": float(np.mean(pred_aligned[:, j])),
            "n_compare_case": n_compare,
            "is_frontier_band": int(frontier_band == (j + 1)) if frontier_band else 0,
        })

    return case_row, band_rows


def make_split_summary(case_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for split in sorted({r["split"] for r in case_rows}):
        rows = [r for r in case_rows if r["split"] == split]
        out.append({
            "split": split,
            "n_cases": len(rows),
            "mean_full_rmse_eV": finite_mean(r["full_rmse_eV"] for r in rows),
            "median_full_rmse_eV": finite_median(r["full_rmse_eV"] for r in rows),
            "mean_full_mae_eV": finite_mean(r["full_mae_eV"] for r in rows),
            "mean_q1_rmse_eV": finite_mean(r["q1_rmse_eV"] for r in rows),
            "mean_q2_rmse_eV": finite_mean(r["q2_rmse_eV"] for r in rows),
            "mean_q3_rmse_eV": finite_mean(r["q3_rmse_eV"] for r in rows),
            "mean_q4_rmse_eV": finite_mean(r["q4_rmse_eV"] for r in rows),
            "n_frontier_available": sum(
                1 for r in rows
                if isinstance(r["frontier_rmse_eV"], (int, float))
                and np.isfinite(float(r["frontier_rmse_eV"]))
            ),
            "mean_frontier_rmse_eV": finite_mean(r["frontier_rmse_eV"] for r in rows),
            "median_frontier_rmse_eV": finite_median(r["frontier_rmse_eV"] for r in rows),
        })
    return out


def make_band_summary(band_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in band_rows:
        grouped[(row["split"], int(row["band_index_1based"]))].append(row)

    out = []
    for (split, band_idx), rows in sorted(grouped.items()):
        out.append({
            "split": split,
            "band_index_1based": band_idx,
            "n_cases_contributing": len(rows),
            "mean_band_rmse_eV": finite_mean(r["band_rmse_eV"] for r in rows),
            "median_band_rmse_eV": finite_median(r["band_rmse_eV"] for r in rows),
            "mean_band_mae_eV": finite_mean(r["band_mae_eV"] for r in rows),
            "mean_band_bias_eV": finite_mean(r["band_bias_eV"] for r in rows),
        })
    return out


def make_normalized_band_summary(
    band_rows: List[Dict[str, Any]],
    nbins: int,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in band_rows:
        pos = float(row["relative_band_position"])
        b = min(nbins - 1, int(math.floor(pos * nbins)))
        grouped[(row["split"], b)].append(row)

    out = []
    for (split, b), rows in sorted(grouped.items()):
        lo = b / nbins
        hi = (b + 1) / nbins
        out.append({
            "split": split,
            "normalized_bin": b + 1,
            "relative_position_low": lo,
            "relative_position_high": hi,
            "relative_position_mid": 0.5 * (lo + hi),
            "n_band_samples": len(rows),
            "mean_band_rmse_eV": finite_mean(r["band_rmse_eV"] for r in rows),
            "median_band_rmse_eV": finite_median(r["band_rmse_eV"] for r in rows),
            "mean_band_bias_eV": finite_mean(r["band_bias_eV"] for r in rows),
        })
    return out


def maybe_make_plots(
    output_dir: Path,
    band_summary: List[Dict[str, Any]],
    normalized_summary: List[Dict[str, Any]],
):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARNING: plots skipped because matplotlib could not be imported: {exc}")
        return

    splits = sorted({r["split"] for r in band_summary})

    # Plot 1: absolute converted band index.
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for split in splits:
        rows = [r for r in band_summary if r["split"] == split]
        x = [r["band_index_1based"] for r in rows]
        y = [r["mean_band_rmse_eV"] for r in rows]
        ax.plot(x, y, label=split)
    ax.set_xlabel("Converted band index")
    ax.set_ylabel("Mean band RMSE (eV)")
    ax.set_title("DeePTB error versus band index")
    ax.legend()
    ax.tick_params(direction="in")
    fig.tight_layout()
    fig.savefig(output_dir / "rmse_vs_band.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: normalized band position, useful because target band counts vary by case.
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for split in sorted({r["split"] for r in normalized_summary}):
        rows = [r for r in normalized_summary if r["split"] == split]
        x = [r["relative_position_mid"] for r in rows]
        y = [r["mean_band_rmse_eV"] for r in rows]
        ax.plot(x, y, marker="o", label=split)
    ax.set_xlabel("Normalized position through compared spectrum (0 = lowest, 1 = highest)")
    ax.set_ylabel("Mean band RMSE (eV)")
    ax.set_title("DeePTB error across the retained DFT spectrum")
    ax.legend()
    ax.tick_params(direction="in")
    fig.tight_layout()
    fig.savefig(output_dir / "normalized_rmse.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Batch-evaluate a DeePTB checkpoint against hBN DFT set.* directories."
    )
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="DeePTB checkpoint, e.g. nnsk.ep30.pth")
    parser.add_argument("--train", type=Path, default=None,
                        help="Training dataset root containing set.* directories.")
    parser.add_argument("--val", type=Path, default=None,
                        help="Validation dataset root containing set.* directories.")
    parser.add_argument("--output", required=True, type=Path,
                        help="Evaluation output directory.")
    parser.add_argument("--case-glob", default="set.*",
                        help="Directory glob under each dataset root (default: set.*).")
    parser.add_argument("--r-max", type=float, default=5.0)
    parser.add_argument("--oer-max", type=float, default=4.0)
    parser.add_argument("--frontier-half-width", type=int, default=10,
                        help="Evaluate +/- this many converted bands around the inferred frontier.")
    parser.add_argument("--normalized-bins", type=int, default=20,
                        help="Bins for normalized band-position aggregation.")
    parser.add_argument("--make-plots", action="store_true",
                        help="Also write RMSE-vs-band PNG diagnostics.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow replacing an existing non-empty output directory.")
    parser.add_argument("--keep-going", action="store_true", default=True,
                        help="Continue after individual case failures (default behavior).")
    args = parser.parse_args()

    if args.train is None and args.val is None:
        parser.error("Provide at least one of --train or --val.")
    if args.frontier_half_width < 0:
        parser.error("--frontier-half-width must be >= 0")
    if args.normalized_bins < 2:
        parser.error("--normalized-bins must be >= 2")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")

    prepare_output_dir(args.output, args.overwrite)

    roots: List[Tuple[str, Path]] = []
    if args.train is not None:
        roots.append(("train", args.train))
    if args.val is not None:
        roots.append(("val", args.val))

    cases = discover_cases(roots, args.case_glob)
    if not cases:
        raise RuntimeError("No set.* directories found.")

    print("=" * 72)
    print("hBN DeePTB batch evaluation")
    print("=" * 72)
    print(f"checkpoint : {args.checkpoint}")
    for split, root in roots:
        n = sum(1 for s, _ in cases if s == split)
        print(f"{split:10s}: {root} ({n} cases)")
    print(f"output     : {args.output}")
    print(f"r_max/oer  : {args.r_max} / {args.oer_max}")
    print(f"frontier   : +/- {args.frontier_half_width} bands when metadata exist")
    print("=" * 72)

    print("\nLoading model once...")
    model = build_model(checkpoint=str(args.checkpoint))

    work_root = args.output / "_deeptb_work"
    work_root.mkdir(parents=True, exist_ok=True)

    case_rows: List[Dict[str, Any]] = []
    band_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    total = len(cases)
    for i, (split, set_dir) in enumerate(cases, start=1):
        print(f"[{i:3d}/{total:3d}] {split:5s} {set_dir.name} ... ", end="", flush=True)
        try:
            case_row, one_band_rows = evaluate_one_case(
                model=model,
                split=split,
                set_dir=set_dir,
                work_root=work_root,
                r_max=args.r_max,
                oer_max=args.oer_max,
                frontier_half_width=args.frontier_half_width,
            )
            case_rows.append(case_row)
            band_rows.extend(one_band_rows)
            fr = case_row["frontier_rmse_eV"]
            fr_text = f", frontier={fr:.3f}" if isinstance(fr, float) and np.isfinite(fr) else ""
            print(f"RMSE={case_row['full_rmse_eV']:.3f} eV{fr_text}")
        except Exception as exc:
            print(f"FAILED: {exc}")
            failures.append({
                "split": split,
                "set_id": set_dir.name,
                "set_dir": str(set_dir),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })

    if not case_rows:
        raise RuntimeError(
            "All evaluations failed. See terminal output and failures.csv."
        )

    case_fields = [
        "split", "set_id", "set_dir",
        "source_case", "source_relative", "case_name",
        "natoms", "B", "C", "N", "O",
        "cb", "cn", "ob", "on", "vb", "vn",
        "defect_count", "defect_family",
        "nkpoints", "dft_target_bands", "deeptb_raw_bands", "comparable_bands",
        "alignment",
        "full_mse_eV2", "full_rmse_eV", "full_mae_eV", "full_max_abs_eV", "full_bias_eV",
        "q1_first_band", "q1_last_band", "q1_rmse_eV",
        "q2_first_band", "q2_last_band", "q2_rmse_eV",
        "q3_first_band", "q3_last_band", "q3_rmse_eV",
        "q4_first_band", "q4_last_band", "q4_rmse_eV",
        "frontier_band_1based", "frontier_source",
        "frontier_first_band", "frontier_last_band",
        "frontier_rmse_eV", "frontier_mae_eV", "frontier_max_abs_eV",
    ]
    write_csv(args.output / "case_metrics.csv", case_rows, case_fields)

    band_fields = [
        "split", "set_id", "case_name", "defect_family",
        "band_index_1based", "relative_band_position",
        "band_rmse_eV", "band_mae_eV", "band_max_abs_eV", "band_bias_eV",
        "dft_band_mean_aligned_eV", "deeptb_band_mean_aligned_eV",
        "n_compare_case", "is_frontier_band",
    ]
    write_csv(args.output / "band_metrics.csv", band_rows, band_fields)

    split_summary = make_split_summary(case_rows)
    split_fields = [
        "split", "n_cases",
        "mean_full_rmse_eV", "median_full_rmse_eV", "mean_full_mae_eV",
        "mean_q1_rmse_eV", "mean_q2_rmse_eV",
        "mean_q3_rmse_eV", "mean_q4_rmse_eV",
        "n_frontier_available",
        "mean_frontier_rmse_eV", "median_frontier_rmse_eV",
    ]
    write_csv(args.output / "split_summary.csv", split_summary, split_fields)

    band_summary = make_band_summary(band_rows)
    band_summary_fields = [
        "split", "band_index_1based", "n_cases_contributing",
        "mean_band_rmse_eV", "median_band_rmse_eV",
        "mean_band_mae_eV", "mean_band_bias_eV",
    ]
    write_csv(
        args.output / "band_summary_by_split.csv",
        band_summary,
        band_summary_fields,
    )

    normalized_summary = make_normalized_band_summary(
        band_rows, args.normalized_bins
    )
    normalized_fields = [
        "split", "normalized_bin",
        "relative_position_low", "relative_position_high", "relative_position_mid",
        "n_band_samples",
        "mean_band_rmse_eV", "median_band_rmse_eV", "mean_band_bias_eV",
    ]
    write_csv(
        args.output / "normalized_band_summary.csv",
        normalized_summary,
        normalized_fields,
    )

    if failures:
        failure_fields = [
            "split", "set_id", "set_dir", "error_type", "error", "traceback"
        ]
        write_csv(args.output / "failures.csv", failures, failure_fields)

    if args.make_plots:
        maybe_make_plots(args.output, band_summary, normalized_summary)

    # Human-readable report
    report_path = args.output / "evaluation_report.txt"
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("hBN DeePTB batch evaluation\n")
        fh.write("=" * 72 + "\n")
        fh.write(f"checkpoint: {args.checkpoint}\n")
        for split, root in roots:
            fh.write(f"{split}_root: {root}\n")
        fh.write(f"r_max: {args.r_max}\n")
        fh.write(f"oer_max: {args.oer_max}\n")
        fh.write("alignment: independent minimum of each comparable DFT/DeePTB window\n")
        fh.write(f"frontier_half_width: {args.frontier_half_width}\n")
        fh.write(f"successful_cases: {len(case_rows)} / {len(cases)}\n")
        fh.write(f"failed_cases: {len(failures)}\n\n")

        fh.write("Split summary\n")
        fh.write("-" * 72 + "\n")
        for row in split_summary:
            fh.write(
                f"{row['split']}: n={row['n_cases']}, "
                f"mean full RMSE={row['mean_full_rmse_eV']:.6f} eV, "
                f"median full RMSE={row['median_full_rmse_eV']:.6f} eV, "
                f"Q1/Q2/Q3/Q4 mean RMSE="
                f"{row['mean_q1_rmse_eV']:.6f}/"
                f"{row['mean_q2_rmse_eV']:.6f}/"
                f"{row['mean_q3_rmse_eV']:.6f}/"
                f"{row['mean_q4_rmse_eV']:.6f} eV"
            )
            if row["n_frontier_available"]:
                fh.write(
                    f", frontier n={row['n_frontier_available']}, "
                    f"mean frontier RMSE={row['mean_frontier_rmse_eV']:.6f} eV"
                )
            fh.write("\n")

        fh.write("\nBest/worst cases by full-window RMSE\n")
        fh.write("-" * 72 + "\n")
        sorted_cases = sorted(case_rows, key=lambda r: float(r["full_rmse_eV"]))
        nshow = min(5, len(sorted_cases))
        fh.write("Best:\n")
        for row in sorted_cases[:nshow]:
            fh.write(
                f"  {row['split']:5s} {row['set_id']:12s} "
                f"RMSE={row['full_rmse_eV']:.6f} eV "
                f"{row['defect_family'] or row['case_name']}\n"
            )
        fh.write("Worst:\n")
        for row in reversed(sorted_cases[-nshow:]):
            fh.write(
                f"  {row['split']:5s} {row['set_id']:12s} "
                f"RMSE={row['full_rmse_eV']:.6f} eV "
                f"{row['defect_family'] or row['case_name']}\n"
            )

        fh.write("\nInterpretation aid\n")
        fh.write("-" * 72 + "\n")
        fh.write(
            "Compare Q1 -> Q4 and normalized_band_summary.csv. "
            "A systematic rise toward Q4 / normalized position ~1 indicates "
            "that prediction error grows toward the high-energy end of the retained spectrum.\n"
        )
        fh.write(
            "Frontier metrics are reported only when conversion metadata support "
            "an occupied-frontier inference; missing values are intentionally not guessed.\n"
        )

    print("\n" + "=" * 72)
    print("Evaluation complete")
    print("=" * 72)
    print(f"Successful cases : {len(case_rows)} / {len(cases)}")
    print(f"Case metrics     : {args.output / 'case_metrics.csv'}")
    print(f"Band metrics     : {args.output / 'band_metrics.csv'}")
    print(f"Split summary    : {args.output / 'split_summary.csv'}")
    print(f"Band summary     : {args.output / 'band_summary_by_split.csv'}")
    print(f"Normalized bands: {args.output / 'normalized_band_summary.csv'}")
    print(f"Report           : {report_path}")
    if failures:
        print(f"Failures         : {args.output / 'failures.csv'}")
    if args.make_plots:
        print(f"Plots            : {args.output / 'rmse_vs_band.png'}")
        print(f"                   {args.output / 'normalized_rmse.png'}")


if __name__ == "__main__":
    main()
