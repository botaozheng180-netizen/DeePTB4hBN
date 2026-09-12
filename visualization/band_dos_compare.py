"""
Plot FHI-aims and DeePTB band structures together with total and
species-projected DOS for one converted hBN structure.

Usage
-----
FHI-aims ground truth only::

    python visualization/band_dos_compare.py \
        --set DATA/set.000000 \
        --output RESULTS/band_dos_dft \
        --plot-content ground-truth \
        --mode full \
        --dos-mode all

DeePTB prediction only::

    python visualization/band_dos_compare.py \
        --checkpoint RUN/checkpoint/nnsk.epN.pth \
        --set DATA/set.000000 \
        --output RESULTS/band_dos_pred \
        --plot-content prediction \
        --mode full \
        --dos-mode all \
        --kmesh 30 30 1 \
        --sigma 0.10

FHI-aims vs DeePTB comparison::

    python visualization/band_dos_compare.py \
        --checkpoint RUN/checkpoint/nnsk.epN.pth \
        --set DATA/set.000000 \
        --output RESULTS/band_dos_compare \
        --plot-content comparison \
        --mode full \
        --dos-mode all \
        --kmesh 30 30 1 \
        --sigma 0.10

``--dos-mode total`` plots only total DOS. ``--dos-mode all`` adds one
projected-DOS panel for every species present in the structure.

Modes
-----
``supervised`` uses the converted ``eigenvalues.npy`` target.
``full`` reloads ``band1*.out``, strips inferred B/C/N/O 1s core bands, and
uses all available non-core FHI-aims bands.

In comparison mode, FHI-aims stays in its native band/DOS energy gauge.
With ``--align loss``, DeePTB is rigidly shifted so the minima of the common
FHI-aims and DeePTB band windows coincide; the same shift is applied to its
DOS.

The script prefers ``KS_DOS_total.dat`` and ``<species>_l_proj_dos.dat``.
If only raw DOS files are available, their energies are converted with
``E_mu = E_raw - mu``. A warning is printed when the FHI-aims DOS energy
window does not cover the full plotted FHI-aims band range.
"""


from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
from ase.data import atomic_numbers
from ase.io import read

from dptb.data import AtomicDataDict
from dptb.nn.build import build_model
from dptb.postprocess.bandstructure.band import Band
from dptb.postprocess.elec_struc_cal import ElecStruCal


CORE_BANDS_PER_ATOM: Dict[str, int] = {
    "H": 0,
    "B": 1,
    "C": 1,
    "N": 1,
    "O": 1,
}

MU_RE = re.compile(
    r"chemical\s+potential\s*,?\s*mu\s*=\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?)",
    re.IGNORECASE,
)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _f(x: str) -> float:
    return float(x.replace("D", "E").replace("d", "e"))


def natural_key(path: str | Path):
    s = Path(path).name
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def load_eigenvalues_npy(path: Path) -> np.ndarray:
    eig = np.load(path)
    if eig.ndim == 3:
        if eig.shape[0] != 1:
            raise ValueError(f"Expected one frame in {path}, got {eig.shape}")
        eig = eig[0]
    elif eig.ndim != 2:
        raise ValueError(f"Expected (1,nk,nb) or (nk,nb), got {eig.shape}")
    return np.asarray(eig, dtype=float)


# -----------------------------------------------------------------------------
# FHI-aims band parsing (kept consistent with band_plot)
# -----------------------------------------------------------------------------
def parse_numeric_band_file(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    with path.open("r", errors="replace") as fh:
        for line in fh:
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
        raise ValueError(f"{path}: inconsistent band counts: {sorted(nbands_set)}")
    nbands = nbands_set.pop()

    kpoints, occs, eigs = [], [], []
    for vals in rows:
        kpoints.append(vals[1:4])
        pairs = np.asarray(vals[4:], dtype=float).reshape(nbands, 2)
        occs.append(pairs[:, 0])
        eigs.append(pairs[:, 1])

    return (
        np.asarray(kpoints, dtype=float),
        np.asarray(occs, dtype=float),
        np.asarray(eigs, dtype=float),
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
            "Cannot infer 1s core stripping for: " + ", ".join(unsupported)
        )
    return int(sum(CORE_BANDS_PER_ATOM[s] for s in symbols))


def source_case_from_report(set_dir: Path) -> Optional[Path]:
    report = set_dir / "conversion_report.json"
    if not report.exists():
        return None
    data = json.loads(report.read_text(encoding="utf-8"))
    source = data.get("source_case")
    return Path(source) if source else None


def load_full_noncore_aims_reference(
    aims_dir: Path,
    atoms,
    expected_kpoints: np.ndarray,
    k_tol: float,
):
    band1 = sorted(aims_dir.glob("band1*.out"), key=natural_key)
    band2 = sorted(aims_dir.glob("band2*.out"), key=natural_key)
    if not band1:
        raise FileNotFoundError(f"No band1*.out in {aims_dir}")
    if band2:
        raise RuntimeError(
            f"Found {len(band2)} band2*.out files.  This script expects the "
            "current single-channel/unpolarized DeePTB workflow."
        )

    kp, _occ, eig_all = combine_aims_segments(band1)
    if kp.shape != expected_kpoints.shape:
        raise ValueError(
            f"Raw FHI-aims k-point shape {kp.shape} != converted {expected_kpoints.shape}"
        )
    max_kdiff = float(np.max(np.abs(kp - expected_kpoints)))
    if max_kdiff > k_tol:
        raise ValueError(
            f"Raw/converted kpoints differ: max |dk|={max_kdiff:.3e} > {k_tol:.3e}"
        )

    n_core = infer_core_band_count(list(atoms.get_chemical_symbols()))
    n_all = int(eig_all.shape[1])
    if n_core >= n_all:
        raise ValueError(f"Inferred {n_core} core bands but only {n_all} DFT bands exist")
    return np.asarray(eig_all[:, n_core:], dtype=float), n_core, n_all


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
        dmin = float(dist[idx])
        if dmin > tolerance:
            raise ValueError(
                f"Could not identify {label}; nearest k={kpoints[idx]}, distance={dmin:.3e}"
            )
        found.append({
            "label": label,
            "index": idx,
            "distance": dmin,
            "kpoint": kpoints[idx].copy(),
            "x": float(xlist[idx]),
        })
    found.sort(key=lambda x: x["index"])
    if len({x["index"] for x in found}) != len(found):
        raise ValueError("High-symmetry detection produced duplicate indices")
    return found


# -----------------------------------------------------------------------------
# FHI-aims DOS parsing
# -----------------------------------------------------------------------------
def _read_header(path: Path, max_lines: int = 40) -> str:
    out = []
    with path.open("r", errors="replace") as fh:
        for _ in range(max_lines):
            line = fh.readline()
            if not line:
                break
            if line.lstrip().startswith("#"):
                out.append(line.rstrip())
            elif line.strip():
                break
    return "\n".join(out)



def load_aims_dos_file(path: Path, raw_mu: Optional[float] = None) -> dict:
    """
    Load one FHI-aims DOS/PDOS file and express its energy column in the SAME
    reference used by the ordinary (non-raw) FHI-aims DOS output.

    Energy-reference handling
    -----------------------
    FHI-aims *_raw.dat files use the raw/vacuum (or periodic internal) energy
    reference, while the ordinary KS_DOS_total.dat and *_l_proj_dos.dat files
    are shifted by the chemical potential:

        E_mu = E_raw - mu

    The band1*.out energies used in this hBN workflow are compared against the
    ordinary FHI-aims DOS convention.  Therefore the script prefers the non-raw files.
    A raw file is accepted only when mu is known (from --aims-mu or another
    non-raw DOS header), in which case it is converted with E_raw - mu.
    """
    header = _read_header(path)
    arr = np.loadtxt(path, comments="#")
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] < 2:
        raise ValueError(f"DOS file needs >=2 columns: {path}")

    energy_file = np.asarray(arr[:, 0], dtype=float)
    values = np.asarray(arr[:, 1:], dtype=float)
    low = header.lower()

    is_raw = ("vacuum level" in low) or path.name.lower().endswith("_raw.dat")
    mu = None

    if is_raw:
        if raw_mu is None:
            raise ValueError(
                f"{path} is a raw FHI-aims DOS file, but its chemical-potential "
                "offset is unknown. Prefer the corresponding non-raw DOS file or "
                "supply --aims-mu."
            )
        mu = float(raw_mu)
        energy_band_ref = energy_file - mu
        reference = "raw-converted-to-chemical-potential"
    else:
        m = MU_RE.search(header)
        if m:
            mu = _f(m.group(1))
            reference = "chemical-potential"
        else:
            reference = "nonraw-assumed-band-reference"
            warnings.warn(
                f"Could not parse chemical potential from {path}; using its "
                "energy column directly as the FHI band-reference gauge."
            )
        # The ordinary FHI-aims DOS file is already in the desired gauge.
        energy_band_ref = energy_file.copy()

    return {
        "path": path,
        "header": header,
        "energy_file": energy_file,
        "energy_band_ref": energy_band_ref,
        "values": values,
        "mu": mu,
        "reference": reference,
        "is_raw": is_raw,
    }


def _case_insensitive_file(directory: Path, wanted: str) -> Optional[Path]:
    exact = directory / wanted
    if exact.exists():
        return exact
    wanted_low = wanted.lower()
    for p in directory.iterdir():
        if p.is_file() and p.name.lower() == wanted_low:
            return p
    return None


def _mu_from_nonraw_dos_header(path: Path) -> Optional[float]:
    if path is None or not path.exists():
        return None
    m = MU_RE.search(_read_header(path))
    return _f(m.group(1)) if m else None


def infer_aims_mu_hint(aims_dir: Path, species: List[str]) -> Optional[float]:
    """
    Try to obtain mu from any ordinary (non-raw) DOS header.  This is used only
    as a fallback for converting *_raw.dat files.
    """
    candidates = [aims_dir / "KS_DOS_total.dat"]
    candidates.extend(aims_dir / f"{s}_l_proj_dos.dat" for s in species)
    for p in candidates:
        mu = _mu_from_nonraw_dos_header(p)
        if mu is not None:
            return mu
    return None


def locate_total_dos(aims_dir: Path) -> Path:
    # Prefer the chemical-potential-referenced file because it is already
    # in the same practical plotting gauge as the FHI-aims band output.
    for name in ("KS_DOS_total.dat", "KS_DOS_total_raw.dat"):
        p = _case_insensitive_file(aims_dir, name)
        if p is not None:
            return p
    raise FileNotFoundError(
        f"Could not find KS_DOS_total.dat or KS_DOS_total_raw.dat in {aims_dir}"
    )


def locate_species_dos(aims_dir: Path, symbol: str) -> Optional[Path]:
    # Same preference as total DOS.
    for name in (
        f"{symbol}_l_proj_dos.dat",
        f"{symbol}_l_proj_dos_raw.dat",
    ):
        p = _case_insensitive_file(aims_dir, name)
        if p is not None:
            return p
    return None


def load_fhi_dos_bundle(
    aims_dir: Path,
    species: List[str],
    dos_mode: str,
    aims_mu: Optional[float] = None,
):
    mu_hint = aims_mu
    if mu_hint is None:
        mu_hint = infer_aims_mu_hint(aims_dir, species)

    total_path = locate_total_dos(aims_dir)
    total = load_aims_dos_file(total_path, raw_mu=mu_hint)

    # If the total non-raw file itself supplied mu, reuse that for any raw PDOS.
    if total.get("mu") is not None:
        mu_hint = total["mu"]

    projected = {}
    missing = []
    if dos_mode == "all":
        for sym in species:
            p = locate_species_dos(aims_dir, sym)
            if p is None:
                missing.append(sym)
                continue
            datum = load_aims_dos_file(p, raw_mu=mu_hint)
            projected[sym] = datum
    return total, projected, missing


# -----------------------------------------------------------------------------
# DeePTB DOS / species PDOS
# -----------------------------------------------------------------------------
def uniform_gamma_kmesh(kmesh: Tuple[int, int, int]) -> np.ndarray:
    """Uniform Gamma-centered fractional reciprocal coordinates in [-0.5,0.5)."""
    axes = []
    for n in kmesh:
        if n < 1:
            raise ValueError("All --kmesh entries must be >=1")
        if n == 1:
            axes.append(np.array([0.0], dtype=float))
        else:
            # Exactly uniform periodic grid; includes Gamma.
            x = np.arange(n, dtype=float) / float(n)
            x[x >= 0.5] -= 1.0
            axes.append(x)
    grid = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([g.reshape(-1) for g in grid])


def model_overlap_enabled(model) -> bool:
    value = getattr(model, "overlap", False)
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().item())
    return bool(value)


def species_orbital_indices(model, atoms, norb_total: int) -> Dict[str, np.ndarray]:
    """Map each chemical species to global TB-basis indices using atom order."""
    if not hasattr(model, "idp"):
        raise AttributeError("Loaded model has no idp/OrbitalMapper")
    idp = model.idp
    if not hasattr(idp, "atom_norb") or not hasattr(idp, "chemical_symbol_to_type"):
        raise AttributeError("Model OrbitalMapper lacks atom_norb/type mapping")

    by_species: Dict[str, List[int]] = {}
    offset = 0
    for sym in atoms.get_chemical_symbols():
        if sym not in idp.chemical_symbol_to_type:
            raise ValueError(f"Species {sym} is absent from the model basis mapping")
        itype = idp.chemical_symbol_to_type[sym]
        n = idp.atom_norb[itype]
        if isinstance(n, torch.Tensor):
            n = int(n.detach().cpu().item())
        else:
            n = int(n)
        by_species.setdefault(sym, []).extend(range(offset, offset + n))
        offset += n

    if offset != norb_total:
        raise ValueError(
            f"OrbitalMapper/structure imply {offset} orbitals, but DeePTB eigenvectors have {norb_total}."
        )
    return {k: np.asarray(v, dtype=int) for k, v in by_species.items()}


def gaussian_add(
    energy_grid: np.ndarray,
    energies: np.ndarray,
    weights: Optional[np.ndarray],
    sigma: float,
    state_block: int,
) -> np.ndarray:
    """Accumulate Gaussian-broadened states without forming one huge matrix."""
    e = np.asarray(energies, dtype=float).reshape(-1)
    if weights is None:
        w = None
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.shape != e.shape:
            raise ValueError("Weights and energies have different flattened shapes")

    out = np.zeros_like(energy_grid, dtype=float)
    pref = 1.0 / (math.sqrt(2.0 * math.pi) * sigma)
    for i0 in range(0, e.size, state_block):
        i1 = min(i0 + state_block, e.size)
        ee = e[i0:i1]
        delta = (energy_grid[:, None] - ee[None, :]) / sigma
        g = np.exp(-0.5 * delta * delta) * pref
        if w is None:
            out += np.sum(g, axis=1)
        else:
            out += g @ w[i0:i1]
    return out


def compute_deeptb_dos(
    model,
    structure_path: Path,
    atoms,
    kmesh: Tuple[int, int, int],
    energy_grid_raw: np.ndarray,
    dos_mode: str,
    sigma: float,
    spin_degeneracy: float,
    k_chunk: int,
    state_block: int,
    r_max: float,
    oer_max: float,
):
    if sigma <= 0:
        raise ValueError("--sigma must be > 0")
    if k_chunk < 1 or state_block < 1:
        raise ValueError("--k-chunk and --state-block must be >=1")

    kpoints = uniform_gamma_kmesh(kmesh)
    nk_total = len(kpoints)
    want_pdos = dos_mode == "all"

    if want_pdos and model_overlap_enabled(model):
        raise NotImplementedError(
            "Species PDOS currently supports overlap=false only.  Use --dos-mode total "
            "for overlap models."
        )

    eig_method = "eigh" if want_pdos else "eigvalsh"
    cal = ElecStruCal(model=model, device=model.device, eig_method=eig_method)
    atomic_options = {"r_max": r_max, "oer_max": oer_max, "pbc": True}

    total_acc = np.zeros_like(energy_grid_raw, dtype=float)
    species_acc: Dict[str, np.ndarray] = {}
    orbital_indices = None
    norb = None

    all_eval_min = np.inf
    all_eval_max = -np.inf

    print(
        f"Calculating DeePTB DOS on {kmesh[0]}x{kmesh[1]}x{kmesh[2]} "
        f"mesh ({nk_total} k-points), mode={dos_mode}"
    )

    for i0 in range(0, nk_total, k_chunk):
        i1 = min(i0 + k_chunk, nk_total)
        kpart = kpoints[i0:i1]
        print(f"  DOS k-points {i0 + 1:5d}-{i1:5d}/{nk_total}", flush=True)

        with torch.no_grad():
            data, eigs = cal.get_eigs(
                data=str(structure_path),
                klist=kpart,
                AtomicData_options=atomic_options,
            )

        eigs = np.asarray(eigs, dtype=float)
        if eigs.ndim != 2:
            raise ValueError(f"Unexpected DOS eigenvalue shape {eigs.shape}")
        all_eval_min = min(all_eval_min, float(np.min(eigs)))
        all_eval_max = max(all_eval_max, float(np.max(eigs)))

        if norb is None:
            norb = eigs.shape[1]

        total_acc += gaussian_add(
            energy_grid_raw, eigs, None, sigma=sigma, state_block=state_block
        )

        if want_pdos:
            vec = data[AtomicDataDict.EIGENVECTOR_KEY]
            vec = to_numpy(vec)
            # Eigh / torch.linalg.eigh convention: [Nk, basis, band], columns
            # are eigenvectors.
            if vec.ndim != 3:
                raise ValueError(f"Unexpected DeePTB eigenvector shape {vec.shape}")
            if vec.shape[0] != eigs.shape[0] or vec.shape[2] != eigs.shape[1]:
                raise ValueError(
                    f"Eigenvector/eigenvalue shape mismatch: {vec.shape} vs {eigs.shape}"
                )

            if orbital_indices is None:
                orbital_indices = species_orbital_indices(model, atoms, vec.shape[1])
                species_acc = {
                    sym: np.zeros_like(energy_grid_raw, dtype=float)
                    for sym in orbital_indices
                }

            for sym, indices in orbital_indices.items():
                weights = np.sum(np.abs(vec[:, indices, :]) ** 2, axis=1)
                species_acc[sym] += gaussian_add(
                    energy_grid_raw,
                    eigs,
                    weights,
                    sigma=sigma,
                    state_block=state_block,
                )

    factor = float(spin_degeneracy) / float(nk_total)
    total_acc *= factor
    for sym in species_acc:
        species_acc[sym] *= factor

    # For an orthogonal complete species partition, sum species PDOS == total.
    pdos_sum_error = None
    if species_acc:
        summed = np.sum(np.stack(list(species_acc.values()), axis=0), axis=0)
        denom = max(float(np.max(np.abs(total_acc))), 1e-12)
        pdos_sum_error = float(np.max(np.abs(summed - total_acc)) / denom)

    return {
        "kpoints": kpoints,
        "energy_raw": energy_grid_raw,
        "total": total_acc,
        "species": species_acc,
        "norb": int(norb) if norb is not None else None,
        "eig_min_raw": float(all_eval_min),
        "eig_max_raw": float(all_eval_max),
        "pdos_sum_relative_max_error": pdos_sum_error,
    }


# -----------------------------------------------------------------------------
# Normalization and metrics/helpers
# -----------------------------------------------------------------------------
def comparison_metrics(pred: np.ndarray, ref: np.ndarray):
    residual = pred - ref
    mse = float(np.mean(residual ** 2))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(residual))),
        "max_abs": float(np.max(np.abs(residual))),
    }


def scaling_factor(energy, total, emin, emax, method: str) -> float:
    if method == "none":
        return 1.0
    mask = (energy >= emin) & (energy <= emax)
    if np.count_nonzero(mask) < 2:
        raise ValueError("Too few DOS points inside plotted energy range for normalization")
    if method == "total-area":
        area = float(np.trapezoid(total[mask], energy[mask]))
        if abs(area) < 1e-14:
            raise ValueError("Total DOS area is zero in plotted window")
        return 1.0 / area
    if method == "total-max":
        peak = float(np.max(np.abs(total[mask])))
        if peak < 1e-14:
            raise ValueError("Total DOS maximum is zero in plotted window")
        return 1.0 / peak
    raise ValueError(method)


def species_order(atoms) -> List[str]:
    symbols = set(atoms.get_chemical_symbols())
    return sorted(symbols, key=lambda s: atomic_numbers.get(s, 999))


def choose_auto_energy_range(ref_band, pred_band):
    lo = float(min(np.min(ref_band), np.min(pred_band)))
    hi = float(max(np.max(ref_band), np.max(pred_band)))
    span = max(hi - lo, 1.0)
    pad = 0.03 * span
    return lo - pad, hi + pad


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------
def assess_fhi_dos_coverage(
    fhi_band_values: np.ndarray,
    fhi_dos_energy: np.ndarray,
    tolerance: float = 1e-6,
) -> dict:
    """Compare plotted FHI band extent with the available FHI DOS source range."""
    band_min = float(np.min(fhi_band_values))
    band_max = float(np.max(fhi_band_values))
    dos_min = float(np.min(fhi_dos_energy))
    dos_max = float(np.max(fhi_dos_energy))

    missing_low = max(0.0, dos_min - band_min)
    missing_high = max(0.0, band_max - dos_max)

    return {
        "band_min": band_min,
        "band_max": band_max,
        "dos_min": dos_min,
        "dos_max": dos_max,
        "missing_low": missing_low,
        "missing_high": missing_high,
        "incomplete_low": missing_low > tolerance,
        "incomplete_high": missing_high > tolerance,
        "incomplete": (missing_low > tolerance) or (missing_high > tolerance),
    }


def print_fhi_dos_coverage_warning(info: dict) -> None:
    if not info["incomplete"]:
        return

    print("\n" + "!" * 78)
    print("WARNING: FHI-aims DOS energy window does not cover the plotted band spectrum")
    print("!" * 78)
    print(
        f"FHI-aims band range: {info['band_min']:.6f} ... "
        f"{info['band_max']:.6f} eV"
    )
    print(
        f"FHI-aims DOS range:  {info['dos_min']:.6f} ... "
        f"{info['dos_max']:.6f} eV"
    )

    if info["incomplete_low"]:
        print(
            f"The FHI-aims DOS does not cover the lowest "
            f"{info['missing_low']:.6f} eV of the plotted band spectrum."
        )
        print(
            f"Missing DOS below {info['dos_min']:.6f} eV reflects the original "
            "FHI-aims DOS calculation window, not absence of electronic states."
        )

    if info["incomplete_high"]:
        print(
            f"The FHI-aims DOS does not cover the highest "
            f"{info['missing_high']:.6f} eV of the plotted band spectrum."
        )
        print(
            f"Missing DOS above {info['dos_max']:.6f} eV reflects the original "
            "FHI-aims DOS calculation window, not absence of electronic states."
        )
    print("!" * 78 + "\n")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Adaptive FHI-aims / DeePTB band + DOS plot with FHI DOS "
            "energy registration and selectable ground-truth/prediction/comparison content."
        )
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="DeePTB checkpoint. Required for prediction/comparison; not needed for ground-truth.",
    )
    p.add_argument("--set", dest="set_dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--plot-content",
        choices=("comparison", "prediction", "ground-truth"),
        default="comparison",
        help=(
            "comparison: FHI-aims + DeePTB; prediction: DeePTB only; "
            "ground-truth: FHI-aims only."
        ),
    )
    p.add_argument(
        "--aims-dir",
        type=Path,
        default=None,
        help="Original FHI-aims folder; otherwise source_case from conversion_report.json.",
    )
    p.add_argument(
        "--aims-mu",
        type=float,
        default=None,
        help=(
            "Optional FHI-aims chemical potential in eV, used only if a raw "
            "*_DOS*_raw.dat file must be converted because the non-raw file is absent."
        ),
    )
    p.add_argument(
        "--mode",
        choices=("supervised", "full"),
        default="full",
        help="Band reference scope. Default: full non-core FHI-aims spectrum.",
    )
    p.add_argument(
        "--dos-mode",
        choices=("total", "all"),
        default="all",
        help="total = total DOS only; all = total + every available species PDOS.",
    )
    p.add_argument(
        "--kmesh", nargs=3, type=int, default=(30, 30, 1), metavar=("NX", "NY", "NZ")
    )
    p.add_argument("--sigma", type=float, default=0.10,
                   help="DeePTB Gaussian DOS broadening in eV.")
    p.add_argument("--n-energy", type=int, default=1800)
    p.add_argument(
        "--spin-degeneracy",
        type=float,
        default=2.0,
        help="State-count factor for spin-unpolarized DeePTB DOS (default 2).",
    )
    p.add_argument(
        "--dos-normalization",
        choices=("none", "total-area", "total-max"),
        default="total-area",
    )
    p.add_argument("--k-chunk", type=int, default=16)
    p.add_argument("--state-block", type=int, default=2048)
    p.add_argument("--r-max", type=float, default=5.0)
    p.add_argument("--oer-max", type=float, default=4.0)
    p.add_argument("--k-tol", type=float, default=2e-6)
    p.add_argument(
        "--align",
        choices=("loss", "none"),
        default="loss",
        help=(
            "loss: keep FHI-aims in its native band/DOS gauge and rigidly shift "
            "DeePTB so its common-window minimum matches the FHI minimum. This is "
            "mathematically equivalent to the EigLoss minimum alignment but keeps "
            "the y-axis in the FHI-aims plotting gauge. none: no DeePTB shift."
        ),
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
    args = parse_args()
    if args.n_energy < 100:
        raise ValueError("--n-energy should be at least 100")
    if args.ref_stride < 1:
        raise ValueError("--ref-stride must be >=1")
    if args.spin_degeneracy <= 0:
        raise ValueError("--spin-degeneracy must be >0")
    if args.plot_content in ("comparison", "prediction") and args.checkpoint is None:
        raise ValueError("--checkpoint is required for prediction/comparison")

    set_dir = args.set_dir
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    structure_path = set_dir / "xdat.traj"
    kpoints_path = set_dir / "kpoints.npy"
    supervised_path = set_dir / "eigenvalues.npy"
    for pth in (structure_path, kpoints_path, supervised_path):
        if not pth.exists():
            raise FileNotFoundError(pth)
    if args.checkpoint is not None and not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    atoms = read(str(structure_path))
    kpoints_band = np.asarray(np.load(kpoints_path), dtype=float)
    ref_supervised_raw = load_eigenvalues_npy(supervised_path)
    n_supervised = ref_supervised_raw.shape[1]

    aims_dir = args.aims_dir or source_case_from_report(set_dir)
    if aims_dir is None:
        raise FileNotFoundError(
            "Need --aims-dir or conversion_report.json -> source_case."
        )
    aims_dir = aims_dir.resolve()
    if not aims_dir.is_dir():
        raise FileNotFoundError(aims_dir)

    # ------------------------------------------------------------------
    # FHI-aims band reference
    # ------------------------------------------------------------------
    n_core = None
    n_dft_all = None
    if args.mode == "supervised":
        ref_band_raw = ref_supervised_raw.copy()
    else:
        ref_band_raw, n_core, n_dft_all = load_full_noncore_aims_reference(
            aims_dir, atoms, kpoints_band, args.k_tol
        )
        ncheck = min(n_supervised, ref_band_raw.shape[1])
        if ncheck:
            d = float(
                np.max(np.abs(ref_supervised_raw[:, :ncheck] - ref_band_raw[:, :ncheck]))
            )
            if d > 2e-5:
                raise ValueError(
                    "Converted eigenvalues.npy is not the expected prefix of the "
                    f"raw non-core FHI bands; max |dE|={d:.3e} eV"
                )

    # Keep FHI-aims in its native band-reference gauge.
    ref_band_plot = ref_band_raw.copy()

    xlist = cumulative_k_distance(kpoints_band, atoms)
    hs = find_hbn_high_symmetry_points(kpoints_band, xlist)
    hs_x = np.asarray([x["x"] for x in hs])
    hs_labels = [x["label"] for x in hs]

    # ------------------------------------------------------------------
    # DeePTB band prediction, only when requested
    # ------------------------------------------------------------------
    model = None
    pred_band_raw = None
    pred_band_plot = None
    pred_to_ref_shift = 0.0

    if args.plot_content in ("comparison", "prediction"):
        print(f"Loading DeePTB model: {args.checkpoint}")
        model = build_model(checkpoint=str(args.checkpoint))
        print(f"Model device: {model.device}")

        bcal = Band(
            model=model,
            use_gui=False,
            results_path=str(output),
            device=model.device,
        )
        status = bcal.get_bands(
            data=str(structure_path),
            kpath_kwargs={
                "kline_type": "array",
                "kpath": kpoints_band,
                "xlist": xlist,
                "high_sym_kpoints": hs_x,
                "labels": hs_labels,
            },
            AtomicData_options={
                "r_max": args.r_max,
                "oer_max": args.oer_max,
                "pbc": True,
            },
        )
        pred_band_raw = to_numpy(status["eigenvalues"])
        if pred_band_raw.ndim == 3:
            pred_band_raw = pred_band_raw[0]
        if pred_band_raw.ndim != 2:
            raise ValueError(f"Unexpected DeePTB band shape {pred_band_raw.shape}")

    if pred_band_raw is not None:
        n_compare = min(ref_band_raw.shape[1], pred_band_raw.shape[1])
    else:
        n_compare = ref_band_raw.shape[1]

    ref_band = ref_band_raw[:, :n_compare].copy()

    if pred_band_raw is not None:
        pred_band = pred_band_raw[:, :n_compare].copy()
        if args.align == "loss":
            ref_min = float(np.min(ref_band))
            pred_min = float(np.min(pred_band))
            pred_to_ref_shift = ref_min - pred_min
        else:
            pred_to_ref_shift = 0.0
        pred_band_plot = pred_band + pred_to_ref_shift
    else:
        pred_band = None

    # Same FHI gauge for the y-axis.  Under loss alignment only DeePTB moves.
    energy_label = "Energy (FHI-aims band reference, eV)"

    first = 1 if args.first_band is None else args.first_band
    last = n_compare if args.last_band is None else args.last_band
    if not (1 <= first <= last <= n_compare):
        raise ValueError(f"Need 1 <= first-band <= last-band <= {n_compare}")
    b0, b1 = first - 1, last

    band_metric = None
    if pred_band_plot is not None:
        band_metric = comparison_metrics(
            pred_band_plot[:, b0:b1], ref_band[:, b0:b1]
        )

    # Auto y-range depends on requested content.
    if args.plot_content == "ground-truth":
        auto_emin = float(np.min(ref_band[:, b0:b1]))
        auto_emax = float(np.max(ref_band[:, b0:b1]))
        span = max(auto_emax - auto_emin, 1.0)
        auto_emin -= 0.03 * span
        auto_emax += 0.03 * span
    elif args.plot_content == "prediction":
        auto_emin = float(np.min(pred_band_plot[:, b0:b1]))
        auto_emax = float(np.max(pred_band_plot[:, b0:b1]))
        span = max(auto_emax - auto_emin, 1.0)
        auto_emin -= 0.03 * span
        auto_emax += 0.03 * span
    else:
        auto_emin, auto_emax = choose_auto_energy_range(
            ref_band[:, b0:b1], pred_band_plot[:, b0:b1]
        )

    plot_emin = auto_emin if args.emin is None else float(args.emin)
    plot_emax = auto_emax if args.emax is None else float(args.emax)
    if plot_emin >= plot_emax:
        raise ValueError("Need --emin < --emax")

    species = species_order(atoms)

    # ------------------------------------------------------------------
    # FHI-aims DOS in the native band-reference gauge
    # ------------------------------------------------------------------
    fhi_total = None
    fhi_species = {}
    missing_species = []
    fhi_energy = None
    fhi_total_y = None
    fhi_species_curves = {}
    fhi_scale = None
    fhi_dos_coverage = None

    if args.plot_content in ("comparison", "ground-truth"):
        fhi_total, fhi_species, missing_species = load_fhi_dos_bundle(
            aims_dir, species, args.dos_mode, aims_mu=args.aims_mu
        )

        # Energy-reference rule:
        # ordinary FHI-aims DOS energy is already in the same practical
        # band-reference gauge.  Do NOT convert it back to raw/vacuum and do
        # NOT subtract a band minimum from it.
        fhi_energy = fhi_total["energy_band_ref"].copy()
        fhi_total_y = fhi_total["values"][:, 0].copy()

        fhi_species_curves = {
            sym: {
                "energy": datum["energy_band_ref"].copy(),
                "total": datum["values"][:, 0].copy(),
                "all_columns": datum["values"].copy(),
                "path": datum["path"],
                "reference": datum["reference"],
                "mu": datum["mu"],
            }
            for sym, datum in fhi_species.items()
        }

        fhi_scale = scaling_factor(
            fhi_energy,
            fhi_total_y,
            plot_emin,
            plot_emax,
            args.dos_normalization,
        )
        fhi_total_y *= fhi_scale
        for sym in fhi_species_curves:
            fhi_species_curves[sym]["total"] *= fhi_scale

        fhi_dos_coverage = assess_fhi_dos_coverage(
            ref_band[:, b0:b1],
            fhi_energy,
        )
        print_fhi_dos_coverage_warning(fhi_dos_coverage)

    # ------------------------------------------------------------------
    # DeePTB DOS, only when requested
    # ------------------------------------------------------------------
    dptb_dos = None
    dptb_energy = None
    dptb_total_y = None
    dptb_species_curves = {}
    dptb_scale = None

    if args.plot_content in ("comparison", "prediction"):
        # We want:
        #   E_plot = E_DeePTB_raw + pred_to_ref_shift
        # so the raw sampling grid is E_plot - pred_to_ref_shift.
        energy_plot_grid = np.linspace(plot_emin, plot_emax, args.n_energy)
        energy_raw_grid = energy_plot_grid - pred_to_ref_shift

        dptb_dos = compute_deeptb_dos(
            model=model,
            structure_path=structure_path,
            atoms=atoms,
            kmesh=tuple(args.kmesh),
            energy_grid_raw=energy_raw_grid,
            dos_mode=args.dos_mode,
            sigma=args.sigma,
            spin_degeneracy=args.spin_degeneracy,
            k_chunk=args.k_chunk,
            state_block=args.state_block,
            r_max=args.r_max,
            oer_max=args.oer_max,
        )
        dptb_energy = dptb_dos["energy_raw"] + pred_to_ref_shift
        dptb_total_y = dptb_dos["total"].copy()
        dptb_species_curves = {
            k: v.copy() for k, v in dptb_dos["species"].items()
        }

        dptb_scale = scaling_factor(
            dptb_energy,
            dptb_total_y,
            plot_emin,
            plot_emax,
            args.dos_normalization,
        )
        dptb_total_y *= dptb_scale
        for sym in dptb_species_curves:
            dptb_species_curves[sym] *= dptb_scale

    dos_xlabel = (
        "DOS (states/eV)"
        if args.dos_normalization == "none"
        else "Normalized DOS"
    )

    # Adaptive species panels depend on requested source(s).
    if args.dos_mode == "total":
        panel_species = []
    elif args.plot_content == "ground-truth":
        panel_species = [s for s in species if s in fhi_species_curves]
    elif args.plot_content == "prediction":
        panel_species = [s for s in species if s in dptb_species_curves]
    else:
        panel_species = [
            s for s in species
            if s in fhi_species_curves and s in dptb_species_curves
        ]

    panel_names = ["Total"] + panel_species
    n_dos_panels = len(panel_names)

    width_ratios = [3.6] + [1.0] * n_dos_panels
    fig_width = 7.0 + 1.55 * n_dos_panels
    fig = plt.figure(figsize=(fig_width, 6.1))
    gs = fig.add_gridspec(
        1, 1 + n_dos_panels, width_ratios=width_ratios, wspace=0.12
    )

    ax_band = fig.add_subplot(gs[0, 0])
    dos_axes = [
        fig.add_subplot(gs[0, i + 1], sharey=ax_band)
        for i in range(n_dos_panels)
    ]

    # ------------------------------------------------------------------
    # Band plotting
    # ------------------------------------------------------------------
    cmap = plt.get_cmap("turbo")

    if args.plot_content == "ground-truth":
        # aimsplot-like debugging view: continuous black FHI-aims bands.
        for j in range(b0, b1):
            ax_band.plot(
                xlist, ref_band[:, j],
                color="black", lw=0.85, alpha=0.95
            )
    elif args.plot_content == "prediction":
        for j in range(b0, b1):
            ax_band.plot(
                xlist, pred_band_plot[:, j],
                color="black", lw=0.85, alpha=0.95
            )
    else:
        for j in range(b0, b1):
            color = cmap(j / (n_compare - 1)) if n_compare > 1 else cmap(0.5)
            ax_band.plot(
                xlist, pred_band_plot[:, j],
                color=color, lw=0.75, alpha=0.95
            )
            ax_band.plot(
                xlist[::args.ref_stride],
                ref_band[::args.ref_stride, j],
                linestyle="None",
                marker="o",
                color=color,
                markersize=2.2,
                alpha=0.72,
            )

    for x in hs_x[1:-1]:
        ax_band.axvline(x, linestyle=":", linewidth=0.8, color="black")
    if plot_emin <= 0.0 <= plot_emax:
        ax_band.axhline(0.0, linestyle=":", linewidth=0.8, color="black")

    ax_band.set_xticks(hs_x)
    ax_band.set_xticklabels(hs_labels)
    ax_band.set_xlim(xlist[0], xlist[-1])
    ax_band.set_ylim(plot_emin, plot_emax)
    ax_band.set_xlabel("k-path")
    ax_band.set_ylabel(energy_label)
    ax_band.tick_params(direction="in")

    if args.plot_content == "comparison":
        ax_band.legend(
            handles=[
                Line2D([0], [0], lw=1.2, label="DeePTB"),
                Line2D(
                    [0], [0], marker="o", linestyle="None",
                    markersize=4, label="DFT (FHI-aims)"
                ),
            ],
            loc="best",
            fontsize=8,
        )

    # ------------------------------------------------------------------
    # DOS plotting
    # ------------------------------------------------------------------
    for iax, (ax, name) in enumerate(zip(dos_axes, panel_names)):
        title = "Total DOS" if name == "Total" else f"{name} DOS"

        if args.plot_content in ("comparison", "ground-truth"):
            if name == "Total":
                e_fhi, y_fhi = fhi_energy, fhi_total_y
            else:
                e_fhi = fhi_species_curves[name]["energy"]
                y_fhi = fhi_species_curves[name]["total"]

            if args.plot_content == "ground-truth":
                ax.plot(y_fhi, e_fhi, color="black", lw=1.05, label="FHI-aims")
            else:
                ax.plot(y_fhi, e_fhi, lw=1.15, label="FHI-aims")

        if args.plot_content in ("comparison", "prediction"):
            if name == "Total":
                e_dp, y_dp = dptb_energy, dptb_total_y
            else:
                e_dp = dptb_energy
                y_dp = dptb_species_curves[name]

            if args.plot_content == "prediction":
                ax.plot(y_dp, e_dp, color="black", lw=1.05, label="DeePTB")
            else:
                ax.plot(
                    y_dp, e_dp, lw=1.15, linestyle="--", label="DeePTB"
                )

        if plot_emin <= 0.0 <= plot_emax:
            ax.axhline(0.0, linestyle=":", linewidth=0.8, color="black")
        ax.axvline(0.0, linewidth=0.7, alpha=0.6, color="black")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(dos_xlabel, fontsize=8)
        ax.tick_params(direction="in", labelleft=False, labelsize=8)
        ax.set_ylim(plot_emin, plot_emax)

        if iax == 0 and args.plot_content == "comparison":
            ax.legend(fontsize=7, loc="best")

    # Title / filenames.
    if args.title:
        fig.suptitle(args.title, y=0.995)
    else:
        if args.plot_content == "comparison":
            metric_txt = (
                f"band RMSE={band_metric['rmse']:.3f} eV  |  "
                if band_metric is not None else ""
            )
            subtitle = (
                f"{set_dir.name}: FHI-aims vs DeePTB  |  {metric_txt}"
                f"DOS {args.kmesh[0]}x{args.kmesh[1]}x{args.kmesh[2]}, "
                f"sigma={args.sigma:.2f} eV"
            )
        elif args.plot_content == "ground-truth":
            subtitle = f"{set_dir.name}: FHI-aims ground truth"
        else:
            subtitle = (
                f"{set_dir.name}: DeePTB prediction  |  "
                f"DOS {args.kmesh[0]}x{args.kmesh[1]}x{args.kmesh[2]}, "
                f"sigma={args.sigma:.2f} eV"
            )
        fig.suptitle(subtitle, y=0.995, fontsize=12)

    fig.subplots_adjust(top=0.91, left=0.07, right=0.99, bottom=0.12)

    stem = {
        "comparison": "band_dos_comparison",
        "prediction": "band_dos_prediction",
        "ground-truth": "band_dos_ground_truth",
    }[args.plot_content]
    png_path = output / f"{stem}.png"
    pdf_path = output / f"{stem}.pdf"
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Save numerical outputs
    # ------------------------------------------------------------------
    np.save(output / "dft_band_reference.npy", ref_band)
    np.save(output / "kpoints_band.npy", kpoints_band)
    np.save(output / "xlist.npy", xlist)
    if pred_band_plot is not None:
        np.save(output / "deeptb_band_aligned_to_fhi.npy", pred_band_plot)

    if fhi_total is not None:
        fhi_npz = {
            "energy_total_band_reference_eV": fhi_energy,
            "total_dos": fhi_total_y,
            "normalization_scale": np.asarray(fhi_scale),
            "mu_eV": np.asarray(
                np.nan if fhi_total["mu"] is None else fhi_total["mu"]
            ),
        }
        for sym, dat in fhi_species_curves.items():
            fhi_npz[f"{sym}_energy_band_reference_eV"] = dat["energy"]
            fhi_npz[f"{sym}_dos"] = dat["total"]
            fhi_npz[f"{sym}_all_pdos_columns_scaled"] = (
                dat["all_columns"] * fhi_scale
            )
        np.savez(output / "fhi_aims_dos.npz", **fhi_npz)

    if dptb_dos is not None:
        dptb_npz = {
            "energy_aligned_to_fhi_eV": dptb_energy,
            "energy_model_raw_eV": dptb_dos["energy_raw"],
            "total_dos": dptb_total_y,
            "kpoints": dptb_dos["kpoints"],
            "prediction_to_fhi_shift_eV": np.asarray(pred_to_ref_shift),
            "normalization_scale": np.asarray(dptb_scale),
        }
        for sym, y in dptb_species_curves.items():
            dptb_npz[f"{sym}_dos"] = y
        np.savez(output / "deeptb_dos.npz", **dptb_npz)

    report_path = output / f"{stem}_report.txt"
    with report_path.open("w", encoding="utf-8") as fh:
        fh.write("FHI-aims / DeePTB band + DOS comparison\n")
        fh.write("=" * 78 + "\n")
        fh.write(f"plot content: {args.plot_content}\n")
        fh.write(f"set_dir: {set_dir}\n")
        fh.write(f"aims_dir: {aims_dir}\n")
        fh.write(f"checkpoint: {args.checkpoint}\n")
        fh.write(f"band mode: {args.mode}\n")
        fh.write(f"DOS mode: {args.dos_mode}\n")
        fh.write(f"species in structure: {species}\n")
        fh.write(f"species panels: {panel_species}\n")
        fh.write(f"band k-points: {len(kpoints_band)}\n")
        fh.write(f"supervised DFT bands: {n_supervised}\n")
        if n_dft_all is not None:
            fh.write(f"DFT all-electron bands: {n_dft_all}\n")
            fh.write(f"DFT 1s core bands stripped: {n_core}\n")
        fh.write(f"DFT bands in selected mode: {ref_band_raw.shape[1]}\n")
        if pred_band_raw is not None:
            fh.write(f"DeePTB raw bands: {pred_band_raw.shape[1]}\n")
            fh.write(f"common compared bands: {n_compare}\n")
            fh.write(
                f"DeePTB rigid shift into FHI gauge: {pred_to_ref_shift:.10f} eV\n"
            )
        fh.write(f"plotted bands: {first}-{last}\n")
        fh.write(f"alignment: {args.align}\n")
        fh.write(
            "FHI band/DOS gauge handling: FHI bands are kept native; ordinary "
            "non-raw DOS energies are used directly. Raw DOS, if necessary, is "
            "converted as E_raw - mu.\n"
        )
        fh.write(
            f"plot energy range: {plot_emin:.6f} .. {plot_emax:.6f} eV\n"
        )
        if band_metric is not None:
            fh.write(f"band RMSE: {band_metric['rmse']:.10f} eV\n")
            fh.write(f"band MAE: {band_metric['mae']:.10f} eV\n")

        if fhi_total is not None:
            fh.write("\nFHI-aims DOS\n")
            fh.write("-" * 78 + "\n")
            fh.write(f"total DOS file: {fhi_total['path']}\n")
            fh.write(
                f"total DOS source reference: {fhi_total['reference']}\n"
            )
            fh.write(f"parsed/provided mu: {fhi_total['mu']}\n")
            fh.write(
                f"FHI DOS energy extent in band gauge: "
                f"{np.min(fhi_energy):.8f} .. {np.max(fhi_energy):.8f} eV\n"
            )
            fh.write(f"FHI normalization scale: {fhi_scale:.12g}\n")
            if fhi_dos_coverage is not None:
                fh.write(
                    f"FHI plotted band range: "
                    f"{fhi_dos_coverage['band_min']:.8f} .. "
                    f"{fhi_dos_coverage['band_max']:.8f} eV\n"
                )
                fh.write(
                    f"FHI DOS source range: "
                    f"{fhi_dos_coverage['dos_min']:.8f} .. "
                    f"{fhi_dos_coverage['dos_max']:.8f} eV\n"
                )
                if fhi_dos_coverage["incomplete_low"]:
                    fh.write(
                        "WARNING: FHI DOS does not cover the lowest "
                        f"{fhi_dos_coverage['missing_low']:.8f} eV of the "
                        "plotted band spectrum. Missing DOS there reflects "
                        "the original DOS calculation window, not absence of states.\n"
                    )
                if fhi_dos_coverage["incomplete_high"]:
                    fh.write(
                        "WARNING: FHI DOS does not cover the highest "
                        f"{fhi_dos_coverage['missing_high']:.8f} eV of the "
                        "plotted band spectrum. Missing DOS there reflects "
                        "the original DOS calculation window, not absence of states.\n"
                    )
            if missing_species:
                fh.write(
                    f"FHI species PDOS missing/skipped: {missing_species}\n"
                )

        if dptb_dos is not None:
            fh.write("\nDeePTB DOS\n")
            fh.write("-" * 78 + "\n")
            fh.write(
                f"kmesh: {tuple(args.kmesh)} ({np.prod(args.kmesh)} k-points)\n"
            )
            fh.write(f"Gaussian sigma: {args.sigma} eV\n")
            fh.write(f"spin degeneracy: {args.spin_degeneracy}\n")
            fh.write(f"TB orbitals: {dptb_dos['norb']}\n")
            fh.write(f"DeePTB normalization scale: {dptb_scale:.12g}\n")
            fh.write(
                f"species-PDOS sum relative max error: "
                f"{dptb_dos['pdos_sum_relative_max_error']}\n"
            )

    print("\n" + "=" * 78)
    print("band_dos_compare complete")
    print("=" * 78)
    print(f"Plot content:          {args.plot_content}")
    print(f"FHI band gauge:        native")
    if fhi_total is not None:
        print(f"FHI DOS file:          {fhi_total['path'].name}")
        print(f"FHI DOS reference:     {fhi_total['reference']}")
        print(f"FHI DOS mu:            {fhi_total['mu']}")
    if pred_band_raw is not None:
        print(f"DeePTB -> FHI shift:   {pred_to_ref_shift:+.8f} eV")
    if band_metric is not None:
        print(f"Band RMSE:             {band_metric['rmse']:.6f} eV")
    print(f"Species panels:        {panel_species}")
    print(f"Saved PNG:             {png_path}")
    print(f"Saved PDF:             {pdf_path}")
    print(f"Saved report:          {report_path}")


if __name__ == "__main__":
    main()
