"""
Compare two DeePTB prediction sources using DFT-defined host-like VBM, CBM, and
band-gap targets across hBN defect datasets.

Usage
-----
Compare two checkpoints or training-output directories::

    python evaluation/compare_frontiers.py \
        --source-a RUN_A \
        --source-b RUN_B \
        --train DATA/train \
        --val DATA/val \
        --output RESULTS/frontier_compare \
        --make-plots

Each source may be a checkpoint file, a DeePTB training directory containing
``nnsk.ep*.pth`` checkpoints, or an ``evaluate_model.py`` output directory
containing reusable ``_deeptb_work`` predictions.  When a training directory is
supplied, the highest numbered ``nnsk.ep*.pth`` checkpoint is selected.

Host-edge heuristic
-------------------
Valence occupancy is inferred from the structure using H=1, B=3, C=4, N=5,
O=6 valence electrons, with optional per-set charges from ``--charge-map``.
The default ``manifold`` rule searches for dispersive, host-like edge bands
near the occupancy frontier.  Connectivity is measured by the number of
neighboring bands that become near-degenerate with the candidate at the same
k-point.  If the strict edge pair gives a gap outside the configurable host-gap
plausibility window, the required neighbor count is relaxed toward one while
the minimum-dispersion criterion remains fixed.

The default model-to-DFT energy alignment is one constant occupied-state median
shift per structure.  This makes VBM/CBM errors meaningful despite the arbitrary
model energy zero; band-gap error is gauge-invariant.

The DFT edge definition is a spectral heuristic, not a localization analysis.
For ambiguous defect states, wavefunction localization, IPR, or projected
character should be used for physical validation.

Outputs include ``frontier_comparison.csv``, ``summary_by_split.csv``,
``comparison_report.txt``, optional ``flagged_cases.csv``/``failures.csv``, and
optional edge/gap error plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from ase.io import read

# DeePTB imports are only needed if a source is a checkpoint/training folder.
# They are kept at module level so environment errors are obvious.
from dptb.nn.build import build_model
from dptb.postprocess.bandstructure.band import Band


VALENCE_ELECTRONS = {
    "H": 1,
    "B": 3,
    "C": 4,
    "N": 5,
    "O": 6,
}

EPOCH_RE = re.compile(r"nnsk\.ep(\d+)\.pth$")


def natural_key(text: str):
    return [int(x) if x.isdigit() else x.lower()
            for x in re.split(r"(\d+)", text)]


def ensure_2d_eigenvalues(arr: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 3:
        if a.shape[0] != 1:
            raise ValueError(f"{name}: expected one frame, got shape {a.shape}")
        a = a[0]
    if a.ndim != 2:
        raise ValueError(f"{name}: expected (nk,nb) or (1,nk,nb), got {a.shape}")
    return np.asarray(a, dtype=float)


def load_deeptb_bandstructure(path: Path) -> np.ndarray:
    obj = np.load(path, allow_pickle=True)
    if obj.shape != () or obj.dtype != object:
        raise ValueError(f"Unexpected DeePTB bandstructure format: {path}")
    d = obj.item()
    if "eigenvalues" not in d:
        raise KeyError(f"{path} does not contain 'eigenvalues'")
    return ensure_2d_eigenvalues(np.asarray(d["eigenvalues"]),
                                 f"{path}:eigenvalues")


def calculate_xlist(structure_path: Path, kpoints: np.ndarray) -> np.ndarray:
    atoms = read(str(structure_path))
    reciprocal = 2.0 * np.pi * np.asarray(atoms.cell.reciprocal())
    kcart = np.asarray(kpoints, dtype=float) @ reciprocal
    x = np.zeros(len(kcart), dtype=float)
    if len(kcart) > 1:
        x[1:] = np.cumsum(np.linalg.norm(np.diff(kcart, axis=0), axis=1))
    return x


def prepare_output_dir(path: Path, overwrite: bool):
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {path}\n"
                "Use a new versioned directory or pass --overwrite."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]):
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames),
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_values(values: Iterable[Any]) -> np.ndarray:
    out = []
    for x in values:
        try:
            f = float(x)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f):
            out.append(f)
    return np.asarray(out, dtype=float)


def rmse(values: Iterable[Any]) -> float:
    x = finite_values(values)
    return float(np.sqrt(np.mean(x * x))) if x.size else math.nan


def mae(values: Iterable[Any]) -> float:
    x = finite_values(values)
    return float(np.mean(np.abs(x))) if x.size else math.nan


def mean(values: Iterable[Any]) -> float:
    x = finite_values(values)
    return float(np.mean(x)) if x.size else math.nan


def resolve_checkpoint(source: Path) -> Optional[Path]:
    """
    Return checkpoint if source is a checkpoint/training folder.
    Return None if source looks like evaluate_model output.
    """
    if source.is_dir() and (source / "_deeptb_work").is_dir():
        return None

    if source.is_file():
        if source.suffix != ".pth":
            raise ValueError(f"Prediction source file is not a .pth checkpoint: {source}")
        return source

    if not source.is_dir():
        raise FileNotFoundError(f"Prediction source does not exist: {source}")

    candidates: List[Tuple[int, Path]] = []
    for base in (source / "checkpoint", source):
        if not base.is_dir():
            continue
        for p in base.glob("nnsk.ep*.pth"):
            m = EPOCH_RE.search(p.name)
            if m:
                candidates.append((int(m.group(1)), p))

    if not candidates:
        raise FileNotFoundError(
            f"No nnsk.ep*.pth checkpoint found in {source} or {source/'checkpoint'}"
        )

    candidates.sort(key=lambda t: t[0])
    return candidates[-1][1]


@dataclass
class PredictionSource:
    name: str
    path: Path
    kind: str                       # "eval_dir" or "checkpoint"
    checkpoint: Optional[Path]
    model: Any = None

    @classmethod
    def create(cls, label: str, source_path: Path):
        ckpt = resolve_checkpoint(source_path)
        if ckpt is None:
            return cls(label, source_path, "eval_dir", None, None)
        return cls(label, source_path, "checkpoint", ckpt, None)

    def initialize(self):
        if self.kind == "checkpoint":
            print(f"Loading {self.name} model once: {self.checkpoint}")
            self.model = build_model(checkpoint=str(self.checkpoint))

    def get_prediction(
        self,
        split: str,
        set_dir: Path,
        output_work_root: Path,
        kpoints: np.ndarray,
        r_max: float,
        oer_max: float,
    ) -> np.ndarray:
        if self.kind == "eval_dir":
            band_file = (
                self.path / "_deeptb_work" / split / set_dir.name /
                "bandstructure.npy"
            )
            if not band_file.is_file():
                raise FileNotFoundError(
                    f"{self.name}: reusable prediction not found: {band_file}"
                )
            return load_deeptb_bandstructure(band_file)

        if self.model is None:
            raise RuntimeError(f"{self.name}: model was not initialized")

        work_dir = output_work_root / self.name / split / set_dir.name
        work_dir.mkdir(parents=True, exist_ok=True)

        structure_path = set_dir / "xdat.traj"
        xlist = calculate_xlist(structure_path, kpoints)
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
            model=self.model,
            use_gui=False,
            results_path=str(work_dir),
            device=self.model.device,
        )
        bcal.get_bands(
            data=str(structure_path),
            kpath_kwargs=kpath_kwargs,
            AtomicData_options=atomic_data_options,
        )

        band_file = work_dir / "bandstructure.npy"
        if not band_file.is_file():
            raise FileNotFoundError(
                f"{self.name}: DeePTB did not write {band_file}"
            )
        return load_deeptb_bandstructure(band_file)


def load_charge_map(path: Optional[Path]) -> Dict[str, float]:
    """
    CSV format:
        set_id,charge
        set.000007,0
        set.000048,-1

    Charge convention:
        +1 means one electron removed.
        -1 means one electron added.
    """
    if path is None:
        return {}
    out: Dict[str, float] = {}
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or "set_id" not in reader.fieldnames or "charge" not in reader.fieldnames:
            raise ValueError("--charge-map requires CSV columns: set_id,charge")
        for row in reader:
            out[str(row["set_id"]).strip()] = float(row["charge"])
    return out


def composition_and_electrons(
    structure_path: Path,
    charge: float,
) -> Tuple[Dict[str, int], float]:
    atoms = read(str(structure_path))
    comp: Dict[str, int] = {}
    total = 0.0
    unknown = []
    for sym in atoms.get_chemical_symbols():
        comp[sym] = comp.get(sym, 0) + 1
    for sym, n in comp.items():
        if sym not in VALENCE_ELECTRONS:
            unknown.append(sym)
        else:
            total += n * VALENCE_ELECTRONS[sym]
    if unknown:
        raise ValueError(
            f"No default valence-electron count for species {sorted(set(unknown))}. "
            "Edit VALENCE_ELECTRONS in the script before using these species."
        )
    # Positive charge = electrons removed.
    total -= charge
    return comp, total


@dataclass
class OccupancyInfo:
    n_electrons: float
    n_full: int                 # number of fully occupied bands
    n_partial: int              # number of partially occupied bands inferred
    first_unoccupied: int       # 1-based strictly unoccupied band
    note: str


def infer_occupancy(
    n_electrons: float,
    spin_deg: int,
    nbands: int,
    integer_tol: float = 1e-8,
) -> OccupancyInfo:
    """
    Electron-count occupancy for a non-spin-resolved, spin-degenerate spectrum.
    """
    if spin_deg <= 0:
        raise ValueError("spin_deg must be positive")

    nearest = round(n_electrons)
    if abs(n_electrons - nearest) > integer_tol:
        raise ValueError(
            f"Electron count {n_electrons} is non-integer. "
            "Use an integer charge map or adapt the occupancy rule."
        )
    ne = int(nearest)

    n_full = ne // spin_deg
    rem = ne % spin_deg
    n_partial = 1 if rem else 0
    first_unoccupied = n_full + n_partial + 1  # 1-based

    if n_full < 1:
        raise ValueError("No fully occupied band inferred.")
    if first_unoccupied > nbands:
        raise ValueError(
            f"First unoccupied band {first_unoccupied} exceeds available DFT bands {nbands}"
        )

    note = (
        f"{ne} valence e-, spin_deg={spin_deg}, "
        f"{n_full} full band(s), {n_partial} partial band(s)"
    )
    return OccupancyInfo(ne, n_full, n_partial, first_unoccupied, note)


@dataclass
class EdgeRuleResult:
    band_1based: int
    energy: float
    dispersion: float
    overlap_count: int
    fallback: bool
    reason: str


@dataclass
class AdaptiveEdgePairResult:
    vbm: EdgeRuleResult
    cbm: EdgeRuleResult
    initial_vbm: EdgeRuleResult
    initial_cbm: EdgeRuleResult
    initial_gap_eV: float
    final_gap_eV: float
    initial_gap_flag: bool
    final_gap_plausible: bool
    adaptive_used: bool
    vbm_requirement: int
    cbm_requirement: int
    unresolved: bool
    reason: str


def gap_in_plausible_range(gap_eV: float, reference_gap_eV: float, gap_tol_eV: float) -> bool:
    return (reference_gap_eV - gap_tol_eV) <= gap_eV <= (reference_gap_eV + gap_tol_eV)


def count_samek_near_degenerate_neighbors(
    eigs: np.ndarray,
    band_0based: int,
    direction: str,
    lookaround: int,
    degeneracy_tol: float,
) -> int:
    """
    Count DISTINCT neighboring bands that come within degeneracy_tol of the
    candidate at at least one SAME k-point.

    This is a closer spectral proxy for "crosses/intersects another band" than
    simply asking whether the two global energy ranges overlap.
    """
    nb = eigs.shape[1]
    if direction == "higher":
        others = range(
            band_0based + 1,
            min(nb, band_0based + 1 + lookaround),
        )
    elif direction == "lower":
        others = range(
            max(0, band_0based - lookaround),
            band_0based,
        )
    else:
        raise ValueError(direction)

    e0 = eigs[:, band_0based]
    count = 0
    for j in others:
        min_samek_sep = float(np.min(np.abs(eigs[:, j] - e0)))
        if min_samek_sep <= degeneracy_tol:
            count += 1
    return count


def detect_vbm_neardeg(
    eigs: np.ndarray,
    occupancy: OccupancyInfo,
    rule: str,
    min_neighbors: int,
    lookaround: int,
    degeneracy_tol: float,
    min_dispersion: float,
    edge_search_bands: int,
) -> EdgeRuleResult:
    top_full_0 = occupancy.n_full - 1

    if rule == "top-full":
        e = eigs[:, top_full_0]
        return EdgeRuleResult(
            band_1based=top_full_0 + 1,
            energy=float(np.max(e)),
            dispersion=float(np.ptp(e)),
            overlap_count=count_samek_near_degenerate_neighbors(
                eigs, top_full_0, "lower", lookaround, degeneracy_tol
            ),
            fallback=False,
            reason="highest fully occupied band",
        )

    stop = max(-1, top_full_0 - edge_search_bands)
    for j in range(top_full_0, stop, -1):
        e = eigs[:, j]
        disp = float(np.ptp(e))
        neighbors = count_samek_near_degenerate_neighbors(
            eigs, j, "lower", lookaround, degeneracy_tol
        )
        if disp >= min_dispersion and neighbors >= min_neighbors:
            return EdgeRuleResult(
                band_1based=j + 1,
                energy=float(np.max(e)),
                dispersion=disp,
                overlap_count=neighbors,
                fallback=False,
                reason=(
                    "highest fully occupied same-k-connected band: "
                    f"dispersion={disp:.4f} eV, near-degenerate lower neighbors={neighbors}"
                ),
            )

    e = eigs[:, top_full_0]
    neighbors = count_samek_near_degenerate_neighbors(
        eigs, top_full_0, "lower", lookaround, degeneracy_tol
    )
    return EdgeRuleResult(
        band_1based=top_full_0 + 1,
        energy=float(np.max(e)),
        dispersion=float(np.ptp(e)),
        overlap_count=neighbors,
        fallback=True,
        reason=(
            "no same-k-connected occupied band found; "
            "fell back to highest fully occupied band"
        ),
    )


def detect_cbm_neardeg(
    eigs: np.ndarray,
    occupancy: OccupancyInfo,
    min_neighbors: int,
    lookaround: int,
    degeneracy_tol: float,
    min_dispersion: float,
    edge_search_bands: int,
) -> EdgeRuleResult:
    first_unocc_0 = occupancy.first_unoccupied - 1
    stop = min(eigs.shape[1], first_unocc_0 + edge_search_bands)

    for j in range(first_unocc_0, stop):
        e = eigs[:, j]
        disp = float(np.ptp(e))
        neighbors = count_samek_near_degenerate_neighbors(
            eigs, j, "higher", lookaround, degeneracy_tol
        )
        if disp >= min_dispersion and neighbors >= min_neighbors:
            return EdgeRuleResult(
                band_1based=j + 1,
                energy=float(np.min(e)),
                dispersion=disp,
                overlap_count=neighbors,
                fallback=False,
                reason=(
                    "lowest unoccupied same-k-connected band: "
                    f"dispersion={disp:.4f} eV, near-degenerate higher neighbors={neighbors}"
                ),
            )

    e = eigs[:, first_unocc_0]
    neighbors = count_samek_near_degenerate_neighbors(
        eigs, first_unocc_0, "higher", lookaround, degeneracy_tol
    )
    return EdgeRuleResult(
        band_1based=first_unocc_0 + 1,
        energy=float(np.min(e)),
        dispersion=float(np.ptp(e)),
        overlap_count=neighbors,
        fallback=True,
        reason=(
            "no same-k-connected unoccupied band found; "
            "fell back to first strictly unoccupied band"
        ),
    )


def detect_dft_edges_adaptive_neardeg(
    eigs: np.ndarray,
    occupancy: OccupancyInfo,
    rule: str,
    min_neighbors: int,
    lookaround: int,
    degeneracy_tol: float,
    min_dispersion: float,
    edge_search_bands: int,
    reference_gap_eV: float,
    gap_tol_eV: float,
) -> AdaptiveEdgePairResult:
    """
    Adaptive DFT host-edge rule based on same-k near-degenerate neighbors.

    Start from the strict neighbor-count requirement. If the resulting host gap
    is implausible, relax the VBM and CBM requirements independently toward one
    while keeping the minimum-dispersion threshold fixed.
    """
    strict_req = max(1, int(min_neighbors))
    reqs = list(range(strict_req, 0, -1))

    v_by_req = {}
    c_by_req = {}
    for req in reqs:
        v_by_req[req] = detect_vbm_neardeg(
            eigs, occupancy, rule,
            req, lookaround, degeneracy_tol, min_dispersion, edge_search_bands
        )
        c_by_req[req] = detect_cbm_neardeg(
            eigs, occupancy,
            req, lookaround, degeneracy_tol, min_dispersion, edge_search_bands
        )

    initial_v = v_by_req[strict_req]
    initial_c = c_by_req[strict_req]
    initial_gap = float(initial_c.energy - initial_v.energy)
    initial_ok = (
        not initial_v.fallback
        and not initial_c.fallback
        and initial_c.band_1based > initial_v.band_1based
        and gap_in_plausible_range(initial_gap, reference_gap_eV, gap_tol_eV)
    )
    initial_flag = not initial_ok

    if initial_ok:
        return AdaptiveEdgePairResult(
            vbm=initial_v, cbm=initial_c,
            initial_vbm=initial_v, initial_cbm=initial_c,
            initial_gap_eV=initial_gap, final_gap_eV=initial_gap,
            initial_gap_flag=False, final_gap_plausible=True,
            adaptive_used=False,
            vbm_requirement=strict_req, cbm_requirement=strict_req,
            unresolved=False,
            reason="strict same-k rule already inside plausible host-gap window",
        )

    candidates = []
    for rv in reqs:
        v = v_by_req[rv]
        if rule == "manifold" and v.fallback:
            continue
        for rc in reqs:
            c = c_by_req[rc]
            if c.fallback or c.band_1based <= v.band_1based:
                continue
            gap = float(c.energy - v.energy)
            relax_cost = (strict_req - rv) + (strict_req - rc)
            frontier_distance = (
                max(0, occupancy.n_full - v.band_1based)
                + max(0, c.band_1based - occupancy.first_unoccupied)
            )
            candidates.append({
                "v": v, "c": c, "rv": rv, "rc": rc,
                "gap": gap,
                "plausible": gap_in_plausible_range(
                    gap, reference_gap_eV, gap_tol_eV
                ),
                "relax_cost": relax_cost,
                "frontier_distance": frontier_distance,
                "gap_distance": abs(gap - reference_gap_eV),
            })

    plausible = [x for x in candidates if x["plausible"]]
    if plausible:
        chosen = min(
            plausible,
            key=lambda x: (
                x["relax_cost"],
                x["frontier_distance"],
                x["gap_distance"],
                -(x["rv"] + x["rc"]),
            ),
        )
        return AdaptiveEdgePairResult(
            vbm=chosen["v"], cbm=chosen["c"],
            initial_vbm=initial_v, initial_cbm=initial_c,
            initial_gap_eV=initial_gap, final_gap_eV=chosen["gap"],
            initial_gap_flag=True, final_gap_plausible=True,
            adaptive_used=True,
            vbm_requirement=chosen["rv"], cbm_requirement=chosen["rc"],
            unresolved=False,
            reason=(
                "strict result flagged; relaxed same-k near-degeneracy "
                f"requirement to VBM>={chosen['rv']}, CBM>={chosen['rc']} "
                "and recovered a plausible host gap"
            ),
        )

    if candidates:
        chosen = min(
            candidates,
            key=lambda x: (
                x["gap_distance"],
                x["relax_cost"],
                x["frontier_distance"],
            ),
        )
        return AdaptiveEdgePairResult(
            vbm=chosen["v"], cbm=chosen["c"],
            initial_vbm=initial_v, initial_cbm=initial_c,
            initial_gap_eV=initial_gap, final_gap_eV=chosen["gap"],
            initial_gap_flag=True, final_gap_plausible=False,
            adaptive_used=(
                chosen["rv"] != strict_req or chosen["rc"] != strict_req
                or chosen["v"].band_1based != initial_v.band_1based
                or chosen["c"].band_1based != initial_c.band_1based
            ),
            vbm_requirement=chosen["rv"], cbm_requirement=chosen["rc"],
            unresolved=True,
            reason=(
                "no same-k candidate entered the plausible host-gap window; "
                "returned closest candidate and flagged unresolved"
            ),
        )

    final_gap = float(initial_c.energy - initial_v.energy)
    return AdaptiveEdgePairResult(
        vbm=initial_v, cbm=initial_c,
        initial_vbm=initial_v, initial_cbm=initial_c,
        initial_gap_eV=initial_gap, final_gap_eV=final_gap,
        initial_gap_flag=True, final_gap_plausible=False,
        adaptive_used=False,
        vbm_requirement=strict_req, cbm_requirement=strict_req,
        unresolved=True,
        reason=(
            "no non-fallback same-k-connected edge pair found even after "
            "relaxing to one neighboring band"
        ),
    )


def edge_energy_at_band(
    eigs: np.ndarray,
    band_1based: int,
    edge: str,
) -> float:
    j = band_1based - 1
    if j < 0 or j >= eigs.shape[1]:
        raise IndexError(
            f"Requested band {band_1based}, but spectrum has {eigs.shape[1]} bands"
        )
    if edge == "vbm":
        return float(np.max(eigs[:, j]))
    if edge == "cbm":
        return float(np.min(eigs[:, j]))
    raise ValueError(edge)


def alignment_shift(
    dft: np.ndarray,
    pred: np.ndarray,
    n_full: int,
    method: str,
) -> float:
    """
    Return s such that pred_aligned = pred + s.
    """
    n = min(dft.shape[1], pred.shape[1])
    if n_full > n:
        raise ValueError(
            f"Need {n_full} fully occupied bands for alignment, "
            f"but only {n} comparable bands exist."
        )

    if method == "none":
        return 0.0

    if method == "lowest":
        return float(np.min(dft[:, :n]) - np.min(pred[:, :n]))

    diff = dft[:, :n_full] - pred[:, :n_full]
    if method == "occupied-median":
        return float(np.median(diff))
    if method == "occupied-mean":
        return float(np.mean(diff))

    raise ValueError(method)


def discover_cases(
    train_root: Optional[Path],
    val_root: Optional[Path],
    case_glob: str,
) -> List[Tuple[str, Path]]:
    roots = []
    if train_root is not None:
        roots.append(("train", train_root))
    if val_root is not None:
        roots.append(("val", val_root))
    if not roots:
        raise ValueError("Provide at least one of --train or --val.")

    cases: List[Tuple[str, Path]] = []
    for split, root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"{split} root does not exist: {root}")
        found = sorted(
            [p for p in root.glob(case_glob) if p.is_dir()],
            key=lambda p: natural_key(p.name),
        )
        for p in found:
            cases.append((split, p))
    if not cases:
        raise RuntimeError("No set.* directories found.")
    return cases


def source_label(path: Path) -> str:
    name = path.name or path.parent.name
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def evaluate_case(
    split: str,
    set_dir: Path,
    source_a: PredictionSource,
    source_b: PredictionSource,
    work_root: Path,
    charge_map: Dict[str, float],
    args,
) -> Dict[str, Any]:

    structure_path = set_dir / "xdat.traj"
    kpoints_path = set_dir / "kpoints.npy"
    dft_path = set_dir / "eigenvalues.npy"
    for p in (structure_path, kpoints_path, dft_path):
        if not p.is_file():
            raise FileNotFoundError(f"{set_dir}: missing {p.name}")

    kpoints = np.asarray(np.load(kpoints_path), dtype=float)
    dft = ensure_2d_eigenvalues(np.load(dft_path), str(dft_path))
    if kpoints.ndim != 2 or kpoints.shape[1] != 3:
        raise ValueError(f"{kpoints_path}: expected (nk,3), got {kpoints.shape}")
    if dft.shape[0] != kpoints.shape[0]:
        raise ValueError(
            f"{set_dir}: DFT nk={dft.shape[0]} != kpoints nk={kpoints.shape[0]}"
        )

    pred_a = source_a.get_prediction(
        split, set_dir, work_root,
        kpoints, args.r_max, args.oer_max
    )
    pred_b = source_b.get_prediction(
        split, set_dir, work_root,
        kpoints, args.r_max, args.oer_max
    )

    if pred_a.shape[0] != dft.shape[0] or pred_b.shape[0] != dft.shape[0]:
        raise ValueError(
            f"{set_dir}: k-point mismatch DFT={dft.shape[0]}, "
            f"A={pred_a.shape[0]}, B={pred_b.shape[0]}"
        )

    charge = float(charge_map.get(set_dir.name, args.default_charge))
    comp, ne = composition_and_electrons(structure_path, charge)
    occ = infer_occupancy(ne, args.spin_deg, dft.shape[1])

    # DFT defines the physical target edge indices.
    # Start with the strict same-k near-degeneracy rule. If the resulting host gap
    # is implausible, relax the connectivity count (3 -> 2 -> 1 by default)
    # while keeping the minimum-dispersion criterion fixed.
    dft_pair = detect_dft_edges_adaptive_neardeg(
        dft, occ, args.vbm_rule,
        args.min_overlaps, args.manifold_lookaround,
        args.degeneracy_tol, args.min_dispersion,
        args.edge_search_bands,
        args.reference_gap, args.gap_tol,
    )
    dft_vbm = dft_pair.vbm
    dft_cbm = dft_pair.cbm

    if dft_cbm.band_1based <= dft_vbm.band_1based:
        raise RuntimeError(
            f"{set_dir.name}: detected CBM band {dft_cbm.band_1based} "
            f"<= VBM band {dft_vbm.band_1based}"
        )

    # Primary model energies use the SAME DFT-resolved edge band indices.
    a_vbm_raw = edge_energy_at_band(pred_a, dft_vbm.band_1based, "vbm")
    a_cbm_raw = edge_energy_at_band(pred_a, dft_cbm.band_1based, "cbm")
    b_vbm_raw = edge_energy_at_band(pred_b, dft_vbm.band_1based, "vbm")
    b_cbm_raw = edge_energy_at_band(pred_b, dft_cbm.band_1based, "cbm")

    dft_eg = dft_cbm.energy - dft_vbm.energy
    a_eg = a_cbm_raw - a_vbm_raw
    b_eg = b_cbm_raw - b_vbm_raw

    shift_a = alignment_shift(dft, pred_a, occ.n_full, args.alignment)
    shift_b = alignment_shift(dft, pred_b, occ.n_full, args.alignment)

    a_vbm_aligned = a_vbm_raw + shift_a
    a_cbm_aligned = a_cbm_raw + shift_a
    b_vbm_aligned = b_vbm_raw + shift_b
    b_cbm_aligned = b_cbm_raw + shift_b

    dEVBM_a = a_vbm_aligned - dft_vbm.energy
    dECBM_a = a_cbm_aligned - dft_cbm.energy
    dEg_a = a_eg - dft_eg

    dEVBM_b = b_vbm_aligned - dft_vbm.energy
    dECBM_b = b_cbm_aligned - dft_cbm.energy
    dEg_b = b_eg - dft_eg

    # Secondary diagnostics: what edge indices would each model infer by itself?
    occ_a = infer_occupancy(ne, args.spin_deg, pred_a.shape[1])
    occ_b = infer_occupancy(ne, args.spin_deg, pred_b.shape[1])
    a_vbm_rule = detect_vbm_neardeg(
        pred_a, occ_a, args.vbm_rule,
        args.min_overlaps, args.manifold_lookaround,
        args.degeneracy_tol, args.min_dispersion,
        args.edge_search_bands,
    )
    a_cbm_rule = detect_cbm_neardeg(
        pred_a, occ_a,
        args.min_overlaps, args.manifold_lookaround,
        args.degeneracy_tol, args.min_dispersion,
        args.edge_search_bands,
    )
    b_vbm_rule = detect_vbm_neardeg(
        pred_b, occ_b, args.vbm_rule,
        args.min_overlaps, args.manifold_lookaround,
        args.degeneracy_tol, args.min_dispersion,
        args.edge_search_bands,
    )
    b_cbm_rule = detect_cbm_neardeg(
        pred_b, occ_b,
        args.min_overlaps, args.manifold_lookaround,
        args.degeneracy_tol, args.min_dispersion,
        args.edge_search_bands,
    )

    row: Dict[str, Any] = {
        "split": split,
        "set_id": set_dir.name,
        "charge": charge,
        "natoms": sum(comp.values()),
        "B": comp.get("B", 0),
        "C": comp.get("C", 0),
        "N": comp.get("N", 0),
        "O": comp.get("O", 0),
        "valence_electrons": occ.n_electrons,
        "fully_occupied_bands": occ.n_full,
        "partially_occupied_bands": occ.n_partial,
        "first_strictly_unoccupied_band": occ.first_unoccupied,
        "occupancy_note": occ.note,

        "connectivity_metric": "same-k-near-degeneracy",
        "reference_gap_eV": args.reference_gap,
        "gap_tolerance_eV": args.gap_tol,
        "dft_initial_vbm_band": dft_pair.initial_vbm.band_1based,
        "dft_initial_cbm_band": dft_pair.initial_cbm.band_1based,
        "dft_initial_Eg_eV": dft_pair.initial_gap_eV,
        "dft_initial_gap_flag": int(dft_pair.initial_gap_flag),
        "dft_adaptive_used": int(dft_pair.adaptive_used),
        "dft_final_gap_plausible": int(dft_pair.final_gap_plausible),
        "dft_unresolved_flag": int(dft_pair.unresolved),
        "dft_vbm_connectivity_requirement": dft_pair.vbm_requirement,
        "dft_cbm_connectivity_requirement": dft_pair.cbm_requirement,
        "dft_adaptive_reason": dft_pair.reason,

        "dft_vbm_band": dft_vbm.band_1based,
        "dft_vbm_raw_eV": dft_vbm.energy,
        "dft_vbm_dispersion_eV": dft_vbm.dispersion,
        "dft_vbm_overlap_count": dft_vbm.overlap_count,
        "dft_vbm_fallback": int(dft_vbm.fallback),
        "dft_vbm_reason": dft_vbm.reason,

        "dft_cbm_band": dft_cbm.band_1based,
        "dft_cbm_raw_eV": dft_cbm.energy,
        "dft_cbm_dispersion_eV": dft_cbm.dispersion,
        "dft_cbm_overlap_count": dft_cbm.overlap_count,
        "dft_cbm_fallback": int(dft_cbm.fallback),
        "dft_cbm_reason": dft_cbm.reason,

        "dft_Eg_eV": dft_eg,

        "A_shift_eV": shift_a,
        "A_vbm_raw_eV": a_vbm_raw,
        "A_vbm_aligned_eV": a_vbm_aligned,
        "A_cbm_raw_eV": a_cbm_raw,
        "A_cbm_aligned_eV": a_cbm_aligned,
        "A_Eg_eV": a_eg,
        "A_delta_EVBM_eV": dEVBM_a,
        "A_delta_ECBM_eV": dECBM_a,
        "A_delta_Eg_eV": dEg_a,
        "A_rule_vbm_band": a_vbm_rule.band_1based,
        "A_rule_cbm_band": a_cbm_rule.band_1based,

        "B_shift_eV": shift_b,
        "B_vbm_raw_eV": b_vbm_raw,
        "B_vbm_aligned_eV": b_vbm_aligned,
        "B_cbm_raw_eV": b_cbm_raw,
        "B_cbm_aligned_eV": b_cbm_aligned,
        "B_Eg_eV": b_eg,
        "B_delta_EVBM_eV": dEVBM_b,
        "B_delta_ECBM_eV": dECBM_b,
        "B_delta_Eg_eV": dEg_b,
        "B_rule_vbm_band": b_vbm_rule.band_1based,
        "B_rule_cbm_band": b_cbm_rule.band_1based,

        # Positive means B reduced the absolute error relative to A.
        "B_abs_error_reduction_EVBM_eV": abs(dEVBM_a) - abs(dEVBM_b),
        "B_abs_error_reduction_ECBM_eV": abs(dECBM_a) - abs(dECBM_b),
        "B_abs_error_reduction_Eg_eV": abs(dEg_a) - abs(dEg_b),

        "B_better_EVBM": int(abs(dEVBM_b) < abs(dEVBM_a)),
        "B_better_ECBM": int(abs(dECBM_b) < abs(dECBM_a)),
        "B_better_Eg": int(abs(dEg_b) < abs(dEg_a)),
    }
    return row


def make_summary(rows: List[Dict[str, Any]], label_a: str, label_b: str):
    out = []
    for split in sorted({r["split"] for r in rows}):
        sub = [r for r in rows if r["split"] == split]
        for tag, label in (("A", label_a), ("B", label_b)):
            out.append({
                "split": split,
                "model": label,
                "n_cases": len(sub),
                "MAE_delta_EVBM_eV": mae(r[f"{tag}_delta_EVBM_eV"] for r in sub),
                "RMSE_delta_EVBM_eV": rmse(r[f"{tag}_delta_EVBM_eV"] for r in sub),
                "bias_delta_EVBM_eV": mean(r[f"{tag}_delta_EVBM_eV"] for r in sub),
                "MAE_delta_ECBM_eV": mae(r[f"{tag}_delta_ECBM_eV"] for r in sub),
                "RMSE_delta_ECBM_eV": rmse(r[f"{tag}_delta_ECBM_eV"] for r in sub),
                "bias_delta_ECBM_eV": mean(r[f"{tag}_delta_ECBM_eV"] for r in sub),
                "MAE_delta_Eg_eV": mae(r[f"{tag}_delta_Eg_eV"] for r in sub),
                "RMSE_delta_Eg_eV": rmse(r[f"{tag}_delta_Eg_eV"] for r in sub),
                "bias_delta_Eg_eV": mean(r[f"{tag}_delta_Eg_eV"] for r in sub),
            })
    return out


def maybe_make_plots(
    output: Path,
    rows: List[Dict[str, Any]],
    label_a: str,
    label_b: str,
):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARNING: plots skipped; matplotlib import failed: {exc}")
        return

    x = np.arange(len(rows))
    ids = [r["set_id"] for r in rows]

    # Edge errors.
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.plot(x, [r["A_delta_EVBM_eV"] for r in rows],
            marker="o", markersize=3, linewidth=1, label=f"{label_a} ΔEVBM")
    ax.plot(x, [r["B_delta_EVBM_eV"] for r in rows],
            marker="o", markersize=3, linewidth=1, label=f"{label_b} ΔEVBM")
    ax.plot(x, [r["A_delta_ECBM_eV"] for r in rows],
            linestyle="--", linewidth=1, label=f"{label_a} ΔECBM")
    ax.plot(x, [r["B_delta_ECBM_eV"] for r in rows],
            linestyle="--", linewidth=1, label=f"{label_b} ΔECBM")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel("Structure index")
    ax.set_ylabel("Aligned edge error (eV)")
    ax.set_title("VBM / CBM errors against DFT")
    ax.legend(ncol=2)
    ax.tick_params(direction="in")
    fig.tight_layout()
    fig.savefig(output / "edge_error_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Gap errors.
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.plot(x, [r["A_delta_Eg_eV"] for r in rows],
            marker="o", markersize=3, linewidth=1, label=label_a)
    ax.plot(x, [r["B_delta_Eg_eV"] for r in rows],
            marker="o", markersize=3, linewidth=1, label=label_b)
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel("Structure index")
    ax.set_ylabel("ΔEg (eV)")
    ax.set_title("Host-gap error against DFT")
    ax.legend()
    ax.tick_params(direction="in")
    fig.tight_layout()
    fig.savefig(output / "gap_error_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description="Compare two arbitrary DeePTB training/evaluation sources using DFT-defined host band edges."
    )
    p.add_argument("--source-a", required=True, type=Path,
                   help="Checkpoint, training output folder, or evaluate_model output folder.")
    p.add_argument("--source-b", required=True, type=Path,
                   help="Checkpoint, training output folder, or evaluate_model output folder.")
    p.add_argument("--label-a", default=None,
                   help="Display label for source A. Default: source basename.")
    p.add_argument("--label-b", default=None,
                   help="Display label for source B. Default: source basename.")
    p.add_argument("--train", type=Path, default=None)
    p.add_argument("--val", type=Path, default=None)
    p.add_argument("--case-glob", default="set.*")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--r-max", type=float, default=5.0)
    p.add_argument("--oer-max", type=float, default=4.0)

    p.add_argument("--spin-deg", type=int, default=2,
                   help="Electrons per fully occupied band (default 2).")
    p.add_argument("--default-charge", type=float, default=0.0,
                   help="Default charge of every structure. +1 removes one electron.")
    p.add_argument("--charge-map", type=Path, default=None,
                   help="Optional CSV with columns set_id,charge.")

    p.add_argument("--vbm-rule", choices=["top-full", "manifold"],
                   default="manifold",
                   help="Host-VBM rule. 'manifold' is safer for occupied in-gap defect states.")
    p.add_argument("--min-overlaps", type=int, default=3,
                   help="Required distinct same-k near-degenerate neighboring bands.")
    p.add_argument("--manifold-lookaround", type=int, default=12,
                   help="Number of lower/higher neighboring bands checked around each candidate.")
    p.add_argument("--degeneracy-tol", type=float, default=0.10,
                   help="Same-k near-degeneracy tolerance in eV (default 0.10).")
    p.add_argument("--min-dispersion", type=float, default=0.15,
                   help="Minimum band dispersion in eV for a manifold-like edge.")
    p.add_argument("--edge-search-bands", type=int, default=24,
                   help="Maximum bands searched away from the occupancy frontier.")
    p.add_argument("--reference-gap", type=float, default=4.672,
                   help="Pristine/host DFT gap used only as a plausibility check (default 4.672 eV).")
    p.add_argument("--gap-tol", type=float, default=0.5,
                   help="Allowed +/- window around --reference-gap before adaptive relaxation (default 0.5 eV).")

    p.add_argument(
        "--alignment",
        choices=["occupied-median", "occupied-mean", "lowest", "none"],
        default="occupied-median",
        help=(
            "Constant model-to-DFT energy gauge for ΔEVBM/ΔECBM. "
            "Default occupied-median is recommended. "
            "'none' reports raw gauge-dependent edge differences."
        ),
    )
    p.add_argument("--make-plots", action="store_true")
    args = p.parse_args()

    if args.min_overlaps < 1:
        p.error("--min-overlaps must be >= 1")
    if args.gap_tol < 0:
        p.error("--gap-tol must be >= 0")
    if args.manifold_lookaround < 1:
        p.error("--manifold-lookaround must be >= 1")
    if args.edge_search_bands < 1:
        p.error("--edge-search-bands must be >= 1")
    if args.min_dispersion < 0:
        p.error("--min-dispersion must be >= 0")

    label_a = args.label_a or source_label(args.source_a)
    label_b = args.label_b or source_label(args.source_b)
    if label_a == label_b:
        label_a += "_A"
        label_b += "_B"

    prepare_output_dir(args.output, args.overwrite)
    charge_map = load_charge_map(args.charge_map)
    cases = discover_cases(args.train, args.val, args.case_glob)

    source_a = PredictionSource.create(label_a, args.source_a)
    source_b = PredictionSource.create(label_b, args.source_b)

    print("=" * 78)
    print("hBN frontier comparison")
    print("=" * 78)
    for s in (source_a, source_b):
        detail = s.checkpoint if s.kind == "checkpoint" else s.path
        print(f"{s.name:20s}: {s.kind:10s} {detail}")
    print(f"cases               : {len(cases)}")
    print(f"VBM rule            : {args.vbm_rule}")
    print(
        f"manifold rule       : >= {args.min_overlaps} near-degenerate neighbors, "
        f"lookaround={args.manifold_lookaround}, "
        f"min dispersion={args.min_dispersion} eV, "
        f"same-k tolerance={args.degeneracy_tol} eV"
    )
    print(f"host-gap window     : {args.reference_gap - args.gap_tol:.3f} to {args.reference_gap + args.gap_tol:.3f} eV")
    print("adaptive rule       : relax same-k neighbor count toward 1; keep minimum dispersion fixed")
    print(f"edge alignment      : {args.alignment}")
    print("=" * 78)

    if args.alignment == "none":
        print(
            "WARNING: --alignment none makes ΔEVBM and ΔECBM depend on the "
            "arbitrary model energy zero. ΔEg remains gauge-invariant."
        )

    source_a.initialize()
    source_b.initialize()

    work_root = args.output / "_deeptb_work"
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for i, (split, set_dir) in enumerate(cases, 1):
        print(f"[{i:3d}/{len(cases):3d}] {split:5s} {set_dir.name} ... ",
              end="", flush=True)
        try:
            row = evaluate_case(
                split, set_dir, source_a, source_b,
                work_root, charge_map, args
            )
            rows.append(row)
            print(
                f"DFT bands V/C={row['dft_vbm_band']}/{row['dft_cbm_band']} | "
                f"ΔEg A={row['A_delta_Eg_eV']:+.3f}, "
                f"B={row['B_delta_Eg_eV']:+.3f} eV"
            )
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

    if not rows:
        raise RuntimeError("All cases failed.")

    fields = [
        "split", "set_id", "charge", "natoms", "B", "C", "N", "O",
        "valence_electrons", "fully_occupied_bands",
        "partially_occupied_bands", "first_strictly_unoccupied_band",
        "occupancy_note",
        "connectivity_metric", "reference_gap_eV", "gap_tolerance_eV",
        "dft_initial_vbm_band", "dft_initial_cbm_band", "dft_initial_Eg_eV",
        "dft_initial_gap_flag", "dft_adaptive_used",
        "dft_final_gap_plausible", "dft_unresolved_flag",
        "dft_vbm_connectivity_requirement", "dft_cbm_connectivity_requirement",
        "dft_adaptive_reason",

        "dft_vbm_band", "dft_vbm_raw_eV", "dft_vbm_dispersion_eV",
        "dft_vbm_overlap_count", "dft_vbm_fallback", "dft_vbm_reason",
        "dft_cbm_band", "dft_cbm_raw_eV", "dft_cbm_dispersion_eV",
        "dft_cbm_overlap_count", "dft_cbm_fallback", "dft_cbm_reason",
        "dft_Eg_eV",

        "A_shift_eV", "A_vbm_raw_eV", "A_vbm_aligned_eV",
        "A_cbm_raw_eV", "A_cbm_aligned_eV", "A_Eg_eV",
        "A_delta_EVBM_eV", "A_delta_ECBM_eV", "A_delta_Eg_eV",
        "A_rule_vbm_band", "A_rule_cbm_band",

        "B_shift_eV", "B_vbm_raw_eV", "B_vbm_aligned_eV",
        "B_cbm_raw_eV", "B_cbm_aligned_eV", "B_Eg_eV",
        "B_delta_EVBM_eV", "B_delta_ECBM_eV", "B_delta_Eg_eV",
        "B_rule_vbm_band", "B_rule_cbm_band",

        "B_abs_error_reduction_EVBM_eV",
        "B_abs_error_reduction_ECBM_eV",
        "B_abs_error_reduction_Eg_eV",
        "B_better_EVBM", "B_better_ECBM", "B_better_Eg",
    ]
    write_csv(args.output / "frontier_comparison.csv", rows, fields)

    summary = make_summary(rows, label_a, label_b)
    summary_fields = [
        "split", "model", "n_cases",
        "MAE_delta_EVBM_eV", "RMSE_delta_EVBM_eV", "bias_delta_EVBM_eV",
        "MAE_delta_ECBM_eV", "RMSE_delta_ECBM_eV", "bias_delta_ECBM_eV",
        "MAE_delta_Eg_eV", "RMSE_delta_Eg_eV", "bias_delta_Eg_eV",
    ]
    write_csv(args.output / "summary_by_split.csv", summary, summary_fields)

    if failures:
        write_csv(
            args.output / "failures.csv",
            failures,
            ["split", "set_id", "set_dir", "error_type", "error", "traceback"],
        )

    flagged_rows = [
        r for r in rows
        if int(r["dft_initial_gap_flag"]) == 1 or int(r["dft_unresolved_flag"]) == 1
    ]
    if flagged_rows:
        write_csv(
            args.output / "flagged_cases.csv",
            flagged_rows,
            fields,
        )

    if args.make_plots:
        maybe_make_plots(args.output, rows, label_a, label_b)

    # Human-readable report.
    report = args.output / "comparison_report.txt"
    with report.open("w", encoding="utf-8") as fh:
        fh.write("hBN DeePTB frontier comparison\n")
        fh.write("=" * 78 + "\n")
        fh.write(f"source A: {label_a} -> {source_a.path}\n")
        if source_a.checkpoint:
            fh.write(f"resolved A checkpoint: {source_a.checkpoint}\n")
        fh.write(f"source B: {label_b} -> {source_b.path}\n")
        if source_b.checkpoint:
            fh.write(f"resolved B checkpoint: {source_b.checkpoint}\n")
        fh.write(f"successful cases: {len(rows)} / {len(cases)}\n")
        fh.write(f"failed cases: {len(failures)}\n")
        fh.write(f"VBM rule: {args.vbm_rule}\n")
        fh.write(
            f"manifold: initial min_overlaps={args.min_overlaps}, "
            f"lookaround={args.manifold_lookaround}, "
            f"same_k_degeneracy_tol={args.degeneracy_tol} eV, "
            f"min_dispersion={args.min_dispersion} eV, "
            f"edge_search_bands={args.edge_search_bands}\n"
        )
        fh.write(
            f"host-gap plausibility window: "
            f"{args.reference_gap - args.gap_tol:.6f} to "
            f"{args.reference_gap + args.gap_tol:.6f} eV\n"
        )
        fh.write(
            "adaptive policy: if strict gap is flagged, relax VBM/CBM "
            "same-k near-degenerate-neighbor counts independently down to 1; minimum "
            "dispersion is never relaxed\n"
        )
        fh.write(f"edge alignment: {args.alignment}\n\n")

        fh.write("Energy-reference note\n")
        fh.write("-" * 78 + "\n")
        fh.write(
            "ΔEVBM and ΔECBM are reported after one constant model-to-DFT "
            "energy shift per structure.  The default occupied-median shift "
            "uses all fully occupied non-core states and does not force either "
            "edge to agree.  Raw model-vs-DFT edge differences are gauge-dependent "
            "because the training loss does not constrain an absolute energy zero.\n"
        )
        fh.write(
            "ΔEg is gauge-invariant and is calculated directly from "
            "(ECBM-EVBM)_model - (ECBM-EVBM)_DFT.\n\n"
        )

        fh.write("Aggregate summary\n")
        fh.write("-" * 78 + "\n")
        for r in summary:
            fh.write(
                f"{r['split']:5s} {r['model']}: n={r['n_cases']}, "
                f"MAE ΔEVBM={r['MAE_delta_EVBM_eV']:.6f} eV, "
                f"MAE ΔECBM={r['MAE_delta_ECBM_eV']:.6f} eV, "
                f"MAE ΔEg={r['MAE_delta_Eg_eV']:.6f} eV\n"
            )

        fh.write("\nB improvement counts relative to A\n")
        fh.write("-" * 78 + "\n")
        for split in sorted({r["split"] for r in rows}):
            sub = [r for r in rows if r["split"] == split]
            fh.write(
                f"{split}: "
                f"|ΔEVBM| improved {sum(r['B_better_EVBM'] for r in sub)}/{len(sub)}, "
                f"|ΔECBM| improved {sum(r['B_better_ECBM'] for r in sub)}/{len(sub)}, "
                f"|ΔEg| improved {sum(r['B_better_Eg'] for r in sub)}/{len(sub)}\n"
            )

        fall_v = sum(int(r["dft_vbm_fallback"]) for r in rows)
        fall_c = sum(int(r["dft_cbm_fallback"]) for r in rows)
        initial_flagged = sum(int(r["dft_initial_gap_flag"]) for r in rows)
        adaptive_used = sum(int(r["dft_adaptive_used"]) for r in rows)
        unresolved = sum(int(r["dft_unresolved_flag"]) for r in rows)
        resolved = sum(
            int(r["dft_initial_gap_flag"]) and int(r["dft_final_gap_plausible"])
            for r in rows
        )
        fh.write("\nRule diagnostics\n")
        fh.write("-" * 78 + "\n")
        fh.write(f"Strict-rule gap flags: {initial_flagged}/{len(rows)}\n")
        fh.write(f"Adaptive relaxation used: {adaptive_used}/{len(rows)}\n")
        fh.write(f"Flagged cases resolved into plausible gap window: {resolved}/{initial_flagged if initial_flagged else 0}\n")
        fh.write(f"Still unresolved after relaxation: {unresolved}/{len(rows)}\n")
        fh.write(f"DFT VBM fallback cases: {fall_v}/{len(rows)}\n")
        fh.write(f"DFT CBM fallback cases: {fall_c}/{len(rows)}\n")
        fh.write(
            "flagged_cases.csv contains every structure whose strict-rule gap "
            "was outside the plausibility window or remained unresolved.\n"
        )

    print("\n" + "=" * 78)
    print("Comparison complete")
    print("=" * 78)
    print(f"Per-case CSV : {args.output / 'frontier_comparison.csv'}")
    print(f"Summary CSV  : {args.output / 'summary_by_split.csv'}")
    print(f"Report       : {report}")
    if failures:
        print(f"Failures     : {args.output / 'failures.csv'}")
    if flagged_rows:
        print(f"Flagged      : {args.output / 'flagged_cases.csv'}")
    if args.make_plots:
        print(f"Plots        : {args.output / 'edge_error_comparison.png'}")
        print(f"               {args.output / 'gap_error_comparison.png'}")


if __name__ == "__main__":
    main()
