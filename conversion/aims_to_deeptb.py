"""
Convert one FHI-aims band-structure calculation into a one-frame DeePTB dataset.

Usage
-----
Recommended adaptive non-core window::

    python scripts/aims_to_deeptb.py CASE_DIR -o OUTPUT_DIR \
        --band-policy adaptive-factor --band-factor 2.0

Keep all available non-core bands::

    python scripts/aims_to_deeptb.py CASE_DIR -o OUTPUT_DIR \
        --band-policy all-noncore

Keep a fixed number of non-core bands::

    python scripts/aims_to_deeptb.py CASE_DIR -o OUTPUT_DIR \
        --band-policy fixed-noncore --noncore-bands N

Input
-----
CASE_DIR must contain ``geometry.in`` and the selected FHI-aims band files
(``band1*.out`` by default).

Output
------
The output directory contains ``xdat.traj``, ``kpoints.npy``,
``eigenvalues.npy``, ``info.json``, and ``conversion_report.json``.

Band handling
-------------
For H/B/C/N/O, the converter can infer frozen 1s core bands automatically:
H contributes zero; B/C/N/O contribute one spatial core band per atom.
``adaptive-factor`` removes these core bands and retains approximately
``F * N_full`` consecutive non-core bands, where ``N_full`` is the initial
contiguous fully occupied non-core manifold.  The selected window is expanded
when necessary so that no band carrying non-negligible occupation is truncated.

DFT eigenvalues are stored in the original FHI-aims energy gauge; no VBM or
Fermi-level shift is applied.  Spin channels are not merged automatically.
Use ``batch_aims_to_deeptb.py`` to convert a directory tree of calculations.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from ase.io import read
from ase.io.trajectory import Trajectory


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"

LONGFORM_RE = re.compile(
    rf"""
    k\s*=\s*(?P<kidx>\d+).*?
    kvec\s*=\s*\(\s*
      (?P<kx>{FLOAT_RE})\s+
      (?P<ky>{FLOAT_RE})\s+
      (?P<kz>{FLOAT_RE})\s*
    \).*?
    band\s*=\s*(?P<band>\d+).*?
    occ\s*=\s*(?P<occ>{FLOAT_RE}).*?
    E\s*=\s*(?P<energy>{FLOAT_RE})
    """,
    re.VERBOSE,
)

# Number of frozen 1s spatial core orbitals per atom for the chemistry
# currently relevant to the hBN project.
#
# B  : 1s2 | 2s2 2p1
# C  : 1s2 | 2s2 2p2
# N  : 1s2 | 2s2 2p3
# O  : 1s2 | 2s2 2p4
#
# H is included as a useful zero-core special case.
CORE_BANDS_PER_ATOM: Dict[str, int] = {
    "H": 0,
    "B": 1,
    "C": 1,
    "N": 1,
    "O": 1,
}


def _f(x: str) -> float:
    """Parse ordinary or Fortran D-exponent floats."""
    return float(x.replace("D", "E").replace("d", "e"))


def natural_key(path: str | Path):
    """Natural-sort filenames, e.g. band1002 before band1010."""
    s = Path(path).name
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def parse_numeric_band_file(
    path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse the standard FHI-aims numeric band*.out format:

        ik  kx ky kz  occ_1 E_1  occ_2 E_2 ...

    Returns
    -------
    kpoints : (nk, 3)
    occupations : (nk, nbands)
    eigenvalues : (nk, nbands)
    """
    rows = []

    with path.open("r", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
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

            rows.append((lineno, vals))

    if not rows:
        raise ValueError(f"{path}: no standard numeric band rows found.")

    nbands_set = {(len(vals) - 4) // 2 for _, vals in rows}
    if len(nbands_set) != 1:
        raise ValueError(
            f"{path}: inconsistent number of bands across rows: "
            f"{sorted(nbands_set)}"
        )

    nbands = nbands_set.pop()

    kpoints, occupations, eigenvalues = [], [], []

    for _, vals in rows:
        kpoints.append(vals[1:4])
        pairs = np.asarray(vals[4:], dtype=float).reshape(nbands, 2)
        occupations.append(pairs[:, 0])
        eigenvalues.append(pairs[:, 1])

    return (
        np.asarray(kpoints, dtype=float),
        np.asarray(occupations, dtype=float),
        np.asarray(eigenvalues, dtype=float),
    )


def parse_longform_band_file(
    path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse diagnostic long-form rows such as:

        k= 1 kvec=(...) band= 51 occ=... E=...

    Every k-point must contain the same consecutive band labels.
    """
    data = {}

    with path.open("r", errors="replace") as f:
        for line in f:
            m = LONGFORM_RE.search(line)
            if not m:
                continue

            kidx = int(m.group("kidx"))
            kvec = np.array(
                [
                    _f(m.group("kx")),
                    _f(m.group("ky")),
                    _f(m.group("kz")),
                ],
                dtype=float,
            )
            band = int(m.group("band"))
            occupation = _f(m.group("occ"))
            energy = _f(m.group("energy"))

            item = data.setdefault(kidx, {"kvec": kvec, "bands": {}})

            if not np.allclose(item["kvec"], kvec, atol=1e-10, rtol=0):
                raise ValueError(f"{path}: inconsistent k-vector for k={kidx}.")

            item["bands"][band] = (occupation, energy)

    if not data:
        raise ValueError(f"{path}: no long-form band rows found.")

    k_indices = sorted(data)
    band_sets = [set(data[k]["bands"]) for k in k_indices]
    common = set.intersection(*band_sets)
    union = set.union(*band_sets)

    if common != union:
        missing = {
            k: sorted(union - set(data[k]["bands"]))
            for k in k_indices
            if set(data[k]["bands"]) != union
        }
        raise ValueError(
            f"{path}: not every k-point contains the same bands. Missing: {missing}"
        )

    bands = sorted(union)

    if bands != list(range(bands[0], bands[-1] + 1)):
        raise ValueError(f"{path}: non-consecutive band labels: {bands}")

    kpoints = np.vstack([data[k]["kvec"] for k in k_indices])
    occupations = np.array(
        [[data[k]["bands"][b][0] for b in bands] for k in k_indices],
        dtype=float,
    )
    eigenvalues = np.array(
        [[data[k]["bands"][b][1] for b in bands] for k in k_indices],
        dtype=float,
    )

    return kpoints, occupations, eigenvalues


def parse_band_file(
    path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Try standard numeric format first, then the diagnostic long form."""
    try:
        kp, occ, eig = parse_numeric_band_file(path)
        return kp, occ, eig, "numeric"

    except ValueError as numeric_err:
        try:
            kp, occ, eig = parse_longform_band_file(path)
            return kp, occ, eig, "longform"

        except ValueError as long_err:
            raise ValueError(
                f"Could not parse {path}.\n"
                f"Numeric parser: {numeric_err}\n"
                f"Long-form parser: {long_err}"
            )


def combine_segments(
    files: List[Path],
    dedupe_boundaries: bool = True,
    atol: float = 1e-8,
):
    """
    Parse and concatenate ordered band-path segments.

    By default, a repeated high-symmetry k-point at the join between adjacent
    segments is retained only once.
    """
    all_k, all_occ, all_eig = [], [], []
    formats, dropped = {}, []

    nbands = None
    last_k = None

    for path in files:
        kp, occ, eig, fmt = parse_band_file(path)
        formats[path.name] = fmt

        if nbands is None:
            nbands = eig.shape[1]
        elif eig.shape[1] != nbands:
            raise ValueError(
                f"Band-count mismatch: {path.name} has {eig.shape[1]} bands, "
                f"expected {nbands}."
            )

        start = 0

        if (
            dedupe_boundaries
            and last_k is not None
            and len(kp) > 0
            and np.allclose(last_k, kp[0], atol=atol, rtol=0)
        ):
            start = 1
            dropped.append(
                {
                    "file": path.name,
                    "kpoint": kp[0].tolist(),
                    "reason": "duplicate segment boundary",
                }
            )

        if start < len(kp):
            all_k.append(kp[start:])
            all_occ.append(occ[start:])
            all_eig.append(eig[start:])
            last_k = kp[-1].copy()

    if not all_k:
        raise ValueError("No k-points remain after combining band segments.")

    return (
        np.concatenate(all_k, axis=0),
        np.concatenate(all_occ, axis=0),
        np.concatenate(all_eig, axis=0),
        formats,
        dropped,
    )


def infer_core_band_count(symbols: List[str]) -> Tuple[Optional[int], List[str]]:
    """
    Infer the number of 1s core bands for the supported light-element chemistry.

    Returns
    -------
    core_count : int or None
        None means automatic inference is not safe for the present elements.
    unsupported : list[str]
        Elements for which this converter has no core-band rule.
    """
    unsupported = sorted(set(symbols) - set(CORE_BANDS_PER_ATOM))

    if unsupported:
        return None, unsupported

    return sum(CORE_BANDS_PER_ATOM[s] for s in symbols), []



def analyze_electronic_structure(
    kpoints: np.ndarray,
    occupations: np.ndarray,
    eigenvalues: np.ndarray,
    inferred_core_bands: Optional[int],
    occ_threshold: float,
) -> Dict:
    """
    Analyze the occupation frontier without interpreting host VBM/CBM.

    DeePTB eigenvalue training does not need semantic labels such as
    "host VBM", "host CBM", or "host band gap". For defect calculations,
    occupation alone cannot reliably distinguish host band edges from
    defect/frontier levels, so the converter reports neutral occupation-based quantities.
    """
    if occupations.shape != eigenvalues.shape:
        raise ValueError(
            "Occupation and eigenvalue arrays must have identical shapes."
        )

    nk, nbands = eigenvalues.shape
    if nk != len(kpoints):
        raise ValueError("k-point count does not match eigenvalue rows.")

    occupied_state_mask = occupations > occ_threshold
    occupied_by_band = occupied_state_mask.any(axis=0)

    occupied_band_indices = np.flatnonzero(occupied_by_band)
    if len(occupied_band_indices) == 0:
        raise ValueError(
            f"No occupied states found with occ_threshold={occ_threshold}."
        )

    last_occ_idx = int(occupied_band_indices[-1])
    first_empty_idx = last_occ_idx + 1 if last_occ_idx + 1 < nbands else None

    holes_below_boundary = [
        int(i + 1)
        for i in range(last_occ_idx + 1)
        if not occupied_by_band[i]
    ]

    nominal_full_occ = float(np.max(occupations))
    fractional_state_mask = (
        (occupations > occ_threshold)
        & (occupations < nominal_full_occ - occ_threshold)
    )
    fractional_band_indices = (
        np.flatnonzero(fractional_state_mask.any(axis=0)) + 1
    ).tolist()

    fully_occupied_by_band = np.all(
        occupations >= nominal_full_occ - occ_threshold, axis=0
    )

    # Adaptive-window reference: count the initial CONTIGUOUS block of
    # fully occupied non-core bands. This matches the desired supercell-
    # scalable convention better than using a fixed absolute band count.
    contiguous_noncore_full_count = None
    last_contiguous_noncore_full_idx = None
    if inferred_core_bands is not None:
        first_noncore_idx = inferred_core_bands
        contiguous_noncore_full_count = 0
        for idx in range(first_noncore_idx, nbands):
            if fully_occupied_by_band[idx]:
                contiguous_noncore_full_count += 1
                last_contiguous_noncore_full_idx = idx
            else:
                break

    # Highest-energy state carrying non-negligible occupation.
    occ_energy_grid = np.where(occupied_state_mask, eigenvalues, -np.inf)
    occ_flat = int(np.argmax(occ_energy_grid))
    occ_k_idx, occ_band_idx = np.unravel_index(
        occ_flat, occ_energy_grid.shape
    )
    highest_occ = float(eigenvalues[occ_k_idx, occ_band_idx])

    lowest_empty = None
    empty_k_idx = None
    empty_band_idx = None
    frontier_sep = None
    frontier_k_relation = None
    empty_bands_available = 0

    if first_empty_idx is not None:
        empty_block = eigenvalues[:, first_empty_idx:]
        empty_flat_local = int(np.argmin(empty_block))
        empty_k_idx, empty_band_local = np.unravel_index(
            empty_flat_local, empty_block.shape
        )
        empty_band_idx = first_empty_idx + int(empty_band_local)
        lowest_empty = float(eigenvalues[empty_k_idx, empty_band_idx])
        frontier_sep = float(lowest_empty - highest_occ)
        empty_bands_available = nbands - first_empty_idx

        same_k = np.allclose(
            kpoints[occ_k_idx],
            kpoints[empty_k_idx],
            atol=1e-8,
            rtol=0,
        )
        frontier_k_relation = "same-k" if same_k else "different-k"

    core_summary = {
        "inferred_core_band_count": inferred_core_bands,
        "core_band_range_1based_inclusive": (
            [1, int(inferred_core_bands)]
            if inferred_core_bands is not None and inferred_core_bands > 0
            else None
        ),
        "core_noncore_energy_separation_eV": None,
        "core_energy_max_eV": None,
        "first_noncore_energy_min_eV": None,
    }

    if (
        inferred_core_bands is not None
        and 0 < inferred_core_bands < nbands
    ):
        ncore = inferred_core_bands
        core_max = float(np.max(eigenvalues[:, :ncore]))
        first_noncore_min = float(np.min(eigenvalues[:, ncore]))
        core_summary.update(
            {
                "core_energy_max_eV": core_max,
                "first_noncore_energy_min_eV": first_noncore_min,
                "core_noncore_energy_separation_eV": (
                    first_noncore_min - core_max
                ),
            }
        )

    noncore_with_any_occ = None
    noncore_fully_occ = None
    if inferred_core_bands is not None:
        noncore_with_any_occ = int(
            np.sum(occupied_by_band[inferred_core_bands:])
        )
        noncore_fully_occ = int(
            np.sum(fully_occupied_by_band[inferred_core_bands:])
        )

    return {
        "interpretation": (
            "occupation-based frontier diagnostics only; "
            "host VBM/CBM are not inferred"
        ),
        "host_band_edges_inferred": False,
        "occupation_threshold": float(occ_threshold),
        "nominal_full_occupation": nominal_full_occ,
        "occupation_contiguous_through_last_band_with_occupation": (
            len(holes_below_boundary) == 0
        ),
        "unexpected_empty_bands_below_boundary_1based": holes_below_boundary,
        "fractionally_occupied_bands_1based": fractional_band_indices,
        "bands_with_any_occupation_total": int(np.sum(occupied_by_band)),
        "fully_occupied_band_count_total": int(
            np.sum(fully_occupied_by_band)
        ),
        "noncore_bands_with_any_occupation": noncore_with_any_occ,
        "noncore_fully_occupied_band_count": noncore_fully_occ,
        "contiguous_noncore_fully_occupied_band_count": (
            int(contiguous_noncore_full_count)
            if contiguous_noncore_full_count is not None else None
        ),
        "last_contiguous_noncore_fully_occupied_band_1based": (
            int(last_contiguous_noncore_full_idx + 1)
            if last_contiguous_noncore_full_idx is not None else None
        ),
        "last_band_with_any_occupation_1based": int(last_occ_idx + 1),
        "first_completely_empty_band_above_boundary_1based": (
            int(first_empty_idx + 1)
            if first_empty_idx is not None else None
        ),
        "empty_bands_available_above_boundary": int(empty_bands_available),
        "highest_occupied_state_energy_eV": highest_occ,
        "highest_occupied_state_band_1based": int(occ_band_idx + 1),
        "highest_occupied_state_kpoint_index_1based": int(occ_k_idx + 1),
        "highest_occupied_state_kpoint": kpoints[occ_k_idx].tolist(),
        "lowest_empty_state_energy_eV": lowest_empty,
        "lowest_empty_state_band_1based": (
            int(empty_band_idx + 1)
            if empty_band_idx is not None else None
        ),
        "lowest_empty_state_kpoint_index_1based": (
            int(empty_k_idx + 1)
            if empty_k_idx is not None else None
        ),
        "lowest_empty_state_kpoint": (
            kpoints[empty_k_idx].tolist()
            if empty_k_idx is not None else None
        ),
        "frontier_state_separation_eV": frontier_sep,
        "frontier_k_relation": frontier_k_relation,
        "core_analysis": core_summary,
    }



def choose_band_window(
    total_bands: int,
    electronic: Dict,
    inferred_core_bands: Optional[int],
    strip_core: bool,
    conduction_bands: Optional[int],
    noncore_bands: Optional[int],
    adaptive_noncore: bool,
    band_policy: Optional[str],
    band_factor: Optional[float],
    first_band: Optional[int],
    last_band: Optional[int],
) -> Tuple[int, int, str, List[str], Dict]:
    """
    Choose original FHI-aims band numbers to store.

    Recommended policies
    -----------------------
    adaptive-factor
        Strip inferred core bands and request

            ceil(F * N_full)

        consecutive non-core bands, where N_full is the initial contiguous
        fully occupied non-core manifold.  The window is enlarged if necessary
        so that every band carrying non-negligible occupation is included.

    all-noncore
        Strip inferred core bands and keep every remaining FHI-aims band.

    fixed-noncore
        Strip inferred core bands and keep exactly --noncore-bands N bands.

    The --adaptive-noncore flag is retained as a compatibility alias for
    adaptive-factor with F=2.0.

    Returned band numbers are 1-based and inclusive.
    """
    allowed_policies = {None, "adaptive-factor", "all-noncore", "fixed-noncore"}
    if band_policy not in allowed_policies:
        raise ValueError(
            f"Unknown band policy {band_policy!r}; expected adaptive-factor, "
            "all-noncore, or fixed-noncore."
        )

    if adaptive_noncore and band_policy is not None:
        raise ValueError(
            "--adaptive-noncore is a compatibility alias and cannot be combined with "
            "--band-policy. Use --band-policy adaptive-factor --band-factor 2.0."
        )

    # Translate the compatibility alias into the explicit policy representation.
    effective_policy = band_policy
    effective_factor = band_factor
    if adaptive_noncore:
        effective_policy = "adaptive-factor"
        effective_factor = 2.0

    # ------------------------------------------------------------------
    # Explicit policy modes
    # ------------------------------------------------------------------
    if effective_policy is not None:
        conflicts = []
        if conduction_bands is not None:
            conflicts.append("--conduction-bands")
        if first_band is not None:
            conflicts.append("--first-band")
        if last_band is not None:
            conflicts.append("--last-band")

        if effective_policy != "fixed-noncore" and noncore_bands is not None:
            conflicts.append("--noncore-bands")

        if conflicts:
            raise ValueError(
                f"--band-policy {effective_policy} cannot be combined with "
                + ", ".join(conflicts) + "."
            )

        if inferred_core_bands is None:
            raise ValueError(
                f"--band-policy {effective_policy} requires automatic "
                "core-band inference."
            )

        first_noncore = inferred_core_bands + 1
        available_noncore = total_bands - inferred_core_bands

        details = {
            "band_policy_effective": effective_policy,
            "band_factor": None,
            "available_noncore_bands": int(available_noncore),
            "contiguous_fully_occupied_noncore_count": electronic.get(
                "contiguous_noncore_fully_occupied_band_count"
            ),
            "factor_target_count": None,
            "frontier_span_noncore": None,
            "requested_count_before_available_cap": None,
            "requested_fixed_noncore_bands": (
                int(noncore_bands) if noncore_bands is not None else None
            ),
        }

        if effective_policy == "adaptive-factor":
            factor = 2.0 if effective_factor is None else float(effective_factor)
            if not np.isfinite(factor) or factor < 1.0:
                raise ValueError(
                    "--band-factor must be a finite number >= 1.0 for "
                    "--band-policy adaptive-factor."
                )

            n_full = electronic.get(
                "contiguous_noncore_fully_occupied_band_count"
            )
            last_full = electronic.get(
                "last_contiguous_noncore_fully_occupied_band_1based"
            )
            if n_full is None or n_full <= 0 or last_full is None:
                raise ValueError(
                    "Could not identify a contiguous fully occupied non-core "
                    "manifold for --band-policy adaptive-factor."
                )

            factor_target_count = int(math.ceil(factor * n_full))
            last_with_occ = electronic[
                "last_band_with_any_occupation_1based"
            ]
            frontier_span = max(0, last_with_occ - first_noncore + 1)

            # Frontier safeguard: never truncate a band that still carries
            # non-negligible occupation, even when factor < 2.
            requested_count = max(factor_target_count, frontier_span)
            selected_count = min(requested_count, available_noncore)

            selected_first = first_noncore
            selected_last = inferred_core_bands + selected_count

            details.update({
                "band_factor": float(factor),
                "factor_target_count": int(factor_target_count),
                "frontier_span_noncore": int(frontier_span),
                "requested_count_before_available_cap": int(requested_count),
            })

            reasons = [
                f"stripped {inferred_core_bands} inferred 1s core bands",
                (
                    "contiguous fully occupied non-core manifold: "
                    f"{first_noncore}..{last_full} ({n_full} bands)"
                ),
                (
                    f"adaptive factor target: ceil({factor:g} x {n_full}) "
                    f"= {factor_target_count}"
                ),
            ]
            if frontier_span > factor_target_count:
                reasons.append(
                    f"expanded to {frontier_span} bands to include every band "
                    "with non-negligible occupation"
                )
            if requested_count > available_noncore:
                reasons.append(
                    f"capped at all {available_noncore} available non-core bands"
                )
            else:
                reasons.append(
                    f"selected {selected_count} of {available_noncore} "
                    "available non-core bands"
                )
            if strip_core:
                reasons.append(
                    "--strip-core is redundant with --band-policy adaptive-factor"
                )

            return (
                selected_first,
                selected_last,
                "adaptive_factor_fully_occupied_noncore",
                reasons,
                details,
            )

        if effective_policy == "all-noncore":
            if effective_factor is not None:
                raise ValueError(
                    "--band-factor is only valid with "
                    "--band-policy adaptive-factor."
                )

            selected_first = first_noncore
            selected_last = total_bands
            selected_count = available_noncore
            details["requested_count_before_available_cap"] = int(
                available_noncore
            )

            reasons = [
                f"stripped {inferred_core_bands} inferred 1s core bands",
                f"kept all {available_noncore} available non-core bands",
            ]
            if strip_core:
                reasons.append(
                    "--strip-core is redundant with --band-policy all-noncore"
                )

            return (
                selected_first,
                selected_last,
                "all_available_noncore",
                reasons,
                details,
            )

        if effective_policy == "fixed-noncore":
            if effective_factor is not None:
                raise ValueError(
                    "--band-factor is only valid with "
                    "--band-policy adaptive-factor."
                )
            if noncore_bands is None:
                raise ValueError(
                    "--band-policy fixed-noncore requires --noncore-bands N."
                )
            if noncore_bands <= 0:
                raise ValueError("--noncore-bands must be a positive integer.")
            if noncore_bands > available_noncore:
                raise ValueError(
                    f"Requested {noncore_bands} non-core bands, but only "
                    f"{available_noncore} are available after stripping "
                    f"{inferred_core_bands} core bands."
                )

            selected_first = first_noncore
            selected_last = inferred_core_bands + noncore_bands
            details["requested_count_before_available_cap"] = int(noncore_bands)

            reasons = [
                f"stripped {inferred_core_bands} inferred 1s core bands",
                f"kept exactly {noncore_bands} consecutive non-core bands",
            ]
            if strip_core:
                reasons.append(
                    "--strip-core is redundant with --band-policy fixed-noncore"
                )

            return (
                selected_first,
                selected_last,
                "fixed_noncore_count",
                reasons,
                details,
            )

    # ------------------------------------------------------------------
    # Compatibility fixed-count mode without --band-policy.
    # ------------------------------------------------------------------
    if band_factor is not None:
        raise ValueError(
            "--band-factor requires --band-policy adaptive-factor."
        )

    if noncore_bands is not None:
        conflicts = []
        if conduction_bands is not None:
            conflicts.append("--conduction-bands")
        if first_band is not None:
            conflicts.append("--first-band")
        if last_band is not None:
            conflicts.append("--last-band")
        if conflicts:
            raise ValueError(
                "--noncore-bands cannot be combined with "
                + ", ".join(conflicts) + "."
            )
        if noncore_bands <= 0:
            raise ValueError("--noncore-bands must be a positive integer.")
        if inferred_core_bands is None:
            raise ValueError(
                "--noncore-bands requires automatic core-band inference, "
                "but one or more elements are unsupported."
            )

        selected_first = inferred_core_bands + 1
        selected_last = inferred_core_bands + noncore_bands
        available = total_bands - inferred_core_bands

        if selected_last > total_bands:
            raise ValueError(
                f"Requested {noncore_bands} non-core bands, but only "
                f"{available} are available after stripping "
                f"{inferred_core_bands} core bands."
            )

        reasons = [
            f"stripped {inferred_core_bands} inferred 1s core bands",
            f"kept exactly {noncore_bands} consecutive non-core bands",
        ]
        if strip_core:
            reasons.append(
                "--strip-core was also supplied; it is redundant in "
                "--noncore-bands mode"
            )

        details = {
            "band_policy_effective": "fixed-noncore",
            "band_factor": None,
            "available_noncore_bands": int(available),
            "contiguous_fully_occupied_noncore_count": electronic.get(
                "contiguous_noncore_fully_occupied_band_count"
            ),
            "factor_target_count": None,
            "frontier_span_noncore": None,
            "requested_count_before_available_cap": int(noncore_bands),
            "requested_fixed_noncore_bands": int(noncore_bands),
        }
        return (
            selected_first,
            selected_last,
            "fixed_noncore_count",
            reasons,
            details,
        )

    # ------------------------------------------------------------------
    # Compatibility/manual selection modes.
    # ------------------------------------------------------------------
    if strip_core and first_band is not None:
        raise ValueError(
            "--strip-core and --first-band are both start-band selectors. "
            "Use only one."
        )

    if conduction_bands is not None and last_band is not None:
        raise ValueError(
            "--conduction-bands and --last-band are both end-band selectors. "
            "Use only one."
        )

    if conduction_bands is not None and conduction_bands < 0:
        raise ValueError("--conduction-bands must be >= 0.")

    reasons = []

    if first_band is not None:
        selected_first = int(first_band)
        reasons.append(f"manual first band = {selected_first}")
    elif strip_core:
        if inferred_core_bands is None:
            raise ValueError(
                "--strip-core requested, but automatic core-band inference "
                "is unavailable for one or more chemical elements."
            )
        selected_first = inferred_core_bands + 1
        reasons.append(
            f"stripped {inferred_core_bands} inferred 1s core bands"
        )
    else:
        selected_first = 1
        reasons.append("kept core bands (no --strip-core)")

    if last_band is not None:
        selected_last = int(last_band)
        reasons.append(f"manual last band = {selected_last}")

    elif conduction_bands is not None:
        first_empty = electronic[
            "first_completely_empty_band_above_boundary_1based"
        ]

        if first_empty is None:
            if conduction_bands == 0:
                selected_last = electronic[
                    "last_band_with_any_occupation_1based"
                ]
            else:
                raise ValueError(
                    "No completely empty band was found above the occupation "
                    f"boundary, so --conduction-bands {conduction_bands} "
                    "cannot be satisfied."
                )
        else:
            available = electronic["empty_bands_available_above_boundary"]
            if conduction_bands > available:
                raise ValueError(
                    f"Requested {conduction_bands} bands above the occupation "
                    f"boundary, but only {available} are available."
                )

            if conduction_bands == 0:
                selected_last = electronic[
                    "last_band_with_any_occupation_1based"
                ]
            else:
                selected_last = first_empty + conduction_bands - 1

        reasons.append(
            "compatibility occupation-based mode: kept all bands through the last "
            f"band with any occupation plus {conduction_bands} empty bands"
        )
        reasons.append(
            "these empty bands are not interpreted as host conduction bands"
        )

    else:
        selected_last = total_bands
        reasons.append("kept all available upper bands")

    if not (1 <= selected_first <= selected_last <= total_bands):
        raise ValueError(
            f"Invalid selected band window {selected_first}:{selected_last}; "
            f"parsed band range is 1:{total_bands}."
        )

    available_noncore = (
        total_bands - inferred_core_bands
        if inferred_core_bands is not None else None
    )
    details = {
        "band_policy_effective": "legacy-or-manual",
        "band_factor": None,
        "available_noncore_bands": available_noncore,
        "contiguous_fully_occupied_noncore_count": electronic.get(
            "contiguous_noncore_fully_occupied_band_count"
        ),
        "factor_target_count": None,
        "frontier_span_noncore": None,
        "requested_count_before_available_cap": None,
        "requested_fixed_noncore_bands": None,
    }
    return selected_first, selected_last, "legacy_or_manual", reasons, details

def write_deeptb_dataset(
    case_dir: Path,
    out_dir: Path,
    geometry_name: str,
    band_glob: str,
    strip_core: bool,
    conduction_bands: Optional[int],
    noncore_bands: Optional[int],
    adaptive_noncore: bool,
    band_policy: Optional[str],
    band_factor: Optional[float],
    first_band: Optional[int],
    last_band: Optional[int],
    occ_threshold: float,
    core_gap_warning: float,
    keep_boundary_duplicates: bool,
    overwrite: bool,
):
    case_dir = case_dir.resolve()
    out_dir = out_dir.resolve()

    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{out_dir} already exists and is not empty. "
            "Use --overwrite if intended."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    geometry_path = case_dir / geometry_name
    if not geometry_path.exists():
        raise FileNotFoundError(f"Geometry not found: {geometry_path}")

    # ASE currently supports FHI-aims geometry.in directly.
    atoms = read(str(geometry_path), format="aims")

    # A geometry with lattice vectors is periodic. Preserve ASE's PBC when
    # available; for the present hBN calculations this is normally all True.
    if not np.any(atoms.pbc) and np.linalg.det(np.asarray(atoms.cell)) != 0:
        atoms.pbc = [True, True, True]

    symbols = atoms.get_chemical_symbols()
    natoms = len(atoms)

    inferred_core_bands, unsupported_core_elements = infer_core_band_count(
        symbols
    )

    band_paths = [
        Path(p) for p in glob.glob(str(case_dir / band_glob))
    ]
    band_paths.sort(key=natural_key)

    if not band_paths:
        raise FileNotFoundError(
            f"No band files match {case_dir / band_glob}"
        )

    # Common FHI-aims convention: band2xxx may be the second collinear spin
    # channel. Spin channels are intentionally not merged automatically.
    possible_spin2 = sorted(
        [Path(p).name for p in glob.glob(str(case_dir / "band2*.out"))],
        key=natural_key,
    )

    (
        kpoints,
        occupations,
        eigenvalues,
        formats,
        dropped,
    ) = combine_segments(
        band_paths,
        dedupe_boundaries=not keep_boundary_duplicates,
    )

    total_bands = eigenvalues.shape[1]

    if inferred_core_bands is not None and inferred_core_bands >= total_bands:
        raise ValueError(
            f"Inferred {inferred_core_bands} core bands from the geometry, "
            f"but only {total_bands} bands were parsed."
        )

    electronic = analyze_electronic_structure(
        kpoints=kpoints,
        occupations=occupations,
        eigenvalues=eigenvalues,
        inferred_core_bands=inferred_core_bands,
        occ_threshold=occ_threshold,
    )

    (
        selected_first,
        selected_last,
        selection_mode,
        selection_reasons,
        selection_details,
    ) = choose_band_window(
        total_bands=total_bands,
        electronic=electronic,
        inferred_core_bands=inferred_core_bands,
        strip_core=strip_core,
        conduction_bands=conduction_bands,
        noncore_bands=noncore_bands,
        adaptive_noncore=adaptive_noncore,
        band_policy=band_policy,
        band_factor=band_factor,
        first_band=first_band,
        last_band=last_band,
    )

    lo = selected_first - 1
    hi = selected_last

    eig_sel = eigenvalues[:, lo:hi]
    occ_sel = occupations[:, lo:hi]

    # DeePTB-SK eigenvalue layout:
    # [nframes, nkpoints, nbands]
    eig_out = eig_sel[np.newaxis, :, :]

    np.save(out_dir / "kpoints.npy", kpoints)
    np.save(out_dir / "eigenvalues.npy", eig_out)

    with Trajectory(str(out_dir / "xdat.traj"), "w") as traj:
        traj.write(atoms)

    info = {
        "nframes": 1,
        "natoms": natoms,
        "pos_type": "ase",
        "pbc": [bool(x) for x in atoms.pbc],
        "bandinfo": {
            # We physically slice the selected DFT bands into eigenvalues.npy,
            # so DeePTB should fit all stored bands.
            "band_min": 0,
            "band_max": int(eig_sel.shape[1]),
            "emin": None,
            "emax": None,
        },
    }

    with (out_dir / "info.json").open("w") as f:
        json.dump(info, f, indent=2)

    core_analysis = electronic["core_analysis"]
    core_gap = core_analysis["core_noncore_energy_separation_eV"]
    core_gap_warning_triggered = (
        core_gap is not None and core_gap < core_gap_warning
    )

    report = {
        "converter_version": 5,
        "source_case": str(case_dir),
        "geometry": geometry_path.name,
        "natoms": natoms,
        "chemical_symbols": symbols,
        "element_counts": {
            symbol: symbols.count(symbol)
            for symbol in sorted(set(symbols))
        },
        "core_inference": {
            "rule": (
                "H:0; B/C/N/O: one 1s core band per atom"
            ),
            "supported": inferred_core_bands is not None,
            "unsupported_elements": unsupported_core_elements,
            "inferred_core_band_count": inferred_core_bands,
            "core_band_range_1based_inclusive": (
                [1, inferred_core_bands]
                if inferred_core_bands is not None
                and inferred_core_bands > 0
                else None
            ),
            "core_noncore_energy_separation_eV": core_gap,
            "core_gap_warning_threshold_eV": core_gap_warning,
            "core_gap_warning_triggered": core_gap_warning_triggered,
        },
        "frontier_diagnostics": {
            key: value
            for key, value in electronic.items()
            if key != "core_analysis"
        },
        "host_band_edge_interpretation": {
            "performed": False,
            "reason": (
                "DeePTB eigenvalue training does not require semantic host "
                "VBM/CBM labels, and occupation alone cannot reliably "
                "distinguish host band edges from defect/frontier states."
            ),
        },
        "band_glob": band_glob,
        "band_files": [p.name for p in band_paths],
        "band_file_formats": formats,
        "possible_second_spin_channel_files": possible_spin2,
        "boundary_duplicates_dropped": dropped,
        "raw_shape_per_frame": list(eigenvalues.shape),
        "selection": {
            "mode": selection_mode,
            "strip_core": strip_core,
            "requested_conduction_bands_legacy": conduction_bands,
            "requested_noncore_bands": noncore_bands,
            "adaptive_noncore_legacy_alias": adaptive_noncore,
            "band_policy_requested": band_policy,
            "band_policy_effective": selection_details.get("band_policy_effective"),
            "band_factor": selection_details.get("band_factor"),
            "available_noncore_bands": selection_details.get("available_noncore_bands"),
            "contiguous_fully_occupied_noncore_count": selection_details.get(
                "contiguous_fully_occupied_noncore_count"
            ),
            "factor_target_count": selection_details.get("factor_target_count"),
            "frontier_span_noncore": selection_details.get("frontier_span_noncore"),
            "requested_count_before_available_cap": selection_details.get(
                "requested_count_before_available_cap"
            ),
            "manual_first_band": first_band,
            "manual_last_band": last_band,
            "reasons": selection_reasons,
            "selected_original_band_range_1based_inclusive": [
                selected_first,
                selected_last,
            ],
            "selected_band_count": int(eig_sel.shape[1]),
        },
        "output_kpoints_shape": list(kpoints.shape),
        "output_eigenvalues_shape": list(eig_out.shape),
        "selected_energy_min_eV": float(np.min(eig_sel)),
        "selected_energy_max_eV": float(np.max(eig_sel)),
        "selected_occupation_min": float(np.min(occ_sel)),
        "selected_occupation_max": float(np.max(occ_sel)),
        "notes": [
            "DFT eigenvalues are preserved without VBM/Fermi shifting.",
            (
                "DeePTB eigvals loss can perform its own relative energy "
                "alignment during fitting."
            ),
            (
                "No host VBM/CBM/band-gap interpretation is required or "
                "attempted by this converter."
            ),
            (
                "Frontier diagnostics are occupation-based only and may "
                "correspond to defect states."
            ),
            (
                "If band2*.out files are present, inspect spin handling "
                "before combining spin channels."
            ),
            (
                "Automatic core stripping is chemistry-specific and is "
                "currently intended for the H/B/C/N/O hBN-defect workflow."
            ),
            (
                "band-policy metadata is recorded in selection so datasets "
                "generated with different supervised spectral windows can be "
                "audited and compared directly."
            ),
        ],
    }

    with (out_dir / "conversion_report.json").open("w") as f:
        json.dump(report, f, indent=2)

    print("=" * 76)
    print("FHI-aims -> DeePTB-SK conversion complete")
    print("=" * 76)
    print(f"Geometry:              {geometry_path}")
    print(f"Atoms:                 {natoms}")
    print(f"Elements:              {report['element_counts']}")
    print(f"Band files:            {len(band_paths)}")
    print(f"Parsed k-points:       {kpoints.shape[0]}")
    print(f"Parsed bands:          {total_bands}")

    print("\nFrontier diagnostics (NOT host VBM/CBM)")
    print("-" * 76)

    if inferred_core_bands is not None:
        if inferred_core_bands > 0:
            print(
                f"Core bands inferred:    {inferred_core_bands} "
                f"(1..{inferred_core_bands})"
            )
        else:
            print("Core bands inferred:    0")
    else:
        print(
            "Core bands inferred:    unavailable "
            f"(unsupported: {unsupported_core_elements})"
        )

    if core_gap is not None:
        print(f"Core/non-core gap:      {core_gap:.6f} eV")
        if core_gap_warning_triggered:
            print(
                f"WARNING: inferred core separation is below "
                f"{core_gap_warning:.3f} eV; inspect the core assignment."
            )

    print(
        f"Last band with occ.:    "
        f"{electronic['last_band_with_any_occupation_1based']}"
    )
    print(
        f"First empty band:       "
        f"{electronic['first_completely_empty_band_above_boundary_1based']}"
    )
    print(
        f"Non-core bands w/occ.:  "
        f"{electronic['noncore_bands_with_any_occupation']}"
    )
    print(
        f"Non-core fully occ.:    "
        f"{electronic['noncore_fully_occupied_band_count']}"
    )
    print(
        f"Contiguous full block:  "
        f"{electronic['contiguous_noncore_fully_occupied_band_count']} "
        f"(ends at band "
        f"{electronic['last_contiguous_noncore_fully_occupied_band_1based']})"
    )
    print(
        f"Highest occ. energy:    "
        f"{electronic['highest_occupied_state_energy_eV']:.6f} eV "
        f"(band {electronic['highest_occupied_state_band_1based']}, "
        f"k={electronic['highest_occupied_state_kpoint']})"
    )

    if electronic["lowest_empty_state_energy_eV"] is not None:
        print(
            f"Lowest empty energy:   "
            f"{electronic['lowest_empty_state_energy_eV']:.6f} eV "
            f"(band {electronic['lowest_empty_state_band_1based']}, "
            f"k={electronic['lowest_empty_state_kpoint']})"
        )
        print(
            f"Frontier separation:   "
            f"{electronic['frontier_state_separation_eV']:.6f} eV "
            f"({electronic['frontier_k_relation']})"
        )
    else:
        print("Lowest empty energy:    not available")
        print("Frontier separation:    not available")

    if electronic["fractionally_occupied_bands_1based"]:
        print(
            "Fractional occupations: "
            f"bands {electronic['fractionally_occupied_bands_1based']}"
        )

    if not electronic[
        "occupation_contiguous_through_last_band_with_occupation"
    ]:
        print(
            "WARNING: occupation ordering is not contiguous below the "
            "last band carrying occupation."
        )

    print(
        "Host band edges:        NOT INFERRED "
        "(not required for DeePTB training)"
    )

    print("\nOutput selection")
    print("-" * 76)
    print(f"Selection mode:         {selection_mode}")
    print(
        f"Band policy:            "
        f"{selection_details.get('band_policy_effective')}"
    )
    if selection_details.get("band_factor") is not None:
        print(
            f"Band factor:            "
            f"{selection_details.get('band_factor')}"
        )
    if selection_details.get("available_noncore_bands") is not None:
        print(
            f"Available non-core:     "
            f"{selection_details.get('available_noncore_bands')}"
        )
    print(
        f"Selected original bands: "
        f"{selected_first}..{selected_last} "
        f"(1-based, inclusive)"
    )
    print(f"Selected band count:     {eig_sel.shape[1]}")
    print(f"kpoints.npy:             {kpoints.shape}")
    print(f"eigenvalues.npy:         {eig_out.shape}")
    print(
        f"Selected energy range:   "
        f"{eig_sel.min():.6f} .. {eig_sel.max():.6f} eV"
    )
    print(f"Output:                  {out_dir}")

    if dropped:
        print(f"Boundary duplicates:     {len(dropped)} removed")

    if possible_spin2:
        print("\nWARNING: possible second spin-channel files detected:")
        for name in possible_spin2[:10]:
            print(f"  {name}")
        if len(possible_spin2) > 10:
            print(f"  ... and {len(possible_spin2) - 10} more")
        print(
            "Spin channels are not merged automatically. This conversion uses "
            f"only files matching {band_glob!r}."
        )


def build_parser():
    p = argparse.ArgumentParser(
        description=(
            "Convert one FHI-aims band calculation to DeePTB-SK format, "
            "with automatic core/occupation analysis."
        )
    )

    p.add_argument(
        "case_dir",
        type=Path,
        help=(
            "FHI-aims calculation directory containing geometry.in "
            "and band files."
        ),
    )

    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Destination DeePTB dataset folder, "
            "e.g. data_hbn/kpath.0"
        ),
    )

    p.add_argument(
        "--geometry",
        default="geometry.in",
        help=(
            "Geometry filename inside case_dir "
            "(default: geometry.in)."
        ),
    )

    p.add_argument(
        "--band-glob",
        default="band1*.out",
        help=(
            "Band-file glob inside case_dir "
            "(default: band1*.out)."
        ),
    )

    p.add_argument(
        "--band-policy",
        choices=["adaptive-factor", "all-noncore", "fixed-noncore"],
        default=None,
        help=(
            "Band-selection policy. adaptive-factor: strip inferred core "
            "bands and keep ceil(F*Nocc) non-core bands with an occupation-"
            "frontier safeguard; all-noncore: strip core and keep every "
            "available non-core FHI-aims band; fixed-noncore: strip core and "
            "keep exactly --noncore-bands N."
        ),
    )

    p.add_argument(
        "--band-factor",
        type=float,
        default=None,
        metavar="F",
        help=(
            "Multiplier F for --band-policy adaptive-factor. Default is 2.0 "
            "when adaptive-factor is selected. Example: --band-factor 1.5."
        ),
    )

    p.add_argument(
        "--adaptive-noncore",
        action="store_true",
        help=(
            "Compatibility alias equivalent to "
            "--band-policy adaptive-factor --band-factor 2.0. "
            "Do not combine it with --band-policy."
        ),
    )

    p.add_argument(
        "--noncore-bands",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Fixed number of consecutive non-core bands after inferred core "
            "stripping. Prefer: --band-policy fixed-noncore "
            "--noncore-bands N."
        ),
    )

    p.add_argument(
        "--strip-core",
        action="store_true",
        help=(
            "Automatically discard inferred 1s core bands. "
            "For the supported hBN chemistry: B/C/N/O each contribute "
            "one core band; H contributes zero."
        ),
    )

    p.add_argument(
        "--conduction-bands",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Compatibility occupation-based mode: keep all bands through "
            "the last band with any occupation plus N completely empty "
            "bands. These are NOT assumed to be host conduction bands."
        ),
    )

    # Manual band-selection escape hatches.
    p.add_argument(
        "--first-band",
        type=int,
        default=None,
        help=(
            "Manual first original DFT band to store, "
            "1-based inclusive. Cannot be combined with --strip-core."
        ),
    )

    p.add_argument(
        "--last-band",
        type=int,
        default=None,
        help=(
            "Manual last original DFT band to store, "
            "1-based inclusive. Cannot be combined with "
            "--conduction-bands."
        ),
    )

    p.add_argument(
        "--occ-threshold",
        type=float,
        default=1e-3,
        help=(
            "Occupation > threshold is treated as occupied "
            "(default: 1e-3)."
        ),
    )

    p.add_argument(
        "--core-gap-warning",
        type=float,
        default=20.0,
        metavar="EV",
        help=(
            "Warn if the inferred core/non-core energy separation is "
            "smaller than this value in eV (default: 20)."
        ),
    )

    p.add_argument(
        "--keep-boundary-duplicates",
        action="store_true",
        help=(
            "Keep repeated high-symmetry k-points at adjacent "
            "band-path segment boundaries."
        ),
    )

    p.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow writing into an existing non-empty output directory."
        ),
    )

    return p


def main():
    args = build_parser().parse_args()

    if args.occ_threshold < 0:
        print(
            "ERROR: --occ-threshold must be >= 0.",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.core_gap_warning < 0:
        print(
            "ERROR: --core-gap-warning must be >= 0.",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.band_factor is not None and (
        not np.isfinite(args.band_factor) or args.band_factor < 1.0
    ):
        print(
            "ERROR: --band-factor must be a finite number >= 1.0.",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.band_policy == "fixed-noncore" and args.noncore_bands is None:
        print(
            "ERROR: --band-policy fixed-noncore requires --noncore-bands N.",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        write_deeptb_dataset(
            case_dir=args.case_dir,
            out_dir=args.output_dir,
            geometry_name=args.geometry,
            band_glob=args.band_glob,
            strip_core=args.strip_core,
            conduction_bands=args.conduction_bands,
            noncore_bands=args.noncore_bands,
            adaptive_noncore=args.adaptive_noncore,
            band_policy=args.band_policy,
            band_factor=args.band_factor,
            first_band=args.first_band,
            last_band=args.last_band,
            occ_threshold=args.occ_threshold,
            core_gap_warning=args.core_gap_warning,
            keep_boundary_duplicates=args.keep_boundary_duplicates,
            overwrite=args.overwrite,
        )

    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
