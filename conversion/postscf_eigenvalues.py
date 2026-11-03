"""Strict reader for single-channel FHI-aims post-SCF eigenvalue blocks.

Coordinates are reciprocal lattice fractions; row order is preserved. Supports
``state occupation eigenvalue`` rows, with energies in eV. Spin-polarized and
SOC spectra require a separate model and are intentionally rejected.
"""
from pathlib import Path
import re

import numpy as np


def parse_postscf_eigenvalues(path: Path):
    kpoints, blocks = [], []
    current = None
    declared_states = None
    declared_kpoints = None
    with path.open() as handle:
        for lineno, line in enumerate(handle, 1):
            lower = line.lower()
            if "occupation number (dn)" in lower or "spin down" in lower:
                raise ValueError(f"{path}:{lineno}: multiple spin channels are unsupported")
            if "eigenvalue" in lower and ("hartree" in lower or "[ha]" in lower):
                raise ValueError(f"{path}:{lineno}: expected eigenvalues in eV, not Hartree")
            for label in ("states", "k-points"):
                match = re.search(rf"number of {label}\s*:\s*(\d+)", lower)
                if match:
                    if label == "states":
                        declared_states = int(match[1])
                    else:
                        declared_kpoints = int(match[1])
            if "k-point in recip. lattice units:" in lower:
                values = line.split(":", 1)[1].split()
                if len(values) != 3:
                    raise ValueError(f"{path}:{lineno}: expected three fractional coordinates")
                kpoints.append([float(x.replace("D", "E").replace("d", "e")) for x in values])
                current = []
                blocks.append(current)
                continue
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "!")):
                continue
            fields = stripped.split()
            if not re.fullmatch(r"\d+", fields[0]):
                continue
            if current is None:
                raise ValueError(f"{path}:{lineno}: state row before fractional k-point header")
            if len(fields) != 3:
                raise ValueError(f"{path}:{lineno}: expected state, occupation, energy; spin/SOC or unknown layout")
            if int(fields[0]) != len(current) + 1:
                raise ValueError(f"{path}:{lineno}: duplicate, missing or unordered state index")
            current.append([float(x.replace("D", "E").replace("d", "e")) for x in fields[1:]])
    if not blocks or not blocks[0] or any(len(b) != len(blocks[0]) for b in blocks):
        raise ValueError(f"{path}: missing or incomplete k-point/state blocks")
    if declared_states is not None and len(blocks[0]) != declared_states:
        raise ValueError(f"{path}: state count differs from header (truncated file)")
    if declared_kpoints is not None and len(blocks) != declared_kpoints:
        raise ValueError(f"{path}: k-point count differs from header (truncated file)")
    kp, data = np.asarray(kpoints), np.asarray(blocks)
    if not np.isfinite(kp).all() or not np.isfinite(data).all():
        raise ValueError(f"{path}: nonfinite coordinates, occupations or energies")
    if np.any(data[:, :, 0] < -1e-8) or np.any(data[:, :, 0] > 2 + 1e-8):
        raise ValueError(f"{path}: occupations outside [0, 2]")
    # A repeated point modulo reciprocal translations usually means repeated
    # spin blocks or concatenated output. Never silently deduplicate a mesh.
    if len(np.unique(np.round(kp % 1, 10), axis=0)) != len(kp):
        raise ValueError(f"{path}: duplicate reciprocal k-points")
    return kp, data[:, :, 0], data[:, :, 1]


def require_scalar_control(case_dir: Path):
    """Reject known incompatible control flags before interpreting occupations."""
    control = case_dir / "control.in"
    if not control.exists():
        raise ValueError("Dense-grid conversion requires control.in to verify spin/SOC settings")
    for line in control.read_text().splitlines():
        fields = line.split("#", 1)[0].lower().split()
        if fields[:2] == ["spin", "collinear"] or fields[:1] == ["include_spin_orbit"]:
            raise ValueError("Dense-grid conversion currently requires spin none and no SOC")
