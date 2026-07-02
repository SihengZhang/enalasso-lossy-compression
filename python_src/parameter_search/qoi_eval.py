#!/usr/bin/env python3
"""Self-contained QoI computation for the parameter-search framework.

This module reimplements the turbulence-statistic (QoI) math, the QoI->field
dependency map, the binary-field loader, and the QoI error metric so that the
parameter-search framework has NO dependency on the ``qpet`` package.

QoI definitions follow ``docs/derived_variables.md``. All fields are 3D arrays
with shape ``(nz, ny, nx)`` (z, y, x) and dtype float32. Perturbations use
Reynolds decomposition: phi' = phi - <phi>_xy, where the horizontal mean is
taken over (y, x) at each height level z.
"""

from pathlib import Path
from typing import Dict

import numpy as np

# The six raw 3D fields used for compression.
FIELD_NAMES = ['U', 'V', 'W', 'PP', 'TABS', 'QV']

# The nine derived QoIs and the raw fields each one depends on. A QoI's value
# is a function ONLY of the listed fields' perturbations, which is what makes
# the staged (single-field then multi-field) search sound.
QOI_FIELD_DEPENDENCIES: Dict[str, tuple] = {
    'wpup': ('W', 'U'),
    'wpvp': ('W', 'V'),
    'wppp': ('W', 'PP'),
    'wptp': ('W', 'TABS'),
    'wpqvp': ('W', 'QV'),
    'uvar': ('U',),
    'vvar': ('V',),
    'wvar': ('W',),
    'wskew': ('W',),
}

# QoIs that depend on exactly one raw field (handled first by the search) vs.
# QoIs that couple two fields (handled after their fields are fixed).
SINGLE_FIELD_QOIS = [q for q, f in QOI_FIELD_DEPENDENCIES.items() if len(f) == 1]
MULTI_FIELD_QOIS = [q for q, f in QOI_FIELD_DEPENDENCIES.items() if len(f) > 1]

QOI_NAMES = list(QOI_FIELD_DEPENDENCIES.keys())

# Ranges below this are treated as zero (QoI auto-passes) to avoid div-by-zero.
_RANGE_EPS = 1e-12


def load_binary_fields(input_dir) -> Dict[str, np.ndarray]:
    """Load the six raw fields from ``{NAME}_{nz}x{ny}x{nx}.raw`` files.

    Mirrors the naming convention used elsewhere in the repo. Returns a dict
    mapping field name -> float32 array of shape (nz, ny, nx).
    """
    input_dir = Path(input_dir)
    fields: Dict[str, np.ndarray] = {}
    for name in FIELD_NAMES:
        matches = sorted(input_dir.glob(f"{name}_*.raw"))
        if not matches:
            raise FileNotFoundError(
                f"No file matching '{name}_*.raw' in {input_dir}")
        filepath = matches[0]
        # Filename is e.g. "U_260x256x256.raw" -> dims "260x256x256".
        dims_str = filepath.stem.split('_', 1)[1]
        dims = tuple(int(d) for d in dims_str.split('x'))
        data = np.fromfile(filepath, dtype=np.float32).reshape(dims)
        fields[name] = data
    return fields


def compute_perturbations(fields: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Reynolds decomposition: phi' = phi - mean over (y, x) per z-level."""
    perturbations: Dict[str, np.ndarray] = {}
    for name, field in fields.items():
        horiz_mean = field.mean(axis=(1, 2), keepdims=True)  # (nz, 1, 1)
        perturbations[name] = field - horiz_mean
    return perturbations


def compute_qois(perturbations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Compute the nine QoI vertical profiles (each length nz).

    Expects perturbation arrays keyed by 'U','V','W','PP','TABS','QV'.
    """
    up = perturbations['U']
    vp = perturbations['V']
    wp = perturbations['W']
    pp = perturbations['PP']
    tp = perturbations['TABS']
    qvp = perturbations['QV']

    qois: Dict[str, np.ndarray] = {}

    # Bilinear (momentum / thermodynamic flux) QoIs: <w' x'>_xy.
    qois['wpup'] = (wp * up).mean(axis=(1, 2))
    qois['wpvp'] = (wp * vp).mean(axis=(1, 2))
    qois['wppp'] = (wp * pp).mean(axis=(1, 2))
    qois['wptp'] = (wp * tp).mean(axis=(1, 2))
    qois['wpqvp'] = (wp * qvp).mean(axis=(1, 2))

    # Quadratic (velocity variance) QoIs: <x'^2>_xy.
    qois['uvar'] = (up * up).mean(axis=(1, 2))
    qois['vvar'] = (vp * vp).mean(axis=(1, 2))
    qois['wvar'] = (wp * wp).mean(axis=(1, 2))

    # Higher-order: vertical velocity skewness <((w'-mu)/sigma)^3>_xy.
    mu = wp.mean(axis=(1, 2), keepdims=True)
    sigma = wp.std(axis=(1, 2), ddof=0, keepdims=True)
    sigma = np.where(sigma > 1e-10, sigma, 1e-10)  # avoid div-by-zero
    z_norm = (wp - mu) / sigma
    qois['wskew'] = (z_norm ** 3).mean(axis=(1, 2))

    return qois


def compute_single_qoi(name: str, fields: Dict[str, np.ndarray]) -> np.ndarray:
    """Compute one QoI profile from only the raw fields it depends on.

    ``fields`` must contain the raw (un-perturbed) arrays for the QoI's
    dependency fields (see ``QOI_FIELD_DEPENDENCIES``). Perturbations are
    computed locally. This avoids touching the other fields, which (by
    separability) do not affect this QoI.
    """
    deps = QOI_FIELD_DEPENDENCIES[name]
    perts = {f: fields[f] - fields[f].mean(axis=(1, 2), keepdims=True)
             for f in deps}
    if name in ('wpup', 'wpvp', 'wppp', 'wptp', 'wpqvp'):
        w, other = deps  # always ('W', X)
        return (perts[w] * perts[other]).mean(axis=(1, 2))
    if name == 'uvar':
        return (perts['U'] ** 2).mean(axis=(1, 2))
    if name == 'vvar':
        return (perts['V'] ** 2).mean(axis=(1, 2))
    if name == 'wvar':
        return (perts['W'] ** 2).mean(axis=(1, 2))
    if name == 'wskew':
        wp = perts['W']
        mu = wp.mean(axis=(1, 2), keepdims=True)
        sigma = wp.std(axis=(1, 2), ddof=0, keepdims=True)
        sigma = np.where(sigma > 1e-10, sigma, 1e-10)
        return (((wp - mu) / sigma) ** 3).mean(axis=(1, 2))
    raise ValueError(f"Unknown QoI '{name}'")


def qoi_value_ranges(qois: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Value range (max - min over z) for each QoI profile."""
    return {name: float(profile.max() - profile.min())
            for name, profile in qois.items()}


def qoi_rel_error(orig_qois: Dict[str, np.ndarray],
                  ranges: Dict[str, float],
                  test_qois: Dict[str, np.ndarray],
                  name: str) -> float:
    """Relative error of one QoI: max_z |orig - test| / range(orig).

    Returns 0.0 when the QoI's value range is ~0 (treated as auto-pass).
    """
    vrange = ranges[name]
    if vrange <= _RANGE_EPS:
        return 0.0
    max_abs = float(np.abs(orig_qois[name] - test_qois[name]).max())
    return max_abs / vrange


def all_qoi_errors(orig_fields: Dict[str, np.ndarray],
                   dec_fields: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Relative error for all nine QoIs given original vs decompressed fields."""
    orig_qois = compute_qois(compute_perturbations(orig_fields))
    test_qois = compute_qois(compute_perturbations(dec_fields))
    ranges = qoi_value_ranges(orig_qois)
    return {name: qoi_rel_error(orig_qois, ranges, test_qois, name)
            for name in QOI_NAMES}
