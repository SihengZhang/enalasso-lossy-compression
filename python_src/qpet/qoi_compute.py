"""Compute QoIs (Quantities of Interest) from field data.

All QoIs are vertical profiles computed from horizontal averages of
turbulent perturbations using Reynolds decomposition.
"""

import numpy as np
from typing import Dict, Tuple


def compute_perturbations(fields: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Compute perturbation fields using Reynolds decomposition.

    For each field φ, compute φ' = φ - ⟨φ⟩_xy where ⟨·⟩_xy is the
    horizontal mean at each height level.

    Args:
        fields: Dictionary mapping field names to 3D arrays (z, y, x)

    Returns:
        Dictionary mapping field names to perturbation arrays (same shape)
    """
    perturbations = {}
    for name, field in fields.items():
        # field shape: (nz, ny, nx)
        # Compute horizontal mean at each height level
        horiz_mean = field.mean(axis=(1, 2), keepdims=True)
        perturbations[name] = field - horiz_mean
    return perturbations


def compute_qois(perturbations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Compute all 9 QoIs from perturbation fields.

    Args:
        perturbations: Dictionary with keys 'U', 'V', 'W', 'PP', 'TABS', 'QV'
                      containing perturbation arrays (z, y, x)

    Returns:
        Dictionary mapping QoI names to vertical profiles (1D arrays of length nz)
    """
    up = perturbations['U']
    vp = perturbations['V']
    wp = perturbations['W']
    pp = perturbations['PP']
    tp = perturbations['TABS']
    qvp = perturbations['QV']

    qois = {}

    # Bilinear QoIs: ⟨w'φ'⟩_xy
    qois['wpup'] = (wp * up).mean(axis=(1, 2))
    qois['wpvp'] = (wp * vp).mean(axis=(1, 2))
    qois['wppp'] = (wp * pp).mean(axis=(1, 2))
    qois['wptp'] = (wp * tp).mean(axis=(1, 2))
    qois['wpqvp'] = (wp * qvp).mean(axis=(1, 2))

    # Quadratic QoIs: ⟨φ'²⟩_xy
    qois['uvar'] = (up * up).mean(axis=(1, 2))
    qois['vvar'] = (vp * vp).mean(axis=(1, 2))
    qois['wvar'] = (wp * wp).mean(axis=(1, 2))

    # Skewness: ⟨((w' - μ)/σ)³⟩_xy
    # Note: μ = ⟨w'⟩_xy ≈ 0 by construction
    mu = wp.mean(axis=(1, 2), keepdims=True)
    sigma = wp.std(axis=(1, 2), ddof=0, keepdims=True)
    # Avoid division by zero
    sigma = np.where(sigma > 1e-10, sigma, 1e-10)
    z_normalized = (wp - mu) / sigma
    qois['wskew'] = (z_normalized ** 3).mean(axis=(1, 2))

    return qois


def compute_qoi_statistics(perturbations: Dict[str, np.ndarray]) -> Dict[str, Dict]:
    """Compute statistics needed for error bound calculation.

    Args:
        perturbations: Dictionary with perturbation arrays

    Returns:
        Dictionary with statistics for each field and QoI
    """
    stats = {}

    for name, p in perturbations.items():
        # Mean of absolute perturbation at each height, then max over heights
        mean_abs_p = np.abs(p).mean(axis=(1, 2))  # shape (nz,)
        stats[name] = {
            'mean_abs_per_z': mean_abs_p,
            'max_mean_abs': mean_abs_p.max(),
        }

    return stats


def compute_qoi_value_ranges(qois: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compute value range for each QoI (max - min over all z levels).

    Args:
        qois: Dictionary mapping QoI names to vertical profiles

    Returns:
        Dictionary mapping QoI names to value ranges
    """
    ranges = {}
    for name, profile in qois.items():
        ranges[name] = float(profile.max() - profile.min())
    return ranges
