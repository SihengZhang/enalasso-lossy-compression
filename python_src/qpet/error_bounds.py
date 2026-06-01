"""Calculate field error bounds from QoI constraints.

Given user-specified error bounds on both fields and QoIs, compute the
most restrictive error bound for each field that satisfies all constraints.

Processing order:
1. Quadratic QoIs (uvar, vvar, wvar) - get initial bounds for U, V, W
2. Skewness (wskew) - refine W bound using wvar error
3. Bilinear QoIs (wpup, wpvp, wppp, wptp, wpqvp) - use known W bound to get bounds for other fields
"""

import numpy as np
from typing import Dict
from .qoi_compute import compute_perturbations, compute_qois, compute_qoi_statistics, compute_qoi_value_ranges


# Mapping from QoI name to the fields it uses
QOI_FIELD_DEPENDENCIES = {
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


def calculate_uvar_bound(
    tau: float,
    qoi_range: float,
    stats: Dict
) -> float:
    """Calculate U field error bound from uvar = ⟨u'²⟩.

    |Δuvar| ≤ 4ε × ⟨|u'|⟩(z) at each height z.
    ε(z) = τ × range(uvar) / (4 × ⟨|u'|⟩(z))
    Take min over all z.
    """
    abs_threshold = tau * qoi_range
    mean_abs_per_z = stats['U']['mean_abs_per_z']

    denominator_per_z = 4 * mean_abs_per_z
    valid_mask = denominator_per_z > 1e-15
    if not valid_mask.any():
        return np.inf

    bounds_per_z = np.full_like(denominator_per_z, np.inf)
    bounds_per_z[valid_mask] = abs_threshold / denominator_per_z[valid_mask]

    return float(bounds_per_z.min())


def calculate_vvar_bound(
    tau: float,
    qoi_range: float,
    stats: Dict
) -> float:
    """Calculate V field error bound from vvar = ⟨v'²⟩.

    |Δvvar| ≤ 4ε × ⟨|v'|⟩(z) at each height z.
    ε(z) = τ × range(vvar) / (4 × ⟨|v'|⟩(z))
    Take min over all z.
    """
    abs_threshold = tau * qoi_range
    mean_abs_per_z = stats['V']['mean_abs_per_z']

    denominator_per_z = 4 * mean_abs_per_z
    valid_mask = denominator_per_z > 1e-15
    if not valid_mask.any():
        return np.inf

    bounds_per_z = np.full_like(denominator_per_z, np.inf)
    bounds_per_z[valid_mask] = abs_threshold / denominator_per_z[valid_mask]

    return float(bounds_per_z.min())


def calculate_wvar_bound(
    tau: float,
    qoi_range: float,
    stats: Dict
) -> float:
    """Calculate W field error bound from wvar = ⟨w'²⟩.

    |Δwvar| ≤ 4ε × ⟨|w'|⟩(z) at each height z.
    ε(z) = τ × range(wvar) / (4 × ⟨|w'|⟩(z))
    Take min over all z.
    """
    abs_threshold = tau * qoi_range
    mean_abs_per_z = stats['W']['mean_abs_per_z']

    denominator_per_z = 4 * mean_abs_per_z
    valid_mask = denominator_per_z > 1e-15
    if not valid_mask.any():
        return np.inf

    bounds_per_z = np.full_like(denominator_per_z, np.inf)
    bounds_per_z[valid_mask] = abs_threshold / denominator_per_z[valid_mask]

    return float(bounds_per_z.min())


def calculate_wskew_bound(
    tau_wskew: float,
    wskew_range: float,
    wvar_abs_error: float,
    perturbations: Dict[str, np.ndarray]
) -> float:
    """Calculate W field error bound from wskew using per-datapoint approach.

    For each datapoint i at height z, define:
        fᵢ = (w'ᵢ)³ / (σ²)^(3/2) = (w'ᵢ)³ / σ³

    wskew = ⟨fᵢ⟩. If each |Δfᵢ| ≤ τ × range(wskew), then |Δwskew| ≤ τ × range(wskew).

    Partial derivatives for fᵢ(w'ᵢ, σ²):
        ∂fᵢ/∂w'ᵢ = 3(w'ᵢ)² / σ³
        ∂fᵢ/∂(σ²) = -3(w'ᵢ)³ / (2σ⁵)

    Error propagation for each datapoint:
        |Δfᵢ| ≤ |∂fᵢ/∂w'ᵢ| × |δw'ᵢ| + |∂fᵢ/∂(σ²)| × |Δσ²|
              ≤ 3(w'ᵢ)²/σ³ × 2ε + 3|(w'ᵢ)³|/(2σ⁵) × Δwvar

    Solving for ε at each datapoint:
        τ × range(wskew) ≥ 6(w'ᵢ)²ε/σ³ + 3|(w'ᵢ)³| × Δwvar/(2σ⁵)
        ε ≤ σ³ × [τ × range(wskew) - 3|(w'ᵢ)³| × Δwvar/(2σ⁵)] / [6(w'ᵢ)²]

    Take min over all datapoints at all heights.
    """
    abs_threshold = tau_wskew * wskew_range
    wp = perturbations['W']
    nz = wp.shape[0]

    # Compute σ at each height
    sigma_per_z = wp.std(axis=(1, 2), ddof=0)  # σ(z)

    global_min_bound = np.inf

    for z_idx in range(nz):
        sigma = sigma_per_z[z_idx]
        if sigma < 1e-10:
            continue

        sigma3 = sigma ** 3
        sigma5 = sigma ** 5
        wp_slice = wp[z_idx]  # shape (ny, nx), these are w'ᵢ values

        # w'ᵢ squared and cubed
        wp_sq = wp_slice ** 2  # (w'ᵢ)²
        wp_cu = wp_slice ** 3  # (w'ᵢ)³

        # σ² error term per datapoint: 3|(w'ᵢ)³| × Δwvar / (2σ⁵)
        sigma2_term = 1.5 * np.abs(wp_cu) * wvar_abs_error / sigma5

        # Available budget per datapoint
        available = abs_threshold - sigma2_term

        # Denominator: 6(w'ᵢ)² / σ³
        denom = 6 * wp_sq / sigma3

        # Compute bound where valid (available > 0, denom > 0)
        valid = (available > 0) & (denom > 1e-15)
        if not valid.any():
            continue

        bounds = np.full_like(wp_slice, np.inf)
        bounds[valid] = available[valid] / denom[valid]

        z_min = bounds.min()
        if z_min < global_min_bound:
            global_min_bound = z_min

    return float(global_min_bound)


def calculate_wpup_bound(
    tau: float,
    qoi_range: float,
    w_bound: float,
    stats: Dict
) -> float:
    """Calculate U field error bound from wpup = ⟨w'u'⟩, given known W bound.

    |Δwpup| ≤ 2ε_W × ⟨|u'|⟩(z) + 2ε_U × ⟨|w'|⟩(z)

    With known ε_W, solve for ε_U at each height z:
        ε_U(z) = (τ × range(wpup) - 2ε_W × ⟨|u'|⟩(z)) / (2 × ⟨|w'|⟩(z))

    Take min over all z.
    """
    abs_threshold = tau * qoi_range
    mean_abs_w_per_z = stats['W']['mean_abs_per_z']
    mean_abs_u_per_z = stats['U']['mean_abs_per_z']

    # Budget consumed by W error
    w_error_contribution = 2 * w_bound * mean_abs_u_per_z

    # Remaining budget for U
    remaining_budget = abs_threshold - w_error_contribution

    # Solve for ε_U
    denominator_per_z = 2 * mean_abs_w_per_z
    valid_mask = (denominator_per_z > 1e-15) & (remaining_budget > 0)
    if not valid_mask.any():
        return np.inf

    bounds_per_z = np.full_like(denominator_per_z, np.inf)
    bounds_per_z[valid_mask] = remaining_budget[valid_mask] / denominator_per_z[valid_mask]

    return float(bounds_per_z.min())


def calculate_wpvp_bound(
    tau: float,
    qoi_range: float,
    w_bound: float,
    stats: Dict
) -> float:
    """Calculate V field error bound from wpvp = ⟨w'v'⟩, given known W bound.

    |Δwpvp| ≤ 2ε_W × ⟨|v'|⟩(z) + 2ε_V × ⟨|w'|⟩(z)

    With known ε_W, solve for ε_V at each height z:
        ε_V(z) = (τ × range(wpvp) - 2ε_W × ⟨|v'|⟩(z)) / (2 × ⟨|w'|⟩(z))
    """
    abs_threshold = tau * qoi_range
    mean_abs_w_per_z = stats['W']['mean_abs_per_z']
    mean_abs_v_per_z = stats['V']['mean_abs_per_z']

    w_error_contribution = 2 * w_bound * mean_abs_v_per_z
    remaining_budget = abs_threshold - w_error_contribution

    denominator_per_z = 2 * mean_abs_w_per_z
    valid_mask = (denominator_per_z > 1e-15) & (remaining_budget > 0)
    if not valid_mask.any():
        return np.inf

    bounds_per_z = np.full_like(denominator_per_z, np.inf)
    bounds_per_z[valid_mask] = remaining_budget[valid_mask] / denominator_per_z[valid_mask]

    return float(bounds_per_z.min())


def calculate_wppp_bound(
    tau: float,
    qoi_range: float,
    w_bound: float,
    stats: Dict
) -> float:
    """Calculate PP field error bound from wppp = ⟨w'p'⟩, given known W bound.

    |Δwppp| ≤ 2ε_W × ⟨|p'|⟩(z) + 2ε_PP × ⟨|w'|⟩(z)

    With known ε_W, solve for ε_PP at each height z:
        ε_PP(z) = (τ × range(wppp) - 2ε_W × ⟨|p'|⟩(z)) / (2 × ⟨|w'|⟩(z))
    """
    abs_threshold = tau * qoi_range
    mean_abs_w_per_z = stats['W']['mean_abs_per_z']
    mean_abs_pp_per_z = stats['PP']['mean_abs_per_z']

    w_error_contribution = 2 * w_bound * mean_abs_pp_per_z
    remaining_budget = abs_threshold - w_error_contribution

    denominator_per_z = 2 * mean_abs_w_per_z
    valid_mask = (denominator_per_z > 1e-15) & (remaining_budget > 0)
    if not valid_mask.any():
        return np.inf

    bounds_per_z = np.full_like(denominator_per_z, np.inf)
    bounds_per_z[valid_mask] = remaining_budget[valid_mask] / denominator_per_z[valid_mask]

    return float(bounds_per_z.min())


def calculate_wptp_bound(
    tau: float,
    qoi_range: float,
    w_bound: float,
    stats: Dict
) -> float:
    """Calculate TABS field error bound from wptp = ⟨w'T'⟩, given known W bound.

    |Δwptp| ≤ 2ε_W × ⟨|T'|⟩(z) + 2ε_TABS × ⟨|w'|⟩(z)

    With known ε_W, solve for ε_TABS at each height z:
        ε_TABS(z) = (τ × range(wptp) - 2ε_W × ⟨|T'|⟩(z)) / (2 × ⟨|w'|⟩(z))
    """
    abs_threshold = tau * qoi_range
    mean_abs_w_per_z = stats['W']['mean_abs_per_z']
    mean_abs_tabs_per_z = stats['TABS']['mean_abs_per_z']

    w_error_contribution = 2 * w_bound * mean_abs_tabs_per_z
    remaining_budget = abs_threshold - w_error_contribution

    denominator_per_z = 2 * mean_abs_w_per_z
    valid_mask = (denominator_per_z > 1e-15) & (remaining_budget > 0)
    if not valid_mask.any():
        return np.inf

    bounds_per_z = np.full_like(denominator_per_z, np.inf)
    bounds_per_z[valid_mask] = remaining_budget[valid_mask] / denominator_per_z[valid_mask]

    return float(bounds_per_z.min())


def calculate_wpqvp_bound(
    tau: float,
    qoi_range: float,
    w_bound: float,
    stats: Dict
) -> float:
    """Calculate QV field error bound from wpqvp = ⟨w'qv'⟩, given known W bound.

    |Δwpqvp| ≤ 2ε_W × ⟨|qv'|⟩(z) + 2ε_QV × ⟨|w'|⟩(z)

    With known ε_W, solve for ε_QV at each height z:
        ε_QV(z) = (τ × range(wpqvp) - 2ε_W × ⟨|qv'|⟩(z)) / (2 × ⟨|w'|⟩(z))
    """
    abs_threshold = tau * qoi_range
    mean_abs_w_per_z = stats['W']['mean_abs_per_z']
    mean_abs_qv_per_z = stats['QV']['mean_abs_per_z']

    w_error_contribution = 2 * w_bound * mean_abs_qv_per_z
    remaining_budget = abs_threshold - w_error_contribution

    denominator_per_z = 2 * mean_abs_w_per_z
    valid_mask = (denominator_per_z > 1e-15) & (remaining_budget > 0)
    if not valid_mask.any():
        return np.inf

    bounds_per_z = np.full_like(denominator_per_z, np.inf)
    bounds_per_z[valid_mask] = remaining_budget[valid_mask] / denominator_per_z[valid_mask]

    return float(bounds_per_z.min())


def calculate_field_bounds(
    fields: Dict[str, np.ndarray],
    user_field_bounds: Dict[str, float],
    user_qoi_bounds: Dict[str, float],
    return_relative: bool = False
) -> Dict[str, float]:
    """Calculate final error bounds for each field.

    Processing order:
    1. Quadratic QoIs (uvar, vvar, wvar) - get initial bounds
    2. Skewness (wskew) - refine W bound using wvar error
    3. Bilinear QoIs - use known W bound to get bounds for other fields

    Args:
        fields: Dictionary mapping field names to 3D arrays (z, y, x)
        user_field_bounds: User-specified relative error bounds for each field
        user_qoi_bounds: User-specified relative error bounds for each QoI
        return_relative: If True, return relative bounds; if False, return absolute

    Returns:
        Dictionary mapping field names to error bounds (absolute or relative)
    """
    # Compute perturbations and statistics
    perturbations = compute_perturbations(fields)
    qois = compute_qois(perturbations)
    stats = compute_qoi_statistics(perturbations)
    qoi_ranges = compute_qoi_value_ranges(qois)

    # Compute value ranges for fields
    field_ranges = {}
    for name, field in fields.items():
        field_ranges[name] = float(field.max() - field.min())

    # Initialize bounds with user-specified field bounds (convert to absolute)
    bounds = {}
    for name in fields.keys():
        user_rel_bound = user_field_bounds.get(name, np.inf)
        bounds[name] = user_rel_bound * field_ranges[name]

    # ========== Step 1: Quadratic QoIs (uvar, vvar, wvar) ==========

    # uvar -> U bound
    tau_uvar = user_qoi_bounds.get('uvar', np.inf)
    if tau_uvar != np.inf and qoi_ranges['uvar'] >= 1e-15:
        bound = calculate_uvar_bound(tau_uvar, qoi_ranges['uvar'], stats)
        bounds['U'] = min(bounds['U'], bound)

    # vvar -> V bound
    tau_vvar = user_qoi_bounds.get('vvar', np.inf)
    if tau_vvar != np.inf and qoi_ranges['vvar'] >= 1e-15:
        bound = calculate_vvar_bound(tau_vvar, qoi_ranges['vvar'], stats)
        bounds['V'] = min(bounds['V'], bound)

    # wvar -> W bound
    tau_wvar = user_qoi_bounds.get('wvar', np.inf)
    wvar_abs_error = np.inf
    if tau_wvar != np.inf and qoi_ranges['wvar'] >= 1e-15:
        bound = calculate_wvar_bound(tau_wvar, qoi_ranges['wvar'], stats)
        bounds['W'] = min(bounds['W'], bound)
        wvar_abs_error = tau_wvar * qoi_ranges['wvar']

    # ========== Step 2: Skewness (wskew) ==========

    tau_wskew = user_qoi_bounds.get('wskew', np.inf)
    if tau_wskew != np.inf and qoi_ranges['wskew'] >= 1e-15:
        # If wvar bound not specified, estimate wvar error from current W bound
        if wvar_abs_error == np.inf:
            max_mean_abs_w = stats['W']['mean_abs_per_z'].max()
            wvar_abs_error = 4 * bounds['W'] * max_mean_abs_w

        bound = calculate_wskew_bound(
            tau_wskew, qoi_ranges['wskew'], wvar_abs_error, perturbations
        )
        bounds['W'] = min(bounds['W'], bound)

    # ========== Step 3: Bilinear QoIs (using known W bound) ==========

    w_bound = bounds['W']

    # wpup -> U bound
    tau_wpup = user_qoi_bounds.get('wpup', np.inf)
    if tau_wpup != np.inf and qoi_ranges['wpup'] >= 1e-15:
        bound = calculate_wpup_bound(tau_wpup, qoi_ranges['wpup'], w_bound, stats)
        bounds['U'] = min(bounds['U'], bound)

    # wpvp -> V bound
    tau_wpvp = user_qoi_bounds.get('wpvp', np.inf)
    if tau_wpvp != np.inf and qoi_ranges['wpvp'] >= 1e-15:
        bound = calculate_wpvp_bound(tau_wpvp, qoi_ranges['wpvp'], w_bound, stats)
        bounds['V'] = min(bounds['V'], bound)

    # wppp -> PP bound
    tau_wppp = user_qoi_bounds.get('wppp', np.inf)
    if tau_wppp != np.inf and qoi_ranges['wppp'] >= 1e-15:
        bound = calculate_wppp_bound(tau_wppp, qoi_ranges['wppp'], w_bound, stats)
        bounds['PP'] = min(bounds['PP'], bound)

    # wptp -> TABS bound
    tau_wptp = user_qoi_bounds.get('wptp', np.inf)
    if tau_wptp != np.inf and qoi_ranges['wptp'] >= 1e-15:
        bound = calculate_wptp_bound(tau_wptp, qoi_ranges['wptp'], w_bound, stats)
        bounds['TABS'] = min(bounds['TABS'], bound)

    # wpqvp -> QV bound
    tau_wpqvp = user_qoi_bounds.get('wpqvp', np.inf)
    if tau_wpqvp != np.inf and qoi_ranges['wpqvp'] >= 1e-15:
        bound = calculate_wpqvp_bound(tau_wpqvp, qoi_ranges['wpqvp'], w_bound, stats)
        bounds['QV'] = min(bounds['QV'], bound)

    if return_relative:
        rel_bounds = {}
        for name in fields.keys():
            vrange = field_ranges[name]
            rel_bounds[name] = bounds[name] / vrange if vrange > 0 else np.inf
        return rel_bounds

    return bounds


def report_bounds(
    fields: Dict[str, np.ndarray],
    user_field_bounds: Dict[str, float],
    user_qoi_bounds: Dict[str, float],
    calculated_bounds: Dict[str, float],
    is_relative: bool = False
) -> str:
    """Generate a report of calculated bounds.

    Args:
        fields: Original field data
        user_field_bounds: User-specified field bounds
        user_qoi_bounds: User-specified QoI bounds
        calculated_bounds: Calculated error bounds (absolute or relative)
        is_relative: If True, calculated_bounds are relative; if False, absolute

    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("QPET Error Bound Calculation Report")
    lines.append("=" * 60)

    # Field ranges
    lines.append("\nField Value Ranges:")
    for name, field in fields.items():
        vrange = field.max() - field.min()
        lines.append(f"  {name:6s}: [{field.min():.4e}, {field.max():.4e}] (range: {vrange:.4e})")

    # User bounds
    lines.append("\nUser-Specified Bounds (relative):")
    lines.append("  Fields:")
    for name in fields.keys():
        lines.append(f"    {name:6s}: {user_field_bounds.get(name, 'N/A')}")
    lines.append("  QoIs:")
    for name in QOI_FIELD_DEPENDENCIES.keys():
        lines.append(f"    {name:8s}: {user_qoi_bounds.get(name, 'N/A')}")

    # Calculated bounds
    bound_type = "relative" if is_relative else "absolute"
    lines.append(f"\nCalculated Field Error Bounds ({bound_type}):")
    for name in fields.keys():
        vrange = fields[name].max() - fields[name].min()
        bound = calculated_bounds[name]
        if is_relative:
            rel_bound = bound
            abs_bound = bound * vrange
        else:
            abs_bound = bound
            rel_bound = bound / vrange if vrange > 0 else np.inf
        lines.append(f"  {name:6s}: {bound:.4e} (abs: {abs_bound:.4e}, rel: {rel_bound:.4e})")

    lines.append("=" * 60)
    return "\n".join(lines)
