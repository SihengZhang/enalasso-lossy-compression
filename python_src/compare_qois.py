#!/usr/bin/env python3
"""Compare QoIs from original data vs compressed/decompressed data."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pysz import sz, szConfig, szErrorBoundMode

from qpet import compute_perturbations, compute_qois, calculate_field_bounds

FIELD_NAMES = ['U', 'V', 'W', 'PP', 'TABS', 'QV']
QOI_NAMES = ['wpup', 'wpvp', 'wppp', 'wptp', 'wpqvp', 'uvar', 'vvar', 'wvar', 'wskew']


def load_binary_fields(input_dir: Path):
    """Load binary field files from directory."""
    fields = {}
    for name in FIELD_NAMES:
        pattern = f"{name}_*.raw"
        matches = list(input_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No file matching {pattern} in {input_dir}")
        filepath = matches[0]
        dims_str = filepath.stem.split('_')[1]
        dims = tuple(int(d) for d in dims_str.split('x'))
        data = np.fromfile(filepath, dtype=np.float32).reshape(dims)
        fields[name] = data
    return fields


def compress_decompress(fields, bounds):
    """Compress and decompress fields with given bounds."""
    decompressed = {}
    for name, field in fields.items():
        config = szConfig()
        config.errorBoundMode = szErrorBoundMode.ABS
        config.absErrorBound = bounds[name]
        compressed, _ = sz.compress(field, config)
        dec_data, _ = sz.decompress(compressed, np.float32, field.shape)
        decompressed[name] = dec_data
    return decompressed


def main():
    input_dir = Path('/archives/disk1/hi_res_sim/data/enalasso_sam3d_20160410era5d25x100_sbmwrm-aer1-flxsstC1.m1.20160410.001500.nc.v0_binary')
    error_bound = 1e-3

    print("Loading original fields...")
    fields = load_binary_fields(input_dir)

    # Calculate error bounds
    user_field_bounds = {name: error_bound for name in FIELD_NAMES}
    user_qoi_bounds = {name: error_bound for name in QOI_NAMES}

    print("Calculating field error bounds...")
    calculated_bounds = calculate_field_bounds(fields, user_field_bounds, user_qoi_bounds)

    print("Compressing and decompressing...")
    decompressed = compress_decompress(fields, calculated_bounds)

    # Compute QoIs
    print("Computing QoIs...")
    orig_pert = compute_perturbations(fields)
    dec_pert = compute_perturbations(decompressed)
    orig_qois = compute_qois(orig_pert)
    dec_qois = compute_qois(dec_pert)

    # Create plot
    nz = 260
    z = np.arange(nz)

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    axes = axes.flatten()

    qoi_titles = {
        'wpup': "$\\overline{w'u'}$",
        'wpvp': "$\\overline{w'v'}$",
        'wppp': "$\\overline{w'p'}$",
        'wptp': "$\\overline{w'T'}$",
        'wpqvp': "$\\overline{w'q_v'}$",
        'uvar': "$\\overline{u'^2}$",
        'vvar': "$\\overline{v'^2}$",
        'wvar': "$\\overline{w'^2}$",
        'wskew': "$S_w$"
    }

    for i, qoi_name in enumerate(QOI_NAMES):
        ax = axes[i]
        orig = orig_qois[qoi_name]
        dec = dec_qois[qoi_name]

        ax.plot(orig, z, 'b-', linewidth=1.5, label='Original')
        ax.plot(dec, z, 'r--', linewidth=1.5, label='Compressed')

        ax.set_ylabel('z index')
        ax.set_xlabel(qoi_name)
        ax.set_title(f'{qoi_titles[qoi_name]} ({qoi_name})')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Compute and display max relative error
        vrange = orig.max() - orig.min()
        if vrange > 0:
            max_rel_err = np.abs(orig - dec).max() / vrange
        else:
            max_rel_err = 0
        ax.text(0.02, 0.98, f'Max rel err: {max_rel_err:.2e}',
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'QoI Comparison: Original vs Compressed (rel error bound = {error_bound})', fontsize=14)
    plt.tight_layout()

    output_path = Path('/home/szhang/enalasso-lossy-compression/LaTeX/qoi_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

    plt.close()


if __name__ == '__main__':
    main()
