#!/usr/bin/env python3
"""
Analyze error distribution of SZ3 compressed/decompressed W field vs ground truth.
"""

import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = '/archives/disk1/hi_res_sim/data/compressed-data'
SHAPE = (96, 260, 256, 256)

def load_binary(filepath):
    """Load binary float32 file."""
    return np.fromfile(filepath, dtype=np.float32).reshape(SHAPE)

def main():
    print("Loading data...")
    w_orig = load_binary(f"{DATA_DIR}/w_96x260x256x256.dat")
    w_rel1e3 = load_binary(f"{DATA_DIR}/w_96x260x256x256_rel1e-3.dat")
    w_rel1e4 = load_binary(f"{DATA_DIR}/w_96x260x256x256_rel1e-4.dat")

    print("Computing errors...")
    err_rel1e3 = w_rel1e3 - w_orig
    err_rel1e4 = w_rel1e4 - w_orig

    # Compute statistics
    value_range = w_orig.max() - w_orig.min()

    stats = {
        'REL 1e-3': {
            'max_abs_err': np.abs(err_rel1e3).max(),
            'mean_abs_err': np.abs(err_rel1e3).mean(),
            'rmse': np.sqrt((err_rel1e3**2).mean()),
            'std': err_rel1e3.std(),
            'rel_max': np.abs(err_rel1e3).max() / value_range,
        },
        'REL 1e-4': {
            'max_abs_err': np.abs(err_rel1e4).max(),
            'mean_abs_err': np.abs(err_rel1e4).mean(),
            'rmse': np.sqrt((err_rel1e4**2).mean()),
            'std': err_rel1e4.std(),
            'rel_max': np.abs(err_rel1e4).max() / value_range,
        }
    }

    print(f"\nOriginal W: min={w_orig.min():.4f}, max={w_orig.max():.4f}, range={value_range:.4f} m/s")
    print("\n=== Error Statistics ===")
    for name, s in stats.items():
        print(f"\n{name}:")
        print(f"  Max absolute error: {s['max_abs_err']:.6f} m/s")
        print(f"  Mean absolute error: {s['mean_abs_err']:.6f} m/s")
        print(f"  RMSE: {s['rmse']:.6f} m/s")
        print(f"  Std dev: {s['std']:.6f} m/s")
        print(f"  Relative max error: {s['rel_max']:.6e} (bound was {name.split()[-1]})")

    # Create histograms
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # REL 1e-3 histogram
    ax0 = axes[0]
    counts, bins, _ = ax0.hist(err_rel1e3.flatten(), bins=200, density=True,
                                alpha=0.7, color='steelblue', edgecolor='none')
    ax0.axvline(x=0, color='red', linestyle='--', linewidth=1, label='Zero error')
    ax0.axvline(x=stats['REL 1e-3']['max_abs_err'], color='orange', linestyle=':',
                linewidth=1.5, label=f"Max err: {stats['REL 1e-3']['max_abs_err']:.4f}")
    ax0.axvline(x=-stats['REL 1e-3']['max_abs_err'], color='orange', linestyle=':', linewidth=1.5)
    ax0.set_xlabel('Error (m/s)')
    ax0.set_ylabel('Density')
    ax0.set_title(f"REL 1e-3 Error Distribution\n(CR=17.5x, Max err={stats['REL 1e-3']['max_abs_err']:.4f} m/s)")
    ax0.legend(loc='upper right')
    ax0.set_xlim(-0.025, 0.025)

    # REL 1e-4 histogram
    ax1 = axes[1]
    counts, bins, _ = ax1.hist(err_rel1e4.flatten(), bins=200, density=True,
                                alpha=0.7, color='forestgreen', edgecolor='none')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=1, label='Zero error')
    ax1.axvline(x=stats['REL 1e-4']['max_abs_err'], color='orange', linestyle=':',
                linewidth=1.5, label=f"Max err: {stats['REL 1e-4']['max_abs_err']:.5f}")
    ax1.axvline(x=-stats['REL 1e-4']['max_abs_err'], color='orange', linestyle=':', linewidth=1.5)
    ax1.set_xlabel('Error (m/s)')
    ax1.set_ylabel('Density')
    ax1.set_title(f"REL 1e-4 Error Distribution\n(CR=8.4x, Max err={stats['REL 1e-4']['max_abs_err']:.5f} m/s)")
    ax1.legend(loc='upper right')
    ax1.set_xlim(-0.0025, 0.0025)

    plt.tight_layout()
    plt.savefig('plots/compression_error_histogram.png', dpi=150, bbox_inches='tight')
    print("\nSaved: plots/compression_error_histogram.png")
    plt.show()

if __name__ == '__main__':
    main()
