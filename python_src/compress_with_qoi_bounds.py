#!/usr/bin/env python3
"""QoI-preserving compression for LASSO-ENA turbulence data.

This script compresses 6 SAM3D fields (U, V, W, PP, TABS, QV) using SZ3
while bounding errors on both the original fields and 9 derived QoIs.

Usage:
    python compress_with_qoi_bounds.py --input-dir <path> --output-dir <path> [options]
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from pysz import sz, szConfig, szErrorBoundMode
from qpet import compute_perturbations, compute_qois, calculate_field_bounds
from qpet.error_bounds import report_bounds, QOI_FIELD_DEPENDENCIES
from qpet.qoi_compute import compute_qoi_value_ranges


FIELD_NAMES = ['U', 'V', 'W', 'PP', 'TABS', 'QV']
QOI_NAMES = ['wpup', 'wpvp', 'wppp', 'uvar', 'vvar', 'wvar', 'wskew', 'wptp', 'wpqvp']


def load_binary_fields(input_dir: Path) -> Dict[str, np.ndarray]:
    """Load binary field files from directory.

    Expects files named like: U_260x256x256.raw
    """
    fields = {}
    for name in FIELD_NAMES:
        pattern = f"{name}_*.raw"
        matches = list(input_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No file matching {pattern} in {input_dir}")
        filepath = matches[0]

        # Parse dimensions from filename
        dims_str = filepath.stem.split('_')[1]
        dims = tuple(int(d) for d in dims_str.split('x'))

        # Load as float32 and reshape
        data = np.fromfile(filepath, dtype=np.float32).reshape(dims)
        fields[name] = data
        print(f"Loaded {name}: shape={dims}, range=[{data.min():.4e}, {data.max():.4e}]")

    return fields


def compress_fields(
    fields: Dict[str, np.ndarray],
    bounds: Dict[str, float],
    output_dir: Path,
    use_relative: bool = False
) -> Dict[str, Tuple[np.ndarray, float]]:
    """Compress all fields with calculated error bounds.

    Args:
        fields: Dictionary mapping field names to arrays
        bounds: Dictionary mapping field names to error bounds
                (absolute if use_relative=False, relative if use_relative=True)
        output_dir: Output directory for compressed files
        use_relative: If True, use REL error bound mode; if False, use ABS mode

    Returns:
        Dictionary mapping field names to (compressed_data, compression_ratio)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for name, field in fields.items():
        config = szConfig()
        if use_relative:
            config.errorBoundMode = szErrorBoundMode.REL
            config.relErrorBound = bounds[name]
            bound_str = f"rel error bound {bounds[name]:.4e}"
        else:
            config.errorBoundMode = szErrorBoundMode.ABS
            config.absErrorBound = bounds[name]
            bound_str = f"abs error bound {bounds[name]:.4e}"

        print(f"Compressing {name} with {bound_str}...")
        start = time.time()
        compressed, ratio = sz.compress(field, config)
        elapsed = time.time() - start

        # Save compressed data
        out_path = output_dir / f"{name}.sz"
        compressed.tofile(out_path)

        results[name] = (compressed, ratio)
        print(f"  Ratio: {ratio:.2f}x, Size: {len(compressed):,} bytes, Time: {elapsed:.2f}s")

    return results


def decompress_fields(
    compressed: Dict[str, Tuple[np.ndarray, float]],
    original_shapes: Dict[str, Tuple[int, ...]]
) -> Dict[str, np.ndarray]:
    """Decompress all fields."""
    decompressed = {}

    for name, (comp_data, _) in compressed.items():
        shape = original_shapes[name]
        dec_data, _ = sz.decompress(comp_data, np.float32, shape)
        decompressed[name] = dec_data

    return decompressed


def verify_bounds(
    original_fields: Dict[str, np.ndarray],
    decompressed_fields: Dict[str, np.ndarray],
    user_field_bounds: Dict[str, float],
    user_qoi_bounds: Dict[str, float]
) -> bool:
    """Verify that all error bounds are satisfied.

    Returns:
        True if all bounds are satisfied, False otherwise
    """
    all_passed = True

    print("\n" + "=" * 60)
    print("VERIFICATION RESULTS")
    print("=" * 60)

    # Verify field error bounds
    print("\nField Errors:")
    for name in FIELD_NAMES:
        orig = original_fields[name]
        dec = decompressed_fields[name]
        max_err = np.abs(orig - dec).max()
        vrange = orig.max() - orig.min()
        rel_err = max_err / vrange if vrange > 0 else 0

        user_bound = user_field_bounds.get(name, np.inf)
        passed = rel_err <= user_bound
        status = "PASS" if passed else "FAIL"
        all_passed = all_passed and passed

        print(f"  {name:6s}: max_rel_err = {rel_err:.4e}, bound = {user_bound:.4e} [{status}]")

    # Compute QoIs from both original and decompressed
    orig_pert = compute_perturbations(original_fields)
    dec_pert = compute_perturbations(decompressed_fields)
    orig_qois = compute_qois(orig_pert)
    dec_qois = compute_qois(dec_pert)
    qoi_ranges = compute_qoi_value_ranges(orig_qois)

    # Verify QoI error bounds
    print("\nQoI Errors:")
    for name in QOI_NAMES:
        orig_q = orig_qois[name]
        dec_q = dec_qois[name]
        max_err = np.abs(orig_q - dec_q).max()
        vrange = qoi_ranges[name]
        rel_err = max_err / vrange if vrange > 0 else 0

        user_bound = user_qoi_bounds.get(name, np.inf)
        passed = rel_err <= user_bound
        status = "PASS" if passed else "FAIL"
        all_passed = all_passed and passed

        print(f"  {name:8s}: max_rel_err = {rel_err:.4e}, bound = {user_bound:.4e} [{status}]")

    print("\n" + "=" * 60)
    print(f"Overall: {'ALL BOUNDS SATISFIED' if all_passed else 'SOME BOUNDS VIOLATED'}")
    print("=" * 60)

    return all_passed


def parse_bounds_arg(arg_str: str, names: list) -> Dict[str, float]:
    """Parse comma-separated bounds argument.

    Format: "1e-3,1e-3,1e-3,1e-3,1e-3,1e-3" or "1e-3" for all same
    """
    if ',' in arg_str:
        values = [float(v) for v in arg_str.split(',')]
        if len(values) != len(names):
            raise ValueError(f"Expected {len(names)} values, got {len(values)}")
        return dict(zip(names, values))
    else:
        value = float(arg_str)
        return {name: value for name in names}


def main():
    parser = argparse.ArgumentParser(
        description='QoI-preserving compression for LASSO-ENA turbulence data'
    )
    parser.add_argument(
        '--input-dir', '-i',
        type=Path,
        required=True,
        help='Input directory containing binary field files'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        required=True,
        help='Output directory for compressed files'
    )
    parser.add_argument(
        '--field-bounds', '-f',
        type=str,
        default='1e-3',
        help='Relative error bounds for fields (U,V,W,PP,TABS,QV). '
             'Single value or comma-separated. Default: 1e-3'
    )
    parser.add_argument(
        '--qoi-bounds', '-q',
        type=str,
        default='1e-3',
        help='Relative error bounds for QoIs (wpup,wpvp,wppp,uvar,vvar,wvar,wskew,wptp,wpqvp). '
             'Single value or comma-separated. Default: 1e-3'
    )
    parser.add_argument(
        '--skip-verify',
        action='store_true',
        help='Skip verification step'
    )
    parser.add_argument(
        '--use-relative',
        action='store_true',
        help='Use relative error bounds for compression (default: absolute)'
    )

    args = parser.parse_args()

    # Parse bounds
    user_field_bounds = parse_bounds_arg(args.field_bounds, FIELD_NAMES)
    user_qoi_bounds = parse_bounds_arg(args.qoi_bounds, QOI_NAMES)

    print("=" * 60)
    print("QoI-Preserving Compression")
    print("=" * 60)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Field bounds: {args.field_bounds}")
    print(f"QoI bounds:   {args.qoi_bounds}")
    print()

    # Load fields
    print("Loading fields...")
    fields = load_binary_fields(args.input_dir)
    shapes = {name: field.shape for name, field in fields.items()}
    print()

    # Calculate error bounds
    print("Calculating field error bounds from QoI constraints...")
    calculated_bounds = calculate_field_bounds(
        fields, user_field_bounds, user_qoi_bounds,
        return_relative=args.use_relative
    )
    print(report_bounds(
        fields, user_field_bounds, user_qoi_bounds, calculated_bounds,
        is_relative=args.use_relative
    ))
    print()

    # Compress
    print("Compressing fields...")
    compressed = compress_fields(
        fields, calculated_bounds, args.output_dir,
        use_relative=args.use_relative
    )
    print()

    # Summary
    total_orig = sum(f.nbytes for f in fields.values())
    total_comp = sum(len(c[0]) for c in compressed.values())
    print(f"Total compression ratio: {total_orig / total_comp:.2f}x")
    print(f"Original size:   {total_orig:,} bytes ({total_orig / 1e6:.1f} MB)")
    print(f"Compressed size: {total_comp:,} bytes ({total_comp / 1e6:.1f} MB)")
    print()

    # Verify
    if not args.skip_verify:
        print("Decompressing for verification...")
        decompressed = decompress_fields(compressed, shapes)
        all_passed = verify_bounds(fields, decompressed, user_field_bounds, user_qoi_bounds)
        return 0 if all_passed else 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
