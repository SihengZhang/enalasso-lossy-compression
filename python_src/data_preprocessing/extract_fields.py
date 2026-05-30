#!/usr/bin/env python3
"""Extract required fields from SAM3D NetCDF files for turbulence statistics."""

import argparse
import sys
import time
from pathlib import Path

import netCDF4 as nc

REQUIRED_VARS = ['time', 'x', 'y', 'z', 'U', 'V', 'W', 'PP', 'TABS', 'QV']

DEFAULT_INPUT_DIR = '/archives/disk1/hi_res_sim/data/enalasso_sam3d_20160410era5d25x100_sbmwrm-aer1-flxsstC1.m1'
DEFAULT_OUTPUT_DIR = '/archives/disk1/hi_res_sim/data/enalasso_sam3d_cleaned'


def extract_fields(input_path: Path, output_path: Path) -> None:
    """Extract required variables from input NetCDF to output NetCDF."""
    with nc.Dataset(input_path, 'r') as src:
        with nc.Dataset(output_path, 'w', format=src.file_format) as dst:
            # Copy dimensions
            for dim_name, dim in src.dimensions.items():
                dst.createDimension(dim_name, None if dim.isunlimited() else len(dim))

            # Copy global attributes
            dst.setncatts({attr: src.getncattr(attr) for attr in src.ncattrs()})

            # Copy only required variables
            for var_name in REQUIRED_VARS:
                if var_name not in src.variables:
                    print(f"  Warning: variable '{var_name}' not found in {input_path.name}")
                    continue

                src_var = src.variables[var_name]
                dst_var = dst.createVariable(
                    var_name,
                    src_var.datatype,
                    src_var.dimensions,
                    zlib=True,
                    complevel=4,
                )
                # Copy variable attributes
                dst_var.setncatts({attr: src_var.getncattr(attr) for attr in src_var.ncattrs()})
                # Copy data
                dst_var[:] = src_var[:]


def main():
    parser = argparse.ArgumentParser(
        description='Extract required fields from SAM3D NetCDF files for turbulence statistics.'
    )
    parser.add_argument(
        '-i', '--input-dir',
        type=Path,
        default=Path(DEFAULT_INPUT_DIR),
        help=f'Input directory (default: {DEFAULT_INPUT_DIR})'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing output files'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of files to process (for testing)'
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(args.input_dir.glob('*.nc.v0'))
    if not input_files:
        print(f"No .nc.v0 files found in {args.input_dir}")
        sys.exit(1)

    if args.limit:
        input_files = input_files[:args.limit]

    total = len(input_files)
    print(f"Processing {total} files...")
    print(f"  Input:  {args.input_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Variables: {', '.join(REQUIRED_VARS)}")
    print()

    start_time = time.time()
    processed = 0
    skipped = 0

    for i, input_path in enumerate(input_files, 1):
        output_path = args.output_dir / input_path.name

        if output_path.exists() and not args.overwrite:
            print(f"[{i}/{total}] Skipping (exists): {input_path.name}")
            skipped += 1
            continue

        print(f"[{i}/{total}] Processing: {input_path.name}", end='', flush=True)
        file_start = time.time()

        extract_fields(input_path, output_path)

        file_elapsed = time.time() - file_start
        input_size = input_path.stat().st_size / (1024 * 1024)
        output_size = output_path.stat().st_size / (1024 * 1024)
        ratio = output_size / input_size * 100

        print(f" -> {output_size:.1f} MB ({ratio:.1f}% of {input_size:.1f} MB) [{file_elapsed:.1f}s]")
        processed += 1

    elapsed = time.time() - start_time
    print()
    print(f"Done. Processed {processed} files, skipped {skipped}. Total time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
