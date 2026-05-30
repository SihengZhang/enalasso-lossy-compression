#!/usr/bin/env python3
"""Extract NetCDF fields to raw binary files for SZ3 compression."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import netCDF4 as nc

FIELDS = ['U', 'V', 'W', 'PP', 'TABS', 'QV']


def extract_to_binary(nc_path: Path, output_dir: Path) -> None:
    """Extract fields from a NetCDF file to raw binary files."""
    # Create output folder with _binary suffix
    folder_name = f"{nc_path.name}_binary"
    out_folder = output_dir / folder_name
    out_folder.mkdir(parents=True, exist_ok=True)

    with nc.Dataset(nc_path, 'r') as ds:
        for field in FIELDS:
            if field not in ds.variables:
                print(f"  Warning: {field} not found, skipping")
                continue

            var = ds.variables[field]
            data = var[:].astype(np.float32)

            # Squeeze time dimension if size 1
            if 'time' in var.dimensions and data.shape[0] == 1:
                data = data.squeeze(axis=0)

            # Build filename with dimensions
            dims_str = 'x'.join(str(d) for d in data.shape)
            out_file = out_folder / f"{field}_{dims_str}.raw"

            # Write raw binary
            data.tobytes()
            with open(out_file, 'wb') as f:
                f.write(data.tobytes())

            size_mb = out_file.stat().st_size / (1024 * 1024)
            print(f"  {field}: {data.shape} -> {out_file.name} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description='Extract NetCDF fields to raw binary files for SZ3 compression.'
    )
    parser.add_argument(
        'input',
        type=Path,
        nargs='+',
        help='Input NetCDF file(s)'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=Path,
        required=True,
        help='Output directory for binary files'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of files to process'
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_files = args.input
    if args.limit:
        input_files = input_files[:args.limit]

    total = len(input_files)
    print(f"Processing {total} file(s) to {args.output_dir}")
    print(f"Fields: {', '.join(FIELDS)}\n")

    start_time = time.time()
    for i, nc_path in enumerate(input_files, 1):
        if not nc_path.exists():
            print(f"[{i}/{total}] File not found: {nc_path}")
            continue

        print(f"[{i}/{total}] {nc_path.name}")
        file_start = time.time()
        extract_to_binary(nc_path, args.output_dir)
        print(f"  Done in {time.time() - file_start:.1f}s\n")

    print(f"Total time: {time.time() - start_time:.1f}s")


if __name__ == '__main__':
    main()
