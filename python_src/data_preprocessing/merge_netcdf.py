#!/usr/bin/env python3
"""Merge multiple NetCDF files along the time dimension into a single 4D file."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import netCDF4 as nc

DEFAULT_INPUT_DIR = '/archives/disk1/hi_res_sim/data/enalasso_sam3d_cleaned'
DEFAULT_OUTPUT_FILE = '/archives/disk1/hi_res_sim/data/enalasso_sam3d_cleaned_merged.nc'


def merge_netcdf(input_dir: Path, output_file: Path) -> None:
    """Merge all NetCDF files in input_dir along time dimension."""
    input_files = sorted(input_dir.glob('*.nc.v0'))
    if not input_files:
        print(f"No .nc.v0 files found in {input_dir}")
        sys.exit(1)

    n_files = len(input_files)
    print(f"Found {n_files} files to merge")

    # Read first file to get structure
    with nc.Dataset(input_files[0], 'r') as src:
        # Get dimensions (excluding time which will be extended)
        dims = {name: len(dim) for name, dim in src.dimensions.items() if name != 'time'}

        # Get variable info
        var_info = {}
        for var_name, var in src.variables.items():
            var_info[var_name] = {
                'dtype': var.dtype,
                'dimensions': var.dimensions,
                'attrs': {attr: var.getncattr(attr) for attr in var.ncattrs()}
            }

        # Get global attributes
        global_attrs = {attr: src.getncattr(attr) for attr in src.ncattrs()}

    print(f"Dimensions: x={dims['x']}, y={dims['y']}, z={dims['z']}, time={n_files}")
    print(f"Variables: {', '.join(var_info.keys())}")

    # Create output file
    print(f"\nCreating output file: {output_file}")
    with nc.Dataset(output_file, 'w', format='NETCDF4') as dst:
        # Create dimensions
        dst.createDimension('time', None)  # unlimited
        for dim_name, dim_size in dims.items():
            dst.createDimension(dim_name, dim_size)

        # Set global attributes
        dst.setncatts(global_attrs)

        # Create variables
        for var_name, info in var_info.items():
            var = dst.createVariable(
                var_name,
                info['dtype'],
                info['dimensions'],
                zlib=True,
                complevel=4,
            )
            var.setncatts(info['attrs'])

        # Process each file
        start_time = time.time()
        for i, input_path in enumerate(input_files):
            print(f"[{i+1}/{n_files}] Reading: {input_path.name}", end='', flush=True)
            file_start = time.time()

            with nc.Dataset(input_path, 'r') as src:
                for var_name in var_info.keys():
                    src_var = src.variables[var_name]
                    if 'time' in src_var.dimensions:
                        # Append along time dimension
                        dst.variables[var_name][i:i+1] = src_var[:]
                    elif i == 0:
                        # Copy non-time variables only once
                        dst.variables[var_name][:] = src_var[:]

            file_elapsed = time.time() - file_start
            print(f" [{file_elapsed:.1f}s]")

        elapsed = time.time() - start_time
        print(f"\nMerge complete. Total time: {elapsed:.1f}s")

    # Report output file size
    output_size = output_file.stat().st_size / (1024 * 1024 * 1024)
    print(f"Output file size: {output_size:.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description='Merge NetCDF files along time dimension into a single 4D file.'
    )
    parser.add_argument(
        '-i', '--input-dir',
        type=Path,
        default=Path(DEFAULT_INPUT_DIR),
        help=f'Input directory (default: {DEFAULT_INPUT_DIR})'
    )
    parser.add_argument(
        '-o', '--output-file',
        type=Path,
        default=Path(DEFAULT_OUTPUT_FILE),
        help=f'Output file (default: {DEFAULT_OUTPUT_FILE})'
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)

    merge_netcdf(args.input_dir, args.output_file)


if __name__ == '__main__':
    main()
