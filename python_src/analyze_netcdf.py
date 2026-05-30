#!/usr/bin/env python3
"""Analyze metadata, dimensions, and variables of a single NetCDF file."""

import sys
from pathlib import Path
import netCDF4 as nc


def analyze_netcdf(filepath: str) -> None:
    """Analyze and print NetCDF file information."""
    print(f"{'='*60}")
    print(f"NetCDF File Analysis: {Path(filepath).name}")
    print(f"{'='*60}\n")

    with nc.Dataset(filepath, 'r') as ds:
        # Global attributes (metadata)
        print("GLOBAL ATTRIBUTES")
        print("-" * 40)
        if ds.ncattrs():
            for attr in ds.ncattrs():
                value = getattr(ds, attr)
                print(f"  {attr}: {value}")
        else:
            print("  (none)")
        print()

        # Dimensions
        print("DIMENSIONS")
        print("-" * 40)
        for dim_name, dim in ds.dimensions.items():
            unlimited = " (unlimited)" if dim.isunlimited() else ""
            print(f"  {dim_name}: {len(dim)}{unlimited}")
        print()

        # Variables summary
        print("VARIABLES")
        print("-" * 40)
        for var_name, var in ds.variables.items():
            dtype = var.dtype
            dims = var.dimensions
            shape = var.shape
            print(f"  {var_name}")
            print(f"    dtype: {dtype}")
            print(f"    dimensions: {dims}")
            print(f"    shape: {shape}")

            # Variable attributes
            if var.ncattrs():
                for attr in var.ncattrs():
                    value = getattr(var, attr)
                    print(f"    {attr}: {value}")
            print()


def main():
    data_dir = Path("/archives/disk1/hi_res_sim/data/enalasso_sam3d_20160410era5d25x100_sbmwrm-aer1-flxsstC1.m1")

    if len(sys.argv) > 1:
        # Use provided filename
        filepath = Path(sys.argv[1])
        if not filepath.is_absolute():
            filepath = data_dir / filepath
    else:
        # Find the first NetCDF file in the directory (handles .nc and .nc.v0)
        nc_files = sorted(data_dir.glob("*.nc")) + sorted(data_dir.glob("*.nc.v0"))
        if not nc_files:
            print(f"No NetCDF files found in {data_dir}")
            sys.exit(1)
        filepath = nc_files[0]
        print(f"No file specified, using first file: {filepath.name}\n")

    if not filepath.exists():
        print(f"File not found: {filepath}")
        sys.exit(1)

    analyze_netcdf(str(filepath))


if __name__ == "__main__":
    main()
