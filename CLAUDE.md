# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Lossy compression research for LASSO-ENA atmospheric simulation data (SAM model output).

## Documentation

- [SAM3D Data Documentation](docs/sam3d_data.md) - Source data metadata, variables, grid specification
- [Derived Variables Documentation](docs/derived_variables.md) - Turbulence statistics and derivation equations

## Data Location

- **Individual timesteps**: `/archives/disk1/hi_res_sim/data/enalasso_sam3d_20160410era5d25x100_sbmwrm-aer1-flxsstC1.m1/`
  - 96 NetCDF files, 15-min intervals, ~400 MB each (38.44 GB total)
- **Merged 4D file**: `/archives/disk1/hi_res_sim/data/enalasso_sam3d_20160410era5d25x100_sbmwrm-aer1-flxsstC1.m1.20160410.nc.v0`
  - Single NetCDF with all 96 timesteps, 80 GB
  - Variables shape: (96, 260, 256, 256)
- Grid: 256×256×260 (x, y, z), 100m horizontal resolution, 25.6 km domain

## Commands

```bash
# Build and install SZ3 compressor (from project root)
mkdir -p external/sz3-build && cd external/sz3-build
cmake ../SZ3 -DCMAKE_INSTALL_PREFIX=../sz3-install -DCMAKE_INSTALL_RPATH='$ORIGIN/../lib' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
make -j$(nproc) && make install
cd ../..

# SZ3 executable
./external/sz3-install/bin/sz3 --help

# Or use Python API
pip install pysz

# Run the turbulence statistics processing script
python3 python_src/process_3dles.py
```

## Dependencies

- [SZ3](https://github.com/szcompressor/SZ3) - Modular error-bounded lossy compression framework (submodule in `external/SZ3`)

## Quick Reference: Derived Variables

| Variable | Equation | Description |
|----------|----------|-------------|
| `wpup` | ⟨w'u'⟩ | Zonal momentum flux |
| `wpvp` | ⟨w'v'⟩ | Meridional momentum flux |
| `wppp` | ⟨w'p'⟩ | Pressure-velocity covariance |
| `uvar` | ⟨u'²⟩ | Zonal velocity variance |
| `vvar` | ⟨v'²⟩ | Meridional velocity variance |
| `wvar` | ⟨w'²⟩ | Vertical velocity variance |
| `wskew` | ⟨((w'-μ)/σ)³⟩ | Vertical velocity skewness |
| `wptp` | ⟨w'T'⟩ | Sensible heat flux |
| `wpqvp` | ⟨w'qv'⟩ | Moisture flux |

See [derived_variables.md](docs/derived_variables.md) for full equations and physical interpretation.
