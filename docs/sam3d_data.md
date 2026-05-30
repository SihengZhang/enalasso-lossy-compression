# SAM3D Data Documentation

## Overview

LASSO-ENA (Large-Eddy Simulation ARM Symbiotic Simulation and Observation - Eastern North Atlantic) simulation output from the SAM (System for Atmospheric Modeling) model.

## Data Source

| Property | Value |
|----------|-------|
| **Model** | SAM v6.10.3 with LASSO modifications |
| **Site** | Eastern North Atlantic (ENA) Atmospheric Observatory |
| **Location** | Graciosa Island, Azores |
| **Simulation Date** | April 10, 2016 |
| **Forcing** | ERA5 reanalysis |
| **Configuration** | `sbmwrm-aer1-flxsstC1` |

## File Information

| Property | Value |
|----------|-------|
| **Path** | `/archives/disk1/hi_res_sim/data/enalasso_sam3d_20160410era5d25x100_sbmwrm-aer1-flxsstC1.m1/` |
| **Format** | NetCDF4 (`.nc.v0`) |
| **Total Files** | 96 |
| **Time Resolution** | 15 minutes |
| **Time Span** | 00:15 - 24:00 UTC |
| **Total Size** | 38.44 GB |
| **Per File Size** | ~400 MB |

## Grid Specification

| Dimension | Points | Range | Resolution | Units |
|-----------|--------|-------|------------|-------|
| x | 256 | 0 - 25,500 | 100 | m |
| y | 256 | 0 - 25,500 | 100 | m |
| z | 260 | 12.5 - 8,087.5 | variable | m |
| time | 1 per file | - | 15 min | days |

**Domain Size**: 25.6 km × 25.6 km × 8.1 km

## Coordinate Variables

| Variable | Dimensions | Units | Description |
|----------|------------|-------|-------------|
| `x` | (256,) | m | X-coordinate |
| `y` | (256,) | m | Y-coordinate |
| `z` | (260,) | m | Height above surface |
| `time` | (1,) | days | Time since simulation start |
| `p` | (260,) | mb | Pressure at each level |

### Pressure Profile

| Level | Height (m) | Pressure (mb) |
|-------|------------|---------------|
| Surface | 12.5 | 1025.5 |
| Mid-level | 2,512.5 | 755.2 |
| Top | 8,087.5 | 361.8 |

## 3D Variables

All 3D variables have shape `(time, z, y, x)` = `(1, 260, 256, 256)` and dtype `float32`.

### Wind Components

| Variable | Long Name | Units | Typical Range |
|----------|-----------|-------|---------------|
| `U` | X Wind Component | m/s | 3 - 19 |
| `V` | Y Wind Component | m/s | -22 - -3 |
| `W` | Z Wind Component | m/s | -7 - 8 |

### Thermodynamic Variables

| Variable | Long Name | Units | Typical Range |
|----------|-----------|-------|---------------|
| `TABS` | Absolute Temperature | K | 235 - 287 |
| `PP` | Pressure Perturbation | Pa | -1 - 1 |
| `QRAD` | Radiative Heating Rate | K/day | -52 - 1 |

### Water Species

| Variable | Long Name | Units | Sparsity |
|----------|-----------|-------|----------|
| `QV` | Water Vapor | g/kg | Dense (0%) |
| `QCL` | Cloud Liquid Water | g/kg | 97% zeros |
| `QCI` | Cloud Ice Water | g/kg | 100% zeros |
| `QPL` | Precipitating Liquid Water | g/kg | 100% zeros |
| `QPI` | Precipitating Ice Water | g/kg | 100% zeros |

### Effective Radii

| Variable | Long Name | Units | Sparsity |
|----------|-----------|-------|----------|
| `REL` | Effective Radius for Cloud Liquid | μm | 86% zeros |
| `REI` | Effective Radius for Cloud Ice | μm | 100% zeros |

## Data Characteristics

### Memory Footprint

- **Points per 3D variable**: 17,039,360 (256 × 256 × 260)
- **Bytes per variable**: 68.2 MB (float32)
- **Memory per timestep** (13 vars): 886 MB
- **Uncompressed full dataset**: 85.1 GB

### Compression Characteristics

| Category | Variables | Notes |
|----------|-----------|-------|
| Dense, high entropy | U, V, W, PP, QRAD, QV | Millions of unique values |
| Dense, low entropy | TABS | ~70K unique values, good for quantization |
| Sparse | QCL, REL | 86-97% zeros |
| All zeros | QCI, QPL, QPI, REI | Can use sparse encoding |

### NetCDF Chunking

All 3D variables use chunking: `[1, 87, 86, 86]`

## Global Attributes

```
product_description: LASSO-ENA simulation from SAM model
sim_name: 20160410era5d25x100_sbmwrm-aer1-flxsst
model_type: SAM v6.10.3 plus LASSO modifications
site_id: ena
platform_id: lasso
facility_designation: C1
data_level: m1
contact: lasso@arm.gov
```

## References

- **LASSO-ENA DOI**: https://doi.org/10.5439/2572639
- **Documentation DOI**: https://doi.org/10.5439/2572640
- **Model Source**: https://code.arm.gov/lasso/lasso-ena-codes/lasso_sam_sbm/-/tree/lasso_ena_noice
- **Processing Code**: https://code.arm.gov/lasso/lasso-ena-codes/stagesam_ena.git

## Example: Reading Data

```python
import netCDF4 as nc

filepath = '/archives/disk1/hi_res_sim/data/enalasso_sam3d_20160410era5d25x100_sbmwrm-aer1-flxsstC1.m1/'
filename = 'enalasso_sam3d_20160410era5d25x100_sbmwrm-aer1-flxsstC1.m1.20160410.001500.nc.v0'

ds = nc.Dataset(filepath + filename, 'r')
u = ds['U'][0, :, :, :]  # Shape: (260, 256, 256)
z = ds['z'][:]           # Height levels
ds.close()
```
