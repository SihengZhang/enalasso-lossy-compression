# Derived Variables Documentation

## Overview

Turbulence statistics derived from SAM3D LES output using Reynolds decomposition. These variables characterize the turbulent transport of momentum, heat, and moisture in the atmospheric boundary layer.

## Reynolds Decomposition

All derived variables use Reynolds decomposition, separating each field into mean and perturbation components:

$$\phi(x, y, z, t) = \overline{\phi}(z, t) + \phi'(x, y, z, t)$$

Where:
- $\overline{\phi}(z, t) = \langle \phi \rangle_{xy}$ is the horizontal mean at each height level
- $\phi'$ is the turbulent perturbation (deviation from the mean)

### Implementation (LINEAR)

All perturbation variables are **linear** transformations:

```python
# Compute perturbation by removing horizontal mean
up = u - u.mean(axis=(0, 2, 3), keepdims=True)   # LINEAR: u' = u - ⟨u⟩
vp = v - v.mean(axis=(0, 2, 3), keepdims=True)   # LINEAR: v' = v - ⟨v⟩
wp = w - w.mean(axis=(0, 2, 3), keepdims=True)   # LINEAR: w' = w - ⟨w⟩
tp = t - t.mean(axis=(0, 2, 3), keepdims=True)   # LINEAR: T' = T - ⟨T⟩
qvp = qv - qv.mean(axis=(0, 2, 3), keepdims=True) # LINEAR: qv' = qv - ⟨qv⟩
```

## Linearity Summary

| Variable | Type | Linear? | Operation |
|----------|------|---------|-----------|
| `up` (u') | Intermediate | **Yes** | u - mean(u) |
| `vp` (v') | Intermediate | **Yes** | v - mean(v) |
| `wp` (w') | Intermediate | **Yes** | w - mean(w) |
| `tp` (T') | Intermediate | **Yes** | T - mean(T) |
| `qvp` (qv') | Intermediate | **Yes** | qv - mean(qv) |
| `wpup` | Derived | No | mean(w' × u') |
| `wpvp` | Derived | No | mean(w' × v') |
| `wppp` | Derived | No | mean(w' × p') |
| `uvar` | Derived | No | mean(u'²) |
| `vvar` | Derived | No | mean(v'²) |
| `wvar` | Derived | No | mean(w'²) |
| `wskew` | Derived | No | mean((w'/σ)³) |
| `wptp` | Derived | No | mean(w' × T') |
| `wpqvp` | Derived | No | mean(w' × qv') |

**Linear variables**: Perturbations (primed variables) are linear transformations of the original fields — they preserve superposition and are computed via subtraction of the horizontal mean.

**Nonlinear variables**: All final statistics involve products or powers of perturbations, making them nonlinear functions of the original data.

---

## Derived Variables

### 1. Momentum Fluxes (NONLINEAR)

Turbulent momentum fluxes represent the vertical transport of horizontal momentum by turbulent eddies. All are **nonlinear** (products of perturbations).

#### `wpup` - Zonal Momentum Flux

$$\overline{w'u'}(z,t) = \langle w'(x,y,z,t) \cdot u'(x,y,z,t) \rangle_{xy}$$

| Property | Value |
|----------|-------|
| **Units** | m²/s² |
| **Physical meaning** | Vertical transport of eastward momentum |
| **Sign convention** | Positive = upward flux of eastward momentum |
| **Typical values** | -0.1 to 0.1 m²/s² |

```python
wpup[i,:] = np.mean(wp * up, axis=(0, 2, 3))
```

#### `wpvp` - Meridional Momentum Flux

$$\overline{w'v'}(z,t) = \langle w'(x,y,z,t) \cdot v'(x,y,z,t) \rangle_{xy}$$

| Property | Value |
|----------|-------|
| **Units** | m²/s² |
| **Physical meaning** | Vertical transport of northward momentum |
| **Sign convention** | Positive = upward flux of northward momentum |

```python
wpvp[i,:] = np.mean(wp * vp, axis=(0, 2, 3))
```

#### `wppp` - Pressure-Velocity Covariance

$$\overline{w'p'}(z,t) = \langle w'(x,y,z,t) \cdot p'(x,y,z,t) \rangle_{xy}$$

| Property | Value |
|----------|-------|
| **Units** | Pa·m/s |
| **Physical meaning** | Pressure work / pressure transport term in TKE budget |
| **Note** | Uses pressure perturbation (PP), not absolute pressure |

```python
wppp[i,:] = np.mean(wp * pp, axis=(0, 2, 3))
```

---

### 2. Velocity Variances (NONLINEAR)

Velocity variances are components of the turbulent kinetic energy (TKE). All are **nonlinear** (squares of perturbations).

$$\text{TKE} = \frac{1}{2}\left(\overline{u'^2} + \overline{v'^2} + \overline{w'^2}\right)$$

#### `uvar` - Zonal Velocity Variance

$$\overline{u'^2}(z,t) = \langle u'(x,y,z,t)^2 \rangle_{xy}$$

| Property | Value |
|----------|-------|
| **Units** | m²/s² |
| **Physical meaning** | Intensity of turbulent fluctuations in x-direction |
| **Relation to TKE** | $\frac{1}{2}\overline{u'^2}$ is x-component of TKE |

```python
uvar[i,:] = np.mean(up * up, axis=(0, 2, 3))
```

#### `vvar` - Meridional Velocity Variance

$$\overline{v'^2}(z,t) = \langle v'(x,y,z,t)^2 \rangle_{xy}$$

| Property | Value |
|----------|-------|
| **Units** | m²/s² |
| **Physical meaning** | Intensity of turbulent fluctuations in y-direction |

```python
vvar[i,:] = np.mean(vp * vp, axis=(0, 2, 3))
```

#### `wvar` - Vertical Velocity Variance

$$\overline{w'^2}(z,t) = \langle w'(x,y,z,t)^2 \rangle_{xy}$$

| Property | Value |
|----------|-------|
| **Units** | m²/s² |
| **Physical meaning** | Intensity of vertical turbulent motions |
| **Note** | Key indicator of convective activity |

```python
wvar[i,:] = np.mean(wp * wp, axis=(0, 2, 3))
```

---

### 3. Higher-Order Statistics (NONLINEAR)

#### `wskew` - Vertical Velocity Skewness (NONLINEAR - cubic)

$$S_w(z,t) = \left\langle \left(\frac{w' - \mu_w}{\sigma_w}\right)^3 \right\rangle_{xy}$$

Where:
- $\mu_w = \langle w' \rangle_{xy}$ (should be ~0 by construction)
- $\sigma_w = \sqrt{\langle w'^2 \rangle_{xy}}$ (standard deviation)

| Property | Value |
|----------|-------|
| **Units** | dimensionless |
| **Physical meaning** | Asymmetry between updrafts and downdrafts |
| **Positive skewness** | Narrow, strong updrafts with broad, weak downdrafts (convective) |
| **Negative skewness** | Broad, weak updrafts with narrow, strong downdrafts (stable) |
| **Typical range** | -2 to 2 |

```python
mu = wp.mean(axis=(0, 2, 3), keepdims=True)
sg = wp.std(axis=(0, 2, 3), ddof=0, keepdims=True)
wskew[i,:] = np.mean(((wp - mu) / sg)**3, axis=(0, 2, 3))
```

---

### 4. Thermodynamic Fluxes (NONLINEAR)

All are **nonlinear** (products of perturbations).

#### `wptp` - Sensible Heat Flux

$$\overline{w'T'}(z,t) = \langle w'(x,y,z,t) \cdot T'(x,y,z,t) \rangle_{xy}$$

| Property | Value |
|----------|-------|
| **Units** | K·m/s |
| **Physical meaning** | Vertical turbulent transport of sensible heat |
| **Sign convention** | Positive = upward heat flux (warming aloft) |
| **Conversion to W/m²** | $H = \rho c_p \overline{w'T'}$ where $\rho \approx 1.2$ kg/m³, $c_p \approx 1005$ J/(kg·K) |

```python
wptp[i,:] = np.mean(wp * tp, axis=(0, 2, 3))
```

#### `wpqvp` - Moisture Flux

$$\overline{w'q_v'}(z,t) = \langle w'(x,y,z,t) \cdot q_v'(x,y,z,t) \rangle_{xy}$$

| Property | Value |
|----------|-------|
| **Units** | (g/kg)·(m/s) |
| **Physical meaning** | Vertical turbulent transport of water vapor |
| **Sign convention** | Positive = upward moisture flux |
| **Relation to latent heat** | $LE = \rho L_v \overline{w'q_v'} \times 10^{-3}$ where $L_v \approx 2.5 \times 10^6$ J/kg |

```python
wpqvp[i,:] = np.mean(wp * qvp, axis=(0, 2, 3))
```

---

## Output Arrays

All derived variables have shape `(n_files, n_levels)` = `(96, 260)`:

| Variable | Shape | Description |
|----------|-------|-------------|
| `time` | (96,) | Time in days since simulation start |
| `wpup` | (96, 260) | Zonal momentum flux profile |
| `wpvp` | (96, 260) | Meridional momentum flux profile |
| `wppp` | (96, 260) | Pressure-velocity covariance profile |
| `uvar` | (96, 260) | Zonal velocity variance profile |
| `vvar` | (96, 260) | Meridional velocity variance profile |
| `wvar` | (96, 260) | Vertical velocity variance profile |
| `wskew` | (96, 260) | Vertical velocity skewness profile |
| `wptp` | (96, 260) | Sensible heat flux profile |
| `wpqvp` | (96, 260) | Moisture flux profile |

## Processing Script

**Script**: `process_3dles.py`

**Input**: Raw SAM3D NetCDF files (U, V, W, PP, TABS, QV)

**Output**: Three plot files in `plots/`:
- `entropy_flux.png` - Heat and moisture fluxes (wptp, wpqvp)
- `momentum_flux.png` - Momentum fluxes (wpup, wpvp, wppp)
- `turbstats.png` - Variances and skewness (uvar, vvar, wvar, wskew)

## Physical Context

These derived variables are fundamental for:

1. **Boundary layer characterization**: wvar and wskew indicate convective vs. stable conditions
2. **Surface flux estimation**: wptp and wpqvp at lowest levels relate to surface heat/moisture exchange
3. **Turbulence closure validation**: Used to validate and tune turbulence parameterizations in coarser models
4. **Cloud-turbulence interaction**: Fluxes near cloud base/top reveal entrainment processes

## References

- Stull, R. B. (1988). *An Introduction to Boundary Layer Meteorology*. Kluwer Academic Publishers.
- Wyngaard, J. C. (2010). *Turbulence in the Atmosphere*. Cambridge University Press.
