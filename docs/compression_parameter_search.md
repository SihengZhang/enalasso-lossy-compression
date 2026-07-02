# QoI-Preserving Compression: Parameter-Search Results

Results of the empirical per-field error-bound search ([`python_src/parameter_search/`](../python_src/parameter_search/)) run across **9 configurations**: three error-bounded compressors (**SZ3**, **zfp**, **SPERR**) at three relative QoI error bounds (**1e-2**, **1e-3**, **1e-4**).

## Setup

- **Data**: one SAM3D timestep (`...20160410.001500...v0_binary`), six raw float32 fields `U, V, W, PP, TABS, QV`, each shape `(260, 256, 256)`; **408.9 MB** total (`408944640 bytes`).
- **QoIs (9, all bounded)**: bilinear fluxes `wpup, wpvp, wppp, wptp, wpqvp`; variances `uvar, vvar, wvar`; skewness `wskew`.
- **Bound semantics**: the value is a *relative* QoI error, `max_z |qoi_orig − qoi_dec| / range_z(qoi_orig)`, applied uniformly to all 9 QoIs.
- **Method**: for each field, binary-search the highest absolute error bound that keeps every dependent QoI within bound (single-field QoIs first, then multi-field with `W` fixed). A QoI that is infeasible even with its fields near-lossless triggers **lossless storage** of those fields (in practice `W`, driven by the hypersensitive `wskew`).
- **Result**: every one of the 9 configurations satisfies all 9 QoIs (using the lossless fallback where noted).

## 1. Overall compression ratio

| compressor | 1e-2 | 1e-3 | 1e-4 |
|---|---|---|---|
| **SZ3** | 30.52x | 18.30x | 10.87x |
| **zfp** | 6.23x | 5.68x | 5.09x |
| **SPERR** | 15.80x | 13.47x | 5.13x |

Best per bound: **1e-2** → sz3 (30.52x), **1e-3** → sz3 (18.30x), **1e-4** → sz3 (10.87x).

Lossless-`W` fallback (compressor could not preserve `wskew`): SZ3 none, zfp 1e-2/1e-3/1e-4, SPERR 1e-4.

## 2. Per-field compression ratio

### QoI bound = 1e-2

| field | SZ3 | zfp | SPERR |
|---|---|---|---|
| U | 138.89x | 17.16x | 76.73x |
| V | 278.43x | 27.18x | 165.33x |
| W | 5.92x | 1.31x | 3.04x |
| PP | 184.59x | 78.70x | 204.24x |
| TABS | 152.21x | 16.18x | 56.15x |
| QV | 203.23x | 33.19x | 114.86x |
| **overall** | **30.52x** | **6.23x** | **15.80x** |

### QoI bound = 1e-3

| field | SZ3 | zfp | SPERR |
|---|---|---|---|
| U | 68.11x | 14.23x | 37.88x |
| V | 92.07x | 17.45x | 64.07x |
| W | 3.75x | 1.31x | 2.91x |
| PP | 103.36x | 48.36x | 97.92x |
| TABS | 77.01x | 11.90x | 35.00x |
| QV | 78.57x | 16.41x | 47.21x |
| **overall** | **18.30x** | **5.68x** | **13.47x** |

### QoI bound = 1e-4

| field | SZ3 | zfp | SPERR |
|---|---|---|---|
| U | 35.10x | 10.49x | 23.38x |
| V | 37.42x | 12.30x | 29.24x |
| W | 2.41x | 1.31x | 1.00x |
| PP | 38.78x | 30.97x | 50.51x |
| TABS | 34.85x | 8.20x | 25.00x |
| QV | 37.35x | 11.72x | 29.48x |
| **overall** | **10.87x** | **5.09x** | **5.13x** |

## 3. Per-field error bounds

Relative error bound found per field (`abs_eb / field_value_range`). `0` = stored losslessly.

### QoI bound = 1e-2

| field | SZ3 | zfp | SPERR |
|---|---|---|---|
| U | 1.61e-03 | 2.86e-02 | 3.19e-03 |
| V | 3.70e-03 | 9.68e-02 | 5.82e-03 |
| W | 4.13e-06 | lossless | 9.00e-08 |
| PP | 4.90e-03 | 1.44e-01 | 6.11e-03 |
| TABS | 8.65e-05 | 4.77e-03 | 1.44e-04 |
| QV | 2.54e-03 | 1.14e-01 | 4.01e-03 |

### QoI bound = 1e-3

| field | SZ3 | zfp | SPERR |
|---|---|---|---|
| U | 4.97e-04 | 1.43e-02 | 7.39e-04 |
| V | 1.04e-03 | 2.42e-02 | 1.67e-03 |
| W | 5.85e-07 | lossless | 6.56e-08 |
| PP | 1.99e-03 | 3.61e-02 | 1.48e-03 |
| TABS | 3.12e-05 | 1.19e-03 | 4.23e-05 |
| QV | 4.78e-04 | 1.42e-02 | 7.15e-04 |

### QoI bound = 1e-4

| field | SZ3 | zfp | SPERR |
|---|---|---|---|
| U | 1.14e-04 | 3.57e-03 | 1.54e-04 |
| V | 2.07e-04 | 6.05e-03 | 2.61e-04 |
| W | 6.56e-08 | lossless | lossless |
| PP | 2.97e-04 | 9.02e-03 | 2.93e-04 |
| TABS | 6.89e-06 | 1.49e-04 | 1.30e-05 |
| QV | 8.92e-05 | 3.56e-03 | 2.00e-04 |

## 4. Per-QoI achieved relative error

Achieved QoI error vs. the target bound. All entries are within bound (`✓`). Values shown as achieved error; `wvar`/`wskew` reach `0` when `W` is stored losslessly.

### QoI bound = 1e-2 (target = 1e-2 for every QoI)

| QoI | SZ3 | zfp | SPERR |
|---|---|---|---|
| wpup | 4.57e-03 ✓ | 1.29e-03 ✓ | 8.19e-03 ✓ |
| wpvp | 7.74e-03 ✓ | 3.56e-03 ✓ | 7.85e-03 ✓ |
| wppp | 8.30e-03 ✓ | 4.37e-03 ✓ | 7.53e-03 ✓ |
| wptp | 6.70e-03 ✓ | 9.09e-03 ✓ | 8.13e-03 ✓ |
| wpqvp | 9.14e-03 ✓ | 9.32e-03 ✓ | 9.37e-03 ✓ |
| uvar | 8.52e-03 ✓ | 3.43e-03 ✓ | 6.22e-03 ✓ |
| vvar | 7.16e-03 ✓ | 3.86e-03 ✓ | 4.07e-03 ✓ |
| wvar | 2.17e-07 ✓ | 0.00e+00 ✓ | 4.82e-08 ✓ |
| wskew | 4.90e-03 ✓ | 0.00e+00 ✓ | 3.91e-03 ✓ |

### QoI bound = 1e-3 (target = 1e-3 for every QoI)

| QoI | SZ3 | zfp | SPERR |
|---|---|---|---|
| wpup | 4.98e-04 ✓ | 4.96e-04 ✓ | 7.60e-04 ✓ |
| wpvp | 3.49e-04 ✓ | 3.13e-04 ✓ | 9.77e-04 ✓ |
| wppp | 9.96e-04 ✓ | 4.33e-04 ✓ | 9.81e-04 ✓ |
| wptp | 7.37e-04 ✓ | 7.57e-04 ✓ | 8.32e-04 ✓ |
| wpqvp | 9.84e-04 ✓ | 4.22e-04 ✓ | 6.18e-04 ✓ |
| uvar | 7.94e-04 ✓ | 8.48e-04 ✓ | 5.31e-04 ✓ |
| vvar | 7.40e-04 ✓ | 5.63e-04 ✓ | 4.94e-04 ✓ |
| wvar | 9.65e-08 ✓ | 0.00e+00 ✓ | 9.65e-08 ✓ |
| wskew | 6.63e-04 ✓ | 0.00e+00 ✓ | 7.44e-04 ✓ |

### QoI bound = 1e-4 (target = 1e-4 for every QoI)

| QoI | SZ3 | zfp | SPERR |
|---|---|---|---|
| wpup | 6.68e-05 ✓ | 7.69e-05 ✓ | 6.89e-05 ✓ |
| wpvp | 8.97e-05 ✓ | 6.90e-05 ✓ | 8.03e-05 ✓ |
| wppp | 7.03e-05 ✓ | 4.96e-05 ✓ | 9.30e-05 ✓ |
| wptp | 6.35e-05 ✓ | 7.85e-05 ✓ | 7.12e-05 ✓ |
| wpqvp | 4.33e-05 ✓ | 3.96e-05 ✓ | 9.22e-05 ✓ |
| uvar | 6.34e-05 ✓ | 9.93e-05 ✓ | 5.94e-05 ✓ |
| vvar | 4.56e-05 ✓ | 8.76e-05 ✓ | 4.60e-05 ✓ |
| wvar | 9.65e-08 ✓ | 0.00e+00 ✓ | 0.00e+00 ✓ |
| wskew | 7.75e-05 ✓ | 0.00e+00 ✓ | 0.00e+00 ✓ |

## 5. Overall performance comparison

**SZ3 wins at every bound.** SZ3 gives the highest overall ratio in all three regimes (30.5x / 18.3x / 10.9x) — about 1.9x SPERR and 4.9x zfp at the loosest bound. It is also the only compressor that preserves `wskew` at every bound, so it never falls back to lossless `W`; its prediction-based scheme keeps the sensitive higher-order statistic while still compressing the smoother fields hard (e.g. `V` at 278x @1e-2).

**SPERR is a strong second — until `wskew` breaks it.** SPERR trails SZ3 by roughly 2x at the looser bounds (15.8x @1e-2, 13.5x @1e-3) but collapses to 5.13x @1e-4. The cause is `wskew`: SPERR has no lossless mode, so at 1e-4 it must store `W` **raw** (ratio 1.00x), and that bottleneck field drags the overall ratio down to near-zfp territory.

**zfp is not competitive here.** zfp is weakest across the board (6.2x / 5.7x / 5.1x) and must store `W` losslessly (reversible mode, ~1.31x) at *every* bound — its block transform cannot keep velocity skewness within even the loosest 1e-2 bound. Because lossless `W` plus modest per-field ratios dominate its budget, zfp is also the *least* bound-sensitive (only 1.22x spread from 1e-2 to 1e-4, vs 2.81x for SZ3 and 3.08x for SPERR).

**`W` is the universal bottleneck.** Across all 27 field-configs, `W` has the lowest per-field ratio. The driver is `wskew`, a normalized third moment that is hypersensitive to error at low-variance heights; it forces `W`'s bound far tighter than any other field (e.g. SZ3 `W` rel_eb 4.1e-06 vs `PP` 4.9e-03 at 1e-2 — a ~1000x gap). Every lossless fallback in this study is `W`.

**Bound sensitivity & per-field behaviour.** Tightening the QoI bound 10x roughly halves the achievable ratio for SZ3 and SPERR at the looser end, then steepens as raw/lossless `W` kicks in; zfp barely moves. The smooth fields carry most of the compression — `PP` is consistently the best-compressing field, followed by `V`/`U`/`QV`, with `TABS` mid-pack (its large ~290 K offset makes its *relative* bound tiny but its ratio still healthy).

**Recommendation.** For this QoI-preserving task, **SZ3 is the clear choice at every tested bound.** SPERR is a reasonable alternative when the bound is loose (>=1e-3) and lossless `W` is not yet forced, but loses its edge at 1e-4. zfp should be avoided here, primarily because it cannot preserve velocity skewness without storing `W` losslessly.

## 6. Appendix: full per-configuration detail

### SZ3 @ 1e-2

Overall ratio **30.52x** (`408944640` → `13398256` bytes); all QoIs within bound: **True**.

| field | abs_eb | rel_eb | ratio | status |
|---|---|---|---|---|
| U | 2.542e-02 | 1.614e-03 | 138.89x | ok |
| V | 6.878e-02 | 3.698e-03 | 278.43x | ok+wpvp |
| W | 2.391e-06 | 4.129e-06 | 5.92x | ok |
| PP | 7.642e-03 | 4.899e-03 | 184.59x | ok |
| TABS | 4.081e-03 | 8.646e-05 | 152.21x | ok |
| QV | 2.006e-02 | 2.536e-03 | 203.23x | ok |

### zfp @ 1e-2

Overall ratio **6.23x** (`408944640` → `65638616` bytes); all QoIs within bound: **True**; lossless fields: `W`.

| field | abs_eb | rel_eb | ratio | status |
|---|---|---|---|---|
| U | 4.500e-01 | 2.857e-02 | 17.16x | ok |
| V | 1.800e+00 | 9.679e-02 | 27.18x | ok |
| W | 0.000e+00 | 0.000e+00 | 1.31x | lossless |
| PP | 2.250e-01 | 1.442e-01 | 78.70x | ok |
| TABS | 2.250e-01 | 4.766e-03 | 16.18x | ok |
| QV | 9.000e-01 | 1.138e-01 | 33.19x | ok |

### SPERR @ 1e-2

Overall ratio **15.80x** (`408944640` → `25879917` bytes); all QoIs within bound: **True**.

| field | abs_eb | rel_eb | ratio | status |
|---|---|---|---|---|
| U | 5.032e-02 | 3.194e-03 | 76.73x | ok+wpup |
| V | 1.082e-01 | 5.820e-03 | 165.33x | ok+wpvp |
| W | 5.212e-08 | 9.000e-08 | 3.04x | infeasible+tightened |
| PP | 9.536e-03 | 6.114e-03 | 204.24x | ok |
| TABS | 6.800e-03 | 1.441e-04 | 56.15x | ok |
| QV | 3.171e-02 | 4.009e-03 | 114.86x | ok |

### SZ3 @ 1e-3

Overall ratio **18.30x** (`408944640` → `22342224` bytes); all QoIs within bound: **True**.

| field | abs_eb | rel_eb | ratio | status |
|---|---|---|---|---|
| U | 7.824e-03 | 4.967e-04 | 68.11x | ok |
| V | 1.937e-02 | 1.042e-03 | 92.07x | ok |
| W | 3.387e-07 | 5.850e-07 | 3.75x | ok |
| PP | 3.100e-03 | 1.987e-03 | 103.36x | ok |
| TABS | 1.475e-03 | 3.124e-05 | 77.01x | ok |
| QV | 3.783e-03 | 4.783e-04 | 78.57x | ok |

### zfp @ 1e-3

Overall ratio **5.68x** (`408944640` → `72011200` bytes); all QoIs within bound: **True**; lossless fields: `W`.

| field | abs_eb | rel_eb | ratio | status |
|---|---|---|---|---|
| U | 2.250e-01 | 1.428e-02 | 14.23x | ok |
| V | 4.500e-01 | 2.420e-02 | 17.45x | ok |
| W | 0.000e+00 | 0.000e+00 | 1.31x | lossless |
| PP | 5.625e-02 | 3.606e-02 | 48.36x | ok |
| TABS | 5.625e-02 | 1.192e-03 | 11.90x | ok |
| QV | 1.125e-01 | 1.422e-02 | 16.41x | ok |

### SPERR @ 1e-3

Overall ratio **13.47x** (`408944640` → `30361744` bytes); all QoIs within bound: **True**.

| field | abs_eb | rel_eb | ratio | status |
|---|---|---|---|---|
| U | 1.164e-02 | 7.392e-04 | 37.88x | ok+wpup |
| V | 3.111e-02 | 1.673e-03 | 64.07x | ok+wpvp |
| W | 3.799e-08 | 6.561e-08 | 2.91x | infeasible+tightened |
| PP | 2.313e-03 | 1.483e-03 | 97.92x | ok |
| TABS | 1.997e-03 | 4.231e-05 | 35.00x | ok |
| QV | 5.658e-03 | 7.152e-04 | 47.21x | ok |

### SZ3 @ 1e-4

Overall ratio **10.87x** (`408944640` → `37610337` bytes); all QoIs within bound: **True**.

| field | abs_eb | rel_eb | ratio | status |
|---|---|---|---|---|
| U | 1.793e-03 | 1.138e-04 | 35.10x | ok+wpup+tightened |
| V | 3.846e-03 | 2.068e-04 | 37.42x | ok+wpvp |
| W | 3.799e-08 | 6.561e-08 | 2.41x | infeasible+tightened |
| PP | 4.633e-04 | 2.970e-04 | 38.78x | ok |
| TABS | 3.252e-04 | 6.890e-06 | 34.85x | ok |
| QV | 7.057e-04 | 8.922e-05 | 37.35x | ok |

### zfp @ 1e-4

Overall ratio **5.09x** (`408944640` → `80398128` bytes); all QoIs within bound: **True**; lossless fields: `W`.

| field | abs_eb | rel_eb | ratio | status |
|---|---|---|---|---|
| U | 5.625e-02 | 3.571e-03 | 10.49x | ok |
| V | 1.125e-01 | 6.049e-03 | 12.30x | ok |
| W | 0.000e+00 | 0.000e+00 | 1.31x | lossless |
| PP | 1.406e-02 | 9.015e-03 | 30.97x | ok |
| TABS | 7.031e-03 | 1.490e-04 | 8.20x | ok |
| QV | 2.812e-02 | 3.556e-03 | 11.72x | ok |

### SPERR @ 1e-4

Overall ratio **5.13x** (`408944640` → `79791213` bytes); all QoIs within bound: **True**; lossless fields: `W`.

| field | abs_eb | rel_eb | ratio | status |
|---|---|---|---|---|
| U | 2.434e-03 | 1.545e-04 | 23.38x | ok+wpup |
| V | 4.858e-03 | 2.612e-04 | 29.24x | ok+wpvp |
| W | 0.000e+00 | 0.000e+00 | 1.00x | lossless |
| PP | 4.574e-04 | 2.933e-04 | 50.51x | ok |
| TABS | 6.124e-04 | 1.297e-05 | 25.00x | ok |
| QV | 1.578e-03 | 1.995e-04 | 29.48x | ok |

