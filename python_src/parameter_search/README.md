# parameter_search

Empirical search for the **highest (loosest) per-field error bound** that still
keeps every derived turbulence QoI within a user-specified relative error bound,
for three interchangeable error-bounded compressors: **SZ3**, **zfp**, **SPERR**.

Unlike `qpet/` (which derives field bounds analytically via error-propagation
formulas), this framework drives the search *empirically*: compress a field at a
candidate bound → decompress → recompute the QoI → check it against the QoI
bound → binary-search the bound. This package is **self-contained** and does not
import `qpet`.

## How it works

Each QoI depends only on its own fields' perturbations (the horizontal mean is
per-field), so a per-field search is well-posed. Search proceeds in stages:

1. **Single-field QoIs** — binary-search `U` via `uvar`, `V` via `vvar`, and `W`
   via both `wvar` and `wskew` (tighter wins). `W` is the universal partner of
   the bilinear QoIs, so it is locked here and held fixed afterwards.
2. **Multi-field QoIs (W fixed)** — search `PP` via `wppp`, `TABS` via `wptp`,
   `QV` via `wpqvp`; re-search `U`/`V` via `wpup`/`wpvp` and keep the tighter
   bound.
3. **W-coupling fallback** — if a bilinear QoI is unmet even with its partner
   near-lossless (W's own error exhausts the budget), tighten `W` and restart
   stage 2 (tightening W only improves every QoI).
4. **Final verification** — compress all six fields at the final bounds,
   recompute all nine QoIs, and report pass/fail + total compression ratio.
5. **Lossless fallback** — if a QoI is *infeasible* (fails even with its fields
   near-lossless — e.g. `wskew`, a hypersensitive normalized third moment), the
   fields that QoI depends on are stored **losslessly** so the QoI is satisfied
   exactly. In practice this forces `W` to lossless. Lossless uses each
   backend's native mode (zfp reversible, SZ3 error-bound 0) and falls back to
   raw storage where there is none (SPERR). Such fields are reported with
   `status = 'lossless'` and listed under `lossless_fields`.

Binary search runs in log space with capped iterations and a safety margin
(default 0.9) to absorb compressor non-monotonicity; the final all-fields
verification is the authoritative check.

## QoI → field dependencies

| QoI | fields | kind |
|-----|--------|------|
| `uvar` | U | single |
| `vvar` | V | single |
| `wvar`, `wskew` | W | single |
| `wpup` | W, U | multi |
| `wpvp` | W, V | multi |
| `wppp` | W, PP | multi |
| `wptp` | W, TABS | multi |
| `wpqvp` | W, QV | multi |

QoI error metric: `max_z |qoi_orig(z) − qoi_test(z)| / range_z(qoi_orig) ≤ τ`.

## Usage

```bash
cd /home/szhang/enalasso-lossy-compression
# zfpy needs its install dir on PYTHONPATH (the package also auto-adds it):
export PYTHONPATH="$PWD/external/zfp-install/lib/python3.13/site-packages:$PYTHONPATH"

# Single compressor
python3 python_src/parameter_search/run_search.py \
    --compressor sz3 --qoi-bound 1e-2 --report /tmp/sz3.json

# Compare all three, with per-QoI overrides
python3 python_src/parameter_search/run_search.py \
    --compare --qoi-bound 1e-2 --qoi-bounds "wppp=1e-3,wskew=2e-2"
```

Key flags: `--input-dir` (default = the `.001500` timestep binary dir),
`--compressor {sz3,zfp,sperr}` **or** `--compare`, `--qoi-bound` (uniform
default 1e-2), `--qoi-bounds name=val,...` (per-QoI overrides), `--max-iters`,
`--margin`, `--report` (JSON), `--quiet`.

## Module layout

| File | Responsibility |
|------|----------------|
| `qoi_eval.py` | QoI formulas, dependency map, value ranges, error metric, binary loader |
| `compressors.py` | `BaseCompressor` ABC + `SZ3Compressor`/`ZfpCompressor`/`SperrCompressor` + `get_compressor` |
| `search.py` | `ParameterSearch`: staged binary search, W fallback, final verification |
| `run_search.py` | CLI driver (single / compare modes, text + JSON reports) |

## Programmatic API

```python
from parameter_search import get_compressor, load_binary_fields, run_search
from parameter_search.qoi_eval import QOI_NAMES

fields = load_binary_fields("/path/to/..._binary")
qoi_bounds = {q: 1e-2 for q in QOI_NAMES}
result = run_search(get_compressor("sz3"), fields, qoi_bounds, verbose=True)
print(result["total_ratio"], result["all_pass"])
```

## Notes

- **SPERR** has no Python binding; it is driven via the `sperr3d` CLI with
  temp files. Its `--dims` takes the fastest-varying dimension first, so a numpy
  array of shape `(nz, ny, nx)` is passed as `--dims nx ny nz`.
- **zfp** often achieves error well below the requested tolerance (safe but
  conservative); the search keys on *measured* QoI error, so this only affects
  the achievable ratio.
- QoIs with ~zero value range auto-pass (avoids divide-by-zero).
