# sparse_qoi_edit — joint data + QoI bounding by sparse edits

A feasibility test for one question:

> Starting from a **loosely** SZ3-compressed field `X' = SZ3⁻¹(SZ3(X, eb_data))`,
> can a gradient-based optimizer make a **small number of edits** so the corrected
> field `X^` satisfies **both** a point-wise data error bound **and** a QoI error
> bound *at the same time*?

Tested on `X = W` (vertical velocity) and `Q = wskew` (vertical-velocity
skewness), the QoI that `docs/compression_parameter_search.md` identifies as the
universal compression bottleneck (a normalized 3rd moment, hyper-sensitive to
small perturbations).

## The idea

Two constraints, on the same field:

* **C1 (data box):** `|X^_i − X_i| ≤ eb_data` for every point `i`.
* **C2 (QoI):**      `|Q(X^)_z − Q(X)_z| ≤ eb_qoi` for every z-level.

`X'` already satisfies C1 (SZ3's ABS guarantee), so we keep `X^` inside the box
`[X − eb_data, X + eb_data]` by **clamping after every step** — C1 then holds by
construction, for free. The ground-truth `X` is the **center** of that box and
satisfies C2 exactly, so a feasible point *always* exists; the only question is
whether a **sparse** edit reaches one. `X^` is a single free-parameter tensor (a
"0-layer neural field") optimized by backprop through the **frozen**,
differentiable `wskew`.

Because `wskew(z)` is a function only of the 256×256 plane at level `z`, the 260
levels are **independent** and the whole thing is one **batched** GPU
optimization (batch = z, 65536 variables per row).

## Algorithm (`sparse_edit.py`)

1. **Projected gradient + adaptive ISTA.** Per-*row*-normalized gradient descent
   on the QoI squared-hinge (per-row normalization equalizes the wildly different
   per-level σ scales — `∂wskew/∂x ∝ 1/σ` — without flattening the within-row
   concentration on the skewness-driving *tail* points that a per-parameter
   optimizer like Adam would destroy), with a geometric **step decay** for precise
   near-boundary convergence, and a **soft-threshold** on `Δ = X^ − X'` (the prox
   of `λ‖·‖₁`, which yields *exact* zeros) ramped while keeping the sparsest
   feasible iterate.
2. **Hard-threshold bisection finalize.** Log-space binary-search a magnitude
   threshold on `|Δ|`: zero the smallest edits, polish the surviving support,
   keep the sparsest iterate that still verifies (mirrors
   `parameter_search/search.py`'s bisection).

Degenerate near-constant levels (σ ≈ 0, e.g. the laminar domain top) are left at
`X'` and auto-pass, matching `qoi_eval`'s div-by-zero guard.

## Run

```bash
python3 python_src/sparse_qoi_edit/run.py \
  --data "data/enalasso_sam3d_20160410era5d25x100_sbmwrm-aer1-flxsstC1.m1.20160410.001500.nc.v0_binary" \
  --eb-data 1e-5 --tau 1e-2 --device cuda:0 --out /tmp/sparse_qoi_W
```

Key flags: `--eb-data` (abs SZ3 / data bound), `--tau` (wskew bound relative to
its z-range, repo convention) or `--eb-qoi-abs` (absolute), `--iters`,
`--polish-iters`, `--device`.

### Why `eb_data = 1e-5`?

`W`'s perturbation σ is tiny (median ≈ 9e-5; max ≈ 0.07), so wskew is fragile.
`eb_data = 1e-5` puts SZ3 at ≈9.6× while leaving ≈35 levels just over the wskew
bound — a **non-trivial but sparse-fixable** regime. Much looser (e.g. `0.02`,
≈480×) makes wskew hopeless (≈87% of points need editing); much tighter makes
`X'` already pass (trivial). The driver prints `X'`'s pre-edit violation so you
can confirm the test is non-trivial.

## What it verifies

The driver re-checks results with the **independent numpy** QoI implementation
(`parameter_search/qoi_eval`), not the torch graph it trained against, and reports:

* **C1** `max|X^ − X|` vs `eb_data`,
* **C2** `max_z |wskew(X^) − wskew(X)| / range` vs `tau` (plus `X'`'s pre-edit
  violation, to show the edit did the work),
* **sparsity** — edit fraction, per-level edit counts, `|Δ|` histogram.

Artifacts in `--out`: `What_<dims>.raw` (float32 `X^`), `report.json`, and
`sparse_qoi_analysis.png` (wskew profiles GT/X'/X^, edits-per-level, `|Δ|`
histogram).

### Representative result (`eb_data=1e-5`, `tau=1e-2`)

| metric | value |
|---|---|
| SZ3 ratio / `X'` pre-edit wskew rel-err | 9.6× / 3.1e-2 (**violates**) |
| C1 `max|X^−X|` | 1.0e-5 ≤ 1e-5 — **PASS** |
| C2 wskew rel-err | < 1e-2 — **PASS** |
| edits | ≈1.4% of points, ≈39 / 260 levels |

So both bounds are met simultaneously by editing a small fraction of points —
the corrected field stays inside the original SZ3 data box yet restores the
fragile skewness QoI.
