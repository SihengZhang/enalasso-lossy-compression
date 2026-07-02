#!/usr/bin/env python3
"""CLI: sparse QoI-preserving edit of an SZ3-decompressed W field for wskew.

Pipeline
--------
1. Load ground-truth ``W`` (binary dir).
2. Build the base ``X' = SZ3_decompress(SZ3(W, eb_data))`` (already satisfies the
   data box ``|X'-W| <= eb_data``).
3. Run :class:`SparseQoIEditor` to produce ``X^`` that jointly satisfies the data
   box AND the wskew bound with as few point edits as possible.
4. INDEPENDENTLY verify with the numpy QoI implementation (not the torch graph)
   and report C1 / C2 / sparsity. Save ``X^`` and plots.

Example
-------
    python3 python_src/sparse_qoi_edit/run.py \
      --data "data/enalasso_..._001500.nc.v0_binary" \
      --eb-data 1e-5 --tau 1e-2 --device cuda:0 --out /tmp/sparse_qoi_W
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Make the sibling packages (parameter_search) importable.
_PYSRC = Path(__file__).resolve().parents[1]
if str(_PYSRC) not in sys.path:
    sys.path.insert(0, str(_PYSRC))

from parameter_search.qoi_eval import (          # noqa: E402
    load_binary_fields, compute_single_qoi)
from parameter_search.compressors import get_compressor   # noqa: E402

from sparse_edit import SparseQoIEditor, SparseQoIConfig   # noqa: E402


_DEFAULT_DATA = ("data/enalasso_sam3d_20160410era5d25x100_sbmwrm-aer1-flxsstC1"
                 ".m1.20160410.001500.nc.v0_binary")


def wskew_np(field: np.ndarray) -> np.ndarray:
    """Independent numpy wskew profile (nz,) for a (nz, ny, nx) W field."""
    return compute_single_qoi("wskew", {"W": field})


def active_mask(field: np.ndarray, sigma_active_rel: float) -> np.ndarray:
    """Per-level active mask matching the editor's criterion (numpy side)."""
    wp = field - field.mean(axis=(1, 2), keepdims=True)
    sig = np.sqrt((wp * wp).mean(axis=(1, 2)))
    return sig > sigma_active_rel * sig.max()


def rel_err(profile_a: np.ndarray, profile_b: np.ndarray,
            vrange: float, mask: np.ndarray) -> float:
    """max_z |a-b| / vrange over masked levels (the repo's QoI metric)."""
    if vrange <= 1e-12:
        return 0.0
    diff = np.abs(profile_a - profile_b)[mask]
    return float(diff.max() / vrange) if diff.size else 0.0


def make_plots(out: Path, z: np.ndarray, wsk_gt, wsk_base, wsk_hat,
               edits_per_level, delta_nonzero):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].plot(wsk_gt, z, label="GT  Q(X)", lw=2)
    axes[0].plot(wsk_base, z, "--", label="SZ3  Q(X')", alpha=0.8)
    axes[0].plot(wsk_hat, z, ":", label="edited  Q(X^)", lw=2)
    axes[0].set_xlabel("wskew"); axes[0].set_ylabel("z level")
    axes[0].set_title("wskew profile"); axes[0].legend()

    axes[1].plot(edits_per_level, z)
    axes[1].set_xlabel("# edited points"); axes[1].set_ylabel("z level")
    axes[1].set_title("edits per z-level")

    if delta_nonzero.size:
        axes[2].hist(delta_nonzero, bins=60)
    axes[2].set_xlabel("|X^ - X'|  (nonzero edits)")
    axes[2].set_ylabel("count")
    axes[2].set_title("edit magnitude histogram")

    fig.tight_layout()
    p = out / "sparse_qoi_analysis.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=_DEFAULT_DATA,
                    help="binary field directory containing W_*.raw")
    ap.add_argument("--eb-data", type=float, default=1e-5,
                    help="absolute SZ3 / data error bound (defines the box). "
                         "Default 1e-5 puts SZ3 at ~9.6x with ~35 wskew-violating "
                         "levels -- a non-trivial but sparse-fixable regime "
                         "(W perturbation sigma is small, ~1e-4 median).")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--tau", type=float, default=1e-2,
                   help="relative-to-range wskew bound (repo convention)")
    g.add_argument("--eb-qoi-abs", type=float, default=None,
                   help="absolute per-level wskew bound (overrides --tau)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--iters", type=int, default=None,
                    help="projected-gradient + ISTA iterations (optimize_iters)")
    ap.add_argument("--polish-iters", type=int, default=None,
                    help="polish steps per bisection probe")
    ap.add_argument("--out", default="/tmp/sparse_qoi_W")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1. Ground truth W.
    print(f"Loading W from {args.data} ...", flush=True)
    W = load_binary_fields(args.data)["W"]            # (nz, ny, nx) float32
    nz, ny, nx = W.shape
    N = ny * nx
    print(f"  W shape={W.shape} dtype={W.dtype}", flush=True)

    # 2. Base X' via SZ3 (absolute error bound).
    print(f"SZ3 round-trip at abs eb={args.eb_data} ...", flush=True)
    sz3 = get_compressor("sz3")
    X_base, comp_bytes = sz3.round_trip(W, args.eb_data)
    sz3_ratio = W.nbytes / comp_bytes
    max_sz3_err = float(np.abs(X_base - W).max())
    print(f"  SZ3 ratio={sz3_ratio:.2f}x  max|X'-W|={max_sz3_err:.3e} "
          f"(<= eb_data={args.eb_data})", flush=True)

    # 3. QoI targets and threshold.
    cfg = SparseQoIConfig(device=args.device, verbose=not args.quiet)
    if args.iters is not None:
        cfg.optimize_iters = args.iters
    if args.polish_iters is not None:
        cfg.polish_iters = args.polish_iters

    wsk_gt = wskew_np(W)                               # (nz,)
    wsk_base = wskew_np(X_base)
    amask = active_mask(W, cfg.sigma_active_rel)
    vrange = float(wsk_gt[amask].max() - wsk_gt[amask].min()) if amask.any() else 0.0

    if args.eb_qoi_abs is not None:
        thr_abs = float(args.eb_qoi_abs)
        bound_desc = f"absolute eb_qoi={thr_abs:.3e}"
    else:
        thr_abs = args.tau * vrange
        bound_desc = (f"tau={args.tau} x range_active={vrange:.3e} "
                      f"-> abs eb_qoi={thr_abs:.3e}")
    print(f"QoI bound: {bound_desc}", flush=True)
    print(f"  active levels={int(amask.sum())}/{nz}  "
          f"X' pre-edit wskew rel-err="
          f"{rel_err(wsk_base, wsk_gt, vrange, amask):.3e} "
          f"(tau={args.tau})", flush=True)

    # 4. Run the editor (batched over z).
    Xg = W.reshape(nz, N)
    Xb = X_base.reshape(nz, N)
    editor = SparseQoIEditor(Xg, Xb, eb_data=args.eb_data, eb_qoi=thr_abs, cfg=cfg)
    result = editor.run()
    X_hat = result["x_hat"].reshape(nz, ny, nx).astype(np.float32)
    st = result["stats"]

    # 5. INDEPENDENT numpy verification.
    max_data_err = float(np.abs(X_hat - W).max())
    wsk_hat = wskew_np(X_hat)
    c2_active = rel_err(wsk_hat, wsk_gt, vrange, amask)
    # repo-style metric over ALL levels (for reference)
    vrange_all = float(wsk_gt.max() - wsk_gt.min())
    c2_all = rel_err(wsk_hat, wsk_gt, vrange_all, np.ones(nz, bool))

    c1_pass = max_data_err <= args.eb_data * (1 + 1e-6)
    c2_pass = (c2_active <= args.tau if args.eb_qoi_abs is None
               else float(np.abs(wsk_hat - wsk_gt)[amask].max()) <= thr_abs)
    edit_frac = st["edit_fraction"]

    # 6. Report.
    print("\n" + "=" * 64)
    print("RESULTS  (independent numpy verification)")
    print("=" * 64)
    print(f"C1 data box : max|X^-W| = {max_data_err:.4e}  "
          f"vs eb_data={args.eb_data:.4e}   -> {'PASS' if c1_pass else 'FAIL'}")
    print(f"C2 wskew    : rel-err(active) = {c2_active:.4e}  "
          f"vs tau={args.tau:.1e}   -> {'PASS' if c2_pass else 'FAIL'}")
    print(f"             (all-levels rel-err = {c2_all:.4e}; "
          f"max abs wskew err = {float(np.abs(wsk_hat-wsk_gt)[amask].max() if amask.any() else 0.0):.3e})")
    print(f"Sparsity    : edits = {st['n_edits']:,} / {nz*N:,}  "
          f"= {edit_frac*100:.4f}%  of points")
    nlev_edited = int((st['edits_per_level'] > 0).sum())
    print(f"             levels edited = {nlev_edited}/{nz}; "
          f"skipped(degenerate) = {st['n_skipped']}")
    print(f"             SZ3 X' pre-edit C2 rel-err = "
          f"{rel_err(wsk_base, wsk_gt, vrange, amask):.4e} "
          f"({'VIOLATES' if rel_err(wsk_base, wsk_gt, vrange, amask) > args.tau else 'already ok -- loosen eb-data'})")
    print(f"Wall time   : {time.time()-t0:.1f}s")
    print("=" * 64)

    # 7. Save artifacts.
    raw_path = out / f"What_{nz}x{ny}x{nx}.raw"
    X_hat.tofile(raw_path)
    z = np.arange(nz)
    plot_path = make_plots(out, z, wsk_gt, wsk_base, wsk_hat,
                           st["edits_per_level"], st["delta_abs_nonzero"])
    report = {
        "data": args.data, "eb_data": args.eb_data, "tau": args.tau,
        "eb_qoi_abs": thr_abs, "sz3_ratio": sz3_ratio,
        "c1_pass": c1_pass, "c2_pass": c2_pass,
        "max_data_err": max_data_err, "c2_rel_err_active": c2_active,
        "c2_rel_err_all_levels": c2_all,
        "x_base_c2_rel_err": rel_err(wsk_base, wsk_gt, vrange, amask),
        "n_edits": st["n_edits"], "edit_fraction": edit_frac,
        "levels_edited": nlev_edited, "n_skipped": st["n_skipped"],
        "edits_per_level": st["edits_per_level"].tolist(),
        "wall_time_s": time.time() - t0,
    }
    with open(out / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {raw_path}\n       {plot_path}\n       {out/'report.json'}")
    return 0 if (c1_pass and c2_pass) else 1


if __name__ == "__main__":
    raise SystemExit(main())
