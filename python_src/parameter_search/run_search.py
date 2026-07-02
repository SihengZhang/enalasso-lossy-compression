#!/usr/bin/env python3
"""CLI driver for the QoI-preserving compression parameter search.

Examples
--------
Single compressor::

    python3 python_src/parameter_search/run_search.py \
        --compressor sz3 --qoi-bound 1e-2 --report /tmp/sz3.json

Compare all three compressors with a per-QoI override::

    python3 python_src/parameter_search/run_search.py \
        --compare --qoi-bound 1e-2 --qoi-bounds wppp=1e-3,wskew=2e-2
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running as a script (python python_src/parameter_search/run_search.py)
# as well as a module (python -m parameter_search.run_search).
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from parameter_search.compressors import available_compressors, get_compressor
    from parameter_search.qoi_eval import FIELD_NAMES, QOI_NAMES, load_binary_fields
    from parameter_search.search import run_search
else:
    from .compressors import available_compressors, get_compressor
    from .qoi_eval import FIELD_NAMES, QOI_NAMES, load_binary_fields
    from .search import run_search

DEFAULT_INPUT_DIR = (
    "/archives/disk1/hi_res_sim/data/"
    "enalasso_sam3d_20160410era5d25x100_sbmwrm-aer1-flxsstC1.m1."
    "20160410.001500.nc.v0_binary"
)


def parse_qoi_bounds(uniform: float, overrides: str) -> dict:
    """Build a per-QoI bound dict from a uniform default + 'name=val,...'."""
    bounds = {q: uniform for q in QOI_NAMES}
    if overrides:
        for item in overrides.split(','):
            item = item.strip()
            if not item:
                continue
            name, _, val = item.partition('=')
            name = name.strip()
            if name not in bounds:
                raise ValueError(f"Unknown QoI '{name}'. Choices: {QOI_NAMES}")
            bounds[name] = float(val)
    return bounds


def print_single_report(res: dict) -> None:
    print(f"\n=== Compressor: {res['compressor']} ===")
    print("Per-field error bounds:")
    print(f"  {'field':6s} {'abs_eb':>12s} {'rel_eb':>12s} {'ratio':>8s}  status")
    for f in FIELD_NAMES:
        d = res['fields'][f]
        print(f"  {f:6s} {d['abs_eb']:12.4e} {d['rel_eb']:12.4e} "
              f"{d['ratio']:8.2f}  {d['status']}")
    print("Per-QoI verification (relative error vs bound):")
    print(f"  {'qoi':7s} {'rel_err':>12s} {'bound':>12s}  pass")
    for q in QOI_NAMES:
        d = res['qois'][q]
        mark = 'PASS' if d['pass'] else 'FAIL'
        print(f"  {q:7s} {d['rel_err']:12.4e} {d['bound']:12.4e}  {mark}")
    print(f"Total compression ratio: {res['total_ratio']:.3f}x "
          f"({res['orig_bytes']} -> {res['compressed_bytes']} bytes)")
    if res.get('lossless_fields'):
        print(f"Fields stored losslessly (infeasible QoI fallback): "
              f"{res['lossless_fields']}")
    if res['infeasible_qois']:
        print(f"Infeasible QoIs (unmet even with lossless fields): "
              f"{res['infeasible_qois']}")
    print(f"All QoIs within bound: {res['all_pass']}")
    if res['verification_failures']:
        print(f"Final verification failures: {res['verification_failures']}")


def print_compare_report(results: dict) -> None:
    names = list(results.keys())
    print("\n=== Comparison: per-field absolute error bound ===")
    header = f"  {'field':6s}" + "".join(f"{n:>14s}" for n in names)
    print(header)
    for f in FIELD_NAMES:
        row = f"  {f:6s}"
        for n in names:
            row += f"{results[n]['fields'][f]['abs_eb']:14.4e}"
        print(row)

    print("\n=== Comparison: total compression ratio ===")
    best = max(names, key=lambda n: results[n]['total_ratio'])
    for n in names:
        flag = "  <-- best" if n == best else ""
        allp = "all QoIs pass" if results[n]['all_pass'] else "QoI FAIL"
        print(f"  {n:6s} {results[n]['total_ratio']:8.3f}x   ({allp}){flag}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input-dir', default=DEFAULT_INPUT_DIR,
                    help='Directory with {NAME}_{dims}.raw fields')
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument('--compressor', choices=available_compressors(),
                     help='Run a single compressor backend')
    grp.add_argument('--compare', action='store_true',
                     help='Run all three compressors and compare')
    ap.add_argument('--qoi-bound', type=float, default=1e-2,
                    help='Uniform relative QoI error bound (default 1e-2)')
    ap.add_argument('--qoi-bounds', default='',
                    help="Per-QoI overrides, e.g. 'wppp=1e-3,wskew=2e-2'")
    ap.add_argument('--max-iters', type=int, default=20,
                    help='Binary-search iterations per field (default 20)')
    ap.add_argument('--margin', type=float, default=0.9,
                    help='Safety margin applied to converged bounds (default 0.9)')
    ap.add_argument('--report', default=None, help='Write JSON report to this path')
    ap.add_argument('--quiet', action='store_true', help='Suppress progress logs')
    args = ap.parse_args(argv)

    qoi_bounds = parse_qoi_bounds(args.qoi_bound, args.qoi_bounds)
    verbose = not args.quiet

    print(f"Loading fields from {args.input_dir} ...", flush=True)
    fields = load_binary_fields(args.input_dir)
    shape = next(iter(fields.values())).shape
    print(f"Loaded {len(fields)} fields, shape {shape}")
    print(f"QoI bounds: {qoi_bounds}")

    search_kwargs = dict(max_iters=args.max_iters, margin=args.margin,
                         verbose=verbose)

    if args.compare:
        targets = available_compressors()
    else:
        targets = [args.compressor or 'sz3']

    results = {}
    for name in targets:
        print(f"\n--- Running parameter search with {name} ---", flush=True)
        compressor = get_compressor(name)
        res = run_search(compressor, fields, qoi_bounds, **search_kwargs)
        results[name] = res
        print_single_report(res)

    if len(results) > 1:
        print_compare_report(results)

    if args.report:
        with open(args.report, 'w') as fh:
            json.dump(results, fh, indent=2)
        print(f"\nWrote JSON report to {args.report}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
