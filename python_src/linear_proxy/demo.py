#!/usr/bin/env python3
"""Self-tests, cross-validation, and real-data driver for the PyTorch engine.

All tunable settings live in ``config.json`` (next to this file). The program:

1. Self-tests on small planes: the variance/bilinear QoIs match the closed-form
   structured bound exactly (constant Hessian -> structured == dense), and for
   wskew the structured O(N) bound is >= the dense per-entry (sampled) bound and
   neither is violated by Monte-Carlo. Gradients are cross-checked against
   ``torch.func.jacrev``.
2. Driver: load a full x-y plane (256x256) of the raw fields from real .raw/.nc
   data (or a synthetic fallback), set X0 from the data and E = abs_eb, and
   report f(X0), ||grad||, the strict bound t = 0.5 E^T B E, and a Monte-Carlo
   check that |F-g| <= t -- on the GPU in float64.

Run:
    python3 python_src/linear_proxy/demo.py
    python3 python_src/linear_proxy/demo.py --qoi wskew --data <dir_or.nc> --zlevel 5 --abs-eb 0.001
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTHON_SRC = os.path.dirname(_HERE)
for _p in (_HERE, _PYTHON_SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qoi_library import QOI_REGISTRY, QOI_NAMES  # noqa: E402
from engine import LinearProxyEngine, resolve_device  # noqa: E402

DEFAULT_CONFIG = os.path.join(_HERE, 'config.json')
_DTYPE = torch.float64


def load_config(path: str = DEFAULT_CONFIG) -> Dict[str, Any]:
    """Load the JSON config (documentation lives in ignored '_comment*' keys)."""
    with open(path, 'r') as fh:
        return json.load(fh)


def _banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# --------------------------------------------------------------------------- #
# Synthetic & real field planes
# --------------------------------------------------------------------------- #
def synthetic_plane(size: int, seed: int, skew: float, device: str) -> torch.Tensor:
    """A skewed synthetic field plane (positive skewness, nonzero variance)."""
    g = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn(size * size, generator=g, dtype=_DTYPE, device=device)
    return z + skew * (z * z - 1.0)            # inject positive skewness


def load_plane(data: str, field: str, zlevel: int, device: str) -> torch.Tensor:
    """Load one z-level (flattened y-x plane) of ``field`` from a dir or .nc."""
    if os.path.isdir(data):
        from parameter_search.qoi_eval import load_binary_fields
        arr = load_binary_fields(data)[field][zlevel]
    elif data.endswith(('.nc', '.nc.v0', '.v0')):
        import netCDF4 as nc
        ds = nc.Dataset(data, 'r')
        try:
            arr = np.asarray(ds[field][0, zlevel, :, :], dtype=np.float64)
        finally:
            ds.close()
    else:
        raise ValueError(f"--data must be a dir of .raw files or a .nc file: {data}")
    return torch.as_tensor(np.ascontiguousarray(arr), dtype=_DTYPE, device=device).flatten()


# --------------------------------------------------------------------------- #
# Self-tests
# --------------------------------------------------------------------------- #
def _engine(qoi_name: str, cfg: Dict[str, Any], device: str) -> LinearProxyEngine:
    return LinearProxyEngine(QOI_REGISTRY[qoi_name], dtype=_DTYPE, device=device,
                             sigma2_floor=cfg['engine']['sigma2_floor'])


def test_constant_hessian(cfg: Dict[str, Any], device: str) -> None:
    """Variance & bilinear QoIs: structured == dense (H constant); MC <= t."""
    eb = 0.01
    P = 16                       # 16x16 plane -> N=256 (dense Hessian feasible)
    for name in ('uvar', 'wpup'):
        eng = _engine(name, cfg, device)
        n_fields = len(QOI_REGISTRY[name].fields)
        X0 = [synthetic_plane(P, seed=k, skew=0.4, device=device) for k in range(n_fields)]
        E = [torch.full_like(x, eb) for x in X0]
        t_struct = eng.structured_bound(X0, E)
        t_dense = eng.dense_bound(X0, E, n_samples=4)        # H constant -> 4 samples ok
        passed, max_err = eng.validate_bound(X0, E, t_struct,
                                             n_samples=cfg['validation']['samples'],
                                             seed=cfg['validation']['seed'])
        print(f"  {name:6s} N={X0[0].numel()*n_fields:>4d}  "
              f"t_struct={t_struct:.6e}  t_dense={t_dense:.6e}  "
              f"MC max_err={max_err:.3e}  util={max_err/t_struct:5.1%}")
        assert abs(t_struct - t_dense) <= 1e-9 * max(1.0, t_struct), \
            f"{name}: structured != dense for constant Hessian"
        assert passed, f"{name}: Monte-Carlo violated the bound"
        # Gradient cross-check vs torch.func.jacrev.
        f0, grads = eng.compute_gradient(*X0)
        from torch.func import jacrev
        z = torch.cat([x.clone() for x in X0])
        sizes = [x.numel() for x in X0]
        gref = jacrev(lambda v: QOI_REGISTRY[name](*torch.split(v, sizes)))(z)
        gmine = torch.cat(grads)
        assert (gref - gmine).abs().max() < 1e-10, f"{name}: gradient mismatch"
    print("  PASS (constant-Hessian closed form == dense; gradients match; MC ok)")


def test_wskew(cfg: Dict[str, Any], device: str) -> None:
    """wskew: structured (O(N)) >= dense (sampled) >= MC; all rigorous-consistent."""
    eb = 0.01
    P = 12                       # 12x12 plane -> N=144
    eng = _engine('wskew', cfg, device)
    X0 = [synthetic_plane(P, seed=7, skew=0.5, device=device)]
    E = [torch.full_like(X0[0], eb)]
    t_struct = eng.structured_bound(X0, E)
    t_dense = eng.dense_bound(X0, E, n_samples=4000, seed=1)
    passed, max_err = eng.validate_bound(X0, E, t_struct, n_samples=40000, seed=2)
    print(f"  wskew  N={X0[0].numel()}  t_struct={t_struct:.6e}  "
          f"t_dense(sampled)={t_dense:.6e}  MC max_err={max_err:.6e}")
    assert t_struct >= t_dense, "structured must upper-bound the dense entrywise bound"
    assert passed, "wskew: Monte-Carlo violated the structured bound"
    assert max_err <= t_dense * (1 + 1e-9), "wskew: MC exceeds dense bound (unexpected)"
    print("  PASS (structured >= dense >= MC; structured is a valid upper bound)")


def run_self_tests(cfg: Dict[str, Any], device: str) -> None:
    _banner("SELF-TESTS (closed forms, structured-vs-dense cross-validation, MC)")
    test_constant_hessian(cfg, device)
    test_wskew(cfg, device)
    _banner("ALL SELF-TESTS PASSED")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_driver(args: argparse.Namespace, cfg: Dict[str, Any], device: str) -> None:
    _banner(f"DRIVER: strict bound for QoI '{args.qoi}' over a full x-y plane")
    if args.qoi not in QOI_REGISTRY:
        raise SystemExit(f"--qoi must be one of {QOI_NAMES}")
    qoi = QOI_REGISTRY[args.qoi]
    eng = _engine(args.qoi, cfg, device)

    if args.data:
        print(f"  Loading z-level {args.zlevel} from {args.data}  fields={qoi.fields}")
        X0 = [load_plane(args.data, f, args.zlevel, device) for f in qoi.fields]
    else:
        size = cfg['synthetic']['size']
        print(f"  No --data: synthetic {size}x{size} field(s)")
        X0 = [synthetic_plane(size, seed=cfg['synthetic']['seed'] + k,
                              skew=cfg['synthetic']['skew'], device=device)
              for k in range(len(qoi.fields))]
    E = [torch.full_like(x, args.abs_eb) for x in X0]
    N = X0[0].numel()
    n_total = N * len(qoi.fields)

    t0 = time.perf_counter()
    f0, grads = eng.compute_gradient(*X0)
    t_grad = time.perf_counter() - t0
    grad_norm = torch.cat(grads).norm().item()

    t1 = time.perf_counter()
    try:
        t = eng.structured_bound(X0, E)
    except ValueError as exc:
        print(f"  f(X0) = {f0:.6g}  ||grad|| = {grad_norm:.6g}")
        print(f"\n  Cannot bound '{args.qoi}' here: {exc}")
        return
    t_bound = time.perf_counter() - t1

    t2 = time.perf_counter()
    passed, max_err = eng.validate_bound(X0, E, t, n_samples=cfg['validation']['samples'],
                                         seed=cfg['validation']['seed'])
    t_mc = time.perf_counter() - t2

    print(f"  {qoi.desc}")
    print(f"  plane N={N} ({int(N**0.5)}x{int(N**0.5)})  variables={n_total}  "
          f"device={device}  dtype=float64")
    print(f"  f(X0) = {f0:.6g}")
    print(f"  ||grad_F(X0)|| = {grad_norm:.6g}")
    print(f"\n  STRICT bound  t = 0.5 * E^T B E = {t:.6e}   (E = abs_eb = {args.abs_eb})")
    print(f"  Monte-Carlo ({cfg['validation']['samples']} samples): max|F-g| = {max_err:.6e}  "
          f"(<= t -> {passed}, utilization {max_err/t if t>0 else 0:.1%})")
    print(f"  wall time: gradient {t_grad:.3f}s | bound {t_bound:.3f}s | MC {t_mc:.3f}s")
    print("\n  t upper-bounds |QoI - linear_proxy| when every field data point in\n"
          "  the plane is compressed within +- abs_eb.")


def build_arg_parser(cfg: Dict[str, Any]) -> argparse.ArgumentParser:
    drv = cfg['driver']
    p = argparse.ArgumentParser(
        description="PyTorch full-plane strict error bound for turbulence QoIs.")
    p.add_argument('--config', default=DEFAULT_CONFIG)
    p.add_argument('--qoi', default=drv['qoi'], help=f"one of {QOI_NAMES}")
    p.add_argument('--data', default=drv['data'],
                   help="dir of {FIELD}_*.raw files, or a .nc file (else synthetic).")
    p.add_argument('--zlevel', type=int, default=drv['zlevel'])
    p.add_argument('--abs-eb', type=float, default=drv['abs_eb'])
    p.add_argument('--device', default=cfg['engine']['device'])
    p.add_argument('--no-tests', action='store_true')
    return p


def main(argv: List[str]) -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=DEFAULT_CONFIG)
    pre_args, _ = pre.parse_known_args(argv)
    cfg = load_config(pre_args.config)

    args = build_arg_parser(cfg).parse_args(argv)
    device = resolve_device(args.device)
    torch.set_grad_enabled(True)

    run_tests = cfg['driver'].get('run_self_tests', True) and not args.no_tests
    if run_tests:
        run_self_tests(cfg, device)
    run_driver(args, cfg, device)


if __name__ == '__main__':
    main(sys.argv[1:])
