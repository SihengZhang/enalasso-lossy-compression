"""Linear-proxy strict error-bound engine (PyTorch, full x-y-plane variables).

For a QoI ``F(X)`` whose variables ``X`` are the data points of an x-y plane
(one or two raw fields), a center ``X0`` (ground truth) and a per-variable
displacement ``E`` (compression error bound, region ``|X-X0| <= E``):

1. Linear proxy ``g(X) = F(X0) + grad_F(X0)^T (X-X0)`` -- gradient via reverse-mode
   autodiff (one backward pass; trivial at n = 65536, GPU-accelerated).
2. Bounding matrix ``B_ij = max_box |d^2F/dx_i dx_j|`` and strict bound
   ``t = 0.5 E^T B E >= |F(X)-g(X)|``. ``B`` is never materialized at scale:
   the QoI Hessians are CONSTANT (variance/bilinear) or DIAGONAL+LOW-RANK
   (wskew), so ``t`` is computed in O(N) closed form. A dense
   ``torch.func.hessian`` + box-sampling mode cross-validates on small planes.

All math is double precision (float64) for bound rigor. See ``qoi_library.py``
for the QoI definitions and Hessian-structure tags.
"""

from typing import Any, List, Sequence, Tuple

import torch
from torch.func import hessian

from qoi_library import QoI, compute_perturbation


def resolve_device(device: str) -> str:
    """Fall back to CPU if CUDA was requested but is unavailable."""
    if device.startswith('cuda') and not torch.cuda.is_available():
        print("[linear_proxy] CUDA requested but unavailable; using CPU.")
        return 'cpu'
    return device


class LinearProxyEngine:
    """Gradient + strict structured error bound for one QoI over a box.

    Args:
        qoi: a :class:`qoi_library.QoI`.
        dtype: torch dtype (float64 recommended for rigor).
        device: 'cuda' or 'cpu' (auto-falls back to cpu).
        sigma2_floor: minimum block variance; the wskew bound raises if the box
            could drive sigma^2 below this (Hessian singularity).
    """

    def __init__(self, qoi: QoI, dtype: torch.dtype = torch.float64,
                 device: str = 'cuda', sigma2_floor: float = 1e-12) -> None:
        self.qoi = qoi
        self.dtype = dtype
        self.device = resolve_device(device)
        self.sigma2_floor = float(sigma2_floor)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _to(self, t: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(t, dtype=self.dtype, device=self.device).flatten()

    def _prep(self, X0: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        xs = [self._to(x) for x in X0]
        n = xs[0].numel()
        if any(x.numel() != n for x in xs):
            raise ValueError("all field planes must have the same length N")
        if len(xs) != len(self.qoi.fields):
            raise ValueError(f"{self.qoi.name} expects {len(self.qoi.fields)} field(s)")
        return xs

    # ------------------------------------------------------------------ #
    # linear proxy: f(X0) and grad_f(X0)
    # ------------------------------------------------------------------ #
    def compute_gradient(self, *X0: torch.Tensor) -> Tuple[float, List[torch.Tensor]]:
        """Return (f0, grads) where grads is one gradient tensor per field."""
        xs = [x.clone().detach().requires_grad_(True) for x in self._prep(X0)]
        f0 = self.qoi(*xs)
        grads = torch.autograd.grad(f0, xs)
        return float(f0.detach()), [g.detach() for g in grads]

    # ------------------------------------------------------------------ #
    # structured O(N) bound (no n x n materialization)
    # ------------------------------------------------------------------ #
    def structured_bound(self, X0: Sequence[torch.Tensor],
                         E: Sequence[torch.Tensor]) -> float:
        """Strict bound t = 0.5 E^T B E via the QoI's Hessian structure."""
        xs = self._prep(X0)
        es = [self._to(e) for e in E]
        N = xs[0].numel()
        kind = self.qoi.kind

        if kind == 'var':
            # H = (2/N) P, P_ij = delta_ij - 1/N. B = |H|.
            # t = (1/N^2)[(N-2) sum E^2 + (sum E)^2].
            e = es[0]
            sE = e.sum(); sE2 = (e * e).sum()
            return float(((N - 2) * sE2 + sE * sE) / (N * N))

        if kind == 'bilinear':
            # Hessian off-diagonal block (1/N)P; w-w and x-x blocks zero.
            # t = (1/N^2)[(N-2) sum E_w*E_x + (sum E_w)(sum E_x)].
            ew, ex = es[0], es[1]
            return float(((N - 2) * (ew * ex).sum() + ew.sum() * ex.sum()) / (N * N))

        if kind == 'wskew':
            return self._wskew_structured_bound(xs[0], es[0])

        raise ValueError(f"unknown QoI kind {kind!r}")

    def _wskew_structured_bound(self, w0: torch.Tensor, E: torch.Tensor) -> float:
        """O(N) strict upper bound for wskew via diagonal+low-rank Hessian.

        H_w = P H_c P (c = P w0) is diagonal + rank-<=6:
            H_w = sqrt(N)[ P diag(d) P
                          - 9 S2^{-5/2} (c2c c^T + c c2c^T)
                          + 15 S3 S2^{-7/2} c c^T ]
        with d_i = 6 c_i S2^{-3/2} - 3 S3 S2^{-5/2}, c2c = P(c^2), and
        P diag(d) P = diag(d) - (1/N)(d 1^T + 1 d^T) + (sum d / N^2) 1 1^T.

        Sum_ij max_box|H_ij| is upper-bounded (triangle inequality + per-factor
        box maxima) in O(N); t = 0.5 eb^2 * that sum (uniform eb = max E).
        A raw-w box of half-width eb implies |delta c_i| <= 2 eb.
        """
        eb = float(E.max())
        N = w0.numel()
        sN = N ** 0.5
        c0 = compute_perturbation(w0)

        # Per-index magnitude of c over the box; S2 (=h) and S3 ranges.
        cmax = c0.abs() + 2.0 * eb
        cmin = (c0.abs() - 2.0 * eb).clamp(min=0.0)
        h_min = float((cmin * cmin).sum())
        h_max = float((cmax * cmax).sum())
        g_absmax = float((cmax ** 3).sum())

        if h_min <= self.sigma2_floor * N:
            raise ValueError(
                f"wskew Hessian singular in box: min sigma^2 ~ {h_min / N:.3e} "
                f"<= floor {self.sigma2_floor:.1e}. Tighten E or skip this z-level.")

        # Worst case for |H| is the smallest denominator h -> h_min.
        # Per-index max|D_i| with D = sqrt(N) d.
        maxD = sN * (6.0 * cmax * h_min ** -1.5 + 3.0 * g_absmax * h_min ** -2.5)
        diag_sum = float(maxD.sum())

        # Rank vectors' box-max L1 sums.
        c2c_max = cmax * cmax + h_max / N          # |P(c^2)|_i <= cmax_i^2 + h_max/N
        S_c2c = float(c2c_max.sum())
        S_c = float(cmax.sum())

        # Contributions to Sum_ij max|H_ij| (each a rigorous upper bound):
        #   diag(D)                                        -> diag_sum
        #   P diag(d) P off-structure (d 1^T, 1 d^T, 11^T) -> 3 * diag_sum
        #   -9 sqrt(N) h^{-5/2}(c2c c^T + c c2c^T)         -> 2 * 9 sqrt(N) h_min^{-5/2} S_c2c S_c
        #   +15 sqrt(N) S3 h^{-7/2} c c^T                  -> 15 sqrt(N) g_absmax h_min^{-7/2} S_c^2
        sumB = diag_sum + 3.0 * diag_sum
        sumB += 2.0 * (9.0 * sN * h_min ** -2.5 * S_c2c * S_c)
        sumB += 15.0 * sN * g_absmax * h_min ** -3.5 * S_c * S_c

        return 0.5 * eb * eb * sumB

    # ------------------------------------------------------------------ #
    # dense bound (small planes only) -- for cross-validation
    # ------------------------------------------------------------------ #
    def dense_bound(self, X0: Sequence[torch.Tensor], E: Sequence[torch.Tensor],
                    n_samples: int = 2000, seed: int = 0,
                    max_vars: int = 6000) -> float:
        """Brute-force t = 0.5 E^T B E with B_ij = max over sampled box of |H_ij|.

        Materializes the full (n x n) Hessian via ``torch.func.hessian`` at the
        center plus ``n_samples`` random points in the box, taking the entrywise
        max. Only feasible for small n (<= ``max_vars``); used to cross-validate
        the structured bound (the sampled max is a lower estimate of the true
        entrywise bound, so structured >= dense is the expected sanity check).
        """
        xs = self._prep(X0)
        es = [self._to(e) for e in E]
        sizes = [x.numel() for x in xs]
        n_total = sum(sizes)
        if n_total > max_vars:
            raise ValueError(f"dense_bound infeasible for n={n_total} > {max_vars}")

        x0_cat = torch.cat(xs)
        e_cat = torch.cat(es)

        def f_cat(z: torch.Tensor) -> torch.Tensor:
            return self.qoi(*torch.split(z, sizes))

        H_fn = hessian(f_cat)
        g = torch.Generator(device=self.device).manual_seed(seed)
        lo, hi = x0_cat - e_cat, x0_cat + e_cat
        B = H_fn(x0_cat).abs()                      # include the center
        for _ in range(n_samples):
            z = lo + (hi - lo) * torch.rand(n_total, generator=g,
                                            dtype=self.dtype, device=self.device)
            B = torch.maximum(B, H_fn(z).abs())
        return float(0.5 * e_cat @ B @ e_cat)

    # ------------------------------------------------------------------ #
    # bound dispatch + Monte-Carlo validation
    # ------------------------------------------------------------------ #
    def error_bound(self, X0: Sequence[torch.Tensor], E: Sequence[torch.Tensor],
                    method: str = 'structured', **kw: Any) -> float:
        if method == 'structured':
            return self.structured_bound(X0, E)
        if method == 'dense':
            return self.dense_bound(X0, E, **kw)
        raise ValueError(f"method must be 'structured' or 'dense', got {method!r}")

    def validate_bound(self, X0: Sequence[torch.Tensor], E: Sequence[torch.Tensor],
                       t: float, n_samples: int = 10000, seed: int = 0,
                       chunk: int = 256) -> Tuple[bool, float]:
        """Monte-Carlo: sample the box, check max|F(X)-g(X)| <= t.

        Returns (passed, max_err). Evaluates F over each batch with ``torch.vmap``
        and processes samples in chunks to bound memory at full plane size.
        """
        xs = self._prep(X0)
        es = [self._to(e) for e in E]
        f0, grads = self.compute_gradient(*xs)
        vF = torch.vmap(self.qoi.fn)
        g = torch.Generator(device=self.device).manual_seed(seed)

        max_err = 0.0
        done = 0
        while done < n_samples:
            b = min(chunk, n_samples - done)
            samples = []
            for x0, e in zip(xs, es):
                u = torch.rand(b, x0.numel(), generator=g,
                               dtype=self.dtype, device=self.device)
                samples.append(x0 + (2.0 * u - 1.0) * e)
            f_vals = vF(*samples)                       # (b,)
            g_vals = torch.full((b,), f0, dtype=self.dtype, device=self.device)
            for s, x0, grad in zip(samples, xs, grads):
                g_vals = g_vals + (s - x0) @ grad
            max_err = max(max_err, (f_vals - g_vals).abs().max().item())
            done += b

        passed = max_err <= t * (1.0 + 1e-9) + 1e-300
        return passed, max_err
