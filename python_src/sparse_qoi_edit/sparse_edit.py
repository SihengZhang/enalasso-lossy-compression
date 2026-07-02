#!/usr/bin/env python3
"""Sparse QoI-preserving edits to an SZ3-decompressed field (PyTorch).

Feasibility test for the question: starting from a *loosely* compressed field
``X' = SZ3_decompress(SZ3(X, eb_data))``, can a gradient-based optimizer make a
*small number of edits* so the corrected field ``X^`` satisfies BOTH

  * C1 (data box) : ``|X^_i - X_i| <= eb_data`` for every point i, and
  * C2 (QoI)      : ``|Q(X^)_z - Q(X)_z| <= eb_qoi`` for every z-level,

simultaneously, where ``Q = wskew`` (vertical-velocity skewness) and ``X = W``.

Key structure exploited here
----------------------------
``wskew(z)`` is a function ONLY of the 256x256 plane at level ``z`` (see
``docs/derived_variables.md`` / ``parameter_search/qoi_eval.py``), so the 260
levels are independent and the whole problem is a single *batched* optimization
with batch dim = z and 65536 free variables per row.

``X^`` is a single free parameter tensor (a "0-layer neural field") optimized by
backprop through the *frozen* differentiable QoI. C1 is enforced as a hard
constraint by clamping ``X^`` into the box ``[X - eb_data, X + eb_data]`` after
every step, so C1 holds at all times by construction. C2 and sparsity are driven
by the loss. Because the ground truth ``X`` always lies in the box and satisfies
C2 exactly, the feasible set is never empty -- the only question is whether a
*sparse* edit reaches it.

Optimization stages
-------------------
1. Feasibility (Adam, per-parameter scaling handles the wide per-level sigma
   spread): minimize the QoI squared-hinge until every level is inside the bound.
2. Sparsify (proximal gradient / ISTA): gradient step on the hinge then a
   soft-threshold on ``Delta = X^ - X'`` (the prox of ``lambda*||.||_1``, which
   yields EXACT zeros), with ``lambda`` ramped adaptively while keeping the best
   feasible iterate.
3. Prune + polish + bisect: log-space binary-search a hard threshold on
   ``|Delta|`` -- zero the smallest edits, polish the surviving support, keep the
   sparsest iterate that still verifies.

The torch ``wskew`` mirrors the numpy verifier in
``parameter_search/qoi_eval.py`` bit-for-bit (mean-subtraction, ddof=0 std,
sigma floor 1e-10, mean of the cubed standardized perturbation) so the training
target and the independent check agree.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Differentiable QoI (batched over z), matching qoi_eval.compute_qois['wskew'].
# --------------------------------------------------------------------------- #
def batched_wskew(x: torch.Tensor, sigma_floor: float = 1e-10) -> torch.Tensor:
    """Per-row vertical-velocity skewness for ``x`` of shape ``(Z, N)``.

    Mirrors ``qoi_eval.compute_single_qoi('wskew', ...)``:
        wp    = x - mean_xy(x)            # Reynolds perturbation
        mu    = mean(wp)                  # ~0 by construction
        sigma = std(wp, ddof=0)           # floored at ``sigma_floor``
        wskew = mean(((wp - mu) / sigma) ** 3)
    Returns a tensor of shape ``(Z,)``.
    """
    wp = x - x.mean(dim=1, keepdim=True)
    mu = wp.mean(dim=1, keepdim=True)
    centered = wp - mu
    var = (centered * centered).mean(dim=1)
    sigma = torch.sqrt(var).clamp_min(sigma_floor)
    z = centered / sigma.unsqueeze(1)
    return (z * z * z).mean(dim=1)


def row_sigma(x: torch.Tensor) -> torch.Tensor:
    """Per-row std (ddof=0) of the Reynolds perturbation, shape ``(Z,)``."""
    wp = x - x.mean(dim=1, keepdim=True)
    return torch.sqrt((wp * wp).mean(dim=1))


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class SparseQoIConfig:
    """Hyper-parameters for the sparse-edit optimizer.

    The optimizer is a per-row-normalized projected gradient descent on the QoI
    squared-hinge, with an adaptive soft-threshold (ISTA) that sparsifies the
    edits, followed by a hard-threshold bisection finalize. Per-row (per-z)
    gradient normalization is essential: ``d(wskew)/dx ~ 1/sigma^3`` so per-level
    gradient magnitudes span orders of magnitude, but normalizing each row by its
    own gradient RMS equalizes the convergence rate across levels WITHOUT
    flattening the within-row concentration on the skewness-sensitive tail points
    (which a per-parameter optimizer like Adam would destroy).
    """

    device: str = "cuda:0"
    sigma_floor: float = 1e-10        # wskew denominator floor (matches qoi_eval)
    # A level is "active" (subject to C2 + optimization) only if its perturbation
    # std is above this; degenerate near-constant planes are left at X' and
    # auto-pass, mirroring qoi_eval's div-by-zero guard.
    sigma_active_rel: float = 1e-6    # relative to max-over-z perturbation std
    margin: float = 0.9               # optimize the LOSS to margin*thr
    feas_margin: float = 0.95         # accept/prune only if err <= feas_margin*thr
                                      # (leaves headroom so the final result is a
                                      #  robust PASS, not a knife-edge one)

    # Projected-gradient + ISTA sparsify
    optimize_iters: int = 3000
    pgd_step_frac: float = 1.0        # step0 = frac * eb_data (per-row unit-RMS)
    step_decay_final: float = 1e-2    # step decays geometrically to this * step0
    grad_eps: float = 1e-20           # guards 0/0 on already-feasible rows
    shrink0_frac: float = 0.01        # initial soft-threshold = frac * eb_data
    shrink_up: float = 1.3            # ramp shrink up when feasible (sparser)
    shrink_down: float = 0.6          # ease shrink when feasibility lost
    # Hard-threshold bisection finalize
    bisect_iters: int = 26
    polish_iters: int = 300

    check_every: int = 25
    verbose: bool = True
    seed: int = 0


# --------------------------------------------------------------------------- #
# Editor
# --------------------------------------------------------------------------- #
class SparseQoIEditor:
    """Edit an SZ3-decompressed field to jointly satisfy a data box and a QoI
    bound with as few point edits as possible.

    Parameters
    ----------
    x_gt, x_base : np.ndarray
        Ground-truth ``X`` and SZ3-decompressed base ``X'``, shape ``(Z, N)``
        (float32). ``X'`` must already satisfy ``|X'-X| <= eb_data``.
    eb_data : float
        Absolute point-wise data error bound (the SZ3 ABS bound; defines the box).
    eb_qoi : np.ndarray | float
        Absolute per-level wskew error bound (shape ``(Z,)`` or scalar). For the
        repo's relative convention pass ``tau * range_z(wskew(X))``.
    """

    def __init__(self, x_gt: np.ndarray, x_base: np.ndarray,
                 eb_data: float, eb_qoi, cfg: Optional[SparseQoIConfig] = None):
        self.cfg = cfg or SparseQoIConfig()
        torch.manual_seed(self.cfg.seed)
        dev = self.cfg.device
        if dev.startswith("cuda") and not torch.cuda.is_available():
            dev = "cpu"
        self.device = torch.device(dev)

        f64 = torch.float64
        self.x_gt = torch.as_tensor(x_gt, dtype=f64, device=self.device)
        self.x_base = torch.as_tensor(x_base, dtype=f64, device=self.device)
        self.Z, self.N = self.x_gt.shape
        self.eb_data = float(eb_data)
        self.lo = self.x_gt - self.eb_data
        self.hi = self.x_gt + self.eb_data

        eb_qoi = np.broadcast_to(np.asarray(eb_qoi, dtype=np.float64),
                                 (self.Z,)).copy()
        self.thr = torch.as_tensor(eb_qoi, dtype=f64, device=self.device)  # (Z,)
        self.thr_in = self.cfg.margin * self.thr

        # Targets derived from Q(X) (precomputed once, frozen).
        with torch.no_grad():
            self.q0 = batched_wskew(self.x_gt, self.cfg.sigma_floor)
            sig0 = row_sigma(self.x_gt)
            sig_active = self.cfg.sigma_active_rel * float(sig0.max())
            self.active = sig0 > sig_active            # (Z,) bool
        self.n_active = int(self.active.sum())
        self.n_skipped = int((~self.active).sum())

    # -- helpers ----------------------------------------------------------- #
    def _qoi_err(self, x: torch.Tensor) -> torch.Tensor:
        """Signed wskew error per level, zeroed on inactive levels."""
        return torch.where(self.active, batched_wskew(x, self.cfg.sigma_floor)
                           - self.q0, torch.zeros_like(self.q0))

    def _hinge_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Squared QoI hinge (target = margin*thr), summed over active levels."""
        err = self._qoi_err(x).abs()
        hinge = torch.relu(err - self.thr_in)
        return (hinge * hinge).sum()

    def _feasible(self, x: torch.Tensor) -> bool:
        """All active levels within ``feas_margin * thr`` (headroom under the
        reported full-bound check)."""
        with torch.no_grad():
            err = self._qoi_err(x).abs()
            return bool((err <= self.cfg.feas_margin * self.thr).all())

    def _n_edits(self, x: torch.Tensor) -> int:
        with torch.no_grad():
            return int((x != self.x_base).sum())

    def _clamp_(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            torch.clamp_(x, self.lo, self.hi)
            # Inactive levels are never edited.
            if self.n_skipped:
                x[~self.active] = self.x_base[~self.active]

    def _log(self, msg: str) -> None:
        if self.cfg.verbose:
            print(msg, flush=True)

    def _norm_grad(self, x: torch.Tensor) -> torch.Tensor:
        """Per-row-normalized hinge gradient (unit RMS per active row)."""
        x = x.detach().requires_grad_(True)
        loss = self._hinge_loss(x)
        (g,) = torch.autograd.grad(loss, x)
        g = torch.nan_to_num(g.detach())
        rms = torch.sqrt((g * g).mean(dim=1, keepdim=True))
        return g / (rms + self.cfg.grad_eps)

    # -- stage 1+2: projected gradient + adaptive ISTA sparsify ----------- #
    def _stage_optimize(self) -> torch.Tensor:
        """Reach feasibility, then ramp a soft-threshold to sparsify the edits,
        always retaining the sparsest feasible iterate seen."""
        step0 = self.cfg.pgd_step_frac * self.eb_data
        x = self.x_base.clone()
        best, best_nnz = None, self.Z * self.N + 1
        shrink = 0.0                       # 0 until first feasibility (warmup)
        shrink0 = self.cfg.shrink0_frac * self.eb_data
        for it in range(self.cfg.optimize_iters):
            # Geometric step decay: coarse early (cross the box), fine late
            # (settle precisely below threshold -- RMS-normalization otherwise
            # removes the squared-hinge's natural near-boundary slowdown).
            step = step0 * self.cfg.step_decay_final ** (it / self.cfg.optimize_iters)
            gs = self._norm_grad(x)
            with torch.no_grad():
                x = x - step * gs
                if shrink > 0.0:           # soft-threshold the edits (ISTA prox)
                    d = x - self.x_base
                    d = torch.sign(d) * torch.relu(d.abs() - shrink)
                    x = self.x_base + d
            x = x.detach()
            self._clamp_(x)
            if it % self.cfg.check_every == 0:
                feas = self._feasible(x)
                nnz = self._n_edits(x)
                if feas:
                    if nnz <= best_nnz:
                        best_nnz, best = nnz, x.clone()
                    shrink = shrink0 if shrink == 0.0 else shrink * self.cfg.shrink_up
                elif best is not None:
                    x = best.clone()           # backtrack to last feasible
                    shrink *= self.cfg.shrink_down
                if self.cfg.verbose and it % (self.cfg.check_every * 20) == 0:
                    nviol = int((self._qoi_err(x).abs() > self.thr).sum())
                    self._log(f"  [opt] it={it} feas={feas} viol={nviol} "
                              f"nnz={nnz} best={best_nnz} shrink={shrink:.2e}")
        if best is None:                       # never feasible -> return last
            self._log("  [opt] WARNING: never reached feasibility")
            return x
        self._log(f"  [opt] best feasible edits={best_nnz}")
        return best

    # -- stage 3: prune + polish + bisect --------------------------------- #
    def _polish(self, x_init: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        """A few normalized hinge-gradient steps restricted to ``support``."""
        step0 = self.cfg.pgd_step_frac * self.eb_data
        x = x_init.clone()
        for it in range(self.cfg.polish_iters):
            step = step0 * self.cfg.step_decay_final ** (it / self.cfg.polish_iters)
            gs = self._norm_grad(x) * support      # only edit kept points
            with torch.no_grad():
                x = x - step * gs
            x = x.detach()
            self._clamp_(x)
        return x

    def _stage_bisect(self, x0: torch.Tensor) -> torch.Tensor:
        d = (x0 - self.x_base).abs()
        nz = d[d > 0]
        if nz.numel() == 0:
            return x0
        lo_t = float(nz.min())
        hi_t = float(nz.max())
        best = x0.clone()
        best_nnz = self._n_edits(x0)
        for _ in range(self.cfg.bisect_iters):
            t = float(np.sqrt(max(lo_t, 1e-300) * hi_t))
            support = (d >= t)
            cand = self.x_base + (x0 - self.x_base) * support
            self._clamp_(cand)
            cand = self._polish(cand, support)
            if self._feasible(cand):
                lo_t = t                       # can prune more aggressively
                nnz = self._n_edits(cand)
                if nnz <= best_nnz:
                    best_nnz, best = nnz, cand.clone()
            else:
                hi_t = t                       # pruned too much; ease off
        self._log(f"  [bisect] final edits={best_nnz}")
        return best

    # -- driver ----------------------------------------------------------- #
    def run(self) -> Dict:
        """Run all stages; return ``X^`` (numpy float32, shape ``(Z, N)``) and
        a statistics dict."""
        self._log(f"Editor: Z={self.Z} N={self.N} active_levels={self.n_active} "
                  f"skipped={self.n_skipped} device={self.device}")
        x = self._stage_optimize()
        x = self._stage_bisect(x)

        with torch.no_grad():
            delta = (x - self.x_base)
            edited = delta != 0
            stats = {
                "feasible": self._feasible(x),
                "n_edits": int(edited.sum()),
                "edit_fraction": float(edited.float().mean()),
                "edits_per_level": edited.sum(dim=1).detach().cpu().numpy(),
                "max_data_err": float((x - self.x_gt).abs().max()),
                "max_qoi_abs_err": float(self._qoi_err(x).abs().max()),
                "n_active": self.n_active,
                "n_skipped": self.n_skipped,
                "delta_abs_nonzero": delta.abs()[edited].detach().cpu().numpy(),
                "wskew_hat": batched_wskew(x, self.cfg.sigma_floor)
                             .detach().cpu().numpy(),
                "wskew_gt": self.q0.detach().cpu().numpy(),
            }
        x_hat = x.detach().to(torch.float32).cpu().numpy()
        return {"x_hat": x_hat, "stats": stats}
