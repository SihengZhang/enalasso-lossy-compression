#!/usr/bin/env python3
"""Binary-search engine for QoI-preserving per-field error bounds.

For each of the six raw fields, find the highest (loosest) absolute compression
error bound such that every derived QoI stays within its relative error bound.

Algorithm (see the project plan):

* Phase 1 -- single-field QoIs. Binary-search ``U`` via ``uvar``, ``V`` via
  ``vvar`` and ``W`` via both ``wvar`` and ``wskew`` (the tighter wins). These
  QoIs depend on exactly one field, so each search is one-dimensional. ``W`` is
  the universal partner for the bilinear QoIs, so it is locked here and held
  fixed (as its decompressed array) for Phase 2.
* Phase 2 -- multi-field QoIs with ``W`` fixed. Search ``PP`` via ``wppp``,
  ``TABS`` via ``wptp``, ``QV`` via ``wpqvp``; re-search ``U`` via ``wpup`` and
  ``V`` via ``wpvp`` and keep the tighter of the Phase-1 / Phase-2 bound.
* W-coupling fallback -- if a bilinear QoI cannot be met even with its partner
  near-lossless (``W``'s own error exhausts the budget), tighten ``W`` and
  restart Phase 2. Tightening ``W`` only improves every other QoI, so it is safe.
* Phase 3 -- final verification. Compress all six fields at the final bounds,
  recompute all nine QoIs, and report pass/fail plus the total compression ratio.
* Lossless fallback -- any QoI that is infeasible even with its fields
  near-lossless (e.g. ``wskew``) has its dependency fields stored losslessly so
  it is satisfied exactly; in practice this forces ``W`` to lossless.

Correctness rests on *separability*: a QoI depends only on its own fields'
perturbations (the horizontal mean is per-field), so a per-field search with the
coupled field(s) at their final state matches the all-fields-compressed result.
"""

import math
from collections import namedtuple
from typing import Dict, List

import numpy as np

from .compressors import BaseCompressor
from .qoi_eval import (
    FIELD_NAMES,
    QOI_FIELD_DEPENDENCIES,
    QOI_NAMES,
    all_qoi_errors,
    compute_perturbations,
    compute_qois,
    compute_single_qoi,
    qoi_rel_error,
    qoi_value_ranges,
)

# Bilinear QoI -> the partner field searched for it (W is the fixed partner).
_BILINEAR_PARTNER = {
    'wppp': 'PP', 'wptp': 'TABS', 'wpqvp': 'QV', 'wpup': 'U', 'wpvp': 'V',
}

BoundResult = namedtuple("BoundResult", ["value", "status"])
# status: 'ok' (binary search converged), 'unconstrained' (loosest bound passes),
#         'infeasible' (even near-lossless fails the QoI).


def binary_search_bound(eval_fn, lo: float, hi: float,
                        max_iters: int = 20, margin: float = 0.9) -> BoundResult:
    """Largest absolute error bound in [lo, hi] for which ``eval_fn`` passes.

    ``eval_fn(eb) -> bool`` returns True when compressing at ``eb`` keeps the
    target QoI(s) within bound. Searches in log space; assumes the QoI error is
    (approximately) monotone in ``eb`` and applies a safety ``margin`` < 1 to
    the converged bound. Capped iterations guard against non-monotonicity.
    """
    if eval_fn(hi):
        return BoundResult(hi, 'unconstrained')
    if not eval_fn(lo):
        return BoundResult(lo, 'infeasible')
    lo_pass, hi_fail = lo, hi
    for _ in range(max_iters):
        mid = math.sqrt(lo_pass * hi_fail)  # geometric (log-space) midpoint
        if eval_fn(mid):
            lo_pass = mid
        else:
            hi_fail = mid
    return BoundResult(lo_pass * margin, 'ok')


class ParameterSearch:
    """Drive the staged binary search for one compressor backend."""

    def __init__(self, compressor: BaseCompressor,
                 fields: Dict[str, np.ndarray],
                 qoi_bounds: Dict[str, float],
                 max_iters: int = 20, margin: float = 0.9,
                 max_w_restarts: int = 5, max_verify_attempts: int = 6,
                 verbose: bool = False):
        self.c = compressor
        self.fields = fields
        self.tau = qoi_bounds
        self.max_iters = max_iters
        self.margin = margin
        self.max_w_restarts = max_w_restarts
        self.max_verify_attempts = max_verify_attempts
        self.verbose = verbose

        # Reference QoIs / ranges from the pristine fields (computed once).
        self.orig_qois = compute_qois(compute_perturbations(fields))
        self.ranges = qoi_value_ranges(self.orig_qois)
        self.field_range = {f: float(fields[f].max() - fields[f].min())
                            for f in FIELD_NAMES}
        self.infeasible_qois: List[str] = []
        self.lossless_fields: List[str] = []

    # -- logging -----------------------------------------------------------
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[{self.c.name}] {msg}", flush=True)

    # -- bracketing --------------------------------------------------------
    def _bracket(self, field: str):
        r = self.field_range[field]
        if r <= 0:
            r = 1.0
        return r * 1e-7, r * 0.5

    # -- per-(field, QoI-set) evaluator -----------------------------------
    def _make_eval(self, search_field: str, qois: List[str],
                   fixed: Dict[str, np.ndarray]):
        """Build eval_fn(eb): compress ``search_field`` at eb, check ``qois``.

        ``fixed`` maps coupled fields (e.g. {'W': W_dec}) to their held arrays;
        any other dependency field falls back to its original (pristine) array.
        """
        def eval_fn(eb: float) -> bool:
            dec, _ = self.c.round_trip(self.fields[search_field], eb)
            for q in qois:
                test_fields = {}
                for dep in QOI_FIELD_DEPENDENCIES[q]:
                    if dep == search_field:
                        test_fields[dep] = dec
                    elif dep in fixed:
                        test_fields[dep] = fixed[dep]
                    else:
                        test_fields[dep] = self.fields[dep]
                test_q = compute_single_qoi(q, test_fields)
                err = qoi_rel_error(self.orig_qois, self.ranges, {q: test_q}, q)
                if err > self.tau[q]:
                    return False
            return True
        return eval_fn

    def _search(self, search_field: str, qois: List[str],
                fixed: Dict[str, np.ndarray]) -> BoundResult:
        lo, hi = self._bracket(search_field)
        eval_fn = self._make_eval(search_field, qois, fixed)
        res = binary_search_bound(eval_fn, lo, hi, self.max_iters, self.margin)
        self._log(f"search {search_field} via {qois}: "
                  f"abs_eb={res.value:.3e} ({res.status})")
        return res

    def _infeasible_among(self, failing: List[str]) -> List[str]:
        """Of the ``failing`` QoIs, which fail even with all deps near-lossless.

        Such a QoI is fundamentally infeasible at the requested bound (no
        compression of its fields can satisfy it), as opposed to a near-miss.
        Run once at the end on the (small) set of failing QoIs.
        """
        near: Dict[str, np.ndarray] = {}
        out = []
        for q in failing:
            test_fields = {}
            for dep in QOI_FIELD_DEPENDENCIES[q]:
                if dep not in near:
                    lo, _ = self._bracket(dep)
                    near[dep] = self.c.round_trip(self.fields[dep], lo)[0]
                test_fields[dep] = near[dep]
            test_q = compute_single_qoi(q, test_fields)
            if qoi_rel_error(self.orig_qois, self.ranges,
                             {q: test_q}, q) > self.tau[q]:
                out.append(q)
        return out

    # -- W refinement (fallback) ------------------------------------------
    def _refine_w(self, qoi: str, partner: str) -> BoundResult:
        """Tighten W so ``qoi`` can pass with ``partner`` near-lossless."""
        lo_p, _ = self._bracket(partner)
        partner_dec, _ = self.c.round_trip(self.fields[partner], lo_p)
        lo, hi = self._bracket('W')

        def eval_fn(eb: float) -> bool:
            w_dec, _ = self.c.round_trip(self.fields['W'], eb)
            test_q = compute_single_qoi(qoi, {'W': w_dec, partner: partner_dec})
            err = qoi_rel_error(self.orig_qois, self.ranges, {qoi: test_q}, qoi)
            return err <= self.tau[qoi]

        res = binary_search_bound(eval_fn, lo, hi, self.max_iters, self.margin)
        self._log(f"refine W for {qoi} (partner {partner} near-lossless): "
                  f"abs_eb={res.value:.3e} ({res.status})")
        return res

    # -- main driver -------------------------------------------------------
    def run(self) -> dict:
        eb: Dict[str, float] = {}
        status: Dict[str, str] = {}

        # ---- Phase 1: single-field QoIs ----
        self._log("Phase 1: single-field QoIs (U via uvar, V via vvar, "
                  "W via wvar+wskew)")
        rU = self._search('U', ['uvar'], {})
        rV = self._search('V', ['vvar'], {})
        rW = self._search('W', ['wvar', 'wskew'], {})
        eb['U'], status['U'] = rU.value, rU.status
        eb['V'], status['V'] = rV.value, rV.status
        eb['W'], status['W'] = rW.value, rW.status

        # ---- Phase 2 (with W-coupling fallback) ----
        self._log("Phase 2: multi-field QoIs (W fixed)")
        restarts = 0
        while True:
            w_dec, _ = self.c.round_trip(self.fields['W'], eb['W'])
            results = {}
            for qoi, partner in _BILINEAR_PARTNER.items():
                results[qoi] = self._search(partner, [qoi], {'W': w_dec})

            # Any bilinear QoI infeasible => W is too loose for it.
            infeasible = [(q, _BILINEAR_PARTNER[q]) for q in results
                          if results[q].status == 'infeasible']
            if infeasible and restarts < self.max_w_restarts:
                qoi, partner = infeasible[0]
                self._log(f"  {qoi} infeasible with W={eb['W']:.3e}; tightening W")
                rW2 = self._refine_w(qoi, partner)
                new_w = min(eb['W'], rW2.value)
                if rW2.status == 'infeasible' or new_w >= eb['W']:
                    # Cannot improve by tightening W further; let the final
                    # verification + infeasibility classification report it.
                    eb['W'] = min(eb['W'], rW2.value)
                    self._log(f"  {qoi} cannot be improved by tightening W; "
                              "stopping W refinement")
                    self._finalize_phase2(eb, status, results)
                    break
                eb['W'] = new_w
                status['W'] = 'ok'
                restarts += 1
                continue

            self._finalize_phase2(eb, status, results)
            break

        # ---- Phase 3: final all-fields verification (+ marginal retries) ----
        result = self._verify(eb, status)
        return result

    def _finalize_phase2(self, eb, status, results) -> None:
        """Commit Phase-2 bounds: PP/TABS/QV directly; U/V take the tighter."""
        eb['PP'], status['PP'] = results['wppp'].value, results['wppp'].status
        eb['TABS'], status['TABS'] = results['wptp'].value, results['wptp'].status
        eb['QV'], status['QV'] = results['wpqvp'].value, results['wpqvp'].status
        for qoi, field in (('wpup', 'U'), ('wpvp', 'V')):
            r = results[qoi]
            if r.value < eb[field]:
                eb[field] = r.value
                status[field] = f"{status[field]}+{qoi}"

    # -- verification ------------------------------------------------------
    def _verify(self, eb: Dict[str, float], status: Dict[str, str]) -> dict:
        self._log("Phase 3: final verification (all fields compressed)")
        last = self.max_verify_attempts - 1
        dec_fields, comp_bytes, errors, failing = {}, {}, {}, []
        for attempt in range(self.max_verify_attempts):
            for f in FIELD_NAMES:
                dec, nbytes = self.c.round_trip(self.fields[f], eb[f])
                dec_fields[f] = dec
                comp_bytes[f] = nbytes
            errors = all_qoi_errors(self.fields, dec_fields)
            failing = [q for q in QOI_NAMES if errors[q] > self.tau[q]]
            if not failing or attempt == last:
                break
            # Tighten the bounds of fields feeding the failing QoIs and retry.
            self._log(f"  verification fail on {failing}; tightening deps "
                      f"(attempt {attempt + 1})")
            for q in failing:
                for dep in QOI_FIELD_DEPENDENCIES[q]:
                    eb[dep] *= 0.9
                    if 'tightened' not in status[dep]:
                        status[dep] = status[dep] + '+tightened'

        # Lossless fallback: any QoI still failing that is infeasible even with
        # its fields near-lossless is satisfied exactly by storing those fields
        # losslessly (this is what makes e.g. wskew force W to lossless).
        if failing:
            infeasible = self._infeasible_among(failing)
            lossless = sorted({dep for q in infeasible
                               for dep in QOI_FIELD_DEPENDENCIES[q]})
            if lossless:
                self._log(f"  infeasible QoIs {infeasible}; storing "
                          f"{lossless} losslessly")
                for f in lossless:
                    dec, nbytes = self.c.lossless_round_trip(self.fields[f])
                    dec_fields[f] = dec
                    comp_bytes[f] = nbytes
                    eb[f] = 0.0
                    status[f] = 'lossless'
                self.lossless_fields = lossless
                errors = all_qoi_errors(self.fields, dec_fields)
                failing = [q for q in QOI_NAMES if errors[q] > self.tau[q]]
            # Anything still failing is genuinely infeasible (should be empty).
            self.infeasible_qois = self._infeasible_among(failing) if failing else []

        return self._build_result(eb, status, comp_bytes, errors, failing)

    def _build_result(self, eb, status, comp_bytes, errors, failing) -> dict:
        orig_total = sum(int(self.fields[f].nbytes) for f in FIELD_NAMES)
        comp_total = sum(comp_bytes[f] for f in FIELD_NAMES)
        fields_out = {}
        for f in FIELD_NAMES:
            r = self.field_range[f]
            fields_out[f] = {
                'abs_eb': eb[f],
                'rel_eb': eb[f] / r if r > 0 else float('inf'),
                'status': status[f],
                'compressed_bytes': comp_bytes[f],
                'ratio': self.fields[f].nbytes / comp_bytes[f],
            }
        qois_out = {
            q: {'rel_err': errors[q], 'bound': self.tau[q],
                'pass': errors[q] <= self.tau[q]}
            for q in QOI_NAMES
        }
        return {
            'compressor': self.c.name,
            'qoi_bounds': dict(self.tau),
            'fields': fields_out,
            'qois': qois_out,
            'orig_bytes': orig_total,
            'compressed_bytes': comp_total,
            'total_ratio': orig_total / comp_total,
            'lossless_fields': sorted(set(self.lossless_fields)),
            'infeasible_qois': sorted(set(self.infeasible_qois)),
            'verification_failures': failing,
            'all_pass': not failing,
        }


def run_search(compressor: BaseCompressor, fields: Dict[str, np.ndarray],
               qoi_bounds: Dict[str, float], **kwargs) -> dict:
    """Convenience wrapper: build a :class:`ParameterSearch` and run it."""
    return ParameterSearch(compressor, fields, qoi_bounds, **kwargs).run()
