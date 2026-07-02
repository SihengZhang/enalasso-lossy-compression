"""Turbulence QoIs as PyTorch computation graphs over full x-y-plane fields.

Each QoI is the x-y-plane average defined in ``docs/derived_variables.md`` and
``process_3dles.py``. The VARIABLES are the raw data points of an entire x-y
plane (flattened to length ``N = ny*nx``) of one or two of the six raw fields
(U, V, W, PP, TABS, QV). Reynolds perturbation ``phi' = phi - mean(phi)`` is a
linear function of the raw field, so every QoI is a differentiable function of
the raw field data points directly -- no reduced ``(w', sigma^2)`` surrogate.

Variable count per QoI:
  - single-field (uvar, vvar, wvar, wskew): n = N
  - bilinear (wpup, wpvp, wppp, wptp, wpqvp): n = 2N (two fields)

The ``kind`` field tags the Hessian structure the bound engine exploits:
  - 'var'      : F = (1/N)||P phi||^2          -> Hessian (2/N)P, CONSTANT
  - 'bilinear' : F = (1/N)(P w).(P x)          -> Hessian off-diag (1/N)P, CONSTANT
  - 'wskew'    : F = sqrt(N) S3 / S2^{3/2}      -> Hessian diagonal + low-rank (varies)
where P = I - (1/N) 11^T is the mean-centering projection.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import torch


def compute_perturbation(phi: torch.Tensor) -> torch.Tensor:
    """Reynolds perturbation phi' = phi - mean(phi) over the flattened plane."""
    return phi - phi.mean()


# --------------------------------------------------------------------------- #
# QoI computation graphs (each returns a 0-dim tensor; autograd-friendly)
# --------------------------------------------------------------------------- #
def _variance(phi: torch.Tensor) -> torch.Tensor:
    """<phi'^2> = (1/N) sum (phi - mean(phi))^2."""
    c = compute_perturbation(phi)
    return (c * c).mean()


def _bilinear(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """<a' b'> = (1/N) sum (a-mean(a))(b-mean(b))."""
    return (compute_perturbation(a) * compute_perturbation(b)).mean()


def _wskew(w: torch.Tensor) -> torch.Tensor:
    """<((w'-mu)/sigma)^3> = sqrt(N) * S3 / S2^{3/2}  (mu = mean(w') = 0).

    Equals mean(w'^3) / sigma^3 with sigma^2 = mean(w'^2). Matches
    process_3dles.py (mu ~ 0 by construction after mean removal).
    """
    c = compute_perturbation(w)
    n = w.numel()
    s2 = (c * c).sum()
    s3 = (c * c * c).sum()
    return (n ** 0.5) * s3 / s2 ** 1.5


@dataclass(frozen=True)
class QoI:
    """A turbulence QoI: its fields, computation graph, and Hessian structure.

    Attributes:
        name:   QoI identifier (e.g. 'wpup').
        fields: ordered raw fields the QoI consumes (1 or 2 of the six fields).
        kind:   'var' | 'bilinear' | 'wskew' -- selects the structured bound.
        fn:     callable(*field_tensors) -> scalar tensor (the QoI value).
        desc:   one-line human-readable description.
    """
    name: str
    fields: Tuple[str, ...]
    kind: str
    fn: Callable[..., torch.Tensor]
    desc: str

    def __call__(self, *field_tensors: torch.Tensor) -> torch.Tensor:
        return self.fn(*field_tensors)


# Registry of all nine QoIs (definitions per docs/derived_variables.md).
QOI_REGISTRY: Dict[str, QoI] = {
    'uvar':  QoI('uvar',  ('U',),        'var',      _variance, "zonal velocity variance <u'^2>"),
    'vvar':  QoI('vvar',  ('V',),        'var',      _variance, "meridional velocity variance <v'^2>"),
    'wvar':  QoI('wvar',  ('W',),        'var',      _variance, "vertical velocity variance <w'^2>"),
    'wpup':  QoI('wpup',  ('W', 'U'),    'bilinear', _bilinear, "zonal momentum flux <w'u'>"),
    'wpvp':  QoI('wpvp',  ('W', 'V'),    'bilinear', _bilinear, "meridional momentum flux <w'v'>"),
    'wppp':  QoI('wppp',  ('W', 'PP'),   'bilinear', _bilinear, "pressure-velocity cov <w'p'>"),
    'wptp':  QoI('wptp',  ('W', 'TABS'), 'bilinear', _bilinear, "sensible heat flux <w'T'>"),
    'wpqvp': QoI('wpqvp', ('W', 'QV'),   'bilinear', _bilinear, "moisture flux <w'qv'>"),
    'wskew': QoI('wskew', ('W',),        'wskew',    _wskew,    "vertical velocity skewness <((w'-mu)/sigma)^3>"),
}

QOI_NAMES = list(QOI_REGISTRY.keys())
