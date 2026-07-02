"""Strict error bounds for full x-y-plane turbulence QoIs (PyTorch engine).

Each QoI is a function of the data points of an entire x-y plane of the raw
fields (n = 65536 for a 256x256 plane; doubled for bilinear QoIs). For a center
``X0`` and per-variable displacement ``E`` (region ``|X-X0| <= E``) the engine
builds the first-order Taylor proxy ``g(X) = F(X0) + grad_F(X0)^T (X-X0)`` and a
strict bound ``t = 0.5 E^T B E >= |F(X)-g(X)|`` with ``B_ij = max_box|d^2F/dx_i
dx_j|``. Gradients come from autograd; ``B`` is never materialized -- the QoI
Hessians are constant (variance/bilinear) or diagonal+low-rank (wskew), so ``t``
is O(N). See ``qoi_library.py`` and ``engine.py``.

The submodules use flat imports (matching how ``demo.py`` is run as a script),
so this package adds its own directory to ``sys.path`` on import to make both
``import linear_proxy`` and ``python3 .../demo.py`` work.
"""

import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)

from engine import LinearProxyEngine, resolve_device
from qoi_library import QOI_REGISTRY, QOI_NAMES, QoI, compute_perturbation

__all__ = [
    'LinearProxyEngine', 'resolve_device',
    'QOI_REGISTRY', 'QOI_NAMES', 'QoI', 'compute_perturbation',
]
