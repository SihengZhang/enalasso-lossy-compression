"""Sparse QoI-preserving edits to a compressed field (W / wskew feasibility test).

See ``sparse_edit.py`` for the optimizer and ``run.py`` for the CLI driver.
"""

from .sparse_edit import (
    SparseQoIEditor,
    SparseQoIConfig,
    batched_wskew,
    row_sigma,
)

__all__ = [
    "SparseQoIEditor",
    "SparseQoIConfig",
    "batched_wskew",
    "row_sigma",
]
