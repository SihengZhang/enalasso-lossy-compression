"""QoI-preserving lossy-compression parameter search.

Find the highest per-field absolute error bound such that all derived
turbulence QoIs stay within a given relative error bound, for any of three
interchangeable error-bounded compressors (SZ3, zfp, SPERR).

Self-contained: this package does not depend on the ``qpet`` module.
"""

from .compressors import (
    BaseCompressor,
    SZ3Compressor,
    ZfpCompressor,
    SperrCompressor,
    available_compressors,
    get_compressor,
)
from .qoi_eval import (
    FIELD_NAMES,
    QOI_FIELD_DEPENDENCIES,
    QOI_NAMES,
    all_qoi_errors,
    compute_perturbations,
    compute_qois,
    compute_single_qoi,
    load_binary_fields,
    qoi_rel_error,
    qoi_value_ranges,
)
from .search import ParameterSearch, binary_search_bound, run_search

__all__ = [
    'BaseCompressor', 'SZ3Compressor', 'ZfpCompressor', 'SperrCompressor',
    'available_compressors', 'get_compressor',
    'FIELD_NAMES', 'QOI_NAMES', 'QOI_FIELD_DEPENDENCIES',
    'compute_perturbations', 'compute_qois', 'compute_single_qoi',
    'qoi_value_ranges', 'qoi_rel_error', 'all_qoi_errors', 'load_binary_fields',
    'ParameterSearch', 'binary_search_bound', 'run_search',
]
