#!/usr/bin/env python3
"""Pluggable error-bounded lossy compressor backends.

Each backend exposes a uniform interface::

    decompressed, compressed_bytes = compressor.round_trip(array, abs_eb)

where ``array`` is a 3D float32 numpy array (nz, ny, nx), ``abs_eb`` is the
absolute point-wise error bound, ``decompressed`` is the round-tripped array
(same shape/dtype), and ``compressed_bytes`` is the compressed stream size in
bytes (used to compute compression ratios).

Three backends are provided so the parameter search can drive SZ3, zfp, and
SPERR interchangeably:

* ``SZ3Compressor``   -- SZ3 via the ``pysz`` Python package (ABS mode).
* ``ZfpCompressor``   -- zfp via the ``zfpy`` bindings (fixed-accuracy).
* ``SperrCompressor`` -- SPERR via the ``sperr3d`` CLI (no Python binding).
"""

import os
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Tuple

import numpy as np

# Project root: .../parameter_search/compressors.py -> python_src -> <root>.
_ROOT = Path(__file__).resolve().parents[2]
_EXTERNAL = _ROOT / "external"


def _ensure_on_path(path: Path) -> None:
    p = str(path)
    if path.is_dir() and p not in sys.path:
        sys.path.insert(0, p)


class BaseCompressor(ABC):
    """Common interface + a small bounded result cache.

    The cache catches immediate repeats (e.g. re-compressing the final bound
    that equals the last tested midpoint) without retaining many large arrays.
    """

    name = "base"

    def __init__(self, cache_size: int = 4):
        self._cache: "OrderedDict[tuple, Tuple[np.ndarray, int]]" = OrderedDict()
        self._cache_size = cache_size

    def round_trip(self, array: np.ndarray, abs_eb: float) -> Tuple[np.ndarray, int]:
        """Compress ``array`` at absolute error bound ``abs_eb`` then decompress."""
        array = np.ascontiguousarray(array, dtype=np.float32)
        key = (id(array), array.shape, float(abs_eb))
        if self._cache_size > 0 and key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        result = self._round_trip(array, float(abs_eb))
        if self._cache_size > 0:
            self._cache[key] = result
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return result

    @abstractmethod
    def _round_trip(self, array: np.ndarray, abs_eb: float) -> Tuple[np.ndarray, int]:
        ...

    def lossless_round_trip(self, array: np.ndarray) -> Tuple[np.ndarray, int]:
        """Exact (lossless) round-trip. Uses the backend's native lossless mode
        when it reproduces the input bit-for-bit, else falls back to raw storage
        (guaranteed exact, ratio 1.0)."""
        array = np.ascontiguousarray(array, dtype=np.float32)
        dec, size = self._lossless(array)
        if dec is not None and dec.shape == array.shape and np.array_equal(dec, array):
            return dec, int(size)
        return array.copy(), int(array.nbytes)  # raw fallback

    def _lossless(self, array: np.ndarray):
        """Native lossless attempt. Returns (decompressed, bytes) or (None, 0)
        when the backend has no lossless mode."""
        return None, 0


class SZ3Compressor(BaseCompressor):
    """SZ3 through the ``pysz`` package, absolute error-bound mode."""

    name = "sz3"

    def __init__(self, cache_size: int = 4):
        super().__init__(cache_size)
        # ``pysz`` is normally pip-installed; fall back to the vendored source.
        _ensure_on_path(_EXTERNAL / "SZ3" / "tools" / "pysz" / "src")
        from pysz import sz, szConfig, szErrorBoundMode  # noqa: F401
        self._sz = sz
        self._szConfig = szConfig
        self._mode = szErrorBoundMode

    def _round_trip(self, array: np.ndarray, abs_eb: float) -> Tuple[np.ndarray, int]:
        config = self._szConfig()
        config.errorBoundMode = self._mode.ABS
        config.absErrorBound = abs_eb
        compressed, _ratio = self._sz.compress(array, config)
        dec, _cfg = self._sz.decompress(compressed, np.float32, array.shape)
        dec = np.asarray(dec, dtype=np.float32).reshape(array.shape)
        return dec, int(np.asarray(compressed).nbytes)

    def _lossless(self, array: np.ndarray):
        # SZ3 reproduces input exactly with an absolute error bound of 0.
        config = self._szConfig()
        config.errorBoundMode = self._mode.ABS
        config.absErrorBound = 0.0
        compressed, _ = self._sz.compress(array, config)
        dec, _ = self._sz.decompress(compressed, np.float32, array.shape)
        dec = np.asarray(dec, dtype=np.float32).reshape(array.shape)
        return dec, int(np.asarray(compressed).nbytes)


class ZfpCompressor(BaseCompressor):
    """zfp through the ``zfpy`` bindings, fixed-accuracy (absolute tolerance).

    Note: zfp often achieves error well below the requested tolerance. That is
    safe (tighter than asked) but conservative; the search drives by *measured*
    QoI error, so this only affects the achievable compression ratio.
    """

    name = "zfp"

    def __init__(self, cache_size: int = 4):
        super().__init__(cache_size)
        try:
            import zfpy  # noqa: F401
        except ImportError:
            # Add the installed zfpy site-packages (python version may vary).
            for sp in sorted((_EXTERNAL / "zfp-install" / "lib").glob(
                    "python3*/site-packages")):
                _ensure_on_path(sp)
            import zfpy  # noqa: F401
        self._zfpy = zfpy

    def _round_trip(self, array: np.ndarray, abs_eb: float) -> Tuple[np.ndarray, int]:
        compressed = self._zfpy.compress_numpy(array, tolerance=abs_eb)
        dec = self._zfpy.decompress_numpy(compressed)
        dec = np.ascontiguousarray(dec, dtype=np.float32).reshape(array.shape)
        return dec, int(len(compressed))

    def _lossless(self, array: np.ndarray):
        # zfp's reversible mode (no tolerance/rate/precision) is lossless.
        compressed = self._zfpy.compress_numpy(array)
        dec = self._zfpy.decompress_numpy(compressed)
        dec = np.ascontiguousarray(dec, dtype=np.float32).reshape(array.shape)
        return dec, int(len(compressed))


class SperrCompressor(BaseCompressor):
    """SPERR through the ``sperr3d`` CLI (point-wise error mode).

    SPERR's ``--dims`` expects the fastest-varying dimension FIRST. Our numpy
    arrays are C-order (nz, ny, nx) with x fastest, so we pass ``nx ny nz``.
    """

    name = "sperr"

    def __init__(self, cache_size: int = 4, binary: str = None):
        super().__init__(cache_size)
        self._bin = binary or str(_EXTERNAL / "SPERR-install" / "bin" / "sperr3d")
        if not Path(self._bin).exists():
            raise FileNotFoundError(f"sperr3d binary not found: {self._bin}")

    def _round_trip(self, array: np.ndarray, abs_eb: float) -> Tuple[np.ndarray, int]:
        nz, ny, nx = array.shape
        tmpdir = tempfile.mkdtemp(prefix="sperr_")
        in_raw = os.path.join(tmpdir, "in.raw")
        bs = os.path.join(tmpdir, "stream.sperr")
        out_raw = os.path.join(tmpdir, "out.raw")
        try:
            array.tofile(in_raw)
            subprocess.run(
                [self._bin, "-c", "--ftype", "32",
                 "--dims", str(nx), str(ny), str(nz),
                 "--pwe", repr(abs_eb),
                 "--bitstream", bs, in_raw],
                check=True, capture_output=True, timeout=300)
            comp_size = os.path.getsize(bs)
            subprocess.run(
                [self._bin, "-d", "--decomp_f", out_raw, bs],
                check=True, capture_output=True, timeout=300)
            dec = np.fromfile(out_raw, dtype=np.float32).reshape(nz, ny, nx)
            return dec, int(comp_size)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode(errors="replace") if exc.stderr else ""
            raise RuntimeError(
                f"SPERR failed (eb={abs_eb}): {stderr.strip()}") from exc
        finally:
            for f in (in_raw, bs, out_raw):
                try:
                    os.remove(f)
                except OSError:
                    pass
            try:
                os.rmdir(tmpdir)
            except OSError:
                pass


_COMPRESSORS = {
    "sz3": SZ3Compressor,
    "zfp": ZfpCompressor,
    "sperr": SperrCompressor,
}


def get_compressor(name: str, **kwargs) -> BaseCompressor:
    """Factory: return a compressor backend by name ('sz3', 'zfp', 'sperr')."""
    key = name.lower()
    if key not in _COMPRESSORS:
        raise ValueError(
            f"Unknown compressor '{name}'. Choices: {sorted(_COMPRESSORS)}")
    return _COMPRESSORS[key](**kwargs)


def available_compressors() -> list:
    """Names of all registered compressor backends."""
    return list(_COMPRESSORS.keys())
