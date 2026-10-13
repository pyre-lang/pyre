# pyre-check: platforms=darwin,linux
"""dlopen callers reject a name with an embedded NUL as ValueError."""
import ctypes

try:
    ctypes.CDLL("foo\0bar")
except ValueError as exc:
    assert "embedded null" in str(exc), exc
else:
    raise SystemExit("ctypes.CDLL accepted an embedded null")

try:
    import _cffi_backend
except ImportError:
    raise SystemExit(0)

try:
    _cffi_backend.FFI().dlopen("foo\0bar")
except ValueError as exc:
    assert "embedded null" in str(exc), exc
else:
    raise SystemExit("_cffi_backend.FFI.dlopen accepted an embedded null")
