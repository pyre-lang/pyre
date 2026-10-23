# pyre-check: gate=1
# CPython-suite gap: `test_zlib` never passes an explicit `None` for an
# integer argument, never flushes a `Decompress` object twice after the end of
# its stream, and never uses a `Compress` object after `flush(Z_FINISH)`.
#
# The integer arguments are `@unwrap_spec(...=int)` / `mode="c_int"` in
# `interp_zlib.py` (clinic `int` in `zlibmodule.c`), so `None` is a TypeError.
# `Decompress.flush` returns `b''` once the stream is released
# (`if not self.stream`), and a released `Compress` raises `zlib.error`.
import zlib

data = zlib.compress(b"hello")

for call in (
    lambda: zlib.compress(b"", None),
    lambda: zlib.compress(b"", wbits=None),
    lambda: zlib.decompress(data, None),
    lambda: zlib.decompress(data, 15, None),
    lambda: zlib.compressobj(None),
    lambda: zlib.compressobj(level=None),
    lambda: zlib.compressobj(method=None),
    lambda: zlib.compressobj(wbits=None),
    lambda: zlib.compressobj(memLevel=None),
    lambda: zlib.compressobj(strategy=None),
    lambda: zlib.decompressobj(None),
    lambda: zlib.decompressobj(wbits=None),
    lambda: zlib.compressobj().flush(None),
    lambda: zlib.decompressobj().decompress(b"", None),
    lambda: zlib.decompressobj().flush(None),
):
    try:
        call()
    except TypeError:
        pass
    else:
        raise AssertionError("None accepted as an integer argument")

# Omitted arguments still take their defaults.
assert zlib.decompress(zlib.compress(b"abc")) == b"abc"
co = zlib.compressobj()
assert zlib.decompress(co.compress(b"xyz") + co.flush()) == b"xyz"
assert zlib.decompressobj().decompress(data) == b"hello"

d = zlib.decompressobj()
assert d.decompress(data) == b"hello"
assert d.eof
assert d.flush() == b""
assert d.flush() == b""
assert d.flush(10) == b""
try:
    d.flush(0)
except ValueError:
    pass
else:
    raise AssertionError("flush(0) accepted")

c = zlib.compressobj()
c.compress(b"abc")
c.flush(zlib.Z_FINISH)
for call in (lambda: c.compress(b"x"), lambda: c.flush(), lambda: c.flush(zlib.Z_SYNC_FLUSH)):
    try:
        call()
    except zlib.error:
        pass
    else:
        raise AssertionError("released compressor accepted more input")

try:
    zlib.decompressobj().decompress(b"", -1)
except ValueError:
    pass
else:
    raise AssertionError("negative max_length accepted")

# `wbits` is `"c_int"` (clinic `int`): out of the C int range is an
# OverflowError, not a wrapped value.
try:
    zlib.decompress(data, 2**32 + 15)
except OverflowError:
    pass
else:
    raise AssertionError("wbits wrapped to a C int")

print("OK")
