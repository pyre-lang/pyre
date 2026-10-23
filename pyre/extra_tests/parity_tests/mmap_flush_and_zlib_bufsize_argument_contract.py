# pyre-check: pypy-diverges: pypy3's `rmmap.flush` takes `size=0` to mean the
# whole mapping, so `flush(PAGESIZE)` and `flush(PAGESIZE, -1)` raise ValueError
# "flush values out of range", and its `zlib.decompress` never reads `bufsize`, so a
# negative one is accepted.
#
# CPython-suite gap: `test_mmap` flushes only the whole mapping or a
# page-aligned range, never a closed mapping with an argument that does not
# convert, and `test_zlib` never passes a negative `bufsize` to the one-shot
# `decompress`.
#
# parity-tests reason: `mmap.flush` converts `offset` and `size` through the
# index protocol before it checks that the mapping is open.  Its `size`
# defaults to "from `offset` to the end", as an explicit -1 does.  Offsets
# stay page-aligned because `msync` refuses an unaligned start, and no case
# flushes zero bytes because macOS `msync` refuses a zero length.
# A negative `bufsize` is a ValueError because the clinic declares it
# non-negative.
import mmap
import zlib


def raises(exc, fn, *args, match=None):
    try:
        fn(*args)
    except exc as e:
        if match is not None:
            assert str(e) == match, (fn, args, str(e))
    else:
        raise AssertionError((fn, args, "did not raise", exc))


P = mmap.PAGESIZE


class Index:
    def __index__(self):
        return P


class IntOnly:
    def __int__(self):
        return 1


m = mmap.mmap(-1, 2 * P)
for args in ((), (P,), (P, -1), (Index(),), (0, Index())):
    assert m.flush(*args) is None, args
for args in ((2 * P + 1,), (-1,), (-(2**63),), (0, 2 * P + 1), (P, P + 1), (0, -2)):
    raises(ValueError, m.flush, *args, match="flush values out of range")
raises(TypeError, m.flush, IntOnly())
raises(TypeError, m.flush, 0, IntOnly())
m.close()
raises(TypeError, m.flush, "x")
raises(TypeError, m.flush, 0, "x")
raises(ValueError, m.flush, 0, 1, match="mmap closed or invalid")

data = zlib.compress(b"x")
raises(ValueError, zlib.decompress, data, 15, -1, match="bufsize cannot be negative")
assert zlib.decompress(data, 15, 0) == b"x"

print("OK")
