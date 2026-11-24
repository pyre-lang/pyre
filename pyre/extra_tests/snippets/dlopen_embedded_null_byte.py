# pyre-check: platforms=darwin,linux
# pyre-check: requires-modules=_ctypes,_cffi_backend
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

# `func.py from_buffer` / `W_CDataFromBuffer`: the cdata owns a `buffer_w`
# lease for its lifetime; `release` / `__exit__` / the finalizer drop it.
import array
import gc

B = _cffi_backend
BCharA = B.new_array_type(B.new_pointer_type(B.new_primitive_type("char")), None)

ba = bytearray(b"abc")
c = B.from_buffer(BCharA, ba)
assert len(c) == 3
try:
    ba.append(100)
except BufferError as exc:
    assert "Existing exports of data" in str(exc), exc
else:
    raise SystemExit("from_buffer cdata did not pin the bytearray export")
assert c[0] == b"a"

assert B.from_buffer(BCharA, memoryview(bytearray(b"xyz")))[1] == b"y"
assert B.from_buffer(BCharA, b"qr")[0] == b"q"
try:
    B.from_buffer(BCharA, "s")
except TypeError as exc:
    assert "unicode object" in str(exc), exc
else:
    raise SystemExit("from_buffer accepted a unicode object")
try:
    B.from_buffer(BCharA, memoryview(b"abcd")[::2])
except BufferError as exc:
    assert "not C-contiguous" in str(exc), exc
else:
    raise SystemExit("from_buffer accepted a strided memoryview")
assert len(B.from_buffer(BCharA, array.array("i", [1, 2]))) == 8

m = memoryview(bytearray(b"zz"))
c_mv = B.from_buffer(BCharA, m)
m.release()
assert "memoryview" in repr(c_mv)

ba_rel = bytearray(b"abc")
c_rel = B.from_buffer(BCharA, ba_rel)
B.release(c_rel)
ba_rel.append(100)
assert bytes(ba_rel) == b"abcd"

ba_with = bytearray(b"abc")
c_with = B.from_buffer(BCharA, ba_with)
with c_with:
    pass
ba_with.append(100)
assert bytes(ba_with) == b"abcd"

ba_gc = bytearray(b"abc")
c_gc = B.from_buffer(BCharA, ba_gc)
del c_gc
gc.collect()
gc.collect()
ba_gc.append(100)
assert bytes(ba_gc) == b"abcd"

try:
    B.from_buffer(BCharA, b"ab", True)
except BufferError as exc:
    assert str(exc) == "Object is not writable.", exc
else:
    raise SystemExit("from_buffer(bytes, require_writable=True) succeeded")

class CountBuf:
    acquires = 0
    releases = 0

    def __init__(self, data):
        self.data = bytearray(data)

    def __buffer__(self, flags):
        type(self).acquires += 1
        return memoryview(self.data)

    def __release_buffer__(self, view):
        type(self).releases += 1

CountBuf.acquires = 0
CountBuf.releases = 0
cb = CountBuf(b"xy")
c_pep = B.from_buffer(BCharA, cb)
assert CountBuf.acquires == 1, CountBuf.acquires
assert CountBuf.releases == 0, CountBuf.releases
B.release(c_pep)
assert CountBuf.acquires == 1, CountBuf.acquires
assert CountBuf.releases == 0, CountBuf.releases
del c_pep
gc.collect()
assert CountBuf.acquires == 1, CountBuf.acquires
assert CountBuf.releases == 0, CountBuf.releases

# `W_MemoryView.buffer_w` / `NonOwningReleaseView`: after `m.release()` the
# cdata still holds the backing, so reads and writes through `c` succeed.
m_hold = memoryview(bytearray(b"abcdefgh"))
c_hold = B.from_buffer(BCharA, m_hold)
m_hold.release()
del m_hold
for _ in range(32):
    _ = bytearray(4096)
    gc.collect()
assert c_hold[0] == b"a", c_hold[0]
assert c_hold[7] == b"h", c_hold[7]
c_hold[0] = b"Z"
assert c_hold[0] == b"Z", c_hold[0]
assert c_hold[1] == b"b", c_hold[1]

# `__buffer__` returns a memoryview of an otherwise-unreferenced bytearray.
# After that view is released the cdata still holds the backing (`self.buf`).
class ViewBuf:
    releases = 0

    def __buffer__(self, flags):
        self.view = memoryview(bytearray(b"wxyz"))
        return self.view

    def __release_buffer__(self, view):
        type(self).releases += 1

ViewBuf.releases = 0
vb = ViewBuf()
c_vb = B.from_buffer(BCharA, vb)
vb.view.release()
del vb.view
for _ in range(32):
    _ = bytearray(4096)
    gc.collect()
assert c_vb[0] == b"w", c_vb[0]
assert c_vb[3] == b"z", c_vb[3]

try:
    B.from_buffer(BCharA, memoryview(b"abcd")[::2], True)
except BufferError as exc:
    assert str(exc) == "Object is not writable.", exc
else:
    raise SystemExit("from_buffer(strided memoryview, require_writable=True) succeeded")

# `func.py get_raw_address_of_string` / `RawBytesCache.wdict`: two calls on
# the same live bytes object return the same pointer.
intptr = B.new_primitive_type("intptr_t")

def _buf_addr(cdata):
    return int(B.cast(intptr, cdata))

b_same = bytes(range(8))
c_same_1 = B.from_buffer(BCharA, b_same)
c_same_2 = B.from_buffer(BCharA, b_same)
assert _buf_addr(c_same_1) == _buf_addr(c_same_2)

# Distinct equal bytes objects are distinct keys (`ll_keyeq` is identity).
b_eq_1 = bytes([101, 113, 117, 97, 108, 45, 99, 111, 110, 116, 101, 110, 116])
b_eq_2 = bytes([101, 113, 117, 97, 108, 45, 99, 111, 110, 116, 101, 110, 116])
assert b_eq_1 == b_eq_2 and b_eq_1 is not b_eq_2
c_eq_1 = B.from_buffer(BCharA, b_eq_1)
c_eq_2 = B.from_buffer(BCharA, b_eq_2)
assert _buf_addr(c_eq_1) != _buf_addr(c_eq_2)

# Dead keys drop their `RawBytes` on collect; a still-live key keeps a
# consistent pointer across `gc.collect()`.
kept_bytes = None
kept_addr = None
for i in range(32):
    x = bytes([i, i, i, i, i, i, i, i])
    c = B.from_buffer(BCharA, x)
    p = _buf_addr(c)
    assert _buf_addr(B.from_buffer(BCharA, x)) == p
    kept_bytes, kept_addr = x, p
    del c
    gc.collect()
assert _buf_addr(B.from_buffer(BCharA, kept_bytes)) == kept_addr

# Native `buffer_w` before a Python `__buffer__` (`ObjSpace._try_buffer_w`).
# A bytes / bytearray subclass that defines `__buffer__` still exports the
# native payload; the override is not called.
ovr_calls = []


class BA(bytearray):
    def __buffer__(self, flags):
        ovr_calls.append("ba")
        return memoryview(bytearray(b"OVERRIDE-LONGER"))


class BY(bytes):
    def __buffer__(self, flags):
        ovr_calls.append("by")
        return memoryview(b"XY")


c_ba = B.from_buffer(BCharA, BA(b"abc"))
assert len(c_ba) == 3, len(c_ba)
assert c_ba[0] == b"a", c_ba[0]
assert ovr_calls == [], ovr_calls
c_by = B.from_buffer(BCharA, BY(b"abcdef"))
assert len(c_by) == 6, len(c_by)
assert c_by[0] == b"a", c_by[0]
assert ovr_calls == [], ovr_calls


class Raise(bytearray):
    def __buffer__(self, flags):
        raise RuntimeError("boom")


c_raise = B.from_buffer(BCharA, Raise(b"q"))
assert len(c_raise) == 1, len(c_raise)
assert c_raise[0] == b"q", c_raise[0]

# ctypes exporter: `wrap_getbuffer` / `NonOwningReleaseView` — `B.release`
# does not run the native release. A second cdata from the same object
# still reads.
ct_arr = (ctypes.c_char * 4)(*b"wxyz")
c_ct = B.from_buffer(BCharA, ct_arr)
assert c_ct[0] == b"w", c_ct[0]
assert c_ct[3] == b"z", c_ct[3]
B.release(c_ct)
gc.collect()
for _ in range(32):
    _ = bytearray(4096)
    gc.collect()
c_ct2 = B.from_buffer(BCharA, ct_arr)
assert c_ct2[0] == b"w", c_ct2[0]
assert c_ct2[3] == b"z", c_ct2[3]

# Concurrent `from_buffer` on fresh bytes plus nursery churn. A waiter
# leaves the running-mutator census so a collection inside `RawBytesCache`
# proceeds while that thread is blocked on the lock.
import threading

_rawbytes_errors = [None] * 4


def _rawbytes_worker(tid):
    try:
        for i in range(200):
            first = i % 251
            c = B.from_buffer(BCharA, bytes([first]) * 64)
            assert c[0] == bytes([first]), (c[0], first)
            if i % 7 == 0:
                _ = bytearray(4096)
    except BaseException as exc:
        _rawbytes_errors[tid] = exc


_rawbytes_threads = [
    threading.Thread(target=_rawbytes_worker, args=(tid,)) for tid in range(4)
]
for t in _rawbytes_threads:
    t.start()
for t in _rawbytes_threads:
    t.join(60)
    assert not t.is_alive()
assert not any(_rawbytes_errors), _rawbytes_errors

# A released from_buffer cdata over bytes keeps the STR, not the bytes
# wrapper (`W_CDataFromBuffer.buf` is `StringBuffer(self._value)`).
_finalized = []


class BS(bytes):
    def __del__(self):
        _finalized.append(1)


bs = BS(b"abcd")
c_bs = B.from_buffer(BCharA, bs)
assert c_bs[0] == b"a"
B.release(c_bs)
del bs
gc.collect()
gc.collect()
assert _finalized == [1], _finalized

_finalized_mv = []


class BSmv(bytes):
    def __del__(self):
        _finalized_mv.append(1)


bs_mv = BSmv(b"abcd")
c_bs_mv = B.from_buffer(BCharA, memoryview(bs_mv))
assert c_bs_mv[0] == b"a"
B.release(c_bs_mv)
del bs_mv
gc.collect()
gc.collect()
assert _finalized_mv == [1], _finalized_mv
