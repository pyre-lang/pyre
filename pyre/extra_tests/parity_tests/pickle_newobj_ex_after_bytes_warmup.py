# CPython-suite gap: `test_complex_newobj_ex` only fails after `test_bytes`
# has compiled `Unpickler.load`; the suite does not isolate that order.
# parity-tests reason: a forced/resumed load frame's type-9 locals array
# must not keep popped young words above `valuestackdepth`.
# parity-env: PYPY_GC_NURSERY=8192
# parity-env: MAJIT_GC_NURSERY_POISON=1
#
# pickle_bytearray_pair_memo:
# `test.test_pickle.CPicklerTests.test_bytearray_memoization` dumps a repeated bytearray
# under every protocol; the suite does not pin a small nursery around BYTEARRAY8.
# `save_bytearray` must copy `view.as_str()` before BYTEARRAY8 / memoize, because the
# bytearray is nursery-movable.

"""Warm `Unpickler.load`, then unpickle a NEWOBJ_EX payload.

`test_bytes` makes the load loop hot. The next NEWOBJ_EX residual forces
that frame and calls `int.__new__`, which collects. Before the trim,
`write_from_resume_data_partial` rewrote every locals-array slot from the
vable boxes — including the popped ones — and the type-9 walker treated
those words as live edges (`holder_type_id=9`, `holder_offset=56`).
"""

import io
import pickle


class ComplexNewObjEx(int):
    def __init__(self, *args, **kwargs):
        raise TypeError("ComplexNewObjEx.__init__ must not run")

    def __getnewargs_ex__(self):
        return ("%X" % self,), {"base": 16}

    def __eq__(self, other):
        return int(self) == int(other) and self.__dict__ == other.__dict__


# Warm `_Unpickler.load` the way `CPicklerTests.test_bytes` does: many
# framed loads of byte strings across protocols.
for proto in range(pickle.HIGHEST_PROTOCOL + 1):
    for n in range(80):
        payload = bytes((n & 0xFF, proto, 0x80, 0xFF))
        restored = pickle._Unpickler(io.BytesIO(pickle.dumps(payload, proto))).load()
        assert restored == payload, (proto, n, restored)

x = ComplexNewObjEx.__new__(ComplexNewObjEx, 0xFACE)
x.abc = 666
for proto in range(pickle.HIGHEST_PROTOCOL + 1):
    stream = io.BytesIO()
    pickle._Pickler(stream, proto).dump(x)
    data = stream.getvalue()
    if proto >= 4:
        assert pickle.NEWOBJ_EX in data, proto
    y = pickle._Unpickler(io.BytesIO(data)).load()
    assert y == x, (proto, y, x)
    assert y.abc == 666, (proto, y.abc)

def pickle_bytearray_pair_memo():
    import _pickle
    import pickle

    assert pickle.dumps is _pickle.dumps
    assert pickle.loads is _pickle.loads

    for proto in range(pickle.HIGHEST_PROTOCOL + 1):
        for payload in (b"", b"xyz", b"xyz" * 100):
            b = bytearray(payload)
            b1, b2 = pickle.loads(pickle.dumps((b, b), proto))
            assert b1 is b2
            assert bytes(b1) == payload
            a, c = bytearray(payload), bytearray(payload)
            d, e = pickle.loads(pickle.dumps((a, c), proto))
            assert d is not e
            assert bytes(d) == payload
            assert bytes(e) == payload
pickle_bytearray_pair_memo()
print("OK")
