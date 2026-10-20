# pyre-check: pypy-diverges: pypy3's Unpickler.memo is a snapshot dict from get_memo_w, so memo.clear() does not reset the unpickler; it also has no sparse w_memo_dict.
# CPython-suite gap: pickletester memo tests never PUT a sparse index, so UnpicklerMemoProxy.copy / clear / set_memo are not checked against memo_dict.
# parity-tests reason: pin CPython 3.15 `_Unpickler_MemoGet` / `_pickle_UnpicklerMemoProxy_copy_impl` / `_pickle_UnpicklerMemoProxy_clear_impl` / `Unpickler_set_memo` / `_pickle_Unpickler___init___impl` observables.

"""Unpickler memo copy/clear/set_memo follow CPython 3.15's dense-array surface."""

import io
import _pickle


def unpickler(data):
    return _pickle.Unpickler(io.BytesIO(data))


def expect(exc, msg, fn):
    try:
        fn()
    except exc as err:
        assert str(err) == msg, err
    else:
        raise AssertionError(msg)


# `_pickle_UnpicklerMemoProxy_copy_impl`: copy reads the dense array only.
u = unpickler(b"I42\np100\n.g100\n.")
assert u.load() == 42
assert u.memo.copy() == {}

# `_pickle_UnpicklerMemoProxy_clear_impl` = `_Unpickler_MemoCleanup` (dense only);
# memo_dict survives, so the next GET still finds the sparse PUT.
u.memo.clear()
assert u.memo.copy() == {}
assert u.load() == 42

# `_pickle_Unpickler___init___impl` / `Unpickler_clear`: Py_CLEAR(memo_dict).
u.__init__(io.BytesIO(b"g100\n."))
expect(
    _pickle.UnpicklingError,
    "Memo value not found at index 100",
    u.load,
)

# Proxy assignment copies the source's dense array only.
src = unpickler(b"I1\np0\nI42\np100\n.")
assert src.load() == 42
assert src.memo.copy() == {0: 1}
p = unpickler(b"g0\n.")
p.memo = src.memo
assert p.load() == 1
assert p.memo.copy() == {0: 1}
p = unpickler(b"g100\n.")
p.memo = src.memo
expect(
    _pickle.UnpicklingError,
    "Memo value not found at index 100",
    p.load,
)

# `Unpickler_set_memo` dict branch: MemoPut against the current memo, then
# install a fresh empty dense array. Sparse keys stay in memo_dict; dense keys
# go with the discarded array.
p = unpickler(b"g100\n.")
p.memo = {100: 7}
assert p.load() == 7

p = unpickler(b"g0\n.")
p.memo = {0: 1}
expect(
    _pickle.UnpicklingError,
    "Memo value not found at index 0",
    p.load,
)

p = unpickler(b"g100\n.g0\n.")
p.memo = {100: 7, 0: 1}
assert p.memo.copy() == {}
assert p.load() == 7
expect(
    _pickle.UnpicklingError,
    "Memo value not found at index 0",
    p.load,
)

# Key / type errors from `Unpickler_set_memo`.
u = unpickler(b"I0\n.")
expect(TypeError, "memo key must be integers", lambda: setattr(u, "memo", {"a": 1}))
expect(
    ValueError,
    "memo key must be positive integers.",
    lambda: setattr(u, "memo", {-1: 1}),
)
expect(
    TypeError,
    "'memo' attribute must be an UnpicklerMemoProxy object or dict, not list",
    lambda: setattr(u, "memo", []),
)

print("OK")
