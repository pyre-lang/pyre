# CPython-suite gap: test_bytes.test_repeat accepts either OverflowError or
# MemoryError for `b * sys.maxsize` and `b *= sys.maxsize` ("Shouldn't bytes
# and bytearray agree on what to raise?"), so it cannot tell the two apart.
# parity-tests reason: bytes repeat checks the product before the elidable
# ll_str_mul and raises OverflowError; bytearray repeat follows rlist.py
# ll_mul / ll_inplace_mul and raises MemoryError, in place or not.
# pyre-check: pypy-diverges: rstr.py ll_str_mul turns the ovfcheck into
# MemoryError, so pypy3 raises MemoryError for `bytes * sys.maxsize` as well.

"""bytes and bytearray repetition raise their own error for an oversized product."""

import sys


def raised(fn):
    try:
        fn()
    except (OverflowError, MemoryError) as e:
        return type(e).__name__, str(e)
    return None


def imul(v, n):
    v *= n


# descr_mul's branches: empty, the operand itself, one byte, several bytes.
b = b"ab"
assert b * 0 == b"" and b * -3 == b"", (b * 0, b * -3)
assert b * 1 is b
assert b"x" * 5 == b"xxxxx", b"x" * 5
assert b * 3 == b"ababab", b * 3

result = raised(lambda: b"ab" * sys.maxsize)
assert result == ("OverflowError", "repeated bytes are too long"), result
result = raised(lambda: imul(b"ab", sys.maxsize))
assert result == ("OverflowError", "repeated bytes are too long"), result
# Fits a Py_ssize_t but cannot be allocated.
result = raised(lambda: b"ab" * (sys.maxsize // 2))
assert result is not None and result[0] == "MemoryError", result

for n in (sys.maxsize, sys.maxsize // 2):
    result = raised(lambda: bytearray(b"ab") * n)
    assert result is not None and result[0] == "MemoryError", (n, result)
    ba = bytearray(b"ab")
    result = raised(lambda: imul(ba, n))
    assert result is not None and result[0] == "MemoryError", (n, result)
    assert ba == bytearray(b"ab"), ba

print("OK")
