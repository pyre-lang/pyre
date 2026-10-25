"""
Plain Python definition of the builtin functions oriented towards
functional programming.
"""


def sorted(iterable, /, *, key=None, reverse=False):
    "sorted(iterable, key=None, reverse=False) --> new sorted list"
    sorted_lst = list(iterable)
    sorted_lst.sort(key=key, reverse=reverse)
    return sorted_lst


# PY_SSIZE_T_MIN / PY_SSIZE_T_MAX.  PyLong_AsLongAndOverflow leaves the
# exact-int phase when the running total no longer fits, so a later float
# cannot enter the compensated path: sum([2**63, 0.1, 1, -(2**63)]) is 0.0.
_SSIZE_MAX = 9223372036854775807
_SSIZE_MIN = -9223372036854775808
_POS_INF = float("inf")
_NEG_INF = float("-inf")


def sum(iterable, /, start=0):
    """sum(sequence[, start]) -> value

Returns the sum of a sequence of numbers (NOT strings) plus the value
of parameter 'start' (which defaults to 0).  When the sequence is
empty, returns start."""
    if isinstance(start, str):
        raise TypeError("sum() can't sum strings [use ''.join(seq) instead]")
    if isinstance(start, bytes):
        raise TypeError("sum() can't sum bytes [use b''.join(seq) instead]")
    if isinstance(start, bytearray):
        raise TypeError("sum() can't sum bytearray [use b''.join(seq) instead]")

    # Avoiding isinstance here, since subclasses can override `+`
    if type(start) is list:
        return _list_sum(iterable, start)

    if type(start) is tuple:
        return _tuple_sum(iterable, start)

    return _regular_sum(iterable, start)


def _cs_add(hi, lo, x):
    # cs_add: improved Kahan-Babuška / Neumaier compensation.
    t = hi + x
    if abs(hi) >= abs(x):
        lo += (hi - t) + x
    else:
        lo += (x - t) + hi
    return t, lo


def _cs_to_double(hi, lo):
    # cs_to_double: skip a zero or non-finite residual so a negative hi
    # keeps its sign and inf - inf cannot turn an infinite total into NaN.
    if lo != 0.0 and lo == lo and lo != _POS_INF and lo != _NEG_INF:
        return hi + lo
    return hi


def _regular_sum(sequence, start):
    # Default implementation for sum (no specialization)
    last = start
    it = iter(sequence)

    # builtin_sum_impl exact-int phase: PyLong_CheckExact total that still
    # fits a pointer-sized signed word.  A bool start skips it
    # (PyLong_CheckExact is false for bool).
    if type(last) is int and _SSIZE_MIN <= last <= _SSIZE_MAX:
        for x in it:
            tx = type(x)
            if tx is int or tx is bool:
                nxt = last + x
                if _SSIZE_MIN <= nxt <= _SSIZE_MAX:
                    last = nxt
                    continue
                last = nxt
            else:
                last = last + x
            break
        else:
            return last

    # builtin_sum_impl float phase: exact float total, Neumaier
    # compensation.  Exact floats and any int (PyLong_Check, including
    # bool and int subclasses) stay on the path; a float subclass leaves.
    # int.__float__ is descr___float__ (payload only, no subclass __float__).
    if type(last) is float:
        hi = last
        lo = 0.0
        for x in it:
            if type(x) is float:
                value = x
            elif isinstance(x, int):
                value = int.__float__(x)
            else:
                last = _cs_to_double(hi, lo) + x
                break
            hi, lo = _cs_add(hi, lo, value)
        else:
            return _cs_to_double(hi, lo)

    # builtin_sum_impl complex phase: independent compensated real and
    # imag.  Exact complex, any int, and any float (PyFloat_Check,
    # including subclasses) stay on the path.
    if type(last) is complex:
        re_hi = last.real
        re_lo = 0.0
        im_hi = last.imag
        im_lo = 0.0
        for x in it:
            if type(x) is complex:
                re_hi, re_lo = _cs_add(re_hi, re_lo, x.real)
                im_hi, im_lo = _cs_add(im_hi, im_lo, x.imag)
            elif isinstance(x, int):
                re_hi, re_lo = _cs_add(re_hi, re_lo, int.__float__(x))
            elif isinstance(x, float):
                re_hi, re_lo = _cs_add(re_hi, re_lo, x)
            else:
                last = complex(
                    _cs_to_double(re_hi, re_lo),
                    _cs_to_double(im_hi, im_lo),
                ) + x
                break
        else:
            return complex(
                _cs_to_double(re_hi, re_lo),
                _cs_to_double(im_hi, im_lo),
            )

    for x in it:
        # Very intentionally *not* +=, that would have different semantics if
        # start was a mutable type, such as a list
        last = last + x
    return last


def _list_sum(sequence, start):
    # Specialization avoiding quadratic complexity for lists
    iterator = iter(sequence)

    try:
        first = next(iterator)
    except StopIteration:
        return start

    if type(first) is not list:
        return _regular_sum(iterator, start + first)

    last = start + first
    for item in iterator:
        if type(item) is list:
            last += item
        else:
            # Non-trivial sum. Use _regular_sum.
            return _regular_sum(iterator, last + item)

    return last


def _tuple_sum(sequence, start):
    # Specialization avoiding quadratic complexity for tuples
    iterator = iter(sequence)

    try:
        first = next(iterator)
    except StopIteration:
        return start

    if type(first) is not tuple:
        return _regular_sum(iterator, start + first)

    last = list(start)
    last.extend(first)
    for item in iterator:
        if type(item) is tuple:
            last.extend(item)
        else:
            # Non-trivial sum. Cast back to tuple and use regular_sum.
            return _regular_sum(iterator, tuple(last) + item)

    return tuple(last)
