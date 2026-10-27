# pyre-check: max-pypy-ratio=175
# PR271 review parity guard.  check.py's correctness oracle is PyPy, so this
# bench only asserts behaviour where 3.14 and PyPy AGREE; it guards the
# tuple-subclass override (P1), the live `__length_hint__` (P2) and the
# type-slot `__getitem__` iteration check (3b).  The 3.14-specific seq-iter
# `next` behaviour (StopIteration treated as exhaustion, retryable after a
# non-IndexError) diverges from PyPy and is documented at baseobjspace.rs
# `next` / `iter_iternext`, so it cannot be asserted against the PyPy oracle
# here.
import operator


class LenSeq:
    def __len__(self) -> int:
        return 7

    def __getitem__(self, i) -> int:
        if i >= 7:
            raise IndexError
        return i


class NoLenSeq:
    def __getitem__(self, i) -> int:
        if i >= 4:
            raise IndexError
        return i


class TupOverride(tuple):
    def __getitem__(self, i) -> int:
        return 1000 + tuple.__getitem__(self, i)


class InstanceGetitem:
    pass


class BadIter:
    def __iter__(self) -> int:
        return 42


class BadListIter(list):
    def __iter__(self) -> int:
        return 99


class LhViaGetattr:
    # __length_hint__ synthesised through __getattr__ — a type-MRO special
    # lookup must NOT see it, so operator.length_hint falls to the default.
    def __getattr__(self, name) -> object:
        if name == "__length_hint__":
            return lambda: 5
        raise AttributeError(name)


class NextViaGetattr:
    def __getattr__(self, name) -> object:
        if name == "__next__":
            return lambda: 1
        raise AttributeError(name)


class IterReturnsGetattrNext:
    def __iter__(self) -> object:
        return NextViaGetattr()


class ListIterNone(list):
    __iter__ = None


class TupleIterNone(tuple):
    __iter__ = None


def drive() -> list:  # noqa: PLR0915 - parity oracle kept intentionally linear
    out = []

    # A generic `__getitem__` cursor iterates lazily to the IndexError (it
    # ignores `__len__`).  Driven hot so the FOR_ITER over the cursor compiles.
    total = 0
    n = 0
    while n < 40000:
        for x in LenSeq():
            total += x
        n += 1
    out.append(("lazy_iter", total))

    # (P2) `__length_hint__` recomputed from the live sequence (len - index),
    # reachable through operator.length_hint as well as the direct call.
    it = iter(LenSeq())
    lh0 = operator.length_hint(it)
    next(it)
    lh1 = operator.length_hint(it)
    out.append(("len_hint", lh0, lh1, it.__length_hint__()))
    out.append(("len_hint_nolen", operator.length_hint(iter(NoLenSeq()))))

    # (P1) tuple-subclass `__getitem__` override honoured in a hot trace (the
    # JIT must not read the raw `wrappeditems` block).
    t = TupOverride([10, 20, 30])
    acc = 0
    n = 0
    while n < 40000:
        acc = t[1]
        n += 1
    out.append(("tuple_override", acc))

    # (3b) an instance-dict `__getitem__` does not enable iteration (special
    # methods resolve on the type, not the instance).
    c = InstanceGetitem()
    c.__getitem__ = lambda i: i
    try:
        list(c)
        out.append(("instance_getitem", "iterable"))
    except TypeError:
        out.append(("instance_getitem", "not_iterable"))

    # (3a) iter() validates that a dispatched `__iter__` returns an iterator;
    # a non-iterator result raises TypeError (the message text differs between
    # 3.14 and PyPy, so only the raise itself is asserted).  Both the generic
    # instance and the list-subclass override path are checked.
    try:
        iter(BadIter())
        out.append(("bad_iter", "no_error"))
    except TypeError:
        out.append(("bad_iter", "typeerror"))
    try:
        iter(BadListIter())
        out.append(("bad_list_iter", "no_error"))
    except TypeError:
        out.append(("bad_list_iter", "typeerror"))

    # length_hint is a type-MRO special lookup: a __getattr__- or
    # instance-synthesised __length_hint__ is ignored (→ default), while a
    # builtin iterator's hint is still found.
    out.append(("lh_via_getattr", operator.length_hint(LhViaGetattr(), 0)))
    y = InstanceGetitem()
    y.__length_hint__ = lambda: 9
    out.append(("lh_via_instance", operator.length_hint(y, 0)))
    out.append(("lh_listiter", operator.length_hint(iter([1, 2, 3, 4]), 0)))

    # iter() validates __next__ with a type-MRO lookup too: a __next__ reachable
    # only via __getattr__ does not make the result an iterator (message text
    # differs across 3.14/PyPy, so only the raise is asserted).
    try:
        iter(IterReturnsGetattrNext())
        out.append(("getattr_next", "no_error"))
    except TypeError:
        out.append(("getattr_next", "typeerror"))

    # An explicit `__iter__ = None` on a list/tuple subclass marks it
    # non-iterable; the message (subclass name) agrees across 3.14 and PyPy.
    try:
        list(ListIterNone([1, 2, 3]))
        out.append(("list_iter_none", "iterable"))
    except TypeError as e:
        out.append(("list_iter_none", str(e)))
    try:
        list(TupleIterNone([1, 2, 3]))
        out.append(("tuple_iter_none", "iterable"))
    except TypeError as e:
        out.append(("tuple_iter_none", str(e)))

    return out


for row in drive():
    print(row)
