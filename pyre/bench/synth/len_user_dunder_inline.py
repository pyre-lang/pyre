# pyre-check: trace-shape=hot_plain:absent=CallMayForceR
# `len(obj)` whose receiver resolves `__len__` to a Python function.  Covers
# the two receiver shapes `descroperation.py _len` dispatches a method for -- a
# user instance and a builtin subclass overriding `__len__` -- plus the ways
# the resolution or the result can change under a hot loop: a second receiver
# class flowing into the same site, a `__len__` reassigned on the class
# mid-loop, and an instance-dict `__len__` (which a special-method lookup must
# ignore).  The three results `len()` refuses -- negative, non-int, and a value
# too large for a machine word -- are exercised hot, so the checks around the
# call are read through the compiled trace rather than only at startup, and so
# is the one it converts: `len()` boxes the machine length it checked, so a
# `__len__` answering `True` gives the int 1.
#
# `hot_deopt_*` leave the operator by the OTHER exit: a map change fails a
# guard inside the inlined `__len__` itself, so the body finishes in the
# blackhole.  `space.index`, `_check_len_result` and that box have to run on
# that exit too, which is what `operator_continuation`'s resume level is for.
#
# `nested` and `nested_refused` are the third exit: a `__len__` whose own body
# runs `len()` leaves an executed effect behind it, which a caller-level guard
# could not resume past, so the route declines the call and keeps the
# interpreter's residual rather than aborting the enclosing loop.
# Deterministic.
#
# No `max-pypy-ratio`: pypy folds every loop here away, so its execution-only
# time sits at the floor whatever this fixture is sized to -- measured, the
# reported ratio read 18x at one size and 47x at four times that, which is the
# clamp talking and not the code.  The fold coverage above and the jitstats
# baselines gate it.


class Sized:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


class Doubled:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n * 2


class TupOverride(tuple):
    def __len__(self):
        return 1000 + tuple.__len__(self)


class Rebind:
    def __len__(self):
        return 4


class Neg:
    def __len__(self):
        return -1


class Boolish:
    # `len()` boxes the machine length it checked, so a `__len__` answering
    # `True` gives the int 1 -- on the traced path as well as the deopt one.
    def __len__(self):
        return True


class Float:
    def __len__(self):
        return 1.5


class Big:
    def __len__(self):
        return 2 ** 100


# A `__len__` whose body calls `len()` itself: inlining it runs a residual,
# which moves the executed-effect odometer the fold reads.
class Nested:
    def __init__(self, items):
        self._items = items

    def __len__(self):
        return len(self._items)


class NestedNeg:
    def __init__(self, items):
        self._items = items

    def __len__(self):
        return -len(self._items)


def hot_plain(o, n):
    acc = 0
    for _ in range(n):
        acc += len(o)
    return acc


def hot_override(t, n):
    acc = 0
    i = 0
    while i < n:
        acc = len(t)
        i += 1
    return acc


def hot_polymorphic(a, b, n):
    acc = 0
    for i in range(n):
        o = a if i % 2 else b
        acc += len(o)
    return acc


def hot_rebind(o, n):
    acc = 0
    for i in range(n):
        if i == n // 2:
            Rebind.__len__ = lambda self: 9
        acc += len(o)
    return acc


def hot_deopt_refused(o, n):
    # The map change fails a guard in the INLINED body rather than a check
    # around it, so the callee finishes in the blackhole and returns through
    # the operator's own tail.  Without that tail `len` hands back `__len__`'s
    # raw box and answers -1 instead of raising.
    total = 0
    caught = 0
    for i in range(n):
        if i == n // 2:
            o.extra = 1
            o.n = -1
        try:
            total += len(o)
        except ValueError:
            caught += 1
    return total, caught


def hot_deopt_converted(o, n):
    # The same exit with a result the operator CONVERTS rather than refuses:
    # `len()` over a `__len__` that answered `True` is the int 1.
    kinds = set()
    for i in range(n):
        if i == n // 2:
            o.extra = 1
            o.n = True
        kinds.add(type(len(o)).__name__)
    return sorted(kinds)


def hot_refused(o, n):
    caught = 0
    for _ in range(n):
        try:
            len(o)
        except (ValueError, TypeError, OverflowError):
            caught += 1
    return caught


def main():
    print("plain", hot_plain(Sized(5), 20000))
    print("override", hot_override(TupOverride([10, 20, 30]), 20000))
    print("polymorphic", hot_polymorphic(Sized(5), Doubled(3), 20000))
    print("rebind", hot_rebind(Rebind(), 20000))
    print("negative", hot_refused(Neg(), 20000))
    print("non_int", hot_refused(Float(), 20000))
    print("too_big", hot_refused(Big(), 20000))
    print("boolish", hot_plain(Boolish(), 20000), type(len(Boolish())).__name__)
    print("nested", hot_plain(Nested([1, 2, 3]), 20000))
    print("nested_refused", hot_refused(NestedNeg([1, 2, 3]), 20000))
    print("deopt_refused", hot_deopt_refused(Sized(5), 20000))
    print("deopt_converted", hot_deopt_converted(Sized(5), 20000))

    # A special method resolves on the type, so an instance-dict entry is not
    # consulted and the inherited length still answers.
    bare = Sized(2)
    bare.__len__ = lambda: 999
    print("instance_dict", len(bare))

    # A length that grows across the loop: the result is a red value, not a
    # constant the trace may bake.
    grow = Sized(0)
    total = 0
    for _ in range(363637):
        grow.n += 1
        total += len(grow)
    print("growing", total)


main()
