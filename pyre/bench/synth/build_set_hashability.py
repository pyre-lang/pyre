# pyre-check: max-pypy-ratio=105
# BUILD_SET (the {...} set literal) hashes every element through
# space.hash_w, so an unhashable element — a list, or an instance whose
# __hash__ is None / raises / returns a non-int — raises instead of silently
# building a set, and a user __hash__ is actually invoked.  Only the
# exception type is printed so the line matches across CPython/PyPy/Pyre.
# Trip count kept clear of the major-collection threshold check.py pins.
# 680000 crossed it: the eval-breaker bailout re-entered through a bridge
# whose guard then failed once more, moving guard_failures for reasons
# outside this fixture. 85000 still arms the breaker on wasm
# (`back_edge_polls=1`) while native stays at 0, and ubuntu wasm then
# reads guard_failures 42 or 43. On this host the breaker first arms at
# 71000; 70000 is the last zero-poll value. 60000 keeps a ~10k margin,
# `back_edge_polls=0` on all three backends, and the same guard_failures;
# loops_compiled stays 3.
N = 60000


class HashRaises:
    def __hash__(self):
        raise ValueError("nope")


class NoHash:
    __hash__ = None


class BadHash:
    def __hash__(self):
        return "not-an-int"


HASH_CALLS = []


class Counted:
    def __init__(self, v):
        self.v = v

    def __hash__(self):
        HASH_CALLS.append(self.v)
        return self.v


def show(label, fn):
    try:
        fn()
        print(label, "NO-RAISE")
    except Exception as e:
        print(label, type(e).__name__)


def main():
    # Hot loop: the JIT-compiled BUILD_SET residual builds 3-element sets
    # whose elements carry a user __hash__; the fallible residual path must
    # stay benign (no spurious raise) when every element is hashable.
    acc = 0
    i = 0
    while i < N:
        s = {Counted(i), Counted(i + 1), Counted(i + 2)}
        acc = acc + len(s)
        i = i + 1
    print("acc", acc)
    print("hash_called", len(HASH_CALLS) > 0)

    # An unhashable list element raises TypeError.
    show("list_elem", lambda: {[1, 2]})

    # __hash__ = None raises TypeError (an infallible identity hash would
    # silently accept this).
    show("nohash_elem", lambda: {NoHash()})

    # A raising __hash__ propagates its own exception (not swallowed).
    show("raising_hash", lambda: {HashRaises()})

    # A __hash__ returning a non-int raises TypeError.
    show("badhash_elem", lambda: {BadHash()})

    # A nested unhashable (tuple containing a list) raises TypeError.
    show("nested_unhashable", lambda: {([1],)})

    # Elements hash left-to-right: the leading raising __hash__ wins over the
    # trailing unhashable list (ValueError, not TypeError).
    show("left_to_right", lambda: {HashRaises(), [1]})

    # Successful-build dedup is unchanged for hashable builtins.
    print("dedup", len({1, 1, 2, "a", "a", (1, 2), (1, 2)}))


main()
