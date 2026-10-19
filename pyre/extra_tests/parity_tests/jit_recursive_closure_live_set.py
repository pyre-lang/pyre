# pyre-check: pypy-diverges: pins _pylong (3.12), absent from the CPython 3.11 pypy3 implements
# CPython-suite gap: closure tests cannot cover pyre JIT residual-call liveness.
# parity-tests reason: guard a closure-owned set across an inlined recursive call,
# and an inlined callee's freevar cell placed past its own fresh cells.

"""A recursive closure must keep its freshly captured set live across a call."""

import _pylong


class Checker:
    def check(self, actual, expected, w):
        assert actual == expected, (w, actual, expected)


def run(checker):
    # `expected_powers` has fresh cells (`seen`, `need`, `visit`) and one
    # freevar (`checker`).  Inlined into this loop, the freevar's cell sits
    # past the fresh cells; seeding it at the first fresh cell's slot made
    # `STORE_DEREF visit` overwrite `checker` with the function.
    def expected_powers(w, base, limit, need_hi):
        seen = set()
        need = set()

        def visit(w):
            if w <= limit or w in seen:
                return
            seen.add(w)
            lo = w >> 1
            hi = w - lo
            need.add(hi if need_hi else lo)
            visit(lo)
            visit(hi)

        visit(w)
        # Keeping only the closure-owned reference live across this residual
        # call used to make the compiled outer frame reload an incomplete set.
        actual = _pylong.compute_powers(w, base, limit, need_hi=need_hi)
        checker.check(actual.keys(), need, w)

    for base in (2, 5):
        for need_hi in (False, True):
            for limit in (1, 11):
                for width in range(250, 550):
                    expected_powers(width, base, limit, need_hi)


run(Checker())
print("OK")
