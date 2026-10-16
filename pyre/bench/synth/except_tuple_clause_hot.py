# pyre-check: max-pypy-ratio=19
# An `except (A, B):` clause builds its match target with `BUILD_TUPLE` on
# every visit, so the tuple is a fresh object each time the handler runs. The
# `CHECK_EXC_MATCH` fold pins the match target it saw while tracing; pinning
# the container rather than its elements makes that guard unsatisfiable on a
# re-allocated tuple, and the handler then side-exits once per visit. The
# elements are `LOAD_GLOBAL` constants already covered by the quasi-immutable
# invalidation guard, so the clause must stay on the compiled path.
#
# The single-class `except A:` loop below is the control: same raise rate, a
# match target that is loaded rather than built. Both loops must agree with the
# interpreter and neither may accumulate side exits.
N = 600000


def tuple_clause(n):
    acc = 0
    i = 0
    while i < n:
        try:
            if i > 2:
                raise TypeError("x")
        except (TypeError, ValueError):
            acc += 9
        i += 1
    return acc


def single_clause(n):
    acc = 0
    i = 0
    while i < n:
        try:
            if i > 2:
                raise TypeError("x")
        except TypeError:
            acc += 9
        i += 1
    return acc


print(tuple_clause(N), single_clause(N))
