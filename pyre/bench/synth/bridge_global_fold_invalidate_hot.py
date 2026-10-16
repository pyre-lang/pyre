# pyre-check: max-pypy-ratio=36
# A read-only module global read on a COLD arm (a branch, a genexpr, or a
# function that gets rebound) compiles as a bridge that const-folds the global
# via a quasi-immutable fold. A mid-loop same-key reassignment (or a function
# rebind) must invalidate that fold so the bridge re-reads the new value; a
# stale fold keeps returning the old value. Exercises an int global, a len() of
# a global set, a str-prefix global, and a rebound global function. Output is
# verified against CPython/PyPy.
N = 400000

CFG = 1
S = {1, 2, 3}
AFFIX = "pre_"


def rd_int(i):
    return CFG if i & 1 else 0


def rd_len(i):
    return len(S) if i & 1 else 0


def chk(s):
    return s.startswith(AFFIX)


def route(n):
    return n * 2


def route2(n):
    return n + 100


def run():
    global CFG, S, AFFIX, route
    total = 0
    for i in range(N):
        total += rd_int(i) + rd_len(i)
        if chk(AFFIX + str(i)):
            total += 1
        total += route(i % 3)
        if i == N // 2:
            CFG = 2
            S = {1, 2, 3, 4, 5}
            AFFIX = "post_"
            route = route2
    return total


print(run())
