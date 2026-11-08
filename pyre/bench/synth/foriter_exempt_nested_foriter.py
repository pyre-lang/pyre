# pyre-check: max-pypy-ratio=3
# The measured reading stays between 0.9 and 2.3, so the ceiling is 3.
# The function-entry door reading its own cell took this off the 44 it needed
# while the door read another cell's answer, asked to trace at every call and
# never entered the compiled loop.
#
# gh#495 guard: fbw_abort_nested_unjournaled_residual prevents the ForIterNext exemption double-advance.
# branch-bearing callee with a SECOND FOR_ITER (nested), not the loop header.
# Two shared generators; inner FOR_ITER advance is a non-header foriter (Finding #2).
# Post-inner declining residual forces abort while inner item in-flight.
# N is large enough that pypy's user time clears startup noise (~0.24s).
# A single-yield generator is inlined at FOR_ITER: `dispatch`'s `except Yield`
# `popvalue` returns the suspended value.
# `MAJIT_STATS` on this loop records `caro_no_merge_entry=0` and one compiled
# trace (`loops_compiled=1`), so gouter/ginner are not left interpreted.
# That compiled loop reads 2.2-4.3x, under this ceiling.
N = 12000000


class Shared:
    def __init__(self):
        self.a = 0
        self.b = 0
        self.t = 0


def gouter(sh, m):
    j = 0
    while j < m:
        sh.a += 1
        yield j
        j += 1


def ginner(sh, m):
    j = 0
    while j < m:
        sh.b += 1
        yield j * 10
        j += 1


def tail(sh):
    sh.t += 1
    return sh.t


def step(go, gi, sh, k):
    if k < 0:
        return 0
    s = 0
    for x in go:
        s += x
        for y in gi:
            s += y
            break
        t = tail(sh)
        s += t & 0
        break
    return s


def run(N):
    sh = Shared()
    go = gouter(sh, N * 10)
    gi = ginner(sh, N * 10)
    acc = 0
    i = 0
    while i < N:
        k = i % 5
        acc += step(go, gi, sh, k)
        i += 1
    return acc, sh.a, sh.b, sh.t


print(run(N))
