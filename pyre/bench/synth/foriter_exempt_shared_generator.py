# pyre-check: max-pypy-ratio=20
# The function-entry door reading its own cell took this off the 63 it needed
# while the door read another cell's answer, asked to trace at every call and
# never entered the compiled loop.  Its pypy baseline straddles
# EXEC_TIME_FLOOR_S the same way its nested-foriter sibling's does, so the
# ceiling is sized the same way -- see the header there.
# gh#495 guard: fbw_abort_nested_unjournaled_residual prevents the ForIterNext exemption double-advance.
# SHARED long generator consumed incrementally. step consumes ONE item (for..break),
# FOR_ITER advance mutates shared counter (exempt). Then a declining nested-residual CALL.
# If the inline sub-walk aborts AFTER the exempt advance and the trait leg re-runs step,
# the SHARED generator is advanced AGAIN -> double advance / dropped item / counter skew.
N = 1200000


class Shared:
    def __init__(self):
        self.pos = 0
        self.tail = 0


def biggen(sh, m):
    j = 0
    while j < m:
        sh.pos += 1
        yield j
        j += 1


def tailhelper(sh):
    sh.tail += 1
    return sh.tail


def step(g, sh, k):
    if k < 0:
        return 0
    got = -1
    for x in g:
        got = x
        break
    t = tailhelper(sh)
    return got + (t & 0)


def run(N):
    sh = Shared()
    g = biggen(sh, N * 10)
    acc = 0
    i = 0
    while i < N:
        k = i % 5
        v = step(g, sh, k)
        acc += v
        i += 1
    return acc, sh.pos, sh.tail


print(run(N))
