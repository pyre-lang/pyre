# pyre-check: max-pypy-ratio=8
# Native CI readings span 2.5x (macos dynasm, run 39064955611) to 6.9x (ubuntu
# cranelift); the ceiling is 15% over the widest, so its derived floor 2.0
# sits under the fastest.
# Deep operand-stack Variables kept across a short-circuit guard, pure and
# mutating.
#
# `g(i)` and `h(i)` are computed Variables held deep on the stack while the
# `p or q` short-circuit guard decides the third tuple element. A guard resume
# must restore the deep-stack g(i)/h(i).
#
# The mutating half holds the same Variables across the same guard, but its
# helpers each append to a shared global list (a non-journaled
# STORE_SUBSCR-class heap effect committed inside a user frame).  A FOR_ITER
# trace that consumes the iterator, aborts on the deep kept-stack guard, and
# then DELIVERS the in-flight item would re-run the body and DOUBLE the
# mutation.  The `log` length must equal 3x the iteration count exactly (g, h,
# and one conditional g/h append per iteration): a doubled delivery
# over-counts, a dropped iteration under-counts.
N = 1454546
log = []


def g(i):
    return i * 2 + 1


def h(i):
    return i * 3 - 1


def g_log(i):
    log.append(i)
    return i * 2 + 1


def h_log(i):
    log.append(-i)
    return i * 3 - 1


def f(n):
    s = 0
    for i in range(n):
        p = (i % 4) != 0
        q = (i % 5) != 0
        t = (g(i), h(i), (g(i) if (p or q) else h(i)))
        s += t[0] - t[1] + t[2]
    return s


def f_log(n):
    s = 0
    for i in range(n):
        p = (i % 4) != 0
        q = (i % 5) != 0
        t = (g_log(i), h_log(i), (g_log(i) if (p or q) else h_log(i)))
        s += t[0] - t[1] + t[2]
    return s


print(f(N))
print(f_log(N))
print(len(log))
