# pyre-check: max-pypy-ratio=28
# The ceiling is twice the slowest ratio observed, 13.9x on the macos runner.
# Deep operand-stack Variables across a nested-call deopt guard.
# Several computed Variables (x, y, z) sit deep on the stack as arguments being
# marshalled for `acc(...)` while `mul(i, i)` (a residual call that can guard /
# deopt) evaluates. A guard resume there must reconstruct the deep-stack
# argument Variables.
def mul(a, b):
    return a * b


def acc(a, b, c, d):
    return a + b - c + d


def f(n):
    s = 0
    for i in range(n):
        x = i + 1
        y = i + 2
        z = i + 3
        s += acc(x, y, z, mul(i, i) % 1000)
    return s


print(f(1440000))
