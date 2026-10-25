# pyre-check: max-pypy-ratio=10
# Function-entry `push` is a no-loop helper; list growth is a residual
# cond_call rather than a deopt. Three loops compile, as on pypy: the `push`
# entry trace, `main`'s loop with `push` inlined, and the app-level
# `_regular_sum` loop for the closing `sum` calls (`app_functional.sum`).
# The ceiling keeps the macOS reading (3.1x locally) above its 2.5x floor
# while leaving room for the Linux runners, where pypy is relatively faster.
# #171/#34: the orthodox list.append fold fires in function-entry (no-loop)
# helper traces, not only loop traces.  `push` is a no-loop helper called in a
# hot loop on two alternating receivers, so it traces from entry (header_pc==0)
# and its spare-capacity resume must reconstruct the alternating receiver
# correctly — a wrong receiver box would route an
# append into the other list and corrupt the cross-checked sums below.
N = 1500000


def push(a, v):
    a.append(v)


def main():
    xs = []
    ys = []
    i = 0
    while i < N:
        push(xs, i)
        push(ys, -i)
        i = i + 1
    ok = (
        len(xs) == N
        and len(ys) == N
        and xs[0] == 0
        and ys[1] == -1
        and xs[N - 1] == N - 1
        and ys[N // 2] == -(N // 2)
    )
    print(sum(xs) + sum(ys), len(xs) + len(ys), ok)


main()
