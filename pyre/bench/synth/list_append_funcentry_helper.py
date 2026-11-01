# pyre-check: max-pypy-ratio=5
# Function-entry `push` is a no-loop helper; list growth is a residual
# cond_call rather than a deopt. Three loops compile, as on pypy: the `push`
# entry trace, `main`'s loop with `push` inlined, and the app-level
# `_int_sum` loop for the closing `sum` calls (`app_functional.sum`).
# The ceiling of 5 keeps the macOS reading (1.8x-2.4x locally at N=4500000)
# above its 1.25x floor while leaving room for the Linux runners, where pypy is
# relatively faster (CI read the ubuntu ratio at about 1.6x the macOS one).
# #171/#34: the orthodox list.append fold fires in function-entry (no-loop)
# helper traces, not only loop traces.  `push` is a no-loop helper called in a
# hot loop on two alternating receivers, so it traces from entry (header_pc==0)
# and its spare-capacity resume must reconstruct the alternating receiver
# correctly — a wrong receiver box would route an
# append into the other list and corrupt the cross-checked sums below.
# N is sized from CI timings. At N=1500000 pypy's execution-only time read
# 0.027-0.036s on the ubuntu and macOS check.py jobs, under the 0.05s floor
# below which the pypy ratio is not gated, and the reading moved between 5x
# and 15x from run to run. Three times that puts pypy near 0.1s while the
# macOS pyre reading (0.24-0.29s) stays under one second.
N = 4500000


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
