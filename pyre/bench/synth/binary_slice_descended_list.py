# pyre-check: max-pypy-ratio=9
# Exact-int BINARY_SLICE on a list: the walker descends
# `binary_slice_values_inner` (no residual CallMayForceR).
# N is large enough that pypy's execution-only time clears
# FLOOR_GATE_MIN_BASELINE_S so the ratio gate is actually evaluated.
# Darwin dynasm 8.3-9.3x; ceiling is twice the slower, one decimal.
N = 16000000


def slice_loop(n):
    xs = list(range(32))
    total = 0
    i = 0
    while i < n:
        a = i & 3
        b = a + 7
        total = total + len(xs[a:b])
        i = i + 1
    return total


assert slice_loop(N) == 7 * N
print("OK")
