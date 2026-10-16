# pyre-check: max-pypy-ratio=14
# Parity fixture for the depth-1 bridge self-recursive inline lift
# (#704). `f` is a tail-recursive
# exact-integer callee whose `acc * 2 + 1` crosses the machine-int boundary
# partway down the recursion, so an overflow guard fires inside the frame the
# lift inlines on a guard-failure bridge. #704's A/B ran on `fib`, which never
# overflows, so this shape — an overflow-guard resume in a bridge-inlined
# recursive frame — was unexercised. Byte-parity against CPython/PyPy here
# guards its resume against a silent wrong-answer regression. No max-pypy-ratio
# gate: with the lift on this shape admits an abort that folds back to residual
# (perf-neutral-to-negative), so the signal is correctness, not speed.
def f(n, acc):
    if n == 0:
        return acc
    return f(n - 1, acc * 2 + 1)


out = []
for i in range(18000):
    out.append(f(64, i) % 1000000007)
print(out[0], out[-1], sum(out) % 1000000007)
