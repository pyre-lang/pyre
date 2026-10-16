# pyre-check: max-pypy-ratio=11
# CHECK_EXC_MATCH against a non-exception target. The loop warms up with a
# valid except class so the trace JIT-compiles, then on the final iteration the
# except target becomes a non-exception (int 5). cmp_exc_match raises TypeError
# when the target is not an exception class / tuple of exception classes; the
# JIT residual path must raise it too (not silently produce a bool and let the
# original exception propagate). The TypeError is caught by an outer handler so
# the output is deterministic. Deterministic.
def f(n):
    caught_value = 0
    caught_type = 0
    for i in range(n):
        target = ValueError if i < n - 1 else 5
        try:
            try:
                raise ValueError("x")
            except target:
                caught_value += 1
        except TypeError:
            caught_type += 1
    return caught_value, caught_type


def main():
    cv, ct = f(3000000)
    print("final", cv, ct)


main()
