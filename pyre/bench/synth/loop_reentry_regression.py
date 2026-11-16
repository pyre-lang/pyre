# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=loop
# Regression guard: a compiled hot loop must be re-entered on every call of its
# enclosing function, not only on the call that compiled it.
#
# Discovered by the synthetic suite as a native-backend self-check. Run it directly:
#
#     python pyre/bench/synth/loop_reentry_regression.py
#     target/release/pyre-cranelift pyre/bench/synth/loop_reentry_regression.py
#
# The regression it guards: is_compatible refused re-entry once the enclosing
# module bound a new top-level name and its globals dict grew, so call 1 ran the
# loop compiled (~sub-ns/iter) while calls 2+ ran it interpreted (~1000 ns/iter)
# with a per-iteration compiled-entry abort. The ~1000x margin makes the 20x gate
# below not timing-flaky.

import time

N = 3_000_000
SLOWDOWN_LIMIT = 20.0  # calls 2+ may be at most 20x slower than call 1


def loop(n):
    i = 0
    total = 0
    while i < n:
        total = total + i
        i = i + 1
    return total


def per_call_ns():
    t0 = time.perf_counter()
    loop(N)
    return (time.perf_counter() - t0) / N * 1e9


first = per_call_ns()          # compiles mid-call, then runs compiled
rest = [per_call_ns() for _ in range(3)]
worst = max(rest)
ratio = worst / first if first > 0 else float("inf")

print(f"call1={first:.2f}ns rest={[round(x, 2) for x in rest]} worst_ratio={ratio:.1f}x")
if ratio > SLOWDOWN_LIMIT:
    print(f"FAIL re-entry broken: later calls {ratio:.0f}x slower than the compiling call")
    raise SystemExit(1)
print("PASS re-entry preserved")
