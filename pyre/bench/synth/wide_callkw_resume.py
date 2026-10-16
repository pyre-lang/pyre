# A CALL_KW past the per-arity helper ceiling (nargs > 13) reached after a hot
# loop.  The loop exit deopts into the blackhole, which walks forward to the
# CALL_KW and hands it back with `abort_permanent`.  The interpreter resumes AT
# that opcode and re-runs it, so every argument has to still be readable out of
# the frame -- the walker keeps a pushed value in its SSA register and syncs
# only `valuestackdepth`, so the marker is what writes the slots back.
#
# Wrong code shows up as `UnboundLocalError: cannot access local variable 'a'`
# raised inside the callee, because *a binds against empty argument slots.
#
# No `max-pypy-ratio`: pyre runs this ~80x pypy because the CALL_KW is refused
# every time and the enclosing frame falls back to the interpreter, and pypy's
# own execution time here sits on the harness measurement floor, so no ratio
# computed against it is a measurement.  The fixture is a wrong-code guard; the
# committed jit-stats and the output comparison are what gate it.
CALLS = []


def take(*a, **k):
    return (a, k)


def run(n):
    i = 0
    while i < n:
        i += 1
    CALLS.append(i)
    return take(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, kw=99)


total = 0
for _ in range(3):
    args, kwargs = run(4200000)
    total += sum(args) + kwargs["kw"]
print(total, len(CALLS))
