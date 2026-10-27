# pyre-check: max-pypy-ratio=11
# A module-scope hot loop calling a helper that returns an inlined list
# comprehension over `range(n)` with `n` a parameter.  Once the loop reaches the
# trace threshold it records the `len(f(<const>))` body; recording the CALL to
# `f` aborts and the walk forward-flushes the caller (module) frame at the CALL
# boundary so the interpreter re-runs the call from there.  That flush rebuilds
# the caller's operand stack from the walk's live/shadow sources — but the
# CALL's `LOAD_CONST`'d argument has no concrete Ref shadow, so its slot
# resolves to NULL.  The flush must decline (fall back to the legacy replay)
# rather than commit the NULL; committing it left the next call's argument slot
# unbound, so `f` raised `UnboundLocalError` on its parameter (`n`).
#
# The trigger is specific: the caller loop must be at MODULE scope (its CALL
# operands come from LOAD_NAME / LOAD_CONST, not LOAD_FAST), the inner
# `range(n)` must be large enough to compile + bridge the comprehension loop,
# and the outer loop must run enough to reach the trace threshold so the two
# transitions coincide.


def f(n):
    return [i for i in range(n)]


t = 0
k = 0
while k < 79200:
    t += len(f(300))
    k += 1
print(t)
