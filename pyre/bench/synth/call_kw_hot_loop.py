# pyre-check: max-pypy-ratio=16
# A CALL_KW directly in the hot loop body (not nested inside an inlined
# callee, unlike call_star_forms_inlined_callee.py), on a callee the seeded
# inline takes.  The keyword overrides a default (`step=1`) and the result
# depends on both the positional and the keyword arg so the call cannot be
# constant-folded away; the exact aggregate makes a single dropped iteration
# observable.
#
# What a dropped term would come from is the pair the inlined route needs at a
# CALL_KW: the permutation that seeds the callee's parameters from `kwnames`
# (fbw_reorder_call_kw_args), and the caller image the seeded frame's guards
# resume on, whose operand region is one slot deeper here than a plain CALL's
# because the names tuple sits above the arguments (caller_operand_slots).
# call_kw_residual_kwargs_callee.py is the other half, covering the same
# opcode on a callee the inline declines.
N = 2400000


def g(x, step=1):
    return x + step


def main():
    total = 0
    for i in range(N):
        total += g(i, step=2)
    print(total)


main()
