# pyre-check: max-pypy-ratio=7.4
# A CALL_KW directly in the hot loop body (not nested inside an inlined
# callee, unlike call_star_forms_inlined_callee.py). The keyword overrides a
# default (`step=1`) and the result depends on both the positional and the
# keyword arg so the call cannot be constant-folded away; the exact aggregate
# makes a single dropped iteration observable.
#
# The keyword binding is the interpreter's own matcher
# (`Arguments.match_signature` / `parse_into_scope`): the callee
# signature is constant, so the `kwnames` permutation unrolls into the
# trace and the seeded frame is the callee body. The caller image the
# seeded frame's guards resume on is one slot deeper than a plain CALL's
# because the names tuple sits above the arguments (caller_operand_slots).
# call_kw_residual_kwargs_callee.py is the other half: the same opcode
# with a `**kw` callee, whose surplus name is stored into that dict.
N = 33600000


def g(x, step=1):
    return x + step


def main():
    total = 0
    for i in range(N):
        total += g(i, step=2)
    print(total)


main()
