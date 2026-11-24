# pyre-check: max-pypy-ratio=3
# Readings 1.1x-1.9x. A ceiling of 3 keeps the reading in 0.75x..3x.
N = 93041900


# UNARY_NEGATIVE in a hot loop lowers to the `unary_negative(value)` HLOp →
# the canonical `inline_call_r_r` to the interpreter's `descroperation::neg`
# body (mirroring UNARY_INVERT).
# Before the HLOp lowering the flow op `neg` reached the assembler with no
# builder mapping and any `-x` in a JIT-compiled loop panicked.
def main():
    acc = 0
    i = 0
    while i < N:
        x = -i
        acc = acc + x
        i = i + 1
    print(acc)


# UNARY_NEGATIVE on INT_MIN: -INT_MIN overflows the machine-int range, so
# descr_neg (intobject.py) takes the long branch and returns 2**63 as a
# W_LongObject.  The codewriter inline-call walks that overflow arm of `neg`,
# so the compiled loop must agree with the long result rather than wrapping
# back to INT_MIN.  The promoted long crosses the loop header as an argument
# and the comparison keeps its bigint call in the body.  Negation reaches the
# `descroperation::neg` body through the canonical inline-call.  The `unary_neg`
# specialization (`try_emit_exact_int_uneg`) used to be consulted at that site
# and at the residual fallback, and declined INT_MIN outright because its
# negation is the 2**63 long; that fold is removed.  An earlier int-negation
# fold pinned the operand with GUARD_VALUE instead, and this loop ran 3x faster
# under it.
def main_int_min():
    m = -9223372036854775807 - 1  # INT_MIN as a machine int
    acc = 0
    i = 0
    while i < N:
        if -m == 9223372036854775808:
            acc = acc + 1
        i = i + 1
    assert acc == N
    print(acc)


class Derived(int):
    def __neg__(self):
        return "NEG"

    def __pos__(self):
        return "POS"

    def __invert__(self):
        return "INV"


def check_guard_transitions():
    int_min = -9223372036854775807 - 1
    promoted = 0
    for value in [7] * 3000 + [int_min] * 3000:
        result = -value
        promoted += result == 9223372036854775808
    assert promoted == 3000

    # A subclass arrives after the exact-int trace is hot without a branch at
    # the operation site; each unary fold must leave through its class guard.
    # The compiled loop GUARD_VALUEs each lambda's `Function.code`, and
    # `must_compile` keys that failing ref by `cast_ptr_to_int(refval)`. A
    # minor collection inside a guard's 200 `trace_eagerness` failures used to
    # move a still-young code object and restart its counter (guard_failures
    # 610/673/671 at a 1/2/4MiB nursery, 790 on windows-latest). The `fill`
    # allocation is larger than check.py's pinned 4MiB nursery, so a minor
    # collection promotes the code objects before the first transition.
    fill = list(map(str, range(50000)))
    del fill
    for op, expected in (
        (lambda value: -value, "NEG"),
        (lambda value: +value, "POS"),
        (lambda value: ~value, "INV"),
    ):
        result = None
        for value in [7] * 3000 + [Derived(7)]:
            result = op(value)
        assert result == expected


main()
main_int_min()
check_guard_transitions()
