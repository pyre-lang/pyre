# pyre-check: max-pypy-ratio=5.2
# macOS dynasm reads 1.4x; the floor, a fourth of the ceiling, sits under it.
# pyre-check: skip-cpython
# cpython 2.40s vs pyre 0.64s (3.8x on the ubuntu runner), and it is not
# gated on — only pypy is.
# Depth > 1 kept-stack branch guards whose kept / merge slot is a HEAP int
# (>= 256, a LOAD_CONST rather than the inline LoadSmallInt).  Sibling of
# `kept_stack_depth_gt1.py`, which keeps only small-int slots (11, 5, 2); here
# every short-circuit merges a heap-magnitude constant so the not-taken arm
# resumes with a hoisted (loop-invariant) heap-int box in a kept operand-stack
# slot.
#
# The short-circuit result flows into a wider expression (`total + (a + ...)`),
# so the merge slot lives at operand-stack depth > 1 across the `goto_if_not`.
# The heap constant is loop-invariant, so it is hoisted once; a merge-slot
# recovery that reads a stale / color-reused box for the hoisted heap-int
# delivers a WRONG value on the merge arm (the census-caught heap-int
# short-circuit miscompile that the small-int siblings do not exercise).
#
# This is also the resume shape most sensitive to the branch-guard resume
# coordinate: under a direct-jitcode-pc resume the not-taken arm is rebuilt at
# the guard pc, so the hoisted heap-int kept slot must be sourced faithfully
# there.  Byte-exact against the interpreter oracle pins that reconstruction.
#
# Asymmetric per-shape weighting makes any transposition or staleness change the
# checksum.  Pure arithmetic -> deterministic checksum across runtimes.
N = 36129033


def or_heap_in_add():
    # depth-2 `or`: stack = [a, (i & 1)] across the guard; the merge slot is
    # `(i & 1)` (truthy) or the heap const 1000000 (falsy, hoisted).
    total = 0
    i = 0
    while i < N:
        a = i + 100000
        total = total + (a + ((i & 1) or 1000000))
        i = i + 1
    return total


def and_heap_in_add():
    # depth-2 `and`: merge slot is 0 (falsy left kept) or the heap const 500000
    # (truthy left -> right).  The heap const only reaches the merge on the
    # truthy arm, so the not-taken (falsy) arm must resume with the small 0.
    total = 0
    i = 0
    while i < N:
        a = i + 100000
        total = total + (a + ((i & 1) and 500000))
        i = i + 1
    return total


def nested_heap_short_circuit():
    # nested `(flag and HEAP) or HEAP`: two heap consts, two chained guards, the
    # inner `and` result kept across the outer `or`.
    acc = 0
    i = 0
    while i < N:
        flag = i & 1
        acc = acc + ((flag and 100000) or 200000)
        i = i + 1
    return acc


def or_chain_heap():
    # `a or b or HEAP`: two guards; the heap const 300000 is the final fallback,
    # kept-merged when both `a` and `b` are falsy.
    acc = 0
    i = 0
    while i < N:
        a = i & 1
        b = i & 2
        acc = acc + (a or b or 300000)
        i = i + 1
    return acc


def main():
    print(
        or_heap_in_add(),
        and_heap_in_add(),
        nested_heap_short_circuit(),
        or_chain_heap(),
    )


main()
