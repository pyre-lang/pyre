# pyre-check: max-pypy-ratio=6.4
# Widest reading 5.7x with 28% cranelift spread; the ceiling covers 2.32s/0.33s.
# Depth > 1 kept-stack branch guards: chained comparisons, nested
# short-circuits and conditional expressions keep two or more operand-stack
# temps live across a `goto_if_not`.  The not-taken arm resumes at a merge
# point the full-body walk has not written; the kept temps are recovered
# from the not-taken edge's `ref_copy` parallel-move trampoline (the depth-1
# positional heuristic does not generalize to depth > 1).
#
# Each shape lives in its own minimal loop so the kept-stack guard is the
# whole trace and a miscompile is not diluted by a multi-shape loop.  Pure
# arithmetic -> deterministic checksum.  Regression guard for the depth > 1
# decline removal: `0 < a < b < 9` must read 375000 (a miscompile gives
# 749949) and `total + (a + ((i & 1) or 5))` 1275003750000 (miscompile:
# 1837502750500).
N = 14258537


def chain2():
    acc = 0
    i = 0
    while i < N:
        a = i & 1
        b = i & 3
        acc = acc + (1 if 0 < a < b < 9 else 0)
        i = i + 1
    return acc


def chain3():
    acc = 0
    i = 0
    while i < N:
        a = i & 7
        b = i & 15
        c = i & 31
        acc = acc + (1 if 0 < a < b < c < 50 else 0)
        i = i + 1
    return acc


def nested_or_in_add():
    total = 0
    i = 0
    while i < N:
        a = i + 100000
        total = total + (a + ((i & 1) or 5))
        i = i + 1
    return total


def nested_short_circuit():
    acc = 0
    i = 0
    while i < N:
        flag = i & 1
        acc = acc + ((flag and 11) or 2)
        i = i + 1
    return acc


def or_chain_value():
    acc = 0
    i = 0
    while i < N:
        a = i & 1
        b = i & 2
        c = i & 4
        acc = acc + (a or b or c)
        i = i + 1
    return acc


def main():
    print(
        chain2(),
        chain3(),
        nested_or_in_add(),
        nested_short_circuit(),
        or_chain_value(),
    )


main()
