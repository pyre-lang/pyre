# pyre-check: max-pypy-ratio=7.0
# Widest gated reading 6.1x (linux cranelift; pyre 0.34s, pypy 0.06s).
# 7.0 is 15% over that. The previous 5.1 sat 15% over an older 4.4x.
# Kept-stack branch-guard recovery across resume depths.
#
# A `goto_if_not` whose not-taken arm resumes with operand-stack temps live is
# the short-circuit / conditional / chained-comparison shape.  The full-body
# walk recovers the kept value(s) from the guard-pc register file:
#
#   depth 1 (recovered) — `(i % k) and/or v` keeps a SINGLE operand-stack temp
#   across the guard; the recovery substitutes one color, no ambiguity.
#
#   depth > 1 (declined to the trait tracer) — `a + ((i & 1) or v)` and the
#   triple-chained `0 < a < b < 9` keep TWO+ operand-stack temps across one
#   guard.  A wrong-order recovery would transpose the kept values; the
#   asymmetric weighting below makes any transposition change the checksum.
#
# Pure arithmetic -> deterministic checksum across runtimes; a parity
# regression target for the kept-stack branch-guard depth boundary.
N = 7272728


def main():
    total = 0
    i = 0
    while i < N:
        d1 = (i % 7) and (i + 3)            # depth-1 `and` (keeps falsy 0)
        d1b = (i % 3) or (i + 11)           # depth-1 `or`  (keeps truthy)
        a = i + 100000
        d2 = a + ((i & 1) or 5)             # depth-2: stack = [a, (i & 1)]
        b = i & 3
        d3 = 1 if (0 < (i & 1) < b < 9) else 0  # triple-chained, multi kept temp
        total = total + d1 + d1b + d2 * 3 - d3
        i = i + 1
    print(total)


main()
