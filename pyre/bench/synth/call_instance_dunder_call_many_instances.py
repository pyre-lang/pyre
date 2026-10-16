"""Regression: inlining a call through the callable's own `__call__` must stay
specialized on the receiver's CLASS, not on the receiver.

`space.lookup` promotes the callable's class version to resolve `__call__`, and
the inline pins exactly that (a class guard plus the type's version tag) while
`self` reaches the callee as a red operand.  A per-instance `GUARD_VALUE` on top
of it specializes the compiled body on whichever instance happened to be tracing,
so every other instance of the same class side-exits.  Bridges absorb the first
few and then saturate, at which point the majority of calls run interpreted.

Measured over 200000 calls before the fix, one callable class:

      instances   bridges   guard_failures
              1         0                1
              2         2              401
              4         5             1203
             16        29            13441
             64        29            87424     <- saturated, 44% of calls

and 17.3x CPython against 3.0x for the same loop on a single instance.  The
gate here is `guard_failures`: it reads 1 with the class guard doing the work
and rises into the thousands the moment the receiver is pinned by identity.

The sum is checked because the two failure modes are not the same bug.  Each
instance carries a distinct `k`, so folding the receiver's field from the
tracing instance would answer with the wrong `k` rather than merely deopt.
"""

N_INSTANCES = 64
N_CALLS = 1200000


class Adder:
    def __init__(self, k):
        self.k = k

    def __call__(self, x):
        return x + self.k


adders = [Adder(i) for i in range(N_INSTANCES)]

total = 0
for i in range(N_CALLS):
    total += adders[i % N_INSTANCES](i)

# sum(i) for i in range(N_CALLS), plus each instance's k once per visit.
expected = N_CALLS * (N_CALLS - 1) // 2
expected += (N_CALLS // N_INSTANCES) * (N_INSTANCES * (N_INSTANCES - 1) // 2)
assert total == expected, f"{total} != {expected}"
print("OK")
