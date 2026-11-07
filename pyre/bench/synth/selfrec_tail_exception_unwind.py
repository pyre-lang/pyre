# pyre-check: max-pypy-ratio=46
# After FBW inline screens dropped (#2600), cranelift CI reads 31.7-39.4x
# (dynasm stays ~8-12x under the pypy floor). 46 is 15% over the slower
# macos cranelift reading (39.4x, pyre-lang #2604).
"""Self-recursive CALL_ASSEMBLER exception unwind regression.

The raise happens while evaluating a tail call's accumulator argument.  It
must cross several suspended copies of the same JIT frame before the outer
handler sees it; the normal recursive-call return slot must never be consumed.
"""

MODULUS = 2_147_483_647


def check(value):
    if value == 0:
        raise ZeroDivisionError("self-recursive unwind")
    return (100 // value) % MODULUS


def recur(depth, accumulator):
    if depth <= 0:
        return accumulator
    return recur(depth - 1, (accumulator + check(depth - 2)) % MODULUS)


checksum = 0
for iteration in range(68_000):
    try:
        checksum = (checksum + recur(iteration % 8, 0)) % MODULUS
    except ZeroDivisionError:
        checksum = (checksum + 17) % MODULUS

print(checksum)
