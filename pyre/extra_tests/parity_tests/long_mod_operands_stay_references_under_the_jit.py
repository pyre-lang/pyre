# CPython-suite gap: longobject tests do not run `%`, `//` or divmod on
# nursery-born longs in a loop hot enough to be traced.
# parity-tests reason: the `_divrem` lhs-remainder test in `long_mod` and
# `divmod` must reach its residual as GC references; an address cast to a
# machine word folds into a constant that a minor collection leaves stale.
#
# divmod_long_int_jit:
# divmod tests do not trace mixed bigint/int results under GC.
# this is the gc-poison owner for pyre's rooted bigint pair result.

"""Long `%`, `//` and divmod keep their operands across a minor collection.

Every step allocates fresh longs, so the default nursery collects many times
while the loops run traced. `mix` covers the reflected int receiver and the
remainder that reuses the dividend's payload; `divmod` covers the same
payload reuse on its remainder half.
"""

import gc

try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

BIGP = 18446744073709551629
BIGA = 100000000000000000000000
ROUNDS = 3000


def mix(h, v):
    return (h * 3 + (v % BIGP)) % BIGP


def floordiv_mix(h, v):
    return (h * 7 + (BIGA + v) // BIGP) % BIGP


def divmod_mix(h, v):
    q, r = divmod(BIGP + v * 5, BIGA)
    return (h * 3 + q + r) % BIGP


# One loop per shape: each traces on its own, the way the defect was found.
h_int = 0
for i in range(ROUNDS):
    h_int = mix(h_int, i)
h_long = 0
for i in range(ROUNDS):
    h_long = mix(h_long, BIGA + i)
h_floordiv = 0
for i in range(ROUNDS):
    h_floordiv = floordiv_mix(h_floordiv, i)
h_divmod = 0
for i in range(ROUNDS):
    h_divmod = divmod_mix(h_divmod, i)

assert h_int == 10169797193228046475, h_int
assert h_long == 13267213028806569560, h_long
assert h_floordiv == 14656875528999295164, h_floordiv
assert h_divmod == 13955497818721129117, h_divmod


def run_divmod_long_int_jit():
    """Keep both bigint divmod halves live across allocation and collection.

    The performance/fold census lives in ``bench/synth/divmod_long_int_pair.py``.
    This small correctness leg remains here because the parity runner can enable
    ``MAJIT_GC_NURSERY_POISON``; the synth runner cannot.
    """

    BIG = (1 << 200) + 12345
    ROUNDS = 3000

    held = []
    for i in range(ROUNDS):
        divisor = 97 if i & 1 else -97
        q, r = divmod(BIG + i, divisor)
        assert q * divisor + r == BIG + i
        assert type(q) is int and type(r) is int
        held.append((i, divisor, q, r))
        if len(held) > 16:
            held.pop(0)
        if i % 256 == 0:
            gc.collect()
        for index, old_divisor, old_q, old_r in held:
            assert old_q * old_divisor + old_r == BIG + index

    # Leave the bigint-pair arm for small quotients and rejected divisors without
    # reusing its recorded result shape.
    tiny = (1 << 70) >> 70
    assert divmod(tiny, -7) == (-1, -6)
    q, r = divmod(BIG, -(1 << 63))
    assert q * -(1 << 63) + r == BIG
    assert divmod(BIG, True) == (BIG, 0)
    try:
        divmod(BIG, False)
    except ZeroDivisionError:
        pass
    else:
        raise AssertionError("divmod by False did not raise")


run_divmod_long_int_jit()
print("OK")
