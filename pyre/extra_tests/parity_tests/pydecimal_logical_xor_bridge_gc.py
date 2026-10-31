# CPython-suite gap: test_decimal.PyIBMTestCases.test_dqInvert runs this
# listcomp, but the suite cannot pin a tiny nursery around the first
# precision-34 xor after a precision-16 warmup, which is when a bridge
# compile can collect.
# parity-tests reason: `_pydecimal.logical_xor` builds
# `"".join([str(int(a)^int(b)) for a,b in zip(opa,opb)])`. A traced loop
# at precision 16 then a precision-34 call (and a non-logical operand)
# compiles a bridge while the nursery is full; the optimizer's
# ResumeDataLoopMemo consts and InputArg refs must stay rooted across
# that compile. A stale nursery word here is a SIGSEGV with no Python
# traceback.
# parity-env: PYPY_GC_NURSERY=1
# parity-env: PYPY_GC_DEBUG=1
# parity-env: MAJIT_GC_NURSERY_POISON=1

"""_pydecimal logical_xor listcomp survives a moving collection at a bridge."""

import faulthandler

try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

faulthandler.enable()

import _pydecimal as dec


def ctx(prec):
    c = dec.Context(prec=prec)
    c.traps[dec.InvalidOperation] = False
    return c


def invert_many(prec, n, seed):
    c = ctx(prec)
    bits = ("01" * prec)[:prec]
    d = dec.Decimal(bits)
    acc = dec.Decimal(0)
    i = 0
    while i < n:
        acc = acc + d.logical_invert(c)
        if i % 31 == 0:
            c.logical_invert(dec.Decimal("2"))
        i = i + 1
    return acc + seed


# Precision 16 is ddInvert; 34 is dqInvert. The crash was the first
# precision-34 xor after the shorter loop had already compiled.
ROUNDS = 400
got16 = invert_many(16, ROUNDS, dec.Decimal(0))
got34 = invert_many(34, ROUNDS, got16)

c16 = ctx(16)
c34 = ctx(34)
assert str(dec.Decimal("0").logical_invert(c16)) == "1111111111111111"
assert str(dec.Decimal("1").logical_invert(c16)) == "1111111111111110"
assert str(dec.Decimal("0").logical_invert(c34)) == "1" * 34
assert str(dec.Decimal("1").logical_invert(c34)) == "1" * 33 + "0"
assert str(c34.logical_invert(dec.Decimal("2"))) == "NaN"

# Mixed-precision loop: _fill_logical's pad length changes each call,
# so the compiled xor body takes a bridge instead of staying in one trace.
mixed = dec.Decimal(0)
i = 0
while i < ROUNDS:
    mixed = mixed + dec.Decimal("01" * 8).logical_invert(c16)
    mixed = mixed + dec.Decimal("01" * 17).logical_invert(c34)
    i = i + 1

assert got16 != 0
assert got34 != 0
assert mixed != 0
print("OK")
