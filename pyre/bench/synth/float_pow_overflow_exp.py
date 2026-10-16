# pyre-check: max-pypy-ratio=12
# A float or complex power coerces each int operand to a double first, so an
# over-range int base or exponent raises OverflowError up front (even
# `1.0 ** huge`, which never reaches the arithmetic), matching float(huge).
# Ordinary float/complex/int powers still compute. A warmup loop exercises the
# float-pow fast path.
def warm(n):
    acc = 0.0
    for i in range(n):
        acc += 2.0 ** (i % 8)
    return acc


def m(label, fn):
    try:
        print(label, "->", repr(fn()))
    except BaseException as e:
        print(label, "!!", type(e).__name__, repr(str(e)))


def main():
    print("warm", warm(2250000))
    # over-range exponent -> OverflowError from the int->float coercion
    m("2.0**huge", lambda: 2.0 ** (10**1000))
    m("pow(2.0,huge)", lambda: pow(2.0, 10**1000))
    m("0.5**huge", lambda: 0.5 ** (10**1000))
    m("1.0**huge", lambda: 1.0 ** (10**1000))
    m("pow(2j,huge)", lambda: pow(2j, 10**1000))
    m("(1+1j)**huge", lambda: (1 + 1j) ** (10**1000))
    # over-range base
    m("huge**2.0", lambda: (10**1000) ** 2.0)
    # control: float() of the same int also overflows
    m("float(huge)", lambda: float(10**1000))
    # ordinary powers still compute
    m("2.0**10", lambda: 2.0 ** 10)
    m("2.0**-2", lambda: 2.0 ** -2)
    m("2**100", lambda: 2 ** 100)
    m("(2+1j)**2", lambda: (2 + 1j) ** 2)
    m("pow(2,10,5)", lambda: pow(2, 10, 5))


main()
