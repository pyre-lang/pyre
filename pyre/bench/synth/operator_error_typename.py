# pyre-check: max-pypy-ratio=22
# pypy's exec time is pinned to the startup-subtraction floor here, so the
# ratio is not a measurement: the ceiling is twice the slowest the macos
# runner observes (11.0x). It read 10, a blanket value from the sweep that
# first gave every fixture a ceiling.
# Binary-operator and comparison TypeError messages name the operand's real
# class (not the placeholder 'object'), and the `**`/pow() message reads
# 'for ** or pow():'. A custom class with no numeric dunders drives every
# operator to its error path; a plain-int warmup loop first exercises the
# working fast paths. (Unary +/-/~/abs messages also carry the real class name
# but their text diverges between the oracles, so they are not asserted here.)
# Deterministic.
class Foo:
    pass


def warm(n):
    acc = 0
    for i in range(n):
        acc += (i << 1) + (i & 3) - (i % 5)
    return acc


def m(label, fn):
    try:
        fn()
        print(label, "no-error")
    except TypeError as e:
        print(label, type(e).__name__, e)


def main():
    print("warm", warm(2250000))
    f = Foo()
    m("add", lambda: 1 + f)
    m("sub", lambda: 1 - f)
    m("mul", lambda: 3 * f)
    m("truediv", lambda: 1 / f)
    m("floordiv", lambda: 1 // f)
    m("mod", lambda: 1 % f)
    m("lshift", lambda: 1 << f)
    m("rshift", lambda: 256 >> f)
    m("and", lambda: 255 & f)
    m("or", lambda: 1 | f)
    m("xor", lambda: 1 ^ f)
    m("add_l", lambda: f + 1)
    m("both", lambda: f << f)
    m("lt", lambda: 1 < f)
    m("gt_l", lambda: f > 1)
    m("pow_op", lambda: f ** 2)
    m("pow_builtin", lambda: pow(f, 2))
    m("pow_str", lambda: "a" ** 2)


main()
