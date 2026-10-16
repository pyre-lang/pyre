# pyre-check: max-pypy-ratio=16.2
# dynasm and cranelift read in the `?` or `~` band, so the floor does not arm.
# Three-argument `pow(base, exp, mod)` requires all operands to be integers.
# A float base rejects the modulus with a TypeError, a complex base with a
# ValueError ("complex modulo"), and the all-integer forms compute the modular
# power. A warmup loop exercises the working integer path first. Only cpython==
# pypy outputs are asserted (the mixed int/float operand errors name the three
# operand types, which the two oracles word differently). Deterministic.
def m(label, fn):
    try:
        print(label, "->", repr(fn()))
    except BaseException as e:
        print(label, "!!", type(e).__name__, repr(str(e)))


def warm(n):
    acc = 0
    for i in range(n):
        acc = (acc + pow(i % 7, 13, 1000)) % 1000
    return acc


class MyInt(int):
    def __pow__(self, exp, mod=None):
        return ("MyInt.pow", int(self), exp, mod)


def main():
    print("warm", warm(660000))
    # All-integer 3-arg forms compute the modular power.
    m("i_i_i", lambda: pow(2, 10, 100))
    # A base whose type overrides __pow__ is honoured for 3-arg pow (the
    # integer fast path must not shadow the override).
    m("sub_i_i", lambda: pow(MyInt(2), 3, 5))
    m("sub_i_2", lambda: pow(MyInt(2), 3))
    m("b_i_i", lambda: pow(True, 10, 100))
    m("i_neg_i", lambda: pow(3, -1, 7))
    m("i_i_N", lambda: pow(2, 10, None))
    # 2-arg forms are unaffected.
    m("f_i_2", lambda: pow(2.0, 10))
    m("i_i_2", lambda: pow(2, 10))
    # A float base rejects a real modulus.
    m("f_i_i", lambda: pow(2.0, 10, 100))
    m("f_f_i", lambda: pow(2.0, 10.0, 100))
    m("f_i_N", lambda: pow(2.0, 10, None))
    # A complex base rejects a modulus with a ValueError.
    m("c_i_i", lambda: pow(2j, 10, 100))


main()
