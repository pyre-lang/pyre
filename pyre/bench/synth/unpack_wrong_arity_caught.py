# pyre-check: max-pypy-ratio=64
# A hot exact-length unpack specializes for the common arity.  The rare
# mismatching tuple must deopt at the UNPACK_SEQUENCE validation call and
# deliver ValueError through this frame's exception handler.

MOD = 1000000007


def main():
    acc = 0
    for i in range(960000):
        try:
            a, b = (i, i + 1, i + 2) if i % 13 == 0 else (i, i + 1)
            acc = (acc + a + b) % MOD
        except ValueError:
            acc = (acc + 17) % MOD

        try:
            x, y, z = (i, i + 1) if i % 7 == 0 else [i, i + 1, i + 2]
            acc = (acc + x + y + z) % MOD
        except ValueError:
            acc = (acc + 29) % MOD
    return acc


print(main())
