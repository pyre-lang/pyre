# `%` and `//` by a small positive constant over a dividend that crosses zero.
#
# Both operators reach the optimizer as the `int_py_mod` / `int_py_div` oopspec
# calls and are expanded against the constant divisor, by magic-number
# multiplication where the backend has a cheap 64x64->128 multiply and through
# the truncating `IntMod` / `IntFloorDiv` primitives plus a sign correction
# where it does not. The two expansions have to agree with each other and with
# Python, and they only differ on a negative dividend -- the correction is
# exactly what a non-negative one lets both of them drop.
#
# The divisors are 7, 3 and 5: positive, none a power of two, so neither the
# `x & (2**k - 1)` nor the `x >> k` arm claims them. Weighted so a sign error
# in any one term moves the checksum.
N = 4000000


def main():
    total = 0
    i = -N
    while i < N:
        total = total + (i % 7) + (i % 3) * 2 + (i % 5) * 3
        total = total + (i // 7) + (i // 3) * 2 + (i // 5) * 3
        i = i + 1
    print(total)


main()
