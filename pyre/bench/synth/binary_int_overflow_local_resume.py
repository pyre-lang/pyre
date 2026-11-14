# pyre-check: max-pypy-ratio=2.8
# Overflowing integer operations must resume with the values that were loaded
# from local slots, including values produced by recursive calls and unpacking.

MODULUS = 1000000007


def factorial_local(n):
    if n <= 1:
        return 1
    a = factorial_local(n - 1)
    return a * n


def factorial_inline(n):
    if n <= 1:
        return 1
    return factorial_inline(n - 1) * n


def factorial_total(function, n, repetitions):
    total = 0
    for i in range(repetitions):
        total = (total + function(n)) % MODULUS
    return total


def fibpair(n):
    if n == 0:
        return (0, 1)
    a, b = fibpair(n - 1)
    return (b, a + b)


def fibpair_total(n, repetitions):
    total = 0
    for i in range(repetitions):
        a, b = fibpair(n)
        total = (total + b % MODULUS) % MODULUS
    return total


def plain_bignum_multiply(repetitions):
    value = 1 << 62
    total = 0
    for i in range(repetitions):
        product = value * (i % 5 + 2)
        total = (total + product % MODULUS) % MODULUS
    return total


print(factorial_total(factorial_local, 21, 200))
print(factorial_total(factorial_inline, 21, 200))
print(factorial_total(factorial_local, 20, 200))
print(fibpair_total(93, 145455))
print(fibpair_total(40, 145455))
print(plain_bignum_multiply(363637))
