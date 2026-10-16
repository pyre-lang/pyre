# pyre-check: max-pypy-ratio=16
# FOR_ITER inside a while-loop: the common real-world pattern.
# The while-loop is the JIT entry; the for-loop body must handle the FOR_ITER
# liveness scoping correctly.  `outer_var` additionally reads the outer
# loop's variable from inside the for-body.


def main():
    total = 0
    n = 0
    while n < 50000:
        for x in range(100):
            total += x
        n += 1
    return total


def outer_var():
    total = 0
    n = 0
    while n < 5000:
        for i in range(200):
            total += i * n
        n += 1
    return total


print(main())
# Expected: 50000 * sum(range(100)) = 50000 * 4950 = 247500000
print(outer_var())
# Expected: sum(n for n in range(5000)) * sum(i for i in range(200))
#         = 12497500 * 19900 = 248700250000
