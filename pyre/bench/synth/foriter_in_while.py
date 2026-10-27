# pyre-check: max-pypy-ratio=13
# FOR_ITER inside a while-loop: the common real-world pattern.
# The while-loop is the JIT entry; the for-loop body must handle the FOR_ITER
# liveness scoping correctly.  `outer_var` additionally reads the outer
# loop's variable from inside the for-body.


def main():
    total = 0
    n = 0
    while n < 200000:
        for x in range(100):
            total += x
        n += 1
    return total


def outer_var():
    total = 0
    n = 0
    while n < 20000:
        for i in range(200):
            total += i * n
        n += 1
    return total


print(main())
# Expected: 200000 * sum(range(100)) = 200000 * 4950 = 990000000
print(outer_var())
# Expected: sum(n for n in range(20000)) * sum(i for i in range(200))
#         = 199990000 * 19900 = 3979801000000
