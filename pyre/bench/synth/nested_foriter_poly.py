# pyre-check: max-pypy-ratio=13
# Nested FOR_ITER with polymorphic type flip in the inner body.
# The inner loop iterates a list that changes type mid-way (int → float),
# triggering a type guard deopt inside a nested FOR_ITER.

def main():
    data = list(range(100)) + [0.5] + list(range(100))
    total = 0
    n = 0
    while n < 40000:
        for x in data:
            total += x
        n += 1
    return total

result = main()
print(result)
# Expected: 40000 * (sum(range(100)) + 0.5 + sum(range(100)))
#         = 40000 * (4950 + 0.5 + 4950) = 40000 * 9900.5 = 396020000.0
