# pyre-check: max-pypy-ratio=2.7
# main CI 10-24..10-26 (6 runs, all native legs): 1.1x-2.0x; ceiling = max x 1.35.
def loop_with_two_backedges(n):
    high = 0
    for i in range(n):
        value = i % 7
        high = value if value > high else high
        if i % 45 == 0:
            high = 0
    return high


result = loop_with_two_backedges(53424391)
assert result == 6
print(result)
