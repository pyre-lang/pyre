# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=main
# A store through the instance must be visible to a later load of the same
# object through the bound method's receiver. The two boxes differ and may
# alias. `OptHeap.force_lazy_sets_for_guard` has to invalidate the other
# array's cached item when it flushes the lazy `setarrayitem`, the way
# `ArrayCachedItem.force_lazy_set` does.
N = 200000


class C:
    def m(self, a, b):
        return self.x + a + b


def main():
    obj = C()
    obj.x = 0
    bm = obj.m
    total = 0
    i = 0
    while i < N:
        obj.x = i
        total += obj.m(1, 2)
        total += bm(1, 2)
        i += 1
    # sum_{i=0}^{N-1} 2 * (i + 1 + 2) = N * (N + 5)
    expect = N * (N + 5)
    if total != expect:
        raise AssertionError("%s != %s" % (total, expect))
    print(total)
    print("PASS")


main()
