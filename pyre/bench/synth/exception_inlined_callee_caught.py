# pyre-check: max-pypy-ratio=2.1
# main CI 10-24..10-26 (6 runs, all native legs): 0.7x-1.5x; ceiling = max x 1.35.
# A ceiling under 3 derives no floor.
N = 131327700


# Data descriptor with __delete__ but no __set__: a read-only data
# descriptor.  Assigning through it raises AttributeError rather than
# shadowing it with an instance-dict entry (descroperation.py:124-126).
class DeleteOnly:
    def __delete__(self, obj):
        pass


class C:
    d = DeleteOnly()


def assign(obj):
    obj.d = 1


def main():
    c = C()
    acc = 0
    i = 0
    # The raising STORE_ATTR lives in the inlined callee `assign`; the
    # handler lives in this virtualizable caller frame.  The JIT records a
    # GuardNoException after the residual store, deopts into the blackhole
    # every iteration, and the caller must resume at its CALL's
    # catch_exception so the exception is caught here rather than escaping.
    while i < N:
        try:
            assign(c)
        except AttributeError:
            acc = acc + 1
        i = i + 1
    print(acc)


main()
