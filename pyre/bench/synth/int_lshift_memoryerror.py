# pyre-check: max-pypy-ratio=12
# A left shift whose result is too large to allocate raises a catchable
# MemoryError instead of aborting the process on the big-int allocation
# failure. Normal and moderately-large shifts still compute, and a negative
# count raises ValueError. A warmup loop exercises the small-shift fast path.
def warm(n):
    acc = 0
    for i in range(n):
        acc ^= (1 << (i % 60))
    return acc


def m(label, fn):
    try:
        r = fn()
        print(label, "ok", r.bit_length())
    except BaseException as e:
        print(label, type(e).__name__, repr(str(e)))


def main():
    print("warm", warm(2700000))
    # normal / moderately large shifts compute
    m("1<<10", lambda: 1 << 10)
    m("255<<8", lambda: 255 << 8)
    m("big_value_overflow", lambda: (10**18) << 4)
    m("1<<1000", lambda: 1 << 1000)
    m("1<<1e6", lambda: 1 << (10**6))
    m("bignum<<100", lambda: (10**50) << 100)
    # negative shift count
    m("1<<-1", lambda: 1 << -1)
    m("bignum<<-1", lambda: (10**50) << -1)
    # result too large to allocate -> catchable MemoryError, not an abort
    m("1<<1e18", lambda: 1 << (10**18))
    m("2<<1e18", lambda: 2 << (10**18))
    m("bignum<<1e18", lambda: (10**50) << (10**18))


main()
