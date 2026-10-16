# `len()` on ranges whose stored length does and does not fit a machine word.
#
# `functional.py:496-497 W_Range.descr_len` returns the precomputed
# `self.w_length` field, and `descr_new` stores a bignum there whenever
# `compute_range_length` leaves the machine range. The fold reads that field
# behind a class guard, so the loaded box needs its own guard: the receiver
# being an exact range says nothing about what its length slot holds, and the
# hot small ranges below would otherwise let a bignum-length range take their
# recorded exit and hand the bignum straight to `len()`.
N = 7200000

SMALL = range(10)
HUGE = range(-4611686018427387904, 4611686018427387904)


def main():
    total = 0
    for _ in range(N):
        total += len(SMALL)
    try:
        len(HUGE)
        huge = "no-error"
    except OverflowError:
        huge = "OverflowError"
    print(total, huge)


main()
# Expected: 72000000 OverflowError
