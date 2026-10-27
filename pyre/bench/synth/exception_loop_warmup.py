# pyre-check: max-pypy-ratio=2.6
# main CI 10-24..10-26 (6 runs, all native legs): 1.2x-1.9x; ceiling = max x 1.35.
# Warm-up-then-raise exception handling: the loop runs cleanly long enough to
# compile, then a nested try/(try-finally)/except starts raising only after the
# warm-up window. The post-warm-up raise is therefore NOT in the recorded trace,
# so the guard failure resumes in the blackhole and the exception must hop
# floordiv -> inner finally -> reraise -> outer except entirely under blackhole
# control. Benches that raise from iteration 1 do not exercise this path.
N = 47040000


def nested(n):
    acc = 0
    i = 0
    while i < n:
        v = 1
        try:
            try:
                if i >= 2000 and i % 19 == 0:
                    acc //= 0
                v = 2
            finally:
                v += 10
        except ZeroDivisionError:
            v += 100
        acc = (acc + v + i % 3) % 1000003
        i += 1
    return acc


def main():
    print(nested(N))


main()
