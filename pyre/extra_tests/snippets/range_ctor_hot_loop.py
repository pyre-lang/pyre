# pyre-check: gate=1
# Machine-int `range(stop)` / `range(start, stop)` / `range(start, stop, step)`
# in a hot loop, plus residual shapes: empty negative stop, zero step,
# bigint bound, and a user `__index__`.
try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass


class Index:
    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


def main():
    total = 0
    empty = 0
    rev = 0
    big = 0
    zero = 0
    indexed = 0
    i = 0
    while i < 400:
        for v in range(-5):
            empty += v
        for v in range(10, 0, -3):
            rev += v
        r = range(2**70)
        big += r.start == 0 and r.stop == 2**70
        try:
            range(0, 10, 0)
        except ValueError:
            zero += 1
        indexed += len(range(Index(4)))
        total += len(range(i % 7))
        i += 1
    assert empty == 0, empty
    assert rev == 400 * (10 + 7 + 4 + 1), rev
    assert big == 400, big
    assert zero == 400, zero
    assert indexed == 400 * 4, indexed
    print(total, empty, rev, zero, indexed)
    print("OK")


if __name__ == "__main__":
    main()
