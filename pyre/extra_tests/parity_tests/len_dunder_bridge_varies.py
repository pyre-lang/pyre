# CPython-suite gap: no suite test calls `len()` on a user `__len__` whose
# answer changes, including a negative one, in a loop hot enough to compile
# a bridge through the operator.
#
# parity-tests reason: the bridge walks `len`'s own tail after `__len__`
# returns. A result baked at trace time, or a tail that skips the length
# check, prints a different total and error count.
#
# A hot `len()` whose `__len__` answers a different value on every call,
# including a negative that raises ValueError. The bridge through the
# operator must re-run the length check; a constant baked at trace time
# would keep answering the first call.
#
# len_dunder_guard_in_bridge_callee:
# no suite test fails a guard inside a `__len__` that a compiled bridge resumed into, with
# the answer going negative, float or bool. the resumed frame chain has to keep `len`'s
# tail between the caller and `__len__`; resuming straight to the caller returns the raw
# dunder result.


class Left:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


class Right:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


def main():
    left = Left(3)
    right = Right(8)
    neg = Left(-1)
    total = 0
    errors = 0
    seen = []
    # Stay on Left long enough to trace, then switch. A bridge that baked
    # the first answer would keep returning 3.
    for i in range(8000):
        if i < 4000:
            obj = left
        elif i % 17 == 0:
            obj = neg
        else:
            obj = right
        try:
            n = len(obj)
        except ValueError:
            errors += 1
            n = -1
        total += n
        if i in (0, 3999, 4000, 4001, 4017, 7999):
            seen.append((i, n))
    assert (total, errors, seen) == (
        41885,
        235,
        [(0, 3), (3999, 3), (4000, 8), (4001, 8), (4017, 8), (7999, 8)],
    ), (total, errors, seen)


main()

def len_dunder_guard_in_bridge_callee():
    class Sized:
        def __init__(self, n):
            self.n = n

        def __len__(self):
            n = self.n
            if n > 1000:
                return -n
            if n == 500:
                return 2.5
            if n == 250:
                return True
            return n


    def main():
        objs = [Sized(i % 64) for i in range(64)]
        total = 0
        counts = {}
        for i in range(12000):
            obj = objs[i % 64]
            if i > 6000 and i % 97 == 0:
                obj = Sized(2000)
            elif i > 7000 and i % 89 == 0:
                obj = Sized(500)
            elif i > 8000 and i % 83 == 0:
                obj = Sized(250)
            try:
                total += len(obj)
            except (ValueError, TypeError) as e:
                key = type(e).__name__
                counts[key] = counts.get(key, 0) + 1
        result = (total, sorted(counts.items()))
        assert result == (372400, [("TypeError", 55), ("ValueError", 62)]), result


    main()
len_dunder_guard_in_bridge_callee()
print("OK")
