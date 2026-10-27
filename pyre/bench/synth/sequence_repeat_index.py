# pyre-check: max-pypy-ratio=22
# str/tuple `*` route the repeat count through getindex_w: a custom __index__
# repeats the sequence just like list/bytes, and an out-of-index-range __index__
# raises OverflowError naming the original operand. A hot loop first warms
# str/tuple `__mul__` with plain ints so the trace JIT compiles the repeat path,
# then a deterministic tail exercises the custom-__index__ cases and catches the
# overflows. Deterministic.
class Three:
    def __index__(self):
        return 3


class BigIndex:
    def __index__(self):
        return 1 << 70


def warm(n):
    acc = 0
    for i in range(n):
        k = i % 5 + 1
        acc += len("ab" * k)
        acc += len((7,) * k)
    return acc


def repeat(w_seq, w_count):
    try:
        return repr(w_seq * w_count)
    except (OverflowError, TypeError) as e:
        return f"{type(e).__name__} {e}"


def main():
    print("warm", warm(1800000))
    print("tuple three", repeat((7,), Three()))
    print("str three", repeat("ab", Three()))
    print("tuple big", repeat((7,), BigIndex()))
    print("str big", repeat("ab", BigIndex()))


main()
