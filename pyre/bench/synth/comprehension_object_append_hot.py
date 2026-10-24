# pyre-check: max-pypy-ratio=32
# The 102 ceiling (twice an old Windows 51.0x) put the derived floor at 25.5x.
# Current readings sit at 9–14x (macos 9.1–11.3, windows 13.6, linux 11.2);
# pypy above FLOOR_GATE_MIN_BASELINE_S then fails the floor. 32 leaves headroom
# above the slowest current reading (13.6x) and the linux cranelift estimate
# (~23x).
# An inlined list comprehension whose LIST_APPEND element lands in a list
# Object-strategy (tuple / None / str / dict / f-string) folds through the #171
# orthodox append. Its Object arm stores a GC ref and runs list_write_barrier,
# an idempotent Void residual. The barrier must be exempt from the FBW in-flight
# FOR_ITER body-effect accounting; otherwise the trace-attempt iteration is
# refuse-dropped and the comprehension yields one element short (len N-1 instead
# of N). int elements take the unboxed fast path (no barrier) and were never
# affected. (A non-empty nested-list element `[[i] …]` is intentionally omitted:
# it declines to the interpreter until the fold threads the inner list's backing
# block into guard-exit resume data.)


def tuple_comp(n):
    return [(i, i) for i in range(n)]


def none_comp(n):
    return [None for i in range(n)]


def str_comp(n):
    return ["s" for i in range(n)]


def dict_comp(n):
    return [{i: i} for i in range(n)]


def fstring_comp(n):
    return [f"{i}" for i in range(n)]


def int_comp(n):
    return [i for i in range(n)]


def main():
    total = 0
    k = 0
    while k < 500:
        total += len(tuple_comp(1000))
        total += len(none_comp(1000))
        total += len(str_comp(1000))
        total += len(dict_comp(1000))
        total += len(fstring_comp(1000))
        total += len(int_comp(1000))
        k += 1
    print(total)


main()
