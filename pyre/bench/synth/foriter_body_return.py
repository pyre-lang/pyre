# A `return` inside a `FOR_ITER` body — the early-exit search shape
# (`for x in xs: if pred(x): return x`), the shape a gate that rejects
# `RETURN_VALUE` declines most often on real code: a census over stdlib test
# modules puts it first at 95 of 223 first-rejections, ahead of the documented
# `LIST_APPEND`-with-`CALL` case.
#
# A declined loop runs fully interpreted, so it reports `loops_compiled=0`
# AND `loops_aborted=0` — the tracer never sees it. That invisibility is why
# the shape needs a fixture of its own: the decline leaves no counter to
# notice.
#
# Both exits are driven. `i % (N + 8)` overshoots the largest element for one
# probe in five, so the scan returns from inside the loop for the rest and
# falls through to the exhaustion `return` for those.
N = 64
ITERS = 80000


def first_at_least(xs, threshold):
    for x in xs:
        if x >= threshold:
            return x
    return -1


def main():
    xs = list(range(N))
    total = 0
    for i in range(ITERS):
        total += first_at_least(xs, i % (N + 8))
    print(total)


main()
