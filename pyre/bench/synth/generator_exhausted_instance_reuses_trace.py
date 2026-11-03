# A FOR_ITER over generators of one function, some of them already
# exhausted before the loop starts. The inner loop compiles on a live
# instance — they share a pycode, so the resume walk pins that code
# rather than generator identity — and later the same compiled site
# sees drained siblings whose frame_ptr is null.
#
# The shape exists because `_send_ex` handles `frame is None` before
# touching last_instr, and the pypy3 trace is getfield of inst_frame
# then guard_nonnull. Without that guard the compiled loop dereferences
# the exhausted frame. The first generator is long-lived so the inner
# FOR_ITER compiles on one yield pc; a `break` leaves that frame
# installed so the loop is not the finished-frame RETURN path.
ROUNDS = 1800
INNER = 4
WARM = 1800


def g(n):
    i = 0
    while i < n:
        yield i
        i += 1


def drain(gen, n):
    # Residual `next()` so `_send_ex` runs `frame_is_finished` and
    # clears `frame_ptr`. A FOR_ITER drain of `g` would compile the
    # same resume walk and leave the finished frame installed.
    i = 0
    while i < n:
        next(gen)
        i += 1
    try:
        next(gen)
    except StopIteration:
        return


def run(rounds, n, warm):
    gens = []
    gens.append(g(warm + 100))
    r = 0
    while r < rounds:
        dead = g(n)
        drain(dead, n)
        gens.append(dead)
        r += 1
    acc = 0
    count = 0
    first = True
    for gen in gens:
        for x in gen:
            acc += x
            count += 1
            if first and count >= warm:
                break
        first = False
    return acc, count


acc, count = run(ROUNDS, INNER, WARM)
assert (acc, count) == (1619100, 1800), (acc, count)
print(acc, count)
