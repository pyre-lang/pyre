# A hot FOR_ITER over many short generators of one function. After each
# drain, `gi_frame` must be None: `_send_ex` runs `frame_is_finished`
# (clears the frame edge, cancels the finalizer) before StopIteration.
# The traced path used to bind a null FOR_ITER item without that teardown,
# so `gi_frame` stayed live after exhaustion.
ROUNDS = 1800


def g():
    yield 1


def run(rounds):
    n = 0
    r = 0
    while r < rounds:
        gen = g()
        for x in gen:
            n += x
        if gen.gi_frame is None:
            n += 1
        r += 1
    return n


total = run(ROUNDS)
assert total == 3600, total
print(total)
