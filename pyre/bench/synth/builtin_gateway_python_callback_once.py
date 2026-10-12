# pyre-check: max-pypy-ratio=12
# macOS reads 3.3x on dynasm and 7.0x on cranelift.
# Regression oracle: a builtin that calls back into Python -- a dict subscript
# into `__hash__` then `__eq__`, `in` over a list into `__eq__`, `int` into
# `__int__`, `str` into `__str__`, `map` into its function -- runs that
# callback exactly once per call, whatever the JIT does with the call that
# carries it.  When the
# guard behind the call fails, the frame goes on from behind the callback,
# never from in front of it.
#
# Every callback counts itself.  For the first SWITCH iterations it does
# nothing else, so the loops compile on the plain path; after that each
# iteration selects one kind of callback and one behaviour for it:
#   raise -- `Boom` leaves the callback and is caught by a `try` around the
#            builtin, in an inlined callee (`use_*`) or in the looping frame
#            itself (`same_frame`);
#   frame -- the callback reads a local out of the frame that called the
#            builtin, forcing it (`use_*` only).
#
# The five counters must equal what an interpreter counts, and the
# checksums over the results, the caught values and the forced locals match.
import sys

N = 3600
SWITCH = 1700

PLAIN, RAISE, FRAME = 0, 1, 2
K_HASH, K_EQ, K_INT, K_STR, K_MAP = 0, 1, 2, 3, 4

calls = [0, 0, 0, 0, 0]
seen = [0]
selected = [-1, PLAIN]


class Boom(Exception):
    pass


def hit(kind):
    calls[kind] += 1
    if selected[0] != kind:
        return
    mode = selected[1]
    if mode == RAISE:
        raise Boom(kind, calls[kind])
    if mode == FRAME:
        # hit <- the dunder <- the frame that called the builtin
        seen[0] += sys._getframe(2).f_locals['i'] * (kind + 1)


class Probe:
    def __init__(self, v):
        self.v = v

    def __hash__(self):
        hit(K_HASH)
        return self.v & 3

    def __eq__(self, other):
        hit(K_EQ)
        return self.v == other.v

    def __int__(self):
        hit(K_INT)
        return self.v + 20

    def __str__(self):
        hit(K_STR)
        return 'p' * (self.v + 1)


def mapped(x):
    hit(K_MAP)
    return x + 30


def pick(i, frame_ok):
    if i < SWITCH:
        selected[0] = -1
        selected[1] = PLAIN
        return
    j = i - SWITCH
    selected[0] = j % 5
    mode = (j // 5) % 3
    if mode == FRAME and not frame_ok:
        mode = RAISE
    selected[1] = mode


def caught(e):
    return e.args[0] * 1000 + e.args[1]


def use_subscript(table, k, i):
    try:
        return table[k]
    except Boom as e:
        return caught(e)


def use_contains(items, k, i):
    try:
        return k in items
    except Boom as e:
        return caught(e)


def use_int(k, i):
    try:
        return int(k)
    except Boom as e:
        return caught(e)


def use_str(k, i):
    try:
        return len(str(k))
    except Boom as e:
        return caught(e)


def use_map(k, i):
    try:
        return sum(map(mapped, (k.v, i & 1)))
    except Boom as e:
        return caught(e)


def fresh():
    # Four keys with four different hashes, so a subscript compares the probe
    # with exactly one stored key; three list items, the match last.
    selected[0] = -1
    selected[1] = PLAIN
    table = {Probe(0): 10, Probe(1): 11, Probe(2): 12, Probe(3): 13}
    calls[:] = [0, 0, 0, 0, 0]
    seen[0] = 0
    return table, [Probe(100), Probe(101), Probe(2)]


def through_callee(n):
    table, items = fresh()
    k = Probe(0)
    acc = 0
    for i in range(n):
        k.v = i & 3
        pick(i, True)
        acc += use_subscript(table, k, i) * 5
        acc += use_contains(items, k, i) * 7
        acc += use_int(k, i) * 11
        acc += use_str(k, i) * 13
        acc += use_map(k, i) * 17
    selected[0] = -1
    return acc


def same_frame(n):
    table, items = fresh()
    k = Probe(0)
    acc = 0
    for i in range(n):
        k.v = i & 3
        pick(i, False)
        try:
            acc += table[k] * 5
        except Boom as e:
            acc += caught(e)
        try:
            acc += (k in items) * 7
        except Boom as e:
            acc += caught(e)
        try:
            acc += int(k) * 11
        except Boom as e:
            acc += caught(e)
        try:
            acc += len(str(k)) * 13
        except Boom as e:
            acc += caught(e)
        try:
            acc += sum(map(mapped, (k.v, i & 1))) * 17
        except Boom as e:
            acc += caught(e)
    selected[0] = -1
    return acc


def report(label, fn):
    print(label, fn(N), calls, seen[0])


def main():
    report('callee', through_callee)
    report('same', same_frame)
    # A second pass enters the loops and bridges the first one compiled.
    report('callee', through_callee)
    report('same', same_frame)
    print('exc', sys.exc_info()[0])


main()
