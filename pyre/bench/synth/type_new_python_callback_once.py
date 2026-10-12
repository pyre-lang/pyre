# pyre-check: max-pypy-ratio=10
# macOS reads 3.1x on dynasm and 4.8x on cranelift.
# Regression oracle: `type(name, bases, ns)` re-enters Python from inside
# `type.__call__` / `type.__new__` -- a metaclass `__new__` and `__init__`, the
# base's `__init_subclass__`, a descriptor's `__set_name__`.  Each of those runs
# exactly once per `type(...)` call, whatever the JIT does with the call that
# carries it: when the guard after that call fails, the frame goes on from
# behind the callback, never from in front of it.
#
# Every callback counts itself.  For the first SWITCH iterations they do
# nothing else, so the loop compiles on the plain path; after that each
# iteration picks one callback and one behaviour for it:
#   raise  -- `Boom` leaves the callback and is caught by a `try` in the frame
#             that made the call (`same_frame`) or in an inlined callee
#             (`build` under `through_callee`);
#   frame  -- the callback reads a loop local out of the calling frame's
#             `f_locals`, forcing that frame;
#   tb     -- as `raise`, and the handler walks `e.__traceback__`.
# `__set_name__` never raises here: 3.11 wraps that in a RuntimeError and 3.12
# stopped, and this fixture pins nothing the two disagree on.
#
# The `frame` behaviour is exercised from an inlined callee only: `same_frame`
# takes the raising behaviours.
#
# The four counters must equal what an interpreter counts, and the
# checksums over the caught values, the forced locals and the traceback shape
# match.
import sys

N = 3600
SWITCH = 1700

PLAIN, RAISE, FRAME, TB = 0, 1, 2, 3
W_NEW, W_INIT, W_SUBCLASS, W_SETNAME = 0, 1, 2, 3

calls = [0, 0, 0, 0]
seen = [0]


class Boom(Exception):
    pass


def act(who, selected, mode):
    # Shared tail of every callback: count, then misbehave when selected.
    calls[who] += 1
    if selected != who:
        return
    if mode == FRAME:
        # The frame that wrote the `type(...)` call: `__set_name__` and
        # `__init_subclass__` run below `Meta.__new__`, so walk to it by name.
        frame = sys._getframe(1)
        while frame.f_code.co_name != 'build':
            frame = frame.f_back
        seen[0] += frame.f_locals['i'] * (who + 1)
    elif mode == RAISE or mode == TB:
        raise Boom(who, calls[who])


class Meta(type):
    def __new__(mcs, name, bases, ns):
        cls = super().__new__(mcs, name, bases, ns)
        act(W_NEW, ns['who'], ns['mode'])
        return cls

    def __init__(cls, name, bases, ns):
        super().__init__(name, bases, ns)
        act(W_INIT, ns['who'], ns['mode'])


class Base(metaclass=Meta):
    who = -1
    mode = PLAIN

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        act(W_SUBCLASS, cls.who, cls.mode)


class Named:
    def __init__(self, who, mode):
        self.who = who
        self.mode = mode
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name
        act(W_SETNAME, self.who, self.mode)


def pick(i):
    if i < SWITCH:
        return -1, PLAIN
    j = i - SWITCH
    who = j & 3
    mode = (j >> 2) % 4
    if who == W_SETNAME and (mode == RAISE or mode == TB):
        mode = FRAME
    return who, mode


def tb_shape(e):
    # depth, and for each level its function name and the line offset inside it
    tb = e.__traceback__
    shape = 0
    depth = 0
    while tb is not None:
        code = tb.tb_frame.f_code
        shape = shape * 31 + len(code.co_name) * 100 + (tb.tb_lineno - code.co_firstlineno)
        depth += 1
        tb = tb.tb_next
    return depth * 1000000 + shape % 1000000


def build(i, who, mode):
    try:
        cls = type('C', (Base,), {'who': who, 'mode': mode, 'slot': Named(who, mode)})
    except Boom as e:
        r = e.args[0] * 1000 + e.args[1]
        if mode == TB:
            r += tb_shape(e)
        return r
    return len(cls.slot.name) + (cls.who == who)


def through_callee(n):
    acc = 0
    for i in range(n):
        who, mode = pick(i)
        acc += build(i, who, mode)
    return acc


def same_frame(n):
    acc = 0
    for i in range(n):
        who, mode = pick(i)
        if mode == FRAME:
            mode = TB if who != W_SETNAME else PLAIN
        try:
            cls = type('C', (Base,), {'who': who, 'mode': mode, 'slot': Named(who, mode)})
            acc += len(cls.slot.name) + (cls.who == who)
        except Boom as e:
            acc += e.args[0] * 1000 + e.args[1]
            if mode == TB:
                acc += tb_shape(e)
    return acc


class Root:
    # No metaclass in play: `type` itself runs the hook.
    hits = [0, 0]

    def __init_subclass__(cls, **kwargs):
        Root.hits[0] += 1
        if cls.trip == RAISE:
            raise Boom(Root.hits[0])
        if cls.trip == FRAME:
            Root.hits[1] += sys._getframe(1).f_locals['i']


def build_plain(i, trip):
    try:
        return type('D', (Root,), {'trip': trip}).trip
    except Boom as e:
        return e.args[0]


def plain_type(n):
    acc = 0
    for i in range(n):
        acc += build_plain(i, PLAIN if i < SWITCH else i % 3)
    return acc, Root.hits


def report(label, fn):
    calls[:] = [0, 0, 0, 0]
    seen[0] = 0
    print(label, fn(N), calls, seen[0])


def main():
    report('callee', through_callee)
    report('same', same_frame)
    # A second pass enters the loops and bridges the first one compiled.
    report('callee', through_callee)
    report('same', same_frame)
    print('plain', plain_type(N))
    print('exc', sys.exc_info()[0])


main()
