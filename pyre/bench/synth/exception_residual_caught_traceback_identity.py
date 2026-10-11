# Regression oracle: an exception raised inside a builtin (a residual call in
# the trace) and caught in the same trace is one object with one history.
# The handler's `e`, `sys.exc_info()[1]`, the value `__exit__` receives, the
# instance a bare `raise` hands to the caller and the one a generator sees
# from `throw` are the same instance; its `__traceback__` names the frames it
# crossed, innermost last, at the lines that raised; `__context__`,
# `__cause__` and `__suppress_context__` record what was being handled.
#
# Each scenario runs N times and returns a number folded into a checksum plus
# the observables themselves; the last iteration's are printed.  Line numbers
# are reported relative to the function that owns the frame.  `deep` changes
# how many frames the exception crosses after SWITCH iterations, so the
# guards on the traceback's shape fail once the loop is compiled.
import sys

N = 2400
SWITCH = 1500


class Boom(Exception):
    pass


def chain(e):
    out = []
    tb = e.__traceback__
    while tb is not None:
        code = tb.tb_frame.f_code
        out.append((code.co_name, tb.tb_lineno - code.co_firstlineno))
        tb = tb.tb_next
    return tuple(out)


def depth(e):
    n = 0
    tb = e.__traceback__
    while tb is not None:
        n += 1
        tb = tb.tb_next
    return n


def leaf(i):
    return int('t' * (1 + (i & 1)))


def middle(i):
    return leaf(i) + 1


def traceback_chain(i):
    try:
        middle(i)
    except ValueError as e:
        return depth(e), chain(e)
    return -1, None


def deep(i):
    try:
        if i < SWITCH or i & 1:
            middle(i)
        else:
            {}[i & 3]
    except (ValueError, KeyError) as e:
        return depth(e), (type(e).__name__, chain(e))
    return -1, None


def identity(i):
    try:
        [].pop()
    except IndexError as e:
        info = sys.exc_info()
        num = (e is info[1]) + (info[2] is e.__traceback__) * 2 + (type(e) is info[0]) * 4
        return num, (num, chain(e))
    return -1, None


def reraise_inner(i, seen):
    try:
        {}[i & 7]
    except KeyError as e:
        seen.append(e)
        raise


def reraise(i):
    seen = []
    try:
        reraise_inner(i, seen)
    except LookupError as e:
        num = (e is seen[0]) + depth(e) * 2
        return num, (num, e.args, chain(e))
    return -1, None


def context_implicit(i):
    try:
        try:
            int('c')
        except ValueError as handled:
            first = handled
            {}[i & 1]
    except KeyError as e:
        ctx = e.__context__
        num = (ctx is first) + (e.__cause__ is None) * 2 + e.__suppress_context__ * 4 + (ctx.__context__ is None) * 8
        return num, (num, type(ctx).__name__, e.args, chain(e), chain(ctx))
    return -1, None


def context_explicit(i):
    try:
        try:
            [].pop()
        except IndexError as handled:
            first = handled
            raise Boom(i & 3) from first
    except Boom as e:
        num = (e.__cause__ is first) + (e.__context__ is first) * 2 + e.__suppress_context__ * 4
        cause = (num, type(e.__cause__).__name__, e.args)
    try:
        try:
            int('n')
        except ValueError:
            raise Boom from None
    except Boom as e:
        num += (e.__cause__ is None) * 8 + (type(e.__context__) is ValueError) * 16 + e.__suppress_context__ * 32
        return num, (cause, num, e.args)
    return -1, None


class Manager:
    def __init__(self):
        self.seen = None

    def __enter__(self):
        return self

    def __exit__(self, typ, val, tb):
        self.seen = (typ, val, tb)
        return typ is KeyError


def with_exit(i):
    m = Manager()
    with m:
        {}[i & 3]
    typ, val, tb = m.seen
    num = (typ is KeyError) + (type(val) is KeyError) * 2 + (val.__traceback__ is tb) * 4 + (tb.tb_next is None) * 8
    try:
        with m:
            int('w')
    except ValueError as e:
        num += (m.seen[1] is e) * 16 + (m.seen[0] is ValueError) * 32
        return num, (num, val.args, chain(val), chain(e), sys.exc_info()[1] is e)
    return -1, None


def catcher(log):
    while True:
        try:
            yield len(log)
        except ValueError as e:
            log.append(e)
        except GeneratorExit as e:
            log.append(type(e).__name__)
            raise


def raiser():
    try:
        yield 1
    finally:
        int('g')


def generator_paths(i):
    log = []
    g = catcher(log)
    next(g)
    try:
        int('y' * (1 + (i & 1)))
    except ValueError as handled:
        thrown = handled
        got = g.throw(thrown)
    num = got + (log[0] is thrown) * 2 + depth(thrown) * 4
    closed = []
    closing = catcher(closed)
    next(closing)
    closing.close()
    num += (closed[0] == 'GeneratorExit') * 32
    r = raiser()
    next(r)
    try:
        r.close()
    except ValueError as e:
        num += depth(e) * 64
        return num, (num, thrown.args, chain(thrown), chain(e), type(e.__context__).__name__)
    return -1, None


def after_handler(i):
    try:
        try:
            int('a')
        except ValueError as handled:
            inner = handled
            inside = sys.exc_info()[1] is inner
            raise Boom(i & 1)
    except Boom as e:
        outer = sys.exc_info()[1] is e
        kept = e.__context__ is inner
    gone = sys.exc_info() == (None, None, None)
    num = inside + outer * 2 + kept * 4 + gone * 8
    return num, (num,)


SCENARIOS = (
    traceback_chain, deep, identity, reraise, context_implicit,
    context_explicit, with_exit, generator_paths, after_handler,
)


def drive(fn, n):
    acc = 0
    last = None
    for i in range(n):
        num, last = fn(i)
        acc += num
    return acc, last


def main():
    for fn in SCENARIOS:
        acc, last = drive(fn, N)
        print(fn.__name__, acc, last)
    print('exc', sys.exc_info())


main()
