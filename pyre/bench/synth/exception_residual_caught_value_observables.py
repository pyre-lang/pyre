# Regression oracle: an exception raised inside a builtin (a residual call in
# the trace) and caught in the same trace shows the handler the exception an
# interpreter would have built -- its class, `args`, the kind-specific
# attributes, `str` and `repr` -- and matches `except` clauses the same way,
# however late the JIT gets round to building the instance.
#
# Sources: `type('\udcff', (), {})` (the name fails to encode), `int('x')`,
# a missing dict key, `[].pop()`, an exhausted iterator and a generator's
# return value, `open` of a missing path, a user function raising from inside
# `sorted(key=...)` and `map`, `raise Cls` and `raise Cls(args)`.
# Handlers: no binding at all, `as e`, a tuple of classes, a base class, a
# non-matching clause in the callee with the matching one in the caller, and
# `sys.exc_info()` inside the handler and behind it.
#
# Each scenario runs N times and returns a number folded into a checksum plus
# the observables themselves; the last iteration's are printed.  `mixed`
# changes its source after SWITCH iterations, so the guards on the exception
# class fail once the loop is compiled.
import sys

N = 2400
SWITCH = 1500
MISSING = '/nonexistent-pyre-fixture/missing'


class Boom(Exception):
    pass


class Loud(Boom):
    def __str__(self):
        return 'loud:' + ','.join(str(a) for a in self.args)


def surrogate(i):
    name = '\udcff' if i & 1 else 'ok\udc80x'
    try:
        type(name, (), {})
    except UnicodeEncodeError as e:
        num = e.start * 3 + e.end + len(e.object)
        return num, (type(e).__name__, e.object, e.start, e.end, e.reason, len(e.args), e.args[1:])
    return -1, None


def int_literal(i):
    text = 'x' * (1 + (i & 1))
    try:
        int(text)
    except ValueError as e:
        return len(str(e)), (type(e).__name__, e.args, str(e), repr(e))
    return -1, None


def missing_key(i):
    key = ('k', i & 3)
    try:
        {}[key]
    except LookupError as e:
        num = (type(e) is KeyError) + (e.args[0] is key) * 2 + len(e.args) * 4
        return num, (type(e).__name__, e.args, str(e), repr(e))
    return -1, None


def empty_pop(i):
    try:
        [].pop()
    except (KeyError, IndexError) as e:
        return len(str(e)), (type(e).__name__, e.args, str(e), repr(e))
    return -1, None


def returning(value):
    return value
    yield


def stop_iteration(i):
    try:
        next(iter(()))
    except StopIteration as e:
        bare = (e.value, e.args)
    try:
        next(returning(i & 7))
    except StopIteration as e:
        return e.value + len(e.args), (bare, e.value, e.args, repr(e))
    return -1, None


def missing_file(i):
    try:
        open(MISSING)
    except OSError as e:
        num = e.errno + len(e.args) + (e.filename == MISSING) * 10
        return num, (type(e).__name__, e.errno, e.strerror, e.filename, e.filename2, e.args, str(e))
    return -1, None


def bad_key(x):
    raise Loud(x, 'key')


def through_builtin(i):
    try:
        sorted((i & 3, 9), key=bad_key)
    except Boom as e:
        first = (type(e).__name__, e.args, str(e))
    try:
        list(map(bad_key, (i & 1,)))
    except Loud as e:
        return first[1][0] + e.args[0], (first, type(e).__name__, e.args, repr(e))
    return -1, None


def raise_forms(i):
    try:
        raise Boom
    except Boom as e:
        cls_form = (type(e).__name__, e.args, str(e), repr(e))
    try:
        raise Loud(i & 3, 'inst')
    except Boom as e:
        return len(cls_form[1]) + e.args[0], (cls_form, type(e).__name__, e.args, str(e))
    return -1, None


def unbound_handler(i):
    hits = 0
    try:
        int('z')
    except ValueError:
        hits += 1
    try:
        {}[i]
    except KeyError:
        hits += 2
    try:
        [].pop()
    except Exception:
        hits += 4
    return hits, (hits, sys.exc_info()[0])


def inner_no_match(i):
    try:
        return int('q' * (1 + (i & 1)))
    except (KeyError, IndexError):
        return -5


def caller_matches(i):
    try:
        inner_no_match(i)
    except ArithmeticError:
        return -2, None
    except ValueError as e:
        return len(e.args[0]), (type(e).__name__, e.args)
    return -1, None


def exc_info_views(i):
    before = sys.exc_info()[0]
    try:
        {}['info']
    except KeyError as e:
        inside = sys.exc_info()
        same = (inside[0] is KeyError) + (inside[1] is e) * 2 + (inside[2] is e.__traceback__) * 4
    after = sys.exc_info()
    return same, (before, same, after)


def mixed(i):
    which = 0 if i < SWITCH else i % 4
    try:
        if which == 0:
            int('m')
        elif which == 1:
            {}['m']
        elif which == 2:
            [].pop()
        else:
            raise Loud(i & 1)
    except (ValueError, LookupError) as e:
        return which + len(e.args), (type(e).__name__, e.args, str(e))
    except Boom as e:
        return 100 + e.args[0], (type(e).__name__, e.args, str(e))
    return -1, None


SCENARIOS = (
    surrogate, int_literal, missing_key, empty_pop, stop_iteration,
    missing_file, through_builtin, raise_forms, unbound_handler,
    caller_matches, exc_info_views, mixed,
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
