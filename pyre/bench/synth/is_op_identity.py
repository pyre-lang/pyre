# `IS_OP` records `runtime_ops::is_op` (`ObjSpace.is_w`, then `w_bool_from`).
#
# No `max-pypy-ratio`: this fixture pins the answers, and most of its time is
# the `semantics` half.  `goto_if_not_same_box` carries the perf gate.
#
# A self-compare (`a is a`) answers at `is_w`'s opening `ptr::eq`, and
# `opimpl_ptr_eq` folds `b1 is b2` (`FASTPATHS_SAME_BOXES`, pyjitpl.py), so
# that arm records no compare and no guard.  Distinct boxes record the
# exact-type gates on `w_two` and then the fall-through.  Each never-taken
# arm adds a huge sentinel, so a wrong predicate balloons the checksum.
#
# `semantics` runs the classes whose `is_w` compares by value — int, float,
# complex, tuple, bytes, str, frozenset.  An `int` subclass shares `INT_TYPE`
# with a plain `int` and still answers by pointer identity; the exact-type
# gate in `W_AbstractIntObject.is_w` is what separates them.
N = 800000


class Plain:
    pass


def hot():
    a = Plain()
    b = Plain()
    acc = 0
    i = 0
    while i < N:
        if a is a:  # same box: constant True
            acc += 1
        if a is not a:  # same box: constant False
            acc += 1000000
        if a is b:  # distinct instances: ptr_eq False
            acc += 1000000
        if a is not b:
            acc += 2
        if a is None:  # NoneType keeps pointer identity
            acc += 1000000
        if a is not None:
            acc += 4
        if b is Plain:  # instance vs its type object
            acc += 1000000
        i += 1
    return acc


def semantics(n):
    # Every operand is built from `n` so the compiler cannot fold the pair to
    # one constant: these must be two runtime objects of equal value.
    out = []
    out.append((n * 0 + 1000) is (n * 0 + 1000))  # int: equal by value
    out.append((n * 0.0 + 1.5) is (n * 0.0 + 1.5))  # float: equal by bits
    out.append((n * 0.0 + 0.0) is -(n * 0.0 + 0.0))  # float: 0.0 is not -0.0
    out.append(complex(n * 0, n * 0) is complex(n * 0, n * 0))  # complex
    out.append(tuple(range(n * 0)) is tuple(range(n * 0)))  # empty tuple
    out.append(tuple(range(n * 0 + 2)) is tuple(range(n * 0 + 2)))  # non-empty
    out.append(bytes(n * 0) is bytes(n * 0))  # empty bytes
    out.append(("a" * (n * 0 + 1)) is ("a" * (n * 0 + 1)))  # 1-char str
    out.append(("ab" * (n * 0 + 1)) is ("ab" * (n * 0 + 1)))  # 2-char str
    out.append(frozenset(range(n * 0)) is frozenset(range(n * 0)))  # empty
    out.append(True is (n < 1))  # bool singletons
    return out


def main():
    print(hot())
    # The value-comparing classes answer `is` differently under CPython and
    # PyPy (`1000 is 1000` is False there, True here), so the answers
    # themselves are not printable in a fixture check.py oracles against both.
    # What IS engine-independent, and what a wrongly-widened fold breaks, is
    # that the compiled answer matches the interpreted one: `first` is the
    # very first, interpreted call and `again` comes after the loop is hot.
    first = semantics(0)
    for _ in range(2000):
        again = semantics(0)
    print(again == first, len(first))


main()
