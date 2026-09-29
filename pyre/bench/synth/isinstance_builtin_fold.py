# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=hot_hit,hot_miss,hot_subclasses,hot_polymorphic,hot_bases,hot_late_base
# pyre-check: trace-shape=hot_hit:absent=CallMayForceR
# Self-checking guard for `isinstance(obj, cls)` answered out of the trace.
#
# `issubtype` (typeobject.py) promotes both types and keys an elidable answer
# on both version tags, so a hot call needs a class guard on the receiver plus
# the two version-tag pins and the answer is then a constant.  Everything the
# constant depends on has to be pinned, and every shape whose answer is not a
# function of those two types has to keep the call:
#
#   hot_hit / hot_miss / hot_subclasses  the arms that fold, over a builtin
#                                        second argument, a base, a subclass
#                                        of a builtin, and a bool/int pair
#   hot_polymorphic                      the receiver class cycles, so an
#                                        answer baked under one class guard
#                                        must side-exit for the others
#   hot_bases / hot_late_base            `__bases__` reassigned mid-loop in
#                                        both directions; `mutated()` walks
#                                        the subclasses and has to revoke a
#                                        baked answer
#   hot_tuple                            a tuple `classinfo` is rebuilt at
#                                        every call, so there is no second
#                                        argument to pin
#   hot_class_property                   the miss arm reads `__class__`, user
#                                        code here, and answers True for a
#                                        receiver whose real type is unrelated
#   hot_abc_register                     `__instancecheck__` on the metaclass,
#                                        with a `register` mid-loop that the
#                                        next iteration has to see
#   hot_fresh_class                      a class per iteration, so a pinned
#                                        `classinfo` must not answer for the
#                                        next one
import abc
import sys

try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

N = 20000
K = 2000


class A:
    pass


class Mid:
    pass


class B(A):
    pass


class MyInt(int):
    pass


def hot_hit(n):
    b = B()
    x = 7
    a = 0
    i = 0
    while i < n:
        if isinstance(x, int):
            a += 1
        if isinstance(b, A):
            a += 1
        i += 1
    return a


def hot_miss(n):
    x = 7
    a = 0
    i = 0
    while i < n:
        if isinstance(x, str):
            a += 1000
        else:
            a += 1
        i += 1
    return a


def hot_subclasses(n):
    x = MyInt(3)
    a = 0
    i = 0
    while i < n:
        a += isinstance(True, int)
        a += isinstance(x, int)
        a += isinstance(5, bool)
        a += isinstance(x, MyInt)
        i += 1
    return a


def hot_polymorphic(n):
    objs = [A(), B(), object(), 7, "s"]
    a = 0
    i = 0
    while i < n:
        if isinstance(objs[i % 5], A):
            a += 1
        i += 1
    return a


def hot_bases(n):
    b = B()
    a = 0
    i = 0
    while i < n:
        if isinstance(b, A):
            a += 1
        if i == n // 2:
            B.__bases__ = (Mid,)
        i += 1
    B.__bases__ = (A,)
    return a


class Late(Mid):
    pass


def hot_late_base(n):
    x = Late()
    a = 0
    i = 0
    while i < n:
        if isinstance(x, A):
            a += 1
        if i == n // 2:
            Late.__bases__ = (A,)
        i += 1
    Late.__bases__ = (Mid,)
    return a


def hot_tuple(n):
    x = 7
    a = 0
    i = 0
    while i < n:
        if isinstance(x, (str, int)):
            a += 1
        i += 1
    return a


class ClassProp:
    @property
    def __class__(self):
        return A


def hot_class_property(n):
    p = ClassProp()
    a = 0
    i = 0
    while i < n:
        if isinstance(p, A):
            a += 1
        i += 1
    return a


class Reg(metaclass=abc.ABCMeta):
    pass


class Plain:
    pass


def hot_abc_register(n):
    p = Plain()
    a = 0
    i = 0
    while i < n:
        if isinstance(p, Reg):
            a += 1
        if i == n // 2:
            Reg.register(Plain)
        i += 1
    return a


def hot_fresh_class(n):
    a = 0
    i = 0
    while i < n:
        class Fresh:
            pass

        if isinstance(Fresh(), Fresh):
            a += 1
        if isinstance(A(), Fresh):
            a += 100
        i += 1
    return a


def main():
    half = N // 2
    for label, got, want in (
        ("hit", hot_hit(N), 2 * N),
        ("miss", hot_miss(N), N),
        ("subclasses", hot_subclasses(N), 3 * N),
        ("polymorphic", hot_polymorphic(N), 2 * (N // 5)),
        ("bases", hot_bases(N), half + 1),
        ("late_base", hot_late_base(N), N - half - 1),
        ("tuple", hot_tuple(N), N),
        ("class_property", hot_class_property(N), N),
        ("abc_register", hot_abc_register(N), N - half - 1),
        ("fresh_class", hot_fresh_class(K), K),
    ):
        if got != want:
            print(f"FAIL {label}: {got} != {want}")
            return 1
    print("PASS isinstance fold")
    return 0


sys.exit(main())
