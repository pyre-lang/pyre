# pyre-check: max-pypy-ratio=35
# BINARY_SUBSCR whose receiver resolves `__getitem__` to a Python function.
# Covers the two receiver shapes whose subscript the type's own MRO owns — a
# user instance and a builtin sequence subclass that overrides `__getitem__` —
# plus the three ways the resolution can change under a hot loop: a second
# receiver class flowing into the same site, an `__getitem__` reassigned on the
# class mid-loop, and an instance-dict `__getitem__` (which a special-method
# lookup must ignore). Deterministic.


class Seq:
    def __getitem__(self, i):
        if i >= 7:
            raise IndexError(i)
        return i * 2


class Neg:
    def __getitem__(self, i):
        return -i


class TupOverride(tuple):
    def __getitem__(self, i):
        return 1000 + tuple.__getitem__(self, i)


class Rebind:
    def __getitem__(self, i):
        return i


class Empty:
    pass


def hot_plain(o, n):
    acc = 0
    for _ in range(n):
        acc += o[3]
    return acc


def hot_caught(o, n):
    caught = 0
    for _ in range(n):
        try:
            o[9]
        except IndexError:
            caught += 1
    return caught


def hot_override(t, n):
    acc = 0
    i = 0
    while i < n:
        acc = t[1]
        i += 1
    return acc


def hot_polymorphic(a, b, n):
    acc = 0
    for i in range(n):
        o = a if i % 2 else b
        acc += o[3]
    return acc


def hot_rebind(o, n):
    acc = 0
    for i in range(n):
        if i == n // 2:
            Rebind.__getitem__ = lambda self, k: k + 100
        acc += o[2]
    return acc


def main():
    print("plain", hot_plain(Seq(), 60000))
    print("caught", hot_caught(Seq(), 60000))
    print("override", hot_override(TupOverride([10, 20, 30]), 60000))
    print("polymorphic", hot_polymorphic(Seq(), Neg(), 60000))
    print("rebind", hot_rebind(Rebind(), 60000))

    # A special method resolves on the type, so an instance-dict entry is not
    # consulted and the subscript still raises.
    bare = Neg()
    bare.__getitem__ = lambda i: 999
    print("instance_dict", bare[4])

    try:
        Empty()[0]
        print("not_subscriptable", "no_error")
    except TypeError:
        print("not_subscriptable", "typeerror")


main()
