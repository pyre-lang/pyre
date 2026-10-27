# pyre-check: max-pypy-ratio=100
# LOAD_SPECIAL `__enter__`/`__exit__` and LOAD_ATTR method lookup on a user
# instance must pin the map (`mapdict.py BaseUserClassMapdict.getclass` /
# `MapdictStorageMixin._get_mapdict_map`). GuardClass alone is the shared
# `INSTANCE_USER_TYPE` payload and would reuse the wrong constant descriptor
# across subclasses. Exact output is the checksum of which methods ran.
try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

WARM = 64
N = 72000


class A:
    def __enter__(self):
        self.n = 1
        return self

    def __exit__(self, exc_type, exc, tb):
        self.n += 10
        return False


class B:
    def __enter__(self):
        self.n = 2
        return self

    def __exit__(self, exc_type, exc, tb):
        self.n += 20
        return False


def alternating_with(n):
    a = A()
    b = B()
    total = 0
    i = 0
    while i < WARM:
        with a:
            pass
        total += a.n
        i += 1
    i = 0
    while i < n:
        x = a if (i & 1) == 0 else b
        with x:
            pass
        total += x.n
        i += 1
    return total


def class_reassign_with(n):
    x = A()
    total = 0
    i = 0
    while i < n:
        with x:
            pass
        total += x.n
        x.__class__ = B if x.__class__ is A else A
        i += 1
    return total


class DA:
    def m(self):
        return 3


class DB:
    def m(self):
        return 4


def _devolve(obj):
    i = 0
    while i < 90:
        setattr(obj, "a%d" % i, i)
        i += 1


def devolved_load_attr(n):
    a = DA()
    b = DB()
    _devolve(a)
    _devolve(b)
    total = 0
    i = 0
    while i < n:
        obj = a if (i & 1) == 0 else b
        total += obj.m()
        i += 1
    return total


print(alternating_with(N) + class_reassign_with(N) + devolved_load_attr(N))
