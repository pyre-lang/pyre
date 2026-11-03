# pyre-check: requires-modules=math
# pyre-check: jitstats-band=guard_failures=6
# Under check.py's child environment guard_failures moves with where minor
# collections land and the loop and bridge counts do not: on the previous
# tree windows CI dynasm read 3214 against the macOS baseline 3208 (run
# 41387529148) at the same 24 loops / 5 bridges. One macOS arm64 dynasm
# binary reads 1213 under check.py, and 1215 or 1219 when run directly at
# every PYPY_GC_NURSERY from 1M to 32M.
# type.__call__ inlined for a class whose __new__ is a Python function and
# whose __init__ is object.__init__.
#
# typeobject.py descr_call / _type_call: the call result is the __new__
# result when type(result).__init__ is object.__init__ (surplus args are a
# no-op once __new__ is overridden).  An earlier emit of this shape compiled
# and fired, but a loop `x = NewOv(i)` then used the instance as the next
# iteration's callable (`TypeError: 'NewOv' object is not callable`).  The
# cases below are that loop, the same loop storing into a list, a __new__
# that returns a different class, a __new__ that raises, Decimal
# construction, an unjournaled `__new__` that sometimes returns an unrelated
# class, and a `__new__` that returns a subclass with its own `__init__`.
import decimal
import io
import itertools

N = 20000


class NewOv:
    def __new__(cls, v):
        o = object.__new__(cls)
        o.v = v
        return o


class Other:
    pass


class DiffNew:
    def __new__(cls, v):
        o = Other()
        o.v = v
        return o


class RaiseNew:
    def __new__(cls, v):
        if v == 15000:
            raise ValueError(v)
        o = object.__new__(cls)
        o.v = v
        return o


def newov(n):
    acc = 0
    for i in range(n):
        x = NewOv(i)
        acc += x.v
    return acc


def newov_list(n):
    xs = []
    for i in range(n):
        xs.append(NewOv(i))
    acc = 0
    for x in xs:
        acc += x.v
    return acc


def diffnew(n):
    acc = 0
    last = None
    for i in range(n):
        last = DiffNew(i)
        acc += last.v
    return acc, type(last).__name__


def raise_new(n):
    acc = 0
    caught = None
    for i in range(n):
        try:
            x = RaiseNew(i)
            acc += x.v
        except ValueError as e:
            caught = e.args[0]
    return acc, caught


def decimal_mul(n):
    total = 0
    for i in range(n):
        total += decimal.Decimal(i) * decimal.Decimal(3)
    return total


print("newov", newov(N))
print("newov_list", newov_list(N))
acc, name = diffnew(N)
print("diffnew", acc, name)
acc, caught = raise_new(N)
print("raise", acc, caught)
print("decimal", decimal_mul(N))

ctr = itertools.count()
buf = io.StringIO()


class DupA:
    def __new__(cls, v):
        next(ctr)
        buf.write("x")
        if v % 4 == 0:
            return Other()
        return object.__new__(cls)


class DupB:
    def __new__(cls, v):
        next(ctr)
        if v % 7 == 0:
            return object.__new__(DupBSub)
        return object.__new__(cls)


class DupBSub(DupB):
    def __init__(self, v):
        self.v = v


def run():
    n = 0
    for i in range(30000):
        n += type(DupA(i)) is Other
    s = 0
    for i in range(30000):
        y = DupB(i)
        s += getattr(y, "v", 0)
    return n, s


print(run(), next(ctr), len(buf.getvalue()))


class CellBase:
    __init__ = object.__init__


CellBase.__init__ = object.__init__


class CellNew(CellBase):
    def __new__(cls, v):
        return object.__new__(cls)


cell_log = []


def cell_init(self, v):
    cell_log.append(v)


def cell_run():
    for i in range(20000):
        if i == 15000:
            CellBase.__init__ = cell_init
        CellNew(i)
    return len(cell_log), cell_log[0]


print("cell", cell_run())


class VarNewBase:
    def __new__(*args):
        return object.__new__(VarNewSub)


class VarNewSub(VarNewBase):
    def __init__(self, *args):
        self.inited = len(args)


def varargs_new(n):
    total = 0
    for i in range(n):
        total += VarNewBase(i, i).inited
    return total


print("varargs_new", varargs_new(N))
