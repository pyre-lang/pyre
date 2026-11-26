# pyre-check: requires-modules=math
# pyre-check: jitstats-band=guard_failures=6
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

# Classes, maps and bound methods used as jitcounter keys live here, before
# the allocating loops. A first attribute store (or first type.__call__)
# mid-burst allocates a young map whose address hashes ResumeGuardDescr.
# must_compile; a minor collection then splits the counter and a function-
# entry trace races a loop trace. Take every transition once up front.


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


class CellBase:
    __init__ = object.__init__


CellBase.__init__ = object.__init__


class CellNew(CellBase):
    def __new__(cls, v):
        return object.__new__(cls)


cell_log = []


def cell_init(self, v):
    cell_log.append(v)


class VarNewBase:
    def __new__(*args):
        return object.__new__(VarNewSub)


class VarNewSub(VarNewBase):
    def __init__(self, *args):
        self.inited = len(args)


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


def run():
    n = 0
    for i in range(30000):
        n += type(DupA(i)) is Other
    s = 0
    for i in range(30000):
        y = DupB(i)
        s += getattr(y, "v", 0)
    return n, s


def cell_run():
    for i in range(20000):
        if i == 15000:
            CellBase.__init__ = cell_init
        CellNew(i)
    return len(cell_log), cell_log[0]


def varargs_new(n):
    total = 0
    for i in range(n):
        total += VarNewBase(i, i).inited
    return total


_keep = []
_keep.append(NewOv(0))
_other_empty = Other()
_keep.append(_other_empty)
_other_v = Other()
_other_v.v = 0
_keep.append(_other_v)
_keep.append(DiffNew(0))
_keep.append(RaiseNew(0))
_keep.append(decimal.Decimal(0))
_keep.append(decimal.Decimal(3))
_keep.append(_keep[-2] * _keep[-1])
_keep.append(object.__new__)
_keep.append(object.__init__)
_keep.append(type)
_keep.append(getattr)
_keep.append(buf.write)
_keep.append(next)
_keep.append(cell_log.append)
_keep.append(DupA(1))
_keep.append(DupA(0))
_keep.append(DupB(1))
_keep.append(DupB(0))
_keep.append(CellNew(0))
_keep.append(VarNewBase(0, 0))
ctr = itertools.count()
buf = io.StringIO()


print("newov", newov(N))
print("newov_list", newov_list(N))
acc, name = diffnew(N)
print("diffnew", acc, name)
acc, caught = raise_new(N)
print("raise", acc, caught)
print("decimal", decimal_mul(N))
print(run(), next(ctr), len(buf.getvalue()))
print("cell", cell_run())
print("varargs_new", varargs_new(N))
