# CPython-suite gap: exception tests omit this PyPy instance-dict storage path.
# parity-tests reason: it guards PyPy/pyre exception mapdict behavior.
#
# exception_user_layout:
# no CPython test checks that a slotted exception subclass keeps the slot out of
# `__dict__` while still raising, pickling, copying, and finalizing. typedef.py
# _getusercls / objspace.py allocate_instance on W_BaseException and the extra-field
# realbases.
#
# object_user_layout:
# no CPython test checks that exact object() and a user-class instance share attribute,
# pickle, copy, and weakref behavior while living on different interp-level layouts.
# typedef.py _getusercls(W_ObjectObject) / W_ObjectObjectUserDictWeakrefable versus
# W_ObjectObject for object().

"""A user attribute on an exception instance lives in its instance dict, which
is mapdict-backed.  Each block below runs the read hot enough to compile, then
perturbs the one thing that has to invalidate the compiled read.
"""

N = 3000


class E(Exception):
    pass


def read_sum(e):
    n = 0
    for _ in range(N):
        n += e.tag
    return n


def read_last(e):
    out = None
    for _ in range(N):
        out = e.tag
    return out


# An unboxed int slot, rewritten in place, then retyped so the slot migrates
# off unboxed storage.
e = E()
e.tag = 7
assert read_sum(e) == 7 * N
e.tag = 9
assert read_sum(e) == 9 * N
e.tag = "s"
assert read_last(e) == "s"

# A boxed slot, then a second attribute grows the map.
boxed = E()
boxed.tag = "seven"
assert read_last(boxed) == "seven"
boxed.other = 1
assert read_last(boxed) == "seven"

# Deleting the attribute.
gone = E()
gone.tag = 11
read_sum(gone)
del gone.tag
try:
    gone.tag
except AttributeError:
    pass
else:
    raise AssertionError("deleted attribute still readable")

# A second instance whose map places the attribute at another index, read at
# the same site.
shifted = E()
shifted.pad = 0
shifted.tag = 13
assert read_sum(shifted) == 13 * N
assert read_last(boxed) == "seven"

# Devolving the dictionary out from under the read.
devolved = E()
devolved.tag = 21
read_sum(devolved)
d = devolved.__dict__
for i in range(200):
    d["k%d" % i] = i
assert read_sum(devolved) == 21 * N
assert devolved.__dict__["tag"] == 21

# Writing through __dict__ rather than through the attribute.
written = E()
written.tag = 31
read_sum(written)
written.__dict__["tag"] = 32
assert read_sum(written) == 32 * N


# A class attribute of the same name: the instance attribute wins until it is
# deleted.
class F(E):
    pass


f = F()
f.tag = 41
assert read_sum(f) == 41 * N
F.tag = 99
assert read_sum(f) == 41 * N
del f.tag
assert read_sum(f) == 99 * N

# A data descriptor added to the class shadows the instance dictionary.
shadowed = E()
shadowed.tag = 3
assert read_sum(shadowed) == 3 * N
E.tag = property(lambda self: 77)
assert read_sum(shadowed) == 77 * N
del E.tag


# A non-exception receiver reaching the same read site.
class O:
    pass


o = O()
o.tag = 5
assert read_sum(o) == 5 * N

# The typed `args` slot is not a dictionary attribute.
assert E(1, 2).args == (1, 2)

def exception_user_layout():
    """Exception instances follow allocate_instance.

    An exact realbase stays on its interp class. Every other builtin exception
    and every app-level subclass uses that realbase's `_getusercls` layout.
    `__dict__` stays the typed dict; `__slots__` live on the map.
    """

    import copy
    import gc
    import pickle
    import weakref


    CASES = (
        ("Exception", Exception, ("m",), {}),
        ("ValueError", ValueError, ("m",), {}),
        ("KeyError", KeyError, ("k",), {}),
        ("OSError", OSError, (1, "x"), {}),
        ("PermissionError", PermissionError, (1, "x"), {}),
        ("ImportError", ImportError, ("m",), {"name": "mod"}),
        ("StopIteration", StopIteration, (5,), {}),
        ("UnicodeDecodeError", UnicodeDecodeError, ("utf-8", b"\xff", 0, 1, "bad"), {}),
        ("SystemExit", SystemExit, (2,), {}),
        # KeyboardInterrupt is not an Exception, so this stays a BaseExceptionGroup
        # (an all-Exception payload would promote to ExceptionGroup).
        ("BaseExceptionGroup", BaseExceptionGroup, ("g", [KeyboardInterrupt()]), {}),
    )


    def publish(cls):
        cls.__module__ = __name__
        globals()[cls.__name__] = cls
        return cls


    def construct(cls, args, kwargs):
        return cls(*args, **kwargs)


    def extra(e):
        out = []
        if isinstance(e, OSError):
            out.append(("errno", e.errno, e.strerror, e.filename))
        if isinstance(e, ImportError):
            out.append(("name", e.name))
        if isinstance(e, StopIteration):
            out.append(("value", e.value))
        if isinstance(e, SystemExit):
            out.append(("code", e.code))
        if isinstance(e, BaseExceptionGroup):
            out.append(
                (
                    "group",
                    e.message,
                    tuple(type(x).__name__ for x in e.exceptions),
                )
            )
        if isinstance(e, UnicodeDecodeError):
            out.append(("unicode", e.encoding, e.start, e.end, e.reason))
        return tuple(out)


    def caught(cls, args, kwargs, handler):
        try:
            raise construct(cls, args, kwargs)
        except handler as e:
            return (type(e).__name__, e.__traceback__ is not None)


    def exercise(tag, base, args, kwargs):
        plain = publish(type("P_" + tag, (base,), {}))
        slots = publish(type("S_" + tag, (base,), {"__slots__": ("a",)}))
        p = construct(plain, args, kwargs)
        p.x = 1
        p.add_note("n")
        s = construct(slots, args, kwargs)
        s.a = 1
        s.x = 2
        s.add_note("n")
        print(
            "plain",
            tag,
            type(p).__name__,
            isinstance(p, base),
            p.args,
            str(p),
            repr(p),
            sorted(p.__dict__),
            p.__notes__,
            extra(p),
        )
        print(
            "slots",
            tag,
            type(s).__name__,
            isinstance(s, base),
            s.args,
            str(s),
            repr(s),
            sorted(s.__dict__),
            "a" in s.__dict__,
            s.a,
            s.__notes__,
            extra(s),
        )
        print(
            "catch",
            tag,
            caught(plain, args, kwargs, plain),
            caught(plain, args, kwargs, base),
            caught(slots, args, kwargs, slots),
            caught(slots, args, kwargs, base),
        )
        try:
            try:
                raise ValueError("ctx")
            except ValueError:
                raise construct(plain, args, kwargs) from ValueError("cause")
        except plain as e:
            print(
                "chain",
                tag,
                None if e.__cause__ is None else type(e.__cause__).__name__,
                e.__context__ is not None,
                e.__traceback__ is not None,
                e.__suppress_context__,
            )
        try:
            try:
                raise ValueError("ctx")
            except ValueError:
                raise construct(plain, args, kwargs) from None
        except plain as e:
            print(
                "fromnone",
                tag,
                e.__cause__ is None,
                e.__context__ is not None,
                e.__suppress_context__,
            )
        fresh = construct(plain, args, kwargs)
        fresh.x = 3
        back = pickle.loads(pickle.dumps(fresh))
        print(
            "pickle",
            tag,
            type(back) is plain,
            back.args,
            sorted(back.__dict__),
            back.x,
            extra(back),
        )
        slotted = construct(slots, args, kwargs)
        slotted.a = 9
        slotted.x = 4
        back = pickle.loads(pickle.dumps(slotted))
        print(
            "spickle",
            tag,
            type(back) is slots,
            getattr(back, "a", None),
            back.x,
            sorted(back.__dict__),
        )
        copied = copy.copy(fresh)
        print(
            "copy",
            tag,
            type(copied) is plain,
            copied.args,
            sorted(copied.__dict__),
            copied is fresh,
            copied.x,
        )
        print("weak", tag, weakref.ref(p)() is p)


    def finalizer(label, base, args, kwargs):
        seen = []

        class D(base):
            def __del__(self):
                seen.append(label)

        def spawn():
            return weakref.ref(D(*args, **kwargs))

        ref = spawn()
        for _ in range(8):
            gc.collect()
            if ref() is None:
                break
        print("del", label, ref() is None, seen)


    for spec in CASES:
        exercise(*spec)

    finalizer("Exception", Exception, ("m",), {})
    finalizer("OSError", OSError, (1, "x"), {})
    finalizer("BaseExceptionGroup", BaseExceptionGroup, ("g", [KeyboardInterrupt()]), {})
exception_user_layout()

class A:
    pass

class B:
    __slots__ = ("a",)

class C(A):
    pass

def object_user_layout():
    """Exact object() and user-class instances.

    `object()` rejects attributes and `__class__` assignment. A plain class,
    a slotted class, and a subclass keep mapdict attributes, slots, pickle,
    copy, weakref, and `__del__`. The hot loop is what makes the JIT record
    the allocation.
    """

    import copy
    import gc
    import pickle
    import weakref


    def err(fn):
        try:
            fn()
            return "ok"
        except Exception as e:
            return type(e).__name__


    o = object()
    a = A()
    a.x = 1
    print("get", a.x)
    del a.x
    print("del", err(lambda: a.x), hasattr(a, "x"))
    a.x = 2
    print("dict", sorted(a.__dict__.items()))
    b = B()
    b.a = 3
    print("slot", b.a, err(lambda: b.__dict__), err(lambda: setattr(b, "z", 1)))
    print("types", type(o).__name__, type(a).__name__, type(b).__name__, type(C()).__name__)
    print(
        "isinstance",
        isinstance(o, object),
        isinstance(a, object),
        isinstance(a, A),
        isinstance(C(), A),
        isinstance(o, A),
    )
    a2 = A()
    a2.x = 4
    a2.__class__ = C
    print("reclass", type(a2).__name__, a2.x, isinstance(a2, A), isinstance(a2, C))
    a2.__class__ = A
    print("reclass2", type(a2).__name__, a2.x)
    print("obj-attr", err(lambda: setattr(o, "x", 1)))
    print("obj-class", err(lambda: setattr(o, "__class__", A)))
    print("eq", o == o, o == object(), a == a, a == A())
    print("hash", type(hash(o)).__name__, hash(a) == hash(a))


    def rshape(s):
        head, sep, tail = s.partition(" object at ")
        return (head, bool(sep), tail.startswith("0x"), tail.endswith(">"))


    print("repr", rshape(repr(o)), rshape(repr(a)), rshape(repr(b)))
    pa = pickle.loads(pickle.dumps(a))
    print("pickle", type(pa).__name__, pa.x)
    pb = pickle.loads(pickle.dumps(b))
    print("pickle-slot", type(pb).__name__, pb.a)
    ca = copy.copy(a)
    print("copy", type(ca).__name__, ca.x, ca is a)
    da = copy.deepcopy(a)
    print("deepcopy", type(da).__name__, da.x, da is a)
    alive = A()
    ref = weakref.ref(alive)
    print("weak", ref() is alive)
    print("weak-obj", err(lambda: weakref.ref(object())))
    seen = []


    class D:
        def __del__(self):
            seen.append("d")


    d = D()
    del d
    gc.collect()
    print("del", seen)


    def hot(n):
        total = 0
        i = 0
        while i < n:
            x = A()
            x.k = i
            total += x.k
            i += 1
        return total


    print("hot", hot(2000))
object_user_layout()
print("OK")
