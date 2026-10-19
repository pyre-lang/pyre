# CPython-suite gap: attribute tests omit PyPy MapDict layouts on builtin subclasses.
# parity-tests reason: this directly targets PyPy/pyre builtin-subclass storage.
#
# builtin_iterator_subclass_mapdict:
# no test stores attributes, slots, weakrefs or __del__ on
# enumerate/map/filter/zip/reversed/super/property subclasses. this targets the typedef.py
# _getusercls mapdict storage on those iterator and descriptor layouts.
#
# builtin_itertools_subclass_mapdict:
# no test stores attributes, slots, weakrefs or __del__ on itertools subclasses. this
# targets the typedef.py _getusercls mapdict storage on those layouts.
#
# builtin_subclass_mapdict_c5:
# subclass instances of deque, Struct, Pickler, Unpickler, GenericAlias and a big int
# still stored __dict__ beside the builtin typeptr. typedef.py _getusercls mapdict storage
# on those layouts.
#
# builtin_subclass_slots_mapdict:
# no test stores __slots__ members on
# float/complex/str/bytearray/list/set/frozenset/array/weakref subclasses under a hot
# loop. this targets the typedef.py _getusercls mapdict storage behind slot members.
#
# io_subclass_mapdict:
# no CPython test stores attributes, slots, weakrefs or __del__ on
# BytesIO/StringIO/Buffered*/TextIOWrapper subclass instances. typedef.py _getusercls
# mapdict storage on those layouts.

import gc


class T(tuple):
    x = "class"


class I(int):
    x = "class"


class S(str):
    x = "class"


class DataDescriptor:
    def __get__(self, obj, owner):
        return "data"

    def __set__(self, obj, value):
        pass


class TD(tuple):
    d = DataDescriptor()


class ID(int):
    d = DataDescriptor()


class SD(str):
    d = DataDescriptor()


shadowed = (T((1, 2)), I(5), S("hi"))
for obj in shadowed:
    obj.x = "instance"
assert tuple(obj.x for obj in shadowed) == ("instance", "instance", "instance")

described = (TD((1,)), ID(5), SD("hi"))
for obj in described:
    obj.__dict__["d"] = "instance"
assert tuple(obj.d for obj in described) == ("data", "data", "data")

print(tuple(obj.x for obj in shadowed))
print(tuple(obj.d for obj in described))


def load_mark(values):
    total = 0
    for value in values:
        total += value.mark
    return total


def store_mark(values):
    for value in values:
        value.mark = 19


int_user = I(5)
str_user = S("user")
tuple_user = T((3, 4))
for value in (int_user, str_user, tuple_user):
    value.mark = 7

exact_values = (5, "interned_mapdict_exact", (1, 2))
for user, exact in zip((int_user, str_user, tuple_user), exact_values):
    try:
        load_mark([user] * 4000 + [exact])
    except AttributeError:
        pass
    else:
        raise AssertionError(type(user).__name__)
    try:
        store_mark([user] * 4000 + [exact])
    except AttributeError:
        pass
    else:
        raise AssertionError(type(user).__name__)
    assert user.mark == 19


def sum_marks(values):
    total = 0
    for value in values:
        total += value.mark
    return total


def sum_ratios(values):
    total = 0.0
    for value in values:
        total += value.ratio
    return total


def bump_marks(values):
    for value in values:
        value.mark = value.mark + 1


# The unboxed int and float slots of a builtin-subclass carrier must read and
# write the same storage the interpreter uses. A receiver test that admits only
# ordinary instances answers zero for every read and drops every write, which
# the loops above never observe because they discard the loaded value.
for carrier in (I(5), S("user"), T((3, 4))):
    carrier_name = type(carrier).__name__
    carrier.mark = 7
    carrier.ratio = 0.5
    assert sum_marks([carrier] * 4000) == 28000, carrier_name
    assert sum_ratios([carrier] * 4000) == 2000.0, carrier_name
    bump_marks([carrier] * 4000)
    assert carrier.mark == 4007, carrier_name


class Before(str):
    pass


class After(str):
    @property
    def chosen(self):
        return "descriptor"


swapped = Before("value")
swapped.__dict__["chosen"] = "instance"
swapped.__class__ = After
assert swapped.chosen == "descriptor"


class SlottedStr(str):
    __slots__ = ("slot", "__dict__")


slotted = SlottedStr("value")
slotted.slot = "slot-value"
slotted.extra = "dict-value"
assert (slotted.slot, slotted.extra) == ("slot-value", "dict-value")

for value in (int_user, str_user, tuple_user):
    assert any(candidate is value for candidate in gc.get_objects())
    assert any(candidate is value for candidate in gc.get_referents([value]))

def builtin_iterator_subclass_mapdict():
    """Subclass instances of enumerate/map/filter/zip/reversed/super/property.

    `typedef.py` `_getusercls` allocates them with mapdict storage: attributes,
    `__dict__`, slots, weakrefs and `__del__` live on that layout.
    """

    import gc
    import weakref


    def exc_name(fn):
        try:
            fn()
        except Exception as e:
            return type(e).__name__
        else:
            return "ok"


    def plain(label, base, make, iterate):
        class S(base):
            pass

        x = make(S)
        object.__setattr__(x, "attr", 1)
        got = object.__getattribute__(x, "attr")
        reflected = object.__getattribute__(x, "__dict__")["attr"]
        object.__delattr__(x, "attr")
        missing = exc_name(lambda: object.__getattribute__(x, "attr"))
        gone = "attr" in object.__getattribute__(x, "__dict__")
        alive = weakref.ref(x)() is x
        if iterate is None:
            produced, same = "skip", "skip"
        else:
            produced = list(iterate(S))
            same = produced == list(iterate(base))
        print(
            "plain",
            label,
            type(x).__name__,
            isinstance(x, base),
            got,
            reflected,
            missing,
            gone,
            alive,
            produced,
            same,
        )


    def weakref_only(label, base, make):
        class S(base):
            __slots__ = ("__weakref__",)

        x = make(S)
        alive = weakref.ref(x)() is x
        dic = exc_name(lambda: object.__getattribute__(x, "__dict__"))
        print("weakref_slots", label, type(x).__name__, isinstance(x, base), alive, dic)


    def slots(label, base, make):
        class S(base):
            __slots__ = ("a",)

        x = make(S)
        object.__setattr__(x, "a", 2)
        got = object.__getattribute__(x, "a")
        object.__delattr__(x, "a")
        missing = exc_name(lambda: object.__getattribute__(x, "a"))
        extra = exc_name(lambda: object.__setattr__(x, "z", 1))
        dic = exc_name(lambda: object.__getattribute__(x, "__dict__"))
        print(
            "slots",
            label,
            type(x).__name__,
            isinstance(x, base),
            got,
            missing,
            extra,
            dic,
        )


    def finalizer(label, base, make):
        seen = []

        class D(base):
            def __del__(self):
                seen.append(label)

        def spawn():
            obj = make(D)
            return weakref.ref(obj)

        ref = spawn()
        gc.collect()
        print("del", label, ref() is None, seen)


    class A(object):
        pass


    class B(A):
        pass


    b = B()


    def make_enum(cls):
        return cls("ab")


    def make_map(cls):
        return cls(lambda z: z + 1, [1, 2])


    def make_filter(cls):
        return cls(None, [0, 1, 2])


    def make_zip(cls):
        return cls([1, 2], [3, 4])


    def make_rev(cls):
        return cls((1, 2, 3))


    def make_super(cls):
        return cls(B, b)


    def make_prop(cls):
        return cls()


    cases = [
        ("enumerate", enumerate, make_enum, make_enum),
        ("map", map, make_map, make_map),
        ("filter", filter, make_filter, make_filter),
        ("zip", zip, make_zip, make_zip),
        ("reversed", reversed, make_rev, make_rev),
        ("super", super, make_super, None),
        ("property", property, make_prop, None),
    ]
    for label, base, make, iterate in cases:
        plain(label, base, make, iterate)
        slots(label, base, make)
        weakref_only(label, base, make)
        finalizer(label, base, make)
builtin_iterator_subclass_mapdict()

def builtin_itertools_subclass_mapdict():
    """Subclass instances of the itertools types that accept subclassing.

    `typedef.py` `_getusercls` allocates them with mapdict storage: attributes,
    `__dict__`, slots, weakrefs and `__del__` live on that layout. `batched` is
    absent: pypy3 is 3.11. `_grouper`, `_tee` and `_tee_dataobject` reject
    subclassing.
    """

    import gc
    import itertools
    import weakref
    from itertools import islice


    def exc_name(fn):
        try:
            fn()
        except Exception as e:
            return type(e).__name__
        else:
            return "ok"


    def plain(label, base, make, iterate):
        class S(base):
            pass

        x = make(S)
        object.__setattr__(x, "attr", 1)
        got = object.__getattribute__(x, "attr")
        reflected = object.__getattribute__(x, "__dict__")["attr"]
        object.__delattr__(x, "attr")
        missing = exc_name(lambda: object.__getattribute__(x, "attr"))
        gone = "attr" in object.__getattribute__(x, "__dict__")
        alive = weakref.ref(x)() is x
        if iterate is None:
            produced, same = "skip", "skip"
        else:
            produced = list(iterate(S))
            same = produced == list(iterate(base))
        print(
            "plain",
            label,
            type(x).__name__,
            isinstance(x, base),
            got,
            reflected,
            missing,
            gone,
            alive,
            produced,
            same,
        )


    def slots(label, base, make):
        class S(base):
            __slots__ = ("a",)

        x = make(S)
        object.__setattr__(x, "a", 2)
        got = object.__getattribute__(x, "a")
        object.__delattr__(x, "a")
        missing = exc_name(lambda: object.__getattribute__(x, "a"))
        extra = exc_name(lambda: object.__setattr__(x, "z", 1))
        dic = exc_name(lambda: object.__getattribute__(x, "__dict__"))
        print(
            "slots",
            label,
            type(x).__name__,
            isinstance(x, base),
            got,
            missing,
            extra,
            dic,
        )


    def finalizer(label, base, make):
        seen = []

        class D(base):
            def __del__(self):
                seen.append(label)

        def spawn():
            obj = make(D)
            return weakref.ref(obj)

        ref = spawn()
        gc.collect()
        print("del", label, ref() is None, seen)


    def make_count(cls):
        return cls()


    def iter_count(cls):
        return islice(cls(), 4)


    def make_repeat(cls):
        return cls(7)


    def iter_repeat(cls):
        return islice(cls(7), 3)


    def make_takewhile(cls):
        return cls(lambda x: x < 3, [1, 2, 3, 4])


    def make_dropwhile(cls):
        return cls(lambda x: x < 3, [1, 2, 3, 4])


    def make_filterfalse(cls):
        return cls(None, [0, 1, 2])


    def make_islice(cls):
        return cls(range(10), 2, 6, 2)


    def make_product(cls):
        return cls([1, 2], [3])


    def make_combinations(cls):
        return cls([1, 2, 3], 2)


    def make_cwr(cls):
        return cls([1, 2], 2)


    def make_permutations(cls):
        return cls([1, 2, 3], 2)


    def make_groupby(cls):
        return cls([1, 1, 2])


    def iter_groupby(cls):
        return [(k, list(g)) for k, g in cls([1, 1, 2])]


    def make_compress(cls):
        return cls([1, 2, 3], [1, 0, 1])


    def make_starmap(cls):
        return cls(lambda a, b: a + b, [(1, 2), (3, 4)])


    def make_accumulate(cls):
        return cls([1, 2, 3])


    def make_zip_longest(cls):
        return cls([1], [2, 3], fillvalue=0)


    def make_pairwise(cls):
        return cls([1, 2, 3])


    def make_cycle(cls):
        return cls([1, 2])


    def iter_cycle(cls):
        return islice(cls([1, 2]), 4)


    def make_chain(cls):
        return cls([1, 2], [3])


    cases = [
        ("count", itertools.count, make_count, iter_count),
        ("repeat", itertools.repeat, make_repeat, iter_repeat),
        ("takewhile", itertools.takewhile, make_takewhile, make_takewhile),
        ("dropwhile", itertools.dropwhile, make_dropwhile, make_dropwhile),
        ("filterfalse", itertools.filterfalse, make_filterfalse, make_filterfalse),
        ("islice", itertools.islice, make_islice, make_islice),
        ("product", itertools.product, make_product, make_product),
        ("combinations", itertools.combinations, make_combinations, make_combinations),
        (
            "combinations_with_replacement",
            itertools.combinations_with_replacement,
            make_cwr,
            make_cwr,
        ),
        ("permutations", itertools.permutations, make_permutations, make_permutations),
        ("groupby", itertools.groupby, make_groupby, iter_groupby),
        ("compress", itertools.compress, make_compress, make_compress),
        ("starmap", itertools.starmap, make_starmap, make_starmap),
        ("accumulate", itertools.accumulate, make_accumulate, make_accumulate),
        ("zip_longest", itertools.zip_longest, make_zip_longest, make_zip_longest),
        ("pairwise", itertools.pairwise, make_pairwise, make_pairwise),
        ("cycle", itertools.cycle, make_cycle, iter_cycle),
        ("chain", itertools.chain, make_chain, make_chain),
    ]
    for label, base, make, iterate in cases:
        plain(label, base, make, iterate)
        slots(label, base, make)
        finalizer(label, base, make)
builtin_itertools_subclass_mapdict()

def builtin_subclass_mapdict_c5():
    """Subclass instances allocate through objspace.py allocate_instance.

    A plain subclass and a `__slots__` subclass keep `type(x) is S`, attribute
    storage, and a working base operation. `__del__` runs after `gc.collect()`.
    """

    import gc
    import io
    import weakref
    import _pickle
    import _struct
    from collections import deque
    from types import GenericAlias


    def exc_name(fn):
        try:
            fn()
        except Exception as e:
            return type(e).__name__
        else:
            return "ok"


    def plain(label, base, make, operate):
        class S(base):
            pass

        x = make(S)
        object.__setattr__(x, "attr", 1)
        got = object.__getattribute__(x, "attr")
        reflected = object.__getattribute__(x, "__dict__")["attr"]
        object.__delattr__(x, "attr")
        missing = exc_name(lambda: object.__getattribute__(x, "attr"))
        gone = "attr" in object.__getattribute__(x, "__dict__")
        row = [
            "plain",
            label,
            type(x) is S,
            isinstance(x, base),
            got,
            reflected,
            missing,
            gone,
        ]
        # Variable-size bases reject weak references. Omit that probe for int:
        # pypy3 accepts it and this file is diffed against pypy3.
        if base is not int:
            row.append(weakref.ref(x)() is x)
        row.append(operate(x))
        print(*row)


    def slots(label, base, make):
        class S(base):
            __slots__ = ("a",)

        x = make(S)
        object.__setattr__(x, "a", 2)
        got = object.__getattribute__(x, "a")
        object.__delattr__(x, "a")
        missing = exc_name(lambda: object.__getattribute__(x, "a"))
        extra = exc_name(lambda: object.__setattr__(x, "z", 1))
        dic = exc_name(lambda: object.__getattribute__(x, "__dict__"))
        print(
            "slots",
            label,
            type(x) is S,
            isinstance(x, base),
            got,
            missing,
            extra,
            dic,
        )


    def finalizer(label, base, make):
        seen = []

        class D(base):
            def __del__(self):
                seen.append(label)

        if base is int:
            def spawn():
                make(D)

            spawn()
            gc.collect()
            print("del", label, seen)
            return

        def spawn():
            obj = make(D)
            return weakref.ref(obj)

        ref = spawn()
        gc.collect()
        print("del", label, ref() is None, seen)


    def make_deque(cls):
        return cls()


    def operate_deque(x):
        x.append(1)
        x.append(2)
        return (x.pop(), x.popleft(), len(x))


    def make_struct(cls):
        return cls("h")


    def operate_struct(x):
        return x.unpack(x.pack(7)) == (7,)


    _BUFFERS = []


    def make_pickler(cls):
        buf = io.BytesIO()
        _BUFFERS.append(buf)
        return cls(buf)


    def operate_pickler(x):
        buf = _BUFFERS[-1]
        x.dump([1, "a"])
        buf.seek(0)
        return _pickle.Unpickler(buf).load() == [1, "a"]


    def make_unpickler(cls):
        buf = io.BytesIO()
        _pickle.Pickler(buf).dump((2, "b"))
        buf.seek(0)
        _BUFFERS.append(buf)
        return cls(buf)


    def operate_unpickler(x):
        return x.load() == (2, "b")


    def make_alias(cls):
        return cls(list, (int,))


    def operate_alias(x):
        return (x.__origin__ is list, x.__args__ == (int,))


    BIG = 10**30


    def make_int(cls):
        return cls(BIG)


    def operate_int(x):
        return (x + 1 - 1 == BIG, int(x) == BIG)


    plain("deque", deque, make_deque, operate_deque)
    slots("deque", deque, make_deque)
    finalizer("deque", deque, make_deque)

    plain("struct", _struct.Struct, make_struct, operate_struct)
    slots("struct", _struct.Struct, make_struct)
    finalizer("struct", _struct.Struct, make_struct)

    plain("pickler", _pickle.Pickler, make_pickler, operate_pickler)
    slots("pickler", _pickle.Pickler, make_pickler)
    finalizer("pickler", _pickle.Pickler, make_pickler)

    plain("unpickler", _pickle.Unpickler, make_unpickler, operate_unpickler)
    slots("unpickler", _pickle.Unpickler, make_unpickler)
    finalizer("unpickler", _pickle.Unpickler, make_unpickler)

    plain("alias", GenericAlias, make_alias, operate_alias)
    slots("alias", GenericAlias, make_alias)
    finalizer("alias", GenericAlias, make_alias)

    plain("int", int, make_int, operate_int)
    finalizer("int", int, make_int)


    class P(_pickle.Pickler):
        pass


    class U(_pickle.Unpickler):
        pass


    round_buf = io.BytesIO()
    P(round_buf).dump((2, "b"))
    round_buf.seek(0)
    print("round", U(round_buf).load() == (2, "b"))
builtin_subclass_mapdict_c5()

def builtin_subclass_slots_mapdict():
    """`__slots__` members of float/complex/str/bytearray/list/set/frozenset/array/weakref subclasses.

    A slots-only subclass has no `__dict__`; a subclass that also asks for
    `__dict__` must still keep the slot value in the slot, not in the dict.
    """

    import array
    import weakref


    class _Referent:
        pass


    _referent = _Referent()

    # int, tuple and bytes reject a nonempty __slots__.
    BASES = [
        (float, (1.5,)),
        (complex, (1 + 2j,)),
        (str, ("ab",)),
        (bytearray, (b"ab",)),
        (list, ()),
        (set, ()),
        (frozenset, ()),
        (array.array, ("i", [1, 2])),
        (weakref.ref, (_referent,)),
    ]

    N = 2000


    def check(base, args):
        SlotsOnly = type("S" + base.__name__, (base,), {"__slots__": ("x",)})
        WithDict = type("D" + base.__name__, (base,), {"__slots__": ("x", "__dict__")})

        total = 0
        for i in range(N):
            s = SlotsOnly(*args)
            s.x = i
            total += s.x
            del s.x
            try:
                s.x
            except AttributeError:
                pass
            else:
                raise AssertionError("deleted slot still readable on %s" % base.__name__)
            try:
                s.y = 1
            except AttributeError:
                pass
            else:
                raise AssertionError("slots-only %s accepted a new attribute" % base.__name__)
            assert not hasattr(s, "__dict__"), base

            d = WithDict(*args)
            d.x = i
            d.y = i
            assert "x" not in d.__dict__, (base, d.__dict__)
            assert d.__dict__ == {"y": i}, (base, d.__dict__)
            total += d.x + d.y

            # Direct member descriptor, typedef.py Member.descr_member_set.
            member = SlotsOnly.x
            s2 = SlotsOnly(*args)
            member.__set__(s2, i)
            assert member.__get__(s2, SlotsOnly) == i
            member.__delete__(s2)
            try:
                member.__get__(s2, SlotsOnly)
            except AttributeError:
                pass
            else:
                raise AssertionError("deleted slot still readable via Member on %s" % base.__name__)
        assert total == 3 * N * (N - 1) // 2, (base, total)


    def check_weakref_only(base, args):
        # typedef.py _getusercls mixes MapdictWeakrefSupport even when the
        # subclass has no __dict__. Types whose typedef is already weakrefable
        # reject a second __weakref__ slot.
        try:
            WeakOnly = type("W" + base.__name__, (base,), {"__slots__": ("__weakref__",)})
        except TypeError:
            return
        x = WeakOnly(*args)
        assert weakref.ref(x)() is x, base
        assert not hasattr(x, "__dict__"), base


    for base, args in BASES:
        check(base, args)
        check_weakref_only(base, args)
builtin_subclass_slots_mapdict()

def io_subclass_mapdict():
    """Subclass instances of the typed _io payloads allocate through _getusercls.

    A plain subclass and a `__slots__` subclass keep `type(x) is S`, attribute
    storage, and a working base operation. `__del__` runs after `gc.collect()`.
    """

    import gc
    import io
    import os
    import tempfile
    import weakref


    def exc_name(fn):
        try:
            fn()
        except Exception as e:
            return type(e).__name__
        else:
            return "ok"


    def plain(label, base, make, operate):
        class S(base):
            pass

        x = make(S)
        object.__setattr__(x, "attr", 1)
        got = object.__getattribute__(x, "attr")
        reflected = object.__getattribute__(x, "__dict__")["attr"]
        object.__delattr__(x, "attr")
        missing = exc_name(lambda: object.__getattribute__(x, "attr"))
        gone = "attr" in object.__getattribute__(x, "__dict__")
        print(
            "plain",
            label,
            type(x) is S,
            isinstance(x, base),
            got,
            reflected,
            missing,
            gone,
            weakref.ref(x)() is x,
            operate(x),
        )


    def slots(label, base, make):
        class S(base):
            __slots__ = ("a",)

        x = make(S)
        object.__setattr__(x, "a", 2)
        got = object.__getattribute__(x, "a")
        object.__delattr__(x, "a")
        missing = exc_name(lambda: object.__getattribute__(x, "a"))
        extra = exc_name(lambda: object.__setattr__(x, "z", 1))
        dic = exc_name(lambda: object.__getattribute__(x, "__dict__"))
        print(
            "slots",
            label,
            type(x) is S,
            isinstance(x, base),
            got,
            missing,
            extra,
            dic,
        )


    def finalizer(label, base, make):
        seen = []

        class D(base):
            def __del__(self):
                seen.append(label)

        def spawn():
            obj = make(D)
            return weakref.ref(obj)

        ref = spawn()
        gc.collect()
        print("del", label, ref() is None, seen)


    def make_bytesio(cls):
        return cls(b"ab")


    def operate_bytesio(x):
        return x.getvalue() == b"ab"


    def make_stringio(cls):
        return cls("ab")


    def operate_stringio(x):
        return x.getvalue() == "ab"


    def make_reader(cls):
        return cls(io.BytesIO(b"ab"))


    def operate_reader(x):
        return x.read() == b"ab"


    def make_writer(cls):
        raw = io.BytesIO()
        x = cls(raw)
        x.write(b"ab")
        x.flush()
        return x


    def operate_writer(x):
        x.flush()
        return True


    def make_random(cls):
        return cls(io.BytesIO(b"ab"))


    def operate_random(x):
        return x.read() == b"ab"


    def make_rwpair(cls):
        return cls(io.BytesIO(b"ab"), io.BytesIO())


    def operate_rwpair(x):
        return x.read() == b"ab"


    def make_text(cls):
        return cls(io.BytesIO(b"ab"), encoding="utf-8")


    def operate_text(x):
        return x.read() == "ab"


    def make_fileio(cls):
        fd, path = tempfile.mkstemp()
        os.write(fd, b"ab")
        os.lseek(fd, 0, os.SEEK_SET)
        return cls(fd, "rb")


    def operate_fileio(x):
        return x.read() == b"ab"


    for label, base, make, operate in [
        ("BytesIO", io.BytesIO, make_bytesio, operate_bytesio),
        ("StringIO", io.StringIO, make_stringio, operate_stringio),
        ("BufferedReader", io.BufferedReader, make_reader, operate_reader),
        ("BufferedWriter", io.BufferedWriter, make_writer, operate_writer),
        ("BufferedRandom", io.BufferedRandom, make_random, operate_random),
        ("BufferedRWPair", io.BufferedRWPair, make_rwpair, operate_rwpair),
        ("TextIOWrapper", io.TextIOWrapper, make_text, operate_text),
        ("FileIO", io.FileIO, make_fileio, operate_fileio),
    ]:
        plain(label, base, make, operate)
        slots(label, base, make)
        finalizer(label, base, make)
io_subclass_mapdict()
print("OK")
