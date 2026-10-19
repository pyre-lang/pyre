# CPython-suite gap: no suite test rebinds a class attribute twice and then
# keeps reading it in a loop hot enough to compile.  The second rebind is the
# one that matters here, and only the second.
#
# `typeobject.py W_TypeObject.setdictvalue` routes every class-namespace store
# through `write_cell`.  The store that replaces the class-body value builds an
# `ObjectMutableCell` and calls `mutated()`, which revokes `_version_tag` here
# and in every subclass.  Every store after that writes `w_value` inside the
# installed cell, `write_cell` returns `None`, and `mutated()` is skipped -- so
# no version tag moves.
#
# parity-tests reason: a trace that answers a class-namespace lookup under a
# version-tag pin alone is therefore correct until the first rebind and wrong
# from the second on, which shows up as a wrong number rather than as a missed
# optimisation.  The fix is upstream's own shape -- read
# `ObjectMutableCell.w_value` off the pinned cell with a `getfield` and guard
# what was read -- so each case below is a rebind the compiled code must see.
#
# Every rebind happens INSIDE its loop: a store after the loop is interpreted
# and would not consult what the trace baked.  The store sits after the read,
# so iterations `0..=SWITCH` read the first value and the rest read the second.
#
# type_dict_cell_rebinds_protocol_slots:
# no suite test rebinds `__next__`, `__iter__`, `__index__`, `__getattr__`,
# `__getattribute__`, `__enter__` or `__exit__` twice in a hot loop.
# `typeobject.py W_TypeObject.setdictvalue` / `write_cell` / `ObjectMutableCell`.
# a version-tag pin is wrong from the second rebind on.
N = 20000
SWITCH = N // 2


class Box:
    def __init__(self, v):
        self.v = v


class Descr:
    def __get__(self, obj, objtype=None):
        return Box(7)


class DataDescr:
    def __get__(self, obj, objtype=None):
        return Box(9)

    def __set__(self, obj, value):
        raise AssertionError('not stored through')


def expect(total, first, second, what):
    wanted = (SWITCH + 1) * first + (N - SWITCH - 1) * second
    assert total == wanted, '%s: %r != %r' % (what, total, wanted)


class Callable:
    def __call__(self):
        return 1


# The class-body value is the first store; this one parks the cell.
Callable.__call__ = lambda self: 1


def instance_call():
    obj = Callable()
    total = 0
    i = 0
    while i < N:
        total += obj()
        if i == SWITCH:
            Callable.__call__ = lambda self: 7
        i += 1
    expect(total, 1, 7, 'stale __call__')


class Methods:
    def m(self):
        return 1


Methods.m = lambda self: 1


def method_becomes_property():
    obj = Methods()
    total = 0
    i = 0
    while i < N:
        total += obj.m()
        if i == SWITCH:
            # The payload stops being a method descriptor, so the pair
            # `LOAD_METHOD` pushes has to change shape with it.
            Methods.m = property(lambda self: (lambda: 7))
        i += 1
    expect(total, 1, 7, 'stale method payload')


class Hashed:
    def __hash__(self):
        return 1


Hashed.__hash__ = lambda self: 1


def user_hash():
    obj = Hashed()
    total = 0
    i = 0
    while i < N:
        total += hash(obj)
        if i == SWITCH:
            Hashed.__hash__ = lambda self: 7
        i += 1
    expect(total, 1, 7, 'stale __hash__')


class Formatted:
    def __format__(self, spec):
        return 'a'


Formatted.__format__ = lambda self, spec: 'a'


def user_format():
    obj = Formatted()
    total = 0
    i = 0
    while i < N:
        total += len(format(obj))
        if i == SWITCH:
            Formatted.__format__ = lambda self, spec: 'abcdefg'
        i += 1
    expect(total, 1, 7, 'stale __format__')


class Added:
    def __add__(self, other):
        return 1


Added.__add__ = lambda self, other: 1


def user_binop():
    obj = Added()
    total = 0
    i = 0
    while i < N:
        total += obj + 0
        if i == SWITCH:
            Added.__add__ = lambda self, other: 7
        i += 1
    expect(total, 1, 7, 'stale __add__')


class Compared:
    def __lt__(self, other):
        return True


Compared.__lt__ = lambda self, other: True


def user_compare():
    obj = Compared()
    total = 0
    i = 0
    while i < N:
        total += 1 if obj < 0 else 7
        if i == SWITCH:
            Compared.__lt__ = lambda self, other: False
        i += 1
    expect(total, 1, 7, 'stale __lt__')


class Err(Exception):
    def __str__(self):
        return 'a'


Err.__str__ = lambda self: 'a'


def exception_str():
    err = Err()
    total = 0
    i = 0
    while i < N:
        total += len(str(err))
        if i == SWITCH:
            Err.__str__ = lambda self: 'abcdefg'
        i += 1
    expect(total, 1, 7, 'stale exception __str__')


class Attrs:
    x = Box(1)


Attrs.x = Box(1)


def type_attr_becomes_descriptor():
    total = 0
    i = 0
    while i < N:
        total += Attrs.x.v
        if i == SWITCH:
            # `type.__getattribute__` has to start running `__get__` on the
            # payload the cell now holds.
            Attrs.x = Descr()
        i += 1
    expect(total, 1, 7, 'stale type attribute')


class Shadow:
    y = Box(1)


Shadow.y = Box(1)


def instance_attr_gains_a_data_descriptor():
    obj = Shadow()
    obj.y = Box(1)
    total = 0
    i = 0
    while i < N:
        total += obj.y.v
        if i == SWITCH:
            # A data descriptor on the type shadows the instance dict, so the
            # storage read the fold settled on stops being the answer.
            Shadow.y = DataDescr()
        i += 1
    expect(total, 1, 9, 'instance attribute kept shadowing a data descriptor')


# The interpreter's own `LOAD_ATTR` cache, not a trace: `LOAD_ATTR_slowpath`
# classifies the class-namespace entry and caches the instance slot under
# `_version_tag`.  A rebind absorbed by the cell moves no tag, so the entry has
# to be rejected at classify time -- which is why the classification reads the
# raw entry and gives up on a `MutableCell`.  `str` is the shape that reaches
# it: a non-data descriptor of an immutable type is the one payload
# `_classify_attr` still calls cacheable.
class Cached:
    x = 'a'


Cached.x = 'b'


def read_cached(obj):
    return obj.x


def absorbed_rebind_beats_the_attr_cache():
    obj = Cached()
    obj.x = 'instance'
    assert read_cached(obj) == 'instance'
    # A data descriptor now shadows the instance dict, and the store that
    # installs it is absorbed by the cell.
    Cached.x = property(lambda self: 'from-property')
    got = read_cached(obj)
    assert got == 'from-property', 'stale LOAD_ATTR cache: %r' % (got,)


absorbed_rebind_beats_the_attr_cache()
instance_call()
method_becomes_property()
user_hash()
user_format()
user_binop()
user_compare()
exception_str()
type_attr_becomes_descriptor()
instance_attr_gains_a_data_descriptor()

N_proto = 20000
SWITCH_proto = N_proto // 2


def expect_proto(total, first, second, what):
    wanted = (SWITCH_proto + 1) * first + (N_proto - SWITCH_proto - 1) * second
    assert total == wanted, '%s: %r != %r' % (what, total, wanted)


class It:
    def __next__(self):
        return 1


It.__next__ = lambda self: 1


def user_next():
    obj = It()
    total = 0
    i = 0
    while i < N_proto:
        total += next(obj)
        if i == SWITCH_proto:
            It.__next__ = lambda self: 7
        i += 1
    expect_proto(total, 1, 7, 'stale __next__')


class Iterable:
    def __iter__(self):
        return iter((1,))


Iterable.__iter__ = lambda self: iter((1,))


def user_iter():
    obj = Iterable()
    total = 0
    i = 0
    while i < N_proto:
        total += next(iter(obj))
        if i == SWITCH_proto:
            Iterable.__iter__ = lambda self: iter((7,))
        i += 1
    expect_proto(total, 1, 7, 'stale __iter__')


class Idx:
    def __index__(self):
        return 1


Idx.__index__ = lambda self: 1


def user_index():
    obj = Idx()
    lst = [0, 10, 0, 0, 0, 0, 0, 70]
    total = 0
    i = 0
    while i < N_proto:
        total += lst[obj]
        if i == SWITCH_proto:
            Idx.__index__ = lambda self: 7
        i += 1
    expect_proto(total, 10, 70, 'stale __index__')


class Ga:
    def __getattr__(self, name):
        return 1


Ga.__getattr__ = lambda self, name: 1


def user_getattr():
    obj = Ga()
    total = 0
    i = 0
    while i < N_proto:
        total += obj.missing
        if i == SWITCH_proto:
            Ga.__getattr__ = lambda self, name: 7
        i += 1
    expect_proto(total, 1, 7, 'stale __getattr__')


class Gattr:
    def __getattribute__(self, name):
        return 1


Gattr.__getattribute__ = lambda self, name: 1


def user_getattribute():
    obj = Gattr()
    total = 0
    i = 0
    while i < N_proto:
        total += obj.x
        if i == SWITCH_proto:
            Gattr.__getattribute__ = lambda self, name: 7
        i += 1
    expect_proto(total, 1, 7, 'stale __getattribute__')


class Cm:
    def __enter__(self):
        return 1

    def __exit__(self, *args):
        return None


Cm.__enter__ = lambda self: 1
Cm.__exit__ = lambda self, *args: None


def user_with():
    obj = Cm()
    total = 0
    i = 0
    while i < N_proto:
        with obj as v:
            total += v
        if i == SWITCH_proto:
            Cm.__enter__ = lambda self: 7

            def counting_exit(self, *args):
                global exit_calls
                exit_calls += 1

            Cm.__exit__ = counting_exit
        i += 1
    expect_proto(total, 1, 7, 'stale __enter__')
    assert exit_calls == N_proto - SWITCH_proto - 1, 'stale __exit__: %r' % exit_calls


exit_calls = 0
user_next()
user_iter()
user_index()
user_getattr()
user_getattribute()
user_with()
print('OK')
