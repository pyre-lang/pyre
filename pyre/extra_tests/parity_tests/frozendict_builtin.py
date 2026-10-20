# pyre-check: pypy-diverges: PyPy 3.11 has no builtin frozendict (PEP 814).
# CPython-suite gap: test_dict.FrozenDictTests covers constructor/copy/merge/hash
# but not dict(fd)/{**fd}/update/|/|=, __getnewargs__, or
# assignment/deletion/missing-attribute error text.
# parity-tests reason: the 3.15 stdlib uses frozendict; pin the builtin surface
# against the 3.15.0 oracle, including the cases the CPython suite leaves out.
# jit-core cannot import collections.abc, copy, or weakref yet: compile()
# rejects the 3.15 `module=` argument used by importlib.

assert frozendict.__mro__ == (frozendict, object)
assert frozendict.__bases__ == (object,)
assert not issubclass(frozendict, dict)
assert frozendict.__module__ == "builtins"

empty = frozendict()
fd = frozendict(a=1, b=2)
assert type(empty) is frozendict
assert not isinstance(fd, dict)
assert frozendict(fd) is fd
assert frozendict({"a": 1, "b": 2}) == fd
assert frozendict(a=1, b=2) == fd
assert frozendict([("a", 1), ("b", 2)]) == fd

assert repr(empty) == "frozendict()"
assert repr(fd) == "frozendict({'a': 1, 'b': 2})"
assert hash(fd) == hash(frozendict(b=2, a=1))
assert hash(fd) == hash(frozenset(fd.items()))
assert hash(empty) == hash(frozenset())

assert fd.copy() is fd
assert empty.copy() is empty
assert fd.__getnewargs__() == ({"a": 1, "b": 2},)
assert type(fd.__getnewargs__()[0]) is dict
assert frozendict[str, int].__origin__ is frozendict

assert fd == {"a": 1, "b": 2}
assert {"a": 1, "b": 2} == fd
assert dict(fd) == {"a": 1, "b": 2}
assert {**fd} == {"a": 1, "b": 2}

d = {"x": 0}
d.update(fd)
assert d == {"x": 0, "a": 1, "b": 2}
assert type({"x": 0} | fd) is dict
assert {"x": 0} | fd == {"x": 0, "a": 1, "b": 2}
d = {"x": 0}
d |= fd
assert d == {"x": 0, "a": 1, "b": 2}

assert type(fd | {"c": 3}) is frozendict
assert fd | {"c": 3} == frozendict(a=1, b=2, c=3)
assert (fd | {}) is fd
assert (fd | frozendict()) is fd
assert (frozendict() | fd) is fd
e = frozendict()
assert (e | {}) is e

assert fd | frozendict(b=9, c=3) == frozendict(a=1, b=9, c=3)
assert hash(frozendict(a=1) | frozendict(b=2)) == hash(frozendict(a=1, b=2))

copy_fd = fd
fd2 = fd
fd2 |= frozendict(c=3)
assert fd2 == frozendict(a=1, b=2, c=3)
assert copy_fd is fd
assert copy_fd == frozendict(a=1, b=2)

assert "a" in fd
assert fd["a"] == 1
assert fd.get("z", 9) == 9
assert list(fd) == ["a", "b"]
assert list(fd.keys()) == ["a", "b"]
assert list(fd.values()) == [1, 2]
assert list(fd.items()) == [("a", 1), ("b", 2)]
assert type(fd.keys()).__name__ == "dict_keys"
assert type(fd.values()).__name__ == "dict_values"
assert type(fd.items()).__name__ == "dict_items"
assert type(fd.keys().mapping).__name__ == "mappingproxy"
assert list(reversed(fd)) == ["b", "a"]
assert len(fd) == 2
assert not empty
assert fd
assert fd.items() ^ frozendict(b=2, c=3).items() == {("a", 1), ("c", 3)}

assert frozendict.fromkeys("ab", 1) == frozendict(a=1, b=1)
assert type(frozendict.fromkeys("ab")) is frozendict

fd.__init__(x=1)
assert fd == frozendict(a=1, b=2)

try:
    dict.__init__(fd, x=1)
except TypeError as e:
    assert "requires a 'dict' object but received a 'frozendict'" in str(e), e
else:
    raise AssertionError("dict.__init__ accepted a frozendict")

try:
    fd["a"] = 3
except TypeError as e:
    assert str(e) == "'frozendict' object does not support item assignment", e
else:
    raise AssertionError("item assignment succeeded")

try:
    del fd["a"]
except TypeError as e:
    assert str(e) == "'frozendict' object does not support item deletion", e
else:
    raise AssertionError("item deletion succeeded")

try:
    fd.update(a=1)
except AttributeError as e:
    assert str(e) == "'frozendict' object has no attribute 'update'", e
else:
    raise AssertionError("update attribute existed")

try:
    fd.foo
except AttributeError as e:
    assert str(e) == "'frozendict' object has no attribute 'foo'", e
else:
    raise AssertionError("missing attribute succeeded")

class NotADict:
    def __iter__(self):
        return iter(())
    def __len__(self):
        return 0
    def __getitem__(self, k):
        raise KeyError(k)

try:
    fd | NotADict()
except TypeError as e:
    assert "unsupported operand type(s) for |" in str(e), e
else:
    raise AssertionError("| accepted a non-dict mapping")

try:
    [] in fd
except TypeError as e:
    assert str(e) == "cannot use 'list' as a frozendict key (unhashable type: 'list')", e
else:
    raise AssertionError("unhashable containment succeeded")

try:
    fd[[]]
except TypeError as e:
    assert str(e) == "cannot use 'list' as a frozendict key (unhashable type: 'list')", e
else:
    raise AssertionError("unhashable getitem succeeded")

try:
    fd.get([])
except TypeError as e:
    assert str(e) == "cannot use 'list' as a frozendict key (unhashable type: 'list')", e
else:
    raise AssertionError("unhashable get succeeded")

try:
    frozendict([([], 1)])
except TypeError as e:
    assert str(e) == "cannot use 'list' as a frozendict key (unhashable type: 'list')", e
else:
    raise AssertionError("unhashable constructor succeeded")

try:
    hash(frozendict(x=[1]))
except TypeError as e:
    assert "unhashable type: 'list'" in str(e), e
else:
    raise AssertionError("hash of unhashable value succeeded")

try:
    frozendict({}, {})
except TypeError as e:
    assert str(e) == "frozendict expected at most 1 argument, got 2", e
else:
    raise AssertionError("two-arg constructor succeeded")

class FrozenDict(frozendict):
    pass

sub = FrozenDict(a=1, b=2)
assert type(sub) is FrozenDict
assert not (FrozenDict(fd) is fd)
assert FrozenDict(fd) == fd
assert type(sub.copy()) is frozendict
assert sub.copy() is not sub
assert sub.copy() == fd
assert type(sub | fd) is frozendict
assert type(fd | sub) is frozendict
assert (fd | FrozenDict()) is not fd
assert repr(sub) == "FrozenDict({'a': 1, 'b': 2})"
assert repr(FrozenDict()) == "FrozenDict()"
assert type(FrozenDict.fromkeys("a")) is FrozenDict
assert FrozenDict.fromkeys("a") == frozendict(a=None)

created = frozendict(x=1)

class FrozenDictSubclass(frozendict):
    def __new__(cls, *args, **kwargs):
        if args or kwargs:
            return super().__new__(cls, *args, **kwargs)
        else:
            return created

got = FrozenDictSubclass.fromkeys("abc")
assert got == frozendict(x=1, a=None, b=None, c=None)
assert type(got) is FrozenDictSubclass
assert created == frozendict(x=1)

assert not hasattr(fd, "__weakref__")

# frozendict is a dict plus its cached hash word, on GIL and free-threaded
# builds alike.
assert empty.__sizeof__() == {}.__sizeof__() + 8
assert frozendict(a=1).__sizeof__() == {"a": 1}.__sizeof__() + 8
assert frozendict.__dict__.__contains__("__hash__")
assert "__ior__" not in frozendict.__dict__
assert "__init__" not in frozendict.__dict__

print("OK")
