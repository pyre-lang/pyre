# pyre-check: pypy-diverges: PyPy 3.11 has no builtin frozendict (PEP 814).
# CPython-suite gap: FrozenDictTests.test_constructor only pins dict.__init__(fd);
# test_eval_frozendict / test_exec_frozendict pin the three eval/exec cases
# but not dict mutator/reader descriptors on a frozendict, nor frozendict
# descriptors on an unrelated receiver.
# parity-tests reason: dict method_owner must refuse a frozendict (not a dict
# subclass); frozendict method_owner must refuse a non-instance; eval/exec
# must accept frozendict globals, plant __builtins__ only when writable, and
# STORE_NAME on that mapping must raise item-assignment TypeError.


def expect_typeerror(fn, contains):
    try:
        fn()
    except TypeError as err:
        text = str(err)
        assert contains in text, (contains, text)
        return
    raise AssertionError("expected TypeError")


fd = frozendict(a=1)
expect_typeerror(
    lambda: dict.__init__(fd, x=1),
    "requires a 'dict' object but received a 'frozendict'",
)
assert fd == frozendict(a=1)

for name, call in (
    ("update", lambda: dict.update(fd)),
    ("pop", lambda: dict.pop(fd, "a")),
    ("popitem", lambda: dict.popitem(fd)),
    ("clear", lambda: dict.clear(fd)),
    ("setdefault", lambda: dict.setdefault(fd, "a")),
    ("__getitem__", lambda: dict.__getitem__(fd, "a")),
    ("keys", lambda: dict.keys(fd)),
    ("values", lambda: dict.values(fd)),
    ("items", lambda: dict.items(fd)),
    ("get", lambda: dict.get(fd, "a")),
    ("copy", lambda: dict.copy(fd)),
    ("__contains__", lambda: dict.__contains__(fd, "a")),
):
    expect_typeerror(
        call,
        f"descriptor '{name}' for 'dict' objects doesn't apply to a 'frozendict' object",
    )

for name, call in (
    ("__setitem__", lambda: dict.__setitem__(fd, "a", 2)),
    ("__delitem__", lambda: dict.__delitem__(fd, "a")),
    ("__ior__", lambda: dict.__ior__(fd, {})),
    ("__len__", lambda: dict.__len__(fd)),
    ("__iter__", lambda: dict.__iter__(fd)),
    ("__or__", lambda: dict.__or__(fd, {})),
    ("__init__", lambda: dict.__init__(fd)),
):
    expect_typeerror(
        call,
        f"descriptor '{name}' requires a 'dict' object but received a 'frozendict'",
    )

lst = [10]
for name, call in (
    ("__getitem__", lambda: frozendict.__getitem__(lst, 0)),
    ("__contains__", lambda: frozendict.__contains__(lst, 0)),
    ("copy", lambda: frozendict.copy(lst)),
    ("keys", lambda: frozendict.keys(lst)),
    ("values", lambda: frozendict.values(lst)),
    ("items", lambda: frozendict.items(lst)),
    ("get", lambda: frozendict.get(lst, 0)),
    ("__reversed__", lambda: frozendict.__reversed__(lst)),
    ("__sizeof__", lambda: frozendict.__sizeof__(lst)),
    ("__getnewargs__", lambda: frozendict.__getnewargs__(lst)),
):
    expect_typeerror(
        call,
        f"descriptor '{name}' for 'frozendict' objects doesn't apply to a 'list' object",
    )

for name, call in (
    ("__len__", lambda: frozendict.__len__(lst)),
    ("__iter__", lambda: frozendict.__iter__(lst)),
    ("__eq__", lambda: frozendict.__eq__(lst, {})),
    ("__ne__", lambda: frozendict.__ne__(lst, {})),
    ("__or__", lambda: frozendict.__or__(lst, {})),
    ("__ror__", lambda: frozendict.__ror__(lst, {})),
    ("__hash__", lambda: frozendict.__hash__(lst)),
    ("__repr__", lambda: frozendict.__repr__(lst)),
):
    expect_typeerror(
        call,
        f"descriptor '{name}' requires a 'frozendict' object but received a 'list'",
    )

expect_typeerror(
    lambda: frozendict.__dict__["fromkeys"](lst, "ab"),
    "descriptor 'fromkeys' for type 'frozendict' needs a type, not a 'list' as arg 2",
)


class Frozen(frozendict):
    pass


sub = Frozen(a=1)
assert frozendict.__getitem__(sub, "a") == 1
expect_typeerror(
    lambda: dict.__init__(sub, x=1),
    "requires a 'dict' object but received a 'Frozen'",
)

ns = frozendict(x=1, data=[], __builtins__=__builtins__)
eval("data.append(x)", ns, ns)
assert ns["data"] == [1]
exec("data.append(x)", ns, ns)
assert ns["data"] == [1, 1]

empty = frozendict()
expect_typeerror(
    lambda: eval("", empty, empty),
    "cannot assign __builtins__ to frozendict globals",
)
expect_typeerror(
    lambda: exec("", empty, empty),
    "cannot assign __builtins__ to frozendict globals",
)

assign_ns = frozendict(__builtins__=__builtins__)
expect_typeerror(
    lambda: exec("x = 1", assign_ns, assign_ns),
    "'frozendict' object does not support item assignment",
)
assert "x" not in assign_ns

print("OK")
