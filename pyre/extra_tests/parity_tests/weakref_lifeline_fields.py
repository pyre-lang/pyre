# CPython-suite gap: test_weakref does not collect a lifeline for each
# make_weakref_descr builtin in one pass.
# parity-tests reason: make_weakref_descr, W_BaseSetObject and W_TypeObject
# store _lifeline_ on the instance; MapdictWeakrefSupport covers a
# _getusercls layout whose base typedef is not weakrefable.

"""Weakref lifelines on builtin instances round-trip and collect.

One instance of each builtin that both runtimes allow `weakref.ref` on,
plus a subclass of `_local` and of `ModuleType`. `weakref.getweakrefcount`
matches `weakref.getweakrefs`, a `WeakValueDictionary` keeps the value
while the owner is alive, and the callback runs once `del` plus
`gc.collect` drops the owner. `type` itself stays alive.
"""

import array
import collections
import gc
import io
import itertools
import mmap
import os
import pickle
import re
import struct
import tempfile
import types
import weakref
import _thread


def exercise(make, after_del=None):
    obj = make()
    fired = []

    def callback(_ref):
        fired.append(1)

    ref = weakref.ref(obj, callback)
    refs = weakref.getweakrefs(obj)
    count = weakref.getweakrefcount(obj)
    assert ref in refs
    assert count == len(refs) and count >= 1
    bag = weakref.WeakValueDictionary()
    bag["item"] = obj
    assert bag["item"] is obj
    assert weakref.getweakrefcount(obj) == count + 1
    del obj
    if after_del is not None:
        after_del()
    for _ in range(8):
        gc.collect()
    assert fired == [1]
    assert ref() is None
    assert "item" not in bag


def expect_type_error(make):
    try:
        weakref.ref(make())
    except TypeError:
        return
    raise AssertionError("weakref.ref should raise TypeError")


def make_generator():
    def gen():
        yield 1

    return gen()


def make_coroutine():
    async def coro():
        return 1

    co = coro()
    co.close()
    return co


def make_async_generator():
    async def agen():
        yield 1

    return agen()


def make_method():
    class Owner:
        def method(self):
            return 1

    return Owner().method


def make_mmap():
    handle = tempfile.TemporaryFile()
    handle.write(b"x" * 64)
    handle.flush()
    mapped = mmap.mmap(handle.fileno(), 64)
    # The file object has to outlive the mapping; the mapping itself is the
    # weakref owner and is the only object `exercise` drops.
    make_mmap.files.append(handle)
    return mapped


make_mmap.files = []


def make_pattern():
    return re.compile("lifeline-pattern-9f3c")


def make_union():
    return type("LifelineA", (), {}) | type("LifelineB", (), {})


def make_code():
    return compile("x = 1", "<lifeline-code>", "exec")


def make_fileio():
    fd, name = tempfile.mkstemp()
    os.close(fd)
    # Windows rejects unlink while FileIO still has the file open
    # (WinError 32). Drop the name after `exercise` has collected it.
    make_fileio.names.append(name)
    return io.FileIO(name, "rb")


make_fileio.names = []


def make_alias():
    return type("LifelineList", (list,), {})[int]


class ListSub(list):
    pass


class DequeSub(collections.deque):
    pass


class SetSub(set):
    pass


class ExcSub(ValueError):
    pass


class SubLocal(_thread._local):
    pass


class SubMod(types.ModuleType):
    pass


exercise(lambda: _thread.RLock())
exercise(lambda: _thread.allocate_lock())
exercise(lambda: _thread._local())
exercise(make_generator)
exercise(make_coroutine)
exercise(make_async_generator)
exercise(lambda: collections.deque())
exercise(lambda: (lambda: None))
exercise(make_method)
exercise(lambda: itertools.tee([1, 2, 3])[0])
exercise(make_mmap)
exercise(lambda: pickle.PickleBuffer(bytearray(b"abc")))
exercise(make_pattern, re.purge)
exercise(lambda: set())
exercise(lambda: frozenset())
exercise(lambda: type("LifelineT", (), {}))
exercise(make_union)
exercise(lambda: types.ModuleType("lifeline-module"))
exercise(make_code)
exercise(lambda: memoryview(bytearray(b"ab")))
exercise(lambda: array.array("B", [1, 2, 3]))
exercise(lambda: struct.Struct("b"))
exercise(lambda: io.BytesIO())
exercise(lambda: io.StringIO())
exercise(lambda: io.BufferedReader(io.BytesIO()))
exercise(lambda: io.BufferedWriter(io.BytesIO()))
exercise(lambda: io.BufferedRandom(io.BytesIO()))
exercise(lambda: io.BufferedRWPair(io.BytesIO(), io.BytesIO()))
exercise(lambda: io.TextIOWrapper(io.BytesIO()))
try:
    exercise(make_fileio)
finally:
    while make_fileio.names:
        name = make_fileio.names.pop()
        try:
            os.unlink(name)
        except OSError:
            pass
exercise(make_alias)
exercise(lambda: type("LifelineUser", (), {})())
exercise(lambda: ListSub())
exercise(lambda: DequeSub())
exercise(lambda: SetSub())
exercise(lambda: ExcSub())
exercise(SubLocal)
exercise(lambda: SubMod("lifeline-submod"))
# `type` itself cannot be collected. The lifeline still has to resolve.
assert weakref.ref(type)() is type

expect_type_error(object)
expect_type_error(lambda: 1)
expect_type_error(lambda: "s")
expect_type_error(lambda: [])
expect_type_error(lambda: {})
expect_type_error(lambda: ())

print("OK")
