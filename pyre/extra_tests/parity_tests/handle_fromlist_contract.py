# CPython-suite gap: `test_import` exercises `from pkg import sub` and star
# imports through the app-level importer only. Nothing there pins the contract
# itself -- which names get bound, which failures are swallowed, and what a
# cached submodule does -- so a runtime that has to implement `_handle_fromlist`
# somewhere other than `importlib._bootstrap` is graded by nothing.
# parity-tests reason: pyre answers `__import__` from two importers. The
# app-level `_bootstrap.__import__` runs wherever `importlib._bootstrap` is in
# `sys.modules`; the native importer stands in while it is not -- during
# startup, and for the whole run of a guest that carries no importlib, where
# `from . import X` reaches nothing else. Both owe the same observable
# behaviour, and only the app-level one is covered by the suite.
#
# handle_fromlist_native_importer:
# nothing in `test_import` reaches an importer other than `importlib._bootstrap`, so a
# runtime that carries a second implementation of `_handle_fromlist` is graded on only one
# of them. pyre's `dunder_import` hands a fromlist to the app-level
# `_bootstrap.__import__` whenever `importlib._bootstrap` is in `sys.modules`, and to a
# native `handle_fromlist` when it is not -- during startup, and for the whole run of a

import json
import os
import shutil
import sys
import tempfile
import types

root = tempfile.mkdtemp(prefix="pyre_hfl_")
os.mkdir(os.path.join(root, "hflpkg"))
for name, body in [
    ("__init__.py", "__all__ = ['a', 'b']\n"),
    ("a.py", "A = 1\n"),
    ("b.py", "B = 2\n"),
    ("c.py", "C = 3\n"),
    # Written up front rather than beside the assertion that imports it: on
    # Windows a directory's last-write time is not updated in time for the
    # next stat of it, so the mtime-keyed `FileFinder` cache does not notice a
    # file created after the package's finder was built and the submodule is
    # not found at all.
    ("d.py", "import nonexistent_module_xyz\n"),
]:
    with open(os.path.join(root, "hflpkg", name), "w") as f:
        f.write(body)
sys.path.insert(0, root)

# `*` expands `__all__` and imports each name in it; a submodule the package
# does not list stays unimported.
mod = __import__("hflpkg", {}, {}, ["*"], 0)
assert mod.__name__ == "hflpkg"
assert hasattr(mod, "a") and hasattr(mod, "b"), "star did not import __all__"
assert not hasattr(mod, "c"), "star imported a name outside __all__"
assert "hflpkg.c" not in sys.modules

# A fromlist name that is neither an attribute nor an importable submodule is
# swallowed: `__import__` returns the package and IMPORT_FROM raises later.
assert __import__("hflpkg", {}, {}, ["zzz"], 0).__name__ == "hflpkg"

# A submodule whose own body raises ModuleNotFoundError for a *different* name
# is a real failure, not a name the fromlist may skip.
try:
    __import__("hflpkg", {}, {}, ["d"], 0)
except ModuleNotFoundError as exc:
    assert exc.name == "nonexistent_module_xyz", exc.name
else:
    raise AssertionError("expected ModuleNotFoundError")

# `sys.modules[name] = None` blocks a name; the fromlist does not swallow that
# either.
sys.modules["hflpkg.blocked"] = None
try:
    __import__("hflpkg", {}, {}, ["blocked"], 0)
except ImportError as exc:
    assert "hflpkg.blocked" in str(exc), str(exc)
else:
    raise AssertionError("expected ImportError for a blocked name")
del sys.modules["hflpkg.blocked"]

# Every item must be a str, and the message names where the item came from.
try:
    __import__("hflpkg", {}, {}, [1], 0)
except TypeError as exc:
    assert str(exc) == "Item in ``from list'' must be str, not int", str(exc)
else:
    raise AssertionError("expected TypeError")

mod.__all__ = ["a", 2]
try:
    __import__("hflpkg", {}, {}, ["*"], 0)
except TypeError as exc:
    assert str(exc) == "Item in hflpkg.__all__ must be str, not int", str(exc)
else:
    raise AssertionError("expected TypeError")
mod.__all__ = ["a", "b"]

# A `sys.modules` hit returns before the parent binding runs, so neither
# spelling puts back an attribute the program deleted.
import json.decoder

decoder = sys.modules["json.decoder"]
delattr(json, "decoder")
import json.decoder

assert not hasattr(json, "decoder"), "`import a.b` rebound a cached submodule"

got = __import__("json", {}, {}, ["decoder"], 0)
assert not hasattr(got, "decoder"), "a fromlist rebound a cached submodule"
assert sys.modules["json.decoder"] is decoder, "the cached submodule was reloaded"

# A fromlist item is an arbitrary `str`, including one carrying a lone
# surrogate. Such a name cannot be an importable module, so the optional-import
# failure is swallowed and the package comes back.
assert __import__("hflpkg", {}, {}, ["\ud800"], 0) is mod
setattr(mod, "\ud800", 5)
assert __import__("hflpkg", {}, {}, ["\ud800"], 0) is mod
delattr(mod, "\ud800")

# `sys.modules` may hold any object with a `__path__`. The child name is
# formatted from `__name__`, so a non-str one is coerced rather than rejected,
# and the resulting import failure names the parent -- not the child -- so it
# is not the one a fromlist swallows.
class Package:
    __path__ = []
    __name__ = 123


sys.modules["hflfake"] = Package()
try:
    __import__("hflfake", {}, {}, ["x"], 0)
except ModuleNotFoundError as exc:
    assert exc.name == "123", exc.name
else:
    raise AssertionError("expected ModuleNotFoundError")
del sys.modules["hflfake"]

# The fromlist is iterated lazily. An item is fully processed before the next is
# requested, so an error a later `__next__` raises cannot preempt the TypeError
# an earlier item already owes...
class NonStrThenRaise:
    def __init__(self):
        self.n = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.n += 1
        if self.n == 1:
            return 1
        raise ValueError("must not be reached before the TypeError")


try:
    __import__("hflpkg", {}, {}, NonStrThenRaise(), 0)
except TypeError as exc:
    assert str(exc) == "Item in ``from list'' must be str, not int", str(exc)
else:
    raise AssertionError("expected TypeError")

# ... and an import an earlier item performed is visible to the `__next__` that
# follows it.
class WatchAfterFirst:
    def __init__(self):
        self.n = 0
        self.saw = None

    def __iter__(self):
        return self

    def __next__(self):
        self.n += 1
        if self.n == 1:
            return "c"
        self.saw = hasattr(sys.modules["hflpkg"], "c")
        raise StopIteration


watcher = WatchAfterFirst()
__import__("hflpkg", {}, {}, watcher, 0)
assert watcher.saw is True, "the earlier item's import was not visible"

# `if not fromlist` is a truth test, not a None test: an empty list answers the
# HEAD package, the same as omitting the argument, while a non-empty one answers
# the leaf.
import hflpkg.a

assert __import__("hflpkg.a", {}, {}, [], 0) is sys.modules["hflpkg"]
assert __import__("hflpkg.a", {}, {}, (), 0) is sys.modules["hflpkg"]
assert __import__("hflpkg.a", {}, {}, None, 0) is sys.modules["hflpkg"]
assert __import__("hflpkg.a") is sys.modules["hflpkg"]
assert __import__("hflpkg.a", {}, {}, ["A"], 0) is sys.modules["hflpkg.a"]
# A name with no dot answers itself either way.
assert __import__("hflpkg", {}, {}, [], 0) is sys.modules["hflpkg"]

# `from pkg import name` reads the package's `__name__` once, for the submodule
# lookup and the ImportError that follows it alike. A module whose `__name__`
# is a property counts the reads, and a second one is observable to any
# `__getattr__`- or descriptor-backed name.
class CountingModule(types.ModuleType):
    reads = 0

    @property
    def __name__(self):
        type(self).reads += 1
        return "pfcount"

    @__name__.setter
    def __name__(self, value):
        pass


counting = CountingModule("pfcount")
sys.modules["pfcount"] = counting
try:
    from pfcount import absent
except ImportError:
    pass
assert CountingModule.reads == 1, CountingModule.reads

# ... and the successful lookup reads it once too.
CountingModule.reads = 0
piece2 = types.ModuleType("pfcount.piece2")
sys.modules["pfcount.piece2"] = piece2
from pfcount import piece2 as got_piece2

assert got_piece2 is piece2
assert CountingModule.reads == 1, CountingModule.reads

# A module name is WTF-8 and may hold a lone surrogate. Neither the
# `sys.modules` key the lookup builds nor the ImportError message may reach an
# accessor that rejects one.
surrogate = types.ModuleType("pfsur")
surrogate.__name__ = "pf\ud800sur"
sys.modules["pfsur"] = surrogate
try:
    from pfsur import nothing
except ImportError as exc:
    assert exc.name == "pf\ud800sur", exc.name
else:
    raise AssertionError("expected ImportError")

# The same name resolves a submodule registered under it.
deep = types.ModuleType("pf\ud800sur.deep")
sys.modules["pf\ud800sur.deep"] = deep
from pfsur import deep as got_deep

assert got_deep is deep


# A second package, so the rows below start from a fromlist state the ones
# above have not already resolved.
os.mkdir(os.path.join(root, "hflpkg2"))
for name, body in [
    ("__init__.py", "__all__ = ['a']\n"),
    ("a.py", "A = 1\n"),
    ("b.py", "B = 2\n"),
]:
    with open(os.path.join(root, "hflpkg2", name), "w") as f:
        f.write(body)


def drop(prefix):
    for key in [k for k in sys.modules if k == prefix or k.startswith(prefix + ".")]:
        del sys.modules[key]


# `if not fromlist` is one truth test, and a stateful `__bool__` counts them.
# A fast path that tests the list before consulting the module cache tests it
# again in whichever importer it then falls through to, so the cold call --
# the one that misses the cache -- is the row that separates them.
class CountingList(list):
    calls = 0

    def __bool__(self):
        type(self).calls += 1
        return list.__len__(self) != 0


__import__("hflpkg2", {}, {}, CountingList(["a"]), 0)
assert CountingList.calls == 1, CountingList.calls
CountingList.calls = 0
__import__("hflpkg2", {}, {}, CountingList(["a"]), 0)
assert CountingList.calls == 1, CountingList.calls


# `elif x == '*'` is a comparison the item answers, not a read of its text: a
# `str` subclass whose `__eq__` denies `'*'` is an ordinary name and leaves
# `__all__` unexpanded.
class NotStar(str):
    def __eq__(self, other):
        return False

    def __hash__(self):
        return str.__hash__(self)


drop("hflpkg2")
pkg2 = __import__("hflpkg2", {}, {}, [NotStar("*")], 0)
assert not hasattr(pkg2, "a"), "a denying __eq__ still expanded __all__"

# `from_name = f'{module.__name__}.{x}'` formats both operands, so a
# `__format__` override on the item names the module that gets imported ...
class FormatsToB(str):
    def __format__(self, spec):
        return "b"


drop("hflpkg2")
__import__("hflpkg2", {}, {}, [FormatsToB("zzz")], 0)
assert "hflpkg2.b" in sys.modules, "the item's __format__ did not build the name"


# ... and one that raises propagates instead of being swallowed the way a
# missing submodule is.
class FormatRaises(str):
    def __format__(self, spec):
        raise ValueError("boom")


drop("hflpkg2")
try:
    __import__("hflpkg2", {}, {}, [FormatRaises("zzz")], 0)
except ValueError as exc:
    assert str(exc) == "boom", str(exc)
else:
    raise AssertionError("expected ValueError")


# The package's own `__name__` is formatted by the same f-string.
class FormatsToPkg(str):
    def __format__(self, spec):
        return "hflpkg2"


drop("hflpkg2")
import hflpkg2

hflpkg2.__name__ = FormatsToPkg("wrongname")
__import__("hflpkg2", {}, {}, ["b"], 0)
assert "hflpkg2.b" in sys.modules, "the __name__ override was not formatted"

sys.path.remove(root)
shutil.rmtree(root, ignore_errors=True)

def handle_fromlist_native_importer():
    import os
    import shutil
    import sys
    import tempfile

    root = tempfile.mkdtemp(prefix="pyre_hfn_")
    os.mkdir(os.path.join(root, "hfnpkg"))
    for name, body in [
        ("__init__.py", "__all__ = ['a', 'b']\n"),
        ("a.py", "A = 1\n"),
        ("b.py", "B = 2\n"),
        ("c.py", "C = 3\n"),
    ]:
        with open(os.path.join(root, "hfnpkg", name), "w") as f:
            f.write(body)
    sys.path.insert(0, root)

    # Everything the rows below need is imported before the switch, so the native
    # importer is asked only for `hfnpkg`.
    saved_bootstrap = sys.modules.pop("importlib._bootstrap", None)


    def drop():
        for key in [k for k in sys.modules if k == "hfnpkg" or k.startswith("hfnpkg.")]:
            del sys.modules[key]


    # `*` expands `__all__`; a submodule outside it stays unimported.
    drop()
    pkg = __import__("hfnpkg", {}, {}, ["*"], 0)
    assert hasattr(pkg, "a") and hasattr(pkg, "b"), "star did not import __all__"
    assert not hasattr(pkg, "c"), "star imported a name outside __all__"

    # `elif x == '*'` is a comparison the item answers: a `str` subclass whose
    # `__eq__` denies `'*'` is an ordinary name, whatever its text reads.
    class NotStar(str):
        def __eq__(self, other):
            return False

        def __hash__(self):
            return str.__hash__(self)


    drop()
    pkg = __import__("hfnpkg", {}, {}, [NotStar("*")], 0)
    assert not hasattr(pkg, "a"), "a denying __eq__ still expanded __all__"

    # `from_name = f'{module.__name__}.{x}'` formats the item, so a `__format__`
    # override names the module that gets imported ...
    class FormatsToB(str):
        def __format__(self, spec):
            return "b"


    drop()
    __import__("hfnpkg", {}, {}, [FormatsToB("zzz")], 0)
    assert "hfnpkg.b" in sys.modules, "the item's __format__ did not build the name"

    # ... and one that raises propagates instead of being swallowed the way a
    # missing submodule is.
    class FormatRaises(str):
        def __format__(self, spec):
            raise ValueError("boom")


    drop()
    try:
        __import__("hfnpkg", {}, {}, [FormatRaises("zzz")], 0)
    except ValueError as exc:
        assert str(exc) == "boom", str(exc)
    else:
        raise AssertionError("expected ValueError")

    # The package's own `__name__` is formatted by the same f-string.
    class FormatsToPkg(str):
        def __format__(self, spec):
            return "hfnpkg"


    drop()
    pkg = __import__("hfnpkg", {}, {}, [], 0)
    pkg.__name__ = FormatsToPkg("wrongname")
    __import__("hfnpkg", {}, {}, ["b"], 0)
    assert "hfnpkg.b" in sys.modules, "the __name__ override was not formatted"

    # `if not fromlist` is one truth test wherever the call lands.
    class CountingList(list):
        calls = 0

        def __bool__(self):
            type(self).calls += 1
            return list.__len__(self) != 0


    drop()
    __import__("hfnpkg", {}, {}, CountingList(["a"]), 0)
    assert CountingList.calls == 1, CountingList.calls
    CountingList.calls = 0
    __import__("hfnpkg", {}, {}, CountingList(["a"]), 0)
    assert CountingList.calls == 1, CountingList.calls

    # A name that is neither an attribute nor an importable submodule is swallowed;
    # an explicit `None` block is not.
    drop()
    assert __import__("hfnpkg", {}, {}, ["zzz"], 0).__name__ == "hfnpkg"
    sys.modules["hfnpkg.blocked"] = None
    try:
        __import__("hfnpkg", {}, {}, ["blocked"], 0)
    except ImportError as exc:
        assert "hfnpkg.blocked" in str(exc), str(exc)
    else:
        raise AssertionError("expected ImportError")
    del sys.modules["hfnpkg.blocked"]

    # A non-str item names where it was found.
    drop()
    try:
        __import__("hfnpkg", {}, {}, [1], 0)
    except TypeError as exc:
        assert str(exc) == "Item in ``from list'' must be str, not int", str(exc)
    else:
        raise AssertionError("expected TypeError")

    if saved_bootstrap is not None:
        sys.modules["importlib._bootstrap"] = saved_bootstrap
    sys.path.remove(root)
    shutil.rmtree(root, ignore_errors=True)
handle_fromlist_native_importer()
print("OK")
