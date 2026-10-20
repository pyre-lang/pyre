# pyre-check: pypy-diverges: PyPy 3.11 has no PEP 810 lazy imports.
# CPython-suite gap: Lib/test/test_lazy_import/ is large and uses the CPython
# test package layout; this file pins the public runtime surface the 3.15
# stdlib and a module-level `lazy import` actually need.
# parity-tests reason: pin sys APIs, SyntaxError text, placeholder repr,
# dict-vs-name reify, and types.LazyImportType against the 3.15.0 oracle.

import ast
import os
import sys
import tempfile
import types

assert sys.get_lazy_imports() == "normal"
assert sys.get_lazy_imports_filter() is None
assert type(sys.lazy_modules) is set
assert sys.flags.lazy_imports == -1
assert sys.flags.n_fields == 22
assert sys.flags.n_sequence_fields == 18
assert len(sys.flags) == 18

assert types.LazyImportType.__name__ == "lazy_import"
assert types.LazyImportType.__module__ == "builtins"

assert ImportCycleError.__bases__ == (ImportError,)
assert callable(__lazy_import__)

try:
    sys.set_lazy_imports(1)
except TypeError as e:
    assert str(e) == "mode must be a string: 'normal' or 'all'", e
else:
    raise AssertionError("non-str mode accepted")

try:
    sys.set_lazy_imports("eager")
except ValueError as e:
    assert str(e) == "mode must be 'normal' or 'all'", e
else:
    raise AssertionError("invalid mode accepted")

try:
    sys.set_lazy_imports_filter(1)
except ValueError as e:
    assert str(e) == "filter provided but is not callable", e
else:
    raise AssertionError("non-callable filter accepted")

sys.set_lazy_imports("normal")
sys.set_lazy_imports_filter(None)
assert sys.get_lazy_imports() == "normal"
assert sys.get_lazy_imports_filter() is None

assert ast.Import._fields[-1] == "is_lazy"
assert "is_lazy" in ast.ImportFrom._fields
assert ast.parse("lazy import os").body[0].is_lazy == 1
assert ast.parse("import os").body[0].is_lazy == 0

for src, msg in (
    ("def f():\n    lazy import os\n", "lazy import not allowed inside functions"),
    ("class C:\n    lazy import os\n", "lazy import not allowed inside classes"),
    (
        "try:\n    lazy import os\nexcept Exception:\n    pass\n",
        "lazy import not allowed inside try/except blocks",
    ),
    ("lazy from os import *\n", "lazy from ... import * is not allowed"),
    (
        "lazy from __future__ import annotations\n",
        "lazy from __future__ import is not allowed",
    ),
    (
        "from os lazy import path\n",
        "use 'lazy from ... ' instead of 'from ... lazy import'",
    ),
    (
        "async def f():\n    lazy import os\n",
        "lazy import not allowed inside functions",
    ),
):
    try:
        compile(src, "<s>", "exec")
    except SyntaxError as e:
        assert e.msg == msg, (src, e.msg)
    else:
        raise AssertionError(f"accepted: {src!r}")

compile("if 1:\n    lazy import os\n", "<s>", "exec")


def _exec_in_func():
    exec("lazy import os")


try:
    _exec_in_func()
except SyntaxError as e:
    assert e.msg == "'lazy import' is only allowed at module level", e.msg
else:
    raise AssertionError("exec lazy import inside a function succeeded")

td = tempfile.mkdtemp()
uid = os.urandom(8).hex()
helper = f"c2lazy_h_{uid}"
user = f"c2lazy_u_{uid}"
user_from = f"c2lazy_f_{uid}"
user_load = f"c2lazy_n_{uid}"
user_resolve = f"c2lazy_r_{uid}"
with open(os.path.join(td, helper + ".py"), "w", encoding="utf-8") as f:
    f.write("MARKER = 1\n")
with open(os.path.join(td, user + ".py"), "w", encoding="utf-8") as f:
    f.write(
        f"lazy import {helper} as alias\n"
        "g = globals()\n"
        "KIND = type(g['alias']).__name__\n"
        "REPR = repr(g['alias'])\n"
        "IS_TYPE = type(g['alias']) is __import__('types').LazyImportType\n"
    )
with open(os.path.join(td, user_from + ".py"), "w", encoding="utf-8") as f:
    f.write(
        f"lazy from {helper} import MARKER as m\n"
        "g = globals()\n"
        "KIND = type(g['m']).__name__\n"
        "REPR = repr(g['m'])\n"
    )
with open(os.path.join(td, user_load + ".py"), "w", encoding="utf-8") as f:
    f.write(f"lazy import {helper} as alias\nKIND = type(alias).__name__\n")
with open(os.path.join(td, user_resolve + ".py"), "w", encoding="utf-8") as f:
    f.write(
        f"lazy import {helper} as alias\n"
        "resolved = globals()['alias'].resolve()\n"
        "RESOLVED_TYPE = type(resolved).__name__\n"
        "RESOLVED_MARKER = resolved.MARKER\n"
    )

sys.path.insert(0, td)
try:
    u = __import__(user)
    assert u.KIND == "lazy_import", u.KIND
    assert u.REPR == f"<lazy_import '{helper}'>", u.REPR
    assert u.IS_TYPE is True
    assert helper not in sys.modules
    assert helper in sys.lazy_modules

    uf = __import__(user_from)
    assert uf.KIND == "lazy_import", uf.KIND
    assert uf.REPR == f"<lazy_import '{helper}.MARKER'>", uf.REPR
    assert helper not in sys.modules

    ur = __import__(user_resolve)
    assert ur.RESOLVED_TYPE == "module"
    assert ur.RESOLVED_MARKER == 1
    assert helper in sys.modules

    ul = __import__(user_load)
    assert ul.KIND == "module"
    assert helper in sys.modules
finally:
    sys.path.remove(td)
    sys.lazy_modules.discard(helper)
    for name in (user, user_from, user_load, user_resolve, helper):
        sys.modules.pop(name, None)

print("OK")
