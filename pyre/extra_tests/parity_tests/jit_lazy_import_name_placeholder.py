# pyre-check: pypy-diverges: PyPy 3.11 has no PEP 810 lazy imports.
# CPython-suite gap: test_lazy_import does not compile a module-level IMPORT_NAME
# / IMPORT_FROM loop under sys.set_lazy_imports("all") at a low JIT threshold.
# parity-tests reason: traced IMPORT_NAME with flags bit 2 clear on a
# !CO_NEWLOCALS code object must take eval_lazy_import_name, and IMPORT_FROM of
# a placeholder must take lazy_import_from. Inspect via globals(); LOAD_NAME
# reifies.

import os
import sys
import tempfile
import types

try:
    import pypyjit

    pypyjit.set_param("threshold=1,function_threshold=1")
except ImportError:
    pass

td = tempfile.mkdtemp(prefix="pyre_jit_lazy_")
helper = "pyre_jit_lazy_placeholder"
eager = "pyre_jit_eager_fnmod"
for name, body in (
    (helper, "MARKER = 1\n"),
    (eager, "MARKER = 2\n"),
):
    with open(os.path.join(td, name + ".py"), "w", encoding="utf-8") as f:
        f.write(body)
sys.path.insert(0, td)
sys.set_lazy_imports("all")

# A surrounding try would compile IMPORT_NAME with the forced-eager bit
# (PEP 810: imports in try/except stay eager so ImportError is catchable).
for _ in range(2000):
    import pyre_jit_lazy_placeholder as alias
    from pyre_jit_lazy_placeholder import MARKER as m

g = globals()
assert type(g["alias"]) is types.LazyImportType, type(g["alias"])
assert type(g["m"]) is types.LazyImportType, type(g["m"])
assert helper not in sys.modules
assert helper in sys.lazy_modules


def function_level():
    import pyre_jit_eager_fnmod as fmod

    return type(fmod).__name__


kind = None
for _ in range(2000):
    kind = function_level()
assert kind == "module", kind
assert eager in sys.modules

sys.path.remove(td)
sys.set_lazy_imports("normal")
sys.lazy_modules.discard(helper)
sys.lazy_modules.discard(helper + ".MARKER")
sys.modules.pop(helper, None)
sys.modules.pop(eager, None)

print("OK")
