# pyre-check: pypy-diverges: PyPy 3.11's WarningMessage has no `module` field (3.15 adds it).
# CPython-suite gap: test_warnings records WarningMessage through the pure
# Python warnings module and does not compile a hot native show_warning loop.
# parity-tests reason: the show_warning residual must pass the 8th `module`
# argument the interpreter passes at do_warn_explicit / call_show_warning.

import warnings

try:
    import pypyjit

    pypyjit.set_param("threshold=1,function_threshold=1")
except ImportError:
    pass


def warn_once():
    warnings.warn("jit-show-warning-module", UserWarning)


with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    for _ in range(4000):
        warn_once()
    assert caught, "no warning recorded"
    last = caught[-1]
    assert str(last.message) == "jit-show-warning-module", last.message
    assert last.category is UserWarning, last.category
    assert last.filename == __file__, last.filename
    assert last.module == "__main__", last.module

print("OK")
