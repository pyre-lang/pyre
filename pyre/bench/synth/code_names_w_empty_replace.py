# pyre-check: selfcheck
# pyre-check: selfcheck-interpreted
# pyre-check: requires-modules=gc
# `pycode.py PyCode.__init__` keeps `co_names_w` as one managed list, empty
# when there are no names, never null. `code.replace()` and `CodeType(...)`
# must therefore succeed for `(lambda: 1).__code__` (empty names) and for a
# non-empty `co_names` override, including after a collection that traces the
# table.
import gc
import sys
import types


def collect():
    gc.collect()


def check_empty():
    co = (lambda: 1).__code__
    assert co.co_names == ()
    replaced = co.replace()
    collect()
    assert replaced.co_names == ()
    CodeType = type(co)
    rebuilt = CodeType(
        co.co_argcount,
        co.co_posonlyargcount,
        co.co_kwonlyargcount,
        co.co_nlocals,
        co.co_stacksize,
        co.co_flags,
        co.co_code,
        co.co_consts,
        (),
        co.co_varnames,
        co.co_filename,
        co.co_name,
        co.co_qualname,
        co.co_firstlineno,
        co.co_linetable,
        co.co_exceptiontable,
        co.co_freevars,
        co.co_cellvars,
    )
    collect()
    assert rebuilt.co_names == ()


def named():
    return a + b


def check_nonempty():
    co = named.__code__
    assert set(co.co_names) == {"a", "b"}
    interned_x = sys.intern("x")
    interned_y = sys.intern("y")
    replaced = co.replace(co_names=(interned_x, interned_y))
    collect()
    assert replaced.co_names == ("x", "y")
    assert replaced.co_names[0] is interned_x
    assert replaced.co_names[1] is interned_y
    CodeType = types.CodeType
    rebuilt = CodeType(
        co.co_argcount,
        co.co_posonlyargcount,
        co.co_kwonlyargcount,
        co.co_nlocals,
        co.co_stacksize,
        co.co_flags,
        co.co_code,
        co.co_consts,
        (interned_x, interned_y),
        co.co_varnames,
        co.co_filename,
        co.co_name,
        co.co_qualname,
        co.co_firstlineno,
        co.co_linetable,
        co.co_exceptiontable,
        co.co_freevars,
        co.co_cellvars,
    )
    collect()
    assert rebuilt.co_names == ("x", "y")
    assert rebuilt.co_names[0] is interned_x
    assert rebuilt.co_names[1] is interned_y


check_empty()
check_nonempty()
print("PASS")
