# CPython-suite gap: `test_exceptions` deletes `__cause__` / `__context__` /
# `__traceback__` and never `__suppress_context__`, and only writes True/False
# to it. It only ever `add_note`s onto an instance that has no `__notes__`
# yet, so a class-level list being the object that grows is invisible. It
# reads `StopIteration.value` on a freshly constructed instance and never
# calls `__init__` again with no arguments.
#
# parity-tests reason: `__suppress_context__` is a `T_BOOL` member, so
# `PyMember_SetOne` refuses a delete with `can't delete numeric/char
# attribute` and a non-bool store with `attribute value type must be bool`;
# `descr_delsuppresscontext` / `descr_setsuppresscontext` word both
# differently. `BaseException_add_note_impl` looks `__notes__` up with
# `PyObject_GetOptionalAttr` and `PyList_Append`s the list it found, so a
# class attribute is that list; `W_BaseException.descr_add_note` goes through
# `getdict` and allocates an instance list. `StopIteration_init` always
# `Py_CLEAR`s `value` and then stores the first positional argument or
# `None`; `W_StopIteration.descr_init` writes `w_value` only when `args_w` is
# non-empty.
#
# pyre-check: pypy-diverges: pypy3 reports `__suppress_context__ may not be
# deleted` and `expected integer, got str object`, leaves a class-level
# `__notes__` list alone (`inst.__notes__ == ['n']`), and keeps
# `e.value is 1` after `StopIteration(1).__init__()`. pypy3 wraps the
# `__set_name__` error in RuntimeError (typeobject.py `_set_names`) instead
# of adding a note on the ValueError.
# test.test_enum TestSpecial.test_init_exception raises from a member
# constructor during class creation; type_set_names attaches an add_note and
# intern/list-append may collect while intern WEAKDICTENTRYARRAY is still young.
# `_rweakvaldict.py ll_set_nonnull` keeps `d` a livevar across `weakref_create`;
# `framework.py` `push_roots` / `shadowstack.py` `expand_pop_roots` reloads the
# OperationError handle after `to_exc_object` / `w_str_new_managed`
# (`add_internal_exception_note`).
# parity-env: PYPY_GC_NURSERY=4096
import gc


def rejects(fn, message):
    try:
        fn()
    except TypeError as err:
        assert str(err) == message, err
    else:
        raise AssertionError("expected TypeError")


def del_suppress_context(e):
    del e.__suppress_context__


def del_cause(e):
    del e.__cause__


def store_suppress_context(e):
    e.__suppress_context__ = "yes"


e = ValueError()
rejects(lambda: del_suppress_context(e), "can't delete numeric/char attribute")
rejects(lambda: del_cause(e), "__cause__ may not be deleted")
rejects(lambda: store_suppress_context(e), "attribute value type must be bool")


class E(ValueError):
    __notes__ = ["class"]


inst = E()
inst.add_note("n")
assert inst.__notes__ is E.__notes__, (inst.__notes__, E.__notes__)
assert inst.__notes__ == ["class", "n"], inst.__notes__

e = StopIteration(1)
e.__init__()
assert e.value is None, e.value
assert e.args == (), e.args

e = StopIteration(1)
e.__init__(2)
assert e.value == 2, e.value
assert e.args == (2,), e.args


class Desc:
    def __set_name__(self, owner, name):
        raise ValueError("boom")


def set_name_notes_stay_live_across_small_nursery_collections():
    caught = 0
    junk = []
    for i in range(80):
        junk.append([object() for _ in range(32)])
        if i % 8 == 0:
            gc.collect()
        try:

            class C:
                x = Desc()
        except ValueError as e:
            caught += 1
            notes = getattr(e, "__notes__", None)
            if not notes:
                raise AssertionError(("missing __notes__", i, e))
            joined = " ".join(str(n) for n in notes)
            if "Error calling __set_name__" not in joined:
                raise AssertionError(("note text", i, notes))
        else:
            raise AssertionError(("expected ValueError", i))

    if caught != 80:
        raise AssertionError(("caught", caught))


set_name_notes_stay_live_across_small_nursery_collections()
print("OK")
