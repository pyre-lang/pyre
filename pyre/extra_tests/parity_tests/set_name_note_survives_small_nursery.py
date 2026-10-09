# CPython-suite gap: test.test_enum TestSpecial.test_init_exception raises
# from a member constructor during class creation; type_set_names attaches
# an add_note and intern/list-append may collect while intern
# WEAKDICTENTRYARRAY is still young.
# parity-tests reason: `_rweakvaldict.py ll_set_nonnull` keeps `d` a livevar
# across `weakref_create`; `framework.py` `push_roots` / `shadowstack.py`
# `expand_pop_roots` reloads the OperationError handle after
# `to_exc_object` / `w_str_new_managed` (`add_internal_exception_note`).
# parity-env: PYPY_GC_NURSERY=4096
# pyre-check: pypy-diverges: pypy3 wraps the __set_name__ error in
# RuntimeError via typeobject.py _set_names chain_exceptions_from_cause;
# CPython 3.14 attaches an add_note on the original ValueError.

"""__set_name__ notes stay live across small-nursery collections."""

import gc


class Desc:
    def __set_name__(self, owner, name):
        raise ValueError("boom")


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
print("OK")
