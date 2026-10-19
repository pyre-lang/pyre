# CPython-suite gap: `test_exceptions` pickles an exception built by the
# constructor. It does not materialise an empty instance dict and then read
# `__reduce__`.
#
# parity-tests reason: pickle replays the tuple `__reduce__` returns. When
# the instance dict already exists, that tuple's state is the dict itself,
# so a later write shows up in the state. A missing empty dict, or a copy
# standing in for it, rebuilds from a different object than the one the
# caller holds.
#
# pyre-check: pypy-diverges: `W_BaseException.descr_reduce` and
# `W_OSError.descr_reduce` append `w_dict` only when `space.is_true` is
# true, so an empty dict stays a 2-tuple. `W_ImportError.descr_reduce`
# copies a truthy dict and skips `is_w(None)`. `W_AttributeError.descr_reduce`
# omits a falsy dict and publishes no `__getstate__`.
#
# exception_member_delete_drops_reduce_slots:
# `test_exceptions` pickles a constructed ImportError or AttributeError and never deletes
# `name` afterwards. `ImportError_getstate` copies every non-NULL slot, explicit `None`
# included. `AttributeError_getstate` copies a non-NULL `name` the same way.
# `PyMember_SetOne` clears `_Py_T_OBJECT` to NULL, so `del e.name` drops the key from the
# reduce state while `e.name = None` keeps it. `readwrite_attrproperty_w` installs no
#
# exception_setstate_setattr_leaves_omitted_slots:
# `test_exceptions` pickles a fully-populated ImportError and never calls `__setstate__`
# with a partial dict, so an omitted path or a mutated state argument is invisible there.
# `BaseException___setstate___impl` setattr's each present key and leaves the others.
# `W_ImportError.descr_setstate` pops name/path/name_from with a None default, so an
# omitted path becomes None and the caller's state dict loses those keys. OSError has no
def reduced(exc):
    out = exc.__reduce__()
    assert len(out) == 3, (type(exc), out)
    return out


def replay(exc):
    out = exc.__reduce__()
    built = out[0](*out[1])
    if len(out) == 3:
        built.__setstate__(out[2])
    return built


bare = ValueError(1)
assert len(bare.__reduce__()) == 2, bare.__reduce__()
bare = ImportError("m")
assert len(bare.__reduce__()) == 2, bare.__reduce__()

exc = ValueError(1)
exc.__dict__
out = reduced(exc)
assert out[0] is ValueError
assert out[1] == (1,)
assert out[2] is exc.__dict__
assert out[2] == {}
built = replay(exc)
assert built.args == (1,)
assert built.__dict__ == {}

exc = ValueError("m")
exc.k = 1
del exc.k
out = reduced(exc)
assert out[2] is exc.__dict__
assert out[2] == {}

exc = ValueError("m")
exc.__dict__ = {}
assert reduced(exc)[2] is exc.__dict__

exc = ValueError("m")
exc.k = 9
assert reduced(exc)[2] is exc.__dict__
built = replay(exc)
assert built.args == ("m",)
assert built.k == 9


class E(ValueError):
    pass


exc = E("m")
exc.__dict__
out = reduced(exc)
assert out[0] is E
assert out[2] is exc.__dict__
assert out[2] == {}

for make in (
    lambda: SyntaxError("m"),
    lambda: UnicodeDecodeError("utf-8", b"a", 0, 1, "r"),
    lambda: SystemExit(1),
    lambda: StopIteration(1),
    lambda: KeyError("k"),
    lambda: MemoryError(),
    lambda: BaseException(),
    lambda: ExceptionGroup("g", [ValueError(1)]),
):
    exc = make()
    exc.__dict__
    out = reduced(exc)
    assert out[0] is type(exc), type(exc)
    assert out[2] is exc.__dict__
    assert out[2] == {}

exc = NameError("n", name="v")
assert exc.name == "v"
exc.__dict__
out = reduced(exc)
assert out[2] is exc.__dict__
assert out[2] == {}
assert exc.name == "v"

exc = OSError(2, "m", "a")
exc.__dict__
out = reduced(exc)
assert out[0] is type(exc)
assert out[1] == (2, "m", "a")
assert out[2] is exc.__dict__
assert out[2] == {}
built = replay(exc)
assert built.filename == "a"
assert built.args == (2, "m")
assert built.__dict__ == {}

exc = FileNotFoundError(2, "m", "a")
exc.note = "keep"
out = reduced(exc)
assert out[0] is FileNotFoundError
assert out[1] == (2, "m", "a")
assert out[2] is exc.__dict__
assert out[2] == {"note": "keep"}

exc = BlockingIOError(11, "again", 4)
assert exc.characters_written == 4
exc.__dict__
out = reduced(exc)
assert out[1] == (11, "again", 4)
assert out[2] is exc.__dict__
assert out[2] == {}
assert "characters_written" not in out[2]
built = replay(exc)
assert built.characters_written == 4

exc = ImportError("m")
exc.__dict__
out = reduced(exc)
assert out[2] is exc.__dict__
assert out[2] == {}

exc = ImportError("m")
exc.k = 9
out = reduced(exc)
assert out[2] is exc.__dict__
assert out[2] == {"k": 9}
built = replay(exc)
assert built.args == ("m",)
assert built.k == 9

exc = ImportError("m", name="n")
holder = exc.__dict__
out = reduced(exc)
assert out[2] is not holder
assert holder == {}
assert out[2] == {"name": "n"}
built = replay(exc)
assert built.name == "n"
assert built.args == ("m",)
assert built.__dict__ == {}

exc = ImportError("nope", name=None)
exc.__dict__
out = reduced(exc)
assert out[2] is not exc.__dict__
assert out[2] == {"name": None}

exc = ModuleNotFoundError("m")
exc.__dict__
out = reduced(exc)
assert out[0] is ModuleNotFoundError
assert out[2] is exc.__dict__
assert out[2] == {}

exc = AttributeError("m")
blank = exc.__dict__
out = reduced(exc)
assert out[2] is not blank
assert blank == {}
assert out[2] == {"args": ("m",)}
state = exc.__getstate__()
assert state == {"args": ("m",)}
assert state is not blank

exc = AttributeError("m", name="a")
exc.k = 1
out = reduced(exc)
assert out[2] is not exc.__dict__
assert out[2] == {"k": 1, "name": "a", "args": ("m",)}


class D(dict):
    pass


exc = ValueError(1)
exc.__dict__ = D()
out = reduced(exc)
assert out[2] is exc.__dict__
assert type(out[2]) is D
assert out[2] == {}

exc = ValueError(1)
exc.__dict__ = D(a=1)
out = reduced(exc)
assert out[2] is exc.__dict__
assert out[2] == {"a": 1}

exc = ImportError("m")
exc.__dict__ = D()
out = reduced(exc)
assert out[2] is exc.__dict__
assert type(out[2]) is D

exc = ImportError("m", name="n")
holder = D(a=1)
exc.__dict__ = holder
out = reduced(exc)
assert out[2] is not holder
assert type(out[2]) is dict
assert out[2] == {"a": 1, "name": "n"}
assert holder == {"a": 1}

exc = ImportError("m", name="n")
exc.__dict__ = D()
out = reduced(exc)
assert out[2] is not exc.__dict__
assert type(out[2]) is dict
assert out[2] == {"name": "n"}
assert exc.__dict__ == {}

def exception_member_delete_drops_reduce_slots():
    def import_error_deleted_name_drops_from_state():
        exc = ImportError("m", name="n", path="p")
        del exc.name
        assert exc.name is None
        state = exc.__reduce__()[2]
        assert state == {"path": "p"}, state


    def import_error_assigned_none_name_stays_in_state():
        exc = ImportError("m", name="n", path="p")
        exc.name = None
        state = exc.__reduce__()[2]
        assert state == {"name": None, "path": "p"}, state


    def import_error_deleted_path_drops_from_state():
        exc = ImportError("m", name="n", path="p")
        del exc.path
        state = exc.__reduce__()[2]
        assert state == {"name": "n"}, state


    def attribute_error_deleted_name_drops_from_state():
        exc = AttributeError("m", name="n")
        del exc.name
        assert exc.name is None
        state = exc.__reduce__()[2]
        assert "name" not in state, state
        assert state["args"] == ("m",)


    def attribute_error_assigned_none_name_stays_in_state():
        exc = AttributeError("m", name="n")
        exc.name = None
        state = exc.__reduce__()[2]
        assert state["name"] is None, state


    import_error_deleted_name_drops_from_state()
    import_error_assigned_none_name_stays_in_state()
    import_error_deleted_path_drops_from_state()
    attribute_error_deleted_name_drops_from_state()
    attribute_error_assigned_none_name_stays_in_state()
exception_member_delete_drops_reduce_slots()

def exception_setstate_setattr_leaves_omitted_slots():
    def import_error_omitted_path_stays():
        exc = ImportError("m", name="n", path="p")
        state = {"name": "z", "extra": 1}
        exc.__setstate__(state)
        assert exc.name == "z", exc.name
        assert exc.path == "p", exc.path
        assert exc.__dict__ == {"extra": 1}, exc.__dict__
        assert state == {"name": "z", "extra": 1}, state


    def import_error_setstate_is_base():
        assert ImportError.__setstate__ is BaseException.__setstate__
        assert ModuleNotFoundError.__setstate__ is BaseException.__setstate__


    def oserror_setstate_writes_errno_slot():
        exc = OSError(2, "m")
        exc.__setstate__({"errno": 9})
        assert exc.errno == 9, exc.errno
        assert str(exc) == "[Errno 9] m", str(exc)
        assert exc.args == (2, "m"), exc.args
        assert exc.__dict__ == {}, exc.__dict__


    import_error_omitted_path_stays()
    import_error_setstate_is_base()
    oserror_setstate_writes_errno_slot()
exception_setstate_setattr_leaves_omitted_slots()
print("OK")
