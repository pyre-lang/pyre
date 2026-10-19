# CPython-suite gap: weakref tests omit this PyPy lifeline across moving collection.
# parity-tests reason: this targets PyPy/pyre weakref ownership in the moving GC.
#
# make_weakref_descr_lifeline:
# no CPython test pins make_weakref_descr _lifeline_ on these interp types. typedef.py
# make_weakref_descr stores _lifeline_ on the instance.
#
# weakref_ref_subclass_layout:
# weakref subclass tests do not combine callbacks with all storage shapes. this guards the
# translated W_Weakref payload and subclass carrier layout.

"""Weakref lifelines remain owned by the GC across collection.

PyPy stores each rweakref inside the GC-traced W_Weakref/WeakrefLifeline
object.  Fill the nursery while both the referents and their weakref objects
are live, then finalize them together; callbacks allocate while the detached
lifeline callback path is active.
"""

import gc
import weakref


class Target:
    pass


callback_count = 0


def callback(dead):
    global callback_count
    assert dead() is None
    callback_count += 1
    # Allocations during a callback can fill the nursery; the detached
    # lifeline must not be revisited by an intervening collection.
    [Target() for _ in range(8)]


count = 2_000
targets = [Target() for _ in range(count)]
refs = [weakref.ref(target, callback) for target in targets]

# Keep every weakref object alive while their referents die as one group.
del targets
gc.collect()

assert callback_count == count
assert all(ref() is None for ref in refs)

def make_weakref_descr_lifeline():
    """Exact instances of make_weakref_descr types accept weakref.ref.

    `typedef.py` `make_weakref_descr` publishes `__weakref__` and a `_lifeline_`
    field. `_getusercls` does not mix `MapdictWeakrefSupport` for those TypeDefs,
    so a subclass of a weakrefable base keeps the same field.
    """

    import io
    import types
    import weakref
    import _thread
    from collections import deque


    def alive(obj):
        return weakref.ref(obj)() is obj


    def f():
        yield 1


    gen = f()
    mod = types.ModuleType("make_weakref_descr_lifeline")
    lock = _thread.allocate_lock()
    dq = deque()
    bio = io.BytesIO()
    local = _thread._local()


    class SubLocal(_thread._local):
        pass


    class SubMod(types.ModuleType):
        pass


    sub_local = SubLocal()
    sub_mod = SubMod("make_weakref_descr_lifeline_sub")

    assert alive(f)
    assert alive(gen)
    assert alive(type)
    assert alive(mod)
    assert alive(lock)
    assert alive(dq)
    assert alive(bio)
    assert alive(local)
    assert alive(sub_local)
    assert alive(sub_mod)
    assert alive(set())
    assert alive(frozenset())
make_weakref_descr_lifeline()

def weakref_ref_subclass_layout():
    import gc
    from weakref import WeakValueDictionary, ref


    class Value:
        def __init__(self, value):
            self.value = value

        def __eq__(self, other):
            return isinstance(other, Value) and self.value == other.value

        def __hash__(self):
            return hash(self.value)


    class PlainRef(ref):
        pass


    class EmptyRef(ref):
        __slots__ = ()


    class KeyedRef(ref):
        __slots__ = ("key",)


    PRIVATE_FIELDS = ("w_obj_weak", "w_callable", "w_hash", "w_slots")


    def collect():
        for _ in range(3):
            gc.collect()


    def check_ref_subclass(ref_type):
        callbacks = []

        def callback(w_ref):
            callbacks.append(w_ref() is None)

        value = Value((ref_type.__name__, 42))
        equal_value = Value((ref_type.__name__, 42))
        w_ref = ref_type(value, callback)
        equal_ref = ref(equal_value)

        assert w_ref() is value
        assert w_ref.__callback__ is callback
        assert hash(w_ref) == hash(value)
        assert w_ref == equal_ref

        if ref_type is PlainRef:
            w_ref.marker = "plain"
            assert w_ref.__dict__ == {"marker": "plain"}
        else:
            assert not hasattr(w_ref, "__dict__")

        if ref_type is KeyedRef:
            w_ref.key = ("module", "name")
            assert w_ref.key == ("module", "name")
            w_ref.key = ("module", "renamed")
            assert w_ref.key == ("module", "renamed")
            assert w_ref() is value
            del w_ref.key
            try:
                w_ref.key
            except AttributeError:
                pass
            else:
                raise AssertionError("deleted slot remained readable")

        for name in PRIVATE_FIELDS:
            assert not hasattr(w_ref, name)
            if hasattr(w_ref, "__dict__"):
                assert name not in w_ref.__dict__

        del value
        collect()
        assert w_ref() is None
        assert callbacks == [True]


    for ref_type in (PlainRef, EmptyRef, KeyedRef):
        check_ref_subclass(ref_type)


    values = WeakValueDictionary()
    key = ("namespace", "spam")
    value = Value("live")
    values[key] = value
    assert values[key] is value
    stored_ref = values.data[key]
    assert stored_ref.key == key
    assert stored_ref() is value
    del value
    collect()
    assert key not in values
weakref_ref_subclass_layout()
print("OK")
