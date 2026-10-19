# CPython-suite gap: the except* corpus asks what a subgroup contains, never
# whether an unfiltered one is the group it was taken from, so nothing there
# reaches the arm this pins.
# parity-tests reason: an identity and an answered class where PyPy differs from
# CPython on purpose.
# pyre-check: pypy-diverges: pins that an unfiltered `subgroup` is rebuilt;
# `app_group.py subgroup` returns `self` there on purpose ("this is the
# difference to split!"), so pypy3 answers `eg.subgroup(cond) is eg` True and a
# `BaseExceptionGroup` subclass rather than `ExceptionGroup`.
#
# exception_group_notes_ignores_a_non_sequence_iterable:
# the except* corpus only ever gives a group a list of notes, so nothing there
# distinguishes "is a sequence" from "is iterable". the sequence test that decides whether
# `__notes__` is copied at all, where PyPy's app-level `list()` accepts more than
# `PySequence_Check` does. `app_group.py _derive_and_copy_attrs` copies `__notes__` with
# `list(self.__notes__)`, which accepts any iterable and runs its iterator, so pypy3
#
# exception_group_rejection_texts:
# `test_exception_group` constructs with positionals and never
# `ExceptionGroup(message=..., exceptions=...)`. `test_fields_are_readonly` asserts only
# the AttributeError type of a store to `message` / `exceptions`. `BaseExceptionGroup_new`
# takes two positional arguments; keywords are not bound, so the call is `__new__() takes
# exactly 2 arguments (0 given)`. `message` and `exceptions` are readonly members, so a

"""`subgroup` rebuilds its result even when the walk dropped nothing.

`exceptiongroup_subgroup` returns the group itself only when the condition
matches that group; once it walks the children it goes through `derive`
whatever the walk found.  So the result of an all-matching `subgroup` is a new
object, and for a `BaseExceptionGroup` subclass the default `derive` builds a
`BaseExceptionGroup`, which is promoted to `ExceptionGroup` because every leaf
is an `Exception`.

Both facts are observable — `is` and `type()` — and a shortcut that answers the
receiver is invisible to any test that only reads `.exceptions`.
"""


class MyEG(BaseExceptionGroup):
    """A subclass that adds nothing, so only `derive` decides the result type."""


value = ValueError("v")
kind = TypeError("t")

# The one arm that does answer the receiver: the condition matches the group.
group = MyEG("m", [value])
assert group.subgroup(MyEG) is group
assert group.subgroup(BaseExceptionGroup) is group
assert group.subgroup(lambda exc: True) is group
assert type(group.subgroup(MyEG)) is MyEG

# Every leaf matches, so the walk drops nothing -- and the result is still new.
plain = ExceptionGroup("m", [value])
taken = plain.subgroup(ValueError)
assert taken is not plain
assert taken.message == "m"
assert taken.exceptions == (value,)

# The same for a subclass, where the rebuild is visible in the class as well.
assert type(group.subgroup(ValueError)) is ExceptionGroup
assert group.subgroup(ValueError).exceptions == (value,)

# Nested: the inner group is unfiltered too, and is rebuilt with the outer.
inner = ExceptionGroup("i", [value])
outer = ExceptionGroup("o", [inner])
rebuilt = outer.subgroup(ValueError)
assert rebuilt is not outer
assert rebuilt.exceptions[0] is not inner
assert rebuilt.exceptions[0].exceptions == (value,)

# A partial match was already rebuilt before, and stays so.
mixed = MyEG("m", [value, kind])
part = mixed.subgroup(ValueError)
assert part is not mixed
assert type(part) is ExceptionGroup
assert part.exceptions == (value,)

# `split` never took the shortcut on either side.
assert plain.split(ValueError)[0] is not plain
assert plain.split(KeyError)[1] is not plain

# The rebuild runs `_derive_and_copy_attrs`, so the metadata follows it.
noted = ExceptionGroup("m", [value])
noted.add_note("n1")
assert noted.subgroup(ValueError).__notes__ == ["n1"]
assert noted.subgroup(ValueError).__notes__ is not noted.__notes__

def exception_group_notes_ignores_a_non_sequence_iterable():
    """`__notes__` is copied only when it is a *sequence*, not merely iterable.

    The gate is `PySequence_Check` — `issequence_w`, i.e. `__getitem__` on something
    that is not a mapping — and a value that fails it is ignored without being
    iterated at all.  A generator and an object carrying only `__iter__` therefore
    leave the derived group with no notes, and a raising `__iter__` never runs.

    `list()` on the same values would copy the generator and raise out of the
    `__iter__`, which is what makes this observable rather than an internal detail.
    """

    def partial(notes):
        group = ExceptionGroup("m", [ValueError(1), TypeError(2)])
        group.__notes__ = notes
        return group


    # A generator is iterable but not a sequence: no notes on the result.
    assert not hasattr(partial(iter(["g"])).subgroup(ValueError), "__notes__")
    yes, no = partial(iter(["g"])).split(ValueError)
    assert not hasattr(yes, "__notes__")
    assert not hasattr(no, "__notes__")


    # An object with only `__iter__` is not a sequence either, so the iterator is
    # never run and its error never surfaces.
    class RaisingIter:
        def __iter__(self):
            raise ValueError("iter boom")


    assert not hasattr(partial(RaisingIter()).subgroup(ValueError), "__notes__")
    yes, no = partial(RaisingIter()).split(ValueError)
    assert not hasattr(yes, "__notes__")
    assert not hasattr(no, "__notes__")

    # A mapping defines `__getitem__` but is not a sequence: also ignored.
    assert not hasattr(partial({"k": "v"}).subgroup(ValueError), "__notes__")
exception_group_notes_ignores_a_non_sequence_iterable()

def exception_group_rejection_texts():
    def rejects(fn, exc_type, message):
        try:
            fn()
        except exc_type as err:
            assert str(err) == message, err
        else:
            raise AssertionError(f"expected {exc_type.__name__}")


    def store(obj, name, value):
        setattr(obj, name, value)


    rejects(
        lambda: ExceptionGroup(message="m", exceptions=[ValueError()]),
        TypeError,
        "BaseExceptionGroup.__new__() takes exactly 2 arguments (0 given)",
    )

    eg = ExceptionGroup("m", [ValueError(1)])
    rejects(lambda: store(eg, "message", "x"), AttributeError, "readonly attribute")
    rejects(
        lambda: store(eg, "exceptions", (ValueError(1),)),
        AttributeError,
        "readonly attribute",
    )
exception_group_rejection_texts()
print("OK")
