# pyre-check: pypy-diverges: pins heap-type special-attr deletes as TypeError "cannot delete '__bases__' attribute of type 'D'"; pypy3 raises AttributeError "cannot delete '__bases__' attribute of immutable type 'D'"
# CPython-suite gap: test_type checks that some special type attributes
# cannot be deleted, without pinning the "of type" versus "immutable type"
# wording or the getset `__delete__` path.
# parity-tests reason: `check_set_special_type_attr` / `type_setattro`:
# heap-type deletes say "of type", a static type's `del` says "cannot set
# ... of immutable type", and a getset `__delete__` still uses the delete
# wording.

"""Special type-attribute deletes report the heap versus static wording."""


class C:
    pass


class D(C):
    pass


class X:
    "elephant"


X.__doc__ = "banana"


def expect(exc, text):
    got = str(exc)
    if text not in got:
        raise AssertionError(repr(got))


for name in ("__bases__", "__doc__", "__name__", "__qualname__", "__module__"):
    try:
        delattr(D, name)
    except TypeError as e:
        expect(e, "cannot delete '%s' attribute of type 'D'" % name)
    else:
        raise AssertionError("del D.%s succeeded" % name)

try:
    type(X).__dict__["__doc__"].__delete__(X)
except TypeError as e:
    expect(e, "cannot delete '__doc__' attribute of type 'X'")
else:
    raise AssertionError("X.__doc__ __delete__ succeeded")
if X.__doc__ != "banana":
    raise AssertionError(X.__doc__)

try:
    del list.__doc__
except TypeError as e:
    expect(e, "cannot set '__doc__' attribute of immutable type 'list'")
else:
    raise AssertionError("del list.__doc__ succeeded")

try:
    del int.__bases__
except TypeError as e:
    expect(e, "cannot set '__bases__' attribute of immutable type 'int'")
else:
    raise AssertionError("del int.__bases__ succeeded")

try:
    type.__dict__["__module__"].__delete__(D)
except TypeError as e:
    expect(e, "cannot delete '__module__' attribute of type 'D'")
else:
    raise AssertionError("D.__module__ __delete__ succeeded")

try:
    type.__dict__["__doc__"].__delete__(list)
except TypeError as e:
    expect(e, "cannot delete '__doc__' attribute of type 'list'")
else:
    raise AssertionError("list.__doc__ __delete__ succeeded")

try:
    del D.__annotate__
except TypeError as e:
    expect(e, "cannot delete __annotate__ attribute")
else:
    raise AssertionError("del D.__annotate__ succeeded")

try:
    del D.__mro__
except AttributeError as e:
    expect(e, "attribute '__mro__' of 'type' objects is not writable")
else:
    raise AssertionError("del D.__mro__ succeeded")

print("OK")
