# CPython-suite gap: type-cache tests omit hot surrogate names and tag-zero MROs.
# parity-tests reason: this targets PyPy/pyre method-cache lookup and invalidation.

"""Surrogate attribute names obey every method-cache invalidation route."""

import sys


N = 3000


def store_delete():
    name = "\udc80cache_invalidation"

    class Base:
        pass

    class Sub(Base):
        pass

    setattr(Base, name, 1)
    for _ in range(N):
        assert getattr(Sub, name) == 1
    setattr(Base, name, 2)
    assert getattr(Sub, name) == 2
    delattr(Base, name)
    try:
        getattr(Sub, name)
    except AttributeError:
        pass
    else:
        raise AssertionError("deleted surrogate attribute remained cached")


def reassign_bases():
    name = "\udc81cache_bases"

    class Base:
        pass

    class Other:
        pass

    class Sub(Base):
        pass

    setattr(Base, name, "base")
    setattr(Other, name, "other")
    for _ in range(N):
        assert getattr(Sub, name) == "base"
    Sub.__bases__ = (Other,)
    assert getattr(Sub, name) == "other"


def tag_zero_mro():
    if sys.implementation.name == "cpython":
        return
    name = "\udc82tag_zero"

    class ClassicLike:
        __bases__ = ()

    classic_like = ClassicLike()

    class Meta(type):
        def mro(cls):
            return [cls, classic_like, object]

    class Subject(metaclass=Meta):
        pass

    setattr(Subject, name, 41)
    for _ in range(N):
        assert getattr(Subject, name) == 41
    setattr(Subject, name, 42)
    assert getattr(Subject, name) == 42


def bases_assignment_rollback_restores_mro():
    import gc

    fail = False

    class Meta(type):
        def mro(cls):
            junk = [object() for _ in range(256)]
            _ = junk[-1]
            name = getattr(cls, "__name__", "")
            if fail and name == "C":
                gc.collect(0)
                raise RuntimeError("mro fail")
            return type.mro(cls)

    class A(metaclass=Meta):
        pass

    class B(A):
        pass

    class C(B):
        pass

    class D:
        pass

    try:
        fail = True
        B.__bases__ = (D,)
    except RuntimeError:
        pass
    print(
        "bases_assignment_rollback",
        [c.__name__ for c in B.__bases__],
        [c.__name__ for c in B.__mro__],
    )


store_delete()
reassign_bases()
tag_zero_mro()
bases_assignment_rollback_restores_mro()
print("OK")
