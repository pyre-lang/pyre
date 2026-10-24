# `app_abc.py SimpleWeakSet`.  The registry and both caches are
# instances of this, so `_get_dump` can hand out the `data` sets and a
# collected entry drops itself through the callback the set installs.
#
# Held at app level rather than rebuilt over the raw set primitives because
# the callback closes over a weakref to the set: the discard has to run with
# the set still reachable but no longer keeping itself alive through it.
from _weakref import ref


class SimpleWeakSet:
    def __init__(self, data=None):
        self.data = set()

        def _remove(item, selfref=ref(self)):
            self = selfref()
            if self is not None:
                self.data.discard(item)

        self._remove = _remove

    def __iter__(self):
        # Weakref callback may remove entry from set.
        # So we make a copy first.
        copy = list(self.data)
        for itemref in copy:
            item = itemref()
            if item is not None:
                yield item

    def __contains__(self, item):
        try:
            wr = ref(item)
        except TypeError:
            return False
        return wr in self.data

    def add(self, item):
        self.data.add(ref(item, self._remove))

    def clear(self):
        self.data.clear()


# `app_abc.py _abc_instancecheck`.  `get_cache_token` is the module builtin
# seeded into this file's globals; it reads the same counter `_abc_register`
# advances.  A separate Python counter would miss that bump.
def _abc_instancecheck(cls, instance):
    """Internal ABC helper for instance checks. Should be never used outside abc module."""
    # `_abc__abc_instancecheck_impl` uses `PyObject_GetOptionalAttr` for
    # `__class__` and falls back to `type(instance)` when that attribute is
    # absent (gh-153772).
    try:
        subclass = instance.__class__
    except AttributeError:
        subclass = type(instance)
    if subclass in cls._abc_cache:
        return True
    subtype = type(instance)
    if subtype is subclass:
        if (
            cls._abc_negative_cache_version == get_cache_token()
            and subclass in cls._abc_negative_cache
        ):
            return False
        return cls.__subclasscheck__(subclass)
    # `app_abc.py _abc_instancecheck` writes this arm as
    # `any(cls.__subclasscheck__(c) for c in (subclass, subtype))`.
    # That genexp lowers to a `FOR_ITER` in this function, and
    # `code_has_for_iter` then refuses the inline from a caller loop.
    # The cache hit never reaches this arm. The two calls short-circuit
    # the same way, and `any` yields a bool.
    if cls.__subclasscheck__(subclass):
        return True
    if cls.__subclasscheck__(subtype):
        return True
    return False
