# pyre-check: pypy-diverges: pins 3.15's accessor slots being re-fixed on every
# class-namespace store; pypy3 leaves `uses_object_getattribute` /
# `uses_object_setattr` memoised, and its `LOAD_ATTR` / `STORE_ATTR` cache
# entries valid, when a cell absorbs the store, so its third access still takes
# `object`'s hook
#
# CPython-suite gap: no suite test rebinds `__getattribute__` or `__setattr__`
# twice on one class, and the second rebind is the one that matters -- it is the
# store an installed cell absorbs.
#
# parity-tests reason: that store moves no `_version_tag`, so the accessor memo
# and the `LOAD_ATTR` / `STORE_ATTR` cache entry keep answering for the hook
# they were filled with.  The attribute itself reads back correctly, so it
# surfaces as a wrong answer rather than as a missed optimisation.
#
# `typeobject.py W_TypeObject.setdictvalue` offers every store to `write_cell`.
# The store that replaces the class-body value installs a `MutableCell`; every
# store after that writes inside the installed cell, `write_cell` returns `None`
# and `setdictvalue` returns early.  That early return is what keeps
# `_version_tag` still -- otherwise a class-attribute store in a hot loop would
# revoke the loop -- and it is also what skips `mutated()`.
#
# Two shortcuts go stale when it does, and both answer "this type inherits
# `object`'s accessor" for a type that stopped inheriting it:
#
#   * `uses_object_getattribute` / `uses_object_setattr`, the memo
#     `getattribute_if_not_from_object` / `setattr_if_not_from_object` fill once
#     a lookup has confirmed the hook is `object`'s.  `mutated()` is the only
#     place they are cleared.
#   * the `MapdictCacheEntry` a `LOAD_ATTR` / `STORE_ATTR` site fills, which is
#     keyed on `_version_tag` (plus `valid_for_store` for the store side).  On a
#     hit the instance slot is read or written without asking
#     `getattribute_if_not_from_object` / `setattr_if_not_from_object` at all.
#
# Either way the attribute itself reads back correctly, so this is a wrong
# answer rather than a missed optimisation, and nothing else catches it.
#
# The sequence below is the smallest one that reaches it, and the order matters:
#   1. custom  -- the class body has no hook, so this store is raw
#   2. default -- replaces a value, so this is the store that parks the cell
#      ... and the access after it is what fills the memo and the cache entry
#   3. custom  -- absorbed by the cell, no tag moves, both left standing


def probe_getattribute(obj):
    try:
        return obj.missing
    except AttributeError:
        return 'attribute-error'


def custom_getattribute(n):
    def hook(self, name):
        return 'getattribute-%d' % n

    return hook


class Read:
    pass


read = Read()
Read.__getattribute__ = custom_getattribute(1)
assert probe_getattribute(read) == 'getattribute-1'
Read.__getattribute__ = object.__getattribute__
assert probe_getattribute(read) == 'attribute-error'
Read.__getattribute__ = custom_getattribute(2)
got = probe_getattribute(read)
assert got == 'getattribute-2', 'stale __getattribute__ memo: %r' % (got,)


def custom_setattr(n):
    def hook(self, name, value):
        object.__setattr__(self, 'seen', 'setattr-%d' % n)

    return hook


class Write:
    pass


write = Write()
Write.__setattr__ = custom_setattr(1)
write.anything = 1
assert write.seen == 'setattr-1'
Write.__setattr__ = object.__setattr__
write.seen = 'plain'
assert write.seen == 'plain'
Write.__setattr__ = custom_setattr(2)
write.anything = 1
assert write.seen == 'setattr-2', 'stale __setattr__ memo: %r' % (write.seen,)


# The `STORE_ATTR` cache is indexed by the name's `co_names` slot, so reaching
# it needs all three stores to name the SAME attribute from one code object.
# Store 2 runs `object.__setattr__`, which is what fills the entry with
# `valid_for_store`; store 3 must miss it and dispatch the rebound hook.
def store_attr_cache_sees_the_rebind():
    class Cached:
        pass

    obj = Cached()
    out = []
    Cached.__setattr__ = custom_setattr(1)
    obj.slot = 1
    out.append(getattr(obj, 'seen', 'plain'))
    Cached.__setattr__ = object.__setattr__
    obj.slot = 1
    out.append(getattr(obj, 'seen', 'plain'))
    Cached.__setattr__ = custom_setattr(2)
    obj.slot = 1
    out.append(getattr(obj, 'seen', 'plain'))
    return out


got = store_attr_cache_sees_the_rebind()
assert got == ['setattr-1', 'setattr-1', 'setattr-2'], 'stale store cache: %r' % (got,)


# The read twin: a `LOAD_ATTR` site whose middle access succeeds caches the
# instance slot under the same tag, and a hit answers from it without asking
# `getattribute_if_not_from_object`.
def load_attr_cache_sees_the_rebind():
    class Cached:
        pass

    obj = Cached()
    object.__setattr__(obj, 'slot', 'plain')
    out = []
    Cached.__getattribute__ = custom_getattribute(1)
    out.append(obj.slot)
    Cached.__getattribute__ = object.__getattribute__
    out.append(obj.slot)
    Cached.__getattribute__ = custom_getattribute(2)
    out.append(obj.slot)
    return out


got = load_attr_cache_sees_the_rebind()
assert got == ['getattribute-1', 'plain', 'getattribute-2'], 'stale load cache: %r' % (got,)


# A subclass resets the same two shortcuts, so `mutated()` walks subclasses.
class Base:
    pass


class Derived(Base):
    pass


derived = Derived()
Base.__getattribute__ = custom_getattribute(3)
assert probe_getattribute(derived) == 'getattribute-3'
Base.__getattribute__ = object.__getattribute__
assert probe_getattribute(derived) == 'attribute-error'
Base.__getattribute__ = custom_getattribute(4)
got = probe_getattribute(derived)
assert got == 'getattribute-4', 'stale subclass memo: %r' % (got,)


# The attribute itself was never wrong, which is why nothing else catches this.
assert Read.__dict__['__getattribute__'] is not object.__getattribute__

print('OK')
