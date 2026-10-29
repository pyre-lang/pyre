# pyre-check: pypy-diverges: pins 3.14's "cannot use ... as a set element" (pypy3 reports only "unhashable type")
# and 3.15's collection markers (pypy3 matches an unregistered `__getitem__` class as a sequence)
# CPython-suite gap: `test_abc` asks membership only through classes whose
# caches it never inspects, so it covers neither the empty-collection answer,
# nor an item that cannot carry a weakref, nor a class whose hash raises, and
# it cannot cover a replaced collection at all -- the reference build keeps the
# three collections in the type's own struct, so there is no attribute there to
# rebind.
#
# `_abc_instancecheck` answers membership with `in` on the cache, so the
# collection's own protocol is what answers, including a subclass whose
# `__contains__` lies, a rebound `.data`, or a shadowed `ref`.
#
# parity-tests reason: every arm is a silent wrong answer or a doubled side
# effect rather than a crash.  A collection read past its own override reports
# a class as registered when it is not, and a probe that answers and then
# re-asks evaluates an observable `__hash__` twice.
#
# The last three arms name a collection this implementation spells as a class
# attribute; where it is not spellable there is nothing to replace and the arm
# does not apply.
import _abc
import gc
import weakref
from abc import ABCMeta


class Plain:
    pass


calls = []


def empty_collection_answers_then_fills():
    class Fresh(metaclass=ABCMeta):
        pass

    class Late(Fresh):
        pass

    obj = Late()
    # The first question is asked against an empty collection, which holds no
    # entry to match; the same question after the walk recorded one must flip.
    assert isinstance(Plain(), Fresh) is False
    assert isinstance(obj, Fresh) is True
    assert isinstance(obj, Fresh) is True, 'the recorded entry was not found'


def unweakreferenceable_class_reaches_the_subclass_check():
    class Target(metaclass=ABCMeta):
        pass

    class Impl(Target):
        pass

    class Claims:
        __class__ = 3

    # An empty collection and a populated one reach the probe differently, and
    # neither may turn the missing weakref into an answer of its own: the
    # membership tests report "absent" and the subclass check that follows is
    # what raises.
    for _ in range(2):
        try:
            isinstance(Claims(), Target)
        except TypeError:
            pass
        else:
            raise AssertionError('a non-class __class__ must reach the check')
        assert isinstance(Impl(), Target) is True


def a_raising_hash_runs_once():
    class Boom(type):
        def __hash__(cls):
            calls.append(cls)
            raise ValueError('boom')

    class Fused(ABCMeta, Boom):
        pass

    class Warm(metaclass=ABCMeta):
        pass

    class Member(Warm):
        pass

    class Unhashable(metaclass=Fused):
        pass

    class Claims:
        __class__ = Unhashable

    # A populated collection, so the probe is really built and really hashed.
    assert isinstance(Member(), Warm) is True
    del calls[:]
    try:
        isinstance(Claims(), Warm)
    except ValueError:
        pass
    else:
        raise AssertionError('the raising hash was swallowed')
    # The membership expression evaluates the hash once.  Answering the probe
    # and then re-asking through the protocol would evaluate it twice, which is
    # observable here and in any hash carrying a side effect.
    assert len(calls) == 1, 'hash ran %d times' % (len(calls),)


def a_raising_hash_names_a_set_element():
    class Unhashable(ABCMeta):
        __hash__ = None

    class Warm(metaclass=ABCMeta):
        pass

    class Member(Warm):
        pass

    class Refuses(metaclass=Unhashable):
        pass

    class Claims:
        __class__ = Refuses

    # A populated collection, so the probe is really built and really hashed.
    assert isinstance(Member(), Warm) is True
    try:
        isinstance(Claims(), Warm)
    except TypeError as exc:
        message = str(exc)
    else:
        raise AssertionError('an unhashable class was accepted')
    # The collection is a set and the failure names the operand as one.  The
    # dict spelling of the same recovery renames it, which is what a reader of
    # the message sees.
    assert 'as a set element' in message, message


def a_collected_entry_stops_matching():
    class Held(metaclass=ABCMeta):
        pass

    class Doomed(Held):
        pass

    assert isinstance(Doomed(), Held) is True
    entries = _abc._get_dump(Held)[1]
    assert len(entries) == 1, 'the hit was not recorded: %r' % (entries,)
    del Doomed
    gc.collect()
    live = [ref for ref in _abc._get_dump(Held)[1] if ref() is not None]
    assert not live, 'a dead entry stayed in the cache: %r' % (live,)


def rebound_collection_is_consulted():
    class Asked(metaclass=ABCMeta):
        pass

    class Liar:
        def __init__(self):
            self.asked = []

        def __contains__(self, item):
            self.asked.append(item)
            return True

        def add(self, item):
            raise AssertionError('a hit records nothing')

    cache = Liar()
    Asked._abc_cache = cache
    assert isinstance(Plain(), Asked) is True, 'the replacement was bypassed'
    assert cache.asked, 'the replacement was never asked'


def replaced_data_is_consulted():
    class Swapped(metaclass=ABCMeta):
        pass

    class Impl(Swapped):
        pass

    obj = Impl()
    assert isinstance(obj, Swapped) is True
    # A list answers `in` by a linear scan rather than by a hashed probe, and a
    # callback-less weakref compares equal to the callback-carrying one the hit
    # above stored, so the entry is still found -- by the other route.
    Swapped._abc_cache.data = [weakref.ref(Impl)]
    assert isinstance(obj, Swapped) is True, 'a replaced data must still answer'


def a_data_subclass_keeps_its_override():
    class Shadowed(metaclass=ABCMeta):
        pass

    class Lying(set):
        def __contains__(self, item):
            return True

    # An empty table whose `__contains__` claims every member.  Reading the
    # table would answer False; the override answers True, and the override is
    # what the membership protocol reaches -- a subclass keeps the base layout,
    # so a layout test alone does not tell the two apart.
    Shadowed._abc_cache.data = Lying()
    assert isinstance(Plain(), Shadowed) is True, 'a data subclass was read past'


def a_mutated_contains_body_is_consulted():
    class Mutated(metaclass=ABCMeta):
        pass

    class Impl(Mutated):
        pass

    assert isinstance(Impl(), Mutated) is True

    collection = type(Mutated._abc_cache)
    original = collection.__contains__.__code__

    def claims_everything(self, item):
        return True

    # Assigning `__code__` leaves the same function installed on the same
    # class, so an identity test over the method alone still matches while the
    # body it runs no longer does.
    collection.__contains__.__code__ = claims_everything.__code__
    try:
        assert isinstance(Plain(), Mutated) is True, 'the mutated body was read past'
    finally:
        collection.__contains__.__code__ = original


def a_non_function_contains_declines():
    class Guarded(metaclass=ABCMeta):
        pass

    class Impl(Guarded):
        pass

    assert isinstance(Impl(), Guarded) is True

    collection = type(Guarded._abc_cache)
    original = collection.__contains__

    class Descriptor:
        def __get__(self, obj, objtype=None):
            return lambda item: True

    # Not a function at all.  Reading the code object off the installed method
    # is only defined for a function, so the shortcut has to reject this before
    # the read rather than by comparing whatever the read returned.
    collection.__contains__ = Descriptor()
    try:
        assert isinstance(Plain(), Guarded) is True, 'a non-function __contains__ was read past'
    finally:
        collection.__contains__ = original


def a_mutated_ref_global_is_consulted():
    class Watched(metaclass=ABCMeta):
        pass

    class Impl(Watched):
        pass

    assert isinstance(Impl(), Watched) is True

    collection = type(Watched._abc_cache)
    namespace = collection.__contains__.__globals__
    original = namespace['ref']
    calls = []

    def watching_ref(item, *rest):
        calls.append(item)
        return original(item, *rest)

    # `ref` is the whole free-variable surface of the app-level body, and
    # rebinding it moves neither the method nor its code object.
    namespace['ref'] = watching_ref
    try:
        isinstance(Impl(), Watched)
        assert calls, 'the rebound `ref` global was not consulted'
    finally:
        namespace['ref'] = original


def a_shadowed_typeerror_global_is_consulted():
    class Caught(metaclass=ABCMeta):
        pass

    class Impl(Caught):
        pass

    class Claims:
        __class__ = 3

    # A populated collection, so the probe is really built and really weakref'd.
    assert isinstance(Impl(), Caught) is True

    collection = type(Caught._abc_cache)
    namespace = collection.__contains__.__globals__
    assert 'TypeError' not in namespace, 'the handler resolves it through builtins'

    class Missing(Exception):
        pass

    # `except TypeError` reads the name as a global before falling back to
    # builtins, so a module entry shadows the builtin and stops the handler
    # catching the weakref failure.  That failure then leaves `__contains__`
    # instead of becoming a `False`, and the message which escapes says which
    # body answered.
    namespace['TypeError'] = Missing
    try:
        isinstance(Claims(), Caught)
    except TypeError as exc:
        message = str(exc)
    else:
        raise AssertionError('an unweakreferenceable class was accepted')
    finally:
        del namespace['TypeError']
    assert 'weak reference' in message, message


def an_empty_cache_still_builds_the_probe():
    import builtins

    class Fresh(metaclass=ABCMeta):
        pass

    class Claims:
        __class__ = 3

    class Missing(Exception):
        pass

    # Nothing has been recorded, so the collection is empty.  `__contains__`
    # still evaluates `ref(item)` first, and an item that cannot carry one
    # raises before `wr in self.data` is ever reached -- so an empty collection
    # does not decide this case.  With the handler's name rebound, that raise
    # is what escapes, exactly as it does for a populated collection.
    assert not _abc._get_dump(Fresh)[1], 'the cache was not empty'
    original = builtins.TypeError
    builtins.TypeError = Missing
    try:
        isinstance(Claims(), Fresh)
    except original as exc:
        message = str(exc)
    else:
        raise AssertionError('an empty cache answered before the probe')
    finally:
        builtins.TypeError = original
    assert 'weak reference' in message, message


def a_rebound_builtin_typeerror_is_consulted():
    import builtins

    class Caught(metaclass=ABCMeta):
        pass

    class Impl(Caught):
        pass

    class Claims:
        __class__ = 3

    assert isinstance(Impl(), Caught) is True

    class Missing(Exception):
        pass

    # The other half of the shadowed-global arm.  With no module entry to find,
    # `except TypeError` reaches the builtins binding, so rebinding it there
    # leaves the handler naming something the weakref failure is not, and the
    # failure escapes instead of becoming a `False`.  `except original` rather
    # than `except TypeError`, because the name in this frame resolves through
    # the same rebound builtins.
    original = builtins.TypeError
    builtins.TypeError = Missing
    try:
        isinstance(Claims(), Caught)
    except original as exc:
        message = str(exc)
    else:
        raise AssertionError('an unweakreferenceable class was accepted')
    finally:
        builtins.TypeError = original
    assert 'weak reference' in message, message


empty_collection_answers_then_fills()
unweakreferenceable_class_reaches_the_subclass_check()
a_raising_hash_runs_once()
a_raising_hash_names_a_set_element()
a_collected_entry_stops_matching()
if hasattr(ABCMeta('_Probe', (), {}), '_abc_cache'):
    rebound_collection_is_consulted()
    replaced_data_is_consulted()
    a_data_subclass_keeps_its_override()
    a_mutated_contains_body_is_consulted()
    a_non_function_contains_declines()
    a_mutated_ref_global_is_consulted()
    a_shadowed_typeerror_global_is_consulted()
    a_rebound_builtin_typeerror_is_consulted()
    an_empty_cache_still_builds_the_probe()


def collection_markers_follow_the_type_flags():
    # [3.15-spec] `MATCH_*` and `__flags__` read the collection marker alone.
    # pypy3 also accepts an unregistered `__getitem__` class as a sequence and
    # a `keys` + `__getitem__` class as a mapping, and shows
    # Py_TPFLAGS_SEQUENCE on `str`.
    import array
    import collections
    import types

    def kind(x):
        match x:
            case [*_]:
                return 'seq'
            case {}:
                return 'map'
            case _:
                return '-'

    class OnlyGetitem:
        def __getitem__(self, i):
            raise IndexError

    class KeysAndGetitem:
        def keys(self):
            return []

        def __getitem__(self, k):
            raise KeyError

    cases = [
        (OnlyGetitem(), '-', 0),
        (KeysAndGetitem(), '-', 0),
        ('s', '-', 0),
        (b'b', '-', 0),
        (bytearray(), '-', 0),
        ([], 'seq', 1),
        ((), 'seq', 1),
        (range(1), 'seq', 1),
        (memoryview(b''), 'seq', 1),
        (array.array('b'), 'seq', 1),
        (collections.deque(), 'seq', 1),
        (collections.UserList(), 'seq', 1),
        ({}, 'map', 2),
        (types.MappingProxyType({}), 'map', 2),
        (collections.OrderedDict(), 'map', 2),
        (collections.ChainMap(), 'map', 2),
        (frozenset(), '-', 0),
        ({}.values(), '-', 0),
    ]
    for value, want_kind, want_bits in cases:
        name = type(value).__name__
        assert kind(value) == want_kind, (name, kind(value), want_kind)
        bits = type(value).__flags__ >> 5 & 3
        assert bits == want_bits, (name, bits, want_bits)


def registration_stamps_heap_types_only():
    # [3.15-spec] `set_flags_recursive` stops on an immutable type.
    import collections.abc
    import contextvars

    class Registered:
        pass

    class Child(Registered):
        pass

    collections.abc.Sequence.register(Registered)
    assert Child.__flags__ >> 5 & 3 == 1
    match Child():
        case [*_]:
            pass
        case _:
            raise AssertionError('registered subclass is not a sequence')
    match contextvars.copy_context():
        case {}:
            raise AssertionError('immutable Context matched a mapping pattern')
        case _:
            pass


collection_markers_follow_the_type_flags()
registration_stamps_heap_types_only()
print('OK')
