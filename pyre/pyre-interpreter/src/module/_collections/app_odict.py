from __pypy__ import reversed_dict, move_to_end, objects_in_repr
from _collections_abc import KeysView, ItemsView, ValuesView

_missing = object()


def _odict_next(it):
    # `od_state` changes on every structural mutation of the dict, and the
    # dict iterator raises RuntimeError on the same changes.  The iterator
    # runs no application code, so only that check reaches the handler.
    try:
        return next(it, it)
    except RuntimeError:
        raise RuntimeError("OrderedDict mutated during iteration") from None


def _odict_keys_equal(a, b):
    'Pairwise key comparison of two OrderedDicts, as `_odict_keys_equal`.'
    it_a = dict.__iter__(a)
    it_b = dict.__iter__(b)
    key_a = _odict_next(it_a)
    key_b = _odict_next(it_b)
    while True:
        if key_a is it_a or key_b is it_b:
            # Equal only when both ran out together.
            return key_a is it_a and key_b is it_b
        # `PyObject_RichCompareBool` treats identical objects as equal.
        res = key_a is key_b or bool(key_a == key_b)
        # The next keys are read before `res` is acted on, so a mutation
        # made by the comparison raises even when the keys differ.
        key_a = _odict_next(it_a)
        key_b = _odict_next(it_b)
        if not res:
            return False


class OrderedDict(dict):
    '''Dictionary that remembers insertion order.

    In PyPy all dicts are ordered anyway.  This is mostly useful as a
    placeholder to mean "this dict must be ordered even on CPython".

    Known difference: iterating over an OrderedDict which is being
    concurrently modified raises RuntimeError in PyPy.  In CPython
    instead we get some behavior that appears reasonable in some
    cases but is nonsensical in other cases.  This is officially
    forbidden by the CPython docs, so we forbid it explicitly for now.
    '''
    __module__ = 'collections'

    def __init__(*args, **kwds):
        '''Initialize an ordered dictionary.  The signature is the same as
        regular dictionaries, but keyword arguments are not recommended because
        their insertion order is arbitrary.

        '''
        if not args:
            raise TypeError("descriptor '__init__' of 'OrderedDict' object "
                            "needs an argument")
        self, *args = args
        if len(args) > 1:
            raise TypeError('expected at most 1 arguments, got %d' % len(args))
        self.__update(*args, **kwds)

    def update(*args, **kwds):
        ''' D.update([E, ]**F) -> None.  Update D from mapping/iterable E and F.
            If E present and has a .keys() method, does:     for k in E: D[k] = E[k]
            If E present and lacks .keys() method, does:     for (k, v) in E: D[k] = v
            In either case, this is followed by: for k, v in F.items(): D[k] = v
        '''
        if not args:
            raise TypeError("descriptor 'update' of 'OrderedDict' object "
                            "needs an argument")
        self, *args = args
        if len(args) > 1:
            raise TypeError('update expected at most 1 arguments, got %d' %
                            len(args))
        if args:
            other = args[0]
            if hasattr(other, "keys"):
                for key in other.keys():
                    self[key] = other[key]
            elif hasattr(other, 'items'):
                for key, value in other.items():
                    self[key] = value
            else:
                for key, value in other:
                    self[key] = value
        for key, value in kwds.items():
            self[key] = value
    __update = update

    @classmethod
    def fromkeys(cls, iterable, value=None):
        # PyPy inherits this from its app-level dict gateway, which accepts
        # keywords.  CPython 3.14's OrderedDict keeps that public contract even
        # though the base dict method itself rejects keywords.
        self = cls()
        for key in iterable:
            self[key] = value
        return self

    def pop(self, key, default=_missing):
        # Spell these two inherited operations at app level so their Python
        # signatures retain PyPy's keyword acceptance on the 3.14 dict base.
        if default is _missing:
            return dict.pop(self, key)
        return dict.pop(self, key, default)

    def setdefault(self, key, default=None):
        return dict.setdefault(self, key, default)

    def __reversed__(self):
        return reversed_dict(self)

    def popitem(self, last=True):
        '''od.popitem() -> (k, v), return and remove a (key, value) pair.
        Pairs are returned in LIFO order if last is true or FIFO order if false.

        '''
        if last:
            return dict.popitem(self)
        else:
            it = dict.__iter__(self)
            try:
                k = next(it)
            except StopIteration:
                raise KeyError('dictionary is empty')
            return (k, self.pop(k))

    def move_to_end(self, key, last=True):
        '''Move an existing element to the end (or beginning if last==False).

        Raises KeyError if the element does not exist.
        When last=True, acts like a fast version of self[key]=self.pop(key).

        '''
        return move_to_end(self, key, last)

    def __repr__(self):
        'od.__repr__() <==> repr(od)'
        if not self:
            return '%s()' % (self.__class__.__name__,)
        currently_in_repr = objects_in_repr()
        if self in currently_in_repr:
            return '...'
        currently_in_repr[self] = 1
        try:
            # [3.14-spec] CPython `odict_repr` displays the ordinary
            # insertion-ordered dict spelling, while PyPy
            # `app_odict.OrderedDict.__repr__` displays `list(self.items())`.
            # Keep PyPy's app-level recursion protocol and change only the
            # observable payload spelling; the temporary dict leaves the
            # surrounding identity marker effective for self-values.
            return '%s(%r)' % (self.__class__.__name__, dict(self))
        finally:
            try:
                del currently_in_repr[self]
            except:
                pass

    def __reduce__(self):
        'Return state information for pickling'
        state = object.__getstate__(self)
        return self.__class__, (), state or None, None, iter(self.items())

    def copy(self):
        'od.copy() -> a shallow copy of od'
        od_copy = self.__class__()
        exact = type(self) is OrderedDict
        it = dict.__iter__(self)
        key = _odict_next(it)
        while key is not it:
            value = dict.__getitem__(self, key) if exact else self[key]
            od_copy[key] = value
            key = _odict_next(it)
        return od_copy

    def __eq__(self, other):
        '''od.__eq__(y) <==> od==y.  Comparison to another OD is order-sensitive
        while comparison to a regular mapping is order-insensitive.

        '''
        if isinstance(other, OrderedDict):
            return dict.__eq__(self, other) and _odict_keys_equal(self, other)
        return dict.__eq__(self, other)

    def __or__(self, other):
        if not isinstance(other, (dict, frozendict)):
            return NotImplemented
        # `odict_or` calls the type with the left operand, not `copy()`.
        copyself = type(self)(self)
        copyself.update(other)
        return copyself

    def __ror__(self, other):
        if not isinstance(other, (dict, frozendict)):
            return NotImplemented
        copy = type(self)(other)
        copy.update(self)
        return copy

    # for __ior__ the dict implementation is fine

    __ne__ = object.__ne__

    def keys(self):
        "D.keys() -> a set-like object providing a view on D's keys"
        return _OrderedDictKeysView(self)

    def items(self):
        "D.items() -> a set-like object providing a view on D's items"
        return _OrderedDictItemsView(self)

    def values(self):
        "D.values() -> an object providing a view on D's values"
        return _OrderedDictValuesView(self)

class _OrderedDictKeysView(KeysView):
    def __iter__(self):
        # PyPy subclasses the native dict view types here, so iteration is a
        # picklable dict iterator.  Python 3.14 makes those types unacceptable
        # as bases; preserve the upstream iterator shape by delegating to the
        # native view explicitly from the ABC-compatible wrapper.
        return iter(dict.keys(self._mapping))

    def __reversed__(self):
        yield from reversed_dict(self._mapping)

class _OrderedDictItemsView(ItemsView):
    def __iter__(self):
        return iter(dict.items(self._mapping))

    def __reversed__(self):
        for key in reversed_dict(self._mapping):
            yield (key, self._mapping[key])

class _OrderedDictValuesView(ValuesView):
    def __iter__(self):
        return iter(dict.values(self._mapping))

    def __reversed__(self):
        for key in reversed_dict(self._mapping):
            yield self._mapping[key]
