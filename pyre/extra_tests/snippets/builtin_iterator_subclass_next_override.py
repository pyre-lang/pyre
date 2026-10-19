# pyre-check: gate=1
# CPython-suite gap: no test overrides `__next__` on a subclass of a builtin
# iterator type and checks that `next()` and `for` call the override.
import itertools


CASES = [
    (enumerate, ([1, 2],)),
    (reversed, ((1, 2),)),
    (map, (abs, [1, 2])),
    (filter, (None, [1, 2])),
    (zip, ([1, 2], [3, 4])),
    (itertools.count, ()),
    (itertools.cycle, ([1, 2],)),
    (itertools.repeat, (1,)),
    (itertools.chain, ([1], [2])),
    (itertools.islice, ([1, 2, 3], 2)),
    (itertools.starmap, (pow, [(2, 3)])),
    (itertools.takewhile, (bool, [1, 0])),
    (itertools.dropwhile, (bool, [1, 0])),
    (itertools.filterfalse, (bool, [0, 1])),
    (itertools.accumulate, ([1, 2],)),
    (itertools.compress, ([1, 2], [1, 1])),
    (itertools.zip_longest, ([1], [2, 3])),
    (itertools.product, ([1, 2],)),
    (itertools.permutations, ([1, 2],)),
    (itertools.combinations, ([1, 2], 1)),
    (itertools.combinations_with_replacement, ([1, 2], 1)),
    (itertools.groupby, ([1, 1, 2],)),
    (itertools.pairwise, ([1, 2, 3],)),
]



def head(base, it):
    # A groupby item carries a fresh grouper; compare only its key.
    v = next(it)
    return v[0] if base is itertools.groupby else v


checked = 0
for base, args in CASES:
    try:
        class Override(base):
            def __next__(self):
                return "override"

        class Plain(base):
            pass
    except TypeError:
        continue
    it = Override(*args)
    assert next(it) == "override", base
    got = []
    for x in Override(*args):
        got.append(x)
        if len(got) == 2:
            break
    assert got == ["override", "override"], (base, got)
    assert head(base, Plain(*args)) == head(base, base(*args)), base
    checked += 1
assert checked >= 5, checked


class CallingSuper(enumerate):
    def __next__(self):
        i, v = super().__next__()
        return v * 10


assert list(CallingSuper([1, 2])) == [10, 20]


def not_an_iterator(obj):
    try:
        next(obj)
    except TypeError as e:
        return str(e)
    raise AssertionError(obj)


assert not_an_iterator(object()) == "'object' object is not an iterator"
assert not_an_iterator([]) == "'list' object is not an iterator"


class NoNext(object):
    pass


assert not_an_iterator(NoNext()) == "'NoNext' object is not an iterator"
