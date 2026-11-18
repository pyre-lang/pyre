# pyre-check: pypy-diverges: pypy3's `__build_class__` does not check the name's type, so a non-str name
# fails later in `type()` with "type() argument 1 must be string, not int" instead of
# "__build_class__: name is not a string".
#
# CPython-suite gap: test_class and test_metaclass pass only exact str names
# to class statements, so a copied name object is invisible there.
# parity-tests reason: `__build_class__` hands the original name object to
# `__prepare__` and the metaclass (compiling.py `build_class`); only
# `type.__new__` converts it to text and rejects a lone surrogate. A copy
# made before that would lose the subclass identity and reject a surrogate
# name a custom metaclass accepts.

"""__build_class__ hands the original name object to __prepare__ and the metaclass."""

class S(str):
    pass

seen = []


def meta(name, bases, ns, **kw):
    seen.append((type(name), name))
    return name

n = S("X")
assert __build_class__(lambda: None, n, metaclass=meta) is n
assert seen == [(S, "X")]


class M(type):
    @classmethod
    def __prepare__(mcls, name, bases, **kw):
        seen.append(("prepare", name))
        return {}
    def __new__(mcls, name, bases, ns, **kw):
        seen.append(("new", name))
        return super().__new__(mcls, str(name), bases, ns)

seen.clear()
n2 = S("Y")
c = __build_class__(lambda: None, n2, metaclass=M)
assert c.__name__ == "Y"
assert [entry[0] for entry in seen] == ["prepare", "new"]
assert all(entry[1] is n2 for entry in seen)

# A lone surrogate reaches a custom metaclass; type() itself rejects it.
lone = "a\udcffb"
assert __build_class__(lambda: None, lone, metaclass=lambda n, b, ns: n) is lone
try:
    __build_class__(lambda: None, lone)
except UnicodeEncodeError:
    pass
else:
    raise AssertionError("type() accepted a lone surrogate name")

try:
    __build_class__(lambda: None, 5)
except TypeError as e:
    assert str(e) == "__build_class__: name is not a string", str(e)
else:
    raise AssertionError("non-str name accepted")

print("OK")
