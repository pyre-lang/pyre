# CPython-suite gap: the suite never sets __defaults__ to a tuple longer than
# the function's positional parameter count, so nothing there distinguishes
# aligning the defaults at the tuple's head from aligning them at its tail.
# parity-tests reason: the divergence is silent — the call returns a value,
# just the wrong element of __defaults__ — so only a value check catches it.
# pyre-check: pypy-diverges: pins 3.14's storage-level kwdefaults lookup (count 0, raising(1)==3); pypy3 3.11 dispatches __getitem__ (count 3000, RuntimeError propagates)

"""__defaults__ longer than the positional parameters binds from its tail.

`argument.py` `_match_signature` computes `def_first = co_argcount -
len(defaults_w)` as a *signed* quantity and then reads `defaults_w[i -
def_first]`.  When `__defaults__` is longer than the positional parameters
`def_first` is negative, which shifts every read towards the end of the
tuple.  Clamping it at zero instead reads the head and silently binds the
wrong value.

`__defaults__` is writable, so the over-long tuple is reachable from plain
Python without any introspection trickery.
"""


def two(a, b):
    return a, b


two.__defaults__ = (100, 101, 102)

# The positional-only binder and the keyword binder must agree about the same
# call: both take defaults[1] and defaults[2], never defaults[0].
assert two() == (101, 102)
assert two(a=9) == (9, 102)
assert two(b=9) == (101, 9)
assert two(9) == (9, 102)

two.__defaults__ = (100, 101, 102, 103, 104)
assert two() == (103, 104)
assert two(a=9) == (9, 104)
assert two(b=9) == (103, 9)

# One default per parameter is the aligned case, and one fewer still binds the
# tail — the same rule, with def_first non-negative.
two.__defaults__ = (100, 101)
assert two(a=9) == (9, 101)
two.__defaults__ = (100,)
assert two(a=9) == (9, 100)


def three(a, b, c):
    return a, b, c


three.__defaults__ = (100, 101, 102, 103, 104)
assert three() == (102, 103, 104)
assert three(a=9) == (9, 103, 104)
assert three(c=9) == (102, 103, 9)

# Keyword-only defaults come from __kwdefaults__ by name and must be untouched
# by the positional alignment.
def kwonly(a, b, *, k=None):
    return a, b, k


kwonly.__defaults__ = (100, 101, 102)
kwonly.__kwdefaults__ = {"k": 200}
assert kwonly() == (101, 102, 200)
assert kwonly(k=9) == (101, 102, 9)

# Keyword-only defaults bind from the mapping's dict storage
# (`PyDict_GetItemRef` / `getitem_str` on the backing dict).  A subclass
# `__getitem__` does not run, and a raise there does not affect the call.
getitem_count = [0]


class CountDict(dict):
    def __getitem__(self, k):
        getitem_count[0] += 1
        return dict.__getitem__(self, k)


def counted(a, *, b=1):
    return a + b


counted.__kwdefaults__ = CountDict(b=2)
counted_ok = 0
for _ in range(3000):
    counted_ok += counted(1)
assert counted_ok == 9000
assert getitem_count[0] == 0, getitem_count[0]


class RaisingDict(dict):
    def __getitem__(self, k):
        raise RuntimeError("kwdefaults getitem")


def raising(a, *, b=1):
    return a + b


raising.__kwdefaults__ = RaisingDict(b=2)
assert raising(1) == 3


class K:
    def __hash__(self):
        return hash("x")

    def __eq__(self, o):
        raise ZeroDivisionError("eq")


def f(*, x=1):
    return x


f.__kwdefaults__ = {K(): 0, "y": 2}
try:
    f()
except ZeroDivisionError as exc:
    assert str(exc) == "eq", exc
else:
    raise AssertionError("kwonly default lookup swallowed colliding key __eq__")

for _ in range(2000):
    try:
        f()
    except ZeroDivisionError as exc:
        assert str(exc) == "eq", exc
    else:
        raise AssertionError("kwonly default lookup swallowed colliding key __eq__")

eq_count = 0


class Key:
    def __hash__(self):
        return hash("x")

    def __eq__(self, o):
        global eq_count
        eq_count += 1
        return False


def h(a, *, x):
    return a


h.__kwdefaults__ = {Key(): 0}

N = 3000
for i in range(N):
    try:
        h(a=i)
    except TypeError:
        pass
assert eq_count == N, eq_count

# `ArgErrTooMany.getmsg` formats `num_args - num_defaults` and that difference
# is negative here.  It has to be reported as a negative number, not wrapped
# through an unsigned width.
def varkw(a, **kw):
    return a, kw


varkw.__defaults__ = (100, 101)
try:
    varkw(1, 2, 3, x=4)
except TypeError as exc:
    assert "takes from -1 to 1 positional arguments" in str(exc), exc
else:
    raise AssertionError("three positionals into a one-parameter function bound")

varkw.__defaults__ = (100, 101, 102)
try:
    varkw(1, 2, 3, x=4)
except TypeError as exc:
    assert "takes from -2 to 1 positional arguments" in str(exc), exc
else:
    raise AssertionError("three positionals into a one-parameter function bound")

# The same call compiled: the binder has a second copy on the traced path, and
# a loop long enough to compile reaches it.
def warm():
    for _ in range(50000):
        assert two(a=9) == (9, 100)
        assert three(c=9) == (102, 103, 9)


two.__defaults__ = (100,)
warm()

print("OK")
