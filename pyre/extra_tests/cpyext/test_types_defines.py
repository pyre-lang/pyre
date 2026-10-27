# cpyext-fixture: cpyext_types
# cpyext-expect: cpyext-types-ok

# End-to-end check for C-defined types: `PyType_Ready`, the slot wrappers,
# `tp_methods`/`tp_members`/`tp_getset` descriptors, inheritance through
# `tp_base` and `PyErr_NewExceptionWithDoc`.

import cpyext_types as m
import sys

# ── the type itself ────────────────────────────────────────────────────
assert m.Point.__name__ == 'Point'
assert m.Point.__module__ == 'cpyext_types'
assert repr(m.Point.__dict__['norm']) == "<method 'norm' of 'cpyext_types.Point' objects>"
assert m.Point.__doc__ == 'a two-dimensional point defined in C'
assert isinstance(m.Point, type)
assert m.flags() == (1, 1)

# A C type whose base is `int` earns the long fast-subclass flag; one based on
# `object` earns none of them.
made, is_long, is_unicode, point_is_long = m.subclass_flags()
assert issubclass(made, int), made
assert (is_long, is_unicode, point_is_long) == (1, 0, 0), \
    (is_long, is_unicode, point_is_long)

# ── tp_new and tp_init ─────────────────────────────────────────────────
p = m.Point()
assert (p.x, p.y) == (0, 0)
assert p.label == ''
assert p.scale == 1.0

p = m.Point(3, 4)
assert (p.x, p.y) == (3, 4)
p = m.Point(3, 4, 'origin')
assert p.label == 'origin'
p = m.Point(y=9, x=1, label='kw')
assert (p.x, p.y, p.label) == (1, 9, 'kw')

try:
    m.Point(1, 2, 3, 4)
except TypeError:
    pass
else:
    raise AssertionError('too many arguments accepted')

# ── tp_members ─────────────────────────────────────────────────────────
p = m.Point(3, 4)
p.x = 10
assert p.x == 10
p.scale = 2.5
assert p.scale == 2.5
p.label = ['not', 'a', 'string']
assert p.label == ['not', 'a', 'string']

try:
    p.y = 1
except AttributeError:
    pass
else:
    raise AssertionError('a READONLY member was written')

assert type(m.Point.__dict__['x']).__name__ == 'member_descriptor'
assert repr(m.Point.__dict__['x']) == "<member 'x' of 'cpyext_types.Point' objects>"
assert m.Point.__dict__['x'].__doc__ == 'the abscissa'
assert m.Point.__dict__['x'].__name__ == 'x'
assert m.Point.__dict__['x'].__objclass__ is m.Point

# ── tp_getset ──────────────────────────────────────────────────────────
p = m.Point(3, 4)
assert p.total == 3 + 4 + 100
p.total = 12
assert (p.x, p.y) == (12, 0)
assert p.frozen == 'frozen'
try:
    p.frozen = 'thawed'
except AttributeError:
    pass
else:
    raise AssertionError('a get-only property was written')

assert type(m.Point.__dict__['total']).__name__ == 'getset_descriptor'
assert repr(m.Point.__dict__['total']) == "<attribute 'total' of 'cpyext_types.Point' objects>"
assert m.Point.__dict__['total'].__doc__ == 'x + y + closure'
assert m.Point.__dict__['total'].__objclass__ is m.Point

# ── tp_methods ─────────────────────────────────────────────────────────
p = m.Point(3, 4)
assert p.norm() == 25
assert p.translate(1, 1) is p
assert (p.x, p.y) == (4, 5)
assert p.named() == 'p:4'
assert p.named('q') == 'q:4'
assert p.named(prefix='r') == 'r:4'
try:
    p.norm(1)
except TypeError:
    pass
else:
    raise AssertionError('METH_NOARGS took an argument')

assert type(m.Point.__dict__['norm']).__name__ == 'method_descriptor'

# METH_CLASS binds the class, METH_STATIC binds nothing, and each is a
# different kind of attribute on the type.
assert type(m.Point.__dict__['origin']).__name__ == 'classmethod_descriptor'
assert type(m.Point.__dict__['units']).__name__ == 'staticmethod'
origin = m.Point.origin()
assert (origin.x, origin.y) == (0, 0), (origin.x, origin.y)
assert (m.Point(1, 2).origin().x, m.Point(1, 2).origin().y) == (0, 0)
assert m.Point.units(21) == 42
assert m.Point(1, 2).units(3) == 6
# `PyClassMethodDescr_Type` names `method_repr`, so a classmethod descriptor
# reports itself the way a method descriptor does.
assert repr(m.Point.__dict__['origin']).startswith("<method 'origin' of ")
assert m.Point.__dict__['norm'].__doc__ == 'squared length'
assert m.Point.__dict__['norm'].__objclass__ is m.Point
# An unbound descriptor takes the receiver as its first argument.
assert m.Point.__dict__['norm'](p) == 4 * 4 + 5 * 5


def raises(kind, message, call):
    try:
        call()
    except kind as error:
        assert str(error) == message, '%r != %r' % (str(error), message)
    else:
        raise AssertionError('%s was not raised: %s' % (kind.__name__, message))


# A descriptor only applies to an instance of the type that declared it: the
# definition it carries names an offset into that type's block, or a function
# that casts the receiver to it.
class Foreign:
    pass


foreign = Foreign()
for kind, spelling, call in [
    ('method', 'unbound', lambda: m.Point.__dict__['translate'](foreign, 1, 1)),
    ('method', '__get__', lambda: m.Point.__dict__['norm'].__get__(foreign)),
    ('member', '__get__', lambda: m.Point.__dict__['x'].__get__(foreign)),
    ('member', '__set__', lambda: m.Point.__dict__['x'].__set__(foreign, 7)),
    ('attribute', '__get__', lambda: m.Point.__dict__['total'].__get__(foreign)),
    ('attribute', '__set__', lambda: m.Point.__dict__['total'].__set__(foreign, 7)),
]:
    name = {'method': 'norm', 'member': 'x', 'attribute': 'total'}[kind]
    if kind == 'method' and spelling == 'unbound':
        name = 'translate'
    raises(
        TypeError,
        "descriptor '%s' for 'cpyext_types.Point' objects "
        "doesn't apply to a 'Foreign' object" % name,
        call,
    )

# A subclass is an instance of the declaring type, so it passes.
assert m.Point.__dict__['norm'](m.Point3(1, 2)) == 1 * 1 + 2 * 2

raises(
    TypeError,
    'unbound method Point.norm() needs an argument',
    lambda: m.Point.__dict__['norm'](),
)

# ── METH_METHOD ────────────────────────────────────────────────────────
# The row is handed the class it was declared in, and its bound carrier is
# `builtin_method` where every other row's is `builtin_function_or_method`.
p = m.Point(3, 4)
assert p.declared_in(1, 2) == ('cpyext_types.Point', 3, 2, 0)
assert p.declared_in(1, k=2) == ('cpyext_types.Point', 3, 1, 1)
assert m.Point.declared_in(p, 1, 2) == ('cpyext_types.Point', 3, 2, 0)
assert type(p.declared_in).__name__ == 'builtin_method'
assert type(p.norm).__name__ == 'builtin_function_or_method'
assert type(p.translate).__name__ == 'builtin_function_or_method'
assert p.declared_in.__self__ is p
# Binding names no module of its own.
assert p.declared_in.__module__ is None
assert p.norm.__module__ is None
assert repr(m.Point.__dict__['declared_in']) == (
    "<method 'declared_in' of 'cpyext_types.Point' objects>"
)
raises(
    TypeError,
    "descriptor 'declared_in' for 'cpyext_types.Point' objects "
    "doesn't apply to a 'Foreign' object",
    lambda: m.Point.declared_in(foreign, 1),
)

# ── tp_repr, tp_str, tp_hash, tp_call ──────────────────────────────────
p = m.Point(3, 4)
assert repr(p) == 'Point(3, 4)'
assert str(p) == '3/4'
assert hash(p) == 3 * 1000003 + 4
assert p(1, 2) == (4, 6)
assert p(dy=10) == (3, 14)
assert p() == (3, 4)

# ── tp_richcompare ─────────────────────────────────────────────────────
small = m.Point(1, 1)
large = m.Point(5, 5)
assert small < large
assert small <= large
assert large > small
assert large >= small
assert small != large
assert small == m.Point(1, 1)
assert not (small == large)
assert (small == 'not a point') is False

# ── tp_base inheritance ────────────────────────────────────────────────
assert issubclass(m.Point3, m.Point)
assert m.is_subtype() == (1, 0)
q = m.Point3(2, 3)
assert isinstance(q, m.Point)
assert (q.x, q.y) == (2, 3)
assert q.z == 0
q.z = 7
assert q.depth() == 7
# Inherited slots: Point3 declares none of these.
assert repr(q) == 'Point(2, 3)'
assert q.norm() == 13
assert q.total == 2 + 3 + 100

# ── tp_iter and tp_iternext ────────────────────────────────────────────
assert list(m.Counter(4)) == [0, 1, 2, 3]
counter = m.Counter(2)
assert iter(counter) is counter
assert next(counter) == 0
assert next(counter) == 1
try:
    next(counter)
except StopIteration:
    pass
else:
    raise AssertionError('the exhausted iterator kept going')

# ── the C side reads its own instances back ────────────────────────────
made = m.make(6, 7)
assert type(made) is m.Point
assert (made.x, made.y, made.label) == (6, 7, 'made')
assert m.is_point(made) is True
assert m.is_point(m.Point3(0, 0)) is True
assert m.is_point(object()) is False
assert m.sum_x(made) == 6

# ── PyErr_NewExceptionWithDoc ──────────────────────────────────────────
assert issubclass(m.TypesError, Exception)
assert m.TypesError.__name__ == 'TypesError'
assert m.TypesError.__module__ == 'cpyext_types'
assert m.TypesError.__doc__ == 'raised by the fixture'
try:
    m.sum_x(object())
except m.TypesError as error:
    assert str(error) == 'not a Point', str(error)
else:
    raise AssertionError('the module exception did not propagate')

# ── the number table ───────────────────────────────────────────────────
v = m.Vec(3)
assert repr(v) == 'Vec(3)'
assert v.value == 3
assert repr(v + m.Vec(4)) == 'Vec(7)'
assert repr(v + 10) == 'Vec(13)'
assert repr(10 + v) == 'Vec(13)'
assert repr(v - 1) == 'Vec(2)'
assert repr(1 - v) == 'Vec(-2)'
assert repr(v * 4) == 'Vec(12)'
assert repr(4 * v) == 'Vec(12)'
assert repr(-v) == 'Vec(-3)'
assert repr(abs(m.Vec(-9))) == 'Vec(9)'
assert repr(v ** 3) == 'Vec(27)'
assert repr(pow(m.Vec(2), 10, 1000)) == 'Vec(24)'
assert int(v) == 3
assert float(v) == 3.0
assert bool(v) is True
assert bool(m.Vec(0)) is False
try:
    v + 'text'
except TypeError:
    pass
else:
    raise AssertionError('an unsupported operand was accepted')

acc = m.Vec(1)
same = acc
acc += 5
assert acc is same
assert acc.value == 6

# ── the sequence table ─────────────────────────────────────────────────
bag = m.Bag(4, 5, 6)
assert len(bag) == 3
assert bag[0] == 4
assert bag[-1] == 6
assert 5 in bag
assert 9 not in bag
bag[1] = 50
assert bag[1] == 50
del bag[0]
assert len(bag) == 2
assert bag[0] == 50
try:
    bag[9]
except IndexError:
    pass
else:
    raise AssertionError('an out-of-range index was accepted')
assert bag * 2 == [50, 6, 50, 6]
assert 2 * bag == [50, 6, 50, 6]
assert list(bag) == [50, 6]

# ── the mapping table ──────────────────────────────────────────────────
table = m.Table()
assert len(table) == 0
table['a'] = 1
table['b'] = 2
assert len(table) == 2
assert table['a'] == 1
try:
    table['missing']
except KeyError:
    pass
else:
    raise AssertionError('a missing key was accepted')
del table['a']
assert len(table) == 1
assert sorted(table.keys()) == ['b']

# ── the abstract protocols, driven from C ──────────────────────────────
p = m.protocol
assert p('add', 2, 3) == 5
assert p('add', 'a', 'b') == 'ab'
assert p('add', [1], [2]) == [1, 2]
assert repr(p('add', m.Vec(1), m.Vec(2))) == 'Vec(3)'
assert p('multiply', 3, 4) == 12
assert p('power', 2, 8) == 256
assert p('negative', 5) == -5
assert p('index', 7) == 7
assert p('float', 7) == 7.0
# `PyNumber_Long` is `int(o)`: a float truncates toward zero, a string or a
# bytes object is parsed, and what has neither answer is refused.
assert p('long', 7) == 7
assert p('long', 12.9) == 12
assert p('long', -12.9) == -12
assert p('long', True) == 1
assert p('long', '  42  ') == 42
assert p('long', b'42') == 42
assert p('long', 10 ** 30) == 10 ** 30
for bad, kind in [('zz', ValueError), ([], TypeError), (None, TypeError)]:
    try:
        p('long', bad)
    except kind:
        pass
    else:
        raise AssertionError('%r was accepted' % (bad,))
assert p('number_check', 7) is True
assert p('number_check', 'x') is False
assert p('as_ssize', 12) == 12

assert p('sequence_check', [1, 2]) is True
assert p('sequence_check', 5) is False
assert p('size', [1, 2, 3]) == 3
assert p('size', bag) == 2
assert p('getitem', [1, 2, 3], 1) == 2
assert p('getitem', bag, 0) == 50
assert p('contains', [1, 2, 3], 2) is True
assert p('contains', bag, 6) is True
assert p('list', (1, 2)) == [1, 2]
assert p('tuple', [1, 2]) == (1, 2)
assert p('seq_index', [7, 8, 9], 8) == 1
assert p('repeat', [1, 2], 2) == [1, 2, 1, 2]

assert p('mapping_check', {'a': 1}) is True
assert p('mapping_check', [1]) is False
assert p('keys', {'a': 1}) == ['a']
assert p('values', {'a': 1}) == [1]
assert p('items', {'a': 1}) == [('a', 1)]
assert p('getstring', {'a': 1}, 'a') == 1
assert p('haskey', {'a': 1}, 'a') is True
assert p('haskey', {'a': 1}, 'z') is False

# ── PyType_FromSpec ────────────────────────────────────────────────────
assert m.Spec.__name__ == 'Spec'
assert m.Spec.__doc__ == 'a heap type built from a spec'
assert m.Spec.__module__ == 'cpyext_types'
s = m.Spec(21)
assert repr(s) == 'Spec(21)'
assert s.code == 21
assert s.double() == 42
assert len(s) == 21
s.code = 4
assert s.double() == 8

# ── Py_tp_vectorcall: the class object's own call ──────────────────────
# The slot answers `Type(...)` instead of `__new__`/`__init__`, and this
# fixture stores a different value down each path so the caller can tell
# which one ran.
assert m.Vectored().value == 1

made = m.Vectored.__new__(m.Vectored)
made.__init__()
assert made.value == 2

# The slot takes no arguments, and refuses them by its own count.
for call in (lambda: m.Vectored(1), lambda: m.Vectored(k=1)):
    try:
        call()
    except IndexError as error:
        assert 'takes no arguments' in str(error), error
    else:
        raise AssertionError('the vectorcall accepted arguments')

# It is the type's own field: a class derived from it is built the ordinary
# way, which is the pair.
class Derived(m.Vectored):
    pass

assert Derived().value == 2
assert m.vectored_slot(m.Vectored) is True
assert m.vectored_slot(m.Spec) is False

# A loop is what the tracer records, and it must answer as the first call did.
for _ in range(2000):
    assert m.Vectored().value == 1

# ── the module and the token a spec type carries ───────────────────────
module, by_def_is_module, module_name, qualified = m.type_owner(m.Spec)
assert module is m
assert by_def_is_module == 1
assert module_name == 'cpyext_types', module_name
assert qualified == 'cpyext_types.Spec', qualified

# Py_TP_USE_SPEC made the spec's own address the token, so Spec answers for
# itself and Extra answers for Spec through its base.
found, owner, extra_found, extra_owner, absent = m.type_token(m.Spec)
assert (found, owner) == (1, m.Spec), (found, owner)
assert (extra_found, extra_owner) == (0, None), (extra_found, extra_owner)
assert absent == 0

found, owner, extra_found, extra_owner, absent = m.type_token(m.Extra)
assert (found, owner) == (1, m.Spec), (found, owner)
assert (extra_found, extra_owner) == (1, m.Extra), (extra_found, extra_owner)
assert absent == 0

# A type built without a token matches none of them.
found, owner, extra_found, extra_owner, absent = m.type_token(m.Point)
assert (found, owner, extra_found, extra_owner, absent) == (0, None, 0, None, 0)

try:
    m.type_token_null(m.Spec)
except SystemError:
    pass
else:
    raise AssertionError('a NULL token was accepted')

# ── a spec declaring storage relative to its base's ────────────────────
assert issubclass(m.Extra, m.Spec)
assert m.type_data_size(m.Extra) >= 16, m.type_data_size(m.Extra)
# Spec declares a whole block: one header plus its own `long`, so the data
# it adds beyond `object`'s is that field alone.
assert m.type_data_size(m.Spec) == 8, m.type_data_size(m.Spec)

e = m.Extra(9)
assert e.code == 9
e.set(3, 0.5)
assert e.get() == (3, 0.5, 9), e.get()
# The base's own storage is untouched by the extra data behind it.
e.code = 11
assert e.get() == (3, 0.5, 11), e.get()

# The same words through descriptors whose offsets were declared relative to
# the extra data and resolved when the type was built.
assert e.tag == 3, e.tag
assert e.weight == 0.5, e.weight
e.tag = 4
assert e.get() == (4, 0.5, 11), e.get()
try:
    e.weight = 1.0
except AttributeError:
    pass
else:
    raise AssertionError('a readonly member was written')

module, by_def_is_module, module_name, qualified = m.type_owner(m.Extra)
assert module is m
assert qualified == 'cpyext_types.Extra', qualified

# A static type carries no module of its own.
try:
    m.type_owner(m.Point)
except TypeError:
    pass
else:
    raise AssertionError('a static type reported a module')

# ── PyType_Freeze ──────────────────────────────────────────────────────
class Mutable:
    pass

Mutable.before = 1
m.freeze(Mutable)
assert Mutable.before == 1
try:
    Mutable.after = 2
except TypeError:
    pass
else:
    raise AssertionError('a frozen class accepted an attribute')

# Freezing adds IMMUTABLETYPE and keeps HEAPTYPE.  The instance layout does not
# change, so an instance reports the same size (`sys.getsizeof` counts the
# pre-header of a heap type with a dict or weakrefs).
assert Mutable.__flags__ & (1 << 9), 'PyType_Freeze keeps Py_TPFLAGS_HEAPTYPE'
assert Mutable.__flags__ & (1 << 8), 'PyType_Freeze sets Py_TPFLAGS_IMMUTABLETYPE'

class Sized:
    pass

size_before = sys.getsizeof(Sized())
m.freeze(Sized)
size_after = sys.getsizeof(Sized())
assert size_after == size_before, (size_after, size_before)

# A class whose base is still mutable cannot be frozen.
class Derived(Mutable):
    pass

class Deeper(Derived):
    pass

try:
    m.freeze(Deeper)
except TypeError:
    pass
else:
    raise AssertionError('a class with a mutable base was frozen')

# ── tp_descr_get and tp_descr_set ──────────────────────────────────────
class Holder:
    field = m.Doubler()

h = Holder()
assert h.field == 0
h.field = 5
assert h.field == 10
del h.field
assert h.field == 0
# A class access hands the descriptor back.
assert type(Holder.field) is m.Doubler

# ── the buffer table ───────────────────────────────────────────────────
blob = m.Blob(b'abcdef')
assert blob.exports() == 0
view = memoryview(blob)
assert blob.exports() == 1
assert view.obj is blob
assert len(view) == 6
assert view.readonly is False
assert view.itemsize == 1
assert view.format == 'B'
assert bytes(view) == b'abcdef'
assert view[0] == ord('a')
assert list(view[1:3]) == [ord('b'), ord('c')]
view.release()
assert blob.exports() == 0

with memoryview(blob) as inner:
    assert inner[5] == ord('f')
    assert blob.exports() == 1
assert blob.exports() == 0

# The window is the exporter's own storage, not a copy of it.
live = m.Blob(b'abc')
with memoryview(live) as writable:
    writable[0] = ord('z')
assert live.read(live) == b'zbc'

# The bytes-like conversions reach `bf_getbuffer` too.
assert bytes(m.Blob(b'xy')) == b'xy'
assert bytearray(m.Blob(b'xy')) == bytearray(b'xy')

# `readinto` writes into an exporter that answers only through the slot, and
# gives the export back when it is done.
import io

target = m.Blob(b'......')
assert io.BytesIO(b'abcdef').readinto(target) == 6
assert target.read(target) == b'abcdef'
assert target.exports() == 0

# A short source fills a prefix and leaves the rest, and reports what it wrote.
short = m.Blob(b'zzzz')
assert io.BytesIO(b'ab').readinto(short) == 2
assert short.read(short) == b'abzz'
assert short.exports() == 0

# PyObject_GetBuffer driven from C, over a C exporter and over a pyre object.
assert m.Blob(b'').read(m.Blob(b'held')) == b'held'
assert m.Blob(b'').read(b'plain bytes') == b'plain bytes'
assert m.Blob(b'').read(bytearray(b'mutable')) == b'mutable'
try:
    m.Blob(b'').read(42)
except TypeError:
    pass
else:
    raise AssertionError('a non-exporter was accepted')

# ── the buffer slots of an interpreter type ────────────────────────────
# A writable request reaches the exporter's own storage, so what C writes
# through it is what Python reads back.
import array
driver = m.Blob(b'')
target = bytearray(b'....')
assert driver.fill(target, ord('Z')) == 4
assert target == bytearray(b'ZZZZ')
window = memoryview(bytearray(b'....'))
assert driver.fill(window, ord('Q')) == 4
assert bytes(window) == b'QQQQ'
numbers = array.array('i', [1, 2, 3])
assert driver.fill(numbers, 0) == 12
assert list(numbers) == [0, 0, 0]

# Read-only storage refuses a writable request rather than handing out a
# copy that would swallow the write.
for immutable in (b'abcd', memoryview(b'abcd')):
    try:
        driver.fill(immutable, ord('Z'))
    except BufferError:
        pass
    else:
        raise AssertionError('a writable export of read-only storage')

# The address is the exporter's, so two exports of one object name it.
held = bytearray(b'abcdef')
assert driver.address(held) == driver.address(held)
assert driver.address(held) != driver.address(bytearray(b'abcdef'))

# The geometry each exporter reports: (len, itemsize, ndim, format,
# shape[0], strides[0]).
assert driver.describe(b'abcdef') == (6, 1, 1, 'B', 6, 1)
assert driver.describe(bytearray(b'abc')) == (3, 1, 1, 'B', 3, 1)
assert driver.describe(array.array('i', [1, 2])) == (8, 4, 1, 'i', 2, 4)
assert driver.describe(memoryview(b'abcd')[1:]) == (3, 1, 1, 'B', 3, 1)

# A view with no dimensions carries no dimension vectors at all: `shape` and
# `strides` are absent rather than empty, and stay absent through a
# `memoryview` and back into C.  `PyBuffer_GetPointer` is handed no index for
# such a view, so a fabricated dimension makes it read one that is not there.
scalar = m.Scalar(1.5)
assert driver.describe(scalar) == (8, 8, 0, 'd', -1, -1)
assert driver.describe(memoryview(scalar)) == (8, 8, 0, 'd', -1, -1)
assert scalar.at(scalar) == 1.5
assert scalar.at(memoryview(scalar)) == 1.5

view = memoryview(scalar)
assert (view.ndim, view.shape, view.strides, view.nbytes) == (0, (), (), 8)
assert view.tolist() == 1.5
assert len(bytes(view)) == 8

# A strided export is refused by a request that cannot describe one.
try:
    driver.read(memoryview(b'abcdef')[::2])
except BufferError:
    pass
else:
    raise AssertionError('a strided export answered a contiguous request')

# The export the acquisition holds is given back, so the exporter can be
# resized again once C releases it.
resizable = bytearray(b'ab')
assert driver.read(resizable) == b'ab'
resizable.append(ord('c'))
assert resizable == bytearray(b'abc')

# A class written in Python earns the slots from `__buffer__`, and keeps
# its base's when it defines none.
class Forwarding:
    def __init__(self, payload):
        self.payload = payload
    def __buffer__(self, flags):
        return memoryview(self.payload)

assert driver.read(Forwarding(bytearray(b'via __buffer__'))) == b'via __buffer__'
assert driver.fill(Forwarding(bytearray(b'..')), ord('Y')) == 2

class DerivedArray(bytearray):
    pass

assert driver.read(DerivedArray(b'derived')) == b'derived'

# A type that exports nothing is refused, not snapshotted.
try:
    driver.read([1, 2, 3])
except TypeError:
    pass
else:
    raise AssertionError('a list answered a buffer request')

# ── the async table ────────────────────────────────────────────────────
ticker = m.Ticker(3)
assert list(ticker.__await__()) == [3, 2, 1]
assert ticker.__aiter__() is ticker
assert ticker.__anext__() == 2
assert ticker.__anext__() == 1
assert ticker.__anext__() == 0
try:
    ticker.__anext__()
except StopAsyncIteration:
    pass
else:
    raise AssertionError('the exhausted async iterator kept going')

# `await` goes through `am_await`, whose iterator's yields reach the caller.
async def use(source):
    return await source

coroutine = use(m.Ticker(2))
assert coroutine.send(None) == 2
assert coroutine.send(None) == 1
try:
    coroutine.send(None)
except StopIteration as stop:
    assert stop.value is None, stop.value
else:
    raise AssertionError('the coroutine did not finish')

# `am_send`, which has no dunder of its own, is reached through PyIter_Send.
stepper = m.Ticker(2)
assert m.send(stepper) == ('next', 1)
assert m.send(stepper) == ('next', 0)
assert m.send(stepper) == ('return', -1)
# A pyre iterator goes through `__next__` / `send` instead.
assert m.send(iter([7, 8])) == ('next', 7)

# ── capsules ───────────────────────────────────────────────────────────
capsule = m.PAYLOAD
assert repr(capsule).startswith('<capsule object "cpyext_types.PAYLOAD" at 0x')
assert m.capsule_read(capsule) == 4242
assert m.capsule_facts(capsule) == ('cpyext_types.PAYLOAD', 1, 0, 1, 1)
assert m.capsule_import() == 4242
try:
    m.capsule_wrong_name(capsule)
except ValueError as error:
    assert 'incorrect name' in str(error), str(error)
else:
    raise AssertionError('a mismatched capsule name was accepted')
try:
    m.capsule_read(object())
except ValueError as error:
    assert 'invalid PyCapsule' in str(error), str(error)
else:
    raise AssertionError('a non-capsule was accepted')

# ── imports ────────────────────────────────────────────────────────────
import sys
assert m.import_attr('sys', 'maxsize') == sys.maxsize
assert m.import_attr('cpyext_types', 'ANSWER_TYPES') == 'types'
assert m.add_module_ref('sys') is sys
fresh = m.add_module_ref('cpyext_fresh_module')
assert fresh.__name__ == 'cpyext_fresh_module'
assert sys.modules['cpyext_fresh_module'] is fresh
try:
    m.import_attr('cpyext_no_such_module', 'x')
except ImportError:
    pass
else:
    raise AssertionError('a missing module imported')

# ── the descriptors a namespace holds ──────────────────────────────────

point = m.Point(3, 4)


def eq(name, got, want):
    assert got == want, '%s: got %r, want %r' % (name, got, want)


# `PyMethodDescr_Check`, and the two fields the callers that agree read off
# the block straight afterwards.
eq('a tp_methods row is one', m.descr_facts(m.Point.receiver),
   (1, 'cpyext_types.Point', 'receiver'))
eq('a bound method is not', m.descr_facts(point.receiver), (0, None, None))
eq('a python function is not', m.descr_facts(eq), (0, None, None))
eq('the name it carries', m.descr_name(m.Point.receiver), 'receiver')

# The row reached the ordinary way binds the instance.
eq('reached through an instance', point.receiver(), ('cpyext_types.Point', '-'))

made = m.new_method_descr()
eq('a descriptor built by hand', m.descr_facts(made),
   (1, 'cpyext_types.Point', 'receiver'))
eq('and it binds an instance', made.__get__(point, m.Point)(),
   ('cpyext_types.Point', '-'))
eq('unbound, the instance is the argument', made(point),
   ('cpyext_types.Point', '-'))

# The class-method spelling of the same row binds the class instead.
classy = m.new_classmethod_descr()
eq('bound to the class', classy.__get__(point, m.Point)(),
   ('type', 'cpyext_types.Point'))
eq('bound with no instance', classy.__get__(None, m.Point)(),
   ('type', 'cpyext_types.Point'))
eq('unbound, the class is the argument', classy(m.Point),
   ('type', 'cpyext_types.Point'))

try:
    classy.__get__(None, int)
except TypeError as error:
    eq('a type it does not apply to', 'requires a subtype of' in str(error), True)
else:
    raise AssertionError('a descriptor reached through a foreign type must refuse')

# A type argument that is not a type is refused rather than cast, which is
# what makes this a `SystemError` instead of a descriptor whose `d_type` is
# not a type at all.
eq('a type argument that is not one', m.new_descr_bad_type(), 'SystemError')

# ── classmethod and staticmethod over whatever was handed in ───────────


def plain(*args):
    return args


class Holder:
    pass


Holder.classy = m.class_method_new(plain)
Holder.staticy = m.static_method_new(plain)
eq('a class method binds the class', Holder.classy(1), (Holder, 1))
eq('a static method binds nothing', Holder.staticy(1), (1,))
eq('reached through an instance too', Holder().classy(2), (Holder, 2))

# ── a type that declares no tp_new ─────────────────────────────────────
# `type_ready_set_new`: a static type deriving straight from `object` without a
# constructor of its own does not take `object`'s, and refuses instantiation.
eq('the flag is set and the slot is empty', m.sealed_flags(), (1 << 7, 0))
sealed = m.seal(41)
eq('the factory still builds one', sealed.token(), 41)
eq('and its methods reach the storage', type(sealed).__name__, 'Sealed')
try:
    type(sealed)()
except TypeError as error:
    eq('the report names the type', str(error),
       "cannot create 'cpyext_types.Sealed' instances")
else:
    raise AssertionError('a type with no tp_new was instantiated')

print('cpyext-types-ok')
