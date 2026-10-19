# pyre-check: pypy-diverges: pins 3.14's "division by zero"; pypy3 reports "integer division or modulo by zero"
# CPython-suite gap: int div/mod tests do not compare cold/hot raising traces.
# parity-tests reason: this guards pyre's JIT exact-int raising specialization.
#
# numeric_binary_subclass_specialization:
# numeric tests do not alternate subclasses at a hot binary site. this guards pyre's JIT
# exact-class specialization guards. pins PyFloat_AsDouble short-circuiting a float
# subclass; pypy3 runs its __float__ override instead

"""Guarded raising arms for exact-int ``//`` and ``%``.

Each traceback check compares a cold execution with the same source site after
its loop is hot, so frame names and line numbers cover both interpreted and
compiled paths without baking source coordinates into the fixture.  The phase
tests reuse one operator site with nonzero, zero, then nonzero divisors to
exercise both directions across the divisor guard and its bridge.
"""

ROUNDS = 6000
MESSAGE = "division by zero"


def signature(exc):
    frames = []
    tb = exc.__traceback__
    while tb is not None:
        frames.append((tb.tb_frame.f_code.co_name, tb.tb_lineno))
        tb = tb.tb_next
    return type(exc) is ZeroDivisionError, str(exc), exc.args, tuple(frames)


def same_floor(rounds):
    found = None
    zero = 0
    for i in range(rounds):
        try:
            i // zero
        except ZeroDivisionError as exc:
            found = signature(exc)
    return found


def same_mod(rounds):
    found = None
    zero = 0
    for i in range(rounds):
        try:
            i % zero
        except ZeroDivisionError as exc:
            found = signature(exc)
    return found


def floor_leaf(value, divisor):
    return value // divisor


def mod_leaf(value, divisor):
    return value % divisor


def one_frame(rounds, leaf):
    found = None
    for i in range(rounds):
        try:
            leaf(i, 0)
        except ZeroDivisionError as exc:
            found = signature(exc)
    return found


def floor_middle_1(value, divisor):
    return floor_leaf(value, divisor)


def floor_middle_2(value, divisor):
    return floor_middle_1(value, divisor)


def mod_middle_1(value, divisor):
    return mod_leaf(value, divisor)


def mod_middle_2(value, divisor):
    return mod_middle_1(value, divisor)


def three_frames(rounds, leaf):
    found = None
    for i in range(rounds):
        try:
            leaf(i, 0)
        except ZeroDivisionError as exc:
            found = signature(exc)
    return found


def flip_floor():
    raised = 0
    completed = 0
    checksum = 0
    for i in range(ROUNDS):
        divisor = 7 if i < 2000 or i >= 4000 else 0
        try:
            checksum += i // divisor
            completed += 1
        except ZeroDivisionError:
            raised += 1
    return raised, completed, checksum


def flip_mod():
    raised = 0
    completed = 0
    checksum = 0
    for i in range(ROUNDS):
        divisor = 7 if i < 2000 or i >= 4000 else 0
        try:
            checksum += i % divisor
            completed += 1
        except ZeroDivisionError:
            raised += 1
    return raised, completed, checksum


def check_signature(cold, hot, expected_names):
    assert cold == hot, (cold, hot)
    exact_type, message, args, frames = hot
    assert exact_type
    assert message == MESSAGE, message
    assert args == (MESSAGE,), args
    assert tuple(name for name, _ in frames) == expected_names, frames
    assert all(lineno > 0 for _, lineno in frames), frames


cold_same_floor = same_floor(1)
cold_same_mod = same_mod(1)
cold_one_floor = one_frame(1, floor_leaf)
cold_one_mod = one_frame(1, mod_leaf)
cold_three_floor = three_frames(1, floor_middle_2)
cold_three_mod = three_frames(1, mod_middle_2)

check_signature(cold_same_floor, same_floor(ROUNDS), ("same_floor",))
check_signature(cold_same_mod, same_mod(ROUNDS), ("same_mod",))
check_signature(cold_one_floor, one_frame(ROUNDS, floor_leaf), ("one_frame", "floor_leaf"))
check_signature(cold_one_mod, one_frame(ROUNDS, mod_leaf), ("one_frame", "mod_leaf"))
check_signature(
    cold_three_floor,
    three_frames(ROUNDS, floor_middle_2),
    ("three_frames", "floor_middle_2", "floor_middle_1", "floor_leaf"),
)
check_signature(
    cold_three_mod,
    three_frames(ROUNDS, mod_middle_2),
    ("three_frames", "mod_middle_2", "mod_middle_1", "mod_leaf"),
)

floor_phase = flip_floor()
mod_phase = flip_mod()
assert floor_phase[:2] == (2000, 4000), floor_phase
assert mod_phase[:2] == (2000, 4000), mod_phase

def numeric_binary_subclass_specialization():
    """Hot numeric binary specializers preserve reflected subclass dispatch.

    The subclass operand has to arrive at the *same* BINARY_OP pc that went hot, so
    each driver iterates a list whose tail holds the subclass.  Feeding it to a
    separate tail expression instead exercises a pc that was never specialized and
    the check passes whether or not the class guard exists.
    """

    HOT = 20000
    TAIL = 3


    class IntOperand(int):
        def __radd__(self, other):
            return ("radd", other, int(self))

        def __rfloordiv__(self, other):
            return ("rfloordiv", other, int(self))

        def __rmul__(self, other):
            return ("rmul", other, int(self))

        def __floordiv__(self, other):
            return ("floordiv", int(self), other)


    class FloatOperand(float):
        def __rtruediv__(self, other):
            return ("rtruediv", other, float(self))

        def __truediv__(self, other):
            return ("truediv", float(self), other)


    def rhs_add(operands):
        out = None
        for operand in operands:
            out = 7 + operand
        return out


    def rhs_floordiv(operands):
        out = None
        for operand in operands:
            try:
                out = 7 // operand
            except ZeroDivisionError:
                out = ("zero",)
        return out


    def rhs_mul(operands):
        out = None
        for operand in operands:
            out = 7 * operand
        return out


    def rhs_truediv(operands):
        out = None
        for operand in operands:
            try:
                out = 7.0 / operand
            except ZeroDivisionError:
                out = ("zero",)
        return out


    def lhs_floordiv(operands):
        out = None
        for operand in operands:
            try:
                out = operand // 3
            except ZeroDivisionError:
                out = ("zero",)
        return out


    def lhs_truediv(operands):
        out = None
        for operand in operands:
            out = operand / 2.0
        return out


    def tail(hot_value, subclass_value):
        return [hot_value] * HOT + [subclass_value] * TAIL


    # A non-zero divisor keeps the value arm live; a zero divisor takes the raising
    # arm.  Both must still reach the subclass's reflected method.
    assert rhs_add(tail(3, IntOperand(3))) == ("radd", 7, 3)
    assert rhs_mul(tail(3, IntOperand(3))) == ("rmul", 7, 3)
    assert rhs_floordiv(tail(3, IntOperand(3))) == ("rfloordiv", 7, 3)
    assert rhs_floordiv(tail(0, IntOperand(0))) == ("rfloordiv", 7, 0)
    assert rhs_truediv(tail(2.0, FloatOperand(2.0))) == ("rtruediv", 7.0, 2.0)
    assert rhs_truediv(tail(0.0, FloatOperand(0.0))) == ("rtruediv", 7.0, 0.0)

    # The left operand position needs its own guard.
    assert lhs_floordiv(tail(7, IntOperand(7))) == ("floordiv", 7, 3)
    assert lhs_floordiv(tail(0, IntOperand(0))) == ("floordiv", 0, 3)
    assert lhs_truediv(tail(7.0, FloatOperand(7.0))) == ("truediv", 7.0, 2.0)

    # A bool operand goes hot through the tagged/bool path, which skips the emitted
    # class guard; the subclass tail must still divert.
    assert rhs_add(tail(True, IntOperand(3))) == ("radd", 7, 3)
    assert rhs_floordiv(tail(True, IntOperand(3))) == ("rfloordiv", 7, 3)


    # ── `__float__` coercion at a hot site ──────────────────────────────────────
    # The float coercions are the same exact-class question one layer down. Reading
    # an int's payload is the answer only for an exact builtin: `PyNumber_Float`
    # (the `float()` constructor) and `PyFloat_AsDouble` (the `math` entry points)
    # both dispatch `nb_float` for a non-float, so a subclass override decides.
    # The layout predicates read `ob_type`, which the subclass shares, so both the
    # interpreter fast path and the trace guard admitted it and read the payload.
    import math


    class ToFloat(int):
        def __float__(self):
            return 99.0


    class ToFloatFromFloat(float):
        def __float__(self):
            return 99.0


    def coerce_hot(operands, fn):
        out = None
        for operand in operands:
            out = fn(operand)
        return out


    # `float()` honors the override for both bases: `PyNumber_Float` short-circuits
    # only `PyFloat_CheckExact`.
    assert coerce_hot(tail(4, ToFloat(4)), float) == 99.0
    assert coerce_hot(tail(4.0, ToFloatFromFloat(4.0)), float) == 99.0

    # `math` uses `PyFloat_AsDouble`, whose short-circuit is `PyFloat_Check` — so a
    # FLOAT subclass keeps its payload while an INT subclass takes the override.
    assert coerce_hot(tail(4, ToFloat(4)), math.sqrt) == math.sqrt(99.0)
    assert coerce_hot(tail(4.0, ToFloatFromFloat(4.0)), math.sqrt) == 2.0
    assert coerce_hot(tail(4, ToFloat(4)), math.cos) == math.cos(99.0)
    assert coerce_hot(tail(4, ToFloat(4)), math.frexp) == math.frexp(99.0)
    assert coerce_hot(tail(4, ToFloat(4)), lambda a: math.ldexp(a, 1)) == 198.0

    # `complex()` reaches the same real-number ladder.
    assert coerce_hot(tail(4, ToFloat(4)), complex) == complex(99.0, 0.0)

    # A float presentation code formats the `PyNumber_Float` conversion.
    assert coerce_hot(tail(4, ToFloat(4)), lambda a: format(a, ".2f")) == "99.00"

    # `loghelper` is the exception that proves the rule: it converts EVERY
    # `PyLong_Check` operand from its payload, subclass included, for both the
    # argument and the base. Routing either through the general coercion would
    # answer a different logarithm.
    assert coerce_hot(tail(4, ToFloat(4)), math.log) == math.log(4)
    assert math.log(100, ToFloat(10)) == 2.0

    # `__index__`-based entry points are likewise payload-only.
    assert coerce_hot(tail(16, ToFloat(16)), math.isqrt) == 4
    assert coerce_hot(tail(0, ToFloat(0)), lambda a: math.ldexp(1.0, a)) == 1.0


    # ── the subclass that does NOT override ─────────────────────────────────────
    # Every gate above sends a strict subclass to a `__float__` lookup. When it
    # overrides nothing, that lookup resolves to the INHERITED `int.__float__` /
    # `float.__float__`, which convert the receiver's payload. Binding either name
    # to its constructor instead makes this case re-enter the lookup that reached
    # it, so these asserts are what separates a payload-only dunder from one.
    class Plain(int):
        pass


    class PlainFloat(float):
        pass


    assert float(Plain(7)) == 7.0
    assert type(float(Plain(7))) is float
    assert Plain(7).__float__() == 7.0
    assert int.__float__(Plain(7)) == 7.0
    assert float(PlainFloat(4.5)) == 4.5
    assert type(float(PlainFloat(4.5))) is float

    assert coerce_hot(tail(4, Plain(4)), float) == 4.0
    assert coerce_hot(tail(4, Plain(4)), math.sqrt) == 2.0
    assert coerce_hot(tail(4, Plain(4)), math.frexp) == math.frexp(4.0)
    assert coerce_hot(tail(4, Plain(4)), complex) == complex(4.0, 0.0)
    assert coerce_hot(tail(4, Plain(4)), lambda a: format(a, ".2f")) == "4.00"
    assert coerce_hot(tail(4.0, PlainFloat(4.0)), math.sqrt) == 2.0
    assert math.log(100, Plain(10)) == 2.0

    # `int.__float__` converts a payload rather than constructing, so a receiver of
    # the wrong layout is a descriptor error, not a conversion.
    try:
        int.__float__("x")
    except TypeError as exc:
        assert "requires a 'int' object" in str(exc), exc
    else:
        raise AssertionError("int.__float__ accepted a str")

    # `rbigint.tofloat()` raises rather than answering inf.
    try:
        (1 << 2000).__float__()
    except OverflowError as exc:
        assert "too large" in str(exc), exc
    else:
        raise AssertionError("no OverflowError for an out-of-range int")
numeric_binary_subclass_specialization()
print("OK")
