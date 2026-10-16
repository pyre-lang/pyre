# Hot STORE to an unboxed FLOAT attribute, the counterpart of
# `mapdict_unboxed_float_read_hot`. A same-type float update writes the value's
# IEEE-754 bit pattern straight into the existing longlong-list slot, and the
# loop now records that as the `setarrayitem_gc` it is, so it leaves NO
# residual.
#
# It used to leave `jit_mapdict_unboxed_write_f`, recorded as
# `[Ref, Int, Int, Float] -> Void`, and that is why this file exists. The shape
# is `(i64, i64, i64, f64) -> ()` on wasm: void, so the typed result family
# cannot express it, and float-carrying, so the uniform word family
# `residual_call_void_true_arity` covers cannot either -- before the backend
# grew a void arm over descr-derived parameter types, every iteration went out
# through the `jit_call` host trampoline: 3998959 crossings for 4000000
# iterations against the int store's 1, and 0.69s of execution against the int
# store's 0.02s. dynasm read 0.02s for both at the same point, so the whole gap
# was the crossing. The int-valued store below was the control that kept the
# attribution honest: same loop, same class shape, a
# `[Ref, Int, Int, Int] -> Void` descr the uniform word family already lowered
# in-module.
#
# Neither loop reaches a residual any more, so NEITHER gates that void arm.
# What this file still measures is the store itself against pypy.
#
# No `max-wasm-ratio` header -- that directive is an ALLOWANCE carved out of
# `WASM_MAX_DYNASM_RATIO`, and this fixture wants the default. Sized so
# dynasm's execution-only time clears FLOOR_GATE_MIN_BASELINE_S; below that the
# gate declines to judge.
#
# The trailing legs pin the shapes the unboxed store must refuse:
#
#   * storing an int into a float slot is a TYPE CHANGE, which revokes
#     unboxing rather than reinterpreting the word;
#   * storing a float into an int slot is the same event in reverse;
#   * NaN and the infinities round-trip as bit patterns, so they must come back
#     as themselves and not as a boxed re-read.


class Cell:
    def __init__(self, v):
        self.v = v


def write_float_hot(n, c):
    for i in range(n):
        c.v = 1.5
    return c.v


def write_int_hot(n, c):
    for i in range(n):
        c.v = 2
    return c.v


def revoked_shapes(n):
    acc = 0.0
    for _ in range(n):
        f = Cell(1.5)
        f.v = 0.25
        acc += f.v
        # Type change: the slot stops being unboxed rather than taking the
        # int's word into a float slot.
        f.v = 3
        acc += f.v
        i = Cell(7)
        i.v = 0.5
        acc += i.v
    return acc


def special_values():
    c = Cell(0.0)
    out = []
    for v in (float("nan"), float("inf"), float("-inf"), -0.0, 1e308):
        c.v = v
        out.append(repr(c.v))
    return out


print(write_float_hot(20000000, Cell(0.0)))
print(write_int_hot(20000000, Cell(0)))
print(revoked_shapes(20000))
print(" ".join(special_values()))
