# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=entry-bridge:bump,loop:_int_sum
# `main`'s loop aborts with ABORT_ESCAPE on every attempt. The
# `f.f_locals[...] = ...` store is the may-force residual `bh_store_subscr_fn`;
# its FrameLocalsProxy write-through forces the traced frame (`force_locals`
# clears the residual-call vable token), and `vable_after_residual_call`
# raises `SwitchToBlackhole(ABORT_ESCAPE)`. What reaches the JIT is `bump`'s
# entry bridge and the `_int_sum` loop. The descent that would record the
# store instead (`try_walker_orthodox_frame_locals_setitem` ->
# `fast_local_index`) stops at the may-force `hash_w_strict` residual inside
# the transparent helper.
# Self-checking regression guard for the frame-escape flush that resumes past
# the abort. The synthetic suite discovers it as a self-checking fixture.
# A residual (may-force) callee stores its own frame; the loop body then reads
# the caller's f_back.f_locals and mutates it through the 3.14 FrameLocalsProxy
# write-through. Reading the redirected caller frame forces it mid-expression:
# the escape flush must commit with the operand-stack mirror (the vable shadow's
# stack region is NULL there) and resume forward AT the escaping opcode. The
# legacy replay-from-loop-entry fallback drops the
# in-flight FOR_ITER iteration instead, so `total` comes up short -- the JIT-only
# regression this guards.
#
# The write-through is a 3.14 FrameLocalsProxy behaviour PyPy 3.11 lacks (its
# f_locals is a snapshot), so cpython and pypy disagree on the mutated value and
# this cannot use synthetic output parity; the invariant is pyre-internal
# instead (a correct JIT reproduces the no-JIT result) and is asserted here.
import sys

LOOPS = 20000
box = [None]


def bump(x):
    box[0] = sys._getframe(0)
    return x


def main():
    total = 0
    marks = 0
    for i in range(LOOPS):
        total += bump(i)
        f = box[0].f_back
        f.f_locals['marks'] = f.f_locals['marks'] + (1 if f.f_lineno > 0 else 0)
    if total != sum(range(LOOPS)):
        print(f"FAIL dropped iteration: total={total} expected {sum(range(LOOPS))}")
        raise SystemExit(1)
    if marks != LOOPS:
        print(f"FAIL write-through dropped: marks={marks} expected {LOOPS}")
        raise SystemExit(1)
    print(f"PASS total={total} marks={marks}")


main()
