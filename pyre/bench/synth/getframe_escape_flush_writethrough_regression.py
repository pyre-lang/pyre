# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=loop:_int_sum,loop:main
# The `root:` arm is measured, not a relaxation: this fixture's loop aborts
# five times with ABORT_ESCAPE and what reaches the JIT is the root trace
# `finish_and_compile` attaches. The declining arm is the `f_locals` LoadAttr
# fold's `standard_virtualizable_box() == Some(obj)` conjunct: the receiver
# here is the residual result of reading `f_back`, a fresh OpRef that can
# never equal the vable box even though its concrete pointer is the standard
# virtualizable (measured: `g.f_locals` fires and `f.f_locals` declines for
# `f is g`). There is no fold for `f_back` at all; adding one would let the
# existing arm fire unchanged. Whether the conjunct is deliberate or a gap is
# NOT settled — the tree reads both ways and the review split on it.
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
