# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=rec
# Regression guard: a bridge walk that entered AFTER the loop had already
# exited used to end by CLOSING THAT SAME LOOP, publishing the loop header as
# the CALLER frame's resume coordinate.
#
# `rec` is self-recursive and carries a `for`, so its recursive call inlines
# into a trace whose root is `rec` itself. The inlined callee re-crosses the
# very `jit_merge_point` the root registered, and the loop-close end-flush then
# wrote `last_instr = 17` (resume at the FOR_ITER, py_pc 18) onto the caller's
# frame over the caller's OWN operand stack -- which at py_pc 34 holds the int
# `acc`, not an iterator:
#
#     TypeError: 'int' object is not an iterator
#
# The invariant is a TRIPLE, not a pair:
#     (last_instr, valuestackdepth, the stack cells [base .. base+depth))
# py_pc 18 and py_pc 34 have the SAME static operand depth (1), so the flush's
# height check and the frame's own consistency both pass
# and only the cell contents witness the tear. That is why four separate
# audits of the publishing WRITERS found nothing: every writer was faithful to
# the coordinate it was handed, and the coordinate was the wrong one.
#
# `opimpl_jit_merge_point` (pyjitpl.py) closes a loop only under
# `if not self.metainterp.portal_call_depth:`; with a portal frame on the
# stack it finishes the frame and takes `do_recursive_call(...,
# assembler_call=True)` instead. pyre's CALL_ASSEMBLER route was gated on
# `framestack.last().w_code != <trace root code>` -- a CODE-identity test
# standing in for that FRAME question, and for a self-recursive function the
# two codes are the same pointer, so the callee read as "the root's own loop".
# The gate now asks `!is_top_level`, which every sub-walk carries.
#
# The four refuted attempts, kept because each names a writer that is NOT at
# fault:
#   1. `adopt_blackhole_crn` stamps `last_instr` alone, as its sibling
#      `apply_blackhole_crn_handoff` does; an audit placed inside it NEVER
#      FIRES here.
#   2. `LiveLastInstrGuard` saves only `last_instr`. Saving/restoring the whole
#      `FrameScalars` pair did not fix it: the guard is FAITHFUL, and the pair
#      is already inconsistent when `enter_frame` captures it.
#   3. The CALL_ASSEMBLER pin in `emit_walker_loop_callee_call_assembler`. Real
#      and live on main (11 emits on `str_search_index_bounds`) -- ZERO events
#      on this reproducer.
#   4. Pairing both words at the publish site: RED, build provenance verified.
#      Supplying two words without the cells cannot work.
#
# Do NOT resurrect the "resumed inside a Cache slot" reading: `pc` in
# `report_stack_underflow` is `next_instr()`, so `last_instr=66 pc=67` means
# resume pc 66, an ordinary opcode boundary. That was an inference, never an
# observation.
#
# Census before the fix: a 128-case sweep was 32 fail / 96 pass, all 32 in the
# `rec_for` family -- self-recursion plus a `for` loop is the whole shape. Red
# on four dynasm binaries incl. one built at origin/main, and on cranelift, so
# it was backend-independent and main-owned. Clean under PYRE_NO_JIT=1, and it
# survived `PYRE_TRACE_EAGERNESS=1000000` (bridging off), `PYRE_WALKABORT_OFF=1`
# and `PYRE_FORITER_CALL_BODY=1`. The two loops in `main` are both needed at
# these trip counts; a single loop needs ~1200 trials to reach it.

DATA = tuple(range(40))
STEP = sum(DATA)


def rec(n):
    if n <= 0:
        return 0
    acc = 0
    for ch in DATA:
        acc = acc + ch
    return acc + rec(n - 1)


def expected(n):
    return STEP * n if n > 0 else 0


def main():
    total = 0
    bad = 0
    for trial in range(50):
        n = trial % 50
        for value in [rec(i) for i in range(n)]:
            total += value
    for trial in range(60):
        n = trial % 20
        for i in range(n):
            value = rec(i)
            if value != expected(i):
                bad += 1
                if bad < 4:
                    print("MISMATCH i=%d got=%r want=%r" % (i, value, expected(i)))
            total += value
    if bad:
        print("FAIL %d wrong rec() results" % bad)
        raise SystemExit(1)
    if total != 17955600:
        print("FAIL total=%d want=17955600" % total)
        raise SystemExit(1)
    print("PASS loop-callee for-header resume total=%d" % total)


main()
