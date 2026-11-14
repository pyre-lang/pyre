# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=entry-bridge:refused_inline
# `main` aborts every recording residual that forces the caller
# (`vable_after_residual_call` → `SwitchToBlackhole(ABORT_ESCAPE)`).
# Self-checking regression guard for a caller's `f_lasti`, read from inside a
# callee the trace left RESIDUAL.  It reported the RESUME coordinate
# (`post_call_pc - 1`) instead of the `CALL` the caller is suspended at.
#
# The rule the histograms below name is not "the first residual call of a
# traced body is wrong" -- that is a symptom.  It is:
#
#     a read is wrong exactly when it executes while COMPILED code is the
#     caller's active executor, and right whenever it executes interpreted.
#
# The read's own force deopts the activation, so at most one such read happens
# per iteration, which is what makes it LOOK like "the first one".
#
# WHAT WAS MEASURED (release dynasm, 20000-iteration loops).  `guard_failures`
# tracked the wrong-answer count to within one on every variant, which is what
# identified the force as the event.
#
#   variant                          reads                      loops  guard_failures
#   one call site (this file)        56 x11074 / 62 x8926           3            8927
#   two sites, same callee           A: 54 x10547 / 60 x9453        3            9453
#                                    B: 128 x20000  (all right)
#   a residual call placed FIRST     all three sites right,         5           19640
#                                    20000/20000 each
#   two sites on opposite arms       A: 82 x1074 / 88 x8926         3            8927
#   of `if i % 2`                    B: 160 x20000  (all right)
#
# Every "all right" row is a site that never executes compiled.  Note the third
# row's loop DOES compile, so "a preceding residual fixes it" was false -- the
# reads simply stopped being compiled.
#
# THE WRONG VALUE WAS NOT AN INSTRUCTION AT ALL.  `dis` on `main` places `CALL`
# at offset 52 and the next real instruction, `STORE_FAST`, at 60, so the CALL
# owns 52..59 -- three cache entries at 54, 56, 58 -- and the wrong answer 58
# is the CALL's LAST CACHE SLOT.  Not a rival instruction offset; not an
# instruction boundary.
#
# WHY THE READER OWNS THE CONVERSION.  The stored `last_instr` is a dispatch
# coordinate, and deliberately so: `next_instr()` is `last_instr + 1`, every
# resume writer stores `pc - 1` against that inverse, and the derived value
# becomes the frame's re-entry point, the bridge walk's `start_pc` and the
# green/decline key.  Storing the `CALL`'s own index instead would resume at
# the CALL's FIRST cache word -- it moves the defect onto the resume path.  So
# the field keeps its meaning and the three PYTHON-VISIBLE readers convert:
# `PyFrame::executing_instr` behind `fget_f_lasti`, the `tb_lasti` mint in
# `record_application_traceback`, and `fset_f_lineno`'s `mark_stacks` index.
#
# All three, together -- that is what an earlier attempt got wrong.  A snap in
# `fget_f_lasti` ALONE does break `traceback_inlined_callee_lasti_regression`
# and `frame_lasti_fold_foreign_frames`, which assert `f_lasti == tb_lasti`:
# `tb_lasti` is minted from the same raw word and has to snap with it.  A raw
# `tb_lasti` of 58 was never right either.  The compiled channel needs nothing:
# `try_walker_specialize_frame_lasti` emits `walker_prove_owned_frame_pc`'s own
# pc, which is already an instruction, so the two channels now AGREE where
# `frame_lasti_fold_callee_and_bridge` asserts they must.
#
# The conversion itself is `pyopcode.rs owning_instruction`, the backward
# mirror of `skip_caches`.  pyre materializes each cache word as an
# `Instruction::Cache` unit and dispatches it as a no-op, so walking back over
# them is a local test rather than a scan -- which is why no backward helper
# had to be invented for this.
#
# A SECOND OBSERVABLE BREAK closed with it, and no fixture reaches it yet: a
# cache-region `last_instr` also made `frame.f_lineno = N` fail with "can't
# jump from unreachable code", because `mark_stacks` steps
# `i + cache_entries + 1` and so leaves every cache index `MARK_UNINITIALIZED`.
# CPython cannot hit it: `TARGET(CALL)` sets `frame->instr_ptr = next_instr`
# BEFORE `next_instr += 4`, so `_PyInterpreterFrame_LASTI` names the CALL for
# the callee's whole lifetime.  That invariant is the one this restores.
#
# REFUTED -- do not repeat these (each was investigated and killed):
#   * "the publish is merely mis-ORDERED": the force's resume image supersedes
#     the slot, so moving the publish before the CALL changes nothing.
#   * `state.rs` `virt_restore_scalars_raw` / `restore_virtualizable_from_raw`
#     as the force's stamp: that function has NO callers.
#   * "the force is one-shot per activation": the token is re-armed before
#     every may-force residual.
#   * "a non-reading residual absorbs the force": a residual that reads no
#     frame field forces nothing; the frame-chain walk does not force.
#   * `flush_active_frame_escape` as the producer: its escape pc is
#     `vstack_cur_pypc`, so it would write `call_pc - 1` -> f_lasti 54, a value
#     absent from every histogram above.
#   * the virtualizable BOXES as the reader's source: the read is a plain heap
#     read through the getset.  (Probed at three portal sites: boxes
#     110/165/185 against executing pcs 115/170/197.)
#   * `walker_frame_executing_py_pc` correctly DECLINES a
#     caller-frame-read-from-inside-a-callee receiver, so it is not in play.
#   * OptHeap eliding the publish: it keeps and hoists the store.
#   * an opcode with a different cache count as a discriminator:
#     `post_call_pc - 1` == `CALL_idx + cache_count` identically.
#
# NOT the same defect as `_pending/loop_callee_for_header_resume.py`, though it
# is the same field: there the triple `(last_instr, valuestackdepth, cells)` is
# torn at a resume; here nothing resumes, a live reader just reads the field.
import opcode
import sys

N = 20000


def refused_inline():
    acc = 0
    for _ in range(1):
        acc += 1
    return sys._getframe(1).f_lasti


def main():
    seen = {}
    for i in range(N):
        v = refused_inline()
        seen[v] = seen.get(v, 0) + 1
    offsets = sorted(seen)
    print("f_lasti values:", [(o, seen[o]) for o in offsets])
    if len(offsets) != 1:
        print(f"FAIL caller f_lasti diverged across the compile: {offsets}")
        return 1
    # Stability alone is not the property under guard: a regression that
    # consistently reports the WRONG opcode reads as one offset too.  The
    # reverted widening answered 42, the `LOAD_GLOBAL` that pushes
    # `refused_inline`, instead of the `CALL` that entered it.  So decode the
    # caller's own bytecode at the offset and require the CALL itself.
    #
    # `refused_inline()` is the only zero-argument call in `main`, so the
    # oparg pins WHICH call: `seen.get(v, 0)` is 2, `range`/`sorted` are 1.
    off = offsets[0]
    code = main.__code__.co_code
    op, oparg = code[off], code[off + 1]
    if op != opcode.opmap["CALL"] or oparg != 0:
        print(
            f"FAIL caller f_lasti {off} names {opcode.opname[op]} arg={oparg}, "
            "not the zero-argument CALL of refused_inline"
        )
        return 1
    print("PASS")
    return 0


sys.exit(main())
