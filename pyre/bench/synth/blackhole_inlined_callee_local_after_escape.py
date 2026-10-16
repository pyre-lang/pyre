# Output guard for an inlined callee whose local is read back after the frame
# escapes.  This file reaches no adopt of any kind: with the constant-depth
# `sys._getframe` fold (#1096) its committed baselines read
# `fbw_blackhole_adopted_single_frame=0`, `loops_aborted=0`, `loops_compiled=2`
# on all three backends.  It does not take the multi-frame arm either; that
# arm is pinned by `getframe_inline_subwalk_multiframe` and
# `getframe_while_inlined_callee_subwalk` among others, all recording a nonzero
# `fbw_blackhole_adopted_multi_frame` on all three backends.
#
# The blackhole obligations described below are carried by the `_declined`
# sibling (`blackhole_inlined_callee_local_after_escape_declined`), which repeats
# this shape at a force the fold declines and records 5 adopts / 5 aborts.  What
# survives here is the printed answer.
#
# An inlined callee assigns a local, the frame then escapes through a residual
# `sys._getframe()`, and an attribute read POSITIONED AFTER that escape reads the
# local back.  The read is executed by the blackhole, not by the walk, so the
# shape holds the adopt to two separate obligations and fails differently on
# each:
#
#   * every LOAD_FAST lowers to `getarrayitem_vable_r` on the level's own frame
#     array, so a level whose locals were left unpublished resumes `tb` as null
#     and the attribute read faults in `object_getattr_miss` -- a hard SIGSEGV
#     (rc=139) with no output at all;
#   * the traceback the callee stored has to name the frame the callee runs on,
#     so a walk-time node anchored on any other object prints `False` here while
#     still exiting 0.
#
# The second is the quieter one and the reason the assertion is an identity
# rather than a liveness check.  Both need the escape to happen inside an
# INLINED callee: the same shape through the single-frame arm was always
# correct.
#
# Deliberately carries no `# pyre-check: max-pypy-ratio=` header: this guards an
# output, and the forcing read makes it a poor perf subject.
import sys

N = 200000


def catches_here(i):
    try:
        raise ValueError(i)
    except ValueError as e:
        tb = e.__traceback__
        f = sys._getframe()
        return (tb.tb_frame is f, tb.tb_lineno - f.f_code.co_firstlineno)


def drive():
    seen = set()
    k = 0
    while k < N:
        seen.add(catches_here(k))
        k += 1
    return sorted(seen)


print(drive())
