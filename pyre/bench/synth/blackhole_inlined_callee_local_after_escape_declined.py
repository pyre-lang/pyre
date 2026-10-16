# Companion to blackhole_inlined_callee_local_after_escape, written to carry
# that file's shape at a force the constant-depth `sys._getframe` arm declines.
#
# !! THE GUARD BELOW IS CURRENTLY UNREACHABLE, and the recorded counters say so:
# `fbw_blackhole_adopted_single_frame=0`, `..._multi_frame=0`, `loops_aborted=0`.
# The prose from "Guard for what an adopted multi-frame blackhole chain owes"
# down describes a path this file no longer takes.  Two independent changes
# closed it, and neither can be undone from inside the fixture:
#
#   * the forcing residual is gone.  `try_walker_specialize_sys_getframe`
#     answers depth 0 at the top walk level out of the portal virtualizable, so
#     `sys._getframe()` folds; and the `.f_locals` read that replaced it as the
#     lever stopped forcing when the proxy force moved to the WRITE accessors
#     only -- correctly, since `fast2locals` is `@jit.unroll_safe` and forces
#     nothing.  `sys._getframe(1)` does still force (measured: one escape, one
#     `fbw_blackhole_adopted_single_frame`), which is the lever
#     `getframe_while_inlined_callee_subwalk` now uses;
#   * but a SINGLE-frame adopt is not this file's obligation.  Both failures
#     below need the escape inside an INLINED callee, and `catches_here` is no
#     longer inlined at all: it carries a try/except, so its whole-body replay
#     scan comes back Dirty/DeferredCall and the unseeded sub-walk screen
#     declines it (`InlineCallee::BranchyHandlerDirty`, census-visible under
#     `PYRE_FBW_INLINE_DIAG=1` as `resolved-inline-decline`).  Restoring this
#     guard means seeding a handler-bearing callee, not editing the driver.
#
# What still runs here is the printed answer, which its non-`_declined` sibling
# also guards.  Keep the file: the day the seeded arm admits a handler-bearing
# callee, this shape is the one that exercises it.
#
# Guard for what an adopted multi-frame blackhole chain owes its inner levels.
#
# An inlined callee assigns a local, the frame then escapes through the
# `.f_locals` read added above -- `sys._getframe()` itself folds here -- and an
# attribute read POSITIONED AFTER that escape reads the local back.  The read is
# executed by the blackhole, not by the walk, so the
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

N = 160000


def catches_here(i):
    try:
        raise ValueError(i)
    except ValueError as e:
        tb = e.__traceback__
        f = sys._getframe()
        _ = f.f_locals
        return (tb.tb_frame is f, tb.tb_lineno - f.f_code.co_firstlineno)


def drive():
    seen = set()
    k = 0
    while k < N:
        seen.add(catches_here(k))
        k += 1
    return sorted(seen)


print(drive())
