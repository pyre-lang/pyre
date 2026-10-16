# Regression guard: a blackhole drive that is declined after the fact must not
# hand its already-executed region back to a replay.
#
# Same shape as `getframe_root_loop_force_blackhole_crn`: the walk roots at
# `main`, whose portal jitcode carries a jit_merge_point at the loop header, and
# a sys._getframe force inside the loop latches a blackhole image that drives to
# the back edge and raises ContinueRunningNormally there.
#
# The difference is the effect in the driven region.  The sibling accumulates
# into a local and adds an already-present element to a set, so BOTH halves of a
# post-drive decline are invisible once the frame is restored: the local comes
# back with the undo, and re-running `set.add` of the same string changes
# nothing.  A list append does not have that property.  With the CRN arm forced
# to decline, that sibling prints its correct 199990000 while a 20000-iteration run of this shape printed
# 20005 appends — one extra per declined drive.
#
# That is the measurement behind `adopt_blackhole_crn`: a decline taken after
# the drive cannot be repaired by any frame-level undo, because the residual
# calls the drive ran are heap effects and not frame state.  So the CRN handoff
# resumes on the frame the blackhole already wrote instead of validating a
# rebuilt register image that could reject it.
import sys


def main():
    total = 0
    seen = []
    for i in range(2400000):
        fr = sys._getframe(0)
        seen.append(fr.f_code.co_name)
        total += i
    print(total, len(seen))


main()
