# Reachability fixture: a force whose blackhole drive REACHES a jit_merge_point.
#
# Companion to getframe_root_loop_force_blackhole_crn. The walk roots at `main`,
# whose `while` loop gives the portal jitcode a merge point, so driving the
# latched image from just past the sys._getframe residual runs to the back edge
# and hands back ContinueRunningNormally rather than a frame terminal.
#
# Every getframe_* fixture that predates this one forces inside a short leaf
# callee whose jitcode has nothing after the residual but a return, so all of
# them ended in DoneWithThisFrameRef and the CRN arm -- the only arm that can
# reject the image -- was never exercised at all.
import sys
_gf = sys._getframe
kept = None
def main():
    global kept
    total = 0
    i = 0
    while i < 5400000:
        kept = _gf()
        total = total + 1
        i = i + 1
    return total
t = main()
print(t, kept.f_code.co_name)
