# The bridge forced-vable escape of `getframe_bridge_force_plain`, with one
# un-journaled store ahead of the forcing call.
#
# `box.n = i` lowers to a Void `store_attr_fn` residual: it writes live heap, so
# it bumps the executed-effect odometer, and no journal covers it. Rolling the
# walk back therefore cannot undo the store, and the legacy entry replay would
# apply it a second time. The escape has to capture its operand-stack mirror and
# resume forward instead, which is what the recorded
# `fbw_blackhole_adopted_single_frame` pins; `fbw_rolled_back_with_effects` back
# above zero means the capture broke and the store is running twice again.
#
# The forcing residual itself never contributes an effect: the force branch
# returns before the odometer bump, so a bridge escape needs a second, earlier
# effectful op to register at all -- which is exactly what this file adds.
#
# !! THE ESCAPE IS GONE FROM THIS FILE.  `_gf(0)` no longer produces a forcing
# residual -- `try_walker_specialize_sys_getframe` answers depth 0 at the top
# walk level out of the portal virtualizable -- so the shape below still
# compiles its loop and its bridge while forcing nothing.  The recorded baseline
# gates only part of that: it carries `fbw_blackhole_adopted_single_frame=0`,
# `..._multi_frame=0`,
# `loops_compiled=1` and `bridges_compiled=1`.  It carries NO `fbw_escape_*`,
# `fbw_force_*` or `sbt_entered` key at all, and an absent key is UNGATED, not
# zero -- those read 0 (and `sbt_entered=1`) only under `MAJIT_STATS=1`, which
# is where the claim above was measured.
#
# !!Raising the depth is not the repair either: from the PORTAL frame `_gf(1)`
# names the module frame, not the traced virtualizable, so it measures zero here
# too.  The lever has to sit one frame BELOW the portal.
# `getframe_bridge_force_from_inlined_callee` carries the bridge forced-escape
# coverage now; on that shape the store ahead of the call stops being a
# discriminator, because the escape captures its image and resumes forward
# whether or not one stands -- which is why this file keeps its own shape rather
# than being converted.
import sys

_gf = sys._getframe


class Box:
    n = 0


box = Box()


def main():
    total = 0
    names = 0
    for i in range(2000000):
        if i % 97 == 0:
            box.n = i
            fr = _gf(0)
            names += len(fr.f_code.co_name)
        total += i
    return total, names, box.n


print(main())
