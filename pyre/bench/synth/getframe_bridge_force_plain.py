# Coverage for the forced-vable escape on a BRIDGE walk, which the rest of the
# corpus never produces: of 138 forced escapes across the synth fixtures, zero
# are `bridge=true`.
#
# Shape, each clause load-bearing:
#   * the `for` loop compiles on the common arm;
#   * `i % 97 == 0` is the rare arm, so its guard fails ~4124 times -- past
#     `DEFAULT_TRACE_EAGERNESS` -- and `start_bridge_tracing` sets
#     `ctx.is_bridge_trace`, making the walk over the rare arm a bridge walk;
#   * `_gf(0)` is a `CallFn` residual returning a Ref that forces the
#     virtualizable, and it is a builtin, so `frame_entry_count()` does not move
#     and no user Python frame is entered;
#   * the call sits directly in the portal frame's loop body, so the framestack
#     is empty and this is not an inline sub-walk.
#
# With nothing else in the rare arm the escape's mirror image resolves and the
# walk adopts a single-frame blackhole terminal. Its sibling
# `getframe_bridge_force_after_store` puts an un-journaled store ahead of the
# forcing call and takes the replay path instead.
#
# !! THE ESCAPE IS GONE FROM THIS FILE.  `_gf(0)` no longer produces a forcing
# residual -- `try_walker_specialize_sys_getframe` answers depth 0 at the top
# walk level out of the portal virtualizable -- so the shape below still
# compiles its loop and its bridge while forcing nothing.  The recorded baseline
# gates only part of that: it carries `fbw_blackhole_adopted_single_frame=0`,
# `..._multi_frame=0`, `loops_compiled=1` and `bridges_compiled=1`.  It carries
# NO `fbw_escape_*`, `fbw_force_*` or `sbt_entered` key at all, and an absent
# key is UNGATED, not zero -- those read 0 (and `sbt_entered=1`) only under
# `MAJIT_STATS=1`, which is where the claim above was measured.
#
# !!Raising the depth is not the repair either: `_gf(1)` restored the escape in
# the `getframe_while_*` fixtures, but from the PORTAL frame depth 1 names the
# module frame, which is not the traced virtualizable, so it measures zero here
# too -- as do `repr(_gf(0).f_code)`, storing `_gf(0)` into a module list, and
# `str(_gf(0).f_lasti)`.
#
# What a bridge walk does with a may-force residual is now answered, and the
# answer is "the same thing a top-level walk does": it forces, escapes, and
# adopts a blackhole terminal.  The lever has to sit one frame BELOW the portal
# so that depth 1 names the portal frame, which makes the terminal multi-frame.
# `getframe_bridge_force_from_inlined_callee` is that fixture and carries the
# bridge forced-escape coverage; this file and its `_after_store` sibling keep
# their compiled loop and bridge, which that shape gives up.
#
# The claim below about "138 forced escapes, zero bridge=true" is a historical
# note, not a live count.
import sys

_gf = sys._getframe


def main():
    total = 0
    names = 0
    for i in range(2000000):
        if i % 97 == 0:
            fr = _gf(0)
            names += len(fr.f_code.co_name)
        total += i
    return total, names


print(main())
