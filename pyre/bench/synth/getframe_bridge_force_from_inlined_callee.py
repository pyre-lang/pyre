# pyre-check: max-pypy-ratio=9.7
# The forced-vable escape on a BRIDGE walk -- the case its two older siblings
# `getframe_bridge_force_plain` and `getframe_bridge_force_after_store` were
# written for and can no longer produce.
#
# What those two are missing is the lever, not the bridge. A residual forces the
# tracing token only when it is handed the TRACED virtualizable itself, and from
# inside the portal frame no `sys._getframe` call names it:
#   * `_gf(0)` / `_gf()` is folded -- `try_walker_specialize_sys_getframe`
#     answers depth 0 at the top walk level out of the portal virtualizable, so
#     there is no residual at all;
#   * `_gf(1)` from the portal frame names its CALLER, the module frame, which
#     is not the traced one, so the escape test returns false.
# Measured at zero on this shape as well: `repr(_gf(0).f_code)`, storing
# `_gf(0)` into a module list, and `str(_gf(0).f_lasti)`.
#
# One frame down, depth 1 lands on the portal frame: `peek()` inlines into
# the rare arm and `_gf(1).f_lasti` now folds at the CALL the portal is
# suspended at, so this shape no longer residualizes or forces.  The two
# older siblings still document why a portal-level `_getframe` cannot be the
# lever; this file keeps the inlined rare-arm shape and pins that the
# printed totals stay the same once the fold compiles.
#
import sys

_gf = sys._getframe


def peek():
    # Depth 1 from inside the inlined callee names `main` -- the traced
    # virtualizable -- so this stays a residual, and the `f_lasti` read off the
    # frame it returns forces the token.
    fr = _gf(1)
    _ = fr.f_lasti
    return fr.f_code.co_name


def main():
    total = 0
    names = 0
    for i in range(12000000):
        if i % 97 == 0:
            names += len(peek())
        total += i
    return total, names


print(main())
