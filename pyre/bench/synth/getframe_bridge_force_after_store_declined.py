# Historical `_declined` companion to `getframe_bridge_force_after_store`.
# The name predates the standard-frame `f_locals` specialization: an exact
# `_getframe(0)` result now creates its CPython 3.14 `FrameLocalsProxy` without
# residualizing the getter or forcing the portal virtualizable.
#
# That is also what the PyPy oracle does through its `@jit.unroll_safe`
# `fast2locals`: one loop, one bridge, no forcings, no virtualizable forcings,
# and no aborts.  The rare bridge and the preceding effectful `box.n = i`
# remain useful coverage that the optimized frame read neither duplicates nor
# loses the store.
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
            names += len(fr.f_locals)
        total += i
    return total, names, box.n


print(main())
