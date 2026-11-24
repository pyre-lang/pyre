# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=main
# The nested `FOR_ITER` operand stack must survive a compiled `_getframe(1)`
# from the inlined callee. The walk answers that call with the portal red
# box and folds `f_lasti` at the CALL, so the loop compiles instead of
# aborting into a multi-frame blackhole.
#
# The walk keeps that virtualizable symbolic, so nothing it pushed or popped
# ever reached the live frame's slot array. When the abort lands on a walk
# that crossed the inner `FOR_ITER`'s exhaust, the outer `FOR_ITER` has
# already run and `GET_ITER` has built a FRESH inner iterator that exists only
# in the walk: resuming without publishing the stack leaves the PREVIOUS
# pass's exhausted iterator on TOS, the inner loop ends one iteration in, and
# every later `i` runs one pass ahead of the `j` beside it. The values the
# body reads stay intact, which is why this reads as a loop-control error and
# not a lost item.
#
# Measured before the stack was published: `k=1041 i=209 j=0`, where `k` says
# the body is at `i=208 j=1` -- the first abort lands at the default hotness
# threshold and every pass after it is shifted. The compiled loop is `main`.
#
# The same read in a `while` loop (`getframe_while_escaping_read_frame_identity`)
# escapes just as often and cannot show this: with no iterator on the operand
# stack there is nothing for a stale image to hold.
import sys

_gf = sys._getframe

N = 5
PASSES = 96000
wrong = []


def leaf(x):
    fr = _gf(1)
    _ = fr.f_lasti
    name = fr.f_code.co_name
    if name != "main":
        wrong.append(name)
    return x + 1


def main():
    total = 0
    k = 0
    for i in range(PASSES):
        for j in range(N):
            total = leaf(total)
            if k != i * N + j:
                print(
                    f"FAIL dropped iteration: k={k} i={i} j={j} "
                    f"expected i={k // N} j={k % N}"
                )
                raise SystemExit(1)
            k += 1
    if total != PASSES * N:
        print(f"FAIL short run: total={total} expected {PASSES * N}")
        raise SystemExit(1)
    if wrong:
        print(f"FAIL caller identity: {wrong[:4]}")
        raise SystemExit(1)
    print(f"PASS total={total}")


main()
