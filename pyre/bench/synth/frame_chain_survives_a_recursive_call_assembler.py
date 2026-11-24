# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=hot,entry-bridge:rec
# The `root:` arm is the premise, not a relaxation: the activations this
# fixture walks are owed precisely because `rec` compiles and its recursive
# call is folded into `CALL_ASSEMBLER`. A `rec` that stopped compiling would
# make every walk below pass without testing anything.
# A warm self-recursive callee reaches its own compiled loop through
# `CALL_ASSEMBLER`, and the callee frame that fold hands the assembler is built
# out of recorded operations rather than by the interpreter's call door.
#
# The warm-up must not walk: `chain` reads `f_code` of frames under tracing,
# `_nonstandard_virtualizable` then `emit_force_virtualizable` forces them,
# and the trace ends `ABORT_ESCAPE`. `rec` takes the leaf as an argument so
# warm-up can pass a list-returning dummy; the measured `DEPTHS` walks pass `chain`.
#
# `pyframe.py execute_frame` runs `ec.enter(self)` before the merge point and
# `ec.leave(...)` in a `finally` after it, which is what puts an activation on
# the `topframeref` / `f_backref` chain `sys._getframe` walks. Upstream's merge
# point is on `PyFrame.dispatch` (`interp_jit.py`), so a CALL_ASSEMBLER that
# replaces it is recorded with the enter already traced ahead of it. pyre's
# portal sits one level up, at `execute_frame`, so the fold jumped over both
# halves: the frame it minted was never linked, and every activation the
# assembler ran was invisible to a stack walk.
#
# THE SHAPE IS THE TEST: `rec(DEPTH)` is exactly DEPTH + 1 activations of `rec`,
# stacked contiguously between the probe and the caller. Measured before the
# fix the walk found ONE `rec` frame at every depth — the outermost, the only
# one that still went through the interpreter — so `f_back` ran straight from
# the innermost body to the module frame.
#
# TWO DEPTHS ARE ALSO A TEST: a chain that is linked for the first level only,
# or that is off by a constant, tracks neither. The two walks must differ by
# exactly the difference between the depths.
import sys

WARM = 3000  # past the loop threshold (1039) many times over
WARM_DEPTH = 20  # the warm-up must RECURSE, or the fold never fires
DEPTHS = (3, 12)


def chain():
    names = []
    frame = sys._getframe()
    while frame is not None:
        names.append(frame.f_code.co_name)
        frame = frame.f_back
    return names


def dummy():
    return []


def rec(n, leaf):
    if n <= 0:
        return leaf()
    return rec(n - 1, leaf)


def hot(n):
    total = 0
    for _ in range(n):
        total = (total + len(rec(WARM_DEPTH, dummy))) % 1000003
    return total


def rec_frames(names):
    # `chain` is names[0]; count the contiguous `rec` activations above the
    # caller that requested the walk.
    count = 0
    for name in names[1:]:
        if name != 'rec':
            break
        count += 1
    return count


def main():
    failures = []
    # Compile `rec` first so the walks below run against compiled code rather
    # than against a cold frame the interpreter would have chained anyway.
    hot(WARM)
    walked = []
    for depth in DEPTHS:
        names = rec(depth, chain)
        got = rec_frames(names)
        walked.append(got)
        if got != depth + 1:
            failures.append(
                'rec(%d): the f_back chain holds %d `rec` frames, owed %d — '
                'the activations CALL_ASSEMBLER ran are missing from it (%r)'
                % (depth, got, depth + 1, names[: depth + 3])
            )
        if names[-1] != '<module>':
            failures.append(
                'rec(%d): the chain ends at %r, not the module frame'
                % (depth, names[-1])
            )
    grew = walked[1] - walked[0]
    owed = DEPTHS[1] - DEPTHS[0]
    if grew != owed:
        failures.append(
            'the chain grew by %d frames from depth %d to depth %d, owed %d — '
            'it does not track the recursion depth'
            % (grew, DEPTHS[0], DEPTHS[1], owed)
        )
    if failures:
        for line in failures:
            print('FAIL', line)
        return 1
    print('PASS a recursive CALL_ASSEMBLER activation stays on the frame chain')
    return 0


sys.exit(main())
