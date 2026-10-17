# pyre-check: max-pypy-ratio=30
# macOS reads 13.2-15.7x on dynasm (19.4x once); the ceiling is about twice
# that. Every failing guard of a generatorentry trace resumes in the
# blackhole, where `compile.py handle_fail` would compile a bridge.
# wasm has its own baseline: it reads guard_failures 38071 against 38067 on
# dynasm and cranelift, with the same loops and bridges. The cause of the
# four extra exits is not identified.
# Regression oracle: a callee entered through a CALL_ASSEMBLER fold reads a
# local out of a frame the trace is being recorded for, forcing it.  The
# recording stops there and execution goes on from behind the call, with the
# call's result, in every frame the recording had open -- never from in front
# of the call, and never in the outermost frame alone.
#
#   rec  -- `rec` and `step` call each other; the recursive call is folded
#           while both are inlined levels of the loop in `run_rec`, and the
#           innermost `rec` reads `step`'s and `rec`'s locals;
#   gen  -- a generator with several yields is resumed by the `for` in
#           `run_gen`, and its body reads the locals of that loop's frame.
#
# The call counter must equal what an interpreter counts, and the checksums
# over the results and the forced locals match.
import sys

N = 36000
SWITCH = 17000

calls = [0]
seen = [0]


def force(where, name, on):
    # Count, and when selected read a local out of the named frame above.
    calls[0] += 1
    if on:
        frame = sys._getframe(1)
        while frame.f_code.co_name != where:
            frame = frame.f_back
        seen[0] += frame.f_locals[name]


def step(d, i, on):
    k = d * 2 + i
    r = rec(d - 1, i, on)
    return r + k


def rec(d, i, on):
    if d == 0:
        force('step', 'k', on)
        force('rec', 'd', on)
        return i
    x = step(d, i, on)
    return x + d


def run_rec(n):
    acc = 0
    for i in range(n):
        acc += rec(3, i, i >= SWITCH and i % 7 == 0)
    return acc


def gen(i, on):
    yield i
    force('run_gen', 'i', on)
    yield i + 1
    force('run_gen', 'acc', on)
    yield i + 2


def run_gen(n):
    acc = 0
    for i in range(n):
        for v in gen(i, i >= 5 and i % 2 == 0):
            acc += v
    return acc % 1000003


def report(label, fn):
    calls[0] = 0
    seen[0] = 0
    print(label, fn(N), calls[0], seen[0] % 1000003)


def main():
    # `gen` first: with `rec` ahead of it no recording of `run_gen` meets the
    # force.
    report('gen', run_gen)
    report('rec', run_rec)
    # A second pass enters the loops and bridges the first one compiled.
    report('gen', run_gen)
    report('rec', run_rec)


main()
