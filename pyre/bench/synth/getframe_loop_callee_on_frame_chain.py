# pyre-check: max-pypy-ratio=16
# pyre-check: skip-backends=wasm
# wasm compiles a chain of bridges on the first guard of the previous bridge
# (19-35 bridges run to run, against 12 on dynasm and cranelift).
# macOS reads 4.0-8.1x on dynasm; the ceiling is twice the slowest.
# Regression oracle: a callee that carries its own loop is entered from an
# inlined call, and its loop runs as a call into its own compiled code.  The
# frames between the code a force runs in and the loop that was traced are
# all on the `f_back` chain while that call runs, in order, both while the
# caller is being recorded and afterwards in the compiled caller.
#
#   loop -- `inner` has a `for`; `use_loop` calls it from the loop in
#           `run_loop`, and `force` reads `inner`'s `j` one frame up and
#           `use_loop`'s `i` two frames up;
#   gen  -- `gen` is resumed by the `for` in `use_gen`, and `force` reads
#           `use_gen`'s `i` and `t` two frames up from inside the generator.
#
# After a call that forced, `use_loop` counts the frames above it: the loop
# callee is no longer among them.
#
# The call counter must equal what an interpreter counts, and the checksum
# over the locals read must match.
import sys

try:
    import pypyjit

    # Counter decay makes guard_failures depend on when collections run.
    pypyjit.set_param("decay=0")
except ImportError:
    pass

N_LOOP = 400000
N_GEN = 12000

calls = [0]
seen = [0]


def force(depth, name, on):
    # Count, and when selected read a local out of a frame above the caller.
    calls[0] += 1
    if on:
        seen[0] += sys._getframe(depth + 1).f_locals[name]


def inner(i, on):
    s = 0
    for j in range(40):
        s += j ^ i
        if j == 25:
            force(1, 'i', on)
            force(0, 'j', on)
    return s


def depth():
    # Frames on the chain above this one.
    n = 0
    f = sys._getframe(1)
    while f is not None:
        n += 1
        f = f.f_back
    return n


def use_loop(i, on):
    r = inner(i, on)
    if on:
        # `inner` has returned, so it is off the chain again: use_loop,
        # run_loop, report, main, <module>.
        seen[0] += depth()
    return r * 3 + i


def run_loop(n):
    acc = 0
    switch = n // 2
    for i in range(n):
        acc += use_loop(i, i >= switch and i % 7 == 0)
    return acc % 1000003


def gen(i, on):
    yield i
    force(1, 'i', on)
    yield i + 1
    force(1, 't', on)
    yield i + 2


def use_gen(i, on):
    t = 0
    for v in gen(i, on):
        t += v
    return t * 3 + i


def run_gen(n):
    acc = 0
    switch = n // 2
    for i in range(n):
        acc += use_gen(i, i >= switch and i % 7 == 0)
    return acc % 1000003


def report(label, fn, n):
    calls[0] = 0
    seen[0] = 0
    print(label, fn(n), calls[0], seen[0] % 1000003)


def main():
    report('loop', run_loop, N_LOOP)
    report('gen', run_gen, N_GEN)
    # A second pass enters the loops and bridges the first one compiled.
    report('loop', run_loop, N_LOOP)
    report('gen', run_gen, N_GEN)


main()
