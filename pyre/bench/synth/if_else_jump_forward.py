# pyre-check: max-pypy-ratio=3
# Native CI readings span 0.8x (macos cranelift) to 1.9x (windows dynasm), so
# the ceiling is 3 and its derived floor 0.75 sits under the fastest of them.
# pyre-check: skip-cpython
# cpython 2.92s vs pyre 0.28s (10.4x on the ubuntu runner), and it is not
# gated on — only pypy is.
# Sized so pypy's own execution clears the measurement floor: below it the
# ratio gate divides by the floor and reads startup rather than this loop.
N = 85775500


def main():
    # `if cond: ... else: ...` inside a hot `while` loop.  The THEN block ends
    # with an unconditional forward jump over the ELSE body (JUMP_FORWARD), so
    # the loop body exercises JUMP_FORWARD on the hot (THEN) path.  The
    # condition is THEN-dominant (true 6/7 of the time) so the compiled trace
    # stabilizes on the THEN arm — which carries the forward jump — while the
    # ELSE arm stays live as an occasional bridge.  Pure integer arithmetic
    # keeps the result identical across CPython, PyPy, and pyre.
    i = 0
    acc = 0
    while i < N:
        if i % 7 != 0:
            acc = acc + (i % 17)
        else:
            acc = acc - (i % 31)
        i = i + 1
    print(acc)


main()
