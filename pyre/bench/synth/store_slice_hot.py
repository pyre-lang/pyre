# pyre-check: max-pypy-ratio=32
# The ceiling is twice the slowest ratio observed (15.8x on the linux runner),
# rounded up.
# STORE_SLICE in a hot loop: `buf[a:b] = [i, i + 1]` with *variable* bounds
# (a, b are locals, not literals) compiles to the STORE_SLICE opcode every
# iteration — literal bounds would fold to a const slice + STORE_SUBSCR.  The
# compiled per-CodeObject jitcode walks the store_slice residual (a `slice`
# object through setitem on the peeked list) instead of an abort_permanent
# marker, so the hot body JIT-compiles rather than declining to the trait leg.
# Output is verified against CPython/PyPy.
N = 1200000


def main():
    total = 0
    i = 0
    a = 1
    b = 3
    buf = [0, 0, 0, 0, 0, 0]
    while i < N:
        buf[a:b] = [i, i + 1]
        total = total + buf[1] + buf[2]
        i = i + 1
    print(total)


main()
