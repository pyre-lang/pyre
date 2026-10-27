# pyre-check: max-pypy-ratio=32
# A callee whose exception handler is large enough that its jumps need
# EXTENDED_ARG, inlined into a hot caller loop. The walker dequeues blocks
# non-sequentially; a block ending on an EXTENDED_ARG code unit must not fold
# its stale high bits into the next block's first instruction argument. If the
# running EXTENDED_ARG accumulator leaked across the block boundary, decoding
# the next block's opcode arg (e.g. COMPARE_OP) with a too-large value aborts
# JIT codegen (Arg::try_from -> InvalidBytecode). Output verified against
# CPython/PyPy.
N = 5760000


def f(i, v2):
    try:
        raise IndexError
    except KeyError:
        for j in range(2):
            if i > 3:
                v2 = (v2 + 0) % 1000000007
                v2 = (v2 + 1) % 1000000007
                v2 = (v2 + 2) % 1000000007
                v2 = (v2 + 3) % 1000000007
                v2 = (v2 + 4) % 1000000007
                v2 = (v2 + 5) % 1000000007
                v2 = (v2 + 6) % 1000000007
                v2 = (v2 + 7) % 1000000007
                v2 = (v2 + 8) % 1000000007
                v2 = (v2 + 9) % 1000000007
                v2 = (v2 + 10) % 1000000007
                v2 = (v2 + 11) % 1000000007
                v2 = (v2 + 12) % 1000000007
                v2 = (v2 + 13) % 1000000007
                v2 = (v2 + 14) % 1000000007
                v2 = (v2 + 15) % 1000000007
                v2 = (v2 + 16) % 1000000007
                v2 = (v2 + 17) % 1000000007
                v2 = (v2 + 18) % 1000000007
                v2 = (v2 + 19) % 1000000007
            try:
                v2 = (i & 5) % 1000000007
            except IndexError:
                v2 = (i + 1) % 1000000007


def run(n):
    acc = 0
    for i in range(n):
        try:
            f(i, i % 5)
        except IndexError:
            acc = (acc + 33) % 1000000007
    return acc


print(run(N))
