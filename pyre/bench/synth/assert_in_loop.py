# pyre-check: max-pypy-ratio=17
# N/ITERS are sized to prove the opcode compiles and to drive the compiled
# loop thousands of times, not to race pypy.
N = 600
ITERS = 48000


def run(n):
    # `assert i >= 0` compiles to LOAD_COMMON_CONSTANT(AssertionError).
    # Before that opcode was lowered, its abort_permanent marker declined
    # the whole loop, forcing the interpreter.
    total = 0
    i = 0
    while i < n:
        assert i >= 0
        total += i
        i += 1
    return total


def main():
    total = 0
    for _ in range(ITERS):
        total += run(N)
    print(total)


main()
