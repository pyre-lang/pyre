# pyre-check: max-pypy-ratio=20.5
#
# Floor is 20.5 / 6 = 3.417x. CI measured cranelift 12.8x (ubuntu) and
# 11.8x (macOS arm64), dynasm 7.1x (macOS arm64); local macOS arm64 dynasm
# 5.7x-5.9x. Ceiling = 1.6 x the largest. Before the recursive call folded
# to CALL_ASSEMBLER here, the same workload ran about 8x slower than now,
# past this ceiling.
#
# A self-recursive function defined and driven in code executed with a
# plain `{}` as globals.  Its `LOAD_GLOBAL` cannot fold to a constant, so
# the walker runs the globals lookup as a residual; an executed residual
# earlier in the body must not stop the recursive call at the inline bound
# from becoming a `CALL_ASSEMBLER`.
#
# Without that, every recursive call past the bound re-entered the portal
# through a residual call, ~60x slower than the same code under a module
# dict.
SRC = '''
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

total = 0
for i in range(30):
    total += fib(i)
'''


def main():
    g = {}
    exec(compile(SRC, "<fib>", "exec"), g)
    print(g["total"])


main()
