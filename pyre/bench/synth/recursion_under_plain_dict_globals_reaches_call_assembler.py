# A self-recursive function defined and driven in code executed with a
# plain `{}` as globals.  Its `LOAD_GLOBAL` cannot fold to a constant, so
# the walker runs the globals lookup as a residual; that lookup reads only
# the dict's own str-keyed storage, so it must not stop the recursive call
# at the inline bound from becoming a `CALL_ASSEMBLER`.
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
for i in range(27):
    total += fib(i)
'''


def main():
    g = {}
    exec(compile(SRC, "<fib>", "exec"), g)
    print(g["total"])


main()
