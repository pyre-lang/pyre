# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=entry-bridge:leaf,entry-bridge:fib,loop:drive
# Inlined leaf reads K (branches->bridges); self-recursive fib reads K.
# Each exec'd under a module dict then two plain dicts with different K.
# LOAD_GLOBAL follows get_w_globals: debugdata override, else pycode.w_globals.
import types

SRC = """
def leaf(i):
    if i % 7 == 0:
        return K * 3
    if i % 5 == 0:
        return K + i
    return K
def mid(i):
    return leaf(i) + leaf(i + 1)
def drive(n):
    t = 0
    for i in range(n):
        t += mid(i)
    return t

def fib(n):
    if n < 2:
        return n * K
    if n % 3 == 0:
        return fib(n - 1) + fib(n - 2) + 0
    return fib(n - 1) + fib(n - 2)

drive_total = drive(3000)
fib_total = sum(fib(i) for i in range(22))
"""


def main():
    code = compile(SRC, "<inline-globals>", "exec")
    ks = {"module": 1, "plain": 1000, "plain2": 77}
    for name, g in (
        ("module", types.ModuleType("m").__dict__),
        ("plain", {}),
        ("plain2", {}),
    ):
        g["K"] = ks[name]
        exec(code, g)
        print(name, g["drive_total"], g["fib_total"])
    print("PASS")


main()
