# pyre-check: max-pypy-ratio=9.2
#
# Floor is 9.2 / 4 = 2.3x (below the local dynasm minimum 2.6x). Local
# macOS arm64 dynasm 2.6x-3.0x, cranelift 3.0x-3.7x. Estimated CI
# cranelift about 3.0x x 2.2 = 6.6x (sibling: CI cranelift 12.8x vs local
# dynasm 5.8x). Ceiling 9.2 leaves 1.4x over that estimate. Before the
# fix the workload ran 13-28s, far past this ceiling.
#
# One code object is exec'd first under a module dict, publishing
# PyCode.w_globals, then under a plain dict. A function whose code
# already recorded a different first globals used to residualize every
# call as opaque bh_call_fn; the inline / CALL_ASSEMBLER frame now
# carries FrameDebugData.w_globals the way PyFrame.__init__ does.
import types

SRC = """
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

total = 0
for i in range(29):
    total += fib(i)
"""


def main():
    code = compile(SRC, "<fib>", "exec")
    g1 = types.ModuleType("m").__dict__
    exec(code, g1)
    print(g1["total"])
    g2 = {}
    exec(code, g2)
    print(g2["total"])
    print("PASS")


main()
