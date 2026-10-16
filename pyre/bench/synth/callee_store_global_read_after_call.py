# pyre-check: max-pypy-ratio=40
# A residual Python call mutates an IntMutableCell in place.  The caller must
# reload the global after the call even though the module-dict binding and its
# quasi-immutable version did not change.
MOD = 999983
COUNT = 2369550
VALUE = 48
FLAG = 1


def step():
    global VALUE, FLAG
    old = VALUE
    VALUE = (VALUE + 1) % MOD
    if old % 251 == 0:
        FLAG = 1 - FLAG
    return old


def run():
    total = 0
    for _ in range(COUNT):
        total = (total + step()) % MOD
        if FLAG == 1:
            total = (total + VALUE) % MOD
    return total


result = run()
print((result + VALUE * 13 + FLAG * 17) % MOD)
