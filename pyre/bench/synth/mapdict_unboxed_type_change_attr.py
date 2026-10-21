# pyre-check: max-pypy-ratio=25
# Changing an unboxed int or float slot to the other type creates a boxed map;
# the promoted-map guard must deopt before the old raw-storage read or write is
# reused for the new representation (mapdict.py, 600-619, 905-916).
#
# The three backends record the same counters here. `ALLOW_UNBOXING_INTS` holds
# on every target, so `run_int`'s slot is unboxed and its read folds under the
# guest exactly as it does natively; a wasm baseline that drifts from the two
# native ones means the int arm stopped unboxing there.
class IntSlot:
    def __init__(self):
        self.x = 0


class FloatSlot:
    def __init__(self):
        self.x = 0.0


def run_int(n):
    obj = IntSlot()
    total = 0
    i = 0
    while i < n:
        obj.x = obj.x + 1
        if i == n // 2:
            obj.x = 1.5
        total += int(obj.x)
        i += 1
    return total, obj.x


def run_float(n):
    obj = FloatSlot()
    total = 0.0
    i = 0
    while i < n:
        obj.x = obj.x + 1.0
        if i == n // 2:
            obj.x = 7
        total += float(obj.x)
        i += 1
    return total, obj.x


print(run_int(3000000))
print(run_float(3000000))
