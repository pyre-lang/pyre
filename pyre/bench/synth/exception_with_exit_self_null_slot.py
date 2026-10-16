# A `with` block whose handler runs on most iterations of a hot `while` loop.
# LOAD_SPECIAL pushes the resolved `__exit__` and the call's `self_or_null`
# slot; that NULL is a live value, and the dense virtualizable shadow cannot
# tell it from an unwritten slot. Carried as a mirror hole rather than as a
# constant, the slot was omitted from the resume image and a blackhole
# resuming into WITH_EXCEPT_START read the bound `__exit__` the same opcode had
# pushed one slot below, calling it with the receiver twice
# ("__exit__() takes 4 positional arguments but 5 were given").
#
# No FOR_ITER: this is the shape that reaches the defect with no opcode gate
# involved, so it stays a witness independently of what the FOR_ITER body scan
# admits. Kept at module scope, where it was first reproduced.
N = 720000


class Context:
    def __init__(self):
        self.exits = 0

    def __enter__(self):
        return 3

    def __exit__(self, exc_type, exc_value, traceback):
        self.exits += 1
        return exc_type is ValueError


context = Context()
total = 0
i = 0
while i < N:
    with context as value:
        total += value
        if i % 10 == 0:
            raise ValueError
        total += i
    i += 1
print(total, context.exits)
# The line carries 3*N + sum(i for i in range(N) if i % 10), then N.
# Expected: 233282160000 720000
