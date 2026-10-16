# pyre-check: max-pypy-ratio=28
# gh#498 guard: a branch resume must keep the current closure-call result
# local, across the three freevar flavours.
#
#   readonly   — the closure only reads a freevar;
#   list_cell  — it appends to a list held in a cell;
#   nonlocal   — it rebinds the freevar itself.
#
# Each flavour keeps its own driver loop so the closure stays a direct call
# from the hot loop, as it was when these lived in separate files.
N = 240000


def run_readonly():
    base = 10
    acc = 0

    def step(k):
        if k == 2:
            return base + 190
        return -1

    i = 0
    while i < N:
        v = step(i % 5)
        acc += v if v != -1 else 0
        i += 1
    return acc


def run_list_cell():
    buf = []
    acc = 0

    def step(k):
        buf.append(k)
        if k < 0:
            return 0
        if k == 2:
            return 200
        return -1

    i = 0
    while i < N:
        v = step(i % 5)
        acc += v if v != -1 else 0
        i += 1
    return acc, len(buf)


def run_nonlocal():
    n = 0
    acc = 0

    def step(k):
        nonlocal n
        n = n + 1
        if k < 0:
            return 0
        if k == 2:
            return 200
        return -1

    i = 0
    while i < N:
        k = i % 5
        v = step(k)
        acc += v if v != -1 else 0
        i += 1
    return acc, n


print(run_readonly())
print(run_list_cell())
print(run_nonlocal())
