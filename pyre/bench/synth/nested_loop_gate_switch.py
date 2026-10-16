# pyre-check: max-pypy-ratio=16
# Two nested hot loops with a guard-set switch keyed on the outer phase; the
# inner late arm (touching a None-or-int local) only executes once the global
# count crosses 5000, forcing bridge deopts mid-run. This once panicked on
# cranelift ("external JUMP target must be a registered LoopTargetDescr").
# Deterministic.
def inner(x, label, hold, gate):
    s = 0
    j = 0
    while j < 40:
        if (x + j) % 3 == 0:
            s += j
        elif (x + j) % 3 == 1:
            s += len(label)
        else:
            if gate and j % 17 == 0:
                if hold is None:
                    hold = j
                else:
                    hold = None
                s += 1 if hold is None else hold
            s += 2
        j += 1
    return s, hold


def main():
    total = 0
    label = "lo"
    hold = None
    count = 0
    outer = 0
    while outer < 1200:
        gate = count > 5000
        s, hold = inner(outer, label, hold, gate)
        total += s
        count += 40
        if outer % 50 == 0:
            label = "hihi" if (outer // 50) % 2 == 1 else "lo"
            hm = -1 if hold is None else hold
            print("chk", outer, count, total, label, hm)
        outer += 1
    hm = -1 if hold is None else hold
    print("final", total, count, label, hm)


main()
