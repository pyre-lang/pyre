# pyre-check: max-pypy-ratio=29
# Polymorphic accumulator slots cycling int/str/None in a hot loop, with rare
# retype arms (int->str, str->None) opening after warm-up so the compiled
# trace takes bridges. A deopt-resumed frame once pushed past the valuestack
# capacity (load_small_int OOB) on this shape. Deterministic.
def step(state, i):
    a, b, c = state
    if i % 5 == 0:
        a += i % 7
    elif i % 5 == 1:
        if b is None:
            b = "n"
        else:
            b = "s" if (i & 2) else "t"
        a += ord(b[0]) % 5
    elif i % 5 == 2:
        if isinstance(c, int):
            c += 2
        else:
            c = len(c)
        a -= 1
    else:
        a += 1
    return a, b, c


def main():
    a = 0
    b = None
    c = 0
    total = 0
    n = 52000
    for i in range(n):
        a, b, c = step((a, b, c), i)
        if i > 6000:
            if i % 101 == 0:
                c = "w" * ((i % 3) + 1)
            if i % 113 == 0:
                b = None
        if isinstance(c, str):
            total += len(c)
        else:
            total += c % 3
        if i % 4000 == 0:
            bs = "-" if b is None else b
            cs = c if isinstance(c, str) else str(c % 1000)
            print("chk", i, a % 100000, bs, cs, total)
    print("final", a, total, b is None, c if isinstance(c, str) else c % 9973)


main()
