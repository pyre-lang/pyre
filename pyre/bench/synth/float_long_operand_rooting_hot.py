# pyre-check: max-pypy-ratio=20
# Float ** long and long ** float: converting the long (`rbigint.tofloat`)
# allocates, so the nursery-born float must be read back from its root
# (`float_operands`; `W_FloatObject.descr_pow` `_to_float`). A stale
# operand is a forwarding word and SIGSEGVs under dynasm.
N = 25000
d = 2**250 + 12345
fd = float(d)


def run(n):
    s = 0
    i = 0
    while i < n:
        x = 1.5 + (i & 7)
        if x ** -d == x ** -fd:
            s += 1
        if d ** -x == fd ** -x:
            s += 1
        i += 1
    return s


print(run(N))
