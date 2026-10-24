# pyre-check: max-pypy-ratio=4.9
# pyre-check: spec-folds=binary_op_descent,compare_op_descent
# pyre-check: trace-shape=bool_arithmetic__main:present=GuardClass,present=GuardNoException
# Merged synth parity smoke suite: independent feature-level hot loops, each
# kept verbatim from its former standalone file with module-level names prefixed
# by the source name. Bug-repro / resume / kept-stack tests are NOT merged (they
# stay isolated so a miscompile is not diluted). check.py runs every *.py here.

# ── int_arithmetic ──
int_arithmetic__N = 19047620

def int_arithmetic__main():
    i = 1
    acc = 0
    while i < int_arithmetic__N:
        acc = acc + i
        acc = acc ^ i << 1
        acc = acc - (i >> 2)
        acc = acc + i % 97
        i = i + 1
    print(acc)
int_arithmetic__main()

# ── float_arithmetic ──
float_arithmetic__N = 7619048

def float_arithmetic__main():
    i = 0
    x = 1.0
    y = 0.25
    while i < float_arithmetic__N:
        x = x + y
        y = y * 1.000001 + 3e-06
        x = x / 1.000002
        if x > 1000.0:
            x = x - 999.5
        i = i + 1
    print(int(x * 1000.0) + int(y * 1000.0))
float_arithmetic__main()

# ── bool_arithmetic ──
bool_arithmetic__N = 14285715

def bool_arithmetic__main():
    i = 0
    acc = 0
    while i < bool_arithmetic__N:
        flag = i % 2 == 0
        other = i % 3 == 0
        acc = acc + flag
        acc = acc + flag * 2
        acc = acc - other
        acc = acc + -flag
        # CPython 3.14 warns for every `~bool` evaluation.  Express the same
        # integer contribution directly so this arithmetic benchmark does not
        # measure 1.5 million warning-registry lookups.
        acc = acc - other - 1
        if flag & other:
            acc = acc + 1
        if flag | other:
            acc = acc + 2
        if flag ^ other:
            acc = acc + 4
        acc = acc + (flag & 3)
        if flag < other:
            acc = acc + 8
        i = i + 1
    print(acc)
bool_arithmetic__main()

# ── bool_compare ──
bool_compare__N = 14285715

def bool_compare__main():
    i = 0
    acc = 0
    while i < bool_compare__N:
        a = i % 17
        b = i % 31
        if a < b and b != 0 or a == 13:
            acc = acc + a
        else:
            acc = acc - b
        i = i + 1
    print(acc)
bool_compare__main()

# ── is_op ──
is_op__N = 1904762

def is_op__main():
    a = object()
    b = object()
    acc = 0
    i = 0
    while i < is_op__N:
        c = a if i & 1 == 0 else b
        if c is a:
            acc = acc + 1
        else:
            acc = acc + 2
        if c is not b:
            acc = acc + 10
        i = i + 1
    print(acc)
is_op__main()

# ── unary_not ──
unary_not__N = 1904762

def unary_not__main():
    acc = 0
    i = 0
    while i < unary_not__N:
        acc = acc + (not i & 1)
        i = i + 1
    print(acc)
unary_not__main()

# ── unary_invert ──
unary_invert__N = 1904762

def unary_invert__main():
    acc = 0
    i = 0
    while i < unary_invert__N:
        acc = acc + ~i
        i = i + 1
    print(acc)
unary_invert__main()
