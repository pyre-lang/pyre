# pyre-check: max-pypy-ratio=33.6
# Widest CI cranelift 29.2x (ubuntu); macOS cranelift 12.8x sits above the
# 8.4x floor. Ceiling = 29.2 x 1.15.
# Merged synth parity smoke suite: independent feature-level hot loops, each
# kept verbatim from its former standalone file with module-level names prefixed
# by the source name. Bug-repro / resume / kept-stack tests are NOT merged (they
# stay isolated so a miscompile is not diluted). check.py runs every *.py here.

# ── dict_delete ──
dict_delete__N = 300000

def dict_delete__main():
    i = 0
    acc = 0
    while i < dict_delete__N:
        d = {0: i, 1: i + 1, 2: i + 2}
        del d[1]
        acc = acc + d[0] + d[2] + len(d)
        i = i + 1
    print(acc)
dict_delete__main()

# ── dict_lookup_update ──
dict_lookup_update__N = 500000

def dict_lookup_update__main():
    d = {}
    i = 0
    while i < 128:
        d[i] = i * 3
        i = i + 1
    i = 0
    acc = 0
    while i < dict_lookup_update__N:
        k = i & 127
        acc = acc + d[k]
        d[k] = d[k] + 1
        i = i + 1
    print(acc + d[17] + len(d))
dict_lookup_update__main()

# ── set_literal ──
set_literal__N = 300000

def set_literal__main():
    acc = 0
    i = 0
    while i < set_literal__N:
        s = {i, i + 1, i + 2}
        acc = acc + len(s)
        i = i + 1
    print(acc)
set_literal__main()

# ── set_membership ──
set_membership__N = 700000

def set_membership__main():
    s = set()
    i = 0
    while i < 256:
        s.add(i * 3)
        i = i + 1
    i = 0
    acc = 0
    while i < set_membership__N:
        if i % 1024 in s:
            acc = acc + i
        else:
            acc = acc - (i & 15)
        i = i + 1
    print(acc + len(s))
set_membership__main()

# ── bool_float_list ──
bool_float_list__N = 1000000

def bool_float_list__main():
    i = 0
    acc = 0.0
    lst = [10, 20]
    flst = [1.5, 2.5]
    while i < bool_float_list__N:
        flag = i % 2 == 0
        acc = acc + flag
        acc = acc + flag * 1.5
        if flag < 0.5:
            acc = acc - 1.0
        acc = acc + lst[flag]
        acc = acc + flst[flag]
        lst[flag] = lst[flag] + 1
        i = i + 1
    print(int(acc), lst[0], lst[1])
bool_float_list__main()
