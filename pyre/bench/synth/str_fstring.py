# pyre-check: max-pypy-ratio=16
# pyre-check: spec-folds=format_with_spec_int
# pyre-check: trace-shape=fstring_simple__main:absent=CallMayForceR,convert_value__main:absent=CallMayForceR,fstring_spec__main:present=Copystrcontent,string_ops__main:absent=CallMayForceR
# Merged synth parity smoke suite: independent feature-level hot loops, each
# kept from its former standalone file with module-level names prefixed by the
# source name. Bug-repro / resume / kept-stack tests are NOT merged (they stay
# isolated so a miscompile is not diluted). check.py runs every *.py here.
#
# Every trip count is half what the standalone files used, which is the one
# deliberate departure from them. At the old counts this file's old-gen use
# crossed the major-collection threshold check.py pins, and the resulting
# eval-breaker bailout re-entered through a bridge whose guard then failed --
# one extra `guard_failures` that appeared or not depending on where the
# crossing landed. That placement moves with total allocation volume, so it
# differed per backend, per host, and per environment size, and no recorded
# baseline could hold: the counter read 657 or 658 for reasons having nothing
# to do with the code under test. Halved, the file does not reach the threshold
# at all (`back_edge_polls=0`) and both backends read 657.
#
# Guard failures arise at trace transitions rather than per iteration, so
# `guard_failures`, `loops_compiled` and `bridges_compiled` are unchanged by the
# halving -- measured 657/6/3 on both native backends at 0.4x, 0.5x and 0.6x,
# against 657 (dynasm) and 658 (cranelift) at the old counts. Trip counts still
# clear the compile threshold by two orders of magnitude. Keep them well below
# the crossing: 0.7x reaches it again.

# ── fstring_simple ──
fstring_simple__N = 100000

def fstring_simple__main():
    acc = 0
    i = 0
    while i < fstring_simple__N:
        s = f'{i}'
        acc = acc + len(s)
        i = i + 1
    print(acc)
fstring_simple__main()

# ── fstring_multi ──
fstring_multi__N = 100000

def fstring_multi__main():
    acc = 0
    i = 0
    while i < fstring_multi__N:
        s = f'{i}-{i}'
        acc = acc + len(s)
        i = i + 1
    print(acc)
fstring_multi__main()

# ── fstring_spec ──
fstring_spec__N = 100000

def fstring_spec__main():
    acc = 0
    i = 0
    while i < fstring_spec__N:
        s = f'{i:05d}'
        acc = acc + len(s)
        i = i + 1
    print(acc)
fstring_spec__main()

# ── convert_value ──
convert_value__N = 100000

def convert_value__main():
    acc = 0
    i = 0
    while i < convert_value__N:
        s = f'{i!r}-{i!s}-{i!a}'
        acc = acc + len(s)
        i = i + 1
    print(acc)
convert_value__main()

# ── string_ops ──
string_ops__N = 125000

def string_ops__main():
    words = ['alpha', 'beta', 'gamma', 'delta']
    i = 0
    acc = 0
    while i < string_ops__N:
        s = words[i & 3]
        t = s + ':' + str(i & 255)
        if t.startswith('a') or t.endswith('7'):
            acc = acc + len(t)
        else:
            acc = acc - len(s)
        i = i + 1
    print(acc)
string_ops__main()

# ── bytes_ops ──
bytes_ops__N = 175000

def bytes_ops__main():
    data = b'abcdefghijklmnopqrstuvwxyz'
    i = 0
    acc = 0
    while i < bytes_ops__N:
        b = data[i % len(data)]
        piece = data[i & 7:(i & 7) + 5]
        acc = acc + b + len(piece)
        i = i + 1
    print(acc)
bytes_ops__main()
