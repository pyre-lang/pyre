# pyre-check: max-pypy-ratio=16
# Readings 5.0x-8.1x (macos dynasm 5.0x failed floor 7.5 against 30, run 37242176195). Floor 4x.
# pyre-check: jitstats-band=guard_failures=6
# The JitCounter decay (`counter.py` `JitCounter.decay_all_counters`, run every
# 32 minor collections) makes `guard_failures` a function of how much the
# process has allocated. When wasm JITFRAMEs were sized to the trace, the
# guest's minor collections changed and wasm read 2130 against dynasm and
# cranelift's 2122, at the same 7 loops / 11 bridges. With `decay=0` all three
# backends read 2111.
#
# CPython (the oracle) has no `pypyjit`; PyPy and pyre do. `set_param` rather
# than an environment variable because the wasm guest sees no environment.
try:
    import pypyjit

    pypyjit.set_param("decay=0")
except ImportError:
    pass

# Merged synth parity smoke suite: independent feature-level hot loops, each
# kept verbatim from its former standalone file with module-level names prefixed
# by the source name. Bug-repro / resume / kept-stack tests are NOT merged (they
# stay isolated so a miscompile is not diluted). check.py runs every *.py here.

# ── comprehensions ──
comprehensions__N = 70000

def comprehensions__main():
    i = 0
    acc = 0
    while i < comprehensions__N:
        xs = [j * j for j in range(8) if j + i & 1 == 0]
        d = {j: j + i for j in xs}
        acc = acc + sum(xs) + len(d)
        i = i + 1
    print(acc)
comprehensions__main()

# ── for_range_loop ──
for_range_loop__N = 600000

def for_range_loop__main():
    acc = 0
    for i in range(for_range_loop__N):
        acc = acc + i % 19
    for i in range(10, 200000, 3):
        acc = acc - i % 11
    print(acc)
for_range_loop__main()

# ── while_nested_break_continue ──
while_nested_break_continue__N = 60000

def while_nested_break_continue__main():
    i = 0
    acc = 0
    while i < while_nested_break_continue__N:
        j = 0
        while j < 12:
            if j == 5:
                j = j + 1
                continue
            if j == 10:
                break
            acc = acc + (i + j) % 23
            j = j + 1
        i = i + 1
    print(acc)
while_nested_break_continue__main()

# ── nested_foriter_sum ──
def nested_foriter_sum__main():
    total = 0
    n = 0
    while n < 20000:
        for x in [1, 2, 3, 4, 5]:
            for y in range(x):
                total += y
        n += 1
    return total
nested_foriter_sum__result = nested_foriter_sum__main()
print(nested_foriter_sum__result)
