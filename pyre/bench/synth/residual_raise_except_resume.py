# pyre-check: max-pypy-ratio=3
# Inlining the except-as-return callee (`warmstate.py can_inline_callable`,
# `pyjitpl.py perform_call`) brings the measured ratio to about 1x
# (dynasm 0.9-1.3x, wasm 2.0x), so the ceiling is 3 (floor 0.75x).
# pyre-check: skip-cpython
# cpython 1.67s vs pyre 0.13s (13x), and it is not gated on — only pypy is.
# A hot loop FOLLOWED by a try/except whose body raises through a
# *may-force residual call* (`//`, subscript) rather than an explicit
# `raise`.  The try/except is reached only after the loop's exit guard
# fails, so the blackhole walks forward and executes the residual call
# concretely on the resume path.  When that call raises, the exception
# must be routed to the in-frame `catch_exception` adjacency the
# codewriter emitted right after the call.  A catch search that starts at
# the call's operand byte (the dispatch loop advances the position only
# after the handler returns) never finds the catch, and the exception
# escapes the handler.  This bench pins that a
# residual-call raise caught on the resume path returns byte-identically.
#
# `list_extend_resume` additionally builds a heap-local list via
# LIST_EXTEND (`[*base]`) *after* the loop, so the loop-exit guard
# failure resumes through the LIST_EXTEND: the walk must execute the
# `list_extend` residual.  Without a walker handler LIST_EXTEND emits an
# abort that, reached on the resume walk, invalidates the live frame and
# crashes (SIGSEGV).
#
# N is sized so PyPy's startup-subtracted user-CPU stays well clear of its own
# empty-program startup. The gate divides by that difference, so a workload
# finishing near startup makes the denominator a residue of two similar
# measurements: at N=2000000 PyPy's whole run is shorter than its startup, the
# difference clamps to the 5ms floor, and unchanged code reports anywhere from
# 3x to 18x depending on runner speed. At this N the ratio is 1.8x (dynasm) /
# 1.9x (cranelift), which is the value the gate is meant to bound.
N = 67110700


def divide_by_zero_const(n):
    i = 0
    while i < n:
        i = i + 1
    try:
        return 1 // 0
    except ZeroDivisionError:
        return -2


def divide_by_zero_computed(n):
    i = 0
    acc = 0
    while i < n:
        acc = acc + 1
        i = i + 1
    d = acc - acc
    try:
        return acc // d
    except ZeroDivisionError:
        return acc + 1


def modulo_by_zero(n):
    i = 0
    while i < n:
        i = i + 1
    try:
        return i % 0
    except ZeroDivisionError:
        return i - 5


def tuple_index_resume(n):
    i = 0
    while i < n:
        i = i + 1
    try:
        return (10, 20, 30)[i]
    except IndexError:
        return -7


def list_extend_resume(n):
    i = 0
    while i < n:
        i = i + 1
    base = [10, 20, 30]
    data = [*base]
    try:
        return data[i]
    except IndexError:
        return data[0] - 100


def main():
    print(divide_by_zero_const(N))
    print(divide_by_zero_computed(N))
    print(modulo_by_zero(N))
    print(tuple_index_resume(N))
    print(list_extend_resume(N))


main()
