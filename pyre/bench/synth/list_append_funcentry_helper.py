# pyre-check: max-pypy-ratio=125
# Function-entry `push` deopts on each list realloc. The orthodox
# append fold compiles two helper traces with 1 guard failure
# on dynasm and cranelift (was 5).
# The trip count puts pypy above the startup-subtraction floor, so this ratio
# is a measurement rather than pyre divided by the floor constant. The ceiling
# is twice the slowest backend observed (61.9x on cranelift).
# #171/#34: the orthodox list.append fold fires in function-entry (no-loop)
# helper traces, not only loop traces.  `push` is a no-loop helper called in a
# hot loop on two alternating receivers, so it traces from entry (header_pc==0)
# and its spare-capacity guard resume must reconstruct the alternating receiver
# correctly across the many realloc deopts — a wrong receiver box would route an
# append into the other list and corrupt the cross-checked sums below.
N = 1500000


def push(a, v):
    a.append(v)


def main():
    xs = []
    ys = []
    i = 0
    while i < N:
        push(xs, i)
        push(ys, -i)
        i = i + 1
    ok = (
        len(xs) == N
        and len(ys) == N
        and xs[0] == 0
        and ys[1] == -1
        and xs[N - 1] == N - 1
        and ys[N // 2] == -(N // 2)
    )
    print(sum(xs) + sum(ys), len(xs) + len(ys), ok)


main()
