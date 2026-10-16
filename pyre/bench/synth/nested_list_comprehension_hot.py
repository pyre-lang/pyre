# pyre-check: max-pypy-ratio=44
# Two function-entry comprehension loops (`single_comp`, `pair_comp`); each
# virtualizes an inner list whose backing is NewArray/NewArrayClear. wasm
# compiles each as its own module.
# An inlined list comprehension whose LIST_APPEND element is a non-empty nested
# list (`[[i] …]` / `[[i, i + 1] …]`). The inner list is virtual; its
# NewArray / NewArrayClear backing has no jitcode-liveness color. The append
# body is not a speculative-replay sub-walk, so that backing is bound at every
# guard-exit deopt without an extra resume-data root. dynasm, cranelift, and
# wasm print the same total.


def single_comp(n):
    return [[i] for i in range(n)]


def pair_comp(n):
    return [[i, i + 1] for i in range(n)]


def main():
    total = 0
    k = 0
    while k < 2000:
        xs = single_comp(1000)
        total += len(xs)
        total += xs[-1][0]
        ys = pair_comp(1000)
        total += len(ys)
        total += ys[-1][1]
        k += 1
    print(total)


main()
