# pyre-check: max-pypy-ratio=6
# Readings 1.7x-3.9x (macOS cranelift 1.7x-2.1x, Linux 2.6x-3.3x, wasm 3.8x-3.9x,
# Windows 2.7x-3.4x); the floor, a fourth of the ceiling, sits under the fastest.
# Bare `raise` (RERAISE) inside a handler re-raises the exception currently
# being handled, which an outer handler then catches. This exercises the
# current-exception slot on the re-raise path: the bare raise must pick up the
# active exception from the slot, and after the inner handler unwinds the slot
# must restore to the outer/None prev. Alternating the re-raise by iteration
# parity keeps both the re-raised and the locally-finished paths hot.
#   even i -> bare re-raise -> outer except (reraised++)
#   odd  i -> handled locally               (caught++)
# With N=30000: caught=15000, reraised=15000. Deterministic.
N = 112420646


def run(n):
    caught = 0
    reraised = 0
    i = 0
    while i < n:
        try:
            try:
                raise ValueError("x")
            except ValueError:
                if (i & 1) == 0:
                    raise
                caught += 1
        except ValueError:
            reraised += 1
        i += 1
    return caught, reraised


def main():
    c, r = run(N)
    print(c, r)


main()
