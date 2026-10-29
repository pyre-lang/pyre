# CPython-suite gap: `test_set` raises from `__hash__` in a set display only
# once, never inside a loop hot enough to be compiled, so it does not count
# the calls a compiled `{...}` makes.
#
# A set display hashes each element exactly once, also when one of the
# hashes raises and the display is abandoned for the handler.
#
# parity-tests reason: a guard that resumes before the hash residual runs
# the user `__hash__` a second time; only the call count shows it.

N = 20000

CALLS = [0]


class K:
    def __init__(self, v):
        self.v = v

    def __hash__(self):
        CALLS[0] += 1
        if self.v % 997 == 0:
            raise ValueError
        return self.v


def main():
    acc = 0
    errs = 0
    for i in range(N):
        try:
            s = {K(i), K(i + 1)}
            acc += len(s)
        except ValueError:
            errs += 1
    print(acc, errs, CALLS[0])
    print("OK")


main()
