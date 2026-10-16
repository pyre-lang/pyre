# No `max-pypy-ratio`: pypy's exec time here sits at the startup-subtraction
# floor, so the printed ratio is a quotient by a clamped denominator rather
# than a measurement, and which side of the floor it lands on flips run to run
# -- which decides whether the ceiling is applied at all.  A ceiling read off
# such a number gates noise; 175 of the 463 fixtures state none for the same
# reason, and the ratio is still printed on every run.  This fixture's gates
# are its three-backend jit-stats baselines.
N = 500000


# Custom operands enter through the binary `+` and rich-compare `<` value
# helpers.  The walker resolves each user dunder and gives it its own inlined
# frame.  Both dunders finish that frame by raising every iteration, and the
# caller catches the exception.  The three jit-stats baselines therefore gate
# one caller loop with no separately compiled dunder portals or bridges, as on
# the PyPy oracle, while the printed result checks both unwind paths.
class Boom:
    def __add__(self, other):
        raise ValueError("add")

    def __lt__(self, other):
        raise ValueError("lt")


def main():
    b = Boom()
    acc = 0
    i = 0
    while i < N:
        try:
            b + 1
        except ValueError:
            acc = acc + 1
        try:
            b < 1
        except ValueError:
            acc = acc + 1
        i = i + 1
    print(acc)


main()
