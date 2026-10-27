# pyre-check: max-pypy-ratio=36
N = 900000


class CountDown:
    def __init__(self, n):
        self.n = n

    def __iter__(self):
        return self

    def __next__(self):
        if self.n <= 0:
            raise StopIteration
        self.n = self.n - 1
        return self.n


def main():
    i = 0
    acc = 0
    while i < N:
        for x in CountDown(5):
            acc = acc + x
        i = i + 1
    print(acc)


main()
