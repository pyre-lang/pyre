class C:
    def __init__(self, pos):
        self._pos = pos

    def m(self, i):
        return self._pos + i


def main(n):
    c = C(7)
    acc = 0
    for i in range(n):
        acc += c.m(i)
    return acc


print(main(3600000))
