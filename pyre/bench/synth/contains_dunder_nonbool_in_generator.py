# A user `__contains__` returning a non-bool, inlined from a generator body
# driven from the caller's loop; the decline must record the residual
# CONTAINS_OP, not abort the trace after the generator's FOR_ITER advanced.

class Bag:
    def __init__(self, k):
        self.k = k
    def __contains__(self, x):
        return x % self.k

def gen(n):
    for i in range(n):
        yield (3 in Bag(i % 4 + 2))

def main():
    t = 0
    for v in gen(3000):
        t += v
    print(t)

main()
