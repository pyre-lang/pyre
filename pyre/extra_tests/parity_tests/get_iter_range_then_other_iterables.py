# CPython-suite gap: the suite does not run one hot `for` loop and one hot
# list comprehension over a `range` thousands of times and then feed the same
# code objects a list, a tuple, a user `__iter__`, a dict and other ranges.
# parity-tests reason: the trace specialises GET_ITER on the receiver being an
# exact `range`; that test has to be a guard in the trace, or a later
# non-range receiver is read through the range layout.

def f(x):
    t = 0
    for i in x:
        t += i
    return t

class It:
    def __iter__(self):
        return iter((7, 8))

r = range(5)
objs = [r] * 4000 + [[1, 2, 3], (4, 5), It(), {6: 0}, r, range(2, 6), range(0, 9, 3)] * 300
s = 0
k = 0
n = len(objs)
while k < n:
    s += f(objs[k])
    k += 1
assert s == 60700, s

def g(x):
    return [i for i in x]
s = 0
k = 0
while k < n:
    s += len(g(objs[k])) + sum(g(objs[k]))
    k += 1
assert s == 86700, s
print("OK")
