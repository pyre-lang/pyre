# The forward __add__ is the last _invoke_binop arm (same type), so its body
# stores are recorded; a later NotImplemented must raise the TypeError
# without running __add__ a second time.


class Acc:
    def __init__(self, n):
        self.n = n
        self.calls = 0

    def __add__(self, other):
        self.calls += 1
        if other.n < 0:
            return NotImplemented
        return Acc(self.n + other.n)


class P:
    def __pow__(self, other):
        self.seen = True
        return NotImplemented


class L:
    pass


class R:
    def __init__(self):
        self.calls = 0

    def __radd__(self, other):
        self.calls += 1
        if self.calls % 1000 == 0:
            return NotImplemented
        return 1


def run(n):
    a = Acc(0)
    one = Acc(1)
    neg = Acc(-1)
    errors = 0
    total = 0
    msg = None
    for i in range(n):
        try:
            total += (a + (neg if i % 1000 == 999 else one)).n
        except TypeError as e:
            errors += 1
            msg = str(e)
    p = P()
    pmsg = None
    for i in range(n // 10):
        try:
            p ** p
        except TypeError as e:
            pmsg = str(e)
    l = L()
    r = R()
    rtotal = 0
    rmsg = None
    rerrors = 0
    for i in range(n):
        try:
            rtotal += l + r
        except TypeError as e:
            rerrors += 1
            rmsg = str(e)
    return a.calls, errors, total, msg, pmsg, r.calls, rerrors, rtotal, rmsg


print(run(60000))
