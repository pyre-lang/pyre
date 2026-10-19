# pyre-check: max-pypy-ratio=40
# Hot loop calls a small non-portal function that does `return K(i)` where
# `K` is a module-level class. Mid-loop rebind of `K` to another class must
# be observed: the inlined callee's LOAD_GLOBAL reads that frame's
# `get_w_globals()` (`pyframe.py PyFrame.get_w_globals`) through the module
# dict `version?` cell (`celldict.py ModuleDictStrategy.getitem_str`), so
# the rebind invalidates the loop. A codewrite-time bake of `K` keeps
# returning instances of the stale class.
N = 400000


class A:
    def __init__(self, i):
        self.i = i

    def kind(self):
        return 1


class B:
    def __init__(self, i):
        self.i = i

    def kind(self):
        return 2


K = A


def make(i):
    return K(i)


def main():
    global K
    total = 0
    for i in range(N):
        total += make(i).kind()
        if i == N // 2:
            K = B
    print(total)


main()
