# Fail-image overlay of a newbool / COMPARE / TO_BOOL guard on frames whose
# locals_cells_stack_w stack base is co_nlocals + ncellvars + nfreevars
# (PyFrame._stack_start). Portal and inlined callee both carry cells and
# freevars; 1-tuple COMPARE_EQ (W_TupleObject, not a specialised pair)
# flips after the loop is compiled so the fail image is the other arm.
N = 8000
LIM = 5000


def outer(k):
    acc = [0]
    t = k

    def step(i, lim):
        captured = lim
        if i < 0:
            def bump(n):
                return captured + n

            return bump(0)
        x = i + t
        flag = i % 1000 == 999
        s = captured
        if flag:
            s = s + 1
        left = (i,)
        right = (lim,)
        if x > lim:
            s = s + (1 if left == right else 2)
        else:
            s = s + (1 if left == right else 2)
        a = True
        b = not flag
        while a and b:
            a = False
            s = s + 1
        acc[0] = acc[0] + s
        return s

    return step, acc, t


def make_driver(seed):
    def driver():
        box = [0]
        if seed < 0:
            def grab():
                return box[0]

            return grab()
        f, acc, t = outer(seed)
        i = 0
        total = 0
        lim = LIM
        while i < N:
            if i % 1000 == 999:
                total = total + 3
            x = i
            left = (x,)
            right = (lim,)
            total = total + (1 if left == right else 2) + f(i, lim)
            flag = i & 1
            if not flag:
                total = total + box[0]
            box[0] = box[0] + (i & 3)
            a = True
            b = i < lim
            while a and b:
                a = False
                total = total + 1
            i = i + 1
        cell_sum = 0
        for c in f.__closure__:
            v = c.cell_contents
            if type(v) is list:
                cell_sum = cell_sum + v[0]
            else:
                cell_sum = cell_sum + v
        return (total, box[0], acc[0], t, seed, cell_sum)

    return driver


def main():
    print(make_driver(7)())


main()
