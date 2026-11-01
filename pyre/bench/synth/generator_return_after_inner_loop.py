# A loop nested inside a yield-bearing loop, then a `return` once the
# outer counter hits its bound. The inner loop's body holds no suspension
# — it runs to completion between two yields — so it is an ordinary counted
# loop and compiles like one. After that loop the frame sometimes RETURNs
# rather than yielding; FOR_ITER must treat a finished frame as loop exit
# (`_send_ex` `frame_finished_execution`) instead of binding the portal's
# implicit None as an item.
def g(m, rounds):
    r = 0
    while True:
        total = 0
        i = 0
        while i < m:
            total += i
            i += 1
        if r >= rounds:
            return
        yield total
        r += 1


def main():
    acc = 0
    n = 0
    for k in range(400):
        for v in g(50 + (k & 3), 6):
            acc += v
            n += 1
    assert (acc, n) == (3122400, 2400), (acc, n)
    print(acc, n)


main()
