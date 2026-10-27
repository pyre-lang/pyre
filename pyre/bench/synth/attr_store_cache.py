# pyre-check: max-pypy-ratio=130
N = 960000


class C:
    pass


def main():
    total = 0
    i = 0
    while i < N:
        o = C()
        o.x = i          # STORE_ATTR add-new: cached transition after warmup
        o.y = i + 1      # second add-new transition
        o.x = o.x + 10   # LOAD hit + STORE in-place write (cache hit)
        total = total + o.x + o.y
        i = i + 1
    print(total)


main()
