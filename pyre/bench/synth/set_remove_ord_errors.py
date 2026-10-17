# pyre-check: max-pypy-ratio=3.5
# set.remove(missing) raises KeyError carrying the missing key itself (so its
# repr shows), matching dict[missing]; ord() on a non-string names the argument's
# real type. A warmup loop exercises the working set-membership and ord fast
# paths first. Deterministic.
def warm(n):
    acc = 0
    s = {1, 2, 3}
    for i in range(n):
        acc += ord("a") + (i % 3 in s)
    return acc


def m(label, fn):
    try:
        fn()
        print(label, "no-error")
    except BaseException as e:
        print(label, type(e).__name__, repr(str(e)), e.args)


def main():
    print("warm", warm(10800000))
    m("remove_int", lambda: {1, 2}.remove(9))
    m("remove_str", lambda: {1, 2}.remove("x"))
    m("remove_tuple", lambda: {(1, 2)}.remove((3, 4)))
    m("ord_int", lambda: ord(65))
    m("ord_none", lambda: ord(None))
    m("ord_list", lambda: ord([1]))
    m("ord_float", lambda: ord(1.5))


main()
