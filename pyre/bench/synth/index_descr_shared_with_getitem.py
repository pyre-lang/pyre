# A class may bind the same one-argument function to both `__index__` and
# `__getitem__`. Descriptor identity does not identify the calling protocol:
# `seq[c]` is `space.index` (`get_and_call_function(w_impl, w_obj)`, zero extra
# args) and must succeed, while `c[i]` is `getitem` (`get_and_call_function(
# w_descr, w_obj, w_key)`) and must raise TypeError every time.
N = 8000


def one(self):
    return 3


class C:
    __index__ = one
    __getitem__ = one


def main():
    c = C()
    seq = list(range(10))
    hits = 0
    errs = 0
    i = 0
    while i < N:
        hits += seq[c]
        try:
            c[i]
        except TypeError:
            errs += 1
        i += 1
    print(hits, errs)


main()
