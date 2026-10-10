# pyre-check: max-pypy-ratio=11
# FOR_ITER over zip of two tuples calls `zip_next` as a residual call until
# a guard inside a descended helper can resume in the helper frame
# (`resume.py` `rebuild_from_resumedata`); the outer `zip()` constructor is
# also residual.

try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

N = 400000
T1 = tuple(range(1, 65))
T2 = tuple(range(10, 650, 10))


def hot(n):
    s = 0
    i = 0
    while i < n:
        for a, b in zip(T1, T2):
            s += a * b
        i += 1
    return s


print(hot(N))
