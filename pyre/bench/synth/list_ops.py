# Merged synth parity smoke suite: independent feature-level hot loops, each
# kept verbatim from its former standalone file with module-level names prefixed
# by the source name. Bug-repro / resume / kept-stack tests are NOT merged (they
# stay isolated so a miscompile is not diluted). check.py runs every *.py here.

try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

# Every independent loop needs one compiled trace; larger counts only repeat
# the same list operation after its result and jitstats shape are fixed.

# ── list_index_update ──
list_index_update__N = 10000

def list_index_update__main():
    xs = [0] * 32
    i = 0
    while i < list_index_update__N:
        j = i & 31
        xs[j] = xs[j] + i
        xs[j + 7 & 31] = xs[j + 7 & 31] - j
        i = i + 1
    print(sum(xs))
list_index_update__main()

# ── list_slicing ──
list_slicing__N = 4000

def list_slicing__main():
    xs = [0, 1, 2, 3, 4, 5, 6, 7]
    i = 0
    while i < list_slicing__N:
        xs[2:5] = [i & 255, i + 1 & 255, i + 2 & 255]
        ys = xs[1:6]
        xs[0] = ys[2]
        i = i + 1
    print(sum(xs))
list_slicing__main()

# ── list_append_pop ──
list_append_pop__N = 10000

def list_append_pop__main():
    xs = []
    i = 0
    acc = 0
    while i < list_append_pop__N:
        xs.append(i)
        if len(xs) > 32:
            acc = acc + xs.pop(0)
        if i & 7 == 0:
            xs.append(i + 3)
            acc = acc - xs.pop()
        i = i + 1
    print(acc + len(xs) + sum(xs))
list_append_pop__main()

# ── list_obj_append_pop ──
list_obj_append_pop__N = 10000

def list_obj_append_pop__main():
    xs = []
    i = 0
    acc = 0
    s = 'x'
    t = 'y'
    while i < list_obj_append_pop__N:
        xs.append(s)
        if len(xs) > 32:
            xs.pop(0)
        if i & 7 == 0:
            xs.append(t)
            xs.pop()
        i = i + 1
    print(len(xs) + acc)
list_obj_append_pop__main()

# ── tuple_unpacking ──
tuple_unpacking__N = 10000

def tuple_unpacking__pair(i):
    return (i, i + 1)

def tuple_unpacking__main():
    i = 0
    acc = 0
    while i < tuple_unpacking__N:
        (a, b) = tuple_unpacking__pair(i)
        (c, d, e) = (b, a + b, a - b)
        acc = acc + c + d + e
        i = i + 1
    print(acc)
tuple_unpacking__main()

# ── build_list_large ──
build_list_large__N = 10000

def build_list_large__main():
    i = 0
    acc = 0
    while i < build_list_large__N:
        t = [i, i + 1, i + 2, i + 3, i + 4, i + 5]
        acc = acc + t[0] + t[2] + t[5]
        i = i + 1
    print(acc)
build_list_large__main()

# ── build_tuple_large ──
build_tuple_large__N = 10000

def build_tuple_large__main():
    i = 0
    acc = 0
    while i < build_tuple_large__N:
        t = (i, i + 1, i + 2, i + 3, i + 4, i + 5)
        acc = acc + t[0] + t[2] + t[5]
        i = i + 1
    print(acc)
build_tuple_large__main()
