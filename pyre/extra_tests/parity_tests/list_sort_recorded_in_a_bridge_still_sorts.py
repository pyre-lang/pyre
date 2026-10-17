# CPython-suite gap: `test_heapq`'s `test_naive_nbest` / `test_naive_nworst`
# call `heap.sort()` right after a hot loop, but they run once per class on
# random data, so the shape below is reached only on some runs.
#
# parity-tests reason: `lst.sort()` with no `key=` / `reverse=` reaches
# `sort_list_in_place_obj` with both keyword slots holding the gateway's NULL
# holes.  A trace recording that call must still execute it, as `pyjitpl.py`
# `do_residual_call` executes every residual call it records.  Here the
# recording bridge (the loop exit of `t`) then aborts inside a nested call, and
# the blackhole resumes forward from the walk's state; a call that was recorded
# but not executed leaves the list unsorted.
import io


def inner():
    s = io.StringIO()
    s.write("x")
    return s.getvalue()


def outer():
    return inner()


def t(data):
    heap = []
    for x in data:
        heap.append(x)
        if len(heap) > 5:
            heap.pop()
    heap.sort()
    outer()
    return heap


for k in range(300):
    h = t([(j * 37 + k) % 11 for j in range(40)])
    assert h == sorted(h), (k, h)

print("OK")
