# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=loop_callee,drive
# A generator portal whose for-loop body calls a loop-bearing callee
# (cut into CALL_ASSEMBLER) must keep the FOR_ITER iterator. pyjitpl.py
# `_opimpl_setarrayitem_vable` then `synchronize_virtualizable_at` writes
# the object the metainterp executes (`virtualizable.py write_box_at`).
# A record-only portal walk that stored `pyframe.py popvalue` None onto
# the live interpreter frame left that slot NULL, and interpreter
# FOR_ITER then faulted in `shortcut_fallback_next`.
#
# Shape taken from `unittest.loader._find_tests`: yield inside the for-loop,
# a nested `yield from` after a loop-bearing helper, then a later consume
# over a different iterator type so the compiled FOR_ITER guard fails.

N = 400
INNER = 8


def loop_callee(n):
    s = 0
    i = 0
    while i < n:
        s += i
        i += 1
    return s


def find_tests(paths, depth=0):
    for path in paths:
        n = loop_callee(INNER + depth)
        yield path, n
        if path == "pkg" and depth < 1:
            yield from find_tests(("pkg/a", "pkg/b"), depth + 1)


def drive(n, names):
    acc = 0
    last = None
    for _ in range(n):
        paths = []
        for path, value in find_tests(names):
            acc += value + len(path)
            paths.append(path)
        last = paths
    return acc, last


def expected_paths(names):
    out = []
    for path in names:
        out.append(path)
        if path == "pkg":
            out.extend(("pkg/a", "pkg/b"))
    return list(out)


def triangle(n):
    return n * (n - 1) // 2


tuple_names = tuple("t%d.py" % i for i in range(12)) + ("pkg",) + tuple(
    "z%d.py" % i for i in range(6)
)
list_names = list(tuple_names)
want_paths = expected_paths(tuple_names)
want_acc = 0
for path in want_paths:
    depth = 1 if path.startswith("pkg/") else 0
    want_acc += triangle(INNER + depth) + len(path)

got_acc, got_paths = drive(N, tuple_names)
if got_paths != want_paths:
    raise SystemExit(
        "FAIL generator FOR_ITER lost the iterator: %r != %r" % (got_paths, want_paths)
    )
if got_acc != want_acc * N:
    raise SystemExit(
        "FAIL generator FOR_ITER wrong totals: %r != %r" % (got_acc, want_acc * N)
    )

got_acc, got_paths = drive(8, list_names)
if got_paths != want_paths or got_acc != want_acc * 8:
    raise SystemExit(
        "FAIL generator FOR_ITER lost the iterator after deopt: %r %r"
        % (got_paths, got_acc)
    )
print("PASS generator FOR_ITER kept the iterator", got_acc)
