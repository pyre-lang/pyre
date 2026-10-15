# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=hot_object,hot_int,hot_float,hot_ascii,hot_nested,hot_break_resume,hot_append_during,hot_truncate_during,hot_strategy_change,hot_exhausted_retained,hot_setstate_sentinel,hot_subclass,hot_empty
# pyre-check: trace-shape=hot_int:absent=CallMayForceR
# `for x in <list>` drives a `list_iterator`. The step is
# `W_FastListIterObject.descr_next` (`list_iter_descr_next`): it re-reads the
# live length, so an append or a truncation under the cursor is observed, and
# exhaustion clears the sequence. The legs below pin that recorded body: the
# four storage strategies, a list mutated underneath a live cursor in both
# directions, a strategy promotion mid-iteration, the exhaustion edge, a
# partially consumed cursor resumed after `break`, a list subclass (its
# `__getitem__` must not run in the loop), and the `__setstate__` negative
# cursor.
try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

N = 20000
K = 2000


def hot_object():
    data = [object(), None, True, (1, 2), "x"]
    seen = 0
    for _ in range(N):
        for item in data:
            if item is None:
                seen += 1
            else:
                seen += 2
    return seen


def hot_int():
    data = [3, 1, 4, 1, 5, 9, 2, 6]
    total = 0
    for _ in range(N):
        for value in data:
            total += value
    return total


def hot_float():
    data = [0.5, 1.25, -2.0, 8.0]
    total = 0.0
    for _ in range(N):
        for value in data:
            total += value
    return total


def hot_ascii():
    data = ["alpha", "beta", "gamma", "delta"]
    total = 0
    for _ in range(N):
        for text in data:
            total += len(text)
    return total


def hot_nested():
    outer = [[1, 2], [3], [4, 5, 6]]
    total = 0
    for _ in range(K):
        for inner in outer:
            for value in inner:
                total += value
    return total


def hot_break_resume():
    # A `break` leaves the cursor mid-list; resuming the SAME iterator must
    # continue where it stopped, not restart.
    total = 0
    for _ in range(K):
        data = [10, 20, 30, 40]
        it = iter(data)
        for value in it:
            total += value
            if value == 20:
                break
        for value in it:
            total += value * 100
    return total


def hot_append_during():
    # `descr_next` re-reads the live length, so an element appended while the
    # cursor is inside the list is yielded.
    total = 0
    for _ in range(K):
        data = [1, 2, 3]
        for value in data:
            total += value
            if value == 3 and len(data) < 5:
                data.append(4)
                data.append(5)
    return total


def hot_truncate_during():
    # A removed tail ends the loop early.
    total = 0
    for _ in range(K):
        data = [1, 2, 3, 4, 5, 6]
        for value in data:
            total += value
            if value == 2:
                del data[3:]
    return total


def hot_strategy_change():
    # An int-strategy list promoted to object storage under a live cursor: the
    # strategy guard must send the next read back to the generic path.
    total = 0
    for _ in range(K):
        data = [1, 2, 3, 4]
        parts = []
        for value in data:
            parts.append(repr(value))
            if value == 1:
                data.append("tail")
        total += len(parts)
    return total


def hot_exhausted_retained():
    # Exhaustion clears the sequence: the retained iterator stays exhausted and
    # reports a zero length hint.
    total = 0
    for _ in range(K):
        it = iter([7, 8])
        for value in it:
            total += value
        total += len(list(it))
        total += it.__length_hint__()
        try:
            next(it)
        except StopIteration:
            total += 1
    return total


def hot_setstate_sentinel():
    # `__setstate__(-1)` is the exhausted sentinel; a later in-range state
    # revives the cursor.  What the negative cursor itself yields is an oracle
    # divergence (cpython stays exhausted, pypy clamps to the front), so the
    # leg walks it without counting and scores only the revive.
    total = 0
    for _ in range(K):
        data = [1, 2, 3]
        it = iter(data)
        it.__setstate__(-1)
        for _value in it:
            pass
        it = iter(data)
        it.__setstate__(2)
        total += sum(it)
    return total


class Doubling(list):
    def __getitem__(self, index):
        return list.__getitem__(self, index) * 2


def hot_subclass():
    # `list.__iter__` yields a `list_iterator` over the subclass instance and
    # reads the storage directly, so the override does NOT reach the loop --
    # only the explicit `data[0]` doubles.  The direct read must still refuse
    # an instance whose `w_class` is not the canonical one.
    data = Doubling([1, 2, 3])
    total = 0
    for _ in range(K):
        for value in data:
            total += value
        total += data[0]
    return total


def hot_empty():
    total = 0
    for _ in range(K):
        for _value in []:
            total += 1
        total += 1
    return total


def main():
    checks = [
        ("hot_object", hot_object(), 9 * N),
        ("hot_int", hot_int(), 31 * N),
        ("hot_float", hot_float(), 7.75 * N),
        ("hot_ascii", hot_ascii(), 19 * N),
        ("hot_nested", hot_nested(), 21 * K),
        ("hot_break_resume", hot_break_resume(), 7030 * K),
        ("hot_append_during", hot_append_during(), 15 * K),
        ("hot_truncate_during", hot_truncate_during(), 6 * K),
        ("hot_strategy_change", hot_strategy_change(), 5 * K),
        ("hot_exhausted_retained", hot_exhausted_retained(), 16 * K),
        ("hot_setstate_sentinel", hot_setstate_sentinel(), 3 * K),
        ("hot_subclass", hot_subclass(), 8 * K),
        ("hot_empty", hot_empty(), K),
    ]
    failed = 0
    for name, got, want in checks:
        if got != want:
            print("FAIL", name, "got", got, "want", want)
            failed = 1
    if failed:
        return 1
    print("PASS for_iter list fold")
    return 0


import sys

sys.exit(main())
