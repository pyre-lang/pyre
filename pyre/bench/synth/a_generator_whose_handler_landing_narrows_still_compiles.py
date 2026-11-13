# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=main
# Every protected range of `prev_next_window` unwinds to the generator's one
# `lasti` handler, which the codewriter coalesces into a single landing block.
# The early `except StopIteration: return` reaches that landing before the
# keyword call binds `window`, the loop after it with `window` bound, so the
# landing is generalized: the landing that was already emitted is retired, and
# it forwards to a fresh landing that later edges keep widening in place.  When
# a later edge left a local unbound, the widened state dropped that slot, while
# the forwarding link still carried the older argument list.  Flattening the
# jitcode then indexed past the landing's inputargs and the process aborted.
from collections import deque


def prev_next_window(iterable):
    iterator = iter(iterable)
    try:
        first = next(iterator)
    except StopIteration:
        return
    window = deque((None, first), maxlen=3)
    try:
        for x in iterator:
            window.append(x)
            yield tuple(window)
    finally:
        window.append(None)
        yield tuple(window)


def main():
    total = 0
    for _ in range(300):
        for prev, cur, nxt in prev_next_window('abcdefghij' * 5):
            total += cur is not None
    assert total == 300 * 50, total
    assert list(prev_next_window('')) == []
    assert list(prev_next_window('ab')) == [(None, 'a', 'b'), ('a', 'b', None)]
    print('PASS a generator whose handler landing narrows still compiles')


main()
