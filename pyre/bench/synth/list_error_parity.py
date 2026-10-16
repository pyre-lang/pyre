# pyre-check: max-pypy-ratio=24
# list error-message parity where 3.14 and PyPy AGREE (check.py oracle is
# PyPy).  list.index's missing-value message DIVERGES — 3.14 says
# "list.index(x): x not in list" (gh-100242 dropped the repr) while PyPy keeps
# the older "%R is not in list" form — so the index message itself is not
# asserted here; only the raise PATH is exercised (hot_index_miss, comparing
# the ValueError count) plus list.pop's empty/out-of-range IndexError
# messages, which agree.  list.remove is likewise omitted (3.14
# "list.remove(x): x not in list" vs PyPy "%R is not in list").
def cap(fn):
    try:
        return ("ok", fn())
    except (ValueError, IndexError, TypeError) as e:
        return (type(e).__name__, str(e))


def drive():
    out = []
    # list.index honouring start: 1 reappears at index 2, so this succeeds.
    out.append(("index_found_range", cap(lambda: [1, 2, 1].index(1, 2))))
    out.append(("pop_empty", cap(lambda: [].pop())))
    out.append(("pop_oob", cap(lambda: [1, 2].pop(9))))
    # hot index-miss so a compiled trace exercises the raise path; only the
    # ValueError count is compared (the message diverges 3.14 vs PyPy).
    hits = 0
    n = 0
    while n < 100000:
        try:
            [1, 2, 3].index(9)
        except ValueError:
            hits += 1
        n += 1
    out.append(("hot_index_miss", hits))
    return out


for row in drive():
    print(row)
