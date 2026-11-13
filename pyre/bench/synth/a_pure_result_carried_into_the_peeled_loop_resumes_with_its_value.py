# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=lev
# The peeled loop reuses `distance + 2` from the previous iteration: the
# preamble's pure result is carried through the label as an imported short box,
# and the next iteration's `substitute = distance + cost` folds onto it.  When
# a guard in that iteration fails, the boxed `substitute` local is a virtual
# whose int field is that carried box, and the resume data must number it.
#
# The carried box was minted without being registered as the host of its
# position, so every positional lookup minted a different box: `register_box`
# and `_gettagged` keyed two objects and the field was written as `UNASSIGNED`.
# The blackhole then read the field from a hole as 0 (a wrong distance, so the
# early bail was missed), or indexed the deadframe with -8192 and panicked.
#
# Where tracing starts inside the call decides which guard fails first, so the
# check recompiles `lev` from source under every threshold in a range.  Each
# round gets a fresh code object and therefore a fresh loop.  The source runs
# in a real module namespace: a plain dict as globals takes a different
# LOAD_GLOBAL path and records a loop that never carries the box.
import types
try:
    from pypyjit import set_param
except ImportError:
    def set_param(*args, **kwargs):
        pass

SOURCE = '''
import sys
_MOVE_COST = 2
_CASE_COST = 1

def _substitution_cost(ch_a, ch_b):
    if ch_a == ch_b:
        return 0
    if ch_a.lower() == ch_b.lower():
        return _CASE_COST
    return _MOVE_COST

def lev(a, b, max_cost):
    if len(b) < len(a):
        a, b = b, a
    if (len(b) - len(a)) * _MOVE_COST > max_cost:
        return max_cost + 1
    row = list(range(_MOVE_COST, _MOVE_COST * (len(a) + 1), _MOVE_COST))
    result = 0
    for bindex in range(len(b)):
        bchar = b[bindex]
        distance = result = bindex * _MOVE_COST
        minimum = sys.maxsize
        for index in range(len(a)):
            substitute = distance + _substitution_cost(bchar, a[index])
            distance = row[index]
            insert_delete = min(result, distance) + _MOVE_COST
            result = min(insert_delete, substitute)
            row[index] = result
            if result < minimum:
                minimum = result
        if minimum > max_cost:
            return max_cost + 1
    return result
'''

CASES = (
    # (a, b, max_cost, expected)
    ('callbacks', 'DEBUG_SAVEALL', 8, 9),
    ('callbacks', 'DEBUG_SAVEALL', 1000, 25),
    ('callbacks', 'DEBUG_LEAK', 7, 8),
)


def main():
    failures = []
    for threshold in range(1, 61):
        set_param(threshold=threshold)
        for a, b, max_cost, want in CASES:
            module = types.ModuleType('lev_round')
            exec(compile(SOURCE, 'lev_round.py', 'exec'), module.__dict__)
            got = module.lev(a, b, max_cost)
            if got != want:
                failures.append('threshold=%d lev(%r, %r, %d) = %r, expected %r'
                                % (threshold, a, b, max_cost, got, want))
    set_param('default')
    if failures:
        for line in failures[:10]:
            print('FAIL', line)
        return 1
    print('PASS a pure result carried into the peeled loop resumes with its value')
    return 0


import sys

sys.exit(main())
