# CPython-suite gap: no suite test inlines a Python caller around an effectful operator dunder.
# parity-tests reason: a post-return operator-tail guard or abort under an
# already-inlined caller must keep the whole frame chain, and a collection
# inside the dunder must not leave the caller image stale.

"""Nested inlined caller around effectful operator dunders.

A hot loop calls a Python function ``f`` so ``f`` is inlined. ``f`` then
runs ``format``, ``len`` and ``bool`` on objects whose dunders are Python
and have effects. After warmup the dunder result changes so the
post-return guard fails and resumes in the operator tail still under
``f``; an invalid result raises in that tail and is caught outside ``f``.
Each dunder allocates, so a minor collection can run inside the body.
Effects run once per call.
"""

try:
    import pypyjit
except ImportError:
    pypyjit = None

if pypyjit is not None:
    pypyjit.set_param("threshold=1,function_threshold=1")

N = 4000
SWITCH = N // 2


def churn():
    return [[i, i + 1] for i in range(64)]


class Fmt:
    calls = 0

    def __init__(self):
        self.out = "s"

    def __format__(self, spec):
        Fmt.calls += 1
        churn()
        return self.out


class LenObj:
    calls = 0

    def __init__(self):
        self.n = 1

    def __len__(self):
        LenObj.calls += 1
        churn()
        return self.n


class TruthObj:
    calls = 0

    def __init__(self):
        self.out = True

    def __bool__(self):
        TruthObj.calls += 1
        churn()
        return self.out


def f(fmt, ln, tr, spec):
    s = format(fmt, spec)
    n = len(ln)
    b = bool(tr)
    return s, n, b


def run_phase(kind):
    Fmt.calls = 0
    LenObj.calls = 0
    TruthObj.calls = 0
    fmt = Fmt()
    ln = LenObj()
    tr = TruthObj()
    raised = 0
    last = None
    for i in range(N):
        if i == SWITCH:
            if kind == "fmt":
                fmt.out = 42
            elif kind == "len":
                ln.n = -1
            else:
                tr.out = 1
        try:
            last = f(fmt, ln, tr, "s")
        except TypeError:
            raised += 1
        except ValueError:
            raised += 1
    return (
        Fmt.calls,
        LenObj.calls,
        TruthObj.calls,
        raised,
        last,
    )


fmt_calls, fmt_len_calls, fmt_truth_calls, fmt_raised, fmt_last = run_phase("fmt")
len_fmt_calls, len_calls, len_truth_calls, len_raised, len_last = run_phase("len")
truth_fmt_calls, truth_len_calls, truth_calls, truth_raised, truth_last = run_phase(
    "truth"
)

print(fmt_calls, fmt_len_calls, fmt_truth_calls, fmt_raised, fmt_last)
print(len_fmt_calls, len_calls, len_truth_calls, len_raised, len_last)
print(truth_fmt_calls, truth_len_calls, truth_calls, truth_raised, truth_last)
print("OK")
