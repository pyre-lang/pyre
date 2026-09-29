# pyre-check: max-pypy-ratio=3
# pyre-check: skip-cpython
# pyre-check: trace-shape=run:absent=CallMayForceR,absent=CallR
# Zero-argument super inside a FOR_ITER body.  N keeps the pypy reference above
# check.py's timing floor; CPython is skipped because that N is impractical.
#
# `load_super_attr_descent` still declines at unpublished `w_method_new`.
# Publishing it in a throwaway dynasm binary exposed an unclosed RootScope
# bracket: before `7d57e55a2dc` the shadow stack grew each iteration.  Best of
# three runs per N, in ns/iteration (`fold` disables only the descent):
#
#                  descent           fold          `PYRE_NO_JIT=1`
#         N     before  after     before  after     before  after
#   250,000      3131    265        353    182       1516    860
#   500,000      3445    169        170     91       1315    774
# 1,000,000      4372    119         87     46       1243    730
# 2,000,000      8009     95         45     23       1181    709
#
# The fixed descent is linear (about 0.049s + 71ns/iteration), but remains 4.2x
# slower than the fold at N=2,000,000, so `w_method_new` stays unpublished.
#
try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

N = 250000000


class Base:
    def val(self):
        return 1


class Child(Base):
    def run(self, n):
        acc = 0
        for _ in range(n):
            acc = acc + super().val()
        return acc


print(Child().run(N))
