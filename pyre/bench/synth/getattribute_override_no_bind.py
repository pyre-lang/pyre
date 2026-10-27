# pyre-check: max-pypy-ratio=710
# Two regressions in one hot loop:
#
# 1. Binding: a non-default `__getattribute__` must suppress LOAD_METHOD
#    self-binding.  The override returns the raw function object
#    (`type(self).__dict__[name]`), never a bound method, so `c.f(i)` calls
#    `f(i)` (a single positional, no implicit self).  `compute_load_method_bound`
#    (shared by the interpreter `load_method` and the blackhole
#    `bh_load_method_self_fn`) must not infer the binding from MRO shape
#    alone: that prepends self and makes `len(args)` 2 instead of 1.
#
# 2. GC: `type.__dict__` caches its canonical `W_DictObject` in the type
#    namespace's off-GC `DictStorage.mirror_target`.  The moving collector did
#    not forward that field, so once the loop's allocations triggered a minor
#    collection the cache dangled and the next `__dict__` access returned a
#    wild pointer (SIGSEGV in `getitem`).  The loop count is large enough to
#    force collections through the blackhole-resumed `__getattribute__`.
class C:
    def f(*args):
        return len(args)

    def __getattribute__(self, name):
        return type(self).__dict__[name]


def run(c, n):
    total = 0
    i = 0
    while i < n:
        total += c.f(i)
        i += 1
    return total


print(run(C(), 1200000))
