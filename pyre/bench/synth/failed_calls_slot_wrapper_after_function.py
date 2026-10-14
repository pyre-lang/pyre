# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=<module>
# Residual TypeError from a user function, then from a slot wrapper
# (`[].__len__`) on the same loop.  `arity_slot` looks inside `format!`,
# which is `ll_int2dec` (rstr `STR`) plus `ll_strconcat`.  A wrap residual
# (`jit_int_str`) concatenated as rstr reads `w_class` as length.
#
# `test_call.test_frames_are_popped_after_failed_calls` drives each
# callable 1000 times so the second / third class compiles a bridge off
# the first loop.
def f():
    pass


class C:
    def m(self):
        pass


callables = [f, C.m, [].__len__]
for c in callables:
    for _ in range(1000):
        try:
            c(None)
        except TypeError:
            pass
print("PASS")
