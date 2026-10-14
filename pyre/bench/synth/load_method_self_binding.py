# pyre-check: spec-folds=load_method_attr
# pyre-check: trace-shape=entry-bridge:helper:max=GetfieldGcR:2
# LOAD_METHOD's fast-method path decides the descriptor and the receiver
# together (`callmethod.py LOAD_METHOD` pushes `w_descr` then `w_obj`).  pyre
# reaches the two halves through separate jitcode residuals -- `load_attr_fn`
# and `load_method_self_fn` -- and the walker folds them independently, so the
# two verdicts can disagree: the descriptor half bakes the function as a
# constant while the receiver half answers PY_NULL, and the CALL then runs one
# argument short.
#
# `receiver_from_fresh_class` is the witness: the method call sits in a callee
# frame whose receiver is a parameter, and the class is rebuilt every
# iteration, so the callee is first traced on its own (not inlined) long after
# the caller loop compiled.  That standalone trace is where the two folds were
# reached in the order that made them disagree.  A static class, or the same
# call at module level, does not reproduce it.
#
# The remaining legs pin the shapes the receiver half must keep answering
# correctly around that fix: a name shadowed by an instance attribute (no
# binding), a bound method reached through `getattr` (no binding), and a
# classmethod (binds the class, not the instance).
try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

N = 4000


def _name(self):
    return "n"


def receiver_from_fresh_class(n):
    def helper(o):
        return o._name()

    def make(i):
        return type("C%d" % i, (object,), {"_name": _name})

    total = 0
    for i in range(n):
        total += len(helper(make(0)()))
    return total


class Shadowed:
    def m(self):
        return "class"


def instance_attr_shadows(n):
    obj = Shadowed()
    obj.m = lambda: "instance"
    total = 0
    for _ in range(n):
        total += len(obj.m())
    return total


def bound_method_through_getattr(n):
    obj = Shadowed()
    total = 0
    for _ in range(n):
        m = obj.m
        total += len(m())
    return total


class WithClassmethod:
    @classmethod
    def cm(cls):
        return cls.__name__


def classmethod_binds_the_class(n):
    total = 0
    for _ in range(n):
        total += len(WithClassmethod.cm())
    return total


print(receiver_from_fresh_class(N))
print(instance_attr_shadows(N))
print(bound_method_through_getattr(N))
print(classmethod_binds_the_class(N))
