# `function.py:673` / `:720` `StaticMethod._immutable_fields_ =
# ['w_function?']` and the `ClassMethod` twin.  The `?` is what lets a tracer
# bake the wrapped callable and equally what registers the invalidation an
# assignment to the slot owes, so re-initialising an installed wrapper revokes
# every loop that folded it.
#
# Re-initialising a wrapper is the case no other pin reaches: it changes no
# type's version tag and leaves the descriptor at the same address, so a fold
# that pinned the DESCRIPTOR still holds while the callable inside it has
# already been replaced.
#
# Each rebind happens INSIDE its loop: a read after the loop is interpreted and
# would not consult what the trace baked.  The wrapped bodies are residual-free
# so the folds stand rather than aborting.
N = 2400000
SWITCH = N // 2


def first_hook(name):
    return 1


def second_hook(name):
    return 2


def first_class_hook(cls, name):
    return 1


def second_class_hook(cls, name):
    return 2


def first_scaled(cls, i):
    return 1


def second_scaled(cls, i):
    return 2


class StaticHook:
    __getattr__ = staticmethod(first_hook)


class ClassHook:
    __getattr__ = classmethod(first_class_hook)


class Scaled:
    scaled = classmethod(first_scaled)


def rebind_staticmethod_hook():
    # The `__getattr__`-hook fold's `staticmethod` arm: the descriptor binds
    # nothing, so the name is the only argument.
    obj = StaticHook()
    descr = StaticHook.__dict__['__getattr__']
    total = 0
    i = 0
    while i < N:
        total += obj.missing
        if i == SWITCH:
            descr.__init__(second_hook)
        i += 1
    # SWITCH+1 reads of 1, then N-SWITCH-1 reads of 2.
    print('staticmethod hook', total)


def rebind_classmethod_hook():
    # The same fold's `classmethod` arm — a distinct code path, since the
    # descriptor leads the positionals with the class it binds.
    obj = ClassHook()
    descr = ClassHook.__dict__['__getattr__']
    total = 0
    i = 0
    while i < N:
        total += obj.missing
        if i == SWITCH:
            descr.__init__(second_class_hook)
        i += 1
    print('classmethod hook', total)


def rebind_classmethod_on_type():
    # The `LOAD_METHOD` classmethod fold for a type receiver, which bakes
    # `__func__` under the class's version tag alone.  Re-initialising the
    # descriptor bumps no version tag, so this is the case that reads a stale
    # callable without the `w_function?` pin.
    descr = Scaled.__dict__['scaled']
    total = 0
    i = 0
    while i < N:
        total += Scaled.scaled(i)
        if i == SWITCH:
            descr.__init__(second_scaled)
        i += 1
    print('classmethod on type', total)


rebind_staticmethod_hook()
rebind_classmethod_hook()
rebind_classmethod_on_type()
