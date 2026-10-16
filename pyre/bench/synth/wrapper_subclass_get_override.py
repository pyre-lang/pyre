# `typeobject.py descr_getattribute` resolves an attribute through
# `type(descr).__get__`, so a `classmethod` / `staticmethod` SUBCLASS that
# overrides `__get__` binds through that override.  A fold that unwraps
# `w_function` in its place calls the wrapped callable directly and never
# reaches the override, which is why every wrapper type test is exact.
#
# One loop per fold that unwraps a wrapper — class receiver, instance
# receiver, and the `__getattr__` hook's two arms — each reading through a
# direct call so the fold is the shape that records, not a lambda's
# megamorphic call site.  Every loop must report the OVERRIDE.
# Sized for the fold, not for throughput: the declining path runs the whole
# descriptor protocol per iteration.
try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

N = 300000


class GetOverridingClassMethod(classmethod):
    def __get__(self, obj, cls=None):
        return overridden


class GetOverridingStaticMethod(staticmethod):
    def __get__(self, obj, cls=None):
        return overridden


def overridden(*args):
    return 'override'


def wrapped_class(cls, x):
    return 'wrapped'


def wrapped_static(x):
    return 'wrapped'


class Attrs:
    cm = GetOverridingClassMethod(wrapped_class)
    sm = GetOverridingStaticMethod(wrapped_static)


class ClassHook:
    __getattr__ = GetOverridingClassMethod(wrapped_class)


class StaticHook:
    __getattr__ = GetOverridingStaticMethod(wrapped_static)


def classmethod_on_type():
    seen = None
    i = 0
    while i < N:
        seen = Attrs.cm(i)
        i += 1
    print('classmethod on type', seen)


def staticmethod_on_type():
    seen = None
    i = 0
    while i < N:
        seen = Attrs.sm(i)
        i += 1
    print('staticmethod on type', seen)


def classmethod_on_instance():
    obj = Attrs()
    seen = None
    i = 0
    while i < N:
        seen = obj.cm(i)
        i += 1
    print('classmethod on instance', seen)


def staticmethod_on_instance():
    obj = Attrs()
    seen = None
    i = 0
    while i < N:
        seen = obj.sm(i)
        i += 1
    print('staticmethod on instance', seen)


def classmethod_getattr_hook():
    obj = ClassHook()
    seen = None
    i = 0
    while i < N:
        seen = obj.missing
        i += 1
    print('classmethod getattr hook', seen)


def staticmethod_getattr_hook():
    obj = StaticHook()
    seen = None
    i = 0
    while i < N:
        seen = obj.missing
        i += 1
    print('staticmethod getattr hook', seen)


classmethod_on_type()
staticmethod_on_type()
classmethod_on_instance()
staticmethod_on_instance()
classmethod_getattr_hook()
staticmethod_getattr_hook()
