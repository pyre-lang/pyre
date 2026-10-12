# pyre-check: max-pypy-ratio=3
# pyre-check: skip-cpython
# The loop count is sized so pypy clears `FLOOR_GATE_MIN_BASELINE_S`.  At
# 200000 iterations pypy exec is under `EXEC_TIME_FLOOR_S` and the ratio
# prints with a `~`, so no ceiling is applied.  320000000 iterations land
# pypy near 0.18s.  Local dynasm reads 1.1x with the `name == 'ping'`
# `jit_ll_streq` folded on constant arguments; 3 leaves room for cranelift
# and a slower host.  cpython cannot run this many inside the reference
# timeout.
# A metaclass resolves `Cls.name()` before the class's own MRO does:
# `type.__getattribute__` lets a metatype DATA descriptor win outright, and a
# metatype `__getattribute__` override produces the value itself. Either way the
# call must use what the metaclass returned, not rebind the class onto it.


class MetaProp(type):
    @property
    def where(cls):
        return lambda: 'meta-prop'


class ByProp(metaclass=MetaProp):
    @classmethod
    def where(cls):
        return 'own-classmethod'


class MetaGetattr(type):
    def __getattribute__(cls, name):
        if name == 'ping':
            return lambda: 'meta-getattr'
        return type.__getattribute__(cls, name)


class ByGetattr(metaclass=MetaGetattr):
    @classmethod
    def ping(cls):
        return 'own-classmethod'


class Plain:
    @classmethod
    def tag(cls):
        return cls.__name__


def main():
    prop = getattr_ = plain = None
    for _ in range(320000000):
        prop = ByProp.where()
        getattr_ = ByGetattr.ping()
        # an ordinary class still binds its classmethod's cls
        plain = Plain.tag()
    print('prop', prop)
    print('getattr', getattr_)
    print('plain', plain)


main()
