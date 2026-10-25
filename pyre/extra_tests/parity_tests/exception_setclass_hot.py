# CPython-suite gap: exception tests never rebind __class__ inside a hot
# loop and then grade type(), isinstance, except matching, and __reduce__.
# parity-tests reason: _getusercls installs mapdict mixins for exception
# USER, so getclass is map.terminator.w_cls (BaseUserClassMapdict.getclass).
# After setclass the header w_class word is stale; JIT readers must pin
# the map, not that header (walker_guard_exact_w_class).

"""Exception setclass is visible to type, isinstance, except, and reduce."""


try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass


class E(Exception):
    pass


class F(Exception):
    pass


def run(n):
    e = E("hot")
    matched = 0
    for i in range(n):
        e.__class__ = F
        assert type(e) is F
        assert isinstance(e, F)
        try:
            raise e
        except F:
            matched += 1
        reduced = e.__reduce__()
        assert reduced[0] is F
        e.__class__ = E
    return matched


assert run(2500) == 2500

# Constant receiver: the instance is a module-global default argument, so the
# trace sees a ConstPtr. Mapdict class is still `map.terminator.w_cls`
# (`BaseUserClassMapdict.getclass`); `self.map` is not in `_immutable_fields_`,
# so `_get_mapdict_map` promotes even a constant receiver. After the loop
# compiles, setclass must be visible to type, except, and reduce.
E_INST = E("const")


def run_const(n, x=E_INST):
    matched = 0
    for i in range(n):
        if i == 200:
            x.__class__ = F
        expected = F if i >= 200 else E
        assert type(x) is expected
        if i >= 200:
            assert isinstance(x, F)
            try:
                raise x
            except F:
                matched += 1
            assert x.__reduce__()[0] is F
    x.__class__ = E
    return matched


assert run_const(2500) == 2300
print("OK")
