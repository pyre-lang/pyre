# CPython-suite gap: test_types/test_descr/test_class cover most-derived
# metaclass for class statements, not type.__new__ winner-differs name
# validation, inherited type.__new__ skip, or keyword forwarding.
# parity-tests reason: _create_new_type must compute the winning metaclass
# and, when the winner's __new__ is not type.__new__, delegate the original
# arguments including keywords before name validation and namespace copy.
# space.text_w(w_name) is a livevar across allocate_instance; a borrowed
# utf8 view of a young name is not.
# parity-env: PYPY_GC_NURSERY=4096
# parity-env: MAJIT_GC_NURSERY_POISON=1

"""type.__new__ metaclass winner: order, type.__new__ identity, kwargs."""


class MetaCustom(type):
    def __new__(mcls, name, bases, ns, extra=None, **kw):
        ns = dict(ns)
        ns["seen_name"] = name
        ns["seen_extra"] = extra
        return type.__new__(mcls, "Fixed", bases, ns)


class BaseCustom(metaclass=MetaCustom):
    pass


class MetaInh(type):
    pass


class BaseInh(metaclass=MetaInh):
    pass


# a. Custom __new__ on a differing winner runs before name validation.
t = type.__new__(type, "bad\x00name", (BaseCustom,), {})
assert t.__name__ == "Fixed", t.__name__
assert t.seen_name == "bad\x00name", t.seen_name
assert type(t) is MetaCustom

# b. Inherited type.__new__ is not delegated; invalid names still fail here.
try:
    type.__new__(type, "bad\x00name", (BaseInh,), {})
except ValueError as e:
    assert "type name must not contain null characters" in str(e), e
else:
    raise AssertionError("inherited type.__new__ must still validate the name")

t = type.__new__(type, "TB", (BaseInh,), {})
assert t.__name__ == "TB"
assert type(t) is MetaInh

# c. Keywords reach the custom __new__, not only __init_subclass__.
t = type.__new__(type, "CK", (BaseCustom,), {}, extra=7)
assert t.__name__ == "Fixed"
assert t.seen_extra == 7
assert type(t) is MetaCustom

# Young names must survive a collecting allocate_instance. A borrowed
# utf8 view of the unicode payload reads poison after the nursery fills.
junk = []
for i in range(400):
    name = "N%d" % i
    t = type(name, (), {})
    assert t.__name__ == name, (t.__name__, name)
    t = type.__new__(type, name, (BaseInh,), {})
    assert t.__name__ == name, (t.__name__, name)
    assert type(t) is MetaInh

    class MetaLoop(type):
        def __new__(mcls, name, bases, ns):
            return type.__new__(mcls, name, bases, ns)

    class T(metaclass=MetaLoop):
        pass

    assert T.__name__ == "T", T.__name__
    junk.append([object() for _ in range(24)])
    if len(junk) > 8:
        junk.pop(0)

print("OK")
