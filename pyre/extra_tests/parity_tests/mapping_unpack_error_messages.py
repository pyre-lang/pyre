# CPython-suite gap: `test_unpack_ex` pins these `{**x}` errors but its row
# stays FAIL for parser messages, so nothing gates them; it never covers
# `f(**x)` on a raising `keys`, nor `dict.update` / `dict()` on `BrokenKeys`.
#
# parity-tests reason: `_DICT_UPDATE` and `_PyEval_FormatKwargsError` rewrite
# an `AttributeError` to a mapping TypeError only when `keys` is absent
# (`PyObject_HasAttrWithError`). `PyMapping_Keys` / `method_output_as_list`
# replace a TypeError from GetIter of the `keys()` result with
# `%T.keys() must return an iterable, not %T`.
class D1:
    def keys(self):
        raise AttributeError("some error")


class I:
    def __iter__(self):
        return self

    def __next__(self):
        raise AttributeError("some error")


class D2:
    def keys(self):
        return I()


class D3:
    def keys(self):
        return ["key"]

    def __getitem__(self, k):
        raise AttributeError("some error")


class BrokenKeys:
    def keys(self):
        return 1


class OnlyKeys:
    def keys(self):
        return ["key"]


class TupleKeys:
    def keys(self):
        return ("k",)

    def __getitem__(self, k):
        return 2


class GenKeys:
    def keys(self):
        yield "k"

    def __getitem__(self, k):
        return 2


def f(**k):
    return k


def show(kind, name, thunk):
    try:
        result = thunk()
    except Exception as e:
        print(kind, name, type(e).__name__, str(e))
    else:
        print(kind, name, "ok", result)


def run(name, o):
    show("{**o}", name, lambda o=o: {**o})
    show("{'a': 1, **o}", name, lambda o=o: {"a": 1, **o})
    show("f(**o)", name, lambda o=o: f(**o))
    show("f(x=1, **o)", name, lambda o=o: f(x=1, **o))
    show("{}.update(o)", name, lambda o=o: {}.update(o))
    show("dict(o)", name, lambda o=o: dict(o))


run("D1", D1())
run("D2", D2())
run("D3", D3())
run("BrokenKeys", BrokenKeys())
run("OnlyKeys", OnlyKeys())
run("int", int())
run("list", [])
show("{**o}", "TupleKeys", lambda: {**TupleKeys()})
show("f(**o)", "TupleKeys", lambda: f(**TupleKeys()))
show("{**o}", "GenKeys", lambda: {**GenKeys()})
show("f(**o)", "GenKeys", lambda: f(**GenKeys()))
print("OK")
