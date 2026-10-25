# No `max-pypy-ratio`: this fixture compiles no loop -- its jitstats record
# `loops_compiled=0` -- so a pypy ratio compares two interpreters' startup
# rather than any generated code, and reads whatever the host's process
# spawn cost happens to be that run. The jitstats baselines gate it.
# A __prepare__ namespace whose keys include a lone surrogate must be read back
# into the class without assuming the keys are valid UTF-8.


class Meta(type):
    @classmethod
    def __prepare__(mcs, name, bases):
        # a lone-surrogate key sits alongside the ordinary ones
        return {chr(0xD800): 'surrogate', 'marker': 'ok'}


class C(metaclass=Meta):
    body_attr = 42


# compiling.py `build_class` runs the body against the mapping `__prepare__`
# returned: a retained dict sees `a = 1` before `NS.get("a")`.  That mapping
# is also the object `LOAD_LOCALS` stores into the compiler-made
# `__classdict__` cell.
NS = None


class PrepareRetained(type):
    @classmethod
    def __prepare__(mcls, name, bases):
        global NS
        NS = {}
        return NS


class Retained(metaclass=PrepareRetained):
    a = 1
    b = NS.get("a", 0)


def main():
    # the ordinary prepared-namespace key survives the surrogate sibling
    print('marker', C.marker)
    # a normal class-body store still lands
    print('body', C.body_attr)
    # the surrogate-named attribute is preserved and retrievable
    print('surrogate', getattr(C, chr(0xD800)))
    print('retained', Retained.b)

    class ProbeDict(dict):
        def __getitem__(self, key):
            if key in ("__classcell__", "__classdictcell__"):
                raise RuntimeError("probed " + str(key))
            return super().__getitem__(key)

        def __missing__(self, key):
            if key in ("__classcell__", "__classdictcell__"):
                raise RuntimeError("missing " + str(key))
            raise KeyError(key)

    class ProbeMeta(type):
        @classmethod
        def __prepare__(mcs, name, bases):
            return ProbeDict()

        def __new__(mcs, name, bases, ns):
            return "sentinel"

    class Probe(metaclass=ProbeMeta):
        def method(self):
            return super()

    print('probe', Probe)

    class SetNameDesc:
        def __set_name__(self, owner, name):
            print("set_name", owner.probe())

    class SetNameOwner:
        d = SetNameDesc()

        def probe():
            return __class__.__name__

    class NonTypeMeta(type):
        def __new__(mcs, name, bases, ns):
            return ns["method"]

    class NonTypeOwner(metaclass=NonTypeMeta):
        def method(self):
            return __class__

    try:
        NonTypeOwner(None)
        print('non_type_cell', 'filled')
    except NameError as exc:
        print('non_type_cell', type(exc).__name__)

    try:
        class D(object, object):
            __classcell__ = 1
        print('dup_classcell', 'ok')
    except Exception as exc:
        print('dup_classcell', type(exc).__name__)

    class Stash:
        stashed = locals()

        def method(self):
            return __class__

    print('classcell_in_stashed', '__classcell__' in Stash.stashed)

    class CellMutDesc:
        def __set_name__(self, owner, name):
            owner.method.__closure__[0].cell_contents = 123

    try:
        class CellMut:
            d = CellMutDesc()

            def method(self):
                return __class__
        print('set_name_cell', 'ok')
    except Exception as exc:
        print('set_name_cell', type(exc).__name__)

    class CellMutSDesc:
        def __set_name__(self, owner, name):
            owner.method.__closure__[0].cell_contents = "s"

    try:
        class CellMutS:
            d = CellMutSDesc()

            def method(self):
                return __class__
        print('set_name_cell_s', 'ok')
    except Exception as exc:
        print('set_name_cell_s', type(exc).__name__)


main()
