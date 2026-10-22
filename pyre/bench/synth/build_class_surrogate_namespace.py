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


main()
