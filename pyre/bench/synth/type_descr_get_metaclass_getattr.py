# No `max-pypy-ratio`: this fixture compiles no loop -- its jitstats record
# `loops_compiled=0` -- so a pypy ratio compares two interpreters' startup
# rather than any generated code, and reads whatever the host's process
# spawn cost happens to be that run. The jitstats baselines gate it.
# descroperation.py _handle_getattribute — when a descriptor reached through a type attribute
# lookup raises AttributeError from its __get__, the metaclass __getattr__ gets
# the final say (the whole getattribute slot is wrapped in try/except and an
# AttributeError consults __getattr__).
# The type getattribute slot is wrapped so even the hardcoded
# __abstractmethods__ AttributeError (raised when the slot is unset) consults
# the metaclass __getattr__ before propagating.


class Meta(type):
    def __getattr__(cls, name):
        return 'meta:' + name


class Raises:
    # A descriptor in the class's own MRO whose __get__(None, C) raises.
    def __get__(self, obj, objtype=None):
        raise AttributeError('descriptor refused')


class C(metaclass=Meta):
    attr = Raises()


def main():
    # C.attr finds the descriptor, calls Raises().__get__(None, C) which raises
    # AttributeError, so the lookup falls back to Meta.__getattr__('attr').
    print('descr_get_raise', C.attr)
    # a plain missing name takes the terminal metaclass __getattr__ path
    print('missing', C.missing)


main()


class abstractmethods_Meta(type):
    def __getattr__(cls, name):
        return 'meta_hook:' + name


class abstractmethods_C(metaclass=abstractmethods_Meta):
    pass


def abstractmethods_metaclass_getattr_main():
    # __abstractmethods__ is unset on C, so the metaclass __getattr__ answers
    print('abstractmethods', abstractmethods_C.__abstractmethods__)
    # a normal missing name takes the same fallback
    print('missing', abstractmethods_C.whatever)


abstractmethods_metaclass_getattr_main()
