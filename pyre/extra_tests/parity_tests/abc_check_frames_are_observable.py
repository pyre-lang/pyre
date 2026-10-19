# CPython-suite gap: `test_abc` never installs a tracer around an ABC check,
# and no suite test walks the traceback of a failure raised behind one, so
# nothing there notices if the forwarding frames stop existing.
# `ABCMeta.__instancecheck__` and `__subclasscheck__` are ordinary Python
# functions in `abc.py`, so each ABC check pushes a frame a program can see.
# `_abc_instancecheck` is app-level source installed with `hidden_applevel`,
# so it is not a frame.  Answering the check from `isinstance` while skipping
# the forwarding method would drop the frame this test pins, and that frame
# is a third of what an ABC check costs.
#
# parity-tests reason: that frame is not an implementation detail.  A tracer
# installed with `sys.settrace` receives a `call` event for it, and a failure
# raised behind the check reports it in the traceback, so removing it changes
# what a debugger shows and what an error says.  Both are pinned here because
# neither is visible from a timing or a wrong answer -- a shortcut that skipped
# the frame would pass every other test in this suite.
#
# CPython, PyPy and pyre agree on both observations today, so this is a parity
# fact rather than a house rule.
#
# abcmeta_type_call_inline:
# `test_abc` never instantiates an ABCMeta-built class in a loop hot enough to compile,
# and no suite test installs `__call__` on a metaclass after such a loop has run.
# `typeobject.py:_type_call` runs unless the metatype supplies a `__call__` of its own. An
# `ABCMeta` subclass supplies `__instancecheck__`, `__subclasscheck__` and `register` and
# leaves `__call__` alone, so a class built with one instantiates through the same path a
import sys
import abc


def instancecheck_is_a_traced_call():
    class Shape(metaclass=abc.ABCMeta):
        pass

    class Circle(Shape):
        pass

    circle = Circle()
    # Warm the positive cache, so what the tracer sees is the *fast* path --
    # the one a shortcut would take over.
    assert isinstance(circle, Shape) is True

    seen = []

    def record(frame, event, arg):
        if event == 'call':
            seen.append(frame.f_code.co_name)
        return None

    sys.settrace(record)
    try:
        isinstance(circle, Shape)
    finally:
        sys.settrace(None)

    assert '__instancecheck__' in seen, seen


def a_failure_behind_the_check_names_the_frames():
    class Refusing(metaclass=abc.ABCMeta):
        @classmethod
        def __subclasshook__(cls, subclass):
            raise ValueError('refused')

    class Probe:
        pass

    try:
        isinstance(Probe(), Refusing)
    except ValueError:
        traceback = sys.exc_info()[2]
    else:
        raise AssertionError('the hook did not raise')

    names = []
    while traceback is not None:
        names.append(traceback.tb_frame.f_code.co_name)
        traceback = traceback.tb_next
    # The check reaches the hook through both forwarding methods, and each one
    # is a frame the report walks.
    assert names.count('__instancecheck__') == 1, names
    assert names.count('__subclasscheck__') == 1, names
    assert names[-1] == '__subclasshook__', names


def the_cache_membership_body_is_not_a_frame():
    class Shape(metaclass=abc.ABCMeta):
        pass

    class Circle(Shape):
        pass

    circle = Circle()
    # A populated positive cache, so the membership test really runs.
    assert isinstance(circle, Shape) is True

    seen = []

    def record(frame, event, arg):
        if event == 'call':
            seen.append(frame.f_code.co_name)
        return None

    sys.settrace(record)
    try:
        isinstance(circle, Shape)
    finally:
        sys.settrace(None)

    # The opposite pin to the two above, and the reason they are not a blanket
    # rule about everything an ABC check touches.  The cache is a weak-set
    # whose membership test is written in Python on one of these runtimes, yet
    # no runtime reports a frame for it: where it is a built-in it has none,
    # and where it is the module's own helper source the frame is hidden --
    # `ApplevelClass.hidden_applevel` marks that code, `frame.hide()` drops it
    # from the trace hook and `record_application_traceback` refuses to record
    # it.  So answering membership without a frame is what every runtime
    # already does, and only the `abc.py` frames above are observable.
    assert seen.count('__contains__') == 0, seen

    class Raising(type):
        def __hash__(cls):
            raise ValueError('refused')

    class Opaque(metaclass=Raising):
        pass

    try:
        isinstance(Opaque(), Shape)
    except ValueError:
        traceback = sys.exc_info()[2]
    else:
        raise AssertionError('the hash did not raise')

    names = []
    while traceback is not None:
        names.append(traceback.tb_frame.f_code.co_name)
        traceback = traceback.tb_next
    assert names.count('__contains__') == 0, names
    assert names[-1] == '__hash__', names


def the_declined_membership_body_is_not_a_frame():
    class Shape(metaclass=abc.ABCMeta):
        pass

    class Circle(Shape):
        pass

    circle = Circle()
    assert isinstance(circle, Shape) is True

    collection = type(Shape._abc_cache)
    namespace = collection.__contains__.__globals__

    class Missing(Exception):
        pass

    seen = []

    def record(frame, event, arg):
        if event == 'call':
            seen.append(frame.f_code.co_name)
        return None

    # A shadowing module entry is one of the things that makes the native
    # answer decline, so this is the run where the app-level body really
    # executes.  It is still not a frame: the body being reached is a fact
    # about which implementation answers, and the frame's absence is a fact
    # about whose code it is.
    namespace['TypeError'] = Missing
    sys.settrace(record)
    try:
        isinstance(circle, Shape)
    finally:
        sys.settrace(None)
        del namespace['TypeError']

    assert seen.count('__contains__') == 0, seen


instancecheck_is_a_traced_call()
a_failure_behind_the_check_names_the_frames()
the_cache_membership_body_is_not_a_frame()
# The last arm names a collection this implementation spells as a class
# attribute; where it is not spellable there is nothing to shadow and the arm
# does not apply.
if hasattr(abc.ABCMeta('_Probe', (), {}), '_abc_cache'):
    the_declined_membership_body_is_not_a_frame()

def abcmeta_type_call_inline():
    import abc

    N = 40000
    SWITCH = N // 2


    class Meta(abc.ABCMeta):
        pass


    class Point(metaclass=Meta):
        def __init__(self, x):
            self.x = x


    class Fixed:
        x = 7


    def inlines():
        # The plain shape: default `__new__`, an `__init__` the walk can enter, and
        # a metaclass that overrides neither.
        total = 0
        i = 0
        while i < N:
            total += Point(i).x
            i += 1
        assert total == N * (N - 1) // 2, 'wrong sum: %r' % (total,)


    def metaclass_gains_call():
        total = 0
        i = 0
        while i < N:
            total += Point(i).x
            if i == SWITCH:
                Meta.__call__ = lambda cls, x: Fixed()
            i += 1
        # `i == SWITCH` is assigned before the rebind, so iterations 0..SWITCH read
        # their own index and the rest read `Fixed.x`.
        expected = SWITCH * (SWITCH + 1) // 2 + (N - SWITCH - 1) * 7
        assert total == expected, 'baked a stale metaclass __call__: %r != %r' % (
            total,
            expected,
        )


    inlines()
    metaclass_gains_call()
abcmeta_type_call_inline()
print('OK')
