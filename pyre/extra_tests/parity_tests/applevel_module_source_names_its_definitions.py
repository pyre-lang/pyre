# CPython-suite gap: no suite test asks a module's own helpers what
# `__module__` they report.  `test_atexit`, `test_ordered_dict`,
# `test_hashlib` and `test_contextvars` all exercise what those objects *do*,
# and each name here is reached through its module, so a wrong owner never
# changes an answer any of them checks.  The name only surfaces when something
# asks the object where it came from -- a traceback header, a `repr`, `help`,
# or `pickle` resolving a reference.
#
# The subject is where the answer comes from.  A class reads `__module__` off
# the globals it is defined in, through a name lookup that reaches the
# builtins module when its own namespace has none -- so a namespace with no
# `__name__` makes every class in it claim `builtins`.  A function reads the
# same entry with a plain dict lookup instead, so the same namespace leaves
# it with `None`.  Both are silent: the object is built, and only its account
# of itself is wrong.
#
# parity-tests reason: several of these are written in Python here and in C on
# CPython, and the namespace such a source runs in is the runtime's own
# construction rather than a real module's dict.  That construction is what
# has to bind the name, and nothing else in this suite would notice if it
# stopped: pickling by reference is the one arm that fails loudly, and it
# fails only for a subset.  CPython and PyPy agree on every arm below.
#
# applevel_module_frames_are_hidden:
# nothing in the suite installs a tracer around a builtin whose implementation happens to
# be written in Python on some runtime, or walks the traceback of a failure raised inside
# one, so nothing there notices when such a helper starts reporting a frame. On CPython
# every name below lives in an extension module and has no frame to report, which is
# exactly why the suite cannot express the question. a frame is not an implementation
import pickle
import sys

import atexit
import collections
import contextvars
import _contextvars
import hashlib
import typing
import _io
import operator

def a_nameless_namespace_answers_two_different_ways():
    namespace = {}
    exec('class Cls: pass\ndef fn(): pass\n', namespace)

    # The two halves of what an unnamed namespace costs.  Neither raises, and
    # a runtime that builds its own module sources this way inherits both.
    assert namespace['Cls'].__module__ == 'builtins', namespace['Cls'].__module__
    assert namespace['fn'].__module__ is None, namespace['fn'].__module__


def a_native_modules_own_helpers_name_it():
    # `atexit` is a C module whose callbacks are pure Python here.  Whichever
    # half a name is written in, it belongs to `atexit`.
    for helper in (atexit.register, atexit.unregister, atexit._ncallbacks):
        assert helper.__module__ == 'atexit', (helper, helper.__module__)


def an_accelerated_class_names_the_module_it_is_reached_through():
    # Each of these is a class an accelerator module defines and a public
    # module re-exports, so the owner it reports is the one a reader can
    # import it from -- not always the one whose source defines it.
    expected = [
        (collections.OrderedDict, 'collections'),
        (collections.defaultdict, 'collections'),
        (hashlib.blake2b, '_blake2'),
        (contextvars.ContextVar, '_contextvars'),
        (contextvars.Context, '_contextvars'),
        (typing.TypeVar, 'typing'),
        (_io.IncrementalNewlineDecoder, '_io'),
    ]
    for cls, owner in expected:
        assert cls.__module__ == owner, (cls, cls.__module__, owner)


def the_reported_owner_is_the_one_pickle_resolves():
    # The arm that makes a wrong owner an error rather than a cosmetic slip.
    # A class and a function both pickle as a reference -- module name plus
    # qualified name -- so the owner has to be a module the loader can import
    # and find the object in.  `None` and `builtins` are each unusable, and
    # the failure lands on the caller pickling an ordinary object.
    for obj in (
        atexit.register,
        atexit.unregister,
        collections.OrderedDict,
        collections.defaultdict,
        hashlib.blake2b,
        contextvars.ContextVar,
        contextvars.Context,
        typing.TypeVar,
        _io.IncrementalNewlineDecoder,
    ):
        restored = pickle.loads(pickle.dumps(obj))
        assert restored is obj, (obj, restored)
        module = sys.modules[obj.__module__]
        assert getattr(module, obj.__name__) is obj, (obj, module)

    assert contextvars.Context is _contextvars.Context
    assert repr(contextvars.Context) == "<class '_contextvars.Context'>"
    assert repr(contextvars.copy_context()).startswith(
        '<_contextvars.Context object at 0x'
    )


a_nameless_namespace_answers_two_different_ways()
a_native_modules_own_helpers_name_it()
an_accelerated_class_names_the_module_it_is_reached_through()
the_reported_owner_is_the_one_pickle_resolves()

def traced(call):
    seen = []

    def record(frame, event, arg):
        if event == 'call':
            seen.append(frame.f_code.co_name)
        return None

    call()  # warm every lazy import and cache outside the trace
    sys.settrace(record)
    try:
        call()
    finally:
        sys.settrace(None)
    return seen


def the_object_reduce_helpers_are_not_frames():
    class Simple:
        pass

    obj = Simple()
    # The protocol-2 reduction walks the instance dict and the slot names, and
    # where that walk is written in Python it is still not the program's.
    assert traced(lambda: obj.__reduce_ex__(2)) == ['<lambda>'], traced(
        lambda: obj.__reduce_ex__(2)
    )

    # Bundled app-level source resolves its own sys module without consulting
    # a program's blocked import entry.
    class Slotted:
        __slots__ = ('value',)

    slotted = Slotted()
    slotted.value = 1
    saved = sys.modules['sys']
    sys.modules['sys'] = None
    try:
        reduction = slotted.__reduce_ex__(2)
    finally:
        sys.modules['sys'] = saved
    assert reduction[1] == (Slotted,)
    assert reduction[2] == (None, {'value': 1})


def the_defaultdict_factory_is_not_a_frame():
    d = collections.defaultdict(int)
    # A miss fills the key it was asked for, and `traced` runs its callable once
    # outside the trace, so a fixed key would leave the traced call an ordinary
    # hit that never reaches the factory -- and the arm would pass however
    # visible that frame became.  Hand each call a key nothing has filled.
    keys = iter(('warm', 'traced'))
    seen = traced(lambda: d[next(keys)])
    assert seen == ['<lambda>'], seen
    assert d['traced'] == 0, d


def the_operator_getters_are_not_frames():
    get = operator.attrgetter('real')
    assert traced(lambda: get(1)) == ['<lambda>'], traced(lambda: get(1))
    assert traced(lambda: operator.itemgetter(0)([7])) == ['<lambda>']


def the_atexit_registry_is_not_a_frame():
    def callback():
        pass

    def register_and_drop():
        atexit.register(callback)
        atexit.unregister(callback)

    assert traced(register_and_drop) == ['register_and_drop'], traced(register_and_drop)


def a_failure_inside_aiter_records_no_frame():
    class NotAnAsyncIterator:
        def __aiter__(self):
            return self

    try:
        aiter(NotAnAsyncIterator())
    except TypeError:
        traceback = sys.exc_info()[2]
    else:
        raise AssertionError('aiter accepted a non-async-iterator')

    names = []
    while traceback is not None:
        names.append(traceback.tb_frame.f_code.co_name)
        traceback = traceback.tb_next
    assert names == ['a_failure_inside_aiter_records_no_frame'], names


def the_caller_a_type_parameter_records_is_this_frame():
    # The arm that fails in the other direction.  A type-parameter object takes
    # its `__module__` from the frame that is running when it is built, and
    # every frame between this one and that read is one of the hidden ones --
    # so the answer has to be this module, not whoever called it.
    assert typing.TypeVar('T').__module__ == '__main__', typing.TypeVar('T').__module__
    assert typing.ParamSpec('P').__module__ == '__main__'
    assert typing.TypeVarTuple('Ts').__module__ == '__main__'

    def built_one_frame_deeper():
        return typing.TypeVar('U').__module__

    assert built_one_frame_deeper() == '__main__', built_one_frame_deeper()


the_object_reduce_helpers_are_not_frames()
the_defaultdict_factory_is_not_a_frame()
the_operator_getters_are_not_frames()
the_atexit_registry_is_not_a_frame()
a_failure_inside_aiter_records_no_frame()
the_caller_a_type_parameter_records_is_this_frame()
print('OK')
