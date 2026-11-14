# `get_and_call_function`: a __getattribute__ defined as a classmethod or
# staticmethod is bound through __get__ before it is called, so it receives the
# arguments the descriptor protocol gives it -- the class, or nothing at all,
# in place of the receiver.  Only an exact `Function` takes the
# `funccall(w_obj, w_name)` shortcut that leads with the receiver.
#
# The hook inlines against the version-tag and map pins that make the
# descriptor lookup constant, instead of costing one opaque `load_attr_fn`
# residual per access holding a fresh interpreter frame.  The twin fixture for
# the miss-side hook is `getattr_hook_binding`.


class ClassmethodGetattribute:
    @classmethod
    def __getattribute__(cls, name):
        return 'cm:%s:%s' % (cls.__name__, name)


class StaticmethodGetattribute:
    @staticmethod
    def __getattribute__(name):
        return 'sm:%s' % name


class PlainGetattribute:
    def __getattribute__(self, name):
        return 'plain:%s' % name


class Shadowed:
    def __getattribute__(self, name):
        return 'hook'


class Both:
    def __getattribute__(self, name):
        if name == 'via_getattr':
            raise AttributeError(name)
        return 'attribute:' + name

    def __getattr__(self, name):
        return 'fallback:' + name


try:
    import pypyjit

    pypyjit.set_param("threshold=20,function_threshold=20")
except ImportError:
    pass

N = 333334


def main():
    # The map-pin loop below stores `later` mid-loop, moving its receiver
    # to the map `Shadowed`'s terminator caches for that attribute.  `warm`
    # takes the transition here, ahead of the allocating loops, so their
    # minor collections have made that map old by the time the store runs:
    # the GUARD_VALUE that fails after the store keys its jitcounter on the
    # map's address (`compile.py` `ResumeGuardDescr.must_compile`
    # `cast_ptr_to_int(refval)`), and a map promoted inside the failure
    # burst would split the count.
    warm = Shadowed()
    warm.later = 'instance'

    # classmethod hook receives the class as its first bound argument
    print('classmethod', ClassmethodGetattribute().nope)
    # staticmethod hook receives only the name
    print('staticmethod', StaticmethodGetattribute().nope)
    # plain function hook stays bound to the instance
    print('plain', PlainGetattribute().nope)

    cm = ClassmethodGetattribute()
    sm = StaticmethodGetattribute()
    plain = PlainGetattribute()
    total = 0
    i = 0
    while i < N:
        if cm.nope == 'cm:ClassmethodGetattribute:nope':
            total += 1
        if sm.nope == 'sm:nope':
            total += 1
        if plain.nope == 'plain:nope':
            total += 1
        i += 1
    print('loop', total)

    # The fold pins the type version, so rebinding the slot mid-loop must
    # switch away from the recorded hook on the very next iteration.
    class Replaced:
        def __getattribute__(self, name):
            return 'first'

    obj = Replaced()
    seen = None
    for i in range(30000):
        seen = obj.later
        if i == 15000:
            Replaced.__getattribute__ = lambda self, name: 'second'
    assert seen == 'second'
    print('version', seen)

    # The fold also pins the receiver map, which the hook's answer does not
    # depend on: a custom slot owns every name whether or not the instance
    # carries one.  A store mid-loop invalidates that pin and must resume on
    # the same answer rather than the stored value.
    obj = Shadowed()
    seen = None
    for i in range(30000):
        seen = obj.later
        if i == 15000:
            obj.later = 'instance'
    assert seen == 'hook'
    print('map', seen, object.__getattribute__(obj, 'later'))

    # A type that also defines __getattr__ declines the fold:
    # `_handle_getattribute` runs the slot, catches its AttributeError and
    # calls the fallback, and the inline entry carries only one of the two
    # frames.
    both = Both()
    hits = 0
    for _ in range(30000):
        if both.plain == 'attribute:plain':
            hits += 1
        if both.via_getattr == 'fallback:via_getattr':
            hits += 1
    print('fallback', hits)


main()
