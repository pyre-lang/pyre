# `StdObjSpace.getattr` (`objspace/std/objspace.py`) wraps the
# `__getattr__` fallback call in `enrich_attribute_error` (`error.py:725-738`),
# which fills an AttributeError's `name` and `obj` in with the attribute that
# was looked up and the object it was looked up on -- and only when the
# exception carries neither yet, so a hook that sets its own pair keeps it.
#
# `try_walker_inline_getattr_hook` walks the hook body in place of the
# `load_attr_fn` residual that used to reach that enrichment, so the inline
# owes the pair itself.  The binding fixture next to this one covers the
# returning path; this one covers the raising path, where the enrichment is the
# only observable difference between the residual and the inline.
#
# A single access answers from the interpreter, so an assertion that runs once
# proves nothing about the compiled trace.  Each loop below runs long enough
# for the hook to be inlined and re-reads the pair on every iteration.
N = 60000


class Miss:
    def __getattr__(self, name):
        raise AttributeError(name)


class Preset:
    def __getattr__(self, name):
        err = AttributeError(name)
        err.name = 'from_hook'
        err.obj = 'from_hook_obj'
        raise err


class Mixed:
    def __getattr__(self, name):
        if name == 'ok':
            return 'answered'
        raise AttributeError(name)


def main():
    miss = Miss()
    preset = Preset()
    mixed = Mixed()

    hits = 0
    for _ in range(N):
        try:
            miss.absent
        except AttributeError as err:
            if err.name == 'absent' and err.obj is miss:
                hits += 1
    print('miss', hits)

    # `enrich_attribute_error` writes only when both slots are unset, so a hook
    # that filled them in itself keeps its own pair.
    hits = 0
    for _ in range(N):
        try:
            preset.absent
        except AttributeError as err:
            if err.name == 'from_hook' and err.obj == 'from_hook_obj':
                hits += 1
    print('preset', hits)

    # The name is the attribute the failing access actually asked for, not the
    # one the recording iteration happened to see.
    hits = 0
    for i in range(N):
        if i % 2 == 0:
            try:
                miss.even
            except AttributeError as err:
                if err.name == 'even':
                    hits += 1
        else:
            try:
                miss.odd
            except AttributeError as err:
                if err.name == 'odd':
                    hits += 1
    print('alternating', hits)

    # A hook that answers some names and raises on others: the trace records
    # one of the two and reaches the other through a guard failure, so the
    # enrichment has to hold on the recorded arm and on the bridge alike.
    hits = 0
    for i in range(N):
        if i % 2 == 0:
            if mixed.ok == 'answered':
                hits += 1
        else:
            try:
                mixed.bad
            except AttributeError as err:
                if err.name == 'bad' and err.obj is mixed:
                    hits += 1
    print('mixed', hits)

    # The pair survives the exception outliving the handler: reading it after
    # the loop must not answer from a slot the next iteration overwrote.
    saved = None
    for _ in range(N):
        try:
            miss.kept
        except AttributeError as err:
            saved = err
    print('saved', saved.name, saved.obj is miss)


main()
