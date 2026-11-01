# No `max-pypy-ratio`: the mutating probes below run once each, so a pypy
# ratio would compare two interpreters' startup rather than generated code.
# The jitstats baselines gate the warmup loop.
class Key:
    def __init__(self, value, owner):
        self.value = value
        self.owner = owner

    def __hash__(self):
        return 9

    def __eq__(self, other):
        container = self.owner[0]
        if container is not None and not self.owner[1]:
            self.owner[1] = True
            marker = ("mutated", self.value)
            if isinstance(container, dict):
                container[marker] = 1
            else:
                container.add(marker)
        return isinstance(other, Key) and self.value == other.value


def run_set(operation):
    owner = [None, False]
    container = set(Key(i, owner) for i in range(8))
    owner[0] = container
    probe = Key(1000, owner)
    if operation == "add":
        container.add(probe)
    elif operation == "contains":
        probe in container
    else:
        container.discard(probe)
    return len(container)


def run_dict(operation):
    owner = [None, False]
    container = {Key(i, owner): 1 for i in range(8)}
    owner[0] = container
    probe = Key(1000, owner)
    if operation == "setitem":
        container[probe] = 1
    elif operation == "getitem":
        container.get(probe)
    elif operation == "delitem":
        try:
            del container[probe]
        except KeyError:
            pass
    else:
        container.pop(probe, None)
    return len(container)


# Trace getitem and setitem of a colliding user key.  owner[1] is already
# True so `__eq__` does not insert the mutation marker; the one-shot probes
# below still mutate once.
owner = [None, True]
container = {Key(i, owner): 1 for i in range(8)}
owner[0] = container
probe = Key(1000, owner)
for _ in range(2000):
    container.get(probe)
    container[probe] = 1

for operation in ("add", "contains", "discard"):
    print("set", operation, run_set(operation))
for operation in ("setitem", "getitem", "delitem", "pop"):
    print("dict", operation, run_dict(operation))
