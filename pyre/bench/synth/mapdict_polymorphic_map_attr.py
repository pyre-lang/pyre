# pyre-check: max-pypy-ratio=14
# One LOAD_ATTR site alternates between maps whose insertion order assigns
# different storage slots to `q`; promoting the map must guard the selected
# storage coordinate, while a class-layout guard alone cannot (mapdict.py _get_mapdict_map).
class A:
    def __init__(self):
        self.p = 1
        self.q = 2


class B:
    def __init__(self):
        self.q = 20
        self.p = 10


def run(n):
    objects = [A(), B()]
    total = 0
    i = 0
    while i < n:
        total += objects[i % 2].q
        i += 1
    return total


print(run(1000000))
