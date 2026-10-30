# CPython-suite gap: test_pickle.test_evil_class_mutating_dict can collect
# inside a dict store's hash probe, but does not pin a nursery that forces
# a moving collection between the callback-free probe and the reentrant retry.
# parity-tests reason: __hash__ collects during the unpickler's dict stores
# inside the dict store's callback-free probe; the reentrant probe must
# reload the dict, key and value.
# parity-env: PYPY_GC_NURSERY=262144
# parity-env: MAJIT_GC_NURSERY_POISON=1

import pickle, random, gc
random.seed(0)
getrandbits = random.getrandbits
ENABLED = False
LOADING = False
class Bad:
    def __eq__(self, other):
        return ENABLED
    def __hash__(self):
        if LOADING:
            gc.collect()
        return 42
    def __reduce__(self):
        if getrandbits(6) == 0:
            collection.clear()
        return (Bad, ())
n = 0
for proto in range(pickle.HIGHEST_PROTOCOL + 1):
    for _ in range(20):
        ENABLED = False
        collection = {Bad(): Bad() for _ in range(20)}
        for bad in collection:
            bad.bad = bad
            bad.collection = collection
        ENABLED = True
        try:
            data = pickle.dumps(collection, proto)
            LOADING = True
            try:
                pickle.loads(data)
            finally:
                LOADING = False
        except RuntimeError as e:
            assert "changed size during iteration" in str(e)
            n += 1
print("ok", n > 0)
print("OK")
