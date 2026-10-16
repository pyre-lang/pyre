# pyre-check: max-pypy-ratio=108
# #171/PR248 §3a: generic sequence-protocol iteration must be LAZY.
# An object with `__getitem__` but no `__iter__` iterates through a sequence
# cursor (iterobject.py W_SeqIterObject.descr_next: `space.getitem` per step,
# ending on IndexError), NOT by materialising the whole sequence up front.
# Two properties a materialising path gets wrong:
#   * `__len__` must NOT bound the walk -- the cursor advances until
#     `__getitem__` raises IndexError, even when `__len__` under-reports.
#   * items are fetched one-at-a-time as the loop body runs, so a side effect
#     in `__getitem__` interleaves with the body instead of running in a batch.
# A WHILE loop drives a hot FOR_ITER over the cursor so the trace compiles.


class Seq:
    def __len__(self):
        # Under-reports the real length; a `__len__`-bounded walk would stop
        # after one item, a lazy walk runs to the IndexError at index 5.
        return 1

    def __getitem__(self, i):
        if i >= 5:
            raise IndexError
        return i


def main():
    total = 0
    order = 0
    n = 0
    while n < 40000:
        step = 0
        for x in Seq():
            # Fold the visit order in so a batch-then-iterate path (which would
            # run every __getitem__ before the first body) diverges.
            total += x
            order += (step + 1) * x
            step += 1
        n += 1
    return total, order


t, o = main()
print(t, o)
