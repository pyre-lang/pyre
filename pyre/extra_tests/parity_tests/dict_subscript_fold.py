# CPython-suite gap: dict subscription tests do not run through pyre's trace fold.
# parity-tests reason: this guards JIT folding, guards, and deoptimization.
#
# dict_unicode_lookup_fold:
# dict tests do not trace Unicode-strategy changes at one site. this guards pyre's JIT
# strategy fold and side exit.

N = 3000


def read_tag(d, k):
    n = 0
    for _ in range(N):
        n += d[k]
    return n


def read_last(d, k):
    out = None
    for _ in range(N):
        out = d[k]
    return out


def read_mixed(d, k):
    out = None
    for _ in range(N):
        out = d[k]
    return out


d = {"tag": 1}
assert read_tag(d, "tag") == N
d["tag"] = 2
assert read_tag(d, "tag") == 2 * N

gone = {"tag": 3}
assert read_tag(gone, "tag") == 3 * N
del gone["tag"]
try:
    read_last(gone, "tag")
except KeyError as exc:
    assert exc.args == ("tag",)
else:
    raise AssertionError("deleted key still readable")


class MissingDict(dict):
    def __missing__(self, key):
        return 41


missing = MissingDict()
assert read_last(missing, "tag") == 41

mixed = {1: "int", "1": "str"}
assert read_mixed(mixed, 1) == "int"
assert read_mixed(mixed, "1") == "str"

computed_key = "".join(["ta", "g"])
assert computed_key == "tag"
assert read_last({"tag": "computed"}, computed_key) == "computed"


class MyStr(str):
    def __hash__(self):
        return str.__hash__(self)

    def __eq__(self, other):
        return str.__eq__(self, other)


incoming_sub = MyStr("tag")
assert read_last({"tag": "incoming"}, incoming_sub) == "incoming"

stored_sub = MyStr("tag")
stored_sub_dict = {stored_sub: "stored"}
assert next(iter(stored_sub_dict)) is stored_sub
assert read_last(stored_sub_dict, "tag") == "stored"

surrogate = "\ud800"
assert read_last({surrogate: "surrogate"}, surrogate) == "surrogate"

devolved = {"tag": 5}
assert read_tag(devolved, "tag") == 5 * N
devolved[1] = 99
assert read_tag(devolved, "tag") == 5 * N


class CountingKey:
    hash_count = 0
    eq_count = 0

    def __init__(self, value):
        self.value = value

    def __hash__(self):
        type(self).hash_count += 1
        return 123

    def __eq__(self, other):
        type(self).eq_count += 1
        return isinstance(other, CountingKey) and self.value == other.value


stored = CountingKey("tag")
probe = CountingKey("tag")
counted = {stored: 7}
CountingKey.hash_count = 0
CountingKey.eq_count = 0
assert read_tag(counted, probe) == 7 * N
assert CountingKey.hash_count == N
assert CountingKey.eq_count == N

CountingKey.hash_count = 0
CountingKey.eq_count = 0
assert read_tag(counted, stored) == 7 * N
assert CountingKey.hash_count == N
assert CountingKey.eq_count == 0


def read_varying_int_keys(d):
    total = 0
    for i in range(N):
        total += d[i]
    return total


def write_new_keys_and_read_first(d):
    total = 0
    for i in range(N):
        d[i] = i * 3
        total += d[0]
    return total


varying = {}
for i in range(N):
    varying[i] = i + 5
assert read_varying_int_keys(varying) == (N * (N - 1)) // 2 + 5 * N

mutating = {0: 11}
assert write_new_keys_and_read_first(mutating) == 0
assert mutating[N - 1] == (N - 1) * 3

def dict_unicode_lookup_fold():
    N = 400


    def check(got, want, label):
        assert got == want, "%s: %r != %r" % (label, got, want)


    # ── the plain shape: one loop-invariant dict, one loop-invariant key ──
    def repeat_same_key(n):
        d = {"tag": 3, "other": 5}
        total = 0
        i = 0
        while i < n:
            total = total + d["tag"]
            i = i + 1
        return total


    # ── keys that are EQUAL but not the same object.  The identity-keyed
    # `dict.get` fold could never reach this; the hash+equality probe must ──
    def fresh_key_each_iteration(n):
        d = {"tag": 7}
        total = 0
        i = 0
        while i < n:
            k = ("tagX")[:-1]
            total = total + d[k]
            i = i + 1
        return total


    # ── the value under one key is rewritten every iteration.  A value-only
    # overwrite keeps the entry index, so the fold may keep it — but the VALUE read
    # must stay live, never folded to the recorded one ──
    def value_overwritten(n):
        d = {"acc": 0}
        i = 0
        while i < n:
            d["acc"] = d["acc"] + i
            i = i + 1
        return d["acc"]


    # ── the KEY SET changes under the loop.  Each insert can move the entry a
    # retained index points at, so a stale index would read the wrong value ──
    def key_set_grows(n):
        d = {"probe": 1}
        total = 0
        i = 0
        while i < n:
            d["k%d" % i] = i
            total = total + d["probe"]
            i = i + 1
        return total


    # ── a delete compacts the table, renumbering every entry after the hole ──
    def key_set_shrinks(n):
        d = {}
        i = 0
        while i < 64:
            d["k%d" % i] = i
            i = i + 1
        d["probe"] = 99
        total = 0
        i = 0
        while i < n:
            victim = "k%d" % (i % 64)
            if victim in d:
                del d[victim]
            total = total + d["probe"]
            i = i + 1
        return total


    # ── two dicts alternating at one call site: a receiver mix-up cross-checks ──
    def two_receivers(n):
        a = {"tag": 11}
        b = {"tag": 22}
        total = 0
        i = 0
        while i < n:
            total = total + (a if i % 2 == 0 else b)["tag"]
            i = i + 1
        return total


    # ── the dict leaves the Unicode strategy mid-loop.  Storing a non-str key
    # promotes it to the Object strategy, so the strategy guard must side-exit and
    # the same call site must keep answering correctly afterwards ──
    def strategy_switches(n):
        d = {"tag": 4}
        total = 0
        i = 0
        while i < n:
            if i == n // 2:
                d[17] = 100
            total = total + d["tag"]
            i = i + 1
        return total + d[17]


    # ── a miss must raise KeyError, not answer a neighbouring entry ──
    def miss_raises(n):
        d = {"present": 1}
        hits = 0
        misses = 0
        i = 0
        while i < n:
            try:
                hits = hits + d["present" if i % 3 else "absent"]
            except KeyError:
                misses = misses + 1
            i = i + 1
        return hits * 1000 + misses


    # ── a dict SUBCLASS defines `__missing__`, so the fold must decline for it and
    # the generic path must reach the override ──
    class WithMissing(dict):
        def __missing__(self, key):
            return len(key)


    def subclass_missing(n):
        d = WithMissing()
        d["here"] = 2
        total = 0
        i = 0
        while i < n:
            total = total + d["here"] + d["absent"]
            i = i + 1
        return total


    # ── a str SUBCLASS key may override `__hash__`/`__eq__`, so it must not take
    # the exact-str probe ──
    class Shouty(str):
        def __hash__(self):
            return hash(str(self).lower())

        def __eq__(self, other):
            return str(self).lower() == str(other).lower()


    def subclass_key(n):
        d = {"tag": 6}
        plain = 0
        shouty = 0
        i = 0
        while i < n:
            plain = plain + d["tag"]
            try:
                shouty = shouty + d[Shouty("TAG")]
            except KeyError:
                shouty = shouty - 1
            i = i + 1
        return plain * 100000 + shouty


    def run():
        check(repeat_same_key(N), 3 * N, "repeat_same_key")
        check(fresh_key_each_iteration(N), 7 * N, "fresh_key_each_iteration")
        check(value_overwritten(N), sum(range(N)), "value_overwritten")
        check(key_set_grows(N), N, "key_set_grows")
        check(key_set_shrinks(N), 99 * N, "key_set_shrinks")
        check(two_receivers(N), (11 + 22) * (N // 2), "two_receivers")
        check(strategy_switches(N), 4 * N + 100, "strategy_switches")
        check(miss_raises(N), expected_miss(), "miss_raises")
        check(subclass_missing(N), (2 + 6) * N, "subclass_missing")
        check(subclass_key(N), expected_subclass_key(), "subclass_key")


    # `miss_raises` / `subclass_key` checksums are written out from the same
    # per-iteration rule rather than reusing the function, so a wrong answer cannot
    # be cancelled by a matching wrong expectation.
    def expected_miss():
        hits = 0
        misses = 0
        for i in range(N):
            if i % 3:
                hits = hits + 1
            else:
                misses = misses + 1
        return hits * 1000 + misses


    def expected_subclass_key():
        # `Shouty("TAG")` hashes as "tag" and compares equal to it, so every
        # iteration finds the entry through the subclass's own protocol.
        return (6 * N) * 100000 + 6 * N


    run()

    # A second pass on already-warm code: every loop above has been traced by now,
    # so this run executes the compiled form rather than building it.
    run()
dict_unicode_lookup_fold()
print("OK")
