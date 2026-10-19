# CPython-suite gap: `test_extcall` unpacks `**` through ordinary dicts and
# mappings, never a dict subclass that overrides `__getitem__` but leaves
# `__iter__` as `dict.__iter__`.
#
# parity-tests reason: `update1` / `_combine_starstarargs_wrapped` compute
# `is_dict` from `findattr(type, "__iter__") is dict.__iter__` and then read
# with `dict_getitem`; a subclass that also overrides `__iter__` uses
# `space.getitem`.  `f(**D)`, `{**D}`, `{'z': 0, **D}`, and `dict.update`
# agree on both oracles; `f(x=1, **D)` is the 3.14 `DICT_MERGE` non-empty
# target (`_dict_merge_loop` value read).
# pyre-check: pypy-diverges: pypy3 3.11 `_dict_merge_loop` uses space.getitem so f(x=1, **D) binds a='OVERRIDE'; python3 3.14 binds a=1.

keys_log = []


class D(dict):
    def __getitem__(self, k):
        return "OVERRIDE"

    def keys(self):
        keys_log.append("keys")
        return dict.keys(self)


class E(dict):
    def __getitem__(self, k):
        return "OVERRIDE"

    def __iter__(self):
        return iter(dict.keys(self))


class M(dict):
    def __missing__(self, k):
        return "missing"


def f(**kw):
    return kw


print(f(**D(a=1)))
print(f(**E(a=1)))
print(f(x=1, **D(a=1)))
print(f(x=1, **E(a=1)))
print({**D(a=1)})
print({**E(a=1)})
print({"z": 0, **D(a=1)})
print({"z": 0, **E(a=1)})
print({"x": 1, **D(a=1)})
print(f(**M(a=1)))
print(keys_log)
d = {}
d.update(D(a=1))
print(d)
e = {}
e.update(E(a=1))
print(e)

hot_star_d = None
hot_star_e = None
hot_kw_d = None
hot_kw_e = None
hot_unp_d = None
hot_unp_e = None
hot_merge_d = None
hot_merge_e = None
hot_upd_d = None
hot_upd_e = None
i = 0
while i < 2000:
    hot_star_d = f(**D(a=1))
    hot_star_e = f(**E(a=1))
    hot_kw_d = f(x=1, **D(a=1))
    hot_kw_e = f(x=1, **E(a=1))
    hot_unp_d = {**D(a=1)}
    hot_unp_e = {**E(a=1)}
    hot_merge_d = {"z": 0, **D(a=1)}
    hot_merge_e = {"z": 0, **E(a=1)}
    hot_upd_d = {}
    hot_upd_d.update(D(a=1))
    hot_upd_e = {}
    hot_upd_e.update(E(a=1))
    i = i + 1
print(hot_star_d)
print(hot_star_e)
print(hot_kw_d)
print(hot_kw_e)
print(hot_unp_d)
print(hot_unp_e)
print(hot_merge_d)
print(hot_merge_e)
print(hot_upd_d)
print(hot_upd_e)
assert hot_star_d == {"a": 1}, hot_star_d
assert hot_star_e == {"a": "OVERRIDE"}, hot_star_e
assert hot_kw_d == {"x": 1, "a": 1}, hot_kw_d
assert hot_kw_e == {"x": 1, "a": "OVERRIDE"}, hot_kw_e
assert hot_unp_d == {"a": 1}, hot_unp_d
assert hot_unp_e == {"a": "OVERRIDE"}, hot_unp_e
assert hot_merge_d == {"z": 0, "a": 1}, hot_merge_d
assert hot_merge_e == {"z": 0, "a": "OVERRIDE"}, hot_merge_e
assert hot_upd_d == {"a": 1}, hot_upd_d
assert hot_upd_e == {"a": "OVERRIDE"}, hot_upd_e
print("OK")
