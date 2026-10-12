# CPython-suite gap: `test_extcall` and `test_builtin` pass `**` mappings whose
# keys are ordinary parameter names (`x`, `sep`, `start`). Nothing there uses
# the identifier `__pyre_kw__`, so a call path that stores a sentinel under that
# key overwrites a real user keyword and the suite still passes.
#
# parity-tests reason: `Arguments.keyword_names_w` / `keywords_w` keep the name
# out of band (`argument.py`). A user keyword spelled `__pyre_kw__` is a real
# item: `dict(**{K:7})` is `{K: 7}`. Each row pins CPython 3.14.6.
#
# pyre-check: pypy-diverges: pypy3 words round/pow/int/str as "takes no keyword
# arguments" (and names `int.__new__`/`str.__new__`); type.__init_subclass__
# names `object` rather than `C`; sum with two extras is "unexpected keyword"
# rather than "takes at most 2 arguments"; applevel sorted reports `sorted()`
# rather than `sort()`.
K = "__pyre_kw__"


def _catch(thunk):
    try:
        return ("ok", thunk())
    except Exception as err:
        return (type(err).__name__, str(err))


def _check(label, thunk, expected):
    got = _catch(thunk)
    if got == expected:
        return
    raise AssertionError("%s: got %r, expected %r" % (label, got, expected))


def dict_starstar_keeps_the_marker_key():
    _check(
        "dict(**{K:7})",
        lambda: dict(**{K: 7}),
        ("ok", {K: 7}),
    )


def dict_starstar_getitem_reads_the_marker_key():
    def thunk():
        return dict(**{K: 7})[K]

    _check(
        "dict(**{K:7})[K]",
        thunk,
        ("ok", 7),
    )


def sum_rejects_the_marker_key():
    _check(
        "sum([1,2], **{K:0})",
        lambda: sum([1, 2], **{K: 0}),
        (
            "TypeError",
            "sum() got an unexpected keyword argument '%s'" % K,
        ),
    )


def round_rejects_the_marker_key():
    _check(
        "round(1.5, **{K:0})",
        lambda: round(1.5, **{K: 0}),
        (
            "TypeError",
            "round() got an unexpected keyword argument '%s'" % K,
        ),
    )


def pow_rejects_the_marker_key():
    _check(
        "pow(2,3,**{K:5})",
        lambda: pow(2, 3, **{K: 5}),
        (
            "TypeError",
            "pow() got an unexpected keyword argument '%s'" % K,
        ),
    )


def int_rejects_the_marker_key():
    _check(
        'int("5", **{K:1})',
        lambda: int("5", **{K: 1}),
        (
            "TypeError",
            "int() got an unexpected keyword argument '%s'" % K,
        ),
    )


def str_rejects_the_marker_key():
    _check(
        'str(b"x", **{K:1})',
        lambda: str(b"x", **{K: 1}),
        (
            "TypeError",
            "str() got an unexpected keyword argument '%s'" % K,
        ),
    )


def type_init_subclass_names_the_class():
    _check(
        'type("C", (), {}, **{K:1})',
        lambda: type("C", (), {}, **{K: 1}).__name__,
        (
            "TypeError",
            "C.__init_subclass__() takes no keyword arguments",
        ),
    )


def sum_two_extras_are_too_many_arguments():
    _check(
        'sum([1], **{K:0, "start":3})',
        lambda: sum([1], **{K: 0, "start": 3}),
        ("TypeError", "sum() takes at most 2 arguments (3 given)"),
    )


def dict_get_takes_no_keyword_arguments():
    _check(
        "{}.get(**{K:1})",
        lambda: {}.get(**{K: 1}),
        ("TypeError", "dict.get() takes no keyword arguments"),
    )


def list_append_takes_no_keyword_arguments():
    _check(
        "[].append(**{K:1})",
        lambda: [].append(**{K: 1}),
        ("TypeError", "list.append() takes no keyword arguments"),
    )


def sorted_reports_sort_for_an_unknown_keyword():
    _check(
        "sorted([], **{K:1})",
        lambda: sorted([], **{K: 1}),
        (
            "TypeError",
            "sort() got an unexpected keyword argument '%s'" % K,
        ),
    )


def user_function_kwargs_keep_the_marker_key():
    def f(**kw):
        return kw

    _check(
        "def f(**kw) with K",
        lambda: f(**{K: 7}),
        ("ok", {K: 7}),
    )


def print_rejects_the_marker_key():
    _check(
        "print(**{K:7})",
        lambda: print(**{K: 7}),
        (
            "TypeError",
            "print() got an unexpected keyword argument '%s'" % K,
        ),
    )


dict_starstar_keeps_the_marker_key()
dict_starstar_getitem_reads_the_marker_key()
sum_rejects_the_marker_key()
round_rejects_the_marker_key()
pow_rejects_the_marker_key()
int_rejects_the_marker_key()
str_rejects_the_marker_key()
type_init_subclass_names_the_class()
sum_two_extras_are_too_many_arguments()
dict_get_takes_no_keyword_arguments()
list_append_takes_no_keyword_arguments()
sorted_reports_sort_for_an_unknown_keyword()
user_function_kwargs_keep_the_marker_key()
print_rejects_the_marker_key()
print("OK")
