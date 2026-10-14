# CPython-suite gap: `test_extcall` covers the shapes a `**` unpack accepts and
# the errors a bad mapping raises, but every key it passes is an exact `str`, so
# nothing there notices an implementation that rebuilds the key.  `test_unicode`
# and `test_userstring` exercise `str` subclasses as values and as dict keys,
# never as the name half of a `**` mapping.  `test_extcall` and `test_builtin`
# pass `**` mappings whose keys are ordinary parameter names (`x`, `sep`,
# `start`); they do not pin an unknown name through builtins that reject
# keywords, constructors that name the subclass, or applevel `sorted`/`sort`.
#
# parity-tests reason: `Arguments._collect_keyword_args` forwards the key the
# mapping supplied -- `space.setitem(w_kwds, w_key, keywords_w[i])` with the
# object out of `keyword_names_w` -- so a subclass instance reaches the callee's
# `**kwargs` dict unchanged, keeps its own `__repr__`, and compares by the
# subclass's rules.  An implementation that carries keyword names as plain text
# and re-wraps them loses that identity, and the loss is invisible to a timing
# run and to every arm that only reads the name.  The error message an unknown
# keyword produces reads the same object, so it is pinned here too.
# `keyword_names_w` / `keywords_w` keep the name out of band (`argument.py`).
# Each unknown-name row pins CPython 3.14.6.
# pyre-check: pypy-diverges: pypy3 words round/pow/int/str as "takes no keyword arguments" (and names `int.__new__`/`str.__new__`); type.__init_subclass__ names `object` rather than `C`; sum with two extras is "unexpected keyword" rather than "takes at most 2 arguments"; applevel sorted reports `sorted()` rather than `sort()`.
class S(str):
    def __repr__(self):
        return "S(%s)" % str.__repr__(self)


def a_star_star_key_reaches_kwargs_unchanged():
    def f(**kw):
        return kw

    kw = f(**{S("a"): 1})
    assert list(map(type, kw)) == [S], list(map(type, kw))
    assert list(map(repr, kw)) == ["S('a')"], list(map(repr, kw))
    assert kw == {"a": 1}, kw


def a_matched_name_is_bound_and_the_rest_keep_their_keys():
    def g(a, **kw):
        return a, kw

    a, kw = g(**{S("a"): 1, S("b"): 2})
    assert a == 1, a
    assert list(map(type, kw)) == [S], list(map(type, kw))


def the_keys_survive_a_sort():
    def f(**kw):
        return kw

    kw = f(**{S("z"): 1, S("a"): 2})
    assert list(map(repr, sorted(kw))) == ["S('a')", "S('z')"], sorted(kw)


def a_literal_keyword_is_an_exact_str():
    def f(**kw):
        return kw

    kw = f(a=1)
    assert list(map(type, kw)) == [str], list(map(type, kw))


def an_unknown_keyword_names_the_key_it_was_given():
    def h(a):
        return a

    try:
        h(**{S("b"): 1})
    except TypeError as exc:
        assert "'b'" in str(exc), str(exc)
    else:
        raise AssertionError("expected TypeError")


def a_duplicate_keyword_is_still_a_type_error():
    def f(a, **kw):
        return a, kw

    try:
        f(1, **{S("a"): 2})
    except TypeError as exc:
        assert "'a'" in str(exc), str(exc)
    else:
        raise AssertionError("expected TypeError")


def every_dispatch_arm_keeps_the_key():
    # The call surface reaches a callee through several arms, and each one
    # rebuilds its argument view before forwarding.  An arm that carries the
    # name half as text rather than as the object loses the key on its own,
    # whatever the arms around it do, so every arm is pinned separately.
    def types_of(kw):
        return list(map(type, kw))

    class WithInit:
        def __init__(self, **kw):
            self.kw = kw

    class WithNew:
        def __new__(cls, **kw):
            self = object.__new__(cls)
            self.kw = kw
            return self

    class Meta(type):
        def __call__(cls, **kw):
            return kw

    class ViaMeta(metaclass=Meta):
        pass

    class Callable:
        def __call__(self, **kw):
            return kw

    class Holder:
        def bound(self, **kw):
            return kw

        @staticmethod
        def static(**kw):
            return kw

        @classmethod
        def classy(cls, **kw):
            return kw

    def plain(**kw):
        return kw

    arms = [
        ("__init__", lambda: WithInit(**{S("a"): 1}).kw),
        ("type.__call__", lambda: type.__call__(WithInit, **{S("a"): 1}).kw),
        ("__new__", lambda: WithNew(**{S("a"): 1}).kw),
        ("metaclass __call__", lambda: ViaMeta(**{S("a"): 1})),
        ("instance __call__", lambda: Callable()(**{S("a"): 1})),
        ("bound method", lambda: Holder().bound(**{S("a"): 1})),
        ("staticmethod", lambda: Holder.static(**{S("a"): 1})),
        ("classmethod", lambda: Holder.classy(**{S("a"): 1})),
        ("builtin dict", lambda: dict(**{S("a"): 1})),
        ("forwarded twice", lambda: plain(**plain(**{S("a"): 1}))),
    ]
    for name, call in arms:
        assert types_of(call()) == [S], (name, types_of(call()))


K = "no_such_keyword"


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


def sum_rejects_an_unknown_keyword():
    _check(
        "sum([1,2], **{K:0})",
        lambda: sum([1, 2], **{K: 0}),
        (
            "TypeError",
            "sum() got an unexpected keyword argument '%s'" % K,
        ),
    )


def round_rejects_an_unknown_keyword():
    _check(
        "round(1.5, **{K:0})",
        lambda: round(1.5, **{K: 0}),
        (
            "TypeError",
            "round() got an unexpected keyword argument '%s'" % K,
        ),
    )


def pow_rejects_an_unknown_keyword():
    _check(
        "pow(2,3,**{K:5})",
        lambda: pow(2, 3, **{K: 5}),
        (
            "TypeError",
            "pow() got an unexpected keyword argument '%s'" % K,
        ),
    )


def int_rejects_an_unknown_keyword():
    _check(
        'int("5", **{K:1})',
        lambda: int("5", **{K: 1}),
        (
            "TypeError",
            "int() got an unexpected keyword argument '%s'" % K,
        ),
    )


def str_rejects_an_unknown_keyword():
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


def print_rejects_an_unknown_keyword():
    _check(
        "print(**{K:7})",
        lambda: print(**{K: 7}),
        (
            "TypeError",
            "print() got an unexpected keyword argument '%s'" % K,
        ),
    )


a_star_star_key_reaches_kwargs_unchanged()
a_matched_name_is_bound_and_the_rest_keep_their_keys()
the_keys_survive_a_sort()
a_literal_keyword_is_an_exact_str()
an_unknown_keyword_names_the_key_it_was_given()
a_duplicate_keyword_is_still_a_type_error()
every_dispatch_arm_keeps_the_key()
sum_rejects_an_unknown_keyword()
round_rejects_an_unknown_keyword()
pow_rejects_an_unknown_keyword()
int_rejects_an_unknown_keyword()
str_rejects_an_unknown_keyword()
type_init_subclass_names_the_class()
sum_two_extras_are_too_many_arguments()
dict_get_takes_no_keyword_arguments()
list_append_takes_no_keyword_arguments()
sorted_reports_sort_for_an_unknown_keyword()
print_rejects_an_unknown_keyword()
print('OK')
