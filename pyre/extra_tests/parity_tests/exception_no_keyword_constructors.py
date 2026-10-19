# CPython-suite gap: `test_exceptions` constructs UnicodeError subclasses
# with five (or four) positionals and never keyword-only field names.
# `ValueError(foo=1)` is also absent from the suite, as are keyword
# calls on MemoryError, KeyError, IndentationError, Warning, and OSError
# subclasses.
#
# parity-tests reason: `_PyArg_NoKeywords` inside `BaseException_init`
# and the UnicodeError inits reports `{Type}() takes no keyword
# arguments`, using the receiver type. `W_UnicodeDecodeError.descr_init`
# / `W_UnicodeEncodeError.descr_init` bind the five fields as interp2app
# arguments, so pypy3 accepts those keywords. `W_BaseException.descr_init`
# names `BaseException.__init__`. `W_SyntaxError.descr_init` names
# `SyntaxError.__init__` for IndentationError and TabError.
# `W_OSError.descr_new` says `OSError does not take keyword arguments`
# for every OSError subclass. `W_SystemExit.descr_init` and
# `W_StopIteration.descr_init` name their own `.__init__`.
#
# pyre-check: pypy-diverges: pypy3 constructs UnicodeDecodeError and
# UnicodeEncodeError from keyword field names successfully.
# UnicodeError, ValueError, MemoryError, and KeyError keyword calls
# raise `BaseException.__init__() takes no keyword arguments`.
# UnicodeTranslateError names `UnicodeTranslateError.__init__`.
# IndentationError and TabError name `SyntaxError.__init__`.
# BlockingIOError and FileNotFoundError raise
# `OSError does not take keyword arguments`. A MemoryError subclass
# still names `BaseException.__init__`. SystemExit and StopIteration
# name `SystemExit.__init__` / `StopIteration.__init__`.
#
# exception_kwarg_count_overflows_before_unknown:
# `test_exceptions` and `test_call` reject a single unknown ImportError keyword (`invalid`
# / `blech`) and never pass more keywords than the ParseTuple kwlist length.
# `vgetargskeywords` reports `takes at most N keyword argument(s) (M given)` when `nargs +
# nkwargs > len`, and only then scans for an unknown key. `NameError_init` /
# `AttributeError_init` / `ImportError_init` parse against an empty positional tuple, so
#
# exception_new_surrogate_kwarg:
# `test_exceptions` constructs through `Type(*args)` and never calls
# `BaseException.__new__(cls, *args, **kw)` with a keyword name that has no UTF-8
# encoding. `descr_new_base_exception` does `args_w, kwds_w = __args__.unpack()` and `#
# ignore kwds`. `Arguments.unpack` keeps keyword names as `text_w` (`self._utf8`), so a
# lone surrogate is ignored rather than encoded. `W_OSError.descr_new` unpacks the same
#
# exception_new_unpack_ignores_keywords:
# `test_exceptions` constructs through `Type(*args)` and never calls
# `BaseException.__new__(cls, *args, **kw)`, which is the `descr_new_base_exception` path
# that unpacks then ignores `kwds_w`. `descr_new_base_exception` does `args_w, kwds_w =
# __args__.unpack()` and `# ignore kwds`, so `BaseException.__new__(ValueError, 1, foo=2)`
# stores `args == (1,)`. `W_OSError.descr_new` unpacks the same way and rejects a
def reject(fn, message):
    try:
        fn()
    except TypeError as err:
        assert str(err) == message, err
    else:
        raise AssertionError("expected TypeError")


reject(
    lambda: UnicodeDecodeError(encoding="u", object=b"x", start=0, end=1, reason="r"),
    "UnicodeDecodeError() takes no keyword arguments",
)
reject(
    lambda: UnicodeEncodeError(encoding="u", object="x", start=0, end=1, reason="r"),
    "UnicodeEncodeError() takes no keyword arguments",
)
reject(
    lambda: UnicodeTranslateError(object="x", start=0, end=1, reason="r"),
    "UnicodeTranslateError() takes no keyword arguments",
)
reject(lambda: UnicodeError(encoding="u"), "UnicodeError() takes no keyword arguments")
reject(lambda: ValueError(foo=1), "ValueError() takes no keyword arguments")
reject(lambda: ValueError().__init__(foo=1), "ValueError() takes no keyword arguments")
reject(lambda: MemoryError(foo=1), "MemoryError() takes no keyword arguments")
reject(lambda: KeyError(foo=1), "KeyError() takes no keyword arguments")
reject(lambda: KeyboardInterrupt(foo=1), "KeyboardInterrupt() takes no keyword arguments")
reject(
    lambda: IndentationError(foo=1),
    "IndentationError() takes no keyword arguments",
)
reject(lambda: TabError(foo=1), "TabError() takes no keyword arguments")
reject(
    lambda: BlockingIOError(foo=1),
    "BlockingIOError() takes no keyword arguments",
)
reject(
    lambda: FileNotFoundError(foo=1),
    "FileNotFoundError() takes no keyword arguments",
)


class ME(MemoryError):
    pass


reject(lambda: ME(foo=1), "ME() takes no keyword arguments")

# One representative per constructor path: the shared `BaseException_init`,
# the types with their own init, and OSError's `__new__`/`__init__` split.
for T in (
    BaseException,
    Exception,
    GeneratorExit,
    SystemExit,
    StopIteration,
    Warning,
    SyntaxError,
    OSError,
    PermissionError,
):
    reject(lambda T=T: T(foo=1), f"{T.__name__}() takes no keyword arguments")


class PE(PermissionError):
    pass


reject(lambda: PE(foo=1), "PE() takes no keyword arguments")

def exception_kwarg_count_overflows_before_unknown():
    def name_error_two_keywords_overflows():
        try:
            NameError(name="x", bogus=1)
        except TypeError as err:
            assert str(err) == "NameError() takes at most 1 keyword argument (2 given)", err
        else:
            raise AssertionError("expected TypeError")


    def name_error_one_unknown_is_unexpected():
        try:
            NameError(bogus=1)
        except TypeError as err:
            assert str(err) == "NameError() got an unexpected keyword argument 'bogus'", err
        else:
            raise AssertionError("expected TypeError")


    def unbound_local_error_names_name_error():
        try:
            UnboundLocalError(name="x", bogus=1)
        except TypeError as err:
            assert str(err) == "NameError() takes at most 1 keyword argument (2 given)", err
        else:
            raise AssertionError("expected TypeError")


    def attribute_error_three_keywords_overflows():
        try:
            AttributeError(name="n", obj=1, extra=1)
        except TypeError as err:
            assert (
                str(err) == "AttributeError() takes at most 2 keyword arguments (3 given)"
            ), err
        else:
            raise AssertionError("expected TypeError")


    def attribute_error_one_unknown_is_unexpected():
        try:
            AttributeError(name="n", bogus=1)
        except TypeError as err:
            assert (
                str(err) == "AttributeError() got an unexpected keyword argument 'bogus'"
            ), err
        else:
            raise AssertionError("expected TypeError")


    def import_error_four_keywords_overflows():
        try:
            ImportError(name="n", path="p", name_from="f", extra=1)
        except TypeError as err:
            assert (
                str(err) == "ImportError() takes at most 3 keyword arguments (4 given)"
            ), err
        else:
            raise AssertionError("expected TypeError")


    def import_error_one_unknown_is_unexpected():
        try:
            ImportError(name="n", bogus=1)
        except TypeError as err:
            assert str(err) == "ImportError() got an unexpected keyword argument 'bogus'", err
        else:
            raise AssertionError("expected TypeError")


    def module_not_found_error_names_import_error():
        try:
            ModuleNotFoundError(name="n", path="p", name_from="f", extra=1)
        except TypeError as err:
            assert (
                str(err) == "ImportError() takes at most 3 keyword arguments (4 given)"
            ), err
        else:
            raise AssertionError("expected TypeError")


    name_error_two_keywords_overflows()
    name_error_one_unknown_is_unexpected()
    unbound_local_error_names_name_error()
    attribute_error_three_keywords_overflows()
    attribute_error_one_unknown_is_unexpected()
    import_error_four_keywords_overflows()
    import_error_one_unknown_is_unexpected()
    module_not_found_error_names_import_error()
exception_kwarg_count_overflows_before_unknown()

def exception_new_surrogate_kwarg():
    LONE = chr(0xD800)

    e = BaseException.__new__(ValueError, 1, **{LONE: 2})
    assert type(e) is ValueError, type(e)
    assert e.args == (1,), e.args

    try:
        OSError(1, **{LONE: 2})
    except TypeError as err:
        assert str(err) == "OSError() takes no keyword arguments", err
    except UnicodeEncodeError as err:
        raise AssertionError("surrogate keyword name must not encode") from err
    else:
        raise AssertionError("expected TypeError")

    try:
        OSError.__new__(OSError, 1, **{LONE: 2})
    except TypeError as err:
        assert str(err) == "OSError() takes no keyword arguments", err
    except UnicodeEncodeError as err:
        raise AssertionError("surrogate keyword name must not encode") from err
    else:
        raise AssertionError("expected TypeError")
exception_new_surrogate_kwarg()

def exception_new_unpack_ignores_keywords():
    e = BaseException.__new__(ValueError, 1, foo=2)
    assert type(e) is ValueError, type(e)
    assert e.args == (1,), e.args

    try:
        OSError.__new__(OSError, 2, "x", filename="a")
    except TypeError as err:
        assert str(err) == "OSError() takes no keyword arguments", err
    else:
        raise AssertionError("expected TypeError")
exception_new_unpack_ignores_keywords()
print("OK")
