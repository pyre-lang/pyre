# CPython-suite gap: clinic/gateway binding for missing required slots,
# explicit None vs omission on code.replace, constructor keyword rejection,
# duplicate named+positional wording, and positional-only counts is spread
# across many tests that never pin these exact calls together.
#
# parity-tests reason: Arguments._match_signature fills required slots or
# raises ArgErrMissing; an explicit None in code.replace is a supplied
# value (pycode.py replace); array.array and _SSLContext/MemoryBIO reject
# unexpected keywords; clinic reports a duplicate as
# "argument for fname() given by name ('x') and position (n)" with n
# counted without the receiver. gc.get_objects validates generation,
# then returns every object for any accepted value.
#
# pyre-check: pypy-diverges: pypy3 words _sre.compile() as Pattern.__new__
# missing three arguments; str.split(" ", sep=",") as "takes no keyword
# arguments"; array.array as "does not take keyword arguments";
# _opcode.stack_effect(opcode=1) succeeds; start_new_thread takes no
# keywords in the text signature but pypy still binds some names
# differently; ssl.SSLContext.__new__ accepts unexpected keywords;
# gc.get_objects(generation) raises NotImplementedError for a
# non-None generation.

import sys


def reject(fn, message=None):
    try:
        fn()
    except TypeError as err:
        if message is not None:
            assert str(err) == message, err
        return str(err)
    raise AssertionError("expected TypeError")


def missing_sre_compile_names_pattern():
    import _sre

    reject(_sre.compile, "compile() missing required argument 'pattern' (pos 1)")


def code_replace_explicit_none_is_supplied():
    code = (lambda: None).__code__
    reject(lambda: code.replace(co_filename=None))
    reject(lambda: code.replace(co_consts=None))
    reject(lambda: code.replace(co_argcount=None))
    assert code.replace(co_name="x").co_name == "x"


def code_replace_rejects_positionals():
    code = (lambda: None).__code__
    reject(lambda: code.replace(1), "replace() takes no positional arguments")


def array_new_rejects_keywords_for_the_exact_type():
    try:
        import array
    except ImportError:
        return
    reject(
        lambda: array.array.__new__(array.array, "i", unexpected=1),
        "array.array() takes no keyword arguments",
    )


def ssl_constructors_reject_unexpected_keywords():
    try:
        import _ssl
    except ImportError:
        return
    ctx = getattr(_ssl, "_SSLContext", None) or getattr(_ssl, "SSLContext", None)
    bio = getattr(_ssl, "MemoryBIO", None)
    if ctx is not None:
        proto = getattr(_ssl, "PROTOCOL_TLS_CLIENT", None) or getattr(
            _ssl, "PROTOCOL_TLS", 2
        )
        reject(lambda: ctx.__new__(ctx, proto, unexpected=1))
    if bio is not None:
        reject(lambda: bio.__new__(bio, unexpected=1))


def split_duplicate_reports_position_without_self():
    reject(
        lambda: "a b".split(" ", sep=","),
        "argument for split() given by name ('sep') and position (1)",
    )


def str_duplicate_encoding_is_clinic_wording():
    reject(
        lambda: str(b"x", "utf-8", encoding="ascii"),
        "argument for str() given by name ('encoding') and position (2)",
    )


def missing_required_abs_and_len():
    reject(lambda: abs())
    reject(lambda: len())


def missing_required_signature_slots():
    reject(lambda: compile(), "compile() missing required argument 'source' (pos 1)")
    reject(lambda: open(), "open() missing required argument 'file' (pos 1)")
    reject(lambda: round(), "round() missing required argument 'number' (pos 1)")
    reject(lambda: pow(), "pow() missing required argument 'base' (pos 1)")
    reject(lambda: getattr())
    reject(lambda: getattr(1))
    reject(lambda: sum())
    try:
        import math

        reject(lambda: math.log())
        reject(lambda: math.isclose())
        reject(lambda: math.isclose(1))
    except ImportError:
        pass
    try:
        import functools

        reject(lambda: functools.reduce())
        reject(lambda: functools.reduce(max))
    except ImportError:
        pass
    assert "a b".split() == ["a", "b"]
    assert (1).to_bytes() == b"\x01"
    assert enumerate(["a", "b"]).__next__() == (0, "a")
    assert list(zip([1], [2])) == [(1, 2)]
    assert list(map(lambda x: x + 1, [1])) == [2]
    lst = [2, 1]
    lst.sort()
    assert lst == [1, 2]
    try:
        import math

        assert math.log(1.0) == 0.0
        assert math.isclose(1.0, 1.0)
        assert math.perm(5) == 120
        assert math.prod([2, 3]) == 6
    except ImportError:
        pass
    try:
        import cmath

        z = cmath.log(1 + 0j)
        assert z.real == 0.0 and z.imag == 0.0
        assert cmath.isclose(1 + 0j, 1 + 0j)
    except ImportError:
        pass
    try:
        import unicodedata

        assert unicodedata.digit("1") == 1
        assert unicodedata.name("A") == "LATIN CAPITAL LETTER A"
    except ImportError:
        pass


def opcode_stack_effect_opcode_is_positional_only():
    import _opcode

    reject(lambda: _opcode.stack_effect(opcode=1))


def thread_start_new_thread_is_positional_only():
    import _thread

    reject(
        lambda: _thread.start_new_thread(function=lambda: None, args=()),
    )


def select_select_is_positional_only():
    try:
        import select
    except ImportError:
        return
    reject(lambda: select.select(rlist=[], wlist=[], xlist=[]))
    # `interp_select.py select(..., w_timeout=None)` — three fd lists
    # bind timeout as None (block). A zero timeout is the non-blocking form.
    # Winsock rejects a call with all three lists empty.
    if sys.platform != "win32":
        assert select.select([], [], [], 0) == ([], [], [])


def posix_stat_omitted_kwonly_uses_defaults():
    import os

    os.stat(".")


def omitted_int_and_subclass_zero():
    assert int() == 0

    class _ZeroSentinel(int):
        pass

    assert _ZeroSentinel() == 0


def omitted_gc_generation_and_stats():
    try:
        import gc
    except ImportError:
        return
    gc.collect()
    gc.get_objects()
    gc.get_stats()

    class Sentinel:
        pass

    sentinel = Sentinel()
    print("none", any(obj is sentinel for obj in gc.get_objects()))
    print("-1", any(obj is sentinel for obj in gc.get_objects(-1)))
    for generation in (0, 1, 2):
        print("list", generation, isinstance(gc.get_objects(generation), list))
    for generation in (3, -2, "x"):
        try:
            gc.get_objects(generation)
        except Exception as exc:
            print("err", generation, type(exc).__name__, str(exc))
    if sys.implementation.name == "pyre":
        assert any(obj is sentinel for obj in gc.get_objects(1))


def contextvar_omitted_default_is_absent():
    try:
        from _contextvars import ContextVar
    except ImportError:
        return
    var = ContextVar("x")
    assert var.get(1) == 1
    var_none = ContextVar("y", default=None)
    assert var_none.get() is None


def ffi_dlopen_omitted_flags():
    try:
        from _cffi_backend import FFI
    except ImportError:
        return
    # `dlopen(None)` names the running process; the Windows `rdynload.dlopen`
    # raises "cannot use None" for it.
    if sys.platform == "win32":
        return
    FFI().dlopen(None)


def omitted_bool_and_cell_contents():
    assert bool() is False
    from types import CellType

    cell = CellType()
    try:
        cell.cell_contents
    except ValueError:
        pass
    else:
        raise AssertionError("empty cell must raise ValueError")


def omitted_binascii_and_crc32_defaults():
    try:
        import binascii
    except ImportError:
        return
    assert binascii.a2b_base64(b"YQ==") == b"a"
    assert binascii.b2a_base64(b"a") == b"YQ==\n"
    assert isinstance(binascii.crc32(b"x"), int)


def omitted_zlib_decompressor_wbits():
    try:
        import zlib
    except ImportError:
        return
    zlib._ZlibDecompressor()


def omitted_socket_init_optionals():
    try:
        import _socket
    except ImportError:
        return
    s = _socket.socket()
    s.close()


def omitted_create_dynamic_file():
    import _imp
    import types

    spec = types.SimpleNamespace(name="no_such_ext", origin="no_such_ext")
    try:
        _imp.create_dynamic(spec)
    except ImportError:
        pass


def fork_exec_omitted_allow_vfork_default():
    try:
        import _posixsubprocess
    except ImportError:
        return
    # 22 positionals omit trailing `allow_vfork` (`Function.defs_w`).
    # Invalid `args` must TypeError on the value, not "takes exactly".
    try:
        _posixsubprocess.fork_exec(
            123,
            [b"exe"],
            True,
            (),
            None,
            [b"env"],
            -1,
            -1,
            -1,
            -1,
            1,
            2,
            3,
            4,
            True,
            True,
            0,
            False,
            [],
            0,
            -1,
            None,
        )
    except TypeError as err:
        assert "takes exactly" not in str(err), err
    else:
        raise AssertionError("expected TypeError")


def posix_spawn_omitted_sigset_defaults_are_empty():
    import os
    import sys

    spawn = getattr(os, "posix_spawn", None)
    if spawn is None:
        return
    # clinic: `setsigmask=()`, `setsigdef=()`, `file_actions=()`. Omitted
    # keywords must bind those empty tuples, not None.
    pid = spawn(sys.executable, [sys.executable, "-c", "pass"], os.environ)
    os.waitpid(pid, 0)


def pickle_dumps_omitted_kwonly_defaults():
    try:
        import pickle
    except ImportError:
        return
    assert pickle.loads(pickle.dumps(1)) == 1


def bytesio_omitted_read_size_is_all():
    import io

    # `interp_bytesio.py` `read_w(w_size=None)` / `convert_size`: omitted
    # size is None and reads the rest of the buffer (`Function.defs_w`).
    buf = io.BytesIO()
    buf.write(b"hello")
    buf.seek(0)
    assert buf.read() == b"hello"
    buf = io.BytesIO(b"hello")
    assert buf.read(1) == b"h"
    assert buf.read() == b"ello"
    assert io.BytesIO().getvalue() == b""
    assert io.BytesIO(b"ab").readline() == b"ab"


def itertools_product_and_batched_omitted_kwonly():
    try:
        import itertools
    except ImportError:
        return
    assert list(itertools.product("ab", "cd")) == [
        ("a", "c"),
        ("a", "d"),
        ("b", "c"),
        ("b", "d"),
    ]
    if hasattr(itertools, "batched"):
        assert list(itertools.batched("abcdef", 2)) == [("a", "b"), ("c", "d"), ("e", "f")]
    assert list(itertools.accumulate([1, 2, 3])) == [1, 3, 6]
    assert list(itertools.zip_longest("ab", "c")) == [("a", "c"), ("b", None)]


def structseq_omitted_dict_and_function_name():
    import os

    st = os.stat(".")
    assert type(st)(tuple(st))
    import types

    def inner():
        pass

    fn = types.FunctionType(inner.__code__, {})
    assert fn.__name__ == "inner"


def register_at_fork_omitted_before_is_absent():
    import os

    if not hasattr(os, "register_at_fork"):
        return
    os.register_at_fork(after_in_child=lambda: None)
    reject(lambda: os.register_at_fork(before=None))
    reject(lambda: os.register_at_fork(1), "register_at_fork() takes no positional arguments")


def omitted_sys_exit_defaults_to_none():
    import sys

    try:
        sys.exit()
    except SystemExit as err:
        assert err.code is None
    else:
        raise AssertionError("expected SystemExit")
    reject(lambda: sys.exit(unexpected=1))


def omitted_set_asyncgen_hooks():
    import sys

    sys.set_asyncgen_hooks()


def contextvar_missing_name_is_exact_arity():
    try:
        from _contextvars import ContextVar
    except ImportError:
        import contextvars

        ContextVar = contextvars.ContextVar
    reject(
        lambda: ContextVar(),
        "ContextVar() takes exactly 1 positional argument (0 given)",
    )


missing_sre_compile_names_pattern()
code_replace_explicit_none_is_supplied()
code_replace_rejects_positionals()
array_new_rejects_keywords_for_the_exact_type()
ssl_constructors_reject_unexpected_keywords()
split_duplicate_reports_position_without_self()
str_duplicate_encoding_is_clinic_wording()
missing_required_abs_and_len()
missing_required_signature_slots()
opcode_stack_effect_opcode_is_positional_only()
thread_start_new_thread_is_positional_only()
select_select_is_positional_only()
posix_stat_omitted_kwonly_uses_defaults()
omitted_int_and_subclass_zero()
omitted_gc_generation_and_stats()
contextvar_omitted_default_is_absent()
ffi_dlopen_omitted_flags()
omitted_bool_and_cell_contents()
omitted_binascii_and_crc32_defaults()
omitted_zlib_decompressor_wbits()
omitted_socket_init_optionals()
omitted_create_dynamic_file()
fork_exec_omitted_allow_vfork_default()
posix_spawn_omitted_sigset_defaults_are_empty()
pickle_dumps_omitted_kwonly_defaults()
bytesio_omitted_read_size_is_all()
register_at_fork_omitted_before_is_absent()
itertools_product_and_batched_omitted_kwonly()
structseq_omitted_dict_and_function_name()
omitted_sys_exit_defaults_to_none()
omitted_set_asyncgen_hooks()
contextvar_missing_name_is_exact_arity()
print("OK")
