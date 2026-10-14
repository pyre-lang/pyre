# pyre-check: skip-cpython
# pyre-check: skip-backends=wasm
# pyre-check: requires-modules=_cffi_backend
# pyre-check: skip-platforms=win32
# pyre-check: trace-shape=hot_lib_call:max=CallMayForceR:2
# `optind` is a libc `int` on unix. Windows has no such export, so pypy
# exits 1 during dlopen and the oracle never starts.
# `_cffi_backend` is built into pypy but is not a CPython builtin, and this
# host's python3 cannot import it, so pypy alone is the oracle here. The module
# is absent on wasm32 altogether -- the fold arm itself is
# `#[cfg(not(target_arch = "wasm32"))]` -- so that backend is skipped rather
# than left to fail on the import.
#
# No `max-pypy-ratio`: absence exempts the fixture from the ratio gate, which
# is what this file wants. Every leg's wall-clock is dominated by the libffi
# trampoline pyre still pays per foreign call -- `direct_libffi_call` is
# unported, so `OS_LIBFFI_CALL` never reaches a dynamic calldescr -- and a
# ceiling here would therefore gate that hole rather than this arm. The
# `load_attr_cffi_lib` cell fold fired 0 times on the synthetic corpus and
# was retired; this file stays staged until a descent of
# `W_LibObject.descr_getattribute` records the same cell read.
#
# What the arm does: `lib.<name>` on a `_cffi_backend.Lib` is a dict-first
# lookup (`lib_obj.py _get_attr` / `W_LibObject.descr_getattribute`), and the
# `Lib`'s own dict is a module-strategy dict, so the same cell fold that serves
# `math.sqrt` serves it. `hot_lib_call` is what that buys and
# `hot_lib_call_hoisted` is the spelling it is brought level with; without the
# arm the in-loop form carries an extra may-force residual per iteration and
# compiles to 81 ops against the hoisted form's 69, with twice the
# `GuardNotForced` guards.
#
# `hot_global_var` is the leg that says what the arm may NOT swallow, and it is
# the reason this file exists as much as the two above. An `OP_GLOBAL_VAR`
# entry resolves to a `W_GlobSupport`, and reading the attribute is
# `read_global_var` against live C memory -- the dict cell holds the support
# object, not the value. Folding it would return the support object itself, and
# no invalidation can rescue that: writing the global goes straight through
# `write_global_var` and never touches the dict version. So the arm must
# decline on `W_GlobSupport`, and this leg's read-after-write is what fails if
# the decline is ever dropped.

from _cffi_backend import FFI

# `int abs(int)`: OP_FUNCTION(1), OP_PRIMITIVE(PRIM_INT), OP_FUNCTION_END.
# Opcodes are big-endian `(arg << 8) | op`.
T_INT_FN = b"\x00\x00\x01\x0D\x00\x00\x07\x01\x00\x00\x00\x0F"
G_DLOPEN_FUNC = b"\x00\x00\x00\x23"      # OP_DLOPEN_FUNC(type index 0)
# A bare `int` at type index 0, reached by an OP_GLOBAL_VAR global.
T_INT = b"\x00\x00\x07\x01"
G_GLOBAL_VAR = b"\x00\x00\x00\x21"       # OP_GLOBAL_VAR(type index 0)

ffi_fn = FFI("cffi_lib_attr_fn", _version=0x2601, _types=T_INT_FN,
             _globals=(G_DLOPEN_FUNC + b"abs", 0))
lib_fn = ffi_fn.dlopen(None)

# `optind` is a plain `int` in libc on every unix this runs on, and dlsym finds
# it because an ABI-mode global carries a null address until `cdlopen_fetch`
# resolves the name.
ffi_var = FFI("cffi_lib_attr_var", _version=0x2601, _types=T_INT,
              _globals=(G_GLOBAL_VAR + b"optind", 0))
lib_var = ffi_var.dlopen(None)


def hot_lib_call(n):
    total = 0
    for i in range(n):
        total += lib_fn.abs(-(i & 255))
    return total


def hot_lib_call_hoisted(n):
    f = lib_fn.abs
    total = 0
    for i in range(n):
        total += f(-(i & 255))
    return total


def hot_global_var(n):
    # Read-after-write through a `W_GlobSupport`. A folded cell would hand back
    # the support object rather than the value, so this sum is the assertion.
    total = 0
    for i in range(n):
        lib_var.optind = i & 7
        total += lib_var.optind
    return total


def main():
    total = 0
    k = 0
    while k < 10:
        total += hot_lib_call(74766)
        total += hot_lib_call_hoisted(74766)
        total += hot_global_var(24922)
        k += 1
    lib_var.optind = 1
    print(total)


main()
