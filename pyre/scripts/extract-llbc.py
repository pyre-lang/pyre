#!/usr/bin/env python3
"""pyre driver for the Charon ULLBC extraction engine.

Declares the pyre crate table and delegates to the neutral engine in
`<repo-root>/scripts/llbc_extract.py`. Artefacts land under
`<repo-root>/build/llbc`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from llbc_extract import (  # noqa: E402
    CrateSpec,
    child_dirs_whose_rs_mention,
    llbc_dest_path,
    run_cli,
)


# Each successful extraction persists the repo-relative Local entries from the
# artefact's own `translated.files` table. That set drives `source=` on later
# checks, supplemented automatically with proc-macro sources, build scripts,
# closure manifests and Cargo.lock. The full cargo closure remains separately
# hashed as `closure=` so inputs the file table cannot prove irrelevant still
# produce a diagnostic when they move.
# `--reconstruct-panic-calls` keeps panic lang-item calls as Charon's
# `Panic` terminator and `--inline-anon-consts` keeps promoted constants
# inline at their use site; both were Charon's default before "Make
# `TerminatorKind::Panic` reconstruction optional" and "Make anon
# constant inlining optional" made them opt-in. Charon documents
# `--inline-anon-consts` as unsound only for a promoted constant used
# at `'static`, which moves it to a local.
CHARON_BODY_SHAPE_ARGS = ["--reconstruct-panic-calls", "--inline-anon-consts"]
# Passed to Charon for every pyre crate. `--hide-marker-traits` drops the
# `Sized` / `Send` / `Sync` clauses from every generic signature: nothing in
# `majit-translate` reads a trait clause, and Charon's own help names the flag
# as a translation speed-up. Measured on `pyre-interpreter`: 97 s -> 81 s of
# CPU for the host pass, with the prepass output unchanged (`jit_trace_gen.rs`,
# `jitcodes_index.bin`, `descrs_index.bin` and the rest of the build script's
# cross-process determinism set compared byte-equal across the flag).
# `--no-typecheck` was tried beside it and made the pass slower.
# `--include libc::_` translates the libc items an extracted crate reaches,
# including const initializers. A foreign NamedConst otherwise stays
# Opaque, so the named-const harvest has no literal. `majit_jitcode::rffi`
# holds the `rffi.py` `RFFI_*` flags `rposix._errno_before` /
# `_errno_after` test, module globals the flow space folds upstream.
# Charon name-matcher `--include` of `core` methods whose bodies would
# otherwise stay `opacity: Foreign` / `body: "Opaque"`. Pattern shape
# matches `libc::_`: `item::_` is subitems only. Inherent-impl methods
# live under an Impl segment, so `core::num::_::div_ceil` matches
# `{uN,usize}::div_ceil`. `--extract-opaque-bodies` translates every
# reached foreign body; keep the narrow list.
CHARON_CORE_INCLUDE = [
    "--include",
    "core::num::_::div_ceil",
    "--include",
    "core::num::_::saturating_add",
    "--include",
    "core::num::_::saturating_sub",
    "--include",
    "core::num::_::is_ascii_whitespace",
    "--include",
    "core::num::_::wrapping_sub",
    "--include",
    "core::char::methods::_::from_u32",
    "--include",
    "core::char::methods::_::len_utf8",
    "--include",
    "core::char::convert::from_u32",
    "--include",
    "core::char::convert::char_try_from_u32",
    "--include",
    "core::char::methods::len_utf8",
    "--include",
    "core::ops::range::_::contains",
    "--include",
    "core::ops::range::_::new",
    "--include",
    "core::ops::range::_::start_bound",
    "--include",
    "core::ops::range::_::end_bound",
    "--include",
    "core::ops::range::RangeBounds::contains",
    # `RangeInclusive` is a foreign struct whose fields (`start`/`end`)
    # stay `kind: Opaque` unless the type itself is included; start_bound
    # then reads those fields as `Adt` and Bound::Included unions
    # `Instance(Adt) ∪ Integer`.
    "--include",
    "core::ops::range::RangeInclusive",
    "--include",
    "core::cmp::impls::_::le",
    "--include",
    "core::cmp::impls::_::lt",
]
CHARON_ARGS = [
    "--hide-marker-traits",
    *CHARON_BODY_SHAPE_ARGS,
    "--include",
    "libc::_",
    "--include",
    "majit_jitcode::rffi::_",
    *CHARON_CORE_INCLUDE,
]


def _majit_rlib_cargo_args() -> list[str]:
    """Features pyre-interpreter enables on majit-rlib for this `CARGO_FEATURES`.

    `node.py new_funcnode` substitutes `rsandbox.get_sandbox_stub` only when
    `db.sandbox`. majit-rlib's `host_env` is not a default; the interpreter's
    default `host_env` enables `majit-rlib/host_env`. `dynasm` is a
    pyre-interpreter / pyre-jit feature majit-rlib does not have, so
    `{features}` is not passed through. `layout_cargo_args` stays unset:
    wasm32 reuses these flags, matching the interpreter layout pass that
    already compiles `majit-rlib/host_env`.
    """
    names = {
        f.strip()
        for f in os.environ.get("CARGO_FEATURES", "dynasm").split(",")
        if f.strip()
    }
    if "sandbox" in names:
        return ["--features", "sandbox"]
    return ["--features", "host_env"]


# `pyre-native` is the stable native/backend boundary.  Charon follows local
# workspace dependencies by default, so merely moving an implementation into
# that crate does *not* keep it out of an extracting crate's ULLBC.  Keep the
# crate root opaque explicitly: callers retain declarations for residual calls
# and opaque pointer/value types, while compression/crypto/TLS bodies are never
# translated into the interpreter or JIT artefacts.  Reusable pure engines live
# in rustpython-common instead; the output audit below guards their foreign
# boundary as well.
PYRE_RUNTIME_CHARON_ARGS = [*CHARON_ARGS, "--opaque", "pyre_native"]


def interpreter_charon_args(crate_dir: Path) -> list[str]:
    """Do not walk `crate::module` as a crate root; translate a module item
    only if something outside it reaches it.

    Charon's own example: `--opaque crate::module --include crate::module::_`.
    rustc still compiles the files. `--start-from crate` keeps the crate
    walk: a non-empty `--start-from` list replaces that default. Modules
    that carry a JIT-hint macro are `--start-from-if-exists` so their
    marker consts stay in the artefact (`harvest_hints_from_llbcs`) even
    when the module is cfg'd out on a layout target (`mmap` /
    `_cffi_backend` on wasm). A hand list of residual modules is not
    kept. `posix` / `gc` / `time` need no special case: core calls reach
    them and `--include crate::module::_` then translates those items.
    """
    args = [
        *PYRE_RUNTIME_CHARON_ARGS,
        "--opaque",
        "crate::module",
        "--include",
        "crate::module::_",
        # Charon has no const-only matcher: `--include PATH` translates that
        # item and its subitems. These are the localsplus flag consts the
        # interpreter reads (`bytecode.rs` `CO_FAST_*`). Naming each const
        # translates its initializer and leaves the rest of
        # `rustpython_compiler_core::bytecode` opaque.
        "--include",
        "rustpython_compiler_core::bytecode::CO_FAST_HIDDEN",
        "--include",
        "rustpython_compiler_core::bytecode::CO_FAST_LOCAL",
        "--include",
        "rustpython_compiler_core::bytecode::CO_FAST_CELL",
        "--include",
        "rustpython_compiler_core::bytecode::CO_FAST_FREE",
        "--start-from",
        "crate",
    ]
    for name in child_dirs_whose_rs_mention(crate_dir / "src" / "module", "majit_macros::"):
        args += ["--start-from-if-exists", f"crate::module::{name}"]
    return args


SPECS: dict[str, CrateSpec] = {
    # `corpus` lives outside the crate graph the metadata walk sees, so its
    # fingerprint inputs are explicit pathspecs.
    "corpus": CrateSpec(
        name="corpus",
        crate_dir=ROOT / "majit" / "charon-corpus",
        output_name="corpus.ullbc",
        fingerprint_pathspecs=[
            "majit/charon-corpus/Cargo.toml",
            "majit/charon-corpus/src/",
        ],
        charon_args=CHARON_BODY_SHAPE_ARGS,
        # A reader fixture, not a build input: nothing consumes its layouts
        # for a cross target.
        layout_targets=(),
    ),
    # `majit-rlib` gets its own artefact rather than riding along in
    # `pyre-object`'s. Charon roots a translation at the crate being extracted
    # and reaches a dependency's items only through a reference, which prunes
    # exactly what the JIT needs from `rbigint`: its hint markers
    # (`_elidable_function_*` and friends) are consts nothing calls, so the
    # reachability walk never reaches them and every `@jit.elidable` on rbigint
    # silently disappears. Extracted as its own crate, it is translated whole.
    # (`--start-from majit_rlib::_` on `pyre-object` does not substitute: it
    # changes opacity, not what cross-crate metadata offers up.)
    "majit-rlib": CrateSpec(
        name="majit-rlib",
        crate_dir=ROOT / "majit" / "majit-rlib",
        output_name="majit-rlib.ullbc",
        charon_args=CHARON_ARGS,
        cargo_args=_majit_rlib_cargo_args(),
    ),
    "pyre-object": CrateSpec(
        name="pyre-object",
        crate_dir=ROOT / "pyre" / "pyre-object",
        output_name="pyre-object.ullbc",
        charon_args=CHARON_ARGS,
    ),
    "pyre-module": CrateSpec(
        name="pyre-module",
        crate_dir=ROOT / "pyre" / "pyre-module",
        output_name="pyre-module.ullbc",
        # Charon follows workspace deps, so without these the module
        # artefact would re-translate interpreter/object bodies the
        # prepass already loads from their own ullbcs.
        charon_args=[
            *PYRE_RUNTIME_CHARON_ARGS,
            "--opaque",
            "pyre_interpreter",
            "--opaque",
            "pyre_object",
        ],
        cargo_args=["--features", "pyre-interpreter/{features}"],
    ),
    "pyre-interpreter": CrateSpec(
        name="pyre-interpreter",
        crate_dir=ROOT / "pyre" / "pyre-interpreter",
        output_name="pyre-interpreter.ullbc",
        charon_args=interpreter_charon_args(ROOT / "pyre" / "pyre-interpreter"),
        cargo_args=["--features", "{features}"],
    ),
    "pyre-jit": CrateSpec(
        name="pyre-jit",
        crate_dir=ROOT / "pyre" / "pyre-jit",
        output_name="pyre-jit.ullbc",
        charon_args=PYRE_RUNTIME_CHARON_ARGS,
        # `--no-default-features` drops `prepass`: `pyre-jit-trace`'s build
        # script then writes its placeholders without build-depending on a
        # host copy of `pyre-interpreter` + `majit-translate`, the one unit
        # set this pass compiled that no artefact ever read. The
        # `pyre-module` default goes with it: it links `pyre-module` into
        # that build script and so brings the host copy of the interpreter
        # back.
        cargo_args=["--no-default-features", "--features", "{features}"],
        # No layout sidecar. The 588 layout-carrying types declared
        # only here — `jit::{flow,codewriter,flatten,regalloc}`,
        # `majit_*`, and closure environments — are the compiler's own
        # data structures rather than the object model traced bytecode
        # reads. Every runtime type reached through a descr comes from
        # pyre-object or pyre-interpreter, both of which do get sidecars.
        layout_targets=(),
    ),
}

DEFAULT_CRATES = [
    "majit-rlib",
    "pyre-object",
    "pyre-interpreter",
    "pyre-module",
    "pyre-jit",
]

# Targets, besides the extraction host, that get a layout sidecar. The
# wasm32 build reads the same `build/llbc` set as the native build, and its
# pointers are 4 bytes wide: without its own field offsets every descr field
# past the first pointer names the wrong bytes.
LAYOUT_TARGETS = ("wasm32-unknown-unknown",)

# The wasm32 compiler pass needs the same `getrandom` backend selection the
# wasm build itself uses (`check.py`'s `WASM_RUSTFLAGS`) — the default
# backend refuses to build for `wasm32-unknown-unknown`.
LAYOUT_TARGET_RUSTFLAGS = '--cfg getrandom_backend="custom"'

# Bump only when pyre's extraction behaviour changes in a way not already
# represented by the effective cargo/Charon/layout flags hashed by the engine.
# The explicit ABI keeps comments, diagnostics, tests, the forwarding shim and
# the Charon installer from invalidating every multi-minute artefact.
EXTRACTION_ABI = "1"

# Workspace resolution is a compiler input. Extraction-tool implementation
# files are deliberately absent; FINGERPRINT_SCHEMA and EXTRACTION_ABI carry
# their output-affecting semantics without coupling cache keys to every edit.
BASE_PATHSPECS = [
    "Cargo.lock",
    "Cargo.toml",
]


def main() -> None:
    specs = SPECS
    crates = DEFAULT_CRATES
    layout_targets = LAYOUT_TARGETS
    out_dir = ROOT / "build" / "llbc"
    run_cli(
        specs,
        crates,
        root=ROOT,
        out_dir=out_dir,
        extraction_abi=EXTRACTION_ABI,
        base_pathspecs=BASE_PATHSPECS,
        metadata_feature_crates=("pyre-interpreter", "pyre-jit"),
        layout_targets=layout_targets,
        layout_target_rustflags=LAYOUT_TARGET_RUSTFLAGS,
    )

    # A crate move without this output check regressed once already: Charon
    # follows workspace dependencies, so `pyre-native` appeared to be outside
    # the interpreter while 83 of its items still had Transparent bodies in
    # `pyre-interpreter.ullbc`.  Scan one item record at a time (the JSON is a
    # single ~GB line, so line iteration or json.load is not viable) and refuse
    # any runtime artefact that translated a native item body.
    modes_without_an_artefact = {
        "--fingerprint",
        "--list-inputs",
        "--self-test",
        "--prune-store",
    }
    if not modes_without_an_artefact.intersection(sys.argv[1:]):
        requested = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
        # `crates` is the set `run_cli` was just asked to extract. Falling
        # back to `DEFAULT_CRATES` here would demand `pyre-module.ullbc` from
        # a core run that deliberately did not produce it.
        check_crates = requested or crates
        dest_dir = llbc_dest_path(out_dir, ROOT)
        for crate in check_crates:
            if crate not in {"pyre-module", "pyre-interpreter", "pyre-jit"}:
                continue
            artefact = dest_dir / SPECS[crate].output_name
            _assert_runtime_library_items_opaque(artefact)


def _assert_runtime_library_items_opaque(path: Path) -> None:
    """Refuse translated native or rustpython-common engine bodies."""
    delimiter = b'{"def_id":'
    native_prefix = b',"item_meta":{"name":[{"Ident":["pyre_native",0]'
    common_prefix = b',"item_meta":{"name":[{"Ident":["rustpython_common",0]'
    # `inet` reaches an artefact only on a target with no host socket layer,
    # since `_socket` imports it exactly there; on the hosts this script
    # extracts from, that entry matches nothing and guards the extraction
    # targets rather than the current ones.
    common_engine_prefixes = (
        common_prefix + b'},{"Ident":["binascii",0]',
        common_prefix + b'},{"Ident":["compression",0]',
        common_prefix + b'},{"Ident":["hashlib",0]',
        common_prefix + b'},{"Ident":["inet",0]',
        common_prefix + b'},{"Ident":["json",0]',
        common_prefix + b'},{"Ident":["encodings",0]},{"Ident":["cjk",0]',
    )
    opacity_key = b'"opacity":"'
    buffer = bytearray()

    def check_record(record: bytes | bytearray) -> None:
        comma = record.find(b",")
        if comma < 0:
            return
        item = record[comma:]
        # A workspace-local body is only ever legitimate as `Opaque`; a crate
        # Charon never enters is `Foreign` instead, and either answer keeps the
        # engine out of the artefact.
        if item.startswith(native_prefix):
            boundary, allowed = "pyre_native", {"Opaque"}
        elif any(item.startswith(prefix) for prefix in common_engine_prefixes):
            boundary, allowed = "rustpython_common engine", {"Opaque", "Foreign"}
        else:
            return
        start = record.find(opacity_key)
        if start < 0:
            raise SystemExit(f"extract-llbc.py: {path} {boundary} item has no opacity")
        start += len(opacity_key)
        end = record.find(b'"', start)
        opacity = bytes(record[start:end]).decode("ascii", errors="replace")
        if opacity not in allowed:
            raise SystemExit(
                f"extract-llbc.py: {path} contains a {opacity} {boundary} item; "
                "library bodies must stay outside runtime ULLBC"
            )

    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            buffer.extend(chunk)
            records = buffer.split(delimiter)
            for record in records[:-1]:
                check_record(record)
            buffer = bytearray(records[-1])
    check_record(buffer)


if __name__ == "__main__":
    main()
