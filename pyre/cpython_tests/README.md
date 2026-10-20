# CPython regression suite for pyre

Runs the vendored CPython 3.14 regression suite (`lib-python/3/test/`) against
a built pyre interpreter and gates on regressions. This is the analog of
PyPy's `lib-python/conftest.py` and RustPython's `-m test` CI, adapted to
pyre's plain-Python runner convention (no pytest / RPython dependency).

## Design

- **Per-module subprocess.** Each test module runs in its own subprocess of
  the built interpreter, exactly like PyPy launches `pypy3-c -m test <module>`.
  Test pollution can't leak between modules.
- **Vendored tests carry PyPy's edits and nothing else.** `lib-python/3/test/`
  is CPython's `Lib/test` plus the modifications PyPy makes to it in its own
  `lib-python` — those are ported and kept (a re-vendor from CPython drops
  them, so they are restored after one). A pyre-specific skip / expected-status
  decision never becomes a test-file edit: it lives in the external
  `baseline.json`, so stdlib upgrades stay a clean drop-in (see
  `lib-python/stdlib-upgrade.txt`).
- **`_testbuffer` is built by the runner.** `test_buffer`'s
  `TestBufferProtocol` — 95 of that module's cases — is `skipUnless(ndarray)`,
  and `ndarray` comes from `Modules/_testbuffer.c`, which CPython builds beside
  its interpreter. The runner compiles `lib_pypy/_testbuffer.c` — that file
  verbatim, and unchanged across 3.14.6 and 3.14.7 — against
  `include/pyre3.15t` and puts it on `PYTHONPATH` for that module alone,
  because several other modules assert on the path their child interpreters
  inherit. An interpreter without `cpyext`, a platform the feature is not
  compiled for, or a missing C compiler leaves it unbuilt and the class
  skipped, as upstream does when the module is absent. The feature also fills
  `EXTENSION_SUFFIXES`, which unskips `test_importlib.extension.*`; those cases
  want `_testsinglephase` and `_testmultiphase`, whose sources include
  `pycore_*` headers pyre does not ship, so `test_importlib` is recorded
  `FAIL`.
- **External baseline.** `baseline.json` records the expected status of every
  module per backend. A module recorded `PASS` that stops passing is a
  regression and fails CI. The default gate runs only the `PASS` subset (for CI
  budget), so newly-passing modules surface under `--strict-baseline`, `--full`,
  or `--update-baseline`, which also run the non-`PASS` modules.

## Usage

```sh
# build the interpreter first
cargo build --release -p pyrex --bin pyre-dynasm --no-default-features --features dynasm,cpyext

# gate against the baseline (default: dynasm, JIT on, script mode)
python3 pyre/cpython_tests/run.py

# explore the whole suite without gating, write a JSON report
python3 pyre/cpython_tests/run.py --full --report /tmp/report.json

# focus on one module / list selection / regenerate the baseline
python3 pyre/cpython_tests/run.py --filter test_json
python3 pyre/cpython_tests/run.py --list
python3 pyre/cpython_tests/run.py --full --update-baseline
```

Key flags: `--backend dynasm|cranelift`, `--no-jit` (`PYRE_NO_JIT=1`),
`--mode script|module|regrtest`, `--jobs N`, `--timeout S`, `--filter SUB`,
`--baseline PATH`, `--update-baseline`, `--strict-baseline`, `--full`,
`--report PATH`.

### Run modes

- `script` (default): `pyre <path>/test_xxx.py` — runs the file directly as
  `__main__`, firing its `unittest.main()`. The most robust mode today; it
  bypasses `runpy` / `importlib.util.find_spec`. A package, and a file with no
  `if __name__ == "__main__"` block of its own, are driven through a
  synthesized unittest entry — running those directly defines their test
  classes and exits 0 without running a single test.
- `module`: `pyre -m test.<module>` — the same module entry via `runpy`,
  without libregrtest's process setup and result protocol.
- `regrtest`: `pyre -m test -v <module>` — CPython's libregrtest, the
  PyPy/RustPython form. Libregrtest imports cleanly and `test_regrtest` passes;
  use this mode when the runner-owned setup and canonical module identity are
  part of the behavior under test.

## Result classes

`PASS` (rc 0, and unittest reported at least one case it did not skip) · `FAIL`
(unittest ran but failed) · `IMPORTERROR` (module could not even be
imported/run — an interpreter or stdlib-compat gap) · `CRASH` (rust panic /
nonzero `internal_compile_panics` / signal) · `TIMEOUT` · `SKIP` (curated out in
the baseline / `KNOWN_SKIPS`, or the run itself opted out: `unittest.SkipTest`
before any test, a denied resource, or a suite whose every case was skipped).

## CI

- `.github/workflows/pyre-ci.yml` job `cargo-test-linux`, in its trailing
  `Run CPython suite` step — gates PRs on the baseline-`PASS` subset, dynasm
  with **JIT on** (`MAJIT_STRICT=1`), on `ubuntu-24.04` (x86_64). The baseline is recorded on linux-x86_64 and the JIT
  codegen is architecture-specific, so local baseline comparisons must use the
  same host.
- `.github/workflows/pyre-cpython-weekly.yml` — non-gating `--full` across
  three lanes (dynasm JIT-on, dynasm JIT-off, cranelift), weekly or on manual
  dispatch, with reports uploaded as artifacts. A module that passes JIT-off
  but not JIT-on is a JIT correctness divergence. It runs on the gate's own
  host and repeats the gate's `--jobs` and `--timeout`, so a lane's verdict for
  a module means the same thing the baseline means by it; a report measured
  under a tighter per-module budget calls a module TIMEOUT for reasons that
  have nothing to do with the module. Because the non-`PASS` modules are only
  ever run here, this is also where a recorded verdict that has since gone
  stale becomes visible.

## Current state and backlog (Phase 0)

The baseline records **262 `PASS`**, 27 `IMPORTERROR`, 61 `SKIP`,
69 `FAIL`, 7 `CRASH`, and 7 `TIMEOUT` (433 recorded rows, stdlib 3.14.6). These are a
snapshot counted from `baseline.json`, which is the authority when they disagree.
`IMPORTERROR` here means unittest never printed a summary — many of those
labels are stale (the modules now import and run). Re-measure with
`--full --report` before treating one as an import gap. The `PASS` set grows
as the gaps below are closed.
(It was 0 `PASS` / 414 `IMPORTERROR` before the
`-m`/`STORE_SLICE`/submodule-binding/`_ast`/PEP-709-hidden-locals fixes
below.)

Building this infra already surfaced and fixed five interpreter gaps:

- **`-m module`** support (`pyrex/src/lib.rs` — `runpy._run_module_as_main`).
- **`STORE_SLICE`** (`obj[i:j] = v`) — `eval.rs` / `runtime_ops.rs`.
- **submodule attribute binding** — `import a.b` now binds `b` on `a`
  (`importing.rs::absolute_import`, the `_find_and_load` step). Fixes
  `import xml.etree.ElementTree`, `importlib.util`, and a broad class of
  stdlib imports.
- **`_ast` node types are now heap types** (`module/_ast/moduledef.rs`,
  built via `type(name, bases, {})` along the ASDL hierarchy). `ast.py` can
  monkeypatch / subclass them. Fixes `import ast` and, transitively,
  `collections`, `json`, `functools`, `enum`, `textwrap`, `argparse`, …
- **PEP 709 hidden comprehension locals at module scope** (`eval.rs`
  `store_name_value` + `pyframe.rs` fast↔locals sync). A module-level inlined
  comprehension makes its iteration variable a `CO_FAST_HIDDEN` fast local
  whose binding lives in the module dict (via `STORE_NAME`), not the NULL
  fast slot. Two parity fixes: **(a)** `STORE_NAME` now writes straight to
  `w_locals` (PyPy `pyopcode.py:855` `space.setitem_str(getorcreatedebug().
  w_locals, …)`) instead of via `getdictscope()` — the old path ran
  `fast2locals` on every store, which deleted the hidden name from the dict
  (the module's `w_locals` *is* its globals). **(b)** `fast2locals` /
  `locals2fast` skip `CO_FAST_HIDDEN` slots (CPython `frameobject.c`; PyPy
  has no PEP 709) so an explicit `locals()`/`vars()` can't erase them either.
  This was **the gateway**: it unblocks `import dis` → `inspect` →
  `unittest`. Repro: `y=[op for op in range(2)]; op=5; i=6; print(op)`
  (byte-identical to CPython 3.14 after the fix).

Known remaining blockers (the backlog the suite tracks), highest-leverage
first:

1. **Stale `IMPORTERROR` labels.** `handle_fromlist` / `IMPORT_FROM` already
   implement `from package import submodule`. `test.support` and
   `unittest.mock` import. Re-run `--full` and promote modules that now
   PASS or FAIL. CPython-only / Windows / display suites are curated
   `KNOWN_SKIPS` or `PLATFORM_GATED`, not import work.
2. **Semantic FAILs** already inside unittest (`test_xml_etree` self-closing
   reparse, `test_syntax` `expected ':'`, POSIX `pwd`/`grp`/`select`,
   `bz2`/`lzma`, `shutil`/`traceback`/`zoneinfo`). `test_site` is the
   PyPy-shaped `lib/pyre3.15t` layout, not a path bug.
3. **Missing PyPy product modules**, ported natively (do not put `lib_pypy`
   on `sys.path`): `_sqlite3`, `_curses`, `_gdbm`/`_dbm`, Linux `crypt`.
   Do not implement `_datetime`, `_zstd`, or the subinterpreter modules.
