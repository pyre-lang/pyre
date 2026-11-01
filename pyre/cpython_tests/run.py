#!/usr/bin/env python3
"""Run the vendored CPython regression suite against a pyre binary.

This mirrors PyPy's `lib-python/conftest.py` mechanism: each CPython test
module is run in its OWN subprocess of the built interpreter, the vendored
test files carry PyPy's own modifications and nothing else, and every
pyre-specific expected-status / skip decision lives in an EXTERNAL baseline
(`baseline.json`) — never as an edit to the test files.

Each module is launched one of three ways, selectable with `--mode`:

  * script (default): `pyre <path>/test_xxx.py` — runs the test file directly
    as `__main__` so its `if __name__ == "__main__": unittest.main()` block
    fires. Needs only `unittest` plus the module's own imports, making it the
    most robust default. A package, and a file carrying no such block,
    go through a synthesized unittest entry instead — running those directly
    exits 0 without testing anything. Runner metadata gives
    resource-heavy or dotted-identity-sensitive modules the corresponding
    libregrtest bootstrap without changing this default.
  * module: `pyre -m test.<module>` — same unittest entry but via `runpy`
    and the canonical dotted module identity.
  * regrtest: `pyre -m test -v <module>` — CPython's own libregrtest, the
    PyPy/RustPython form. Libregrtest and `test_regrtest` run cleanly.

A (module, backend) result is classified as:

  PASS        rc 0 (unittest printed OK) and at least one test actually ran
  SKIP        the module opted out: it bailed with unittest.SkipTest, was
              denied a resource, or ran nothing but skips
  FAIL        clean nonzero exit (test failures/errors)
  CRASH       rust panic / nonzero internal_compile_panics / signal
  TIMEOUT     exceeded --timeout
  IMPORTERROR module could not even be imported (an interpreter/stdlib gap;
              this is the Phase-0 backlog the suite self-populates)

Gating (default): the run executes only the baseline-PASS subset — the
non-PASS modules carry no gate signal and running the whole suite at the
per-module timeout overruns the CI budget. A module recorded PASS that no
longer passes is a REGRESSION and makes the run exit nonzero. Newly-passing
modules surface only when the non-PASS modules are actually run, so use
`--strict-baseline` (gates on unrecorded improvements), `--full`, or
`--update-baseline` to detect them. `SKIP` baseline entries are not run.

The baseline is a shared file plus a per-host overlay,
`baseline.<sys.platform>-<machine>.json`, consulted first. Whichever host
records a (module, backend) first sets the shared verdict; every other host
writes an entry only where it disagrees, and drops it again once the two
agree. Two separate mechanisms sit next to this: `PLATFORM_GATED` excludes
modules CPython skips wholesale on a host, and `KNOWN_SKIPS` is a decision
about the module rather than the host, so it stays shared.

Usage:
    python3 pyre/cpython_tests/run.py [--backend dynasm|cranelift]
        [--no-jit] [--mode script|module|regrtest] [--jobs N]
        [--timeout SECONDS] [--filter SUBSTR] [--list]
        [--baseline PATH] [--update-baseline] [--strict-baseline] [--full]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TARGET_RELEASE = ROOT / "target" / "release"
TESTDIR = ROOT / "lib-python" / "3" / "test"
STDLIB_VERSION_FILE = ROOT / "lib-python" / "stdlib-version.txt"
DEFAULT_BASELINE = HERE / "baseline.json"
HOST_TAG = f"{sys.platform}-{platform.machine()}"

EXE = ".exe" if sys.platform == "win32" else ""
BIN_NAME = {"dynasm": "pyre-dynasm", "cranelift": "pyre-cranelift"}

STATUSES = ("PASS", "FAIL", "CRASH", "TIMEOUT", "IMPORTERROR", "SKIP")

# A live module is named once at this age so a cancelled log still identifies
# it. Progress marks wait for a result, so without this a module that hangs
# until the job is killed leaves only unmarked dots.
STILL_RUNNING_AFTER = 120

# Modules deliberately not run, ported from PyPy's lib-python/conftest.py
# `testmap` skip reasons. These are CPython implementation-detail or
# environment-specific tests that are never expected to pass on an
# alternative interpreter; keep them out of the gate regardless of state.
KNOWN_SKIPS = {
    "test.test_dis": "implementation detail",
    "test.test_dict_version": "implementation detail",
    "test.test_gc": "implementation detail",
    "test.test_frozen": "implementation detail",
    "test.test_frame": "implementation detail",
    "test.test_fileutils": "CPython internal test helper absent on PyPy",
    "test.test_sys_setprofile": "implementation detail",
    "test.test_sys_settrace": "implementation detail",
    "test.test_bytecode_helper": "implementation detail",
    "test.test_peepholer": "implementation detail",
    "test.test_opcache": "implementation detail",
    "test.test_refcounting": "implementation detail (refcount semantics)",
    "test.test_capi": "needs C-API",
    "test.test_cppext": "needs C compiler / C-API",
    "test.test_cext": "needs C compiler / C-API",
    "test.test_stable_abi_ctypes": "needs ctypes.pythonapi",
    "test.test_ossaudiodev": "needs low level audio",
    "test.test_winsound": "needs audio hardware",
    "test.test_tk": "needs display",
    "test.test_ttk": "needs display",
    "test.test_idle": "needs display",
    "test.test_tools": "CPython internal build details",
    "test.test_zipfile64": "demands too many resources",
    "test.test_largefile": "demands too many resources",
    "test.test_embed": "needs embedded CPython",
    "test.test_xpickle": "spawns per-version pythonX.Y subprocesses (PATH-dependent, can deadlock)",
    "test.test_c_locale_coercion": "asserts child stderr is empty, but MAJIT_STATS=1 prints a [jit-stats] line to every process's stderr",
    # `test/test_interpreters/__init__.py` raises SkipTest("GIL disabled")
    # when `test.support.Py_GIL_DISABLED` is set, and pyre reports
    # `Py_GIL_DISABLED = 1` (abiflags `t`) on every host. The package therefore
    # opts out everywhere, which is a fact about the build rather than about
    # one host.
    "test.test_interpreters": "package skips itself on a free-threaded build (Py_GIL_DISABLED)",
    # `test/test_types.py` imports `_datetime` at module scope. PyPy reaches
    # `datetime` through its pure-Python implementation and ships no `_datetime`
    # extension module, so that import cannot succeed here without the stub
    # `AGENTS.md` rules out; the module stays blocked on a decision about the
    # module inventory rather than on an interpreter gap.
    "test.test_types": "imports `_datetime`, an extension module PyPy does not provide",
    # CPython-only internals / helpers.  PyPy's conftest skips the same
    # class of modules; none of these have a PyPy owner, so they are not
    # a pyre porting target (AGENTS.md module-presence rule).
    "test.test__interpchannels": "CPython subinterpreter API; PyPy has no owner",
    "test.test__interpreters": "CPython subinterpreter API; PyPy has no owner",
    "test.test_crossinterp": "CPython subinterpreter API; PyPy has no owner",
    "test.test_thread_local_bytecode": "CPython specializing-interpreter internals",
    "test.test_generated_cases": "CPython bytecode case generator",
    "test.test_optimizer": "CPython specializing optimizer",
    "test.test_external_inspection": "CPython remote-debugging helper",
    "test.test_clinic": "CPython Argument Clinic",
    "test.test_asdl_parser": "CPython Parser/asdl internals",
    "test.test_build_details": "CPython build metadata",
    "test.test_getpath": "CPython internal details",
    "test.test_launcher": "CPython Windows py.exe launcher",
    "test.test_winconsoleio": "imports `_testconsole`, a CPython test helper PyPy does not provide",
    "test.test_gdb": "CPython gdb hooks",
    "test.test_perf_profiler": "CPython perf profiler hooks",
    "test.test_perfmaps": "CPython perf map hooks",
    "test.test_xxlimited": "CPython internal details",
    "test.test_xxtestfuzz": "CPython internal details",
    "test.test_android": "Android-only",
    "test.test_zstd": "no PyPy owner for the _zstd accelerator",
    # Display-backed Tk.  Headless `test_tcl` / `test_ttk_textonly` /
    # `test_turtle` stay off this table: PyPy runs them, and they do not
    # need a screen.
    "test.test_tkinter": "needs display",
}

# Whole modules that CPython skips at import on some platforms. Off-platform
# modules are neither run nor written to the baseline, avoiding false
# `PASS -> SKIP` regressions. Keep each predicate aligned with CPython's guard;
# modules that only skip individual cases do not belong here. A host that
# disagrees about a module for any other reason records it in its own
# `host_baseline_path` overlay instead.
PLATFORM_GATED = {
    # `if not is_apple: raise unittest.SkipTest("Apple-specific")`, where
    # `test.support.is_apple` is darwin plus the mobile Apple platforms.
    "test.test_apple": (
        lambda p: p in ("darwin", "ios", "tvos", "watchos"),
        "Apple-specific",
    ),
    # `if sys.platform[:3] == 'win': raise unittest.SkipTest(...)`
    "test.test_threadsignals": (
        lambda p: p[:3] != "win",
        "signals not testable on Windows",
    ),
    # `if support.is_emscripten or support.is_wasi: raise unittest.SkipTest(...)`
    "test.test_selectors": (
        lambda p: p not in ("emscripten", "wasi"),
        "cannot create socketpair on Emscripten/WASI",
    ),
    # `if support.is_wasi: raise unittest.SkipTest(...)`
    "test.test_urllib_response": (
        lambda p: p != "wasi",
        "cannot create socket on WASI",
    ),
    # `if not support.has_fork_support: raise unittest.SkipTest(...)`, and
    # `has_fork_support` is `hasattr(os, "fork")` minus the platforms that
    # have the call but cannot use it: `is_emscripten`, `is_wasi`,
    # `is_apple_mobile` and `is_android`.
    "test.test_fork1": (
        lambda p: p not in ("win32", "emscripten", "wasi", "ios", "tvos", "watchos", "android"),
        "os.fork() not available",
    ),
    # `if sys.platform == "win32": raise unittest.SkipTest("fork is not
    # available on Windows")`, in the test package's `__init__`.
    "test.test_multiprocessing_fork": (
        lambda p: p != "win32",
        "fork is not available on Windows",
    ),
    # `if sys.platform == "win32": raise unittest.SkipTest("forkserver is
    # not available on Windows")`, in the test package's `__init__`.
    "test.test_multiprocessing_forkserver": (
        lambda p: p != "win32",
        "forkserver is not available on Windows",
    ),
    # `if not hasattr(os, "openpty"): raise unittest.SkipTest(...)`
    "test.test_openpty": (
        lambda p: p not in ("win32", "emscripten", "wasi"),
        "os.openpty() not available",
    ),
    # `fcntl = import_module('fcntl')`.  The module needs `sys/ioctl.h` and
    # `fcntl.h`, which win32 has not, and WASI names it in the SDK's
    # unsupported set next to `mmap` and `termios`.  Emscripten builds it:
    # the browser block's own `SET_NA` lists only `readline`.
    "test.test_fcntl": (
        lambda p: p not in ("win32", "wasi"),
        "no fcntl module",
    ),
    # `syslog = import_helper.import_module("syslog")`
    "test.test_syslog": (
        lambda p: p not in ("win32", "emscripten", "wasi"),
        "no syslog module",
    ),
    # `termios = import_module('termios')`
    "test.test_tty": (
        lambda p: p not in ("win32", "emscripten", "wasi"),
        "no termios module",
    ),
    # Windows-only modules.  Each file raises SkipTest or uses
    # `import_module(..., required_on=['win'])` on a POSIX host; keep them
    # out of the shared baseline so a POSIX IMPORTERROR is not recorded as
    # an interpreter gap.
    "test.test_msvcrt": (
        lambda p: p == "win32",
        "windows related tests",
    ),
    "test.test_winconsoleio": (
        lambda p: p == "win32",
        "test only relevant on win32",
    ),
    "test.test_winapi": (
        lambda p: p == "win32",
        "Windows-only _winapi",
    ),
    "test.test_winreg": (
        lambda p: p == "win32",
        "Windows-only winreg",
    ),
    "test.test_wmi": (
        lambda p: p == "win32",
        "Windows-only _wmi",
    ),
    "test.test_startfile": (
        lambda p: p == "win32",
        "Windows-only os.startfile",
    ),
}


def platform_gate(module: str) -> str | None:
    """The reason `module` does not apply to this host, or `None` if it does."""
    gate = PLATFORM_GATED.get(module)
    if gate is None:
        return None
    applies, reason = gate
    return None if applies(sys.platform) else reason


# `Modules/_testbuffer.c` is the exporter `test_buffer`'s `TestBufferProtocol`
# is `skipUnless`d on; `_testsinglephase` and `_testmultiphase` are the two
# `test_importlib.extension` loads as C extensions.  CPython builds all three
# beside the interpreter; pyre builds them here against the same headers
# `pyrex`'s cpyext fixtures compile against.
TEST_EXTENSION_SOURCES = {
    "_testbuffer": ROOT / "lib_pypy" / "_testbuffer.c",
    "_testsinglephase": ROOT / "lib_pypy" / "_testsinglephase.c",
    "_testmultiphase": ROOT / "lib_pypy" / "_testmultiphase.c",
}
CPYEXT_INCLUDE = ROOT / "include" / "pyre3.15t"
# The headers CPython keeps out of an installed tree and builds these modules
# against inside its own: `_PyNamespace_New` and the argument parser Argument
# Clinic writes calls to both live behind one.
CPYEXT_INTERNAL_INCLUDE = CPYEXT_INCLUDE / "internal"
# `PYTHONPATH` is language-visible and does not reach a child started with
# `-E` or `-I`, which `script_helper.run_python_until_end` does whenever the
# caller passes no environment of its own — `test_importlib.extension`'s
# `test_nonmodule_cases` runs `_test_nonmodule_cases.py` that way.  So the
# extensions go on the interpreter's own default `sys.path` instead, whose
# only entry here is `lib-python/3`.  `_testmultiphase_build.py` does the
# same, compiling into `lib_pypy` — a source directory `compute_lib_pypy_path`
# puts on the default path — because `test.importlib` imports the module as a
# C extension and so cannot be served by a Python shim.  Only these three
# names become importable; the artefacts are ignored by git.
EXTENSION_DIR = ROOT / "lib-python" / "3"
NEEDS_TEST_EXTENSION = {"test.test_buffer", "test.test_importlib"}

# These modules intentionally assert on a child interpreter's exact stderr.
# MAJIT_STATS is runner instrumentation rather than language-visible output,
# so do not inject it into those modules or any subprocesses they spawn.
STATS_STDERR_INCOMPATIBLE = {
    "test.test_inspect",
    # test_regrtest exercises both line-oriented `--list-*` output and the
    # worker JSON protocol through several generations of child interpreters.
    # Its subprocess helpers intentionally sanitize their environments, so the
    # root-only ancestry marker cannot keep MAJIT_STATS out of every child.
    "test.test_regrtest",
}

# ── classification ───────────────────────────────────────────────────

# Width of the panic MESSAGE kept next to the `panicked at file:line:col:`
# location. The collector's marking-bug report is the widest one pyre emits:
# it names the holder, the child, both generations, whether the holder is in
# the remembered set, and formats two `[usize; 8]` word dumps.
PANIC_BODY_CHARS = 2000


def panic_message_body(lines: list[str]) -> str:
    """The panic message following a `panicked at file:line:col:` line.

    Rust's default hook writes the location line, then the message, then a
    `note:` / `stack backtrace:` trailer, so the message is everything in
    between. It is not one line: `{:#x?}` is alternate debug, which puts every
    element of an array on its own line, so a report carrying a word dump
    spans ~20 of them. Joining them is what keeps `holder_in_remembered`,
    `child_gen` and `child_vtable_type_id` — the fields that separate a stale
    remembered-set entry from a mis-declared GC offset — in a CI log, which is
    the only place a crash that does not reproduce locally is ever observed.
    """
    body: list[str] = []
    used = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("note:", "stack backtrace:")):
            break
        body.append(stripped[: PANIC_BODY_CHARS - used])
        used += len(body[-1])
        if used >= PANIC_BODY_CHARS:
            break
    return " ".join(body)


def jit_panic_reason(stderr: str) -> str | None:
    """A JIT-level rust panic or nonzero internal_compile_panics, else None.

    Mirrors check.py `_jit_panic_reason`: pyre's panic hook suppresses
    legitimate trace-abandon panics, so anything here is a real crash.
    """
    if not stderr:
        return None
    if "panicked" in stderr:
        lines = stderr.splitlines()
        for idx, line in enumerate(lines):
            if "panicked" in line:
                reason = f"rust panic: {line.strip()[:80]}"
                # The location line alone cannot triage a crash from a CI log.
                # 200 cut the GC's `invalid major child` diagnostic at
                # `holder_words`, which is where the fields that name the
                # culprit start; and those dumps are `{:#x?}` arrays printed one
                # element per line, so a first-line excerpt truncates them at
                # any width. Keep the whole message body instead.
                body = panic_message_body(lines[idx + 1:])
                if body:
                    reason += f" | {body}"
                return reason
        return "rust panic"
    for line in stderr.splitlines():
        if line.startswith("[jit-stats]") and "internal_compile_panics=" in line:
            field = line.split("internal_compile_panics=", 1)[1].split()[0]
            try:
                if int(field) > 0:
                    return f"internal_compile_panics={field}"
            except ValueError:
                pass
    return None


def last_stderr_line(err: str) -> str:
    """The last non-empty stderr line — the only evidence a dying run leaves."""
    for line in reversed(err.splitlines()):
        if line.strip():
            return line.strip()
    return ""


def traceback_verdict(lines: list[str], header: int) -> str:
    """The exception line closing the traceback unittest printed at `header`.

    A block runs from the `FAIL:`/`ERROR:` header to the `====` rule opening
    the next one (or unittest's closing `Ran N tests`). Every frame inside a
    traceback is indented, so the exception is the first unindented line after
    the `Traceback` banner. The banner is what arms the search rather than the
    exception simply being the block's last unindented line, because an
    assertion failure prints its diff below the `AssertionError` and those
    lines are unindented too.

    A chained exception prints one traceback per link, joined by `The above
    exception ...` / `During handling ...`. The last link is the one the test
    actually failed with, so each banner overwrites the previous answer.

    The exception line alone is not always the cause. `assertSetEqual`,
    `assertDictEqual` and `assertMultiLineEqual` put a header on the
    `AssertionError` line ("Items in the first set but not the second:") and
    the values that differ on the unindented lines below it, so a digest that
    stops at the exception says only that some set differed. The first few of
    those lines follow the verdict — for a case that only fails on the CI host,
    the differing value is what identifies which assertion in the test ran into
    the defect.
    """
    verdict = "(no traceback)"
    detail: list[str] = []
    armed = False
    capturing = False
    for line in lines[header + 1:]:
        stripped = line.strip()
        if stripped.startswith("====") or stripped.startswith("Ran "):
            break
        if stripped.startswith("Traceback ("):
            armed = True
            capturing = False
            continue
        # Continuation lines of a traceback are indented, its separator rules
        # are punctuation, and blank lines end nothing.
        if not stripped or line[:1].isspace() or set(stripped) <= {"-", "="}:
            continue
        if stripped.startswith("File "):
            continue
        if armed:
            verdict = stripped[:160]
            detail = []
            armed = False
            capturing = True
        elif capturing:
            if len(detail) < 4:
                detail.append(stripped[:80])
        elif verdict == "(no traceback)":
            # No banner opened this block — a bare `SkipTest` reason, say.
            verdict = stripped[:160]
    return " ".join([verdict, *detail])


def failure_digest(out: str, err: str) -> str:
    """Which cases unittest reported against, and its closing verdict.

    A FAIL's tail is not its cause: this runner sets `MAJIT_STATS`, so the last
    thing every process writes is the JIT summary, and `last_stderr_line` reads
    back `Compilation time: 8.1ms` for a module whose real answer is two named
    assertions several hundred lines above it. Every FAIL in the suite recorded
    that same line, which named no test at all.

    unittest writes `FAIL: <case>` / `ERROR: <case>` headers and one closing
    `FAILED (...)`; those are the runner's own account of what went wrong, and
    they are what a CI log needs to carry.

    The header names the case but not the cause, and a case that only fails on
    the CI host cannot be re-run locally to find out — so each header carries
    the closing line of its traceback (the exception type and message), which
    is the one line that says what actually went wrong.
    """
    lines = f"{out}\n{err}".splitlines()
    cases: list[str] = []
    verdict = ""
    for idx, line in enumerate(lines):
        line = line.strip()
        if line.startswith(("FAIL: ", "ERROR: ")):
            entry = f"{line} -> {traceback_verdict(lines, idx)}"
            if entry not in cases:
                cases.append(entry)
        elif line.startswith("FAILED ("):
            verdict = line
    shown = cases[:4]
    if len(cases) > len(shown):
        shown.append(f"(+{len(cases) - len(shown)} more)")
    return " | ".join(part for part in (verdict, *shown) if part)


# The Windows loader refusing to start a process, which is not the module
# crashing but the module never running.
STATUS_DLL_INIT_FAILED = 0xC0000142

# Windows kills a faulting process by setting its exit status to the NTSTATUS
# it died of, so these arrive where a POSIX signal number would.
NTSTATUS_NAMES = {
    0xC0000005: "ACCESS_VIOLATION",
    0xC0000017: "NO_MEMORY",
    0xC000001D: "ILLEGAL_INSTRUCTION",
    0xC0000094: "INTEGER_DIVIDE_BY_ZERO",
    0xC00000FD: "STACK_OVERFLOW",
    0xC000013A: "CONTROL_C_EXIT",
    0xC0000135: "DLL_NOT_FOUND",
    0xC0000139: "ENTRYPOINT_NOT_FOUND",
    STATUS_DLL_INIT_FAILED: "DLL_INIT_FAILED",
    0xC0000374: "HEAP_CORRUPTION",
    0xC0000409: "STACK_BUFFER_OVERRUN",
}


def death_signal(rc: int) -> str:
    """`SIGBUS` for a run killed by a signal, else a bare return code.

    POSIX `subprocess` reports a signal death as a negative return code; a
    shell-mediated one arrives as 128+n. Naming the signal is what separates
    "the interpreter faulted" from "the interpreter aborted an assertion",
    and those two want completely different investigations.

    Windows has no signal to name and reports the NTSTATUS instead, which
    reaches here as a ten-digit number that reads as nothing: three modules
    of a full run were filed as interpreter crashes on the strength of one,
    and it was the loader saying it had not started them.
    """
    if sys.platform == "win32":
        named = NTSTATUS_NAMES.get(rc)
        if named is not None:
            return named
        if rc >= 0xC0000000:
            return f"NTSTATUS 0x{rc:08X}"
    num = -rc if rc < 0 else rc - 128
    try:
        return signal.Signals(num).name
    except ValueError:
        return f"rc={rc}"


def unittest_tally(out: str, err: str) -> tuple[int, int] | None:
    """`(tests run, tests skipped)`, totalled over unittest's own summaries.

    `None` when the module wrote no summary at all.

    Every `Ran N tests` and the `skipped=N` of the verdict closing it are added
    up rather than the last summary winning: a test file may run more than one
    suite, and one that prints a captured child interpreter's output carries
    that child's summary too. Totalling them keeps the caller's "nothing ran"
    test conservative, since one suite that did run real tests outweighs
    another that skipped everything.
    """
    ran = skipped = 0
    counted = False
    for line in f"{out}\n{err}".splitlines():
        line = line.strip()
        if line.startswith("Ran ") and " test" in line:
            count = line.split()[1]
            if count.isdigit():
                ran += int(count)
                counted = True
        elif line.startswith(("OK (", "FAILED (")) and line.endswith(")"):
            for field in line[line.index("(") + 1:-1].split(", "):
                name, _, value = field.partition("=")
                if name == "skipped" and value.isdigit():
                    skipped += int(value)
    return (ran, skipped) if counted else None


def classify(rc: int, out: str, err: str) -> tuple[str, str]:
    """Map a finished run to (status, detail).

    The FAIL vs IMPORTERROR split keys on whether unittest actually ran:
    a `Ran N tests` / `FAILED (` line means the module imported and the test
    framework executed (a genuine test FAIL), while its absence means the
    module could not even be loaded — an interpreter/stdlib gap, which is the
    Phase-0 backlog this suite is meant to surface.

    The PASS vs SKIP split on a clean exit keys on whether anything was
    actually exercised: a run whose every case was skipped exits 0 and prints
    `OK`, and calling that PASS records a gate entry for a module that carries
    no signal. `test.test_interpreters` was recorded PASS while its five
    submodules were nothing but `unittest.loader` placeholders for a missing
    `_interpreters` (`loader.py` `_make_skipped_test` turns an import-time
    `SkipTest` into a skipped test rather than a failure), so the entry claimed
    coverage of a package that ran no test at all.
    """
    panic = jit_panic_reason(err)
    if panic:
        return "CRASH", panic
    if rc < 0 or rc > 128:
        # Keep the stderr tail. A signal death is precisely the case where no
        # test framework got to report anything, so the last line the
        # interpreter wrote is all the evidence there is — and dropping it
        # left CI runs that could only be re-run, never diagnosed.
        # The last line of a Rust abort is the backtrace note. Keep the
        # lines above it too; the location is not in that trailer, and a
        # 200-character cut dropped it on the only run that sees the crash.
        tail = [
            line.strip()
            for line in err.splitlines()
            if line.strip() and not line.lstrip().startswith("[jit-stats]")
        ]
        detail = f"signal/abort {death_signal(rc)} rc={rc} " + " || ".join(tail[-8:])
        return "CRASH", detail.strip()
    if rc == 0:
        denied = next(
            (
                line[len(RESOURCE_DENIED_MARKER):]
                for line in out.splitlines()
                if line.startswith(RESOURCE_DENIED_MARKER)
            ),
            None,
        )
        if denied is not None:
            return "SKIP", denied[:120]
        tally = unittest_tally(out, err)
        # No summary at all is the same absence of a result, one step earlier:
        # the process exited cleanly without unittest ever reporting, so there
        # is nothing to have passed. A module launched without a driver — an
        # `-m` run of a file with no `__main__` block — reaches exactly this,
        # and recording it PASS claims coverage the run never had.
        if tally is None:
            return "SKIP", "no unittest summary"
        # Both halves of `ran == skipped` are the same absence of a result:
        # every case opted out, or the suite was empty. libregrtest keeps a
        # state of its own for the second (`single.py:92` raises
        # `TestDidNotRun` on a run with no tests, no skips and no errors,
        # which `result.py` prints as "ran no tests").
        if tally[0] == tally[1]:
            ran = tally[0]
            return "SKIP", f"ran {ran}, every test skipped" if ran else "ran no tests"
        return "PASS", ""
    last = last_stderr_line(err)
    ran = "Ran " in out or "Ran " in err or "FAILED (" in out or "FAILED (" in err
    # A module that bails with unittest.SkipTest before any test runs is
    # opting out (a missing optional C extension or wrong platform) — CPython
    # skips it too, so it is honestly SKIP, not an interpreter gap.
    if not ran and "SkipTest" in err:
        return "SKIP", f"rc={rc} {last}"[:120]
    if ran:
        # An IMPORTERROR never reached unittest, so its tail is all there is;
        # a FAIL has unittest's own account, and only that names a test.
        # Wide enough for four cases that each now carry their exception line;
        # at 300 the digest was cut off inside the first case's name.
        return "FAIL", f"rc={rc} {failure_digest(out, err) or last}"[:900]
    return "IMPORTERROR", f"rc={rc} {last}"[:120]


# ── discovery ────────────────────────────────────────────────────────

def discover_modules(filter_substring: str | None) -> list[str]:
    """Dotted names (`test.test_xxx`) for every CPython test module/package."""
    names: list[str] = []
    for p in sorted(TESTDIR.glob("test_*.py")):
        names.append(f"test.{p.stem}")
    for d in sorted(TESTDIR.glob("test_*")):
        if d.is_dir() and (d / "__init__.py").exists():
            names.append(f"test.{d.name}")
    names = sorted(set(names))
    if filter_substring:
        names = [n for n in names if filter_substring in n]
    return names


# ── execution ────────────────────────────────────────────────────────

def module_path(module: str) -> Path:
    """Filesystem path to run a dotted `test.<name>` module as a script:
    a `test_xxx.py` file, or a package's `__main__.py`."""
    name = module.split(".", 1)[1]
    pyfile = TESTDIR / f"{name}.py"
    if pyfile.exists():
        return pyfile
    return TESTDIR / name / "__main__.py"


def is_package(module: str) -> bool:
    """A `test.<name>` whose `<name>` is a package directory, not a file."""
    name = module.split(".", 1)[1]
    return (TESTDIR / name / "__init__.py").exists()


# The guard as the language accepts it, not as it is usually typed:
# `test_coroutines.py:2527`, `test_posixpath.py:1169` and
# `test_robotparser.py:819` all spell it `if __name__=="__main__":` with no
# spaces, and a fixed-string test reads those three as guardless.
MAIN_GUARD = re.compile(r"^\s*if\s+__name__\s*==\s*(['\"])__main__\1\s*:", re.MULTILINE)


# Files whose `__main__` block is not their test entry: test_patma's runs a
# pyperf benchmark of the suite, so running the file imports `pyperf` and
# reports an import error instead of running its 328 tests.  libregrtest
# imports the module and builds the suite itself, which the dotted driver does.
BENCHMARK_MAIN_MODULES = {
    "test.test_patma",
}


def runs_its_own_tests(module: str) -> bool:
    """Whether running the module's file as `__main__` runs its tests.

    Script mode bets on the file's own `if __name__ == "__main__":
    unittest.main()` block. Not every test file carries one — libregrtest
    imports the module and builds the suite itself, so a file written only for
    that entry needs none — and running such a file directly defines its test
    classes, runs nothing, and exits 0. That is a clean exit which asserted
    nothing, and it was recorded as PASS: `test_type_params` alone collects
    over a hundred tests that never ran.
    """
    if module in BENCHMARK_MAIN_MODULES:
        return False
    try:
        source = module_path(module).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return True
    return MAIN_GUARD.search(source) is not None


# Driver that imports a test *package* with real package context (so its
# `__init__.py` / `load_tests` discovers the sub-suite) and runs it through
# unittest, emitting the `Ran N`/`OK`/`FAILED (` markers classify() keys on.
#
# The `if __name__ == "__main__"` guard is the "Safe importing of main module"
# idiom multiprocessing requires: a spawn-started child re-imports this file as
# `__mp_main__` (`multiprocessing/spawn.py:_fixup_main_from_path`), and without
# the guard it would re-run the whole suite inside the child. Every test file
# run directly carries the same guard.
_DOTTED_DRIVER = (
    "import sys, unittest\n"
    "mod = {module!r}\n"
    'if __name__ == "__main__":\n'
    "    __import__(mod)\n"
    "    unittest.main(module=sys.modules[mod], argv=['pyre'], verbosity=2)\n"
)

# Only the resource drivers below write this line, and `classify` honours it
# only from them. A test module that prints its own "skipped: ..." says nothing
# about the run's outcome, so the marker has to be one nothing else emits.
RESOURCE_DENIED_MARKER = "pyre-cpython-suite: ResourceDenied: "

# Same `__mp_main__` guard as _DOTTED_DRIVER.
_RESOURCE_DRIVER = (
    "import runpy\n"
    "from test import support\n"
    'if __name__ == "__main__":\n'
    "    support.use_resources = {{}}\n"
    "    try:\n"
    "        runpy.run_path({path!r}, run_name='__main__')\n"
    "    except support.ResourceDenied as exc:\n"
    "        print({marker!r} + str(exc))\n"
)

_RESOURCE_MODULE_DRIVER = (
    "import runpy\n"
    "from test import support\n"
    "mod = {module!r}\n"
    'if __name__ == "__main__":\n'
    "    support.use_resources = {{}}\n"
    "    try:\n"
    "        runpy.run_module(mod, run_name='__main__', alter_sys=True)\n"
    "    except support.ResourceDenied as exc:\n"
    "        print({marker!r} + str(exc))\n"
)

# test_descr and test_enum assert fully qualified class/enum names. Running
# their files as __main__ changes those names even on CPython, so preserve the
# dotted identity that libregrtest gives them.  test_regrtest and the thread
# suites repeatedly spawn child interpreters and import/filter their canonical
# `test.test_*` modules; running the parent as a second `__main__` module would
# test a different module identity from both libregrtest and those children.
#
# test_runpy's file as __main__ does not run test_runpy at all — it starts
# libregrtest over the whole suite (measured: 491 modules on CPython 3.14,
# 492 here), so script mode can only ever time out on it.
#
# Four 3.15 tests fail as __main__ on CPython 3.15.0 too and pass when imported
# as `test.<name>` (measured): test_complex's `test_constructor` re-emits a
# caught warning with `__main__`'s globals, and `warnings.warn_explicit` then
# asks the frozen-only loader of `importlib._bootstrap_external` for its
# source; test_zipimport_support's
# `test_import_filter_syntax_warnings_by_module` fails its
# `import test_pkg.test_mod` with "'test_pkg' is not a package" when the
# file's own directory is on `sys.path`, where `test/test_pkg.py` shadows the
# zip's package. test_extcall's and test_unpack_ex's doctests expect
# `test.test_extcall.f()`-style qualified names in their tracebacks, which name
# `__main__` when the file runs as a script.
DOTTED_IDENTITY_MODULES = {
    "test.test_complex",
    "test.test_descr",
    "test.test_enum",
    "test.test_extcall",
    "test.test_regrtest",
    "test.test_runpy",
    "test.test_thread",
    "test.test_threading",
    "test.test_unpack_ex",
    "test.test_zipimport_support",
}

# These suites create child interpreters or threads of their own.  Running them
# inside the module-level worker pool can exhaust process/thread resources and
# turn an otherwise passing module into a timeout or an ENOMEM child launch.
SERIAL_MODULES = {
    "test.test_regrtest",
    "test.test_thread",
    "test.test_threading",
    # `TestExecutablePrependedZip.test_execute_zip{2,64}` each launches the
    # pyre executable embedded behind a shell prefix.  On the Linux CI runner,
    # that extra interpreter was the fourth one beside `--jobs 3` and both
    # launches failed with ENOMEM.  The tests pass when their parent is the
    # only module worker, which is the same isolation this set gives the
    # subprocess-heavy suites above.
    "test.test_zipfile",
}

# Modules whose resource-gated tests are ruinous when test.support.use_resources
# is left as None: test_datetime's load_tests then appends an exhaustive test
# class for every installed system timezone, test_urllibnet performs live
# external HTTP requests, and test_zipimport's Zip64 arms build archives past
# 4 GiB, which cost 290s and over 6 GB of RSS.  libregrtest and PyPy's conftest
# use an empty default resource set; the driver also turns a module-level
# ResourceDenied into a clean skip.
DEFAULT_RESOURCE_MODULES = {
    "test.test_datetime",
    "test.test_urllibnet",
    # `EnsurePipTest.test_with_pip` is 115s of this module's 133s, nearly all
    # of it waiting on two pip bootstraps and two uninstalls in subprocesses.
    # `pyre/extra_tests/pip/run.py` covers that ground in the same CI job,
    # offline and in a fraction of the time.
    "test.test_venv",
    "test.test_zipimport",
    # `is_resource_enabled` treats `use_resources is None` as "every resource
    # is on". `DirectoryTestCase.test_files` then parse/unparses every
    # `Lib/*.py` and `Lib/test/*.py` (600+ files) instead of the 10-file
    # sample plus the grammar always-set. That arm is what `-u cpu` is for.
    "test.test_unparse",
    # Four `@requires_resource('cpu')` methods (float/Fraction sqrt rounding
    # and the NormalDist/kde suites) are the only ones that blow the 300s
    # module budget; the other ~400 cases finish in seconds.
    "test.test_statistics",
}


def build_cmd(binary: Path, module: str, mode: str) -> list[str]:
    if mode == "regrtest":
        # CPython libregrtest; <module> is the bare basename (test_xxx).
        return [str(binary), "-m", "test", "-v", module.split(".", 1)[1]]
    if mode == "module":
        return [str(binary), "-m", module]
    # script (default): run the file directly as __main__.
    return [str(binary), str(module_path(module))]


def build_test_extensions(binary: Path) -> dict[str, str]:
    """Compile the test extensions for `binary`; answer, per extension, why
    one is missing.

    An interpreter built without `cpyext`, a platform the feature is not
    compiled for, and a missing C compiler each leave every extension unbuilt.
    A source that fails on its own leaves the others usable, so its reason is
    reported alone. The tests that need one skip exactly as they do upstream
    when the module is absent, rather than failing the run.
    """
    if sys.platform not in ("darwin", "linux"):
        reason = f"cpyext is built for macos and linux, not {sys.platform}"
        return dict.fromkeys(TEST_EXTENSION_SOURCES, reason)
    named = subprocess.run(
        [str(binary), "-c", "import _imp; print(_imp.extension_suffixes()[0])"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    suffix = named.stdout.strip().splitlines()[-1] if named.stdout.strip() else ""
    if named.returncode != 0 or not suffix:
        reason = "the interpreter reported no extension suffix"
        return dict.fromkeys(TEST_EXTENSION_SOURCES, reason)
    unbuilt = {}
    for name, source in TEST_EXTENSION_SOURCES.items():
        reason = build_one_test_extension(binary, name, source, suffix)
        if reason:
            unbuilt[name] = reason
    return unbuilt


def build_one_test_extension(binary: Path, name: str, source: Path,
                             suffix: str) -> str:
    """Compile one extension and prove the interpreter can import it, or say
    why not."""
    if not source.is_file():
        return f"{source} is missing"
    compile_cmd = [
        os.environ.get("CC", "cc"), str(source),
        "-I", str(CPYEXT_INCLUDE), "-I", str(CPYEXT_INTERNAL_INCLUDE),
        "-o", str(EXTENSION_DIR / f"{name}{suffix}"), "-O2",
    ]
    compile_cmd += (
        ["-bundle", "-undefined", "dynamic_lookup"] if sys.platform == "darwin"
        else ["-fPIC", "-shared"]
    )
    built = subprocess.run(
        compile_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if built.returncode != 0:
        return f"{compile_cmd[0]}: {last_stderr_line(built.stderr)}"
    # Compiling proves the headers agree, not that the interpreter can load
    # what they describe: a build without cpyext resolves none of the symbols.
    # `-E` because that is the interpreter the tests reach it through, so the
    # check answers for the default `sys.path` rather than for this process's
    # environment.
    loaded = subprocess.run(
        [str(binary), "-E", "-c", f"import {name}"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if loaded.returncode != 0:
        return f"built but not importable: {last_stderr_line(loaded.stderr)}"
    return ""


def reap(proc: "subprocess.Popen") -> None:
    """End a timed-out module and everything it started.

    `Popen.kill` reaches the child alone.  A module that spawned
    `multiprocessing` workers leaves them alive holding the very stdout and
    stderr this runner reads, and on Windows a spawned worker holds them the
    same way under CPython as under pyre — measured both ways.  The drain that
    collects the partial output then never sees EOF, so a run that meant to
    record one TIMEOUT stops instead, for as long as the orphans live.
    `taskkill /T` walks the tree while the child still parents it.

    POSIX fills the exception's buffers before raising and needs the child
    reaped and nothing read, which is what `subprocess.run` does there too.
    """
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True)
    proc.kill()
    (proc.communicate if sys.platform == "win32" else proc.wait)()


def run_module(binary: Path, module: str, mode: str, timeout: int,
               env: dict) -> tuple[str, str]:
    # Run in a throwaway cwd so a test writing into '.' never touches the
    # repo (the stdlib is resolved relative to the executable, not cwd).
    # `ignore_cleanup_errors` because a test that leaves a file open takes the
    # whole run down on Windows, where an open handle blocks the unlink; the
    # directory is disposable, and losing one is not a test result.
    with tempfile.TemporaryDirectory(
        prefix="pyre-cpytest-", ignore_cleanup_errors=True
    ) as cwd:
        # A package has no `__main__.py` to run as a bare script (and running a
        # submodule file directly breaks its relative imports), and a file with
        # no `__main__` block runs nothing on EITHER entry — `pyre -m test.x`
        # imports it, defines its classes and exits 0 just as the bare script
        # does.  Both shapes get the synthesized unittest entry in script and
        # module mode alike; `regrtest` builds the suite itself and needs no
        # driver.  `DOTTED_IDENTITY_MODULES` stays script-only: it exists to
        # restore the dotted identity that `-m` already gives.
        if mode != "regrtest" and is_package(module):
            driver_name = "_pyre_pkg_main.py"
        elif mode != "regrtest" and not runs_its_own_tests(module):
            driver_name = "_pyre_dotted_main.py"
        elif mode == "script" and module in DOTTED_IDENTITY_MODULES:
            driver_name = "_pyre_dotted_main.py"
        else:
            driver_name = None
        if driver_name is not None:
            driver = Path(cwd) / driver_name
            driver.write_text(_DOTTED_DRIVER.format(module=module), encoding="utf-8")
            cmd = [str(binary), str(driver)]
        elif mode == "script" and module in DEFAULT_RESOURCE_MODULES:
            # Match libregrtest's runner-owned default resource policy while
            # retaining direct-file __main__ semantics.
            driver = Path(cwd) / "_pyre_script_main.py"
            driver.write_text(
                _RESOURCE_DRIVER.format(
                    path=str(module_path(module)),
                    marker=RESOURCE_DENIED_MARKER,
                ),
                encoding="utf-8",
            )
            cmd = [str(binary), str(driver)]
        elif mode == "module" and module in DEFAULT_RESOURCE_MODULES:
            # `pyre -m` would otherwise leave use_resources as None, which
            # means every optional resource is enabled. Preserve module entry
            # semantics while applying libregrtest's empty default resource set.
            driver = Path(cwd) / "_pyre_resource_module_main.py"
            driver.write_text(
                _RESOURCE_MODULE_DRIVER.format(
                    module=module,
                    marker=RESOURCE_DENIED_MARKER,
                ),
                encoding="utf-8",
            )
            cmd = [str(binary), str(driver)]
        else:
            cmd = build_cmd(binary, module, mode)
        module_env = env
        if module in STATS_STDERR_INCOMPATIBLE:
            if module_env is env:
                module_env = env.copy()
            module_env.pop("MAJIT_STATS", None)
        if mode == "script" and module == "test.test_regrtest":
            # libregrtest/setup.py `setup_process` keeps a non-ASCII
            # environment value present for every test process.  Script mode
            # deliberately bypasses that setup, but test_regrtest verifies the
            # runner-owned invariant itself, so reproduce the same environment
            # rather than grading the runner test outside its contract.
            if module_env is env:
                module_env = env.copy()
            module_env.setdefault("PYTHONREGRTEST_UNICODE_GUARD", "æ")
        relaunched = False
        while True:
            # `subprocess.run(timeout=)` cannot be used here: its own recovery
            # drains the pipes after killing the child alone, and `reap`
            # documents why that never returns for a module holding spawned
            # workers.  `with`, as `subprocess.run` has: one module of the four
            # hundred leaking its two pipe handles is one the run cannot
            # afford.
            with subprocess.Popen(
                cmd, cwd=cwd, env=module_env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace",
            ) as proc:
                try:
                    out, err = proc.communicate(timeout=timeout)
                except subprocess.TimeoutExpired as expired:
                    # A bare "timeout 120s" says only that the module is slow,
                    # never which case it stopped in — and unittest's progress
                    # dots, the one record of how far it got, were being
                    # thrown away here. `text=True` decodes what
                    # `communicate()` returns; the timeout path raises with the
                    # raw chunks it had joined, which is bytes on POSIX and str
                    # on Windows, so both spellings have to be accepted.
                    partial = expired.stderr or ""
                    if isinstance(partial, bytes):
                        partial = partial.decode("utf-8", "replace")
                    reap(proc)
                    return "TIMEOUT", f"timeout {timeout}s {last_stderr_line(partial)}"[:300]
            # The resource the loader ran out of is one this suite spends
            # itself, sixteen modules at a time, so the second launch of the
            # same command usually succeeds.  Nothing is hidden by trying: an
            # interpreter that genuinely cannot load fails twice.
            if relaunched or proc.returncode != STATUS_DLL_INIT_FAILED:
                break
            relaunched = True
    return classify(proc.returncode, out or "", err or "")


# ── baseline ─────────────────────────────────────────────────────────

def stdlib_version() -> str:
    try:
        text = STDLIB_VERSION_FILE.read_text(encoding="utf-8")
    except OSError:
        return "unknown"
    for token in text.split():
        if token.startswith("v") and token[1:2].isdigit():
            return token.lstrip("v")
    return "unknown"


def load_baseline(path: Path) -> dict:
    if not path.exists():
        return {"stdlib_version": stdlib_version(), "modules": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def host_baseline_path(path: Path) -> Path:
    """Per-host overlay beside the shared baseline, `baseline.<host>.json`.

    Same shape as the shared file and consulted first, so a host records only
    the modules it genuinely disagrees with.  The jit-stats baselines overlay
    on `sys.platform` alone; a verdict also has to separate the architectures
    within a platform, because the dynasm backend emits different machine code
    on each and a miscompile is not portable.
    """
    return path.with_name(f"{path.stem}.{HOST_TAG}{path.suffix}")


# Metadata keys stored beside per-backend verdicts in a module entry.
NON_BACKEND_ENTRY_KEYS = ("reason", "provenance")


def cell_sources(baseline: dict, overlay: dict) -> tuple[dict, dict]:
    """The two files a cell can come from, in the order they are consulted.

    One helper rather than the pair spelled at each site: `expected_cell` and
    `cell_provenance` have to agree about precedence, or a status read from the
    host overlay is graded against an axis read from the shared file.
    """
    return (overlay, baseline)


def expected_cell(baseline: dict, overlay: dict, module: str,
                  backend: str) -> tuple[str | None, str | None]:
    """Return a status and the backend that recorded it.

    The host overlay is consulted before the shared file, and a cell it holds
    wins — including when it answers by borrowing `dynasm`. An overlay row that
    holds no cell for this backend is not an answer, though: `write_baseline`
    mints a row carrying only the backend whose status diverged on this host,
    so a row exists for every module any backend ever diverged on. Treating
    that as an answer would drop the shared file's expectation for every OTHER
    backend, and a module recorded PASS there would stop being one — no longer
    selected by the gate, and no longer a regression when it fails.

    Missing backend cells fall back to dynasm; returning the source makes that
    substitution visible to selection and reporting.
    """
    for source in cell_sources(baseline, overlay):
        entry = source.get("modules", {}).get(module)
        if entry is None:
            continue
        own = entry.get(backend)
        if own:
            return own, backend
        borrowed = entry.get("dynasm")
        if borrowed:
            return borrowed, "dynasm"
    return None, None


def cell_provenance(baseline: dict, overlay: dict, module: str,
                    backend: str) -> dict | None:
    """Return the recorded measurement axis, or None for legacy cells.

    Reads the same source order as `expected_cell`, and falls through the same
    way, so the axis returned is the one under the status that lookup returned.
    Reading only the shared file here would pair an overlay host's status with
    the axis of a cell it overrode; stopping at a row that holds no cell for
    this backend would pair a shared-file status with no axis at all.
    """
    for source in cell_sources(baseline, overlay):
        entry = source.get("modules", {}).get(module)
        if entry is None:
            continue
        prov = entry.get("provenance", {}).get(backend)
        if prov is not None:
            return prov
    return None


def run_axis(args: argparse.Namespace, binary: Path) -> dict:
    """Return the configuration that gives this run's statuses meaning."""
    return {
        "backend": args.backend,
        # Name only, not the resolved path: the absolute path is host state and
        # would rewrite every cell on a different machine without any of them
        # having been re-measured.
        "binary": binary.name,
        "mode": args.mode,
        "jit": not args.no_jit,
        "stdlib_version": stdlib_version(),
    }


def axis_drift_report(baseline: dict, overlay: dict, selected: list[str],
                      axis: dict) -> list[str]:
    """Report axis drift for cells recorded by the requested backend.

    Borrowed cells are reported separately. Drift is informational because
    legacy cells do not carry provenance.
    """
    unrecorded = 0
    differing: dict[str, int] = {}
    compared = 0
    for m in selected:
        _status, src = expected_cell(baseline, overlay, m, axis["backend"])
        if src != axis["backend"]:
            continue
        prov = cell_provenance(baseline, overlay, m, src)
        if prov is None:
            unrecorded += 1
            continue
        compared += 1
        for key, value in axis.items():
            if key == "backend":
                continue
            if prov.get(key) != value:
                differing[key] = differing.get(key, 0) + 1
    own = compared + unrecorded
    lines = [f"{compared} of {own} own-backend cells carry a recorded axis, "
             f"{unrecorded} predate it"]
    for key in sorted(differing):
        lines.append(f"  axis drift: {differing[key]} of {compared} recorded "
                     f"under a different {key} (this run: {axis[key]!r})")
    return lines


# ── main ─────────────────────────────────────────────────────────────

def positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value!r}")
    return ivalue


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=("dynasm", "cranelift"), default="dynasm")
    ap.add_argument("--binary", help="explicit interpreter path (overrides --backend)")
    ap.add_argument("--no-jit", action="store_true", help="set PYRE_NO_JIT=1")
    ap.add_argument("--mode", choices=("script", "module", "regrtest"), default="script")
    ap.add_argument("--jobs", type=positive_int, default=os.cpu_count() or 4)
    ap.add_argument("--timeout", type=positive_int, default=300)
    ap.add_argument("--filter", help="only modules whose dotted name contains this")
    ap.add_argument("--list", action="store_true", help="list selected modules and exit")
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--update-baseline", action="store_true",
                    help="rewrite the baseline from observed results (non-gating)")
    ap.add_argument("--strict-baseline", action="store_true",
                    help="also fail on newly-passing modules not recorded as PASS")
    ap.add_argument("--full", action="store_true",
                    help="run every module and only report (never gate, never skip)")
    ap.add_argument("--report", type=Path,
                    help="write the per-module results as JSON to this path")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    # The report is box-drawn and the module names it echoes come from the
    # stdlib, so a console whose codepage cannot spell one of those characters
    # used to kill the run with a `UnicodeEncodeError` before it reached a
    # single test — the header alone does it on cp949.  The error paths below
    # spell host paths to stderr, which has the same codepage, so reconfigure
    # both streams.
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

    binary = Path(args.binary) if args.binary else TARGET_RELEASE / f"{BIN_NAME[args.backend]}{EXE}"
    binary = binary.resolve()
    if not args.list and not binary.exists():
        print(f"error: interpreter not found: {binary}", file=sys.stderr)
        features = args.backend
        if sys.platform in ("darwin", "linux"):
            features += ",cpyext"
        print("       build it with: cargo build --release -p pyrex --bin "
              f"{BIN_NAME[args.backend]} --no-default-features --features {features}",
              file=sys.stderr)
        return 2

    if not TESTDIR.is_dir():
        print(f"error: CPython test suite not found: {TESTDIR}", file=sys.stderr)
        return 2

    baseline = load_baseline(args.baseline)
    overlay_path = host_baseline_path(args.baseline)
    overlay = load_baseline(overlay_path) if overlay_path.exists() else {"modules": {}}
    modules = discover_modules(args.filter)
    # Derive once for reports, baseline updates, and drift checks.
    axis = run_axis(args, binary)

    if args.list:
        for m in modules:
            print(m)
        print(f"\n{len(modules)} modules", file=sys.stderr)
        return 0

    env = dict(os.environ)
    env["MAJIT_STRICT"] = "1"
    env["MAJIT_STATS"] = "1"
    # A SIGSEGV in a gated module (test_decimal's IBM invert) dies with no
    # Python traceback. `classify` keeps the stderr tail; this is what
    # writes the faulting PC into that tail.
    env.setdefault("PYTHONFAULTHANDLER", "1")
    # Several regression tests spawn sys.executable and assert the child's
    # stdout/stderr exactly. Keep the machine-readable panic counters on the
    # direct pyre process without leaking diagnostic output into its children.
    env["PYRE_MAJIT_STATS_ROOT_ONLY"] = "1"
    env.pop("PYRE_MAJIT_STATS_ANCESTOR", None)
    if args.no_jit:
        env["PYRE_NO_JIT"] = "1"

    unbuilt = build_test_extensions(binary)
    for name, reason in sorted(unbuilt.items()):
        print(f"warning: {name} not built ({reason}) — "
              f"{'/'.join(sorted(NEEDS_TEST_EXTENSION))} will skip the classes "
              "that need it", file=sys.stderr)

    # Decide which modules to actually run.
    #
    # The gate only protects modules the baseline records as PASS: a PASS that
    # regresses fails CI. In the plain gate run it therefore runs *only* those
    # PASS modules — running the whole suite at the per-module timeout overruns
    # the CI job budget, and the non-PASS modules carry no gate signal. The
    # exploratory lanes still see everything: `--full` reports the full suite,
    # `--update-baseline` re-records it, and `--strict-baseline` must observe
    # non-PASS modules to flag newly-passing ones.
    gate_pass_only = not (args.full or args.update_baseline or args.strict_baseline)
    to_run: list[str] = []
    skipped: list[str] = []
    off_platform: list[tuple[str, str]] = []
    deselected = 0
    # Selected modules whose expected status came from another backend.
    borrowed_gated: list[str] = []
    for m in modules:
        # Even full/update runs must not overwrite another platform's result.
        gate_reason = platform_gate(m)
        if gate_reason is not None:
            off_platform.append((m, gate_reason))
            skipped.append(m)
            continue
        exp, exp_src = expected_cell(baseline, overlay, m, args.backend)
        is_skip = (exp == "SKIP") or (m in KNOWN_SKIPS)
        if is_skip and not args.full and not args.update_baseline:
            skipped.append(m)
            continue
        if gate_pass_only and exp != "PASS":
            deselected += 1
            continue
        to_run.append(m)
        if exp_src is not None and exp_src != args.backend:
            borrowed_gated.append(m)

    # Print the same derived axis that baseline updates record.
    print(f"pyre CPython suite — backend={axis['backend']} mode={axis['mode']} "
          f"jit={'on' if axis['jit'] else 'off'} jobs={args.jobs}")
    print(f"binary: {binary}")
    overlay_count = len(overlay.get("modules", {}))
    print(f"baseline: {args.baseline.name} + "
          + (f"{overlay_path.name} ({overlay_count} host entries)"
             if overlay_count else f"no {HOST_TAG} overlay"))
    for m, reason in off_platform:
        print(f"  off-platform on {sys.platform}: {m} ({reason})")
    extra = f", {deselected} not gated (non-PASS)" if deselected else ""
    print(f"{len(to_run)} to run, {len(skipped)} skipped{extra}, "
          f"timeout={args.timeout}s")
    # Expose how much of this run was gated against another backend.
    print(f"{len(borrowed_gated)} of {len(to_run)} gated against another "
          f"backend's recorded status")
    for line in axis_drift_report(baseline, overlay, to_run, axis):
        print(line)
    print()

    results: dict[str, tuple[str, str]] = {}
    done = 0
    progress_on_line = 0
    running: dict[str, float] = {}
    announced_running: set[str] = set()
    io_lock = threading.Lock()
    stop_watch = threading.Event()

    def end_progress_line(with_counter: bool = False) -> None:
        """Close the current mark line. Callers hold `io_lock`."""
        nonlocal progress_on_line
        if with_counter:
            sys.stdout.write(f" {done}/{len(to_run)}\n")
            progress_on_line = 0
        elif progress_on_line:
            sys.stdout.write("\n")
            progress_on_line = 0

    def record_result(module: str, result: tuple[str, str]) -> None:
        nonlocal done, progress_on_line
        status, detail = result
        # Every member of STATUSES needs a mark, SKIP included: a module
        # that runs and bails with SkipTest is classified SKIP (:229),
        # which is a different event from a module deselected before the
        # run, and without its own mark it printed as "?" — the character
        # reserved for a status this table does not know.
        mark = {"PASS": "·", "FAIL": "F", "CRASH": "C", "TIMEOUT": "T",
                "IMPORTERROR": "i", "SKIP": "s"}.get(status, "?")
        # Name FAIL/CRASH/TIMEOUT/IMPORTERROR as soon as they are recorded so
        # a cancelled job log still identifies the module. End the current
        # mark line first, then continue marks on a fresh line. The
        # `done/total` counter stays on every 80th result, independent of
        # those breaks.
        named = status not in ("PASS", "SKIP")
        with io_lock:
            results[module] = (status, detail)
            done += 1
            sys.stdout.write(mark)
            progress_on_line += 1
            if done % 80 == 0:
                end_progress_line(with_counter=True)
            elif named:
                end_progress_line()
            if named:
                preview = detail.splitlines()[0].strip()[:200] if detail else ""
                sys.stdout.write(f"  {module}: {status}")
                if preview:
                    sys.stdout.write(f" {preview}")
                sys.stdout.write("\n")
            sys.stdout.flush()

    def run_tracked(module: str) -> tuple[str, str]:
        with io_lock:
            running[module] = time.monotonic()
        try:
            return run_module(binary, module, args.mode, args.timeout, env)
        finally:
            with io_lock:
                running.pop(module, None)

    def watch_still_running() -> None:
        while not stop_watch.wait(1.0):
            now = time.monotonic()
            with io_lock:
                overdue: list[tuple[str, int]] = []
                for module, started in running.items():
                    if module in announced_running:
                        continue
                    elapsed = now - started
                    if elapsed >= STILL_RUNNING_AFTER:
                        announced_running.add(module)
                        overdue.append((module, int(elapsed)))
                if not overdue:
                    continue
                end_progress_line()
                for module, elapsed in overdue:
                    sys.stdout.write(
                        f"  still running after {elapsed}s: {module}\n"
                    )
                sys.stdout.flush()

    watcher = threading.Thread(
        target=watch_still_running, name="still-running", daemon=True,
    )
    watcher.start()
    try:
        parallel_modules = [m for m in to_run if m not in SERIAL_MODULES]
        serial_modules = [m for m in to_run if m in SERIAL_MODULES]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futs = {
                pool.submit(run_tracked, m): m
                for m in parallel_modules
            }
            for fut in concurrent.futures.as_completed(futs):
                m = futs[fut]
                record_result(m, fut.result())
        for m in serial_modules:
            record_result(m, run_tracked(m))
    finally:
        stop_watch.set()
        watcher.join(timeout=2)
    print()

    # Summary by status.
    counts = {s: 0 for s in STATUSES}
    for status, _ in results.values():
        counts[status] = counts.get(status, 0) + 1
    # `+=`, not `=`: SKIP arrives from two places — modules deselected before
    # the run (`skipped`) and modules that ran and bailed with SkipTest
    # (counted from `results` just above). Assigning dropped the second set,
    # so the buckets did not sum to what was selected: a `--filter ctypes`
    # run reported "2 to run" over "?F" and a summary totalling 1.
    counts["SKIP"] += len(skipped)
    print("\n── summary ──")
    for s in STATUSES:
        print(f"  {s:12s} {counts[s]}")
    # Reconcile rather than trust: the buckets partition the selection, so
    # their sum is derivable and a disagreement means a status escaped the
    # table. Computed, not read off the counters that produced it.
    accounted = sum(counts[s] for s in STATUSES)
    expected = len(to_run) + len(skipped)
    if accounted != expected:
        print(f"  {'UNACCOUNTED':12s} {expected - accounted}  "
              f"(buckets sum to {accounted}, {expected} were selected)")
        # Nonzero, and before the analysis below. `check.py` reads this
        # runner's exit code and never this line, so printing alone hands it a
        # pass over a summary short by exactly this many modules. Returning
        # here also stops `--report` and `--update-baseline` from recording a
        # table that is missing a result.
        return 1

    # Regression / improvement analysis vs baseline.
    regressions: list[str] = []
    improvements: list[str] = []
    for m, (status, detail) in sorted(results.items()):
        exp, exp_src = expected_cell(baseline, overlay, m, args.backend)
        # Name the backend a verdict is measured against whenever it is not the
        # one under test. Without it a regression line reads as this backend
        # having gone from PASS to FAIL, when what happened may be that this
        # backend was never recorded passing.
        via = "" if exp_src in (None, args.backend) else f" (expected from {exp_src})"
        if exp == "PASS" and status != "PASS":
            regressions.append(f"{m}: PASS -> {status}{via}  {detail}")
        elif exp != "PASS" and status == "PASS":
            improvements.append(f"{m}: {exp or 'new'} -> PASS{via}")

    if improvements:
        print(f"\n── improvements ({len(improvements)}) ──")
        for line in improvements:
            print(f"  + {line}")

    if args.report:
        report = {
            **axis,
            "counts": counts,
            "modules": {m: {"status": s, "detail": d}
                        for m, (s, d) in sorted(results.items())},
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
        print(f"\nreport written: {args.report}")

    if args.update_baseline:
        # `axis`, not `args.backend`: the cells this writes record the axis
        # they were measured along, and `run_axis` is the one place that
        # derives it.
        written = write_baseline(args.baseline, baseline, overlay, results, axis)
        recorded = sum(1 for s, _ in results.values() if s == "PASS")
        for target in written:
            print(f"\nbaseline written: {target} ({recorded} PASS recorded)")
        # Read the file as written, not as loaded: this bless repairs some of
        # what these would otherwise name, and a report naming a cell the same
        # run just corrected is wrong the moment it prints.
        for line in phantom_row_report(baseline) + stale_curated_cell_report(baseline):
            print(line)
        return 0

    if regressions:
        print(f"\n── REGRESSIONS ({len(regressions)}) ──")
        for line in regressions:
            print(f"  - {line}")
        # `--full` is report-only (the weekly exploratory lanes rely on it);
        # never let a current PASS regression fail an exploratory run.
        if not args.full:
            return 1

    if args.strict_baseline and improvements and not args.full:
        print("\nstrict: newly-passing modules must be recorded "
              "(run --update-baseline)")
        return 1

    print("\nno regressions" if not regressions else "\n(--full: regressions reported, not gated)")
    return 0


def phantom_row_report(baseline: dict) -> list[str]:
    """Lines naming baseline rows that no longer have a discoverable module.

    Rows are reported rather than deleted because the caller may be running a
    filtered subset. Discovery is repeated without that filter here.
    """
    rows = baseline.get("modules", {})
    phantom = sorted(set(rows) - set(discover_modules(None)))
    if not phantom:
        return []
    return [f"{len(phantom)} of {len(rows)} baseline rows have no discoverable "
            f"module (carried unchanged, not gated):"] + [f"  ? {m}" for m in phantom]


def stale_curated_cell_report(baseline: dict) -> list[str]:
    """Lines naming cells on a curated-skip row that still record a measurement.

    `KNOWN_SKIPS` already governs selection, so these cells are inert. Preserve
    their last measured status in case the module leaves the curated skip set.
    """
    rows = baseline.get("modules", {})
    stale = [(m, backend, status)
             for m, entry in sorted(rows.items()) if m in KNOWN_SKIPS
             for backend, status in sorted(entry.items())
             if backend not in NON_BACKEND_ENTRY_KEYS and status != "SKIP"]
    if not stale:
        return []
    return [f"{len(stale)} cell(s) on {len({m for m, _, _ in stale})} curated-skip "
            f"row(s) still record a measurement (reported, not synced — the "
            f"curated table governs selection, so these are stale and inert):"] + [
        f"  * {m} [{backend}] {status}" for m, backend, status in stale]


def write_baseline(path: Path, baseline: dict, overlay: dict, results: dict,
                   axis: dict) -> list[Path]:
    """Record `results`, splitting them between the shared baseline and this
    host's overlay.  Returns the files actually written.

    A verdict the shared file has never seen establishes the shared answer, so
    whichever host records first sets it and no host is privileged.  After
    that a host writes to its own overlay only where it disagrees, and a run
    that comes back into agreement drops the overlay entry again -- so an
    overlay never outlives the divergence that created it.

    A cell's provenance travels with the status into whichever of the two files
    ends up holding it, so an overlay cell is a measurement with an axis under
    it rather than a bare status.
    """
    backend = axis["backend"]
    modules = baseline.setdefault("modules", {})
    overlay_modules = overlay.setdefault("modules", {})
    # Keep the legacy file-level value; per-cell provenance is authoritative for
    # a measurement's stdlib version.
    baseline["stdlib_version"] = stdlib_version()
    overlay_dirty = False
    # The containing cell already identifies the backend.
    cell_axis = {k: v for k, v in axis.items() if k != "backend"}
    for m, (status, _detail) in results.items():
        # Defensive: never overwrite another platform's recorded result.
        if platform_gate(m) is not None:
            continue
        # A curated KNOWN_SKIP stays SKIP regardless of what the run observed
        # (it is a "do not run" decision, not a result). Modules absent from
        # `results` (phantom skips that no longer exist) are simply not added.
        # It is a decision about the module rather than about this host, so it
        # is shared and never overlaid.
        if m in KNOWN_SKIPS:
            entry = modules.setdefault(m, {})
            entry[backend] = "SKIP"
            # Curated reasons are row-wide and refreshed from their source of
            # truth. Warn before replacing a different recorded rationale.
            recorded = entry.get("reason")
            if recorded is not None and recorded != KNOWN_SKIPS[m]:
                # Leading blank line: the improvements block prints immediately
                # above, and an unseparated `  ! ` line reads as one of its rows.
                print(f"\n  ! {m}: recorded `reason` replaced by the curated text")
                print(f"      was    : {recorded}")
                print(f"      curated: {KNOWN_SKIPS[m]}")
            entry["reason"] = KNOWN_SKIPS[m]
            # A curated skip is a decision, not a measurement, so it has no
            # measurement provenance.
            prov = entry.get("provenance")
            if prov is not None:
                prov.pop(backend, None)
                if not prov:
                    del entry["provenance"]
            # The skip is a decision about the module rather than about this
            # host, so it is shared and any host entry for it is dropped.
            overlay_dirty |= overlay_modules.pop(m, None) is not None
            continue
        shared = modules.get(m, {}).get(backend)
        if shared is None:
            entry = modules.setdefault(m, {})
            entry[backend] = status
            entry.setdefault("provenance", {})[backend] = dict(cell_axis)
            continue
        if status == shared:
            host_entry = overlay_modules.get(m)
            if host_entry is not None and host_entry.pop(backend, None) is not None:
                overlay_dirty = True
                # The axis goes with the cell it described. Left behind it would
                # be provenance for a status this file no longer records.
                host_prov = host_entry.get("provenance")
                if host_prov is not None:
                    host_prov.pop(backend, None)
                    if not host_prov:
                        del host_entry["provenance"]
                # Neither `reason` nor a leftover `provenance` is a verdict, so
                # neither can keep the entry alive once no backend still records
                # one. Spelled through the constant rather than as a literal:
                # this is the second site walking an entry generically, and two
                # spellings of "which keys are not backends" is how they drift.
                if not any(k not in NON_BACKEND_ENTRY_KEYS for k in host_entry):
                    del overlay_modules[m]
            continue
        host_entry = overlay_modules.setdefault(m, {})
        if host_entry.get(backend) != status:
            host_entry[backend] = status
            host_entry.setdefault("provenance", {})[backend] = dict(cell_axis)
            overlay_dirty = True

    written = [path]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    overlay_path = host_baseline_path(path)
    if overlay_dirty or overlay_path.exists():
        if overlay_modules:
            overlay["host"] = HOST_TAG
            overlay["stdlib_version"] = stdlib_version()
            overlay_path.write_text(
                json.dumps(overlay, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
            written.append(overlay_path)
        elif overlay_path.exists():
            overlay_path.unlink()
            written.append(overlay_path)
    return written


if __name__ == "__main__":
    sys.exit(main())
