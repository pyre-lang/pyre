#!/usr/bin/env python3
"""pyre pre-merge check: correctness + regression guard + comparison

Cross-platform Python translation of pyre/check.sh.
"""

import argparse
import ast
import difflib
import hashlib
import math
import os
import re
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import oracle_cache

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

EXE = ".exe" if sys.platform == "win32" else ""
# pyre's compat target is CPython 3.15; its native modules (`_sre.MAGIC`) and the
# vendored `lib-python/3` are coupled to that version. Take a version-matched
# oracle or none at all: a stale `python3` on PATH (an older system CPython)
# diverges from pypy on version-sensitive error text and trips a spurious
# cpython-vs-pypy baseline mismatch, and its timings gate the ratios.
CPYTHON_TARGET = (3, 15)


def _probe_interpreter(command):
    """What the interpreter reports as its version and its own path.

    `None` when the command did not run at all, which on Windows is what a
    `python3.15` naming an extensionless shim rather than an executable does —
    `shutil.which` finds it and `CreateProcess` cannot start it.
    """
    probe = "import sys; print(sys.version_info[0], sys.version_info[1]); print(sys.executable)"
    try:
        proc = subprocess.run(
            [command, "-c", probe], capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    lines = (proc.stdout or "").splitlines()
    if len(lines) < 2:
        return None
    try:
        major, minor = lines[0].split()
    except ValueError:
        return None
    return (int(major), int(minor)), lines[1].strip() or command


def _resolve_python3():
    """The oracle interpreter, as the absolute path it reports for itself.

    A bare name would be resolved against the PATH of whichever environment
    spawns it, and the timed runs hand the child a curated one.
    """
    named = os.environ.get("PYRE_CHECK_PYTHON3")
    candidates = [named] if named else ["python3.15", "python3", "python"]
    rejected = []
    for candidate in candidates:
        if named is None and shutil.which(candidate) is None:
            continue
        probed = _probe_interpreter(candidate)
        if probed is None:
            rejected.append(f"  {candidate}: did not run")
            continue
        version, executable = probed
        if version == CPYTHON_TARGET:
            return executable
        rejected.append("  %s: %d.%d" % (candidate, *version))
    wanted = "%d.%d" % CPYTHON_TARGET
    raise SystemExit(
        f"no CPython {wanted} to measure against — pyre targets it, and an "
        f"older one disagrees on version-sensitive behaviour and timings.\n"
        + ("\n".join(rejected) or "  (no candidate on PATH)")
        + "\nName one with PYRE_CHECK_PYTHON3."
    )


_PYTHON3 = None


def python3():
    """The oracle interpreter, resolved once and remembered.

    Resolved on first use rather than at import, because `_resolve_python3`
    exits the process when no CPython 3.15 is on PATH -- and `--check-headers`
    reads fixture files, never runs an oracle, and runs on a job that has no
    reason to carry one.
    """
    global _PYTHON3
    if _PYTHON3 is None:
        _PYTHON3 = _resolve_python3()
    return _PYTHON3

PYPY3 = os.environ.get("PYRE_CHECK_PYPY3") or (
    "pypy3" if shutil.which("pypy3") else "pypy"
)


def _detect_pyre_stdlib():
    """Stdlib directory to pin pyre to.

    pyre's native modules (`_sre`, ...) are coupled to one CPython version
    via `_sre.MAGIC`; the vendored `lib-python/3` is the version-matched
    copy. pyre's own `detect_stdlib_path` already resolves it through
    `find_intree_stdlib`, but a host `python3` whose `re`/`_sre` MAGIC
    disagrees (e.g. an older CPython on the dev PATH, or a PyPy that
    shadows CPython on the CI PATH) would mismatch if reached. Pin the
    vendored copy explicitly so the result is independent of PATH.
    The host `python3` stdlib is only a last resort for an out-of-tree run.
    """
    intree = Path(__file__).resolve().parent.parent / "lib-python" / "3"
    if intree.is_dir():
        return str(intree)
    try:
        proc = subprocess.run(
            [python3(), "-c", "import sysconfig; print(sysconfig.get_paths()['stdlib'])"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    path = (proc.stdout or "").strip()
    return path if path and os.path.isdir(path) else None


_UNSET = object()
_PYRE_STDLIB = _UNSET


def pyre_stdlib():
    """[`_detect_pyre_stdlib`]'s answer, computed once and remembered.

    Deferred with [`python3`]: the out-of-tree fallback spawns the oracle.
    """
    global _PYRE_STDLIB
    if _PYRE_STDLIB is _UNSET:
        _PYRE_STDLIB = _detect_pyre_stdlib()
    return _PYRE_STDLIB


# Which wasm runtime the `pyre-wasm-runner` uses (`--wasm-engine`). wasmtime
# (cranelift) is fast in steady state but recompiles the ~14MB module on every
# process start; wasmi is a pure-Rust interpreter with near-zero startup cost
# but slower hot loops. Forwarded to the runner via PYRE_WASM_ENGINE; ignored
# by the dynasm/cranelift backends.
WASM_ENGINE = "wasmtime"

# The wasm backend has no perf-ratio gate (its run_bench vs_cpython/vs_pypy are
# None) — its per-bench timeout is only a hang guard. wasm legitimately runs a
# few× slower than the native backends the base timeouts were tuned for, and
# slower still on loaded CI runners, so give it extra headroom by default to
# avoid flaky timeouts. Overridable with --wasm-timeout-scale.
WASM_TIMEOUT_SCALE = 4.0
# Ceiling on wasm's execution-only user-CPU as a multiple of dynasm's on the
# same fixture, both startup-subtracted. Unlike the vs-cpython / vs-pypy gates
# this baseline is not a per-bench tuned constant: it is dynasm's own time from
# the same invocation, so the pair sees one host under one load and there is no
# recorded number that can age. Gated only when dynasm actually ran the fixture
# in this run and its execution-only time clears FLOOR_GATE_MIN_BASELINE_S;
# below that the denominator is startup noise, and the comparison table marks
# the ratio `~` the way the pypy gates already do.
#
# The number is meant to be a target the backend is held to, so it moves only
# when the backend does, and never to bless a fixture: it started at 3x, went to
# 4x when 3x was collecting per-fixture `max-wasm-ratio` allowances one at a
# time (fannkuch, fib_recursive, raise_catch_loop,
# short_circuit_value_kept_stack) rather than measuring anything, and came back
# to 3.5 once every one of those allowances had been removed and the fixtures
# that still exceeded it -- fib_recursive and loop_callee_shared_mutation, both
# bound by CALL_ASSEMBLER returns -- were fixed rather than exempted. It then
# went 3.5 -> 3.8 -> 4.0 chasing str_getitem_len_hot, which measured 4.1x on
# ubuntu and stayed red under each wider ceiling anyway; that fixture reads
# 2.9x now that its bridge has stopped crossing modules once per iteration, and
# its allowance is gone, so it is not what holds the ceiling up any more.
#
# What does is the rest of the census, and it is worth stating because the
# ceiling has now been widened twice by a fixture that was never going to be
# caught by widening it. Two ubuntu runs of one base gate 299 fixtures between
# them, and the worst is foriter_make_function_body at 3.74x, then
# short_circuit_boxed_int_cross_fn 3.0x/3.5x and pickle_terminal_raise_resume
# steady at 3.4x.
#
# Those were all base-gated readings at the time: this paragraph used to except
# builtin_folds_hot and math_folds_hot as allowance-holders, then no fixture in
# the tree carried a `max-wasm-ratio` header. The ratios are from a base that
# has since moved -- one later ubuntu run of 249 gated fixtures read
# pickle_terminal_raise_resume worst at 3.26x with foriter_make_function_body
# absent from the table, which under the union rule below is a reason to census
# again, not a refutation of 3.74x.
#
# The Function.call_args parity port deliberately changed that census: Python
# dunders that the old shortcut kept on the plain evaluator now enter the
# recursive portal just as PyCode.funcrun -> PyFrame.run does in PyPy.  The
# float_subclass_binop_dispatch and pickle_terminal_raise_resume fixtures carry
# fitted allowances for the resulting wasm module-compilation tax.  Their
# headers record both oracle topology and the measurements; removing the new
# traces to regain this base ceiling would restore the very deviation the port
# fixed.
#
# Under the highest-observed-plus-15% rule the allowances are fitted with, that
# 3.74x asks for 4.30x -- ABOVE this constant. 4.0 is therefore not slack: it
# leaves its worst gated fixture 7% of headroom where the rule wants 15%, and
# short_circuit_boxed_int_cross_fn's own 0.49x swing between those two runs is
# the scale that headroom is being measured against. ⛔That is a reason to fix
# or exempt foriter_make_function_body, NOT to raise the constant to 4.3: the
# ceiling has been widened for a fixture twice already and stayed red both
# times.
#
# Census the union of runs, not their intersection. A fixture whose jit-stats
# gate fails prints SNAPDIFF where its time and ratio go, so it leaves the table
# of whichever run it failed in -- which is exactly how foriter_make_function_body
# stayed out of the first version of this paragraph, and 3.5 looked reachable.
WASM_MAX_DYNASM_RATIO = 4.0
# The headroom every `max-wasm-ratio` allowance is fitted with, and the same
# margin the summary requires before it will call one outgrown -- a fixture
# reading just under the ceiling is not one that can lose its allowance.
WASM_RATIO_FIT_HEADROOM = 1.15
# Native Windows CI can spend substantially more wall time than reported
# process user-CPU while antivirus and concurrent matrix jobs contend for the
# runner.  Keep the timeout as a hang guard by granting native backends 2x
# headroom; performance still gates on measured user-CPU below.
WIN_NATIVE_TIMEOUT_SCALE = 2.0

BENCH_DIR = "pyre/bench"
SYNTHETIC_BENCH_DIR = "pyre/bench/synth"
CPYTHON_SUITE_BASELINE = "pyre/cpython_tests/baseline.json"
# Per module, matching the CI job. `test.test_asyncio` alone needs ~2m47s of
# wall time, so a smaller per-module limit turns a slow module into a fake
# regression.
CPYTHON_SUITE_MODULE_TIMEOUT_S = 300
CPYTHON_SUITE_TIMEOUT_S = 2400
SNAP_DIR = "pyre/check.snap"
BENCH_COMPARE_BUFFER_S = 0.005
# Windows process CPU accounting advances in scheduler ticks (normally 1/64 s).
# Keep the execution floor at least one observable tick on that platform.
WIN_TIMER_QUANTUM_S = 1.0 / 64
# Empty-program user-CPU startup is MEASURED (mean of STARTUP_SAMPLES runs)
# and subtracted from every timed run before ratios and perf gates are
# computed, so a fixed per-process cost does not inflate the ratio/gate of a
# short bench. Never hardcode the startup.
# Floored so a bench at/below its own startup (or noise) cannot drive a time
# to <= 0 and blow up a ratio.
#
# It is measured per BASELINE interpreter and not per pyre backend: every pyre
# backend subtracts pypy's. A pyre-specific estimate puts a second independent
# error in the numerator of a ratio whose denominator already carries one,
# while a shared subtrahend is a common term whose error largely cancels. It
# also keeps the subtraction from forgiving pyre the startup it owes -- what a
# pyre process spends above pypy to reach the first bytecode is a gap pyre has
# to close, not one to net out -- and it makes the wasm-vs-dynasm ratio a
# comparison of two whole processes rather than of two differently-estimated
# remainders.
#
# The startup is a divisor for every short bench: the baseline's exec time is
# `bench - startup`, so a startup estimate that lands high collapses that
# denominator and inflates every ratio computed against it.  A CI runner
# measured pypy startup at 0.031s where the same job on the same platform had
# measured 0.013s, and three unrelated benches failed their gates in that run
# while their pyre exec times had gone *down*.  That reading is the one the
# estimator below is chosen against: 0.031s is what a loaded sampling window
# reports, and the pass that carried it also saw 0.013s.
#
# The estimator is the MEAN, and the reason is that on one platform the samples
# are quantised far more coarsely than the quantity they measure.  Windows
# charges process CPU by sampling at WIN_TIMER_QUANTUM_S, so a single reading is
# a count of ticks the process happened to be caught running for -- an unbiased
# estimate of the true cost, but one that can only take lattice values.
# Measured here over 40 empty-program runs, pypy read 0.000s 31 times, 0.016s 8
# times and 0.031s once: mean 0.0039s against a wall-clock mean of 0.026s.  The
# cost is about four milliseconds, and NO sample ever reports that.  A median
# picks a lattice point and so reads 0.016s, four times the cost; across 47 CI
# sampling passes it reads two ticks, 0.031s, in 19 of them.  A minimum picks
# the other lattice point and reads 0.000s.  Averaging is what recovers the
# value between them, and it is the only one of the three that converges on the
# quantity as samples are added.
#
# Off windows the reading is not quantised and the three estimators agree to
# within a millisecond -- ubuntu 0.012s median against a 0.009s minimum, macos
# 0.016s against 0.014s -- so this changes almost nothing there.  It is a
# correction to one platform's instrument, not a policy about how much to
# subtract, which is why it is not a widening: on the platform it moves, the
# value it replaces was four times the cost it claimed to be.
#
# What rides on it is the denominator.  A startup estimate is subtracted from a
# baseline whose whole execution time is often the same size: measured across
# ten CI runs the median `exec` for pypy is 8ms on ubuntu, 10ms on macos and
# 16ms on windows.  An estimate that lands high collapses that denominator onto
# EXEC_TIME_FLOOR_S, and a baseline pinned there is not a measurement --
# `failed_bound` declines BOTH bounds on it, so the fixture's recorded ceiling
# goes unapplied.  Over-subtraction does not make the gate stricter; it makes
# the gate absent.
#
# The mean is the one estimator of the three that a single slow sample can move,
# which is what STARTUP_SAMPLES answers: the sampling error of a mean of N falls
# as sqrt(N), and each pass prints the range it was drawn from so an outlier is
# visible rather than inferred.
#
# This supersedes the MINIMUM, taken here briefly, and the MEDIAN recorded
# through `bc67f721c42`.  That commit's counter-example -- nine short benches
# carried over their gates by an estimate 15ms under the median -- described
# benches gated on a baseline pinned to EXEC_TIME_FLOOR_S, and `6a098e07ce2`
# later made a pinned baseline decline both bounds, so it no longer describes
# this code.  What survives of it is that an estimate from an idle window is
# subtracted from benches that were not idle; the mean is drawn from the same
# window as the median was, so that argument does not distinguish them.
#
# That last condition is what STARTUP_REFRESH_S exists to meet.  A single
# upfront measurement is subtracted from benches that run for the rest of the
# job, so it can come from a load window those benches never saw: one macos job
# read every interpreter's startup at about twice the neighbouring job's
# (dynasm 0.151s against 0.066s) while its raw bench times moved by a fifth of
# that, which lands whole in `exec` and collapsed a 0.074s exec to 0.009s.
# Re-measuring on a timer keeps the subtrahend within one interval of the run
# it is taken from, and every pass prints the samples it was drawn from
# ([min..max]) so the drift is readable in the log rather than inferred from a
# neighbouring job.
STARTUP_SAMPLES = 7
# Re-measured no more often than this, and only at a fixture boundary: the
# pairing argument above requires a fixture's numerator and denominator to be
# reduced by the SAME estimate, which a refresh between one fixture's backends
# would break.
#
# Sized against the phase it has to resolve, which is NOT the job's wall time:
# of a ~50min ubuntu job, the part between the startup line and the last
# comparison is 592s (macos 402s, windows 661s) -- the rest is cargo.  Over
# that, 300s bought 1 refresh on ubuntu and macos; 120s buys 4/3/5, and each
# pass costs one empty-program spawn per interpreter per sample, which the
# pass's own line reports so this can be re-sized from a measurement rather
# than an estimate.  Lengthening it trades that resolution back for an estimate
# allowed to drift further from what it is subtracted from.
STARTUP_REFRESH_S = 120
EXEC_TIME_FLOOR_S = WIN_TIMER_QUANTUM_S if sys.platform == "win32" else 0.005
# `exec` is `bench - startup`, and startup is a measured mean rather than a
# constant, so whatever the estimate is off by lands whole in `exec`.  That
# error is real, but NO allowance is granted for it, and a standing allowance
# was tried and removed.  It read `0.5 * startup`, added to the baseline on
# every recorded ceiling and to the measured side on every floor.
#
# It was withdrawn because its evidence did not survive being checked against
# the 234 job logs carrying a startup line across 125 CI runs.  Consecutive
# same-platform movement |delta|/max has median 0.125 and p90 0.45, putting 0.5
# near p95 rather than mid-range; on ubuntu and macos only 6 of 432 pairs reach
# it (ubuntu dynasm median 0.07, macos dynasm 0.14).  What does reach it is
# windows, where every reading is an exact multiple of WIN_TIMER_QUANTUM_S --
# cpython reads only 0.000s or 0.016s, so its pairs move by a full 1.0 as pure
# quantization, which `_compare_buffer` already pays for separately there.  The
# four pairs the number was fitted on turned out to be two macos lines from
# different branches four hours apart, with ten macos runs in between reading
# 0.130 0.077 0.069 0.076 0.072 0.076 0.118 0.092 0.079 0.106 for dynasm: they
# bracket the range rather than measuring an adjacent step.
#
# It was also the wrong shape.  It multiplied the READING, so it grew where the
# startup was largest rather than where it was least certain, and vanished
# exactly where the clock is coarsest -- windows cpython measured 0.000s in 17
# of 56 runs, yielding no allowance at all.  Widening the floor was worse than
# widening the ceiling: the floor is what reports a fixture that got FASTER
# than its recorded ceiling, so an allowance there dampens the signal a
# performance change is supposed to produce.  And it was the only gate
# adjustment the comparison table could not show, because `_ratio` never
# applied it.
#
# What replaces it is not a threshold.  A failing gate reports how far the
# startup estimate would have to be wrong to erase the failure, in units of the
# startup it subtracted (`_gate_fail_detail`), so a red carries its own
# measurement context and a small one can be told from a real regression
# without a constant deciding in advance.
# A floor failure is only trustworthy when the baseline clears the execution
# floor enough for small relative error: execution time is the difference
# between two independently measured values.
#
# Three times the floor is not enough of a clearance to fail a bench on.  At
# that size the baseline is 3 scheduler ticks of Windows CPU accounting, each
# side of the ratio carries +-1 tick, and a "you got fast" verdict is being
# read off a number with a third of its own magnitude in quantization error.
# One CI run failed `comprehension_accumulators` on windows for reading 0.3x
# against a pypy baseline of 0.05s -- 3.2 ticks -- while the same fixture read
# 1.6x and 2.5x on macos.  Ten times the floor holds that error near a tenth,
# and it leaves the gate armed on 24 fixtures whose pypy time is real work.
FLOOR_GATE_MIN_BASELINE_S = 10 * EXEC_TIME_FLOOR_S
# The pypy performance floor is derived from the ceiling and is never stated
# per bench. It is one fourth of the ceiling, and only when that fourth is at
# least 0.75: a ceiling of 20 keeps the reading in 5x..20x, a ceiling of 3
# keeps it in 0.75x..3x. A ceiling under 3 has no floor, so a reading faster
# than pypy is not a failure. A wider runner-to-runner spread is a measurement
# or fixture defect to investigate, not slack for this global policy to absorb.
PERF_GATE_FLOOR_RATIO = 0.75
PERF_GATE_FLOOR_DIVISOR = 4
# A single slow sample is retried before failing a performance gate. Windows
# needs more samples because its process CPU accounting is scheduler-tick
# quantized (see WIN_TIMER_QUANTUM_S above).
PERF_RETRY_RUNS = 5 if sys.platform == "win32" else 3
SYNTHETIC_CPYTHON_REFERENCE_TIMEOUT_S = 5
CARGO_CONFIG = {
    "dynasm": {
        # `cpyext`, because the CPython suite step that rides this backend's
        # Linux leg builds `_testbuffer`, `_testsinglephase` and
        # `_testmultiphase` against the binary built here: without the feature
        # `test_buffer` skips `TestBufferProtocol` wholesale, which is 95 of
        # that module's cases, and `test_importlib.extension` skips too.
        # `pyre-module` is a pyrex default that `--no-default-features` drops;
        # without it the binary lacks the builtin modules the product ships.
        # `mimalloc` is a default for the same reason and drops the same way,
        # and it is the global allocator every frame allocation goes through
        # (`FrameBox::new` -> `external_malloc(alloc_young=True)`): leaving it
        # out measured a no-argument Python call at 541 ns against 381 ns with
        # it, so a binary without it gates a slower interpreter than the one
        # the release ships.  `scripts/build-jit-core.sh` already passes it.
        "extra": [
            "--no-default-features",
            "--features",
            "dynasm,cpyext,pyre-module,mimalloc",
        ],
        "bin": "pyre-dynasm",
    },
    "cranelift": {
        # The core: no `pyre-module`, so its build neither compiles nor
        # translates the builtin modules that crate owns.  `mimalloc` for the
        # reason the dynasm leg gives, and so both legs measure one allocator.
        "extra": ["--no-default-features", "--features", "cranelift,mimalloc"],
        "bin": "pyre-cranelift",
    },
    # The wasm backend is not a `pyrex` binary: it is the wasm32 build of
    # `pyre-wasm` (full interpreter+JIT) executed by the native `pyre-wasm-runner`
    # under wasmtime. `build_backend` special-cases it (see `wasm=True`).
    "wasm": {
        "bin": "pyre-wasm-runner",
        "wasm": True,
    },
}

# Module + linker flags for the wasm32 build of `pyre-wasm`:
#   * `--export-table` exposes `__indirect_function_table` so the runner's
#     `jit_call` trampoline can dispatch residual calls by table index;
#   * `getrandom_backend="custom"` selects getrandom's custom backend (see
#     `pyre-wasm/src/lib.rs`) so the module carries no wasm-bindgen imports.
# `pyre-wasm` builds to the same `pyre_wasm.wasm` filename for both the `web`
# and `wasm-host` features, so a later build of the other flavour would clobber
# the native-host module. Copy the wasm-host build to a distinct, stable path the
# runner reads, immune to that overwrite. `pyre/pyre-wasm/build-web.sh` does the
# mirror image for the web flavour (snapshot -> pyre_wasm.web.wasm, fed to
# wasm-bindgen).
WASM_BUILD_OUTPUT = "target/wasm32-unknown-unknown/release/pyre_wasm.wasm"
WASM_MODULE_PATH = "target/wasm32-unknown-unknown/release/pyre_wasm.wasm-host.wasm"
# The JIT's trace-abort signal (InvalidLoop / speculative-fold failure) is
# propagated as a `Result`/deferred flag through the optimizer rather than a
# panic, so the build needs neither unwinding nor `-Z build-std`: it runs on the
# precompiled wasm32 std with the default `panic=abort`, on the stable toolchain.
# `--export-table` exposes the indirect-call table the runner patches for JIT
# re-entry; `--growable-table` drops its fixed maximum so the host can append
# compiled trace functions for inter-trace call_indirect chaining;
# `getrandom_backend="custom"` selects the no-import getrandom backend.
WASM_RUSTFLAGS = (
    '-C link-arg=--export-table -C link-arg=--growable-table '
    '--cfg getrandom_backend="custom"'
)
# Stable toolchain, no build-std. Kept as a (possibly empty) arg list so the
# build invocation can splat it uniformly.
WASM_CARGO_TOOLCHAIN = []
WASM_BUILD_STD_FLAGS = []

# ── ANSI helpers ─────────────────────────────────────────────────────

def red(s):    return f"\033[31m{s}\033[0m"
def yellow(s): return f"\033[33m{s}\033[0m"
def green(s):  return f"\033[32m{s}\033[0m"
def dim(s):    return f"\033[2m{s}\033[0m"
def bold(s):   return f"\033[1m{s}\033[0m"

# ── cargo ────────────────────────────────────────────────────────────

def cargo_finished_line(proc):
    """cargo's `Finished` line, or a stand-in when it printed none.

    Not the last line of the output: cargo ends with its future-incompat
    `note:`, which says nothing about the build that just ran. `Finished` is
    the line that names the profile cargo actually produced.
    """
    lines = (proc.stdout or "").splitlines() + (proc.stderr or "").splitlines()
    return next(
        (line.strip() for line in reversed(lines) if line.strip().startswith("Finished")),
        "cargo build done",
    )

# ── Child-process user CPU time and peak RSS ─────────────────────────

# Peak RSS in MiB of the most recent `run_timed` child, or None when the run
# timed out or the platform cannot report it. A single slot rather than a
# return-tuple member: `run_timed`'s 4-tuple has many callers and only the
# memory gate reads this. Runs are sequential, so the slot is unambiguous.
_LAST_RUN_MAXRSS_MB = [None]


def last_run_peak_rss_mb():
    """Peak RSS (MiB) of the child `run_timed` most recently reaped."""
    return _LAST_RUN_MAXRSS_MB[0]



def _run_timed_unix(args, timeout_s, env=None, keep_output=False):
    # `os.wait4` rather than `getrusage(RUSAGE_CHILDREN)` around
    # `subprocess.run`: `ru_maxrss` on RUSAGE_CHILDREN is a running maximum
    # over every reaped child, not a counter, so a before/after difference
    # attributes nothing to this run. wait4 reports the rusage of this child
    # alone, which is what a per-fixture memory gate needs. `ru_utime` comes
    # from the same struct, so the timing path gains a per-child number too
    # instead of a difference of process-wide totals.
    #
    # The child's pipes are replaced by temporary files: `communicate()` would
    # reap the child itself and leave nothing for wait4, and draining pipes by
    # hand risks the classic full-buffer deadlock.
    import resource  # noqa: F401  (documents the rusage struct's origin)
    import tempfile
    import threading

    timed_out = False

    with tempfile.TemporaryFile() as out_f, tempfile.TemporaryFile() as err_f:
        proc = subprocess.Popen(args, stdout=out_f, stderr=err_f, env=env)

        def _kill():
            nonlocal timed_out
            timed_out = True
            proc.kill()

        timer = threading.Timer(timeout_s, _kill) if timeout_s else None
        if timer is not None:
            timer.start()
        try:
            _pid, status, usage = os.wait4(proc.pid, 0)
        finally:
            if timer is not None:
                timer.cancel()
        # Popen still believes the child is running; tell it otherwise so its
        # finalizer neither waits nor warns.
        proc.returncode = -(status & 0x7F) if status & 0x7F else (status >> 8)

        if timed_out and not keep_output:
            _LAST_RUN_MAXRSS_MB[0] = None
            return "", 0.0, 124, ""

        out_f.seek(0)
        err_f.seek(0)
        stdout_bytes = out_f.read()
        stderr_bytes = err_f.read()
        if timed_out:
            _LAST_RUN_MAXRSS_MB[0] = None
            return (
                stdout_bytes.decode("utf-8", errors="replace"),
                0.0,
                124,
                stderr_bytes.decode("utf-8", errors="replace"),
            )

    # ru_maxrss is kilobytes on Linux and bytes on macOS/BSD.
    scale = 1024 * 1024 if sys.platform == "darwin" else 1024
    _LAST_RUN_MAXRSS_MB[0] = usage.ru_maxrss / scale
    return (
        stdout_bytes.decode("utf-8", errors="replace"),
        max(usage.ru_utime, 0.0),
        proc.returncode,
        stderr_bytes.decode("utf-8", errors="replace"),
    )


def _run_timed_win32(args, timeout_s, env=None, keep_output=False):
    import ctypes
    from ctypes import wintypes

    # pypy3.exe on Windows is a launcher that spawns the real interpreter as a
    # child process; GetProcessTimes on Popen's handle only sees the launcher
    # and reports ~0s. A JobObject aggregates user/kernel time across all
    # descendant processes, which is what we actually want.
    class _IOCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class _JobBasic(ctypes.Structure):
        _fields_ = [
            ("TotalUserTime", ctypes.c_int64),
            ("TotalKernelTime", ctypes.c_int64),
            ("ThisPeriodTotalUserTime", ctypes.c_int64),
            ("ThisPeriodTotalKernelTime", ctypes.c_int64),
            ("TotalPageFaultCount", ctypes.c_uint32),
            ("TotalProcesses", ctypes.c_uint32),
            ("ActiveProcesses", ctypes.c_uint32),
            ("TotalTerminatedProcesses", ctypes.c_uint32),
        ]

    class _JobBasicAndIo(ctypes.Structure):
        _fields_ = [("BasicInfo", _JobBasic), ("IoInfo", _IOCounters)]

    kernel32 = ctypes.windll.kernel32
    job = kernel32.CreateJobObjectW(None, None)

    proc = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
    )
    # Assigning right after Popen catches launchers like pypy3.exe before they
    # spawn their interpreter child; descendants inherit job membership.
    assigned = bool(kernel32.AssignProcessToJobObject(job, int(proc._handle)))

    try:
        stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout_bytes, stderr_bytes = proc.communicate()
        kernel32.CloseHandle(job)
        if not keep_output:
            return "", 0.0, 124, ""
        return (
            (stdout_bytes or b"").decode("utf-8", errors="replace"),
            0.0,
            124,
            (stderr_bytes or b"").decode("utf-8", errors="replace"),
        )

    utime = 0.0
    JobObjectBasicAndIoAccountingInformation = 8
    if assigned:
        info = _JobBasicAndIo()
        if kernel32.QueryInformationJobObject(
            job, JobObjectBasicAndIoAccountingInformation,
            ctypes.byref(info), ctypes.sizeof(info), None,
        ):
            utime = info.BasicInfo.TotalUserTime / 1e7
    else:
        # Job assignment refused (e.g. already in a non-nestable job on older
        # Windows). Fall back to per-process times.
        ct = wintypes.FILETIME()
        et = wintypes.FILETIME()
        kt = wintypes.FILETIME()
        ut = wintypes.FILETIME()
        if kernel32.GetProcessTimes(
            int(proc._handle),
            ctypes.byref(ct), ctypes.byref(et),
            ctypes.byref(kt), ctypes.byref(ut),
        ):
            utime = ((ut.dwHighDateTime << 32) | ut.dwLowDateTime) / 1e7
    kernel32.CloseHandle(job)
    return (
        stdout_bytes.decode("utf-8", errors="replace"),
        utime,
        proc.returncode,
        (stderr_bytes or b"").decode("utf-8", errors="replace"),
    )


def run_timed(args, timeout_s=None, env=None, keep_output_on_timeout=False):
    """Run *args*, return (stdout_str, user_cpu_seconds, returncode, stderr_str).

    returncode 124 = timeout (matching coreutils convention). *env* (when
    given) replaces the child environment (pass a full os.environ copy plus
    extras).

    `keep_output_on_timeout` still returns whatever the child wrote before it
    was killed. The default discards it: a timed-out benchmark has no output
    to compare. The trace-shape gate keeps it, because the optimized trace is
    dumped at compile time, long before a log-heavy run spends its budget.
    """
    if sys.platform == "win32":
        # No wait4 counterpart; the memory gate reads None and stays inert.
        _LAST_RUN_MAXRSS_MB[0] = None
        out, t, rc, err = _run_timed_win32(
            args, timeout_s, env, keep_output=keep_output_on_timeout,
        )
    else:
        out, t, rc, err = _run_timed_unix(
            args, timeout_s, env, keep_output=keep_output_on_timeout,
        )
    # PyPy/CPython on Windows emit CRLF in stdout text mode; Rust's println!
    # emits LF on all platforms. Normalize so output comparisons aren't
    # platform-sensitive (and snapshots stay portable).
    return out.replace("\r\n", "\n"), t, rc, err


# Names the child is allowed to inherit. Everything else in this process's
# environment is dropped before the child sees it — see `child_env_base`.
ENV_ALLOWLIST = (
    # Locating the loader, the shell a subprocess needs, and scratch space.
    "PATH",
    "PATHEXT",
    "COMSPEC",
    "SHELL",
    "TEMP",
    "TMP",
    "TMPDIR",
    # Windows refuses to start a process, resolve a system DLL or initialise
    # WinSock without these.
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "WINDIR",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
    # The home a `~` expands to, and the account naming it.
    "HOME",
    "HOMEDRIVE",
    "HOMEPATH",
    "USERPROFILE",
    "APPDATA",
    "LOCALAPPDATA",
    "PROGRAMDATA",
    "USER",
    "USERNAME",
    "LOGNAME",
    # Shared-library search, which a build out of the source tree depends on.
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "DYLD_FALLBACK_LIBRARY_PATH",
    # The locale chain, because an unset chain resolves to the C locale and
    # changes how the child decodes its own stdio — not something to vary
    # under a run whose stdout is diffed against an oracle. On Windows the
    # chain answers nothing, which is why `child_env_base` names the codec
    # outright rather than leaving it to these.
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "PYTHONIOENCODING",
)

# Prefixes a developer sets deliberately to steer this run, kept so an A/B
# stays reachable without editing this file.
ENV_ALLOWLIST_PREFIXES = ("PYRE_", "MAJIT_", "PYPY_", "PYTHON")


def child_env_base():
    """The environment a measured child starts from: an allowlist, not a copy.

    pyre copies every environment entry into `posix.environ` at startup
    (`interp_posix.rs:882 create_environ`, PyPy's `_convertenviron`), so the
    environment's total size is startup allocation like any other. That
    allocation moves where the nursery fills, which moves where a collection
    lands, which changes how many times a compiled trace re-enters a guarded
    branch afterwards — and `guard_failures` counts those re-entries.

    Measured on one binary with nothing but unrelated variables added:
    `str_fstring` read 658 or 659, entirely from one guard in trace 6 firing
    51 times or 52. `loops_compiled` and `bridges_compiled` did not move at
    all, so nothing about what was compiled differed — only how often the
    compiled code re-entered one branch. That fixture no longer demonstrates
    it: its trip counts were halved so it stops short of the major-collection
    threshold entirely, which is the durable fix for one fixture and not for
    the input.

    A CI runner injects dozens of variables a developer's shell does not
    (GITHUB_*, RUNNER_*, JAVA_HOME, and the rest), and those are exactly the
    class of input shown above to move a counter. Dropping them removes that
    input rather than relaxing the comparison. It is not established that it
    is the *only* per-host input — this pin narrows what a counter can depend
    on; it does not prove a counter now depends on nothing but the tree.

    The allowlist is not a fixed environment either: the `PYTHON` prefix keeps
    whatever a host names that way (PYTHONLOCATION, PYTHON3_ROOT_DIR on the
    runners), because the same prefix carries the PYTHONSAFEPATH and
    PYTHONDONTWRITEBYTECODE overrides a developer sets deliberately and that
    `LAUNCH_ENV_NAMES` forwards to the wasm guest. What it removes is the bulk.

    Any non-empty `PYRE_CHECK_INHERIT_ENV` restores the old whole-environment
    copy, so the effect of this normalisation stays measurable.
    """
    if os.environ.get("PYRE_CHECK_INHERIT_ENV"):
        return dict(os.environ)
    env = {}
    for name, value in os.environ.items():
        upper = name.upper()
        if upper in ENV_ALLOWLIST or upper.startswith(ENV_ALLOWLIST_PREFIXES):
            env[name] = value
    # Name the stdio codec the locale chain cannot name on every host.
    # Dropping LANG/LC_ALL/LC_CTYPE settles the child's stdio encoding on a
    # POSIX host, where the chain is what `config_get_locale_encoding` reads.
    # It settles nothing on Windows: there the answer is `cp%u` of GetACP(),
    # a machine setting no variable reaches, so a cp1252 runner and a cp949
    # workstation open their streams on different codecs and import different
    # `encodings.*` modules to do it. That import is startup work like any
    # other and it reaches jit-stats the way the bytecode cache above does --
    # measured, `import_none_sentinel` reads loops_compiled=1 against a
    # recorded 0 here, and naming utf-8 restores the recorded value exactly.
    #
    # The value spells the handler as well as the codec, because the handler
    # is the answer both hosts already give: Windows asks for surrogateescape
    # and a scrubbed POSIX chain resolves to the C locale, which asks for the
    # same. Naming a bare encoding would settle it at strict instead and so
    # change the streams it is meant to hold still. stderr keeps
    # backslashreplace either way -- `init_sys_streams` hardcodes it.
    #
    # An explicit PYTHONIOENCODING still wins, so the A/B stays reachable.
    env.setdefault("PYTHONIOENCODING", "utf-8:surrogateescape")
    return env


# 4MB: env.py NURSERY_SIZE_UNKNOWN_CACHE / collector.rs DEFAULT_NURSERY_SIZE
PYPY_GC_NURSERY_PIN = str(4 * 1024 * 1024)


def pyre_env():
    """Child environment for pyre runs: strict JIT plus one-line stats.

    MAJIT_STRICT=1 re-raises internal compile panics instead of silently
    falling back to the interpreter, so a JIT bug surfaces as a crash here
    rather than as correct-but-uncompiled output. MAJIT_STATS=1 prints the
    `[jit-stats]` line that `_jit_panic_reason` inspects.

    PYRE_DESCR_SPELLING_GATE=1 makes the descr-universe counters in that line
    name their members. The counters are gated at zero
    (JITSTATS_BADNESS_FIELDS), and a bare `descr_set_ambiguous 0 -> 1` says
    nothing about which container disagreed — on a platform the developer
    cannot reproduce, the name is the whole diagnosis. Naming costs one stderr
    line per member and only ever prints when a member fails to resolve.
    """
    env = child_env_base()
    env["MAJIT_STRICT"] = "1"
    env["MAJIT_STATS"] = "1"
    env["PYRE_DESCR_SPELLING_GATE"] = "1"
    # Pin the nursery, so the minor-collection schedule is a property of the
    # tree rather than of the machine. `default_nursery_size` (collector.rs)
    # falls back to `estimate_best_nursery_size`, which is half the L2 cache
    # once that exceeds 8MB and `DEFAULT_NURSERY_SIZE` (4MB, env.py's
    # `NURSERY_SIZE_UNKNOWN_CACHE`) otherwise. That probe is macOS-only —
    # `get_l2cache` is `-1` under `cfg(not(target_os = "macos"))` — so two
    # macOS machines can disagree while every Linux and Windows host already
    # sits on the 4MB fallback.
    #
    # `guard_failures` is what moved: it counts bailouts, and a collection
    # landing a little earlier or later moves them. Measured — a macos runner
    # reported 19 differences against baselines recorded here, every one of
    # them `guard_failures` and most of them identical on both backends (so the
    # difference is in the schedule, not in codegen: no `loops_compiled` or
    # `bridges_compiled` moved at all). Sweeping `PYPY_GC_NURSERY` reproduced
    # that runner's numbers exactly at 6MB — `fib_loop` 189, `dict_set` 609,
    # `listcomp_hot` 239, `nested_list_comprehension_hot` 402,
    # `recursive_call_frame_relocation` 636, `str_fstring` 657,
    # `closure_per_call` 415, `type_name_setter` 2, all eight. A 12MB L2 halves
    # to exactly that.
    #
    # 4MB is the documented unknown-cache value and what the non-macOS hosts
    # compute anyway, so pinning it costs nothing there and makes a macOS
    # recorder agree with them. It no longer decides `guard_failures` — the
    # threshold pin below removes the collection those readings were counting,
    # and the baselines are recorded with both pins in place — but it still
    # fixes the minor schedule and the nursery term the threshold derives from.
    env.setdefault("PYPY_GC_NURSERY", PYPY_GC_NURSERY_PIN)
    # Pin the major-collection threshold too, because the nursery pin above only
    # fixes one term of it. `min_heap_size` is `PYPY_GC_MIN`, else `nursery * 8`
    # (collector.rs:648), floored by `nursery * major_collection_threshold`
    # (:662), and installed as `next_major_collection_threshold` (:702) — so a
    # 4MB nursery also sets a 32MB threshold, and where old-gen use crosses it
    # (:2437 `threshold_reached`) stayed free to move.
    #
    # Crossing it used to reach `guard_failures`, and no longer does. The major
    # step's finalizer trigger arms the eval-breaker word, and every compiled
    # loop's back edge polls that word through a real guard —
    # `RawLoadI(&EVAL_BREAKER_WORD) -> IntAnd(word, JIT_BREAKER_MASK) ->
    # IntIsTrue -> GuardFalse` (trace_opcode.rs) — whose failure was counted
    # like any other. It is now tallied separately as `back_edge_polls`
    # (`is_back_edge_poll_guard` marks the guard, `record_guard_failure_event`
    # splits the total), which is what removes the whole class rather than
    # avoiding it; the rest of this note is kept because it describes the pins
    # that are still in place and what they were measured to do. The trace is
    # peeled, so that
    # poll appears twice, and which copy catches the armed bit sets the price:
    # the loop-body copy costs one bailout, the peeled-preamble copy costs two,
    # because resuming from it re-enters at the loop head and fails the
    # preamble's FOR_ITER guard once more. That is the whole of the
    # `recursive_call_frame_relocation` 638/637 knife edge — 638 preamble, 637
    # body, 636 no crossing at all — with `loops_compiled` and
    # `bridges_compiled` identical across all three, and it is why the windows
    # runner disagreed with itself between jobs rather than against the tree.
    #
    # Pushing the threshold past a fixture's working set removes the event
    # instead of relocating it. Measured: `recursive_call_frame_relocation`
    # reads 636 at every nursery from 3968KB to 8MB, and `closure_per_call`
    # reads 414 where it had alternated 415/416 — the pair a per-platform
    # overlay could not hold.
    #
    # "Past every fixture's working set" is what this pin was believed to do
    # and is not what it does — the two fixtures above are not the suite.
    # Measured across all 404 dynasm synthetic fixtures, comparing each one's
    # count at this 256MB pin against the same fixture at 8GB (where no major
    # collection happens at all): 11 of them still cross, for 21 poll failures
    # in total. `bytes_split_whitespace_maxsplit` reads 4 here and 1 there,
    # `build_set_hashability` 4 and 1, `str_fstring` 658 and 657. So the pin
    # narrows the class and does not close it, which is why the counter split
    # above exists — those 11 were the fixtures that flipped between hosts.
    #
    # The residual is not more of the same class, though, and calling those 21
    # "poll failures" is wrong. Taking `str_fstring` apart guard by guard: its
    # 658-vs-657 is one guard in trace 6 firing 52 times or 51, and that guard
    # is an ordinary branch, not the breaker poll. The poll is what bails out;
    # re-entry after the bailout takes the branch once more, and that hit is
    # what the counter records. So the counter split above removed the poll
    # class exactly as described, and what is left is a real deopt the split
    # cannot absorb and no GC pin closes, because it is not a crossing being
    # counted. What closes it for a fixture is keeping the fixture short of the
    # crossing — see `str_fstring.py`'s header, where halving the trip counts
    # is what settled this one. Holding the startup allocation still
    # (`child_env_base`) narrows what can move the placement; it does not
    # remove the placement. Host RAM cannot re-open it: `max_delta`
    # (0.125 * total memory) enters only as an upper bound at collector.rs:3570
    # and the `min_heap_size` floor is applied after it (:2452), so the floor
    # wins on every host.
    #
    # "Every fixture's working set" is the criterion, but the value was chosen
    # against the two fixtures that motivated it, and `str_fstring` sits above
    # it. That fixture merges six hot loops into one run, and at 256MB it still
    # crossed: it reads `guard_failures` 658 or 659 as a function of nothing but
    # how much the process allocated before the loops got hot. Startup
    # allocation is the whole input — a 49-variable environment reads 659 and a
    # 9-variable one 658, with the same binary, the same nursery and the same
    # fixture, non-monotonically in the variable count. That is the same
    # preamble-vs-body knife edge described above, and it is what the three
    # per-platform overlays were holding: each host has its own startup
    # allocation volume, so each landed on its own side of the crossing.
    #
    # 512MB clears it. `str_fstring` then reads 657 at every nursery from 4MB to
    # 32MB and under every perturbation of that startup volume — the same
    # invariance that says "no crossing at all" for the two fixtures above.
    # Eight other fixtures move with it, none of them a knife edge at either
    # value: `arith_int_bool` 2214 -> 2212, `build_set_hashability` 4 -> 2,
    # `bytes_split_whitespace_maxsplit` 3 -> 2,
    # `delete_negative_open_slice_hot` 1405 -> 1404, `instance_dict_reassign`
    # 2 -> 1, `newslice_step_hot` 5 -> 3 and `unpack_ex_hot` 3 -> 2 lose a
    # crossing, and `inlined_helper_mutation` 3 -> 4 relocates one. No
    # `loops_compiled` or `bridges_compiled` moves anywhere in the suite, which
    # is the signature that says the collection schedule is the only thing that
    # changed.
    #
    # The cost is memory, and it is paid by the one fixture that was over the
    # old threshold: `str_fstring` peaks at 618MB against 519MB. Every other
    # fixture is flat to within a megabyte (`closure_per_call` 179 -> 180MB,
    # `recursive_call_frame_relocation` 159 -> 160MB), because a threshold only
    # costs what a fixture actually retains.
    #
    # Both pins reach the wasm backend only because `pyre-wasm-runner` hands
    # them to the guest explicitly (`pyre_set_gc_env`): a wasm32-unknown-unknown
    # module's `std::env` is permanently empty, so setting them here does
    # nothing on its own there.
    #
    # Reaching the guest is not the same as clearing the crossing there. 256MB
    # removed it on both native backends and did not in the guest, whose
    # baselines held `recursive_call_frame_relocation` 638 and
    # `closure_per_call` 415 against their 636 and 414 — the same knife edge
    # described above, one step along it. Historically the two ran different
    # interpreter allocation models: `PYRE_GC_INTERP` was off natively and on in
    # the guest, so only the guest counted interpreter-created int/float boxes
    # toward the old-gen threshold. The gate is now default-on everywhere and
    # the allocation model is shared, but the recorded counter boundary outlived
    # that convergence, because the threshold was still inside the guest's
    # working set.
    #
    # The value below is past it, so the guest now lands exactly on the native
    # counts: both fixtures re-recorded to 636 and 414, identical across all
    # three backends. It buys that agreement with memory — `closure_per_call`
    # peaked at 459MB under the guest at 384MB against 358MB at 256MB. The
    # sensitivity this note is here to name outlives the convergence: a change
    # in interpreter-path allocation volume moves the wasm numbers while the
    # native ones sit still.
    # selfrec_bridge_nontail_promote now keeps its wasm overflow bridges
    # compiled. Its live allocation volume crosses 512 MiB: wasm and Windows
    # cranelift then add one guard exit (634 vs 633), with the same two loops,
    # four bridges and zero aborts. At 1 GiB every backend retains 633 without
    # changing a baseline. The wasm reading goes from one major collection
    # to zero, at a measured linear-memory cost of 586 MB -> 602 MB. Keep this
    # fixture beyond the same collection boundary as the cases above.
    # The same preset removes two/one major collections from wasm's
    # build_set_hashability/minmax_key_rooting: guards become 43/257 instead
    # of 42/256. Both the old and new guest reproduce those exact numbers
    # when changing only this minimum, so those baselines follow the preset.
    env.setdefault("PYPY_GC_MIN", str(1024 * 1024 * 1024))
    # Keep the bench directory off `sys.path` (`-P`), so the jit-stats counters
    # describe the fixture rather than the directory it happens to sit in.
    #
    # `sys.path[0]` is the script's own directory, and `pyre/bench/synth` holds
    # well over a thousand entries. The import machinery's scan of that entry is
    # itself a Python loop, and at that size it crosses the 1039 compile
    # threshold — so a fixture with no loop at all in its body still records
    # `loops_compiled=1`, produced by a single `import` statement. Measured, one
    # fixture unchanged and only the directory it runs from varied: 400 files
    # compile nothing, 800 compile that loop, and the real directory is well past
    # both. It is invariant to PYPY_GC_NURSERY (1MB..16MB), so it is not the
    # collection schedule.
    #
    # That is what made the counters disagree between platforms: path handling
    # differs per OS, so the same ambient loop lands either side of the
    # threshold. Every jit-stats diff CI reported was a uniform +-1 on
    # `loops_compiled` — and setting this flag reproduces the Linux numbers
    # exactly on macOS (`arith_int_bool` 8 -> 7 with guard_failures 2213 -> 2211,
    # `gc_deque_backing_list` 6 -> 5 / 204 -> 203, `struct_pack_unpack`
    # 2 -> 1 / 2 -> 1).
    #
    # `-P` alone, not `-I`: the latter also implies `-E`, which drops
    # PYTHONIOENCODING and so changes how the child resolves its stdio encoding —
    # not something to fold into a run whose stdout is diffed against an oracle.
    #
    # An explicit PYTHONSAFEPATH in the environment wins, so the old behaviour
    # stays reachable for an A/B: pyre reads the variable as a presence flag, so
    # passing it empty is what turns the flag back off.
    env.setdefault("PYTHONSAFEPATH", "1")
    # Write no bytecode cache, so a reading describes the tree rather than what
    # earlier runs left on disk. Importing a module the cache already holds
    # unmarshals where importing it cold compiles, and that difference reaches
    # jit-stats the same way the ambient import loop above does: `import json,
    # base64, textwrap` alone drops 30 `__pycache__/*.pyre315.pyc` files into
    # the pinned stdlib, so the first run of a fresh checkout and every run
    # after it measure different work.
    #
    # It also removes a class of platform disagreement rather than papering
    # over it. On Windows nothing pyre wrote was readable until FileIO put an
    # adopted descriptor in binary mode — `_write_atomic` wraps a raw fd, whose
    # CRT text mode turned every `\n` into `\r\n` — so the same tree read one
    # set of counters while the cache was dead and another once it worked.
    #
    # This suppresses writing only; a stale cache from before this pin is still
    # read, so an A/B against an older tree wants `__pycache__` cleared first.
    # `launch_env::finalize` folds the name as an integer flag, so an explicit
    # PYTHONDONTWRITEBYTECODE=0 turns it back off. It reaches the wasm guest
    # too: the name is in `LAUNCH_ENV_NAMES`, which the runner forwards through
    # `pyre_set_launch_env`.
    #
    # Only pyre is pinned this way. The oracles are spawned with the inherited
    # environment (`run_timed([python3(), script])`), so they keep writing and
    # reading their own caches and import warm from their first run onward,
    # while pyre now imports cold on every run. Startup subtraction covers the
    # imports an empty program performs, not the ones a fixture adds on top, so
    # for the handful that `import json` / `re` / `pickle` the ratio carries
    # that difference. Determinism is the trade taken here: a warm pyre depends
    # on what earlier runs left behind, and a ratio that moves with disk state
    # is worse than one that is consistently cold.
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    # Pin the vendored, `_sre.MAGIC`-matched stdlib so pyre never picks up a
    # version-mismatched host `python3` off the PATH. An explicit PYRE_STDLIB
    # in the environment wins.
    stdlib = pyre_stdlib()
    if stdlib and "PYRE_STDLIB" not in env:
        env["PYRE_STDLIB"] = stdlib
    # Point the wasm runner at the built module by absolute path so it resolves
    # regardless of the child's working directory (ignored by other backends).
    if "PYRE_WASM_MODULE" not in env and Path(WASM_MODULE_PATH).exists():
        env["PYRE_WASM_MODULE"] = str(Path(WASM_MODULE_PATH).resolve())
    # Pick the wasm runtime engine (ignored by other backends). An explicit
    # PYRE_WASM_ENGINE in the environment wins over the --wasm-engine default.
    if "PYRE_WASM_ENGINE" not in env:
        env["PYRE_WASM_ENGINE"] = WASM_ENGINE
    return env


def pypy_env():
    """Child environment for pypy oracle runs: inherited env plus a pinned nursery.

    PyPy sizes its nursery from the host cache: `get_L2cache_linux2_system_cpu_index`
    (env.py) sums L2+L3, and `best_nursery_size_for_L2cache` halves it when it
    is over 8MB, else `NURSERY_SIZE_UNKNOWN_CACHE` (4MB). Ubuntu runners with a
    32MB L3 therefore run a ~16MB nursery, so allocation-heavy fixtures depend
    on the runner CPU. Measured: `pypy3 pyre/bench/synth/type_name_surrogate_reject.py`
    takes 0.05s at PYPY_GC_NURSERY=1MB, 0.06s at 4MB, 0.13s at 32MB —
    `W_TypeObject.add_subclass` (typeobject.py) scans `weak_subclasses` for a
    dead slot, and a larger nursery leaves more dead types uncollected.

    pyre runs under the same 4MB pin (`pyre_env`), so both sides of a ratio
    see the same minor-collection schedule.
    """
    env = dict(os.environ)
    env.setdefault("PYPY_GC_NURSERY", PYPY_GC_NURSERY_PIN)
    return env


def effective_wasm_module():
    """The module path the wasm runner will actually load.

    `pyre_env` defaults `PYRE_WASM_MODULE` to the built module but leaves an
    inherited one alone, so the default is not what runs whenever the caller
    set it. A gate that asks about `WASM_MODULE_PATH` under an override checks
    a file the run never opens, and clears the way for the stale module the
    override names.

    Both `pyre_env` and the runner read the key by presence, so an empty value
    survives into the child and is opened as the empty path; resolving it here
    by falsiness would vouch for the built module while every invocation failed.
    """
    override = os.environ.get("PYRE_WASM_MODULE")
    return WASM_MODULE_PATH if override is None else override


def same_file(one, other):
    """Whether two paths name one file, tolerating a missing one.

    The override is compared against the built module to decide whether the
    build settled the question, and it is written by a caller who may spell it
    absolutely, through a symlink, or relative to a different directory.
    `os.path.samefile` answers that but raises when either side is absent,
    which here is not an error: a missing file is simply not the built one, and
    the existence check above has already spoken for the module itself.
    """
    try:
        return os.path.samefile(one, other)
    except OSError:
        return False


def _dump_failed_run(output, stderr, limit=40):
    """Print the tail of a failed run's captured streams.

    A self-checking guard reports only its exit code, so without this the
    traceback that explains the failure never reaches the log and a
    platform-specific break has to be reconstructed from the source. Bounded to
    the last *limit* lines of each stream so a runaway script cannot flood CI.
    """
    for label, text in (("stdout", output), ("stderr", stderr)):
        if not text or not text.strip():
            continue
        lines = text.rstrip().splitlines()
        clipped = lines[-limit:]
        print(f"─── {label} ───")
        if len(clipped) < len(lines):
            print(dim(f"... {len(lines) - len(clipped)} earlier line(s) omitted"))
        for line in clipped:
            print(line)
    print("────────────────────")


def _dump_output_mismatch(actual, expected, limit=80):
    """Print a bounded unified diff for a backend stdout mismatch."""
    diff = list(
        difflib.unified_diff(
            expected.splitlines(),
            actual.splitlines(),
            fromfile="expected (pypy)",
            tofile="actual",
            lineterm="",
        )
    )
    if not diff:
        # Preserve otherwise-invisible trailing-newline/line-ending details.
        print(f"    expected repr: {expected!r}")
        print(f"    actual repr:   {actual!r}")
        return
    print("─── output diff ───")
    for line in diff[:limit]:
        print(line)
    if len(diff) > limit:
        print(dim(f"... {len(diff) - limit} later diff line(s) omitted"))
    print("────────────────────")


def _first_stderr_line(stderr):
    """Return `"  <first non-empty stderr line>"`, or `""` when there is none.

    A traceback additionally carries its final line, which is the one that
    names the exception.

    A crash reason is otherwise just an exit code, which on a binary that dies
    before it writes any stdout says nothing about why. The empty answer is
    itself informative: it names a process that produced no diagnostic at all.

    An app-level crash is the one case where the first line carries nothing:
    every Python traceback opens with the same banner, so a fixture that dies
    on an assertion reports only `Traceback (most recent call last):` and the
    CI log holds no other copy of the frames. The line that names the failure
    is the last one, so append it — the same reason `_jit_panic_reason`
    appends the message line that follows a Rust panic's location.
    """
    lines = [line.strip() for line in (stderr or "").splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return ""
    reason = lines[0][:160]
    exception = _traceback_exception_line((stderr or "").splitlines())
    if exception:
        reason += f" | {exception[:400]}"
    return f"  {reason}"


def _traceback_exception_line(raw):
    """The line naming the exception in the last traceback `raw` holds.

    A traceback's frames are indented and the line naming the exception is not,
    so the block ends at the first unindented line after the banner. The
    stderr's final line is not that line whenever anything writes after the
    interpreter unwinds, and something does: the trace logger flushes its
    `=== JIT Statistics ===` summary at exit, whose last line is
    `Compilation time: <n>ms`. A fixture that died on `ValueError` then reported
    that timing as its reason, which names nothing -- the crash of
    `synth/exception_reused_object_tb_not_doubled` on linux/x86 dynasm reached CI
    as `Traceback (most recent call last): | Compilation time: 1.4ms`.

    Chained reports repeat the banner and separate the blocks with their own
    unindented line, so the scan starts at the LAST banner: that block ends with
    the exception the process actually died on.
    """
    banner = "Traceback (most recent call last):"
    starts = [i for i, line in enumerate(raw) if line.strip().startswith(banner)]
    if not starts:
        return ""
    for line in raw[starts[-1] + 1:]:
        if line.strip() and not line[0].isspace():
            return line.strip()
    return ""


# Width of the panic MESSAGE kept next to the `panicked at file:line:col:`
# location. The collector's marking-bug report is the widest one pyre emits:
# it names the holder, the child, both generations, whether the holder is in
# the remembered set, and formats two `[usize; 8]` word dumps.
PANIC_BODY_CHARS = 2000


def _panic_message_body(lines):
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
    body = []
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


def _jit_panic_reason(stderr):
    """Return a failure reason if *stderr* shows a JIT-level Rust panic or a
    nonzero internal_compile_panics stat, else None.

    A Rust panic prints 'panicked at' via the default hook (InvalidLoop is
    suppressed by pyre's panic hook, so legitimate trace aborts never appear
    here). A nonzero internal_compile_panics in the `[jit-stats]` line means an
    internal compile bug fell back to the interpreter (only reachable in a
    non-strict build; under MAJIT_STRICT the panic re-raises and shows up as
    'panicked' plus a nonzero exit instead).
    """
    if not stderr:
        return None
    if "panicked" in stderr:
        lines = stderr.splitlines()
        for idx, line in enumerate(lines):
            if "panicked" in line:
                reason = f"rust panic: {line.strip()[:80]}"
                # The location line alone is not enough to triage a flaky
                # crash from CI logs, and no excerpt width is enough either.
                # At 200 the varsize-length diagnostic was cut
                # mid-`nursery_start`, dropping both `forwarded=` and
                # `site=` — the two fields that name which path reached the
                # object. At 400 the two `invalid type_id` diagnostics,
                # about 800 characters once their eight-word holder and
                # child dumps are in, were cut inside the first dump. At
                # 1200 the major collector's own was cut at `site=`,
                # dropping both generations, the remembered-set membership
                # and the enclosing container. Those dumps are `{:#x?}`
                # arrays, which print one element per line, so a first-line
                # excerpt truncates them at any width; keep the whole
                # message body instead.
                body = _panic_message_body(lines[idx + 1 :])
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


BOOTSTRAP_FAILURE_MARKER = "pyre: importlib bootstrap failed:"


def _importlib_bootstrap_failure(stderr):
    """Return the reported reason if *stderr* says the importlib bootstrap
    failed, else None.

    `init_importlib_bootstrap` failing is non-fatal in the interpreter by
    design: a build with no reachable stdlib keeps serving imports from the
    native importer, the minimal-importer role. Under this script the stdlib is
    always reachable, so the only way to reach it is a builtin module the
    bootstrap needs and the binary does not have, and every number measured
    afterwards describes the fallback rather than the product.
    """
    if not stderr:
        return None
    for line in stderr.splitlines():
        if line.startswith(BOOTSTRAP_FAILURE_MARKER):
            return line.strip()
    return None


# ── Helpers ──────────────────────────────────────────────────────────

def _jit_stats_merged(stderr):
    """Every `[jit-stats]` line merged into one `key -> value` map, UNFILTERED.

    `None` when the run printed no such line at all, which is a different
    condition from printing one with no keys.

    Split out from `_jit_stats_snapshot` because the two callers want opposite
    things. A committed baseline may only contain the host-stable surface, so
    the snapshot narrows to `JITSTATS_SNAPSHOT_FIELDS`; a *non-vacuity* check
    has to read a counter precisely because it is NOT recorded — see
    `JITSTATS_DENOMINATOR_FIELDS`. Narrowing first would hide it.
    """
    fields = {}
    seen = False
    for line in stderr.splitlines():
        if not line.startswith("[jit-stats]"):
            continue
        seen = True
        for token in line[len("[jit-stats]") :].split():
            if "=" in token:
                key, value = token.split("=", 1)
                fields[key] = value
    return fields if seen else None


def _jit_stats_field(stderr, field):
    """One integer counter off the merged map, or `None`.

    `None` covers a run that printed no `[jit-stats]` line, one that printed
    no such key, and one whose value does not read as an integer. None of
    those is the same answer as a printed zero, which is why the caller
    distinguishes them.

    The unfiltered map and not the snapshot, for the reason
    [`_jit_stats_merged`] gives: a non-vacuity check reads a counter precisely
    because it is not part of the recorded surface.
    """
    fields = _jit_stats_merged(stderr or "")
    if not fields or field not in fields:
        return None
    try:
        return int(fields[field])
    except ValueError:
        return None


def _jit_stats_snapshot(stderr, ungated=()):
    """Return every jit-stats line merged into normalized key/value text.

    The interpreter emits more than one `[jit-stats]` line — the counters, the
    mc_diag tallies, the descr-set universe. Keeping only the last one dropped
    `loops_aborted` and `internal_compile_panics` from the snapshot, which
    silently disarmed `_jit_stats_change`: it read both counters as
    absent on the current side, so `cur > base` could never hold no matter how
    far they rose. Later lines still win on a repeated key.

    Merging is what keeps that true as lines are added. Selecting one line
    instead — the last whose first token is not a bare diagnostic name like
    `mc_diag` / `heap_buckets` / `bridge_diag` / `fbw_diag` / `exec_hist` —
    re-disarms the floor the moment a second key=value line joins the first,
    because the newcomer displaces the counters. The bare-name lines need no
    special case here: their name token carries no `=` and is skipped with
    every other non-assignment token.

    The merged set is then narrowed to `JITSTATS_SNAPSHOT_FIELDS`, because what
    a run may read and what a committed baseline may contain are two different
    questions: merging arms the floor, the filter keeps the recorded surface
    host-stable.

    *ungated* names counters this fixture's header exempted
    (`synth_ungated_jitstats`). They are dropped here rather than at the
    comparison, so the recorded baseline never carries them either: a field
    absent from both sides compares equal, which is what makes one filter serve
    the record and the gate at once.
    """
    fields = _jit_stats_merged(stderr)
    if fields is None:
        return None
    fields = {
        k: v
        for k, v in fields.items()
        if k in JITSTATS_SNAPSHOT_FIELDS and k not in ungated
    }
    # This watches what the JIT compiles, never how well. A regression that
    # changes no structure (for example, an extra spill) is invisible here.
    return "".join(f"{key}={fields[key]}\n" for key in sorted(fields))


def _parse_jit_stats(snapshot):
    """Split a normalized jit-stats snapshot into its `field -> value` map. A
    snapshot that was never taken reads as no fields at all, so every caller can
    treat "absent" and "absent from this side" the same way."""
    if snapshot is None:
        return {}
    return dict(line.split("=", 1) for line in snapshot.splitlines())


# Every counter a `.jitstats` baseline records is gated, in BOTH directions: the
# recorded surface and the gated surface are the same set, so a baseline can
# never carry a number nobody checks. A fixture header may drop a structural
# counter from that set (`synth_ungated_jitstats`), and it leaves both halves at
# once for exactly this reason.
#
# Both directions, because an improvement has to be recorded too. A counter that
# falls is a real change in what the JIT compiles, and leaving the fall ungated
# means the committed baseline quietly stops describing the tree — the next
# regression is then measured against a number that was already stale. Gating a
# fixture on an exact match is upstream's own instrument, not a stricter one
# invented here: `OpMatcher.match` (test_pypy_c/model.py) compares a compiled
# loop operation by operation against a literal expected trace, so an
# improvement that reorders or removes an operation fails that assertion until
# the expected trace is updated. These counters are a coarser projection of the
# same idea.
#
# The exact comparison is affordable, but not because the compile decision is
# free of per-run inputs — it is not. `JitCounter` does index the loop half by
# a hash rather than an address, but the guard half is built from one:
# `current_object_addr_as_int(self) * 777767777 + intval * 1442968193`
# (compile.py:780-781, ported at pyjitpl.rs:10841-10845). Upstream spells it
# the same way, so this is orthodoxy rather than drift, and it is not what
# moved these numbers.
#
# What moved them is that `guard_failures` is not a compile decision at all.
# Each counter is a plain increment at the event (`record_guard_failure_event`,
# pyjitpl.rs), so the total is the runtime taken-count of every guard site —
# the same run reports `mc_diag mc_entered` with the identical value, because
# the counter is the metainterp-entry count. An already-compiled guard running
# once more moves it while `loops_compiled` and `bridges_compiled` stay put,
# which is exactly the shape every disagreement took. The GC schedule was
# reaching it through the back-edge safepoint poll; both halves of that
# schedule are pinned above, and with them pinned the numbers agree across
# hosts and across tree vintages — `recursive_call_frame_relocation` reads 636
# and `closure_per_call` 414 on macOS and on Linux alike, where unpinned the
# same nursery gave 638 on one and 637 on the other.
#
# Measured across runners as well as run-to-run — one CI run reported
# `list_length_hint_validate guard_failures 828 -> 4923` and
# `exception_args_virtual 401 -> 1002` with identical digits on ubuntu-24.04,
# macos-latest and windows-latest. That measurement is what retired the +25%
# band `guard_failures` used to carry for cross-runner drift that had never
# been measured.
#
# `internal_compile_panics` counts an internal compile bug falling back to the
# interpreter, so 0 is its only healthy value. `loops_aborted` is not absolute:
# an abort is either a give-up the tracer is supposed to perform (an over-long
# trace hits `blackhole_if_trace_too_long`, pyjitpl.py) or a walker
# coverage gap that is tracked elsewhere — the synth fixtures below carry
# `LoopBearingCalleeInlineUnsupported`, the walker's open gap on inlining a
# loop-bearing callee. Either way the baseline pins the count those known
# declines produce today, and a rise means a loop the tracer used to finish has
# stopped finishing.
#
# `descr_set_absent`, `descr_set_ambiguous` and `descr_set_stale_absent` are the
# descr-universe invariants. Upstream cannot reach any of the three —
# `effectinfo.py:492-494` builds its sets out of the descr objects `cpu.*descrof`
# just created in the same process, so no member of a raw set can fail to
# resolve — and it expresses this class of condition with a plain `assert`
# (`descr.py`, `effectinfo.py:486,525`). Pyre resolves the sets across a  allow-line-citation
# build-time/runtime split where all three are reachable in principle and the
# runtime answer is a sound degradation rather than a crash, so the assertion is
# expressed as a counter that must not rise off zero. A field absent from a
# baseline reads as 0, so these gate from the first run without re-recording
# anything.
#
# `fbw_rolled_back_with_effects` counts walks that ended UNCOMMITTED after a
# residual had already run an irreversible effect — it wrote live heap outside
# the journals, or it entered a Python frame. The store journal cannot undo
# either, so the legacy replay the caller falls back to applies those effects a
# second time. Upstream has no counterpart state: `_copy_data_from_miframe`
# (blackhole.py) copies the register banks unconditionally and has no failing
# path, so every abort there leaves a resumable image. Pyre's mirror is partial,
# so an abort can arrive with no image to adopt, and the difference is a wrong
# answer rather than a slow one — three such bugs (a locals-typed vable overlay,
# an unpublished escape root stack, an unmodelled LOAD_SPECIAL) were each found
# by reading this counter's population by hand. Gating it is what makes the
# fourth one fail CI instead of surfacing as an intermittent asyncio failure.
# Like `loops_aborted` it is not absolute: the baseline pins the instances that
# exist today and a rise means a new one.
#
# `field_pos_spec_misplaced` and `field_pos_attached_misplaced` are the same kind
# of assertion one layer down: a field descr's `index_in_parent` (`descr.py`)
# must name the slot its own parent puts the field in, which upstream gets for
# free because `heaptracker.py:60-72` and `:96-112` are one walker. Both are
# counted where the producer's number is still readable — before
# `get_field_descr` re-derives it — because the counter that looks like it
# already gates this, `field_pos_rederived`, cannot: `derive_index_in_parent`
# overwrites the number it disagrees with, and skips the check entirely whenever
# the parent is not yet published, which is most of them. So a printed
# `field_pos_rederived=0` never meant the producers agreed, and gating it would
# have gated nothing.
JITSTATS_BADNESS_FIELDS = (
    "loops_aborted",
    "internal_compile_panics",
    "descr_set_absent",
    "descr_set_ambiguous",
    "descr_set_stale_absent",
    "fbw_rolled_back_with_effects",
    "field_pos_spec_misplaced",
    "field_pos_attached_misplaced",
    "fbw_store_journal_rollback_failed",
    # Three of the four `fbw_diag` tripwires. Unlike the rest of that line's
    # tallies these are not workload counts: each marks a leg the walker takes
    # only when a cheaper one was unavailable, and `trace.rs` says of them that
    # a nonzero value "is itself the news". Their healthy value is exactly
    # zero, which is what makes them badness counters and bars them from being
    # banded.
    #
    # Listing a counter that is zero on both sides costs no re-record:
    # `_jit_stats_change` reads a field missing from either side as "0", so
    # every existing baseline keeps matching without being touched. That is
    # what makes arming these free, and it is also why each one's zero has to
    # be established on the tree first — a counter that is nonzero anywhere
    # fails every baseline that does not name it.
    #
    # Established by running the full corpus on all three backends with these
    # listed: zero fixture diffs and no baseline written. `fbw_midbody_latch`
    # is deliberately NOT here — the same run found it at 1 on five fixtures
    # natively and six on wasm, so its leg is now reachable and gating it at
    # zero would be asserting something false. Its hazardous subset,
    # `fbw_midbody_latch_new_unjournaled`, is still zero everywhere, which is
    # the half that matters and the half that is armed.
    "fbw_midbody_latch_new_unjournaled",
    "fbw_escape_plain_fallback",
    "fbw_escape_plain_fallback_unclean",
    # `fbw_foriter_item_dropped` counts a FOR_ITER item the walk consumed
    # concretely and no leg handed back, so the caller's loop runs one
    # iteration fewer than the program says. It is the same kind of counter as
    # `fbw_rolled_back_with_effects` and for the same reason: the refusals that
    # produce it are each correct in isolation (delivering would double a
    # committed effect, or push onto a stack the frame is not parked on), and
    # the price of being correct is a WRONG ANSWER rather than a slow one.
    #
    # Its witness is `class_stmt_after_attr_store_keeps_the_iteration`: before
    # the class-body qmut abort latched its operand stack, a `class` statement
    # after `obj.attr = v` in a loop body lost 5 of 10000 iterations on both
    # native backends, and the only reason anyone found it was reading a census
    # by hand. Nothing in the corpus reaches either refusal now — the four
    # profiler fixtures that take an in-flight item deliver all 19 of them —
    # which is what makes zero the value to arm.
    "fbw_foriter_item_dropped",
)

# A zero-valued badness counter is meaningful only if its source census ran.
# Pair counters whose census has a reliable denominator and reject runs where
# that denominator is absent or zero. The denominator is not baselined because
# it describes a host-dependent descriptor population; only non-vacuity is
# stable across hosts. `field_pos_attached_misplaced` is intentionally absent:
# its checked population can legitimately be empty for compiled fixtures.
JITSTATS_DENOMINATOR_FIELDS = {
    "field_pos_spec_misplaced": "field_pos_spec_checked",
}

# The count-valued counters, and what a move in either direction means:
#
# * `guard_failures` counts every guard failure that re-enters the metainterp,
#   so it is what moves when a compiled loop's guards start failing more often —
#   a cost `max-pypy-ratio` cannot see, because that ceiling is one absolute
#   number per fixture: recursion_memo_branch sits near 8.9x under a limit of
#   20, so it absorbs a 2x regression and still passes. Narrowing
#   `nested_break_bridge_resume_hazard` so that fixture's `main` compiles moves
#   the counter 1757 -> 2613 (+49%) and measures slower, and no other gate here
#   notices.
# * `loops_compiled` closes the blind spot the others leave open: a change that
#   makes the tracer stop admitting a frame at all — a widened
#   `unsupported_jit_shape` predicate, a pre-trace decline — aborts nothing
#   (`loops_aborted` holds) and *lowers* `guard_failures` (the compiled guards
#   that used to fail are gone), so a fall here is the signal that a hot loop
#   went back to running interpreted.
#   This counts all compilations, including straight-line traces, so it cannot
#   distinguish a loop-shaped trace from a flat trace. Inspect `Label` and
#   `Jump` operations when that distinction matters.
# * `bridges_compiled` is that same signal one level down: guards that stop
#   earning a bridge keep re-entering the metainterp forever. It was left
#   ungated for a long time on the grounds that it "moves in both directions
#   under ordinary tuning", which made a bridge collapse (27 -> 0) invisible for
#   as long as `guard_failures` stayed inside its band — the whole dead-bridge
#   class this suite exists to catch.
# * `retraces_compiled` records that a requested retrace was assembled and
#   attached. A fall means the retrace path stopped producing an artifact;
#   without this positive signal, `loops_aborted` can hold at zero and the
#   accompanying fall in `guard_failures` looks like an improvement.
# * `fbw_blackhole_adopted_single_frame` and
#   `fbw_blackhole_adopted_multi_frame` count successful full-body-walk
#   blackhole adoptions: the walk handed the interpreter a resumable image
#   instead of making its caller replay the region. A fall means the adoption
#   path stopped firing and the legacy replay path came back; a rise means more
#   walks avoided replay and the baseline should say so.
#
# A move is not *always* a defect: two loops merging into one trace lowers
# `loops_compiled` while raising coverage. That is the bargain `loops_aborted`
# already makes — an intentional structural change re-records its baseline, and
# the gate is what forces someone to look at the delta and say why.

# What a committed `.jitstats` baseline is allowed to contain. `_jit_stats_snapshot`
# merges every `[jit-stats]` line so the floor above cannot be disarmed by
# reshuffling them, but the merged set also carries keys that are not comparable
# against a file checked into the repo:
#
# * `all_descrs` and `descr_set_resolved` are host-dependent because Charon/LLBC
#   extraction runs per host. Committing either count would make otherwise
#   equivalent hosts disagree.
# * the `mc_diag` tallies include contention counters (`busy_skip`,
#   `stfe_cell_busy`, `stfe_tick`) that move with machine load.
# * `PYRE_WASM_JIT_STATS=1` adds wall-clock (`compile_ms`) and repeated
#   `exec_hist` lines whose keys collide across lines; check.py never sets it,
#   but a developer who has it exported would otherwise poison a `--snapshot`
#   re-record.
#
# The badness fields are all zero in a healthy run and the three structural
# counts are host-stable, so this surface needs no re-record when a new
# diagnostic line is added.
JITSTATS_SNAPSHOT_FIELDS = JITSTATS_BADNESS_FIELDS + (
    "loops_compiled",
    "bridges_compiled",
    "retraces_compiled",
    "guard_failures",
    "fbw_blackhole_adopted_single_frame",
    "fbw_blackhole_adopted_multi_frame",
)

# `back_edge_polls` is deliberately absent, and is the one counter that must
# stay absent. It reports how many times a compiled loop left machine code
# because the eval-breaker word was armed — a measure of when a collection
# landed, not of anything the compiler decided. Recording it would move the
# schedule-sensitivity this split was made to remove onto a new key instead of
# removing it. It is printed on the `[jit-stats]` line either way, so a reader
# diagnosing a `guard_failures` move can still see it.

# The remaining `fbw_diag` tallies — `fbw_walks`, `fbw_escape_*`, `fbw_force_*`
# — stay absent, and unlike `back_edge_polls` that is a decision to revisit
# rather than a rule. They were emitted with BARE keys (`portal_only`,
# `by_portal`, …) until they were namespaced, so no committed baseline carries
# them under any spelling. Two reasons to leave them ungated for now, and the
# three tripwires above are listed precisely because NEITHER reason reaches
# them:
#
# * They are workload counts, not badness counters: none has a healthy value,
#   and neither direction is a defect on its own — the same reason
#   `bridges_compiled` sits in neither regression list. Gating needs a polarity,
#   and nothing here has measured one yet. The tripwires do have one: zero.
# * A count that is nonzero on the fbw fixtures reads as 0 on the baseline side
#   the moment it is listed here (`_jit_stats_change` treats an absent field as
#   "0"), so adding one would fail every affected baseline until all of them are
#   re-recorded. That re-record is the decision, and it should be made
#   deliberately with a polarity in hand, not as a side effect of naming a key.
#   A counter that is zero on BOTH sides costs no such re-record, which is why
#   the tripwires could be armed without touching a baseline.
#
# `fbw_midbody_latch` is ungated for a third reason that is neither of those.
# It reads as a tripwire but has no healthy value: the corpus reaches its leg
# on five fixtures natively and those five plus `str_search_index_bounds` on
# wasm, each exactly once. Arming it at zero would assert a falsehood.
#
# That is not something this branch caused, and not a recent drift either. A
# control build of the base commit, full LLBC extraction and both native
# binaries, reads 1 on the same five fixtures with the same
# `fbw_midbody_latch_new_unjournaled=0` — the six fixture files are
# byte-identical across the two trees, and the increment guard
# (`is_top_inline && !unjournaled_before_subwalk`) is byte-identical too. The
# leg was already reached when the counter was added; `trace.rs` used to say
# the opposite and now records the measurement.
#
# Its hazardous subset is armed above and zero on every fixture of the corpus
# on all three backends, which is what bounds the cost of leaving the
# population count ungated.
#
# They are on the `[jit-stats] fbw_diag` line on both the native and the wasm
# reader, so a human can read them without a gate.


# Which way each counter has to move to be a regression rather than a gain.
# Both outcomes fail — a baseline that stopped describing the tree has to be
# re-recorded either way, and letting a gain pass is how the eight baselines
# this split was written for went stale. The labels exist so a reader can tell
# the two apart at a glance: REGRESSED is "something got worse, go look",
# IMPROVED is "something got better, re-record it".
#
# `loops_compiled` is inverted against the badness fields: it is the counter
# that falls when the tracer stops admitting a frame at all, which aborts
# nothing and *lowers* `guard_failures`, so a fall is the regression and a rise
# is the gain — within the limit recorded above, that it counts compiles and not
# loop-shaped ones. `retraces_compiled` has the same polarity: a fall means an
# assembled retrace stopped being attached. The blackhole adoption counters are
# inverted the same way: a fall means the interpreter stopped receiving an
# image and went back to replay.
JITSTATS_REGRESSION_ON_RISE = JITSTATS_BADNESS_FIELDS + (
    "guard_failures",
)
JITSTATS_REGRESSION_ON_FALL = (
    "loops_compiled",
    "retraces_compiled",
    "fbw_blackhole_adopted_single_frame",
    "fbw_blackhole_adopted_multi_frame",
)

# How many fresh runs a disagreeing fixture gets before its counters are called
# stable. Only a fixture that already disagrees pays this, so the common path is
# untouched.
#
# The gate's premise is that these counters are a function of the program. Where
# that holds, a repeat reproduces the first run exactly and the disagreement is
# gated as before. Where it does not — measured on this tree: two runs of the
# same `pyre-dynasm` binary, minutes apart on a loaded machine, disagreed by
# `loops_compiled` +2 on five unrelated fixtures — the number being compared is
# not a property of the tree, and failing on it reports load as if it were a
# code change.
#
# So instability is MEASURED, never declared: nothing is annotated as flaky, and
# a fixture stops being gated only by demonstrating, in this same invocation,
# that it does not reproduce itself. The cost is that a genuinely
# nondeterministic regression now warns instead of failing — which is the right
# trade only because a counter that will not reproduce cannot be gated on
# exactly anyway, and the warning still names it.
JITSTATS_STABILITY_RUNS = 2

# `bridges_compiled` is deliberately in neither: it is not interpretable on its
# own, in either direction. A fall to 0 is the dead-bridge regression this gate
# was built to catch when guards still fail, and a gain when the guards stopped
# failing (`list_length_hint_validate` fell 4 -> 0 as guard_failures fell
# 828 -> 1). A rise is more guards crossing the bridge threshold, which is
# either wider coverage or a guard storm. Undecidable counters are reported as
# regressions, so the ambiguous case is the one a human is asked to look at.


def _jit_stats_change(saved, current, bands=None):
    """Return `(regressions, improvements, banded)` change descriptions.

    Both directions gate: a rise means the JIT started aborting traces, hitting
    internal compile panics or failing guards it did not before, and a fall
    means it stopped compiling something it used to compile. Neither may pass
    unrecorded — see the surface comment above.

    A band reclassifies an integer-valued move at or below its symmetric width,
    whether that move would be a regression or an improvement. The move is
    reported in `banded` rather than hidden. A band cannot reclassify a
    non-integer value or a move outside the band, and the fixture parser rejects
    bands on badness counters whose healthy value must remain exactly zero.

    A field missing from either side reads as "0", so a baseline recorded before
    a counter existed still matches a run that reports it as 0, and adding an
    invariant counter costs no re-record. The cost of that convenience is that a
    field absent from BOTH sides compares equal forever: a backend whose
    [jit-stats] line stops naming a counter disarms that counter's gate on every
    one of its baselines, silently. Whenever a backend's line changes shape,
    re-record its whole baseline surface rather than trusting the run that
    follows."""
    bands = bands or {}
    old_fields = _parse_jit_stats(saved)
    new_fields = _parse_jit_stats(current)
    regressions, improvements, banded = [], [], []
    for field in JITSTATS_SNAPSHOT_FIELDS:
        old, new = old_fields.get(field, "0"), new_fields.get(field, "0")
        if old == new:
            continue
        try:
            old_int, new_int = int(old), int(new)
        except ValueError:
            # A counter that stopped being an integer is not a gain.
            rose = None
        else:
            if field in bands and abs(new_int - old_int) <= bands[field]:
                banded.append(
                    f"{field} {old} -> {new} (within band {bands[field]})"
                )
                continue
            rose = new_int > old_int
        if rose is not None and (
            (rose and field in JITSTATS_REGRESSION_ON_FALL)
            or (not rose and field in JITSTATS_REGRESSION_ON_RISE)
        ):
            improvements.append(f"{field} {old} -> {new}")
        else:
            regressions.append(f"{field} {old} -> {new}")
    return regressions, improvements, banded


def _jit_stats_vacuous(stderr):
    """Return why a gated badness counter is vacuous this run, or "".

    A badness counter's zero means "clean" only when its source census ran.
    Require a nonzero denominator without baselining its host-dependent value.
    This detects total loss of a census, not a producer that merely omits some
    census updates.
    """
    fields = _jit_stats_merged(stderr)
    if fields is None:
        return ""  # the caller already failed this run for the missing line
    for numerator, denominator in JITSTATS_DENOMINATOR_FIELDS.items():
        raw = fields.get(denominator)
        if raw is not None and raw.isdigit() and int(raw) > 0:
            continue
        return (
            f"{denominator}={raw if raw is not None else '<absent>'} — "
            f"the census behind `{numerator}` did not run, so that counter's 0 "
            f"means 'nothing was checked', not 'nothing is misplaced', and its "
            f"gate passed without inspecting anything"
        )
    return ""


def _jit_stats_context(current):
    """Return " (observed loops_compiled=N bridges_compiled=M)", or "".

    Report what the run compiled alongside what moved. A `guard_failures` change
    costs nothing to explain when the same run also compiled more loops or
    bridges — each new guard fails `trace_eagerness` times before its bridge
    attaches — but a gate that names only the fields that moved leaves the
    reader unable to tell that case from a real defect without re-running the
    bench. On a platform the developer does not have (the CI runners),
    re-running is not an option and the row is unjudgeable without this.
    """
    fields = _parse_jit_stats(current)
    context = " ".join(
        f"{f}={fields[f]}"
        for f in ("loops_compiled", "bridges_compiled")
        if f in fields
    )
    return f" (observed {context})" if context else ""


def scaled_timeout(base, scale):
    v = base * scale
    return int(v) if v == int(v) else float(f"{v:.3f}".rstrip("0").rstrip("."))


def fmt_time(t):
    if t is None or t == "-":
        return "-"
    if isinstance(t, (int, float)):
        return f"{t:.2f}s"
    try:
        return f"{float(t):.2f}s"
    except ValueError:
        return t


HEADER_SCAN_LINES = 20


def _header_lines(path):
    """The window a `# pyre-check:` directive has to sit in to be honored.

    Every reader below scans these lines and no further. A directive past the
    window is not read at all, which is the safe direction for all of them:
    the fixture keeps the ungated, unexempted behaviour instead of quietly
    acquiring an allowance nobody reviewed. Every directive that carves out an
    allowance also has to be followed by a comment saying why, so the reason
    is reviewable next to the workload.
    """
    with open(path, encoding="utf-8") as source:
        for _ in range(HEADER_SCAN_LINES):
            line = source.readline()
            if not line:
                return
            yield line


def _header_directive(path, prefix):
    """The first header line carrying `prefix`, as `(value, line)`, else None.

    First one wins: a second copy of the same directive is not an error for
    the readers that use this, it is simply not read.
    """
    for line in _header_lines(path):
        if line.startswith(prefix):
            return line[len(prefix):].strip(), line
    return None


def _positive_float(raw, line, path, kind):
    """A directive value that has to parse as a finite number above zero.

    `float()` accepts `inf` and `nan`, and neither is a threshold: a reading
    compares below `inf` forever, and every comparison against `nan` is False,
    so either one disarms the gate it configures while still looking like a
    number in the header. `<= 0` does not catch them — `nan <= 0` is False —
    so the finiteness test is separate.
    """
    try:
        value = float(raw)
    except ValueError as e:
        raise ValueError(f"invalid {kind} in {path}: {line.strip()}") from e
    if not math.isfinite(value):
        raise ValueError(f"{kind} must be finite in {path}: {line.strip()}")
    if value <= 0:
        raise ValueError(f"{kind} must be positive in {path}")
    return value


def _header_name_list(path, prefix, allowed, noun, exemption, reject=(), reject_noun=""):
    """A comma-separated directive value, every name checked against `allowed`.

    An unknown name is an error rather than a silent no-op: a typo would
    otherwise read as "exempted", which is the one failure mode an exemption
    directive must not have. `reject` names members of `allowed` that this
    particular directive still may not exempt.
    """
    found = _header_directive(path, prefix)
    if found is None:
        return ()
    raw, line = found
    names = [name for name in (part.strip() for part in raw.split(",")) if name]
    if not names:
        raise ValueError(f"empty {exemption} in {path}: {line.strip()}")
    unknown = [name for name in names if name not in allowed]
    if unknown:
        raise ValueError(
            f"unknown {noun}(s) {unknown} in {exemption} in {path}: {line.strip()}"
        )
    banned = [name for name in names if name in reject]
    if banned:
        raise ValueError(
            f"{reject_noun}(s) {banned} cannot be exempted in {path}: {line.strip()}"
        )
    return tuple(names)


def synth_perf_gate(path):
    """Read an optional per-fixture performance limit from its header.

    Synthetic tests own their limits, for example:
        # pyre-check: max-pypy-ratio=30
    Keeping the gate beside the workload makes a changed loop count or known
    slow path reviewable with the test that needs the allowance.

    The limit is read against the native backends only; `run_synthetic_bench`
    exempts wasm.

    A fixture states its ceiling and nothing else: the floor that goes with it
    is `perf_gate_floor`'s business, so a leftover `min-pypy-ratio` header is
    rejected rather than ignored.
    """
    prefix = "# pyre-check: max-pypy-ratio="
    retired = "# pyre-check: min-pypy-ratio="
    ratio = None
    for line in _header_lines(path):
        if line.startswith(retired):
            raise ValueError(
                f"{path} states a pypy floor: {line.strip()}\n"
                "the floor is derived from max-pypy-ratio; delete the line"
            )
        if line.startswith(prefix) and ratio is None:
            ratio = _positive_float(
                line[len(prefix):].strip(), line, path, "synthetic performance gate"
            )
    return ratio


def wasm_ratio_gate(path):
    """Read an optional per-fixture ceiling on wasm's ratio to dynasm:
        # pyre-check: max-wasm-ratio=6

    Absence means `WASM_MAX_DYNASM_RATIO`, not "no ceiling" -- the opposite of
    `synth_perf_gate` and `synth_rss_gate`, whose absence exempts a fixture
    entirely. A directive here is therefore an allowance carved out of a gate
    that already applies, so a run names every fixture that used one: an
    allowance that is quietly the reason a suite is green is worse than no gate.

    Read for every bench rather than synthetic ones alone, because the fixtures
    that need an allowance are not all synthetic.

    Fixtures carrying one must record their structural reason. Historically
    that was the wasm backend reaching CPython-level objects through interpreter
    round-trips and allocating each on a Rust heap the wasm path does not yet
    collect. Function-entry-heavy fixtures can instead be dominated by wasm's
    one-module-per-trace compilation tax after an orthodox portal change; their
    headers must pin the corresponding PyPy topology so an allowance cannot
    hide over-tracing. The allowances are set at the highest ratio observed
    plus 15% (`WASM_RATIO_FIT_HEADROOM`, which `print_summary` also requires
    before it will call an allowance outgrown), and each fixture records the
    ratios it was fitted to. ubuntu-24.04 is the only runner that builds wasm,
    so it supplies most of them; a fixture sitting on the gate locally is fitted
    here instead, which is why an allowance can name a host no CI run measures.

    15% because the ratio moves with host load even though both sides are
    user-CPU and measured in the same invocation: fannkuch reads 2.9x on an
    idle machine here and 3.1x on the same machine under a load average of 41.
    An allowance fitted to an idle reading would be a gate that fails on a busy
    runner.

    It is a p90 and not a safety factor, which this paragraph got wrong while
    it cited "~3% between runs": over the 273 fixtures two ubuntu runs of one
    base both gate, |delta|/max has median 4.7%, p75 8.8%, p90 14.9% and a tail
    to 67% (mapdict_frozen_unboxing_fold 1.79x then 0.59x). An allowance that
    clears its fixture's highest reading by less than 15% is fitted to one
    sample rather than to the instrument.
    """
    found = _header_directive(path, "# pyre-check: max-wasm-ratio=")
    if found is None:
        return None
    return _positive_float(*found, path, "wasm ratio gate")


def perf_gate_floor(ceiling):
    """The pypy ratio a bench with this ceiling must not read below.

    One fourth of the ceiling when that fourth is at least 0.75, and no floor
    otherwise. A ceiling of 20 keeps the reading in 5x..20x, a ceiling of 3
    keeps it in 0.75x..3x, and a ceiling under 3 has no floor. A reading faster
    than pypy does not fail a bench whose ceiling is under 3.
    """
    floor = ceiling / PERF_GATE_FLOOR_DIVISOR
    if floor < PERF_GATE_FLOOR_RATIO:
        return None
    return floor


def synth_rss_gate(path):
    """Read an optional per-fixture peak-RSS ceiling from its header:
        # pyre-check: max-rss-mb=200

    A second axis beside `max-pypy-ratio`, not a replacement: the ratio gate
    is blind to memory, so a change that trades retained bytes for speed
    passes it while regressing badly. Frames are the standing example —
    every interpreted call retains an old-gen frame, and nothing in this
    harness has ever measured that.

    Absent directive means no ceiling, so adding this gate cannot fail a
    fixture that does not opt in. Read against native backends only, like
    the ratio gate.
    """
    found = _header_directive(path, "# pyre-check: max-rss-mb=")
    if found is None:
        return None
    return _positive_float(*found, path, "synthetic memory gate")


def synth_skip_platforms(path):
    """Read an optional host exemption:
        # pyre-check: skip-platforms=win32

    The named `sys.platform` values never start the fixture. Use it when the
    oracle itself cannot run there, not to hide a pyre failure. The header
    line must be followed by a comment naming the missing host facility.
    """
    return _header_name_list(
        path,
        "# pyre-check: skip-platforms=",
        ("darwin", "linux", "win32"),
        "platform",
        "synthetic platform exemption",
    )


def synth_skip_backends(path):
    """Read an optional per-fixture backend exemption from its header:
        # pyre-check: skip-backends=wasm

    For a fixture that guards a mechanism a backend does not implement at
    all, so the fixture can only ever restate that known gap. The named
    backends print `skip` instead of running, the way
    `run_bench(skip_backends=...)` already does.

    This is not a way to park a failure: the header line must be followed by
    a comment saying which mechanism is missing. A `pyre-module` module the
    binary was built without is not a backend mechanism; that fixture says
    `requires-modules=` instead (`synth_requires_modules`).

    The exemption covers a backend's *execution* only. It deliberately does
    NOT cover the three baseline-failure paths below (cpython crash, pypy
    crash, cpython/pypy output mismatch): those mean the fixture itself is
    broken, and that must stay loud on every backend rather than be hidden
    behind an exemption written for a different reason.
    """
    return _header_name_list(
        path,
        "# pyre-check: skip-backends=",
        ALL_BACKENDS,
        "backend",
        "synthetic backend exemption",
    )


# The builtin modules `pyre-module` owns, one per entry under its `module/`
# directory. A binary built without that crate (`build-jit-core.sh`, the
# cranelift leg) has none of them.
PYRE_MODULE_NAMES = tuple(
    sorted(
        entry.stem
        for entry in (
            Path(__file__).resolve().parent / "pyre-module" / "src" / "module"
        ).iterdir()
        if entry.is_dir() or (entry.suffix == ".rs" and entry.stem != "mod")
    )
)


# Leads the line the `sys.builtin_module_names` probe prints, so the answer is
# told apart from anything else the binary writes to stdout.
BUILTIN_MODULES_MARKER = "pyre-check-builtin-modules:"


def synth_requires_modules(path):
    """Read the `pyre-module` modules a fixture imports from its header:
        # pyre-check: requires-modules=math

    Whether a binary has them depends on how it was built, not on its
    backend, so `Check._fixture_skips` asks each binary's
    `sys.builtin_module_names` and skips the fixture only where a named
    module is missing. The same fixture runs on every backend built with
    `pyre-module`.

    A name must be one `pyre-module` owns: a misspelt one would be missing
    from every binary and would silently skip the fixture everywhere.
    """
    return _header_name_list(
        path,
        "# pyre-check: requires-modules=",
        PYRE_MODULE_NAMES,
        "pyre-module module",
        "synthetic module requirement",
    )


def synth_jitstats_bands(path):
    """Read an optional per-fixture jit-stats band from its header:
        # pyre-check: jitstats-band=guard_failures=8

    The band is symmetric around the recorded baseline and absorbs moves in
    both directions: a delta within the band is not signal either way. An
    absorbed move is reported on that run rather than swallowed, so the
    allowance is visible whenever it is used and not only in the fixture
    header. It is only for schedule-sensitive counters; badness counters must
    remain exactly zero. The directive must be followed by a comment describing
    the measured variance. A repeated directive, and a duplicate, unknown or
    non-positive field, are errors.
    """
    prefix = "# pyre-check: jitstats-band="
    bands = None
    for line in _header_lines(path):
        if not line.startswith(prefix):
            continue
        if bands is not None:
            raise ValueError(f"duplicate jit-stats band directive in {path}: {line.strip()}")
        bands = {}
        for entry in line[len(prefix):].strip().split(","):
            parts = entry.split("=")
            if len(parts) != 2 or not all(part.strip() for part in parts):
                raise ValueError(f"invalid jit-stats band in {path}: {line.strip()}")
            field, raw_width = (part.strip() for part in parts)
            if field not in JITSTATS_SNAPSHOT_FIELDS:
                raise ValueError(
                    f"unknown jit-stats band field {field!r} in {path}: {line.strip()}"
                )
            if field in JITSTATS_BADNESS_FIELDS:
                raise ValueError(f"badness counter cannot be banded in {path}: {line.strip()}")
            if field in bands:
                raise ValueError(
                    f"duplicate jit-stats band field {field!r} in {path}: {line.strip()}"
                )
            try:
                width = int(raw_width)
            except ValueError as e:
                raise ValueError(f"invalid jit-stats band width in {path}: {line.strip()}") from e
            if width <= 0:
                raise ValueError(
                    f"jit-stats band width must be positive in {path}: {line.strip()}"
                )
            bands[field] = width
    return bands if bands is not None else {}


EXPECTED_PREFIX = "# Expected"

# What an `# Expected:` claim may name. The corpus states its expectations as
# arithmetic over the fixture's own sizes, so the evaluator has to reach the
# aggregate builtins those use and nothing else.
EXPECTED_BUILTINS = {
    "abs": abs, "divmod": divmod, "float": float, "int": int, "len": len,
    "max": max, "min": min, "range": range, "round": round, "sum": sum,
}

# The node kinds an expectation is allowed to be built from: literals, the
# names bound above or by the fixture, calls to those names, comprehensions,
# and arithmetic. Anything else -- an attribute, a lambda, a subscript of a
# name the fixture did not bind -- is not an expectation, it is prose.
_EXPECTED_NODES = (
    ast.Expression, ast.Constant, ast.Name, ast.Load, ast.Call, ast.keyword,
    ast.Tuple, ast.List, ast.Set, ast.Dict, ast.Starred,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.GeneratorExp, ast.ListComp, ast.SetComp, ast.DictComp, ast.comprehension,
    ast.operator, ast.unaryop, ast.boolop, ast.cmpop,
)


def _expected_constants(path):
    """The fixture's own module-level numeric constants.

    An expectation is worth more when it derives from the size the fixture
    actually runs -- `2 * N` stays true when `N` moves, and a literal does
    not. Only plain `NAME = <number>` bindings are read; anything computed is
    left out rather than partially evaluated.
    """
    constants = {}
    try:
        module = ast.parse(Path(path).read_text(encoding="utf-8"))
    except SyntaxError:
        return constants
    for node in module.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        value = node.value
        if isinstance(target, ast.Name) and isinstance(value, ast.Constant):
            if isinstance(value.value, (int, float)) and not isinstance(value.value, bool):
                constants[target.id] = value.value
    return constants


def _expected_value(text, constants):
    """`text` evaluated as an expectation, or None when it is not one."""
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, _EXPECTED_NODES):
            return None
        if isinstance(node, ast.Name) and node.id not in EXPECTED_BUILTINS \
                and node.id not in constants:
            return None
        # `4000 ('i', 'marker')` parses as a call on a literal. Only a name
        # can be called here, so rejecting it keeps that shape prose rather
        # than compiling something that can only fail at eval.
        if isinstance(node, ast.Call) and not isinstance(node.func, ast.Name):
            return None
    try:
        return eval(  # noqa: S307 - the allowlist above is the guard
            compile(tree, "<expected>", "eval"),
            {"__builtins__": {}, **EXPECTED_BUILTINS, **constants},
        )
    except Exception:
        return None


def _expected_claims(path):
    """The lines a fixture states its own stdout will contain.

        # Expected: 200010000
        # Expected output: (40000, 40000, 40000)

    A claim whose label ends the line introduces a block, one claim per
    following indented comment line; otherwise the following indented comment
    lines continue the claim they follow, which is how the corpus already
    wraps its longer literals.

        # Expected output:
        #   4000 ('i', 'marker')
        #   4000 ('i', 'marker', 'odd_only')

    Returns `(claim, line)` pairs. Every `# Expected` line yields at least one
    claim: a line that names nothing checkable is the rot this reader exists
    to catch, so it is reported rather than skipped.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    claims = []
    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        if not line.startswith(EXPECTED_PREFIX):
            continue
        head = line[len(EXPECTED_PREFIX):]
        text = head.split(":", 1)[1].strip() if ":" in head else ""
        continuation = []
        while index < len(lines) and lines[index].startswith("#  "):
            continuation.append(lines[index].lstrip("#").strip())
            index += 1
        if text:
            claims.append((" ".join([text, *continuation]).strip(), line))
        else:
            claims.extend((entry, line) for entry in continuation)
            if not continuation:
                claims.append(("", line))
    return claims


def _expected_mismatches(path, output):
    """The fixture's `# Expected` claims that its own output does not bear out.

    Each claim names one whole line of stdout, matched with internal
    whitespace collapsed so a claim may stay aligned inside its comment
    block. A claim spelled as arithmetic ending in its own result -- the
    `20000 * 90 = 1800000` shape the corpus uses -- is taken at that result,
    and one spelled as an expression is evaluated against the fixture's own
    constants.
    """
    claims = _expected_claims(path)
    if not claims:
        return []
    printed = {" ".join(line.split()) for line in output.splitlines() if line.strip()}
    constants = _expected_constants(path)
    mismatches = []
    for claim, line in claims:
        candidates = [claim]
        # `a = b = value`: the corpus states the arithmetic and then its
        # result. `==` and the ordering comparisons are not that shape.
        tail = re.split(r"(?<![=!<>])=(?!=)", claim)[-1].strip()
        if tail != claim:
            candidates.append(tail)
        for candidate in candidates:
            normalized = " ".join(candidate.split())
            if normalized in printed:
                break
            value = _expected_value(candidate, constants)
            if value is not None and str(value) in printed:
                break
        else:
            mismatches.append((claim, line))
    return mismatches


def _synth_header_flag(path, directive, malformed):
    """Read a valueless per-fixture header flag.

    Shared by the cpython opt-outs below, which differ only in the directive
    they spell and in what a malformed spelling is called.
    """
    for line in _header_lines(path):
        if line.strip() == directive:
            return True
        if line.startswith(directive):
            raise ValueError(f"{malformed} in {path}: {line.strip()}")
    return False


def synth_skip_cpython(path):
    """Read an optional per-fixture cpython opt-out from its header:
        # pyre-check: skip-cpython

    For a fixture sized past what cpython can usefully run. The ratio gate is
    pypy's; cpython is only a reference, and once the workload is big enough
    for pypy's execution time to be a measurement rather than a clock tick,
    cpython can be minutes behind. `SYNTHETIC_CPYTHON_REFERENCE_TIMEOUT_S`
    already drops it in that case, but only after spending that timeout on
    every run — this says so up front.

    What it gives up is the cpython/pypy output cross-check, which is the one
    thing that catches a fixture whose two baselines disagree. So the header
    line must be followed by a comment saying why the size is what it is,
    the way `skip-backends` does, and a fixture cpython can still run has no
    business carrying it.
    """
    return _synth_header_flag(
        path,
        "# pyre-check: skip-cpython",
        "synthetic cpython opt-out takes no value",
    )


def synth_no_cpython(path):
    """Read an optional per-fixture exemption from the reference oracle:
        # pyre-check: no-cpython

    A synthetic fixture is normally rejected outright when the reference and
    pypy disagree, because a workload the two do not agree on cannot say
    anything about pyre. That leaves a DELIBERATE divergence from the
    reference untestable anywhere in the tree: a parity fixture has to pass
    under the reference as well, and the vendored suite measures a divergence
    as a regression rather than as coverage. So behaviour pyre takes from pypy
    precisely because the reference does something else — `type.__basicsize__`
    and its three siblings being absent, `__flags__` carrying pypy's six-bit
    mask, `__sizeof__` not existing — had nowhere to be pinned.

    This header names pypy as the fixture's only oracle: the reference is not
    run, and the output is compared against pypy alone. It is only correct for
    a fixture whose whole subject IS such a divergence; anything else wants
    both references, and the disagreement it would otherwise report is the
    point of that check. Follow the header with a comment naming the
    divergence.
    """
    return _synth_header_flag(
        path,
        "# pyre-check: no-cpython",
        "malformed reference-oracle exemption",
    )


def synth_selfcheck(path):
    """Read an optional self-checking fixture marker from its header:
        # pyre-check: selfcheck

    These fixtures assert their own invariant and print ``PASS`` instead of
    producing stable output for the cpython/pypy parity comparison.  Keeping
    the marker on the fixture lets the synthetic suite discover it normally
    without a second, hard-coded list in ``main``.
    """
    return _synth_header_flag(
        path,
        "# pyre-check: selfcheck",
        "synthetic selfcheck marker takes no value",
    )


def synth_selfcheck_interpreted(path):
    """Read an optional per-fixture JIT-floor opt-out from its header:
        # pyre-check: selfcheck-interpreted

    A selfcheck fixture asserts its own invariant, so it passes whenever the
    invariant holds — including on a run where the JIT compiled nothing at all,
    which for a guard against a *compiled* mis-admission is a pass that
    establishes nothing. `run_selfcheck` therefore requires the run to have
    compiled what the fixture names, and this is how a fixture says its
    invariant is not about compiled code: the three that carry it
    (`oserror_errno_fields_regression`, `posix_replace_regression`,
    `settrace_local_tracer_armed_before_entry`) guard interpreter-level
    behaviour and compile nothing their assertion is about.

    It declares the empty set rather than turning a threshold off: a
    bookkeeping loop that happens to compile is neither required nor
    forbidden. A fixture that does reach the JIT, but only as a
    FINISH-only entry bridge, declares
    `selfcheck-compiles=entry-bridge:<name>` rather than this.

    An opt-out rather than an opt-in, so a fixture that quietly stops being
    compiled is reported rather than passing on in silence.
    """
    return _synth_header_flag(
        path,
        "# pyre-check: selfcheck-interpreted",
        "synthetic selfcheck-interpreted marker takes no value",
    )


def synth_selfcheck_compiles(path):
    """Read what a selfcheck fixture requires the JIT to compile:
        # pyre-check: selfcheck-compiles=hot,inner
        # pyre-check: selfcheck-compiles=entry-bridge:callee_d1

    Each entry is a code object name — what `get_printable_location`
    (interp_jit.py) puts first in the string it renders for a green key —
    optionally prefixed by the compile arm that has to mint it. A bare name
    means `loop`.

    The arm is part of the declaration because `loops_compiled` is not a count
    of loops: measured on `wrapper_subclass_load_method_self`, eight bumps are
    seven `loop` arms and one `root` (`majit_metainterp::loop_census` says
    which arm is which). So `root:` is not a weaker form of the same claim,
    and a fixture whose loop degrades into one is still reported.

    This replaces a bare number, `selfcheck-loops=8`, read off
    `loops_compiled`, which was also a whole-process figure — any hot loop in
    the file cleared the floor, which is why
    `settrace_local_tracer_armed_before_entry` had to count its own events
    with `list.count` instead of a `for` loop.
    """
    found = _header_directive(path, "# pyre-check: selfcheck-compiles=")
    if found is None:
        raise ValueError(
            f"{path} is a selfcheck fixture that does not say what it needs "
            "compiled. Run it with PYRE_LOOP_CENSUS=1, read the "
            "`[loop-census] <arm> <name>` lines, and declare the shapes the "
            "guard is about:\n"
            "    # pyre-check: selfcheck-compiles=<name>[,<arm>:<name>...]\n"
            "A bare name means the `loop` arm. A fixture whose invariant is "
            "not about compiled code carries "
            "`# pyre-check: selfcheck-interpreted` instead."
        )
    raw, line = found
    entries = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        arm, _, name = part.rpartition(":")
        arm = arm or "loop"
        if name.isdigit():
            raise ValueError(
                f"{path} declares a compiled-trace COUNT, not a name: "
                f"{line.strip()}\n"
                "`selfcheck-compiles` names the shapes that must compile. Run "
                "the fixture with PYRE_LOOP_CENSUS=1 and read the "
                "`[loop-census]` lines for the arms and names."
            )
        if arm not in SELFCHECK_COMPILE_ARMS:
            raise ValueError(
                f"unknown compile arm {arm!r} in {path}: {line.strip()}\n"
                "The arms the census emits are: "
                + ", ".join(sorted(SELFCHECK_COMPILE_ARMS))
            )
        entries.append((arm, name))
    if not entries:
        raise ValueError(f"empty compiled-trace list in {path}: {line.strip()}")
    return tuple(entries)


def synth_ungated_jitstats(path):
    """Read an optional per-fixture jit-stats exemption from its header:
        # pyre-check: ungated-jitstats=bridges_compiled,guard_failures

    For a fixture whose named counters are demonstrably not a function of the
    tree — the same binary, the same fixture and the same child environment
    read two different values. The names are dropped from both sides of the
    comparison and from the recorded baseline, so the file carries no number
    nobody checks, which is the rule the recorded surface is built on.

    Only the structural counters may be named. The badness fields are
    assertions that must hold in every run (`internal_compile_panics` is an
    internal compile bug, the descr-universe counters are invariants upstream
    spells as `assert`), and a run that reaches one has a defect no
    per-run input excuses, so naming one here is an error rather than an
    exemption.

    The header must carry what was measured, so the next reader can check the claim
    rather than take it.

    Read from `_apply_snapshot_gate`, so unlike the rest of this family it
    reaches the regular benches too — the gate it exempts is the one they
    share, and a counter that will not reproduce is not a synthetic-only
    condition.
    """
    return _header_name_list(
        path,
        "# pyre-check: ungated-jitstats=",
        JITSTATS_SNAPSHOT_FIELDS,
        "counter",
        "jit-stats exemption",
        reject=JITSTATS_BADNESS_FIELDS,
        reject_noun="badness counter",
    )


def synth_spec_folds(path):
    """Read an optional list of trace-time folds a fixture must make fire:
        # pyre-check: spec-folds=setslice,newlist

    A throughput ceiling cannot gate one fold out of several. The legs of a
    multi-fold fixture sum into one number, so retiring the least valuable one
    moves that number by less than the span the same fixture reads across the
    runners (`PERF_GATE_FLOOR_DIVISOR` records how wide that is). A jit-stats
    baseline covers only part of the set: suppressing one fold of a
    multi-fold fixture can move no gated counter at all.

    So this gates coverage instead, which is exact and the same on every host:
    each named fold has to fire at least once. That is the failure the other
    two instruments are blind to -- a fold that stops firing reads exactly
    like a fold nobody wrote a leg for, and `builtin_zip` sat at `fired=0` in
    the fixture written to gate it for as long as it read its `call_kw`
    operands in the wrong order.

    It does not claim the folds are worth their code; it claims the fixture
    still reaches them. Pair it with the ceiling, which gates the sum.
    """
    found = _header_directive(path, "# pyre-check: spec-folds=")
    if found is None:
        return ()
    raw, line = found
    names = tuple(n for n in (part.strip() for part in raw.split(",")) if n)
    if not names:
        raise ValueError(f"empty fold list in {path}: {line.strip()}")
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise ValueError(f"repeated fold label(s) {repeated} in {path}: {line.strip()}")
    return names


def synth_trace_shapes(path):
    """Read the optimized-trace shape a fixture's named loop must have:
        # pyre-check: trace-shape=main:absent=CallMayForceR,present=IntAdd
        # pyre-check: trace-shape=entry-bridge:leaf:max=GuardTrue:2

    The loop label is the code-object name `selfcheck-compiles` already uses:
    the first field of a `[loop-census] <arm> <name>` line. A bare name means
    the `loop` arm. `entry-bridge:<name>:...` (and the other census arms)
    select that compile.

    The body is the `jit-log-opt-loop` section `MAJIT_LOG` dumps for that
    compile — the last such section before the census line, which is the
    optimized trace rather than the peeled assembly dumped ahead of it.
    `PYRE_LOOP_CENSUS` names the compile. Both variables already exist.

    An assertion is `absent=<token>`, `present=<token>` or `max=<token>:<N>`.
    A token is an opcode spelled the way the dump prints it (`IntAdd`,
    `CallMayForceR`) or any other text on that operation line (a residual
    helper, an oopspec, a callee path). Every compiled trace of the label
    has to satisfy every assertion. A missing trace, a trace with no
    operations, or a trace dumped as an op-count table because it was too
    long to print, fails: each of those is indistinguishable from a trace
    that truly lacks the operation.
    """
    found = _header_directive(path, "# pyre-check: trace-shape=")
    if found is None:
        return ()
    raw, line = found
    try:
        groups = _parse_trace_shape(raw)
    except ValueError as e:
        raise ValueError(f"invalid trace-shape in {path}: {line.strip()} ({e})") from e
    return groups


def synth_fixture_headers(path):
    """Every directive `path` owes, read in one pass.

    Which of the readers above apply depends on the fixture's kind, so that
    rule lives here rather than being restated at each consumer. Raises
    `ValueError` on this fixture's first unusable directive.
    """
    selfcheck = synth_selfcheck(path)
    headers = {
        "selfcheck": selfcheck,
        "skip_backends": synth_skip_backends(path),
        "requires_modules": synth_requires_modules(path),
        "skip_platforms": synth_skip_platforms(path),
        "spec_folds": synth_spec_folds(path),
        "trace_shapes": synth_trace_shapes(path),
    }
    if selfcheck:
        interpreted = synth_selfcheck_interpreted(path)
        headers["interpreted"] = interpreted
        headers["want_compiles"] = (
            () if interpreted else synth_selfcheck_compiles(path)
        )
    else:
        headers["max_pypy_ratio"] = synth_perf_gate(path)
        headers["max_rss_mb"] = synth_rss_gate(path)
        headers["skip_cpython"] = synth_skip_cpython(path)
        headers["no_cpython"] = synth_no_cpython(path)
        # `_apply_snapshot_gate` reads these two per backend and nothing there
        # catches a `ValueError`, so an unusable value in either arrived as a
        # traceback out of a backend run. Read here for the raise alone; the
        # gate still reads them itself, once it knows which backend it is.
        synth_ungated_jitstats(path)
        synth_jitstats_bands(path)
    return headers


def report_unusable_headers(errors):
    """Name every fixture whose header could not be read, not just the first.

    `pyre/bench/synth/` is shared, so a directive this script requires reds
    every fixture that arrives without it, and stopping at the first one costs
    a CI run per name. Fixtures that fail the same way share a block, so the
    guidance the reader wrote is printed once instead of once per fixture.
    """
    groups = {}
    for path, message in errors:
        groups.setdefault(message.replace(str(path), "<fixture>"), []).append(path)
    print(f"{red('ERROR')}: {len(errors)} fixture(s) with a header this suite cannot read")
    for message, paths in groups.items():
        print()
        for path in paths:
            print(f"  {path}")
        for line in message.splitlines():
            print(f"    {line}")


def synthetic_bench_paths(pattern):
    """The fixtures *pattern* names under [`SYNTHETIC_BENCH_DIR`]."""
    paths = sorted(Path(SYNTHETIC_BENCH_DIR).glob(pattern))
    if not paths and not Path(pattern).suffix:
        paths = sorted(Path(SYNTHETIC_BENCH_DIR).glob(f"{pattern}.py"))
    return [p for p in paths if p.is_file() and p.suffix == ".py"]


def read_synthetic_headers(paths):
    """`({path: headers}, [(path, message)])` -- the readable and the rest."""
    headers, unusable = {}, []
    for path in paths:
        try:
            headers[path] = synth_fixture_headers(path)
        except ValueError as e:
            unusable.append((path, str(e)))
    return headers, unusable


def check_synthetic_headers(pattern):
    """`--check-headers`: read every fixture header and report, building nothing.

    This pass touches files and nothing else -- no binary, no reference
    interpreter, no cargo -- so it belongs beside `cargo fmt --check`, which
    every expensive job in the workflow already waits on. Run from the suite it
    cannot answer until the build it sits behind has finished, and an authoring
    error costs a whole CI run to hear about. Returns a process exit status.
    """
    paths = synthetic_bench_paths(pattern)
    if not paths:
        print(f"{red('ERROR')}: no synthetic benchmarks matched {pattern!r}")
        return 1
    unusable = read_synthetic_headers(paths)[1]
    if unusable:
        report_unusable_headers(unusable)
        return 1
    broken = _expected_reader_selftest()
    for case, detail in broken:
        print(f"{red('ERROR')}: the `# Expected` reader mis-reads {case}: {detail}")
    broken_shape = _trace_shape_reader_selftest()
    for case, detail in broken_shape:
        print(f"{red('ERROR')}: the trace-shape reader mis-reads {case}: {detail}")
    empty = [(path, line) for path in paths
             for claim, line in _expected_claims(path) if not claim.strip()]
    for path, line in empty:
        print(f"{red('ERROR')}: {path} states an expectation with nothing in it: {line.strip()}")
    if broken or broken_shape or empty:
        return 1
    claims = sum(len(_expected_claims(path)) for path in paths)
    print(
        f"{len(paths)} synthetic fixture header(s) read, {claims} expectation(s) "
        f"parsed, pattern={pattern!r}"
    )
    return 0


# `(label, comment lines, stdout, should the reader object)`. The corpus proves
# the reader accepts what the fixtures say; only a table can prove it still
# objects when one of them stops being true, which is the direction that lets an
# expectation rot. Runs in `--check-headers`, which builds nothing.
EXPECTED_READER_CASES = (
    ("a literal", "# Expected: 200010000\n", "200010000\n", False),
    ("a rotted literal", "# Expected: 200010000\n", "200010001\n", True),
    ("arithmetic and its result", "# Expected: 20000 * 90 = 1800000\n", "1800000\n", False),
    ("a rotted result", "# Expected: 20000 * 90 = 1800000\n", "1800001\n", True),
    ("an expression over a fixture constant", "# Expected: 2 * N\nN = 7\n", "14\n", False),
    ("a rotted expression", "# Expected: 2 * N\nN = 7\n", "15\n", True),
    ("a block", "# Expected output:\n#   a 1\n#   b 2\n", "a 1\nb 2\n", False),
    ("a rotted block line", "# Expected output:\n#   a 1\n#   b 2\n", "a 1\nb 3\n", True),
    ("a wrapped literal", "# Expected output: [1,\n#    2]\n", "[1, 2]\n", False),
    ("an aligned block line", "# Expected output:\n#   a    1\n", "a 1\n", False),
    ("prose", "# Expected: whatever the loop settles on\n", "5\n", True),
    ("an expectation with nothing in it", "# Expected:\n", "5\n", True),
    ("a name the fixture never bound", "# Expected: os.sep\n", "/\n", True),
)


def _expected_reader_selftest():
    """The cases `_expected_mismatches` reads wrongly, as `(label, detail)`."""
    broken = []
    with tempfile.TemporaryDirectory() as directory:
        fixture = Path(directory) / "fixture.py"
        for label, header, output, should_object in EXPECTED_READER_CASES:
            fixture.write_text(header, encoding="utf-8")
            objected = bool(_expected_mismatches(str(fixture), output))
            if objected != should_object:
                broken.append((
                    label,
                    "objected" if objected else "stayed silent",
                ))
    return broken


# `absent=<token>` / `present=<token>` / `max=<token>:<N>`. `\S+` stops at
# whitespace; the trailing `:digits` of `max` is split off by backtracking so
# a callee path may itself contain colons.
_TRACE_SHAPE_ASSERTION_RE = re.compile(
    r"^(absent|present)=(\S+)$|^(max)=(\S+):(\d+)$"
)
_TRACE_SHAPE_OP_LINE_RE = re.compile(
    r"^\s+(?:v\d+ = )?([A-Za-z_][A-Za-z0-9_]*)\("
)
_TRACE_SHAPE_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# debug.rs wire format: `[<hex ts>] {<category>` … `[<hex ts>] <category>}`.
_TRACE_SECTION_OPEN_RE = re.compile(r"^\[[0-9a-fA-F]+\] \{(\S+)\s*$")
_TRACE_SECTION_CLOSE_RE = re.compile(r"^\[[0-9a-fA-F]+\] (\S+)\}\s*$")
_TRACE_LOOP_CENSUS_RE = re.compile(r"^\[loop-census\] (\S+) (\S+)")
# pyjitpl.rs refuses to print an optimized trace past this and emits an
# op-count table instead. A count of zero over that table is not "the op is
# absent".
_TRACE_TRUNCATED_MARK = "[trace too large for full dump, showing op counts]"


def _split_trace_loop_spec(prefix):
    """`(arm, name)` for a loop label. A bare name is the `loop` arm."""
    if ":" in prefix:
        arm, name = prefix.split(":", 1)
        if arm not in SELFCHECK_COMPILE_ARMS:
            raise ValueError(f"unknown compile arm {arm!r}")
        if not name or ":" in name:
            raise ValueError(f"loop name {name!r} is not a single code-object name")
        return arm, name
    if not prefix:
        raise ValueError("missing loop name")
    return "loop", prefix


def _parse_trace_shape_assertion(text):
    """`(kind, token, limit)` — limit is None except for `max`."""
    match = _TRACE_SHAPE_ASSERTION_RE.fullmatch(text)
    if match is None:
        raise ValueError(f"bad assertion {text!r}")
    if match.group(1):
        kind, token = match.group(1), match.group(2)
        return kind, token, None
    return match.group(3), match.group(4), int(match.group(5))


def _parse_trace_shape(raw):
    """`(arm, name, assertions)` groups from one directive value.

    Each comma-separated piece is either `<loop>:<assertion>`, which starts a
    group, or a bare assertion, which continues the group just opened. The
    loop label may itself contain one colon (`<arm>:<name>`).
    """
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError("empty trace-shape")
    groups = []
    current = None
    for part in parts:
        found = None
        for kind in ("absent=", "present=", "max="):
            at = part.find(kind)
            if at != -1 and (found is None or at < found[0]):
                found = (at, kind)
        if found is None:
            raise ValueError(f"no assertion in {part!r}")
        at, _kind = found
        prefix = part[:at].rstrip(":")
        assertion = _parse_trace_shape_assertion(part[at:])
        if prefix:
            arm, name = _split_trace_loop_spec(prefix)
            current = [arm, name, [assertion]]
            groups.append(current)
        elif current is None:
            raise ValueError(f"assertion before its loop: {part!r}")
        else:
            current[2].append(assertion)
    return tuple(
        (arm, name, tuple(assertions))
        for arm, name, assertions in groups
    )


def _trace_token_hits(body, token):
    """How many operation lines name `token` as their opcode or on the line."""
    if _TRACE_SHAPE_IDENT_RE.fullmatch(token):
        bound = re.compile(
            r"(?<![A-Za-z0-9_])" + re.escape(token) + r"(?![A-Za-z0-9_])"
        )
        def on_line(line):
            return bound.search(line) is not None
    else:
        def on_line(line):
            return token in line
    hits = 0
    for line in body.splitlines():
        match = _TRACE_SHAPE_OP_LINE_RE.match(line)
        if match is None:
            continue
        if match.group(1) == token or on_line(line):
            hits += 1
    return hits


def _trace_op_lines(body):
    return sum(
        1 for line in body.splitlines() if _TRACE_SHAPE_OP_LINE_RE.match(line)
    )


def optimized_loop_traces(stderr):
    """`(arm, name) -> [body, ...]` for every compile the op log names.

    A compile dumps `jit-log-opt-loop` at least once (the peeled assembly,
    then the optimized trace) and afterwards prints one `[loop-census]` line.
    That line is inside the enclosing `jit-tracing` section, so it is not a
    top-level event. It takes the `jit-log-opt-loop` section that closed most
    recently, which is the optimized trace rather than the peeled assembly
    ahead of it. A census with no such section waiting is not a trace, and a
    census printed from inside the dump itself is not one either.
    """
    stack = []
    pending = None
    traces = {}
    for line in (stderr or "").splitlines():
        opened = _TRACE_SECTION_OPEN_RE.match(line)
        if opened:
            stack.append((opened.group(1), []))
            continue
        closed = _TRACE_SECTION_CLOSE_RE.match(line)
        if closed and stack and stack[-1][0] == closed.group(1):
            category, body = stack.pop()
            if category == "jit-log-opt-loop":
                pending = "\n".join(body)
            continue
        census = _TRACE_LOOP_CENSUS_RE.match(line)
        if (
            census
            and pending is not None
            and all(category != "jit-log-opt-loop" for category, _body in stack)
        ):
            key = (census.group(1), census.group(2))
            traces.setdefault(key, []).append(pending)
            pending = None
        if stack:
            stack[-1][1].append(line)
    return traces


def trace_shape_failures(stderr, groups):
    """The assertions in `groups` that the optimized traces do not meet."""
    traces = optimized_loop_traces(stderr)
    failures = []
    for arm, name, assertions in groups:
        bodies = traces.get((arm, name), [])
        label = name if arm == "loop" else f"{arm}:{name}"
        if not bodies:
            failures.append(f"trace-shape {label}: no optimized trace")
            continue
        for index, body in enumerate(bodies):
            where = label if len(bodies) == 1 else f"{label}#{index + 1}"
            if _TRACE_TRUNCATED_MARK in body:
                failures.append(f"trace-shape {where}: optimized trace truncated")
                continue
            if _trace_op_lines(body) == 0:
                failures.append(f"trace-shape {where}: optimized trace has no operations")
                continue
            for kind, token, limit in assertions:
                hits = _trace_token_hits(body, token)
                if kind == "absent" and hits:
                    failures.append(
                        f"trace-shape {where}: absent {token} (saw {hits})"
                    )
                elif kind == "present" and not hits:
                    failures.append(
                        f"trace-shape {where}: present {token} (saw 0)"
                    )
                elif kind == "max" and hits > limit:
                    failures.append(
                        f"trace-shape {where}: max {token}:{limit} (saw {hits})"
                    )
    return failures


def _trace_shape_reader_selftest():
    """Cases the trace-shape parser or the op-log reader gets wrong."""
    broken = []

    def check(label, cond):
        if not cond:
            broken.append((label, "mismatch"))

    try:
        parsed = _parse_trace_shape(
            "main:absent=CallMayForceR,present=IntAdd,max=GuardTrue:2"
        )
    except ValueError as e:
        parsed = e
    check(
        "one loop, three assertions",
        parsed == ((
            "loop", "main", (
                ("absent", "CallMayForceR", None),
                ("present", "IntAdd", None),
                ("max", "GuardTrue", 2),
            ),
        ),),
    )
    try:
        armed = _parse_trace_shape("entry-bridge:leaf:absent=foo::bar,max=GuardTrue:0")
    except ValueError as e:
        armed = e
    check(
        "an arm-qualified loop and a callee path",
        armed == ((
            "entry-bridge", "leaf", (
                ("absent", "foo::bar", None),
                ("max", "GuardTrue", 0),
            ),
        ),),
    )
    try:
        two = _parse_trace_shape("hot:absent=CallR,inner:present=IntAdd")
    except ValueError as e:
        two = e
    check(
        "two loops on one line",
        two == (
            ("loop", "hot", (("absent", "CallR", None),)),
            ("loop", "inner", (("present", "IntAdd", None),)),
        ),
    )
    for label, text in (
        ("empty", ""),
        ("no assertion", "main"),
        ("unknown arm", "not-an-arm:leaf:absent=CallR"),
        ("assertion before its loop", "absent=CallR"),
        ("max without a count", "main:max=GuardTrue"),
    ):
        try:
            _parse_trace_shape(text)
        except ValueError:
            continue
        broken.append((label, "accepted"))

    log = "\n".join([
        "[1] {jit-log-opt-loop",
        "--- peeled trace (assembled) ---",
        "  v1 = CallR(1) descr=<runtime_helper: BinaryOp>",
        "[2] jit-log-opt-loop}",
        "[3] {jit-log-opt-loop",
        "--- trace (after opt) --- [3 ops]",
        "  v2 = IntAdd(1, 2)",
        "  GuardTrue(v2)",
        "  Jump()",
        "[4] jit-log-opt-loop}",
        "[loop-census] loop main #14 FOR_ITER",
        "[5] {jit-log-opt-loop",
        "# Loop 2 : entry bridge with 1 ops",
        "  v0 = CallR(9) descr=<foo::bar>",
        "[6] jit-log-opt-loop}",
        "[loop-census] entry-bridge leaf #0 LOAD_FAST",
        "[9] {jit-tracing",
        "[a] {jit-log-opt-loop",
        "--- peeled trace (assembled) ---",
        "  v1 = CallR(1) descr=<runtime_helper: BinaryOp>",
        "[b] jit-log-opt-loop}",
        "[c] {jit-log-opt-loop",
        "--- trace (after opt) --- [1 ops]",
        "  v9 = GuardClass(v0)",
        "[d] jit-log-opt-loop}",
        "[loop-census] loop nested #4 LOAD_FAST",
        "[e] jit-tracing}",
    ])
    groups = (
        ("loop", "main", (
            ("absent", "CallR", None),
            ("absent", "BinaryOp", None),
            ("present", "IntAdd", None),
            ("max", "GuardTrue", 1),
        )),
        ("entry-bridge", "leaf", (
            ("present", "foo::bar", None),
            ("absent", "IntAdd", None),
        )),
    )
    check("fold-shaped trace passes", trace_shape_failures(log, groups) == [])
    # The peeled section named CallR and BinaryOp. Binding it would fail
    # `absent`, so a pass here is the proof the census took the later section.
    check(
        "Call is not CallR",
        trace_shape_failures(log, (
            ("loop", "main", (("absent", "Call", None), ("present", "CallR", None))),
        )) == ["trace-shape main: present CallR (saw 0)"],
    )
    check(
        "a max the trace exceeds",
        trace_shape_failures(log, (
            ("loop", "main", (("max", "GuardTrue", 0),)),
        )) == ["trace-shape main: max GuardTrue:0 (saw 1)"],
    )
    check(
        "a loop the census did not name",
        trace_shape_failures(log, (
            ("loop", "other", (("present", "IntAdd", None),)),
        )) == ["trace-shape other: no optimized trace"],
    )
    check(
        "a census nested in jit-tracing binds the optimized trace",
        trace_shape_failures(log, (
            ("loop", "nested", (
                ("present", "GuardClass", None),
                ("absent", "CallR", None),
                ("absent", "BinaryOp", None),
            )),
        )) == [],
    )
    truncated = "\n".join([
        "[1] {jit-log-opt-loop",
        "--- trace (after opt) --- [10001 ops]",
        _TRACE_TRUNCATED_MARK,
        "  IntAdd: 10",
        "[2] jit-log-opt-loop}",
        "[loop-census] loop main #4 LOAD_FAST",
    ])
    check(
        "a truncated dump is not an absent op",
        trace_shape_failures(truncated, (
            ("loop", "main", (("absent", "IntAdd", None),)),
        )) == ["trace-shape main: optimized trace truncated"],
    )
    empty_body = "\n".join([
        "[1] {jit-log-opt-loop",
        "--- trace (after opt) --- [0 ops]",
        "[2] jit-log-opt-loop}",
        "[loop-census] loop main #4 LOAD_FAST",
    ])
    check(
        "a banner with no operations is not an absent op",
        trace_shape_failures(empty_body, (
            ("loop", "main", (("absent", "IntAdd", None),)),
        )) == ["trace-shape main: optimized trace has no operations"],
    )

    with tempfile.TemporaryDirectory() as directory:
        fixture = Path(directory) / "fixture.py"
        fixture.write_text(
            "# pyre-check: trace-shape=hot:absent=CallMayForceR,present=IntAdd\n",
            encoding="utf-8",
        )
        check(
            "a header line",
            synth_trace_shapes(fixture) == ((
                "loop", "hot", (
                    ("absent", "CallMayForceR", None),
                    ("present", "IntAdd", None),
                ),
            ),),
        )
        buried = ["# a comment\n"] * HEADER_SCAN_LINES
        buried.append("# pyre-check: trace-shape=hot:absent=CallR\n")
        fixture.write_text("".join(buried), encoding="utf-8")
        check("a directive past the scan window", synth_trace_shapes(fixture) == ())
    return broken


# `[spec-census] fold=<label> consulted=N fired=N suppressed=N site=... parent=...`
SPEC_CENSUS_FOLD_RE = re.compile(r"^\[spec-census\] fold=(\S+) .*?\bfired=(\d+)\b", re.M)


def spec_fold_census(binary, path, timeout_s, wasm=False):
    """`({label: fired}, returncode)` for every fold the run registered.

    The binary owns the label set, so a declared name absent from this map is
    a typo rather than a fold that stayed quiet, and the caller reports the
    two differently.

    The wasm guest reads no environment, so `PYRE_FBW_SPEC_CENSUS` cannot
    reach it; `pyre-wasm-runner` arms the same census through the
    `pyre_fbw_spec_census_enable` export and prints the identical
    `[spec-census]` lines, so one parser serves both. Setting the name the
    other backend ignores would only earn the runner's inert-knob warning.
    """
    env = pyre_env()
    env["PYRE_WASM_SPEC_CENSUS" if wasm else "PYRE_FBW_SPEC_CENSUS"] = "1"
    _, _, code, err = run_timed([binary, path], timeout_s=timeout_s, env=env)
    if code != 0:
        # A census panic is a runtime failure, not a missing fold. Preserve
        # the evidence just as the ordinary benchmark failure path does.
        if err:
            print("\n─── spec-fold census stderr ───")
            print(err.rstrip())
        return None, code
    return {m.group(1): int(m.group(2)) for m in SPEC_CENSUS_FOLD_RE.finditer(err)}, 0


# The arms `majit_metainterp::loop_census` tags a compile with. Spelled out
# here so a typo in a fixture header is an authoring error rather than a shape
# that silently never matches.
SELFCHECK_COMPILE_ARMS = frozenset({"loop", "retrace", "root", "entry-bridge"})

SELFCHECK_LOOP_CENSUS_RE = re.compile(r"^\[loop-census\] (\S+) (\S+)", re.M)


def selfcheck_compiled_census(stderr):
    """`(kind, name)` for every trace `PYRE_LOOP_CENSUS=1` saw compiled.

    *kind* is the compile arm ([`SELFCHECK_COMPILE_ARMS`]); *name* is the code
    object the green key points at.

    One line per compiled trace, so a name repeats when a function is compiled
    more than once; the caller only asks whether a pair is present. A trace
    whose green key carries no code pointer prints `<unnamed>` rather than
    nothing, because a missing line and "nothing compiled" must not look alike.
    """
    return [m.groups() for m in SELFCHECK_LOOP_CENSUS_RE.finditer(stderr or "")]


def default_binary(backend):
    name = CARGO_CONFIG[backend]["bin"]
    return f"./target/release/{name}{EXE}"


# Repository-root files every crate is built against.
ROOT_BUILD_INPUTS = ("Cargo.toml", "Cargo.lock", ".cargo/config.toml", "rust-toolchain.toml")


def workspace_member_dirs():
    """Directories listed in the root `Cargo.toml` `members` array.

    A `.rs` file only reaches a compiler if it belongs to a member crate, so
    this is what separates a build input from a bench fixture or a baseline
    sitting elsewhere in the tree. It does not settle the whole input set: one
    of those sources can embed a file from anywhere in the tree, which is what
    [`embedded_inputs_outside_members`] recovers. `pyre/pyrex/tests/gate_triage_complete.rs`
    derives its own search roots the same way, for the same reason — including
    the line anchor, which is what keeps `default-members = [` above from
    being read as this array.
    """
    try:
        manifest = Path("Cargo.toml").read_text(encoding="utf-8")
    except OSError:
        # No readable manifest is no member list, which the caller already
        # treats as an unenumerable tree and fails open on. Letting the read
        # raise instead would abort the whole run on a traceback.
        return []
    listing = re.search(r"^\s*members\s*=\s*\[(.*?)\]", manifest, re.S | re.M)
    if not listing:
        return []
    members = []
    for member in re.findall(r'"([^"]+)"', listing.group(1)):
        # Cargo accepts a glob member. There is none today; expanding here
        # keeps a future one from being read as a literal directory name that
        # matches nothing and silently narrows the input set.
        if any(char in member for char in "*?["):
            members.extend(sorted(str(path) for path in Path().glob(member)))
        else:
            members.append(member)
    return members


# The package whose build script reads `MAJIT_MIR_FRONTEND_LLBC`. `cargo` runs a
# build script with its own package directory as the working directory, so that
# is what a relative entry in the override is relative to — not this script's.
LLBC_OVERRIDE_BASE = "pyre/pyre-jit-trace"


def llbc_input_paths():
    """The LLBC artefacts the JIT front end will actually read.

    `majit_translate::build_semantic_program_via_active_frontend` reads them from
    `MAJIT_MIR_FRONTEND_LLBC` (an OS path-list) before falling back to the
    workspace `build/llbc`, so a run under that override is built against
    different bytes than the default glob names.

    A relative entry is resolved against `LLBC_OVERRIDE_BASE` rather than taken
    as written. The reader is `pyre-jit-trace/build.rs`, and a build script's
    working directory is its own package, so `../../build/llbc/pyre-jit.ullbc`
    is the workspace artefact to the build and two levels above the repository
    to a caller resolving from the root. Taken as written it names a file that
    is not there, which the digest records as an unreadable input — the same
    value however often the real artefact is rewritten, so `--build=no` would
    approve a generated JIT front end built from an LLBC that has since moved.
    """
    override = os.environ.get("MAJIT_MIR_FRONTEND_LLBC")
    if override:
        return [
            entry if os.path.isabs(entry)
            else os.path.normpath(os.path.join(LLBC_OVERRIDE_BASE, entry)).replace(os.sep, "/")
            for entry in override.split(os.pathsep) if entry
        ]
    return [str(path) for path in Path("build/llbc").glob("*.ullbc")]


# `include_str!("x")`, `include_bytes!("x")` and `include!("x")` with a plain
# literal. A path built with `concat!`/`env!` is deliberately unmatched: those
# name `OUT_DIR`, which is under `target/` and so outside the tree this set is
# derived from.
EMBED_MACRO = re.compile(r'include(?:_str|_bytes)?!\s*\(\s*"([^"\\]*)"')


def embedded_inputs_outside_members(member_sources, members, listed):
    """Files a member crate embeds from outside every member directory.

    `include_str!` and its siblings read their file at compile time, so it is a
    build input exactly as the `.rs` around it is, and `rustc` records it in the
    depinfo that makes `cargo` rebuild when it changes. The path is written
    relative to the source file, so it can climb out of the crate and out of the
    workspace entirely: `majit-metainterp/src/ruleopt/mod.rs` embeds
    `rpython/jit/metainterp/ruleopt/real.rules`, which the member-directory
    filter alone drops. Left out, editing that file rebuilds the artefact
    without moving the fingerprint, and a later `--build=no` run approves a
    binary built from the previous text.

    Scanned rather than listed by hand, so a second one cannot be added to the
    tree without the gate seeing it.

    Conservative in one direction on purpose: an embed under `#[cfg(test)]`
    reaches the unit tests and not the measured artefact, and this counts it
    anyway rather than track cfgs. The cost is a rebuild nobody needed; the
    alternative fails the other way, which is the one that reports a
    measurement the binary does not support.
    """
    outside = set()
    for source in member_sources:
        if not source.endswith(".rs"):
            continue
        try:
            text = Path(source).read_text(encoding="utf-8", errors="replace")
        except OSError:
            # Raced with a delete, or unreadable. The file was in the listing a
            # moment ago, so dropping its embeds is no worse than the enclosing
            # enumeration already is about a tree changing under it.
            continue
        if "include" not in text:
            continue
        for embedded in EMBED_MACRO.findall(text):
            resolved = os.path.normpath(os.path.join(os.path.dirname(source), embedded))
            if resolved.startswith("..") or os.path.isabs(resolved):
                # Outside the repository, so no digest over the tree can cover
                # it; the enumeration says what it saw and nothing more.
                continue
            resolved = resolved.replace(os.sep, "/")
            if resolved.startswith(members) or resolved not in listed:
                continue
            outside.add(resolved)
    return sorted(outside)


def build_input_paths():
    """Every file a release artefact is built from, or `None` when the tree
    cannot be enumerated — which makes the freshness gate below fail open
    rather than block a checkout that has no git.

    Deliberately derived from the tree alone, never from `target/`. Reading the
    build scripts' own `cargo:rerun-if-changed=` declarations back out of
    `target/*/build/*/output` was tried and removed: the paths there are
    written relative to each crate's own directory, so all but one of the 57
    failed to resolve from the repository root and the whole mechanism
    contributed nothing but a dependence on which profiles and targets happened
    to have been built.

    Untracked files count, which is why the enumeration is not just
    `git ls-files`: a new `.rs` under a member crate compiles into the artefact
    before it is ever staged. Left out, it would contribute nothing to the
    digest, so editing it would not move the fingerprint and a later
    `--build=no` run would accept an artefact built from code the digest never
    saw. Worse, deleting it again would return the digest to its earlier value
    while the artefact still held its code. `--others --exclude-standard` adds
    exactly the untracked-and-not-ignored files, so `target/` and the rest of
    `.gitignore` stay out.

    One class is knowingly outside the set: an input a build script reads from
    outside its crate. The only one in the tree is the `lib-python/3` closure
    `pyre-interpreter/build.rs` embeds, and it is guarded by `wasm_vfs`, a
    feature no artefact this script measures is built with. Enabling it for the
    wasm-host build would mean adding that closure here. A file embedded by the
    ordinary sources rather than by a build script is a different matter and is
    covered — see [`embedded_inputs_outside_members`].
    """
    try:
        listing = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
    except OSError:
        return None
    if listing.returncode != 0:
        return None
    members = tuple(member + "/" for member in workspace_member_dirs())
    if not members:
        return None
    listed = listing.stdout.split("\0")
    # Every file under a member crate, whatever its suffix: the CJK codec
    # `.c`/`.h` sources and the app-level `.py` bodies reach the binary exactly
    # as the `.rs` files do.
    paths = [p for p in listed if p.startswith(members) or p in ROOT_BUILD_INPUTS]
    # A member crate can embed a file from outside every member directory, and
    # the filter above is by path alone, so it cannot see one.
    paths += embedded_inputs_outside_members(paths, members, frozenset(listed))
    # The LLBC is a build input the fingerprint gate already tracks by content,
    # but it is generated rather than tracked by git.
    paths += llbc_input_paths()
    return sorted(set(paths))


# Environment the build reads that this script does not spell in a constant.
#
# The first five mirror `pyre/pyre-jit-trace/build/prepass.rs` `LOWERING_GATE_ENV`,
# which folds them into `codegen_cache_key` and declares each one
# `cargo::rerun-if-env-changed`. They select a *lowering*: `PYRE_MIR_FRAMESTATE=0`
# sends `majit-translate`'s mir front end down a different path
# (`front/mir.rs` `framestate_enabled`), and the generated `jit_trace_gen.rs`
# lands in `OUT_DIR` under `target/`, which [`build_input_paths`] deliberately
# never reads. So nothing else in the fingerprint can see the difference, and
# without them a run under a gate stamps a digest a later unset run reproduces
# exactly — `--build=no` then measures a differently-lowered JIT.
#
# `RUSTFLAGS` and `CARGO_ENCODED_RUSTFLAGS` are cargo's own: the native builds
# inherit whatever is set, and only the wasm module build overrides `RUSTFLAGS`
# with `WASM_RUSTFLAGS` (which the recipe already names).
#
# A divergence from `LOWERING_GATE_ENV` is the usual duplicated-list rot;
# `pyre/pyrex/tests/gate_triage_complete.rs` is where a new gate is registered,
# and a gate that changes the artefact belongs here as well.
BUILD_RECIPE_ENV = (
    "PYRE_DYN_INDIRECT",
    "PYRE_FNPTR_INDIRECT",
    "PYRE_MIR_FRAMESTATE",
    "PYRE_OPTION_RESIDUAL_NARROW",
    "PYRE_TUPLE_PER_SHAPE_CLASSDEF",
    "RUSTFLAGS",
    "CARGO_ENCODED_RUSTFLAGS",
)


def build_recipe_digest():
    """A digest of the options an artefact is built *with*.

    The input set above is the files a build reads; this is the command run
    over them. `--no-default-features --features dynasm`, the wasm link args,
    the toolchain and the build-std flags each select what lands in the
    artefact, and none of them is a file under a member directory — this
    script is a build input by no path rule, so editing `--growable-table` out
    of `WASM_RUSTFLAGS` moved nothing in the fingerprint and a later
    `--build=no` run approved a module built with it still in.

    The whole table at once rather than the row for one backend: a stamp
    naming only its own row could not be compared without also recording which
    row it was, the table drives every artefact, and an edit to it is rare
    enough that rebuilding all of them is the cheaper mistake.

    What it does not cover is an option spelled inline in the build functions
    rather than held here — `--target wasm32-unknown-unknown` is the one such
    literal today. Hashing this script whole would cover those and would also
    make every comment edit invalidate every artefact, which in a tree where
    the script is edited far more often than the recipe is the worse trade.

    The recipe is not only what this script spells: `build_backend` runs
    `cargo` with no `env=`, so the ambient environment reaches the compiler
    and the build scripts too. [`BUILD_RECIPE_ENV`] names the part of it that
    selects what lands in the artefact, and an unset variable hashes
    distinguishably from an empty one — the same distinction
    `prepass.rs`'s `codegen_cache_key` draws with `write_os(var_os(key))`.
    """
    recipe = "\0".join((
        repr(CARGO_CONFIG),
        WASM_RUSTFLAGS,
        repr(WASM_CARGO_TOOLCHAIN),
        repr(WASM_BUILD_STD_FLAGS),
        WASM_BUILD_OUTPUT,
        WASM_MODULE_PATH,
        repr([(key, os.environ.get(key)) for key in BUILD_RECIPE_ENV]),
    ))
    return hashlib.sha256(recipe.encode("utf-8")).hexdigest()


# Sentinel distinguishing "not computed yet" from "computed, and there is no
# answer": a `None` result must be cached too, or one unenumerable call would
# leave a later call free to answer differently within the same run.
_FINGERPRINT_UNSET = object()
_BUILD_INPUTS_FINGERPRINT = _FINGERPRINT_UNSET


def build_inputs_fingerprint():
    """A digest of every build input's *content*, computed once per run.

    Content and not mtime: a concurrent `git checkout <ref> -- .`, a branch
    switch, or an editor rewriting a file it did not change re-stamps whole
    subtrees, and a gate reading mtimes then blocks a build that is in fact
    current. Hashing the roughly one thousand inputs (about 1GB, most of it
    the LLBC) costs under a second, against the multi-minute build
    `--build=no` exists to skip.

    Computed after a build rather than before it: `cargo` may rewrite
    `Cargo.lock`, which is itself an input, so the state that produced the
    artefact is the state that exists once the build has finished. Nothing
    else a build writes is in the set — that is what keeps the two orders
    equivalent for every other input.

    "Once per run" therefore holds only between builds, and
    [`invalidate_build_inputs_fingerprint`] is what ends a memoised value's
    life. Without that call the rule above would bind the FIRST stamped
    artefact alone: a run that builds several backends stamps after each one,
    and every stamp past the first would carry a digest read before the build
    that produced it. A later `--build=no` run then reads the tree as it
    actually is and rejects artefacts that are in fact current.
    """
    global _BUILD_INPUTS_FINGERPRINT
    if _BUILD_INPUTS_FINGERPRINT is not _FINGERPRINT_UNSET:
        return _BUILD_INPUTS_FINGERPRINT
    paths = build_input_paths()
    if paths is None:
        _BUILD_INPUTS_FINGERPRINT = None
        return None
    _BUILD_INPUTS_FINGERPRINT = digest_input_paths(paths)
    return _BUILD_INPUTS_FINGERPRINT


def digest_input_paths(paths):
    """A digest over the named paths' contents, and over the build recipe."""
    digest = hashlib.sha256()
    # The recipe first: an artefact is what its inputs and its build options
    # together produce, and neither alone identifies it.
    digest.update(build_recipe_digest().encode("ascii"))
    digest.update(b"\0")
    for path in paths:
        # The name goes in before the bytes are read, so a file that is
        # tracked but unreadable is distinguishable both from one that is
        # absent from the list and from one that is empty.
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        try:
            handle = open(path, "rb")
        except OSError:
            digest.update(b"\0")
            continue
        # Each file contributes a fixed-width value behind a one-byte tag,
        # rather than its bytes inline: feeding the content straight in leaves
        # the boundary between one file's tail and the next file's name
        # unmarked, so a file holding `b"b\0x"` at path `a` hashes to the same
        # bytes as an empty `a` beside a `b` holding `x`. Two input sets that
        # collide read as one unchanged tree, which is the answer this gate
        # exists to refuse.
        content = hashlib.sha256()
        with handle:
            while chunk := handle.read(1 << 20):
                content.update(chunk)
        digest.update(b"\1")
        digest.update(content.digest())
    return digest.hexdigest()


# The one input `cargo` itself rewrites during a build: a lock the build
# resolved is the build doing its job, not the tree moving underneath it.
# Excluded from a window comparison only when [`settle_cargo_resolution`]
# could not settle it up front.
CARGO_WRITTEN_INPUTS = ("Cargo.lock",)

# What `inputs` says in a stamp whose artefact was built over a tree that
# changed while it was building. It is not a digest and can equal none, so the
# comparison refuses it; `require_fresh_artefacts` names it to say why.
STAMP_TREE_MOVED = "tree-moved-during-build"

_BUILD_WINDOW_WITNESS = None
_BUILD_WINDOW_EXCLUDED = CARGO_WRITTEN_INPUTS
_CARGO_RESOLUTION_SETTLED = False


def settle_cargo_resolution():
    """Run a build's dependency resolution before the build window opens.

    `cargo` writes `Cargo.lock` during resolution, which happens once at the
    start of a build. While the lock is excluded from the witness, an
    *external* rewrite landing anywhere in the window — a `git checkout` of
    another branch's lock, a sibling worktree's `cargo update` — is
    indistinguishable from that resolution, and the stamp then records
    [`build_inputs_fingerprint`] over the new lock against a binary compiled
    from the old resolution. `build_input_paths` does list `Cargo.lock`
    (`ROOT_BUILD_INPUTS`), so the two readings disagree about exactly one file
    and nothing refuses the result.

    `cargo metadata` performs the same resolution a build performs and writes
    the same lock, without compiling; a lock already in sync makes it a
    sub-second no-op against a multi-minute build. Once it has returned the
    lock is settled for the rest of the window, so the lock can be judged like
    every other input instead of excluded from the comparison.

    Returns whether that held. A resolution this cannot run — no `cargo` on
    `PATH`, an offline tree whose index is cold — leaves the exclusion in
    place rather than turning every build into a tree-moved refusal.
    """
    global _CARGO_RESOLUTION_SETTLED
    if _CARGO_RESOLUTION_SETTLED:
        return True
    try:
        proc = subprocess.run(
            ["cargo", "metadata", "--format-version", "1", "--quiet"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    if proc.returncode != 0:
        return False
    _CARGO_RESOLUTION_SETTLED = True
    return True


def build_window_witness(excluded=()):
    """A digest of the inputs a build is about to read.

    Uncached, unlike [`build_inputs_fingerprint`]: the whole point is to be
    read twice and compared, so a memoised second read would answer with the
    first. Both readings must pass the same `excluded`, or the comparison
    reports a difference the tree does not have.
    """
    paths = build_input_paths()
    if paths is None:
        return None
    return digest_input_paths([p for p in paths if p not in excluded])


def open_build_window():
    """Note the tree state a build is starting from.

    `build_inputs_fingerprint` is read *after* a build, so that it describes
    the `Cargo.lock` the build may have resolved. That ordering is what lets an
    edit landing mid-build be recorded against an artefact compiled before it:
    the stamp would then name a tree the binary does not contain, and a later
    `--build=no` run would find them in agreement and measure it. Reading the
    tree here as well makes that window observable, which is the only way to
    tell the two orders apart.
    """
    global _BUILD_WINDOW_WITNESS, _BUILD_WINDOW_EXCLUDED
    _BUILD_WINDOW_EXCLUDED = () if settle_cargo_resolution() else CARGO_WRITTEN_INPUTS
    _BUILD_WINDOW_WITNESS = build_window_witness(_BUILD_WINDOW_EXCLUDED)


def invalidate_build_inputs_fingerprint():
    """Drop the memoised digest so the next read enumerates the tree again.

    Called once per completed build, before its artefacts are stamped. The
    memoisation that survives is the one it exists for: the read-only
    `--build=no` path, where nothing writes to the tree between reads.
    """
    global _BUILD_INPUTS_FINGERPRINT
    _BUILD_INPUTS_FINGERPRINT = _FINGERPRINT_UNSET


def artefact_fingerprint_path(artefact):
    return Path(str(artefact) + ".inputs")


def artefact_content_digest(artefact):
    """The artefact's own bytes, or `None` if it cannot be read.

    35MB of binary hashes in about a tenth of a second, against the
    multi-minute build the stamp exists to skip.
    """
    digest = hashlib.sha256()
    try:
        with open(artefact, "rb") as handle:
            while chunk := handle.read(1 << 20):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def stamp_artefact_inputs(artefact):
    """Record the inputs an artefact was just built from, beside it.

    The artefact's own digest goes in beside the input digest, because the
    sidecar cannot otherwise tell which binary it is vouching for. A `cargo
    build` outside this script overwrites the executable and leaves the
    sidecar alone; restoring the tree to the state the sidecar names would
    then make the input digests agree while the executable holds different
    code, and `--build=no` would record baselines against it.

    A failure here is not worth losing a finished build over, but it is worth
    saying: the next `--build=no` run would otherwise report the artefact as
    unstamped with no trace of why.
    """
    fingerprint = build_inputs_fingerprint()
    if fingerprint is None:
        return
    if (_BUILD_WINDOW_WITNESS is not None
            and _BUILD_WINDOW_WITNESS != build_window_witness(_BUILD_WINDOW_EXCLUDED)):
        # Some input changed while cargo was reading them, so the artefact
        # holds neither the tree the build started from nor the one that is
        # here now. Stamping the current tree against it would make a later
        # `--build=no` run agree with a binary that does not contain it.
        print(
            f"  warning: the tree changed while {artefact} was building, so "
            "its inputs are not\n           the tree that is here now; it is "
            "stamped as unusable for --build=no."
        )
        fingerprint = STAMP_TREE_MOVED
    content = artefact_content_digest(artefact)
    if content is None:
        print(f"  warning: could not read {artefact} to stamp it")
        return
    stamp = artefact_fingerprint_path(artefact)
    try:
        stamp.write_text(
            f"inputs {fingerprint}\nartefact {content}\n", encoding="utf-8"
        )
    except OSError as exc:
        print(f"  warning: could not write the build-input stamp {stamp}: {exc}")


def read_artefact_stamp(stamp):
    """The `(inputs, artefact)` digests a stamp records, or `None`.

    `None` covers both an unreadable stamp and one written in an older format
    — either way this script did not write it in a shape it can check, which
    is the "built outside this script" case the caller already reports.
    """
    try:
        text = stamp.read_text(encoding="utf-8")
    except OSError:
        return None
    fields = {}
    for line in text.splitlines():
        key, _, value = line.partition(" ")
        if value:
            fields[key] = value.strip()
    if "inputs" not in fields or "artefact" not in fields:
        return None
    return fields["inputs"], fields["artefact"]


def require_fresh_artefacts(artefacts, reason, remedy):
    """Refuse to measure an artefact that was built from different sources.

    `--build=no` skips every artefact of a backend, and wasm has two: the
    native runner and the wasm module the runner loads. A module left behind by
    an earlier build produces a fully green run — and a set of recorded
    baselines — for code that is not in it, with nothing in the output saying
    so.

    `reason` names the circumstance that put the artefact under question and
    `remedy` says what closes it, because skipping the build is not the only
    way to arrive here: `PYRE_WASM_MODULE` names the module the runner loads
    whether or not a build ran, so a normal run can measure a module this
    invocation did not produce.

    Each build stamps its artefact with `build_inputs_fingerprint` and with the
    artefact's own content digest; an artefact carrying no stamp was built
    outside this script, so its inputs are unknown and the run continues with a
    note rather than a refusal. An artefact whose bytes no longer match the
    stamp was rebuilt outside this script since, which puts it in the same
    unknown-inputs class however well the input digest agrees.
    """
    fingerprint = build_inputs_fingerprint()
    if fingerprint is None:
        return
    for artefact in artefacts:
        recorded = read_artefact_stamp(artefact_fingerprint_path(artefact))
        if recorded is None:
            print(
                f"  note: {artefact} carries no build-input stamp; "
                "its freshness is unchecked"
            )
            continue
        recorded_inputs, recorded_content = recorded
        if recorded_content != artefact_content_digest(artefact):
            fault = "has been rebuilt since it was stamped, so what it contains is unknown."
        elif recorded_inputs == STAMP_TREE_MOVED:
            fault = ("was built while the tree was changing, so which sources "
                     "it contains was never established.")
        elif recorded_inputs != fingerprint:
            fault = ("was built from different sources, so it does not contain "
                     "the current tree.")
        else:
            continue
        print(f"ERROR: {reason}, but {artefact}\n       {fault}\n       {remedy}")
        sys.exit(1)


def require_fresh_llbc():
    """Refuse to measure when `build/llbc/` no longer matches the tree.

    Field offsets are read out of those artefacts, so one that predates an edit
    names the wrong bytes — and a miscompiled field access returns a NUMBER
    rather than an error, which makes an unsound run indistinguishable from a
    real one. `pyre-jit-trace`'s `build.rs` refuses on exactly that, but it can
    only refuse while a build is running. `--build=no` is the path where
    nothing else asks, and it is the path a moving tree lands in: a branch that
    moves after the binaries were built leaves them measurable and the offsets
    they read stale, with nothing in the output saying so.

    The artefact stamps `require_fresh_artefacts` reads answer a different
    question — whether *this script* produced the binary from these sources —
    and an artefact built by a bare `cargo build` carries no stamp at all, so
    that gate notes and continues. This one does not depend on who built what:
    it asks the extractor whether what is on disk describes the tree that is on
    disk.
    """
    script = Path(__file__).resolve().parent / "scripts" / "extract-llbc.py"
    if not script.is_file():
        return
    proc = subprocess.run(
        [sys.executable, str(script), "--check"],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        return
    print(red("LLBC artefacts under build/llbc/ are STALE."))
    for line in ((proc.stdout or "") + (proc.stderr or "")).splitlines():
        if line.startswith("  ") and ": artefact says" in line:
            print(line)
    print("Field offsets come from them, so a run on this tree would measure "
          "the wrong bytes.")
    print("Re-extract, then re-run this script:")
    print("    python3 pyre/scripts/extract-llbc.py "
          "majit-rlib pyre-object pyre-interpreter pyre-module pyre-jit")
    sys.exit(1)


# Relative tolerance for wasm float outputs ONLY (see `wasm_outputs_match`).
WASM_FLOAT_RTOL = 1e-9


def wasm_outputs_match(output, expected):
    """Bench-output comparison for the wasm backend, allowing a bounded
    float divergence — used ONLY for benches explicitly opted in with
    run_bench(..., wasm_float_tol=True). No blanket wasm allowance exists;
    today only nbody is marked, and every other wasm bench stays byte-exact.

    Root cause (measured, not guessed): the wasm guest and native pyre run
    the SAME libm source, but libm's transcendentals (`pow` is the FreeBSD
    `e_pow.c` hi/lo split) bottom out on ARCH-SPECIFIC building blocks
    (`libm/src/math/arch/aarch64.rs`). On aarch64 those land on hardware-tuned
    ops that match the platform macOS libm — so native pyre matches
    CPython/PyPy bit-for-bit — while wasm32 gets the generic software
    fallbacks. That is a ~0.5-ULP-per-op gap that accumulates (nbody: 5M pow
    calls -> ~1490 ULP, 3e-13 relative, in the printed energy). It is an
    unfixable target-ISA codegen gap, not a miscompile: an FMA-fusion
    hypothesis was tested and refuted (fp-contract=fast on the wasm build is
    a no-op — wasm32 scalar has no FMA instruction — and libm's pow uses no
    mul_add), forcing native unfused would break native parity with the
    platform reference, and host-libm callbacks were rejected (they make wasm
    output vary by host machine, which we do not want).

    So float tokens are compared with a relative tolerance `WASM_FLOAT_RTOL`
    (1e-9 — ~4 orders looser than the observed 3e-13 drift, ~3 orders
    TIGHTER than the smallest real value bug seen, 5.7e-6). Every non-float
    token (ints, strings, non-finite floats) still must match byte-for-byte,
    so an int off-by-one can never slip through."""
    if output == expected:
        return True
    out_lines, exp_lines = output.splitlines(), expected.splitlines()
    if len(out_lines) != len(exp_lines):
        return False
    for out_line, exp_line in zip(out_lines, exp_lines):
        if out_line == exp_line:
            continue
        out_toks, exp_toks = out_line.split(), exp_line.split()
        if len(out_toks) != len(exp_toks):
            return False
        for out_tok, exp_tok in zip(out_toks, exp_toks):
            if out_tok == exp_tok:
                continue
            # Only finite float-shaped tokens (decimal point or exponent)
            # get tolerance; ints/strings/nan/inf must match byte-for-byte
            # above, so an int off-by-one can never slip through.
            def _floaty(tok):
                return "." in tok or "e" in tok or "E" in tok
            if not (_floaty(out_tok) and _floaty(exp_tok)):
                return False
            try:
                a, b = float(out_tok), float(exp_tok)
            except ValueError:
                return False
            if not (math.isfinite(a) and math.isfinite(b)):
                return False
            if abs(a - b) > WASM_FLOAT_RTOL * max(abs(a), abs(b)):
                return False
    return True


# Backends rendered in fixed-column displays, in order. Any enabled backend not
# listed here still runs and is counted; it just falls outside the fixed columns.
ALL_BACKENDS = ("dynasm", "cranelift", "wasm")


def _jitstats_shared_path(source):
    return source.with_name(f"{source.stem}.jitstats")


def _jitstats_specific_path(source, backend):
    return source.with_name(f"{source.stem}.{backend}.jitstats")


def _jitstats_texts_match(a, b):
    """True when two baselines are the same snapshot under the gate.

    `_jit_stats_change` reads a field missing from either side as 0, so a
    printed zero and an omitted key record one baseline. Merging keeps a
    single file for that case.
    """
    left, right = _parse_jit_stats(a), _parse_jit_stats(b)
    keys = set(left) | set(right)
    return all(left.get(key, "0") == right.get(key, "0") for key in keys)


def _jitstats_overlay_path(source, backend):
    """Platform overlay, if one is committed for this host. Else None.

    A `<name>.<backend>.<platform>.github-actions.jitstats` wins on that
    GitHub runner; otherwise `<name>.<backend>.<platform>.jitstats` wins on
    the platform. These stay per backend: they exist only where one host
    disagrees, and folding them into the shared file would hide that.
    """
    if os.environ.get("GITHUB_ACTIONS") == "true":
        github_runner = source.with_name(
            f"{source.stem}.{backend}.{sys.platform}.github-actions.jitstats"
        )
        if github_runner.exists():
            return github_runner
    per_platform = source.with_name(
        f"{source.stem}.{backend}.{sys.platform}.jitstats"
    )
    if per_platform.exists():
        return per_platform
    return None


def jitstats_baseline_path(script, backend):
    """Committed baseline consulted for `backend` on `script`.

    Resolution order: a platform overlay, then `<name>.<backend>.jitstats`
    when this backend's counters differ, then `<name>.jitstats` shared by
    every backend whose snapshot gates as equal. When nothing is committed,
    the path is the per-backend name the missing-baseline error reports.
    """
    source = Path(script)
    overlay = _jitstats_overlay_path(source, backend)
    if overlay is not None:
        return overlay
    specific = _jitstats_specific_path(source, backend)
    if specific.exists():
        return specific
    shared = _jitstats_shared_path(source)
    if shared.exists():
        return shared
    return specific


def _choose_shared_jitstats_text(members):
    """Exact bytes to store for a cluster that already gates as equal.

    Prefer the text most members already have. A tie prefers the shorter
    one, which drops a key another backend only prints as 0.
    """
    counts = {}
    for _backend, text in members:
        counts[text] = counts.get(text, 0) + 1
    return max(counts, key=lambda text: (counts[text], -len(text)))


def rewrite_jitstats_baselines(script, contents):
    """Write the smallest baseline set that still gates as `contents`.

    `contents` maps a backend to its snapshot text, or None when that
    backend has no baseline. Callers omit backends they are not placing.
    At least two backends whose texts gate as equal share
    `<stem>.jitstats`. A backend that disagrees keeps
    `<stem>.<backend>.jitstats`. Platform overlays are left alone.
    """
    source = Path(script)
    shared_path = _jitstats_shared_path(source)
    clusters = []
    for backend, text in contents.items():
        if text is None:
            continue
        for cluster in clusters:
            if _jitstats_texts_match(cluster[0][1], text):
                cluster.append((backend, text))
                break
        else:
            clusters.append([(backend, text)])

    order = {backend: index for index, backend in enumerate(ALL_BACKENDS)}

    def cluster_rank(cluster):
        earliest = min(order.get(backend, len(order)) for backend, _text in cluster)
        return (len(cluster), -earliest)

    shared_backends = set()
    if clusters:
        shared_cluster = max(clusters, key=cluster_rank)
        if len(shared_cluster) >= 2:
            shared_text = _choose_shared_jitstats_text(shared_cluster)
            shared_backends = {backend for backend, _text in shared_cluster}
            current = (
                shared_path.read_text(encoding="utf-8")
                if shared_path.exists()
                else None
            )
            if current != shared_text:
                shared_path.write_text(shared_text, encoding="utf-8", newline="")
    if not shared_backends and shared_path.exists():
        shared_path.unlink()

    for backend, text in contents.items():
        specific = _jitstats_specific_path(source, backend)
        if backend in shared_backends or text is None:
            specific.unlink(missing_ok=True)
            continue
        current = specific.read_text(encoding="utf-8") if specific.exists() else None
        if current != text:
            specific.write_text(text, encoding="utf-8", newline="")


def store_jitstats_baseline(script, backend, text, absent=()):
    """Record `text` for `backend` and re-fold sibling baselines.

    `text` None removes this backend's baseline. An existing platform
    overlay is updated in place and does not participate in the fold:
    recording one host must not rewrite the baseline the other hosts read.

    `absent` names backends whose binary lacks a module the fixture's
    `requires-modules=` header names. They sit out the fold like a
    `skip-backends=` one: they never ran the fixture, so splitting the shared
    file must not hand them a baseline of their own.
    """
    source = Path(script)
    overlay = _jitstats_overlay_path(source, backend)
    if overlay is not None:
        if text is None:
            overlay.unlink(missing_ok=True)
        else:
            overlay.write_text(text, encoding="utf-8", newline="")
        return

    skipped = set(synth_skip_backends(script)) | set(absent)
    backends = [name for name in ALL_BACKENDS if name not in skipped]
    if backend not in backends:
        backends.append(backend)
    for name in ALL_BACKENDS:
        if name not in backends and _jitstats_specific_path(source, name).exists():
            backends.append(name)

    shared_path = _jitstats_shared_path(source)
    shared_text = (
        shared_path.read_text(encoding="utf-8") if shared_path.exists() else None
    )
    contents = {}
    for name in backends:
        if name == backend:
            contents[name] = text
            continue
        specific = _jitstats_specific_path(source, name)
        if specific.exists():
            contents[name] = specific.read_text(encoding="utf-8")
        else:
            contents[name] = shared_text

    if text is None:
        # No snapshot must not inherit the shared file on the next run.
        # Materialize the siblings, then drop the shared baseline.
        for name, sibling in contents.items():
            if name == backend or sibling is None:
                continue
            specific = _jitstats_specific_path(source, name)
            current = (
                specific.read_text(encoding="utf-8") if specific.exists() else None
            )
            if current != sibling:
                specific.write_text(sibling, encoding="utf-8", newline="")
        shared_path.unlink(missing_ok=True)
        _jitstats_specific_path(source, backend).unlink(missing_ok=True)
        return

    rewrite_jitstats_baselines(script, contents)


def _wasm_target_installed():
    """Whether the wasm backend can be built here.

    The only extra prerequisite over the native backends is the
    `wasm32-unknown-unknown` rustup target (the wasmtime runtime is embedded
    in `pyre-wasm-runner`, not an external tool). If it is missing, the wasm
    build would `rustup target add`-fail, so wasm stays out of the default set.
    """
    try:
        proc = subprocess.run(
            ["rustup", "target", "list", "--installed"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0 and "wasm32-unknown-unknown" in proc.stdout.split()


# wasm joins the defaults only where its target is installed, so a plain
# `check.py` on an unconfigured machine still runs just the native backends.
DEFAULT_BACKENDS = ("dynasm", "cranelift")
if _wasm_target_installed():
    DEFAULT_BACKENDS = (*DEFAULT_BACKENDS, "wasm")

# ── Check runner ─────────────────────────────────────────────────────

class Check:
    def __init__(self, args):
        self.args = args
        self.results = []
        self.comparisons = []
        # Synthetic fixtures whose cpython reference run exceeded its budget,
        # so pypy alone was the baseline for them.
        self.cpython_reference_skips = []
        self.oracle_store, self.oracle_cache_line = oracle_cache.resolve_store()
        self.oracle_refresh = bool(getattr(args, "no_oracle_cache", False))
        self.cpython_declared_skips = []
        # Per-backend bookkeeping, keyed by backend name so any backend
        # (dynasm / cranelift / wasm / a single `--pyre-path` binary) is tracked
        # uniformly.
        self.pass_count = {}
        self.fail_count = {}
        self.pyre = {}
        # backend -> frozenset of its binary's `sys.builtin_module_names`, or
        # None when the probe failed. Filled on first use by
        # `_builtin_modules`.
        self.builtin_modules = {}
        # baseline interpreter -> measured empty-program user-CPU startup (s).
        # Populated by measure_startups(); missing key => 0.0 (no subtraction).
        # Read through `_startup_key`, which maps every pyre backend to pypy.
        self.startup = {}
        # time.monotonic() of the pass that produced self.startup; None until
        # the first one. Read by maybe_refresh_startups() to age the estimate.
        self.startup_sampled_at = None
        # (backend, bench name) -> raw user-CPU of that backend's run in THIS
        # invocation. Feeds the wasm-vs-dynasm gate, which needs a baseline
        # measured on the same host under the same load; a recorded one would
        # age out of the machine it was taken on.
        self.bench_elapsed = {}
        # Fixtures where the wasm run had no dynasm time to divide by -- absent
        # or under the noise floor -- so WASM_MAX_DYNASM_RATIO was never
        # applied. Reported in the summary: an unevaluated gate otherwise
        # prints the same green as a satisfied one.
        self.wasm_ratio_ungated = []
        self.pypy_ratio_ungated = []
        # (bench name, ceiling, measured ratio or None) for fixtures whose
        # header raised the ratio above WASM_MAX_DYNASM_RATIO, for the same
        # reason as the line above: a gate widened for a fixture and a gate the
        # fixture satisfied both print green. The measured ratio rides along
        # because an allowance the backend has since outgrown is invisible
        # otherwise -- one sat at 4.8x over a fixture measuring 2.9x, and
        # finding that took parsing 434 comparison rows out of a job log.
        # `None` means the gate was declined this run, so there is no ratio to
        # compare it against. Reported from `print_summary` rather than beside
        # that line, because three of the fixtures carrying an allowance are
        # regular benches and a `--no-synthetic` run would otherwise not name
        # them.
        self.wasm_ratio_allowed = []
        self.snapshot_diffs = []
        self.snapshot_missing = []
        self.jitstats_diffs = []
        # Benches whose gated counters all moved the way their field calls a
        # gain. Still failures — the baseline stopped describing the tree — but
        # tracked apart so the summary can say which reds need investigating and
        # which only need re-recording.
        self.jitstats_improvements = []
        # Benches whose counters did not reproduce across repeats in this same
        # invocation. Reported, never failed — see JITSTATS_STABILITY_RUNS.
        self.jitstats_unstable = []
        # Benches whose gated counters moved only inside a declared per-fixture
        # band. Reported, never failed.
        self.jitstats_banded = []
        # "<backend>/<bench>: <counters>" for fixtures whose header exempted a
        # counter (`synth_ungated_jitstats`). An exempted counter and a matching
        # one print the same green, so the exemptions are named in the summary:
        # a gate nobody applied should not read as a gate nobody had to.
        self.jitstats_ungated = []
        self.jitstats_missing = []
        # Benches whose run printed no `[jit-stats]` line at all. Tracked apart
        # from `jitstats_missing` because the two say opposite things: a missing
        # baseline is a bench nobody recorded, an absent line is a bench that
        # stopped reporting.
        self.jitstats_absent = []
        # Benches whose run printed the line, but with a badness counter's
        # census denominator at zero — the counter's gate passed on an
        # instrument that inspected nothing. Tracked apart from
        # `jitstats_absent` because the line IS there and every other gate on it
        # is still meaningful; only the paired counters are void.
        self.jitstats_vacuous = []

    # ── backend helpers ──

    def enabled(self, backend):
        return bool(self._pyre(backend))

    def _pyre(self, backend):
        return self.pyre.get(backend, "")

    def _builtin_modules(self, backend):
        """The backend binary's `sys.builtin_module_names`, probed once.

        None when the probe itself fails: the fixtures that asked are then
        run rather than skipped, so a binary that cannot start is reported
        by them instead of hidden behind a skip. An exit status of 0 is not
        enough, because a wasm guest without an exit-status export reports 0
        after an uncaught exception: the answer counts only when its marked
        line is there and names `sys`, which every build has.
        """
        if backend not in self.builtin_modules:
            with tempfile.TemporaryDirectory() as tmp:
                probe = Path(tmp) / "builtin_modules.py"
                probe.write_text(
                    "import sys\n"
                    f"print({BUILTIN_MODULES_MARKER!r}, *sys.builtin_module_names)\n",
                    encoding="utf-8",
                )
                output, _, code, _ = run_timed(
                    [self._pyre(backend), str(probe)],
                    timeout_s=scaled_timeout(30, self._timeout_scale(backend)),
                    env=pyre_env(),
                )
            names = next(
                (
                    frozenset(line.split()[1:])
                    for line in output.splitlines()
                    if line.split()[:1] == [BUILTIN_MODULES_MARKER]
                ),
                None,
            )
            self.builtin_modules[backend] = (
                names if code == 0 and names and "sys" in names else None
            )
        return self.builtin_modules[backend]

    def _fixture_skips(self, header):
        """{backend: reason} for every enabled backend a fixture skips.

        The reason is empty for a `skip-backends=` exemption and names the
        missing modules for a `requires-modules=` one.
        """
        skips = {backend: "" for backend in header["skip_backends"]}
        required = header["requires_modules"]
        for backend in ALL_BACKENDS:
            if not required or backend in skips or not self.enabled(backend):
                continue
            present = self._builtin_modules(backend)
            if present is None:
                continue
            missing = [name for name in required if name not in present]
            if missing:
                skips[backend] = f"no {', '.join(missing)}"
        return skips

    @staticmethod
    def _print_skip(backend, reason):
        sys.stdout.write(f"    {backend:<10s}")
        print(dim(f"skip ({reason})" if reason else "skip"))

    def _timeout_scale(self, backend):
        if backend == "dynasm" and self.args.dynasm_timeout_scale is not None:
            return self.args.dynasm_timeout_scale
        if backend == "cranelift" and self.args.cranelift_timeout_scale is not None:
            return self.args.cranelift_timeout_scale
        if backend == "wasm":
            if self.args.wasm_timeout_scale is not None:
                return self.args.wasm_timeout_scale
            # Default wasm headroom composes with --timeout-scale.
            return WASM_TIMEOUT_SCALE * self.args.timeout_scale
        if sys.platform == "win32":
            return WIN_NATIVE_TIMEOUT_SCALE * self.args.timeout_scale
        return self.args.timeout_scale

    def _sample_startups(self):
        """One sampling pass: ({key: [user-CPU seconds]}, wall seconds spent).

        Only the baseline interpreters are spawned -- see `_startup_key`, which
        is what routes a pyre backend to pypy's reading.

        A target whose empty program exits non-zero yields an empty list, which
        `_adopt_startups` reads as "this pass has no estimate for it" rather
        than as zero.

        The wall cost is returned so every pass states its own price in the
        log: it is what STARTUP_REFRESH_S is traded against.
        """
        started = time.monotonic()
        with tempfile.NamedTemporaryFile(
            "w", suffix=".py", delete=False
        ) as handle:
            empty_path = handle.name  # zero-length program
        try:
            targets = [
                ("cpython", [python3(), empty_path], None),
                ("pypy", [PYPY3, empty_path], pypy_env()),
            ]
            samples = {}
            for key, cmd, env in targets:
                got = []
                for _ in range(STARTUP_SAMPLES):
                    _out, elapsed, code, _err = run_timed(
                        cmd, timeout_s=60, env=env,
                    )
                    if code != 0:
                        got = []
                        break
                    got.append(elapsed)
                samples[key] = got
            return samples, time.monotonic() - started
        finally:
            try:
                os.unlink(empty_path)
            except OSError:
                pass

    def _adopt_startups(self, samples):
        """Take a pass as the current estimate; return its rendered parts.

        A target this pass failed to sample keeps the estimate it already had:
        on the first pass that is 0.0, which is the no-subtraction default, and
        on a refresh it is the previous pass's, which beats dropping to zero
        mid-job and inflating every later exec time for that interpreter. Such
        a target is rendered `[unsampled]` so the value is not read as fresh.
        """
        parts = []
        for key, got in samples.items():
            if got:
                self.startup[key] = statistics.fmean(got)
                spread = f"[{min(got):.3f}..{max(got):.3f}]"
            else:
                spread = "[unsampled]"
            parts.append(f"{key}={self.startup.get(key, 0.0):.3f}s{spread}")
        self.startup_sampled_at = time.monotonic()
        return " ".join(parts)

    def measure_startups(self):
        """Measure each baseline interpreter's empty-program user-CPU cost.

        Runs an empty script STARTUP_SAMPLES times per interpreter, records
        the mean user-CPU in self.startup and prints it beside the range its
        samples spanned. This is the fixed per-process cost of interpreter
        init, added to every bench regardless of workload; subtracting it
        yields a comparison of the work the fixture asked for. Re-measured
        during the run by `maybe_refresh_startups`. No-op under
        --no-startup-subtract.
        """
        if self.args.no_startup_subtract:
            return
        samples, cost = self._sample_startups()
        print(dim(
            f"startup (empty-program user-CPU, mean of {STARTUP_SAMPLES} "
            f"in {cost:.1f}s, shown as mean[min..max]; ratios/gates are "
            f"execution-only): "
            + self._adopt_startups(samples)
        ))

    def maybe_refresh_startups(self):
        """Re-measure startup once the estimate in hand is STARTUP_REFRESH_S old.

        Call at a fixture boundary only -- see STARTUP_REFRESH_S. Prints the
        pass in the same shape as the opening line, so the job's log carries
        the whole series and a suspicious `exec` can be read against the
        estimate that was actually subtracted from it.
        """
        if self.args.no_startup_subtract or self.startup_sampled_at is None:
            return
        if time.monotonic() - self.startup_sampled_at < STARTUP_REFRESH_S:
            return
        samples, cost = self._sample_startups()
        print(dim(
            f"startup re-measured in {cost:.1f}s (mean[min..max]): "
            + self._adopt_startups(samples)
        ))

    @staticmethod
    def _startup_key(key):
        """Whose startup reading is subtracted for interpreter/backend *key*.

        Every pyre backend answers "pypy", so a pyre-vs-pypy ratio and a
        wasm-vs-dynasm ratio both subtract one estimate from both sides. Only
        cpython, which no pyre run is being held to, keeps its own.
        """
        return "cpython" if key == "cpython" else "pypy"

    def _exec_time(self, key, t):
        """Startup-subtracted user-CPU for interpreter *key*, floored.

        Returns *t* unchanged when subtraction is disabled or *t* is
        unavailable ("-"/None). Never returns <= 0 (floored to
        EXEC_TIME_FLOOR_S) so downstream ratios cannot divide by ~0.
        """
        if t is None or t == "-":
            return t
        value = float(t)
        if self.args.no_startup_subtract:
            return value
        return max(
            value - self.startup.get(self._startup_key(key), 0.0),
            EXEC_TIME_FLOOR_S,
        )

    def _set_pyre(self, backend, path):
        self.pyre[backend] = path

    # ── comparison table ──

    def _comp_index(self, name):
        for i, c in enumerate(self.comparisons):
            if c["name"] == name:
                return i
        return -1

    def _append_comparison(self, backend, name, t_cpython, t_pypy, pyre_field, note=""):
        idx = self._comp_index(name)
        if idx == -1:
            entry = {
                "name": name,
                "cpython": fmt_time(t_cpython),
                "pypy": fmt_time(t_pypy),
            }
            for b in ALL_BACKENDS:
                entry[b] = "-"
            self.comparisons.append(entry)
            idx = len(self.comparisons) - 1
        else:
            self.comparisons[idx]["cpython"] = fmt_time(t_cpython)
            self.comparisons[idx]["pypy"] = fmt_time(t_pypy)

        cell = pyre_field
        if note:
            note = note.strip("()")
            if note.endswith(" vs pypy"):
                note = note[: -len(" vs pypy")]
            cell = f"{pyre_field:>6s}   {note:>5s}"
        self.comparisons[idx][backend] = cell

    # ── record result ──

    def _record(self, backend, passed, name, detail):
        if passed:
            self.pass_count[backend] = self.pass_count.get(backend, 0) + 1
        else:
            self.results.append(f"{red('FAIL')} {backend} {name}  {detail}")
            self.fail_count[backend] = self.fail_count.get(backend, 0) + 1

    # ── snapshot gate ──

    def _snapshot_path(self, backend, name, suffix):
        return Path(SNAP_DIR) / backend / f"{name}.{suffix}"

    def _jitstats_repeats(self, backend, script, timeout, ungated=()):
        """Re-run the fixture and return each repeat's jit-stats snapshot.

        Returns None if a repeat did not exit cleanly, so a crashing re-run
        cannot argue a real diff away — the caller then gates on the first run
        as if no repeat had happened.

        *ungated* is the caller's, so a repeat is narrowed the same way the run
        being compared was: narrowing one side only would report every exempted
        counter as instability.
        """
        effective_timeout = scaled_timeout(timeout, self._timeout_scale(backend))
        snapshots = []
        for _ in range(JITSTATS_STABILITY_RUNS):
            _output, _elapsed, code, stderr = run_timed(
                [self._pyre(backend), script],
                timeout_s=effective_timeout, env=pyre_env(),
            )
            if code != 0:
                return None
            snapshots.append(_jit_stats_snapshot(stderr, ungated))
        return snapshots

    def _jitstats_baseline_path(self, backend, script):
        # The committed structural-stats baseline sits beside its benchmark
        # source, not in the gitignored check.snap scratch tree that holds the
        # local .out/.time snapshots. Backends whose snapshots gate as equal
        # share `<name>.jitstats`. A backend that disagrees keeps
        # `<name>.<backend>.jitstats`, and that file wins over the shared one.
        #
        # A `<name>.<backend>.<platform>.github-actions.jitstats` beside it wins
        # on that GitHub runner; otherwise a
        # `<name>.<backend>.<platform>.jitstats` wins on the platform. These are
        # for counters a host genuinely disagrees on, and keep the comparison
        # exact everywhere instead of widening the gate for everyone: a band
        # that tolerated the disagreement would tolerate it on every fixture
        # and all three backends, and the band this gate replaced is exactly
        # what let eight baselines go stale unnoticed.
        #
        # No fixture has one at present, and the last set to need them is worth
        # recording because of how it ended. `str_fstring` was carried by three:
        # the runners read guard_failures 658/659 on ubuntu dynasm/cranelift,
        # the inverse 659/658 on macos, and 658/658 on windows, with every other
        # gated counter identical (six loops, three bridges) and each mismatch
        # reproducing in both stability reruns. The split was stable per host, so
        # it read as a host disagreement.
        #
        # It was not one. The knife edge is a major-collection crossing landing
        # either side of the peeled preamble (see `pyre_env`), and its input is
        # how much the process allocated before the loops got hot — a quantity
        # every host computes differently and nothing about the platform fixes.
        # Adding check.py's two wasm environment keys, semantically ignored by
        # the native backend, moves a local macOS cranelift build 658 -> 659
        # while the number of major collections stays at 11; the guard sequence
        # gains one trace-6 fail-5 exit that the compiled-trace log maps to the
        # GuardFalse on the eval-breaker poll. Raising `PYPY_GC_MIN` past the
        # fixture's working set removed the crossing, and all three overlays
        # came out with it.
        #
        # That is the shape to look for before writing one: a counter that
        # splits per host and moves under a perturbation with no semantic
        # content is not a host disagreement, it is a boundary the harness has
        # not pinned yet. An overlay would have frozen it per platform instead.
        #
        # `child_env_base` since removed the environment term of that volume:
        # a child starts from an allowlist rather than a copy of the runner's
        # environment. That closes one input, not the class -- any other change
        # in allocation volume still moves the point, which is why taking the
        # fixture off the threshold is what actually settled it.
        #
        # An overlay also shadows the shared baseline permanently, so one written
        # against a value that later converges becomes a failure main does not
        # have: `recursive_call_frame_relocation` was given one for 637, and
        # windows later converged on its shared value. Read all three runners
        # back before adding one, and prefer removing the fixture's dependence on
        # the host input to recording the host.
        return jitstats_baseline_path(script, backend)

    def _apply_snapshot_gate(
        self, backend, name, script, output, stderr, elapsed, timeout,
    ):
        status, reason = "ok", ""
        out_path = self._snapshot_path(backend, name, "out")
        time_path = self._snapshot_path(backend, name, "time")
        jitstats_path = self._jitstats_baseline_path(backend, script)
        ungated = synth_ungated_jitstats(script)
        if ungated:
            self.jitstats_ungated.append(f"{backend}/{name}: {', '.join(ungated)}")
        jitstats = _jit_stats_snapshot(stderr, ungated)
        jitstats_bands = synth_jitstats_bands(script)

        # The jit-stats gate — enforced on EVERY run, so a structural JIT change
        # reddens the default `pyre/check.py` (locally, and in the bare CI
        # invocation) the instant it lands, with no --snapshot-diff needed. This
        # catches the inline-abort class (e.g. an inlined-callee LOAD_GLOBAL fold
        # that stops resolving: loops_aborted 0 -> 5) and the dead-bridge class
        # (bridges_compiled 27 -> 0). Record mode is writing a fresh baseline, so
        # it is exempt.
        #
        # Each of the three conditions below is a way for the gate to be silently
        # disarmed rather than a way for it to pass, which is why all three fail
        # rather than being skipped:
        #
        # * No `[jit-stats]` line at all. `pyre_env` sets MAJIT_STATS=1 for every
        #   pyre run, so a run that prints none is not a bench without counters —
        #   it is a bench whose counters went missing, and reading that as
        #   "nothing to compare" skips the baseline check and the comparison
        #   together.
        # * No committed baseline. Every gate then runs against nothing and the
        #   bench prints PASS, so "never recorded" is indistinguishable from
        #   "recorded and clean" — which is how the floor came to cover 3 of 340
        #   synthetic fixtures without anyone noticing.
        # * A counter that moved, in either direction.
        # * A badness counter reading 0 because its census never ran, rather
        #   than because nothing is wrong — `JITSTATS_DENOMINATOR_FIELDS`. The
        #   other three disarms are visible as an absence (no line, no file, no
        #   move); this one is invisible, because a disarmed census and a clean
        #   one print the same digit.
        if self.args.snapshot_mode != "record":
            if jitstats is None:
                self.jitstats_absent.append(f"{backend}/{name}")
                return "fail", (
                    "no [jit-stats] line in this run's stderr — MAJIT_STATS=1 is "
                    "set for every pyre run, so every jit-stats gate below was "
                    "skipped rather than passed"
                )
            if not jitstats_path.exists():
                self.jitstats_missing.append(f"{backend}/{name}")
                return "fail", (
                    f"no committed jit-stats baseline ({jitstats_path.name}) — record it with "
                    f"`pyre/check.py --snapshot --backend {backend}`"
                )
            vacuous = _jit_stats_vacuous(stderr)
            if vacuous:
                self.jitstats_vacuous.append(f"{backend}/{name}")
                return "fail", vacuous
            regressions, improvements, banded = _jit_stats_change(
                jitstats_path.read_text(encoding="utf-8"), jitstats, jitstats_bands
            )
            if regressions or improvements:
                repeats = self._jitstats_repeats(backend, script, timeout, ungated)
                drifted = next(
                    (s for s in repeats or () if s != jitstats), None
                )
                if drifted is not None:
                    # Drift compares two samples directly and is deliberately unbanded.
                    moved = _jit_stats_change(jitstats, drifted)
                    self.jitstats_unstable.append(f"{backend}/{name}")
                    return "unstable", (
                        "jit-stats unstable — re-running the same binary moved "
                        + ", ".join(moved[0] + moved[1])
                        + ", so this run's counters are not a property of the "
                        "tree and the baseline comparison ("
                        + ", ".join(regressions + improvements)
                        + ") is not gated"
                    )
                parts = []
                if regressions:
                    parts.append("regressed: " + ", ".join(regressions))
                if improvements:
                    parts.append("improved: " + ", ".join(improvements))
                if banded:
                    parts.append("within band: " + ", ".join(banded))
                reason = (
                    "jit-stats change — " + "; ".join(parts)
                    + _jit_stats_context(jitstats)
                    + f" (re-record with `pyre/check.py --snapshot --backend {backend} "
                    f"--synthetic-pattern {Path(script).stem}`)"
                )
                if regressions:
                    self.jitstats_diffs.append(f"{backend}/{name}")
                    return "regressed", reason
                self.jitstats_improvements.append(f"{backend}/{name}")
                return "improved", reason
            if banded:
                # Assigned, not returned: every other verdict above is a
                # failure, so leaving early costs those nothing. A band means
                # the run is fine, and returning here would carry the fixture
                # past the `--snapshot-diff` and `--threshold` gates below on
                # every run of a host that sits inside its band.
                self.jitstats_banded.append(f"{backend}/{name}")
                status, reason = "banded", (
                    "jit-stats within band — " + ", ".join(banded)
                    + _jit_stats_context(jitstats)
                )

        if self.args.snapshot_mode == "record":
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # `newline=""` on every baseline: these files are committed,
            # and a text-mode write on Windows would store each line feed as
            # CRLF, so the recorder would rewrite every line of a baseline it
            # did not change.
            out_path.write_text(output, encoding="utf-8", newline="")
            time_path.write_text(f"{elapsed:.2f}", encoding="utf-8", newline="")
            # Fold after writing: a backend that now matches its siblings
            # drops `<name>.<backend>.jitstats` in favor of `<name>.jitstats`,
            # and one that diverges is split back out of the shared file.
            store_jitstats_baseline(
                script, backend, jitstats,
                self._fixture_skips({
                    "skip_backends": (),
                    "requires_modules": synth_requires_modules(script),
                }),
            )

        if self.args.snapshot_mode == "diff":
            if not out_path.exists():
                self.snapshot_missing.append(f"{backend}/{name}")
            else:
                saved_out = out_path.read_text(encoding="utf-8")
                if output != saved_out:
                    self.snapshot_diffs.append(f"{backend}/{name}")
                    return "fail", "snapshot output diff"

            # The jit-stats half of this mode is gone: the gate above already
            # ran the same comparison on every run, so reaching here means it
            # found a baseline and found it equal.

        if (
            self.args.threshold is not None
            and elapsed is not None
            and elapsed != "-"
            and time_path.exists()
        ):
            saved_time_str = time_path.read_text(encoding="utf-8").strip()
            if saved_time_str and saved_time_str != "-":
                saved_time = float(saved_time_str)
                limit = saved_time * (1 + self.args.threshold / 100.0)
                if float(elapsed) > limit:
                    return "fail", f"threshold {elapsed:.2f}s > baseline {saved_time_str}s +{self.args.threshold}%"

        return status, reason

    # ── build ──

    def build_backend(self, backend):
        # `pyre-jit-trace/build.rs` compares each `build/llbc/*.ullbc` against
        # what its crate's sources hash to now and fails the build on a
        # mismatch. Every number this script produces is read out of a binary
        # whose field offsets come from those artefacts, so a stale one does
        # not fail: it answers, wrongly and quietly.
        #
        # This used to set `PYRE_LLBC_STRICT=1` to arm that gate. It was
        # already armed: `freshness_policy` maps an unset switch and `=1` to
        # the same `FreshnessMode::Strict`, so the assignment selected no
        # behaviour build.rs would not have taken anyway — while build.rs
        # declares `cargo::rerun-if-env-changed=PYRE_LLBC_STRICT`, so it made
        # every alternation between a cargo invocation that set the variable
        # and one that did not re-run the build script and recompile
        # `pyre-jit-trace -> pyre-jit -> pyrex`. Measured on a 16-core Windows
        # box: 1s to repeat a build at the same value, 123s and 122s to switch
        # it, 0.4s once nothing sets it. That is a working-tree cost rather
        # than a CI one — rust-cache cleans workspace crates before saving,
        # so the CI job rebuilds them either way.
        cfg = CARGO_CONFIG[backend]
        if cfg.get("wasm"):
            return self.build_wasm_backend()
        print(f"Building {cfg['bin']} (release, backend={backend})...")
        open_build_window()
        cmd = [
            "cargo", "build", "--release", "-p", "pyrex",
            "--bin", cfg["bin"], *cfg["extra"],
        ]
        print("  $ " + " ".join(cmd))
        cargo_path = shutil.which("cargo") or "(not found on PATH)"
        # `capture_output` holds cargo's progress until it exits and this
        # script's stdout is block-buffered under a harness, so without the
        # flush the whole build is one gap with nothing in it. On the Windows
        # runner it was 2766s of a 3392s step, every build line stamped at the
        # instant the buffer finally drained; the next largest gap anywhere in
        # that step was 31s.
        print(f"  cargo resolved to: {cargo_path}", flush=True)
        started = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        elapsed = time.perf_counter() - started
        if proc.returncode != 0:
            print(f"ERROR: cargo build failed (exit {proc.returncode}) after {elapsed:.1f}s")
            if proc.stdout:
                print("─── cargo stdout ───")
                print(proc.stdout.rstrip())
            if proc.stderr:
                print("─── cargo stderr ───")
                print(proc.stderr.rstrip())
            print("────────────────────")
            cargo_output = (proc.stderr or "") + (proc.stdout or "")
            if "LLBC STALE" in cargo_output:
                # build.rs refuses a stale artefact by default, which is the
                # failure that got us here. It already printed the exact
                # `extract-llbc.py` line naming the crates that moved, so
                # repeat the reason rather than the command.
                print(red("LLBC artefacts under build/llbc/ are STALE."))
                print("Field offsets come from them, so a run on this tree "
                      "would measure the wrong bytes.")
                print("Re-extract with the command build.rs printed above, "
                      "then re-run this script.")
                sys.exit(1)
            llbc_missing = (
                # translator runtime panic (majit-translate/src/lib.rs)
                "no LLBC source resolved" in cargo_output
                # pyre-jit-trace/build.rs compile-time preflight
                or "Extract the LLBC" in cargo_output
            )
            if llbc_missing:
                # The JIT front-end has no MIR to lower because build/llbc/
                # is empty. This is a setup step, not a toolchain fault — the
                # rustup diagnostics below would be noise, so point at the
                # producer instead.
                print(red("LLBC artefacts are missing under build/llbc/."))
                print("Run the extractor first, then re-run this script:")
                # No crate arguments: extract the full DEFAULT_CRATES set
                # (majit-rlib, pyre-object, pyre-interpreter, pyre-jit). The
                # exact eval::eval_loop_jit portal lives in pyre-jit.ullbc, so
                # a production JIT build requires all four artifacts.
                print("    scripts/extract-llbc.py")
            else:
                self._print_cargo_diagnostics(cargo_path)
            sys.exit(1)
        # The wall clock beside cargo's own figure is what makes a build that
        # recompiled the world distinguishable from a cache hit.
        print(f"  {cargo_finished_line(proc)} — {elapsed:.1f}s wall", flush=True)
        # The digest has to be read back after this build, not carried over
        # from an earlier one in the same run.
        invalidate_build_inputs_fingerprint()
        stamp_artefact_inputs(default_binary(backend))

    def build_wasm_backend(self):
        """Build the wasm32 `pyre-wasm` module and the native `pyre-wasm-runner`.

        The runner loads the module (via `$PYRE_WASM_MODULE`, defaulting to
        `WASM_MODULE_PATH`) and executes it under wasmtime, so two artefacts are
        produced: the wasm module (needs the export-table / custom-getrandom
        flags) and the host runner binary.
        """
        open_build_window()
        steps = [
            (
                "pyre-wasm (wasm32, --features wasm-host)",
                [
                    "cargo", *WASM_CARGO_TOOLCHAIN, "build", "--release",
                    "-p", "pyre-wasm",
                    "--target", "wasm32-unknown-unknown",
                    "--no-default-features", "--features", "wasm-host",
                    *WASM_BUILD_STD_FLAGS,
                ],
                {
                    **os.environ,
                    "RUSTFLAGS": WASM_RUSTFLAGS,
                },
            ),
            (
                "pyre-wasm-runner (native wasmtime host)",
                ["cargo", "build", "--release", "-p", "pyre-wasm-runner"],
                None,
            ),
        ]
        for label, cmd, env in steps:
            print(f"Building {label}...")
            # Flushed for the same reason `build_backend` flushes: the command
            # is the only thing naming the gap it is about to spend.
            print("  $ " + " ".join(cmd), flush=True)
            started = time.perf_counter()
            proc = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8",
                errors="replace", env=env,
            )
            elapsed = time.perf_counter() - started
            if proc.returncode != 0:
                print(f"ERROR: cargo build failed (exit {proc.returncode}) after {elapsed:.1f}s")
                if proc.stdout:
                    print("─── cargo stdout ───")
                    print(proc.stdout.rstrip())
                if proc.stderr:
                    print("─── cargo stderr ───")
                    print(proc.stderr.rstrip())
                print("────────────────────")
                sys.exit(1)
            print(f"  {cargo_finished_line(proc)} — {elapsed:.1f}s wall", flush=True)
        if not Path(WASM_BUILD_OUTPUT).exists():
            print(f"ERROR: wasm module not produced at {WASM_BUILD_OUTPUT}")
            sys.exit(1)
        # Snapshot the wasm-host build to a stable path so a later `web` build of
        # the same crate cannot overwrite the module the runner loads. Copy when
        # the bytes actually changed: rewriting an identical file would bump its
        # mtime, and the runner's `<module>.cwasm` compiled cache is keyed by
        # the module's content hash, so an identical rewrite buys nothing.
        src_bytes = Path(WASM_BUILD_OUTPUT).read_bytes()
        dst = Path(WASM_MODULE_PATH)
        if not dst.exists() or dst.read_bytes() != src_bytes:
            dst.write_bytes(src_bytes)
        # As in `build_backend`: this build resolves target-specific
        # dependencies, so `Cargo.lock` may differ from the state the native
        # backends were stamped against.
        invalidate_build_inputs_fingerprint()
        stamp_artefact_inputs(WASM_MODULE_PATH)
        stamp_artefact_inputs(default_binary("wasm"))

        if WASM_ENGINE == "wasmtime":
            self._warm_wasm_cache()

    def _warm_wasm_cache(self):
        """Compile the wasmtime `.cwasm` cache once here, untimed.

        wasmtime recompiles the whole ~14MB module (~5s) on a cold start. The
        runner caches that compilation in `<module>.cwasm`, but the wasm build
        is non-deterministic, so each rebuild yields a fresh module that
        invalidates the cache. Warming it in the build phase moves that fixed
        cost out of every measured benchmark (including the first), so the
        reported times reflect Python execution, not module compilation.

        Fuel metering emits a different module, cached separately as
        `<module>.fuel.cwasm`, so warm that one too: the wasm runtime codegen
        regression tests all run under PYRE_WASM_JIT_STATS, and they start six
        runners at once — without this every one of them recompiles.
        """
        runner = default_binary("wasm")
        if not Path(runner).exists():
            return
        env = dict(os.environ)
        env["PYRE_WASM_MODULE"] = str(Path(WASM_MODULE_PATH).resolve())
        env["PYRE_WASM_ENGINE"] = "wasmtime"
        print("Warming wasmtime module cache (.cwasm)...")
        for extra in ({}, {"PYRE_WASM_JIT_STATS": "1"}):
            try:
                subprocess.run(
                    [runner, "--engine", "wasmtime", os.devnull],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    env={**env, **extra}, timeout=120,
                )
            except (OSError, subprocess.SubprocessError):
                pass  # best-effort; a cold first bench just pays the compile once

    def _print_cargo_diagnostics(self, cargo_path):
        """Dump the toolchain state when cargo refuses to run.

        Targets the macOS CI failure mode where Swatinem/rust-cache restores
        a stale `~/.cargo/bin/cargo` rustup proxy after a runner-image bump.
        The new rustup's state under `~/.rustup/` then mismatches the old
        proxy, which falls back to rustup-init mode and clap-rejects `build`.
        """
        print("─── cargo diagnostics ───")
        print(f"PATH = {os.environ.get('PATH', '')}")
        print(f"CARGO_HOME = {os.environ.get('CARGO_HOME', '(unset)')}")
        print(f"RUSTUP_HOME = {os.environ.get('RUSTUP_HOME', '(unset)')}")
        if sys.platform != "win32":
            for which_cmd in (["which", "-a", "cargo"], ["which", "-a", "rustup"]):
                self._run_diag(which_cmd)
            cargo_dir = os.path.dirname(cargo_path) if os.path.sep in cargo_path else ""
            if cargo_dir:
                self._run_diag(["ls", "-laHi", cargo_dir])
            if cargo_dir and os.path.isfile(cargo_path):
                self._run_diag(["file", cargo_path])
        self._run_diag(["cargo", "--version"])
        self._run_diag(["rustup", "show"])
        print("─────────────────────────")

    @staticmethod
    def _run_diag(cmd):
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=10,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            print(f"  $ {' '.join(cmd)}\n    [error: {e}]")
            return
        out = (proc.stdout or "") + (proc.stderr or "")
        print(f"  $ {' '.join(cmd)} (exit {proc.returncode})")
        for line in out.rstrip().splitlines():
            print(f"    {line}")

    # ── warmup ──

    def warmup(self, script):
        sys.stdout.write(f"  {'warmup':<10s}")
        sys.stdout.flush()
        for runner in [python3(), PYPY3]:
            try:
                subprocess.run(
                    [runner, script],
                    stdout=subprocess.DEVNULL,
                    timeout=30,
                )
            except Exception:
                pass
        for backend in ALL_BACKENDS:
            if self.enabled(backend):
                try:
                    # stderr is captured, not inherited: this is the one run per
                    # backend that happens before any fixture, so it is where a
                    # binary that cannot bootstrap its own import machinery has
                    # to be caught. Nothing else reads a warmup's stderr, and it
                    # carried the `[jit-stats]` dump into the log for no reader.
                    warm = subprocess.run(
                        [self._pyre(backend), script],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        timeout=30,
                        env=pyre_env(),
                    )
                except Exception:
                    continue
                reason = _importlib_bootstrap_failure(
                    warm.stderr.decode("utf-8", "replace")
                )
                if reason:
                    print(red("BOOTSTRAP"))
                    print(f"  {backend}: {reason}")
                    print("  The interpreter falls back to the native importer "
                          "when this happens, so every fixture on this backend "
                          "runs a different import path than the other backends "
                          "and its jit-stats are not comparable to theirs.")
                    print("  The missing builtin module names the defect: this "
                          "backend's feature set has to carry whatever "
                          "`importlib._bootstrap_external` imports on this "
                          "platform.")
                    sys.exit(1)
        print(dim("done"))

    # ── single-backend bench run ──

    def _retry_performance_gate(
        self, backend, script, timeout, baseline_cmd, expected_output,
        baseline_key,
    ):
        """Median (pyre, baseline) timings, plus the samples they came from.

        A retry is deliberately limited to the failed gate: correctness has
        already passed, and a nonzero exit, changed output, or JIT panic in any
        retry remains a failure rather than being hidden by a median.

        The sample lists are returned rather than reduced away because the
        median alone does not say whether the runs it summarises agreed --
        `_retry_sample_note` reports that span on the line the verdict prints.
        """
        attempts = PERF_RETRY_RUNS
        effective_timeout = scaled_timeout(timeout, self._timeout_scale(backend))
        # The wasm gate's baseline is a pyre binary, not an oracle, so it needs
        # the pyre environment the numerator already runs under — otherwise the
        # ratio compares a pinned run against an unpinned one. It was also the
        # one pyre spawn left outside `pyre_env()`, which is how a bytecode
        # cache still appeared under a run that pins PYTHONDONTWRITEBYTECODE.
        baseline_is_pyre = baseline_key in ALL_BACKENDS
        if baseline_is_pyre:
            baseline_env = pyre_env()
        elif baseline_key == "pypy":
            baseline_env = pypy_env()
        else:
            baseline_env = None
        pyre_times = []
        baseline_times = []
        for _ in range(attempts):
            baseline_output, baseline_time, baseline_code, baseline_stderr = run_timed(
                baseline_cmd, timeout_s=effective_timeout, env=baseline_env,
            )
            if baseline_code != 0:
                return None
            # A pyre baseline is under test itself, so it answers to the same
            # bar as the numerator: a median built on a denominator that
            # printed the wrong answer or panicked out of the JIT would let the
            # failing gate pass on a time no correct run produced.
            if baseline_is_pyre and (
                baseline_output != expected_output
                or _jit_panic_reason(baseline_stderr)
            ):
                return None
            output, elapsed, code, stderr = run_timed(
                [self._pyre(backend), script], timeout_s=effective_timeout,
                env=pyre_env(),
            )
            if (
                code != 0
                or output != expected_output
                or _jit_panic_reason(stderr)
            ):
                return None
            baseline_times.append(baseline_time)
            pyre_times.append(elapsed)
        return (
            statistics.median(pyre_times), statistics.median(baseline_times),
            pyre_times, baseline_times,
        )

    def _subtracted_startup(self, key):
        """The startup `_exec_time` actually subtracted for *key*.

        The unit a gate failure's margin is reported in: nothing is subtracted
        under --no-startup-subtract, so there is no error to report against.
        """
        if self.args.no_startup_subtract:
            return 0.0
        return self.startup.get(self._startup_key(key), 0.0)

    def _baseline_exec_time_clamped(self, baseline, baseline_time):
        """Whether startup subtraction pinned a baseline to its floor."""
        exec_b = self._exec_time(baseline, baseline_time)
        return (
            not self.args.no_startup_subtract
            and exec_b not in (None, "-")
            and float(exec_b) <= EXEC_TIME_FLOOR_S
        )

    def _baseline_exec_time_thin(self, baseline, baseline_time):
        """Whether a baseline clears the floor but not the floor gate's bar.

        Reporting only: the ceiling accepts any value above
        `EXEC_TIME_FLOOR_S`, while the floor gate requires
        `FLOOR_GATE_MIN_BASELINE_S`. Mark ratios in the intervening band so
        they are not mistaken for values accepted by both directions.
        """
        exec_b = self._exec_time(baseline, baseline_time)
        return (
            not self.args.no_startup_subtract
            and exec_b not in (None, "-")
            and EXEC_TIME_FLOOR_S < float(exec_b) < FLOOR_GATE_MIN_BASELINE_S
        )

    @staticmethod
    def _compare_buffer(limit):
        """Clock granularity the comparison has to absorb, at this *limit*.

        Shared with `_gate_fail_detail`, which reports a failure's margin and
        so has to reconstruct the same bound the gate applied.
        """
        # Each execution-only value is a difference between two independently
        # measured quantities (bench - empty-program startup).  On Windows each
        # quantity may be off by one scheduler tick, so the comparison
        # ``pyre <= baseline * limit`` needs 2q on the left and 2q*limit on the
        # right.  Adding one tick to every raw sample cannot model this: those
        # additions cancel during startup subtraction.
        #
        # The right-hand ``baseline * limit`` amplifies the baseline's own
        # measurement error by `limit`, and off Windows that term was missing:
        # the buffer carried only the left-hand `BENCH_COMPARE_BUFFER_S` and no
        # allowance for a noisy denominator.  A baseline near EXEC_TIME_FLOOR_S
        # is a difference of two medians each granular to about one floor, so its
        # residual run-to-run error is ~EXEC_TIME_FLOOR_S, and the budget it sets
        # swings by `limit * EXEC_TIME_FLOOR_S`.  This makes the effective
        # ceiling `limit * (1 + EXEC_TIME_FLOOR_S / baseline)`: the grace is
        # `floor / baseline`, real only when the baseline is close to the floor
        # -- exactly where the denominator cannot be trusted -- and vanishing
        # for a baseline that is real work.  `exception_traceback_loop_forms`
        # read 35.6x on one runner and 37.1x on another against an unchanged
        # ~2.3s exec, a 0.06s baseline wobbling +-0.004s across the 36x line.
        #
        # That term stays: it is the granularity of the clock, and it is what a
        # baseline sitting at the floor is worth.  It is deliberately the ONLY
        # term -- an exec time also carries whatever its startup estimate was
        # off by, a separate quantity of a different size, and no allowance is
        # granted for it.  See the STARTUP_SAMPLES block for what was tried.
        buf = BENCH_COMPARE_BUFFER_S
        if sys.platform == "win32":
            return buf + 2 * WIN_TIMER_QUANTUM_S * (1 + limit)
        return buf + limit * EXEC_TIME_FLOOR_S

    def _performance_gate_passed(
        self, backend, script, timeout, elapsed, limit, baseline_time,
        baseline_cmd, expected_output, baseline_key, minimum=None,
    ):
        """Check one performance ratio, retrying a failure by median.

        The pass/fail decision compares execution-only (startup-subtracted)
        times; the returned elapsed/baseline stay raw for reporting. The
        returned bound is ``ceiling`` or ``floor`` when the gate fails.
        """
        compare_buffer = self._compare_buffer(limit)

        def failed_bound(measured, baseline_value):
            exec_measured = self._exec_time(backend, measured)
            exec_baseline = self._exec_time(baseline_key, baseline_value)
            # A baseline pinned to EXEC_TIME_FLOOR_S by startup subtraction is
            # not a measurement of the baseline, it is the floor constant. The
            # ratio built on it is `exec_measured / EXEC_TIME_FLOOR_S`, so the
            # recorded ceiling it is compared against is an absolute wall-clock
            # budget of `limit * EXEC_TIME_FLOOR_S` seconds -- and that budget
            # was fitted on whichever host last wrote the header. Applying it
            # elsewhere compares two hosts' wall clocks with no baseline
            # standing between them, which is what the printed line already
            # says ("ratio not a measurement") while failing the run on it.
            # Measured: `class_reassign_hot` read 27.0x and 27.3x on one host
            # and 49.2x on a CI runner against the same code, because only the
            # numerator moves.
            #
            # The earlier reading kept the ceiling armed here because
            # measured/floor is a LOWER bound on the true ratio, so a failure
            # does prove the fixture is at least that many times slower than
            # pypy. It is still not a bound this ceiling can judge: the
            # recorded number carries the same clamp, so no pypy measurement
            # enters the comparison on either side. Three consecutive `main`
            # runs failed exactly this way on three different fixtures and two
            # runners -- global_cell_shortpreamble_hot 24.1x > 19x,
            # class_reassign_hot 49.2x > 47x, reentrant_key_eq_mutation 10.3x >
            # 5x -- which is a population, not three regressions.
            #
            # Only the ceiling relaxes: the floor declines to arm below
            # FLOOR_GATE_MIN_BASELINE_S for the neighbouring reason, which a
            # clamped baseline is always under. The number stays visible either
            # way -- the comparison table prints it with a `~`. A fixture that
            # wants a ratio gate has to give pypy enough work to measure.
            if self._baseline_exec_time_clamped(baseline_key, baseline_value):
                return None
            # A startup estimate that came out high under-states every exec
            # derived from it. Against pypy and against dynasm both sides
            # subtract the same reading, so the understatement is common and
            # pushes the ratio away from 1 -- toward whichever bound the
            # fixture already sits nearer. Against cpython the two sides
            # subtract different readings and only one side moves. Both
            # distortions are real and neither is bought off here.
            # `_gate_fail_detail` reports instead how far the estimate would
            # have to be wrong to erase the failure, in units of the startup
            # subtracted, so a small margin is legible as one without a
            # constant deciding in advance that every fixture gets it.
            if exec_measured > exec_baseline * limit + compare_buffer:
                return "ceiling"
            # The measured side enters amplified by `limit / minimum`, so its
            # error is amplified with it -- which is also why the floor's
            # margin is reported against `limit / minimum`, not flat.
            if (
                minimum is not None
                and exec_baseline >= FLOOR_GATE_MIN_BASELINE_S
                and exec_measured * (limit / minimum) + compare_buffer
                < exec_baseline * limit
            ):
                return "floor"
            return None

        bound = failed_bound(elapsed, baseline_time)
        if bound is None:
            return True, None, elapsed, baseline_time, ""

        retry = self._retry_performance_gate(
            backend, script, timeout, baseline_cmd, expected_output,
            baseline_key,
        )
        if retry is not None:
            median_elapsed, median_baseline, measured_times, baseline_times = retry
            bound = failed_bound(median_elapsed, median_baseline)
            return (
                bound is None, bound, median_elapsed, median_baseline,
                self._retry_sample_note(
                    backend, measured_times, baseline_key, baseline_times,
                ),
            )

        return False, bound, elapsed, baseline_time, ""

    def _retry_sample_note(
        self, backend, measured_times, baseline_key, baseline_times,
    ) -> str:
        """`median N` plus the span each side's samples actually covered.

        The retry medians PERF_RETRY_RUNS interleaved pairs and, until this
        note, reported only that it had: the samples themselves were reduced
        and dropped. A verdict read off them rests on those samples agreeing,
        and nothing on the line said whether they did.

        `_startup_error_margin` does not answer it. That states the margin in
        units of the startup subtracted on the side the bound is distorted by,
        which measures a different error and is smallest exactly when the
        estimate is stable -- so a red produced by the fixture's own
        run-to-run variance is reported against the one term that did not
        move, in a unit that says nothing about how far the samples spread.
        That span is what this failure mode moves, so it is stated beside the
        median it was reduced to, in the same shape as the startup line's
        `[min..max]`.

        No threshold: like `_startup_error_margin`, this measures the verdict's
        context rather than deciding in advance which readings to believe.
        """
        parts = []
        for key, times in (
            (backend, measured_times), (baseline_key, baseline_times),
        ):
            if not times:
                continue
            low, high, mid = min(times), max(times), statistics.median(times)
            span = f" {(high - low) / mid:.0%}" if mid > 0 else ""
            parts.append(f"{key} {low:.2f}-{high:.2f}s{span}")
        note = f"median {PERF_RETRY_RUNS}"
        return f"{note}; {', '.join(parts)}" if parts else note

    def _gate_fail_detail(
        self, backend, baseline, measured, baseline_time, limit, bound,
        minimum=None,
    ):
        """One-line FAIL detail using the exact numbers the gate compared.

        The gate decides on startup-subtracted exec times
        rather than raw run times, so those exec times are printed alongside
        their true measured ratio and the failed gate threshold. Every number
        on the line is arithmetically self-consistent: exec_measured /
        exec_baseline equals the shown ratio, which is outside the shown gate.

        The line closes with what it would take for the startup estimate to be
        the whole story, since no allowance is granted for that error: see
        `_startup_error_margin`.
        """
        exec_m = self._exec_time(backend, measured)
        exec_b = self._exec_time(baseline, baseline_time)
        if exec_b in (None, "-") or float(exec_b) <= 0:
            ratio = "-"
        else:
            ratio = f"{float(exec_m) / float(exec_b):.1f}x"
        if bound == "floor":
            detail = (
                f"exec {exec_m:.2f}s vs {baseline} {exec_b:.2f}s  "
                f"ratio {ratio} < gate {float(minimum):g}x; "
                f"max-pypy-ratio ceiling {float(limit):g}x needs re-recording"
            )
        else:
            detail = (
                f"exec {exec_m:.2f}s > {baseline} {exec_b:.2f}s  "
                f"ratio {ratio} > gate {float(limit):g}x"
            )
        margin = self._startup_error_margin(
            backend, baseline, exec_m, exec_b, limit, bound, minimum,
        )
        return detail + margin if margin else detail

    def _startup_error_margin(
        self, backend, baseline, exec_m, exec_b, limit, bound, minimum,
    ):
        """How wrong the startup estimate has to be for this failure to vanish.

        `exec` is `bench - startup`, so an over-stated startup understates
        every exec built on it. Which execs those are decides the shape of the
        correction, and there are two cases.

        When the sides subtract DIFFERENT readings -- a cpython baseline --
        only the bound's own side moves: the ceiling's denominator, or the
        floor's numerator.

        When they subtract the SAME reading -- both graded against pypy -- the
        understatement is common, and taking a constant off both sides of a
        ratio of unequal sides pushes it away from 1. So the correction that
        erases the verdict is an over-statement, and its leverage is the gate's
        multiplier minus 1: a gate of exactly 1x is unmoved by any shared
        error, and nothing is reported for it.

        Solving the failing bound for the correction that puts it exactly on
        its threshold gives the size of the startup error the verdict rests on,
        and dividing by the startup actually subtracted states it in the only
        unit that travels between hosts.

        `0.05 x` says the run is arguing about a twentieth of an empty-program
        spawn and should not be believed; `3.0 x` says no plausible mis-reading
        reaches it. This replaces the withdrawn standing allowance, which
        answered the same question with one constant for every fixture.
        """
        # Nothing was subtracted, so there is no subtraction error to report.
        side = self._startup_key(baseline if bound == "ceiling" else backend)
        startup = self._subtracted_startup(side)
        if startup <= 0:
            return ""
        shared = self._startup_key(backend) == self._startup_key(baseline)
        buffer_s = self._compare_buffer(limit)
        if bound == "ceiling":
            # exec_m - d > (exec_b - d[shared]) * limit + buffer
            slack = float(exec_m) - float(exec_b) * limit - buffer_s
            leverage = limit - 1 if shared else limit
        else:
            if not minimum:
                return ""
            # (exec_m - d) * (limit / minimum) + buffer
            #     < (exec_b - d[shared]) * limit
            amplified = limit / minimum
            slack = (
                float(exec_b) * limit - float(exec_m) * amplified - buffer_s
            )
            leverage = amplified - limit if shared else amplified
        if slack <= 0 or leverage <= 0:
            return ""
        needed = slack / leverage / startup
        return f"; needs {needed:.3g}x {side} startup to be noise"

    def _run_backend_bench(
        self, backend, name, script, timeout,
        vs_cpython, vs_pypy, t_cpython, t_pypy, pypy_output,
        wasm_float_tol=False, max_rss_mb=None,
    ):
        pyre_bin = self._pyre(backend)
        effective_timeout = scaled_timeout(timeout, self._timeout_scale(backend))

        sys.stdout.write(f"    {backend:<10s}")
        sys.stdout.flush()

        output, elapsed, code, stderr = run_timed(
            [pyre_bin, script], timeout_s=effective_timeout, env=pyre_env(),
        )
        # Read before any gate re-runs the fixture and overwrites the slot.
        peak_rss_mb = last_run_peak_rss_mb()

        panic_reason = _jit_panic_reason(stderr)
        if panic_reason:
            self._record(backend, False, name, panic_reason)
            print(f"{red('JIT-PANIC')}  {panic_reason}")
            self._append_comparison(backend, name, t_cpython, t_pypy, "FAIL")
            return

        if code != 0:
            if code == 124:
                self._record(backend, False, name, f"timeout (>{effective_timeout}s)")
                print(f"{red('TIMEOUT')}  >{effective_timeout}s")
            else:
                # A crash whose stderr never reaches the log is undiagnosable
                # from a CI run, and a binary that dies before it can print
                # anything is exactly the case worth naming.
                detail = _first_stderr_line(stderr)
                self._record(backend, False, name, f"crash (exit {code}){detail}")
                print(f"{red('CRASH')} (exit {code}){detail}")
            self._append_comparison(backend, name, t_cpython, t_pypy, "FAIL")
            return

        # Every backend requires byte-identical output. The sole exception is a
        # bench explicitly marked wasm_float_tol=True AND run on the wasm
        # backend: wasm32's scalar ISA cannot reproduce the platform libm's
        # arch-tuned pow, so its transcendental-heavy output drifts by a bounded
        # amount (see `wasm_outputs_match`). This is opt-in per bench, never a
        # blanket wasm allowance — every other wasm bench stays byte-exact.
        if backend == "wasm" and wasm_float_tol:
            matched = wasm_outputs_match(output, pypy_output)
        else:
            matched = output == pypy_output
        if not matched:
            self._record(backend, False, name, "wrong output")
            print(f"{red('WRONG')}")
            _dump_output_mismatch(output, pypy_output)
            self._append_comparison(backend, name, t_cpython, t_pypy, "WRONG")
            return

        # Recorded before the gates so a later median retry cannot make one
        # backend's stored time depend on which gates its own run happened to
        # trip. dynasm runs before wasm in ALL_BACKENDS, so the wasm gate below
        # finds this fixture's dynasm time already in place.
        self.bench_elapsed[(backend, name)] = elapsed

        def _ratio(elapsed_val, pypy_val):
            num = self._exec_time(backend, elapsed_val)
            den = self._exec_time("pypy", pypy_val)
            if den in (None, "-") or float(den) <= 0 or num in (None, "-"):
                return "-"
            # Display only: neither marker reaches a verdict. `~` says no gate
            # was applied; `?` says one was, against a baseline under the bar
            # the floor gate uses. An unmarked ratio now means both directions
            # of the gate accepted the denominator, which is what a reader has
            # been entitled to assume all along.
            if self._baseline_exec_time_clamped("pypy", pypy_val):
                marker = "~"
            elif self._baseline_exec_time_thin("pypy", pypy_val):
                marker = "?"
            else:
                marker = ""
            return f"{marker}{float(num) / float(den):.1f}x"

        ratio = _ratio(elapsed, t_pypy) if elapsed > 0 else "-"

        # Ratio and memory failures do not invalidate the run's counters, so
        # keep comparing or recording the jit-stats baseline after they fail.
        failures = []
        retry_notes = []
        if vs_cpython and t_cpython not in (None, "-"):
            passed, bound, checked_elapsed, checked_baseline, gate_retry_note = self._performance_gate_passed(
                backend, script, timeout, elapsed, vs_cpython, float(t_cpython),
                [python3(), script], pypy_output, "cpython",
            )
            if gate_retry_note:
                retry_notes.append(gate_retry_note)
            if not passed:
                detail = self._gate_fail_detail(
                    backend, "cpython", checked_elapsed, checked_baseline,
                    vs_cpython, bound,
                )
                suffix = f" ({gate_retry_note})" if gate_retry_note else ""
                failures.append(
                    (
                        f"{red('SLOWER')}  pyre {detail}{suffix}",
                        detail,
                        fmt_time(f"{elapsed:.2f}"),
                        f"({ratio} vs pypy)",
                    )
                )
            if passed and gate_retry_note:
                elapsed = checked_elapsed
                ratio = _ratio(elapsed, t_pypy)

        if vs_pypy and t_pypy not in (None, "-"):
            # A clamped baseline declines BOTH bounds in `_performance_gate_passed`,
            # so a fixture can carry a `max-pypy-ratio` that no run has ever
            # applied. The comparison table already prints `~` on the row, but a
            # marker per row is not a set anyone can review, and absence of the
            # directive exempts a fixture entirely -- so a ceiling that never
            # arms reads exactly like one that holds. Named for the same reason
            # the wasm ratio is.
            if (
                self._baseline_exec_time_clamped("pypy", float(t_pypy))
                and name not in self.pypy_ratio_ungated
            ):
                self.pypy_ratio_ungated.append(name)
            minimum = perf_gate_floor(vs_pypy)
            passed, bound, checked_elapsed, checked_baseline, gate_retry_note = self._performance_gate_passed(
                backend, script, timeout, elapsed, vs_pypy, float(t_pypy),
                [PYPY3, script], pypy_output, "pypy", minimum,
            )
            if gate_retry_note:
                retry_notes.append(gate_retry_note)
            if not passed:
                detail = self._gate_fail_detail(
                    backend, "pypy", checked_elapsed, checked_baseline,
                    vs_pypy, bound, minimum,
                )
                suffix = f" ({gate_retry_note})" if gate_retry_note else ""
                label = "FASTER" if bound == "floor" else "SLOWER"
                failures.append(
                    (
                        f"{red(label)}  pyre {detail}{suffix}",
                        detail,
                        fmt_time(f"{elapsed:.2f}"),
                        f"({ratio} vs pypy)",
                    )
                )
            if passed and gate_retry_note:
                elapsed = checked_elapsed
                t_pypy = checked_baseline
                ratio = _ratio(elapsed, t_pypy)

        # `# pyre-check: max-rss-mb=` — the axis the ratio gates cannot see.
        # Checked after them so a fixture that is both slow and fat reports the
        # slowness first, while still reporting both failures.
        if max_rss_mb is not None and peak_rss_mb is not None and peak_rss_mb > max_rss_mb:
            detail = f"peak RSS {peak_rss_mb:.0f}MB > {max_rss_mb:.0f}MB"
            failures.append(
                (
                    f"{red('FATTER')}  pyre {detail}",
                    detail,
                    fmt_time(f"{elapsed:.2f}"),
                    f"({ratio} vs pypy)",
                )
            )

        # wasm carries no tuned per-fixture ratio, so it gates against dynasm's
        # execution-only time from this same invocation instead. Placed after
        # the other ratio gates so a fixture that is slow against several
        # baselines still reports each one.
        if backend == "wasm":
            dynasm_elapsed = self.bench_elapsed.get(("dynasm", name))
            dynasm_exec = (
                self._exec_time("dynasm", dynasm_elapsed)
                if dynasm_elapsed is not None
                else None
            )
            # Both sides of this ratio are measured in this invocation, so the
            # denominator carries the full startup-subtraction error rather
            # than a recorded constant's stability. `_baseline_exec_time_clamped`
            # would only reject one already pinned to EXEC_TIME_FLOOR_S, which
            # leaves the band up to FLOOR_GATE_MIN_BASELINE_S dividing by
            # something the same size as its own error.
            ceiling = wasm_ratio_gate(script) or WASM_MAX_DYNASM_RATIO
            allowed = ceiling > WASM_MAX_DYNASM_RATIO
            if dynasm_exec in (None, "-") or (
                float(dynasm_exec) < FLOOR_GATE_MIN_BASELINE_S
            ):
                self.wasm_ratio_ungated.append(name)
                if allowed:
                    self.wasm_ratio_allowed.append((name, ceiling, None))
            else:
                # This ceiling is meant to come back down as the backend closes
                # the gap, so it has to mean the number it states: a standing
                # allowance underneath it would be re-fitted along with it, and
                # the tightening would buy less than it says it does. That is
                # now true of every ratio gate rather than this one alone, so
                # the opt-out this call site used to carry is gone. The
                # denominator concern it does have is handled above, by
                # declining the gate outright and naming the fixture in the
                # summary rather than quietly widening the bound.
                passed, bound, checked_elapsed, checked_baseline, gate_retry_note = (
                    self._performance_gate_passed(
                        backend, script, timeout, elapsed,
                        ceiling, dynasm_elapsed,
                        [self._pyre("dynasm"), script], pypy_output, "dynasm",
                    )
                )
                if gate_retry_note:
                    retry_notes.append(gate_retry_note)
                if allowed:
                    # The values the gate judged, not the first sample: a
                    # median retry is what an allowance fitted to a slow
                    # reading would otherwise be reported against.
                    self.wasm_ratio_allowed.append((
                        name, ceiling,
                        float(self._exec_time(backend, checked_elapsed))
                        / float(self._exec_time("dynasm", checked_baseline)),
                    ))
                if not passed:
                    detail = self._gate_fail_detail(
                        backend, "dynasm", checked_elapsed, checked_baseline,
                        ceiling, bound,
                    )
                    suffix = f" ({gate_retry_note})" if gate_retry_note else ""
                    failures.append(
                        (
                            f"{red('SLOWER')}  pyre {detail}{suffix}",
                            detail,
                            fmt_time(f"{elapsed:.2f}"),
                            f"({ratio} vs pypy)",
                        )
                    )

        snap_status, snap_reason = self._apply_snapshot_gate(
            backend, name, script, output, stderr, elapsed, timeout,
        )
        if snap_status != "ok":
            if snap_status in ("unstable", "banded"):
                # Warned, not failed: unstable counters cannot be gated, while
                # banded counters stayed inside their declared allowance.
                note_label = snap_status.upper()
                note_line = f"{yellow(note_label)}  {snap_reason}"
            else:
                # `improved` is still a failure; the label only tells the
                # reader whether to investigate or just re-record.
                label = "IMPROVED" if snap_status == "improved" else "SNAPDIFF"
                paint = yellow if snap_status == "improved" else red
                failures.append(
                    (f"{paint(label)}  {snap_reason}", snap_reason, label, None)
                )
                note_line = None
        else:
            note_line = None

        if failures:
            self._record(
                backend, False, name,
                "; ".join(detail for _, detail, _, _ in failures),
            )
            for index, (line, _, _, _) in enumerate(failures):
                print(f"{' ' * 14 if index else ''}{line}")
            if note_line is not None:
                print(f"{' ' * 14}{note_line}")
            _, _, comparison_cell, comparison_note = failures[0]
            self._append_comparison(
                backend, name, t_cpython, t_pypy,
                comparison_cell, comparison_note,
            )
            return

        if note_line is not None:
            self._record(backend, True, name, f"{elapsed:.2f}s")
            print(note_line)
            self._append_comparison(backend, name, t_cpython, t_pypy, note_label)
            return

        self._record(backend, True, name, f"{elapsed:.2f}s")
        retry_note = "; ".join(retry_notes)
        suffix = f" ({retry_note})" if retry_note else ""
        print(f"{green('PASS')}  {elapsed:.2f}s{suffix}")
        self._append_comparison(
            backend, name, t_cpython, t_pypy,
            fmt_time(f"{elapsed:.2f}"), f"({ratio} vs pypy)",
        )

    # ── top-level bench entry ──

    def run_bench(
        self, name, script, timeout,
        dynasm_vs_cpython=None, dynasm_vs_pypy=None,
        cranelift_vs_cpython=None, cranelift_vs_pypy=None,
        skip_backends=(), *, wasm_float_tol=False,
    ):
        """Run one benchmark on each enabled backend."""
        self.maybe_refresh_startups()
        need_cpython = False
        if (
            self.enabled("dynasm")
            and "dynasm" not in skip_backends
            and dynasm_vs_cpython
        ):
            need_cpython = True
        if (
            self.enabled("cranelift")
            and "cranelift" not in skip_backends
            and cranelift_vs_cpython
        ):
            need_cpython = True
        if self.args.full:
            need_cpython = True

        print(f"  {name}")

        t_cpython = "-"
        cpython_code = 0
        if need_cpython:
            sys.stdout.write(f"    {'cpython':<10s}")
            sys.stdout.flush()
            cpython_output, t_cpu, cpython_code, _ = run_timed([python3(), script])
            t_cpython = t_cpu
            if cpython_code != 0:
                print(f"{red('CRASH')} (exit {cpython_code})")
            else:
                print(f"{dim('done')}  {t_cpython:.2f}s")

        sys.stdout.write(f"    {'pypy':<10s}")
        sys.stdout.flush()
        pypy_output, pypy_cpu, pypy_code, _ = run_timed(
            [PYPY3, script], env=pypy_env(),
        )
        t_pypy = pypy_cpu if pypy_code == 0 else "-"
        if pypy_code != 0:
            print(f"{red('CRASH')} (exit {pypy_code})")
            for backend in ALL_BACKENDS:
                if self.enabled(backend):
                    self._record(backend, False, name, "pypy crash")
                    self._append_comparison(backend, name, t_cpython, "-", "FAIL")
            return
        print(f"{dim('done')}  {t_pypy:.2f}s")

        for backend, vs_cpython, vs_pypy in [
            ("dynasm", dynasm_vs_cpython, dynasm_vs_pypy),
            ("cranelift", cranelift_vs_cpython, cranelift_vs_pypy),
            ("wasm", None, None),
        ]:
            if not self.enabled(backend):
                continue
            if backend in skip_backends:
                sys.stdout.write(f"    {backend:<10s}")
                print(dim("skip"))
                self._append_comparison(
                    backend, name, t_cpython, t_pypy, "skip",
                )
                continue
            if vs_cpython and cpython_code != 0:
                sys.stdout.write(f"    {backend:<10s}")
                print(f"{red('FAIL')}  missing cpython baseline")
                self._record(backend, False, name, "cpython crash")
                self._append_comparison(backend, name, "-", t_pypy, "FAIL")
                continue
            self._run_backend_bench(
                backend, name, script, timeout,
                vs_cpython, vs_pypy, t_cpython, t_pypy, pypy_output,
                wasm_float_tol=wasm_float_tol,
            )

    # ── self-checking regression guard ──

    def run_selfcheck(self, name, script, timeout, expect="PASS", skip_backends=None,
                      require_jit=True, spec_folds=(), trace_shapes=(),
                      want_compiles=()):
        """Run a self-checking regression script on each enabled backend.

        The script asserts its own invariant (exit 0 AND prints *expect*);
        check.py only honors the exit code and the required marker. Used for
        guards whose signal is an asserted invariant, not byte-identical
        output, so they cannot go through `run_bench` or synthetic parity.

        *skip_backends* maps each backend the guard does not apply to onto
        the reason printed beside its `skip` (e.g. a `time`-module timing
        guard cannot run on the wasm guest, which has no `time` module); see
        `_fixture_skips`.

        With *require_jit* the run must also have compiled every
        `(arm, name)` in *want_compiles*. A self-asserted invariant is
        satisfied by an interpreted run, so without this a fixture guarding a
        compiled mis-admission passes while establishing nothing — and it would
        go on passing if a shape it guards stopped reaching the JIT, which is
        the change most likely to make the guard vacuous.
        `synth_selfcheck_interpreted` turns it off for a fixture whose
        invariant is not about compiled code.

        *want_compiles* names those shapes, so the answer is about them and not
        about the file, and carries which compile arm has to mint each one, so
        a loop that degrades into a root trace is reported rather than counted.
        Both come from the same run's `PYRE_LOOP_CENSUS=1` lines — no second
        process, and the question is asked once per backend, which is where it
        has to be asked: a fold admitted on dynasm can be declined on wasm, and
        only a per-backend census sees that. `synth_selfcheck_compiles` carries
        the list and records why it replaced a count.

        *spec_folds* is the `# pyre-check: spec-folds=` list, read the same way
        `run_synthetic_bench` reads it. A fixture written to guard a trace-time
        fold asserts the ANSWER, and the residual answers identically -- so
        without the census the fixture passes just as well with the fold gone,
        which is the one failure it exists to catch.

        *trace_shapes* is the `# pyre-check: trace-shape=` list. It asks the
        same question of the optimized trace instead of the fold counter: the
        named loop's compiled operations have to match every assertion.
        """
        skip_backends = skip_backends or {}
        print(f"  {name}")
        if trace_shapes and not self._check_trace_shapes(
            name, script, trace_shapes, timeout, "-", "-", skip_backends,
        ):
            return
        if spec_folds and not self._check_spec_folds(
            name, script, spec_folds, timeout, "-", "-", skip_backends,
        ):
            return
        for backend in ALL_BACKENDS:
            if not self.enabled(backend):
                continue
            if backend in skip_backends:
                self._print_skip(backend, skip_backends[backend])
                self._append_comparison(backend, name, "-", "-", "skip")
                continue
            effective_timeout = scaled_timeout(timeout, self._timeout_scale(backend))
            sys.stdout.write(f"    {backend:<10s}")
            sys.stdout.flush()
            # The census rides this run: a selfcheck fixture is not measured
            # against a ratio gate (its cpython/pypy cells are `-`), so the
            # extra stderr lines cost nothing a gate reads, and asking here
            # keeps the answer per backend.
            selfcheck_env = pyre_env()
            selfcheck_env["PYRE_LOOP_CENSUS"] = "1"
            output, elapsed, code, stderr = run_timed(
                [self._pyre(backend), script],
                timeout_s=effective_timeout,
                env=selfcheck_env,
            )
            panic_reason = _jit_panic_reason(stderr)
            if panic_reason:
                self._record(backend, False, name, panic_reason)
                print(f"{red('JIT-PANIC')}  {panic_reason}")
                self._append_comparison(backend, name, "-", "-", "FAIL")
                continue
            if code != 0:
                detail = (
                    f"timeout (>{effective_timeout}s)" if code == 124
                    else f"exit {code}"
                )
                self._record(backend, False, name, detail)
                print(f"{red('FAIL')}  {detail}")
                _dump_failed_run(output, stderr)
                self._append_comparison(backend, name, "-", "-", "FAIL")
                continue
            if expect not in output:
                self._record(backend, False, name, f"missing '{expect}'")
                print(f"{red('FAIL')}  missing '{expect}'")
                _dump_failed_run(output, stderr)
                self._append_comparison(backend, name, "-", "-", "FAIL")
                continue
            if require_jit:
                saw = selfcheck_compiled_census(stderr)
                missing = [want for want in want_compiles if want not in saw]
                if missing:
                    if not saw:
                        seen = "the run compiled nothing at all"
                    else:
                        seen = "compiled: " + ", ".join(
                            f"{arm}:{name}" for arm, name in sorted(set(saw))
                        )
                    wanted = ", ".join(f"{arm}:{name}" for arm, name in missing)
                    detail = (
                        f"nothing compiled for {wanted} — {seen}. A shape this "
                        "guard covers stopped reaching the JIT the way it was "
                        "declared to, so its assertion no longer says anything "
                        "about that shape"
                    )
                    self._record(backend, False, name, detail)
                    print(f"{red('FAIL')}  {detail}")
                    _dump_failed_run(output, stderr)
                    self._append_comparison(backend, name, "-", "-", "FAIL")
                    continue
            self._record(backend, True, name, "")
            print(f"{green('PASS')}  {elapsed:.2f}s")
            self._append_comparison(backend, name, "-", "-", f"{elapsed:.2f}s")

    # ── vendored CPython suite ──

    def run_cpython_suite(self):
        """Gate the vendored CPython test suite against its recorded baseline.

        Every other stage in this file runs code we wrote, so it can only
        catch a defect in a shape somebody already thought of.  This one runs
        CPython's own tests, and it is the stage that reports a JIT miscompile
        nobody wrote a fixture for: `test.test_datetime` failed on a
        specialised-pair subscript fold while the whole synthetic corpus and
        every parity fixture stayed green.

        The baseline records one verdict per module per backend and follows
        the host: `run.py` reads a `baseline.<platform>-<machine>.json`
        overlay before the shared file, so a verdict a host disagrees with
        (dynasm's codegen is arch-specific) is recorded there rather than
        making the stage unusable off one machine.

        Off by default and reached only through `--cpython-suite`: the suite
        costs more wall time than every other stage here put together, and the
        `cargo-test-linux` CI job already runs it on its own schedule.  Pass the
        flag locally when a change could move a verdict the synthetic corpus
        does not cover.
        """
        name = "cpython-suite"
        backend = "dynasm"
        print(f"  {name}")
        if not self.enabled(backend):
            sys.stdout.write(f"    {backend:<10s}")
            print(dim("skip (backend not enabled)"))
            self._append_comparison(backend, name, "-", "-", "skip")
            return
        sys.stdout.write(f"    {backend:<10s}")
        sys.stdout.flush()
        output, elapsed, code, stderr = run_timed(
            [
                python3(), str(Path(__file__).parent / "cpython_tests" / "run.py"),
                "--binary", str(self._pyre(backend)),
                "--baseline", str(CPYTHON_SUITE_BASELINE),
                "--jobs", str(max(1, (os.cpu_count() or 4) - 1)),
                "--timeout", str(CPYTHON_SUITE_MODULE_TIMEOUT_S),
            ],
            timeout_s=CPYTHON_SUITE_TIMEOUT_S,
            env=pyre_env(),
        )
        # `N to run,` rather than the runner's own "no regressions": a
        # selection that came out empty prints every counter as zero and
        # still exits 0, which reads as a pass and tests nothing.
        #
        # The default runner gates only modules whose recorded baseline passes.
        # Report skipped and non-gated counts so this subset is not mistaken
        # for the full suite.
        selected = re.search(
            r"^(\d+) to run, (\d+) skipped(?:, (\d+) not gated \(non-PASS\))?,",
            output,
            re.M,
        )
        if code == 124:
            detail = f"timeout (>{CPYTHON_SUITE_TIMEOUT_S}s)"
        elif selected is None:
            detail = "runner printed no selection line"
        elif selected.group(1) == "0":
            detail = "selected 0 modules"
        elif code != 0:
            regressions = re.findall(r"^  - (\S+): (.*)$", output, re.M)
            detail = "; ".join(f"{m} {d}" for m, d in regressions) or f"exit {code}"
        else:
            detail = ""
        if detail:
            self._record(backend, False, name, detail)
            print(f"{red('FAIL')}  {detail}")
            _dump_failed_run(output, stderr)
            self._append_comparison(backend, name, "-", "-", "FAIL")
            return
        self._record(backend, True, name, "")
        ran, not_run, degated = (
            selected.group(1), selected.group(2), selected.group(3) or "0",
        )
        print(
            f"{green('PASS')}  {ran} gated, {degated} not gated, "
            f"{not_run} skipped, {elapsed:.0f}s"
        )
        self._append_comparison(backend, name, "-", "-", f"{elapsed:.0f}s")

    # ── synthetic parity suite ──

    def _check_spec_folds(
        self, name, path, spec_folds, timeout, t_cpython, t_pypy, skip_backends=(),
    ):
        """True if every fold the fixture declares fired; else record and report.

        The census reads the same on every backend that shares the trace
        walker, so this runs once rather than per backend, preferring a native
        one: it needs no export lookup and its stderr carries no runner
        chatter. A wasm-only run reads the census back through the guest's
        `pyre_fbw_spec_census` export instead of skipping, which would leave
        the gate vacuous exactly where nothing else observes the walker.

        Because the backend is chosen here rather than by the caller, *timeout*
        is the fixture's own unscaled budget: the scale belongs to a backend,
        and which one that is is not known until the line above. A caller that
        hands over its own already-scaled figure squares the scale.

        *skip_backends* holds the backends `_fixture_skips` opts the fixture
        out of; the census never runs on one of them.
        """
        sys.stdout.write(f"    {'folds':<10s}")
        sys.stdout.flush()
        backend = next(
            (
                b for b in ALL_BACKENDS
                if self.enabled(b) and b != "wasm" and b not in skip_backends
            ),
            None,
        )
        if backend is None:
            if not self.enabled("wasm") or "wasm" in skip_backends:
                print(dim("skip (no backend enabled)"))
                return True
            backend = "wasm"
        census, code = spec_fold_census(
            self._pyre(backend),
            path,
            scaled_timeout(timeout, self._timeout_scale(backend)),
            wasm=backend == "wasm",
        )
        if census is None:
            detail = f"spec-folds census run failed (exit {code})"
            print(f"{red('FAIL')}  {detail}")
        else:
            unknown = [f for f in spec_folds if f not in census]
            quiet = [f for f in spec_folds if census.get(f) == 0]
            if unknown:
                detail = f"unknown fold label(s): {', '.join(unknown)}"
                print(f"{red('FAIL')}  {detail}")
            elif quiet:
                detail = f"declared fold(s) never fired: {', '.join(quiet)}"
                print(f"{red('FAIL')}  {detail}")
            else:
                print(f"{dim('done')}  {len(spec_folds)} fired")
                return True
        for b in ALL_BACKENDS:
            if self.enabled(b) and b not in skip_backends:
                self._record(b, False, name, detail)
                self._append_comparison(b, name, t_cpython, t_pypy, "FAIL")
        return False

    def _check_trace_shapes(
        self, name, path, trace_shapes, timeout, t_cpython, t_pypy, skip_backends=(),
    ):
        """True if every declared trace shape holds; else record and report.

        One native run, the same choice `_check_spec_folds` makes: the
        optimized trace is produced before a backend lowers it, so a second
        backend would re-read the same operations. The run sets `MAJIT_LOG`
        and `PYRE_LOOP_CENSUS`, which already exist; the loop name in the
        census is what selects the `jit-log-opt-loop` section.

        *skip_backends* holds the backends `_fixture_skips` opts the fixture
        out of; the shape run never uses one of them, so a binary built
        without a `requires-modules=` module does not read as a missing trace.
        """
        sys.stdout.write(f"    {'shape':<10s}")
        sys.stdout.flush()
        backend = next(
            (
                b for b in ALL_BACKENDS
                if self.enabled(b) and b != "wasm" and b not in skip_backends
            ),
            None,
        )
        if backend is None:
            if not any(
                self.enabled(b) and b not in skip_backends for b in ALL_BACKENDS
            ):
                print(dim("skip (no backend enabled)"))
                return True
            detail = "trace-shape needs a native backend (no optimized-trace log on wasm)"
            print(f"{red('FAIL')}  {detail}")
            for b in ALL_BACKENDS:
                if self.enabled(b) and b not in skip_backends:
                    self._record(b, False, name, detail)
                    self._append_comparison(b, name, t_cpython, t_pypy, "FAIL")
            return False
        env = pyre_env()
        env["MAJIT_LOG"] = "1"
        env["PYRE_LOOP_CENSUS"] = "1"
        _, _, code, err = run_timed(
            [self._pyre(backend), path],
            timeout_s=scaled_timeout(timeout, self._timeout_scale(backend)),
            env=env,
            keep_output_on_timeout=True,
        )
        failures = trace_shape_failures(err, trace_shapes) if err else [
            "trace-shape: no optimized trace",
        ]
        # A killed run can still have compiled the loop: the dump happens at
        # compile time, and the log of the iterations after it is what spends
        # the budget. A trace that meets every assertion is the gate's answer.
        # Any other non-zero status is the program failing, which this check
        # does not excuse — a trace that happens to match is not a pass.
        if not failures and code in (0, 124):
            print(f"{dim('done')}  {len(trace_shapes)} loop(s)")
            return True
        if failures:
            detail = "; ".join(failures)
        else:
            # The trace met every assertion and the process still failed.
            # The shape is not what went wrong.
            detail = f"trace-shape run failed (exit {code})"
            if err:
                tail = "\n".join(err.rstrip().splitlines()[-40:])
                print("\n─── trace-shape stderr (tail) ───")
                print(tail)
        print(f"{red('FAIL')}  {detail}")
        for b in ALL_BACKENDS:
            if self.enabled(b) and b not in skip_backends:
                self._record(b, False, name, detail)
                self._append_comparison(b, name, t_cpython, t_pypy, "FAIL")
        return False

    def run_synthetic_bench(self, path, timeout, headers):
        self.maybe_refresh_startups()
        name = f"synth/{Path(path).stem}"
        effective_timeout = scaled_timeout(timeout, self.args.timeout_scale)
        max_pypy_ratio = headers["max_pypy_ratio"]
        max_rss_mb = headers["max_rss_mb"]
        skip_backends = self._fixture_skips(headers)
        skip_cpython = headers["skip_cpython"]
        no_cpython = headers["no_cpython"]
        spec_folds = headers["spec_folds"]
        trace_shapes = headers["trace_shapes"]

        print(f"  {name}")
        # `# pyre-check: skip-platforms=` — the oracle cannot run on this host.
        if sys.platform in headers["skip_platforms"]:
            print(f"    {dim('skip')} ({sys.platform})")
            return

        sys.stdout.write(f"    {'cpython':<10s}")
        sys.stdout.flush()
        if skip_cpython or no_cpython:
            # `# pyre-check: skip-cpython` — the fixture is sized past what
            # cpython can usefully run, so running it would spend the whole
            # reference timeout to arrive at the same drop.
            # `# pyre-check: no-cpython` — the fixture's subject is a place
            # pyre follows pypy and the reference does something else, so the
            # reference is not an oracle here. Same shape as the timeout arm
            # below: `cpython_output = None` leaves pypy as the only baseline
            # the backends are compared against.
            cpython_output = None
            # Spelled the same way as the timeout skip below, so the comparison
            # table's cpython column reads the same for every kind of skip.
            t_cpython = "skip (pypy oracle)" if no_cpython else "skip (opt)"
            print(dim(t_cpython))
            self.cpython_declared_skips.append(name)
        else:
            # Output oracle and display-only time: `_run_backend_bench` is
            # passed `vs_cpython=None`, so no gate reads `t_cpython`. The
            # non-synthetic `run_bench` path does feed a gate and is not cached.
            timeout_s = SYNTHETIC_CPYTHON_REFERENCE_TIMEOUT_S

            def _run_cpython_reference():
                output, elapsed, code, _err = run_timed(
                    [python3(), path], timeout_s=timeout_s,
                )
                return output, elapsed, code

            if self.oracle_store is None:
                cpython_output, cpython_time, cpython_code = _run_cpython_reference()
                cpython_cached = False
            else:
                key = oracle_cache.entry_key(
                    kind="synth",
                    python=python3(),
                    argv=[path],
                    timeout=timeout_s,
                    fixture=path,
                    env=None,
                )
                cpython_output, cpython_time, cpython_code, cpython_cached = (
                    oracle_cache.lookup_or_run(
                        self.oracle_store,
                        key,
                        self.oracle_refresh,
                        _run_cpython_reference,
                        oracle_cache.fixture_label(path),
                    )
                )
            cached_tag = " (cached)" if cpython_cached else ""
            if cpython_code == 124:
                # Not a crash: the fixture is sized for the JITs and cpython is
                # only the reference. Drop it and let pypy be the baseline the
                # backends are compared against, rather than spend the fixture's
                # whole timeout on an interpreter nothing is gated on.
                cpython_output = None
                t_cpython = f"skip (>{SYNTHETIC_CPYTHON_REFERENCE_TIMEOUT_S:g}s)"
                print(dim(t_cpython + cached_tag))
                self.cpython_reference_skips.append(name)
            elif cpython_code != 0:
                print(f"{red('CRASH')} (exit {cpython_code})")
                for backend in ALL_BACKENDS:
                    if self.enabled(backend):
                        self._record(backend, False, name, "cpython crash")
                        self._append_comparison(backend, name, "-", "-", "FAIL")
                return
            else:
                t_cpython = cpython_time
                print(f"{dim('done')}  {t_cpython:.2f}s{cached_tag}")

        sys.stdout.write(f"    {'pypy':<10s}")
        sys.stdout.flush()
        pypy_output, pypy_time, pypy_code, _ = run_timed(
            [PYPY3, path], timeout_s=effective_timeout, env=pypy_env(),
        )
        if pypy_code != 0:
            print(f"{red('CRASH')} (exit {pypy_code})")
            for backend in ALL_BACKENDS:
                if self.enabled(backend):
                    self._record(backend, False, name, "pypy crash")
                    self._append_comparison(backend, name, t_cpython, "-", "FAIL")
            return
        t_pypy = pypy_time
        print(f"{dim('done')}  {t_pypy:.2f}s")

        if cpython_output is not None and cpython_output != pypy_output:
            print(f"    {'baseline':<10s}{red('WRONG')}  cpython output differs from pypy")
            for backend in ALL_BACKENDS:
                if self.enabled(backend):
                    self._record(backend, False, name, "cpython/pypy output mismatch")
                    self._append_comparison(
                        backend, name, t_cpython, t_pypy, "BASEFAIL",
                    )
            return

        # The oracles agree, so this output is what the fixture is for -- and
        # the place to hold its own `# Expected` comments to it. They were
        # documentation nothing read, so they rotted: three named a count the
        # fixture had not printed since its size was last doubled.
        expected_mismatches = _expected_mismatches(path, pypy_output)
        if expected_mismatches:
            claim, line = expected_mismatches[0]
            detail = f"{line.strip()} (fixture prints something else)"
            print(f"    {'expected':<10s}{red('STALE')}  {detail}")
            for backend in ALL_BACKENDS:
                if self.enabled(backend):
                    self._record(backend, False, name, f"stale expectation: {claim}")
                    self._append_comparison(
                        backend, name, t_cpython, t_pypy, "BASEFAIL",
                    )
            return

        if trace_shapes and not self._check_trace_shapes(
            name, path, trace_shapes, timeout, t_cpython, t_pypy, skip_backends,
        ):
            return

        if spec_folds and not self._check_spec_folds(
            name, path, spec_folds, timeout, t_cpython, t_pypy, skip_backends,
        ):
            return

        for backend in ALL_BACKENDS:
            if not self.enabled(backend):
                continue
            # `# pyre-check: skip-backends=` — the fixture guards a mechanism
            # this backend does not implement, so running it here would only
            # restate that gap. `# pyre-check: requires-modules=` — this
            # backend's binary was built without a module the fixture imports.
            if backend in skip_backends:
                self._print_skip(backend, skip_backends[backend])
                self._append_comparison(backend, name, t_cpython, t_pypy, "skip")
                continue
            # wasm carries no perf-ratio gate, matching `run_bench` (which has no
            # `wasm_vs_*` parameter at all): it legitimately runs a few× slower
            # than the native backends a fixture's ratio is tuned for, so one
            # ratio cannot gate both — see WASM_TIMEOUT_SCALE. Its per-bench
            # timeout stays the hang guard.
            vs_pypy = None if backend == "wasm" else max_pypy_ratio
            self._run_backend_bench(
                backend, name, path, timeout,
                None, vs_pypy, t_cpython, t_pypy, pypy_output,
                max_rss_mb=None if backend == "wasm" else max_rss_mb,
            )

    def run_synthetic_suite(self):
        pattern = self.args.synthetic_pattern
        paths = synthetic_bench_paths(pattern)
        if not paths:
            print(f"{red('ERROR')}: no synthetic benchmarks matched {pattern!r}")
            sys.exit(1)

        print(bold("synthetic parity suite"))
        print(dim(f"{len(paths)} benchmark(s), pattern={pattern!r}"))
        # Read every fixture's header before running any of them: a missing or
        # malformed directive is an authoring error, and it is reported with
        # the others of its kind rather than one per run. `--check-headers`
        # runs this same pass with nothing built, so ordinarily it has already
        # answered by the time the suite reaches here.
        headers, unusable = read_synthetic_headers(paths)
        if unusable:
            report_unusable_headers(unusable)
            sys.exit(1)

        for path in paths:
            header = headers[path]
            if header["selfcheck"]:
                self.run_selfcheck(
                    f"synth/{path.stem}",
                    str(path),
                    self.args.synthetic_timeout,
                    skip_backends=self._fixture_skips(header),
                    require_jit=not header["interpreted"],
                    spec_folds=header["spec_folds"],
                    trace_shapes=header["trace_shapes"],
                    want_compiles=header["want_compiles"],
                )
            else:
                self.run_synthetic_bench(
                    str(path), self.args.synthetic_timeout, header,
                )
        # A fixture that loses its cpython reference also loses the
        # cpython/pypy output cross-check, so the count belongs in the summary
        # rather than only in the per-fixture line it scrolled past. A fixture
        # appearing here for the first time is a fixture whose baseline just
        # got weaker.
        if self.cpython_reference_skips:
            names = ", ".join(self.cpython_reference_skips)
            print(
                dim(
                    f"cpython reference skipped (>{SYNTHETIC_CPYTHON_REFERENCE_TIMEOUT_S:g}s) "
                    f"for {len(self.cpython_reference_skips)}: {names}"
                )
            )
        # A `max-pypy-ratio` whose baseline was pinned to the floor every time
        # it ran is a recorded number nobody checked.
        if self.pypy_ratio_ungated:
            names = ", ".join(self.pypy_ratio_ungated)
            print(
                dim(
                    "max-pypy-ratio not evaluated for "
                    f"{len(self.pypy_ratio_ungated)} fixture(s): pypy's "
                    "execution-only time was clamped to the floor, so no ratio "
                    f"gate was applied to them: {names}"
                )
            )
        # A wasm run with no usable dynasm denominator beside it leaves
        # WASM_MAX_DYNASM_RATIO with nothing to divide by. Said out loud
        # because the per-fixture line for an unevaluated gate is the same
        # green as a satisfied one.
        if self.wasm_ratio_ungated:
            print(
                dim(
                    f"wasm/dynasm {WASM_MAX_DYNASM_RATIO:g}x ratio not evaluated for "
                    f"{len(self.wasm_ratio_ungated)} fixture(s): dynasm did not run "
                    f"them in this invocation, or its execution-only time stayed "
                    f"under {FLOOR_GATE_MIN_BASELINE_S * 1000:g}ms"
                )
            )
        # Counters a fixture's header took out of its own gate. Named for the
        # same reason as the two above: the fixture prints green either way, and
        # the exemption is only reviewable if it is said out loud.
        if self.jitstats_ungated:
            print(
                dim(
                    f"jit-stats counters ungated by header for "
                    f"{len(self.jitstats_ungated)}: "
                    + "; ".join(self.jitstats_ungated)
                )
            )
        # The same weaker baseline, asked for rather than discovered. Listed
        # apart so the two are not read as one number: this one only grows when
        # someone writes the header line.
        if self.cpython_declared_skips:
            names = ", ".join(self.cpython_declared_skips)
            print(
                dim(
                    f"cpython reference skipped (header) "
                    f"for {len(self.cpython_declared_skips)}: {names}"
                )
            )

    # ── printing ──

    def print_backend_config(self):
        parts = []
        for b in ALL_BACKENDS:
            if self.enabled(b):
                label = b if b != "wasm" else f"wasm/{WASM_ENGINE}"
                parts.append(f"{label}={self._pyre(b)}(x{self._timeout_scale(b)})")
        if parts:
            print(f"backend: {' '.join(parts)}")
        print(self.oracle_cache_line)

    def print_comparison_table(self):
        if not self.comparisons:
            return
        cols = [b for b in ALL_BACKENDS if self.enabled(b)]
        if not cols:
            return

        print(bold("Comparison"))

        header = f"  {'benchmark':<35s} {'cpython':>8s} {'pypy':>8s}"
        header += "".join(f" {b:>18s}" for b in cols)
        print(header)
        if any("~" in c[b] for c in self.comparisons for b in cols):
            print(
                "  ~ pypy exec clamped to floor; ratio is not a measurement, "
                "and no ratio gate is applied to it"
            )
        # Sniffing the rendered cell, like the line above: no other cell value
        # contains a `?` (they are times, ratios, FAIL/WRONG/UNSTABLE/skip/-).
        if any("?" in c[b] for c in self.comparisons for b in cols):
            print(
                "  ? pypy exec is above the floor but under "
                "FLOOR_GATE_MIN_BASELINE_S; the ceiling is applied, the floor "
                "gate declines the same baseline as too small to judge"
            )
        print("  " + "─" * (54 + 19 * len(cols)))
        for c in self.comparisons:
            row = f"  {c['name']:<35s} {c['cpython']:>8s} {c['pypy']:>8s}"
            row += "".join(f" {c[b]:>18s}" for b in cols)
            print(row)

    def print_summary(self):
        print()
        self.print_comparison_table()
        print()

        if self.wasm_ratio_allowed:
            rows = sorted(set(self.wasm_ratio_allowed))
            names = ", ".join(
                f"{name} {ceiling:g}x "
                + (f"(measured {measured:.1f}x)" if measured is not None
                   else "(gate declined this run)")
                for name, ceiling, measured in rows
            )
            print(
                dim(
                    f"wasm/dynasm ratio raised above {WASM_MAX_DYNASM_RATIO:g}x by "
                    f"`# pyre-check: max-wasm-ratio` for: {names}"
                )
            )
            # `* WASM_RATIO_FIT_HEADROOM` and not a bare comparison: without
            # it math_folds_hot, which reads 3.9x under a 4.0 ceiling, is named
            # as outgrown on the strength of 0.07x. The caveat is there because
            # a run only measures its own host and an allowance may be fitted
            # to another -- math_folds_hot's 13x is a darwin-arm64 reading of
            # 11.3x, which no ubuntu run can see.
            outgrown = [
                n for n, _, m in rows
                if m is not None
                and m * WASM_RATIO_FIT_HEADROOM <= WASM_MAX_DYNASM_RATIO
            ]
            if outgrown:
                print(
                    dim(
                        f"  {len(outgrown)} of those measured far enough under "
                        f"{WASM_MAX_DYNASM_RATIO:g}x on THIS host to fit "
                        f"without an allowance; check what host the fixture's "
                        f"note fitted it to before removing: "
                        f"{', '.join(outgrown)}"
                    )
                )

        if self.results:
            print("─" * 53)
            for r in self.results:
                print(f"  {r}")
            print("─" * 53)

        failed_runs = 0
        enabled_runs = 0
        for b in ALL_BACKENDS:
            if not self.enabled(b):
                continue
            enabled_runs += 1
            if self.fail_count.get(b, 0) > 0:
                failed_runs += 1

        # The jit-stats gate runs on every non-record run, so its tally is
        # reported on every non-record run — not only under --snapshot-diff,
        # where it used to live and where the bare CI invocation never looked.
        if self.args.snapshot_mode != "record":
            for paint, label, benches in (
                (red, "jit-stats regressed", self.jitstats_diffs),
                (yellow, "jit-stats improved (re-record)", self.jitstats_improvements),
                (yellow, "jit-stats unstable (not gated)", self.jitstats_unstable),
                (yellow, "jit-stats within band (not gated)", self.jitstats_banded),
                (red, "jit-stats baseline missing", self.jitstats_missing),
                (red, "jit-stats line absent", self.jitstats_absent),
                (red, "jit-stats census vacuous", self.jitstats_vacuous),
            ):
                if benches:
                    print(
                        f"{paint(label)}: {len(benches)} bench(es)"
                        f" — {' '.join(benches)}"
                    )

        if self.args.snapshot_mode or self.args.threshold is not None:
            if self.args.snapshot_mode == "record":
                print(dim(f"snapshot recorded under {SNAP_DIR}/"))
            elif self.args.snapshot_mode == "diff":
                if self.snapshot_diffs:
                    print(
                        f"{red('snapshot diff')}: {len(self.snapshot_diffs)} bench(es)"
                        f" — {' '.join(self.snapshot_diffs)}"
                    )
                if self.snapshot_missing:
                    print(
                        f"{dim('snapshot missing')}: {len(self.snapshot_missing)} bench(es)"
                        f" — {' '.join(self.snapshot_missing)}"
                    )
                if not self.snapshot_diffs and not self.snapshot_missing:
                    print(dim("snapshot diff: clean"))
            if self.args.threshold is not None:
                print(dim(f"threshold: ±{self.args.threshold}% vs baseline"))
            print()

        for b in ALL_BACKENDS:
            if not self.enabled(b):
                continue
            p = self.pass_count.get(b, 0)
            f = self.fail_count.get(b, 0)
            if f > 0:
                print(f"{red('FAILED')}: {b} {f} failed, {p} passed")
            else:
                print(f"{green('ALL PASSED')}: {b} {p}/{p}")

        if failed_runs > 0:
            print(f"{red('FAILED')}: {failed_runs} backend run(s) failed")
        else:
            print(f"{green('ALL PASSED')}: {enabled_runs}/{enabled_runs} backend run(s)")

        return 1 if failed_runs > 0 else 0


# ── Argument parsing ─────────────────────────────────────────────────

def parse_args():
    def positive_float(value):
        f = float(value)
        if f <= 0:
            raise argparse.ArgumentTypeError("must be greater than 0")
        return f

    def parse_backend_specs(specs):
        if specs is None:
            return list(DEFAULT_BACKENDS)

        backends = []
        for spec in specs:
            for backend in spec.split(","):
                backend = backend.strip()
                if not backend:
                    continue
                if backend not in CARGO_CONFIG:
                    choices = ", ".join(CARGO_CONFIG)
                    raise argparse.ArgumentTypeError(
                        f"invalid backend {backend!r}; choose from: {choices}"
                    )
                if backend not in backends:
                    backends.append(backend)

        if not backends:
            return list(DEFAULT_BACKENDS)
        return backends

    parser = argparse.ArgumentParser(
        description="pyre pre-merge check: correctness + regression guard + comparison",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--backend",
        action="append",
        nargs="?",
        const=",".join(DEFAULT_BACKENDS),
        default=None,
        metavar="BACKENDS",
        help="comma-separated backend list; may be repeated "
        f"(default: {','.join(DEFAULT_BACKENDS)})",
    )
    parser.add_argument(
        "--build",
        choices=("only", "yes", "no"),
        nargs="?",
        const="yes",
        default="yes",
        help="yes (the default) builds each backend and then measures it; no "
        "measures the release binaries already on disk without building; only "
        "builds and stamps them and exits without measuring, so that a caller "
        "can time the build apart from the run",
    )
    parser.add_argument(
        "--wasm-engine",
        choices=["wasmtime", "wasmi"],
        default="wasmtime",
        help="wasm runtime for the wasm backend: wasmtime (cranelift JIT, fast "
        "but recompiles the module each start) or wasmi (interpreter, near-zero "
        "startup, slower loops)",
    )
    parser.add_argument(
        "--no-startup-subtract",
        action="store_true",
        help="do not measure/subtract the baseline interpreter's empty-program "
        "user-CPU startup from ratios and perf gates (report raw times incl. "
        "startup)",
    )
    parser.add_argument("--timeout-scale", type=float, default=1.0)
    parser.add_argument("--dynasm-timeout-scale", type=float, default=None)
    parser.add_argument("--cranelift-timeout-scale", type=float, default=None)
    parser.add_argument("--wasm-timeout-scale", type=float, default=None)
    parser.add_argument("--snapshot", dest="snapshot_mode", action="store_const", const="record")
    parser.add_argument("--snapshot-diff", dest="snapshot_mode", action="store_const", const="diff")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--full",
        action="store_true",
        help="also run cpython on benchmarks without a vs_cpython gate (comparison only)",
    )
    parser.add_argument(
        "--no-synthetic",
        action="store_true",
        help="skip pyre/bench/synth feature-parity benchmarks",
    )
    parser.add_argument(
        "--cpython-suite",
        action="store_true",
        help="also run the vendored CPython suite gate (pyre/cpython_tests); "
             "off by default because it dominates this script's wall time",
    )
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="run only pyre/bench/synth feature-parity benchmarks",
    )
    parser.add_argument(
        "--synthetic-pattern",
        default="*.py",
        help="glob pattern under pyre/bench/synth for synthetic runs",
    )
    parser.add_argument(
        "--synthetic-timeout",
        type=positive_float,
        default=20.0,
        help="per-script timeout in seconds for synthetic benchmarks",
    )
    parser.add_argument(
        "--check-headers",
        action="store_true",
        help="read every synthetic fixture's `# pyre-check:` header, report the "
             "unusable ones and exit; builds and runs nothing",
    )
    parser.add_argument(
        "--no-oracle-cache",
        action="store_true",
        help="do not read cached CPython reference results (still write them)",
    )
    parser.add_argument(
        "--prune-oracle-cache",
        action="store_true",
        help="prune stale CPython oracle-cache entries and exit",
    )
    parser.add_argument("pyre_path", nargs="?", default="")
    args = parser.parse_args()
    # Answered here, ahead of the backend resolution below: the check is over
    # the fixture files, and nothing about it needs a backend to exist.
    if args.check_headers:
        sys.exit(check_synthetic_headers(args.synthetic_pattern))
    if args.prune_oracle_cache:
        sys.exit(oracle_cache.prune_cli())
    try:
        args.backends = parse_backend_specs(args.backend)
    except argparse.ArgumentTypeError as e:
        parser.error(str(e))

    if args.pyre_path and args.backend is None:
        parser.error("[path/to/pyre] requires --backend when running a single binary")

    if args.pyre_path and len(args.backends) != 1:
        parser.error("[path/to/pyre] can only be used with exactly one --backend")

    if args.synthetic_only and args.no_synthetic:
        parser.error("--synthetic-only cannot be combined with --no-synthetic")

    if args.snapshot_mode == "record" and args.threshold is not None:
        print("NOTE: --threshold ignored in --snapshot record mode")
        args.threshold = None

    return args


# ── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    global WASM_ENGINE
    WASM_ENGINE = args.wasm_engine
    chk = Check(args)

    backends = args.backends

    # Once per run, before any backend is measured: the check is over the tree,
    # not over a backend.
    #
    # The offsets in `build/llbc/` describe an artefact only if that artefact
    # was built from them, so the question is per leg and it is about the thing
    # the leg *executes*. For a native backend that is the binary, and a
    # positional one names a binary built somewhere else — the reason
    # `require_fresh_artefacts` skips its stamp on the same argument. For wasm
    # it is the module, not the binary: `pyre-wasm-runner` has no build.rs and
    # no pyre dependency, so the host carries no generated offset and an
    # external one changes nothing, while `PYRE_WASM_MODULE` can point the run
    # at a module built elsewhere. A module that does not exist answers
    # `same_file` False and is left to the existence check below, which says
    # far more about it than a stale-LLBC message would.
    def leg_reads_local_llbc(backend):
        if backend == "wasm":
            return same_file(effective_wasm_module(), WASM_MODULE_PATH)
        return not args.pyre_path

    if args.build == "no" and any(leg_reads_local_llbc(b) for b in backends):
        require_fresh_llbc()

    for backend in backends:
        if args.build != "no":
            chk.build_backend(backend)
        pyre_bin = args.pyre_path if args.pyre_path else default_binary(backend)
        if not Path(pyre_bin).exists():
            alt = pyre_bin + EXE
            if Path(alt).exists():
                pyre_bin = alt
        if not os.access(pyre_bin, os.X_OK) and not Path(pyre_bin).exists():
            if args.build == "no":
                print(
                    f"ERROR: --build=no requested for backend '{backend}', but "
                    f"the executable is missing: {pyre_bin}"
                )
            else:
                print(
                    f"ERROR: build failed for backend '{backend}' "
                    f"(missing executable: {pyre_bin})"
                )
            sys.exit(1)
        wasm_module = effective_wasm_module() if backend == "wasm" else None
        # Asked whether or not a build ran: `PYRE_WASM_MODULE` names the module
        # the runner loads, and a build cannot supply one that is not there.
        if backend == "wasm" and not Path(wasm_module).is_file():
            skipped = "--build=no requested" if args.build == "no" else "PYRE_WASM_MODULE set"
            print(
                f"ERROR: {skipped} for backend 'wasm', but the "
                f"wasm-host module is missing: {wasm_module}"
            )
            sys.exit(1)
        if args.build == "no":
            # `--pyre-path` names a binary from outside this tree, so its age
            # says nothing about these sources. The module is a separate
            # artefact and is this tree's own unless overridden, so it is asked
            # about either way; one from elsewhere carries no stamp and draws
            # the unchecked-freshness note rather than a refusal.
            artefacts = [] if args.pyre_path else [pyre_bin]
            if backend == "wasm":
                artefacts.append(wasm_module)
            if artefacts:
                require_fresh_artefacts(
                    artefacts,
                    f"--build=no requested for backend '{backend}'",
                    "Re-run with --build=yes to rebuild it.",
                )
        elif backend == "wasm" and not same_file(wasm_module, WASM_MODULE_PATH):
            # The build above produced `WASM_MODULE_PATH` and stamped it, but
            # the override names something else, so that is what the benchmarks
            # will load and what any baseline they record describes. Building a
            # module nothing runs is not a freshness check on the one that does.
            require_fresh_artefacts(
                [wasm_module],
                f"PYRE_WASM_MODULE names the module backend '{backend}' will load",
                "Unset PYRE_WASM_MODULE to run the module this build produced.",
            )
        chk._set_pyre(backend, pyre_bin)

    if args.build == "only":
        # Stop here, with every artefact built, existence-checked and stamped
        # by `build_backend`. A caller splits the run in two so that a build
        # and a measurement are timed apart -- which is otherwise impossible,
        # because one invocation reports one duration for both.
        #
        # This costs the second invocation one extra `build_inputs_fingerprint`
        # (under a second against a multi-minute build) and nothing else. The
        # build-window witness is opened and closed inside `build_backend`, so
        # it does not span the split, and the stamps the freshness gate reads
        # are files. Splitting in fact exercises that gate harder than a single
        # run does: the process that measures is no longer the process that
        # built, so every run reads a stamp across a process boundary instead
        # of trusting one it wrote itself.
        print()
        print(f"built and stamped: {', '.join(backends)}")
        sys.exit(0)

    print()
    print(bold("pyre pre-merge check"))
    chk.print_backend_config()
    print()
    # After the warmup, not before it. The startup pass runs on whatever state
    # the cargo build left behind -- cold page cache for each binary, and a
    # machine that has just had every core busy -- while the benches it is
    # subtracted from run one process at a time on paged-in binaries. The
    # warmup is the one thing between the two, so it belongs on the build's
    # side of the measurement.
    if not args.synthetic_only:
        chk.warmup(f"{BENCH_DIR}/int_loop.py")
    chk.measure_startups()
    if not args.synthetic_only:
        print()

        B = BENCH_DIR

        # fib_recursive / raise_catch / nbody / fannkuch are heavy on wasm
        # (interpreter round-trips + per-object Rust-heap allocation that the
        # wasm path does not yet collect, so they hit their timeout or the
        # linear-memory budget). They are intentionally NOT skipped on wasm:
        # we run them so the wasm leak/perf work can be driven down one bench
        # at a time, accepting the timeout state until each is fixed.

        # Gate values are set from the gate's OWN metric — user-CPU minus the
        # measured empty-program startup, min-of-5 interleaved — at ~2.5x the
        # measured ratio, so a runner 2.5x slower than an idle local box still
        # passes. A gate is only tightened when its baseline is healthy
        # (baseline exec >= ~5x that runtime's startup); where it is not, the
        # workload is sized up instead (see spectral_norm.py). Benches with
        # under ~3x headroom are deliberately left alone: nested_loop and
        # fib_recursive-vs-cpython are the known slow-runner flakes.
        # raise_catch's workload is sized so its execution time clears the
        # startup-health threshold instead of relying on extra gate headroom.
        # int_loop-cranelift recurrently measures 2.0-2.1x on the CI
        # runner (2x-tight), so its c_vs_py gate is raised 2->3.
        #             name              script                          timeout  d_vs_cp  d_vs_py  c_vs_cp  c_vs_py
        chk.run_bench("int_loop",       f"{B}/int_loop.py",             5,       None,    2,       None,    3)
        # float_loop's pypy ceilings follow the oracle to PyPy 8.0.0. Its x86
        # backend breaks cvtsi2sd's false dependency on the destination
        # register (`genop_cast_int_to_float`), which halves pypy's execution of
        # this loop on x86 and leaves aarch64 unchanged. pyre's x86 loop already
        # does the same, and its loop runs within ~15% of pypy's. What remains
        # is pyre's startup above pypy's, which stays in every pyre reading
        # (see fib_recursive below), against a pypy execution that is now
        # 0.13-0.20s. Measured on one runner with the three interpreters
        # interleaved: windows dynasm 2.0x, ubuntu dynasm 1.6x. Both ceilings
        # are fitted to the widest reading with a quarter's headroom.
        chk.run_bench("float_loop",     f"{B}/float_loop.py",           5,       None,    2.5,     None,    2.5)
        # fib_loop reads 0.8x on windows and 2.3x on ubuntu, and the ceiling is 3.
        chk.run_bench("fib_loop",       f"{B}/fib_loop.py",             5,       2,       3,       2,       3)
        # inline_helper's cranelift ceiling moves for the startup-subtraction
        # reason recorded on fib_recursive below, which reached that bench,
        # fib_loop and spectral_norm but not this row.  It is not a
        # measurement: run 33591256963 (before `6aabe927ce1`) and run
        # 33750103555 read ubuntu dynasm at the SAME pypy 0.22s and the SAME
        # pyre 0.32s, and the reported ratio still moved 1.1x -> 1.5x.  Only
        # the arithmetic changed.
        #
        # The leftover pyre startup in the numerator is a fixed surcharge; the
        # loop is 150e6 so that term is a small fraction of a 1.5x budget.
        # Cranelift stays at 1.9: it is the leg that failed at 1.5 before the
        # lengthening.
        chk.run_bench("inline_helper",  f"{B}/inline_helper.py",        5,       None,    1.5,     None,    1.9)
        # fib_recursive cranelift-vs-pypy reads 0.8x (PR ubuntu, run
        # 37225763286) .. 2.5x (main ubuntu, run 37213275404); 2.9 is 15%
        # over that and under 3, so it derives no floor.  Dynasm stays at
        # 2.7.  The cpython gates stay at 2.
        chk.run_bench("fib_recursive",  f"{B}/fib_recursive.py",        5,       2,       2.7,     2,       2.9)
        chk.run_bench("nested_loop",    f"{B}/nested_loop.py",          5,       None,    2,       None,    3)
        # Windows dynasm run 34593191789 measured 1.6x against a 1.5
        # ceiling; 1.7 covers that reading with the same ~15% headroom
        # used on spectral_norm below. Ubuntu cranelift stayed inside 2.5.
        chk.run_bench("raise_catch",    f"{B}/raise_catch_loop.py",     5,       None,    1.7,     None,    2.5)
        # Run 33363045302 measured spectral_norm at 0.4-1.4x on the healthy
        # pypy baselines; windows' clamped baseline displayed an indicative
        # 2.0x.  A 2.3 ceiling covers that reading with 15% headroom.
        #
        # Neither carries a cpython gate, and an absent gate is what stops
        # cpython being spawned for the row at all. On the ubuntu runner
        # cpython takes 11.38s against the slowest backend's 0.45s here and
        # 4.42s against 0.85s on nbody, so the gates they used to carry (1x,
        # and 0.5x/1x) sat 25x and 3x inside their margin while the pypy gate
        # beside them reads 0.4x-1.4x: pypy trips on the smaller
        # regression of the two in both rows. Those two spawns were three
        # quarters of every cpython second this file spends outside the
        # synthetic suite.
        # Both ceilings then move for the startup-subtraction reason recorded on
        # fib_recursive above.  spectral_norm carries it worse than fib does
        # because its pypy execution is barely over a tenth of a second, so a
        # 0.05s-0.11s startup deficit is most of a whole multiple: the readings
        # become 1.60x-2.83x, widest on windows dynasm, and a 3.5 ceiling covers
        # that with a quarter's headroom.
        chk.run_bench("spectral_norm",  f"{B}/spectral_norm.py",       15,       None,    3.5,     None,    3.5)
        chk.run_bench("nbody",          f"{B}/nbody.py",               10,       None,    3,       None,    4,    wasm_float_tol=True)
        # fannkuch is almost nothing but cross-loop JUMP, which the two
        # backends pay differently: cranelift's closing_jump carries a LABEL's
        # values through jitframe slots (`emit_attached_loop_dispatch`) where
        # dynasm remaps them in registers.  That is why its ceiling here is the
        # wider of the pair, and why this bench alone needs the spread.
        # Cranelift reads 3.2-5.0x; 7 covers that reading. Dynasm's 5 keeps
        # the reading in 1.25x..5x.
        chk.run_bench("fannkuch",       f"{B}/fannkuch.py",            30,       1,       5,       2,       7)
        # The branchy-inlined-callee guard (gh#343) lives in the synthetic parity
        # suite as bridge_branchy_callee.py, gated against pypy by
        # `# pyre-check: max-pypy-ratio`; a decline that keeps every crossing
        # interpreted (~34x) balloons the ratio past the limit.

    if not args.no_synthetic:
        print()
        chk.run_synthetic_suite()

    if args.cpython_suite and not args.synthetic_only:
        print()
        print(bold("vendored CPython suite"))
        chk.run_cpython_suite()

    rc = chk.print_summary()
    oracle_cache.maybe_prune(chk.oracle_store)
    sys.exit(rc)


if __name__ == "__main__":
    main()
