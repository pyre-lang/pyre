#!/usr/bin/env python3
"""Parity baseline runner.

Runs every script under `pyre/extra_tests/parity_tests/` against:
  - CPython 3.15 (`PYRE_CHECK_PYTHON3`, else `python3.15` on PATH),
  - the pyre-dynasm binary (release build),
  - the pyre-cranelift binary (release build, if present).

A script passes when:
  - the process exits with code 0,
  - the last non-empty stdout line equals "OK".

Any divergence between CPython and a pyre backend is a parity
regression: the runtime has drifted from CPython observable
semantics.

A script whose subject is platform-specific names the platforms it
applies to in its header (`# pyre-check: platforms=linux,darwin`) and is
skipped elsewhere.

PyPy runs too when one is on PATH, as a cross-check rather than a
reference: it is a second implementation of the same language, so it
catches a script that pins something pyre and CPython agree on only by
accident. PyPy 7.3.x implements CPython 3.11, so a script pinning
behaviour introduced later — or pinning something PyPy does its own way —
cannot hold there and says so in its header:

    # pyre-check: pypy-diverges: <what this pins, and what pypy3 reports>

A script carrying that must fail under PyPy and a script without it must
pass, so the directive cannot quietly outlive the divergence it names.

A script whose defect is only reachable under a particular runtime
configuration declares it with one or more

    # parity-env: NAME=VALUE

lines, which are added to the environment of every runner for that script
only. Without this a script can keep passing after the shape it exercises
stops being reachable, i.e. cover nothing while still looking green.

A fix that ships ON, with a switch that restores the defect, has an arm
nothing runs: the corpus covers the fix, and the switch is exercised once by
whoever measured it. A script names that arm with

    # pyre-check: regresses-under: NAME=VALUE [NAME=VALUE ...] -- <reason>

and is then run a second time on each pyre backend with those names set,
where it must FAIL. Passing there is reported as XPASS, the same way a
`pypy-diverges` script that starts passing is: either the switch stopped
changing the answer, or the script stopped reaching the shape, and both
turn the default-ON run green for a reason its header no longer states.

Usage:
    python3 pyre/extra_tests/parity_tests/run.py
        [--dynasm-only|--cranelift-only] [--gc-poison]

`PYRE_CHECK_PYPY3` names the PyPy binary; setting it empty turns the
cross-check off.

`--gc-poison` fills reclaimed nursery bytes with a poison pattern for the
pyre backends. A dangling reference into reclaimed nursery memory usually
still decodes as a plausible object, so without poison that whole class of
GC defect passes silently; with it the run aborts at the first stale read.

Exit code is 0 iff every (script, backend) pair passed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
TARGET_RELEASE = ROOT / "target" / "release"

EXE = ".exe" if sys.platform == "win32" else ""

PLATFORMS_PREFIX = "# pyre-check: platforms="
CPYTHON_GAP_PREFIX = "# CPython-suite gap:"
PARITY_REASON_PREFIX = "# parity-tests reason:"

# The version whose observable behaviour these scripts pin, which is the one
# pyre's native modules (`_sre.MAGIC`) and the vendored `lib-python/3` are
# coupled to. An older CPython disagrees with it on error text, dunder surfaces
# and structure sizes — dozens of failures that are not parity failures, and
# that bury the ones that are. Comparing a backend against the wrong reference
# measures nothing, so a run that cannot find the right one stops rather than
# reporting what it found.
CPYTHON_TARGET = (3, 15)

PYPY_DIVERGES_PREFIX = "# pyre-check: pypy-diverges:"

# The CPython version PyPy implements, which is what makes it a cross-check and
# not a second reference: it is three releases behind the one these scripts
# pin. A PyPy on some other target has not been classified against this corpus,
# so the lane skips rather than reporting what it found.
PYPY_TARGET = (3, 11)


def _runs_here(path: Path) -> bool:
    """Whether this platform is one the script's expectations hold on.

    A script may name the `sys.platform` values it applies to in its header,
    above the docstring, with the reason beside it:

        # pyre-check: platforms=linux,darwin

    Elsewhere it is skipped, because what it pins is platform-specific and the
    reference CPython fails it too — comparing a backend against a failing
    reference measures nothing. A script with no marker runs everywhere.
    """
    with path.open(encoding="utf-8") as source:
        for _ in range(20):
            line = source.readline()
            if not line:
                break
            if not line.startswith(PLATFORMS_PREFIX):
                continue
            named = line[len(PLATFORMS_PREFIX):].strip().split(",")
            return sys.platform in [name.strip() for name in named]
    return True


def _scripts() -> tuple[list[Path], list[Path]]:
    """The scripts to run on this platform, and the ones skipped."""
    out = []
    skipped = []
    for p in sorted(HERE.glob("*.py")):
        if p.name == "run.py":
            continue
        header = p.read_text(encoding="utf-8").splitlines()[:20]
        missing = [
            prefix
            for prefix in (CPYTHON_GAP_PREFIX, PARITY_REASON_PREFIX)
            if not any(line.startswith(prefix) for line in header)
        ]
        if missing:
            names = ", ".join(missing)
            raise RuntimeError(f"{p.name}: missing parity header field(s): {names}")
        for line in header:
            if line.startswith(PYPY_DIVERGES_PREFIX):
                if not line[len(PYPY_DIVERGES_PREFIX):].strip():
                    raise RuntimeError(
                        f"{p.name}: `{PYPY_DIVERGES_PREFIX}` with no reason — the "
                        f"directive exists to say which divergence is expected"
                    )
        # Read here for the raise alone. The arm below reads it again once it
        # knows which runners are pyre's, and an unparseable directive there
        # would arrive as a traceback out of the middle of the table.
        _regresses_under(p)
        (out if _runs_here(p) else skipped).append(p)
    return out, skipped


def _pypy_divergence(script: Path) -> str | None:
    """The divergence from PyPy this script declares, if it declares one."""
    header = script.read_text(encoding="utf-8").splitlines()[:20]
    for line in header:
        if line.startswith(PYPY_DIVERGES_PREFIX):
            return line[len(PYPY_DIVERGES_PREFIX):].strip()
    return None


_ENV_DIRECTIVE = re.compile(r"^#\s*parity-env:\s*(\w+)=(\S*)\s*$", re.MULTILINE)


def _script_env(script: Path) -> dict[str, str]:
    """Environment a script pins for itself with `# parity-env: NAME=VALUE`."""
    text = script.read_text(encoding="utf-8", errors="replace")
    return {m.group(1): m.group(2) for m in _ENV_DIRECTIVE.finditer(text)}


REGRESSES_UNDER_PREFIX = "# pyre-check: regresses-under:"

# `# pyre-check: regresses-under: NAME=VALUE [NAME=VALUE ...] -- <reason>`.
# The reason is part of the syntax rather than a convention: this directive
# asserts a failure, and a failure nobody wrote down reads as the script being
# broken by whoever next runs it.
_REGRESSES_UNDER = re.compile(
    r"^#\s*pyre-check:\s*regresses-under:\s*"
    r"((?:\w+=\S*\s+)*\w+=\S*)"
    r"\s+--\s+(\S.*?)\s*$"
)


def _regresses_under(script: Path) -> list[tuple[dict[str, str], str]]:
    """The configurations this script declares it must fail under, with reasons.

    Raises on a directive that does not parse, so a typo reads as an authoring
    error rather than as a script that quietly declares nothing.
    """
    out = []
    for line in script.read_text(encoding="utf-8").splitlines()[:20]:
        if not line.startswith(REGRESSES_UNDER_PREFIX):
            continue
        parsed = _REGRESSES_UNDER.match(line)
        if parsed is None:
            raise RuntimeError(
                f"{script.name}: `{REGRESSES_UNDER_PREFIX}` needs "
                f"`NAME=VALUE [NAME=VALUE ...] -- <reason>`, got: {line.strip()}"
            )
        settings = dict(pair.split("=", 1) for pair in parsed.group(1).split())
        out.append((settings, parsed.group(2)))
    return out


TIMEOUT = 30


class Failure(NamedTuple):
    """One (script, runner) pair that did not pass, and the evidence."""

    script: str
    backend: str
    # What the runner concluded, in one line — fits beside the table row.
    reason: str
    # The child's stderr, verbatim. A traceback is the answer to "why", and it
    # only reads as one when its own line breaks survive.
    stderr: str


def _run(
    cmd: list[str], script: Path, env: dict[str, str] | None
) -> tuple[bool, str, str]:
    """Whether the script passed, why it did not, and the child's stderr.

    The reason is kept apart from the stderr rather than formatted into it: a
    `repr` of a whole traceback is one unreadable line, and the run wants the
    short verdict beside the row and the traceback in the report at the end.
    """
    try:
        proc = subprocess.run(
            cmd + [str(script)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TIMEOUT,
            env=None if env is None else {**os.environ, **env},
        )
    except subprocess.TimeoutExpired as expired:
        # `text=True` decodes what `communicate()` RETURNS; the timeout path
        # raises with the raw chunks it had joined so far, so what arrives here
        # is bytes on POSIX and str on Windows. Reporting the bytes would print
        # one `b'...\n...'` line, which is the unreadable shape this report
        # exists to avoid.
        partial = expired.stderr or ""
        if isinstance(partial, bytes):
            partial = partial.decode("utf-8", "replace")
        return False, f"timed out after {TIMEOUT}s", partial
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    last = lines[-1] if lines else ""
    if proc.returncode == 0 and last == "OK":
        return True, "", ""
    if proc.returncode == 0:
        # The script ran to completion and never announced itself. Reporting
        # this as `rc=0 last=''` reads like an interpreter that produced
        # nothing, and it is reported once per runner, so three of them make a
        # sound fixture that forgot its last line look like a pyre failure.
        reason = f"exited 0 without a final 'OK' line (last non-empty line {last!r})"
    else:
        reason = f"exited {proc.returncode} (last non-empty stdout line {last!r})"
    return False, reason, proc.stderr


PROBE = "import sys; print(sys.version_info[0], sys.version_info[1]); print(sys.executable)"


def _probe(command: str) -> tuple[tuple[int, int], str] | None:
    """What the interpreter reports as its version and its own path.

    `None` when the command did not run at all, which on Windows is what a
    `python3.15` that names an extensionless shim rather than an executable
    does — `shutil.which` finds it and `CreateProcess` cannot start it.
    """
    try:
        proc = subprocess.run(
            [command, "-c", PROBE],
            capture_output=True,
            text=True,
            timeout=30,
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


def _cpython() -> str:
    """The reference interpreter, which has to be the version the scripts pin.

    `PYRE_CHECK_PYTHON3` names it outright; otherwise the version-qualified
    name is tried before the bare ones, because a `python3` on PATH is
    whichever CPython the system happens to ship. What comes back is the path
    the interpreter reports for itself, so the scripts are spawned by a name
    that does not depend on the PATH they inherit.
    """
    named = os.environ.get("PYRE_CHECK_PYTHON3")
    candidates = [named] if named else ["python3.15", "python3", "python"]
    rejected = []
    for candidate in candidates:
        if named is None and shutil.which(candidate) is None:
            continue
        probed = _probe(candidate)
        if probed is None:
            rejected.append(f"  {candidate}: did not run")
            continue
        version, executable = probed
        if version == CPYTHON_TARGET:
            return executable
        rejected.append("  %s: %d.%d" % (candidate, *version))
    wanted = "%d.%d" % CPYTHON_TARGET
    raise SystemExit(
        f"no CPython {wanted} to compare against — the parity scripts pin its "
        f"behaviour, and an older one fails them for reasons that are not "
        f"parity failures.\n"
        + ("\n".join(rejected) or "  (no candidate on PATH)")
        + "\nName one with PYRE_CHECK_PYTHON3."
    )


PYPY_PROBE = (
    "import sys; print(sys.implementation.name); "
    "print(sys.version_info[0], sys.version_info[1]); print(sys.executable)"
)


def _pypy() -> tuple[str | None, str]:
    """The PyPy to cross-check against, and why there is none when there isn't.

    Unlike [`_cpython`] this never stops the run: PyPy is a cross-check, and a
    host without one still measures everything the references can decide.
    Which is also why its absence is announced — a lane that silently does not
    run reads exactly like a lane that ran and agreed.
    """
    named = os.environ.get("PYRE_CHECK_PYPY3")
    if named is not None and not named.strip():
        return None, "PYRE_CHECK_PYPY3 is set empty"
    candidate = named or "pypy3"
    if named is None and shutil.which(candidate) is None:
        return None, "no pypy3 on PATH"
    try:
        proc = subprocess.run(
            [candidate, "-c", PYPY_PROBE], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None, f"{candidate} did not run"
    lines = (proc.stdout or "").splitlines()
    if proc.returncode != 0 or len(lines) < 3:
        return None, f"{candidate} did not report its version"
    if lines[0].strip() != "pypy":
        return None, f"{candidate} is {lines[0].strip()}, not pypy"
    try:
        version = tuple(int(part) for part in lines[1].split())
    except ValueError:
        return None, f"{candidate} did not report its version"
    if version != PYPY_TARGET:
        wanted = "%d.%d" % PYPY_TARGET
        return None, (
            "%s implements CPython %d.%d, and the `pypy-diverges:` directives "
            "were classified against %s" % (candidate, *version, wanted)
        )
    return lines[2].strip() or candidate, ""


def _report(failures: list[Failure]) -> None:
    """The evidence for every failure, printed last and on stdout.

    Last because a CI log is read from its end, and the table above it is two
    hundred rows: a reader who has to scroll up for the reason reruns the job
    instead. On stdout because the table is — these lines used to go to stderr,
    which is unbuffered where a piped stdout is not, so the whole report
    arrived in the log *above* the run's own header and the last thing before
    the runner's non-zero exit was a passing row.
    """
    print("=" * 72)
    print(f"{len(failures)} failure(s)")
    for failure in failures:
        print()
        print(f"  {failure.script} [{failure.backend}]: {failure.reason}")
        for line in failure.stderr.strip().splitlines():
            print(f"      {line}")
    print("=" * 72)


def _annotate(failures: list[Failure]) -> None:
    """One GitHub Actions error annotation per failure.

    An annotation shows on the pull request and the job summary, so the
    reason survives even for a reader who never opens the log. The message is
    a single line by the format's own rule, so it carries the verdict and the
    exception line rather than the whole traceback.
    """
    for failure in failures:
        spoken = [line.strip() for line in failure.stderr.splitlines() if line.strip()]
        tail = spoken[-1] if spoken else ""
        message = f"{failure.backend}: {failure.reason}"
        if tail:
            message += f" | {tail}"
        escaped = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        path = (HERE / failure.script).relative_to(ROOT).as_posix()
        print(f"::error file={path},title=parity::{escaped}")


class Runner(NamedTuple):
    """One interpreter the corpus is run against."""

    name: str
    cmd: list[str]
    env: dict[str, str] | None
    # Whether this runner reads pyre's own configuration. `regresses-under`
    # only says anything on one that does: every other runtime ignores the
    # name, passes, and would be reported as the control having gone dead.
    pyre: bool


def _runners(
    only_dynasm: bool, only_cranelift: bool, gc_poison: bool
) -> list[Runner]:
    runners: list[Runner] = []
    runners.append(Runner("cpython", [_cpython()], None, False))
    pyre_env = {"MAJIT_GC_NURSERY_POISON": "1"} if gc_poison else None
    dynasm = TARGET_RELEASE / f"pyre-dynasm{EXE}"
    cranelift = TARGET_RELEASE / f"pyre-cranelift{EXE}"
    if not only_cranelift and dynasm.exists():
        runners.append(Runner("dynasm", [str(dynasm)], pyre_env, True))
    if not only_dynasm and cranelift.exists():
        runners.append(Runner("cranelift", [str(cranelift)], pyre_env, True))
    return runners


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynasm-only", action="store_true")
    parser.add_argument("--cranelift-only", action="store_true")
    parser.add_argument("--gc-poison", action="store_true")
    args = parser.parse_args()

    # A piped stdout is block-buffered, so without line buffering the rows
    # arrive in the log in one burst at the end and a run that dies mid-way
    # names none of the scripts it got through. The encoding is pinned because
    # the report echoes a child's stderr, and several of these scripts are
    # about names no console codepage can spell — printing one of those on a
    # non-UTF-8 console used to kill the runner with a `UnicodeEncodeError`
    # instead of reporting the failure it was in the middle of explaining.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

    runners = _runners(args.dynasm_only, args.cranelift_only, args.gc_poison)
    pypy, pypy_off = _pypy()
    scripts, skipped = _scripts()
    if not scripts:
        print("no parity test scripts found", file=sys.stderr)
        return 1

    print(f"runners: {[runner.name for runner in runners]}")
    if pypy is None:
        print(f"pypy cross-check: off — {pypy_off}")
    else:
        print(f"pypy cross-check: {pypy}")
    print(f"scripts: {len(scripts)}")
    for script in skipped:
        print(f"skipped ({sys.platform} not in its platforms): {script.name}")
    print()

    failures: list[Failure] = []
    for script in scripts:
        name = script.name
        pinned = _script_env(script)
        row: list[str] = [f"  {name:<36s}"]
        reasons: list[str] = []
        for runner in runners:
            merged = {**(runner.env or {}), **pinned}
            ok, reason, err = _run(runner.cmd, script, merged or None)
            row.append(f"{runner.name}={'OK' if ok else 'FAIL'}")
            if not ok:
                failures.append(Failure(name, runner.name, reason, err))
                reasons.append(f"      {runner.name}: {reason}")
        for settings, why in _regresses_under(script):
            spelled = " ".join(f"{key}={value}" for key, value in settings.items())
            for runner in runners:
                if not runner.pyre:
                    continue
                ok = _run(
                    runner.cmd,
                    script,
                    {**(runner.env or {}), **pinned, **settings},
                )[0]
                if not ok:
                    row.append(f"{runner.name}:regresses=xfail")
                    continue
                # The default-ON run above is green either way, so this is the
                # only place the switch it depends on is measured at all.
                row.append(f"{runner.name}:regresses=XPASS")
                stale = (
                    f"passed with {spelled} set, which its header says it "
                    f"regresses under: {why!r} — either the switch stopped "
                    f"changing the answer or the script stopped reaching the "
                    f"shape, and the passing run above no longer covers it"
                )
                failures.append(Failure(name, f"{runner.name} {spelled}", stale, ""))
                reasons.append(f"      {runner.name} [{spelled}]: {stale}")
        if pypy is not None:
            declared = _pypy_divergence(script)
            ok, reason, err = _run([pypy], script, pinned or None)
            if declared is None:
                row.append(f"pypy={'OK' if ok else 'FAIL'}")
                if not ok:
                    failures.append(Failure(name, "pypy", reason, err))
                    reasons.append(f"      pypy: {reason}")
            elif ok:
                # The directive named a divergence that is no longer there, so
                # what it now does is exempt this script from the lane for a
                # reason that has stopped being true.
                row.append("pypy=XPASS")
                stale = (
                    f"passed, but its header declares a divergence: {declared!r} "
                    f"— drop the `{PYPY_DIVERGES_PREFIX}` line"
                )
                failures.append(Failure(name, "pypy", stale, ""))
                reasons.append(f"      pypy: {stale}")
            else:
                row.append("pypy=xfail")
        print(" ".join(row))
        for line in reasons:
            print(line)

    print()
    if not failures:
        print("all parity tests pass")
        return 0
    _report(failures)
    if os.environ.get("GITHUB_ACTIONS") == "true":
        _annotate(failures)
    return 1


if __name__ == "__main__":
    sys.exit(main())
