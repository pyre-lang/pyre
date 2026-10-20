#!/usr/bin/env python3
"""Runner for the cpyext end-to-end tests.

Each `test_*.py` under `pyre/extra_tests/cpyext/` is the script that used to
live inside a `pyre/pyrex/tests/cpyext_*.rs` integration binary.  The Rust
side only compiled a C fixture and compared stdout, so this runner does that
itself and the scripts stay Python.

A script's leading comment block carries directives:

    # cpyext-fixture: <name> [<suffix>]   (repeatable; compiles fixtures/<name>.c)
    # cpyext-env: NAME=<file>             (NAME=<tmpdir>/<file>)
    # cpyext-expect: <marker>             (stripped stdout must equal this)
    # cpyext-expect-exit: <code>          (default 0)
    # cpyext-expect-stderr-contains: <text>

The fixture is compiled the way the Rust `Fixtures::compile_with_suffix` did:
`$CC` or `cc`, `-I include/pyre3.15t`, `-o`, `-Werror`, then
`-bundle -undefined dynamic_lookup` on macOS or `-fPIC -shared` elsewhere.
The native suffix matches `_imp.extension_suffixes()` (darwin; x86_64 or
aarch64 linux-gnu; plain linux-gnu).  A second word on `cpyext-fixture`
overrides it, for example `# cpyext-fixture: foo .abi3.so`.

Each script gets a fresh temporary directory on `PYTHONPATH` and is run as
`<pyre> -S <script>`.

Usage:
    python3 pyre/extra_tests/cpyext/run.py [--pyre PATH]
                                           [--filter SUBSTRING]
                                           [--list]
                                           [--jobs N]
                                           [--timeout SECONDS]

Exit code is 0 iff every script passed.  On Windows, cpyext fixtures are
unsupported and the runner exits 0.
"""

from __future__ import annotations

import argparse
import os
import platform
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FIXTURES = ROOT / "pyre" / "pyrex" / "tests" / "fixtures"
INCLUDE = ROOT / "include" / "pyre3.15t"
DEFAULT_PYRE = ROOT / "target" / "release" / "pyre-dynasm"

_DIRECTIVE = re.compile(r"# cpyext-([a-z0-9-]+):\s*(.*)$")


def extension_suffix() -> str:
    """The suffix `_imp.extension_suffixes()` reports for this host."""
    if sys.platform == "darwin":
        return ".pyre315-darwin.so"
    machine = platform.machine()
    if machine == "x86_64":
        return ".pyre315-x86_64-linux-gnu.so"
    if machine == "aarch64":
        return ".pyre315-aarch64-linux-gnu.so"
    return ".pyre315-linux-gnu.so"


def _scripts(filter_substring: str | None) -> list[Path]:
    out = []
    for path in sorted(HERE.glob("test_*.py")):
        if filter_substring and filter_substring not in path.name:
            continue
        out.append(path)
    return out


def _parse_header(text: str) -> dict:
    """Directives from the leading comment block.  Later comments are script text."""
    fixtures: list[tuple[str, str | None]] = []
    env: list[tuple[str, str]] = []
    expect: str | None = None
    expect_exit: int | None = None
    stderr_contains: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "" or stripped.startswith("#!"):
            continue
        if not stripped.startswith("#"):
            break
        match = _DIRECTIVE.match(stripped)
        if match is None:
            continue
        key, rest = match.group(1), match.group(2).strip()
        if key == "fixture":
            parts = rest.split()
            if len(parts) == 1:
                fixtures.append((parts[0], None))
            elif len(parts) == 2:
                fixtures.append((parts[0], parts[1]))
            else:
                raise SystemExit(f"bad cpyext-fixture directive: {stripped}")
        elif key == "env":
            name, sep, filename = rest.partition("=")
            if not sep or not name or not filename:
                raise SystemExit(f"bad cpyext-env directive: {stripped}")
            env.append((name, filename))
        elif key == "expect":
            expect = rest
        elif key == "expect-exit":
            expect_exit = int(rest)
        elif key == "expect-stderr-contains":
            stderr_contains = rest
        else:
            raise SystemExit(f"unknown cpyext directive: {stripped}")
    return {
        "fixtures": fixtures,
        "env": env,
        "expect": expect,
        "expect_exit": expect_exit,
        "stderr_contains": stderr_contains,
    }


def _compile(name: str, suffix: str, directory: Path) -> str:
    source = FIXTURES / f"{name}.c"
    extension = directory / f"{name}{suffix}"
    cc = os.environ["CC"] if "CC" in os.environ else "cc"
    command = [
        cc,
        str(source),
        "-I",
        str(INCLUDE),
        "-o",
        str(extension),
        "-Werror",
    ]
    if sys.platform == "darwin":
        command.extend(["-bundle", "-undefined", "dynamic_lookup"])
    else:
        command.extend(["-fPIC", "-shared"])
    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        return proc.stderr
    return ""


def _run_one(script: Path, pyre: Path, timeout: int) -> tuple[str, bool, str]:
    header = _parse_header(script.read_text(encoding="utf-8"))
    try:
        tmp = tempfile.TemporaryDirectory(prefix="pyre-cpyext-")
    except OSError as exc:
        return script.name, False, f"temp dir: {exc}"
    with tmp:
        directory = Path(tmp.name)
        for name, suffix in header["fixtures"]:
            err = _compile(name, suffix or extension_suffix(), directory)
            if err:
                return script.name, False, f"C compiler failed:\n{err}"
        child_env = os.environ.copy()
        child_env["PYTHONPATH"] = str(directory)
        for name, filename in header["env"]:
            child_env[name] = str(directory / filename)
        try:
            proc = subprocess.run(
                [str(pyre), "-S", str(script)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                env=child_env,
            )
        except subprocess.TimeoutExpired as expired:
            stdout = expired.stdout or ""
            stderr = expired.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", "replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", "replace")
            return (
                script.name,
                False,
                f"timeout after {timeout}s\nstdout:\n{stdout}\nstderr:\n{stderr}",
            )
        expected_exit = 0 if header["expect_exit"] is None else header["expect_exit"]
        problems = []
        if proc.returncode != expected_exit:
            problems.append(f"exit {proc.returncode}, expected {expected_exit}")
        if header["expect"] is not None and proc.stdout.strip() != header["expect"]:
            problems.append(
                f"stdout {proc.stdout.strip()!r} != {header['expect']!r}"
            )
        if (
            header["stderr_contains"] is not None
            and header["stderr_contains"] not in proc.stderr
        ):
            problems.append(
                f"stderr does not contain {header['stderr_contains']!r}"
            )
        if problems:
            detail = "\n".join(problems)
            return (
                script.name,
                False,
                f"{detail}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}",
            )
        return script.name, True, ""


def main() -> int:
    if os.name == "nt":
        print("cpyext fixtures are unsupported")
        return 0
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pyre",
        default=str(DEFAULT_PYRE),
        help="pyre binary (default: target/release/pyre-dynasm)",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="run only scripts whose name contains this substring",
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    scripts = _scripts(args.filter)
    if args.list:
        for script in scripts:
            print(script)
        return 0
    pyre = Path(args.pyre)
    if not pyre.is_file():
        print(f"FAIL pyre binary not found: {pyre}", file=sys.stderr)
        return 1
    jobs = max(1, args.jobs)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(
            pool.map(lambda script: _run_one(script, pyre, args.timeout), scripts)
        )
    failed = 0
    for name, ok, detail in results:
        if ok:
            print(f"PASS {name}")
        else:
            failed += 1
            print(f"FAIL {name}")
            if detail:
                print(detail)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
