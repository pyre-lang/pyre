#!/usr/bin/env python3
"""End-to-end gate for the interpreter's own pip, resolved entirely offline.

Drives a release pyre binary through the sequence a user performs when they
install something:

    -m venv  ->  ensurepip  ->  pip install (wheel)
             ->  pip install (PEP 517 build, isolated)
             ->  console script  ->  metadata  ->  uninstall

Every artefact comes from the checkout: the pip wheel `ensurepip` bundles, and
the setuptools wheel under `lib-python/3/test/wheeldata` that the suite's own
venv helpers glob for.  Nothing is resolved from an index, so the gate cannot
fail because a package server is slow, and it cannot pass because a stale
wheel was lying in a cache.  One check spends a second proving exactly that,
before the first check that resolves anything: a plain `pip download` of a
name only an index can answer has to fail.

This lives beside `snippets/` and `parity_tests/` rather than in them because
it is one long stateful sequence -- the build in a later check consumes the
archive an earlier one wrote -- and because it needs a per-check timeout an
order of magnitude above what a snippet gets.

The reference CPython is not in the gating set: pip succeeding there says
nothing about pyre.  It is used only as a control, and only after something
has already failed, to separate "pyre broke" from "the fixture rotted".

Usage:
    python3 pyre/extra_tests/pip/run.py [--dynasm-only|--cranelift-only]
                                        [--with-network] [--keep]
                                        [--no-cpython-control]
                                        [--timeout SECONDS]

Exit code is 0 iff every (backend, check) pair passed.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Callable, NamedTuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
TARGET_RELEASE = ROOT / "target" / "release"
FIXTURES = HERE / "fixtures"
MKSDIST = HERE / "mksdist.py"

# The two wheel directories the checkout already carries. `_bundled` is what
# `ensurepip` installs into a fresh venv; `wheeldata` is what the suite's own
# `setup_venv_with_pip_setuptools` helper globs, and it is the whole reason a
# build with real isolation can be gated without an index.
BUNDLED = ROOT / "lib-python" / "3" / "ensurepip" / "_bundled"
WHEELDATA = ROOT / "lib-python" / "3" / "test" / "wheeldata"

EXE = ".exe" if sys.platform == "win32" else ""
SCRIPTS = "Scripts" if sys.platform == "win32" else "bin"

# Per check, not per run. The slowest check builds a wheel in an isolated
# environment it has to populate first, which measures in single-digit seconds
# on an unloaded machine; the margin is for a shared runner, and the timeout
# report names the check and echoes what the child had written so far, so a
# hit reads as a diagnosis rather than as one word.
TIMEOUT = 300

# The reference the control leg has to be, for the same reason the parity
# runner pins it: an older interpreter disagrees about things that are not
# what this gate measures.
CPYTHON_TARGET = (3, 15)

SDIST = "stpkg-0.2.0.tar.gz"
SDIST_MEMBERS = {
    "stpkg-0.2.0/pyproject.toml",
    "stpkg-0.2.0/stpkg.py",
    "stpkg-0.2.0/PKG-INFO",
}


class Failed(Exception):
    """A check that did not hold, and the evidence for saying so."""

    def __init__(self, reason: str, evidence: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.evidence = evidence


class Failure(NamedTuple):
    """One (check, backend) pair that did not pass."""

    check: str
    backend: str
    reason: str
    evidence: str


class Result(NamedTuple):
    """What a spawned command did."""

    argv: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        return self.stdout + self.stderr

    def describe(self) -> str:
        spelled = " ".join(self.argv)
        return f"$ {spelled}\n{self.output}"


def _sole_wheel(directory: Path, stem: str) -> tuple[str, str]:
    """The one `stem-*.whl` in `directory`, as its name and its version.

    Read rather than written down, so a stdlib sync that bumps either bundled
    wheel needs no edit here: a version literal in this file would turn that
    sync red for a reason that has nothing to do with the runtime.
    """
    found = sorted(directory.glob(f"{stem}-*.whl"))
    if len(found) != 1:
        names = ", ".join(path.name for path in found) or "<none>"
        raise SystemExit(f"expected exactly one {stem} wheel in {directory}, found: {names}")
    name, version, *_ = found[0].name.split("-")
    return name, version


class Context:
    """One backend's run: where it works, and what it has established so far."""

    def __init__(
        self, backend: str, interpreter: str, root: Path, network: bool, timeout: int
    ) -> None:
        self.backend = backend
        self.interpreter = interpreter
        self.root = root
        self.network = network
        self.timeout = timeout
        self.tmp = root / "tmp"
        self.src = root / "src"
        self.venv = root / "venv"
        self.dist = root / "dist"
        # What the venv's own pip answers for its version, set by the import
        # check and compared against everywhere else.
        self.pipver = ""

    @property
    def python(self) -> Path:
        return self.venv / SCRIPTS / f"python{EXE}"

    def script(self, name: str) -> Path:
        return self.venv / SCRIPTS / f"{name}{EXE}"

    def env(self) -> dict[str, str]:
        """The child environment, pinned so no check inherits a wheel source.

        Every `PIP_*` the caller had is dropped, not just the ones with an
        obvious reach: pip takes a long option's value from the matching
        `PIP_*` name, and `--find-links` is one of them. `--no-index` closes
        the index and leaves link directories open, so a developer or a runner
        with a configured wheelhouse would resolve the isolated build's
        backend from outside the checkout and still see a green gate. The
        `pip download` guard cannot catch that on its own -- it asks for a
        name a wheelhouse has no reason to carry.

        A configuration file is another such source and is disowned in both
        modes: the networked leg wants the default index rather than whichever
        mirror the host is pointed at. What is left is the dead index URL,
        which turns any resolve that still gets out into an immediate error
        rather than a timeout -- and the guard asserts it does.
        """
        env = {
            name: value for name, value in os.environ.items() if not name.startswith("PIP_")
        }
        env.update(
            {
                "PIP_CONFIG_FILE": os.devnull,
                "PIP_NO_INPUT": "1",
                "PIP_NO_CACHE_DIR": "1",
                "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                "TMPDIR": str(self.tmp),
                "TEMP": str(self.tmp),
                "TMP": str(self.tmp),
            }
        )
        if not self.network:
            env.update(
                {
                    "PIP_INDEX_URL": "http://127.0.0.1:1/simple",
                    "PIP_RETRIES": "0",
                    "PIP_TIMEOUT": "5",
                }
            )
        for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
            env.pop(name, None)
        return env


def _spawn(ctx: Context, argv: list[str], cwd: Path | None = None) -> Result:
    """Run a command in the pinned environment and decode what it said."""
    spelled = [str(part) for part in argv]
    try:
        proc = subprocess.run(
            spelled,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=ctx.timeout,
            cwd=None if cwd is None else str(cwd),
            env=ctx.env(),
        )
    except subprocess.TimeoutExpired as expired:
        # The timeout path raises with whatever chunks it had joined, which is
        # bytes on POSIX however the call was configured; printing those would
        # put the one thing worth reading inside a `b'...'` repr.
        partial = expired.stderr or ""
        if isinstance(partial, bytes):
            partial = partial.decode("utf-8", "replace")
        raise Failed(f"timed out after {ctx.timeout}s", f"$ {' '.join(spelled)}\n{partial}")
    return Result(spelled, proc.returncode, proc.stdout, proc.stderr)


def _ok(result: Result) -> Result:
    if result.returncode != 0:
        raise Failed(f"exited {result.returncode}", result.describe())
    return result


def _contains(result: Result, needle: str) -> None:
    if needle not in result.output:
        raise Failed(f"output does not contain {needle!r}", result.describe())


def _lines(result: Result) -> list[str]:
    return [line for line in result.stdout.splitlines() if line.strip()]


def _under(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


# the checks, in the order a user meets them


def check_hermetic_guard(ctx: Context) -> None:
    """A name only an index can answer must not resolve.

    This is the gate's self-test. Every later check passes `--no-index` and
    names a directory in the checkout, so all of them would keep passing if
    the environment silently grew a working index -- and then the gate would
    be measuring a package server. Here the guard is inverted: a plain
    `pip download` with no local link directory has to fail, and it has to
    fail for the stated reason rather than by crashing.
    """
    result = _spawn(
        ctx,
        [ctx.python, "-P", "-m", "pip", "download", "--no-deps", "--dest", ctx.root / "dl", "six"],
        cwd=ctx.root,
    )
    if result.returncode == 0:
        raise Failed("an index answered, so the run is not hermetic", result.describe())
    if "No matching distribution found" not in result.output:
        raise Failed("failed, but not by finding no distribution", result.describe())


def check_venv(ctx: Context) -> None:
    """`-m venv` builds a usable environment that points back at its base."""
    _ok(_spawn(ctx, [ctx.interpreter, "-m", "venv", ctx.venv], cwd=ctx.root))
    if not ctx.python.exists():
        raise Failed(f"no interpreter at {ctx.python}")
    config = ctx.venv / "pyvenv.cfg"
    if not config.exists():
        raise Failed(f"no pyvenv.cfg at {config}")
    settings = {}
    for line in config.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep:
            settings[key.strip()] = value.strip()
    if "home" not in settings:
        raise Failed("pyvenv.cfg has no home key", config.read_text(encoding="utf-8"))
    recorded = settings.get("executable")
    if recorded and not os.path.samefile(recorded, ctx.interpreter):
        raise Failed(
            f"pyvenv.cfg executable is {recorded!r}, not {ctx.interpreter!r}",
            config.read_text(encoding="utf-8"),
        )


def check_pip_import(ctx: Context) -> None:
    """The venv's pip imports, from the venv, and agrees with ensurepip."""
    probe = (
        "import pip, ensurepip\n"
        "print(pip.__version__)\n"
        "print(pip.__file__)\n"
        "print(ensurepip.version())\n"
    )
    result = _ok(_spawn(ctx, [ctx.python, "-P", "-c", probe], cwd=ctx.root))
    reported = _lines(result)
    if len(reported) != 3:
        raise Failed("expected three lines", result.describe())
    version, location, bundled = reported
    if version != bundled:
        raise Failed(f"pip reports {version}, ensurepip bundles {bundled}", result.describe())
    if not _under(Path(location), ctx.venv):
        raise Failed(f"pip came from {location}, not from the venv", result.describe())
    ctx.pipver = version


def check_pip_cli(ctx: Context) -> None:
    """The console entry point runs and names the same pip the import did."""
    result = _ok(_spawn(ctx, [ctx.python, "-P", "-m", "pip", "--version"], cwd=ctx.root))
    if not re.match(rf"^pip {re.escape(ctx.pipver)} from .*\(python 3\.\d+\)$", result.stdout.strip()):
        raise Failed(f"unexpected version line for pip {ctx.pipver}", result.describe())


def check_offline_wheel_install(ctx: Context) -> None:
    """A real wheel installs from a local directory, over a running copy.

    Reinstalling pip on top of itself is the unglamorous half of an install:
    the existing distribution's RECORD has to be read and its files removed
    while the tool doing the removing is the one being replaced, and the
    console script has to come back working. The wheel is the one the
    checkout ships, so this is a couple of thousand members rather than a toy.
    """
    name, version = _sole_wheel(BUNDLED, "pip")
    result = _ok(
        _spawn(
            ctx,
            [
                ctx.python, "-P", "-m", "pip", "install",
                "--no-index", "--find-links", BUNDLED, "--force-reinstall", "pip",
            ],
            cwd=ctx.root,
        )
    )
    _contains(result, f"Successfully installed {name}-{version}")
    after = _ok(_spawn(ctx, [ctx.python, "-P", "-m", "pip", "--version"], cwd=ctx.root))
    if not after.stdout.strip().startswith(f"pip {version} "):
        raise Failed("the reinstalled pip does not report itself", after.describe())


def check_pep517_without_build_deps(ctx: Context) -> None:
    """A PEP 517 build whose backend needs nothing installed.

    Separates the hook protocol from what populates the build environment: if
    this passes and the setuptools build below does not, the isolation
    machinery works and the thing it failed to install is the subject.
    """
    result = _ok(
        _spawn(
            ctx,
            [ctx.python, "-P", "-m", "pip", "install", "--no-index", ctx.src / "tinypkg"],
            cwd=ctx.root,
        )
    )
    _contains(result, "Successfully built tinypkg")
    _contains(result, "Successfully installed tinypkg-0.1.0")
    # From the run root and with `-P`, so nothing but the install can answer
    # the import -- the fixture directory holds a `tinypkg.py` that would
    # satisfy it just as well.
    imported = _ok(
        _spawn(ctx, [ctx.python, "-P", "-c", "import tinypkg; print(tinypkg.hello())"], cwd=ctx.root)
    )
    if imported.stdout.strip() != "hello from tinypkg":
        raise Failed("the installed module did not answer", imported.describe())


def check_sdist_build(ctx: Context) -> None:
    """The runtime writes a source distribution the driver can read back."""
    archive = ctx.dist / SDIST
    _ok(_spawn(ctx, [ctx.python, "-P", MKSDIST, ctx.src / "stpkg", archive], cwd=ctx.root))
    if not archive.exists():
        raise Failed(f"no archive at {archive}")
    with tarfile.open(archive) as opened:
        members = set(opened.getnames())
    missing = SDIST_MEMBERS - members
    if missing:
        raise Failed(f"archive is missing {sorted(missing)}", f"members: {sorted(members)}")


def check_sdist_to_wheel_isolated(ctx: Context) -> None:
    """A source distribution builds through a real isolated environment.

    The build backend is resolved and installed by a nested pip into a
    throwaway prefix -- the outer `--no-index --find-links` reach it, which is
    what makes an isolated build possible with no index at all. `-v` is what
    surfaces the nested install's own line, and that line is the evidence the
    isolation ran rather than being skipped.
    """
    name, version = _sole_wheel(WHEELDATA, "setuptools")
    result = _ok(
        _spawn(
            ctx,
            [
                ctx.python, "-P", "-m", "pip", "install", "-v",
                "--no-index", "--find-links", WHEELDATA, ctx.dist / SDIST,
            ],
            cwd=ctx.root,
        )
    )
    _contains(result, "Installing build dependencies")
    _contains(result, f"Successfully installed {name}-{version}")
    _contains(result, "Created wheel for stpkg")
    _contains(result, "Successfully built stpkg")
    _contains(result, "Successfully installed stpkg-0.2.0")


def check_console_script(ctx: Context) -> None:
    """The generated console script runs on its own and points into the venv."""
    script = ctx.script("stpkg-hi")
    if not script.exists():
        raise Failed(f"no console script at {script}")
    result = _ok(_spawn(ctx, [script], cwd=ctx.root))
    if result.stdout.strip() != "hi from stpkg":
        raise Failed("the console script did not answer", result.describe())
    if sys.platform != "win32":
        # Windows gets an executable shim instead, whose target is not
        # readable as text.
        first = script.read_text(encoding="utf-8", errors="replace").splitlines()[0]
        if not first.startswith("#!") or not os.path.samefile(first[2:].strip(), ctx.python):
            raise Failed(f"console script shebang is {first!r}", first)


def check_entry_point_metadata(ctx: Context) -> None:
    """The installed distribution's metadata reads back through the stdlib."""
    probe = (
        "import importlib.metadata as m\n"
        "d = m.distribution('stpkg')\n"
        "print(d.version)\n"
        "print(sorted((e.name, e.value) for e in d.entry_points))\n"
    )
    result = _ok(_spawn(ctx, [ctx.python, "-P", "-c", probe], cwd=ctx.root))
    reported = _lines(result)
    expected = ["0.2.0", "[('stpkg-hi', 'stpkg:main')]"]
    if reported != expected:
        raise Failed(f"metadata reads {reported}, expected {expected}", result.describe())


def check_pip_list(ctx: Context) -> None:
    """Everything installed so far is what pip reports as installed."""
    result = _ok(_spawn(ctx, [ctx.python, "-P", "-m", "pip", "list", "--format=freeze"], cwd=ctx.root))
    listed = set(_lines(result))
    _, pipver = _sole_wheel(BUNDLED, "pip")
    expected = {f"pip=={pipver}", "stpkg==0.2.0", "tinypkg==0.1.0"}
    missing = expected - listed
    if missing:
        raise Failed(f"pip list is missing {sorted(missing)}", result.describe())


def check_uninstall(ctx: Context) -> None:
    """Uninstalling removes the modules and the scripts that came with them."""
    result = _ok(
        _spawn(ctx, [ctx.python, "-P", "-m", "pip", "uninstall", "-y", "stpkg", "tinypkg"], cwd=ctx.root)
    )
    _contains(result, "Successfully uninstalled stpkg-0.2.0")
    _contains(result, "Successfully uninstalled tinypkg-0.1.0")
    # Both, separately: one command reported two uninstalls, and a module left
    # behind by either of them is the thing worth catching.
    for module in ("stpkg", "tinypkg"):
        gone = _spawn(ctx, [ctx.python, "-P", "-c", f"import {module}"], cwd=ctx.root)
        if gone.returncode == 0:
            raise Failed(f"{module} still imports after being uninstalled", gone.describe())
        if "ModuleNotFoundError" not in gone.output:
            raise Failed(f"the {module} import failed for some other reason", gone.describe())
    if ctx.script("stpkg-hi").exists():
        raise Failed(f"{ctx.script('stpkg-hi')} outlived its distribution")


def check_network_download(ctx: Context) -> None:
    """An index answers over TLS. Only under `--with-network`."""
    result = _ok(
        _spawn(
            ctx,
            [ctx.python, "-P", "-m", "pip", "download", "--no-deps", "--dest", ctx.root / "net", "six"],
            cwd=ctx.root,
        )
    )
    _contains(result, "Saved")


HERMETIC: list[tuple[str, Callable[[Context], None]]] = [
    ("venv", check_venv),
    ("pip-import", check_pip_import),
    ("pip-cli", check_pip_cli),
    # Standing between the checks that only need the venv and the first one
    # that resolves anything.
    ("hermetic-guard", check_hermetic_guard),
    ("offline-wheel-install", check_offline_wheel_install),
    ("pep517-no-build-deps", check_pep517_without_build_deps),
    ("sdist-build", check_sdist_build),
    ("sdist-to-wheel-isolated", check_sdist_to_wheel_isolated),
    ("console-script", check_console_script),
    ("entry-point-metadata", check_entry_point_metadata),
    ("pip-list", check_pip_list),
    ("uninstall", check_uninstall),
]

NETWORKED: list[tuple[str, Callable[[Context], None]]] = [
    ("network-download", check_network_download),
]


def _checks(network: bool) -> list[tuple[str, Callable[[Context], None]]]:
    # The guard asserts there is no index, so it is the one check a networked
    # run has to drop rather than reorder.
    if not network:
        return HERMETIC
    return [pair for pair in HERMETIC if pair[0] != "hermetic-guard"] + NETWORKED


def _prepare(root: Path) -> None:
    """Lay out one run's working tree.

    The fixtures are copied because installing from a source directory writes
    build artefacts beside it, and the originals are tracked files.
    """
    (root / "tmp").mkdir(parents=True, exist_ok=True)
    (root / "dist").mkdir(parents=True, exist_ok=True)
    shutil.copytree(FIXTURES, root / "src")


def _sequence(
    backend: str, interpreter: str, root: Path, network: bool, timeout: int, verbose: bool
) -> list[Failure]:
    """Run every check for one interpreter, stopping at the first failure.

    Stopping is not a policy choice: the checks share one venv and each builds
    on the last, so a later check after a failure would report the earlier
    defect a second time under a name that does not describe it.
    """
    _prepare(root)
    ctx = Context(backend, interpreter, root, network, timeout)
    for name, check in _checks(network):
        try:
            check(ctx)
        except Failed as failed:
            print(f"  {name:<26s} {backend}=FAIL  ({failed.reason})")
            return [Failure(name, backend, failed.reason, failed.evidence)]
        if verbose:
            print(f"  {name:<26s} {backend}=OK")
    return []


def _probe(command: str) -> tuple[tuple[int, int], str] | None:
    """What an interpreter reports as its version and its own path."""
    probe = "import sys; print(sys.version_info[0], sys.version_info[1]); print(sys.executable)"
    try:
        proc = subprocess.run([command, "-c", probe], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    reported = (proc.stdout or "").splitlines()
    if len(reported) < 2:
        return None
    try:
        major, minor = reported[0].split()
    except ValueError:
        return None
    return (int(major), int(minor)), reported[1].strip() or command


def _cpython() -> str | None:
    """The reference interpreter, if one of the right version is around.

    Unlike the parity runner this does not stop when there is none: the
    reference is a control here, not a comparand, and a run that found a real
    failure should still report it.
    """
    named = os.environ.get("PYRE_CHECK_PYTHON3")
    for candidate in [named] if named else ["python3.15", "python3", "python"]:
        if named is None and shutil.which(candidate) is None:
            continue
        probed = _probe(candidate)
        if probed is not None and probed[0] == CPYTHON_TARGET:
            return probed[1]
    return None


def _control(failure: Failure, network: bool, timeout: int, keep: bool) -> str:
    """Whether the reference interpreter fails the same check.

    Answers the question a red gate raises first: did the runtime break, or
    did the fixture rot? It costs nothing on a green run because it is only
    reached once something has already failed.
    """
    reference = _cpython()
    if reference is None:
        return "fixture control: no CPython %d.%d found, so not run" % CPYTHON_TARGET
    root = Path(tempfile.mkdtemp(prefix="pyre-pip-control-"))
    try:
        failures = _sequence("cpython", reference, root, network, timeout, verbose=False)
    finally:
        if not keep:
            shutil.rmtree(root, ignore_errors=True)
    if not failures:
        return f"fixture control: cpython passed every check, so {failure.check} is a pyre defect"
    if failures[0].check == failure.check:
        return (
            f"fixture control: cpython fails {failure.check} too "
            f"({failures[0].reason}) -- the fixture or the bundled wheels rotted"
        )
    return (
        f"fixture control: cpython got no further than {failures[0].check} "
        f"({failures[0].reason}), so this run proves nothing either way"
    )


def _report(failures: list[Failure], control: str) -> None:
    print("=" * 72)
    print(f"{len(failures)} failure(s)")
    print(control)
    for failure in failures:
        print()
        print(f"  {failure.check} [{failure.backend}]: {failure.reason}")
        for line in failure.evidence.strip().splitlines():
            print(f"      {line}")
    print("=" * 72)


def _annotate(failures: list[Failure], control: str) -> None:
    """One GitHub Actions error annotation per failure."""
    path = HERE.relative_to(ROOT).joinpath("run.py").as_posix()
    for failure in failures:
        message = f"{failure.backend}: {failure.check}: {failure.reason} | {control}"
        escaped = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        print(f"::error file={path},title=pip::{escaped}")


def _backends(only_dynasm: bool, only_cranelift: bool) -> list[tuple[str, str]]:
    """The release binaries to drive.

    Named individually rather than by globbing `pyre*`: `target/release/pyre`
    is whatever the last build wrote there, and a sandbox build of that name
    has no filesystem and no stdlib to find.
    """
    backends = []
    dynasm = TARGET_RELEASE / f"pyre-dynasm{EXE}"
    cranelift = TARGET_RELEASE / f"pyre-cranelift{EXE}"
    if not only_cranelift and dynasm.exists():
        backends.append(("dynasm", str(dynasm)))
    if not only_dynasm and cranelift.exists():
        backends.append(("cranelift", str(cranelift)))
    return backends


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynasm-only", action="store_true")
    parser.add_argument("--cranelift-only", action="store_true")
    parser.add_argument(
        "--with-network",
        action="store_true",
        help="also resolve from a real index; never used by the merge gate",
    )
    parser.add_argument("--keep", action="store_true", help="keep the working tree and print its path")
    parser.add_argument("--no-cpython-control", action="store_true")
    parser.add_argument("--timeout", type=int, default=TIMEOUT, help="seconds per check")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

    for directory in (BUNDLED, WHEELDATA):
        if not directory.is_dir():
            print(f"missing wheel directory: {directory}", file=sys.stderr)
            return 1

    backends = _backends(args.dynasm_only, args.cranelift_only)
    if not backends:
        print(f"no pyre release binary under {TARGET_RELEASE}", file=sys.stderr)
        return 1

    checks = _checks(args.with_network)
    print(f"backends: {[name for name, _ in backends]}")
    print(f"checks: {len(checks)}{' (with network)' if args.with_network else ' (hermetic)'}")
    print()

    failures: list[Failure] = []
    for backend, interpreter in backends:
        root = Path(tempfile.mkdtemp(prefix=f"pyre-pip-{backend}-"))
        print(f"{backend}: {interpreter}")
        found = _sequence(backend, interpreter, root, args.with_network, args.timeout, verbose=True)
        failures.extend(found)
        if found or args.keep:
            print(f"  working tree kept at {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)
        print()

    if not failures:
        print("pip end-to-end passes on every backend")
        return 0

    control = "fixture control: skipped"
    if not args.no_cpython_control:
        control = _control(failures[0], args.with_network, args.timeout, args.keep)
    _report(failures, control)
    if os.environ.get("GITHUB_ACTIONS") == "true":
        _annotate(failures, control)
    return 1


if __name__ == "__main__":
    sys.exit(main())
