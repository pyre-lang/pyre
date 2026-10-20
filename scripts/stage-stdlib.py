#!/usr/bin/env python3
"""Stage Pyre's release stdlib.

This is the stdlib half of `pypy/tool/release/package.py:192-390 create_package`,
ported here rather than invoked: every function below has a counterpart there and
is cited by line.  Read that file before changing the layout.

The rest of `create_package` has no counterpart here.  `:233-240` runs the
`lib_pypy` cffi build scripts for `_ssl`, `sqlite3`, `_blake2`, `_sha3`,
`_tkinter` and the rest, which Pyre implements natively; `smartstrip` and
`make_portable` post-process a translated binary; and `:449-486` writes the zip
or tarball itself, which `dist` does for us out of the `include` entry in
`dist-workspace.toml`.  What is left of it would not run unmodified anyway:
`:221` spells the implementation directory `pypy{ver}`, so it would stage where
`sysconfig._get_implementation` does not look.

Two deliberate departures, both noted at the site: `lib_pypy` is not overlaid,
and the bundled pip wheel is verified against `ensurepip`.
"""

from __future__ import annotations

import argparse
import ast
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

# package.py:221 `IMPLEMENTATION = 'pypy{}'.format(python_ver)`.  The trailing
# `t` is the free-threaded ABI flag: `sysconfig` derives `abi_thread` from
# `Py_GIL_DISABLED` and the posix schemes put it in the directory name.
IMPLEMENTATION = "pyre3.15t"


# The suffix `cpyext::extension_suffix` publishes for each release target.
# The importer looks for exactly this name, so a file staged under any other
# one is never found -- which is what a host-derived suffix produces when the
# archive is cross-compiled.
EXTENSION_SUFFIXES = {
    "aarch64-apple-darwin": ".pyre315-darwin.so",
    "x86_64-apple-darwin": ".pyre315-darwin.so",
    "aarch64-unknown-linux-gnu": ".pyre315-aarch64-linux-gnu.so",
    "x86_64-unknown-linux-gnu": ".pyre315-x86_64-linux-gnu.so",
}


def host_target() -> str | None:
    """The release triple this host builds loadable extensions for."""
    machine = platform.machine().lower()
    architecture = {
        "x86_64": "x86_64",
        "amd64": "x86_64",
        "aarch64": "aarch64",
        "arm64": "aarch64",
    }.get(machine)
    if architecture is None:
        return None
    if sys.platform == "darwin":
        return f"{architecture}-apple-darwin"
    if sys.platform.startswith("linux"):
        return f"{architecture}-unknown-linux-gnu"
    return None


def build_sqlite3_cffi(destination: Path, targets: list[str]) -> bool:
    """PyPy `package.py:create_package`'s `_sqlite3_build.py` entry.

    The generated library must be compiled by PyPy so cffi selects its
    `_cffi_pypyinit_*` form.  Pyre's extension loader consumes that immutable
    type context directly; renaming only the import suffix does not alter the
    native ABI.

    Answers whether an owner for `_sqlite3` was staged.
    """
    # cffi compiles through the host toolchain, so the library it produces
    # loads only on the host's own triple.  A cross-compiled archive -- and
    # Windows, where host dlopen is not built at all -- gets no extension
    # rather than one its interpreter can never load.
    host = host_target()
    staged = [target for target in targets if target == host and target in EXTENSION_SUFFIXES]
    if not staged:
        return False
    pypy = os.environ.get("PYPY3") or shutil.which("pypy3")
    if not pypy:
        raise SystemExit("staging sqlite3 requires pypy3 (or PYPY3)")
    helper = r"""
import ctypes.util
import sys

source_root, output_root = sys.argv[1:]
if sys.platform == "darwin":
    original_find_library = ctypes.util.find_library
    ctypes.util.find_library = lambda name: (
        "/usr/lib/libsqlite3.dylib"
        if name == "sqlite3" and original_find_library(name) is None
        else original_find_library(name)
    )
sys.path.insert(0, source_root)
import _sqlite3_build
print("PYRE_SQLITE3_CFFI=" + _sqlite3_build._ffi.compile(
    tmpdir=output_root,
    verbose=True,
))
"""
    with tempfile.TemporaryDirectory(prefix="pyre-sqlite3-cffi-") as temporary:
        result = subprocess.run(
            [pypy, "-c", helper, str(ROOT / "lib_pypy"), temporary],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        marker = "PYRE_SQLITE3_CFFI="
        outputs = [line.removeprefix(marker) for line in result.stdout.splitlines() if line.startswith(marker)]
        if len(outputs) != 1:
            raise SystemExit("_sqlite3_build.py did not report exactly one extension path")
        for target in staged:
            shutil.copy2(outputs[0], destination / f"_sqlite3_cffi{EXTENSION_SUFFIXES[target]}")
    shutil.copy2(ROOT / "lib_pypy" / "_sqlite3.py", destination / "_sqlite3.py")
    return True


def ignored(_directory: str, names: list[str]) -> set[str]:
    ignored_names = {"__pycache__"} & set(names)
    ignored_names.update(name for name in names if name.endswith((".pyc", ".pyo", ".o", ".obj")))
    return ignored_names


def ensurepip_version(ensurepip: Path) -> str:
    """Read the bundled version without importing the stdlib being staged."""
    module = ast.parse(ensurepip.read_text(encoding="utf-8"), filename=str(ensurepip))
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "_PIP_VERSION" for target in statement.targets):
            value = ast.literal_eval(statement.value)
            if isinstance(value, str):
                return value
    raise SystemExit(f"cannot determine _PIP_VERSION from {ensurepip}")


def verify_bundled_pip(destination: Path) -> None:
    """No package.py counterpart.

    `ensurepip` derives the wheel it installs from `_PIP_VERSION`, so a wheel
    that does not match the name it builds fails when a user bootstraps pip
    rather than when the release is built.
    """
    version = ensurepip_version(destination / "ensurepip" / "__init__.py")
    wheel_dir = destination / "ensurepip" / "_bundled"
    expected = wheel_dir / f"pip-{version}-py3-none-any.whl"
    wheels = sorted(wheel_dir.glob("pip-*.whl"))
    if wheels != [expected]:
        names = ", ".join(path.name for path in wheels) or "<none>"
        raise SystemExit(
            f"ensurepip expects {expected.name}, staged pip wheels are: {names}"
        )
    try:
        with zipfile.ZipFile(expected) as wheel:
            corrupt = wheel.testzip()
            if corrupt is not None:
                raise SystemExit(f"corrupt member in {expected}: {corrupt}")
            if "pip/__init__.py" not in wheel.namelist():
                raise SystemExit(f"{expected} does not contain pip/__init__.py")
    except zipfile.BadZipFile as error:
        raise SystemExit(f"invalid bundled pip wheel {expected}: {error}") from error


def stdlib_destination(assets_root: Path) -> Path:
    """package.py:222-227 — where the release keeps the stdlib on this platform.

    Windows holds the whole thing in `Lib`: that is where `sysconfig`'s `nt`
    scheme puts `stdlib` (`{installed_base}/Lib`) and where
    `site.getsitepackages` looks for `site-packages` (`<prefix>/Lib`).  Every
    other platform uses `<platlibdir>/<implementation><version>`, and
    `sys.platlibdir` is `lib`.
    """
    if os.name == "nt":
        return assets_root / "Lib"
    return assets_root / "lib" / IMPLEMENTATION


def stage(assets_root: Path, targets: list[str]) -> None:
    destination = stdlib_destination(assets_root).resolve()
    allowed_root = (ROOT / "dist-assets").resolve()
    if destination == allowed_root or allowed_root not in destination.parents:
        raise SystemExit(f"destination must be below {allowed_root}: {destination}")

    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    # pypy/tool/release/package.py copies lib-python first, then overlays
    # lib_pypy into the same implementation-version directory.  Pyre selects
    # only the app-level owners it still needs: overlaying all of lib_pypy
    # would expose CPython-only test shims such as `_testcapi` and suppress
    # intended fallbacks for modules pyre owns natively.
    shutil.copytree(ROOT / "lib-python" / "3", destination, ignore=ignored)
    # PyPy 5578: `_pypy_abi3_tags` lives in lib_pypy.  Overlay just that
    # file so site.py can import it without putting the rest of lib_pypy
    # on sys.path.
    abi3_tags = ROOT / "lib_pypy" / "_pypy_abi3_tags.py"
    if abi3_tags.is_file():
        shutil.copy2(abi3_tags, destination / "_pypy_abi3_tags.py")
    if not build_sqlite3_cffi(destination, targets):
        # `sqlite3/dbapi2.py` opens with `from _sqlite3 import *`, so without an
        # owner the package raises from inside itself on every import.  Ship
        # the absence instead: `import sqlite3` then reports a missing module,
        # which is what the archive actually offers.
        shutil.rmtree(destination / "sqlite3", ignore_errors=True)
        print(
            "stage-stdlib: no loadable _sqlite3 owner for "
            f"{', '.join(targets) or 'this host'}; the sqlite3 package is not staged",
            file=sys.stderr,
        )

    if not (destination / "site.py").is_file():
        raise SystemExit("staged stdlib does not contain site.py")
    verify_bundled_pip(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "assets_root",
        type=Path,
        help="directory whose contents dist copies into the archive root",
    )
    parser.add_argument(
        "targets",
        nargs="*",
        help="Rust target triples this archive is built for (default: the host)",
    )
    args = parser.parse_args()
    targets = args.targets or [target for target in [host_target()] if target]
    stage(args.assets_root, targets)


if __name__ == "__main__":
    main()
