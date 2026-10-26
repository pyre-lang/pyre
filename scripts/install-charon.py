#!/usr/bin/env python3
"""Fetch or build the pinned Charon release into pyre's shared build cache."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional


CHARON_VERSION_DEFAULT = "nightly-2026.10.25"


def default_charon_dest(shared: Path, platform_key: str, version: str) -> Path:
    """Versioned install dir. Sibling worktrees pin other tags beside this one."""
    return shared / "charon" / platform_key / version


def default_charon_src(shared: Path, platform_key: str, version: str) -> Path:
    """Versioned source checkout. An explicit `CHARON_SRC` replaces this whole path."""
    return shared / "charon-src" / platform_key / version


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def platform_info() -> tuple[str, Optional[str], str, bool]:
    system = platform.system()
    machine = platform.machine().lower()
    force_source = os.environ.get("CHARON_FROM_SOURCE") == "1"

    if system == "Darwin" and machine in {"arm64", "aarch64"}:
        return "darwin-arm64", "charon-macos-aarch64.tar.gz", "", force_source
    if system == "Darwin" and machine == "x86_64":
        return "darwin-x86_64", "charon-macos-x86_64.tar.gz", "", force_source
    if system == "Linux" and machine in {"arm64", "aarch64"}:
        return "linux-aarch64", "charon-linux-aarch64.tar.gz", "", force_source
    if system == "Linux" and machine == "x86_64":
        return "linux-x86_64", "charon-linux-x86_64.tar.gz", "", force_source
    if system == "Windows" or system.startswith(("MSYS", "MINGW", "CYGWIN")):
        return "windows", None, ".exe", True

    raise SystemExit(
        f"install-charon.py: unsupported platform {system}-{machine}\n"
        "  prebuilt: darwin-aarch64, darwin-x86_64, linux-aarch64, linux-x86_64\n"
        "  set CHARON_FROM_SOURCE=1 to build from source instead"
    )


def prepend_msvc_link(env: dict[str, str]) -> None:
    if platform_info()[0] != "windows":
        return
    vswhere = Path(
        "C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe"
    )
    if not vswhere.exists():
        print(
            "warn: vswhere not found; MSVC link.exe may be shadowed by Git's link.exe",
            file=sys.stderr,
        )
        return
    try:
        install = subprocess.run(
            [
                str(vswhere),
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        install = ""
    if not install:
        return
    msvc_root = Path(install) / "VC" / "Tools" / "MSVC"
    candidates = sorted(msvc_root.glob("*/bin/Hostx64/x64/link.exe"))
    if not candidates:
        candidates = sorted(msvc_root.glob("*/bin/hostx64/x64/link.exe"))
    if candidates:
        env["PATH"] = str(candidates[-1].parent) + os.pathsep + env.get("PATH", "")
    else:
        print(
            f"warn: MSVC link.exe not found under {install}; Git's link.exe may shadow it",
            file=sys.stderr,
        )


def download(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url) as response, dest.open("wb") as output:
        shutil.copyfileobj(response, output)


def main() -> None:
    version = os.environ.get("CHARON_VERSION", CHARON_VERSION_DEFAULT)
    root = repo_root()
    shared = Path(os.environ.get("PYRE_SHARED_BUILD", root.parent / ".pyre-build"))
    platform_key, asset, exe, from_source = platform_info()
    charon_dest = Path(
        os.environ.get(
            "CHARON_DEST", default_charon_dest(shared, platform_key, version)
        )
    )

    stamp = charon_dest / ".installed-version"
    charon_bin = charon_dest / f"charon{exe}"
    if charon_bin.exists() and stamp.exists():
        current = stamp.read_text().strip()
        if current == version:
            print(f"charon {version} already installed at {charon_dest}")
            return
        print(f"charon at {charon_dest} is {current}; replacing with {version}")

    charon_dest.mkdir(parents=True, exist_ok=True)

    if from_source:
        charon_src = Path(
            os.environ.get(
                "CHARON_SRC", default_charon_src(shared, platform_key, version)
            )
        )
        if (charon_src / ".git").is_dir():
            print(f"updating {charon_src} to {version}")
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(charon_src),
                    "fetch",
                    "--depth",
                    "1",
                    "origin",
                    f"refs/tags/{version}:refs/tags/{version}",
                ],
                check=True,
            )
            subprocess.run(["git", "-C", str(charon_src), "checkout", "-q", version], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(charon_src),
                    "submodule",
                    "update",
                    "--init",
                    "--recursive",
                    "--depth",
                    "1",
                ],
                check=True,
            )
        else:
            print(f"cloning charon {version} into {charon_src}")
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    version,
                    "--recurse-submodules",
                    "--shallow-submodules",
                    "https://github.com/AeneasVerif/charon.git",
                    str(charon_src),
                ],
                check=True,
            )

        print("building charon (cargo build --release; first run installs the pinned nightly)")
        env = os.environ.copy()
        prepend_msvc_link(env)
        subprocess.run(["cargo", "build", "--release"], cwd=charon_src / "charon", env=env, check=True)
        release = charon_src / "charon" / "target" / "release"
        shutil.copy2(release / f"charon{exe}", charon_dest / f"charon{exe}")
        shutil.copy2(release / f"charon-driver{exe}", charon_dest / f"charon-driver{exe}")
    else:
        if asset is None:
            raise SystemExit("install-charon.py: no release asset for this platform")
        url = f"https://github.com/AeneasVerif/charon/releases/download/{version}/{asset}"
        print(f"fetching {url}")
        with tempfile.TemporaryDirectory() as tmp_dir:
            archive = Path(tmp_dir) / asset
            download(url, archive)
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(tmp_dir)
            shutil.move(str(Path(tmp_dir) / "charon"), charon_dest / "charon")
            shutil.move(str(Path(tmp_dir) / "charon-driver"), charon_dest / "charon-driver")

    stamp.write_text(version + "\n")
    print()
    print(f"installed: {charon_bin}")
    subprocess.run([str(charon_bin), "version"], check=False)
    print()
    print("next: trigger the rustc nightly install (one-time, ~1 minute):")
    print(f"  {charon_bin} toolchain-path")


if __name__ == "__main__":
    main()
