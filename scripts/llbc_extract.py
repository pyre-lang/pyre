"""Charon ULLBC extraction engine with source-fingerprint skip logic.

Import-only library: a per-repo driver declares its crate table as a dict of
`CrateSpec`s and calls `run_cli(...)`. This module carries ZERO crate names —
every crate (its dir, cargo flags, fingerprint inputs, output artefact) is
declared by the driver, so the engine stays neutral about which consumer
(pyre, or an external interpreter crate) it is extracting for.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import platform
import posixpath
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import typing
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath


def install_charon():
    """`scripts/install-charon.py`, the single owner of the Charon pin and layout."""
    path = Path(__file__).resolve().parent / "install-charon.py"
    spec = importlib.util.spec_from_file_location("install_charon", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"extract-llbc.py: cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Domain separator for the bytes hashed by the repository fingerprints. Bump this
# only when the fingerprint algorithm or the meaning of its input set changes.
# Ordinary refactors, diagnostics and comments in this file do not change what
# Charon compiles and must not invalidate every multi-minute LLBC artefact.
FINGERPRINT_SCHEMA = "4"


@dataclass
class CrateSpec:
    """One extractable crate.

    - `crate_dir`: absolute dir where `charon cargo` runs (holds Cargo.toml/src).
    - `output_name`: artefact filename under the driver's output dir
      (e.g. `<crate>.ullbc`).
    - `cargo_args`: extra flags passed after `--`; each arg may contain the
      `{features}` placeholder, substituted with the active cargo feature set
      (e.g. `["--features", "{features}"]` or `["--no-default-features"]`).
    - `charon_args`: extra flags passed to Charon itself, before the `--`
      separator (e.g. `["--include", "somecrate::module::_"]` to translate the
      bodies of items in a foreign dependency instead of keeping them opaque).
      Same `{features}` placeholder substitution as `cargo_args`.
    - `fingerprint_pathspecs`: explicit git pathspecs (relative to the driver's
      `root`) used as the conservative bootstrap set when no artefact-derived
      `.readfiles` sidecar exists. `None` derives that fallback from a `cargo
      metadata` dependency walk instead.
    - `layout_targets`: target triples, besides the extraction host, this
      crate also emits a layout sidecar for (see `layout_sidecar_name`).
      `None` takes the driver's default; `()` opts out. Every listed target
      must compile under `layout_cargo_args`.
    - `layout_cargo_args`: `cargo_args` replacement for the sidecar passes.
      A cross target usually needs a different feature set than the host
      (a native code generator does not build for wasm32). `None` reuses
      `cargo_args`.
    """

    name: str
    crate_dir: Path
    output_name: str
    cargo_args: list[str] = field(default_factory=list)
    charon_args: list[str] = field(default_factory=list)
    fingerprint_pathspecs: list[str] | None = None
    layout_targets: tuple[str, ...] | None = None
    layout_cargo_args: list[str] | None = None

    def layout_sidecar_name(self, target: str) -> str:
        """Artefact name of this crate's `target` layout sidecar."""
        return f"{Path(self.output_name).stem}.{target}.layouts.ullbc"


def child_dirs_whose_rs_mention(root: Path, needle: str) -> list[str]:
    """Immediate subdirectories of `root` that have a `.rs` file containing `needle`.

    Used to find interpreter `src/module/<name>/` trees that carry a JIT
    hint macro, so Charon can `--start-from-if-exists` those modules (and
    their marker consts) without a hand-maintained name list.
    """
    if not root.is_dir():
        return []
    names: list[str] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        for rs in child.rglob("*.rs"):
            try:
                text = rs.read_text(encoding="utf-8")
            except OSError:
                continue
            if needle in text:
                names.append(child.name)
                break
    return names


@dataclass
class Engine:
    """Resolved driver configuration threaded through the extraction helpers."""

    specs: dict[str, CrateSpec]
    default_crates: list[str]
    root: Path
    out_dir: Path
    base_pathspecs: list[str]
    charon_root: Path
    cargo_features: str
    extraction_abi: str
    metadata_feature_crates: tuple[str, ...] = ()
    layout_targets: tuple[str, ...] = ()
    layout_target_rustflags: str = ""
    # Files/directories a driver declares by hand because the git channel
    # cannot carry them. Two kinds, and BOTH are hashed by content:
    #
    #   * outside `root` — the engine module itself, a sibling checkout. `git
    #     ls-files` refuses a pathspec that leaves the repository.
    #   * inside `root` but IGNORED — an uncommitted `.cargo/config.toml`, a
    #     lockfile a repo gitignores. `_collect_inputs` resolves pathspecs
    #     through `ls-files ∪ ls-files --others --exclude-standard`, and an
    #     ignored file is in neither, so declaring it as a pathspec is inert.
    #     `refuse_inert_pathspecs` names this field as the remedy, so keep it
    #     accepting in-root paths: labels come from `os.path.relpath`, which
    #     spells an in-root path without any `..`.
    #
    # Patched path deps need no declaration — `_collect_inputs` discovers those.
    external_inputs: tuple[Path, ...] = ()

    def spec(self, crate: str) -> CrateSpec:
        try:
            return self.specs[crate]
        except KeyError:
            known = " ".join(self.specs)
            raise SystemExit(
                f"extract-llbc.py: unknown crate '{crate}'\n  known: {known}"
            )


def platform_info() -> tuple[str, str]:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin" and machine in {"arm64", "aarch64"}:
        return "darwin-arm64", "charon"
    if system == "Darwin" and machine == "x86_64":
        return "darwin-x86_64", "charon"
    if system == "Linux" and machine in {"arm64", "aarch64"}:
        return "linux-aarch64", "charon"
    if system == "Linux" and machine == "x86_64":
        return "linux-x86_64", "charon"
    if system == "Windows" or system.startswith(("MSYS", "MINGW", "CYGWIN")):
        return "windows", "charon.exe"
    raise SystemExit(f"extract-llbc.py: unsupported platform {system}-{machine}")


def expand_features(arg: str, cargo_features: str) -> str:
    features = [f.strip() for f in cargo_features.split(",") if f.strip()]
    if "{features}" not in arg or len(features) <= 1:
        # No placeholder, or a single/absent feature: whole-string
        # substitution already yields the right flag.
        return arg.format(features=cargo_features)
    # Multiple features: a template like `crate/{features}` prefixes the
    # placeholder, so splicing the raw `a,b` list into one slot only
    # prefixes the first feature. Expand the template per feature and
    # rejoin so each feature keeps the prefix.
    return ",".join(arg.format(features=feature) for feature in features)


def crate_flags(spec: CrateSpec, cargo_features: str) -> list[str]:
    # Charon's rustc is the nightly-2026-09-17 toolchain (1.100.0-nightly).
    # That is newer than every rust-version this workspace's crates declare,
    # so cargo does not need `--ignore-rust-version`.
    return [expand_features(arg, cargo_features) for arg in spec.cargo_args]


def charon_crate_flags(spec: CrateSpec, cargo_features: str) -> list[str]:
    return [expand_features(arg, cargo_features) for arg in spec.charon_args]


def crate_layout_flags(
    spec: CrateSpec, cargo_features: str, flags: list[str]
) -> list[str]:
    """Cargo args for this crate's layout sidecar passes.

    `layout_cargo_args` when the spec declares them (a cross target usually
    needs a different feature set), otherwise the host extraction `flags`.
    """
    if spec.layout_cargo_args is None:
        return flags
    return [
        "--ignore-rust-version",
        *[expand_features(arg, cargo_features) for arg in spec.layout_cargo_args],
    ]


def features_in_cargo_flags(crate: str, flags: list[str]) -> list[str]:
    """Feature names a cargo arg list enables, workspace-qualified.

    A bare name is crate-relative (the extraction passes run cargo inside the
    crate directory) and must be spelled `crate/feature` for the
    workspace-level `cargo metadata`. `--no-default-features` is ignored:
    dropping it can only widen the graph, and the fingerprint may
    over-approximate but must never miss an input.
    """
    features: list[str] = []
    index = 0
    while index < len(flags):
        arg = flags[index]
        if arg in ("--features", "-F") and index + 1 < len(flags):
            value = flags[index + 1]
            index += 2
        elif arg.startswith(("--features=", "-F=")):
            value = arg.split("=", 1)[1]
            index += 1
        else:
            index += 1
            continue
        for feature in value.replace(",", " ").split():
            features.append(feature if "/" in feature else f"{crate}/{feature}")
    return features


def run_capture(args: list[str], *, cwd: Path) -> str:
    return subprocess.run(
        args,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    ).stdout


_host_triple_cache: dict[Path, str] = {}


def host_triple(root: Path) -> str:
    if value := os.environ.get("LLBC_TARGET_TRIPLE"):
        return value
    cache_key = root.resolve()
    if value := _host_triple_cache.get(cache_key):
        return value
    rustc_info = run_capture(["rustc", "-vV"], cwd=root)
    for line in rustc_info.splitlines():
        if line.startswith("host: "):
            value = line.removeprefix("host: ")
            _host_triple_cache[cache_key] = value
            return value
    raise SystemExit("extract-llbc.py: could not determine rustc host triple")


_metadata_cache: dict[tuple, dict] = {}


def metadata(
    eng: Engine,
    cargo_features: str,
    platform: str,
    extra_features: tuple[str, ...] = (),
) -> dict:
    """`cargo metadata` filtered to `platform`, memoised per full query.

    The platform matters: `--filter-platform` drops dependencies gated on other
    targets, so the host's graph does not see a `[target.'cfg(...)']` path dep
    that only the cross-target layout sidecar pass compiles.
    `extra_features` carries workspace-qualified features a layout sidecar
    pass enables beyond the host pass (`layout_cargo_args`), so the graph
    sees the dependencies they pull in. The memo key spans every input that
    shapes the result — including the engine's workspace root and feature
    crates, which differ between engines sharing this module.
    """
    key = (
        str(eng.root),
        eng.metadata_feature_crates,
        cargo_features,
        platform,
        extra_features,
    )
    if key in _metadata_cache:
        return _metadata_cache[key]

    metadata_features: list[str] = []
    for feature in cargo_features.split(","):
        feature = feature.strip()
        if feature:
            metadata_features.extend(
                f"{crate}/{feature}" for crate in eng.metadata_feature_crates
            )
    metadata_features.extend(extra_features)

    args = [
        "cargo",
        "metadata",
        # `--locked` because this call is reached from `--list-inputs`, which
        # callers run as a read-only membership check before deciding whether
        # a file they are about to edit is a fingerprint input. Without it
        # `cargo metadata` refreshes `Cargo.lock` when the lock is out of
        # date — and `Cargo.lock` is itself one of the inputs this driver
        # fingerprints, so the check would edit the thing it is asked about.
        # The hazard is conditional (a current lock makes it a no-op), which
        # is the worse shape: it is harmless until it is not.
        #
        # A stale lock now fails loudly here instead of being silently
        # updated. That is the intent: refreshing the lock is an action a
        # caller should take deliberately, not one a query performs for them.
        "--locked",
        "--format-version=1",
        "--filter-platform",
        platform,
    ]
    if metadata_features:
        args.extend(["--features", ",".join(metadata_features)])
    result = json.loads(run_capture(args, cwd=eng.root))
    _metadata_cache[key] = result
    return result


_INCLUDE_CALL = re.compile(r"\binclude(?:_str|_bytes)?!\s*\(\s*")
# A plain, escape-free string literal is the only spelling this resolves. A raw
# literal or one carrying a `\` escape falls through to the refusal below rather
# than being decoded wrongly.
_INCLUDE_PATH = re.compile(r"\"([^\"\\]*)\"")


def include_closure(root: Path, files: set[Path]) -> set[Path]:
    """The files `include*!` reads that no crate pathspec covers.

    `include!` / `include_str!` / `include_bytes!` resolve their argument
    against the directory of the file holding the macro, so an argument that
    climbs out of the crate is still a compiler input while sitting outside
    every pathspec the enumeration builds. `majit-metainterp`'s `REAL_RULES`
    is the live instance: it includes PyPy's `real.rules` out of the vendored
    `rpython/` tree, and editing that file changes what the crate compiles
    without moving a digest that never hashed it.

    Only a plain string literal is resolvable from the text, so an argument in
    any other spelling stops extraction instead of being skipped: a skip is
    indistinguishable from a fresh artefact, which is the failure this walk
    exists to remove. The two spellings in the tree today are allowed by shape
    and each is checked below; a third has to be audited the same way before it
    can pass.

    An argument matched inside a comment or a string names either nothing or a
    file the crate does not read; both only widen the digest, which costs a
    re-extraction and never a wrong offset.

    `#[path]` is the other way a crate compiles a file outside its own tree and
    is deliberately not walked: of the four in the workspace, the only pair
    whose `mod` file any crate enumerates redirects into `pyre-jit-trace/src/`,
    which that same crate's closure already covers. Walk it when a redirection
    appears that a pathspec does not reach.

    A refusal raised here reaches whoever runs extraction and whoever runs
    `--fingerprint` from CI, but not `pyre-jit-trace`'s build script: that one
    collapses every non-answer from this driver to silence on purpose
    (`pyre/pyre-jit-trace/build.rs` `llbc_fingerprint_output`).
    """
    extra: set[Path] = set()
    pending = [path for path in files if path.suffix == ".rs"]
    queued = set(pending)
    while pending:
        rel_source = pending.pop()
        source = root / rel_source
        if not source.is_file():
            continue
        text = read_input_bytes(root, rel_source).decode("utf-8", errors="replace")
        for call in _INCLUDE_CALL.finditer(text):
            line = text.count("\n", 0, call.start()) + 1
            where = f"{rel_source.as_posix()}:{line}"
            literal = _INCLUDE_PATH.match(text, call.end())
            if literal is None:
                _reject_unresolvable_include(where, text[call.end() : call.end() + 80])
                continue
            target = (source.parent / literal.group(1)).resolve()
            if not target.is_file():
                continue
            try:
                rel = target.relative_to(root)
            except ValueError:
                raise SystemExit(
                    f"extract-llbc.py: {where} includes {target}, which is "
                    "outside the repository. The digest is keyed by "
                    "repo-relative paths, so that file cannot be hashed and "
                    "the artefact would read as fresh across edits to it."
                ) from None
            if rel in files or rel in extra:
                continue
            extra.add(rel)
            if rel.suffix == ".rs" and rel not in queued:
                queued.add(rel)
                pending.append(rel)
    return extra


def _reject_unresolvable_include(where: str, tail: str) -> None:
    """Let through the spellings whose inputs are covered; refuse the rest."""
    # `include!()` with no argument only occurs in prose about the macro.
    if tail.startswith(")"):
        return
    # `concat!(env!("OUT_DIR"), ...)` names a file the build script generates
    # into the cargo out dir, from sources this enumeration already carries.
    if tail.startswith("concat!") and 'env!("OUT_DIR")' in tail:
        return
    # A `macro_rules!` parameter, resolved at each call site. The one instance
    # is the `appleveldefs:` arm of `pyre_module_init!`, whose call sites pass
    # sibling `app_*.py` names that the crate's own `src/` pathspec enumerates.
    if tail.startswith("$"):
        return
    raise SystemExit(
        f"extract-llbc.py: {where} includes `{tail.splitlines()[0].strip()}`, "
        "which this fingerprint cannot resolve to a path. Whatever it names "
        "would be a compiler input the digest never hashes, so the artefact "
        "would read as fresh across edits to it. Either give it a plain "
        "string literal or add its shape to _reject_unresolvable_include with "
        "the reason its inputs are already enumerated."
    )


def refuse_inert_pathspecs(eng: Engine) -> None:
    """Refuse a declared fingerprint input that cannot contribute anything.

    A pathspec is an input only if git will list a file under it, because
    `_collect_inputs` resolves the in-root channel through `git ls-files` and
    nothing else. A pathspec matching zero files is therefore a silent no-op —
    and a no-op that READS as coverage everywhere a human looks: in the
    driver's list, in the comments that cite it, and in the design decisions
    that rest on it. A driver carried `Cargo.lock` in exactly that state while
    a comment eight lines above called it a declared input and used it to
    justify skipping the more expensive `cargo metadata` walk, so the cheaper
    option was resting on coverage that did not exist.

    This lives in the ENGINE rather than in one driver because every consumer
    of the engine has the same hole: the check was first written against one
    driver's `BASE_PATHSPECS`, and each other driver kept the defect until it
    grew its own copy. Refusing here catches the whole class at the one place
    that resolves pathspecs, and it would have fired the day either known
    offender was added.

    Scope is the DECLARED pathspecs — `base_pathspecs`, and each spec's
    `fingerprint_pathspecs`. The ones `_collect_inputs` discovers from `cargo
    metadata` are facts about the build rather than a human's claim of
    coverage, and refusing on those would fail a driver over a dependency's
    layout it does not control. Every spec is checked, not only the crates
    requested on this run: a pathspec that rots in a spec nobody asked for
    today is then found on the next run, rather than the next time that one
    crate happens to be extracted.

    Git is the authority for this check. `Path.exists()` below feeds the
    message and nothing else; do not replace the git query with it. An
    ignored file EXISTS on disk and is invisible to `git ls-files`, which is
    the entire defect. Filesystem-reachable and git-reachable are different
    predicates, and a check that conflates them answers a question nobody
    asked. (The defect is not `exists()`; it is using `exists()` to answer a
    question about git. An `is_file()` test on a module about to be imported is
    a genuine filesystem question and stays correct.)

    Drivers may also declare `external_inputs`. Those inputs are hashed by
    reading their bytes rather than through git, so filesystem existence is
    the correct predicate for that separate channel. The authority must match
    the channel that carries the input.

    `ls-files ∪ ls-files --others --exclude-standard` is the definition of
    git-reachable used here, matching `_collect_inputs` exactly. `ls-files
    --error-unmatch` alone is TRACKED-only, and would call a brand-new unadded
    file inert when the engine does hash it.

    The two failure modes need different remedies, so they are reported apart:
    a path that EXISTS but is ignored is a coverage hole, a path that does not
    exist is a typo or a file that moved.
    """

    def git_lists(*args: str) -> bool:
        result = subprocess.run(
            ["git", "-C", str(eng.root), *args],
            check=True,
            stdout=subprocess.PIPE,
        )
        return bool(result.stdout.strip())

    declared: list[tuple[str, list[str]]] = [("base_pathspecs", eng.base_pathspecs)]
    for name, spec in eng.specs.items():
        if spec.fingerprint_pathspecs is not None:
            declared.append(
                (f"specs[{name!r}].fingerprint_pathspecs", spec.fingerprint_pathspecs)
            )

    inert: list[tuple[str, str, bool]] = []
    for label, pathspecs in declared:
        for pathspec in pathspecs:
            # `--others` only when the tracked query came back empty: the live
            # case is one subprocess per pathspec instead of two, and this runs
            # on every invocation including the `--check` a build script makes.
            if git_lists("ls-files", "--", pathspec):
                continue
            if git_lists("ls-files", "--others", "--exclude-standard", "--", pathspec):
                continue
            inert.append((label, pathspec, (eng.root / pathspec).exists()))
    if not inert:
        return

    lines = ["extract-llbc.py: declared fingerprint pathspecs that match no file:"]
    for label, pathspec, on_disk in inert:
        if on_disk:
            lines.append(
                f"  {label}: {pathspec!r} — EXISTS on disk but git will not list"
                f" it (ignored?), so it contributes nothing to the fingerprint."
                f" Either stop declaring it or cover it another way — an"
                f" out-of-root or ignored file goes in `external_inputs=`,"
                f" which is hashed by content; do not leave it here reading as"
                f" coverage."
            )
        else:
            lines.append(
                f"  {label}: {pathspec!r} — does not exist. Renamed, moved, or a"
                f" typo: whatever it was meant to fingerprint is now"
                f" unfingerprinted."
            )
    raise SystemExit("\n".join(lines))


def _package_closure(
    target_ids: list[str],
    by_id: dict,
    resolve_nodes: dict,
) -> list[dict]:
    """Path-dependency packages reachable from `target_ids`.

    Pure graph reachability over cargo-metadata shapes: `by_id` maps package id
    to package, and `resolve_nodes` maps package id to its resolve node. Kept
    separate from `_collect_inputs` so its graph semantics stay testable
    without invoking Cargo.
    """
    seen: set[str] = set()
    stack = list(target_ids)
    closure = []
    while stack:
        package_id = stack.pop()
        if package_id in seen:
            continue
        seen.add(package_id)
        package = by_id[package_id]
        closure.append(package)

        for dep in resolve_nodes.get(package_id, {}).get("deps", []):
            dep_kinds = dep.get("dep_kinds", [])
            # An empty `dep_kinds` is a normal (non-dev) edge; only drop deps
            # whose every listed kind is `dev`.
            if dep_kinds and all(kind.get("kind") == "dev" for kind in dep_kinds):
                continue
            dep_package = by_id.get(dep["pkg"])
            if dep_package is not None and dep_package.get("source") is None:
                stack.append(dep_package["id"])
    return closure


def nested_repo_entries(entries: list[str]) -> list[str]:
    """`--others` entries git reported as directories instead of descending.

    `git ls-files --others` walks an ordinary untracked directory and lists the
    FILES in it. At a directory holding its own `.git` it stops, and emits the
    DIRECTORY, with a trailing slash. So a trailing slash in that output is
    git's own statement that it did not look inside.

    Keyed on the slash rather than on an `is_dir()` probe because the slash is
    the producer's record of what it did, where a filesystem call re-derives a
    guess about it — and the two can disagree while a build runs.

    Measured, git 2.50.1, all under one covered pathspec: a nested repo with
    commits and one freshly `git init`ed both carry the slash; an ordinary
    untracked directory of sources, a directory whose files are all ignored, an
    empty directory, and an ignored nested repo produce no directory entry at
    all. The benign shapes are the ones that matter here — this refusal is
    fail-closed, so a predicate that flagged any new module directory would
    stop every extraction in the repository.
    """
    return sorted(entry for entry in entries if entry.endswith("/"))


def refuse_nested_repos(entries: list[str]) -> None:
    """Refuse when a nested repository sits under a fingerprinted pathspec.

    Not a conservative over-hash like the untracked leg above it: this is the
    one shape where the fingerprint is WRONG rather than merely wide. The
    directory entry enters the input list, `source_fingerprint` finds it is not
    a file, and it hashes through the `<deleted>` branch — one constant token.
    Editing, adding, or deleting files inside such a repository leaves the
    digest byte-identical. The stamp would then claim to cover a path whose
    contents cannot move it, and that digest is also the CI cache key.

    Refusing rather than recursing is the direction the untracked leg's own
    argument points: it prefers a cost that is always a re-extraction over an
    answer that is sometimes wrong. Recursing would mean deciding what a nested
    repository's commit means for a source hash, which is a larger question
    than this guard needs to answer.
    """
    nested = nested_repo_entries(entries)
    if not nested:
        return
    # Each parenthesised pair is ONE message line: without the parentheses a
    # dropped comma reads exactly like the wrapping, and would silently join two
    # lines of the refusal into one.
    lines = [
        (
            "extract-llbc.py: a nested git repository sits under a fingerprinted"
            " pathspec:"
        ),
        *(f"    {entry}" for entry in nested),
        "",
        (
            "`git ls-files --others` does not descend into one, so every file"
            " inside is"
        ),
        (
            "absent from `closure=` while the directory itself hashes to a single"
            " constant"
        ),
        (
            "token — the digest does not move when those contents change, grow, or"
            " are"
        ),
        "deleted outright. The stamp would claim a coverage it does not have.",
        "",
        (
            "Move it outside the fingerprinted pathspecs, or add it to"
            " `.gitignore`"
        ),
        "(an ignored nested repo is excluded here and is not compiled either).",
        (
            "Registering it as a SUBMODULE does not fix this: a gitlink arrives on"
            " the"
        ),
        (
            "tracked leg, carries no trailing slash, and hashes through the same"
            " branch."
        ),
    ]
    raise SystemExit("\n".join(lines))


_inputs_cache: dict[tuple, tuple[list[Path], list[Path]]] = {}


_FILES_KEY = b'"files":['


def ullbc_files_table(path: Path, chunk_size: int = 1 << 22) -> list[dict]:
    """Stream the `translated.files` array out of a large one-line artefact."""
    buffer = b""
    start: int | None = None
    scan = 0
    depth = 0
    in_string = False
    escaped = False
    with path.open("rb") as artefact:
        while True:
            piece = artefact.read(chunk_size)
            if not piece:
                raise SystemExit(f"extract-llbc.py: {path} has no translated.files table")
            buffer += piece
            if start is None:
                index = buffer.find(_FILES_KEY)
                if index < 0:
                    buffer = buffer[-(len(_FILES_KEY) - 1) :]
                    continue
                start = index + len(_FILES_KEY) - 1
                scan = start
            while scan < len(buffer):
                byte = buffer[scan]
                if in_string:
                    if escaped:
                        escaped = False
                    elif byte == ord("\\"):
                        escaped = True
                    elif byte == ord('"'):
                        in_string = False
                elif byte == ord('"'):
                    in_string = True
                elif byte in (ord("["), ord("{")):
                    depth += 1
                elif byte in (ord("]"), ord("}")):
                    depth -= 1
                    if depth == 0:
                        try:
                            table = json.loads(buffer[start : scan + 1])
                        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                            raise SystemExit(
                                f"extract-llbc.py: cannot parse {path}'s files table: {exc}"
                            ) from exc
                        if not isinstance(table, list):
                            raise SystemExit(
                                f"extract-llbc.py: {path}'s translated.files is not a list"
                            )
                        return table
                scan += 1


def is_repo_relative_name(value: str) -> bool:
    """Whether an artefact file-table name can be resolved against the root.

    A Windows DRIVE disqualifies a name even with no root after it. `C:Users/x`
    is drive-relative, so neither `PurePosixPath.is_absolute` nor
    `PureWindowsPath.is_absolute` reports it absolute, while `root / value`
    still DISCARDS `root` and resolves against that drive's working directory.
    Charon spells the toolchain's own sources that way on Windows, and the
    resolved file exists, so the walk read rustc's `std/src/lib.rs` and refused
    the `include_str!` reaching out of it — after two absolute-path checks had
    both passed the name through.
    """
    if not value:
        return False
    if PurePosixPath(value).is_absolute():
        return False
    windows = PureWindowsPath(value)
    return not windows.is_absolute() and not windows.drive


def normalized_local_name(value: str) -> PurePosixPath:
    """A file-table name as a normalised repo-relative POSIX path.

    Charon separates with backslashes on Windows, and `posixpath` does not read
    those as separators: the whole spelling arrives as ONE component, so a
    `../` chain survives normalisation and the name never equals the
    forward-slash path `git ls-files` reports for the same file. The live
    instance is `_immutables_map/mod.rs`, whose `include_str!` climbs five
    levels to `lib_pypy/_immutables_map.py`; unnormalised, it failed the
    coverage assertion on the windows leg alone. `PureWindowsPath` reads both
    separators, so canonicalise through it first. Nothing else changes shape —
    no file in this repository carries a backslash in its name.
    """
    return PurePosixPath(posixpath.normpath(PureWindowsPath(value).as_posix()))


def artefact_local_readfiles(path: Path) -> dict[str, set[Path]]:
    """Repo-relative `Local` files in an artefact, grouped by crate name."""
    by_crate: dict[str, set[Path]] = {}
    for entry in ullbc_files_table(path):
        name = entry.get("name")
        if not isinstance(name, dict) or "Local" not in name:
            continue
        value = name["Local"]
        if not isinstance(value, str):
            raise SystemExit(f"extract-llbc.py: {path} has a non-string Local filename")
        if not is_repo_relative_name(value):
            continue
        normal = normalized_local_name(value)
        if normal == PurePosixPath("..") or normal.parts[:1] == ("..",):
            raise SystemExit(
                f"extract-llbc.py: {path} has a Local filename outside the repository: {value}"
            )
        crate_name = entry.get("crate_name")
        if not isinstance(crate_name, str) or not crate_name:
            raise SystemExit(
                f"extract-llbc.py: {path} has a repo-local file with no crate_name: {value}"
            )
        by_crate.setdefault(crate_name, set()).add(Path(*normal.parts))
    return by_crate


def readfiles_path(eng: Engine, crate: str) -> Path:
    """Sidecar carrying the artefact-declared repository read set."""
    dest = Path(os.environ.get("LLBC_DEST", eng.out_dir))
    if not dest.is_absolute():
        dest = eng.root / dest
    output = dest / eng.spec(crate).output_name
    return output.with_suffix(output.suffix + ".readfiles")


def load_readfiles(
    eng: Engine, crate: str, path: Path | None = None
) -> set[Path] | None:
    """Read a persisted artefact read set, or request the wide fallback.

    `path` overrides the worktree sidecar. Store import hashes this tree
    against another checkout's recorded read set, so the walk must not
    silently pick up whatever `.readfiles` sits beside this worktree's
    artefact.
    """
    path = readfiles_path(eng, crate) if path is None else path
    if not path.is_file():
        return None
    result: set[Path] = set()
    for value in path.read_text(encoding="utf-8").splitlines():
        normal = normalized_local_name(value)
        escapes = normal == PurePosixPath("..") or normal.parts[:1] == ("..",)
        if not is_repo_relative_name(value) or escapes:
            raise SystemExit(f"extract-llbc.py: invalid path in {path}: {value!r}")
        result.add(Path(*normal.parts))
    return result


# File contents read while hashing and while scanning for `include*!`
# arguments, keyed by repo-relative path and validated against the file's
# stat on every hit, so a caller that rewrites an input between two walks
# (the selftest does, per case) reads the new bytes without having to know
# about this memo. Size and mtime alone would pass a metadata-preserving
# rewrite (`cp -p` of a same-sized file), so the inode number and the ctime
# join them: a replacement changes the inode, and an in-place write moves
# ctime, which no file API sets backwards. `forget_collected_inputs` drops
# the memo with the input walks it serves. One `--fingerprint --per-crate`
# over the four pyre crates hashes the interpreter's closure three times
# over (the `source=` fallback, `closure=`, and the include scan) and the
# jit's on top, which on a windows runner is the cost of the step — every
# read there goes through the real-time scanner, where a stat does not.
_file_bytes_cache: dict[Path, tuple[tuple[int, int, int, int], bytes]] = {}


def read_input_bytes(root: Path, path: Path) -> bytes:
    """`(root / path).read_bytes()`, memoised while the file's stat holds."""
    stat = (root / path).stat()
    seen = (stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_ctime_ns)
    hit = _file_bytes_cache.get(path)
    if hit is not None and hit[0] == seen:
        return hit[1]
    data = (root / path).read_bytes()
    _file_bytes_cache[path] = (seen, data)
    return data


def forget_collected_inputs() -> None:
    """Drop the memoised input sets, because the tree they describe has moved.

    The memo below answers "what does this crate compile" for a tree as it was
    when the walk ran, so anything in this process that CHANGES the tree has to
    call this before the next walk reads it. Two do, and each would otherwise
    get a stale answer at exactly the moment it needs a fresh one: `extract`,
    whose whole post-build re-hash exists to catch a tree that moved while
    Charon ran, and `self_test`, whose probe file exists precisely to move the
    input set.
    """
    _inputs_cache.clear()
    _file_bytes_cache.clear()


def _collect_inputs(
    eng: Engine,
    crates: list[str],
    cargo_features: str,
    *,
    derived: bool = False,
    readfiles_override: Path | None = None,
) -> tuple[list[Path], list[Path]]:
    """Split the fingerprint's inputs by which channel can carry them.

    Returns `(repo-relative files, external inputs)`:

    * repo-relative files — the artefact-derived set for `source=`, or the
      conservative cargo closure for `closure=` and bootstrap fallback.
    * `external` — absolute paths the git channel cannot carry: anything
      outside `eng.root`, which `git ls-files` refuses to name at all, plus
      whatever a driver declares in `external_inputs` (which is also how an
      in-root but IGNORED file gets covered). Hashed by content instead, by
      `external_fingerprint`.

    The second list used to be dropped on the floor. The dependency walk below
    admits path deps by `source is None`, which is how cargo spells BOTH an
    in-tree workspace member and a `[patch]` redirect into a sibling checkout —
    so a patched dependency was discovered, found to be out-of-root, and
    discarded with no diagnostic. An out-of-tree consumer that patches a
    dependency to a sibling workspace has this shape, including proc-macro
    crates whose expansions become part of the extracted item bodies.

    `majit-macros` is a proc macro, so its expansion IS the extracted crate's
    item bodies. It was dropped by TWO independent gates, and finding the
    first hid the second: this out-of-root discard, and the target-kind filter
    below, which admitted neither `proc-macro` nor `cdylib` until it was
    inverted into a deny-list. Both are fixed; do not read that as the class
    being closed, since each was found only after the previous one was.

    Memoised, because none of this is cheap: two `git ls-files` subprocesses per
    call plus `include_closure`, which reads the text of every `.rs` input. One
    `extract` of one crate reaches here five times — twice through `stamp_for`,
    then the provenance closure, `window_writes` and the post-build digest — and
    `--fingerprint` twice, under a caller-imposed budget (120s in
    `pyre-jit-trace/build.rs`) past which freshness degrades to UNKNOWN for
    every crate. The key spells out the engine fields this walk reads, except
    the SPEC TABLE, which the crate names stand for: `run_cli` builds one
    Engine from a driver's specs and nothing edits them afterwards.
    `forget_collected_inputs` is how a caller that moved the tree says so.
    The optional `readfiles_override` is part of the key so a store import
    that hashes against another entry's read set cannot reuse a walk that
    used this worktree's sidecar.
    """
    key = (
        str(eng.root),
        eng.metadata_feature_crates,
        tuple(eng.base_pathspecs),
        eng.external_inputs,
        eng.layout_targets,
        tuple(crates),
        cargo_features,
        derived,
        None if readfiles_override is None else str(readfiles_override),
    )
    if key in _inputs_cache:
        return _inputs_cache[key]

    root = eng.root
    target_names: list[str] = []
    # The derived set names its two manifests outright rather than reading
    # `base_pathspecs`, which today holds exactly those two. The base set is the
    # conservative one: a caller widening it means "always hash this", which the
    # wide channel already honours, and inheriting it here would let one broad
    # entry undo the narrowing this whole path exists to produce.
    pathspecs = ["Cargo.lock", "Cargo.toml"] if derived else list(eng.base_pathspecs)
    direct_files: set[Path] = set()
    external: list[Path] = [Path(p).resolve() for p in eng.external_inputs]

    for crate in crates:
        spec = eng.spec(crate)
        if derived:
            readfiles = load_readfiles(eng, crate, path=readfiles_override)
            if readfiles is None:
                return _collect_inputs(eng, crates, cargo_features, derived=False)
            direct_files.update(readfiles)
            if spec.fingerprint_pathspecs is None:
                target_names.append(crate)
            else:
                manifest = (spec.crate_dir / "Cargo.toml").resolve()
                if manifest.is_relative_to(root):
                    pathspecs.append(manifest.relative_to(root).as_posix())
                else:
                    external.append(manifest)
        elif spec.fingerprint_pathspecs is not None:
            pathspecs.extend(spec.fingerprint_pathspecs)
        else:
            target_names.append(crate)

    if target_names:
        # The stamp covers the host artefact AND every layout sidecar, so the
        # fingerprint must hash the union of the closures those passes compile:
        # a path dep gated on a cross target alone would otherwise change
        # without invalidating the sidecar it shaped.
        platforms = {host_triple(root)}
        for name in target_names:
            platforms.update(crate_layout_targets(eng, eng.spec(name)))

        # A layout sidecar pass may enable features the host pass does not
        # (`layout_cargo_args`); fold them into every metadata query so the
        # union closure sees the dependencies those features pull in.
        layout_features: set[str] = set()
        for name in target_names:
            spec = eng.spec(name)
            if spec.layout_cargo_args is not None:
                layout_features.update(
                    features_in_cargo_flags(
                        name, crate_layout_flags(spec, cargo_features, [])
                    )
                )
        extra_features = tuple(sorted(layout_features))

        for platform in sorted(platforms):
            meta = metadata(eng, cargo_features, platform, extra_features)
            packages = meta["packages"]
            by_name = {package["name"]: package for package in packages}
            by_id = {package["id"]: package for package in packages}
            resolve_nodes = {
                node["id"]: node for node in meta.get("resolve", {}).get("nodes", [])
            }

            missing = [name for name in target_names if name not in by_name]
            if missing:
                raise SystemExit(
                    "extract-llbc.py: unknown crate(s): " + ", ".join(sorted(missing))
                )

            closure = _package_closure(
                [by_name[name]["id"] for name in target_names],
                by_id,
                resolve_nodes,
            )

            for package in closure:
                package_dir = Path(package["manifest_path"]).resolve().parent
                if package_dir.is_relative_to(root):
                    rel_dir = package_dir.relative_to(root).as_posix()
                    pathspecs.append(f"{rel_dir}/Cargo.toml")
                else:
                    external.append(package_dir / "Cargo.toml")
                for target in package["targets"]:
                    kinds = set(target["kind"])
                    # This is a deny-list so new library-like target kinds are
                    # admitted by default. The old allow-list omitted
                    # `proc-macro` and `cdylib`; proc-macro expansions are
                    # inlined into consuming crates, so their sources must be
                    # fingerprinted.
                    #
                    # The package loop above appends `Cargo.toml` BEFORE this
                    # loop runs, which is what made the omission invisible: the
                    # dropped crate still showed up in `--list-inputs`, so it
                    # read as covered while none of its sources were hashed.
                    #
                    # Naming what is NOT compiled into the artefact is a short,
                    # closed list; naming what IS grows silently with every
                    # crate type cargo adds. Getting this wrong now costs a
                    # re-extraction, never a wrong answer — the same direction
                    # the untracked-file leg below is deliberately biased in.
                    if kinds & {"example", "test", "bench"}:
                        continue
                    if derived and not kinds & {"proc-macro", "custom-build"}:
                        continue
                    src_path = Path(target["src_path"]).resolve()
                    if src_path.is_relative_to(root):
                        rel_src = src_path.relative_to(root).as_posix()
                        pathspecs.append(rel_src)
                        if "custom-build" not in kinds:
                            pathspecs.append(str(Path(rel_src).parent) + "/")
                    else:
                        # Same shape as the in-root arm, one channel over: the
                        # entry point always, its directory only for a real
                        # target. A `custom-build`'s src_path is the crate's
                        # own `build.rs`, so its parent is the CRATE ROOT —
                        # walking that would pull in `target/` and every
                        # sibling target's sources.
                        external.append(src_path)
                        if "custom-build" not in kinds:
                            external.append(src_path.parent)

    def ls_files_raw(*extra_flags: str) -> list[str]:
        result = subprocess.run(
            ["git", "ls-files", "-z", *extra_flags, "--", *pathspecs],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
        )
        return [raw.decode("utf-8") for raw in result.stdout.split(b"\0") if raw]

    def ls_files(*extra_flags: str) -> set[Path]:
        # `Path()` normalises the trailing slash away, so a caller that needs to
        # tell a directory entry from a file entry must read `ls_files_raw`.
        # That erasure is why the nested-repo blindness was invisible here.
        return {Path(entry) for entry in ls_files_raw(*extra_flags)}

    # Charon reads the working tree, so include untracked, non-ignored files.
    # Keep the raw spelling until nested repositories have been rejected:
    # converting a directory entry to `Path` erases its trailing slash.
    #
    others = ls_files_raw("--others", "--exclude-standard")
    refuse_nested_repos(others)
    files = ls_files() | {Path(entry) for entry in others} | direct_files
    # Both modes. Charon does record an `include_str!` target that sits inside
    # the crate — `_io_app.py` and its siblings arrive through `direct_files`
    # like any source — so the derived set is not blind to those. What it is
    # blind to is an argument that climbs OUT of the crate: `pyre-native`'s
    # `include_bytes!("../../../lib-python/3/test/certdata/keycert3.pem")`
    # reaches a file the artefact's file table never names and no pathspec
    # covers, and dropping this leg left editing it invisible to `source=`.
    # That is the same escape the walk was written for, one input set along.
    files |= include_closure(root, files)
    result = (sorted(files, key=lambda path: path.as_posix()), external)
    _inputs_cache[key] = result
    return result


def fingerprint_inputs(
    eng: Engine,
    crates: list[str],
    cargo_features: str,
    *,
    readfiles_override: Path | None = None,
) -> list[Path]:
    """Artefact-derived repo-relative inputs hashed into `source=`."""
    combined: set[Path] = set()
    for crate in crates:
        combined.update(
            _collect_inputs(
                eng,
                [crate],
                cargo_features,
                derived=True,
                readfiles_override=readfiles_override,
            )[0]
        )
    return sorted(combined, key=lambda path: path.as_posix())


def closure_inputs(eng: Engine, crates: list[str], cargo_features: str) -> list[Path]:
    """Conservative cargo-closure inputs hashed into `closure=`."""
    return _collect_inputs(eng, crates, cargo_features)[0]


def external_input_groups(
    eng: Engine, crates: list[str], cargo_features: str
) -> list[tuple[str, list[tuple[str, Path]]]]:
    """Out-of-root inputs GROUPED by the root that declared or discovered them.

    `[(root label, [(file label, absolute path), …]), …]`, sorted by root label
    and, within a root, by file label.

    Deliberately NOT named `external_inputs`: that is the driver-facing keyword
    of `run_cli`, and a parameter of that name shadows the module-level
    function for the whole body.

    The group, not the file and not the whole set, is the unit `external=`
    records a digest for, and that choice is the only reason `check` can name a
    mover. One digest over every out-of-root file fires identically whether a
    doc comment moved in a crate the artefact barely references or
    `majit-macros` did — a proc macro whose expansion IS the extracted crate's
    item bodies, so its movement voids every measurement already read out of
    the artefact. Same remedy (`--force`), very different consequence, and a
    single digest cannot tell them apart. Per-file reporting is the other
    extreme and becomes unreadable for a large declared closure.

    Roots are deduped by label — the metadata walk runs once per layout
    platform and rediscovers the same packages — but files are NOT deduped
    across roots. `<crate>/src/lib.rs` and `<crate>/src/` are both roots the
    walk emits, so an edit to `lib.rs` moves both, and that pair of movers is
    the narrowing a reader wants rather than noise.

    The label — not the absolute path — is what enters the digest, so two
    checkouts of the same layout in different directories agree. Labels are
    relative to the artefact root, so a stamp survives relocating a group of
    sibling repositories as a unit. It does not survive rearranging them
    relative to each other, which is correct because that changes which
    sources are compiled.

    Directories are expanded here rather than at collection time so the walk
    sees the working tree at hashing time, matching `source=`'s
    save-not-commit behaviour for the in-root half.
    """
    _, roots = _collect_inputs(eng, crates, cargo_features)

    def label_for(path: Path) -> str:
        return Path(os.path.relpath(path, eng.root)).as_posix()

    groups: dict[str, list[tuple[str, Path]]] = {}
    for entry in roots:
        root_label = label_for(entry)
        if root_label in groups:
            continue
        if entry.is_dir():
            found: dict[str, Path] = {}
            for sub in entry.rglob("*"):
                # `target/` is build output and dot-dirs are VCS/tooling state;
                # neither is a compiler input, and `target/` alone is tens of GB.
                parts = set(sub.relative_to(entry).parts[:-1])
                if "target" in parts or any(p.startswith(".") for p in parts):
                    continue
                if sub.is_file():
                    found[label_for(sub)] = sub
            groups[root_label] = sorted(found.items())
        else:
            # A file, or a declared root that does not exist. The absent case
            # still gets a group, so it moves the digest when it appears —
            # exactly as a deleted tracked file does in `source=`.
            groups[root_label] = [(root_label, entry)]
    return sorted(groups.items())


def external_group_digest(root_label: str, files: list[tuple[str, Path]]) -> str:
    """Digest of one out-of-root group.

    The root label is folded in first so an EMPTY group (a declared directory
    holding no file the filter admits) still gets a digest of its own rather
    than colliding with every other empty group.
    """
    digest = hashlib.sha256()
    digest.update(root_label.encode("utf-8"))
    digest.update(b"\0")
    for label, path in files:
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes() if path.is_file() else b"<absent>")
        digest.update(b"\0")
    return digest.hexdigest()


def external_fingerprint(eng: Engine, crates: list[str], cargo_features: str) -> str:
    """`external=`'s stamp value: one `<root label>=<digest>` per group.

    Empty is the common case (a consumer whose whole dependency closure lives
    inside its own repo, which is every pyre crate today) and it is spelled as
    an empty value rather than a digest-of-nothing, so the stamp says plainly
    that nothing outside the repo was folded in.

    `shlex.quote` per entry, space-separated: a root label is a filesystem path
    and may hold a space or a quote, and the value has to survive as ONE line
    of a line-oriented stamp. `parse_packed` is the inverse.

    One stamp KEY, not one key per root. `STAMP_KEYS` is a fixed schema and
    `check` refuses a stamp missing any member of it; deriving the key set from
    the very declaration whose movement is under test would mean a dropped
    input takes its own coverage assertion with it, and "the field is gone"
    would read the same as "there was never such a field".
    """
    groups = external_input_groups(eng, crates, cargo_features)
    return " ".join(
        shlex.quote(f"{root_label}={external_group_digest(root_label, files)}")
        for root_label, files in groups
    )


def parse_packed(value: str) -> dict[str, str]:
    """Split a packed `<label>=<value>` stamp field into its entries.

    Two fields pack a list into one line this way, `external=` and
    `artefacts=`; this is the inverse of both encodings. `rpartition`, not
    `partition`: a label may contain `=`, and the value that follows never
    does.
    """
    entries: dict[str, str] = {}
    for token in shlex.split(value):
        label, sep, digest = token.rpartition("=")
        if sep:
            entries[label] = digest
    return entries


def external_diff(crate: str, recorded: str, expected: str) -> list[str]:
    """Name WHICH out-of-root input moved, rather than printing two digests.

    `external=` carries one digest per root precisely so this can be written.
    With a single digest there is nothing to say here beyond "something outside
    the repo changed" — a guard that fires without informing, which is the
    shape this field is structured to avoid.
    """
    was = parse_packed(recorded)
    now = parse_packed(expected)
    lines: list[str] = []
    for label in sorted(set(was) | set(now)):
        if label not in now:
            lines.append(
                f"{crate}: external: {label!r} is no longer a declared input, "
                f"but the artefact was built with it folded in"
            )
        elif label not in was:
            lines.append(
                f"{crate}: external: {label!r} is an input the artefact never "
                f"covered"
            )
        elif was[label] != now[label]:
            lines.append(f"{crate}: external: {label!r} changed")
    if not lines:
        # The packed values differ while every root agrees: an encoding this
        # engine did not write, or a reordering. Reporting nothing here would
        # turn a mismatch into a silent pass.
        lines.append(
            f"{crate}: external: {recorded!r} does not match {expected!r} "
            f"although every root agrees — the value was not written by this "
            f"engine"
        )
    return lines


def artefact_token(path: Path) -> str:
    """One artefact's contribution to `artefacts=`: its size and its digest.

    `missing` rather than a raise. Both sides of the comparison run this, and
    the side reading the tree has to be able to say "the file the stamp
    describes is not there" instead of ending the run.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return f"{size}:{digest.hexdigest()}"


def describe_artefact(token: str) -> str:
    """Render one `artefacts=` token for a reader.

    The size is spelled out and the digest abbreviated: a digest pair says
    only "these differ", where a size pair usually says how.
    """
    if token == "missing":
        return "not there"
    size, sep, digest = token.partition(":")
    if not sep:
        return f"{token!r}, which is not a token this engine writes"
    return f"{size} bytes, sha256 {digest[:12]}"


def artefact_diff(crate: str, recorded: str, expected: str) -> list[str]:
    """Name WHICH artefact moved, rather than printing two packed lists."""
    was = parse_packed(recorded)
    now = parse_packed(expected)
    lines: list[str] = []
    for name in sorted(set(was) | set(now)):
        if name not in now:
            lines.append(
                f"{crate}: artefacts: {name!r} is no longer one of this "
                f"crate's artefacts, but the stamp was written over it"
            )
        elif name not in was:
            lines.append(
                f"{crate}: artefacts: {name!r} is an artefact the stamp never "
                f"covered"
            )
        elif was[name] != now[name]:
            lines.append(
                f"{crate}: artefacts: {name}: stamped "
                f"{describe_artefact(was[name])}, on disk "
                f"{describe_artefact(now[name])}"
            )
    if not lines:
        # The packed values differ while every artefact agrees: an encoding
        # this engine did not write. Reporting nothing here would turn a
        # mismatch into a silent pass.
        lines.append(
            f"{crate}: artefacts: {recorded!r} does not match {expected!r} "
            f"although every artefact agrees — the value was not written by "
            f"this engine"
        )
    return lines


def artefacts_fingerprint(eng: Engine, spec: CrateSpec, dest_dir: Path) -> str:
    """`artefacts=`'s stamp value: the bytes this stamp was written beside.

    Every other field describes an INPUT, and together they answer "would a
    fresh extraction write different bytes here". None of them answers "are
    these still the bytes that extraction wrote". Charon writes its artefact
    straight to its final path and `write_layout_sidecar` writes the reduced
    sidecar the same way, so a run killed part-way through either leaves a
    short file behind — while the previous run's stamp, describing sources
    that have not moved, stays on disk and goes on reading current. The
    0-byte test in `check` catches only the kill that landed before the first
    byte.

    Cost, since `check` recomputes this on every run and a build script runs
    one: 0.5 s of CPU, 0.7 s of wall clock, over the 980 MB the four pyre
    artefacts and their sidecars occupy, against the 2.6 s of CPU a check
    costs without it. Measured with the artefacts in the page cache; a cold
    read adds its own time.
    """
    names = [spec.output_name]
    names += [spec.layout_sidecar_name(t) for t in crate_layout_targets(eng, spec)]
    # `shlex.quote` per entry for the same reason `external=` does it: the
    # value has to survive as ONE line of a line-oriented stamp, and
    # `parse_packed` splits it back with `shlex.split`.
    return " ".join(
        shlex.quote(f"{name}={artefact_token(dest_dir / name)}") for name in names
    )


# The two fields that pack a list into one value. Printing either as an opaque
# string would hand the reader kilobytes and no answer, so each has a renderer
# that names the entry that moved.
PACKED_FIELD_DIFFS = {"external": external_diff, "artefacts": artefact_diff}


def _repo_fingerprint(
    eng: Engine,
    crates: list[str],
    cargo_features: str,
    inputs: list[Path],
) -> str:
    """Hash extraction configuration and one repository input set."""
    digest = hashlib.sha256()
    # Keep the cache/staleness key tied to extraction semantics rather than to
    # the implementation bytes of this driver. The driver-provided ABI covers
    # repo-specific extraction behaviour; the remaining fields automatically
    # cover the effective per-crate flags and target-layout configuration.
    config_fields = [
        ("fingerprint_schema", FINGERPRINT_SCHEMA),
        ("extraction_abi", eng.extraction_abi),
        ("cargo_features", cargo_features),
        ("layout_target_rustflags", eng.layout_target_rustflags),
    ]
    for crate in sorted(crates):
        spec = eng.spec(crate)
        flags = crate_flags(spec, cargo_features)
        config_fields.extend(
            [
                (f"{crate}.cargo_flags", "\x1f".join(flags)),
                (
                    f"{crate}.charon_flags",
                    "\x1f".join(charon_crate_flags(spec, cargo_features)),
                ),
                (
                    f"{crate}.layout_targets",
                    "\x1f".join(crate_layout_targets(eng, spec)),
                ),
                (
                    f"{crate}.layout_flags",
                    "\x1f".join(crate_layout_flags(spec, cargo_features, flags)),
                ),
            ]
        )
    for key, value in config_fields:
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    for path in inputs:
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        full_path = eng.root / path
        if full_path.is_file():
            digest.update(read_input_bytes(eng.root, path))
        elif full_path.exists():
            # A directory in the input list is never a source file. Hashing it
            # below would fold it to one constant token that no edit inside it
            # can move — the nested-repo defect, and equally a registered
            # submodule, which `refuse_nested_repos` cannot see because a
            # gitlink arrives tracked and unslashed.
            raise SystemExit(
                "extract-llbc.py: a fingerprint input is a DIRECTORY, not a"
                f" file:\n    {path.as_posix()}\n"
                "Hashing it would record one constant token for the whole"
                " subtree.\nIf it is a submodule or a nested repository, move"
                " it out of the\nfingerprinted pathspecs or ignore it."
            )
        else:
            # The `--cached` half of the enumeration includes tracked paths
            # deleted in the working tree. A deletion is part of the source
            # state and must change the fingerprint instead of making
            # extraction unusable until commit.
            digest.update(b"<deleted>")
        digest.update(b"\0")
    return digest.hexdigest()


def source_fingerprint(
    eng: Engine,
    crates: list[str],
    cargo_features: str,
    *,
    readfiles_override: Path | None = None,
) -> str:
    """Hash the artefact-declared read set and mandatory generated-code inputs."""
    return _repo_fingerprint(
        eng,
        crates,
        cargo_features,
        fingerprint_inputs(
            eng, crates, cargo_features, readfiles_override=readfiles_override
        ),
    )


def closure_fingerprint(eng: Engine, crates: list[str], cargo_features: str) -> str:
    """Hash the former wide cargo-closure input set without changing its meaning."""
    return _repo_fingerprint(
        eng, crates, cargo_features, closure_inputs(eng, crates, cargo_features)
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


def shared_build_dir(root: Path) -> Path:
    """Machine-wide cache root: Charon install, its cargo target, the LLBC store.

    `PYRE_SHARED_BUILD` relocates the whole tree; otherwise it sits beside the
    checkout so sibling worktrees share one store. Callers that already
    specialised this (an explicit `CHARON_TARGET_DIR`, an explicit
    `LLBC_STORE`) still win over the default.
    """
    return Path(os.environ.get("PYRE_SHARED_BUILD", root.parent / ".pyre-build"))


def charon_paths(charon_root: Path) -> tuple[str, Path, Path]:
    platform_key, charon_exe = platform_info()
    shared = shared_build_dir(charon_root)
    pin = install_charon()
    version = os.environ.get("CHARON_VERSION", pin.CHARON_VERSION_DEFAULT)
    charon_dest = Path(
        os.environ.get(
            "CHARON_DEST", pin.default_charon_dest(shared, platform_key, version)
        )
    )
    return platform_key, charon_dest, charon_dest / charon_exe


def charon_cargo_target_dir(root: Path) -> Path:
    """Cargo target dir for Charon's nightly rustc, not the workspace `target/`.

    Charon's rustc is a different compiler from the stable build. Sharing
    `target/` with it either rebuilds the whole graph (metadata rustc hash)
    or, with incremental on, fuses CGUs from the two sessions. A dedicated
    directory lets CI cache the nightly deps without bloating rust-cache.
    Peer worktrees must not share one cache: two checkouts compiling the
    same crate at different revisions produce phantom E0425 errors against
    a mixed incremental store, so the default is a subdirectory keyed by
    a hash of the resolved worktree path — basename collides when two
    checkouts share a folder name. An explicit `CHARON_TARGET_DIR` still
    wins.
    """
    if value := os.environ.get("CHARON_TARGET_DIR"):
        return Path(value)
    shared = shared_build_dir(root)
    key = hashlib.sha256(os.fsencode(root.resolve())).hexdigest()[:16]
    return shared / "charon-target" / key


def llbc_dest_path(out_dir: Path, root: Path) -> Path:
    """Where artefacts live, without creating it.

    Split from `llbc_dest` for `--fingerprint`, which now reports on those
    files: a read-only instrument that silently creates a directory is one
    whose answer is no longer about the tree it was asked about.
    """
    dest = Path(os.environ.get("LLBC_DEST", out_dir))
    if not dest.is_absolute():
        dest = root / dest
    return dest


def llbc_dest(out_dir: Path, root: Path) -> Path:
    dest = llbc_dest_path(out_dir, root)
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def invalidate_fingerprint(stamp_path: Path, readfiles: Path) -> None:
    """Remove both parts of the certificate for an artefact's current bytes."""
    stamp_path.unlink(missing_ok=True)
    readfiles.unlink(missing_ok=True)


def merge_readfiles(into: dict[str, set[Path]], other: dict[str, set[Path]]) -> None:
    """Fold one artefact's read set into another's, keyed by file-table crate."""
    for crate_name, files in other.items():
        into.setdefault(crate_name, set()).update(files)


def write_readfiles(path: Path, by_crate: dict[str, set[Path]]) -> None:
    """Persist a sorted, LF-terminated artefact read set."""
    files = sorted(
        {item for crate_files in by_crate.values() for item in crate_files},
        key=lambda item: item.as_posix(),
    )
    with path.open("w", encoding="utf-8", newline="\n") as output:
        output.write("".join(f"{item.as_posix()}\n" for item in files))


def assert_readfiles_resolve(root: Path, dest: Path, by_crate: dict[str, set[Path]]) -> None:
    """Refuse a read set whose entries do not name files in this repository.

    Charon roots a `Local` name at the cargo workspace of the crate it
    translated, which is the repository root only while that crate belongs to
    the top-level workspace. A crate carrying its own `Cargo.lock` gets names
    rooted at itself instead -- `majit/charon-corpus` reports `src/lib.rs` --
    and those resolve against the repository root to paths that do not exist.
    Such a set hashes nothing, so edits to the real sources would move
    `closure=` alone and the artefact would keep reading fresh. Every entry was
    a file Charon had just read, so any that cannot be found names a rooting
    this function does not handle.
    """
    unresolved = {
        crate_name: sorted(
            (item for item in files if not (root / item).is_file()),
            key=lambda item: item.as_posix(),
        )
        for crate_name, files in by_crate.items()
    }
    unresolved = {name: files for name, files in unresolved.items() if files}
    if not unresolved:
        return
    lines = [f"extract-llbc.py: {dest.name} names read files that are not in the repository:"]
    for crate_name, files in sorted(unresolved.items()):
        lines.append(f"  {crate_name}:")
        lines.extend(f"    {item.as_posix()}" for item in files)
    lines.append("  a crate outside the top-level cargo workspace roots its Local names at itself")
    raise SystemExit("\n".join(lines))


def assert_artefact_readfiles_covered(
    dest: Path,
    by_crate: dict[str, set[Path]],
    derived: set[Path],
) -> None:
    """Refuse an artefact whose file-table crates escape the derived input set."""
    uncovered = {
        crate_name: sorted(files - derived, key=lambda item: item.as_posix())
        for crate_name, files in by_crate.items()
        if files - derived
    }
    if not uncovered:
        return
    lines = [
        f"extract-llbc.py: {dest.name} has file-table crates outside its derived fingerprint:"
    ]
    for crate_name, files in sorted(uncovered.items()):
        lines.append(f"  {crate_name}:")
        lines.extend(f"    {item.as_posix()}" for item in files)
    raise SystemExit("\n".join(lines))


def charon_version(charon_dest: Path) -> str:
    stamp = charon_dest / ".installed-version"
    return stamp.read_text().strip() if stamp.exists() else "unknown"


# Placeholder `stamp_for(..., include_closure=False)` writes instead of
# walking the cargo closure. `stamp_skip_ok` treats it as "not computed"
# rather than "closure moved".
CLOSURE_UNCOMPUTED = "-"


def stamp_for(
    eng: Engine,
    *,
    crate: str,
    platform_key: str,
    charon_stamp: str,
    cargo_features: str,
    flags: list[str],
    charon_flags: list[str],
    layout_targets: list[str],
    layout_flags: list[str],
    artefacts: str,
    include_closure: bool = True,
) -> str:
    closure = (
        closure_fingerprint(eng, [crate], cargo_features)
        if include_closure
        else CLOSURE_UNCOMPUTED
    )
    return "\n".join(
        [
            f"crate={crate}",
            f"platform={platform_key}",
            f"charon={charon_stamp}",
            f"fingerprint_schema={FINGERPRINT_SCHEMA}",
            f"extraction_abi={eng.extraction_abi}",
            f"features={cargo_features}",
            f"flags={' '.join(flags)}",
            f"charon_flags={' '.join(charon_flags)}",
            f"layout_targets={' '.join(layout_targets)}",
            # The sidecar passes' own inputs: their cargo args and RUSTFLAGS
            # shape the extracted layouts, so changing either must re-extract.
            f"layout_flags={' '.join(layout_flags)}",
            f"layout_rustflags={eng.layout_target_rustflags}",
            f"source={source_fingerprint(eng, [crate], cargo_features)}",
            # The old wide source hash remains as a residual detector. A move
            # here alone warns: the artefact's declared read set did not move,
            # but a new trait impl outside that set can still affect resolution.
            # Skip does not need this walk: `closure=` is not a gate.
            f"closure={closure}",
            # Inputs `source=` structurally cannot cover: everything outside
            # this repo. Kept as its own field rather than folded into
            # `source=` so `check`'s per-field diff can say WHICH side moved —
            # "your own sources changed" and "a patched dependency in another
            # checkout changed" call for different actions, and collapsing
            # them would repeat the mistake this field exists to fix.
            f"external={external_fingerprint(eng, [crate], cargo_features)}",
            # The one field about the OUTPUT rather than its inputs: the
            # artefact bytes, and the layout sidecars beside them, as they
            # stood when this stamp was written. See `artefacts_fingerprint`.
            f"artefacts={artefacts}",
        ]
    )


# Every field `stamp_for` writes, in order. A stamp that does not carry all of
# them was not written by this engine — or was truncated mid-write — and the
# comparison in `check` would then quietly skip whatever is absent.
STAMP_KEYS = (
    "crate",
    "platform",
    "charon",
    "fingerprint_schema",
    "extraction_abi",
    "features",
    "flags",
    "charon_flags",
    "layout_targets",
    "layout_flags",
    "layout_rustflags",
    "source",
    "closure",
    "external",
    "artefacts",
)


def parse_stamp(text: str) -> dict[str, str]:
    """Split a stamp into its `key=value` fields.

    Only used to replay `features=` and to render a readable per-field diff;
    the verdict in `check` is an exact text comparison, so a line this drops
    (blank, or without a `=`) still fails the check.
    """
    fields: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            fields[key] = value
    return fields


# `closure=` is a residual detector, not a skip/stale gate. `check` already
# warns when it alone moves; `extract` used to require the whole stamp to
# match, so a comment in a cargo-closure crate re-extracted every consumer
# even when that crate was absent from the artefact's file table. Keep the
# two verdicts on the same field set.
STAMP_GATE_KEYS = tuple(key for key in STAMP_KEYS if key != "closure")


def stamp_skip_ok(recorded_text: str, expected: str) -> tuple[bool, bool]:
    """Whether extract may skip, and whether `closure=` alone moved.

    `recorded_text` is the on-disk stamp (CRLF folded to LF). `expected` is
    the string `stamp_for` just produced, without its trailing newline.
    Missing any `STAMP_KEYS` field refuses the skip: a stamp that predates a
    field cannot certify what that field now covers.
    """
    recorded = parse_stamp(recorded_text.replace("\r\n", "\n"))
    want = parse_stamp(expected)
    if any(key not in recorded for key in STAMP_KEYS):
        return False, False
    gates_match = all(recorded[key] == want.get(key) for key in STAMP_GATE_KEYS)
    want_closure = want.get("closure")
    if want_closure == CLOSURE_UNCOMPUTED:
        closure_moved = False
    else:
        closure_moved = recorded.get("closure") != want_closure
    return gates_match, closure_moved


# Fields hashed into the store bucket. `source=` is the entry name, not the
# bucket: two worktrees with the same extraction configuration but different
# artefact-declared read sets must not collide. `closure=` and `artefacts=`
# are omitted on purpose — the former is a residual detector, the latter
# hashes bytes that include Charon's `dest_file` absolute path, which differs
# per checkout even when the translated crate does not.
STORE_BUCKET_KEYS = (
    "crate",
    "platform",
    "charon",
    "fingerprint_schema",
    "extraction_abi",
    "features",
    "flags",
    "charon_flags",
    "layout_targets",
    "layout_flags",
    "layout_rustflags",
    "external",
)


def llbc_store_max_age_days() -> int:
    raw = os.environ.get("LLBC_STORE_MAX_AGE_DAYS")
    if raw:
        return int(raw)
    return 3


def llbc_store_dir(root: Path) -> tuple[Path | None, str]:
    """`(store path or None, one-line status)`.

    `LLBC_STORE` set to a path always wins, including under CI. The tokens
    `off`, `0`, and the empty string disable it. CI disables the default
    path so hosted runners do not share a cache they never reuse; an
    explicit path is how a job that wants the store opts in.
    """
    if "LLBC_STORE" in os.environ:
        raw = os.environ["LLBC_STORE"]
        if raw in ("off", "0", ""):
            reason = "LLBC_STORE empty" if raw == "" else f"LLBC_STORE={raw}"
            return None, f"llbc store: disabled ({reason})"
        store = Path(raw)
        return store, f"llbc store: {store}"
    if os.environ.get("CI"):
        return None, "llbc store: disabled (CI)"
    store = shared_build_dir(root) / "llbc-store"
    return store, f"llbc store: {store}"


def llbc_store_bucket(fields: dict[str, str]) -> str:
    payload = "\n".join(f"{key}={fields[key]}" for key in STORE_BUCKET_KEYS)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def llbc_store_entry_dir(store: Path, fields: dict[str, str]) -> Path:
    return (
        store
        / fields["platform"]
        / fields["crate"]
        / llbc_store_bucket(fields)
        / fields["source"]
    )


def clone_file(src: Path, dst: Path) -> None:
    """Copy `src` onto `dst`, cloning/reflinking when the filesystem allows it.

    Never hardlink. Charon rewrites `--dest-file` in place on the next
    extraction; a shared inode would corrupt the store entry the clone
    was meant to preserve.
    """
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.clone-{os.getpid()}-{os.urandom(4).hex()}")
    try:
        cloned = False
        system = platform.system()
        try:
            if system == "Darwin":
                subprocess.run(
                    ["cp", "-c", str(src), str(tmp)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                cloned = True
            elif system == "Linux":
                subprocess.run(
                    ["cp", "--reflink=auto", str(src), str(tmp)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                cloned = True
        except (OSError, subprocess.CalledProcessError):
            cloned = False
        if not cloned:
            tmp.unlink(missing_ok=True)
            shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def touch_llbc_store_entry(entry: Path) -> None:
    marker = entry / "last-used"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()


def llbc_store_payload_files(
    eng: Engine, spec: CrateSpec, dest_dir: Path
) -> list[Path]:
    dest = dest_dir / spec.output_name
    names = [spec.output_name]
    names += [spec.layout_sidecar_name(t) for t in crate_layout_targets(eng, spec)]
    files = [dest_dir / name for name in names]
    files.append(dest.with_suffix(dest.suffix + ".readfiles"))
    files.append(dest.with_suffix(dest.suffix + ".fingerprint"))
    files.append(dest.with_suffix(dest.suffix + ".provenance"))
    return files


def publish_llbc_store_entry(
    store: Path, fields: dict[str, str], sources: list[Path]
) -> Path:
    """Copy `sources` into the store entry for `fields`. Returns the entry path.

    Materialises under `<entry>.tmp-<pid>-<rand>` and `os.rename`s into
    place. An existing entry is left unchanged; only `last-used` is touched.
    """
    entry = llbc_store_entry_dir(store, fields)
    if entry.exists():
        touch_llbc_store_entry(entry)
        return entry
    present = [Path(path) for path in sources if Path(path).is_file()]
    if not present:
        raise FileNotFoundError("llbc store publish: no artefact files to copy")
    bucket = entry.parent
    bucket.mkdir(parents=True, exist_ok=True)
    tmp = bucket / f"{entry.name}.tmp-{os.getpid()}-{os.urandom(4).hex()}"
    try:
        tmp.mkdir()
        for src in present:
            clone_file(src, tmp / src.name)
        (tmp / "last-used").touch()
        try:
            os.rename(tmp, entry)
        except OSError:
            shutil.rmtree(tmp, ignore_errors=True)
            if entry.exists():
                touch_llbc_store_entry(entry)
                return entry
            raise
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return entry


def _llbc_store_entry_used_at(entry: Path) -> float:
    marker = entry / "last-used"
    try:
        return (marker if marker.is_file() else entry).stat().st_mtime
    except OSError:
        return 0.0


def _llbc_store_dir_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def iter_llbc_store_entries(bucket: Path) -> list[Path]:
    if not bucket.is_dir():
        return []
    entries = [
        child
        for child in bucket.iterdir()
        if child.is_dir() and ".tmp-" not in child.name
    ]
    entries.sort(key=_llbc_store_entry_used_at, reverse=True)
    return entries


def prune_llbc_store(store: Path, *, max_age_days: int) -> None:
    """Drop unused store entries, leftover tmp dirs, then empty parents."""
    if not store.is_dir():
        print("llbc store: pruned 0 entries, 0 bytes")
        return
    now = time.time()
    entry_cutoff = now - max_age_days * 86400
    tmp_cutoff = now - 86400
    removed = 0
    total = 0
    for child in sorted(store.glob("*/*/*/*")):
        if not child.is_dir():
            continue
        is_tmp = ".tmp-" in child.name
        try:
            if is_tmp:
                used = child.stat().st_mtime
                cutoff = tmp_cutoff
            else:
                used = _llbc_store_entry_used_at(child)
                cutoff = entry_cutoff
        except OSError:
            continue
        if used >= cutoff:
            continue
        size = _llbc_store_dir_size(child)
        shutil.rmtree(child, ignore_errors=True)
        removed += 1
        total += size
        print(f"llbc store: pruned {child.relative_to(store)} ({size} bytes)")
    for dirpath, _, _ in os.walk(store, topdown=False):
        path = Path(dirpath)
        if path == store:
            continue
        try:
            next(path.iterdir())
        except StopIteration:
            try:
                path.rmdir()
            except OSError:
                pass
        except OSError:
            pass
    print(f"llbc store: pruned {removed} entries, {total} bytes")


def maybe_prune_llbc_store(store: Path) -> bool:
    """Run a default-age prune when `.last-prune` is missing or older than 24 h.

    Returns True if prune ran.
    """
    if not store.is_dir():
        return False
    marker = store / ".last-prune"
    now = time.time()
    try:
        if marker.is_file() and now - marker.stat().st_mtime < 24 * 3600:
            return False
    except OSError:
        pass
    prune_llbc_store(store, max_age_days=llbc_store_max_age_days())
    marker.touch()
    return True


def _llbc_store_on_skip(
    store: Path,
    eng: Engine,
    spec: CrateSpec,
    dest_dir: Path,
    stamp_path: Path,
) -> None:
    recorded = parse_stamp(stamp_path.read_text(encoding="utf-8", errors="replace"))
    entry = llbc_store_entry_dir(store, recorded)
    if entry.exists():
        touch_llbc_store_entry(entry)
        return
    publish_llbc_store_entry(
        store, recorded, llbc_store_payload_files(eng, spec, dest_dir)
    )


def import_from_llbc_store(
    eng: Engine,
    *,
    store: Path,
    spec: CrateSpec,
    crate: str,
    dest_dir: Path,
    dest: Path,
    stamp: str,
    stamp_path: Path,
    readfiles: Path,
    cargo_features: str,
    platform_key: str,
    charon_stamp: str,
    flags: list[str],
    charon_flags: list[str],
    layout_flags: list[str],
) -> bool:
    """Clone a matching store entry into `dest_dir`. True if the crate is done."""
    fields = parse_stamp(stamp)
    bucket = llbc_store_entry_dir(store, fields).parent
    layout_targets = crate_layout_targets(eng, spec)
    names = [spec.output_name]
    names += [spec.layout_sidecar_name(t) for t in layout_targets]
    for entry in iter_llbc_store_entries(bucket):
        entry_readfiles = entry / readfiles.name
        if not entry_readfiles.is_file():
            continue
        digest = source_fingerprint(
            eng, [crate], cargo_features, readfiles_override=entry_readfiles
        )
        if digest != entry.name:
            continue
        for name in names:
            src = entry / name
            if not src.is_file():
                raise FileNotFoundError(f"llbc store entry missing {name}: {entry}")
            clone_file(src, dest_dir / name)
        clone_file(entry_readfiles, readfiles)
        forget_collected_inputs()
        current_stamp = stamp_for(
            eng,
            crate=crate,
            platform_key=platform_key,
            charon_stamp=charon_stamp,
            cargo_features=cargo_features,
            flags=flags,
            charon_flags=charon_flags,
            layout_targets=layout_targets,
            layout_flags=layout_flags,
            artefacts=artefacts_fingerprint(eng, spec, dest_dir),
        )
        stored_path = entry / stamp_path.name
        stored = parse_stamp(
            stored_path.read_text(encoding="utf-8", errors="replace")
            if stored_path.is_file()
            else ""
        )
        want = parse_stamp(current_stamp)
        mismatched = [
            key
            for key in STAMP_GATE_KEYS
            if key != "artefacts" and stored.get(key) != want.get(key)
        ]
        if mismatched:
            invalidate_fingerprint(stamp_path, readfiles)
            forget_collected_inputs()
            details = ", ".join(
                f"{key}: stored={stored.get(key)!r} now={want.get(key)!r}"
                for key in mismatched
            )
            print(
                f"    llbc store entry {entry} does not match this tree"
                f" ({details}); extracting"
            )
            return False
        with stamp_path.open("w", encoding="utf-8", newline="\n") as output:
            output.write(current_stamp + "\n")
        provenance_path = dest.with_suffix(dest.suffix + ".provenance")
        entry_provenance = entry / provenance_path.name
        text = ""
        if entry_provenance.is_file():
            text = entry_provenance.read_text(encoding="utf-8", errors="replace")
            if text and not text.endswith("\n"):
                text += "\n"
        imported_at = datetime.datetime.now().isoformat(timespec="seconds")
        text += f"imported_from={entry}\nimported_at={imported_at}\n"
        provenance_path.write_text(text, encoding="utf-8")
        touch_llbc_store_entry(entry)
        print(f"=== imported {crate} from llbc store ({entry}) ===")
        return True
    return False


def prune_store_cli(eng: Engine, args: argparse.Namespace) -> None:
    store, status = llbc_store_dir(eng.root)
    print(status)
    if store is not None:
        prune_llbc_store(store, max_age_days=args.store_max_age_days)


def file_mtime(path: Path) -> str:
    return datetime.datetime.fromtimestamp(path.stat().st_mtime).isoformat(
        timespec="seconds"
    )


# Provenance records what the repository looked like around the build. It is
# stored beside the artefact, reported by `check`, and never used as a gate.
#
# This does not go in the fingerprint stamp and must not be moved there.
# Two independent reasons, either one sufficient:
#
#   * `check` compares the stamp as exact text and already has an arm for a
#     stamp that "carries content this engine did not write". Extra lines in
#     the stamp file are a REFUSAL, not extra information.
#   * a commit sha moves on EVERY commit, including the overwhelming majority
#     that touch nothing this artefact reads. A git field inside a compared
#     stamp would report every artefact stale after every unrelated commit,
#     which deletes the entire point of a content-addressed fingerprint —
#     `source=` exists precisely so that an unrelated commit does not force a
#     re-extraction.
#
# So the two files answer different questions and are kept apart on purpose:
# the stamp answers "is this artefact current", the provenance answers "what
# was going on when it was built". Only the first is allowed to fail a build.
#
# The first sidecars this code writes may describe a tree containing the engine
# change that introduced them. Extraction semantics are represented by
# FINGERPRINT_SCHEMA and the driver ABI rather than by hashing this file's
# implementation bytes, so ordinary diagnostic edits do not stale every large
# artefact.
#
# More generally, a nonzero
# `dirty_in_closure` IS A REPORT, NOT A DEFECT. An extraction is normally
# batched into a window several people's work lands in, so a sidecar routinely
# names files somebody else was editing. What it tells you is narrow and worth
# saying exactly: the repository fingerprints hash the WORKING TREE, so the artefact is correct
# FOR THAT TREE and is not reproducible from any commit. Kill such a run only
# if commit-reproducible artefacts are what you came for; the freshness
# question was already answered, by the stamp, and the answer was yes.
def git_head(root: Path) -> str:
    """`HEAD`'s sha, or a LABELLED reason it is unavailable.

    Never raises, and never returns the empty string. An unlabelled blank would
    read as "no commit" when it means "could not ask", and those call for
    opposite reactions from whoever reads the sidecar.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return "unavailable(git-not-runnable)"
    if result.returncode != 0:
        return "unavailable(rev-parse-failed)"
    return result.stdout.decode("utf-8", errors="replace").strip() or "unavailable(empty)"


def git_dirty(root: Path) -> tuple[str, list[tuple[str, str]]]:
    """`(status, [(porcelain XY, path), …])` for the worktree at `root`.

    `-uall` is load-bearing. The default collapses an untracked directory into
    a single `?? dir/` entry, while the fingerprint's own untracked leg
    (`git ls-files --others --exclude-standard`) works per FILE. Classifying a
    directory against a set of files would report `?? majit/examples/new/` as
    outside the closure while every file under it is inside it — the one
    reading a person would act on, inverted.

    The status is returned separately rather than folded into an empty list,
    because "the tree is clean" and "git could not be asked" both produce no
    entries and demand opposite reactions.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "-z", "-uall"],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return "unavailable(git-not-runnable)", []
    if result.returncode != 0:
        return "unavailable(status-failed)", []

    fields = result.stdout.decode("utf-8", errors="replace").split("\0")
    entries: list[tuple[str, str]] = []
    index = 0
    while index < len(fields):
        entry = fields[index]
        index += 1
        if not entry:
            continue
        # Porcelain v1 spends exactly two columns on the status and one on the
        # separator, so slice rather than split: a path may itself begin with a
        # space, and `-z` means it is not quoted to hide that.
        code, path = entry[:2], entry[3:]
        # An `R`/`C` entry spends a SECOND NUL-separated field on the original
        # path. Not consuming it re-reads that path as a status code and mints
        # a junk entry — and, worse, shifts every entry after it by one.
        if "R" in code or "C" in code:
            if index < len(fields):
                path = f"{path} (from {fields[index]})"
                index += 1
        entries.append((code, path))
    return "ok", entries


# Ceiling on the per-file `dirty=` listing. Not a ceiling on the COUNTS, which
# stay exact however far past this the tree is.
DIRTY_LINE_LIMIT = 200


def provenance_for(
    *,
    crate: str,
    head_before: str,
    head_after: str,
    started: str,
    finished: str,
    dirty_status: str,
    dirty: list[tuple[str, str]],
    dirty_before: int | None,
    closure: set[str],
) -> str:
    """The provenance sidecar's text: the HEAD pair, and dirt CLASSIFIED.

    Both halves exist because each detects a distinct failure.

    `head_before`/`head_after` are a pair because a build that straddles a
    commit is only visible as one. A single post-hoc `rev-parse` records the
    later revision and hides that source changed while the artefact was built.

    Every dirty path is classified against this crate's own fingerprint
    closure, because the bare porcelain is not actionable: three files dirty
    during an extraction is a run to kill if they are inputs and a non-event if
    they are not, and only the classification distinguishes those. A reader
    handed the unclassified list has to reconstruct the closure by hand, which
    is the step nobody takes.

    `in-closure` covers the in-root half only. The out-of-root inputs
    `external=` hashes live in sibling checkouts this `git status` never sees,
    so `outside` means "not an input from this repository", never "cannot
    affect the artefact".

    A `dirty=` line is `<XY> <where> <path>` with the field widths FIXED at the
    front: `value[:2]` is the porcelain status, `value[3:]` is
    `<where> <path>`. Splitting the whole value on whitespace does not work and
    the reason is easy to miss — a porcelain code is two columns of which
    either may be a SPACE (` M`, `A `), so ` M in-closure x.rs` splits into an
    empty first field. Positions, not separators.
    """
    lines = [
        f"crate={crate}",
        f"head_before={head_before}",
        f"head_after={head_after}",
        f"started={started}",
        f"finished={finished}",
        f"dirty_status={dirty_status}",
    ]
    # A count, not a second list: the pre-build snapshot is here for the same
    # reason the HEAD pair is — so a tree that moved during the build is
    # legible — and repeating the whole listing to say so would bury the one
    # that describes the stamped state.
    #
    # This is the complete unclassified porcelain output. `dirty_in_closure` and
    # `dirty_outside` below are a partition of a LATER snapshot, so the only
    # valid comparison is against their SUM; reading `dirty_before` against
    # either half alone compares a total to a part. And the two are SUPPOSED to
    # be free to differ — that difference is the whole instrument, and making
    # them agree would delete it.
    lines.append(
        f"dirty_before={'unknown' if dirty_before is None else dirty_before}"
    )
    classified = [
        (code, path, "in-closure" if path in closure else "outside")
        for code, path in dirty
    ]
    in_closure = [entry for entry in classified if entry[2] == "in-closure"]
    outside = [entry for entry in classified if entry[2] == "outside"]
    # COUNTS ARE ALWAYS EXACT; only the per-file listing is bounded. `-uall`
    # asks git to expand untracked directories file by file, so a single
    # untracked tree of generated files could otherwise write a sidecar of
    # unbounded size into `build/`. In-closure entries are listed first and
    # never dropped: they are the actionable set, and a truncation that ate
    # them would remove the only lines anyone acts on.
    lines.append(f"dirty_in_closure={len(in_closure)}")
    lines.append(f"dirty_outside={len(outside)}")
    shown = in_closure + outside[: max(0, DIRTY_LINE_LIMIT - len(in_closure))]
    # Always written, including as `0`, so a reader never has to decide what an
    # absent key meant.
    lines.append(f"dirty_omitted={len(classified) - len(shown)}")
    for code, path, where in shown:
        lines.append(f"dirty={code} {where} {path}")
    return "\n".join(lines)


def parse_provenance(text: str) -> tuple[dict[str, str], list[str]]:
    """`({single-valued key: value}, [dirty line, …])`.

    `dirty=` repeats, so it is returned as its own list; collapsing it into the
    dict would keep the last entry and silently report one dirty file where
    there were thirty.
    """
    fields: dict[str, str] = {}
    dirty: list[str] = []
    for line in text.splitlines():
        key, sep, value = line.partition("=")
        if not sep:
            continue
        if key == "dirty":
            dirty.append(value)
        else:
            fields[key] = value
    return fields, dirty


def dirty_report_lines(fields: dict[str, str], in_closure: list[str]) -> list[str]:
    """The dirty half of `report_provenance`'s output, as text.

    Split out from the printer so `--self-test` can drive it. Both conditional
    lines below are SILENT on a healthy sidecar — the delta is zero and nothing
    is omitted — so a run against the real tree cannot tell a correct branch
    from a dead one, and every sidecar this repository has ever written is
    healthy in exactly that way. Keeping it a pure function also keeps the
    self-test off the disk, so a diagnostic check cannot perturb extraction.

    Two moments, on two lines. Printed as one line with the pre-build count in a
    parenthetical, this reads as a single measurement whose parts should sum,
    and it silently mixes three bases: a stamp-time in-closure count, a
    stamp-time outside count, and a pre-build UNCLASSIFIED whole-tree total.
    That reading is what has to be prevented — the numbers were never wrong, and
    a disagreement between the moments is the instrument, not a fault.
    """
    at_stamp_in = fields.get("dirty_in_closure", "?")
    at_stamp_out = fields.get("dirty_outside", "?")
    before = fields.get("dirty_before", "?")
    lines = [
        f"    dirty when stamped: {at_stamp_in} in this crate's closure,"
        f" {at_stamp_out} outside",
        f"    dirty before the build: {before} (whole tree, unclassified)",
    ]
    # Computed, not left to the reader to do in their head. The two TOTALS are
    # comparable — `dirty_before` and `in_closure + outside` are both whole
    # porcelain — and it is the naive comparison against one HALF, or against a
    # count of the rows below, that is invalid. Silent at zero, and silent when
    # `dirty_before` is `unknown`, which is not the same as zero.
    if at_stamp_in.isdigit() and at_stamp_out.isdigit() and before.isdigit():
        delta = int(at_stamp_in) + int(at_stamp_out) - int(before)
        if delta:
            lines.append(
                f"    ⇒ the tree MOVED during the build: {delta:+d} dirty"
                " path(s) between the two snapshots — the pair exists to say so"
            )
    # The `dirty=` rows are a bounded DISPLAY while the counts above are exact,
    # so counting the rows is not a check on the counts. Said here because the
    # bound lives in the writer and no reader of this output would find it.
    omitted = fields.get("dirty_omitted", "0")
    if omitted not in ("0", "?"):
        lines.append(
            f"      WARNING: {omitted} further dirty path(s) not listed — do not count"
            " the rows, the counts above are the exact ones"
        )
    lines.extend(f"      {entry}" for entry in in_closure)
    return lines


def report_provenance(dest: Path) -> None:
    """Print an artefact's provenance. Returns nothing and gates nothing.

    Absence is not a staleness signal. Every
    artefact extracted before this sidecar existed lacks it, and so does every
    artefact copied in from elsewhere; refusing on that would turn a reporting
    aid into a second, weaker freshness gate answering a question `source=`
    already answers properly.

    The in-closure dirty entries are printed in full and the outside ones only
    counted, because that is the asymmetry the classification exists to
    express: an input dirty during the build is the thing to act on, and a
    hundred unrelated edits are the noise that hid it.
    """
    path = dest.with_suffix(dest.suffix + ".provenance")
    if not path.exists():
        print(
            "    no provenance sidecar (predates it, or was written by another"
            " tool) — not a staleness signal"
        )
        return
    fields, dirty = parse_provenance(path.read_text())
    before, after = fields.get("head_before", "?"), fields.get("head_after", "?")
    if before == after:
        print(f"    built at HEAD {before}, {fields.get('started', '?')}")
    else:
        # Said plainly rather than as a warning: the artefact may be perfectly
        # current — `source=` is content-addressed and does not care which
        # commit the tree sat on. What a straddle costs is the ability to name
        # ONE tree this artefact came from.
        print(f"    WARNING: built across a commit: {before} -> {after}")
    if "imported_from" in fields or "imported_at" in fields:
        print(
            f"    imported from {fields.get('imported_from', '?')} at"
            f" {fields.get('imported_at', '?')}"
        )
    status = fields.get("dirty_status", "?")
    if status != "ok":
        print(f"    dirty state at build time: {status} — neither clean nor known")
        return
    # Positional, per `provenance_for`: the status occupies two fixed columns,
    # so the classification starts at offset 3. A substring test would also
    # match a PATH that happened to contain the word.
    in_closure = [entry for entry in dirty if entry[3:].startswith("in-closure ")]
    for line in dirty_report_lines(fields, in_closure):
        print(line)


def crate_layout_targets(eng: Engine, spec: CrateSpec) -> list[str]:
    """Cross targets this crate emits a layout sidecar for.

    The extraction host is never in the list: its layouts already live in
    the crate's own artefact.
    """
    extra = eng.layout_targets if spec.layout_targets is None else spec.layout_targets
    if not extra:
        return []
    host = host_triple(eng.root)
    return [t for t in extra if t != host]


def scan_for_keys(
    path: Path, keys: list[bytes], chunk_size: int = 1 << 22
) -> dict[bytes, int]:
    """Byte offset of the first occurrence of each key, in one streamed pass."""
    found: dict[bytes, int] = {}
    keep = max(len(key) for key in keys) - 1
    buffer = b""
    base = 0
    with path.open("rb") as artefact:
        while len(found) < len(keys):
            piece = artefact.read(chunk_size)
            if not piece:
                break
            buffer += piece
            for key in keys:
                if key not in found:
                    index = buffer.find(key)
                    if index >= 0:
                        found[key] = base + index
            base += len(buffer) - keep
            buffer = buffer[-keep:]
    return found


# First decode window of `write_layout_sidecar`, in bytes; it widens fourfold
# until the value closes. Wide enough for every scalar field in one read and
# for the smaller crates' `type_decls`, small enough that a miss costs little.
LAYOUT_DECODE_WINDOW = 1 << 22


def write_layout_sidecar(source: Path, dest: Path) -> None:
    """Reduce a full `.ullbc` to its type declarations.

    A cross-target extraction is only wanted for the struct layouts Charon
    resolved for that target; its function bodies are compiled under
    different `cfg`s and must not displace the host's. Dropping every
    declaration list but `type_decls` keeps the artefact a loadable
    `.ullbc` — the reader needs no special case, and the consumer merges
    it ahead of the host artefacts so its layouts win.

    Only the fields kept are decoded, and only a window of the file around
    each is ever in memory: the artefact is three quarters of a GB of
    one-line JSON, and `read_text` on it peaked at 7 GB of RSS (measured on
    the interpreter's), the entire memory of a 7 GB macos runner. The keys
    are located by one streamed byte scan; each value is then decoded from
    a window that starts at the value and widens until the value closes.
    """
    decoder = json.JSONDecoder()
    keys = {
        name: f'"{name}":'.encode("ascii")
        for name in ("charon_version", "has_errors", "crate_name", "type_decls")
    }
    offsets = scan_for_keys(source, list(keys.values()))
    size = source.stat().st_size

    def field(name: str, *, want: type):
        key = keys[name]
        if key not in offsets:
            raise SystemExit(f"extract-llbc.py: {source.name} has no `{name}` field")
        start = offsets[key] + len(key)
        window = LAYOUT_DECODE_WINDOW
        with source.open("rb") as artefact:
            while True:
                artefact.seek(start)
                # A window cut inside a multi-byte character only loses bytes
                # PAST the value being decoded; a value that does not close
                # inside the window is retried with a wider one.
                text = artefact.read(window).decode("utf-8", errors="replace")
                # `raw_decode` does not skip leading whitespace of its own.
                at = 0
                while at < len(text) and text[at] in " \t\r\n":
                    at += 1
                try:
                    value, _ = decoder.raw_decode(text, at)
                except json.JSONDecodeError as exc:
                    if start + window < size:
                        window *= 4
                        continue
                    # The substring scan is unscoped, so `"{name}":` could
                    # match inside a string literal before the real key; then
                    # `start` points at non-JSON and decoding fails.  Report
                    # it the same way the rest of this file does instead of
                    # surfacing a bare traceback.
                    raise SystemExit(
                        f"extract-llbc.py: {source.name} `{name}` did not decode as"
                        f" JSON ({exc}); the reducer's key scan matched a non-field"
                        " occurrence."
                    ) from exc
                break
        # Shape guard: a wrong (earlier) match can still decode as valid JSON of
        # the wrong type.  `type_decls` in particular must reach the sidecar
        # intact — it is the sole source of the target field offsets.
        if not isinstance(value, want):
            raise SystemExit(
                f"extract-llbc.py: {source.name} `{name}` decoded as "
                f"{type(value).__name__}, expected {want.__name__}; the reducer's "
                f"key scan matched a non-field occurrence."
            )
        return value

    slim = {
        "charon_version": field("charon_version", want=str),
        "has_errors": field("has_errors", want=bool),
        "translated": {
            "crate_name": field("crate_name", want=str),
            "type_decls": field("type_decls", want=list),
            "fun_decls": [],
        },
    }
    dest.write_text(json.dumps(slim), encoding="utf-8")


def charon_toolchain_path(charon_bin: Path, root: Path) -> Path:
    """The rustup toolchain directory Charon's driver runs under.

    The LAST NON-EMPTY LINE, not the whole capture: `charon toolchain-path`
    shares its stream with rustup, which announces an install on a runner that
    does not have the pinned nightly yet -- a macos job printed "The required
    toolchain is not installed. Installing...", the toolchain it had just
    installed, and only then the path. Reading the capture whole made a Path of
    all three, and the `rustc -vV` built from it went looking for a file named
    after the notice.
    """
    lines = [
        line.strip()
        for line in run_capture(
            [str(charon_bin), "toolchain-path"], cwd=root
        ).splitlines()
        if line.strip()
    ]
    if not lines:
        raise SystemExit(
            f"extract-llbc.py: {charon_bin} toolchain-path printed no path"
        )
    return Path(lines[-1])


def ensure_charon_std(charon_bin: Path, targets: list[str], root: Path) -> None:
    """Install the target `std` Charon's pinned toolchain needs.

    Charon drives its own nightly rustc, so a target installed for the
    build toolchain is not necessarily installed for Charon's. Adding it is
    idempotent and fast when already present.
    """
    toolchain_name = charon_toolchain_path(charon_bin, root).name
    for target in targets:
        command = ["rustup", "target", "add", target, "--toolchain", toolchain_name]
        if subprocess.run(command).returncode != 0:
            raise SystemExit(
                f"extract-llbc.py: could not install '{target}' std for Charon's "
                f"toolchain '{toolchain_name}'.\n"
                f"  run: {' '.join(command)}\n"
                "  or drop the target from LLBC_LAYOUT_TARGETS to extract host "
                "layouts only (a cross-target build then reads the host's field "
                "offsets)."
            )


def charon_host_triple(charon_bin: Path, root: Path) -> str:
    """The target triple Charon's `cargo` passes as `--target`.

    `charon cargo` always names the host explicitly so its driver can tell
    build scripts and proc macros (host units) from the crate it translates,
    which is why the extraction lands under `target/<host-triple>/debug`
    rather than `target/debug`. Read it from Charon's own rustc: the build
    toolchain can be a different host spelling on the same machine.
    """
    toolchain = charon_toolchain_path(charon_bin, root)
    rustc = toolchain / "bin" / ("rustc.exe" if os.name == "nt" else "rustc")
    for line in run_capture([str(rustc), "-vV"], cwd=root).splitlines():
        if line.startswith("host:"):
            return line[len("host:"):].strip()
    raise SystemExit(f"extract-llbc.py: {rustc} -vV printed no `host:` line")


def cargo_unit_location(crate_dir: Path) -> tuple[Path, str]:
    """`(target_directory, package_name)` for the crate Charon builds in `crate_dir`.

    Resolved through `cargo metadata --no-deps` from the crate's own directory
    rather than the workspace root: a crate outside the workspace (`corpus`
    carries its own `Cargo.lock`) has its own `target/`, and the package name
    is what cargo spells the `.fingerprint/<name>-<hash>` entry with, which is
    the manifest `name`, not the driver's spec key.
    """
    result = json.loads(
        run_capture(
            ["cargo", "metadata", "--no-deps", "--locked", "--format-version=1"],
            cwd=crate_dir,
        )
    )
    manifest = (crate_dir / "Cargo.toml").resolve()
    for package in result["packages"]:
        if Path(package["manifest_path"]).resolve() == manifest:
            return Path(result["target_directory"]), package["name"]
    raise SystemExit(
        f"extract-llbc.py: `cargo metadata --no-deps` in {crate_dir} lists no "
        f"package whose manifest is {manifest}"
    )


def invalidate_cargo_unit(target_dir: Path, triple: str, package: str) -> int:
    """Drop cargo's fingerprint for `package` in the `<triple>/debug` layout.

    Charon writes the `.ullbc` only while rustc actually compiles the crate,
    and a warm cache makes the inner `cargo build` skip rustc and emit
    nothing. The fingerprint is the one record cargo consults to decide that,
    so removing it forces exactly this unit to recompile — every feature
    variant of it, since the hash in the directory name is not known here —
    while its dependencies stay cached.

    This replaces a `touch` of the crate root. cargo's freshness check is
    mtime-based, so the touch also re-staled the same sources for every OTHER
    layout on disk (the host units beside this one, each cross-target layout,
    and the stable build's `target/release`), and their dependents with them:
    one extraction recompiled the majit stack six times over an unchanged
    tree. A missing fingerprint is local to one layout and touches no input.

    Returns the number of fingerprint entries removed; zero on a cold cache,
    which needs no forcing.
    """
    fingerprints = target_dir / triple / "debug" / ".fingerprint"
    removed = 0
    if not fingerprints.is_dir():
        return removed
    for entry in fingerprints.glob(f"{package}-*"):
        # `<name>-<hash>`: the hash is 16 hex digits, so a package whose name
        # extends this one (`pyre-jit` vs `pyre-jit-trace`) cannot match.
        suffix = entry.name[len(package) + 1 :]
        if len(suffix) != 16 or not all(c in "0123456789abcdef" for c in suffix):
            continue
        shutil.rmtree(entry, ignore_errors=True)
        removed += 1
    return removed


def window_writes(
    inputs: list[Path],
    root: Path,
    lo_ns: int,
    hi_ns: int,
) -> tuple[list[tuple[int, Path]], int, list[str]]:
    """Closure inputs written while this crate was being extracted.

    The stamp comparison beside this one is an ENDPOINT check: it hashes the
    tree before the build and again after. That is blind to the write that
    matters most — an edit that opens and closes INSIDE the build re-converges,
    so both hashes agree, `git status` and `git diff` are clean, and the
    artefact is stamped over a tree that briefly was not the one it names.

    A restore is itself a write, so mtime lands inside the window even when the
    bytes came back. That is the only channel that retains the event, and it
    retains it for a short time: mtime holds ONE value, so the next ordinary
    write destroys it. That is why this runs here, in the producer, at the
    moment the window closes — a sweep run later is not a weaker version of
    this check, it silently answers a different question.

    Returns `(candidates, covered, unresolved)`. `candidates` is the word on
    purpose: this names files, never an author.

    KNOWN LIMITS, printed by the caller rather than filed elsewhere:
      * An mtime-preserving restore (`cp -p`) defeats it completely.
      * It has never produced a real-world true positive; it is proven to fire
        on a constructed edit-and-restore only.

    The extractor itself writes no closure input: it forces Charon's recompile
    through cargo's fingerprint (`invalidate_cargo_unit`), so every candidate
    reported here is a peer's write.
    """
    candidates: list[tuple[int, Path]] = []
    unresolved: list[str] = []
    for rel in inputs:
        try:
            st = (root / rel).stat()
        except OSError as exc:
            # Reported, never skipped silently: an input the sweep could not
            # read is a hole in its own denominator.
            unresolved.append(f"{rel.as_posix()} ({exc.strerror})")
            continue
        if not lo_ns <= st.st_mtime_ns <= hi_ns:
            continue
        candidates.append((st.st_mtime_ns, rel))
    candidates.sort()
    return candidates, len(inputs) - len(unresolved), unresolved


def finish_charon_pass(proc: subprocess.Popen, command: list[str], dest: Path) -> None:
    """Wait for a Charon pass and refuse a run that produced no artefact."""
    if proc.wait() != 0:
        raise subprocess.CalledProcessError(proc.returncode, command)
    # Fail loud rather than letting a missing artefact surface later
    # as an opaque build.rs panic ("target/llbc/ is missing …").
    if not dest.exists() or dest.stat().st_size == 0:
        raise SystemExit(
            f"extract-llbc.py: Charon emitted no artefact at {dest}\n"
            "  the crate compiled but produced no MIR — "
            "inspect the Charon output above"
        )


def physical_memory_bytes() -> int | None:
    """Installed RAM, or None where the platform does not say (Windows)."""
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return None


def layout_passes_in_parallel() -> bool:
    """Whether a crate's cross-target layout pass runs beside its host pass.

    Each Charon pass over `pyre-interpreter` peaks near 5 GB and is
    single-threaded, so two at once want a second core to spare and ~10 GB
    headroom over whatever else the box is doing. Hosted CI runners (4 cores,
    and 7 GB on macOS) stay sequential; a developer machine goes parallel.
    `LLBC_PARALLEL_LAYOUTS=0|1` overrides the guess either way.
    """
    override = os.environ.get("LLBC_PARALLEL_LAYOUTS")
    if override in ("0", "1"):
        return override == "1"
    if override is not None:
        raise SystemExit(
            f"extract-llbc.py: LLBC_PARALLEL_LAYOUTS={override!r}; expected 0 or 1"
        )
    memory = physical_memory_bytes()
    return (os.cpu_count() or 1) >= 8 and memory is not None and memory >= 16 << 30


def extract(eng: Engine, args: argparse.Namespace) -> None:
    cargo_features = eng.cargo_features
    platform_key, charon_dest, charon_bin = charon_paths(eng.charon_root)
    store, store_status = llbc_store_dir(eng.root)
    print(store_status)

    if not charon_bin.exists():
        raise SystemExit(
            f"extract-llbc.py: charon not installed at {charon_bin}\n"
            "  run: scripts/install-charon.py"
        )

    dest_dir = llbc_dest(eng.out_dir, eng.root)
    charon_stamp = charon_version(charon_dest)
    unstamped: list[str] = []
    env = os.environ.copy()
    prepend_msvc_link(env)
    # Dedicated dir so Charon's nightly rlibs are not mixed with the
    # stable `target/` rust-cache, and so `invalidate_cargo_unit` sees
    # the same layout `charon cargo` writes. Set on `os.environ` too:
    # `cargo_unit_location` runs `cargo metadata` without an env dict.
    charon_target = charon_cargo_target_dir(eng.root)
    charon_target.mkdir(parents=True, exist_ok=True)
    env["CARGO_TARGET_DIR"] = str(charon_target)
    os.environ["CARGO_TARGET_DIR"] = str(charon_target)
    print(f"charon cargo target: {charon_target}")

    # Charon reads MIR straight from rustc; the compiled binary is discarded
    # and only the `.ullbc` is kept, so debuginfo is dead weight here. Drop it
    # to skip DWARF generation across the whole extraction graph. The nightly
    # extraction build fingerprints separately from the stable build (distinct
    # rustc), so this never thrashes the runtime build's cache, and the LLBC is
    # independent of debuginfo so the artefact is byte-identical.
    env.setdefault("CARGO_PROFILE_DEV_DEBUG", "0")
    # The extracted image must describe the interpreter the JIT accompanies,
    # and that one is built by `cargo build --release`, where
    # `debug_assertions` is off. The dev profile leaves it on (the workspace
    # `[profile.dev]` comment says so outright), so every `debug_assert!` and
    # every `cfg(debug_assertions)` arm in the closure was being translated
    # into the jitcodes: `w_bytearray_len` is `debug_assert_eq!(ba.length,
    # (*ba.data).len()); ba.length`, and its body came out 44 bytes — a
    # `residual_call` on an un-lowered `__len`, an allocation and two stores
    # for the `assert_eq!` operand pair, then the field read the function
    # actually is.
    #
    # The divergence runs the dangerous way. Compiled code carrying a check
    # the interpreter does not have can fail where the interpreter succeeds;
    # compiled code missing a debug-only check merely loses a diagnostic. The
    # `cfg(debug_assertions)` arms in the closure are all of the second kind —
    # `gc_roots`' `ShadowStackWalk` re-entrancy guard, whose own doc says
    # "release builds match upstream's raw root-stack access without that
    # diagnostic" — so off is both the matching and the safer setting.
    #
    # Set rather than defaulted, unlike `CARGO_PROFILE_DEV_DEBUG` above: that
    # one cannot change the LLBC, this one is the LLBC. An inherited override
    # would produce an artefact its own stamp misdescribes.
    env["CARGO_PROFILE_DEV_DEBUG_ASSERTIONS"] = "false"
    # The dev profile enables incremental compilation, which splits a crate into
    # codegen units and reuses object files from the previous session. Charon
    # drives rustc for the crates it extracts while plain rustc builds the rest,
    # and the two disagree about which CGUs are still valid: the resulting
    # archive fuses objects from different sessions, so CGU-local symbols
    # (`...drop_in_place...llvm.<hash>`) are referenced by one object and
    # defined in none. The failure surfaces far from its cause, as
    # `Undefined symbols for architecture arm64` while LINKING the
    # `pyre-jit-trace` BUILD SCRIPT, which is a host binary and therefore the
    # first thing in the graph that has to resolve real code out of
    # `libmajit_translate`. The tell is that the `.rcgu.o` files named in the
    # linker error carry a different session tag than the loose ones on disk.
    # Extraction only needs MIR, so incremental buys nothing here.
    env["CARGO_INCREMENTAL"] = "0"
    # Dependency build scripts run while Charon extracts a target crate. They
    # must not recursively demand the very LLBC artefact currently being
    # produced (pyre-jit -> pyre-jit-trace -> pyre-jit.ullbc). Consumers may
    # use this explicit extraction mode to emit compile-only placeholders;
    # `rerun-if-env-changed` ensures the following normal build regenerates
    # production artifacts from the completed LLBC set.
    env["MAJIT_LLBC_EXTRACTION"] = "1"
    host_config_env = {
        "CARGO_UNSTABLE_HOST_CONFIG": "true",
        "CARGO_UNSTABLE_TARGET_APPLIES_TO_HOST": "true",
    }
    host_config = [
        "--config",
        "target-applies-to-host=false",
    ]

    prepared_std: set[str] = set()
    charon_host = charon_host_triple(charon_bin, eng.root)
    parallel_layouts = layout_passes_in_parallel()
    if parallel_layouts and any(
        crate_layout_targets(eng, eng.spec(crate))
        for crate in args.crates or eng.default_crates
    ):
        print(
            "layout passes run beside the host pass (LLBC_PARALLEL_LAYOUTS=0"
            " to serialise them)"
        )

    for crate in args.crates or eng.default_crates:
        spec = eng.spec(crate)
        path = spec.crate_dir
        flags = crate_flags(spec, cargo_features)
        charon_flags = charon_crate_flags(spec, cargo_features)
        if not path.is_dir():
            raise SystemExit(f"extract-llbc.py: missing crate dir for '{crate}' at {path}")

        dest = dest_dir / spec.output_name
        stamp_path = dest.with_suffix(dest.suffix + ".fingerprint")
        readfiles = dest.with_suffix(dest.suffix + ".readfiles")
        if args.force:
            # A forced run bootstraps conservatively. The new artefact supplies
            # the narrower read set only after Charon has successfully emitted it.
            invalidate_fingerprint(stamp_path, readfiles)
            forget_collected_inputs()
        layout_flags = crate_layout_flags(spec, cargo_features, flags)
        stamp = stamp_for(
            eng,
            crate=crate,
            platform_key=platform_key,
            charon_stamp=charon_stamp,
            cargo_features=cargo_features,
            flags=flags,
            charon_flags=charon_flags,
            layout_targets=crate_layout_targets(eng, spec),
            layout_flags=layout_flags,
            artefacts=artefacts_fingerprint(eng, spec, dest_dir),
            include_closure=False,
        )

        sidecars = [
            dest_dir / spec.layout_sidecar_name(t) for t in crate_layout_targets(eng, spec)
        ]
        recorded_ok = False
        closure_moved = False
        if stamp_path.exists():
            recorded_text = stamp_path.read_bytes().decode(
                "utf-8", errors="replace"
            )
            recorded_ok, closure_moved = stamp_skip_ok(recorded_text, stamp)
        if (
            not args.force
            and dest.exists()
            and dest.stat().st_size > 0
            and all(s.exists() and s.stat().st_size > 0 for s in sidecars)
            and recorded_ok
        ):
            if closure_moved:
                print(
                    f"=== skipping {crate} -> {dest} "
                    "(source unchanged; closure moved) ==="
                )
            else:
                print(f"=== skipping {crate} -> {dest} (fingerprint unchanged) ===")
            # No provenance is written here on purpose. The sidecar describes
            # the artefact, and the artefact is the one the previous run built:
            # re-stamping it with the current HEAD would claim this checkout produced
            # bytes it did not.
            if store is not None:
                try:
                    _llbc_store_on_skip(store, eng, spec, dest_dir, stamp_path)
                except Exception as exc:
                    print(
                        f"warn: llbc store: skip record failed for {crate}: {exc}",
                        file=sys.stderr,
                    )
            continue

        if store is not None and not args.force:
            try:
                if import_from_llbc_store(
                    eng,
                    store=store,
                    spec=spec,
                    crate=crate,
                    dest_dir=dest_dir,
                    dest=dest,
                    stamp=stamp,
                    stamp_path=stamp_path,
                    readfiles=readfiles,
                    cargo_features=cargo_features,
                    platform_key=platform_key,
                    charon_stamp=charon_stamp,
                    flags=flags,
                    charon_flags=charon_flags,
                    layout_flags=layout_flags,
                ):
                    continue
            except Exception as exc:
                print(
                    f"warn: llbc store: import failed for {crate}: {exc}",
                    file=sys.stderr,
                )

        # The stamp certifies the exact artefact bytes from the previous
        # successful extraction. This run may replace those bytes before any
        # of the validation below fails, so the old certificate must stop
        # being publishable before Charon can write to `dest`.
        if not args.force:
            invalidate_fingerprint(stamp_path, readfiles)
            forget_collected_inputs()

        # Opening half of the provenance pair, taken before the Charon build
        # that is about to run for minutes. See `provenance_for`.
        head_before = git_head(eng.root)
        started = datetime.datetime.now().isoformat(timespec="seconds")
        # The window is opened from THIS process's clock, not from an
        # argument: a bound supplied by a caller is a bound nobody can
        # check, and a guessed one reads exactly like a measured one.
        window_lo_ns = time.time_ns()
        dirty_status_before, dirty_entries_before = git_dirty(eng.root)
        dirty_before = (
            len(dirty_entries_before) if dirty_status_before == "ok" else None
        )

        print(f"=== extracting {crate} -> {dest} ===")
        # Skip used `include_closure=False`. The window check below compares
        # closure=/external= from before Charon to after; a placeholder
        # `closure=-` against the real digest would look like the tree moved.
        stamp = stamp_for(
            eng,
            crate=crate,
            platform_key=platform_key,
            charon_stamp=charon_stamp,
            cargo_features=cargo_features,
            flags=flags,
            charon_flags=charon_flags,
            layout_targets=crate_layout_targets(eng, spec),
            layout_flags=layout_flags,
            artefacts=artefacts_fingerprint(eng, spec, dest_dir),
        )
        # Once the fingerprint skip above is past, the artefact is known
        # absent or stale and must be (re)generated, so this unit must reach
        # rustc whatever cargo's cache says; see `invalidate_cargo_unit`.
        # Dependency crates stay cached (their MIR reaches Charon via rlib
        # metadata), so re-runs remain cheap.
        target_dir, package = cargo_unit_location(path)
        invalidate_cargo_unit(target_dir, charon_host, package)

        host_env = {**env, **host_config_env}
        command = [
            str(charon_bin),
            "cargo",
            "--ullbc",
            "--dest-file",
            str(dest),
            *charon_flags,
            "--",
            *flags,
            *host_config,
        ]
        layout_passes: list[
            tuple[
                str, Path, Path, subprocess.Popen | None, typing.BinaryIO | None, float
            ]
        ] = []
        # Wall-clock marks for the per-phase figures printed below. Under CI
        # stdout is block-buffered, so every line of a crate's extraction
        # lands with one timestamp and the figures in the text are the only
        # record of where the minutes went.
        phase_started = time.monotonic()
        host_pass = subprocess.Popen(command, cwd=path, env=host_env)
        try:
            # One extra extraction per cross target, reduced to its type
            # declarations. Charon's own `--targets` aggregation would fold
            # every target's layouts into this one artefact, but it aggregates
            # the *bodies* too: a `cfg`-differing function lands two or three
            # times over, and a portal that must resolve to an exact graph no
            # longer does. Extracting each target separately keeps the host
            # artefact's bodies untouched, and dropping everything but
            # `type_decls` from the cross-target one leaves only what a
            # cross-target build actually lacks — its own field offsets.
            #
            # The cross-target pass does not depend on the host one, and each is
            # a single-threaded Charon translation of the same crate (the bulk of
            # the pass), so on a machine with the cores and memory for two they
            # run side by side; see `layout_passes_in_parallel`. A parallel pass
            # gets its own target directory: cargo locks each layout's
            # `.cargo-lock`, and both passes would otherwise serialise on the
            # host layout they share for build scripts and proc macros.
            for target in crate_layout_targets(eng, spec):
                if target not in prepared_std:
                    ensure_charon_std(charon_bin, [target], eng.root)
                    prepared_std.add(target)
                sidecar = dest_dir / spec.layout_sidecar_name(target)
                full = sidecar.with_suffix(sidecar.suffix + ".full")
                layout_env = dict(env)
                if eng.layout_target_rustflags:
                    layout_env["RUSTFLAGS"] = (
                        layout_env.get("RUSTFLAGS", "") + " " + eng.layout_target_rustflags
                    ).strip()
                layout_target_dir = target_dir
                if parallel_layouts:
                    layout_target_dir = target_dir / "llbc-layouts"
                    layout_env["CARGO_TARGET_DIR"] = str(layout_target_dir)
                invalidate_cargo_unit(layout_target_dir, target, package)
                layout_command = [
                    str(charon_bin),
                    "cargo",
                    "--ullbc",
                    "--dest-file",
                    str(full),
                    "--targets",
                    target,
                    *charon_flags,
                    "--",
                    *layout_flags,
                ]
                print(f"=== extracting {crate} layouts for {target} -> {sidecar} ===")
                if parallel_layouts:
                    # Output is held back and printed only on failure: two
                    # interleaved cargo streams are unreadable, and the host
                    # pass is the one whose progress a reader follows.
                    #
                    # Held in a file, not a pipe. Nothing reads this stream until
                    # the host pass has finished, and a Charon pass writes far
                    # more than a pipe buffer holds, so a pipe stops the child at
                    # the first full buffer -- for exactly the span the two
                    # passes exist to overlap.
                    log = tempfile.TemporaryFile()
                    proc = subprocess.Popen(
                        layout_command,
                        cwd=path,
                        env=layout_env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    layout_passes.append(
                        (target, sidecar, full, proc, log, time.monotonic())
                    )
                else:
                    layout_passes.append(
                        (target, sidecar, full, None, None, time.monotonic())
                    )
                    # Sequential: the host pass holds the shared layout's lock,
                    # so run this one only after it has finished.
                    if host_pass is not None:
                        finish_charon_pass(host_pass, command, dest)
                        host_pass = None
                        print(f"    host pass: {time.monotonic() - phase_started:.0f} s")
                        phase_started = time.monotonic()
                    subprocess.run(layout_command, cwd=path, env=layout_env, check=True)
                    print(f"    {target} pass: {time.monotonic() - phase_started:.0f} s")
                    phase_started = time.monotonic()

            if host_pass is not None:
                finish_charon_pass(host_pass, command, dest)
                host_pass = None
                print(f"    host pass: {time.monotonic() - phase_started:.0f} s")
                phase_started = time.monotonic()

            layout_tables: dict[str, set[Path]] = {}
            for target, sidecar, full, proc, log, launched in layout_passes:
                if proc is not None:
                    returncode = proc.wait()
                    with log:
                        if returncode != 0:
                            log.seek(0)
                            sys.stdout.write(log.read().decode("utf-8", errors="replace"))
                            raise subprocess.CalledProcessError(returncode, proc.args)
                    # Its own duration, from ITS launch: the pass ran beside
                    # the host pass, so the wait measured from the host pass's
                    # end is only the tail it added to the wall clock.
                    print(
                        f"    {target} pass: {time.monotonic() - launched:.0f} s"
                        f" ({time.monotonic() - phase_started:.0f} s beyond the"
                        " host pass)"
                    )
                    phase_started = time.monotonic()
                if not full.exists() or full.stat().st_size == 0:
                    raise SystemExit(
                        f"extract-llbc.py: Charon emitted no {target} artefact at {full}"
                    )
                write_layout_sidecar(full, sidecar)
                # This artefact was compiled on its own, so its file table names
                # sources the host build never reaches: everything behind a
                # `cfg(target_arch)`, such as the wasm backend `pyre-interpreter`
                # picks up through `majit-metainterp` on wasm32. Only `type_decls`
                # survive into the sidecar, so take the read set before the full
                # artefact goes. Without it an edit to a wasm-only source moves
                # `closure=` alone, which is a warning, and the layouts this
                # sidecar froze are consumed as current.
                merge_readfiles(layout_tables, artefact_local_readfiles(full))
                full.unlink()
                print(
                    f"    wrote {sidecar} ({sidecar.stat().st_size} bytes),"
                    f" reduced in {time.monotonic() - phase_started:.0f} s"
                )
                phase_started = time.monotonic()
        finally:
            # A raise anywhere above -- a pass that failed, an artefact
            # that is missing, a layout std that will not build -- leaves
            # every pass not yet waited for still running, each a cargo
            # and a Charon holding multiple GB. Below this point there is
            # nothing left to reap.
            for _, _, _, pending, _, _ in layout_passes:
                if pending is not None and pending.poll() is None:
                    pending.kill()
            if host_pass is not None and host_pass.poll() is None:
                host_pass.kill()
        # Persist the read set before computing the stamp it governs. The
        # artefacts are the oracle: repo-relative Local entries are exactly the
        # Rust sources Charon parsed, across the host translation and every
        # cross-target one folded in above, while proc macros, build scripts
        # and manifests are added by the cargo-metadata walk.
        file_table = artefact_local_readfiles(dest)
        merge_readfiles(file_table, layout_tables)
        assert_readfiles_resolve(eng.root, dest, file_table)
        write_readfiles(readfiles, file_table)

        # Every walk from here on reports on the tree AFTER the builds above,
        # so the memoised pre-build input sets are dropped first. The three
        # readers below — the provenance closure, `window_writes` and the
        # post-build digest — exist to see a tree that moved while Charon ran,
        # and an input set carried over from before it would hide exactly the
        # movement they are looking for. They still share one walk with each
        # other, which is the intent.
        forget_collected_inputs()
        derived = set(fingerprint_inputs(eng, [crate], cargo_features))
        try:
            assert_artefact_readfiles_covered(dest, file_table, derived)
        except BaseException:
            invalidate_fingerprint(stamp_path, readfiles)
            forget_collected_inputs()
            raise

        # Closing half of the provenance pair, and the sidecar write.
        #
        # This precedes the two guards below, both of which can end the run,
        # because the artefact at `dest` has ALREADY been rewritten by the
        # Charon build above. The invariant worth holding is "the provenance
        # beside an artefact describes the bytes in it" — leaving the write
        # until after the guards would, on a guard failure, leave the previous
        # run's provenance sitting beside this run's bytes, which is the same
        # lie the skip path is careful not to tell.
        #
        # It is also the reason this is not gated: an artefact the stamp
        # refuses is exactly the one whose provenance a reader needs.
        head_after = git_head(eng.root)
        finished = datetime.datetime.now().isoformat(timespec="seconds")
        dirty_status, dirty_entries = git_dirty(eng.root)
        closure = {
            path.as_posix()
            for path in fingerprint_inputs(eng, [crate], cargo_features)
        }
        provenance_path = dest.with_suffix(dest.suffix + ".provenance")
        provenance_path.write_text(
            provenance_for(
                crate=crate,
                head_before=head_before,
                head_after=head_after,
                started=started,
                finished=finished,
                dirty_status=dirty_status,
                dirty=dirty_entries,
                dirty_before=dirty_before,
                closure=closure,
            )
            + "\n"
        )
        if head_after != head_before:
            print(
                f"    NOTE: HEAD moved during this build"
                f" ({head_before[:11]} -> {head_after[:11]}); recorded in"
                f" {provenance_path.name}. Whether that matters is answered by"
                f" the source-hash check below, not by this line."
            )

        # `stamp` was computed before the Charon build. The post-build source
        # hash legitimately adopts the new artefact's read set, so movement is
        # checked through the unchanged wide closure and external channels. If
        # either moved, the artefact straddles two trees and belongs to neither;
        # a stamp that looks authoritative is worse than none.
        # Sited BEFORE the stamp decision so it reports on both branches:
        # when the stamp is refused, "the tree moved" is exactly when a reader
        # wants the file names, and when it is accepted this is the only check
        # that saw the middle of the window at all.
        moved, covered, unresolved = window_writes(
            fingerprint_inputs(eng, [crate], cargo_features),
            eng.root,
            window_lo_ns,
            time.time_ns(),
        )
        print(
            f"    window writes: {len(moved)} candidate(s) over {covered} of"
            f" {covered + len(unresolved)} closure inputs"
        )
        for _, rel in moved:
            print(f"        {rel.as_posix()}")
        for line in unresolved:
            print(f"        UNRESOLVED {line}")
        if moved:
            print(
                "      Those paths were written while this artefact was being"
                " built. Candidates, not an accusation, and blind to an"
                " mtime-preserving restore."
            )
        current_stamp = stamp_for(
            eng,
            crate=crate,
            platform_key=platform_key,
            charon_stamp=charon_stamp,
            cargo_features=cargo_features,
            flags=flags,
            charon_flags=charon_flags,
            layout_targets=crate_layout_targets(eng, spec),
            layout_flags=layout_flags,
            artefacts=artefacts_fingerprint(eng, spec, dest_dir),
        )
        before = parse_stamp(stamp)
        after = parse_stamp(current_stamp)
        # Skip used `include_closure=False`. The stamp recomputed just
        # before Charon must carry a real `closure=` digest so a
        # closure-only edit during the multi-minute build is visible.
        if before.get("closure") == CLOSURE_UNCOMPUTED:
            raise SystemExit(
                f"extract-llbc.py: pre-build stamp for {crate} has"
                " closure=-; the post-skip stamp_for must compute a"
                " real closure digest before Charon starts"
            )
        closure_moved = before.get("closure") != after.get("closure")
        external_moved = before.get("external") != after.get("external")
        if closure_moved or external_moved:
            invalidate_fingerprint(stamp_path, readfiles)
            forget_collected_inputs()
            unstamped.append(crate)
            moved_fields = [
                field
                for field, moved in (
                    ("closure", closure_moved),
                    ("external", external_moved),
                )
                if moved
            ]
            print(
                f"    REFUSING to stamp {dest.name}: the tree moved during its"
                f" build ({', '.join(moved_fields)}), so no input hash"
                f" describes this artefact. Left unstamped — reported as"
                f" freshness UNKNOWN, not as fresh."
            )
            continue
        # The line ending is pinned, not left to the platform: the stamp is
        # compared against a string built in memory with "\n", so a writer
        # that translates makes the artefact describe the same tree
        # differently on Windows than everywhere else.
        with stamp_path.open("w", encoding="utf-8", newline="\n") as output:
            output.write(current_stamp + "\n")
        print(
            f"    wrote {dest} ({dest.stat().st_size} bytes),"
            f" stamped in {time.monotonic() - phase_started:.0f} s"
        )
        if store is not None:
            try:
                publish_llbc_store_entry(
                    store,
                    parse_stamp(current_stamp),
                    llbc_store_payload_files(eng, spec, dest_dir),
                )
            except Exception as exc:
                print(
                    f"warn: llbc store: publish failed for {crate}: {exc}",
                    file=sys.stderr,
                )

    print()
    if unstamped:
        raise SystemExit(
            "extract-llbc.py: the tree moved while extracting "
            + ", ".join(unstamped)
            + ".\n  Those artefacts are written but deliberately unstamped."
            "\n  Re-run with --force once the tree is quiet."
        )
    if store is not None:
        try:
            maybe_prune_llbc_store(store)
        except Exception as exc:
            print(f"warn: llbc store prune failed: {exc}", file=sys.stderr)
    print(f"all extractions complete. artefacts under: {dest_dir}")


def check(eng: Engine, args: argparse.Namespace) -> None:
    """Refuse an artefact whose stamp does not match the tree it sits beside.

    Or whose own bytes are not the ones that stamp was written beside: a
    Charon pass writes straight to the artefact's final path, so a run killed
    part-way leaves a short file under a stamp that still describes unmoved
    sources. See `artefacts_fingerprint`.

    `extract` already computes this comparison — it is the `skipping <crate>
    (fingerprint unchanged)` test — but only a caller who runs the extractor
    ever benefits from it. A consumer that opens `<out_dir>/<crate>.ullbc` as
    found gets whatever was last written there, and a stale artefact does not
    fail: it answers, in detail, about sources that no longer exist. So this is
    the same comparison, run by the consumer, reported through the exit status.
    Nonzero means DO NOT READ the artefact.

    Every way of reaching the end WITHOUT having compared the stamp against a
    freshly computed one is a refusal, not a pass: an absent stamp, an empty
    stamp, a stamp missing fields. Treating "nothing to compare" as "nothing
    wrong" is what makes a gate decorative — a harness reads the exit status,
    not the output.

    Two things are deliberately not gated, and one is simply not covered here
    — the list is written out because an incomplete "what this does not check"
    reads as a complete one, which is the same shape as an allow-list that
    silently omits a member:

      * `features=` is replayed out of the stamp rather than compared. The
        stamp records the configuration the artefact was built under, and the
        question here is whether the artefact is current FOR that
        configuration; comparing it against this process's `CARGO_FEATURES`
        would refuse a byte-correct artefact whenever the caller left the
        default in place. Replaying keeps `flags=`, `charon_flags=`,
        `layout_flags=`, `source=`, `closure=` and `external=` real comparisons
        — all six are recomputed from the replayed value. (`external=` belongs on
        that list because the dependency walk it feeds off is itself run
        per feature set, so a feature that pulls in a patched path dep
        changes which out-of-root inputs exist.)
      * `closure=` is a residual warning when `source=` still matches. The
        artefact's own file table, proc macros, build scripts and manifests are
        unchanged, but something else in the cargo dependency closure moved.
        A new trait impl outside the read set can still affect method
        resolution, so the warning tells the reader when to re-extract without
        making unrelated closure edits fail the build.
      * artefact and stamp mtimes are printed, never gated. `extract` writes
        both in one pass, and the source digest already answers the question a
        timestamp only approximates.

    Nothing here re-extracts. Re-extraction runs a whole-crate Charon build and
    writes into the working tree, so it stays a human's scheduling decision and
    the refusal names the command instead. No cost figure is quoted anywhere in
    this function: what it costs depends on how warm the cargo cache is, and a
    number carried over from one cold run reads as measured when it is not.
    `LLBC_DEST` points the check at a directory other than the driver's default,
    which is what a caller who must not disturb the live artefacts uses.

    Exit 0 has to be EARNED, never fallen into. `verified` records the crates
    that positively matched, and the epilogue refuses unless every requested
    crate is in it — so an empty crate list, or any future branch that forgets
    to file a failure, refuses instead of reporting success by default.
    """
    platform_key, charon_dest, _ = charon_paths(eng.charon_root)
    charon_stamp = charon_version(charon_dest)
    dest_dir = llbc_dest(eng.out_dir, eng.root)
    # `crates` is what was requested, and the epilogue's `unaccounted` check
    # depends on it staying that way. Rebuilding this list from resolved specs
    # would drop a name the loop could not process before the epilogue ever saw
    # it, turning "this crate was never confirmed" back into silent success.
    crates = args.crates or eng.default_crates

    stale: list[str] = []
    verified: set[str] = set()
    # Feature set to re-extract each crate under: the one its stamp records, so
    # a mixed set does not collapse into one wrong remedy command. A crate whose
    # stamp never got read keeps this process's feature set.
    stamped_features: dict[str, str] = {crate: eng.cargo_features for crate in crates}

    for crate in crates:
        spec = eng.spec(crate)
        dest = dest_dir / spec.output_name
        stamp_path = dest.with_suffix(dest.suffix + ".fingerprint")
        print(f"=== checking {crate} -> {dest} ===")

        if not dest.exists():
            stale.append(f"{crate}: no artefact at {dest}")
            continue
        if dest.stat().st_size == 0:
            stale.append(
                f"{crate}: artefact {dest} is 0 bytes — Charon aborted before "
                f"writing it"
            )
            continue
        print(f"    artefact {dest.stat().st_size} bytes, mtime {file_mtime(dest)}")
        # Before the stamp arms, all of which can `continue`. Provenance is
        # most useful on exactly the paths that refuse.
        report_provenance(dest)

        # `artefacts=` covers these too, and says more about them, but this
        # line is the one that names what a reader loses — and the only one
        # that survives the arms below, each of which `continue`s on a stamp
        # that cannot be compared at all.
        for target in crate_layout_targets(eng, spec):
            sidecar = dest_dir / spec.layout_sidecar_name(target)
            if not sidecar.exists() or sidecar.stat().st_size == 0:
                stale.append(
                    f"{crate}: {target} layout sidecar {sidecar} is missing or "
                    f"empty — field offsets for that target would be read out "
                    f"of the extraction host's layouts"
                )

        if not stamp_path.exists():
            stale.append(
                f"{crate}: no fingerprint stamp at {stamp_path} — the artefact "
                f"records nothing about the sources it came from, so there is "
                f"nothing to compare it against"
            )
            continue
        stamp_bytes = stamp_path.read_bytes()
        if not stamp_bytes:
            stale.append(
                f"{crate}: fingerprint stamp {stamp_path} is 0 bytes — it "
                f"records no source digest, so it cannot match the tree"
            )
            continue
        # Bytes and not `read_text`, so a truncated or foreign stamp is seen
        # as it is — but a stamp written before the writer above pinned its
        # line endings holds CRLF on Windows, and comparing that against an
        # LF string reports every one of them as carrying content this
        # engine did not write.
        text = stamp_bytes.decode("utf-8", errors="replace").replace(
            "\r\n", "\n"
        )
        if not text.strip():
            stale.append(
                f"{crate}: fingerprint stamp {stamp_path} holds only whitespace"
            )
            continue
        print(f"    stamp {len(stamp_bytes)} bytes, mtime {file_mtime(stamp_path)}")

        recorded = parse_stamp(text)
        missing = [key for key in STAMP_KEYS if key not in recorded]
        if missing:
            # Order matters: the LIKELIEST cause is listed first, and it is the
            # one the two original guesses both excluded. Adding a key to
            # STAMP_KEYS makes EVERY stamp already on disk land here at once,
            # so right after such a change this arm fires for a reason that is
            # neither "a foreign tool wrote it" nor "it was truncated" — the
            # stamp is this engine's own output from before the field existed.
            # Saying only those two sends a reader hunting a corruption that is
            # not there. Measured after `external=` was added: all five live
            # pyre stamps refuse through here, none of them damaged.
            stale.append(
                f"{crate}: fingerprint stamp {stamp_path} has no "
                f"{', '.join(missing)} field — written before this engine "
                f"recorded that field, or by another tool, or truncated. "
                f"Re-extract either way; the artefact predates what the stamp "
                f"is now required to cover, so it cannot be compared"
            )
            continue

        features = recorded["features"]
        stamped_features[crate] = features
        flags = crate_flags(spec, features)
        expected = stamp_for(
            eng,
            crate=crate,
            platform_key=platform_key,
            charon_stamp=charon_stamp,
            cargo_features=features,
            flags=flags,
            charon_flags=charon_crate_flags(spec, features),
            layout_targets=crate_layout_targets(eng, spec),
            layout_flags=crate_layout_flags(spec, features, flags),
            artefacts=artefacts_fingerprint(eng, spec, dest_dir),
        )
        skip_ok, closure_moved = stamp_skip_ok(text, expected)
        if skip_ok:
            if closure_moved:
                print(
                    f"    WARNING: {crate}.ullbc's declared inputs are unchanged, "
                    "but something else in its dependency closure moved; if a trait "
                    "impl or proc macro changed, re-extract"
                )
            print(f"    fingerprint matches the tree (source={recorded['source']})")
            verified.add(crate)
            continue

        want = parse_stamp(expected)
        differing = [key for key in STAMP_KEYS if recorded[key] != want.get(key)]
        if not differing:
            # The text differs while every modelled field agrees, so the stamp
            # carries content this engine does not write. Reporting nothing
            # here would turn a mismatch into a silent pass.
            stale.append(
                f"{crate}: fingerprint stamp {stamp_path} does not match "
                f"byte-for-byte although every field agrees — it carries "
                f"content this engine did not write"
            )
            continue
        for key in differing:
            render = PACKED_FIELD_DIFFS.get(key)
            if render is not None:
                stale.extend(render(crate, recorded[key], want.get(key, "")))
                continue
            stale.append(
                f"{crate}: {key}: artefact says {recorded[key]!r}, "
                f"tree says {want.get(key)!r}"
            )

    # Files in the artefact directory that no crate claims. Reported, never
    # gated: nothing reads them, so they cannot make a read unsound — but the
    # directory is what a person browses, and a sidecar left behind by a
    # retired layout target sits beside the live ones looking every bit as
    # current. Scanned against EVERY spec's names rather than the requested
    # crates, so checking one crate does not report the other three's
    # artefacts as strays.
    claimed = set()
    for spec in eng.specs.values():
        claimed.add(spec.output_name)
        claimed.update(
            spec.layout_sidecar_name(target)
            for target in crate_layout_targets(eng, spec)
        )
    strays = sorted(
        path.name for path in dest_dir.glob("*.ullbc") if path.name not in claimed
    )
    if strays:
        print()
        print(f"    NOTE: {len(strays)} file(s) under {dest_dir} no crate claims:")
        for name in strays:
            print(f"        {name}")

    # Exit 0 requires a positive match for every requested crate, and at least
    # one crate to have been requested. Both halves matter: an empty list has no
    # unaccounted crates, so testing only for those would let "checked nothing"
    # through as success — the exact shape this whole function exists to refuse.
    unaccounted = [crate for crate in crates if crate not in verified]
    if not stale and (not crates or unaccounted):
        stale.append(
            "checked nothing for: " + (" ".join(unaccounted) or "(no crates requested)")
        )
    if not stale:
        print()
        print(f"llbc artefacts current: {' '.join(crates)}")
        return

    driver = sys.argv[0] or "scripts/extract-llbc.py"
    # The per-crate lines above went to stdout, which is block-buffered when
    # this runs under a harness; flush so the verdict does not land before the
    # evidence it was drawn from.
    sys.stdout.flush()
    print(file=sys.stderr)
    print("=" * 72, file=sys.stderr)
    print("LLBC STALE — do not read these artefacts", file=sys.stderr)
    for line in stale:
        print(f"  {line}", file=sys.stderr)
    print(
        "\n"
        "  Re-extract before reading. This runs a whole-crate Charon build and\n"
        "  writes into the working tree, so nothing here runs it for you —\n"
        "  how long it takes depends on how warm the cargo cache is:",
        file=sys.stderr,
    )
    for features in dict.fromkeys(stamped_features.values()):
        group = [c for c in crates if stamped_features[c] == features]
        print(
            f"      CARGO_FEATURES={features} python3 {driver} --force "
            f"{' '.join(group)}",
            file=sys.stderr,
        )
    # Two things a reader needs at exactly this moment and can find nowhere
    # near it: what makes the re-extraction stick, and what they may still do
    # meanwhile. Both are printed here rather than left in a doc, because this
    # banner is where someone learns they are blocked.
    print(
        "\n"
        "  WARNING: re-extracting while a declared input is dirty re-stales the result\n"
        "  the moment that edit changes again. The fingerprint is over the\n"
        "  WORKING TREE, so an uncommitted in-closure edit is invisible to\n"
        "  `git log` and to a --check taken a minute earlier. On a shared\n"
        "  checkout, require this intersection to be EMPTY first:\n"
        f"      python3 {driver} --list-inputs {' '.join(crates)} | sort -u > cl.txt\n"
        "      git status --porcelain | sed 's/^...//' | sort -u        > dirty.txt\n"
        "      comm -12 dirty.txt cl.txt          # must print nothing\n"
        "  Both sides are repo-relative, so they compare directly. If it comes\n"
        "  back empty, confirm the instrument is live before believing it —\n"
        "  `Cargo.lock` is an input, so seeding it into dirty.txt must produce\n"
        "  a hit.\n"
        "\n"
        "  Meanwhile: COMPILING is available under PYRE_LLBC_STRICT=0, which\n"
        "  demotes this to a warning rather than silencing it. MEASURING is\n"
        "  not. Stale offsets can name the wrong bytes, and a miscompiled field\n"
        "  access returns a NUMBER rather than an error, so an unsound run is\n"
        "  indistinguishable from a real one. The criterion is whether the work\n"
        "  READS FIELD OFFSETS, not which crate it lives in: a pure source-text\n"
        "  audit is exempt, anything that runs a trace never is.",
        file=sys.stderr,
    )
    print("=" * 72, file=sys.stderr)
    raise SystemExit(1)


def self_test_ullbc_files_table() -> None:
    """Exercise chunk boundaries, escapes and Local-file filtering."""
    table = [
        {
            "id": 0,
            "name": {"Local": "crate/src/lib.rs"},
            "crate_name": "crate",
            "contents": 'brackets ][ and an escaped quote: "',
        },
        {
            "id": 1,
            "name": {"Local": "/rustc/library/core/src/lib.rs"},
            "crate_name": "core",
            "contents": "",
        },
        {
            "id": 2,
            "name": {"Virtual": "generated.rs"},
            "crate_name": "generated",
            "contents": "",
        },
        # The spellings Charon uses for the toolchain's own sources on Windows.
        # Neither reads as absolute to `PurePosixPath`, and the drive-relative
        # one does not read as absolute to `PureWindowsPath` either, so both
        # reached the read set before `is_repo_relative_name` judged them by
        # their drive instead. `root / value` then discards the root and
        # resolves against that drive, which is how `std/src/lib.rs` came to be
        # scanned for `include*!`.
        {
            "id": 3,
            "name": {"Local": "C:Users/runner/.rustup/toolchains/x/std/src/lib.rs"},
            "crate_name": "std",
            "contents": "",
        },
        {
            "id": 4,
            "name": {"Local": "C:/Users/runner/.rustup/toolchains/x/core/src/lib.rs"},
            "crate_name": "core",
            "contents": "",
        },
        # A repo-relative name separated the way the windows leg spells it,
        # carrying the `../` chain `_immutables_map/mod.rs` uses to reach
        # `lib_pypy/`. `posixpath` reads no separator here, so before
        # `normalized_local_name` the whole thing stayed one component and the
        # `..` survived into the read set, where it matched nothing git lists.
        {
            "id": 5,
            "name": {"Local": r"crate\src\module\m\..\..\..\..\shared\app.py"},
            "crate_name": "crate",
            "contents": "",
        },
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "probe.ullbc"
        path.write_text(
            json.dumps({"translated": {"files": table}, "tail": []}, separators=(",", ":"))
        )
        parsed = ullbc_files_table(path, chunk_size=7)
        local = artefact_local_readfiles(path)
    expected = {"crate": {Path("crate/src/lib.rs"), Path("shared/app.py")}}
    if parsed != table or local != expected:
        raise SystemExit(
            f"self-test FAILED: files-table parser returned parsed={parsed!r}, local={local!r}"
        )
    print("  files-table streaming parser: ok")


def self_test_readfiles_rooting() -> None:
    """Merge a cross-target read set, then refuse one rooted at a nested crate."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "crate" / "src").mkdir(parents=True)
        (root / "crate" / "src" / "lib.rs").write_text("")
        (root / "crate" / "src" / "wasm.rs").write_text("")
        dest = root / "probe.ullbc"

        # What the layout passes contribute: a source only the cross-target
        # translation reached, under a crate the host table already lists, plus
        # a crate the host table does not mention at all.
        merged = {"crate": {Path("crate/src/lib.rs")}}
        merge_readfiles(
            merged,
            {"crate": {Path("crate/src/wasm.rs")}, "dep": {Path("crate/src/lib.rs")}},
        )
        expected = {
            "crate": {Path("crate/src/lib.rs"), Path("crate/src/wasm.rs")},
            "dep": {Path("crate/src/lib.rs")},
        }
        if merged != expected:
            raise SystemExit(f"self-test FAILED: merge_readfiles produced {merged!r}")
        assert_readfiles_resolve(root, dest, merged)

        # `majit/charon-corpus` carries its own `Cargo.lock`, so Charon roots
        # its names at that crate and reports `src/lib.rs`. Read from here the
        # name resolves nowhere, and a set of those would hash nothing.
        try:
            assert_readfiles_resolve(root, dest, {"nested": {Path("src/lib.rs")}})
        except SystemExit as refusal:
            if "src/lib.rs" not in str(refusal):
                raise SystemExit(f"self-test FAILED: refusal omits the entry: {refusal}") from None
        else:
            raise SystemExit("self-test FAILED: a crate-rooted read set was accepted")
    print("  read-set rooting and cross-target merge: ok")


def self_test_artefact_tokens() -> None:
    """Drive `artefact_token` over states a live artefact cannot be caught in.

    The failure `artefacts=` exists for — a write cut short — leaves a file
    that is neither absent nor 0 bytes, and that is the one state the artefact
    checks around it cannot tell from a complete extraction.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "probe.ullbc"
        path.write_bytes(b"x" * 4096)
        whole = artefact_token(path)
        path.write_bytes(b"x" * 4095)
        short = artefact_token(path)
        path.write_bytes(b"y" * 4096)
        swapped = artefact_token(path)
        path.unlink()
        gone = artefact_token(path)
    if gone != "missing":
        raise SystemExit(
            f"self-test FAILED: artefact_token reports {gone!r} for a file "
            f"that is not there"
        )
    if len({whole, short, swapped, gone}) != 4:
        raise SystemExit(
            f"self-test FAILED: artefact_token does not separate {whole!r} "
            f"(whole), {short!r} (truncated), {swapped!r} (same size, other "
            f"bytes) and {gone!r} (absent)"
        )
    if describe_artefact(short) == describe_artefact(swapped):
        raise SystemExit(
            f"self-test FAILED: a truncated and a swapped artefact both read "
            f"as {describe_artefact(short)!r}"
        )
    print("  artefact tokens separate whole/truncated/swapped/absent: ok")


def self_test_provenance_rendering() -> None:
    """Drive `dirty_report_lines` over cases the live sidecars cannot produce.

    Both of its conditional lines are silent on every artefact this repository
    has ever stamped — the delta is zero and nothing is omitted — so `--check`
    against the real tree exercises neither, and a dead branch would read
    exactly like a correct one. The movement line especially: it exists for a
    build that straddled a peer's edit, which is the case nobody can arrange on
    demand and everybody needs reported.

    The three negatives are load-bearing, not padding. A renderer that emitted
    the movement line unconditionally satisfies every positive row here and is
    worse than no line at all, because it accuses a clean build of moving; and
    `unknown` is not zero, so a renderer that coerced it would report a delta
    against a count it never had.
    """
    base = {
        "dirty_before": "3",
        "dirty_in_closure": "0",
        "dirty_outside": "3",
        "dirty_omitted": "0",
    }
    cases = [
        ("steady tree", {}, False, False),
        ("grew by 2", {"dirty_in_closure": "1", "dirty_outside": "4"}, True, False),
        ("shrank by 2", {"dirty_before": "5"}, True, False),
        ("moved inside the closure", {"dirty_in_closure": "1"}, True, False),
        ("pre-build count unknown", {"dirty_before": "unknown"}, False, False),
        ("listing truncated", {"dirty_omitted": "7"}, False, True),
        ("moved AND truncated", {"dirty_before": "9", "dirty_omitted": "4"}, True, True),
    ]

    print("  provenance rendering")
    failures = []
    for label, over, want_moved, want_omitted in cases:
        text = "\n".join(dirty_report_lines({**base, **over}, []))
        got_moved = "MOVED during the build" in text
        got_omitted = "do not count" in text
        verdict = "ok" if (got_moved, got_omitted) == (want_moved, want_omitted) else "FAILED"
        print(f"    {verdict:<6} {label}: moved={got_moved} truncated={got_omitted}")
        if verdict == "FAILED":
            failures.append(
                f"{label!r}: expected moved={want_moved} truncated={want_omitted},"
                f" got moved={got_moved} truncated={got_omitted}"
            )
    if failures:
        sys.stdout.flush()  # as in check(): the verdict must not outrun its evidence
        for line in failures:
            print(f"self-test FAILED: {line}", file=sys.stderr)
        raise SystemExit(1)


def self_test_nested_repo_detection() -> None:
    """Drive the nested-repo refusal on a synthetic tree, BOTH arms.

    `self_test` below draws its probe from `fingerprint_inputs`, i.e. from a
    directory the fingerprint already covers, so it can never place one where
    coverage fails — it would keep passing whatever this guard did. Hence a
    separate case with its own tree.

    Runs real `git` against a real nested repository rather than handing
    `nested_repo_entries` a list of strings: the load-bearing fact is what git
    DOES with a nested repo, and a list written here would only ever agree with
    what its author believed. If a future git stops emitting the trailing
    slash, this fails; a string fixture would not.

    The BENIGN arm carries as much weight as the hazard one. This refusal is
    fail-closed, so a predicate that flagged every untracked directory would
    satisfy a detection-only test and then stop every extraction that follows a
    new module directory.

    Everything happens inside a temporary directory. `self_test` writes its
    probe into the shared worktree, a brief real mutation; doing that with a
    nested repo would be a worse one, because a peer extracting in that instant
    would take this very refusal instead of merely seeing a moved hash.
    """
    def ls_others(root: Path) -> list[str]:
        result = subprocess.run(
            ["git", "ls-files", "-z", "--others", "--exclude-standard", "--", "covered"],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
        )
        return [raw.decode("utf-8") for raw in result.stdout.split(b"\0") if raw]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        def git(*args: str, cwd: Path | None = None) -> None:
            subprocess.run(
                ["git", *args],
                cwd=cwd or root,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        git("init", "-q", ".")
        git("config", "user.email", "self-test@example.invalid")
        git("config", "user.name", "llbc self-test")
        (root / "covered").mkdir()
        (root / "covered" / "tracked.rs").write_text("// tracked\n")
        git("add", "-A")
        git("commit", "-q", "-m", "base")

        # Benign: an ordinary untracked directory holding a new module. This is
        # also the liveness control — the leg must LIST it, so a git that
        # returned nothing at all cannot pass this test by being empty.
        (root / "covered" / "plain").mkdir()
        (root / "covered" / "plain" / "new_module.rs").write_text("// new\n")
        benign_entries = ls_others(root)
        benign_flagged = nested_repo_entries(benign_entries)

        # Hazard: a nested repository under the same covered path.
        nested = root / "covered" / "nested"
        nested.mkdir()
        git("init", "-q", ".", cwd=nested)
        (nested / "inside.rs").write_text("// invisible to the outer repo\n")
        hazard_entries = ls_others(root)
        hazard_flagged = nested_repo_entries(hazard_entries)

    print("    benign tree    " + " ".join(benign_entries))
    print(f"      flagged      {benign_flagged}")
    print("    + nested repo  " + " ".join(hazard_entries))
    print(f"      flagged      {hazard_flagged}")

    failures = []
    if "covered/plain/new_module.rs" not in benign_entries:
        failures.append(
            "the untracked leg did not list a new module — the test measured nothing"
        )
    if benign_flagged:
        failures.append(
            f"an ordinary untracked directory was flagged {benign_flagged} —"
            " this refusal would stop every extraction after a new module dir"
        )
    if hazard_flagged != ["covered/nested/"]:
        failures.append(
            f"a nested repository was not flagged: got {hazard_flagged} —"
            " the fingerprint would hash its whole subtree as one constant"
        )
    if failures:
        sys.stdout.flush()  # as in check(): the verdict must not outrun its evidence
        for line in failures:
            print(f"self-test FAILED: {line}", file=sys.stderr)
        raise SystemExit(1)
    print("\nself-test passed: nested-repo refusal (hazard flagged, benign not)")


def self_test_derived_include_closed(
    eng: Engine, crates: list[str], cargo_features: str
) -> None:
    """Prove the derived input set is closed under `include*!`.

    The derived set comes from the artefact's file table, and that table lists
    SOURCE files. `include_str!` and `include_bytes!` embed a file's contents
    as a constant without making it one, so the file carries no span and never
    appears there — while editing it still changes what the crate compiles.
    `include_closure` is what recovers those, and it is the only leg of
    `_collect_inputs` the derived mode could drop without any hash moving,
    which is why it gets an assertion of its own rather than a comment.

    Not an A/B/A: there is nothing to mutate. The property is structural, so
    it is read off the set itself, and it fails exactly when the closure stops
    being applied to the derived inputs.
    """
    for crate in crates:
        derived = set(fingerprint_inputs(eng, [crate], cargo_features))
        missing = sorted(include_closure(eng.root, derived) - derived)
        print(f"    include-closed  {crate}: {len(derived)} input(s)")
        if missing:
            sys.stdout.flush()  # as in check(): the verdict must not outrun its evidence
            print(
                f"self-test FAILED: {crate}'s derived input set is not closed under"
                f" `include*!` — these are compiled in but unhashed:",
                file=sys.stderr,
            )
            for path in missing:
                print(f"    {path.as_posix()}", file=sys.stderr)
            raise SystemExit(1)
    print(f"self-test passed: derived inputs include-closed: {' '.join(crates)}")


def self_test(eng: Engine, crates: list[str], cargo_features: str) -> None:
    """A/B/A the wide closure hash against a new untracked Rust source.

    The blind spot being guarded is silent by construction: `git ls-files` lists
    tracked paths only, so before `fingerprint_inputs` folded in `--others
    --exclude-standard`, adding a brand-new module left the hash unmoved and
    `--check` blessed an artefact built from different source. A wrong answer
    there prints nothing, so the guard needs a demonstration rather than a
    reading of the patch. Both legs are required: that the hash *moves* for the
    new file, and that removing the probe *restores* it, so a change which
    invalidated every stamp cannot pass by refusing everything.

    The probe exists for one pair of fingerprint calls and is removed in a
    `finally`. On a shared worktree that is a real if brief mutation — anyone
    fingerprinting the same crate in that instant sees the moved closure hash.
    """
    inputs = closure_inputs(eng, crates, cargo_features)
    # Deepest `.rs`: exact-file pathspecs (Cargo.toml, target roots, build.rs)
    # sit shallow, so a deeply nested file was listed by a `dir/` pathspec and
    # its directory is therefore covered for new files too.
    covered = max((p for p in inputs if p.suffix == ".rs"), key=lambda p: (len(p.parts), p.as_posix()), default=None)
    if covered is None:
        raise SystemExit(f"self-test: no .rs input for {' '.join(crates)}")
    probe = eng.root / covered.parent / "__llbc_self_test_probe.rs"
    if probe.exists():
        raise SystemExit(f"self-test: {probe} already exists; refusing to clobber it")
    fallback = any(load_readfiles(eng, crate) is None for crate in crates)
    source_before = source_fingerprint(eng, crates, cargo_features)
    closure_before = closure_fingerprint(eng, crates, cargo_features)
    try:
        probe.write_text("// transient probe written by --self-test; safe to delete\n")
        # Both legs of this test are a deliberate change to the input SET, which
        # is the one thing `_collect_inputs` memoises. Reading a carried-over set
        # here would report the guard blind and the removal residual — a failure
        # printed by the cache rather than by the code under test.
        forget_collected_inputs()
        source_moved = source_fingerprint(eng, crates, cargo_features)
        closure_moved = closure_fingerprint(eng, crates, cargo_features)
        listed = probe.relative_to(eng.root) in set(
            closure_inputs(eng, crates, cargo_features)
        )
    finally:
        probe.unlink(missing_ok=True)
        forget_collected_inputs()
    source_after = source_fingerprint(eng, crates, cargo_features)
    closure_after = closure_fingerprint(eng, crates, cargo_features)
    print(f"    probe          {probe.relative_to(eng.root).as_posix()}")
    print(f"    source before  {source_before}")
    print(f"    source +probe  {source_moved}  (wide fallback: {fallback})")
    print(f"    closure before {closure_before}")
    print(f"    closure +probe {closure_moved}  (listed as an input: {listed})")
    failures = []
    if closure_moved == closure_before:
        failures.append("a new untracked .rs did not move closure= — the residual guard is blind")
    if fallback != (source_moved != source_before):
        failures.append(
            "source= did not follow its bootstrap rule — it must use the wide set only "
            "when a requested crate has no .readfiles"
        )
    if source_after != source_before or closure_after != closure_before:
        failures.append("removing the probe did not restore both fingerprints — it left residue")
    if failures:
        sys.stdout.flush()  # as in check(): the verdict must not outrun its evidence
        for line in failures:
            print(f"self-test FAILED: {line}", file=sys.stderr)
        raise SystemExit(1)
    print(f"\nself-test passed: {' '.join(crates)}")


def run_cli(
    specs: dict[str, CrateSpec],
    default_crates: list[str],
    *,
    root: Path,
    out_dir: Path,
    extraction_abi: str,
    base_pathspecs: list[str] | None = None,
    charon_root: Path | None = None,
    metadata_feature_crates: tuple[str, ...] = (),
    layout_targets: tuple[str, ...] = (),
    layout_target_rustflags: str = "",
    external_inputs: tuple[Path, ...] = (),
) -> None:
    """Argparse UX shared by every driver (positional crates, --force, …)."""
    # This file writes en/em dashes and arrows in its diagnostics, and Windows
    # hands a console the ANSI codepage (cp949 here), which cannot encode them.
    # A print then raises UnicodeEncodeError from inside the extraction, which
    # is how a "REFUSING to stamp" notice became a traceback that also skipped
    # the crates queued behind it. Encode as UTF-8 and never let a character
    # class decide whether an artefact gets built.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    all_crates = " ".join(specs)
    parser = argparse.ArgumentParser(
        description="Extract Charon ULLBC artefacts with source-fingerprint skip logic."
    )
    parser.add_argument("crates", nargs="*", help=f"known: {all_crates}")
    # The dispatch below takes the FIRST of these that is set and returns, so
    # accepting two would perform one and drop the other with no output saying
    # so — `--check --self-test` would run the self-test and report success for
    # a freshness check that never ran. That is the shape this file refuses
    # everywhere else, and argparse can refuse it here.
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--fingerprint", action="store_true")
    parser.add_argument(
        "--per-crate",
        action="store_true",
        help="with --fingerprint: one `<field>[<crate>]=` line per crate and"
        " field, instead of one digest over all the crates named. One process"
        " shares the cargo metadata and include walks every crate needs, where"
        " one process per crate repeats them",
    )
    modes.add_argument("--list-inputs", action="store_true")
    modes.add_argument(
        "--prune-store",
        action="store_true",
        help="remove unused LLBC store entries; never extracts",
    )
    parser.add_argument(
        "--store-max-age-days",
        type=int,
        default=None,
        help="with --prune-store: drop entries unused longer than N days "
        "(default LLBC_STORE_MAX_AGE_DAYS or 3)",
    )
    modes.add_argument(
        "--check",
        action="store_true",
        help="compare the existing artefacts against the tree and exit nonzero "
        "if any is stale; never extracts",
    )
    modes.add_argument(
        "--self-test",
        action="store_true",
        help="prove the source fingerprint still sees a new untracked .rs; "
        "writes and removes one transient probe file, never extracts",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=os.environ.get("LLBC_FORCE_REEXTRACT") == "1",
    )
    args = parser.parse_args()

    # Default feature set is `dynasm`, matching the default JIT backend. The
    # ULLBC that feeds trace codegen is backend-agnostic — `dynasm` and
    # `cranelift` extraction yield byte-identical generated code — so the
    # lighter backend skips compiling the cranelift-codegen tree (~33 crates)
    # the dynasm build never needs. A driver whose crates ignore `{features}`
    # is unaffected by this default. Override with `CARGO_FEATURES`.
    cargo_features = os.environ.get("CARGO_FEATURES", "dynasm")
    # `LLBC_LAYOUT_TARGETS` overrides the driver's cross-target layout set
    # (comma-separated); the empty string extracts host layouts only.
    layout_override = os.environ.get("LLBC_LAYOUT_TARGETS")
    if layout_override is not None:
        layout_targets = tuple(t.strip() for t in layout_override.split(",") if t.strip())
    eng = Engine(
        specs=specs,
        default_crates=default_crates,
        root=root,
        out_dir=out_dir,
        base_pathspecs=list(base_pathspecs) if base_pathspecs else ["Cargo.lock", "Cargo.toml"],
        charon_root=charon_root or root,
        cargo_features=cargo_features,
        extraction_abi=extraction_abi,
        metadata_feature_crates=metadata_feature_crates,
        layout_targets=layout_targets,
        layout_target_rustflags=layout_target_rustflags,
        external_inputs=tuple(external_inputs),
    )
    # Before ANY subcommand, including the read-only instruments: an inert
    # pathspec makes `--list-inputs` and `--fingerprint` report a coverage the
    # stamp does not have, and those are the two outputs a reader trusts when
    # deciding whether the guard is working.
    refuse_inert_pathspecs(eng)

    crates = args.crates or default_crates
    if args.list_inputs:
        for path in fingerprint_inputs(eng, crates, cargo_features):
            print(path.as_posix())
        # Marked, because the two channels are hashed into different stamp
        # fields and an unmarked union would read as one list of repo-relative
        # paths — which is what made the out-of-root inputs easy to miss. The
        # root is named on every line because it, not the file, is the unit
        # `external=` records a digest for and `check` reports a move against.
        for root_label, files in external_input_groups(eng, crates, cargo_features):
            if not files:
                # A declared root the filter emptied. It still holds a digest,
                # so listing nothing for it would under-report the coverage.
                print(f"external[{root_label}]: (no files)")
            for label, _ in files:
                print(f"external[{root_label}]:{label}")
        return
    if args.fingerprint:
        # BOTH fields. Printing only `source=` here previously cost a session:
        # three legs returned identical hashes that were consistent with every
        # hypothesis, because the field under test was not in the output.
        groups = (
            [(f"[{crate}]", [crate]) for crate in crates]
            if args.per_crate
            else [("", crates)]
        )
        dest_dir = llbc_dest_path(eng.out_dir, eng.root)
        for suffix, group in groups:
            print(f"source{suffix}={source_fingerprint(eng, group, cargo_features)}")
            print(f"closure{suffix}={closure_fingerprint(eng, group, cargo_features)}")
            print(f"external{suffix}={external_fingerprint(eng, group, cargo_features)}")
            # The one field here that describes an OUTPUT.  Without it the
            # build-time gate -- which asks this subcommand and nothing else --
            # accepts a truncated or swapped artefact whose sources have not
            # moved, which is the case `artefacts=` was written for.  Absent
            # files answer `missing`, so asking before extraction (the CI cache
            # key does) costs nothing and reads as "not there yet".
            print(
                f"artefacts{suffix}="
                + " ".join(
                    artefacts_fingerprint(eng, eng.spec(crate), dest_dir)
                    for crate in group
                )
            )
        return
    if args.per_crate:
        parser.error("--per-crate only applies with --fingerprint")
    if args.self_test:
        # The parser test runs first because it mutates only a temporary file;
        # the live probe below briefly changes the shared working tree.
        self_test_ullbc_files_table()
        # Also temporary-file only, and it covers the two steps between the
        # parser and the stamp: folding the cross-target tables in, and
        # refusing names that do not resolve from the repository root.
        self_test_readfiles_rooting()
        # Same reason, and the only cover `artefacts=` can have: every artefact
        # this repository has stamped was written whole.
        self_test_artefact_tokens()
        # Same reason, one artefact along: `--check` renders provenance on every
        # run and exercises neither of its conditional lines, because ordinary
        # sidecars describe steady, complete extractions.
        self_test_provenance_rendering()
        # Same reason again: this one builds its own repository in a
        # temporary directory and never touches the worktree.
        self_test_nested_repo_detection()
        # Reads the worktree without writing to it, so it precedes the probe.
        self_test_derived_include_closed(eng, crates, cargo_features)
        self_test(eng, crates, cargo_features)
        return
    if args.check:
        check(eng, args)
        return
    if args.prune_store:
        if args.store_max_age_days is None:
            args.store_max_age_days = llbc_store_max_age_days()
        prune_store_cli(eng, args)
        return
    extract(eng, args)


# This module is the shared extraction ENGINE. Every consumer imports it — the
# per-repo wrappers call `run_cli` with their own `SPECS` — and nothing has ever
# invoked it as a program.
#
# Without this block, running it directly defined `main()` and fell off the end:
# zero output, **exit 0**, for every argument including `--check`, `--fingerprint`
# and `--help`. A freshness check aimed here reported success without executing,
# which is worse than the staleness it was run to detect.
#
# The body refuses instead of dispatching. Adding a `main()` call would make
# direct invocation *work* and mint a second, under-specified entry point: the
# engine alone cannot know which crates to extract or where their sources live —
# that is exactly what a wrapper's `CrateSpec` and `PYRE_ROOT` resolution supply.
if __name__ == "__main__":
    raise SystemExit(
        "llbc_extract.py is the shared extraction ENGINE, not an entry point.\n"
        "Use the repository's scripts/extract-llbc.py wrapper instead.\n"
        "The wrapper supplies the CrateSpec table this module needs."
    )
