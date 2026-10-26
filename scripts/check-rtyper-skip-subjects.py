#!/usr/bin/env python3
"""Hold the two-phase prepass's Skip set to a named, shrinking list.

`dual_gate_publish_concretetypes` publishes a graph from the real rtyper
when the two-phase prepass cached it, and Skips it to the legacy walker
when the prepass failed.  Every Skip is a graph the legacy walker still
owns, so the Skip set is the remaining distance to deleting it -- the
epic is done when the set is empty.

A count alone cannot carry that: the same total can hide one family
closed and another opened, and a family that reappears after a port is
the failure most worth catching.  So this gate records the NAMES.  A
subject that was not in the baseline fails; a subject that has left it is
progress the baseline is meant to absorb.

The reading is a byproduct of a build that already happens.  The prepass
runs inside `pyre-jit-trace`'s build script, and `record_reason` -- whose
only call site is this gate -- prints one line per Skip as soon as the
decline census is on at any level.  `MAJIT_DECLINE_LOG=1` is therefore
the whole instrument: no verbose switch, no second pipeline, and no other
gate's per-event lines in the channel.

    python3 scripts/check-rtyper-skip-subjects.py

`--update` rewrites the baseline from this run, for a change that pays
subjects off.  `--no-build` skips the build and reads the census already
on disk, for a second look at a reading just taken.

The set is read against the corpus it was measured on.  The prepass reads
`target/llbc/*.ullbc`, and a re-extraction brings interpreter code the
baseline never saw: its graphs land in this set exactly like a regression
would.  A rise measured over a corpus the baseline does not name is a
ratchet failure; refresh with `--update` only after attributing the
additions to that move.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT / "target" / "release" / "build"
PACKAGE = "pyre-jit-trace"

# `decline::gate::DUAL_GATE`.  Spelled here rather than derived, because a
# rename on the Rust side must break this gate loudly: a prefix that stops
# matching would otherwise read as "no graph was skipped", which is the one
# answer this file exists to make unforgeable.
GATE = "codewriter::dual_gate_publish_concretetypes"

# The corpus this set was measured on.  Each crate's `source=` is the hash
# the STALE check itself compares, so naming the LLBC crates names the
# corpus without hashing the artefacts.
CORPUS_CRATES = (
    "majit-rlib",
    "pyre-object",
    "pyre-interpreter",
    "pyre-module",
    "pyre-jit",
)

# `decline::dump_to_stderr`'s banner. Its presence is what tells a census
# from a build that ran with the switch off -- both print no `[decline]`
# line when nothing declined, and only one of them is a reading.
CENSUS_BANNER = "=== majit decline census [analyze_pipeline]"

# `[decline] {gate} {class} {subject}: {reason}`.  The class vocabulary is
# bounded and contains no `": "`, and a graph name carries no space, so the
# subject is the last token before the first `": "`.
DECLINE = re.compile(r"^\[decline\] " + re.escape(GATE) + r" (?P<body>.*?): ", re.M)

# The summary prints a gate header even when its only row is the ACCEPT arm.
# That makes a genuine zero-Skip run distinguishable from a stale `GATE`
# spelling that matched no event at all.
GATE_SUMMARY = re.compile(r"^  " + re.escape(GATE) + r"$", re.M)

# The `_{hash:016x}` suffix `clause_spec.spec_leaf` appends to a
# specialized graph's name.
SPEC_DIGEST = re.compile(r"_[0-9a-f]{16}$")


def corpus_key() -> tuple[str, list[str]]:
    """What the prepass read, named so a moved corpus is visible.

    Returns the digest and the crates it could NOT name.  An unstamped
    artefact is not an unchanged one: the digest then covers less than the
    corpus does, so the miss is reported rather than folded into the hash,
    where it would read as a corpus that simply differs.

    A corpus this cannot fully name is still not grounds to soften the
    ratchet.  A moved-corpus failure turns on the key having *changed*;
    absence of evidence that it moved is not evidence that it did, and
    treating it as such would let deleting a stamp file excuse any
    addition.
    """
    parts = []
    unstamped = []
    for crate in CORPUS_CRATES:
        path = ROOT / "target" / "llbc" / f"{crate}.ullbc.fingerprint"
        source = ""
        if path.is_file():
            for line in path.read_text().splitlines():
                if line.startswith("source="):
                    source = line[len("source="):]
                    break
        if source:
            parts.append(f"{crate}={source}")
        else:
            unstamped.append(crate)
    if not parts:
        return "", unstamped
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16], unstamped


def platform_key() -> str:
    """The census is one number per platform, not one number.

    The interpreter's `cfg` arms differ across them, so a Linux corpus
    holds graphs a macOS one does not and the Skip sets cannot be
    satisfied from each other.
    """
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "darwin":
        return "darwin"
    return sys.platform


def baseline_path() -> pathlib.Path:
    """One file per platform, rather than one file with platform sections.

    The value of recording names instead of a count is that a pull request
    shows them: a family closed and a family opened are one line each in the
    diff.  Sections would put every platform's set in one diff and lose that,
    and a platform whose set is absent must read as "never measured here",
    not as an empty set some other platform's run satisfied.
    """
    return ROOT / "majit" / f"rtyper-skip-subjects.{platform_key()}.txt"


def scan_for_stderr() -> pathlib.Path:
    """The one build script stderr on disk that holds a census.

    Only for `--no-build`, where nothing names the directory.  A package can
    have several `build/<pkg>-<hash>/` directories -- one per feature
    resolution -- and most hold a stderr from some other build: the LLBC
    extraction pass writes placeholders there and censuses nothing.  So the
    file is chosen by the census summary being IN it: an mtime says when a
    file was touched, not what it records.
    """
    found = [
        (p.stat().st_mtime, p)
        for p in sorted(BUILD_DIR.glob(f"{PACKAGE}-*/stderr"))
        if CENSUS_BANNER in p.read_text(errors="replace")
    ]
    if not found:
        sys.exit(
            f"error: no build script stderr under {BUILD_DIR.relative_to(ROOT)}/"
            f"{PACKAGE}-*/ holds a `{CENSUS_BANNER}` line.\n"
            "  The census is off in every build on disk, so there is nothing "
            "to read. Take one by dropping --no-build."
        )
    return max(found)[1]


def run_build() -> pathlib.Path:
    """Build with the census on, and return the stderr cargo wrote it to.

    Cargo names the directory rather than this guessing it.  A cached build
    script still reports `build-script-executed` -- cargo replays the
    recorded output -- so this resolves the same path whether the script
    reran or cargo certified that it need not, and never falls back to
    another feature resolution's older, still-censused stderr.
    """
    env = dict(os.environ, MAJIT_DECLINE_LOG="1")
    proc = subprocess.run(
        ["cargo", "build", "--release", "-p", PACKAGE, "--message-format=json"],
        cwd=ROOT, env=env, capture_output=True, encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0:
        # The build script's own errors (a STALE corpus, most often) are on
        # cargo's stderr, so print it rather than only the exit code: a
        # reader who has to rerun to find out why has been told nothing.
        sys.exit(
            f"error: `cargo build --release -p {PACKAGE}` exited "
            f"{proc.returncode}\n{proc.stderr}"
        )
    out_dirs = []
    for line in proc.stdout.splitlines():
        if '"build-script-executed"' not in line:
            continue
        message = json.loads(line)
        if message.get("package_id", "").split("#")[0].endswith(PACKAGE) or (
            f"/{PACKAGE}#" in message.get("package_id", "")
        ):
            out_dirs.append(pathlib.Path(message["out_dir"]))
    if not out_dirs:
        sys.exit(
            f"error: cargo reported no build script for {PACKAGE}.\n"
            "  Nothing ran the prepass, so there is no census to read."
        )
    # `out_dir` is `<build>/<pkg>-<hash>/out`; the stderr sits beside it.
    return out_dirs[-1].parent / "stderr"


def parse(text: str) -> dict[str, set[str]]:
    """`class -> {subject}` for every Skip in the census.

    The caller separately requires [`GATE_SUMMARY`] when this returns empty,
    distinguishing a terminal zero-Skip reading from a gate that never ran.
    """
    found: dict[str, set[str]] = {}
    for m in DECLINE.finditer(text):
        body = m.group("body")
        cls, _, subject = body.rpartition(" ")
        if not cls:
            continue
        found.setdefault(cls, set()).add(subject)
    return found


def read_baseline(
    path: pathlib.Path | None = None,
) -> tuple[bool, dict[str, str], dict[str, set[str]]]:
    path = baseline_path() if path is None else path
    if not path.is_file():
        return False, {}, {}
    header: dict[str, str] = {}
    want: dict[str, set[str]] = {}
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            for field in line[1:].split():
                key, _, value = field.partition("=")
                if value:
                    header[key] = value
            continue
        if not line.strip():
            continue
        cls, _, subject = line.partition("\t")
        want.setdefault(cls, set()).add(subject)
    return True, header, want


def write_baseline(header: dict[str, str], got: dict[str, set[str]]) -> None:
    lines = [
        "# The graphs the two-phase prepass still Skips to the legacy walker.",
        "# Ratcheted by scripts/check-rtyper-skip-subjects.py: a name may",
        "# leave this list, and may not join it.  Empty is the epic's",
        "# done-when.",
        "# " + " ".join(f"{k}={v}" for k, v in sorted(header.items()) if v),
    ]
    for cls in sorted(got):
        for subject in sorted(got[cls]):
            lines.append(f"{cls}\t{subject}")
    baseline_path().write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--update", action="store_true",
                    help="rewrite the baseline from this run")
    ap.add_argument("--no-build", action="store_true",
                    help="read the existing build script stderr instead of rebuilding")
    args = ap.parse_args()

    stderr_path = scan_for_stderr() if args.no_build else run_build()
    if not stderr_path.is_file():
        sys.exit(f"error: {stderr_path} does not exist; the build script "
                 "produced no stderr.")
    census = stderr_path.read_text(errors="replace")
    if CENSUS_BANNER not in census:
        sys.exit(
            f"error: {stderr_path.relative_to(ROOT)} holds no "
            f"`{CENSUS_BANNER}` line.\n"
            "  The build script ran with the census off, so this file records "
            "nothing about what was declined."
        )
    got = parse(census)
    if not got and not GATE_SUMMARY.search(census):
        sys.exit(
            f"error: {stderr_path.relative_to(ROOT)} holds no `[decline] {GATE}` "
            "line and no summary row for that gate.\n"
            "  The census ran, but it never observed the dual gate under the "
            "spelling this checker pins; treating that as zero Skips would hide "
            "a renamed or unreachable gate."
        )

    corpus, unstamped = corpus_key()
    header = {"corpus": corpus, "platform": platform_key()}
    baseline_exists, recorded, want = read_baseline()

    if args.update:
        write_baseline(header, got)
        total = sum(len(s) for s in got.values())
        print(f"wrote {baseline_path().relative_to(ROOT)}: {total} subjects in "
              f"{len(got)} classes [{header['platform']}]")
        return 0

    if not baseline_exists:
        sys.exit(
            f"error: {baseline_path().relative_to(ROOT)} does not exist.\n"
            "  Seed it from a run on this platform: "
            "python3 scripts/check-rtyper-skip-subjects.py --update"
        )

    flat_got = {(c, s) for c, subjects in got.items() for s in subjects}
    flat_want = {(c, s) for c, subjects in want.items() for s in subjects}
    # Split the pair-level diff by NAME, because the name is what the
    # invariant is stated over.  A graph that lands in a different Skip
    # class shows up as one removal and one addition, and calling that
    # addition "newly Skipped" is false -- the name never left the set.
    # Relocation is the ordinary shape of progress here: a graph that
    # stops declining on its first blocker reaches a later one and is
    # categorised by that instead.
    def module_move_key(name: str) -> str:
        # `#1854` moved optional modules from `pyre-interpreter` to
        # `pyre-module`. The graph is the same subject; only the crate
        # prefix changed, so both spellings compare as one name.
        for prefix in (
            "pyre_interpreter::module::",
            "pyre_module::module::",
        ):
            if name.startswith(prefix):
                name = "module::" + name[len(prefix) :]
                break
        # `clause_spec.spec_leaf` ends a specialized graph's name with an
        # FNV-1a digest of the spelled trait refs and types. A corpus move
        # that respells one (a new impl, a changed path) mints a new digest
        # for the same `{leaf}__spec_{readable}` copy, so the digest is not
        # part of the subject.
        if "__spec_" in name:
            name = SPEC_DIGEST.sub("", name)
        return name

    pair_added = ({(c, module_move_key(s)) for c, s in flat_got}
                  - {(c, module_move_key(s)) for c, s in flat_want})
    pair_gone = ({(c, module_move_key(s)) for c, s in flat_want}
                 - {(c, module_move_key(s)) for c, s in flat_got})
    gone_names = {s for _, s in pair_gone}
    added_names = {s for _, s in pair_added}
    added = sorted((c, s) for c, s in pair_added if s not in gone_names)
    gone = sorted((c, s) for c, s in pair_gone if s not in added_names)
    relocated = sorted(
        (s, before, now)
        for s in gone_names & added_names
        for before in (c for c, t in pair_gone if t == s)
        for now in (c for c, t in pair_added if t == s)
    )

    print(f"rtyper skip-subject ratchet [{header['platform']}]: "
          f"{len(flat_got)} subjects, baseline {len(flat_want)}")
    for cls in sorted(set(got) | set(want)):
        now, before = len(got.get(cls, ())), len(want.get(cls, ()))
        mark = "" if now == before else f"   (baseline {before})"
        print(f"  {cls:44} {now}{mark}")

    if unstamped:
        # Not a failure -- an artefact can predate the stamp -- but it must
        # not pass silently: the key below covers less of the corpus than the
        # prepass read, so a move confined to these crates is invisible here.
        print(f"\nNOTE: the corpus key omits {', '.join(unstamped)} (no "
              f".fingerprint stamp beside the artefact); a move confined to "
              f"those crates will not show as one. Re-extract them to close "
              f"the gap.")

    moved = bool(header["corpus"]) and header["corpus"] != recorded.get("corpus")
    if moved:
        print(
            f"\nNOTE: the baseline names corpus {recorded.get('corpus', '(unrecorded)')}"
            f" and this run read {header['corpus']}. Interpreter graphs the"
            f" baseline never saw are in this set; attribute an addition"
            f" before paying it down."
        )

    if relocated:
        print(f"\n{len(relocated)} subject(s) changed Skip class — still "
              f"skipped, so not an addition; record with --update:")
        for subject, before, now in relocated[:20]:
            print(f"  - {subject}: {before} -> {now}")
        if len(relocated) > 20:
            print(f"  ... and {len(relocated) - 20} more")

    if gone:
        print(f"\n{len(gone)} subject(s) no longer skipped — pay the baseline "
              f"down with --update:")
        for cls, subject in gone[:20]:
            print(f"  - {cls}: {subject}")
        if len(gone) > 20:
            print(f"  ... and {len(gone) - 20} more")

    if added:
        label = ("FAIL — additions this run cannot attribute"
                 if moved else "FAIL")
        print(f"\n{label} — {len(added)} graph(s) newly Skipped to the legacy "
              f"walker:")
        for cls, subject in added[:40]:
            print(f"  - {cls}: {subject}")
        if len(added) > 40:
            print(f"  ... and {len(added) - 40} more")
        if moved:
            print("  Re-extract onto the recorded corpus and rerun to "
                  "attribute these, or refresh the baseline: "
                  "python3 scripts/check-rtyper-skip-subjects.py --update")
            return 1
        print("  Each name is a graph the real rtyper stopped handling. Fix "
              "the class it landed in, or record it with --update if the "
              "Skip is intended.")
        return 1

    print("\nOK — no graph newly Skipped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
