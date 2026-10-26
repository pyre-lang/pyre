#!/usr/bin/env python3
"""Hold the hand-written GC root brackets to what the analysis can prove.

`rpython/memory/gctransform/` inserts the shadow-stack bracket automatically:
`framework.py` brackets every operation that can reach the collector,
`shadowstack.py` emits the `gc_push_roots` / `gc_pop_roots` pair, and
`expand_pop_roots` turns the pop into one `gc_restore_root` per variable.
The graph-wide framework marker insertion and `shadowcolor` stages are now
automatic. Native interpreter paths are also compiled directly by rustc, so
they retain their existing source `push_roots` brackets and this gate stops
those brackets being taken on trust. See the module docs on
`majit-translate/src/memory/gctransform/mod.rs` for that current port boundary.

Some of the reported numbers are invariants at zero and are held there.  The
rest are a backlog: they are ratcheted, so a change may pay them down but not
add to them.

The baseline holds one entry per platform.  The scan reads an artefact built
from this platform's sources, and the interpreter's `cfg` arms differ across
them, so the counts do too -- a baseline written from one platform cannot be
satisfied from another, and `--update` rewrites only the entry it measured.

The ratchet counts every unbracketed call in the artefact, this branch's and
main's together.  A base that has moved brings code the baseline never saw
into the same number a regression would land in.  The gate holds that number:
a rise fails wherever it is measured, and a rise on main is fixed on main.
The run still says when the base has moved, so the reader can see whose code
the count now includes.  A column the baseline entry does not record yet is
printed as unrecorded and is not a failure; `--update` records it.

Run the analysis and compare:

    cargo build -p majit-translate --release --example gc-root-reachability
    python3 scripts/check-gc-root-brackets.py

`--update` rewrites the baseline from the current run, for a change that pays
the backlog down.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASELINE = ROOT / "majit" / "gc-root-brackets.baseline.json"
EXAMPLE = ROOT / "target" / "release" / "examples" / "gc-root-reachability"

# Every production LLBC crate. Donors join the call graph; only subject
# bodies are walked. `gc_ptr_type_ids` now also reads `register_module` /
# `module_ns_store` signatures, so a `--opaque pyre_object` module
# artefact still has an artefact-local `PyObjectRef` id.
LLBC = (
    "target/llbc/majit-rlib.ullbc",
    "target/llbc/pyre-object.ullbc",
    "target/llbc/pyre-interpreter.ullbc",
    "target/llbc/pyre-module.ullbc",
    "target/llbc/pyre-jit.ullbc",
)
SUBJECTS = (
    "target/llbc/pyre-interpreter.ullbc",
    "target/llbc/pyre-module.ullbc",
)

# (key, regex).  Every one of these must match exactly once, in this order.
# `tier 1` is printed twice, and `counting unresolved dispatch` is printed
# twice; telling each pair apart is positional.  `short_brackets` ends on the
# count so the following pattern can read `unread_brackets` off the same line.
PATTERNS = [
    ("unmatched_seeds",
     r"collecting-alloc seeds:.*?UNMATCHED patterns: \[(?P<v>.*?)\]"),
    ("brackets_reaching_no_collection",
     r"cannot reach any collection\s*: (?P<v>\d+)"),
    ("short_brackets",
     r"of those withheld: \d+ pin every live pointer, (?P<v>\d+)"),
    ("unread_brackets",
     r"are SHORT a root, (?P<v>\d+) could not be read"),
    ("stale_pins",
     r"pins whose argument the body still reads afterwards: (?P<v>\d+)"),
    ("unbracketed_calls",
     r"unbracketed calls that can collect with a live PyObjectRef: (?P<v>\d+) in (?P<w>\d+) fn"),
    ("unresolved_collecting_calls",
     r"counting unresolved dispatch as collecting too: (?P<v>\d+) in (?P<w>\d+) fn"),
    ("tier1_calls",
     r"tier 1 \(callee IS a dispatch seed\): (?P<v>\d+) call\(s\) in (?P<w>\d+) fn"),
    ("tier15_calls",
     r"tier 1\.5 \(live ptr later addressed as list/dict\): (?P<v>\d+) call\(s\) in (?P<w>\d+) fn"),
    ("tier15_unresolved_calls",
     r"tier 1\.5 counting unresolved dispatch too: (?P<v>\d+) call\(s\)"),
    ("frames_across_collecting",
     r"frame carried across a call that can collect: (?P<v>\d+) in (?P<w>\d+) fn"),
    ("frames_unresolved_calls",
     r"counting unresolved dispatch as collecting too: (?P<v>\d+) call\(s\)"),
    ("frame_tier1_calls",
     r"tier 1 \(callee IS a dispatch seed\): (?P<v>\d+) call\(s\) in (?P<w>\d+) fn"),
]

# Held at zero rather than ratcheted.  A frame carried across a collecting call
# whose callee is a dispatch seed is a stale frame, not a backlog entry.
# `tier1_calls` measures zero and is held there with it.  `tier15_calls`, the
# alias-closed column, was paid down to zero and is now held there.
INVARIANT_ZERO = ("frame_tier1_calls", "tier1_calls", "tier15_calls")

# A crate with no `PyFrame` (optional modules) skips the frame scan.
# That is not a liveness skip: `PyObjectRef` was already measured.
FRAME_KEYS = (
    "frames_across_collecting",
    "frames_unresolved_calls",
    "frame_tier1_calls",
)
FRAME_SKIPPED = "frame scan skipped"

# Printed only by the liveness scan.  When that scan is skipped these columns
# are zero, so the shape error stays on the unbracketed-call line the scan
# exists to produce.
LIVENESS_KEYS = (
    "short_brackets",
    "unread_brackets",
    "stale_pins",
    "unresolved_collecting_calls",
    "tier15_unresolved_calls",
)
LIVENESS_SKIPPED = "liveness scan skipped"

# Ratcheted: may fall, may not rise.  A column the baseline entry does not
# record yet is unrecorded and fails until `--update` stores it.
RATCHET = (
    "unbracketed_calls",
    "frames_across_collecting",
    "brackets_reaching_no_collection",
    "short_brackets",
    "unread_brackets",
    "stale_pins",
    "unresolved_collecting_calls",
    "unresolved_collecting_calls_fns",
    "tier15_unresolved_calls",
    "frames_unresolved_calls",
)


def platform_key() -> str:
    """The name this run's numbers are recorded under.

    The scan reads an artefact extracted from this platform's build, and the
    interpreter's `cfg` arms differ across them -- a Linux artefact carries
    calls a macOS one does not.  The counts are therefore not one number but
    one per platform, and a baseline written from one of them cannot be
    satisfied from another: a macOS `--update` would leave the Linux gate
    permanently red by exactly the difference between the two.
    """
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "darwin":
        return "darwin"
    return sys.platform


def merge_base() -> str:
    """The upstream commit this branch is measured against.

    The numbers below count every unbracketed call in the artefact.  A rebase
    brings interpreter code the baseline never saw, and its calls land in this
    count exactly like a regression would -- so record what the baseline was
    taken against, and say when that has moved.  The rise fails either way,
    and a rise on main is fixed on main.
    """
    # A shallow CI checkout is grafted: it holds no `main` ref, and the
    # merge commit's parent list is truncated away, so nothing in the
    # repository can name the base.  The workflow knows it and passes it in.
    supplied = os.environ.get("PYRE_GC_GATE_BASE", "").strip()
    if supplied:
        return supplied
    for base in ("origin/main", "upstream/main"):
        proc = subprocess.run(["git", "merge-base", base, "HEAD"], cwd=ROOT,
                              capture_output=True, encoding="utf-8")
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    # A pull-request checkout holds no `main` ref at all: the action fetches
    # `refs/pull/N/merge` and nothing else, so every `merge-base` above fails
    # and the base reads as unknown -- which is precisely the run that most
    # needs it, since that merge commit carries whatever `main` gained since
    # the baseline.  Its first parent *is* the base tip, and naming it needs
    # only the commit object already in hand.
    proc = subprocess.run(["git", "rev-list", "--parents", "-n", "1", "HEAD"],
                          cwd=ROOT, capture_output=True, encoding="utf-8")
    if proc.returncode == 0:
        parts = proc.stdout.split()
        if len(parts) == 3:
            return parts[1]
    return ""


def merge_counts(left: dict, right: dict) -> dict:
    """Add one subject's numbers to another's."""
    merged: dict = {}
    for key in set(left) | set(right):
        a, b = left.get(key), right.get(key)
        if isinstance(a, list) or isinstance(b, list):
            merged[key] = sorted(set(a or []) | set(b or []))
        else:
            merged[key] = (a or 0) + (b or 0)
    return merged


def run_one(subject: str, donors: list[str]) -> str:
    # Option<PyObjectRef> and &[PyObjectRef] locals count as GC pointers.
    env = dict(
        os.environ,
        GC_JOIN_WITH=",".join(donors),
        GC_OPTION_REFS="1",
        GC_SLICE_ARGS="1",
    )
    proc = subprocess.run(
        [str(EXAMPLE), subject],
        cwd=ROOT,
        env=env,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        sys.exit(f"error: analysis exited {proc.returncode}\n{proc.stderr}")
    return proc.stdout


def run_analysis() -> dict:
    if not EXAMPLE.exists():
        sys.exit(
            f"error: {EXAMPLE.relative_to(ROOT)} is not built.\n"
            "  cargo build -p majit-translate --release "
            "--example gc-root-reachability"
        )
    missing = [p for p in LLBC if not (ROOT / p).is_file()]
    if missing:
        sys.exit(
            "error: LLBC artefacts missing: " + ", ".join(missing) + "\n"
            "  python3 scripts/extract-llbc.py majit-rlib pyre-object "
            "pyre-interpreter pyre-module pyre-jit"
        )
    got: dict | None = None
    for subject in SUBJECTS:
        donors = [path for path in LLBC if path != subject]
        part = parse(run_one(subject, donors))
        got = part if got is None else merge_counts(got, part)
    assert got is not None
    return got


def parse(report: str) -> dict:
    """Read the numbers, and refuse to report a clean run over an empty read.

    A gate whose pass is indistinguishable from a gate that matched nothing is
    a gate nobody can trust, so a pattern that does not appear where it is
    expected is an error rather than a missing key.  A skipped liveness scan
    or a skipped frame scan has no lines for the columns that scan prints;
    those columns are zero.
    """
    got: dict = {}
    pos = 0
    liveness_skipped = LIVENESS_SKIPPED in report
    frame_skipped = FRAME_SKIPPED in report
    for key, pattern in PATTERNS:
        m = re.compile(pattern, re.S).search(report, pos)
        if m is None:
            skipped = (liveness_skipped and key in LIVENESS_KEYS) or (
                frame_skipped and key in FRAME_KEYS
            )
            if skipped:
                got[key] = 0
                # The fn-count is the same line.  A skipped scan records it as
                # zero too, matching the call count above.
                if key == "unresolved_collecting_calls":
                    got["unresolved_collecting_calls_fns"] = 0
                continue
            sys.exit(
                f"error: the analysis report has no `{key}` line after "
                f"offset {pos}. The report shape changed; this gate reads it "
                f"positionally and cannot tell a zero from an absence.\n"
                f"--- report ---\n{report}"
            )
        pos = m.end()
        if key == "unmatched_seeds":
            names = [s.strip().strip('"') for s in m.group("v").split(",")]
            got[key] = sorted(n for n in names if n)
        else:
            got[key] = int(m.group("v"))
            if "w" in m.groupdict() and m.group("w") is not None:
                got[key + "_fns"] = int(m.group("w"))
    return got


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--update", action="store_true",
                    help="rewrite the baseline from this run")
    args = ap.parse_args()

    got = run_analysis()
    got["base"] = merge_base()
    key = platform_key()

    recorded = json.loads(BASELINE.read_text()) if BASELINE.is_file() else {}

    if args.update:
        # Only this platform's entry: the others were measured on artefacts
        # this run never saw and are not ours to rewrite.
        recorded[key] = got
        BASELINE.write_text(json.dumps(recorded, indent=2, sort_keys=True) + "\n")
        print(f"wrote {BASELINE.relative_to(ROOT)} [{key}]")
        for k in sorted(got):
            print(f"  {k}: {got[k]}")
        return 0

    if key not in recorded:
        sys.exit(
            f"error: {BASELINE.relative_to(ROOT)} records no `{key}` entry "
            f"(it has: {', '.join(sorted(recorded)) or 'nothing'}).\n"
            "  Seed it from a run on this platform: "
            "python3 scripts/check-gc-root-brackets.py --update"
        )
    want = recorded[key]

    # Printed whichever way this ends, so a reader can see what was measured
    # rather than infer it from silence.
    print(f"gc root bracket gate — measured [{key}]:")
    return compare(got, want)


def compare(got: dict, want: dict) -> int:
    """Score one run against one platform's baseline entry.

    Returns 0 when every invariant is zero and no ratcheted column rose.
    A ratcheted column the baseline does not record yet fails as unrecorded.
    A rise fails even when this run's base is not the base
    the baseline recorded: the gate holds main's own code too, and a rise on
    main is fixed on main.
    """
    got = dict(got)
    for k in sorted(got):
        if k not in want:
            mark = "   (unrecorded)"
        else:
            base = want[k]
            mark = "" if got[k] == base else f"   (baseline {base})"
        print(f"  {k:34} {got[k]}{mark}")

    base_now = got.get("base", "")
    if base_now and base_now != want.get("base"):
        recorded = want.get("base")
        recorded = recorded[:12] if recorded else "(unrecorded)"
        print(
            f"\nNOTE: the baseline was taken against {recorded}"
            f" and this run sits on {base_now[:12]}. Interpreter code the"
            f" baseline never saw is in this count; attribute a rise before"
            f" paying it down."
        )

    bad = []
    got.pop("base", None)
    for k in INVARIANT_ZERO:
        if got[k] != 0:
            bad.append(f"{k} is {got[k]}, and this one is held at zero: a live "
                       f"pointer addressed as a relocatable object across an "
                       f"unbracketed collecting call is a use-after-move, not "
                       f"a backlog entry.")
    for k in RATCHET:
        if k not in want:
            bad.append(
                f"{k} is {got[k]} (unrecorded). Seed this platform's "
                f"baseline with --update so the column is enforced."
            )
        elif got[k] > want[k]:
            bad.append(
                f"{k} rose {want[k]} -> {got[k]}. Root the new call, "
                f"or pay the baseline down with --update."
            )
    if "unmatched_seeds" in want and got["unmatched_seeds"] != want["unmatched_seeds"]:
        bad.append(
            f"the unmatched seed set changed: {want['unmatched_seeds']} -> "
            f"{got['unmatched_seeds']}. A seed that matches nothing empties "
            f"half the closure silently, so this is checked rather than "
            f"trusted."
        )

    if bad:
        print("\nFAIL")
        for b in bad:
            print(f"  - {b}")
        return 1
    print("\nOK — invariants at zero, backlog not raised.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
