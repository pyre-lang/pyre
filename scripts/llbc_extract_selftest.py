#!/usr/bin/env python3
"""Check `include_closure` against every `include*!` argument spelling.

`fingerprint_inputs` decides which bytes the LLBC staleness stamp — and the CI
cache key built from it — covers. An argument spelling that walks off the end
of that set does not fail anything: the artefact simply reads as fresh while
the crate compiles something else. Nothing downstream can notice, so the check
has to live here.

Runs on a synthetic tree in a temp dir: no cargo, no git, well under a second.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import redirect_stdout

ROOT = pathlib.Path(__file__).resolve().parent


def load_engine():
    """Import the engine by path; it is a library the drivers import, not a package."""
    spec = importlib.util.spec_from_file_location(
        "llbc_extract", ROOT / "llbc_extract.py"
    )
    module = importlib.util.module_from_spec(spec)
    # `@dataclass` resolves annotations through `sys.modules[cls.__module__]`.
    sys.modules["llbc_extract"] = module
    spec.loader.exec_module(module)
    return module


def run(engine) -> int:
    root = pathlib.Path(tempfile.mkdtemp()).resolve()
    outside = pathlib.Path(tempfile.mkdtemp()).resolve()
    try:
        (root / "src").mkdir()
        (root / "data").mkdir()
        (root / "data" / "rules.txt").write_text("rules")
        (outside / "far.txt").write_text("far")

        # FOLD: the file joins the digest. PASS: nothing to add, and nothing
        # unresolved was passed over. REFUSE: extraction stops.
        cases = [
            ("literal outside the pathspecs", 'include_str!("../data/rules.txt");', "FOLD data/rules.txt"),
            ("literal already enumerated", 'include_str!("lib.rs");', "PASS"),
            ("nonexistent path", 'include_str!("../nope.txt");', "PASS"),
            ("concat!(env!(OUT_DIR))", 'include!(concat!(env!("OUT_DIR"), "/g.rs"));', "PASS"),
            ("macro_rules! parameter", "include_str!($appfile);", "PASS"),
            ("prose about the macro", "// how include!() behaves", "PASS"),
            ("raw literal", 'include_str!(r"../data/rules.txt");', "REFUSE"),
            ("literal carrying an escape", 'include_str!("../data\\\\rules.txt");', "REFUSE"),
            ("path through a constant", "include_str!(RULES_PATH);", "REFUSE"),
        ]

        # One `..` per component of `root/src` below the filesystem anchor.
        chain = "../" * len(root.parts) + (outside / "far.txt").relative_to(
            outside.anchor
        ).as_posix()
        cases.append(("`..` chain leaving the repo", f'include_str!("{chain}");', "REFUSE"))

        try:
            (root / "src" / "link.txt").symlink_to(outside / "far.txt")
            cases.append(("symlink leaving the repo", 'include_str!("link.txt");', "REFUSE"))
        except OSError as exc:
            # Windows grants symlink creation only under Developer Mode.
            print(f"skip: symlink case ({exc})")

        failures = 0
        for name, body, expected in cases:
            (root / "src" / "lib.rs").write_text(body + "\n")
            try:
                extra = engine.include_closure(root, {pathlib.Path("src/lib.rs")})
                sorted_extra = sorted(path.as_posix() for path in extra)
                actual = "FOLD " + ",".join(sorted_extra) if extra else "PASS"
            except SystemExit:
                actual = "REFUSE"
            if actual == expected:
                print(f"ok   {name}: {actual}")
            else:
                failures += 1
                print(f"FAIL {name}: expected {expected}, got {actual}")
        return failures
    finally:
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(outside, ignore_errors=True)


def run_fingerprint(engine) -> int:
    """Exercise the complete source key against a synthetic Git working tree."""
    root = pathlib.Path(tempfile.mkdtemp()).resolve()
    try:
        (root / "src").mkdir()
        (root / "scripts").mkdir()
        (root / "Cargo.toml").write_text("[package]\nname='probe'\nversion='0.0.0'\n")
        (root / ".gitignore").write_text("src/ignored.rs\n")
        (root / "src" / "lib.rs").write_text("pub fn answer() -> i64 { 42 }\n")
        tool = root / "scripts" / "tool.py"
        tool.write_text("# implementation detail\n")
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(
            ["git", "add", ".gitignore", "Cargo.toml", "src/lib.rs", "scripts/tool.py"],
            cwd=root,
            check=True,
        )

        spec = engine.CrateSpec(
            name="probe",
            crate_dir=root,
            output_name="probe.ullbc",
            fingerprint_pathspecs=["src/"],
            layout_targets=(),
        )

        def make_engine(*, extraction_abi="probe-1", rustflags=""):
            return engine.Engine(
                specs={"probe": spec},
                default_crates=["probe"],
                root=root,
                out_dir=root / "out",
                base_pathspecs=["Cargo.toml"],
                charon_root=root,
                cargo_features="",
                extraction_abi=extraction_abi,
                layout_target_rustflags=rustflags,
            )

        def fingerprint(eng):
            return engine.source_fingerprint(eng, ["probe"], "")

        failures = 0

        def expect(name, condition):
            nonlocal failures
            if condition:
                print(f"ok   {name}")
            else:
                failures += 1
                print(f"FAIL {name}")

        eng = make_engine()
        base = fingerprint(eng)

        tool.write_text("# diagnostics and comments changed\n")
        expect("tool implementation is outside the source key", fingerprint(eng) == base)

        untracked = root / "src" / "new.rs"
        untracked.write_text("pub fn new_item() {}\n")
        # The input walk is memoised for one stable extraction snapshot. This
        # test deliberately changes the input set, so begin a new snapshot
        # before asking whether the added path moves the fingerprint.
        engine.forget_collected_inputs()
        before_stage = fingerprint(eng)
        expect("untracked source moves the source key", before_stage != base)
        subprocess.run(["git", "add", "src/new.rs"], cwd=root, check=True)
        expect("staging identical bytes preserves the source key", fingerprint(eng) == before_stage)

        ignored = root / "src" / "ignored.rs"
        ignored.write_text("build output\n")
        expect("ignored file stays outside the source key", fingerprint(eng) == before_stage)

        expect(
            "extraction ABI moves the cache and staleness key",
            fingerprint(make_engine(extraction_abi="probe-2")) != before_stage,
        )
        expect(
            "effective layout configuration moves the key",
            fingerprint(make_engine(rustflags="--cfg changed")) != before_stage,
        )
        return failures
    finally:
        shutil.rmtree(root, ignore_errors=True)


def run_layout_sidecar(engine) -> int:
    """Reduce a synthetic full artefact whose shape mirrors Charon's.

    `type_decls` is made wider than the reducer's first decode window so the
    widening retry runs, the bodies around it carry non-ASCII text so a window
    edge can fall inside a multi-byte character, and `has_errors` sits at the
    tail of the file, past every body, the way Charon writes it.
    """
    root = pathlib.Path(tempfile.mkdtemp()).resolve()
    try:
        type_decls = [
            {"item_meta": {"name": f"ty{i}", "doc": "\u00e9" * 64}, "layout": [i]}
            for i in range(60_000)
        ]
        fun_decls = [{"body": "\u2603" * 512, "index": i} for i in range(2_000)]
        artefact = {
            "charon_version": "probe",
            "translated": {
                "crate_name": "probe",
                "files": [{"name": {"Local": "src/lib.rs"}}],
                "type_decls": type_decls,
                "fun_decls": fun_decls,
            },
            "has_errors": False,
        }
        full = root / "probe.full"
        full.write_text(json.dumps(artefact, ensure_ascii=False), encoding="utf-8")
        dest = root / "probe.layouts.ullbc"
        engine.write_layout_sidecar(full, dest)
        slim = json.loads(dest.read_text(encoding="utf-8"))
        expected = {
            "charon_version": "probe",
            "has_errors": False,
            "translated": {
                "crate_name": "probe",
                "type_decls": type_decls,
                "fun_decls": [],
            },
        }
        encoded = json.dumps(type_decls, ensure_ascii=False).encode("utf-8")
        wide = len(encoded) > engine.LAYOUT_DECODE_WINDOW
        ok = slim == expected and wide
        print(f"{'ok  ' if ok else 'FAIL'} layout sidecar keeps type_decls and drops bodies")
        return 0 if ok else 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


def run_stamp_skip(engine) -> int:
    """Skip extract when only `closure=` moved; refuse when `source=` moved."""
    failures = 0

    def expect(name, condition):
        nonlocal failures
        if condition:
            print(f"ok   {name}")
        else:
            failures += 1
            print(f"FAIL {name}")

    keys = engine.STAMP_KEYS
    base = {key: f"{key}-v1" for key in keys}
    expected = "\n".join(f"{key}={base[key]}" for key in keys)

    recorded = expected + "\n"
    ok, closure_moved = engine.stamp_skip_ok(recorded, expected)
    expect("identical stamp skips and reports closure still", ok and not closure_moved)

    crlf = expected.replace("\n", "\r\n") + "\r\n"
    ok, closure_moved = engine.stamp_skip_ok(crlf, expected)
    expect("CRLF stamp folds to the same skip", ok and not closure_moved)

    closure_only = "\n".join(
        f"{key}={base[key] if key != 'closure' else 'closure-v2'}" for key in keys
    )
    ok, closure_moved = engine.stamp_skip_ok(closure_only + "\n", expected)
    expect("closure-only move still skips", ok and closure_moved)

    source_moved = "\n".join(
        f"{key}={base[key] if key != 'source' else 'source-v2'}" for key in keys
    )
    ok, closure_moved = engine.stamp_skip_ok(source_moved + "\n", expected)
    expect("source move refuses the skip", not ok and not closure_moved)

    missing = "\n".join(f"{key}={base[key]}" for key in keys if key != "artefacts")
    ok, _ = engine.stamp_skip_ok(missing + "\n", expected)
    expect("stamp missing a gate field refuses the skip", not ok)

    uncomputed = "\n".join(
        f"{key}={engine.CLOSURE_UNCOMPUTED if key == 'closure' else base[key]}"
        for key in keys
    )
    ok, closure_moved = engine.stamp_skip_ok(recorded, uncomputed)
    expect(
        "uncomputed closure still skips and does not claim it moved",
        ok and not closure_moved,
    )

    return failures


def run_child_dirs_whose_rs_mention(engine) -> int:
    """Immediate subdirs that contain the needle, not siblings that do not."""
    failures = 0

    def expect(name, condition):
        nonlocal failures
        if condition:
            print(f"ok   {name}")
        else:
            failures += 1
            print(f"FAIL {name}")

    root = pathlib.Path(tempfile.mkdtemp()).resolve()
    try:
        (root / "keep" / "nested").mkdir(parents=True)
        (root / "skip").mkdir()
        (root / "keep" / "nested" / "lib.rs").write_text("#[majit_macros::elidable]\nfn f() {}\n")
        (root / "skip" / "lib.rs").write_text("fn g() {}\n")
        (root / "file.rs").write_text("#[majit_macros::elidable]\nfn h() {}\n")
        names = engine.child_dirs_whose_rs_mention(root, "majit_macros::")
        expect("hinted subdirectory is named", names == ["keep"])
        expect("missing root is empty", engine.child_dirs_whose_rs_mention(root / "nope", "x") == [])
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return failures


def run_charon_target_dir(engine) -> int:
    """Charon's cargo target is not the workspace `target/`."""
    failures = 0

    def expect(name, condition):
        nonlocal failures
        if condition:
            print(f"ok   {name}")
        else:
            failures += 1
            print(f"FAIL {name}")

    import hashlib
    import os

    root = pathlib.Path(tempfile.mkdtemp()).resolve()
    saved = {
        key: os.environ.pop(key, None)
        for key in ("CHARON_TARGET_DIR", "PYRE_SHARED_BUILD")
    }
    try:
        default = engine.charon_cargo_target_dir(root)
        key = hashlib.sha256(os.fsencode(root.resolve())).hexdigest()[:16]
        expect(
            "default target is keyed by a hash of the resolved worktree path",
            default == root.parent / ".pyre-build" / "charon-target" / key,
        )
        os.environ["PYRE_SHARED_BUILD"] = str(root / "shared")
        expect(
            "PYRE_SHARED_BUILD relocates the target",
            engine.charon_cargo_target_dir(root)
            == root / "shared" / "charon-target" / key,
        )
        os.environ["CHARON_TARGET_DIR"] = str(root / "explicit")
        expect(
            "CHARON_TARGET_DIR wins over the shared dir",
            engine.charon_cargo_target_dir(root) == root / "explicit",
        )
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(root, ignore_errors=True)
    return failures


def run_llbc_store(engine) -> int:
    """Content-addressed LLBC store: path, clone, publish, prune, disablement."""
    failures = 0

    def expect(name, condition):
        nonlocal failures
        if condition:
            print(f"ok   {name}")
        else:
            failures += 1
            print(f"FAIL {name}")

    saved = {
        key: os.environ.pop(key, None)
        for key in (
            "LLBC_STORE",
            "CI",
            "PYRE_SHARED_BUILD",
            "LLBC_STORE_MAX_AGE_DAYS",
        )
    }
    root = pathlib.Path(tempfile.mkdtemp()).resolve()
    store = pathlib.Path(tempfile.mkdtemp()).resolve()
    try:
        fields = {key: f"{key}-v1" for key in engine.STAMP_KEYS}
        path1 = engine.llbc_store_entry_dir(store, fields)
        expect(
            "entry name is source=",
            path1.name == fields["source"],
        )
        expect(
            "bucket is 32 hex chars",
            len(path1.parent.name) == 32
            and all(c in "0123456789abcdef" for c in path1.parent.name),
        )
        twin = dict(fields)
        twin["closure"] = "closure-other"
        twin["artefacts"] = "artefacts-other"
        path2 = engine.llbc_store_entry_dir(store, twin)
        expect("bucket/entry ignore closure= and artefacts=", path1 == path2)
        moved_source = dict(fields)
        moved_source["source"] = "source-v2"
        path_source = engine.llbc_store_entry_dir(store, moved_source)
        expect(
            "source= is the entry and stays in the same bucket",
            path_source.parent == path1.parent and path_source.name == "source-v2",
        )
        moved_flags = dict(fields)
        moved_flags["flags"] = "flags-v2"
        path_flags = engine.llbc_store_entry_dir(store, moved_flags)
        expect("flags= moves the bucket", path_flags.parent != path1.parent)
        expect(
            "bucket/entry derivation is deterministic",
            path1 == engine.llbc_store_entry_dir(store, fields),
        )
        payload = "\n".join(
            f"{key}={fields[key]}" for key in engine.STORE_BUCKET_KEYS
        )
        expect(
            "bucket digest matches STORE_BUCKET_KEYS join",
            path1.parent.name
            == hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32],
        )
        expect(
            "STORE_BUCKET_KEYS omit closure and artefacts",
            "closure" not in engine.STORE_BUCKET_KEYS
            and "artefacts" not in engine.STORE_BUCKET_KEYS
            and "source" not in engine.STORE_BUCKET_KEYS,
        )

        src = root / "clone-src.bin"
        dst = root / "clone-dst.bin"
        src.write_bytes(b"clone-payload-bytes")
        engine.clone_file(src, dst)
        expect("clone_file copies bytes", dst.read_bytes() == src.read_bytes())
        expect(
            "clone_file uses a distinct inode",
            src.stat().st_ino != dst.stat().st_ino,
        )

        dest_dir = root / "out"
        dest_dir.mkdir()
        artefact = dest_dir / "probe.ullbc"
        readfiles = dest_dir / "probe.ullbc.readfiles"
        fingerprint = dest_dir / "probe.ullbc.fingerprint"
        provenance = dest_dir / "probe.ullbc.provenance"
        artefact.write_bytes(b"ULLBC-v1")
        readfiles.write_text("src/lib.rs\n")
        fingerprint.write_text("crate=probe\n")
        provenance.write_text("crate=probe\n")
        pub_fields = dict(fields)
        pub_fields["crate"] = "probe"
        pub_fields["source"] = "abc" * 8
        sources = [artefact, readfiles, fingerprint, provenance]
        entry = engine.publish_llbc_store_entry(store, pub_fields, sources)
        expect("publish creates the entry dir", entry.is_dir())
        expect(
            "publish clones the artefact",
            (entry / "probe.ullbc").read_bytes() == b"ULLBC-v1",
        )
        expect("publish writes last-used", (entry / "last-used").is_file())
        leftover = list(entry.parent.glob("*.tmp-*"))
        expect("publish leaves no tmp dir", leftover == [])

        marker = entry / "last-used"
        past = time.time() - 1000
        os.utime(marker, (past, past))
        artefact.write_bytes(b"ULLBC-v2-should-not-land")
        again = engine.publish_llbc_store_entry(store, pub_fields, sources)
        expect("second publish returns the existing entry", again == entry)
        expect(
            "publish does not overwrite an existing entry",
            (entry / "probe.ullbc").read_bytes() == b"ULLBC-v1",
        )
        expect(
            "existing publish only touches last-used",
            marker.stat().st_mtime > past + 500,
        )

        now = time.time()
        old_fields = dict(fields)
        old_fields["crate"] = "probe"
        old_fields["source"] = "old" * 8
        new_fields = dict(fields)
        new_fields["crate"] = "probe"
        new_fields["source"] = "new" * 8
        stale_fields = dict(fields)
        stale_fields["platform"] = "goneplat"
        stale_fields["crate"] = "gonecrate"
        stale_fields["source"] = "stale" * 5
        old_entry = engine.publish_llbc_store_entry(store, old_fields, sources)
        new_entry = engine.publish_llbc_store_entry(store, new_fields, sources)
        stale_entry = engine.publish_llbc_store_entry(store, stale_fields, sources)
        os.utime(old_entry / "last-used", (now - 15 * 86400, now - 15 * 86400))
        os.utime(new_entry / "last-used", (now - 1 * 86400, now - 1 * 86400))
        os.utime(stale_entry / "last-used", (now - 15 * 86400, now - 15 * 86400))
        tmp_dir = (
            store
            / fields["platform"]
            / "probe"
            / "deadbucket"
            / "tmpentry.tmp-1-dead"
        )
        tmp_dir.mkdir(parents=True)
        (tmp_dir / "junk").write_bytes(b"x" * 8)
        os.utime(tmp_dir, (now - 2 * 86400, now - 2 * 86400))

        engine.prune_llbc_store(store, max_age_days=14)
        expect("prune removes a 15-day-old entry", not old_entry.exists())
        expect("prune keeps a 1-day-old entry", new_entry.is_dir())
        expect("prune removes an old .tmp- dir", not tmp_dir.exists())
        expect(
            "prune removes empty parents of a stale platform",
            not (store / "goneplat").exists(),
        )
        expect(
            "prune removes the empty tmp bucket",
            not (store / fields["platform"] / "probe" / "deadbucket").exists(),
        )

        marker_prune = store / ".last-prune"
        marker_prune.touch()
        # Recreate a 15-day-old entry under a fresh crate so the 24 h gate
        # has something it would delete if it ran.
        gate_fields = dict(fields)
        gate_fields["crate"] = "gatecrate"
        gate_fields["source"] = "gate" * 8
        gate_entry = engine.publish_llbc_store_entry(store, gate_fields, sources)
        os.utime(gate_entry / "last-used", (now - 15 * 86400, now - 15 * 86400))
        ran = engine.maybe_prune_llbc_store(store)
        expect("fresh .last-prune skips the 24 h prune", ran is False)
        expect("24 h gate leaves the old entry", gate_entry.is_dir())
        os.utime(marker_prune, (now - 25 * 3600, now - 25 * 3600))
        ran = engine.maybe_prune_llbc_store(store)
        expect("stale .last-prune runs prune", ran is True)
        expect("periodic prune removes the old entry", not gate_entry.exists())
        expect(
            "periodic prune refreshes .last-prune",
            marker_prune.stat().st_mtime > now - 60,
        )

        os.environ["LLBC_STORE"] = "off"
        path, msg = engine.llbc_store_dir(root)
        expect(
            "LLBC_STORE=off disables the store",
            path is None and "disabled" in msg and "off" in msg,
        )
        os.environ["LLBC_STORE"] = "0"
        path, msg = engine.llbc_store_dir(root)
        expect(
            "LLBC_STORE=0 disables the store",
            path is None and "disabled" in msg,
        )
        os.environ["LLBC_STORE"] = ""
        path, msg = engine.llbc_store_dir(root)
        expect(
            "empty LLBC_STORE disables the store",
            path is None and "disabled" in msg,
        )
        os.environ.pop("LLBC_STORE", None)
        os.environ["CI"] = "1"
        path, msg = engine.llbc_store_dir(root)
        expect(
            "CI=1 disables the store by default",
            path is None and "disabled" in msg and "CI" in msg,
        )
        os.environ["LLBC_STORE"] = str(store)
        path, msg = engine.llbc_store_dir(root)
        expect(
            "explicit LLBC_STORE path wins over CI",
            path == store and "disabled" not in msg,
        )
        os.environ.pop("LLBC_STORE", None)
        os.environ.pop("CI", None)
        path, msg = engine.llbc_store_dir(root)
        expect(
            "default store is <shared>/llbc-store",
            path == engine.shared_build_dir(root) / "llbc-store",
        )

        dest = root / "probe.ullbc"
        dest.write_bytes(b"x")
        dest.with_suffix(dest.suffix + ".provenance").write_text(
            "crate=probe\n"
            "head_before=aaa\n"
            "head_after=aaa\n"
            "started=t0\n"
            "dirty_status=ok\n"
            "dirty_before=0\n"
            "dirty_in_closure=0\n"
            "dirty_outside=0\n"
            "dirty_omitted=0\n"
            "imported_from=/tmp/llbc-entry\n"
            "imported_at=2026-01-01T00:00:00\n"
        )
        parsed, dirty = engine.parse_provenance(
            dest.with_suffix(dest.suffix + ".provenance").read_text()
        )
        expect(
            "parse_provenance keeps imported_from",
            parsed.get("imported_from") == "/tmp/llbc-entry"
            and parsed.get("imported_at") == "2026-01-01T00:00:00"
            and dirty == [],
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            engine.report_provenance(dest)
        rendered = buf.getvalue()
        expect(
            "report_provenance displays the import",
            "imported from /tmp/llbc-entry" in rendered
            and "2026-01-01T00:00:00" in rendered,
        )
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(store, ignore_errors=True)
    return failures


def main() -> None:
    engine = load_engine()
    failures = (
        run(engine)
        + run_fingerprint(engine)
        + run_layout_sidecar(engine)
        + run_stamp_skip(engine)
        + run_child_dirs_whose_rs_mention(engine)
        + run_charon_target_dir(engine)
        + run_llbc_store(engine)
    )
    if failures:
        raise SystemExit(f"llbc_extract_selftest: {failures} failed")
    print("llbc_extract_selftest: all passed")


if __name__ == "__main__":
    main()
