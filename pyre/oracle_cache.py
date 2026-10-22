#!/usr/bin/env python3
"""Content-addressed CPython reference-result cache.

`check.py`'s synthetic suite and `parity_tests/run.py` re-run the same CPython
oracle on fixtures whose bytes (and local imports) have not changed. This
store is machine-wide so sibling worktrees and concurrent sessions share it.

Only cacheable CPython results are stored: exit 0, and the timeout verdict
(exit 124). A crash or any other nonzero code is always re-run. Times that
feed a performance gate are never written here.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = 1
TIMEOUT_CODE = 124
CACHEABLE_CODES = frozenset((0, TIMEOUT_CODE))
PRUNE_INTERVAL_S = 24 * 3600
STRAY_TMP_MAX_AGE_S = 24 * 3600
DEFAULT_MAX_AGE_DAYS = 30
LAST_PRUNE_NAME = ".last-prune"

_PYTHON_VERSION_CACHE: dict[str, str] = {}


def repo_root() -> Path:
    # This file lives in `pyre/`; the checkout is its parent.
    return Path(__file__).resolve().parent.parent


def shared_build_dir(environ=None) -> Path:
    """Same rule as `scripts/llbc_extract.py` `shared_build_dir`, copied so
    this module stays stdlib-only and does not import the extractor."""
    env = os.environ if environ is None else environ
    named = env.get("PYRE_SHARED_BUILD")
    if named:
        return Path(named).expanduser()
    return repo_root().parent / ".pyre-build"


def resolve_store(environ=None) -> tuple[Path | None, str]:
    """`(store, status line)`. `store` is None when the cache is disabled.

    `PYRE_ORACLE_CACHE` set to a path enables that store (including under CI).
    `off` / `0` / empty disable it. CI with the variable unset disables it.
    Otherwise the default is `<shared>/oracle-cache`.
    """
    env = os.environ if environ is None else environ
    raw = env.get("PYRE_ORACLE_CACHE")
    if raw is not None:
        stripped = raw.strip()
        if stripped in ("", "off", "0"):
            reason = "empty" if stripped == "" else stripped
            return None, f"oracle cache: disabled ({reason})"
        path = Path(stripped).expanduser()
        return path, f"oracle cache: {path}"
    if env.get("CI"):
        return None, "oracle cache: disabled (CI)"
    path = shared_build_dir(env) / "oracle-cache"
    return path, f"oracle cache: {path}"


def python_version(python: str) -> str:
    """`sys.version` of *python*, probed once per realpath per process."""
    real = os.path.realpath(python)
    cached = _PYTHON_VERSION_CACHE.get(real)
    if cached is not None:
        return cached
    try:
        proc = subprocess.run(
            [python, "-c", "import sys; sys.stdout.write(sys.version)"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        version = ""
    else:
        version = proc.stdout if proc.returncode == 0 else ""
    _PYTHON_VERSION_CACHE[real] = version
    return version


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError:
        return b""


def hash_fixture(path) -> str:
    return hashlib.sha256(_read_bytes(Path(path))).hexdigest()


def _top_level_import_names(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            if not node.module:
                continue
            names.append(node.module.split(".", 1)[0])
    return names


def hash_local_modules(fixture_path) -> str:
    """sha256 over every local module the fixture's import statements can see.

    Stdlib names are ignored: they are covered by `python=` in the key. A
    package directory contributes every `*.py` under it, sorted by relative
    path, so a change in a submodule misses.
    """
    fixture_path = Path(fixture_path)
    directory = fixture_path.parent
    source = _read_bytes(fixture_path).decode("utf-8", "replace")
    hasher = hashlib.sha256()
    for name in sorted(set(_top_level_import_names(source))):
        file_path = directory / f"{name}.py"
        pkg_path = directory / name
        if file_path.is_file():
            hasher.update(b"F")
            hasher.update(name.encode("utf-8"))
            hasher.update(b"\0")
            hasher.update(_read_bytes(file_path))
            hasher.update(b"\0")
        if pkg_path.is_dir():
            hasher.update(b"P")
            hasher.update(name.encode("utf-8"))
            hasher.update(b"\0")
            py_files = sorted(
                pkg_path.rglob("*.py"),
                key=lambda path: path.relative_to(pkg_path).as_posix(),
            )
            for py_path in py_files:
                rel = py_path.relative_to(pkg_path).as_posix()
                hasher.update(rel.encode("utf-8"))
                hasher.update(b"\0")
                hasher.update(_read_bytes(py_path))
                hasher.update(b"\0")
    return hasher.hexdigest()


def _env_value(env) -> str:
    if not env:
        return "inherit"
    items = {str(key): str(value) for key, value in env.items()}
    return json.dumps(items, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def entry_key(*, kind, python, argv, timeout, fixture, env=None) -> str:
    """sha256 of the newline-joined `name=value` lines that identify a run.

    `python=` uses the executable's realpath plus `st_size` / `st_mtime_ns` so
    a rebuilt interpreter at the same path misses. `args=` includes the
    effective timeout so a budget change misses. `env=inherit` when the
    runner did not set extra variables on the child.
    """
    python = str(python)
    real = os.path.realpath(python)
    st = os.stat(real)
    argv = [str(arg) for arg in argv]
    fixture = str(fixture)
    args_value = json.dumps(
        {"argv": argv, "timeout": timeout},
        sort_keys=True,
        separators=(",", ":"),
    )
    lines = [
        f"schema={SCHEMA}",
        f"kind={kind}",
        f"python={real} {st.st_size} {st.st_mtime_ns}",
        f"python_version={json.dumps(python_version(python), ensure_ascii=False)}",
        f"args={args_value}",
        f"fixture={hash_fixture(fixture)}",
        f"local={hash_local_modules(fixture)}",
        f"env={_env_value(env)}",
    ]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def fixture_label(path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(repo_root()).as_posix()
    except ValueError:
        return path.as_posix()


def _entry_path(store: Path, key: str) -> Path:
    return store / key[:2] / f"{key}.json"


def read_entry(store: Path, key: str) -> dict | None:
    """Load a cacheable entry, or None on miss / corruption.

    A hit updates the entry's mtime, which is the last-use stamp prune reads.
    """
    path = _entry_path(store, key)
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("schema") != SCHEMA:
        return None
    if not isinstance(data.get("code"), int) or data["code"] not in CACHEABLE_CODES:
        return None
    if not isinstance(data.get("output"), str):
        return None
    if not isinstance(data.get("elapsed"), (int, float)):
        return None
    try:
        os.utime(path, None)
    except OSError:
        pass
    return data


def write_entry(store: Path, key: str, *, code, output, elapsed, fixture="") -> None:
    if code not in CACHEABLE_CODES:
        return
    dest = _entry_path(store, key)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA,
        "code": int(code),
        "output": output,
        "elapsed": float(elapsed),
        "created": datetime.now(timezone.utc).isoformat(),
        "fixture": fixture,
    }
    fd, tmp = tempfile.mkstemp(
        prefix=f".{key}.", suffix=".tmp", dir=str(dest.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp, dest)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def lookup_or_run(store, key, refresh, run, fixture=""):
    """`(output, elapsed, code, cached)`.

    *run* is `() -> (output, elapsed, code)`. A disabled store (*store* is
    None) just calls *run*. `--no-oracle-cache` sets *refresh* so the child
    still runs and the result is rewritten.
    """
    if store is not None and not refresh:
        hit = read_entry(store, key)
        if hit is not None:
            return hit["output"], float(hit["elapsed"]), int(hit["code"]), True
    output, elapsed, code = run()
    if store is not None and code in CACHEABLE_CODES:
        try:
            write_entry(
                store, key,
                code=code, output=output, elapsed=elapsed, fixture=fixture,
            )
        except OSError as exc:
            print(f"warning: oracle cache write failed: {exc}", file=sys.stderr)
    return output, elapsed, code, False


def _max_age_s(environ=None) -> float:
    env = os.environ if environ is None else environ
    raw = env.get("PYRE_ORACLE_CACHE_MAX_AGE_DAYS", str(DEFAULT_MAX_AGE_DAYS))
    try:
        days = float(raw)
    except (TypeError, ValueError):
        days = float(DEFAULT_MAX_AGE_DAYS)
    return days * 86400


def prune(store: Path, *, force: bool = False, now: float | None = None) -> int:
    """Drop stale entries and stray temp files. Returns the number removed.

    Automatic callers pass *force* False and are gated on `.last-prune` so
    this walks the store at most once per 24 h. `--prune-oracle-cache` forces
    a walk. Errors from individual unlinks are skipped; the stamp is still
    touched so a half-finished walk does not repeat immediately.
    """
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)
    now = time.time() if now is None else now
    stamp = store / LAST_PRUNE_NAME
    if not force and stamp.is_file():
        try:
            if now - stamp.stat().st_mtime < PRUNE_INTERVAL_S:
                return 0
        except OSError:
            pass
    max_age = _max_age_s()
    removed = 0
    for dirpath, _dirnames, filenames in os.walk(store):
        for name in filenames:
            if name == LAST_PRUNE_NAME and Path(dirpath) == store:
                continue
            path = Path(dirpath) / name
            try:
                age = now - path.stat().st_mtime
            except OSError:
                continue
            drop = False
            if name.endswith(".json"):
                drop = age > max_age
            elif name.endswith(".tmp"):
                drop = age > STRAY_TMP_MAX_AGE_S
            if not drop:
                continue
            try:
                path.unlink()
            except OSError:
                continue
            removed += 1
    try:
        stamp.touch()
    except OSError:
        pass
    return removed


def maybe_prune(store, *, force: bool = False) -> int:
    if store is None:
        return 0
    try:
        return prune(Path(store), force=force)
    except OSError as exc:
        print(f"warning: oracle cache prune failed: {exc}", file=sys.stderr)
        return 0


def prune_cli() -> int:
    store, line = resolve_store()
    print(line)
    if store is None:
        return 0
    removed = maybe_prune(store, force=True)
    print(f"pruned {removed} oracle-cache file(s)")
    return 0


def _self_test() -> int:
    failed: list[str] = []

    def check(cond, msg):
        if cond:
            print(f"  ok  {msg}")
        else:
            print(f"  FAIL {msg}")
            failed.append(msg)

    # Disable / enable resolution, independent of this process's real env.
    store, msg = resolve_store({"PYRE_ORACLE_CACHE": "off"})
    check(store is None and "disabled" in msg and "off" in msg, "off disables")
    store, msg = resolve_store({"PYRE_ORACLE_CACHE": "0"})
    check(store is None and "disabled" in msg, "0 disables")
    store, msg = resolve_store({"PYRE_ORACLE_CACHE": ""})
    check(store is None and "disabled" in msg, "empty disables")
    store, msg = resolve_store({"CI": "1"})
    check(store is None and "disabled" in msg and "CI" in msg, "CI=1 disables")
    store, msg = resolve_store({"CI": "1", "PYRE_ORACLE_CACHE": "/tmp/oracle-x"})
    check(store == Path("/tmp/oracle-x"), "CI=1 with an explicit path enables")
    store, msg = resolve_store({})
    check(store is not None and str(store).endswith("oracle-cache"), "default store path")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fixture = tmp_path / "fixture.py"
        fixture.write_text("print(1)\n", encoding="utf-8")
        python = sys.executable
        base = dict(
            kind="synth",
            python=python,
            argv=[str(fixture)],
            timeout=5,
            fixture=str(fixture),
            env=None,
        )
        key = entry_key(**base)

        fixture.write_text("print(2)\n", encoding="utf-8")
        check(entry_key(**base) != key, "key changes when fixture bytes change")
        fixture.write_text("print(1)\n", encoding="utf-8")
        check(entry_key(**base) == key, "key is stable when fixture bytes are restored")

        local_mod = tmp_path / "localmod.py"
        local_mod.write_text("VALUE = 1\n", encoding="utf-8")
        fixture.write_text("import localmod\nprint(localmod.VALUE)\n", encoding="utf-8")
        key_local = entry_key(**base)
        check(key_local != key, "key changes when a local import appears")
        local_mod.write_text("VALUE = 2\n", encoding="utf-8")
        check(entry_key(**base) != key_local, "key changes when a local module changes")
        local_mod.write_text("VALUE = 1\n", encoding="utf-8")
        check(entry_key(**base) == key_local, "key is stable when a local module is restored")
        (tmp_path / "unrelated.py").write_text("x = 0\n", encoding="utf-8")
        check(
            entry_key(**base) == key_local,
            "an unused file next to the fixture does not change the key",
        )

        dummy = tmp_path / "dummy-python"
        dummy.write_bytes(b"not-an-interpreter\n")
        dummy_base = dict(base)
        dummy_base["python"] = str(dummy)
        key_dummy = entry_key(**dummy_base)
        st = dummy.stat()
        os.utime(dummy, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
        check(entry_key(**dummy_base) != key_dummy, "key changes when python mtime changes")

        other_timeout = dict(base)
        other_timeout["timeout"] = 30
        fixture.write_text("import localmod\nprint(localmod.VALUE)\n", encoding="utf-8")
        check(entry_key(**other_timeout) != key_local, "key changes when timeout changes")
        check(entry_key(**base) == key_local, "key is stable otherwise")

        cache = tmp_path / "store"
        calls = []

        def run_crash():
            calls.append("crash")
            return "boom\n", 0.05, 1

        out, elapsed, code, cached = lookup_or_run(
            cache, key_local, False, run_crash, fixture_label(fixture),
        )
        check(
            (out, elapsed, code, cached) == ("boom\n", 0.05, 1, False),
            "crash result is returned as a miss",
        )
        check(read_entry(cache, key_local) is None, "crash codes are not stored")

        def run_ok():
            calls.append("ok")
            return "ok-out\n", 0.12, 0

        out, elapsed, code, cached = lookup_or_run(
            cache, key_local, False, run_ok, fixture_label(fixture),
        )
        check(not cached and code == 0, "successful miss is stored")
        out2, elapsed2, code2, cached2 = lookup_or_run(
            cache, key_local, False, run_ok, fixture_label(fixture),
        )
        check(cached2 and (out2, elapsed2, code2) == (out, elapsed, code), "successful hit")
        check(calls.count("ok") == 1, "a hit does not re-run CPython")

        timeout_key = entry_key(**other_timeout)

        def run_timeout():
            calls.append("timeout")
            return "", 0.0, TIMEOUT_CODE

        lookup_or_run(cache, timeout_key, False, run_timeout, fixture_label(fixture))
        _out, _elapsed, code, cached = lookup_or_run(
            cache, timeout_key, False, run_timeout, fixture_label(fixture),
        )
        check(cached and code == TIMEOUT_CODE, "timeout (124) is stored and hit")

        def run_refresh():
            calls.append("refresh")
            return "refreshed\n", 0.3, 0

        out, elapsed, code, cached = lookup_or_run(
            cache, key_local, True, run_refresh, fixture_label(fixture),
        )
        check(not cached and out == "refreshed\n", "refresh skips the hit and rewrites")

        now = time.time()
        old_key = hashlib.sha256(b"old").hexdigest()
        young_key = hashlib.sha256(b"young").hexdigest()
        write_entry(cache, old_key, code=0, output="old", elapsed=1.0, fixture="old.py")
        write_entry(cache, young_key, code=0, output="young", elapsed=1.0, fixture="young.py")
        old_path = _entry_path(cache, old_key)
        young_path = _entry_path(cache, young_key)
        os.utime(old_path, (now - 31 * 86400, now - 31 * 86400))
        os.utime(young_path, (now - 1 * 86400, now - 1 * 86400))
        removed = prune(cache, force=True, now=now)
        check(removed >= 1 and not old_path.is_file(), "prune removes a 31-day-old entry")
        check(young_path.is_file(), "prune keeps a 1-day-old entry")

        stale_again = hashlib.sha256(b"stale-again").hexdigest()
        write_entry(
            cache, stale_again, code=0, output="stale", elapsed=1.0, fixture="stale.py",
        )
        stale_path = _entry_path(cache, stale_again)
        os.utime(stale_path, (now - 31 * 86400, now - 31 * 86400))
        skipped = prune(cache, force=False, now=now)
        check(skipped == 0 and stale_path.is_file(), "24h .last-prune gate skips a second walk")
        forced = prune(cache, force=True, now=now)
        check(forced >= 1 and not stale_path.is_file(), "forced prune ignores the 24h gate")

    if failed:
        print(f"{len(failed)} self-test failure(s)")
        return 1
    print("self-test passed")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv == ["--self-test"]:
        return _self_test()
    print("usage: oracle_cache.py --self-test", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
