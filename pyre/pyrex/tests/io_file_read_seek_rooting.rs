//! `TextIOWrapper.read` / `readline` / `seek`, `BufferedReader` /
//! `BufferedRandom` read paths, `FileIO`, and `BufferedRWPair` each hold
//! `&mut self` across allocating calls. A 64K nursery makes a minor
//! collection land in that window on a host whose default nursery would
//! otherwise hold the whole run.

#![cfg(feature = "dynasm")]

use std::path::PathBuf;
use std::process::Command;

const PYRE: &str = env!("CARGO_BIN_EXE_pyre-dynasm");

const PROGRAM: &str = r#"
import io, os, sys

base = sys.argv[0] + '.workdir'
os.makedirs(base, exist_ok=True)

def once(i):
    path = os.path.join(base, 'f%d' % i)
    wpath = os.path.join(base, 'w%d' % i)
    raw = io.FileIO(path, 'w')
    raw.write(b'line1\nline2\nline3\n')
    raw.close()

    f = io.FileIO(path, 'r')
    data = f.read()
    f.seek(0)
    first = f.read(6)
    _ = repr(f)
    buf = bytearray(6)
    n = f.readinto(buf)
    f.close()
    if data != b'line1\nline2\nline3\n' or first != b'line1\n' or n != 6:
        raise SystemExit('fileio')
    w = io.FileIO(path, 'r+')
    w.truncate(4)
    w.seek(0)
    truncated = w.read()
    w.close()
    if truncated != b'line':
        raise SystemExit('fileio truncate %r' % truncated)

    # Restore the original bytes so later readline/seek/textio/rwpair
    # still exercise the multi-line paths against the same contents.
    restored = io.FileIO(path, 'w')
    restored.write(b'line1\nline2\nline3\n')
    restored.close()

    f = open(path, 'rb')
    line = f.readline()
    pos = f.tell()
    f.seek(0)
    allb = f.read()
    f.close()
    if line != b'line1\n' or allb != b'line1\nline2\nline3\n' or pos != 6:
        raise SystemExit('reader %r %r %r' % (line, allb, pos))

    f = open(path, 'r+b')
    line = f.readline()
    f.seek(0)
    f.write(b'XXXX')
    f.flush()
    f.seek(0)
    allb = f.read()
    f.close()
    if not allb.startswith(b'XXXX'):
        raise SystemExit('random %r' % allb)

    f = open(path, 'r', encoding='utf-8', newline='\n')
    line = f.readline()
    pos = f.tell()
    f.seek(0)
    allt = f.read()
    f.close()
    if 'XXXX' not in allt:
        raise SystemExit('textio %r' % allt)

    r = io.FileIO(path, 'r')
    w = io.FileIO(wpath, 'w')
    pair = io.BufferedRWPair(r, w)
    got = pair.read()
    pair.write(b'ok')
    pair.flush()
    pair.close()
    if b'XXXX' not in got:
        raise SystemExit('rwpair %r' % got)
    os.unlink(path)
    os.unlink(wpath)

total = 0
i = 0
while i < 80:
    once(i)
    total += 1
    i += 1
print(total)
"#;

fn run(label: &str, env: &[(&str, &str)]) -> String {
    let mut path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    path.push(format!("io_file_read_seek_rooting_{label}.py"));
    std::fs::write(&path, PROGRAM).expect("write the program under test");

    let mut cmd = Command::new(PYRE);
    cmd.arg(&path);
    for (k, v) in env {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("run the program under test");
    assert!(
        out.status.success(),
        "{label}: {}\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr),
    );
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

#[test]
fn default_nursery_io_file_read_seek() {
    assert_eq!(run("default", &[]), "80");
}

#[test]
fn small_nursery_io_file_read_seek() {
    assert_eq!(run("64k", &[("PYPY_GC_NURSERY", "64K")]), "80");
}
