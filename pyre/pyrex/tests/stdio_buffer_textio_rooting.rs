//! A `TextIOWrapper` subclass wrapping `BufferedWriter(BytesIO)` is
//! nursery-born. `__init__`, `write`, and `flush` each allocate; a small
//! nursery makes a minor collection land while those methods still hold
//! `&mut self`, which they reload from a shadow-stack slot.
//!
//! `PYPY_GC_NURSERY=64K` makes that window deterministic on a host whose
//! default nursery would otherwise hold the whole run.

#![cfg(feature = "dynasm")]

use std::path::PathBuf;
use std::process::Command;

const PYRE: &str = env!("CARGO_BIN_EXE_pyre-dynasm");

const PROGRAM: &str = "\
import io
class StdIOBuffer(io.TextIOWrapper):
    def __init__(self, initial_value='', newline='\\n'):
        initial_value = initial_value.encode('utf-8')
        super().__init__(io.BufferedWriter(io.BytesIO(initial_value)),
                         'utf-8', newline=newline)
    def getvalue(self):
        self.flush()
        return self.buffer.raw.getvalue().decode('utf-8')

total = 0
i = 0
while i < 200:
    buf = StdIOBuffer()
    buf.write('hello %d\\n' % i)
    total += len(buf.getvalue())
    buf.close()
    i += 1
print(total)
";

// [3.15-spec] follows v3.15.2 and `test_truncate_size_changes_stream`:
// re-checks closed after convert_size, so a close in `__index__` raises
// ValueError; truncate leaves tell() as the callback left it (`truncate_w`
// restores the pre-conversion position).
const CHECK_ORDER_PROGRAM: &str = r#"
import io

class CloseOnIndex:
    def __init__(self, stream):
        self.stream = stream
    def __index__(self):
        self.stream.close()
        return 1

b = io.BytesIO(b"abc")
try:
    b.read(CloseOnIndex(b))
except ValueError:
    pass
else:
    raise SystemExit("read close-on-index did not raise")

b = io.BytesIO(b"abc")
try:
    b.readline(CloseOnIndex(b))
except ValueError:
    pass
else:
    raise SystemExit("readline close-on-index did not raise")

b = io.BytesIO(b"abc")
v = b.getbuffer()
indexed = []
class RecordIndex:
    def __index__(self):
        indexed.append(1)
        return 1
try:
    b.truncate(RecordIndex())
except BufferError:
    pass
else:
    raise SystemExit("truncate export did not raise")
if indexed:
    raise SystemExit("truncate called __index__")
del v

b = io.BytesIO(b"abc")
try:
    b.truncate(CloseOnIndex(b))
except ValueError:
    pass
else:
    raise SystemExit("truncate close-on-index did not raise")

b = io.BytesIO(b"abcdef")
b.seek(2)
class SeekOnIndex:
    def __index__(self):
        b.seek(5)
        return 3
got = b.truncate(SeekOnIndex())
if got != 3:
    raise SystemExit("truncate returned %r" % (got,))
if b.tell() != 5:
    raise SystemExit("truncate tell %r" % (b.tell(),))

b = io.BytesIO(b"abcdef")
b.seek(4)
got = b.truncate(2)
if got != 2:
    raise SystemExit("truncate plain returned %r" % (got,))
if b.tell() != 4:
    raise SystemExit("truncate plain tell %r" % (b.tell(),))
print("OK")
"#;

fn run_source(label: &str, source: &str, env: &[(&str, &str)]) -> String {
    let mut path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    path.push(format!("stdio_buffer_textio_rooting_{label}.py"));
    std::fs::write(&path, source).expect("write the program under test");

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

fn run(label: &str, env: &[(&str, &str)]) -> String {
    run_source(label, PROGRAM, env)
}

#[test]
fn default_nursery_stdio_buffer_textio() {
    assert_eq!(run("default", &[]), "1890");
}

#[test]
fn small_nursery_stdio_buffer_textio() {
    assert_eq!(run("64k", &[("PYPY_GC_NURSERY", "64K")]), "1890");
}

#[test]
fn bytesio_checks_closed_and_exports_before_callbacks() {
    assert_eq!(run_source("check-order", CHECK_ORDER_PROGRAM, &[]), "OK");
}
