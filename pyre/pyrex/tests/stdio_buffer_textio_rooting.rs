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

fn run(label: &str, env: &[(&str, &str)]) -> String {
    let mut path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    path.push(format!("stdio_buffer_textio_rooting_{label}.py"));
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
fn default_nursery_stdio_buffer_textio() {
    assert_eq!(run("default", &[]), "1890");
}

#[test]
fn small_nursery_stdio_buffer_textio() {
    assert_eq!(run("64k", &[("PYPY_GC_NURSERY", "64K")]), "1890");
}
