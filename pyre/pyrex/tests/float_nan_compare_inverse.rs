//! A NaN operand makes every ordered comparison false, so `x > y` is not
//! `not (x <= y)`.  The loop computes both relations on the same operands
//! until it is compiled; the optimized trace must not answer the second
//! with the first negated (`rewrite.py find_rewritable_bool` over
//! `resoperation.py opboolinvers`, whose float LT/GE and GT/LE pairs do
//! not hold for NaN).

#![cfg(feature = "dynasm")]

use std::path::PathBuf;
use std::process::Command;

const PYRE: &str = env!("CARGO_BIN_EXE_pyre-dynasm");

const PROGRAM: &str = "\
def main():
    vals = [0.5, float(\"nan\"), -1.5, float(\"inf\")]
    others = [0, 3, -1, True, 2.5, float(\"nan\"), 2**70]
    acc = [0] * 8
    for _ in range(20000):
        for x in vals:
            for y in others:
                acc[0] += x <= y
                acc[1] += x > y
                acc[2] += x < y
                acc[3] += x >= y
                acc[4] += y <= x
                acc[5] += y > x
                acc[6] += y < x
                acc[7] += y >= x
    print(acc)


main()
";

fn run(label: &str, env: &[(&str, &str)]) -> String {
    let mut path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    path.push(format!("float_nan_compare_inverse_{label}.py"));
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
fn nan_ordered_compare_is_not_the_inverse_relation() {
    assert_eq!(
        run("default", &[]),
        "[200000, 160000, 200000, 160000, 160000, 200000, 160000, 200000]"
    );
}
