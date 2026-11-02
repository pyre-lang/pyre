/// Brainfuck interpreter benchmarks — interpreter vs JIT.
pub mod interp;
pub mod jit_interp;

use std::time::Instant;

/// Simple but long-running: count from N to 0.
/// BF: set cell0=N via repeated +, then loop: [-]
fn countdown_bf(n: usize) -> Vec<u8> {
    let mut prog = Vec::with_capacity(n + 3);
    prog.extend(std::iter::repeat_n(b'+', n));
    prog.extend_from_slice(b"[-]");
    prog
}

/// Multiply: cell1 = a * b via nested loop.
fn multiply_bf(a: u8, b: u8) -> Vec<u8> {
    let mut prog = Vec::new();
    prog.extend(std::iter::repeat_n(b'+', usize::from(a)));
    prog.push(b'[');
    prog.push(b'>');
    prog.extend(std::iter::repeat_n(b'+', usize::from(b)));
    prog.extend_from_slice(b"<-]");
    prog
}

/// Checks the interpreted trip count for multiplication programs of different
/// parity. This gate validates arithmetic output; JIT liveness is covered by
/// the tests in `jit_interp`.
fn trip_count_gate() {
    for a in [113u8, 114] {
        let b = 2u8;
        let mut prog = multiply_bf(a, b);
        prog.extend_from_slice(b">.");
        let mut jit = jit_interp::JitBrainInterp::new();
        let out = jit.run(&prog);
        assert_eq!(
            out.chars().count(),
            1,
            "multiply({a}*{b}) printed {out:?}; the gate expects exactly one cell"
        );
        let got = out.chars().next().unwrap() as u32;
        let expected = u32::from(a) * u32::from(b);
        assert_eq!(
            got,
            expected,
            "multiply({a}*{b}) left {got} in cell 1, so the loop ran {} passes \
             rather than {a}",
            got / u32::from(b)
        );
        println!("[trip-count] multiply({a}*{b}) = {got} — exactly {a} passes");
    }
}

fn main() {
    trip_count_gate();

    // Benchmark 1: countdown (hot tight loop)
    let n = 100_000;
    let countdown = countdown_bf(n);
    println!("=== countdown({n}) ===");
    {
        let start = Instant::now();
        interp::interpret(&countdown);
        println!("  interp: {:?}", start.elapsed());
    }
    {
        let mut jit = jit_interp::JitBrainInterp::new();
        let start = Instant::now();
        jit.run(&countdown);
        println!("  JIT:    {:?}", start.elapsed());
    }

    // Benchmark 2: multiply (nested loop)
    let mul = multiply_bf(255, 255);
    println!("\n=== multiply(255*255) ===");
    {
        let start = Instant::now();
        interp::interpret(&mul);
        println!("  interp: {:?}", start.elapsed());
    }
    {
        let mut jit = jit_interp::JitBrainInterp::new();
        let start = Instant::now();
        jit.run(&mul);
        println!("  JIT:    {:?}", start.elapsed());
    }

    // Benchmark 3: large multiply (stresses nested loop JIT)
    let large_mul = multiply_bf(200, 200);
    println!("\n=== multiply(200*200) x100 ===");
    {
        let start = Instant::now();
        for _ in 0..100 {
            interp::interpret(&large_mul);
        }
        println!("  interp: {:?}", start.elapsed());
    }
    {
        let mut jit = jit_interp::JitBrainInterp::new();
        let start = Instant::now();
        for _ in 0..100 {
            jit.run(&large_mul);
        }
        println!("  JIT:    {:?}", start.elapsed());
    }

    // Benchmark 4: very large countdown
    let big_n = 1_000_000;
    let big_countdown = countdown_bf(big_n);
    println!("\n=== countdown({big_n}) ===");
    {
        let start = Instant::now();
        interp::interpret(&big_countdown);
        println!("  interp: {:?}", start.elapsed());
    }
    {
        let mut jit = jit_interp::JitBrainInterp::new();
        let start = Instant::now();
        jit.run(&big_countdown);
        println!("  JIT:    {:?}", start.elapsed());
    }
}
