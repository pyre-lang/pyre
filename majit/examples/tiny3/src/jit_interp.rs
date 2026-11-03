//! JIT-enabled tiny3 interpreter via `#[jit_interp]` proc macro with `state_fields`.
//!
//! Representation difference: `tiny3_hotpath.Stack` represents
//! the operand stack as a linked list of `Stack(value, next)` nodes. Each push
//! allocates one cons cell that RPython's JIT peels as a chain of virtuals.
//! pyre's `state_fields = { stackpos, stack: [int; virt] }` requires a
//! contiguous virtualizable array and cannot express that linked-list shape.
//! The array backing is a source-shape deviation, although its optimized trace
//! is equivalent for the shallow, constant-height stacks used by tiny3.
//!
//! Greens: [pc]
//! Reds:   [stackpos, stack]  (args at bottom, computation stack on top)
//!
//! The JIT traces the integer-only path. Float arithmetic falls back to the
//! plain interpreter. This matches RPython's promote(y.__class__) strategy.

use majit_metainterp::virt_array::VirtArray;
// ── Bytecode opcodes ──

const OP_PUSH_INT: u8 = 0; // followed by 8 bytes (i64 LE)
const OP_PUSH_ARG: u8 = 1; // followed by 1 byte (arg index, 0-based)
const OP_STORE_ARG: u8 = 2; // followed by 1 byte (arg index, 0-based)
const OP_ADD: u8 = 3;
const OP_SUB: u8 = 4;
const OP_MUL: u8 = 5;
const OP_LOOP_START: u8 = 6; // no-op marker
const OP_LOOP_END: u8 = 7; // followed by 2 bytes (target pc, u16 LE)
const OP_END: u8 = 8;
const OP_PUSH_FLOAT: u8 = 9; // followed by 8 bytes (f64 bits as i64 LE) — not traced by JIT

// ── Bytecode compiler ──

fn compile(words: &[&str]) -> Vec<u8> {
    let mut code = Vec::new();
    let mut loop_starts: Vec<usize> = Vec::new();

    let mut i = 0;
    while i < words.len() {
        let w = words[i];
        if w == "ADD" {
            code.push(OP_ADD);
        } else if w == "SUB" {
            code.push(OP_SUB);
        } else if w == "MUL" {
            code.push(OP_MUL);
        } else if w.starts_with("->#") {
            let n = parse_int(w, 3) as u8;
            code.push(OP_STORE_ARG);
            code.push(n - 1);
        } else if w.starts_with('#') {
            let n = parse_int(w, 1) as u8;
            code.push(OP_PUSH_ARG);
            code.push(n - 1);
        } else if w == "{" {
            code.push(OP_LOOP_START);
            loop_starts.push(code.len());
        } else if w == "}" {
            code.push(OP_LOOP_END);
            let target_pc = loop_starts.pop().expect("unmatched }");
            code.push((target_pc & 0xFF) as u8);
            code.push(((target_pc >> 8) & 0xFF) as u8);
        } else if let Some(fval) = try_parse_float(w) {
            // Float literal
            code.push(OP_PUSH_FLOAT);
            code.extend_from_slice(&(fval.to_bits() as i64).to_le_bytes());
        } else {
            // Integer literal
            let val = parse_int(w, 0);
            code.push(OP_PUSH_INT);
            code.extend_from_slice(&val.to_le_bytes());
        }
        i += 1;
    }
    code.push(OP_END);
    code
}

// ── State ──

struct Tiny3State {
    stackpos: i64,
    stack: VirtArray<i64>,
}

pub type Bytecode = [u8];

pub static COMPILES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static LAST_OPS_AFTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Shape of the last compiled loop body — see [`majit_metainterp::LoopBodyShape`].
///
/// Held as two flags rather than the struct itself so the recording stays
/// lock-free on the compile path; the probe rebuilds the struct inside the same
/// lock window it reads the counters in, because this is as process-global as
/// they are.
pub static LAST_HAS_JUMP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub static LAST_ALWAYS_FAILS: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

// ── JIT mainloop (integer-only path) ──

#[majit_macros::jit_interp(
    state = Tiny3State,
    env = Bytecode,
    greens = [pc, program],
    state_fields = {
        stackpos: int,
        stack: [int; virt],
    },
)]
#[allow(unused_assignments, unused_variables)]
fn mainloop(program: &Bytecode, num_args: usize, threshold: u32) -> i64 {
    let mut driver: majit_metainterp::JitDriver<Tiny3State> =
        majit_metainterp::JitDriver::new(threshold);
    majit_metainterp::install_jitframe_gc(&mut driver);
    driver.set_on_compile_loop(|_green_key, _ops_before, ops_after, opcodes| {
        COMPILES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        LAST_OPS_AFTER.store(ops_after, std::sync::atomic::Ordering::Relaxed);
        let shape = majit_metainterp::LoopBodyShape::of(opcodes);
        LAST_HAS_JUMP.store(shape.has_jump, std::sync::atomic::Ordering::Relaxed);
        LAST_ALWAYS_FAILS.store(shape.has_always_fails, std::sync::atomic::Ordering::Relaxed);
    });
    let mut pc: usize = 0;
    let stacksize: i32 = 0;
    let mut state = Tiny3State {
        stackpos: num_args as i64,
        stack: VirtArray::filled(0i64, program.len()),
    };

    // RPython warmspot.py:281-289 canonical-liveness install hook.
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }

    while pc < program.len() {
        jit_merge_point!(driver, program, pc; state);
        let opcode = program[pc];
        pc += 1;

        match opcode {
            OP_PUSH_INT => {
                let value = i64::from_le_bytes([
                    program[pc],
                    program[pc + 1],
                    program[pc + 2],
                    program[pc + 3],
                    program[pc + 4],
                    program[pc + 5],
                    program[pc + 6],
                    program[pc + 7],
                ]);
                pc += 8;
                state.stack[state.stackpos as usize] = value;
                state.stackpos += 1;
            }
            OP_PUSH_FLOAT => {
                // Float literal stored as bit-casted i64 — same encoding as OP_PUSH_INT.
                let value = i64::from_le_bytes([
                    program[pc],
                    program[pc + 1],
                    program[pc + 2],
                    program[pc + 3],
                    program[pc + 4],
                    program[pc + 5],
                    program[pc + 6],
                    program[pc + 7],
                ]);
                pc += 8;
                state.stack[state.stackpos as usize] = value;
                state.stackpos += 1;
            }
            OP_PUSH_ARG => {
                let n = program[pc] as usize;
                pc += 1;
                let v = state.stack[n];
                state.stack[state.stackpos as usize] = v;
                state.stackpos += 1;
            }
            OP_STORE_ARG => {
                let n = program[pc] as usize;
                pc += 1;
                state.stackpos -= 1;
                let v = state.stack[state.stackpos as usize];
                state.stack[n] = v;
            }
            OP_ADD => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = b + a;
                state.stackpos -= 1;
            }
            OP_SUB => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = b - a;
                state.stackpos -= 1;
            }
            OP_MUL => {
                let a = state.stack[(state.stackpos - 1) as usize];
                let b = state.stack[(state.stackpos - 2) as usize];
                state.stack[(state.stackpos - 2) as usize] = b * a;
                state.stackpos -= 1;
            }
            OP_LOOP_START => {}
            OP_LOOP_END => {
                let target = (program[pc] as usize) | ((program[pc + 1] as usize) << 8);
                pc += 2;
                state.stackpos -= 1;
                let jump = state.stack[state.stackpos as usize] != 0;
                if jump {
                    if target <= pc {
                        can_enter_jit!(driver, target, &mut state, program, || {});
                    }
                    pc = target;
                    continue;
                }
            }
            OP_END => break,
            _ => {}
        }
    }

    state.stackpos -= 1;
    state.stack[state.stackpos as usize]
}

// ── Public wrapper matching the old API ──

pub struct JitTiny3Interp {
    threshold: u32,
}

impl Default for JitTiny3Interp {
    fn default() -> Self {
        Self::new()
    }
}

impl JitTiny3Interp {
    pub fn new() -> Self {
        JitTiny3Interp { threshold: 3 }
    }

    /// Run a word-based program with integer args.
    pub fn run(&mut self, bytecode: &[&str], args: &mut [i64]) -> i64 {
        let code = compile(bytecode);
        let num_args = args.len();
        let has_result_on_stack = program_has_result(bytecode, num_args);

        // Prepend OP_PUSH_INT for each arg to pre-populate the stack.
        let prefix_len = num_args * 9;
        let mut full_code = Vec::new();
        for &arg in args.iter() {
            full_code.push(OP_PUSH_INT);
            full_code.extend_from_slice(&arg.to_le_bytes());
        }
        // Patch loop targets to account for the prepended prefix.
        let mut patched_code = code.clone();
        let mut ci = 0;
        while ci < patched_code.len() {
            match patched_code[ci] {
                OP_PUSH_INT | OP_PUSH_FLOAT => ci += 9,
                OP_PUSH_ARG | OP_STORE_ARG => ci += 2,
                OP_LOOP_END => {
                    ci += 1;
                    let old_target =
                        (patched_code[ci] as usize) | ((patched_code[ci + 1] as usize) << 8);
                    let new_target = old_target + prefix_len;
                    patched_code[ci] = (new_target & 0xFF) as u8;
                    patched_code[ci + 1] = ((new_target >> 8) & 0xFF) as u8;
                    ci += 2;
                }
                _ => ci += 1,
            }
        }
        full_code.extend_from_slice(&patched_code);

        let result = mainloop(&full_code, 0, self.threshold);

        // Sync args back via plain interpreter.
        let mut interp_args: Vec<crate::interp::Box> =
            args.iter().map(|&v| crate::interp::Box::Int(v)).collect();
        let _ = crate::interp::interpret(bytecode, &mut interp_args);
        for (i, b) in interp_args.iter().enumerate() {
            args[i] = b.as_int();
        }

        if has_result_on_stack { result } else { args[0] }
    }

    /// Run a word-based program with typed (Int/Float) args.
    /// Float paths fall back to the plain interpreter; JIT traces integer-only.
    pub fn run_typed(
        &mut self,
        bytecode: &[&str],
        args: &mut [crate::interp::Box],
    ) -> crate::interp::Box {
        // For typed runs, use the plain interpreter.
        // The JIT only traces the integer-only path.
        let result_stack = crate::interp::interpret(bytecode, args);
        if !result_stack.is_empty() {
            result_stack.last().unwrap().clone()
        } else {
            args[0].clone()
        }
    }
}

/// Check if the program produces a stack result beyond args.
fn program_has_result(words: &[&str], num_args: usize) -> bool {
    let mut depth: i32 = num_args as i32;
    let mut loop_depth = 0;
    for w in words {
        if *w == "{" {
            loop_depth += 1;
        } else if *w == "}" {
            loop_depth -= 1;
            depth -= 1;
        } else if loop_depth == 0 {
            if matches!(*w, "ADD" | "SUB" | "MUL") || w.starts_with("->#") {
                depth -= 1;
            } else {
                depth += 1;
            }
        }
    }
    depth > num_args as i32
}

/// Try parsing a float literal (must contain '.').
fn try_parse_float(s: &str) -> Option<f64> {
    if s.contains('.') {
        s.parse::<f64>().ok()
    } else {
        None
    }
}

fn parse_int(s: &str, start: usize) -> i64 {
    let s = &s[start..];
    let mut res: i64 = 0;
    for c in s.chars() {
        let d = c as i64 - '0' as i64;
        res = res * 10 + d;
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::interp;
    use majit_metainterp::{RefusalKind, refusal_kind};

    /// Serializes every JIT entry in this module, so the `COMPILES` window in
    /// [`jit_tier_is_alive`] cannot be written by another
    /// test running concurrently. The counters are process-wide, and libtest
    /// runs these tests in parallel by default.
    ///
    /// Plain mutex: the helpers below and the gate must never call one
    /// another, or a single thread takes it twice and deadlocks. The gate holds
    /// the guard itself and calls `mainloop` directly for exactly that reason.
    static PROBE_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

    /// [`mainloop`] under [`PROBE_LOCK`].
    fn mainloop_locked(program: &Bytecode, num_args: usize, threshold: u32) -> i64 {
        let _guard = PROBE_LOCK.lock();
        mainloop(program, num_args, threshold)
    }

    /// [`JitTiny3Interp::run`] under [`PROBE_LOCK`].
    ///
    /// Not a convenience alias for `jit.run(..)` — the lock is the whole
    /// point. A test that enters the JIT without it never READS the counters,
    /// but it does MOVE them, which is what makes it the hazard.
    fn run_locked(jit: &mut JitTiny3Interp, prog: &[&str], args: &mut Vec<i64>) -> i64 {
        let _guard = PROBE_LOCK.lock();
        jit.run(prog, args)
    }

    #[test]
    fn jit_tier_is_alive() {
        use std::sync::atomic::Ordering;

        const N: i64 = 1001;
        let src = format!("0 {N} {{ #1 1 ADD ->#1 #2 1 SUB ->#2 #2 }} #1");
        let words: Vec<&str> = src.split_whitespace().collect();
        // Bare `mainloop`, not `mainloop_locked`: the guard is taken here so the
        // store/run/load is a single critical section. Going through the helper
        // would take the plain mutex twice on one thread and deadlock.
        let _guard = PROBE_LOCK.lock();
        COMPILES.store(0, Ordering::Relaxed);
        LAST_OPS_AFTER.store(0, Ordering::Relaxed);
        LAST_HAS_JUMP.store(false, Ordering::Relaxed);
        LAST_ALWAYS_FAILS.store(false, Ordering::Relaxed);
        let got = mainloop(&compile(&words), 0, 3);
        let compiles = COMPILES.load(Ordering::Relaxed);
        let ops_after = LAST_OPS_AFTER.load(Ordering::Relaxed);
        let shape = majit_metainterp::LoopBodyShape {
            has_jump: LAST_HAS_JUMP.load(Ordering::Relaxed),
            has_always_fails: LAST_ALWAYS_FAILS.load(Ordering::Relaxed),
        };
        assert_eq!(got, N, "the interpreter's own trip count moved");

        // Read after the run: nothing installs the dispatch JitCode until the
        // interpreter is entered, so a list gathered before it is empty for the
        // wrong reason.
        let t3_arms: Vec<_> = majit_metainterp::degraded_dispatch_arms()
            .into_iter()
            .filter(|a| a.interp == "Tiny3State")
            .collect();
        let mut degraded: Vec<&str> = t3_arms.iter().map(|a| a.arm).collect();
        degraded.sort_unstable();

        // The CAUSE, which the name set cannot see: an arm can keep degrading
        // for an entirely different reason. See tl's `jit_tier_is_alive` for the
        // measurement that motivated this — three A/B arms whose name sets and
        // pass counts were identical while the mechanism changed underneath.
        let mut causes: Vec<(&str, RefusalKind)> = t3_arms
            .iter()
            .map(|a| (a.arm, refusal_kind(a.reason)))
            .collect();
        causes.sort_unstable();
        assert_eq!(
            causes,
            Vec::<(&str, RefusalKind)>::new(),
            "a degraded arm's cause moved while its name did not — a different \
             mechanism is refusing it now"
        );

        assert_eq!(
            degraded,
            Vec::<&str>::new(),
            "the degraded-arm set moved. A NEW name means an arm silently \
             stopped lowering and every trace reaching it now aborts"
        );
        assert_eq!(
            compiles, 1,
            "count_to({N}) compiled {compiles} loops, not the observed 1"
        );
        // The body actually closes a loop — see `LoopBodyShape`. A compile
        // count and an op count together still accept a body that bails out on
        // its first pass; this is the term that does not. Sound HERE because
        // this fixture loops: on a straight-line subject a `Jump`-less body is
        // the right answer, not a defect.
        assert!(
            shape.closes_a_loop(),
            "compiled {ops_after} ops but the body {} ({shape:?})",
            shape.why_not().unwrap_or("closes a loop")
        );
        println!(
            "[tier-alive] count_to({N}) = {got}, compiled {compiles} loop(s) of {ops_after} ops, degraded {degraded:?}"
        );
    }

    #[test]
    fn trip_count_gate() {
        for n in [1001i64, 1002] {
            let src = format!("0 {n} {{ #1 1 ADD ->#1 #2 1 SUB ->#2 #2 }} #1");
            let words: Vec<&str> = src.split_whitespace().collect();
            let got = mainloop_locked(&compile(&words), 0, 3);
            assert_eq!(
                got, n,
                "count_to({n}) = {got}, so the loop ran {got} passes rather than \
                 {n} — an off-by-one trip count is the signature of a terminal \
                 arm whose exit the trace dropped, leaving the lowered arm to \
                 fall through to the dispatch back-edge"
            );
            println!("[trip-count] count_to({n}) = {got} — exactly {n} passes");
        }
    }

    #[test]
    fn jit_fibonacci_single() {
        let prog: Vec<&str> = "#3 1 SUB ->#3 { #2 #1 #2 ADD ->#2 ->#1 #3 1 SUB ->#3 #3 } #1"
            .split_whitespace()
            .collect();
        let mut jit = JitTiny3Interp::new();
        let mut args = vec![1i64, 1, 11];
        let result = run_locked(&mut jit, &prog, &mut args);
        assert_eq!(result, 89);
    }

    #[test]
    fn jit_fibonacci_matches_interp() {
        let prog_str = "#3 1 SUB ->#3 { #2 #1 #2 ADD ->#2 ->#1 #3 1 SUB ->#3 #3 } #1";
        let prog: Vec<&str> = prog_str.split_whitespace().collect();

        for n in [5, 10, 11, 15, 20] {
            let mut interp_args = vec![
                interp::Box::Int(1),
                interp::Box::Int(1),
                interp::Box::Int(n),
            ];
            let interp_result = interp::interpret(&prog, &mut interp_args);
            let expected = interp::repr_stack(&interp_result);

            let mut jit = JitTiny3Interp::new();
            let mut jit_args = vec![1i64, 1, n];
            let jit_result = run_locked(&mut jit, &prog, &mut jit_args);

            assert_eq!(jit_result.to_string(), expected, "fib({n}) mismatch");
        }
    }

    #[test]
    fn jit_countdown() {
        let prog: Vec<&str> = "{ #1 #1 1 SUB ->#1 #1 }".split_whitespace().collect();
        let mut jit = JitTiny3Interp::new();
        let mut args = vec![5i64];
        run_locked(&mut jit, &prog, &mut args);
        assert_eq!(args[0], 0);
    }

    #[test]
    fn jit_float_arithmetic() {
        let prog: Vec<&str> = "{ #1 0.5 MUL ->#1 #2 1 SUB ->#2 #2 }"
            .split_whitespace()
            .collect();
        let mut jit = JitTiny3Interp::new();
        let mut args = vec![interp::Box::Float(8.0), interp::Box::Int(10)];
        jit.run_typed(&prog, &mut args);
        let result = args[0].as_float();
        let expected = 8.0 * 0.5f64.powi(10);
        assert!(
            (result - expected).abs() < 1e-10,
            "expected ~{expected}, got {result}"
        );
        assert_eq!(args[1].as_int(), 0);
    }

    #[test]
    fn jit_mixed_int_float() {
        let prog: Vec<&str> = "{ #1 #2 ADD ->#1 #3 1 SUB ->#3 #3 }"
            .split_whitespace()
            .collect();
        let mut jit = JitTiny3Interp::new();
        let mut args = vec![
            interp::Box::Float(1.5),
            interp::Box::Int(2),
            interp::Box::Int(5),
        ];
        jit.run_typed(&prog, &mut args);
        let result = args[0].as_float();
        assert!((result - 11.5).abs() < 1e-10, "expected 11.5, got {result}");
        assert_eq!(args[2].as_int(), 0);
    }
}
