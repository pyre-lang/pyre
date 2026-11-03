//! Eager compilation and checked-vs-raw backend entry cost (dynasm).
//!
//! cargo run --release -p majit --no-default-features --features dynasm --example eager_vs_jit
//!
//! Despite the historical filename, this does not run a tracing JIT.
//! Both execution panels use the SAME compiled token and machine code.
//! EAGER_BENCH_ITERS controls calls per sample (default 50,000).
//! Wall-clock minima are diagnostic, not a replacement for the thread-CPU
//! benchmark board. Run on external power before drawing performance conclusions.

use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

use majit::backend::{Backend, JitCellToken, RawExecResult};
use majit::eager::CompiledIr;
use majit::ir::descr::make_finish_descr;
use majit::ir::operand::Operand;
use majit::ir::{InputArg, InputArgRc, Op, OpCode, OpRc, OpRef, Type, Value};
use majit_backend_dynasm::runner::DynasmBackend;

const SAMPLES: usize = 7;
const WARMUP: u64 = 1_000;

fn backend() -> DynasmBackend {
    let mut cpu = DynasmBackend::new();
    majit::backend::make_and_attach_done_descrs(&mut [&mut cpu as &mut dyn Backend]);
    cpu.set_propagate_exception_descr(Arc::new(majit::backend::PropagateExceptionDescr::new()));
    cpu.setup_once();
    cpu
}

fn add_ir() -> (Vec<InputArgRc>, Vec<OpRc>) {
    let x = InputArg::new_int_rc(0);
    let y = InputArg::new_int_rc(1);
    let add = OpRc::new(Op::new(
        OpCode::IntAdd,
        &[
            Operand::from_bound_inputarg(&x),
            Operand::from_bound_inputarg(&y),
        ],
    ));
    add.pos().set(OpRef::int_op(1));
    let finish = OpRc::new(Op::with_descr(
        OpCode::Finish,
        &[Operand::from_bound_op(&add)],
        make_finish_descr(0, vec![Type::Int]),
    ));
    (vec![x, y], vec![add, finish])
}

fn arguments(i: u64) -> [Value; 2] {
    [Value::Int(i as i64), Value::Int(2)]
}

fn check(exit: RawExecResult, i: u64) {
    assert!(exit.is_finish);
    assert_eq!(
        exit.typed_outputs,
        vec![Value::Int((i as i64).wrapping_add(2))]
    );
}

fn minimum_ns_per_call(n: u64, mut run: impl FnMut(u64)) -> f64 {
    for i in 0..WARMUP {
        run(i);
    }
    (0..SAMPLES)
        .map(|_| {
            let start = Instant::now();
            for i in 0..n {
                run(i);
            }
            start.elapsed().as_secs_f64() * 1e9 / n as f64
        })
        .fold(f64::INFINITY, f64::min)
}

fn main() {
    let iterations = std::env::var("EAGER_BENCH_ITERS")
        .map(|s| {
            s.parse::<u64>()
                .expect("EAGER_BENCH_ITERS must be an integer")
        })
        .unwrap_or(50_000);
    assert!(iterations > 0, "EAGER_BENCH_ITERS must be positive");
    println!(
        "# backend=dynasm calls/sample={iterations} samples={SAMPLES} timer=wall minimum=ns/call"
    );
    println!("# Same code/token; eager panel first, raw panel second; no tracing or batching.");
    let mut cpu = backend();
    // A single standalone CPU and compilation: token 1 is unique in this example.
    let token = Arc::new(JitCellToken::new(1));
    let (inputs, ops) = add_ir();
    let start = Instant::now();
    // SAFETY: integer-only, well-formed IR, initialized CPU, fresh token.
    let mut compiled = unsafe { CompiledIr::compile(&mut cpu, Arc::clone(&token), &inputs, &ops) }
        .expect("eager compile");
    let compile_ns = start.elapsed().as_secs_f64() * 1e9;

    let start = Instant::now();
    // SAFETY: the IR uses integers only; no pointers, allocation or host calls.
    let first = unsafe { compiled.execute(black_box(&arguments(40))) }.unwrap();
    let first_ns = start.elapsed().as_secs_f64() * 1e9;
    check(first, 40);
    for i in [0, 1, 42, i64::MAX as u64, u64::MAX] {
        check(unsafe { compiled.execute(&arguments(i)) }.unwrap(), i);
    }

    let eager_ns = minimum_ns_per_call(iterations, |i| {
        // SAFETY: same integer-only program and two correctly typed arguments.
        black_box(unsafe { compiled.execute(black_box(&arguments(i))) }.unwrap());
    });
    drop(compiled);

    // Keep the dynamic Backend dispatch used inside CompiledIr::execute.
    let cpu: &dyn Backend = &cpu;
    for i in [0, 1, 42, i64::MAX as u64, u64::MAX] {
        check(cpu.execute_token_raw(&token, &arguments(i)), i);
    }
    let raw_ns = minimum_ns_per_call(iterations, |i| {
        black_box(cpu.execute_token_raw(&token, black_box(&arguments(i))));
    });
    println!("compile (one cold sample)    {compile_ns:>12.1} ns");
    println!("first eager call            {first_ns:>12.1} ns");
    println!("steady eager checked        {eager_ns:>12.1} ns/call");
    println!("steady backend raw          {raw_ns:>12.1} ns/call");
    println!("# Setup/lowering excluded; steady includes argument construction and result drop.");
    println!("# Sequential panels can drift; neither path removes JITFRAME or GC entry costs.");
}
