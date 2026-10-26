//! Opaque Rust float predicates must lower to ll_math's arithmetic graphs.
use majit_charon_reader::Llbc;
use majit_translate::{
    front::mir::lower_function,
    model::{CallTarget, OpKind},
};

#[test]
#[ignore = "requires extracted pyre-interpreter.ullbc"]
fn float_power_lowers_infinity_predicates_to_arithmetic() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../target/llbc/pyre-interpreter.ullbc"
    );
    let llbc = Llbc::load(path).expect("extract interpreter before running this test");
    let graph = lower_function(&llbc, "float_pow_inner").expect("lower float power");
    let ops: Vec<_> = graph.blocks.iter().flat_map(|b| &b.operations).collect();
    assert!(!ops.iter().any(|op| matches!(&op.kind,
        OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
        if segments.last().is_some_and(|s| s == "is_infinite"))));
    assert!(ops.iter().any(|op| matches!(op.kind,
        OpKind::ConstFloat(bits) if bits == (2.0_f64.powi(1020)).to_bits())));
    // ll_math_isinf's jitted expression must reject finite extrema and NaN,
    // and accept both signs of infinity without inspecting integer bits.
    let large = 2.0_f64.powi(1020);
    for x in [
        0.0,
        -0.0,
        1.0,
        -1.0,
        f64::MAX,
        f64::MIN,
        f64::MIN_POSITIVE,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        assert_eq!((x + large) == x, x.is_infinite());
    }
}
