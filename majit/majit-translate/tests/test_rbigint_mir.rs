//! Translation coverage for the RPython `rbigint` port.
//!
//! `RBigInt` is part of the interpreter's translated program, not a foreign
//! arithmetic library.  Its inherent methods therefore have to remain real
//! flow graphs under the same `[module::RBigInt, method]` keys emitted at
//! call sites.

mod common;

use common::{
    INTERPRETER_LLBC, MODULE_LLBC, load_llbc, load_llbc_if_present, lower_named, rbigint_llbcs,
};
use majit_charon_reader::Llbc;
use majit_translate::{
    HostStaticAddrs,
    front::{
        llbc_hints::harvest_hints_from_llbcs,
        mir::{
            build_semantic_program_from_llbcs_with_static_addrs_and_function_names,
            build_semantic_program_from_llbcs_with_static_addrs_and_module_paths,
        },
    },
    model::{CallTarget, OpKind, ValueType},
};

const RBIGINT_RS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../majit/majit-rlib/src/rbigint.rs"
);

/// `_AsDouble` consumes `bit_length()` with `?`, so the `Result<i64,
/// RBigIntError>` it returns is destructured by a live `Try::branch` /
/// `from_residual` diamond.  A one-word scalar residual would erase that
/// `Result` and leave the diamond reading `__discriminant` off an integer, so
/// `scalar_residual_for_method` maps neither `bit_length` nor `bit_count` —
/// matching upstream, which mints no residual for them either.
#[test]
fn as_double_effect_graph_keeps_upstream_bit_length_call() {
    let Some(llbcs) = rbigint_llbcs() else {
        return;
    };
    let graph = lower_named(llbcs[0], "rbigint::_AsDouble")
        .or_else(|_| lower_named(llbcs[0], "_AsDouble"))
        .expect("lower rbigint::_AsDouble");
    let calls: Vec<String> = graph
        .blocks
        .iter()
        .flat_map(|block| block.operations.iter())
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => segments.last().cloned(),
            OpKind::Call {
                target: CallTarget::Method { name, .. },
                ..
            } => Some(name.clone()),
            _ => None,
        })
        .collect();
    assert!(
        calls.iter().any(|name| name == "bit_length"),
        "_AsDouble must retain upstream RBigInt::bit_length for effect analysis: {calls:?}",
    );
    assert!(
        calls.iter().all(|name| name != "jit_bigint_bit_length"),
        "_AsDouble must not acquire a one-word residual for a Result-returning \
         call: {calls:?}",
    );
}

/// `RBigInt::floordiv` is `self.divmod(other)?`. The `Result` is not stamped
/// `Ok`: the call stays `RBigInt::divmod`, and `?` switches on
/// `Result<(RBigInt, RBigInt), RBigIntError>`'s discriminant, `Err` payload
/// included. `RBigInt::divmod`'s `return this.int_divmod(...)` is the
/// function result and stays a `Result` call too. `expect` consumers
/// (`integer_divmod_pair`) keep the pair residual.
#[test]
fn floordiv_keeps_divmod_result_for_try_branch() {
    let Some(llbcs) = rbigint_llbcs() else {
        return;
    };
    let call_leaves = |graph: &majit_translate::model::FunctionGraph| -> Vec<String> {
        graph
            .blocks
            .iter()
            .flat_map(|block| block.operations.iter())
            .filter_map(|op| match &op.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } => segments.last().cloned(),
                OpKind::Call {
                    target: CallTarget::Method { name, .. },
                    ..
                } => Some(name.clone()),
                _ => None,
            })
            .collect()
    };
    let reads_field = |graph: &majit_translate::model::FunctionGraph, name: &str, owner: &str| {
        graph.blocks.iter().any(|block| {
            block.operations.iter().any(|op| {
                matches!(
                    &op.kind,
                    OpKind::FieldRead { field, .. }
                        if field.name == name
                            && field
                                .owner_root
                                .as_deref()
                                .is_some_and(|root| root.contains(owner))
                )
            })
        })
    };

    let graph = lower_named(llbcs[0], "floordiv").expect("lower floordiv");
    let calls = call_leaves(&graph);
    assert!(
        calls.iter().any(|name| name == "divmod"),
        "floordiv must keep RBigInt::divmod: {calls:?}"
    );
    assert!(
        calls.iter().all(|name| name != "jit_bigint_divmod"),
        "floordiv must not stamp divmod's Result Ok: {calls:?}"
    );
    assert!(
        reads_field(
            &graph,
            "__discriminant",
            "Result<(RBigInt,RBigInt),RBigIntError>",
        ),
        "floordiv's ? must switch on divmod's Result discriminant"
    );
    assert!(
        reads_field(
            &graph,
            "__pos_0",
            "Result<(RBigInt,RBigInt),RBigIntError>::Err",
        ),
        "floordiv's ? must read the Err payload"
    );

    let divmod = lower_named(llbcs[0], "divmod").expect("lower divmod");
    let divmod_calls = call_leaves(&divmod);
    assert!(
        divmod_calls.iter().any(|name| name == "int_divmod"),
        "divmod's returned int_divmod stays a Result call: {divmod_calls:?}"
    );
    assert!(
        divmod_calls
            .iter()
            .all(|name| name != "jit_bigint_int_divmod"),
        "divmod must not stamp the returned int_divmod Ok: {divmod_calls:?}"
    );
}

/// The same erasure, on the side the `?` diamond does not cover.
/// `rbigint_to_compiler_bigint` spells its `bit_length()` consumption as
/// `.expect(..)`, so a one-word residual leaves `Result::expect` applied to a
/// machine word.  Unlike `_AsDouble` this graph carries no `dont_look_inside`
/// marker, which is why the caller's hint could never have been the predicate
/// that kept the mapping sound.
#[test]
fn compiler_bigint_conversion_keeps_upstream_bit_length_call() {
    if !std::path::Path::new(INTERPRETER_LLBC).is_file() {
        eprintln!(
            "skipping: {INTERPRETER_LLBC} is missing; run \
             `python3 scripts/extract-llbc.py pyre-interpreter`"
        );
        return;
    }
    let llbc = load_llbc(INTERPRETER_LLBC);
    let graph =
        lower_named(llbc, "rbigint_to_compiler_bigint").expect("lower rbigint_to_compiler_bigint");
    let residuals: Vec<String> = graph
        .blocks
        .iter()
        .flat_map(|block| block.operations.iter())
        .filter_map(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => Some(segments.join("::")),
            _ => None,
        })
        .collect();
    assert!(
        residuals
            .iter()
            .all(|path| !path.ends_with("jit_bigint_bit_length")),
        "an `.expect()` consumer must not receive a one-word residual: {residuals:?}",
    );
}

/// `numdigits` reads `self._size.abs()`.  `rint.py rtype_abs` lowers the
/// `Signed` `abs` to the `int_abs` op, not a call, so the graph carries the
/// unary `abs` and no residual `core::num::<Impl>::abs`.
#[test]
fn numdigits_lowers_i64_abs_to_the_abs_op() {
    let Some(llbcs) = rbigint_llbcs() else {
        return;
    };
    let graph = lower_named(llbcs[0], "rbigint::<Impl>::numdigits")
        .or_else(|_| lower_named(llbcs[0], "numdigits"))
        .expect("lower RBigInt::numdigits");
    let ops: Vec<_> = graph
        .blocks
        .iter()
        .flat_map(|block| block.operations.iter())
        .collect();
    assert!(
        ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::UnaryOp { op, result_ty: ValueType::Int, .. } if op == "abs"
        )),
        "`i64::abs` must lower to the unary `abs` op: {ops:?}",
    );
    assert!(
        !ops.iter().any(|op| matches!(
            &op.kind,
            OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                if segments.last().is_some_and(|s| s == "abs")
        )),
        "`i64::abs` must not stay a residual call: {ops:?}",
    );
}

fn assert_source_order(source: &str, fragments: &[&str]) {
    let mut cursor = 0;
    for fragment in fragments {
        let relative = source[cursor..]
            .find(fragment)
            .unwrap_or_else(|| panic!("missing `{fragment}` after byte {cursor}"));
        cursor += relative + fragment.len();
    }
}

#[test]
fn mapped_rbigint_methods_and_helpers_follow_upstream_source_order() {
    let source = std::fs::read_to_string(RBIGINT_RS).expect("read Rust rbigint port");
    let core_start = source.find("impl RBigInt {").expect("primary RBigInt impl");
    let compatibility_start = source
        .find("/// Rust ecosystem compatibility surface")
        .expect("compatibility RBigInt impl");
    let core = &source[core_start..compatibility_start];

    // rpython/rlib/rbigint.py.  Each suffix is one concrete graph
    // emitted by the corresponding `@specialize.argtype(0)` decorator.
    assert_source_order(
        &source[..core_start],
        &[
            "fn _mask_digit(",
            "fn _mask_udigit(",
            "fn _mask_widedigit(",
            "fn _mask_uwidedigit(",
            "fn _widen_digit(",
            "fn _unsigned_widen_digit(",
            "fn _store_digit(",
            "fn _store_udigit(",
            "fn _store_widedigit(",
            "fn _store_uwidedigit(",
            "fn _load_unsigned_digit(",
        ],
    );

    // rpython/rlib/rbigint.py. Rust-only allocation and storage
    // helpers may appear between these methods, but every translated upstream
    // method must retain the original relative order. Python special methods
    // represented by Rust traits are deliberately outside this inherent impl.
    assert_source_order(
        core,
        &[
            "fn get_sign(",
            "fn _set_sign(",
            "fn digit(",
            "fn widedigit(",
            "fn uwidedigit(",
            "fn udigit(",
            "fn setdigit(",
            "fn setdigit_udigit(",
            "fn setdigit_widedigit(",
            "fn setdigit_uwidedigit(",
            "fn numdigits(",
            "fn fromint(",
            "fn frombool(",
            "fn fromlong(",
            "fn fromfloat(",
            "fn _fromfloat_finite(",
            "fn fromrarith_int(",
            "fn fromrarith_uint(",
            "fn fromdecimalstr(",
            "fn fromstr(",
            "fn _from_numberstring_parser(",
            "fn frombytes(",
            "fn tobytes(",
            "fn toint(",
            "fn _toint_helper(",
            "fn fits_int(",
            "fn tolonglong(",
            "fn tobool(",
            "fn touint(",
            "fn _touint_helper(",
            "fn toulonglong(",
            "fn uintmask(",
            "fn ulonglongmask(",
            "fn tofloat(",
            "fn format(",
            "fn repr(",
            "fn str(",
            "fn eq(",
            "fn int_eq(",
            "fn ne(",
            "fn int_ne(",
            "fn lt(",
            "fn int_lt(",
            "fn le(",
            "fn int_le(",
            "fn gt(",
            "fn int_gt(",
            "fn ge(",
            "fn int_ge(",
            "fn hash(",
            "fn add(",
            "fn int_add(",
            "fn add_int_int_bigint_result(",
            "fn _add_int_int_helper(",
            "fn sub(",
            "fn int_sub(",
            "fn sub_int_int_bigint_result(",
            "fn mul(",
            "fn int_mul(",
            "fn mul_int_int_bigint_result(",
            "fn truediv(",
            "fn floordiv(",
            "fn div(",
            "fn int_floordiv(",
            "fn int_div(",
            "fn r#mod(",
            "fn int_mod(",
            "fn int_mod_int_result(",
            "fn divmod(",
            "fn _divmod_small(",
            "fn int_divmod(",
            "fn pow(",
            "fn int_pow(",
            "fn neg(",
            "fn abs(",
            "fn invert(",
            "fn lshift(",
            "fn lqshift(",
            "fn lshift_int_int_bigint_result(",
            "fn rshift(",
            "fn rqshift(",
            "fn abs_rshift_and_mask(",
            "fn from_list_n_bits(",
            "fn and_(",
            "fn int_and_(",
            "fn xor(",
            "fn int_xor(",
            "fn or_(",
            "fn int_or_(",
            "fn oct(",
            "fn hex(",
            "fn log(",
            "fn tolong(",
            "fn _normalize(",
            "fn bit_length(",
            "fn bit_count(",
            "fn gcd(",
            "fn isqrt(",
        ],
    );
    assert!(
        !core.contains("fn from_u128("),
        "Rust ecosystem compatibility API leaked into the upstream method block"
    );
    for signed_result in [
        "pub fn numdigits(&self) -> i64",
        "pub fn bit_length(&self) -> Result<i64, RBigIntError>",
        "pub fn bit_count(&self) -> Result<i64, RBigIntError>",
    ] {
        assert!(
            core.contains(signed_result),
            "upstream Signed result changed shape: {signed_result}"
        );
    }

    // rpython/rlib/rbigint.py. Owner structs and Rust GC hooks are
    // allowed between mapped helpers; the upstream algorithm sequence is not.
    let helpers = &source[compatibility_start..];
    assert_source_order(
        helpers,
        &[
            "fn _help_mult(",
            "fn digits_from_nonneg_long(",
            "fn digits_from_nonneg_ulong(",
            "fn digits_for_most_neg_long(",
            "fn args_from_rarith_int1(",
            "fn args_from_rarith_int(",
            "fn args_from_rarith_uint1(",
            "fn args_from_rarith_uint(",
            "fn args_from_long(",
            "fn _x_add(",
            "fn _x_int_add(",
            "fn _x_sub(",
            "fn _x_int_sub(",
            "fn _x_mul(",
            "fn _kmul_split(",
            "fn _k_mul(",
            "fn _inplace_divrem1(",
            "fn _divrem1(",
            "fn _int_rem_core(",
            "fn _v_iadd(",
            "fn _v_isub(",
            "fn _muladd1(",
            "fn _v_lshift(",
            "fn _v_rshift(",
            "fn _x_divrem(",
            "fn _divrem(",
            "fn _extract_digits(",
            "fn div2n1n(",
            "fn div3n2n(",
            "fn _full_digits_lshift_then_or(",
            "fn _divmod_fast_pos(",
            "fn divmod_big(",
            "fn _x_int_lt(",
            "fn _AsScaledDouble(",
            "fn _AsDouble(",
            "fn _loghelper_ln(",
            "fn _loghelper_log10(",
            "fn _loghelper_log2(",
            "fn bits_in_digit(",
            "fn bit_length_int(",
            "fn bit_count_digit(",
            "fn _bitcount64(",
            "fn _bitcount64_ops(",
            "fn _truediv_result(",
            "fn _truediv_overflow",
            "fn _bigint_true_divide(",
            "fn _format_base2_notzero(",
            "fn get_cached_parts(",
            "fn _format_int_general(",
            "fn _format_int10(",
            "fn _format_int10_18digits(",
            "fn _format_recursive_decimal(",
            "fn _format_recursive_general(",
            "fn _format_lowest_level_divmod_int_results(",
            "fn _format(",
            "fn _bitwise_and(",
            "fn _bitwise_or(",
            "fn _bitwise_xor(",
            "fn _int_bitwise_and(",
            "fn _int_bitwise_or(",
            "fn _int_bitwise_xor(",
            "fn _AsLongLong(",
            "fn _AsULonglong_ignore_sign(",
            "fn make_unsigned_mask_conversion(",
            "fn _As_unsigned_mask(",
            "fn _hash(",
            "fn digits_max_for_base(",
            "fn _decimalstr_to_bigint(",
            "fn parse_digit_string(",
            "fn _str_to_int_big_w5pow(",
            "fn _str_to_int_big_inner10(",
            "fn _str_to_int_big_base10(",
            "fn parse_string_from_binary_base(",
            "fn gcd_binary(",
            "fn lehmer_xgcd(",
            "fn gcd_lehmer(",
            "fn frombytes_int(",
            "fn tobytes_int(",
        ],
    );
    for runtime_discriminator in ["enum LogKind", "enum FormatIntKind", "enum BitOp"] {
        assert!(
            !helpers.contains(runtime_discriminator),
            "upstream @specialize.arg graph was collapsed behind {runtime_discriminator}"
        );
    }
}

#[test]
fn rbigint_impl_methods_preserve_upstream_elidable_markers() {
    let Some(llbcs) = rbigint_llbcs() else {
        return;
    };
    let hints = harvest_hints_from_llbcs(&llbcs);
    // Exhaustive mapping of every `@jit.elidable` method in the pinned
    // rpython/rlib/rbigint.py.  This is a performance contract, not just
    // metadata: missing one changes whether callers residualize a pure bigint
    // operation or trace through its implementation. `fromrarith_uint` is the
    // unsigned Rust specialization of upstream's
    // `@specialize.argtype(0) fromrarith_int`.
    for name in [
        "fromint",
        "frombool",
        "fromfloat",
        "_fromfloat_finite",
        "fromrarith_int",
        "fromrarith_uint",
        "fromdecimalstr",
        "fromstr",
        "frombytes",
        "tobytes",
        "toint",
        "_toint_helper",
        "tolonglong",
        "touint",
        "_touint_helper",
        "toulonglong",
        "uintmask",
        "ulonglongmask",
        "tofloat",
        "format",
        "repr",
        "str",
        "eq",
        "int_eq",
        "lt",
        "int_lt",
        "hash",
        "add",
        "int_add",
        "add_int_int_bigint_result",
        "sub",
        "int_sub",
        "sub_int_int_bigint_result",
        "mul",
        "int_mul",
        "mul_int_int_bigint_result",
        "truediv",
        "int_floordiv",
        "int_mod",
        "int_mod_int_result",
        "divmod",
        "int_divmod",
        "pow",
        "int_pow",
        "neg",
        "abs",
        "invert",
        "lshift",
        "lqshift",
        "lshift_int_int_bigint_result",
        "rshift",
        "rqshift",
        "abs_rshift_and_mask",
        "and_",
        "int_and_",
        "xor",
        "int_xor",
        "or_",
        "int_or_",
        "oct",
        "hex",
        "log",
        "bit_length",
        "bit_count",
    ] {
        let path = format!("rbigint::<Impl>::{name}");
        assert!(
            hints
                .get(&path)
                .is_some_and(|values| values.iter().any(|hint| hint == "elidable")),
            "missing upstream @jit.elidable marker for {path}: {hints:?}"
        );
    }
    // Exhaustive set of mapped core methods that upstream deliberately does
    // not mark elidable. They are projections, relational compositions,
    // mutators, or entry points whose pure callees carry the effect.
    for name in [
        "get_sign",
        "_set_sign",
        "digit",
        "widedigit",
        "uwidedigit",
        "udigit",
        "setdigit",
        "setdigit_udigit",
        "setdigit_widedigit",
        "setdigit_uwidedigit",
        "numdigits",
        "_from_numberstring_parser",
        "fits_int",
        "tobool",
        "ne",
        "int_ne",
        "le",
        "int_le",
        "gt",
        "int_gt",
        "ge",
        "int_ge",
        "_add_int_int_helper",
        "floordiv",
        "div",
        "mod",
        "_divmod_small",
        "int_div",
        "from_list_n_bits",
        "_normalize",
        "gcd",
        "isqrt",
    ] {
        let path = format!("rbigint::<Impl>::{name}");
        assert!(
            !hints
                .get(&path)
                .is_some_and(|values| values.iter().any(|hint| hint == "elidable")),
            "{path} is an upstream thin projection, not an @jit.elidable body"
        );
    }
    for name in [
        "_divrem1",
        "_divrem",
        "bit_length_int",
        "_bitcount64",
        "gcd_binary",
        "gcd_lehmer",
        "frombytes_int",
        "tobytes_int",
    ] {
        let path = format!("rbigint::{name}");
        assert!(
            hints
                .get(&path)
                .is_some_and(|values| values.iter().any(|hint| hint == "elidable")),
            "missing upstream @jit.elidable marker for {path}: {hints:?}"
        );
    }
    for name in ["fromlong", "tolong"] {
        let path = format!("rbigint::<Impl>::{name}");
        assert!(
            hints
                .get(&path)
                .is_some_and(|values| values.iter().any(|hint| hint == "not_rpython")),
            "missing upstream @not_rpython marker for {path}: {hints:?}"
        );
    }
    assert!(
        hints
            .get("rbigint::args_from_long")
            .is_some_and(|values| values.iter().any(|hint| hint == "not_rpython")),
        "missing upstream @not_rpython marker for rbigint::args_from_long: {hints:?}"
    );
    for name in [
        "digit",
        "widedigit",
        "uwidedigit",
        "udigit",
        "setdigit",
        "setdigit_udigit",
        "setdigit_widedigit",
        "setdigit_uwidedigit",
        "numdigits",
        "lshift",
        "lqshift",
        "_normalize",
    ] {
        let path = format!("rbigint::<Impl>::{name}");
        assert!(
            hints
                .get(&path)
                .is_some_and(|values| values.iter().any(|hint| hint == "always_inline")),
            "missing upstream _always_inline_ marker for {path}: {hints:?}"
        );
    }
    for name in [
        "int_in_valid_range",
        "_load_unsigned_digit",
        "_bitcount64_ops",
        "_format_lowest_level_divmod_int_results",
    ] {
        let path = format!("rbigint::{name}");
        assert!(
            hints
                .get(&path)
                .is_some_and(|values| values.iter().any(|hint| hint == "always_inline")),
            "missing upstream _always_inline_=True marker for {path}: {hints:?}"
        );
    }
    assert!(
        hints
            .get("rbigint::<Impl>::rshift")
            .is_some_and(|values| values.iter().any(|hint| hint == "always_inline_try")),
        "missing upstream _always_inline_='try' marker for rbigint::<Impl>::rshift: {hints:?}"
    );
    assert!(
        !hints.contains_key("rbigint::<Impl>::rshift::try_rshift"),
        "_always_inline_try_ must not also match the shorter _always_inline_ prefix"
    );
    assert!(
        hints
            .get("rbigint::_AsDouble")
            .is_some_and(|values| values.iter().any(|hint| hint == "dont_look_inside")),
        "missing upstream @jit.dont_look_inside marker for rbigint::_AsDouble: {hints:?}"
    );
    for (path, effect) in [
        ("longobject::jit_bigint_from_i64", "elidable_or_memerror"),
        ("longobject::jit_bigint_eq", "elidable_cannot_raise"),
    ] {
        let values = hints
            .get(path)
            .unwrap_or_else(|| panic!("missing pointer-ABI wrapper hints for {path}"));
        assert!(values.iter().any(|hint| hint == "elidable"));
        assert!(
            values.iter().any(|hint| hint == effect),
            "{path} must carry {effect}, got {values:?}"
        );
    }
}

#[test]
fn rbigint_inherent_constructors_keep_their_owner_and_graph() {
    let Some(llbcs) = rbigint_llbcs() else {
        return;
    };
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_module_paths(
        &llbcs,
        HostStaticAddrs::default(),
        &["rbigint", "longobject"],
    )
    .expect("lower rbigint module");

    assert_eq!(
        program.struct_fields.field_type("RBigInt", "_digits"),
        Some("[i64]"),
        "RBigInt._digits must retain rbigint.py's GcArray(Signed) annotation"
    );

    for name in [
        "new",
        "setdigit",
        "setdigit_udigit",
        "setdigit_widedigit",
        "setdigit_uwidedigit",
        "fromint",
        "fromrarith_int",
        "fromrarith_uint",
    ] {
        let function = program
            .functions
            .iter()
            .find(|function| {
                function.name == name
                    && function.self_ty_root.as_deref() == Some("rbigint::RBigInt")
            })
            .unwrap_or_else(|| panic!("missing rbigint::RBigInt::{name} graph"));
        assert_eq!(function.module_path, "rbigint::<Impl>");
    }
    for name in [
        "_mask_digit",
        "_mask_udigit",
        "_mask_widedigit",
        "_mask_uwidedigit",
        "_store_digit",
        "_store_udigit",
        "_store_widedigit",
        "_store_uwidedigit",
        "digits_from_nonneg_long",
        "digits_from_nonneg_ulong",
        "args_from_rarith_int1",
        "args_from_rarith_int",
        "args_from_rarith_uint1",
        "args_from_rarith_uint",
        "_loghelper_ln",
        "_loghelper_log10",
        "_loghelper_log2",
        "_format_recursive_decimal",
        "_format_recursive_general",
        "_bitwise_and",
        "_bitwise_or",
        "_bitwise_xor",
        "_int_bitwise_and",
        "_int_bitwise_or",
        "_int_bitwise_xor",
    ] {
        assert!(
            program
                .functions
                .iter()
                .any(|function| function.name == name && function.module_path == "rbigint"),
            "missing specialized upstream helper graph rbigint::{name}"
        );
    }

    let input_types = |name: &str, owner: Option<&str>| {
        let function = program
            .functions
            .iter()
            .find(|function| {
                function.name == name
                    && function.self_ty_root.as_deref() == owner
                    && function.module_path
                        == if owner.is_some() {
                            "rbigint::<Impl>"
                        } else {
                            "rbigint"
                        }
            })
            .unwrap_or_else(|| panic!("missing typed rbigint graph {owner:?}::{name}"));
        function
            .graph()
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .filter_map(|operation| match &operation.kind {
                OpKind::Input { ty, .. } => Some(ty.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    for (name, expected) in [
        ("_mask_digit", ValueType::Int),
        ("_mask_udigit", ValueType::Unsigned),
        ("_mask_widedigit", ValueType::Int128),
        ("_mask_uwidedigit", ValueType::UInt128),
        ("_store_digit", ValueType::Int),
        ("_store_udigit", ValueType::Unsigned),
        ("_store_widedigit", ValueType::Int128),
        ("_store_uwidedigit", ValueType::UInt128),
        ("digits_from_nonneg_long", ValueType::Int),
        ("digits_from_nonneg_ulong", ValueType::Unsigned),
        ("args_from_rarith_int1", ValueType::Int),
        ("args_from_rarith_int", ValueType::Int),
        ("args_from_rarith_uint1", ValueType::Unsigned),
        ("args_from_rarith_uint", ValueType::Unsigned),
    ] {
        assert_eq!(
            input_types(name, None),
            vec![expected],
            "wrong @specialize.argtype graph carrier for rbigint::{name}"
        );
    }
    for (name, expected) in [
        ("setdigit", ValueType::Int),
        ("setdigit_udigit", ValueType::Unsigned),
        ("setdigit_widedigit", ValueType::Int128),
        ("setdigit_uwidedigit", ValueType::UInt128),
    ] {
        assert_eq!(
            input_types(name, Some("rbigint::RBigInt")).last(),
            Some(&expected),
            "wrong @specialize.argtype value carrier for RBigInt::{name}"
        );
        assert_eq!(
            input_types(name, Some("rbigint::RBigInt")).get(1),
            Some(&ValueType::Int),
            "RPython digit index must remain Signed for RBigInt::{name}"
        );
    }
    for name in ["digit", "widedigit", "uwidedigit", "udigit"] {
        assert_eq!(
            input_types(name, Some("rbigint::RBigInt")).last(),
            Some(&ValueType::Int),
            "RPython digit index must remain Signed for RBigInt::{name}"
        );
    }
    let digit_graph = program
        .functions
        .iter()
        .find(|function| {
            function.name == "digit" && function.self_ty_root.as_deref() == Some("rbigint::RBigInt")
        })
        .expect("rbigint::RBigInt::digit graph");
    assert!(
        digit_graph.graph().blocks.iter().any(|block| {
            block.operations.iter().any(|operation| {
                matches!(
                    &operation.kind,
                    OpKind::FieldRead { field, .. } if field.name == "_digits"
                )
            })
        }),
        "RBigInt::digit must project the upstream _digits field directly"
    );
    assert!(
        !digit_graph.graph().blocks.iter().any(|block| {
            block
                .operations
                .iter()
                .any(|operation| match &operation.kind {
                    OpKind::Call {
                        target: CallTarget::Method { name, .. },
                        ..
                    } => matches!(name.as_str(), "digits" | "digits_mut"),
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } => matches!(
                        segments.last().map(String::as_str),
                        Some("digits" | "digits_mut")
                    ),
                    _ => false,
                })
        }),
        "RBigInt::digit must not retain Rust's slice-view adapter"
    );
    for name in ["lqshift", "rqshift"] {
        assert_eq!(
            input_types(name, Some("rbigint::RBigInt")).last(),
            Some(&ValueType::Int),
            "RPython quick-shift count must remain Signed for RBigInt::{name}"
        );
    }

    for name in [
        "add",
        "int_add",
        "sub",
        "int_sub",
        "int_mul",
        "int_floordiv",
        "pow",
        "int_pow",
        "abs",
        "lshift",
        "rshift",
    ] {
        let function = program
            .functions
            .iter()
            .find(|function| {
                function.name == name
                    && function.self_ty_root.as_deref() == Some("rbigint::RBigInt")
            })
            .unwrap_or_else(|| panic!("missing rbigint::RBigInt::{name} graph"));
        assert!(
            !function
                .graph()
                .blocks
                .iter()
                .flat_map(|block| &block.operations)
                .any(|operation| matches!(
                    &operation.kind,
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } if segments.last().is_some_and(|leaf| leaf == "jit_bigint_clone")
                )),
            "RBigInt::{name} must preserve upstream `return self/other` as a GC-reference alias"
        );
    }
    for name in [
        "prebuilt",
        "zero",
        "one",
        "negative_one",
        "five",
        "frombool",
    ] {
        let candidates: Vec<_> = program
            .functions
            .iter()
            .filter(|function| {
                function.name == name
                    && function.self_ty_root.as_deref() == Some("rbigint::RBigInt")
            })
            .collect();
        assert!(
            !candidates.is_empty(),
            "missing rbigint::RBigInt::{name} graph"
        );
        assert!(
            candidates.iter().all(|function| {
                function
                    .graph()
                    .blocks
                    .iter()
                    .flat_map(|block| &block.operations)
                    .all(|operation| {
                        !matches!(
                            &operation.kind,
                            OpKind::Call {
                                target: CallTarget::FunctionPath { segments, .. },
                                ..
                            } if segments.last().is_some_and(|leaf| leaf == "jit_bigint_clone")
                        )
                    })
            }),
            "RBigInt::{name} must return the process-global prebuilt payload, not a clone"
        );
    }

    let neg_candidates: Vec<_> = program
        .functions
        .iter()
        .filter(|function| {
            function.name == "neg" && function.self_ty_root.as_deref() == Some("rbigint::RBigInt")
        })
        .collect();
    let neg = neg_candidates
        .iter()
        .copied()
        .find(|function| {
            function
                .graph()
                .blocks
                .iter()
                .flat_map(|block| &block.operations)
                .all(|operation| {
                    !matches!(
                        &operation.kind,
                        OpKind::Call {
                            target: CallTarget::FunctionPath { segments, .. },
                            ..
                        } if segments.last().is_some_and(|leaf| leaf == "jit_bigint_neg")
                    )
                })
        })
        .expect("inherent rbigint::RBigInt::neg graph, distinct from the Neg trait adapter");
    let neg_calls: Vec<Vec<String>> = neg
        .graph()
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|operation| match &operation.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => Some(segments.clone()),
            _ => None,
        })
        .collect();
    assert!(
        neg_calls.iter().any(|segments| segments
            .last()
            .is_some_and(|leaf| leaf == "jit_bigint_clone")),
        "RBigInt::neg must retain the fresh shallow-copy handle upstream constructs: {neg_calls:?}"
    );
    let range_obj_to_bigint = program
        .functions
        .iter()
        .find(|function| {
            function.name == "range_obj_to_bigint" && function.module_path.ends_with("functional")
        })
        .expect("functional::range_obj_to_bigint graph");
    let range_bigint_calls: Vec<_> = range_obj_to_bigint
        .graph()
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|operation| match &operation.kind {
            OpKind::Call { target, .. } => Some(target),
            _ => None,
        })
        .collect();
    assert!(
        !range_bigint_calls.iter().any(|target| {
            matches!(
                target,
                CallTarget::FunctionPath { segments, .. }
                    if segments.last().is_some_and(|leaf| leaf == "jit_bigint_clone")
            ) || matches!(
                target,
                CallTarget::Method { name, .. } if name == "translated_alias"
            )
        }),
        "range_obj_to_bigint must preserve bigint_w's borrowed rbigint payload without a \
         residual ownership marker: {range_bigint_calls:?}"
    );
    for function in &program.functions {
        assert!(
            !function
                .graph()
                .blocks
                .iter()
                .flat_map(|block| &block.operations)
                .any(|operation| matches!(
                    &operation.kind,
                    OpKind::Call {
                        target: CallTarget::Method { name, .. },
                        ..
                    } if name == "translated_alias"
                )),
            "Rust's translated_alias ownership marker must be erased in every translated \
             RBigInt consumer, but survived in {}::{}",
            function.module_path,
            function.name
        );
    }
    // `list: &[i64]` is a raw `{items, length}` slice view: two words, the
    // item address and the item count.  The bit width stays Signed.
    let from_list_types = input_types("from_list_n_bits", Some("rbigint::RBigInt"));
    assert!(
        matches!(
            from_list_types.as_slice(),
            [ValueType::Unsigned, ValueType::Unsigned, ValueType::Int]
        ),
        "from_list_n_bits takes the slice pair and a Signed bit width: \
         {from_list_types:?}"
    );
    for name in ["_loghelper_ln", "_loghelper_log10", "_loghelper_log2"] {
        let types = input_types(name, None);
        assert!(
            matches!(types.as_slice(), [ValueType::Ref(_)]),
            "specialized logarithm graph must carry only the RBigInt argument: {types:?}"
        );
    }
    for name in ["_bitwise_and", "_bitwise_or", "_bitwise_xor"] {
        let types = input_types(name, None);
        // `rbigint.py _bitwise(a, op, b)` specialized on `op`: the graph
        // takes the two bigint payloads only. `framework.py push_roots`
        // brackets the digit-list malloc inside the graph, so no root span
        // is a parameter; and/or/xor must remain distinct specialized graphs.
        assert!(
            matches!(types.as_slice(), [ValueType::Ref(_), ValueType::Ref(_)]),
            "specialized bigint bitwise graph must carry exactly the two payloads: {types:?}"
        );
    }
    for name in ["_int_bitwise_and", "_int_bitwise_or", "_int_bitwise_xor"] {
        let types = input_types(name, None);
        assert!(
            matches!(types.as_slice(), [ValueType::Ref(_), ValueType::Int]),
            "specialized int bitwise graph must not carry an operation discriminator: {types:?}"
        );
    }
    assert_eq!(
        input_types("_format_recursive_decimal", None),
        input_types("_format_recursive_general", None),
        "recursive formatting specializations must differ only in their fixed callable"
    );
    for name in [
        "_kmul_split",
        "_v_iadd",
        "_v_isub",
        "_v_lshift",
        "_v_rshift",
        "_extract_digits",
        "div2n1n",
        "div3n2n",
        "_full_digits_lshift_then_or",
        "_format_recursive_decimal",
        "_format_recursive_general",
        "_format_lowest_level_divmod_int_results",
        "get_cached_parts",
        "digits_max_for_base",
        "_decimalstr_to_bigint",
        "_str_to_int_big_w5pow",
        "_str_to_int_big_inner10",
        "_str_to_int_big_base10",
        "tobytes_int",
    ] {
        let types = input_types(name, None);
        assert!(
            !types.contains(&ValueType::Unsigned),
            "upstream logical size/index leaked as Rust Unsigned in rbigint::{name}: {types:?}"
        );
    }
    for old_runtime_dispatch in [
        "_loghelper",
        "_format_recursive",
        "_bitwise",
        "_int_bitwise",
    ] {
        assert!(
            !program.functions.iter().any(|function| {
                function.name == old_runtime_dispatch && function.module_path == "rbigint"
            }),
            "runtime-dispatch graph rbigint::{old_runtime_dispatch} survived @specialize.arg port"
        );
    }

    let x_divrem = program
        .functions
        .iter()
        .find(|function| function.name == "_x_divrem" && function.module_path == "rbigint")
        .expect("rbigint::_x_divrem wide-digit graph");
    let x_divrem_ops = x_divrem
        .graph()
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .collect::<Vec<_>>();
    assert!(
        x_divrem_ops.iter().any(|operation| matches!(
            &operation.kind,
            OpKind::BinOp {
                result_ty: majit_translate::model::ValueType::Int128
                    | majit_translate::model::ValueType::UInt128,
                ..
            }
        )),
        "_x_divrem must preserve 128-bit LONG_TYPE/ULONG_TYPE binop results"
    );
    for name in ["fromlong", "tolong"] {
        assert!(
            !program.functions.iter().any(|function| {
                function.name == name
                    && function.self_ty_root.as_deref() == Some("rbigint::RBigInt")
            }),
            "upstream @not_rpython method RBigInt::{name} must not have a flow graph"
        );
    }
    assert!(
        !program
            .functions
            .iter()
            .any(|function| function.name == "args_from_long"),
        "upstream @not_rpython helper args_from_long must not have a flow graph"
    );

    let fromint = program
        .functions
        .iter()
        .find(|function| {
            function.name == "fromint"
                && function.self_ty_root.as_deref() == Some("rbigint::RBigInt")
        })
        .expect("fromint graph");
    let mut saw_mask = false;
    for block in &fromint.graph().blocks {
        for operation in &block.operations {
            match &operation.kind {
                OpKind::ConstInt(value) if *value == i64::MAX => {
                    saw_mask = true;
                }
                OpKind::ConstUInt(value) if *value == i64::MAX as u64 => {
                    saw_mask = true;
                }
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } => {
                    assert_ne!(
                        segments.last().map(String::as_str),
                        Some("MASK"),
                        "a NamedConst read must fold instead of becoming a call"
                    );
                }
                _ => {}
            }
        }
    }
    assert!(
        saw_mask,
        "fromint must contain the folded 63-bit digit mask"
    );

    let constructor = program
        .functions
        .iter()
        .find(|function| {
            function.name == "new" && function.self_ty_root.as_deref() == Some("rbigint::RBigInt")
        })
        .expect("new graph");
    for block in &constructor.graph().blocks {
        for operation in &block.operations {
            if let OpKind::Call { target, .. } = &operation.kind {
                assert_ne!(
                    target
                        .path_segments()
                        .and_then(|parts| parts.last().copied()),
                    Some("is_ascii"),
                    "the translated constructor must not retain debug formatting machinery: \
                     {target:?}"
                );
            }
        }
    }

    let gt = program
        .functions
        .iter()
        .find(|function| {
            function.name == "gt" && function.self_ty_root.as_deref() == Some("rbigint::RBigInt")
        })
        .expect("RBigInt::gt graph");
    assert!(
        gt.graph()
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .any(|operation| matches!(
                &operation.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments
                    == &["pyre_object", "longobject", "jit_bigint_lt"]
            )),
        "the local Method-form `other.lt(self)` must become a scalar residual"
    );

    let is_zero = program
        .functions
        .iter()
        .find(|function| {
            function.name == "is_zero"
                && function.self_ty_root.as_deref() == Some("rbigint::RBigInt")
        })
        .expect("RBigInt::is_zero graph");
    assert!(
        is_zero
            .graph()
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .any(|operation| matches!(
                &operation.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments
                    == &["pyre_object", "longobject", "jit_bigint_sign_i64"]
            )),
        "the local Method-form `get_sign()` must become a scalar residual"
    );

    for (name, forbidden_leaf) in [("int_pow", "pow"), ("rshift", "rqshift")] {
        let function = program
            .functions
            .iter()
            .find(|function| {
                function.name == name
                    && function.self_ty_root.as_deref() == Some("rbigint::RBigInt")
            })
            .unwrap_or_else(|| panic!("missing rbigint::RBigInt::{name} graph"));
        for operation in function
            .graph()
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
        {
            if let OpKind::Call { target, .. } = &operation.kind {
                assert_ne!(
                    target
                        .path_segments()
                        .and_then(|parts| parts.last().copied()),
                    Some(forbidden_leaf),
                    "PyPy's specialized RBigInt::{name} body must not delegate to \
                     RBigInt::{forbidden_leaf}: {target:?}"
                );
            }
        }
    }
}

/// Callers whose mixed long/int arm must reach the machine-word residual
/// named beside it, and must not fall back to a full `RBigInt::int_*` call.
const MIXED_INT_RESIDUAL_CALLERS: &[(&str, &str)] = &[
    ("long_add", "jit_bigint_int_add"),
    ("long_sub", "jit_bigint_int_sub"),
    ("long_mul", "jit_bigint_int_mul"),
    ("long_bitand", "jit_bigint_int_and"),
    ("long_bitor", "jit_bigint_int_or"),
    ("long_bitxor", "jit_bigint_int_xor"),
];

/// `(module suffix, caller)` pairs that must borrow their long operand's
/// payload instead of allocating a translated `jit_bigint_clone`.
const BORROWED_PAYLOAD_CALLERS: &[(&str, &str)] = &[
    ("objspace::descroperation", "long_pow"),
    ("objspace::descroperation", "try_int_long_pow_with_modulo"),
    ("objspace::descroperation", "long_lshift"),
    ("objspace::descroperation", "long_rshift"),
    ("objspace::descroperation", "long_floordiv"),
    ("objspace::descroperation", "long_int_floordiv"),
    ("objspace::descroperation", "long_mod"),
    ("objspace::descroperation", "long_int_mod"),
    ("objspace::descroperation", "integer_divmod_pair"),
    ("objspace::descroperation", "bigint_mod_inverse"),
    ("objspace::descroperation", "complex_richcompare"),
    ("objspace::descroperation", "compare_slot"),
    ("baseobjspace", "compute_slice_indices3_big"),
    ("baseobjspace", "uint_w"),
    ("baseobjspace", "truncatedint_w"),
    ("baseobjspace", "getindex_w"),
    ("baseobjspace", "index_int_w_preserve_negative"),
    ("baseobjspace", "space_index"),
    ("builtins", "builtin_abs"),
    ("builtins", "ensure_baseint_result"),
    ("builtins", "int_to_decimal_string"),
    ("typedef", "int_descr_new"),
    ("typedef", "slice_method_indices"),
    ("builtins", "format_index_radix"),
    ("type_methods", "format_with_spec"),
    ("builtins", "obj_to_bigint"),
    ("objspace::std::formatting", "arg_to_bigint"),
    ("module::math::interp_math", "comb"),
    ("module::math::interp_math", "perm"),
];

/// Floor-division callers and the floor-semantics residual each must take.
const FLOOR_DIVMOD_CALLERS: &[(&str, &str)] = &[
    ("long_floordiv", "jit_bigint_div_floor"),
    ("long_mod", "jit_bigint_mod_floor"),
];

/// Unary callers that must borrow their input and allocate only the result.
const UNARY_RESIDUAL_CALLERS: &[(&str, &str)] = &[
    ("bigint_neg", "jit_bigint_neg"),
    ("bigint_invert", "jit_bigint_invert"),
];

/// Graphs the test reaches for one at a time rather than through a table.
const SINGLE_GRAPHS: &[&str] = &[
    "bigint_and",
    "long_int_compare",
    "long_pow",
    "long_add",
    "long_lshift",
    "int_lshift",
    "long_int_divmod",
    "integer_divmod_pair",
];

/// Every graph name the assertions below look up.
///
/// This is what scopes the lowering: `pyre-interpreter.ullbc` declares tens
/// of thousands of functions and the module filter alone still admits
/// thousands of bodies, none of which these assertions read. Deriving the
/// list from the same tables the assertions iterate keeps the two from
/// drifting — a caller added to a table is lowered by construction, and a
/// name that goes missing fails the lookup it was added for.
fn asserted_graph_names() -> Vec<&'static str> {
    let mut names: Vec<&'static str> = SINGLE_GRAPHS.to_vec();
    names.extend(MIXED_INT_RESIDUAL_CALLERS.iter().map(|(caller, _)| *caller));
    names.extend(BORROWED_PAYLOAD_CALLERS.iter().map(|(_, caller)| *caller));
    names.extend(FLOOR_DIVMOD_CALLERS.iter().map(|(caller, _)| *caller));
    names.extend(UNARY_RESIDUAL_CALLERS.iter().map(|(caller, _)| *caller));
    names
}

#[test]
fn dependent_crate_rbigint_identity_retargets_opaque_llbc_declaration() {
    if !std::path::Path::new(INTERPRETER_LLBC).is_file() {
        eprintln!(
            "skipping: {INTERPRETER_LLBC} is missing; run \
             `python3 scripts/extract-llbc.py pyre-interpreter`"
        );
        return;
    }

    let interpreter = load_llbc(INTERPRETER_LLBC);
    let mut llbcs: Vec<&Llbc> = vec![interpreter];
    if let Some(module) = load_llbc_if_present(MODULE_LLBC) {
        llbcs.push(module);
    }
    let hints = harvest_hints_from_llbcs(&llbcs);
    for path in [
        "objspace::descroperation::jit_bigint_add",
        "objspace::descroperation::jit_bigint_int_add",
        "objspace::descroperation::jit_bigint_int_sub",
        "objspace::descroperation::jit_bigint_int_mul",
        "objspace::descroperation::jit_bigint_int_and",
        "objspace::descroperation::jit_bigint_int_or",
        "objspace::descroperation::jit_bigint_int_xor",
        "objspace::descroperation::jit_bigint_shl",
        "objspace::descroperation::jit_bigint_pow_nomod",
        "objspace::descroperation::jit_bigint_lshift_int_int_result",
        "objspace::descroperation::jit_bigint_div",
        "objspace::descroperation::jit_bigint_rem",
        "objspace::descroperation::jit_bigint_div_floor",
        "objspace::descroperation::jit_bigint_mod_floor",
        "jit_compiler_bigint_to_rbigint",
    ] {
        let values = hints
            .get(path)
            .unwrap_or_else(|| panic!("missing pointer-ABI wrapper hints for {path}"));
        assert!(values.iter().any(|hint| hint == "elidable"));
        assert!(
            values.iter().any(|hint| hint == "elidable_or_memerror"),
            "{path} must be allocation-only elidable, got {values:?}"
        );
    }
    // rbigint.py `_make_int_comparison` only reads existing digits and
    // returns a bool.  These wrappers are elidable and cannot raise, not the
    // allocation-only arithmetic contract above.
    for path in [
        "objspace::descroperation::jit_bigint_int_eq",
        "objspace::descroperation::jit_bigint_int_ne",
        "objspace::descroperation::jit_bigint_int_lt",
        "objspace::descroperation::jit_bigint_int_le",
        "objspace::descroperation::jit_bigint_int_gt",
        "objspace::descroperation::jit_bigint_int_ge",
        "objspace::descroperation::jit_bigint_divrem_returns_lhs_remainder",
    ] {
        let values = hints
            .get(path)
            .unwrap_or_else(|| panic!("missing pointer-ABI wrapper hints for {path}"));
        assert!(values.iter().any(|hint| hint == "elidable"));
        assert!(
            values.iter().any(|hint| hint == "elidable_cannot_raise"),
            "{path} comparison must not retain an exception edge: {values:?}"
        );
        assert!(
            !values.iter().any(|hint| hint == "elidable_or_memerror"),
            "{path} comparison must not advertise an allocation edge: {values:?}"
        );
    }
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        &llbcs,
        HostStaticAddrs::default(),
        &[
            "objspace::descroperation",
            "module::math::interp_math",
            "baseobjspace",
            "builtins",
            "typedef",
            "type_methods",
            "objspace::std::formatting",
        ],
        &asserted_graph_names(),
    )
    .expect("lower descroperation module");
    let helper = program
        .functions
        .iter()
        .find(|function| {
            function.name == "bigint_and"
                && function.module_path.ends_with("objspace::descroperation")
        })
        .expect("descroperation::bigint_and graph");

    let calls: Vec<Vec<String>> = helper
        .graph()
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|operation| {
            if let OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } = &operation.kind
            {
                Some(segments.clone())
            } else {
                None
            }
        })
        .collect();
    let call_targets: Vec<String> = helper
        .graph()
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|operation| {
            if let OpKind::Call { target, .. } = &operation.kind {
                Some(format!("{target:?}"))
            } else {
                None
            }
        })
        .collect();
    let residuals = calls
        .iter()
        .filter(|segments| {
            segments.as_slice()
                == [
                    "pyre_interpreter",
                    "objspace",
                    "descroperation",
                    "jit_bigint_and",
                ]
        })
        .count();
    assert_eq!(
        residuals, 1,
        "the dependent-crate opaque declaration must retain RBigInt identity; \
         calls={calls:?}; targets={call_targets:?}"
    );

    for &(caller_name, residual_name) in MIXED_INT_RESIDUAL_CALLERS {
        let caller = program
            .functions
            .iter()
            .find(|function| {
                function.name == caller_name
                    && function.module_path.ends_with("objspace::descroperation")
            })
            .unwrap_or_else(|| panic!("descroperation::{caller_name} graph"));
        let calls: Vec<Vec<String>> = caller
            .graph()
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .filter_map(|operation| match &operation.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } => Some(segments.clone()),
                _ => None,
            })
            .collect();
        assert!(
            calls
                .iter()
                .any(|segments| segments.last().is_some_and(|leaf| leaf == residual_name)),
            "{caller_name} must preserve PyPy's mixed-long/int {residual_name} call: {calls:?}"
        );
        assert!(
            !calls.iter().any(|segments| {
                segments.last().is_some_and(|leaf| {
                    matches!(
                        leaf.as_str(),
                        "int_add" | "int_sub" | "int_mul" | "int_and_" | "int_or_" | "int_xor"
                    )
                })
            }),
            "{caller_name} retained an untranslated RBigInt::int_* call: {calls:?}"
        );
        assert!(
            !calls.iter().any(|segments| {
                segments
                    .last()
                    .is_some_and(|leaf| leaf == "jit_bigint_clone")
            }),
            "{caller_name} must borrow W_LongObject.num like PyPy's asbigint(), \
             not allocate cloned rbigint payloads: {calls:?}"
        );
    }

    let mixed_compare = program
        .functions
        .iter()
        .find(|function| {
            function.name == "long_int_compare"
                && function.module_path.ends_with("objspace::descroperation")
        })
        .expect("descroperation::long_int_compare graph");
    let mixed_compare_calls: Vec<Vec<String>> = mixed_compare
        .graph()
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|operation| match &operation.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => Some(segments.clone()),
            _ => None,
        })
        .collect();
    for residual_name in [
        "jit_bigint_int_eq",
        "jit_bigint_int_ne",
        "jit_bigint_int_lt",
        "jit_bigint_int_le",
        "jit_bigint_int_gt",
        "jit_bigint_int_ge",
    ] {
        assert!(
            mixed_compare_calls
                .iter()
                .any(|segments| segments.last().is_some_and(|leaf| leaf == residual_name)),
            "long_int_compare must target {residual_name}: {mixed_compare_calls:?}"
        );
    }
    assert!(
        !mixed_compare_calls.iter().any(|segments| {
            segments.last().is_some_and(|leaf| {
                matches!(
                    leaf.as_str(),
                    "int_eq" | "int_ne" | "int_lt" | "int_le" | "int_gt" | "int_ge"
                )
            })
        }),
        "long_int_compare retained an untranslated RBigInt::int_* call: \
         {mixed_compare_calls:?}"
    );

    let long_pow = program
        .functions
        .iter()
        .find(|function| {
            function.name == "long_pow"
                && function.module_path.ends_with("objspace::descroperation")
        })
        .expect("descroperation::long_pow graph");
    let pow_calls: Vec<Vec<String>> = long_pow
        .graph()
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|operation| match &operation.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => Some(segments.clone()),
            _ => None,
        })
        .collect();
    assert!(
        pow_calls.iter().any(|segments| segments
            == &[
                "pyre_interpreter",
                "objspace",
                "descroperation",
                "jit_bigint_pow_nomod",
            ]),
        "long_pow must erase Rust Result to the elidable pointer ABI: {pow_calls:?}"
    );
    assert!(
        !pow_calls.iter().any(|segments| segments
            .last()
            .is_some_and(|leaf| leaf == "bigint_pow_nomod")),
        "the host Result wrapper must not survive in the translated graph: {pow_calls:?}"
    );
    assert!(
        pow_calls.iter().any(|segments| segments
            .last()
            .is_some_and(|leaf| leaf == "jit_bigint_int_pow_nomod")),
        "long_pow must retarget bigint_int_pow_nomod: {pow_calls:?}"
    );
    assert!(
        !pow_calls.iter().any(|segments| segments
            .last()
            .is_some_and(|leaf| leaf == "bigint_int_pow_nomod")),
        "long_pow retained bigint_int_pow_nomod: {pow_calls:?}"
    );
    assert!(
        pow_calls.iter().any(|segments| segments
            .last()
            .is_some_and(|leaf| leaf == "jit_bigint_int_eq")),
        "long_pow int_eq(1) must retarget to jit_bigint_int_eq: {pow_calls:?}"
    );
    assert!(
        !pow_calls
            .iter()
            .any(|segments| segments.last().is_some_and(|leaf| leaf == "int_eq")),
        "long_pow retained RBigInt::int_eq: {pow_calls:?}"
    );

    let long_add = program
        .functions
        .iter()
        .find(|function| {
            function.name == "long_add"
                && function.module_path.ends_with("objspace::descroperation")
        })
        .expect("descroperation::long_add graph");
    let add_calls: Vec<Vec<String>> = long_add
        .graph()
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|operation| match &operation.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => Some(segments.clone()),
            _ => None,
        })
        .collect();
    assert!(
        add_calls.iter().any(|segments| segments
            == &[
                "pyre_interpreter",
                "objspace",
                "descroperation",
                "jit_bigint_add",
            ]),
        "long_add must retarget bigint_add to the elidable pointer ABI: {add_calls:?}"
    );
    assert!(
        !add_calls
            .iter()
            .any(|segments| segments.last().is_some_and(|leaf| leaf == "bigint_add")),
        "the host bigint_add wrapper must not survive in the translated graph: {add_calls:?}"
    );

    let module_loaded = llbcs.len() > 1;
    for &(module_suffix, caller_name) in BORROWED_PAYLOAD_CALLERS {
        if module_suffix.starts_with("module::") && !module_loaded {
            continue;
        }
        let caller = program
            .functions
            .iter()
            .find(|function| {
                function.name == caller_name && function.module_path.ends_with(module_suffix)
            })
            .unwrap_or_else(|| panic!("{module_suffix}::{caller_name} graph"));
        assert!(
            !caller
                .graph()
                .blocks
                .iter()
                .flat_map(|block| &block.operations)
                .any(|operation| matches!(
                    &operation.kind,
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } if segments.last().is_some_and(|leaf| leaf == "jit_bigint_clone")
                )),
            "{caller_name} must borrow long operand payloads rather than allocate \
             translated rbigint clones"
        );
    }

    let long_lshift = program
        .functions
        .iter()
        .find(|function| {
            function.name == "long_lshift"
                && function.module_path.ends_with("objspace::descroperation")
        })
        .expect("descroperation::long_lshift graph");
    let lshift_calls: Vec<Vec<String>> = long_lshift
        .graph()
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|operation| match &operation.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => Some(segments.clone()),
            _ => None,
        })
        .collect();
    assert!(
        lshift_calls.iter().any(|segments| segments
            == &[
                "pyre_interpreter",
                "objspace",
                "descroperation",
                "jit_bigint_lshift_count",
            ]),
        "long_lshift must erase Rust Result to the elidable pointer ABI: {lshift_calls:?}"
    );
    assert!(
        !lshift_calls.iter().any(|segments| segments
            .last()
            .is_some_and(|leaf| leaf == "bigint_lshift_count")),
        "the host Result wrapper must not survive in the translated graph: {lshift_calls:?}"
    );

    let int_lshift = program
        .functions
        .iter()
        .find(|function| {
            function.name == "int_lshift"
                && function.module_path.ends_with("objspace::descroperation")
        })
        .expect("descroperation::int_lshift graph");
    let int_lshift_calls: Vec<Vec<String>> = int_lshift
        .graph()
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|operation| match &operation.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } => Some(segments.clone()),
            _ => None,
        })
        .collect();
    assert!(
        int_lshift_calls.iter().any(|segments| segments
            == &[
                "pyre_interpreter",
                "objspace",
                "descroperation",
                "jit_bigint_lshift_int_int_result",
            ]),
        "int_lshift must retain rbigint's Signed×Signed specialized residual: \
         {int_lshift_calls:?}"
    );
    assert!(
        !int_lshift_calls.iter().any(|segments| segments
            .last()
            .is_some_and(|leaf| leaf == "bigint_lshift_int_int_result")),
        "the specialized host Result wrapper must not survive: {int_lshift_calls:?}"
    );

    for &(caller_name, residual_name) in FLOOR_DIVMOD_CALLERS {
        let caller = program
            .functions
            .iter()
            .find(|function| {
                function.name == caller_name
                    && function.module_path.ends_with("objspace::descroperation")
            })
            .unwrap_or_else(|| panic!("descroperation::{caller_name} graph"));
        let calls: Vec<Vec<String>> = caller
            .graph()
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .filter_map(|operation| match &operation.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } => Some(segments.clone()),
                _ => None,
            })
            .collect();
        assert!(
            calls
                .iter()
                .any(|segments| { segments.last().is_some_and(|leaf| leaf == residual_name) }),
            "{caller_name} must target {residual_name}: {calls:?}"
        );
    }

    for (caller_name, residual_name, forbidden) in [
        (
            "long_int_floordiv",
            "jit_bigint_int_div_floor",
            "bigint_int_floordiv_nonzero",
        ),
        (
            "long_int_mod",
            "jit_bigint_int_mod_int_result",
            "bigint_int_modulo_int_result_nonzero",
        ),
        ("long_rshift", "jit_bigint_shr", "bigint_rshift"),
        ("long_bitxor", "jit_bigint_xor", "xor"),
        ("long_int_divmod", "jit_bigint_int_divmod", "int_divmod"),
        ("integer_divmod_pair", "jit_bigint_int_divmod", "int_divmod"),
        ("integer_divmod_pair", "jit_bigint_divmod", "divmod"),
    ] {
        let caller = program
            .functions
            .iter()
            .find(|function| {
                function.name == caller_name
                    && function.module_path.ends_with("objspace::descroperation")
            })
            .unwrap_or_else(|| panic!("descroperation::{caller_name} graph"));
        let calls: Vec<Vec<String>> = caller
            .graph()
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .filter_map(|operation| match &operation.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } => Some(segments.clone()),
                _ => None,
            })
            .collect();
        assert!(
            calls
                .iter()
                .any(|segments| segments.last().is_some_and(|leaf| leaf == residual_name)),
            "{caller_name} must target {residual_name}: {calls:?}"
        );
        assert!(
            !calls
                .iter()
                .any(|segments| segments.last().is_some_and(|leaf| leaf == forbidden)),
            "{caller_name} retained {forbidden}: {calls:?}"
        );
        // The pair residual binds its destination to the bare
        // `(RBigInt, RBigInt)` payload, so no `Ok` field may be read off it,
        // including through the temporary `with_roots!` returns it in.
        if residual_name.ends_with("divmod") {
            let ok_reads: Vec<String> = caller
                .graph()
                .blocks
                .iter()
                .flat_map(|block| &block.operations)
                .filter_map(|operation| match &operation.kind {
                    OpKind::FieldRead { field, .. } => field.owner_root.clone(),
                    _ => None,
                })
                .filter(|owner| owner.starts_with("core::result::Result<(RBigInt,RBigInt)"))
                .collect();
            assert!(
                ok_reads.is_empty(),
                "{caller_name} reads a Result payload off the pair tuple: {ok_reads:?}"
            );
        }
    }

    for &(caller_name, residual_name) in UNARY_RESIDUAL_CALLERS {
        let caller = program
            .functions
            .iter()
            .find(|function| {
                function.name == caller_name
                    && function.module_path.ends_with("objspace::descroperation")
            })
            .unwrap_or_else(|| panic!("descroperation::{caller_name} graph"));
        let calls: Vec<Vec<String>> = caller
            .graph()
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .filter_map(|operation| match &operation.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } => Some(segments.clone()),
                _ => None,
            })
            .collect();
        assert!(
            calls
                .iter()
                .any(|segments| segments.last().is_some_and(|leaf| leaf == residual_name)),
            "{caller_name} must target {residual_name}: {calls:?}"
        );
        assert!(
            !calls.iter().any(|segments| segments
                .last()
                .is_some_and(|leaf| leaf == "jit_bigint_clone")),
            "{caller_name} must borrow its input and allocate only the unary result: {calls:?}"
        );
    }
}

#[test]
fn rbigint_add_residual_calls_the_rbigint_add_body_once() {
    let Some(llbcs) = rbigint_llbcs() else {
        return;
    };
    let program = build_semantic_program_from_llbcs_with_static_addrs_and_function_names(
        &llbcs,
        HostStaticAddrs::default(),
        &["longobject"],
        &["jit_bigint_add", "w_long_from_i64", "box_bigint_constant"],
    )
    .expect("lower longobject::jit_bigint_add and w_long_from_i64");
    let wrapper = program
        .functions
        .iter()
        .find(|function| function.name == "jit_bigint_add" && function.module_path == "longobject")
        .expect("longobject::jit_bigint_add graph");
    // `rbigint.add` is `@jit.elidable`: the front retargets the inherent
    // call on two exact `RBigInt` operands to its pointer-ABI residual
    // (`bigint_binop_residual_for_method`), so the wrapper enters the
    // residual once and never the Rust by-value `impl Add` shim.
    let add_residual = [
        "pyre_interpreter",
        "objspace",
        "descroperation",
        "jit_bigint_add",
    ];
    let mut add_residual_calls = 0;
    let mut calls = Vec::new();
    for block in &wrapper.graph().blocks {
        for operation in &block.operations {
            let OpKind::Call { target, .. } = &operation.kind else {
                continue;
            };
            calls.push(target.clone());
            match target {
                CallTarget::FunctionPath { segments, .. } => {
                    assert!(
                        !matches!(segments.as_slice(), [.., owner, leaf]
                            if owner == "<Impl>" && leaf == "add"),
                        "the Rust by-value RBigInt trait shim must not enter the JIT graph: \
                         {target:?}"
                    );
                    if segments == &add_residual {
                        add_residual_calls += 1;
                    }
                }
                CallTarget::Method { name, .. } => {
                    assert!(
                        name != "add",
                        "rbigint.add must reach the graph as its residual: {target:?}"
                    );
                }
                _ => {}
            }
        }
    }
    assert_eq!(
        add_residual_calls, 1,
        "the GC-reference residual must enter exactly one rbigint.add: {calls:?}"
    );

    let constructor_caller = program
        .functions
        .iter()
        .find(|function| function.name == "w_long_from_i64" && function.module_path == "longobject")
        .expect("longobject::w_long_from_i64 graph");
    let constructor_residuals = constructor_caller
        .graph()
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter(|operation| {
            matches!(
                &operation.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } if segments
                    == &[
                        "pyre_object",
                        "longobject",
                        "jit_bigint_from_i64",
                    ]
            )
        })
        .count();
    assert_eq!(
        constructor_residuals, 1,
        "RBigInt::from(i64) must return one GC reference through its residual"
    );

    // `long_pow`'s zero-exponent arm is `BigInt::from(1)`, an `Operand::Const`
    // of `I32`. The place-only constructor gate used to leave that call as
    // `rbigint::RBigInt::from`.
    if std::path::Path::new(INTERPRETER_LLBC).is_file() {
        let interpreter = load_llbc(INTERPRETER_LLBC);
        let long_pow = lower_named(interpreter, "long_pow").expect("lower long_pow");
        let long_pow_calls: Vec<Vec<String>> = long_pow
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .filter_map(|operation| match &operation.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } => Some(segments.clone()),
                _ => None,
            })
            .collect();
        assert!(
            long_pow_calls.iter().any(|segments| {
                segments
                    .last()
                    .is_some_and(|leaf| leaf == "jit_bigint_from_i64")
            }),
            "long_pow constant word constructors must retarget: {long_pow_calls:?}"
        );
        assert!(
            !long_pow_calls.iter().any(|segments| {
                segments.len() >= 2
                    && segments[segments.len() - 2] == "RBigInt"
                    && segments.last().is_some_and(|leaf| leaf == "from")
            }),
            "long_pow must not residualize rbigint::RBigInt::from: {long_pow_calls:?}"
        );
    }

    let clone_caller = program
        .functions
        .iter()
        .find(|function| {
            function.name == "box_bigint_constant" && function.module_path == "longobject"
        })
        .expect("longobject::box_bigint_constant graph");
    assert!(
        clone_caller
            .graph()
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .any(|operation| matches!(
                &operation.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    result_ty: ValueType::Ref(None),
                    ..
                } if segments.last().is_some_and(|leaf| leaf == "jit_bigint_clone")
            )),
        "RBigInt::clone must allocate a translated shallow-copy payload, not collapse to an alias"
    );
}
