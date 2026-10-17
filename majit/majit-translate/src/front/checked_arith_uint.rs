//! `usize::checked_{add,sub,mul}()` → native unsigned-overflow ops + a
//! virtualized `Option`.
//!
//! ## Positioning
//!
//! `core::num::<Impl>::checked_add` / `checked_sub` / `checked_mul` on
//! **unsigned** operands
//! are Opaque core bodies (Charon cannot extract `core`), so the caller
//! carries a residual `checked_*` call the rtyper census cannot type.
//! [`crate::front::checked_arith`] lowers the **signed** match-shape into a
//! `*_ovf` op + `OverflowError` edge, but that path is wrong for unsigned
//! arithmetic: RPython has no unsigned overflow op (`r_uint` wraps), so a
//! signed `mul_ovf` would diverge for operands in `(i64::MAX, usize::MAX]`.
//!
//! The unsigned overflow tests have native, wrapping-safe forms:
//!   - `checked_mul(x, y)` overflows iff the high word of the widening
//!     product is non-zero: `uint_mul_high(x, y) != 0`.
//!   - `checked_add(x, y)` overflows iff the sum wrapped below an addend
//!     (carry): `uint_lt(x + y, x)`.
//!   - `checked_sub(x, y)` underflows iff the subtrahend is above the
//!     minuend (borrow): `uint_lt(x, y)`.
//!
//! ## The rewrite (`rewire_one_checked_arith_uint_site`)
//!
//! Block A's last op is the residual `opt = checked_*(x, y)` call.  The
//! rewrite drops that call and emits the native sequence followed by a
//! Some/None construction diamond forwarding to the original continuation:
//!   - `checked_mul`: `lo = mul(x, y)`, `hi = uint_mul_high(x, y)`,
//!     `disc = eq(hi, 0)`, value = `lo`.
//!   - `checked_add`: `sum = add(x, y)`, `ovf = uint_lt(sum, x)`,
//!     `disc = eq(ovf, 0)`, value = `sum`.
//!   - `checked_sub`: `diff = sub(x, y)`, `ovf = uint_lt(x, y)`,
//!     `disc = eq(ovf, 0)`, value = `diff`.
//!   - branch on `disc`; Some carries `value`, None carries no payload.
//!
//! Unlike [`crate::front::checked_arith`] / [`crate::front::option_try`] this
//! does not match a particular consumer diamond: it produces valid variant
//! instances that downstream `?` / match / let-else consume unchanged.
//! Ordinary annotation joins them to the enum base; no payload is stored on
//! that base (rpython/annotator/classdesc.py::ClassDef._generalize_attr).
//!
//! It is **fail-safe**: any structural mismatch returns `Err`, the caller
//! leaves the residual call untouched, and the census Skip / legacy-walker
//! fallback is unchanged.  Gating (an unsigned `checked_{add,sub,mul}` producer
//! still present as its `Call`) is applied both at the recording site (only
//! unsigned operands are recorded) and here (the producer must still be the
//! `Call`, so a site [`crate::front::checked_arith`] already rewrote is
//! skipped).

use crate::flowspace::model::Variable;
use crate::front::bool_then::emit_option_variant_dynamic;
use crate::model::{FunctionGraph, OpKind, SpaceOperation, ValueType};

/// A recorded unsigned `checked_{add,sub,mul}` call whose result is an
/// `Option<T>`, captured during body lowering with the `Option`/`Some` owners
/// and payload type resolved from the destination type.
#[derive(Clone)]
pub(crate) struct CheckedArithUintSite {
    /// The `checked_*` call result (the `Option<T>` value) — locates block A
    /// and is replaced on its outgoing edge by each arm's concrete instance.
    pub opt: Variable,
    /// The `Option` enum root `name_path` — the `__discriminant` field owner
    /// and the ctor owner.
    pub option_owner: String,
    /// The `Option::Some` variant `name_path` — the `__pos_0` payload owner.
    pub some_owner: String,
    /// The `Option` payload `T` — the `Some::__pos_0` field kind.
    pub payload_ty: ValueType,
}

/// Rewrite every recorded unsigned `checked_{add,sub,mul}` site into the native
/// overflow-test + virtualized `Option` shape.  Returns the number of sites
/// rewritten; declined sites keep their residual call (census Skip).
pub(crate) fn rewire_checked_arith_uint_sites(
    graph: &mut FunctionGraph,
    sites: &[CheckedArithUintSite],
) -> usize {
    let mut rewritten = 0;
    for site in sites {
        match rewire_one_checked_arith_uint_site(graph, site) {
            Ok(()) => rewritten += 1,
            Err(_decline) => {
                if std::env::var_os("MAJIT_MIR_FRONTEND_DEBUG").is_some() {
                    eprintln!(
                        "[checked_arith_uint] {} decline at {:?}: {_decline}",
                        graph.name, site.opt
                    );
                }
                // Leave the residual `checked_*` call; the unregistered callee
                // makes the rtyper census Skip this graph (no regression).
            }
        }
    }
    rewritten
}

/// The unsigned `checked_*` operators this pass lowers.
#[derive(Clone, Copy)]
enum UintArith {
    Add,
    Sub,
    Mul,
}

impl UintArith {
    fn from_leaf(leaf: &str) -> Option<Self> {
        match leaf {
            "checked_add" => Some(UintArith::Add),
            "checked_sub" => Some(UintArith::Sub),
            "checked_mul" => Some(UintArith::Mul),
            _ => None,
        }
    }
}

/// The unsigned checked-arithmetic pass uses one JIT-bank operation and
/// its carry/borrow test.  Charon's flattened [`ValueType::Unsigned`]
/// erases the source width, so retain the literal atom at the capture
/// gate: a narrow `u8`/`u16`/`u32` overflow is not necessarily a bank
/// overflow.
///
/// The bank is 8 bytes. `int_add` / `uint_lt` / `uint_mul_high` lower to
/// the wasm `I64Add` / `I64LtU` / 64×64 high half, with no `intmask`
/// back to a 4-byte target word (`rarithmetic.intmask` / `r_uint` at
/// `LONG_BIT`). A 4-byte `usize` is the same width as `u32` and must
/// keep the residual call. `u64` matches the bank only when the target
/// word is 8 bytes; on a 4-byte target it is `UnsignedLongLong`
/// (`unsignedlonglong_repr`, `ullong_*`), not this `uint_*` op.
pub(crate) fn is_word_sized_uint_atom_for(atom: &str, word_bytes: usize) -> bool {
    match atom {
        "Usize" | "U64" => word_bytes == 8,
        _ => false,
    }
}

/// `I64` / `U64` are the 8-byte JIT int bank. `Isize` / `Usize` are that
/// bank only when the target word is 8 bytes; a 4-byte word wraps like
/// `i32` / `u32` and must not be lowered to an unmasked `int_*` op.
pub(crate) fn is_jit_bank_int_atom(atom: &str, word_bytes: usize) -> bool {
    match atom {
        "I64" | "U64" => true,
        "Isize" | "Usize" => word_bytes == 8,
        _ => false,
    }
}

/// `Signed` max for a `word_bytes`-wide machine word (`sys.maxint`).
pub(crate) fn signed_word_max(word_bytes: usize) -> i64 {
    match word_bytes {
        8 => i64::MAX,
        4 => i32::MAX as i64,
        other => {
            let bits = other.saturating_mul(8);
            if bits == 0 || bits >= 64 {
                i64::MAX
            } else {
                (1i64 << (bits - 1)) - 1
            }
        }
    }
}

/// `Unsigned` max for a `word_bytes`-wide machine word.
pub(crate) fn unsigned_word_max(word_bytes: usize) -> u64 {
    match word_bytes {
        8 => u64::MAX,
        4 => u32::MAX as u64,
        other => {
            let bits = other.saturating_mul(8);
            if bits == 0 || bits >= 64 {
                u64::MAX
            } else {
                (1u64 << bits) - 1
            }
        }
    }
}

pub(crate) fn is_word_sized_uint_atom(atom: &str) -> bool {
    is_word_sized_uint_atom_for(atom, crate::layout::target_word_size())
}

/// Unsigned atoms that fit in [`crate::model::ValueType::Unsigned`].
/// Narrow `u8`/`u16`/`u32` and word-sized `u64`/`usize` are accepted;
/// `U128` declines because the saturating_sub / div_ceil rewrites type
/// results as that word-sized bank.
pub(crate) fn is_non_u128_uint_atom(atom: &str) -> bool {
    atom != "U128"
}

/// Destination `Option<Self>` payload first (covers both-const
/// `1usize.checked_add(2)`), then either operand Place.  A readable
/// narrow payload declines even if an operand atom looks word-sized.
pub(crate) fn unsigned_word_atom<'a>(
    dest_payload_atom: Option<&'a str>,
    operand_atoms: impl IntoIterator<Item = Option<&'a str>>,
) -> Option<&'a str> {
    unsigned_word_atom_for(
        dest_payload_atom,
        operand_atoms,
        crate::layout::target_word_size(),
    )
}

pub(crate) fn unsigned_word_atom_for<'a>(
    dest_payload_atom: Option<&'a str>,
    operand_atoms: impl IntoIterator<Item = Option<&'a str>>,
    word_bytes: usize,
) -> Option<&'a str> {
    let atom = dest_payload_atom.or_else(|| operand_atoms.into_iter().flatten().next())?;
    is_word_sized_uint_atom_for(atom, word_bytes).then_some(atom)
}

fn rewire_one_checked_arith_uint_site(
    graph: &mut FunctionGraph,
    site: &CheckedArithUintSite,
) -> Result<(), String> {
    let name = graph.name.clone();
    let opt = &site.opt;

    // Block A: the `checked_*()` residual call producing `opt`, closed by
    // `lower_call` with a single forwarding exit.
    let a = graph
        .blocks
        .iter()
        .position(|b| {
            b.operations
                .iter()
                .any(|op| op.result.as_ref() == Some(opt))
        })
        .ok_or_else(|| format!("{name}: checked_* result var has no producer block"))?;

    // The dest Option is recast onto its instantiation (`Option<usize>`).
    // Drop that identity recast so the callee is again the last op.
    crate::front::result_exc::collapse_trailing_recasts_onto(graph, a, opt);

    // The call must still be A's last op (a
    // `[..]::checked_add/checked_sub/checked_mul`
    // 2-arg FunctionPath call); a site `checked_arith`'s first pass already
    // rewrote is now a `*_ovf` BinOp producer — skip it.  Recognition is
    // by the callee (`FunDecl` / FunctionPath leaf), not dest-type spelling.
    // The overflow op is resolved here (before any mutation) so an
    // unsupported leaf declines without touching the graph.
    let call_idx = graph.blocks[a]
        .operations
        .iter()
        .position(|op| {
            op.result.as_ref() == Some(opt)
                && matches!(
                    &op.kind,
                    OpKind::Call { target, args, .. }
                        if args.len() == 2
                            && crate::front::checked_arith::is_checked_arith_target(target)
                )
        })
        .ok_or_else(|| format!("{name}: no 2-arg checked_* callee producing {opt:?}"))?;
    if call_idx + 1 != graph.blocks[a].operations.len() {
        return Err(format!(
            "{name}: block {a} last op is not the 2-arg checked_* call producing {opt:?}"
        ));
    }
    let (lhs, rhs, arith) = match &graph.blocks[a].operations[call_idx] {
        SpaceOperation {
            result: Some(r),
            kind: OpKind::Call { target, args, .. },
        } if r == opt
            && args.len() == 2
            && crate::front::checked_arith::is_checked_arith_target(target) =>
        {
            let leaf =
                crate::front::checked_arith::checked_arith_callee_leaf(target).unwrap_or_default();
            let arith = UintArith::from_leaf(leaf).ok_or_else(|| {
                format!("{name}: unsigned checked lowering does not handle {leaf}")
            })?;
            (
                args[0].clone().into_variable(),
                args[1].clone().into_variable(),
                arith,
            )
        }
        _ => {
            return Err(format!(
                "{name}: block {a} last op is not the 2-arg checked_* call producing {opt:?}"
            ));
        }
    };

    crate::front::bool_then::validate_dynamic_option_exit(graph, graph.blocks[a].id)?;

    // All structural validation passed; mutate the graph.

    let a_id = graph.blocks[a].id;
    // Drop the residual call (A's last op) so `opt` is produced solely by the
    // virtualized ctor appended below, then re-emit the native sequence in
    // place of the removed call. The Some payload is written on the
    // instantiated `Option<T>` ClassDef recorded at the call (`site.some_owner`).
    graph.blocks[a].operations.truncate(call_idx);

    // The two operators differ only in the payload value and the overflow
    // test; build the `Option` once from the resulting `(value, disc)`.
    let (value, disc) = match arith {
        UintArith::Mul => {
            let lo = push_binop(
                graph,
                a_id,
                "mul",
                lhs.clone(),
                rhs.clone(),
                ValueType::Unsigned,
            );
            let hi = push_binop(graph, a_id, "uint_mul_high", lhs, rhs, ValueType::Unsigned);
            let disc = push_no_overflow_disc(graph, a_id, hi);
            (lo, disc)
        }
        UintArith::Add => {
            let sum = push_binop(
                graph,
                a_id,
                "add",
                lhs.clone(),
                rhs.clone(),
                ValueType::Unsigned,
            );
            // Unsigned carry: the wrapped sum is strictly below an addend.
            let ovf = push_binop(
                graph,
                a_id,
                "uint_lt",
                sum.clone(),
                lhs.clone(),
                ValueType::Int,
            );
            let disc = push_no_overflow_disc(graph, a_id, ovf);
            (sum, disc)
        }
        UintArith::Sub => {
            let diff = push_binop(
                graph,
                a_id,
                "sub",
                lhs.clone(),
                rhs.clone(),
                ValueType::Unsigned,
            );
            // Unsigned borrow: the subtrahend is strictly above the minuend.
            // The difference is emitted either way and read only when the
            // discriminant says `Some`, as the wrapped sum above is.
            let ovf = push_binop(graph, a_id, "uint_lt", lhs, rhs, ValueType::Int);
            let disc = push_no_overflow_disc(graph, a_id, ovf);
            (diff, disc)
        }
    };

    emit_option_variant_dynamic(
        graph,
        a_id,
        opt.clone(),
        &site.option_owner,
        disc,
        Some((&site.some_owner, value, site.payload_ty.clone())),
    );
    Ok(())
}

/// `disc = eq(test, 0)` — the `Option` tag: `1` (`Some`) when the overflow
/// `test` is `0` (no overflow), `0` (`None`) when it is non-zero.
fn push_no_overflow_disc(
    graph: &mut FunctionGraph,
    block: crate::model::BlockId,
    test: Variable,
) -> Variable {
    let zero = push_const_int(graph, block, 0);
    push_binop(graph, block, "eq", test, zero, ValueType::Int)
}

pub(crate) fn push_binop(
    graph: &mut FunctionGraph,
    block: crate::model::BlockId,
    op: &str,
    lhs: Variable,
    rhs: Variable,
    result_ty: ValueType,
) -> Variable {
    let res = graph.alloc_value_var();
    graph.block_mut(block).operations.push(SpaceOperation {
        result: Some(res.clone()),
        kind: OpKind::BinOp {
            op: op.to_string(),
            lhs,
            rhs,
            result_ty,
        },
    });
    res
}

pub(crate) fn push_const_int(
    graph: &mut FunctionGraph,
    block: crate::model::BlockId,
    value: i64,
) -> Variable {
    let res = graph.alloc_value_var();
    graph.block_mut(block).operations.push(SpaceOperation {
        result: Some(res.clone()),
        kind: OpKind::ConstInt(value),
    });
    res
}

#[cfg(test)]
mod tests {
    #[test]
    fn checked_unsigned_capture_rejects_narrow_integer_atoms() {
        for atom in ["U8", "U16", "U32", "U128"] {
            assert!(!is_word_sized_uint_atom(atom));
        }
        assert!(is_word_sized_uint_atom("Usize"));
        assert_eq!(
            is_word_sized_uint_atom("U64"),
            crate::layout::target_word_size() == 8
        );
    }

    #[test]
    fn word_sized_uint_follows_unsigned_lowleveltype() {
        assert!(is_word_sized_uint_atom_for("Usize", 8));
        assert!(is_word_sized_uint_atom_for("U64", 8));
        assert!(!is_word_sized_uint_atom_for("U32", 8));
        assert!(
            !is_word_sized_uint_atom_for("Usize", 4),
            "4-byte usize is u32-wide; the JIT int bank does not wrap there"
        );
        assert!(
            !is_word_sized_uint_atom_for("U64", 4),
            "u64 on a 4-byte target is UnsignedLongLong, not the uint_* word op"
        );
        assert!(!is_word_sized_uint_atom_for("U32", 4));
    }

    #[test]
    fn four_byte_usize_wrapping_shl_is_not_a_jit_bank_op() {
        assert!(
            !is_jit_bank_int_atom("Usize", 4),
            "usize::wrapping_shl on a 4-byte word must stay a residual call"
        );
        assert!(!is_jit_bank_int_atom("Isize", 4));
        assert!(is_jit_bank_int_atom("Usize", 8));
        assert!(is_jit_bank_int_atom("Isize", 8));
        assert!(is_jit_bank_int_atom("U64", 4));
        assert!(is_jit_bank_int_atom("I64", 4));
        assert!(!is_jit_bank_int_atom("U32", 8));
        assert!(!is_jit_bank_int_atom("I32", 8));
    }

    #[test]
    fn four_byte_usize_checked_add_is_not_an_unchecked_bank_op() {
        assert!(!is_word_sized_uint_atom_for("Usize", 4));
        assert_eq!(
            unsigned_word_atom_for(Some("Usize"), [None, None], 4),
            None,
            "usize::checked_add on a 4-byte word must stay a residual call"
        );
        assert_eq!(
            unsigned_word_atom_for(Some("Usize"), [None, None], 8),
            Some("Usize")
        );
    }

    use super::*;
    use crate::model::CallTarget;

    fn checked_target(leaf: &str) -> CallTarget {
        CallTarget::FunctionPath {
            segments: ["core", "num", "<Impl>", leaf]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            fun_decl_id: None,
        }
    }

    fn site_for(opt: &Variable) -> CheckedArithUintSite {
        CheckedArithUintSite {
            opt: opt.clone(),
            option_owner: "core::option::Option<usize>".to_string(),
            some_owner: "core::option::Option<usize>::Some".to_string(),
            payload_ty: ValueType::Unsigned,
        }
    }

    /// The two operands [`build_checked_site`] fed the `checked_*` call.
    ///
    /// Matched by their literal values rather than by position: the rewrite
    /// appends a `0` of its own for the discriminant test, so a positional
    /// read would pick that up once the pass has run.
    fn const_operand_vars(graph: &FunctionGraph, a: usize) -> (Variable, Variable) {
        let mut operands = graph.blocks[a]
            .operations
            .iter()
            .filter(|op| matches!(op.kind, OpKind::ConstInt(3) | OpKind::ConstInt(8)))
            .filter_map(|op| op.result.clone());
        let x = operands.next().expect("the first operand");
        let y = operands.next().expect("the second operand");
        (x, y)
    }

    /// The op names / owners the virtualized `Option` tail carries, in order.
    fn tail_binops(graph: &FunctionGraph, a: usize) -> Vec<String> {
        graph.blocks[a]
            .operations
            .iter()
            .filter_map(|op| match &op.kind {
                OpKind::BinOp { op: name, .. } => Some(name.clone()),
                _ => None,
            })
            .collect()
    }

    fn build_checked_site(leaf: &str) -> (FunctionGraph, Variable, usize) {
        let mut g = FunctionGraph::new("test_checked_uint");
        let a = g.startblock;
        let x = g.push_op_var(a, OpKind::ConstInt(3), true).unwrap();
        let y = g.push_op_var(a, OpKind::ConstInt(8), true).unwrap();
        // Block A's last op = the residual unsigned `checked_*(x, y)`.
        let opt = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: checked_target(leaf),
                    args: crate::model::call_args(vec![x, y]),
                    result_ty: ValueType::Ref(Some("core::option::Option<usize>".into())),
                },
                true,
            )
            .unwrap();
        // A continuation consuming `opt` (single forwarding goto exit).
        let (cont, _) = g.create_block_with_arg_vars(1);
        g.set_return(cont, None);
        g.set_goto(a, cont, vec![opt.clone()]);
        (g, opt, a.0)
    }

    #[test]
    fn checked_mul_lowers_to_uint_mul_high_and_virtualized_option() {
        let (mut g, opt, a) = build_checked_site("checked_mul");
        let rewritten = rewire_checked_arith_uint_sites(&mut g, &[site_for(&opt)]);
        assert_eq!(rewritten, 1, "the unsigned checked_mul site must rewrite");

        // No residual `checked_*` call survives in block A.
        assert!(
            !g.blocks[a].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { .. },
                    ..
                }
            )),
            "the residual checked_mul call must be gone"
        );
        // Overflow tests: `mul` (low word) + `uint_mul_high` (high word) + `eq`.
        assert_eq!(tail_binops(&g, a), vec!["mul", "uint_mul_high", "eq"]);

        assert_eq!(g.blocks[a].exits.len(), 2);
        let mut variants = Vec::new();
        for exit in &g.blocks[a].exits {
            let arm = g.block(exit.target);
            let variant = arm
                .operations
                .iter()
                .find_map(|op| match &op.kind {
                    OpKind::Call {
                        target: CallTarget::SyntheticTransparentCtor { name, .. },
                        ..
                    } => Some(name.as_str()),
                    _ => None,
                })
                .expect("arm constructor");
            variants.push(variant);
            let payloads: Vec<_> = arm
                .operations
                .iter()
                .filter_map(|op| match &op.kind {
                    OpKind::FieldWrite { field, .. } if field.name == "__pos_0" => Some(field),
                    _ => None,
                })
                .collect();
            assert_eq!(payloads.len(), usize::from(variant == "Some"));
            if let Some(field) = payloads.first() {
                assert_eq!(
                    field.owner_root.as_deref(),
                    Some("core::option::Option<usize>::Some")
                );
            }
        }
        variants.sort();
        assert_eq!(variants, vec!["None", "Some"]);
    }

    #[test]
    fn checked_add_lowers_to_uint_lt_carry_test() {
        let (mut g, opt, a) = build_checked_site("checked_add");
        let rewritten = rewire_checked_arith_uint_sites(&mut g, &[site_for(&opt)]);
        assert_eq!(rewritten, 1, "the unsigned checked_add site must rewrite");
        // Carry test: `add` (wrapping sum) + `uint_lt(sum, x)` + `eq`.
        assert_eq!(tail_binops(&g, a), vec!["add", "uint_lt", "eq"]);
    }

    #[test]
    fn checked_add_rewrites_through_instantiation_recast() {
        let mut g = FunctionGraph::new("test_checked_uint_recast");
        let a = g.startblock;
        let x = g.push_op_var(a, OpKind::ConstInt(3), true).unwrap();
        let y = g.push_op_var(a, OpKind::ConstInt(8), true).unwrap();
        let opt = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: checked_target("checked_add"),
                    args: crate::model::call_args(vec![x, y]),
                    result_ty: ValueType::Ref(Some("core::option::Option<usize>".into())),
                },
                true,
            )
            .unwrap();
        let narrowed = g
            .push_op_var(
                a,
                crate::model::cast_instance_call("Option<usize>", opt.clone()),
                true,
            )
            .unwrap();
        let (cont, _) = g.create_block_with_arg_vars(1);
        g.set_return(cont, None);
        g.set_goto(a, cont, vec![narrowed]);
        let rewritten = rewire_checked_arith_uint_sites(&mut g, &[site_for(&opt)]);
        assert_eq!(
            rewritten, 1,
            "the unsigned checked_add callee must rewrite through the dest recast"
        );
        assert_eq!(tail_binops(&g, a.0), vec!["add", "uint_lt", "eq"]);
        let some_owners: Vec<_> = g
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .filter_map(|op| match &op.kind {
                OpKind::FieldWrite { field, .. } if field.name == "__pos_0" => {
                    field.owner_root.clone()
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            some_owners.iter().map(String::as_str).collect::<Vec<_>>(),
            vec!["core::option::Option<usize>::Some"]
        );
    }

    #[test]
    fn checked_sub_lowers_to_uint_lt_borrow_test() {
        let (mut g, opt, a) = build_checked_site("checked_sub");
        let rewritten = rewire_checked_arith_uint_sites(&mut g, &[site_for(&opt)]);
        assert_eq!(rewritten, 1, "the unsigned checked_sub site must rewrite");
        // Borrow test: `sub` (wrapping difference) + `uint_lt(x, y)` + `eq`.
        assert_eq!(tail_binops(&g, a), vec!["sub", "uint_lt", "eq"]);

        // Which operand order makes the test a borrow is the whole of this
        // arm's correctness: `x < y` underflows, while the carry arm's
        // `sum < x` is a different question about different values. Reversed,
        // the tag would answer `Some` on exactly the inputs that underflow.
        let (x, y) = const_operand_vars(&g, a);
        let borrow = g.blocks[a]
            .operations
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::BinOp {
                    op: name, lhs, rhs, ..
                } if name == "uint_lt" => Some((lhs.clone(), rhs.clone())),
                _ => None,
            })
            .expect("the borrow test must be present");
        assert_eq!(borrow, (x, y));
    }

    #[test]
    fn declines_when_producer_is_no_longer_the_checked_call() {
        // A site `checked_arith` already rewrote (producer is now a `mul_ovf`
        // BinOp, not the `Call`) must be left untouched.
        let mut g = FunctionGraph::new("test_already_rewritten");
        let a = g.startblock;
        let x = g.push_op_var(a, OpKind::ConstInt(3), true).unwrap();
        let y = g.push_op_var(a, OpKind::ConstInt(8), true).unwrap();
        let opt = g
            .push_op_var(
                a,
                OpKind::BinOp {
                    op: "mul_ovf".to_string(),
                    lhs: x,
                    rhs: y,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let (cont, _) = g.create_block_with_arg_vars(1);
        g.set_return(cont, None);
        g.set_goto(a, cont, vec![opt.clone()]);

        let rewritten = rewire_checked_arith_uint_sites(&mut g, &[site_for(&opt)]);
        assert_eq!(rewritten, 0, "a non-Call producer must decline");
        assert!(
            g.blocks[a.0].operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::BinOp { op, .. } if op == "mul_ovf"
            )),
            "the mul_ovf producer must survive untouched"
        );
    }

    #[test]
    fn both_const_and_const_rhs_reach_word_sized_unsigned() {
        assert_eq!(
            unsigned_word_atom(Some("Usize"), [None, None]),
            Some("Usize")
        );
        assert_eq!(
            unsigned_word_atom(None, [Some("Usize"), None]),
            Some("Usize")
        );
        assert_eq!(
            unsigned_word_atom(Some("U64"), [None, None]).is_some(),
            crate::layout::target_word_size() == 8
        );
    }

    #[test]
    fn narrow_unsigned_dest_payload_declines() {
        assert_eq!(unsigned_word_atom(Some("U32"), [Some("U64"), None]), None);
        assert_eq!(unsigned_word_atom(Some("U8"), [None, None]), None);
        assert_eq!(unsigned_word_atom(None, [Some("U16"), None]), None);
        assert_eq!(unsigned_word_atom(None, [None, None]), None);
    }
}
