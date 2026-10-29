//! Result/`map_err` / try-successor rewiring of core callees.

use super::*;

/// `core::result::<Impl>::map_err` / `std::result::Result::map_err`.
///
/// The leaf name `map_err` is not enough: an extension trait can publish
/// the same method on `Result` and Charon still emits `CallTarget::Method`.
/// Gate on the resolved callee path so a foreign `map_err` stays residual.
pub(in crate::front::mir) fn is_core_result_map_err_path(path: &str) -> bool {
    matches!(
        path.split("::").collect::<Vec<_>>().as_slice(),
        ["core" | "std", "result", "<Impl>" | "Result", "map_err"]
            | ["result", "<Impl>" | "Result", "map_err"]
    )
}

pub(crate) fn is_core_clone_impls_clone_path(path: &str) -> bool {
    matches!(
        path.split("::").collect::<Vec<_>>().as_slice(),
        ["core", "clone", "impls", "<Impl>", "clone"]
    )
}

pub(crate) fn is_core_default_path(path: &str) -> bool {
    matches!(
        path.split("::").collect::<Vec<_>>().as_slice(),
        ["core", "default", "<Impl>", "default"] | ["core", "ptr", "mut_ptr", "<Impl>", "default"]
    )
}

/// `<(A, B, …) as Default>::default` — Charon spells the tuple impl as
/// `core::tuple::<Impl>::default`. Distinct from [`is_core_default_path`]
/// (scalar / thin-pointer `core::default::<Impl>::default`).
pub(crate) fn is_core_tuple_default_path(path: &str) -> bool {
    matches!(
        path.split("::").collect::<Vec<_>>().as_slice(),
        ["core", "tuple", "<Impl>", "default"]
    )
}

pub(in crate::front::mir) fn is_core_ptr_add_path(path: &str) -> bool {
    matches!(
        path.split("::").collect::<Vec<_>>().as_slice(),
        ["core", "ptr", "mut_ptr" | "const_ptr", "<Impl>", "add"]
    )
}

pub(in crate::front::mir) fn is_core_ptr_sub_path(path: &str) -> bool {
    matches!(
        path.split("::").collect::<Vec<_>>().as_slice(),
        ["core", "ptr", "mut_ptr" | "const_ptr", "<Impl>", "sub"]
    )
}

pub(in crate::front::mir) fn is_core_ptr_write_path(path: &str) -> bool {
    path == "core::ptr::write" || path == "core::ptr::mut_ptr::<Impl>::write"
}

pub(in crate::front::mir) fn is_alloc_vec_new_segments(segments: &[String]) -> bool {
    matches!(
        segments,
        [a, b, c, d]
            if a == "alloc" && b == "vec" && (c == "<Impl>" || c == "Vec") && d == "new"
    )
}

pub(in crate::front::mir) fn is_core_option_method(path: &str, leaf: &str) -> bool {
    matches!(
        path.split("::").collect::<Vec<_>>().as_slice(),
        ["core" | "std", "option", "<Impl>" | "Option", method] if *method == leaf
    )
}

pub(in crate::front::mir) fn is_core_result_method(path: &str, leaf: &str) -> bool {
    matches!(
        path.split("::").collect::<Vec<_>>().as_slice(),
        ["core" | "std", "result", "<Impl>" | "Result", method] if *method == leaf
    )
}

/// Option/Result methods whose body is a discriminant switch, optionally
/// calling a closure argument.  RPython never has these callees: the same
/// source is an if/else in the flow graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::front::mir) enum DiscCombinator {
    OptionFilter,
    /// `Option::ok_or(opt, err)`: `Some` rebuilds `Ok(payload)`, `None`
    /// rebuilds `Err(err)`.  The receiver tag follows the Option's
    /// representation (`emit_option_recv_tag`).
    OptionOkOr,
    /// `Option::ok_or_else(opt, f)`: `Some` rebuilds `Ok(payload)`, `None`
    /// calls the niladic `f` and rebuilds `Err(f())`.  The receiver tag
    /// follows the Option's representation (`emit_option_recv_tag`).
    OptionOkOrElse,
    ResultMap,
    ResultAndThen,
    ResultUnwrapOrElse,
    ResultOrElse,
    ResultOk,
    ResultErr,
    ResultIsOk,
    ResultIsErr,
}

impl DiscCombinator {
    pub(in crate::front::mir) fn from_method(
        name: &str,
        nargs: usize,
        callee_path: Option<&str>,
        recv_ty: Option<&TyRef>,
        llbc: &Llbc,
    ) -> Option<Self> {
        let path = callee_path?;
        let _ = (recv_ty, llbc);
        match (name, nargs) {
            ("filter", 2) if is_core_option_method(path, "filter") => Some(Self::OptionFilter),
            ("ok_or", 2) if is_core_option_method(path, "ok_or") => Some(Self::OptionOkOr),
            ("ok_or_else", 2) if is_core_option_method(path, "ok_or_else") => {
                Some(Self::OptionOkOrElse)
            }
            ("map", 2) if is_core_result_method(path, "map") => Some(Self::ResultMap),
            ("and_then", 2) if is_core_result_method(path, "and_then") => Some(Self::ResultAndThen),
            ("unwrap_or_else", 2) if is_core_result_method(path, "unwrap_or_else") => {
                Some(Self::ResultUnwrapOrElse)
            }
            ("or_else", 2) if is_core_result_method(path, "or_else") => Some(Self::ResultOrElse),
            ("ok", 1) if is_core_result_method(path, "ok") => Some(Self::ResultOk),
            ("err", 1) if is_core_result_method(path, "err") => Some(Self::ResultErr),
            ("is_ok", 1) if is_core_result_method(path, "is_ok") => Some(Self::ResultIsOk),
            ("is_err", 1) if is_core_result_method(path, "is_err") => Some(Self::ResultIsErr),
            _ => None,
        }
    }

    pub(in crate::front::mir) fn needs_closure(self) -> bool {
        matches!(
            self,
            Self::OptionFilter
                | Self::OptionOkOrElse
                | Self::ResultMap
                | Self::ResultAndThen
                | Self::ResultUnwrapOrElse
                | Self::ResultOrElse
        )
    }

    pub(in crate::front::mir) fn method_name(self) -> &'static str {
        match self {
            Self::OptionFilter => "filter",
            Self::OptionOkOr => "ok_or",
            Self::OptionOkOrElse => "ok_or_else",
            Self::ResultMap => "map",
            Self::ResultAndThen => "and_then",
            Self::ResultUnwrapOrElse => "unwrap_or_else",
            Self::ResultOrElse => "or_else",
            Self::ResultOk => "ok",
            Self::ResultErr => "err",
            Self::ResultIsOk => "is_ok",
            Self::ResultIsErr => "is_err",
        }
    }
}

#[derive(Clone)]
pub(in crate::front::mir) struct DiscCombinatorSite {
    pub(in crate::front::mir) kind: DiscCombinator,
    pub(in crate::front::mir) result_var: Variable,
    /// The combinator call's arguments — locate the call once `result_exc`
    /// has renamed its result to the `Ok` payload.
    pub(in crate::front::mir) call_args: Vec<LinkArg>,
    pub(in crate::front::mir) recv_owner: String,
    pub(in crate::front::mir) recv_tag0_owner: String,
    pub(in crate::front::mir) recv_tag1_owner: String,
    pub(in crate::front::mir) payload0_ty: ValueType,
    pub(in crate::front::mir) payload1_ty: ValueType,
    pub(in crate::front::mir) payload0_class: Option<String>,
    pub(in crate::front::mir) payload1_class: Option<String>,
    pub(in crate::front::mir) payload0_is_raw_u8: bool,
    pub(in crate::front::mir) payload1_is_raw_u8: bool,
    pub(in crate::front::mir) result_owner: String,
    pub(in crate::front::mir) result_tag0_owner: String,
    pub(in crate::front::mir) result_tag1_owner: String,
    pub(in crate::front::mir) result_payload0_ty: ValueType,
    pub(in crate::front::mir) result_payload1_ty: ValueType,
    pub(in crate::front::mir) result_payload0_class: Option<String>,
    pub(in crate::front::mir) result_payload1_class: Option<String>,
    pub(in crate::front::mir) call_once_owner: String,
    /// The callable is this function item (no closure env): the arm calls it
    /// directly.
    pub(in crate::front::mir) fn_item_segments: Option<Vec<String>>,
    /// The function item is an ADT constructor: the arm builds this aggregate
    /// instead of calling it.
    pub(in crate::front::mir) fn_item_ctor: Option<AggregateShape>,
    pub(in crate::front::mir) args_tuple_suffix: String,
    pub(in crate::front::mir) call_result_ty: ValueType,
    pub(in crate::front::mir) call_result_class: Option<String>,
    /// The receiver is a `Result<_, carrier>`.  Its `map`/`and_then` is
    /// rewritten into the if/else on the receiver before `result_exc` turns
    /// the carrier into exception edges, so the receiver call is caught and
    /// rewrapped like any other `match` on it, and the built `Ok`/`Err`
    /// shells reach the callee rule.
    pub(in crate::front::mir) carrier_recv: bool,
    /// Receiver is a one-word niche `Option` (nullable pointer). Its tag is
    /// a null test, and the `Some` payload is the pointer itself.
    pub(in crate::front::mir) recv_niche: bool,
    /// Receiver is `Option<NonZero*>`: `None` is 0 and `Some` is the integer
    /// itself (`rint.py` `IntegerRepr.rtype_bool`).
    pub(in crate::front::mir) scalar_niche: bool,
    /// The combinator's result is that same nullable pointer.
    pub(in crate::front::mir) result_niche: bool,
    /// The combinator's result is an `Option<NonZero*>` word.
    pub(in crate::front::mir) result_scalar_niche: bool,
    /// `None` narrowed to the payload class, when the null must share it
    /// (`option_niche_null_cast`).
    pub(in crate::front::mir) niche_null_cast: Option<(String, ValueType)>,
    /// Int-banked niche (`Option<fn>` / raw Address): `Some` is that
    /// int-bank null (`null_fn` / `null_raw`; `rnone.py` `rtype_is_None`).
    /// `None` is a ref-bank niche (`push_niche_null`).
    pub(in crate::front::mir) int_null: Option<crate::model::NicheIntNull>,
}

#[derive(Clone)]
pub(in crate::front::mir) struct ResultTrySite {
    pub(in crate::front::mir) branch_result_var: Variable,
    pub(in crate::front::mir) result_owner: String,
    pub(in crate::front::mir) ok_owner: String,
    pub(in crate::front::mir) err_owner: String,
    pub(in crate::front::mir) ok_ty: ValueType,
    pub(in crate::front::mir) err_ty: ValueType,
    pub(in crate::front::mir) ok_is_raw_u8: bool,
    pub(in crate::front::mir) err_is_raw_u8: bool,
    pub(in crate::front::mir) err_class: Option<String>,
    pub(in crate::front::mir) recv_err_ast: String,
}

#[derive(Clone)]
pub(in crate::front::mir) struct ResultTryReturnOwners {
    pub(in crate::front::mir) result_owner: String,
    pub(in crate::front::mir) err_owner: String,
    pub(in crate::front::mir) err_ty: ValueType,
    pub(in crate::front::mir) err_ast: String,
}

#[derive(Default, Debug, Clone)]
pub(in crate::front::mir) struct ResultTryStats {
    pub(in crate::front::mir) rewritten: usize,
}

/// How an `Rvalue::Aggregate` builds its value
/// ([`Lowering::aggregate_shape`]).
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum AggregateShape {
    /// A fieldless enum variant (or the reserved `None` of an
    /// `Option<fieldless enum>`) is its discriminant integer.
    Discriminant(i64),
    /// `None` of `Option<NonZeroU*>`: the unsigned zero, not a signed `ConstInt`.
    UnsignedZero,
    /// A zero-sized value: no constructor and no register.
    Void,
    /// `None` of a niche pointer `Option`: a null pointer.
    NicheNull,
    /// The value is operand `index` itself: a niche `Some`, an
    /// `UnsafeCell`, or (`transparent`) a `repr(transparent)` wrapper.
    Operand { index: usize, transparent: bool },
    /// A variant/struct/tuple constructor plus one `FieldWrite` per operand.
    Ctor(AggregateCtor),
}

/// Unsuffixed `core::ops::range::{leaf}` aggregate. Accepts both the
/// historical `["core","ops","range"]` + `Range<T>` split and the class
/// identity split `["core"]` + `ops::range::Range<T>`.
pub(in crate::front::mir) fn is_core_ops_range_adt_ctor(
    owner_path: &[String],
    ctor_name: &str,
    leaf: &str,
) -> bool {
    let joined = if owner_path.is_empty() {
        ctor_name.to_string()
    } else {
        format!("{}::{ctor_name}", owner_path.join("::"))
    };
    let base = joined.split_once('<').map_or(joined.as_str(), |(b, _)| b);
    base == format!("core::ops::range::{leaf}")
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct AggregateCtor {
    pub(in crate::front::mir) target: CallTarget,
    pub(in crate::front::mir) result_owner: String,
    pub(in crate::front::mir) owner_id: Option<majit_ir::descr::StructId>,
    pub(in crate::front::mir) owner_path: Vec<String>,
    pub(in crate::front::mir) ctor_name: String,
    /// One row per operand.
    pub(in crate::front::mir) fields: Vec<AggregateField>,
}

#[derive(Clone, Debug, PartialEq)]
pub(in crate::front::mir) struct AggregateField {
    pub(in crate::front::mir) name: String,
    /// Zero-sized: no slot, no write.
    pub(in crate::front::mir) void: bool,
    /// Instance root the stored value is narrowed to.
    pub(in crate::front::mir) narrow_root: Option<String>,
    /// Shared `&[u8]` field: store `{data, len}` via `FatData`/`FatLen`,
    /// not `pair_len_field_name`.
    pub(in crate::front::mir) window: bool,
    /// Declared field type is `*mut u8` / `*const u8` (`rffi.py` `CCHARP`).
    pub(in crate::front::mir) raw_u8: bool,
}

/// Emit `ctor`'s constructor and field writes into `block` over the operand
/// values `args`, returning the constructed value.

pub(in crate::front::mir) fn emit_aggregate_ctor_with_pairs(
    graph: &mut FunctionGraph,
    block: BlockId,
    ctor: &AggregateCtor,
    args: &[Variable],
    pair_lens: &[Option<Variable>],
) -> Result<Variable, LowerError> {
    if pair_lens.iter().all(Option::is_none) {
        return emit_aggregate_ctor(graph, block, ctor, args);
    }
    let res = graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
    graph.block_mut(block).operations.push(SpaceOperation {
        result: Some(res.clone()),
        kind: OpKind::Call {
            target: ctor.target.clone(),
            args: Vec::new(),
            result_ty: ValueType::Ref(Some(ctor.result_owner.clone())),
        },
    });
    for (i, field) in ctor.fields.iter().enumerate() {
        if field.void {
            continue;
        }
        let Some(value) = args.get(i) else {
            continue;
        };
        if let Some(Some(len)) = pair_lens.get(i) {
            if field.window {
                for (part, stored, ty) in [
                    (
                        crate::model::VecFieldPart::FatData,
                        value.clone(),
                        ValueType::Unsigned,
                    ),
                    (
                        crate::model::VecFieldPart::FatLen,
                        len.clone(),
                        ValueType::Int,
                    ),
                ] {
                    graph.block_mut(block).operations.push(SpaceOperation {
                        result: None,
                        kind: OpKind::FieldWrite {
                            base: res.clone(),
                            field: crate::model::FieldDescriptor::new(
                                field.name.clone(),
                                Some(ctor.result_owner.clone()),
                            )
                            .with_owner_id(ctor.owner_id)
                            .with_base_is_deref(false)
                            .with_vec_part(part)
                            .with_owner_declared_gc(Some(false)),
                            value: crate::model::LinkArg::Value(stored),
                            ty,
                        },
                    });
                }
                continue;
            }
            let len_name = pair_len_field_name(&field.name);
            for (name, value) in [(field.name.clone(), value.clone()), (len_name, len.clone())] {
                graph.block_mut(block).operations.push(SpaceOperation {
                    result: None,
                    kind: OpKind::FieldWrite {
                        base: res.clone(),
                        field: crate::model::FieldDescriptor::new(
                            name,
                            Some(ctor.result_owner.clone()),
                        )
                        .with_owner_id(ctor.owner_id)
                        .with_base_is_deref(false),
                        value: crate::model::LinkArg::Value(value),
                        ty: ValueType::Unsigned,
                    },
                });
            }
            continue;
        }
        let (value, ty) = ctor_ccharp_field_store(graph, block, ctor, field, value)?;
        graph.block_mut(block).operations.push(SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: res.clone(),
                field: crate::model::FieldDescriptor {
                    name: field.name.clone(),
                    owner_root: Some(ctor.result_owner.clone()),
                    owner_id: ctor.owner_id,
                    // Built by value: the store is not a dereference of a
                    // live container (`base_is_local_aggregate`).
                    base_is_deref: Some(false),
                    taken_by_address: false,
                    inline_vec: false,
                    vec_part: None,
                    owner_declared_gc: None,
                    host_index: None,
                    scalar_word: None,
                },
                value: crate::model::LinkArg::Value(value),
                ty,
            },
        });
    }
    Ok(res)
}

pub(in crate::front::mir) fn emit_aggregate_ctor(
    graph: &mut FunctionGraph,
    block: BlockId,
    ctor: &AggregateCtor,
    args: &[Variable],
) -> Result<Variable, LowerError> {
    let res = graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
    graph.block_mut(block).operations.push(SpaceOperation {
        result: Some(res.clone()),
        kind: OpKind::Call {
            target: ctor.target.clone(),
            args: Vec::new(),
            result_ty: ValueType::Ref(Some(ctor.result_owner.clone())),
        },
    });
    for (field, value) in ctor.fields.iter().zip(args) {
        if field.void {
            continue;
        }
        let (value, ty) = ctor_ccharp_field_store(graph, block, ctor, field, value)?;
        graph.block_mut(block).operations.push(SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base: res.clone(),
                field: crate::model::FieldDescriptor {
                    name: field.name.clone(),
                    owner_root: Some(ctor.result_owner.clone()),
                    owner_id: ctor.owner_id,
                    // Built by value: the store is not a dereference of a
                    // live container (`base_is_local_aggregate`).
                    base_is_deref: Some(false),
                    taken_by_address: false,
                    inline_vec: false,
                    vec_part: None,
                    owner_declared_gc: None,
                    host_index: None,
                    scalar_word: None,
                },
                value: crate::model::LinkArg::Value(value),
                ty,
            },
        });
    }
    Ok(res)
}

/// Store a CCHARP operand as the int-banked field (`rffi.py` `CCHARP`).
/// `rptr.py` `PtrRepr.rtype_setattr` / `rtuple.py` `newtuple` take the
/// operand with `hop.inputarg` (converted to the field repr); a missing
/// conversion is a `TyperError`. CCHARP paints Unsigned only for a raw `*u8`
/// field. A `cast_instance` Ref retype would union `SomePtr` with
/// `SomeInstance`.
fn ctor_ccharp_field_store(
    graph: &mut FunctionGraph,
    block: BlockId,
    ctor: &AggregateCtor,
    field: &AggregateField,
    value: &Variable,
) -> Result<(Variable, ValueType), LowerError> {
    if ctor_field_is_raw_u8(ctor, field) {
        if value.concretetype().as_ref().is_some_and(lltype_is_ccharp) {
            return Ok((value.clone(), ValueType::Unsigned));
        }
        if ctor_operand_is_int_or_untyped(value) {
            stamp_as_ccharp(value);
            return Ok((value.clone(), ValueType::Unsigned));
        }
        return Err(LowerError::Unsupported(format!(
            "raw *u8 field {}.{} store of {:?} (PtrRepr.rtype_setattr / newtuple hop.inputarg)",
            ctor.result_owner,
            field.name,
            value.concretetype(),
        )));
    }
    let stored = match &field.narrow_root {
        Some(root) => {
            let narrowed = graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
            graph.block_mut(block).operations.push(SpaceOperation {
                result: Some(narrowed.clone()),
                kind: crate::model::cast_instance_call(root, value.clone()),
            });
            narrowed
        }
        None => value.clone(),
    };
    Ok((
        stored,
        positional_ctor_field_value_type(&ctor.result_owner, &field.name),
    ))
}

/// Untyped (`concretetype` missing) or int-kind (`history.py` `getkind` int).
fn ctor_operand_is_int_or_untyped(value: &Variable) -> bool {
    let Some(ct) = value.concretetype() else {
        return true;
    };
    crate::model::try_getkind(&ct, true, true, true)
        .is_ok_and(|kind| matches!(kind, crate::model::ConcreteType::Signed))
}

/// Tuple intern spelling `*mut u8` / `*const u8`, or a struct field whose
/// declared type is that raw byte pointer.
fn ctor_field_is_raw_u8(ctor: &AggregateCtor, field: &AggregateField) -> bool {
    if field.raw_u8 {
        return true;
    }
    positional_shape_metadata(&ctor.result_owner).is_some_and(|(rows, _)| {
        rows.iter()
            .any(|row| row.name == field.name && (row.ty == "*mut u8" || row.ty == "*const u8"))
    })
}

/// `TupleRepr` field bank of a positional ctor write (`rtuple.py`).
///
/// [`positional_shape_metadata`] intern's `__pos_N` from the shape
/// spelling; a raw `*mut usize` item is `int` (`history.py` `getkind`).
/// The write has to use that same bank so setattr and getattr agree.
pub(in crate::front::mir) fn positional_ctor_field_value_type(
    result_owner: &str,
    field_name: &str,
) -> ValueType {
    crate::virtualizable_decl::overlay_virtualizable_llfield_value_type(
        result_owner,
        field_name,
        positional_shape_metadata(result_owner)
            .and_then(|(_, attrs)| {
                attrs
                    .into_iter()
                    .find(|(name, _)| name == field_name)
                    .map(|(_, vt)| vt)
            })
            .unwrap_or(ValueType::Ref(None)),
    )
}

/// Build `shape` in `block` from operand values already in hand — a call of
/// a constructor function item.  `Err` for a shape that has no operands to
/// build from.
pub(crate) fn emit_aggregate_value(
    graph: &mut FunctionGraph,
    block: BlockId,
    shape: &AggregateShape,
    args: &[Variable],
) -> Result<Variable, String> {
    match shape {
        AggregateShape::Discriminant(tag) => Ok(graph
            .push_op_var(block, OpKind::ConstInt(*tag), true)
            .expect("ConstInt produces a value")),
        AggregateShape::UnsignedZero => Ok(graph
            .push_op_var(block, OpKind::ConstUInt(0), true)
            .expect("ConstUInt produces a value")),
        AggregateShape::Operand { index, .. } => args
            .get(*index)
            .cloned()
            .ok_or_else(|| format!("constructor call has no operand {index}")),
        AggregateShape::Ctor(ctor) if ctor.fields.len() == args.len() => {
            emit_aggregate_ctor(graph, block, ctor, args).map_err(|e| e.to_string())
        }
        other => Err(format!("constructor call cannot build {other:?}")),
    }
}

pub(in crate::front::mir) fn rewire_disc_combinator_sites(
    graph: &mut FunctionGraph,
    sites: &[DiscCombinatorSite],
    spec: crate::ErrorCarrierSpec<'_>,
) -> usize {
    let rewritten = sites
        .iter()
        .filter(|site| rewire_one_disc_combinator(graph, site, spec).is_ok())
        .count();
    crate::front::result_exc::split_trailing_ops_off_canraise_blocks(graph);
    rewritten
}

pub(in crate::front::mir) fn rewire_one_disc_combinator(
    graph: &mut FunctionGraph,
    site: &DiscCombinatorSite,
    spec: crate::ErrorCarrierSpec<'_>,
) -> Result<(), String> {
    match site.kind {
        DiscCombinator::ResultIsOk | DiscCombinator::ResultIsErr => {
            rewire_result_is_ok_err(graph, site)
        }
        _ => rewire_disc_combinator_diamond(graph, site, spec),
    }
}

pub(in crate::front::mir) fn locate_combinator_call(
    graph: &FunctionGraph,
    site: &DiscCombinatorSite,
    method: &str,
    name: &str,
) -> Result<(usize, usize, Variable, Vec<Variable>), String> {
    // `result_exc` renames a can-raise call's result to its `Ok` payload
    // (`remint_call_as_payload`); the call then keeps only its arguments.
    let is_call = |op: &SpaceOperation| {
        op.result.as_ref() == Some(&site.result_var)
            || (!site.call_args.is_empty()
                && matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::Method { name: m, .. },
                        args,
                        ..
                    } if m == method && *args == site.call_args
                ))
    };
    let a = graph
        .blocks
        .iter()
        .position(|block| block.operations.iter().any(is_call))
        .ok_or_else(|| format!("{name}: {method} result var has no producer block"))?;
    let ci = graph.blocks[a]
        .operations
        .iter()
        .position(is_call)
        .ok_or_else(|| format!("{name}: {method} call not found in block {a}"))?;
    let result_var = graph.blocks[a].operations[ci]
        .result
        .as_ref()
        .ok_or_else(|| format!("{name}: {method} call has no result"))?;
    let ops_len = graph.blocks[a].operations.len();
    let flow_result = if ci + 1 == ops_len {
        result_var.clone()
    } else if ci + 2 == ops_len {
        let cast = &graph.blocks[a].operations[ci + 1];
        match cast.result.as_ref() {
            Some(narrowed) if crate::model::cast_instance_of(&cast.kind, result_var).is_some() => {
                narrowed.clone()
            }
            _ => {
                return Err(format!(
                    "{name}: {method} call is not the last op of block {a}"
                ));
            }
        }
    } else {
        return Err(format!(
            "{name}: {method} call is not the last op of block {a}"
        ));
    };
    let args = match &graph.blocks[a].operations[ci].kind {
        OpKind::Call {
            target: CallTarget::Method { name: m, .. },
            args,
            ..
        } if m == method => args.iter().map(|arg| arg.clone().into_variable()).collect(),
        other => {
            return Err(format!(
                "{name}: {method} producer is not a method call: {other:?}"
            ));
        }
    };
    Ok((a, ci, flow_result, args))
}

pub(in crate::front::mir) fn rewire_result_is_ok_err(
    graph: &mut FunctionGraph,
    site: &DiscCombinatorSite,
) -> Result<(), String> {
    let name = graph.name.clone();
    let method = site.kind.method_name();
    let (a, ci, flow_result, args) = locate_combinator_call(graph, site, method, &name)?;
    if args.len() != 1 {
        return Err(format!("{name}: {method} is not a one-arg call"));
    }
    if flow_result != site.result_var {
        return Err(format!("{name}: {method} carried a trailing cast"));
    }
    let recv = args[0].clone();
    let a_id = graph.blocks[a].id;
    graph.blocks[a].operations.truncate(ci);
    let disc = emit_enum_disc_read(graph, a_id, recv, &site.recv_owner);
    let tag = graph
        .push_op_var(
            a_id,
            OpKind::ConstInt(if site.kind == DiscCombinator::ResultIsOk {
                0
            } else {
                1
            }),
            true,
        )
        .expect("ConstInt produces a value");
    graph.block_mut(a_id).operations.push(SpaceOperation {
        result: Some(site.result_var.clone()),
        kind: OpKind::BinOp {
            op: "eq".to_string(),
            lhs: disc,
            rhs: tag,
            result_ty: ValueType::Bool,
        },
    });
    Ok(())
}

pub(in crate::front::mir) fn rewire_disc_combinator_diamond(
    graph: &mut FunctionGraph,
    site: &DiscCombinatorSite,
    spec: crate::ErrorCarrierSpec<'_>,
) -> Result<(), String> {
    use crate::front::option_variant::{close_goto_mixed, reproduce_exit_args};

    let name = graph.name.clone();
    let method = site.kind.method_name();
    let (a, ci, flow_result, args) = locate_combinator_call(graph, site, method, &name)?;
    let recv = args
        .first()
        .cloned()
        .ok_or_else(|| format!("{name}: {method} missing receiver"))?;
    let extra = args.get(1).cloned();
    if site.kind == DiscCombinator::OptionFilter {
        return rewire_option_filter_diamond(graph, site, a, ci, recv, extra, flow_result, &name);
    }
    let exception_lowered = matches!(graph.blocks[a].exitswitch, Some(ExitSwitch::LastException));
    let [exit] = graph.blocks[a].exits.as_slice() else {
        if exception_lowered && graph.blocks[a].exits.len() == 2 {
            return rewire_result_map_last_exception(
                graph,
                site,
                spec,
                (a, ci),
                recv,
                extra,
                flow_result,
            );
        }
        return Err(format!(
            "{name}: {method} block does not have a single exit"
        ));
    };
    if graph.blocks[a].exitswitch.is_some()
        || exit.exitcase.is_some()
        || exit.last_exception.is_some()
        || exit.last_exc_value.is_some()
    {
        return Err(format!("{name}: {method} exit is not a plain goto"));
    }
    let saved_exit = exit.clone();
    let b_target = saved_exit.target;
    let mut carried: Vec<Variable> = Vec::new();
    for arg in &saved_exit.args {
        if let LinkArg::Value(v) = arg
            && *v != flow_result
            && !carried.contains(v)
        {
            carried.push(v.clone());
        }
    }

    let (then_is_tag1, then_needs_recv, then_needs_extra, else_needs_recv, else_needs_extra) =
        match site.kind {
            DiscCombinator::OptionFilter => unreachable!("filter has its own diamond"),
            DiscCombinator::OptionOkOr | DiscCombinator::OptionOkOrElse => {
                (true, true, false, false, true)
            }
            DiscCombinator::ResultMap | DiscCombinator::ResultAndThen => {
                (false, true, true, true, false)
            }
            DiscCombinator::ResultUnwrapOrElse | DiscCombinator::ResultOrElse => {
                (false, true, false, true, true)
            }
            DiscCombinator::ResultOk => (false, true, false, false, false),
            DiscCombinator::ResultErr => (true, true, false, false, false),
            DiscCombinator::ResultIsOk | DiscCombinator::ResultIsErr => unreachable!(),
        };
    // A function-item callable is named, not threaded: only a closure env or
    // a plain value argument flows into the arm.
    let extra = extra.filter(|_| site.fn_item_segments.is_none());
    let mut then_sources = carried.clone();
    if then_needs_recv && !then_sources.contains(&recv) {
        then_sources.push(recv.clone());
    }
    if then_needs_extra
        && let Some(extra) = extra.as_ref()
        && !then_sources.contains(extra)
    {
        then_sources.push(extra.clone());
    }
    let mut else_sources = carried.clone();
    if else_needs_recv && !else_sources.contains(&recv) {
        else_sources.push(recv.clone());
    }
    if else_needs_extra
        && let Some(extra) = extra.as_ref()
        && !else_sources.contains(extra)
    {
        else_sources.push(extra.clone());
    }

    let (then_bb, then_inputs) = graph.create_block_with_arg_vars(then_sources.len());
    let (else_bb, else_inputs) = graph.create_block_with_arg_vars(else_sources.len());

    let then_value = build_disc_arm(
        graph,
        site,
        then_bb,
        &recv,
        &then_sources,
        &then_inputs,
        extra.as_ref(),
        true,
        &name,
    )?;
    let then_args = reproduce_exit_args(
        &saved_exit,
        &flow_result,
        &then_value,
        &then_sources,
        &then_inputs,
        &name,
    )?;
    close_goto_mixed(graph, then_bb, b_target, then_args);

    let else_value = build_disc_arm(
        graph,
        site,
        else_bb,
        &recv,
        &else_sources,
        &else_inputs,
        extra.as_ref(),
        false,
        &name,
    )?;
    let else_args = reproduce_exit_args(
        &saved_exit,
        &flow_result,
        &else_value,
        &else_sources,
        &else_inputs,
        &name,
    )?;
    close_goto_mixed(graph, else_bb, b_target, else_args);

    let a_id = graph.blocks[a].id;
    let ops_len = graph.blocks[a].operations.len();
    let remove_upto = if ci + 2 == ops_len { ci + 1 } else { ci };
    for _ in ci..=remove_upto {
        graph.blocks[a].operations.remove(ci);
    }
    // Option receivers switch on their representation
    // (`emit_option_recv_tag`). Result combinators fall through to
    // `__discriminant`.
    let disc = emit_option_recv_tag(graph, a_id, recv, site);
    if then_is_tag1 {
        graph.set_branch(a_id, disc, then_bb, then_sources, else_bb, else_sources);
    } else {
        graph.set_branch(a_id, disc, else_bb, else_sources, then_bb, then_sources);
    }
    let _ = spec;
    Ok(())
}

pub(in crate::front::mir) fn rewire_option_filter_diamond(
    graph: &mut FunctionGraph,
    site: &DiscCombinatorSite,
    a: usize,
    ci: usize,
    recv: Variable,
    extra: Option<Variable>,
    flow_result: Variable,
    name: &str,
) -> Result<(), String> {
    use crate::front::option_map_or::emit_narrow;
    use crate::front::option_variant::{
        close_goto_mixed, emit_option_variant, map_source, reproduce_exit_args,
    };

    let env = match site.fn_item_segments {
        Some(_) => None,
        None => Some(extra.ok_or_else(|| format!("{name}: filter missing predicate"))?),
    };
    let [exit] = graph.blocks[a].exits.as_slice() else {
        return Err(format!("{name}: filter block does not have a single exit"));
    };
    if graph.blocks[a].exitswitch.is_some()
        || exit.exitcase.is_some()
        || exit.last_exception.is_some()
        || exit.last_exc_value.is_some()
    {
        return Err(format!("{name}: filter exit is not a plain goto"));
    }
    let saved_exit = exit.clone();
    let b_target = saved_exit.target;
    let mut carried: Vec<Variable> = Vec::new();
    for arg in &saved_exit.args {
        if let LinkArg::Value(v) = arg
            && *v != flow_result
            && !carried.contains(v)
        {
            carried.push(v.clone());
        }
    }

    let mut some_sources = carried.clone();
    if !some_sources.contains(&recv) {
        some_sources.push(recv.clone());
    }
    if let Some(env) = env.as_ref()
        && !some_sources.contains(env)
    {
        some_sources.push(env.clone());
    }
    let none_sources = carried.clone();
    let (some_bb, some_inputs) = graph.create_block_with_arg_vars(some_sources.len());
    let (none_bb, none_inputs) = graph.create_block_with_arg_vars(none_sources.len());

    // A niche result is the pointer or null. An aggregate `None` would make
    // the caller's null test read a live object. A scalar niche is the
    // integer 0 (`rint.py` `IntegerRepr.rtype_bool`).
    let none = emit_option_result_none(graph, none_bb, site);
    let none_args = reproduce_exit_args(
        &saved_exit,
        &flow_result,
        &none,
        &none_sources,
        &none_inputs,
        name,
    )?;
    close_goto_mixed(graph, none_bb, b_target, none_args);

    let recv_in = map_source(&some_sources, &some_inputs, &recv)
        .ok_or_else(|| format!("{name}: filter receiver not threaded"))?;
    let env_in = disc_callable_env(site, &some_sources, &some_inputs, env.as_ref(), name)?;
    // Niche `Some` is the option word itself. `__pos_0` would load a field
    // of that object (pointer) or of an integer (scalar).
    let payload = option_recv_payload(graph, some_bb, recv_in, site);
    let pred = emit_disc_callable(
        graph,
        some_bb,
        site,
        env_in,
        (
            payload.clone(),
            site.payload1_ty.clone(),
            site.payload1_class.clone(),
        ),
    )?;

    let mut keep_sources = some_inputs.clone();
    keep_sources.push(payload.clone());
    let (keep_bb, keep_inputs) = graph.create_block_with_arg_vars(keep_sources.len());
    let (drop_bb, drop_inputs) = graph.create_block_with_arg_vars(some_inputs.len());
    let keep_payload = keep_inputs
        .last()
        .cloned()
        .ok_or_else(|| format!("{name}: filter keep arm missing payload"))?;
    let keep_payload = emit_narrow(graph, keep_bb, keep_payload, &site.result_payload1_class);
    let some = if site.result_niche || site.result_scalar_niche {
        keep_payload
    } else {
        emit_option_variant(
            graph,
            keep_bb,
            &site.result_owner,
            1,
            Some((
                &site.result_tag1_owner,
                keep_payload,
                site.result_payload1_ty.clone(),
            )),
        )
    };
    let drop_none = emit_option_result_none(graph, drop_bb, site);

    let mut keep_exit = Vec::new();
    let mut drop_exit = Vec::new();
    for arg in &saved_exit.args {
        match arg {
            LinkArg::Const(c) => {
                keep_exit.push(LinkArg::Const(c.clone()));
                drop_exit.push(LinkArg::Const(c.clone()));
            }
            LinkArg::Value(v) if *v == flow_result => {
                keep_exit.push(LinkArg::Value(some.clone()));
                drop_exit.push(LinkArg::Value(drop_none.clone()));
            }
            LinkArg::Value(v) => {
                let in_some = map_source(&some_sources, &some_inputs, v)
                    .ok_or_else(|| format!("{name}: filter exit arg not threaded"))?;
                let in_keep = map_source(&keep_sources, &keep_inputs, &in_some)
                    .ok_or_else(|| format!("{name}: filter keep exit arg not threaded"))?;
                let in_drop = map_source(&some_inputs, &drop_inputs, &in_some)
                    .ok_or_else(|| format!("{name}: filter drop exit arg not threaded"))?;
                keep_exit.push(LinkArg::Value(in_keep));
                drop_exit.push(LinkArg::Value(in_drop));
            }
        }
    }
    close_goto_mixed(graph, keep_bb, b_target, keep_exit);
    close_goto_mixed(graph, drop_bb, b_target, drop_exit);
    graph.set_branch(
        some_bb,
        pred,
        keep_bb,
        keep_sources,
        drop_bb,
        some_inputs.clone(),
    );

    let a_id = graph.blocks[a].id;
    let ops_len = graph.blocks[a].operations.len();
    let remove_upto = if ci + 2 == ops_len { ci + 1 } else { ci };
    for _ in ci..=remove_upto {
        graph.blocks[a].operations.remove(ci);
    }
    let disc = emit_option_recv_tag(graph, a_id, recv, site);
    graph.set_branch(a_id, disc, some_bb, some_sources, none_bb, none_sources);
    Ok(())
}

/// The arm's closure env for `site`: the threaded `extra`, or `None` for a
/// function-item callable.
pub(in crate::front::mir) fn disc_callable_env(
    site: &DiscCombinatorSite,
    sources: &[Variable],
    inputs: &[Variable],
    extra: Option<&Variable>,
    name: &str,
) -> Result<Option<Variable>, String> {
    if site.fn_item_segments.is_some() {
        return Ok(None);
    }
    let method = site.kind.method_name();
    let extra = extra.ok_or_else(|| format!("{name}: {method} missing closure"))?;
    crate::front::option_variant::map_source(sources, inputs, extra)
        .map(Some)
        .ok_or_else(|| format!("{name}: {method} env not threaded"))
}

/// Run `site`'s callable on `arg` in `block`: `call_once(env, (arg,))` for a
/// closure, a direct call for a function item, and the constructed aggregate
/// for a constructor function item.
pub(in crate::front::mir) fn emit_disc_callable(
    graph: &mut FunctionGraph,
    block: BlockId,
    site: &DiscCombinatorSite,
    env: Option<Variable>,
    arg: (Variable, ValueType, Option<String>),
) -> Result<Variable, String> {
    let result_ty = if site.kind == DiscCombinator::OptionFilter {
        ValueType::Bool
    } else {
        site.call_result_ty.clone()
    };
    crate::front::option_closure_select::emit_callable(
        graph,
        block,
        env,
        &site.call_once_owner,
        site.fn_item_segments.as_deref(),
        site.fn_item_ctor.as_ref(),
        Some(arg),
        result_ty,
        &site.args_tuple_suffix,
    )
}

pub(in crate::front::mir) fn build_disc_arm(
    graph: &mut FunctionGraph,
    site: &DiscCombinatorSite,
    block: BlockId,
    recv: &Variable,
    sources: &[Variable],
    inputs: &[Variable],
    extra: Option<&Variable>,
    is_then: bool,
    name: &str,
) -> Result<Variable, String> {
    use crate::front::option_map_or::emit_narrow;
    use crate::front::option_variant::{emit_option_variant, emit_sum_variant, map_source};

    match (site.kind, is_then) {
        (DiscCombinator::OptionFilter, _) => unreachable!("filter has its own diamond"),
        (DiscCombinator::OptionOkOr, true) | (DiscCombinator::OptionOkOrElse, true) => {
            let recv = map_source(sources, inputs, recv)
                .ok_or_else(|| format!("{name}: ok_or receiver not threaded"))?;
            let payload = option_recv_payload(graph, block, recv, site);
            let payload = emit_narrow(graph, block, payload, &site.payload1_class);
            Ok(emit_sum_variant(
                graph,
                block,
                &site.result_owner,
                "Ok",
                0,
                Some((
                    &site.result_tag0_owner,
                    payload,
                    site.result_payload0_ty.clone(),
                )),
            ))
        }
        (DiscCombinator::OptionOkOr, false) => {
            let extra = extra.ok_or_else(|| format!("{name}: ok_or missing err value"))?;
            let err = map_source(sources, inputs, extra)
                .ok_or_else(|| format!("{name}: ok_or err not threaded"))?;
            let err = emit_narrow(graph, block, err, &site.result_payload1_class);
            Ok(emit_sum_variant(
                graph,
                block,
                &site.result_owner,
                "Err",
                1,
                Some((
                    &site.result_tag1_owner,
                    err,
                    site.result_payload1_ty.clone(),
                )),
            ))
        }
        (DiscCombinator::OptionOkOrElse, false) => {
            let env = disc_callable_env(site, sources, inputs, extra, name)?;
            let err = crate::front::option_closure_select::emit_callable(
                graph,
                block,
                env,
                &site.call_once_owner,
                site.fn_item_segments.as_deref(),
                site.fn_item_ctor.as_ref(),
                None,
                site.call_result_ty.clone(),
                &site.args_tuple_suffix,
            )?;
            let err = emit_narrow(graph, block, err, &site.call_result_class);
            Ok(emit_sum_variant(
                graph,
                block,
                &site.result_owner,
                "Err",
                1,
                Some((
                    &site.result_tag1_owner,
                    err,
                    site.result_payload1_ty.clone(),
                )),
            ))
        }
        (DiscCombinator::ResultMap, true) | (DiscCombinator::ResultAndThen, true) => {
            let recv = map_source(sources, inputs, recv)
                .ok_or_else(|| format!("{name}: Result receiver not threaded into Ok arm"))?;
            let env = disc_callable_env(site, sources, inputs, extra, name)?;
            let payload = emit_payload_read(
                graph,
                block,
                recv,
                &site.recv_tag0_owner,
                site.payload0_ty.clone(),
                site.payload0_is_raw_u8,
            );
            let mapped = emit_disc_callable(
                graph,
                block,
                site,
                env,
                (
                    payload,
                    site.payload0_ty.clone(),
                    site.payload0_class.clone(),
                ),
            )?;
            let mapped = emit_narrow(graph, block, mapped, &site.call_result_class);
            if site.kind == DiscCombinator::ResultAndThen {
                Ok(mapped)
            } else {
                Ok(emit_sum_variant(
                    graph,
                    block,
                    &site.result_owner,
                    "Ok",
                    0,
                    Some((
                        &site.result_tag0_owner,
                        mapped,
                        site.result_payload0_ty.clone(),
                    )),
                ))
            }
        }
        (DiscCombinator::ResultMap, false) | (DiscCombinator::ResultAndThen, false) => {
            let recv = map_source(sources, inputs, recv)
                .ok_or_else(|| format!("{name}: Result receiver not threaded into Err arm"))?;
            let payload = emit_payload_read(
                graph,
                block,
                recv,
                &site.recv_tag1_owner,
                site.payload1_ty.clone(),
                site.payload1_is_raw_u8,
            );
            let payload = emit_narrow(graph, block, payload, &site.payload1_class);
            Ok(emit_sum_variant(
                graph,
                block,
                &site.result_owner,
                "Err",
                1,
                Some((
                    &site.result_tag1_owner,
                    payload,
                    site.result_payload1_ty.clone(),
                )),
            ))
        }
        (DiscCombinator::ResultUnwrapOrElse, true) | (DiscCombinator::ResultOrElse, true) => {
            let recv = map_source(sources, inputs, recv)
                .ok_or_else(|| format!("{name}: Result receiver not threaded into Ok arm"))?;
            if site.kind == DiscCombinator::ResultOrElse {
                Ok(recv)
            } else {
                Ok(emit_payload_read(
                    graph,
                    block,
                    recv,
                    &site.recv_tag0_owner,
                    site.payload0_ty.clone(),
                    site.payload0_is_raw_u8,
                ))
            }
        }
        (DiscCombinator::ResultUnwrapOrElse, false) | (DiscCombinator::ResultOrElse, false) => {
            let recv = map_source(sources, inputs, recv)
                .ok_or_else(|| format!("{name}: Result receiver not threaded into Err arm"))?;
            let env = disc_callable_env(site, sources, inputs, extra, name)?;
            let payload = emit_payload_read(
                graph,
                block,
                recv,
                &site.recv_tag1_owner,
                site.payload1_ty.clone(),
                site.payload1_is_raw_u8,
            );
            emit_disc_callable(
                graph,
                block,
                site,
                env,
                (
                    payload,
                    site.payload1_ty.clone(),
                    site.payload1_class.clone(),
                ),
            )
        }
        (DiscCombinator::ResultOk, true) => {
            let recv = map_source(sources, inputs, recv)
                .ok_or_else(|| format!("{name}: Result::ok receiver not threaded"))?;
            let payload = emit_payload_read(
                graph,
                block,
                recv,
                &site.recv_tag0_owner,
                site.payload0_ty.clone(),
                site.payload0_is_raw_u8,
            );
            let payload = emit_narrow(graph, block, payload, &site.payload0_class);
            Ok(emit_option_variant(
                graph,
                block,
                &site.result_owner,
                1,
                Some((
                    &site.result_tag1_owner,
                    payload,
                    site.result_payload1_ty.clone(),
                )),
            ))
        }
        (DiscCombinator::ResultOk, false) => Ok(emit_option_variant(
            graph,
            block,
            &site.result_owner,
            0,
            None,
        )),
        (DiscCombinator::ResultErr, true) => {
            let recv = map_source(sources, inputs, recv)
                .ok_or_else(|| format!("{name}: Result::err receiver not threaded"))?;
            let payload = emit_payload_read(
                graph,
                block,
                recv,
                &site.recv_tag1_owner,
                site.payload1_ty.clone(),
                site.payload1_is_raw_u8,
            );
            let payload = emit_narrow(graph, block, payload, &site.payload1_class);
            Ok(emit_option_variant(
                graph,
                block,
                &site.result_owner,
                1,
                Some((
                    &site.result_tag1_owner,
                    payload,
                    site.result_payload1_ty.clone(),
                )),
            ))
        }
        (DiscCombinator::ResultErr, false) => Ok(emit_option_variant(
            graph,
            block,
            &site.result_owner,
            0,
            None,
        )),
        (DiscCombinator::ResultIsOk | DiscCombinator::ResultIsErr, _) => unreachable!(),
    }
}

/// Rewrite a Result combinator whose call `result_exc` already made can-raise.
#[allow(clippy::too_many_arguments)]
pub(in crate::front::mir) fn rewire_result_map_last_exception(
    graph: &mut FunctionGraph,
    site: &DiscCombinatorSite,
    spec: crate::ErrorCarrierSpec<'_>,
    (a, ci): (usize, usize),
    recv: Variable,
    extra: Option<Variable>,
    flow_result: Variable,
) -> Result<(), String> {
    use crate::front::option_variant::map_source;

    // `result_exc` already turned this Result-returning combinator into a
    // can-raise site: the normal exit carries the `Ok` payload and the
    // exception exit receives the `Err`.  `map`'s callable returns that
    // payload; `and_then`'s returns the carrier `Result`, so it is itself a
    // can-raise call.  A constructor `and_then` builds a `Result` value the
    // raising form has no slot for, so it keeps the residual.
    let name = graph.name.clone();
    let method = site.kind.method_name();
    let raising_callable = match site.kind {
        DiscCombinator::ResultMap => false,
        DiscCombinator::ResultAndThen if site.fn_item_ctor.is_none() => true,
        _ => {
            return Err(format!("{name}: {method} LastException form left residual"));
        }
    };
    if ci + 1 != graph.blocks[a].operations.len() {
        return Err(format!("{name}: {method} is not the raising operation"));
    }
    let normal = graph.blocks[a]
        .exits
        .iter()
        .find(|link| link.exitcase.is_none())
        .cloned()
        .ok_or_else(|| format!("{name}: {method} has no normal exit"))?;
    let raise = graph.blocks[a]
        .exits
        .iter()
        .find(|link| link.exitcase.is_some())
        .cloned()
        .ok_or_else(|| format!("{name}: {method} has no exception exit"))?;
    if raise
        .last_exception
        .as_ref()
        .and_then(LinkArg::as_variable)
        .is_none()
        || raise
            .last_exc_value
            .as_ref()
            .and_then(LinkArg::as_variable)
            .is_none()
    {
        return Err(format!(
            "{name}: {method} exception exit lacks its exception pair"
        ));
    }
    let arm = CombinatorArmExits {
        normal: &normal,
        raise: &raise,
        flow_result: &flow_result,
        raising_callable,
    };
    if drop_rebuilt_receiver_shell(graph, site, (a, ci), &recv, extra.as_ref(), &arm).is_ok() {
        return Ok(());
    }
    // The receiver stays a value.  Its `Err` arm raises the carrier the way a
    // `?` on it would; a local handler expecting the rebuilt shell is not
    // proven here.
    if raise.target != graph.exceptblock {
        return Err(format!("{name}: {method} LastException form left residual"));
    }
    let exc_vars = [&raise.last_exception, &raise.last_exc_value];
    let mut ok_sources: Vec<Variable> = Vec::new();
    let mut err_sources: Vec<Variable> = Vec::new();
    // A raising callable leaves the `Ok` arm on both exits.
    let ok_links: &[&Link] = if raising_callable {
        &[&normal, &raise]
    } else {
        &[&normal]
    };
    for (links, sources) in [
        (ok_links, &mut ok_sources),
        (&[&raise][..], &mut err_sources),
    ] {
        for arg in links.iter().flat_map(|link| &link.args) {
            if let LinkArg::Value(v) = arg
                && *v != flow_result
                && !exc_vars.iter().any(|e| e.as_ref() == Some(arg))
                && !sources.contains(v)
            {
                sources.push(v.clone());
            }
        }
    }
    if !ok_sources.contains(&recv) {
        ok_sources.push(recv.clone());
    }
    if site.fn_item_segments.is_none()
        && let Some(env) = extra.as_ref()
        && !ok_sources.contains(env)
    {
        ok_sources.push(env.clone());
    }
    if !err_sources.contains(&recv) {
        err_sources.push(recv.clone());
    }
    let (ok_bb, ok_inputs) = graph.create_block_with_arg_vars(ok_sources.len());
    let (err_bb, err_inputs) = graph.create_block_with_arg_vars(err_sources.len());

    // `Ok(x)` → `f(x)` on the normal exit.
    let recv_ok = map_source(&ok_sources, &ok_inputs, &recv)
        .ok_or_else(|| format!("{name}: {method} receiver not threaded into Ok arm"))?;
    let payload = emit_payload_read(
        graph,
        ok_bb,
        recv_ok,
        &site.recv_tag0_owner,
        site.payload0_ty.clone(),
        site.payload0_is_raw_u8,
    );
    let env = disc_callable_env(site, &ok_sources, &ok_inputs, extra.as_ref(), &name)?;
    arm.close_with_callable(graph, site, ok_bb, env, payload, &|v| {
        map_source(&ok_sources, &ok_inputs, v)
    })?;

    // `Err(e)` → raise `e` on the exception exit.
    let recv_err = map_source(&err_sources, &err_inputs, &recv)
        .ok_or_else(|| format!("{name}: {method} receiver not threaded into Err arm"))?;
    let error = emit_payload_read(
        graph,
        err_bb,
        recv_err,
        &site.recv_tag1_owner,
        site.payload1_ty.clone(),
        site.payload1_is_raw_u8,
    );
    crate::front::result_map_err::raise_carrier_on_exception_edge(
        graph,
        err_bb,
        error,
        &raise,
        &err_sources,
        &err_inputs,
        spec,
        &name,
    )?;

    let a_id = graph.blocks[a].id;
    graph.blocks[a].operations.truncate(ci);
    let disc = emit_enum_disc_read(graph, a_id, recv, &site.recv_owner);
    graph.set_branch(a_id, disc, err_bb, err_sources, ok_bb, ok_sources);
    Ok(())
}

/// The two exits of a combinator call `result_exc` made can-raise.
pub(in crate::front::mir) struct CombinatorArmExits<'l> {
    pub(in crate::front::mir) normal: &'l Link,
    pub(in crate::front::mir) raise: &'l Link,
    /// The call's result on `normal`: the `Ok` payload.
    pub(in crate::front::mir) flow_result: &'l Variable,
    /// The callable itself raises the carrier's `Err` (`and_then`).
    pub(in crate::front::mir) raising_callable: bool,
}

impl CombinatorArmExits<'_> {
    /// Run the site's callable on `payload` at the end of `block` and leave on
    /// the normal exit with its result, remapping every other value through
    /// `remap`.  A raising callable is the block's can-raise operation: its
    /// exception exit is a copy of the combinator's, with a fresh exception
    /// pair.
    pub(in crate::front::mir) fn close_with_callable(
        &self,
        graph: &mut FunctionGraph,
        site: &DiscCombinatorSite,
        block: BlockId,
        env: Option<Variable>,
        payload: Variable,
        remap: &dyn Fn(&Variable) -> Option<Variable>,
    ) -> Result<(), String> {
        let name = graph.name.clone();
        let result_ty = if self.raising_callable {
            site.result_payload0_ty.clone()
        } else {
            site.call_result_ty.clone()
        };
        let value = crate::front::option_closure_select::emit_callable(
            graph,
            block,
            env,
            &site.call_once_owner,
            site.fn_item_segments.as_deref(),
            site.fn_item_ctor.as_ref(),
            Some((
                payload,
                site.payload0_ty.clone(),
                site.payload0_class.clone(),
            )),
            result_ty,
            &site.args_tuple_suffix,
        )
        .map_err(|msg| format!("{name}: {msg}"))?;
        let value = if self.raising_callable {
            value
        } else {
            crate::front::option_map_or::emit_narrow(graph, block, value, &site.call_result_class)
        };
        let remap_arg = |arg: &LinkArg, fresh: &[(&Option<LinkArg>, &Variable)]| match arg {
            LinkArg::Value(v) if v == self.flow_result => Some(LinkArg::Value(value.clone())),
            LinkArg::Value(_) => fresh
                .iter()
                .find(|(slot, _)| slot.as_ref() == Some(arg))
                .map(|(_, var)| LinkArg::Value((*var).clone()))
                .or_else(|| arg.as_variable().and_then(remap).map(LinkArg::Value)),
            LinkArg::Const(c) => Some(LinkArg::Const(c.clone())),
        };
        let normal_args = self
            .normal
            .args
            .iter()
            .map(|arg| remap_arg(arg, &[]))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| format!("{name}: combinator normal exit arg not threaded"))?;
        let normal = Link::new_mixed(normal_args, self.normal.target, None);
        if !self.raising_callable {
            graph.set_control_flow_metadata(block, None, vec![normal]);
            return Ok(());
        }
        let etype = graph.alloc_value_var();
        let evalue = graph.alloc_value_var();
        let raise_args = self
            .raise
            .args
            .iter()
            .map(|arg| {
                remap_arg(
                    arg,
                    &[
                        (&self.raise.last_exception, &etype),
                        (&self.raise.last_exc_value, &evalue),
                    ],
                )
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| format!("{name}: combinator exception exit arg not threaded"))?;
        let mut raise = Link::new_mixed(raise_args, self.raise.target, self.raise.exitcase.clone());
        raise.llexitcase = self.raise.llexitcase.clone();
        raise.last_exception = Some(LinkArg::Value(etype));
        raise.last_exc_value = Some(LinkArg::Value(evalue));
        graph.set_control_flow_metadata(
            block,
            Some(ExitSwitch::LastException),
            vec![normal, raise],
        );
        Ok(())
    }
}

/// Drop the `Ok`/`Err` shells `result_exc::catch_and_rewrap` rebuilt for
/// the combinator's receiver alone.  The normal arm of the producing call
/// applies the callable to its payload; its exception arm forwards the caught
/// exception pair straight to the combinator's exception exit, so no carrier
/// round trip happens.  Every check runs before the first edit; `Err` leaves
/// the graph unchanged.
pub(in crate::front::mir) fn drop_rebuilt_receiver_shell(
    graph: &mut FunctionGraph,
    site: &DiscCombinatorSite,
    (a, ci): (usize, usize),
    recv: &Variable,
    env: Option<&Variable>,
    arm: &CombinatorArmExits<'_>,
) -> Result<(), String> {
    use crate::front::result_exc::{count_var_uses, delete_ops, result_ctor_kind, shell_build_ops};

    let block_a = &graph.blocks[a];
    let a_id = block_a.id;
    let only_use = |graph: &FunctionGraph, var: &Variable, ops: usize, links: usize| {
        let uses = count_var_uses(graph, var);
        uses.op_uses == ops && uses.link_uses == links
    };
    let Some(recv_pos) = block_a.inputargs.iter().position(|v| v == recv) else {
        return Err("receiver is not an inputarg of the combinator block".to_string());
    };
    if block_a.inputargs.iter().filter(|v| *v == recv).count() != 1 || !only_use(graph, recv, 1, 0)
    {
        return Err("receiver shell has another consumer".to_string());
    }
    // Only function-item defines the call alone reads may precede the call.
    let OpKind::Call {
        args: call_args, ..
    } = &block_a.operations[ci].kind
    else {
        return Err("combinator is not a call".to_string());
    };
    for op in &block_a.operations[..ci] {
        let OpKind::Call { target, args, .. } = &op.kind else {
            return Err("combinator block computes more than the call".to_string());
        };
        let define = args.is_empty() && crate::model::fn_const_segments(target).is_some();
        if !define
            || !op.result.as_ref().is_some_and(|r| {
                call_args.contains(&LinkArg::Value(r.clone())) && only_use(graph, r, 1, 0)
            })
        {
            return Err("combinator block computes more than the call".to_string());
        }
    }
    let env_pos = match (&site.fn_item_segments, env) {
        (None, Some(env)) => Some(
            block_a
                .inputargs
                .iter()
                .position(|v| v == env)
                .ok_or("closure env is not an inputarg of the combinator block")?,
        ),
        _ => None,
    };
    // Every value leaving the block is one of its inputargs (not the shell),
    // the call result, or the exception pair.
    for link in [arm.normal, arm.raise] {
        for arg in &link.args {
            if let LinkArg::Value(v) = arg
                && v != arm.flow_result
                && Some(arg) != link.last_exception.as_ref()
                && Some(arg) != link.last_exc_value.as_ref()
                && (v == recv || !block_a.inputargs.contains(v))
            {
                return Err("combinator exit carries a value the block computes".to_string());
            }
        }
    }
    let preds: Vec<usize> = graph
        .blocks
        .iter()
        .enumerate()
        .filter(|(_, block)| block.exits.iter().any(|link| link.target == a_id))
        .map(|(i, _)| i)
        .collect();
    let [p0, p1] = preds.as_slice() else {
        return Err("combinator block is not private to two rebuilt shells".to_string());
    };
    let shell_of = |block: usize| -> Result<(Variable, bool, Vec<LinkArg>), String> {
        let b = &graph.blocks[block];
        let [exit] = b.exits.as_slice() else {
            return Err(format!("shell block {block} has more than one exit"));
        };
        if b.exitswitch.is_some() || exit.exitcase.is_some() {
            return Err(format!("shell block {block} exit is not a plain goto"));
        }
        let Some(LinkArg::Value(shell)) = exit.args.get(recv_pos) else {
            return Err(format!("shell block {block} does not pass a shell"));
        };
        let is_err = b
            .operations
            .iter()
            .find(|op| op.result.as_ref() == Some(shell))
            .and_then(|op| match &op.kind {
                OpKind::Call { target, .. } => result_ctor_kind(target),
                _ => None,
            })
            .ok_or_else(|| format!("shell block {block} does not build a Result shell"))?;
        if !only_use(graph, shell, 1, 1) {
            return Err(format!("shell of block {block} has another consumer"));
        }
        Ok((shell.clone(), is_err, exit.args.clone()))
    };
    let (s0, err0, args0) = shell_of(*p0)?;
    let (s1, err1, args1) = shell_of(*p1)?;
    let ((n, ok_shell, n_args), (e, err_shell, e_args)) = match (err0, err1) {
        (false, true) => ((*p0, s0, args0), (*p1, s1, args1)),
        (true, false) => ((*p1, s1, args1), (*p0, s0, args0)),
        _ => return Err("predecessors are not one Ok and one Err shell".to_string()),
    };
    let payload_of = |block: usize, shell: &Variable| {
        graph.blocks[block]
            .operations
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::FieldWrite {
                    base,
                    field,
                    value: LinkArg::Value(value),
                    ..
                } if base == shell && field.name == "__pos_0" => Some(value.clone()),
                _ => None,
            })
    };
    let ok_payload = payload_of(n, &ok_shell).ok_or("Ok shell has no payload")?;
    let err_payload = payload_of(e, &err_shell).ok_or("Err shell has no payload")?;
    // The Err arm is `catch_and_rewrap`'s handler: its last two inputargs are
    // the caught exception pair, and its carrier is that value or its
    // `from_exc_object` rebuild, read by nothing but the shell.
    let e_inputs = &graph.blocks[e].inputargs;
    let [.., caught_type, caught_value] = e_inputs.as_slice() else {
        return Err("Err shell block does not receive an exception pair".to_string());
    };
    let e_id = graph.blocks[e].id;
    let caught_by_producer = graph.blocks.iter().any(|block| {
        block.exits.iter().any(|link| {
            link.target == e_id
                && link.exitcase.is_some()
                && link.args.len() == e_inputs.len()
                && link.last_exception.as_ref() == link.args.get(e_inputs.len() - 2)
                && link.last_exc_value.as_ref() == link.args.last()
        })
    });
    if !caught_by_producer {
        return Err("Err shell block is not the producer's exception handler".to_string());
    }
    let mut err_build = shell_build_ops(graph, e, &err_shell, false)?;
    if err_payload != *caught_value {
        let rebuild = graph.blocks[e]
            .operations
            .iter()
            .position(|op| {
                op.result.as_ref() == Some(&err_payload)
                    && matches!(&op.kind, OpKind::Call { args, .. }
                        if args.len() == 1 && args[0] == LinkArg::Value(caught_value.clone()))
            })
            .ok_or("Err shell payload is not the caught exception")?;
        if !only_use(graph, &err_payload, 1, 0) {
            return Err("Err shell payload has another consumer".to_string());
        }
        err_build.push(rebuild);
        err_build.sort_unstable();
    }
    let ok_build = shell_build_ops(graph, n, &ok_shell, false)?;
    let a_inputs = graph.blocks[a].inputargs.clone();
    let via = |args: &[LinkArg], v: &Variable| {
        a_inputs
            .iter()
            .position(|input| input == v)
            .and_then(|i| args.get(i))
            .and_then(LinkArg::as_variable)
            .cloned()
    };
    let n_env = match env_pos {
        Some(pos) => Some(
            n_args
                .get(pos)
                .and_then(LinkArg::as_variable)
                .cloned()
                .ok_or("closure env reaches the Ok shell block as a constant")?,
        ),
        None => None,
    };
    let e_exit_args = arm
        .raise
        .args
        .iter()
        .map(|arg| match arg {
            _ if Some(arg) == arm.raise.last_exception.as_ref() => {
                Some(LinkArg::Value(caught_type.clone()))
            }
            _ if Some(arg) == arm.raise.last_exc_value.as_ref() => {
                Some(LinkArg::Value(caught_value.clone()))
            }
            LinkArg::Value(v) => via(&e_args, v).map(LinkArg::Value),
            LinkArg::Const(c) => Some(LinkArg::Const(c.clone())),
        })
        .collect::<Option<Vec<_>>>()
        .ok_or("combinator exception exit arg does not reach the Err shell block")?;
    let n_remap = |v: &Variable| via(&n_args, v);
    if arm
        .normal
        .args
        .iter()
        .chain(arm.raise.args.iter().filter(|_| arm.raising_callable))
        .filter_map(LinkArg::as_variable)
        .any(|v| {
            v != arm.flow_result
                && Some(&LinkArg::Value(v.clone())) != arm.raise.last_exception.as_ref()
                && Some(&LinkArg::Value(v.clone())) != arm.raise.last_exc_value.as_ref()
                && n_remap(v).is_none()
        })
    {
        return Err("combinator exit arg does not reach the Ok shell block".to_string());
    }

    // Every check has run.
    delete_ops(graph, n, ok_build);
    delete_ops(graph, e, err_build);
    let n_id = graph.blocks[n].id;
    arm.close_with_callable(graph, site, n_id, n_env, ok_payload, &n_remap)?;
    graph.set_control_flow_metadata(
        e_id,
        None,
        vec![Link::new_mixed(e_exit_args, arm.raise.target, None)],
    );
    Ok(())
}

/// Option receiver tag by the Option's own representation, shared by
/// `ok_or`, `ok_or_else`, and `filter`.
///
/// Pointer niche: `rnone.py` `rtype_is_None` / `CanBeNull.rtype_bool`
/// (`ptr_nonzero`) via [`FunctionGraph::push_niche_is_some`]. Scalar
/// niche: `rint.py` `IntegerRepr.rtype_bool` (`int_is_true`) as `ne`
/// against 0. Aggregate: `__discriminant`.
pub(in crate::front::mir) fn emit_option_recv_tag(
    graph: &mut FunctionGraph,
    block: BlockId,
    recv: Variable,
    site: &DiscCombinatorSite,
) -> Variable {
    if site.recv_niche {
        emit_niche_some_tag(
            graph,
            block,
            recv,
            site.int_null.as_ref(),
            &site.niche_null_cast,
        )
    } else if site.scalar_niche {
        let disc = graph.alloc_value_var();
        let zero = graph.alloc_value_var();
        graph.block_mut(block).operations.push(SpaceOperation {
            result: Some(zero.clone()),
            kind: nonzero_option_zero(&site.recv_owner),
        });
        graph.block_mut(block).operations.push(SpaceOperation {
            result: Some(disc.clone()),
            kind: OpKind::BinOp {
                op: "ne".to_string(),
                lhs: recv,
                rhs: zero,
                result_ty: ValueType::Int,
            },
        });
        disc
    } else {
        emit_enum_disc_read(graph, block, recv, &site.recv_owner)
    }
}

/// Niche payload is the option word. Aggregate `Some` is `__pos_0`.
pub(in crate::front::mir) fn option_recv_payload(
    graph: &mut FunctionGraph,
    block: BlockId,
    recv: Variable,
    site: &DiscCombinatorSite,
) -> Variable {
    if site.recv_niche || site.scalar_niche {
        recv
    } else {
        emit_payload_read(
            graph,
            block,
            recv,
            &site.recv_tag1_owner,
            site.payload1_ty.clone(),
            site.payload1_is_raw_u8,
        )
    }
}

/// `None` of the combinator's result Option, by that Option's representation.
pub(in crate::front::mir) fn emit_option_result_none(
    graph: &mut FunctionGraph,
    block: BlockId,
    site: &DiscCombinatorSite,
) -> Variable {
    use crate::front::option_variant::emit_option_variant;
    if site.result_niche {
        emit_niche_null(graph, block, site.int_null.as_ref(), &site.niche_null_cast)
    } else if site.result_scalar_niche {
        graph
            .push_op_var(block, nonzero_option_zero(&site.result_owner), true)
            .expect("scalar None produces a value")
    } else {
        emit_option_variant(graph, block, &site.result_owner, 0, None)
    }
}

/// `Some` tag of a niche `Option`: `eq(is_(ptr, null), false)`.
///
/// `is_` is the null test. `ne` compares string and list values, so it is
/// not a discriminant. The `None` null shares the payload class when
/// `null_cast` is set (`push_niche_null_ptr`). `int_null` is the int-bank
/// null (`rnone.py` `rtype_is_None`); every other niche uses
/// [`FunctionGraph::push_niche_null`].
pub(in crate::front::mir) fn emit_niche_some_tag(
    graph: &mut FunctionGraph,
    block: BlockId,
    recv: Variable,
    int_null: Option<&crate::model::NicheIntNull>,
    null_cast: &Option<(String, ValueType)>,
) -> Variable {
    let disc = graph.alloc_value_var();
    graph.push_niche_is_some(block, recv, int_null, null_cast.as_ref(), disc.clone());
    disc
}

pub(in crate::front::mir) fn emit_niche_null(
    graph: &mut FunctionGraph,
    block: BlockId,
    int_null: Option<&crate::model::NicheIntNull>,
    null_cast: &Option<(String, ValueType)>,
) -> Variable {
    match int_null {
        Some(int_null) => int_null.push(graph, block),
        None => graph.push_niche_null(block, null_cast.as_ref()),
    }
}

/// Reproduce a diamond arm's goto args: the placeholder dest slot is
/// the arm's produced value, every other live Variable is re-sourced
/// from the arm's threaded inputarg (`option_variant::map_source`).
pub(in crate::front::mir) fn remap_diamond_goto_args(
    link_args: &[Variable],
    placeholder: &Variable,
    produced: &Variable,
    sources: &[Variable],
    inputs: &[Variable],
) -> Vec<Variable> {
    link_args
        .iter()
        .map(|v| {
            if v == placeholder {
                produced.clone()
            } else {
                crate::front::option_variant::map_source(sources, inputs, v)
                    .expect("diamond edge arg carried")
            }
        })
        .collect()
}

pub(in crate::front::mir) fn emit_enum_disc_read(
    graph: &mut FunctionGraph,
    block: BlockId,
    recv: Variable,
    owner: &str,
) -> Variable {
    let disc = graph.alloc_value_var();
    graph.block_mut(block).operations.push(SpaceOperation {
        result: Some(disc.clone()),
        kind: OpKind::FieldRead {
            base: recv,
            field: FieldDescriptor {
                name: "__discriminant".to_string(),
                owner_root: Some(owner.to_string()),
                owner_id: None,
                base_is_deref: None,
                taken_by_address: false,
                inline_vec: false,
                vec_part: None,
                owner_declared_gc: None,
                host_index: None,
                scalar_word: None,
            },
            ty: ValueType::Int,
            pure: true,
        },
    });
    disc
}

pub(in crate::front::mir) fn emit_payload_read(
    graph: &mut FunctionGraph,
    block: BlockId,
    recv: Variable,
    owner: &str,
    ty: ValueType,
    field_is_raw_u8: bool,
) -> Variable {
    let payload = graph.alloc_value_var();
    graph.block_mut(block).operations.push(SpaceOperation {
        result: Some(payload.clone()),
        kind: OpKind::FieldRead {
            base: recv,
            field: FieldDescriptor {
                name: "__pos_0".to_string(),
                owner_root: Some(owner.to_string()),
                owner_id: None,
                base_is_deref: None,
                taken_by_address: false,
                inline_vec: false,
                vec_part: None,
                owner_declared_gc: None,
                host_index: None,
                scalar_word: None,
            },
            ty: ty.clone(),
            pure: true,
        },
    });
    if field_is_raw_u8 {
        stamp_as_ccharp(&payload);
    }
    payload
}

/// Two Charon type-decl layouts describe the same bits: size, alignment,
/// per-variant field offsets, and the discriminant encoding.
pub(in crate::front::mir) fn type_layouts_equal(
    a: &majit_charon_reader::ullbc::TypeLayout,
    b: &majit_charon_reader::ullbc::TypeLayout,
) -> bool {
    if a.size != b.size || a.align != b.align {
        return false;
    }
    if a.variant_layouts.len() != b.variant_layouts.len() {
        return false;
    }
    if a.variant_layouts
        .iter()
        .zip(&b.variant_layouts)
        .any(|(left, right)| left.field_offsets != right.field_offsets)
    {
        return false;
    }
    if a.discriminator != b.discriminator {
        return false;
    }
    let transparent = |layout: &majit_charon_reader::ullbc::TypeLayout| {
        layout.repr.as_ref().is_some_and(|repr| repr.transparent)
    };
    transparent(a) == transparent(b)
}

/// Replace a `Result::branch` call with `same_as` when the two type-decl
/// layouts are equal. A missing layout, or two layouts that differ, leaves
/// the call for [`lower_result_branch_to_control_flow`].
pub(in crate::front::mir) fn rewrite_result_branch_for_layouts(
    kind: OpKind,
    recv_layout: Option<&majit_charon_reader::ullbc::TypeLayout>,
    flow_layout: Option<&majit_charon_reader::ullbc::TypeLayout>,
) -> OpKind {
    let (Some(recv_layout), Some(flow_layout)) = (recv_layout, flow_layout) else {
        return kind;
    };
    if !type_layouts_equal(recv_layout, flow_layout) {
        return kind;
    }
    let OpKind::Call {
        target,
        args,
        result_ty,
    } = &kind
    else {
        return kind;
    };
    let CallTarget::Method {
        name,
        receiver_root,
        ..
    } = target
    else {
        return kind;
    };
    if name != "branch"
        || !receiver_root.as_deref().is_some_and(|root| {
            majit_ir::descr::strip_instantiation_suffix(majit_ir::descr::class_root_leaf(root))
                == "Result"
        })
    {
        return kind;
    }
    let Some(operand) = args.first().and_then(LinkArg::as_variable).cloned() else {
        return kind;
    };
    OpKind::UnaryOp {
        op: "same_as".to_string(),
        operand,
        result_ty: result_ty.clone(),
    }
}

/// `Try::branch` on a `Result` that the diamond rewrites did not consume
/// and that was not already emitted as `same_as`.
///
/// Read the discriminant and build `Continue` or `Break`.
pub(in crate::front::mir) fn lower_result_branch_to_control_flow(graph: &mut FunctionGraph) {
    let mut sites = Vec::new();
    for (bi, block) in graph.blocks.iter().enumerate() {
        for (oi, op) in block.operations.iter().enumerate() {
            let OpKind::Call {
                target,
                args,
                result_ty,
            } = &op.kind
            else {
                continue;
            };
            let CallTarget::Method {
                name,
                receiver_root,
                ..
            } = target
            else {
                continue;
            };
            if name != "branch"
                || !receiver_root.as_deref().is_some_and(|root| {
                    majit_ir::descr::strip_instantiation_suffix(majit_ir::descr::class_root_leaf(
                        root,
                    )) == "Result"
                })
            {
                continue;
            }
            let Some(operand) = args.first().and_then(LinkArg::as_variable).cloned() else {
                continue;
            };
            let Some(result) = op.result.clone() else {
                continue;
            };
            let payloads = match target {
                CallTarget::Method {
                    branch_payloads, ..
                } => branch_payloads.clone(),
                _ => None,
            };
            sites.push((
                bi,
                oi,
                operand,
                result,
                result_ty.clone(),
                receiver_root
                    .clone()
                    .unwrap_or_else(|| "core::result::Result".to_string()),
                payloads,
            ));
        }
    }
    for (bi, oi, operand, result, result_ty, result_owner, payloads) in sites.into_iter().rev() {
        lower_result_branch_as_match(
            graph,
            BlockId(bi),
            oi,
            &operand,
            &result,
            &result_owner,
            &result_ty,
            payloads.as_ref(),
        );
    }
}

pub(in crate::front::mir) fn control_flow_owner(result_ty: &ValueType) -> String {
    match result_ty {
        ValueType::Ref(Some(owner)) if !owner.is_empty() => owner.clone(),
        _ => "core::ops::control_flow::ControlFlow".to_string(),
    }
}

pub(in crate::front::mir) fn push_enum_field_read(
    graph: &mut FunctionGraph,
    block: BlockId,
    base: Variable,
    owner: &str,
    field: &str,
    ty: ValueType,
    field_is_raw_u8: bool,
) -> Variable {
    let read = graph.alloc_value_var();
    graph.block_mut(block).operations.push(SpaceOperation {
        result: Some(read.clone()),
        kind: OpKind::FieldRead {
            base,
            field: FieldDescriptor {
                name: field.to_string(),
                owner_root: Some(owner.to_string()),
                owner_id: None,
                base_is_deref: None,
                taken_by_address: false,
                inline_vec: false,
                vec_part: None,
                owner_declared_gc: None,
                host_index: None,
                scalar_word: None,
            },
            ty: ty.clone(),
            pure: true,
        },
    });
    if field_is_raw_u8 {
        stamp_as_ccharp(&read);
    }
    read
}

/// Replace one `Result::branch` call with a discriminant switch that builds
/// `ControlFlow::Continue` or `ControlFlow::Break`.
/// Register bank of one enum variant's payload, taken from that variant's
/// field on the type decl. A `TypeVar` field is the corresponding generic
/// argument of `ty`. Unit and other zero-sized payloads are `None`: the
/// match emits no read and no field for them.
pub(in crate::front::mir) fn adt_variant_payload_type(
    ty: &TyRef,
    variant: &str,
    llbc: &Llbc,
    tombstoned: &std::collections::HashSet<String>,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> Option<ValueType> {
    let node = tyref_node(ty, llbc).and_then(|node| strip_ty_wrappers(node, llbc))?;
    let def_id = adt_node_def_id(node)?;
    let td = llbc.type_by_id(def_id)?;
    let TypeDeclKind::Enum(variants) = &td.kind else {
        return None;
    };
    let field = variants
        .iter()
        .find(|decl| decl.name == variant)?
        .fields
        .first()?;
    let concrete = substitute_typevar_field(&field.ty, node, llbc)?;
    if is_unit_type(&concrete, llbc)
        || (tyref_is_void_zst(&concrete, llbc)
            && !tyref_keeps_empty_class_instance(&concrete, llbc, tombstoned))
    {
        return None;
    }
    let value = tyref_enum_payload_value_type(
        &concrete,
        llbc,
        tombstoned,
        gc_struct_ids,
        raw_alloc_flavor_structs,
    );
    if value == ValueType::Void {
        None
    } else {
        Some(value)
    }
}

pub(in crate::front::mir) fn substitute_typevar_field(
    field_ty: &TyRef,
    owner_adt: &serde_json::Value,
    llbc: &Llbc,
) -> Option<TyRef> {
    let node = tyref_node(field_ty, llbc).and_then(|node| strip_ty_indirections(node, llbc))?;
    if let Some(index) = typevar_bound_index(node) {
        let arg = type_decl_ref_generics(owner_adt.get("Adt")?.as_object()?, llbc)?
            .get("types")?
            .as_array()?
            .get(index as usize)?;
        return serde_json::from_value(arg.clone()).ok();
    }
    Some(clone_tyref(field_ty))
}

pub(in crate::front::mir) fn lower_result_branch_as_match(
    graph: &mut FunctionGraph,
    block: BlockId,
    op_idx: usize,
    operand: &Variable,
    result: &Variable,
    result_owner: &str,
    result_ty: &ValueType,
    payloads: Option<&crate::model::ResultBranchPayloads>,
) {
    use crate::front::option_variant::{emit_sum_variant, map_source};

    let cf_owner = control_flow_owner(result_ty);
    let mut prefix_defined = graph.block(block).inputargs.clone();
    for op in graph.block(block).operations.iter().take(op_idx) {
        if let Some(produced) = &op.result {
            prefix_defined.push(produced.clone());
        }
    }
    prefix_defined.retain(|var| var != result);
    if !prefix_defined.iter().any(|var| var == operand) {
        prefix_defined.push(operand.clone());
    }

    graph.block_mut(block).operations.remove(op_idx);
    let tail = graph.block_mut(block).operations.split_off(op_idx);
    let exitswitch = graph.block_mut(block).exitswitch.take();
    let exits = std::mem::take(&mut graph.block_mut(block).exits);

    let (cont, inputs) = graph.create_block_with_arg_vars(prefix_defined.len() + 1);
    let phi = inputs.last().expect("phi").clone();
    let mut remap_map: std::collections::HashMap<Variable, Variable> = prefix_defined
        .iter()
        .cloned()
        .zip(inputs.iter().cloned())
        .collect();
    remap_map.insert(result.clone(), phi);
    let remap = |var: &Variable| remap_map.get(var).cloned().unwrap_or_else(|| var.clone());
    graph.block_mut(cont).operations = tail
        .into_iter()
        .map(|op| SpaceOperation {
            result: op.result.clone(),
            kind: crate::inline::remap_op_kind(&op.kind, &remap),
        })
        .collect();
    graph.block_mut(cont).exitswitch = match exitswitch {
        Some(ExitSwitch::Value(var)) => Some(ExitSwitch::Value(remap(&var))),
        Some(ExitSwitch::Fused { opname, args }) => Some(ExitSwitch::Fused {
            opname,
            args: args.into_iter().map(|arg| remap(&arg)).collect(),
        }),
        other => other,
    };
    graph.block_mut(cont).exits = exits
        .into_iter()
        .map(|mut link| {
            for arg in &mut link.args {
                if let Some(var) = arg.as_variable()
                    && let Some(mapped) = remap_map.get(var)
                {
                    *arg = LinkArg::Value(mapped.clone());
                }
            }
            link
        })
        .collect();

    let disc = push_enum_field_read(
        graph,
        block,
        operand.clone(),
        result_owner,
        "__discriminant",
        ValueType::Int,
        false,
    );

    let mut arm_ids = Vec::new();
    let mut arm_sources = Vec::new();
    for (tag, flow_variant, result_variant) in [(0i64, "Continue", "Ok"), (1i64, "Break", "Err")] {
        let sources = prefix_defined.clone();
        let (arm, arm_inputs) = graph.create_block_with_arg_vars(sources.len());
        let operand_arm = map_source(&sources, &arm_inputs, operand).expect("operand threaded");
        let (read_ty, write_ty) = match (tag, payloads) {
            (0, Some(payloads)) => (payloads.ok.clone(), payloads.continue_ty.clone()),
            (1, Some(payloads)) => (payloads.err.clone(), payloads.break_ty.clone()),
            // A call lowered without type decls keeps the reference bank the
            // match used before the decls were consulted.
            _ => (Some(ValueType::Ref(None)), Some(ValueType::Ref(None))),
        };
        let payload_owner = format!("{cf_owner}::{flow_variant}");
        let payload = match (read_ty, write_ty) {
            (Some(read_ty), Some(write_ty)) => {
                let read = push_enum_field_read(
                    graph,
                    arm,
                    operand_arm,
                    &format!("{result_owner}::{result_variant}"),
                    "__pos_0",
                    read_ty.clone(),
                    false,
                );
                // Kinds differ: `Break` holds `Result<Infallible, E>`, so
                // store `Err(e)` rather than the bare `E`.
                let (value, field_ty) = if tag == 1 && read_ty != write_ty {
                    let err_field = format!("{result_owner}::Err");
                    let wrapped = emit_sum_variant(
                        graph,
                        arm,
                        result_owner,
                        "Err",
                        1,
                        Some((err_field.as_str(), read, read_ty)),
                    );
                    (wrapped, write_ty)
                } else {
                    (read, write_ty)
                };
                Some((payload_owner, value, field_ty))
            }
            _ => None,
        };
        let built = emit_sum_variant(
            graph,
            arm,
            &cf_owner,
            flow_variant,
            tag,
            payload
                .as_ref()
                .map(|(owner, value, ty)| (owner.as_str(), value.clone(), ty.clone())),
        );
        let mut args: Vec<Variable> = sources
            .iter()
            .map(|var| map_source(&sources, &arm_inputs, var).expect("prefix threaded"))
            .collect();
        args.push(built);
        graph.set_goto(arm, cont, args);
        arm_ids.push(arm);
        arm_sources.push(sources);
    }
    let break_sources = arm_sources.pop().expect("break");
    let continue_sources = arm_sources.pop().expect("continue");
    let break_arm = arm_ids.pop().expect("break");
    let continue_arm = arm_ids.pop().expect("continue");
    graph.set_branch(
        block,
        disc,
        break_arm,
        break_sources,
        continue_arm,
        continue_sources,
    );
}

pub(in crate::front::mir) fn rewire_result_try_call_sites(
    graph: &mut FunctionGraph,
    sites: &[ResultTrySite],
    return_owners: Option<ResultTryReturnOwners>,
) -> ResultTryStats {
    let mut stats = ResultTryStats::default();
    for site in sites {
        if rewire_one_result_try_site(graph, site, return_owners.as_ref()).is_ok() {
            stats.rewritten += 1;
        }
    }
    stats
}

pub(in crate::front::mir) fn rewire_one_result_try_site(
    graph: &mut FunctionGraph,
    site: &ResultTrySite,
    return_owners: Option<&ResultTryReturnOwners>,
) -> Result<(), String> {
    use crate::flowspace::model::Constant;
    use crate::front::option_map_or::emit_narrow;
    use crate::front::option_variant::{emit_sum_variant, map_source};
    use crate::front::result_exc::{
        assert_block_pure_besides, assert_single_pred, back_substitute, collapse_pos0_read,
        diamond_forward_chain, locate_control_flow_question, unreachable_continue_target,
    };

    let name = graph.name.clone();
    let Some(return_owners) = return_owners else {
        return Err(format!("{name}: enclosing function does not return Result"));
    };
    if return_owners.err_ast != site.recv_err_ast {
        return Err(format!(
            "{name}: Result `?` FromResidual would convert {} into {}",
            site.recv_err_ast, return_owners.err_ast
        ));
    }
    if graph.blocks[graph.returnblock.0].inputargs.len() != 1 {
        return Err(format!(
            "{name}: Result-returning function returnblock is not unary"
        ));
    }

    let b = graph
        .blocks
        .iter()
        .position(|block| {
            block
                .operations
                .iter()
                .any(|op| op.result.as_ref() == Some(&site.branch_result_var))
        })
        .ok_or_else(|| format!("{name}: Result branch result var has no producer block"))?;
    let branch_idx = graph.blocks[b]
        .operations
        .iter()
        .position(|op| op.result.as_ref() == Some(&site.branch_result_var))
        .ok_or_else(|| format!("{name}: Result branch op not found in block {b}"))?;
    let res_b = match &graph.blocks[b].operations[branch_idx].kind {
        OpKind::Call {
            target: CallTarget::Method { name: m, .. },
            args,
            ..
        } if m == "branch" && args.len() == 1 => args[0].clone(),
        // Equal Result and ControlFlow layouts are emitted as `same_as`
        // of the Result operand. The diamond still reads that Result.
        OpKind::UnaryOp { op, operand, .. } if op == "same_as" => LinkArg::Value(operand.clone()),
        other => {
            return Err(format!(
                "{name}: Result branch producer is not a one-arg branch method call: {other:?}"
            ));
        }
    };
    assert_single_pred(graph, b, &name)?;
    // `tyref_to_value_type` paints split-eligible `ControlFlow<B, C>` as
    // `Ref(Some(root))`, so `lower_call` recasts the `branch` result to
    // that instantiation. Peel it so the diamond still matches.
    let (cf_forwarded, recast_b) =
        crate::front::result_exc::peel_recast_chain_from(graph, b, &site.branch_result_var);
    let mut recognized_b = vec![branch_idx];
    recognized_b.extend(&recast_b);
    assert_block_pure_besides(graph, b, &recognized_b, "branch", &name)?;

    let (a, res_a) = result_try_predecessor_carrying(graph, b, &res_b, &name)?;
    let question = locate_control_flow_question(graph, b, &cf_forwarded, &name)?;
    let c = question.disc_block;
    let cf_c = question.cf_at_disc.clone();
    if c != b {
        assert_single_pred(graph, c, &name)?;
        let recognized: Vec<usize> = question.disc_idx.into_iter().collect();
        assert_block_pure_besides(graph, c, &recognized, "discriminant", &name)?;
    }
    let continue_link = question.continue_link.clone();
    let break_link = question.break_link.clone();
    let cf_disc_var = question.disc_var.clone();
    let (e_block, residual_var) = result_try_verify_break_arm(graph, &break_link, &cf_c, &name)?;
    // The from_residual block still owns the original successor — the
    // RootScope close lives there, not in the residual call itself.
    // Jumping the rewritten Err arm at returnblock would skip that
    // chain.  Map the successor's args back to the Result producer; a
    // value that cannot be mapped declines the site.
    let (err_sources, err_specs, err_target) = result_try_err_successor(
        graph,
        a,
        &question.path,
        &break_link,
        e_block,
        &residual_var,
        &res_a,
        &name,
    )?;

    enum ContinueArg {
        Const(crate::flowspace::model::Constant),
        Payload,
        Mapped(Variable),
    }
    let mut continue_specs = Vec::new();
    let mut payload_positions = Vec::new();
    let mut ok_sources = Vec::new();
    if !ok_sources.contains(&res_a) {
        ok_sources.push(res_a.clone());
    }
    if let Some(continue_link) = &continue_link {
        continue_specs.reserve(continue_link.args.len());
        for (i, arg) in continue_link.args.iter().enumerate() {
            match arg {
                LinkArg::Const(cst) => continue_specs.push(ContinueArg::Const(cst.clone())),
                LinkArg::Value(v) if *v == cf_c => {
                    continue_specs.push(ContinueArg::Payload);
                    payload_positions.push(i);
                }
                LinkArg::Value(v) if cf_disc_var.as_ref() == Some(v) => {
                    continue_specs.push(ContinueArg::Const(Constant::new(ConstValue::Int(0))));
                }
                LinkArg::Value(v) => {
                    let v_a = back_substitute(
                        graph,
                        &diamond_forward_chain(a, &question.path),
                        v,
                        &name,
                    )?;
                    if !ok_sources.contains(&v_a) {
                        ok_sources.push(v_a.clone());
                    }
                    continue_specs.push(ContinueArg::Mapped(v_a));
                }
            }
        }
        if payload_positions.len() > 1 {
            return Err(format!(
                "{name}: ControlFlow value threaded into {} continue-arm slots",
                payload_positions.len()
            ));
        }
    }

    let (ok_bb, ok_inputs) = if continue_link.is_some() {
        graph.create_block_with_arg_vars(ok_sources.len())
    } else {
        (unreachable_continue_target(graph), Vec::new())
    };
    let (err_bb, err_inputs) = graph.create_block_with_arg_vars(err_sources.len());

    if let Some(continue_link) = &continue_link {
        let res_in_ok = map_source(&ok_sources, &ok_inputs, &res_a)
            .ok_or_else(|| format!("{name}: Result value not threaded into Ok arm"))?;
        let payload = emit_payload_read(
            graph,
            ok_bb,
            res_in_ok,
            &site.ok_owner,
            site.ok_ty.clone(),
            site.ok_is_raw_u8,
        );
        let ok_link_args = continue_specs
            .iter()
            .map(|spec| match spec {
                ContinueArg::Const(cst) => Ok(LinkArg::Const(cst.clone())),
                ContinueArg::Payload => Ok(LinkArg::Value(payload.clone())),
                ContinueArg::Mapped(v_a) => map_source(&ok_sources, &ok_inputs, v_a)
                    .map(LinkArg::Value)
                    .ok_or_else(|| format!("{name}: continue arg not threaded into Ok arm")),
            })
            .collect::<Result<Vec<_>, _>>()?;
        graph.set_control_flow_metadata(
            ok_bb,
            None,
            vec![Link::new_mixed(ok_link_args, continue_link.target, None)],
        );
        for pos in payload_positions {
            collapse_pos0_read(graph, continue_link.target, pos, &name)?;
        }
    }

    let res_in_err = map_source(&err_sources, &err_inputs, &res_a)
        .ok_or_else(|| format!("{name}: Result value not threaded into Err arm"))?;
    let err_payload = emit_payload_read(
        graph,
        err_bb,
        res_in_err,
        &site.err_owner,
        site.err_ty.clone(),
        site.err_is_raw_u8,
    );
    let err_payload = emit_narrow(graph, err_bb, err_payload, &site.err_class);
    let err_shell = emit_sum_variant(
        graph,
        err_bb,
        &return_owners.result_owner,
        "Err",
        1,
        Some((
            &return_owners.err_owner,
            err_payload,
            return_owners.err_ty.clone(),
        )),
    );
    let err_link_args = err_specs
        .iter()
        .map(|spec| match spec {
            ResultTryErrArg::Const(cst) => Ok(LinkArg::Const(cst.clone())),
            ResultTryErrArg::Shell => Ok(LinkArg::Value(err_shell.clone())),
            ResultTryErrArg::Mapped(v_a) => map_source(&err_sources, &err_inputs, v_a)
                .map(LinkArg::Value)
                .ok_or_else(|| format!("{name}: break-arm arg not threaded into Err arm")),
        })
        .collect::<Result<Vec<_>, _>>()?;
    graph.set_control_flow_metadata(
        err_bb,
        None,
        vec![Link::new_mixed(err_link_args, err_target, None)],
    );

    let a_id = graph.blocks[a].id;
    let disc = emit_enum_disc_read(graph, a_id, res_a, &site.result_owner);
    graph.set_control_flow_metadata(
        a_id,
        Some(ExitSwitch::Value(disc)),
        vec![
            Link::new_mixed(
                if continue_link.is_some() {
                    ok_sources.iter().cloned().map(LinkArg::Value).collect()
                } else {
                    Vec::new()
                },
                ok_bb,
                Some(ExitCase::Const(ConstValue::Int(0))),
            ),
            Link::new_mixed(
                err_sources.iter().cloned().map(LinkArg::Value).collect(),
                err_bb,
                Some(ExitCase::Const(ConstValue::Int(1))),
            ),
        ],
    );
    Ok(())
}

pub(in crate::front::mir) fn result_try_predecessor_carrying(
    graph: &FunctionGraph,
    block: usize,
    var_in_block: &Variable,
    name: &str,
) -> Result<(usize, Variable), String> {
    let preds: Vec<usize> = graph
        .blocks
        .iter()
        .enumerate()
        .filter_map(|(i, b)| b.exits.iter().any(|l| l.target.0 == block).then_some(i))
        .collect();
    let [pred] = preds.as_slice() else {
        return Err(format!(
            "{name}: diamond block {block} has {} predecessors, expected 1",
            preds.len()
        ));
    };
    let pos = graph.blocks[block]
        .inputargs
        .iter()
        .position(|v| v == var_in_block)
        .ok_or_else(|| format!("{name}: branch receiver is not a block {block} inputarg"))?;
    let [link] = graph.blocks[*pred].exits.as_slice() else {
        return Err(format!(
            "{name}: Result-producing block {pred} has multiple exits"
        ));
    };
    match link.args.get(pos) {
        Some(LinkArg::Value(v)) => Ok((*pred, v.clone())),
        other => Err(format!(
            "{name}: predecessor arg at position {pos} is {other:?}, expected Value"
        )),
    }
}

pub(in crate::front::mir) fn result_try_verify_break_arm(
    graph: &FunctionGraph,
    break_link: &Link,
    cf_c: &Variable,
    name: &str,
) -> Result<(usize, Variable), String> {
    use crate::front::result_exc::{assert_block_pure_besides, peel_recast_chain_from};
    let pos = break_link
        .args
        .iter()
        .position(|a| matches!(a, LinkArg::Value(v) if v == cf_c))
        .ok_or_else(|| format!("{name}: break arm does not carry the ControlFlow value"))?;
    let e_block = break_link.target.0;
    let cf_e = graph.blocks[e_block]
        .inputargs
        .get(pos)
        .cloned()
        .ok_or_else(|| format!("{name}: break arm target lacks inputarg {pos}"))?;
    let ops = &graph.blocks[e_block].operations;
    let payload = ops.iter().enumerate().find_map(|(i, op)| match &op.kind {
        OpKind::FieldRead { base, field, .. } if *base == cf_e && field.name == "__pos_0" => {
            op.result.clone().map(|r| (i, r))
        }
        _ => None,
    });
    let Some((pos0_idx, payload_var)) = payload else {
        return Err(format!(
            "{name}: break arm block {e_block} lacks the __pos_0 residual read"
        ));
    };
    let residual = ops.iter().enumerate().find_map(|(i, op)| match &op.kind {
        OpKind::Call {
            target: CallTarget::Method { name: m, .. },
            args,
            ..
        } if m == "from_residual"
            && crate::front::result_exc::from_residual_arg_is_payload(ops, args, &payload_var) =>
        {
            op.result.clone().map(|r| (i, r))
        }
        _ => None,
    });
    let Some((from_residual_idx, residual_result)) = residual else {
        return Err(format!(
            "{name}: break arm block {e_block} lacks the from_residual call"
        ));
    };
    let (residual_var, recast_indices) = peel_recast_chain_from(graph, e_block, &residual_result);
    let mut recognized = vec![pos0_idx, from_residual_idx];
    if let OpKind::Call { args, .. } = &ops[from_residual_idx].kind
        && let [LinkArg::Value(arg)] = args.as_slice()
        && let Some(bridge) =
            crate::front::result_exc::payload_bridge_indices(ops, arg, &payload_var, 1)
    {
        recognized.extend(bridge);
    }
    recognized.extend(recast_indices);
    assert_block_pure_besides(graph, e_block, &recognized, "break arm", name)?;
    Ok((e_block, residual_var))
}

pub(in crate::front::mir) enum ResultTryErrArg {
    Const(crate::flowspace::model::Constant),
    Shell,
    Mapped(Variable),
}

/// Map the from_residual block's successor back to the Result producer.
///
/// The rewritten Err arm must land on that successor so a RootScope close
/// (or any other forwarded live value) still runs.  A value produced inside
/// the diamond that is not the residual itself cannot be mapped, and the
/// site stays residual rather than skipping the close.
pub(in crate::front::mir) fn result_try_err_successor(
    graph: &FunctionGraph,
    a: usize,
    path: &[usize],
    break_link: &Link,
    e_block: usize,
    residual_var: &Variable,
    res_a: &Variable,
    name: &str,
) -> Result<(Vec<Variable>, Vec<ResultTryErrArg>, BlockId), String> {
    use crate::front::result_exc::{back_substitute, diamond_forward_chain};
    let [e_exit] = graph.blocks[e_block].exits.as_slice() else {
        return Err(format!(
            "{name}: break arm block {e_block} does not have a single exit"
        ));
    };
    if graph.blocks[e_block].exitswitch.is_some()
        || e_exit.exitcase.is_some()
        || e_exit.last_exception.is_some()
        || e_exit.last_exc_value.is_some()
    {
        return Err(format!(
            "{name}: break arm block {e_block} exit is not a plain goto"
        ));
    }
    let err_target = e_exit.target;
    let mut err_sources = vec![res_a.clone()];
    let mut err_specs = Vec::with_capacity(e_exit.args.len());
    for arg in &e_exit.args {
        match arg {
            LinkArg::Const(cst) => err_specs.push(ResultTryErrArg::Const(cst.clone())),
            LinkArg::Value(v) if v == residual_var => err_specs.push(ResultTryErrArg::Shell),
            LinkArg::Value(v) => {
                let pos = graph.blocks[e_block]
                    .inputargs
                    .iter()
                    .position(|x| x == v)
                    .ok_or_else(|| {
                        format!(
                            "{name}: break-arm successor carries a value produced in \
                             block {e_block} that is not the residual"
                        )
                    })?;
                match break_link.args.get(pos) {
                    Some(LinkArg::Const(cst)) => {
                        err_specs.push(ResultTryErrArg::Const(cst.clone()));
                    }
                    Some(LinkArg::Value(cv)) => {
                        let v_a =
                            back_substitute(graph, &diamond_forward_chain(a, path), cv, name)?;
                        if !err_sources.contains(&v_a) {
                            err_sources.push(v_a.clone());
                        }
                        err_specs.push(ResultTryErrArg::Mapped(v_a));
                    }
                    other => {
                        return Err(format!(
                            "{name}: break-arm inputarg {pos} is {other:?}, expected a value"
                        ));
                    }
                }
            }
        }
    }
    if err_target == graph.returnblock
        && (err_specs.len() != 1 || !matches!(err_specs.first(), Some(ResultTryErrArg::Shell)))
    {
        return Err(format!(
            "{name}: break arm does not forward the residual as the unary return"
        ));
    }
    Ok((err_sources, err_specs, err_target))
}

#[cfg(test)]
mod tests;
