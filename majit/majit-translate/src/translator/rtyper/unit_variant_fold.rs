//! Pre-jtransform fold of unit-variant `SyntheticTransparentCtor` calls.
//!
//! RPython parity: `rtyper/rpbc.py::SingleFrozenPBCRepr` resolves a
//! frozen PBC constructor that has no arguments (a unit variant
//! `StepResult::Continue`, `JitAction::Continue`, …) into a singleton
//! `Constant(prebuilt_instance_ptr)` before `codewriter/jtransform`
//! ever sees the call.  See also
//! `rclass.InstanceRepr.get_reusable_prebuilt_instance`.
//!
//! Pyre's frontend (`front::mir`) lowers a unit-variant path
//! expression to `OpKind::Call { target: SyntheticTransparentCtor,
//! args: [] }`.  The companion fold inside
//! `translator/rtyper/flowspace_adapter.rs::legacy_const_define_hlvalue`
//! covers graphs that traverse the Match arm of the dual gate, but
//! graphs registered directly through `register_function_graph` can
//! take the Skip arm and bypass that fold.
//! The residual `Call` op then survives into jtransform and is emitted
//! as a `residual_call_r/d>r` wrapper, which blocks the JitCode walker.
//!
//! This pass operates directly on `model::FunctionGraph` after
//! `lower_indirect_calls` and before `Transformer::transform`, so it
//! catches both gate arms.  The prebuilt instance is
//! `intern_enum_variant_host(...).reusable_prebuilt_instance()` on the
//! interned class in `struct_root_classes` —
//! `InstanceRepr.get_reusable_prebuilt_instance` (rclass.py).  Without
//! a session bookkeeper this pass does not mint (same as jtransform
//! `transform_body`).  The assembler's `emit_const_r` recovers
//! `(qualname, tag)` from the interned variant class host
//! (`unit_variant_const_from_host`).

use std::rc::Rc;

use crate::annotator::bookkeeper::Bookkeeper;
use crate::flowspace::model::{ConstValue, HostObject};
use crate::model::{CallTarget, FunctionGraph, OpKind};

/// Class-dict constant carrying a unit variant's declared discriminant.
///
/// Upstream `ClassDesc.classdict[name] = Constant(value)`
/// (`classdesc.py` `create_new_attribute` / `add_source_attribute`) and
/// `rclass.InstanceRepr.initialize_prebuilt_data` reading a missing
/// instance attr from `classdef.classdesc.read_attribute`.  The host
/// carrier is the same one `stamp_host_virtualizable` uses for
/// `_virtualizable_` (`cls.__dict__` via `HostObject::class_set`).
const UNIT_VARIANT_DISCRIMINANT_ATTR: &str = "__discriminant";

/// Stamp `tag` onto `class` as a class-level constant.  First stamp wins;
/// a later call with the same tag is a no-op.
fn stamp_unit_variant_discriminant(class: &HostObject, tag: i64) {
    match class.class_get(UNIT_VARIANT_DISCRIMINANT_ATTR) {
        None => class.class_set(UNIT_VARIANT_DISCRIMINANT_ATTR, ConstValue::Int(tag)),
        Some(ConstValue::Int(existing)) => {
            debug_assert_eq!(
                existing,
                tag,
                "unit-variant {} carries two tags",
                class.qualname()
            );
        }
        Some(other) => {
            debug_assert!(
                false,
                "unit-variant {} __discriminant is not an int: {other:?}",
                class.qualname()
            );
        }
    }
}

/// Recover `(qualname, tag)` from a prebuilt instance whose class host
/// carries [`UNIT_VARIANT_DISCRIMINANT_ATTR`].  The class host is the
/// interned variant host (`value.__class__`); the tag is the variant's
/// declared discriminant, stamped when the ctor was interned.
pub(crate) fn unit_variant_const_from_host(obj: &HostObject) -> Option<(String, i64)> {
    let class = obj.instance_class()?;
    match class.class_get(UNIT_VARIANT_DISCRIMINANT_ATTR) {
        Some(ConstValue::Int(tag)) => Some((class.qualname().to_string(), tag)),
        _ => None,
    }
}

/// Class host for a unit-variant (or zero-length aggregate) singleton.
///
/// Annotation already interned the variant through
/// [`Bookkeeper::intern_enum_variant_host`].
/// Minting a fresh `HostObject::new_class` here produced a second ClassDesc
/// with no bases, so `InstanceRepr.convert_const` (`rclass.py` —
/// `bk.getuniqueclassdef(value.__class__)`) rejected the prebuilt instance
/// as not an instance of the variant ClassDef.  Reuse the interned host so
/// `value.__class__` and the annotation ClassDef are one identity.
fn intern_unit_variant_class(bk: &Rc<Bookkeeper>, qualname: &str) -> HostObject {
    if let Some((owner, variant)) = qualname.rsplit_once('.') {
        let mut segments: Vec<String> = owner.split('.').map(str::to_string).collect();
        segments.push(variant.to_string());
        if is_synthetic_unit_variant_path(&segments) {
            let enum_root = owner.replace('.', "::");
            return bk.intern_enum_variant_host(&enum_root, variant);
        }
    }
    bk.intern_class_by_qualname(qualname)
}

/// Find-or-mint the prebuilt singleton instance for an allowlisted
/// unit-variant ctor (`StepResult::Continue`, `LoopResult::Done`, …).
/// Returns the same `HostObject` Arc across all calls against the
/// same bookkeeper, mirroring `InstanceRepr.get_reusable_prebuilt_instance`
/// caching on the per-rtyper `instance_reprs` map.  Returns `None`
/// only if `HostObject::Class` cannot produce a prebuilt instance,
/// which by construction never happens for the allowlisted set
/// (every allowlisted path is a unit-variant enum with no fields,
/// so `reusable_prebuilt_instance()` always materialises the
/// `OnceLock` instance — see
/// `majit-translate/src/flowspace/model.rs`).
///
/// The instance is `interned_class.reusable_prebuilt_instance()` —
/// one prebuilt per class, as
/// `InstanceRepr.get_reusable_prebuilt_instance` (rclass.py).  The
/// declared discriminant is a class-dict constant on that host
/// (`ClassDesc.classdict` / `initialize_prebuilt_data`).
pub(crate) fn intern_unit_variant_prebuilt_instance(
    bookkeeper: &Rc<Bookkeeper>,
    qualname: &str,
    tag: Option<i64>,
) -> Option<HostObject> {
    let class_obj = intern_unit_variant_class(bookkeeper, qualname);
    if let Some(tag) = tag {
        stamp_unit_variant_discriminant(&class_obj, tag);
    }
    class_obj.reusable_prebuilt_instance()
}

/// Pyre-side `Class::Variant` unit-variant ctors.  These are valid
/// as bare path-expression values; `flowspace_adapter` pre-folds them
/// to `Hlvalue::Constant(ConstValue::HostObject(prebuilt_instance))`
/// before the rtyper sees a call (mirrors PyPy `rtyper` resolving
/// `SomePBC([InstanceDesc(<unit-variant>)])` to a singleton constant
/// before `jtransform`).  Read by [`fold_unit_variant_ctors`] here and
/// by `flowspace_adapter::is_synthetic_unit_variant_call`.
///
/// An LLBC-derived graph spells the ctor with its full module path and
/// generic instantiation (`resolve_aggregate_adt` pushes the
/// per-instantiation leaf, e.g. `pyre_interpreter::pyopcode::
/// StepResult<*mut PyObject>::Continue`), so the owner segment is
/// compared instantiation-stripped, the same way `result_ctor_kind`
/// strips `Result<T, E>` before comparing.
pub(crate) fn is_synthetic_unit_variant_path(segments: &[String]) -> bool {
    let [head @ .., owner, name] = segments else {
        return false;
    };
    let owner_base = owner.split_once('<').map_or(owner.as_str(), |(b, _)| b);
    match (owner_base, name.as_str()) {
        ("LoopResult", "Done" | "ContinueRunningNormally")
        | ("JitAction", "Return" | "Continue")
        | ("CompareOp", "Lt" | "Le" | "Gt" | "Ge" | "Eq" | "Ne") => head.is_empty(),
        ("StepResult", "Continue") => head.is_empty() || head == ["pyre_interpreter", "pyopcode"],
        _ => false,
    }
}

/// Whether `name` is a shaped positional aggregate carrying no items:
/// `Tuple<>` or `Array<T;0>`.
///
/// The shape suffix is the aggregate's low-level identity, so an empty one is
/// decidable from the name alone. `Tuple` and `Array` without a suffix are the
/// bare roots and are not this: the bare `Tuple` is the unit handled above, and
/// a bare `Array` carries no length to be zero.
pub(crate) fn is_zero_length_shaped_aggregate(name: &str) -> bool {
    is_zero_length_shaped_tuple(name) || is_zero_length_shaped_array(name)
}

/// `Tuple<>` — `rtuple.TUPLE_TYPE` of an empty field list is Void.
fn is_zero_length_shaped_tuple(name: &str) -> bool {
    majit_ir::descr::is_shaped_tuple_name(name)
        && name
            .strip_prefix("Tuple<")
            .and_then(|rest| rest.strip_suffix('>'))
            .is_some_and(|args| args.trim().is_empty())
}

/// `Array<T;0>` — a prebuilt zero-length `GcArray` (`_ll_prebuilt_empty_array`).
pub(crate) fn is_zero_length_shaped_array(name: &str) -> bool {
    majit_ir::descr::is_shaped_array_name(name)
        && name
            .rsplit_once(';')
            .and_then(|(_, len)| len.strip_suffix('>'))
            .is_some_and(|len| len.trim() == "0")
}

/// Rewrite `OpKind::Call { target: SyntheticTransparentCtor, args: [] }`
/// ops whose qualified path matches
/// [`is_synthetic_unit_variant_path`] into
/// `OpKind::ConstRef(prebuilt_instance)`, mirroring
/// `rtyper/rpbc.py::SingleFrozenPBCRepr`.
pub fn fold_unit_variant_ctors(graph: &mut FunctionGraph, bookkeeper: Option<&Rc<Bookkeeper>>) {
    // Unit `()` is `lltype.Void` (`rtuple.TUPLE_TYPE` returns
    // `Void` for an empty field list; `getkind(Void) == 'void'`). Flatten
    // allocates no register for it and `jtransform` drops it from call
    // arguments. Do this before the ref-constant folds so a fieldless
    // struct never becomes a `ConstRefNull` register.
    // A named empty class is still an instance: intern it below to the
    // prebuilt singleton (`InstanceRepr.convert_const`).
    //
    // `erase_zero_sized_ctors` already strips the ctor results it
    // classified. A second pass over every Void-typed call actual
    // (`call.py getcalldescr` still sees those on the flow graph and
    // skips them by concretetype) would drop a Skip-arm `s_None`
    // projection of a declared-Int residual (the one-word guard
    // release) before residual rewrite can stamp `FUNC.ARGS`.
    erase_zero_sized_ctors(graph);
    // Interning a prebuilt instance needs the session bookkeeper
    // (`rtyper.annotator.bookkeeper` / `getuniqueclassdef(value.__class__)`).
    // Without one, skip the ctor-to-constant rewrite rather than minting
    // a throwaway Bookkeeper or a process-global instance cache.
    let Some(bk) = bookkeeper else {
        return;
    };
    let field_bases = collect_field_bases(graph);
    for block in graph.blocks.iter_mut() {
        for op in block.operations.iter_mut() {
            let OpKind::Call {
                target:
                    CallTarget::SyntheticTransparentCtor {
                        name,
                        owner_path,
                        variant_tag,
                        is_struct,
                        ..
                    },
                args,
                ..
            } = &op.kind
            else {
                continue;
            };
            if !args.is_empty() {
                continue;
            }
            // Unit `()` used only as a call operand is erased above
            // (`lltype.Void`, no register). A unit that is returned is a
            // dead shell once the void return drops it; `ConstRefNull` is
            // the pure op `prune_dead_phis` deletes. It is not a live ref.
            if owner_path.is_empty() && name == "Tuple" {
                op.kind = OpKind::ConstRefNull;
                continue;
            }
            // A zero-length shaped aggregate that is read as a call argument.
            // `Tuple<>` is Void (`rtuple.TUPLE_TYPE` of an empty field list).
            // `Array<T;0>` — the empty argument slice `&[]` — is a prebuilt
            // zero-length `GcArray`: `ListRepr.convert_const` /
            // `FixedSizeListRepr.prepare_const` malloc immortal with `n == 0`,
            // and `_ll_prebuilt_empty_array` memos one empty `GcArray` per
            // item type. Intern the instance so `emit_const_r` can pool that
            // prebuilt pointer; `history.CONST_NULL` is only a genuine NULL.
            //
            // Leaving it as a traced allocation is what
            // `positional_shape_metadata` derives with zero rows and no
            // collector-issued type id, which the walker rejects with
            // `UnregisteredNewGcType` after the descent has already run.
            if owner_path.is_empty()
                && is_zero_length_shaped_aggregate(name)
                && let Some(instance) = intern_unit_variant_prebuilt_instance(bk, name, None)
            {
                op.kind = OpKind::ConstRef(instance);
                continue;
            }
            // A fieldless named struct (`PyreEnv`, `KwargsDictStrategy`, a
            // zero-capture closure env) is an instance of a class with no
            // fields. Payloads of a struct ctor go through FieldWrites, so
            // an empty-arg ctor with no field traffic is that empty class.
            // Intern under the crate-relative STRUCT path
            // `synthetic_transparent_ctor_owner` / `rewrite_op_malloc`
            // uses for this ctor (`ClassDesc` / `getuniqueclassdef`
            // identity, `rclass.py` `getinstancerepr`). The ctor leaf
            // alone is not that object: a shared `closure` / `closure#N`
            // leaf is ambiguous across TypeDecls.
            //
            // Every such ctor yields the one prebuilt instance instead of a
            // fresh `malloc` (`rclass.py` `InstanceRepr.new_instance`). A
            // fieldless Rust struct is zero-sized and has no address identity
            // (`Box::new` of it does not allocate, `ptr::eq` between two of them
            // is unspecified), so no caller can tell the instances apart. PyPy
            // reaches the same shape for its stateless strategy objects through
            // `space.fromcache(Strategy)`.
            if *is_struct {
                let fieldless = op
                    .result
                    .as_ref()
                    .is_some_and(|result| !field_bases.iter().any(|base| base == result));
                if fieldless {
                    let name_path = if owner_path.is_empty() {
                        name.clone()
                    } else {
                        format!("{}::{name}", owner_path.join("::"))
                    };
                    let intern_key = crate::model::crate_relative_struct_owner(
                        &name_path,
                        graph.name.split("::").next(),
                    );
                    if let Some(instance) =
                        intern_unit_variant_prebuilt_instance(bk, &intern_key, None)
                    {
                        op.kind = OpKind::ConstRef(instance);
                        continue;
                    }
                }
            }
            let mut segments = owner_path.clone();
            segments.push(name.clone());
            if !is_synthetic_unit_variant_path(&segments) {
                continue;
            }
            let qualname = segments.join(".");
            let Some(instance) = intern_unit_variant_prebuilt_instance(bk, &qualname, *variant_tag)
            else {
                continue;
            };
            op.kind = OpKind::ConstRef(instance);
        }
    }
}

/// Drop a unit `()` constructor from the graph.
///
/// `rtuple.py` `TUPLE_TYPE` returns `lltype.Void` for an empty field list,
/// and `jtransform.py` `add_in_correct_list` skips `kind == 'void'`. A
/// zero-arg `Tuple` used only as a call operand is that case: the value
/// gets `concretetype = Void`, the constructor op goes away, and every
/// call argument list loses it. A named empty class is an instance, not
/// Void; those constructors intern to a prebuilt constant instead.
fn erase_zero_sized_ctors(graph: &mut FunctionGraph) {
    use crate::model::{ConcreteType, FunctionGraph};

    let mut void_vars = Vec::new();
    for block in &graph.blocks {
        for op in &block.operations {
            let OpKind::Call {
                target:
                    CallTarget::SyntheticTransparentCtor {
                        name, owner_path, ..
                    },
                args,
                ..
            } = &op.kind
            else {
                continue;
            };
            if !args.is_empty() {
                continue;
            }
            let Some(result) = op.result.clone() else {
                continue;
            };
            if !(owner_path.is_empty() && name == "Tuple") {
                continue;
            }
            if !zst_uses_are_call_args(graph, &result) {
                continue;
            }
            void_vars.push(result);
        }
    }
    if void_vars.is_empty() {
        return;
    }
    for var in &void_vars {
        FunctionGraph::set_concretetype_of_inline(var, ConcreteType::Void);
    }
    let is_void = |var: &crate::flowspace::model::Variable| void_vars.iter().any(|z| z == var);
    for block in &mut graph.blocks {
        block.operations.retain(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::SyntheticTransparentCtor { .. },
                args,
                ..
            } if args.is_empty() => match op.result.as_ref() {
                Some(result) => !is_void(result),
                None => true,
            },
            _ => true,
        });
        for op in &mut block.operations {
            strip_void_call_operands(&mut op.kind, &is_void);
        }
    }
}

/// `true` when every use of `var` is a call operand.
///
/// A returned unit, a phi, or a field store still names the value, so
/// the constructor stays. Call operands are dropped with the value.
fn zst_uses_are_call_args(graph: &FunctionGraph, var: &crate::flowspace::model::Variable) -> bool {
    use crate::inline::op_variable_refs;
    for block in &graph.blocks {
        if block.inputargs.iter().any(|arg| arg == var) {
            return false;
        }
        for op in &block.operations {
            if op.result.as_ref() == Some(var) {
                continue;
            }
            if !op_variable_refs(&op.kind).iter().any(|used| used == var) {
                continue;
            }
            if !op_is_call(&op.kind) {
                return false;
            }
        }
        for link in &block.exits {
            if link.args.iter().any(|arg| arg.as_variable() == Some(var)) {
                return false;
            }
        }
    }
    true
}

fn op_is_call(kind: &OpKind) -> bool {
    matches!(kind, OpKind::Call { .. } | OpKind::IndirectCall { .. })
}

fn call_keeps_void_receiver(kind: &OpKind) -> bool {
    // `try_handle_jit_marker` treats `args[0]` as the driver and reads
    // greens from the rest. Dropping the void receiver here makes `pc`
    // look like the driver.
    let leaf = match kind {
        OpKind::Call {
            target: CallTarget::Method { name, .. },
            ..
        } => name.as_str(),
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            ..
        } => match segments.last() {
            Some(leaf) => leaf.as_str(),
            None => return false,
        },
        _ => return false,
    };
    matches!(leaf, "jit_merge_point" | "can_enter_jit" | "loop_header")
}

fn strip_void_call_operands(
    kind: &mut OpKind,
    is_void: &impl Fn(&crate::flowspace::model::Variable) -> bool,
) {
    if call_keeps_void_receiver(kind) {
        return;
    }
    match kind {
        OpKind::Call { args, .. } => {
            args.retain(|arg| arg.as_variable().map(|var| !is_void(var)).unwrap_or(true));
        }
        OpKind::IndirectCall { args, .. } => {
            args.retain(|var| !is_void(var));
        }
        _ => {}
    }
}

fn collect_field_bases(graph: &FunctionGraph) -> Vec<crate::flowspace::model::Variable> {
    let mut field_bases = Vec::new();
    for block in &graph.blocks {
        for op in &block.operations {
            match &op.kind {
                OpKind::FieldRead { base, .. }
                | OpKind::FieldWrite { base, .. }
                | OpKind::InteriorFieldRead { base, .. }
                | OpKind::InteriorFieldWrite { base, .. }
                | OpKind::VableFieldRead { base, .. }
                | OpKind::VableFieldWrite { base, .. } => field_bases.push(base.clone()),
                _ => {}
            }
        }
    }
    field_bases
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_length_shapes_are_recognised() {
        assert!(is_zero_length_shaped_aggregate("Tuple<>"));
        assert!(is_zero_length_shaped_aggregate("Array<*mut PyObject;0>"));
        assert!(is_zero_length_shaped_aggregate("Array<u8;0>"));
    }

    /// The bare roots are not shaped, and a shape that carries items is not
    /// empty — both keep the allocation.
    #[test]
    fn bare_roots_and_non_empty_shapes_are_rejected() {
        assert!(!is_zero_length_shaped_aggregate("Tuple"));
        assert!(!is_zero_length_shaped_aggregate("Array"));
        assert!(!is_zero_length_shaped_aggregate("Tuple<*mut PyObject>"));
        assert!(!is_zero_length_shaped_aggregate("Array<*mut PyObject;1>"));
        assert!(!is_zero_length_shaped_aggregate("Array<u8;10>"));
    }

    /// A length that merely ENDS in `0` is not length zero.
    #[test]
    fn a_length_ending_in_zero_is_not_zero() {
        assert!(!is_zero_length_shaped_aggregate("Array<u8;20>"));
        assert!(!is_zero_length_shaped_aggregate("Array<u8;100>"));
    }

    /// The qualified, generic-instantiated spelling an LLBC graph carries
    /// must match alongside the bare two-segment form; unrelated paths
    /// and other variants of the same enum must not.
    #[test]
    fn qualified_instantiated_unit_variant_paths_match() {
        let q = |s: &[&str]| s.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert!(is_synthetic_unit_variant_path(&q(&[
            "StepResult",
            "Continue"
        ])));
        assert!(is_synthetic_unit_variant_path(&q(&[
            "pyre_interpreter",
            "pyopcode",
            "StepResult<*mut PyObject>",
            "Continue",
        ])));
        assert!(!is_synthetic_unit_variant_path(&q(&[
            "pyre_interpreter",
            "pyopcode",
            "StepResult<*mut PyObject>",
            "Yield",
        ])));
        assert!(!is_synthetic_unit_variant_path(&q(&[
            "other_crate",
            "StepResult",
            "Continue",
        ])));
        assert!(!is_synthetic_unit_variant_path(&q(&[
            "JitAction",
            "Continue",
            "Extra"
        ])));
        assert!(!is_synthetic_unit_variant_path(&q(&["Continue"])));
    }

    /// The assembler recovers `(qualname, tag)` from the interned class
    /// host's `__discriminant` classdict constant.
    #[test]
    fn bookkeeper_prebuilt_carries_discriminant_on_class() {
        let bk = Rc::new(Bookkeeper::new());
        let instance =
            intern_unit_variant_prebuilt_instance(&bk, "JitAction.Return", Some(1)).unwrap();
        let (qualname, tag) = unit_variant_const_from_host(&instance)
            .expect("class host carries the declared discriminant");
        assert_eq!(tag, 1);
        let class = bk.intern_enum_variant_host("JitAction", "Return");
        assert_eq!(qualname, class.qualname());
        assert_eq!(
            instance.instance_class(),
            Some(&class),
            "instance class is the interned variant host"
        );
    }

    /// One prebuilt instance per shape, shared across graphs, as
    /// `InstanceRepr.get_reusable_prebuilt_instance` caches per classdef.
    #[test]
    fn one_prebuilt_instance_per_shape() {
        let bk = Rc::new(Bookkeeper::new());
        let a = intern_unit_variant_prebuilt_instance(&bk, "Array<*mut PyObject;0>", None).unwrap();
        let b = intern_unit_variant_prebuilt_instance(&bk, "Array<*mut PyObject;0>", None).unwrap();
        let other = intern_unit_variant_prebuilt_instance(&bk, "Array<u8;0>", None).unwrap();
        assert_eq!(a.identity_id(), b.identity_id());
        assert_ne!(a.identity_id(), other.identity_id());
    }

    /// A fieldless named struct built in the prologue, live across a merge
    /// point, and passed to a method is `Constant(prebuilt instance)`
    /// (`InstanceRepr.convert_const` / `get_reusable_prebuilt_instance`).
    #[test]
    fn zst_across_merge_point_is_prebuilt_instance() {
        use crate::model::{FunctionGraph, ValueType, call_args};

        let mut graph = FunctionGraph::new("zst_merge");
        let entry = graph.startblock;
        let zst = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_struct_ctor(
                        vec!["grain".into(), "vm".into(), "jit".into()],
                        "Marker",
                    ),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(Some("Marker".into())),
                },
                true,
            )
            .unwrap();
        let tag = graph.push_op_var(entry, OpKind::ConstInt(7), true).unwrap();
        FunctionGraph::set_concretetype_of_inline(&tag, crate::model::ConcreteType::Signed);
        graph.push_op_var(entry, OpKind::Live, false);
        graph.push_op_var(
            entry,
            OpKind::JitMergePoint {
                jitdriver_index: 0,
                greens_i: vec![tag.clone()],
                greens_r: Vec::new(),
                greens_f: Vec::new(),
                reds_i: Vec::new(),
                reds_r: Vec::new(),
                reds_f: Vec::new(),
                green_classes: String::new(),
                red_classes: String::new(),
            },
            false,
        );
        graph.push_op_var(
            entry,
            OpKind::Call {
                target: CallTarget::Method {
                    name: "dispatch_cold".into(),
                    receiver_root: None,
                    resolved_path: None,
                    fun_decl_id: None,
                    branch_payloads: None,
                },
                args: call_args(vec![zst.clone(), tag.clone()]),
                result_ty: ValueType::Void,
            },
            false,
        );
        graph.set_goto(entry, entry, Vec::new());

        let bk = Rc::new(Bookkeeper::new());
        fold_unit_variant_ctors(&mut graph, Some(&bk));
        assert_ne!(
            FunctionGraph::concretetype_of(&zst),
            crate::model::ConcreteType::Void,
            "the named empty class is an instance, not Void"
        );
        let mut saw_ctor = false;
        let mut saw_prebuilt = false;
        let mut call_passes_zst = false;
        for block in &graph.blocks {
            for op in &block.operations {
                match &op.kind {
                    OpKind::Call {
                        target: CallTarget::SyntheticTransparentCtor { .. },
                        ..
                    } => saw_ctor = true,
                    OpKind::ConstRef(instance) if op.result.as_ref() == Some(&zst) => {
                        saw_prebuilt = true;
                        let class = instance
                            .instance_class()
                            .expect("prebuilt instance has a class");
                        let leaf = class
                            .qualname()
                            .rsplit(['.', ':'])
                            .next()
                            .unwrap_or(class.qualname());
                        assert_eq!(leaf, "Marker", "intern leaf matches Input class_root");
                    }
                    OpKind::Call {
                        target: CallTarget::Method { name, .. },
                        args,
                        ..
                    } if name == "dispatch_cold" => {
                        call_passes_zst = args.iter().any(|arg| arg.as_variable() == Some(&zst));
                    }
                    _ => {}
                }
            }
        }
        assert!(!saw_ctor, "fieldless ctor survived");
        assert!(
            saw_prebuilt,
            "named empty class interned as prebuilt instance"
        );
        assert!(call_passes_zst, "the call must pass the instance");
    }

    /// Two zero-capture closures sharing the leaf intern as distinct
    /// ClassDescs (`getuniqueclassdef` identity, `rclass.py`
    /// `getinstancerepr`). The malloc owner is the crate-relative
    /// STRUCT path, not the ambiguous `closure` leaf.
    #[test]
    fn fieldless_closures_intern_by_owner_path() {
        use crate::model::{FunctionGraph, ValueType};

        let mut graph = FunctionGraph::new("pyre_interpreter::mod::f");
        let entry = graph.startblock;
        let a = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_struct_ctor(
                        vec!["pyre_interpreter".into(), "mod".into(), "f".into()],
                        "closure",
                    ),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(Some("pyre_interpreter::mod::f::closure".into())),
                },
                true,
            )
            .unwrap();
        let b = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_struct_ctor(
                        vec!["pyre_interpreter".into(), "other".into(), "g".into()],
                        "closure",
                    ),
                    args: Vec::new(),
                    result_ty: ValueType::Ref(Some("pyre_interpreter::other::g::closure".into())),
                },
                true,
            )
            .unwrap();
        graph.set_goto(entry, entry, Vec::new());

        let bk = Rc::new(Bookkeeper::new());
        fold_unit_variant_ctors(&mut graph, Some(&bk));
        let class_of = |var: &crate::flowspace::model::Variable| -> String {
            for block in &graph.blocks {
                for op in &block.operations {
                    if let OpKind::ConstRef(instance) = &op.kind
                        && op.result.as_ref() == Some(var)
                    {
                        return instance
                            .instance_class()
                            .expect("prebuilt instance has a class")
                            .qualname()
                            .to_string();
                    }
                }
            }
            panic!("expected ConstRef for {var:?}");
        };
        let qa = class_of(&a);
        let qb = class_of(&b);
        assert_eq!(qa, "mod::f::closure");
        assert_eq!(qb, "other::g::closure");
        assert_ne!(qa, qb);
    }

    /// Skip-arm `legacy_resolve` can leave a declared-Int residual actual
    /// as Void. `call.py getcalldescr` still sees that Variable on the
    /// flow graph; dropping it here makes residual rewrite pass `[]`.
    #[test]
    fn fold_keeps_a_void_actual_of_a_non_zst_residual() {
        use crate::model::{ConcreteType, FunctionGraph, SpaceOperation, ValueType, call_args};

        let mut graph = FunctionGraph::new("release");
        let depth = graph.alloc_value_var();
        FunctionGraph::set_concretetype_of_inline(&depth, ConcreteType::Void);
        graph
            .block_mut(graph.startblock)
            .operations
            .push(SpaceOperation {
                result: None,
                kind: OpKind::Call {
                    target: CallTarget::function_path([
                        "pyre_interpreter",
                        "eval",
                        "frame_anchor_release",
                    ]),
                    args: call_args(vec![depth.clone()]),
                    result_ty: ValueType::Void,
                },
            });
        fold_unit_variant_ctors(&mut graph, None);
        let OpKind::Call { args, .. } = &graph.block(graph.startblock).operations[0].kind else {
            panic!("expected the residual Call");
        };
        assert_eq!(
            args.len(),
            1,
            "getcalldescr still sees the Void actual so residual rewrite can stamp FUNC.ARGS"
        );
        assert_eq!(args[0].as_variable(), Some(&depth));
    }
}
