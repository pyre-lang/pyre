//! `ll_copy` (`rlist.py`) of a fixed GcArray into a raw `Vec`.
//!
//! Spine A still names `<[T]>::to_vec`. A `FixedSizeListRepr` self is
//! retargeted to `ll_vec_copy_from_fixed_list_{segment}`, which the JIT
//! inlines. The item copy is `ll_copy_list_to_raw_array` (`rrawarray.py`):
//! `@jit.dont_look_inside`, so the GC-address `raw_memcopy` stays residual.
//! `get_raw_buf` is inlined in that helper (`cast_ptr_to_adr` + `adr_add`
//! of `itemoffsetof`). `ll_list.ll_length()` is `ll_fixed_length` inside
//! the copy graph; the outer `ll_copy` calls it too.
//!
//! `Signed` and `Unsigned` stay distinct graphs (`lltype_to_annotation`)
//! and share the word-sized native leaf `ll_copy_list_to_raw_array_i`.
//! `getarraysize` ignores the item type, so the first `ll_fixed_length`
//! registration wins; the measured site is `GCREF`.

use crate::codewriter::call::{CallControl, effect_graph_from_rtyped_helper};
use crate::flowspace::model::{ConstValue, Variable};
use crate::model::{CallTarget, FunctionGraph, OpKind};
use crate::parse::CallPath;
use crate::translator::rtyper::flowspace_adapter::LegacyToTyped;
use crate::translator::rtyper::lltypesystem::lltype::{GCREF, GcKind, LowLevelType, PtrTarget};
use crate::translator::rtyper::rlist::{
    build_ll_copy_list_to_raw_array_helper_graph, build_ll_fixed_length_helper_graph,
    build_ll_vec_copy_from_fixed_list_into_graph, items_array_ptr_lltype,
};
use crate::translator::rtyper::rrustvec::{
    rust_vec_item_key_segment, rust_vec_item_lltype, vec_item_kind_of,
};
use majit_ir::rvec::VecItemKind;

fn item_lltype(segment: &str) -> LowLevelType {
    match segment {
        "Signed" => LowLevelType::Signed,
        "Unsigned" => LowLevelType::Unsigned,
        "Float" => LowLevelType::Float,
        "GCREF" => (*GCREF).clone(),
        other => panic!("ll_vec_copy_from_fixed_list: unknown segment {other}"),
    }
}

/// Native `HELPER_FNADDRS` leaf. `Signed` and `Unsigned` share `_i`
/// (`VecItemKind::Int` is the word-sized int leaf, as `ll_vec_newlist_hint_i`).
fn native_leaf(segment: &str) -> &'static str {
    match segment {
        "Signed" | "Unsigned" => "ll_copy_list_to_raw_array_i",
        "Float" => "ll_copy_list_to_raw_array_f",
        "GCREF" => "ll_copy_list_to_raw_array_r",
        other => panic!("ll_copy_list_to_raw_array: unknown segment {other}"),
    }
}

fn unwrap_flow(py: crate::flowspace::pygraph::PyGraph) -> crate::flowspace::model::FunctionGraph {
    let crate::flowspace::pygraph::PyGraph { graph, .. } = py;
    std::rc::Rc::try_unwrap(graph)
        .unwrap_or_else(|_| panic!("fresh helper graph is uniquely owned"))
        .into_inner()
}

/// `ll_fixed_length` (`lltypesystem/rlist.py`): `getarraysize` on the
/// `Ptr(GcArray)`. Item type is ignored, so one opname graph is enough.
fn ensure_ll_fixed_length(cc: &mut CallControl, item: &LowLevelType) {
    let path = CallPath::from_segments(["ll_fixed_length"]);
    if cc.has_opname_graph(&path) || cc.jitcode_handle(&path).is_some() {
        return;
    }
    let py = build_ll_fixed_length_helper_graph("ll_fixed_length", items_array_ptr_lltype(item))
        .expect("ll_fixed_length helper graph");
    cc.register_opname_helper_graph(path, unwrap_flow(py));
}

/// `ll_copy_list_to_raw_array` (`rrawarray.py`).
///
/// `GraphFunc._jit_look_inside_ = Some(false)` is already set on the
/// synthesized graph. [`effect_graph_from_rtyped_helper`] pushes
/// `"dont_look_inside"`, and [`CallControl::add_candidate_graph_if_look_inside`]
/// refuses it (`look_inside_graph`). The call is `CallKind::Residual`
/// with the fnaddr [`CallControl::alias_function_fnaddr`] copies from the
/// `#[dont_look_inside_cannot_raise]` leaf
/// (`register_macro_helper_trace_fnaddr`). Not an opname graph: Spine B
/// must not inline the `raw_memcopy`.
fn ensure_copy_residual(cc: &mut CallControl, segment: &str, item: &LowLevelType) {
    let helper_name = format!("ll_copy_list_to_raw_array_{segment}");
    let path = CallPath::from_segments([helper_name.as_str()]);
    if cc.has_function_graph(&path) {
        return;
    }
    let py = build_ll_copy_list_to_raw_array_helper_graph(&helper_name, item)
        .expect("ll_copy_list_to_raw_array helper graph");
    let graph = effect_graph_from_rtyped_helper(&py.graph.borrow());
    cc.register_function_graph(path.clone(), graph);
    cc.add_candidate_graph_if_look_inside(path.clone());
    let src = CallPath::from_segments(["lltypesystem", "rvec", native_leaf(segment)]);
    if cc.lookup_function_fnaddr(&src).is_some() {
        cc.alias_function_fnaddr(path.clone(), &src);
    }
    cc.mark_cannot_raise_assertion(path);
}

/// `ll_copy` (`rlist.py`) of one fixed list. No `_jit_look_inside_`,
/// so the lowered graph stays a regular callee. The first registration
/// of a segment wins: later sites reuse that graph. Argument kinds stay
/// `Ref` then the raw header's `Int`.
fn ensure_outer(
    cc: &mut CallControl,
    segment: &str,
    src_item: &LowLevelType,
    result_ll: &LowLevelType,
) {
    let name = format!("ll_vec_copy_from_fixed_list_{segment}");
    let path = CallPath::from_segments([name.as_str()]);
    if cc.has_opname_graph(&path) || cc.jitcode_handle(&path).is_some() {
        return;
    }
    let py = build_ll_vec_copy_from_fixed_list_into_graph(&name, src_item, result_ll)
        .expect("ll_vec_copy_from_fixed_list helper graph");
    cc.register_opname_helper_graph(path, unwrap_flow(py));
}

/// Register `ll_fixed_length`, then the residual copy, then the inlined
/// outer. Drain is LIFO, so the outer is assembled before `ll_fixed_length`.
pub(crate) fn ensure_fixed_list_to_vec(cc: &mut CallControl, segment: &str) {
    let item = item_lltype(segment);
    let result_ll = crate::translator::rtyper::rrustvec::rust_vec_lltype(&item);
    ensure_ll_fixed_length(cc, &item);
    ensure_copy_residual(cc, segment, &item);
    ensure_outer(cc, segment, &item, &result_ll);
}

/// Same registration when the source list item and the result header
/// differ. The copy residual stays the source segment's graph
/// (`ll_copy_list_to_raw_array_{segment}`): both pointers are one word
/// and the native leaf does not read the header's item type.
fn ensure_fixed_list_to_vec_into(
    cc: &mut CallControl,
    segment: &str,
    src_item: &LowLevelType,
    result_ll: &LowLevelType,
) {
    ensure_ll_fixed_length(cc, src_item);
    ensure_copy_residual(cc, segment, src_item);
    ensure_outer(cc, segment, src_item, result_ll);
}

/// Full lowleveltype. The typed twin wins when it has one; a missing twin
/// or an unset concretetype falls back to the legacy variable.
fn lltype_of(map: &LegacyToTyped, var: &Variable) -> Option<LowLevelType> {
    if let Some(twin) = map.get(var)
        && let Some(ty) = twin.concretetype()
    {
        return Some(ty);
    }
    var.concretetype()
}

/// `FixedSizeListRepr` storage: `Ptr(GcArray(ITEM))` (`Array::gc`), not
/// the raw nolength buffer of a `Vec` header. `None` for any other pointer.
fn fixed_gcarray_item(ty: &LowLevelType) -> Option<&LowLevelType> {
    let LowLevelType::Ptr(ptr) = ty else {
        return None;
    };
    let PtrTarget::Array(array) = &ptr.TO else {
        return None;
    };
    if array._gckind != GcKind::Gc {
        return None;
    }
    if array._hints.get("nolength") == Some(&ConstValue::Bool(true)) {
        return None;
    }
    Some(&array.OF)
}

/// Retarget a one-arg `<[T]>::to_vec` whose self is a fixed GcArray of
/// the result header's item. Any other self keeps `to_vec`.
pub(crate) fn rewrite_fixed_list_to_vec(
    graph: &mut FunctionGraph,
    value_to_var: &LegacyToTyped,
    cc: &mut CallControl,
) {
    let mut hits = Vec::new();
    for block in graph.iter_blocks() {
        for (index, op) in block.operations.iter().enumerate() {
            let OpKind::Call { target, args, .. } = &op.kind else {
                continue;
            };
            let CallTarget::FunctionPath { segments, .. } = target else {
                continue;
            };
            if !crate::translator::rtyper::call_registry::is_alloc_slice_to_vec(segments) {
                continue;
            }
            let runtime = crate::model::call_arg_vars(args);
            if runtime.len() != 1 {
                continue;
            }
            let Some(result) = op.result.as_ref() else {
                continue;
            };
            let Some(header) = lltype_of(value_to_var, result) else {
                continue;
            };
            let Some(header_item) = rust_vec_item_lltype(&header).cloned() else {
                continue;
            };
            let Some(self_ty) = lltype_of(value_to_var, &runtime[0]) else {
                continue;
            };
            let Some(self_item) = fixed_gcarray_item(&self_ty).cloned() else {
                continue;
            };
            // Equal items share a segment. A `GCREF` list copied into the
            // external GC pointer (`OBJECTPTR` for `SomeRustVec::for_kind`)
            // still uses the `GCREF` helper; the outer returns `header`.
            let segment = if self_item == header_item {
                let Some(segment) = rust_vec_item_key_segment(&header_item) else {
                    continue;
                };
                segment
            } else if self_item == *GCREF
                && vec_item_kind_of(&header_item) == Some(VecItemKind::Ref)
            {
                "GCREF"
            } else {
                continue;
            };
            hits.push((
                block.id,
                index,
                segment.to_string(),
                self_item,
                header.clone(),
            ));
        }
    }
    for (block_id, index, segment, self_item, header) in hits {
        if rust_vec_item_lltype(&header) == Some(&self_item) {
            ensure_fixed_list_to_vec(cc, &segment);
        } else {
            ensure_fixed_list_to_vec_into(cc, &segment, &self_item, &header);
        }
        let name = format!("ll_vec_copy_from_fixed_list_{segment}");
        let op = &mut graph.block_mut(block_id).operations[index];
        if let OpKind::Call { target, .. } = &mut op.kind {
            *target = CallTarget::function_path([name.as_str()]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codewriter::call::{CallKind, is_symbolic_fnaddr};
    use crate::codewriter::jtransform::{GraphTransformConfig, Transformer};
    use crate::codewriter::jtransform_opname::lower_graph;
    use crate::model::{CallFuncPtr, SpaceOperation, ValueType};
    use crate::translator::rtyper::rlist::{
        build_ll_vec_copy_from_fixed_list_helper_graph, rust_vec_items_array_ptr_lltype,
    };
    use crate::translator::rtyper::rrustvec::rust_vec_lltype;
    use majit_ir::descr::ExtraEffect;
    use majit_ir::rvec::{
        VEC_LEN_WORD, VEC_PTR_WORD, VecItemKind, VecOp, vec_helper_path, vec_item_malloc_path,
        vec_word_offset,
    };

    fn no_twins() -> LegacyToTyped {
        LegacyToTyped::new()
    }

    fn register_copy_fnaddrs(cc: &mut CallControl) {
        cc.register_macro_helper_trace_fnaddr(
            "majit_rlib::lltypesystem::rvec::ll_copy_list_to_raw_array_i",
            0x5151,
        );
        cc.register_macro_helper_trace_fnaddr(
            "majit_rlib::lltypesystem::rvec::ll_copy_list_to_raw_array_f",
            0x5152,
        );
        cc.register_macro_helper_trace_fnaddr(
            "majit_rlib::lltypesystem::rvec::ll_copy_list_to_raw_array_r",
            0x5153,
        );
    }

    fn to_vec_graph(self_ty: LowLevelType, header: LowLevelType, extra_len: bool) -> FunctionGraph {
        let mut graph = FunctionGraph::new("to_vec_site");
        let self_var = graph.alloc_value_var();
        self_var.set_concretetype(Some(self_ty));
        let mut args = vec![self_var];
        if extra_len {
            let len = graph.alloc_value_var();
            len.set_concretetype(Some(LowLevelType::Signed));
            args.push(len);
        }
        let result = graph.alloc_value_var();
        result.set_concretetype(Some(header));
        graph.push_op_with_result_var(
            graph.startblock,
            OpKind::Call {
                target: CallTarget::function_path(["alloc", "slice", "<Impl>", "to_vec"]),
                args: crate::model::call_args(args),
                result_ty: ValueType::Int,
            },
            result,
        );
        graph
    }

    fn call_leaf(graph: &FunctionGraph) -> String {
        for block in graph.iter_blocks() {
            for op in &block.operations {
                if let OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    ..
                } = &op.kind
                {
                    return segments.join("::");
                }
            }
        }
        panic!("no call");
    }

    fn call_op(name: &str) -> SpaceOperation {
        SpaceOperation {
            result: None,
            kind: OpKind::Call {
                target: CallTarget::function_path([name]),
                args: crate::model::call_args(vec![]),
                result_ty: ValueType::Void,
            },
        }
    }

    #[test]
    fn rewrite_retargets_only_a_fixed_gcarray_of_the_header_item() {
        let mut cc = CallControl::new();
        register_copy_fnaddrs(&mut cc);
        let map = no_twins();

        let mut signed = to_vec_graph(
            items_array_ptr_lltype(&LowLevelType::Signed),
            rust_vec_lltype(&LowLevelType::Signed),
            false,
        );
        super::rewrite_fixed_list_to_vec(&mut signed, &map, &mut cc);
        assert_eq!(call_leaf(&signed), "ll_vec_copy_from_fixed_list_Signed");

        let mut mismatch = to_vec_graph(
            items_array_ptr_lltype(&LowLevelType::Float),
            rust_vec_lltype(&LowLevelType::Signed),
            false,
        );
        super::rewrite_fixed_list_to_vec(&mut mismatch, &map, &mut cc);
        assert_eq!(
            call_leaf(&mismatch),
            "alloc::slice::<Impl>::to_vec",
            "item mismatch stays to_vec"
        );

        let mut raw = to_vec_graph(
            rust_vec_items_array_ptr_lltype(&LowLevelType::Signed),
            rust_vec_lltype(&LowLevelType::Signed),
            false,
        );
        super::rewrite_fixed_list_to_vec(&mut raw, &map, &mut cc);
        assert_eq!(
            call_leaf(&raw),
            "alloc::slice::<Impl>::to_vec",
            "nolength raw self stays to_vec"
        );

        let mut two = to_vec_graph(
            items_array_ptr_lltype(&LowLevelType::Signed),
            rust_vec_lltype(&LowLevelType::Signed),
            true,
        );
        super::rewrite_fixed_list_to_vec(&mut two, &map, &mut cc);
        assert_eq!(
            call_leaf(&two),
            "alloc::slice::<Impl>::to_vec",
            "two runtime args stay to_vec"
        );

        let mut gcref = to_vec_graph(
            items_array_ptr_lltype(&*GCREF),
            rust_vec_lltype(&*GCREF),
            false,
        );
        super::rewrite_fixed_list_to_vec(&mut gcref, &map, &mut cc);
        assert_eq!(call_leaf(&gcref), "ll_vec_copy_from_fixed_list_GCREF");

        // `externalvsinternal` stores the list as `GCREF`. The header item
        // is the external root instance pointer (`OBJECTPTR`).
        let mut external = to_vec_graph(
            items_array_ptr_lltype(&*GCREF),
            rust_vec_lltype(&*crate::translator::rtyper::rclass::OBJECTPTR),
            false,
        );
        super::rewrite_fixed_list_to_vec(&mut external, &map, &mut cc);
        assert_eq!(
            call_leaf(&external),
            "ll_vec_copy_from_fixed_list_GCREF",
            "GCREF list into an external GC-pointer Vec"
        );
    }

    fn assert_residual(cc: &CallControl, name: &str, addr: i64) {
        let path = CallPath::from_segments([name]);
        assert!(cc.has_function_graph(&path), "missing graph {name}");
        assert!(!cc.is_candidate(&path), "{name} must stay residual");
        let graph = cc
            .function_graphs()
            .get(&path)
            .unwrap_or_else(|| panic!("missing helper {name}"));
        assert_eq!(graph.func.jit_look_inside, Some(false));
        assert!(graph.hints.iter().any(|hint| hint == "dont_look_inside"));
        assert_eq!(cc.guess_call_kind(&call_op(name)), CallKind::Residual);
        assert_eq!(cc.lookup_function_fnaddr(&path), Some(addr));
        let binding = cc.fnaddr_binding_for_target(&CallTarget::function_path([name]));
        assert!(!binding.symbolic, "{name}");
        assert!(!is_symbolic_fnaddr(binding.addr));
        assert_eq!(binding.addr, addr);
        assert!(
            binding.path.contains("ll_copy_list_to_raw_array"),
            "{}",
            binding.path
        );
    }

    #[test]
    fn copy_helper_is_residual_and_outer_is_regular() {
        let mut cc = CallControl::new();
        register_copy_fnaddrs(&mut cc);
        super::ensure_fixed_list_to_vec(&mut cc, "Signed");
        super::ensure_fixed_list_to_vec(&mut cc, "Unsigned");
        super::ensure_fixed_list_to_vec(&mut cc, "Float");
        super::ensure_fixed_list_to_vec(&mut cc, "GCREF");

        assert_residual(&cc, "ll_copy_list_to_raw_array_Signed", 0x5151);
        assert_residual(&cc, "ll_copy_list_to_raw_array_Unsigned", 0x5151);
        assert_residual(&cc, "ll_copy_list_to_raw_array_Float", 0x5152);
        assert_residual(&cc, "ll_copy_list_to_raw_array_GCREF", 0x5153);

        let signed = cc
            .function_graphs()
            .get(&CallPath::from_segments([
                "ll_copy_list_to_raw_array_Signed",
            ]))
            .expect("signed copy");
        let unsigned = cc
            .function_graphs()
            .get(&CallPath::from_segments([
                "ll_copy_list_to_raw_array_Unsigned",
            ]))
            .expect("unsigned copy");
        assert!(!std::rc::Rc::ptr_eq(&signed, &unsigned));
        drop(signed);
        drop(unsigned);

        let outer = CallPath::from_segments(["ll_vec_copy_from_fixed_list_Signed"]);
        assert!(cc.is_candidate(&outer));
        assert_eq!(
            cc.guess_call_kind(&call_op("ll_vec_copy_from_fixed_list_Signed")),
            CallKind::Regular
        );
        let unsigned_outer = CallPath::from_segments(["ll_vec_copy_from_fixed_list_Unsigned"]);
        assert!(cc.is_candidate(&unsigned_outer));
        assert_ne!(outer, unsigned_outer);

        let py = build_ll_vec_copy_from_fixed_list_helper_graph(
            "ll_vec_copy_from_fixed_list_Signed",
            &LowLevelType::Signed,
        )
        .expect("signed outer");
        let model = lower_graph(&py.graph.borrow());
        assert_eq!(model.func.jit_look_inside, None);
        assert!(!model.hints.iter().any(|hint| hint == "dont_look_inside"));
        assert!(
            model.iter_block_ops().any(|(_, op)| matches!(
                &op.kind,
                OpKind::FieldRead { field, .. }
                    if field.name == "ptr" && field.owner_root.as_deref() == Some("RustVec")
            )),
            "pre-transform getfield ptr"
        );

        let hint = vec_helper_path(VecOp::NewHint, VecItemKind::Int);
        cc.mark_oopspec(
            CallPath::from_segments(hint.split("::")),
            "newlist_hint(lengthhint)".to_string(),
        );
        let lowered = Transformer::new(&GraphTransformConfig::default())
            .with_callcontrol(&mut cc)
            .transform(&model);
        assert_transformed_outer(&lowered.graph);
    }

    fn const_int_of(graph: &FunctionGraph, var: &Variable) -> Option<i64> {
        for block in graph.iter_blocks() {
            for op in &block.operations {
                if op.result.as_ref() == Some(var)
                    && let OpKind::ConstInt(n) = &op.kind
                {
                    return Some(*n);
                }
            }
        }
        None
    }

    fn def_kind<'a>(graph: &'a FunctionGraph, var: &Variable) -> Option<&'a OpKind> {
        for block in graph.iter_blocks() {
            for op in &block.operations {
                if op.result.as_ref() == Some(var) {
                    return Some(&op.kind);
                }
            }
        }
        None
    }

    fn origin(graph: &FunctionGraph, mut var: Variable) -> Variable {
        for _ in 0..8 {
            match def_kind(graph, &var) {
                Some(OpKind::UnaryOp { op, operand, .. })
                    if crate::codewriter::jtransform::is_same_as_follow_unary(op) =>
                {
                    var = operand.clone();
                }
                _ => break,
            }
        }
        var
    }

    fn assert_transformed_outer(graph: &FunctionGraph) {
        let mut paths = Vec::new();
        let mut saw_ptr_field = false;
        let mut saw_memcopy = false;
        let mut saw_copy_inline = false;
        let mut saw_copy_residual = false;
        for block in graph.iter_blocks() {
            for op in &block.operations {
                match &op.kind {
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } => paths.push(segments.join("::")),
                    OpKind::ConstFnAddr { path, .. } => paths.push(path.clone()),
                    OpKind::FieldRead { field, .. } if field.name == "ptr" => {
                        saw_ptr_field = true;
                    }
                    OpKind::UnaryOp { op, .. } if op == "raw_memcopy" => saw_memcopy = true,
                    OpKind::InlineCall { jitcode, .. }
                        if jitcode.0.name.contains("ll_copy_list_to_raw_array") =>
                    {
                        saw_copy_inline = true;
                    }
                    OpKind::CallResidual {
                        funcptr,
                        descriptor,
                        args_i,
                        args_r,
                        result_kind,
                        ..
                    } => {
                        let (addr, path, symbolic) = match funcptr {
                            CallFuncPtr::Value(fp) => match def_kind(graph, fp) {
                                Some(OpKind::ConstFnAddr {
                                    value,
                                    path,
                                    symbolic,
                                }) => (*value, path.clone(), *symbolic),
                                _ => continue,
                            },
                            CallFuncPtr::Target(target) => (0, target.to_string(), true),
                        };
                        paths.push(path.clone());
                        if !path.contains("ll_copy_list_to_raw_array") {
                            continue;
                        }
                        assert!(!symbolic, "{path}");
                        assert_eq!(addr, 0x5151, "{path}");
                        assert_eq!(*result_kind, 'v');
                        assert_eq!(args_r.len(), 1);
                        assert_eq!(args_i.len(), 1);
                        assert_eq!(descriptor.extra_info.extraeffect, ExtraEffect::CannotRaise);
                        saw_copy_residual = true;
                    }
                    _ => {}
                }
            }
        }
        let dump = paths.join("\n");
        assert!(
            paths.iter().any(|p| p.contains("raw_malloc_varsize_char")),
            "header malloc missing:\n{dump}"
        );
        let item_path = vec_item_malloc_path(VecItemKind::Int, false);
        assert!(
            paths
                .iter()
                .any(|p| p.contains(item_path) || p.contains("vec_item_malloc")),
            "item malloc missing:\n{dump}"
        );
        assert!(!saw_ptr_field, "FieldRead ptr survived");
        assert!(!saw_memcopy, "raw_memcopy must stay inside the residual");
        assert!(!saw_copy_inline, "copy helper must not be an inline_call");
        assert!(saw_copy_residual, "copy residual missing:\n{dump}");

        let word = crate::layout::target_word_size();
        let ptr_off = vec_word_offset(VEC_PTR_WORD, word) as i64;
        let len_off = vec_word_offset(VEC_LEN_WORD, word) as i64;
        let mut saw_ptr_load = false;
        let mut saw_zero = false;
        let mut saw_len = false;
        for block in graph.iter_blocks() {
            for op in &block.operations {
                match &op.kind {
                    OpKind::RawLoad { offset, .. }
                        if const_int_of(graph, offset) == Some(ptr_off) =>
                    {
                        saw_ptr_load = true;
                    }
                    OpKind::RawStore { offset, value, .. }
                        if const_int_of(graph, offset) == Some(len_off) =>
                    {
                        if const_int_of(graph, value) == Some(0) {
                            saw_zero = true;
                        }
                        let src = origin(graph, value.clone());
                        if let Some(OpKind::InlineCall { jitcode, .. }) = def_kind(graph, &src)
                            && jitcode.0.name.contains("ll_fixed_length")
                        {
                            saw_len = true;
                        }
                    }
                    _ => {}
                }
            }
        }
        assert!(
            saw_ptr_load,
            "RawLoad of VEC_PTR_WORD missing, offset={ptr_off}"
        );
        assert!(
            saw_zero && saw_len,
            "len word stores: zero={saw_zero} copy_len={saw_len} offset={len_off}"
        );
    }
}
