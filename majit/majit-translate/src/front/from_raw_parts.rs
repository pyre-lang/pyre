//! `slice::from_raw_parts{,_mut}(chars, len)` of a length-prefixed GC string
//! header → identity on that header.
//!
//! ## Positioning
//!
//! `core::slice::raw::from_raw_parts` is Opaque in the LLBC.  A handful of
//! adapters already lower it during body lowering by aliasing the destination
//! to the block header — `bytes_block_chars` (address-taken `chars` FieldRead),
//! `BytesArray`/`UnicodeArray::as_slice` (items-view), `RBigInt::digits`
//! (capacity-length typed items).  Those folds key on the *enclosing function
//! name*, so a helper with the same callee shape
//! (`utf8_payload_bytes`: `from_raw_parts(header.add(CHARS_OFFSET), len)`)
//! never reaches them and stays a Phase-A leaf.
//!
//! The translated STR / `BytesBlock` is a length-prefixed GC struct whose
//! `chars` array begins at the descr `base_size`.  A slice over that array is
//! the header itself in the list/string model: `__len` and byte reads dispatch
//! through `arraylen_gc` / `strgetitem`.  Aliasing the slice to the header is
//! therefore the same identity those name-gated folds already emit.
//!
//! Arbitrary `from_raw_parts(p, n)` is not that shape.  `n` may be a logical
//! length below capacity (object-list pop, a buffer exporter's window); dropping
//! it would let a consumer observe capacity.  This pass only fires when the
//! pointer is recovered as:
//! - an address-taken `chars` FieldRead (the `bytes_block_chars` shape), or
//! - a `<*const u8>::add(header, 2*word)` whose length operand is the same
//!   header's `bh_lowlevel_string_len` / `length` field (`utf8_payload_bytes`).
//!
//! Recognition follows SSA copies (`resolve_to_producer_op`) and peels
//! `lltype.cast_ptr_to_int` / `cast_int_to_ptr` so a length read spelled
//! `bh_lowlevel_string_len(header as i64)` still names the header.  When the
//! header dominates the residual but is not live in its block (the add and
//! the `from_raw_parts` sit in successive blocks, and the edge forwards only
//! `(len, chars)`), the rewrite threads the header across that unique
//! predecessor edge.  A merge, a bank mismatch (`value_type_bank`), or any
//! other unproven shape keeps the residual call.
//!
//! Fail-safe: any other residual is left untouched.

use crate::flowspace::model::Variable;
use crate::front::mir::{resolve_to_producer_op, value_type_bank};
use crate::model::{CallTarget, FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueType};

/// Rewrite every residual `from_raw_parts{,_mut}` whose pointer is a
/// length-prefixed GC-string header view into `same_as(header)`.  Returns
/// the number of sites rewritten.
pub(crate) fn rewire_from_raw_parts_sites(graph: &mut FunctionGraph) -> usize {
    let mut sites: Vec<(usize, usize, Variable, ValueType, bool)> = Vec::new();
    for (bi, block) in graph.blocks.iter().enumerate() {
        for (oi, op) in block.operations.iter().enumerate() {
            let Some((header, bytes_block)) = header_for_from_raw_parts(graph, op) else {
                continue;
            };
            let result_ty = match &op.kind {
                OpKind::Call { result_ty, .. } => result_ty.clone(),
                _ => continue,
            };
            let canonical = canonical_source(graph, &header);
            sites.push((bi, oi, canonical, result_ty, bytes_block));
        }
    }
    let mut rewritten = 0usize;
    for (bi, oi, canonical, result_ty, _bytes_block) in sites.into_iter().rev() {
        let Some(live) = ensure_live_rep(graph, bi, &canonical) else {
            continue;
        };
        let Some(live_ty) = var_value_type(graph, &live) else {
            continue;
        };
        if value_type_bank(&live_ty) != value_type_bank(&result_ty) {
            continue;
        }
        let result = graph.blocks[bi].operations[oi].result.clone();
        if let Some(result) = result
            && crate::front::mir::forward_identity(graph, &result, &live)
        {
            rewritten += 1;
            continue;
        }
        // `live` is this block's input. The call result is a different
        // variable the edge already named, so the copy stays.
        graph.blocks[bi].operations[oi].kind = OpKind::UnaryOp {
            op: "same_as".to_string(),
            operand: live,
            result_ty,
        };
        rewritten += 1;
    }
    rewritten
}

fn header_for_from_raw_parts(
    graph: &FunctionGraph,
    op: &SpaceOperation,
) -> Option<(Variable, bool)> {
    let OpKind::Call { target, args, .. } = &op.kind else {
        return None;
    };
    if args.len() != 2 || !is_from_raw_parts_target(target) {
        return None;
    }
    let ptr = args[0].clone().into_variable();
    let len = args[1].clone().into_variable();
    if let Some(header) = chars_field_header(graph, &ptr) {
        let header = peel_ptr_int_casts(graph, &header);
        return len_names_same_header(graph, &len, &header).then_some((header, true));
    }
    let header = ptr_add_chars_offset_header(graph, &ptr)?;
    let header = peel_ptr_int_casts(graph, &header);
    len_names_same_header(graph, &len, &header).then_some((header, false))
}

fn is_from_raw_parts_target(target: &CallTarget) -> bool {
    let CallTarget::FunctionPath { segments, .. } = target else {
        return false;
    };
    matches!(
        segments.last().map(String::as_str),
        Some("from_raw_parts" | "from_raw_parts_mut")
    ) && segments.iter().any(|s| s == "slice" || s == "raw")
}

fn is_ptr_add_target(target: &CallTarget) -> bool {
    let CallTarget::FunctionPath { segments, .. } = target else {
        return false;
    };
    matches!(
        segments.last().map(String::as_str),
        Some("add" | "wrapping_add")
    ) && segments
        .iter()
        .any(|s| s == "const_ptr" || s == "mut_ptr" || s == "ptr")
}

fn is_ptr_int_cast_target(target: &CallTarget) -> bool {
    let CallTarget::FunctionPath { segments, .. } = target else {
        return false;
    };
    matches!(
        segments.last().map(String::as_str),
        Some("cast_ptr_to_int" | "cast_int_to_ptr")
    )
}

fn producer_op<'a>(graph: &'a FunctionGraph, var: &Variable) -> Option<&'a SpaceOperation> {
    let (block_id, idx) = resolve_to_producer_op(graph, var)?;
    graph
        .blocks
        .iter()
        .find(|block| block.id == block_id)?
        .operations
        .get(idx)
}

fn peel_ptr_int_casts(graph: &FunctionGraph, var: &Variable) -> Variable {
    let mut current = var.clone();
    let mut seen: Vec<u64> = Vec::new();
    loop {
        if seen.contains(&current.id()) {
            return current;
        }
        seen.push(current.id());
        let Some(op) = producer_op(graph, &current) else {
            return current;
        };
        match &op.kind {
            OpKind::Call { target, args, .. }
                if args.len() == 1 && is_ptr_int_cast_target(target) =>
            {
                current = args[0].clone().into_variable();
            }
            OpKind::UnaryOp { op, operand, .. } if op == "same_as" => {
                current = operand.clone();
            }
            _ => return current,
        }
    }
}

fn canonical_source(graph: &FunctionGraph, var: &Variable) -> Variable {
    let peeled = peel_ptr_int_casts(graph, var);
    match producer_op(graph, &peeled) {
        Some(op) => match &op.kind {
            OpKind::Call { target, args, .. }
                if args.len() == 1 && is_ptr_int_cast_target(target) =>
            {
                canonical_source(graph, &args[0].clone().into_variable())
            }
            OpKind::UnaryOp { op, operand, .. } if op == "same_as" => {
                canonical_source(graph, operand)
            }
            _ => op.result.clone().unwrap_or(peeled),
        },
        None => peeled,
    }
}

fn chars_field_header(graph: &FunctionGraph, ptr: &Variable) -> Option<Variable> {
    match &producer_op(graph, ptr)?.kind {
        OpKind::FieldRead { base, field, .. }
            if field.name == "chars" && field.taken_by_address =>
        {
            Some(base.clone())
        }
        _ => None,
    }
}

fn is_chars_offset(n: i64) -> bool {
    // `llmemory.offsetof(STR, 'chars')` — hash, then len, then chars.
    // pyre lays that out as `LOWLEVEL_STRING_CHARS_OFFSET` (two words).
    n == (2 * crate::layout::target_word_size()) as i64
}

fn const_int_of(graph: &FunctionGraph, var: &Variable) -> Option<i64> {
    match &producer_op(graph, var)?.kind {
        OpKind::ConstInt(n) => Some(*n),
        OpKind::ConstUInt(n) => i64::try_from(*n).ok(),
        _ => None,
    }
}

fn ptr_add_chars_offset_header(graph: &FunctionGraph, ptr: &Variable) -> Option<Variable> {
    match &producer_op(graph, ptr)?.kind {
        OpKind::Call { target, args, .. } if args.len() == 2 && is_ptr_add_target(target) => {
            let offset = args[1].clone().into_variable();
            is_chars_offset(const_int_of(graph, &offset)?).then(|| args[0].clone().into_variable())
        }
        OpKind::BinOp { op, lhs, rhs, .. } if op == "add" || op == "int_add" => {
            is_chars_offset(const_int_of(graph, rhs)?).then(|| lhs.clone())
        }
        _ => None,
    }
}

fn len_names_same_header(graph: &FunctionGraph, len: &Variable, header: &Variable) -> bool {
    let Some(op) = producer_op(graph, len) else {
        return false;
    };
    let header_src = canonical_source(graph, header);
    match &op.kind {
        OpKind::FieldRead { base, field, .. } if field.name == "length" || field.name == "len" => {
            canonical_source(graph, base) == header_src
        }
        OpKind::Call { target, args, .. } if !args.is_empty() => {
            let leaf = match target {
                CallTarget::FunctionPath { segments, .. } => segments.last().map(String::as_str),
                CallTarget::Method { name, .. } => Some(name.as_str()),
                _ => None,
            };
            let names_len = leaf.is_some_and(|leaf| {
                leaf == "bh_lowlevel_string_len" || leaf.ends_with("lowlevel_string_len")
            });
            names_len && canonical_source(graph, &args[0].clone().into_variable()) == header_src
        }
        _ => false,
    }
}

fn var_value_type(graph: &FunctionGraph, var: &Variable) -> Option<ValueType> {
    match &producer_op(graph, var)?.kind {
        OpKind::Input { ty, .. } => Some(ty.clone()),
        OpKind::Call { result_ty, .. }
        | OpKind::UnaryOp { result_ty, .. }
        | OpKind::FieldRead { ty: result_ty, .. } => Some(result_ty.clone()),
        OpKind::BinOp { result_ty, .. } => Some(result_ty.clone()),
        OpKind::ConstInt(_) => Some(ValueType::Int),
        OpKind::ConstUInt(_) => Some(ValueType::Unsigned),
        OpKind::ConstBool(_) => Some(ValueType::Bool),
        _ => None,
    }
}

fn var_live_in_block(graph: &FunctionGraph, block_idx: usize, var: &Variable) -> bool {
    let block = &graph.blocks[block_idx];
    block.inputargs.iter().any(|input| input == var)
        || block
            .operations
            .iter()
            .any(|op| op.result.as_ref() == Some(var))
}

fn live_rep(graph: &FunctionGraph, block_idx: usize, canonical: &Variable) -> Option<Variable> {
    let block = &graph.blocks[block_idx];
    block
        .inputargs
        .iter()
        .cloned()
        .chain(block.operations.iter().filter_map(|op| op.result.clone()))
        .find(|var| &canonical_source(graph, var) == canonical)
}

fn unique_pred_block(graph: &FunctionGraph, block_idx: usize) -> Option<usize> {
    let target = graph.blocks[block_idx].id;
    let mut found = None;
    for (pi, pred) in graph.blocks.iter().enumerate() {
        if pred.exits.iter().any(|exit| exit.target == target) {
            if found.is_some() {
                return None;
            }
            found = Some(pi);
        }
    }
    found
}

fn ensure_live_rep(
    graph: &mut FunctionGraph,
    block_idx: usize,
    canonical: &Variable,
) -> Option<Variable> {
    ensure_live_rep_visited(graph, block_idx, canonical, &mut Vec::new())
}

fn ensure_live_rep_visited(
    graph: &mut FunctionGraph,
    block_idx: usize,
    canonical: &Variable,
    visited: &mut Vec<usize>,
) -> Option<Variable> {
    if visited.contains(&block_idx) {
        return None;
    }
    if let Some(rep) = live_rep(graph, block_idx, canonical) {
        return Some(rep);
    }
    if var_live_in_block(graph, block_idx, canonical) {
        return Some(canonical.clone());
    }
    visited.push(block_idx);
    let pred = unique_pred_block(graph, block_idx)?;
    let src = ensure_live_rep_visited(graph, pred, canonical, visited)?;
    let new_var = graph.alloc_value_var();
    let target = graph.blocks[block_idx].id;
    graph.blocks[block_idx].inputargs.push(new_var.clone());
    for exit in &mut graph.blocks[pred].exits {
        if exit.target == target {
            exit.args.push(LinkArg::Value(src.clone()));
        }
    }
    Some(new_var)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::FieldDescriptor;

    fn from_raw_parts_target() -> CallTarget {
        CallTarget::FunctionPath {
            segments: ["core", "slice", "raw", "from_raw_parts"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            fun_decl_id: None,
        }
    }

    fn ptr_add_target() -> CallTarget {
        CallTarget::FunctionPath {
            segments: ["core", "ptr", "const_ptr", "<Impl>", "add"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            fun_decl_id: None,
        }
    }

    fn cast_ptr_to_int_target() -> CallTarget {
        CallTarget::FunctionPath {
            segments: [
                "rpython",
                "rtyper",
                "lltypesystem",
                "lltype",
                "cast_ptr_to_int",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
            fun_decl_id: None,
        }
    }

    fn string_len_target() -> CallTarget {
        CallTarget::FunctionPath {
            segments: ["pyre_object", "lowlevel_string", "bh_lowlevel_string_len"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            fun_decl_id: None,
        }
    }

    fn residual_from_raw_parts(g: &FunctionGraph) -> bool {
        g.blocks.iter().flat_map(|b| &b.operations).any(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target, .. } if is_from_raw_parts_target(target)
            )
        })
    }

    fn str_header(g: &mut FunctionGraph, block: crate::model::BlockId) -> Variable {
        g.push_op_var(
            block,
            OpKind::Input {
                name: "value".into(),
                ty: ValueType::Str,
                class_root: None,
            },
            true,
        )
        .unwrap()
    }

    #[test]
    fn chars_field_from_raw_parts_is_bytes_block_str() {
        let mut g = FunctionGraph::new("test_from_raw_parts_chars");
        let a = g.startblock;
        let header = g
            .push_op_var(
                a,
                OpKind::Input {
                    name: "block".into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let chars = g
            .push_op_var(
                a,
                OpKind::FieldRead {
                    base: header.clone(),
                    field: FieldDescriptor {
                        name: "chars".to_string(),
                        owner_root: Some("BytesBlock".to_string()),
                        owner_id: None,
                        base_is_deref: None,
                        taken_by_address: true,
                        inline_vec: false,
                        vec_part: None,
                        owner_declared_gc: None,
                        host_index: None,
                        scalar_word: None,
                    },
                    ty: ValueType::Ref(None),
                    pure: false,
                },
                true,
            )
            .unwrap();
        let len = g
            .push_op_var(
                a,
                OpKind::FieldRead {
                    base: header.clone(),
                    field: FieldDescriptor {
                        name: "length".to_string(),
                        owner_root: Some("BytesBlock".to_string()),
                        owner_id: None,
                        base_is_deref: None,
                        taken_by_address: false,
                        inline_vec: false,
                        vec_part: None,
                        owner_declared_gc: None,
                        host_index: None,
                        scalar_word: None,
                    },
                    ty: ValueType::Unsigned,
                    pure: false,
                },
                true,
            )
            .unwrap();
        let slice = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: from_raw_parts_target(),
                    args: crate::model::call_args(vec![chars, len]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        g.set_return(a, Some(slice.clone()));

        assert_eq!(rewire_from_raw_parts_sites(&mut g), 1);
        assert!(
            !residual_from_raw_parts(&g),
            "header-view from_raw_parts must be gone"
        );
        assert_eq!(g.blocks[a.0].exits[0].args[0].as_variable(), Some(&header));
        let slice_op = g.blocks[a.0]
            .operations
            .iter()
            .find(|op| op.result.as_ref() == Some(&slice));
        if let Some(slice_op) = slice_op {
            match &slice_op.kind {
                OpKind::Call {
                    result_ty: ValueType::Str,
                    ..
                } => panic!("BytesBlock view must not be Str"),
                _ => {}
            }
        }
    }

    #[test]
    fn utf8_payload_add_from_raw_parts_aliases_header() {
        let mut g = FunctionGraph::new("test_from_raw_parts_utf8");
        let a = g.startblock;
        let header = str_header(&mut g, a);
        let chars_off = (2 * crate::layout::target_word_size()) as i64;
        let offset = g.push_op_var(a, OpKind::ConstInt(chars_off), true).unwrap();
        let chars = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: ptr_add_target(),
                    args: crate::model::call_args(vec![header.clone(), offset]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let len = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: string_len_target(),
                    args: crate::model::call_args(vec![header.clone()]),
                    result_ty: ValueType::Unsigned,
                },
                true,
            )
            .unwrap();
        let slice = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: from_raw_parts_target(),
                    args: crate::model::call_args(vec![chars, len]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        g.set_return(a, Some(slice));

        assert_eq!(rewire_from_raw_parts_sites(&mut g), 1);
        assert!(!residual_from_raw_parts(&g));
        assert!(
            g.blocks[a.0]
                .operations
                .iter()
                .all(|op| { !matches!(&op.kind, OpKind::UnaryOp { op, .. } if op == "same_as") })
        );
        assert_eq!(g.blocks[a.0].exits[0].args[0].as_variable(), Some(&header));
    }

    #[test]
    fn pointer_sub_of_chars_offset_stays_residual() {
        let mut g = FunctionGraph::new("test_from_raw_parts_sub");
        let a = g.startblock;
        let header = str_header(&mut g, a);
        let chars_off = (2 * crate::layout::target_word_size()) as i64;
        let offset = g.push_op_var(a, OpKind::ConstInt(chars_off), true).unwrap();
        let chars = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: ["core", "ptr", "const_ptr", "<Impl>", "sub"]
                            .iter()
                            .map(|s| s.to_string())
                            .collect(),
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![header.clone(), offset]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let len = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: string_len_target(),
                    args: crate::model::call_args(vec![header]),
                    result_ty: ValueType::Unsigned,
                },
                true,
            )
            .unwrap();
        let slice = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: from_raw_parts_target(),
                    args: crate::model::call_args(vec![chars, len]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        g.set_return(a, Some(slice));
        assert_eq!(rewire_from_raw_parts_sites(&mut g), 0);
        assert!(residual_from_raw_parts(&g));
    }

    #[test]
    fn host_width_offset_is_not_the_other_targets_chars_offset() {
        let mut g = FunctionGraph::new("test_from_raw_parts_other_width");
        let a = g.startblock;
        let header = str_header(&mut g, a);
        let other = if crate::layout::target_word_size() == 8 {
            8
        } else {
            16
        };
        let offset = g.push_op_var(a, OpKind::ConstInt(other), true).unwrap();
        let chars = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: ptr_add_target(),
                    args: crate::model::call_args(vec![header.clone(), offset]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let len = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: string_len_target(),
                    args: crate::model::call_args(vec![header]),
                    result_ty: ValueType::Unsigned,
                },
                true,
            )
            .unwrap();
        let slice = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: from_raw_parts_target(),
                    args: crate::model::call_args(vec![chars, len]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        g.set_return(a, Some(slice));
        assert_eq!(rewire_from_raw_parts_sites(&mut g), 0);
    }

    #[test]
    fn utf8_payload_ssa_cast_split_aliases_header() {
        // Measured `utf8_payload_bytes` shape: length is
        // `bh_lowlevel_string_len(cast_ptr_to_int(header))`, the add lives in
        // the predecessor, and the `from_raw_parts` block is forwarded only
        // `(len, chars)`.
        let mut g = FunctionGraph::new("test_from_raw_parts_utf8_ssa");
        let a = g.startblock;
        let header = str_header(&mut g, a);

        let (b_len, b_len_args) = g.create_block_with_arg_vars(1);
        let h_len = b_len_args[0].clone();
        g.set_goto(a, b_len, vec![header.clone()]);
        let as_int = g
            .push_op_var(
                b_len,
                OpKind::Call {
                    target: cast_ptr_to_int_target(),
                    args: crate::model::call_args(vec![h_len.clone()]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let len = g
            .push_op_var(
                b_len,
                OpKind::Call {
                    target: string_len_target(),
                    args: crate::model::call_args(vec![as_int]),
                    result_ty: ValueType::Unsigned,
                },
                true,
            )
            .unwrap();

        let (b_add, b_add_args) = g.create_block_with_arg_vars(2);
        let h_add = b_add_args[0].clone();
        let len_add = b_add_args[1].clone();
        g.set_goto(b_len, b_add, vec![h_len, len.clone()]);
        let chars_off = (2 * crate::layout::target_word_size()) as u64;
        let offset = g
            .push_op_var(b_add, OpKind::ConstUInt(chars_off), true)
            .unwrap();
        let chars = g
            .push_op_var(
                b_add,
                OpKind::Call {
                    target: ptr_add_target(),
                    args: crate::model::call_args(vec![h_add, offset]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();

        let (b_frp, b_frp_args) = g.create_block_with_arg_vars(2);
        let len_frp = b_frp_args[0].clone();
        let chars_frp = b_frp_args[1].clone();
        g.set_goto(b_add, b_frp, vec![len_add, chars]);
        let slice = g
            .push_op_var(
                b_frp,
                OpKind::Call {
                    target: from_raw_parts_target(),
                    args: crate::model::call_args(vec![chars_frp, len_frp]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        g.set_return(b_frp, Some(slice));

        assert_eq!(rewire_from_raw_parts_sites(&mut g), 1);
        assert!(
            !residual_from_raw_parts(&g),
            "SSA+cast STR header view must be gone"
        );
        let frp = &g.blocks[b_frp.0];
        assert!(
            frp.operations
                .iter()
                .all(|op| { !matches!(&op.kind, OpKind::UnaryOp { op, .. } if op == "same_as") }),
            "the slice is the threaded header, with no same_as copy"
        );
        assert_eq!(
            frp.inputargs.len(),
            3,
            "header must be threaded onto the unique predecessor edge"
        );
        assert_eq!(frp.exits[0].args[0].as_variable(), Some(&frp.inputargs[2]));
    }

    #[test]
    fn chars_field_with_constant_length_stays_residual() {
        let mut g = FunctionGraph::new("test_from_raw_parts_chars_const_len");
        let a = g.startblock;
        let header = g
            .push_op_var(
                a,
                OpKind::Input {
                    name: "block".into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let chars = g
            .push_op_var(
                a,
                OpKind::FieldRead {
                    base: header,
                    field: FieldDescriptor {
                        name: "chars".to_string(),
                        owner_root: Some("BytesBlock".to_string()),
                        owner_id: None,
                        base_is_deref: None,
                        taken_by_address: true,
                        inline_vec: false,
                        vec_part: None,
                        owner_declared_gc: None,
                        host_index: None,
                        scalar_word: None,
                    },
                    ty: ValueType::Ref(None),
                    pure: false,
                },
                true,
            )
            .unwrap();
        let len = g.push_op_var(a, OpKind::ConstInt(1), true).unwrap();
        let slice = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: from_raw_parts_target(),
                    args: crate::model::call_args(vec![chars, len]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        g.set_return(a, Some(slice));

        assert_eq!(rewire_from_raw_parts_sites(&mut g), 0);
        assert!(
            residual_from_raw_parts(&g),
            "a chars view whose length is not the STR length must stay residual"
        );
    }

    #[test]
    fn word_offset_is_not_the_chars_displacement() {
        // `len` sits one word in; `chars` sits at `2 * word`. Accepting
        // every literal in {8, 16} aliases a length-word address.
        let mut g = FunctionGraph::new("test_from_raw_parts_word_ofs");
        let a = g.startblock;
        let header = str_header(&mut g, a);
        let offset = g
            .push_op_var(
                a,
                OpKind::ConstInt(crate::layout::target_word_size() as i64),
                true,
            )
            .unwrap();
        let chars = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: ptr_add_target(),
                    args: crate::model::call_args(vec![header.clone(), offset]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let len = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: string_len_target(),
                    args: crate::model::call_args(vec![header]),
                    result_ty: ValueType::Unsigned,
                },
                true,
            )
            .unwrap();
        let slice = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: from_raw_parts_target(),
                    args: crate::model::call_args(vec![chars, len]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        g.set_return(a, Some(slice));

        assert_eq!(rewire_from_raw_parts_sites(&mut g), 0);
        assert!(residual_from_raw_parts(&g));
    }

    #[test]
    fn int_bank_header_rep_is_not_aliased() {
        let mut g = FunctionGraph::new("test_from_raw_parts_int_rep");
        let a = g.startblock;
        let header = str_header(&mut g, a);
        let as_int = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: cast_ptr_to_int_target(),
                    args: crate::model::call_args(vec![header.clone()]),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let (b, b_args) = g.create_block_with_arg_vars(1);
        let live_int = b_args[0].clone();
        g.set_goto(a, b, vec![as_int]);
        let offset = g
            .push_op_var(
                b,
                OpKind::ConstInt((2 * crate::layout::target_word_size()) as i64),
                true,
            )
            .unwrap();
        let chars = g
            .push_op_var(
                b,
                OpKind::Call {
                    target: ptr_add_target(),
                    args: crate::model::call_args(vec![live_int.clone(), offset]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let len = g
            .push_op_var(
                b,
                OpKind::Call {
                    target: string_len_target(),
                    args: crate::model::call_args(vec![live_int]),
                    result_ty: ValueType::Unsigned,
                },
                true,
            )
            .unwrap();
        let slice = g
            .push_op_var(
                b,
                OpKind::Call {
                    target: from_raw_parts_target(),
                    args: crate::model::call_args(vec![chars, len]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        g.set_return(b, Some(slice));

        assert_eq!(rewire_from_raw_parts_sites(&mut g), 0);
        assert!(
            residual_from_raw_parts(&g),
            "same_as must not take an int-bank cast of the header"
        );
        assert!(
            !g.blocks
                .iter()
                .flat_map(|block| &block.operations)
                .any(|op| { matches!(&op.kind, OpKind::UnaryOp { op, .. } if op == "same_as") })
        );
    }

    #[test]
    fn arbitrary_from_raw_parts_is_left_residual() {
        let mut g = FunctionGraph::new("test_from_raw_parts_arbitrary");
        let a = g.startblock;
        let ptr = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let len = g.push_op_var(a, OpKind::ConstInt(4), true).unwrap();
        let slice = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: from_raw_parts_target(),
                    args: crate::model::call_args(vec![ptr, len]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        g.set_return(a, Some(slice));

        assert_eq!(rewire_from_raw_parts_sites(&mut g), 0);
        assert!(
            residual_from_raw_parts(&g),
            "an unproven ptr+len view must stay residual"
        );
    }

    #[test]
    fn exporter_window_from_raw_parts_is_left_residual() {
        let mut g = FunctionGraph::new("test_from_raw_parts_exporter");
        let a = g.startblock;
        let buf = g
            .push_op_var(
                a,
                OpKind::Input {
                    name: "self".into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let ptr = g
            .push_op_var(
                a,
                OpKind::FieldRead {
                    base: buf.clone(),
                    field: FieldDescriptor {
                        name: "address".to_string(),
                        owner_root: Some("WritableBuffer".to_string()),
                        owner_id: None,
                        base_is_deref: None,
                        taken_by_address: false,
                        inline_vec: false,
                        vec_part: None,
                        owner_declared_gc: None,
                        host_index: None,
                        scalar_word: None,
                    },
                    ty: ValueType::Ref(None),
                    pure: false,
                },
                true,
            )
            .unwrap();
        let len = g
            .push_op_var(
                a,
                OpKind::FieldRead {
                    base: buf,
                    field: FieldDescriptor {
                        name: "length".to_string(),
                        owner_root: Some("WritableBuffer".to_string()),
                        owner_id: None,
                        base_is_deref: None,
                        taken_by_address: false,
                        inline_vec: false,
                        vec_part: None,
                        owner_declared_gc: None,
                        host_index: None,
                        scalar_word: None,
                    },
                    ty: ValueType::Unsigned,
                    pure: false,
                },
                true,
            )
            .unwrap();
        let slice = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: from_raw_parts_target(),
                    args: crate::model::call_args(vec![ptr, len]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        g.set_return(a, Some(slice));

        assert_eq!(rewire_from_raw_parts_sites(&mut g), 0);
        assert!(
            residual_from_raw_parts(&g),
            "an exporter window is not a GC-string header"
        );
    }

    #[test]
    fn int_header_from_raw_parts_is_left_residual() {
        let mut g = FunctionGraph::new("test_from_raw_parts_int_header");
        let a = g.startblock;
        let header = g.push_op_var(a, OpKind::ConstInt(0), true).unwrap();
        let offset = g.push_op_var(a, OpKind::ConstInt(16), true).unwrap();
        let chars = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: ptr_add_target(),
                    args: crate::model::call_args(vec![header.clone(), offset]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let len = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: string_len_target(),
                    args: crate::model::call_args(vec![header]),
                    result_ty: ValueType::Unsigned,
                },
                true,
            )
            .unwrap();
        let slice = g
            .push_op_var(
                a,
                OpKind::Call {
                    target: from_raw_parts_target(),
                    args: crate::model::call_args(vec![chars, len]),
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        g.set_return(a, Some(slice));

        assert_eq!(rewire_from_raw_parts_sites(&mut g), 0);
        assert!(
            residual_from_raw_parts(&g),
            "an int-bank header must not alias a ref-bank slice"
        );
    }

    /// Loads the real pyre-object LLBC, so ignored by default.
    #[test]
    #[ignore]
    fn utf8_payload_bytes_folds_header_length_view() {
        let path = crate::runtime_names::artifacts::OBJECT_ULLBC;
        let llbc = majit_charon_reader::Llbc::load(path).expect("load pyre-object LLBC");
        let graph = crate::front::mir::lower_function(
            &llbc,
            "pyre_object::unicodeobject::utf8_payload_bytes",
        )
        .expect("lower utf8_payload_bytes");
        assert!(
            !residual_from_raw_parts(&graph),
            "utf8_payload_bytes must alias the STR header; graph: {graph:#?}"
        );
    }
}
