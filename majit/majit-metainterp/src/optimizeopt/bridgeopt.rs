//! Serializer / deserializer for bridge-side optimizer knowledge.
//!
//! Ports `rpython/jit/metainterp/optimizeopt/bridgeopt.py`:
//!
//! * `serialize_optimizer_knowledge` (bridgeopt.py) writes the
//!   known-class bitfield + heap field/array triples + loopinvariant
//!   call-result tuples onto a guard's `rd_numb` stream when finishing
//!   resume data.
//! * `deserialize_optimizer_knowledge` (bridgeopt.py) reads those
//!   sections back at bridge-compile time and applies the facts directly
//!   onto the bridge optimizer (`Optimizer::make_constant_class`,
//!   `import_heap_knowledge`, `import_loopinvariant_knowledge`). RPython
//!   has no separate "BridgeKnowledge" struct or per-guard pass — facts
//!   are written into the standard optimizer state and consumed by the
//!   existing OptIntBounds / OptHeap / OptVirtualize passes.
//!
//! `decode_box` lives with the resume tag helpers, but is re-exported here
//! because the upstream import path is `optimizeopt.bridgeopt.decode_box`.
//! `decoded_box_to_opref` is a small helper for folding a typed
//! `Const{Int,Float,Ptr}` from that decoded form back into the optimizer's
//! constant pool.

use majit_ir::OpRef;

use crate::optimizeopt::OptContext;
pub use crate::resume::decode_box;

/// bridgeopt.py tag_box.
pub fn tag_box(
    opref: OpRef,
    liveboxes_from_env: &crate::resume::LiveboxMap,
    memo: &mut crate::resume::ResumeDataLoopMemo,
    env: &dyn majit_ir::BoxEnv,
    new_liveboxes: &crate::resume::LiveboxMap,
) -> Result<i16, crate::resume::TagOverflow> {
    memo._gettagged(opref, env, liveboxes_from_env, new_liveboxes)
}

/// bridgeopt.py deserialize_optimizer_knowledge.
///
/// Read optimizer knowledge from the guard's rd_numb and apply it
/// directly to the optimizer passes. RPython parity: the function
/// takes the optimizer and applies knowledge inline, never returning
/// an intermediate struct.
/// bridgeopt.py signature:
/// deserialize_optimizer_knowledge(optimizer, resumestorage, frontend_boxes, liveboxes)
///
/// bridgeopt.py `serialize_optimizer_knowledge(optimizer,
/// numb_state, liveboxes, liveboxes_from_env, memo)`.
///
/// Emits three serialized sections on every guard (RPython emits zeros
/// when the optheap/optrewrite caches are empty; the deserializer relies
/// on the sections always being present):
///
/// 1. known-class bitfield per Ref livebox (bridgeopt.py)
/// 2. heap field + array item triples (bridgeopt.py)
/// 3. loopinvariant call results (bridgeopt.py)
///
/// RPython splits the memo-side wrapper (`_add_optimizer_sections`,
/// resume.py) from the serialize core (`serialize_optimizer_knowledge`,
/// bridgeopt.py). pyre keeps the same split: this free function
/// carries the core, and `ResumeDataLoopMemo::_add_optimizer_sections`
/// forwards.
pub fn serialize_optimizer_knowledge(
    memo: &mut crate::resume::ResumeDataLoopMemo,
    numb_state: &mut crate::resume::NumberingState,
    liveboxes: &[Option<majit_ir::operand::Operand>],
    new_liveboxes: &crate::resume::LiveboxMap,
    env: &dyn majit_ir::BoxEnv,
    optimizer_knowledge: Option<&crate::resume::OptimizerKnowledgeForResume>,
) -> Result<(), crate::resume::TagOverflow> {
    // bridgeopt.py `available_boxes = {}` followed by
    // `available_boxes[box] = None`. Upstream's membership dict maps to
    // LiveboxMap (identity-keyed: linear below 16, hash-indexed above).
    let mut available_boxes = crate::resume::LiveboxMap::new();
    for b in liveboxes.iter().flatten() {
        // bridgeopt.py serialize_optimizer_knowledge tests the numbered box
        // itself for membership in liveboxes_from_env.
        if numb_state.liveboxes.contains_key(b) {
            available_boxes.insert(b.clone(), 0);
        }
    }

    // `serialize_optimizer_knowledge` records a known-class bit for each Ref
    // livebox by calling `getptrinfo(box).get_known_class(cpu)`.
    // The actual class pointer is recovered at deserialization time
    // via cpu.cls_of_box(frontend_boxes[i]).
    //
    // Read the numbered object's intrinsic type, just as fail_arg_types does.
    // Resolving a position again could classify a different box and make
    // serialization and deserialization disagree on the class-bit positions.
    //
    // bridgeopt.py `serialize_optimizer_knowledge`:
    //     for box in liveboxes:
    //         if box is None or box.type != "r":
    //             continue
    // `initialize_state_from_guard_failure` later removes the same None
    // entries before `deserialize_optimizer_knowledge` sees `liveboxes`, so
    // neither side assigns a class-bit position to a resume hole.
    let mut bitfield: i32 = 0;
    let mut shifts = 0;
    for slot in liveboxes.iter() {
        let Some(b) = slot else {
            continue;
        };
        let livebox_tp = b.type_();
        if livebox_tp != majit_ir::Type::Ref {
            continue;
        }
        bitfield <<= 1;
        // `bridgeopt.serialize_optimizer_knowledge` obtains `info` with
        // `getptrinfo(box)` and records whether it has a known class.
        // known_class = info is not None and info.get_known_class(cpu) is not None
        if env.has_known_class(b) {
            bitfield |= 1;
        }
        shifts += 1;
        if shifts == 6 {
            numb_state.append_int(bitfield as i64);
            bitfield = 0;
            shifts = 0;
        }
    }
    if shifts > 0 {
        numb_state.append_int((bitfield << (6 - shifts)) as i64);
    }

    // Serialize heap and loop-invariant knowledge after the class bitfield.
    let Some(knowledge) = optimizer_knowledge else {
        // No optimizer knowledge means three empty sections.
        numb_state.append_int(0); // struct fields count
        numb_state.append_int(0); // array items count
        numb_state.append_int(0); // loopinvariant count
        return Ok(());
    };
    // bridgeopt.py:93: triples_struct = optimizer.optheap.serialize_optheap(available_boxes)
    let is_const =
        |opref: OpRef| opref.is_constant() || env.is_const(&env.get_box_replacement_operand(opref));
    let field_ok = |obj: OpRef, val: OpRef| {
        (is_const(obj) || available_boxes.contains_key(&env.get_box_replacement_operand(obj)))
            && (is_const(val)
                || available_boxes.contains_key(&env.get_box_replacement_operand(val)))
    };
    let triples_struct: Vec<(OpRef, i32, OpRef)> = knowledge
        .heap_fields
        .iter()
        .copied()
        .filter(|&(obj, _, val)| field_ok(obj, val))
        .collect();
    numb_state.append_int(triples_struct.len() as i64);
    for &(obj, descr_idx, val) in &triples_struct {
        let obj_tag = tag_box(obj, &numb_state.liveboxes, memo, env, new_liveboxes)?;
        numb_state.writer.append_short(obj_tag as i32);
        numb_state.append_int(descr_idx as i64);
        let val_tag = tag_box(val, &numb_state.liveboxes, memo, env, new_liveboxes)?;
        numb_state.writer.append_short(val_tag as i32);
    }
    // bridgeopt.py:102-108: array items
    let triples_array: Vec<(OpRef, i64, i32, OpRef)> = knowledge
        .heap_arrayitems
        .iter()
        .copied()
        .filter(|&(obj, _, _, val)| field_ok(obj, val))
        .collect();
    numb_state.append_int(triples_array.len() as i64);
    for &(obj, index, descr_idx, val) in &triples_array {
        let obj_tag = tag_box(obj, &numb_state.liveboxes, memo, env, new_liveboxes)?;
        numb_state.writer.append_short(obj_tag as i32);
        // bridgeopt.py:106 numb_state.append_int(index) — pass the original
        // index unchanged; resumecode.py enforces SHORT range on the
        // i64 value, panicking instead of silently wrapping a too-large
        // index into an i32.
        numb_state.append_int(index);
        numb_state.append_int(descr_idx as i64);
        let val_tag = tag_box(val, &numb_state.liveboxes, memo, env, new_liveboxes)?;
        numb_state.writer.append_short(val_tag as i32);
    }

    // bridgeopt.py:113-122: loopinvariant results
    let tuples_loopinvariant: Vec<(i64, OpRef)> = knowledge
        .loopinvariant_results
        .iter()
        .copied()
        .filter(|&(_, result)| {
            is_const(result)
                || available_boxes.contains_key(&env.get_box_replacement_operand(result))
        })
        .collect();
    numb_state.append_int(tuples_loopinvariant.len() as i64);
    for &(const_ptr, result) in &tuples_loopinvariant {
        let const_tag = memo.getconst_int(const_ptr)?;
        numb_state.writer.append_short(const_tag as i32);
        let result_tag = tag_box(result, &numb_state.liveboxes, memo, env, new_liveboxes)?;
        numb_state.writer.append_short(result_tag as i32);
    }
    Ok(())
}

/// `frontend_boxes`: runtime values from guard failure (RPython Box objects
///   with concrete references). Used by `cpu.cls_of_box` to read vtable.
/// `cpu`: `optimizer.cpu` (model.py `AbstractCPU`).  Dispatches
///   `cpu.cls_of_box(frontend_boxes[i])` for bridgeopt.py
///   `make_constant_class`.
#[expect(
    clippy::too_many_arguments,
    reason = "The parameter order mirrors the corresponding RPython metainterpreter routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and frame ownership"
)]
pub fn deserialize_optimizer_knowledge(
    rd_numb: &[u8],
    rd_consts: &[majit_ir::Const],
    frontend_boxes: &[i64],
    liveboxes: &[OpRef],
    livebox_types: &[majit_ir::Type],
    all_descrs: &[majit_ir::descr::DescrRef],
    cpu: std::sync::Arc<dyn crate::cpu::Cpu>,
    optimizer: &mut super::optimizer::Optimizer,
    ctx: &mut OptContext,
) {
    use crate::resume::DecodedBox;
    use majit_ir::resumecode::Reader;

    let mut reader = Reader::new(rd_numb);
    // bridgeopt.py:126: assert len(frontend_boxes) == len(liveboxes)
    assert!(
        frontend_boxes.len() == liveboxes.len(),
        "frontend_boxes.len()={} != liveboxes.len()={}",
        frontend_boxes.len(),
        liveboxes.len(),
    );

    // bridgeopt.py:130-131: skip resume section
    let startcount = reader.next_item();
    reader.jump((startcount - 1) as usize);

    // bridgeopt.py:133-146: class knowledge
    let mut bitfield: i32 = 0;
    let mut mask: i32 = 0;
    for (i, &livebox) in liveboxes.iter().enumerate() {
        // bridgeopt.py:135 reads `box.type` (intrinsic on the Box).
        // pyre's parallel side table must cover `liveboxes`.
        let tp = livebox_types.get(i).copied().unwrap_or_else(|| {
            panic!(
                "missing livebox_types[{}] (liveboxes.len()={}): \
                 RPython bridgeopt.py:135 reads box.type intrinsically; \
                 pyre's parallel array must match liveboxes length",
                i,
                liveboxes.len()
            )
        });
        if tp != majit_ir::Type::Ref {
            continue;
        }
        if mask == 0 {
            bitfield = reader.next_item();
            mask = 0b100000;
        }
        let class_known = (bitfield & mask) != 0;
        mask >>= 1;
        if class_known {
            // bridgeopt.py:145-146:
            //   cls = optimizer.cpu.cls_of_box(frontend_boxes[i])
            //   optimizer.make_constant_class(box, cls)
            // RPython's type system guarantees frontend_boxes[i] is a valid
            // GcRef when box.type == "r" and class_known is set. Our raw
            // i64 encoding requires a nonnull check (RPython's
            // `box.nonnull()` equivalent, info.py get_known_class).
            let raw_ref = frontend_boxes[i];
            if raw_ref != 0 {
                // bridgeopt.py:145 `optimizer.cpu.cls_of_box(box)` — the
                // runtime box is a `ConstPtr` carrying the GcRef payload.
                let const_box = majit_ir::operand::Operand::const_from_value(majit_ir::Value::Ref(
                    majit_ir::GcRef(raw_ref as usize),
                ));
                let cls = cpu.cls_of_box(&const_box);
                // optimizer.py `make_constant_class` updates
                // `_forwarded` after `get_box_replacement`. `livebox` is a
                // bridge livebox (= inputarg materialized by
                // `ensure_inputarg_bindings`, which runs before
                // `deserialize_optimizer_knowledge`), so it always resolves
                // and the class info install is never skipped.
                if let Some(b) = ctx.get_box_replacement_operand_opt(livebox) {
                    super::optimizer::Optimizer::make_constant_class(ctx, &b, cls, true);
                }
            }
        }
    }

    // bridgeopt.py:148-158: heap knowledge (struct fields)
    let length = reader.next_item();
    let mut result_struct = Vec::new();
    for _ in 0..length {
        let tagged = reader.next_item() as i16;
        let box1 = decode_box(tagged, rd_consts, liveboxes);
        let descr_index = reader.next_item();
        let tagged2 = reader.next_item() as i16;
        let box2 = decode_box(tagged2, rd_consts, liveboxes);
        // bridgeopt.py:155: descr = metainterp_sd.all_descrs[descr_index]
        let descr = &all_descrs[descr_index as usize];
        let opref1 = decoded_box_to_opref(&box1, ctx);
        let opref2 = decoded_box_to_opref(&box2, ctx);
        result_struct.push((opref1, descr.clone(), opref2));
    }
    // bridgeopt.py:159-169: heap knowledge (array items)
    let length = reader.next_item();
    let mut result_array = Vec::new();
    for _ in 0..length {
        let tagged = reader.next_item() as i16;
        let box1 = decode_box(tagged, rd_consts, liveboxes);
        let index = reader.next_item() as i64;
        let descr_index = reader.next_item();
        let tagged2 = reader.next_item() as i16;
        let box2 = decode_box(tagged2, rd_consts, liveboxes);
        // bridgeopt.py:166: descr = metainterp_sd.all_descrs[descr_index]
        let descr = &all_descrs[descr_index as usize];
        let opref1 = decoded_box_to_opref(&box1, ctx);
        let opref2 = decoded_box_to_opref(&box2, ctx);
        result_array.push((opref1, index, descr.clone(), opref2));
    }
    // bridgeopt.py:170-171: optimizer.optheap.deserialize_optheap(...)
    if !result_struct.is_empty() || !result_array.is_empty() {
        optimizer.import_heap_knowledge(&result_struct, &result_array, ctx);
    }

    // bridgeopt.py:173-185: call_loopinvariant knowledge
    let length = reader.next_item();
    let mut result_loopinvariant = Vec::new();
    for _ in 0..length {
        let tagged1 = reader.next_item() as i16;
        let const_box = decode_box(tagged1, rd_consts, liveboxes);
        // bridgeopt.py:179-180: assert isinstance(const, ConstInt); i = const.getint()
        let DecodedBox::Const(majit_ir::Const::Int(const_int)) = const_box else {
            panic!(
                "bridgeopt: loopinvariant entry must be ConstInt, got {:?}",
                const_box
            );
        };
        let tagged2 = reader.next_item() as i16;
        let box2 = decode_box(tagged2, rd_consts, liveboxes);
        let opref2 = decoded_box_to_opref(&box2, ctx);
        // bridgeopt.py:183: result_loopinvariant.append((i, box))
        // No sentinel check — ConstInt(0) is a valid func_ptr value.
        result_loopinvariant.push((const_int, opref2));
    }
    // bridgeopt.py:184-185: optimizer.optrewrite.deserialize_optrewrite(...)
    if !result_loopinvariant.is_empty() {
        optimizer.import_loopinvariant_knowledge(&result_loopinvariant);
    }
}

/// Convert a DecodedBox to an OpRef for the bridge optimizer context.
///
/// RPython's deserialize path passes Const/Box objects directly. In majit,
/// constants must be registered in the optimizer's context to get an OpRef.
fn decoded_box_to_opref(decoded: &crate::resume::DecodedBox, ctx: &mut OptContext) -> OpRef {
    use crate::resume::DecodedBox;
    use majit_ir::Const;
    match decoded {
        DecodedBox::LiveBox(opref) => *opref,
        DecodedBox::Const(Const::Int(v)) => ctx.make_constant_int(*v),
        DecodedBox::Const(Const::Ref(r)) => ctx.make_constant_ref(*r),
        DecodedBox::Const(Const::Float(f)) => ctx.make_constant_float(*f),
    }
}
