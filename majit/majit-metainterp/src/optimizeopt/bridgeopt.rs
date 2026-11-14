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
//! Optimizer-side consumption uses `decode_knowledge_box`, which returns
//! the live InputArg / Const box already in hand.

use majit_ir::OpRef;
use majit_ir::operand::Operand;

use crate::optimizeopt::OptContext;
pub use crate::resume::decode_box;

/// bridgeopt.py tag_box.
pub fn tag_box(
    box_: &Operand,
    liveboxes_from_env: &crate::resume::LiveboxMap,
    memo: &mut crate::resume::ResumeDataLoopMemo,
    env: &dyn majit_ir::BoxEnv,
    new_liveboxes: &crate::resume::LiveboxMap,
) -> Result<i16, crate::resume::TagOverflow> {
    memo._gettagged(box_, env, liveboxes_from_env, new_liveboxes)
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
    let field_ok = |obj: &Operand, val: &Operand| {
        // bridgeopt.py serialize_optheap membership is after
        // `AbstractValue.get_box_replacement`. Walk the box, then test
        // Const / available_boxes (`ResumeDataVirtualAdder._gettagged`).
        let obj = obj.get_box_replacement(false);
        let val = val.get_box_replacement(false);
        (obj.is_constant() || env.is_const(&obj) || available_boxes.contains_key(&obj))
            && (val.is_constant() || env.is_const(&val) || available_boxes.contains_key(&val))
    };
    let triples_struct: Vec<&(Operand, i32, Operand)> = knowledge
        .heap_fields
        .iter()
        .filter(|(obj, _, val)| field_ok(obj, val))
        .collect();
    numb_state.append_int(triples_struct.len() as i64);
    for (obj, descr_idx, val) in triples_struct {
        let obj_tag = tag_box(obj, &numb_state.liveboxes, memo, env, new_liveboxes)?;
        numb_state.writer.append_short(obj_tag as i32);
        numb_state.append_int(*descr_idx as i64);
        let val_tag = tag_box(val, &numb_state.liveboxes, memo, env, new_liveboxes)?;
        numb_state.writer.append_short(val_tag as i32);
    }
    // bridgeopt.py:102-108: array items
    let triples_array: Vec<&(Operand, i64, i32, Operand)> = knowledge
        .heap_arrayitems
        .iter()
        .filter(|(obj, _, _, val)| field_ok(obj, val))
        .collect();
    numb_state.append_int(triples_array.len() as i64);
    for (obj, index, descr_idx, val) in triples_array {
        let obj_tag = tag_box(obj, &numb_state.liveboxes, memo, env, new_liveboxes)?;
        numb_state.writer.append_short(obj_tag as i32);
        // bridgeopt.py:106 numb_state.append_int(index) — pass the original
        // index unchanged; resumecode.py enforces SHORT range on the
        // i64 value, panicking instead of silently wrapping a too-large
        // index into an i32.
        numb_state.append_int(*index);
        numb_state.append_int(*descr_idx as i64);
        let val_tag = tag_box(val, &numb_state.liveboxes, memo, env, new_liveboxes)?;
        numb_state.writer.append_short(val_tag as i32);
    }

    // bridgeopt.py:113-122: loopinvariant results
    let tuples_loopinvariant: Vec<&(i64, Operand)> = knowledge
        .loopinvariant_results
        .iter()
        .filter(|(_, result)| {
            let result = result.get_box_replacement(false);
            result.is_constant() || env.is_const(&result) || available_boxes.contains_key(&result)
        })
        .collect();
    numb_state.append_int(tuples_loopinvariant.len() as i64);
    for (const_ptr, result) in tuples_loopinvariant {
        let const_tag = memo.getconst_int(*const_ptr)?;
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
    use majit_ir::resumecode::Reader;

    let mut reader = Reader::new(rd_numb);
    // bridgeopt.py deserialize_optimizer_knowledge:
    // assert len(frontend_boxes) == len(liveboxes)
    assert!(
        frontend_boxes.len() == liveboxes.len(),
        "frontend_boxes.len()={} != liveboxes.len()={}",
        frontend_boxes.len(),
        liveboxes.len(),
    );

    // Class-C once: resume numbering `liveboxes` → the bound InputArg
    // already in `ctx.inputargs` (setup ran `ensure_inputarg_bindings`).
    let livebox_boxes: Vec<Operand> = liveboxes
        .iter()
        .map(|&r| {
            ctx.inputargs
                .iter()
                .find(|b| b.to_opref() == r)
                .cloned()
                .or_else(|| {
                    // const / resop livebox (rare): the bound producer in
                    // input_ops / new_operations. Never rematerialize.
                    ctx.input_ops
                        .iter()
                        .find(|op| op.pos().get() == r)
                        .map(Operand::from_bound_op)
                })
                .expect("decoded livebox must retain its bound bridge InputArg")
        })
        .collect();

    // bridgeopt.py deserialize_optimizer_knowledge: skip resume section
    let startcount = reader.next_item();
    reader.jump((startcount - 1) as usize);

    // bridgeopt.py deserialize_optimizer_knowledge: class knowledge
    let mut bitfield: i32 = 0;
    let mut mask: i32 = 0;
    for (i, livebox_box) in livebox_boxes.iter().enumerate() {
        // bridgeopt.py deserialize_optimizer_knowledge reads `box.type`.
        // The parallel `livebox_types` array is the type the serializer
        // counted (`fail_arg_types`); InputArg.type_() can diverge after
        // unboxing, so the bitfield still walks the array.
        let tp = livebox_types.get(i).copied().unwrap_or_else(|| {
            panic!(
                "missing livebox_types[{}] (liveboxes.len()={}): \
                 bridgeopt.py deserialize_optimizer_knowledge reads box.type; \
                 the parallel array must match liveboxes length",
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
            // bridgeopt.py deserialize_optimizer_knowledge:
            //   cls = optimizer.cpu.cls_of_box(frontend_boxes[i])
            //   optimizer.make_constant_class(box, cls)
            // RPython's type system guarantees frontend_boxes[i] is a valid
            // GcRef when box.type == "r" and class_known is set. Our raw
            // i64 encoding requires a nonnull check (`box.nonnull()`,
            // info.py get_known_class).
            let raw_ref = frontend_boxes[i];
            if raw_ref != 0 {
                // bridgeopt.py deserialize_optimizer_knowledge
                // `optimizer.cpu.cls_of_box(box)` — the runtime box is a
                // `ConstPtr` carrying the GcRef payload.
                let const_box = majit_ir::operand::Operand::const_from_value(majit_ir::Value::Ref(
                    majit_ir::GcRef(raw_ref as usize),
                ));
                let cls = cpu.cls_of_box(&const_box);
                // optimizer.py `make_constant_class` updates
                // `_forwarded` after `get_box_replacement`.
                super::optimizer::Optimizer::make_constant_class(ctx, livebox_box, cls, true);
            }
        }
    }

    // bridgeopt.py deserialize_optimizer_knowledge: heap knowledge (struct fields)
    let length = reader.next_item();
    let mut result_struct = Vec::new();
    for _ in 0..length {
        let tagged = reader.next_item() as i16;
        let object = decode_knowledge_box(tagged, rd_consts, liveboxes, &livebox_boxes);
        let descr_index = reader.next_item();
        let tagged2 = reader.next_item() as i16;
        let value = decode_knowledge_box(tagged2, rd_consts, liveboxes, &livebox_boxes);
        // bridgeopt.py deserialize_optimizer_knowledge:
        // descr = metainterp_sd.all_descrs[descr_index]
        let descr = &all_descrs[descr_index as usize];
        result_struct.push((object, descr.clone(), value));
    }
    // bridgeopt.py deserialize_optimizer_knowledge: heap knowledge (array items)
    let length = reader.next_item();
    let mut result_array = Vec::new();
    for _ in 0..length {
        let tagged = reader.next_item() as i16;
        let object = decode_knowledge_box(tagged, rd_consts, liveboxes, &livebox_boxes);
        let index = reader.next_item() as i64;
        let descr_index = reader.next_item();
        let tagged2 = reader.next_item() as i16;
        let value = decode_knowledge_box(tagged2, rd_consts, liveboxes, &livebox_boxes);
        // bridgeopt.py deserialize_optimizer_knowledge:
        // descr = metainterp_sd.all_descrs[descr_index]
        let descr = &all_descrs[descr_index as usize];
        result_array.push((object, index, descr.clone(), value));
    }
    // bridgeopt.py deserialize_optimizer_knowledge:
    // optimizer.optheap.deserialize_optheap(...)
    if !result_struct.is_empty() || !result_array.is_empty() {
        optimizer.import_heap_knowledge(&result_struct, &result_array, ctx);
    }

    // bridgeopt.py deserialize_optimizer_knowledge: call_loopinvariant knowledge
    let length = reader.next_item();
    let mut result_loopinvariant = Vec::new();
    for _ in 0..length {
        let tagged1 = reader.next_item() as i16;
        let const_box = decode_knowledge_box(tagged1, rd_consts, liveboxes, &livebox_boxes);
        // bridgeopt.py deserialize_optimizer_knowledge:
        // assert isinstance(const, ConstInt); i = const.getint()
        let Some(const_int) = const_box.const_int() else {
            panic!(
                "bridgeopt: loopinvariant entry must be ConstInt, got {:?}",
                const_box
            );
        };
        let tagged2 = reader.next_item() as i16;
        let result = decode_knowledge_box(tagged2, rd_consts, liveboxes, &livebox_boxes);
        // bridgeopt.py deserialize_optimizer_knowledge:
        // result_loopinvariant.append((i, box))
        // No sentinel check — ConstInt(0) is a valid func_ptr value.
        result_loopinvariant.push((const_int, result));
    }
    // bridgeopt.py deserialize_optimizer_knowledge:
    // optimizer.optrewrite.deserialize_optrewrite(...)
    if !result_loopinvariant.is_empty() {
        optimizer.import_loopinvariant_knowledge(&result_loopinvariant);
    }
}

fn decode_knowledge_box(
    tagged: i16,
    rd_consts: &[majit_ir::Const],
    liveboxes: &[OpRef],
    livebox_boxes: &[Operand],
) -> Operand {
    use crate::resume::{NULLREF, TAG_CONST_OFFSET, TAGBOX, TAGCONST, TAGINT, tagged_eq, untag};
    let (num, tag_type) = untag(tagged);
    match tag_type {
        TAGBOX => {
            debug_assert_eq!(liveboxes.len(), livebox_boxes.len());
            livebox_boxes.get(num as usize).cloned().unwrap_or_else(|| {
                panic!(
                    "bridgeopt decode_knowledge_box TAGBOX out-of-range: \
                         idx={} liveboxes.len()={} (corrupt knowledge stream — \
                         see decode_box TAGBOX)",
                    num as usize,
                    liveboxes.len()
                )
            })
        }
        TAGINT => Operand::const_(majit_ir::Const::Int(num as i64)),
        TAGCONST => {
            if tagged_eq(tagged, NULLREF) {
                Operand::const_(majit_ir::Const::from_gcref(majit_ir::GcRef::NULL))
            } else {
                let idx = (num - TAG_CONST_OFFSET) as usize;
                let c = rd_consts.get(idx).copied().unwrap_or_else(|| {
                    panic!(
                        "bridgeopt decode_knowledge_box TAGCONST out-of-range: \
                         idx={} rd_consts.len()={} (corrupt knowledge stream — \
                         see decode_box TAGCONST)",
                        idx,
                        rd_consts.len()
                    )
                });
                Operand::const_(c)
            }
        }
        _ => unreachable!(
            "bridgeopt decode_knowledge_box: unexpected tag type {}",
            tag_type
        ),
    }
}
