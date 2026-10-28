//! Per-block local state and liveness (`rpython/flowspace/framestate.py`).

use super::*;

/// One block's slot-indexed `Option<Variable>` row, stored by bound slot.
///
/// Same reasoning as [`PackedFrameState`]: the row is sized by the body's
/// local count but holds a live-in handful, and one row per block makes
/// the table `blocks * locals`.
#[derive(Clone, Default)]
pub(in crate::front::mir) struct PackedLocalRow {
    pub(in crate::front::mir) len: usize,
    pub(in crate::front::mir) bound: Vec<(u32, crate::flowspace::model::Variable)>,
}

impl PackedLocalRow {
    pub(in crate::front::mir) fn pack(row: &[Option<crate::flowspace::model::Variable>]) -> Self {
        Self {
            len: row.len(),
            bound: row
                .iter()
                .enumerate()
                .filter_map(|(i, v)| v.clone().map(|v| (i as u32, v)))
                .collect(),
        }
    }

    pub(in crate::front::mir) fn unpack(&self) -> Vec<Option<crate::flowspace::model::Variable>> {
        let mut row = vec![None; self.len];
        for (i, v) in &self.bound {
            row[*i as usize] = Some(v.clone());
        }
        row
    }

    pub(in crate::front::mir) fn set(
        &mut self,
        slot: usize,
        var: crate::flowspace::model::Variable,
    ) {
        match self.bound.binary_search_by_key(&(slot as u32), |(i, _)| *i) {
            Ok(at) => self.bound[at].1 = var,
            Err(at) => self.bound.insert(at, (slot as u32, var)),
        }
    }

    pub(in crate::front::mir) fn unset(&mut self, slot: usize) {
        if let Ok(at) = self.bound.binary_search_by_key(&(slot as u32), |(i, _)| *i) {
            self.bound.remove(at);
        }
    }
}

/// One flowspace word, or the two words of a pair slice, held in the MIR
/// local itself. A slice iterator adds the index cursor beside the pair.
#[derive(Clone)]
pub(in crate::front::mir) enum LocalValue {
    One(crate::flowspace::model::Variable),
    Pair(
        crate::flowspace::model::Variable,
        crate::flowspace::model::Variable,
    ),
    SliceIter {
        items: crate::flowspace::model::Variable,
        index: crate::flowspace::model::Variable,
    },
    /// The address word of a slice `get` / `first` / `last` option. The item
    /// kind is the MIR type of the local, read at the use.
    ItemAddr(crate::flowspace::model::Variable),
}

impl LocalValue {
    /// The single word of a [`LocalValue::One`]. A pair or a slice iterator
    /// is not one word.
    pub(in crate::front::mir) fn one(
        &self,
    ) -> Result<crate::flowspace::model::Variable, LowerError> {
        match self {
            LocalValue::One(var) | LocalValue::ItemAddr(var) => Ok(var.clone()),
            // The cursor, never the item pointer.
            LocalValue::SliceIter { index, .. } => Ok(index.clone()),
            LocalValue::Pair(_, _) => Err(LowerError::Unsupported(
                "pair-slice local read as one word".to_string(),
            )),
        }
    }

    /// Both words of a pair slice. A slice iterator's words are its index
    /// cursor and its item pointer, the two words the shadow slot carries.
    pub(in crate::front::mir) fn pair(
        &self,
    ) -> Result<
        (
            crate::flowspace::model::Variable,
            crate::flowspace::model::Variable,
        ),
        LowerError,
    > {
        match self {
            LocalValue::Pair(ptr, len) => Ok((ptr.clone(), len.clone())),
            LocalValue::SliceIter { index, items, .. } => Ok((index.clone(), items.clone())),
            LocalValue::One(_) | LocalValue::ItemAddr(_) => Err(LowerError::Unsupported(
                "one-word local read as a pair slice".to_string(),
            )),
        }
    }

    /// The word a block inputarg or link arg carries for this slot. The
    /// length of a pair, and the item pointer of a slice iterator, travel
    /// in the shadow slot.
    pub(in crate::front::mir) fn link_word(&self) -> crate::flowspace::model::Variable {
        match self {
            LocalValue::One(var) | LocalValue::Pair(var, _) | LocalValue::ItemAddr(var) => {
                var.clone()
            }
            LocalValue::SliceIter { index, .. } => index.clone(),
        }
    }

    /// Every word this slot holds. A bounds fork forwards each one.
    pub(in crate::front::mir) fn words(&self) -> Vec<crate::flowspace::model::Variable> {
        match self {
            LocalValue::One(var) | LocalValue::ItemAddr(var) => vec![var.clone()],
            LocalValue::Pair(ptr, len) => vec![ptr.clone(), len.clone()],
            LocalValue::SliceIter { items, index } => vec![items.clone(), index.clone()],
        }
    }

    /// Replace words that `map` copied onto the in-bounds block.
    pub(in crate::front::mir) fn remap_words(
        &mut self,
        map: &std::collections::HashMap<u64, crate::flowspace::model::Variable>,
    ) {
        let map_word = |var: &mut crate::flowspace::model::Variable| {
            if let Some(copy) = map.get(&var.id()) {
                *var = copy.clone();
            }
        };
        match self {
            LocalValue::One(var) | LocalValue::ItemAddr(var) => map_word(var),
            LocalValue::Pair(ptr, len) => {
                map_word(ptr);
                map_word(len);
            }
            LocalValue::SliceIter { items, index } => {
                map_word(items);
                map_word(index);
            }
        }
    }
}

pub(in crate::front::mir) fn link_words_of(
    row: &[Option<LocalValue>],
) -> Vec<Option<crate::flowspace::model::Variable>> {
    row.iter()
        .map(|slot| slot.as_ref().map(LocalValue::link_word))
        .collect()
}

/// A [`FrameState`] held in a per-block table with its two slot-indexed
/// rows compressed to the slots that are actually bound.
///
/// `entries` / `locals_w` are sized by the body's local count and read
/// positionally, so a table holding one per block costs
/// `blocks * locals` cells.  On a large body that product dominates the
/// whole front end — `transform_graph_to_jitcode` has 15 335 MIR blocks
/// and 14 215 locals, yet averages ~33 bound slots per block, so the
/// dense rows are 0.2% occupied.  Packing keeps the same values and the
/// same row lengths; a row is rebuilt dense at each use, one at a time.
#[derive(Clone)]
pub(in crate::front::mir) struct PackedFrameState {
    /// The other `FrameState` fields, verbatim; its `entries` /
    /// `locals_w` are left empty and carried in the two sparse rows.
    pub(in crate::front::mir) rest: FrameState,
    pub(in crate::front::mir) entries_len: usize,
    pub(in crate::front::mir) entries: Vec<(u32, crate::flowspace::model::Variable)>,
    pub(in crate::front::mir) locals_w_len: usize,
    pub(in crate::front::mir) locals_w: Vec<(u32, crate::flowspace::model::Hlvalue)>,
}

impl PackedFrameState {
    pub(in crate::front::mir) fn pack(mut fs: FrameState) -> Self {
        let entries_len = fs.entries.len();
        let locals_w_len = fs.locals_w.len();
        let entries = std::mem::take(&mut fs.entries)
            .into_iter()
            .enumerate()
            .filter_map(|(i, v)| v.map(|v| (i as u32, v)))
            .collect();
        let locals_w = std::mem::take(&mut fs.locals_w)
            .into_iter()
            .enumerate()
            .filter_map(|(i, v)| v.map(|v| (i as u32, v)))
            .collect();
        Self {
            rest: fs,
            entries_len,
            entries,
            locals_w_len,
            locals_w,
        }
    }

    pub(in crate::front::mir) fn unpack(&self) -> FrameState {
        let mut fs = self.rest.clone();
        fs.entries = vec![None; self.entries_len];
        for (i, v) in &self.entries {
            fs.entries[*i as usize] = Some(v.clone());
        }
        fs.locals_w = vec![None; self.locals_w_len];
        for (i, v) in &self.locals_w {
            fs.locals_w[*i as usize] = Some(v.clone());
        }
        fs
    }
}

/// Per-block extra-live MIR locals for the deferred array-write base /
/// index of a workspace `index` / `index_mut` call.
///
/// `arr[i] = v` lowers to `_p = index_mut(_s, _i)` (recorded as an
/// `index_elem_alias`, [`Lowering::lower_call`]) followed — across the
/// call's own block split — by `*_p = v`, which
/// [`Lowering::emit_projection_write`] re-expresses as `ArrayWrite {
/// base: _s, index: _i, .. }`.  That `ArrayWrite` is a synthetic use of
/// `_s` / `_i` the MIR never spells, so plain liveness drops them before
/// the write block and the base reaches the rtyper as an undefined
/// operand.  Return, per block, the base/index locals of every such
/// `_p` deref-written in it, for [`compute_mir_liveness`] to mark live;
/// the backward fixpoint then threads them from their definition (which
/// dominates the `_p` use).
pub(in crate::front::mir) fn compute_index_write_extra_live(
    body: &Unstructured,
    llbc: &Llbc,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
) -> Vec<Vec<usize>> {
    let mut index_call: std::collections::HashMap<usize, (Option<usize>, Option<usize>)> =
        std::collections::HashMap::new();
    let forward_blocks = forward_reachable_mask(llbc, body);
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc) else {
            continue;
        };
        let CallFunc::Regular(reg) = &call.func else {
            continue;
        };
        let PlaceKind::Local(p) = call.dest.kind else {
            continue;
        };
        // The workspace `index_mut` call, `Vec<T>::index_mut`, and brick
        // 3's `*base.add(idx)` all defer their `*p = v` write to a later
        // block, where the base / index operands are no longer spelled (the
        // `ArrayWrite` is synthesised by `emit_projection_write`).  The Vec
        // arm must be gated on the identical escape guard the intercept
        // applies, so the two stay in lockstep — threading a live-set entry
        // for a write the intercept declined to lift would leave the base
        // undefined.
        let is_deferred_write_producer = is_workspace_index_regular(reg, llbc)
            || (is_vec_index_mut_call(
                reg,
                call.args.get(1).and_then(operand_tyref),
                llbc,
                gc_struct_ids,
                &std::collections::HashSet::<String>::new(),
            ) && index_mut_result_is_element(llbc, body, p as usize))
            || is_list_items_elem_ptr_add_parts(
                reg,
                call.args.len(),
                operand_local(call.args.first()),
                call.args.first().and_then(operand_tyref),
                p as usize,
                body,
                llbc,
            )
            || is_typed_items_elem_ptr_add_parts(
                reg,
                call.args.len(),
                operand_local(call.args.first()),
                call.args.first().and_then(operand_tyref),
                p as usize,
                body,
                llbc,
            )
            || is_string_items_elem_ptr_add_parts(
                reg,
                call.args.len(),
                operand_local(call.args.first()),
                call.args.first().and_then(operand_tyref),
                p as usize,
                body,
                llbc,
            );
        if !is_deferred_write_producer {
            continue;
        }
        index_call.insert(
            p as usize,
            (
                operand_local(call.args.first()),
                operand_local(call.args.get(1)),
            ),
        );
    }
    let mut extra = vec![Vec::new(); body.body.len()];
    if index_call.is_empty() {
        return extra;
    }
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        for stmt in &bb.statements {
            let Ok(StmtKind::Assign(place, _)) = stmt.stmt_kind_ref() else {
                continue;
            };
            if let Some(p) = deref_write_base_local(place)
                && let Some((base, idx)) = index_call.get(&p)
            {
                extra[bb_idx].extend(base.iter().copied());
                extra[bb_idx].extend(idx.iter().copied());
            }
        }
    }
    extra
}

/// Per-block extra-live MIR locals for a struct-field `ptr::add`.
///
/// `(*p.add(i)).field` lowers to `getinteriorfield`
/// (`rewrite_op_getinteriorfield`) whose base is the GcArray header and
/// whose index is `i`. The field block never spells those operands, so
/// plain liveness drops them before the use. The header is the add base
/// when nothing produces it, `entries_item_ptr`'s argument, or
/// `entry_ptr`'s receiver — the same recovery [`struct_field_ptr_add_trace`]
/// records on the alias.
pub(in crate::front::mir) fn compute_interior_field_extra_live(
    body: &Unstructured,
    llbc: &Llbc,
    tombstoned: &std::collections::HashSet<String>,
) -> Vec<Vec<usize>> {
    let mut extra = vec![Vec::new(); body.body.len()];
    let forward_blocks = forward_reachable_mask(llbc, body);
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        let Ok(TermKind::Call { call, .. }) = bb.term_ref(llbc) else {
            continue;
        };
        let CallFunc::Regular(reg) = &call.func else {
            continue;
        };
        let PlaceKind::Local(dest) = call.dest.kind else {
            continue;
        };
        let Some(traced) = struct_field_ptr_add_trace(
            reg,
            call.args.len(),
            operand_local(call.args.first()),
            call.args.first().and_then(operand_tyref),
            operand_local(call.args.get(1)),
            dest as usize,
            body,
            llbc,
            tombstoned,
        ) else {
            continue;
        };
        let header_local = match traced.header {
            TracedPtrAddHeader::EntriesItem { local } => local,
            TracedPtrAddHeader::EntryPtr { recv_local, .. } => recv_local,
        };
        for (use_idx, use_bb) in body.body.iter().enumerate() {
            if !forward_blocks[use_idx] {
                continue;
            }
            if block_has_struct_field_value_use(llbc, use_bb, dest as usize) {
                extra[use_idx].push(header_local);
                extra[use_idx].push(traced.index_local);
            }
        }
    }
    extra
}

pub(in crate::front::mir) fn compute_mir_liveness(
    llbc: &Llbc,
    body: &Unstructured,
    extra_live: &[Vec<usize>],
    glue_call_drops: &bit_set::BitSet,
) -> Vec<bit_set::BitSet> {
    use bit_set::BitSet;

    let n_blocks = body.body.len();
    let n_locals = body.locals.locals.len();
    // Same closure as whole-body scans: a panic-abort default still
    // contributes uses, and an `on_unwind` cleanup block does not.
    let forward = forward_reachable_mask(llbc, body);
    let mut uses = Vec::with_capacity(n_blocks);
    let mut defs = Vec::with_capacity(n_blocks);
    for bb_idx in 0..n_blocks {
        if forward[bb_idx] {
            uses.push(BitSet::with_capacity(n_locals));
            defs.push(BitSet::with_capacity(n_locals));
        } else {
            uses.push(BitSet::new());
            defs.push(BitSet::new());
        }
    }
    let mut succs = vec![Vec::<usize>::new(); n_blocks];
    let mut preds = vec![Vec::<usize>::new(); n_blocks];

    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward[bb_idx] {
            continue;
        }
        for stmt in &bb.statements {
            let Ok(kind) = stmt.stmt_kind_ref() else {
                continue;
            };
            match kind {
                StmtKind::Assign(place, rvalue) => {
                    mark_rvalue_uses(rvalue, &mut uses[bb_idx], &defs[bb_idx], n_locals);
                    mark_place_write(place, &mut uses[bb_idx], &mut defs[bb_idx], n_locals);
                }
                StmtKind::PlaceMention(place) => {
                    mark_place_use(place, &mut uses[bb_idx], &defs[bb_idx], n_locals)
                }
                StmtKind::Assert(assert) => {
                    mark_operand_use(&assert.cond, &mut uses[bb_idx], &defs[bb_idx], n_locals)
                }
                StmtKind::StorageLive(_)
                | StmtKind::StorageDead(_)
                | StmtKind::Borrowck(_)
                | StmtKind::Unknown => {}
            }
        }
        let Ok(term) = bb.term_ref(llbc) else {
            continue;
        };
        match term {
            TermKind::Return => mark_local_use(0, &mut uses[bb_idx], &defs[bb_idx], n_locals),
            TermKind::Goto { target } => {
                push_successor(&mut succs[bb_idx], &mut preds, bb_idx, *target, n_blocks)
            }
            TermKind::Switch { discr, targets } => {
                mark_operand_use(discr, &mut uses[bb_idx], &defs[bb_idx], n_locals);
                match targets {
                    SwitchTargets::If(a, b) => {
                        push_successor(&mut succs[bb_idx], &mut preds, bb_idx, *a, n_blocks);
                        push_successor(&mut succs[bb_idx], &mut preds, bb_idx, *b, n_blocks);
                    }
                    SwitchTargets::SwitchInt(_, arms, default) => {
                        for (_, bb) in arms {
                            push_successor(&mut succs[bb_idx], &mut preds, bb_idx, *bb, n_blocks);
                        }
                        push_successor(&mut succs[bb_idx], &mut preds, bb_idx, *default, n_blocks);
                    }
                }
            }
            TermKind::Call { call, target, .. } => {
                mark_call_uses(call, &mut uses[bb_idx], &defs[bb_idx], n_locals);
                mark_place_write(&call.dest, &mut uses[bb_idx], &mut defs[bb_idx], n_locals);
                push_successor(&mut succs[bb_idx], &mut preds, bb_idx, *target, n_blocks);
            }
            TermKind::Assert { assert, target, .. } => {
                mark_operand_use(&assert.cond, &mut uses[bb_idx], &defs[bb_idx], n_locals);
                push_successor(&mut succs[bb_idx], &mut preds, bb_idx, *target, n_blocks);
            }
            TermKind::Drop { place, target, .. } => {
                // Glue calls read the dropped local; erased drops do not.
                if glue_call_drops.contains(bb_idx) {
                    mark_place_use(place, &mut uses[bb_idx], &defs[bb_idx], n_locals);
                }
                push_successor(&mut succs[bb_idx], &mut preds, bb_idx, *target, n_blocks)
            }
            TermKind::UnwindResume
            | TermKind::UnwindTerminate
            | TermKind::Abort(_)
            | TermKind::Panic { .. }
            | TermKind::UndefinedBehavior
            | TermKind::Unknown => {}
        }
    }

    // Synthetic deferred-array-write uses: the base/index of a
    // workspace index call feed an `ArrayWrite` the MIR never spells
    // (see `compute_index_write_extra_live`).  Mark them live-used in
    // the block that holds the `*p = v` write — unless that block also
    // defines them — so the backward fixpoint threads them in from
    // their definition, which dominates the `_p` use.
    for (bb_idx, locals) in extra_live.iter().enumerate().take(n_blocks) {
        if !forward.get(bb_idx).copied().unwrap_or(false) {
            continue;
        }
        for &local_idx in locals {
            if local_idx < n_locals && !defs[bb_idx].contains(local_idx) {
                uses[bb_idx].insert(local_idx);
            }
        }
    }

    let mut live_in = Vec::with_capacity(n_blocks);
    for bb_idx in 0..n_blocks {
        if forward[bb_idx] {
            live_in.push(BitSet::with_capacity(n_locals));
        } else {
            live_in.push(BitSet::new());
        }
    }
    let mut worklist: std::collections::VecDeque<usize> =
        (0..n_blocks).rev().filter(|&bb| forward[bb]).collect();
    let mut in_worklist = forward.clone();
    while let Some(bb_idx) = worklist.pop_front() {
        in_worklist[bb_idx] = false;
        let mut new_in = BitSet::with_capacity(n_locals);
        for &succ in &succs[bb_idx] {
            new_in.union_with(&live_in[succ]);
        }
        new_in.difference_with(&defs[bb_idx]);
        new_in.union_with(&uses[bb_idx]);
        if new_in != live_in[bb_idx] {
            live_in[bb_idx] = new_in;
            for &pred in &preds[bb_idx] {
                if !in_worklist[pred] {
                    worklist.push_back(pred);
                    in_worklist[pred] = true;
                }
            }
        }
    }

    live_in
}

pub(in crate::front::mir) fn push_successor(
    out: &mut Vec<usize>,
    preds: &mut [Vec<usize>],
    source: usize,
    target: u64,
    n_blocks: usize,
) {
    let target = target as usize;
    if target < n_blocks && !out.contains(&target) {
        out.push(target);
        preds[target].push(source);
    }
}

pub(in crate::front::mir) fn mark_call_uses(
    call: &CallPayload,
    uses: &mut bit_set::BitSet,
    defs: &bit_set::BitSet,
    n_locals: usize,
) {
    if let CallFunc::Dynamic(op, _) = &call.func {
        mark_operand_use(op, uses, defs, n_locals);
    }
    for arg in &call.args {
        mark_operand_use(arg, uses, defs, n_locals);
    }
}

pub(in crate::front::mir) fn mark_rvalue_uses(
    rvalue: &Rvalue,
    uses: &mut bit_set::BitSet,
    defs: &bit_set::BitSet,
    n_locals: usize,
) {
    match rvalue {
        Rvalue::Use(op, _)
        | Rvalue::UnaryOp(_, op)
        | Rvalue::Cast(_, op, _)
        | Rvalue::Repeat(op, _, _, _)
        | Rvalue::ShallowInitBox(op, _) => mark_operand_use(op, uses, defs, n_locals),
        Rvalue::BinaryOp(_, lhs, rhs) => {
            mark_operand_use(lhs, uses, defs, n_locals);
            mark_operand_use(rhs, uses, defs, n_locals);
        }
        Rvalue::Ref { place, .. }
        | Rvalue::RawPtr { place, .. }
        | Rvalue::Len(place)
        | Rvalue::Discriminant(place) => mark_place_use(place, uses, defs, n_locals),
        Rvalue::Aggregate(_, operands) => {
            for op in operands {
                mark_operand_use(op, uses, defs, n_locals);
            }
        }
        Rvalue::NullaryOp(_, _) | Rvalue::Unknown => {}
    }
}

pub(in crate::front::mir) fn mark_operand_use(
    op: &Operand,
    uses: &mut bit_set::BitSet,
    defs: &bit_set::BitSet,
    n_locals: usize,
) {
    match op {
        Operand::Copy(place) | Operand::Move(place) => mark_place_use(place, uses, defs, n_locals),
        Operand::Const(_) => {}
    }
}

pub(in crate::front::mir) fn mark_place_use(
    place: &Place,
    uses: &mut bit_set::BitSet,
    defs: &bit_set::BitSet,
    n_locals: usize,
) {
    match &place.kind {
        PlaceKind::Local(i) => mark_local_use(*i as usize, uses, defs, n_locals),
        PlaceKind::Projection(inner, elem) => {
            mark_place_use(inner, uses, defs, n_locals);
            mark_projection_index_offset_use(elem, uses, defs, n_locals);
        }
        PlaceKind::Global { .. } | PlaceKind::Unknown => {}
    }
}

pub(in crate::front::mir) fn mark_place_write(
    place: &Place,
    uses: &mut bit_set::BitSet,
    defs: &mut bit_set::BitSet,
    n_locals: usize,
) {
    match &place.kind {
        PlaceKind::Local(i) => mark_local_def(*i as usize, defs, n_locals),
        PlaceKind::Projection(inner, elem) => {
            mark_place_use(inner, uses, defs, n_locals);
            mark_projection_index_offset_use(elem, uses, defs, n_locals);
        }
        PlaceKind::Global { .. } | PlaceKind::Unknown => {}
    }
}

/// A place whose projection chain contains an `Index { offset, .. }` or
/// an `Offset { bytes, .. }` reads that operand's local at lowering
/// time — `resolve_place` / `emit_projection_write` route `Index.offset`
/// through `index_offset_var` → `resolve_operand`, and
/// `resolve_offset_place` does the same for `Offset.bytes`.  The operand
/// lives in the projection element, not in `inner`, so the surrounding
/// `mark_place_use` / `mark_place_write` recursion (which descends only
/// into `inner`) would miss it.  Mark it as a use so a loop-carried
/// index or byte-count local receives a block inputarg and threads
/// across the back-edge, instead of failing loud as an uninitialised
/// local at the loop header.  `from_end` is ignored, matching
/// `index_offset_var`.  Upstream `_compute_liveness_must_continue`
/// marks every register argument alive.
pub(in crate::front::mir) fn mark_projection_index_offset_use(
    elem: &ProjectionElem,
    uses: &mut bit_set::BitSet,
    defs: &bit_set::BitSet,
    n_locals: usize,
) {
    let ProjectionElem::Tagged(v) = elem else {
        return;
    };
    let obj = match v.as_object() {
        Some(m) => m,
        None => return,
    };
    let operand = obj
        .get("Index")
        .and_then(serde_json::Value::as_object)
        .and_then(|m| m.get("offset"))
        .or_else(|| {
            obj.get("Offset")
                .and_then(serde_json::Value::as_object)
                .and_then(|m| m.get("bytes"))
        });
    let Some(offset) = operand else {
        return;
    };
    if let Ok(op) = serde_json::from_value::<Operand>(offset.clone()) {
        mark_operand_use(&op, uses, defs, n_locals);
    }
}

pub(in crate::front::mir) fn mark_local_use(
    local_idx: usize,
    uses: &mut bit_set::BitSet,
    defs: &bit_set::BitSet,
    n_locals: usize,
) {
    if local_idx >= n_locals || defs.contains(local_idx) {
        return;
    }
    uses.insert(local_idx);
}

pub(in crate::front::mir) fn mark_local_def(
    local_idx: usize,
    defs: &mut bit_set::BitSet,
    n_locals: usize,
) {
    if local_idx < n_locals {
        defs.insert(local_idx);
    }
}
