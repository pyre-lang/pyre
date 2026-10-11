//! Take the shadow-stack root bracket out of a body before it is lowered.
//!
//! Upstream never puts a root bracket in a jitcode.  `gc_push_roots` /
//! `gc_pop_roots` are genop'd by `ShadowStackFrameworkGCTransformer`
//! (`memory/gctransform/shadowstack.py` `push_roots` / `pop_roots`), and that
//! transformer runs out of the C backend's database long after
//! `warmspot.py` `make_jitcodes` has read the graphs.  pyre spells the bracket
//! in interpreter source, so this pass removes it from what the codewriter
//! reads, the way the gctransformer's absence does upstream.  The native build
//! keeps executing the source bracket unchanged.
//!
//! The bracket is scalar-replaced: every slot a body publishes becomes one
//! synthetic MIR local.  A pin writes the slot local and answers the value it
//! was handed, a read-back answers the slot local, a `set` rewrites it, and the
//! opener, `base()`, `shadow_stack_len()`, the normalize calls and the close
//! all disappear.  The flow-graph builder then carries each slot through the
//! body's blocks like any other local, so a merge or a loop needs nothing of
//! its own.  The references the slots held stay rooted by the jitcode's own
//! root sets (the jitframe gcmap, the blackhole's `registers_r`, the recorder).
//!
//! The rewrite needs every slot index to be a compile-time offset from the
//! depth the body was entered at, so it runs an abstract interpretation of the
//! stack depth first.  A callee is assumed to leave the depth where it found
//! it: a body whose own depth at `Return` is not zero is refused, so the only
//! functions that could break that assumption are the ones this pass never
//! rewrites.  A callee that *calls* such a function but still returns at
//! depth 0 (its own `RootScope` Close rewinds) is depth-neutral: it does not
//! read the caller's slots by index, and a collection inside it roots the
//! caller's replaced slots through the jitframe gcmap, the blackhole's
//! `registers_r`, and the recorder.  Observes at entry depth is a refusal
//! (every live slot is the caller's). Inside this body's Open it is
//! allowed for neutrality and for erasure (`eval_slice_index` pin then
//! `getindex_w`): the callee reads our pins, which become slot locals,
//! and it does not leave extra slots. A LeavesAbove or ParamSlots
//! callee is the same split for neutrality, but erasure still refuses
//! — after the rewrite nothing rewinds a leftover
//! (`_fix_graph_after_inlining`). Anything the interpretation cannot
//! model refuses the whole body, which then lowers exactly as it did
//! before.
//!
//! Unwind edges are not followed: the flow-graph builder does not lower an
//! `on_unwind` cleanup chain either, and one chain is shared by calls made at
//! different depths.

use std::collections::{HashMap, VecDeque};

use majit_charon_reader::Llbc;
use majit_charon_reader::ullbc::{
    FunDecl, Operand, Place, PlaceKind, ProjectionElem, Rvalue, SpanRef, StmtKind, TermKind, TyRef,
    Unstructured,
};
use serde_json::{Value, json};

/// One shadow-stack operation, keyed by the callee's path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Leaf {
    Open,
    Close,
    Base,
    Len,
    Pin,
    Publish,
    Normalize,
    NormalizeMoved,
    Get,
    Set,
    ReloadTop,
    /// `shadow_stack_copy_range` / `_into_vec`: a Get of argument 0 for
    /// the walk, Unmodeled for erase.
    CopyRange,
    /// `RootedItems::new`: a second Open for the walk, Unmodeled for erase.
    ItemsOpen,
    /// `RootedItems::push`.
    ItemsPush,
    /// `RootedItems::{len, is_empty, assert_owns_the_top}`.
    ItemsNop,
    /// `RootedItems::{get, take}`: own-slot reads for the walk.
    ItemsGet,
    /// `RootedItems` drop / `drop_in_place`.
    ItemsClose,
    /// Touches the stack in a way this pass does not model.
    Unmodeled,
}

/// Classify a `gc_roots` callee, and say whether it takes the guard as its
/// first argument.  `None` for a callee outside the module.
///
/// Charon names an inherent method `gc_roots::<Impl>::leaf`, so the type a
/// method belongs to is read off the call: the receiver's type for a method,
/// the destination's for the guard's constructor.
fn classify_call(
    call: &majit_charon_reader::ullbc::CallPayload,
    llbc: &Llbc,
) -> Option<(Leaf, bool)> {
    let path = callee_path(call, llbc)?;
    let segments: Vec<&str> = path.split("::").collect();
    if !segments.contains(&super::ROOT_SCOPE_MODULE) {
        return None;
    }
    let leaf = *segments.last()?;
    let names_type = |ty: &majit_charon_reader::ullbc::TyRef, name: &str| {
        super::tyref_class_root(ty, llbc).is_some_and(|root| root.rsplit("::").next() == Some(name))
    };
    let receiver_ty = match call.args.first() {
        Some(Operand::Copy(p) | Operand::Move(p)) => Some(&p.ty),
        _ => None,
    };
    let is_method =
        segments.contains(&super::ROOT_SCOPE_TYPE) || segments.iter().any(|s| s.starts_with('<'));
    if is_method {
        if names_type(&call.dest.ty, super::ROOT_SCOPE_TYPE) && leaf == "new" {
            return Some((Leaf::Open, false));
        }
        if names_type(&call.dest.ty, "RootedItems") && leaf == "new" {
            return Some((Leaf::ItemsOpen, false));
        }
        if names_type(&call.dest.ty, "RootedOnceRef")
            || receiver_ty.is_some_and(|ty| names_type(ty, "RootedOnceRef"))
        {
            // Process-global MiniMark slot, not the thread shadow stack.
            return None;
        }
        if receiver_ty.is_some_and(|ty| names_type(ty, "RootedItems")) {
            let kind = match leaf {
                "push" => Leaf::ItemsPush,
                "len" | "is_empty" | "assert_owns_the_top" => Leaf::ItemsNop,
                "get" | "take" => Leaf::ItemsGet,
                "drop" | "drop_in_place" => Leaf::ItemsClose,
                _ => Leaf::Unmodeled,
            };
            return Some((kind, true));
        }
        if !receiver_ty.is_some_and(|ty| names_type(ty, super::ROOT_SCOPE_TYPE)) {
            // `RootStack` and the walkers: the stack itself.
            return Some((Leaf::Unmodeled, false));
        }
        let kind = match leaf {
            "base" => Leaf::Base,
            "pin_root" => Leaf::Pin,
            "get" => Leaf::Get,
            "publish" | "pin_roots" => Leaf::Publish,
            "normalize" => Leaf::Normalize,
            "normalize_moved" => Leaf::NormalizeMoved,
            "set" => Leaf::Set,
            "drop" | "drop_in_place" => Leaf::Close,
            _ => Leaf::Unmodeled,
        };
        return Some((kind, true));
    }
    let kind = match leaf {
        "push_roots" => Leaf::Open,
        "root_scope_close" => return Some((Leaf::Close, true)),
        "pin_root" => Leaf::Pin,
        "pin_roots" | "publish_roots" => Leaf::Publish,
        "normalize_roots" => Leaf::Normalize,
        "shadow_stack_len" => Leaf::Len,
        "shadow_stack_get" => Leaf::Get,
        "shadow_stack_set" => Leaf::Set,
        "reload_top_root" => Leaf::ReloadTop,
        "shadow_stack_copy_range" | "shadow_stack_copy_range_into_vec" => Leaf::CopyRange,
        "mark_prebuilt_roots_dirty"
        | "prebuilt_roots_dirty"
        | "clear_prebuilt_roots_dirty"
        | "increase_root_stack_depth"
        | "root_stack_depth" => return None,
        // Everything else in the module reads or rewrites the stack itself.
        _ => Leaf::Unmodeled,
    };
    Some((kind, false))
}

/// What a special local stands for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Special {
    /// A `RootScope` guard opened at this depth.
    Guard(usize),
    /// A borrow of a guard.
    Alias(usize),
    /// A slot index: the entry depth plus this offset.
    Index(usize),
    /// A checked-add pair whose `.0` is this offset.
    IndexPair(usize),
}

/// A call or assert this pass rewrites to a `Goto`, with the statements that
/// replace it.
struct TermRewrite {
    target: u64,
    stmts: Vec<Value>,
}

struct Plan {
    specials: HashMap<usize, Special>,
    /// Statements to drop, per block.
    removed: HashMap<usize, Vec<usize>>,
    /// Statements to insert after a statement, per block.
    inserted_after: HashMap<(usize, usize), Vec<Value>>,
    terms: HashMap<usize, TermRewrite>,
    slot_count: usize,
    slot_ty: Option<Value>,
    /// Blocks no path from the entry reaches once the rewritten calls lose
    /// their unwind edges.
    unreachable: Vec<usize>,
}

type Refusal = &'static str;

fn place_local(place: &Place) -> Option<usize> {
    match place.kind {
        PlaceKind::Local(l) => Some(l as usize),
        _ => None,
    }
}

fn operand_local(op: &Operand) -> Option<usize> {
    match op {
        Operand::Copy(p) | Operand::Move(p) => place_local(p),
        Operand::Const(_) => None,
    }
}

/// The base local of `*local`.
fn deref_of_local(place: &Place) -> Option<usize> {
    let PlaceKind::Projection(inner, ProjectionElem::Atom(elem)) = &place.kind else {
        return None;
    };
    (elem == "Deref").then(|| place_local(inner)).flatten()
}

/// `local.0` / `local.1` of a tuple, or `(*local).0` of a borrow.
fn tuple_field_of_local(place: &Place) -> Option<(usize, u64)> {
    let PlaceKind::Projection(inner, ProjectionElem::Tagged(elem)) = &place.kind else {
        return None;
    };
    let field = elem.get("Field")?.as_array()?;
    let index = field.get(1)?.as_u64()?;
    let base = place_local(inner).or_else(|| deref_of_local(inner))?;
    Some((base, index))
}

/// Whether a `Ref` / `RawPtr` kind writes through the borrowed place.
fn ref_kind_mutates(kind: &Value) -> bool {
    match kind {
        Value::String(s) => s.contains("Mut"),
        Value::Object(m) => m.keys().any(|k| k.contains("Mut")),
        _ => false,
    }
}

/// Assignments of each local: `Assign` destinations, call destinations,
/// and `&mut` / `*mut` borrows of a local. `StorageLive` / `StorageDead`
/// are not definitions.
fn local_def_counts(body: &Unstructured, llbc: &Llbc) -> Vec<u32> {
    let n = body.locals.locals.len();
    let mut counts = vec![0u32; n];
    let bump = |counts: &mut [u32], l: Option<usize>| {
        if let Some(l) = l
            && let Some(c) = counts.get_mut(l)
        {
            *c = c.saturating_add(1);
        }
    };
    for block in &body.body {
        for stmt in &block.statements {
            let Ok(kind) = stmt.stmt_kind_ref() else {
                continue;
            };
            match kind {
                StmtKind::Assign(place, rvalue) => {
                    bump(&mut counts, place_local(place));
                    match rvalue {
                        Rvalue::Ref {
                            place: src, kind, ..
                        }
                        | Rvalue::RawPtr {
                            place: src, kind, ..
                        } => {
                            if ref_kind_mutates(kind) {
                                bump(
                                    &mut counts,
                                    place_local(src).or_else(|| deref_of_local(src)),
                                );
                            }
                        }
                        _ => {}
                    }
                }
                StmtKind::StorageLive(_) | StmtKind::StorageDead(_) => {}
                _ => {}
            }
        }
        if let Ok(TermKind::Call { call, .. }) = block.term_ref(llbc) {
            bump(&mut counts, place_local(&call.dest));
        }
    }
    counts
}

/// An unsigned scalar constant operand.
fn const_usize(op: &Operand, llbc: &Llbc) -> Option<usize> {
    let Operand::Const(v) = op else {
        return None;
    };
    let lit = llbc.const_expr_literal(v)?;
    let scalar = lit.get("Scalar")?;
    let lit = scalar.get("Unsigned").or_else(|| scalar.get("Signed"))?;
    lit.as_array()?.get(1)?.as_str()?.parse().ok()
}

/// A compile-time slot-index `+ k` or `- k`. `"*Checked"` yields a pair;
/// `"Add"` / `"Sub"` / `{"Add": "Wrap"}` / `{"Sub": "Wrap"}` a value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IndexArith {
    Add { checked: bool },
    Sub { checked: bool },
}

fn binop_is_index_arith(op: &Value) -> Option<IndexArith> {
    match op {
        Value::String(s) if s == "AddChecked" => Some(IndexArith::Add { checked: true }),
        Value::String(s) if s == "Add" => Some(IndexArith::Add { checked: false }),
        Value::Object(m) if m.contains_key("Add") => Some(IndexArith::Add { checked: false }),
        Value::String(s) if s == "SubChecked" => Some(IndexArith::Sub { checked: true }),
        Value::String(s) if s == "Sub" => Some(IndexArith::Sub { checked: false }),
        Value::Object(m) if m.contains_key("Sub") => Some(IndexArith::Sub { checked: false }),
        _ => None,
    }
}

fn copy_operand_json(op: &Value) -> Value {
    match op.as_object() {
        Some(m) if m.contains_key("Move") => json!({ "Copy": m["Move"].clone() }),
        _ => op.clone(),
    }
}

fn local_place_json(local: usize, ty: &Value) -> Value {
    json!({ "kind": { "Local": local }, "ty": ty })
}

fn assign_json(dest: Value, rvalue: Value) -> Value {
    json!({ "Assign": [dest, rvalue] })
}

/// Locals this place names that are bound as specials, including a
/// tuple-field or deref of one (`AddChecked` overflow bit, `*guard`).
#[allow(dead_code)]
fn specials_in_place(place: &Place, specials: &HashMap<usize, Special>) -> Vec<Special> {
    let mut out = Vec::new();
    let push = |out: &mut Vec<Special>, local: usize| {
        if let Some(s) = specials.get(&local) {
            out.push(*s);
        }
    };
    if let Some(l) = place_local(place) {
        push(&mut out, l);
    }
    if let Some((pair, _)) = tuple_field_of_local(place) {
        push(&mut out, pair);
    }
    if let Some(l) = deref_of_local(place) {
        push(&mut out, l);
    }
    out
}

fn specials_in_json(v: &Value, specials: &HashMap<usize, Special>) -> Vec<Special> {
    let mut out = Vec::new();
    fn walk(v: &Value, specials: &HashMap<usize, Special>, out: &mut Vec<Special>) {
        match v {
            Value::Object(map) => {
                if let Some(local) = map.get("Local").and_then(Value::as_u64)
                    && let Some(s) = specials.get(&(local as usize))
                {
                    out.push(*s);
                }
                for nested in map.values() {
                    walk(nested, specials, out);
                }
            }
            Value::Array(items) => {
                for nested in items {
                    walk(nested, specials, out);
                }
            }
            _ => {}
        }
    }
    walk(v, specials, &mut out);
    out
}

/// Overflow bit of an erased `AddChecked` (`IndexPair.1`). The add is a
/// compile-time slot offset, so the overflow flag is false by construction.
fn assert_is_erased_index_overflow(
    assert: &majit_charon_reader::ullbc::AssertStmt,
    specials: &HashMap<usize, Special>,
) -> bool {
    let pair = match &assert.cond {
        Operand::Copy(p) | Operand::Move(p) => tuple_field_of_local(p),
        Operand::Const(_) => None,
    };
    matches!(
        pair.and_then(|(pair, field)| (field == 1).then_some(specials.get(&pair))),
        Some(Some(Special::IndexPair(_)))
    )
}

/// `BoundsCheck { len, index }` in Charon's `AssertKind` payload.
fn bounds_check_len_and_index(check_kind: &Value) -> Option<(&Value, &Value)> {
    let bc = check_kind.get("BoundsCheck")?;
    if let Some(map) = bc.as_object() {
        return Some((map.get("len")?, map.get("index")?));
    }
    let arr = bc.as_array()?;
    (arr.len() >= 2).then_some((&arr[0], &arr[1]))
}

/// Compile-time slot offset named by an operand JSON node.
fn json_index_offset(v: &Value, specials: &HashMap<usize, Special>) -> Option<usize> {
    match specials_in_json(v, specials).as_slice() {
        [Special::Index(k)] | [Special::IndexPair(k)] => Some(*k),
        _ => None,
    }
}

/// Length of the erased slot array: a constant, or `shadow_stack_len`
/// (an `Index` special holding the current depth).
fn json_slot_array_len(
    v: &Value,
    specials: &HashMap<usize, Special>,
    const_locals: &HashMap<usize, usize>,
    llbc: &Llbc,
) -> Option<usize> {
    if let Some(n) = json_const_usize(v, llbc) {
        return Some(n);
    }
    if let Some(local) = json_operand_local(v)
        && let Some(n) = const_locals.get(&local)
    {
        return Some(*n);
    }
    match specials_in_json(v, specials).as_slice() {
        [Special::Index(d)] | [Special::IndexPair(d)] => Some(*d),
        _ => None,
    }
}

fn json_operand_local(v: &Value) -> Option<usize> {
    v.get("Copy")
        .or_else(|| v.get("Move"))
        .and_then(|p| p.get("kind"))
        .and_then(|k| k.get("Local"))
        .and_then(Value::as_u64)
        .map(|n| n as usize)
}

fn json_const_usize(v: &Value, llbc: &Llbc) -> Option<usize> {
    let lit = llbc.const_expr_literal(v.get("Const").unwrap_or(v))?;
    let scalar = lit.get("Scalar")?;
    let lit = scalar.get("Unsigned").or_else(|| scalar.get("Signed"))?;
    lit.as_array()?.get(1)?.as_str()?.parse().ok()
}

/// Assert that is true by construction of the erased bracket: the overflow
/// bit of `len + k`, or a `BoundsCheck` whose index is an erased slot
/// offset and whose length is that same slot array's length (`index < len`).
/// Anything else keeps the bracket.
fn assert_is_erased_index_add_overflow(
    assert: &majit_charon_reader::ullbc::AssertStmt,
    specials: &HashMap<usize, Special>,
) -> bool {
    if assert.expected {
        return false;
    }
    let Some(arr) = assert.check_kind.get("Overflow").and_then(Value::as_array) else {
        return false;
    };
    let is_index_arith = match arr.first() {
        Some(Value::String(s)) => {
            s == "Add" || s == "AddChecked" || s == "Sub" || s == "SubChecked"
        }
        Some(Value::Object(m)) => m.contains_key("Add") || m.contains_key("Sub"),
        _ => false,
    };
    if !is_index_arith || arr.len() < 3 {
        return false;
    }
    // A compile-time slot offset plus or minus a small usize cannot
    // overflow a machine word: the offsets this pass binds are the
    // published slot count, and a `len - k` it accepted already
    // proved `k` fits.
    json_index_offset(&arr[1], specials).is_some() || json_index_offset(&arr[2], specials).is_some()
}

fn assert_is_slot_index_bounds_check(
    assert: &majit_charon_reader::ullbc::AssertStmt,
    specials: &HashMap<usize, Special>,
    const_locals: &HashMap<usize, usize>,
    llbc: &Llbc,
) -> bool {
    if assert_is_erased_index_overflow(assert, specials)
        || assert_is_erased_index_add_overflow(assert, specials)
    {
        return true;
    }
    if !assert.expected {
        return false;
    }
    let Some((len, index)) = bounds_check_len_and_index(&assert.check_kind) else {
        return false;
    };
    let Some(k) = json_index_offset(index, specials) else {
        return false;
    };
    json_slot_array_len(len, specials, const_locals, llbc).is_some_and(|n| k < n)
}

/// Build the rewrite, or say why this body keeps its bracket.
fn analyze(body: &Unstructured, llbc: &Llbc) -> Result<Option<Plan>, Refusal> {
    let n_blocks = body.body.len();
    let n_locals = body.locals.locals.len();
    let mut classified: HashMap<usize, (Leaf, bool)> = HashMap::new();
    for (bb, block) in body.body.iter().enumerate() {
        if let Ok(TermKind::Call { call, .. }) = block.term_ref(llbc)
            && let Some(class) = classify_call(call, llbc)
        {
            classified.insert(bb, class);
        }
    }
    if classified.is_empty() {
        return Ok(None);
    }
    if !llbc.stack_sensitive_fns_complete() {
        return Err("callee-stack-effects-unknown");
    }
    // A callee that leaves slots, indexes through a parameter or returns
    // an index refuses here, whether or not it also reads below entry:
    // after the rewrite nothing rewinds a leftover
    // (`_fix_graph_after_inlining`). A callee that only reads below entry
    // is checked against the entry depth in the walk below.
    for block in &body.body {
        if let Ok(TermKind::Call { call, .. }) = block.term_ref(llbc)
            && classify_call(call, llbc).is_none()
            && callee_path(call, llbc)
                .as_deref()
                .is_some_and(|path| callee_effect_of(llbc, path).refuses_erase())
        {
            return Err("calls-stack-sensitive-fn");
        }
    }
    if classified
        .values()
        .any(|(leaf, _)| leaf_unmodeled_for_erase(*leaf))
    {
        return Err("unmodeled-stack-op");
    }

    let mut plan = Plan {
        specials: HashMap::new(),
        removed: HashMap::new(),
        inserted_after: HashMap::new(),
        terms: HashMap::new(),
        slot_count: 0,
        slot_ty: None,
        unreachable: Vec::new(),
    };
    let bind =
        |specials: &mut HashMap<usize, Special>, local: usize, special: Special| match specials
            .insert(local, special)
        {
            Some(previous) if previous != special => Err("special-rebound"),
            _ => Ok(()),
        };
    let slot_local = |k: usize| n_locals + k;

    let def_counts = local_def_counts(body, llbc);
    let single_def = |dest: usize| def_counts.get(dest).copied() == Some(1);
    let mut const_locals: HashMap<usize, usize> = HashMap::new();
    let mut array_lens: HashMap<usize, usize> = HashMap::new();
    let mut depth_in: Vec<Option<usize>> = vec![None; n_blocks];
    let mut queue: VecDeque<usize> = VecDeque::new();
    if n_blocks == 0 {
        return Ok(None);
    }
    depth_in[0] = Some(0);
    queue.push_back(0);
    let mut visited = vec![false; n_blocks];
    let reach = |depth_in: &mut Vec<Option<usize>>,
                 queue: &mut VecDeque<usize>,
                 target: u64,
                 depth: usize|
     -> Result<(), Refusal> {
        let target = target as usize;
        if target >= n_blocks {
            return Ok(());
        }
        match depth_in[target] {
            Some(d) if d != depth => Err("depth-disagrees-at-merge"),
            Some(_) => Ok(()),
            None => {
                depth_in[target] = Some(depth);
                queue.push_back(target);
                Ok(())
            }
        }
    };

    while let Some(bb) = queue.pop_front() {
        if visited[bb] {
            continue;
        }
        visited[bb] = true;
        let mut depth = depth_in[bb].expect("queued blocks have a depth");
        let block = &body.body[bb];
        // Statements: the index, pair and alias definitions.
        for (si, stmt) in block.statements.iter().enumerate() {
            let Ok(StmtKind::Assign(place, rvalue)) = stmt.stmt_kind_ref() else {
                continue;
            };
            let Some(dest) = place_local(place) else {
                continue;
            };
            match rvalue {
                Rvalue::Aggregate(kind, operands) if kind.get("Array").is_some() => {
                    if single_def(dest) {
                        array_lens.insert(dest, operands.len());
                    }
                }
                Rvalue::Len(src) => {
                    let base = place_local(src).or_else(|| deref_of_local(src));
                    if single_def(dest)
                        && let Some(n) = base.and_then(|l| array_lens.get(&l).copied())
                    {
                        const_locals.insert(dest, n);
                    }
                }
                Rvalue::Ref { place: src, .. } | Rvalue::RawPtr { place: src, .. } => {
                    let base = place_local(src).or_else(|| deref_of_local(src));
                    if single_def(dest)
                        && let Some(n) = base.and_then(|l| array_lens.get(&l).copied())
                    {
                        array_lens.insert(dest, n);
                    }
                }
                // Cast / UnaryOp may change the value (narrowing, negation,
                // bit-not). `constfold.fold_op_list` evaluates them; this
                // pass does not, so the fact is dropped.
                Rvalue::Use(Operand::Copy(src) | Operand::Move(src), _) => {
                    if single_def(dest)
                        && let Some((pair, 1)) = tuple_field_of_local(src)
                        && let Some(n) = array_lens.get(&pair).copied()
                    {
                        const_locals.insert(dest, n);
                    }
                }
                _ => {}
            }
            let special = match rvalue {
                Rvalue::Ref { place: src, .. } | Rvalue::RawPtr { place: src, .. } => {
                    let guard = place_local(src)
                        .and_then(|l| match plan.specials.get(&l) {
                            Some(Special::Guard(_)) => Some(l),
                            _ => None,
                        })
                        .or_else(|| {
                            deref_of_local(src).and_then(|l| match plan.specials.get(&l) {
                                Some(Special::Alias(g)) => Some(*g),
                                _ => None,
                            })
                        });
                    guard.map(Special::Alias)
                }
                Rvalue::Use(op, _) => match op {
                    Operand::Copy(src) | Operand::Move(src) => {
                        if let Some(l) = place_local(src) {
                            if single_def(dest) {
                                if let Some(n) = const_locals.get(&l).copied() {
                                    const_locals.insert(dest, n);
                                }
                                if let Some(n) = array_lens.get(&l).copied() {
                                    array_lens.insert(dest, n);
                                }
                            }
                            match plan.specials.get(&l) {
                                Some(Special::Index(k)) => Some(Special::Index(*k)),
                                Some(Special::Alias(g)) => Some(Special::Alias(*g)),
                                Some(Special::Guard(d)) => Some(Special::Guard(*d)),
                                _ => None,
                            }
                        } else if let Some((pair, 0)) = tuple_field_of_local(src) {
                            match plan.specials.get(&pair) {
                                Some(Special::IndexPair(k)) => Some(Special::Index(*k)),
                                _ => None,
                            }
                        } else {
                            None
                        }
                    }
                    Operand::Const(_) => {
                        if single_def(dest)
                            && let Some(n) = const_usize(op, llbc)
                        {
                            const_locals.insert(dest, n);
                        }
                        None
                    }
                },
                Rvalue::BinaryOp(op, lhs, rhs) => match binop_is_index_arith(op) {
                    Some(arith) => {
                        let index_of = |o: &Operand| {
                            operand_local(o).and_then(|l| match plan.specials.get(&l) {
                                Some(Special::Index(k)) => Some(*k),
                                _ => None,
                            })
                        };
                        // `len + k` and `len - k` are compile-time offsets
                        // from the depth at the Len/Base. `zip_two_tuple_next`
                        // names a just-pinned slot as `shadow_stack_len() - 1`.
                        let offset = match (arith, index_of(lhs), index_of(rhs)) {
                            (IndexArith::Add { checked }, Some(k), None) => {
                                const_usize(rhs, llbc).map(|c| (k + c, checked))
                            }
                            (IndexArith::Add { checked }, None, Some(k)) => {
                                const_usize(lhs, llbc).map(|c| (k + c, checked))
                            }
                            (IndexArith::Sub { checked }, Some(k), None) => const_usize(rhs, llbc)
                                .and_then(|c| k.checked_sub(c).map(|n| (n, checked))),
                            _ => None,
                        };
                        offset.map(|(k, checked)| {
                            if checked {
                                Special::IndexPair(k)
                            } else {
                                Special::Index(k)
                            }
                        })
                    }
                    None => None,
                },
                _ => None,
            };
            if let Some(special) = special {
                bind(&mut plan.specials, dest, special)?;
                plan.removed.entry(bb).or_default().push(si);
            }
        }
        let term_json = block.terminator.kind_value();
        match block.term_ref(llbc) {
            Ok(TermKind::Call { call, target, .. }) => {
                let Some(&(leaf, method)) = classified.get(&bb) else {
                    // Observes at Known(0) reads the caller's slots.
                    // Inside our Open the callee reads our pins, which
                    // become slot locals (`eval_slice_index` / `getindex_w`).
                    if callee_path(call, llbc)
                        .as_deref()
                        .is_some_and(|path| callee_effect_of(llbc, path).observes)
                        && depth == 0
                    {
                        return Err("calls-stack-sensitive-fn");
                    }
                    reach(&mut depth_in, &mut queue, *target, depth)?;
                    continue;
                };
                let args_json = &term_json["Call"]["call"]["args"];
                let dest_json = &term_json["Call"]["call"]["dest"];
                let dest = place_local(&call.dest);
                let arg0 = usize::from(method);
                let guard = if method {
                    let receiver = call.args.first().and_then(operand_local);
                    match receiver.and_then(|l| plan.specials.get(&l)) {
                        Some(Special::Alias(g)) => Some(*g),
                        _ => return Err("receiver-not-a-guard-borrow"),
                    }
                } else {
                    None
                };
                let guard_depth = |g: usize| match plan.specials.get(&g) {
                    Some(Special::Guard(d)) => Ok(*d),
                    _ => Err("guard-not-opened"),
                };
                let index_arg = |i: usize| -> Result<usize, Refusal> {
                    match call
                        .args
                        .get(i)
                        .and_then(operand_local)
                        .and_then(|l| plan.specials.get(&l))
                    {
                        Some(Special::Index(k)) => Ok(*k),
                        _ => Err("index-not-static"),
                    }
                };
                let mut stmts = Vec::new();
                match leaf {
                    Leaf::Open => {
                        let dest = dest.ok_or("guard-dest-projected")?;
                        bind(&mut plan.specials, dest, Special::Guard(depth))?;
                    }
                    Leaf::Close => {
                        let d = guard_depth(guard.ok_or("close-without-guard")?)?;
                        if d > depth {
                            return Err("close-above-depth");
                        }
                        depth = d;
                    }
                    Leaf::Base => {
                        let d = guard_depth(guard.ok_or("base-without-guard")?)?;
                        bind(
                            &mut plan.specials,
                            dest.ok_or("index-dest-projected")?,
                            Special::Index(d),
                        )?;
                    }
                    Leaf::Len => {
                        bind(
                            &mut plan.specials,
                            dest.ok_or("index-dest-projected")?,
                            Special::Index(depth),
                        )?;
                    }
                    Leaf::Pin => {
                        let value = &args_json[arg0];
                        let ty = value
                            .as_object()
                            .and_then(|m| m.values().next())
                            .and_then(|p| p.get("ty"))
                            .cloned()
                            .ok_or("pin-value-untyped")?;
                        plan.slot_ty.get_or_insert(ty.clone());
                        let copied = copy_operand_json(value);
                        stmts.push(assign_json(
                            local_place_json(slot_local(depth), &ty),
                            json!({ "Use": [copied, "Yes"] }),
                        ));
                        stmts.push(assign_json(
                            dest_json.clone(),
                            json!({ "Use": [copied, "Yes"] }),
                        ));
                        depth += 1;
                        plan.slot_count = plan.slot_count.max(depth);
                    }
                    Leaf::Publish => {
                        let slice = call.args.get(arg0).and_then(operand_local);
                        let (agg_si, operands) = resolve_published_array(block, slice)
                            .ok_or("publish-array-unresolved")?;
                        let mut inserted = Vec::new();
                        for (i, element) in operands.iter().enumerate() {
                            let ty = element
                                .as_object()
                                .and_then(|m| m.values().next())
                                .and_then(|p| p.get("ty"))
                                .cloned()
                                .ok_or("publish-element-untyped")?;
                            plan.slot_ty.get_or_insert(ty.clone());
                            inserted.push(assign_json(
                                local_place_json(slot_local(depth + i), &ty),
                                json!({ "Use": [copy_operand_json(element), "Yes"] }),
                            ));
                        }
                        plan.inserted_after
                            .entry((bb, agg_si))
                            .or_default()
                            .extend(inserted);
                        if let Some(dest) = dest {
                            bind(&mut plan.specials, dest, Special::Index(depth))?;
                        }
                        depth += operands.len();
                        plan.slot_count = plan.slot_count.max(depth);
                    }
                    Leaf::Normalize => {}
                    Leaf::NormalizeMoved => {
                        let ty = dest_json["ty"].clone();
                        stmts.push(assign_json(
                            dest_json.clone(),
                            json!({ "Use": [{ "Const": [{ "Bool": false }, ty] }, "Yes"] }),
                        ));
                    }
                    Leaf::Get => {
                        let k = index_arg(arg0)?;
                        if k >= depth {
                            return Err("get-above-depth");
                        }
                        let ty = plan.slot_ty.clone().ok_or("get-before-any-pin")?;
                        stmts.push(assign_json(
                            dest_json.clone(),
                            json!({ "Use": [{ "Copy": local_place_json(slot_local(k), &ty) }, "Yes"] }),
                        ));
                    }
                    Leaf::Set => {
                        let k = index_arg(arg0)?;
                        if k >= depth {
                            return Err("set-above-depth");
                        }
                        let ty = plan.slot_ty.clone().ok_or("set-before-any-pin")?;
                        stmts.push(assign_json(
                            local_place_json(slot_local(k), &ty),
                            json!({ "Use": [copy_operand_json(&args_json[arg0 + 1]), "Yes"] }),
                        ));
                    }
                    Leaf::ReloadTop => {
                        if depth == 0 {
                            return Err("reload-empty-stack");
                        }
                        let ty = plan.slot_ty.clone().ok_or("reload-before-any-pin")?;
                        stmts.push(assign_json(
                            dest_json.clone(),
                            json!({ "Use": [{ "Copy": local_place_json(slot_local(depth - 1), &ty) }, "Yes"] }),
                        ));
                    }
                    Leaf::Unmodeled
                    | Leaf::CopyRange
                    | Leaf::ItemsOpen
                    | Leaf::ItemsPush
                    | Leaf::ItemsNop
                    | Leaf::ItemsGet
                    | Leaf::ItemsClose => unreachable!("refused above"),
                }
                plan.terms.insert(
                    bb,
                    TermRewrite {
                        target: *target,
                        stmts,
                    },
                );
                reach(&mut depth_in, &mut queue, *target, depth)?;
            }
            Ok(TermKind::Drop { place, target, .. }) => {
                let dropped = place_local(place);
                if let Some(Special::Guard(d)) = dropped.and_then(|l| plan.specials.get(&l)) {
                    // A drop flag the artefact does not carry can leave the
                    // guard unopened on this path; the depth says so.
                    if *d > depth {
                        return Err("close-above-depth");
                    }
                    depth = *d;
                    plan.terms.insert(
                        bb,
                        TermRewrite {
                            target: *target,
                            stmts: Vec::new(),
                        },
                    );
                    reach(&mut depth_in, &mut queue, *target, depth)?;
                } else if dropped.is_some_and(|l| is_root_scope_local(body, llbc, l)) {
                    return Err("drop-of-unopened-guard");
                } else {
                    reach(&mut depth_in, &mut queue, *target, depth)?;
                }
            }
            Ok(TermKind::Assert { assert, target, .. }) => {
                if assert_is_slot_index_bounds_check(assert, &plan.specials, &const_locals, llbc) {
                    plan.terms.insert(
                        bb,
                        TermRewrite {
                            target: *target,
                            stmts: Vec::new(),
                        },
                    );
                }
                reach(&mut depth_in, &mut queue, *target, depth)?;
            }
            Ok(TermKind::Goto { target }) => reach(&mut depth_in, &mut queue, *target, depth)?,
            Ok(TermKind::Switch { targets, .. }) => match targets {
                majit_charon_reader::ullbc::SwitchTargets::If(a, b) => {
                    reach(&mut depth_in, &mut queue, *a, depth)?;
                    reach(&mut depth_in, &mut queue, *b, depth)?;
                }
                majit_charon_reader::ullbc::SwitchTargets::SwitchInt(_, arms, default) => {
                    for (_, t) in arms {
                        reach(&mut depth_in, &mut queue, *t, depth)?;
                    }
                    reach(&mut depth_in, &mut queue, *default, depth)?;
                }
            },
            Ok(TermKind::Return) => {
                if depth != 0 {
                    // 10.04 `Drop` of the guard lives in an `is_cleanup`
                    // block; `unstructured` leaves those out and appends
                    // one `UnwindResume`. Return is then the normal closer.
                    if body
                        .body
                        .last()
                        .is_some_and(|bb| matches!(bb.term(llbc), Ok(TermKind::UnwindResume)))
                    {
                        depth = 0;
                    } else {
                        return Err("returns-with-published-slots");
                    }
                }
            }
            Ok(
                TermKind::UnwindResume
                | TermKind::UnwindTerminate
                | TermKind::Abort(_)
                | TermKind::Panic { .. }
                | TermKind::UndefinedBehavior,
            ) => {}
            _ => return Err("unknown-terminator"),
        }
    }

    // Local 0 is the return place. An Index bound there is a helper that
    // answers a slot (`pin_self`: pin, then `shadow_stack_len() - 1`).
    // The Sub/Len statements that defined it are stripped as specials, and
    // `Return` does not mention local 0, so the rewritten body would return
    // an unbound local. Callers keep that index (`from_slot`) and lower the
    // call as `getattr(recv, method)` (`CallTarget::Method`); a broken
    // callee graph blocks that getattr. Refuse: the Pin lives in the
    // caller's RootScope and is only erasable if the helper is inlined.
    if matches!(
        plan.specials.get(&0),
        Some(Special::Index(_) | Special::IndexPair(_))
    ) {
        return Err("returns-slot-index");
    }

    // Every surviving mention of a guard, borrow or index has to be one of the
    // shapes rewritten above; anything else would read a local no block binds.
    let mut watched = bit_set::BitSet::new();
    for local in plan.specials.keys() {
        watched.insert(*local);
    }
    for (bb, block) in body.body.iter().enumerate() {
        if !visited[bb] {
            continue;
        }
        let removed = plan.removed.get(&bb);
        for (si, stmt) in block.statements.iter().enumerate() {
            if removed.is_some_and(|r| r.contains(&si)) {
                continue;
            }
            if matches!(
                stmt.stmt_kind_ref(),
                Ok(StmtKind::StorageLive(_) | StmtKind::StorageDead(_) | StmtKind::Borrowck(_))
            ) {
                continue;
            }
            if super::mentions_local(stmt.kind_value(), &watched) {
                return Err("unmodeled-use-in-statement");
            }
        }
        if plan.terms.contains_key(&bb) {
            continue;
        }
        if super::mentions_local(block.terminator.kind_value(), &watched) {
            // BoundsCheck / overflow asserts on erased slot-index
            // arithmetic (`shadow_stack_len` + `+ k`, or the array
            // index of `publish_roots(&[...])`). Failure is the
            // panic edge; the offsets are compile-time so the check
            // is statically true. Guard/Alias mentions stay: rewriting
            // those Goto'd past Drop of a surviving RootScope.
            let target = match block.term_ref(llbc) {
                Ok(TermKind::Assert { assert, target, .. })
                    if assert_is_slot_index_bounds_check(
                        assert,
                        &plan.specials,
                        &const_locals,
                        llbc,
                    ) =>
                {
                    *target
                }
                _ => return Err("unmodeled-use-in-terminator"),
            };
            plan.terms.insert(
                bb,
                TermRewrite {
                    target,
                    stmts: Vec::new(),
                },
            );
        }
    }
    plan.unreachable = (0..n_blocks).filter(|bb| !visited[*bb]).collect();
    Ok(Some(plan))
}

fn leaf_unmodeled_for_erase(leaf: Leaf) -> bool {
    matches!(
        leaf,
        Leaf::Unmodeled
            | Leaf::CopyRange
            | Leaf::ItemsOpen
            | Leaf::ItemsPush
            | Leaf::ItemsNop
            | Leaf::ItemsGet
            | Leaf::ItemsClose
    )
}

fn is_rooted_items_local(body: &Unstructured, llbc: &Llbc, local: usize) -> bool {
    let Some(decl) = body.locals.locals.get(local) else {
        return false;
    };
    super::output_adt_def_id_free(&decl.ty, llbc)
        .and_then(|id| llbc.type_by_id(id))
        .is_some_and(|t| {
            let path = t.item_meta.name_path();
            path.rsplit("::").next() == Some("RootedItems")
                && path.split("::").any(|s| s == super::ROOT_SCOPE_MODULE)
        })
}

pub(super) fn is_root_scope_local(body: &Unstructured, llbc: &Llbc, local: usize) -> bool {
    let Some(decl) = body.locals.locals.get(local) else {
        return false;
    };
    super::output_adt_def_id_free(&decl.ty, llbc)
        .and_then(|id| llbc.type_by_id(id))
        .is_some_and(|t| super::gc_root_scope_type_path(&t.item_meta.name_path()))
}

/// Follow a published slice back to the array literal it borrows, in the same
/// block: `_a = [x, y]; _r = &_a; _s = &*_r; _p = _s as &[_]`.  Returns the
/// literal's statement index and its element operands.
///
/// Statement JSON is read off this block, the CFG after unwind-only blocks
/// are dropped, so the index matches the rewrite.
fn resolve_published_array(
    block: &majit_charon_reader::ullbc::BasicBlock,
    slice: Option<usize>,
) -> Option<(usize, Vec<Value>)> {
    let mut want = slice?;
    for (si, stmt) in block.statements.iter().enumerate().rev() {
        let Ok(StmtKind::Assign(place, rvalue)) = stmt.stmt_kind_ref() else {
            continue;
        };
        if place_local(place) != Some(want) {
            continue;
        }
        match rvalue {
            Rvalue::UnaryOp(_, op) | Rvalue::Use(op, _) | Rvalue::Cast(_, op, _) => {
                want = operand_local(op)?;
            }
            Rvalue::Ref { place: src, .. } => {
                want = place_local(src).or_else(|| deref_of_local(src))?;
            }
            Rvalue::Aggregate(kind, _) if kind.get("Array").is_some() => {
                let operands = stmt.kind_value()["Assign"][1]["Aggregate"][1]
                    .as_array()?
                    .clone();
                return Some((si, operands));
            }
            _ => return None,
        }
    }
    None
}

/// Rewrite `fd`'s body with its root brackets scalar-replaced.
///
/// `Ok(None)`: the body has no bracket.  `Err`: it has one this pass refuses;
/// the reason names the shape.
pub(super) fn erase_shadow_stack(
    fd: &FunDecl,
    body: &Unstructured,
    llbc: &Llbc,
) -> Result<Option<Unstructured>, Refusal> {
    if fd.body.is_none() {
        return Ok(None);
    }
    // The root-stack runtime itself implements the leaves this pass removes;
    // its bodies stay as written for the callers that keep their brackets.
    if fd.item_meta.name_path().contains("::gc_roots::") {
        return Ok(None);
    }
    // `body` is the CFG after unwind-only blocks are dropped (and promoted
    // constants spliced).  The rewrite has to use that numbering: the
    // extracted artefact still has those blocks, and mixing the two is
    // how a same-block array publish was refused as unresolved.
    let mut u = unstructured_json_from_body(body);
    let plan = match analyze(body, llbc)? {
        Some(plan) => plan,
        None => return Ok(None),
    };
    let span = u["span"].clone();
    // The slot locals.
    if plan.slot_count > 0 {
        let ty = plan.slot_ty.clone().ok_or("slot-untyped")?;
        let locals = u["locals"]["locals"].as_array_mut().ok_or("body-json")?;
        let base = locals.len();
        for k in 0..plan.slot_count {
            locals.push(json!({
                "index": base + k,
                "name": null,
                "span": span,
                "ty": ty,
            }));
        }
    }
    let blocks = u["body"].as_array_mut().ok_or("body-json")?;
    for (bb, block) in blocks.iter_mut().enumerate() {
        // An unreachable block may still name a guard or an index whose
        // definition is gone; nothing runs it.
        if plan.unreachable.contains(&bb) {
            block["statements"] = Value::Array(Vec::new());
            block["terminator"]["kind"] = json!({ "Abort": "UnwindTerminate" });
            continue;
        }
        let removed = plan.removed.get(&bb);
        let term = plan.terms.get(&bb);
        let has_inserts = plan.inserted_after.keys().any(|(b, _)| *b == bb);
        if removed.is_none() && term.is_none() && !has_inserts {
            continue;
        }
        let stmt_span = block["terminator"]["span"].clone();
        let make_stmt = |kind: Value| {
            json!({
                "span": stmt_span,
                "kind": kind,
                "comments_before": [],
            })
        };
        let old = block["statements"].as_array().cloned().unwrap_or_default();
        let mut new = Vec::with_capacity(old.len());
        for (si, stmt) in old.into_iter().enumerate() {
            if !removed.is_some_and(|r| r.contains(&si)) {
                new.push(stmt);
            }
            if let Some(extra) = plan.inserted_after.get(&(bb, si)) {
                new.extend(extra.iter().cloned().map(make_stmt));
            }
        }
        if let Some(term) = term {
            new.extend(term.stmts.iter().cloned().map(make_stmt));
            block["terminator"]["kind"] = json!({ "Goto": { "target": term.target } });
        }
        block["statements"] = Value::Array(new);
    }
    serde_json::from_value::<Unstructured>(u)
        .map(Some)
        .map_err(|_| "rewritten-body-unparsable")
}

fn span_ref_json(span: &SpanRef) -> Value {
    match span {
        SpanRef::Inline(data) => json!({
            "data": {
                "file_id": data.file_id,
                "beg": { "line": data.beg.line, "col": data.beg.col },
                "end": { "line": data.end.line, "col": data.end.col },
            }
        }),
        SpanRef::Deduplicated(id) => json!({ "Deduplicated": id }),
    }
}

fn ty_ref_json(ty: &TyRef) -> Value {
    match ty {
        TyRef::Dedup { id } => json!({ "Deduplicated": id }),
        TyRef::Inline { value: (id, v) } => json!({ "Value": [*id, v] }),
        TyRef::Other(v) => v.clone(),
    }
}

/// JSON for `body` as the rewrite sees it: same blocks, same statement
/// indices, same terminator targets as the typed CFG.
fn unstructured_json_from_body(body: &Unstructured) -> Value {
    let locals: Vec<Value> = body
        .locals
        .locals
        .iter()
        .map(|loc| {
            json!({
                "index": loc.index,
                "name": loc.name,
                "span": span_ref_json(&loc.span),
                "ty": ty_ref_json(&loc.ty),
            })
        })
        .collect();
    let blocks: Vec<Value> = body
        .body
        .iter()
        .map(|bb| {
            let statements: Vec<Value> = bb
                .statements
                .iter()
                .map(|st| {
                    json!({
                        "span": span_ref_json(&st.span),
                        "kind": st.kind_value().clone(),
                    })
                })
                .collect();
            json!({
                "statements": statements,
                "terminator": {
                    "span": bb.terminator.span.as_ref().map(span_ref_json),
                    "kind": bb.terminator.kind_value().clone(),
                },
                "is_cleanup": bb.is_cleanup,
            })
        })
        .collect();
    json!({
        "span": span_ref_json(&body.span),
        "locals": {
            "arg_count": body.locals.arg_count,
            "locals": locals,
        },
        "body": blocks,
    })
}

/// [`erase_shadow_stack`], falling back to the body as extracted.  A refusal is
/// counted under its reason.
pub(super) fn erase_or_keep(fd: &FunDecl, body: Unstructured, llbc: &Llbc) -> Unstructured {
    match erase_shadow_stack(fd, &body, llbc) {
        Ok(Some(erased)) => erased,
        Ok(None) => body,
        Err(reason) => {
            crate::decline::record(
                crate::decline::gate::SHADOW_STACK_ERASE,
                reason,
                format_args!("{}", fd.item_meta.name_path()),
            );
            body
        }
    }
}

/// Census-path only: every local body's walk under the registered
/// effects must equal the registered effect. A registered Observes
/// whose own walk is clean (`why=-`) is the sticky-fixpoint bug.
fn debug_check_registered_effects_match_rewalk(llbc: &Llbc) {
    let effect = |path: &str| callee_effect_of(llbc, path);
    for fd in llbc.iter_local_fns() {
        let Some(body) = fd.unstructured() else {
            continue;
        };
        let name = fd.item_meta.name_path();
        let registered = effect(&name);
        let walked = stack_walk(&body, llbc, &effect).effect();
        if registered != walked {
            panic!(
                "stack-effect invariant: {name} registered {} rewalk {} why={}",
                registered.as_str(),
                walked.as_str(),
                stack_walk(&body, llbc, &effect)
                    .why
                    .as_deref()
                    .unwrap_or("-")
            );
        }
    }
}

/// Every local body with a root bracket, bucketed by what the scalar
/// replacement did with it: `"erased"`, or the refusal reason.  The subject
/// list names the bodies in each bucket.
pub fn census(llbc: &Llbc) -> std::collections::BTreeMap<&'static str, Vec<String>> {
    ensure_stack_sensitive_fns(llbc);
    debug_check_registered_effects_match_rewalk(llbc);
    let detail = std::env::var("CENSUS_DETAIL").is_ok();
    let mut out: std::collections::BTreeMap<&'static str, Vec<String>> = Default::default();
    for fd in llbc.iter_local_fns() {
        let Some(body) = fd.unstructured() else {
            continue;
        };
        let bucket = match erase_shadow_stack(fd, &body, llbc) {
            Ok(Some(_)) => "erased",
            Ok(None) => continue,
            Err(reason) => reason,
        };
        let name = fd.item_meta.name_path();
        if detail {
            let walk = stack_walk(&body, llbc, &|path| callee_effect_of(llbc, path));
            let callee = first_non_none_callee(&body, llbc);
            eprintln!(
                "[census-detail] {name}:{bucket} why={} callee={} effect={}",
                walk.why.as_deref().unwrap_or("-"),
                callee.as_ref().map(|(p, _)| p.as_str()).unwrap_or("-"),
                callee
                    .as_ref()
                    .map(|(_, effect)| effect.as_str())
                    .unwrap_or("-"),
            );
        }
        out.entry(bucket).or_default().push(name);
    }
    out
}

// ---------------------------------------------------------------------------
// Stack effects a caller cannot see past
// ---------------------------------------------------------------------------

/// Stack depth relative to a body's entry.
/// `Known(d)` is exact. `AtLeast(d)` is a lower bound: a publish of
/// unknown length or a LeavesAbove callee ran at that depth.
/// `Unknown` is no bound (a Close of an unknown guard, or a join of
/// incompatible depths).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Depth {
    Known(usize),
    /// Lower bound relative to entry. `Above` was `AtLeast(1)`.
    AtLeast(usize),
    Unknown,
}

impl Depth {
    /// The body holds at least one slot of its own.
    fn above_entry(self) -> bool {
        matches!(self, Depth::Known(1..) | Depth::AtLeast(1..))
    }

    fn lower_bound(self) -> Option<usize> {
        match self {
            Depth::Known(d) | Depth::AtLeast(d) => Some(d),
            Depth::Unknown => None,
        }
    }

    fn join(self, other: Depth) -> Depth {
        if self == other {
            self
        } else {
            match (self.lower_bound(), other.lower_bound()) {
                (Some(a), Some(b)) => {
                    let m = a.min(b);
                    if self == Depth::Known(m) && other == Depth::Known(m) {
                        Depth::Known(m)
                    } else {
                        Depth::AtLeast(m)
                    }
                }
                _ => Depth::Unknown,
            }
        }
    }

    fn add(self, n: usize) -> Depth {
        match self {
            Depth::Known(d) => Depth::Known(d + n),
            Depth::AtLeast(d) => Depth::AtLeast(d + n),
            Depth::Unknown => Depth::Unknown,
        }
    }

    fn checked_sub(self, n: usize) -> Option<Depth> {
        match self {
            Depth::Known(d) => d.checked_sub(n).map(Depth::Known),
            Depth::AtLeast(_) | Depth::Unknown => Some(Depth::Unknown),
        }
    }

    /// The depth after a callee that may leave slots behind, or a
    /// publish of unknown length: `Known(d)` becomes `AtLeast(d)`.
    fn after_unknown_push(self) -> Depth {
        match self {
            Depth::Known(d) | Depth::AtLeast(d) => Depth::AtLeast(d),
            Depth::Unknown => Depth::Unknown,
        }
    }
}

/// Effect of a callee on the caller's shadow stack.
/// `observes` and `leaves_above` are independent: a body that reads
/// below entry can also return with slots still published, and a caller
/// whose bracket is erased has no Close left to rewind them.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
struct CalleeEffect {
    observes: bool,
    leaves_above: bool,
    param_slots: Vec<u8>,
    /// Every reachable `Return` yields an own index (>= entry).
    returns_index: bool,
}

impl CalleeEffect {
    fn none() -> Self {
        Self::default()
    }

    /// Reads below entry and returns at entry depth on every path.
    #[cfg(test)]
    fn observes() -> Self {
        Self {
            observes: true,
            leaves_above: false,
            param_slots: Vec::new(),
            returns_index: false,
        }
    }

    /// Reads below entry and may end at any depth.
    fn worst() -> Self {
        Self {
            observes: true,
            leaves_above: true,
            param_slots: Vec::new(),
            returns_index: false,
        }
    }

    /// The erased caller cannot keep this call: something the callee
    /// publishes or indexes outlives it, and nothing rewinds a leftover
    /// once the caller's Open / Close are gone.
    fn refuses_erase(&self) -> bool {
        self.leaves_above || !self.param_slots.is_empty() || self.returns_index
    }

    #[allow(dead_code)]
    fn leaves_above() -> Self {
        Self {
            observes: false,
            leaves_above: true,
            param_slots: Vec::new(),
            returns_index: false,
        }
    }

    fn is_none(&self) -> bool {
        !self.observes && !self.leaves_above && self.param_slots.is_empty() && !self.returns_index
    }

    fn as_str(&self) -> &'static str {
        if self.observes && self.leaves_above {
            "ObservesLeaves"
        } else if self.observes {
            "Observes"
        } else if !self.param_slots.is_empty() {
            "ParamSlots"
        } else if self.leaves_above {
            "LeavesAbove"
        } else if self.returns_index {
            "ReturnsIndex"
        } else {
            "None"
        }
    }

    #[allow(dead_code)]
    fn join(&self, other: &Self) -> Self {
        let mut param_slots = self.param_slots.clone();
        for &p in &other.param_slots {
            if !param_slots.contains(&p) {
                param_slots.push(p);
            }
        }
        param_slots.sort_unstable();
        Self {
            observes: self.observes || other.observes,
            leaves_above: self.leaves_above || other.leaves_above,
            param_slots,
            returns_index: self.returns_index || other.returns_index,
        }
    }
}

/// Registered effect of `path`: `None` if not sensitive or proven
/// neutral; otherwise the union of the sets it was registered in. A
/// sensitive path in none of them comes from an artefact that did not
/// classify it, and is taken at its worst.
fn callee_effect_of(llbc: &Llbc, path: &str) -> CalleeEffect {
    if llbc.is_stack_depth_neutral_fn(path) {
        return CalleeEffect::none();
    }
    let param_slots = llbc.stack_param_slots(path).unwrap_or_default();
    let leaves = llbc.is_stack_leaves_above_fn(path);
    let returns_index = llbc.is_stack_returns_index_fn(path);
    let observes = llbc.is_stack_observes_fn(path);
    let effect = CalleeEffect {
        observes,
        leaves_above: leaves,
        param_slots,
        returns_index,
    };
    if effect.is_none() && llbc.is_stack_sensitive_fn(path) {
        return CalleeEffect::worst();
    }
    effect
}

fn first_non_none_callee(body: &Unstructured, llbc: &Llbc) -> Option<(String, CalleeEffect)> {
    for block in &body.body {
        if let Ok(TermKind::Call { call, .. }) = block.term_ref(llbc)
            && classify_call(call, llbc).is_none()
            && let Some(path) = callee_path(call, llbc)
        {
            let effect = callee_effect_of(llbc, &path);
            if !effect.is_none() {
                return Some((path, effect));
            }
        }
    }
    None
}

/// The callee path a call terminator names, for a statically resolved callee.
fn callee_path(call: &majit_charon_reader::ullbc::CallPayload, llbc: &Llbc) -> Option<String> {
    let majit_charon_reader::ullbc::CallFunc::Regular(reg) = &call.func else {
        return None;
    };
    let id = super::regular_call_fun_decl_id(&reg.kind)?;
    llbc.fn_by_id(id).map(|fd| fd.item_meta.name_path())
}

/// Result of the per-path shadow-stack walk.
struct StackWalk {
    /// `None` when every reachable return / unwind restores the entry
    /// depth and no instruction reads or writes a slot below it.
    why: Option<String>,
    /// A reachable block opened a `RootScope`. Unreachable open/close
    /// pairs do not set this.
    saw_open: bool,
    /// Some instruction reads or writes a slot below entry.
    reads_below: bool,
    /// Some reachable return / unwind is `Known(d>=1)` or `AtLeast(_)`.
    leaves_above: bool,
    /// Some reachable return / unwind is `Unknown`, or the walk bailed
    /// because depth grew in a loop.
    unknown_exit: bool,
    /// Parameter positions this body uses as shadow-stack indices.
    param_slots: Vec<u8>,
    /// Every reachable `Return` yields a local-0 own index (the value
    /// itself or a struct field) with a lower bound relative to entry.
    returns_index: bool,
}

impl StackWalk {
    fn effect(&self) -> CalleeEffect {
        // An exit the walk could not bound may sit below entry (the
        // caller's slots are gone) or above it (slots are left), so it
        // sets both.
        let observes = self.reads_below || self.unknown_exit;
        CalleeEffect {
            observes,
            leaves_above: self.leaves_above || self.unknown_exit,
            param_slots: if observes {
                Vec::new()
            } else {
                self.param_slots.clone()
            },
            returns_index: self.returns_index && !observes,
        }
    }
}

/// Depth-neutral: every reachable path restores the entry depth, the body
/// never reads a caller-owned slot, Observes callees are refused only at
/// entry depth (inside our Open they see our pins; Close rewinds),
/// LeavesAbove is allowed, no parameter-indexed slot access, no returned
/// own-index summary, and a `RootScope` opens on some reachable path.
fn body_is_depth_neutral(
    body: &Unstructured,
    llbc: &Llbc,
    effect: &dyn Fn(&str) -> CalleeEffect,
) -> bool {
    let walk = stack_walk(body, llbc, effect);
    walk.why.is_none() && walk.saw_open && walk.param_slots.is_empty() && !walk.returns_index
}

fn clear_tracked(
    dest: usize,
    params: &mut HashMap<usize, u8>,
    param_pairs: &mut HashMap<usize, u8>,
    indices: &mut HashMap<usize, Depth>,
    pairs: &mut HashMap<usize, Depth>,
    field_indices: &mut HashMap<(usize, u64), Depth>,
    field_params: &mut HashMap<(usize, u64), u8>,
) {
    params.remove(&dest);
    param_pairs.remove(&dest);
    indices.remove(&dest);
    pairs.remove(&dest);
    field_indices.retain(|&(l, _), _| l != dest);
    field_params.retain(|&(l, _), _| l != dest);
}

fn copy_field_maps(
    src: usize,
    dest: usize,
    field_indices: &mut HashMap<(usize, u64), Depth>,
    field_params: &mut HashMap<(usize, u64), u8>,
) {
    let idx: Vec<(u64, Depth)> = field_indices
        .iter()
        .filter(|((l, _), _)| *l == src)
        .map(|((_, f), k)| (*f, *k))
        .collect();
    for (f, k) in idx {
        field_indices.insert((dest, f), k);
    }
    let ps: Vec<(u64, u8)> = field_params
        .iter()
        .filter(|((l, _), _)| *l == src)
        .map(|((_, f), p)| (*f, *p))
        .collect();
    for (f, p) in ps {
        field_params.insert((dest, f), p);
    }
}

fn stack_walk(
    body: &Unstructured,
    llbc: &Llbc,
    effect: &dyn Fn(&str) -> CalleeEffect,
) -> StackWalk {
    let n_blocks = body.body.len();
    if n_blocks == 0 {
        return StackWalk {
            why: None,
            saw_open: false,
            reads_below: false,
            leaves_above: false,
            unknown_exit: false,
            param_slots: Vec::new(),
            returns_index: false,
        };
    }
    let mut why = String::new();
    let mut saw_open = false;
    let mut depth_in: Vec<Option<Depth>> = vec![None; n_blocks];
    let mut guards: HashMap<usize, Depth> = HashMap::new();
    let mut aliases: HashMap<usize, usize> = HashMap::new();
    let mut indices: HashMap<usize, Depth> = HashMap::new();
    let mut pairs: HashMap<usize, Depth> = HashMap::new();
    let mut params: HashMap<usize, u8> = HashMap::new();
    let mut param_pairs: HashMap<usize, u8> = HashMap::new();
    let mut field_indices: HashMap<(usize, u64), Depth> = HashMap::new();
    let mut field_params: HashMap<(usize, u64), u8> = HashMap::new();
    let arg_count = body.locals.arg_count as usize;
    for i in 1..=arg_count {
        if let Ok(p) = u8::try_from(i - 1) {
            params.insert(i, p);
        }
    }
    let mut param_slots: Vec<u8> = Vec::new();
    depth_in[0] = Some(Depth::Known(0));
    let mut queue: VecDeque<usize> = VecDeque::from([0]);
    let mut reads_below = false;
    // The first read below entry. The walk goes on past it so the exits
    // are still judged: a reader that also leaves slots is a different
    // callee from one that returns at its entry depth.
    let mut read_why: Option<String> = None;
    let mut leaves_above = false;
    let mut unknown_exit = false;
    let mut returns_index: Option<bool> = None;
    let mut rounds = 0usize;
    while let Some(bb) = queue.pop_front() {
        rounds += 1;
        if rounds > n_blocks * 8 + 64 {
            // A depth that keeps growing round a loop: slots pile up.
            return StackWalk {
                why: Some(read_why.unwrap_or_else(|| "depth-grows-in-loop".into())),
                saw_open,
                reads_below,
                leaves_above,
                unknown_exit: true,
                param_slots,
                returns_index: false,
            };
        }
        let mut depth = depth_in[bb].expect("queued blocks have a depth");
        let block = &body.body[bb];
        for stmt in &block.statements {
            let Ok(StmtKind::Assign(place, rvalue)) = stmt.stmt_kind_ref() else {
                continue;
            };
            let Some(dest) = place_local(place) else {
                continue;
            };
            match rvalue {
                Rvalue::Ref { place: src, .. } | Rvalue::RawPtr { place: src, .. } => {
                    clear_tracked(
                        dest,
                        &mut params,
                        &mut param_pairs,
                        &mut indices,
                        &mut pairs,
                        &mut field_indices,
                        &mut field_params,
                    );
                    let guard = place_local(src)
                        .filter(|l| guards.contains_key(l))
                        .or_else(|| deref_of_local(src).and_then(|l| aliases.get(&l).copied()));
                    if let Some(g) = guard {
                        aliases.insert(dest, g);
                    }
                    // A borrow of an own index (or of a struct that
                    // carries one) still names that index, so a method
                    // call on `storage` can pass the `&mut` to a
                    // ParamSlots callee.
                    if let Some(l) = place_local(src).or_else(|| deref_of_local(src)) {
                        if let Some(&k) = indices.get(&l) {
                            indices.insert(dest, k);
                        }
                        if let Some(&p) = params.get(&l) {
                            params.insert(dest, p);
                        }
                        copy_field_maps(l, dest, &mut field_indices, &mut field_params);
                    }
                }
                Rvalue::Use(Operand::Copy(src) | Operand::Move(src), _) => {
                    let mut param_from = None;
                    let mut index_from = None;
                    let mut fields_from = None;
                    if let Some(l) = place_local(src) {
                        index_from = indices.get(&l).copied();
                        if let Some(g) = aliases.get(&l).copied() {
                            aliases.insert(dest, g);
                        }
                        if let Some(d) = guards.get(&l).copied() {
                            let entry = guards.entry(dest).or_insert(d);
                            *entry = entry.join(d);
                        }
                        param_from = params.get(&l).copied();
                        fields_from = Some(l);
                    } else if let Some((base, field)) = tuple_field_of_local(src) {
                        // Checked-add pairs: `.0` is the sum. Unary ADT
                        // wrappers (Option::Some of an own index): the
                        // payload field is the same usize, so it stays
                        // >= entry. A usize field of a parameter is a
                        // value the caller supplied, so it records that
                        // parameter in param_slots when used as a Get.
                        if field == 0 {
                            index_from = pairs.get(&base).copied();
                            param_from = param_pairs.get(&base).copied();
                        }
                        if index_from.is_none() {
                            index_from = field_indices
                                .get(&(base, field))
                                .copied()
                                .or_else(|| indices.get(&base).copied());
                        }
                        if param_from.is_none() {
                            param_from = field_params
                                .get(&(base, field))
                                .copied()
                                .or_else(|| params.get(&base).copied());
                        }
                    }
                    clear_tracked(
                        dest,
                        &mut params,
                        &mut param_pairs,
                        &mut indices,
                        &mut pairs,
                        &mut field_indices,
                        &mut field_params,
                    );
                    if let Some(k) = index_from {
                        indices.insert(dest, k);
                    }
                    if let Some(p) = param_from {
                        params.insert(dest, p);
                    }
                    if let Some(src_local) = fields_from {
                        copy_field_maps(src_local, dest, &mut field_indices, &mut field_params);
                    }
                }
                Rvalue::Aggregate(_, operands) => {
                    clear_tracked(
                        dest,
                        &mut params,
                        &mut param_pairs,
                        &mut indices,
                        &mut pairs,
                        &mut field_indices,
                        &mut field_params,
                    );
                    // Each operand that is an own index stays one in that
                    // field: the struct stores the same usize. A unary wrap
                    // (Option::Some) is the same fact on the dest itself.
                    for (i, op) in operands.iter().enumerate() {
                        let Some(l) = operand_local(op) else {
                            continue;
                        };
                        if let Some(&k) = indices.get(&l) {
                            field_indices.insert((dest, i as u64), k);
                        }
                        if let Some(&p) = params.get(&l) {
                            field_params.insert((dest, i as u64), p);
                        }
                    }
                    if let [op] = operands.as_slice() {
                        if let Some(k) = operand_local(op).and_then(|l| indices.get(&l).copied()) {
                            indices.insert(dest, k);
                        } else if let Some(p) =
                            operand_local(op).and_then(|l| params.get(&l).copied())
                        {
                            params.insert(dest, p);
                        }
                    }
                }
                Rvalue::BinaryOp(op, lhs, rhs) => {
                    clear_tracked(
                        dest,
                        &mut params,
                        &mut param_pairs,
                        &mut indices,
                        &mut pairs,
                        &mut field_indices,
                        &mut field_params,
                    );
                    if let Some(arith) = binop_is_index_arith(op) {
                        let base =
                            |o: &Operand| operand_local(o).and_then(|l| indices.get(&l).copied());
                        // Unsigned add cannot produce a smaller index:
                        // Known(k)+x stays AtLeast(k), param p+x stays param p.
                        let offset = match (arith, base(lhs), base(rhs)) {
                            (IndexArith::Add { checked }, Some(k), _) => {
                                Some((add_unsigned_index(k, rhs, llbc), checked))
                            }
                            (IndexArith::Add { checked }, None, Some(k)) => {
                                Some((add_unsigned_index(k, lhs, llbc), checked))
                            }
                            (IndexArith::Sub { checked }, Some(k), None) => const_usize(rhs, llbc)
                                .and_then(|c| k.checked_sub(c).map(|n| (n, checked))),
                            _ => None,
                        };
                        if let Some((k, checked)) = offset {
                            if checked {
                                pairs.insert(dest, k);
                            } else {
                                indices.insert(dest, k);
                            }
                        } else {
                            let param_of = |o: &Operand| {
                                operand_local(o).and_then(|l| params.get(&l).copied())
                            };
                            let kept = match (arith, param_of(lhs), param_of(rhs)) {
                                (IndexArith::Add { checked }, Some(p), _) => Some((p, checked)),
                                (IndexArith::Add { checked }, None, Some(p)) => Some((p, checked)),
                                _ => None,
                            };
                            if let Some((p, checked)) = kept {
                                if checked {
                                    param_pairs.insert(dest, p);
                                } else {
                                    params.insert(dest, p);
                                }
                            }
                        }
                    }
                }
                _ => {
                    clear_tracked(
                        dest,
                        &mut params,
                        &mut param_pairs,
                        &mut indices,
                        &mut pairs,
                        &mut field_indices,
                        &mut field_params,
                    );
                }
            }
        }
        // `None`: the block ends the walk.  Otherwise the successors and the
        // depth they receive.
        let mut successors: Vec<u64> = Vec::new();
        match block.term_ref(llbc) {
            Ok(TermKind::Call { call, target, .. }) => {
                successors.push(*target);
                if let Some(dest) = place_local(&call.dest) {
                    clear_tracked(
                        dest,
                        &mut params,
                        &mut param_pairs,
                        &mut indices,
                        &mut pairs,
                        &mut field_indices,
                        &mut field_params,
                    );
                }
                let path = callee_path(call, llbc);
                match classify_call(call, llbc) {
                    Some((leaf, method)) => {
                        let arg0 = usize::from(method);
                        let guard = call
                            .args
                            .first()
                            .and_then(operand_local)
                            .and_then(|l| aliases.get(&l).copied());
                        let items_guard = call.args.first().and_then(operand_local).and_then(|l| {
                            aliases
                                .get(&l)
                                .copied()
                                .or_else(|| guards.contains_key(&l).then_some(l))
                        });
                        match leaf {
                            Leaf::Open | Leaf::ItemsOpen => {
                                saw_open = true;
                                if let Some(dest) = place_local(&call.dest) {
                                    let entry = guards.entry(dest).or_insert(depth);
                                    *entry = entry.join(depth);
                                }
                            }
                            Leaf::Close | Leaf::ItemsClose => {
                                let g = if matches!(leaf, Leaf::ItemsClose) {
                                    items_guard
                                } else {
                                    guard
                                };
                                depth = g
                                    .and_then(|g| guards.get(&g).copied())
                                    .unwrap_or(Depth::Unknown);
                            }
                            Leaf::Base => {
                                if let (Some(dest), Some(g)) = (place_local(&call.dest), guard) {
                                    indices.insert(
                                        dest,
                                        guards.get(&g).copied().unwrap_or(Depth::Unknown),
                                    );
                                }
                            }
                            Leaf::Len => {
                                if let Some(dest) = place_local(&call.dest) {
                                    indices.insert(dest, depth);
                                }
                            }
                            Leaf::Pin | Leaf::ItemsPush => depth = depth.add(1),
                            Leaf::Publish => {
                                let slice = call.args.get(arg0).and_then(operand_local);
                                if let Some(dest) = place_local(&call.dest) {
                                    indices.insert(dest, depth);
                                }
                                match resolve_published_array_len(block, slice) {
                                    Some(n) => depth = depth.add(n),
                                    None => depth = depth.after_unknown_push(),
                                }
                            }
                            Leaf::Normalize | Leaf::NormalizeMoved | Leaf::ItemsNop => {}
                            Leaf::Get | Leaf::Set | Leaf::CopyRange => {
                                if !call
                                    .args
                                    .get(arg0)
                                    .and_then(operand_local)
                                    .is_some_and(|l| {
                                        accept_index(l, &indices, &params, &mut param_slots)
                                    })
                                {
                                    reads_below = true;
                                    why = format!("index-not-own bb{bb}");
                                }
                            }
                            Leaf::ItemsGet => {
                                if items_guard.and_then(|g| guards.get(&g).copied()).is_none() {
                                    reads_below = true;
                                    why = format!("index-not-own bb{bb}");
                                }
                            }
                            Leaf::ReloadTop => {
                                if !matches!(depth, Depth::Known(d) if d > 0) {
                                    reads_below = true;
                                    why = "reload-top-below".into();
                                }
                            }
                            Leaf::Unmodeled => {
                                reads_below = true;
                                depth = Depth::Unknown;
                                why = format!("unmodeled {}", path.as_deref().unwrap_or(""));
                            }
                        }
                    }
                    None => {
                        let callee = path
                            .as_deref()
                            .map(effect)
                            .unwrap_or_else(CalleeEffect::none);
                        if callee.observes {
                            // Unproven stack-sensitive callee. At
                            // Known(0) every live slot is the caller's,
                            // so the body is not neutral. Inside our
                            // own Open (`eval_slice_index` pin then
                            // `getindex_w`) only the depth is unknown:
                            // Close rewinds our guard, and
                            // `is_stack_depth_neutral_fn` still holds
                            // so BINARY_SLICE can erase its bracket.
                            if !depth.above_entry() {
                                reads_below = true;
                                why = format!(
                                    "unproven-stack-sensitive {}",
                                    path.as_deref().unwrap_or("")
                                );
                            }
                            // A reader that may also leave slots, or end
                            // below its entry, leaves no bound at all.
                            depth = if callee.leaves_above {
                                Depth::Unknown
                            } else {
                                depth
                            };
                        } else {
                            // pin() returns shadow_stack_len() captured at
                            // callee entry, an own index >= entry; the
                            // caller dest is the depth at the call.
                            let depth_at_call = depth;
                            if callee.leaves_above {
                                depth = depth.after_unknown_push();
                            }
                            for &p in &callee.param_slots {
                                if !call
                                    .args
                                    .get(p as usize)
                                    .and_then(operand_local)
                                    .is_some_and(|l| {
                                        accept_index(l, &indices, &params, &mut param_slots)
                                    })
                                {
                                    reads_below = true;
                                    why = format!("index-not-own bb{bb}");
                                    break;
                                }
                            }
                            if callee.returns_index
                                && let Some(dest) = place_local(&call.dest)
                                && depth_at_call.lower_bound().is_some()
                            {
                                // pin() returns the entry Len; a struct
                                // whose field is an own index is the same
                                // bound at the call-site depth.
                                indices.insert(dest, depth_at_call);
                            }
                        }
                    }
                }
            }
            Ok(TermKind::Drop { place, target, .. }) => {
                successors.push(*target);
                if let Some(g) = place_local(place).filter(|l| guards.contains_key(l)) {
                    depth = guards[&g];
                } else if place_local(place).is_some_and(|l| {
                    is_root_scope_local(body, llbc, l) || is_rooted_items_local(body, llbc, l)
                }) {
                    depth = Depth::Unknown;
                }
            }
            Ok(TermKind::Assert { target, .. }) => successors.push(*target),
            Ok(TermKind::Goto { target }) => successors.push(*target),
            Ok(TermKind::Switch { targets, .. }) => match targets {
                majit_charon_reader::ullbc::SwitchTargets::If(a, b) => {
                    successors.extend([*a, *b]);
                }
                majit_charon_reader::ullbc::SwitchTargets::SwitchInt(_, arms, default) => {
                    successors.extend(arms.iter().map(|(_, t)| *t));
                    successors.push(*default);
                }
            },
            Ok(TermKind::Return) => {
                let this_returns_index = indices.get(&0).is_some_and(|d| d.lower_bound().is_some())
                    || field_indices
                        .iter()
                        .any(|((l, _), d)| *l == 0 && d.lower_bound().is_some());
                returns_index = Some(returns_index.unwrap_or(true) && this_returns_index);
                note_exit(
                    depth,
                    "returns-at",
                    &mut why,
                    &mut leaves_above,
                    &mut unknown_exit,
                );
            }
            Ok(TermKind::UnwindResume | TermKind::UnwindTerminate) => note_exit(
                depth,
                "unwinds-at",
                &mut why,
                &mut leaves_above,
                &mut unknown_exit,
            ),
            _ => {}
        }
        if reads_below && read_why.is_none() {
            read_why = Some(why.clone());
        }
        for target in successors {
            let target = target as usize;
            if target >= n_blocks {
                continue;
            }
            let next = match depth_in[target] {
                None => depth,
                Some(old) => old.join(depth),
            };
            if depth_in[target] != Some(next) {
                depth_in[target] = Some(next);
                queue.push_back(target);
            }
        }
    }
    let why = if reads_below {
        read_why
    } else if unknown_exit || leaves_above {
        Some(why)
    } else {
        None
    };
    // A reader's parameter-indexed accesses add nothing to "reads below".
    if reads_below {
        param_slots.clear();
    }
    param_slots.sort_unstable();
    param_slots.dedup();
    StackWalk {
        why,
        saw_open,
        reads_below,
        leaves_above,
        unknown_exit,
        param_slots,
        returns_index: returns_index == Some(true),
    }
}

/// `Known(k) + c` stays `Known(k+c)` when `c` is a constant; any other
/// unsigned addend drops to `AtLeast(k)`. Adding cannot yield a smaller
/// unsigned index than the tracked lower bound.
fn add_unsigned_index(k: Depth, other: &Operand, llbc: &Llbc) -> Depth {
    match const_usize(other, llbc) {
        Some(c) => k.add(c),
        None => match k {
            Depth::Known(d) | Depth::AtLeast(d) => Depth::AtLeast(d),
            Depth::Unknown => Depth::Unknown,
        },
    }
}

fn accept_index(
    local: usize,
    indices: &HashMap<usize, Depth>,
    params: &HashMap<usize, u8>,
    param_slots: &mut Vec<u8>,
) -> bool {
    if let Some(d) = indices.get(&local) {
        return matches!(d, Depth::Known(_) | Depth::AtLeast(_));
    }
    if let Some(&p) = params.get(&local) {
        if !param_slots.contains(&p) {
            param_slots.push(p);
        }
        return true;
    }
    false
}

fn note_exit(
    depth: Depth,
    kind: &str,
    why: &mut String,
    leaves_above: &mut bool,
    unknown_exit: &mut bool,
) {
    match depth {
        Depth::Known(0) => {}
        Depth::Unknown => {
            *unknown_exit = true;
            *why = format!("{kind} {depth:?}");
        }
        Depth::Known(_) | Depth::AtLeast(_) => {
            *leaves_above = true;
            if !*unknown_exit && why.is_empty() {
                *why = format!("{kind} {depth:?}");
            }
        }
    }
}

/// [`resolve_published_array`] without the JSON: only the literal's length.
fn resolve_published_array_len(
    block: &majit_charon_reader::ullbc::BasicBlock,
    slice: Option<usize>,
) -> Option<usize> {
    let mut want = slice?;
    for stmt in block.statements.iter().rev() {
        let Ok(StmtKind::Assign(place, rvalue)) = stmt.stmt_kind_ref() else {
            continue;
        };
        if place_local(place) != Some(want) {
            continue;
        }
        match rvalue {
            Rvalue::UnaryOp(_, op) | Rvalue::Use(op, _) | Rvalue::Cast(_, op, _) => {
                want = operand_local(op)?;
            }
            Rvalue::Ref { place: src, .. } => {
                want = place_local(src).or_else(|| deref_of_local(src))?;
            }
            Rvalue::Aggregate(kind, operands) if kind.get("Array").is_some() => {
                return Some(operands.len());
            }
            _ => return None,
        }
    }
    None
}

/// The functions of `llbc` whose shadow-stack effect a caller cannot see
/// past.  Callees from artefacts linked earlier are answered by what they
/// registered on `llbc` ([`Llbc::is_stack_sensitive_fn`],
/// [`Llbc::is_stack_leaves_above_fn`]).
pub fn discover_stack_sensitive_fns(llbc: &Llbc) -> Vec<String> {
    discover_stack_fn_effects(llbc).0
}

/// Local bodies that never read below entry but can return with slots
/// still published above it. Harvested in link order like
/// [`discover_stack_sensitive_fns`].
pub fn discover_stack_leaves_above_fns(llbc: &Llbc) -> Vec<String> {
    discover_stack_fn_effects(llbc).1
}

/// Local bodies that index the shadow stack through a parameter.
pub fn discover_stack_param_slots_fns(llbc: &Llbc) -> Vec<(String, Vec<u8>)> {
    discover_stack_fn_effects(llbc).2
}

/// Local bodies whose every reachable return is an own index relative
/// to entry. Harvested in link order like
/// [`discover_stack_sensitive_fns`].
pub fn discover_stack_returns_index_fns(llbc: &Llbc) -> Vec<String> {
    discover_stack_fn_effects(llbc).3
}

/// Sensitive, leaves-above, param-slots, returns-index and observes
/// sets from one fixpoint, so harvest does not recompute the body walks.
pub fn discover_stack_fn_effects(
    llbc: &Llbc,
) -> (
    Vec<String>,
    Vec<String>,
    Vec<(String, Vec<u8>)>,
    Vec<String>,
    Vec<String>,
) {
    discover_stack_effects(llbc)
}

/// Least consistent assignment of per-body [`stack_walk`] over this artefact.
///
/// Seed every local body against all-`None` (own-body effects only). Then
/// Gauss-Seidel: recompute a body from scratch against the live classes
/// (previous crates through [`callee_effect_of`]). Jacobi rounds oscillate
/// on a caller that Observes only because a `returns_index` callee is not
/// yet in the previous map; processing Observes first lets that caller
/// drop once the callee's summary is live. A revisit cap freezes still-
/// changing bodies as Observes.
fn discover_stack_effects(
    llbc: &Llbc,
) -> (
    Vec<String>,
    Vec<String>,
    Vec<(String, Vec<u8>)>,
    Vec<String>,
    Vec<String>,
) {
    let fds: Vec<&FunDecl> = llbc
        .iter_local_fns()
        .filter(|fd| fd.body.is_some())
        .collect();
    let names: Vec<String> = fds.iter().map(|fd| fd.item_meta.name_path()).collect();
    let local: std::collections::HashSet<&str> = names.iter().map(String::as_str).collect();
    let mut callers: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, fd) in fds.iter().enumerate() {
        let Some(body) = fd.unstructured() else {
            continue;
        };
        for block in &body.body {
            if let Ok(TermKind::Call { call, .. }) = block.term_ref(llbc)
                && let Some(path) = callee_path(call, llbc)
            {
                callers.entry(path).or_default().push(i);
            }
        }
    }
    let mut class: HashMap<String, CalleeEffect> = HashMap::new();
    let none_effect = |path: &str| {
        if local.contains(path) {
            CalleeEffect::none()
        } else {
            callee_effect_of(llbc, path)
        }
    };
    let mut queued = vec![false; fds.len()];
    let mut observes_ids: Vec<usize> = Vec::new();
    let mut other_ids: Vec<usize> = Vec::new();
    for (i, fd) in fds.iter().enumerate() {
        let Some(body) = fd.unstructured() else {
            continue;
        };
        let new = stack_walk(&body, llbc, &none_effect).effect();
        if new.observes {
            observes_ids.push(i);
        } else {
            other_ids.push(i);
        }
        if !new.is_none() {
            class.insert(names[i].clone(), new);
        }
    }
    let mut queue: VecDeque<usize> = VecDeque::new();
    for i in observes_ids.into_iter().chain(other_ids) {
        queued[i] = true;
        queue.push_back(i);
    }
    let mut visits = vec![0u8; fds.len()];
    const VISIT_CAP: u8 = 8;
    while let Some(i) = queue.pop_front() {
        queued[i] = false;
        if visits[i] >= VISIT_CAP {
            let old = class
                .get(&names[i])
                .cloned()
                .unwrap_or_else(CalleeEffect::none);
            class.insert(names[i].clone(), CalleeEffect::worst());
            if old != CalleeEffect::worst() {
                for &caller in callers.get(&names[i]).into_iter().flatten() {
                    if queued[caller] {
                        continue;
                    }
                    queued[caller] = true;
                    queue.push_back(caller);
                }
            }
            continue;
        }
        visits[i] = visits[i].saturating_add(1);
        let Some(body) = fds[i].unstructured() else {
            continue;
        };
        let effect = |path: &str| {
            if local.contains(path) {
                class.get(path).cloned().unwrap_or_else(CalleeEffect::none)
            } else {
                callee_effect_of(llbc, path)
            }
        };
        let new = stack_walk(&body, llbc, &effect).effect();
        let old = class
            .get(&names[i])
            .cloned()
            .unwrap_or_else(CalleeEffect::none);
        if new == old {
            continue;
        }
        if new.is_none() {
            class.remove(&names[i]);
        } else {
            class.insert(names[i].clone(), new.clone());
        }
        for &caller in callers.get(&names[i]).into_iter().flatten() {
            if queued[caller] {
                continue;
            }
            queued[caller] = true;
            if new.observes {
                queue.push_back(caller);
            } else {
                // Callee dropped Observes or gained a summary: re-walk
                // Observes callers first so they can drop.
                queue.push_front(caller);
            }
        }
    }
    let mut sensitive: Vec<String> = Vec::new();
    let mut leaves: Vec<String> = Vec::new();
    let mut param_slots: Vec<(String, Vec<u8>)> = Vec::new();
    let mut returns_index: Vec<String> = Vec::new();
    let mut observes: Vec<String> = Vec::new();
    for (name, effect) in class {
        if effect.is_none() {
            continue;
        }
        sensitive.push(name.clone());
        if effect.leaves_above {
            leaves.push(name.clone());
        }
        if effect.observes {
            observes.push(name.clone());
        }
        if !effect.param_slots.is_empty() {
            param_slots.push((name.clone(), effect.param_slots));
        }
        if effect.returns_index {
            returns_index.push(name);
        }
    }
    sensitive.sort();
    observes.sort();
    leaves.sort();
    param_slots.sort_by(|a, b| a.0.cmp(&b.0));
    returns_index.sort();
    (sensitive, leaves, param_slots, returns_index, observes)
}

/// Local bodies proven depth-neutral: every path restores the entry
/// depth, no caller-owned slot is read, Observes callees are refused,
/// and LeavesAbove callees are allowed (fixpoint, pessimistic start).
/// Harvested in link order like [`discover_stack_sensitive_fns`].
pub fn discover_depth_neutral_fns(llbc: &Llbc) -> Vec<String> {
    let fds: Vec<&FunDecl> = llbc
        .iter_local_fns()
        .filter(|fd| fd.body.is_some())
        .collect();
    let mut proven: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut changed = true;
    while changed {
        changed = false;
        for fd in &fds {
            let name = fd.item_meta.name_path();
            if proven.contains(&name) {
                continue;
            }
            let Some(body) = fd.unstructured() else {
                continue;
            };
            let effect = |path: &str| {
                if proven.contains(path) {
                    CalleeEffect::none()
                } else {
                    callee_effect_of(llbc, path)
                }
            };
            if body_is_depth_neutral(&body, llbc, &effect) {
                proven.insert(name);
                changed = true;
            }
        }
    }
    let mut out: Vec<String> = proven.into_iter().collect();
    out.sort();
    out
}

/// Classify `llbc`'s own functions and mark its set complete, for a caller
/// that lowers one artefact on its own.  Callees from other artefacts count
/// as insensitive here; the linked translation registers them first.
pub fn ensure_stack_sensitive_fns(llbc: &Llbc) {
    if llbc.stack_sensitive_fns_complete() {
        return;
    }
    let (found, leaves, params, ret_idx, observes) = discover_stack_effects(llbc);
    llbc.register_stack_sensitive_fns(found);
    llbc.register_stack_observes_fns(observes);
    llbc.register_stack_leaves_above_fns(leaves);
    llbc.register_stack_param_slots_fns(params);
    llbc.register_stack_returns_index_fns(ret_idx);
    llbc.register_stack_depth_neutral_fns(discover_depth_neutral_fns(llbc));
    llbc.mark_stack_sensitive_fns_complete();
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_charon_reader::Llbc;
    use std::collections::HashMap;

    fn span() -> Value {
        json!({
            "data": {
                "file_id": 0,
                "beg": {"line": 0, "col": 0},
                "end": {"line": 0, "col": 0}
            },
            "generated_from_span": null
        })
    }

    fn fixture_llbc() -> Llbc {
        let ident = |s: &str| json!({"Ident": [s, 0]});
        let push_roots = json!({
            "def_id": 1,
            "item_meta": {
                "name": [
                    ident("pyre_object"),
                    ident("gc_roots"),
                    ident("push_roots")
                ],
                "span": span(),
                "source_text": null,
                "attr_info": {
                    "attributes": [],
                    "inline": null,
                    "rename": null,
                    "public": true
                },
                "is_local": true
            },
            "signature": {
                "is_unsafe": false,
                "inputs": [],
                "output": {"Deduplicated": 0}
            },
            "body": "Opaque"
        });
        let file = json!({
            "charon_version": "t",
            "has_errors": false,
            "translated": {
                "crate_name": "c",
                "fun_decls": [null, push_roots]
            }
        });
        let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc");
        llbc.mark_stack_sensitive_fns_complete();
        llbc
    }

    fn analyze_open_then(sink: Value) -> Result<Option<Plan>, Refusal> {
        let ty = json!({"Deduplicated": 0});
        let place = |i: u64| json!({"kind": {"Local": i}, "ty": ty});
        let local = |i: u64| json!({"index": i, "name": null, "span": span(), "ty": ty});
        let bb = |kind: Value| {
            json!({
                "statements": [],
                "terminator": {"span": span(), "kind": kind},
                "is_cleanup": false
            })
        };
        let open = json!({"Call": {
            "call": {
                "func": {"Regular": {"kind": {"Fun": 1}, "generics": {}}},
                "args": [],
                "dest": place(1)
            },
            "target": 1,
            "on_unwind": 2
        }});
        let raw = json!({
            "span": span(),
            "locals": {"arg_count": 0, "locals": [local(0), local(1)]},
            "body": [
                bb(open),
                bb(sink),
                bb(json!("UnwindResume"))
            ]
        });
        let body: Unstructured = serde_json::from_value(raw.clone()).expect("fixture body");
        analyze(&body, &fixture_llbc())
    }

    fn assert_erase_accepted(sink: Value, label: &str) {
        match analyze_open_then(sink) {
            Ok(Some(_)) => {}
            Ok(None) => panic!("{label}: expected accepted erase, got no bracket"),
            Err(reason) => panic!("{label}: expected accepted erase, got {reason}"),
        }
    }

    #[test]
    fn analyze_accepts_undefined_behavior_sink_like_abort() {
        assert_erase_accepted(json!({"Abort": "UndefinedBehavior"}), "Abort");
        assert_erase_accepted(json!("UndefinedBehavior"), "UndefinedBehavior");
    }

    #[test]
    fn analyze_accepts_unwind_terminate_sink_like_abort() {
        assert_erase_accepted(json!({"Abort": "UnwindTerminate"}), "Abort");
        assert_erase_accepted(json!("UnwindTerminate"), "UnwindTerminate");
    }

    fn opaque_fun(def_id: u64, name: &[&str], body: Value) -> Value {
        let ident = |s: &str| json!({"Ident": [s, 0]});
        json!({
            "def_id": def_id,
            "item_meta": {
                "name": name.iter().map(|s| ident(s)).collect::<Vec<_>>(),
                "span": span(),
                "source_text": null,
                "attr_info": {
                    "attributes": [],
                    "inline": null,
                    "rename": null,
                    "public": true
                },
                "is_local": true
            },
            "signature": {
                "is_unsafe": false,
                "inputs": [],
                "output": {"Deduplicated": 0}
            },
            "body": body
        })
    }

    /// A `pin_roots` of a same-block array, with an unwind-only block in
    /// front of it in the artefact.  The CFG drop of that block must not
    /// make the array unresolvable.
    #[test]
    fn erase_accepts_publish_when_cleanup_blocks_precede_the_array() {
        let ty = json!({"Deduplicated": 0});
        let place = |i: u64| json!({"kind": {"Local": i}, "ty": ty});
        let local = |i: u64| json!({"index": i, "name": null, "span": span(), "ty": ty});
        let stmt = |kind: Value| json!({"span": span(), "kind": kind});
        let bb = |statements: Vec<Value>, kind: Value, is_cleanup: bool| {
            json!({
                "statements": statements,
                "terminator": {"span": span(), "kind": kind},
                "is_cleanup": is_cleanup
            })
        };
        let call = |fun: u64, args: Vec<Value>, dest: u64, target: u64, on_unwind: u64| {
            json!({"Call": {
                "call": {
                    "func": {"Regular": {"kind": {"Fun": fun}, "generics": {}}},
                    "args": args,
                    "dest": place(dest)
                },
                "target": target,
                "on_unwind": on_unwind
            }})
        };
        let array = json!({"Assign": [
            place(2),
            {"Aggregate": [
                {"Array": [ty, {"Deduplicated": 1}, null]},
                [
                    {"Move": place(3)},
                    {"Move": place(4)}
                ]
            ]}
        ]});
        let unstructured = json!({
            "span": span(),
            "locals": {
                "arg_count": 0,
                "locals": (0..6).map(local).collect::<Vec<_>>()
            },
            "body": [
                bb(vec![], call(1, vec![], 1, 2, 1), false),
                bb(vec![], json!("UnwindResume"), true),
                bb(
                    vec![stmt(array)],
                    call(2, vec![json!({"Move": place(2)})], 5, 3, 1),
                    false
                ),
                bb(vec![], json!("UndefinedBehavior"), false)
            ]
        });
        let file = json!({
            "charon_version": "t",
            "has_errors": false,
            "translated": {
                "crate_name": "c",
                "fun_decls": [
                    null,
                    opaque_fun(1, &["pyre_object", "gc_roots", "push_roots"], json!("Opaque")),
                    opaque_fun(2, &["pyre_object", "gc_roots", "pin_roots"], json!("Opaque")),
                    opaque_fun(
                        3,
                        &["pyre_object", "celldict", "write_cell"],
                        json!({ "Unstructured": unstructured })
                    )
                ]
            }
        });
        let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc");
        llbc.mark_stack_sensitive_fns_complete();
        let fd = llbc.fn_by_id(3).expect("subject");
        let body = fd.unstructured().expect("stripped body");
        assert_eq!(
            body.body.len(),
            4,
            "three kept blocks plus one unwind resume"
        );
        assert!(
            body.body.iter().all(|bb| !bb.is_cleanup),
            "unwind-only block should have been dropped"
        );
        match erase_shadow_stack(fd, &body, &llbc) {
            Ok(Some(_)) => {}
            Ok(None) => panic!("expected accepted erase, got no bracket"),
            Err(reason) => panic!("expected accepted erase, got {reason}"),
        }
    }

    fn stack_ops_llbc() -> Llbc {
        let file = json!({
            "charon_version": "t",
            "has_errors": false,
            "translated": {
                "crate_name": "c",
                "fun_decls": [
                    null,
                    opaque_fun(1, &["pyre_object", "gc_roots", "push_roots"], json!("Opaque")),
                    opaque_fun(2, &["pyre_object", "gc_roots", "shadow_stack_get"], json!("Opaque")),
                    opaque_fun(3, &["pyre_object", "gc_roots", "shadow_stack_len"], json!("Opaque")),
                    opaque_fun(4, &["pyre_object", "gc_roots", "pin_root"], json!("Opaque"))
                ]
            }
        });
        let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc");
        llbc.mark_stack_sensitive_fns_complete();
        llbc
    }

    fn ty() -> Value {
        json!({"Deduplicated": 0})
    }

    fn place(i: u64) -> Value {
        json!({"kind": {"Local": i}, "ty": ty()})
    }

    fn local_decl(i: u64) -> Value {
        json!({"index": i, "name": null, "span": span(), "ty": ty()})
    }

    fn bb(kind: Value) -> Value {
        json!({
            "statements": [],
            "terminator": {"span": span(), "kind": kind},
            "is_cleanup": false
        })
    }

    fn call_fun(fun: u64, args: Vec<Value>, dest: u64, target: u64) -> Value {
        json!({"Call": {
            "call": {
                "func": {"Regular": {"kind": {"Fun": fun}, "generics": {}}},
                "args": args,
                "dest": place(dest)
            },
            "target": target,
            "on_unwind": 99
        }})
    }

    fn drop_local(local: u64, target: u64) -> Value {
        json!({"Drop": {
            "place": place(local),
            "fn_ptr": {"kind": {"Fun": 0}, "generics": {}},
            "target": target,
            "on_unwind": 99
        }})
    }

    fn body_of(n_locals: u64, blocks: Vec<Value>) -> Unstructured {
        body_of_args(0, n_locals, blocks)
    }

    fn body_of_args(arg_count: u64, n_locals: u64, blocks: Vec<Value>) -> Unstructured {
        let raw = json!({
            "span": span(),
            "locals": {
                "arg_count": arg_count,
                "locals": (0..n_locals).map(local_decl).collect::<Vec<_>>()
            },
            "body": blocks
        });
        serde_json::from_value(raw).expect("fixture body")
    }

    fn is_neutral(body: &Unstructured, llbc: &Llbc) -> bool {
        body_is_depth_neutral(body, llbc, &|_| CalleeEffect::none())
    }

    #[test]
    fn depth_neutral_refuses_scope_leaked_on_one_branch() {
        let llbc = stack_ops_llbc();
        let body = body_of(
            3,
            vec![
                bb(call_fun(1, vec![], 1, 1)),
                bb(call_fun(4, vec![json!({"Copy": place(0)})], 2, 2)),
                bb(json!({"Switch": {
                    "discr": {"Copy": place(0)},
                    "targets": {"If": [3, 4]}
                }})),
                bb(drop_local(1, 5)),
                bb(json!({"Goto": {"target": 5}})),
                bb(json!("Return")),
            ],
        );
        assert!(
            !is_neutral(&body, &llbc),
            "a branch that skips Close must not be depth-neutral"
        );
    }

    #[test]
    fn depth_neutral_refuses_balanced_scope_plus_caller_slot_read() {
        let llbc = stack_ops_llbc();
        let body = body_of(
            4,
            vec![
                bb(call_fun(1, vec![], 1, 1)),
                bb(call_fun(3, vec![], 2, 2)),
                // Local 0 is not a Len/Base index: untracked, so index-not-own.
                bb(call_fun(2, vec![json!({"Copy": place(0)})], 3, 3)),
                bb(drop_local(1, 4)),
                bb(json!("Return")),
            ],
        );
        assert!(
            !is_neutral(&body, &llbc),
            "Get of an untracked index must not be depth-neutral"
        );
    }

    #[test]
    fn depth_neutral_refuses_open_and_close_in_unreachable_blocks() {
        let llbc = stack_ops_llbc();
        let body = body_of(
            2,
            vec![
                bb(json!("Return")),
                bb(call_fun(1, vec![], 1, 2)),
                bb(drop_local(1, 3)),
                bb(json!("Return")),
            ],
        );
        assert!(
            !is_neutral(&body, &llbc),
            "open/close only in unreachable blocks must not be depth-neutral"
        );
    }

    #[test]
    fn depth_neutral_accepts_eval_slice_index_style_balanced_body() {
        let llbc = stack_ops_llbc();
        let body = body_of(
            3,
            vec![
                bb(call_fun(1, vec![], 1, 1)),
                bb(call_fun(4, vec![json!({"Copy": place(0)})], 2, 2)),
                bb(drop_local(1, 3)),
                bb(json!("Return")),
            ],
        );
        assert!(
            is_neutral(&body, &llbc),
            "open, pin, close on every path must be depth-neutral"
        );
    }

    fn stack_ops_llbc_with_callee() -> Llbc {
        let file = json!({
            "charon_version": "t",
            "has_errors": false,
            "translated": {
                "crate_name": "c",
                "fun_decls": [
                    null,
                    opaque_fun(1, &["pyre_object", "gc_roots", "push_roots"], json!("Opaque")),
                    opaque_fun(2, &["pyre_object", "gc_roots", "shadow_stack_get"], json!("Opaque")),
                    opaque_fun(3, &["pyre_object", "gc_roots", "shadow_stack_len"], json!("Opaque")),
                    opaque_fun(4, &["pyre_object", "gc_roots", "pin_root"], json!("Opaque")),
                    opaque_fun(5, &["other", "unproven_callee"], json!("Opaque")),
                    opaque_fun(6, &["pyre_object", "gc_roots", "pin_roots"], json!("Opaque")),
                    opaque_fun(7, &["other", "leaves_above_callee"], json!("Opaque")),
                    opaque_fun(8, &["other", "reload_slot"], json!("Opaque")),
                    opaque_fun(9, &["other", "pin_helper"], json!("Opaque")),
                    opaque_fun(10, &["other", "storage_helper"], json!("Opaque")),
                    opaque_fun(11, &["other", "observes_and_leaves"], json!("Opaque"))
                ]
            }
        });
        let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc");
        llbc.register_stack_sensitive_fns([
            "other::unproven_callee".into(),
            "other::leaves_above_callee".into(),
            "other::reload_slot".into(),
            "other::pin_helper".into(),
            "other::storage_helper".into(),
            "other::observes_and_leaves".into(),
        ]);
        llbc.register_stack_observes_fns([
            "other::unproven_callee".into(),
            "other::observes_and_leaves".into(),
        ]);
        llbc.register_stack_leaves_above_fns([
            "other::leaves_above_callee".into(),
            "other::pin_helper".into(),
            "other::observes_and_leaves".into(),
        ]);
        llbc.register_stack_param_slots_fns([("other::reload_slot".into(), vec![0])]);
        llbc.register_stack_returns_index_fns([
            "other::pin_helper".into(),
            "other::storage_helper".into(),
        ]);
        llbc.mark_stack_sensitive_fns_complete();
        llbc
    }

    fn bb_stmts(statements: Vec<Value>, kind: Value) -> Value {
        json!({
            "statements": statements,
            "terminator": {"span": span(), "kind": kind},
            "is_cleanup": false
        })
    }

    fn assign_local(dest: u64, rvalue: Value) -> Value {
        json!({
            "kind": {"Assign": [place(dest), rvalue]},
            "comments_before": [],
            "span": span()
        })
    }

    fn usize_lit(k: u64) -> Value {
        json!({"Const": [{"Integer": {"Unsigned": ["Usize", k.to_string()]}}, ty()]})
    }

    /// `zip_two_tuple_next` / `tuple_iter_descr_next`: pin, then name the
    /// slot as `shadow_stack_len() - 1`, then get it.
    fn len_minus_one_body(sub_op: Value) -> Unstructured {
        let copy = |i| json!({"Copy": place(i)});
        body_of(
            6,
            vec![
                bb(call_fun(1, vec![], 1, 1)),
                bb(call_fun(4, vec![copy(0)], 2, 2)),
                bb(call_fun(3, vec![], 3, 3)),
                bb_stmts(
                    vec![assign_local(
                        4,
                        json!({"BinaryOp": [sub_op, copy(3), usize_lit(1)]}),
                    )],
                    call_fun(2, vec![copy(4)], 5, 4),
                ),
                bb(drop_local(1, 5)),
                bb(json!("Return")),
            ],
        )
    }

    #[test]
    fn analyze_accepts_len_minus_one_after_pin() {
        let llbc = stack_ops_llbc();
        let body = len_minus_one_body(json!("Sub"));
        match analyze(&body, &llbc) {
            Ok(Some(_)) => {}
            Ok(None) => panic!("expected accepted erase, got no bracket"),
            Err(reason) => panic!("expected accepted erase, got {reason}"),
        }
        assert!(
            is_neutral(&body, &llbc),
            "len-1 of a just-pinned slot must be depth-neutral"
        );
    }

    /// `pin_self`: pin, `len - 1`, return that index. The return place is
    /// the Index special; erase would strip its assignment. Last block is
    /// `UnwindResume` so the cleanup-drop depth hack does not refuse first.
    #[test]
    fn analyze_refuses_returning_the_len_minus_one_slot_index() {
        let llbc = stack_ops_llbc();
        let copy = |i| json!({"Copy": place(i)});
        let body = body_of(
            4,
            vec![
                bb(call_fun(4, vec![copy(1)], 2, 1)),
                bb(call_fun(3, vec![], 3, 2)),
                bb_stmts(
                    vec![assign_local(
                        0,
                        json!({"BinaryOp": [json!("Sub"), copy(3), usize_lit(1)]}),
                    )],
                    json!("Return"),
                ),
                bb(json!("UnwindResume")),
            ],
        );
        match analyze(&body, &llbc) {
            Err("returns-slot-index") => {}
            Ok(Some(_)) => panic!("expected returns-slot-index, got accepted"),
            Ok(None) => panic!("expected returns-slot-index, got no bracket"),
            Err(reason) => panic!("expected returns-slot-index, got {reason}"),
        }
    }

    #[test]
    fn analyze_accepts_len_minus_one_wrap_after_pin() {
        let llbc = stack_ops_llbc();
        let body = len_minus_one_body(json!({"Sub": "Wrap"}));
        match analyze(&body, &llbc) {
            Ok(Some(_)) => {}
            Ok(None) => panic!("expected accepted erase, got no bracket"),
            Err(reason) => panic!("expected accepted erase, got {reason}"),
        }
    }

    #[test]
    fn analyze_accepts_len_minus_one_subchecked_after_pin() {
        let llbc = stack_ops_llbc();
        let copy = |i| json!({"Copy": place(i)});
        let field0 = json!({
            "kind": {"Projection": [place(4), {"Field": [null, 0]}]},
            "ty": ty()
        });
        let body = body_of(
            7,
            vec![
                bb(call_fun(1, vec![], 1, 1)),
                bb(call_fun(4, vec![copy(0)], 2, 2)),
                bb(call_fun(3, vec![], 3, 3)),
                bb_stmts(
                    vec![
                        assign_local(
                            4,
                            json!({"BinaryOp": ["SubChecked", copy(3), usize_lit(1)]}),
                        ),
                        assign_local(5, json!({"Use": [{"Move": field0}, "No"]})),
                    ],
                    call_fun(2, vec![copy(5)], 6, 4),
                ),
                bb(drop_local(1, 5)),
                bb(json!("Return")),
            ],
        );
        match analyze(&body, &llbc) {
            Ok(Some(_)) => {}
            Ok(None) => panic!("expected accepted erase, got no bracket"),
            Err(reason) => panic!("expected accepted erase, got {reason}"),
        }
        assert!(
            is_neutral(&body, &llbc),
            "SubChecked len-1 of a just-pinned slot must be depth-neutral"
        );
    }

    fn stmt(kind: Value) -> Value {
        json!({"span": span(), "kind": kind})
    }

    #[test]
    fn depth_neutral_refuses_unproven_stack_sensitive_callee() {
        let llbc = stack_ops_llbc_with_callee();
        let observes = |path: &str| {
            if path.ends_with("unproven_callee") {
                CalleeEffect::observes()
            } else {
                CalleeEffect::none()
            }
        };
        // After Close the depth is Known(0): every remaining slot is the
        // caller's, so an Observes callee is not neutral.
        let after_close = body_of(
            4,
            vec![
                bb(call_fun(1, vec![], 1, 1)),
                bb(call_fun(4, vec![json!({"Copy": place(0)})], 2, 2)),
                bb(drop_local(1, 3)),
                bb(call_fun(5, vec![], 4, 4)),
                bb(json!("Return")),
            ],
        );
        assert!(
            !body_is_depth_neutral(&after_close, &llbc, &observes),
            "Observes callee at Known(0) must not be depth-neutral"
        );
        assert!(
            body_is_depth_neutral(&after_close, &llbc, &|_| CalleeEffect::none()),
            "the same body is depth-neutral once the callee is proven"
        );
        // Unproven callees inside our Open/Close are depth-neutral:
        // they see our pins, and Close rewinds the guard.
        let inside_scope = body_of(
            4,
            vec![
                bb(call_fun(1, vec![], 1, 1)),
                bb(call_fun(4, vec![json!({"Copy": place(0)})], 2, 2)),
                bb(call_fun(5, vec![], 3, 3)),
                bb(call_fun(5, vec![], 3, 4)),
                bb(drop_local(1, 5)),
                bb(json!("Return")),
            ],
        );
        assert!(
            body_is_depth_neutral(&inside_scope, &llbc, &observes),
            "unproven callees inside our Open/Close are depth-neutral"
        );
    }

    #[test]
    fn open_pin_callee_close_observes_vs_leaves_above() {
        let llbc = stack_ops_llbc_with_callee();
        let effect = |path: &str| callee_effect_of(&llbc, path);
        let observes_inside = body_of(
            4,
            vec![
                bb(call_fun(1, vec![], 1, 1)),
                bb(call_fun(4, vec![json!({"Copy": place(0)})], 2, 2)),
                bb(call_fun(5, vec![], 3, 3)),
                bb(drop_local(1, 4)),
                bb(json!("Return")),
            ],
        );
        assert!(
            body_is_depth_neutral(&observes_inside, &llbc, &effect),
            "Open -> Pin -> Observes -> Close is depth-neutral"
        );
        match analyze(&observes_inside, &llbc) {
            Ok(Some(_)) => {}
            Ok(None) => panic!("expected erasure of Open/Pin/Observes/Close"),
            Err(reason) => panic!("expected erasure of Open/Pin/Observes/Close, got {reason}"),
        }
        let leaves_inside = body_of(
            4,
            vec![
                bb(call_fun(1, vec![], 1, 1)),
                bb(call_fun(4, vec![json!({"Copy": place(0)})], 2, 2)),
                bb(call_fun(7, vec![], 3, 3)),
                bb(drop_local(1, 4)),
                bb(json!("Return")),
            ],
        );
        assert!(
            body_is_depth_neutral(&leaves_inside, &llbc, &effect),
            "Open -> Pin -> LeavesAbove -> Close is depth-neutral"
        );
        match analyze(&leaves_inside, &llbc) {
            Err("calls-stack-sensitive-fn") => {}
            Ok(Some(_)) => panic!("expected calls-stack-sensitive-fn, got accepted"),
            Ok(None) => panic!("expected calls-stack-sensitive-fn, got no bracket"),
            Err(reason) => panic!("expected calls-stack-sensitive-fn, got {reason}"),
        }
        // A callee that reads below entry and also leaves slots: Close
        // still rewinds it, so the body stays neutral, but the erased
        // body would have no Close left.
        let observes_and_leaves_inside = body_of(
            4,
            vec![
                bb(call_fun(1, vec![], 1, 1)),
                bb(call_fun(4, vec![json!({"Copy": place(0)})], 2, 2)),
                bb(call_fun(11, vec![], 3, 3)),
                bb(drop_local(1, 4)),
                bb(json!("Return")),
            ],
        );
        assert!(
            body_is_depth_neutral(&observes_and_leaves_inside, &llbc, &effect),
            "Open -> Pin -> Observes+LeavesAbove -> Close is depth-neutral"
        );
        match analyze(&observes_and_leaves_inside, &llbc) {
            Err("calls-stack-sensitive-fn") => {}
            Ok(Some(_)) => panic!("expected calls-stack-sensitive-fn, got accepted"),
            Ok(None) => panic!("expected calls-stack-sensitive-fn, got no bracket"),
            Err(reason) => panic!("expected calls-stack-sensitive-fn, got {reason}"),
        }
        let only_observes_and_leaves =
            body_of(2, vec![bb(call_fun(11, vec![], 1, 1)), bb(json!("Return"))]);
        assert_eq!(
            stack_walk(&only_observes_and_leaves, &llbc, &effect).effect(),
            CalleeEffect::worst(),
            "a reader's leftover reaches its caller's summary"
        );
        let only_leaves = body_of(2, vec![bb(call_fun(7, vec![], 1, 1)), bb(json!("Return"))]);
        let walk = stack_walk(&only_leaves, &llbc, &effect);
        assert_eq!(
            walk.effect(),
            CalleeEffect::leaves_above(),
            "a body that only calls LeavesAbove and returns is LeavesAbove"
        );
    }

    #[test]
    fn param_slots_callee_own_index_is_neutral_untracked_is_observes() {
        let llbc = stack_ops_llbc_with_callee();
        let effect = |path: &str| callee_effect_of(&llbc, path);
        let callee = body_of_args(
            1,
            3,
            vec![
                bb(call_fun(2, vec![json!({"Copy": place(1)})], 2, 1)),
                bb(json!("Return")),
            ],
        );
        let walk = stack_walk(&callee, &llbc, &effect);
        assert!(
            !walk.effect().observes && walk.effect().param_slots == vec![0],
            "Get of a parameter index is ParamSlots, not Observes: {:?}",
            walk.effect()
        );
        assert!(
            !body_is_depth_neutral(&callee, &llbc, &effect),
            "a param_slots body is not depth-neutral"
        );
        let own_index = body_of(
            5,
            vec![
                bb(call_fun(1, vec![], 1, 1)),
                bb(call_fun(4, vec![json!({"Copy": place(0)})], 2, 2)),
                bb(call_fun(3, vec![], 3, 3)),
                bb(call_fun(8, vec![json!({"Copy": place(3)})], 4, 4)),
                bb(drop_local(1, 5)),
                bb(json!("Return")),
            ],
        );
        assert!(
            body_is_depth_neutral(&own_index, &llbc, &effect),
            "caller passing its own Len index to a param_slots callee is depth-neutral"
        );
        match analyze(&own_index, &llbc) {
            Err("calls-stack-sensitive-fn") => {}
            Ok(Some(_)) => panic!("expected calls-stack-sensitive-fn, got accepted"),
            Ok(None) => panic!("expected calls-stack-sensitive-fn, got no bracket"),
            Err(reason) => panic!("expected calls-stack-sensitive-fn, got {reason}"),
        }
        let untracked = body_of(
            4,
            vec![
                bb(call_fun(1, vec![], 1, 1)),
                bb(call_fun(4, vec![json!({"Copy": place(0)})], 2, 2)),
                bb(call_fun(8, vec![json!({"Copy": place(0)})], 3, 3)),
                bb(drop_local(1, 4)),
                bb(json!("Return")),
            ],
        );
        let walk = stack_walk(&untracked, &llbc, &effect);
        assert!(
            walk.effect().observes,
            "caller passing an untracked index to a param_slots callee is Observes: why={:?}",
            walk.why
        );
        match analyze(&untracked, &llbc) {
            Err("calls-stack-sensitive-fn") => {}
            Ok(Some(_)) => panic!("expected calls-stack-sensitive-fn, got accepted"),
            Ok(None) => panic!("expected calls-stack-sensitive-fn, got no bracket"),
            Err(reason) => panic!("expected calls-stack-sensitive-fn, got {reason}"),
        }
        // B.1: param p + any unsigned operand stays param p.
        let param_plus = body_of_args(
            1,
            4,
            vec![
                bb_stmts(
                    vec![assign_local(
                        2,
                        json!({"BinaryOp": ["Add", json!({"Copy": place(1)}), json!({"Copy": place(3)})]}),
                    )],
                    call_fun(2, vec![json!({"Copy": place(2)})], 3, 1),
                ),
                bb(json!("Return")),
            ],
        );
        let walk = stack_walk(&param_plus, &llbc, &effect);
        assert!(
            !walk.effect().observes && walk.effect().param_slots == vec![0],
            "Get of param+unsigned is ParamSlots, not Observes: {:?}",
            walk.effect()
        );
        // B.1: Known(k) + unsigned local is AtLeast(k), still own.
        let known_plus = body_of(
            6,
            vec![
                bb(call_fun(1, vec![], 1, 1)),
                bb(call_fun(4, vec![json!({"Copy": place(0)})], 2, 2)),
                bb(call_fun(3, vec![], 3, 3)),
                bb_stmts(
                    vec![assign_local(
                        4,
                        json!({"BinaryOp": ["Add", json!({"Copy": place(3)}), json!({"Copy": place(5)})]}),
                    )],
                    call_fun(2, vec![json!({"Copy": place(4)})], 5, 4),
                ),
                bb(drop_local(1, 5)),
                bb(json!("Return")),
            ],
        );
        assert!(
            body_is_depth_neutral(&known_plus, &llbc, &effect),
            "Get of Len+unsigned must be own (AtLeast of the Len bound)"
        );
        // B.2: pin-style helper returns the entry-depth index; caller Get of dest is own.
        let pin_then_get = body_of(
            5,
            vec![
                bb(call_fun(1, vec![], 1, 1)),
                bb(call_fun(9, vec![], 3, 2)),
                bb(call_fun(2, vec![json!({"Copy": place(3)})], 4, 3)),
                bb(drop_local(1, 4)),
                bb(json!("Return")),
            ],
        );
        assert!(
            body_is_depth_neutral(&pin_then_get, &llbc, &effect),
            "Get of a returns_index dest (depth at the call) must be own"
        );
    }

    #[test]
    fn pin_style_body_returns_own_index_at_entry_depth() {
        let llbc = stack_ops_llbc();
        let copy = |i| json!({"Copy": place(i)});
        // Len into the return place, then Pin: every Return yields Known(0).
        let pin_style = body_of(
            3,
            vec![
                bb(call_fun(3, vec![], 0, 1)),
                bb(call_fun(4, vec![copy(1)], 2, 2)),
                bb(json!("Return")),
            ],
        );
        let walk = stack_walk(&pin_style, &llbc, &|_| CalleeEffect::none());
        assert!(
            walk.returns_index && walk.effect().leaves_above && !walk.effect().observes,
            "Len then Pin then Return of the Len dest is LeavesAbove+returns_index: {:?}",
            walk.effect()
        );
        let misses = body_of(
            3,
            vec![
                bb(call_fun(3, vec![], 1, 1)),
                bb(call_fun(4, vec![copy(1)], 2, 2)),
                bb(json!("Return")),
            ],
        );
        let walk = stack_walk(&misses, &llbc, &|_| CalleeEffect::none());
        assert!(
            !walk.returns_index,
            "Return of unbound local 0 is not returns_index: {:?}",
            walk.effect()
        );
    }

    #[test]
    fn option_wrap_of_own_index_stays_own() {
        let llbc = stack_ops_llbc();
        let copy = |i| json!({"Copy": place(i)});
        let some = json!({"Adt": [
            {"builtin": null, "generics": {
                "const_generics": [], "regions": [], "trait_refs": [], "types": [ty()]
            }, "id": 1},
            1,
            null
        ]});
        let payload = json!({
            "kind": {"Projection": [place(4), {"Field": [1, 0]}]},
            "ty": ty()
        });
        // Open, Pin, Len, Some(len), extract payload, Get: the Option
        // wrap is identity on the usize, so the Get is still own.
        let body = body_of(
            7,
            vec![
                bb(call_fun(1, vec![], 1, 1)),
                bb(call_fun(4, vec![copy(0)], 2, 2)),
                bb(call_fun(3, vec![], 3, 3)),
                bb_stmts(
                    vec![
                        assign_local(4, json!({"Aggregate": [some, [copy(3)]]})),
                        assign_local(5, json!({"Use": [{"Copy": payload}, "No"]})),
                    ],
                    call_fun(2, vec![copy(5)], 6, 4),
                ),
                bb(drop_local(1, 5)),
                bb(json!("Return")),
            ],
        );
        assert!(
            is_neutral(&body, &llbc),
            "Get of Option::Some(Len) payload must be own"
        );
    }

    #[test]
    fn field_of_param_used_as_index_is_param_slots() {
        let llbc = stack_ops_llbc();
        let field0 = json!({
            "kind": {"Projection": [place(1), {"Field": [null, 0]}]},
            "ty": ty()
        });
        // Get of a usize field of argument 0: the caller supplied that
        // value, so the body is ParamSlots not Observes.
        let callee = body_of_args(
            1,
            4,
            vec![
                bb_stmts(
                    vec![assign_local(2, json!({"Use": [{"Copy": field0}, "No"]}))],
                    call_fun(2, vec![json!({"Copy": place(2)})], 3, 1),
                ),
                bb(json!("Return")),
            ],
        );
        let walk = stack_walk(&callee, &llbc, &|_| CalleeEffect::none());
        assert!(
            !walk.effect().observes && walk.effect().param_slots == vec![0],
            "Get of param.field is ParamSlots: {:?}",
            walk.effect()
        );
        let deref_field = json!({
            "kind": {"Projection": [
                {"kind": {"Projection": [place(1), "Deref"]}, "ty": ty()},
                {"Field": [null, 0]}
            ]},
            "ty": ty()
        });
        let via_ref = body_of_args(
            1,
            4,
            vec![
                bb_stmts(
                    vec![assign_local(
                        2,
                        json!({"Use": [{"Copy": deref_field}, "No"]}),
                    )],
                    call_fun(2, vec![json!({"Copy": place(2)})], 3, 1),
                ),
                bb(json!("Return")),
            ],
        );
        let walk = stack_walk(&via_ref, &llbc, &|_| CalleeEffect::none());
        assert!(
            !walk.effect().observes && walk.effect().param_slots == vec![0],
            "Get of (*param).field is ParamSlots: {:?}",
            walk.effect()
        );
    }

    #[test]
    fn struct_return_with_index_field_is_returns_index() {
        let llbc = stack_ops_llbc();
        let copy = |i| json!({"Copy": place(i)});
        let adt = json!({"Adt": [
            {"builtin": null, "generics": {
                "const_generics": [], "regions": [], "trait_refs": [], "types": []
            }, "id": 1},
            0,
            null
        ]});
        // Len into a field of the return struct: the body returns an
        // own-index carrier.
        let helper = body_of(
            3,
            vec![
                bb(call_fun(3, vec![], 1, 1)),
                bb_stmts(
                    vec![assign_local(
                        0,
                        json!({"Aggregate": [adt, [copy(1), copy(2)]]}),
                    )],
                    json!("Return"),
                ),
            ],
        );
        let walk = stack_walk(&helper, &llbc, &|_| CalleeEffect::none());
        assert!(
            walk.returns_index && !walk.effect().observes,
            "Return of a struct holding a Len dest is returns_index: {:?}",
            walk.effect()
        );
        let llbc = stack_ops_llbc_with_callee();
        let effect = |path: &str| callee_effect_of(&llbc, path);
        // Caller treats the dest as an own-index carrier and passes it
        // to a ParamSlots callee.
        let caller = body_of(
            5,
            vec![
                bb(call_fun(1, vec![], 1, 1)),
                bb(call_fun(4, vec![copy(0)], 2, 2)),
                bb(call_fun(10, vec![], 3, 3)),
                bb(call_fun(8, vec![copy(3)], 4, 4)),
                bb(drop_local(1, 5)),
                bb(json!("Return")),
            ],
        );
        assert!(
            body_is_depth_neutral(&caller, &llbc, &effect),
            "ParamSlots of a returns_index struct dest must be own"
        );
        let ref_of = json!({
            "kind": {"Local": 3},
            "ty": ty()
        });
        let via_ref = body_of(
            6,
            vec![
                bb(call_fun(1, vec![], 1, 1)),
                bb(call_fun(4, vec![copy(0)], 2, 2)),
                bb(call_fun(10, vec![], 3, 3)),
                bb_stmts(
                    vec![assign_local(
                        4,
                        json!({"Ref": {
                            "kind": "Mut",
                            "place": ref_of,
                            "ptr_metadata": {"Const": null}
                        }}),
                    )],
                    call_fun(8, vec![copy(4)], 5, 4),
                ),
                bb(drop_local(1, 5)),
                bb(json!("Return")),
            ],
        );
        assert!(
            body_is_depth_neutral(&via_ref, &llbc, &effect),
            "ParamSlots of a &mut returns_index dest must be own"
        );
    }

    #[test]
    fn bounds_assert_not_erased_when_length_local_is_reassigned() {
        let llbc = stack_ops_llbc_with_callee();
        let array = json!({"Assign": [
            place(2),
            {"Aggregate": [
                {"Array": [ty(), {"Deduplicated": 1}, null]},
                [
                    {"Move": place(7)},
                    {"Move": place(8)},
                    {"Move": place(9)}
                ]
            ]}
        ]});
        let len_of_array = json!({"Assign": [
            place(5),
            {"Len": place(2)}
        ]});
        let unknown = json!({"Assign": [
            place(5),
            {"Use": [{"Copy": place(6)}, "Yes"]}
        ]});
        let assert_term = json!({"Assert": {
            "assert": {
                "cond": {"Copy": place(3)},
                "expected": true,
                "check_kind": {"BoundsCheck": {
                    "len": {"Copy": place(5)},
                    "index": {"Copy": place(3)}
                }}
            },
            "target": 6,
            "on_unwind": 99
        }});
        let body = body_of(
            10,
            vec![
                bb(call_fun(1, vec![], 1, 1)),
                bb_stmts(
                    vec![stmt(array)],
                    call_fun(6, vec![json!({"Move": place(2)})], 3, 2),
                ),
                bb(json!({"Switch": {
                    "discr": {"Copy": place(0)},
                    "targets": {"If": [3, 4]}
                }})),
                bb_stmts(vec![stmt(len_of_array)], json!({"Goto": {"target": 5}})),
                bb_stmts(vec![stmt(unknown)], json!({"Goto": {"target": 5}})),
                bb_stmts(vec![], assert_term),
                bb(drop_local(1, 7)),
                bb(json!("Return")),
            ],
        );
        match analyze(&body, &llbc) {
            Ok(Some(plan)) => assert!(
                !plan.terms.contains_key(&5),
                "bounds assert must not be erased when the length local is assigned on two paths"
            ),
            Err(reason) => assert_eq!(
                reason, "unmodeled-use-in-terminator",
                "keeping the bracket is the sound answer when the length local disagrees across paths"
            ),
            Ok(None) => panic!("body has a root bracket"),
        }
    }

    #[test]
    fn bounds_check_against_unrelated_length_is_refused() {
        let specials = HashMap::from([(5usize, Special::Index(0))]);
        let assert: majit_charon_reader::ullbc::AssertStmt = serde_json::from_value(json!({
            "cond": {"Copy": place(5)},
            "expected": true,
            "check_kind": {"BoundsCheck": {
                "len": {"Copy": place(9)},
                "index": {"Copy": place(5)}
            }}
        }))
        .expect("assert");
        assert!(
            !assert_is_slot_index_bounds_check(
                &assert,
                &specials,
                &HashMap::new(),
                &fixture_llbc()
            ),
            "BoundsCheck against a length that is not the erased slot array must refuse"
        );
    }

    #[test]
    fn bounds_check_of_erased_slot_index_against_slot_len_is_accepted() {
        let specials = HashMap::from([(5usize, Special::Index(0)), (6usize, Special::Index(3))]);
        let assert: majit_charon_reader::ullbc::AssertStmt = serde_json::from_value(json!({
            "cond": {"Copy": place(5)},
            "expected": true,
            "check_kind": {"BoundsCheck": {
                "len": {"Copy": place(6)},
                "index": {"Copy": place(5)}
            }}
        }))
        .expect("assert");
        assert!(
            assert_is_slot_index_bounds_check(&assert, &specials, &HashMap::new(), &fixture_llbc()),
            "index 0 of a 3-slot erased array is true by construction"
        );
    }
}
