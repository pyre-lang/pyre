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
//! (`_fix_graph_after_inlining`). A call whose resolved path is the
//! same declaration `alias_extended_functions_to_origportal` shares with the
//! rewritten origportal takes that portal's effect: the portal body is
//! RootScope-balanced. Anything the interpretation cannot model refuses
//! the whole body, which then lowers exactly as it did before.
//!
//! Unwind edges are not followed: the flow-graph builder does not lower an
//! `on_unwind` cleanup chain either, and one chain is shared by calls made at
//! different depths.

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;

use majit_charon_reader::Llbc;
use majit_charon_reader::ullbc::{
    CallFunc, FunDecl, Operand, Place, PlaceKind, ProjectionElem, RegularCall, Rvalue, SpanRef,
    StmtKind, TermKind, TyRef, Unstructured, UnwindKind, operand_json_place,
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
    /// `DictOperationGuard::new`: Open plus publish of the dict and the
    /// refs slice. The constructor returns the bracket it opened
    /// (`harvest_scope_owning_identities`); one leaf so a caller that also
    /// has its own `with_roots!` (`w_module_dict_setitem_str_internal`) can
    /// erase. Upstream: `ShadowStackFrameworkGCTransformer.push_roots`.
    GuardNew,
    /// `DictOperationGuard::root(k)`: Get of the k-th slot this guard published.
    GuardRoot,
    /// Touches the stack in a way this pass does not model.
    Unmodeled,
}

/// Classify a `gc_roots` callee, and say whether it takes the guard as
/// its first argument. `None` for a callee outside that module.
/// `FrameAnchor` ops are identities (`frame_anchor::classify`); they do
/// not touch the stack and are not leaves here.
///
/// Charon names an inherent method `gc_roots::<Impl>::leaf`, so the type a
/// method belongs to is read off the call: the receiver's type for a method,
/// the destination's for the guard's constructor.
fn classify_call(
    call: &majit_charon_reader::ullbc::CallPayload,
    llbc: &Llbc,
) -> Option<(Leaf, bool)> {
    let fd = callee_fun_decl(call, llbc)?;
    let path = fd.item_meta.name_path();
    let segments: Vec<&str> = path.split("::").collect();
    let leaf = *segments.last()?;
    let names_type = |ty: &majit_charon_reader::ullbc::TyRef, name: &str| {
        super::tyref_class_root(ty, llbc).is_some_and(|root| root.rsplit("::").next() == Some(name))
    };
    let receiver_ty = match call.args.first() {
        Some(Operand::Copy(p) | Operand::Move(p)) => Some(&p.ty),
        _ => None,
    };
    // `DictOperationGuard` lives in `dictmultiobject`, not `gc_roots`.
    // Charon names the constructor `dictmultiobject::<Impl>::new`, so the
    // type is read off the destination (and the receiver for `root`).
    if (names_type(&call.dest.ty, "DictOperationGuard") && leaf == "new")
        || (leaf == "new" && segments.contains(&"DictOperationGuard"))
    {
        return Some((Leaf::GuardNew, false));
    }
    if receiver_ty.is_some_and(|ty| names_type(ty, "DictOperationGuard"))
        || segments.contains(&"DictOperationGuard")
    {
        match leaf {
            "root" => return Some((Leaf::GuardRoot, true)),
            "drop" | "drop_in_place" => return Some((Leaf::Close, true)),
            "new" => {}
            _ => {}
        }
    }
    if !segments.contains(&super::ROOT_SCOPE_MODULE) {
        return None;
    }
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
        "reload_top_root" | "reload_top_root_at" => Leaf::ReloadTop,
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

/// `FrameAnchor` ops are identities (`frame_anchor::classify`); they do
/// not touch the shadow stack. A handler that also opens a `RootScope`
/// must still erase: treat these calls as depth-neutral and leave them
/// for the MIR identity rewrite.
fn call_is_anchor_identity(call: &majit_charon_reader::ullbc::CallPayload, llbc: &Llbc) -> bool {
    super::frame_anchor::classify(call, llbc).is_some()
}

/// Open depth named by a Guard, its Alias, or the depth word itself.
/// Second element is the Guard skew; an Index depth-word has skew 0.
fn special_open_guard(specials: &HashMap<usize, Special>, local: usize) -> Option<(usize, usize)> {
    match specials.get(&local) {
        Some(Special::Guard { depth, skew }) => Some((*depth, *skew)),
        Some(Special::Index { k, .. }) => Some((*k, 0)),
        Some(Special::Alias(g)) => match specials.get(g) {
            Some(Special::Guard { depth, skew }) => Some((*depth, *skew)),
            _ => None,
        },
        _ => None,
    }
}

/// What a special local stands for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Special {
    /// A `RootScope` guard opened at this depth, with the join skew
    /// (`hi - lo`) recorded at Open so a Close/Drop restores both.
    Guard { depth: usize, skew: usize },
    /// A borrow of a guard.
    Alias(usize),
    /// A slot index. `relative` is a `Leaf::Len` (or arithmetic on one):
    /// the offset tracks the analysed depth. Absolute indices are a
    /// guard base, a constant, or arithmetic on those.
    Index { k: usize, relative: bool },
    /// A checked-add pair whose `.0` is this offset.
    IndexPair { k: usize, relative: bool },
    /// `pin_roots` dest of an incoming `&[PyObjectRef]`
    /// (`pin_roots_incoming_slice`). Not a scalar slot: `Leaf::Len` and
    /// merge depth ignore it. `get(base + k)` is that list at `k`.
    SliceBase { slice_local: usize },
    /// `base + k` of a [`Special::SliceBase`].
    SliceIndex { slice_local: usize, k: usize },
    /// Checked-add pair whose `.0` is a [`Special::SliceIndex`].
    SliceIndexPair { slice_local: usize, k: usize },
}

/// `drop(roots)` is `core::mem::drop` of the guard. rustc emits a Call that
/// moves the `RootScope`, not a `Drop` terminator, so NLL can keep the
/// bracket live across the preceding call. That Call is the same rewind as
/// `TermKind::Drop` / `root_scope_close`. Only the full paths
/// `core::mem::drop` and `core::ptr::drop_in_place` match.
fn mem_drop_scope_local(
    call: &majit_charon_reader::ullbc::CallPayload,
    llbc: &Llbc,
) -> Option<usize> {
    let path = callee_path(call, llbc)?;
    if path != "core::mem::drop" && path != "core::ptr::drop_in_place" {
        return None;
    }
    match call.args.first()? {
        Operand::Move(place) | Operand::Copy(place) => place_local(place),
        _ => None,
    }
}

fn mem_drop_guard(
    call: &majit_charon_reader::ullbc::CallPayload,
    llbc: &Llbc,
    specials: &HashMap<usize, Special>,
) -> Option<(usize, usize)> {
    let local = mem_drop_scope_local(call, llbc)?;
    match specials.get(&local) {
        Some(Special::Guard { depth, skew }) => Some((*depth, *skew)),
        Some(Special::Alias(g)) => match specials.get(g) {
            Some(Special::Guard { depth, skew }) => Some((*depth, *skew)),
            _ => None,
        },
        _ => None,
    }
}

fn mem_drop_guard_depth(
    call: &majit_charon_reader::ullbc::CallPayload,
    llbc: &Llbc,
    specials: &HashMap<usize, Special>,
) -> Option<usize> {
    mem_drop_guard(call, llbc, specials).map(|(d, _)| d)
}

impl Special {
    fn as_index(self) -> Option<(usize, bool)> {
        match self {
            Special::Index { k, relative } | Special::IndexPair { k, relative } => {
                Some((k, relative))
            }
            _ => None,
        }
    }

    fn index(k: usize, relative: bool, checked: bool) -> Self {
        if checked {
            Special::IndexPair { k, relative }
        } else {
            Special::Index { k, relative }
        }
    }
}

/// Compatible rebind of one local: same `Index`/`IndexPair` `k` (AND of
/// `relative`) or same `Guard` depth (max of `skew`). Anything else is
/// `special-rebound`.
fn merge_special(previous: Special, new: Special) -> Result<Special, Refusal> {
    match (previous, new) {
        (
            Special::Index {
                k: k1,
                relative: r1,
            },
            Special::Index {
                k: k2,
                relative: r2,
            },
        ) if k1 == k2 => Ok(Special::Index {
            k: k1,
            relative: r1 && r2,
        }),
        (
            Special::IndexPair {
                k: k1,
                relative: r1,
            },
            Special::IndexPair {
                k: k2,
                relative: r2,
            },
        ) if k1 == k2 => Ok(Special::IndexPair {
            k: k1,
            relative: r1 && r2,
        }),
        (
            Special::Guard {
                depth: d1,
                skew: s1,
            },
            Special::Guard {
                depth: d2,
                skew: s2,
            },
        ) if d1 == d2 => Ok(Special::Guard {
            depth: d1,
            skew: s1.max(s2),
        }),
        (a, b) if a == b => Ok(a),
        _ => Err("special-rebound"),
    }
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
    /// Per-slot type. RootScope pins `PyObjectRef`. A mixed body needs
    /// both when a Guard also publishes a different pointee.
    slot_tys: HashMap<usize, Value>,
    /// Blocks no path from the entry reaches once the rewritten calls lose
    /// their unwind edges.
    unreachable: Vec<usize>,
}

type Refusal = &'static str;

/// Record the pointee type at `depth`. A reused depth with a different
/// type is a refusal: first-type-wins would leave Get/Set on the earlier
/// local while a later pin writes the later type.
fn record_slot_ty(plan: &mut Plan, depth: usize, ty: &Value) -> Result<(), Refusal> {
    match plan.slot_tys.entry(depth) {
        std::collections::hash_map::Entry::Vacant(v) => {
            v.insert(ty.clone());
            Ok(())
        }
        std::collections::hash_map::Entry::Occupied(o) => {
            if o.get() == ty {
                Ok(())
            } else {
                Err("slot-type-mismatch")
            }
        }
    }
}

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
/// are not definitions. Only blocks in `forward_blocks` are counted: an
/// `on_unwind` cleanup block is still in the body.
fn local_def_counts(body: &Unstructured, llbc: &Llbc, forward_blocks: &[bool]) -> Vec<u32> {
    let n = body.locals.locals.len();
    let mut counts = vec![0u32; n];
    let bump = |counts: &mut [u32], l: Option<usize>| {
        if let Some(l) = l
            && let Some(c) = counts.get_mut(l)
        {
            *c = c.saturating_add(1);
        }
    };
    for (bb, block) in body.body.iter().enumerate() {
        if !forward_blocks[bb] {
            continue;
        }
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
                        } if ref_kind_mutates(kind) => {
                            bump(
                                &mut counts,
                                place_local(src).or_else(|| deref_of_local(src)),
                            );
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
    match operand_json_place(op) {
        Some(place) => json!({ "Read": place.clone() }),
        None => op.clone(),
    }
}

fn operand_json_ty(value: &Value) -> Option<Value> {
    operand_json_place(value).and_then(|p| p.get("ty")).cloned()
}

fn local_place_json(local: usize, ty: &Value) -> Value {
    json!({ "kind": { "Local": local }, "ty": ty })
}

fn assign_json(dest: Value, rvalue: Value) -> Value {
    json!({ "Assign": [dest, rvalue] })
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
        Some(Some(
            Special::IndexPair { .. } | Special::SliceIndexPair { .. }
        ))
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
        [Special::Index { k, .. }] | [Special::IndexPair { k, .. }] => Some(*k),
        [Special::SliceBase { .. }] => Some(0),
        [Special::SliceIndex { k, .. } | Special::SliceIndexPair { k, .. }] => Some(*k),
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
        [Special::Index { k: d, .. }] | [Special::IndexPair { k: d, .. }] => Some(*d),
        _ => None,
    }
}

fn json_operand_local(v: &Value) -> Option<usize> {
    operand_json_place(v)
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

/// How a closure body names the slot it reads.
///
/// `Capture` / `CapturePlusArg` are [`super::function_is_root_slot_getter`].
/// `CapturePlusConst` is `|| shadow_stack_get(captured + k)`: the addend is a
/// literal in the body, which that helper traces as opaque.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlotGetter {
    Capture,
    CapturePlusArg,
    CapturePlusConst(u64),
}

/// Classify a callee as a root-slot getter.  Reuses
/// [`super::function_is_root_slot_getter`] for the two kinds `RootBracketPlan`
/// already answers, then ports `classify_root_slot_getter_body` for a
/// `captured + const` body.
fn slot_getter_of(
    llbc: &Llbc,
    fun_id: u64,
    cache: &mut HashMap<u64, Option<SlotGetter>>,
) -> Option<SlotGetter> {
    if let Some(&cached) = cache.get(&fun_id) {
        return cached;
    }
    let name_of = |reg: &RegularCall| super::regular_call_name_path(reg, llbc);
    let got = match super::function_is_root_slot_getter(llbc, fun_id, &name_of) {
        Some(super::RootSlotGetter::Capture) => Some(SlotGetter::Capture),
        Some(super::RootSlotGetter::CapturePlusArg) => Some(SlotGetter::CapturePlusArg),
        None => classify_capture_plus_const(llbc, fun_id, &name_of),
    };
    cache.insert(fun_id, got);
    got
}

/// `classify_root_slot_getter_body` with a `Const` usize treated as an addend
/// rather than an opaque leaf, so `|| shadow_stack_get(base + 1)` is a Get of
/// slot `k + 1`.
fn classify_capture_plus_const(
    llbc: &Llbc,
    fun_id: u64,
    name_of: &impl Fn(&RegularCall) -> Option<String>,
) -> Option<SlotGetter> {
    let fd = llbc.fn_by_id(fun_id)?;
    let raw = fd.body.as_ref()?;
    if raw.get().len() > 16 * 1024 {
        return None;
    }
    let body = fd.unstructured()?;
    if body.body.len() > 8 || body.locals.arg_count != 2 {
        return None;
    }
    let mut defs: HashMap<usize, Rvalue> = HashMap::new();
    let mut getter_arg: Option<Operand> = None;
    let mut getter_call: Option<(usize, usize)> = None;
    let mut calls = 0usize;
    let forward_blocks = super::forward_reachable_mask(llbc, &body);
    for (bb_idx, bb) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        for stmt in &bb.statements {
            match stmt.stmt_kind_ref().ok()? {
                StmtKind::StorageLive(_)
                | StmtKind::StorageDead(_)
                | StmtKind::Borrowck(_)
                | StmtKind::PlaceMention(_)
                | StmtKind::Assert(_) => {}
                StmtKind::Assign(place, value) => {
                    let PlaceKind::Local(dest) = place.kind else {
                        return None;
                    };
                    let dest = dest as usize;
                    if dest <= 2 || defs.insert(dest, value.clone()).is_some() {
                        return None;
                    }
                    if !super::root_slot_rvalue_is_traceable(value) {
                        return None;
                    }
                }
                StmtKind::Unknown => return None,
            }
        }
        let term = serde_json::from_str::<TermKind>(bb.terminator.kind.get()).ok()?;
        match term {
            TermKind::Assert { .. }
            | TermKind::Goto { .. }
            | TermKind::Return
            | TermKind::UnwindResume
            | TermKind::UnwindTerminate => {}
            TermKind::Call { call, target, .. } => {
                calls += 1;
                if calls > 1 {
                    return None;
                }
                let CallFunc::Regular(reg) = &call.func else {
                    return None;
                };
                let path = name_of(reg)?;
                let leaf = path.rsplit("::").next().unwrap_or("");
                if leaf != "shadow_stack_get"
                    || !path
                        .split("::")
                        .any(|segment| segment == super::ROOT_SCOPE_MODULE)
                    || call.args.len() != 1
                    || !matches!(call.dest.kind, PlaceKind::Local(0))
                {
                    return None;
                }
                getter_call = Some((bb_idx, target as usize));
                getter_arg = call.args.into_iter().next();
            }
            TermKind::Drop { .. }
            | TermKind::Switch { .. }
            | TermKind::Abort(_)
            | TermKind::Panic { .. }
            | TermKind::UndefinedBehavior
            | TermKind::Unknown => return None,
        }
    }
    let (getter_bb, normal_target) = getter_call?;
    if !super::getter_returns_follow_call(llbc, &body, getter_bb, normal_target) {
        return None;
    }
    let trace = const_slot_trace_operand(
        llbc,
        &defs,
        getter_arg.as_ref()?,
        &mut bit_set::BitSet::new(),
    );
    if trace.opaque || trace.env != 1 || trace.arg != 0 || trace.adds != 1 {
        return None;
    }
    Some(SlotGetter::CapturePlusConst(trace.const_sum))
}

#[derive(Clone, Copy)]
struct ConstSlotTrace {
    env: u32,
    arg: u32,
    adds: u32,
    const_sum: u64,
    opaque: bool,
}

impl ConstSlotTrace {
    fn env_leaf() -> Self {
        Self {
            env: 1,
            arg: 0,
            adds: 0,
            const_sum: 0,
            opaque: false,
        }
    }

    fn arg_leaf() -> Self {
        Self {
            env: 0,
            arg: 1,
            adds: 0,
            const_sum: 0,
            opaque: false,
        }
    }

    fn const_leaf(n: u64) -> Self {
        Self {
            env: 0,
            arg: 0,
            adds: 0,
            const_sum: n,
            opaque: false,
        }
    }

    fn opaque() -> Self {
        Self {
            env: 0,
            arg: 0,
            adds: 0,
            const_sum: 0,
            opaque: true,
        }
    }

    fn join_add(self, other: Self) -> Self {
        Self {
            env: self.env.saturating_add(other.env),
            arg: self.arg.saturating_add(other.arg),
            adds: self.adds.saturating_add(other.adds).saturating_add(1),
            const_sum: self.const_sum.saturating_add(other.const_sum),
            opaque: self.opaque || other.opaque,
        }
    }
}

fn const_slot_trace_operand(
    llbc: &Llbc,
    defs: &HashMap<usize, Rvalue>,
    op: &Operand,
    visiting: &mut bit_set::BitSet,
) -> ConstSlotTrace {
    match op {
        Operand::Copy(place) | Operand::Move(place) => {
            const_slot_trace_place(llbc, defs, place, visiting)
        }
        Operand::Const(_) => match const_usize(op, llbc) {
            Some(n) => ConstSlotTrace::const_leaf(n as u64),
            None => ConstSlotTrace::opaque(),
        },
    }
}

fn const_slot_trace_place(
    llbc: &Llbc,
    defs: &HashMap<usize, Rvalue>,
    place: &Place,
    visiting: &mut bit_set::BitSet,
) -> ConstSlotTrace {
    let Some(root) = super::place_root_through_deref_or_field(place) else {
        return ConstSlotTrace::opaque();
    };
    const_slot_trace_local(llbc, defs, root, visiting)
}

fn const_slot_trace_local(
    llbc: &Llbc,
    defs: &HashMap<usize, Rvalue>,
    local: usize,
    visiting: &mut bit_set::BitSet,
) -> ConstSlotTrace {
    if local == 1 {
        return ConstSlotTrace::env_leaf();
    }
    if local == 2 {
        return ConstSlotTrace::arg_leaf();
    }
    if !visiting.insert(local) {
        return ConstSlotTrace::opaque();
    }
    let Some(value) = defs.get(&local) else {
        visiting.remove(local);
        return ConstSlotTrace::opaque();
    };
    let trace = match value {
        Rvalue::Use(op, _) => const_slot_trace_operand(llbc, defs, op, visiting),
        Rvalue::BinaryOp(op, lhs, rhs) if super::root_slot_sum_is_checked(op).is_some() => {
            const_slot_trace_operand(llbc, defs, lhs, visiting)
                .join_add(const_slot_trace_operand(llbc, defs, rhs, visiting))
        }
        _ => ConstSlotTrace::opaque(),
    };
    visiting.remove(local);
    trace
}

/// What a rewritten getter call answers.
enum GetterAnswer {
    /// `|| shadow_stack_get(captured)` of a scalar slot. `relative` is
    /// inherited from the captured [`Special::Index`] (a Len vs a base
    /// or constant).
    Scalar { k: usize, relative: bool },
    /// `|| shadow_stack_get(captured + k)` of an incoming slice
    /// (`pin_roots_incoming_slice`).
    Slice { slice_local: usize, k: usize },
}

/// Peel a closure call of a [`SlotGetter`] down to the static slot index and
/// the temporaries that only build the capture (`peel_closure_capture`,
/// `peel_captured_index`, `peel_getter_addend`).
fn try_slot_getter_call(
    body: &Unstructured,
    llbc: &Llbc,
    call: &majit_charon_reader::ullbc::CallPayload,
    specials: &HashMap<usize, Special>,
    assigned: &HashMap<usize, usize>,
    call_dests: &bit_set::BitSet,
    forward_blocks: &[bool],
    cache: &mut HashMap<u64, Option<SlotGetter>>,
) -> Option<(GetterAnswer, Vec<usize>)> {
    let CallFunc::Regular(reg) = &call.func else {
        return None;
    };
    let fun_id = super::regular_call_fun_decl_id(&reg.kind)?;
    let kind = slot_getter_of(llbc, fun_id, cache)?;
    let receiver = call.args.first()?;
    let mut bases = HashMap::new();
    let mut candidates = bit_set::BitSet::new();
    for (local, spec) in specials {
        if matches!(
            spec,
            Special::Index { .. }
                | Special::IndexPair { .. }
                | Special::SliceBase { .. }
                | Special::SliceIndex { .. }
                | Special::SliceIndexPair { .. }
        ) {
            bases.insert(*local, 0);
            candidates.insert(0);
        }
    }
    if bases.is_empty() {
        return None;
    }
    let offsets: HashMap<usize, super::RootSlotSum> = HashMap::new();
    let len_index: HashMap<usize, (usize, usize)> = HashMap::new();
    let aliases: HashMap<usize, usize> = HashMap::new();
    let index = super::RootGetterIndex {
        assigned,
        bases: &bases,
        offsets: &offsets,
        len_index: &len_index,
        candidates: &candidates,
        aliases: &aliases,
        call_dests,
        forward_blocks,
    };
    let peeled = match super::peel_closure_capture(llbc, body, &index, receiver) {
        super::CapturePeel::Ready(peeled) => peeled,
        _ => return None,
    };
    let (index_local, _, index_carriers) =
        super::peel_captured_index(llbc, body, &index, &peeled.capture)?;
    let peel_kind = match kind {
        SlotGetter::Capture | SlotGetter::CapturePlusConst(_) => super::RootSlotGetter::Capture,
        SlotGetter::CapturePlusArg => super::RootSlotGetter::CapturePlusArg,
    };
    let addend = super::peel_getter_addend(llbc, body, &index, peel_kind, &call.args)?;
    let body_addend = match kind {
        SlotGetter::CapturePlusConst(n) => n,
        _ => 0,
    };
    let extra = usize::try_from(body_addend)
        .ok()?
        .checked_add(usize::try_from(addend.addend).ok()?)?;
    let answer = match specials.get(&index_local) {
        Some(Special::Index { k, relative } | Special::IndexPair { k, relative }) => {
            GetterAnswer::Scalar {
                k: k.checked_add(extra)?,
                relative: *relative,
            }
        }
        Some(Special::SliceBase { slice_local }) => GetterAnswer::Slice {
            slice_local: *slice_local,
            k: extra,
        },
        Some(
            Special::SliceIndex { slice_local, k } | Special::SliceIndexPair { slice_local, k },
        ) => GetterAnswer::Slice {
            slice_local: *slice_local,
            k: k.checked_add(extra)?,
        },
        _ => return None,
    };
    let mut carriers = peeled.carriers;
    carriers.extend(index_carriers);
    carriers.extend(addend.carriers);
    if carriers
        .iter()
        .any(|local| *local == index_local || specials.contains_key(local))
    {
        return None;
    }
    Some((answer, carriers))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum WalkMode {
    Probe,
    Emit,
}

/// One walk of the erase CFG. `Probe` pushes the per-block entry lattice
/// (`depth_in` / `lo_in` / `open_in`) to a fixpoint and does not write
/// the rewrite `Plan`. `Emit` walks each reachable block once at those
/// entry values and fills the plan.
///
/// `gctransform/framework.py` computes livevars per block from the
/// finished graph before `push_roots` inserts the stores; the analysis
/// never rewrites while it is still iterating.
struct EraseWalk<'a> {
    mode: WalkMode,
    body: &'a Unstructured,
    llbc: &'a Llbc,
    classified: &'a HashMap<usize, (Leaf, bool)>,
    forward_blocks: &'a [bool],
    succs: &'a [Vec<u64>],
    n_blocks: usize,
    n_locals: usize,
    def_counts: &'a [u32],
    stmt_assigned: &'a HashMap<usize, usize>,
    call_dests: &'a bit_set::BitSet,
    depth_in: &'a mut Vec<Option<usize>>,
    lo_in: &'a mut Vec<Option<usize>>,
    open_in: &'a mut Vec<Option<usize>>,
    frozen_depth: &'a [Option<usize>],
    frozen_lo: &'a [Option<usize>],
    queue: &'a mut VecDeque<usize>,
    visited: &'a mut [bool],
    specials: &'a mut HashMap<usize, Special>,
    plan: Option<&'a mut Plan>,
    const_locals: &'a mut HashMap<usize, usize>,
    array_lens: &'a mut HashMap<usize, usize>,
    getter_cache: &'a mut HashMap<u64, Option<SlotGetter>>,
    getter_bbs: &'a mut bit_set::BitSet,
    capture_temps: &'a mut bit_set::BitSet,
}

impl<'a> EraseWalk<'a> {
    fn emit(&self) -> bool {
        self.mode == WalkMode::Emit
    }

    fn slot_local(&self, k: usize) -> usize {
        self.n_locals + k
    }

    fn single_def(&self, dest: usize) -> bool {
        self.def_counts.get(dest).copied() == Some(1)
    }

    /// Bind one local to a special. Same `Index`/`IndexPair` `k` or same
    /// `Guard` depth is a compatible rebind via [`merge_special`]. Absolute
    /// wins (`relative: a && b`) so an absolute Get still sees the join
    /// floor; a larger skew is the lower floor. Probe overwrites an
    /// incompatible pair because it re-walks blocks at a new lattice
    /// (Open then binds a new `Guard` depth); Emit walks once and refuses.
    fn bind(&mut self, local: usize, special: Special) -> Result<(), Refusal> {
        match self.specials.get(&local).copied() {
            None => {
                self.specials.insert(local, special);
                Ok(())
            }
            Some(previous) => match merge_special(previous, special) {
                Ok(merged) => {
                    self.specials.insert(local, merged);
                    Ok(())
                }
                Err(_) if !self.emit() => {
                    self.specials.insert(local, special);
                    Ok(())
                }
                Err(reason) => Err(reason),
            },
        }
    }

    fn check_slot_read(
        &self,
        k: usize,
        relative: bool,
        depth: usize,
        lo: usize,
        set: bool,
    ) -> Result<(), Refusal> {
        if !self.emit() {
            return Ok(());
        }
        if relative || lo == depth {
            if k >= depth {
                return Err(if set {
                    "set-above-depth"
                } else {
                    "get-above-depth"
                });
            }
        } else if k >= lo {
            return Err("absolute-get-above-join-floor");
        }
        Ok(())
    }

    fn reach(
        &mut self,
        from: usize,
        target: u64,
        depth: usize,
        lo: usize,
        open: usize,
    ) -> Result<(), Refusal> {
        let target = target as usize;
        if target >= self.n_blocks {
            return Ok(());
        }
        match self.mode {
            WalkMode::Emit => {
                // Probe joins every successor, including Drop/Close, so
                // the frozen lattice is the fixpoint. Incoming must sit
                // inside it; a miss is a two-phase bug mapped to
                // `depth-disagrees-at-merge` in release.
                let frozen_hi = self.frozen_depth[target];
                let frozen_lo = self.frozen_lo[target];
                debug_assert!(
                    frozen_hi.is_some_and(|h| depth <= h),
                    "emit incoming depth exceeds phase-1 lattice"
                );
                debug_assert!(
                    frozen_lo.is_some_and(|l| lo >= l),
                    "emit incoming lo is below phase-1 lattice"
                );
                if !frozen_hi.is_some_and(|h| depth <= h) || !frozen_lo.is_some_and(|l| lo >= l) {
                    return Err("depth-disagrees-at-merge");
                }
                if !self.visited[target] {
                    self.queue.push_back(target);
                }
                Ok(())
            }
            WalkMode::Probe => {
                // Scalar depth only: an incoming slice is not a slot
                // (`pin_roots_incoming_slice`). A Drop of the guard rewinds to
                // the open depth, so a slice-only early return may join a
                // `pin_root` arm there (`__majit_wrap_dict_descr_keys`).
                // Join the Drop/Close entry too: skipping it leaves the
                // first-visit lattice, and Emit then walks the rewind at
                // that depth with the final Guard specials.
                match (self.depth_in[target], self.lo_in[target]) {
                    (Some(d), Some(l)) if d == depth && l == lo => Ok(()),
                    (None, _) => {
                        self.depth_in[target] = Some(depth);
                        self.lo_in[target] = Some(lo);
                        self.open_in[target] = Some(open);
                        self.queue.push_back(target);
                        Ok(())
                    }
                    (Some(d), Some(l)) => {
                        // Incoming depths disagree at a CFG join, including a
                        // Drop/Close: that block postdominates itself, so the
                        // join is accepted and `slot_count` takes max.
                        // `stack_walk` `Depth::join` takes a min lower bound for
                        // neutrality; this walk takes max for `slot_count`.
                        // `gctransform/shadowstack.py`
                        // `ShadowStackFrameworkGCTransformer.push_roots` /
                        // `gctransform/framework.py` `get_livevars_for_roots` is
                        // one livevar set around the collecting op, not a
                        // per-arm stack pointer. Extra-pin Gets stay on the
                        // writing arm (SSA slot locals). A back edge that grows
                        // depth is pins accumulating around a loop: refuse
                        // rather than let the lattice climb.
                        // `gctransform/framework.py` computes livevars per block
                        // from the finished graph before `push_roots` inserts the
                        // stores; the analysis never rewrites while it is still
                        // iterating.
                        if d != depth && dominates(self.succs, self.n_blocks, target, from) {
                            return Err("depth-disagrees-at-merge");
                        }
                        if d != depth
                            && !join_postdominated_by_this_guard_rewind(
                                self.body,
                                self.llbc,
                                self.classified,
                                self.specials,
                                self.succs,
                                self.n_blocks,
                                target,
                            )
                        {
                            return Err("depth-disagrees-at-merge");
                        }
                        let joined_hi = d.max(depth);
                        let joined_lo = l.min(lo);
                        if joined_hi != d || joined_lo != l {
                            self.depth_in[target] = Some(joined_hi);
                            self.lo_in[target] = Some(joined_lo);
                            if depth == joined_hi {
                                self.open_in[target] = Some(open);
                            }
                            self.visited[target] = false;
                            self.queue.push_back(target);
                        }
                        Ok(())
                    }
                    _ => Ok(()),
                }
            }
        }
    }

    fn guard_open(&self, g: usize) -> Result<(usize, usize), Refusal> {
        match self.specials.get(&g) {
            Some(Special::Guard { depth, skew }) => Ok((*depth, *skew)),
            _ => Err("guard-not-opened"),
        }
    }

    fn slot_index_of(
        &self,
        call: &majit_charon_reader::ullbc::CallPayload,
        i: usize,
        llbc: &Llbc,
    ) -> Result<(usize, bool), Refusal> {
        if let Some((k, relative)) = call
            .args
            .get(i)
            .and_then(operand_local)
            .and_then(|l| self.specials.get(&l))
            .and_then(|s| s.as_index())
        {
            return Ok((k, relative));
        }
        if let Some(k) = call.args.get(i).and_then(|op| const_usize(op, llbc)) {
            return Ok((k, false));
        }
        Err("index-not-static")
    }

    fn record_removed(&mut self, bb: usize, si: usize) {
        if let Some(plan) = self.plan.as_mut() {
            plan.removed.entry(bb).or_default().push(si);
        }
    }

    fn record_term(&mut self, bb: usize, target: u64, stmts: Vec<Value>) {
        if let Some(plan) = self.plan.as_mut() {
            plan.terms.insert(bb, TermRewrite { target, stmts });
        }
    }

    fn insert_after(&mut self, key: (usize, usize), inserted: Vec<Value>) {
        if let Some(plan) = self.plan.as_mut() {
            plan.inserted_after.entry(key).or_default().extend(inserted);
        }
    }

    fn bump_slot_count(&mut self, depth: usize) {
        if let Some(plan) = self.plan.as_mut() {
            plan.slot_count = plan.slot_count.max(depth);
        }
    }

    fn ensure_slot_ty(&mut self, ty: Value) {
        if let Some(plan) = self.plan.as_mut() {
            plan.slot_ty.get_or_insert(ty);
        }
    }

    fn record_slot_ty(&mut self, depth: usize, ty: &Value) -> Result<(), Refusal> {
        if let Some(plan) = self.plan.as_mut() {
            record_slot_ty(plan, depth, ty)?;
        }
        Ok(())
    }

    fn slot_ty_at(&self, k: usize) -> Option<Value> {
        self.plan
            .as_ref()
            .and_then(|p| p.slot_tys.get(&k).cloned().or_else(|| p.slot_ty.clone()))
    }
}

/// A pin raises `hi`. It raises `lo` only while the stack is aligned
/// (`lo == hi`): after a disagreeing join the floor is the slots every
/// arm holds, and later pins land at different source indices on the
/// short arm, so the `[lo, hi)` band stays absolute-unsafe.
fn pin_depth(depth: &mut usize, lo: &mut usize, n: usize) {
    if *lo == *depth {
        *lo = lo.saturating_add(n);
    }
    *depth = depth.saturating_add(n);
}

fn run_erase_walk(cx: &mut EraseWalk<'_>) -> Result<(), Refusal> {
    let body = cx.body;
    let llbc = cx.llbc;
    let n_blocks = cx.n_blocks;
    let classified = cx.classified;
    let mut rounds = 0usize;
    while let Some(bb) = cx.queue.pop_front() {
        rounds += 1;
        // Lattice is finite: hi rises only at disagreeing forward joins
        // and back-edge growth refuses; lo falls toward 0.
        debug_assert!(
            rounds
                <= n_blocks
                    .saturating_mul(n_blocks.saturating_add(8))
                    .saturating_add(64),
            "erase depth lattice should terminate"
        );
        if cx.visited[bb] {
            continue;
        }
        cx.visited[bb] = true;
        let (mut depth, mut lo, mut open) = match cx.mode {
            WalkMode::Emit => (
                cx.frozen_depth[bb].expect("emit block is phase-1 reachable"),
                cx.frozen_lo[bb].expect("emit block is phase-1 reachable"),
                cx.open_in[bb].expect("queued blocks have an open count"),
            ),
            WalkMode::Probe => (
                cx.depth_in[bb].expect("queued blocks have a depth"),
                cx.lo_in[bb].expect("queued blocks have a lo"),
                cx.open_in[bb].expect("queued blocks have an open count"),
            ),
        };
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
                    if cx.single_def(dest) {
                        cx.array_lens.insert(dest, operands.len());
                    }
                }
                Rvalue::Len(src) => {
                    let base = place_local(src).or_else(|| deref_of_local(src));
                    if cx.single_def(dest)
                        && let Some(n) = base.and_then(|l| cx.array_lens.get(&l).copied())
                    {
                        cx.const_locals.insert(dest, n);
                    }
                }
                Rvalue::Ref { place: src, .. } | Rvalue::RawPtr { place: src, .. } => {
                    let base = place_local(src).or_else(|| deref_of_local(src));
                    if cx.single_def(dest)
                        && let Some(n) = base.and_then(|l| cx.array_lens.get(&l).copied())
                    {
                        cx.array_lens.insert(dest, n);
                    }
                }
                // Cast / UnaryOp may change the value (narrowing, negation,
                // bit-not). `constfold.fold_op_list` evaluates them; this
                // pass does not, so the fact is dropped.
                Rvalue::Use(Operand::Copy(src) | Operand::Move(src), _) => {
                    if cx.single_def(dest)
                        && let Some((pair, 1)) = tuple_field_of_local(src)
                        && let Some(n) = cx.array_lens.get(&pair).copied()
                    {
                        cx.const_locals.insert(dest, n);
                    }
                }
                _ => {}
            }
            let special = match rvalue {
                Rvalue::Ref { place: src, .. } | Rvalue::RawPtr { place: src, .. } => {
                    let guard = place_local(src)
                        .and_then(|l| match cx.specials.get(&l) {
                            Some(Special::Guard { .. }) => Some(l),
                            _ => None,
                        })
                        .or_else(|| {
                            deref_of_local(src).and_then(|l| match cx.specials.get(&l) {
                                Some(Special::Alias(g)) => Some(*g),
                                _ => None,
                            })
                        });
                    guard.map(Special::Alias)
                }
                Rvalue::Use(op, _) => match op {
                    Operand::Copy(src) | Operand::Move(src) => {
                        if let Some(l) = place_local(src) {
                            if cx.single_def(dest) {
                                if let Some(n) = cx.const_locals.get(&l).copied() {
                                    cx.const_locals.insert(dest, n);
                                }
                                if let Some(n) = cx.array_lens.get(&l).copied() {
                                    cx.array_lens.insert(dest, n);
                                }
                            }
                            match cx.specials.get(&l).copied() {
                                Some(Special::Index { k, relative }) => {
                                    Some(Special::Index { k, relative })
                                }
                                Some(Special::Alias(g)) => Some(Special::Alias(g)),
                                Some(Special::Guard { depth, skew }) => {
                                    Some(Special::Guard { depth, skew })
                                }
                                Some(Special::SliceBase { slice_local }) => {
                                    Some(Special::SliceBase { slice_local })
                                }
                                Some(Special::SliceIndex { slice_local, k }) => {
                                    Some(Special::SliceIndex { slice_local, k })
                                }
                                Some(Special::SliceIndexPair { slice_local, k }) => {
                                    Some(Special::SliceIndexPair { slice_local, k })
                                }
                                _ => None,
                            }
                        } else if let Some((base, field)) = tuple_field_of_local(src) {
                            match (field, cx.specials.get(&base).copied()) {
                                (0, Some(Special::IndexPair { k, relative })) => {
                                    Some(Special::Index { k, relative })
                                }
                                (0, Some(Special::SliceIndexPair { slice_local, k })) => {
                                    Some(Special::SliceIndex { slice_local, k })
                                }
                                (0, Some(Special::Guard { depth, .. })) => Some(Special::Index {
                                    k: depth,
                                    relative: false,
                                }),
                                (0, Some(Special::Index { k, relative })) => {
                                    Some(Special::Index { k, relative })
                                }
                                _ => None,
                            }
                        } else {
                            None
                        }
                    }
                    Operand::Const(_) => {
                        if cx.single_def(dest)
                            && let Some(n) = const_usize(op, llbc)
                        {
                            cx.const_locals.insert(dest, n);
                        }
                        None
                    }
                },
                Rvalue::BinaryOp(op, lhs, rhs) => match binop_is_index_arith(op) {
                    Some(arith) => {
                        let spec_of = |o: &Operand| {
                            operand_local(o).and_then(|l| cx.specials.get(&l).copied())
                        };
                        let slice_add = |spec: Option<Special>, other: &Operand, checked: bool| {
                            let c = const_usize(other, llbc)?;
                            match spec {
                                Some(Special::SliceBase { slice_local }) => Some(if checked {
                                    Special::SliceIndexPair { slice_local, k: c }
                                } else {
                                    Special::SliceIndex { slice_local, k: c }
                                }),
                                Some(Special::SliceIndex { slice_local, k }) => {
                                    let k = k.checked_add(c)?;
                                    Some(if checked {
                                        Special::SliceIndexPair { slice_local, k }
                                    } else {
                                        Special::SliceIndex { slice_local, k }
                                    })
                                }
                                _ => None,
                            }
                        };
                        if let Some(special) = match arith {
                            IndexArith::Add { checked } => slice_add(spec_of(lhs), rhs, checked)
                                .or_else(|| slice_add(spec_of(rhs), lhs, checked)),
                            IndexArith::Sub { .. } => None,
                        } {
                            Some(special)
                        } else {
                            let index_of = |o: &Operand| match spec_of(o) {
                                Some(
                                    Special::Index { k, relative }
                                    | Special::IndexPair { k, relative },
                                ) => Some((k, relative)),
                                _ => None,
                            };
                            // `len + k` and `len - k` are compile-time offsets
                            // from the depth at the Len/Base. `zip_two_tuple_next`
                            // names a just-pinned slot as `shadow_stack_len() - 1`.
                            let offset = match (arith, index_of(lhs), index_of(rhs)) {
                                (IndexArith::Add { checked }, Some((k, relative)), None) => {
                                    const_usize(rhs, llbc).map(|c| (k + c, checked, relative))
                                }
                                (IndexArith::Add { checked }, None, Some((k, relative))) => {
                                    const_usize(lhs, llbc).map(|c| (k + c, checked, relative))
                                }
                                (IndexArith::Sub { checked }, Some((k, relative)), None) => {
                                    const_usize(rhs, llbc).and_then(|c| {
                                        k.checked_sub(c).map(|n| (n, checked, relative))
                                    })
                                }
                                _ => None,
                            };
                            offset
                                .map(|(k, checked, relative)| Special::index(k, relative, checked))
                        }
                    }
                    None => None,
                },
                _ => None,
            };
            if let Some(special) = special {
                cx.bind(dest, special)?;
                cx.record_removed(bb, si);
            }
        }
        let term_json = block.terminator.kind_value();
        match block.term_ref(llbc) {
            Ok(TermKind::Call { call, target, .. }) => {
                let Some(&(leaf, method)) = classified.get(&bb) else {
                    if let Some((d, skew)) = mem_drop_guard(call, llbc, &cx.specials) {
                        if d > depth {
                            return Err("close-above-depth");
                        }
                        depth = d;
                        lo = d - skew;
                        open = open.saturating_sub(1);
                        cx.record_term(bb, *target, Vec::new());
                        cx.reach(bb, *target, depth, lo, open)?;
                        continue;
                    }
                    // A closure whose body is `shadow_stack_get` of a captured
                    // index (or `captured + const`) is a Get of that slot.
                    // `RootBracketPlan.capture_temps` already answers this at
                    // lowering; the codewriter must not see the bracket
                    // (`gctransform` `push_roots` runs after `make_jitcodes`).
                    if let Some((answer, carriers)) = try_slot_getter_call(
                        body,
                        llbc,
                        call,
                        &cx.specials,
                        &cx.stmt_assigned,
                        cx.call_dests,
                        cx.forward_blocks,
                        cx.getter_cache,
                    ) {
                        let dest_json = &term_json["Call"]["dest"];
                        let stmts = match answer {
                            GetterAnswer::Scalar { k, relative } => {
                                cx.check_slot_read(k, relative, depth, lo, false)?;
                                if !cx.emit() {
                                    cx.reach(bb, *target, depth, lo, open)?;
                                    continue;
                                }
                                let ty = cx.slot_ty_at(k).ok_or("get-before-any-pin")?;
                                vec![assign_json(
                                    dest_json.clone(),
                                    json!({ "Use": [{ "Read": local_place_json(cx.slot_local(k), &ty) }, "Yes"] }),
                                )]
                            }
                            GetterAnswer::Slice { slice_local, k } => {
                                if cx.emit() && open == 0 {
                                    return Err("get-above-depth");
                                }
                                if !cx.emit() {
                                    cx.reach(bb, *target, depth, lo, open)?;
                                    continue;
                                }
                                vec![assign_json(
                                    dest_json.clone(),
                                    slice_elem_use(body, slice_local, dest_json, k)?,
                                )]
                            }
                        };
                        cx.record_term(bb, *target, stmts);
                        cx.getter_bbs.insert(bb);
                        for local in carriers {
                            cx.capture_temps.insert(local);
                        }
                        cx.reach(bb, *target, depth, lo, open)?;
                        continue;
                    }
                    // Observes at Known(0) reads the caller's slots.
                    // Inside our Open the callee reads our pins, which
                    // become slot locals (`eval_slice_index` / `getindex_w`).
                    if call_is_anchor_identity(call, llbc) {
                        cx.reach(bb, *target, depth, lo, open)?;
                        continue;
                    }
                    let effect =
                        resolved_callee_effect(call, llbc, &|id| callee_effect_of(llbc, id));
                    if effect.observes && depth == 0 {
                        if crate::decline::enabled() {
                            eprintln!(
                                "[erase] observes-at-entry {} as {}",
                                callee_path(call, llbc).as_deref().unwrap_or(""),
                                effect.as_str()
                            );
                        }
                        return Err("calls-stack-sensitive-fn");
                    }
                    cx.reach(bb, *target, depth, lo, open)?;
                    continue;
                };
                let args_json = &term_json["Call"]["call"]["args"];
                let dest_json = &term_json["Call"]["dest"];
                let dest = place_local(&call.dest);
                let arg0 = usize::from(method);
                let guard = if method {
                    let receiver = call.args.first().and_then(operand_local);
                    match receiver.and_then(|l| cx.specials.get(&l).map(|s| (l, s))) {
                        Some((_, Special::Alias(g))) => Some(*g),
                        Some((l, Special::Guard { .. })) => Some(l),
                        _ => return Err("receiver-not-a-guard-borrow"),
                    }
                } else {
                    None
                };
                let mut stmts = Vec::new();
                match leaf {
                    Leaf::Open => {
                        let dest = dest.ok_or("guard-dest-projected")?;
                        cx.bind(
                            dest,
                            Special::Guard {
                                depth,
                                skew: depth - lo,
                            },
                        )?;
                        open += 1;
                    }
                    Leaf::GuardNew => {
                        let dest = dest.ok_or("guard-dest-projected")?;
                        cx.bind(
                            dest,
                            Special::Guard {
                                depth,
                                skew: depth - lo,
                            },
                        )?;
                        let obj = args_json.get(0).ok_or("pin-value-untyped")?;
                        let ty = operand_json_ty(obj).ok_or("pin-value-untyped")?;
                        cx.ensure_slot_ty(ty.clone());
                        cx.record_slot_ty(depth, &ty)?;
                        if cx.emit() {
                            stmts.push(assign_json(
                                local_place_json(cx.slot_local(depth), &ty),
                                json!({ "Use": [copy_operand_json(obj), "Yes"] }),
                            ));
                        }
                        let extra = match call.args.get(1).and_then(operand_local) {
                            Some(slice) => {
                                let (agg_si, operands) =
                                    resolve_published_array(block, Some(slice))
                                        .ok_or("publish-array-unresolved")?;
                                let mut inserted = Vec::new();
                                for (i, element) in operands.iter().enumerate() {
                                    let ty = operand_json_ty(element)
                                        .ok_or("publish-element-untyped")?;
                                    cx.ensure_slot_ty(ty.clone());
                                    cx.record_slot_ty(depth + 1 + i, &ty)?;
                                    inserted.push(assign_json(
                                        local_place_json(cx.slot_local(depth + 1 + i), &ty),
                                        json!({ "Use": [copy_operand_json(element), "Yes"] }),
                                    ));
                                }
                                cx.insert_after((bb, agg_si), inserted);
                                operands.len()
                            }
                            None => 0,
                        };
                        pin_depth(&mut depth, &mut lo, 1 + extra);
                        cx.bump_slot_count(depth);
                    }
                    Leaf::Close => {
                        let (d, skew) = if let Some(g) = guard {
                            cx.guard_open(g)?
                        } else {
                            call.args
                                .first()
                                .and_then(operand_local)
                                .and_then(|l| special_open_guard(&cx.specials, l))
                                .ok_or("close-without-guard")?
                        };
                        if d > depth {
                            return Err("close-above-depth");
                        }
                        depth = d;
                        lo = d - skew;
                        open = open.saturating_sub(1);
                    }
                    Leaf::Base => {
                        let (d, _) = cx.guard_open(guard.ok_or("base-without-guard")?)?;
                        cx.bind(
                            dest.ok_or("index-dest-projected")?,
                            Special::Index {
                                k: d,
                                relative: false,
                            },
                        )?;
                    }
                    Leaf::Len => {
                        cx.bind(
                            dest.ok_or("index-dest-projected")?,
                            Special::Index {
                                k: depth,
                                relative: true,
                            },
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
                        cx.ensure_slot_ty(ty.clone());
                        cx.record_slot_ty(depth, &ty)?;
                        if cx.emit() {
                            let copied = copy_operand_json(value);
                            stmts.push(assign_json(
                                local_place_json(cx.slot_local(depth), &ty),
                                json!({ "Use": [copied, "Yes"] }),
                            ));
                            stmts.push(assign_json(
                                dest_json.clone(),
                                json!({ "Use": [copied, "Yes"] }),
                            ));
                        }
                        pin_depth(&mut depth, &mut lo, 1);
                        cx.bump_slot_count(depth);
                    }
                    Leaf::Publish => {
                        let slice = call.args.get(arg0).and_then(operand_local);
                        // Same-block array first; then an aggregate in a
                        // dominating predecessor (`pin_roots_slice_values`
                        // shape); then an incoming `&[PyObjectRef]`
                        // (`pin_roots_incoming_slice`). `gctransform`
                        // `push_roots` runs after `make_jitcodes`.
                        if let Some((agg_bb, agg_si, operands)) =
                            resolve_published_array(block, slice)
                                .map(|(si, ops)| (bb, si, ops))
                                .or_else(|| {
                                    resolve_published_array_in_body(
                                        body,
                                        slice,
                                        &cx.stmt_assigned,
                                        cx.forward_blocks,
                                    )
                                })
                        {
                            let mut inserted = Vec::new();
                            for (i, element) in operands.iter().enumerate() {
                                let ty = element
                                    .as_object()
                                    .and_then(|m| m.values().next())
                                    .and_then(|p| p.get("ty"))
                                    .cloned()
                                    .ok_or("publish-element-untyped")?;
                                cx.ensure_slot_ty(ty.clone());
                                cx.record_slot_ty(depth + i, &ty)?;
                                inserted.push(assign_json(
                                    local_place_json(cx.slot_local(depth + i), &ty),
                                    json!({ "Use": [copy_operand_json(element), "Yes"] }),
                                ));
                            }
                            cx.insert_after((agg_bb, agg_si), inserted);
                            if let Some(dest) = dest {
                                cx.bind(
                                    dest,
                                    Special::Index {
                                        k: depth,
                                        relative: false,
                                    },
                                )?;
                            }
                            pin_depth(&mut depth, &mut lo, operands.len());
                            cx.bump_slot_count(depth);
                        } else if let Some((slice_local, _)) = slice.and_then(|s| {
                            super::pin_roots_incoming_slice(
                                body,
                                s,
                                &cx.stmt_assigned,
                                cx.forward_blocks,
                            )
                        }) {
                            if let Some(dest) = dest {
                                cx.bind(dest, Special::SliceBase { slice_local })?;
                            }
                        } else {
                            return Err("publish-array-unresolved");
                        }
                    }
                    Leaf::Normalize => {}
                    Leaf::NormalizeMoved => {
                        if cx.emit() {
                            let ty = dest_json["ty"].clone();
                            stmts.push(assign_json(
                                dest_json.clone(),
                                json!({ "Use": [{ "Const": [{ "Bool": false }, ty] }, "Yes"] }),
                            ));
                        }
                    }
                    Leaf::Get => {
                        let index_local = call.args.get(arg0).and_then(operand_local);
                        let const_k = call.args.get(arg0).and_then(|op| const_usize(op, llbc));
                        match index_local.and_then(|l| cx.specials.get(&l).copied()) {
                            Some(
                                Special::Index { k, relative } | Special::IndexPair { k, relative },
                            ) => {
                                cx.check_slot_read(k, relative, depth, lo, false)?;
                                if cx.emit() {
                                    let ty = cx.slot_ty_at(k).ok_or("get-before-any-pin")?;
                                    stmts.push(assign_json(
                                        dest_json.clone(),
                                        json!({ "Use": [{ "Read": local_place_json(cx.slot_local(k), &ty) }, "Yes"] }),
                                    ));
                                }
                            }
                            Some(Special::SliceBase { slice_local }) => {
                                if cx.emit() && open == 0 {
                                    return Err("get-above-depth");
                                }
                                if cx.emit() {
                                    stmts.push(assign_json(
                                        dest_json.clone(),
                                        slice_elem_use(body, slice_local, dest_json, 0)?,
                                    ));
                                }
                            }
                            Some(
                                Special::SliceIndex { slice_local, k }
                                | Special::SliceIndexPair { slice_local, k },
                            ) => {
                                if cx.emit() && open == 0 {
                                    return Err("get-above-depth");
                                }
                                if cx.emit() {
                                    stmts.push(assign_json(
                                        dest_json.clone(),
                                        slice_elem_use(body, slice_local, dest_json, k)?,
                                    ));
                                }
                            }
                            _ if const_k.is_some() => {
                                let k = const_k.unwrap();
                                cx.check_slot_read(k, false, depth, lo, false)?;
                                if cx.emit() {
                                    let ty = cx.slot_ty_at(k).ok_or("get-before-any-pin")?;
                                    stmts.push(assign_json(
                                        dest_json.clone(),
                                        json!({ "Use": [{ "Read": local_place_json(cx.slot_local(k), &ty) }, "Yes"] }),
                                    ));
                                }
                            }
                            _ => return Err("index-not-static"),
                        }
                    }
                    Leaf::GuardRoot => {
                        let (d, _) = cx.guard_open(guard.ok_or("receiver-not-a-guard-borrow")?)?;
                        let k = call
                            .args
                            .get(1)
                            .and_then(|op| {
                                const_usize(op, llbc).or_else(|| {
                                    operand_local(op).and_then(|l| cx.const_locals.get(&l).copied())
                                })
                            })
                            .ok_or("index-not-static")?;
                        let slot = d.checked_add(k).ok_or("get-above-depth")?;
                        cx.check_slot_read(slot, false, depth, lo, false)?;
                        if cx.emit() {
                            let ty = cx.slot_ty_at(slot).ok_or("get-before-any-pin")?;
                            stmts.push(assign_json(
                                dest_json.clone(),
                                json!({ "Use": [{ "Read": local_place_json(cx.slot_local(slot), &ty) }, "Yes"] }),
                            ));
                        }
                    }
                    Leaf::Set => {
                        let (k, relative) = cx.slot_index_of(call, arg0, llbc)?;
                        cx.check_slot_read(k, relative, depth, lo, true)?;
                        if cx.emit() {
                            let ty = cx.slot_ty_at(k).ok_or("set-before-any-pin")?;
                            stmts.push(assign_json(
                                local_place_json(cx.slot_local(k), &ty),
                                json!({ "Use": [copy_operand_json(&args_json[arg0 + 1]), "Yes"] }),
                            ));
                        }
                    }
                    Leaf::ReloadTop => {
                        if depth == 0 {
                            return Err("reload-empty-stack");
                        }
                        if cx.emit() {
                            let ty = cx.slot_ty_at(depth - 1).ok_or("reload-before-any-pin")?;
                            stmts.push(assign_json(
                                dest_json.clone(),
                                json!({ "Use": [{ "Read": local_place_json(cx.slot_local(depth - 1), &ty) }, "Yes"] }),
                            ));
                        }
                    }
                    Leaf::Unmodeled
                    | Leaf::CopyRange
                    | Leaf::ItemsOpen
                    | Leaf::ItemsPush
                    | Leaf::ItemsNop
                    | Leaf::ItemsGet
                    | Leaf::ItemsClose => unreachable!("refused above"),
                }
                cx.record_term(bb, *target, stmts);
                cx.reach(bb, *target, depth, lo, open)?;
            }
            Ok(TermKind::Drop { place, target, .. }) => {
                let dropped = place_local(place);
                if let Some(Special::Guard { depth: d, skew }) =
                    dropped.and_then(|l| cx.specials.get(&l)).copied()
                {
                    // A drop flag the artefact does not carry can leave the
                    // guard unopened on this path; the depth says so.
                    if d > depth {
                        return Err("close-above-depth");
                    }
                    depth = d;
                    lo = d - skew;
                    open = open.saturating_sub(1);
                    cx.record_term(bb, *target, Vec::new());
                    cx.reach(bb, *target, depth, lo, open)?;
                } else if dropped.is_some_and(|l| is_root_scope_local(body, llbc, l)) {
                    return Err("drop-of-unopened-guard");
                } else {
                    cx.reach(bb, *target, depth, lo, open)?;
                }
            }
            Ok(TermKind::Assert { assert, target, .. }) => {
                if assert_is_slot_index_bounds_check(assert, &cx.specials, &cx.const_locals, llbc) {
                    cx.record_term(bb, *target, Vec::new());
                }
                cx.reach(bb, *target, depth, lo, open)?;
            }
            Ok(TermKind::Goto { target }) => cx.reach(bb, *target, depth, lo, open)?,
            Ok(TermKind::Switch { targets, .. }) => match targets {
                majit_charon_reader::ullbc::SwitchTargets::If(a, b) => {
                    cx.reach(bb, *a, depth, lo, open)?;
                    cx.reach(bb, *b, depth, lo, open)?;
                }
                majit_charon_reader::ullbc::SwitchTargets::SwitchInt(_, arms, default) => {
                    for (_, t) in arms {
                        cx.reach(bb, *t, depth, lo, open)?;
                    }
                    cx.reach(bb, *default, depth, lo, open)?;
                }
            },
            Ok(TermKind::Return) => {
                if depth != 0 {
                    // 10.04 `Drop` of the guard lives in an `is_cleanup`
                    // block. `FunDecl::unstructured` keeps those blocks, so
                    // `UnwindResume` is among them and a later
                    // `UnwindTerminate` can be last. Return is then the
                    // normal closer.
                    if !body
                        .body
                        .iter()
                        .any(|bb| matches!(bb.term_ref(llbc), Ok(TermKind::UnwindResume)))
                    {
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

    Ok(())
}

/// Build the rewrite, or say why this body keeps its bracket.
fn analyze(body: &Unstructured, llbc: &Llbc) -> Result<Option<Plan>, Refusal> {
    let n_blocks = body.body.len();
    let n_locals = body.locals.locals.len();
    let mut classified: HashMap<usize, (Leaf, bool)> = HashMap::new();
    let forward_blocks = super::forward_reachable_mask(llbc, body);
    for (bb, block) in body.body.iter().enumerate() {
        if !forward_blocks[bb] {
            continue;
        }
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
    for (bb, block) in body.body.iter().enumerate() {
        if !forward_blocks[bb] {
            continue;
        }
        if let Ok(TermKind::Call { call, .. }) = block.term_ref(llbc)
            && classify_call(call, llbc).is_none()
            && !call_is_anchor_identity(call, llbc)
        {
            let effect = resolved_callee_effect(call, llbc, &|id| callee_effect_of(llbc, id));
            if effect.refuses_erase() {
                if crate::decline::enabled() {
                    eprintln!(
                        "[erase] refuses_erase {} as {}",
                        callee_path(call, llbc).as_deref().unwrap_or(""),
                        effect.as_str()
                    );
                }
                return Err("calls-stack-sensitive-fn");
            }
        }
    }
    if classified
        .values()
        .any(|(leaf, _)| leaf_unmodeled_for_erase(*leaf))
    {
        return Err("unmodeled-stack-op");
    }

    let def_counts = local_def_counts(body, llbc, &forward_blocks);
    let mut stmt_assigned: HashMap<usize, usize> = HashMap::new();
    let mut call_dests = bit_set::BitSet::new();
    for (bb, block) in body.body.iter().enumerate() {
        if !forward_blocks[bb] {
            continue;
        }
        for stmt in &block.statements {
            if let Ok(StmtKind::Assign(place, _)) = stmt.stmt_kind_ref()
                && let Some(dest) = place_local(place)
            {
                *stmt_assigned.entry(dest).or_default() += 1;
            }
        }
        if let Ok(TermKind::Call { call, .. }) = block.term_ref(llbc)
            && let Some(dest) = place_local(&call.dest)
        {
            call_dests.insert(dest);
        }
    }
    let mut const_locals: HashMap<usize, usize> = HashMap::new();
    let mut array_lens: HashMap<usize, usize> = HashMap::new();
    let mut getter_cache: HashMap<u64, Option<SlotGetter>> = HashMap::new();
    if n_blocks == 0 {
        return Ok(None);
    }
    let succs: Vec<Vec<u64>> = (0..n_blocks)
        .map(|bb| erase_forward_successors(body, llbc, bb))
        .collect();
    let mut depth_in: Vec<Option<usize>> = vec![None; n_blocks];
    let mut lo_in: Vec<Option<usize>> = vec![None; n_blocks];
    // Guards currently open. Incoming-slice Gets are allowed only while
    // this is non-zero: Open does not change scalar depth, so Close of a
    // slice-only bracket is still depth 0.
    let mut open_in: Vec<Option<usize>> = vec![None; n_blocks];
    depth_in[0] = Some(0);
    lo_in[0] = Some(0);
    open_in[0] = Some(0);

    let mut probe_specials = HashMap::new();
    let mut queue: VecDeque<usize> = VecDeque::from([0]);
    let mut visited = vec![false; n_blocks];
    let mut dummy_getter_bbs = bit_set::BitSet::new();
    let mut dummy_capture = bit_set::BitSet::new();
    {
        let mut cx = EraseWalk {
            mode: WalkMode::Probe,
            body,
            llbc,
            classified: &classified,
            forward_blocks: &forward_blocks,
            succs: &succs,
            n_blocks,
            n_locals,
            def_counts: &def_counts,
            stmt_assigned: &stmt_assigned,
            call_dests: &call_dests,
            depth_in: &mut depth_in,
            lo_in: &mut lo_in,
            open_in: &mut open_in,
            frozen_depth: &[],
            frozen_lo: &[],
            queue: &mut queue,
            visited: &mut visited,
            specials: &mut probe_specials,
            plan: None,
            const_locals: &mut const_locals,
            array_lens: &mut array_lens,
            getter_cache: &mut getter_cache,
            getter_bbs: &mut dummy_getter_bbs,
            capture_temps: &mut dummy_capture,
        };
        run_erase_walk(&mut cx)?;
    }

    let frozen_depth = depth_in.clone();
    let frozen_lo = lo_in.clone();
    visited.fill(false);
    queue.clear();
    queue.push_back(0);
    let mut plan = Plan {
        specials: HashMap::new(),
        removed: HashMap::new(),
        inserted_after: HashMap::new(),
        terms: HashMap::new(),
        slot_count: 0,
        slot_ty: None,
        slot_tys: HashMap::new(),
        unreachable: Vec::new(),
    };
    let mut emit_specials = HashMap::new();
    let mut getter_bbs = bit_set::BitSet::new();
    let mut capture_temps = bit_set::BitSet::new();
    {
        let mut cx = EraseWalk {
            mode: WalkMode::Emit,
            body,
            llbc,
            classified: &classified,
            forward_blocks: &forward_blocks,
            succs: &succs,
            n_blocks,
            n_locals,
            def_counts: &def_counts,
            stmt_assigned: &stmt_assigned,
            call_dests: &call_dests,
            depth_in: &mut depth_in,
            lo_in: &mut lo_in,
            open_in: &mut open_in,
            frozen_depth: &frozen_depth,
            frozen_lo: &frozen_lo,
            queue: &mut queue,
            visited: &mut visited,
            specials: &mut emit_specials,
            plan: Some(&mut plan),
            const_locals: &mut const_locals,
            array_lens: &mut array_lens,
            getter_cache: &mut getter_cache,
            getter_bbs: &mut getter_bbs,
            capture_temps: &mut capture_temps,
        };
        run_erase_walk(&mut cx)?;
    }
    plan.specials = emit_specials;

    // Local 0 is the return place. An Index bound there is a helper that
    // answers a slot (`pin_self`: pin, then `shadow_stack_len() - 1`).
    // The Sub/Len statements that defined it are stripped as specials, and
    // `Return` does not mention local 0, so the rewritten body would return
    // an unbound local. Callers keep that index (`from_slot`) and lower the
    // call as `getattr(recv, method)` (`CallTarget::Method`); a broken
    // callee graph blocks that getattr. Refuse: the Pin lives in the
    // caller's RootScope and is only erasable if the helper is inlined.
    // A Guard bound there is an opener whose result is the bracket itself
    // (a trait method that returns `RootScope`): the erase would strip the
    // write of local 0 the same way. The caller keeps that guard and the
    // Open is modeled at its call site
    // (`shadowstack.py ShadowStackFrameworkGCTransformer.push_roots`).
    if matches!(
        plan.specials.get(&0),
        Some(Special::Index { .. } | Special::IndexPair { .. } | Special::Guard { .. })
    ) {
        return Err("returns-slot-index");
    }

    // Temporaries that only build a getter capture (borrow of the index, the
    // closure aggregate, the receiver borrow, the argument tuple) bind
    // nothing once the call is a slot copy.  Another use — passing the
    // closure, storing it — would still run the real `shadow_stack_get`.
    if !capture_temps.is_empty() {
        if !super::capture_carriers_stay_inside(body, &capture_temps, &getter_bbs, &forward_blocks)
        {
            return Err("capture-escapes");
        }
        for (bb, block) in body.body.iter().enumerate() {
            if !forward_blocks[bb] || !visited[bb] {
                continue;
            }
            for (si, stmt) in block.statements.iter().enumerate() {
                if let Ok(StmtKind::Assign(place, _)) = stmt.stmt_kind_ref()
                    && place_local(place).is_some_and(|dest| capture_temps.contains(dest))
                {
                    plan.removed.entry(bb).or_default().push(si);
                }
            }
        }
    }

    // Every surviving mention of a guard, borrow or index has to be one of the
    // shapes rewritten above; anything else would read a local no block binds.
    let mut watched = bit_set::BitSet::new();
    for local in plan.specials.keys() {
        watched.insert(*local);
    }
    for (bb, block) in body.body.iter().enumerate() {
        if !forward_blocks[bb] {
            continue;
        }
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

/// Drop of the guard / `root_scope_close` rewinds to the open depth, so
/// predecessors may hold different scalar depths. Slots are locals; the
/// close does not read them (`__majit_wrap_dict_descr_keys` early return
/// vs `pin_root`).
fn block_rewinds_scope(
    body: &Unstructured,
    llbc: &Llbc,
    classified: &HashMap<usize, (Leaf, bool)>,
    specials: &HashMap<usize, Special>,
    bb: usize,
) -> bool {
    if matches!(classified.get(&bb), Some((Leaf::Close, _))) {
        return true;
    }
    match body.body.get(bb).and_then(|b| b.term_ref(llbc).ok()) {
        Some(TermKind::Drop { place, .. }) => {
            matches!(
                place_local(place).and_then(|l| specials.get(&l)),
                Some(Special::Guard { .. })
            ) || place_local(place).is_some_and(|l| is_root_scope_local(body, llbc, l))
        }
        Some(TermKind::Call { call, .. }) => mem_drop_guard_depth(call, llbc, specials).is_some(),
        _ => false,
    }
}

/// Forward successors the erase walk follows: `on_unwind` is not a
/// successor (`analyze` does not walk cleanup).
fn erase_forward_successors(body: &Unstructured, llbc: &Llbc, bb: usize) -> Vec<u64> {
    let Some(block) = body.body.get(bb) else {
        return Vec::new();
    };
    match block.term_ref(llbc) {
        Ok(TermKind::Call { target, .. })
        | Ok(TermKind::Drop { target, .. })
        | Ok(TermKind::Assert { target, .. })
        | Ok(TermKind::Goto { target }) => vec![*target],
        Ok(TermKind::Switch { targets, .. }) => match targets {
            majit_charon_reader::ullbc::SwitchTargets::If(a, b) => vec![*a, *b],
            majit_charon_reader::ullbc::SwitchTargets::SwitchInt(_, arms, default) => {
                let mut s: Vec<u64> = arms.iter().map(|(_, t)| *t).collect();
                s.push(*default);
                s
            }
        },
        _ => Vec::new(),
    }
}

fn is_erase_exit(body: &Unstructured, llbc: &Llbc, bb: usize) -> bool {
    matches!(
        body.body.get(bb).and_then(|b| b.term_ref(llbc).ok()),
        Some(
            TermKind::Return
                | TermKind::UnwindResume
                | TermKind::UnwindTerminate
                | TermKind::Abort(_)
                | TermKind::Panic { .. }
                | TermKind::UndefinedBehavior
        )
    )
}

/// A path that returns to the caller with this body's Open still live.
/// UB / abort / panic / unwind-terminate are sinks: they do not hand a
/// livevar set back, so they do not need a rewind for the join to be
/// sound (`get_livevars_for_roots` is around collecting ops on returning
/// paths).
fn is_erase_returning_exit(body: &Unstructured, llbc: &Llbc, bb: usize) -> bool {
    matches!(
        body.body.get(bb).and_then(|b| b.term_ref(llbc).ok()),
        Some(TermKind::Return | TermKind::UnwindResume)
    )
}

fn reachable_avoiding(
    succs: &[Vec<u64>],
    n: usize,
    start: usize,
    goal: usize,
    avoid: usize,
) -> bool {
    if start == avoid {
        return false;
    }
    if start == goal {
        return true;
    }
    let mut seen = vec![false; n];
    let mut queue = VecDeque::new();
    queue.push_back(start);
    seen[start] = true;
    while let Some(bb) = queue.pop_front() {
        for &s in succs.get(bb).into_iter().flatten() {
            let s = s as usize;
            if s >= n || s == avoid || seen[s] {
                continue;
            }
            if s == goal {
                return true;
            }
            seen[s] = true;
            queue.push_back(s);
        }
    }
    false
}

fn dominates(succs: &[Vec<u64>], n: usize, dominator: usize, node: usize) -> bool {
    dominator == node || !reachable_avoiding(succs, n, 0, node, dominator)
}

/// The set of Close/Drop of this body's open guard postdominates `join`
/// on the erase CFG: every *returning* path from `join` goes through some
/// rewind. rustc copies the Drop onto each return, so no single rewind
/// block postdominates; the set does. UB / abort / panic sinks (shared
/// unreachable defaults) do not need a rewind.
fn join_postdominated_by_this_guard_rewind(
    body: &Unstructured,
    llbc: &Llbc,
    classified: &HashMap<usize, (Leaf, bool)>,
    specials: &HashMap<usize, Special>,
    succs: &[Vec<u64>],
    n: usize,
    join: usize,
) -> bool {
    let mut seen = vec![false; n];
    let mut queue = VecDeque::new();
    queue.push_back(join);
    seen[join] = true;
    while let Some(bb) = queue.pop_front() {
        if block_rewinds_scope(body, llbc, classified, specials, bb) {
            continue;
        }
        if is_erase_returning_exit(body, llbc, bb) {
            return false;
        }
        if is_erase_exit(body, llbc, bb) {
            continue;
        }
        for &s in succs.get(bb).into_iter().flatten() {
            let s = s as usize;
            if s >= n || seen[s] {
                continue;
            }
            seen[s] = true;
            queue.push_back(s);
        }
    }
    true
}

/// `(*slice)[k]` as a `Use` rvalue. Lowering's `pair_getitem` /
/// `guard_item_index` is the bounds check rustc would emit for that
/// Index; a separate Assert terminator would need extra blocks.
fn slice_elem_use(
    body: &Unstructured,
    slice_local: usize,
    dest_place: &Value,
    k: usize,
) -> Result<Value, Refusal> {
    let slice_ty = body
        .locals
        .locals
        .get(slice_local)
        .map(|loc| ty_ref_json(&loc.ty))
        .ok_or("slice-untyped")?;
    let elem_ty = dest_place.get("ty").cloned().ok_or("get-dest-untyped")?;
    let local_place = json!({ "kind": { "Local": slice_local }, "ty": slice_ty });
    let deref = json!({ "kind": { "Projection": [local_place, "Deref"] }, "ty": slice_ty });
    let offset = json!({
        "Const": [
            { "Integer": { "Unsigned": ["Usize", k.to_string()] } },
            { "Deduplicated": 0 }
        ]
    });
    let index_place = json!({
        "kind": {
            "Projection": [deref, { "Index": { "offset": offset, "from_end": false } }]
        },
        "ty": elem_ty
    });
    Ok(json!({ "Use": [{ "Read": index_place }, "Yes"] }))
}

/// Unique forward `Assign` of `local`, with its block and statement index.
fn unique_assign_stmt<'a>(
    body: &'a Unstructured,
    local: usize,
    assigned: &HashMap<usize, usize>,
    forward_blocks: &[bool],
) -> Option<(usize, usize, &'a majit_charon_reader::ullbc::Statement)> {
    if assigned.get(&local) != Some(&1) {
        return None;
    }
    for (bb, block) in body.body.iter().enumerate() {
        if !forward_blocks.get(bb).copied().unwrap_or(false) {
            continue;
        }
        for (si, stmt) in block.statements.iter().enumerate() {
            if let Ok(StmtKind::Assign(place, _)) = stmt.stmt_kind_ref()
                && place_local(place) == Some(local)
            {
                return Some((bb, si, stmt));
            }
        }
    }
    None
}

/// Same walk as [`resolve_published_array`], across forward-reachable
/// single-def locals, so an aggregate built in a predecessor is resolved.
fn resolve_published_array_in_body(
    body: &Unstructured,
    slice: Option<usize>,
    assigned: &HashMap<usize, usize>,
    forward_blocks: &[bool],
) -> Option<(usize, usize, Vec<Value>)> {
    let mut local = slice?;
    for _ in 0..8 {
        let assigns = assigned.get(&local).copied().unwrap_or(0);
        if assigns != 1 {
            return None;
        }
        let (bb, si, stmt) = unique_assign_stmt(body, local, assigned, forward_blocks)?;
        let Ok(StmtKind::Assign(_, rvalue)) = stmt.stmt_kind_ref() else {
            return None;
        };
        match rvalue {
            Rvalue::UnaryOp(_, op) | Rvalue::Use(op, _) | Rvalue::Cast(_, op, _) => {
                local = operand_local(op)?;
            }
            Rvalue::Ref { place: src, .. } | Rvalue::RawPtr { place: src, .. } => {
                local = place_local(src).or_else(|| deref_of_local(src))?;
            }
            Rvalue::Aggregate(kind, _) if kind.get("Array").is_some() => {
                let operands = stmt.kind_value()["Assign"][1]["Aggregate"][1]
                    .as_array()?
                    .clone();
                return Some((bb, si, operands));
            }
            _ => return None,
        }
    }
    None
}

/// Follow a published slice back to the array literal it borrows, in the same
/// block: `_a = [x, y]; _r = &_a; _s = &*_r; _p = _s as &[_]`.  Returns the
/// literal's statement index and its element operands.
///
/// The index is this block's own statement index. Cleanup blocks stay in the
/// body at their extracted indices and are not renumbered out of the way.
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
    if fd.item_meta.name_path().contains("::gc_roots::") || super::frame_anchor::is_helper(fd, llbc)
    {
        return Ok(None);
    }
    // `body` keeps every extracted block at its original index (promoted
    // constants already spliced). The walk records only blocks reached from
    // bb0; `on_unwind` is not a successor, so a cleanup block stays in the
    // body and is not entered. Rewriting against any other numbering is how
    // a same-block array publish was refused as unresolved.
    //
    // The JSON rebuild parses every terminator kind. A body with no bracket
    // never reads that tree (`analyze` returns `None`), so build it only
    // once a plan exists.
    let plan = match analyze(body, llbc)? {
        Some(plan) => plan,
        None => return Ok(None),
    };
    let mut u = unstructured_json_from_body(body);
    let span = u["span"].clone();
    // The slot locals.
    if plan.slot_count > 0 {
        let locals = u["locals"]["locals"].as_array_mut().ok_or("body-json")?;
        let base = locals.len();
        for k in 0..plan.slot_count {
            let ty = plan
                .slot_tys
                .get(&k)
                .cloned()
                .or_else(|| plan.slot_ty.clone())
                .ok_or("slot-untyped")?;
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
                "kind": match bb.kind {
                    UnwindKind::Regular => "Regular",
                    UnwindKind::Cleanup => "Cleanup",
                    UnwindKind::Terminate => "Terminate",
                },
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
/// `callee_effect_of` must not answer `None` when the field tables
/// hold a non-`None` product (depth-neutral overlay vs tables).
fn debug_check_registered_effects_match_rewalk(llbc: &Llbc) {
    let collect_all = std::env::var_os("PYRE_SS_COLLECT_ALL").is_some();
    let effect = |identity: &str| callee_effect_of(llbc, identity);
    let mut mismatches = 0usize;
    let mut overlay_hides = 0usize;
    for fd in llbc.iter_local_fns() {
        let Some(body) = fd.unstructured() else {
            continue;
        };
        let identity = super::scope_owner_identity(llbc, fd);
        let registered = effect(&identity);
        let walked = stack_walk(&body, llbc, &effect).effect();
        let assembled = assembled_effect_from_tables(llbc, &identity);
        if !assembled.is_none() && registered.is_none() {
            overlay_hides += 1;
            let name = fd.item_meta.name_path();
            let msg = format!(
                "stack-effect overlay: {name} identity={identity} tables {} callee_effect_of None",
                assembled.describe()
            );
            if collect_all {
                eprintln!("[table-overlay-disagree] {msg}");
            } else {
                panic!("{msg}");
            }
        }
        if registered != walked {
            mismatches += 1;
            let name = fd.item_meta.name_path();
            let why = stack_walk(&body, llbc, &effect)
                .why
                .as_deref()
                .unwrap_or("-")
                .to_string();
            let msg = format!(
                "stack-effect invariant: {name} identity={identity} registered {} rewalk {} why={why}",
                registered.describe(),
                walked.describe(),
            );
            if collect_all {
                eprintln!("[mismatch] {msg}");
            } else {
                panic!("{msg}");
            }
        }
    }
    if collect_all {
        eprintln!(
            "[mismatch-summary] crate={} mismatches={} overlay_hides={}",
            llbc.crate_name(),
            mismatches,
            overlay_hides
        );
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
            let walk = stack_walk(&body, llbc, &|identity| callee_effect_of(llbc, identity));
            let callee = first_non_none_callee(&body, llbc);
            eprintln!(
                "[census-detail] {name}:{bucket} why={} callee={} effect={}",
                walk.why.as_deref().unwrap_or("-"),
                callee.as_ref().map(|(p, _)| p.as_str()).unwrap_or("-"),
                callee
                    .as_ref()
                    .map(|(_, effect)| effect.describe())
                    .unwrap_or_else(|| "-".to_string()),
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

    #[cfg(test)]
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

    /// `GraphAnalyzer.bottom_result`: most optimistic answer.
    /// `returns_index` is a must-property (`true ≼ false`), so bottom
    /// still claims it. `RaiseAnalyzer.bottom_result` is `False`
    /// (cannot raise) for the same reason: unseen / in-progress graphs
    /// are optimistic until `Dependency.merge_with_result` / `leave_with`.
    fn bottom() -> Self {
        Self {
            observes: false,
            leaves_above: false,
            param_slots: Vec::new(),
            returns_index: true,
        }
    }

    /// Pointwise join: `observes` / `leaves_above` OR, `param_slots`
    /// union (sorted, dedup), `returns_index` AND.
    /// `BoolGraphAnalyzer.join_two_results` is OR; the must-field uses
    /// the dual. `WriteAnalyzer.join_two_results` is set union.
    fn join_two_results(&self, other: &Self) -> Self {
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
            returns_index: self.returns_index && other.returns_index,
        }
    }

    /// Product order: `self ≼ other` iff other is at least as severe
    /// on every field (`false ≼ true` OR-fields, `⊆` on `param_slots`,
    /// `true ≼ false` on `returns_index`).
    fn leq(&self, other: &Self) -> bool {
        (!self.observes || other.observes)
            && (!self.leaves_above || other.leaves_above)
            && self
                .param_slots
                .iter()
                .all(|p| other.param_slots.contains(p))
            && (!other.returns_index || self.returns_index)
    }

    fn describe(&self) -> String {
        format!(
            "{}(observes={} leaves_above={} param_slots={:?} returns_index={})",
            self.as_str(),
            self.observes,
            self.leaves_above,
            self.param_slots,
            self.returns_index
        )
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
}

/// The origportal is the configured jitdriver's `portal` graph that
/// `rewrite_jit_merge_point` turns into `return portal_runner(*args)`
/// (`warmspot.py rewrite_jit_merge_point`).
fn resolved_callee_is_origportal_alias(llbc: &Llbc, path: &str) -> bool {
    llbc.is_rewritten_origportal_fn(&crate::front::mir::strip_crate_prefix(path))
}

/// Origportal is a property of the FunDecl, matched by the configured
/// jitdriver path (`rewritten_origportal_fns`).
fn is_origportal_alias_fd(llbc: &Llbc, fd: &FunDecl) -> bool {
    resolved_callee_is_origportal_alias(llbc, &fd.item_meta.name_path())
}

/// Registered effect of `identity`: `None` if not sensitive or proven
/// neutral; otherwise the union of the sets it was registered in. A
/// sensitive identity in none of them comes from an artefact that did
/// not classify it, and is taken at its worst. `identity` is
/// [`super::scope_owner_identity`], matching
/// `GraphAnalyzer.analyze_direct_call` / `DependencyTracker` keyed by
/// graph, never by `ItemMeta::name_path`. Origportal aliases are
/// answered as none by [`resolved_callee_effect`] / [`is_origportal_alias_fd`],
/// not here: identities are not config paths.
fn assembled_effect_from_tables(llbc: &Llbc, identity: &str) -> CalleeEffect {
    CalleeEffect {
        observes: llbc.is_stack_observes_fn(identity),
        leaves_above: llbc.is_stack_leaves_above_fn(identity),
        param_slots: llbc.stack_param_slots(identity).unwrap_or_default(),
        returns_index: llbc.is_stack_returns_index_fn(identity),
    }
}

fn callee_effect_of(llbc: &Llbc, identity: &str) -> CalleeEffect {
    let assembled = assembled_effect_from_tables(llbc, identity);
    if llbc.is_stack_depth_neutral_fn(identity) {
        return CalleeEffect::none();
    }
    if assembled.is_none() && llbc.is_stack_sensitive_fn(identity) {
        return CalleeEffect::worst();
    }
    assembled
}

fn first_non_none_callee(body: &Unstructured, llbc: &Llbc) -> Option<(String, CalleeEffect)> {
    let forward_blocks = super::forward_reachable_mask(llbc, body);
    for (bb_idx, block) in body.body.iter().enumerate() {
        if !forward_blocks[bb_idx] {
            continue;
        }
        if let Ok(TermKind::Call { call, .. }) = block.term_ref(llbc)
            && classify_call(call, llbc).is_none()
            && !call_is_anchor_identity(call, llbc)
            && let Some(identity) = callee_identity(call, llbc)
        {
            let effect = resolved_callee_effect(call, llbc, &|id| callee_effect_of(llbc, id));
            if !effect.is_none() {
                return Some((identity, effect));
            }
        }
    }
    None
}

fn callee_fun_decl<'a>(
    call: &majit_charon_reader::ullbc::CallPayload,
    llbc: &'a Llbc,
) -> Option<&'a FunDecl> {
    let majit_charon_reader::ullbc::CallFunc::Regular(reg) = &call.func else {
        return None;
    };
    let id = super::regular_call_fun_decl_id(&reg.kind)?;
    llbc.fn_by_id(id)
}

/// The callee's `ItemMeta::name_path()`, for a statically resolved callee.
/// Name consumers (`mem_drop_scope_local`, observes-at-entry logging)
/// use this. Stack-effect tables key by [`callee_identity`].
fn callee_path(call: &majit_charon_reader::ullbc::CallPayload, llbc: &Llbc) -> Option<String> {
    callee_fun_decl(call, llbc).map(|fd| fd.item_meta.name_path())
}

/// [`super::scope_owner_identity`] of the resolved callee FunDecl.
/// `GraphAnalyzer.analyze_direct_call` / `DependencyTracker` key by graph;
/// this is that graph's existing identity string.
fn callee_identity(call: &majit_charon_reader::ullbc::CallPayload, llbc: &Llbc) -> Option<String> {
    callee_fun_decl(call, llbc).map(|fd| super::scope_owner_identity(llbc, fd))
}

/// Shared Call → effect resolution. An origportal-alias callee is none
/// before identity-keyed tables or `effect` are consulted.
fn resolved_callee_effect(
    call: &majit_charon_reader::ullbc::CallPayload,
    llbc: &Llbc,
    effect: &dyn Fn(&str) -> CalleeEffect,
) -> CalleeEffect {
    let Some(fd) = callee_fun_decl(call, llbc) else {
        return CalleeEffect::none();
    };
    if is_origportal_alias_fd(llbc, fd) {
        return CalleeEffect::none();
    }
    effect(&super::scope_owner_identity(llbc, fd))
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
        // sets both. The four fields are a product: Observes does not
        // clear `param_slots` / `returns_index`.
        let observes = self.reads_below || self.unknown_exit;
        CalleeEffect {
            observes,
            leaves_above: self.leaves_above || self.unknown_exit,
            param_slots: self.param_slots.clone(),
            returns_index: self.returns_index,
        }
    }
}

/// Depth-neutral: the walk's product is `None` and a `RootScope` opens
/// on some reachable path (`effect().is_none() && saw_open`).
#[cfg(test)]
fn body_is_depth_neutral(
    body: &Unstructured,
    llbc: &Llbc,
    effect: &dyn Fn(&str) -> CalleeEffect,
) -> bool {
    let walk = stack_walk(body, llbc, effect);
    walk.effect().is_none() && walk.saw_open
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
                            Leaf::Open | Leaf::ItemsOpen | Leaf::GuardNew => {
                                saw_open = true;
                                if let Some(dest) = place_local(&call.dest) {
                                    let entry = guards.entry(dest).or_insert(depth);
                                    *entry = entry.join(depth);
                                }
                                if matches!(leaf, Leaf::GuardNew) {
                                    depth = depth.add(1);
                                    let slice = call.args.get(1).and_then(operand_local);
                                    match resolve_published_array_len(block, slice) {
                                        Some(n) => depth = depth.add(n),
                                        None if call.args.get(1).is_none() => {}
                                        None => depth = depth.after_unknown_push(),
                                    }
                                }
                            }
                            Leaf::Close | Leaf::ItemsClose => {
                                let g = if matches!(leaf, Leaf::ItemsClose) {
                                    items_guard
                                } else {
                                    guard
                                };
                                if let Some(d) = g.and_then(|g| guards.get(&g).copied()) {
                                    depth = d;
                                } else if let Some(l) = call.args.get(arg0).and_then(operand_local)
                                {
                                    depth = guards
                                        .get(&l)
                                        .copied()
                                        .or_else(|| indices.get(&l).copied())
                                        .or_else(|| {
                                            aliases.get(&l).and_then(|g| guards.get(g).copied())
                                        })
                                        .unwrap_or(Depth::Unknown);
                                } else {
                                    depth = Depth::Unknown;
                                }
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
                            Leaf::GuardRoot => {
                                if guard.and_then(|g| guards.get(&g).copied()).is_none() {
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
                        if let Some(local) = mem_drop_scope_local(call, llbc) {
                            if let Some(d) = guards.get(&local).copied().or_else(|| {
                                aliases.get(&local).and_then(|g| guards.get(g).copied())
                            }) {
                                depth = d;
                            }
                        }
                        if !call_is_anchor_identity(call, llbc) {
                            let callee = resolved_callee_effect(call, llbc, effect);
                            // Four fields independently (`GraphAnalyzer` product
                            // transfer). Observes does not gate param-slots or
                            // `returns_index`. At Known(0) every live slot is
                            // the caller's, so Observes is not neutral. Inside
                            // our own Open (`eval_slice_index` pin then
                            // `getindex_w`) Observes does not set `reads_below`:
                            // Close rewinds our guard.
                            if callee.observes && !depth.above_entry() {
                                reads_below = true;
                                why = format!(
                                    "unproven-stack-sensitive {}",
                                    path.as_deref().unwrap_or("")
                                );
                            }
                            let depth_at_call = depth;
                            if callee.leaves_above {
                                // ObservesLeaves may be an unbounded exit
                                // (below or above); LeavesAbove alone is a
                                // publish of unknown length at this depth.
                                depth = if callee.observes {
                                    Depth::Unknown
                                } else {
                                    depth.after_unknown_push()
                                };
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
/// [`Llbc::is_stack_leaves_above_fn`]), keyed by
/// [`super::scope_owner_identity`].
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

/// Sensitive, leaves-above, param-slots, returns-index, observes, and
/// depth-neutral sets from one fixpoint, so harvest does not recompute
/// the body walks.
pub fn discover_stack_fn_effects(
    llbc: &Llbc,
) -> (
    Vec<String>,
    Vec<String>,
    Vec<(String, Vec<u8>)>,
    Vec<String>,
    Vec<String>,
    Vec<String>,
) {
    discover_stack_effects(llbc)
}

fn local_callee_answer(
    llbc: &Llbc,
    identity: &str,
    local: &HashSet<&str>,
    class: &HashMap<String, CalleeEffect>,
) -> CalleeEffect {
    if local.contains(identity) {
        class
            .get(identity)
            .cloned()
            .unwrap_or_else(CalleeEffect::bottom)
    } else {
        callee_effect_of(llbc, identity)
    }
}

/// Least consistent assignment of per-body [`stack_walk`] over this artefact.
///
/// `GraphAnalyzer.analyze` / `analyze_direct_call`: every local body
/// starts at [`CalleeEffect::bottom`] (`GraphAnalyzer.bottom_result`).
/// A body's answer is `join_two_results(previous, walk)`
/// (`Dependency.merge_with_result`). On a change, requeue that body's
/// callers (`DependencyTracker`). FIFO worklist; no visit cap; no
/// `CalleeEffect::worst()` freeze. Cross-artefact callees keep
/// [`callee_effect_of`]. Termination follows from finite height
/// (`param_slots` bounded by arity). Depth-neutral is the same
/// assignment: `effect().is_none() && saw_open`.
///
/// `class`, `identities[i]`, and `callers` are keyed by
/// [`super::scope_owner_identity`], matching
/// `GraphAnalyzer.analyze_direct_call` / `DependencyTracker` which key
/// by graph object. `ItemMeta::name_path` is a label, not an identity.
fn discover_stack_effects(
    llbc: &Llbc,
) -> (
    Vec<String>,
    Vec<String>,
    Vec<(String, Vec<u8>)>,
    Vec<String>,
    Vec<String>,
    Vec<String>,
) {
    let started = Instant::now();
    let fds: Vec<&FunDecl> = llbc
        .iter_local_fns()
        .filter(|fd| fd.body.is_some())
        .collect();
    let identities: Vec<String> = fds
        .iter()
        .map(|fd| super::scope_owner_identity(llbc, fd))
        .collect();
    let local: HashSet<&str> = identities.iter().map(String::as_str).collect();
    let mut callers: HashMap<String, Vec<usize>> = HashMap::new();
    // Concrete assignment, including resolved None. Missing is only the
    // pre-init state; lookup must not revive `bottom` (`returns_index`)
    // for a body already joined to None.
    let mut class: HashMap<String, CalleeEffect> = HashMap::new();
    for identity in &identities {
        class.insert(identity.clone(), CalleeEffect::bottom());
    }
    let mut last_walk: Vec<Option<(Vec<(String, CalleeEffect)>, CalleeEffect, bool)>> =
        vec![None; fds.len()];
    for (i, fd) in fds.iter().enumerate() {
        let Some(body) = fd.unstructured_shared() else {
            continue;
        };
        let forward_blocks = super::forward_reachable_mask(llbc, &body);
        for (bb_idx, block) in body.body.iter().enumerate() {
            if !forward_blocks[bb_idx] {
                continue;
            }
            if let Ok(TermKind::Call { call, .. }) = block.term_ref(llbc)
                && let Some(identity) = callee_identity(call, llbc)
            {
                callers.entry(identity).or_default().push(i);
            }
        }
    }
    let mut queued = vec![false; fds.len()];
    let mut queue: VecDeque<usize> = VecDeque::new();
    for i in 0..fds.len() {
        queued[i] = true;
        queue.push_back(i);
    }
    let mut pops = 0usize;
    let mut walks = 0usize;
    let mut skips = 0usize;
    let mut max_visits = 0u32;
    let mut visits = vec![0u32; fds.len()];
    while let Some(i) = queue.pop_front() {
        queued[i] = false;
        pops += 1;
        visits[i] = visits[i].saturating_add(1);
        if visits[i] > max_visits {
            max_visits = visits[i];
        }
        let walk_effect = if let Some((queried, prev, _)) = &last_walk[i]
            && queried.iter().all(|(identity, answer)| {
                local_callee_answer(llbc, identity, &local, &class) == *answer
            }) {
            skips += 1;
            prev.clone()
        } else if is_origportal_alias_fd(llbc, fds[i]) {
            CalleeEffect::none()
        } else {
            let Some(body) = fds[i].unstructured_shared() else {
                let none = CalleeEffect::none();
                let old = class
                    .get(&identities[i])
                    .cloned()
                    .unwrap_or_else(CalleeEffect::bottom);
                let joined = old.join_two_results(&none);
                debug_assert!(
                    old.leq(&joined),
                    "stack-effect answer decreased: {} -> {}",
                    old.describe(),
                    joined.describe()
                );
                if joined != old {
                    class.insert(identities[i].clone(), joined);
                    for &caller in callers.get(&identities[i]).into_iter().flatten() {
                        if queued[caller] {
                            continue;
                        }
                        queued[caller] = true;
                        queue.push_back(caller);
                    }
                }
                continue;
            };
            walks += 1;
            let queried = std::cell::RefCell::new(Vec::new());
            let effect = |identity: &str| {
                let answer = local_callee_answer(llbc, identity, &local, &class);
                if local.contains(identity) {
                    queried
                        .borrow_mut()
                        .push((identity.to_string(), answer.clone()));
                }
                answer
            };
            let walk = stack_walk(&body, llbc, &effect);
            let walk_effect = walk.effect();
            last_walk[i] = Some((queried.into_inner(), walk_effect.clone(), walk.saw_open));
            walk_effect
        };
        let old = class
            .get(&identities[i])
            .cloned()
            .unwrap_or_else(CalleeEffect::bottom);
        let joined = old.join_two_results(&walk_effect);
        debug_assert!(
            old.leq(&joined),
            "stack-effect answer decreased: {} -> {}",
            old.describe(),
            joined.describe()
        );
        if joined == old {
            continue;
        }
        class.insert(identities[i].clone(), joined);
        for &caller in callers.get(&identities[i]).into_iter().flatten() {
            if queued[caller] {
                continue;
            }
            queued[caller] = true;
            queue.push_back(caller);
        }
    }
    let mut sensitive: Vec<String> = Vec::new();
    let mut leaves: Vec<String> = Vec::new();
    let mut param_slots: Vec<(String, Vec<u8>)> = Vec::new();
    let mut returns_index: Vec<String> = Vec::new();
    let mut observes: Vec<String> = Vec::new();
    let mut neutrals: Vec<String> = Vec::new();
    for (i, identity) in identities.iter().enumerate() {
        let effect = class
            .get(identity)
            .cloned()
            .unwrap_or_else(CalleeEffect::none);
        let saw_open = last_walk[i]
            .as_ref()
            .map(|(_, _, open)| *open)
            .unwrap_or(false);
        if effect.is_none() {
            if saw_open {
                neutrals.push(identity.clone());
            }
            continue;
        }
        sensitive.push(identity.clone());
        if effect.leaves_above {
            leaves.push(identity.clone());
        }
        if effect.observes {
            observes.push(identity.clone());
        }
        if !effect.param_slots.is_empty() {
            param_slots.push((identity.clone(), effect.param_slots));
        }
        if effect.returns_index {
            returns_index.push(identity.clone());
        }
    }
    sensitive.sort();
    observes.sort();
    leaves.sort();
    param_slots.sort_by(|a, b| a.0.cmp(&b.0));
    returns_index.sort();
    neutrals.sort();
    if std::env::var_os("PYRE_SS_STATS").is_some() {
        eprintln!(
            "[discover-stats] crate={} pops={pops} walks={walks} skips={skips} max_visits={max_visits} neutrals={} ms={} bodies={}",
            llbc.crate_name(),
            neutrals.len(),
            started.elapsed().as_millis(),
            fds.len()
        );
    }
    (
        sensitive,
        leaves,
        param_slots,
        returns_index,
        observes,
        neutrals,
    )
}

/// Local bodies proven depth-neutral under the same assignment
/// [`discover_stack_effects`] publishes: `effect().is_none() && saw_open`.
/// Keys are [`super::scope_owner_identity`] (`GraphAnalyzer` keys by graph).
pub fn discover_depth_neutral_fns(llbc: &Llbc) -> Vec<String> {
    discover_stack_effects(llbc).5
}

/// Classify `llbc`'s own functions and mark its set complete, for a caller
/// that lowers one artefact on its own.  Callees from other artefacts count
/// as insensitive here; the linked translation registers them first.
pub fn ensure_stack_sensitive_fns(llbc: &Llbc) {
    if llbc.stack_sensitive_fns_complete() {
        return;
    }
    let (found, leaves, params, ret_idx, observes, neutrals) = discover_stack_effects(llbc);
    llbc.register_stack_sensitive_fns(found);
    llbc.register_stack_observes_fns(observes);
    llbc.register_stack_leaves_above_fns(leaves);
    llbc.register_stack_param_slots_fns(params);
    llbc.register_stack_returns_index_fns(ret_idx);
    llbc.register_stack_depth_neutral_fns(neutrals);
    llbc.mark_stack_sensitive_fns_complete();
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_charon_reader::Llbc;
    use std::collections::HashMap;

    #[test]
    fn origportal_alias_is_a_registered_rewritten_portal() {
        let llbc = fixture_llbc();
        assert!(!resolved_callee_is_origportal_alias(
            &llbc,
            "eval::eval_loop_jit"
        ));
        llbc.register_rewritten_origportal_fns(["eval::eval_loop_jit".into()]);
        assert!(resolved_callee_is_origportal_alias(
            &llbc,
            "pyre_interpreter::eval::eval_loop_jit"
        ));
        assert!(!resolved_callee_is_origportal_alias(
            &llbc,
            "eval::dispatch_frame"
        ));
        assert!(!resolved_callee_is_origportal_alias(
            &llbc,
            "pyre_interpreter::eval::dispatch_frame"
        ));
        assert!(!resolved_callee_is_origportal_alias(&llbc, "eval_loop_jit"));
        assert!(!resolved_callee_is_origportal_alias(
            &llbc,
            "other::eval_loop_jit"
        ));
        assert!(!resolved_callee_is_origportal_alias(
            &llbc,
            "eval::eval_loop"
        ));
        assert!(!resolved_callee_is_origportal_alias(
            &llbc,
            "eval::eval_loop_jit::inner"
        ));
    }

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
                "kind": "Regular"
            })
        };
        let open = json!({"Call": {
            "call": {
                "func": {"Regular": {"kind": {"Fun": 1}, "generics": {}}},
                "args": []}, "dest": place(1),
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

    fn opaque_fun_sig(
        def_id: u64,
        name: &[&str],
        inputs: Vec<Value>,
        output: Value,
        body: Value,
    ) -> Value {
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
                "inputs": inputs,
                "output": output
            },
            "body": body
        })
    }

    fn frame_anchor_adt() -> Value {
        json!({
            "Adt": {
                "id": 0,
                "generics": {
                    "regions": [],
                    "types": [],
                    "const_generics": [],
                    "trait_refs": []
                }
            }
        })
    }

    fn frame_anchor_type_decl() -> Value {
        let ident = |s: &str| json!({"Ident": [s, 0]});
        json!({
            "def_id": 0,
            "item_meta": {
                "name": [
                    ident("pyre_interpreter"),
                    ident("eval"),
                    ident("FrameAnchor")
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
            "kind": {"Struct": []}
        })
    }

    /// A `pin_roots` of a same-block array, with a cleanup block in front of
    /// that array block. The body keeps every extracted block at its original
    /// index; erase still accepts the publish.
    #[test]
    fn erase_accepts_publish_when_cleanup_blocks_precede_the_array() {
        let ty = json!({"Deduplicated": 0});
        let place = |i: u64| json!({"kind": {"Local": i}, "ty": ty});
        let local = |i: u64| json!({"index": i, "name": null, "span": span(), "ty": ty});
        let stmt = |kind: Value| json!({"span": span(), "kind": kind});
        let bb = |statements: Vec<Value>, kind: Value, is_cleanup: bool| {
            let unwind = if is_cleanup { "Cleanup" } else { "Regular" };
            json!({
                "statements": statements,
                "terminator": {"span": span(), "kind": kind},
                "kind": unwind
            })
        };
        let call = |fun: u64, args: Vec<Value>, dest: u64, target: u64, on_unwind: u64| {
            json!({"Call": {
                "call": {
                    "func": {"Regular": {"kind": {"Fun": fun}, "generics": {}}},
                    "args": args}, "dest": place(dest),
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
            "extracted blocks stay at their original indices"
        );
        assert!(
            body.body[1].is_cleanup
                && matches!(body.body[1].term_ref(&llbc), Ok(TermKind::UnwindResume)),
            "block 1 is the cleanup UnwindResume"
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
                "type_decls": [frame_anchor_type_decl()],
                "fun_decls": [
                    null,
                    opaque_fun(1, &["pyre_object", "gc_roots", "push_roots"], json!("Opaque")),
                    opaque_fun(2, &["pyre_object", "gc_roots", "shadow_stack_get"], json!("Opaque")),
                    opaque_fun(3, &["pyre_object", "gc_roots", "shadow_stack_len"], json!("Opaque")),
                    opaque_fun(4, &["pyre_object", "gc_roots", "pin_root"], json!("Opaque")),
                    opaque_fun(5, &["pyre_interpreter", "eval", "frame_anchor_push"], json!("Opaque")),
                    opaque_fun(6, &["pyre_interpreter", "eval", "frame_anchor_live"], json!("Opaque")),
                    opaque_fun(7, &["pyre_interpreter", "eval", "frame_anchor_release"], json!("Opaque")),
                    opaque_fun(8, &["pyre_object", "gc_roots", "pin_roots"], json!("Opaque"))
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
        bb_stmts(vec![], kind)
    }

    fn bb_cleanup(kind: Value) -> Value {
        json!({
            "statements": [],
            "terminator": {"span": span(), "kind": kind},
            "kind": "Cleanup"
        })
    }

    fn call_fun(fun: u64, args: Vec<Value>, dest: u64, target: u64) -> Value {
        json!({"Call": {
            "call": {
                "func": {"Regular": {"kind": {"Fun": fun}, "generics": {}}},
                "args": args}, "dest": place(dest),
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
                bb(call_fun(4, vec![json!({"Read": place(0)})], 2, 2)),
                bb(json!({"Switch": {
                    "discr": {"Read": place(0)},
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
                bb(call_fun(2, vec![json!({"Read": place(0)})], 3, 3)),
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
                bb(call_fun(4, vec![json!({"Read": place(0)})], 2, 2)),
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
                    opaque_fun(11, &["other", "observes_and_leaves"], json!("Opaque")),
                    opaque_fun(12, &["core", "mem", "drop"], json!("Opaque"))
                ]
            }
        });
        let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc");
        let id = |suffix: &str| stack_fn_identity(&llbc, suffix);
        llbc.register_stack_sensitive_fns([
            id("unproven_callee"),
            id("leaves_above_callee"),
            id("reload_slot"),
            id("pin_helper"),
            id("storage_helper"),
            id("observes_and_leaves"),
        ]);
        llbc.register_stack_observes_fns([id("unproven_callee"), id("observes_and_leaves")]);
        llbc.register_stack_leaves_above_fns([
            id("leaves_above_callee"),
            id("pin_helper"),
            id("observes_and_leaves"),
        ]);
        llbc.register_stack_param_slots_fns([(id("reload_slot"), vec![0])]);
        llbc.register_stack_returns_index_fns([id("pin_helper"), id("storage_helper")]);
        llbc.mark_stack_sensitive_fns_complete();
        llbc
    }

    fn stack_fn_identity(llbc: &Llbc, suffix: &str) -> String {
        llbc.iter_fun_decls()
            .find(|fd| {
                let path = fd.item_meta.name_path();
                path == suffix || path.ends_with(&format!("::{suffix}"))
            })
            .map(|fd| super::super::scope_owner_identity(llbc, fd))
            .unwrap_or_else(|| panic!("missing FunDecl ending with {suffix}"))
    }

    fn bb_stmts(statements: Vec<Value>, kind: Value) -> Value {
        json!({
            "statements": statements,
            "terminator": {"span": span(), "kind": kind},
            "kind": "Regular"
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

    /// A cleanup block that reassigns a local is not a second definition.
    #[test]
    fn local_def_counts_skips_cleanup_reassign() {
        let llbc = fixture_llbc();
        let use_copy0 = json!({"Use": [{"Read": place(0)}, "Yes"]});
        let body = body_of(
            2,
            vec![
                bb_stmts(vec![assign_local(1, use_copy0.clone())], json!("Return")),
                json!({
                    "statements": [assign_local(1, use_copy0)],
                    "terminator": {"span": span(), "kind": "UnwindResume"},
                    "kind": "Cleanup"
                }),
            ],
        );
        let forward_blocks = super::super::forward_reachable_mask(&llbc, &body);
        let counts = local_def_counts(&body, &llbc, &forward_blocks);
        assert_eq!(counts.get(1).copied(), Some(1));
    }

    /// `zip_two_tuple_next` / `tuple_iter_descr_next`: pin, then name the
    /// slot as `shadow_stack_len() - 1`, then get it.
    fn len_minus_one_body(sub_op: Value) -> Unstructured {
        let copy = |i| json!({"Read": place(i)});
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
    fn analyze_accepts_frame_anchor_push_live_release() {
        let llbc = stack_ops_llbc();
        let copy = |i| json!({"Read": place(i)});
        let body = body_of(
            5,
            vec![
                bb(call_fun(5, vec![copy(1)], 2, 1)),
                bb(call_fun(6, vec![copy(2)], 3, 2)),
                bb(call_fun(7, vec![copy(2)], 4, 3)),
                bb(json!("Return")),
            ],
        );
        match analyze(&body, &llbc) {
            Ok(None) => {}
            Ok(Some(_)) => panic!("FrameAnchor ops are identities, no bracket"),
            Err(reason) => panic!("FrameAnchor ops are identities, got {reason}"),
        }
        for bb in 0..3 {
            let Ok(TermKind::Call { call, .. }) = body.body[bb].term_ref(&llbc) else {
                panic!("expected Call");
            };
            assert!(
                call_is_anchor_identity(call, &llbc),
                "bb{bb} must be a FrameAnchor identity"
            );
            assert_eq!(classify_call(call, &llbc), None);
        }
        assert!(
            stack_walk(&body, &llbc, &|_| CalleeEffect::none())
                .effect()
                .is_none(),
            "FrameAnchor push/live/release must be depth-neutral"
        );
        let fd: FunDecl =
            serde_json::from_value(opaque_fun(9, &["other", "subject"], json!("Opaque")))
                .expect("subject FunDecl");
        match erase_shadow_stack(&fd, &body, &llbc) {
            Ok(None) => {}
            Ok(Some(_)) => panic!("FrameAnchor identities are not a bracket"),
            Err(reason) => panic!("FrameAnchor identities are not a bracket, got {reason}"),
        }
    }

    /// `eval::{from_raw,live,drop}` without a `FrameAnchor` path segment:
    /// dest / receiver type is the owner, as is the helper signature.
    #[test]
    fn analyze_accepts_from_raw_live_by_frame_anchor_ty() {
        let anchor = frame_anchor_adt();
        let file = json!({
            "charon_version": "t",
            "has_errors": false,
            "translated": {
                "crate_name": "c",
                "type_decls": [frame_anchor_type_decl()],
                "fun_decls": [
                    null,
                    opaque_fun_sig(
                        1,
                        &["pyre_interpreter", "eval", "from_raw"],
                        vec![ty()],
                        anchor.clone(),
                        json!("Opaque")
                    ),
                    opaque_fun_sig(
                        2,
                        &["pyre_interpreter", "eval", "live"],
                        vec![anchor.clone()],
                        ty(),
                        json!("Opaque")
                    ),
                    opaque_fun_sig(
                        3,
                        &["pyre_interpreter", "eval", "drop"],
                        vec![anchor.clone()],
                        ty(),
                        json!("Opaque")
                    )
                ]
            }
        });
        let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc");
        llbc.mark_stack_sensitive_fns_complete();
        let place_a = |i: u64| json!({"kind": {"Local": i}, "ty": anchor.clone()});
        let call = |fun: u64, args: Vec<Value>, dest: u64, dest_ty: Value, target: u64| {
            json!({"Call": {
                "call": {
                    "func": {"Regular": {"kind": {"Fun": fun}, "generics": {}}},
                    "args": args
                },
                "dest": {"kind": {"Local": dest}, "ty": dest_ty},
                "target": target,
                "on_unwind": 99
            }})
        };
        let body = body_of(
            5,
            vec![
                bb(call(
                    1,
                    vec![json!({"Read": place(1)})],
                    2,
                    anchor.clone(),
                    1,
                )),
                bb(call(2, vec![json!({"Read": place_a(2)})], 3, ty(), 2)),
                bb(call(3, vec![json!({"Move": place_a(2)})], 4, ty(), 3)),
                bb(json!("Return")),
            ],
        );
        match analyze(&body, &llbc) {
            Ok(None) => {}
            Ok(Some(_)) => panic!("FrameAnchor ops are identities, no bracket"),
            Err(reason) => panic!("FrameAnchor ops are identities, got {reason}"),
        }
        let fd = llbc.fn_by_id(1).expect("from_raw");
        assert!(
            super::super::frame_anchor::is_helper(fd, &llbc),
            "from_raw typed as FrameAnchor is a helper identity"
        );
        match erase_shadow_stack(fd, &body, &llbc) {
            Ok(None) => {}
            Ok(Some(_)) => panic!("from_raw helper must skip erase"),
            Err(reason) => panic!("from_raw helper must skip erase, got {reason}"),
        }
    }

    #[test]
    fn other_drop_is_not_a_scope_close() {
        let file = json!({
            "charon_version": "t",
            "has_errors": false,
            "translated": {
                "crate_name": "c",
                "fun_decls": [
                    null,
                    opaque_fun(1, &["other", "drop"], json!("Opaque"))
                ]
            }
        });
        let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc");
        llbc.mark_stack_sensitive_fns_complete();
        let body = body_of(
            2,
            vec![
                bb(call_fun(1, vec![json!({"Move": place(1)})], 0, 1)),
                bb(json!("Return")),
            ],
        );
        let Ok(TermKind::Call { call, .. }) = body.body[0].term_ref(&llbc) else {
            panic!("expected Call");
        };
        assert_eq!(mem_drop_scope_local(call, &llbc), None);
    }

    #[test]
    fn other_frame_anchor_push_is_not_anchor_open() {
        let file = json!({
            "charon_version": "t",
            "has_errors": false,
            "translated": {
                "crate_name": "c",
                "type_decls": [frame_anchor_type_decl()],
                "fun_decls": [
                    null,
                    opaque_fun(1, &["other", "frame_anchor_push"], json!("Opaque"))
                ]
            }
        });
        let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc");
        llbc.mark_stack_sensitive_fns_complete();
        let body = body_of(1, vec![bb(call_fun(1, vec![], 0, 1)), bb(json!("Return"))]);
        let Ok(TermKind::Call { call, .. }) = body.body[0].term_ref(&llbc) else {
            panic!("expected Call");
        };
        assert_eq!(classify_call(call, &llbc), None);
        assert!(
            !call_is_anchor_identity(call, &llbc),
            "other::frame_anchor_push is not a FrameAnchor identity"
        );
    }

    #[test]
    fn same_module_helper_is_not_a_frame_anchor_call() {
        let file = json!({
            "charon_version": "t",
            "has_errors": false,
            "translated": {
                "crate_name": "c",
                "type_decls": [frame_anchor_type_decl()],
                "fun_decls": [
                    null,
                    opaque_fun(1, &["pyre_interpreter", "eval", "some_helper"], json!("Opaque"))
                ]
            }
        });
        let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc");
        llbc.mark_stack_sensitive_fns_complete();
        let body = body_of(1, vec![bb(call_fun(1, vec![], 0, 1)), bb(json!("Return"))]);
        let Ok(TermKind::Call { call, .. }) = body.body[0].term_ref(&llbc) else {
            panic!("expected Call");
        };
        assert_eq!(classify_call(call, &llbc), None);
    }

    /// RootScope open, nested FrameAnchor identities, then get of an
    /// incoming slice. Slice Get is accepted only while `open > 0`;
    /// FrameAnchor ops do not close the still-active RootScope.
    #[test]
    fn anchor_close_inside_root_scope_keeps_scope_open() {
        let llbc = stack_ops_llbc();
        let copy = |i| json!({"Read": place(i)});
        let body = body_of_args(
            1,
            8,
            vec![
                bb(call_fun(1, vec![], 2, 1)),
                bb(call_fun(5, vec![copy(0)], 3, 2)),
                bb(call_fun(7, vec![copy(3)], 4, 3)),
                bb(call_fun(8, vec![copy(1)], 5, 4)),
                bb(call_fun(2, vec![copy(5)], 6, 5)),
                bb(drop_local(2, 6)),
                bb(json!("Return")),
            ],
        );
        match analyze(&body, &llbc) {
            Ok(Some(_)) => {}
            Ok(None) => panic!("expected accepted erase, got no bracket"),
            Err(reason) => panic!("expected accepted erase, got {reason}"),
        }
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

    /// `Drop` of the guard lives in an `is_cleanup` block. `unstructured`
    /// keeps those blocks, so a later `UnwindTerminate` can be last while
    /// `UnwindResume` remains in the body. Return with published slots is
    /// accepted when a retained block is `UnwindResume`.
    #[test]
    fn analyze_accepts_return_with_slots_when_unwind_resume_is_not_last() {
        let llbc = stack_ops_llbc();
        let copy = |i| json!({"Read": place(i)});
        let body = body_of(
            3,
            vec![
                bb(call_fun(1, vec![], 1, 1)),
                bb(call_fun(4, vec![copy(0)], 2, 2)),
                bb(json!("Return")),
                bb_cleanup(json!("UnwindResume")),
                bb_cleanup(json!("UnwindTerminate")),
            ],
        );
        match analyze(&body, &llbc) {
            Ok(Some(_)) => {}
            Ok(None) => panic!("expected accepted erase, got no bracket"),
            Err(reason) => panic!("expected accepted erase, got {reason}"),
        }
    }

    /// `pin_self`: pin, `len - 1`, return that index. The return place is
    /// the Index special; erase would strip its assignment. Last block is
    /// `UnwindResume` so the cleanup-drop depth hack does not refuse first.
    #[test]
    fn analyze_refuses_returning_the_len_minus_one_slot_index() {
        let llbc = stack_ops_llbc();
        let copy = |i| json!({"Read": place(i)});
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

    /// Identity of `SharedOpcodeHandler::anchor` (`eval::<Impl>::anchor`):
    /// output is FrameAnchor, leaf `anchor`, no FrameAnchor path segment.
    /// Skip like `new` so the host push inside is not erased into a slot.
    #[test]
    fn is_frame_anchor_helper_skips_anchor_returning_frame_anchor() {
        let anchor = frame_anchor_adt();
        let file = json!({
            "charon_version": "t",
            "has_errors": false,
            "translated": {
                "crate_name": "c",
                "type_decls": [frame_anchor_type_decl()],
                "fun_decls": [
                    null,
                    opaque_fun_sig(
                        1,
                        &["eval", "<Impl>", "anchor"],
                        vec![ty()],
                        anchor.clone(),
                        json!("Opaque")
                    )
                ]
            }
        });
        let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc");
        llbc.mark_stack_sensitive_fns_complete();
        let fd = llbc.fn_by_id(1).expect("anchor");
        assert!(
            super::super::frame_anchor::is_helper(fd, &llbc),
            "anchor returning FrameAnchor is a helper identity"
        );
        let body = body_of(2, vec![bb(json!("Return"))]);
        match erase_shadow_stack(fd, &body, &llbc) {
            Ok(None) => {}
            Ok(Some(_)) => panic!("anchor identity must skip erase"),
            Err(reason) => panic!("anchor identity must skip erase, got {reason}"),
        }
    }

    /// Returning a FrameAnchor is a Ref identity of the frame, not a
    /// published slot index. Analyze sees no bracket.
    #[test]
    fn analyze_refuses_returning_an_open_frame_anchor() {
        let llbc = stack_ops_llbc();
        let copy = |i| json!({"Read": place(i)});
        let body = body_of(
            2,
            vec![
                bb(call_fun(5, vec![copy(1)], 0, 1)),
                bb(json!("Return")),
                bb(json!("UnwindResume")),
            ],
        );
        match analyze(&body, &llbc) {
            Ok(None) => {}
            Ok(Some(_)) => panic!("returning FrameAnchor is not a bracket"),
            Err(reason) => panic!("returning FrameAnchor is not a bracket, got {reason}"),
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
        let copy = |i| json!({"Read": place(i)});
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
        let unproven = stack_fn_identity(&llbc, "unproven_callee");
        let observes = |identity: &str| {
            if identity == unproven {
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
                bb(call_fun(4, vec![json!({"Read": place(0)})], 2, 2)),
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
                bb(call_fun(4, vec![json!({"Read": place(0)})], 2, 2)),
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
        let effect = |identity: &str| callee_effect_of(&llbc, identity);
        let observes_inside = body_of(
            4,
            vec![
                bb(call_fun(1, vec![], 1, 1)),
                bb(call_fun(4, vec![json!({"Read": place(0)})], 2, 2)),
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
                bb(call_fun(4, vec![json!({"Read": place(0)})], 2, 2)),
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
                bb(call_fun(4, vec![json!({"Read": place(0)})], 2, 2)),
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
        // `drop(roots)` is `core::mem::drop` of the guard (a Call, not a
        // `Drop` terminator). Same rewind as `TermKind::Drop`.
        let mem_drop_close = body_of(
            3,
            vec![
                bb(call_fun(1, vec![], 1, 1)),
                bb(call_fun(4, vec![json!({"Read": place(0)})], 2, 2)),
                bb(call_fun(12, vec![json!({"Move": place(1)})], 3, 3)),
                bb(json!("Return")),
            ],
        );
        assert!(
            body_is_depth_neutral(&mem_drop_close, &llbc, &effect),
            "Open -> Pin -> mem::drop(guard) is depth-neutral"
        );
        match analyze(&mem_drop_close, &llbc) {
            Ok(Some(_)) => {}
            Ok(None) => panic!("expected erasure of Open/Pin/mem::drop"),
            Err(reason) => panic!("expected erasure of Open/Pin/mem::drop, got {reason}"),
        }
    }

    #[test]
    fn param_slots_callee_own_index_is_neutral_untracked_is_observes() {
        let llbc = stack_ops_llbc_with_callee();
        let effect = |identity: &str| callee_effect_of(&llbc, identity);
        let callee = body_of_args(
            1,
            3,
            vec![
                bb(call_fun(2, vec![json!({"Read": place(1)})], 2, 1)),
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
                bb(call_fun(4, vec![json!({"Read": place(0)})], 2, 2)),
                bb(call_fun(3, vec![], 3, 3)),
                bb(call_fun(8, vec![json!({"Read": place(3)})], 4, 4)),
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
                bb(call_fun(4, vec![json!({"Read": place(0)})], 2, 2)),
                bb(call_fun(8, vec![json!({"Read": place(0)})], 3, 3)),
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
                        json!({"BinaryOp": ["Add", json!({"Read": place(1)}), json!({"Read": place(3)})]}),
                    )],
                    call_fun(2, vec![json!({"Read": place(2)})], 3, 1),
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
                bb(call_fun(4, vec![json!({"Read": place(0)})], 2, 2)),
                bb(call_fun(3, vec![], 3, 3)),
                bb_stmts(
                    vec![assign_local(
                        4,
                        json!({"BinaryOp": ["Add", json!({"Read": place(3)}), json!({"Read": place(5)})]}),
                    )],
                    call_fun(2, vec![json!({"Read": place(4)})], 5, 4),
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
                bb(call_fun(2, vec![json!({"Read": place(3)})], 4, 3)),
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
        let copy = |i| json!({"Read": place(i)});
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

    fn effect_params(slots: Vec<u8>) -> CalleeEffect {
        CalleeEffect {
            observes: false,
            leaves_above: false,
            param_slots: slots,
            returns_index: false,
        }
    }

    #[test]
    fn callee_effect_join_leq_laws() {
        let bottom = CalleeEffect::bottom();
        let none = CalleeEffect::none();
        let observes = CalleeEffect::observes();
        let leaves = CalleeEffect::leaves_above();
        let params = effect_params(vec![0, 2]);
        let worst = CalleeEffect::worst();
        for x in [&bottom, &none, &observes, &leaves, &params, &worst] {
            assert!(x.leq(x), "leq reflexive {}", x.describe());
            assert_eq!(
                x.join_two_results(x),
                *x,
                "join idempotent {}",
                x.describe()
            );
            assert!(bottom.leq(x), "bottom ≼ {}", x.describe());
        }
        assert!(none.leq(&observes));
        assert!(none.leq(&leaves));
        assert!(none.leq(&params));
        assert!(observes.leq(&worst));
        assert!(leaves.leq(&worst));
        let joined = params.join_two_results(&observes);
        assert_eq!(observes.join_two_results(&params), joined);
        assert!(params.leq(&joined));
        assert!(observes.leq(&joined));
        assert!(joined.observes);
        assert_eq!(joined.param_slots, vec![0, 2]);
        assert!(!joined.returns_index);
        assert!(!joined.leaves_above);
        let a = effect_params(vec![0]);
        let b = effect_params(vec![1]);
        let u = a.join_two_results(&b);
        assert_eq!(u.param_slots, vec![0, 1]);
        assert!(a.leq(&u) && b.leq(&u));
        let must = CalleeEffect::bottom().join_two_results(&none);
        assert!(!must.returns_index, "true ≼ false, join is AND");
    }

    #[test]
    fn join_param_slots_with_observes_keeps_both() {
        let joined = effect_params(vec![0]).join_two_results(&CalleeEffect::observes());
        assert!(joined.observes && joined.param_slots == vec![0]);
        assert!(!joined.leaves_above && !joined.returns_index);
    }

    fn unstructured_fun_body(n_locals: u64, blocks: Vec<Value>) -> Value {
        json!({
            "Unstructured": {
                "span": span(),
                "locals": {
                    "arg_count": 0,
                    "locals": (0..n_locals).map(local_decl).collect::<Vec<_>>()
                },
                "body": blocks
            }
        })
    }

    #[test]
    fn recursive_pair_converges_without_visit_cap() {
        let file = json!({
            "charon_version": "t",
            "has_errors": false,
            "translated": {
                "crate_name": "c",
                "fun_decls": [
                    null,
                    opaque_fun(
                        1,
                        &["other", "rec_a"],
                        unstructured_fun_body(
                            2,
                            vec![bb(call_fun(2, vec![], 1, 1)), bb(json!("Return"))],
                        ),
                    ),
                    opaque_fun(
                        2,
                        &["other", "rec_b"],
                        unstructured_fun_body(
                            2,
                            vec![bb(call_fun(1, vec![], 1, 1)), bb(json!("Return"))],
                        ),
                    )
                ]
            }
        });
        let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc");
        let (sens, leaves, params, ret, obs, neutrals) = discover_stack_effects(&llbc);
        assert!(sens.is_empty(), "sens={sens:?}");
        assert!(leaves.is_empty(), "leaves={leaves:?}");
        assert!(params.is_empty(), "params={params:?}");
        assert!(ret.is_empty(), "ret={ret:?}");
        assert!(obs.is_empty(), "obs={obs:?}");
        assert!(neutrals.is_empty(), "neutrals={neutrals:?}");
    }

    #[test]
    fn returns_index_cycle_stays_true() {
        let file = json!({
            "charon_version": "t",
            "has_errors": false,
            "translated": {
                "crate_name": "c",
                "fun_decls": [
                    null,
                    opaque_fun(
                        1,
                        &["pyre_object", "gc_roots", "shadow_stack_len"],
                        json!("Opaque"),
                    ),
                    opaque_fun(
                        2,
                        &["other", "ret_a"],
                        unstructured_fun_body(
                            3,
                            vec![
                                bb(call_fun(1, vec![], 0, 1)),
                                bb(call_fun(3, vec![], 2, 2)),
                                bb(json!("Return")),
                            ],
                        ),
                    ),
                    opaque_fun(
                        3,
                        &["other", "ret_b"],
                        unstructured_fun_body(
                            3,
                            vec![
                                bb(call_fun(1, vec![], 0, 1)),
                                bb(call_fun(2, vec![], 2, 2)),
                                bb(json!("Return")),
                            ],
                        ),
                    )
                ]
            }
        });
        let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc");
        let (sens, _leaves, _params, ret, obs, _) = discover_stack_effects(&llbc);
        assert!(
            ret.iter().any(|id| id.contains("ret_a")) && ret.iter().any(|id| id.contains("ret_b")),
            "cycle must keep returns_index: ret={ret:?} sens={sens:?} obs={obs:?}"
        );
        assert!(obs.is_empty(), "obs={obs:?}");
    }

    #[test]
    fn observes_param_slots_untracked_inside_open_is_observes() {
        let llbc = stack_ops_llbc_with_callee();
        let id = stack_fn_identity(&llbc, "unproven_callee");
        llbc.register_stack_param_slots_fns([(id, vec![0])]);
        let effect = |identity: &str| callee_effect_of(&llbc, identity);
        let untracked = body_of(
            4,
            vec![
                bb(call_fun(1, vec![], 1, 1)),
                bb(call_fun(4, vec![json!({"Read": place(0)})], 2, 2)),
                bb(call_fun(5, vec![json!({"Read": place(0)})], 3, 3)),
                bb(drop_local(1, 4)),
                bb(json!("Return")),
            ],
        );
        let walk = stack_walk(&untracked, &llbc, &effect);
        assert!(
            walk.effect().observes,
            "Observes+ParamSlots callee with an untracked index inside Open is Observes: {:?}",
            walk.effect()
        );
        let own = body_of(
            5,
            vec![
                bb(call_fun(1, vec![], 1, 1)),
                bb(call_fun(4, vec![json!({"Read": place(0)})], 2, 2)),
                bb(call_fun(3, vec![], 3, 3)),
                bb(call_fun(5, vec![json!({"Read": place(3)})], 4, 4)),
                bb(drop_local(1, 5)),
                bb(json!("Return")),
            ],
        );
        assert!(
            body_is_depth_neutral(&own, &llbc, &effect),
            "own index to Observes+ParamSlots inside Open is depth-neutral"
        );
        match analyze(&own, &llbc) {
            Err("calls-stack-sensitive-fn") => {}
            Ok(Some(_)) => panic!("expected calls-stack-sensitive-fn, got accepted"),
            Ok(None) => panic!("expected calls-stack-sensitive-fn, got no bracket"),
            Err(reason) => panic!("expected calls-stack-sensitive-fn, got {reason}"),
        }
    }

    #[test]
    fn option_wrap_of_own_index_stays_own() {
        let llbc = stack_ops_llbc();
        let copy = |i| json!({"Read": place(i)});
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
                        assign_local(5, json!({"Use": [{"Read": payload}, "No"]})),
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
                    vec![assign_local(2, json!({"Use": [{"Read": field0}, "No"]}))],
                    call_fun(2, vec![json!({"Read": place(2)})], 3, 1),
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
                        json!({"Use": [{"Read": deref_field}, "No"]}),
                    )],
                    call_fun(2, vec![json!({"Read": place(2)})], 3, 1),
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
        let copy = |i| json!({"Read": place(i)});
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
        let effect = |identity: &str| callee_effect_of(&llbc, identity);
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
            {"Use": [{"Read": place(6)}, "Yes"]}
        ]});
        let assert_term = json!({"Assert": {
            "assert": {
                "cond": {"Read": place(3)},
                "expected": true,
                "check_kind": {"BoundsCheck": {
                    "len": {"Read": place(5)},
                    "index": {"Read": place(3)}
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
                    "discr": {"Read": place(0)},
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
        let specials = HashMap::from([(
            5usize,
            Special::Index {
                k: 0,
                relative: false,
            },
        )]);
        let assert: majit_charon_reader::ullbc::AssertStmt = serde_json::from_value(json!({
            "cond": {"Read": place(5)},
            "expected": true,
            "check_kind": {"BoundsCheck": {
                "len": {"Read": place(9)},
                "index": {"Read": place(5)}
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
        let specials = HashMap::from([
            (
                5usize,
                Special::Index {
                    k: 0,
                    relative: false,
                },
            ),
            (
                6usize,
                Special::Index {
                    k: 3,
                    relative: true,
                },
            ),
        ]);
        let assert: majit_charon_reader::ullbc::AssertStmt = serde_json::from_value(json!({
            "cond": {"Read": place(5)},
            "expected": true,
            "check_kind": {"BoundsCheck": {
                "len": {"Read": place(6)},
                "index": {"Read": place(5)}
            }}
        }))
        .expect("assert");
        assert!(
            assert_is_slot_index_bounds_check(&assert, &specials, &HashMap::new(), &fixture_llbc()),
            "index 0 of a 3-slot erased array is true by construction"
        );
    }

    fn getter_llbc() -> Llbc {
        let capture_body = json!({
            "Unstructured": {
                "span": span(),
                "locals": {
                    "arg_count": 2,
                    "locals": (0..4).map(local_decl).collect::<Vec<_>>()
                },
                "body": [
                    bb(call_fun(3, vec![json!({"Read": place(1)})], 0, 1)),
                    bb(json!("Return"))
                ]
            }
        });
        let field0 = json!({
            "kind": {"Projection": [place(3), {"Field": [null, 0]}]},
            "ty": ty()
        });
        let plus_const_body = json!({
            "Unstructured": {
                "span": span(),
                "locals": {
                    "arg_count": 2,
                    "locals": (0..5).map(local_decl).collect::<Vec<_>>()
                },
                "body": [
                    bb_stmts(
                        vec![assign_local(
                            3,
                            json!({"BinaryOp": [
                                "AddChecked",
                                {"Read": place(1)},
                                usize_lit(1)
                            ]}),
                        )],
                        json!({"Assert": {
                            "assert": {
                                "cond": {"Move": {
                                    "kind": {"Projection": [place(3), {"Field": [null, 1]}]},
                                    "ty": ty()
                                }},
                                "expected": false,
                                "check_kind": {"Overflow": [
                                    {"Add": "Wrap"},
                                    {"Read": place(1)},
                                    usize_lit(1)
                                ]}
                            },
                            "target": 1,
                            "on_unwind": 99
                        }}),
                    ),
                    bb_stmts(
                        vec![assign_local(4, json!({"Use": [{"Move": field0}, "Yes"]}))],
                        call_fun(3, vec![json!({"Move": place(4)})], 0, 2),
                    ),
                    bb(json!("Return"))
                ]
            }
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
                    opaque_fun(3, &["pyre_object", "gc_roots", "shadow_stack_get"], json!("Opaque")),
                    opaque_fun(4, &["pyre_object", "gc_roots", "pin_root"], json!("Opaque")),
                    opaque_fun(5, &["fixture", "capture", "call"], capture_body),
                    opaque_fun(6, &["fixture", "plus_const", "call"], plus_const_body),
                    opaque_fun(7, &["fixture", "other"], json!("Opaque"))
                ]
            }
        });
        let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc");
        llbc.mark_stack_sensitive_fns_complete();
        llbc
    }

    fn copy_local(i: u64) -> Value {
        json!({"Read": place(i)})
    }

    fn move_local(i: u64) -> Value {
        json!({"Move": place(i)})
    }

    fn borrow_local(dest: u64, src: u64) -> Value {
        assign_local(
            dest,
            json!({"Ref": {"place": place(src), "kind": "Shared", "ptr_metadata": null}}),
        )
    }

    fn closure_aggregate(dest: u64, capture: u64) -> Value {
        assign_local(
            dest,
            json!({"Aggregate": [{"Adt": [3026, null]}, [move_local(capture)]]}),
        )
    }

    fn empty_unit(dest: u64) -> Value {
        assign_local(dest, json!({"Aggregate": [{"Adt": [0, null]}, []]}))
    }

    fn pin_two_array_stmts() -> Vec<Value> {
        vec![
            assign_local(8, json!({"Use": [copy_local(0), "Yes"]})),
            assign_local(9, json!({"Use": [copy_local(0), "Yes"]})),
            assign_local(
                6,
                json!({"Aggregate": [
                    {"Array": [ty(), {"Deduplicated": 1}, null]},
                    [move_local(8), move_local(9)]
                ]}),
            ),
            borrow_local(7, 6),
        ]
    }

    fn capture_of_base(borrow: u64, closure: u64, recv: u64, unit: u64) -> Vec<Value> {
        vec![
            borrow_local(borrow, 4),
            closure_aggregate(closure, borrow),
            borrow_local(recv, closure),
            empty_unit(unit),
        ]
    }

    /// `long_rshift` shape: Open, `pin_roots(&[a, b])`, closures over the
    /// base, `shadow_stack_get` of the captured index and of `captured + 1`.
    fn long_rshift_shape_body(call_after_close: bool) -> Unstructured {
        let mut blocks = vec![
            bb(call_fun(1, vec![], 1, 1)),
            bb_stmts(
                pin_two_array_stmts(),
                call_fun(2, vec![move_local(7)], 4, 2),
            ),
        ];
        let capture_a = capture_of_base(12, 11, 18, 19);
        if call_after_close {
            blocks.push(bb_stmts(capture_a, drop_local(1, 3)));
            blocks.push(bb(call_fun(5, vec![copy_local(18), copy_local(19)], 30, 4)));
            blocks.push(bb(json!("Return")));
        } else {
            blocks.push(bb_stmts(
                capture_a,
                call_fun(5, vec![copy_local(18), copy_local(19)], 30, 3),
            ));
            blocks.push(bb_stmts(
                capture_of_base(14, 13, 20, 21),
                call_fun(6, vec![copy_local(20), copy_local(21)], 31, 4),
            ));
            blocks.push(bb(drop_local(1, 5)));
            blocks.push(bb(json!("Return")));
        }
        body_of(32, blocks)
    }

    #[test]
    fn analyze_erases_long_rshift_closure_capture() {
        let llbc = getter_llbc();
        let body = long_rshift_shape_body(false);
        let plan = match analyze(&body, &llbc) {
            Ok(Some(plan)) => plan,
            Ok(None) => panic!("expected accepted erase, got no bracket"),
            Err(reason) => panic!("expected accepted erase, got {reason}"),
        };
        let n_locals = 32;
        let slot0 = local_place_json(n_locals, &ty());
        let slot1 = local_place_json(n_locals + 1, &ty());
        let rewrite0 = plan.terms.get(&2).expect("capture get rewritten");
        assert_eq!(
            rewrite0.stmts,
            vec![json!({"Assign": [
                place(30),
                {"Use": [{"Read": slot0}, "Yes"]}
            ]})],
            "|| shadow_stack_get(base) is a copy of slot 0"
        );
        let rewrite1 = plan.terms.get(&3).expect("plus-const get rewritten");
        assert_eq!(
            rewrite1.stmts,
            vec![json!({"Assign": [
                place(31),
                {"Use": [{"Read": slot1}, "Yes"]}
            ]})],
            "|| shadow_stack_get(base + 1) is a copy of slot 1"
        );
    }

    #[test]
    fn analyze_refuses_capture_escapes_when_closure_is_passed() {
        let llbc = getter_llbc();
        let blocks = vec![
            bb(call_fun(1, vec![], 1, 1)),
            bb_stmts(
                pin_two_array_stmts(),
                call_fun(2, vec![move_local(7)], 4, 2),
            ),
            bb_stmts(
                capture_of_base(12, 11, 18, 19),
                call_fun(5, vec![copy_local(18), copy_local(19)], 30, 3),
            ),
            bb(call_fun(7, vec![copy_local(11)], 31, 4)),
            bb(drop_local(1, 5)),
            bb(json!("Return")),
        ];
        let body = body_of(32, blocks);
        match analyze(&body, &llbc) {
            Err("capture-escapes") => {}
            Ok(Some(_)) => panic!("expected capture-escapes, got accepted"),
            Ok(None) => panic!("expected capture-escapes, got no bracket"),
            Err(reason) => panic!("expected capture-escapes, got {reason}"),
        }
    }

    #[test]
    fn analyze_refuses_closure_get_after_close() {
        let llbc = getter_llbc();
        let body = long_rshift_shape_body(true);
        match analyze(&body, &llbc) {
            Err("get-above-depth") => {}
            Ok(Some(_)) => panic!("expected get-above-depth, got accepted"),
            Ok(None) => panic!("expected get-above-depth, got no bracket"),
            Err(reason) => panic!("expected get-above-depth, got {reason}"),
        }
    }

    #[test]
    fn analyze_getter_body_still_refuses_index_not_static() {
        let llbc = getter_llbc();
        let fd = llbc.fn_by_id(5).expect("capture getter");
        let body = fd.unstructured().expect("getter body");
        match analyze(&body, &llbc) {
            Err("index-not-static") => {}
            Ok(Some(_)) => panic!("expected index-not-static, got accepted"),
            Ok(None) => panic!("expected index-not-static, got no bracket"),
            Err(reason) => panic!("expected index-not-static, got {reason}"),
        }
    }

    fn incoming_slice_llbc() -> Llbc {
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
                    opaque_fun(5, &["pyre_object", "gc_roots", "pin_roots"], json!("Opaque"))
                ]
            }
        });
        let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture Llbc");
        llbc.mark_stack_sensitive_fns_complete();
        llbc
    }

    fn slice_index_use_json(slice_local: u64, dest: u64, k: u64) -> Value {
        let deref = json!({
            "kind": {"Projection": [place(slice_local), "Deref"]},
            "ty": ty()
        });
        json!({"Assign": [
            place(dest),
            {"Use": [{"Read": {
                "kind": {"Projection": [deref, {"Index": {
                    "offset": {"Const": [
                        {"Integer": {"Unsigned": ["Usize", k.to_string()]}},
                        {"Deduplicated": 0}
                    ]},
                    "from_end": false
                }}]},
                "ty": ty()
            }}, "Yes"]}
        ]})
    }

    /// `__majit_wrap_dict_descr_keys` shape: incoming `&[PyObjectRef]`,
    /// `get(base)`, then `len` / `pin_root` / `get(len)`.
    fn dict_descr_keys_shape_body(get_after_close: bool) -> Unstructured {
        let mut blocks = vec![
            bb(call_fun(1, vec![], 2, 1)),
            bb(call_fun(5, vec![copy_local(1)], 3, 2)),
        ];
        if get_after_close {
            blocks.push(bb(drop_local(2, 3)));
            blocks.push(bb(call_fun(2, vec![copy_local(3)], 4, 4)));
            blocks.push(bb(json!("Return")));
        } else {
            blocks.push(bb(call_fun(2, vec![copy_local(3)], 4, 3)));
            blocks.push(bb(call_fun(3, vec![], 5, 4)));
            blocks.push(bb(call_fun(4, vec![copy_local(4)], 6, 5)));
            blocks.push(bb(call_fun(2, vec![copy_local(5)], 7, 6)));
            blocks.push(bb(drop_local(2, 7)));
            blocks.push(bb(json!("Return")));
        }
        body_of_args(1, 8, blocks)
    }

    #[test]
    fn analyze_erases_incoming_slice_then_scalar_pin() {
        let llbc = incoming_slice_llbc();
        let body = dict_descr_keys_shape_body(false);
        let plan = match analyze(&body, &llbc) {
            Ok(Some(plan)) => plan,
            Ok(None) => panic!("expected accepted erase, got no bracket"),
            Err(reason) => panic!("expected accepted erase, got {reason}"),
        };
        let rewrite_base = plan.terms.get(&2).expect("get(base) rewritten");
        assert_eq!(
            rewrite_base.stmts,
            vec![slice_index_use_json(1, 4, 0)],
            "get(base) of an incoming slice is (*args)[0]"
        );
        let n_locals = 8;
        let slot0 = local_place_json(n_locals, &ty());
        let rewrite_len = plan.terms.get(&5).expect("get(len) rewritten");
        assert_eq!(
            rewrite_len.stmts,
            vec![json!({"Assign": [
                place(7),
                {"Use": [{"Read": slot0}, "Yes"]}
            ]})],
            "get(len) after pin_root is scalar slot 0; the slice adds no depth"
        );
        assert_eq!(plan.slot_count, 1, "incoming slice is not a scalar slot");
    }

    #[test]
    fn analyze_erases_array_aggregate_in_predecessor_block() {
        let llbc = incoming_slice_llbc();
        let array = json!({"Aggregate": [
            {"Array": [ty(), {"Deduplicated": 1}, null]},
            [move_local(3), move_local(4)]
        ]});
        let body = body_of(
            6,
            vec![
                bb(call_fun(1, vec![], 1, 1)),
                bb_stmts(vec![assign_local(2, array)], json!({"Goto": {"target": 2}})),
                bb(call_fun(5, vec![move_local(2)], 5, 3)),
                bb(drop_local(1, 4)),
                bb(json!("Return")),
            ],
        );
        match analyze(&body, &llbc) {
            Ok(Some(_)) => {}
            Ok(None) => panic!("expected accepted erase, got no bracket"),
            Err(reason) => panic!("expected accepted erase, got {reason}"),
        }
    }

    #[test]
    fn analyze_refuses_incoming_slice_get_with_runtime_index() {
        let llbc = incoming_slice_llbc();
        let body = body_of_args(
            1,
            6,
            vec![
                bb(call_fun(1, vec![], 2, 1)),
                bb(call_fun(5, vec![copy_local(1)], 3, 2)),
                bb_stmts(
                    vec![assign_local(
                        4,
                        json!({"BinaryOp": ["Add", copy_local(3), copy_local(0)]}),
                    )],
                    call_fun(2, vec![copy_local(4)], 5, 3),
                ),
                bb(drop_local(2, 4)),
                bb(json!("Return")),
            ],
        );
        match analyze(&body, &llbc) {
            Err("index-not-static") => {}
            Ok(Some(_)) => panic!("expected index-not-static, got accepted"),
            Ok(None) => panic!("expected index-not-static, got no bracket"),
            Err(reason) => panic!("expected index-not-static, got {reason}"),
        }
    }

    #[test]
    fn analyze_refuses_incoming_slice_get_after_close() {
        let llbc = incoming_slice_llbc();
        let body = dict_descr_keys_shape_body(true);
        match analyze(&body, &llbc) {
            Err("get-above-depth") => {}
            Ok(Some(_)) => panic!("expected get-above-depth, got accepted"),
            Ok(None) => panic!("expected get-above-depth, got no bracket"),
            Err(reason) => panic!("expected get-above-depth, got {reason}"),
        }
    }

    /// Early return at slice-only depth 0 joins `pin_root` (depth 1) at
    /// the guard's Drop; the Drop rewinds, so the join is accepted.
    #[test]
    fn analyze_erases_incoming_slice_early_return_join_at_drop() {
        let llbc = incoming_slice_llbc();
        let body = body_of_args(
            1,
            8,
            vec![
                bb(call_fun(1, vec![], 2, 1)),
                bb(call_fun(5, vec![copy_local(1)], 3, 2)),
                bb(call_fun(2, vec![copy_local(3)], 4, 3)),
                bb(json!({"Switch": {
                    "discr": {"Read": place(4)},
                    "targets": {"If": [4, 5]}
                }})),
                bb(json!({"Goto": {"target": 8}})),
                bb(call_fun(3, vec![], 5, 6)),
                bb(call_fun(4, vec![copy_local(4)], 6, 7)),
                bb(call_fun(2, vec![copy_local(5)], 7, 8)),
                bb(drop_local(2, 9)),
                bb(json!("Return")),
            ],
        );
        match analyze(&body, &llbc) {
            Ok(Some(plan)) => {
                assert_eq!(plan.slot_count, 1, "only the scalar pin is a slot");
            }
            Ok(None) => panic!("expected accepted erase, got no bracket"),
            Err(reason) => panic!("expected accepted erase, got {reason}"),
        }
    }

    /// Extra `pin_root` on one arm joining a non-Drop continuation, then
    /// Drop of this body's guard. A join postdominated by that Drop lets
    /// `slot_count` take max over arms.
    #[test]
    fn analyze_erases_extra_pin_join_at_non_drop_then_drop() {
        let llbc = stack_ops_llbc();
        let body = body_of(
            4,
            vec![
                bb(call_fun(1, vec![], 1, 1)),
                bb(call_fun(4, vec![json!({"Read": place(0)})], 2, 2)),
                bb(json!({"Switch": {
                    "discr": {"Read": place(2)},
                    "targets": {"If": [3, 4]}
                }})),
                bb(json!({"Goto": {"target": 6}})),
                bb(call_fun(4, vec![json!({"Read": place(0)})], 3, 5)),
                bb(json!({"Goto": {"target": 6}})),
                bb(json!({"Goto": {"target": 7}})),
                bb(drop_local(1, 8)),
                bb(json!("Return")),
            ],
        );
        match analyze(&body, &llbc) {
            Ok(Some(plan)) => {
                assert_eq!(plan.slot_count, 2, "max over arms, extra pin is slot 1");
            }
            Ok(None) => panic!("expected accepted erase, got no bracket"),
            Err(reason) => panic!("expected accepted erase, got {reason}"),
        }
    }

    /// Pins that accumulate around a loop join with no postdominating Drop
    /// still refuse: the lattice would climb, and there is no rewind of
    /// this body's open guard on every path to an exit.
    #[test]
    fn analyze_refuses_extra_pin_loop_join_without_drop() {
        let llbc = stack_ops_llbc();
        let body = body_of(
            4,
            vec![
                bb(call_fun(1, vec![], 1, 1)),
                bb(call_fun(4, vec![json!({"Read": place(0)})], 2, 2)),
                bb(json!({"Switch": {
                    "discr": {"Read": place(2)},
                    "targets": {"If": [3, 4]}
                }})),
                bb(call_fun(4, vec![json!({"Read": place(0)})], 3, 2)),
                bb(json!("Return")),
            ],
        );
        match analyze(&body, &llbc) {
            Err("depth-disagrees-at-merge") => {}
            Ok(Some(_)) => panic!("expected depth-disagrees-at-merge, got accepted"),
            Ok(None) => panic!("expected depth-disagrees-at-merge, got no bracket"),
            Err(reason) => panic!("expected depth-disagrees-at-merge, got {reason}"),
        }
    }

    /// Get of the extra pin is rewritten on the writing arm only (SSA
    /// slot locals). The join does not grow a Get of slot 1.
    #[test]
    fn analyze_erases_extra_pin_get_stays_on_writing_arm() {
        let llbc = stack_ops_llbc();
        let n_locals: u64 = 7;
        let body = body_of(
            n_locals,
            vec![
                bb(call_fun(1, vec![], 1, 1)),
                bb(call_fun(4, vec![json!({"Read": place(0)})], 2, 2)),
                bb(json!({"Switch": {
                    "discr": {"Read": place(2)},
                    "targets": {"If": [3, 4]}
                }})),
                bb(json!({"Goto": {"target": 8}})),
                bb(call_fun(3, vec![], 4, 5)),
                bb(call_fun(4, vec![json!({"Read": place(0)})], 5, 6)),
                bb(call_fun(2, vec![copy_local(4)], 6, 7)),
                bb(json!({"Goto": {"target": 8}})),
                bb(json!({"Goto": {"target": 9}})),
                bb(drop_local(1, 10)),
                bb(json!("Return")),
            ],
        );
        let plan = match analyze(&body, &llbc) {
            Ok(Some(plan)) => plan,
            Ok(None) => panic!("expected accepted erase, got no bracket"),
            Err(reason) => panic!("expected accepted erase, got {reason}"),
        };
        assert_eq!(plan.slot_count, 2, "max over arms");
        let slot1 = local_place_json(n_locals as usize + 1, &ty());
        let rewrite_get = plan.terms.get(&6).expect("Get of extra pin rewritten");
        assert_eq!(
            rewrite_get.stmts,
            vec![json!({"Assign": [
                place(6),
                {"Use": [{"Read": slot1}, "Yes"]}
            ]})],
            "Get of the extra pin is Read of slot 1 on the writing arm"
        );
        assert!(
            plan.terms.get(&3).is_none(),
            "the no-extra-pin arm has no Get of slot 1"
        );
        assert!(
            plan.terms.get(&8).is_none(),
            "the join does not Get the extra pin"
        );
    }

    /// Two-phase walk: the short arm reaches the join first. After the join
    /// a Len, a pin, and a Get of that Len in a block also reached from the
    /// extra-pin arm already at `hi`. Emit walks at the Probe fixpoint: the
    /// Get copies the pin after the join (`slot == hi`).
    #[test]
    fn analyze_rewrites_len_after_disagreeing_join_at_final_depth() {
        let llbc = stack_ops_llbc();
        let n_locals: u64 = 7;
        let body = body_of(
            n_locals,
            vec![
                bb(call_fun(1, vec![], 1, 1)),
                bb(call_fun(4, vec![json!({"Read": place(0)})], 2, 2)),
                bb(json!({"Switch": {
                    "discr": {"Read": place(2)},
                    "targets": {"If": [3, 4]}
                }})),
                bb(json!({"Goto": {"target": 6}})),
                bb(call_fun(4, vec![json!({"Read": place(0)})], 3, 5)),
                bb(json!({"Switch": {
                    "discr": {"Read": place(3)},
                    "targets": {"If": [6, 8]}
                }})),
                bb(call_fun(3, vec![], 4, 7)),
                bb(call_fun(4, vec![json!({"Read": place(0)})], 5, 8)),
                bb(call_fun(2, vec![copy_local(4)], 6, 9)),
                bb(drop_local(1, 10)),
                bb(json!("Return")),
            ],
        );
        let plan = match analyze(&body, &llbc) {
            Ok(Some(plan)) => plan,
            Ok(None) => panic!("expected accepted erase, got no bracket"),
            Err(reason) => panic!("expected accepted erase, got {reason}"),
        };
        assert_eq!(plan.slot_count, 3, "obj + extra pin + pin after join");
        let hi = 2usize;
        let slot_hi = local_place_json(n_locals as usize + hi, &ty());
        let rewrite_get = plan.terms.get(&8).expect("Get of Len index rewritten");
        assert_eq!(
            rewrite_get.stmts,
            vec![json!({"Assign": [
                place(6),
                {"Use": [{"Read": slot_hi}, "Yes"]}
            ]})],
            "Get of the Len index copies the pin after the join (slot hi)"
        );
    }

    /// After a disagreeing join, `guard.get(1)` is an absolute index in
    /// the `[lo, hi)` band and refuses `absolute-get-above-join-floor`.
    #[test]
    fn analyze_refuses_absolute_get_in_join_band() {
        let llbc = stack_ops_llbc();
        let body = body_of(
            6,
            vec![
                bb(call_fun(1, vec![], 1, 1)),
                bb(call_fun(4, vec![json!({"Read": place(0)})], 2, 2)),
                bb(json!({"Switch": {
                    "discr": {"Read": place(2)},
                    "targets": {"If": [3, 4]}
                }})),
                bb(json!({"Goto": {"target": 6}})),
                bb(call_fun(4, vec![json!({"Read": place(0)})], 3, 5)),
                bb(json!({"Goto": {"target": 6}})),
                bb(call_fun(4, vec![json!({"Read": place(0)})], 4, 7)),
                bb(call_fun(2, vec![usize_lit(1)], 5, 8)),
                bb(drop_local(1, 9)),
                bb(json!("Return")),
            ],
        );
        match analyze(&body, &llbc) {
            Err("absolute-get-above-join-floor") => {}
            Ok(Some(_)) => panic!("expected absolute-get-above-join-floor, got accepted"),
            Ok(None) => panic!("expected absolute-get-above-join-floor, got no bracket"),
            Err(reason) => panic!("expected absolute-get-above-join-floor, got {reason}"),
        }
    }

    /// Same CFG as [`analyze_refuses_absolute_get_in_join_band`], slot 0
    /// sits below `lo`.
    #[test]
    fn analyze_accepts_absolute_get_below_join_floor() {
        let llbc = stack_ops_llbc();
        let n_locals: u64 = 6;
        let body = body_of(
            n_locals,
            vec![
                bb(call_fun(1, vec![], 1, 1)),
                bb(call_fun(4, vec![json!({"Read": place(0)})], 2, 2)),
                bb(json!({"Switch": {
                    "discr": {"Read": place(2)},
                    "targets": {"If": [3, 4]}
                }})),
                bb(json!({"Goto": {"target": 6}})),
                bb(call_fun(4, vec![json!({"Read": place(0)})], 3, 5)),
                bb(json!({"Goto": {"target": 6}})),
                bb(call_fun(4, vec![json!({"Read": place(0)})], 4, 7)),
                bb(call_fun(2, vec![usize_lit(0)], 5, 8)),
                bb(drop_local(1, 9)),
                bb(json!("Return")),
            ],
        );
        let plan = match analyze(&body, &llbc) {
            Ok(Some(plan)) => plan,
            Ok(None) => panic!("expected accepted erase, got no bracket"),
            Err(reason) => panic!("expected accepted erase, got {reason}"),
        };
        let slot0 = local_place_json(n_locals as usize, &ty());
        let rewrite_get = plan.terms.get(&7).expect("Get of slot 0 rewritten");
        assert_eq!(
            rewrite_get.stmts,
            vec![json!({"Assign": [
                place(5),
                {"Use": [{"Read": slot0}, "Yes"]}
            ]})],
            "absolute get below the join floor copies slot 0"
        );
    }

    /// `ll_set_nonnull`: one arm binds the shared index local as a
    /// `pin_roots` dest (`relative: false`), the other as a `Len`
    /// (`relative: true`) then `pin_root`. Same `k` is a compatible
    /// rebind; absolute wins and the Get sits below the join floor.
    #[test]
    fn analyze_erases_pin_roots_and_len_bound_to_the_same_index_local() {
        let llbc = stack_ops_llbc();
        let array = json!({"Aggregate": [
            {"Array": [ty(), {"Deduplicated": 1}, null]},
            [{"Move": place(0)}]
        ]});
        let body = body_of(
            6,
            vec![
                bb(call_fun(1, vec![], 1, 1)),
                bb(json!({"Switch": {
                    "discr": {"Read": place(0)},
                    "targets": {"If": [2, 3]}
                }})),
                bb_stmts(
                    vec![assign_local(3, array)],
                    call_fun(8, vec![json!({"Move": place(3)})], 2, 4),
                ),
                bb(call_fun(3, vec![], 2, 7)),
                bb(call_fun(2, vec![copy_local(2)], 4, 5)),
                bb(drop_local(1, 6)),
                bb(json!("Return")),
                bb(call_fun(4, vec![json!({"Read": place(0)})], 5, 4)),
            ],
        );
        match analyze(&body, &llbc) {
            Ok(Some(_)) => {}
            Ok(None) => panic!("expected accepted erase, got no bracket"),
            Err(reason) => panic!("expected accepted erase, got {reason}"),
        }
    }

    /// Two arms bind the same index local to different `k`. That is
    /// `special-rebound`: a join Get would read one slot on both arms.
    #[test]
    fn analyze_refuses_dual_def_with_different_slot_index() {
        let llbc = stack_ops_llbc();
        let body = body_of(
            6,
            vec![
                bb(call_fun(1, vec![], 1, 1)),
                bb(json!({"Switch": {
                    "discr": {"Read": place(0)},
                    "targets": {"If": [2, 3]}
                }})),
                bb(call_fun(3, vec![], 2, 4)),
                bb(call_fun(4, vec![json!({"Read": place(0)})], 5, 7)),
                bb(call_fun(2, vec![copy_local(2)], 4, 5)),
                bb(drop_local(1, 6)),
                bb(json!("Return")),
                bb(call_fun(3, vec![], 2, 4)),
            ],
        );
        match analyze(&body, &llbc) {
            Err("special-rebound") => {}
            Ok(Some(_)) => panic!("expected special-rebound, got accepted"),
            Ok(None) => panic!("expected special-rebound, got no bracket"),
            Err(reason) => panic!("expected special-rebound, got {reason}"),
        }
    }

    /// `build_instance`: extra-pin join, then an inner Open/Drop. Probe
    /// first visits the inner Drop at the short-arm depth; the high arm
    /// reaches it later. Joining the rewind entry lets Emit walk it at
    /// the fixpoint with the final Guard specials.
    #[test]
    fn analyze_erases_inner_rewind_after_extra_pin_join() {
        let llbc = stack_ops_llbc();
        let body = body_of(
            4,
            vec![
                bb(call_fun(1, vec![], 1, 1)),
                bb(call_fun(4, vec![json!({"Read": place(0)})], 2, 2)),
                bb(json!({"Switch": {
                    "discr": {"Read": place(2)},
                    "targets": {"If": [3, 4]}
                }})),
                bb(json!({"Goto": {"target": 5}})),
                bb(call_fun(4, vec![json!({"Read": place(0)})], 2, 5)),
                bb(call_fun(1, vec![], 3, 6)),
                bb(drop_local(3, 7)),
                bb(drop_local(1, 8)),
                bb(json!("Return")),
            ],
        );
        match analyze(&body, &llbc) {
            Ok(Some(plan)) => {
                assert_eq!(plan.slot_count, 2, "max over extra-pin arms");
            }
            Ok(None) => panic!("expected accepted erase, got no bracket"),
            Err(reason) => panic!("expected accepted erase, got {reason}"),
        }
    }

    /// `w_module_dict_setitem_str_internal` erases: `DictOperationGuard::new`
    /// is `ReturnsIndex`, so the module-strategy `with_roots!` is not residual
    /// `push_roots`.
    #[test]
    fn w_module_dict_setitem_str_internal_bracket_is_scalar_replaced() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../target/llbc/pyre-object.ullbc"
        );
        if !std::path::Path::new(path).is_file() {
            return;
        }
        let llbc = Llbc::load(path).expect("load pyre-object llbc");
        ensure_stack_sensitive_fns(&llbc);
        let fd = llbc
            .iter_local_fns()
            .find(|fd| {
                fd.item_meta
                    .name_path()
                    .ends_with("::w_module_dict_setitem_str_internal")
            })
            .expect("w_module_dict_setitem_str_internal present");
        let body = fd.unstructured().expect("body");
        let erased = match erase_shadow_stack(fd, &body, &llbc) {
            Ok(Some(erased)) => erased,
            Ok(None) => panic!("expected a bracket to erase"),
            Err(reason) => panic!("expected accepted erase, got {reason}"),
        };
        for block in &erased.body {
            let Ok(TermKind::Call { call, .. }) = block.term_ref(&llbc) else {
                continue;
            };
            if callee_path(call, &llbc)
                .as_deref()
                .is_some_and(|p| p.rsplit("::").next() == Some("push_roots"))
            {
                panic!("push_roots survived erase of w_module_dict_setitem_str_internal");
            }
        }
    }
}
