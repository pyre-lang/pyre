//! Port of `rpython/translator/backendopt/storesink.py`.
//!
//! Removes superfluous `getfield` and `cast_pointer` operations using
//! a super-local heap-cache: each block inherits its predecessor's
//! cache when it has exactly one predecessor; merge blocks (and the
//! startblock) start with an empty cache. `setfield` invalidates
//! cache entries for the same `(concretetype, fieldname)` pair, and
//! any side-effecting op clears the cache entirely.

use std::collections::HashMap;

use crate::flowspace::model::{
    BlockKey, BlockRef, ConcretetypePlaceholder, ConstValue, Constant, FunctionGraph, Hlvalue,
    LinkRef, SpaceOperation, Variable, mkentrymap,
};
use crate::translator::backendopt::removenoops;
use crate::translator::rtyper::lltypesystem::lloperation::ll_operations;
use crate::translator::rtyper::lltypesystem::lltype::{
    LowLevelType, LowLevelValue, PtrTarget, cast_pointer,
};
use crate::translator::simplify;

/// RPython `OK_OPS` (`storesink.py`). Allow-listed names that
/// upstream's `has_side_effects` reports as side-effect-free even
/// though `LLOp.sideeffects` is True.
const OK_OPS: &[&str] = &[
    "debug_assert",
    "debug_assert_not_none",
    "jit_force_virtualizable",
];

/// RPython `has_side_effects(op)` (`storesink.py`).
fn has_side_effects(opname: &str) -> bool {
    if OK_OPS.contains(&opname) {
        return false;
    }
    match ll_operations().get(opname) {
        Some(op) => op.sideeffects,
        // Upstream `:14-16 except AttributeError: return True`.
        None => true,
    }
}

/// Cache key for the super-local heap state.
///
/// Upstream uses a Python tuple `(arg0, field_or_type)`. The Rust
/// port enumerates the two shapes explicitly so they can share a
/// single `HashMap<CacheKey, Hlvalue>`:
/// - `Field` keys cache `getfield` / invalidate on `setfield`
///   (`:90-110, :132-136`).
/// - `Cast` keys cache `cast_pointer` (`:111-129`). The result type
///   is the discriminator since two casts to different types over
///   the same pointer have different semantics.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
enum CacheKey {
    Field(Hlvalue, String),
    Cast(Hlvalue, ConcretetypePlaceholder),
}

type Cache = HashMap<CacheKey, Hlvalue>;

/// RPython `storesink_graph(graph)` (`storesink.py`).
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
pub fn storesink_graph(graph: &FunctionGraph) {
    let entrymap = mkentrymap(graph);
    let start_key = BlockKey::of(&graph.startblock);

    // Upstream `:26-28`: merge blocks (>1 incoming) and startblock
    // are starting points; each gets an empty cache.
    let mut todo: Vec<(BlockRef, Option<Cache>, Option<LinkRef>)> = Vec::new();
    for (block_key, prev_links) in &entrymap {
        if prev_links.len() > 1 || block_key == &start_key {
            // Recover the block from any incoming link's target.
            // Every link in `prev_links` targets the same block.
            let block = prev_links
                .first()
                .and_then(|l| l.borrow().target.clone())
                .expect("mkentrymap guarantees synthetic start link or real link target");
            todo.push((block, None, None));
        }
    }

    let mut added_some_same_as = false;
    let mut visited = 0usize;

    while let Some((block, cache_opt, inputlink)) = todo.pop() {
        visited += 1;
        let mut cache = cache_opt.unwrap_or_default();

        // Upstream `:38-40`: only walk blocks that carry operations.
        // `inputlink` is unused once `cache` is established because
        // `_translate_cache` has already remapped the entries.
        let _ = inputlink;
        let has_ops = !block.borrow().operations.is_empty();
        if has_ops && _storesink_block(&block, &mut cache) {
            added_some_same_as = true;
        }

        let exits: Vec<LinkRef> = block.borrow().exits.to_vec();
        for link in exits {
            let target = match link.borrow().target.clone() {
                Some(t) => t,
                None => continue,
            };
            let target_key = BlockKey::of(&target);
            // Upstream `:42-44`: only chain into blocks with one
            // incoming link — merge blocks were already seeded with
            // an empty cache up front.
            if entrymap.get(&target_key).map(Vec::len).unwrap_or(0) == 1 {
                let new_cache = _translate_cache(&cache, &link);
                todo.push((target, Some(new_cache), Some(link)));
            }
        }
    }

    // Upstream `:46 assert visited == len(entrymap)`. Python's
    // `assert` is unconditional (controlled by `python -O`, not by
    // build mode); pyre's parity carrier is `assert!` so the
    // invariant fires on release builds as well.
    assert_eq!(
        visited,
        entrymap.len(),
        "storesink.py:46 visited count mismatches entrymap"
    );

    if added_some_same_as {
        removenoops::remove_same_as(graph);
        simplify::transform_dead_op_vars(graph, None);
    }
}

/// RPython `_translate_cache(cache, link)` (`storesink.py`).
///
/// Maps cache entries through link.args → block.inputargs. If a
/// cached variable is not yet plumbed across the edge, append a
/// fresh inputarg + a corresponding link arg so the cache survives
/// the rename. `local_versions` upstream is the per-call mapping;
/// we materialise it as `HashMap<Variable, Variable>`.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn _translate_cache(cache: &Cache, link: &LinkRef) -> Cache {
    let target = match link.borrow().target.clone() {
        Some(t) => t,
        None => return Cache::new(),
    };
    // Upstream `:52-53 if link.target.operations == (): # exit or
    // except block`. `operations == ()` is the upstream marker for
    // a final block — set only by `mark_final()` flipping
    // `Block.operations` from a list to the empty tuple. Plain
    // empty-`Vec` regular blocks must keep their cache.
    if target.borrow().is_final_block() {
        return Cache::new();
    }

    // upstream `:55 local_versions = {var1: var2 for var1, var2 in
    // zip(link.args, block.inputargs)}` — pre-existing
    // link-arg → inputarg mapping (Variables only).
    let mut local_versions: HashMap<Variable, Variable> = HashMap::new();
    {
        let l = link.borrow();
        let t = target.borrow();
        for (la, ia) in l.args.iter().zip(t.inputargs.iter()) {
            if let (Some(Hlvalue::Variable(lv)), Hlvalue::Variable(iv)) = (la.as_ref(), ia) {
                local_versions.insert(lv.clone(), iv.clone());
            }
        }
    }

    let mut new_cache: Cache = Cache::new();

    // Closure mirroring upstream `:56-67 _translate_arg`. Captures
    // `local_versions`, `link`, `target` by mutable reference.
    let translate_arg = |arg: &Hlvalue,
                         local_versions: &mut HashMap<Variable, Variable>,
                         link: &LinkRef,
                         target: &BlockRef|
     -> Hlvalue {
        match arg {
            Hlvalue::Variable(v) => {
                if let Some(rep) = local_versions.get(v).cloned() {
                    return Hlvalue::Variable(rep);
                }
                // upstream `:59-64`: append a fresh inputarg, plumb
                // it through the link's args.
                let fresh = Variable::new();
                if let Some(ct) = v.concretetype() {
                    fresh.set_concretetype(Some(ct));
                }
                link.borrow_mut().args.push(Some(arg.clone()));
                target
                    .borrow_mut()
                    .inputargs
                    .push(Hlvalue::Variable(fresh.clone()));
                local_versions.insert(v.clone(), fresh.clone());
                Hlvalue::Variable(fresh)
            }
            // upstream `:65 return arg` — Constants pass through.
            Hlvalue::Constant(_) => arg.clone(),
        }
    };

    for (key, res) in cache {
        // Upstream `:70`: only carry entries whose anchor variable
        // is still live across the edge (or which is anchored on a
        // Constant — Constants are always live).
        let key_anchor = match key {
            CacheKey::Field(anchor, _) | CacheKey::Cast(anchor, _) => anchor,
        };
        let anchor_alive = match key_anchor {
            Hlvalue::Variable(v) => local_versions.contains_key(v),
            Hlvalue::Constant(_) => true,
        };
        if !anchor_alive {
            continue;
        }
        let new_anchor = translate_arg(key_anchor, &mut local_versions, link, &target);
        let new_res = translate_arg(res, &mut local_versions, link, &target);
        let new_key = match key {
            CacheKey::Field(_, fname) => CacheKey::Field(new_anchor, fname.clone()),
            CacheKey::Cast(_, ct) => CacheKey::Cast(new_anchor, ct.clone()),
        };
        new_cache.insert(new_key, new_res);
    }
    new_cache
}

/// RPython `_storesink_block(block, cache, inputlink)` (`storesink.py`).
///
/// Returns `true` when the block grew at least one `same_as` op
/// (cache hit replacing a getfield / cast_pointer).
///
/// TODO: the upstream constant-folding sub-paths at `:93-103`
/// (immutable-field read on `Constant(_ptr)`) and `:113-122`
/// (`lltype.cast_pointer` on a Constant pointer) are intentionally
/// not lifted here. They depend on `lltype._ptr` field access and
/// `lltype.cast_pointer`'s `_parentable` chain — both of which are
/// TODOs in `lltype.rs`.
/// Upstream itself notes at `:118-119` that "constfold also handles
/// the case", so the cache-side port is complete on its own:
/// constfold catches the constant pointer reads, storesink catches
/// the redundant-load elimination across SSA pointers.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn _storesink_block(block: &BlockRef, cache: &mut Cache) -> bool {
    let mut replacements: HashMap<Hlvalue, Hlvalue> = HashMap::new();
    let mut added_some_same_as = false;

    // Closure mirroring upstream `:80-83 replace`.
    let do_replace =
        |op: &mut SpaceOperation, res: Hlvalue, replacements: &mut HashMap<Hlvalue, Hlvalue>| {
            op.opname = "same_as".to_string();
            op.args = vec![res.clone()];
            replacements.insert(op.result.clone(), res);
        };

    let get_rep = |arg: &Hlvalue, replacements: &HashMap<Hlvalue, Hlvalue>| -> Hlvalue {
        replacements
            .get(arg)
            .cloned()
            .unwrap_or_else(|| arg.clone())
    };

    let mut b = block.borrow_mut();
    for op in b.operations.iter_mut() {
        match op.opname.as_str() {
            "getfield" => {
                if op.args.len() < 2 {
                    continue;
                }
                let arg0 = get_rep(&op.args[0], &replacements);
                let field_name = match &op.args[1] {
                    Hlvalue::Constant(c) => match c.value.as_text() {
                        Some(s) => s.to_string(),
                        // upstream's `field = op.args[1].value` is a
                        // Python str. The Rust ConstValue carrier may
                        // be ByteStr or UniStr; either way
                        // `as_text()` extracts the field name.
                        // Non-string carriers abort the cache.
                        None => continue,
                    },
                    Hlvalue::Variable(_) => continue,
                };
                // Upstream `:93-103`: reading an immutable field of
                // a non-null constant pointer folds to the field
                // value directly.
                if let Hlvalue::Constant(c) = &arg0
                    && let Some(folded) = fold_constant_getfield(c, &field_name)
                {
                    do_replace(op, folded, &mut replacements);
                    added_some_same_as = true;
                    continue;
                }
                // Upstream `:104-110`: cache the (anchor, field) ->
                // result pair for later getfield calls.
                let key = CacheKey::Field(arg0, field_name);
                if let Some(res) = cache.get(&key).cloned() {
                    do_replace(op, res, &mut replacements);
                    added_some_same_as = true;
                } else {
                    cache.insert(key, op.result.clone());
                }
            }
            "cast_pointer" => {
                if op.args.is_empty() {
                    continue;
                }
                let arg0 = get_rep(&op.args[0], &replacements);
                let result_ct = match concretetype_of(&op.result) {
                    Some(ct) => ct,
                    None => continue,
                };
                // Upstream `:113-122`: a constant cast_pointer folds
                // to `Constant(cast_pointer(target_type, src_ptr))`.
                // `lltype.cast_pointer` may raise `RuntimeError` for
                // an invalid cast on unreachable code (mirrors
                // upstream `:117-119 except RuntimeError: pass`); we
                // treat the Err arm the same way and fall through to
                // the cache path.
                if let Hlvalue::Constant(c) = &arg0
                    && let Some(folded) = fold_constant_cast_pointer(c, &result_ct)
                {
                    do_replace(op, folded, &mut replacements);
                    added_some_same_as = true;
                    continue;
                }
                let key = CacheKey::Cast(arg0, result_ct);
                if let Some(res) = cache.get(&key).cloned() {
                    do_replace(op, res, &mut replacements);
                    added_some_same_as = true;
                } else {
                    cache.insert(key, op.result.clone());
                }
            }
            "setarrayitem" | "setinteriorfield" | "malloc" | "malloc_varsize" => {
                // Upstream `:130-131 pass`: leave cache intact.
            }
            "setfield" => {
                if op.args.len() < 3 {
                    continue;
                }
                let target = get_rep(&op.args[0], &replacements);
                let field_name = match &op.args[1] {
                    Hlvalue::Constant(c) => match c.value.as_text() {
                        Some(s) => s.to_string(),
                        None => continue,
                    },
                    Hlvalue::Variable(_) => continue,
                };
                let target_ct = concretetype_of(&target);
                clear_cache_for(cache, target_ct.as_ref(), &field_name);
                cache.insert(CacheKey::Field(target, field_name), op.args[2].clone());
            }
            other if has_side_effects(other) => {
                cache.clear();
            }
            _ => {}
        }
    }
    added_some_same_as
}

/// Mirror of upstream's nested `clear_cache_for(cache, concretetype,
/// fieldname)` (`storesink.py`). Drops any cache entry whose
/// anchor has the same concrete pointer-type AND the same field
/// name, since a `setfield` on `concretetype.field` may have
/// invalidated it.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn clear_cache_for(
    cache: &mut Cache,
    concretetype: Option<&ConcretetypePlaceholder>,
    fieldname: &str,
) {
    cache.retain(|key, _| match key {
        CacheKey::Field(anchor, fname) => {
            let anchor_ct = concretetype_of(anchor);
            !(anchor_ct.as_ref() == concretetype && fname == fieldname)
        }
        CacheKey::Cast(_, _) => true,
    });
}

/// Mirror of upstream `storesink.py:93-103`. When `arg0` is a non-null
/// `Constant(_ptr)` with a struct type whose `field` is marked
/// immutable, fold the `getfield` to a `Constant(field_value,
/// field_concretetype)`. Returns `None` (and falls through to the
/// cache path) when any of the upstream guards fails:
///
/// * `arg0.value` is not an `LLPtr` (e.g. void, primitive constant)
/// * `arg0.concretetype.TO` is not a `Struct`
/// * the struct type does not declare `field` immutable
/// * the pointer is null (delayed pointers also fail this check via
///   `_obj` returning `Err(DelayedPointer)`)
/// * the field's `LowLevelValue` cannot be encoded as a `ConstValue`
///   (e.g. nested struct/array — those need a different carrier)
///
/// Upstream guard `:97 not isinstance(arg0.value._obj, int)` excludes
/// tagged-int pointers. Pyre's [`_ptr_obj`] enum at
/// `lltype.rs` enumerates `Func | Struct | Array | Opaque`
/// only — there is no `Int` variant, so a `_ptr` whose `_obj` is an
/// `int` is structurally inhabit-impossible. The upstream guard is
/// therefore satisfied by Rust's exhaustive match; no runtime check
/// needed.
fn fold_constant_getfield(c: &Constant, field: &str) -> Option<Hlvalue> {
    let ConstValue::LLPtr(ptr) = &c.value else {
        return None;
    };
    if !ptr.nonzero() {
        // Upstream `:96 arg0.value` — exclude null pointers.
        return None;
    }
    let concretetype = c.concretetype.as_ref()?;
    let LowLevelType::Ptr(ptrtype) = concretetype else {
        return None;
    };
    let PtrTarget::Struct(struct_type) = &ptrtype.TO else {
        return None;
    };
    if !struct_type._immutable_field(field) {
        return None;
    }
    // Upstream `:100-101`: read the LL field and its concretetype.
    // The tagged-int guard at upstream `:97` is structurally
    // satisfied by `_ptr_obj`'s closed enum — see the docstring.
    let llres = ptr.getattr(field).ok()?;
    let field_concretetype = struct_type.getattr_field_type(field)?;
    let const_value = lowlevel_value_to_const_value(&llres)?;
    let mut folded = Constant::new(const_value);
    folded.concretetype = Some(field_concretetype);
    Some(Hlvalue::Constant(folded))
}

/// Mirror of upstream `storesink.py:113-122`. When `arg0` is a
/// `Constant(_ptr)` and `lltype.cast_pointer` accepts the cast,
/// fold the `cast_pointer` to a `Constant(casted_ptr,
/// op.result.concretetype)`. Upstream's `RuntimeError` arm — invalid
/// casts in unreachable code — surfaces here as `Err(_)` from
/// [`cast_pointer`]; we silently drop the fold and fall through to
/// the cache path, matching `:117-119 pass`.
fn fold_constant_cast_pointer(
    c: &Constant,
    result_ct: &ConcretetypePlaceholder,
) -> Option<Hlvalue> {
    let ConstValue::LLPtr(ptr) = &c.value else {
        return None;
    };
    let LowLevelType::Ptr(target_ptr_type) = result_ct else {
        return None;
    };
    let casted = match cast_pointer(target_ptr_type, ptr) {
        Ok(p) => p,
        Err(_) => return None,
    };
    let mut folded = Constant::new(ConstValue::LLPtr(Box::new(casted)));
    folded.concretetype = Some(result_ct.clone());
    Some(Hlvalue::Constant(folded))
}

/// Map an `lltype.LowLevelValue` to the carrier `ConstValue` so a
/// folded `getfield` can land in a `Constant`. Mirrors upstream
/// `storesink.py res = Constant(llres, concretetype)` —
/// `Constant.value` carries the LL field value directly. The Rust
/// port's `ConstValue` variants have to map each `LowLevelValue`
/// arm to the carrier the rest of the optimizer uses for that
/// concretetype:
///
/// * `Signed` / `Unsigned` → `Int(i64)` (concretetype distinguishes
///   the lltype, e.g. `LowLevelType::Unsigned` keeps the
///   `Constant.concretetype` slot precise).
/// * `Bool` → `Bool`.
/// * `Float` / `LongFloat` → `Float(bits)` (already an `f64::to_bits`
///   carrier — no conversion needed).
/// * `SingleFloat` → `Float(f64::to_bits(f32 as f64))`. Upstream
///   stores the raw f32 numeric value; pyre's `ConstValue::Float`
///   is the only float carrier and stores f64 bits, so promote.
/// * `Char(c)` → `ByteStr(vec![c as u8])`. Upstream `_ptr.field`
///   for a `Char`-typed slot returns a Python str of length 1; the
///   Rust port carries Char-typed Constants as `ByteStr` (see
///   `opimpl.rs::op_cast_int_to_char` and `rstr.rs` setitem).
/// * `UniChar(c)` → `UniStr(c.to_string())`. UniChar-typed
///   Constants are `UniStr` everywhere else in the port.
/// * `Ptr` / `Address` → corresponding `LL*` carriers.
///
/// Returns `None` for `Void` / `Struct` / `Array` / `Opaque` /
/// `InteriorPtr` — those container values have no flat `ConstValue`
/// variant today, so the fold-to-Constant fast path is skipped and
/// the cache path takes over. This is a TODO
/// versus upstream `storesink.py:100-102`, which lifts every
/// `getattr(arg0.value, field)` into a Constant unconditionally.
/// Convergence path: extend `ConstValue` with `LLStruct(_struct)` /
/// `LLArray(_array)` / `LLOpaque(_opaque)` / `LLInteriorPtr(...)`
/// and route them through the rest of the optimizer.
fn lowlevel_value_to_const_value(v: &LowLevelValue) -> Option<ConstValue> {
    match v {
        LowLevelValue::Signed(n) => Some(ConstValue::Int(*n)),
        LowLevelValue::Unsigned(u) => Some(ConstValue::UInt(*u)),
        LowLevelValue::SignedLongLongLong(n) => Some(ConstValue::Int128(*n)),
        LowLevelValue::UnsignedLongLongLong(u) => Some(ConstValue::UInt128(*u)),
        LowLevelValue::Bool(b) => Some(ConstValue::Bool(*b)),
        LowLevelValue::Float(bits) => Some(ConstValue::Float(*bits)),
        LowLevelValue::LongFloat(bits) => Some(ConstValue::LongFloat(*bits)),
        LowLevelValue::SingleFloat(bits) => Some(ConstValue::SingleFloat(*bits)),
        LowLevelValue::Char(c) => {
            // Upstream `_ptr.field` for a Char-typed slot is a
            // Python str of length 1. The Rust port carries
            // Char-typed Constants as `ByteStr(vec![byte])` (see
            // `opimpl.rs`'s `op_cast_int_to_char`).
            Some(ConstValue::ByteStr(vec![*c as u8]))
        }
        LowLevelValue::UniChar(c) => Some(ConstValue::uni_str(c.to_string())),
        LowLevelValue::Ptr(p) => Some(ConstValue::LLPtr(p.clone())),
        LowLevelValue::Address(a) => Some(ConstValue::LLAddress(a.clone())),
        // Void / Struct / Array / Opaque / InteriorPtr have no flat
        // ConstValue equivalent today — defer (TODO: documented
        // above).
        _ => None,
    }
}

fn concretetype_of(value: &Hlvalue) -> Option<ConcretetypePlaceholder> {
    match value {
        Hlvalue::Variable(v) => v.concretetype(),
        Hlvalue::Constant(c) => c.concretetype.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::{Block, BlockRefExt, Constant, Hlvalue, Link, SpaceOperation};

    fn const_field(name: &str) -> Hlvalue {
        Hlvalue::Constant(Constant::new(ConstValue::byte_str(name)))
    }

    #[test]
    fn unichar_field_value_equals_equivalent_unistr() {
        let v = lowlevel_value_to_const_value(&LowLevelValue::UniChar('A')).expect("UniChar");
        assert_eq!(v, ConstValue::uni_str("A"));
        assert_eq!(v, ConstValue::UniChar(u32::from('A')));
    }

    #[test]
    fn redundant_getfield_after_first_load_collapses_to_same_as() {
        // Build:
        //   block0(p):
        //     v1 = getfield(p, "x")
        //     v2 = getfield(p, "x")
        //     return v1 + v2     # only the structural shape matters
        let p = Variable::named("p");
        let v1 = Variable::named("v1");
        let v2 = Variable::named("v2");
        let block = Block::shared(vec![Hlvalue::Variable(p.clone())]);
        block.borrow_mut().operations.push(SpaceOperation::new(
            "getfield",
            vec![Hlvalue::Variable(p.clone()), const_field("x")],
            Hlvalue::Variable(v1.clone()),
        ));
        block.borrow_mut().operations.push(SpaceOperation::new(
            "getfield",
            vec![Hlvalue::Variable(p), const_field("x")],
            Hlvalue::Variable(v2.clone()),
        ));
        let graph = FunctionGraph::new("entry", block.clone());
        let return_target = graph.returnblock.clone();
        block.closeblock(vec![
            Link::new(vec![Hlvalue::Variable(v2)], Some(return_target), None).into_ref(),
        ]);

        storesink_graph(&graph);

        let b = block.borrow();
        // After storesink: the second getfield is rewritten to
        // same_as(v1) — but `remove_same_as` then drops it from the
        // op list, forwarding v2 → v1 in the link arg.
        assert_eq!(
            b.operations.len(),
            1,
            "second getfield should fold into same_as and be removed"
        );
        assert_eq!(b.operations[0].opname, "getfield");

        // The terminating link's arg must reference v1 (the first
        // getfield's result), not v2.
        let l = b.exits[0].borrow();
        assert!(matches!(
            &l.args[0],
            Some(Hlvalue::Variable(v)) if v.id() == v1.id()
        ));
    }

    #[test]
    fn setfield_invalidates_matching_cache_entry() {
        // Build:
        //   block0(p, val):
        //     v1 = getfield(p, "x")
        //     setfield(p, "x", val)
        //     v2 = getfield(p, "x")
        //     return v2
        // After storesink, v2 must NOT collapse to v1 (cache cleared
        // by setfield), but it should fold to `val` (cache populated
        // by the setfield itself per :136).
        let p = Variable::named("p");
        let val = Variable::named("val");
        let v1 = Variable::named("v1");
        let v2 = Variable::named("v2");
        let block = Block::shared(vec![
            Hlvalue::Variable(p.clone()),
            Hlvalue::Variable(val.clone()),
        ]);
        block.borrow_mut().operations.push(SpaceOperation::new(
            "getfield",
            vec![Hlvalue::Variable(p.clone()), const_field("x")],
            Hlvalue::Variable(v1.clone()),
        ));
        block.borrow_mut().operations.push(SpaceOperation::new(
            "setfield",
            vec![
                Hlvalue::Variable(p.clone()),
                const_field("x"),
                Hlvalue::Variable(val.clone()),
            ],
            Hlvalue::Variable(Variable::named("setres")),
        ));
        block.borrow_mut().operations.push(SpaceOperation::new(
            "getfield",
            vec![Hlvalue::Variable(p), const_field("x")],
            Hlvalue::Variable(v2.clone()),
        ));
        let graph = FunctionGraph::new("entry", block.clone());
        let return_target = graph.returnblock.clone();
        block.closeblock(vec![
            Link::new(vec![Hlvalue::Variable(v2)], Some(return_target), None).into_ref(),
        ]);

        storesink_graph(&graph);

        let b = block.borrow();
        // After storesink + remove_same_as + transform_dead_op_vars:
        //   - second getfield → same_as(val) → removed
        //   - link arg becomes `val` directly, so v1's first
        //     getfield becomes dead and is dropped by
        //     transform_dead_op_vars
        // Only the side-effecting setfield remains.
        let opnames: Vec<&str> = b.operations.iter().map(|o| o.opname.as_str()).collect();
        assert_eq!(opnames, vec!["setfield"]);
        let l = b.exits[0].borrow();
        assert!(matches!(
            &l.args[0],
            Some(Hlvalue::Variable(v)) if v.id() == val.id()
        ));
        let _ = v1;
    }

    #[test]
    fn side_effecting_op_clears_the_cache() {
        // Build:
        //   block0(p):
        //     v1 = getfield(p, "x")
        //     debug_print()       # canfold=false → side effect
        //     v2 = getfield(p, "x")
        //     return v2
        // The cache is cleared by debug_print, so v2 stays as a
        // genuine getfield (no fold).
        let p = Variable::named("p");
        let v1 = Variable::named("v1");
        let v2 = Variable::named("v2");
        let block = Block::shared(vec![Hlvalue::Variable(p.clone())]);
        block.borrow_mut().operations.push(SpaceOperation::new(
            "getfield",
            vec![Hlvalue::Variable(p.clone()), const_field("x")],
            Hlvalue::Variable(v1),
        ));
        block.borrow_mut().operations.push(SpaceOperation::new(
            "debug_print",
            vec![],
            Hlvalue::Variable(Variable::named("dbg")),
        ));
        block.borrow_mut().operations.push(SpaceOperation::new(
            "getfield",
            vec![Hlvalue::Variable(p), const_field("x")],
            Hlvalue::Variable(v2.clone()),
        ));
        let graph = FunctionGraph::new("entry", block.clone());
        let return_target = graph.returnblock.clone();
        // FunctionGraph::new builds a returnblock with one inputarg;
        // the closing link must supply one arg to match.
        block.closeblock(vec![
            Link::new(vec![Hlvalue::Variable(v2)], Some(return_target), None).into_ref(),
        ]);

        storesink_graph(&graph);

        let b = block.borrow();
        let opnames: Vec<&str> = b.operations.iter().map(|o| o.opname.as_str()).collect();
        // Both getfields survive — debug_print cleared the cache
        // before the second one.
        assert_eq!(opnames, vec!["getfield", "debug_print", "getfield"]);
    }

    #[test]
    fn fold_constant_getfield_reads_immutable_struct_field_via_constant() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            _ptr_obj, MallocFlavor, Ptr, Struct, malloc,
        };

        // GcStruct `S` with a single immutable Signed field `x` —
        // upstream's `_hints={'immutable': True}` carrier marks the
        // whole struct as immutable.
        let s = Struct::gc_with_hints(
            "S",
            vec![("x".into(), LowLevelType::Signed)],
            vec![("immutable".into(), ConstValue::Bool(true))],
        );
        let s_type = LowLevelType::Struct(Box::new(s));
        let ptr = malloc(s_type.clone(), None, MallocFlavor::Gc, false).unwrap();
        // Populate `x = 42` via the runtime container.
        let _ptr_obj::Struct(obj) = ptr._obj().expect("ptr is non-null") else {
            panic!("malloc returned non-Struct _ptr_obj");
        };
        obj._setattr("x", LowLevelValue::Signed(42));
        let ptrtype = Ptr::from_container_type(s_type).unwrap();

        let mut c = Constant::new(ConstValue::LLPtr(Box::new(ptr)));
        c.concretetype = Some(LowLevelType::Ptr(Box::new(ptrtype)));

        let folded = fold_constant_getfield(&c, "x").expect("immutable field folds");
        let Hlvalue::Constant(folded_c) = folded else {
            panic!("fold should return Constant Hlvalue");
        };
        assert_eq!(folded_c.value, ConstValue::Int(42));
        assert!(matches!(folded_c.concretetype, Some(LowLevelType::Signed)));
    }

    #[test]
    fn fold_constant_getfield_skips_non_immutable_field() {
        use crate::translator::rtyper::lltypesystem::lltype::{MallocFlavor, Ptr, Struct, malloc};

        // Same struct but without the `immutable` hint — fold must
        // return None so the cache path runs instead.
        let s = Struct::gc("S", vec![("x".into(), LowLevelType::Signed)]);
        let s_type = LowLevelType::Struct(Box::new(s));
        let ptr = malloc(s_type.clone(), None, MallocFlavor::Gc, false).unwrap();
        let ptrtype = Ptr::from_container_type(s_type).unwrap();

        let mut c = Constant::new(ConstValue::LLPtr(Box::new(ptr)));
        c.concretetype = Some(LowLevelType::Ptr(Box::new(ptrtype)));

        assert!(fold_constant_getfield(&c, "x").is_none());
    }

    #[test]
    fn ok_ops_do_not_clear_the_cache() {
        // Same as side_effecting_op_clears_the_cache but with
        // debug_assert (in OK_OPS). The second getfield should
        // collapse.
        let p = Variable::named("p");
        let v1 = Variable::named("v1");
        let v2 = Variable::named("v2");
        let block = Block::shared(vec![Hlvalue::Variable(p.clone())]);
        block.borrow_mut().operations.push(SpaceOperation::new(
            "getfield",
            vec![Hlvalue::Variable(p.clone()), const_field("x")],
            Hlvalue::Variable(v1.clone()),
        ));
        block.borrow_mut().operations.push(SpaceOperation::new(
            "debug_assert",
            vec![],
            Hlvalue::Variable(Variable::named("dbg")),
        ));
        block.borrow_mut().operations.push(SpaceOperation::new(
            "getfield",
            vec![Hlvalue::Variable(p), const_field("x")],
            Hlvalue::Variable(v2),
        ));
        let graph = FunctionGraph::new("entry", block.clone());
        let return_target = graph.returnblock.clone();
        block.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(v1.clone())],
                Some(return_target),
                None,
            )
            .into_ref(),
        ]);

        storesink_graph(&graph);

        let b = block.borrow();
        let opnames: Vec<&str> = b.operations.iter().map(|o| o.opname.as_str()).collect();
        // OK: cache survives debug_assert, second getfield folds out.
        assert_eq!(opnames, vec!["getfield", "debug_assert"]);
    }
}
