//! `rpython/memory/gctransform/framework.py get_livevars_for_roots` —
//! which GC pointers are live across an operation that can collect.
//!
//! Upstream reads the answer straight off the flow graph with
//! `hop.livevars_after_op()`, then brackets the operation with
//! `push_roots` / `pop_roots`. The flowspace transformer now performs that
//! insertion automatically for translated graphs. Native interpreter paths
//! also run as rustc-compiled Rust, so this audit computes the corresponding
//! live set over ULLBC and reports where their source bracket is *missing*.
//!
//! # One deliberate divergence from upstream
//!
//! `get_livevars_for_roots` drops the current operation's own arguments for a
//! moving GC — *"moving GCs don't borrow, so the caller does not need to keep
//! the arguments alive"*.  That holds upstream because the shadowstack walks
//! the caller's frame and rewrites its copy in place.  pyre's shadow stack
//! holds only what was explicitly pinned, so an argument the caller still uses
//! after the call is just as stale as any other local.  Arguments are therefore
//! kept in the live set, and reported in a separate column so the two shapes
//! stay distinguishable.

use std::collections::{HashMap, HashSet, VecDeque};

use majit_charon_reader::ullbc::{
    BasicBlock, CallFunc, CallKind, FunId, Operand, Place, PlaceKind, ProjectionElem, Rvalue,
    StmtKind, SwitchTargets, TermKind, TyRef, TypeDeclKind, Unstructured,
};

/// One call that can collect, with GC pointers live across it and no bracket.
pub struct Finding {
    pub func: u64,
    pub func_name: String,
    /// Source path of [`Finding::line`], read out of the same span, so the
    /// two cannot name different files.  Empty when the artefact carries no
    /// file table.
    pub file: String,
    pub line: u64,
    pub callee_id: u64,
    pub callee_name: String,
    /// Live across the call and *not* passed to it — the shape both proven
    /// defects had.
    pub live_non_arg: Vec<String>,
    /// Live across the call because the call itself takes them.
    pub live_arg: Vec<String>,
    /// Of the live pointers, those the function later hands to a callee whose
    /// name says it addresses a kind whose header a minor collection
    /// relocates.  A stale pointer that is only
    /// stored or returned is a different (and rarer) problem; this column is
    /// where a stale pointer is actually dereferenced as a movable object.
    pub movable_use: Vec<String>,
}

/// What the scan could and could not account for.
///
/// A finding count means nothing without these: a body whose terminator the
/// reader could not parse loses successors, which shrinks every live set
/// computed from it, and a call sitting downstream of a `push_roots` is
/// withheld rather than cleared.
#[derive(Default)]
pub struct ScanStats {
    pub bodies_scanned: usize,
    /// Bodies holding two live root scopes at once somewhere in them.  The
    /// pin set carries an owner, so this on its own no longer withholds a
    /// body; the count stays because a nested body that is unread is unread
    /// for one of the other reasons and the distinction is worth measuring.
    pub bodies_with_nested_scopes: usize,
    /// Calls counted in [`Self::withheld_contents_opaque`] that came from a
    /// body in [`Self::bodies_with_nested_scopes`].
    pub withheld_opaque_from_nested: usize,
    /// Bodies holding a terminator this reader could not classify.  Their
    /// liveness is incomplete, so they are reported rather than counted clean.
    pub unparsed_terminator_bodies: usize,
    /// Bodies holding a statement this reader could not classify.  A statement
    /// that does not parse, and one that parses as `StmtKind::Unknown`, both
    /// contribute no uses, so the live set computed over such a body is a lower
    /// bound and a clean result over it is one too.
    pub unparsed_statement_bodies: usize,
    /// Collecting calls withheld because a `push_roots` dominates them.
    /// Whether that scope is still alive at the call is a drop-placement
    /// question this pass cannot answer, so they are neither reported nor
    /// silently dropped.
    pub withheld_under_a_bracket: usize,
    /// Of [`Self::withheld_under_a_bracket`], those whose bracket pins every
    /// GC pointer live across the call.  A call with nothing live counts here:
    /// any bracket covers an empty set.
    pub withheld_bracket_covers: usize,
    /// Those where a pointer live across the call is not in the bracket's
    /// pinned set.  The bracket exists, so the finding scan withholds the
    /// call -- and the root it would have needed is not in it.  Listed in
    /// [`Self::short_brackets`].
    pub withheld_bracket_short: usize,
    /// Those whose pinned set could not be read: a body holding a statement or
    /// terminator this reader could not parse, a scope-local slot overwrite,
    /// or a pin whose argument does not trace back to a set.  Neither graded
    /// nor claimed clean -- a set read short would accuse correct code.
    pub withheld_contents_opaque: usize,
    /// Of [`Self::withheld_bracket_short`], those missing a root the body
    /// produced itself, which no caller's bracket can be covering.
    pub withheld_bracket_short_body_local: usize,
    /// Of [`Self::withheld_bracket_short`], those missing a root this body goes
    /// on to address as a movable kind.  A caller's pin does not answer for
    /// these, so they do not depend on the intra-procedural limit.
    pub withheld_bracket_short_movable: usize,
    /// The [`ScanStats::withheld_bracket_short`] calls, named.
    pub short_brackets: Vec<ShortBracket>,
    /// Pins whose argument local is still read after the pin normalised the
    /// slot.  Not a missing root -- a stale word.  See [`StalePinRead`].
    pub pin_arg_read_after: usize,
    /// Of those, the ones reading a local this body later addresses as a
    /// movable kind, where the stale word is dereferenced.
    pub pin_arg_read_after_movable: usize,
    /// The [`Self::pin_arg_read_after`] pins, named.
    pub stale_pin_reads: Vec<StalePinRead>,
    /// Bodies that hand at least one GC pointer to a movable-addressing callee.
    ///
    /// The `movable` columns are all filters over this set, so a zero here says
    /// the ranking never had anything to rank -- a different fact from "nothing
    /// ranked".  Worth a counter of its own: the marker list, the id space the
    /// callees are resolved in, and the operand shape a call argument takes each
    /// silence the columns the same way, and only this number separates "the
    /// scan found no such call" from "the scan found them and none was stale".
    pub bodies_with_movable_args: usize,
    /// Locals summed over [`Self::bodies_with_movable_args`], so a body that
    /// contributes one argument is told apart from one that contributes twenty.
    pub movable_arg_locals: usize,
    /// [`Self::withheld_contents_opaque`] split by the first reason the body's
    /// pinned set went unread.
    pub opaque_by_reason: std::collections::BTreeMap<&'static str, usize>,
    /// One entry per body with unread contents: function, file, first
    /// reason, and how many withheld calls it cost.
    pub opaque_bodies: Vec<(String, String, &'static str, usize)>,
}

/// A pin whose argument the body goes on to read.
///
/// `pin_root` returns the word its slot holds *after* the publish, because the
/// publish is a safepoint: a foreign collection can forward the value between
/// the caller's copy and the query, leaving the caller's local pointing at a
/// forwarding stub.  Reading the returned word is the fix; `let _ =` opts out
/// and thereby asserts the kind never moves, which is what
/// [`Self::movable`] checks.  The assertion is worth checking because it has
/// been written down and been wrong: `getitem_tuple` and `getitem_str` each
/// stated in prose that their receiver never moves, and both receivers are
/// nursery-allocated.
pub struct StalePinRead {
    pub func_name: String,
    pub file: String,
    pub line: u64,
    /// Which pin was called.  `with_roots!` expands to `pin_roots` and runs its
    /// body before reading the slots back, so it shows this shape by
    /// construction; a hand-written `let _ = pin_root(x)` does not.
    pub pin_name: String,
    /// The locals handed to the pin and still read afterwards.
    pub locals: Vec<String>,
    /// Of those, the ones this body later addresses as a movable kind,
    /// which is where a stale word is dereferenced rather than merely carried.
    pub movable: Vec<String>,
}

/// A bracketed call whose bracket does not pin everything live across it.
///
/// The complement of a [`Finding`]: there the bracket is absent, here it is
/// present and short.  `postprocess_double_check` asserts the same property
/// upstream after `shadowcolor.py` has run; this reads it off the shipped
/// hand-written brackets, which no pass has ever checked.
pub struct ShortBracket {
    pub func_name: String,
    pub file: String,
    pub line: u64,
    pub callee_name: String,
    /// Live across the call and not pinned by the bracket that dominates it.
    pub missing: Vec<String>,
    /// Of [`Self::missing`], those the body produced itself rather than took as
    /// a parameter.  The scan is intra-procedural, so a missing *parameter* may
    /// be pinned by the caller -- `build_class_inner` says in prose that its
    /// only caller keeps its arguments pinned for the whole call, and no read
    /// of this body can see that.  Nothing outside the body can have pinned a
    /// local it produced, so these are the ones a caller cannot account for.
    pub missing_local: Vec<String>,
    /// Of [`Self::missing`], those this body later hands to a callee whose name
    /// says it addresses a movable kind -- one whose header a
    /// minor collection relocates.  A caller's pin cannot rescue one of these:
    /// the collection rewrites the caller's slot, not this body's copy, so the
    /// word in hand is a corpse either way.  The bracket-contents twin of the
    /// `tier 1.5` column, which the gate holds at zero.
    pub missing_movable: Vec<String>,
    /// What the bracket does pin here, so the two columns can be read
    /// together: an empty one is a scope opened over no roots at all.
    pub pinned: Vec<String>,
}

/// Callee names that say the pointer is addressed as a movable object.
///
/// A stale `PyObjectRef` handed to one of these is dereferenced as a corpse
/// rather than merely stored or returned.  The set is not the whole moving
/// heap: it is the part of it a callee's *name* identifies, and the entries
/// below record which kind each spelling was checked against.
///
/// A new spelling that matches an existing prefix (a `w_list_*` helper)
/// re-ranks an already-unresolved live pointer as tier 1.5; it does not add
/// an unrooted call.
pub const MOVABLE_GC_MARKERS: &[&str] = &[
    "w_list_",
    "w_dict_",
    "list_concat",
    "list_repeat",
    "sequence_repeat",
    "require_list",
    "require_dict",
    "dict_method_",
    "list_method_",
    // The families whose headers the interpreter allocates in the nursery, so
    // a pointer reaching one of these is addressed as an object that relocates.
    // Spelled at the moving constructor rather than at the family prefix: a
    // `w_weakref_lifeline_*` is `allocate_stable` and a `w_str_new` is
    // immortal, so the wider spelling would name a pointer that cannot move.
    // `w_dict_view_*` needs no entry of its own -- the `w_dict_` prefix already
    // spells it.
    //
    // `str` has no entry at all, and deliberately.  Its mobility is decided per
    // constructor -- `w_str_from_wtf8_managed_collecting` is the one nursery
    // spelling, while `w_str_from_wtf8_managed` is `try_gc_alloc_stable` and
    // `w_str_new` is immortal -- so the accessors every one of them shares
    // cannot say which kind the pointer reaching them is.  Naming a constructor
    // here would add nothing either: they take a `Wtf8Buf`, never a
    // `PyObjectRef`, so no GC-pointer local is ever an argument of one.  The
    // same reading rules out the items-block and JITFRAME allocators, whose
    // arguments are a capacity and a type id.
    // `w_method_` is spelled at the accessors rather than at the family
    // prefix, because `w_method_new` takes the new method's three members --
    // not a `Method` -- and roots and re-reads them itself.  Under the family
    // prefix every argument of that constructor counted, so an unrelated local
    // passed as a member was reported as addressed through a method it is only
    // stored in.  `w_tuple_new` needs no such split: it takes a slice, so no
    // `PyObjectRef` local is ever one of its arguments.
    "w_tuple_",
    "w_method_get_",
    "w_method_set_",
    "w_gc_weakref_box_",
    "w_weakref_new",
];

/// The callees that say a pointer reaching them is addressed as a movable
/// object, resolved once for the whole scan.
///
/// `hops` is how far past the named callee to look.  At 0 only the callee's own
/// name is read, which is what the `tier 1.5` column has always done -- and a
/// thin wrapper defeats it: `module_ns_store` is one line forwarding to
/// `w_dict_setitem_str_no_proxy` and matches no marker itself.  A zero at 0 hops
/// therefore says the marker was not the immediate callee's name, not that
/// nothing is addressed as a list or dict.  Each hop admits a caller of
/// something already in the set.
pub fn movable_callee_ids(
    cg: &super::framework::CallGraph,
    markers: &[&str],
    hops: u32,
) -> HashSet<u64> {
    let mut out: HashSet<u64> = cg
        .names
        .iter()
        .filter(|(_, n)| markers.iter().any(|m| n.contains(m)))
        .map(|(&id, _)| id)
        .collect();
    for _ in 0..hops {
        let grown: HashSet<u64> = cg
            .callees
            .iter()
            .filter(|(id, cs)| !out.contains(id) && cs.iter().any(|c| out.contains(c)))
            .map(|(&id, _)| id)
            .collect();
        if grown.is_empty() {
            break;
        }
        out.extend(grown);
    }
    out
}

fn ty_id(t: &TyRef) -> Option<u64> {
    match t {
        TyRef::Dedup { id } => Some(*id),
        TyRef::Inline { value: (id, _) } => Some(*id),
        TyRef::Other(_) => None,
    }
}

/// The type ids `PyObjectRef` is spelled with in *this* artefact.
///
/// Read off signatures pyre already declares in those terms rather than
/// hard-coded, because a dedup id is artefact-local: `pin_root` takes one and
/// `shadow_stack_get` returns one.  An empty result means the analysis would
/// silently find nothing, so callers must report it.
pub fn gc_ptr_type_ids(llbc: &majit_charon_reader::Llbc) -> HashSet<u64> {
    let mut out = HashSet::new();
    for fd in llbc.iter_local_fns() {
        let name = fd.item_meta.name_path();
        // `pin_root` / `shadow_stack_get` live in `pyre-object`. An
        // optional-module artefact extracted `--opaque pyre_object`
        // does not carry those names, but it does carry its own
        // `register_module` / `module_ns_store` signatures, which
        // take the same artefact-local `PyObjectRef` id.
        if name.ends_with("gc_roots::pin_root")
            || name.ends_with("::register_module")
            || name.ends_with("module_ns_store")
        {
            if let Some(t) = fd.signature.inputs.first().and_then(ty_id) {
                out.insert(t);
            }
        } else if name.ends_with("gc_roots::shadow_stack_get")
            && let Some(t) = ty_id(&fd.signature.output)
        {
            out.insert(t);
        }
    }
    // `PyError::pin`'s receiver names the handle, the way `pin_root` names
    // `PyObjectRef`. `Result<_, PyError>` is a different hash-cons id per
    // instantiation; collect those whose error slot is that handle.
    let pyerror = pyerror_type_ids(llbc);
    out.extend(pyerror.iter().copied());
    out.extend(result_pyerror_type_ids(llbc, &pyerror).iter().copied());
    out
}

/// The type ids of `&[PyObjectRef]` in *this* artefact.
///
/// A builtin receives its arguments as a native slice: a copy the collector
/// does not rewrite, so an element read after a collecting call is the same
/// stale word a bare local would be.  Each borrow region is its own type id,
/// so the spellings are read off every body's locals, as for `Option`.
pub fn gc_slice_type_ids(llbc: &majit_charon_reader::Llbc, gc_tys: &HashSet<u64>) -> HashSet<u64> {
    let id_of = |v: &serde_json::Value| {
        v.get("Deduplicated")
            .and_then(serde_json::Value::as_u64)
            .or_else(|| v.pointer("/Value/0").and_then(serde_json::Value::as_u64))
    };
    let body_of = |v: &serde_json::Value| {
        v.pointer("/Value/1")
            .cloned()
            .or_else(|| id_of(v).and_then(|id| llbc.dedup_body(id).cloned()))
    };
    let mut seen = HashSet::new();
    let mut out = HashSet::new();
    for fd in llbc.iter_local_fns() {
        let Some(body) = fd.unstructured() else {
            continue;
        };
        for l in &body.locals.locals {
            let Some(t) = ty_id(&l.ty) else { continue };
            if gc_tys.contains(&t) || !seen.insert(t) {
                continue;
            }
            let elem = llbc
                .dedup_body(t)
                .and_then(|b| b.pointer("/Ref/1").cloned())
                .and_then(|inner| body_of(&inner))
                .and_then(|inner| inner.pointer("/Slice/0").and_then(id_of));
            if elem.is_some_and(|e| gc_tys.contains(&e)) {
                out.insert(t);
            }
        }
    }
    out
}

/// The type ids of `Option<PyObjectRef>` in *this* artefact.
///
/// A GC pointer held as `Option<PyObjectRef>` goes stale across a collecting
/// call exactly like a bare one: the payload is the same word, and nothing
/// publishes it.  Read off every body's locals, so the answer covers only
/// the spellings a body actually holds.
pub fn gc_option_type_ids(llbc: &majit_charon_reader::Llbc, gc_tys: &HashSet<u64>) -> HashSet<u64> {
    let mut seen = HashSet::new();
    let mut out = HashSet::new();
    for fd in llbc.iter_local_fns() {
        let Some(body) = fd.unstructured() else {
            continue;
        };
        for l in &body.locals.locals {
            let Some(t) = ty_id(&l.ty) else { continue };
            if gc_tys.contains(&t) || !seen.insert(t) {
                continue;
            }
            let is_option = llbc
                .dedup_to_adt_def_id(t)
                .and_then(|def| llbc.type_by_id(def))
                .is_some_and(|td| td.item_meta.name_path() == "core::option::Option");
            if !is_option {
                continue;
            }
            let arg = llbc
                .dedup_body(t)
                .and_then(|v| v.pointer("/Adt/generics/types/0"))
                .and_then(|v| serde_json::from_value::<TyRef>(v.clone()).ok())
                .and_then(|r| ty_id(&r));
            if arg.is_some_and(|a| gc_tys.contains(&a)) {
                out.insert(t);
            }
        }
    }
    out
}

fn pyerror_def_ids(llbc: &majit_charon_reader::Llbc) -> HashSet<u64> {
    llbc.iter_type_decls()
        .filter(|td| td.item_meta.name_path().ends_with("::error::PyError"))
        .map(|td| td.def_id)
        .collect()
}

fn json_ty_id(v: &serde_json::Value) -> Option<u64> {
    if let Some(id) = v.get("Deduplicated").and_then(|x| x.as_u64()) {
        return Some(id);
    }
    v.get("Value")
        .and_then(|x| x.as_array())
        .and_then(|a| a.first())
        .and_then(|x| x.as_u64())
}

fn adt_is_pyerror(llbc: &majit_charon_reader::Llbc, id: u64, defs: &HashSet<u64>) -> bool {
    llbc.dedup_to_adt_def_id(id)
        .is_some_and(|d| defs.contains(&d))
}

/// Ids of the `PyError` handle. Prefer `PyError::pin`'s receiver; if that
/// spelling is a reference, also accept any signature that names the ADT.
fn pyerror_type_ids(llbc: &majit_charon_reader::Llbc) -> HashSet<u64> {
    let defs = pyerror_def_ids(llbc);
    let mut out = HashSet::new();
    if defs.is_empty() {
        return out;
    }
    fn consider(
        llbc: &majit_charon_reader::Llbc,
        defs: &HashSet<u64>,
        out: &mut HashSet<u64>,
        ty: &TyRef,
    ) {
        let Some(id) = ty_id(ty) else {
            return;
        };
        if adt_is_pyerror(llbc, id, defs) {
            out.insert(id);
        }
    }
    for fd in llbc.iter_fun_decls() {
        let name = fd.item_meta.name_path();
        if name.contains("::error::") && name.ends_with("::pin") {
            for inp in &fd.signature.inputs {
                consider(llbc, &defs, &mut out, inp);
            }
        }
    }
    if out.is_empty() {
        for fd in llbc.iter_fun_decls() {
            consider(llbc, &defs, &mut out, &fd.signature.output);
            for inp in &fd.signature.inputs {
                consider(llbc, &defs, &mut out, inp);
            }
        }
    }
    out
}

fn result_pyerror_type_ids(
    llbc: &majit_charon_reader::Llbc,
    pyerror_ids: &HashSet<u64>,
) -> HashSet<u64> {
    let py_defs = pyerror_def_ids(llbc);
    let result_defs: HashSet<u64> = llbc
        .iter_type_decls()
        .filter(|td| td.item_meta.name_path() == "core::result::Result")
        .map(|td| td.def_id)
        .collect();
    let mut seen = HashSet::new();
    let mut ids = HashSet::new();
    if result_defs.is_empty() || pyerror_ids.is_empty() {
        return ids;
    }
    let mut classify = |ty: &TyRef| {
        let Some(id) = ty_id(ty) else {
            return;
        };
        if !seen.insert(id) {
            return;
        }
        if !llbc
            .dedup_to_adt_def_id(id)
            .is_some_and(|d| result_defs.contains(&d))
        {
            return;
        }
        let Some(body) = llbc.dedup_body(id) else {
            return;
        };
        let Some(slot) = body
            .get("Adt")
            .and_then(|a| a.get("generics"))
            .and_then(|g| g.get("types"))
            .and_then(|t| t.get(1))
        else {
            return;
        };
        let is_py = if let Some(err_id) = json_ty_id(slot) {
            pyerror_ids.contains(&err_id) || adt_is_pyerror(llbc, err_id, &py_defs)
        } else if let Some(def) = slot
            .get("Adt")
            .and_then(|a| a.get("id"))
            .and_then(|i| i.as_u64())
        {
            py_defs.contains(&def)
        } else {
            false
        };
        if is_py {
            ids.insert(id);
        }
    };
    for fd in llbc.iter_fun_decls() {
        classify(&fd.signature.output);
        for ty in &fd.signature.inputs {
            classify(ty);
        }
    }
    ids
}

/// The type ids a `PyFrame` pointer is spelled with in *this* artefact.
///
/// The frame is the second thing a minor collection can leave a body holding a
/// corpse of, and it is the harder one: a `PyObjectRef` at least has a root
/// stack it can be pinned on, whereas the running frame is carried as a bare
/// `&mut PyFrame` that no walker reaches.  `eval::FrameAnchor` is the reload
/// point, and a body that takes one and re-reads `live()` kills its stale local
/// at the call, so [`scan`] needs no bracket set for this kind — the liveness
/// answer already distinguishes a reloaded frame from a carried one.
///
/// The interpreter spells the pointer three ways and all three go stale, so
/// each is read off a signature pyre already declares in those terms rather
/// than hard-coded: a dedup id is artefact-local.  An empty result means the
/// scan would silently find nothing, so callers must report it.
pub fn frame_ptr_type_ids(llbc: &majit_charon_reader::Llbc) -> HashSet<u64> {
    // Free functions, so the name is the whole match: an inherent method
    // carries its `impl` block as an opaque segment and cannot be named here.
    const FIRST_INPUT: &[&str] = &[
        "eval::install_current_frame",   // &mut PyFrame
        "eval::handle_exception",        // &mut PyFrame
        "executioncontext::force_frame", // *mut PyFrame
        "call::enter_recursive_frame",   // *const PyFrame
    ];
    let mut out = HashSet::new();
    for fd in llbc.iter_local_fns() {
        let name = fd.item_meta.name_path();
        if FIRST_INPUT
            .iter()
            .any(|p| name == *p || name.ends_with(&format!("::{p}")))
            && let Some(t) = fd.signature.inputs.first().and_then(ty_id)
        {
            out.insert(t);
        }
    }
    out
}

/// The local a place is rooted at, looking through every projection.
fn place_local(p: &Place) -> Option<u64> {
    match &p.kind {
        PlaceKind::Local(i) => Some(*i),
        PlaceKind::Projection(base, _) => place_local(base),
        _ => None,
    }
}

/// Whether the place names a whole local (so assigning it kills the old value).
fn bare_local(p: &Place) -> Option<u64> {
    match &p.kind {
        PlaceKind::Local(i) => Some(*i),
        _ => None,
    }
}

/// `x.PtrMetadata` -- a fat pointer's length, not the words behind it.
fn is_metadata_place(p: &Place) -> bool {
    matches!(&p.kind, PlaceKind::Projection(_, ProjectionElem::Atom(e)) if e == "PtrMetadata")
}

fn use_operand(o: &Operand, out: &mut HashSet<u64>) {
    match o {
        Operand::Copy(p) | Operand::Move(p) if is_metadata_place(p) => {}
        Operand::Copy(p) | Operand::Move(p) => {
            if let Some(l) = place_local(p) {
                out.insert(l);
            }
        }
        Operand::Const(_) => {}
    }
}

fn use_rvalue(r: &Rvalue, out: &mut HashSet<u64>) {
    match r {
        // A slice's length lives in the fat pointer, not in the elements the
        // collector would have to forward.
        Rvalue::Len(_) => {}
        Rvalue::Use(o, _) | Rvalue::UnaryOp(_, o) => use_operand(o, out),
        Rvalue::BinaryOp(_, a, b) => {
            use_operand(a, out);
            use_operand(b, out);
        }
        Rvalue::Ref { place, .. } | Rvalue::RawPtr { place, .. } => {
            if let Some(l) = place_local(place) {
                out.insert(l);
            }
        }
        Rvalue::Aggregate(_, ops) => {
            for o in ops {
                use_operand(o, out);
            }
        }
        Rvalue::Discriminant(p) => {
            if let Some(l) = place_local(p) {
                out.insert(l);
            }
        }
        Rvalue::Cast(_, o, _) | Rvalue::Repeat(o, _, _, _) | Rvalue::ShallowInitBox(o, _) => {
            use_operand(o, out)
        }
        Rvalue::NullaryOp(_, _) | Rvalue::Unknown => {}
    }
}

/// How a local that a pin call was handed came to hold its value.
///
/// Only the shapes `pin_roots(&[a, b])` lowers to are followed; anything else
/// leaves the pinned set unread rather than guessed at.
#[derive(Clone, Debug, PartialEq, Eq)]
enum PinSrc {
    /// `_t = [a, b, c]` -- the pinned set itself.
    Aggregate(Vec<u64>),
    /// `_t = &_u`, `_t = _u as _`, `_t = _u` -- the same value under a second
    /// name, so keep walking.
    Alias(u64),
}

fn pin_src(r: &Rvalue) -> Option<PinSrc> {
    let one = |o: &Operand| {
        let mut s = HashSet::new();
        use_operand(o, &mut s);
        s.into_iter().next()
    };
    match r {
        Rvalue::Aggregate(_, ops) => {
            let mut v = Vec::new();
            for o in ops {
                v.extend(one(o));
            }
            Some(PinSrc::Aggregate(v))
        }
        Rvalue::Ref { place, .. } | Rvalue::RawPtr { place, .. } => {
            place_local(place).map(PinSrc::Alias)
        }
        Rvalue::Use(o, _) | Rvalue::Cast(_, o, _) => one(o).map(PinSrc::Alias),
        _ => None,
    }
}

/// Source of a pin-preserving copy: `dest = src`, or a cast of that local.
///
/// `pin_src` follows projections to chase a pin *argument*. A field load is a
/// new pointer; `gct_direct_call` roots the live variables around the
/// collecting op, not a pointer read from one of them.
fn pin_copy_src(r: &Rvalue) -> Option<u64> {
    let op = match r {
        Rvalue::Use(o, _) | Rvalue::Cast(_, o, _) => o,
        _ => return None,
    };
    operand_place(op).and_then(bare_local)
}

/// Every local the value reaching a pin argument is spelled by.
///
/// The whole alias chain is kept, not only its end: a pin publishes a *value*,
/// so every local holding that value at the pin is covered by it, and the
/// liveness answer names whichever one the body went on to use.
fn chase_pinned(l: u64, defs: &HashMap<u64, PinSrc>, out: &mut HashSet<u64>, depth: u32) {
    if !out.insert(l) || depth > 8 {
        return;
    }
    match defs.get(&l) {
        Some(PinSrc::Aggregate(v)) => {
            for &m in v {
                chase_pinned(m, defs, out, depth + 1);
            }
        }
        Some(PinSrc::Alias(next)) => chase_pinned(*next, defs, out, depth + 1),
        None => {}
    }
}

/// Every local an argument to a movable-addressing callee is spelled by.
///
/// The alias half of [`chase_pinned`], and needed for the same reason on the
/// other side of the intersection: a call argument is materialised as its own
/// temporary (`_t = copy _obj; w_tuple_getitem(move _t, ..)`), so matching the
/// argument local against the pinned local compares two spellings of one value
/// and finds nothing.  Aggregates are not followed here -- `pin_roots(&[a, b])`
/// pins each element, but a callee handed a container is addressing the
/// container, not the elements.
fn chase_arg_aliases(l: u64, defs: &HashMap<u64, PinSrc>, out: &mut HashSet<u64>, depth: u32) {
    if !out.insert(l) || depth > 8 {
        return;
    }
    if let Some(PinSrc::Alias(next)) = defs.get(&l) {
        chase_arg_aliases(*next, defs, out, depth + 1);
    }
}

/// The functions that write a livevar onto the root stack.
///
/// `push_roots` opens the scope and takes no arguments; the set is named
/// separately, so the contents are read off these and not off the opener.
/// `publish_roots` is `pin_roots` without the normalize half, and pins just
/// the same.
///
/// The free functions and the scope-local `pin_root` / `publish` forms.
/// `scan` models the matching `RootScope` Drop at the same time, so a method
/// pin cannot leak into calls after its guard was truncated.  Scope-local
/// `set` overwrites a coloured slot and remains opaque until the analysis
/// carries a slot-to-root map; guessing there could claim false coverage.
fn is_pin_fn(name: &str) -> bool {
    name.ends_with("::pin_root")
        || name.ends_with("::pin_roots")
        || name.ends_with("::publish_roots")
        || (name.contains("gc_roots::<Impl>") && name.ends_with("::publish"))
}

fn reads_root_slot(name: &str) -> bool {
    name.ends_with("gc_roots::shadow_stack_get")
        || (name.contains("gc_roots::<Impl>") && name.ends_with("::get"))
}

/// `PyError::pin`. `ends_with("::pin")` does not match `pin_root`.
///
/// The receiver stays in the pinned set: the object is rooted for the call.
/// It is not a pin *argument* the immediate check treats as still being read.
/// `pin` writes the forwarded word back, and that word is fresh only until
/// the next collecting call. `shadowstack.py expand_pop_roots` reloads every
/// live variable after `pop_roots`; only that reloaded word is safe.
fn is_pyerror_handle_pin(name: &str) -> bool {
    name.contains("::error::") && name.ends_with("::pin")
}

/// `PyError::reload`. `ends_with("::reload")` does not match `reload_global`.
fn is_pyerror_handle_reload(name: &str) -> bool {
    name.contains("::error::") && name.ends_with("::reload")
}

/// `&mut self` methods that store the forwarded carrier back into the local.
/// `to_exc_object` does, through its own `pin` / `reload`. `set_exc_object`
/// writes a field and leaves the handle word as it was.
fn is_pyerror_handle_writeback(name: &str) -> bool {
    name.contains("::error::") && (name.ends_with("::pin") || name.ends_with("::to_exc_object"))
}

/// Drop `PyError::pin`'s receiver out of the pin-argument set.
///
/// Chase the `&mut` temporary (`mut_borrow_of`) and single-assignment aliases.
/// The same local stays in the pinned set, which is what coverage reads.
fn drop_pyerror_pin_receiver(
    args: &[Operand],
    mut_borrow_of: &HashMap<u64, u64>,
    defs: &HashMap<u64, PinSrc>,
    args_only: &mut HashSet<u64>,
) {
    let Some(arg0) = args.first() else {
        return;
    };
    let mut seed = HashSet::new();
    use_operand(arg0, &mut seed);
    let mut drop_set = HashSet::new();
    for local in seed {
        chase_handle_local(local, mut_borrow_of, defs, &mut drop_set, 0);
    }
    for local in drop_set {
        args_only.remove(&local);
    }
}

fn chase_handle_local(
    local: u64,
    mut_borrow_of: &HashMap<u64, u64>,
    defs: &HashMap<u64, PinSrc>,
    out: &mut HashSet<u64>,
    depth: u32,
) {
    if !out.insert(local) || depth > 8 {
        return;
    }
    if let Some(&next) = mut_borrow_of.get(&local) {
        chase_handle_local(next, mut_borrow_of, defs, out, depth + 1);
    }
    if let Some(PinSrc::Alias(next)) = defs.get(&local) {
        chase_handle_local(*next, mut_borrow_of, defs, out, depth + 1);
    }
}

/// Locals whose single assignment builds a reference or a raw pointer.
///
/// Copying one of these copies the address of the stack slot, not the GC
/// word. `PyError::reload` is lowered as `_t = &mut err` followed by the
/// call, and that borrow is not a read of the carrier.
fn address_bearing_locals(blocks: &[BasicBlock], forward: &[bool]) -> HashSet<u64> {
    let mut defined = HashSet::new();
    let mut out = HashSet::new();
    for (b, blk) in blocks.iter().enumerate() {
        if !is_forward(forward, b) {
            continue;
        }
        for st in &blk.statements {
            let Ok(StmtKind::Assign(place, rv)) = st.stmt_kind() else {
                continue;
            };
            let Some(dest) = bare_local(&place) else {
                continue;
            };
            if !defined.insert(dest) {
                out.remove(&dest);
                continue;
            }
            if matches!(rv, Rvalue::Ref { .. } | Rvalue::RawPtr { .. }) {
                out.insert(dest);
            }
        }
    }
    out
}

fn operand_place(op: &Operand) -> Option<&Place> {
    match op {
        Operand::Copy(place) | Operand::Move(place) if !is_metadata_place(place) => Some(place),
        _ => None,
    }
}

/// Locals a value operand names, following copy aliases and stopping at a
/// reference. A reference is not a load of the word.
fn operand_value_loads(
    op: &Operand,
    address_locals: &HashSet<u64>,
    defs: &HashMap<u64, PinSrc>,
    mut_borrow_of: &HashMap<u64, u64>,
) -> HashSet<u64> {
    let Some(place) = operand_place(op) else {
        return HashSet::new();
    };
    let Some(local) = place_local(place) else {
        return HashSet::new();
    };
    if address_locals.contains(&local) || mut_borrow_of.contains_key(&local) {
        return HashSet::new();
    }
    chase_value_local(local, address_locals, defs, mut_borrow_of)
}

fn chase_value_local(
    local: u64,
    address_locals: &HashSet<u64>,
    defs: &HashMap<u64, PinSrc>,
    mut_borrow_of: &HashMap<u64, u64>,
) -> HashSet<u64> {
    let mut out = HashSet::new();
    let mut stack = vec![local];
    let mut depth = 0u32;
    while let Some(local) = stack.pop() {
        if address_locals.contains(&local) || mut_borrow_of.contains_key(&local) {
            continue;
        }
        if !out.insert(local) || depth > 8 {
            continue;
        }
        depth += 1;
        if let Some(PinSrc::Alias(next)) = defs.get(&local) {
            stack.push(*next);
        }
    }
    out
}

fn rvalue_value_loads(
    rv: &Rvalue,
    address_locals: &HashSet<u64>,
    defs: &HashMap<u64, PinSrc>,
    mut_borrow_of: &HashMap<u64, u64>,
) -> HashSet<u64> {
    let mut out = HashSet::new();
    match rv {
        Rvalue::Ref { .. }
        | Rvalue::RawPtr { .. }
        | Rvalue::Len(_)
        | Rvalue::NullaryOp(_, _)
        | Rvalue::Unknown => {}
        Rvalue::Use(op, _)
        | Rvalue::UnaryOp(_, op)
        | Rvalue::Cast(_, op, _)
        | Rvalue::Repeat(op, _, _, _)
        | Rvalue::ShallowInitBox(op, _) => {
            out.extend(operand_value_loads(op, address_locals, defs, mut_borrow_of));
        }
        Rvalue::BinaryOp(_, left, right) => {
            out.extend(operand_value_loads(
                left,
                address_locals,
                defs,
                mut_borrow_of,
            ));
            out.extend(operand_value_loads(
                right,
                address_locals,
                defs,
                mut_borrow_of,
            ));
        }
        Rvalue::Aggregate(_, ops) => {
            for op in ops {
                out.extend(operand_value_loads(op, address_locals, defs, mut_borrow_of));
            }
        }
        // The enum tag of the local itself is not the carrier word. A
        // discriminant projected through a field dereferences that word.
        Rvalue::Discriminant(place) => {
            if bare_local(place).is_none()
                && !is_metadata_place(place)
                && let Some(local) = place_local(place)
            {
                out.extend(chase_value_local(
                    local,
                    address_locals,
                    defs,
                    mut_borrow_of,
                ));
            }
        }
    }
    out
}

/// Locals a call argument addresses. The borrow is not itself a load; the
/// callee is, unless it is `PyError::reload` or a slot read into the local.
fn operand_address_roots(
    op: &Operand,
    address_locals: &HashSet<u64>,
    defs: &HashMap<u64, PinSrc>,
    mut_borrow_of: &HashMap<u64, u64>,
) -> HashSet<u64> {
    let Some(place) = operand_place(op) else {
        return HashSet::new();
    };
    let Some(local) = place_local(place) else {
        return HashSet::new();
    };
    if !address_locals.contains(&local) && !mut_borrow_of.contains_key(&local) {
        return HashSet::new();
    }
    let mut out = HashSet::new();
    chase_handle_local(local, mut_borrow_of, defs, &mut out, 0);
    out
}

/// `PyError` / `Result<_, PyError>` locals in `watched` that a path reads
/// before a reload. `shadowstack.py gc_restore_root` is that reload: the
/// word from before `push_roots` is not the word after the collecting call.
///
/// One collecting call flags at most once. The walk starts at the call's
/// successors, so the call's own arguments are the pre-call word.
fn pyerror_stale_after_collect(
    blocks: &[BasicBlock],
    terms: &[Option<TermKind>],
    names: &HashMap<u64, String>,
    mut_borrow_of: &HashMap<u64, u64>,
    defs: &HashMap<u64, PinSrc>,
    address_locals: &HashSet<u64>,
    origin: usize,
    watched: &HashSet<u64>,
    forward: &[bool],
) -> HashSet<u64> {
    let Some(term) = terms.get(origin).and_then(|term| term.as_ref()) else {
        return HashSet::new();
    };
    let n = blocks.len();
    let mut stale_in: Vec<HashSet<u64>> = vec![HashSet::new(); n];
    let mut queued = vec![false; n];
    let mut queue = VecDeque::new();
    for succ in successors(term, forward) {
        let succ = succ as usize;
        if succ >= n {
            continue;
        }
        stale_in[succ].clone_from(watched);
        queued[succ] = true;
        queue.push_back(succ);
    }
    let mut flagged = HashSet::new();
    while let Some(block) = queue.pop_front() {
        queued[block] = false;
        let mut stale = stale_in[block].clone();
        for st in &blocks[block].statements {
            let Ok(kind) = st.stmt_kind() else {
                continue;
            };
            note_stmt_loads(
                &kind,
                &mut stale,
                &mut flagged,
                address_locals,
                defs,
                mut_borrow_of,
            );
        }
        if let Some(term) = terms.get(block).and_then(|term| term.as_ref()) {
            note_term_loads(
                term,
                names,
                &mut stale,
                &mut flagged,
                address_locals,
                defs,
                mut_borrow_of,
            );
            for succ in successors(term, forward) {
                let succ = succ as usize;
                if succ >= n {
                    continue;
                }
                let before = stale_in[succ].len();
                stale_in[succ].extend(stale.iter().copied());
                if stale_in[succ].len() != before && !queued[succ] {
                    queued[succ] = true;
                    queue.push_back(succ);
                }
            }
        }
    }
    flagged.retain(|local| watched.contains(local));
    flagged
}

fn note_stmt_loads(
    kind: &StmtKind,
    stale: &mut HashSet<u64>,
    flagged: &mut HashSet<u64>,
    address_locals: &HashSet<u64>,
    defs: &HashMap<u64, PinSrc>,
    mut_borrow_of: &HashMap<u64, u64>,
) {
    match kind {
        // `PlaceMention` and storage markers do not load the carrier word.
        // A dead local cannot be read later on this path.
        StmtKind::StorageDead(local) => {
            stale.remove(local);
        }
        StmtKind::PlaceMention(_)
        | StmtKind::StorageLive(_)
        | StmtKind::Borrowck(_)
        | StmtKind::Unknown => {}
        StmtKind::Assert(assert) => {
            flag_value_loads(
                &operand_value_loads(&assert.cond, address_locals, defs, mut_borrow_of),
                stale,
                flagged,
            );
        }
        StmtKind::Assign(place, rv) => {
            let loaded = rvalue_value_loads(rv, address_locals, defs, mut_borrow_of);
            flag_value_loads(&loaded, stale, flagged);
            // A kill that does not read the old word is the fresh value.
            // `gc_restore_root` has that shape when the slot read is a call
            // whose destination is the local; a plain assignment of another
            // local is the same.
            if let Some(dest) = bare_local(place)
                && stale.contains(&dest)
                && !loaded.contains(&dest)
            {
                stale.remove(&dest);
            }
        }
    }
}

fn note_term_loads(
    term: &TermKind,
    names: &HashMap<u64, String>,
    stale: &mut HashSet<u64>,
    flagged: &mut HashSet<u64>,
    address_locals: &HashSet<u64>,
    defs: &HashMap<u64, PinSrc>,
    mut_borrow_of: &HashMap<u64, u64>,
) {
    match term {
        TermKind::Switch { discr, .. } => {
            flag_value_loads(
                &operand_value_loads(discr, address_locals, defs, mut_borrow_of),
                stale,
                flagged,
            );
        }
        TermKind::Assert { assert, .. } => {
            flag_value_loads(
                &operand_value_loads(&assert.cond, address_locals, defs, mut_borrow_of),
                stale,
                flagged,
            );
        }
        // User `Drop` glue would load the word. `PyError` has none, and a
        // glue-less `Drop` is not a read. `framework.py` stops at the last
        // real use for the same reason.
        TermKind::Drop { .. }
        | TermKind::Return
        | TermKind::Goto { .. }
        | TermKind::UnwindResume
        | TermKind::UnwindTerminate
        | TermKind::Abort(_)
        | TermKind::Panic { .. }
        | TermKind::UndefinedBehavior
        | TermKind::Unknown => {}
        TermKind::Call { call, .. } => {
            let name = call_name(call, names);
            let mut values = HashSet::new();
            let mut addressed = HashSet::new();
            for arg in &call.args {
                values.extend(operand_value_loads(
                    arg,
                    address_locals,
                    defs,
                    mut_borrow_of,
                ));
                addressed.extend(operand_address_roots(
                    arg,
                    address_locals,
                    defs,
                    mut_borrow_of,
                ));
            }
            let reload = name.is_some_and(is_pyerror_handle_reload);
            let slot_read = name.is_some_and(reads_root_slot);
            let writeback = name.is_some_and(is_pyerror_handle_writeback);
            if !reload {
                flag_value_loads(&values, stale, flagged);
                for local in &addressed {
                    if stale.contains(local) {
                        flagged.insert(*local);
                    }
                }
            }
            let dest = bare_local(&call.dest);
            for local in stale.clone() {
                let reloaded = reload && addressed.contains(&local);
                let from_slot = slot_read && dest == Some(local) && !values.contains(&local);
                let written = writeback && addressed.contains(&local);
                let killed =
                    dest == Some(local) && !values.contains(&local) && !addressed.contains(&local);
                if reloaded || from_slot || written || killed {
                    stale.remove(&local);
                }
            }
        }
    }
}

fn call_name<'a>(
    call: &majit_charon_reader::ullbc::CallPayload,
    names: &'a HashMap<u64, String>,
) -> Option<&'a str> {
    let CallFunc::Regular(reg) = &call.func else {
        return None;
    };
    let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
        return None;
    };
    names.get(id).map(String::as_str)
}

fn flag_value_loads(loaded: &HashSet<u64>, stale: &HashSet<u64>, flagged: &mut HashSet<u64>) {
    for local in loaded {
        if stale.contains(local) {
            flagged.insert(*local);
        }
    }
}

/// `CallKind::Fun(FunId::Regular)` — the only call shape [`scan`] treats as
/// a resolved callee. `push_roots` and the pin helpers are read off these.
fn regular_fun_id(func: &CallFunc) -> Option<u64> {
    match func {
        CallFunc::Regular(reg) => match &reg.kind {
            CallKind::Fun(FunId::Regular { id }) => Some(*id),
            _ => None,
        },
        _ => None,
    }
}

/// The predicate [`scan`] uses to find `bracket_blocks`: this call opens a
/// root scope.
fn opens_root_scope(func: &CallFunc, push_roots: &HashSet<u64>) -> bool {
    regular_fun_id(func).is_some_and(|id| push_roots.contains(&id))
}

/// Hash-cons ids of `RootScope`, read off `push_roots`'s return type.
fn root_scope_type_ids(
    llbc: &majit_charon_reader::Llbc,
    push_roots: &HashSet<u64>,
) -> HashSet<u64> {
    let mut out = HashSet::new();
    for &id in push_roots {
        let Some(fd) = llbc.fn_by_id(id) else {
            continue;
        };
        if let Some(t) = ty_id(&fd.signature.output) {
            out.insert(t);
        }
    }
    out
}

fn pointer_pointee(node: &serde_json::Value) -> Option<&serde_json::Value> {
    if let Some(arr) = node.get("Ref").and_then(serde_json::Value::as_array) {
        return arr.get(1);
    }
    if let Some(arr) = node.get("RawPtr").and_then(serde_json::Value::as_array) {
        return arr.first();
    }
    None
}

fn resolve_ty_node<'a>(
    node: &'a serde_json::Value,
    llbc: &'a majit_charon_reader::Llbc,
) -> Option<&'a serde_json::Value> {
    if let Some(id) = node.get("Deduplicated").and_then(serde_json::Value::as_u64) {
        return llbc.dedup_body(id);
    }
    if let Some(pair) = node.get("Value").and_then(serde_json::Value::as_array) {
        return pair.get(1);
    }
    Some(node)
}

/// Parameter locals that already hold a `RootScope` at entry: the guard
/// itself, or a wrapper whose immediate field is one (`WritableBuffer`).
fn incoming_root_scopes(
    body: &Unstructured,
    llbc: &majit_charon_reader::Llbc,
    root_scope_tys: &HashSet<u64>,
) -> Vec<u64> {
    if root_scope_tys.is_empty() {
        return Vec::new();
    }
    body.locals
        .locals
        .iter()
        .filter(|local| {
            local.index >= 1
                && local.index <= body.locals.arg_count
                && ty_holds_root_scope(&local.ty, llbc, root_scope_tys)
        })
        .map(|local| local.index)
        .collect()
}

fn ty_holds_root_scope(
    ty: &TyRef,
    llbc: &majit_charon_reader::Llbc,
    root_scope_tys: &HashSet<u64>,
) -> bool {
    if ty_id(ty).is_some_and(|t| root_scope_tys.contains(&t)) {
        return true;
    }
    let Some(node) = ty_body(ty, llbc) else {
        return false;
    };
    let pointee = pointer_pointee(node).unwrap_or(node);
    if json_ty_id(pointee).is_some_and(|t| root_scope_tys.contains(&t)) {
        return true;
    }
    let Some(resolved) = resolve_ty_node(pointee, llbc) else {
        return false;
    };
    if json_ty_id(resolved).is_some_and(|t| root_scope_tys.contains(&t)) {
        return true;
    }
    let adt_id = resolved
        .get("Adt")
        .and_then(adt_nominal_def_id)
        .or_else(|| json_ty_id(pointee).and_then(|id| llbc.dedup_to_adt_def_id(id)));
    let Some(decl) = adt_id.and_then(|id| llbc.type_by_id(id)) else {
        return false;
    };
    let field_is_scope = |ty: &TyRef| ty_id(ty).is_some_and(|t| root_scope_tys.contains(&t));
    match &decl.kind {
        TypeDeclKind::Struct(fields) | TypeDeclKind::Union(fields) => {
            fields.iter().any(|field| field_is_scope(&field.ty))
        }
        TypeDeclKind::Enum(variants) => variants
            .iter()
            .any(|variant| variant.fields.iter().any(|field| field_is_scope(&field.ty))),
        _ => false,
    }
}

fn drop_of_root_scope(place: &Place, root_scope_tys: &HashSet<u64>) -> bool {
    ty_id(&place.ty).is_some_and(|t| root_scope_tys.contains(&t))
}

/// A function that pins into its caller's open root scope.
///
/// It calls [`is_pin_fn`] (or another such function) and never calls
/// `push_roots` itself. `pinned_params` are 0-based positions among the call
/// arguments; `returns_pinned` means the return place holds a pin result or
/// a slot read.
#[derive(Clone, Debug, PartialEq, Eq)]
struct PinHelperSummary {
    pinned_params: HashSet<usize>,
    returns_pinned: bool,
}

/// One resolved call, reduced to the locals its arguments name.
#[derive(Clone, Debug, PartialEq, Eq)]
struct HelperCallFact {
    callee: u64,
    callee_name: String,
    /// Locals [`use_operand`] named for each argument, before [`chase_pinned`].
    arg_locals: Vec<Vec<u64>>,
    /// Locals assigned on some path before this call, including statements
    /// in its block. A later assignment does not retire the incoming parameter.
    assigned_before: HashSet<u64>,
    /// Bare local the call writes, when the destination is one local.
    dest: Option<u64>,
}

/// What [`summarize_pin_helpers`] needs from one body. Built by
/// [`helper_body_fact`]; tests construct it directly.
#[derive(Clone)]
struct HelperBodyFact {
    has_push_roots: bool,
    /// MIR parameters are locals `1..=arg_count`. Local 0 is the return place.
    arg_count: u64,
    defs: HashMap<u64, PinSrc>,
    calls: Vec<HelperCallFact>,
    /// Bare locals whose single assignment is a pin call or a slot read.
    pin_result_locals: HashSet<u64>,
    /// Per-block indices into [`Self::calls`]. Empty means the tests' single
    /// block: every call runs, then the function returns.
    block_calls: Vec<Vec<usize>>,
    /// Normal successors only (`Call.target`, not `on_unwind`). Empty when
    /// [`Self::block_calls`] is empty.
    successors: Vec<Vec<usize>>,
}

struct PinAssignIndex {
    defs: HashMap<u64, PinSrc>,
    /// Bare locals whose single assignment is a call, and that were not
    /// overwritten later. A pin result is one of these.
    call_dests: HashSet<u64>,
    /// `_t = &mut _l` — the callee writes the live word back through `_t`.
    mut_borrow_of: HashMap<u64, u64>,
}

/// Single-assignment locals the pin-argument chase will follow.
///
/// A local assigned twice is not a chain worth following. Call destinations
/// are marked assigned so a later statement does not alias them, and they are
/// not given a [`PinSrc`]: the value came from the callee.
fn index_pin_assigns(
    blocks: &[BasicBlock],
    terms: &[Option<TermKind>],
    forward: &[bool],
) -> PinAssignIndex {
    let mut defs: HashMap<u64, PinSrc> = HashMap::new();
    let mut defined: HashSet<u64> = HashSet::new();
    // `_t = &mut _l`: a callee handed `_t` owns keeping `_l` current, the
    // way `try_dispatch_binary_special` pins both operands and writes the
    // live words back through its `&mut` parameters.
    let mut mut_borrow_of: HashMap<u64, u64> = HashMap::new();
    let mut call_dests: HashSet<u64> = HashSet::new();
    for (b, blk) in blocks.iter().enumerate() {
        if !is_forward(forward, b) {
            continue;
        }
        for st in &blk.statements {
            let Ok(StmtKind::Assign(place, rv)) = st.stmt_kind() else {
                continue;
            };
            let Some(d) = bare_local(&place) else {
                continue;
            };
            if !defined.insert(d) {
                defs.remove(&d);
                mut_borrow_of.remove(&d);
                call_dests.remove(&d);
                continue;
            }
            // A two-phase call argument reborrows: `_t = &mut _l;
            // _u = &TwoPhaseMut (*_t)`, so a deref of a recorded borrow
            // names the same local.
            if let Rvalue::Ref { place, kind, .. } = &rv
                && matches!(kind.as_str(), Some("Mut" | "TwoPhaseMut"))
                && let Some(l) = bare_local(place).or_else(|| match &place.kind {
                    PlaceKind::Projection(base, ProjectionElem::Atom(elem)) if elem == "Deref" => {
                        bare_local(base).and_then(|t| mut_borrow_of.get(&t).copied())
                    }
                    _ => None,
                })
            {
                mut_borrow_of.insert(d, l);
            }
            if let Some(src) = pin_src(&rv) {
                defs.insert(d, src);
            }
        }
        if let Some(TermKind::Call { call, .. }) = &terms[b]
            && let Some(d) = bare_local(&call.dest)
        {
            if !defined.insert(d) {
                defs.remove(&d);
                call_dests.remove(&d);
            } else {
                call_dests.insert(d);
            }
        }
    }
    PinAssignIndex {
        defs,
        call_dests,
        mut_borrow_of,
    }
}

/// Parameter positions (0-based) among `1..=arg_count` that `seeds` reaches.
fn param_positions_reaching(
    seeds: &[u64],
    defs: &HashMap<u64, PinSrc>,
    assigned: &HashSet<u64>,
    arg_count: u64,
) -> HashSet<usize> {
    let mut chased = HashSet::new();
    for &local in seeds {
        chase_pinned(local, defs, &mut chased, 0);
    }
    chased
        .into_iter()
        .filter(|local| *local >= 1 && *local <= arg_count && !assigned.contains(local))
        .map(|local| (local - 1) as usize)
        .collect()
}

fn call_pins_params(
    call: &HelperCallFact,
    body: &HelperBodyFact,
    nested: &HashMap<u64, PinHelperSummary>,
) -> HashSet<usize> {
    let mut out = HashSet::new();
    if is_pin_fn(&call.callee_name) {
        for seeds in &call.arg_locals {
            out.extend(param_positions_reaching(
                seeds,
                &body.defs,
                &call.assigned_before,
                body.arg_count,
            ));
        }
        return out;
    }
    let Some(summary) = nested.get(&call.callee) else {
        return out;
    };
    for &position in &summary.pinned_params {
        let Some(seeds) = call.arg_locals.get(position) else {
            continue;
        };
        out.extend(param_positions_reaching(
            seeds,
            &body.defs,
            &call.assigned_before,
            body.arg_count,
        ));
    }
    out
}

fn block_layout(body: &HelperBodyFact) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    if body.block_calls.is_empty() {
        return (vec![(0..body.calls.len()).collect()], vec![Vec::new()]);
    }
    (body.block_calls.clone(), body.successors.clone())
}

/// Parameter positions pinned on every normal path from entry to a return.
/// A pin that lives on only one arm of a switch, or after an early return,
/// is not in the set: the caller scan would otherwise treat the helper as
/// bracketing a use that the skipped arm never rooted.
fn must_pinned_params(
    body: &HelperBodyFact,
    nested: &HashMap<u64, PinHelperSummary>,
) -> HashSet<usize> {
    let (block_calls, successors) = block_layout(body);
    let n = block_calls.len();
    if n == 0 {
        return HashSet::new();
    }
    let block_pins: Vec<HashSet<usize>> = block_calls
        .iter()
        .map(|idxs| {
            let mut pins = HashSet::new();
            for &i in idxs {
                if let Some(call) = body.calls.get(i) {
                    pins.extend(call_pins_params(call, body, nested));
                }
            }
            pins
        })
        .collect();
    // Forward must: join is intersection, transfer is union with this block's
    // pins. Entry has nothing yet.
    let mut in_set: Vec<Option<HashSet<usize>>> = vec![None; n];
    in_set[0] = Some(HashSet::new());
    let mut work: Vec<usize> = vec![0];
    while let Some(b) = work.pop() {
        let mut out = in_set[b].clone().unwrap_or_default();
        out.extend(&block_pins[b]);
        for &s in successors.get(b).map(|v| v.as_slice()).unwrap_or(&[]) {
            if s >= n {
                continue;
            }
            match &in_set[s] {
                None => {
                    in_set[s] = Some(out.clone());
                    work.push(s);
                }
                Some(have) => {
                    let joined: HashSet<usize> = have.intersection(&out).copied().collect();
                    if joined != *have {
                        in_set[s] = Some(joined);
                        work.push(s);
                    }
                }
            }
        }
    }
    let mut exits = Vec::new();
    for b in 0..n {
        if successors.get(b).map(|s| s.is_empty()).unwrap_or(true) && in_set[b].is_some() {
            exits.push(b);
        }
    }
    if exits.is_empty() {
        return HashSet::new();
    }
    let mut must = {
        let mut out = in_set[exits[0]].clone().unwrap_or_default();
        out.extend(&block_pins[exits[0]]);
        out
    };
    for &b in &exits[1..] {
        let mut out = in_set[b].clone().unwrap_or_default();
        out.extend(&block_pins[b]);
        must = must.intersection(&out).copied().collect();
    }
    must
}

fn direct_pinned_params(body: &HelperBodyFact) -> HashSet<usize> {
    must_pinned_params(body, &HashMap::new())
}

/// Local 0 holds a pin result or a slot read, following single-assignment
/// aliases only. An aggregate or a second assignment is not that, and stays
/// unpinned. A collecting call after the pin that produced that local drops
/// the claim: `framework.py get_livevars_for_roots` is live-at-call.
fn returns_pinned_word(
    body: &HelperBodyFact,
    collecting: &HashSet<u64>,
    helpers: &HashSet<u64>,
) -> bool {
    pin_source_local(body).is_some() && !collects_after_pin(body, collecting, helpers)
}

/// The pin/slot-read local that local 0 aliases, when that chain is a single
/// word rather than a wider aggregate.
fn pin_source_local(body: &HelperBodyFact) -> Option<u64> {
    let mut local = 0u64;
    let mut seen = HashSet::new();
    loop {
        if !seen.insert(local) {
            return None;
        }
        if body.pin_result_locals.contains(&local) {
            return Some(local);
        }
        match body.defs.get(&local) {
            Some(PinSrc::Alias(next)) => local = *next,
            // A one-field newtype of the pin word (`PyError(raw)`) is that
            // word. A wider aggregate is a container, not the root.
            Some(PinSrc::Aggregate(fields)) if fields.len() == 1 => local = fields[0],
            _ => return None,
        }
    }
}

fn call_collects(call: &HelperCallFact, collecting: &HashSet<u64>) -> bool {
    collecting.contains(&call.callee)
}

/// Call that wrote `source`, preferring a recorded dest. Without dest, the
/// first pin/slot-read is the producer, so a later pin still invalidates an
/// unmarked return of the first pin.
fn producer_call_index(body: &HelperBodyFact, source: u64) -> Option<usize> {
    let dest_hit = body.calls.iter().enumerate().rev().find_map(|(i, call)| {
        if (is_pin_fn(&call.callee_name) || reads_root_slot(&call.callee_name))
            && call.dest == Some(source)
        {
            Some(i)
        } else {
            None
        }
    });
    if dest_hit.is_some() {
        return dest_hit;
    }
    body.calls.iter().enumerate().find_map(|(i, call)| {
        if is_pin_fn(&call.callee_name) || reads_root_slot(&call.callee_name) {
            Some(i)
        } else {
            None
        }
    })
}

/// A later `pin_root` waits at a safepoint (`gc_roots::normalize_published_slot`).
/// A nested pin helper does too, because it calls `pin_root`. A collecting
/// callee is `CollectAnalyzer.analyze`. A slot read reloads the current word.
fn call_is_later_safepoint(
    call: &HelperCallFact,
    collecting: &HashSet<u64>,
    helpers: &HashSet<u64>,
) -> bool {
    if is_pin_fn(&call.callee_name) {
        return true;
    }
    if helpers.contains(&call.callee) {
        return true;
    }
    call_collects(call, collecting) && !reads_root_slot(&call.callee_name)
}

/// True when a collecting call can run after the pin that produced the return.
///
/// `collecting` is [`CallGraph::reaching`] of
/// [`COLLECTING_SEEDS`](super::framework::COLLECTING_SEEDS), so a wrapper such
/// as `w_weakref_new` is included. `helpers` are the summarized pin helpers:
/// a nested one pins at a safepoint.
fn collects_after_pin(
    body: &HelperBodyFact,
    collecting: &HashSet<u64>,
    helpers: &HashSet<u64>,
) -> bool {
    let Some(source) = pin_source_local(body) else {
        return false;
    };
    let Some(producer) = producer_call_index(body, source) else {
        return false;
    };
    if body.block_calls.is_empty() {
        return body
            .calls
            .iter()
            .enumerate()
            .any(|(i, call)| i > producer && call_is_later_safepoint(call, collecting, helpers));
    }
    let n = body.block_calls.len();
    let mut producer_block = None;
    for (b, indices) in body.block_calls.iter().enumerate() {
        if indices.contains(&producer) {
            producer_block = Some(b);
            break;
        }
    }
    let Some(pb) = producer_block else {
        return false;
    };
    for &i in &body.block_calls[pb] {
        if i > producer && call_is_later_safepoint(&body.calls[i], collecting, helpers) {
            return true;
        }
    }
    let mut stack = Vec::new();
    if let Some(succ) = body.successors.get(pb) {
        stack.extend(succ.iter().copied());
    }
    if stack.is_empty() {
        return false;
    }
    let mut seen = vec![false; n];
    while let Some(b) = stack.pop() {
        if seen[b] {
            continue;
        }
        seen[b] = true;
        for &i in &body.block_calls[b] {
            if call_is_later_safepoint(&body.calls[i], collecting, helpers) {
                return true;
            }
        }
        if let Some(succ) = body.successors.get(b) {
            stack.extend(succ.iter().copied());
        }
    }
    false
}

fn body_calls_pin(body: &HelperBodyFact) -> bool {
    body.calls.iter().any(|call| is_pin_fn(&call.callee_name))
}

/// Which of `bodies` are pin helpers, and which of their parameters they pin.
///
/// A body is a helper when it does not open a root scope and it calls
/// [`is_pin_fn`] or, transitively, another helper. `pinned_params` starts
/// from the locals that reach a pin argument and then grows by parameters
/// handed to a nested helper at one of *its* pinned positions.
fn summarize_pin_helpers(
    bodies: &HashMap<u64, HelperBodyFact>,
    collecting: &HashSet<u64>,
) -> HashMap<u64, PinHelperSummary> {
    let opens_nested = |body: &HelperBodyFact| {
        body.has_push_roots
            || body
                .calls
                .iter()
                .any(|call| bodies.get(&call.callee).is_some_and(|b| b.has_push_roots))
    };
    let mut helpers: HashSet<u64> = bodies
        .iter()
        .filter(|(_, body)| !opens_nested(body) && body_calls_pin(body))
        .map(|(&id, _)| id)
        .collect();
    loop {
        let grown: Vec<u64> = bodies
            .iter()
            .filter(|(id, body)| {
                !helpers.contains(*id)
                    && !opens_nested(body)
                    && body.calls.iter().any(|call| helpers.contains(&call.callee))
            })
            .map(|(&id, _)| id)
            .collect();
        if grown.is_empty() {
            break;
        }
        helpers.extend(grown);
    }

    let mut summaries: HashMap<u64, PinHelperSummary> = helpers
        .iter()
        .map(|&id| {
            let body = &bodies[&id];
            (
                id,
                PinHelperSummary {
                    pinned_params: direct_pinned_params(body),
                    returns_pinned: returns_pinned_word(body, collecting, &helpers),
                },
            )
        })
        .collect();

    // Nested helpers contribute only on paths that always call them. Each
    // round may add a must-pin, and a helper has finitely many parameters.
    loop {
        let mut changed = false;
        for &id in &helpers {
            let next = must_pinned_params(&bodies[&id], &summaries);
            let have = &mut summaries
                .get_mut(&id)
                .expect("helper id is in the summary map")
                .pinned_params;
            if next != *have {
                *have = next;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    summaries
}

/// Superset of the helper ids, from the call graph alone.
///
/// [`CallGraph::callees`](super::framework::CallGraph::callees) records every
/// regular callee [`opens_root_scope`] would see, plus some trait-default
/// edges. Those extra edges can only add candidates; [`helper_body_fact`]
/// re-reads the body with [`opens_root_scope`] and [`summarize_pin_helpers`]
/// drops whoever does not qualify.
fn helper_candidate_ids(
    callees: &HashMap<u64, HashSet<u64>>,
    names: &HashMap<u64, String>,
    push_roots: &HashSet<u64>,
) -> HashSet<u64> {
    let mut openers: HashSet<u64> = callees
        .iter()
        .filter(|(_, cs)| cs.iter().any(|callee| push_roots.contains(callee)))
        .map(|(&id, _)| id)
        .collect();
    loop {
        let grown: Vec<u64> = callees
            .iter()
            .filter(|(id, cs)| {
                !openers.contains(*id) && cs.iter().any(|callee| openers.contains(callee))
            })
            .map(|(&id, _)| id)
            .collect();
        if grown.is_empty() {
            break;
        }
        openers.extend(grown);
    }
    let pins = |cs: &HashSet<u64>| {
        cs.iter()
            .any(|callee| names.get(callee).is_some_and(|name| is_pin_fn(name)))
    };
    let mut helpers: HashSet<u64> = callees
        .iter()
        .filter(|(id, cs)| !openers.contains(*id) && pins(cs))
        .map(|(&id, _)| id)
        .collect();
    loop {
        let grown: Vec<u64> = callees
            .iter()
            .filter(|(id, cs)| {
                !helpers.contains(*id)
                    && !openers.contains(*id)
                    && cs.iter().any(|callee| helpers.contains(callee))
            })
            .map(|(&id, _)| id)
            .collect();
        if grown.is_empty() {
            break;
        }
        helpers.extend(grown);
    }
    helpers
}

/// Locals assigned on some path into each block.
///
/// A successor is queued the first time it is reached, including when the
/// incoming set is empty. A call destination is part of the outgoing set
/// and not of the block's own entry.
fn assigned_at_block_entry(
    assigns: &[HashSet<u64>],
    call_dests: &[HashSet<u64>],
    succ: &[Vec<usize>],
) -> Vec<HashSet<u64>> {
    let n = assigns.len();
    let mut entry = vec![HashSet::new(); n];
    let mut queued = vec![false; n];
    let mut seen = vec![false; n];
    let mut work = Vec::new();
    if n > 0 {
        queued[0] = true;
        seen[0] = true;
        work.push(0usize);
    }
    while let Some(b) = work.pop() {
        queued[b] = false;
        let mut out = entry[b].clone();
        out.extend(&assigns[b]);
        if b < call_dests.len() {
            out.extend(&call_dests[b]);
        }
        for &s in succ.get(b).into_iter().flatten() {
            if s >= n {
                continue;
            }
            let before = entry[s].len();
            entry[s].extend(&out);
            if (!seen[s] || entry[s].len() != before) && !queued[s] {
                seen[s] = true;
                queued[s] = true;
                work.push(s);
            }
        }
    }
    entry
}

fn helper_body_fact(
    llbc: &majit_charon_reader::Llbc,
    body: &Unstructured,
    push_roots: &HashSet<u64>,
    names: &HashMap<u64, String>,
) -> Option<HelperBodyFact> {
    let forward = crate::front::mir::forward_reachable_mask(llbc, body);
    let terms: Vec<Option<TermKind>> = body.body.iter().map(|blk| blk.term(llbc).ok()).collect();
    // A terminator this reader cannot classify may be the `push_roots` that
    // disqualifies the helper, or the only pin. Either misread is worse than
    // leaving the caller unread. Cleanup reached only through `on_unwind`
    // is outside the flow graph, matching `strip_cleanup_blocks`.
    if terms
        .iter()
        .enumerate()
        .any(|(b, term)| is_forward(&forward, b) && matches!(term, None | Some(TermKind::Unknown)))
    {
        return None;
    }
    let index = index_pin_assigns(&body.body, &terms, &forward);
    let mut has_push_roots = false;
    let mut calls = Vec::new();
    let mut pin_result_locals = HashSet::new();
    let n = terms.len();
    let mut block_calls = vec![Vec::new(); n];
    let mut succ = vec![Vec::new(); n];
    let mut call_dests = vec![HashSet::new(); n];
    for (b, term) in terms.iter().enumerate() {
        if !is_forward(&forward, b) {
            continue;
        }
        if let Some(t) = term {
            // `Call.target` only: `on_unwind` is the `UnwindResume` stub
            // `strip_cleanup_blocks` would have substituted.
            succ[b] = match t {
                TermKind::Call { target, .. }
                | TermKind::Assert { target, .. }
                | TermKind::Drop { target, .. } => vec![*target as usize],
                other => successors(other, &forward)
                    .into_iter()
                    .map(|s| s as usize)
                    .collect(),
            };
        }
        let Some(TermKind::Call { call, .. }) = term else {
            continue;
        };
        if let Some(dest) = bare_local(&call.dest) {
            call_dests[b].insert(dest);
        }
        if opens_root_scope(&call.func, push_roots) {
            has_push_roots = true;
        }
        let Some(callee) = regular_fun_id(&call.func) else {
            continue;
        };
        let callee_name = names.get(&callee).cloned().unwrap_or_default();
        if (is_pin_fn(&callee_name) || reads_root_slot(&callee_name))
            && let Some(dest) = bare_local(&call.dest)
            && index.call_dests.contains(&dest)
        {
            pin_result_locals.insert(dest);
        }
        let mut arg_locals = Vec::with_capacity(call.args.len());
        for arg in &call.args {
            let mut seed = HashSet::new();
            use_operand(arg, &mut seed);
            arg_locals.push(seed.into_iter().collect());
        }
        block_calls[b].push(calls.len());
        calls.push(HelperCallFact {
            callee,
            callee_name,
            arg_locals,
            assigned_before: HashSet::new(),
            dest: bare_local(&call.dest),
        });
    }
    let mut assigns = vec![HashSet::new(); n];
    for (b, blk) in body.body.iter().enumerate() {
        if !is_forward(&forward, b) {
            continue;
        }
        for st in &blk.statements {
            if let Ok(StmtKind::Assign(place, _)) = st.stmt_kind()
                && let Some(d) = bare_local(&place)
            {
                assigns[b].insert(d);
            }
        }
    }
    let entry = assigned_at_block_entry(&assigns, &call_dests, &succ);
    for (b, idxs) in block_calls.iter().enumerate() {
        let mut before = entry[b].clone();
        before.extend(&assigns[b]);
        for &i in idxs {
            calls[i].assigned_before = before.clone();
        }
    }
    Some(HelperBodyFact {
        has_push_roots,
        arg_count: body.locals.arg_count,
        defs: index.defs,
        calls,
        pin_result_locals,
        block_calls,
        successors: succ,
    })
}

/// Pin helpers in this artefact, keyed by function id.
///
/// An empty `push_roots` makes every pin-caller a helper, and no bracket is
/// ever active to grade — frame scans pass that empty set. Skip the walk.
fn pin_helper_summaries(
    llbc: &majit_charon_reader::Llbc,
    cg: &super::framework::CallGraph,
    push_roots: &HashSet<u64>,
    donors: &[(&majit_charon_reader::Llbc, &super::framework::CallGraph)],
) -> HashMap<u64, PinHelperSummary> {
    if push_roots.is_empty() {
        return HashMap::new();
    }
    let candidates = helper_candidate_ids(&cg.callees, &cg.names, push_roots);
    let mut bodies = HashMap::with_capacity(candidates.len());
    for id in candidates {
        let Some(fd) = llbc.fn_by_id(id) else {
            continue;
        };
        let Some(body) = fd.unstructured() else {
            continue;
        };
        let Some(fact) = helper_body_fact(llbc, &body, push_roots, &cg.names) else {
            continue;
        };
        bodies.insert(id, fact);
    }
    // A cross-crate helper is often an opaque declaration here. Its body
    // lives in the donor that extracted it. The callee ids inside that body
    // stay the donor's; `summarize_pin_helpers` keys off names for `pin_root`.
    if !donors.is_empty() {
        let mut donor_by_name: HashMap<&str, HelperBodyFact> = HashMap::new();
        for (dllbc, dcg) in donors {
            let push: HashSet<u64> = dcg
                .names
                .iter()
                .filter(|(_, n)| n.ends_with("gc_roots::push_roots"))
                .map(|(&id, _)| id)
                .collect();
            if push.is_empty() {
                continue;
            }
            let donor_collecting = collecting_for_pin_summaries(dcg);
            for id in helper_candidate_ids(&dcg.callees, &dcg.names, &push) {
                let Some(name) = dcg.names.get(&id) else {
                    continue;
                };
                if donor_by_name.contains_key(name.as_str()) {
                    continue;
                }
                let Some(fd) = dllbc.fn_by_id(id) else {
                    continue;
                };
                let Some(body) = fd.unstructured() else {
                    continue;
                };
                let Some(fact) = helper_body_fact(dllbc, &body, &push, &dcg.names) else {
                    continue;
                };
                // Only a helper whose return place is the pinned word. Importing
                // an argument-only pin (`RootedItems::push`) marks every later
                // read of that argument stale, and a pin set this artefact
                // cannot name makes the caller unread.
                // Donor callee ids are the donor's; `donor_collecting` is
                // `CallGraph::reaching` of the seeds in that id space, so a
                // wrapper such as `w_weakref_new` is included.
                if !returns_pinned_word(&fact, &donor_collecting, &HashSet::new()) {
                    continue;
                }
                donor_by_name.insert(name.as_str(), fact);
            }
        }
        for (&id, name) in &cg.names {
            if bodies.contains_key(&id) {
                continue;
            }
            let readable = llbc.fn_by_id(id).and_then(|fd| fd.unstructured()).is_some();
            if readable {
                continue;
            }
            if let Some(fact) = donor_by_name.get(name.as_str()) {
                bodies.insert(id, fact.clone());
            }
        }
    }
    let collecting = collecting_for_pin_summaries(cg);
    summarize_pin_helpers(&bodies, &collecting)
}

/// Functions that can collect, for pin-return summaries.
///
/// `CollectAnalyzer` is the transitive closure of operations that can malloc
/// GC. `CallGraph::reaching` of [`COLLECTING_SEEDS`](super::framework::COLLECTING_SEEDS)
/// is that set, so a wrapper such as `w_weakref_new` is included even though
/// it is not itself a seed.
fn collecting_for_pin_summaries(cg: &super::framework::CallGraph) -> HashSet<u64> {
    let (seeds, _) = cg.seeds_for(super::framework::COLLECTING_SEEDS);
    cg.reaching(&seeds)
}

/// True when `b` is in [`crate::front::mir::forward_reachable_mask`].
/// An `on_unwind`-only cleanup block is not.
fn is_forward(forward: &[bool], b: usize) -> bool {
    forward.get(b).copied().unwrap_or(false)
}

/// CFG successors of `t` as the flow graph sees them.
///
/// `Call` / `Assert` / `Drop` keep `target`. `on_unwind` is kept only when
/// that block is already forward-reachable — a normal edge also lands
/// there, the case `strip_cleanup_blocks` left unchanged. An unwind-only
/// cleanup is the terminal `UnwindResume` stub the stripped body had: no
/// statements, no calls, no further successors.
fn successors(t: &TermKind, forward: &[bool]) -> Vec<u64> {
    match t {
        TermKind::Goto { target } => vec![*target],
        TermKind::Switch { targets, .. } => match targets {
            SwitchTargets::If(a, b) => vec![*a, *b],
            SwitchTargets::SwitchInt(_, arms, d) => {
                let mut v: Vec<u64> = arms.iter().map(|(_, b)| *b).collect();
                v.push(*d);
                v
            }
        },
        TermKind::Call {
            target, on_unwind, ..
        }
        | TermKind::Assert {
            target, on_unwind, ..
        }
        | TermKind::Drop {
            target, on_unwind, ..
        } => {
            let mut v = vec![*target];
            if is_forward(forward, *on_unwind as usize) && *on_unwind != *target {
                v.push(*on_unwind);
            }
            v
        }
        _ => vec![],
    }
}

/// Report every call that can collect and carries an unrooted live GC pointer.
///
/// `push_roots` are the `gc_roots::push_roots` function ids.  Coverage is
/// judged **per call**, not per function: a call is withheld only when every
/// path to it runs through a `push_roots`, which is exactly "this block is
/// unreachable from entry once the bracket blocks are removed".  A function
/// that brackets one branch and leaves another bare is therefore still
/// reported on the bare branch.
///
/// `gc_tys` selects which pointer kind is being judged — [`gc_ptr_type_ids`]
/// for a managed reference, [`frame_ptr_type_ids`] for the running frame — and
/// `movable_callees` ranks the findings for that kind
/// ([`movable_callee_ids`], or empty where the kind has no such call).
pub fn scan(
    llbc: &majit_charon_reader::Llbc,
    cg: &super::framework::CallGraph,
    reach: &HashSet<u64>,
    push_roots: &HashSet<u64>,
    gc_tys: &HashSet<u64>,
    movable_callees: &HashSet<u64>,
    donors: &[(&majit_charon_reader::Llbc, &super::framework::CallGraph)],
) -> (Vec<Finding>, ScanStats) {
    let mut findings = Vec::new();
    let mut stats = ScanStats::default();
    // One summary for the artefact: a helper pins into the caller's open
    // scope, so which parameters it publishes is a property of the callee.
    let pin_helpers = pin_helper_summaries(llbc, cg, push_roots, donors);
    let root_scope_tys = root_scope_type_ids(llbc, push_roots);
    // `PyError` and `Result<_, PyError>` are the words `expand_pop_roots`
    // would reload. A pin roots the object; only a later reload freshens the
    // local. Computed once: the ids are a property of the artefact.
    let pyerror_ids = pyerror_type_ids(llbc);
    let mut pyerror_word_ids = pyerror_ids.clone();
    pyerror_word_ids.extend(result_pyerror_type_ids(llbc, &pyerror_ids));
    // Calls that read a slice's length or test a word against null.  A moved
    // object's stale address is still non-null, so neither answer changes.
    let metadata_fns: HashSet<u64> = cg
        .names
        .iter()
        .filter(|(_, n)| {
            matches!(
                n.as_str(),
                "core::slice::<Impl>::len"
                    | "core::slice::<Impl>::is_empty"
                    | "core::ptr::mut_ptr::<Impl>::is_null"
                    | "core::ptr::const_ptr::<Impl>::is_null"
            )
        })
        .map(|(id, _)| *id)
        .collect();
    // Drop glue is a property of the type, not of the local that holds it.
    // One answer per hash-cons id is reused across every body in the artefact.
    let drop_owners = explicit_drop_owners(llbc);
    let mut drop_memo: HashMap<u64, bool> = HashMap::new();
    for fd in llbc.iter_local_fns() {
        let id = fd.def_id;
        if !reach.contains(&id) {
            continue;
        }
        let Some(body) = fd.unstructured() else {
            continue;
        };
        // Locals whose static type is a GC pointer.  Nothing else can go stale.
        let gc_locals: HashMap<u64, String> = body
            .locals
            .locals
            .iter()
            .filter(|l| ty_id(&l.ty).is_some_and(|t| gc_tys.contains(&t)))
            .map(|l| {
                (
                    l.index,
                    l.name.clone().unwrap_or_else(|| format!("_{}", l.index)),
                )
            })
            .collect();
        if gc_locals.is_empty() {
            continue;
        }

        stats.bodies_scanned += 1;
        let n = body.body.len();
        // Cleanup reached only through `on_unwind` is outside the flow
        // graph. One mask per body; walks below reuse it.
        let forward = crate::front::mir::forward_reachable_mask(llbc, &body);
        let terms: Vec<Option<TermKind>> = body.body.iter().map(|b| b.term(llbc).ok()).collect();
        let unparsed_terms = terms.iter().enumerate().any(|(b, t)| {
            is_forward(&forward, b) && (t.is_none() || matches!(t, Some(TermKind::Unknown)))
        });
        if unparsed_terms {
            // Successors are unknown for that block, so every live set derived
            // from it is a lower bound.  Count the body; do not pretend it is
            // clean.
            stats.unparsed_terminator_bodies += 1;
        }
        let unparsed_stmts = body.body.iter().enumerate().any(|(b, blk)| {
            is_forward(&forward, b)
                && blk
                    .statements
                    .iter()
                    .any(|st| matches!(st.stmt_kind(), Err(_) | Ok(StmtKind::Unknown)))
        });
        if unparsed_stmts {
            // `transfer_stmt` reads no uses out of either shape, so the live
            // sets below can only be smaller than the truth.
            stats.unparsed_statement_bodies += 1;
        }

        // Blocks whose terminator opens a root scope, and the blocks that are
        // still reachable from entry without them — those are the ones no
        // bracket can dominate.
        let bracket_blocks: HashSet<usize> = (0..n)
            .filter(|&b| {
                is_forward(&forward, b)
                    && match &terms[b] {
                        Some(TermKind::Call { call, .. }) => {
                            opens_root_scope(&call.func, push_roots)
                        }
                        _ => false,
                    }
            })
            .collect();
        // Active root-scope locals at each block entry, one sorted set per
        // feasible path state.  Merely removing every opener block from the
        // CFG (the old implementation) made its bracket last until function
        // exit: a collecting call after `RootScope::drop` was silently read as
        // covered.  Charon's Drop place tells us exactly which guard ends.
        let mut active_at: Vec<HashSet<Vec<u64>>> = vec![HashSet::new(); n];
        if n > 0 {
            // A wrapper such as `WritableBuffer` already holds the acquire
            // pin in `_roots`; `BufferView.releasebuffer` has no keep-alive
            // of its own. Seed that guard so `drop` is not a nested scope.
            active_at[0].insert(incoming_root_scopes(&body, llbc, &root_scope_tys));
        }
        let mut scope_work = if n > 0 { vec![0usize] } else { Vec::new() };
        while let Some(b) = scope_work.pop() {
            let states: Vec<Vec<u64>> = active_at[b].iter().cloned().collect();
            let Some(term) = &terms[b] else {
                continue;
            };
            for state in states {
                let mut normal = state.clone();
                let mut unwind = state;
                match term {
                    TermKind::Call {
                        call,
                        target,
                        on_unwind,
                    } => {
                        let opens = opens_root_scope(&call.func, push_roots);
                        // Pushed rather than inserted in id order: a pin is
                        // rewound by the *innermost* live guard, because
                        // `RootScope::drop` truncates the shadow stack to the
                        // length it saved, so ownership needs nesting order.
                        if opens
                            && let Some(scope) = bare_local(&call.dest)
                            && !normal.contains(&scope)
                        {
                            normal.push(scope);
                        }
                        for (succ, next) in [(*target, normal), (*on_unwind, unwind)] {
                            let s = succ as usize;
                            // Unwind-only cleanup is the stripped body's
                            // `UnwindResume` stub: no statements, no further
                            // successors, so it is not entered.
                            if s < n && is_forward(&forward, s) && active_at[s].insert(next) {
                                scope_work.push(s);
                            }
                        }
                        continue;
                    }
                    // Truncated rather than removed, for the same reason the
                    // push above is a push: `RootScope::drop` truncates the
                    // shadow stack to the length the dropped guard saved, so
                    // dropping an outer guard retires every guard opened after
                    // it.  Removing only the named scope would leave the stack
                    // claiming a guard the runtime has already released.
                    TermKind::Drop { place, .. } => {
                        if drop_of_root_scope(place, &root_scope_tys)
                            && let Some(scope) = place_local(place)
                        {
                            if let Some(i) = normal.iter().position(|&s| s == scope) {
                                normal.truncate(i);
                            }
                            if let Some(i) = unwind.iter().position(|&s| s == scope) {
                                unwind.truncate(i);
                            }
                        }
                    }
                    _ => {}
                }
                for succ in successors(term, &forward) {
                    let s = succ as usize;
                    if s < n && is_forward(&forward, s) && active_at[s].insert(normal.clone()) {
                        scope_work.push(s);
                    }
                }
            }
        }
        let unbracketed: HashSet<usize> = (0..n)
            .filter(|&b| active_at[b].iter().any(Vec::is_empty))
            .collect();
        // The scope that owns a pin made at `b`: the innermost guard live
        // there.  `None` where the block is reached with two different scope
        // stacks, because then the owner is path-dependent and no single
        // answer is right.
        let owner_at: Vec<Option<u64>> = (0..n)
            .map(|b| {
                let mut it = active_at[b].iter();
                let first = it.next()?;
                let innermost = *first.last()?;
                for other in it {
                    if other.last() != Some(&innermost) {
                        return None;
                    }
                }
                Some(innermost)
            })
            .collect();
        // The one scope stack live at `b`, innermost last.  `None` where the
        // block is reached with two different stacks.
        let stack_at: Vec<Option<Vec<u64>>> = (0..n)
            .map(|b| {
                let mut it = active_at[b].iter();
                let first = it.next()?;
                if it.next().is_some() {
                    return None;
                }
                Some(first.clone())
            })
            .collect();
        // The scope local a scope-closing Drop names, so the dataflow can
        // retire that guard's pins and leave an enclosing guard's alone.
        let closed_scope: Vec<Option<u64>> = (0..n)
            .map(|b| {
                if !is_forward(&forward, b) {
                    return None;
                }
                match &terms[b] {
                    Some(TermKind::Drop { place, .. })
                        if drop_of_root_scope(place, &root_scope_tys) =>
                    {
                        place_local(place)
                            .filter(|scope| active_at[b].iter().any(|a| a.contains(scope)))
                    }
                    _ => None,
                }
            })
            .collect();
        let has_nested_scopes = active_at
            .iter()
            .flat_map(HashSet::iter)
            .any(|scopes| scopes.len() > 1);
        let term_closes_root_scope: Vec<bool> = (0..n)
            .map(|b| {
                if !is_forward(&forward, b) {
                    return false;
                }
                match &terms[b] {
                    Some(TermKind::Drop { place, .. })
                        if drop_of_root_scope(place, &root_scope_tys) =>
                    {
                        place_local(place)
                            .is_some_and(|scope| active_at[b].iter().any(|a| a.contains(&scope)))
                    }
                    _ => false,
                }
            })
            .collect();
        // Reachable from entry, brackets and all. `forward_reachable_mask`
        // is that set: `on_unwind` is not an edge.
        let reachable: HashSet<usize> = (0..n).filter(|&b| is_forward(&forward, b)).collect();

        // What each pin call names.  `bracket_blocks` says a scope is open; it
        // does not say what is in it, and a scope that pins the wrong set
        // silences this scan without protecting anything.
        //
        // A local assigned twice is not a chain worth following, so the map is
        // built over single-assignment locals only -- which is every temporary
        // `pin_roots(&[..])` lowers through.
        let PinAssignIndex {
            defs,
            mut_borrow_of,
            ..
        } = index_pin_assigns(&body.body, &terms, &forward);

        // A body whose pinned set cannot be read is not a body with an empty
        // one: grading it would turn "not understood" into "root missing".
        // The pin-set analysis below carries the owning guard alongside each
        // root local, so an inner Drop retires its own pins and leaves an
        // enclosing guard's alone; nesting on its own no longer withholds a
        // body.  This counts how many still hold two live scopes, because a
        // body that also trips one of the opacity tests below is unread for
        // that reason and the two populations are easy to confuse.
        if has_nested_scopes {
            stats.bodies_with_nested_scopes += 1;
        }
        let mut opaque_reason: Option<&'static str> = if unparsed_terms {
            Some("unparsed-terminator")
        } else if unparsed_stmts {
            Some("unparsed-statement")
        } else {
            None
        };
        let mut saw_pin_call = false;
        let mut term_pins: Vec<HashSet<u64>> = vec![HashSet::new(); n];
        // The locals a pin was *handed*, as distinct from the word it hands
        // back.  `pin_root` returns the normalized word because the publish is
        // itself a safepoint -- a foreign collection can forward the value
        // between the caller's copy and the query -- so a body that goes on
        // reading the local it passed in is reading a possible forwarding stub.
        let mut term_pin_args: Vec<HashSet<u64>> = vec![HashSet::new(); n];
        let mut term_pin_names: Vec<String> = vec![String::new(); n];
        for b in 0..n {
            if !is_forward(&forward, b) {
                continue;
            }
            let Some(TermKind::Call { call, .. }) = &terms[b] else {
                continue;
            };
            let CallFunc::Regular(reg) = &call.func else {
                continue;
            };
            let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
                continue;
            };
            let Some(name) = cg.names.get(id) else {
                continue;
            };
            if name.contains("gc_roots::<Impl>") && name.ends_with("::set") {
                // These overwrite an existing coloured slot.  Without a
                // slot→root map, retaining the old root or replacing the
                // wrong one could both claim false coverage.
                opaque_reason.get_or_insert("slot-set");
                continue;
            }
            // Reading a slot back yields the word the slot holds now, which
            // is rooted by whatever pinned it; the index it takes is not a
            // root, so only the result is read here.
            let reads_a_slot_back = reads_root_slot(name);
            if is_pyerror_handle_pin(name) {
                // `PyError::pin` pins the receiver through `&mut self` and
                // writes the forwarded word back. The helper summary does
                // not see that: `pin_root` is handed `self.0`, a field, so
                // `pinned_params` is empty. Name the receiver here so the
                // open bracket covers the handle, and leave pin-args empty
                // so the writeback is not a stale read of the pre-pin word.
                saw_pin_call = true;
                let mut pinned: HashSet<u64> = HashSet::new();
                if let Some(arg0) = call.args.first() {
                    let mut seed = HashSet::new();
                    use_operand(arg0, &mut seed);
                    for local in seed {
                        chase_handle_local(local, &mut_borrow_of, &defs, &mut pinned, 0);
                    }
                }
                pinned.retain(|local| gc_locals.contains_key(local));
                term_pin_args[b] = HashSet::new();
                term_pin_names[b] = name.clone();
                term_pins[b] = pinned;
                continue;
            }
            if !is_pin_fn(name) && !reads_a_slot_back {
                // A helper pins into this body's open scope. Its own fresh
                // object names no local here, so an empty set is not a pin
                // we failed to read.
                let Some(helper) = pin_helpers.get(id) else {
                    continue;
                };
                saw_pin_call = true;
                let mut pinned: HashSet<u64> = HashSet::new();
                for (i, arg) in call.args.iter().enumerate() {
                    if !helper.pinned_params.contains(&i) {
                        continue;
                    }
                    let mut seed: HashSet<u64> = HashSet::new();
                    use_operand(arg, &mut seed);
                    for local in seed {
                        chase_pinned(local, &defs, &mut pinned, 0);
                    }
                }
                let mut args_only = pinned.clone();
                if let Some(dest) = bare_local(&call.dest) {
                    if helper.returns_pinned {
                        pinned.insert(dest);
                    }
                    // `let obj = helper(obj)` rebinds. The destination is the
                    // word handed back only when the helper returns one;
                    // either way it is not an argument still being read.
                    args_only.remove(&dest);
                }
                if is_pyerror_handle_pin(name) {
                    // `PyError::pin` writes the forwarded word back through
                    // `&mut self`. The receiver stays in `pinned` so coverage
                    // still sees the root. It is not a pin argument: counting
                    // it there treats the writeback as a read of the pre-pin
                    // word. `shadowstack.py expand_pop_roots` reloads after
                    // the collecting call; this writeback is that reload for
                    // the one local the pin itself updated.
                    drop_pyerror_pin_receiver(&call.args, &mut_borrow_of, &defs, &mut args_only);
                }
                args_only.retain(|local| gc_locals.contains_key(local));
                term_pin_args[b] = args_only;
                term_pin_names[b] = name.clone();
                pinned.retain(|local| gc_locals.contains_key(local));
                if pinned.is_empty() && !helper.pinned_params.is_empty() {
                    opaque_reason.get_or_insert("pin-names-nothing");
                }
                term_pins[b] = pinned;
                continue;
            }
            saw_pin_call = true;
            let mut pinned: HashSet<u64> = HashSet::new();
            if !reads_a_slot_back {
                for a in &call.args {
                    let mut seed: HashSet<u64> = HashSet::new();
                    use_operand(a, &mut seed);
                    for l in seed {
                        chase_pinned(l, &defs, &mut pinned, 0);
                    }
                }
            }
            // What a pin hands back is the word now in the slot, so the local
            // it binds is rooted exactly as the argument was.  `pin_root` is
            // `#[must_use]` for that reason -- a collection may have forwarded
            // the value, and the returned word is the one the caller must go
            // on to use.  `let obj = pin_root(obj)` rebinds, so the local the
            // body uses afterwards is a *different* one from the local passed
            // in, and pinning only the argument would read the rebound name as
            // unrooted.
            let mut args_only = pinned.clone();
            if let Some(d) = bare_local(&call.dest) {
                pinned.insert(d);
                args_only.remove(&d);
            }
            args_only.retain(|l| gc_locals.contains_key(l));
            term_pin_args[b] = args_only;
            term_pin_names[b] = name.clone();
            pinned.retain(|l| gc_locals.contains_key(l));
            if pinned.is_empty() {
                // A pin that named nothing we could resolve is a pin we do not
                // understand, not one that pinned nothing.
                opaque_reason.get_or_insert("pin-names-nothing");
            }
            term_pins[b] = pinned;
        }
        // Locals already reported as a pin argument stay on that counter.
        // The post-collect walk below answers a different question: a rooted
        // `PyError` whose word was not reloaded after a collecting call.
        // `framework.py push_roots` / `pop_roots` reloads every live variable;
        // the original local is not that reloaded word.
        let pin_arg_union: HashSet<u64> = term_pin_args.iter().flatten().copied().collect();
        let address_of_local = address_bearing_locals(&body.body, &forward);
        let pyerror_locals: HashSet<u64> = body
            .locals
            .locals
            .iter()
            .filter(|l| ty_id(&l.ty).is_some_and(|t| pyerror_word_ids.contains(&t)))
            .map(|l| l.index)
            .collect();
        // A scope-closing Drop this reader cannot name retires an unknown set,
        // and a pin made where two scope stacks meet has a path-dependent
        // owner.  Either would let one guard's Drop release another's pins.
        // Blocks that neither pin nor close a scope need no owner.
        if (0..n).any(|b| {
            is_forward(&forward, b)
                && ((term_closes_root_scope[b]
                    && (closed_scope[b].is_none() || stack_at[b].is_none()))
                    || (!term_pins[b].is_empty() && owner_at[b].is_none()))
        }) {
            opaque_reason.get_or_insert("scope-owner-ambiguous");
        }
        if !bracket_blocks.is_empty() && !saw_pin_call {
            // A scope is open and nothing in this body names what went into
            // it. A helper that pins into this scope was already recorded
            // above; what remains opens its own scope, as `RootedItems` does.
            // An unread set, not an empty one.
            opaque_reason.get_or_insert("no-pin-in-body");
        }

        // Every block, including a non-forward cleanup, is a predecessor of
        // its terminator's successors. Jacobi intersected those outs — zero
        // when the predecessor is not forward — into each successor.
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
        for b in 0..n {
            if let Some(t) = &terms[b] {
                for s in successors(t, &forward) {
                    if (s as usize) < n {
                        preds[s as usize].push(b);
                    }
                }
            }
        }
        // Statement-order pin transfer. A copy is `dest = src` or a cast of
        // that local: it clears `dest` and writes `src`'s bits, so `a = a`
        // keeps the pin. A projected load is a new pointer and kills
        // `dest`. The call destination is killed after these statements,
        // because `gct_direct_call` reads livevars at the collecting op
        // (`push_roots`), after the block's statements and before the call
        // overwrites its destination.
        enum RawPinXfer {
            Kill(u64),
            Copy { src: u64, dest: u64 },
        }
        let mut stmt_xfers: Vec<Vec<RawPinXfer>> = (0..n).map(|_| Vec::new()).collect();
        for (b, blk) in body.body.iter().enumerate() {
            if !is_forward(&forward, b) {
                continue;
            }
            for st in &blk.statements {
                match st.stmt_kind() {
                    Ok(StmtKind::Assign(place, rvalue)) => {
                        if let Some(d) = bare_local(&place) {
                            if let Some(src) = pin_copy_src(&rvalue) {
                                stmt_xfers[b].push(RawPinXfer::Copy { src, dest: d });
                            } else {
                                stmt_xfers[b].push(RawPinXfer::Kill(d));
                            }
                        }
                    }
                    Ok(StmtKind::StorageLive(i)) | Ok(StmtKind::StorageDead(i)) => {
                        stmt_xfers[b].push(RawPinXfer::Kill(i));
                    }
                    _ => {}
                }
            }
        }
        // Forward *must*-analysis: a root counts as pinned at a call only when
        // every path reaching it pinned that local and nothing has reassigned
        // it since.  Starts at the universe and intersects down, so a block
        // whose predecessors disagree keeps only what they all hold.
        // Each element carries the guard that owns it, so a Drop retires that
        // guard's pins and leaves an enclosing guard's in place.  The owner is
        // the innermost scope live where the pin was made.
        let all_scopes: HashSet<u64> = active_at
            .iter()
            .flat_map(HashSet::iter)
            .flat_map(|a| a.iter().copied())
            .collect();
        // The element is a subset of `locals x scopes` and the entry value is
        // that whole product, so a `HashSet` of pairs per block costs
        // `blocks * locals * scopes` hash entries -- for the largest
        // interpreter bodies, gigabytes, three times over per round because
        // each round rebuilt both block-indexed vectors.  The same answer is
        // `blocks * locals * scopes` **bits**.
        //
        // Bit `l * scopes.len() + s`, so one local's owners are contiguous:
        // clearing every owner of a local is what a kill and a call
        // destination each do, while retiring a scope is rarer and takes a
        // strided mask built once.
        let mut scopes: Vec<u64> = all_scopes.iter().copied().collect();
        scopes.sort_unstable();
        // The domain carries every local a pin *names*, not only the GC ones:
        // a terminator adds `(local, owner)` for whatever it pinned.  Only the
        // GC locals seed the entry value, which is what starting from
        // `gc_locals x scopes` meant.
        let mut locals: Vec<u64> = gc_locals.keys().copied().collect();
        for pins in &term_pins {
            locals.extend(pins.iter().copied());
        }
        locals.sort_unstable();
        locals.dedup();
        let local_at: HashMap<u64, usize> =
            locals.iter().enumerate().map(|(i, &l)| (l, i)).collect();
        let scope_at: HashMap<u64, usize> =
            scopes.iter().enumerate().map(|(i, &x)| (x, i)).collect();
        let nscopes = scopes.len();
        let words = (locals.len() * nscopes).div_ceil(64);
        let bit_set = |v: &mut [u64], i: usize| v[i / 64] |= 1u64 << (i % 64);
        let bit_get = |v: &[u64], i: usize| v[i / 64] >> (i % 64) & 1 == 1;
        // One scope's pairs, which are strided and so need a mask; a local's
        // are a run and are cleared in place.
        let scope_masks: Vec<Vec<u64>> = (0..nscopes)
            .map(|si| {
                let mut m = vec![0u64; words];
                for li in 0..locals.len() {
                    bit_set(&mut m, li * nscopes + si);
                }
                m
            })
            .collect();
        // What each block does to the set, resolved once so the rounds below
        // are word operations. Statement transfers stay in order. The call
        // destination is killed after them: a copy into that local must not
        // restore a pin the call is about to overwrite.
        enum PinXfer {
            Kill(usize),
            Copy { src: usize, dest: usize },
        }
        struct BlockEdit {
            stmts: Vec<PinXfer>,
            call_dest_kill: Option<usize>,
            retire_scopes: Vec<usize>,
            retire_all: bool,
            add_bits: Vec<usize>,
        }
        let edits: Vec<BlockEdit> = (0..n)
            .map(|b| {
                if !is_forward(&forward, b) {
                    return BlockEdit {
                        stmts: Vec::new(),
                        call_dest_kill: None,
                        retire_scopes: Vec::new(),
                        retire_all: false,
                        add_bits: Vec::new(),
                    };
                }
                // A tracked destination always drops its previous pin. It
                // gains one only when the source is a tracked bare local
                // (`gct_direct_call` livevars).
                let stmts: Vec<PinXfer> = stmt_xfers[b]
                    .iter()
                    .filter_map(|xfer| match *xfer {
                        RawPinXfer::Kill(local) => local_at.get(&local).copied().map(PinXfer::Kill),
                        RawPinXfer::Copy { src, dest } => {
                            let dest = *local_at.get(&dest)?;
                            match local_at.get(&src) {
                                Some(&src) => Some(PinXfer::Copy { src, dest }),
                                None => Some(PinXfer::Kill(dest)),
                            }
                        }
                    })
                    .collect();
                let mut call_dest_kill = None;
                let mut retire_scopes: Vec<usize> = Vec::new();
                let mut retire_all = false;
                if term_closes_root_scope[b] {
                    // `RootScope::drop` truncates the shadow stack to the
                    // length it saved, so a guard takes its own pins *and*
                    // those of every guard opened after it -- dropping out of
                    // order takes more than one scope's worth.  An enclosing
                    // guard keeps its own.
                    match (closed_scope[b], stack_at[b].as_ref()) {
                        (Some(scope), Some(stack)) => {
                            let from = stack.iter().position(|&x| x == scope).unwrap_or(0);
                            retire_scopes = stack[from..]
                                .iter()
                                .filter_map(|x| scope_at.get(x).copied())
                                .collect();
                        }
                        // A Drop this reader cannot place retires an unknown
                        // set.
                        _ => retire_all = true,
                    }
                }
                if let Some(TermKind::Call { call, .. }) = &terms[b]
                    && let Some(d) = bare_local(&call.dest)
                    && let Some(&di) = local_at.get(&d)
                {
                    call_dest_kill = Some(di);
                }
                let add_bits = match owner_at[b].and_then(|o| scope_at.get(&o).copied()) {
                    Some(si) => term_pins[b]
                        .iter()
                        .filter_map(|l| local_at.get(l).map(|&li| li * nscopes + si))
                        .collect(),
                    None => Vec::new(),
                };
                BlockEdit {
                    stmts,
                    call_dest_kill,
                    retire_scopes,
                    retire_all,
                    add_bits,
                }
            })
            .collect();
        // `PinXfer` is local to this analysis, so the transfer lives in
        // closures rather than free functions. A copy saves the source bits
        // before clearing the destination, which is what keeps `a = a`.
        let clear_local_pins = |bits: &mut [u64], li: usize| {
            for si in 0..nscopes {
                let i = li * nscopes + si;
                bits[i / 64] &= !(1u64 << (i % 64));
            }
        };
        let apply_pin_statements = |bits: &mut [u64], stmts: &[PinXfer]| {
            for xfer in stmts {
                match *xfer {
                    PinXfer::Kill(li) => clear_local_pins(bits, li),
                    PinXfer::Copy { src, dest } => {
                        if src == dest {
                            continue;
                        }
                        let mut saved = vec![0u8; nscopes.div_ceil(8)];
                        for si in 0..nscopes {
                            let i = src * nscopes + si;
                            if bits[i / 64] >> (i % 64) & 1 == 1 {
                                saved[si / 8] |= 1 << (si % 8);
                            }
                        }
                        clear_local_pins(bits, dest);
                        for si in 0..nscopes {
                            if saved[si / 8] & (1 << (si % 8)) != 0 {
                                let i = dest * nscopes + si;
                                bits[i / 64] |= 1u64 << (i % 64);
                            }
                        }
                    }
                }
            }
        };
        // Block-indexed in-sets. Entry, unreachable, and predecessor-less
        // blocks stay empty. Every other forward block starts at the
        // universe and shrinks by intersecting predecessor outs. The meet
        // is monotone, so a worklist that reapplies a smaller out reaches
        // the same greatest fixpoint as a round of every block at once.
        // A long MIR chain otherwise needs one synchronous round per hop.
        let mut pinned_in = vec![0u64; n * words];
        for b in 0..n {
            if b == 0 || !reachable.contains(&b) || preds[b].is_empty() {
                continue;
            }
            let blk = &mut pinned_in[b * words..(b + 1) * words];
            for (li, l) in locals.iter().enumerate() {
                if !gc_locals.contains_key(l) {
                    continue;
                }
                for si in 0..nscopes {
                    bit_set(blk, li * nscopes + si);
                }
            }
        }
        let mut succs: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (s, ps) in preds.iter().enumerate() {
            for &p in ps {
                succs[p].push(s);
            }
        }
        let mut queued = vec![false; n];
        let mut work = VecDeque::new();
        // Every block is processed at least once so a non-forward
        // predecessor's zero out-set meets each successor. Requeue is
        // shrink-only, and those in-sets start empty and never shrink.
        for b in 0..n {
            queued[b] = true;
            work.push_back(b);
        }
        let mut out = vec![0u64; words];
        while let Some(b) = work.pop_front() {
            queued[b] = false;
            if words == 0 {
                continue;
            }
            out.copy_from_slice(&pinned_in[b * words..(b + 1) * words]);
            if is_forward(&forward, b) {
                let e = &edits[b];
                apply_pin_statements(&mut out, &e.stmts);
                if let Some(li) = e.call_dest_kill {
                    clear_local_pins(&mut out, li);
                }
                if e.retire_all {
                    out.fill(0);
                } else {
                    for &si in &e.retire_scopes {
                        for (w, m) in out.iter_mut().zip(&scope_masks[si]) {
                            *w &= !m;
                        }
                    }
                }
                for &i in &e.add_bits {
                    bit_set(&mut out, i);
                }
            } else {
                out.fill(0);
            }
            for &s in &succs[b] {
                if s == 0 || !reachable.contains(&s) {
                    continue;
                }
                let sat = s * words;
                let mut shrunk = false;
                for w in 0..words {
                    let v = pinned_in[sat + w] & out[w];
                    if v != pinned_in[sat + w] {
                        pinned_in[sat + w] = v;
                        shrunk = true;
                    }
                }
                if shrunk && !queued[s] {
                    queued[s] = true;
                    work.push_back(s);
                }
            }
        }

        let mut live_in: Vec<HashSet<u64>> = vec![HashSet::new(); n];
        // Backward liveness to a fixed point.  The bodies are small; a plain
        // worklist over predecessors converges in a handful of rounds.
        let mut changed = true;
        while changed {
            changed = false;
            for b in (0..n).rev() {
                if !is_forward(&forward, b) {
                    continue;
                }
                let Some(t) = &terms[b] else { continue };
                let mut live: HashSet<u64> = HashSet::new();
                for s in successors(t, &forward) {
                    if let Some(sl) = live_in.get(s as usize) {
                        live.extend(sl.iter().copied());
                    }
                }
                transfer_term(
                    t,
                    &mut live,
                    &metadata_fns,
                    llbc,
                    &drop_owners,
                    &mut drop_memo,
                );
                for st in body.body[b].statements.iter().rev() {
                    if let Ok(k) = st.stmt_kind() {
                        transfer_stmt(&k, &mut live, &gc_locals);
                    }
                }
                live.retain(|l| gc_locals.contains_key(l));
                if live != live_in[b] {
                    live_in[b] = live;
                    changed = true;
                }
            }
        }

        // Locals this body hands to a callee that addresses a nursery-allocated
        // kind.  Built once: it does not depend on which collecting call is
        // being judged, so a local that reaches such a callee anywhere in the
        // body is in the set for every call the body makes.
        let mut movable_args: HashSet<u64> = HashSet::new();
        for (b, other) in body.body.iter().enumerate() {
            if !is_forward(&forward, b) {
                continue;
            }
            let Ok(TermKind::Call { call: c2, .. }) = other.term(llbc) else {
                continue;
            };
            let CallFunc::Regular(r2) = &c2.func else {
                continue;
            };
            let CallKind::Fun(FunId::Regular { id: cid }) = &r2.kind else {
                continue;
            };
            if !movable_callees.contains(cid) {
                continue;
            }
            for a in &c2.args {
                let mut seed: HashSet<u64> = HashSet::new();
                use_operand(a, &mut seed);
                for l in seed {
                    chase_arg_aliases(l, &defs, &mut movable_args, 0);
                }
            }
        }
        // Count only the GC pointers: a `Wtf8Buf` or a capacity handed to a
        // marked callee is an argument the `movable` filters can never select,
        // so counting it here would report a live ranking that is not one.
        let movable_gc_args = movable_args
            .iter()
            .filter(|l| gc_locals.contains_key(*l))
            .count();
        if movable_gc_args > 0 {
            stats.bodies_with_movable_args += 1;
            stats.movable_arg_locals += movable_gc_args;
        }

        // Every pin whose argument the body goes on to read.  Independent of
        // the bracket question above: this is not a missing root but a stale
        // *word*, and the pin call is itself where the forwarding could have
        // happened.  `#[must_use]` on `pin_root` puts the choice in the open --
        // use the returned word, or write `let _ =` and thereby claim the kind
        // never moves.  That claim is what this checks.
        for b in 0..n {
            if !is_forward(&forward, b) {
                continue;
            }
            if term_pin_args[b].is_empty() {
                continue;
            }
            let Some(TermKind::Call {
                target, on_unwind, ..
            }) = &terms[b]
            else {
                continue;
            };
            let mut after: HashSet<u64> = HashSet::new();
            for s in [*target, *on_unwind] {
                let s = s as usize;
                if !is_forward(&forward, s) {
                    continue;
                }
                if let Some(sl) = live_in.get(s) {
                    after.extend(sl.iter().copied());
                }
            }
            let still: Vec<u64> = term_pin_args[b]
                .iter()
                .filter(|l| after.contains(l))
                .copied()
                .collect();
            if still.is_empty() {
                continue;
            }
            stats.pin_arg_read_after += 1;
            let mut movable: Vec<String> = still
                .iter()
                .filter(|l| movable_args.contains(l))
                .map(|l| gc_locals[l].clone())
                .collect();
            if !movable.is_empty() {
                stats.pin_arg_read_after_movable += 1;
            }
            let mut locals: Vec<String> = still.iter().map(|l| gc_locals[l].clone()).collect();
            locals.sort();
            movable.sort();
            let at = term_span(llbc, &body.body[b].terminator.span, &fd.item_meta.span);
            stats.stale_pin_reads.push(StalePinRead {
                func_name: fd.item_meta.name_path(),
                file: llbc.file_path(at.file_id).unwrap_or_default().to_string(),
                line: at.beg.line,
                pin_name: term_pin_names[b].clone(),
                locals,
                movable,
            });
        }

        // Now re-walk, and at every collecting Call read the live-after set.
        for (b, bb) in body.body.iter().enumerate() {
            if !is_forward(&forward, b) {
                continue;
            }
            let Some(TermKind::Call {
                call,
                target,
                on_unwind,
            }) = &terms[b]
            else {
                continue;
            };
            let CallFunc::Regular(reg) = &call.func else {
                continue;
            };
            let CallKind::Fun(FunId::Regular { id: callee }) = &reg.kind else {
                continue;
            };
            let mut callee_node = cg.callee_node(llbc, &[], *callee, &reg.generics);
            // A spec copy that can collect stays the generic FunDecl, the
            // scan-side half of `CallGraph::demote_specializations_in`.
            if callee_node != *callee && reach.contains(&callee_node) {
                callee_node = *callee;
            }
            if !reach.contains(&callee_node) {
                continue;
            }
            let mut after: HashSet<u64> = HashSet::new();
            for s in [*target, *on_unwind] {
                let s = s as usize;
                if !is_forward(&forward, s) {
                    continue;
                }
                if let Some(sl) = live_in.get(s) {
                    after.extend(sl.iter().copied());
                }
            }
            if let Some(d) = bare_local(&call.dest) {
                after.remove(&d);
            }
            for a in &call.args {
                let mut used: HashSet<u64> = HashSet::new();
                use_operand(a, &mut used);
                for t in used {
                    if let Some(l) = mut_borrow_of.get(&t) {
                        after.remove(l);
                    }
                }
            }
            after.retain(|l| gc_locals.contains_key(l));
            // One span answers every column below, and it is the
            // terminator's own: the call being reported *is* the terminator,
            // so the block's last statement names a line that runs after it.
            // A terminator with no span at all falls back to the function's.
            let at = term_span(llbc, &bb.terminator.span, &fd.item_meta.span);
            if !unbracketed.contains(&b) {
                stats.withheld_under_a_bracket += 1;
                // Withholding the call is right -- a bracket does dominate it
                // -- but "a bracket is open" and "this root is in it" are two
                // questions, and only the first was ever asked.  Grade the
                // second here so a bracket that pins the wrong set stops
                // reading as coverage.
                if opaque_reason.is_some() || !reachable.contains(&b) {
                    stats.withheld_contents_opaque += 1;
                    let reason = opaque_reason.unwrap_or("unreachable-block");
                    *stats.opaque_by_reason.entry(reason).or_default() += 1;
                    let fname = fd.item_meta.name_path();
                    match stats.opaque_bodies.last_mut() {
                        Some(last) if last.0 == fname => last.3 += 1,
                        _ => stats.opaque_bodies.push((
                            fname,
                            llbc.file_path(at.file_id).unwrap_or_default().to_string(),
                            reason,
                            1,
                        )),
                    }
                    if has_nested_scopes {
                        stats.withheld_opaque_from_nested += 1;
                    }
                    continue;
                }
                // Drop the owner here: coverage asks only whether the local
                // is pinned by *some* live guard.
                let blk = &pinned_in[b * words..(b + 1) * words];
                // Pins at the call, after this block's statements and the
                // call-destination kill. Not the block-exit set: that also
                // retires scopes and adds the terminator's own pins.
                let mut at_call = blk.to_vec();
                apply_pin_statements(&mut at_call, &edits[b].stmts);
                if let Some(li) = edits[b].call_dest_kill {
                    clear_local_pins(&mut at_call, li);
                }
                let held: HashSet<u64> = locals
                    .iter()
                    .enumerate()
                    .filter(|(li, _)| (0..nscopes).any(|si| bit_get(&at_call, li * nscopes + si)))
                    .map(|(_, l)| *l)
                    .collect();
                let mut missing: Vec<String> = after
                    .iter()
                    .filter(|l| !held.contains(l))
                    .map(|l| gc_locals[l].clone())
                    .collect();
                // Held across the call means the object is rooted. It does not
                // mean the local still holds the forwarded word.
                // `gc_restore_root` is the reload; a read of the pre-call
                // local after this point is the stale word.
                let watched: HashSet<u64> = after
                    .iter()
                    .copied()
                    .filter(|l| {
                        held.contains(l) && pyerror_locals.contains(l) && !pin_arg_union.contains(l)
                    })
                    .collect();
                if !watched.is_empty() {
                    let flagged = pyerror_stale_after_collect(
                        &body.body,
                        &terms,
                        &cg.names,
                        &mut_borrow_of,
                        &defs,
                        &address_of_local,
                        b,
                        &watched,
                        &forward,
                    );
                    if !flagged.is_empty() {
                        stats.pin_arg_read_after += 1;
                        let mut movable: Vec<String> = flagged
                            .iter()
                            .filter(|l| movable_args.contains(l))
                            .filter_map(|l| gc_locals.get(l).cloned())
                            .collect();
                        if !movable.is_empty() {
                            stats.pin_arg_read_after_movable += 1;
                        }
                        let mut locals_named: Vec<String> = flagged
                            .iter()
                            .filter_map(|l| gc_locals.get(l).cloned())
                            .collect();
                        locals_named.sort();
                        movable.sort();
                        stats.stale_pin_reads.push(StalePinRead {
                            func_name: fd.item_meta.name_path(),
                            file: llbc.file_path(at.file_id).unwrap_or_default().to_string(),
                            line: at.beg.line,
                            pin_name: cg.names.get(callee).cloned().unwrap_or_default(),
                            locals: locals_named,
                            movable,
                        });
                    }
                }
                if missing.is_empty() {
                    stats.withheld_bracket_covers += 1;
                    continue;
                }
                // Locals `1..=arg_count` are this body's parameters; 0 is the
                // return place.
                let params = 1..=body.locals.arg_count;
                let mut missing_local: Vec<String> = after
                    .iter()
                    .filter(|l| !held.contains(l) && !params.contains(*l))
                    .map(|l| gc_locals[l].clone())
                    .collect();
                // A caller's pin keeps the object alive; it does not keep
                // *this* body's copy correct.  A collection rewrites the slot
                // the caller reads back from, not the callee's local, so a
                // missing root of a kind that relocates is stale here however
                // well the caller bracketed it.  `build_class_inner` says as
                // much the other way round -- its borrowed arguments are safe
                // because they are tuples, which never move.
                let mut missing_movable: Vec<String> = after
                    .iter()
                    .filter(|l| !held.contains(l) && movable_args.contains(l))
                    .map(|l| gc_locals[l].clone())
                    .collect();
                missing.sort();
                missing_local.sort();
                missing_movable.sort();
                if !missing_local.is_empty() {
                    stats.withheld_bracket_short_body_local += 1;
                }
                if !missing_movable.is_empty() {
                    stats.withheld_bracket_short_movable += 1;
                }
                let mut pinned: Vec<String> = held
                    .iter()
                    .filter_map(|l| gc_locals.get(l).cloned())
                    .collect();
                pinned.sort();
                stats.withheld_bracket_short += 1;
                stats.short_brackets.push(ShortBracket {
                    func_name: fd.item_meta.name_path(),
                    file: llbc.file_path(at.file_id).unwrap_or_default().to_string(),
                    line: at.beg.line,
                    callee_name: cg.names.get(callee).cloned().unwrap_or_default(),
                    missing,
                    missing_local,
                    missing_movable,
                    pinned,
                });
                continue;
            }
            if after.is_empty() {
                continue;
            }
            let mut args: HashSet<u64> = HashSet::new();
            for a in &call.args {
                use_operand(a, &mut args);
            }
            let mut non_arg: Vec<String> = after
                .iter()
                .filter(|l| !args.contains(l))
                .map(|l| gc_locals[l].clone())
                .collect();
            let mut in_arg: Vec<String> = after
                .iter()
                .filter(|l| args.contains(l))
                .map(|l| gc_locals[l].clone())
                .collect();
            non_arg.sort();
            in_arg.sort();
            // Does any live pointer reach a movable-addressing callee in this
            // body?  Anywhere in the body, not only in the dominated
            // successors — this is a ranking signal, not a proof.
            let mut movable_use: Vec<String> = after
                .iter()
                .filter(|l| movable_args.contains(l))
                .map(|l| gc_locals[l].clone())
                .collect();
            movable_use.sort();
            // One span answers both columns, and it is the terminator's
            // own: the call being reported *is* the terminator, so the
            // block's last statement names a line that runs after it.
            // A terminator with no span at all falls back to the function's.
            let at = term_span(llbc, &bb.terminator.span, &fd.item_meta.span);
            findings.push(Finding {
                func: id,
                func_name: fd.item_meta.name_path(),
                file: llbc.file_path(at.file_id).unwrap_or_default().to_string(),
                line: at.beg.line,
                callee_id: *callee,
                callee_name: cg.names.get(callee).cloned().unwrap_or_default(),
                live_non_arg: non_arg,
                live_arg: in_arg,
                movable_use,
            });
        }
    }
    (findings, stats)
}

fn transfer_stmt(k: &StmtKind, live: &mut HashSet<u64>, tracked: &HashMap<u64, String>) {
    match k {
        StmtKind::Assign(p, r) => {
            if let Some(d) = bare_local(p) {
                // `_t = &*args` / `_t = copy x` into a tracked local reads `x`
                // only if `_t` is read later: the reborrow a `args.len()` call
                // takes is dead once the length is out.  Only for a tracked
                // destination -- an untracked one is never in `live`, so its
                // later reads are invisible here.
                let pure = matches!(r, Rvalue::Use(..) | Rvalue::Ref { .. } | Rvalue::Cast(..));
                let was_live = live.remove(&d);
                if pure && tracked.contains_key(&d) && !was_live {
                    return;
                }
            } else if let Some(l) = place_local(p) {
                live.insert(l);
            }
            use_rvalue(r, live);
        }
        StmtKind::StorageLive(i) | StmtKind::StorageDead(i) => {
            live.remove(i);
        }
        StmtKind::Assert(a) => use_operand(&a.cond, live),
        // `PlaceMention` is a borrowck marker, not a read.
        // `get_livevars_for_roots` stops at the last real use.
        StmtKind::PlaceMention(_) | StmtKind::Borrowck(_) | StmtKind::Unknown => {}
    }
}

fn term_span<'a>(
    llbc: &'a majit_charon_reader::Llbc,
    term: &'a Option<majit_charon_reader::ullbc::SpanRef>,
    item: &'a majit_charon_reader::ullbc::SpanRef,
) -> &'a majit_charon_reader::ullbc::SpanData {
    let chosen = match term {
        Some(span) => span,
        None => item,
    };
    llbc.span_data(chosen)
        .expect("span id is not in the artefact span table")
}

fn transfer_term(
    t: &TermKind,
    live: &mut HashSet<u64>,
    metadata_fns: &HashSet<u64>,
    llbc: &majit_charon_reader::Llbc,
    drop_owners: &HashSet<u64>,
    drop_memo: &mut HashMap<u64, bool>,
) {
    match t {
        TermKind::Call { call, .. } => {
            if let Some(d) = bare_local(&call.dest) {
                live.remove(&d);
            } else if let Some(l) = place_local(&call.dest) {
                live.insert(l);
            }
            let metadata_only = matches!(
                &call.func,
                CallFunc::Regular(reg)
                    if matches!(&reg.kind, CallKind::Fun(FunId::Regular { id }) if metadata_fns.contains(id))
            );
            if metadata_only {
                return;
            }
            for a in &call.args {
                use_operand(a, live);
            }
        }
        TermKind::Switch { discr, .. } => use_operand(discr, live),
        TermKind::Assert { assert, .. } => use_operand(&assert.cond, live),
        // A `Drop` reads the local only when its drop glue runs a user
        // `Drop` impl. Upstream livevars stop at the last real use; glue
        // that does not run user code is not one.
        TermKind::Drop { place, .. } => {
            if let Some(l) = place_local(place)
                && ty_may_run_user_drop(&place.ty, llbc, drop_owners, drop_memo, &[], &[], 0)
            {
                live.insert(l);
            }
        }
        _ => {}
    }
}

/// Self types of `core::ops::drop::Drop` impls in this artefact.
///
/// An impl whose trait decl is missing counts too: that is the same
/// "cannot prove dropless" answer as `type_decl_has_explicit_drop`.
fn explicit_drop_owners(llbc: &majit_charon_reader::Llbc) -> HashSet<u64> {
    let mut owners = HashSet::new();
    for row in llbc.trait_impls_raw() {
        let Some(impl_trait) = row.get("impl_trait") else {
            continue;
        };
        let Some(def_id) = impl_trait
            .get("generics")
            .and_then(|generics| generics.get("types"))
            .and_then(serde_json::Value::as_array)
            .and_then(|types| types.first())
            .and_then(|owner| impl_self_def_id(llbc, owner))
        else {
            continue;
        };
        let drop_or_unknown = impl_trait
            .get("id")
            .and_then(serde_json::Value::as_u64)
            .and_then(|trait_id| llbc.trait_by_id(trait_id))
            .is_none_or(|decl| decl.item_meta.name_path() == "core::ops::drop::Drop");
        if drop_or_unknown {
            owners.insert(def_id);
        }
    }
    owners
}

/// Nominal ADT def id of an impl's first type argument.
///
/// `Value: [id, body]` and `Deduplicated: id` are the two spellings
/// `resolve_tyexpr_to_adt_def_id_free` accepts. Tuple and `str` have no
/// nominal owner; `Box` does.
fn impl_self_def_id(llbc: &majit_charon_reader::Llbc, ty: &serde_json::Value) -> Option<u64> {
    if let Some(pair) = ty.get("Value").and_then(serde_json::Value::as_array)
        && let Some(body) = pair.get(1)
    {
        return body.get("Adt").and_then(adt_nominal_def_id);
    }
    if let Some(id) = ty.get("Deduplicated").and_then(serde_json::Value::as_u64) {
        return llbc.dedup_to_adt_def_id(id);
    }
    ty.get("Adt").and_then(adt_nominal_def_id)
}

fn adt_nominal_def_id(adt: &serde_json::Value) -> Option<u64> {
    match adt.get("builtin").and_then(serde_json::Value::as_str) {
        None | Some("Box") => adt.get("id").and_then(serde_json::Value::as_u64),
        Some(_) => None,
    }
}

fn ty_body<'a>(
    ty: &'a TyRef,
    llbc: &'a majit_charon_reader::Llbc,
) -> Option<&'a serde_json::Value> {
    match ty {
        TyRef::Inline { value: (_, body) } => Some(body),
        TyRef::Other(body) => Some(body),
        TyRef::Dedup { id } => llbc.dedup_body(*id),
    }
}

/// Whether dropping `ty` can run a user `Drop` impl.
///
/// `subst` binds type variables of `ty`; `outer` binds type variables that
/// occur inside those arguments. A generic parameter with no binding, a
/// trait object, a missing decl, or a walk deeper than 16 types counts.
/// The memo is the hash-cons id of a closed type, so two instantiations of
/// one ADT do not share an answer. It is not keyed by a local.
fn ty_may_run_user_drop(
    ty: &TyRef,
    llbc: &majit_charon_reader::Llbc,
    owners: &HashSet<u64>,
    memo: &mut HashMap<u64, bool>,
    subst: &[serde_json::Value],
    outer: &[serde_json::Value],
    depth: usize,
) -> bool {
    let closed = subst.is_empty() && outer.is_empty();
    let key = if closed { ty_id(ty) } else { None };
    if let Some(id) = key
        && let Some(answer) = memo.get(&id)
    {
        return *answer;
    }
    if let Some(id) = key {
        // A cycle of closed types has no user destructor of its own.
        memo.insert(id, false);
    }
    let answer = match ty_body(ty, llbc) {
        Some(body) => value_may_run_user_drop(body, llbc, owners, memo, subst, outer, depth),
        None => true,
    };
    if let Some(id) = key {
        memo.insert(id, answer);
    }
    answer
}

fn value_may_run_user_drop(
    node: &serde_json::Value,
    llbc: &majit_charon_reader::Llbc,
    owners: &HashSet<u64>,
    memo: &mut HashMap<u64, bool>,
    subst: &[serde_json::Value],
    outer: &[serde_json::Value],
    depth: usize,
) -> bool {
    if depth > 16 {
        return true;
    }
    let closed = subst.is_empty() && outer.is_empty();
    let mut node = node;
    let mut memo_id = None;
    let mut peeled = false;
    for _ in 0..24 {
        if let Some(id) = node.get("Deduplicated").and_then(serde_json::Value::as_u64) {
            if closed {
                if let Some(answer) = memo.get(&id).copied() {
                    if let Some(outer_id) = memo_id {
                        memo.insert(outer_id, answer);
                    }
                    return answer;
                }
                memo.insert(id, false);
                memo_id = Some(id);
            }
            match llbc.dedup_body(id) {
                Some(body) => node = body,
                None => {
                    if let Some(id) = memo_id {
                        memo.insert(id, true);
                    }
                    return true;
                }
            }
            continue;
        }
        if let Some(pair) = node.get("Value").and_then(serde_json::Value::as_array)
            && pair.len() == 2
        {
            if closed
                && memo_id.is_none()
                && let Some(id) = pair[0].as_u64()
            {
                if let Some(answer) = memo.get(&id).copied() {
                    return answer;
                }
                memo.insert(id, false);
                memo_id = Some(id);
            }
            node = &pair[1];
            continue;
        }
        peeled = true;
        break;
    }
    let answer = if peeled {
        type_node_may_run_user_drop(node, llbc, owners, memo, subst, outer, depth)
    } else {
        true
    };
    if let Some(id) = memo_id {
        memo.insert(id, answer);
    }
    answer
}

fn type_node_may_run_user_drop(
    node: &serde_json::Value,
    llbc: &majit_charon_reader::Llbc,
    owners: &HashSet<u64>,
    memo: &mut HashMap<u64, bool>,
    subst: &[serde_json::Value],
    outer: &[serde_json::Value],
    depth: usize,
) -> bool {
    if node.as_str() == Some("Never")
        || node.get("Scalar").is_some()
        || node.get("Ref").is_some()
        || node.get("RawPtr").is_some()
        || node.get("FnDef").is_some()
        || node.get("FnPtr").is_some()
    {
        return false;
    }
    if let Some(type_var) = node.get("TypeVar") {
        let Some(index) = typevar_index(type_var) else {
            return true;
        };
        let Some(arg) = subst.get(index) else {
            return true;
        };
        return value_may_run_user_drop(arg, llbc, owners, memo, outer, &[], depth + 1);
    }
    if node.get("DynTrait").is_some() || node.get("Dynamic").is_some() {
        return true;
    }
    if let Some(adt) = node.get("Adt") {
        return adt_may_run_user_drop(adt, llbc, owners, memo, subst, depth);
    }
    if let Some(element) = node
        .get("Array")
        .or_else(|| node.get("Slice"))
        .and_then(serde_json::Value::as_array)
        .and_then(|parts| parts.first())
    {
        return value_may_run_user_drop(element, llbc, owners, memo, subst, outer, depth + 1);
    }
    true
}

/// Index of a `TypeVar` bound at the innermost binder, or `None` when the
/// binder is not that one (the argument list in hand does not cover it).
fn typevar_index(type_var: &serde_json::Value) -> Option<usize> {
    let pair = type_var
        .get("Bound")
        .and_then(serde_json::Value::as_array)
        .or_else(|| type_var.as_array())?;
    let debruijn = pair.first()?.as_u64()?;
    let index = pair.get(1)?.as_u64()?;
    (debruijn == 0).then_some(index as usize)
}

fn adt_may_run_user_drop(
    adt: &serde_json::Value,
    llbc: &majit_charon_reader::Llbc,
    owners: &HashSet<u64>,
    memo: &mut HashMap<u64, bool>,
    enclosing: &[serde_json::Value],
    depth: usize,
) -> bool {
    let args: &[serde_json::Value] = adt
        .get("generics")
        .and_then(|generics| generics.get("types"))
        .and_then(serde_json::Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    // A tuple's elements are its type arguments. Other non-nominal builtins
    // (`str`, and anything this reader does not model) stay uses.
    if adt.get("builtin").and_then(serde_json::Value::as_str) == Some("Tuple") {
        return args.iter().any(|arg| {
            value_may_run_user_drop(arg, llbc, owners, memo, enclosing, &[], depth + 1)
        });
    }
    let Some(def_id) = adt_nominal_def_id(adt) else {
        return true;
    };
    if owners.contains(&def_id) {
        return true;
    }
    let Some(decl) = llbc.type_by_id(def_id) else {
        return true;
    };
    match &decl.kind {
        TypeDeclKind::Struct(fields) | TypeDeclKind::Union(fields) => fields.iter().any(|field| {
            ty_may_run_user_drop(&field.ty, llbc, owners, memo, args, enclosing, depth + 1)
        }),
        TypeDeclKind::Enum(variants) => variants.iter().any(|variant| {
            variant.fields.iter().any(|field| {
                ty_may_run_user_drop(&field.ty, llbc, owners, memo, args, enclosing, depth + 1)
            })
        }),
        // No field list. A destructor of this shape can still run the
        // instantiation's type arguments; an argument that is itself
        // dropless contributes nothing.
        TypeDeclKind::Opaque => args
            .iter()
            .any(|arg| value_may_run_user_drop(arg, llbc, owners, memo, enclosing, &[], depth + 1)),
        TypeDeclKind::Alias(body) => {
            value_may_run_user_drop(body, llbc, owners, memo, args, enclosing, depth + 1)
        }
        TypeDeclKind::Unknown => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_charon_reader::ullbc::{
        Operand, Place, PlaceKind, ProjectionElem, Rvalue, StmtKind, TyRef,
    };

    fn ty() -> TyRef {
        TyRef::Dedup { id: 0 }
    }

    fn local(i: u64) -> Place {
        Place {
            kind: PlaceKind::Local(i),
            ty: ty(),
        }
    }

    fn mv(i: u64) -> Operand {
        Operand::Move(local(i))
    }

    fn chased(l: u64, defs: &HashMap<u64, PinSrc>) -> Vec<u64> {
        let mut out = HashSet::new();
        chase_pinned(l, defs, &mut out, 0);
        let mut v: Vec<u64> = out.into_iter().collect();
        v.sort();
        v
    }

    /// The set is named by `pin_roots`, never by the opener.  Reading the
    /// contents off `push_roots` would find no argument at all and report
    /// every bracket as empty.
    #[test]
    fn the_scope_opener_is_not_where_a_root_set_is_named() {
        assert!(is_pin_fn("pyre_object::gc_roots::pin_root"));
        assert!(is_pin_fn("pyre_object::gc_roots::pin_roots"));
        assert!(is_pin_fn("pyre_object::gc_roots::publish_roots"));
        assert!(!is_pin_fn("pyre_object::gc_roots::push_roots"));
    }

    /// `pin_roots(&[a, b])` lowers to an array build, a borrow, and the call.
    #[test]
    fn a_pin_argument_traces_back_to_the_array_it_was_built_from() {
        let mut defs = HashMap::new();
        defs.insert(2, PinSrc::Aggregate(vec![10, 11]));
        defs.insert(
            3,
            pin_src(&Rvalue::Ref {
                place: local(2),
                kind: serde_json::Value::Null,
                ptr_metadata: serde_json::Value::Null,
            })
            .expect("a borrow is a followable alias"),
        );
        assert_eq!(chased(3, &defs), vec![2, 3, 10, 11]);
    }

    /// The borrow is `&[T; N]` and the parameter is `&[T]`, so an unsize cast
    /// sits between them on some lowerings and on others it does not.
    #[test]
    fn an_unsize_cast_between_the_array_and_the_slice_is_walked_through() {
        let mut defs = HashMap::new();
        defs.insert(2, PinSrc::Aggregate(vec![10]));
        defs.insert(3, PinSrc::Alias(2));
        defs.insert(
            4,
            pin_src(&Rvalue::Cast(serde_json::Value::Null, mv(3), ty()))
                .expect("a cast is a followable alias"),
        );
        assert_eq!(chased(4, &defs), vec![2, 3, 4, 10]);
    }

    /// A pin publishes a value, so every local spelling that value at the pin
    /// is covered by it -- keeping only the end of the chain would read a
    /// covered root as missing and accuse correct code.
    #[test]
    fn a_pin_covers_every_local_the_value_is_spelled_by() {
        let mut defs = HashMap::new();
        defs.insert(
            1,
            pin_src(&Rvalue::Use(mv(0), serde_json::Value::Null)).expect("a use is an alias"),
        );
        assert_eq!(chased(1, &defs), vec![0, 1]);
    }

    /// A local defined by a call has no followable definition; the pin still
    /// names that local and nothing more.
    #[test]
    fn an_argument_with_no_followable_definition_names_only_itself() {
        assert_eq!(chased(7, &HashMap::new()), vec![7]);
    }

    /// Two locals aliasing each other must not walk forever.  A body is read
    /// from an artefact, so no shape can be ruled out by construction.
    #[test]
    fn an_alias_cycle_terminates() {
        let mut defs = HashMap::new();
        defs.insert(0, PinSrc::Alias(1));
        defs.insert(1, PinSrc::Alias(0));
        assert_eq!(chased(0, &defs), vec![0, 1]);
    }

    /// A shape this reader does not model leaves the set unread rather than
    /// guessed at: `RootedItems` fills its slots through a method, and reading
    /// that as an empty pin would report every root in it as missing.
    #[test]
    fn an_unmodelled_rvalue_yields_no_source_at_all() {
        assert!(pin_src(&Rvalue::Unknown).is_none());
        assert!(pin_src(&Rvalue::Len(local(1))).is_none());
    }

    fn field_of(i: u64) -> Place {
        Place {
            kind: PlaceKind::Projection(
                Box::new(local(i)),
                ProjectionElem::Tagged(serde_json::json!({"Field": [null, 0]})),
            ),
            ty: ty(),
        }
    }

    /// `dest = src` and a cast of that local copy the pin; a field load does not.
    #[test]
    fn only_a_bare_local_copy_or_cast_transfers_a_pin() {
        assert_eq!(
            pin_copy_src(&Rvalue::Use(mv(2), serde_json::Value::Null)),
            Some(2)
        );
        assert_eq!(
            pin_copy_src(&Rvalue::Cast(serde_json::Value::Null, mv(2), ty())),
            Some(2)
        );
        assert_eq!(
            pin_copy_src(&Rvalue::Use(
                Operand::Copy(field_of(2)),
                serde_json::Value::Null
            )),
            None
        );
        assert_eq!(
            pin_src(&Rvalue::Use(
                Operand::Copy(field_of(2)),
                serde_json::Value::Null
            )),
            Some(PinSrc::Alias(2))
        );
    }

    fn helper_fact(
        arg_count: u64,
        has_push_roots: bool,
        calls: Vec<(&str, u64, Vec<Vec<u64>>)>,
    ) -> HelperBodyFact {
        HelperBodyFact {
            has_push_roots,
            arg_count,
            defs: HashMap::new(),
            pin_result_locals: HashSet::new(),
            calls: calls
                .into_iter()
                .map(|(name, callee, arg_locals)| HelperCallFact {
                    callee,
                    callee_name: name.to_string(),
                    arg_locals,
                    assigned_before: HashSet::new(),
                    dest: None,
                })
                .collect(),
            block_calls: Vec::new(),
            successors: Vec::new(),
        }
    }

    fn summary_of(bodies: HashMap<u64, HelperBodyFact>, id: u64) -> Option<PinHelperSummary> {
        summarize_pin_helpers(&bodies, &HashSet::new()).remove(&id)
    }

    /// A pin on only one arm is not a must-pin: the caller scan would
    /// otherwise treat the skipped arm as bracketed.
    #[test]
    fn a_helper_that_pins_on_one_arm_names_no_parameter() {
        let mut body = helper_fact(
            1,
            false,
            vec![("pyre_object::gc_roots::pin_root", 9, vec![vec![1]])],
        );
        // block 0 switches to a pin arm and a bare return.
        body.block_calls = vec![Vec::new(), vec![0], Vec::new()];
        body.successors = vec![vec![1, 2], Vec::new(), Vec::new()];
        let bodies = HashMap::from([(1, body)]);
        assert_eq!(
            summary_of(bodies, 1),
            Some(PinHelperSummary {
                pinned_params: HashSet::new(),
                returns_pinned: false,
            })
        );
    }

    /// `pin_root(param)` — parameter local 1 is argument position 0.
    #[test]
    fn a_helper_that_pins_its_first_parameter_names_that_position() {
        let bodies = HashMap::from([(
            1,
            helper_fact(
                1,
                false,
                vec![("pyre_object::gc_roots::pin_root", 9, vec![vec![1]])],
            ),
        )]);
        assert_eq!(
            summary_of(bodies, 1),
            Some(PinHelperSummary {
                pinned_params: HashSet::from([0]),
                returns_pinned: false,
            })
        );
    }

    /// `pin_root(fresh)` — the pinned local is not a parameter.
    #[test]
    fn a_helper_that_pins_only_a_fresh_value_names_no_parameter() {
        let bodies = HashMap::from([(
            1,
            helper_fact(
                1,
                false,
                vec![("pyre_object::gc_roots::pin_root", 9, vec![vec![5]])],
            ),
        )]);
        assert_eq!(
            summary_of(bodies, 1),
            Some(PinHelperSummary {
                pinned_params: HashSet::new(),
                returns_pinned: false,
            })
        );
    }

    /// A pin into a nested owned scope is not a pin into the caller.
    #[test]
    fn a_pin_caller_that_opens_a_scope_through_another_function_is_not_a_helper() {
        let mut bodies = HashMap::new();
        bodies.insert(
            1,
            helper_fact(
                1,
                true,
                vec![("pyre_object::gc_roots::pin_root", 9, vec![vec![1]])],
            ),
        );
        bodies.insert(
            2,
            helper_fact(
                1,
                false,
                vec![
                    ("pyre_object::gc_roots::pin_root", 9, vec![vec![1]]),
                    ("nested_opener", 1, vec![vec![1]]),
                ],
            ),
        );
        let sums = summarize_pin_helpers(&bodies, &HashSet::new());
        assert!(!sums.contains_key(&1));
        assert!(!sums.contains_key(&2));
    }

    /// `RootedItems::new` opens its own scope. It is not a helper, and neither
    /// is a function whose only pin goes through it.
    #[test]
    fn a_function_that_opens_its_own_root_scope_is_not_a_helper() {
        let mut bodies = HashMap::new();
        bodies.insert(
            1,
            helper_fact(
                1,
                true,
                vec![("pyre_object::gc_roots::pin_root", 9, vec![vec![1]])],
            ),
        );
        bodies.insert(
            2,
            helper_fact(
                1,
                false,
                vec![("pyre_object::gc_roots::RootedItems::new", 1, vec![vec![1]])],
            ),
        );
        let sums = summarize_pin_helpers(&bodies, &HashSet::new());
        assert!(!sums.contains_key(&1));
        assert!(!sums.contains_key(&2));
    }

    /// `outer` hands local 3, an alias of its parameter, to `inner`'s pinned
    /// position. The other argument is not pinned.
    #[test]
    fn a_nested_helper_pins_the_parameter_its_caller_handed_over() {
        let mut outer = helper_fact(
            2,
            false,
            vec![(
                "pyre_interpreter::builtins::pin_into_caller",
                1,
                vec![vec![3], vec![2]],
            )],
        );
        outer.defs.insert(3, PinSrc::Alias(1));
        let inner = helper_fact(
            1,
            false,
            vec![("pyre_object::gc_roots::pin_root", 9, vec![vec![1]])],
        );
        let mut bodies = HashMap::new();
        bodies.insert(1, inner);
        bodies.insert(2, outer);
        let sums = summarize_pin_helpers(&bodies, &HashSet::new());
        assert_eq!(
            sums.get(&1).map(|summary| summary.pinned_params.clone()),
            Some(HashSet::from([0]))
        );
        assert_eq!(
            sums.get(&2),
            Some(&PinHelperSummary {
                pinned_params: HashSet::from([0]),
                returns_pinned: false,
            })
        );
    }

    /// The return place is pinned when it aliases a pin result or is a
    /// one-field newtype of one, and not when the assignment packs several
    /// fields.
    #[test]
    fn the_return_place_is_pinned_only_when_it_aliases_a_pin_result() {
        let mut aliased = helper_fact(
            1,
            false,
            vec![("pyre_object::gc_roots::pin_root", 9, vec![vec![1]])],
        );
        aliased.pin_result_locals.insert(2);
        aliased.defs.insert(0, PinSrc::Alias(2));
        let mut aggregate = helper_fact(
            0,
            false,
            vec![("pyre_object::gc_roots::pin_root", 9, vec![vec![4]])],
        );
        aggregate.pin_result_locals.insert(2);
        aggregate.defs.insert(0, PinSrc::Aggregate(vec![2]));
        let mut wide = helper_fact(
            0,
            false,
            vec![("pyre_object::gc_roots::pin_root", 9, vec![vec![4]])],
        );
        wide.pin_result_locals.insert(2);
        wide.defs.insert(0, PinSrc::Aggregate(vec![2, 5]));
        let mut slot = helper_fact(
            0,
            false,
            vec![
                ("pyre_object::gc_roots::pin_root", 9, vec![vec![4]]),
                ("pyre_object::gc_roots::shadow_stack_get", 8, vec![vec![3]]),
            ],
        );
        slot.pin_result_locals.insert(0);
        let bodies = HashMap::from([(1, aliased), (2, aggregate), (3, slot), (4, wide)]);
        let sums = summarize_pin_helpers(&bodies, &HashSet::new());
        assert!(sums[&1].returns_pinned);
        assert!(sums[&2].returns_pinned);
        assert!(sums[&3].returns_pinned);
        assert!(!sums[&4].returns_pinned);
        assert!(sums[&2].pinned_params.is_empty());
    }

    /// A collection between the pin and the return drops `returns_pinned`.
    #[test]
    fn a_collecting_call_after_a_pin_is_not_a_pinned_return() {
        let mut body = helper_fact(
            1,
            false,
            vec![
                ("pyre_object::gc_roots::pin_root", 9, vec![vec![1]]),
                ("gc_hook::try_gc_collect", 11, vec![]),
            ],
        );
        body.pin_result_locals.insert(0);
        body.defs.insert(0, PinSrc::Alias(2));
        body.pin_result_locals.insert(2);
        let bodies = HashMap::from([(1, body)]);
        let sums = summarize_pin_helpers(&bodies, &HashSet::from([11]));
        assert!(!sums[&1].returns_pinned);
    }

    /// A wrapper id in `collecting` still drops the pinned return.
    /// Production fills that set with `CallGraph::reaching` of the seeds.
    #[test]
    fn a_wrapper_id_passed_as_collecting_is_not_a_pinned_return() {
        let mut body = helper_fact(
            1,
            false,
            vec![
                ("pyre_object::gc_roots::pin_root", 9, vec![vec![1]]),
                ("pyre_object::weakref::w_weakref_new", 20, vec![vec![1]]),
            ],
        );
        body.pin_result_locals.insert(0);
        let bodies = HashMap::from([(1, body)]);
        let sums = summarize_pin_helpers(&bodies, &HashSet::from([20]));
        assert!(!sums[&1].returns_pinned);
    }

    /// A second `pin_root` is a safepoint. The word returned by the first
    /// pin is not updated.
    #[test]
    fn a_later_pin_after_a_returned_pin_is_not_a_pinned_return() {
        let mut body = helper_fact(
            1,
            false,
            vec![
                ("pyre_object::gc_roots::pin_root", 9, vec![vec![1]]),
                ("pyre_object::gc_roots::pin_root", 9, vec![vec![2]]),
            ],
        );
        body.pin_result_locals.insert(0);
        let bodies = HashMap::from([(1, body)]);
        let sums = summarize_pin_helpers(&bodies, &HashSet::new());
        assert!(!sums[&1].returns_pinned);

        let mut blocked = helper_fact(
            1,
            false,
            vec![
                ("pyre_object::gc_roots::pin_root", 9, vec![vec![1]]),
                ("pyre_object::gc_roots::pin_root", 9, vec![vec![2]]),
            ],
        );
        blocked.pin_result_locals.insert(0);
        blocked.block_calls = vec![vec![0, 1]];
        blocked.successors = vec![vec![]];
        let bodies = HashMap::from([(1, blocked)]);
        let sums = summarize_pin_helpers(&bodies, &HashSet::new());
        assert!(!sums[&1].returns_pinned);
    }

    /// The second pin's own result is current. An earlier unrelated pin is
    /// not a safepoint after the producer.
    #[test]
    fn a_return_of_a_later_pin_is_still_a_pinned_return() {
        let mut body = helper_fact(
            1,
            false,
            vec![
                ("pyre_object::gc_roots::pin_root", 9, vec![vec![1]]),
                ("pyre_object::gc_roots::pin_root", 9, vec![vec![2]]),
            ],
        );
        body.calls[0].dest = Some(3);
        body.calls[1].dest = Some(0);
        body.pin_result_locals.insert(0);
        body.pin_result_locals.insert(3);
        let bodies = HashMap::from([(1, body)]);
        let sums = summarize_pin_helpers(&bodies, &HashSet::new());
        assert!(sums[&1].returns_pinned);

        let mut blocked = helper_fact(
            1,
            false,
            vec![
                ("pyre_object::gc_roots::pin_root", 9, vec![vec![1]]),
                ("pyre_object::gc_roots::pin_root", 9, vec![vec![2]]),
            ],
        );
        blocked.calls[0].dest = Some(3);
        blocked.calls[1].dest = Some(0);
        blocked.pin_result_locals.insert(0);
        blocked.pin_result_locals.insert(3);
        blocked.block_calls = vec![vec![0, 1]];
        blocked.successors = vec![vec![]];
        let bodies = HashMap::from([(1, blocked)]);
        let sums = summarize_pin_helpers(&bodies, &HashSet::new());
        assert!(sums[&1].returns_pinned);
    }

    /// A nested summarized pin helper pins at a safepoint. The word copied
    /// by the earlier pin is stale.
    #[test]
    fn a_nested_pin_helper_after_a_pin_is_not_a_pinned_return() {
        let inner = helper_fact(
            1,
            false,
            vec![("pyre_object::gc_roots::pin_root", 9, vec![vec![1]])],
        );
        let mut outer = helper_fact(
            1,
            false,
            vec![
                ("pyre_object::gc_roots::pin_root", 9, vec![vec![1]]),
                ("nested_pin_helper", 2, vec![vec![4]]),
            ],
        );
        outer.calls[0].dest = Some(0);
        outer.pin_result_locals.insert(0);
        let bodies = HashMap::from([(1, outer.clone()), (2, inner.clone())]);
        let sums = summarize_pin_helpers(&bodies, &HashSet::new());
        assert!(sums.contains_key(&2));
        assert!(!sums[&1].returns_pinned);

        let mut blocked = outer;
        blocked.block_calls = vec![vec![0, 1]];
        blocked.successors = vec![vec![]];
        let bodies = HashMap::from([(1, blocked), (2, inner)]);
        let sums = summarize_pin_helpers(&bodies, &HashSet::new());
        assert!(!sums[&1].returns_pinned);
    }

    /// `CollectAnalyzer` / `CallGraph::reaching` includes a wrapper of a
    /// seed, so summarizing a pin then `w_weakref_new` drops `returns_pinned`.
    #[test]
    fn collecting_for_pin_summaries_includes_a_wrapper_of_a_seed() {
        let mut names = HashMap::new();
        names.insert(1, "helper".into());
        names.insert(20, "pyre_object::weakref::w_weakref_new".into());
        names.insert(11, "gc_hook::try_gc_alloc_collecting_rooted".into());
        let mut callees: HashMap<u64, HashSet<u64>> = HashMap::new();
        callees.insert(1, HashSet::from([20]));
        callees.insert(20, HashSet::from([11]));
        let mut callers: HashMap<u64, HashSet<u64>> = HashMap::new();
        callers.insert(20, HashSet::from([1]));
        callers.insert(11, HashSet::from([20]));
        let cg = super::super::framework::CallGraph {
            names,
            keys: HashMap::new(),
            callees,
            callers,
            indirect: HashSet::new(),
            opaque: super::super::framework::OpaqueCensus::default(),
            spec_memo: HashMap::new(),
        };
        let collecting = collecting_for_pin_summaries(&cg);
        assert!(collecting.contains(&11));
        assert!(
            collecting.contains(&20),
            "w_weakref_new reaches a collecting seed"
        );

        let mut body = helper_fact(
            1,
            false,
            vec![
                ("pyre_object::gc_roots::pin_root", 9, vec![vec![1]]),
                ("pyre_object::weakref::w_weakref_new", 20, vec![vec![1]]),
            ],
        );
        body.pin_result_locals.insert(0);
        let bodies = HashMap::from([(1, body)]);
        let sums = summarize_pin_helpers(&bodies, &collecting);
        assert!(!sums[&1].returns_pinned);
    }

    /// An assignment after the pin does not erase the parameter the pin saw.
    #[test]
    fn an_assignment_after_the_pin_keeps_the_pinned_parameter() {
        let mut body = helper_fact(
            1,
            false,
            vec![("pyre_object::gc_roots::pin_root", 9, vec![vec![1]])],
        );
        body.calls[0].assigned_before.clear();
        let bodies = HashMap::from([(1, body)]);
        let sums = summarize_pin_helpers(&bodies, &HashSet::new());
        assert_eq!(sums[&1].pinned_params, HashSet::from([0]));

        let mut replaced = helper_fact(
            1,
            false,
            vec![("pyre_object::gc_roots::pin_root", 9, vec![vec![1]])],
        );
        replaced.calls[0].assigned_before.insert(1);
        let bodies = HashMap::from([(1, replaced)]);
        let sums = summarize_pin_helpers(&bodies, &HashSet::new());
        assert!(sums[&1].pinned_params.is_empty());
    }

    /// The call-graph prefilter keeps a pin-caller and its non-bracketing
    /// caller, and drops a function that opens a scope.
    #[test]
    fn candidate_ids_skip_a_function_that_opens_a_root_scope() {
        let mut callees: HashMap<u64, HashSet<u64>> = HashMap::new();
        let mut names = HashMap::new();
        callees.insert(1, HashSet::from([9]));
        names.insert(1, "helper".into());
        names.insert(9, "pyre_object::gc_roots::pin_root".into());
        callees.insert(2, HashSet::from([7, 9]));
        names.insert(2, "bracket".into());
        names.insert(7, "pyre_object::gc_roots::push_roots".into());
        callees.insert(3, HashSet::from([1]));
        names.insert(3, "outer".into());
        callees.insert(4, HashSet::from([2]));
        names.insert(4, "caller_of_bracket".into());
        callees.insert(5, HashSet::from([2, 9]));
        names.insert(5, "pin_and_open".into());
        let ids = helper_candidate_ids(&callees, &names, &HashSet::from([7]));
        assert!(ids.contains(&1));
        assert!(ids.contains(&3));
        assert!(!ids.contains(&2));
        assert!(!ids.contains(&4));
        assert!(!ids.contains(&5));
        assert!(!ids.contains(&9));
    }

    fn item_meta(path: &[&str]) -> serde_json::Value {
        let span = serde_json::json!({
            "data": {"file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}
        });
        serde_json::json!({
            "name": path.iter().map(|s| serde_json::json!({"Ident": [s, 0]})).collect::<Vec<_>>(),
            "span": span.clone(),
            "source_text": null,
            "is_local": true,
            "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true}
        })
    }

    fn generics(types: serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "regions": [], "types": types, "const_generics": [], "trait_refs": []
        })
    }

    fn adt(id: u64, types: serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "Adt": {"id": id, "generics": generics(types), "builtin": null}
        })
    }

    fn inline(id: u64, body: serde_json::Value) -> serde_json::Value {
        serde_json::json!({"Value": [id, body]})
    }

    fn raw_ptr() -> serde_json::Value {
        serde_json::json!({"RawPtr": [null, "Mut"]})
    }

    fn field(ty: serde_json::Value) -> serde_json::Value {
        serde_json::json!({"name": "_0", "is_positional": true, "ty": ty})
    }

    fn struct_decl(id: u64, name: &str, fields: serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "def_id": id,
            "item_meta": item_meta(&[name]),
            "kind": {"Struct": fields}
        })
    }

    fn trait_decl(id: u64, path: &[&str]) -> serde_json::Value {
        serde_json::json!({"def_id": id, "item_meta": item_meta(path)})
    }

    fn trait_impl(trait_id: u64, adt_id: u64) -> serde_json::Value {
        serde_json::json!({
            "impl_trait": {
                "id": trait_id,
                "generics": {"types": [inline(1, adt(adt_id, serde_json::json!([])))]}
            }
        })
    }

    /// Locals live across the collecting call that sits between the payload's
    /// last real use and the `Drop` of that payload.
    fn live_across_drop(
        type_decls: serde_json::Value,
        trait_decls: serde_json::Value,
        trait_impls: serde_json::Value,
        payload_ty: serde_json::Value,
    ) -> Vec<String> {
        let span = serde_json::json!({
            "data": {"file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}
        });
        let scalar = inline(1, serde_json::json!({"Scalar": "Bool"}));
        let gens = generics(serde_json::json!([]));
        let place =
            |id: u64, ty: &serde_json::Value| serde_json::json!({"kind": {"Local": id}, "ty": ty});
        let call = |fun: u64,
                    args: serde_json::Value,
                    dest: u64,
                    dest_ty: &serde_json::Value,
                    target: u64| {
            serde_json::json!({
                "statements": [],
                "kind": "Regular",
                "terminator": {"span": span.clone(), "kind": {"Call": {
                    "call": {
                        "func": {"Regular": {"kind": {"Fun": fun}, "generics": gens.clone()}},
                        "args": args,
                        "dest": place(dest, dest_ty)
                    },
                    "target": target,
                    "on_unwind": 3
                }}}
            })
        };
        let holder = serde_json::json!({
            "def_id": 0,
            "item_meta": item_meta(&["fixture", "holder"]),
            "signature": {"is_unsafe": false, "inputs": [], "output": scalar.clone()},
            "body": {"Unstructured": {
                "span": span.clone(),
                "locals": {"arg_count": 0, "locals": [
                    {"index": 0, "name": null, "span": span.clone(), "ty": scalar.clone()},
                    {"index": 1, "name": "tmp", "span": span.clone(), "ty": scalar.clone()},
                    {"index": 2, "name": "payload", "span": span.clone(), "ty": payload_ty.clone()}
                ]},
                "body": [
                    call(2, serde_json::json!([{"Copy": place(2, &payload_ty)}]), 1, &scalar, 1),
                    call(1, serde_json::json!([]), 0, &scalar, 2),
                    {"statements": [], "kind": "Regular", "terminator": {"span": span.clone(), "kind": {"Drop": {
                        "place": place(2, &payload_ty),
                        "fn_ptr": {"kind": {"Fun": 1}, "generics": gens.clone()},
                        "target": 3,
                        "on_unwind": 3
                    }}}},
                    {"statements": [], "kind": "Regular", "terminator": {"span": span.clone(), "kind": "Return"}}
                ]
            }}
        });
        let opaque = |id: u64, name: &str| {
            serde_json::json!({
                "def_id": id,
                "item_meta": item_meta(&["fixture", name]),
                "signature": {"is_unsafe": false, "inputs": [], "output": scalar.clone()},
                "body": "Opaque"
            })
        };
        let file = serde_json::json!({
            "charon_version": "0.1.201",
            "has_errors": false,
            "translated": {
                "crate_name": "fixture",
                "type_decls": type_decls,
                "fun_decls": [holder, opaque(1, "collect"), opaque(2, "touch")],
                "global_decls": [],
                "trait_decls": trait_decls,
                "trait_impls": trait_impls
            }
        });
        let llbc = majit_charon_reader::Llbc::from_slice(file.to_string().as_bytes())
            .expect("drop fixture parses");
        let cg = super::super::framework::build(&llbc);
        let mut reach = HashSet::new();
        reach.insert(0);
        reach.insert(1);
        let mut gc_tys = HashSet::new();
        gc_tys.insert(100);
        let (findings, stats) = scan(
            &llbc,
            &cg,
            &reach,
            &HashSet::new(),
            &gc_tys,
            &HashSet::new(),
            &[],
        );
        assert_eq!(
            stats.unparsed_terminator_bodies, 0,
            "the fixture's terminators must parse"
        );
        assert_eq!(
            stats.bodies_scanned, 1,
            "the payload must be a tracked local"
        );
        findings
            .into_iter()
            .flat_map(|finding| finding.live_non_arg)
            .collect()
    }

    /// The `Drop` after the payload's last real use is not itself a use, so
    /// the collecting call between them sees a dead local.
    #[test]
    fn a_drop_of_a_dropless_local_after_its_last_use_leaves_it_dead() {
        let payload = struct_decl(
            0,
            "Handle",
            serde_json::json!([field(inline(7, raw_ptr()))]),
        );
        let ty = inline(100, adt(0, serde_json::json!([])));
        let live = live_across_drop(
            serde_json::json!([payload]),
            serde_json::json!([]),
            serde_json::json!([]),
            ty,
        );
        assert!(live.is_empty(), "dropless drop kept {live:?} live");
    }

    /// A type with a `Drop` impl is read by its destructor, so the local is
    /// live at the collecting call that precedes that `Drop`.
    #[test]
    fn a_drop_of_a_type_with_a_drop_impl_keeps_it_live() {
        let payload = struct_decl(0, "Guard", serde_json::json!([]));
        let ty = inline(100, adt(0, serde_json::json!([])));
        let live = live_across_drop(
            serde_json::json!([payload]),
            serde_json::json!([trait_decl(0, &["core", "ops", "drop", "Drop"])]),
            serde_json::json!([trait_impl(0, 0)]),
            ty,
        );
        assert_eq!(live, vec!["payload".to_string()]);
    }

    /// The destructor runs for a field too, not only for the outer type's
    /// own impl.
    #[test]
    fn a_drop_of_a_struct_whose_field_has_a_drop_impl_keeps_it_live() {
        let outer = struct_decl(
            0,
            "Outer",
            serde_json::json!([field(inline(8, adt(1, serde_json::json!([]))))]),
        );
        let inner = struct_decl(1, "Inner", serde_json::json!([]));
        let ty = inline(100, adt(0, serde_json::json!([])));
        let live = live_across_drop(
            serde_json::json!([outer, inner]),
            serde_json::json!([trait_decl(0, &["core", "ops", "drop", "Drop"])]),
            serde_json::json!([trait_impl(0, 1)]),
            ty,
        );
        assert_eq!(live, vec!["payload".to_string()]);
    }

    /// `Result<RawPtr, RawPtr>`: the enum's fields are type variables, and
    /// neither instantiation argument runs a user destructor.
    #[test]
    fn a_drop_of_an_instantiated_enum_of_raw_pointers_leaves_it_dead() {
        let result = serde_json::json!({
            "def_id": 0,
            "item_meta": item_meta(&["Result"]),
            "kind": {"Enum": [
                {"name": "Ok", "fields": [field(inline(5, serde_json::json!({"TypeVar": {"Bound": [0, 0]}})))]},
                {"name": "Err", "fields": [field(inline(6, serde_json::json!({"TypeVar": {"Bound": [0, 1]}})))]}
            ]}
        });
        let ty = inline(
            100,
            adt(
                0,
                serde_json::json!([inline(101, raw_ptr()), inline(102, raw_ptr())]),
            ),
        );
        let live = live_across_drop(
            serde_json::json!([result]),
            serde_json::json!([]),
            serde_json::json!([]),
            ty,
        );
        assert!(live.is_empty(), "enum of raw pointers kept {live:?} live");
    }

    /// An unbound type variable can be anything, so the `Drop` counts.
    #[test]
    fn a_drop_of_a_generic_parameter_keeps_it_live() {
        let ty = inline(100, serde_json::json!({"TypeVar": {"Bound": [0, 0]}}));
        let live = live_across_drop(
            serde_json::json!([]),
            serde_json::json!([]),
            serde_json::json!([]),
            ty,
        );
        assert_eq!(live, vec!["payload".to_string()]);
    }

    /// No declaration means the destructor contract is unknown.
    #[test]
    fn a_drop_of_a_missing_decl_keeps_it_live() {
        let ty = inline(100, adt(0, serde_json::json!([])));
        let live = live_across_drop(
            serde_json::json!([]),
            serde_json::json!([]),
            serde_json::json!([]),
            ty,
        );
        assert_eq!(live, vec!["payload".to_string()]);
    }

    /// An opaque body has no fields. Its type arguments are what a
    /// destructor can still run, and a raw pointer does not.
    #[test]
    fn a_drop_of_an_opaque_type_of_raw_pointers_leaves_it_dead() {
        let wrapper = serde_json::json!({
            "def_id": 0,
            "item_meta": item_meta(&["Vec"]),
            "kind": "Opaque"
        });
        let ty = inline(100, adt(0, serde_json::json!([inline(101, raw_ptr())])));
        let live = live_across_drop(
            serde_json::json!([wrapper]),
            serde_json::json!([]),
            serde_json::json!([]),
            ty,
        );
        assert!(
            live.is_empty(),
            "opaque raw-pointer wrapper kept {live:?} live"
        );
    }

    /// The same opaque shape keeps the local live when an argument has a
    /// `Drop` impl.
    #[test]
    fn a_drop_of_an_opaque_type_carrying_a_drop_type_keeps_it_live() {
        let wrapper = serde_json::json!({
            "def_id": 0,
            "item_meta": item_meta(&["Vec"]),
            "kind": "Opaque"
        });
        let inner = struct_decl(1, "Guard", serde_json::json!([]));
        let ty = inline(
            100,
            adt(
                0,
                serde_json::json!([inline(101, adt(1, serde_json::json!([])))]),
            ),
        );
        let live = live_across_drop(
            serde_json::json!([wrapper, inner]),
            serde_json::json!([trait_decl(0, &["core", "ops", "drop", "Drop"])]),
            serde_json::json!([trait_impl(0, 1)]),
            ty,
        );
        assert_eq!(live, vec!["payload".to_string()]);
    }

    /// `Copy` is not `Drop`. A missing trait decl is not `Copy` either: the
    /// impl cannot be shown to be something other than a destructor.
    #[test]
    fn a_copy_impl_does_not_keep_the_local_live_but_an_unresolved_impl_does() {
        let payload = struct_decl(0, "Bits", serde_json::json!([]));
        let ty = inline(100, adt(0, serde_json::json!([])));
        let copy = live_across_drop(
            serde_json::json!([payload.clone()]),
            serde_json::json!([trait_decl(0, &["core", "marker", "Copy"])]),
            serde_json::json!([trait_impl(0, 0)]),
            ty.clone(),
        );
        assert!(copy.is_empty(), "Copy kept {copy:?} live");
        let unresolved = live_across_drop(
            serde_json::json!([payload]),
            serde_json::json!([]),
            serde_json::json!([trait_impl(9, 0)]),
            ty,
        );
        assert_eq!(unresolved, vec!["payload".to_string()]);
    }

    /// `PlaceMention` is a borrowck marker, not a read. A collecting call
    /// between the last real use and a mention sees a dead local.
    #[test]
    fn a_place_mention_is_not_a_use() {
        let mut live = HashSet::new();
        let tracked = HashMap::from([(1u64, "b".to_string())]);
        transfer_stmt(&StmtKind::PlaceMention(local(1)), &mut live, &tracked);
        assert!(live.is_empty(), "PlaceMention kept {live:?} live");
    }

    #[test]
    fn empty_entry_still_reaches_a_later_assignment() {
        let assigns = vec![HashSet::new(), HashSet::from([1]), HashSet::new()];
        let dests = vec![HashSet::new(), HashSet::new(), HashSet::new()];
        let succ = vec![vec![1], vec![2], vec![]];
        let entry = assigned_at_block_entry(&assigns, &dests, &succ);
        assert!(entry[2].contains(&1));
        assert!(!entry[1].contains(&1));
    }

    #[test]
    fn call_dest_reaches_the_successor_not_its_own_block() {
        let assigns = vec![HashSet::new(), HashSet::new()];
        let dests = vec![HashSet::from([4]), HashSet::new()];
        let succ = vec![vec![1], vec![]];
        let entry = assigned_at_block_entry(&assigns, &dests, &succ);
        assert!(!entry[0].contains(&4));
        assert!(entry[1].contains(&4));
    }

    /// Body whose only Call to `cleanup_only` (id 2) sits in a cleanup
    /// block reached through `on_unwind`.
    fn llbc_with_cleanup_only_call() -> majit_charon_reader::Llbc {
        let span = serde_json::json!({
            "data": {"file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}
        });
        let scalar = serde_json::json!({"Value": [1, {"Scalar": "Bool"}]});
        let gc_ty = serde_json::json!({"Deduplicated": 100});
        let gens = serde_json::json!({
            "regions": [], "types": [], "const_generics": [], "trait_refs": []
        });
        let place =
            |id: u64, ty: &serde_json::Value| serde_json::json!({"kind": {"Local": id}, "ty": ty});
        let call_bb = |fun: u64,
                       dest: u64,
                       dest_ty: &serde_json::Value,
                       target: u64,
                       on_unwind: u64,
                       is_cleanup: bool| {
            serde_json::json!({
                "statements": [],
                "kind": if is_cleanup { "Cleanup" } else { "Regular" },
                "terminator": {"span": span.clone(), "kind": {"Call": {
                    "call": {
                        "func": {"Regular": {"kind": {"Fun": fun}, "generics": gens.clone()}},
                        "args": [],
                        "dest": place(dest, dest_ty)
                    },
                    "target": target,
                    "on_unwind": on_unwind
                }}}
            })
        };
        let holder = serde_json::json!({
            "def_id": 0,
            "item_meta": item_meta(&["fixture", "holder"]),
            "signature": {"is_unsafe": false, "inputs": [], "output": scalar.clone()},
            "body": {"Unstructured": {
                "span": span.clone(),
                "locals": {"arg_count": 0, "locals": [
                    {"index": 0, "name": null, "span": span.clone(), "ty": scalar.clone()},
                    {"index": 1, "name": "payload", "span": span.clone(), "ty": gc_ty.clone()}
                ]},
                "body": [
                    call_bb(1, 0, &scalar, 1, 2, false),
                    {"statements": [], "kind": "Regular", "terminator": {
                        "span": span.clone(), "kind": "Return"
                    }},
                    call_bb(2, 0, &scalar, 3, 3, true),
                    {"statements": [], "kind": "Cleanup", "terminator": {
                        "span": span.clone(), "kind": "UnwindResume"
                    }}
                ]
            }}
        });
        let opaque = |id: u64, name: &str| {
            serde_json::json!({
                "def_id": id,
                "item_meta": item_meta(&["fixture", name]),
                "signature": {"is_unsafe": false, "inputs": [], "output": scalar.clone()},
                "body": "Opaque"
            })
        };
        let file = serde_json::json!({
            "charon_version": "0.1.201",
            "has_errors": false,
            "translated": {
                "crate_name": "fixture",
                "type_decls": [],
                "fun_decls": [holder, opaque(1, "normal_callee"), opaque(2, "cleanup_only")],
                "global_decls": [],
                "trait_decls": [],
                "trait_impls": []
            }
        });
        majit_charon_reader::Llbc::from_slice(file.to_string().as_bytes())
            .expect("cleanup-call fixture parses")
    }

    fn pin_facts(
        fact: &HelperBodyFact,
    ) -> (
        bool,
        u64,
        HashMap<u64, PinSrc>,
        HashSet<u64>,
        Vec<HelperCallFact>,
    ) {
        (
            fact.has_push_roots,
            fact.arg_count,
            fact.defs.clone(),
            fact.pin_result_locals.clone(),
            fact.calls.clone(),
        )
    }

    /// A Call that exists only on the `on_unwind` cleanup chain is not a
    /// pin/liveness fact, matching `strip_cleanup_blocks` of the same body.
    #[test]
    fn cleanup_only_call_is_absent_from_pin_facts_and_scan() {
        use majit_charon_reader::ullbc::strip_cleanup_blocks;

        let llbc = llbc_with_cleanup_only_call();
        let cg = super::super::framework::build(&llbc);
        let holder = cg.callees.get(&0).expect("holder is a local fn");
        assert!(holder.contains(&1), "forward Call to normal_callee");
        assert!(
            !holder.contains(&2),
            "cleanup-only Call must not be an edge: {holder:?}"
        );

        let fd = llbc.fn_by_id(0).expect("holder decl");
        let body = fd.unstructured().expect("Unstructured body");
        let stripped = strip_cleanup_blocks(fd.unstructured().expect("Unstructured body"));
        let names = &cg.names;
        let push = HashSet::new();
        let retained = helper_body_fact(&llbc, &body, &push, names).expect("retained fact");
        let stripped_fact =
            helper_body_fact(&llbc, &stripped, &push, names).expect("stripped fact");
        assert_eq!(
            pin_facts(&retained),
            pin_facts(&stripped_fact),
            "pin facts must equal strip_cleanup_blocks"
        );
        assert!(
            retained.calls.iter().all(|c| c.callee != 2),
            "cleanup-only callee must not appear in pin facts: {:?}",
            retained.calls.iter().map(|c| c.callee).collect::<Vec<_>>()
        );

        let mut reach = HashSet::new();
        reach.insert(0);
        reach.insert(1);
        reach.insert(2);
        let mut gc_tys = HashSet::new();
        gc_tys.insert(100);
        let (findings, stats) = scan(
            &llbc,
            &cg,
            &reach,
            &HashSet::new(),
            &gc_tys,
            &HashSet::new(),
            &[],
        );
        assert_eq!(stats.unparsed_terminator_bodies, 0);
        assert!(
            findings.iter().all(|f| f.callee_id != 2),
            "scan must not report the cleanup-only callee: {:?}",
            findings
                .iter()
                .map(|f| (f.callee_id, &f.callee_name))
                .collect::<Vec<_>>()
        );
    }

    /// Open a root scope, pin `pin_arg`, assign `assign_dest`, then a
    /// collecting call with `assign_dest` still live.
    fn short_bracket_after_pin_assign(
        pin_arg: u64,
        assign_dest: u64,
        assign_rvalue: serde_json::Value,
        extra_locals: serde_json::Value,
    ) -> ScanStats {
        let span = serde_json::json!({
            "data": {"file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}
        });
        let scalar = serde_json::json!({"Value": [1, {"Scalar": "Bool"}]});
        let gc_ty = serde_json::json!({"Deduplicated": 100});
        let scope_ty = serde_json::json!({"Deduplicated": 200});
        let gens = serde_json::json!({
            "regions": [], "types": [], "const_generics": [], "trait_refs": []
        });
        let place =
            |id: u64, ty: &serde_json::Value| serde_json::json!({"kind": {"Local": id}, "ty": ty});
        let call_bb = |fun: u64,
                       args: serde_json::Value,
                       dest: u64,
                       dest_ty: &serde_json::Value,
                       target: u64,
                       statements: serde_json::Value| {
            serde_json::json!({
                "statements": statements,
                "kind": "Regular",
                "terminator": {"span": span.clone(), "kind": {"Call": {
                    "call": {
                        "func": {"Regular": {"kind": {"Fun": fun}, "generics": gens.clone()}},
                        "args": args,
                        "dest": place(dest, dest_ty)
                    },
                    "target": target,
                    "on_unwind": 5
                }}}
            })
        };
        let mut locals = serde_json::json!([
            {"index": 0, "name": null, "span": span.clone(), "ty": scalar.clone()},
            {"index": 1, "name": "scope", "span": span.clone(), "ty": scope_ty.clone()},
            {"index": 2, "name": "obj", "span": span.clone(), "ty": gc_ty.clone()},
            {"index": 3, "name": "loaded", "span": span.clone(), "ty": gc_ty.clone()},
            {"index": 4, "name": "tmp", "span": span.clone(), "ty": scalar.clone()}
        ]);
        if let Some(extra) = extra_locals.as_array() {
            locals
                .as_array_mut()
                .expect("locals is an array")
                .extend(extra.iter().cloned());
        }
        let assign = serde_json::json!({
            "span": span.clone(),
            "kind": {"Assign": [place(assign_dest, &gc_ty), assign_rvalue]}
        });
        let holder = serde_json::json!({
            "def_id": 0,
            "item_meta": item_meta(&["fixture", "holder"]),
            "signature": {"is_unsafe": false, "inputs": [], "output": scalar.clone()},
            "body": {"Unstructured": {
                "span": span.clone(),
                "locals": {"arg_count": 0, "locals": locals},
                "body": [
                    call_bb(3, serde_json::json!([]), 1, &scope_ty, 1, serde_json::json!([])),
                    call_bb(
                        2,
                        serde_json::json!([{"Copy": place(pin_arg, &gc_ty)}]),
                        pin_arg,
                        &gc_ty,
                        2,
                        serde_json::json!([])
                    ),
                    call_bb(
                        1,
                        serde_json::json!([]),
                        4,
                        &scalar,
                        3,
                        serde_json::json!([assign])
                    ),
                    call_bb(
                        4,
                        serde_json::json!([{"Copy": place(assign_dest, &gc_ty)}]),
                        4,
                        &scalar,
                        4,
                        serde_json::json!([])
                    ),
                    {"statements": [], "kind": "Regular", "terminator": {
                        "span": span.clone(), "kind": "Return"
                    }},
                    {"statements": [], "kind": "Cleanup", "terminator": {
                        "span": span.clone(), "kind": "UnwindResume"
                    }}
                ]
            }}
        });
        let opaque = |id: u64, path: &[&str], output: &serde_json::Value| {
            serde_json::json!({
                "def_id": id,
                "item_meta": item_meta(path),
                "signature": {"is_unsafe": false, "inputs": [], "output": output.clone()},
                "body": "Opaque"
            })
        };
        let file = serde_json::json!({
            "charon_version": "0.1.201",
            "has_errors": false,
            "translated": {
                "crate_name": "fixture",
                "type_decls": [],
                "fun_decls": [
                    holder,
                    opaque(1, &["fixture", "collect"], &scalar),
                    opaque(2, &["pyre_object", "gc_roots", "pin_root"], &gc_ty),
                    opaque(3, &["pyre_object", "gc_roots", "push_roots"], &scope_ty),
                    opaque(4, &["fixture", "touch"], &scalar)
                ],
                "global_decls": [],
                "trait_decls": [],
                "trait_impls": []
            }
        });
        let llbc = majit_charon_reader::Llbc::from_slice(file.to_string().as_bytes())
            .expect("pin-assign fixture parses");
        let cg = super::super::framework::build(&llbc);
        let mut reach = HashSet::new();
        reach.insert(0);
        reach.insert(1);
        let mut push_roots = HashSet::new();
        push_roots.insert(3);
        let mut gc_tys = HashSet::new();
        gc_tys.insert(100);
        let (_, stats) = scan(
            &llbc,
            &cg,
            &reach,
            &push_roots,
            &gc_tys,
            &HashSet::new(),
            &[],
        );
        assert_eq!(
            stats.unparsed_terminator_bodies, 0,
            "terminators must parse"
        );
        assert_eq!(stats.unparsed_statement_bodies, 0, "statements must parse");
        assert_eq!(
            stats.withheld_contents_opaque, 0,
            "pin transfer must be readable: {:?}",
            stats.opaque_by_reason
        );
        stats
    }

    /// A field load is a new pointer. Pinning the aggregate does not pin
    /// the loaded word (`gct_direct_call` livevars).
    #[test]
    fn a_field_load_does_not_inherit_the_source_pin() {
        let gc_ty = serde_json::json!({"Deduplicated": 100});
        let field = serde_json::json!({
            "kind": {"Projection": [
                {"kind": {"Local": 2}, "ty": gc_ty.clone()},
                {"Field": [null, 0]}
            ]},
            "ty": gc_ty.clone()
        });
        let stats = short_bracket_after_pin_assign(
            2,
            3,
            serde_json::json!({"Use": [{"Copy": field}, null]}),
            serde_json::json!([]),
        );
        assert_eq!(stats.withheld_under_a_bracket, 1);
        assert_eq!(
            stats.short_brackets.len(),
            1,
            "short={} covers={}",
            stats.withheld_bracket_short,
            stats.withheld_bracket_covers
        );
        assert_eq!(
            stats.short_brackets[0].missing,
            vec!["loaded".to_string()],
            "pinned={:?}",
            stats.short_brackets[0].pinned
        );
    }

    /// Assigning an untracked source onto a pinned local kills the pin.
    #[test]
    fn assigning_an_untracked_source_clears_the_destination_pin() {
        let span = serde_json::json!({
            "data": {"file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}
        });
        let scalar = serde_json::json!({"Value": [1, {"Scalar": "Bool"}]});
        let stats = short_bracket_after_pin_assign(
            3,
            3,
            serde_json::json!({"Use": [
                {"Move": {"kind": {"Local": 5}, "ty": scalar.clone()}},
                null
            ]}),
            serde_json::json!([{
                "index": 5, "name": "bits", "span": span, "ty": scalar
            }]),
        );
        assert_eq!(stats.withheld_under_a_bracket, 1);
        assert_eq!(
            stats.short_brackets.len(),
            1,
            "short={} covers={}",
            stats.withheld_bracket_short,
            stats.withheld_bracket_covers
        );
        assert_eq!(
            stats.short_brackets[0].missing,
            vec!["loaded".to_string()],
            "pinned={:?}",
            stats.short_brackets[0].pinned
        );
    }

    /// A successor that also has a non-forward predecessor must meet that
    /// predecessor's zero out-set, as the Jacobi rounds did. The cleanup
    /// block's in-set starts empty and never shrinks, so only a worklist
    /// that processes every block applies the meet.
    #[test]
    fn a_non_forward_predecessor_clears_the_successor_pin() {
        let span = serde_json::json!({
            "data": {"file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}
        });
        let scalar = serde_json::json!({"Value": [1, {"Scalar": "Bool"}]});
        let gc_ty = serde_json::json!({"Deduplicated": 100});
        let scope_ty = serde_json::json!({"Deduplicated": 200});
        let gens = serde_json::json!({
            "regions": [], "types": [], "const_generics": [], "trait_refs": []
        });
        let place =
            |id: u64, ty: &serde_json::Value| serde_json::json!({"kind": {"Local": id}, "ty": ty});
        let call_bb = |fun: u64,
                       args: serde_json::Value,
                       dest: u64,
                       dest_ty: &serde_json::Value,
                       target: u64| {
            serde_json::json!({
                "statements": [],
                "kind": "Regular",
                "terminator": {"span": span.clone(), "kind": {"Call": {
                    "call": {
                        "func": {"Regular": {"kind": {"Fun": fun}, "generics": gens.clone()}},
                        "args": args,
                        "dest": place(dest, dest_ty)
                    },
                    "target": target,
                    "on_unwind": 5
                }}}
            })
        };
        let holder = serde_json::json!({
            "def_id": 0,
            "item_meta": item_meta(&["fixture", "holder"]),
            "signature": {"is_unsafe": false, "inputs": [], "output": scalar.clone()},
            "body": {"Unstructured": {
                "span": span.clone(),
                "locals": {"arg_count": 0, "locals": [
                    {"index": 0, "name": null, "span": span.clone(), "ty": scalar.clone()},
                    {"index": 1, "name": "scope", "span": span.clone(), "ty": scope_ty.clone()},
                    {"index": 2, "name": "obj", "span": span.clone(), "ty": gc_ty.clone()},
                    {"index": 3, "name": "tmp", "span": span.clone(), "ty": scalar.clone()}
                ]},
                "body": [
                    call_bb(3, serde_json::json!([]), 1, &scope_ty, 1),
                    call_bb(
                        2,
                        serde_json::json!([{"Copy": place(2, &gc_ty)}]),
                        2,
                        &gc_ty,
                        2
                    ),
                    call_bb(1, serde_json::json!([]), 3, &scalar, 3),
                    call_bb(
                        4,
                        serde_json::json!([{"Copy": place(2, &gc_ty)}]),
                        3,
                        &scalar,
                        4
                    ),
                    {"statements": [], "kind": "Regular", "terminator": {
                        "span": span.clone(), "kind": "Return"
                    }},
                    {"statements": [], "kind": "Cleanup", "terminator": {
                        "span": span.clone(), "kind": {"Goto": {"target": 2}}
                    }}
                ]
            }}
        });
        let opaque = |id: u64, path: &[&str], output: &serde_json::Value| {
            serde_json::json!({
                "def_id": id,
                "item_meta": item_meta(path),
                "signature": {"is_unsafe": false, "inputs": [], "output": output.clone()},
                "body": "Opaque"
            })
        };
        let file = serde_json::json!({
            "charon_version": "0.1.201",
            "has_errors": false,
            "translated": {
                "crate_name": "fixture",
                "type_decls": [],
                "fun_decls": [
                    holder,
                    opaque(1, &["fixture", "collect"], &scalar),
                    opaque(2, &["pyre_object", "gc_roots", "pin_root"], &gc_ty),
                    opaque(3, &["pyre_object", "gc_roots", "push_roots"], &scope_ty),
                    opaque(4, &["fixture", "touch"], &scalar)
                ],
                "global_decls": [],
                "trait_decls": [],
                "trait_impls": []
            }
        });
        let llbc = majit_charon_reader::Llbc::from_slice(file.to_string().as_bytes())
            .expect("non-forward pred fixture parses");
        let cg = super::super::framework::build(&llbc);
        let mut reach = HashSet::new();
        reach.insert(0);
        reach.insert(1);
        let mut push_roots = HashSet::new();
        push_roots.insert(3);
        let mut gc_tys = HashSet::new();
        gc_tys.insert(100);
        let (_, stats) = scan(
            &llbc,
            &cg,
            &reach,
            &push_roots,
            &gc_tys,
            &HashSet::new(),
            &[],
        );
        assert_eq!(
            stats.unparsed_terminator_bodies, 0,
            "terminators must parse"
        );
        assert_eq!(stats.unparsed_statement_bodies, 0, "statements must parse");
        assert_eq!(
            stats.withheld_contents_opaque, 0,
            "pin transfer must be readable: {:?}",
            stats.opaque_by_reason
        );
        assert_eq!(stats.withheld_under_a_bracket, 1);
        assert_eq!(
            stats.short_brackets.len(),
            1,
            "short={} covers={}",
            stats.withheld_bracket_short,
            stats.withheld_bracket_covers
        );
        assert_eq!(
            stats.short_brackets[0].missing,
            vec!["obj".to_string()],
            "pinned={:?}",
            stats.short_brackets[0].pinned
        );
        assert!(
            stats.short_brackets[0].pinned.is_empty(),
            "the successor in-set must be empty on the pin: {:?}",
            stats.short_brackets[0].pinned
        );
    }

    /// Pin facts of every Unstructured body in pyre-object equal those of
    /// `strip_cleanup_blocks` of that body. Call lists are the liveness
    /// scan's per-body input at reasonable cost.
    #[test]
    fn object_ullbc_pin_facts_match_strip_cleanup_blocks() {
        use majit_charon_reader::ullbc::strip_cleanup_blocks;

        let path = crate::runtime_names::artifacts::OBJECT_ULLBC;
        let llbc = majit_charon_reader::Llbc::load(path).expect("load pyre-object LLBC");
        let cg = super::super::framework::build(&llbc);
        let push: HashSet<u64> = cg
            .names
            .iter()
            .filter(|(_, n)| n.ends_with("gc_roots::push_roots"))
            .map(|(&id, _)| id)
            .collect();
        for fd in llbc.iter_fun_decls() {
            let Some(body) = fd.unstructured() else {
                continue;
            };
            let stripped = strip_cleanup_blocks(fd.unstructured().expect("Unstructured body"));
            let retained = helper_body_fact(&llbc, &body, &push, &cg.names);
            let stripped_fact = helper_body_fact(&llbc, &stripped, &push, &cg.names);
            match (retained, stripped_fact) {
                (None, None) => {}
                (Some(a), Some(b)) => {
                    assert_eq!(
                        pin_facts(&a),
                        pin_facts(&b),
                        "pin facts of {} (id {}) must equal strip_cleanup_blocks",
                        fd.item_meta.name_path(),
                        fd.def_id
                    );
                }
                (a, b) => panic!(
                    "pin-fact presence of {} (id {}) diverged from strip_cleanup_blocks: retained={} stripped={}",
                    fd.item_meta.name_path(),
                    fd.def_id,
                    a.is_some(),
                    b.is_some()
                ),
            }
        }
    }
}
