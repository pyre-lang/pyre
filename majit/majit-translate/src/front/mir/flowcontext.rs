//! `FlowContext` / `build_flow` (`rpython/flowspace/flowcontext.py`).

use super::*;

/// `objectmodel.keepalive_until_here`.
pub(crate) const KEEPALIVE_UNTIL_HERE_PATH: &str =
    "rpython::rlib::objectmodel::keepalive_until_here";

/// The `(base, index)` operands of a devirtualized workspace index call,
/// recorded for the paired `*p = v` write.  Each operand keeps the
/// resolving-block Variable plus its source MIR local (`None` for a
/// constant operand) so the write site can re-resolve through
/// `local_var` after the operand is rebound across the call's block
/// split.  See [`Lowering::index_elem_alias`].
#[derive(Clone)]
pub(in crate::front::mir) struct IndexElemAlias {
    pub(in crate::front::mir) base_local: Option<usize>,
    pub(in crate::front::mir) base_var: Variable,
    pub(in crate::front::mir) index_local: Option<usize>,
    pub(in crate::front::mir) index_var: Variable,
    /// External item bank the source-level list exposes.  This can differ
    /// from the Rust pointee spelling: `[str]` is physically `GcArray(GCREF)`
    /// but its get/set boundary is `StringRepr`.
    pub(in crate::front::mir) item_ty: ValueType,
    /// Identity the paired write must repeat, so `arr[i] = v` keys the
    /// same ARRAY as the `arr[i]` read that recorded this alias.
    pub(in crate::front::mir) array_type_id: Option<String>,
    /// The item kind when `base_var` is the header address of a `Vec` of
    /// one-word items: the paired write then calls `ll_vec_setitem_fast`.
    pub(in crate::front::mir) rust_vec: Option<majit_ir::rvec::VecItemKind>,
    /// List spelling the paired read and write narrow the base through.
    /// `None` when the base is already that list (a `Vec` / slice place).
    /// An object-items pointer is not: its descr key is
    /// [`OBJECT_REF_GCARRAY_TYPE_ID`], which `project_struct_field_type`
    /// does not read as a list, so `getitem` would stay on the
    /// classdef-less pointer.
    pub(in crate::front::mir) list_root: Option<String>,
    /// `BytesArray` / `UnicodeArray` when the paired read substituted
    /// `ll_items` (`block`) for the `{ block, len }` wrapper. The write
    /// repeats that FieldRead off the remapped wrapper (`ll_setitem_fast`).
    pub(in crate::front::mir) items_owner: Option<&'static str>,
    /// Write-only `index_mut`. `ll_setitem_fast` stores the item and does
    /// not read the old one, so the local's Variable is not an element
    /// value and `edge_args` does not pass it.
    pub(in crate::front::mir) no_element: bool,
}

/// Header of a `(*p.add(i)).field` access.
///
/// `getinteriorfield` (`rewrite_op_getinteriorfield`) addresses
/// `entries[index].field` from the GcArray object. The descr adds
/// `get_array_token`'s items offset (`array_items_base`), so the base is
/// that object, not the item pointer `entries_item_ptr` / `entry_ptr` return.
#[derive(Clone)]
pub(in crate::front::mir) enum InteriorHeader {
    /// The add base is already the array object, or `entries_item_ptr`'s
    /// argument is. Both are one MIR local.
    Local { local: usize, var: Variable },
    /// `entry_ptr(&dict)`. The use site emits `FieldRead entries` off the
    /// receiver, then the interior op off that result.
    Entries {
        recv_local: usize,
        recv_var: Variable,
        field: FieldDescriptor,
        ty: ValueType,
    },
}

/// A `ptr::add` whose result is used only as `(*dest).field`. The field
/// itself is known at the use, so the add records the header and index
/// and aliases the destination to the item pointer for every other consumer.
#[derive(Clone)]
pub(in crate::front::mir) struct InteriorFieldAlias {
    pub(in crate::front::mir) header: InteriorHeader,
    pub(in crate::front::mir) index_local: usize,
    pub(in crate::front::mir) index_var: Variable,
    /// `GcArray<Entry>` / `GcArray<Point>`. `extract_element_type_from_str`
    /// takes the angle-bracket interior, so the leaf is
    /// `strip_instantiation_suffix`'d. The `GcArray<>` wrapper is the
    /// length-prefixed ARRAY spelling: `nolength_from_array_type_id` is
    /// false, matching `interiorfielddescrof_keyed`.
    pub(in crate::front::mir) array_type_id: String,
}

/// A native conversion's temporary aggregate tail, consumed immediately after
/// MIR framestate threading. This is frontend work order, not a ClassDef or
/// per-box side table: annotation only sees ordinary variant constructors and
/// branch/join links (flowcontext.py::BlockRecorder.guessbool).
pub(in crate::front::mir) struct TaggedPairAggregateSite {
    pub(in crate::front::mir) block: BlockId,
    pub(in crate::front::mir) operation_index: usize,
    pub(in crate::front::mir) result: Variable,
    pub(in crate::front::mir) owner: String,
    pub(in crate::front::mir) payload_owner: String,
    pub(in crate::front::mir) disc: Variable,
    pub(in crate::front::mir) payload: Variable,
    pub(in crate::front::mir) payload_ty: ValueType,
}

pub(in crate::front::mir) struct Lowering<'a> {
    pub(in crate::front::mir) graph: FunctionGraph,
    pub(in crate::front::mir) llbc: &'a Llbc,
    /// Cross-crate struct field harvest. An Opaque scalar-word projection
    /// reads the declared name from the defining crate's harvest; `__pos_N`
    /// is only the missing-row fallback. `None` in unit tests that do not
    /// lower Opaque scalar fields.
    pub(in crate::front::mir) struct_fields:
        Option<&'a crate::front::semantic::StructFieldRegistry>,
    /// Harvested `_alloc_flavor_ = raw` structs. `get_type_flag` banks a
    /// pointer to one as an unsigned int word (`descr.py get_type_flag`).
    pub(in crate::front::mir) raw_alloc_flavor_structs: std::collections::HashSet<String>,
    /// Harvested `#[dont_look_inside]` marker set (keyed
    /// `strip_crate_prefix(name_path())`, the same spelling
    /// `stamp_return_token` uses at the whole-program registration loop).
    /// A residual-marked inherent method must not route as
    /// `CallTarget::Method`: that surfaces `getattr(recv, method)` on a
    /// classed receiver whose classdict carries no method source, which
    /// blocks at annotate. The call-target builder declines the Method
    /// hint for a callee in this set so it routes as a `FunctionPath` the
    /// registry resolves to the same residual fnaddr.
    pub(in crate::front::mir) dont_look_inside: &'a std::collections::HashSet<String>,
    /// Leaves whose `struct_origins` entry `harden_duplicate_leaf_metadata`
    /// cleared. A duplicated leaf is not a class; [`adt_node_class_root_with`]
    /// paints the declaration path instead. Owned by the program build
    /// that computed `struct_origins`, and borrowed for this lowering.
    pub(in crate::front::mir) tombstoned_leaves: &'a std::collections::HashSet<String>,
    /// Gc structs and enums for this lowering. Absence is Raw.
    pub(in crate::front::mir) gc_struct_ids:
        &'a std::collections::HashSet<majit_ir::descr::StructId>,
    /// `_immutable_fields_` harvested once for this artefact.
    pub(in crate::front::mir) immutable_fields: &'a ImmutableFieldsTable,
    pub(in crate::front::mir) body: &'a Unstructured,
    pub(in crate::front::mir) static_addrs: crate::HostStaticAddrs<'a>,
    /// Configured receiver-type paths whose marker methods identify a
    /// `JitDriver`.  The same roots feed `jit_marker_key_from_target` in the
    /// codewriter; carrying them into the front end lets a read of the
    /// corresponding prebuilt driver const become a value constant before
    /// the nullary residual-call fallback.
    pub(in crate::front::mir) jitdriver_receiver_roots: &'a [String],
    pub(in crate::front::mir) arg_count: usize,
    /// `local_var[i] = Some(LocalValue::One(var))` once MIR local `i` has been bound to
    /// a flowspace Variable. Slot 0 is the return value, 1..arg_count
    /// are arguments, the rest are introduced lazily by the first
    /// `Assign` that writes them. Local 0 stays `None` until a Return
    /// terminator wires it up — the Return path reads MIR local 0 and
    /// drops a `Link([value], returnblock)` so we never need to mint
    /// a Variable for it.
    pub(in crate::front::mir) local_var: Vec<Option<LocalValue>>,
    /// Shadow slot carrying a pair's length, or a slice iterator's item
    /// pointer, across block inputargs and link args. Readers take the
    /// words from [`LocalValue`], not from this table.
    pub(in crate::front::mir) len_shadow: Vec<Option<usize>>,
    /// Extra slot carrying a window's GC owner (`rffi.get_raw_address_of_string`
    /// / list header) across block inputargs. Not a startblock ABI param.
    pub(in crate::front::mir) owner_shadow: Vec<Option<usize>>,
    /// The local slot holding the trailing length-slot parameter of a body
    /// that returns a pair slice.
    pub(in crate::front::mir) ret_len_slot_local: Option<usize>,

    /// `raw_array[i]` is the role MIR local `i` plays in a `[T; N]` array
    /// kept in a raw item buffer (see [`raw_array_roles`]).
    pub(in crate::front::mir) raw_array: Vec<Option<RawArrayRole>>,
    /// MIR locals whose value is the address word of a slice `get` /
    /// `first` / `last` option (and `copied` / `cloned` of one). One-shot,
    /// from the body, like [`raw_array`].
    pub(in crate::front::mir) item_addr: Vec<bool>,
    /// `slice_next_iter[opt] = Some(iter)` when `opt` is the destination of
    /// `next` on pair-slice iterator local `iter`. One-shot, from the body.
    pub(in crate::front::mir) slice_next_iter: Vec<Option<usize>>,
    /// `block_id[i]` = FunctionGraph BlockId for MIR basic block `i`,
    /// when `i` is in the lowered closure. Incoming edges always target
    /// this entry. A bounds fork does not retarget it.
    pub(in crate::front::mir) block_id: MirBlockMap,
    /// Graph block that currently receives operations for MIR `i`. Starts
    /// equal to [`Self::block_id`]; a bounds fork moves it to the in-bounds
    /// tail so later ops and the terminator stay after the guard.
    pub(in crate::front::mir) emit_block: MirBlockMap,
    /// [`forward_reachable_mask`] of this body. Scans that classify a
    /// local from every assignment consult it and skip the rest.
    pub(in crate::front::mir) forward_reached: Vec<bool>,
    /// Locals some forward-reached `Call` terminator writes whole, for
    /// [`Self::local_is_call_dest`]; built on first query.
    pub(in crate::front::mir) call_dest_locals: std::cell::OnceCell<bit_set::BitSet>,
    /// Root locals of some forward-reached projected `Assign` destination,
    /// for [`Self::local_has_projected_store`]; built on first query.
    pub(in crate::front::mir) projected_store_roots: std::cell::OnceCell<bit_set::BitSet>,
    /// MIR locals that are live when entering each block. Non-entry
    /// blocks receive these through `Block.inputargs`, and predecessor
    /// edges pass the matching current Variables via `Link.args`.
    pub(in crate::front::mir) block_live_in: Vec<bit_set::BitSet>,
    /// MIR locals this body moves out of.  A root-bracket guard among them is
    /// dropped under an initialisation flag the artefact does not carry, so
    /// its bracket is left open rather than closed on a path that may not own
    /// it (see [`moved_out_locals`]).
    pub(in crate::front::mir) root_scope_moved_locals: MovedOutLocals<'a>,
    /// Root brackets this body keeps out of its jitcode entirely
    /// (see [`RootBracketPlan`]).
    pub(in crate::front::mir) root_bracket: RootBracketPlan,
    /// rbigint owner-root handles this body lowers as the value they root
    /// (see [`OwnerRootPlan`]).
    pub(in crate::front::mir) owner_root: OwnerRootPlan,
    /// The `RBigIntGcRoot::new` call that re-roots the producer value the
    /// current call terminator is lowering (see
    /// [`Self::owner_root_reroot_call`]).
    pub(in crate::front::mir) owner_root_reroot: Option<CallPayload>,
    pub(in crate::front::mir) block_entry_local_var: Vec<PackedLocalRow>,
    /// Fresh block-input id → the predecessor Variable `FrameState.copy`
    /// minted it from. The target body is lowered before pass 2 writes
    /// those pairs as link args, so a use in the target reads the map.
    pub(in crate::front::mir) input_copied_from: std::collections::HashMap<u64, u64>,
    /// Header addresses of a `Vec<PyObjectRef>`. A whole-list mutable
    /// view (`deref_mut`, `as_mut_slice`, `index_mut(RangeFull)`) aliases
    /// `&mut vec` / `&mut vec[..]` to this word. A block input copied from
    /// it is recorded too.
    pub(in crate::front::mir) object_vec_headers: std::collections::HashSet<u64>,
    /// Definitely-initialised `Vec` Drops recorded while lowering. The
    /// `ll_vec_free_*` is emitted after block links are in SSA-final form
    /// (framestate Pass 2 / monotonic `edge_args`), because
    /// `rewrite_op_free` follows the header through those links and a
    /// Pass-1 leftover inputarg position does not match a live-in link.
    pub(in crate::front::mir) pending_vec_frees:
        Vec<(BlockId, Variable, majit_ir::rvec::VecItemKind)>,
    pub(in crate::front::mir) block_entry_positional_aggregate_locals:
        Vec<std::collections::HashMap<usize, String>>,
    pub(in crate::front::mir) block_positional_seen: Vec<bit_set::BitSet>,
    pub(in crate::front::mir) block_positional_conflict: Vec<bit_set::BitSet>,
    pub(in crate::front::mir) block_entry_string_byte_view_locals: Vec<bit_set::BitSet>,
    pub(in crate::front::mir) block_byte_view_seen: Vec<bit_set::BitSet>,
    pub(in crate::front::mir) block_byte_view_conflict: Vec<bit_set::BitSet>,
    pub(in crate::front::mir) block_entry_list_view_locals: Vec<bit_set::BitSet>,
    pub(in crate::front::mir) block_list_view_seen: Vec<bit_set::BitSet>,
    pub(in crate::front::mir) block_list_view_conflict: Vec<bit_set::BitSet>,
    /// Maps each MIR local whose current binding was produced by a
    /// positional [`Rvalue::Aggregate`] (tuple / array / closure — any
    /// kind for which [`Lowering::resolve_aggregate_adt`] returns
    /// `None`) to the `owner_root` its construction-side `FieldWrite`
    /// chain used.  Such a local holds a synthetic transparent-ctor
    /// base with a `__pos_<i>` `FieldWrite` chain, so its `.N` reads
    /// must resolve to a symmetric `FieldRead __pos_<N>` in
    /// [`Lowering::resolve_place`] — carrying the *same* `owner_root` —
    /// rather than collapsing to the base Variable.  The stored owner
    /// is required because Charon's tuple `Aggregate` kind serialises
    /// as `{"Adt": [{"id": 0, "builtin": "Tuple", ..}, ..]}` (owner_root `"Adt"`)
    /// while the matching `Field` projection container serialises as
    /// `{"Tuple": N}`, so the read side cannot re-derive the
    /// construction owner from its own payload.  Excludes the
    /// `*Checked` `(value, bool)`-as-`BinOp` locals (those are bound by
    /// [`Rvalue::BinaryOp`], never an Aggregate), so their `.0` reads
    /// still fall through.
    pub(in crate::front::mir) positional_aggregate_locals: std::collections::HashMap<usize, String>,
    /// MIR locals bound by a scalar [`Rvalue::BinaryOp`] anywhere in the
    /// body.  A `*Checked (value, bool)` operation lowers to a single
    /// scalar `BinOp` (the [`Rvalue::BinaryOp`] arm), so the destination
    /// local — though MIR-typed `(numeric, bool)` — holds one scalar
    /// Variable, not a tuple.  Its `.0` projection therefore collapses to
    /// that scalar instead of extracting a tuple element; a `.1` read is
    /// the Rust overflow bit, which the JIT IR does not model, so it fails
    /// loud in [`Lowering::resolve_place`] rather than aliasing the
    /// overflow bool to the arithmetic value.  A MIR local's type is
    /// fixed, so a local bound by `BinaryOp` is a scalar at every read
    /// site (its `(numeric, bool)` type can never alias a genuine data
    /// tuple); a single function-wide set needs no per-block propagation.
    /// Distinguishes the collapse case from a genuine Ref tuple `.N` read
    /// in [`Lowering::resolve_place`].
    pub(in crate::front::mir) binop_result_locals: std::collections::HashSet<usize>,
    /// Whether this function contains a builder-form accumulator. The canonical
    /// lowering sets it directly from [`graph_has_builder_accumulator`]; ctor /
    /// append / terminal-materialisation sites then emit the
    /// `__majit_stringbuilder_{new,append,build}` markers (`flowspace_adapter` →
    /// `newstringbuilder` / `append` / `build`) instead of a parallel
    /// `ll_strconcat` graph. Whether a given local *is* such an
    /// accumulator is [`accum`](Self::accum), filled once from
    /// [`is_builder_mode_accumulator`] when this lowering starts.
    pub(in crate::front::mir) builder_mode: bool,
    /// Per-local string-builder recognizer answers for this body.
    pub(in crate::front::mir) accum: AccumulatorFacts,
    /// MIR locals bound by a devirtualized workspace `Index::index` /
    /// `IndexMut::index_mut` call, mapped to the `(base, index)`
    /// operand pair.  Those impls bottom out at raw-slice
    /// construction (`as_mut_slice` → `from_raw_parts`), which has no
    /// graph lowering, so the callsite is lowered as RPython's
    /// getarrayitem instead and the paired `*p = v` write
    /// ([`Lowering::emit_projection_write`] `Deref` arm) consults
    /// this map to emit `ArrayWrite` rather than the opaque
    /// `__deref_write` marker.  The base/index are kept as both the
    /// resolving-block Variable and their source MIR local: the write
    /// usually lands in a later block (the index call terminates its
    /// own block), where the operands have been rebound to fresh
    /// inputarg Variables, so the consumer re-resolves through
    /// `local_var` by local and only falls back to the recorded
    /// Variable for a base/index without a backing local (a constant).
    pub(in crate::front::mir) index_elem_alias: std::collections::HashMap<usize, IndexElemAlias>,
    /// MIR locals bound by `ptr::add` whose result is used only as
    /// `(*dest).field` on a struct pointee. The use emits
    /// `InteriorFieldRead` / `InteriorFieldWrite` off the recovered GcArray
    /// header (`getinteriorfield`), not off the item pointer. Same
    /// block-split re-resolve as [`Lowering::index_elem_alias`]: the header
    /// and index are kept as the resolving-block Variable plus the source
    /// MIR local.
    pub(in crate::front::mir) interior_field_alias:
        std::collections::HashMap<usize, InteriorFieldAlias>,
    /// MIR locals bound by `_i = &place` where `place` is a
    /// `core::sync::atomic` slot, mapped to the place they borrowed.
    ///
    /// `<Atomic*>::store(&self, v, ord)` and `core::mem::{replace,swap,take}`
    /// are calls, so the write target survives only as an operand.  The load
    /// fold gets away with aliasing the receiver Variable because a read
    /// needs no place, but a write does.  Every single-assignment borrow is
    /// recorded; the atomic store still fires only for an atomic method.
    ///
    /// Only locals outside [`Lowering::multi_assigned_locals`] enter, the
    /// same restriction [`Lowering::const_discriminant_locals`] carries and
    /// for the same reason: MIR locals are not SSA and this map is not a
    /// flow-state, so a local borrowed from `&a` in one block and `&b` in
    /// another would hand the store whichever block happened to be lowered
    /// last rather than the definition that reaches it — a write to the
    /// wrong field.  A local assigned once in the whole body carries the
    /// reaching place at every use of it; a rebound one stays a call.
    pub(in crate::front::mir) atomic_ref_place: std::collections::HashMap<usize, Place>,
    /// Parameter locals declared `&mut T` where `T` is a GC reference.
    /// The local holds the one-field cell, not the reference word.
    pub(in crate::front::mir) gc_mut_ref_params: std::collections::HashSet<usize>,
    /// MIR locals whose value is the address of a primitive, produced by a
    /// call that returns `&T` / `&mut T` (`MutexGuard::deref` → `&bool`).
    ///
    /// `Rvalue::Ref` does not enter: that borrow aliases the scalar, and
    /// [`Lowering::atomic_ref_place`] already names the place. A copy or
    /// move of one of these locals carries the same address. Only locals
    /// outside [`Lowering::multi_assigned_locals`] enter, for the same
    /// reason as [`Lowering::atomic_ref_place`].
    pub(in crate::front::mir) scalar_address_locals: Vec<usize>,
    /// Function-local primitive scalars whose address is a raw scalar
    /// pointer argument. `local_var` holds the Char-varsize cell pointer;
    /// reads `raw_load`, writes `raw_store`.
    pub(in crate::front::mir) cell_locals: bit_set::BitSet,
    pub(in crate::front::mir) cell_descr: Vec<Option<(ValueType, usize, bool)>>,
    pub(in crate::front::mir) cell_has_storage_live: bit_set::BitSet,
    /// MIR locals bound to a `core::sync::atomic::Ordering` value, mapped to
    /// the variant's name.
    ///
    /// Charon does not hand the ordering to `load` / `store` as a constant
    /// operand: it assigns `_i = Ordering::<V>` as a fieldless-enum
    /// `Aggregate` and passes `move _i`, so the variant is only knowable from
    /// that statement.  [`Lowering::is_atomic_store`]'s arm folds a `Relaxed`
    /// store and nothing else — a `Release` store lowered as an unordered
    /// `FieldWrite` would DROP a barrier the residual call it replaces still
    /// performs (`w_type_set_version_tag` publishes `version_tag` with
    /// `Release` against an `Acquire` reader).  Which makes reading the WRONG
    /// variant a dropped barrier, so this map carries the single-assignment
    /// restriction for the same reason [`Lowering::atomic_ref_place`] does.
    pub(in crate::front::mir) atomic_ordering_locals: std::collections::HashMap<usize, String>,
    /// `FunctionDesc.cachedgraph` for this lowering. `None` outside the
    /// whole-program walk.
    pub(in crate::front::mir) spec:
        Option<&'a std::cell::RefCell<crate::front::clause_spec::SpecQueue>>,
    /// This body is a `cachedgraph` copy, so clause refs have been
    /// replaced and a `TraitImpl` call names the impl method.
    pub(in crate::front::mir) spec_body: bool,
    /// MIR locals whose enum discriminant is a translation-time
    /// constant: single-assignment locals bound by an always-`Ok`
    /// decomposed conversion ([`Lowering::try_lower_usize_try_from`]).
    /// `Rvalue::Discriminant` on such a local emits `ConstInt(tag)`
    /// instead of the synthetic FieldRead, which lets the
    /// `lower_switch` Constant fold drop the statically dead
    /// `Err`/panic arm even though MIR calls terminate their block
    /// (the switch always sits in a successor block).  Only locals
    /// outside [`Lowering::multi_assigned_locals`] enter this map, so
    /// a re-bound local can never carry a stale tag.
    pub(in crate::front::mir) const_discriminant_locals: std::collections::HashMap<usize, i64>,
    /// Integer fields of a single-assignment local bound to a multi-field
    /// integer NamedConst. Indexed by MIR local. A field projection emits
    /// the literal; reading the local as one word is refused. The whole
    /// struct is not a flow op: an unregistered nullary call of the const
    /// would still fail annotation.
    pub(in crate::front::mir) const_struct_fields: Vec<Option<Vec<u64>>>,
    /// MIR locals assigned more than once anywhere in the body
    /// (statement assigns + call destinations).  Guard set for
    /// [`Lowering::const_discriminant_locals`].
    pub(in crate::front::mir) multi_assigned_locals: std::collections::HashSet<usize>,
    /// String variables whose Rust source obtained a `&[u8]` through
    /// `str`/`Wtf8::as_bytes`. RPython stores UTF-8/WTF-8 in a byte string,
    /// where `ord(s[i])` is the scalar byte read; the consumer gate below
    /// uses this provenance to emit exactly that pair of flowspace ops.
    /// Restored per block from [`Lowering::block_entry_string_byte_view_locals`],
    /// same shape as [`Lowering::positional_aggregate_locals`].
    pub(in crate::front::mir) string_byte_view_locals: Vec<usize>,
    /// Shared `&[u8]` locals that alias a W2 list owner (`Vec<u8>` /
    /// `Box<[u8]>` / `&mut [u8]`). One list word; converted to a window
    /// only at a window boundary. Restored per block from
    /// [`Lowering::block_entry_list_view_locals`].
    pub(in crate::front::mir) list_view_locals: Vec<usize>,
    /// Per-local: the list-view owner is a fixed-size list (`Box<[u8]>` /
    /// `[u8; N]`). Whole-body, from [`body_owner_view_locals`].
    pub(in crate::front::mir) list_view_fixed: Vec<bool>,
    /// Field-read results whose declared type is an inline fat pointer
    /// (`Box<[T]>`, `Box<str>`, `Box<dyn Trait>`). The length is the
    /// metadata word and an index uses the data word; neither is a
    /// length-prefixed GcArray. One field feeds both uses, so the mark
    /// sits on the result variable rather than on the shared read.
    pub(in crate::front::mir) fat_box_vars: Vec<Variable>,
    /// Results of calls whose declared type is a fat `&dyn` or `Box<dyn>`.
    /// `FunctionGraph::set_return` carries one word, so
    /// [`crate::front::dyn_fat`] splices these callees and reads `method_*`
    /// from the metadata word of the fat field.
    pub(in crate::front::mir) dyn_return_calls: Vec<Variable>,
    /// MIR locals bound by [`Lowering::is_prebuilt_once_lock_get_or_init`].
    /// Each holds the `&usize` a `OnceLock<usize>` singleton hands back, and
    /// its word is the registered GC object, so a later `*local as *mut T`
    /// is that reference unchanged rather than a `cast_int_to_ptr`.
    pub(in crate::front::mir) prebuilt_once_value_locals: Vec<usize>,
    /// MIR locals holding the `ll_items(l)` view returned by the string-list
    /// slice adapters, each paired with the list's LOGICAL length.  Rust
    /// spells the element as a concrete raw pointer, but both byte and
    /// unicode strategies expose the same external `str` over one internal
    /// `GcArray(GCREF)`.
    ///
    /// The pair exists because the fold drops the fat slice's second word.
    /// That word is `l.length`, which upstream keeps strictly apart from the
    /// array header's `ll_fixed_length` / `arraylen_gc`
    /// (`rlist.py, 395-397`: `ll_length` reads `l.length` while
    /// `ll_getitem_fast` asserts against it).  A `Rvalue::Len` over one of
    /// these locals answers with the recorded length, never with `__len` of
    /// the header, which is the CAPACITY.
    pub(in crate::front::mir) string_array_view_locals: Vec<(usize, Variable)>,
    /// Twin of [`Self::string_array_view_locals`] for `IntArray` /
    /// `FloatArray` slice views. Float pair slices are outside
    /// [`PAIR_SLICE_ITEM_KINDS`], so a length read of the `[f64]` view
    /// would otherwise be `ll_fixed_length` of the items block.
    pub(in crate::front::mir) numeric_array_view_locals: Vec<(usize, Variable)>,
    /// Result `Variable`s of calls to callees whose declared result is
    /// `Result<T, PyError>`.  Each
    /// heads a `Try::branch` diamond that
    /// [`crate::front::result_exc::rewire_result_exc_call_sites`]
    /// rewires into `ExitSwitch::LastException` exits after the body
    /// lowering completes.  The paired `Option<String>` is the callee's
    /// per-instantiation `<…>` suffix (Ref-shaped payloads only) keying
    /// the rebuilt `Ok`/`Err` shells' ClassDef per instantiation.
    pub(in crate::front::mir) result_exc_call_results: Vec<(Variable, Option<String>, ValueType)>,
    /// Foreign `Option::ok_or_else(opt, closure)` results consumed immediately
    /// by `?`, recorded for the combined value-or-exception rewrite in
    /// [`crate::front::result_exc::rewire_option_ok_or_else_try_sites`].
    /// This cannot share `result_exc_call_results`: the Option combinator is
    /// not itself a translated scoped callee, so its `Result` shell must be
    /// removed together with the Option select.
    pub(in crate::front::mir) option_ok_or_else_try_sites:
        Vec<crate::front::result_exc::OptionOkOrElseTrySite>,
    /// Carrier `Result<T, PyError>` combinator sites (`unwrap_or` / `expect`
    /// / `unwrap`) fused by
    /// [`crate::front::result_exc::rewire_carrier_combinator_sites`] into the
    /// call's `LastException` join before the caller rule can catch-and-rewrap
    /// the call. `expect` / `unwrap` raise the implicit `AssertionError`
    /// (`core` `Result::expect`; `simplify.py` `remove_assertion_errors`;
    /// `interp_deque.py` `W_Deque.trimleft` / `popleft`).
    pub(in crate::front::mir) carrier_combinator_sites:
        Vec<crate::front::result_exc::CarrierCombinatorSite>,
    /// `Result::map_err(result, closure)` sites lowered to an explicit
    /// Ok/Err closure-select before the exception-link pass consumes the
    /// newly built result shells.
    pub(in crate::front::mir) result_map_err_sites:
        Vec<crate::front::result_map_err::ResultMapErrSite>,
    pub(in crate::front::mir) result_expect_sites: Vec<crate::front::option_unwrap::UnwrapSite>,
    /// `Iterator::next()` call results (`Option<T>`-typed) recorded for
    /// the `next`-diamond rewiring pass (`front::iter_next`) that runs
    /// after the body lowering completes.  The paired [`ValueType`] is the
    /// element `T` with the `&` a slice iterator adds peeled off — the only
    /// place the element's kind is still readable, since the fold's op
    /// carries the iterator and not the container's item type.
    pub(in crate::front::mir) next_call_results: Vec<(Variable, ValueType)>,
    /// `Map::collect` construction sites recorded for the loop rewrite
    /// `front::iter_adapter` synthesizes before the `next`-diamond runs
    /// (see [`crate::front::iter_adapter::MapCollectSite`]).
    pub(in crate::front::mir) map_collect_sites: Vec<crate::front::iter_adapter::MapCollectSite>,
    /// `i64::checked_{add,sub,mul}()` call results (`Option<i64>`-typed)
    /// recorded for the checked-arith rewiring pass
    /// (`front::checked_arith`) that runs after the body lowering
    /// completes, rewriting each into an `*_ovf` op + OverflowError edge.
    pub(in crate::front::mir) checked_arith_call_results: Vec<Variable>,
    /// `Option::ok_or_else` continuations that may directly consume one of
    /// the signed checked-arith results above.  Type-owned closure/Result
    /// metadata is captured here so `front::checked_arith` can fuse that
    /// continuation into the `*_ovf` normal/overflow edges.
    pub(in crate::front::mir) checked_arith_ok_or_else_sites:
        Vec<crate::front::checked_arith::CheckedArithOkOrElseSite>,
    /// Unsigned `usize::checked_{add,mul}()` call sites (`Option<usize>`-
    /// typed) recorded for the native-overflow rewiring pass
    /// (`front::checked_arith_uint`) that runs after `front::checked_arith`,
    /// replacing each residual with `uint_mul_high`/`uint_lt`-based overflow
    /// tests + a virtualized `Option`.  Kept separate from
    /// `checked_arith_call_results` because it needs the `Option`/`Some`
    /// owners + payload type resolved at the recording site.
    pub(in crate::front::mir) checked_arith_uint_sites:
        Vec<crate::front::checked_arith_uint::CheckedArithUintSite>,
    pub(in crate::front::mir) tagged_pair_aggregate_sites: Vec<TaggedPairAggregateSite>,
    /// `Layout::from_size_align(size, const_align).ok()` call sites
    /// (`Option<Layout>`-typed) recorded for the native bound-check rewiring
    /// pass (`front::from_size_align`), which replaces the `from_size_align` +
    /// `ok` residual pair with a `uint_lt` bound test + a virtualized nested
    /// `Option<Layout>` aggregate.  The `Option`/`Some`/`Layout` owners and the
    /// payload type are resolved here from the `.ok()` destination type.
    pub(in crate::front::mir) from_size_align_sites:
        Vec<crate::front::from_size_align::FromSizeAlignSite>,
    pub(in crate::front::mir) from_size_align_expect_sites:
        Vec<crate::front::from_size_align::FromSizeAlignExpectSite>,
    /// `Layout::from_size_align(size, align)` producers (`Result<Layout, _>`),
    /// rewritten independently of a following `.ok()` / `.expect()` combinator.
    pub(in crate::front::mir) from_size_align_call_sites:
        Vec<crate::front::from_size_align::FromSizeAlignCallSite>,
    /// `Layout::size` / `Layout::align` field accessors on a virtualized Layout.
    pub(in crate::front::mir) layout_accessor_sites:
        Vec<crate::front::from_size_align::LayoutAccessorSite>,
    /// `Try::branch(opt)` call sites where `opt: Option<T>`, recorded for
    /// the `Option` `?` rewiring pass (`front::option_try`) that runs after
    /// body lowering completes.
    pub(in crate::front::mir) option_try_sites: Vec<crate::front::option_try::OptionTrySite>,
    /// `bool::then(cond, closure)` call sites recorded for the
    /// short-circuit `Option` diamond the `front::bool_then` post-pass
    /// synthesizes after the body lowering completes.
    pub(in crate::front::mir) bool_then_sites: Vec<crate::front::bool_then::BoolThenSite>,
    /// `<[T]>::first(slice)` call sites recorded for the length-checked
    /// `Option<&T>` diamond the `front::slice_first` post-pass synthesizes
    /// after the body lowering completes (see
    /// [`crate::front::slice_first::SliceFirstSite`]).
    pub(in crate::front::mir) slice_first_sites: Vec<crate::front::slice_first::SliceFirstSite>,
    /// `<[T]>::get(slice, i)` call sites recorded for the bounds-checked
    /// `Option<&T>` diamond the `front::slice_get` post-pass synthesizes
    /// after the body lowering completes (see
    /// [`crate::front::slice_get::SliceGetSite`]).
    pub(in crate::front::mir) slice_get_sites: Vec<crate::front::slice_get::SliceGetSite>,
    /// `<[u8]>::is_ascii(slice)` call sites recorded for the
    /// `rutf8.check_ascii` rewrite (`first_non_ascii_char(s) < 0`) the
    /// `front::slice_ascii` post-pass applies after body lowering (see
    /// [`crate::front::slice_ascii::SliceIsAsciiSite`]).
    pub(in crate::front::mir) slice_is_ascii_sites:
        Vec<crate::front::slice_ascii::SliceIsAsciiSite>,
    /// Word-sized `{u64,usize}::saturating_mul(a, b)` call results recorded
    /// for the unsigned high-word clamp diamond `front::saturating_mul`
    /// synthesizes after body lowering.
    pub(in crate::front::mir) saturating_mul_sites: Vec<Variable>,
    /// `core::intrinsics::saturating_add` sites recorded for the clamp
    /// diamond `front::saturating_intrinsic` synthesizes after body
    /// lowering (same post-pass as `saturating_mul`).
    pub(in crate::front::mir) saturating_add_sites:
        Vec<crate::front::saturating_intrinsic::SaturatingAddSite>,
    /// `core::intrinsics::saturating_sub` sites recorded for the clamp
    /// diamond `front::saturating_intrinsic` synthesizes after body
    /// lowering (same post-pass as `saturating_mul`).
    pub(in crate::front::mir) saturating_sub_sites:
        Vec<crate::front::saturating_intrinsic::SaturatingSubSite>,
    /// Exclusive int-`Range { start, end }` aggregates (`for _ in a..b`)
    /// and inclusive `RangeInclusive::new(a, b)` calls (`for _ in a..=b`)
    /// recorded for the `range()` / `range(a, b + 1)` + list-iter divert
    /// the `front::range_iter` post-pass synthesizes so the `next`-diamond
    /// folds the loop (see [`crate::front::range_iter::RangeNewSite`]).
    pub(in crate::front::mir) range_iter_new_sites: Vec<crate::front::range_iter::RangeNewSite>,
    /// RangeFrom aggregates retained for the orthodox
    /// `getslice(start, None)` rewrite. The slice-index consumer proves the
    /// bound is a `usize`, hence non-negative as required by RPython's
    /// `decompose_slice_args`.
    pub(in crate::front::mir) slice_index_rangefrom_sites:
        Vec<crate::front::slice_index::SliceIndexRangeFromSite>,
    /// RangeTo aggregates retained as candidates for the slice-index
    /// recognizer. The end's unsigned coloring is captured from MIR here,
    /// before the operand can become a block inputarg.
    pub(in crate::front::mir) slice_index_rangeto_sites:
        Vec<crate::front::slice_index::SliceIndexRangeToSite>,
    /// Full `Range { start, end }` aggregates retained for the orthodox
    /// `getslice(start, stop)` rewrite.  The same aggregate may also be a
    /// `range_iter` candidate; both passes are consumer-gated.
    pub(in crate::front::mir) slice_index_range_sites:
        Vec<crate::front::slice_index::SliceIndexRangeSite>,
    /// `Option::unwrap(opt)` call sites recorded for the discriminant guard the
    /// `front::option_unwrap` post-pass synthesizes (see
    /// [`crate::front::option_unwrap::UnwrapSite`]).
    pub(in crate::front::mir) unwrap_sites: Vec<crate::front::option_unwrap::UnwrapSite>,
    /// `Option::expect(opt, msg)` call sites recorded for the discriminant
    /// guard the `front::option_expect` post-pass synthesizes — the two-arg
    /// sibling of `unwrap` (the panic message is dropped; see
    /// [`crate::front::option_expect::ExpectSite`]).
    pub(in crate::front::mir) expect_sites: Vec<crate::front::option_expect::ExpectSite>,
    /// `Option::map_or(opt, default, closure)` call sites recorded for the
    /// discriminant closure-select the `front::option_map_or` post-pass
    /// synthesizes (see [`crate::front::option_map_or::MapOrSite`]).
    pub(in crate::front::mir) map_or_sites: Vec<crate::front::option_map_or::MapOrSite>,
    /// `Option::map`/`and_then`/`unwrap_or_else(opt, closure)` call sites
    /// recorded for the discriminant closure-select the
    /// `front::option_closure_select` post-pass synthesizes (see
    /// [`crate::front::option_closure_select::ClosureSelectSite`]).
    pub(in crate::front::mir) closure_select_sites:
        Vec<crate::front::option_closure_select::ClosureSelectSite>,
    /// Option/Result combinators whose body is a discriminant switch (plus an
    /// optional closure `call_once`).  Captured here so the post-pass can
    /// replace the opaque core residual with the if/else the flow graph
    /// would have had if the source had been written that way.
    pub(in crate::front::mir) disc_combinator_sites: Vec<DiscCombinatorSite>,
    /// `Try::branch(res)` sites whose receiver is a non-carrier `Result`.
    /// The Option sibling is [`Lowering::option_try_sites`]; exception-carrier
    /// Results stay on [`Lowering::result_exc_call_results`].
    pub(in crate::front::mir) result_try_sites: Vec<ResultTrySite>,
    /// Result-var ids of one-word niche `Option` discriminant reads folded to
    /// a pointer null-test (`ne(base, null_mut())`, `build_rvalue`
    /// `Rvalue::Discriminant` niche arm).  Such a discriminant is a `SomeBool`
    /// (the `ne` result), so a `SwitchInt` terminator on it must close with
    /// `ExitCase::Bool` — not `ExitCase::Const(Int)` — or the bool exitswitch
    /// disagrees with its exitcases at rtype (`BoolRepr::convert_const(Int)`
    /// → "not a bool").  `lower_switch` consults this set to route the
    /// terminator through the `If` (`set_branch`) bool path; genuine
    /// two-variant enum tag switches (real `__discriminant` reads) are absent
    /// and keep the `Int`-exitcase path.
    pub(in crate::front::mir) niche_disc_vars: std::collections::HashSet<u64>,
    /// Source GC pointer of each `cast_ptr_to_int` / `p as usize` result.
    /// A following `cast_int_to_ptr` is `jtransform.py rewrite_op_cast_opaque_ptr`.
    pub(in crate::front::mir) cast_ptr_to_int_src: std::collections::HashMap<Variable, Variable>,
    /// `r_uint` result of `p as usize` when `p` is already the signed address
    /// of a Raw struct (`history.getkind`). Not a GC pointer, so it must not
    /// join [`Self::cast_ptr_to_int_src`]: that map folds `as *mut T` back to
    /// the source. `GcRef` construction still retypes this address.
    pub(in crate::front::mir) raw_address_uint_src: std::collections::HashMap<Variable, Variable>,
    /// Transparent `GcRef(p as usize)` results. The wrapper's Rust field is
    /// `usize`, but `GcRef::NULL` is `ConstRefNull` (`Ptr(GCREF)`), so the
    /// constructed word has to stay that pointer. `rgc.cast_ptr_to_adr`
    /// keeps the address kind the same way.
    pub(in crate::front::mir) pointer_word_vars: std::collections::HashSet<Variable>,
    /// Reads of an erased `OwnerRootGuard` word. `get().0 as *mut T` names
    /// the wrapper's `usize` field, but the stored word is the reference
    /// `new` wrote. A `cast_int_to_ptr` of that read would take the integer
    /// bank, which does not hold the reference.
    pub(in crate::front::mir) guard_word_vars: std::collections::HashSet<Variable>,
    /// Arrays that received a [`Self::pointer_word_vars`] store. A later
    /// `getitem` is that same word, not the wrapper's inner integer.
    pub(in crate::front::mir) pointer_word_arrays: std::collections::HashSet<Variable>,
}

/// `PyGraph.__init__` (`flowspace/pygraph.py`): the startblock of a
/// declared function, one named inputarg per formal parameter, built from
/// the code object's locals alone. Returns the local-to-Variable table
/// with the parameters bound.
#[cfg(any(test, feature = "mir-frontend"))]
pub(in crate::front::mir) fn pygraph_initial_block(
    graph: &mut FunctionGraph,
    locals: &majit_charon_reader::ullbc::Locals,
    llbc: &Llbc,
    generics: Option<&serde_json::Value>,
    tombstoned_leaves: &std::collections::HashSet<String>,
    gc_mut_ref_params: &std::collections::HashSet<usize>,
    gc_struct_ids: &std::collections::HashSet<majit_ir::descr::StructId>,
    keep_zst_self: bool,
    raw_alloc_flavor_structs: &std::collections::HashSet<String>,
) -> Vec<Option<Variable>> {
    let n_locals = locals.locals.len();
    let mut local_var: Vec<Option<Variable>> = vec![None; n_locals];

    let arg_count = locals.arg_count as usize;
    let len_shadow = pair_len_shadows_of(locals, llbc, &[], &[], &[]);
    let n_slots = n_locals + len_shadow.iter().flatten().count();
    let ret_len_slot_local = ret_len_slot_index(locals, llbc, &len_shadow, n_slots);
    // Arguments become startblock inputargs in source order
    // (RPython parity: `flowcontext.py` `init_locals_stack` fills
    // `locals_w`; arguments are `model.py` `Block` inputargs).
    //
    // Each parameter is also emitted as a paired `OpKind::Input { name,
    // ty }` op into the startblock.  Downstream consumers
    // — `flowspace_adapter::derive_subject_inputcells`
    // (`translator/rtyper/flowspace_adapter.rs`),
    // `graph_non_void_arg_types` (`codewriter/call.rs`),
    // `type_state` (`codewriter/type_state.rs`) — locate
    // each inputarg's declared `ValueType` by scanning the leading
    // `OpKind::Input` ops with `op.result == &arg`.  Without the
    // Input op, `derive_subject_inputcells` fails-loud at
    // `flowspace_adapter.rs` for any MIR-built graph that
    // reaches the real-rtyper dual-gate.
    //
    // A pair-slice parameter is two inputargs, the item pointer and then
    // the length, the two words the fat reference arrives in.
    let mut startblock_args: Vec<Variable> = Vec::with_capacity(arg_count);
    let mut input_ops: Vec<SpaceOperation> = Vec::with_capacity(arg_count);
    for (i, (local, slot)) in locals
        .locals
        .iter()
        .zip(local_var.iter_mut())
        .enumerate()
        .take(arg_count + 1)
        .skip(1)
    {
        let name = local.name.clone().unwrap_or_else(|| format!("arg{i}"));
        let var = graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        // Register a stable name so canonical comparison can spot
        // arg-renames.  Names live on the value via `name_value_var`
        // (mirrors the `parse.rs` arg-binding path).
        graph.name_value_var(&var, name.clone());
        *slot = Some(var.clone());
        let ty = tyref_to_value_type_with(
            &local.ty,
            llbc,
            tombstoned_leaves,
            gc_struct_ids,
            raw_alloc_flavor_structs,
        );
        // A parameter with no runtime representation (`Arg<T>`'s
        // `PhantomData` marker is the case in point — rustc gives it no
        // ABI slot) has to read as `Void` here, matching upstream's
        // `getkind(lltype.Void)`: `NON_VOID_ARGS` (the caller's actual
        // arguments) and `FUNC.ARGS` filtered the same way (this
        // graph's declared parameters via `graph_non_void_arg_types`)
        // both filter by the identical is-not-Void test, so a param the
        // caller's own concretetype tracking already treats as
        // void-carrying must agree here or `getcalldescr` sees a
        // caller/callee arity mismatch and hard fails. Scoped to the
        // declared-parameter fallback only — not
        // `tyref_to_value_type` itself, which construction sites
        // (`AggregateKind` lowering et al.) also call, and which must
        // keep minting a real value for a zero-sized type until they
        // erase it the way rtyper does everywhere. Guarded on the
        // generic `Ref(None)` fallback so a real (non-zero-sized) Ref
        // param is never touched, and a fieldless enum — already
        // resolved to `Int` above regardless of its own zero-sized
        // layout — never reaches this arm.
        //
        // An empty class instance (`KwargsDictStrategy`) is still a
        // `SomeInstance` (`description.py` `MethodDesc.func_args`);
        // `tyref_input_class_root` names it, so the Input stays `Ref`
        // and `derive_subject_inputcells` seeds the ClassDef.
        // `Arg<T>` is a zero-sized marker (`tyref_is_bytecode_arg_marker`)
        // even when the instantiation has a class_root; it stays Void.
        //
        // A reference to a primitive is the same case in the other
        // bank. `lltype.Ptr.__new__` refuses a non-container pointee,
        // so `&i64` is not a GC ref: `Rvalue::Ref` aliases the scalar
        // and `history.py` `getkind` banks it as `int`. Recording
        // `Ref(None)` makes `FUNC.ARGS` disagree with that caller.
        // A raw pointer to a wider scalar is `int` (`getkind` of a raw
        // `Ptr`). A byte pointer stays `Ref`: pyre erases a GC
        // reference to `*mut u8`. A nested `Ref` of a primitive (`&&i64`,
        // `PartialOrd<&B> for &A`) is the same scalar. See
        // [`tyref_scalar_pointee_value_type`] /
        // [`tyref_nested_ref_to_scalar_value_type`].
        //
        // An `indirectcalltarget` trait impl keeps ZST `&self` as Ref
        // (RPython `DictStrategy` instance / `history.getkind` of a Gc
        // Ptr): rustc still passes the pointer, and RPython method
        // `indirect_call` always has self. See [`fundecl_keeps_zst_self`].
        let mut ty = input_param_value_type(
            ty,
            &local.ty,
            llbc,
            keep_zst_self,
            i == 1,
            tombstoned_leaves,
        );
        // Same membership as [`gc_mut_ref_param_locals`], including the
        // residual exclusion: a cell parameter is the `MutRef<T>` the
        // caller passes, not `T`. Seeding `T` hoists `value` onto `T`'s class.
        let cell_root = if gc_mut_ref_params.contains(&i) {
            tyref_gc_mut_ref_cell_root(
                &local.ty,
                llbc,
                tombstoned_leaves,
                gc_struct_ids,
                raw_alloc_flavor_structs,
            )
        } else {
            None
        };
        // `class_root` carries the param's named-ADT leaf so
        // `derive_subject_inputcells` can seed the receiver's
        // `ClassDef`; only `Ref`-typed params consume it there.  A
        // generic param (`&T` where `T: Trait`, incl. trait default
        // bodies' `&Self`) has no ADT leaf — carry the bound
        // trait's qualified path instead, which the adapter
        // resolves through the unique-impl map
        // (`trait_unique_impls`, keyed by qualified path).
        // RPython's GC transformer casts a GC helper's pointer *argument*
        // to `llmemory.GCREF` before the call —
        // `framework.py gct_gc_identityhash` does
        // `[v_ptr] = hop.spaceop.args; v_ptr = hop.genop("cast_opaque_ptr",
        // [v_ptr], resulttype=llmemory.GCREF)`.  Rust's `PyObjectRef`
        // parameter is the physical carrier for that opaque slot, not a
        // W_Root instance.  Preserve the GCREF input boundary explicitly so
        // StringRepr and InstanceRepr callers never meet in one
        // source-level FunctionDesc cell.
        //
        // Key it on the LAST parameter, never on index 1: both spellings
        // `gc_root_pin_path` admits take exactly one `PyObjectRef`, but the
        // `RootScope` method form puts `&self` first, so stamping index 1
        // there annotates the receiver and leaves the GC pointer untouched.
        let class_root = if let Some(root) = &cell_root {
            Some(root.clone())
        } else if i == arg_count && gc_root_pin_path(&graph.name) {
            Some("GCREF".to_string())
        } else {
            match &ty {
                ValueType::Ref(_) => {
                    tyref_ref_param_class_root(&local.ty, llbc, tombstoned_leaves, generics)
                        .or_else(|| {
                            // `Box` / `NonNull` of a flavor-raw ADT is
                            // `Ref(Some(root))` at the front; `tyref_ref_param_class_root`
                            // declines the alloc wrapper. Name the pointee so
                            // `derive_subject_inputcells` seeds `SomeInstance`.
                            // `Arc` / `Rc` point at the refcount header, not `T`
                            // (`deref_cast_root`).
                            raw_address_owner_root(
                                &local.ty,
                                llbc,
                                tombstoned_leaves,
                                gc_struct_ids,
                                raw_alloc_flavor_structs,
                            )
                        })
                }
                // A fieldless enum is `Int`-colored (`tyref_to_value_type`),
                // so it takes the non-`Ref` arm and would otherwise carry no
                // `class_root`.  Its variant-name metadata is a side table
                // keyed by the enum type (the RPython "names by int" model),
                // not a field on the value; carry the crate-stripped enum
                // path so the `Debug`-fmt collapse can recover the enum
                // identity from the value's origin (`debug_enum_disc_owner`)
                // now that no `__discriminant` field read remains to scavenge.
                // `derive_subject_inputcells` only consumes `class_root` on
                // the `Ref` arm, so the annotation seed stays a plain
                // `SomeInteger`.
                // A function pointer is `Int`-colored (`history.getkind` of
                // `Ptr(FuncType)`); carry the `fn(inputs) -> output` /
                // `Option<fn(inputs) -> output>` spelling so
                // `derive_subject_inputcells` seeds `SomePtr(FuncType)`
                // (`fn_ptr_somevalue_for_spelling`) instead of
                // `valuetype_to_someshell`'s `SomeInteger`.
                // A `Vec` of one-word items is `Int`-colored (its header
                // address); carry its spelling so
                // `derive_subject_inputcells` seeds the `SomeRustVec`.
                // `char` is the same Int bank (`getkind` of UniChar);
                // carry `"char"` so the adapter seeds `SomeUnicodeCodePoint`.
                _ => tyref_char_class_root(&local.ty, llbc)
                    .or_else(|| tyref_fn_ptr_input_class_root(&local.ty, llbc))
                    .or_else(|| {
                        tyref_rust_vec_item_kind(&local.ty, llbc)
                            .map(|_| tyref_to_ast_string(&local.ty, llbc))
                    })
                    .or_else(|| tyref_fieldless_enum_class_root(&local.ty, llbc))
                    .or_else(|| {
                        if tyref_is_borrowed_fieldless_enum_free(&local.ty, llbc) {
                            None
                        } else {
                            raw_address_owner_root(
                                &local.ty,
                                llbc,
                                tombstoned_leaves,
                                gc_struct_ids,
                                raw_alloc_flavor_structs,
                            )
                        }
                    }),
            }
        };
        if let Some(root) = cell_root {
            ty = ValueType::Ref(Some(root));
        }
        let len_slot = len_shadow[i];
        let (ty, class_root) = match len_slot {
            Some(_) if tyref_is_u8_window(&local.ty, llbc) => {
                // Window data word is CCHARP (`rffi.py` `CCHARP`). The
                // register bank stays Unsigned so `getcalldescr` sees
                // Int-kind; `class_root` lets `derive_subject_inputcells`
                // seed `SomePtr` so `ann_direct_ptradd` (`s_p`) unions.
                (ValueType::Unsigned, Some("CCHARP".to_string()))
            }
            Some(_) => (ValueType::Unsigned, None),
            None if tyref_raw_u8_ptr_param_is_ccharp(&local.ty, llbc) => {
                // Every raw `*u8` parameter is CCHARP (`rffi.py` `CCHARP`).
                // Same bank as a window data word (`getkind` of a raw Ptr).
                (ValueType::Unsigned, Some("CCHARP".to_string()))
            }
            None => (ty, class_root),
        };
        input_ops.push(SpaceOperation {
            result: Some(var.clone()),
            kind: OpKind::Input {
                name: name.clone(),
                ty,
                class_root,
            },
        });
        startblock_args.push(var);
        if len_slot.is_some() {
            let len_var = graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
            let len_name = format!("{name}_len");
            graph.name_value_var(&len_var, len_name.clone());
            input_ops.push(SpaceOperation {
                result: Some(len_var.clone()),
                kind: OpKind::Input {
                    name: len_name,
                    ty: ValueType::Unsigned,
                    class_root: None,
                },
            });
            startblock_args.push(len_var);
        }
    }
    if ret_len_slot_local.is_some() {
        let var = graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
        let name = "ret_len_slot".to_string();
        graph.name_value_var(&var, name.clone());
        input_ops.push(SpaceOperation {
            result: Some(var.clone()),
            kind: OpKind::Input {
                name,
                ty: ValueType::Unsigned,
                class_root: None,
            },
        });
        startblock_args.push(var);
    }
    // Startblock gets the args as its inputargs. The startblock is
    // BlockId(0), already created by `FunctionGraph::new`.
    for var in &startblock_args {
        graph.push_inputarg_var(graph.startblock, var.clone());
    }
    // Push the paired `OpKind::Input` ops into the startblock so
    // `derive_subject_inputcells` can project each inputarg's
    // declared ValueType to a SomeValue shell.
    graph
        .block_mut(graph.startblock)
        .operations
        .extend(input_ops);
    local_var
}

/// A merge phi inherits `concretetype` only when both predecessors agree
/// (`unionof` of equal lowleveltypes). Disagreement leaves it unset.
fn copy_agreed_concretetypes(prev: &FrameState, ex: &FrameState, merged: &FrameState) {
    let n = merged
        .entries
        .len()
        .min(prev.entries.len())
        .min(ex.entries.len());
    for i in 0..n {
        let (Some(dst), Some(a), Some(b)) = (&merged.entries[i], &prev.entries[i], &ex.entries[i])
        else {
            continue;
        };
        if dst.id() == a.id() || dst.id() == b.id() {
            continue;
        }
        let (Some(ct_a), Some(ct_b)) = (a.concretetype(), b.concretetype()) else {
            continue;
        };
        if ct_a == ct_b {
            dst.set_concretetype(Some(ct_a));
        }
    }
}

impl<'a> Lowering<'a> {
    pub(in crate::front::mir) fn new(
        llbc: &'a Llbc,
        name: String,
        body: &'a Unstructured,
        static_addrs: crate::HostStaticAddrs<'a>,
        jitdriver_receiver_roots: &'a [String],
        generics: Option<&serde_json::Value>,
        dont_look_inside: &'a std::collections::HashSet<String>,
        tombstoned_leaves: &'a std::collections::HashSet<String>,
        gc_struct_ids: &'a std::collections::HashSet<majit_ir::descr::StructId>,
        immutable_fields: &'a ImmutableFieldsTable,
        raw_alloc_flavor_structs: &std::collections::HashSet<String>,
        accum: &AccumulatorFacts,
        root_stack: &RootStackAnalyzer<'_>,
        self_dont_look_inside: bool,
        keep_zst_self: bool,
    ) -> Result<Self, LowerError> {
        let gc_mut_ref_params = gc_mut_ref_param_locals(
            &body.locals,
            llbc,
            tombstoned_leaves,
            gc_struct_ids,
            self_dont_look_inside,
            raw_alloc_flavor_structs,
        );
        let n_locals = body.locals.locals.len();
        let (len_shadow, str_view_locals, list_view_locals, list_view_fixed) =
            pair_len_shadows(body, llbc, &name, accum);
        let mut graph = FunctionGraph::new(name);
        let raw_array = raw_array_roles(body, llbc);
        let slice_next_iter = slice_next_iters(body, llbc);
        let item_addr = item_addr_locals(body, llbc, &slice_next_iter);
        let owner_start = n_locals + len_shadow.iter().flatten().count();
        let owner_shadow = pair_owner_shadows_of(&len_shadow, owner_start);
        let n_slots = owner_start + owner_shadow.iter().flatten().count();
        // A body returning a pair slice or a window takes the address of
        // the caller's length slot as a trailing parameter. A STR-view
        // local 0 has no length shadow and still needs the slot.
        let ret_len_slot_local = ret_len_slot_index(&body.locals, llbc, &len_shadow, n_slots);
        let n_slots = n_slots + usize::from(ret_len_slot_local.is_some());
        let mut local_var: Vec<Option<LocalValue>> = vec![None; n_slots];

        let arg_count = body.locals.arg_count as usize;
        // Arguments become startblock inputargs in source order
        // (RPython parity: `flowcontext.py` `init_locals_stack` fills
        // `locals_w`; arguments are `model.py` `Block` inputargs).
        //
        // Each parameter is also emitted as a paired `OpKind::Input { name,
        // ty }` op into the startblock.  Downstream consumers
        // — `flowspace_adapter::derive_subject_inputcells`
        // (`translator/rtyper/flowspace_adapter.rs`),
        // `graph_non_void_arg_types` (`codewriter/call.rs`),
        // `type_state` (`codewriter/type_state.rs`) — locate
        // each inputarg's declared `ValueType` by scanning the leading
        // `OpKind::Input` ops with `op.result == &arg`.  Without the
        // Input op, `derive_subject_inputcells` fails-loud at
        // `flowspace_adapter.rs` for any MIR-built graph that
        // reaches the real-rtyper dual-gate.
        //
        // A pair-slice parameter is two inputargs, the item pointer and then
        // the length, the two words the fat reference arrives in.
        let mut startblock_args: Vec<Variable> = Vec::with_capacity(arg_count);
        let mut input_ops: Vec<SpaceOperation> = Vec::with_capacity(arg_count);
        let mut len_params: Vec<(usize, Variable)> = Vec::new();
        for (i, (local, slot)) in body
            .locals
            .locals
            .iter()
            .zip(local_var.iter_mut())
            .enumerate()
            .take(arg_count + 1)
            .skip(1)
        {
            let name = local.name.clone().unwrap_or_else(|| format!("arg{i}"));
            let var = graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
            // `_handle_list_call` reads `args[0].concretetype`
            // (`jtransform.py` `LIST = deref(args[0].concretetype)`).
            // A one-word `Vec` parameter is `RustVecRepr.lowleveltype`.
            stamp_rust_vec_header(&var, &local.ty, llbc);
            // Every raw `*u8` parameter is CCHARP (`rffi.py` `CCHARP`).
            // Live-in inputargs stamp the same way; `FrameState.copy` inherits it.
            stamp_raw_u8_ccharp(&var, &local.ty, llbc);
            // Register a stable name so canonical comparison can spot
            // arg-renames.  Names live on the value via `name_value_var`
            // (mirrors the `parse.rs` arg-binding path).
            graph.name_value_var(&var, name.clone());
            *slot = Some(LocalValue::One(var.clone()));
            if str_view_locals.get(i).copied().unwrap_or(false) {
                stamp_as_rstr_str(&var);
            }
            let ty = tyref_to_value_type_with(
                &local.ty,
                llbc,
                tombstoned_leaves,
                gc_struct_ids,
                raw_alloc_flavor_structs,
            );
            // A parameter with no runtime representation (`Arg<T>`'s
            // `PhantomData` marker is the case in point — rustc gives it no
            // ABI slot) has to read as `Void` here, matching upstream's
            // `getkind(lltype.Void)`: `NON_VOID_ARGS` (the caller's actual
            // arguments) and `FUNC.ARGS` filtered the same way (this
            // graph's declared parameters via `graph_non_void_arg_types`)
            // both filter by the identical is-not-Void test, so a param the
            // caller's own concretetype tracking already treats as
            // void-carrying must agree here or `getcalldescr` sees a
            // caller/callee arity mismatch and hard fails. Scoped to the
            // declared-parameter fallback only — not
            // `tyref_to_value_type` itself, which construction sites
            // (`AggregateKind` lowering et al.) also call, and which must
            // keep minting a real value for a zero-sized type until they
            // erase it the way rtyper does everywhere. Guarded on the
            // generic `Ref(None)` fallback so a real (non-zero-sized) Ref
            // param is never touched, and a fieldless enum — already
            // resolved to `Int` above regardless of its own zero-sized
            // layout — never reaches this arm.
            // An empty class instance (`KwargsDictStrategy`) is still a
            // `SomeInstance` (`description.py` `MethodDesc.func_args`);
            // `tyref_input_class_root` names it, so the Input stays `Ref`.
            // `Arg<T>` stays Void (`tyref_is_bytecode_arg_marker`).
            // Same banks as [`pygraph_initial_block`]: a reference to a
            // primitive is the scalar word (`history.py` `getkind` of
            // `&i64` is `int`), and a raw pointer to a wider scalar is
            // `int`. A byte pointer stays `Ref`.
            //
            // An `indirectcalltarget` trait impl keeps ZST `&self` as Ref
            // (the same kind as a GC impl in the family). See
            // [`fundecl_keeps_zst_self`].
            let mut ty = input_param_value_type(
                ty,
                &local.ty,
                llbc,
                keep_zst_self,
                i == 1,
                tombstoned_leaves,
            );
            // `class_root` carries the param's named-ADT leaf so
            // `derive_subject_inputcells` can seed the receiver's
            // `ClassDef`; only `Ref`-typed params consume it there.  A
            // generic param (`&T` where `T: Trait`, incl. trait default
            // bodies' `&Self`) has no ADT leaf — carry the bound
            // trait's qualified path instead, which the adapter
            // resolves through the unique-impl map
            // (`trait_unique_impls`, keyed by qualified path).
            // RPython's GC transformer casts a GC helper's pointer *argument*
            // to `llmemory.GCREF` before the call — `gct_gc_identityhash`
            // (`framework.py`) does
            // `[v_ptr] = hop.spaceop.args; v_ptr = hop.genop("cast_opaque_ptr",
            // [v_ptr], resulttype=llmemory.GCREF)`.  Rust's `PyObjectRef`
            // parameter is the physical carrier for that opaque slot, not a
            // W_Root instance.  Preserve the GCREF input boundary explicitly so
            // StringRepr and InstanceRepr callers never meet in one
            // source-level FunctionDesc cell.
            //
            // Key it on the LAST parameter, never on index 1: both spellings
            // `gc_root_pin_path` admits take exactly one `PyObjectRef`, but the
            // `RootScope` method form puts `&self` first, so stamping index 1
            // there annotates the receiver and leaves the GC pointer untouched.
            // Same membership as [`gc_mut_ref_param_locals`]: a cell
            // parameter is the `MutRef<T>` the caller passes, not `T`.
            let cell_root = if gc_mut_ref_params.contains(&i) {
                tyref_gc_mut_ref_cell_root(
                    &local.ty,
                    llbc,
                    tombstoned_leaves,
                    gc_struct_ids,
                    raw_alloc_flavor_structs,
                )
            } else {
                None
            };
            let class_root = if let Some(root) = &cell_root {
                Some(root.clone())
            } else if i == arg_count && gc_root_pin_path(&graph.name) {
                Some("GCREF".to_string())
            } else {
                match &ty {
                    ValueType::Ref(_) => {
                        tyref_ref_param_class_root(&local.ty, llbc, tombstoned_leaves, generics)
                    }
                    // A fieldless enum is `Int`-colored (`tyref_to_value_type`),
                    // so it takes the non-`Ref` arm and would otherwise carry no
                    // `class_root`.  Its variant-name metadata is a side table
                    // keyed by the enum type (the RPython "names by int" model),
                    // not a field on the value; carry the crate-stripped enum
                    // path so the `Debug`-fmt collapse can recover the enum
                    // identity from the value's origin (`debug_enum_disc_owner`)
                    // now that no `__discriminant` field read remains to scavenge.
                    // `derive_subject_inputcells` only consumes `class_root` on
                    // the `Ref` arm, so the annotation seed stays a plain
                    // `SomeInteger`.
                    // A function pointer is `Int`-colored (`history.getkind` of
                    // `Ptr(FuncType)`); carry the `fn(inputs) -> output` /
                    // `Option<fn(inputs) -> output>` spelling so
                    // `derive_subject_inputcells` seeds `SomePtr(FuncType)`
                    // (`fn_ptr_somevalue_for_spelling`) instead of
                    // `valuetype_to_someshell`'s `SomeInteger`.
                    // A `Vec` of one-word items is `Int`-colored (its header
                    // address); carry its spelling so
                    // `derive_subject_inputcells` seeds the `SomeRustVec`.
                    // `char` is the same Int bank (`getkind` of UniChar);
                    // carry `"char"` so the adapter seeds `SomeUnicodeCodePoint`.
                    _ => tyref_char_class_root(&local.ty, llbc)
                        .or_else(|| tyref_fn_ptr_input_class_root(&local.ty, llbc))
                        .or_else(|| {
                            tyref_rust_vec_item_kind(&local.ty, llbc)
                                .map(|_| tyref_to_ast_string(&local.ty, llbc))
                        })
                        .or_else(|| tyref_fieldless_enum_class_root(&local.ty, llbc))
                        .or_else(|| {
                            if tyref_is_borrowed_fieldless_enum_free(&local.ty, llbc) {
                                None
                            } else {
                                raw_address_owner_root(
                                    &local.ty,
                                    llbc,
                                    tombstoned_leaves,
                                    gc_struct_ids,
                                    raw_alloc_flavor_structs,
                                )
                            }
                        }),
                }
            };
            if let Some(root) = cell_root {
                ty = ValueType::Ref(Some(root));
            }
            let len_slot = len_shadow[i];
            let (ty, class_root) = match len_slot {
                Some(_) if tyref_is_u8_window(&local.ty, llbc) => {
                    // Window data word is CCHARP (`rffi.py` `CCHARP`). The
                    // register bank stays Unsigned so `getcalldescr` sees
                    // Int-kind; `class_root` lets `derive_subject_inputcells`
                    // seed `SomePtr` so `ann_direct_ptradd` (`s_p`) unions.
                    (ValueType::Unsigned, Some("CCHARP".to_string()))
                }
                Some(_) => (ValueType::Unsigned, None),
                None if str_view_locals.get(i).copied().unwrap_or(false)
                    && tyref_is_u8_window(&local.ty, llbc) =>
                {
                    (ValueType::Str, None)
                }
                None if tyref_raw_u8_ptr_param_is_ccharp(&local.ty, llbc) => {
                    // Every raw `*u8` parameter is CCHARP (`rffi.py` `CCHARP`).
                    // Same bank as a window data word (`getkind` of a raw Ptr).
                    (ValueType::Unsigned, Some("CCHARP".to_string()))
                }
                None => (ty, class_root),
            };
            input_ops.push(SpaceOperation {
                result: Some(var.clone()),
                kind: OpKind::Input {
                    name: name.clone(),
                    ty,
                    class_root,
                },
            });
            startblock_args.push(var);
            if let Some(len_slot) = len_slot {
                let len_var = graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                let len_name = format!("{name}_len");
                graph.name_value_var(&len_var, len_name.clone());
                input_ops.push(SpaceOperation {
                    result: Some(len_var.clone()),
                    kind: OpKind::Input {
                        name: len_name,
                        ty: ValueType::Unsigned,
                        class_root: None,
                    },
                });
                startblock_args.push(len_var.clone());
                len_params.push((len_slot, len_var));
            }
        }
        if let Some(ret_slot) = ret_len_slot_local {
            let var = graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
            let name = "ret_len_slot".to_string();
            graph.name_value_var(&var, name.clone());
            input_ops.push(SpaceOperation {
                result: Some(var.clone()),
                kind: OpKind::Input {
                    name,
                    ty: ValueType::Unsigned,
                    class_root: None,
                },
            });
            startblock_args.push(var.clone());
            len_params.push((ret_slot, var));
        }
        for (len_slot, len_var) in len_params {
            local_var[len_slot] = Some(LocalValue::One(len_var));
        }
        // Startblock gets the args as its inputargs. The startblock is
        // BlockId(0), already created by `FunctionGraph::new`.
        for var in &startblock_args {
            graph.push_inputarg_var(graph.startblock, var.clone());
        }
        // Push the paired `OpKind::Input` ops into the startblock so
        // `derive_subject_inputcells` can project each inputarg's
        // declared ValueType to a SomeValue shell.
        graph
            .block_mut(graph.startblock)
            .operations
            .extend(input_ops);

        // `FlowContext.build_flow` records a block only when `pendingblocks`
        // reaches it. Edges `successor_blocks` does not emit — every
        // `on_unwind`, and a `SwitchInt` panic default whose arms are
        // outside the niche shape — never enqueue a block, so those MIR
        // blocks get no `BlockId`, no per-block row, and no dead stub. A
        // niche-shaped `SwitchInt` keeps that default
        // (`lower_niche_option_switch`).
        let forward_reached = forward_reachable_mask(llbc, body);
        let lowered = lowered_reachable_mask(llbc, body);
        let block_id = MirBlockMap::from_lowered(&mut graph, &lowered);
        let cell_plan = collect_cell_scalar_locals(llbc, body, arg_count, &forward_reached);
        let mut extra_live = compute_index_write_extra_live(body, llbc, gc_struct_ids);
        let interior_live = compute_interior_field_extra_live(body, llbc, tombstoned_leaves);
        if extra_live.len() < interior_live.len() {
            extra_live.resize(interior_live.len(), Vec::new());
        }
        for (bb_idx, locals) in interior_live.into_iter().enumerate() {
            if bb_idx < extra_live.len() {
                extra_live[bb_idx].extend(locals);
            }
        }
        let glue_call_drops = glue_call_drop_blocks(body, llbc);
        // A root-bracket guard's `Drop` lowers to a call that takes the guard
        // by reference, so the dropping block reads it.  Every other drop is
        // pure control flow as far as liveness is concerned; without this the
        // guard reaches no drop block's inputargs and the close has no place
        // to name.  `glue_call_drops` is the same fact for the `drop_in_place`
        // spelling #1689 already keeps live.
        let root_scope_moved_locals = MovedOutLocals::new(body, llbc);
        let root_bracket = analyze_root_brackets(body, llbc, &root_scope_moved_locals, root_stack);
        if extra_live.len() < body.body.len() {
            extra_live.resize(body.body.len(), Vec::new());
        }
        for (bb_idx, local) in root_scope_drop_sites(body, llbc, &root_scope_moved_locals) {
            // An erased bracket emits no close, so its guard is not read at
            // the drop and must not be kept live for one.
            if root_bracket.is_erased_scope(local) {
                continue;
            }
            extra_live[bb_idx].push(local);
        }
        // A `get` this pass answers with the pinned value is a use of that
        // value the MIR never spells, the same synthetic-use shape
        // `compute_index_write_extra_live` handles for a deferred array write.
        for (bb_idx, value_local) in &root_bracket.get_sites {
            extra_live[*bb_idx].push(*value_local);
        }
        for (bb_idx, slice_local, _) in root_bracket
            .slice_gets
            .iter()
            .chain(root_bracket.slice_copies.iter())
        {
            extra_live[*bb_idx].push(*slice_local);
        }
        // The STR chars view aliases its slice to `base`, which the
        // `from_raw_parts` block does not spell. Keep `base` live there so
        // the alias names the block's own input, the same synthetic-use
        // shape as a deferred array write.
        for (bb_idx, base) in str_chars_view_extra_live(body, llbc) {
            if bb_idx < extra_live.len() {
                extra_live[bb_idx].push(base);
            }
        }
        // `StorageDead` of a window local is the last use of its GC owner
        // (`objectmodel.keepalive_until_here`).
        let owner_dead = window_owner_storage_dead_extra_live(body, &owner_shadow);
        if extra_live.len() < owner_dead.len() {
            extra_live.resize(owner_dead.len(), Vec::new());
        }
        for (bb_idx, locals) in owner_dead.into_iter().enumerate() {
            extra_live[bb_idx].extend(locals);
        }
        let cell_extra = cell_scope_extra_live(llbc, body, &cell_plan, &forward_reached);
        if extra_live.len() < cell_extra.len() {
            extra_live.resize(cell_extra.len(), Vec::new());
        }
        for (bb_idx, locals) in cell_extra.into_iter().enumerate() {
            extra_live[bb_idx].extend(locals);
        }
        let mut block_live_in =
            compute_mir_liveness(llbc, body, &extra_live, &glue_call_drops, &cell_plan.locals);
        // The erased guard, its borrows, its `base()` results, the
        // `base + k` and `shadow_stack_len` indices, the `pin_roots`
        // argument temporaries and the closure-capture temporaries bind no
        // Variable, so no block may ask a predecessor to pass one. The
        // pinned values stay live: `get_sites` already pushed them.
        if root_bracket.enabled {
            for live in &mut block_live_in {
                for scope in root_bracket.scopes.iter() {
                    live.remove(scope);
                }
                for alias in root_bracket.aliases.keys() {
                    live.remove(*alias);
                }
                for base in root_bracket.base_results.keys() {
                    live.remove(*base);
                }
                for temp in root_bracket.slot_temps.keys() {
                    live.remove(*temp);
                }
                for temp in root_bracket.pin_temps.keys() {
                    live.remove(*temp);
                }
                for temp in root_bracket.capture_temps.keys() {
                    live.remove(*temp);
                }
            }
        }
        // A pair slice's length is live wherever its pointer is, and the
        // return length slot everywhere.
        let raw_array_storage: Vec<usize> = raw_array
            .iter()
            .enumerate()
            .filter_map(|(local, role)| {
                matches!(role, Some(RawArrayRole::Storage { .. })).then_some(local)
            })
            .collect();
        for live in &mut block_live_in {
            let lens: Vec<usize> = live
                .iter()
                .filter_map(|local| len_shadow.get(local).copied().flatten())
                .collect();
            live.extend(lens);
            live.extend(ret_len_slot_local);
            live.extend(raw_array_storage.iter().copied());
        }
        // Owner slots are not startblock ABI params (`(data, len)` only).
        for (mir_bb, live) in block_live_in.iter_mut().enumerate() {
            if mir_bb == 0 {
                continue;
            }
            let owners: Vec<usize> = live
                .iter()
                .filter_map(|local| owner_shadow.get(local).copied().flatten())
                .collect();
            live.extend(owners);
        }
        // Panic-abort blocks stay in the fixpoint above so their uses still
        // flow into the switch that targets them. Only the lowered closure
        // keeps a row.
        let block_live_in: Vec<bit_set::BitSet> = block_live_in
            .into_iter()
            .enumerate()
            .filter(|(mir_bb, _)| lowered.get(*mir_bb).copied().unwrap_or(false))
            .map(|(_, live)| live)
            .collect();
        let nrows = block_id.nrows();
        let mut block_entry_local_var = vec![
            PackedLocalRow {
                len: n_slots,
                bound: Vec::new()
            };
            nrows
        ];
        let block_entry_positional_aggregate_locals = vec![std::collections::HashMap::new(); nrows];
        // A `[T; N]` array borrowed as a slice lives in a raw item buffer
        // for the whole body, the frame slot a native frame gives it: the
        // buffer is allocated on entry, reused by every construction of the
        // array and freed at each return.
        for (local, role) in raw_array.iter().enumerate() {
            let Some(RawArrayRole::Storage { kind, len }) = role else {
                continue;
            };
            let length = graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
            let items = graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
            graph.block_mut(graph.startblock).operations.extend([
                SpaceOperation {
                    result: Some(length.clone()),
                    kind: OpKind::ConstUInt(*len),
                },
                SpaceOperation {
                    result: Some(items.clone()),
                    kind: OpKind::Call {
                        target: CallTarget::FunctionPath {
                            segments: majit_ir::rvec::slice_helper_path(
                                majit_ir::rvec::SliceOp::BufferNew,
                                *kind,
                            )
                            .split("::")
                            .map(str::to_string)
                            .collect(),
                            fun_decl_id: None,
                            specialization: crate::parse::GraphSpecialization::Regular,
                        },
                        args: crate::model::call_args(vec![length]),
                        result_ty: ValueType::Unsigned,
                    },
                },
            ]);
            local_var[local] = Some(LocalValue::One(items));
        }
        for local in cell_plan.locals.iter() {
            if cell_plan.has_storage_live.contains(local) {
                continue;
            }
            let Some((_, itemsize, _)) = cell_plan.descr.get(local).and_then(|d| d.as_ref()) else {
                continue;
            };
            let start = graph.startblock;
            let ptr = emit_raw_cell_malloc(&mut graph, start, *itemsize);
            local_var[local] = Some(LocalValue::One(ptr));
        }
        if !block_entry_local_var.is_empty() {
            block_entry_local_var[0] = PackedLocalRow::pack(&link_words_of(&local_var));
        }
        for (mir_bb, bid) in block_id.iter_lowered() {
            if mir_bb == 0 {
                continue;
            }
            let row = block_id.row(mir_bb);
            for local_idx in 0..n_slots {
                if !block_live_in
                    .get(row)
                    .is_some_and(|s| s.contains(local_idx))
                {
                    continue;
                }
                let var = graph.alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
                // Loop-header inputargs are these variables
                // (`lower_framestate` Pass 0). `list.getitem` /
                // `list.setitem` read `args[0].concretetype`.
                if let Some(local) = body.locals.locals.get(local_idx) {
                    stamp_rust_vec_header(&var, &local.ty, llbc);
                    stamp_raw_u8_ccharp(&var, &local.ty, llbc);
                }
                if owner_shadow.iter().any(|slot| *slot == Some(local_idx)) {
                    crate::model::FunctionGraph::set_concretetype_of_inline(
                        &var,
                        crate::model::ConcreteType::GcRef,
                    );
                }
                graph.push_inputarg_var(bid, var.clone());
                block_entry_local_var[row].set(local_idx, var);
            }
        }
        Ok(Self {
            graph,
            llbc,
            struct_fields: None,
            raw_alloc_flavor_structs: raw_alloc_flavor_structs.clone(),
            dont_look_inside,
            tombstoned_leaves,
            gc_struct_ids,
            immutable_fields,
            body,
            static_addrs,
            jitdriver_receiver_roots,
            arg_count,
            local_var,
            len_shadow,
            owner_shadow,
            ret_len_slot_local,
            raw_array,
            item_addr,
            slice_next_iter,
            emit_block: block_id.clone(),
            block_id,
            forward_reached,
            call_dest_locals: std::cell::OnceCell::new(),
            projected_store_roots: std::cell::OnceCell::new(),
            block_live_in,
            block_entry_local_var,
            input_copied_from: std::collections::HashMap::new(),
            object_vec_headers: std::collections::HashSet::new(),
            block_entry_positional_aggregate_locals,
            block_positional_seen: vec![bit_set::BitSet::with_capacity(n_locals); nrows],
            block_positional_conflict: vec![bit_set::BitSet::with_capacity(n_locals); nrows],
            block_entry_string_byte_view_locals: {
                let mut rows = vec![bit_set::BitSet::with_capacity(n_locals); nrows];
                for (i, marked) in str_view_locals.iter().enumerate() {
                    if *marked {
                        for row in &mut rows {
                            row.insert(i);
                        }
                    }
                }
                rows
            },
            block_byte_view_seen: vec![bit_set::BitSet::with_capacity(n_locals); nrows],
            block_byte_view_conflict: vec![bit_set::BitSet::with_capacity(n_locals); nrows],
            block_entry_list_view_locals: {
                let mut rows = vec![bit_set::BitSet::with_capacity(n_locals); nrows];
                for (i, marked) in list_view_locals.iter().enumerate() {
                    if *marked {
                        for row in &mut rows {
                            row.insert(i);
                        }
                    }
                }
                rows
            },
            block_list_view_seen: vec![bit_set::BitSet::with_capacity(n_locals); nrows],
            block_list_view_conflict: vec![bit_set::BitSet::with_capacity(n_locals); nrows],
            positional_aggregate_locals: std::collections::HashMap::new(),
            binop_result_locals: compute_binop_result_locals(llbc, body),
            builder_mode: false,
            accum: accum.clone(),
            index_elem_alias: std::collections::HashMap::new(),
            interior_field_alias: std::collections::HashMap::new(),
            atomic_ref_place: std::collections::HashMap::new(),
            gc_mut_ref_params,
            scalar_address_locals: Vec::new(),
            cell_locals: cell_plan.locals,
            cell_descr: cell_plan.descr,
            cell_has_storage_live: cell_plan.has_storage_live,
            atomic_ordering_locals: std::collections::HashMap::new(),
            spec: None,
            spec_body: false,
            const_discriminant_locals: std::collections::HashMap::new(),
            const_struct_fields: vec![None; n_locals],
            multi_assigned_locals: compute_multi_assigned_locals(llbc, body),
            string_byte_view_locals: str_view_locals
                .iter()
                .enumerate()
                .filter_map(|(i, marked)| marked.then_some(i))
                .collect(),
            list_view_locals: list_view_locals
                .iter()
                .enumerate()
                .filter_map(|(i, marked)| marked.then_some(i))
                .collect(),
            list_view_fixed,
            fat_box_vars: Vec::new(),
            dyn_return_calls: Vec::new(),
            prebuilt_once_value_locals: Vec::new(),
            string_array_view_locals: Vec::new(),
            numeric_array_view_locals: Vec::new(),
            result_exc_call_results: Vec::new(),
            option_ok_or_else_try_sites: Vec::new(),
            carrier_combinator_sites: Vec::new(),
            result_map_err_sites: Vec::new(),
            result_expect_sites: Vec::new(),
            next_call_results: Vec::new(),
            map_collect_sites: Vec::new(),
            checked_arith_call_results: Vec::new(),
            checked_arith_ok_or_else_sites: Vec::new(),
            checked_arith_uint_sites: Vec::new(),
            tagged_pair_aggregate_sites: Vec::new(),
            from_size_align_sites: Vec::new(),
            from_size_align_expect_sites: Vec::new(),
            from_size_align_call_sites: Vec::new(),
            layout_accessor_sites: Vec::new(),
            option_try_sites: Vec::new(),
            bool_then_sites: Vec::new(),
            slice_first_sites: Vec::new(),
            slice_get_sites: Vec::new(),
            slice_is_ascii_sites: Vec::new(),
            saturating_mul_sites: Vec::new(),
            saturating_add_sites: Vec::new(),
            saturating_sub_sites: Vec::new(),
            range_iter_new_sites: Vec::new(),
            slice_index_rangefrom_sites: Vec::new(),
            slice_index_rangeto_sites: Vec::new(),
            slice_index_range_sites: Vec::new(),
            unwrap_sites: Vec::new(),
            expect_sites: Vec::new(),
            map_or_sites: Vec::new(),
            closure_select_sites: Vec::new(),
            disc_combinator_sites: Vec::new(),
            result_try_sites: Vec::new(),
            niche_disc_vars: std::collections::HashSet::new(),
            cast_ptr_to_int_src: std::collections::HashMap::new(),
            raw_address_uint_src: std::collections::HashMap::new(),
            pointer_word_vars: std::collections::HashSet::new(),
            guard_word_vars: std::collections::HashSet::new(),
            pointer_word_arrays: std::collections::HashSet::new(),
            root_scope_moved_locals,
            root_bracket,
            owner_root: analyze_owner_roots(body, llbc),
            owner_root_reroot: None,
            pending_vec_frees: Vec::new(),
        })
    }

    /// Switch this lowering to builder form so the ctor / append /
    /// terminal materialisation sites of a builder-mode accumulator
    /// ([`is_builder_mode_accumulator`]) emit the `StringBuilder` markers
    /// instead of `ll_strconcat`. The canonical frontend selects this before
    /// lowering whenever [`graph_has_builder_accumulator`] succeeds.
    /// Returns whether any accumulator was found. Per-local membership is
    /// [`accum`](Self::accum); this records only the mode flag.
    pub(in crate::front::mir) fn enable_builder_mode(&mut self) -> bool {
        self.builder_mode = true;
        // Same "any builder-mode accumulator?" question as the pre-check, over
        // the owned-builder candidates only (see [`owned_builder_dest_locals`]).
        self.accum.has_builder
    }

    pub(in crate::front::mir) fn accumulator_is_builder_mode(&self, c: usize) -> bool {
        match self.accum.builder_mode.get(c) {
            Some(flag) => *flag,
            None => {
                let cache = ScanCache::new(self.body.locals.locals.len());
                cache.precompute_fresh(self.body, self.llbc);
                is_builder_mode_accumulator(&cache, self.body, self.llbc, c)
            }
        }
    }

    pub(in crate::front::mir) fn accumulator_of_append_arg(&self, rt: usize) -> Option<usize> {
        match self.accum.append_accumulator.get(rt) {
            Some(hit) => *hit,
            None => append_accumulator_of_arg_temp(self.body, self.llbc, rt),
        }
    }

    pub(in crate::front::mir) fn piece_accumulator_of_append_arg(
        &self,
        rt: usize,
    ) -> Option<usize> {
        match self.accum.append_piece_accumulator.get(rt) {
            Some(hit) => *hit,
            None => append_piece_accumulator_of_arg_temp(self.body, self.llbc, rt),
        }
    }

    pub(in crate::front::mir) fn accumulator_of_identity_arg(&self, rt: usize) -> Option<usize> {
        match self.accum.identity_accumulator.get(rt) {
            Some(hit) => *hit,
            None => identity_accumulator_of_arg_temp(self.body, self.llbc, rt),
        }
    }

    /// Owned `String` / `Wtf8Buf` accumulator `mem::take` rebinds.
    /// Walks `atomic_ref_place` through a two-phase `&mut` the way
    /// [`append_accumulator_of_arg_temp`] walks append receivers, then
    /// falls back to a MIR scan of [`take_receiver_of_borrow`].
    pub(in crate::front::mir) fn string_accumulator_of_take_arg(
        &self,
        take_arg: Option<usize>,
    ) -> Option<usize> {
        let rt = take_arg?;
        if let Some(MemSlot::Place(place)) = self.mem_slot(Some(rt)) {
            let place = self.concrete_borrow_place(place);
            if let PlaceKind::Local(i) = place.kind {
                let i = i as usize;
                if tyref_is_owned_str_builder(&place.ty, self.llbc)
                    || self
                        .body
                        .locals
                        .locals
                        .get(i)
                        .is_some_and(|local| tyref_is_owned_str_builder(&local.ty, self.llbc))
                {
                    return Some(i);
                }
            }
        }
        let cache = ScanCache::new(self.body.locals.locals.len());
        cache.precompute_fresh(self.body, self.llbc);
        let forward_blocks = forward_reachable_mask(self.llbc, self.body);
        for candidate in owned_builder_dest_locals(self.body, self.llbc) {
            if !is_fresh_str_builder(&cache, self.body, self.llbc, candidate) {
                continue;
            }
            for (bb_idx, block) in self.body.body.iter().enumerate() {
                if !forward_blocks[bb_idx] {
                    continue;
                }
                for stmt in &block.statements {
                    let Ok(StmtKind::Assign(place, Rvalue::Ref { place: source, .. })) =
                        stmt.stmt_kind_ref()
                    else {
                        continue;
                    };
                    if matches!(source.kind, PlaceKind::Local(i) if i as usize == candidate)
                        && let PlaceKind::Local(direct_borrow) = place.kind
                        && take_receiver_of_borrow(self.body, self.llbc, direct_borrow as usize)
                            == Some(rt)
                    {
                        return Some(candidate);
                    }
                }
            }
        }
        None
    }

    pub(in crate::front::mir) fn lower(&mut self, order: BlockOrder) -> Result<(), LowerError> {
        // Each MIR basic block is a FlowGraph block.  Locals live across
        // a successor edge are explicit `Link.args` into the target
        // block's `inputargs`, mirroring FlowContext.mergeblock rather
        // than relying on a function-wide slot table.
        for mir_bb in self.block_processing_order(order) {
            self.lower_block(mir_bb)?;
        }
        Ok(())
    }

    /// Block processing order over the lowered closure.
    ///
    /// `Linear` is MIR index order. `ReversePostorder` is
    /// [`Self::model_rpo`]: the reverse-postorder of [`Self::model_succs`]
    /// from bb0. Blocks `pendingblocks` would never reach are absent.
    pub(in crate::front::mir) fn block_processing_order(&self, order: BlockOrder) -> Vec<usize> {
        if matches!(order, BlockOrder::Linear) {
            return self.block_id.iter_lowered().map(|(mir, _)| mir).collect();
        }
        self.model_rpo()
    }

    // Framestate-threaded lowering (acyclic GAP-B path)

    /// Snapshot the current `local_var` table as a [`FrameState`].  Only
    /// the locals projection is populated — MIR has no value stack /
    /// pending exception / block-stack, so those stay at the `Default`
    /// (empty) shape.  `entries` is indexed by MIR local index; every
    /// framestate produced in this lowering uses the same indexing, so
    /// [`FrameState::union`] / [`FrameState::getvariables`] /
    /// [`FrameState::getoutputargs`] — all purely positional over
    /// `entries` / `locals_w` — line up across predecessors and merge
    /// targets without consulting any external slot table.
    pub(in crate::front::mir) fn link_words(&self) -> Vec<Option<Variable>> {
        link_words_of(&self.local_var)
    }

    /// Rebuild pair and slice-iterator locals from the link words a
    /// [`FrameState`] or a block-entry row carries. The shadow slot holds
    /// the length, or the item pointer of a slice iterator.
    pub(in crate::front::mir) fn hydrate_words(
        &self,
        words: Vec<Option<Variable>>,
    ) -> Vec<Option<LocalValue>> {
        let mut out: Vec<Option<LocalValue>> = words
            .iter()
            .cloned()
            .map(|word| word.map(LocalValue::One))
            .collect();
        for (local, shadow) in self.len_shadow.iter().enumerate() {
            let Some(shadow) = *shadow else {
                continue;
            };
            let Some(LocalValue::One(first)) = out.get(local).cloned().flatten() else {
                continue;
            };
            let Some(LocalValue::One(second)) = out.get(shadow).cloned().flatten() else {
                continue;
            };
            let is_iter = self
                .body
                .locals
                .locals
                .get(local)
                .is_some_and(|loc| self.pair_iter_kind(&loc.ty).is_some());
            let owner = self
                .owner_shadow
                .get(local)
                .copied()
                .flatten()
                .and_then(|slot| match out.get(slot).cloned().flatten() {
                    Some(LocalValue::One(v)) => Some(v),
                    _ => None,
                });
            out[local] = Some(if is_iter {
                LocalValue::SliceIter {
                    index: first,
                    items: second,
                }
            } else {
                LocalValue::Pair(first, second, owner)
            });
        }
        for (local, is_addr) in self.item_addr.iter().enumerate() {
            if *is_addr && let Some(LocalValue::One(addr)) = out.get(local).cloned().flatten() {
                out[local] = Some(LocalValue::ItemAddr(addr));
            }
        }
        out
    }

    pub(in crate::front::mir) fn getstate(&self) -> FrameState {
        FrameState {
            entries: self.link_words(),
            ..Default::default()
        }
    }

    /// Restore `local_var` from a [`FrameState`] produced by
    /// [`Self::getstate`] or [`FrameState::union`].  Slots are sized to
    /// the fixed local count; a `None` slot marks a local undefined on at
    /// least one predecessor path (union None-kill), so a later read
    /// fails-loud through [`Self::resolve_place`] as "uninitialised
    /// local" — the same use-before-def signal the monotonic path emits.
    pub(in crate::front::mir) fn setstate(&mut self, fs: &FrameState) {
        let n = self.local_var.len();
        let words = (0..n)
            .map(|i| fs.entries.get(i).cloned().flatten())
            .collect();
        self.local_var = self.hydrate_words(words);
    }

    /// Record `FrameState.copy`'s fresh input → predecessor Variable.
    ///
    /// Pass 2 writes the same pairs as link args after the target body
    /// is lowered. A use lowered in the target, such as the Acquire load
    /// of a quasi-immutable field, reads the predecessor operation through
    /// this map.
    pub(in crate::front::mir) fn note_copied_block_inputs(
        &mut self,
        source: &FrameState,
        copied: &FrameState,
    ) {
        let n = source.entries.len().min(copied.entries.len());
        for slot in 0..n {
            let (Some(src), Some(dst)) = (&source.entries[slot], &copied.entries[slot]) else {
                continue;
            };
            if src.id() != dst.id() {
                self.input_copied_from.insert(dst.id(), src.id());
                // `Variable.copy` (`flowspace/model.py`) keeps `concretetype`.
                copy_concretetype(src, dst);
            }
            if self.var_is_object_vec_header(src) {
                self.object_vec_headers.insert(dst.id());
            }
        }
    }

    /// `header` is the address of a `Vec<PyObjectRef>`.
    pub(in crate::front::mir) fn note_object_vec_header(&mut self, header: &Variable) {
        self.object_vec_headers.insert(header.id());
    }

    /// Whether `var` is that header, or a block input copied from one.
    pub(in crate::front::mir) fn var_is_object_vec_header(&self, var: &Variable) -> bool {
        let mut id = var.id();
        // A chain longer than this, or a cycle, has no header within reach.
        for _ in 0..65 {
            if self.object_vec_headers.contains(&id) {
                return true;
            }
            match self.input_copied_from.get(&id).copied() {
                Some(next) => id = next,
                None => return false,
            }
        }
        false
    }

    /// Whether MIR `local` currently holds a [`Self::note_object_vec_header`]
    /// address: the dest of a whole-list mut view (`deref_mut`,
    /// `as_mut_slice`, `index_mut(RangeFull)`).
    pub(in crate::front::mir) fn local_is_object_vec_header(&self, local: usize) -> bool {
        self.local_var
            .get(local)
            .and_then(|slot| slot.as_ref())
            .and_then(|value| value.one().ok())
            .is_some_and(|var| self.var_is_object_vec_header(&var))
    }

    /// Successor MIR blocks along the edges [`Self::lower_terminator`]
    /// actually materialises in the model graph.  This EXCLUDES
    /// `on_unwind` (dropped by `lower_call` and the `Assert` / `Drop`
    /// arms): the reachability / acyclicity / RPO that drive framestate
    /// threading must follow only the edges that exist in the produced
    /// `FunctionGraph`, otherwise an orphan cleanup chain would appear as
    /// a live merge predecessor.
    pub(in crate::front::mir) fn model_succs(&self, mir_bb: usize) -> Vec<usize> {
        successor_blocks(self.llbc, self.body, mir_bb, false)
    }

    /// True when MIR block `bb`'s terminator is a panic/abort stub
    /// (`Abort` / `Panic` / `UnwindResume`).  rustc lowers the out-of-range
    /// `default` arm of an enum-discriminant `SwitchInt` to such a block
    /// — an unreachable UB stub with no flowgraph analogue.  Excluding it
    /// from the switch's successors keeps the orphan `set_raise`
    /// cleanup chain (with its undefined etype/evalue) out of the produced
    /// graph instead of leaving it as a live, undefined-operand block.
    pub(in crate::front::mir) fn switch_default_targets_panic_abort(&self, bb: u64) -> bool {
        terminator_is_panic_abort(self.llbc, self.body, bb)
    }

    /// Loop headers of the model-reachable component (from bb0 over
    /// [`Self::model_succs`]): the targets of back-edges.  Indexed by MIR
    /// block; `true` iff the block is the target of at least one retreating
    /// edge in the DFS — i.e. a successor still grey (on the DFS stack)
    /// when its predecessor reaches it.  Returns all-`false` for an
    /// acyclic body, so passing this to [`Self::lower_framestate`] reduces
    /// that path to the plain two-pass RPO walk.
    ///
    /// For a reducible body (every retreating edge's target dominates its
    /// source — the shape structured Rust control flow always produces)
    /// these are exactly the natural loop headers, independent of DFS
    /// order.  The framestate path uses them to pre-seed each loop
    /// header's entry state with its pre-bound live-in phis so the
    /// back-edge can thread into them; an irreducible body may mis-seed
    /// and produce an undefined-operand graph, which the pass-2 / final
    /// self-validation guards reject into the monotonic fallback.
    pub(in crate::front::mir) fn mir_model_loop_headers(&self) -> Vec<bool> {
        let n = self.body.body.len();
        let mut headers = vec![false; n];
        if n == 0 {
            return headers;
        }
        // 0 = white, 1 = grey (on stack), 2 = black (done).
        let mut state = vec![0u8; n];
        let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
        state[0] = 1;
        while let Some(&(node, idx)) = stack.last() {
            let succs = self.model_succs(node);
            if idx < succs.len() {
                stack.last_mut().unwrap().1 += 1;
                let nxt = succs[idx];
                match state[nxt] {
                    0 => {
                        state[nxt] = 1;
                        stack.push((nxt, 0));
                    }
                    1 => headers[nxt] = true, // a grey successor is a back-edge target
                    _ => {}
                }
            } else {
                state[node] = 2;
                stack.pop();
            }
        }
        headers
    }

    /// Reverse-postorder of the model-reachable component over
    /// [`Self::model_succs`].  Only blocks reachable from bb0 appear.
    /// In an acyclic graph RPO visits every predecessor of a block before
    /// the block itself, which is exactly the order the two-pass framestate
    /// threading relies on.
    pub(in crate::front::mir) fn model_rpo(&self) -> Vec<usize> {
        let n = self.body.body.len();
        if n == 0 {
            return vec![];
        }
        let mut state = vec![0u8; n];
        let mut postorder: Vec<usize> = Vec::with_capacity(n);
        let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
        state[0] = 1;
        while let Some(&(node, idx)) = stack.last() {
            let succs = self.model_succs(node);
            if idx < succs.len() {
                stack.last_mut().unwrap().1 += 1;
                let nxt = succs[idx];
                if state[nxt] == 0 {
                    state[nxt] = 1;
                    stack.push((nxt, 0));
                }
            } else {
                state[node] = 2;
                postorder.push(node);
                stack.pop();
            }
        }
        postorder.into_iter().rev().collect()
    }

    /// Stub `bb_id` as a dead bare-raise block: clear its body / exits /
    /// exitswitch, close it with a `set_raise`, and mark it `dead`.  Used
    /// for an `If`-arm a const-bool discriminant folded away
    /// (see [`Self::lower_framestate`]).  That arm is in the lowered
    /// closure — `model_succs` still lists both arms — so it has a graph
    /// block, and the fold is what makes the entry empty.  The real-path
    /// `function_graph_to_flowspace` prunes `dead` blocks
    /// (`remove_dead_blocks` parity), so the stub's orphan etype/evalue
    /// never reach the rtyper as undefined operands.  A block outside the
    /// closure is never recorded, so it needs no stub.
    pub(in crate::front::mir) fn stub_dead_block(&mut self, bb_id: BlockId) {
        let blk = self.graph.block_mut(bb_id);
        blk.operations.clear();
        blk.exits.clear();
        blk.exitswitch = None;
        self.graph.set_raise(bb_id, "mir-dead");
        self.graph.block_mut(bb_id).dead = true;
    }

    /// Lower the body threading locals as block inputargs / phis via
    /// per-block [`FrameState`]s, instead of the function-wide monotonic
    /// `local_var` table.  Handles both acyclic and cyclic bodies; this
    /// drains the GAP-B "undefined operand" census skips where a
    /// reassigned local reaches a merge with path-dependent values the
    /// monotonic single-slot scheme cannot represent, and puts cyclic
    /// bodies on the same orthodox flowspace shape as acyclic ones.
    ///
    /// `loop_headers` (from [`Self::mir_model_loop_headers`]) marks the
    /// back-edge targets.  For an acyclic body it is all-`false` and this
    /// reduces exactly to the two-pass RPO walk.
    ///
    /// Two passes over the model-reachable blocks in reverse-postorder:
    ///
    ///   Pass 0 (cyclic only) — pre-seed every loop header's entry
    ///   framestate with the live-in phis `new` pre-bound for it
    ///   (`block_entry_local_var[H]`).  The RPO walk visits a loop header
    ///   before its back-edge predecessor (latch), so without a seed the
    ///   header's entry would miss the back-edge's contribution; seeding
    ///   it with the live-in Variables makes the header's inputargs real
    ///   phis that both the forward predecessor(s) and the latch thread
    ///   into during pass 2.
    ///
    ///   Pass 1 — for each block, `setstate` to its accumulated entry
    ///   framestate, set its (non-startblock) inputargs to the entry
    ///   framestate's variables, lower its statements + terminator
    ///   (reusing the monotonic per-op lowering, which closes
    ///   return / raise correctly and emits placeholder empty-args
    ///   goto / branch links because every successor still has empty
    ///   inputargs at close time), snapshot the exit framestate, and
    ///   union that exit into each model successor's entry framestate —
    ///   except a successor that is a loop header, whose pre-seeded entry
    ///   is fixed (the predecessor still threads into it in pass 2).
    ///
    ///   Pass 2 — re-argument each goto / branch link from the
    ///   predecessor exit / target entry framestates via `getoutputargs`,
    ///   preserving the link's target, exitcase, and the block's
    ///   `exitswitch`.
    ///
    /// Blocks outside the lowered closure are never recorded
    /// (`FlowContext.build_flow` / `pendingblocks`), so there is no orphan
    /// cleanup chain to stub.  bb0 (the startblock) keeps its
    /// `OpKind::Input`-paired parameter inputargs untouched — the
    /// opcode-dispatch arm extractor depends on that shape — so a
    /// back-edge into bb0 (which would demand reseeding the parameter
    /// slots as phis) declines to the monotonic fallback.
    pub(in crate::front::mir) fn lower_framestate(
        &mut self,
        loop_headers: &[bool],
    ) -> Result<(), LowerError> {
        let n = self.body.body.len();
        if n == 0 {
            return Ok(());
        }
        // A back-edge into bb0 would force the parameter slots to become
        // phis, but bb0 must keep its `OpKind::Input`-paired parameter
        // inputargs (the opcode-dispatch arm extractor depends on that
        // shape).  Decline such bodies to the monotonic fallback.
        if loop_headers.first().copied().unwrap_or(false) {
            return Err(LowerError::Unsupported(
                "framestate: back-edge into startblock bb0 — declines to monotonic".to_string(),
            ));
        }
        let rpo = self.model_rpo();
        let returnblock = self.graph.returnblock;
        let exceptblock = self.graph.exceptblock;

        // BlockId → MIR bb inverse.  returnblock / exceptblock (and any
        // non-MIR block) map to `usize::MAX` — they are merge sinks the
        // accumulation skips.
        let mut block_to_mir = vec![usize::MAX; self.graph.blocks.len()];
        for (mir, bid) in self.block_id.iter_lowered() {
            block_to_mir[bid.0] = mir;
        }

        let nrows = self.block_id.nrows();
        let mut entry_state: Vec<Option<PackedFrameState>> = vec![None; nrows];
        let mut exit_state: Vec<Option<PackedFrameState>> = vec![None; nrows];
        // bb0 enters with the parameter bindings established in `new`.
        entry_state[0] = Some(PackedFrameState::pack(self.getstate()));

        // Pass 0 (cyclic only) — pre-seed each loop header's entry
        // framestate with the live-in phis `new` pre-bound for it
        // (`block_entry_local_var[H]`, a fresh `Variable` per live-in
        // local already pushed as a `block_id[H]` inputarg).  The RPO
        // walk reaches a loop header before its latch, so this seed — not
        // a forward-predecessor union — is what makes the header's
        // inputargs the loop phis; the forward predecessor(s) and the
        // latch both thread into them in pass 2 via `getoutputargs`.
        // bb0 is never reseeded (guarded above + it is not in the merge
        // set), so its parameter inputargs / `Input` ops stay intact.
        for (h, &is_header) in loop_headers.iter().enumerate().take(n) {
            if is_header && h != 0 && self.block_id.is_lowered(h) {
                let row = self.block_id.row(h);
                entry_state[row] = Some(PackedFrameState::pack(FrameState {
                    entries: self.block_entry_local_var[row].unpack(),
                    ..Default::default()
                }));
            }
        }

        // Pass 1 — RPO walk: setstate, inputargs, lower, snapshot, union.
        for &bb in &rpo {
            let row = self.block_id.row(bb);
            let st = match entry_state[row].as_ref().map(PackedFrameState::unpack) {
                Some(st) => st,
                None => {
                    // No live predecessor edge reached this block.  RPO
                    // over `model_succs` visits every model-predecessor
                    // before `bb`, and each unions its *lowered* exits
                    // into `bb`'s entry; a still-empty entry here means
                    // every model edge into `bb` was an `If`-arm that
                    // `lower_switch` folded away on a const-bool
                    // discriminant (a translation-time `const` gate such
                    // as `if WITHPREBUILTINT`), so `bb` is unreachable in
                    // the produced graph even though the raw `model_succs`
                    // closure — which reads the terminator before the fold —
                    // recorded a block for it.  Stub that block dead; the
                    // real-path adapter's reachability prune removes it, and
                    // threading dead state would only mint phis the legacy
                    // fallback cannot type.  `bb` is never bb0 (the
                    // startblock is seeded and visited first), so a bare
                    // raise stub is well-formed.
                    self.stub_dead_block(self.block_id[bb]);
                    continue;
                }
            };
            self.setstate(&st);
            let bb_id = self.block_id[bb];
            // The startblock keeps its parameter inputargs + Input ops;
            // it is never a merge target in an acyclic body.
            if bb != 0 {
                let inputargs = st.getvariables(&self.graph);
                self.graph.block_mut(bb_id).inputargs = inputargs;
                // Anchor the block body to the framestate locals.
                // `lower_block` reloads `self.local_var` from
                // `block_entry_local_var[bb]` (the construction-time
                // per-block-entry Variables), so without this the body
                // would bind locals to those construction identities
                // while the inputargs above (and the `getoutputargs`
                // link args in Pass 2) carry the union-minted framestate
                // identities.  The mismatch leaves the body's operand
                // Variables defined as neither an inputarg nor an op
                // result, so `perform_register_allocation` assigns them
                // no colour and `assembler::lookup_coloring` aborts.
                // `self.local_var` here is exactly `setstate(&st)`'s
                // positional projection of the framestate entries, whose
                // Variable cells are the same identities `getvariables`
                // threaded into `inputargs`.
                self.block_entry_local_var[row] = PackedLocalRow::pack(&self.link_words());
            }
            self.lower_block(bb)?;
            // A bounds fork or whole-enum move closes this MIR block's
            // head and leaves the real terminator on the tail / join.
            // Incoming edges still target `block_id`; successors are the
            // emit cursor's exits.
            let bb_id = self.emit_bb(bb);
            let mut ex = self.getstate();
            // Scrub phantom locals before threading.  A slot bound to a
            // Variable that is neither an inputarg nor an op result of this
            // block has no definition in the produced graph — e.g. a MIR
            // local the body never assigns, conservatively kept live by
            // `compute_mir_liveness` and prebound in `new`, that the
            // fully-inlined source leaves dead (the unread `MaybeUninit`
            // scratch a monomorphized `core::ptr::swap` carries).  The
            // monotonic path threads only `target_input_locals` (live-in)
            // via per-block prebind inputargs, so its copy is always a
            // defined block inputarg; the framestate threads every `Some`
            // slot positionally, so an undefined phantom would reach Pass 2
            // as a `Link.arg` defined at no source.  Drop it to `None`: a
            // dead local is irrelevant, and a later read of a genuinely
            // live-but-undefined slot still fails loud as an uninitialised
            // local through `resolve_place` — the same use-before-def
            // signal the monotonic `edge_args` raises.
            for slot in ex.entries.iter_mut() {
                if let Some(v) = slot
                    && !self.graph.variable_defined_in_block(bb_id, v)
                {
                    *slot = None;
                }
            }
            exit_state[row] = Some(PackedFrameState::pack(ex.clone()));
            // Union this exit into each model successor's entry state.
            // Successors are read off the just-closed exits (the model
            // edges), skipping the return / except sinks and any
            // non-MIR target.
            let succ_targets: Vec<BlockId> = self
                .graph
                .block(bb_id)
                .exits
                .iter()
                .map(|l| l.target)
                .collect();
            for tgt in succ_targets {
                if tgt == returnblock || tgt == exceptblock {
                    continue;
                }
                let tmir = block_to_mir.get(tgt.0).copied().unwrap_or(usize::MAX);
                if tmir == usize::MAX {
                    continue;
                }
                // A loop-header successor keeps its pre-seeded entry
                // (Pass 0): its live-in phis are fixed, and this
                // predecessor threads into them in pass 2.  Unioning the
                // exit in would re-mint the phi Variables on the forward
                // edge and lose the seed the latch must thread into.
                if loop_headers.get(tmir).copied().unwrap_or(false) {
                    continue;
                }
                let trow = self.block_id.row(tmir);
                let merged = match entry_state[trow]
                    .take()
                    .as_ref()
                    .map(PackedFrameState::unpack)
                {
                    // First edge to reach `tmir` — `flowcontext.py`
                    // `make_next_block`, whose whole body is `newstate =
                    // state.copy()` before `SpamBlock(newstate)`.
                    // `FrameState.copy` is "make a copy of this state in
                    // which all Variables are fresh" (framestate.py), and
                    // the freshness is what the flowspace shape rests on: a
                    // block's inputargs are Variables *distinct* from the
                    // link args feeding them, so per-Variable liveness can
                    // tell "read in the predecessor" apart from "read in the
                    // target".  Carrying `ex` through unchanged instead made
                    // the target's inputargs literally the predecessor's
                    // Variables — measured at 142598 of 153927 blocks — and
                    // `prune_dead_phis` (`transform_dead_op_vars`) then kept
                    // every link arg whose Variable happened to be read
                    // anywhere in the predecessor, however dead it was in the
                    // target.  Each slot is copied independently, so two
                    // slots holding one Variable become two distinct
                    // inputargs; `remove_duplicate_inputargs`
                    // (`remove_identical_vars_SSA`) recombines the ones that
                    // really are one phi column.
                    None => {
                        let copied = ex.copy(&mut self.graph);
                        self.note_copied_block_inputs(&ex, &copied);
                        copied
                    }
                    Some(prev) => {
                        let merged = prev.union(&ex, &mut self.graph).ok_or_else(|| {
                            LowerError::Unsupported(format!(
                                "framestate: union of predecessors failed at bb{tmir}"
                            ))
                        })?;
                        copy_agreed_concretetypes(&prev, &ex, &merged);
                        merged
                    }
                };
                entry_state[trow] = Some(PackedFrameState::pack(merged));
            }
        }

        // Pass 2 — re-argument the goto / branch links from framestates.
        for &bb in &rpo {
            let row = self.block_id.row(bb);
            // A block Pass 1 stubbed dead (empty-entry const-fold orphan)
            // has no exit state and a bare-raise body — its links carry no
            // framestate args to re-argument.  Skip it; the final guard
            // and the real-path dead-block prune both ignore it too.
            if self.graph.block(self.block_id[bb]).dead {
                continue;
            }
            let bb_id = self.emit_bb(bb);
            let ex = exit_state[row]
                .as_ref()
                .map(PackedFrameState::unpack)
                .ok_or_else(|| {
                    LowerError::Unsupported(format!(
                        "framestate: bb{bb} missing exit state in pass 2"
                    ))
                })?;
            let exits_meta: Vec<(usize, BlockId)> = self
                .graph
                .block(bb_id)
                .exits
                .iter()
                .enumerate()
                .map(|(i, l)| (i, l.target))
                .collect();
            for (idx, tgt) in exits_meta {
                if tgt == returnblock || tgt == exceptblock {
                    continue;
                }
                let tmir = block_to_mir.get(tgt.0).copied().unwrap_or(usize::MAX);
                if tmir == usize::MAX {
                    continue;
                }
                let tgt_state = entry_state[self.block_id.row(tmir)]
                    .as_ref()
                    .map(PackedFrameState::unpack)
                    .ok_or_else(|| {
                        LowerError::Unsupported(format!(
                            "framestate: bb{tmir} missing entry state in pass 2"
                        ))
                    })?;
                // `try_getoutputargs` (not the panicking `getoutputargs`):
                // a loop header pre-seeded with live-in phis bypasses the
                // union's None-kill, so a phantom slot scrubbed to `None`
                // in this predecessor's exit could leave a target
                // `Variable` slot unbound in `self`.  Decline such a
                // mismatch to the monotonic fallback instead of panicking.
                let outputargs =
                    ex.try_getoutputargs(&tgt_state, &self.graph)
                        .ok_or_else(|| {
                            LowerError::Unsupported(format!(
                                "framestate: getoutputargs phantom-slot mismatch on edge bb{bb} -> \
                         bb{tmir} in graph {:?} — declines to monotonic",
                                self.graph.name,
                            ))
                        })?;
                // Self-validation: every output arg is an exit-state cell
                // of `bb`, hence must be defined in `bb_id` (as an
                // inputarg or op result).  If the threading would emit a
                // Link.arg undefined at its source block, bail to the
                // monotonic path rather than hand a malformed graph to
                // the downstream rtyper / legacy fallback (which both
                // consume this same graph and would fault on the
                // undefined operand).  This is the `flowspace_adapter`
                // "every referenced operand must be defined as a block
                // inputarg or op result" invariant, checked at the
                // threading site.
                for arg in &outputargs {
                    if let LinkArg::Value(var) = arg
                        && !self.graph.variable_defined_in_block(bb_id, var)
                    {
                        if std::env::var_os("MAJIT_MIR_FRAMESTATE_DEBUG").is_some() {
                            self.debug_dump_undefined_link_arg(
                                bb, bb_id, tgt, tmir, var, &ex, &tgt_state,
                            );
                        }
                        return Err(LowerError::Unsupported(format!(
                            "framestate: threaded Link.arg (var id={}) undefined in source \
                                 bb{bb} (BlockId {:?}) for edge -> bb{tmir} (BlockId {:?}) \
                                 in graph {:?}",
                            var.id(),
                            bb_id,
                            tgt,
                            self.graph.name,
                        )));
                    }
                }
                self.graph.block_mut(bb_id).exits[idx].args = outputargs;
            }
        }

        // Final adapter-rejection guard.  The real-path `flowspace_adapter`
        // rejects a graph if any non-dead block has a `Link.args` operand
        // that is not defined in its source block (an inputarg or op
        // result).  A model-reachable raise lowers to `set_raise`, whose
        // orphan etype/evalue are exactly such operands, so the graph is a
        // guaranteed real-path Skip — Pass 2 leaves those exceptblock /
        // returnblock links untouched, so scan every reachable block's
        // links here.  Threading framestate phis into a graph that will
        // Skip buys no census drain (it cannot Match) and hands the phis to
        // the legacy fallback, which cannot type non-startblock inputargs
        // and miscolours them `GcRef` (assembler Ref/Int mismatch).
        // Decline such graphs to the monotonic lowering, which the legacy
        // path consumes exactly as today; the decline is drain-neutral
        // because a Match is impossible while the orphan operand stands.
        for &bb in &rpo {
            if self.graph.block(self.block_id[bb]).dead {
                continue;
            }
            let bb_id = self.emit_bb(bb);
            let undefined: Option<u64> = self
                .graph
                .block(bb_id)
                .exits
                .iter()
                .flat_map(|l| l.args.iter())
                .find_map(|arg| match arg {
                    LinkArg::Value(var) if !self.graph.variable_defined_in_block(bb_id, var) => {
                        Some(var.id())
                    }
                    _ => None,
                });
            if let Some(id) = undefined {
                return Err(LowerError::Unsupported(format!(
                    "framestate: declines graph {:?} — reachable bb{bb} (BlockId {:?}) has a \
                     Link.args operand (var id={id}) undefined at its source (real-path adapter \
                     would Skip; monotonic lowering is legacy-safe)",
                    self.graph.name, bb_id,
                )));
            }
        }

        Ok(())
    }

    /// Diagnostic dump for a framestate threading that would emit a
    /// `Link.arg` undefined at its source block.  Gated behind
    /// `MAJIT_MIR_FRAMESTATE_DEBUG`.  Prints the source / target block
    /// shapes and both framestates so the alignment bug can be located.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::front::mir) fn debug_dump_undefined_link_arg(
        &self,
        bb: usize,
        bb_id: BlockId,
        tgt: BlockId,
        tmir: usize,
        var: &Variable,
        ex: &FrameState,
        tgt_state: &FrameState,
    ) {
        let id_list = |entries: &[Option<Variable>]| -> String {
            entries
                .iter()
                .map(|e| match e {
                    Some(v) => v.id().to_string(),
                    None => "_".to_string(),
                })
                .collect::<Vec<_>>()
                .join(",")
        };
        let src = self.graph.block(bb_id);
        let src_inputargs: Vec<u64> = src.inputargs.iter().map(|v| v.id()).collect();
        let src_op_results: Vec<u64> = src
            .operations
            .iter()
            .filter_map(|op| op.result.as_ref().map(|v| v.id()))
            .collect();
        eprintln!(
            "[FRAMESTATE undefined-link-arg] graph={:?}\n  edge bb{bb}({:?}) -> bb{tmir}({:?})\n  \
             undefined var id={}\n  src.inputargs ids=[{:?}]\n  src.op_result ids=[{:?}]\n  \
             ex.entries ids=[{}]\n  tgt_state.entries ids=[{}]\n  \
             tgt_state.locals_w.len={} ex.locals_w.len={}",
            self.graph.name,
            bb_id,
            tgt,
            var.id(),
            src_inputargs,
            src_op_results,
            id_list(&ex.entries),
            id_list(&tgt_state.entries),
            tgt_state.locals_w.len(),
            ex.locals_w.len(),
        );
    }

    pub(in crate::front::mir) fn lower_block(&mut self, mir_bb: usize) -> Result<(), LowerError> {
        let bb: &BasicBlock = &self.body.body[mir_bb];
        let row = self.block_id.row(mir_bb);
        self.local_var = self.hydrate_words(self.block_entry_local_var[row].unpack());
        // Framestate scrubs a write-only `index_mut` place (`ll_setitem_fast`
        // has no loaded item), so the entry row hydrates it as `None`.
        // Bind a fresh Variable with no element value so `mem_slot` still
        // answers `MemSlot::Index`.
        self.bind_unvalued_index_places();
        self.positional_aggregate_locals =
            self.block_entry_positional_aggregate_locals[row].clone();
        self.string_byte_view_locals = self.block_entry_string_byte_view_locals[row]
            .iter()
            .collect();
        self.list_view_locals = self.block_entry_list_view_locals[row].iter().collect();

        // 1. Statements -> SpaceOperations on the corresponding block.
        for (s_idx, st) in bb.statements.iter().enumerate() {
            let kind = st.stmt_kind().map_err(LowerError::Schema)?;
            self.lower_statement(mir_bb, s_idx, kind)?;
        }

        // 2. Terminator -> block exits (close the block).
        let term = bb.term(self.llbc).map_err(LowerError::Schema)?;
        self.lower_terminator(mir_bb, term)
    }

    // Statements

    pub(in crate::front::mir) fn lower_statement(
        &mut self,
        mir_bb: usize,
        s_idx: usize,
        kind: StmtKind,
    ) -> Result<(), LowerError> {
        match kind {
            // `StorageLive` carries no IR except for a scoped Char-varsize
            // cell (`lltype.py` `ScopedAlloc.__enter__`). `StorageDead` of
            // a window pair plants `objectmodel.keepalive_until_here` on
            // the GC owner after the last use of the window local; a cell
            // local is `raw_free`'d (`ScopedAlloc.__exit__`).
            StmtKind::StorageLive(local) => {
                let local = local as usize;
                if self.is_cell_local(local) && self.cell_has_storage_live.contains(local) {
                    self.emit_cell_free(mir_bb, local)?;
                    self.emit_cell_malloc_bind(mir_bb, local)?;
                }
                Ok(())
            }
            StmtKind::StorageDead(local) => {
                let local = local as usize;
                if self.is_cell_local(local) {
                    self.emit_cell_free(mir_bb, local)?;
                }
                self.emit_window_owner_keepalive(mir_bb, local);
                Ok(())
            }
            StmtKind::Borrowck(_) => Ok(()),

            // `let _ = place` — read for side-effect tracking only.
            // The JIT does not need to materialize anything.
            StmtKind::PlaceMention(_) => Ok(()),

            // Statement-level Rust overflow / bounds assertion. Charon
            // emits every assert as a *terminator* (`TermKind::Assert`,
            // handled in `lower_terminator`), so this arm is not reached
            // by the current corpus; it is kept as defensive handling
            // for the paired `Assign(AddChecked) + Assert(!overflow)`
            // shape. Stripping is correct either way — same rationale as
            // the terminator arm: a Rust-debug check with no
            // Python-observable meaning.
            StmtKind::Assert(_) => Ok(()),

            StmtKind::Assign(place, rvalue) => self.lower_assign(mir_bb, place, rvalue),

            StmtKind::Unknown => Err(LowerError::Unsupported(format!(
                "bb{mir_bb} stmt#{s_idx}: unknown StmtKind"
            ))),
        }
    }

    pub(in crate::front::mir) fn lower_assign(
        &mut self,
        mir_bb: usize,
        dest: Place,
        rvalue: Rvalue,
    ) -> Result<(), LowerError> {
        // The destination place's post-projection type is the rvalue's
        // result type (for both a `Local` slot and a `place.field`
        // write).  `build_rvalue` reads it to pick a cast's result bank.
        // `_a = &_scope` for a bracket [`RootBracketPlan`] erased: the borrow
        // has no reader left, so it mints no Variable and the guard stays
        // unbound for the rest of the body.
        if let PlaceKind::Local(alias) = dest.kind
            && let Some(scope) = self.root_bracket.aliases.get(&(alias as usize)).copied()
            && self.root_bracket.is_erased_scope(scope)
            && matches!(&rvalue, Rvalue::Ref { place, .. } if matches!(place.kind, PlaceKind::Local(l) if l as usize == scope))
        {
            return Ok(());
        }
        // `_t = copy _slot` for an erased bracket's `base()` result: the index
        // has no reader left either, since every `get` it fed is answered by
        // the pinned value, so the temporary mints no Variable.
        if let PlaceKind::Local(index) = dest.kind
            && let Some(scope) = self
                .root_bracket
                .base_results
                .get(&(index as usize))
                .copied()
            && self.root_bracket.is_erased_scope(scope)
            && matches!(&rvalue, Rvalue::Use(_, _))
        {
            return Ok(());
        }
        // `_s = base + k` and `_i = _s.0` for the same bracket: the slot index
        // a `get` names past the first pin.  That `get` is answered by the
        // value its pin published, so the index has no reader either.
        if let PlaceKind::Local(index) = dest.kind
            && let Some(scope) = self.root_bracket.slot_temps.get(&(index as usize)).copied()
            && self.root_bracket.is_erased_scope(scope)
            && matches!(&rvalue, Rvalue::Use(_, _) | Rvalue::BinaryOp(..))
        {
            return Ok(());
        }
        // The array, borrows and cast a `pin_roots(&[..])` argument is built
        // through: the erased pin was their only reader.
        if let PlaceKind::Local(temp) = dest.kind
            && let Some(scope) = self.root_bracket.pin_temps.get(&(temp as usize)).copied()
            && self.root_bracket.is_erased_scope(scope)
            && matches!(
                &rvalue,
                Rvalue::Aggregate(..)
                    | Rvalue::Ref { .. }
                    | Rvalue::UnaryOp(..)
                    | Rvalue::Use(_, _)
            )
        {
            return Ok(());
        }
        // The borrow, aggregate and argument tuple that build a closure over
        // an erased index. The call is answered by the pinned value, so these
        // bind nothing.
        if let PlaceKind::Local(temp) = dest.kind
            && let Some(scope) = self
                .root_bracket
                .capture_temps
                .get(&(temp as usize))
                .copied()
            && self.root_bracket.is_erased_scope(scope)
            && matches!(
                &rvalue,
                Rvalue::Ref { .. } | Rvalue::Aggregate(..) | Rvalue::Use(_, _)
            )
        {
            return Ok(());
        }
        // A borrow of an erased owner-root handle is the handle's value, and
        // so is a reborrow through it ([`OwnerRootPlan`]).
        if let PlaceKind::Local(borrow) = dest.kind
            && let Some(handle) = self.owner_root.handle_of_borrow(borrow as usize)
            && matches!(&rvalue, Rvalue::Ref { .. })
        {
            let value = self.local_var[handle].as_ref().ok_or_else(|| {
                LowerError::Unsupported(format!(
                    "bb{mir_bb}: borrow of erased owner-root handle {handle} before its definition"
                ))
            })?.one()?;
            self.local_var[borrow as usize] = Some(LocalValue::One(value));
            return Ok(());
        }
        if let PlaceKind::Local(i) = dest.kind
            && self.len_shadow[i as usize].is_some()
        {
            return self.lower_pair_assign(mir_bb, i as usize, rvalue);
        }
        // A raw-buffer array's literal stores into its buffer; a borrow of it
        // is its item pointer.
        if let PlaceKind::Local(i) = dest.kind {
            match self.raw_array[i as usize] {
                Some(RawArrayRole::Storage { .. }) => {
                    return self.lower_raw_array_assign(mir_bb, i as usize, rvalue);
                }
                Some(RawArrayRole::Borrow(storage)) => {
                    self.local_var[i as usize] = self.local_var[storage].clone();
                    return Ok(());
                }
                None => {}
            }
        }
        // `&mut it` over a pair-slice iterator is the iterator itself, as a
        // borrow is everywhere in this value model; the referent is kept for
        // the `next` that reads through it.
        if let PlaceKind::Local(i) = dest.kind
            && let Rvalue::Ref { place, .. } | Rvalue::RawPtr { place, .. } = &rvalue
            && let Some(place) = self.pair_iter_referent(place)
            && let PlaceKind::Local(iter) = place.kind
        {
            let iter_value = self.local_var[iter as usize].clone().ok_or_else(|| {
                LowerError::Unsupported(format!(
                    "bb{mir_bb}: borrow of slice iterator local {iter} before any Assign"
                ))
            })?;
            self.local_var[i as usize] = Some(iter_value.clone());
            if let (Some(shadow), LocalValue::SliceIter { items, .. }) = (
                self.len_shadow.get(i as usize).copied().flatten(),
                &iter_value,
            ) {
                self.local_var[shadow] = Some(LocalValue::One(items.clone()));
            }
            return Ok(());
        }
        if self.cell_address_escapes_scope(&dest, &rvalue) {
            return Err(LowerError::Unsupported(format!(
                "bb{mir_bb}: raw scalar spill address would escape the storage scope"
            )));
        }
        let dest_ty = clone_tyref(&dest.ty);
        match dest.kind {
            PlaceKind::Local(i) => {
                // `_i = &atomic_slot` — remember the referent so a later
                // `<Atomic*>::store` through this local can write to the
                // place rather than to the value it resolved to.  Read
                // before `build_rvalue` consumes the rvalue.
                if let Some(place) = borrowed_place_referent(&rvalue) {
                    let local = i as usize;
                    let cell_reborrow = self
                        .reborrow_cell_param(&place)
                        .or_else(|| self.deref_cell_param_local(&place))
                        .is_some();
                    // `&raw const *p` is the address `p`, not the loaded
                    // word. Recording that local as a scalar alias makes
                    // the later `*q` skip `RawLoad`. A cell reborrow is
                    // recorded even when the local is assigned again.
                    if self.raw_primitive_reborrow_inner(&place).is_none()
                        && (cell_reborrow || !self.multi_assigned_locals.contains(&local))
                    {
                        self.atomic_ref_place.insert(local, place);
                    }
                }
                if !self.multi_assigned_locals.contains(&(i as usize))
                    && let Rvalue::Use(Operand::Copy(src) | Operand::Move(src), _) = &rvalue
                    && let PlaceKind::Local(src_local) = src.kind
                    && self.scalar_address_locals.contains(&(src_local as usize))
                    && !self.scalar_address_locals.contains(&(i as usize))
                {
                    // `_j = copy _i` where `_i` is a call's `&T`. The copy
                    // is that same address (`Rvalue::Use` aliases the
                    // Variable), so `*_j` still has to `RawLoad`.
                    self.scalar_address_locals.push(i as usize);
                }
                if !self.multi_assigned_locals.contains(&(i as usize)) {
                    // `_i = Ordering::<V>` — the ordering the store arm has
                    // to read before it may fold.
                    if let Some(name) = self.atomic_ordering_variant(&rvalue) {
                        self.atomic_ordering_locals.insert(i as usize, name);
                    }
                }
                // Capture the construction `owner_root` if this binding
                // is a positional aggregate, before `build_rvalue`
                // consumes the rvalue, so `.N` reads of the local can
                // later emit a symmetric `FieldRead __pos_<N>` carrying
                // the same owner (see `resolve_place`).
                let positional_owner = self.positional_aggregate_owner(&rvalue, &dest_ty);
                // `as_bytes()` marks its dest; a later `Copy` / `&*view`
                // assign must keep the mark so `slice::len` / `Rvalue::Len`
                // still emit `__strlen` rather than `arraylen_gc`.
                let dest_local = i as usize;
                let inherit_byte_view = match &rvalue {
                    Rvalue::Use(op, _) | Rvalue::Cast(_, op, _) => {
                        self.operand_is_string_byte_view(op)
                            || operand_tyref(op).is_some_and(|ty| tyref_is_w1_str(ty, self.llbc))
                    }
                    Rvalue::Ref { place, .. } | Rvalue::RawPtr { place, .. } => {
                        self.string_byte_view_locals
                            .iter()
                            .any(|&local| place_references_local(place, local))
                            || tyref_is_w1_str(&place.ty, self.llbc)
                    }
                    _ => false,
                };
                if inherit_byte_view {
                    if !self.string_byte_view_locals.contains(&dest_local) {
                        self.string_byte_view_locals.push(dest_local);
                    }
                } else {
                    // Last-write-wins, same as `positional_aggregate_locals`
                    // below: a later assign of an ordinary `&[u8]` must not
                    // keep `strlen`/`strgetitem` on a `W_UnicodeObject`.
                    self.string_byte_view_locals
                        .retain(|&local| local != dest_local);
                }
                let inherit_list_view = match &rvalue {
                    Rvalue::Use(op, _) | Rvalue::Cast(_, op, _) => {
                        self.operand_is_list_view(op)
                            || operand_tyref(op).is_some_and(|ty| tyref_is_u8_list(ty, self.llbc))
                    }
                    Rvalue::Ref { place, .. } | Rvalue::RawPtr { place, .. } => {
                        self.place_is_list_view(place)
                            || tyref_is_u8_list(&place.ty, self.llbc)
                            || matches!(
                                &place.kind,
                                PlaceKind::Projection(inner, ProjectionElem::Atom(atom))
                                    if atom == "Deref"
                                        && tyref_is_u8_list(&inner.ty, self.llbc)
                            )
                    }
                    _ => false,
                };
                if inherit_list_view {
                    if !self.list_view_locals.contains(&dest_local) {
                        self.list_view_locals.push(dest_local);
                    }
                } else {
                    self.list_view_locals.retain(|&local| local != dest_local);
                }
                // `_j = copy (*_i)` reads the singleton word out of the
                // `get_or_init` borrow; `_j` carries the same object.
                if let Rvalue::Use(Operand::Copy(src) | Operand::Move(src), _) = &rvalue
                    && !self.multi_assigned_locals.contains(&dest_local)
                    && self
                        .prebuilt_once_value_locals
                        .iter()
                        .any(|&local| place_references_local(src, local))
                {
                    self.prebuilt_once_value_locals.push(dest_local);
                }
                // `let q = p` keeps the cell ABI. `gc_mut_ref_param_root`
                // otherwise only names the original parameter local.
                if let Rvalue::Use(Operand::Copy(src) | Operand::Move(src), _) = &rvalue
                    && !self.multi_assigned_locals.contains(&dest_local)
                    && let PlaceKind::Local(src_local) = src.kind
                    && self.gc_mut_ref_params.contains(&(src_local as usize))
                {
                    self.gc_mut_ref_params.insert(dest_local);
                }
                // `_j = move _i` of a whole always-`Ok` local is the same
                // payload, so `_j` keeps the constant tag its `expect` /
                // `Discriminant` readers fold against (`with_roots!` returns
                // its body's `Result` through such a temporary).
                if let Rvalue::Use(Operand::Copy(src) | Operand::Move(src), _) = &rvalue
                    && let PlaceKind::Local(src_local) = src.kind
                    && !self.multi_assigned_locals.contains(&dest_local)
                    && let Some(&tag) = self.const_discriminant_locals.get(&(src_local as usize))
                {
                    self.const_discriminant_locals.insert(dest_local, tag);
                }
                // A single-assignment copy of a multi-field integer NamedConst
                // keeps the field literals. The struct itself is not a flow
                // value: emitting the const's nullary call leaves that path
                // unregistered. `fold_op_list` keeps any operation it cannot
                // prove safe, so a projected store, address-taking, or
                // whole-value consumer leaves the aggregate. A whole-local
                // copy is safe when that dest is itself only field-read
                // (or copied onward the same way).
                if !self.multi_assigned_locals.contains(&dest_local)
                    && !self.local_has_projected_store(dest_local)
                    && let Some(fields) = self.take_const_struct_fields(&rvalue)
                    && const_struct_uses_are_field_reads(self.llbc, self.body, dest_local)
                {
                    self.const_struct_fields[dest_local] = Some(fields);
                    self.local_var[dest_local] = None;
                    self.positional_aggregate_locals.remove(&dest_local);
                    return Ok(());
                }
                if self.is_cell_local(dest_local) {
                    let (op, result_var) = self.build_rvalue(mir_bb, rvalue, &dest_ty)?;
                    if let Some(op) = op {
                        let bb_id = self.emit_bb(mir_bb);
                        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                            result: Some(result_var.clone()),
                            kind: op,
                        });
                    }
                    self.emit_cell_store(mir_bb, dest_local, result_var)?;
                    return Ok(());
                }
                if self.list_view_locals.contains(&dest_local)
                    && tyref_is_u8_window(&dest_ty, self.llbc)
                {
                    let list = match &rvalue {
                        Rvalue::Use(op, _) | Rvalue::Cast(_, op, _) => {
                            Some(self.resolve_operand(mir_bb, op.clone())?)
                        }
                        Rvalue::Ref { place, .. } | Rvalue::RawPtr { place, .. } => {
                            Some(match &place.kind {
                                PlaceKind::Projection(inner, ProjectionElem::Atom(atom))
                                    if atom == "Deref" =>
                                {
                                    self.resolve_place(mir_bb, inner.as_ref().clone())?
                                }
                                _ => self.resolve_place(mir_bb, place.clone())?,
                            })
                        }
                        _ => None,
                    };
                    if let Some(list) = list {
                        self.bind_list_view_local(dest_local, list);
                        self.positional_aggregate_locals.remove(&dest_local);
                        return Ok(());
                    }
                }
                let (op, result_var) = self.build_rvalue(mir_bb, rvalue, &dest_ty)?;
                if let Some(op) = op {
                    let bb_id = self.emit_bb(mir_bb);
                    self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                        result: Some(result_var.clone()),
                        kind: op,
                    });
                }
                // The destination local takes on the freshly-minted
                // result Variable. Subsequent reads of the local
                // resolve to this Variable until the next Assign
                // overwrites the slot. A pair-slice item address
                // (`IterMut` payload) keeps the address, so `*slot = v`
                // writes through `ptr`.
                if inherit_byte_view && tyref_is_one_word_str_view(&dest_ty, self.llbc) {
                    stamp_as_rstr_str(&result_var);
                }
                self.local_var[i as usize] = Some(
                    if self.item_addr.get(i as usize).copied().unwrap_or(false) {
                        LocalValue::ItemAddr(result_var.clone())
                    } else {
                        LocalValue::One(result_var.clone())
                    },
                );
                // Keep the aggregate-local map in sync with the
                // last-write-wins slot: a non-aggregate rebind clears
                // the marker so the slot's reads collapse again.
                match positional_owner {
                    Some(owner) => {
                        self.positional_aggregate_locals.insert(i as usize, owner);
                    }
                    None => {
                        self.positional_aggregate_locals.remove(&(i as usize));
                    }
                }
                Ok(())
            }
            PlaceKind::Projection(inner, elem) => {
                // `place.field = val` / `*p = val` / `p[i] = val`.
                // Compute the rvalue, then emit a write op keyed by the
                // projection element. The destination local is NOT
                // updated — the write goes through indirection, the
                // base local remains the same Variable.
                if (tyref_is_u8_window(&dest_ty, self.llbc)
                    || (self.rvalue_is_str_view_source(&rvalue)
                        && tyref_is_bare_u8_slice(&dest_ty, self.llbc)))
                    && let Some((ptr, len)) = self.window_rvalue_parts(mir_bb, &rvalue)?
                {
                    let ptr = self.ensure_window_ccharp(mir_bb, ptr);
                    return self.emit_window_field_write(mir_bb, *inner, elem, ptr, len);
                }
                let value = match self.field_write_inline_const(&inner, &elem, &rvalue, &dest_ty) {
                    // An int / bool / float field write keeps its constant
                    // operand inline (`LinkArg::Const`) instead of
                    // materialising a `ConstInt` / `ConstFloat` + copy into
                    // a register: the assembler then takes the short `c`
                    // byte (`setfield_gc_i/rcd`) or a pool slot
                    // (`/rid` for a wide int, `/rfd` for a float).
                    // Mirrors RPython keeping the `Constant` box as a
                    // `setfield_gc` argument and deferring the
                    // short-vs-pool choice to `assembler.py` `emit_const`.
                    // `Constant::new` defers `concretetype` to the rtyper
                    // like every front-end synthesized constant; the
                    // assembler recovers the value kind from the
                    // self-describing Int / Bool / Float `ConstValue`
                    // variant via `constant_kind` (`getkind` fallback), so
                    // the `setfield_gc_<kind>` opname is keyed correctly.
                    Some(const_value) => {
                        LinkArg::Const(crate::flowspace::model::Constant::new(const_value))
                    }
                    None => {
                        let (op, value_var) = self.build_rvalue(mir_bb, rvalue, &dest_ty)?;
                        // If `build_rvalue` produced an op, emit it first
                        // so `value_var` is bound before the write reads
                        // it.
                        if let Some(op) = op {
                            let bb_id = self.emit_bb(mir_bb);
                            self.graph.block_mut(bb_id).operations.push(SpaceOperation {
                                result: Some(value_var.clone()),
                                kind: op,
                            });
                        }
                        LinkArg::Value(value_var)
                    }
                };
                self.emit_projection_write(mir_bb, *inner, elem, value, &dest_ty)
            }
            _ => Err(LowerError::Unsupported(format!(
                "bb{mir_bb}: Assign to {:?} destination not yet supported",
                place_kind_label(&dest.kind)
            ))),
        }
    }

    /// Re-resolve a recorded [`IndexElemAlias`] operand to the Variable
    /// live in the current block: prefer the current `local_var` binding
    /// of its source MIR local (rebound to a fresh inputarg if the
    /// operand was threaded across the index call's block split), and
    /// fall back to the Variable captured at the index call for a
    /// constant operand with no backing local.
    pub(in crate::front::mir) fn realias_operand(
        &self,
        local: Option<usize>,
        recorded: Variable,
    ) -> Variable {
        local
            .and_then(|l| self.local_var.get(l).and_then(|slot| slot.as_ref()))
            .and_then(|value| value.one().ok())
            .unwrap_or(recorded)
    }

    // Terminators

    pub(in crate::front::mir) fn lower_terminator(
        &mut self,
        mir_bb: usize,
        term: TermKind,
    ) -> Result<(), LowerError> {
        let bb_id = self.emit_bb(mir_bb);
        match term {
            TermKind::Return => {
                if let Some(ret) = self
                    .local_var
                    .get(0)
                    .and_then(|slot| slot.as_ref())
                    .and_then(|value| value.one().ok())
                    && self.variable_is_cell_ptr(&ret)
                {
                    return Err(LowerError::Unsupported(format!(
                        "bb{mir_bb}: raw scalar spill address would escape the storage scope"
                    )));
                }
                // A `-> ()` body materializes its implicit return as a
                // unit aggregate (`_0 = ()`), which lowers to a
                // Ref-typed transparent ctor.  Feeding that into the
                // return block colors the result kind 'r', contradicting
                // the declared void kind ('v').  RPython filters void
                // out of return links (NON_VOID); mirror that by routing
                // an empty void return.
                // The raw buffers of the body's arrays end with the frame.
                for storage in 0..self.raw_array.len() {
                    if matches!(self.raw_array[storage], Some(RawArrayRole::Storage { .. })) {
                        let items = self.local_var[storage]
                            .as_ref()
                            .and_then(|value| value.one().ok())
                            .expect("a raw array buffer is bound in every block");
                        self.emit_path_call(
                            mir_bb,
                            majit_ir::rvec::SLICE_BUFFER_FREE,
                            vec![items],
                            ValueType::Void,
                        );
                    }
                }
                self.free_bound_cells(mir_bb)?;
                if is_unit_type(&self.body.locals.locals[0].ty, self.llbc) {
                    self.graph.set_return(bb_id, None);
                    return Ok(());
                }
                // A pair slice returns its pointer and stores its length
                // through the caller's slot. A STR-view local 0 becomes a
                // window here (`rffi.get_raw_address_of_string`).
                if let Some(ret_slot) = self.ret_len_slot_local {
                    let ret_src = self.local_var[0].clone();
                    let (ptr, len, owner) = match ret_src {
                        Some(LocalValue::One(header))
                            if self.local_holds_str_view(0) || self.var_is_str_repr(&header) =>
                        {
                            let (ptr, len, owner) = self.emit_str_to_window(mir_bb, header);
                            (self.ensure_window_ccharp(mir_bb, ptr), len, Some(owner))
                        }
                        Some(LocalValue::One(list)) if self.local_holds_list_view(0) => {
                            let (ptr, len) = self.emit_list_view_to_window(mir_bb, list.clone())?;
                            (ptr, len, Some(list))
                        }
                        _ => {
                            let (ptr, len) = self.pair_local_parts(mir_bb, 0)?;
                            (ptr, len, self.window_owner_of_local(0))
                        }
                    };
                    let slot = self.local_var[ret_slot]
                        .as_ref()
                        .and_then(|value| value.one().ok())
                        .expect("return length slot is bound in every block");
                    self.emit_path_call(
                        mir_bb,
                        majit_ir::rvec::SLICE_LEN_SLOT_STORE,
                        vec![slot, len],
                        ValueType::Void,
                    );
                    // After the last use of `(data, len)` in this graph
                    // (`rtype_keepalive_until_here`). The owner does not
                    // cross the return (`rffi.get_raw_address_of_string`).
                    if let Some(owner) = owner {
                        self.emit_keepalive(mir_bb, owner);
                    }
                    self.graph.set_return(bb_id, Some(ptr));
                    return Ok(());
                }
                let ret = self.local_var[0]
                    .as_ref()
                    .ok_or_else(|| {
                        LowerError::Unsupported(format!(
                            "bb{mir_bb}: Return without any Assign to MIR local 0"
                        ))
                    })?
                    .one()?;
                self.graph.set_return(bb_id, Some(ret));
                Ok(())
            }
            TermKind::Abort(_) | TermKind::Panic { .. } | TermKind::UndefinedBehavior => {
                self.free_bound_cells(mir_bb)?;
                // A Rust panic-abort (`unreachable!()`, `panic!`,
                // failed `unwrap`), Charon `TerminatorKind::Panic`,
                // or Charon `TerminatorKind::UndefinedBehavior`.
                // Python-level exceptions never
                // reach here — they ride the `Result<_, PyError>`
                // Switch/Return edges as ordinary control flow — so
                // an Abort or Panic marks a "shouldn't occur at run-time"
                // path, exactly the implicit-exception raise of
                // `RaiseImplicit.nomoreblocks`
                // (`flowcontext.py`).  Closing the block
                // with `[Constant(AssertionError),
                // Constant(AssertionError(msg))]` lets
                // `remove_assertion_errors` (simplify.py)
                // prune the branch — e.g. the `else unreachable!()`
                // arm of a per-variant `let Instruction::X {..} =`
                // re-match folds away together with its discriminant
                // switch.
                self.graph.set_raise_implicit(bb_id, "AssertionError");
                melt_abort_message_chain(&mut self.graph, bb_id);
                Ok(())
            }
            TermKind::UnwindResume | TermKind::UnwindTerminate => {
                self.free_bound_cells(mir_bb)?;
                // Unwind-table cleanup resume, or Charon
                // `TerminatorKind::UnwindTerminate`.  Its only inbound edges
                // are `on_unwind` edges, all of which this lowering
                // drops, so the block is unreachable — close it as a
                // bare exception propagation; the flowspace adapter
                // converts only the reachable closure and never sees
                // it.  Python-level exceptions never reach here: they
                // ride the `Result<_, PyError>` Switch/Return edges as
                // ordinary control flow.
                self.graph.set_raise(bb_id, "mir-unwind");
                Ok(())
            }
            TermKind::Goto { target } => {
                let target_bb = self.block_id[target as usize];
                let args = self.edge_args(mir_bb, target as usize)?;
                self.graph.set_goto(bb_id, target_bb, args);
                Ok(())
            }
            TermKind::Assert {
                target, on_unwind, ..
            } => {
                // A Rust-level overflow / bounds / division-by-zero check
                // whose `on_unwind` successor is a bare UnwindResume panic
                // path. These are debug-build artifacts release builds
                // elide, with no Python-observable meaning: Python ints are
                // arbitrary-precision (no machine OverflowError), and any
                // IndexError / ZeroDivisionError is produced by an explicit
                // value-level guard that lowers to a `Result` Switch and is
                // already carried by the ArrayRead / BinOp op's canraise.
                // Strip the check — branch unconditionally to the success
                // continuation — leaving the panic path unreachable.
                // RPython does the same (`backendopt/removeassert.py`).
                let _ = on_unwind;
                let target_bb = self.block_id[target as usize];
                let args = self.edge_args(mir_bb, target as usize)?;
                self.graph.set_goto(bb_id, target_bb, args);
                Ok(())
            }
            TermKind::Switch { discr, targets } => self.lower_switch(mir_bb, discr, targets),
            TermKind::Call {
                call,
                target,
                on_unwind,
            } => {
                self.lower_call(mir_bb, call, target as usize, on_unwind as usize)?;
                // An intercept that closed the block itself never reached the
                // re-rooting `new`, and the handle would hold the bare value.
                if self.owner_root_reroot.take().is_some() {
                    return Err(LowerError::Unsupported(format!(
                        "bb{mir_bb}: owner-root producer call closed before its re-rooting `new`"
                    )));
                }
                Ok(())
            }
            // RootScope drop closes the shadow-stack bracket opened by
            // `push_roots`.  #1689 spells that as the Charon `drop_in_place`
            // glue call; this branch spells the same rewind as the named
            // `root_scope_close` residual so a crate that only imports the
            // opaque guard can still name it.  Emit the named residual, not
            // both — a second close would truncate an already-rewound stack.
            // FrameAnchor drop is the matching rewind for `frame_anchor_new`;
            // matching it here keeps that glue residual live.  Other drops
            // keep the legacy goto until their glue bodies and residual
            // callees are available to the translator.
            TermKind::Drop {
                place,
                fn_ptr,
                target,
                on_unwind,
            } => {
                // The destructor effect. The goto below is this terminator's
                // edge; `core::mem::drop` emits the same effect and then its
                // own call edge. A local `Vec` whose header this graph did
                // not allocate is the glue call, when this block holds that
                // header word. Freeing an unproven header would double-free
                // it. A conditional drop whose header never reached this
                // block, and a projection drop, keep the goto.
                let handled = self.emit_place_drop(mir_bb, &place, Some(&fn_ptr))?;
                let bound_vec_header = match &place.kind {
                    PlaceKind::Local(local)
                        if tyref_rust_vec_item_kind(&place.ty, self.llbc).is_some() =>
                    {
                        self.local_var
                            .get(*local as usize)
                            .and_then(|slot| slot.as_ref())
                            .and_then(|value| value.one().ok())
                            .is_some()
                    }
                    _ => false,
                };
                if !handled && bound_vec_header {
                    return self.lower_drop_as_glue_call(
                        mir_bb,
                        place,
                        fn_ptr,
                        target as usize,
                        on_unwind as usize,
                    );
                }
                let _ = on_unwind;
                let target_bb = self.block_id[target as usize];
                let args = self.edge_args(mir_bb, target as usize)?;
                self.graph.set_goto(bb_id, target_bb, args);
                Ok(())
            }
            TermKind::Unknown => Err(LowerError::Unsupported(format!(
                "bb{mir_bb}: unknown TermKind"
            ))),
        }
    }

    pub(in crate::front::mir) fn lower_switch(
        &mut self,
        mir_bb: usize,
        discr: Operand,
        targets: SwitchTargets,
    ) -> Result<(), LowerError> {
        let bb_id = self.emit_bb(mir_bb);
        let discr_var = self.resolve_operand(mir_bb, discr)?;
        match targets {
            SwitchTargets::If(then_bb, else_bb) => {
                // Constant-bool discriminant: take the known arm
                // unconditionally.  `flowcontext.py guessbool
                // FlowContext.guessbool` returns a Constant condition's
                // value directly instead of forking the recorder, so a
                // translation-time `const` gate like
                // `if WITHPREBUILTINT { ... }` leaves no branch in the
                // upstream graph.  The discriminant here is a fresh var
                // whose defining op sits in this same block when the
                // read folded to a constant (`static_addr_op` /
                // `global_literal_init_op` chain); the untaken target
                // stays in `block_id` but drops out via the adapter's
                // reachability prune.
                let const_cond = self
                    .graph
                    .block(bb_id)
                    .operations
                    .iter()
                    .rev()
                    .find(|op| op.result.as_ref() == Some(&discr_var))
                    .and_then(|op| match op.kind {
                        OpKind::ConstBool(b) => Some(b),
                        _ => None,
                    });
                if let Some(cond) = const_cond {
                    let taken = if cond { then_bb } else { else_bb };
                    let args = self.edge_args(mir_bb, taken as usize)?;
                    self.graph
                        .set_goto(bb_id, self.block_id[taken as usize], args);
                    return Ok(());
                }
                // Route through `set_branch` so the cond gets the
                // upstream `bool` UnaryOp wrap before becoming the
                // exitswitch (flowcontext.py POP_JUMP_IF_FALSE
                // `Variable.bool().eval(self)`).  Necessary because the
                // MIR discriminant for an `If` target can be a Ref
                // (e.g. a SyntheticTransparentCtor result) whereas
                // codewriter/assembler.rs::FlatOp::GotoIfNot expects
                // `cond.kind == RegKind::Int`.  `true_args` / `false_args`
                // carry each target block's input arguments; `set_branch`
                // asserts their arity against the block's `inputargs`.
                let then_args = self.edge_args(mir_bb, then_bb as usize)?;
                let else_args = self.edge_args(mir_bb, else_bb as usize)?;
                self.graph.set_branch(
                    bb_id,
                    discr_var,
                    self.block_id[then_bb as usize],
                    then_args,
                    self.block_id[else_bb as usize],
                    else_args,
                );
                Ok(())
            }
            SwitchTargets::SwitchInt(_int_ty, arms, default) => {
                // A `SwitchInt` on a one-word niche `Option` discriminant
                // switches on the pointer null-test `ne(base, null)` — a
                // `SomeBool`, not an `Int` tag.  RPython never switches on a
                // bool: `None`/`Some` on a maybe-null `Ptr` is a True/False
                // `If` (`ptr_nonzero`).  Route it through `set_branch` so the
                // bool exitswitch closes with `ExitCase::Bool`, matching the
                // `If` terminator path; a `SwitchInt` here would emit
                // `ExitCase::Const(Int)`, which rtype rejects against the
                // `BoolRepr` exitswitch (`BoolRepr::convert_const(Int)` → "not
                // a bool").  Genuine two-variant enum tag switches (real
                // `__discriminant` reads, `Int`/`IntegerRepr`) are not recorded
                // in `niche_disc_vars` and keep the `Int`-exitcase path below.
                if self.niche_disc_vars.contains(&discr_var.id())
                    || (arms.len() == 1 && arms[0].0.get("Ref").is_some())
                {
                    // A one-arm `Ref`+`DynTrait` SwitchInt is the `Some`
                    // side of a niche `Option` / pointer switch
                    // (`ptr_nonzero` `If`). `ExitCase::Const(Int)` would
                    // reject against the bool exitswitch.
                    return self.lower_niche_option_switch(mir_bb, bb_id, discr_var, arms, default);
                }
                let mut links: Vec<Link> = Vec::new();
                for (scalar, bb) in arms {
                    let case = scalar_to_const_value(&scalar, self.llbc).ok_or_else(|| {
                        LowerError::Unsupported(format!(
                            "bb{mir_bb}: SwitchInt case scalar shape not yet supported: {scalar}"
                        ))
                    })?;
                    let args = self.edge_args(mir_bb, bb as usize)?;
                    links.push(
                        Link::from_variables(
                            &self.graph,
                            args,
                            self.block_id[bb as usize],
                            Some(ExitCase::Const(case)),
                        )
                        .with_prevblock(bb_id)
                        .with_llexitcase_from_exitcase(),
                    );
                }
                if !self.switch_default_targets_panic_abort(default) {
                    let default_args = self.edge_args(mir_bb, default as usize)?;
                    links.push(
                        Link::from_variables(
                            &self.graph,
                            default_args,
                            self.block_id[default as usize],
                            Some(ExitCase::Const(ConstValue::UniStr("default".into()))),
                        )
                        .with_prevblock(bb_id),
                    );
                }
                self.graph.block_mut(bb_id).exitswitch = Some(ExitSwitch::Value(discr_var));
                self.graph.closeblock(bb_id, links);
                Ok(())
            }
        }
    }

    /// Close a `SwitchInt` whose discriminant is a one-word niche `Option`
    /// null-test (`ne(base, null)`, a `SomeBool`) as a True/False `If` via
    /// [`FunctionGraph::set_branch`], so the bool exitswitch closes with
    /// `ExitCase::Bool` instead of `ExitCase::Const(Int)`.  `Some` = tag 1 =
    /// non-null → the `if_true` arm; `None` = tag 0 = null → the `if_false`
    /// arm.  A two-variant match lowers to one explicit `(scalar, bb)` arm
    /// plus a `default` covering the complementary tag, so whichever of {0,1}
    /// is absent from `arms` is taken from `default` (the same 0/1/default
    /// triage as [`crate::front::result_exc::split_diamond_exits`]); a
    /// both-explicit shape (`arms.len() == 2`, no live default) is tolerated.
    pub(in crate::front::mir) fn lower_niche_option_switch(
        &mut self,
        mir_bb: usize,
        bb_id: BlockId,
        discr_var: Variable,
        arms: Vec<(serde_json::Value, u64)>,
        default: u64,
    ) -> Result<(), LowerError> {
        let mut some_bb: Option<u64> = None;
        let mut none_bb: Option<u64> = None;
        for (scalar, bb) in &arms {
            let case = scalar_to_const_value(scalar, self.llbc).ok_or_else(|| {
                LowerError::Unsupported(format!(
                    "bb{mir_bb}: niche-Option SwitchInt case scalar shape not supported: {scalar}"
                ))
            })?;
            match case {
                ConstValue::Int(1) | ConstValue::UInt(1) => some_bb = Some(*bb),
                ConstValue::Int(0) | ConstValue::UInt(0) => none_bb = Some(*bb),
                other => {
                    return Err(LowerError::Unsupported(format!(
                        "bb{mir_bb}: niche-Option SwitchInt has non-0/1 case {other:?}"
                    )));
                }
            }
        }
        // The tag absent from `arms` is the `default` target; if both are
        // explicit there is no live default to place.
        match (some_bb, none_bb) {
            (Some(_), None) => none_bb = Some(default),
            (None, Some(_)) => some_bb = Some(default),
            (Some(_), Some(_)) => {}
            (None, None) => {
                return Err(LowerError::Unsupported(format!(
                    "bb{mir_bb}: niche-Option SwitchInt lacks a 0/1 case"
                )));
            }
        }
        let some_bb = some_bb.expect("some arm resolved above");
        let none_bb = none_bb.expect("none arm resolved above");
        let true_args = self.edge_args(mir_bb, some_bb as usize)?;
        let false_args = self.edge_args(mir_bb, none_bb as usize)?;
        self.graph.set_branch(
            bb_id,
            discr_var,
            self.block_id[some_bb as usize],
            true_args,
            self.block_id[none_bb as usize],
            false_args,
        );
        Ok(())
    }

    /// A write-only `index_mut` place has no loaded element
    /// (`ll_setitem_fast`). After framestate scrubs the undefined
    /// Variable, the block entry hydrates that local as `None`. Bind a
    /// fresh Variable with no element value so `mem_slot` still answers
    /// `MemSlot::Index`.
    pub(in crate::front::mir) fn bind_unvalued_index_places(&mut self) {
        let locals: Vec<usize> = self
            .index_elem_alias
            .iter()
            .filter(|(_, alias)| alias.no_element)
            .map(|(local, _)| *local)
            .collect();
        for local in locals {
            if self
                .local_var
                .get(local)
                .and_then(|slot| slot.as_ref())
                .is_some()
            {
                continue;
            }
            if local >= self.local_var.len() {
                continue;
            }
            let place = self
                .graph
                .alloc_value_var_with_type(crate::model::ConcreteType::Unknown);
            self.local_var[local] = Some(LocalValue::One(place));
        }
    }

    /// Drop `local` from not-yet-lowered block inputargs.
    ///
    /// `ll_setitem_fast` does not produce an element value, so `edge_args`
    /// must not pass one. A block that already uses the construction-time
    /// inputarg keeps it: removing the defining inputarg would leave that
    /// use undefined.
    pub(in crate::front::mir) fn drop_unvalued_index_local(&mut self, local: usize) {
        let targets: Vec<(usize, BlockId)> = self.block_id.iter_lowered().collect();
        for (mir_bb, bid) in targets {
            if mir_bb == 0 {
                continue;
            }
            let row = self.block_id.row(mir_bb);
            let (old_len, pos) = {
                let Some(live) = self.block_live_in.get(row) else {
                    continue;
                };
                if !live.contains(local) {
                    continue;
                }
                let Some(pos) = live.iter().position(|idx| idx == local) else {
                    continue;
                };
                (live.count(), pos)
            };
            let untouched = {
                let block = self.graph.block(bid);
                block.operations.is_empty()
                    && block.exits.is_empty()
                    && block.exitswitch.is_none()
                    && block.inputargs.len() == old_len
            };
            if !untouched {
                continue;
            }
            self.block_live_in[row].remove(local);
            self.block_entry_local_var[row].unset(local);
            self.graph.block_mut(bid).inputargs.remove(pos);
            for pred in &mut self.graph.blocks {
                for link in &mut pred.exits {
                    if link.target == bid && link.args.len() == old_len {
                        link.args.remove(pos);
                    }
                }
            }
        }
    }

    pub(in crate::front::mir) fn edge_args(
        &mut self,
        from_bb: usize,
        target_bb: usize,
    ) -> Result<Vec<Variable>, LowerError> {
        // A place `drop_unvalued_index_local` removed from `block_live_in`
        // is absent here. `ll_setitem_fast` has no loaded item to pass.
        let local_indices = self.target_input_locals(target_bb)?;
        let mut converted_lens: Vec<(usize, Variable)> = Vec::new();
        let mut converted_owners: Vec<(usize, Variable)> = Vec::new();
        let mut args = Vec::with_capacity(local_indices.len());
        for local_idx in local_indices.iter().copied() {
            if let Some((_, len)) = converted_lens
                .iter()
                .find(|(shadow, _)| *shadow == local_idx)
            {
                args.push(len.clone());
                continue;
            }
            if let Some((_, owner)) = converted_owners
                .iter()
                .find(|(shadow, _)| *shadow == local_idx)
            {
                args.push(owner.clone());
                continue;
            }
            let shadow = self.len_shadow.get(local_idx).copied().flatten();
            let convert_on_link = (self.local_holds_str_view(local_idx)
                || self.local_holds_list_view(local_idx))
                && shadow.is_some_and(|slot| local_indices.contains(&slot));
            if convert_on_link {
                let header = self
                    .local_var
                    .get(local_idx)
                    .and_then(|slot| slot.as_ref())
                    .and_then(|value| value.one().ok())
                    .ok_or_else(|| {
                        LowerError::Unsupported("owner view joins a window".to_string())
                    })?;
                let (ptr, len) = if self.local_holds_str_view(local_idx) {
                    let (ptr, len, owner) = self.emit_str_to_window(from_bb, header.clone());
                    if let Some(slot) = self.owner_shadow.get(local_idx).copied().flatten() {
                        converted_owners.push((slot, owner));
                    }
                    (self.ensure_window_ccharp(from_bb, ptr), len)
                } else {
                    let (ptr, len) = self.emit_list_view_to_window(from_bb, header.clone())?;
                    if let Some(slot) = self.owner_shadow.get(local_idx).copied().flatten() {
                        converted_owners.push((slot, header.clone()));
                    }
                    (ptr, len)
                };
                if let Some(slot) = shadow {
                    converted_lens.push((slot, len));
                }
                self.merge_positional_aggregate_state(target_bb, local_idx);
                self.merge_string_byte_view_state(target_bb, local_idx, false);
                self.merge_list_view_state(target_bb, local_idx, false);
                args.push(ptr);
                continue;
            }
            let var = match self
                .local_var
                .get(local_idx)
                .and_then(|slot| slot.as_ref())
                .map(LocalValue::link_word)
            {
                Some(var) => var,
                None => {
                    if self
                        .owner_shadow
                        .iter()
                        .any(|slot| *slot == Some(local_idx))
                    {
                        args.push(self.emit_gc_null(from_bb));
                        continue;
                    }
                    let owner_is_str = self.len_shadow.iter().enumerate().any(|(owner, slot)| {
                        *slot == Some(local_idx)
                            && (self.local_holds_str_view(owner)
                                || self
                                    .local_var
                                    .get(owner)
                                    .and_then(|entry| entry.as_ref())
                                    .and_then(|value| value.one().ok())
                                    .is_some())
                    });
                    if owner_is_str {
                        return Err(LowerError::Unsupported(
                            "STR view joins a window".to_string(),
                        ));
                    }
                    let n_locals = self.body.locals.locals.len();
                    let ty = if local_idx < n_locals {
                        tyref_to_ast_string(&self.body.locals.locals[local_idx].ty, self.llbc)
                    } else {
                        "len_shadow_or_slot".to_string()
                    };
                    return Err(LowerError::Unsupported(format!(
                        "bb{from_bb}: edge to bb{target_bb} needs live MIR local {local_idx} ({ty}), \
                         but it is uninitialised"
                    )));
                }
            };
            self.merge_positional_aggregate_state(target_bb, local_idx);
            self.merge_string_byte_view_state(
                target_bb,
                local_idx,
                self.local_holds_str_view(local_idx),
            );
            self.merge_list_view_state(target_bb, local_idx, self.local_holds_list_view(local_idx));
            args.push(var);
        }
        Ok(args)
    }

    pub(in crate::front::mir) fn target_input_locals(
        &self,
        target_bb: usize,
    ) -> Result<Vec<usize>, LowerError> {
        if !self.block_id.is_lowered(target_bb) {
            return Err(LowerError::Schema(format!(
                "edge references unknown target bb{target_bb}"
            )));
        }
        let row = self.block_id.row(target_bb);
        if target_bb == 0 {
            let mut locals = Vec::with_capacity(self.arg_count);
            for local_idx in 1..=self.arg_count {
                locals.push(local_idx);
                locals.extend(self.len_shadow[local_idx]);
            }
            locals.extend(self.ret_len_slot_local);
            if let Some(live_set) = self.block_live_in.get(row) {
                for local_idx in live_set.iter() {
                    if !locals.contains(&local_idx) {
                        return Err(LowerError::Unsupported(format!(
                            "edge to startblock bb0 requires non-argument MIR local {local_idx}"
                        )));
                    }
                }
            }
            return Ok(locals);
        }
        Ok(self
            .block_live_in
            .get(row)
            .map(|live_set| live_set.iter().collect())
            .unwrap_or_default())
    }

    /// GC null for a window-owner join where this predecessor has none
    /// (`push_niche_null` / `ConstRefNull`).
    pub(in crate::front::mir) fn emit_gc_null(&mut self, mir_bb: usize) -> Variable {
        let bb_id = self.emit_bb(mir_bb);
        let null = self
            .graph
            .alloc_value_var_with_type(crate::model::ConcreteType::GcRef);
        self.graph.block_mut(bb_id).operations.push(SpaceOperation {
            result: Some(null.clone()),
            kind: OpKind::ConstRefNull,
        });
        null
    }

    /// `objectmodel.keepalive_until_here` as the host call the annotator
    /// accepts (`robjmodel_keepalive_until_here`). `normalize_unary_op_name`
    /// rejects a unary op named `keepalive`.
    pub(in crate::front::mir) fn emit_keepalive(&mut self, mir_bb: usize, owner: Variable) {
        let _ = self.emit_path_call(
            mir_bb,
            KEEPALIVE_UNTIL_HERE_PATH,
            vec![owner],
            ValueType::Void,
        );
    }

    /// True when `owner` is `emit_gc_null`'s `ConstRefNull`, or a
    /// `FrameState.copy` of one (`input_copied_from`).
    fn owner_is_gc_null(&self, owner: &Variable) -> bool {
        let mut id = owner.id();
        let mut seen = std::collections::HashSet::new();
        loop {
            let produced = self.graph.blocks.iter().any(|block| {
                block.operations.iter().any(|op| {
                    matches!(op.kind, OpKind::ConstRefNull)
                        && op.result.as_ref().is_some_and(|r| r.id() == id)
                })
            });
            if produced {
                return true;
            }
            if !seen.insert(id) || seen.len() > 64 {
                return false;
            }
            match self.input_copied_from.get(&id).copied() {
                Some(next) => id = next,
                None => return false,
            }
        }
    }

    /// `StorageDead` of a window local: keepalive the GC owner, then drop it.
    pub(in crate::front::mir) fn emit_window_owner_keepalive(
        &mut self,
        mir_bb: usize,
        local: usize,
    ) {
        let Some(owner) = self.window_owner_of_local(local) else {
            return;
        };
        // A defined null is an empty GC root (`liveness.py` keeps Registers).
        if !self.owner_is_gc_null(&owner) {
            self.emit_keepalive(mir_bb, owner);
        }
        if let Some(LocalValue::Pair(_, _, slot)) =
            self.local_var.get_mut(local).and_then(Option::as_mut)
        {
            *slot = None;
        }
        if let Some(owner_slot) = self.owner_shadow.get(local).copied().flatten() {
            self.local_var[owner_slot] = None;
        }
    }

    pub(in crate::front::mir) fn merge_positional_aggregate_state(
        &mut self,
        target_bb: usize,
        local_idx: usize,
    ) {
        if !self.block_id.is_lowered(target_bb) {
            return;
        }
        let row = self.block_id.row(target_bb);
        if self.block_positional_conflict[row].contains(local_idx) {
            return;
        }
        let incoming = self.positional_aggregate_locals.get(&local_idx).cloned();
        if !self.block_positional_seen[row].contains(local_idx) {
            self.block_positional_seen[row].insert(local_idx);
            if let Some(owner) = incoming {
                self.block_entry_positional_aggregate_locals[row].insert(local_idx, owner);
            }
            return;
        }
        let current = self.block_entry_positional_aggregate_locals[row]
            .get(&local_idx)
            .cloned();
        if current != incoming {
            self.block_positional_conflict[row].insert(local_idx);
            self.block_entry_positional_aggregate_locals[row].remove(&local_idx);
        }
    }

    pub(in crate::front::mir) fn merge_string_byte_view_state(
        &mut self,
        target_bb: usize,
        local_idx: usize,
        incoming: bool,
    ) {
        if !self.block_id.is_lowered(target_bb) {
            return;
        }
        let row = self.block_id.row(target_bb);
        if self.block_byte_view_conflict[row].contains(local_idx) {
            return;
        }
        if !self.block_byte_view_seen[row].contains(local_idx) {
            self.block_byte_view_seen[row].insert(local_idx);
            if incoming {
                self.block_entry_string_byte_view_locals[row].insert(local_idx);
            }
            return;
        }
        let current = self.block_entry_string_byte_view_locals[row].contains(local_idx);
        if current != incoming {
            self.block_byte_view_conflict[row].insert(local_idx);
            self.block_entry_string_byte_view_locals[row].remove(local_idx);
        }
    }

    pub(in crate::front::mir) fn merge_list_view_state(
        &mut self,
        target_bb: usize,
        local_idx: usize,
        incoming: bool,
    ) {
        if !self.block_id.is_lowered(target_bb) {
            return;
        }
        let row = self.block_id.row(target_bb);
        if self.block_list_view_conflict[row].contains(local_idx) {
            return;
        }
        if !self.block_list_view_seen[row].contains(local_idx) {
            self.block_list_view_seen[row].insert(local_idx);
            if incoming {
                self.block_entry_list_view_locals[row].insert(local_idx);
            }
            return;
        }
        let current = self.block_entry_list_view_locals[row].contains(local_idx);
        if current != incoming {
            self.block_list_view_conflict[row].insert(local_idx);
            self.block_entry_list_view_locals[row].remove(local_idx);
        }
    }
}

#[cfg(test)]
mod tests;
