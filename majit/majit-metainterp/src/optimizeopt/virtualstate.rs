/// Virtual state export/import for loop peeling.
///
/// Translated from rpython/jit/metainterp/optimizeopt/virtualstate.py.
///
/// After unrolling one iteration (the "preamble"), the optimizer captures the
/// abstract state of each value carried across the back-edge. On the next
/// iteration, this exported state is compared against the incoming values.
/// If compatible, the optimizer can directly apply known information (virtuals,
/// bounds, classes) without re-discovering it.
///
/// Key types:
/// - `VirtualState`: a snapshot of abstract info for all loop-carried values
/// - `VirtualStateInfo`: per-value abstract info (constant, virtual, class, etc.)
/// - State comparison determines if a compiled loop body can be reused
///
/// **Sharing**: VirtualStateInfo is wrapped in `Rc<...>` so the export tree
/// becomes a DAG mirroring RPython's reference-shared
/// AbstractVirtualStateInfo objects. When two parents reference the same
/// underlying box, they share the same `Rc<VirtualStateInfoNode>` and the
/// position-numbered enum_forced_boxes dedup (virtualstate.py, 274,
/// 352) prevents revisiting it.
use std::cell::Cell;
use std::ops::Deref;
use std::rc::Rc;

use indexmap::IndexSet;

use majit_ir::descr::descr_identity;
use majit_ir::operand::Operand;
use majit_ir::{DescrRef, GcRef, Op, OpCode, OpRef, Type, Value};

/// virtualstate.py: VirtualStatesCantMatch — raised when two virtual states
/// are incompatible and cannot be merged for bridge compilation.
#[derive(Clone, Debug)]
pub struct VirtualStatesCantMatch {
    pub msg: String,
}

impl VirtualStatesCantMatch {
    pub fn new(msg: &str) -> Self {
        VirtualStatesCantMatch {
            msg: msg.to_string(),
        }
    }
}

impl Default for VirtualStatesCantMatch {
    fn default() -> Self {
        Self::new("virtual states cannot match")
    }
}

/// virtualstate.py `GenerateGuardState` line-by-line port.
///
/// ```python
/// class GenerateGuardState(object):
///     def __init__(self, optimizer=None, guards=None, renum=None,
///                  bad=None, force_boxes=False):
///         self.optimizer = optimizer
///         self.cpu = optimizer.cpu
///         if guards is None:
///             guards = []
///         self.extra_guards = guards
///         if renum is None:
///             renum = {}
///         self.renum = renum
///         if bad is None:
///             bad = {}
///         self.bad = bad
///         self.force_boxes = force_boxes
/// ```
///
/// Pyre packs the per-`generate_guards` state into one struct instead
/// of threading 4-5 separate parameters through the recursion. The
/// `optimizer` + `cpu` slot is realised as `ctx: &mut OptContext`;
/// `ctx.cpu` is the `crate::cpu::Cpu` trait object that mirrors
/// `optimizer.cpu`, and `get_runtime_field/item/interiorfield` helpers
/// live on `OptContext`. `extra_guards` is the output buffer
/// owned by the caller. `renum` tracks position aliasing per
/// virtualstate.py. `bad` tracks the per-node "did this node
/// fail to match" set per virtualstate.py/:98 (Python object
/// identity → raw pointer identity here). `force_boxes` mirrors
/// virtualstate.py directly.
pub(crate) struct GenerateGuardState<'a> {
    pub ctx: &'a mut OptContext,
    pub extra_guards: &'a mut Vec<GuardRequirement>,
    pub renum: crate::FxIndexMap<i32, i32>,
    pub bad: crate::FxIndexSet<*const VirtualStateInfoNode>,
    pub force_boxes: bool,
}

impl std::fmt::Display for VirtualStatesCantMatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VirtualStatesCantMatch: {}", self.msg)
    }
}

impl std::error::Error for VirtualStatesCantMatch {}

/// virtualstate.py `not_virtual(cpu, type, info)` dispatch parity:
/// returns true when `incoming` belongs to the **same non-virtual class**
/// as the expected type.
///
/// RPython isinstance dispatch (virtualstate.py):
///   NotVirtualStateInfoPtr._generate_guards rejects any `other` that is
///   not isinstance(other, NotVirtualStateInfoPtr). Virtual/VArray/VStruct
///   are VirtualStateInfo subclasses, NOT NotVirtualStateInfoPtr, so they
///   fail this check. The force_boxes=True path is handled separately in
///   generate_guards_for_entry (line 1229) BEFORE this function is called.
pub(crate) fn info_type_matches(expected: Type, incoming: &VirtualStateInfo) -> bool {
    // history.py:182 / virtualstate.py not_virtual leaves: every value
    // box is int/ref/float. Void is never a value box class, so neither
    // `expected = Void` nor `Unknown(Void)` should ever flow through.
    if expected == Type::Void {
        panic!(
            "info_type_matches: expected=Type::Void has no value-box parity; \
             void-result ops are not value boxes (resoperation.py:260)"
        );
    }
    if let VirtualStateInfo::Unknown(Type::Void) = incoming {
        panic!(
            "info_type_matches: incoming Unknown(Void) — VirtualState leaves \
             are int/ref/float (virtualstate.py:655)"
        );
    }
    match (expected, incoming) {
        // LEVEL_UNKNOWN on both: compare the explicit type tags.
        (Type::Int, VirtualStateInfo::Unknown(Type::Int))
        | (Type::Float, VirtualStateInfo::Unknown(Type::Float))
        | (Type::Ref, VirtualStateInfo::Unknown(Type::Ref)) => true,
        (_, VirtualStateInfo::Unknown(_)) => false,
        // Int expected: incoming must be Int-typed constant or IntBounded.
        (Type::Int, VirtualStateInfo::IntBounded(_)) => true,
        (Type::Int, VirtualStateInfo::Constant(Value::Int(_))) => true,
        (Type::Int, _) => false,
        // Float expected: incoming must be Float constant.
        (Type::Float, VirtualStateInfo::Constant(Value::Float(_))) => true,
        (Type::Float, _) => false,
        // Ref expected: incoming must be a NotVirtualStateInfoPtr subclass
        // (NonNull / KnownClass / Ref-typed Constant). Virtual* variants
        // are NOT accepted — virtualstate.py:
        //   if not isinstance(other, NotVirtualStateInfoPtr):
        //       raise VirtualStatesCantMatch(...)
        (Type::Ref, VirtualStateInfo::NonNull)
        | (Type::Ref, VirtualStateInfo::KnownClass { .. }) => true,
        (Type::Ref, VirtualStateInfo::Constant(Value::Ref(_))) => true,
        (Type::Ref, _) => false,
        // Void: only Void-typed constants (rare).
        (Type::Void, VirtualStateInfo::Constant(Value::Void)) => true,
        (Type::Void, _) => false,
    }
}

use crate::optimizeopt::OptContext;
use crate::optimizeopt::info::{PtrInfo, PtrInfoExt};
use crate::optimizeopt::intutils::{IntBound, IntBoundMakeGuards};

/// Abstract info for one value at the loop boundary.
///
/// Mirrors the hierarchy in RPython's `AbstractVirtualStateInfo` and its subclasses:
/// `VirtualStateInfoConst`, `VirtualStateInfoVirtual`, `VirtualStateInfoNotVirtual`, etc.
///
/// In RPython every subclass is a Python class instance carrying mutable
/// `position` (and `position_in_notvirtuals` for NotVirtual leaves). Pyre
/// stores those per-instance attrs on [`VirtualStateInfoNode`] (the wrapper
/// around `Rc<VirtualStateInfoNode>` in shared trees) so each unique node has
/// its own slot, matching RPython's class-attr identity.
#[derive(Clone, Debug)]
pub enum VirtualStateInfo {
    /// Value is a known constant.
    Constant(Value),
    /// Value is a virtual instance with known fields.
    ///
    /// **Invariant**: `fields` NEVER contains typeptr (offset 0).
    /// Mirrors the `VirtualInfo.fields` invariant (RPython
    /// heaptracker.py all_fielddescrs excludes typeptr).
    /// Enforced at export/import boundaries via
    /// `debug_assert_no_typeptr_in_virtual_fields`.
    Virtual {
        descr: DescrRef,
        /// Known class as the immortal vtable address (`ConstInt(vtable)`,
        /// virtualstate.py:748) — a plain integer, never a traced ref.
        known_class: Option<i64>,
        /// ob_type field descriptor for force path (pyre offset 0).
        ob_type_descr: Option<DescrRef>,
        /// Field values as VirtualStateInfo (recursive). Excludes typeptr.
        fields: Vec<(u32, Rc<VirtualStateInfoNode>)>,
        /// Original field descriptors in parent-local slot order.
        /// virtualstate.py AbstractVirtualStructStateInfo.fielddescrs
        /// is a flat list and box access uses `fielddescrs[i].get_index()`.
        field_descrs: Vec<DescrRef>,
    },
    /// virtualstate.py: VArrayStateInfo — virtual array with known elements.
    VArray {
        descr: DescrRef,
        /// virtualstate.py:257-260 / :707-710: an unwritten array slot has
        /// `fieldstate[i] is None` and is skipped by every state walk.
        items: Vec<Option<Rc<VirtualStateInfoNode>>>,
        /// virtualstate.py: lenbound — known bounds on array length.
        /// None means unbounded.
        lenbound: Option<IntBound>,
    },
    /// virtualstate.py: VStructStateInfo — virtual struct.
    VStruct {
        descr: DescrRef,
        fields: Vec<(u32, Rc<VirtualStateInfoNode>)>,
        /// virtualstate.py AbstractVirtualStructStateInfo.fielddescrs
        /// stored as a flat parent-local list.
        field_descrs: Vec<DescrRef>,
    },
    /// virtualstate.py: VArrayStructStateInfo(arraydescr, fielddescrs, length)
    VArrayStruct {
        descr: DescrRef,
        /// virtualstate.py:289: self.fielddescrs — InteriorFieldDescr per field slot.
        fielddescrs: Vec<DescrRef>,
        /// virtualstate.py:316-320 / :707-710: dense element/field slots keep
        /// `None` for unwritten fields.
        element_fields: Vec<Vec<(u32, Option<Rc<VirtualStateInfoNode>>)>>,
    },
    /// Value has a known class (non-null).
    ///
    /// virtualstate.py NotVirtualStateInfoPtr with level=LEVEL_KNOWNCLASS.
    /// Implicitly Ref-typed in RPython via the class hierarchy; pyre keeps
    /// the explicit invariant that this variant is only emitted for Ref.
    /// `class_ptr` is the immortal vtable address (`ConstInt(vtable)`,
    /// virtualstate.py/748) — a plain integer, never a traced ref.
    KnownClass { class_ptr: i64 },
    /// Value is known non-null.
    ///
    /// virtualstate.py NotVirtualStateInfoPtr with level=LEVEL_NONNULL.
    /// Implicitly Ref-typed; emitted only for Ref.
    NonNull,
    /// Value has known integer bounds.
    ///
    /// virtualstate.py NotVirtualStateInfoInt with intbound set.
    /// Implicitly Int-typed; emitted only for Int.
    IntBounded(IntBound),
    /// No useful info (anything is compatible at this level).
    ///
    /// virtualstate.py NotVirtualStateInfo base class with
    /// LEVEL_UNKNOWN. In RPython the concrete class
    /// (`NotVirtualStateInfoInt` / `NotVirtualStateInfoPtr` / base
    /// `NotVirtualStateInfo` for Float) tags the type via
    /// `isinstance`-based dispatch in `_generate_guards`. pyre's flat
    /// OpRef namespace lets the optimizer unbox Ref→Int within a trace,
    /// so the type must be carried explicitly to prevent a ref-slot
    /// entry from accepting an int-typed incoming (and vice versa).
    /// This field is the direct counterpart of RPython's
    /// `NotVirtualStateInfoInt(cpu, 'i', info)` vs
    /// `NotVirtualStateInfoPtr(cpu, 'r', info)` constructor discriminant.
    Unknown(Type),
}

/// Per-instance wrapper around [`VirtualStateInfo`] carrying RPython's
/// `AbstractVirtualStateInfo.position` (virtualstate.py) and
/// `NotVirtualStateInfo.position_in_notvirtuals` (virtualstate.py)
/// class attributes.
///
/// In RPython each subclass instance holds these directly; pyre's flat
/// `Rc<VirtualStateInfo>` enum value cannot host per-instance mutability,
/// so the position cells live on this wrapper. Sharing semantics match
/// RPython exactly: when two parents reference the same logical box, the
/// same `Rc<VirtualStateInfoNode>` is shared, both parents see the same
/// position values, and the `state.position > self.position` dedup works
/// without an external side map.
///
/// [`Deref`] forwards to the inner enum so existing pattern-matching
/// against `VirtualStateInfo` variants stays valid.
#[derive(Debug)]
pub struct VirtualStateInfoNode {
    pub info: VirtualStateInfo,
    /// virtualstate.py `NotVirtualStateInfoPtr.lenbound`.
    /// Only non-virtual pointer leaves populate this; all other nodes keep
    /// the default `None`.
    pub lenbound: Option<IntBound>,
    /// virtualstate.py `AbstractVirtualStateInfo.position`. Default -1.
    /// Set by [`VirtualState::enum_top_level`] during construction.
    pub position: Cell<i32>,
    /// virtualstate.py `NotVirtualStateInfo.position_in_notvirtuals`.
    /// Only meaningful for the NotVirtual leaf variants
    /// (`Constant`/`KnownClass`/`NonNull`/`IntBounded`/`Unknown`); other
    /// variants leave this at -1.
    pub position_in_notvirtuals: Cell<i32>,
}

impl VirtualStateInfoNode {
    pub fn new(info: VirtualStateInfo) -> Self {
        Self::new_with_lenbound(info, None)
    }

    pub fn new_with_lenbound(info: VirtualStateInfo, lenbound: Option<IntBound>) -> Self {
        VirtualStateInfoNode {
            info,
            lenbound,
            position: Cell::new(-1),
            position_in_notvirtuals: Cell::new(-1),
        }
    }

    pub fn new_rc(info: VirtualStateInfo) -> Rc<Self> {
        Rc::new(Self::new(info))
    }

    pub fn new_rc_with_lenbound(info: VirtualStateInfo, lenbound: Option<IntBound>) -> Rc<Self> {
        Rc::new(Self::new_with_lenbound(info, lenbound))
    }

    /// virtualstate.py `AbstractVirtualStateInfo.enum`.
    /// ```python
    /// def enum(self, virtual_state):
    ///     if self.position != -1:
    ///         return
    ///     virtual_state.info_counter += 1
    ///     self.position = virtual_state.info_counter
    ///     self._enum(virtual_state)
    /// ```
    pub fn enum_into(&self, state: &mut VirtualState) {
        if self.position.get() != -1 {
            return;
        }
        state.info_counter += 1;
        self.position.set(state.info_counter);
        self._enum(state);
    }

    /// virtualstate.py: `_enum` per subclass. Dispatches over the inner
    /// enum, mirroring RPython's class-level method dispatch:
    /// - virtualstate.py AbstractVirtualStructStateInfo._enum
    /// - virtualstate.py VArrayStateInfo._enum
    /// - virtualstate.py VArrayStructStateInfo._enum
    /// - virtualstate.py NotVirtualStateInfo._enum
    fn _enum(&self, state: &mut VirtualState) {
        match &self.info {
            VirtualStateInfo::Constant(_) => {
                // virtualstate.py: LEVEL_CONSTANT short-circuit.
            }
            VirtualStateInfo::Virtual { fields, .. } | VirtualStateInfo::VStruct { fields, .. } => {
                for (_, child) in fields {
                    child.enum_into(state);
                }
            }
            VirtualStateInfo::VArray { items, .. } => {
                // virtualstate.py _enum: absent fieldstate entries do not
                // participate in position or notvirtual numbering.
                for child in items.iter().flatten() {
                    child.enum_into(state);
                }
            }
            VirtualStateInfo::VArrayStruct { element_fields, .. } => {
                for fields in element_fields {
                    for (_, child) in fields {
                        // virtualstate.py _enum: skip `None` fieldstate.
                        if let Some(child) = child {
                            child.enum_into(state);
                        }
                    }
                }
            }
            VirtualStateInfo::KnownClass { .. }
            | VirtualStateInfo::NonNull
            | VirtualStateInfo::IntBounded(_)
            | VirtualStateInfo::Unknown(_) => {
                // virtualstate.py NotVirtualStateInfo._enum:
                //     if self.level == LEVEL_CONSTANT: return
                //     self.position_in_notvirtuals = virtual_state.numnotvirtuals
                //     virtual_state.numnotvirtuals += 1
                //
                // The Constant case is dispatched above; reaching here
                // means self is a non-constant NotVirtual leaf.
                self.position_in_notvirtuals
                    .set(state.numnotvirtuals as i32);
                state.numnotvirtuals += 1;
            }
        }
    }
}

impl Deref for VirtualStateInfoNode {
    type Target = VirtualStateInfo;
    fn deref(&self) -> &VirtualStateInfo {
        &self.info
    }
}

impl Clone for VirtualStateInfoNode {
    /// Cloning resets position/position_in_notvirtuals to -1 because the
    /// cloned node is a fresh instance — RPython's `__init__` does not
    /// inherit position from a source instance, and `enum` re-assigns
    /// positions when called on the new VirtualState.
    fn clone(&self) -> Self {
        VirtualStateInfoNode {
            info: self.info.clone(),
            lenbound: self.lenbound.clone(),
            position: Cell::new(-1),
            position_in_notvirtuals: Cell::new(-1),
        }
    }
}

impl VirtualStateInfo {
    /// Whether this info represents a virtual (not yet allocated) value.
    pub fn is_virtual(&self) -> bool {
        matches!(
            self,
            VirtualStateInfo::Virtual { .. }
                | VirtualStateInfo::VArray { .. }
                | VirtualStateInfo::VStruct { .. }
                | VirtualStateInfo::VArrayStruct { .. }
        )
    }

    /// virtualstate.py implicit Box.type parity: return the Type of this
    /// state entry, as if the originating `box.type` were still accessible.
    /// Returns `None` only for Void/unreachable constants, since every
    /// RPython Box has one of the three types.
    pub fn info_type(&self) -> Option<Type> {
        match self {
            VirtualStateInfo::Unknown(tp) => Some(*tp),
            VirtualStateInfo::Constant(v) => Some(v.get_type()),
            VirtualStateInfo::IntBounded(_) => Some(Type::Int),
            VirtualStateInfo::NonNull
            | VirtualStateInfo::KnownClass { .. }
            | VirtualStateInfo::Virtual { .. }
            | VirtualStateInfo::VArray { .. }
            | VirtualStateInfo::VStruct { .. }
            | VirtualStateInfo::VArrayStruct { .. } => Some(Type::Ref),
        }
    }
}

fn visit_const_value(value: &mut Value, visitor: &mut dyn FnMut(&mut GcRef)) {
    if let Value::Ref(gcref) = value {
        visitor(gcref);
    }
}

fn visit_info_node(
    ptr: *const VirtualStateInfoNode,
    visitor: &mut dyn FnMut(&mut GcRef),
    seen: &mut Vec<*const VirtualStateInfoNode>,
) {
    if ptr.is_null() || seen.contains(&ptr) {
        return;
    }
    seen.push(ptr);
    // SAFETY: `ptr` is an `Rc<VirtualStateInfoNode>` payload held by a
    // live `VirtualState`. The walk is stop-the-world. `TargetToken.virtual_state`
    // is mutated in place; this writes the shared node so every clone sees
    // the new address.
    let node_mut = unsafe { &mut *(ptr as *mut VirtualStateInfoNode) };
    visit_state_info(&mut node_mut.info, visitor, seen);
}

fn visit_state_info(
    info: &mut VirtualStateInfo,
    visitor: &mut dyn FnMut(&mut GcRef),
    seen: &mut Vec<*const VirtualStateInfoNode>,
) {
    match info {
        VirtualStateInfo::Constant(value) => visit_const_value(value, visitor),
        VirtualStateInfo::Virtual { fields, .. } => {
            // known_class is an immortal vtable integer, not a traced ref.
            for (_, child) in fields {
                visit_info_node(Rc::as_ptr(child), visitor, seen);
            }
        }
        VirtualStateInfo::VArray { items, .. } => {
            for child in items.iter_mut().flatten() {
                visit_info_node(Rc::as_ptr(child), visitor, seen);
            }
        }
        VirtualStateInfo::VStruct { fields, .. } => {
            for (_, child) in fields {
                visit_info_node(Rc::as_ptr(child), visitor, seen);
            }
        }
        VirtualStateInfo::VArrayStruct { element_fields, .. } => {
            for fields in element_fields {
                for (_, child) in fields {
                    if let Some(child) = child {
                        visit_info_node(Rc::as_ptr(child), visitor, seen);
                    }
                }
            }
        }
        // KnownClass.class_ptr is an immortal vtable integer, not a traced ref.
        VirtualStateInfo::KnownClass { .. }
        | VirtualStateInfo::NonNull
        | VirtualStateInfo::IntBounded(_)
        | VirtualStateInfo::Unknown(_) => {}
    }
}

/// A complete snapshot of abstract state at a loop boundary.
///
/// The `state` vector has one entry per loop-carried variable (matching the
/// `Jump`/`Label` args). During loop peeling, this is exported at the end
/// of the preamble and imported at the loop header.
///
/// Top-level entries are stored as `Rc<VirtualStateInfoNode>` so that
/// aliased loop-carried variables (two jump args resolving to the same
/// box) share a single state object — matching RPython's
/// `VirtualStateConstructor.create_state` cache where the same box always
/// returns the same `AbstractVirtualStateInfo` instance. The dedup walker
/// `enum_forced_boxes_for_entry` uses each node's `position` cell to
/// short-circuit revisits, mirroring `state.position > self.position`.
#[derive(Debug)]
pub struct VirtualState {
    /// Abstract info for each loop-carried variable, in order matching Label/Jump args.
    pub state: Vec<Rc<VirtualStateInfoNode>>,
    /// virtualstate.py `VirtualState.numnotvirtuals`. Maintained by
    /// [`Self::enum_top_level`].
    numnotvirtuals: usize,
    /// virtualstate.py `VirtualState.info_counter`. Increments through
    /// `enum_into` to assign per-instance positions.
    info_counter: i32,
    /// ConstPtr table indexes interned from this state's `Constant` refs.
    ///
    /// `TargetToken.virtual_state` is a field of a GC object; MiniMark traces
    /// `NotVirtualStateInfo.constbox` through that graph. The compiled-loop
    /// holder walk (`MetaInterp::walk_rd_consts_refs`) is that graph here:
    /// it stamps these indexes via `trace_index` and forwards the inline
    /// copies. The list lives as long as this state, independent of which
    /// mutator created the token.
    const_ptr_indexes: Vec<u32>,
}

impl VirtualState {
    /// Visit every inline `ConstPtr.value` / known-class Ref stored below this
    /// virtual state.
    ///
    /// RPython keeps `TargetToken.virtual_state` in the GC-traced object graph.
    /// The Rust port stores the same graph in `Rc<VirtualStateInfoNode>`; during
    /// a stop-the-world GC walk we mutate the node payloads in place so all
    /// shared references observe the forwarded address.
    pub fn walk_const_ptr_refs_mut(&mut self, visitor: &mut dyn FnMut(&mut GcRef)) {
        let mut seen: Vec<*const VirtualStateInfoNode> = Vec::new();
        for node in &self.state {
            visit_info_node(Rc::as_ptr(node), visitor, &mut seen);
        }
    }

    pub fn new(state: Vec<VirtualStateInfo>) -> Self {
        let state: Vec<Rc<VirtualStateInfoNode>> = state
            .into_iter()
            .map(VirtualStateInfoNode::new_rc)
            .collect();
        Self::from_shared_rcs(state)
    }

    /// Construct directly from already-shared `Rc`s. Used by `export_state`
    /// so two top-level jump args resolving to the same box collapse onto
    /// the same `Rc<VirtualStateInfoNode>` (matching RPython's
    /// `VirtualStateConstructor.create_state` box-keyed cache).
    pub fn from_shared_rcs(state: Vec<Rc<VirtualStateInfoNode>>) -> Self {
        let mut vs = VirtualState {
            state,
            numnotvirtuals: 0,
            info_counter: -1,
            const_ptr_indexes: Vec::new(),
        };
        vs.enum_top_level();
        vs
    }

    /// Record interned ConstPtr indexes this state still names.
    ///
    /// `finalize_short_preamble` assigns `TargetToken.virtual_state` and
    /// then records the indexes that graph (and the short preamble) hold.
    /// `walk_rd_consts_refs` stamps them for the token's life. Replaces any
    /// previous list. Partial-trace ops are not part of that graph.
    pub(crate) fn retain_const_ptr_indexes(&mut self, indexes: &[u32]) {
        let mut indexes = indexes.to_vec();
        indexes.sort_unstable();
        indexes.dedup();
        indexes.retain(|&i| i != 0);
        self.const_ptr_indexes = indexes;
    }

    /// Stamp interned `ConstPtr.value` slots this state still names.
    ///
    /// The inline `Constant(Value::Ref)` copies are a different holder;
    /// [`Self::walk_const_ptr_refs_mut`] forwards those. `trace_index` is
    /// idempotent inside one wave.
    pub(crate) fn trace_const_ptr_indexes(&self, visitor: &mut dyn FnMut(&mut GcRef)) {
        for &index in &self.const_ptr_indexes {
            majit_ir::const_ptr_table::trace_index(index, visitor);
        }
    }

    /// virtualstate.py `VirtualState.__init__` per-state walk:
    /// ```python
    /// self.info_counter = -1
    /// self.numnotvirtuals = 0
    /// for s in state:
    ///     if s:
    ///         s.enum(self)
    /// ```
    /// Resets per-instance position cells before walking so that repeated
    /// calls (e.g., after `refresh_from_gc`) start from a clean
    /// RPython-equivalent state.
    pub fn enum_top_level(&mut self) {
        self.info_counter = -1;
        self.numnotvirtuals = 0;
        Self::reset_positions(&self.state);
        // Borrow split: clone the top-level Rcs into a temporary list so
        // the per-node `enum_into` can take `&mut self`. Rc::clone is
        // refcount-only. RPython `VirtualState.__init__` walks `state`
        // directly; keep the clone on the stack for the usual few inputs.
        let top: smallvec::SmallVec<[Rc<VirtualStateInfoNode>; 8]> =
            self.state.iter().cloned().collect();
        for node in &top {
            node.enum_into(self);
        }
    }

    /// Walk all reachable nodes and reset position cells to -1 so the
    /// next `enum_top_level` traversal mirrors a fresh RPython
    /// VirtualState.__init__ over fresh subclass instances.
    ///
    /// RPython `__init__` has no visited set (fresh instances). The set
    /// here only breaks shared-Rc DAGs; a linear SmallVec stays off the
    /// 48 B IndexMap-bucket class on the regex graphs.
    fn reset_positions(state: &[Rc<VirtualStateInfoNode>]) {
        let mut visited: smallvec::SmallVec<[usize; 8]> = smallvec::SmallVec::new();
        for node in state {
            Self::reset_positions_walk(node, &mut visited);
        }
    }

    fn reset_positions_walk(
        node: &Rc<VirtualStateInfoNode>,
        visited: &mut smallvec::SmallVec<[usize; 8]>,
    ) {
        let key = Rc::as_ptr(node) as usize;
        if visited.contains(&key) {
            return;
        }
        visited.push(key);
        node.position.set(-1);
        node.position_in_notvirtuals.set(-1);
        match &node.info {
            VirtualStateInfo::Virtual { fields, .. } | VirtualStateInfo::VStruct { fields, .. } => {
                for (_, child) in fields {
                    Self::reset_positions_walk(child, visited);
                }
            }
            VirtualStateInfo::VArray { items, .. } => {
                for child in items.iter().flatten() {
                    Self::reset_positions_walk(child, visited);
                }
            }
            VirtualStateInfo::VArrayStruct { element_fields, .. } => {
                for fields in element_fields {
                    for (_, child) in fields {
                        if let Some(child) = child {
                            Self::reset_positions_walk(child, visited);
                        }
                    }
                }
            }
            VirtualStateInfo::Constant(_)
            | VirtualStateInfo::KnownClass { .. }
            | VirtualStateInfo::NonNull
            | VirtualStateInfo::IntBounded(_)
            | VirtualStateInfo::Unknown(_) => {}
        }
    }

    /// Counts the leaves in a single top-level state entry, deduping shared
    /// `Rc<VirtualStateInfoNode>` subtrees via the caller-supplied visited map.
    /// The visited map (Rc::as_ptr → first imported box, NONE for the
    /// counting path) must be threaded across all top-level state entries
    /// in a single VirtualState walk so cross-entry shared substates are
    /// counted exactly once. Both the top-level Rc identity and the
    /// recursive nested Rcs participate in the dedup.
    pub fn count_forced_boxes_for_entry_static(
        rc: &Rc<VirtualStateInfoNode>,
        visited: &mut crate::FxIndexMap<usize, majit_ir::operand::Operand>,
    ) -> usize {
        // RPython virtualstate.py enum first-visit guard via
        // `position == -1` — every visited node is recorded so a later
        // visit returns 0 without re-counting. Pyre's parallel:
        //
        // - If the entry is already present (real box from
        //   `import_virtual_state_from_label_args_recurse`
        //   in `optimizer.rs`, or `NONE` from a prior counting visit),
        //   return 0. unroll.py `setinfo_from_preamble` parity:
        //   preserve the existing real box rather than overwriting.
        // - Otherwise insert `NONE` to mark this node visited, then
        //   recurse. Without this insert, repeated top-level visits
        //   to leaf variants (Unknown/NonNull/IntBounded/KnownClass)
        //   would each return 1 because they don't recurse through
        //   `_rc` and therefore never insert themselves; the dedup
        //   would silently fail and `numnotvirtuals` would over-count.
        let key = Rc::as_ptr(rc) as usize;
        if visited.contains_key(&key) {
            return 0;
        }
        visited.insert(key, majit_ir::operand::Operand::None);
        Self::count_forced_boxes_for_entry(rc, visited)
    }

    fn count_forced_boxes_for_entry(
        info: &VirtualStateInfo,
        visited: &mut crate::FxIndexMap<usize, majit_ir::operand::Operand>,
    ) -> usize {
        match info {
            VirtualStateInfo::Constant(_) => 0,
            VirtualStateInfo::Virtual { fields, .. } | VirtualStateInfo::VStruct { fields, .. } => {
                fields
                    .iter()
                    .map(|(_, child)| Self::count_forced_boxes_for_entry_rc(child, visited))
                    .sum()
            }
            VirtualStateInfo::VArray { items, .. } => items
                .iter()
                .flatten()
                .map(|child| Self::count_forced_boxes_for_entry_rc(child, visited))
                .sum(),
            VirtualStateInfo::VArrayStruct { element_fields, .. } => element_fields
                .iter()
                .flat_map(|fields| fields.iter().filter_map(|(_, child)| child.as_ref()))
                .map(|child| Self::count_forced_boxes_for_entry_rc(child, visited))
                .sum(),
            VirtualStateInfo::KnownClass { .. }
            | VirtualStateInfo::NonNull
            | VirtualStateInfo::IntBounded(_)
            | VirtualStateInfo::Unknown(_) => 1,
        }
    }

    /// Rc::as_ptr dedup wrapper for `count_forced_boxes_for_entry`,
    /// mirroring `enum_forced_boxes_recurse`.
    ///
    /// virtualstate.py `enum()` returns without touching
    /// `self.position` when `position != -1`; pyre must preserve any
    /// real box the caller already wrote into `visited` (e.g. from
    /// `import_virtual_state_from_label_args_recurse`). A blind
    /// `insert(.., NONE)` would overwrite that real box before the
    /// `is_some()` check, leaking NONE into downstream lookups.
    fn count_forced_boxes_for_entry_rc(
        rc: &Rc<VirtualStateInfoNode>,
        visited: &mut crate::FxIndexMap<usize, majit_ir::operand::Operand>,
    ) -> usize {
        let key = Rc::as_ptr(rc) as usize;
        if visited.contains_key(&key) {
            return 0;
        }
        visited.insert(key, majit_ir::operand::Operand::None);
        Self::count_forced_boxes_for_entry(rc, visited)
    }

    /// Number of non-virtual values (need concrete OpRefs at loop entry).
    /// virtualstate.py: num_boxes()
    pub fn num_boxes(&self) -> usize {
        self.numnotvirtuals
    }

    /// Total number of entries (virtual + non-virtual).
    pub fn num_entries(&self) -> usize {
        self.state.len()
    }

    /// Number of virtual entries.
    pub fn num_virtuals(&self) -> usize {
        self.state.iter().filter(|s| s.is_virtual()).count()
    }

    /// Whether this state has any virtual objects.
    pub fn has_virtuals(&self) -> bool {
        self.state.iter().any(|s| s.is_virtual())
    }

    /// virtualstate.py `make_inputargs(inputargs, optimizer, force_boxes=False)`.
    ///
    /// ```python
    /// def make_inputargs(self, inputargs, optimizer, force_boxes=False):
    ///     if optimizer.optearlyforce:
    ///         optimizer = optimizer.optearlyforce
    ///     assert len(inputargs) == len(self.state)
    ///     boxes = [None] * self.numnotvirtuals
    ///     # We try twice. The first time around we allow boxes to be forced
    ///     # which might change the virtual state if the box appear in more
    ///     # than one place among the inputargs.
    ///     if force_boxes:
    ///         for i in range(len(inputargs)):
    ///             self.state[i].enum_forced_boxes(boxes, inputargs[i], optimizer, True)
    ///     for i in range(len(inputargs)):
    ///         self.state[i].enum_forced_boxes(boxes, inputargs[i], optimizer)
    ///     return boxes
    /// ```
    ///
    /// Returns `Err(VirtualStatesCantMatch::default())` to mirror RPython's
    /// `raise VirtualStatesCantMatch` thrown from `enum_forced_boxes`.
    /// `enum_forced_boxes_for_entry` performs the
    /// `optimizer = optimizer.optearlyforce` redirection around each forced
    /// leaf by temporarily selecting `optearlyforce_idx` as the emitting pass.
    pub fn make_inputargs(
        &self,
        concrete_refs: &[OpRef],
        optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
        ctx: &mut OptContext,
        force_boxes: bool,
    ) -> Result<Vec<Operand>, VirtualStatesCantMatch> {
        // boxes = [None] * self.numnotvirtuals
        let mut boxes = vec![Operand::None; self.num_boxes()];
        // virtualstate.py `enum_forced_boxes` receives the box. Build each
        // input box once; both passes walk the same objects. A NONE or const
        // has no producer to bind (`Operand::from_opref`).
        let concrete_boxes: Vec<Operand> = concrete_refs
            .iter()
            .map(|&r| {
                if r.is_none() || r.is_constant() {
                    Operand::from_opref(r)
                } else {
                    ctx.materialize_operand_at(r)
                }
            })
            .collect();
        // virtualstate.py:664-667 — first pass with `force_boxes=True`.
        // RPython writes into the SAME `boxes` array on both passes; the
        // values converge after force because subsequent
        // `get_box_replacement` reads return the forced opref.
        // Shared-substate dedup uses `position` cells assigned during
        // `enum_top_level` (virtualstate.py:196, 274, 352).
        if force_boxes {
            for (idx, node) in self.state.iter().enumerate() {
                let box_ = concrete_boxes.get(idx).cloned().unwrap_or(Operand::None);
                Self::enum_forced_boxes_for_entry(
                    node, &box_, optimizer, ctx, &mut boxes, /* force_boxes */ true,
                )?;
            }
        }
        // virtualstate.py:668-669 — second pass with `force_boxes=False`,
        // unconditional. Mirrors RPython exactly.
        for (idx, node) in self.state.iter().enumerate() {
            let box_ = concrete_boxes.get(idx).cloned().unwrap_or(Operand::None);
            Self::enum_forced_boxes_for_entry(
                node, &box_, optimizer, ctx, &mut boxes, /* force_boxes */ false,
            )?;
        }
        Ok(boxes)
    }

    /// virtualstate.py `make_inputargs_and_virtuals(inputargs, optimizer, force_boxes=False)`.
    ///
    /// ```python
    /// def make_inputargs_and_virtuals(self, inputargs, optimizer, force_boxes=False):
    ///     inpargs = self.make_inputargs(inputargs, optimizer, force_boxes)
    ///     virtuals = []
    ///     for i in range(len(inputargs)):
    ///         if not isinstance(self.state[i], NotVirtualStateInfo):
    ///             virtuals.append(inputargs[i])
    ///     return inpargs, virtuals
    /// ```
    pub fn make_inputargs_and_virtuals(
        &self,
        concrete_refs: &[OpRef],
        optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
        ctx: &mut OptContext,
        force_boxes: bool,
    ) -> Result<(Vec<Operand>, Vec<OpRef>), VirtualStatesCantMatch> {
        let inputargs = self.make_inputargs(concrete_refs, optimizer, ctx, force_boxes)?;
        let virtuals: Vec<OpRef> = self
            .state
            .iter()
            .enumerate()
            .filter(|(_, info)| info.is_virtual())
            .filter_map(|(i, _)| concrete_refs.get(i).copied())
            .collect();
        Ok((inputargs, virtuals))
    }

    /// Return `make_inputargs_and_virtuals` plus the close-loop live-value
    /// source positions for each emitted non-virtual LABEL arg.
    ///
    /// The positions follow the `make_inputargs` output order from
    /// virtualstate.py, not a separate top-level `end_args` filter.
    /// Top-level virtual entries are skipped, so a LABEL arg produced by a
    /// virtual field maps to that field's concrete live-value slot when it is
    /// present in `concrete_refs`.
    #[expect(
        clippy::type_complexity,
        reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
    )]
    pub fn make_inputargs_and_virtuals_with_source_positions(
        &self,
        concrete_refs: &[OpRef],
        optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
        ctx: &mut OptContext,
        force_boxes: bool,
    ) -> Result<(Vec<Operand>, Vec<OpRef>, Vec<usize>), VirtualStatesCantMatch> {
        let (inputargs, virtuals) =
            self.make_inputargs_and_virtuals(concrete_refs, optimizer, ctx, force_boxes)?;
        let label_source_positions = self
            .label_source_positions_for_inputargs(concrete_refs, &inputargs, ctx)
            // No corresponding close-loop live slot: direct LABEL packing
            // is unavailable for this trace, so leave metadata empty and
            // let pack_front_target_live_values decline.
            .unwrap_or_default();
        Ok((inputargs, virtuals, label_source_positions))
    }

    fn label_source_positions_for_inputargs(
        &self,
        concrete_refs: &[OpRef],
        inputargs: &[Operand],
        ctx: &mut OptContext,
    ) -> Option<Vec<usize>> {
        if concrete_refs.len() != self.state.len() {
            return None;
        }
        // Resolve each candidate ONCE and look the answer up, rather than
        // rescanning `concrete_refs` per inputarg.
        //
        // virtualstate.py `_enum` assigns `position_in_notvirtuals` in
        // constant time on the same walk that will later place the box, and
        // virtualstate.py `enum_forced_boxes` writes straight to
        // `boxes[self.position_in_notvirtuals]`. Upstream never searches for
        // the slot, so its cost is one pass. Recovering the mapping by a
        // nested scan makes it quadratic in the entry count, and
        // `get_replacement_opref` is not a cheap comparison: it walks the
        // producer maps and then the forwarding chain, so the same candidate
        // was being re-resolved once per inputarg.
        //
        // Both lists are one entry per loop-carried value, so a virtualizable
        // array contributes one entry per declared element whether or not the
        // loop reads it (pyjitpl.py:3329 `original_boxes +=
        // self.virtualizable_boxes`). The quadratic term is therefore paid in
        // the square of the declared array length.
        //
        // Reverse iteration with an overwriting insert leaves the LOWEST
        // matching index in the map, which is what the first-match-wins scan
        // this replaces returned.
        let mut index_by_resolved: rustc_hash::FxHashMap<OpRef, usize> =
            rustc_hash::FxHashMap::with_capacity_and_hasher(
                concrete_refs.len(),
                Default::default(),
            );
        for (index, &candidate) in concrete_refs.iter().enumerate().rev() {
            if self.state[index].is_virtual() || candidate.is_constant() {
                continue;
            }
            let resolved_candidate = ctx.get_replacement_opref(candidate);
            index_by_resolved.insert(resolved_candidate, index);
        }
        inputargs
            .iter()
            .map(|inputarg| {
                let resolved_inputarg = inputarg.to_opref();
                index_by_resolved
                    .get(&resolved_inputarg)
                    .copied()
                    .or_else(|| {
                        let r = inputarg.to_opref();
                        r.is_input_arg().then_some(r.raw() as usize)
                    })
            })
            .collect()
    }

    /// Walk the virtual state tree the way RPython's `enum_forced_boxes`
    /// does (virtualstate.py AbstractVirtualStructStateInfo,
    /// 263-275 VArrayStateInfo, 333-354 VArrayStructStateInfo,
    /// 412-425 NotVirtualStateInfo) and write one entry per non-virtual,
    /// non-constant leaf into `boxes`.
    ///
    /// RPython has one method per state subclass; majit dispatches via
    /// `match` over the `VirtualStateInfo` enum (Rust enum vs Python
    /// class hierarchy — same dispatch, different shape).
    ///
    /// The Virtual / VStruct / VArray / VArrayStruct branches mirror
    /// the line `if info is None or not info.is_virtual(): raise
    /// VirtualStatesCantMatch()` (virtualstate.py, 266, 336):
    /// returning `Err(VirtualStatesCantMatch::default())` is the majit equivalent of raising
    /// `VirtualStatesCantMatch`.
    ///
    /// The leaf branch mirrors virtualstate.py — when the
    /// resolved box is virtual but the slot is non-virtual, force it
    /// through the optimizer if `force_boxes=True`, otherwise raise.
    ///
    /// **Shared-substate dedup**: RPython's `state.position > self.position`
    /// guard (virtualstate.py, 274, 352) skips revisiting a shared
    /// `AbstractVirtualStateInfo` so each unique state object's
    /// `NotVirtualStateInfo` gets exactly one slot. The Rust port carries
    /// `position` on each `VirtualStateInfoNode` (set by `enum_top_level`)
    /// and applies the same comparison at every recursive call site.
    fn enum_forced_boxes_for_entry(
        node: &VirtualStateInfoNode,
        box_: &Operand,
        optimizer: &mut crate::optimizeopt::optimizer::Optimizer,
        ctx: &mut OptContext,
        boxes: &mut [Operand],
        force_boxes: bool,
    ) -> Result<(), VirtualStatesCantMatch> {
        match &node.info {
            VirtualStateInfo::Constant(_) => Ok(()),
            VirtualStateInfo::Virtual { fields, .. } | VirtualStateInfo::VStruct { fields, .. } => {
                // virtualstate.py enum_forced_boxes:
                //     box = get_box_replacement(box)
                //     info = getptrinfo(box)
                //     if info is None or not info.is_virtual():
                //         raise VirtualStatesCantMatch()
                //     else:
                //         assert isinstance(info, AbstractStructPtrInfo)
                // operand-routing reader; cached once so the per-field walk below
                // doesn't re-clone PtrInfo per iteration.
                let info_snapshot = ctx
                    .resolve_operand_operand_opt(box_)
                    .as_ref()
                    .and_then(|b| ctx.peek_ptr_info(b));
                let is_virtual = info_snapshot.as_ref().is_some_and(|pi| pi.is_virtual());
                if !is_virtual {
                    return Err(VirtualStatesCantMatch::default());
                }
                // virtualstate.py:192-198: walk min(len(fielddescrs),
                // len(info._fields)) entries — RPython explicitly comments
                // that the min() guards against unvalidated callers.
                let info_field_count = info_snapshot
                    .as_ref()
                    .map(|pi| match pi {
                        PtrInfo::Virtual(vinfo) => vinfo.fields.len(),
                        PtrInfo::VirtualStruct(vinfo) => vinfo.fields.len(),
                        _ => 0,
                    })
                    .unwrap_or(0);
                let walk_count = fields.len().min(info_field_count);
                let field_boxes: Vec<Operand> = fields
                    .iter()
                    .take(walk_count)
                    .map(|(field_idx, _)| {
                        info_snapshot
                            .as_ref()
                            .and_then(|info| info.getfield(*field_idx))
                            .and_then(|e| e.as_value())
                            .unwrap_or(Operand::None)
                    })
                    .collect();
                for ((_, field_state), field_box) in
                    fields.iter().take(walk_count).zip(field_boxes.iter())
                {
                    // virtualstate.py `if state.position > self.position`
                    if field_state.position.get() > node.position.get() {
                        Self::enum_forced_boxes_for_entry(
                            field_state,
                            field_box,
                            optimizer,
                            ctx,
                            boxes,
                            force_boxes,
                        )?;
                    }
                }
                Ok(())
            }
            VirtualStateInfo::VArray { items, .. } => {
                // virtualstate.py VArrayStateInfo.enum_forced_boxes
                // operand-routing reader; cached once so the per-item walk
                // below doesn't re-clone PtrInfo per iteration.
                let info_snapshot = ctx
                    .resolve_operand_operand_opt(box_)
                    .as_ref()
                    .and_then(|b| ctx.peek_ptr_info(b));
                let is_virtual = info_snapshot.as_ref().is_some_and(|pi| pi.is_virtual());
                if !is_virtual {
                    return Err(VirtualStatesCantMatch::default());
                }
                // virtualstate.py:268-269: explicit length check.
                //     if len(self.fieldstate) > info.getlength():
                //         raise VirtualStatesCantMatch
                let array_len = info_snapshot
                    .as_ref()
                    .map(|pi| match pi {
                        PtrInfo::VirtualArray(ainfo) => ainfo.items.len(),
                        _ => 0,
                    })
                    .unwrap_or(0);
                if items.len() > array_len {
                    return Err(VirtualStatesCantMatch::default());
                }
                for (index, item_state) in items.iter().enumerate() {
                    let item_box = info_snapshot
                        .as_ref()
                        .and_then(|info| info.getitem(index))
                        .and_then(|e| e.as_value())
                        .unwrap_or(Operand::None);
                    // virtualstate.py:272-275: an absent fieldstate is not a
                    // forced box and contributes no notvirtual slot.
                    let Some(item_state) = item_state else {
                        continue;
                    };
                    // virtualstate.py `if state.position > self.position`
                    if item_state.position.get() > node.position.get() {
                        Self::enum_forced_boxes_for_entry(
                            item_state,
                            &item_box,
                            optimizer,
                            ctx,
                            boxes,
                            force_boxes,
                        )?;
                    }
                }
                Ok(())
            }
            VirtualStateInfo::VArrayStruct { element_fields, .. } => {
                // virtualstate.py VArrayStructStateInfo.enum_forced_boxes:
                //   for i in range(self.length):
                //       for descr in self.fielddescrs:
                //           index = i * len(self.fielddescrs) + descr.get_index()
                //           fieldstate = self.fieldstate[index]
                //           itembox = opinfo._items[i * len(self.fielddescrs) +
                //                                   descr.get_index()]
                //           if fieldstate is None:
                //               if itembox is not None:
                //                   raise VirtualStatesCantMatch
                //               continue
                //           if fieldstate.position > self.position:
                //               fieldstate.enum_forced_boxes(boxes, itembox, ...)
                //
                let runtime_fields: Vec<Vec<(u32, Operand)>> = match ctx
                    .resolve_operand_operand_opt(box_)
                    .as_ref()
                    .and_then(|b| ctx.peek_ptr_info(b))
                {
                    Some(crate::optimizeopt::info::PtrInfo::VirtualArrayStruct(vinfo)) => vinfo
                        .element_fields
                        .iter()
                        .map(|row| row.iter().map(|(i, b)| (*i, b.clone())).collect())
                        .collect(),
                    _ => return Err(VirtualStatesCantMatch::default()),
                };
                if runtime_fields.len() != element_fields.len() {
                    return Err(VirtualStatesCantMatch::default());
                }
                for (elem_idx, fields) in element_fields.iter().enumerate() {
                    let runtime_elem = &runtime_fields[elem_idx];
                    for (rt_field_idx, _) in runtime_elem {
                        if !fields.iter().any(|(fdidx, _)| fdidx == rt_field_idx) {
                            return Err(VirtualStatesCantMatch::default());
                        }
                    }
                    for (field_idx, field_state) in fields {
                        // Look up the runtime ref via the proper interior
                        // accessor (info.py:_compute_index parity), not the
                        // VirtualArray-only `getitem(flat_index)` helper.
                        let item_box = runtime_elem
                            .iter()
                            .find(|(fdidx, _)| fdidx == field_idx)
                            .map(|(_, op)| op.clone())
                            .unwrap_or(Operand::None);
                        // virtualstate.py:347-350: absent state requires an
                        // equally unwritten item and then skips recursion.
                        let Some(field_state) = field_state else {
                            if !item_box.to_opref().is_none() {
                                return Err(VirtualStatesCantMatch::default());
                            }
                            continue;
                        };
                        // virtualstate.py `if state.position > self.position`
                        if field_state.position.get() > node.position.get() {
                            Self::enum_forced_boxes_for_entry(
                                field_state,
                                &item_box,
                                optimizer,
                                ctx,
                                boxes,
                                force_boxes,
                            )?;
                        }
                    }
                }
                Ok(())
            }
            VirtualStateInfo::KnownClass { .. }
            | VirtualStateInfo::NonNull
            | VirtualStateInfo::IntBounded(_)
            | VirtualStateInfo::Unknown(_) => {
                // virtualstate.py NotVirtualStateInfo.enum_forced_boxes:
                //     if self.level == LEVEL_CONSTANT: return
                //     assert 0 <= self.position_in_notvirtuals
                //     assert optimizer is not None
                //     box = get_box_replacement(box)
                //     if box.type == 'r':
                //         info = getptrinfo(box)
                //         if info and info.is_virtual():
                //             if force_boxes:
                //                 info.force_box(box, optimizer)
                //             else:
                //                 raise VirtualStatesCantMatch
                //     boxes[self.position_in_notvirtuals] = box
                let resolved_box = ctx.resolve_operand_operand_opt(box_);
                let resolved: Operand = resolved_box.clone().unwrap_or_else(|| box_.clone());
                let forced = match resolved_box.as_ref().and_then(|b| ctx.peek_ptr_info(b)) {
                    // RPython: Virtualizable refs stay virtual across iterations.
                    Some(PtrInfo::Virtualizable(_)) => resolved,
                    Some(ptr_info) if ptr_info.is_virtual() => {
                        if !force_boxes {
                            return Err(VirtualStatesCantMatch::default());
                        }
                        // virtualstate.py `VirtualState.make_inputargs`:
                        // `if optimizer.optearlyforce: optimizer =
                        // optimizer.optearlyforce`.  The selected
                        // `NotVirtualStateInfo.enum_forced_boxes` therefore
                        // calls `info.force_box(box, OptEarlyForce)`, and the
                        // allocation/initializing stores enter at
                        // `OptEarlyForce.next_optimization` (OptHeap).  Merely
                        // calling `Optimizer::force_box` does not perform that
                        // redirection: it routes from `ctx.current_pass_idx`,
                        // which is normally OptHeap after the preceding flush.
                        // Preserve the caller's pass cursor around the exact
                        // upstream owner switch.
                        let saved_pass_idx = ctx.current_pass_idx;
                        ctx.current_pass_idx = ctx.optearlyforce_idx;
                        let forced = optimizer.force_box(
                            resolved_box
                                .as_ref()
                                .expect("virtual info lives on the box"),
                            ctx,
                        );
                        ctx.current_pass_idx = saved_pass_idx;
                        forced
                    }
                    _ => resolved,
                };
                // boxes[self.position_in_notvirtuals] = box
                // virtualstate.py:421 — each non-constant NotVirtual leaf
                // has its `position_in_notvirtuals` assigned during
                // `enum_top_level` / `_enum`, so the slot is deterministic
                // and aliased nodes naturally collapse onto the same slot.
                let slot_i32 = node.position_in_notvirtuals.get();
                debug_assert!(
                    slot_i32 >= 0,
                    "NotVirtual leaf reached enum_forced_boxes without position_in_notvirtuals \
                     assigned by enum_top_level"
                );
                let slot = slot_i32 as usize;
                let resolved_for_store = if forced.is_none() {
                    Operand::None
                } else {
                    ctx.resolve_operand_operand(&forced)
                };
                // virtualstate.py NotVirtualStateInfo{Int,Ptr}: Box.type
                // immutability. RPython dispatches `isinstance(self,
                // NotVirtualStateInfoInt)` vs `NotVirtualStateInfoPtr` on a
                // fixed-type Box, so a Ref-typed slot can never receive an
                // Int/Float source (and vice versa). pyre's flat-OpRef
                // forwarding via `get_box_replacement` can technically
                // bridge types; reject the cross here so the unrolled
                // trace falls back to `jump_to_preamble` exactly as RPython
                // raises VirtualStatesCantMatch.
                if let (Some(expected), Some(actual)) = (
                    node.info.info_type(),
                    ctx.opref_type(resolved_for_store.to_opref()),
                ) && expected != actual
                {
                    if crate::majit_log_enabled() {
                        eprintln!(
                            "[label-type-mismatch] slot={} expected={:?} actual={:?} \
                                 source={:?} resolved={:?}",
                            slot,
                            expected,
                            actual,
                            box_.to_opref(),
                            resolved_for_store.to_opref()
                        );
                    }
                    return Err(VirtualStatesCantMatch::default());
                }
                if let Some(dst) = boxes.get_mut(slot) {
                    *dst = resolved_for_store;
                }
                Ok(())
            }
        }
    }

    /// virtualstate.py `generalization_of(self, other, optimizer)`.
    ///
    /// `self` is the target loop state's requirement and `other` is the
    /// incoming state. Returns true if `self` can accept `other` *without*
    /// emitting any runtime guard — a pure structural generalization check.
    ///
    /// Mirrors the upstream exactly: build a `GenerateGuardState`, then for
    /// every entry call the per-entry `generate_guards` with `op = None` and
    /// `runtime_op = None`. Per the upstream docstring (virtualstate.py)
    /// "if None is passed in for op, no guard is ever generated, and this
    /// function degenerates to a generalization check"; with `runtime_box`
    /// `None` every guard-emitting arm fails fast, so the walk reduces to
    /// structure plus the renum alias-consistency check (virtualstate.py):
    /// two positions mapping to the same virtual node in `self` must map
    /// consistently in `other`. The previous `is_compatible` zip skipped that
    /// renum check entirely.
    ///
    /// `ctx` corresponds to the upstream `optimizer`; it is threaded into
    /// `GenerateGuardState` but never dereferenced, because no arm that reads
    /// a runtime value is reachable when `runtime_box` is `None`.
    pub fn generalization_of(&self, other: &VirtualState, ctx: &mut OptContext) -> bool {
        // virtualstate.py `assert len(self.state) == len(other.state)`.
        // pyre returns false rather than asserting so an arity-mismatched
        // target is skipped (the `pick_virtual_state` contract), matching the
        // Err-not-assert choice in `generate_guards`.
        if self.state.len() != other.state.len() {
            return false;
        }
        // virtualstate.py `state = GenerateGuardState(optimizer)` —
        // force_boxes defaults False. `extra_guards` is a throwaway buffer:
        // with op / runtime_op None no guard is ever pushed, and the result
        // is discarded regardless.
        let mut guards: Vec<GuardRequirement> = Vec::new();
        let mut state = GenerateGuardState {
            ctx,
            extra_guards: &mut guards,
            renum: crate::FxIndexMap::default(),
            bad: crate::FxIndexSet::default(),
            force_boxes: false,
        };
        // virtualstate.py:640-642 `for i in range(len(self.state)):
        //     self.state[i].generate_guards(other.state[i], None, None, state)`
        // catching VirtualStatesCantMatch → return False.
        for (i, (expected, incoming)) in self.state.iter().zip(other.state.iter()).enumerate() {
            if Self::generate_guards_for_entry_recursive(
                i,
                expected,
                incoming,
                OpRef::NONE,
                None,
                &mut state,
            )
            .is_err()
            {
                return false;
            }
        }
        true
    }

    /// virtualstate.py `generate_guards(self, other, boxes, runtime_boxes, optimizer)`.
    ///
    /// Generate guards to bridge from `other` state to `self` state.
    /// Returns `Ok(guards)` if the incoming state can be accepted with
    /// runtime guards, `Err(VirtualStatesCantMatch::default())` if fundamentally incompatible
    /// (`VirtualStatesCantMatch`).
    ///
    /// `boxes`: the actual OpRefs at each position (the guard's first
    /// argument in GUARD_VALUE etc.). `runtime_boxes`: the concrete runtime
    /// values at the jump point, used as an educated guess to decide whether
    /// emitting a guard is profitable (virtualstate.py). Both are
    /// always concrete, position-aligned lists — virtualstate.py
    /// asserts `len(self.state) == len(other.state) == len(boxes)
    /// == len(runtime_boxes)`.
    ///
    /// `force_boxes`: when true, Virtual incoming values can be accepted by
    /// NonVirtual targets (the virtual will be forced later by
    /// make_inputargs). Matches RPython's force_boxes.
    pub fn generate_guards(
        &self,
        other: &VirtualState,
        boxes: &[OpRef],
        runtime_boxes: &[OpRef],
        ctx: &mut OptContext,
        force_boxes: bool,
    ) -> Result<Vec<GuardRequirement>, VirtualStatesCantMatch> {
        if self.state.len() != other.state.len() {
            return Err(VirtualStatesCantMatch::default());
        }
        // virtualstate.py generate_guards `assert (len(self.state) == len(other.state)
        // == len(boxes) == len(runtime_boxes))`. The state-length pair is
        // handled above (Err, not assert — an arity-mismatched target is
        // skipped rather than crashing). The remaining two equalities are a
        // hard invariant: `boxes` (the jump args, get_box_replacement'd) and
        // `runtime_boxes` (the concrete jump-close values) are built from the
        // same jump arglist as `other` (export_state mints one state entry per
        // arg), so both align positionally with `self.state`. Release-active
        // `assert` mirrors the upstream `assert`, and the strict `boxes[i]` /
        // `runtime_boxes[i]` indexing below relies on it (no silent NONE
        // masking of a misaligned list).
        assert_eq!(
            boxes.len(),
            self.state.len(),
            "generate_guards: boxes ({}) not aligned with state ({})",
            boxes.len(),
            self.state.len(),
        );
        assert_eq!(
            runtime_boxes.len(),
            self.state.len(),
            "generate_guards: runtime_boxes ({}) not aligned with state ({})",
            runtime_boxes.len(),
            self.state.len(),
        );
        // virtualstate.py `GenerateGuardState.__init__` constructs
        // the per-call state container: `optimizer`, `cpu`,
        // `extra_guards`, `renum`, `bad`, `force_boxes`. pyre packs
        // these into `GenerateGuardState` and threads it through the
        // recursive entry helpers — matching the upstream `state`
        // parameter shape exactly. `extra_guards` is the output buffer;
        // `renum`/`bad` are populated during recursion.
        let mut guards: Vec<GuardRequirement> = Vec::new();
        let mut state = GenerateGuardState {
            ctx,
            extra_guards: &mut guards,
            renum: crate::FxIndexMap::default(),
            bad: crate::FxIndexSet::default(),
            force_boxes,
        };
        // virtualstate.py `GenerateGuardState.renum` and :84-94
        // `AbstractVirtualStateInfo.generate_guards` alias-consistency:
        //
        //     if self.position in state.renum:
        //         if state.renum[self.position] != other.position:
        //             raise VirtualStatesCantMatch(...)
        //         # else: already-seen position, skip _generate_guards
        //     else:
        //         state.renum[self.position] = other.position
        //         self._generate_guards(...)
        //
        // Catches the case where two distinct incoming positions map to
        // the same expected virtual node (the trace assumes two values
        // are aliased but the incoming disagrees), and short-circuits
        // duplicate visits to a node already proven compatible. The same
        // `IndexMap` instance is threaded through every recursive call
        // (virtualstate.py struct field, :260-261 array item,
        // :325-326 interior field) so nested virtual nodes share the
        // alias namespace with their top-level parents. Now lives on
        // `state.renum` (GenerateGuardState struct field).

        for (i, (expected, incoming)) in self.state.iter().zip(other.state.iter()).enumerate() {
            // virtualstate.py indexes `boxes[i]` / `runtime_boxes[i]`
            // directly; the asserts above guarantee i is in range. The
            // per-entry `runtime_box` is `Option` because nested struct/array
            // recursion has no runtime value for sub-fields (passes `None`);
            // at the top level it is always present.
            let box_opref = boxes[i];
            let runtime_box = Some(runtime_boxes[i]);
            if Self::generate_guards_for_entry_recursive(
                i,
                expected,
                incoming,
                box_opref,
                runtime_box,
                &mut state,
            )
            .is_err()
            {
                if crate::log_jtet_enabled() {
                    let runtime_value = runtime_box.and_then(|rb| {
                        state
                            .ctx
                            .get_box_replacement_operand_opt(rb)
                            .and_then(|b| state.ctx.get_constant_box(&b))
                    });
                    // The constant-folded reading above is not what the
                    // Constant arm decides on: that one calls `runtime_value_of`,
                    // which reads the box's OBSERVED value. Print both, or a
                    // refusal that had an observed value all along reads as a
                    // box with no value at all.
                    let observed = runtime_box.and_then(|rb| state.ctx.runtime_value_of(rb));
                    eprintln!(
                        "[jit][jte] virtualstate mismatch index={i} box={box_opref:?} runtime={runtime_box:?} runtime_value={runtime_value:?} observed={observed:?} expected={expected:?} incoming={incoming:?}"
                    );
                }
                return Err(VirtualStatesCantMatch::default());
            }
        }

        Ok(guards)
    }

    /// Recursive port of `AbstractVirtualStateInfo.generate_guards` and each
    /// subclass's `_generate_guards` implementation. Every recursive call
    /// performs the alias-consistency check in the shared `renum` namespace.
    ///
    /// `runtime_box` supplies RPython's concrete "educated guess" for emitting
    /// non-permanent guards. Nested virtuals obtain their concrete fields and
    /// items through `OptContext::get_runtime_field`, `get_runtime_item`, and
    /// `get_runtime_interiorfield`, corresponding to the methods of the same
    /// names on `GenerateGuardState`. Those helpers return fresh const-pool
    /// operands so compile-time virtual fields remain distinct from runtime
    /// values. A missing placeholder or unreadable descriptor propagates
    /// `None`, causing guards that require runtime evidence to reject the match.
    ///
    /// With `force_boxes`, a non-virtual target may accept a compatible virtual
    /// incoming value through `NotVirtualStateInfoPtr._generate_virtual_guards`.
    fn generate_guards_for_entry_recursive(
        arg_idx: usize,
        expected: &VirtualStateInfoNode,
        incoming: &VirtualStateInfoNode,
        box_opref: OpRef,
        runtime_box: Option<OpRef>,
        state: &mut GenerateGuardState,
    ) -> Result<(), VirtualStatesCantMatch> {
        // `AbstractVirtualStateInfo.generate_guards` requires an assigned
        // position. Pyre assigns
        // positions in `enum_top_level`; sentinel -1 means a node was
        // never enumerated, which is a constructor bug. RPython's
        // `assert` is always-on (no debug gate); pyre matches with
        // `assert!` rather than `debug_assert!` so release builds also
        // fail-fast on the constructor bug.
        let exp_pos = expected.position.get();
        let inc_pos = incoming.position.get();
        assert!(
            exp_pos >= 0,
            "expected entry {arg_idx} has unassigned position (enum_top_level not run?)"
        );
        // virtualstate.py:84-94: alias check + dedup. A matching prior
        // entry short-circuits the recursion (already proven compatible).
        // virtualstate.py:86 `state.bad[self] = state.bad[other] = None`
        // — the per-node identity is recorded so debug_print can flag
        // failing nodes.
        match state.renum.get(&exp_pos).copied() {
            Some(prev) if prev != inc_pos => {
                state.bad.insert(expected as *const _);
                state.bad.insert(incoming as *const _);
                if crate::log_jtet_enabled() {
                    eprintln!(
                        "[jit][jte] renum alias mismatch arg_idx={arg_idx} expected.position={exp_pos} \
                         prior_incoming={prev} current_incoming={inc_pos}"
                    );
                }
                return Err(VirtualStatesCantMatch::default());
            }
            Some(_) => return Ok(()),
            None => {
                state.renum.insert(exp_pos, inc_pos);
            }
        }
        let expected_info = &expected.info;
        let incoming_info = &incoming.info;
        // virtualstate.py NotVirtualStateInfoPtr._generate_guards:
        //
        //     if state.force_boxes and isinstance(other, VirtualStateInfo):
        //         return self._generate_virtual_guards(other, box, runtime_box, state)
        //
        // The force-to-virtual path is defined ONLY on NotVirtualStateInfoPtr
        // — a non-virtual *Ref* leaf. NotVirtualStateInfoInt/Float use the
        // base NotVirtualStateInfo._generate_guards, which has no force
        // branch. And `other` must be the virtual-instance class
        // VirtualStateInfo (`Virtual`), never VArrayStateInfo /
        // VStructStateInfo / VArrayStructStateInfo. There is no reverse
        // force: a virtual `self` against a non-virtual `other` falls through
        // to the main match and raises VirtualStatesCantMatch
        // (AbstractVirtualStructStateInfo._generate_guards via
        // _generalization_of_structpart, virtualstate.py).
        if state.force_boxes
            && matches!(incoming_info, VirtualStateInfo::Virtual { .. })
            && matches!(
                expected_info,
                VirtualStateInfo::NonNull
                    | VirtualStateInfo::KnownClass { .. }
                    | VirtualStateInfo::Unknown(Type::Ref)
                    | VirtualStateInfo::Constant(Value::Ref(_))
            )
        {
            // virtualstate.py _generate_virtual_guards: a virtual is
            // never null, so only the type need be checked.
            return match expected_info {
                // :566-568 LEVEL_CONSTANT → cannot unify a constant with a virtual.
                VirtualStateInfo::Constant(_) => Err(VirtualStatesCantMatch::default()),
                // :570-571 LEVEL_KNOWNCLASS → known_class.same_constant(other.known_class).
                VirtualStateInfo::KnownClass { class_ptr } => match incoming_info {
                    VirtualStateInfo::Virtual { known_class, .. }
                        if known_class.as_ref() == Some(class_ptr) =>
                    {
                        Ok(())
                    }
                    _ => Err(VirtualStatesCantMatch::default()),
                },
                // LEVEL_NONNULL / LEVEL_UNKNOWN(Ref) → pass, no guard.
                _ => Ok(()),
            };
        }
        // `AbstractVirtualStructStateInfo._generate_guards` has no force-box
        // relaxation: a virtual target requires a matching virtual incoming.
        // The converse relaxation belongs to
        // `NotVirtualStateInfoPtr._generate_virtual_guards` and is handled by
        // the `state.force_boxes` branch for a non-virtual target.

        // `NotVirtualStateInfo._generate_guards` rejects virtual incoming
        // values before constant or pointer-specific matching. Keeping that
        // check in `info_type_matches` also ensures the `Constant` match arm is
        // reached only for non-virtual incoming state.
        if !expected_info.is_virtual()
            && let Some(expected_type) = expected_info.info_type()
            && !info_type_matches(expected_type, incoming_info)
        {
            return Err(VirtualStatesCantMatch::default());
        }

        // virtualstate.py NotVirtualStateInfoPtr._generate_guards:
        // compare the incoming pointer length bound before dispatching on
        // LEVEL_NONNULL / LEVEL_KNOWNCLASS / the base NotVirtual level.
        // An incoming pointer without length information has the default
        // nonnegative length range.
        if let Some(expected_bound) = expected.lenbound.as_ref() {
            let default_incoming_bound;
            let incoming_bound = match incoming.lenbound.as_ref() {
                Some(bound) => bound,
                None => {
                    default_incoming_bound = IntBound::nonnegative();
                    &default_incoming_bound
                }
            };
            assert!(expected_bound.are_knownbits_implied());
            if !incoming_bound.is_within_range(expected_bound.lower, expected_bound.upper) {
                state.bad.insert(expected as *const _);
                state.bad.insert(incoming as *const _);
                if crate::log_jtet_enabled() {
                    eprintln!(
                        "[jit][jte] virtualstate length-bound mismatch arg_idx={arg_idx} \
                         expected={expected_bound:?} incoming={incoming_bound:?}"
                    );
                }
                return Err(VirtualStatesCantMatch::new("length bound does not match"));
            }
        }

        // virtualstate.py try/except VirtualStatesCantMatch wrapper.
        // If `_generate_guards` raises, RPython marks `self` and `other`
        // in `state.bad` so debug_print can flag the failing nodes:
        //
        //     try:
        //         self._generate_guards(other, op, runtime_op, state)
        //     except VirtualStatesCantMatch as e:
        //         state.bad[self] = state.bad[other] = None
        //         ...
        //         raise e
        //
        // Pyre returns `Err(VirtualStatesCantMatch::default())` instead of
        // raising; `VirtualStateInfo::generate_guards` populates `state.bad`
        // before propagating the error. Per-arm
        // pointer-identity keys (`expected as *const _`) match Python
        // object-identity dict keying.
        let result = match (expected_info, incoming_info) {
            // virtualstate.py:387-389: Unknown target accepts anything of
            // the same type (the isinstance check above already enforced
            // the type agreement).
            (VirtualStateInfo::Unknown(_), _) => Ok(()),

            // ── Constant target ── (virtualstate.py:396-405)
            //
            //     if other.level == LEVEL_CONSTANT:
            //         if self.constbox.same_constant(other.constbox):
            //             return
            //         raise VirtualStatesCantMatch("different constants")
            //
            // virtualstate.py settles a LEVEL_CONSTANT incoming
            // STATICALLY, ahead of the runtime-box test at :400-405: two
            // different constants can never match, whatever value the runtime
            // box happens to carry at the jump point. Falling through to the
            // runtime test instead would emit a GUARD_VALUE against a target
            // constant the incoming state already contradicts.
            // `Value`'s PartialEq is the `same_constant` port
            // (`impl PartialEq for Value` in `majit-ir/src/value.rs` —
            // bitwise for floats,
            // history.py:292).
            (VirtualStateInfo::Constant(a), VirtualStateInfo::Constant(b)) if a == b => Ok(()),
            // virtualstate.py `raise VirtualStatesCantMatch("different constants")`.
            (VirtualStateInfo::Constant(_), VirtualStateInfo::Constant(_)) => {
                Err(VirtualStatesCantMatch::default())
            }
            (VirtualStateInfo::Constant(val), _) => {
                // virtualstate.py:400-405: emit GUARD_VALUE only when the
                // concrete runtime box already equals the target constant.
                // virtualstate.py: `runtime_box.constbox()` reads `.value`
                // directly on `InputArg*`, so live InputArgs participate —
                // route through `runtime_value_of`.
                if runtime_box
                    .and_then(|runtime_box| state.ctx.runtime_value_of(runtime_box))
                    .is_some_and(|runtime_value| runtime_value == *val)
                {
                    state.extra_guards.push(GuardRequirement::GuardValue {
                        arg_index: arg_idx,
                        box_opref,
                        expected_value: *val,
                    });
                    Ok(())
                } else {
                    Err(VirtualStatesCantMatch::default())
                }
            }

            // ── KnownClass target ── (virtualstate.py _generate_guards_knownclass)
            //
            // Three sub-branches with distinct opcodes per upstream:
            // - LEVEL_UNKNOWN (:600-606) → GUARD_NONNULL_CLASS
            // - LEVEL_NONNULL (:607-613) → GUARD_CLASS
            // - LEVEL_KNOWNCLASS (:614-617) → no guard, identity check
            // - LEVEL_CONSTANT (:618-624) → no guard, static cls_of_box check
            //
            // Every "with runtime guard" branch reads the runtime class
            // via `cpu.cls_of_box(runtime_box)` (:601-602, :608-609,
            // :620-621). Pyre mirrors this with `ctx.cls_of_box`, which
            // walks the OpRef to its `Value::Ref(gcref)` and invokes
            // the plumbed `default_cls_of_box` hook on `gcref.raw()`.
            // The optimizer-tracked PtrInfo is NOT consulted here —
            // RPython explicitly uses the cpu hook on the concrete
            // runtime box, not `getptrinfo`.
            (
                VirtualStateInfo::KnownClass { class_ptr: c1 },
                VirtualStateInfo::KnownClass { class_ptr: c2 },
            ) if c1 == c2 => Ok(()),
            (VirtualStateInfo::KnownClass { class_ptr }, VirtualStateInfo::Unknown(_)) => {
                // virtualstate.py LEVEL_UNKNOWN branch.
                let Some(rb) = runtime_box else {
                    return Err(VirtualStatesCantMatch::default());
                };
                // virtualstate.py:601 `cpu.cls_of_box(runtime_box)` reads the
                // runtime box's own ref (getref_base), no _forwarded walk.
                let Some(runtime_cls) = state.ctx.runtime_cls_of(rb) else {
                    return Err(VirtualStatesCantMatch::default());
                };
                if runtime_cls != *class_ptr {
                    return Err(VirtualStatesCantMatch::default());
                }
                state
                    .extra_guards
                    .push(GuardRequirement::GuardNonnullClass {
                        arg_index: arg_idx,
                        box_opref,
                        expected_class: *class_ptr,
                    });
                Ok(())
            }
            (VirtualStateInfo::KnownClass { class_ptr }, VirtualStateInfo::NonNull) => {
                // virtualstate.py LEVEL_NONNULL branch.
                let Some(rb) = runtime_box else {
                    return Err(VirtualStatesCantMatch::default());
                };
                // virtualstate.py:608 `cpu.cls_of_box(runtime_box)` reads the
                // runtime box's own ref (getref_base), no _forwarded walk.
                let Some(runtime_cls) = state.ctx.runtime_cls_of(rb) else {
                    return Err(VirtualStatesCantMatch::default());
                };
                if runtime_cls != *class_ptr {
                    return Err(VirtualStatesCantMatch::default());
                }
                state.extra_guards.push(GuardRequirement::GuardClass {
                    arg_index: arg_idx,
                    box_opref,
                    expected_class: *class_ptr,
                });
                Ok(())
            }
            (
                VirtualStateInfo::KnownClass { class_ptr },
                VirtualStateInfo::Constant(Value::Ref(r)),
            ) if !r.is_null() => {
                // virtualstate.py LEVEL_CONSTANT branch.
                // Static check only — `cls_of_box(other.constbox)` against
                // self.known_class. No guard emitted; pass-or-fail decides.
                // The runtime_box gate at :601/:608 is absent here because
                // the constant's class is statically known from `r` itself.
                let const_cls = PtrInfo::Constant(*r).get_known_class(state.ctx.cpu.as_ref());
                if const_cls.as_ref() == Some(class_ptr) {
                    Ok(())
                } else {
                    Err(VirtualStatesCantMatch::default())
                }
            }

            // ── NonNull target ── (virtualstate.py _generate_guards_nonnull)
            (VirtualStateInfo::NonNull, VirtualStateInfo::NonNull)
            | (VirtualStateInfo::NonNull, VirtualStateInfo::KnownClass { .. }) => Ok(()),
            (VirtualStateInfo::NonNull, VirtualStateInfo::Constant(Value::Ref(r))) => {
                if !r.is_null() {
                    Ok(())
                } else {
                    Err(VirtualStatesCantMatch::default())
                }
            }
            (VirtualStateInfo::NonNull, VirtualStateInfo::Unknown(_)) => {
                // virtualstate.py _generate_guards_nonnull, LEVEL_UNKNOWN:
                //   if runtime_box is not None and runtime_box.nonnull():
                //       extra_guards.append(ResOperation(rop.GUARD_NONNULL, [box]))
                //   else:
                //       raise VirtualStatesCantMatch("other not known to be nonnull")
                // `runtime_box.nonnull()` reads the box's own ref
                // (getref_base()), so gate on the runtime value being a
                // non-null ref — not on mere OpRef presence. A null field
                // read wrapped as ConstRef(NULL) must NOT emit GUARD_NONNULL.
                if runtime_box.is_some_and(|rb| state.ctx.runtime_nonnull(rb)) {
                    state.extra_guards.push(GuardRequirement::GuardNonnull {
                        arg_index: arg_idx,
                        box_opref,
                    });
                    Ok(())
                } else {
                    Err(VirtualStatesCantMatch::default())
                }
            }
            // NonNull accepts any virtual (virtual is always nonnull).
            (VirtualStateInfo::NonNull, other) if other.is_virtual() => Ok(()),

            // ── IntBounded target ── (virtualstate.py _generate_guards_unkown)
            //
            //   other_intbound = other.intbound        # other is an int leaf
            //   if self.intbound is None: return
            //   if other_intbound.is_within_range(self.lower, self.upper): return
            //   if runtime_box is not None and self.intbound.contains(runtime_box.getint()):
            //       self.intbound.make_guards(box, extra_guards, optimizer); return
            //   raise VirtualStatesCantMatch
            //
            // `self.intbound` is always present here (IntBounded wraps it); the
            // upstream `is None` always-match case maps to the `Unknown(Int)`
            // leaf, handled by the `(Unknown(_), _)` arm above. The incoming int
            // leaf supplies `other_intbound`: IntBounded carries it directly, a
            // Constant(Int) contributes the singleton `[v, v]`, and an
            // `Unknown(Int)` leaf has no static bound (treated as
            // not-statically-within, deferring to the runtime check). The type
            // gate (`info_type_matches`) above guarantees `incoming` is one of
            // these three for an IntBounded target.
            (VirtualStateInfo::IntBounded(b1), incoming_int) => {
                let statically_within = match incoming_int {
                    VirtualStateInfo::IntBounded(b2) => b2.is_within_range(b1.lower, b1.upper),
                    VirtualStateInfo::Constant(Value::Int(v)) => b1.contains(*v),
                    // Unknown(Int): no static bound → fall to the runtime check.
                    _ => false,
                };
                if statically_within {
                    Ok(())
                } else {
                    // virtualstate.py:493-498 runtime fallback: emit
                    // `IntBound.make_guards` when the concrete runtime int falls
                    // within self's bound, else VirtualStatesCantMatch.
                    // `runtime_box.getint()` reads the box's own _resint (no
                    // _forwarded walk); gate on the runtime int actually falling
                    // within bounds — not on mere OpRef presence. The emitted op
                    // stream is identical to upstream: `GuardBounds.to_ops` calls
                    // `IntBound::make_guards` (intutils.py) verbatim, so
                    // the same INT_GE/INT_LE/INT_AND + GUARD_TRUE/GUARD_VALUE
                    // sequence is produced. Only materialization is deferred:
                    // `generate_guards` may reject a later position and discard
                    // the whole `extra_guards` list (loop at :1388), and pyre's
                    // const creation / OpRef-position allocation are
                    // non-rollbackable global side-effects (unlike RPython's
                    // throwaway ResOperation objects). Recording a
                    // `GuardRequirement` intent here and materializing it in
                    // `to_ops` only after a successful match keeps the const pool
                    // untouched on a rejected match.
                    let within = runtime_box
                        .and_then(|rb| state.ctx.runtime_value_of(rb))
                        .and_then(|v| match v {
                            Value::Int(i) => Some(i),
                            _ => None,
                        })
                        .is_some_and(|i| b1.contains(i));
                    if within {
                        state.extra_guards.push(GuardRequirement::GuardBounds {
                            arg_index: arg_idx,
                            box_opref,
                            bounds: b1.clone(),
                        });
                        Ok(())
                    } else {
                        Err(VirtualStatesCantMatch::default())
                    }
                }
            }

            // ── Virtual targets ──
            // virtualstate.py AbstractVirtualStructStateInfo._generate_guards
            // virtualstate.py VirtualStateInfo._generalization_of_structpart
            //
            // Structural prelude (descr / known_class / fielddescrs
            // length / per-position fielddescrs identity) then recurse
            // positionally over `fieldstate`. RPython at :155-160
            // strictly compares `fielddescrs[i] is self.fielddescrs[i]`
            // — same order, same identity — and uses
            // `fielddescrs[i].get_index()` to look up `_fields`. pyre
            // mirrors this via `field_descrs` (parent-local order) +
            // positional pairing.
            (
                VirtualStateInfo::Virtual {
                    known_class: ekc,
                    fields: ef,
                    field_descrs: efd,
                    ..
                },
                VirtualStateInfo::Virtual {
                    known_class: ikc,
                    fields: if_,
                    field_descrs: ifd,
                    ..
                },
            ) => {
                // virtualstate.py VirtualStateInfo._generalization_of_structpart
                // checks `known_class.same_constant` only. SizeDescr Arc
                // identity is not part of that check: two `new_with_vtable`
                // results of one class do not share one descr allocation.
                // known_class.same_constant(other.known_class) is True only for
                // two equal ConstInts; an absent (None) class on either side
                // rejects, mirroring same_constant's isinstance gate.
                match (ekc, ikc) {
                    (Some(c1), Some(c2)) if c1 == c2 => {}
                    _ => return Err(VirtualStatesCantMatch::default()),
                }
                // virtualstate.py:149-151: opinfo = getptrinfo(box) +
                // assert opinfo.is_virtual() AND isinstance(opinfo,
                // AbstractStructPtrInfo). pyre's Virtual variant carries
                // the per-virtual-field OpRef vector at `info.fields`.
                let parent_fields =
                    Self::peek_parent_field_oprefs(state.ctx, box_opref, runtime_box, |info| {
                        match info {
                            PtrInfo::Virtual(v) => {
                                Some(v.fields.iter().map(|(i, b)| (*i, b.to_opref())).collect())
                            }
                            _ => None,
                        }
                    });
                Self::generate_guards_recurse_positional_fields(
                    arg_idx,
                    efd,
                    ifd,
                    ef,
                    if_,
                    parent_fields.as_deref(),
                    runtime_box,
                    None,
                    state,
                )
            }

            // virtualstate.py VStructStateInfo — same shape as
            // Virtual but keyed by typedescr instead of vtable class.
            (
                VirtualStateInfo::VStruct {
                    descr: ed,
                    fields: ef,
                    field_descrs: efd,
                },
                VirtualStateInfo::VStruct {
                    descr: id,
                    fields: if_,
                    field_descrs: ifd,
                },
            ) => {
                // virtualstate.py VStructStateInfo._generalization_of_structpart:
                // `self.typedescr is other.typedescr`. Object identity
                // (Arc::as_ptr).
                if descr_identity(ed) != descr_identity(id) {
                    return Err(VirtualStatesCantMatch::default());
                }
                // virtualstate.py (VStructStateInfo path):
                // `opinfo = getptrinfo(box)` + isinstance check, then read
                // `opinfo._fields[descr.get_index()]` per field. pyre's
                // VirtualStruct stores the per-virtual-field OpRef vector
                // identically to Virtual.
                let parent_fields =
                    Self::peek_parent_field_oprefs(state.ctx, box_opref, runtime_box, |info| {
                        match info {
                            PtrInfo::VirtualStruct(s) => {
                                Some(s.fields.iter().map(|(i, b)| (*i, b.to_opref())).collect())
                            }
                            _ => None,
                        }
                    });
                Self::generate_guards_recurse_positional_fields(
                    arg_idx,
                    efd,
                    ifd,
                    ef,
                    if_,
                    parent_fields.as_deref(),
                    runtime_box,
                    None,
                    state,
                )
            }

            // virtualstate.py VArrayStateInfo._generate_guards —
            // arraydescr identity + length, then recurse over items.
            (
                VirtualStateInfo::VArray {
                    descr: ed,
                    items: ei,
                    ..
                },
                VirtualStateInfo::VArray {
                    descr: id,
                    items: ii,
                    ..
                },
            ) => {
                // virtualstate.py:244 `self.arraydescr is not other.arraydescr`.
                // Object identity (Arc::as_ptr).
                if descr_identity(ed) != descr_identity(id) || ei.len() != ii.len() {
                    return Err(VirtualStatesCantMatch::default());
                }
                // virtualstate.py:251-256: `opinfo = getptrinfo(box)`,
                // `fieldbox = opinfo._items[i]`, `fieldbox_runtime =
                // state.get_runtime_item(runtime_box, arraydescr, i)`.
                // pyre's VirtualArray stores `_items` as a dense `Vec<OpRef>`.
                let parent_items: Option<Vec<OpRef>> = if runtime_box.is_some() {
                    state
                        .ctx
                        .get_box_replacement_operand_opt(box_opref)
                        .and_then(|b| state.ctx.getptrinfo(&b))
                        .and_then(|info| match info {
                            PtrInfo::VirtualArray(a) => {
                                Some(a.items.iter().map(|b| b.to_opref()).collect())
                            }
                            _ => None,
                        })
                } else {
                    None
                };
                for (i, (ec, ic)) in ei.iter().zip(ii.iter()).enumerate() {
                    // virtualstate.py:257-260: expected `None` skips the
                    // slot; expected state against incoming `None` cannot
                    // match. Only present pairs recurse.
                    let (ec, ic) = match (ec, ic) {
                        (None, _) => continue,
                        (Some(_), None) => return Err(VirtualStatesCantMatch::default()),
                        (Some(ec), Some(ic)) => (ec, ic),
                    };
                    // virtualstate.py:251-256 + :72-76: `fieldbox` and
                    // `fieldbox_runtime` are both `None` when the parent's
                    // `_items[i]` slot is unset. pyre's VirtualArrayInfo
                    // initialises items as `vec![OpRef::NONE; length]`
                    // (`ptr_info.rs`'s `PtrInfo::virtual_array`), so a NONE
                    // slot is the parity-equivalent
                    // of RPython's `_items[i] is None` and must propagate
                    // as `runtime_box=None`. Threading `Some(OpRef::NONE)`
                    // instead would let downstream NonNull / IntBounded
                    // arms emit a guard whose `box_opref` falls back to
                    // `args[arg_index]` (in `to_ops`), promoting "no
                    // guard for this item" into "guard on the parent".
                    let inner = parent_items
                        .as_ref()
                        .and_then(|items| items.get(i).copied())
                        .filter(|opref| !opref.is_none());
                    // virtualstate.py:253-256: `fieldbox_runtime =
                    // state.get_runtime_item(runtime_box, arraydescr, i)`
                    // when both `runtime_box` and the parent's `_items[i]`
                    // are set. `get_runtime_item` reads at
                    // `array_ptr + base_size + i * itemsize` per
                    // `ArrayDescr.base_size()` / `item_size()`.
                    let recurse_runtime = match (runtime_box, inner) {
                        (Some(rb), Some(_)) => state.ctx.get_runtime_item(rb, ed, i),
                        _ => None,
                    };
                    let recurse_box = inner.unwrap_or(OpRef::NONE);
                    Self::generate_guards_for_entry_recursive(
                        arg_idx,
                        ec,
                        ic,
                        recurse_box,
                        recurse_runtime,
                        state,
                    )?;
                }
                Ok(())
            }

            // virtualstate.py VArrayStructStateInfo._generate_guards —
            // arraydescr + length + fielddescrs identity then recurse over
            // (element × field) cells.
            (
                VirtualStateInfo::VArrayStruct {
                    descr: ed,
                    fielddescrs: efd,
                    element_fields: eef,
                },
                VirtualStateInfo::VArrayStruct {
                    descr: id,
                    fielddescrs: ifd,
                    element_fields: ief,
                },
            ) => {
                // virtualstate.py:295 `self.arraydescr is not other.arraydescr`.
                // Object identity (Arc::as_ptr).
                if descr_identity(ed) != descr_identity(id) || eef.len() != ief.len() {
                    return Err(VirtualStatesCantMatch::default());
                }
                if efd.len() != ifd.len() {
                    return Err(VirtualStatesCantMatch::default());
                }
                for (a, b) in efd.iter().zip(ifd.iter()) {
                    // virtualstate.py VArrayStructStateInfo:
                    // `if descr is not other.fielddescrs[j]: raise
                    // VirtualStatesCantMatch`. Object identity, not
                    // numeric index. pyre's Arc::as_ptr-keyed
                    // `descr_identity` (`majit-ir/src/descr.rs`) is the
                    // parity port —
                    // DescrRef::index() is `u32::MAX` for cache-route
                    // minted field descrs (the `Descr::index` default in
                    // `majit-ir/src/descr.rs`), so an index
                    // compare would collapse distinct descrs.
                    if descr_identity(a) != descr_identity(b) {
                        return Err(VirtualStatesCantMatch::default());
                    }
                }
                for (elem_idx, (e_fields, i_fields)) in eef.iter().zip(ief.iter()).enumerate() {
                    // virtualstate.py VArrayStructStateInfo:
                    // efd and ifd are equal per the identity check above,
                    // so each element struct shares the same fielddescrs.
                    //
                    // virtualstate.py `opinfo = getptrinfo(box)` +
                    // :321-324: `fieldbox = opinfo._items[index]`,
                    // `fieldbox_runtime = state.get_runtime_interiorfield(
                    // runtime_box, descr, i)`. pyre's VirtualArrayStruct
                    // stores `element_fields[elem_idx]` as a per-element
                    // `(descr_idx, OpRef)` vector mirroring `_items`.
                    let parent_fields =
                        Self::peek_parent_field_oprefs(state.ctx, box_opref, runtime_box, |info| {
                            match info {
                                PtrInfo::VirtualArrayStruct(a) => {
                                    a.element_fields.get(elem_idx).map(|row| {
                                        row.iter().map(|(i, b)| (*i, b.to_opref())).collect()
                                    })
                                }
                                _ => None,
                            }
                        });
                    // virtualstate.py:316-320: the positional helper's
                    // missing-field cases are the direct `None` semantics:
                    // expected absent skips, incoming absent against an
                    // expected state rejects.
                    let expected_present: Vec<(u32, Rc<VirtualStateInfoNode>)> = e_fields
                        .iter()
                        .filter_map(|(idx, child)| {
                            child.as_ref().map(|child| (*idx, Rc::clone(child)))
                        })
                        .collect();
                    let incoming_present: Vec<(u32, Rc<VirtualStateInfoNode>)> = i_fields
                        .iter()
                        .filter_map(|(idx, child)| {
                            child.as_ref().map(|child| (*idx, Rc::clone(child)))
                        })
                        .collect();
                    Self::generate_guards_recurse_positional_fields(
                        arg_idx,
                        efd,
                        ifd,
                        &expected_present,
                        &incoming_present,
                        parent_fields.as_deref(),
                        runtime_box,
                        // virtualstate.py:316 `for descr in self.fielddescrs:
                        //     ... fieldbox_runtime = state.get_runtime_interiorfield(
                        //         runtime_box, descr, i)` — the array element
                        // index `i` is the outer loop variable. pyre's
                        // `expected_field_descrs[j]` is an InteriorFieldDescr
                        // built from `ArrayDescr.get_all_interiorfielddescrs()`
                        // (`virtualize.rs`'s `optimize_new_array`), so the
                        // recurse helper
                        // routes through `ctx.get_runtime_interiorfield`.
                        Some(elem_idx),
                        state,
                    )?;
                }
                Ok(())
            }

            // Fundamentally incompatible: VirtualStatesCantMatch.
            _ => Err(VirtualStatesCantMatch::default()),
        };
        if result.is_err() {
            // virtualstate.py:98: `state.bad[self] = state.bad[other] = None`.
            // Pointer-identity keys match Python object-identity dict
            // keys for the per-node "did this node fail" set.
            state.bad.insert(expected as *const _);
            state.bad.insert(incoming as *const _);
        }
        result
    }

    /// virtualstate.py AbstractVirtualStructStateInfo._generate_guards
    /// strict positional parity:
    ///
    /// ```python
    /// if len(self.fielddescrs) != len(other.fielddescrs):
    ///     raise VirtualStatesCantMatch("field descrs don't match")
    /// for i in range(len(self.fielddescrs)):
    ///     if other.fielddescrs[i] is not self.fielddescrs[i]:
    ///         raise VirtualStatesCantMatch("field descrs don't match")
    ///     ...
    ///     self.fieldstate[i].generate_guards(other.fieldstate[i], ...)
    /// ```
    ///
    /// Pyre stores `field_descrs` (parent-local order, line-by-line
    /// `get_all_fielddescrs()`) and a parallel `fields` Vec keyed by
    /// `field_slot_index` (FieldDescr.index_in_parent when a parent
    /// SizeDescr is bound, else Descr.index — `heap.rs`). Iterate
    /// `field_descrs` positionally, match by RPython `is`-identity
    /// (Arc::as_ptr via `descr_identity`), then resolve each
    /// (fielddescr, fieldstate) pair via the parent-local slot index.
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython metainterpreter routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and frame ownership"
    )]
    fn generate_guards_recurse_positional_fields(
        arg_idx: usize,
        expected_field_descrs: &[DescrRef],
        incoming_field_descrs: &[DescrRef],
        expected_fields: &[(u32, Rc<VirtualStateInfoNode>)],
        incoming_fields: &[(u32, Rc<VirtualStateInfoNode>)],
        parent_field_oprefs: Option<&[(u32, OpRef)]>,
        parent_runtime_box: Option<OpRef>,
        // virtualstate.py VArrayStructStateInfo path: when
        // `Some(i)`, each `expected_field_descrs[j]` is an
        // `InteriorFieldDescr` and the concrete heap read goes through
        // `ctx.get_runtime_interiorfield(runtime_box, descr, i)`
        // (virtualstate.py). When `None`, the Virtual / VStruct
        // path applies: each descr is a `FieldDescr` and the read goes
        // through `ctx.get_runtime_field(runtime_box, descr)`
        // (virtualstate.py).
        element_idx: Option<usize>,
        state: &mut GenerateGuardState,
    ) -> Result<(), VirtualStatesCantMatch> {
        // virtualstate.py:155: len check, raises "field descrs don't match".
        if expected_field_descrs.len() != incoming_field_descrs.len() {
            return Err(VirtualStatesCantMatch::default());
        }
        for i in 0..expected_field_descrs.len() {
            // virtualstate.py: `other.fielddescrs[i] is not self.fielddescrs[i]`.
            // RPython uses Python object identity — pyre's port is
            // `descr_identity` (`majit-ir/src/descr.rs`, Arc::as_ptr).
            // DescrRef::index()
            // is the dense u32 GC tid; for cache-route minted field
            // descrs that value is `u32::MAX` (the `Descr::index`
            // default), so an
            // index-based check would collapse distinct descrs together
            // and admit cross-struct false matches.
            if descr_identity(&expected_field_descrs[i])
                != descr_identity(&incoming_field_descrs[i])
            {
                return Err(VirtualStatesCantMatch::default());
            }
            // virtualstate.py: `opinfo._fields[self.fielddescrs[i].get_index()]`.
            // `get_index()` is the parent-local field slot index, not
            // the global Descr.index(); pyre's `info.fields` and
            // `element_fields[i]` are populated via `field_slot_index`
            // (`heap.rs`) / `parent_list_slot` (`virtualize.rs`). The
            // mint on `index_in_parent` can predate a later sibling
            // (`args_w` 1 → 2), so look the field up on the completed
            // parent list. Fall back to `Descr::index()` for descrs
            // without a parent.
            let descr_idx = expected_field_descrs[i]
                .as_field_descr()
                .filter(|fd| fd.get_parent_descr().is_some())
                .map(crate::optimizeopt::virtualize::parent_list_slot)
                .unwrap_or_else(|| expected_field_descrs[i].index());
            let expected_child = expected_fields
                .iter()
                .find(|(idx, _)| *idx == descr_idx)
                .map(|(_, v)| v);
            let incoming_child = incoming_fields
                .iter()
                .find(|(idx, _)| *idx == descr_idx)
                .map(|(_, v)| v);
            // virtualstate.py:161-167: when both the parent's `runtime_box`
            // and its `opinfo._fields[descr.get_index()]` are available,
            // thread the inner field's `fieldbox` AND `fieldbox_runtime`
            // into the recursion. The `fieldbox` (compile-time abstract
            // value) comes from `opinfo._fields[idx]`; `fieldbox_runtime`
            // is a *separate* concrete heap read via
            // `state.get_runtime_field(runtime_box, descr)`
            // (virtualstate.py) — calls `cpu.bh_getfield_gc_*` on
            // the parent's runtime pointer.
            //
            // virtualstate.py + :161-167: when the parent's
            // `opinfo._fields[descr.get_index()]` is `None`, RPython
            // passes `fieldbox=None, fieldbox_runtime=None` and any
            // downstream guard becomes a no-op. pyre's NONE-placeholder
            // slot (`ptr_info.rs`'s `PtrInfo::virtual_array`) carries the
            // same "unset" meaning and
            // must be filtered out before being passed as a runtime
            // sentinel — otherwise `to_ops`'s `args[arg_index]` fallback would
            // promote a missing-field guard onto the top-level arg.
            let inner_box_opref = parent_field_oprefs
                .and_then(|f| f.iter().find(|(idx, _)| *idx == descr_idx))
                .map(|(_, opref)| *opref)
                .filter(|opref| !opref.is_none());
            // virtualstate.py:163-166 / :321-322: only call the
            // concrete read when both `runtime_box` and `fieldbox`
            // (the optimizer-tracked inner OpRef) are present.
            // Dispatch on `element_idx`: VArrayStruct (Some(elem))
            // uses `get_runtime_interiorfield(rb, descr, elem)`,
            // Virtual/VStruct (None) uses `get_runtime_field(rb, descr)`.
            let recurse_runtime = match (parent_runtime_box, inner_box_opref) {
                (Some(prb), Some(_)) => match element_idx {
                    Some(elem) => {
                        state
                            .ctx
                            .get_runtime_interiorfield(prb, &expected_field_descrs[i], elem)
                    }
                    None => state.ctx.get_runtime_field(prb, &expected_field_descrs[i]),
                },
                _ => None,
            };
            let recurse_box = inner_box_opref.unwrap_or(OpRef::NONE);
            // virtualstate.py:171-173: both fieldstate[i] None → skip;
            // expected None vs incoming Some → still skip (RPython only
            // recurses when expected is set); expected Some vs incoming
            // None → VirtualStatesCantMatch.
            match (expected_child, incoming_child) {
                (None, _) => continue,
                (Some(_), None) => return Err(VirtualStatesCantMatch::default()),
                (Some(e), Some(i)) => {
                    Self::generate_guards_for_entry_recursive(
                        arg_idx,
                        e,
                        i,
                        recurse_box,
                        recurse_runtime,
                        state,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// virtualstate.py:148-152 / :252-253 / :309-312 `opinfo = getptrinfo(box)`
    /// when `runtime_box is not None`. Returns a clone of the relevant
    /// inner-field OpRef vector (Virtual/VStruct/VArrayStruct[elem_idx])
    /// or `None` when the parent has no runtime_box, no recorded PtrInfo,
    /// or a PtrInfo of an incompatible variant.
    fn peek_parent_field_oprefs(
        ctx: &OptContext,
        parent_box_opref: OpRef,
        parent_runtime_box: Option<OpRef>,
        extract: impl FnOnce(&PtrInfo) -> Option<Vec<(u32, OpRef)>>,
    ) -> Option<Vec<(u32, OpRef)>> {
        parent_runtime_box?;
        let b = ctx.get_box_replacement_operand_opt(parent_box_opref)?;
        let info = ctx.getptrinfo(&b)?;
        extract(&info)
    }

    /// virtualstate.py: debug_print(hdr, bad, metainterp_sd)
    /// Format the virtual state for debugging.
    pub fn debug_print(&self) -> String {
        let mut out = String::new();
        for (i, info) in self.state.iter().enumerate() {
            let kind = match &info.info {
                VirtualStateInfo::Constant(_) => "Constant",
                VirtualStateInfo::Virtual { .. } => "Virtual",
                VirtualStateInfo::VArray { .. } => "VArray",
                VirtualStateInfo::VStruct { .. } => "VStruct",
                VirtualStateInfo::VArrayStruct { .. } => "VArrayStruct",
                VirtualStateInfo::KnownClass { .. } => "KnownClass",
                VirtualStateInfo::NonNull => "NonNull",
                VirtualStateInfo::IntBounded(_) => "IntBounded",
                VirtualStateInfo::Unknown(_) => "Unknown",
            };
            out.push_str(&format!("  [{i}] {kind}\n"));
        }
        out
    }
}

impl VirtualState {
    /// virtualstate.py: force_boxes(optimizer) — force all virtual entries
    /// to be materialized. After calling this, all entries become non-virtual
    /// (Constant, KnownClass, NonNull, IntBounded, or Unknown).
    ///
    /// Returns the number of virtuals that were forced.
    pub fn force_boxes(&mut self) -> usize {
        let mut count = 0;
        for slot in &mut self.state {
            if slot.is_virtual() {
                // virtualstate.py: forced virtuals become NonNull
                // (they were allocated, so they're always non-null).
                let new_kind = match &slot.info {
                    VirtualStateInfo::Virtual { known_class, .. } => {
                        if let Some(cls) = *known_class {
                            VirtualStateInfo::KnownClass { class_ptr: cls }
                        } else {
                            VirtualStateInfo::NonNull
                        }
                    }
                    _ => VirtualStateInfo::NonNull,
                };
                *slot = VirtualStateInfoNode::new_rc(new_kind);
                count += 1;
            }
        }
        // Forcing breaks any prior Rc sharing, so position cells
        // assigned to the previous instance graph must be re-derived
        // against the rewritten state.
        self.enum_top_level();
        count
    }

    /// Get the lenbound of a virtual array at the given index, if any.
    pub fn getlenbound(&self, index: usize) -> Option<&IntBound> {
        match self.state.get(index).map(|rc| &rc.info) {
            Some(VirtualStateInfo::VArray { lenbound, .. }) => lenbound.as_ref(),
            _ => None,
        }
    }
}

impl Clone for VirtualState {
    /// Share the `AbstractVirtualStateInfo` graph. RPython `VirtualState`
    /// is one object; assignment (`TargetToken.virtual_state = vs`,
    /// `ExportedState.virtual_state = vs`) never copies the instance tree.
    /// `VirtualState.__init__` / `from_shared_rcs` is the only constructor
    /// that walks `enum`; `refresh_from_gc` and `force_boxes` rewrite this
    /// instance in place. A deep copy of the node tree was a compile-time
    /// `IndexMap` + per-node `Rc` on every `token.virtual_state.clone()`.
    fn clone(&self) -> Self {
        VirtualState {
            state: self.state.clone(),
            numnotvirtuals: self.numnotvirtuals,
            info_counter: self.info_counter,
            const_ptr_indexes: self.const_ptr_indexes.clone(),
        }
    }
}

/// A guard that must be emitted to make an incoming state compatible.
///
/// virtualstate.py generate_guards: boxes parameter provides the actual OpRef at each
/// position. `box_opref` is the concrete OpRef used as the guard's first
/// argument; `arg_index` is the position in the state vector.
#[derive(Clone, Debug)]
pub enum GuardRequirement {
    /// Emit GUARD_CLASS on the arg at this index.
    /// virtualstate.py NotVirtualStateInfoPtr._generate_guards_knownclass,
    /// LEVEL_NONNULL branch: `ResOperation(rop.GUARD_CLASS, [box, self.known_class])`.
    GuardClass {
        arg_index: usize,
        box_opref: OpRef,
        /// Immortal vtable address (`ConstInt(vtable)`), not a traced ref.
        expected_class: i64,
    },
    /// Emit GUARD_NONNULL_CLASS on the arg at this index.
    /// virtualstate.py NotVirtualStateInfoPtr._generate_guards_knownclass,
    /// LEVEL_UNKNOWN branch: `ResOperation(rop.GUARD_NONNULL_CLASS, [box, self.known_class])`.
    /// Distinct from `GuardClass` — the LEVEL_UNKNOWN incoming has no
    /// proven non-nullness, so the combined guard is required.
    GuardNonnullClass {
        arg_index: usize,
        box_opref: OpRef,
        /// Immortal vtable address (`ConstInt(vtable)`), not a traced ref.
        expected_class: i64,
    },
    /// Emit GUARD_NONNULL on the arg at this index.
    GuardNonnull { arg_index: usize, box_opref: OpRef },
    /// Emit GUARD_VALUE on the arg at this index.
    GuardValue {
        arg_index: usize,
        box_opref: OpRef,
        expected_value: Value,
    },
    /// Emit integer bounds guards on the arg at this index.
    GuardBounds {
        arg_index: usize,
        box_opref: OpRef,
        bounds: IntBound,
    },
}

impl GuardRequirement {
    /// Convert this guard requirement into the concrete Op stream that
    /// upstream `extra_guards` would have appended. RPython creates
    /// ConstInt/ConstPtr inline in ResOperation args
    /// (`NotVirtualStateInfo._generate_guards`,
    /// `NotVirtualStateInfoPtr._generate_guards_knownclass`,
    /// `IntBound.make_guards`); pyre builds those constant args with
    /// `Operand::const_from_value`.
    ///
    /// Most variants emit a single guard; `GuardBounds` expands to the
    /// int_ge/int_le/int_and pairs of `IntBound::make_guards`
    /// (intutils.py). The caller (`unroll.rs`'s
    /// `jump_to_existing_trace_impl`) iterates the
    /// returned `Vec` and applies `rd_resume_position` /
    /// `ResumeAtPositionDescr` only to entries that pass `is_guard()` —
    /// matching unroll.py `if isinstance(guard, GuardResOp)`. The
    /// interleaved INT_GE / INT_LE / INT_AND producers in this stream
    /// are NOT GuardResOp and therefore skip the resume stamp.
    pub fn to_ops(&self, args: &[OpRef], ctx: &mut OptContext) -> Vec<majit_ir::OpRc> {
        match self {
            GuardRequirement::GuardClass {
                arg_index,
                box_opref,
                expected_class,
            } => {
                let arg = if !box_opref.is_none() {
                    *box_opref
                } else {
                    match args.get(*arg_index) {
                        Some(a) => *a,
                        None => return Vec::new(),
                    }
                };
                // virtualstate.py:610 GUARD_CLASS [box, self.known_class].
                // The class operand is a ConstInt vtable address:
                // backend/model.py `cls_of_box()` returns
                // `ConstInt(ptr2int(obj.typeptr))`, and backend regalloc
                // reads `op.getarg(1).getint()`.
                let arg_b = ctx.materialize_operand_at(arg);
                let class_b =
                    majit_ir::operand::Operand::const_from_value(Value::Int(*expected_class));
                let mut op = Op::new(OpCode::GuardClass, &[arg_b.clone(), class_b.clone()]);
                op.setfailargs(Default::default());
                vec![majit_ir::OpRc::new(op)]
            }
            GuardRequirement::GuardNonnullClass {
                arg_index,
                box_opref,
                expected_class,
            } => {
                let arg = if !box_opref.is_none() {
                    *box_opref
                } else {
                    match args.get(*arg_index) {
                        Some(a) => *a,
                        None => return Vec::new(),
                    }
                };
                // virtualstate.py:603 GUARD_NONNULL_CLASS [box, self.known_class].
                // The class operand is the same ConstInt vtable address used
                // by GUARD_CLASS.
                let arg_b = ctx.materialize_operand_at(arg);
                let class_b =
                    majit_ir::operand::Operand::const_from_value(Value::Int(*expected_class));
                let mut op = Op::new(OpCode::GuardNonnullClass, &[arg_b.clone(), class_b.clone()]);
                op.setfailargs(Default::default());
                vec![majit_ir::OpRc::new(op)]
            }
            GuardRequirement::GuardNonnull {
                arg_index,
                box_opref,
            } => {
                let arg = if !box_opref.is_none() {
                    *box_opref
                } else {
                    match args.get(*arg_index) {
                        Some(a) => *a,
                        None => return Vec::new(),
                    }
                };
                let mut op = Op::new(OpCode::GuardNonnull, &[ctx.materialize_operand_at(arg)]);
                op.setfailargs(Default::default());
                vec![majit_ir::OpRc::new(op)]
            }
            GuardRequirement::GuardValue {
                arg_index,
                box_opref,
                expected_value,
            } => {
                let arg = if !box_opref.is_none() {
                    *box_opref
                } else {
                    match args.get(*arg_index) {
                        Some(a) => *a,
                        None => return Vec::new(),
                    }
                };
                // virtualstate.py `NotVirtualStateInfo._generate_guards`:
                // ResOperation(GUARD_VALUE, [box, self.constbox]). Preserve
                // the Const object's type: ConstPtr must not be represented
                // as ConstInt. history.py has no `ConstVoid` class —
                // `LEVEL_CONSTANT` cannot be Void (mirrors the unreachable!
                // in `_generate_guards` LEVEL_CONSTANT arm above).
                let arg_b = ctx.materialize_operand_at(arg);
                let val_b = match expected_value {
                    Value::Void => unreachable!("LEVEL_CONSTANT cannot be Void"),
                    v => majit_ir::operand::Operand::const_from_value(*v),
                };
                let mut op = Op::new(OpCode::GuardValue, &[arg_b.clone(), val_b.clone()]);
                op.setfailargs(Default::default());
                vec![majit_ir::OpRc::new(op)]
            }
            GuardRequirement::GuardBounds {
                arg_index,
                box_opref,
                bounds,
            } => {
                let arg = if !box_opref.is_none() {
                    *box_opref
                } else {
                    match args.get(*arg_index) {
                        Some(a) => *a,
                        None => return Vec::new(),
                    }
                };
                // intutils.py IntBound.make_guards parity:
                // upstream appends INT_GE/INT_LE/INT_AND followed by
                // GUARD_TRUE/GUARD_VALUE pairs into `extra_guards`. Each
                // GUARD_* receives the producer `ResOperation` as its
                // first arg via Python-object identity (intutils.py);
                // pyre passes `&mut OptContext` directly so a fresh Int
                // OpRef is installed on the producer's `pos` before the
                // consumer guard captures the args.
                let mut emitted = Vec::new();
                let arg = ctx.materialize_operand_at(arg);
                bounds.make_guards(&arg, &mut emitted, ctx);
                // Tag GUARD_TRUE / GUARD_VALUE with empty fail_args; the
                // non-guard INT_GE / INT_LE / INT_AND producers keep the
                // default. The caller (`unroll.rs`'s
                // `jump_to_existing_trace_impl`) gates the
                // rd_resume_position / descr stamp on `is_guard()` per
                // unroll.py `isinstance(guard, GuardResOp)`.
                for op in &emitted {
                    if matches!(op.opcode, OpCode::GuardTrue | OpCode::GuardValue) {
                        op.setfailargs(Default::default());
                    }
                }
                emitted
            }
        }
    }
}

/// Export the abstract state of loop-carried values.
///
/// Given the current optimization context and PtrInfo table (from the virtualize pass),
/// create a VirtualState snapshot for the given OpRefs (typically the Jump args).
///
/// virtualstate.py: VirtualStateConstructor.make_virtual_state()
pub(crate) fn export_state(oprefs: &[OpRef], ctx: &OptContext) -> VirtualState {
    // virtualstate.py VirtualStateConstructor.create_state caches by
    // resolved box: if two different oprefs (or two field references) resolve
    // to the same target, they share the SAME `VirtualStateInfo` Python
    // object — and consequently the same `position` /
    // `position_in_notvirtuals`. The Rust port mirrors this with an
    // `Rc<VirtualStateInfoNode>` cache shared across the whole export, including
    // recursive nested-field calls AND top-level jump args.
    //
    // virtualstate.py `create_state` does `box = get_box_replacement(box)`
    // inside the per-value walk, so we don't pre-resolve here.
    let mut cache = ExportCache::new();
    let state: Vec<Rc<VirtualStateInfoNode>> = oprefs
        .iter()
        .map(|opref| export_single_value(*opref, ctx, &mut cache))
        .collect();
    // virtualstate.py VirtualState.__init__ assigns positions via
    // _enum so subsequent walks dedup shared Rc'd subtrees via
    // `state.position > self.position`.
    VirtualState::from_shared_rcs(state)
}

/// Same walk as [`export_state`], keyed by the already-resolved boxes.
/// RPython `get_virtual_state(end_args)` holds Box objects, not positions.
pub(crate) fn export_state_operands(
    operands: &[majit_ir::operand::Operand],
    ctx: &OptContext,
) -> VirtualState {
    let mut cache = ExportCache::new();
    let state: Vec<Rc<VirtualStateInfoNode>> = operands
        .iter()
        .map(|operand| export_single_operand(operand, ctx, &mut cache))
        .collect();
    VirtualState::from_shared_rcs(state)
}

/// Bookkeeping shared across `export_single_value` recursion: the DAG cache
/// (fully constructed nodes only) plus an `in_progress` set used to detect
/// back-edges. Splitting the two prevents the previous "insert Unknown stub
/// then overwrite" pattern from leaking the stub to in-flight recursive
/// callers.
pub(crate) struct ExportCache {
    // Keyed by operand identity (`Rc::ptr_eq`, const by value) — the resolved
    // box `get_box_replacement(opref)`. virtualstate.py create_state keys `self.info`
    // (a plain identity dict) by `get_box_replacement(box)` and relies on "same
    // resolved box → same key" for finished / in_progress dedup. A bound
    // producer resolves to its one canonical `Rc` (memoized `from_bound_op`),
    // and value-equal constants merge via the operand's `same_constant` (matching
    // RPython's shared-instance dedup; the benign over-merge is on not-virtual
    // const nodes that emit no fail-arg box).
    //
    // bind-at-alloc invariant: every position reaching export resolves to a
    // bound box (debug-asserted in `export_single_value`). Production forced
    // end-args and `ProducedShortOp.res` (`PreambleOp::add_op_to_short`
    // `materialize_operand_at`) are bound at creation; an unbound position would
    // mint a fresh operand from the opref per visit and split the cache, so the
    // assert traps that as a bind-at-alloc gap rather than silently
    // mis-deduping. `IndexMap`/`IndexSet` are Vec-backed and compare by `Eq` only
    // (never hash), so no GC pointer is hashed here.
    pub finished: crate::FxIndexMap<majit_ir::operand::Operand, Rc<VirtualStateInfoNode>>,
    pub in_progress: crate::FxIndexSet<majit_ir::operand::Operand>,
}

impl ExportCache {
    pub fn new() -> Self {
        Self {
            finished: crate::FxIndexMap::default(),
            in_progress: crate::FxIndexSet::default(),
        }
    }
}

/// virtualstate.py `VirtualStateConstructor.create_state_or_none`.
/// This is the single export boundary that maps an unwritten virtual slot to
/// absent fieldstate; every present operand follows the ordinary state cache.
fn create_state_or_none(
    operand: &majit_ir::operand::Operand,
    ctx: &OptContext,
    cache: &mut ExportCache,
) -> Option<Rc<VirtualStateInfoNode>> {
    // virtualstate.py `create_state_or_none`: `if box is None: return None`. The
    // absent-slot sentinel is `Operand::None`, not `to_opref() == None`:
    // a bound ResOp whose `pos` was never stamped still has `box.type`
    // (`Op.type_`) and must be exported.
    if operand.is_none() {
        None
    } else {
        let opref = operand.to_opref();
        if opref.is_none() {
            // Bound ResOp whose `pos` was never stamped: keep the
            // operand so `Op.type_` can still pick the not_virtual leaf.
            Some(export_single_operand(operand, ctx, cache))
        } else {
            Some(export_single_value(opref, ctx, cache))
        }
    }
}

/// Export abstract info for a single value, sharing `Rc<VirtualStateInfoNode>`
/// across recursive calls so the resulting tree is a DAG: aliased boxes
/// converge on a single shared `VirtualStateInfo`. virtualstate.py
/// VirtualStateConstructor.create_state.
///
/// **Cycle handling**: RPython does
///
/// ```text
/// result = info.visitor_dispatch_virtual_type(self)
/// self.info[box] = result            # ← cache the empty state
/// info.visitor_walk_recursive(box, self)
/// result.fieldstate = [...]          # ← fill afterwards
/// ```
///
/// so a cycle (`A.f -> B`, `B.f -> A`) closes on the same Python object.
/// Rust's `Rc<VirtualStateInfoNode>` is immutable after construction, and
/// `Rc::new_cyclic`'s `Weak<T>` cannot upgrade during the closure body,
/// so we cannot mirror the "cache empty, then mutate" pattern without
/// switching every consumer to `Rc<RefCell<...>>`. Until that refactor
/// lands, `in_progress` detects the back-edge explicitly and the cycle
/// child returns a fresh `Unknown` Rc instead of aliasing onto a stale
/// stub: distinct nodes never collapse, the parent VirtualStateInfo
/// reflects the genuine acyclic prefix, and downstream code that
/// requires real cycle preservation can detect the dropped edge by
/// checking for `Unknown` in a virtual subtree.
fn export_single_value(
    opref: OpRef,
    ctx: &OptContext,
    cache: &mut ExportCache,
) -> Rc<VirtualStateInfoNode> {
    export_single_operand(&ctx.get_box_replacement_operand(opref), ctx, cache)
}

fn export_single_operand(
    operand: &majit_ir::operand::Operand,
    ctx: &OptContext,
    cache: &mut ExportCache,
) -> Rc<VirtualStateInfoNode> {
    // virtualstate.py:713-716 `box = get_box_replacement(box)` then keyed
    // lookup on `self.info`: resolve the forwarding chain BEFORE the cache
    // lookup so two field references forwarding to the same target collapse
    // onto the same VirtualStateInfo. Key the DAG cache by the resolved box's
    // identity (`Rc::ptr_eq`, const by value) — the bound producer's one
    // canonical `Rc`.
    let box_ = operand.get_box_replacement(false);
    let opref = box_.to_opref();
    // bind-at-alloc invariant (see ExportCache): every position reaching
    // export resolves to a bound box, so `box_` is a stable canonical `Rc`
    // rather than a fresh `from_opref` placeholder that would split the cache.
    debug_assert!(
        !box_.is_none() || opref.is_none(),
        "export_single_value: unbound position {opref:?} reached export — \
         bind-at-alloc invariant violated (every value reaching create_state \
         must be a bound box; virtualstate.py:711-720)"
    );
    // virtualstate.py:714-716: cache hit returns the cached state directly.
    if let Some(cached) = cache.finished.get(&box_) {
        return Rc::clone(cached);
    }
    // Cycle: this opref is currently being exported on the parent stack.
    // Return a fresh Unknown leaf so the back-edge is visibly non-virtual
    // — distinct from any real Unknown elsewhere in the tree because
    // each cycle entry allocates its own Rc.
    //
    // Verified 2026-04-10: this branch fires zero times across all 10
    // benchmarks in pyre/check.py (int_loop, float_loop, fib_loop,
    // fib_recursive, nested_loop, nbody, fannkuch, raise_catch_loop,
    // spectral_norm, inline_helper). The cyclic-virtual-graph regression
    // (pyre returns a fresh `Unknown` leaf for a back-edge instead of
    // closing the cycle on one node, per `export_single_value`'s doc) is
    // therefore latent — no
    // benchmark constructs the necessary self-referential structures.
    if cache.in_progress.contains(&box_) {
        // Fallback to Ref for the cycle leaf: pyre's virtual DAGs only
        // form through ptr fields, so the only reachable cycles are on
        // Ref-typed nodes. Matches `not_virtual(cpu, 'r', None)` in
        // RPython where a cycle child resolves to NotVirtualStateInfoPtr
        // with LEVEL_UNKNOWN.
        return VirtualStateInfoNode::new_rc(VirtualStateInfo::Unknown(Type::Ref));
    }
    let key = box_.clone();
    cache.in_progress.insert(key.clone());

    let info = export_single_value_inner(&box_, ctx, cache);
    // virtualstate.py NotVirtualStateInfoPtr.__init__: retain the
    // widened ArrayPtrInfo / StrPtrInfo length bound on the per-instance
    // pointer leaf. Virtual pointer infos have their own state variants and
    // do not populate NotVirtualStateInfoPtr.lenbound.
    let lenbound = if matches!(
        &info,
        VirtualStateInfo::Constant(Value::Ref(_))
            | VirtualStateInfo::KnownClass { .. }
            | VirtualStateInfo::NonNull
            | VirtualStateInfo::Unknown(Type::Ref)
    ) {
        ctx.peek_ptr_info(&box_)
            .and_then(|mut ptr_info| ptr_info.getlenbound(None))
            .map(|bound| bound.widen())
    } else {
        None
    };
    let rc = VirtualStateInfoNode::new_rc_with_lenbound(info, lenbound);
    cache.in_progress.swap_remove(&key);
    cache.finished.insert(key, Rc::clone(&rc));
    rc
}

fn export_single_value_inner(
    box_: &majit_ir::operand::Operand,
    ctx: &OptContext,
    cache: &mut ExportCache,
) -> VirtualStateInfo {
    let opref = box_.to_opref();
    // virtualstate.py `visit_not_virtual` dispatches via
    // `not_virtual(cpu, value.type, optimizer.getinfo(value))`; when
    // `info.is_constant()` is true the resulting state is LEVEL_CONSTANT
    // (`info.py:not_virtual` builds `NotVirtualStateInfo*` with constant
    // box stored). pyre's `get_constant(opref)` walks `_forwarded` and
    // returns Some exactly when the chain terminates at a Const Box,
    // i.e. when PyPy's `info.is_constant()` is true. Mirror that:
    // export LEVEL_CONSTANT regardless of OpRef namespace.
    if let Some(value) = ctx.get_constant_box(box_) {
        return VirtualStateInfo::Constant(value);
    }

    // operand-routing PtrInfo read (info.py op.get_forwarded()).
    if let Some(info) = ctx.peek_ptr_info(box_) {
        let info_fielddescrs = info.all_fielddescrs_from_descr();
        match info {
            PtrInfo::Virtual(vinfo) => {
                // virtualstate.py `create_state`: fieldboxes go through
                // `create_state_or_none`. An unwritten slot is `None` and
                // stays absent from the sparse field list.
                let fields = vinfo
                    .fields
                    .iter()
                    .filter_map(|(field_idx, field_ref)| {
                        create_state_or_none(field_ref, ctx, cache)
                            .map(|field_state| (*field_idx, field_state))
                    })
                    .collect();
                return VirtualStateInfo::Virtual {
                    descr: vinfo.descr.clone(),
                    known_class: vinfo.known_class,
                    ob_type_descr: vinfo.ob_type_descr.clone(),
                    fields,
                    field_descrs: info_fielddescrs,
                };
            }
            PtrInfo::VirtualArray(vinfo) => {
                // virtualstate.py:724-725: every fieldbox is routed through
                // `create_state_or_none`, preserving unwritten slots.
                let items: Vec<Option<Rc<VirtualStateInfoNode>>> = vinfo
                    .items
                    .iter()
                    .map(|item_ref| create_state_or_none(item_ref, ctx, cache))
                    .collect();
                let len = items.len();
                return VirtualStateInfo::VArray {
                    descr: vinfo.descr.clone(),
                    items,
                    lenbound: Some(IntBound::from_constant(len as i64)),
                };
            }
            PtrInfo::VirtualStruct(vinfo) => {
                // virtualstate.py `create_state`: same `create_state_or_none`
                // walk as Virtual / VArray. Unwritten slots stay absent.
                let fields = vinfo
                    .fields
                    .iter()
                    .filter_map(|(field_idx, field_ref)| {
                        create_state_or_none(field_ref, ctx, cache)
                            .map(|field_state| (*field_idx, field_state))
                    })
                    .collect();
                return VirtualStateInfo::VStruct {
                    descr: vinfo.descr.clone(),
                    fields,
                    field_descrs: info_fielddescrs,
                };
            }
            PtrInfo::VirtualArrayStruct(vinfo) => {
                let element_fields = vinfo
                    .element_fields
                    .iter()
                    .map(|fields| {
                        fields
                            .iter()
                            .map(|(field_idx, field_ref)| {
                                // virtualstate.py:724-725: retain the dense
                                // field slot while making unwritten state absent.
                                let field_state = create_state_or_none(field_ref, ctx, cache);
                                (*field_idx, field_state)
                            })
                            .collect()
                    })
                    .collect();
                return VirtualStateInfo::VArrayStruct {
                    descr: vinfo.descr.clone(),
                    fielddescrs: vinfo.fielddescrs.clone(),
                    element_fields,
                };
            }
            PtrInfo::VirtualRawBuffer(_) | PtrInfo::VirtualRawSlice(_) => {
                // walkvirtual.py: VirtualVisitor.visit_vrawbuffer /
                // visit_vrawslice raise NotImplementedError. RPython's
                // VirtualStateConstructor inherits the abstract base and
                // does not override either method, so a virtual raw buffer
                // or raw slice reaching state export means an earlier pass
                // (force_at_end_of_preamble) failed to materialize it.
                //
                // info.py RawBufferPtrInfo.is_virtual returns True;
                // info.py RawSlicePtrInfo.is_virtual returns True when
                // the parent is virtual. Both should be forced before this
                // boundary. Match upstream by panicking — the alternative
                // (silently exporting an invented virtual-state shape) is
                // the pyre-side divergence flagged by audit.
                panic!(
                    "export_state: virtual raw buffer/slice reached state export — \
                     visit_vrawbuffer/visit_vrawslice has no VirtualStateConstructor \
                     override (walkvirtual.py:20-24); the buffer must be forced \
                     before the loop boundary"
                );
            }
            PtrInfo::NonNull { .. } => {
                return VirtualStateInfo::NonNull;
            }
            PtrInfo::Constant(gcref) => {
                return VirtualStateInfo::Constant(Value::Ref(gcref));
            }
            PtrInfo::Virtualizable(_) => {
                // Virtualizable objects are treated as non-null in virtual state
                return VirtualStateInfo::NonNull;
            }
            PtrInfo::Instance(iinfo) => {
                // info.py InstancePtrInfo(None, class_const) becomes
                // VirtualStateInfo::KnownClass when only the class is
                // known. Otherwise it's an opaque non-null instance.
                if let Some(class_ptr) = iinfo.known_class {
                    return VirtualStateInfo::KnownClass { class_ptr };
                }
                return VirtualStateInfo::NonNull;
            }
            PtrInfo::Struct(_) | PtrInfo::Array(_) => {
                return VirtualStateInfo::NonNull;
            }
            PtrInfo::Str(_) => {
                return VirtualStateInfo::NonNull;
            }
        }
    }

    // virtualstate.py not_virtual(cpu, box.type, info): the subclass
    // is picked by `box.type` which is ALWAYS set on RPython Boxes.
    // pyre's OptContext::opref_type reconstructs it from value_types
    // (seeded from trace_inputargs) / producing-op result_type.
    let tp = ctx
        .opref_type(opref)
        .or_else(|| {
            // A bound ResOp whose `pos` was never stamped has no OpRef
            // variant tag; `Op.type_` is still `opclasses[opnum].type`.
            let t = box_.type_();
            (t != Type::Void).then_some(t)
        })
        .unwrap_or_else(|| {
            // Two different failures reach this line: the absent-operand sentinel
            // is unexpected outside `create_state_or_none`, while a named ref
            // means value_types lacks an entry for a real box. Keep both loud.
            let seen = if box_.is_none() {
                "the absent-operand sentinel reached export_state"
            } else {
                "no type recorded for a ref that does exist"
            };
            panic!(
                "not_virtual: opref_type({opref:?}) found no type — {seen}; \
             RPython box.type is always set (virtualstate.py:360)",
            );
        });
    // virtualstate.py NotVirtualStateInfoInt.__init__: an int leaf's
    // info is `getintbound(op)` (optimizer.py — always an IntBound for a
    // non-constant int), and the constructor widens it (`info.widen_update()`)
    // and stores it as `self.intbound`. Carry that here so the loop/bridge close
    // can emit `IntBound.make_guards` (virtualstate.py): when a peeled
    // loop's exit guard has been const-folded away (because the loop-variant
    // bound proved it redundant), the bound on this leaf is what makes the bridge
    // that re-enters the loop re-emit the bound check — without it the re-entered
    // loop has no exit and runs forever. A widened-unbounded bound carries no
    // information, so collapse it back to the always-match `Unknown` leaf
    // (behaviorally identical to NotVirtualStateInfoInt with an unbounded
    // `intbound`: `_generate_guards_unkown`'s `is_within_range(MININT, MAXINT)`
    // is always true → no guard).
    if tp == Type::Int
        && let Some(widened) = ctx
            .get_box_replacement_operand_opt(opref)
            .and_then(|b| ctx.peek_intbound_box(&b))
            .map(|b| b.widen())
            .filter(|b| !b.is_unbounded())
    {
        return VirtualStateInfo::IntBounded(widened);
    }
    VirtualStateInfo::Unknown(tp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizeopt::info::{VirtualArrayInfo, VirtualStructInfo};
    use majit_ir::operand::Operand;
    use majit_ir::{Descr, FieldDescr, GcRef, Type};
    use std::sync::Arc;

    #[derive(Debug)]
    struct TestDescr(u32);
    impl Descr for TestDescr {
        fn index(&self) -> u32 {
            self.0
        }
    }
    impl FieldDescr for TestDescr {
        fn offset(&self) -> usize {
            self.0 as usize
        }
        fn field_size(&self) -> usize {
            8
        }
        fn field_type(&self) -> Type {
            Type::Int
        }
    }

    fn test_descr(idx: u32) -> DescrRef {
        Arc::new(TestDescr(idx))
    }

    // ── Per-entry structural checks ──
    //
    // virtualstate.py generalization_of drives the per-entry
    // generate_guards with op=None (virtualstate.py), which
    // degenerates to a structural generalization check that emits no
    // guard. Each pair below is wrapped as a one-entry VirtualState so
    // the entry-level discrimination is exercised through that production
    // path rather than a separate compatibility helper.

    /// Wrap a single `VirtualStateInfo` as a one-entry `VirtualState`.
    fn vs1(info: VirtualStateInfo) -> VirtualState {
        VirtualState::new(vec![info])
    }

    fn vs1_with_lenbound(info: VirtualStateInfo, lenbound: Option<IntBound>) -> VirtualState {
        VirtualState::from_shared_rcs(vec![VirtualStateInfoNode::new_rc_with_lenbound(
            info, lenbound,
        )])
    }

    #[test]
    fn test_unknown_type_discrimination() {
        // virtualstate.py NotVirtualStateInfoInt._generate_guards:
        // isinstance(other, NotVirtualStateInfoInt) check enforces type.
        // Unknown(Int) accepts Int-typed incoming only, not Ref-typed NonNull.
        let mut ctx = OptContext::new(128);
        let unknown_int = vs1(VirtualStateInfo::Unknown(Type::Int));
        assert!(
            unknown_int.generalization_of(&vs1(VirtualStateInfo::Unknown(Type::Int)), &mut ctx)
        );
        assert!(
            unknown_int
                .generalization_of(&vs1(VirtualStateInfo::Constant(Value::Int(42))), &mut ctx)
        );
        assert!(!unknown_int.generalization_of(&vs1(VirtualStateInfo::NonNull), &mut ctx));
        assert!(
            !unknown_int.generalization_of(&vs1(VirtualStateInfo::Unknown(Type::Ref)), &mut ctx)
        );

        // Unknown(Ref) accepts Ref-typed incoming: NonNull, KnownClass, etc.
        let unknown_ref = vs1(VirtualStateInfo::Unknown(Type::Ref));
        assert!(unknown_ref.generalization_of(&vs1(VirtualStateInfo::NonNull), &mut ctx));
        assert!(
            unknown_ref.generalization_of(&vs1(VirtualStateInfo::Unknown(Type::Ref)), &mut ctx)
        );
        assert!(
            !unknown_ref.generalization_of(&vs1(VirtualStateInfo::Unknown(Type::Int)), &mut ctx)
        );
    }

    #[test]
    fn test_constant_compatibility() {
        let mut ctx = OptContext::new(128);
        let c1 = vs1(VirtualStateInfo::Constant(Value::Int(42)));
        assert!(c1.generalization_of(&vs1(VirtualStateInfo::Constant(Value::Int(42))), &mut ctx));
        assert!(!c1.generalization_of(&vs1(VirtualStateInfo::Constant(Value::Int(99))), &mut ctx));
        assert!(!c1.generalization_of(&vs1(VirtualStateInfo::Unknown(Type::Int)), &mut ctx));
    }

    #[test]
    fn test_nonnull_compatibility() {
        let mut ctx = OptContext::new(128);
        let nn = vs1(VirtualStateInfo::NonNull);
        assert!(nn.generalization_of(&vs1(VirtualStateInfo::NonNull), &mut ctx));
        assert!(nn.generalization_of(
            &vs1(VirtualStateInfo::KnownClass { class_ptr: 0x100 }),
            &mut ctx
        ));
        assert!(!nn.generalization_of(&vs1(VirtualStateInfo::Unknown(Type::Int)), &mut ctx));
    }

    #[test]
    fn test_pointer_lenbound_is_checked_by_generalization_and_guard_generation() {
        // virtualstate.py NotVirtualStateInfoPtr._generate_guards:
        // a narrower incoming length range is accepted, while a range that
        // escapes the expected bound is rejected before LEVEL_NONNULL dispatch.
        let expected = vs1_with_lenbound(
            VirtualStateInfo::NonNull,
            Some(IntBound::bounded(0, 10).widen()),
        );
        let within = vs1_with_lenbound(
            VirtualStateInfo::NonNull,
            Some(IntBound::bounded(2, 8).widen()),
        );
        let too_wide = vs1_with_lenbound(
            VirtualStateInfo::NonNull,
            Some(IntBound::bounded(0, 20).widen()),
        );
        let no_bound = vs1(VirtualStateInfo::NonNull);
        let nonnegative =
            vs1_with_lenbound(VirtualStateInfo::NonNull, Some(IntBound::nonnegative()));

        let mut ctx = OptContext::new(128);
        assert!(expected.generalization_of(&within, &mut ctx));
        assert!(!expected.generalization_of(&too_wide, &mut ctx));
        assert!(!expected.generalization_of(&no_bound, &mut ctx));
        assert!(nonnegative.generalization_of(&no_bound, &mut ctx));

        let runtime = ctx.make_constant_ref(GcRef(0x100));
        assert!(
            expected
                .generate_guards(&within, &[OpRef::ref_op(10)], &[runtime], &mut ctx, false,)
                .is_ok()
        );
        assert!(
            expected
                .generate_guards(&too_wide, &[OpRef::ref_op(10)], &[runtime], &mut ctx, false,)
                .is_err()
        );

        let cloned = expected.clone();
        assert_eq!(cloned.state[0].lenbound.as_ref().unwrap().upper, 10);
    }

    #[test]
    fn test_known_class_compatibility() {
        let mut ctx = OptContext::new(128);
        let kc1 = vs1(VirtualStateInfo::KnownClass { class_ptr: 0x100 });
        assert!(kc1.generalization_of(
            &vs1(VirtualStateInfo::KnownClass { class_ptr: 0x100 }),
            &mut ctx
        ));
        assert!(!kc1.generalization_of(
            &vs1(VirtualStateInfo::KnownClass { class_ptr: 0x200 }),
            &mut ctx
        ));
        assert!(!kc1.generalization_of(&vs1(VirtualStateInfo::Unknown(Type::Int)), &mut ctx));
    }

    #[test]
    fn test_virtual_array_compatibility() {
        let mut ctx = OptContext::new(128);
        let descr = test_descr(1);
        let a1 = vs1(VirtualStateInfo::VArray {
            descr: descr.clone(),
            items: vec![
                Some(VirtualStateInfoNode::new_rc(VirtualStateInfo::Constant(
                    Value::Int(1),
                ))),
                Some(VirtualStateInfoNode::new_rc(VirtualStateInfo::Unknown(
                    Type::Int,
                ))),
            ],
            lenbound: None,
        });
        let a2 = vs1(VirtualStateInfo::VArray {
            descr: descr.clone(),
            items: vec![
                Some(VirtualStateInfoNode::new_rc(VirtualStateInfo::Constant(
                    Value::Int(1),
                ))),
                Some(VirtualStateInfoNode::new_rc(VirtualStateInfo::Constant(
                    Value::Int(2),
                ))),
            ],
            lenbound: None,
        });
        let a3 = vs1(VirtualStateInfo::VArray {
            descr: descr.clone(),
            items: vec![Some(VirtualStateInfoNode::new_rc(
                VirtualStateInfo::Constant(Value::Int(1)),
            ))],
            lenbound: None,
        });

        assert!(a1.generalization_of(&a2, &mut ctx)); // same length, first matches, second is Unknown
        assert!(!a1.generalization_of(&a3, &mut ctx)); // different length
    }

    #[test]
    fn test_int_bounded_compatibility() {
        let mut ctx = OptContext::new(128);
        let b1 = vs1(VirtualStateInfo::IntBounded(IntBound::bounded(0, 100)));
        // b2 is within b1
        assert!(b1.generalization_of(
            &vs1(VirtualStateInfo::IntBounded(IntBound::bounded(10, 50))),
            &mut ctx
        ));
        // b3 exceeds b1
        assert!(!b1.generalization_of(
            &vs1(VirtualStateInfo::IntBounded(IntBound::bounded(-10, 200))),
            &mut ctx
        ));
        // 42 is within [0, 100]
        assert!(b1.generalization_of(&vs1(VirtualStateInfo::Constant(Value::Int(42))), &mut ctx));
    }

    // ── VirtualState tests ──

    #[test]
    fn test_virtual_state_compatible() {
        let s1 = VirtualState::new(vec![
            VirtualStateInfo::Unknown(Type::Int),
            VirtualStateInfo::NonNull,
        ]);
        let s2 = VirtualState::new(vec![
            VirtualStateInfo::Constant(Value::Int(42)),
            VirtualStateInfo::KnownClass { class_ptr: 0x100 },
        ]);

        let mut ctx = OptContext::new(128);
        assert!(s1.generalization_of(&s2, &mut ctx));
    }

    #[test]
    fn test_virtual_state_generalization_direction_matches_rpython() {
        // RPython: NotVirtualStateInfoPtr(LEVEL_UNKNOWN) generalizes
        // NotVirtualStateInfoPtr(LEVEL_NONNULL) — same type family.
        // Cross-type (Int target, Ref incoming) is always rejected.
        let target = VirtualState::new(vec![VirtualStateInfo::Unknown(Type::Ref)]);
        let incoming = VirtualState::new(vec![VirtualStateInfo::NonNull]);

        let mut ctx = OptContext::new(128);
        assert!(target.generalization_of(&incoming, &mut ctx));
        assert!(!incoming.generalization_of(&target, &mut ctx));

        // Cross-type: Int target does NOT accept Ref incoming
        let int_target = VirtualState::new(vec![VirtualStateInfo::Unknown(Type::Int)]);
        assert!(!int_target.generalization_of(&incoming, &mut ctx));
    }

    #[test]
    fn test_virtual_state_incompatible_length() {
        let s1 = VirtualState::new(vec![VirtualStateInfo::Unknown(Type::Int)]);
        let s2 = VirtualState::new(vec![
            VirtualStateInfo::Unknown(Type::Int),
            VirtualStateInfo::Unknown(Type::Int),
        ]);

        let mut ctx = OptContext::new(128);
        assert!(!s1.generalization_of(&s2, &mut ctx));
    }

    #[test]
    fn test_virtual_state_generate_guards() {
        // KnownClass and NonNull are Ref-typed; incoming must also be Ref.
        // RPython: NotVirtualStateInfoPtr._generate_guards requires
        // isinstance(other, NotVirtualStateInfoPtr).
        //
        // virtualstate.py LEVEL_UNKNOWN branch additionally
        // requires `self.known_class.same_constant(cpu.cls_of_box(runtime_box))`.
        // RPython reads the class directly from the runtime box's
        // `value` attribute via `cpu.cls_of_box(runtime_box)` — no
        // `get_known_class` (optimizer ptrinfo) fallback. Pyre mirrors
        // this by routing the const-pool gcref through the `Cpu` hook
        // installed below.
        let s1 = VirtualState::new(vec![
            VirtualStateInfo::KnownClass { class_ptr: 0x100 },
            VirtualStateInfo::NonNull,
        ]);
        let s2 = VirtualState::new(vec![
            VirtualStateInfo::Unknown(Type::Ref),
            VirtualStateInfo::Unknown(Type::Ref),
        ]);

        let mut ctx = OptContext::new(128);
        // virtualstate.py:601 `cpu.cls_of_box(runtime_box)`. Seed a
        // const-pool runtime_box for slot 0 whose raw payload is the
        // sentinel `0x200`, install a `Cpu::cls_of_box` hook that maps
        // `0x200 -> 0x100`, so the runtime read matches `known_class`.
        // Slot 1 (NonNull) only requires runtime_box.is_some(); reuse
        // the same const-pool ref.
        ctx.cpu = crate::cpu::cpu_from_bh_classof_fn(|raw| if raw == 0x200 { 0x100 } else { 0 });
        let rb0 = ctx.make_constant_ref(GcRef(0x200));
        let rb1 = ctx.make_constant_ref(GcRef(0x300));
        let boxes = vec![OpRef::ref_op(100), OpRef::ref_op(101)];
        let b0 = ctx.materialize_operand_at(boxes[0]);
        ctx.set_ptr_info(
            &b0,
            crate::optimizeopt::info::PtrInfo::known_class(0x100, false),
        );
        let runtime_boxes = vec![rb0, rb1];
        let guards = s1
            .generate_guards(&s2, &boxes, &runtime_boxes, &mut ctx, false)
            .unwrap();
        assert_eq!(guards.len(), 2);
        // Unknown incoming → GUARD_NONNULL_CLASS (:603).
        assert!(matches!(
            &guards[0],
            GuardRequirement::GuardNonnullClass { arg_index: 0, box_opref, .. } if *box_opref == OpRef::ref_op(100)
        ));
        assert!(matches!(
            &guards[1],
            GuardRequirement::GuardNonnull { arg_index: 1, box_opref } if *box_opref == OpRef::ref_op(101)
        ));
    }

    #[test]
    fn test_constant_guard_requires_matching_runtime_box() {
        let expected = VirtualState::new(vec![VirtualStateInfo::Constant(Value::Int(7))]);
        let incoming = VirtualState::new(vec![VirtualStateInfo::Unknown(Type::Int)]);
        let boxes = vec![OpRef::int_op(10)];

        let mut ctx = OptContext::new(128);
        let matching_runtime = ctx.make_constant_int(7);
        let guards = expected
            .generate_guards(&incoming, &boxes, &[matching_runtime], &mut ctx, false)
            .unwrap();
        assert_eq!(guards.len(), 1);

        let mismatching_runtime = ctx.make_constant_int(8);
        assert!(
            expected
                .generate_guards(&incoming, &boxes, &[mismatching_runtime], &mut ctx, false)
                .is_err()
        );
    }

    #[test]
    fn test_differing_constants_reject_before_the_runtime_box_test() {
        // virtualstate.py: a LEVEL_CONSTANT incoming is settled
        // statically — `raise VirtualStatesCantMatch("different constants")` —
        // and never reaches the runtime-box GUARD_VALUE path at :400-405.
        // Without the explicit arm a runtime box that happens to equal the
        // TARGET constant makes pyre emit a GUARD_VALUE where upstream reports
        // CantMatch.
        let expected = VirtualState::new(vec![VirtualStateInfo::Constant(Value::Int(7))]);
        let incoming = VirtualState::new(vec![VirtualStateInfo::Constant(Value::Int(8))]);
        let boxes = vec![OpRef::int_op(10)];

        let mut ctx = OptContext::new(128);
        let runtime = ctx.make_constant_int(7);
        assert!(
            expected
                .generate_guards(&incoming, &boxes, &[runtime], &mut ctx, false)
                .is_err()
        );
    }

    #[test]
    fn test_constant_target_rejects_virtual_incoming() {
        // virtualstate.py:392-394 (Int/Float leaves) / :525-529 (Ptr leaf): a
        // virtual incoming can never satisfy a NotVirtual expected. In pyre the
        // rejection is performed by the `info_type_matches` gate
        // (in `generate_guards_for_entry_recursive`) before the Constant
        // match arm is reached;
        // this test pins that gate so the arm stays unreachable for virtuals.
        let mut ctx = OptContext::new(128);
        let descr = test_descr(1);
        let expected =
            VirtualState::new(vec![VirtualStateInfo::Constant(Value::Ref(GcRef(0x1234)))]);
        let incoming = VirtualState::new(vec![VirtualStateInfo::VArray {
            descr,
            items: vec![Some(VirtualStateInfoNode::new_rc(
                VirtualStateInfo::Constant(Value::Int(1)),
            ))],
            lenbound: None,
        }]);
        let boxes = vec![OpRef::ref_op(10)];
        let runtime = ctx.make_constant_ref(GcRef(0x1234));
        assert!(
            expected
                .generate_guards(&incoming, &boxes, &[runtime], &mut ctx, false)
                .is_err()
        );
    }

    #[test]
    fn test_guard_value_preserves_ref_constant_type() {
        let mut ctx = OptContext::new(128);
        let expected = GcRef(0x1234);
        let emitted = GuardRequirement::GuardValue {
            arg_index: 0,
            box_opref: OpRef::ref_op(10),
            expected_value: Value::Ref(expected),
        }
        .to_ops(&[OpRef::ref_op(10)], &mut ctx);
        assert_eq!(emitted.len(), 1);
        let guard = &emitted[0];

        assert_eq!(guard.opcode, OpCode::GuardValue);
        assert_eq!(
            ctx.get_box_replacement_operand_opt(guard.arg(1).to_opref())
                .and_then(|cb| cb.const_value()),
            Some(Value::Ref(expected))
        );
    }

    // ── Export/Import tests ──

    #[test]
    fn test_export_nonvirtual_array_preserves_widened_lenbound() {
        // virtualstate.py NotVirtualStateInfoPtr.__init__: the
        // non-virtual ArrayPtrInfo leaf retains getlenbound(None).widen().
        let mut ctx = OptContext::new(32);
        let array_ref = OpRef::ref_op(10);
        let array_box = ctx.materialize_operand_at(array_ref);
        ctx.set_ptr_info(
            &array_box,
            PtrInfo::array(test_descr(20), IntBound::bounded(0, 10)),
        );

        let state = export_state(&[array_ref], &ctx);
        assert!(matches!(state.state[0].info, VirtualStateInfo::NonNull));
        let lenbound = state.state[0]
            .lenbound
            .as_ref()
            .expect("non-virtual array length bound");
        assert_eq!((lenbound.lower, lenbound.upper), (0, 10));
        assert!(lenbound.are_knownbits_implied());
    }

    #[test]
    fn test_export_virtual_array_skips_unwritten_slot() {
        // virtualstate.py create_state_or_none / :724-725: an unwritten fieldbox exports
        // as absent fieldstate. virtualstate.py:272-280 then gives it neither
        // a forced box nor a position_in_notvirtuals slot.
        let mut ctx = OptContext::new(32);
        let array_ref = OpRef::ref_op(10);
        let written_ref = OpRef::int_op(11);
        let array_box = ctx.materialize_operand_at(array_ref);
        let written_box = ctx.materialize_operand_at(written_ref);
        ctx.set_ptr_info(
            &array_box,
            PtrInfo::VirtualArray(VirtualArrayInfo {
                descr: test_descr(21),
                clear: false,
                items: vec![Operand::None, written_box],
                last_guard_pos: -1,
                avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
            }),
        );

        let state = export_state(&[array_ref], &ctx);
        let VirtualStateInfo::VArray { items, .. } = &state.state[0].info else {
            panic!("exported state is not a virtual array");
        };
        assert!(items[0].is_none(), "unwritten slot must stay absent");
        let written_state = items[1].as_ref().expect("written slot state");
        assert_eq!(written_state.position_in_notvirtuals.get(), 0);

        // Hand-computed LABEL/JUMP arity: virtual head 0 + absent slot 0 +
        // one written Unknown(Int) leaf 1 = one non-virtual argument.
        assert_eq!(state.num_boxes(), 1);
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let inputargs = state
            .make_inputargs(&[array_ref], &mut optimizer, &mut ctx, false)
            .expect("exported virtual array must enumerate its written slot");
        assert_eq!(
            inputargs.iter().map(|b| b.to_opref()).collect::<Vec<_>>(),
            vec![written_ref]
        );
    }

    #[test]
    fn test_export_virtual_struct_skips_unwritten_slot() {
        // virtualstate.py `create_state_or_none`: an unwritten
        // struct fieldbox is None and must not reach not_virtual.
        let mut ctx = OptContext::new(32);
        let struct_ref = OpRef::ref_op(10);
        let written_ref = OpRef::int_op(11);
        let struct_box = ctx.materialize_operand_at(struct_ref);
        let written_box = ctx.materialize_operand_at(written_ref);
        ctx.set_ptr_info(
            &struct_box,
            PtrInfo::VirtualStruct(VirtualStructInfo {
                fields: vec![(0, Operand::None), (1, written_box)].into(),
                ..VirtualStructInfo::new(test_descr(21))
            }),
        );

        let state = export_state(&[struct_ref], &ctx);
        let VirtualStateInfo::VStruct { fields, .. } = &state.state[0].info else {
            panic!("exported state is not a virtual struct");
        };
        assert_eq!(fields.len(), 1, "unwritten slot must stay absent");
        assert_eq!(fields[0].0, 1);
        assert_eq!(state.num_boxes(), 1);
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let inputargs = state
            .make_inputargs(&[struct_ref], &mut optimizer, &mut ctx, false)
            .expect("exported virtual struct must enumerate its written slot");
        assert_eq!(
            inputargs.iter().map(|b| b.to_opref()).collect::<Vec<_>>(),
            vec![written_ref]
        );
    }

    #[test]
    fn test_make_inputargs_skips_virtual_entries() {
        let descr = test_descr(7);
        let state = VirtualState::new(vec![
            VirtualStateInfo::Unknown(Type::Int),
            VirtualStateInfo::VStruct {
                descr: descr.clone(),
                fields: vec![],
                field_descrs: Vec::new(),
            },
            VirtualStateInfo::NonNull,
        ]);

        let mut ctx = OptContext::new(16);
        // virtualstate.py requires `info.is_virtual()` for the
        // VStruct walker to descend; mirror by attaching a virtual
        // PtrInfo to the corresponding OpRef.
        let b11 = ctx.materialize_operand_at(OpRef::ref_op(11));
        ctx.set_ptr_info(&b11, PtrInfo::VirtualStruct(VirtualStructInfo::new(descr)));
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let inputargs = state
            .make_inputargs(
                &[OpRef::int_op(10), OpRef::ref_op(11), OpRef::ref_op(12)],
                &mut optimizer,
                &mut ctx,
                false,
            )
            .expect("make_inputargs");
        assert_eq!(
            inputargs.iter().map(|b| b.to_opref()).collect::<Vec<_>>(),
            vec![OpRef::int_op(10), OpRef::ref_op(12)]
        );
    }

    #[test]
    fn test_make_inputargs_and_virtuals_returns_virtual_boxes() {
        let descr = test_descr(9);
        let state = VirtualState::new(vec![
            VirtualStateInfo::Unknown(Type::Int),
            VirtualStateInfo::VStruct {
                descr: descr.clone(),
                fields: vec![],
                field_descrs: Vec::new(),
            },
            VirtualStateInfo::NonNull,
        ]);

        let mut ctx = OptContext::new(16);
        let b21 = ctx.materialize_operand_at(OpRef::ref_op(21));
        ctx.set_ptr_info(&b21, PtrInfo::VirtualStruct(VirtualStructInfo::new(descr)));
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let (inputargs, virtuals) = state
            .make_inputargs_and_virtuals(
                &[OpRef::int_op(20), OpRef::ref_op(21), OpRef::ref_op(22)],
                &mut optimizer,
                &mut ctx,
                false,
            )
            .expect("make_inputargs_and_virtuals");
        assert_eq!(
            inputargs.iter().map(|b| b.to_opref()).collect::<Vec<_>>(),
            vec![OpRef::int_op(20), OpRef::ref_op(22)]
        );
        assert_eq!(virtuals, vec![OpRef::ref_op(21)]);
    }

    #[test]
    fn test_label_source_positions_follow_make_inputargs_for_virtual_field() {
        let descr = test_descr(10);
        let mut ctx = OptContext::new(32);
        let object = OpRef::ref_op(10);
        let field = OpRef::int_op(11);
        let scalar = OpRef::int_op(12);
        ctx.materialize_operand_at(object);
        // The field box `enum_forced_boxes` walks is this producer, not a
        // second SameAs minted at the same position.
        let field_box = ctx.materialize_operand_at(field);
        ctx.materialize_operand_at(scalar);

        let object_box = ctx
            .get_box_replacement_operand_opt(object)
            .expect("object box is bound");
        let mut info = PtrInfo::virtual_obj(descr, None);
        info.setfield(0, field_box);
        ctx.set_ptr_info(&object_box, info);

        let state = export_state(&[object, field, scalar], &ctx);
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let (inputargs, virtuals, source_positions) = state
            .make_inputargs_and_virtuals_with_source_positions(
                &[object, field, scalar],
                &mut optimizer,
                &mut ctx,
                false,
            )
            .expect("make_inputargs_and_virtuals_with_source_positions");

        assert_eq!(
            inputargs.iter().map(|b| b.to_opref()).collect::<Vec<_>>(),
            vec![field, scalar]
        );
        assert_eq!(virtuals, vec![object]);
        assert_eq!(
            source_positions,
            vec![1, 2],
            "virtualstate.py:655-683 make_inputargs skips the top-level virtual; \
             unroll.py:462-465 LABEL packing must source the field slot, not \
             the virtual object slot"
        );
    }

    /// The export preview in `optimize_with_constants_and_inputs_at` builds the
    /// state from a list of oprefs and immediately re-matches it against that
    /// SAME list, so `make_inputargs_and_virtuals` cannot raise
    /// `VirtualStatesCantMatch` there: every `state[i]` was derived from
    /// `args[i]`.
    ///
    /// That self-match is why the optimizer's `building_bridge` branch — which
    /// keeps a preview mismatch non-fatal on the bridge path (unroll.py:193,
    /// 207-210) — is unreachable today; probing it with five virtual-carrying
    /// fixtures, two of which do compile bridges, produced zero hits.
    ///
    /// Upstream matches a preamble's exported state against a DIFFERENT loop's
    /// stored state in `jump_to_existing_trace` (unroll.py). Whoever moves
    /// the preview to that shape will break this test — and that is the signal
    /// that the `building_bridge` branch has become live and needs its own
    /// coverage.
    #[test]
    fn export_state_re_matched_against_its_own_args_cannot_fail() {
        let descr = test_descr(10);
        let mut ctx = OptContext::new(32);
        let object = OpRef::ref_op(10);
        let field = OpRef::int_op(11);
        let scalar = OpRef::int_op(12);
        ctx.materialize_operand_at(object);
        // The field box `enum_forced_boxes` walks is this producer, not a
        // second SameAs minted at the same position.
        let field_box = ctx.materialize_operand_at(field);
        ctx.materialize_operand_at(scalar);

        let object_box = ctx
            .get_box_replacement_operand_opt(object)
            .expect("object box is bound");
        let mut info = PtrInfo::virtual_obj(descr, None);
        info.setfield(0, field_box);
        ctx.set_ptr_info(&object_box, info);

        let args = [object, field, scalar];
        let state = export_state(&args, &ctx);
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let matched = state.make_inputargs_and_virtuals(&args, &mut optimizer, &mut ctx, false);

        assert!(
            matched.is_ok(),
            "a state exported from `args` must re-match `args`; \
             a failure here means the preview is no longer a self-match and the \
             optimizer's `building_bridge` branch is now reachable"
        );
    }

    /// virtualstate.py:196 / 274 / 352 — `state.position > self.position`
    /// shared-substate dedup parity. When two top-level state entries
    /// reference the same `Rc<VirtualStateInfoNode>` (an aliased nested box),
    /// the leaves under that subtree must be enumerated exactly once into
    /// the inputargs slot vector — matching RPython's per-state-object
    /// `position_in_notvirtuals` allocation.
    #[test]
    fn test_make_inputargs_dedups_shared_substate() {
        let descr = test_descr(13);
        // Two top-level VStruct entries that share the SAME Rc'd field.
        // After dedup the field's leaf occupies a single slot.
        let shared_field: Rc<VirtualStateInfoNode> =
            VirtualStateInfoNode::new_rc(VirtualStateInfo::NonNull);
        let outer_a = VirtualStateInfo::VStruct {
            descr: descr.clone(),
            fields: vec![(0, Rc::clone(&shared_field))],
            field_descrs: Vec::new(),
        };
        let outer_b = VirtualStateInfo::VStruct {
            descr: descr.clone(),
            fields: vec![(0, Rc::clone(&shared_field))],
            field_descrs: Vec::new(),
        };
        let state = VirtualState::new(vec![outer_a, outer_b]);
        // The dedup walker should report a single non-virtual leaf slot
        // (matching RPython numnotvirtuals on the same shared object).
        assert_eq!(state.num_boxes(), 1);

        let inner_field_value = OpRef::ref_op(31);
        let inner_field_op = crate::history::test_support::rooted_resop_operand(Type::Ref, 31);
        let outer_a_ref = OpRef::ref_op(40);
        let outer_b_ref = OpRef::ref_op(41);
        let mut ctx = OptContext::new(64);
        // Both outer boxes resolve to a virtual struct whose field 0 is
        // the shared inner OpRef.
        let outer_a_box = ctx.materialize_operand_at(outer_a_ref);
        let outer_b_box = ctx.materialize_operand_at(outer_b_ref);
        ctx.set_ptr_info(
            &outer_a_box,
            PtrInfo::VirtualStruct(VirtualStructInfo {
                fields: vec![(0, inner_field_op.clone())].into(),
                ..VirtualStructInfo::new(descr.clone())
            }),
        );
        ctx.set_ptr_info(
            &outer_b_box,
            PtrInfo::VirtualStruct(VirtualStructInfo {
                fields: vec![(0, inner_field_op.clone())].into(),
                ..VirtualStructInfo::new(descr)
            }),
        );

        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let inputargs = state
            .make_inputargs(&[outer_a_ref, outer_b_ref], &mut optimizer, &mut ctx, false)
            .expect("make_inputargs");
        assert_eq!(
            inputargs.iter().map(|b| b.to_opref()).collect::<Vec<_>>(),
            vec![inner_field_value]
        );
    }

    /// Top-level Rc dedup parity: when two jump args resolve to the
    /// same box, RPython's VirtualStateConstructor cache returns the
    /// same AbstractVirtualStateInfo Python object. The Rust port shares
    /// the top-level `Rc<VirtualStateInfoNode>` directly via
    /// `from_shared_rcs` so `numnotvirtuals` reflects the deduped
    /// slot count.
    #[test]
    fn test_top_level_rc_aliasing_dedups_slots() {
        let shared_leaf: Rc<VirtualStateInfoNode> =
            VirtualStateInfoNode::new_rc(VirtualStateInfo::NonNull);
        // Both top-level state entries are the SAME Rc, mirroring
        // VirtualStateConstructor returning the cached object for
        // aliased jump args.
        let state =
            VirtualState::from_shared_rcs(vec![Rc::clone(&shared_leaf), Rc::clone(&shared_leaf)]);
        assert_eq!(state.num_boxes(), 1);

        let outer_ref = OpRef::ref_op(50);
        let mut ctx = OptContext::new(64);
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let inputargs = state
            .make_inputargs(&[outer_ref, outer_ref], &mut optimizer, &mut ctx, false)
            .expect("make_inputargs");
        // Single deduped slot, written by the first top-level visit.
        assert_eq!(
            inputargs.iter().map(|b| b.to_opref()).collect::<Vec<_>>(),
            vec![outer_ref]
        );
    }

    /// `VirtualState` assignment shares the info graph (`VirtualState.__init__`
    /// is the constructor; later copies are the same Python object).
    #[test]
    fn clone_shares_node_identity() {
        let vs = VirtualState::new(vec![VirtualStateInfo::NonNull]);
        let cloned = vs.clone();
        assert!(Rc::ptr_eq(&vs.state[0], &cloned.state[0]));
        assert_eq!(vs.num_boxes(), cloned.num_boxes());
        assert_eq!(vs.state[0].position.get(), cloned.state[0].position.get());
    }

    /// The compiled-loop holder walk forwards `Constant` refs stored in the
    /// state, including a nested field, and leaves a vtable integer alone.
    #[test]
    fn retained_walk_forwards_constant_refs_in_the_state() {
        let descr = test_descr(3);
        let mut state = VirtualState::new(vec![
            VirtualStateInfo::Constant(Value::Ref(GcRef(0x1000))),
            VirtualStateInfo::VStruct {
                descr,
                fields: vec![(
                    0,
                    VirtualStateInfoNode::new_rc(VirtualStateInfo::Constant(Value::Ref(GcRef(
                        0x2000,
                    )))),
                )],
                field_descrs: Vec::new(),
            },
            VirtualStateInfo::KnownClass { class_ptr: 7 },
        ]);
        state.walk_const_ptr_refs_mut(&mut |gcref| {
            gcref.0 += 0x10;
        });
        match &state.state[0].info {
            VirtualStateInfo::Constant(Value::Ref(gcref)) => assert_eq!(gcref.0, 0x1010),
            other => panic!("top-level constant became {other:?}"),
        }
        match &state.state[1].info {
            VirtualStateInfo::VStruct { fields, .. } => match &fields[0].1.info {
                VirtualStateInfo::Constant(Value::Ref(gcref)) => assert_eq!(gcref.0, 0x2010),
                other => panic!("nested constant became {other:?}"),
            },
            other => panic!("struct became {other:?}"),
        }
        match &state.state[2].info {
            VirtualStateInfo::KnownClass { class_ptr } => assert_eq!(*class_ptr, 7),
            other => panic!("known class became {other:?}"),
        }
    }

    #[test]
    fn test_make_inputargs_recursively_extracts_virtual_fields() {
        let descr = test_descr(11);
        let field_value = OpRef::ref_op(21);
        let field_value_op = crate::history::test_support::rooted_resop_operand(Type::Ref, 21);
        let virtual_ref = OpRef::ref_op(20);
        let state = VirtualState::new(vec![VirtualStateInfo::VStruct {
            descr: descr.clone(),
            fields: vec![
                (
                    0,
                    VirtualStateInfoNode::new_rc(VirtualStateInfo::Constant(Value::Int(7))),
                ),
                (8, VirtualStateInfoNode::new_rc(VirtualStateInfo::NonNull)),
            ],
            field_descrs: Vec::new(),
        }]);
        let mut ctx = OptContext::new(32);
        let virtual_box = ctx.materialize_operand_at(virtual_ref);
        ctx.set_ptr_info(
            &virtual_box,
            PtrInfo::VirtualStruct(VirtualStructInfo {
                fields: vec![(0, Operand::None), (8, field_value_op.clone())].into(),
                ..VirtualStructInfo::new(descr)
            }),
        );

        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        let inputargs = state
            .make_inputargs(&[virtual_ref], &mut optimizer, &mut ctx, false)
            .expect("make_inputargs");
        assert_eq!(
            inputargs.iter().map(|b| b.to_opref()).collect::<Vec<_>>(),
            vec![field_value]
        );
    }

    #[test]
    fn test_force_boxes() {
        let descr = test_descr(0);
        let mut state = VirtualState::new(vec![
            VirtualStateInfo::Virtual {
                descr: descr.clone(),
                known_class: Some(0x1000),
                ob_type_descr: None,
                fields: vec![],
                field_descrs: Vec::new(),
            },
            VirtualStateInfo::NonNull,
            VirtualStateInfo::VArray {
                descr,
                items: vec![Some(VirtualStateInfoNode::new_rc(
                    VirtualStateInfo::Unknown(Type::Int),
                ))],
                lenbound: None,
            },
            VirtualStateInfo::Unknown(Type::Int),
        ]);
        assert_eq!(state.num_virtuals(), 2);
        let forced = state.force_boxes();
        assert_eq!(forced, 2);
        assert_eq!(state.num_virtuals(), 0);
        // Virtual with known_class becomes KnownClass
        assert!(matches!(
            &state.state[0].info,
            VirtualStateInfo::KnownClass { .. }
        ));
        // VirtualArray becomes NonNull
        assert!(matches!(&state.state[2].info, VirtualStateInfo::NonNull));
    }

    #[test]
    fn test_make_inputargs_with_optimizer_retries_virtual_into_nonvirtual_slot() {
        let descr = test_descr(12);
        let virtual_ref = OpRef::ref_op(20);
        let state = VirtualState::new(vec![VirtualStateInfo::NonNull]);
        // Generous Ref-typed inputarg pool for the test fixture.
        let mut ctx = OptContext::with_inputarg_types(32, &vec![Type::Ref; 1024]);
        let virtual_box = ctx.materialize_operand_at(virtual_ref);
        ctx.set_ptr_info(
            &virtual_box,
            PtrInfo::VirtualStruct(VirtualStructInfo::new(descr)),
        );
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();

        assert!(
            state
                .make_inputargs_and_virtuals(&[virtual_ref], &mut optimizer, &mut ctx, false,)
                .is_err()
        );

        let (inputargs, virtuals) = state
            .make_inputargs_and_virtuals(&[virtual_ref], &mut optimizer, &mut ctx, true)
            .expect("force_boxes=True should retry instead of failing");
        // After forcing, the virtual struct is replaced by a concrete
        // allocation at a new position. The inputarg should be that
        // forced allocation ref (which is what ctx.get_replacement
        // resolves the original virtual_ref to).
        assert_eq!(inputargs.len(), 1);
        assert_eq!(
            inputargs[0].to_opref(),
            ctx.get_replacement_opref(virtual_ref)
        );
        assert!(virtuals.is_empty());
    }

    #[test]
    fn test_make_inputargs_forces_through_earlyforce_next() {
        // virtualstate.py `VirtualState.make_inputargs` replaces `optimizer`
        // with `optimizer.optearlyforce` before enum_forced_boxes.  A force
        // requested after heap.flush() must consequently enqueue the
        // allocation at earlyforce.next (= heap), not after the stale heap
        // cursor.  Otherwise OptHeap misses the initializing stores and the
        // imported short preamble re-emits redundant GETFIELD_GC operations.
        let descr = test_descr(13);
        let virtual_ref = OpRef::ref_op(20);
        let state = VirtualState::new(vec![VirtualStateInfo::NonNull]);
        let mut ctx = OptContext::with_inputarg_types(32, &vec![Type::Ref; 1024]);
        let virtual_box = ctx.materialize_operand_at(virtual_ref);
        ctx.set_ptr_info(
            &virtual_box,
            PtrInfo::VirtualStruct(VirtualStructInfo::new(descr)),
        );
        let mut optimizer = crate::optimizeopt::optimizer::Optimizer::new();
        ctx.optearlyforce_idx = 5;
        ctx.current_pass_idx = 6;

        state
            .make_inputargs(&[virtual_ref], &mut optimizer, &mut ctx, true)
            .expect("force_boxes=True should materialize the virtual");

        assert_eq!(ctx.current_pass_idx, 6, "restore the caller's pass cursor");
        assert_eq!(ctx.extra_operations_after.len(), 1);
        assert_eq!(
            ctx.extra_operations_after.front().map(|(start, _)| *start),
            Some(6),
            "forced operations must enter at OptEarlyForce.next"
        );
    }
}
