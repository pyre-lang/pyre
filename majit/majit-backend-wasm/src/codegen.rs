/// IR → wasm bytecode compilation.
///
/// Generates a wasm module from majit IR ops using `wasm-encoder`.
/// Generated function signature: `(param $frame_ptr i32) (result i32)`
///
/// Frame layout in shared linear memory (items base = `jf_frame`):
///   offset 0:       dispatch key (i64). The exit descr lives in `jf_descr`.
///   offset 8:       slot[0] (i64)
///   offset 16:      slot[1] (i64)
///   ...
///
/// The residual-call trampoline scratch is stored separately at the static
/// base returned by `jit_call_area_addr`.
use std::collections::HashMap;
use std::sync::Arc;

use majit_backend::BackendError;
use majit_gc::header::{GcHeader, TYPE_ID_MASK};
use majit_ir::forwarding::Forwarded;
use majit_ir::operand::Operand;
use majit_ir::{InputArg, InputArgRc, Op, OpCode, OpRef, Type, Value};
use wasm_encoder::{
    BlockType, CodeSection, ConstExpr, EntityType, ExportKind, ExportSection, Function,
    FunctionSection, GlobalSection, GlobalType, ImportSection, InstructionSink, MemArg, MemoryType,
    Module, RefType, TableType, TypeSection, ValType,
};

/// Insert GETARRAYITEM loads for LABEL args that have no producer.
///
/// `patch_new_loop_to_load_virtualizable_fields` rewrites root inputarg
/// slots only. A body LABEL still carries virtualstate boxes for
/// `locals_cells_stack_w` items the peel never wrote. Native regalloc
/// binds those boxes to the same location the heap load would use; wasm
/// locals start at zero, so a later deopt writes a null local back as
/// bytecode state. Clone an existing array load in this trace — same
/// descr, same array pointer — and store into the missing LABEL id.
pub fn materialize_unbound_label_args(inputargs: &[InputArgRc], ops: &mut Vec<Op>) {
    let produced: std::collections::HashSet<u32> = ops
        .iter()
        .filter_map(result_value_raw)
        .chain(inputargs.iter().map(|ia| ia.index))
        .collect();
    let Some(label) = ops.iter().find(|op| op.opcode == OpCode::Label) else {
        return;
    };
    let label_args: Vec<OpRef> = label.getarglist().iter().map(|a| a.to_opref()).collect();
    let mut missing: Vec<(usize, u32)> = Vec::new();
    for (i, opref) in label_args.iter().enumerate() {
        // Peeled live-ins are InputArg* block parameters, not residual
        // virtualizable slots. Do not replace them with a heap load.
        if opref.is_none() || opref.is_constant() || opref.is_input_arg() {
            continue;
        }
        let raw = opref.raw();
        if !produced.contains(&raw) {
            missing.push((i, raw));
        }
    }
    if missing.is_empty() {
        return;
    }
    // Only a GETARRAYITEM whose result is already a LABEL arg is the
    // frame-locals array. A list/tuple items load in the same trace is the
    // wrong array; cloning it writes a foreign object into a vable slot.
    let mut template: Option<(Op, i64, usize)> = None;
    for (i, opref) in label_args.iter().enumerate() {
        if opref.is_none() || opref.is_constant() {
            continue;
        }
        let raw = opref.raw();
        let Some(op) = ops.iter().find(|op| {
            matches!(
                op.opcode,
                OpCode::GetarrayitemGcR | OpCode::GetarrayitemGcI | OpCode::GetarrayitemGcF
            ) && op.pos().get() != OpRef::NONE
                && !op.pos().get().is_constant()
                && op.pos().get().raw() == raw
        }) else {
            continue;
        };
        if op.num_args() < 2 {
            continue;
        }
        let Some(index) = op.arg(1).to_opref().inline_const_bits() else {
            continue;
        };
        template = Some((op.clone(), index, i));
        break;
    }
    let Some((template, template_index, template_label_i)) = template else {
        return;
    };
    let insert_at = ops
        .iter()
        .position(|op| op.opcode == OpCode::Label)
        .unwrap_or(0);
    let mut loads = Vec::new();
    for (label_i, raw) in missing {
        let array_index = template_index + (label_i as i64 - template_label_i as i64);
        if array_index < 0 {
            continue;
        }
        let load = template.clone();
        load.setarg(
            1,
            majit_ir::operand::Operand::const_from_value(majit_ir::Value::Int(array_index)),
        );
        let result_ty = match template.opcode {
            OpCode::GetarrayitemGcI => Type::Int,
            OpCode::GetarrayitemGcF => Type::Float,
            _ => Type::Ref,
        };
        load.pos().set(OpRef::op_typed(raw, result_ty));
        loads.push(load);
    }
    if !loads.is_empty() {
        ops.splice(insert_at..insert_at, loads);
    }
}

/// Frame slot byte offset: slot[i] is at frame_ptr + 8 + i * 8.
pub const FRAME_SLOT_BASE: u64 = 8;
pub(crate) const SLOT_SIZE: u64 = 8;

/// Scratch i64 locals reserved past the value locals for `emit_umulhi`
/// (al, ah, bl, bh, mid1).
const UMULHI_SCRATCH: u32 = 5;

/// Compile-time ConstPtr identities of the `GcTable`s this emit owns, in
/// slot order under each table's `base_addr`.
///
/// `GcTable::compile_key` is the address `store_info_on_descr` leaves in the
/// assembler's constant-pointer table. A later collection forwards the slot;
/// the key does not change. Lookup is the emitting region's base, so two
/// tables that reuse one nursery address do not share a slot.
struct ConstPtrTables {
    entries: Vec<(u32, Vec<usize>)>,
}

impl ConstPtrTables {
    fn push(&mut self, base: u32, keys: &[usize]) {
        if keys.is_empty() || self.entries.iter().any(|(b, _)| *b == base) {
            return;
        }
        self.entries.push((base, keys.to_vec()));
    }

    fn slot(&self, base: u32, compile_key: usize) -> Option<(u32, u32)> {
        let keys = self.entries.iter().find(|(b, _)| *b == base)?.1.as_slice();
        let index = keys.iter().position(|&key| key == compile_key)? as u32;
        Some((base, index))
    }
}

/// Value boxes occupy wasm locals, numbered by opencoder `_index`
/// (`InputArg*` / `IntOp` / `FloatOp` / `RefOp`). Void ops are
/// `VoidOp(_count)` (`opencoder.py` `_op_end`); that payload shares the
/// integer space with a later value box, so it must not index a local.
fn value_box_raw(r: OpRef) -> Option<u32> {
    if r.is_none() || r.is_constant() || r.is_temp_var() {
        return None;
    }
    if r.ty() == Some(Type::Void) {
        return None;
    }
    Some(r.raw())
}

fn result_value_raw(op: &Op) -> Option<u32> {
    if op.result_type() == Type::Void {
        None
    } else {
        value_box_raw(op.pos().get())
    }
}

fn widen_value_id(r: OpRef, end: &mut u32) {
    if let Some(id) = value_box_raw(r) {
        *end = (*end).max(id + 1);
    }
}

/// Dense wasm-local assignment for the sparse value-id namespace.
struct ValueLocals {
    by_id: Vec<Option<u32>>,
    types: Vec<ValType>,
    /// First non-parameter local.  Ordinary traces have one frame-pointer
    /// parameter; parameter-entry bridges have that plus their fail values.
    first_local: u32,
}

impl ValueLocals {
    fn mark(
        by_id: &mut [Option<u32>],
        id_types: &mut [ValType],
        has_authoritative_type: &mut [bool],
        id: u32,
        ty: ValType,
        authoritative: bool,
    ) {
        let i = id as usize;
        assert!(i < by_id.len(), "value id {id} exceeds pre-pass bounds");
        by_id[i] = Some(0);
        // InputArg::tp and Op::result_type describe the defining value.  An
        // operand may be visited before its producer, so its embedded tag
        // only supplies a type while no definition has claimed this id.
        if authoritative || !has_authoritative_type[i] {
            id_types[i] = ty;
        }
        has_authoritative_type[i] |= authoritative;
    }

    fn collect(inputargs: &[InputArgRc], ops: &[Op], num_vars: u32, first_local: u32) -> Self {
        let mut by_id = vec![None; num_vars as usize];
        let mut id_types = vec![ValType::I64; num_vars as usize];
        let mut has_authoritative_type = vec![false; num_vars as usize];

        for ia in inputargs {
            Self::mark(
                &mut by_id,
                &mut id_types,
                &mut has_authoritative_type,
                ia.index,
                if ia.tp.get() == Type::Float {
                    ValType::F64
                } else {
                    ValType::I64
                },
                true,
            );
        }
        for op in ops {
            if let Some(id) = result_value_raw(op) {
                Self::mark(
                    &mut by_id,
                    &mut id_types,
                    &mut has_authoritative_type,
                    id,
                    if op.result_type() == Type::Float {
                        ValType::F64
                    } else {
                        ValType::I64
                    },
                    true,
                );
            }
            for arg in op.getarglist() {
                let arg = arg.to_opref();
                if let Some(id) = value_box_raw(arg) {
                    Self::mark(
                        &mut by_id,
                        &mut id_types,
                        &mut has_authoritative_type,
                        id,
                        if arg.ty() == Some(Type::Float) {
                            ValType::F64
                        } else {
                            ValType::I64
                        },
                        false,
                    );
                }
            }
            if let Some(failargs) = op.getfailargs() {
                for arg in failargs {
                    let arg = arg.to_opref();
                    if let Some(id) = value_box_raw(arg) {
                        Self::mark(
                            &mut by_id,
                            &mut id_types,
                            &mut has_authoritative_type,
                            id,
                            if arg.ty() == Some(Type::Float) {
                                ValType::F64
                            } else {
                                ValType::I64
                            },
                            false,
                        );
                    }
                }
            }
        }

        // RPython's SAME_AS is a location remap, not a move. Preserve that
        // identity at the wasm-local boundary: an SSA alias can share
        // its source local for its whole lifetime. Ref homes remain value-id
        // based below, so an aliased Ref that needs a distinct force/resume
        // home still mirrors this shared local into that home after the op.
        // LABEL args are mutable phi locations in this backend: terminal JUMP
        // rebinds them on every iteration. Never coalesce either side with a
        // label local, because two equal values at SAME_AS can diverge after
        // the back-edge assignment.
        let mut label_value_ids = vec![false; num_vars as usize];
        for id in ops
            .iter()
            .filter(|op| op.opcode == OpCode::Label)
            .flat_map(|op| op.getarglist().into_iter().map(|arg| arg.to_opref()))
            .filter_map(value_box_raw)
        {
            label_value_ids[id as usize] = true;
        }
        let mut alias_source = vec![None; num_vars as usize];
        for op in ops {
            if !matches!(
                op.opcode,
                OpCode::SameAsI | OpCode::SameAsR | OpCode::SameAsF
            ) {
                continue;
            }
            let result = op.pos().get();
            let source = op.arg(0).to_opref();
            let Some(result_id) = value_box_raw(result) else {
                continue;
            };
            let Some(source_id) = value_box_raw(source) else {
                continue;
            };
            if label_value_ids[result_id as usize] || label_value_ids[source_id as usize] {
                continue;
            }
            let dst = result_id as usize;
            let src = source_id as usize;
            if dst < alias_source.len()
                && src < by_id.len()
                && by_id[src].is_some()
                && id_types[dst] == id_types[src]
            {
                alias_source[dst] = Some(src);
            }
        }
        // Loop-closing defs that do not interfere with their LABEL slot
        // share that slot's local (x86 colors the same way). SAME_AS
        // above refuses LABEL ids; this edge is the one JUMP rewrite.
        for (jid, lid) in jump_phi_coalesce_pairs(ops) {
            let dst = jid as usize;
            let src = lid as usize;
            if dst < alias_source.len()
                && src < by_id.len()
                && by_id[src].is_some()
                && by_id[dst].is_some()
                && alias_source[dst].is_none()
                && id_types[dst] == id_types[src]
            {
                alias_source[dst] = Some(src);
            }
        }

        // One wasm local per SSA root. A loop-carried Int whose last body
        // read is before a later def must still occupy that local on the
        // backedge. x86 `consider_label` force_spill parks it in a frame
        // slot (`is_last_real_use_before`) so the register can be reused;
        // wasm Ints have no such slot.
        let mut types = Vec::new();
        let mut root_locals = vec![None; num_vars as usize];
        for id in 0..by_id.len() {
            if by_id[id].is_none() {
                continue;
            }
            let mut root = id;
            let mut remaining = alias_source.len();
            while let Some(source) = alias_source[root] {
                // SAME_AS edges are SSA-backward and therefore acyclic.  Keep
                // the bound nevertheless so malformed IR fails closed into a
                // distinct local instead of looping in backend compilation.
                if remaining == 0 || source >= alias_source.len() {
                    root = id;
                    break;
                }
                root = source;
                remaining -= 1;
            }
            let local = if let Some(local) = root_locals[root] {
                local
            } else {
                let local = types.len() as u32 + first_local;
                root_locals[root] = Some(local);
                types.push(id_types[root]);
                local
            };
            by_id[id] = Some(local);
        }
        Self {
            by_id,
            types,
            first_local,
        }
    }

    fn local(&self, id: u32) -> u32 {
        self.by_id
            .get(id as usize)
            .copied()
            .flatten()
            .unwrap_or_else(|| panic!("wasm value local is unmapped for id {id}"))
    }

    fn ty(&self, id: u32) -> ValType {
        self.types[(self.local(id) - self.first_local) as usize]
    }

    fn count(&self) -> u32 {
        self.types.len() as u32
    }

    fn types(&self) -> &[ValType] {
        &self.types
    }

    /// Local index immediately after the dense value-local range.
    fn end_local(&self) -> u32 {
        self.first_local + self.count()
    }

    /// Last dense value-local index, used as the base before scratch locals.
    fn last_local(&self) -> u32 {
        self.end_local() - 1
    }
}

/// Call area layout in the historical fixed frame geometry.
///
/// These offsets are the host trampoline's ABI, not a private detail: a caller
/// writes the callee's function-table index, the argument count and the
/// arguments into this block, invokes the import, and reads the callee's
/// result back from it. Whoever satisfies that import reads the same block
/// from the other side, so both ends name these constants instead of each
/// restating the numbers.
pub const CALL_RESULT_OFS: u64 = 2000;
pub const CALL_FUNC_OFS: u64 = 2008;
pub const CALL_NARGS_OFS: u64 = 2016;
pub const CALL_ARGS_OFS: u64 = 2024;
/// `CallDescr.result_size` for the residual, in bytes. Zero is void.
pub const CALL_RESULT_SIZE_OFS: u64 = CALL_ARGS_OFS + (MAX_CALL_ARGS as u64) * SLOT_SIZE;
/// Host-allocated sret slot. Size is the descr `result_size`, capped by
/// [`MAX_SRET_BYTES`].
pub const CALL_SRET_OFS: u64 = CALL_RESULT_SIZE_OFS + SLOT_SIZE;

/// Arguments the call area has room for. A residual call with more arguments
/// than this has nowhere to put them, so a caller checks its arity against
/// this bound rather than writing past the end of the frame.
pub const MAX_CALL_ARGS: usize = 16;

/// One-word sret blob. A descr whose `result_size` exceeds a word is an
/// aggregate `bh_call_*` cannot describe.
pub const MAX_SRET_BYTES: usize = 8;

const STATIC_CALL_RESULT_OFS: u64 = 0;
const STATIC_CALL_FUNC_OFS: u64 = SLOT_SIZE;
const STATIC_CALL_NARGS_OFS: u64 = 2 * SLOT_SIZE;
const STATIC_CALL_ARGS_OFS: u64 = 3 * SLOT_SIZE;
const STATIC_CALL_RESULT_SIZE_OFS: u64 = STATIC_CALL_ARGS_OFS + (MAX_CALL_ARGS as u64) * SLOT_SIZE;

/// Minimum frame allocation size in bytes to accommodate the call area.
///
/// Derived from where the arguments start and how many fit, so raising
/// [`MAX_CALL_ARGS`] cannot leave the frame one argument short of the area it
/// is sized to hold.
pub const MIN_FRAME_BYTES: usize = CALL_SRET_OFS as usize + MAX_SRET_BYTES;

/// Per-token layout of a wasm execution frame. Every frozen geometry retains
/// the historical host-trampoline call area even though emitted code uses the
/// module-static scratch area. CA callee frames allocate only the prefix ending
/// after the Ref homes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameGeometry {
    /// Number of value slots before the dispatch key (including frame[0]).
    pub value_slots: usize,
    /// Byte offset of the call trampoline result word.
    pub call_result_ofs: u64,
    pub call_func_ofs: u64,
    pub call_nargs_ofs: u64,
    pub call_args_ofs: u64,
    /// Byte offset of the resume-at-LABEL key.
    pub dispatch_key_ofs: u64,
    /// Byte offset of Ref-home zero.
    pub home_slot_base: u64,
    /// Number of Ref-home slots the layout reserves.
    pub home_slots: usize,
    /// Number of slots at the END of the Ref-home region reserved for
    /// resume-at-LABEL live-ins.  Ordinary per-trace Ref homes grow upward
    /// from `home_slot_base`; these captures grow from the frozen boundary and
    /// therefore survive execution of a chained bridge, whose own home map may
    /// use the low slots.  The published `jf_gcmap` marks the used ordinary
    /// prefix plus these captures, not the unused reserved tail.
    pub label_ref_slots: usize,
    /// Start of the GUARD_NOT_FORCED(_2) failarg spill area. It contains one
    /// i64 slot per value slot and is disjoint from exits, dispatch, homes and
    /// the residual-call area.
    pub force_slot_base: u64,
    /// Bytes through the end of Ref homes. CA callee frames allocate exactly
    /// this many item bytes; the unused tail call area is intentionally omitted.
    pub ca_frame_bytes: u32,
    /// Full bytes in the frame layout, including the tail call area. Host entry
    /// frames and every chained bridge use this geometry and allocation size.
    /// An extended geometry keeps the source bytes through [`Self::tail_base`]
    /// and stores the grown total here.
    pub frame_bytes: u32,
    /// Value slots addressed at [`FRAME_SLOT_BASE`]. Equals [`Self::value_slots`]
    /// until [`Self::extend`] appends the overflow past [`Self::tail_base`].
    pub prefix_value_slots: usize,
    /// Ordinary Ref homes addressed at [`Self::home_slot_base`]. Label captures
    /// stay at the end of this prefix; extra ordinary homes go to the tail.
    pub prefix_ordinary_homes: usize,
    /// Ordinary Ref homes past [`Self::prefix_ordinary_homes`], one per tail row.
    pub extra_ordinary_homes: usize,
    /// Byte offset of tail row 0. Zero when the geometry is not extended.
    /// A later [`Self::extend`] keeps this base: row `j` stays three words
    /// `[value_j, force_j, home_j]`.
    pub tail_base: u64,
}

impl FrameGeometry {
    /// result, function, nargs, args, result_size, sret blob.
    pub const CALL_AREA_SLOTS: usize = 3 + MAX_CALL_ARGS + 1 + MAX_SRET_BYTES / 8;

    /// Historical fixed geometry, used by direct codegen tests and by callers
    /// that deliberately need the arena-compatible layout.
    pub const fn fixed() -> Self {
        Self {
            value_slots: MIN_FRAME_BYTES / 8,
            call_result_ofs: CALL_RESULT_OFS,
            call_func_ofs: CALL_FUNC_OFS,
            call_nargs_ofs: CALL_NARGS_OFS,
            call_args_ofs: CALL_ARGS_OFS,
            dispatch_key_ofs: DISPATCH_KEY_OFS,
            home_slot_base: HOME_SLOT_BASE,
            home_slots: 0,
            label_ref_slots: 0,
            force_slot_base: (MIN_FRAME_BYTES + SLOT_SIZE as usize) as u64,
            ca_frame_bytes: (MIN_FRAME_BYTES * 2 + SLOT_SIZE as usize) as u32,
            frame_bytes: (MIN_FRAME_BYTES * 2 + SLOT_SIZE as usize) as u32,
            prefix_value_slots: MIN_FRAME_BYTES / 8,
            prefix_ordinary_homes: 0,
            extra_ordinary_homes: 0,
            tail_base: 0,
        }
    }

    /// Compact frozen geometry for one token:
    /// `[value slots | dispatch key | Ref homes | force slots | call area]`.
    /// `value_slots` includes frame[0].  The trailing call area is always
    /// present, even for direct-only source traces, because later bridges are
    /// compiled against this immutable geometry.
    pub fn compact(value_slots: usize, home_slots: usize, label_ref_slots: usize) -> Self {
        debug_assert!(label_ref_slots <= home_slots);
        let value_slots = value_slots.max(1);
        let dispatch_key_ofs = (value_slots as u64) * SLOT_SIZE;
        let home_slot_base = dispatch_key_ofs + SLOT_SIZE;
        let force_slot_base = home_slot_base + home_slots as u64 * SLOT_SIZE;
        let ca_frame_bytes = force_slot_base + value_slots as u64 * SLOT_SIZE;
        let call_result_ofs = ca_frame_bytes;
        let call_func_ofs = call_result_ofs + SLOT_SIZE;
        let call_nargs_ofs = call_func_ofs + SLOT_SIZE;
        let call_args_ofs = call_nargs_ofs + SLOT_SIZE;
        let frame_bytes = call_result_ofs + Self::CALL_AREA_SLOTS as u64 * SLOT_SIZE;
        Self {
            value_slots,
            call_result_ofs,
            call_func_ofs,
            call_nargs_ofs,
            call_args_ofs,
            dispatch_key_ofs,
            home_slot_base,
            home_slots,
            label_ref_slots,
            force_slot_base,
            ca_frame_bytes: ca_frame_bytes as u32,
            frame_bytes: frame_bytes as u32,
            prefix_value_slots: value_slots,
            prefix_ordinary_homes: home_slots - label_ref_slots,
            extra_ordinary_homes: 0,
            tail_base: 0,
        }
    }

    /// `assembler.py _check_frame_depth`: keep every offset of `self`.
    /// Overflow value slots, force args, and ordinary Ref homes share a
    /// tail of fixed-stride rows at [`Self::tail_base`]. Row `j` is
    /// `[value_j, force_j, home_j]`. A second extend keeps `tail_base` and
    /// the prefix counts, so an index already published does not move.
    /// A trace that already fits is returned unchanged.
    pub fn extend(self, value_slots: usize, ordinary_homes: usize) -> Self {
        let values = value_slots.max(self.value_slots);
        let homes = ordinary_homes.max(self.addressable_ordinary_homes());
        if values == self.value_slots && homes == self.addressable_ordinary_homes() {
            return self;
        }
        let mut ext = self;
        if ext.tail_base == 0 {
            ext.tail_base = self.frame_bytes as u64;
        }
        ext.value_slots = values;
        ext.extra_ordinary_homes = homes - self.prefix_ordinary_homes;
        let extra_values = values - self.prefix_value_slots;
        let rows = extra_values.max(ext.extra_ordinary_homes) as u64;
        let growth = 3 * rows * SLOT_SIZE;
        ext.frame_bytes = u32::try_from(ext.tail_base)
            .expect("tail_base")
            .checked_add(u32::try_from(growth).expect("tail growth"))
            .expect("extended frame_bytes");
        ext
    }

    /// True when [`Self::extend`] appended a tail past the source layout.
    pub const fn has_tail(self) -> bool {
        self.tail_base != 0
    }

    /// Frame depth, in Signed items, that `compile_loop` installs on the
    /// token's `frame_info`. A CALL_ASSEMBLER caller allocates the callee frame
    /// from it and stores the arguments at `_ll_initial_locs` before the
    /// callee's `_check_frame_depth` runs, so a tailed geometry covers the
    /// whole tail; otherwise the callee frame is the homes prefix
    /// (`ca_frame_bytes`).
    pub const fn ca_frame_depth(self) -> usize {
        let bytes = if self.has_tail() {
            self.frame_bytes
        } else {
            self.ca_frame_bytes
        };
        bytes as usize / std::mem::size_of::<isize>()
    }

    /// Signed item count `JitFrame::init` stores as `jf_frame.length`.
    pub const fn signed_item_count(self) -> usize {
        self.frame_bytes as usize / std::mem::size_of::<isize>()
    }

    /// Fail-arg / input slot `slot`. Indices below the source prefix stay at
    /// [`FRAME_SLOT_BASE`]; the rest are the value word of tail row
    /// `slot - prefix_value_slots`.
    pub fn spill_slot_ofs(self, slot: u64) -> u64 {
        let prefix = self.prefix_value_slots as u64;
        if slot < prefix || !self.has_tail() {
            FRAME_SLOT_BASE + slot * SLOT_SIZE
        } else {
            self.tail_base + 3 * (slot - prefix) * SLOT_SIZE
        }
    }

    /// Physical `jf_frame` item of [`Self::spill_slot_ofs`]: `(offset -
    /// FRAME_SLOT_BASE) / SLOT_SIZE`. Identity when the geometry has no tail,
    /// so a recorded fail location and `FRAME_SLOT_BASE + loc * 8` name the
    /// same byte the guest stored. Tail items are three apart.
    pub fn spill_slot_index(self, slot: u64) -> u64 {
        (self.spill_slot_ofs(slot) - FRAME_SLOT_BASE) / SLOT_SIZE
    }

    /// Physical item of the first tail value word. Later tail values are
    /// three items apart (`value_tail_index + 3*j`).
    pub fn value_tail_index(self) -> u64 {
        (self.tail_base - FRAME_SLOT_BASE) / SLOT_SIZE
    }

    /// GUARD_NOT_FORCED(_2) failarg `slot`. The source reserves
    /// `prefix_value_slots` words at [`Self::force_slot_base`]; the rest are
    /// the force word of tail row `slot - prefix_value_slots`.
    pub fn force_slot_ofs(self, slot: u64) -> u64 {
        let prefix = self.prefix_value_slots as u64;
        if slot < prefix || !self.has_tail() {
            self.force_slot_base + slot * SLOT_SIZE
        } else {
            self.tail_base + (3 * (slot - prefix) + 1) * SLOT_SIZE
        }
    }

    /// Byte offset of the first tail force word. Zero when [`Self::has_tail`]
    /// is false. Later force words are three slots apart, matching the
    /// stride of [`Self::spill_slot_index`].
    pub fn force_tail_base(self) -> u64 {
        if !self.has_tail() {
            return 0;
        }
        self.tail_base + SLOT_SIZE
    }

    /// LABEL scalar capture. In-prefix slots keep `slot * SLOT_SIZE`.
    /// A tail slot is the value word of that row, the same bytes as
    /// [`Self::spill_slot_ofs`].
    pub fn capture_value_ofs(self, slot: usize) -> u64 {
        let slot = slot as u64;
        let prefix = self.prefix_value_slots as u64;
        if slot < prefix || !self.has_tail() {
            slot * SLOT_SIZE
        } else {
            self.spill_slot_ofs(slot)
        }
    }

    /// Ordinary Ref home `h`. Homes below the source prefix stay at
    /// `home_slot_base`. Home `h >= prefix_ordinary_homes` is the home word
    /// of tail row `h - prefix_ordinary_homes`. Label captures are not
    /// ordinary homes and keep `home_slot_base`.
    pub fn home_ofs(self, h: u64) -> u64 {
        let prefix = self.prefix_ordinary_homes as u64;
        if h < prefix {
            self.home_slot_base + h * SLOT_SIZE
        } else {
            self.tail_base + (3 * (h - prefix) + 2) * SLOT_SIZE
        }
    }

    /// Ordinary homes this geometry can address, prefix plus tail.
    pub const fn addressable_ordinary_homes(self) -> usize {
        self.ordinary_home_slots() + self.extra_ordinary_homes
    }

    /// Host entry writes `nargs` values at [`Self::spill_slot_ofs`] and the
    /// dispatch key at [`Self::dispatch_key_ofs`]. Each arg slot must end at
    /// or before the key, and the key must end at or before the Ref homes.
    pub fn debug_assert_dispatch_entry_slots(self, nargs: u64) {
        let key = self.dispatch_key_ofs;
        for i in 0..nargs {
            let end = self.spill_slot_ofs(i) + SLOT_SIZE;
            debug_assert!(
                end <= key,
                "entry arg {i} ends at {end}, past dispatch key {key}"
            );
        }
        let key_end = key + SLOT_SIZE;
        debug_assert!(
            key_end <= self.home_slot_base,
            "dispatch key ends at {key_end}, past homes at {}",
            self.home_slot_base
        );
    }

    /// Low Ref homes available to the trace currently executing on this
    /// geometry.  The high `label_ref_slots` belong to the source loop's LABEL
    /// capture plan and must not be cleared or reused by a chained bridge.
    pub const fn ordinary_home_slots(self) -> usize {
        self.home_slots - self.label_ref_slots
    }
}

/// Byte offset of the Ref-home region within the frame. Each Ref value that is
/// live across a collecting call is given a dedicated home slot here: it is
/// null-initialized at trace entry and written on every definition
/// (store-on-def), so a home slot only ever holds null or a valid GcRef.
/// A collecting allocation registers these slots as GC roots and forwards them,
/// then the trace reloads the live Ref locals from their homes — making object
/// movement transparent without rooting Refs that never cross a collection.
///
/// In compact geometries this region follows the dispatch key and precedes the
/// trailing call area. Inert while `wasm_jit_alloc` is no-collect (epic B): the
/// extra stores write a region nothing reads until the allocator collects.
pub const HOME_SLOT_BASE: u64 = MIN_FRAME_BYTES as u64 + SLOT_SIZE;

/// Historical fixed-geometry resume-at-LABEL dispatch key (one reserved frame
/// slot, between the call area and the Ref-home region). 0 = preamble/host entry (the `vec![0i64]`
/// frame is always 0 here on a fresh `execute_token`); non-zero = a
/// loop-closing bridge re-entering a single-label peeled loop at its LABEL,
/// skipping the preamble. Compact geometries derive this offset from their
/// value-slot count and put the call area after the homes.
pub const DISPATCH_KEY_OFS: u64 = MIN_FRAME_BYTES as u64;
const _: () = assert!(HOME_SLOT_BASE == DISPATCH_KEY_OFS + SLOT_SIZE);

fn mem64(offset: u64) -> MemArg {
    MemArg {
        offset,
        align: 3,
        memory_index: 0,
    }
}

fn mem32(offset: u64) -> MemArg {
    memarg(offset, 2)
}

fn memarg(offset: u64, align: u32) -> MemArg {
    MemArg {
        offset,
        align,
        memory_index: 0,
    }
}

/// A small lookbehind buffer for local wasm instruction folds.
///
/// Every instruction method used by this emitter is spelled out below.  In
/// particular, this type deliberately does not implement `Deref`: reaching
/// the underlying sink without flushing would reorder pending instructions.
/// What `end` closes. Only `if` splits the straight-line `jf_gcmap` fact:
/// an arm may store, and the join must keep the value from before the `if`.
enum GcmapCtrl {
    Block,
    /// Loop header is a back-edge target, so the body cannot inherit a
    /// pre-loop map. The fall-through after `end` cannot either.
    Loop,
    /// `jf_gcmap` value known on every path that entered this `if`.
    If(i64),
}

struct PeepSink<'sink, 'buf> {
    sink: &'sink mut InstructionSink<'buf>,
    pending: Vec<PendingInstruction>,
    /// `local 0 - FIRST_ITEM_OFFSET`, the jitframe base. `0` means this
    /// function has no header (`compute_home_gcmap` is off) and must not
    /// write `jf_gcmap`.
    gcmap_frame_local: u32,
    /// Pointer last stored to `jf_gcmap` on the straight-line path.
    /// `i64::MIN` means the next push must store.
    gcmap_known: i64,
    gcmap_ctrl: Vec<GcmapCtrl>,
    /// `gc_ll_descr.write_barrier_descr` as `_reload_frame_if_necessary`
    /// reads it; `None` when the module has no collector barrier or no
    /// collecting site.
    frame_wb: Option<FrameWriteBarrier>,
}

/// The jitframe barrier operands: `wasm_jit_write_barrier`, its
/// `(i64) -> i64` residual type, and the flag byte it tests.
#[derive(Clone, Copy)]
struct FrameWriteBarrier {
    fn_ptr: i64,
    type_idx: u32,
    flag_byteofs: i32,
    if_flag: u8,
    /// Function index of the outlined body. `None` emits the body here,
    /// which is also how that function is built.
    helper: Option<u32>,
}

#[derive(Clone, Copy)]
enum PendingInstruction {
    LocalSet(u32),
    I64Const(i64),
    I32Const(i32),
    I64ExtendI32U,
    I64ExtendI32S,
}

macro_rules! forward_zero {
    ($($method:ident),* $(,)?) => {
        $(
            fn $method(&mut self) -> &mut Self {
                self.flush();
                self.sink.$method();
                self
            }
        )*
    };
}

macro_rules! forward_one {
    ($($method:ident($arg:ident: $ty:ty)),* $(,)?) => {
        $(
            fn $method(&mut self, $arg: $ty) -> &mut Self {
                self.flush();
                self.sink.$method($arg);
                self
            }
        )*
    };
}

macro_rules! forward_two {
    ($($method:ident($first:ident: $first_ty:ty, $second:ident: $second_ty:ty)),* $(,)?) => {
        $(
            fn $method(&mut self, $first: $first_ty, $second: $second_ty) -> &mut Self {
                self.flush();
                self.sink.$method($first, $second);
                self
            }
        )*
    };
}

#[allow(dead_code)]
impl<'sink, 'buf> PeepSink<'sink, 'buf> {
    fn new(sink: &'sink mut InstructionSink<'buf>) -> Self {
        Self {
            sink,
            pending: Vec::with_capacity(2),
            gcmap_frame_local: 0,
            gcmap_known: i64::MIN,
            gcmap_ctrl: Vec::new(),
            frame_wb: None,
        }
    }

    /// Items base is in local 0. Refresh the jitframe base local from it.
    fn sync_gcmap_frame(&mut self) {
        let frame = self.gcmap_frame_local;
        if frame == 0 {
            return;
        }
        self.local_get(0);
        self.i32_const(majit_backend::jitframe::FIRST_ITEM_OFFSET as i32);
        self.i32_sub();
        self.local_set(frame);
    }

    fn gcmap_tracking(&self) -> bool {
        self.gcmap_frame_local != 0
    }

    /// Commit every buffered instruction in program order.
    fn flush(&mut self) {
        for instruction in self.pending.drain(..) {
            match instruction {
                PendingInstruction::LocalSet(local) => {
                    self.sink.local_set(local);
                }
                PendingInstruction::I64Const(value) => {
                    self.sink.i64_const(value);
                }
                PendingInstruction::I32Const(value) => {
                    self.sink.i32_const(value);
                }
                PendingInstruction::I64ExtendI32U => {
                    self.sink.i64_extend_i32_u();
                }
                PendingInstruction::I64ExtendI32S => {
                    self.sink.i64_extend_i32_s();
                }
            }
        }
    }

    fn local_set(&mut self, local: u32) -> &mut Self {
        self.flush();
        self.pending.push(PendingInstruction::LocalSet(local));
        self
    }

    fn local_get(&mut self, local: u32) -> &mut Self {
        if matches!(self.pending.last(), Some(PendingInstruction::LocalSet(previous)) if *previous == local)
        {
            self.pending.pop();
            self.flush();
            self.sink.local_tee(local);
        } else {
            self.flush();
            self.sink.local_get(local);
        }
        self
    }

    fn i64_const(&mut self, value: i64) -> &mut Self {
        if !matches!(self.pending.as_slice(), [PendingInstruction::I64Const(_)]) {
            self.flush();
        }
        self.pending.push(PendingInstruction::I64Const(value));
        self
    }

    /// Fold two adjacent i64 constants, or remove an identity constant that
    /// is the right operand of a value already committed to the sink.
    fn fold_i64_binary(
        &mut self,
        right_identity: Option<i64>,
        fold: impl FnOnce(i64, i64) -> i64,
    ) -> bool {
        if let [
            PendingInstruction::I64Const(lhs),
            PendingInstruction::I64Const(rhs),
        ] = self.pending.as_slice()
        {
            let value = fold(*lhs, *rhs);
            self.pending.clear();
            self.pending.push(PendingInstruction::I64Const(value));
            true
        } else if let [PendingInstruction::I64Const(rhs)] = self.pending.as_slice()
            && right_identity == Some(*rhs)
        {
            self.pending.clear();
            true
        } else {
            false
        }
    }

    fn i64_add(&mut self) -> &mut Self {
        if !self.fold_i64_binary(Some(0), i64::wrapping_add) {
            self.flush();
            self.sink.i64_add();
        }
        self
    }

    fn i64_sub(&mut self) -> &mut Self {
        if !self.fold_i64_binary(Some(0), i64::wrapping_sub) {
            self.flush();
            self.sink.i64_sub();
        }
        self
    }

    fn i64_mul(&mut self) -> &mut Self {
        if !self.fold_i64_binary(Some(1), i64::wrapping_mul) {
            self.flush();
            self.sink.i64_mul();
        }
        self
    }

    fn i64_and(&mut self) -> &mut Self {
        if !self.fold_i64_binary(Some(-1), |lhs, rhs| lhs & rhs) {
            self.flush();
            self.sink.i64_and();
        }
        self
    }

    fn i64_or(&mut self) -> &mut Self {
        if !self.fold_i64_binary(Some(0), |lhs, rhs| lhs | rhs) {
            self.flush();
            self.sink.i64_or();
        }
        self
    }

    fn i64_xor(&mut self) -> &mut Self {
        if !self.fold_i64_binary(Some(0), |lhs, rhs| lhs ^ rhs) {
            self.flush();
            self.sink.i64_xor();
        }
        self
    }

    fn i64_shl(&mut self) -> &mut Self {
        if !self.fold_i64_binary(Some(0), |lhs, rhs| lhs.wrapping_shl(rhs as u32 & 63)) {
            self.flush();
            self.sink.i64_shl();
        }
        self
    }

    fn i64_shr_s(&mut self) -> &mut Self {
        if !self.fold_i64_binary(Some(0), |lhs, rhs| lhs >> (rhs as u32 & 63)) {
            self.flush();
            self.sink.i64_shr_s();
        }
        self
    }

    fn i64_shr_u(&mut self) -> &mut Self {
        if !self.fold_i64_binary(Some(0), |lhs, rhs| {
            ((lhs as u64) >> (rhs as u32 & 63)) as i64
        }) {
            self.flush();
            self.sink.i64_shr_u();
        }
        self
    }

    fn i32_wrap_i64(&mut self) -> &mut Self {
        // Inspect the tail before removing it, the way every other fold here
        // does: `pending` is a lookbehind buffer, not an operand stack, so a
        // tail this fold does not consume still owes its instruction to
        // `flush`.
        match self.pending.last().copied() {
            Some(PendingInstruction::I64Const(value)) => {
                self.pending.pop();
                self.pending
                    .push(PendingInstruction::I32Const(value as u64 as u32 as i32));
            }
            Some(PendingInstruction::I64ExtendI32U) | Some(PendingInstruction::I64ExtendI32S) => {
                // wrap(extend_{u,s}(x)) == x for an i32 address already on
                // the stack.
                self.pending.pop();
            }
            _ => {
                self.flush();
                self.sink.i32_wrap_i64();
            }
        }
        self
    }

    fn i64_extend_i32_u(&mut self) -> &mut Self {
        if let Some(PendingInstruction::I32Const(value)) = self.pending.last().copied() {
            self.pending.pop();
            self.pending
                .push(PendingInstruction::I64Const(value as u32 as u64 as i64));
        } else {
            self.flush();
            self.pending.push(PendingInstruction::I64ExtendI32U);
        }
        self
    }

    fn i64_extend_i32_s(&mut self) -> &mut Self {
        if let Some(PendingInstruction::I32Const(value)) = self.pending.last().copied() {
            self.pending.pop();
            self.pending
                .push(PendingInstruction::I64Const(value as i64));
        } else {
            self.flush();
            self.pending.push(PendingInstruction::I64ExtendI32S);
        }
        self
    }

    fn i32_const(&mut self, value: i32) -> &mut Self {
        if !matches!(self.pending.as_slice(), [PendingInstruction::I32Const(_)]) {
            self.flush();
        }
        self.pending.push(PendingInstruction::I32Const(value));
        self
    }

    fn fold_i32_binary(
        &mut self,
        right_identity: Option<i32>,
        fold: impl FnOnce(i32, i32) -> i32,
    ) -> bool {
        if let [
            PendingInstruction::I32Const(lhs),
            PendingInstruction::I32Const(rhs),
        ] = self.pending.as_slice()
        {
            let value = fold(*lhs, *rhs);
            self.pending.clear();
            self.pending.push(PendingInstruction::I32Const(value));
            true
        } else if let [PendingInstruction::I32Const(rhs)] = self.pending.as_slice()
            && right_identity == Some(*rhs)
        {
            self.pending.clear();
            true
        } else {
            false
        }
    }

    fn i32_mul(&mut self) -> &mut Self {
        if !self.fold_i32_binary(Some(1), i32::wrapping_mul) {
            self.flush();
            self.sink.i32_mul();
        }
        self
    }

    fn i32_sub(&mut self) -> &mut Self {
        if !self.fold_i32_binary(Some(0), i32::wrapping_sub) {
            self.flush();
            self.sink.i32_sub();
        }
        self
    }

    fn i32_and(&mut self) -> &mut Self {
        if !self.fold_i32_binary(Some(-1), |lhs, rhs| lhs & rhs) {
            self.flush();
            self.sink.i32_and();
        }
        self
    }

    fn i32_or(&mut self) -> &mut Self {
        if !self.fold_i32_binary(Some(0), |lhs, rhs| lhs | rhs) {
            self.flush();
            self.sink.i32_or();
        }
        self
    }

    fn i32_xor(&mut self) -> &mut Self {
        if !self.fold_i32_binary(Some(0), |lhs, rhs| lhs ^ rhs) {
            self.flush();
            self.sink.i32_xor();
        }
        self
    }

    fn i32_shl(&mut self) -> &mut Self {
        if !self.fold_i32_binary(Some(0), |lhs, rhs| lhs.wrapping_shl(rhs as u32 & 31)) {
            self.flush();
            self.sink.i32_shl();
        }
        self
    }

    fn i32_shr_u(&mut self) -> &mut Self {
        if !self.fold_i32_binary(Some(0), |lhs, rhs| {
            ((lhs as u32) >> (rhs as u32 & 31)) as i32
        }) {
            self.flush();
            self.sink.i32_shr_u();
        }
        self
    }

    fn i32_add(&mut self) -> &mut Self {
        if !self.fold_i32_binary(Some(0), i32::wrapping_add) {
            self.flush();
            self.sink.i32_add();
        }
        self
    }

    fn br_table<V: IntoIterator<Item = u32>>(&mut self, labels: V, default: u32) -> &mut Self
    where
        V::IntoIter: ExactSizeIterator,
    {
        self.flush();
        self.sink.br_table(labels, default);
        self
    }

    fn else_(&mut self) -> &mut Self {
        self.flush();
        if self.gcmap_tracking()
            && let Some(GcmapCtrl::If(saved)) = self.gcmap_ctrl.last()
        {
            self.gcmap_known = *saved;
        }
        self.sink.else_();
        self
    }

    fn end(&mut self) -> &mut Self {
        self.flush();
        if self.gcmap_tracking() {
            match self.gcmap_ctrl.pop() {
                // The join runs whether or not the arm stored, so the
                // pre-`if` fact does not survive. Arms still see it on entry.
                Some(GcmapCtrl::If(_)) => self.gcmap_known = i64::MIN,
                // `br` lands at the end of a block, and a loop's back edge
                // lands at its header. Neither path has the body's stores.
                Some(GcmapCtrl::Loop | GcmapCtrl::Block) => self.gcmap_known = i64::MIN,
                None => {}
            }
        }
        self.sink.end();
        self
    }

    fn block(&mut self, block_type: BlockType) -> &mut Self {
        self.flush();
        if self.gcmap_tracking() {
            self.gcmap_ctrl.push(GcmapCtrl::Block);
        }
        self.sink.block(block_type);
        self
    }

    fn if_(&mut self, block_type: BlockType) -> &mut Self {
        self.flush();
        if self.gcmap_tracking() {
            self.gcmap_ctrl.push(GcmapCtrl::If(self.gcmap_known));
        }
        self.sink.if_(block_type);
        self
    }

    fn loop_(&mut self, block_type: BlockType) -> &mut Self {
        self.flush();
        if self.gcmap_tracking() {
            // Back-edge target: a store before the loop does not dominate.
            self.gcmap_known = i64::MIN;
            self.gcmap_ctrl.push(GcmapCtrl::Loop);
        }
        self.sink.loop_(block_type);
        self
    }

    forward_zero!(
        drop,
        f64_abs,
        f64_add,
        f64_convert_i64_s,
        f64_div,
        f64_eq,
        f64_floor,
        f64_ge,
        f64_gt,
        f64_le,
        f64_lt,
        f64_mul,
        f64_ne,
        f64_neg,
        f64_sqrt,
        f64_reinterpret_i64,
        f64_sub,
        f32_demote_f64,
        f32_reinterpret_i32,
        f64_promote_f32,
        i32_eq,
        i32_eqz,
        i32_gt_u,
        i32_lt_u,
        i32_ne,
        i64_div_s,
        i64_eq,
        i64_eqz,
        i64_extend32_s,
        i64_ge_s,
        i64_ge_u,
        i64_gt_s,
        i64_gt_u,
        i64_le_s,
        i64_le_u,
        i64_lt_s,
        i64_lt_u,
        i64_ne,
        i32_reinterpret_f32,
        i64_reinterpret_f64,
        i64_rem_s,
        i64_trunc_sat_f64_s,
        return_,
        select,
        unreachable,
    );

    forward_one!(
        br(label: u32),
        br_if(label: u32),
        call(function: u32),
        f32_load(memarg: MemArg),
        f32_store(memarg: MemArg),
        f64_load(memarg: MemArg),
        f64_store(memarg: MemArg),
        i32_load(memarg: MemArg),
        i64_load16_s(memarg: MemArg),
        i64_load16_u(memarg: MemArg),
        i64_load32_s(memarg: MemArg),
        i64_load32_u(memarg: MemArg),
        i64_load8_s(memarg: MemArg),
        i64_load8_u(memarg: MemArg),
        i64_store16(memarg: MemArg),
        i64_store32(memarg: MemArg),
        i64_store8(memarg: MemArg),
        i32_load16_s(memarg: MemArg),
        i32_load16_u(memarg: MemArg),
        i32_load8_s(memarg: MemArg),
        i32_load8_u(memarg: MemArg),
        i32_store(memarg: MemArg),
        i32_store16(memarg: MemArg),
        i32_store8(memarg: MemArg),
        i64_load(memarg: MemArg),
        i64_store(memarg: MemArg),
        local_tee(local: u32),
        memory_fill(mem: u32),
        return_call(function: u32),
    );

    forward_two!(
        call_indirect(table_index: u32, type_index: u32),
        return_call_indirect(table_index: u32, type_index: u32),
        memory_copy(dst_mem: u32, src_mem: u32),
    );
}

impl Drop for PeepSink<'_, '_> {
    fn drop(&mut self) {
        self.flush();
    }
}

fn runtime_addr(get: fn() -> usize) -> i32 {
    get() as i32
}

fn emit_call_area_addr(sink: &mut PeepSink<'_, '_>) {
    sink.i32_const(runtime_addr(crate::jit_call_area_addr));
}

/// Invoke the residual-call trampoline, which reads its scratch at
/// `base + offset`. The scratch no longer lives in the frame, so the pair is
/// always the static call area at offset zero, and the base-only import — whose
/// host side adds a baked `CALL_RESULT_OFS` — can no longer be used.
/// Call a `(i64 save_err)->i64` errno helper (`write_real_errno` /
/// `read_real_errno`) through the residual type family at `base`.
fn emit_errno_helper_call(sink: &mut PeepSink<'_, '_>, base: u32, save_err: i64, fn_ptr: i64) {
    sink.i64_const(save_err);
    sink.i32_const(fn_ptr as i32);
    sink.call_indirect(0, base + 1);
    sink.drop();
}

fn emit_jit_call(sink: &mut PeepSink<'_, '_>, jit_call_idx: u32) {
    emit_call_area_addr(sink);
    sink.i32_const(0);
    sink.call(jit_call_idx);
}

/// `CallDescr.result_size`. A missing descr is a void residual (`0`).
fn call_descr_result_facts(op: &Op) -> i64 {
    let Some(descr) = op.getdescr() else {
        return 0;
    };
    match descr.as_call_descr() {
        Some(cd) => cd.result_size() as i64,
        None => 0,
    }
}

fn emit_store_call_result_facts(sink: &mut PeepSink<'_, '_>, result_size: i64) {
    emit_call_area_addr(sink);
    sink.i64_const(result_size);
    sink.i64_store(mem64(STATIC_CALL_RESULT_SIZE_OFS));
}

/// `assembler.py` raw call through the host trampoline. `home` is the result
/// value to fill; `None` drops the result the way `COND_CALL_N` does.
fn emit_residual_trampoline_call(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    jit_call: u32,
    func: OpRef,
    call_args: &[OpRef],
    site_gcmap: &[i64],
    op_idx: usize,
    op: &Op,
    home: Option<u32>,
) -> Result<(), BackendError> {
    if call_args.len() > MAX_CALL_ARGS {
        return Err(BackendError::Unsupported(format!(
            "wasm codegen: residual call has {} arguments; the call area holds {MAX_CALL_ARGS}",
            call_args.len()
        )));
    }
    emit_call_area_addr(sink);
    emit_resolve(sink, constants, value_types, func);
    sink.i64_store(mem64(STATIC_CALL_FUNC_OFS));
    emit_call_area_addr(sink);
    sink.i64_const(call_args.len() as i64);
    sink.i64_store(mem64(STATIC_CALL_NARGS_OFS));
    for (i, arg) in call_args.iter().enumerate() {
        emit_call_area_addr(sink);
        emit_resolve(sink, constants, value_types, *arg);
        sink.i64_store(mem64(STATIC_CALL_ARGS_OFS + i as u64 * SLOT_SIZE));
    }
    emit_store_call_result_facts(sink, call_descr_result_facts(op));
    emit_push_site(sink, site_gcmap, op_idx);
    emit_jit_call(sink, jit_call);
    if let Some(vi) = home {
        emit_call_area_addr(sink);
        sink.i64_load(mem64(STATIC_CALL_RESULT_OFS));
        if value_types.ty(vi) == ValType::F64 {
            sink.f64_reinterpret_i64();
        } else {
            sign_extend_trampolined_int(sink, op);
        }
        sink.local_set(value_types.local(vi));
    }
    Ok(())
}

/// Emit a width-correct integer load. The element address (i32) must be on
/// the stack; the result is an i64, sign- or zero-extended from `size`
/// bytes. Word-sized fields are 4 bytes on wasm32 (`isize`/`usize`/pointer),
/// 8 bytes on 64-bit; reading a fixed 8 bytes here would fold in the next
/// field's bytes on wasm32.
fn emit_sized_int_load(sink: &mut PeepSink<'_, '_>, offset: u64, size: usize, signed: bool) {
    // The i64 family loads and extends in one instruction, so the widening
    // never has to be spelled separately.
    match (size, signed) {
        (4, true) => sink.i64_load32_s(memarg(offset, 2)),
        (4, false) => sink.i64_load32_u(memarg(offset, 2)),
        (2, true) => sink.i64_load16_s(memarg(offset, 1)),
        (2, false) => sink.i64_load16_u(memarg(offset, 1)),
        (1, true) => sink.i64_load8_s(memarg(offset, 0)),
        (1, false) => sink.i64_load8_u(memarg(offset, 0)),
        _ => sink.i64_load(mem64(offset)),
    };
}

/// Emit a width-correct integer store. The stack must hold
/// `[addr_i32, value_i64]`; the low `size` bytes of the value are stored.
/// A fixed 8-byte store would clobber the adjacent field/item (or run past
/// the array end) for word-sized fields and pointer array items on wasm32.
fn emit_sized_int_store(sink: &mut PeepSink<'_, '_>, offset: u64, size: usize) {
    // The i64 family truncates as it stores, so the narrowing never has to be
    // spelled separately.
    match size {
        4 => sink.i64_store32(memarg(offset, 2)),
        2 => sink.i64_store16(memarg(offset, 1)),
        1 => sink.i64_store8(memarg(offset, 0)),
        _ => sink.i64_store(mem64(offset)),
    };
}

/// A GCREF is a machine word. The rewriter's size immediate can still be
/// 8 when a descr was built with host `sizeof(void*)`; an `i64` load/store
/// would then fold the next field (or the GC header of the next object)
/// into the pointer on wasm32.
fn wasm_ref_access_size(is_ref: bool, size: usize) -> usize {
    if is_ref {
        std::mem::size_of::<usize>()
    } else {
        size
    }
}

fn value_is_f64(value_types: &ValueLocals, val: OpRef) -> bool {
    if val.is_constant() {
        return val.ty() == Some(Type::Float);
    }
    value_types.ty(val.raw()) == ValType::F64
}

fn emit_float_store(
    sink: &mut PeepSink<'_, '_>,
    offset: u64,
    size: usize,
) -> Result<(), BackendError> {
    // `get_type_flag`: only `lltype.Float` is FLAG_FLOAT, and it is 8 bytes.
    // A singlefloat store is an integer store, not `f32.demote` + `f32.store`.
    if size != 8 {
        return Err(BackendError::Unsupported(format!(
            "wasm codegen: float store has size {size} (get_type_flag: lltype.Float is 8 bytes)"
        )));
    }
    sink.f64_store(mem64(offset));
    Ok(())
}

/// Address the GC rewrite's descriptor-free `base + offset` memory form.
/// A non-negative constant that fits wasm's unsigned `MemArg` displacement is
/// returned to the caller; dynamic and negative offsets are folded into the
/// i32 address with wasm32 wrapping semantics.
fn emit_gc_offset_addr(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    base: OpRef,
    offset: OpRef,
) -> u64 {
    emit_resolve(sink, constants, value_types, base);
    sink.i32_wrap_i64();
    if let Some(offset) = const_operand_value(constants, offset) {
        if let Ok(offset) = u32::try_from(offset) {
            return offset as u64;
        }
        sink.i32_const(offset as i32);
    } else {
        emit_resolve(sink, constants, value_types, offset);
        sink.i32_wrap_i64();
    }
    sink.i32_add();
    0
}

/// Address the GC rewrite's `base + index * scale + offset` form.  Scale and
/// base offset are rewriter immediates; the index itself remains a runtime
/// value.  Negative offsets (notably accesses into the GC header) cannot use a
/// wasm `MemArg`, so they are added to the address explicitly.
fn emit_gc_indexed_addr(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    op: &Op,
    scale_arg: usize,
    offset_arg: usize,
) -> Result<u64, BackendError> {
    let scale = const_operand_value(constants, op.arg(scale_arg).to_opref()).ok_or_else(|| {
        BackendError::RetryableUnsupported(format!(
            "wasm codegen: {:?} scale is not constant",
            op.opcode
        ))
    })?;
    let scale = u64::try_from(scale).map_err(|_| {
        BackendError::Unsupported(format!(
            "wasm codegen: {:?} has negative scale {scale}",
            op.opcode
        ))
    })?;
    let offset =
        const_operand_value(constants, op.arg(offset_arg).to_opref()).ok_or_else(|| {
            BackendError::RetryableUnsupported(format!(
                "wasm codegen: {:?} base offset is not constant",
                op.opcode
            ))
        })?;

    emit_resolve(sink, constants, value_types, op.arg(0).to_opref());
    sink.i32_wrap_i64();
    emit_resolve(sink, constants, value_types, op.arg(1).to_opref());
    sink.i32_wrap_i64();
    emit_scale_index(sink, scale);
    sink.i32_add();
    if let Ok(offset) = u32::try_from(offset) {
        return Ok(offset as u64);
    }
    sink.i32_const(offset as i32);
    sink.i32_add();
    Ok(0)
}

fn gc_rewrite_access_size(
    op: &Op,
    constants: &indexmap::IndexMap<u32, i64>,
    arg: usize,
) -> Result<(usize, bool), BackendError> {
    let encoded = const_operand_value(constants, op.arg(arg).to_opref()).ok_or_else(|| {
        BackendError::RetryableUnsupported(format!(
            "wasm codegen: {:?} item size is not constant",
            op.opcode
        ))
    })?;
    let size = encoded.unsigned_abs() as usize;
    if !matches!(size, 1 | 2 | 4 | 8) {
        return Err(BackendError::Unsupported(format!(
            "wasm codegen: {:?} has unsupported item size {encoded}",
            op.opcode
        )));
    }
    Ok((size, encoded < 0))
}

/// Dense census of every non-constant Ref-typed value (input arg / op result),
/// independent of whether it needs a home slot. Write-barrier selection still
/// needs the full Ref type set after homes are shrunk to only values live across
/// collecting calls.
struct RefValues {
    /// `Vec<bool>` is a wasteful container in general, but justified here: the
    /// set is built and dropped within one `build_wasm_module` call, sized to
    /// the trace's value count (tens to low hundreds), and only ever
    /// point-queried. At that size a direct byte index beats a bitset's
    /// shift/mask, and the workspace pulls in no bitset crate; it matches the
    /// backend's other id-indexed flag vectors (`label_resume_safety`,
    /// `failguard`).
    by_id: Vec<bool>,
}

impl RefValues {
    fn mark(by_id: &mut Vec<bool>, id: u32) {
        let i = id as usize;
        if i >= by_id.len() {
            by_id.resize(i + 1, false);
        }
        by_id[i] = true;
    }

    fn collect(inputargs: &[InputArgRc], ops: &[Op]) -> Self {
        let mut by_id = Vec::new();
        for ia in inputargs {
            if ia.tp.get() == Type::Ref {
                Self::mark(&mut by_id, ia.index);
            }
        }
        for op in ops {
            if let Some(id) = result_value_raw(op) {
                if op.result_type() == Type::Ref {
                    Self::mark(&mut by_id, id);
                }
            }
        }
        Self { by_id }
    }

    fn contains(&self, v: OpRef) -> bool {
        value_box_raw(v)
            .and_then(|id| self.by_id.get(id as usize).copied())
            .unwrap_or(false)
    }
}

/// Maps each homed Ref-typed value (input arg / op result) to a compact
/// home-slot index `0..len`, where its current `GcRef` is mirrored into the
/// frame's GC-root region (`HOME_SLOT_BASE + home * 8`) so a collecting
/// allocation inside the trace can forward it.
///
/// Keyed by value id (`OpRef::raw()` / input `index`), which is the dense
/// `[0, num_vars)` value-id space; a flat vector indexed by that id is the
/// natural fit — no hashing, and iteration is
/// in id order, so the emitted module stays deterministic without sorting. The
/// `is_constant` guard lives in one place (`home`): a constant `raw()` is a
/// distinct namespace that must never alias a value's home.
struct RefHomes {
    /// `by_id[raw] = home index`, or `NONE` where the value is not a Ref home.
    /// Sized to the last Ref id; queries for higher ids miss via `get`.
    by_id: Vec<u32>,
    len: usize,
}

impl RefHomes {
    const NONE: u32 = u32::MAX;

    fn assign(by_id: &mut Vec<u32>, next: &mut u32, id: u32) {
        let i = id as usize;
        if i >= by_id.len() {
            by_id.resize(i + 1, Self::NONE);
        }
        if by_id[i] == Self::NONE {
            by_id[i] = *next;
            *next += 1;
        }
    }

    fn collect(
        inputargs: &[InputArgRc],
        ops: &[Op],
        include_ca_collects: bool,
        forced_refs: &[OpRef],
        regions: &[InlinedRegionSpan],
    ) -> Self {
        let liveness = HomeLiveness::collect_with_regions(inputargs, ops, regions);
        let collect_positions = collecting_call_positions(ops, include_ca_collects);
        let ref_values = RefValues::collect(inputargs, ops);
        let mut by_id = Vec::new();
        let mut next = 0u32;
        for ia in inputargs {
            if ia.tp.get() == Type::Ref && liveness.live_across_any(ia.index, &collect_positions) {
                Self::assign(&mut by_id, &mut next, ia.index);
            }
        }
        for op in ops {
            if let Some(id) = result_value_raw(op) {
                if op.result_type() == Type::Ref && liveness.live_across_any(id, &collect_positions)
                {
                    Self::assign(&mut by_id, &mut next, id);
                }
            }
        }
        if include_ca_collects {
            // The CA arm allocates its callee frame before it resolves this
            // CALL_ASSEMBLER's arguments. Those Ref operands are used at (not
            // after) this op, so ordinary `live_across` deliberately excludes
            // them; they nevertheless need homes through the prior allocation.
            for op in ops.iter().filter(|op| op.opcode.is_call_assembler()) {
                for arg in op.getarglist() {
                    let arg = arg.to_opref();
                    if ref_values.contains(arg) {
                        Self::assign(&mut by_id, &mut next, arg.raw());
                    }
                }
            }
        }
        // `store_force_descr` publishes the bracketing guard's fail arguments
        // into the frame and leaves the bracket armed past the op, so a force
        // arriving later is what reads them; x86 keeps that guard's gcmap as
        // `finish_gcmap` for the same reason.  Ordinary liveness stops at the
        // guard — nothing consumes them after it — so a Ref that crosses no
        // collecting call would take no home and `emit_force_arm` would publish
        // its raw pointer into the untraced exit slots.  Give every one of them
        // a traced home to name instead.
        for op in ops
            .iter()
            .filter(|op| matches!(op.opcode, OpCode::GuardNotForced | OpCode::GuardNotForced2))
        {
            for arg in exit_fail_args(op) {
                if ref_values.contains(arg) {
                    Self::assign(&mut by_id, &mut next, arg.raw());
                }
            }
        }
        // Resume-at-LABEL Ref captures must also have an ordinary home.  The
        // high capture slot preserves the value while another bridge executes
        // on this frame; the ordinary home participates in the existing
        // post-collection local reload machinery once the target resumes.
        for &r in forced_refs {
            if ref_values.contains(r) {
                Self::assign(&mut by_id, &mut next, r.raw());
            }
        }
        // Same JUMP→LABEL coloring as ValueLocals: one home for the
        // coalesced pair so store-on-def of the def already updates
        // the slot the next iteration reloads.
        for (jid, lid) in jump_phi_coalesce_pairs(ops) {
            let j = jid as usize;
            let l = lid as usize;
            let jh = by_id.get(j).copied().unwrap_or(Self::NONE);
            let lh = by_id.get(l).copied().unwrap_or(Self::NONE);
            match (jh != Self::NONE, lh != Self::NONE) {
                (true, true) => {
                    if j < by_id.len() {
                        by_id[j] = lh;
                    }
                }
                (false, true) => {
                    if j >= by_id.len() {
                        by_id.resize(j + 1, Self::NONE);
                    }
                    by_id[j] = lh;
                }
                (true, false) => {
                    if l >= by_id.len() {
                        by_id.resize(l + 1, Self::NONE);
                    }
                    by_id[l] = jh;
                }
                (false, false) => {}
            }
        }
        RefHomes {
            by_id,
            len: next as usize,
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    /// Home index of value id `id` (caller guarantees it is a value, not a
    /// constant — e.g. an input-arg index).
    fn home_id(&self, id: u32) -> Option<u32> {
        match self.by_id.get(id as usize) {
            Some(&h) if h != Self::NONE => Some(h),
            _ => None,
        }
    }

    /// Home index of `v`, or `None` if it is a constant, a void op, or not a Ref home.
    fn home(&self, v: OpRef) -> Option<u32> {
        value_box_raw(v).and_then(|id| self.home_id(id))
    }

    /// `(value id, home index)` pairs in id order (deterministic).
    fn iter(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.by_id
            .iter()
            .copied()
            .enumerate()
            .filter(|&(_, h)| h != Self::NONE)
            .map(|(id, h)| (id as u32, h))
    }
}

#[derive(Clone, Copy)]
enum LabelCaptureStorage {
    /// Absolute frame value-slot index (slot zero is the fail index).
    ValueSlot(usize),
    /// Ordinal within the high, GC-rooted LABEL-capture home region.
    RefSlot(usize),
}

/// Backend-only preservation plan for values that remain live across a peeled
/// LABEL without appearing in that LABEL's semantic argument list.  RPython's
/// assembler keeps such values in the frozen frame; wasm locals disappear on
/// a tail-call re-entry, so we explicitly mirror that storage shape here.
struct LabelResumeData {
    per_label: Vec<Vec<OpRef>>,
    uncapturable: Vec<bool>,
    capture_by_id: Vec<Option<LabelCaptureStorage>>,
    captured_refs: Vec<OpRef>,
    scalar_slots: usize,
    ref_slots: usize,
}

/// Where one inlined bridge region starts in a merged analysis stream, and
/// which value ids carry that region's own live-ins.
struct InlinedRegionSpan {
    ops_start: usize,
    inputarg_ids: Vec<u32>,
}

impl InlinedRegionSpan {
    /// The regions occupy the tail of the merged stream in `inlined_bridges`
    /// order, so their starts run back from the end of `ops`. `bridges` must be
    /// the rebased copies the merged stream was built from, so the recorded ids
    /// are the ids that stream reads.
    fn collect(ops_len: usize, bridges: &[InlinedBridge]) -> Vec<Self> {
        let mut start =
            ops_len.saturating_sub(bridges.iter().map(|bridge| bridge.ops.len()).sum::<usize>());
        bridges
            .iter()
            .map(|bridge| {
                let span = Self {
                    ops_start: start,
                    inputarg_ids: bridge.inputargs.iter().map(|ia| ia.index).collect(),
                };
                start += bridge.ops.len();
                span
            })
            .collect()
    }
}

impl LabelResumeData {
    fn collect(inputargs: &[InputArgRc], ops: &[Op]) -> Self {
        Self::collect_with_regions(inputargs, ops, &[], inputargs.len())
    }

    fn collect_with_regions(
        inputargs: &[InputArgRc],
        ops: &[Op],
        regions: &[InlinedRegionSpan],
        entry_arity: usize,
    ) -> Self {
        let (_, num_vars) = collect_guards_and_vars(inputargs, ops);
        let ref_values = RefValues::collect(inputargs, ops);
        let normal_value_slots = normal_frame_value_slots_for(inputargs, ops, entry_arity);
        let mut has_producer = vec![false; num_vars as usize];
        let mut is_input = vec![false; num_vars as usize];
        for ia in inputargs {
            if let Some(v) = is_input.get_mut(ia.index as usize) {
                *v = true;
            }
        }
        for op in ops {
            if let Some(id) = result_value_raw(op)
                && let Some(v) = has_producer.get_mut(id as usize)
            {
                *v = true;
            }
            if op.opcode == OpCode::Label {
                for a in op.getarglist().iter() {
                    let opref = a.to_opref();
                    // Peeled InputArgRef live-ins are real LABEL params.
                    // A producerless RefOp is a residual virtualizable slot.
                    if opref != OpRef::NONE
                        && !opref.is_constant()
                        && !matches!(opref, OpRef::RefOp(_))
                        && let Some(v) = has_producer.get_mut(opref.raw() as usize)
                    {
                        *v = true;
                    }
                }
            }
        }
        let mut per_label = Vec::new();
        let mut uncapturable = Vec::new();

        // Only the labels the entry dispatch can land on need a capture plan;
        // an in-body label is never resumed, so reserving frame slots for its
        // live-ins would only inflate the frozen geometry.
        let resumable = resumable_label_count(ops);
        for (label_pos, label) in ops
            .iter()
            .enumerate()
            .filter(|(_, op)| op.opcode == OpCode::Label)
            .take(resumable)
        {
            let mut available = vec![false; num_vars as usize];
            let mut defined_before = vec![false; num_vars as usize];
            // Producer-less int/float ids are folded constant-pool seeds.
            // Codegen binds them before the entry dispatch, so they dominate
            // both the key-0 path and every LABEL resume and need no frame
            // capture. A producerless LABEL RefOp is a residual
            // virtualizable slot, not a seed — wasm would bind it to a
            // null local. Peeled InputArgRef live-ins stay seeds.
            let mut unbound_label_ref = vec![false; num_vars as usize];
            for op in ops {
                if op.opcode != OpCode::Label {
                    continue;
                }
                for a in op.getarglist() {
                    let opref = a.to_opref();
                    if opref == OpRef::NONE
                        || opref.is_constant()
                        || !matches!(opref, OpRef::RefOp(_))
                    {
                        continue;
                    }
                    let id = opref.raw() as usize;
                    if !has_producer.get(id).copied().unwrap_or(false)
                        && !is_input.get(id).copied().unwrap_or(false)
                        && let Some(v) = unbound_label_ref.get_mut(id)
                    {
                        *v = true;
                    }
                }
            }
            for (id, produced) in has_producer.iter().copied().enumerate() {
                if !produced && !is_input[id] && !unbound_label_ref[id] {
                    available[id] = true;
                    defined_before[id] = true;
                }
            }
            for ia in inputargs {
                if let Some(v) = defined_before.get_mut(ia.index as usize) {
                    *v = true;
                }
            }
            for op in &ops[..label_pos] {
                if let Some(id) = result_value_raw(op)
                    && let Some(v) = defined_before.get_mut(id as usize)
                {
                    *v = true;
                }
            }
            for arg in label.getarglist() {
                if let Some(id) = value_box_raw(arg.to_opref())
                    && let Some(v) = available.get_mut(id as usize)
                {
                    *v = true;
                }
            }
            // An appended region's live-ins reach it only through the
            // guard-fail branch that is the region's sole predecessor, and that
            // branch assigns them. Nothing the entry dispatch can land on
            // reaches a region's first read without passing it, so those ids
            // are dead until written here. Treating them as live would reserve
            // one frozen-frame slot per region live-in at every resumable
            // label, and the resume loader would reload a value the guard
            // overwrites before anything reads it.
            for region in regions {
                if region.ops_start <= label_pos {
                    continue;
                }
                for &id in &region.inputarg_ids {
                    if let Some(v) = available.get_mut(id as usize) {
                        *v = true;
                    }
                }
            }

            let mut missing = Vec::new();
            let mut bad = false;
            // An appended region is not a predecessor of this label's
            // resume. Its body reads would only reserve frozen slots
            // the resume loader never reloads — the guard-fail branch
            // is the region's sole predecessor.
            let scan_end = regions
                .iter()
                .filter(|region| region.ops_start > label_pos)
                .map(|region| region.ops_start)
                .min()
                .unwrap_or(ops.len());
            for op in &ops[label_pos + 1..scan_end] {
                let mut reads: Vec<OpRef> = op.getarglist().iter().map(|a| a.to_opref()).collect();
                if let Some(failargs) = op.getfailargs() {
                    reads.extend(failargs.iter().map(|a| a.to_opref()));
                }
                for r in reads {
                    let Some(id) = value_box_raw(r) else {
                        continue;
                    };
                    let id = id as usize;
                    if !available.get(id).copied().unwrap_or(false) {
                        if !defined_before.get(id).copied().unwrap_or(false) {
                            bad = true;
                            continue;
                        }
                        missing.push(r);
                        if let Some(v) = available.get_mut(id) {
                            *v = true;
                        }
                    }
                }
                if let Some(id) = result_value_raw(op)
                    && let Some(v) = available.get_mut(id as usize)
                {
                    *v = true;
                }
            }
            per_label.push(missing);
            uncapturable.push(bad);
        }

        let mut capture_by_id = vec![None; num_vars as usize];
        let mut captured_refs = Vec::new();
        let mut scalar_slots = 0usize;
        let mut ref_slots = 0usize;
        for &r in per_label.iter().flatten() {
            let id = r.raw() as usize;
            if capture_by_id[id].is_some() {
                continue;
            }
            let storage = if ref_values.contains(r) {
                captured_refs.push(r);
                let slot = LabelCaptureStorage::RefSlot(ref_slots);
                ref_slots += 1;
                slot
            } else {
                let slot = LabelCaptureStorage::ValueSlot(normal_value_slots + scalar_slots);
                scalar_slots += 1;
                slot
            };
            capture_by_id[id] = Some(storage);
        }

        Self {
            per_label,
            uncapturable,
            capture_by_id,
            captured_refs,
            scalar_slots,
            ref_slots,
        }
    }

    fn storage(&self, r: OpRef) -> Option<LabelCaptureStorage> {
        value_box_raw(r).and_then(|id| self.capture_by_id.get(id as usize).copied().flatten())
    }

    fn shortage(&self, frame: FrameGeometry) -> Option<super::FrameShortage> {
        if self.ref_slots > frame.label_ref_slots {
            return Some(super::FrameShortage::new(
                super::FrameShortageKind::LabelResumeRefSlots,
                self.ref_slots,
                frame.label_ref_slots,
            ));
        }
        for storage in self.capture_by_id.iter().flatten() {
            match storage {
                LabelCaptureStorage::ValueSlot(slot) if *slot >= frame.value_slots => {
                    return Some(super::FrameShortage::new(
                        super::FrameShortageKind::LabelResumeCaptureSlots,
                        slot + 1,
                        frame.value_slots,
                    ));
                }
                LabelCaptureStorage::RefSlot(slot) if *slot >= frame.label_ref_slots => {
                    return Some(super::FrameShortage::new(
                        super::FrameShortageKind::LabelResumeCaptureSlots,
                        slot + 1,
                        frame.label_ref_slots,
                    ));
                }
                LabelCaptureStorage::ValueSlot(_) | LabelCaptureStorage::RefSlot(_) => {}
            }
        }
        None
    }

    fn supported_by(&self, frame: FrameGeometry) -> bool {
        self.shortage(frame).is_none()
    }

    fn frame_offset(&self, storage: LabelCaptureStorage, frame: FrameGeometry) -> u64 {
        match storage {
            LabelCaptureStorage::ValueSlot(slot) => frame.capture_value_ofs(slot),
            LabelCaptureStorage::RefSlot(slot) => {
                frame.home_slot_base + (frame.ordinary_home_slots() + slot) as u64 * SLOT_SIZE
            }
        }
    }
}

/// Number of Ref-home slots a trace with these `inputargs`/`ops` reserves,
/// matching the `num_ref_homes` [`build_wasm_module`] returns. Lets a CA-arena
/// caller size the callee frame and the GC walker for a (wider) bridge's home
/// region before codegen runs.
pub fn count_ref_homes(inputargs: &[InputArgRc], ops: &[Op]) -> usize {
    // This pre-sizing query is used for CA bridges before `CaParams` exists, so
    // count CALL_ASSEMBLER as a collecting position to match CA codegen.
    let resume = LabelResumeData::collect(inputargs, ops);
    RefHomes::collect(inputargs, ops, true, &resume.captured_refs, &[]).len()
}

/// Number of high GC-rooted homes reserved exclusively for LABEL live-ins.
pub fn label_ref_capture_slots(inputargs: &[InputArgRc], ops: &[Op]) -> usize {
    LabelResumeData::collect(inputargs, ops).ref_slots
}

/// Mark the homes this module initializes: the used ordinary prefix and the
/// LABEL-capture tail. Frozen geometry reserves extra ordinary slots so a
/// later bridge can fit; those unused reserved words stay unmarked so
/// recycled nursery bytes are not traced. assembler.py writes `jf_gcmap`
/// for live slots only.
pub fn build_home_gcmap(
    frame: FrameGeometry,
    used_ordinary: usize,
    used_labels: usize,
) -> Box<[usize]> {
    let sign = std::mem::size_of::<isize>();
    let bits_per_word = std::mem::size_of::<usize>() * 8;
    let prefix = frame.prefix_ordinary_homes.min(frame.ordinary_home_slots());
    let ordinary = used_ordinary.min(frame.addressable_ordinary_homes());
    let in_prefix = ordinary.min(prefix);
    let label_base = frame.ordinary_home_slots();
    let label_n = used_labels.min(frame.label_ref_slots);
    if ordinary == 0 && label_n == 0 {
        // One empty data word: a non-null jf_gcmap that traces nothing.
        return vec![1usize, 0usize].into_boxed_slice();
    }
    let mut offsets = Vec::new();
    for h in 0..in_prefix {
        offsets.push(frame.home_slot_base as usize + h * 8);
    }
    for h in prefix..ordinary {
        offsets.push(frame.home_ofs(h as u64) as usize);
    }
    for h in label_base..label_base + label_n {
        offsets.push(frame.home_slot_base as usize + h * 8);
    }
    let last_index = offsets.iter().copied().max().unwrap_or(0) / sign;
    let num_words = last_index / bits_per_word + 1;
    let mut buf = vec![0usize; 1 + num_words];
    buf[0] = num_words;
    for offset in offsets {
        let index = offset / sign;
        buf[1 + index / bits_per_word] |= 1usize << (index % bits_per_word);
    }
    buf.into_boxed_slice()
}

/// Item index of ordinary Ref home `home`. Tail homes go through
/// [`FrameGeometry::home_ofs`]; prefix homes stay at `home_slot_base`.
fn home_item_index(frame: FrameGeometry, home: u32) -> u32 {
    let sign = std::mem::size_of::<isize>();
    (frame.home_ofs(home as u64) as usize / sign) as u32
}

/// Intern equal bitmaps so each distinct safepoint map is one parked block.
struct ParkedGcmaps {
    sink: usize,
    cached: Vec<(Box<[usize]>, i64)>,
}

impl ParkedGcmaps {
    fn intern(&mut self, mut indices: Vec<u32>) -> i64 {
        if indices.is_empty() {
            return 0;
        }
        indices.sort_unstable();
        indices.dedup();
        let map = gcmap_for_item_indices(&indices);
        for (bits, ptr) in &self.cached {
            if bits.as_ref() == map.as_ref() {
                return *ptr;
            }
        }
        let ptr = crate::release::park_gcmap_raw(self.sink, map.clone()) as i64;
        self.cached.push((map, ptr));
        ptr
    }
}

/// First op index at which each value id's home has been stored.
///
/// `regalloc.py` `get_gcmap` marks a slot only while a live box is bound
/// there. Built in one pass: an input is stored before op 0, an inlined
/// region's live-in at `region.ops_start` (`emit_guard_inline_bridge_move`),
/// and a producer at `i` from op `i + 1`. A LABEL phi dated at the label is
/// not a store on the key-0 path (`consider_label` spills only when the
/// allocator wrote the slot). "Stored before `at`" is `threshold <= at`.
struct HomeStoreAt {
    /// `i32::MAX` — the home is never stored.
    at: Vec<i32>,
}

impl HomeStoreAt {
    fn build(inputargs: &[InputArgRc], ops: &[Op], regions: &[InlinedRegionSpan]) -> Self {
        let mut at = Vec::new();
        let note = |at: &mut Vec<i32>, id: u32, when: i32| {
            let i = id as usize;
            if i >= at.len() {
                at.resize(i + 1, i32::MAX);
            }
            if at[i] == i32::MAX {
                at[i] = when;
            }
        };
        for ia in inputargs {
            note(&mut at, ia.index, 0);
        }
        // Region live-ins override a function input: the id is not stored on
        // the entry path, only from the region's first op.
        for region in regions {
            let when = i32::try_from(region.ops_start).unwrap_or(i32::MAX);
            for &id in &region.inputarg_ids {
                let i = id as usize;
                if i >= at.len() {
                    at.resize(i + 1, i32::MAX);
                }
                at[i] = when;
            }
        }
        for (i, op) in ops.iter().enumerate() {
            let Some(id) = result_value_raw(op) else {
                continue;
            };
            let when = i32::try_from(i + 1).unwrap_or(i32::MAX);
            note(&mut at, id, when);
        }
        Self { at }
    }

    fn threshold(&self, raw: u32) -> i32 {
        self.at.get(raw as usize).copied().unwrap_or(i32::MAX)
    }

    fn stored_before(&self, raw: u32, at_op: usize) -> bool {
        self.threshold(raw) <= i32::try_from(at_op).unwrap_or(i32::MAX)
    }
}

/// `get_gcmap` marks a frame slot only when a live Ref binding is in it.
/// An inlined region is a separate entry: only the live-in move at
/// `ops_start` and stores inside the region have written the slot.
/// A LABEL does not clear homes. The key-0 path stored them before the
/// `br` over the resume loader, and that loader (plus capture restore)
/// rewrites the same ordinary homes on the resume path.
fn ref_home_stored_for_site(
    raw: u32,
    at: usize,
    store_at: &HomeStoreAt,
    ops: &[Op],
    regions: &[InlinedRegionSpan],
) -> bool {
    if let Some(region) = regions.iter().rev().find(|region| at >= region.ops_start) {
        let end = regions
            .iter()
            .find(|later| later.ops_start > region.ops_start)
            .map(|later| later.ops_start)
            .unwrap_or(ops.len());
        if at < end {
            let t = store_at.threshold(raw);
            return t != i32::MAX && (t as usize) >= region.ops_start && (t as usize) <= at;
        }
    }
    store_at.stored_before(raw, at)
}

/// `(value id, home index)` pairs `get_gcmap` marks at `at`: Ref homes live
/// across the op whose store dominates it. Id order matches `RefHomes::iter`.
/// The parked bitmap and the post-call reload both use this vec, so a reload
/// never reads a slot the map did not trace.
fn site_live_homes(
    ref_homes: &RefHomes,
    liveness: &HomeLiveness,
    store_at: &HomeStoreAt,
    ops: &[Op],
    regions: &[InlinedRegionSpan],
    at: usize,
) -> Vec<(u32, u32)> {
    let mut homes = Vec::new();
    for (raw, h) in ref_homes.iter() {
        if !liveness.live_across(raw, at) {
            continue;
        }
        if !ref_home_stored_for_site(raw, at, store_at, ops, regions) {
            continue;
        }
        homes.push((raw, h));
    }
    homes
}

fn collecting_site(op: &Op) -> bool {
    (op.opcode.is_call() && call_can_collect(op))
        || op.opcode.is_malloc()
        || op.opcode.is_call_assembler()
}

/// One parked pointer per op, plus the `(raw, home)` list that pointer marks.
/// `0` is not a collecting site, or the exact `get_gcmap` set is empty.
/// Identical bitmaps share one pointer. The home list is what
/// [`emit_reload_refs_from_homes`] reloads.
fn build_site_gcmaps(
    frame: FrameGeometry,
    ref_homes: &RefHomes,
    liveness: &HomeLiveness,
    store_at: &HomeStoreAt,
    ops: &[Op],
    regions: &[InlinedRegionSpan],
    sink: usize,
    park: bool,
) -> (Vec<i64>, Vec<Vec<(u32, u32)>>) {
    let mut parked = ParkedGcmaps {
        sink,
        cached: Vec::new(),
    };
    let mut ptrs = Vec::with_capacity(ops.len());
    let mut homes = Vec::with_capacity(ops.len());
    for (at, op) in ops.iter().enumerate() {
        if !collecting_site(op) {
            ptrs.push(0);
            homes.push(Vec::new());
            continue;
        }
        let live = site_live_homes(ref_homes, liveness, store_at, ops, regions, at);
        let ptr = if park {
            let indices = live
                .iter()
                .map(|&(_, h)| home_item_index(frame, h))
                .collect();
            parked.intern(indices)
        } else {
            0
        };
        ptrs.push(ptr);
        homes.push(live);
    }
    (ptrs, homes)
}

/// assembler.py `push_gcmap(..., store=True)`: one `i32.store`/`i64.store`
/// at `jf_gcmap`, addressed from the jitframe-base local. A `call` here
/// would spill every live wasm local. A straight-line path that already
/// holds `ptr` emits nothing; `pop_gcmap` still stores 0.
fn emit_push_gcmap(sink: &mut PeepSink<'_, '_>, ptr: i64) {
    if ptr == 0 || sink.gcmap_frame_local == 0 || sink.gcmap_known == ptr {
        return;
    }
    emit_gcmap_store(sink, ptr);
    sink.gcmap_known = ptr;
}

/// assembler.py `pop_gcmap`: one store of 0. Same no-call constraint as
/// [`emit_push_gcmap`]. Not elided: the call returns with the map still
/// installed, and the next straight-line push must see 0.
fn emit_pop_gcmap(sink: &mut PeepSink<'_, '_>) {
    if sink.gcmap_frame_local == 0 {
        return;
    }
    emit_gcmap_store(sink, 0);
    sink.gcmap_known = 0;
}

fn emit_gcmap_store(sink: &mut PeepSink<'_, '_>, ptr: i64) {
    use majit_backend::jitframe::{JF_GCMAP_OFS, SIZEOFSIGNED};
    sink.local_get(sink.gcmap_frame_local);
    if SIZEOFSIGNED == 4 {
        sink.i32_const(ptr as i32);
        sink.i32_store(memarg(JF_GCMAP_OFS as u64, 2));
    } else {
        sink.i64_const(ptr);
        sink.i64_store(memarg(JF_GCMAP_OFS as u64, 3));
    }
}

fn emit_push_site(sink: &mut PeepSink<'_, '_>, site_gcmap: &[i64], op_idx: usize) {
    let ptr = site_gcmap.get(op_idx).copied().unwrap_or(0);
    emit_push_gcmap(sink, ptr);
}

fn emit_pop_site(sink: &mut PeepSink<'_, '_>, site_gcmap: &[i64], op_idx: usize) {
    if site_gcmap.get(op_idx).copied().unwrap_or(0) != 0 {
        emit_pop_gcmap(sink);
    }
}

/// First free value position — one past the highest id any value reference in
/// the trace occupies (input args, op results, and every op argument, including
/// a folded value the constants pool alone binds).
/// `majit_gc::rewrite::remove_ref_constants` numbers the
/// `LoadFromGcTable` results it emits from here upward, so the operand
/// numbering the optimizer produced stays untouched. Same id set
/// `collect_guards_and_vars` sizes `num_vars` from, so the loads land inside
/// the locals the function declares.
pub fn next_value_pos(inputargs: &[InputArgRc], ops: &[Op]) -> u32 {
    collect_guards_and_vars(inputargs, ops).1
}

/// Positional frame slots required for a token's inputs and guard spills.
/// Slot zero is the fail index; the returned count therefore also gives the
/// first free slot for the call trampoline.
///
/// The GUARD_VALUE counter slot is reserved unconditionally, including for a
/// trace whose own guards spill nothing. A bridge runs in its source token's
/// frame, whose offsets froze when that token was compiled, and `compile_bridge`
/// refuses a bridge whose `frame_value_slots` exceeds `source_frame.value_slots`
/// — a refusal the guard descr makes permanent, so the guard blackholes
/// for the rest of the run. Reserving only when THIS trace spills would let a
/// loop with no GUARD_VALUE freeze a frame one slot too narrow for the first
/// bridge that promotes a value, which is the ordinary way a bridge acquires
/// one. Upstream never faces the question: `regalloc.py prepare_op_guard_value`
/// names a slot in the register save area `_push_all_regs_to_frame` writes at
/// every exit, so a slot always exists and no frame is ever sized for it.
fn normal_frame_value_slots(inputargs: &[InputArgRc], ops: &[Op]) -> usize {
    normal_frame_value_slots_for(inputargs, ops, inputargs.len())
}

/// Widest positional transfer a LABEL resume loader or a JUMP stores through
/// [`FrameGeometry::spill_slot_ofs`]. Entry and fail-arg spills are counted
/// separately.
fn positional_transfer_arity(ops: &[Op]) -> usize {
    ops.iter()
        .filter(|op| matches!(op.opcode, OpCode::Label | OpCode::Jump))
        .map(|op| op.num_args())
        .max()
        .unwrap_or(0)
}

fn normal_frame_value_slots_for(inputargs: &[InputArgRc], ops: &[Op], entry_arity: usize) -> usize {
    let (guards, _) = collect_guards_and_vars(inputargs, ops);
    let max_fail_args = guards
        .iter()
        .map(|g| live_fail_arg_count(g.meta_descr.as_ref(), g.fail_arg_refs.len()))
        .max()
        .unwrap_or(0);
    // `spill_slot_ofs(i)` is `FRAME_SLOT_BASE + i * SLOT_SIZE` in the prefix.
    // The dispatch key starts at `value_slots * SLOT_SIZE`, so arg `i` lands
    // on it when `i + 1 == value_slots`. Reserving the fail index, the
    // widest transfer, and the GUARD_VALUE counter keeps every such store
    // strictly below the key.
    let value_area = max_fail_args
        .max(entry_arity)
        .max(positional_transfer_arity(ops));
    1 + value_area + 1
}

/// The trace-wide GUARD_VALUE counter slot, or `None` when no guard needs one.
///
/// The first slot past the value area every exit writes into, so it is free in
/// every exit's layout, and `normal_frame_value_slots` reserves it.
fn counter_slot(inputargs: &[InputArgRc], ops: &[Op]) -> Option<usize> {
    let (guards, _) = collect_guards_and_vars(inputargs, ops);
    if guards.iter().all(|g| g.counter_value_spill.is_none()) {
        return None;
    }
    let max_fail_args = guards
        .iter()
        .map(|g| live_fail_arg_count(g.meta_descr.as_ref(), g.fail_arg_refs.len()))
        .max()
        .unwrap_or(0);
    Some(max_fail_args.max(inputargs.len()))
}

pub fn frame_value_slots(inputargs: &[InputArgRc], ops: &[Op]) -> usize {
    normal_frame_value_slots(inputargs, ops) + LabelResumeData::collect(inputargs, ops).scalar_slots
}

/// `CondCallGcWb` / `CondCallGcWbArray` are the rewriter's already-decided
/// barrier ops (`rewrite.py gen_write_barrier` / `gen_write_barrier_array`);
/// the stored value is no longer on the op, so the base is `arg(0)`.
fn write_barrier_base(op: &Op, ref_values: &RefValues) -> Option<OpRef> {
    if matches!(op.opcode, OpCode::CondCallGcWb | OpCode::CondCallGcWbArray) {
        return Some(op.arg(0).to_opref());
    }
    let val = op.arg(ref_store_value_arg(op)?).to_opref();
    // `contains` returns false for constants, matching the gate's `not ConstPtr`.
    ref_values.contains(val).then(|| op.arg(0).to_opref())
}
/// `llsupport/gc.py WriteBarrierDescr` as the emitted barrier reads it, paired
/// with the addresses of the two helpers its arms call.
///
/// A zero `cards_set` is the collector saying it has no cards, which is the
/// gate `x86/assembler.py _write_barrier_fastpath` spells as
/// `if array and descr.jit_wb_cards_set`; the dynasm backends read the same
/// field off the same descriptor.
#[derive(Clone, Copy)]
pub struct WriteBarrierHelpers {
    /// `wasm_jit_write_barrier`, the `remember_young_pointer` entry point.
    pub fn_ptr: i64,
    /// `wasm_jit_write_barrier_from_array`, the
    /// `jit_remember_young_pointer_from_array` entry point.
    pub array_fn_ptr: i64,
    /// `jit_wb_if_flag_byteofs`: where the flag byte sits relative to the
    /// object pointer. Negative, because the header precedes the object.
    pub flag_byteofs: i32,
    /// `jit_wb_if_flag_singlebyte`.
    pub if_flag: u8,
    /// `jit_wb_cards_set_singlebyte`.
    pub cards_set: u8,
    /// `jit_wb_card_page_shift`.
    pub card_page_shift: u32,
}

impl WriteBarrierHelpers {
    /// Take the geometry from a collector's descriptor. The addresses are the
    /// backend's own exported helpers, so they are supplied separately.
    pub fn new(fn_ptr: i64, array_fn_ptr: i64, descr: &majit_gc::WriteBarrierDescr) -> Self {
        Self {
            fn_ptr,
            array_fn_ptr,
            flag_byteofs: descr.jit_wb_if_flag_byteofs,
            if_flag: descr.jit_wb_if_flag_singlebyte,
            cards_set: descr.jit_wb_cards_set_singlebyte as u8,
            card_page_shift: descr.jit_wb_card_page_shift,
        }
    }

    /// The geometry the running collector advertises, for a caller that has no
    /// descriptor in hand.
    pub fn for_current_gc(fn_ptr: i64, array_fn_ptr: i64) -> Self {
        Self::new(
            fn_ptr,
            array_fn_ptr,
            &majit_gc::WriteBarrierDescr::for_current_gc(),
        )
    }

    /// The `TEST8` mask: the TRACK_YOUNG_PTRS byte, widened to also catch
    /// CARDS_SET when this store can mark a card, so one test covers both
    /// arms (`_write_barrier_fastpath`: `mask = jit_wb_if_flag_singlebyte |
    /// -0x80`).
    fn flag_mask(&self, card_marking: bool) -> i32 {
        let mask = if card_marking {
            self.if_flag | self.cards_set
        } else {
            self.if_flag
        };
        i32::from(mask)
    }
}

/// Push the barrier flag byte, the operand of `TEST8 [obj + jit_wb_if_flag_byteofs]`.
fn emit_load_wb_flag_byte(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    wb: &WriteBarrierHelpers,
    base_ref: OpRef,
) {
    emit_resolve(sink, constants, value_types, base_ref);
    sink.i32_wrap_i64();
    sink.i32_const(wb.flag_byteofs);
    sink.i32_add();
    sink.i32_load8_u(memarg(0, 0));
}

/// Push `index >> card_page_shift`, the card bit's index.
fn emit_push_card_bitindex(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    wb: &WriteBarrierHelpers,
    index: OpRef,
) {
    emit_resolve(sink, constants, value_types, index);
    sink.i32_wrap_i64();
    if wb.card_page_shift != 0 {
        sink.i32_const(wb.card_page_shift as i32);
        sink.i32_shr_u();
    }
}

/// Push the address `incminimark.py get_card` computes:
/// `obj - HEADER + ~(bitindex >> 3)`.
///
/// `WriteBarrierSlowPath` builds it in the same order — shift, `NOT`, subtract
/// the header, add the base — so that only the last term needs the object.
fn emit_push_card_addr(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    wb: &WriteBarrierHelpers,
    base_ref: OpRef,
    index: OpRef,
) {
    emit_push_card_bitindex(sink, constants, value_types, wb, index);
    sink.i32_const(3);
    sink.i32_shr_u();
    sink.i32_const(-1);
    sink.i32_xor();
    sink.i32_const(majit_gc::header::GcHeader::SIZE as i32);
    sink.i32_sub();
    emit_resolve(sink, constants, value_types, base_ref);
    sink.i32_wrap_i64();
    sink.i32_add();
}

/// `*get_card(obj, bitindex >> 3) |= 1 << (bitindex & 7)`.
///
/// The address is built twice rather than parked in a local: every term is
/// pure arithmetic, which the guest optimizer both folds and commons, while a
/// local would have to be carved out of the `UintMulHigh` scratch pool.
///
/// `remember_young_pointer_from_array2` returns early when the bit is already
/// set; the inlined form ORs unconditionally, exactly as
/// `WriteBarrierSlowPath` does.
fn emit_inline_card_mark(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    wb: &WriteBarrierHelpers,
    base_ref: OpRef,
    index: OpRef,
) {
    emit_push_card_addr(sink, constants, value_types, wb, base_ref, index);
    emit_push_card_addr(sink, constants, value_types, wb, base_ref, index);
    sink.i32_load8_u(memarg(0, 0));
    sink.i32_const(1);
    emit_push_card_bitindex(sink, constants, value_types, wb, index);
    sink.i32_const(7);
    sink.i32_and();
    sink.i32_shl();
    sink.i32_or();
    sink.i32_store8(memarg(0, 0));
}

/// Call a one-arg `(i64)->i64` barrier helper and drop the dummy 0 result.
fn emit_wb_helper_call(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    residual_type_base: u32,
    fn_ptr: i64,
    base_ref: OpRef,
    gcmap_ptr: i64,
    ca_reload_fn_ptr: i64,
    jf_top_addr: Option<u32>,
) {
    // assembler.py `push_gcmap` on the write-barrier slow path. The fast
    // path does not call and does not publish a map.
    emit_push_gcmap(sink, gcmap_ptr);
    emit_resolve(sink, constants, value_types, base_ref);
    sink.i32_const(fn_ptr as i32);
    sink.call_indirect(0, residual_type_base + 1);
    sink.drop();
    if gcmap_ptr != 0 {
        // The helper may move the frame. `pop_gcmap` has to clear the
        // forwarded jitframe, the same order as `_reload_frame_if_necessary`
        // before `pop_gcmap`.
        emit_reload_frame_if_necessary(
            sink,
            Some(residual_type_base),
            ca_reload_fn_ptr,
            jf_top_addr,
        );
        emit_pop_gcmap(sink);
    }
}

/// Emit a write-barrier check on `base_ref` for a rewriter `CondCallGcWb` /
/// `CondCallGcWbArray`.
///
/// `_write_barrier_fastpath`: one test of the flag byte, and only a flagged
/// object enters the body. A `card_index` store then follows
/// `WriteBarrierSlowPath` — CARDS_SET already armed marks the card inline,
/// otherwise `jit_remember_young_pointer_from_array` runs and the same test
/// decides again on its return. A field store calls `wasm_jit_write_barrier`.
///
/// The `(i64)->i64` residual type is declared for every module that contains
/// one of these ops (`direct_helper_i64_arity`). Operand-stack-neutral: every
/// push is consumed by a store, the call, or the result drop.
#[allow(clippy::too_many_arguments)]
fn emit_write_barrier(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    residual_type_base: Option<u32>,
    wb: &WriteBarrierHelpers,
    base_ref: OpRef,
    card_index: Option<OpRef>,
    gcmap_ptr: i64,
    ca_reload_fn_ptr: i64,
    jf_top_addr: Option<u32>,
) -> Result<(), BackendError> {
    let Some(base) = residual_type_base else {
        return Err(BackendError::Unsupported(
            "wasm codegen: write barrier has no residual call type".into(),
        ));
    };
    emit_load_wb_flag_byte(sink, constants, value_types, wb, base_ref);
    sink.i32_const(wb.flag_mask(card_index.is_some()));
    sink.i32_and();
    sink.if_(BlockType::Empty);
    match card_index {
        Some(index) => {
            emit_load_wb_flag_byte(sink, constants, value_types, wb, base_ref);
            sink.i32_const(i32::from(wb.cards_set));
            sink.i32_and();
            sink.if_(BlockType::Empty);
            emit_inline_card_mark(sink, constants, value_types, wb, base_ref, index);
            sink.else_();
            emit_wb_helper_call(
                sink,
                constants,
                value_types,
                base,
                wb.array_fn_ptr,
                base_ref,
                gcmap_ptr,
                ca_reload_fn_ptr,
                jf_top_addr,
            );
            emit_load_wb_flag_byte(sink, constants, value_types, wb, base_ref);
            sink.i32_const(i32::from(wb.cards_set));
            sink.i32_and();
            sink.if_(BlockType::Empty);
            emit_inline_card_mark(sink, constants, value_types, wb, base_ref, index);
            sink.end();
            sink.end();
        }
        None => {
            emit_wb_helper_call(
                sink,
                constants,
                value_types,
                base,
                wb.fn_ptr,
                base_ref,
                gcmap_ptr,
                ca_reload_fn_ptr,
                jf_top_addr,
            );
        }
    }
    sink.end();
    Ok(())
}

/// assembler.py `_reload_frame_if_necessary` tail:
/// `_write_barrier_fastpath(mc, wbdescr, [ebp], array=False, is_frame=True)`.
/// Local 0 was just reloaded; a frame the collection promoted has
/// TRACK_YOUNG_PTRS set and joins the remembered set before the home stores
/// that follow. Frames never use card marking. The helper cannot collect
/// (`_build_wb_slowpath(for_frame=True)`), so no gcmap is pushed and the
/// frame is not reloaded again. Off-GC frames reserve a zeroed header too
/// (`alloc_off_gc_jitframe`), so their flag-byte read is valid.
fn emit_frame_write_barrier(sink: &mut PeepSink<'_, '_>) {
    let Some(wb) = sink.frame_wb else {
        return;
    };
    // The flag test does not depend on the call site. A hot trace calls one
    // copy of this body; local 0 is the items base on both sides.
    if let Some(helper) = wb.helper {
        sink.local_get(0);
        sink.call(helper);
        return;
    }
    sink.local_get(0);
    sink.i32_const(majit_backend::jitframe::FIRST_ITEM_OFFSET as i32);
    sink.i32_sub();
    sink.i32_const(wb.flag_byteofs);
    sink.i32_add();
    sink.i32_load8_u(memarg(0, 0));
    sink.i32_const(i32::from(wb.if_flag));
    sink.i32_and();
    sink.if_(BlockType::Empty);
    sink.local_get(0);
    sink.i32_const(majit_backend::jitframe::FIRST_ITEM_OFFSET as i32);
    sink.i32_sub();
    sink.i64_extend_i32_u();
    sink.i32_const(wb.fn_ptr as i32);
    sink.call_indirect(0, wb.type_idx);
    sink.drop();
    sink.end();
}

/// Date a use at `at` unless it is an owner-defined value leaking into an
/// appended region's body. Those reads are not successors of the owner's
/// collect points — the guard-fail branch is the region's only predecessor
/// — so they must not keep a Ref home live across the owner
/// (`regalloc.py` `Lifetime.last_usage` on the reachable path).
fn note_home_use(
    last_use: &mut [i32],
    def_pos: &[i32],
    regions: &[InlinedRegionSpan],
    raw: usize,
    at: i32,
) {
    if let Some(region) = regions
        .iter()
        .rev()
        .find(|region| at as usize >= region.ops_start)
    {
        let region_def = region.ops_start as i32 - 1;
        if def_pos[raw] < region_def && !region.inputarg_ids.contains(&(raw as u32)) {
            return;
        }
    }
    last_use[raw] = at;
}

/// Per-value def / last-use op positions over the trace, used to filter the
/// post-collection Ref reloads ([`emit_reload_refs_from_homes`]) down to
/// values that are both already defined and still read — the wasm-shaped
/// analog of the native regalloc reloading a spilled box on its next use
/// (llsupport/regalloc.py `longevity`) instead of eagerly rebinding every
/// home.
///
/// Positions: inputs are defined at `-1`; an op result at its op index; a
/// LABEL's args additionally at the label's index (a loop-carried value
/// re-enters the body there — a def index past a reload site must not hide
/// the stale local from the reload on the next iteration). Uses are op args
/// plus guard fail args; the loop-closing JUMP's args are op args, so
/// loop-carried values stay live through the backedge.
struct HomeLiveness {
    def_pos: Vec<i32>,
    last_use: Vec<i32>,
}

impl HomeLiveness {
    fn collect_with_regions(
        inputargs: &[InputArgRc],
        ops: &[Op],
        regions: &[InlinedRegionSpan],
    ) -> Self {
        let mut n = inputargs
            .iter()
            .map(|ia| ia.index as usize + 1)
            .max()
            .unwrap_or(0);
        for op in ops {
            if let Some(id) = result_value_raw(op) {
                n = n.max(id as usize + 1);
            }
        }
        let mut def_pos = vec![i32::MAX; n];
        let mut last_use = vec![-1i32; n];
        for ia in inputargs {
            def_pos[ia.index as usize] = -1;
        }
        for (i, op) in ops.iter().enumerate() {
            if let Some(id) = result_value_raw(op)
                && (id as usize) < n
            {
                let d = &mut def_pos[id as usize];
                *d = (*d).min(i as i32);
            }
            for a in op.getarglist().iter() {
                let a = a.to_opref();
                let Some(id) = value_box_raw(a) else {
                    continue;
                };
                if (id as usize) >= n {
                    continue;
                }
                note_home_use(&mut last_use, &def_pos, regions, id as usize, i as i32);
                // A LABEL arg is a phi def (`consider_label`). A
                // producerless RefOp is a residual virtualizable slot,
                // not a phi: dating it here would hide the unwritten
                // local from the post-collection reload.
                if op.opcode == OpCode::Label && !matches!(a, OpRef::RefOp(_)) {
                    let d = &mut def_pos[id as usize];
                    *d = (*d).min(i as i32);
                }
            }
            if let Some(fa) = op.getfailargs() {
                for a in fa.iter() {
                    let Some(id) = value_box_raw(a.to_opref()) else {
                        continue;
                    };
                    if (id as usize) < n {
                        note_home_use(&mut last_use, &def_pos, regions, id as usize, i as i32);
                    }
                }
            }
        }
        // An appended region's live-ins are written by the guard-fail branch
        // that is the region's sole predecessor, and that branch jumps straight
        // into the region. Their entry in the merged input-arg list would
        // otherwise date them to trace entry, making them live across every
        // collecting call in the owner's body: each would take a Ref home and
        // be reloaded there on every iteration, for a value nothing in the
        // owner reads. Date them to the region instead, so they stay live
        // across the region's own collect points and nowhere else.
        for region in regions {
            let defined_at = region.ops_start as i32 - 1;
            for &id in &region.inputarg_ids {
                if let Some(d) = def_pos.get_mut(id as usize) {
                    *d = defined_at;
                }
            }
        }
        // `store_force_descr` keeps the GUARD_NOT_FORCED_2 gcmap as
        // `_finish_gcmap` and `genop_finish` publishes it. Those homes have
        // to stay traced at every later safepoint, or a collection frees the
        // object and the retained map follows a dead nursery word.
        let end = ops.len() as i32;
        for op in ops.iter().filter(|op| op.opcode == OpCode::GuardNotForced2) {
            let Some(fa) = op.getfailargs() else {
                continue;
            };
            for a in fa.iter() {
                let a = a.to_opref();
                if a.ty() != Some(Type::Ref) {
                    continue;
                }
                let Some(id) = value_box_raw(a) else {
                    continue;
                };
                let raw = id as usize;
                if raw < last_use.len() {
                    last_use[raw] = last_use[raw].max(end);
                }
            }
        }
        Self { def_pos, last_use }
    }

    /// Value `raw` is defined before op `at` and read after it — i.e. its
    /// local holds a value a collection at op `at` could invalidate.
    fn live_across(&self, raw: u32, at: usize) -> bool {
        let raw = raw as usize;
        raw < self.def_pos.len() && self.def_pos[raw] < at as i32 && self.last_use[raw] > at as i32
    }

    fn live_across_any(&self, raw: u32, positions: &[usize]) -> bool {
        positions.iter().any(|&at| self.live_across(raw, at))
    }

    /// Index of the last op that reads `raw` (arg or fail arg), or `-1` when
    /// nothing reads it. `regalloc.py` spells this `Lifetime.last_usage`.
    fn last_use(&self, raw: u32) -> i32 {
        self.last_use.get(raw as usize).copied().unwrap_or(-1)
    }

    fn defined_at(&self, raw: u32) -> i32 {
        self.def_pos.get(raw as usize).copied().unwrap_or(i32::MAX)
    }
}

/// Whether a call has to be followed by the frame and home reloads.
///
/// callbuilder.py splits this in two: `emit_no_collect` prepares the
/// arguments and calls, while `emit` additionally pushes a gcmap and reloads a
/// possibly-forwarded frame afterwards. Without a gcmap the collector cannot
/// reach the home slots, so a callee that cannot collect leaves the JitFrame
/// and every home exactly where they were and the reload only re-reads what
/// the locals already hold. `x86/assembler.py:2205-2209` takes the same
/// decision from the same bit, as does the cranelift residual call emission.
///
/// Two families keep their reloads whatever the effect info says, because
/// upstream pushes their gcmap without ever consulting it:
///
/// - `CALL_RELEASE_GIL` — `x86/assembler.py:2200-2203` dispatches to
///   `emit_call_release_gil` before the `check_can_collect()` test, and
///   `push_gcmap_for_call_release_gil` is unconditional; `emit`'s own
///   docstring reads "not for CALL_RELEASE_GIL". The bit describes the callee,
///   while another thread may collect for the span the GIL is released.
/// - `COND_CALL_VALUE` — `x86/regalloc.py consider_cond_call` builds the gcmap with
///   `get_gcmap()` and reads no effect info at all.
///
/// A call carrying no call descr also keeps its reloads: this narrows a
/// conservative answer where the effect info is there to narrow it, and never
/// widens one.
fn call_can_collect(op: &Op) -> bool {
    if matches!(
        op.opcode,
        OpCode::CallReleaseGilI
            | OpCode::CallReleaseGilN
            | OpCode::CallReleaseGilF
            | OpCode::CondCallValueI
            | OpCode::CondCallValueR
    ) {
        return true;
    }
    op.with_call_descr(|descr| descr.get_extra_info().check_can_collect())
        .unwrap_or(true)
}

/// Static collecting-call positions whose gcmap-visible homes may be forwarded.
/// Every `is_malloc` op (post-rewrite `CallMallocNursery*` included) routes
/// through a collecting allocator. A residual call earns a
/// position only where [`call_can_collect`] admits it, which is the predicate
/// the reload side already applies: a home exists so a collection can forward
/// the value, so a call that cannot collect would buy a home that nothing ever
/// reloads, and its store-on-def and back-edge refresh are stores cranelift
/// does not remove. A value live across some other collecting position still
/// takes a home from that position.
fn collecting_call_positions(ops: &[Op], include_ca_collects: bool) -> Vec<usize> {
    ops.iter()
        .enumerate()
        .filter_map(|(i, op)| {
            ((op.opcode.is_call() && call_can_collect(op)) || op.opcode.is_malloc())
                .then_some(i)
                .or_else(|| (include_ca_collects && op.opcode.is_call_assembler()).then_some(i))
        })
        .collect()
}

/// Null-initialise the home slots of `range` that `wanted` selects.
///
/// The slots are adjacent 8-byte words, so a run of them is a memset: one
/// `memory.fill` writes what would otherwise be a `local.get`/`i64.const`/
/// `i64.store` triple per slot. A trace with hundreds of homes spells that
/// triple hundreds of times in its entry prologue, and the module is charged
/// for it twice — once in what the host hands cranelift, and again on every
/// entry that executes it.
///
/// A run pays for the fill's own operands only once it is long enough:
/// the triple encodes in 7-9 bytes against the fill's 12-15, so a run of one
/// or two slots stays a store. Both forms write exactly the same bytes.
/// `_check_frame_depth` (assembler.py): if `jf_frame.length` is below this
/// module's item count, `wasm_realloc_frame` grows it and local 0 becomes
/// the new items base.
///
/// `gcmap_ptr` is the parked map of Ref spill slots live at this entry
/// (`IncreaseStackSlowPath.generate_body` calls `push_gcmap(..., store=True)`
/// before `CALL(realloc_frame)`). Zero means this entry has no Ref input.
fn emit_check_frame_depth(
    sink: &mut PeepSink<'_, '_>,
    depth_items: usize,
    realloc_fn_ptr: i64,
    residual_type_base: u32,
    gcmap_ptr: i64,
    result_local: u32,
    ca_reload_fn_ptr: i64,
    jf_top_addr: Option<u32>,
    propagate_exception_descr: usize,
) {
    let len_size = majit_backend::jitframe::SIZEOFSIGNED as i32;
    sink.local_get(0);
    sink.i32_const(len_size);
    sink.i32_sub();
    sink.i32_load(mem32(0));
    sink.i32_const(depth_items as i32);
    sink.i32_lt_u();
    sink.if_(BlockType::Empty);
    // IncreaseStackSlowPath.generate_body: push_gcmap(store=True) before
    // CALL(realloc_frame). Homes are still null; the map names Ref inputs
    // sitting in their spill slots.
    if gcmap_ptr != 0 {
        emit_store_header_word(
            sink,
            majit_backend::jitframe::JF_GCMAP_OFS as u64,
            gcmap_ptr as usize,
        );
    }
    sink.local_get(0);
    sink.i64_extend_i32_u();
    sink.i64_const(depth_items as i64);
    sink.i32_const(realloc_fn_ptr as i32);
    sink.call_indirect(0, residual_type_base + 2);
    sink.i32_wrap_i64();
    // A 0 return is MemoryError (`emit_memory_error_on_truthy`). Local 0
    // stays the old frame; only a live items base is installed.
    sink.local_set(result_local);
    sink.local_get(result_local);
    sink.i32_eqz();
    emit_memory_error_on_truthy(
        sink,
        Some(residual_type_base),
        ca_reload_fn_ptr,
        jf_top_addr,
        propagate_exception_descr,
    );
    sink.local_get(result_local);
    sink.local_set(0);
    sink.sync_gcmap_frame();
    sink.end();
}

fn emit_reload_refs_from_homes(
    sink: &mut PeepSink<'_, '_>,
    value_types: &ValueLocals,
    homes: &[(u32, u32)],
    skip_raw: Option<u32>,
    frame: FrameGeometry,
    site_gcmap: &[i64],
    at_op: usize,
) {
    // `homes` is the set `site_live_homes` parked into `jf_gcmap` for this
    // op. A home that no path has stored yet is absent, so the reload cannot
    // clobber the local with nursery bytes (`arena_reset`).
    for &(raw, h) in homes {
        if Some(raw) == skip_raw {
            continue;
        }
        sink.local_get(0);
        sink.i64_load(mem64(frame.home_ofs(h as u64)));
        sink.local_set(value_types.local(raw));
    }
    // assembler.py `pop_gcmap` after `_reload_frame_if_necessary`.
    emit_pop_site(sink, site_gcmap, at_op);
}

/// RPython `_reload_frame_if_necessary` (x86 `assembler.py:1369`) for wasm
/// trace bodies: a collecting direct call may have forwarded the running
/// JitFrame, while wasm local 0 still holds its old ITEMS base.
fn emit_reload_frame_if_necessary(
    sink: &mut PeepSink<'_, '_>,
    residual_type_base: Option<u32>,
    ca_reload_fn_ptr: i64,
    jf_top_addr: Option<u32>,
) {
    if let Some(top_addr) = jf_top_addr {
        // assembler.py:1369-1377: reload the possibly-forwarded top JitFrame
        // directly from the shadow-stack cell. Unlike the helper-table call,
        // this does not need the residual direct-call type to be declared.
        emit_ca_reload_top(sink, top_addr);
        sink.local_set(0);
        emit_frame_write_barrier(sink);
    } else if let Some(base) = residual_type_base.filter(|_| ca_reload_fn_ptr != 0) {
        sink.i32_const(ca_reload_fn_ptr as i32);
        sink.call_indirect(0, base);
        sink.i32_wrap_i64();
        sink.local_set(0);
        sink.sync_gcmap_frame();
        emit_frame_write_barrier(sink);
    } else {
        // No shadow top and no reload helper: no active GC, or an embedder
        // that never pushed a JitFrame. Reloading from a shadow stack that
        // never held this frame would install an unrelated one.
    }
}

/// CA-arm-only variant of [`emit_reload_frame_if_necessary`]. The direct CA
/// configuration owns an inline shadow-stack top cell; all other call sites
/// retain their pre-existing helper reload.
fn emit_reload_ca_frame_if_necessary(
    sink: &mut PeepSink<'_, '_>,
    residual_type_base: Option<u32>,
    ca_reload_fn_ptr: i64,
    ca_inline: Option<CaInlineParams>,
) {
    if let Some(inline) = ca_inline {
        debug_assert!(residual_type_base.is_some());
        emit_ca_reload_top(sink, inline.jf_top_addr);
        sink.local_set(0);
        emit_frame_write_barrier(sink);
    } else {
        emit_reload_frame_if_necessary(sink, residual_type_base, ca_reload_fn_ptr, None);
    }
}

/// assembler.py `_reload_frame_if_necessary`: `top[-WORD]` is the top
/// jitframe pointer. The wasm CA ABI carries its ITEMS base in local 0.
fn emit_ca_reload_top(sink: &mut PeepSink<'_, '_>, top_addr: u32) {
    sink.i32_const(top_addr as i32);
    sink.i32_load(mem32(0));
    sink.i32_const(4);
    sink.i32_sub();
    sink.i32_load(mem32(0));
    if sink.gcmap_frame_local != 0 {
        let frame = sink.gcmap_frame_local;
        sink.local_tee(frame);
    }
    sink.i32_const(majit_backend::jitframe::FIRST_ITEM_OFFSET as i32);
    sink.i32_add();
}

fn emit_word_store(sink: &mut PeepSink<'_, '_>, offset: u64) {
    if majit_backend::jitframe::SIZEOFSIGNED == 4 {
        sink.i32_wrap_i64();
        sink.i32_store(memarg(offset, 2));
    } else {
        sink.i64_store(memarg(offset, 3));
    }
}

/// assembler.py `_call_footer_shadowstack`: `SUB [rootstacktop], 2*WORD`.
fn emit_ca_pop_shadowstack(sink: &mut PeepSink<'_, '_>, top_addr: u32) {
    let ss_word = std::mem::size_of::<usize>() as i32;
    sink.i32_const(top_addr as i32);
    sink.i32_const(top_addr as i32);
    sink.i32_load(mem32(0));
    sink.i32_const(2 * ss_word);
    sink.i32_sub();
    sink.i32_store(mem32(0));
}

/// CA return footer: `_call_footer_shadowstack`, the x86 `SUB`.
///
/// `genop_finish` publishes `_finish_gcmap` (or NULL) before the footer, and
/// a CA callee's `FINISH` does that publish inside generated wasm. A callee
/// that retains `GUARD_NOT_FORCED_2` keeps that map while its force token is
/// armed, because the frame stays reachable after the pop; an unarmed one
/// clears it. The flag is the callee's and lives on the stable dispatch
/// cell, not the pre-call snapshot: a GNF2 bridge can attach while this
/// invocation is inside the callee. The caller's own ops do not describe
/// the frame being popped, including after `redirect_call_assembler`.
fn emit_ca_pop_footer(
    sink: &mut PeepSink<'_, '_>,
    inline: CaInlineParams,
    scratch: u32,
    dispatch_entry: i32,
) {
    use majit_backend::jitframe::{JF_FORCE_DESCR_OFS, JF_GCMAP_OFS, SIZEOFSIGNED};
    sink.i32_const(dispatch_entry);
    sink.i32_load(mem32(crate::failguard::WASM_CA_DISPATCH_HAS_GNF2_OFS));
    sink.if_(BlockType::Empty);
    let ss_word = std::mem::size_of::<usize>() as i32;
    // `assembler.py` `_reload_frame_if_necessary`: `top[-WORD]` is the
    // jitframe, not the CA items base (which a collection may have moved).
    sink.i32_const(inline.jf_top_addr as i32);
    sink.i32_load(mem32(0));
    sink.i32_const(ss_word);
    sink.i32_sub();
    sink.i32_load(mem32(0));
    sink.local_tee(scratch);
    if SIZEOFSIGNED == 4 {
        sink.i32_load(memarg(JF_FORCE_DESCR_OFS as u64, 2));
        sink.i32_eqz();
    } else {
        sink.i64_load(memarg(JF_FORCE_DESCR_OFS as u64, 3));
        sink.i64_eqz();
    }
    sink.if_(BlockType::Empty);
    sink.local_get(scratch);
    if SIZEOFSIGNED == 4 {
        sink.i32_const(0);
        sink.i32_store(memarg(JF_GCMAP_OFS as u64, 2));
    } else {
        sink.i64_const(0);
        sink.i64_store(memarg(JF_GCMAP_OFS as u64, 3));
    }
    sink.end();
    sink.end();
    emit_ca_pop_shadowstack(sink, inline.jf_top_addr);
}

/// While a CA callee is pushed, its caller's `jf_ptr` is `top[-3 * WORD]`.
fn emit_ca_reload_caller(sink: &mut PeepSink<'_, '_>, top_addr: u32) {
    sink.i32_const(top_addr as i32);
    sink.i32_load(mem32(0));
    sink.i32_const(12);
    sink.i32_sub();
    sink.i32_load(mem32(0));
    if sink.gcmap_frame_local != 0 {
        let frame = sink.gcmap_frame_local;
        sink.local_tee(frame);
    }
    sink.i32_const(majit_backend::jitframe::FIRST_ITEM_OFFSET as i32);
    sink.i32_add();
}

/// `_call_header_shadowstack` (`assembler.py`). `ca_cfp_local` holds the
/// object base returned by `CallMallocNurseryVarsizeFrame`.
///
/// The two stores run in the guest when `jf_top` / `jf_limit` are published.
/// A full shadow stack, or a build that did not publish those cells, calls
/// `wasm_jit_ca_push_frame`: the stack then lives in host TLS.
/// Zero the callee's reserved Ref-home region before the frame is a root.
///
/// `build_home_gcmap` marks only the used prefix and the LABEL tail, but a
/// redirect widens that set up to `home_slots` without reallocating the
/// frame. Slots past the homes are not in the map. A zero-length region
/// emits nothing.
fn emit_clear_reserved_homes(
    sink: &mut PeepSink<'_, '_>,
    object_local: u32,
    target_local: u32,
    scratch: u32,
) {
    use crate::failguard::{WASM_CA_TARGET_HOME_SLOT_BASE_OFS, WASM_CA_TARGET_HOME_SLOTS_OFS};
    sink.local_get(target_local);
    sink.i32_load(mem32(WASM_CA_TARGET_HOME_SLOTS_OFS));
    sink.i32_const(3);
    sink.i32_shl();
    sink.local_tee(scratch);
    sink.i32_eqz();
    sink.if_(BlockType::Empty);
    sink.else_();
    sink.local_get(object_local);
    sink.i32_const(majit_backend::jitframe::FIRST_ITEM_OFFSET as i32);
    sink.i32_add();
    sink.local_get(target_local);
    sink.i32_load(mem32(WASM_CA_TARGET_HOME_SLOT_BASE_OFS));
    sink.i32_add();
    sink.i32_const(0);
    sink.local_get(scratch);
    sink.memory_fill(0);
    sink.end();
}

fn emit_ca_push_frame(
    sink: &mut PeepSink<'_, '_>,
    inline: Option<&CaInlineParams>,
    residual_type_base: Option<u32>,
    ca_push_fn_ptr: i64,
    jit_call_idx: Option<u32>,
    ca_cfp_local: u32,
    alloc_scratch_local: u32,
) -> Result<(), BackendError> {
    if let Some(inline) = inline {
        let ss_word = std::mem::size_of::<usize>() as i32;
        sink.i32_const(inline.jf_top_addr as i32);
        sink.i32_load(mem32(0));
        sink.local_tee(alloc_scratch_local);
        sink.i32_const(2 * ss_word);
        sink.i32_add();
        sink.i32_const(inline.jf_limit_addr as i32);
        sink.i32_load(mem32(0));
        sink.i32_gt_u();
        sink.if_(BlockType::Empty);
        let Some(base) = residual_type_base else {
            return Err(BackendError::Unsupported(
                "wasm codegen: CALL_ASSEMBLER shadow-stack overflow needs a push helper type"
                    .into(),
            ));
        };
        sink.local_get(ca_cfp_local);
        sink.i64_extend_i32_u();
        sink.i32_const(ca_push_fn_ptr as i32);
        sink.call_indirect(0, base + 1);
        sink.drop();
        sink.else_();
        sink.local_get(alloc_scratch_local);
        sink.i64_const(1);
        emit_word_store(sink, 0);
        sink.local_get(alloc_scratch_local);
        sink.local_get(ca_cfp_local);
        sink.i64_extend_i32_u();
        emit_word_store(sink, ss_word as u64);
        sink.i32_const(inline.jf_top_addr as i32);
        sink.local_get(alloc_scratch_local);
        sink.i32_const(2 * ss_word);
        sink.i32_add();
        sink.i32_store(mem32(0));
        sink.end();
        return Ok(());
    }
    if let Some(base) = residual_type_base {
        sink.local_get(ca_cfp_local);
        sink.i64_extend_i32_u();
        sink.i32_const(ca_push_fn_ptr as i32);
        sink.call_indirect(0, base + 1);
        sink.drop();
        return Ok(());
    }
    let jit_call = jit_call_idx.ok_or_else(|| {
        BackendError::Unsupported(
            "wasm codegen: CALL_ASSEMBLER needs jit_call to push the frame".into(),
        )
    })?;
    emit_call_area_addr(sink);
    sink.i64_const(ca_push_fn_ptr);
    sink.i64_store(mem64(STATIC_CALL_FUNC_OFS));
    emit_call_area_addr(sink);
    sink.i64_const(1);
    sink.i64_store(mem64(STATIC_CALL_NARGS_OFS));
    emit_call_area_addr(sink);
    sink.local_get(ca_cfp_local);
    sink.i64_extend_i32_u();
    sink.i64_store(mem64(STATIC_CALL_ARGS_OFS));
    emit_store_call_result_facts(sink, 8);
    emit_jit_call(sink, jit_call);
    Ok(())
}

/// Information about a guard exit collected during pre-scan.
/// The live-position mask a guard's bridge inputargs were filtered by.
///
/// `pyjitpl.rs initialize_state_from_guard_failure` builds the bridge history
/// from `rd_locs`: a position is live when its entry is not `0xFFFF`, and a
/// descr whose `rd_locs` has not been sized to the fail-arg list (a synthetic
/// one that never reached codegen) declares every position live. Any arity a
/// backend compares against `bridge.inputargs.len()` has to come from that same
/// table — `OpRef::is_none()` is this backend's own IR-level hole set and is a
/// different mask, so counting with it refuses bridges whose arity was fine.
pub fn live_fail_arg_mask(meta_descr: Option<&majit_ir::DescrRef>, n: usize) -> Vec<bool> {
    match meta_descr.and_then(|d| d.as_fail_descr()) {
        Some(fd) if fd.rd_locs().len() == n => {
            fd.rd_locs().iter().map(|&pos| pos != 0xFFFF).collect()
        }
        _ => vec![true; n],
    }
}

/// How many of a guard's fail-arg positions reach its bridge as inputargs.
pub fn live_fail_arg_count(meta_descr: Option<&majit_ir::DescrRef>, n: usize) -> usize {
    live_fail_arg_mask(meta_descr, n)
        .iter()
        .filter(|l| **l)
        .count()
}

/// One past the highest live LOGICAL resume position. The frontend keeps
/// these coordinates, while `GuardExit.fail_locs` maps them to compact
/// physical slots, like BaseAssembler.store_info_on_descr. This extent sizes
/// the frontend counter coordinate, never the generated frame's spill area.
pub fn live_fail_arg_extent(meta_descr: Option<&majit_ir::DescrRef>, n: usize) -> usize {
    live_fail_arg_mask(meta_descr, n)
        .iter()
        .rposition(|&live| live)
        .map_or(0, |i| i + 1)
}

/// BaseAssembler.rebuild_faillocs_from_descr skips holes when binding bridge
/// inputs. Our compact spill assigns those live positions slot 0, 1, ... in
/// the same order, so a frame-entry bridge reads that physical sequence.
pub fn frame_entry_reads_live_positions(
    fail_descr: &dyn majit_ir::FailDescr,
    bridge_inputs: usize,
) -> bool {
    let n = fail_descr.fail_arg_types().len();
    let locs = fail_descr.rd_locs();
    if locs.len() != n {
        return bridge_inputs == n;
    }
    let live = locs.iter().filter(|&&pos| pos != 0xFFFF).count();
    bridge_inputs == live
}

fn live_exit_fail_args(op: &Op) -> Vec<OpRef> {
    let args = exit_fail_args(op);
    let live = live_fail_arg_mask(op.getdescr().as_ref(), args.len());
    args.into_iter()
        .zip(live)
        .filter_map(|(arg, live)| live.then_some(arg))
        .collect()
}

pub struct GuardExit {
    /// llsupport/assembler.py BaseAssembler.store_info_on_descr: logical
    /// resume positions name physical spill locations; holes own no slot.
    pub fail_locs: Vec<Option<usize>>,
    pub fail_index: u32,
    pub fail_arg_refs: Vec<OpRef>,
    pub fail_arg_types: Vec<Type>,
    pub is_finish: bool,
    /// Signed-item indices of the Ref homes kept alive by
    /// GUARD_NOT_FORCED(_2).  This is the wasm equivalent of the guard token's
    /// compile-time `gcmap` which `store_force_descr` saves as
    /// `assembler._finish_gcmap`; it must not be reconstructed from runtime
    /// fail-argument words after FINISH.
    pub force_ref_home_indices: Vec<u32>,
    /// `GUARD_NOT_FORCED_2`: this guard's homes are `assembler._finish_gcmap`.
    pub publishes_finish_gcmap: bool,
    /// Cell address the exit stores in `jf_descr`. Filled before emit.
    pub descr_cell: usize,
    /// Compile-time map the exit stores in `jf_gcmap`. `0` stores a null map.
    pub exit_gcmap_ptr: usize,
    /// The GUARD_VALUE operand this exit parks in the trace's counter slot,
    /// for `make_a_counter_per_value`. `None` when the guard is not a
    /// GUARD_VALUE, or when its operand is already one of the fail arguments
    /// and so already has a slot. See `counter_value_spill`.
    pub counter_value_spill: Option<OpRef>,
    /// Guest address of this guard's bridge-target cell. `0` when the
    /// module has no bridge dispatch. Stamped onto `adr_jump_offset`.
    pub bridge_cell: u32,
    /// `op.descr` snapshot — passed through to `WasmFailDescr.meta_descr`
    /// so `get_latest_descr_arc` can return the canonical metainterp Arc
    /// (parity with dynasm/cranelift's `meta_descr` forwarding).
    pub meta_descr: Option<majit_ir::DescrRef>,
}

/// Pre-fetched GC-type-guard metadata for the wasm codegen.
///
/// RPython's `genop_guard_guard_*` methods call into
/// `self.cpu.gc_ll_descr` at codegen time to obtain the TYPE_INFO
/// table base, the `infobits` offset / byte mask, the subclassrange
/// field offset, and the `(subclassrange_min, subclassrange_max)`
/// bounds for the constant expected-class pointer. The wasm backend
/// has no direct handle on a `GcAllocator` at this layer, so the
/// caller (`WasmBackend::compile_loop`) pre-fetches each of those
/// values and bundles them here.
///
/// Parity references:
///  * `llsupport/gc.py:162` / `gc.py:318` — `supports_guard_gc_type`
///  * `llsupport/gc.py` — `get_translated_info_for_typeinfo`
///  * `llsupport/gc.py` — `get_translated_info_for_guard_is_object`
///  * `x86/assembler.py` — `cpu.subclassrange_min_offset`
///  * `x86/assembler.py:1971-1974` — constant-time
///    `(vtable_ptr.subclassrange_min, vtable_ptr.subclassrange_max)`
///
/// The default sets `supports_guard_gc_type = false`, matching
/// `AbstractCPU.supports_guard_gc_type` in `backend/model.py`; the
/// codegen arms assert this flag before reading any other field.
#[derive(Clone, Default)]
pub struct GuardGcTypeInfo {
    pub supports_guard_gc_type: bool,
    /// `get_translated_info_for_typeinfo()` = (base, shift, sizeof_ti).
    pub base_type_info: usize,
    pub shift_by: u8,
    pub sizeof_ti: usize,
    /// `get_translated_info_for_guard_is_object()`
    ///     = (infobits_offset, T_IS_RPYTHON_INSTANCE_BYTE).
    pub infobits_offset: usize,
    pub is_object_flag: u8,
    /// `cpu.subclassrange_min_offset` (x86/assembler.py).
    pub subclassrange_min_offset: usize,
    /// `(vtable_ptr.subclassrange_min, vtable_ptr.subclassrange_max)`
    /// looked up by constant classptr. Empty when
    /// `supports_guard_gc_type == false`.
    pub subclass_ranges: HashMap<i64, (i64, i64)>,
}

/// Descr-derived wasm type the direct arm would use for this op.
///
/// CallN's void-word vs true-void result follows the oracle's real result
/// when one is known; otherwise it follows `result_size`. Shared by the
/// direct-vs-trampoline predicate and the emitter's type-index choice.
fn expected_direct_wasm_sig(
    op: &Op,
    constants: &indexmap::IndexMap<u32, i64>,
) -> Option<TypedResidualSig> {
    expected_direct_wasm_sig_at(op, constants, residual_func_ofs(op.opcode))
}

fn expected_direct_wasm_sig_at(
    op: &Op,
    constants: &indexmap::IndexMap<u32, i64>,
    func_arg: usize,
) -> Option<TypedResidualSig> {
    let descr = op.getdescr()?;
    let cd = descr.as_call_descr()?;
    let arg_types = cd.arg_types();
    let arg_classes = cd.arg_classes();
    let mut params = Vec::with_capacity(arg_types.len());
    for (idx, ty) in arg_types.iter().enumerate() {
        // `descr.py map_type_to_argclass`: `'S'` is an int-bank value passed
        // as C `float`. `CallBuilder64.prepare_arguments` moves it with MOVD32.
        if arg_classes.as_bytes().get(idx) == Some(&b'S') {
            params.push(ValType::F32);
            continue;
        }
        params.push(match ty {
            Type::Float => ValType::F64,
            Type::Int | Type::Ref => ValType::I64,
            Type::Void => return None,
        });
    }
    let nargs = op.num_args().saturating_sub(func_arg + 1);
    if params.len() != nargs {
        return None;
    }
    let is_void_op = matches!(
        op.opcode,
        OpCode::CallN
            | OpCode::CallPureN
            | OpCode::CallLoopinvariantN
            | OpCode::CallMayForceN
            | OpCode::CallReleaseGilN
            | OpCode::CondCallN
    );
    let mut result = if is_void_op {
        if cd.result_type() != Type::Void {
            return None;
        }
        match cd.result_size() {
            0 => None,
            8 => Some(ValType::I64),
            _ => return None,
        }
    } else {
        if op.result_type() != cd.result_type() {
            return None;
        }
        match cd.result_type() {
            Type::Float => Some(ValType::F64),
            // `descr.py` result `'S'`: the callee returns C `float` and
            // `singlefloat2int` keeps the bits. The IR result stays `Int`.
            Type::Int if cd.result_class() == 'S' => Some(ValType::F32),
            Type::Int | Type::Ref => Some(ValType::I64),
            Type::Void => return None,
        }
    };
    // A void op's `result_size` is the historical dummy-word bit. When the
    // table names the callee, its declared result wins (`get_result_size`
    // cannot see an i32/void split the table already published).
    if is_void_op
        && let Some(addr) = const_funcptr_addr(op, constants, func_arg)
        && let Some(real) = crate::residual_target_sig(addr)
    {
        match real.result {
            None => result = None,
            Some(crate::FuncSigVal::I64) | Some(crate::FuncSigVal::I32) => {
                result = Some(ValType::I64)
            }
            Some(crate::FuncSigVal::F32) | Some(crate::FuncSigVal::F64) => {}
        }
    }
    Some((params, result))
}

fn const_funcptr_addr(
    op: &Op,
    constants: &indexmap::IndexMap<u32, i64>,
    func_arg: usize,
) -> Option<i64> {
    let func_ptr = op.getarglist().get(func_arg).map(|arg| arg.to_opref())?;
    func_ptr
        .is_constant()
        .then(|| resolve_const_bits(constants, func_ptr))
}

fn func_sig_val_to_valtype(val: crate::FuncSigVal) -> ValType {
    match val {
        crate::FuncSigVal::I32 => ValType::I32,
        crate::FuncSigVal::I64 => ValType::I64,
        crate::FuncSigVal::F32 => ValType::F32,
        crate::FuncSigVal::F64 => ValType::F64,
    }
}

fn wasm_sig_to_typed(sig: &crate::WasmSig) -> TypedResidualSig {
    (
        sig.params
            .iter()
            .copied()
            .map(func_sig_val_to_valtype)
            .collect(),
        sig.result.map(func_sig_val_to_valtype),
    )
}

/// True when `real` differs from `expected` only by a narrower wasm spelling
/// of the same descr slot: i32 for an i64, or f32 for an f64 or for i64 bits.
fn table_type_variance(expected: &TypedResidualSig, real: &TypedResidualSig) -> bool {
    if expected.0.len() != real.0.len() || expected.1.is_some() != real.1.is_some() {
        return false;
    }
    for (want, got) in expected.0.iter().zip(&real.0) {
        if !narrow_valtype(*want, *got) {
            return false;
        }
    }
    match (expected.1, real.1) {
        (None, None) => true,
        (Some(want), Some(got)) => narrow_valtype(want, got),
        _ => false,
    }
}

fn narrow_valtype(want: ValType, got: ValType) -> bool {
    matches!(
        (want, got),
        (a, b) if a == b
            || (want == ValType::I64 && matches!(got, ValType::I32 | ValType::F32))
            || (want == ValType::F64 && got == ValType::F32)
    )
}

/// `_genop_call` emits `CallDescr.get_arg_types` / `get_result_type` /
/// `get_result_size`. The descr is the call type. A constant callee may
/// narrow that to the table's i32 or f32 spelling. A slot the table does
/// not hold (a host import) keeps `jit_call`.
fn residual_callee_direct_emit_sig_at(
    op: &Op,
    constants: &indexmap::IndexMap<u32, i64>,
    func_arg: usize,
    expected: &TypedResidualSig,
) -> Option<TypedResidualSig> {
    let Some(func_ptr) = op.getarglist().get(func_arg).map(|arg| arg.to_opref()) else {
        return None;
    };
    if !func_ptr.is_constant() {
        // The descr is the type, the same call the native backends emit.
        // A constant callee below may narrow that to the table's i32 or f32.
        return Some(expected.clone());
    }
    let addr = resolve_const_bits(constants, func_ptr);
    match crate::residual_target_sig(addr) {
        Some(real) => {
            let real_typed = wasm_sig_to_typed(&real);
            if real_typed == *expected {
                Some(expected.clone())
            } else if table_type_variance(expected, &real_typed) {
                Some(real_typed)
            } else {
                None
            }
        }
        None if cfg!(target_arch = "wasm32") => None,
        None => Some(expected.clone()),
    }
}

fn residual_direct_emit_sig(
    op: &Op,
    constants: &indexmap::IndexMap<u32, i64>,
) -> Option<TypedResidualSig> {
    let expected = expected_direct_wasm_sig(op, constants)?;
    residual_callee_direct_emit_sig_at(op, constants, residual_func_ofs(op.opcode), &expected)
}

/// Direct uniform-word shapes for COND_CALL. Conditional calls place the
/// predicate at arg 0 and the callee at arg 1, unlike ordinary residual CALLs.
/// Returning `(arity, returns_word)` distinguishes COND_CALL_VALUE from the
/// void form whose descr may still carry the historical dummy-word ABI.
fn conditional_call_word_shape(
    op: &Op,
    constants: &indexmap::IndexMap<u32, i64>,
) -> Option<(usize, bool)> {
    if !matches!(
        op.opcode,
        OpCode::CondCallN | OpCode::CondCallValueI | OpCode::CondCallValueR
    ) {
        return None;
    }
    let descr = op.getdescr()?;
    let cd = descr.as_call_descr()?;
    let returns_word = matches!(op.opcode, OpCode::CondCallValueI | OpCode::CondCallValueR);
    if returns_word {
        if !matches!(cd.result_type(), Type::Int | Type::Ref)
            || cd.result_type() != op.result_type()
        {
            return None;
        }
    } else if cd.result_type() != Type::Void || !matches!(cd.result_size(), 0 | 8) {
        return None;
    }
    if cd
        .arg_types()
        .iter()
        .any(|ty| !matches!(ty, Type::Int | Type::Ref))
    {
        return None;
    }
    let nargs = op.getarglist().len().saturating_sub(2);
    if cd.arg_types().len() != nargs {
        return None;
    }
    let expected = expected_direct_wasm_sig_at(op, constants, 1)?;
    let emit = residual_callee_direct_emit_sig_at(op, constants, 1, &expected)?;
    if emit.0.iter().any(|t| *t != ValType::I64) {
        return None;
    }
    let emit_word = emit.1 == Some(ValType::I64);
    if returns_word != emit_word && returns_word {
        return None;
    }
    Some((emit.0.len(), emit_word))
}

fn conditional_call_i64_arity(op: &Op, constants: &indexmap::IndexMap<u32, i64>) -> Option<usize> {
    conditional_call_word_shape(op, constants)
        .filter(|(_, returns_word)| *returns_word)
        .map(|(arity, _)| arity)
}

fn conditional_call_true_void_arity(
    op: &Op,
    constants: &indexmap::IndexMap<u32, i64>,
) -> Option<usize> {
    conditional_call_word_shape(op, constants)
        .filter(|(_, returns_word)| !*returns_word)
        .map(|(arity, _)| arity)
}

/// Mixed `i32`/`i64`/`f64` signature of a `COND_CALL`, when the uniform word
/// families do not already cover it. The callee sits at arg 1.
fn conditional_call_typed_sig(
    op: &Op,
    constants: &indexmap::IndexMap<u32, i64>,
) -> Option<TypedResidualSig> {
    if !matches!(
        op.opcode,
        OpCode::CondCallN | OpCode::CondCallValueI | OpCode::CondCallValueR
    ) {
        return None;
    }
    if conditional_call_i64_arity(op, constants).is_some()
        || conditional_call_true_void_arity(op, constants).is_some()
    {
        return None;
    }
    let expected = expected_direct_wasm_sig_at(op, constants, 1)?;
    residual_callee_direct_emit_sig_at(op, constants, 1, &expected)
}

/// `_emit_call` / `simple_call`: one lowering for CALL and COND_CALL.
/// Arguments follow the descr `ValType`, the func slot is the table index,
/// then `call_indirect`. The caller handles the result.
fn emit_typed_residual_call(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    args: &[Operand],
    params: &[ValType],
    func: OpRef,
    site_gcmap: &[i64],
    op_idx: usize,
    type_idx: u32,
) {
    for (arg, ty) in args.iter().zip(params) {
        match *ty {
            ValType::F64 => {
                emit_resolve_f64(sink, constants, value_types, arg.to_opref());
            }
            ValType::I32 => {
                emit_resolve(sink, constants, value_types, arg.to_opref());
                sink.i32_wrap_i64();
            }
            ValType::F32 => {
                let arg = arg.to_opref();
                if arg.ty() == Some(Type::Float) {
                    emit_resolve_f64(sink, constants, value_types, arg);
                    sink.f32_demote_f64();
                } else {
                    emit_resolve(sink, constants, value_types, arg);
                    sink.i32_wrap_i64();
                    sink.f32_reinterpret_i32();
                }
            }
            _ => {
                emit_resolve(sink, constants, value_types, arg.to_opref());
            }
        }
    }
    emit_resolve(sink, constants, value_types, func);
    sink.i32_wrap_i64();
    emit_push_site(sink, site_gcmap, op_idx);
    sink.call_indirect(0, type_idx);
}

/// `callbuilder.py` `load_result`: a result narrower than a word is sign- or
/// zero-extended from `result_size`. An i32 wasm result whose descr is a full
/// word zero-extends (wasm32 pointer / `usize`). An f32 float promotes; an f32
/// whose descr is an int is a singlefloat bit pattern.
fn widen_direct_call_result(sink: &mut PeepSink<'_, '_>, op: &Op, result_ty: Option<ValType>) {
    match result_ty {
        Some(ValType::I32) => {
            let signed = op.getdescr().is_some_and(|descr| {
                descr.as_call_descr().is_some_and(|cd| {
                    cd.result_type() == Type::Int && cd.is_result_signed() && cd.result_size() < 8
                })
            });
            if signed {
                sink.i64_extend_i32_s();
            } else {
                sink.i64_extend_i32_u();
            }
        }
        Some(ValType::F32) => {
            if op.result_type() == Type::Float {
                sink.f64_promote_f32();
            } else {
                sink.i32_reinterpret_f32();
                sink.i64_extend_i32_u();
            }
        }
        _ => {}
    }
}

/// Host `jit_call` stores an i32 result zero-extended. `load_result` then
/// sign-extends a signed int narrower than a word. The wasm C ABI already
/// extended a 1- or 2-byte return to i32, so one `i64.extend_i32_s` covers it.
fn sign_extend_trampolined_int(sink: &mut PeepSink<'_, '_>, op: &Op) {
    let narrow_signed = op.getdescr().is_some_and(|descr| {
        descr.as_call_descr().is_some_and(|cd| {
            cd.result_type() == Type::Int && cd.is_result_signed() && cd.result_size() < 8
        })
    });
    if narrow_signed {
        sink.i32_wrap_i64();
        sink.i64_extend_i32_s();
    }
}

/// If `op` is a residual CALL whose ABI is uniformly i64 (all Int/Ref args and
/// an Int/Ref result), return its argument count — eligible for a direct
/// `call_indirect` of type `(i64×n) -> i64`. `None` keeps the `jit_call`
/// trampoline: void / float / release-GIL / assembler calls, a missing
/// call descr, or an arg-count/descr-shape mismatch (defensive).
///
/// This includes `CallMayForce{I,R}` when their ABI is uniformly i64: the force
/// protocol rides the frame's own data region
/// (`emit_force_bracket_before_call` before the call, `GuardNotForced` after),
/// which neither lowering touches, so a direct call is sound. `CallReleaseGilI`
/// is the same ABI with the callee at arg 1 (`direct_call_release_gil`);
/// wasm32 has no GIL to drop, so the call itself is an ordinary residual.
/// Float / assembler calls and non-reflectable descrs remain on
/// the trampoline. `COND_CALL` declines instead of using it.
fn residual_func_ofs(opcode: OpCode) -> usize {
    usize::from(matches!(
        opcode,
        OpCode::CallReleaseGilI | OpCode::CallReleaseGilF | OpCode::CallReleaseGilN
    ))
}

fn residual_call_i64_arity(op: &Op, constants: &indexmap::IndexMap<u32, i64>) -> Option<usize> {
    use OpCode::*;
    if !matches!(
        op.opcode,
        CallI
            | CallR
            | CallPureI
            | CallPureR
            | CallLoopinvariantI
            | CallLoopinvariantR
            | CallMayForceI
            | CallMayForceR
            | CallReleaseGilI
    ) {
        return None;
    }
    if !matches!(op.result_type(), Type::Int | Type::Ref) {
        return None;
    }
    let descr = op.getdescr()?;
    let cd = descr.as_call_descr()?;
    let arg_types = cd.arg_types();
    if arg_types
        .iter()
        .any(|t| !matches!(t, Type::Int | Type::Ref))
    {
        return None;
    }
    // Ordinary CALL: func at arg 0, call args at `[1..]`.
    // CALL_RELEASE_GIL: savebox at 0, func at 1, call args at `[2..]`.
    let func_ofs = residual_func_ofs(op.opcode);
    let nargs = op.num_args().saturating_sub(func_ofs + 1);
    if arg_types.len() != nargs {
        return None;
    }
    let emit = residual_direct_emit_sig(op, constants)?;
    if emit.0.iter().any(|t| *t != ValType::I64) || emit.1 != Some(ValType::I64) {
        return None;
    }
    Some(nargs)
}

/// Wasm parameter types and result type of a residual call lowered directly.
/// A residual callee's wasm signature taken from its call descr: the
/// parameter sequence, and the result -- `None` for a callee that returns
/// nothing, which wasm spells as an empty result list rather than a type.
type TypedResidualSig = (Vec<ValType>, Option<ValType>);

/// If `op` is a residual float CALL with only float arguments, return its wasm
/// parameter types — eligible for a direct `call_indirect` returning `f64`.
/// Float-result targets are not audited for a uniform word ABI: a `Ref` or
/// `Int` argument may actually be an `i32` pointer, such as
/// `jit_bigint_to_f64_or_inf`. `None` keeps the `jit_call` trampoline:
/// non-float / release-GIL / assembler calls, a missing call descr, a
/// non-float argument or result type, or an arg-count/descr-shape mismatch
/// (defensive).
///
/// This includes `CallMayForceF`: the force protocol rides the frame's own data
/// region (`emit_force_bracket_before_call` before the call, `GuardNotForced`
/// after), which neither lowering touches, so a direct call is sound.
fn residual_call_typed_sig(
    op: &Op,
    constants: &indexmap::IndexMap<u32, i64>,
) -> Option<TypedResidualSig> {
    use OpCode::*;
    if !matches!(
        op.opcode,
        CallF | CallPureF | CallLoopinvariantF | CallMayForceF | CallReleaseGilF
    ) && !matches!(
        op.opcode,
        CallI | CallPureI | CallLoopinvariantI | CallMayForceI | CallReleaseGilI
    ) && !matches!(
        op.opcode,
        CallR | CallPureR | CallLoopinvariantR | CallMayForceR
    ) && !matches!(
        op.opcode,
        CallN | CallPureN | CallLoopinvariantN | CallMayForceN | CallReleaseGilN
    ) {
        return None;
    }
    // The uniform families are preferred wherever they can express the callee,
    // and the emit arm tries them first. Declining here rather than at the call
    // site keeps the four predicates disjoint, so the signature collected for an
    // op is always the signature its emit reaches -- a type collected for an op
    // the i64 family claims would declare an index nothing branches to.
    if residual_call_i64_arity(op, constants).is_some()
        || residual_call_void_word_arity(op, constants).is_some()
        || residual_call_void_true_arity(op, constants).is_some()
    {
        return None;
    }
    residual_direct_emit_sig(op, constants)
}

/// Void-recorded counterpart of [`residual_call_i64_arity`]: an eligible
/// void residual CALL whose descr records the dummy-word C ABI
/// (`result_size == 8`, minted by `make_call_descr_void_word_abi`) — the
/// callee is really `(i64×n) -> i64` with the result ignored, so it lowers
/// through the same i64 type family with a trailing `drop`. This includes
/// `CallMayForceN` with the word ABI: the force protocol rides the frame's own
/// data region (`emit_force_bracket_before_call` before the call,
/// `GuardNotForced` after), which neither lowering touches, so a direct call is
/// sound.
/// Float / release-GIL / assembler calls and non-reflectable descrs
/// remain on the trampoline. `COND_CALL` declines instead of using it.
fn residual_call_void_word_arity(
    op: &Op,
    constants: &indexmap::IndexMap<u32, i64>,
) -> Option<usize> {
    use OpCode::*;
    if !matches!(
        op.opcode,
        CallN | CallPureN | CallLoopinvariantN | CallMayForceN | CallReleaseGilN
    ) {
        return None;
    }
    let descr = op.getdescr()?;
    let cd = descr.as_call_descr()?;
    if cd.result_type() != Type::Void {
        return None;
    }
    let arg_types = cd.arg_types();
    if arg_types
        .iter()
        .any(|t| !matches!(t, Type::Int | Type::Ref))
    {
        return None;
    }
    let func_ofs = residual_func_ofs(op.opcode);
    let nargs = op.num_args().saturating_sub(func_ofs + 1);
    if arg_types.len() != nargs {
        return None;
    }
    let emit = residual_direct_emit_sig(op, constants)?;
    if emit.0.iter().any(|t| *t != ValType::I64) || emit.1 != Some(ValType::I64) {
        return None;
    }
    Some(nargs)
}

/// True-void counterpart of [`residual_call_void_word_arity`]: an eligible
/// void residual CALL whose descr records a `()` result (`result_size == 0`).
/// Int/Ref-only arguments lower through the `(i64×n) -> ()` type family with
/// no result to drop. Float / release-GIL / assembler calls,
/// non-reflectable descrs, and descr/operand arity mismatches remain on the
/// trampoline. `COND_CALL` declines instead of using it.
fn residual_call_void_true_arity(
    op: &Op,
    constants: &indexmap::IndexMap<u32, i64>,
) -> Option<usize> {
    use OpCode::*;
    if !matches!(
        op.opcode,
        CallN | CallPureN | CallLoopinvariantN | CallMayForceN | CallReleaseGilN
    ) {
        return None;
    }
    let descr = op.getdescr()?;
    let cd = descr.as_call_descr()?;
    if cd.result_type() != Type::Void {
        return None;
    }
    let arg_types = cd.arg_types();
    if arg_types
        .iter()
        .any(|t| !matches!(t, Type::Int | Type::Ref))
    {
        return None;
    }
    let func_ofs = residual_func_ofs(op.opcode);
    let nargs = op.num_args().saturating_sub(func_ofs + 1);
    if arg_types.len() != nargs {
        return None;
    }
    let emit = residual_direct_emit_sig(op, constants)?;
    if emit.0.iter().any(|t| *t != ValType::I64) || emit.1.is_some() {
        return None;
    }
    Some(nargs)
}

/// Arity of `op`'s in-module `(i64×n) -> i64` lowering, if it has one: an
/// eligible residual CALL (word-result or word-ABI void), a
/// `CallMallocNursery*` slow path (the `wasm_jit_alloc*` helper targets are
/// plain `extern "C" fn(i64×n) -> i64` table entries), or a
/// `CondCallGcWb*` (`wasm_jit_write_barrier` takes 1 arg). All of these share
/// the i64-result residual-call type family, so one max covers them. True-void
/// residuals use a separate result family and arity census.
fn direct_helper_i64_arity(
    op: &Op,
    ref_values: &RefValues,
    constants: &indexmap::IndexMap<u32, i64>,
) -> Option<usize> {
    if let Some(n) = conditional_call_i64_arity(op, constants) {
        return Some(n);
    }
    if let Some(n) = residual_call_i64_arity(op, constants) {
        return Some(n);
    }
    if let Some(n) = residual_call_void_word_arity(op, constants) {
        return Some(n);
    }
    match op.opcode {
        // wasm_jit_alloc(type_id, size)
        OpCode::CallMallocNursery => Some(2),
        OpCode::CallMallocNurseryHeaderless | OpCode::ThreadlocalrefGet => Some(1),
        OpCode::CallMallocNurseryVarsizeFrame => Some(2),
        // wasm_jit_alloc_array(type_id, base_size, item_size, length, len_offset)
        OpCode::NewArray
        | OpCode::NewArrayClear
        | OpCode::CallMallocNurseryVarsize
        | OpCode::Newstr
        | OpCode::Newunicode => Some(5),
        // rewrite_ops_for_gc lowers SETFIELD_GC / SETARRAYITEM_GC to these.
        OpCode::CondCallGcWb => Some(1),
        OpCode::CondCallGcWbArray => Some(2),
        // wasm_jit_write_barrier(base)
        _ => write_barrier_base(op, ref_values).map(|_| 1),
    }
}

/// Whether this trace emits a host `jit_call` / `jit_call_compact` trampoline
/// invocation and therefore needs the corresponding function import.
///
/// Keep this in lockstep with the individual emission arms below: the uniform
/// i64, typed float, and true-void residual families, `CallMallocNursery*`,
/// and write barriers are direct when the call descr and the callee's table
/// type agree. A `COND_CALL` whose descr does not establish that signature
/// declines. A mismatch, a host import, and string allocation retain the
/// trampoline for ordinary calls.
fn has_trampoline_calls(
    inputargs: &[InputArgRc],
    ops: &[Op],
    constants: &indexmap::IndexMap<u32, i64>,
    emit_ca: bool,
) -> bool {
    let ref_values = RefValues::collect(inputargs, ops);
    ops.iter().any(|op| match op.opcode {
        // `build_function` handles an enabled CALL_ASSEMBLER before the generic
        // CALL arm, lowering it directly to the callee-loop table slot. It
        // therefore never uses the host call area.
        opcode if opcode.is_call_assembler() && emit_ca => false,
        // These live in resoperation's CALL range for effect classification,
        // but neither emission arm calls anything: CheckMemoryError is an
        // inline null/exit test and RecordKnownResult is optimizer metadata.
        OpCode::CheckMemoryError | OpCode::RecordKnownResult => false,
        // Every residual CALL uses the trampoline unless its exact lowering
        // predicate supplies an i64, typed float, or true-void helper ABI.
        _ if op.opcode.is_call() => {
            direct_helper_i64_arity(op, &ref_values, constants).is_none()
                && residual_call_typed_sig(op, constants).is_none()
                && residual_call_void_true_arity(op, constants).is_none()
                && conditional_call_true_void_arity(op, constants).is_none()
                && conditional_call_typed_sig(op, constants).is_none()
        }
        // `CallMallocNursery*` and `CondCallGcWb*` are covered by
        // `direct_helper_i64_arity`, so their direct-family arms do not touch
        // the frame call area.
        _ => false,
    })
}

fn collect_guards_and_vars(inputargs: &[InputArgRc], ops: &[Op]) -> (Vec<GuardExit>, u32) {
    let mut guards = Vec::new();
    let mut max_var: u32 = 0;

    for ia in inputargs {
        if ia.index + 1 > max_var {
            max_var = ia.index + 1;
        }
    }

    let mut fail_index = 0u32;
    for op in ops {
        if let Some(id) = result_value_raw(op) {
            max_var = max_var.max(id + 1);
        }
        // Every value an op reads occupies a local, whether or not the trace
        // also contains an op that produces it: constant folding and the short
        // preamble leave a folded value bound only by the constants pool, and
        // `unbound_pool_const_seeds` materializes it in the prologue. Counting
        // only op results would under-size `num_vars` for such a value and,
        // through `next_value_pos`, let `remove_ref_constants` reuse its id for
        // a `LoadFromGcTable` — whose store then lands after the read, so the
        // read returns the zero wasm initializes the local to.
        for a in op.getarglist().iter() {
            widen_value_id(a.to_opref(), &mut max_var);
        }
        if let Some(fa) = op.getfailargs() {
            for a in fa.iter() {
                widen_value_id(a.to_opref(), &mut max_var);
            }
        }

        if op.opcode.is_guard() || op.opcode == OpCode::Finish {
            let fail_args: Vec<OpRef> = op
                .getfailargs()
                .map(|fa| fa.iter().map(|a| a.to_opref()).collect())
                .unwrap_or_else(|| op.getarglist().iter().map(|a| a.to_opref()).collect());
            let fail_arg_types = op
                .get_fail_arg_types()
                .unwrap_or_else(|| fail_args.iter().map(|_| Type::Int).collect());

            let meta_descr = op.getdescr();
            // `regalloc.py consider_guard_value` — stamp the per-value
            // counter here, where the native backends stamp it during guard
            // layout, so the `status == 0` gate of `store_hash`
            // (`compile.py`) leaves it alone and `must_compile` hashes
            // the (guard, failing value) pair. Without it a guard whose failing
            // value never repeats accumulates in one bucket and compiles
            // another bridge every `trace_eagerness` failures, without bound.
            // The compared operand of a GUARD_VALUE is a promoted value the
            // resume re-derives, so it is almost never one of the guard's own
            // fail arguments: 0 of 16 on a synthetic polymorphic call site, 0
            // of 21 on pyre/bench/fannkuch.py. Reading its slot out of the fail
            // arguments alone therefore left the stamp unwritten on nearly
            // every GUARD_VALUE, and an unstamped guard keeps the per-guard
            // hash: every `trace_eagerness` failures compile another bridge for
            // a value that never repeats, without bound
            // (`foriter_make_function_body`: 47 bridges, 0 of them entered).
            //
            // `regalloc.py consider_guard_value` hands
            // `all_reg_indexes[x.value]` — a deadframe slot, not a fail-argument
            // position — so an operand the guard does not carry is still
            // readable. This backend's slot space is the exit's own frame
            // slots, so give such an operand one past the last fail argument
            // and spill it there (`emit_guard_fail_args_spill`);
            // `normal_frame_value_slots` reserves it and
            // `resolve_guard_value_operand` reads it back through
            // `get_value_direct`.
            // Stamping, sizing and emission use the same live-position mask.
            // A compared operand carried only in a logical hole still needs
            // its own readable counter slot.
            let counter_value_spill = counter_value_spill(op, &fail_args);
            if op.opcode == OpCode::GuardValue
                && let Some(fd) = meta_descr.as_ref().and_then(|d| d.as_fail_descr())
            {
                let arg0 = op.arg(0).to_opref();
                // The parked case is stamped after this loop, where the
                // trace-wide slot is known.
                if counter_value_spill.is_none()
                    && let Some(idx) = live_fail_arg_position(op, &fail_args, arg0)
                {
                    let type_tag = match fail_arg_types.get(idx) {
                        Some(Type::Ref) => majit_backend::STATUS_TY_REF,
                        Some(Type::Float) => majit_backend::STATUS_TY_FLOAT,
                        _ => majit_backend::STATUS_TY_INT,
                    };
                    fd.make_a_counter_per_value(idx as u32, type_tag);
                }
            }
            let mut next_slot = 0;
            let fail_locs = live_fail_arg_mask(meta_descr.as_ref(), fail_args.len())
                .into_iter()
                .map(|live| {
                    live.then(|| {
                        let slot = next_slot;
                        next_slot += 1;
                        slot
                    })
                })
                .collect();
            guards.push(GuardExit {
                fail_locs,
                fail_index,
                fail_arg_refs: fail_args,
                fail_arg_types,
                is_finish: op.opcode == OpCode::Finish,
                force_ref_home_indices: Vec::new(),
                publishes_finish_gcmap: false,
                descr_cell: 0,
                exit_gcmap_ptr: 0,
                counter_value_spill,
                bridge_cell: 0,
                meta_descr,
            });
            fail_index += 1;
        }
    }

    (guards, max_var)
}

/// Park every GUARD_VALUE counter on one slot past the owner's value area.
///
/// Merged analysis concatenates region InputArgs into one id namespace; those
/// ids are not simultaneous entry slots. The physical slot and the stamp
/// index are the owner's function-entry arity, the same width
/// `normal_frame_value_slots` reserved when the token froze.
fn park_guard_value_counters(guards: &mut [GuardExit], entry_arity: usize) {
    if guards.iter().all(|g| g.counter_value_spill.is_none()) {
        return;
    }
    let value_area = guards
        .iter()
        .map(|g| live_fail_arg_extent(g.meta_descr.as_ref(), g.fail_arg_refs.len()))
        .max()
        .unwrap_or(0)
        .max(entry_arity);
    let physical_value_area = guards
        .iter()
        .map(|g| live_fail_arg_count(g.meta_descr.as_ref(), g.fail_arg_refs.len()))
        .max()
        .unwrap_or(0)
        .max(entry_arity);
    for g in guards {
        if g.counter_value_spill.is_some() {
            g.fail_locs.resize(value_area + 1, None);
            g.fail_locs[value_area] = Some(physical_value_area);
        }
        if let Some(operand) = g.counter_value_spill
            && let Some(fd) = g.meta_descr.as_ref().and_then(|d| d.as_fail_descr())
        {
            let type_tag = match operand.ty() {
                Some(Type::Ref) => majit_backend::STATUS_TY_REF,
                Some(Type::Float) => majit_backend::STATUS_TY_FLOAT,
                _ => majit_backend::STATUS_TY_INT,
            };
            fd.make_a_counter_per_value(value_area as u32, type_tag);
        }
    }
}

/// Number of guard/finish exits a module will need bridge-dispatch cells for.
/// Cell ownership belongs to the compiled trace, outside module generation.
pub fn guard_exit_count(inputargs: &[InputArgRc], ops: &[Op]) -> usize {
    collect_guards_and_vars(inputargs, ops).0.len()
}

/// Dense wasm-local assignment and type lookup for each addressed SSA value.
fn collect_value_types(
    inputargs: &[InputArgRc],
    ops: &[Op],
    num_vars: u32,
    first_local: u32,
) -> ValueLocals {
    ValueLocals::collect(inputargs, ops, num_vars, first_local)
}

/// Assign each Ref-typed value (input arg or op result) a dense home-slot
/// index, keyed by its value id (`raw()`), the same id its wasm local uses
/// (the dense wasm local is assigned separately). Input args and op results
/// share one value-id space (see
/// `collect_guards_and_vars`), so a single map covers both. Int / Float /
/// Void values are skipped — only GC references need a forwarding home.
/// Allocate the per-guard bridge-slot cell array for inter-trace chaining and
/// return `(base address in the shared linear memory, owner)`.
///
/// One zero-initialised i32 cell per guard, indexed by `fail_index`;
/// `compile_bridge` writes the bridge's table slot into the matching cell. The
/// returned `Box<[u32]>` is the array's owner — the caller stores it on the
/// compiled loop (or, for a bridge, on its source loop's owned-cells list) so
/// it is freed on `Drop`. The base address aliases the box's heap buffer, which
/// is stable across moves of the owning box, so baking it into the module here
/// stays valid for the loop's lifetime.
///
/// On native the trace is never executed, so the dispatch is omitted and no
/// cells are needed — returning `(0, None)` keeps the emitted module
/// byte-identical to the pre-chaining output and allocates nothing.
pub fn alloc_bridge_cells(num_guards: usize) -> (u32, Option<Box<[u32]>>) {
    // `Box<[u32; 0]>` has a non-null dangling `as_mut_ptr()`.  The pointer is
    // not a dispatch table, so preserve the no-dispatch representation even
    // on wasm where allocating that empty box would otherwise make the
    // epilogue load an uninitialised bridge-slot local.
    if num_guards == 0 {
        return (0, None);
    }
    #[cfg(target_arch = "wasm32")]
    {
        let mut cells = vec![0u32; num_guards].into_boxed_slice();
        let base = cells.as_mut_ptr() as usize as u32;
        (base, Some(cells))
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = num_guards;
        (0, None)
    }
}

/// Parameters for the guest→guest `CALL_ASSEMBLER` `call_indirect` arm.
/// `emit_ca == false` (the default) keeps every emitted module byte-identical
/// to the pre-feature backend.
#[derive(Clone, Default)]
pub struct CaParams {
    /// Emit the dedicated `CALL_ASSEMBLER` arm.
    pub emit_ca: bool,
    /// Geometry and entry metadata, keyed by the CALL_ASSEMBLER callee token.
    /// Every entry describes exactly the JitFrame allocated for that target.
    pub targets: HashMap<u64, CaTarget>,
    /// `__indirect_function_table` slot of `wasm_ca_resume_deopt`
    /// (`lib.rs::ca_deopt_helper_slot`). When a callee `call_indirect` returns a
    /// non-finish `fail_index` (a guard deopt), the CA arm `call_indirect`s this
    /// slot to blackhole-resume the callee on the host and read back its result,
    /// instead of trapping. `0` (unset) ⇒ no helper, so `compile_bridge` declines
    /// the CA lift before reaching codegen.
    pub deopt_helper_slot: u32,
    /// `__indirect_function_table` slot of `wasm_jit_ca_push_frame`.
    /// Used when `jf_top` is not published: the shadow stack is host TLS.
    /// `(i64)->i64`.
    pub ca_push_fn_ptr: i64,
    /// `__indirect_function_table` slot of `lib.rs::wasm_jit_ca_pop_frame`,
    /// called on CA-arm exit to pop the callee frame off the jitframe shadow
    /// stack (strict LIFO).
    pub ca_pop_fn_ptr: i64,
    /// `__indirect_function_table` slot of `lib.rs::wasm_jit_ca_reload_frame`,
    /// called after the recursive call to recover this level's possibly-moved
    /// nursery frame from the jitframe shadow stack.
    pub ca_reload_fn_ptr: i64,
    /// Address of the active jitframe shadow-stack top cell, baked for every
    /// trace body so post-collecting-call local-0 reloads can match
    /// assembler.py without a helper round trip. `None` keeps the existing
    /// helper/trampoline behavior when compilation has no active GC.
    pub jf_top_addr: Option<u32>,
    /// `__indirect_function_table` slot of
    /// `lib.rs::wasm_jit_ca_reload_caller_frame`, called while the callee is
    /// still pushed to recover this invocation's possibly-moved local-0 frame.
    pub ca_reload_caller_fn_ptr: i64,
    /// Active-GC state for the direct CA-only inline allocation/frame path.
    /// `None` retains the helpers (including under gc_stress).
    pub inline: Option<CaInlineParams>,
    /// `build_home_gcmap` pointer published after the fresh-entry home/input
    /// stores, and again on each keyed LABEL resume after those slots are
    /// already valid or newly marked ones have been nulled. A resume whose
    /// live map already covers this pointer leaves `jf_gcmap` unchanged.
    /// Used only when [`Self::compute_home_gcmap`] is false. Zero leaves
    /// `jf_gcmap` unset in the generated module (tests). assembler.py
    /// writes `jf_gcmap` at safepoints once those slots are live.
    pub home_gcmap_ptr: i64,
    /// When set, leak a map from this module's `RefHomes` and LABEL captures
    /// (raised to the `home_gcmap_min_*` floors) instead of
    /// [`Self::home_gcmap_ptr`]. Re-emission and out-of-line bridges need
    /// the floors so a later keyed tail-call cannot drop the source loop's
    /// already-initialized homes.
    pub compute_home_gcmap: bool,
    pub home_gcmap_min_ordinary: usize,
    pub home_gcmap_min_labels: usize,
    /// True when `home_gcmap_min_*` is a previous publication's floor
    /// (re-emission or a bridge that must cover the source loop). False
    /// on a first compile, where min=0 must not wipe live homes on keyed
    /// resume. Distinguishes a 0→N merge from an initial compile.
    pub home_gcmap_has_prior: bool,
    /// True only when a re-emitted module must null newly marked LABEL
    /// homes for a keyed caller whose map did not include them. Re-emission
    /// itself leaves this false and drops stale owner attachments when
    /// the LABEL tail grows; key-0 still clears the full used-label range.
    /// A compiled bridge with more captures than its source also leaves
    /// this false: it writes those slots on the first crossing.
    pub home_gcmap_null_grown_labels: bool,
    /// `LoopAsmResources` that owns maps published by this build. `0` leaks
    /// the map (`allocate_gcmap`'s `Box::into_raw`) for a direct codegen test.
    pub gcmap_sink: usize,
    /// Guest address of `[descr_cell, gcmap]` pairs, one pair per guard,
    /// indexed by `guard_idx - fail_index_base`. `0` in a direct codegen
    /// test: the exit still loads a pair, from offset 0, so two builds of
    /// the same trace stay byte-identical. `compile_loop` parks the real
    /// table on `LoopAsmResources` and passes its address.
    pub exit_table_base: u32,
    /// Guest address of this owner's `{generation, slot}` cell. `0` emits
    /// no back-edge check (codegen tests, host-less compiles). A local JUMP
    /// loads `generation` the way `GuardNotInvalidated` loads its flag and,
    /// when the cell is newer than the generation baked into this module,
    /// tail-calls `slot` with the same jitframe.
    pub resume_entry_addr: u32,
    /// Generation this module was compiled as. The running copy redirects
    /// only when the cell is strictly greater, so a module compiled ahead
    /// of an unbumped cell does not bounce back to the old slot.
    pub resume_generation: u32,
    /// `compile_loop` / `compile_bridge` snapshot of this cpu's six exit
    /// cells (`runner.rs` `AttachedDescrPtrs` captured at entry). `0` is
    /// unattached: direct codegen tests leave the fields at default.
    pub attached: majit_backend::AttachedDescrPtrs,
    /// `__indirect_function_table` slot of `wasm_realloc_frame`. Zero omits
    /// `_check_frame_depth` so a frame that already fits stays byte-identical.
    /// `(i64 items, i64 depth) -> i64` at residual type base + 2.
    pub realloc_fn_ptr: i64,
    /// Signed item count `_check_frame_depth` compares against. Zero uses
    /// [`FrameGeometry::signed_item_count`] of this module. `assemble_bridge`
    /// passes `max(frame_depth, jump target jfi_frame_depth)` without changing
    /// this module's spill or home offsets.
    pub frame_depth_items: usize,
    /// Geometry of this module's cross-module JUMP target. `None` stores at
    /// this module's own offsets. An inlined region carries the same geometry
    /// on [`ExternalJump`] instead, because the owner's params describe the
    /// owner loop.
    pub external_jump_frame: Option<FrameGeometry>,
}

/// Per-CALL_ASSEMBLER target dispatch baked into the corresponding wasm arm.
/// Frame geometry is deliberately not stored here: PyPy permits
/// `redirect_call_assembler` to replace a temporary callback with a deeper
/// real loop, so every mutable target field is loaded through the stable entry.
#[derive(Clone)]
pub struct CaTarget {
    /// Stable guest-memory [`WasmCaDispatchEntry`](crate::failguard::WasmCaDispatchEntry)
    /// address.  The call slot, finish index, and deopt metadata are loaded
    /// through it at runtime so pending->real install and redirects do not
    /// require patching an already-compiled wasm module.
    pub dispatch_entry: u32,
}

/// Direct CA fast-path values baked at bridge compilation time.
#[derive(Clone, Copy)]
pub struct CaInlineParams {
    pub nursery_free_addr: u32,
    pub nursery_top_addr: u32,
    pub jf_top_addr: u32,
    pub jf_limit_addr: u32,
    pub jitframe_tid: u32,
    /// `max_nursery_object_size` — a runtime frame whose aligned total
    /// meets or exceeds this goes through the collecting helper, the
    /// same exclusive bound `can_use_nursery_malloc` uses.
    pub large_threshold: usize,
}

/// Inline nursery-bump fast-path parameters for post-rewrite
/// `CallMallocNursery*` ops (rewrite.py's malloc fast path over the
/// gc.py `get_nursery_free_addr`/`get_nursery_top_addr` surface,
/// which the x86 backend lowers as `malloc_cond`: load free, bump, compare
/// top, call the slow path only on overflow). `None` keeps every allocation
/// on the `wasm_jit_alloc` helper call.
#[derive(Clone)]
pub struct NurseryAllocParams {
    /// Linear-memory address of the GC's `nursery_free` bump pointer.
    pub free_addr: u32,
    /// Linear-memory address of the GC's `nursery_top` limit pointer.
    pub top_addr: u32,
    /// `max_nursery_object_size` / `JIT_max_size_of_young_obj` — the exclusive
    /// `large_object` boundary, so the inline path applies strictly below it.
    pub large_threshold: usize,
    /// Type ids whose allocation is a plain bump + header write (no
    /// destructor / weakref side-list registration).
    pub plain_tids: std::collections::HashSet<u32>,
}
/// `Nursery::alloc` 8-aligns the total. The inline VarsizeFrame and
/// headerless bumps write `nursery_free` themselves, so a wasm32 size that
/// is only word-aligned has to be raised here or the next object header is
/// misaligned.
fn aligned_varsize_frame_bump(size: i64) -> Option<u32> {
    let size = u32::try_from(size).ok()?;
    Some(size.checked_add(7)? & !7)
}

/// `gen_initialize_tid` immediately after `CallMallocNursery`: a constant
/// Signed `HDR.tid` store of the type id into the header word at
/// `obj - HDR_SIZE`.
struct NurseryTidStore {
    tid: i64,
    offset: i64,
    width: usize,
}

fn nursery_header_tid_store(
    next: &Op,
    malloc_result: OpRef,
    constants: &indexmap::IndexMap<u32, i64>,
) -> Option<NurseryTidStore> {
    if next.opcode != OpCode::GcStore || next.num_args() < 4 {
        return None;
    }
    if next.arg(0).to_opref() != malloc_result {
        return None;
    }
    let offset = const_operand_value(constants, next.arg(1).to_opref())?;
    if offset != -(GcHeader::SIZE as i64) {
        return None;
    }
    let width = const_operand_value(constants, next.arg(3).to_opref())?;
    if width != std::mem::size_of::<usize>() as i64 {
        return None;
    }
    let tid = const_operand_value(constants, next.arg(2).to_opref())?;
    Some(NurseryTidStore {
        tid,
        offset,
        width: width as usize,
    })
}

/// `CALL_MALLOC_NURSERY_VARSIZE` slow arm: the arity-5 array helper
/// (`wasm_jit_alloc_array(type_id, base_size, item_size, length, len_offset)`).
/// x86 `MallocCondVarsizeSlowPath.generate_body` `push_gcmap`s before the
/// call; the caller's `emit_reload_refs_from_homes` pops it.
fn emit_alloc_array_helper(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    type_id: i64,
    base_size: i64,
    itemsize: OpRef,
    length: OpRef,
    len_offset: i64,
    new_array_fn_ptr: i64,
    residual_type_base: u32,
    site_gcmap: &[i64],
    op_idx: usize,
) {
    sink.i64_const(type_id);
    sink.i64_const(base_size);
    emit_resolve(sink, constants, value_types, itemsize);
    emit_resolve(sink, constants, value_types, length);
    sink.i64_const(len_offset);
    sink.i32_const(new_array_fn_ptr as i32);
    emit_push_site(sink, site_gcmap, op_idx);
    sink.call_indirect(0, residual_type_base + 5);
}

pub(crate) const BUILTIN_STRING_HASH_OFFSET: usize = majit_backend::BUILTIN_STRING_HASH_OFFSET;
pub(crate) const BUILTIN_STRING_HASH_SIZE: usize = std::mem::size_of::<usize>();
pub(crate) const BUILTIN_STRING_LEN_OFFSET: usize = majit_backend::BUILTIN_STRING_LEN_OFFSET;
pub(crate) const BUILTIN_STR_TOKEN_BASE_SIZE: usize = majit_backend::BUILTIN_STR_TOKEN_BASE_SIZE;
pub(crate) const BUILTIN_UNICODE_TOKEN_BASE_SIZE: usize = 2 * std::mem::size_of::<usize>();

#[derive(Debug)]
struct BuiltinFieldDescr {
    offset: usize,
    field_size: usize,
    field_type: Type,
    signed: bool,
}

impl majit_ir::Descr for BuiltinFieldDescr {
    fn as_field_descr(&self) -> Option<&dyn majit_ir::FieldDescr> {
        Some(self)
    }
}

impl majit_ir::FieldDescr for BuiltinFieldDescr {
    fn offset(&self) -> usize {
        self.offset
    }
    fn field_size(&self) -> usize {
        self.field_size
    }
    fn field_type(&self) -> Type {
        self.field_type
    }
    fn is_field_signed(&self) -> bool {
        self.signed
    }
}

#[derive(Debug)]
struct BuiltinArrayDescr {
    base_size: usize,
    item_size: usize,
    type_id: u32,
    item_type: Type,
    signed: bool,
    len_descr: Arc<BuiltinFieldDescr>,
}

impl majit_ir::Descr for BuiltinArrayDescr {
    fn as_array_descr(&self) -> Option<&dyn majit_ir::ArrayDescr> {
        Some(self)
    }
}

impl majit_ir::ArrayDescr for BuiltinArrayDescr {
    fn base_size(&self) -> usize {
        self.base_size
    }
    fn item_size(&self) -> usize {
        self.item_size
    }
    fn type_id(&self) -> u32 {
        self.type_id
    }
    fn item_type(&self) -> Type {
        self.item_type
    }
    fn is_item_signed(&self) -> bool {
        self.signed
    }
    fn len_descr(&self) -> Option<&dyn majit_ir::FieldDescr> {
        Some(self.len_descr.as_ref())
    }
}

pub(crate) fn builtin_string_array_descr(opcode: OpCode) -> Option<majit_ir::DescrRef> {
    let (base_size, item_size, type_id) = match opcode {
        OpCode::Newstr
        | OpCode::Strlen
        | OpCode::Strgetitem
        | OpCode::Strsetitem
        | OpCode::Copystrcontent => (
            BUILTIN_STR_TOKEN_BASE_SIZE,
            1,
            majit_gc::lowlevel_str_type_id(),
        ),
        OpCode::Newunicode
        | OpCode::Unicodelen
        | OpCode::Unicodegetitem
        | OpCode::Unicodesetitem
        | OpCode::Copyunicodecontent => (
            BUILTIN_UNICODE_TOKEN_BASE_SIZE,
            4,
            majit_gc::lowlevel_unicode_type_id(),
        ),
        _ => return None,
    };
    let len_descr = Arc::new(BuiltinFieldDescr {
        offset: BUILTIN_STRING_LEN_OFFSET,
        field_size: BUILTIN_STRING_HASH_SIZE,
        field_type: Type::Int,
        signed: false,
    });
    Some(Arc::new(BuiltinArrayDescr {
        base_size,
        item_size,
        type_id,
        item_type: Type::Int,
        signed: false,
        len_descr,
    }))
}

pub(crate) fn builtin_string_hash_field_descr(opcode: OpCode) -> Option<majit_ir::DescrRef> {
    if !matches!(opcode, OpCode::Strhash | OpCode::Unicodehash) {
        return None;
    }
    Some(Arc::new(BuiltinFieldDescr {
        offset: BUILTIN_STRING_HASH_OFFSET,
        field_size: BUILTIN_STRING_HASH_SIZE,
        field_type: Type::Int,
        signed: true,
    }))
}

/// Cranelift/dynasm `inject_builtin_string_descrs`: vstring mints
/// `NEWSTR/1/r` with no descr, and rewrite.py fills it from `str_descr`.
///
/// `setdescr` is interior-mutable, so this takes a shared slice: the GC
/// rewrite runs it on the incoming stream before it boxes the ops, and
/// codegen runs it again on a stream that never went through the rewrite.
pub(crate) fn inject_builtin_string_descrs(ops: &[Op]) {
    for op in ops {
        if op.has_descr() {
            continue;
        }
        if let Some(descr) = builtin_string_array_descr(op.opcode) {
            op.setdescr(descr);
        } else if let Some(descr) = builtin_string_hash_field_descr(op.opcode) {
            op.setdescr(descr);
        }
    }
}

fn needs_builtin_string_descr(op: &Op) -> bool {
    !op.has_descr()
        && (builtin_string_array_descr(op.opcode).is_some()
            || builtin_string_hash_field_descr(op.opcode).is_some())
}
/// `__indirect_function_table` indices of the allocation helpers a compiled
/// trace calls for `CallMallocNursery*` slow paths.
#[derive(Clone, Copy, Default)]
pub struct AllocHelpers {
    pub new_fn_ptr: i64,
    pub new_array_fn_ptr: i64,
    pub headerless_fn_ptr: i64,
    pub threadlocal_fn_ptr: i64,
    pub fmod_fn_ptr: i64,
    /// callbuilder.py `write_real_errno` / `read_real_errno`, `(i64)->i64`.
    pub write_real_errno_fn_ptr: i64,
    pub read_real_errno_fn_ptr: i64,
}

type BuildWasmModuleOutput = (Vec<u8>, Vec<GuardExit>, usize, usize);

/// Counts entries into an out-of-line bridge module and calls out once there
/// have been enough of them to pay for merging that bridge into its owner.
///
/// The callee only records the request — the merge itself runs after the trace
/// returns, because the host holds the driver mutably across the whole
/// compiled run.
#[derive(Clone, Copy)]
pub struct InlineTripProbe {
    /// Address of this bridge's own `u64` entry counter.
    pub counter_addr: u32,
    /// Entry count at which the callback fires — once, on equality.
    pub threshold: u64,
    /// `__indirect_function_table` index of the `(i64) -> i64` callback.
    pub trip_fn_ptr: i64,
    /// The callback's only argument: address of the `PendingInlineSlot`.
    pub pending_slot: i64,
}

/// Owned inputs for one wasm module build.  A loop retains this after its
/// first build so it can emit the same trace again without revisiting mutable
/// backend state such as the constants pool or GC-reference interning pass.
pub struct ModuleBuildInputs {
    pub inputargs: Vec<InputArgRc>,
    /// These are the post-intern operations.  Re-interning them would lose the
    /// already allocated GC-table base encoded by `gc_table_base`.
    pub ops: Vec<Op>,
    /// Loop-closing bridge regions emitted inside this loop's wasm function.
    /// Each is retained in its own trace's numbering; `build_wasm_module`
    /// rebases them onto a private id range before merging, because the owner
    /// and every region number their values independently from zero.
    pub inlined_bridges: Vec<InlinedBridge>,
    pub constants: indexmap::IndexMap<u32, i64>,
    pub vtable_offset: Option<usize>,
    pub classptr_to_typeid: HashMap<i64, u32>,
    pub guard_gc_type_info: GuardGcTypeInfo,
    pub alloc: AllocHelpers,
    pub wb: WriteBarrierHelpers,
    pub nursery: Option<NurseryAllocParams>,
    pub invalidated_flag_addr: u32,
    pub gc_table_base: u32,
    /// `GcTable::compile_key` for each slot at `gc_table_base`, in order.
    pub gc_const_keys: Vec<usize>,
    pub fail_index_base: u32,
    pub bridge_cells_base: u32,
    /// Per-guard cell addresses in collect order. A non-zero entry is the
    /// guard's existing cell; re-emission passes these so the new module
    /// loads the same cells instead of a fresh array.
    pub guard_cell_addrs: Vec<u32>,
    /// A bridge reached from an armed guard takes its fail values as `i64`
    /// parameters after the frame pointer. Float bits use the same i64 carrier,
    /// so a single function type per arity covers every failure signature.
    pub bridge_entry_arity: Option<usize>,
    /// Emit fixed-arity guard-to-bridge parameter tail-call arms for this module.
    pub bridge_param_dispatch: bool,
    /// Guest-memory counters baked into an armed trace-entry census module.
    /// `None` keeps the generated module byte-identical to the normal path.
    pub trace_entry_census: Option<crate::TraceEntryCensusStorage>,
    /// Entry counter and callback for a bridge whose merge into its owner is
    /// deferred until it has been crossed often enough to pay for the owner's
    /// re-emission. `None` keeps the generated module byte-identical.
    pub inline_trip: Option<InlineTripProbe>,
    pub external_jump_slot: u32,
    pub external_jump_key: u32,
    /// The target loop's `trace_wide` table slot, when it published one. The
    /// slot the host appends beside `external_jump_slot` holds a fixed-arity
    /// parameter entry, so a loop-closing JUMP can hand its args over as wasm
    /// parameters instead of writing them to frame slots the target's narrow
    /// shim would immediately read back. `0` = the target has no wide entry.
    pub external_jump_wide_slot: u32,
    pub frame: FrameGeometry,
    pub ca: CaParams,
}

/// The cross-module target of a region's closing JUMP, as `compile_bridge`
/// resolved it for the bridge the region stands in for.
///
/// The target's fixed-arity parameter entry is deliberately absent. Calling it
/// needs its wasm type declared in the calling module, and a module declares
/// that type from its OWN `external_jump_wide_slot` — which the owner of a
/// merged region does not have. The narrow entry a region tail-calls instead
/// reads the same values back out of the frame slots stored here, so the merge
/// costs that round trip and nothing else.
#[derive(Clone, Debug)]
pub struct ExternalJump {
    /// `__indirect_function_table` slot of the target loop's entry.
    pub slot: u32,
    /// Resume-at-LABEL dispatch key: `target label ordinal + 1`, or `0` when
    /// the target is not peeled.
    pub key: u32,
    /// Target loop's frame. Spill and dispatch-key stores use these offsets
    /// (`remap_frame_layout`); the narrow shim reloads the same words.
    pub frame: FrameGeometry,
}

pub struct InlinedBridge {
    /// Per-trace fail index of the guard that enters this region.
    pub source_fail_index: u32,
    /// Where this region's closing JUMP goes when it names a LABEL published
    /// by ANOTHER module. `None` is the in-module case: the JUMP rebinds the
    /// owner's own loop args and lowers to a `br`.
    pub external_jump: Option<ExternalJump>,
    /// Emit this region's block outside the header `loop` and its body past
    /// that loop's `end`, rather than inside it.
    ///
    /// Forced when the source guard is in the peeled preamble, which has not
    /// entered the loop the inside blocks are opened in. It is also the only
    /// placement left to a region attaching AFTER one of those, because merging
    /// is append-only: an outside region's ops are the tail of the merged
    /// stream, and splicing anything ahead of them would renumber the exits its
    /// own sub-bridges' dispatch cells are keyed by.
    pub outside_loop: bool,
    pub trace_id: u64,
    pub inputargs: Vec<InputArgRc>,
    pub ops: Vec<Op>,
    /// Base of this already-interned region's GC table. Each region retains
    /// its own roots; codegen selects it by the LoadFromGcTable producer.
    pub gc_table_base: u32,
    /// `GcTable::compile_key` for each slot at `gc_table_base`, in order.
    pub gc_const_keys: Vec<usize>,
    /// The constant pool registered for this region's own trace. A pool is
    /// per-trace (`Backend::set_constants_pool` names the next compile), and
    /// its value-id keys — the folded values that have no producing op — are
    /// in that trace's numbering, so the merge rebases them with the region.
    pub constants: indexmap::IndexMap<u32, i64>,
}

/// Whether the exact operation stream emitted for `inputs` has a local loop
/// back-edge target.  An inline bridge transfers with `br`, which can only
/// target the wasm `loop` opened for that LABEL.
pub fn merged_stream_has_loop_label(inputs: &ModuleBuildInputs) -> bool {
    let mut ops = inputs.ops.clone();
    for bridge in &inputs.inlined_bridges {
        ops.extend(bridge.ops.iter().cloned());
    }
    find_loop_label_index(&ops).is_some_and(|label_idx| label_idx < inputs.ops.len())
}

/// Whether the guard at exit ordinal `fail_index` sits in the peeled preamble,
/// ahead of the loop header LABEL.
///
/// A region attaching to such a guard cannot take the loop-body placement: its
/// block would be opened inside the `loop` the preamble has not entered.
/// `build_function` gives this class its own blocks outside that loop and
/// emits their bodies after it closes, which is why the ordinal decides the
/// emission order of the merged stream.
///
/// `fail_index` is the exit ordinal within the owner's own stream — the
/// numbering `collect_guards_and_vars` assigns and `InlinedBridge` records as
/// `source_fail_index`. The merged stream appends every region after the
/// owner, so the same ordinal reads the same guard there.
pub fn source_guard_precedes_loop_label(ops: &[Op], fail_index: u32) -> bool {
    match (exit_op_index(ops, fail_index), find_loop_label_index(ops)) {
        (Some(pos), Some(label_idx)) => pos < label_idx,
        _ => false,
    }
}

/// An outside-loop region skips its target LABEL's capture loader. The
/// source path must therefore have crossed that LABEL already. Share this
/// structural check between emission and deferred-inline eligibility.
/// A failed install restores the out-of-line cell, so a compile-time
/// miss (sibling peel not yet attached) can still arm a trip.
pub fn outside_region_labels_initialized(
    owner_ops: &[Op],
    source_fail_index: u32,
    region_ops: &[Op],
) -> bool {
    let Some(guard_pos) = exit_op_index(owner_ops, source_fail_index) else {
        return false;
    };
    region_ops
        .iter()
        .filter(|op| op.opcode == OpCode::Jump)
        .all(|jump| {
            find_jump_target_label_index(owner_ops, jump).is_some_and(|label| label < guard_pos)
        })
}

/// Whether these ops carry a `GUARD_NOT_INVALIDATED`, so their validity is
/// watched through the invalidation flag of whichever module emits them.
pub fn has_invalidation_guard(ops: &[Op]) -> bool {
    ops.iter()
        .any(|op| op.opcode == OpCode::GuardNotInvalidated)
}

/// Position in `ops` of the exit with ordinal `fail_index`.
fn exit_op_index(ops: &[Op], fail_index: u32) -> Option<usize> {
    ops.iter()
        .enumerate()
        .filter(|(_, op)| op.opcode.is_guard() || op.opcode == OpCode::Finish)
        .nth(fail_index as usize)
        .map(|(pos, _)| pos)
}

impl Clone for InlinedBridge {
    fn clone(&self) -> Self {
        Self {
            source_fail_index: self.source_fail_index,
            external_jump: self.external_jump.clone(),
            outside_loop: self.outside_loop,
            trace_id: self.trace_id,
            inputargs: self.inputargs.iter().cloned().collect(),
            ops: self.ops.clone(),
            gc_table_base: self.gc_table_base,
            gc_const_keys: self.gc_const_keys.clone(),
            constants: self.constants.clone(),
        }
    }
}

impl Clone for ModuleBuildInputs {
    fn clone(&self) -> Self {
        Self {
            inputargs: self.inputargs.iter().cloned().collect(),
            ops: self.ops.clone(),
            inlined_bridges: self.inlined_bridges.clone(),
            constants: self.constants.clone(),
            vtable_offset: self.vtable_offset,
            classptr_to_typeid: self.classptr_to_typeid.clone(),
            guard_gc_type_info: self.guard_gc_type_info.clone(),
            alloc: self.alloc,
            wb: self.wb,
            nursery: self.nursery.clone(),
            invalidated_flag_addr: self.invalidated_flag_addr,
            gc_table_base: self.gc_table_base,
            gc_const_keys: self.gc_const_keys.clone(),
            fail_index_base: self.fail_index_base,
            bridge_cells_base: self.bridge_cells_base,
            guard_cell_addrs: self.guard_cell_addrs.clone(),
            bridge_entry_arity: self.bridge_entry_arity,
            bridge_param_dispatch: self.bridge_param_dispatch,
            trace_entry_census: self.trace_entry_census,
            inline_trip: self.inline_trip,
            external_jump_slot: self.external_jump_slot,
            external_jump_key: self.external_jump_key,
            external_jump_wide_slot: self.external_jump_wide_slot,
            frame: self.frame,
            ca: self.ca.clone(),
        }
    }
}

/// One past the highest value id `inputargs`/`ops` define or read. Mirrors the
/// `max_var` half of `collect_guards_and_vars` without its guard collection,
/// which stamps per-value counters onto guard descrs and must run once only.
fn value_id_end(inputargs: &[InputArgRc], ops: &[Op]) -> u32 {
    let mut end: u32 = 0;
    for ia in inputargs {
        if ia.index + 1 > end {
            end = ia.index + 1;
        }
    }
    for op in ops {
        if let Some(id) = result_value_raw(op) {
            end = end.max(id + 1);
        }
        for a in op.getarglist().iter() {
            widen_value_id(a.to_opref(), &mut end);
        }
        if let Some(fa) = op.getfailargs() {
            for a in fa.iter() {
                widen_value_id(a.to_opref(), &mut end);
            }
        }
    }
    end
}

/// Move every value id a region defines or reads up by `offset`, returning the
/// rebased region and the width of the id range it now occupies.
///
/// The owner trace and each region are separately recorded traces, so both
/// number their values from zero and their ids overlap. A region is entered by
/// `local.set`ting the id each of its input args carries
/// (`emit_guard_inline_bridge_move`) and leaves through the loop header, so an
/// id it shares with an owner value that is live across the back edge
/// overwrites that value for every following iteration. Rebasing onto a
/// disjoint range is what makes the merged stream's single local namespace
/// sound.
///
/// `TempVar` ids live in a reserved high strip and constants in their own
/// namespace; neither indexes a value local, so both pass through unchanged.
fn rebase_region_value_ids(
    bridge: &InlinedBridge,
    offset: u32,
) -> Result<(InlinedBridge, u32), BackendError> {
    use majit_ir::operand::Operand;

    let shift = |r: OpRef| -> OpRef {
        match value_box_raw(r) {
            Some(id) => r.with_raw(id + offset),
            None => r,
        }
    };

    let width = value_id_end(&bridge.inputargs, &bridge.ops);
    // `with_raw` keeps the variant, but the emitters classify by raw payload
    // (`OpRef::raw_is_constant`), so an id shifted to or past the limit reads
    // as a constant and its result is skipped. Decline instead: the merged
    // stream is an optimization, and no renumbering is correct once the
    // region's range no longer fits below the limit.
    if offset
        .checked_add(width)
        .is_none_or(|end| end > OpRef::VALUE_ID_LIMIT)
    {
        return Err(BackendError::Unsupported(format!(
            "wasm backend: inlined bridge value ids exceed the value-id space \
             (offset {offset}, width {width})"
        )));
    }
    let inputargs: Vec<InputArgRc> = bridge
        .inputargs
        .iter()
        .map(|ia| InputArgRc::new(InputArg::from_type(ia.tp.get(), ia.index + offset)))
        .collect();
    // `Op::clone` gives the copy its own arg/failarg slots, but the operands in
    // them keep pointing at the region's original producers, whose `pos` this
    // must not touch — the region is retained for the next re-emission. So each
    // moved reference is rebound to a synthetic producer carrying the new id.
    let ops: Vec<Op> = bridge.ops.to_vec();
    for op in &ops {
        op.pos().set(shift(op.pos().get()));
        for (i, arg) in op.getarglist().iter().enumerate() {
            let before = arg.to_opref();
            let after = shift(before);
            if after != before {
                op.setarg(i, Operand::bound_from_opref(after));
            }
        }
        if let Some(mut fail_args) = op.getfailargs() {
            let mut moved = false;
            for slot in fail_args.iter_mut() {
                let before = slot.to_opref();
                let after = shift(before);
                if after != before {
                    *slot = Operand::bound_from_opref(after);
                    moved = true;
                }
            }
            if moved {
                op.setfailargs(fail_args);
            }
        }
    }

    Ok((
        InlinedBridge {
            source_fail_index: bridge.source_fail_index,
            external_jump: bridge.external_jump.clone(),
            outside_loop: bridge.outside_loop,
            trace_id: bridge.trace_id,
            inputargs,
            ops,
            gc_table_base: bridge.gc_table_base,
            gc_const_keys: bridge.gc_const_keys.clone(),
            constants: bridge.constants.clone(),
        },
        width,
    ))
}

/// Build a wasm module from majit IR.
pub fn build_wasm_module(
    inputs: &ModuleBuildInputs,
) -> Result<BuildWasmModuleOutput, BackendError> {
    build_wasm_module_reporting_shortage(inputs, &mut None)
}

/// [`build_wasm_module`], also naming the frame layout shortage when the build
/// declines because `inputs.frame` is too small. A caller that may grow the
/// frame (`_check_frame_depth`) extends it by that shortage and builds again.
pub(crate) fn build_wasm_module_reporting_shortage(
    inputs: &ModuleBuildInputs,
    shortage_out: &mut Option<super::FrameShortage>,
) -> Result<BuildWasmModuleOutput, BackendError> {
    let ModuleBuildInputs {
        inputargs,
        ops,
        inlined_bridges,
        constants,
        vtable_offset,
        classptr_to_typeid,
        guard_gc_type_info,
        alloc,
        wb,
        nursery,
        invalidated_flag_addr,
        gc_table_base,
        gc_const_keys,
        fail_index_base,
        bridge_cells_base,
        guard_cell_addrs,
        bridge_entry_arity,
        bridge_param_dispatch,
        trace_entry_census,
        inline_trip,
        external_jump_slot,
        external_jump_key,
        external_jump_wide_slot,
        frame,
        ca,
    } = inputs;
    let ops_with_string_descrs;
    let ops: &[Op] = if ops.iter().any(needs_builtin_string_descr) {
        let cloned = ops.to_vec();
        inject_builtin_string_descrs(&cloned);
        ops_with_string_descrs = cloned;
        &ops_with_string_descrs
    } else {
        ops
    };
    // A bridge region has no function-entry loads, but its InputArgs and ops
    // still need locals, liveness, homes, guard exits, and call signatures.
    // Analyse the complete function as one stream while keeping `inputargs`
    // below as the actual function-entry list.
    // A normal module is emitted directly from its retained vectors.  Keep
    // that path allocation-free: code generation runs in the guest process,
    // so transient merged-stream allocations can otherwise perturb the next
    // collection boundary before any bridge is attached.
    let mut merged_inputargs = Vec::new();
    let mut merged_ops = Vec::new();
    let mut gc_table_bases = HashMap::new();
    let mut rebased_bridges: Vec<InlinedBridge> = Vec::new();
    let mut rebased_constants = indexmap::IndexMap::new();
    let (analysis_inputargs, analysis_ops): (&[InputArgRc], &[Op]) = if inlined_bridges.is_empty() {
        (inputargs, ops)
    } else {
        merged_inputargs.extend(inputargs.iter().cloned());
        merged_ops.extend(ops.iter().cloned());
        // The merged stream has one local namespace, so every region has to be
        // moved off the ids the owner and the earlier regions already use.
        rebased_constants = constants.clone();
        let mut next_value_id = value_id_end(inputargs, ops);
        for bridge in inlined_bridges {
            let (bridge, width) = rebase_region_value_ids(bridge, next_value_id)?;
            // The pool is keyed by value position for a folded value with no
            // producing op, so rebasing the region's ids moved its reads off
            // its own entries. Replay that window at the offset, and drop a
            // key another trace left inside it, or `unbound_pool_const_seeds`
            // either declines a resolvable value or seeds an unrelated one's
            // bits. Keys outside the window are left alone: rewriting them
            // would overwrite the entries the owner's own operations read.
            for id in 0..width {
                match bridge.constants.get(&id) {
                    Some(&bits) => {
                        rebased_constants.insert(id + next_value_id, bits);
                    }
                    None => {
                        rebased_constants.shift_remove(&(id + next_value_id));
                    }
                }
            }
            next_value_id += width;
            merged_inputargs.extend(bridge.inputargs.iter().cloned());
            for op in &bridge.ops {
                if op.opcode == OpCode::LoadFromGcTable {
                    gc_table_bases.insert(op.pos().get().raw(), bridge.gc_table_base);
                }
            }
            merged_ops.extend(bridge.ops.iter().cloned());
            rebased_bridges.push(bridge);
        }
        inject_builtin_string_descrs(&merged_ops);
        (&merged_inputargs, &merged_ops)
    };
    // Guard-entry moves and region emission must name the rebased ids, not the
    // ids the retained regions still carry.
    let emitted_bridges: &[InlinedBridge] = if inlined_bridges.is_empty() {
        inlined_bridges
    } else {
        &rebased_bridges
    };
    let region_spans = InlinedRegionSpan::collect(analysis_ops.len(), emitted_bridges);
    let constants = if inlined_bridges.is_empty() {
        constants
    } else {
        &rebased_constants
    };
    let (mut guards, num_vars) = collect_guards_and_vars(analysis_inputargs, analysis_ops);
    park_guard_value_counters(&mut guards, inputargs.len());
    record_physical_fail_locs(&mut guards, *frame);

    // An inlined bridge branches back into the owner with wasm `br`.  The
    // merged stream must therefore contain the local LABEL that opens the
    // wasm loop; a label-less cross-loop bridge has no in-function target.
    if !inlined_bridges.is_empty() && !merged_stream_has_loop_label(inputs) {
        return Err(BackendError::Unsupported(
            "wasm backend: inlined bridge stream has no local loop LABEL".into(),
        ));
    }
    for bridge in inlined_bridges {
        if bridge.ops.is_empty() {
            return Err(BackendError::Unsupported(
                "wasm backend: inlined bridge stream has an empty region".into(),
            ));
        }
        // The back edge rebinds the target LABEL's args from the region's
        // closing JUMP as a parallel move bounded by `min(jump, label)`, so a
        // JUMP naming fewer args leaves the remaining loop-carried locals
        // holding whatever the failing iteration left in them. Nothing
        // downstream reports that: wasm offset 0 is valid linear memory, so a
        // stale or zero Ref is read as an object instead of trapping.
        // `resolve_cross_loop_jump_target` refuses an arity mismatch before a
        // region is ever retained; this asserts the same invariant where the
        // move is emitted, rather than trusting a check in another file.
        if let Some(jump) = bridge
            .ops
            .last()
            .filter(|op| op.opcode == OpCode::Jump && bridge.external_jump.is_none())
        {
            let label_args = find_label_args(analysis_ops, jump);
            let jump_arity = jump.getarglist().len();
            if jump_arity < label_args.len() {
                return Err(BackendError::Unsupported(format!(
                    "wasm backend: inlined bridge JUMP rebinds {jump_arity} of the \
                     target LABEL's {} args",
                    label_args.len()
                )));
            }
        }
        // A region can carry a CALL_ASSEMBLER this build has no arm for. The
        // dedicated arm is selected by `ca.emit_ca`, which is decided when the
        // OWNER is compiled, and it reads the callee's geometry out of
        // `ca.targets`; a region merged in later brings its own callee. An op
        // that misses that arm does not fail — it falls through to the ordinary
        // residual-call arm, which lowers arg 0 as an
        // `__indirect_function_table` slot, and a CALL_ASSEMBLER's arg 0 is the
        // callee's first frame slot. That calls whatever the slot happens to
        // index and returns its result as the callee's, which is a silent wrong
        // answer rather than a trap. `wasm_unsupported_trace_reason` asks this
        // question of every trace's own ops; the merged stream is the one place
        // it is never re-asked, so ask it here.
        for op in &bridge.ops {
            if !op.opcode.is_call_assembler() {
                continue;
            }
            let target = op
                .getdescr()
                .and_then(|descr| descr.as_call_descr().and_then(|d| d.call_target_token()));
            if !ca.emit_ca || target.is_none_or(|token| !ca.targets.contains_key(&token)) {
                return Err(BackendError::Unsupported(format!(
                    "wasm backend: inlined bridge carries {:?}, which the owner \
                     build has no CALL_ASSEMBLER arm for",
                    op.opcode
                )));
            }
        }
        let source_guard = guards
            .get(bridge.source_fail_index as usize)
            .ok_or_else(|| {
                BackendError::Unsupported(
                    "wasm backend: inlined bridge source guard is outside the owner stream".into(),
                )
            })?;
        let source_args = live_fail_arg_count(
            source_guard.meta_descr.as_ref(),
            source_guard.fail_arg_refs.len(),
        );
        if source_args != bridge.inputargs.len() {
            return Err(BackendError::Unsupported(format!(
                "wasm backend: inlined bridge input arity {} differs from source guard arity {source_args}",
                bridge.inputargs.len(),
            )));
        }
    }

    // `fail_index` stays a per-module ordinal for bridge-cell addressing.
    // The exit names itself by the descr cell in `jf_descr`, so a chained
    // module does not share an index space. `build_function` seeds
    // `guard_idx` with this base; mirror that on `GuardExit.fail_index`.
    for (index, g) in guards.iter_mut().enumerate() {
        g.fail_index += *fail_index_base;
        let from_list = guard_cell_addrs
            .get(index)
            .copied()
            .filter(|&addr| addr != 0);
        let preexisting = g
            .meta_descr
            .as_ref()
            .and_then(|meta| meta.as_fail_descr())
            .map(|fd| fd.adr_jump_offset())
            .filter(|&addr| addr != 0);
        let addr = if let Some(addr) = from_list.or(preexisting.map(|addr| addr as u32)) {
            addr
        } else if *bridge_cells_base != 0 {
            *bridge_cells_base
                + (g.fail_index - *fail_index_base) * std::mem::size_of::<u32>() as u32
        } else {
            0
        };
        // `adr_jump_offset` is stamped by the caller once the module is
        // accepted (`patch_pending_failure_recoveries`). A build that is
        // declined after this point drops the cell array, so a stamp written
        // here would leave the descr naming freed memory for the next compile.
        g.bridge_cell = addr;
    }
    let cell_addrs: Vec<u32> = guards.iter().map(|g| g.bridge_cell).collect();

    // Inter-trace chaining: a loop trace's guard exits dispatch to a compiled
    // bridge in-module via `call_indirect` through the shared
    // `__indirect_function_table` (see the epilogue in `build_function`)
    // instead of returning the guard index to the host and round-tripping
    // through the interpreter. Each guard owns one i32 cell in a contiguous
    // `[u32]` array (indexed by `fail_index`) holding its bridge's table slot,
    // `0` = no bridge yet. The array lives in the shared linear memory so the
    // trace reads it and `compile_bridge` (guest-side) writes it. On native
    // builds the trace is never executed, so `alloc_bridge_cells` returns 0 and
    // the dispatch is omitted entirely — the module stays byte-identical.
    // Label-less traces still want guard cells: the self-recursive
    // CALL_ASSEMBLER case chains a guard exit of a Label-less recursion LOOP
    // into its CA bridge, and a BRIDGE's own guards chain nested sub-bridges the
    // same way (a hot guard inside a chained bridge would otherwise round-trip
    // to the host forever). So any guarded trace wants dispatch cells.
    let bridge_dispatch = *bridge_cells_base != 0 || cell_addrs.iter().any(|&addr| addr != 0);
    // All boundary values use an i64 carrier, including raw Float bits. That
    // makes the call type depend only on arity while preserving f64 payloads.
    let bridge_param_arities: Vec<usize> = if *bridge_param_dispatch && bridge_dispatch {
        let mut arities: Vec<usize> = guards
            .iter()
            .map(|guard| live_fail_arg_count(guard.meta_descr.as_ref(), guard.fail_arg_refs.len()))
            .collect();
        arities.sort_unstable();
        arities.dedup();
        arities
    } else {
        Vec::new()
    };

    // Frame value slots (inputs at entry, fail-arg spills at guard exit) occupy
    // `[1, 1 + max(num inputs, max fail args))`. They precede the dispatch key,
    // Ref homes, and the always-present tail call area; a chained bridge must
    // fit the source token's frozen value-slot count before it can share that
    // frame.
    let label_resume = LabelResumeData::collect_with_regions(
        &analysis_inputargs,
        &analysis_ops,
        &region_spans,
        inputargs.len(),
    );
    let max_value_slots =
        normal_frame_value_slots_for(&analysis_inputargs, &analysis_ops, inputargs.len())
            + label_resume.scalar_slots;
    if max_value_slots > frame.value_slots {
        let shortage = super::FrameShortage::new(
            super::FrameShortageKind::FrameValueSlots,
            max_value_slots,
            frame.value_slots,
        );
        if !inlined_bridges.is_empty() {
            super::record_inline_geometry(shortage.kind, shortage.needed, shortage.available);
        }
        *shortage_out = Some(shortage);
        return Err(BackendError::Unsupported(format!(
            "wasm backend: {} frame value slots exceed frozen frame layout ({})",
            shortage.needed, shortage.available,
        )));
    }

    let label_param_entry = has_label_param_entry(inputargs, ops, *frame, *bridge_entry_arity);
    let entry_param_count = 1 + if label_param_entry {
        crate::FROZEN_LABEL_PARAM_ARITY
    } else {
        bridge_entry_arity.unwrap_or(0)
    } as u32;
    if let Some(arity) = bridge_entry_arity
        && *arity != inputargs.len()
    {
        return Err(BackendError::Unsupported(format!(
            "wasm backend: bridge parameter arity {arity} differs from input arity {}",
            inputargs.len(),
        )));
    }
    let value_types = collect_value_types(
        &analysis_inputargs,
        &analysis_ops,
        num_vars,
        entry_param_count,
    );
    let ref_values = RefValues::collect(&analysis_inputargs, &analysis_ops);
    let ref_homes = RefHomes::collect(
        &analysis_inputargs,
        &analysis_ops,
        ca.emit_ca,
        &label_resume.captured_refs,
        &region_spans,
    );
    let num_ref_homes = ref_homes.len();
    // x86 `store_force_descr` keeps the guard token's already-computed gcmap
    // as `_finish_gcmap`.  Record the same static locations now that RefHomes
    // has assigned them.  Runtime values are deliberately not inspected:
    // constants and GcRefs are values, not tagged frame coordinates.
    let sign = std::mem::size_of::<isize>();
    for (guard, op) in guards.iter_mut().zip(
        analysis_ops
            .iter()
            .filter(|op| op.opcode.is_guard() || op.opcode == OpCode::Finish),
    ) {
        if !matches!(op.opcode, OpCode::GuardNotForced | OpCode::GuardNotForced2) {
            continue;
        }
        guard.publishes_finish_gcmap = op.opcode == OpCode::GuardNotForced2;
        let live = live_fail_arg_extent(guard.meta_descr.as_ref(), guard.fail_arg_refs.len());
        for (&arg, &tp) in guard
            .fail_arg_refs
            .iter()
            .zip(&guard.fail_arg_types)
            .take(live)
        {
            if tp == Type::Ref
                && let Some(home) = ref_homes.home(arg)
            {
                let offset = frame.home_ofs(home as u64) as usize;
                guard.force_ref_home_indices.push((offset / sign) as u32);
            }
        }
    }
    // `generate_quick_failure` stores `guardtok.gcmap` into `jf_gcmap` and
    // the descr into `jf_descr` before the recovery stub returns. The cell
    // address is stable for the loop's `LoopAsmResources`. A FINISH after
    // GUARD_NOT_FORCED_2 publishes that guard's `_finish_gcmap` (plus the
    // result Ref when there is one), matching `genop_finish`.
    let mut pending_finish: Vec<u32> = Vec::new();
    for guard in guards.iter_mut() {
        guard.descr_cell = crate::failguard::alloc_exit_cell(ca.gcmap_sink, guard.fail_index);
        let mut indices = if guard.is_finish {
            pending_finish.clone()
        } else {
            ref_spill_item_indices(guard, sign)
        };
        if guard.is_finish && guard.fail_arg_types.first() == Some(&Type::Ref) {
            if let Some(slot) = physical_fail_slot(guard, 0) {
                let bit = ((FRAME_SLOT_BASE as usize + slot * 8) / sign) as u32;
                if !indices.contains(&bit) {
                    indices.push(bit);
                }
            }
        }
        if guard.publishes_finish_gcmap {
            pending_finish.clone_from(&guard.force_ref_home_indices);
        }
        guard.exit_gcmap_ptr = if indices.is_empty() {
            0
        } else {
            crate::release::park_gcmap_raw(ca.gcmap_sink, gcmap_for_item_indices(&indices))
        };
    }
    let shortage = if num_ref_homes > frame.addressable_ordinary_homes() {
        Some(super::FrameShortage::new(
            super::FrameShortageKind::OrdinaryRefHomes,
            num_ref_homes,
            frame.addressable_ordinary_homes(),
        ))
    } else {
        label_resume.shortage(*frame)
    };
    if let Some(shortage) = shortage {
        if !inlined_bridges.is_empty() {
            super::record_inline_geometry(shortage.kind, shortage.needed, shortage.available);
        }
        *shortage_out = Some(shortage);
        let reason = match shortage.kind {
            super::FrameShortageKind::OrdinaryRefHomes => format!(
                "wasm backend: {} ordinary ref homes exceed frozen frame layout ({})",
                shortage.needed, shortage.available,
            ),
            super::FrameShortageKind::LabelResumeRefSlots => format!(
                "wasm backend: {} LABEL ref captures exceed label resume layout ({} label ref slots)",
                shortage.needed, shortage.available,
            ),
            super::FrameShortageKind::LabelResumeCaptureSlots => format!(
                "wasm backend: {} LABEL capture slots exceed label resume layout ({})",
                shortage.needed, shortage.available,
            ),
            super::FrameShortageKind::FrameValueSlots => {
                unreachable!("value-slot shortage was checked above")
            }
        };
        return Err(BackendError::Unsupported(reason));
    }

    // CA frames execute the source loop and this bridge on the same frozen
    // geometry.  `compile_bridge` rejects a bridge that needs more slots, so
    // no global floor or speculative slack is needed here.

    // This exact lowering census controls the host-trampoline import. Direct
    // residual helpers, including the CA arm's inline fast path, use
    // `call_indirect` and need no import, although their frozen frame still
    // keeps the tail call area for future bridges.
    let needs_call =
        has_trampoline_calls(&analysis_inputargs, &analysis_ops, constants, ca.emit_ca);
    // In-module residual calls: the largest
    // eligible `(i64×n)->i64` arity in this trace — residual CALLs (word
    // result or word-ABI void) plus the `CallMallocNursery*` / write-barrier
    // helper targets, which share the same uniform-i64 ABI — or `None` if there
    // are none. Each distinct arity `0..=max` gets its own function type
    // (declared below) so those arms can `call_indirect` with a static type.
    let residual_max_arity = {
        let scanned = analysis_ops
            .iter()
            .filter_map(|op| direct_helper_i64_arity(op, &ref_values, constants))
            // The `write_real_errno` / `read_real_errno` helpers around a
            // CALL_RELEASE_GIL with a `save_err` are `(i64)->i64`.
            .chain(
                analysis_ops
                    .iter()
                    .filter(|op| {
                        op.opcode.is_call_release_gil()
                            && const_operand_value(constants, op.arg(0).to_opref())
                                .is_some_and(|save_err| save_err != 0)
                    })
                    .map(|_| 1),
            )
            .max();
        // `emit_frame_write_barrier` after each frame reload calls the
        // one-argument `wasm_jit_write_barrier`, which no trace operation names.
        let scanned = if wb.fn_ptr != 0 && analysis_ops.iter().any(collecting_site) {
            Some(scanned.map_or(1, |arity| arity.max(1)))
        } else {
            scanned
        };
        if ca.emit_ca {
            // The CA arm's frame helpers (`wasm_jit_ca_reload_frame()`,
            // `wasm_jit_ca_pop_frame(frame_base)`, and
            // `wasm_jit_ca_push_frame(frame_ptr)`) lower through
            // this same `(i64×n)->i64` family; make sure arity 2 is declared,
            // which declares the full 0..=2 range including reload's arity 0.
            Some(scanned.map_or(2, |m| m.max(2)))
        } else if ca.ca_reload_fn_ptr != 0 {
            // Every trace body can reload its own frame after a collecting
            // direct call, even though only bridges emit the CA arm.
            Some(scanned.map_or(0, |m| m))
        } else {
            scanned
        }
    };
    // `_check_frame_depth` calls `wasm_realloc_frame(items, depth) -> items`,
    // the same `(i64, i64) -> i64` family as residual arity 2.
    // `_check_frame_depth` runs when the bridge must grow the live frame,
    // including a compact bridge whose jump target is deeper (`assemble_bridge`).
    let emit_frame_realloc = ca.realloc_fn_ptr != 0;
    let residual_max_arity = if emit_frame_realloc {
        Some(residual_max_arity.unwrap_or(0).max(2))
    } else {
        residual_max_arity
    };
    // Typed float residual calls use the descr-derived wasm type instead
    // of the uniform i64 helper family. Preserve first-use order so a given
    // trace gets stable type indices while declaring each signature once.
    let mut typed_residual_sigs = Vec::new();
    for op in analysis_ops {
        if let Some(sig) = residual_call_typed_sig(op, constants)
            .or_else(|| conditional_call_typed_sig(op, constants))
            && !typed_residual_sigs.contains(&sig)
        {
            typed_residual_sigs.push(sig);
        }
    }
    if analysis_ops.iter().any(|op| op.opcode == OpCode::FloatMod) {
        let sig = (vec![ValType::F64, ValType::F64], Some(ValType::F64));
        if !typed_residual_sigs.contains(&sig) {
            typed_residual_sigs.push(sig);
        }
    }
    // True-void residual calls use `(i64×n) -> ()`, a separate family from the
    // i64- and f64-result types. As with the uniform i64 family, declaring
    // `0..=max` makes each type index a base plus the call arity.
    let true_void_residual_max_arity = analysis_ops
        .iter()
        .filter_map(|op| {
            residual_call_void_true_arity(op, constants)
                .or_else(|| conditional_call_true_void_arity(op, constants))
        })
        .max();
    // The shared indirect-function table backs direct residual helpers as well
    // as host-trampoline dispatch, chained bridges, and CA recursion.
    let needs_table = needs_call
        || bridge_dispatch
        || residual_max_arity.is_some()
        || !typed_residual_sigs.is_empty()
        || true_void_residual_max_arity.is_some()
        || ca.emit_ca
        || inline_trip.is_some();
    // `ca.emit_ca` forces the direct helper family to include arities 0..=2,
    // so all CA frame-helper trampoline `else` arms below are baseline-only.
    debug_assert!(!ca.emit_ca || residual_max_arity.is_some());

    let mut module = Module::new();

    // Type section
    let mut types = TypeSection::new();
    // Type 0 remains the loop/host entry signature. A bridge parameter entry
    // receives a separate type so its terminal JUMP can still call a loop.
    types.ty().function(vec![ValType::I32], vec![ValType::I32]);
    let mut next_type_idx = 1u32;
    let bridge_entry_type_idx = bridge_entry_arity.map(|arity| {
        let idx = next_type_idx;
        next_type_idx += 1;
        types.ty().function(
            std::iter::once(ValType::I32)
                .chain(std::iter::repeat_n(ValType::I64, arity))
                .collect::<Vec<_>>(),
            vec![ValType::I32],
        );
        idx
    });
    // Declared by the module that *defines* a parameter entry and by one that
    // only *calls* another module's: a loop-closing JUMP naming a published
    // wide slot needs the callee's type to `return_call_indirect` it.
    let label_param_type_idx = (label_param_entry || *external_jump_wide_slot != 0).then(|| {
        let idx = next_type_idx;
        next_type_idx += 1;
        types.ty().function(
            std::iter::once(ValType::I32)
                .chain(std::iter::repeat_n(
                    ValType::I64,
                    crate::FROZEN_LABEL_PARAM_ARITY,
                ))
                .collect::<Vec<_>>(),
            vec![ValType::I32],
        );
        idx
    });
    let mut bridge_param_type_indices = indexmap::IndexMap::new();
    if let (Some(arity), Some(idx)) = (*bridge_entry_arity, bridge_entry_type_idx) {
        bridge_param_type_indices.insert(arity, idx);
    }
    let jit_call_type_idx = if needs_call {
        let idx = next_type_idx;
        next_type_idx += 1;
        types
            .ty()
            .function(vec![ValType::I32, ValType::I32], vec![]);
        Some(idx)
    } else {
        None
    };
    // Residual-call types follow: `(i64×n) -> i64` for arity `n`, indexed by
    // `residual_type_base + n`. `residual_type_base` = the count of types above.
    let residual_type_base = next_type_idx;
    if let Some(max) = residual_max_arity {
        for n in 0..=max {
            types
                .ty()
                .function(vec![ValType::I64; n], vec![ValType::I64]);
        }
        next_type_idx += max as u32 + 1;
    }
    // CA deopt-helper type `(i64 frame_ptr, i64 compiled_ptr) -> i64`. The CA arm
    // `call_indirect`s `wasm_ca_resume_deopt` through it when a self-recursive
    // callee leaves its trace through a guard (a deopt). Declared after the
    // residual-call type family so its index is independent of which residual
    // arities the bridge happens to use.
    let ca_helper_type_idx = next_type_idx;
    if ca.emit_ca {
        types
            .ty()
            .function(vec![ValType::I64, ValType::I64], vec![ValType::I64]);
        next_type_idx += 1;
    }
    // Typed residual types follow all pre-existing direct helper types. Both
    // the parameter sequence and the result come from the call descr (`i64`
    // for Int/Ref, `f64` for Float); the emitter uses this map to select the
    // exact `call_indirect` type for each callee.
    let typed_residual_type_base = next_type_idx;
    let typed_residual_type_indices = typed_residual_sigs
        .iter()
        .cloned()
        .enumerate()
        .map(|(offset, sig)| (sig, typed_residual_type_base + offset as u32))
        .collect::<indexmap::IndexMap<TypedResidualSig, u32>>();
    for (params, result) in typed_residual_type_indices.keys() {
        types
            .ty()
            .function(params.clone(), result.iter().copied().collect::<Vec<_>>());
    }
    next_type_idx += typed_residual_type_indices.len() as u32;
    let true_void_residual_type_base = next_type_idx;
    if let Some(max) = true_void_residual_max_arity {
        for n in 0..=max {
            types.ty().function(vec![ValType::I64; n], vec![]);
        }
        next_type_idx += max as u32 + 1;
    }
    // Deferred-merge trip callback `(i64 pending_slot) -> i64`, declared before
    // the bridge-parameter arities so an armed probe cannot shift their
    // indices.
    let inline_trip_type_idx = next_type_idx;
    if inline_trip.is_some() {
        types.ty().function(vec![ValType::I64], vec![ValType::I64]);
        next_type_idx += 1;
    }
    for arity in bridge_param_arities {
        if bridge_param_type_indices.contains_key(&arity) {
            continue;
        }
        bridge_param_type_indices.insert(arity, next_type_idx);
        next_type_idx += 1;
        types.ty().function(
            std::iter::once(ValType::I32)
                .chain(std::iter::repeat_n(ValType::I64, arity))
                .collect::<Vec<_>>(),
            vec![ValType::I32],
        );
    }
    // Shared guard-exit spill functions, declared after every other family so
    // an added arity cannot shift an index a call site already baked.
    let spill_arities = spill_helper_arities(&guards, *frame);
    let mut spill_helper_type_indices: Vec<u32> = Vec::with_capacity(spill_arities.len());
    for &arity in &spill_arities {
        spill_helper_type_indices.push(next_type_idx);
        next_type_idx += 1;
        types.ty().function(
            std::iter::once(ValType::I32)
                .chain(std::iter::repeat_n(ValType::I64, arity))
                .collect::<Vec<_>>(),
            Vec::new(),
        );
    }
    // After the spill types so neither a residual `call_indirect` nor a spill
    // function's type index moves. The body is `(param i32)`: the items base.
    let frame_wb_sites = if wb.fn_ptr != 0
        && residual_max_arity.is_some()
        && (ca.jf_top_addr.is_some() || ca.ca_reload_fn_ptr != 0 || ca.emit_ca)
    {
        analysis_ops.iter().filter(|op| collecting_site(op)).count()
    } else {
        0
    };
    let frame_wb_type_idx = if outline_frame_write_barrier(frame_wb_sites) {
        let idx = next_type_idx;
        next_type_idx += 1;
        types.ty().function(vec![ValType::I32], vec![]);
        Some(idx)
    } else {
        None
    };
    // After the frame-barrier type, for the same reason: a later family must
    // not shift either helper's type index. The blob is a host box, not a
    // wasm data segment, so declaring the type does not put it in the module.
    let mut inline_fail: Vec<u32> = emitted_bridges
        .iter()
        .map(|bridge| fail_index_base.wrapping_add(bridge.source_fail_index))
        .collect();
    inline_fail.sort_unstable();
    let planned_guard_exits = plan_guard_exit_outline(
        analysis_ops,
        inputargs,
        &ref_homes,
        *frame,
        constants,
        *gc_table_base,
        &gc_table_bases,
        ca.exit_table_base,
        *fail_index_base,
        *invalidated_flag_addr,
        &inline_fail,
    );
    let guard_exit_addrs = if planned_guard_exits.save > GUARD_EXIT_OUTLINE_COST
        && !planned_guard_exits.words.is_empty()
    {
        let base = crate::release::park_guard_exit_blob(
            ca.gcmap_sink,
            planned_guard_exits.words.into_boxed_slice(),
        );
        // A zero address is the inline sentinel (`emit_outlined_guard_exit`).
        // `exit_table_base == 0` never reaches here; this only rejects a box
        // whose low 32 bits are zero.
        if base == 0 {
            None
        } else {
            Some(
                planned_guard_exits
                    .rel
                    .into_iter()
                    .map(|off| {
                        if off == u32::MAX {
                            0
                        } else {
                            base.wrapping_add(off)
                        }
                    })
                    .collect::<Vec<_>>(),
            )
        }
    } else {
        None
    };
    let guard_exit_type_idx = if guard_exit_addrs.is_some() {
        let idx = next_type_idx;
        next_type_idx += 1;
        types
            .ty()
            .function(vec![ValType::I32, ValType::I32, ValType::I64], vec![]);
        Some(idx)
    } else {
        None
    };
    // These types are last in the section. Keep the cursor consumed so a later
    // family cannot reuse an index.
    let _ = next_type_idx;
    module.section(&types);

    // Import section
    let mut imports = ImportSection::new();
    imports.import(
        "env",
        "memory",
        MemoryType {
            minimum: 1,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        },
    );
    if needs_call {
        // Import jit_call trampoline as function index 0
        imports.import(
            "env",
            "jit_call_compact",
            EntityType::Function(jit_call_type_idx.expect("jit_call type")),
        );
    }
    if needs_table {
        // Import the host's shared indirect function table as table index 0.
        // `jit_call`'s residual dispatch and the epilogue bridge
        // `call_indirect` both index it; the host registers every compiled
        // trace (and bridge) into this table by slot. A table import does not
        // shift the function index space, so `trace_func_idx` still depends
        // only on whether `jit_call` (a function import) is present.
        imports.import(
            "env",
            "__indirect_function_table",
            EntityType::Table(TableType {
                element_type: RefType::FUNCREF,
                table64: false,
                minimum: 0,
                maximum: None,
                shared: false,
            }),
        );
    }
    module.section(&imports);

    // Function section
    let mut functions = FunctionSection::new();
    if label_param_entry {
        // The narrow shim keeps type 0 so the host and every type-0 indirect
        // call still enter here; the wide entry follows it.
        functions.function(0);
        functions.function(label_param_type_idx.expect("a parameter entry declares its own type"));
    } else {
        functions.function(bridge_entry_type_idx.unwrap_or(0));
    }
    for &type_idx in &spill_helper_type_indices {
        functions.function(type_idx);
    }
    if let Some(type_idx) = frame_wb_type_idx {
        functions.function(type_idx);
    }
    if let Some(type_idx) = guard_exit_type_idx {
        functions.function(type_idx);
    }
    module.section(&functions);

    // Only armed modules carry this global. The runner reads it after
    // instantiation to give `PYRE_WASM_DUMP_ALL_TRACES` the same trace id the
    // census reports; it is omitted entirely from ordinary trace modules.
    if let Some(census) = trace_entry_census {
        let mut globals = GlobalSection::new();
        globals.global(
            GlobalType {
                val_type: ValType::I64,
                mutable: false,
                shared: false,
            },
            &ConstExpr::i64_const(census.trace_id as i64),
        );
        module.section(&globals);
    }

    // Export section: trace function index depends on whether we imported jit_call
    let trace_func_idx = if needs_call { 1 } else { 0 };
    let mut exports = ExportSection::new();
    exports.export("trace", ExportKind::Func, trace_func_idx);
    if label_param_entry {
        exports.export("trace_wide", ExportKind::Func, trace_func_idx + 1);
    }
    if trace_entry_census.is_some() {
        exports.export("trace_entry_census_id", ExportKind::Global, 0);
    }
    module.section(&exports);

    // Code section
    let mut codes = CodeSection::new();
    let jit_call_idx = if needs_call { Some(0u32) } else { None };
    // The spill functions follow this module's entry function(s) in the
    // function and code sections alike, so their indices start past them.
    let first_spill_func_idx = trace_func_idx + if label_param_entry { 2 } else { 1 };
    let spill_helper_indices: indexmap::IndexMap<usize, u32> = spill_arities
        .iter()
        .enumerate()
        .map(|(i, &arity)| (arity, first_spill_func_idx + i as u32))
        .collect();
    let func = build_function(
        inputargs,
        &analysis_inputargs,
        &analysis_ops,
        emitted_bridges,
        constants,
        num_vars,
        &value_types,
        jit_call_idx,
        *vtable_offset,
        classptr_to_typeid,
        guard_gc_type_info,
        *alloc,
        wb,
        nursery.as_ref(),
        &ref_values,
        &ref_homes,
        &label_resume,
        *bridge_cells_base,
        &cell_addrs,
        bridge_dispatch,
        *bridge_entry_arity,
        &bridge_param_type_indices,
        *invalidated_flag_addr,
        *gc_table_base,
        gc_const_keys,
        &gc_table_bases,
        *fail_index_base,
        *external_jump_slot,
        *external_jump_key,
        label_param_type_idx
            .filter(|_| *external_jump_wide_slot != 0)
            .map(|type_idx| (*external_jump_wide_slot, type_idx)),
        *frame,
        residual_max_arity.map(|_| residual_type_base),
        &typed_residual_type_indices,
        true_void_residual_max_arity.map(|_| true_void_residual_type_base),
        ca.clone(),
        ca_helper_type_idx,
        *trace_entry_census,
        label_param_entry,
        inline_trip.map(|probe| (probe, inline_trip_type_idx)),
        &spill_helper_indices,
        frame_wb_type_idx.map(|_| first_spill_func_idx + spill_arities.len() as u32),
        guard_exit_addrs.as_deref().map(|addrs| GuardExitOutline {
            func: first_spill_func_idx
                + spill_arities.len() as u32
                + u32::from(frame_wb_type_idx.is_some()),
            addrs,
        }),
    )?;
    if label_param_entry {
        codes.function(&build_label_param_shim(*frame, trace_func_idx + 1));
    }
    codes.function(&func);
    for &arity in &spill_arities {
        codes.function(&build_spill_helper(arity));
    }
    if frame_wb_type_idx.is_some() {
        codes.function(&build_frame_wb_helper(FrameWriteBarrier {
            fn_ptr: wb.fn_ptr,
            type_idx: residual_type_base + 1,
            flag_byteofs: wb.flag_byteofs,
            if_flag: wb.if_flag,
            helper: None,
        }));
    }
    if guard_exit_type_idx.is_some() {
        codes.function(&build_guard_exit_helper());
    }
    module.section(&codes);

    let used_labels = label_resume.ref_slots.max(ca.home_gcmap_min_labels);
    Ok((module.finish(), guards, num_ref_homes, used_labels))
}

fn build_label_param_shim(frame: FrameGeometry, wide_func_idx: u32) -> Function {
    let mut func = Function::new(Vec::new());
    let mut raw_sink = func.instructions();
    let mut sink = PeepSink::new(&mut raw_sink);

    sink.local_get(0);
    for k in 0..crate::FROZEN_LABEL_PARAM_ARITY {
        sink.local_get(0);
        // The narrow entry reloads what a JUMP stored with `spill_slot_ofs`.
        sink.i64_load(mem64(frame.spill_slot_ofs(k as u64)));
    }
    sink.return_call(wide_func_idx);
    sink.end();
    sink.flush();
    drop(sink);

    func
}

/// What a wasm function costs the host compiler before its body counts, in
/// units of the body instructions the same cost would buy.
///
/// Collapsing 30.9 KB of `for_iter_list_fold` spill runs into 97 extra
/// functions cut cranelift's compile time for its 44 modules from 130.8 ms to
/// 117.7 ms, where the bytes alone were worth 21.1 ms; the difference puts a
/// function's own fixed cost near 0.06 ms, about forty instructions of body.
/// Charging it here keeps the near-break-even counts out.
const SPILL_HELPER_FIXED_INSTRS: usize = 40;

/// Operators in the inline frame-barrier body (`emit_frame_write_barrier`
/// with no helper), and in the `local.get 0; call` site that replaces it.
const FRAME_WB_BODY_OPS: usize = 17;
const FRAME_WB_CALL_OPS: usize = 2;

/// `_reload_frame_if_necessary` tails into `_write_barrier_fastpath` at every
/// collecting site. The flag test and the helper `call_indirect` do not
/// depend on which call just returned, so one `(param i32)` function can hold
/// them. Admitted only when the copied operators pay for that function; the
/// fixed cost is [`SPILL_HELPER_FIXED_INSTRS`].
fn outline_frame_write_barrier(sites: usize) -> bool {
    sites * (FRAME_WB_BODY_OPS - FRAME_WB_CALL_OPS) > FRAME_WB_BODY_OPS + SPILL_HELPER_FIXED_INSTRS
}

/// Fail-argument counts worth a shared spill function, from the guard exits
/// this module is about to emit.
///
/// A guard exit writes its fail arguments to the positional exit slots, three
/// wasm instructions each (`local.get 0`, the value, `i64.store`). Those runs
/// are the largest single thing a trace module contains — 42% of the
/// instructions across `synth/for_iter_list_fold`'s 44 modules. The host
/// compiles every one of those modules with cranelift before the trace can
/// run, at roughly 0.6 ms per kilobyte handed to it against a per-module fixed
/// cost of about 0.2 ms, so what the module costs to admit is very nearly what
/// it weighs. The spill is purely positional, so one function per argument
/// count serves every exit of that count and the call site costs one
/// instruction per argument instead of three.
///
/// A count is admitted only when the exits that share it pay for the function:
/// `uses * (3n - (n + 2))` saved against `3n + 1` emitted, plus
/// [`SPILL_HELPER_FIXED_INSTRS`] for the function itself. Counts of one
/// argument never do (the call site is the same size as the stores), and a
/// count used once never does. An arity admitted here that no exit reaches —
/// a guard whose region was merged branches instead of spilling — costs its
/// unused body and nothing else, so the estimate may over-admit safely.
fn spill_helper_arities(guards: &[GuardExit], frame: FrameGeometry) -> Vec<usize> {
    let mut uses: HashMap<usize, usize> = HashMap::new();
    for guard in guards {
        *uses
            .entry(live_fail_arg_count(
                guard.meta_descr.as_ref(),
                guard.fail_arg_refs.len(),
            ))
            .or_default() += 1;
    }
    let mut arities: Vec<usize> = uses
        .into_iter()
        .filter(|&(arity, uses)| {
            arity >= 2
                && uses >= 2
                && uses * (2 * arity - 2) > 3 * arity + 1 + SPILL_HELPER_FIXED_INSTRS
        })
        .map(|(arity, _)| arity)
        .collect();
    // `HashMap` iteration order is not stable across runs, and two compilations
    // of the same trace must emit byte-identical modules (`compile_module_cached`
    // keys its host handle on the bytes).
    arities.sort_unstable();
    // A tail slot is not at `FRAME_SLOT_BASE + i*8`. The shared helper only
    // writes that prefix, so an arity that reaches the tail stays inline.
    if frame.has_tail() {
        arities.retain(|&arity| arity <= frame.prefix_value_slots);
    }
    arities
}

/// One shared spill function: `(i32 frame_ptr, i64 x arity) -> ()`, writing its
/// arguments to the positional exit slots `frame[1..=arity]`.
fn build_spill_helper(arity: usize) -> Function {
    let mut func = Function::new(Vec::new());
    let mut raw_sink = func.instructions();
    let mut sink = PeepSink::new(&mut raw_sink);
    for i in 0..arity {
        sink.local_get(0);
        sink.local_get(1 + i as u32);
        sink.i64_store(mem64(FRAME_SLOT_BASE + i as u64 * SLOT_SIZE));
    }
    sink.end();
    sink.flush();
    drop(sink);
    func
}

/// The outlined `_write_barrier_fastpath` for a frame. Parameter 0 is the
/// items base the trace passes from local 0.
fn build_frame_wb_helper(wb: FrameWriteBarrier) -> Function {
    let mut func = Function::new(Vec::new());
    let mut raw_sink = func.instructions();
    let mut sink = PeepSink::new(&mut raw_sink);
    sink.frame_wb = Some(FrameWriteBarrier { helper: None, ..wb });
    emit_frame_write_barrier(&mut sink);
    sink.end();
    sink.flush();
    drop(sink);
    func
}

/// Homes one outlined guard exit can name. Past this the exit stays inline.
const GUARD_EXIT_MAX_HOMES: usize = 48;
const GUARD_EXIT_FLAG_EXC: u32 = 1;
const GUARD_EXIT_FLAG_COUNTER: u32 = 2;
/// Byte offset of the source-home array inside a record. The header is
/// `n, flags, exit_word_addr, counter_ofs` (four u32s).
const GUARD_EXIT_SRC_BYTE: u32 = 16;
const GUARD_EXIT_DEST_BYTE: u32 = GUARD_EXIT_SRC_BYTE + (GUARD_EXIT_MAX_HOMES as u32) * 4;
const GUARD_EXIT_REC_WORDS: usize = 4 + 2 * GUARD_EXIT_MAX_HOMES;
/// Body of [`build_guard_exit_helper`] plus [`SPILL_HELPER_FIXED_INSTRS`].
/// Padded past the counted operators so a handful of exits does not admit
/// a function that fails to pay for itself.
const GUARD_EXIT_OUTLINE_COST: usize = 160;

/// `(param i32 frame, i32 record, i64 counter) -> ()`.
///
/// A failing guard used to reload every fail-arg home, call the positional
/// spill helper, then reload `jf_descr` / `jf_gcmap` from the exit table.
/// Those runs are most of a pickle trace module: dynasm patches
/// `emit_op_guard_not_invalidated` into a NOP (`invalidate_loop`), and wasm
/// has to spill on the cold edge instead. The homes and the two header
/// words differ per exit only in their immediates, so one helper reads them
/// from a host box parked like [`crate::release::LoopAsmResources::exit_table`].
/// The bridge-cell `local.set` and the `br` stay at the site: a callee cannot
/// write the caller's locals or branch to the caller's block.
fn build_guard_exit_helper() -> Function {
    // params: 0 frame, 1 record, 2 counter value. Locals: 3 index, 4 count, 5 value.
    let mut func = Function::new(vec![(2, ValType::I32), (1, ValType::I64)]);
    let mut raw_sink = func.instructions();
    let mut sink = PeepSink::new(&mut raw_sink);

    sink.local_get(1);
    sink.i32_load(mem32(0));
    sink.local_set(4);
    sink.i32_const(0);
    sink.local_set(3);
    sink.block(BlockType::Empty);
    sink.loop_(BlockType::Empty);
    sink.local_get(3);
    sink.local_get(4);
    sink.i32_lt_u();
    sink.i32_eqz();
    sink.br_if(1);
    sink.local_get(1);
    sink.local_get(3);
    sink.i32_const(2);
    sink.i32_shl();
    sink.i32_add();
    sink.i32_load(mem32(GUARD_EXIT_SRC_BYTE as u64));
    sink.local_get(0);
    sink.i32_add();
    sink.i64_load(mem64(0));
    sink.local_set(5);
    sink.local_get(0);
    sink.local_get(1);
    sink.local_get(3);
    sink.i32_const(2);
    sink.i32_shl();
    sink.i32_add();
    sink.i32_load(mem32(GUARD_EXIT_DEST_BYTE as u64));
    sink.i32_add();
    sink.local_get(5);
    sink.i64_store(mem64(0));
    sink.local_get(3);
    sink.i32_const(1);
    sink.i32_add();
    sink.local_set(3);
    sink.br(0);
    sink.end();
    sink.end();

    sink.local_get(1);
    sink.i32_load(mem32(4));
    sink.i32_const(GUARD_EXIT_FLAG_COUNTER as i32);
    sink.i32_and();
    sink.if_(BlockType::Empty);
    sink.local_get(0);
    sink.local_get(1);
    sink.i32_load(mem32(12));
    sink.i32_add();
    sink.local_get(2);
    sink.i64_store(mem64(0));
    sink.end();

    emit_dynamic_exit_header(&mut sink, majit_backend::jitframe::JF_DESCR_OFS as u64, 0);
    emit_dynamic_exit_header(&mut sink, majit_backend::jitframe::JF_GCMAP_OFS as u64, 4);

    sink.local_get(1);
    sink.i32_load(mem32(4));
    sink.i32_const(GUARD_EXIT_FLAG_EXC as i32);
    sink.i32_and();
    sink.if_(BlockType::Empty);
    emit_store_guard_exc_raw(&mut sink);
    sink.end();

    sink.end();
    sink.flush();
    drop(sink);
    func
}

/// `jf_descr` / `jf_gcmap` from the record's exit-word address. Word 0 is
/// the descr cell; word 1 is four bytes later, matching
/// [`emit_load_exit_word`]'s wasm32 stride.
fn emit_dynamic_exit_header(sink: &mut PeepSink<'_, '_>, field: u64, word_byte: u32) {
    emit_header_base(sink);
    sink.local_get(1);
    sink.i32_load(mem32(8));
    if word_byte != 0 {
        sink.i32_const(word_byte as i32);
        sink.i32_add();
    }
    sink.i32_load(mem32(0));
    sink.i32_store(memarg(field, 2));
}

/// Byte offset of a fail arg `emit_resolve_failarg` would load from a Ref
/// home, or `None` when the exit resolves it any other way.
fn home_load_offset(
    opref: OpRef,
    gc_table_slots: &HashMap<u32, (u32, i64)>,
    ref_homes: &RefHomes,
    frame: FrameGeometry,
) -> Option<u64> {
    if opref.is_none() || opref.is_constant() {
        return None;
    }
    // `emit_resolve_failarg` prefers a preamble `LoadFromGcTable` over the home.
    if gc_table_slots.contains_key(&opref.raw()) {
        return None;
    }
    let home = ref_homes.home(opref)?;
    Some(frame.home_ofs(home as u64))
}

fn guard_exit_emits_spill(
    op: &Op,
    guard_idx: u32,
    inline_fail: &[u32],
    invalidated_flag_addr: u32,
) -> bool {
    if inline_fail.binary_search(&guard_idx).is_ok() {
        return false;
    }
    if matches!(op.opcode, OpCode::Finish | OpCode::GuardNotForced2) {
        return false;
    }
    if op.opcode == OpCode::GuardNotInvalidated && invalidated_flag_addr == 0 {
        return false;
    }
    op.opcode.is_guard()
}

/// Smaller of the two inline shapes: the spill-helper call, not the direct
/// stores. Direct stores are larger, so this under-admits them.
fn guard_exit_outline_save(n: usize, exc: bool, counter: bool) -> usize {
    let inline = 2 * n + 14 + usize::from(exc) * 13 + usize::from(counter) * 3;
    let site = 4 + usize::from(counter);
    inline.saturating_sub(site)
}

struct PlannedGuardExits {
    /// Per guard-local index, byte offset of the record in [`Self::words`],
    /// or `u32::MAX` when that exit stays inline.
    rel: Vec<u32>,
    words: Vec<u32>,
    save: usize,
}

fn plan_guard_exit_outline(
    ops: &[Op],
    inputargs: &[InputArgRc],
    ref_homes: &RefHomes,
    frame: FrameGeometry,
    constants: &indexmap::IndexMap<u32, i64>,
    gc_table_base: u32,
    gc_table_bases: &HashMap<u32, u32>,
    exit_table_base: u32,
    fail_index_base: u32,
    invalidated_flag_addr: u32,
    inline_fail: &[u32],
) -> PlannedGuardExits {
    if exit_table_base == 0 {
        return PlannedGuardExits {
            rel: Vec::new(),
            words: Vec::new(),
            save: 0,
        };
    }
    let gc_table_slots = gc_table_failarg_slots(ops, constants, gc_table_base, gc_table_bases);
    let counter_slot = counter_slot(inputargs, ops);
    let mut rel = Vec::new();
    let mut words = Vec::new();
    let mut save = 0usize;
    let mut guard_idx = fail_index_base;
    for op in ops {
        if !(op.opcode.is_guard() || op.opcode == OpCode::Finish) {
            continue;
        }
        let outlined = guard_exit_emits_spill(op, guard_idx, inline_fail, invalidated_flag_addr)
            .then(|| {
                outline_guard_record(
                    op,
                    guard_idx,
                    fail_index_base,
                    exit_table_base,
                    &gc_table_slots,
                    ref_homes,
                    frame,
                    counter_slot,
                )
            })
            .flatten();
        if let Some((record, site_save)) = outlined {
            rel.push((words.len() * 4) as u32);
            words.extend(record);
            save += site_save;
        } else {
            rel.push(u32::MAX);
        }
        guard_idx += 1;
    }
    PlannedGuardExits { rel, words, save }
}

fn outline_guard_record(
    op: &Op,
    guard_idx: u32,
    fail_index_base: u32,
    exit_table_base: u32,
    gc_table_slots: &HashMap<u32, (u32, i64)>,
    ref_homes: &RefHomes,
    frame: FrameGeometry,
    counter_slot: Option<usize>,
) -> Option<(Vec<u32>, usize)> {
    let args = live_exit_fail_args(op);
    if args.len() > GUARD_EXIT_MAX_HOMES {
        return None;
    }
    let mut srcs = Vec::with_capacity(args.len());
    let mut dests = Vec::with_capacity(args.len());
    for (i, arg) in args.iter().enumerate() {
        let src = home_load_offset(*arg, gc_table_slots, ref_homes, frame)?;
        let dest = frame.spill_slot_ofs(i as u64);
        srcs.push(u32::try_from(src).ok()?);
        dests.push(u32::try_from(dest).ok()?);
    }
    let counter = counter_value_spill(op, &exit_fail_args(op)).zip(counter_slot);
    let counter_ofs = match counter {
        Some((_, slot)) => Some(u32::try_from(frame.spill_slot_ofs(slot as u64)).ok()?),
        None => None,
    };
    let exc = matches!(
        op.opcode,
        OpCode::GuardNoException | OpCode::GuardException | OpCode::GuardNotForced
    );
    let mut flags = 0u32;
    if exc {
        flags |= GUARD_EXIT_FLAG_EXC;
    }
    if counter_ofs.is_some() {
        flags |= GUARD_EXIT_FLAG_COUNTER;
    }
    let local = (guard_idx - fail_index_base) as usize;
    // `emit_load_exit_word` addresses pair `local` as two wasm32 words.
    let exit_addr = exit_table_base.wrapping_add((local * 2 * 4) as u32);
    let mut record = vec![0u32; GUARD_EXIT_REC_WORDS];
    record[0] = srcs.len() as u32;
    record[1] = flags;
    record[2] = exit_addr;
    record[3] = counter_ofs.unwrap_or(0);
    let src_base = (GUARD_EXIT_SRC_BYTE / 4) as usize;
    let dest_base = (GUARD_EXIT_DEST_BYTE / 4) as usize;
    for (i, src) in srcs.iter().enumerate() {
        record[src_base + i] = *src;
    }
    for (i, dest) in dests.iter().enumerate() {
        record[dest_base + i] = *dest;
    }
    let site_save = guard_exit_outline_save(srcs.len(), exc, counter_ofs.is_some());
    Some((record, site_save))
}

/// Absolute guest addresses of outlined records, indexed by
/// `guard_idx - fail_index_base`. Zero keeps that exit inline.
#[derive(Clone, Copy)]
struct GuardExitOutline<'a> {
    func: u32,
    addrs: &'a [u32],
}

#[allow(clippy::too_many_arguments)]
fn build_function(
    entry_inputargs: &[InputArgRc],
    inputargs: &[InputArgRc],
    ops: &[Op],
    inlined_bridges: &[InlinedBridge],
    constants: &indexmap::IndexMap<u32, i64>,
    num_vars: u32,
    value_types: &ValueLocals,
    jit_call_idx: Option<u32>,
    vtable_offset: Option<usize>,
    classptr_to_typeid: &HashMap<i64, u32>,
    guard_gc_type_info: &GuardGcTypeInfo,
    alloc: AllocHelpers,
    wb: &WriteBarrierHelpers,
    nursery: Option<&NurseryAllocParams>,
    _ref_values: &RefValues,
    ref_homes: &RefHomes,
    label_resume: &LabelResumeData,
    cells_base: u32,
    cell_addrs: &[u32],
    bridge_dispatch: bool,
    bridge_entry_arity: Option<usize>,
    bridge_param_type_indices: &indexmap::IndexMap<usize, u32>,
    invalidated_flag_addr: u32,
    gc_table_base: u32,
    gc_const_keys: &[usize],
    gc_table_bases: &HashMap<u32, u32>,
    fail_index_base: u32,
    external_jump_slot: u32,
    // Resume-at-LABEL dispatch key the terminal external JUMP writes before
    // tail-calling `external_jump_slot`: `target label ordinal + 1`, so the
    // target's entry `br_table` lands on that label's resume loader. `0` when
    // the target is not peeled (no dispatch reads the slot).
    external_jump_key: u32,
    // `(wide table slot, wasm type index)` of the target's fixed-arity
    // parameter entry, when it published one. The terminal external JUMP then
    // passes its args as parameters instead of through the frame slots the
    // target's narrow shim reads back.
    external_jump_wide: Option<(u32, u32)>,
    frame: FrameGeometry,
    // Base wasm type index of the `(i64×n)->i64` residual-call types (type
    // `residual_type_base + n` for arity `n`), or `None` when the trace has no
    // eligible residual call / `CallMallocNursery*` / write barrier, so those
    // arms always use the `jit_call` path.
    residual_type_base: Option<u32>,
    // Exact wasm type indices for direct typed residual calls, keyed by their
    // descr-derived parameter sequence and result. Float SSA values are
    // converted to/from their i64 bit carrier around the call.
    typed_residual_type_indices: &indexmap::IndexMap<TypedResidualSig, u32>,
    // Base wasm type index of the `(i64×n) -> ()` true-void residual-call
    // types (type `true_void_residual_type_base + n` for arity `n`), or
    // `None` when the trace has no eligible true-void residual call.
    true_void_residual_type_base: Option<u32>,
    // Self-recursive CALL_ASSEMBLER arm (`PYRE_WASM_CA`). `ca.emit_ca` off keeps
    // the body byte-identical.
    ca: CaParams,
    // wasm type index of the CA deopt helper `(i64, i64) -> i64`, declared in the
    // module type section when `ca.emit_ca`. The CA arm uses it to `call_indirect`
    // `ca.deopt_helper_slot` for a deopted callee.
    ca_helper_type_idx: u32,
    trace_entry_census: Option<crate::TraceEntryCensusStorage>,
    label_param_entry: bool,
    inline_trip: Option<(InlineTripProbe, u32)>,
    spill_helper_indices: &indexmap::IndexMap<usize, u32>,
    frame_wb_func_idx: Option<u32>,
    guard_exit_outline: Option<GuardExitOutline<'_>>,
) -> Result<Function, BackendError> {
    // The CA arm requires residual types (the setup above forces arity >= 2
    // whenever it is emitted). Its `jit_call` fallback branches are retained
    // solely for a trace that declared no residual type family at all.
    debug_assert!(!ca.emit_ca || residual_type_base.is_some());
    let value_locals_end = value_types.end_local();
    // Resume-at-LABEL shape, needed here because `resume_dispatch` costs a
    // local. A peeled loop wraps its preamble in a dispatch so a loop-closing
    // bridge can re-enter AT any LABEL up to the header; labels after the
    // header sit inside the `loop` and get no resume arm
    // (`resumable_label_count`).
    let key_dispatch = is_resumable_peeled(ops);
    let num_labels = if key_dispatch {
        resumable_label_count(ops)
    } else {
        0
    };
    let bridge_op_count = inlined_bridges
        .iter()
        .map(|bridge| bridge.ops.len())
        .sum::<usize>();
    let bridge_start = ops.len().checked_sub(bridge_op_count).ok_or_else(|| {
        BackendError::Unsupported(
            "wasm backend: inlined bridge operations are not contained in the merged stream".into(),
        )
    })?;
    // `InlinedBridge::outside_loop` names the placement. The outside ones are
    // emitted past the header `loop`'s `end`, so their ops are the tail of the
    // merged stream and every inside one has to precede them.
    let body_region_count = inlined_bridges
        .iter()
        .position(|bridge| bridge.outside_loop)
        .unwrap_or(inlined_bridges.len());
    let outside_region_count = inlined_bridges.len() - body_region_count;
    if inlined_bridges[body_region_count..]
        .iter()
        .any(|bridge| !bridge.outside_loop)
    {
        return Err(BackendError::Unsupported(
            "wasm backend: an inside-loop inline region follows an outside-loop one".into(),
        ));
    }
    if inlined_bridges.iter().any(|bridge| {
        !bridge.outside_loop && source_guard_precedes_loop_label(ops, bridge.source_fail_index)
    }) {
        return Err(BackendError::Unsupported(
            "wasm backend: a preamble-sourced inline region is placed inside the loop".into(),
        ));
    }
    // A region's block closes where its own ops begin, so a guard branching
    // into it must sit before them. That one inequality is what makes a
    // region's ordinal usable as a branch depth (`emit_guard_exit` counts it
    // from the family's still-open blocks), and it is also what refuses the two
    // structurally impossible attachments: a region reached from a guard in a
    // LATER region of the same family, and a body region reached from a guard
    // in an outside one, whose header `loop` has already closed.
    {
        let mut start = bridge_start;
        for bridge in inlined_bridges {
            let guard_pos = exit_op_index(ops, bridge.source_fail_index).ok_or_else(|| {
                BackendError::Unsupported(
                    "wasm backend: inlined bridge source guard is outside the merged stream".into(),
                )
            })?;
            if guard_pos >= start {
                return Err(BackendError::Unsupported(
                    "wasm backend: an inline region is reached from a guard at or past its \
                     own ops, so the block it branches to has already closed"
                        .into(),
                ));
            }
            start += bridge.ops.len();
        }
    }
    let outside_start = bridge_start
        + inlined_bridges[..body_region_count]
            .iter()
            .map(|bridge| bridge.ops.len())
            .sum::<usize>();
    if outside_region_count > 0 && !key_dispatch {
        return Err(BackendError::Unsupported(
            "wasm backend: an outside-loop inline region needs an entry dispatch".into(),
        ));
    }
    // An outside-loop region re-enters PAST its target LABEL's resume loader,
    // so it takes that label's captures from the frame slots the fall-through
    // path writes as it crosses the label — and a fresh entry clears them. A
    // region reached from the loop body has crossed every resumable label by
    // then; one reached from the preamble has only crossed the labels ahead of
    // its own guard.
    {
        let mut start = outside_start;
        for bridge in &inlined_bridges[body_region_count..] {
            if bridge.external_jump.is_some() {
                // A region that leaves by a cross-module tail call re-enters
                // no LABEL of this function, so it crosses none of them.
                start += bridge.ops.len();
                continue;
            }
            if !outside_region_labels_initialized(
                ops,
                bridge.source_fail_index,
                &ops[start..start + bridge.ops.len()],
            ) {
                return Err(BackendError::Unsupported(
                    "wasm backend: an outside-loop inline region closes at a LABEL its \
                         own entry path had not crossed"
                        .into(),
                ));
            }
            start += bridge.ops.len();
        }
    }
    // A region whose closing JUMP names a resumable LABEL other than the header
    // cannot `br` to the `loop`. Wrap the dispatch in a `loop` such a region
    // re-enters through instead, and give the entry `br_table` a second bucket
    // per label: key `num_labels + 1 + j` lands PAST label j's resume loader,
    // so the region hands its values over in locals rather than through the
    // frame slots the loader reads. A preamble-sourced region always leaves by
    // that route: it is emitted past the `end` of the header `loop`, so no `br`
    // reaches it.
    let resume_dispatch = key_dispatch
        && (outside_region_count > 0
            || ops[bridge_start..].iter().any(|op| {
                op.opcode == OpCode::Jump && jump_resume_ordinal(ops, op, num_labels).is_some()
            }));

    // Value locals occupy the dense local range beginning at 1; reserve
    // `UMULHI_SCRATCH` i64 locals past them for the `UintMulHigh`
    // 32-bit-split expansion, plus one i64 local for the pending overflow flag.
    // One i32 local past those holds a bridge table slot while a guard arm
    // performs its direct indirect tail call (or while the frame-entry
    // dispatcher is enabled without parameter entries).
    let ovf_flag_local = value_locals_end + UMULHI_SCRATCH;
    let bridge_slot_local = ovf_flag_local + 1;
    // The CALL_ASSEMBLER arm needs three more i32 locals: the current callee
    // frame, its returned fail index, and the immutable runtime-target snapshot
    // loaded from the stable dispatch cell.  Keeping the snapshot address in a
    // local makes function/geometry/GC-map selection coherent across redirect.
    let ca_cfp_local = bridge_slot_local + 1;
    let ca_fi_local = ca_cfp_local + 1;
    let ca_target_local = ca_fi_local + 1;
    // Extra i32 scratches when the inline nursery-bump fast path is armed:
    // one holds the loaded `nursery_free` across the bump/commit sequence;
    // runtime varsize array allocation also needs one for the computed
    // total/new-free word.
    let base_i32_locals: u32 = 1 + if ca.emit_ca { 3 } else { 0 };
    // CALL_ASSEMBLER's push/pop footer uses the two alloc scratches even when
    // the nursery fast path is off. Those locals used to alias the prologue
    // gcmap pair; per-site maps do not reserve that pair.
    let extra_alloc_i32 = u32::from(nursery.is_some() || ca.inline.is_some() || ca.emit_ca) * 2;
    let alloc_scratch_local = bridge_slot_local + base_i32_locals;
    let alloc_size_local = alloc_scratch_local + 1;
    // A keyed census must preserve the raw dispatch value until `br_table`.
    // Its counter-address scratch cannot share `bridge_slot_local`, because
    // the latter would replace the selector with a guest-memory address.
    let trace_entry_key_local = bridge_slot_local + base_i32_locals + extra_alloc_i32;
    let trace_entry_needs_key_local = trace_entry_census.is_some() && is_resumable_peeled(ops);
    // `resume_dispatch` keeps the entry key in a local so a region can rewrite
    // it and branch back into the dispatch; without it the key is consumed
    // straight off the frame load.
    let resume_key_local = trace_entry_key_local + u32::from(trace_entry_needs_key_local);
    // One i32: jitframe base (`local 0 - FIRST_ITEM_OFFSET`) so `jf_gcmap`
    // is `i32.store`/`i64.store` at a constant offset. Header-less traces
    // pass an items pointer of 0 and must not allocate or write it.
    let gcmap_frame_local = ca
        .compute_home_gcmap
        .then_some(resume_key_local + u32::from(resume_dispatch));
    // One i32 for the `_check_frame_depth` result. That pointer must not
    // reuse `bridge_slot_local`; the epilogue loads a table index through
    // it, and a frame address is out of range.
    let realloc_result_local =
        resume_key_local + u32::from(resume_dispatch) + u32::from(gcmap_frame_local.is_some());
    debug_assert_eq!(bridge_slot_local, ovf_flag_local + 1);
    debug_assert_eq!(ca_cfp_local, bridge_slot_local + 1);
    debug_assert_eq!(ca_fi_local, ca_cfp_local + 1);
    debug_assert_eq!(ca_target_local, ca_fi_local + 1);
    debug_assert_eq!(alloc_scratch_local, bridge_slot_local + base_i32_locals);
    debug_assert_eq!(alloc_size_local, alloc_scratch_local + 1);
    // rewrite.py clears `gcrefs_recently_loaded` at LABEL. A failarg that
    // is a preamble `LoadFromGcTable` (or SameAs of one) is not a LABEL
    // arg, so the guard must rematerialize the load — cranelift
    // `resolve_failarg_opref` / `GC_TABLE_VAR_INDEX`.
    let gc_table_slots = gc_table_failarg_slots(ops, constants, gc_table_base, gc_table_bases);
    let mut const_tables = ConstPtrTables {
        entries: Vec::new(),
    };
    const_tables.push(gc_table_base, gc_const_keys);
    for bridge in inlined_bridges {
        const_tables.push(bridge.gc_table_base, &bridge.gc_const_keys);
    }
    let inline_guards: Vec<InlineGuard<'_>> = inlined_bridges
        .iter()
        .enumerate()
        .map(|(region, bridge)| InlineGuard {
            guard_idx: fail_index_base + bridge.source_fail_index,
            inputargs: &bridge.inputargs,
            // Blocks open in reverse attach order, making region 0 of each
            // family innermost. The guard `if` contributes the final +1 in
            // `emit_guard_if_exit`.
            region_ordinal: if region < body_region_count {
                region as u32
            } else {
                (region - body_region_count) as u32
            },
            outside_loop: region >= body_region_count,
        })
        .collect();
    let guard_dispatch = BridgeDispatch {
        cells_base,
        cell_addrs,
        fail_index_base,
        bridge_slot_local,
        enabled: bridge_dispatch,
        param_type_indices: bridge_param_type_indices,
        inline_guards: &inline_guards,
        outside_region_base: 0,
        closed_body_regions: 0,
        closed_outside_regions: 0,
        ref_homes,
        frame,
        counter_slot: counter_slot(entry_inputargs, ops).map(|slot| slot as u64),
        spill_helpers: spill_helper_indices,
        exit_table_base: ca.exit_table_base,
        gc_table_slots: &gc_table_slots,
        const_tables: &const_tables,
        const_table_base: gc_table_base,
        attached: ca.attached,
        guard_exit: guard_exit_outline,
    };
    let mut locals = Vec::new();
    let mut start = 0;
    while start < value_types.types().len() {
        let ty = value_types.types()[start];
        let mut end = start + 1;
        while end < value_types.types().len() && value_types.types()[end] == ty {
            end += 1;
        }
        locals.push(((end - start) as u32, ty));
        start = end;
    }
    if let Some((count, ValType::I64)) = locals.last_mut() {
        *count += UMULHI_SCRATCH + 1;
    } else {
        locals.push((UMULHI_SCRATCH + 1, ValType::I64));
    }
    locals.push((
        base_i32_locals
            + extra_alloc_i32
            + u32::from(trace_entry_needs_key_local)
            + u32::from(resume_dispatch)
            + u32::from(gcmap_frame_local.is_some())
            // `_check_frame_depth` result; see `realloc_result_local`.
            + 1,
        ValType::I32,
    ));
    let mut func = Function::new(locals);
    let mut raw_sink = func.instructions();
    let mut sink = PeepSink::new(&mut raw_sink);
    if let Some(frame_local) = gcmap_frame_local {
        sink.gcmap_frame_local = frame_local;
    }
    if wb.fn_ptr != 0
        && ops.iter().any(collecting_site)
        && let Some(base) = residual_type_base
    {
        sink.frame_wb = Some(FrameWriteBarrier {
            fn_ptr: wb.fn_ptr,
            type_idx: base + 1,
            flag_byteofs: wb.flag_byteofs,
            if_flag: wb.if_flag,
            helper: frame_wb_func_idx,
        });
    }

    // assembler.py `_check_frame_depth` at bridge / entry-bridge entry.
    // The depth is a constant of this module; the running length is the
    // JitFrame `jf_frame` length word immediately before local 0.
    if ca.realloc_fn_ptr != 0
        && let Some(base) = residual_type_base
    {
        let gcmap_ptr = realloc_entry_gcmap(frame, entry_inputargs, ca.gcmap_sink);
        let depth_items = if ca.frame_depth_items != 0 {
            ca.frame_depth_items
        } else {
            frame.signed_item_count()
        };
        emit_check_frame_depth(
            &mut sink,
            depth_items,
            ca.realloc_fn_ptr,
            base,
            gcmap_ptr,
            realloc_result_local,
            ca.ca_reload_fn_ptr,
            ca.jf_top_addr,
            ca.attached.propagate_exception_descr,
        );
    }

    // Bind the folded constants the optimizer left under a plain op position
    // (see `unbound_pool_const_seeds`). Emitted before every block so the
    // binding dominates the whole body, including a resume-at-LABEL entry.
    for (raw, bits) in unbound_pool_const_seeds(inputargs, ops, constants, num_vars)? {
        sink.i64_const(bits);
        if value_types.ty(raw) == ValType::F64 {
            sink.f64_reinterpret_i64();
        }
        sink.local_set(value_types.local(raw));
    }

    // A peeled loop arrives as `[preamble..][LABEL][body..][JUMP]`: the
    // preamble runs once on entry, the LABEL is the loop-back target, and
    // JUMP branches back to it. Emit the `loop` at the LABEL selected by the
    // terminal JUMP's descr (not merely the last LABEL) so multi-label traces
    // re-execute the complete loop body.
    let loop_label_idx = find_loop_label_index(ops);
    let has_loop = loop_label_idx.is_some();

    // Def / last-use positions for the post-collection Ref reload filter. The
    // spans must match the ones `RefHomes` was built from, or a home would be
    // reserved and never reloaded (or the reverse).
    let region_spans = InlinedRegionSpan::collect(ops.len(), inlined_bridges);
    let liveness = HomeLiveness::collect_with_regions(inputargs, ops, &region_spans);
    // `get_gcmap` at each collecting op. The pointer is a module constant;
    // `push_gcmap` / `pop_gcmap` bracket the call.
    // `compute_home_gcmap` is the production switch (`compile_loop`): only
    // then is `local 0 - FIRST_ITEM_OFFSET` a real `jf_gcmap`.
    let store_at = HomeStoreAt::build(inputargs, ops, &region_spans);
    let (site_gcmap, site_homes) = build_site_gcmaps(
        frame,
        ref_homes,
        &liveness,
        &store_at,
        ops,
        &region_spans,
        ca.gcmap_sink,
        ca.compute_home_gcmap,
    );
    sink.sync_gcmap_frame();

    // Resume-at-LABEL: a peeled loop wraps its preamble in a dispatch so a
    // loop-closing bridge can re-enter AT any LABEL — key = label ordinal + 1
    // — skipping the code before it, in-module instead of round-tripping
    // through the host. Keyed on the peeled shape (single- OR multi-label);
    // every other trace (non-peeled loop, straight-line, bridge) keeps its
    // byte-identical layout. Each label gets a (past_loader, loader) block
    // pair; the entry `br_table` jumps to the keyed label's resume loader,
    // and the fall-through path `br`s over each loader. Key 0 (and any
    // out-of-range key) runs the function from its entry (the preamble).
    // Every count below is over the resumable prefix — labels 0..=header.
    let all_label_args: Vec<Vec<OpRef>> = ops
        .iter()
        .filter(|op| op.opcode == OpCode::Label)
        .take(num_labels)
        .map(|op| op.getarglist().iter().map(|a| a.to_opref()).collect())
        .collect();

    // The enclosing exit block gives each guard and Finish a direct path to
    // the bridge-dispatch epilogue after it has spilled its fail arguments.
    sink.block(BlockType::Empty); // A $hot_exit
    if resume_dispatch {
        // The key must survive into the `loop` a region branches back to, so
        // read it once here, outside that loop. The census stays outside too:
        // it counts entries into the module, and an in-module re-dispatch is
        // not one.
        sink.local_get(0);
        sink.i64_load(mem64(frame.dispatch_key_ofs));
        sink.i32_wrap_i64();
        sink.local_set(resume_key_local);
        if let Some(census) = trace_entry_census {
            emit_trace_entry_census(&mut sink, census, bridge_slot_local, Some(resume_key_local));
        }
        sink.loop_(BlockType::Empty); // R $resume
        // One block per preamble-sourced region, outside the (B_j, C_j) label
        // pairs and outside the header `loop`, so a guard anywhere in the
        // function can `br` to it. Region 0 is innermost, matching the order
        // the `end`s below close them in.
        for _ in 0..outside_region_count {
            sink.block(BlockType::Empty); // P_k
        }
    }
    if key_dispatch {
        // Per resumable label j (opened outermost = the loop header):
        //   block $past_loader_j (B_j) — the fall-through path br's over the
        //     label-j resume loader.
        //   block $loader_j (C_j) — the `br_table` lands here (its end) for
        //     key j+1: the label-j resume loader.
        // block $dispatch (D) — key 0 br's here: run from the entry.
        for _ in 0..num_labels {
            sink.block(BlockType::Empty); // B_j (j descending)
            sink.block(BlockType::Empty); // C_j
        }
        sink.block(BlockType::Empty); // D $dispatch
        if resume_dispatch {
            sink.local_get(resume_key_local);
        } else {
            sink.local_get(0);
            sink.i64_load(mem64(frame.dispatch_key_ofs));
            sink.i32_wrap_i64();
            // Without a census the key is already where `br_table` wants it.
            // Only the census needs it a second time, so only the census pays
            // to keep a copy: a `tee`/`get` pair here costs every entry into a
            // peeled module.
            if let Some(census) = trace_entry_census {
                let dispatch_key_local = if trace_entry_needs_key_local {
                    trace_entry_key_local
                } else {
                    bridge_slot_local
                };
                sink.local_tee(dispatch_key_local);
                emit_trace_entry_census(
                    &mut sink,
                    census,
                    bridge_slot_local,
                    Some(dispatch_key_local),
                );
                sink.local_get(dispatch_key_local);
            }
        }
        // Depths at this point, innermost first: D=0, then (C_j, B_j) pairs
        // with C_j at 2j+1 and B_j at 2j+2. Entry j+1 of the table targets
        // C_j — label j's resume loader; entry 0 and the default target D (the
        // entry path). Under `resume_dispatch` a second bucket per label,
        // `num_labels + 1 + j`, targets B_j: past that loader, for a region
        // that has already put the label args in their locals.
        let br_targets: Vec<u32> = std::iter::once(0)
            .chain((0..num_labels as u32).map(|j| 2 * j + 1))
            .chain(
                resume_dispatch
                    .then(|| (0..num_labels as u32).map(|j| 2 * j + 2))
                    .into_iter()
                    .flatten(),
            )
            .collect();
        sink.br_table(br_targets, 0);
        sink.end(); // end D $dispatch — key-0 entry path continues here
    } else if let Some(census) = trace_entry_census {
        emit_trace_entry_census(&mut sink, census, bridge_slot_local, None);
    }

    // `jf_gcmap` stays null until a safepoint (`push_gcmap`). A slot is
    // marked only after a store dominates that safepoint (`get_gcmap`).

    // Load inputs from frame into locals, and store Ref inputs to their homes.
    // The input value lives at the frame slot its producer wrote it to: the
    // caller fills slot `k` for the k-th input — `execute_token` for a loop
    // entry, `emit_guard_spill`'s positional fail-arg spill for a bridge entry —
    // so read from the POSITIONAL slot `k`, not `ia.index` (a value number that
    // equals `k` for a loop but not for a bridge, whose live-in args carry their
    // trace value numbers). `ValueLocals` maps each body value id to its dense
    // local index. For `key_dispatch` this runs on
    // the key-0 (preamble) path only — past the `br_if` above — so a resuming
    // bridge never scatters its frame-passed label values into the function
    // inputargs' home slots; those stay null-initialized (GC-safe) and the
    // resume loader sets the live label-arg homes.
    for (k, ia) in entry_inputargs.iter().enumerate() {
        let local_idx = value_types.local(ia.index);
        if bridge_entry_arity.is_some() || label_param_entry {
            // Parameter entries carry raw i64 words after frame_ptr. Float
            // values use their IEEE bit pattern, matching the guard boundary.
            sink.local_get(k as u32 + 1);
            if value_types.ty(ia.index) == ValType::F64 {
                sink.f64_reinterpret_i64();
            }
        } else {
            let offset = frame.spill_slot_ofs(k as u64);
            sink.local_get(0);
            // Value slots are 8 bytes apart (`SLOT_SIZE`), but a Ref store is
            // narrowed to the pointer width. A full `i64.load` then keeps
            // whatever the previous occupant left in the high half, and
            // `PtrEq` against a zero-extended `topframeref` fails on every
            // call-assembler entry. Load the low word the way `GcLoadR` does.
            if ia.tp.get() == Type::Ref {
                sink.i64_load32_u(memarg(offset, 2));
            } else {
                sink.i64_load(mem64(offset));
            }
            if value_types.ty(ia.index) == ValType::F64 {
                sink.f64_reinterpret_i64();
            }
        }
        sink.local_set(local_idx);
        if let Some(h) = ref_homes.home_id(ia.index) {
            sink.local_get(0);
            sink.local_get(local_idx);
            sink.i64_store(mem64(frame.home_ofs(h as u64)));
        }
    }
    // Past the entry loader, so the count is one per entry on the same path
    // the inputs are loaded on.
    if let Some((probe, type_idx)) = inline_trip {
        emit_inline_trip_probe(&mut sink, probe, type_idx);
    }

    // Seed with the fail-index base so each guard/finish exit writes
    // `base + local` into `frame[0]` (every trace passes the next free index
    // of the global fail-index space, `failguard::fail_descr_base`). The local
    // `guard_idx` counter and `collect_guards_and_vars`'s `fail_index` counter
    // increment in lockstep over the same ops, so the value written matches the
    // returned `GuardExit.fail_index` (also offset by the base).
    let mut guard_idx = fail_index_base;
    let mut in_loop_body = false;
    let mut labels_passed = 0usize;
    let mut ovf_flag_live = false;
    let mut fused_guard_at: Option<usize> = None;
    let mut fused_condcall_at: Option<usize> = None;
    let mut skip_nursery_tid_store_at: Option<usize> = None;
    // `_finish_gcmap` is retained only for GUARD_NOT_FORCED_2
    // (`store_force_descr` / `genop_finish`). A leftover `jf_force_descr`
    // from the GUARD_NOT_FORCED that follows CALL_ASSEMBLER is not that map.
    // The CA pop footer loads the callee's flag from the snapshot; this
    // module's ops do not describe the frame being popped.

    // A merged region whose closing JUMP names a LABEL published by another
    // module leaves this function the way its out-of-line bridge did — by
    // tail-calling that module — because `br` cannot cross one. Its ops are in
    // this stream, so the transfer has to be selected per operation rather
    // than per function.
    let external_jump_by_op: Vec<Option<&ExternalJump>> =
        if inlined_bridges.iter().any(|b| b.external_jump.is_some()) {
            let mut by_op = vec![None; ops.len()];
            let mut start = bridge_start;
            for bridge in inlined_bridges {
                for slot in &mut by_op[start..start + bridge.ops.len()] {
                    *slot = bridge.external_jump.as_ref();
                }
                start += bridge.ops.len();
            }
            by_op
        } else {
            Vec::new()
        };

    let mut table_base_by_op = vec![gc_table_base; ops.len()];
    if !inlined_bridges.is_empty() {
        let mut start = bridge_start;
        for bridge in inlined_bridges {
            for slot in &mut table_base_by_op[start..start + bridge.ops.len()] {
                *slot = bridge.gc_table_base;
            }
            start += bridge.ops.len();
        }
    }

    for (op_idx, op) in ops.iter().enumerate() {
        if skip_nursery_tid_store_at == Some(op_idx) {
            skip_nursery_tid_store_at = None;
            continue;
        }
        if op.opcode == OpCode::Label && key_dispatch && labels_passed < num_labels {
            // End of the segment before label j (key-0 / earlier-label path).
            // Branch over the resume loader, then close C_j, emit the loader
            // (resume path only), and close B_j. From inside C_j, `br 1`
            // targets B_j's end, skipping the loader.
            // Preserve every non-argument live-in while its pre-LABEL local is
            // still available. Scalar bits use frozen value slots; Refs use
            // the high, GC-rooted capture region so a chained bridge cannot
            // overwrite them with its own low home mapping.
            for &r in &label_resume.per_label[labels_passed] {
                let storage = label_resume
                    .storage(r)
                    .expect("LABEL live-in has assigned capture storage");
                sink.local_get(0);
                emit_resolve(&mut sink, constants, value_types, r);
                sink.i64_store(mem64(label_resume.frame_offset(storage, frame)));
            }
            sink.br(1); // segment done -> past_loader_j, over the resume loader
            sink.end(); // end C_j (the br_table lands here for key j+1)
            // Resume loader: a loop-closing bridge wrote each label arg into
            // frame slot i (positionally, matching the in-loop JUMP move);
            // load them into the label-arg locals and refresh their Ref
            // homes, mirroring the JUMP's ref-home refresh below. The
            // fall-through path skipped this via the `br 1` above.
            for (i, la) in all_label_args[labels_passed].iter().enumerate() {
                if label_param_entry {
                    sink.local_get(i as u32 + 1);
                } else {
                    sink.local_get(0);
                    sink.i64_load(mem64(frame.spill_slot_ofs(i as u64)));
                }
                if value_types.ty(la.raw()) == ValType::F64 {
                    sink.f64_reinterpret_i64();
                }
                sink.local_set(value_types.local(la.raw()));
                if let Some(h) = ref_homes.home(*la) {
                    sink.local_get(0);
                    sink.local_get(value_types.local(la.raw()));
                    sink.i64_store(mem64(frame.home_ofs(h as u64)));
                }
            }
            // Restore backend-only live-ins after the semantic LABEL args.
            emit_label_capture_restore(
                &mut sink,
                label_resume,
                value_types,
                ref_homes,
                frame,
                labels_passed,
            );
            sink.end(); // end B_j $past_loader
            labels_passed += 1;
        }
        if Some(op_idx) == loop_label_idx {
            sink.loop_(BlockType::Empty);
            for _ in 0..body_region_count {
                sink.block(BlockType::Empty);
            }
            in_loop_body = true;
        }
        // The loop's normal body ends with its JUMP, which branches around all
        // regions. Closing one block before each attached region makes its
        // body reachable only from the guard that branched to that block.
        // Preamble-sourced regions follow the body-sourced ones, and the header
        // `loop` closes before the first of them: their blocks were opened
        // outside it, so their bodies cannot be emitted inside it.
        let in_outside_region = outside_region_count > 0 && op_idx >= outside_start;
        let mut started_body_regions = 0usize;
        let mut started_outside_regions = 0usize;
        if has_loop && op_idx >= bridge_start {
            let mut start = bridge_start;
            for (region, bridge) in inlined_bridges.iter().enumerate() {
                let outside_placed = region >= body_region_count;
                if op_idx == start {
                    if outside_placed && in_loop_body {
                        // A well-formed body ends in a branch, so this return
                        // is unreachable; emit it anyway so a malformed one
                        // cannot walk out of the loop into a region body.
                        sink.local_get(0);
                        sink.return_();
                        sink.end(); // end loop
                        in_loop_body = false;
                    }
                    sink.end();
                }
                if op_idx >= start {
                    if outside_placed {
                        started_outside_regions += 1;
                    } else {
                        started_body_regions += 1;
                    }
                }
                start += bridge.ops.len();
            }
        }
        // Compute the remaining nesting directly from this operation's
        // position, so a label-less stream cannot close a block that was never
        // opened. The body blocks exist only inside the `loop`; the preamble
        // ones are open from the resume `loop` to the end of the function.
        let open_region_blocks = |opened: usize, total: usize| -> Result<u32, BackendError> {
            let remaining = total.checked_sub(opened).ok_or_else(|| {
                BackendError::Unsupported(
                    "wasm backend: inlined bridge region bookkeeping exceeded its open blocks"
                        .into(),
                )
            })?;
            u32::try_from(remaining).map_err(|_| {
                BackendError::Unsupported(
                    "wasm backend: too many inlined bridge regions for wasm branch depth".into(),
                )
            })
        };
        let open_bridge_blocks = if in_loop_body {
            open_region_blocks(started_body_regions, body_region_count)?
        } else {
            0
        };
        let open_outside_blocks =
            open_region_blocks(started_outside_regions, outside_region_count)?;
        // Depth (from statement level) of the enclosing `block` that guard
        // exits `br` to, counted outward from this operation: the
        // (B_j, C_j) pairs a preamble segment still sits inside, the header
        // `loop`, the region blocks open here, the resume `loop`, and block A.
        // Without `key_dispatch` the preamble is at 0, and a straight-line
        // trace uses the universal hot exit block at 0.
        let block_exit_depth = if !has_loop {
            if open_bridge_blocks + open_outside_blocks != 0 {
                return Err(BackendError::Unsupported(
                    "wasm backend: inlined bridge regions require a local loop LABEL".into(),
                ));
            }
            0u32
        } else {
            let label_blocks = if key_dispatch && !in_loop_body {
                2 * (num_labels - labels_passed) as u32
            } else {
                0
            };
            label_blocks
                + u32::from(in_loop_body)
                + open_bridge_blocks
                + open_outside_blocks
                + u32::from(resume_dispatch)
        };
        // A region's block, from this operation's statement level. Region 0 is
        // the innermost of each family, so the ordinal adds to the base.
        let outside_region_base =
            block_exit_depth - open_outside_blocks - u32::from(resume_dispatch);
        let guard_dispatch = BridgeDispatch {
            outside_region_base,
            closed_body_regions: started_body_regions as u32,
            closed_outside_regions: started_outside_regions as u32,
            const_table_base: table_base_by_op[op_idx],
            ..guard_dispatch
        };
        // The guard whose condition the previous op already pushed and tested.
        // `block_exit_depth` is unchanged across the pair: only a LABEL moves
        // `labels_passed` or opens the `loop`, and a guard is neither.
        if fused_guard_at == Some(op_idx) {
            fused_guard_at = None;
            continue;
        }
        if let Some(kind) = cond_kind_of(op.opcode) {
            match next_op_can_accept_cc(
                ops,
                op_idx,
                op.pos().get(),
                &liveness,
                label_resume,
                ref_homes,
            ) {
                Some(next)
                    if matches!(
                        next.opcode,
                        OpCode::CondCallN | OpCode::CondCallValueI | OpCode::CondCallValueR
                    ) =>
                {
                    // Leave the comparison's i32 on the stack. CondCallN
                    // calls when the predicate is nonzero; CondCallValue
                    // calls when it is zero. Both arms consume this i32
                    // instead of re-resolving and `i64.eqz`.
                    push_cond(&mut sink, constants, value_types, op, kind);
                    fused_condcall_at = Some(op_idx + 1);
                }
                Some(guard) => {
                    push_guard_failure_cond(
                        &mut sink,
                        constants,
                        value_types,
                        op,
                        kind,
                        guard.opcode,
                    );
                    emit_guard_if_exit(
                        &mut sink,
                        constants,
                        value_types,
                        guard_idx,
                        guard,
                        block_exit_depth,
                        guard_dispatch,
                    );
                    guard_idx += 1;
                    fused_guard_at = Some(op_idx + 1);
                }
                None => emit_cond(&mut sink, constants, value_types, op, kind),
            }
            // A comparison result is never a Ref, so the store-on-def tail has
            // nothing to do for it.
            continue;
        }
        // The whole-function target is the one a label-less bridge module
        // carries; a region brings its own.
        let jump_external: Option<(u32, u32, Option<(u32, u32)>, FrameGeometry)> =
            if op.opcode == OpCode::Jump {
                match external_jump_by_op.get(op_idx).copied().flatten() {
                    Some(ext) => Some((ext.slot, ext.key, None, ext.frame)),
                    None if !has_loop => Some((
                        external_jump_slot,
                        external_jump_key,
                        external_jump_wide,
                        ca.external_jump_frame.unwrap_or(frame),
                    )),
                    None => None,
                }
            } else {
                None
            };
        match op.opcode {
            OpCode::Label => {}

            OpCode::Jump if jump_external.is_some() => {
                let (external_jump_slot, external_jump_key, external_jump_wide, jump_frame) =
                    jump_external.expect(
                        "the arm guard just established this JUMP has a cross-module target",
                    );
                // A JUMP in a trace with no local LABEL closes back into a
                // *separate* loop module (a loop-closing bridge). There is no
                // enclosing `loop` to `br` to, so hand the jump args — the
                // loop's next inputargs, in inputarg order — to the target and
                // `return_call_indirect` its table slot. The tail call reuses
                // this frame instead of nesting, so the loop⇄bridge cycle holds
                // at constant stack depth.
                //
                // A target that published a parameter entry takes them as wasm
                // parameters; otherwise they go through the frame input slots
                // the way `execute_token` fills them. The jump args are this
                // bridge's SSA locals (or constants), and the input slots are a
                // disjoint frame region from any Ref home slot a resolve might
                // load, so storing each pair in turn cannot feed a clobbered
                // read (unlike the local back-edge's parallel move into shared
                // loop locals).
                let jump_args = op.getarglist();
                // Set the resume-at-LABEL dispatch key so a peeled target
                // re-enters at the JUMP's target LABEL — skipping the code
                // before it — instead of re-running the function from its
                // entry. `compile_bridge` resolves the target label ordinal
                // from the JUMP descr and passes `ordinal + 1` here; the
                // target's entry `br_table` lands on that label's resume
                // loader. Harmless for a non-peeled target, which has no
                // dispatch and ignores the slot (`external_jump_key` 0).
                //
                // The key travels through the frame either way: it selects the
                // entry `br_table` arm, which runs before any parameter is
                // read, so it is not one of the values the wide entry takes.
                let store_dispatch_key = |sink: &mut PeepSink<'_, '_>| {
                    sink.local_get(0); // frame_ptr
                    sink.i64_const(external_jump_key as i64); // dispatch key
                    // Target's `br_table` loads its own `dispatch_key_ofs`.
                    sink.i64_store(mem64(jump_frame.dispatch_key_ofs));
                };
                if let Some((wide_slot, wide_type_idx)) = external_jump_wide
                    .filter(|_| jump_args.len() <= crate::FROZEN_LABEL_PARAM_ARITY)
                {
                    // The target's narrow entry is a shim that loads
                    // `FROZEN_LABEL_PARAM_ARITY` frame slots and tail-calls the
                    // wide one, so storing the args here only to have them read
                    // straight back is a round trip through memory. Both its
                    // entry input loader and every LABEL resume loader read the
                    // parameters, so hand the values over directly.
                    store_dispatch_key(&mut sink);
                    sink.local_get(0); // frame_ptr argument to the loop
                    for k in 0..crate::FROZEN_LABEL_PARAM_ARITY {
                        match jump_args.get(k) {
                            Some(jump_arg) => {
                                emit_resolve(&mut sink, constants, value_types, jump_arg.to_opref())
                            }
                            // `compile_bridge` accepts this JUMP only when its
                            // arity equals the target label's argument count,
                            // and the loader reads exactly that many, so the
                            // parameters past it are never read. They exist to
                            // make one function type serve every arity.
                            None => {
                                sink.i64_const(0);
                            }
                        }
                    }
                    sink.i32_const(wide_slot as i32); // wide table slot
                    emit_push_site(&mut sink, &site_gcmap, op_idx);
                    sink.return_call_indirect(0, wide_type_idx);
                } else {
                    for (i, jump_arg) in jump_args.iter().enumerate() {
                        sink.local_get(0); // frame_ptr
                        emit_resolve(&mut sink, constants, value_types, jump_arg.to_opref());
                        // Narrow shim reloads `spill_slot_ofs` of the target.
                        sink.i64_store(mem64(jump_frame.spill_slot_ofs(i as u64)));
                    }
                    store_dispatch_key(&mut sink);
                    sink.local_get(0); // frame_ptr argument to the loop
                    sink.i32_const(external_jump_slot as i32); // table slot
                    emit_push_site(&mut sink, &site_gcmap, op_idx);
                    sink.return_call_indirect(0, 0); // table 0, type 0: (i32) -> i32
                }
            }

            OpCode::Jump => {
                // `patch_jump_for_descr` makes a replacement the next
                // iteration of the running loop. The module cannot be
                // rewritten, so the back-edge loads the owner's cell and
                // tail-calls the slot `replace_module` already swapped.
                // Address 0 keeps this arm a plain `br` (tests).
                if ca.resume_entry_addr != 0 {
                    sink.i32_const(ca.resume_entry_addr as i32);
                    sink.i32_load(mem32(0));
                    sink.i32_const(ca.resume_generation as i32);
                    sink.i32_gt_u();
                    sink.if_(BlockType::Empty);
                    let jump_args = op.getarglist();
                    for (i, jump_arg) in jump_args.iter().enumerate() {
                        sink.local_get(0);
                        let opref = jump_arg.to_opref();
                        if !opref.is_constant() && value_types.ty(opref.raw()) == ValType::F64 {
                            emit_resolve_f64(&mut sink, constants, value_types, opref);
                            sink.i64_reinterpret_f64();
                        } else {
                            emit_resolve(&mut sink, constants, value_types, opref);
                        }
                        sink.i64_store(mem64(frame.spill_slot_ofs(i as u64)));
                    }
                    // Peeled: key = label ordinal + 1 lands on that LABEL's
                    // resume loader. Header included. No local label, or a
                    // non-peeled loop whose entry is the loop, uses key 0.
                    let dispatch_key = jump_label_ordinal(ops, op)
                        .filter(|&j| key_dispatch && j < num_labels)
                        .map(|j| j + 1)
                        .unwrap_or(0);
                    sink.local_get(0);
                    sink.i64_const(dispatch_key as i64);
                    sink.i64_store(mem64(frame.dispatch_key_ofs));
                    sink.local_get(0);
                    sink.i32_const((ca.resume_entry_addr + 4) as i32);
                    sink.i32_load(mem32(0));
                    emit_push_site(&mut sink, &site_gcmap, op_idx);
                    sink.return_call_indirect(0, 0);
                    sink.end();
                }
                // The jump rebinds the loop's label args to the jump args — a
                // parallel move. A jump arg may read a target local that another
                // pair overwrites (e.g. the swap `x, y = y, x` → x<-y, y<-x), so
                // resolving-then-storing each pair in turn would feed a clobbered
                // value to a later read. Do all reads first (push every resolved
                // jump arg onto the operand stack), then all writes (pop into the
                // targets in reverse, the stack being LIFO).
                let label_args = find_label_args(ops, op);
                let jump_args = op.getarglist();
                let n = jump_args.len().min(label_args.len());
                // A pair whose jump arg IS its label arg rebinds the local to
                // the value it already holds, so its read/write contributes
                // nothing: every read precedes every write, and no other pair
                // writes the same target (a LABEL's args are distinct boxes,
                // asserted below), so dropping the pair leaves every remaining
                // read and write unchanged. The home-refresh loop below already
                // skips this case for the same reason.
                let moved: Vec<usize> = (0..n)
                    .filter(|&i| {
                        let jarg = jump_args[i].to_opref();
                        if jarg.is_constant() {
                            return true;
                        }
                        let larg = label_args[i];
                        if larg.is_constant() {
                            return true;
                        }
                        jarg.raw() != larg.raw()
                            && value_types.local(jarg.raw()) != value_types.local(larg.raw())
                    })
                    .collect();
                debug_assert!(
                    {
                        let mut seen: Vec<u32> = label_args[..n].iter().map(|a| a.raw()).collect();
                        seen.sort_unstable();
                        seen.windows(2).all(|w| w[0] != w[1])
                    },
                    "LABEL args must be distinct for the identity-pair skip to be a no-op"
                );
                for &i in &moved {
                    let label_arg = label_args[i];
                    if value_types.ty(label_arg.raw()) == ValType::F64 {
                        emit_resolve_f64(
                            &mut sink,
                            constants,
                            value_types,
                            jump_args[i].to_opref(),
                        );
                    } else {
                        emit_resolve(&mut sink, constants, value_types, jump_args[i].to_opref());
                    }
                }
                for &i in moved.iter().rev() {
                    sink.local_set(value_types.local(label_args[i].raw()));
                }
                // The parallel move rebinds loop-carried locals without going
                // through store-on-def, so a Ref label arg that is REBOUND to a
                // new value has a stale home slot; refresh it before branching
                // back so the next iteration's reload-after-allocation sees the
                // current value. Skip identity self-moves (jump arg == label
                // arg): the value is loop-invariant, so the home written by the
                // entry/resume loader already holds it and re-storing it every
                // iteration is redundant.
                for i in 0..n {
                    let la = label_args[i];
                    if let Some(h) = ref_homes.home(la) {
                        // Skip the refresh for a loop-invariant self-move (the jump arg
                        // is the label arg itself, so the value flows back unchanged and
                        // the home written by the entry/resume loader is still current).
                        // A constant jump arg is never a self-move, and OpRef::raw() must
                        // not be called on an inline constant, so guard the comparison.
                        let jarg = jump_args[i].to_opref();
                        if !jarg.is_constant()
                            && (jarg.raw() == la.raw()
                                || value_types.local(jarg.raw()) == value_types.local(la.raw()))
                        {
                            continue;
                        }
                        sink.local_get(0);
                        sink.local_get(value_types.local(la.raw()));
                        sink.i64_store(mem64(frame.home_ofs(h as u64)));
                    }
                }
                // A region closing at a LABEL it has no `br` to re-enters the
                // dispatch at the key that lands past that label's resume
                // loader: the parallel move above already left the label args
                // in their locals, so none of them goes through a frame slot.
                // The captures still take the loader's restore. A preamble
                // region sits past the `end` of the header `loop`, so it leaves
                // that way even when it names the header itself.
                let resume_at = if !resume_dispatch {
                    None
                } else if in_outside_region {
                    Some(
                        jump_label_ordinal(ops, op)
                            .filter(|&j| j < num_labels)
                            .ok_or_else(|| {
                                BackendError::Unsupported(
                                    "wasm backend: an outside-loop inline region closes at \
                                     no resumable LABEL"
                                        .into(),
                                )
                            })?,
                    )
                } else {
                    jump_resume_ordinal(ops, op, num_labels)
                };
                match resume_at {
                    Some(j) => {
                        emit_label_capture_restore(
                            &mut sink,
                            label_resume,
                            value_types,
                            ref_homes,
                            frame,
                            j,
                        );
                        sink.i32_const((num_labels + 1 + j) as i32);
                        sink.local_set(resume_key_local);
                        sink.br(block_exit_depth - 1);
                    }
                    None => {
                        sink.br(open_bridge_blocks);
                    }
                }
            }

            OpCode::Finish => {
                // x86 `genop_finish` else-arm stores `jf_gcmap = 0` when
                // there is no `_finish_gcmap`, and calls that store
                // redundant. `_finish_gcmap` is only retained for
                // `GUARD_NOT_FORCED_2`. A leftover `jf_force_descr` from
                // the `GUARD_NOT_FORCED` after `CALL_ASSEMBLER` is not
                // that map, and the CA caller footer on this path is
                // only `_call_footer_shadowstack` (`SUB`). The nursery
                // bump already wrote the callee gcmap; leaving it until
                // the pop is what the collector sees in the window
                // before the frame is unrooted. A host-entered loop
                // still publishes in `execute_token`.
                emit_guard_exit(
                    &mut sink,
                    constants,
                    value_types,
                    guard_idx,
                    op,
                    block_exit_depth,
                    guard_dispatch,
                    // Emitted at statement level: this exit is unconditional,
                    // so no `if` stands between it and the region blocks.
                    0,
                );
                guard_idx += 1;
            }

            // ── Guards ──
            OpCode::GuardTrue | OpCode::VecGuardTrue => {
                emit_guard_true(
                    &mut sink,
                    constants,
                    value_types,
                    guard_idx,
                    op,
                    block_exit_depth,
                    guard_dispatch,
                );
                guard_idx += 1;
            }
            OpCode::GuardFalse | OpCode::VecGuardFalse => {
                emit_guard_false(
                    &mut sink,
                    constants,
                    value_types,
                    guard_idx,
                    op,
                    block_exit_depth,
                    guard_dispatch,
                );
                guard_idx += 1;
            }
            OpCode::GuardValue => {
                // GUARD_VALUE checks bit-equality against the promoted constant:
                // Value::eq (value.rs) compares floats by to_bits() (0.0 != -0.0,
                // NaN == same-bit NaN, per history.py same_constant), which the
                // dynasm/cranelift siblings implement as an integer bit-compare.
                // emit_resolve pushes an F64 operand's i64 bits, so i64_ne is the
                // correct compare for both int and float — an IEEE f64.ne would
                // wrongly pass -0.0 == +0.0 (and fail NaN == same-bit NaN).
                emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                emit_resolve(&mut sink, constants, value_types, op.arg(1).to_opref());
                sink.i64_ne();
                emit_guard_if_exit(
                    &mut sink,
                    constants,
                    value_types,
                    guard_idx,
                    op,
                    block_exit_depth,
                    guard_dispatch,
                );
                guard_idx += 1;
            }
            OpCode::GuardNonnull => {
                emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                sink.i64_eqz();
                emit_guard_if_exit(
                    &mut sink,
                    constants,
                    value_types,
                    guard_idx,
                    op,
                    block_exit_depth,
                    guard_dispatch,
                );
                guard_idx += 1;
            }
            OpCode::GuardIsnull => {
                emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                sink.i64_const(0);
                sink.i64_ne();
                emit_guard_if_exit(
                    &mut sink,
                    constants,
                    value_types,
                    guard_idx,
                    op,
                    block_exit_depth,
                    guard_dispatch,
                );
                guard_idx += 1;
            }
            OpCode::GuardClass | OpCode::GuardNonnullClass => {
                // x86/assembler.py _cmp_guard_class:
                //   offset = self.cpu.vtable_offset
                //   if offset is not None: CMP(mem(loc_ptr, offset), classptr)
                //   else:
                //       assert isinstance(loc_classptr, ImmedLoc)
                //       expected_typeid = gc_ll_descr.
                //           get_typeid_from_classptr_if_gcremovetypeptr(...)
                //       _cmp_guard_gc_type(loc_ptr, ImmedLoc(expected_typeid))
                if let Some(off_usize) = vtable_offset {
                    let off = off_usize as u64;
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                    sink.i32_wrap_i64(); // struct ptr (i64) → i32 address
                    // The typeptr (`ob_type`) is a pointer-width field: 4
                    // bytes on wasm32. Reading it as i64 would fold in the
                    // following field's bytes and never match the class
                    // immediate. Load 4 bytes and zero-extend.
                    sink.i64_load32_u(memarg(off, 2));
                    emit_resolve(&mut sink, constants, value_types, op.arg(1).to_opref());
                    sink.i64_ne();
                } else {
                    // x86/assembler.py `_cmp_guard_class` hands the
                    // gcremovetypeptr case to `_cmp_guard_gc_type`, whose
                    // layout keeps the type id in the object's first word.
                    // majit keeps it in the GC header word placed immediately
                    // before the payload — the lower `TYPE_ID_BITS` of
                    // `majit_gc::header::GcHeader`'s `tid_and_flags`, the
                    // address the `GuardGcType`, `GuardIsObject` and
                    // `GuardSubclass` arms read. Under that layout `obj[0]` is
                    // a payload field, so comparing it against a type id
                    // answers a different question.
                    //
                    // Reading the header instead is not enough on its own
                    // here: this arm evaluates the class compare
                    // unconditionally and ORs the null test in afterwards, so
                    // for a NULL receiver `obj - GcHeader::SIZE` addresses
                    // below linear memory and traps instead of failing the
                    // guard — `genop_guard_guard_nonnull_class` avoids that
                    // with a forward jump this arm does not have. Decline: a
                    // frontend that emits GUARD_CLASS configures the vtable
                    // offset (pyre passes `OB_TYPE_OFFSET`), so no trace pays
                    // for the decline.
                    return Err(BackendError::Unsupported(format!(
                        "wasm backend: {:?} with cpu.vtable_offset = None \
                         (gcremovetypeptr) is unsupported; the type id lives \
                         in the GC header and this arm has no null-safe \
                         header compare",
                        op.opcode
                    )));
                }
                if op.opcode == OpCode::GuardNonnullClass {
                    // x86/assembler.py genop_guard_guard_nonnull_class wraps
                    // `_cmp_guard_class` in `CMP(ptr, 1)` plus a forward `B`
                    // jump, so a NULL receiver reaches the guard already
                    // failing and never has its class read. Here the class
                    // compare above has already run — harmlessly, since the
                    // only shape this arm lowers reads the vtable offset, and
                    // a NULL receiver puts that read inside the first page —
                    // so the guard's answer is the disjunction.
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                    sink.i64_eqz();
                    sink.i32_or();
                }
                emit_guard_if_exit(
                    &mut sink,
                    constants,
                    value_types,
                    guard_idx,
                    op,
                    block_exit_depth,
                    guard_dispatch,
                );
                guard_idx += 1;
            }
            OpCode::GuardNoOverflow => {
                // RPython: 0 args — overflow flag implicit from preceding ovf op.
                // If the optimizer proved the operation cannot overflow, the
                // overflow op is absent and this guard is redundant.
                if !ovf_flag_live {
                    guard_idx += 1;
                    continue;
                }
                ovf_flag_live = false;
                sink.local_get(ovf_flag_local);
                sink.i64_const(0);
                sink.i64_ne();
                emit_guard_if_exit(
                    &mut sink,
                    constants,
                    value_types,
                    guard_idx,
                    op,
                    block_exit_depth,
                    guard_dispatch,
                );
                guard_idx += 1;
            }
            OpCode::GuardOverflow => {
                assert!(ovf_flag_live, "GuardOverflow without preceding overflow op");
                ovf_flag_live = false;
                sink.local_get(ovf_flag_local);
                sink.i64_eqz();
                emit_guard_if_exit(
                    &mut sink,
                    constants,
                    value_types,
                    guard_idx,
                    op,
                    block_exit_depth,
                    guard_dispatch,
                );
                guard_idx += 1;
            }
            OpCode::GuardNotInvalidated => {
                // PRE-EXISTING-ADAPTATION.  `opassembler.py
                // emit_op_guard_not_invalidated` leaves a same-width no-op that
                // `aarch64/runner.py invalidate_loop` overwrites with a branch, so
                // the guard costs nothing per iteration; the dynasm backends do
                // that.  A wasm module is code the host engine compiles, and
                // nothing in the guest can rewrite an instruction of it, so the
                // guard site observes the owning loop token's invalidation flag
                // on every entry instead.  On wasm32 the Arc allocation lives
                // in shared linear memory, so its pointer is addressable by the
                // trace.  There is no convergence path while the module is the
                // unit of compilation.
                if invalidated_flag_addr != 0 {
                    sink.i32_const(invalidated_flag_addr as i32);
                    // The flag byte is the test: `emit_guard_if_exit` opens
                    // with an `if`, which is already `!= 0`.
                    sink.i32_load8_u(memarg(0, 0));
                    emit_guard_if_exit(
                        &mut sink,
                        constants,
                        value_types,
                        guard_idx,
                        op,
                        block_exit_depth,
                        guard_dispatch,
                    );
                }
                guard_idx += 1;
            }
            OpCode::GuardNotForced => {
                // x86/assembler.py genop_guard_guard_not_forced:
                // `CMP [rbp + jf_descr], 0`, fail when nonzero. `Backend::force`
                // stamps that mark on its way out, so this guard is what turns a
                // force that landed inside the preceding call into a deopt: the
                // trace must not run on holding virtualized fields the force has
                // already written back, and the virtuals `handle_async_forcing`
                // materialized are attached for THIS exit's resume to consume.
                // `exit_table_base == 0` is a direct codegen test whose frame
                // sits at address 0: the taken bit lives in `frame[0]`. A real
                // compile compares `jf_descr` (`genop_guard_guard_not_forced`).
                if guard_dispatch.exit_table_base == 0 {
                    const FORCE_TAKEN_BIT: i64 = 1 << 32;
                    const FORCE_TAKEN_HALF_OFS: u64 = 4;
                    const _: () = assert!(FORCE_TAKEN_BIT == 1 << 32);
                    sink.local_get(0);
                    sink.i32_load(memarg(FORCE_TAKEN_HALF_OFS, 2));
                    sink.i32_const(1);
                    sink.i32_and();
                } else {
                    emit_header_base(&mut sink);
                    sink.i32_load(memarg(majit_backend::jitframe::JF_DESCR_OFS as u64, 2));
                }
                emit_guard_if_exit(
                    &mut sink,
                    constants,
                    value_types,
                    guard_idx,
                    op,
                    block_exit_depth,
                    guard_dispatch,
                );
                guard_idx += 1;
            }
            OpCode::GuardNotForced2 => {
                // x86/regalloc.py consider_guard_not_forced_2 answers with
                // `assembler.store_force_descr`, not with a branch: unlike
                // GUARD_NOT_FORCED this one is not paired with a preceding call
                // to test, it is what `store_token_in_vable` emits before a
                // FINISH so a force arriving while the virtualizable is still
                // armed can still rebuild a deadframe. Arm, do not test.
                emit_force_arm(
                    &mut sink,
                    constants,
                    value_types,
                    ref_homes,
                    frame,
                    op,
                    exit_index(op, guard_idx, guard_dispatch.attached),
                    None,
                    guard_dispatch.const_tables,
                    guard_dispatch.const_table_base,
                    guard_dispatch.exit_table_base,
                    guard_idx.wrapping_sub(guard_dispatch.fail_index_base),
                );
                guard_idx += 1;
            }
            OpCode::GuardNoException => {
                // x86/assembler.py generate_guard_no_exception:
                // `CMP(pos_exception, imm0)` — fail the guard when a pending
                // exception is present, keyed on the exception TYPE slot
                // (pos_exception), the same slot GuardException reads and the
                // one llgraph's `last_exception is not None` tests. The slot
                // lives in the host's shared linear memory; load it by absolute
                // address (the trace imports env.memory).
                sink.i32_const(runtime_addr(crate::jit_exc_type_addr));
                sink.i64_load(mem64(0));
                sink.i64_const(0);
                sink.i64_ne();
                emit_guard_if_exit(
                    &mut sink,
                    constants,
                    value_types,
                    guard_idx,
                    op,
                    block_exit_depth,
                    guard_dispatch,
                );
                guard_idx += 1;
            }
            OpCode::GuardException => {
                // x86/assembler.py genop_guard_guard_exception:
                //   load pos_exception; CMP expected; guard on equal; then
                //   _store_and_reset_exception: resloc = pos_exc_value;
                //   pos_exception = 0; pos_exc_value = 0.
                let exc_type_addr = runtime_addr(crate::jit_exc_type_addr);
                let exc_value_addr = runtime_addr(crate::jit_exc_value_addr);
                sink.i32_const(exc_type_addr);
                sink.i64_load(mem64(0));
                emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                sink.i64_ne();
                emit_guard_if_exit(
                    &mut sink,
                    constants,
                    value_types,
                    guard_idx,
                    op,
                    block_exit_depth,
                    guard_dispatch,
                );
                guard_idx += 1;
                // Success path: capture the caught exception into the result
                // var, then clear both slots.
                let vi = op.pos().get().raw();
                if !OpRef::raw_is_constant(vi) {
                    sink.i32_const(exc_value_addr);
                    sink.i64_load(mem64(0));
                    sink.local_set(value_types.local(vi));
                }
                sink.i32_const(exc_type_addr);
                sink.i64_const(0);
                sink.i64_store(mem64(0));
                sink.i32_const(exc_value_addr);
                sink.i64_const(0);
                sink.i64_store(mem64(0));
            }

            // ── Integer arithmetic ──
            OpCode::IntAdd => emit_binop(&mut sink, constants, value_types, op, BinOp::I64Add),
            OpCode::IntSub => emit_binop(&mut sink, constants, value_types, op, BinOp::I64Sub),
            OpCode::IntMul => emit_binop(&mut sink, constants, value_types, op, BinOp::I64Mul),
            OpCode::IntFloorDiv => {
                emit_binop(&mut sink, constants, value_types, op, BinOp::I64DivS)
            }
            OpCode::IntMod => emit_binop(&mut sink, constants, value_types, op, BinOp::I64RemS),
            OpCode::IntAnd => emit_binop(&mut sink, constants, value_types, op, BinOp::I64And),
            OpCode::IntOr => emit_binop(&mut sink, constants, value_types, op, BinOp::I64Or),
            OpCode::IntXor => emit_binop(&mut sink, constants, value_types, op, BinOp::I64Xor),
            OpCode::IntLshift => emit_binop(&mut sink, constants, value_types, op, BinOp::I64Shl),
            OpCode::IntRshift => emit_binop(&mut sink, constants, value_types, op, BinOp::I64ShrS),
            OpCode::UintRshift => emit_binop(&mut sink, constants, value_types, op, BinOp::I64ShrU),
            // High 64 bits of the unsigned 64×64→128 product. The optimizer
            // emits this for division/modulo-by-constant strength reduction;
            // wasm has no mul-high instruction, so expand via 32-bit split.
            OpCode::UintMulHigh => emit_umulhi(
                &mut sink,
                constants,
                value_types,
                op,
                value_types.last_local(),
            ),

            // Overflow variants: compute result + overflow flag
            OpCode::IntAddOvf | OpCode::IntSubOvf | OpCode::IntMulOvf => {
                let binop = match op.opcode {
                    OpCode::IntAddOvf => BinOp::I64Add,
                    OpCode::IntSubOvf => BinOp::I64Sub,
                    OpCode::IntMulOvf => BinOp::I64Mul,
                    _ => unreachable!(),
                };
                // Every overflow form leaves its predicate on the stack, so
                // any of them can hand it straight to an adjacent guard.
                let fused_guard = next_ovf_guard(ops, op_idx);
                ovf_flag_live = match emit_ovf_binop(
                    &mut sink,
                    constants,
                    value_types,
                    op,
                    binop,
                    value_types.last_local(),
                    ovf_flag_local,
                    fused_guard.map(|guard| guard.opcode),
                ) {
                    OvfFlag::Absent => false,
                    OvfFlag::InLocal => true,
                    OvfFlag::FusedCond => {
                        let guard = fused_guard.expect("fused overflow condition requires guard");
                        emit_guard_if_exit(
                            &mut sink,
                            constants,
                            value_types,
                            guard_idx,
                            guard,
                            block_exit_depth,
                            guard_dispatch,
                        );
                        guard_idx += 1;
                        fused_guard_at = Some(op_idx + 1);
                        false
                    }
                };
            }

            // ── Unary ops ──
            OpCode::IntNeg => emit_unary_vi(
                &mut sink,
                constants,
                value_types,
                op,
                |s| {
                    s.i64_const(0);
                },
                |s| {
                    s.i64_sub();
                },
            ),
            OpCode::IntInvert => emit_unary_vi(
                &mut sink,
                constants,
                value_types,
                op,
                |s| {
                    s.i64_const(-1);
                },
                |s| {
                    s.i64_xor();
                },
            ),
            // resoperation.py `int_between(a, b, c)` is the three-operand
            // range test `a <= b < c`, signed on all three. jtransform lowers
            // the name directly (`jtransform_opname.rs`), and the bigint
            // compare path mints it as `int_between(-1, i2 >> 48, 1)`, so a
            // trace can carry it.
            OpCode::IntBetween => {
                let vi = op.pos().get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                    emit_resolve(&mut sink, constants, value_types, op.arg(1).to_opref());
                    sink.i64_le_s();
                    emit_resolve(&mut sink, constants, value_types, op.arg(1).to_opref());
                    emit_resolve(&mut sink, constants, value_types, op.arg(2).to_opref());
                    sink.i64_lt_s();
                    sink.i32_and();
                    sink.i64_extend_i32_u();
                    sink.local_set(value_types.local(vi));
                }
            }

            // `float_mod` is C `fmod`: the interpreter evaluates it as Rust's
            // `a % b`, which truncates toward zero. Wasm has no float
            // remainder instruction, and synthesizing it from div/floor is
            // observably wrong. Call the interpreter module's exact helper
            // through the shared table, staying guest-side rather than
            // declining the whole trace or crossing the host trampoline.
            OpCode::FloatMod => {
                let vi = op.pos().get().raw();
                if !OpRef::raw_is_constant(vi) {
                    let sig = (vec![ValType::F64, ValType::F64], Some(ValType::F64));
                    let type_idx = typed_residual_type_indices.get(&sig).ok_or_else(|| {
                        BackendError::Unsupported(
                            "wasm codegen: FloatMod helper signature was not declared".into(),
                        )
                    })?;
                    emit_resolve_f64(&mut sink, constants, value_types, op.arg(0).to_opref());
                    emit_resolve_f64(&mut sink, constants, value_types, op.arg(1).to_opref());
                    sink.i32_const(alloc.fmod_fn_ptr as i32);
                    emit_push_site(&mut sink, &site_gcmap, op_idx);
                    sink.call_indirect(0, *type_idx);
                    sink.local_set(value_types.local(vi));
                }
            }

            // ── Extended integer ops ──
            OpCode::IntSignext => {
                // int_signext(val, num_bytes): sign-extend from num_bytes width
                let vi = op.pos().get().raw();
                if !OpRef::raw_is_constant(vi) {
                    let arg1 = op.arg(1).to_opref();
                    if let Some(num_bytes) = const_operand_value(constants, arg1) {
                        emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                        let shift = 64 - num_bytes * 8;
                        if shift > 0 && shift < 64 {
                            sink.i64_const(shift);
                            sink.i64_shl();
                            sink.i64_const(shift);
                            sink.i64_shr_s();
                        }
                    } else {
                        // support.py `int_signext`: widths 1..=8 sign-extend;
                        // larger positive widths are the identity.  The IR
                        // producer guarantees a positive width. Wasm shifts
                        // mask their count, so compute both candidates and
                        // select the shifted one only inside the valid range.
                        let value = op.arg(0).to_opref();
                        let emit_shift = |sink: &mut PeepSink<'_, '_>| {
                            sink.i64_const(8);
                            emit_resolve(sink, constants, value_types, arg1);
                            sink.i64_sub();
                            sink.i64_const(3);
                            sink.i64_shl();
                        };
                        emit_resolve(&mut sink, constants, value_types, value);
                        emit_shift(&mut sink);
                        sink.i64_shl();
                        emit_shift(&mut sink);
                        sink.i64_shr_s();
                        emit_resolve(&mut sink, constants, value_types, value);
                        emit_resolve(&mut sink, constants, value_types, arg1);
                        sink.i64_const(8);
                        sink.i64_le_s();
                        sink.select();
                    }
                    sink.local_set(value_types.local(vi));
                }
            }
            OpCode::IntForceGeZero => {
                // max(val, 0)
                let vi = op.pos().get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                    // `select` answers with the FIRST value when its condition
                    // holds, so the condition is the one that keeps `val`.
                    let tmp_local = value_types.local(vi); // reuse result local as temp
                    sink.local_tee(tmp_local);
                    sink.i64_const(0);
                    sink.local_get(tmp_local);
                    sink.i64_const(0);
                    sink.i64_ge_s();
                    sink.select();
                    sink.local_set(value_types.local(vi));
                }
            }

            // ── Float floor/mod ──
            OpCode::FloatFloorDiv => {
                let vi = op.pos().get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve_f64(&mut sink, constants, value_types, op.arg(0).to_opref());
                    emit_resolve_f64(&mut sink, constants, value_types, op.arg(1).to_opref());
                    sink.f64_div();
                    sink.f64_floor();
                    sink.local_set(value_types.local(vi));
                }
            }

            // ── Float/Int conversions ──
            OpCode::CastFloatToInt => {
                let vi = op.pos().get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve_f64(&mut sink, constants, value_types, op.arg(0).to_opref());
                    sink.i64_trunc_sat_f64_s();
                    sink.local_set(value_types.local(vi));
                }
            }
            OpCode::CastIntToFloat => {
                let vi = op.pos().get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                    sink.f64_convert_i64_s();
                    sink.local_set(value_types.local(vi));
                }
            }
            OpCode::ConvertFloatBytesToLonglong => {
                let vi = op.pos().get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                    sink.local_set(value_types.local(vi));
                }
            }
            OpCode::ConvertLonglongBytesToFloat => {
                let vi = op.pos().get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                    sink.f64_reinterpret_i64();
                    sink.local_set(value_types.local(vi));
                }
            }
            OpCode::CastFloatToSinglefloat => {
                let vi = op.pos().get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve_f64(&mut sink, constants, value_types, op.arg(0).to_opref());
                    sink.f32_demote_f64();
                    sink.i32_reinterpret_f32();
                    sink.i64_extend_i32_u();
                    sink.local_set(value_types.local(vi));
                }
            }
            OpCode::CastSinglefloatToFloat => {
                let vi = op.pos().get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                    sink.i32_wrap_i64();
                    sink.f32_reinterpret_i32();
                    sink.f64_promote_f32();
                    sink.local_set(value_types.local(vi));
                }
            }

            // ── Pointer/Int conversions ──
            OpCode::CastPtrToInt => {
                // `cast_ptr_to_int` produces `Signed` (a machine word). On
                // wasm32 a pointer is 4 bytes, so the value carried in the i64
                // value ABI must be the 32-bit pointer reinterpreted as a
                // signed word — a sign-extending widen, not the zero-extension
                // a Ref receives on entry (`i64_extend_i32_u` loads, or a Rust
                // residual shim's `ptr as i64`). Without this, a tagged small
                // int with the top payload bit set (`(v<<1)|1` for v<0 or large
                // v, rtagged.py `ll_unboxed_to_int`) reads back with a zero
                // high half, and the trailing arithmetic `IntRshift(,1)` untag
                // (a 64-bit `i64.shr_s`) recovers the wrong value. `i32.wrap` +
                // `i64.extend_i32_s` is a no-op for a real heap pointer (top bit
                // clear on a <2GB linear memory), so this is the width-correct
                // lowering for both tagged and boxed operands.
                let vi = op.pos().get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                    sink.i32_wrap_i64();
                    sink.i64_extend_i32_s();
                    sink.local_set(value_types.local(vi));
                }
            }
            OpCode::CastIntToPtr | OpCode::CastOpaquePtr => {
                let vi = op.pos().get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                    sink.local_set(value_types.local(vi));
                }
            }

            OpCode::VirtualRefI | OpCode::VirtualRefR | OpCode::VirtualRefFinish => {
                // OptSimplify.optimize_VIRTUAL_REF owns this rewrite; backend
                // aliases would conceal a missing frontend transformation.
                return Err(BackendError::Unsupported(format!(
                    "wasm backend: {:?} must be lowered by the optimizer",
                    op.opcode
                )));
            }

            // ── SameAs (forwarding) ──
            OpCode::SameAsI | OpCode::SameAsR => {
                let vi = op.pos().get().raw();
                if !OpRef::raw_is_constant(vi) {
                    let source = op.arg(0).to_opref();
                    if source.is_none()
                        || source.is_constant()
                        || value_types.local(source.raw()) != value_types.local(vi)
                    {
                        emit_resolve(&mut sink, constants, value_types, source);
                        sink.local_set(value_types.local(vi));
                    }
                }
            }
            OpCode::SameAsF => {
                let vi = op.pos().get().raw();
                if !OpRef::raw_is_constant(vi) {
                    let source = op.arg(0).to_opref();
                    if source.is_none()
                        || source.is_constant()
                        || value_types.local(source.raw()) != value_types.local(vi)
                    {
                        emit_resolve_f64(&mut sink, constants, value_types, source);
                        sink.local_set(value_types.local(vi));
                    }
                }
            }

            // ── Field access (direct memory operations) ──
            OpCode::GetfieldGcI | OpCode::GetfieldRawI => {
                let vi = op.pos().get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref()); // struct ptr (i64)
                    sink.i32_wrap_i64(); // convert to i32 address
                    let field_offset = field_offset_from_descr(op);
                    let (size, signed) = field_size_sign_from_descr(op);
                    emit_sized_int_load(&mut sink, field_offset, size, signed);
                    sink.local_set(value_types.local(vi));
                }
            }
            OpCode::GetfieldGcR | OpCode::GetfieldRawR => {
                let vi = op.pos().get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                    sink.i32_wrap_i64();
                    let field_offset = field_offset_from_descr(op);
                    // Pointer load. The imported memory is not shared, so
                    // this stays `i64.load32_u` when the descr's
                    // `load_is_acquire` is set. The host sample of that
                    // field is `bh_getfield_gc_r`.
                    sink.i64_load32_u(memarg(field_offset, 2));
                    sink.local_set(value_types.local(vi));
                }
            }
            OpCode::SetfieldGc => {
                panic!("wasm codegen: SetfieldGc must have been lowered by rewrite_ops_for_gc");
            }
            OpCode::SetfieldRaw => {
                emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref()); // struct ptr
                sink.i32_wrap_i64();
                let field_offset = field_offset_from_descr(op);
                if field_is_float_from_descr(op) {
                    emit_resolve_f64(&mut sink, constants, value_types, op.arg(1).to_opref());
                    emit_float_store(&mut sink, field_offset, field_size_sign_from_descr(op).0)?;
                } else {
                    emit_resolve(&mut sink, constants, value_types, op.arg(1).to_opref()); // value
                    let size = setfield_store_size_from_descr(op);
                    emit_sized_int_store(&mut sink, field_offset, size);
                }
            }

            // ── Float field access ──
            OpCode::GetfieldGcF | OpCode::GetfieldRawF => {
                let vi = op.pos().get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                    sink.i32_wrap_i64();
                    let field_offset = field_offset_from_descr(op);
                    emit_float_load(&mut sink, field_offset, field_size_sign_from_descr(op).0)?;
                    sink.local_set(value_types.local(vi));
                }
            }

            // ── Array access ──
            OpCode::ArraylenGc => {
                let vi = op.pos().get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref()); // array ptr
                    sink.i32_wrap_i64();
                    let (len_offset, len_size) = array_len_layout_from_descr(op);
                    // The length is a word-sized field (`Signed`/`WORD`): read it
                    // at its real width, like `bh_arraylen_gc`. A fixed i64_load
                    // would fold the next field into the high half on wasm32.
                    emit_sized_int_load(&mut sink, len_offset, len_size, false);
                    sink.local_set(value_types.local(vi));
                }
            }
            OpCode::GetarrayitemGcI | OpCode::GetarrayitemGcPureI | OpCode::GetarrayitemRawI => {
                let vi = op.pos().get().raw();
                if !OpRef::raw_is_constant(vi) {
                    let base_size = emit_array_addr(&mut sink, constants, value_types, op);
                    let (item_size, signed) = array_item_access_size_sign(op);
                    emit_sized_int_load(&mut sink, base_size, item_size, signed);
                    sink.local_set(value_types.local(vi));
                }
            }
            OpCode::GetarrayitemGcR | OpCode::GetarrayitemGcPureR | OpCode::GetarrayitemRawR => {
                let vi = op.pos().get().raw();
                if !OpRef::raw_is_constant(vi) {
                    let base_size = emit_array_addr(&mut sink, constants, value_types, op);
                    let (item_size, signed) = array_item_access_size_sign(op);
                    emit_sized_int_load(&mut sink, base_size, item_size, signed);
                    sink.local_set(value_types.local(vi));
                }
            }
            OpCode::GetarrayitemGcF | OpCode::GetarrayitemGcPureF | OpCode::GetarrayitemRawF => {
                let vi = op.pos().get().raw();
                if !OpRef::raw_is_constant(vi) {
                    let base_size = emit_array_addr(&mut sink, constants, value_types, op);
                    emit_float_load(&mut sink, base_size, array_item_access_size_sign(op).0)?;
                    sink.local_set(value_types.local(vi));
                }
            }
            OpCode::SetarrayitemGc => {
                panic!("wasm codegen: SetarrayitemGc must have been lowered by rewrite_ops_for_gc");
            }
            OpCode::SetarrayitemRaw => {
                let base_size = emit_array_addr(&mut sink, constants, value_types, op);
                if array_item_is_float_from_descr(op) {
                    emit_resolve_f64(&mut sink, constants, value_types, op.arg(2).to_opref());
                    emit_float_store(&mut sink, base_size, array_item_access_size_sign(op).0)?;
                } else {
                    emit_resolve(&mut sink, constants, value_types, op.arg(2).to_opref()); // value
                    // A Ref item is pointer-width (4 bytes on wasm32). Storing a
                    // fixed 8 bytes would clobber the next item, or run past the
                    // array end on the last item and corrupt the heap.
                    let (item_size, _signed) = array_item_access_size_sign(op);
                    emit_sized_int_store(&mut sink, base_size, item_size);
                }
            }

            // Same contract as `SetfieldGc` / `SetarrayitemGc`: a Ref
            // interior store must have been rewritten to COND_CALL_GC_WB
            // plus a raw store. Reaching here would write the pointer
            // without the barrier.
            OpCode::SetinteriorfieldGc => {
                panic!(
                    "wasm codegen: SetinteriorfieldGc must have been lowered by rewrite_ops_for_gc"
                );
            }
            // Pre-rewrite interior-field, string and raw-memory ops. The rewriter
            // consumes these; reaching codegen with one is a producer bug.
            OpCode::GetinteriorfieldGcI
            | OpCode::GetinteriorfieldGcR
            | OpCode::GetinteriorfieldGcF
            | OpCode::SetinteriorfieldRaw
            | OpCode::Strlen
            | OpCode::Unicodelen
            | OpCode::Strgetitem
            | OpCode::Unicodegetitem
            | OpCode::Strsetitem
            | OpCode::Unicodesetitem
            | OpCode::Strhash
            | OpCode::Unicodehash
            | OpCode::Copystrcontent
            | OpCode::Copyunicodecontent
            | OpCode::RawLoadI
            | OpCode::RawLoadF
            | OpCode::RawStore
            | OpCode::GuardAlwaysFails => {
                return Err(BackendError::Unsupported(format!(
                    "wasm codegen: {:?} reached codegen without the GC rewrite",
                    op.opcode
                )));
            }
            // ── GC rewrite memory ops ──
            // These descriptor-free forms carry their complete layout in
            // operands. Supporting them here lets wasm consume the same
            // `GcRewriterImpl` output as the native backends.
            OpCode::GcLoadIndexedI | OpCode::GcLoadIndexedR | OpCode::GcLoadIndexedF => {
                let vi = op.pos().get().raw();
                if !OpRef::raw_is_constant(vi) {
                    if op.num_args() < 5 {
                        return Err(BackendError::Unsupported(format!(
                            "wasm codegen: {:?} expects [base, index, scale, offset, size]",
                            op.opcode
                        )));
                    }
                    let offset = emit_gc_indexed_addr(&mut sink, constants, value_types, op, 2, 3)?;
                    let (size, signed) = gc_rewrite_access_size(op, constants, 4)?;
                    let size = wasm_ref_access_size(op.opcode == OpCode::GcLoadIndexedR, size);
                    if op.opcode == OpCode::GcLoadIndexedF {
                        // `get_type_flag`: a float-kind load is 8 bytes.
                        if size == 8 {
                            sink.f64_load(mem64(offset));
                        } else {
                            return Err(BackendError::Unsupported(format!(
                                "wasm codegen: {:?} float load has size {size} (get_type_flag: lltype.Float is 8 bytes)",
                                op.opcode
                            )));
                        }
                    } else {
                        emit_sized_int_load(&mut sink, offset, size, signed);
                    }
                    sink.local_set(value_types.local(vi));
                }
            }
            OpCode::GcLoadI | OpCode::GcLoadR | OpCode::GcLoadF => {
                let vi = op.pos().get().raw();
                if !OpRef::raw_is_constant(vi) {
                    if op.num_args() < 3 {
                        return Err(BackendError::Unsupported(format!(
                            "wasm codegen: {:?} expects [base, offset, size]",
                            op.opcode
                        )));
                    }
                    let offset = emit_gc_offset_addr(
                        &mut sink,
                        constants,
                        value_types,
                        op.arg(0).to_opref(),
                        op.arg(1).to_opref(),
                    );
                    let (size, signed) = gc_rewrite_access_size(op, constants, 2)?;
                    let size = wasm_ref_access_size(op.opcode == OpCode::GcLoadR, size);
                    if op.opcode == OpCode::GcLoadF {
                        // `get_type_flag`: a float-kind load is 8 bytes.
                        if size == 8 {
                            sink.f64_load(mem64(offset));
                        } else {
                            return Err(BackendError::Unsupported(format!(
                                "wasm codegen: {:?} float load has size {size} (get_type_flag: lltype.Float is 8 bytes)",
                                op.opcode
                            )));
                        }
                    } else {
                        emit_sized_int_load(&mut sink, offset, size, signed);
                    }
                    sink.local_set(value_types.local(vi));
                }
            }
            OpCode::GcStore => {
                if op.num_args() < 4 {
                    return Err(BackendError::Unsupported(
                        "wasm codegen: GcStore expects [base, offset, value, size]".into(),
                    ));
                }
                let offset = emit_gc_offset_addr(
                    &mut sink,
                    constants,
                    value_types,
                    op.arg(0).to_opref(),
                    op.arg(1).to_opref(),
                );
                let (size, _) = gc_rewrite_access_size(op, constants, 3)?;
                let val = op.arg(2).to_opref();
                let size = wasm_ref_access_size(val.ty() == Some(Type::Ref), size);
                // Size 8 stays the bit-preserving integer store below.
                // `get_type_flag` never pairs a float value with another width.
                if value_is_f64(value_types, val) && size != 8 {
                    return Err(BackendError::Unsupported(format!(
                        "wasm codegen: float memory store has size {size} (get_type_flag: lltype.Float is 8 bytes)"
                    )));
                }
                emit_resolve(&mut sink, constants, value_types, val);
                emit_sized_int_store(&mut sink, offset, size);
            }
            OpCode::GcStoreIndexed => {
                if op.num_args() < 6 {
                    return Err(BackendError::Unsupported(
                        "wasm codegen: GcStoreIndexed expects \
                         [base, index, value, scale, offset, size]"
                            .into(),
                    ));
                }
                let offset = emit_gc_indexed_addr(&mut sink, constants, value_types, op, 3, 4)?;
                let (size, _) = gc_rewrite_access_size(op, constants, 5)?;
                let val = op.arg(2).to_opref();
                let size = wasm_ref_access_size(val.ty() == Some(Type::Ref), size);
                // Size 8 stays the bit-preserving integer store below.
                // `get_type_flag` never pairs a float value with another width.
                if value_is_f64(value_types, val) && size != 8 {
                    return Err(BackendError::Unsupported(format!(
                        "wasm codegen: float memory store has size {size} (get_type_flag: lltype.Float is 8 bytes)"
                    )));
                }
                emit_resolve(&mut sink, constants, value_types, val);
                emit_sized_int_store(&mut sink, offset, size);
            }

            // ── Exception handling ──
            OpCode::SaveException => {
                // x86/assembler.py genop_save_exception:
                //   _store_and_reset_exception → resloc = [pos_exc_value];
                //   [pos_exception] = 0; [pos_exc_value] = 0.
                // The result is the caught exception the resumed handler reads,
                // so it must be written even though the slots themselves are
                // shared with the host: skipping the op leaves the local null.
                let vi = op.pos().get().raw();
                if !OpRef::raw_is_constant(vi) {
                    sink.i32_const(runtime_addr(crate::jit_exc_value_addr));
                    sink.i64_load(mem64(0));
                    sink.local_set(value_types.local(vi));
                }
                sink.i32_const(runtime_addr(crate::jit_exc_type_addr));
                sink.i64_const(0);
                sink.i64_store(mem64(0));
                sink.i32_const(runtime_addr(crate::jit_exc_value_addr));
                sink.i64_const(0);
                sink.i64_store(mem64(0));
            }
            OpCode::SaveExcClass => {
                // x86/assembler.py genop_save_exc_class:
                //   MOV resloc, [pos_exception]
                let vi = op.pos().get().raw();
                if !OpRef::raw_is_constant(vi) {
                    sink.i32_const(runtime_addr(crate::jit_exc_type_addr));
                    sink.i64_load(mem64(0));
                    sink.local_set(value_types.local(vi));
                }
            }
            OpCode::RestoreException => {
                // x86/assembler.py _restore_exception:
                //   MOV [pos_exc_value], excvalloc
                //   MOV [pos_exception], exctploc
                sink.i32_const(runtime_addr(crate::jit_exc_value_addr));
                emit_resolve(&mut sink, constants, value_types, op.arg(1).to_opref());
                sink.i64_store(mem64(0));
                sink.i32_const(runtime_addr(crate::jit_exc_type_addr));
                emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                sink.i64_store(mem64(0));
            }

            // ── Conditional calls ──
            OpCode::CondCallGcWb => {
                // rewrite.py `gen_write_barrier`: one-arg COND_CALL_GC_WB
                // before the lowered GC_STORE. The SETFIELD_GC arm never
                // sees that store, so the barrier has to run here.
                emit_write_barrier(
                    &mut sink,
                    constants,
                    value_types,
                    residual_type_base,
                    wb,
                    op.arg(0).to_opref(),
                    None,
                    site_gcmap.get(op_idx).copied().unwrap_or(0),
                    ca.ca_reload_fn_ptr,
                    ca.jf_top_addr,
                )?;
            }
            OpCode::CondCallGcWbArray => {
                // rewrite.py `gen_write_barrier_array`: cards_set == 0
                // falls through to the plain remembered barrier.
                let card =
                    (wb.cards_set != 0 && wb.array_fn_ptr != 0).then(|| op.arg(1).to_opref());
                emit_write_barrier(
                    &mut sink,
                    constants,
                    value_types,
                    residual_type_base,
                    wb,
                    op.arg(0).to_opref(),
                    card,
                    site_gcmap.get(op_idx).copied().unwrap_or(0),
                    ca.ca_reload_fn_ptr,
                    ca.jf_top_addr,
                )?;
            }
            OpCode::CondCallN => {
                // x86/assembler.py `genop_discard_cond_call`: TEST cond; JZ
                // skip; CALL. The predicate is arg 0, the callee is arg 1, and
                // the rest are the call's own arguments. The call descr's
                // word, true-void, or table signature is a direct
                // `call_indirect`. A descr the table does not confirm
                // declines the trace.
                //
                // `do_conditional_call` asserts the callee forces no virtual or
                // virtualizable, so unlike the CALL arm this needs no force
                // bracket.
                let func = op.arg(1).to_opref();
                let call_args = &op.getarglist()[2..];
                let cond_on_stack = fused_condcall_at == Some(op_idx);
                if cond_on_stack {
                    fused_condcall_at = None;
                }
                // A fused comparison already left the nonzero-means-call i32
                // on the stack. The unfused word test cannot `i32.wrap_i64`
                // (bits above 32 would read as false), so it uses `i64.eqz`
                // and puts the call in the else arm.
                if cond_on_stack {
                    sink.if_(BlockType::Empty);
                } else {
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                    sink.i64_eqz();
                    sink.if_(BlockType::Empty);
                    sink.else_();
                }
                if let (Some(base), Some(nargs)) = (
                    residual_type_base,
                    conditional_call_i64_arity(op, constants),
                ) {
                    for arg in call_args {
                        emit_resolve(&mut sink, constants, value_types, arg.to_opref());
                    }
                    emit_resolve(&mut sink, constants, value_types, func);
                    sink.i32_wrap_i64();
                    emit_push_site(&mut sink, &site_gcmap, op_idx);
                    sink.call_indirect(0, base + nargs as u32);
                    // COND_CALL_N ignores the historical dummy word.
                    sink.drop();
                } else if let (Some(base), Some(nargs)) = (
                    true_void_residual_type_base,
                    conditional_call_true_void_arity(op, constants),
                ) {
                    for arg in call_args {
                        emit_resolve(&mut sink, constants, value_types, arg.to_opref());
                    }
                    emit_resolve(&mut sink, constants, value_types, func);
                    sink.i32_wrap_i64();
                    emit_push_site(&mut sink, &site_gcmap, op_idx);
                    sink.call_indirect(0, base + nargs as u32);
                } else if let Some((sig, &type_idx)) = conditional_call_typed_sig(op, constants)
                    .and_then(|sig| {
                        typed_residual_type_indices
                            .get(&sig)
                            .map(|type_idx| (sig, type_idx))
                    })
                {
                    let (params, result_ty) = &sig;
                    emit_typed_residual_call(
                        &mut sink,
                        constants,
                        value_types,
                        call_args,
                        params,
                        func,
                        &site_gcmap,
                        op_idx,
                        type_idx,
                    );
                    if result_ty.is_some() {
                        sink.drop();
                    }
                } else if op.getdescr().is_some()
                    && !cfg!(all(feature = "web", target_arch = "wasm32"))
                {
                    // Browser `jit_glue.js` (`web` on wasm32) reads every
                    // argument as a low i32 and writes an i32 result.
                    // `wasm-host` keeps i32/i64/f32/f64, so it still calls.
                    let jit_call = jit_call_idx.expect("COND_CALL needs jit_call");
                    let arg_refs: Vec<OpRef> = call_args.iter().map(|arg| arg.to_opref()).collect();
                    emit_residual_trampoline_call(
                        &mut sink,
                        constants,
                        value_types,
                        jit_call,
                        func,
                        &arg_refs,
                        &site_gcmap,
                        op_idx,
                        op,
                        None,
                    )?;
                } else if cfg!(all(feature = "web", target_arch = "wasm32")) {
                    return Err(BackendError::Unsupported(
                        "wasm codegen: COND_CALL has no web trampoline that preserves its signature"
                            .into(),
                    ));
                } else {
                    return Err(BackendError::Unsupported(
                        "wasm codegen: COND_CALL has no direct residual signature".into(),
                    ));
                }
                // COND_CALL sits inside the CALL opcode range, so a Ref living
                // across it already owns a home. Only the arm that called can
                // have collected, so the reload belongs on it: after the `if`
                // it would run on the untaken arm too, re-reading slots that
                // nothing moved, once per iteration of a loop whose whole
                // reason for a COND_CALL is that the call is rare.
                if call_can_collect(op) {
                    emit_reload_frame_if_necessary(
                        &mut sink,
                        residual_type_base,
                        ca.ca_reload_fn_ptr,
                        ca.jf_top_addr,
                    );
                    emit_reload_refs_from_homes(
                        &mut sink,
                        value_types,
                        site_homes.get(op_idx).map(Vec::as_slice).unwrap_or(&[]),
                        None,
                        frame,
                        &site_gcmap,
                        op_idx,
                    );
                }
                sink.end();
            }
            OpCode::CondCallValueI | OpCode::CondCallValueR => {
                // x86/regalloc.py `consider_cond_call`, COND_CALL_VALUE arm:
                // "Calls the function when args[0] is equal to 0 or NULL.
                // Returns the result from the function call if done, or args[0]
                // if it was not 0/NULL." Upstream forces the result into
                // args[0]'s register and lets the call overwrite it; the result
                // local plays that register's part here.
                //
                // The operand roles are COND_CALL's: predicate, callee, then the
                // call's own arguments. Sharing the generic CALL arm read args[0]
                // as the callee and args[1] as the first argument, and called
                // unconditionally.
                //
                // `do_conditional_call` asserts the callee forces no virtual or
                // virtualizable, so unlike the CALL arm this needs no force
                // bracket.
                let vi = op.pos().get().raw();
                let has_result = !OpRef::raw_is_constant(vi);
                let cond = op.arg(0).to_opref();
                let func = op.arg(1).to_opref();
                let call_args = &op.getarglist()[2..];
                let cond_on_stack = fused_condcall_at == Some(op_idx);
                if cond_on_stack {
                    fused_condcall_at = None;
                }

                if cond_on_stack {
                    // Comparison left a 0/1 i32. CondCallValue calls on zero
                    // and otherwise returns that predicate.
                    sink.i64_extend_i32_u();
                    if has_result {
                        sink.local_tee(value_types.local(vi));
                    }
                    sink.i64_eqz();
                } else {
                    if has_result {
                        emit_resolve(&mut sink, constants, value_types, cond);
                        sink.local_set(value_types.local(vi));
                    }
                    // The predicate is a full word: `i32.wrap_i64` would read a
                    // value whose only set bits are above 32 as NULL and call on a
                    // live one, so the test has to be `i64.eqz`.
                    emit_resolve(&mut sink, constants, value_types, cond);
                    sink.i64_eqz();
                }
                sink.if_(BlockType::Empty);
                if let (Some(base), Some(nargs)) = (
                    residual_type_base,
                    conditional_call_i64_arity(op, constants),
                ) {
                    for arg in call_args {
                        emit_resolve(&mut sink, constants, value_types, arg.to_opref());
                    }
                    emit_resolve(&mut sink, constants, value_types, func);
                    sink.i32_wrap_i64();
                    emit_push_site(&mut sink, &site_gcmap, op_idx);
                    sink.call_indirect(0, base + nargs as u32);
                    if has_result {
                        sink.local_set(value_types.local(vi));
                    } else {
                        sink.drop();
                    }
                } else if let Some((sig, &type_idx)) = conditional_call_typed_sig(op, constants)
                    .and_then(|sig| {
                        typed_residual_type_indices
                            .get(&sig)
                            .map(|type_idx| (sig, type_idx))
                    })
                {
                    let (params, result_ty) = &sig;
                    emit_typed_residual_call(
                        &mut sink,
                        constants,
                        value_types,
                        call_args,
                        params,
                        func,
                        &site_gcmap,
                        op_idx,
                        type_idx,
                    );
                    if has_result {
                        widen_direct_call_result(&mut sink, op, *result_ty);
                        sink.local_set(value_types.local(vi));
                    } else if result_ty.is_some() {
                        sink.drop();
                    }
                } else if op.getdescr().is_some()
                    && !cfg!(all(feature = "web", target_arch = "wasm32"))
                {
                    let jit_call = jit_call_idx.expect("COND_CALL_VALUE needs jit_call");
                    let arg_refs: Vec<OpRef> = call_args.iter().map(|arg| arg.to_opref()).collect();
                    emit_residual_trampoline_call(
                        &mut sink,
                        constants,
                        value_types,
                        jit_call,
                        func,
                        &arg_refs,
                        &site_gcmap,
                        op_idx,
                        op,
                        has_result.then_some(vi),
                    )?;
                } else if cfg!(all(feature = "web", target_arch = "wasm32")) {
                    return Err(BackendError::Unsupported(
                        "wasm codegen: COND_CALL_VALUE has no web trampoline that preserves its signature"
                            .into(),
                    ));
                } else {
                    return Err(BackendError::Unsupported(
                        "wasm codegen: COND_CALL_VALUE has no direct residual signature".into(),
                    ));
                }
                // Only the arm that called can have collected, so the reload
                // sits on it rather than after the `if`.
                if call_can_collect(op) {
                    emit_reload_frame_if_necessary(
                        &mut sink,
                        residual_type_base,
                        ca.ca_reload_fn_ptr,
                        ca.jf_top_addr,
                    );
                    emit_reload_refs_from_homes(
                        &mut sink,
                        value_types,
                        site_homes.get(op_idx).map(Vec::as_slice).unwrap_or(&[]),
                        has_result.then_some(vi),
                        frame,
                        &site_gcmap,
                        op_idx,
                    );
                }
                sink.end();
            }

            // x86/assembler.py genop_guard_guard_gc_type:
            // GUARD_GC_TYPE: args[0] = object ref, args[1] = expected
            // type_id. The majit runtime stores the typeid in the GC
            // header word placed immediately before the object payload
            // (`majit_gc::header::GcHeader::tid_and_flags`, lower
            // `TYPE_ID_BITS`). The cranelift backend lowers the same op this way
            // (compiler.rs GuardGcType branch). This is NOT the RPython
            // gcremovetypeptr layout — pyre's GC keeps the typeid in the
            // header, not at `obj[0]`.
            OpCode::GuardGcType => {
                let _ = classptr_to_typeid; // typeid is already an immediate
                if op.num_args() >= 2 {
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                    // header address = obj - GcHeader::SIZE
                    sink.i64_const(GcHeader::SIZE as i64);
                    sink.i64_sub();
                    sink.i32_wrap_i64();
                    // Load the physical header prefix. On wasm32 its upper
                    // four bytes are ABI padding; TYPE_ID_MASK selects the
                    // 16-bit type-id half of the logical RPython word.
                    sink.i64_load(mem64(0));
                    // Mask lower TYPE_ID_BITS to extract the type id
                    sink.i64_const(TYPE_ID_MASK as i64);
                    sink.i64_and();
                    // Compare against expected_typeid (arg1 — already an
                    // i64 in the constant pool or a frame slot).
                    emit_resolve(&mut sink, constants, value_types, op.arg(1).to_opref());
                    sink.i64_ne();
                    emit_guard_if_exit(
                        &mut sink,
                        constants,
                        value_types,
                        guard_idx,
                        op,
                        block_exit_depth,
                        guard_dispatch,
                    );
                }
                guard_idx += 1;
            }
            // x86/assembler.py genop_guard_guard_is_object.
            //     assert self.cpu.supports_guard_gc_type
            //     [loc_object, loc_typeid] = locs
            //     if IS_X86_32:
            //         self.mc.MOVZX16(loc_typeid, mem(loc_object, 0))
            //     else:
            //         self.mc.MOV32(loc_typeid, mem(loc_object, 0))
            //     base_type_info, shift_by, sizeof_ti = (
            //         self.cpu.gc_ll_descr
            //             .get_translated_info_for_typeinfo())
            //     infobits_offset, IS_OBJECT_FLAG = (
            //         self.cpu.gc_ll_descr
            //             .get_translated_info_for_guard_is_object())
            //     loc_infobits = addr_add(imm(base_type_info),
            //                             loc_typeid,
            //                             scale=shift_by,
            //                             offset=infobits_offset)
            //     self.mc.TEST8(loc_infobits, imm(IS_OBJECT_FLAG))
            //     self.guard_success_cc = rx86.Conditions['NZ']
            //     self.implement_guard(guard_token)
            OpCode::GuardIsObject => {
                // assembler.py:1925 assert self.cpu.supports_guard_gc_type
                assert!(
                    guard_gc_type_info.supports_guard_gc_type,
                    "x86/assembler.py:1925: assert self.cpu.\
                     supports_guard_gc_type (GcAllocator has not \
                     installed a TYPE_INFO layout)"
                );
                // assembler.py MOV32 loc_typeid, mem(loc_object, 0).
                // majit's GC header sits at obj - GcHeader::SIZE; the
                // typeid occupies the lower TYPE_ID_BITS of that word.
                emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                sink.i64_const(GcHeader::SIZE as i64);
                sink.i64_sub();
                sink.i32_wrap_i64();
                sink.i64_load(mem64(0));
                sink.i64_const(TYPE_ID_MASK as i64);
                sink.i64_and();
                // Stack: [..., loc_typeid]

                // assembler.py:1938-1939 addr_add(imm(base_type_info),
                //     loc_typeid, scale=shift_by, offset=infobits_offset)
                if guard_gc_type_info.shift_by > 0 {
                    sink.i64_const(guard_gc_type_info.shift_by as i64);
                    sink.i64_shl();
                }
                sink.i64_const(guard_gc_type_info.base_type_info as i64);
                sink.i64_add();
                sink.i32_wrap_i64();
                // Stack: [..., loc_type_info(i32 addr)]

                // assembler.py:1940 TEST8 [loc_infobits], IS_OBJECT_FLAG. The
                // `offset=infobits_offset` of the address computation above is
                // a constant, so it rides in the load's own MemArg.
                sink.i32_load8_u(memarg(guard_gc_type_info.infobits_offset as u64, 0));
                sink.i32_const(guard_gc_type_info.is_object_flag as i32);
                sink.i32_and();
                // assembler.py:1942 guard_success_cc = Conditions['NZ']:
                // guard passes when byte & flag != 0; fail when == 0.
                sink.i32_eqz();
                emit_guard_if_exit(
                    &mut sink,
                    constants,
                    value_types,
                    guard_idx,
                    op,
                    block_exit_depth,
                    guard_dispatch,
                );
                guard_idx += 1;
            }
            // x86/assembler.py genop_guard_guard_subclass.
            //     assert self.cpu.supports_guard_gc_type
            //     [loc_object, loc_check_against_class, loc_tmp] = locs
            //     offset = self.cpu.vtable_offset
            //     offset2 = self.cpu.subclassrange_min_offset
            //     if offset is not None:
            //         self.mc.MOV_rm(loc_tmp, (loc_object, offset))
            //         self.mc.MOV_rm(loc_tmp, (loc_tmp, offset2))
            //     else:
            //         self.mc.MOV32(loc_tmp, mem(loc_object, 0))
            //         base_type_info, shift_by, sizeof_ti = (
            //             gc_ll_descr.get_translated_info_for_typeinfo())
            //         self.mc.MOV(loc_tmp, addr_add(
            //             imm(base_type_info), loc_tmp,
            //             scale=shift_by,
            //             offset=sizeof_ti + offset2))
            //     vtable_ptr = loc_check_against_class.getint()
            //     vtable_ptr = rffi.cast(rclass.CLASSTYPE, vtable_ptr)
            //     check_min = vtable_ptr.subclassrange_min
            //     check_max = vtable_ptr.subclassrange_max
            //     self.mc.SUB_ri(loc_tmp, check_min)
            //     self.mc.CMP_ri(loc_tmp, check_max - check_min)
            //     self.guard_success_cc = Conditions['B']
            //     self.implement_guard(guard_token)
            OpCode::GuardSubclass => {
                // assembler.py:1946 assert self.cpu.supports_guard_gc_type
                assert!(
                    guard_gc_type_info.supports_guard_gc_type,
                    "x86/assembler.py:1946: assert self.cpu.\
                     supports_guard_gc_type (GcAllocator has not \
                     installed a TYPE_INFO / rclass.CLASSTYPE layout)"
                );

                // assembler.py:1971 vtable_ptr = loc_check_against_class
                //   .getint(): the bounds are resolved at codegen time,
                //   so arg1 must be an immediate class pointer.
                let class_arg = op.arg(1).to_opref();
                // history.py — inline-Const carries its class pointer directly.
                let loc_check_against_class = class_arg.const_int_value().unwrap_or_else(|| {
                    panic!(
                        "x86/assembler.py:1971 vtable_ptr = \
                             loc_check_against_class.getint(): \
                             GUARD_SUBCLASS requires arg1 to be a \
                             ConstInt immediate class pointer"
                    )
                });
                // assembler.py:1973-1974: vtable_ptr.subclassrange_{min,max}
                let (check_min, check_max) = guard_gc_type_info
                    .subclass_ranges
                    .get(&loc_check_against_class)
                    .copied()
                    .unwrap_or((0, 0));

                // assembler.py:1950-1951 offset / offset2.
                let offset2 = guard_gc_type_info.subclassrange_min_offset;
                if let Some(vtable_off) = vtable_offset {
                    // assembler.py:1953-1956
                    //     MOV_rm(loc_tmp, (loc_object, offset))
                    //     MOV_rm(loc_tmp, (loc_tmp, offset2))
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                    sink.i32_wrap_i64();
                    sink.i64_load(mem64(vtable_off as u64));
                    sink.i32_wrap_i64();
                    // subclassrange_min is an 8-byte i64 on every target
                    // (pyobject.rs `PyType::subclassrange_min: AtomicI64`); read
                    // the full field width, not the wasm32 4-byte `usize`, or the
                    // guard truncates/sign-extends the object's min.
                    emit_sized_int_load(
                        &mut sink,
                        offset2 as u64,
                        std::mem::size_of::<i64>(),
                        true,
                    );
                } else {
                    // assembler.py:1957-1969 gcremovetypeptr path.
                    //     MOV32 loc_tmp, mem(loc_object, 0)
                    //     base_type_info, shift_by, sizeof_ti = ...
                    //     MOV loc_tmp, [base_type_info
                    //         + (loc_tmp << shift_by)
                    //         + sizeof_ti + offset2]
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                    sink.i64_const(GcHeader::SIZE as i64);
                    sink.i64_sub();
                    sink.i32_wrap_i64();
                    sink.i64_load(mem64(0));
                    sink.i64_const(TYPE_ID_MASK as i64);
                    sink.i64_and();
                    if guard_gc_type_info.shift_by > 0 {
                        sink.i64_const(guard_gc_type_info.shift_by as i64);
                        sink.i64_shl();
                    }
                    sink.i64_const(guard_gc_type_info.base_type_info as i64);
                    sink.i64_add();
                    sink.i64_const((guard_gc_type_info.sizeof_ti + offset2) as i64);
                    sink.i64_add();
                    sink.i32_wrap_i64();
                    // 8-byte i64 subclassrange_min (see the vtable path above).
                    emit_sized_int_load(&mut sink, 0, std::mem::size_of::<i64>(), true);
                }
                // Stack: [..., loc_tmp (i64)]

                // assembler.py:1976-1978 unsigned comparison:
                //     (loc_tmp - check_min) <u (check_max - check_min)
                sink.i64_const(check_min);
                sink.i64_sub();
                sink.i64_const(check_max - check_min);
                // assembler.py:1979 guard_success_cc = Conditions['B']:
                // guard passes when sub <u limit; fail when sub >=u limit.
                sink.i64_ge_u();
                emit_guard_if_exit(
                    &mut sink,
                    constants,
                    value_types,
                    guard_idx,
                    op,
                    block_exit_depth,
                    guard_dispatch,
                );
                guard_idx += 1;
            }
            // `reached_loop_header` mints this op only to donate its
            // `rd_resume_position` to the guards `jump_to_existing_trace` and
            // `inline_short_preamble` stamp; both `optimize_GUARD_FUTURE_CONDITION`
            // arms then consume it into `patchguardop` and emit nothing, which is
            // why nothing under `rpython/jit/backend` lowers it. Reaching a
            // backend means the optimizer did not consume it, and there is
            // nothing correct to lower it to: it is nullary, so there is no
            // condition to test, and an exit publishing neither `frame[0]` nor
            // the fail args leaves the resume reading whatever the frame last
            // held.
            OpCode::GuardFutureCondition => {
                return Err(BackendError::Unsupported(
                    "wasm backend: GuardFutureCondition is unsupported (the optimizer \
                     consumes it into patchguardop and no backend lowers it); \
                     declining the trace"
                        .to_string(),
                ));
            }

            // ── Quasi-immutable / record / assert ──
            OpCode::QuasiimmutField
            | OpCode::RecordExactClass
            | OpCode::RecordExactValueI
            | OpCode::RecordExactValueR
            | OpCode::RecordKnownResult
            | OpCode::AssertNotNone => {
                // Metadata-only ops, no codegen needed. `RecordKnownResult`
                // is an optimizer hint with no backend arm upstream at all —
                // `optimize_RECORD_KNOWN_RESULT` consumes it and simplify
                // drops it — so one that still reaches here owes no code.
            }

            OpCode::Newstr | OpCode::Newunicode => {
                let vi = op.pos().get().raw();
                let Some(base) = residual_type_base else {
                    return Err(BackendError::Unsupported(
                        "wasm codegen: Newstr/Newunicode needs a residual alloc helper".into(),
                    ));
                };
                let descr = op.getdescr().ok_or_else(|| {
                    BackendError::Unsupported(
                        "wasm codegen: Newstr/Newunicode is missing its ArrayDescr".into(),
                    )
                })?;
                let ad = descr.as_array_descr().ok_or_else(|| {
                    BackendError::Unsupported(
                        "wasm codegen: Newstr/Newunicode descr is not an ArrayDescr".into(),
                    )
                })?;
                let len_offset = ad.len_descr().map_or(0i64, |ld| ld.offset() as i64);
                sink.i64_const(ad.type_id() as i64);
                sink.i64_const(ad.base_size() as i64);
                sink.i64_const(ad.item_size() as i64);
                emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                sink.i64_const(len_offset);
                sink.i32_const(alloc.new_array_fn_ptr as i32);
                emit_push_site(&mut sink, &site_gcmap, op_idx);
                sink.call_indirect(0, base + 5);
                if !OpRef::raw_is_constant(vi) {
                    sink.local_set(value_types.local(vi));
                } else {
                    sink.drop();
                }
                emit_memory_error_check(
                    &mut sink,
                    constants,
                    value_types,
                    op.pos().get(),
                    residual_type_base,
                    ca.ca_reload_fn_ptr,
                    ca.jf_top_addr,
                    ca.attached.propagate_exception_descr,
                );
                // rewrite.py `clear_varsize_gc_fields` FLAG_STR / FLAG_UNICODE:
                // `emit_setfield(result, 0, descr=hash_descr)`. Both layouts
                // keep `hash` at offset 0 (`rewrite.rs clear_varsize_gc_fields`).
                if !OpRef::raw_is_constant(vi) {
                    sink.local_get(value_types.local(vi));
                    sink.i32_wrap_i64();
                    sink.i32_const(0);
                    sink.i32_store(MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    });
                }
                let skip = (!OpRef::raw_is_constant(vi)).then_some(vi);
                emit_reload_frame_if_necessary(
                    &mut sink,
                    residual_type_base,
                    ca.ca_reload_fn_ptr,
                    ca.jf_top_addr,
                );
                emit_reload_refs_from_homes(
                    &mut sink,
                    value_types,
                    site_homes.get(op_idx).map(Vec::as_slice).unwrap_or(&[]),
                    skip,
                    frame,
                    &site_gcmap,
                    op_idx,
                );
            }
            // ── Misc ops ──
            OpCode::NurseryPtrIncrement => {
                let vi = op.pos().get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                    emit_resolve(&mut sink, constants, value_types, op.arg(1).to_opref());
                    sink.i64_add();
                    sink.local_set(value_types.local(vi));
                }
            }
            // rewrite.py `CALL_MALLOC_NURSERY(ConstInt(size))`: size is the
            // already-rounded header+payload total. Fast path is malloc_cond
            // (bump, write the physical header word, return free+HDR). Slow
            // path is `wasm_jit_alloc(0, payload)`. When the following
            // `gen_initialize_tid` store is the next op, the fast path writes
            // the tid in the same physical header store (flags and padding
            // stay zero) and the tid store runs only on this slow arm.
            OpCode::CallMallocNursery => {
                let vi = op.pos().get().raw();
                let size_const = const_operand_value(constants, op.arg(0).to_opref());
                let bump_size = size_const.and_then(|size| u32::try_from(size).ok());
                let payload =
                    bump_size.map(|size| i64::from(size.saturating_sub(GcHeader::SIZE as u32)));
                let Some(base) = residual_type_base else {
                    return Err(BackendError::Unsupported(
                        "wasm codegen: CallMallocNursery needs a residual alloc helper".into(),
                    ));
                };
                let inlined = matches!((nursery, bump_size, payload), (Some(_), Some(_), Some(_)));
                if let (Some(na), Some(bump_size), Some(payload)) = (nursery, bump_size, payload) {
                    let header_tid = if !OpRef::raw_is_constant(vi) {
                        ops.get(op_idx + 1).and_then(|next| {
                            nursery_header_tid_store(next, op.pos().get(), constants)
                        })
                    } else {
                        None
                    };
                    let header_word = header_tid.as_ref().map(|s| s.tid).unwrap_or(0);
                    if header_tid.is_some() {
                        skip_nursery_tid_store_at = Some(op_idx + 1);
                    }
                    sink.i32_const(na.free_addr as i32);
                    sink.i32_load(MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    });
                    sink.local_tee(alloc_scratch_local);
                    sink.i32_const(bump_size as i32);
                    sink.i32_add();
                    sink.local_tee(alloc_size_local);
                    sink.i32_const(na.top_addr as i32);
                    sink.i32_load(MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    });
                    sink.i32_gt_u();
                    sink.if_(BlockType::Result(ValType::I64));
                    sink.i64_const(0);
                    sink.i64_const(payload);
                    sink.i32_const(alloc.new_fn_ptr as i32);
                    emit_push_site(&mut sink, &site_gcmap, op_idx);
                    sink.call_indirect(0, base + 2);
                    emit_reload_frame_if_necessary(
                        &mut sink,
                        residual_type_base,
                        ca.ca_reload_fn_ptr,
                        ca.jf_top_addr,
                    );
                    emit_reload_refs_from_homes(
                        &mut sink,
                        value_types,
                        site_homes.get(op_idx).map(Vec::as_slice).unwrap_or(&[]),
                        (!OpRef::raw_is_constant(vi)).then_some(vi),
                        frame,
                        &site_gcmap,
                        op_idx,
                    );
                    if let Some(tid_store) = header_tid {
                        sink.local_tee(value_types.local(vi));
                        sink.i64_eqz();
                        sink.if_(BlockType::Empty);
                        sink.else_();
                        emit_resolve(&mut sink, constants, value_types, op.pos().get());
                        sink.i32_wrap_i64();
                        sink.i32_const(tid_store.offset as i32);
                        sink.i32_add();
                        sink.i64_const(tid_store.tid);
                        emit_sized_int_store(&mut sink, 0, tid_store.width);
                        sink.end();
                        sink.local_get(value_types.local(vi));
                    }
                    sink.else_();
                    sink.i32_const(na.free_addr as i32);
                    sink.local_get(alloc_size_local);
                    sink.i32_store(MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    });
                    // Physical header word. Tid is the low half when the
                    // following tid store was folded in; otherwise zero.
                    // Payload stays dirty (`malloc_zero_filled = False`).
                    sink.local_get(alloc_scratch_local);
                    sink.i64_const(header_word);
                    sink.i64_store(MemArg {
                        offset: 0,
                        align: 3,
                        memory_index: 0,
                    });
                    sink.local_get(alloc_scratch_local);
                    sink.i32_const(GcHeader::SIZE as i32);
                    sink.i32_add();
                    sink.i64_extend_i32_u();
                    sink.end();
                } else {
                    sink.i64_const(0);
                    if let Some(payload) = payload {
                        sink.i64_const(payload);
                    } else {
                        emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                        sink.i64_const(GcHeader::SIZE as i64);
                        sink.i64_sub();
                    }
                    sink.i32_const(alloc.new_fn_ptr as i32);
                    emit_push_site(&mut sink, &site_gcmap, op_idx);
                    sink.call_indirect(0, base + 2);
                }
                if !OpRef::raw_is_constant(vi) {
                    sink.local_set(value_types.local(vi));
                } else {
                    sink.drop();
                }
                emit_memory_error_check(
                    &mut sink,
                    constants,
                    value_types,
                    op.pos().get(),
                    residual_type_base,
                    ca.ca_reload_fn_ptr,
                    ca.jf_top_addr,
                    ca.attached.propagate_exception_descr,
                );
                if !inlined {
                    let skip = (!OpRef::raw_is_constant(vi)).then_some(vi);
                    emit_reload_frame_if_necessary(
                        &mut sink,
                        residual_type_base,
                        ca.ca_reload_fn_ptr,
                        ca.jf_top_addr,
                    );
                    emit_reload_refs_from_homes(
                        &mut sink,
                        value_types,
                        site_homes.get(op_idx).map(Vec::as_slice).unwrap_or(&[]),
                        skip,
                        frame,
                        &site_gcmap,
                        op_idx,
                    );
                }
                // `wasm_jit_alloc` can return old-gen when the nursery cannot
                // hold the request.
            }
            OpCode::CallMallocNurseryHeaderless => {
                let vi = op.pos().get().raw();
                let size_const = const_operand_value(constants, op.arg(0).to_opref());
                let bump_size = size_const.and_then(aligned_varsize_frame_bump);
                let Some(base) = residual_type_base else {
                    return Err(BackendError::Unsupported(
                        "wasm codegen: CallMallocNurseryHeaderless needs a residual alloc helper"
                            .into(),
                    ));
                };
                if alloc.headerless_fn_ptr == 0 {
                    return Err(BackendError::Unsupported(
                        "wasm codegen: CallMallocNurseryHeaderless has no headerless helper".into(),
                    ));
                }
                let inlined = matches!((nursery, bump_size), (Some(_), Some(_)));
                if let (Some(na), Some(bump_size)) = (nursery, bump_size) {
                    sink.i32_const(na.free_addr as i32);
                    sink.i32_load(MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    });
                    sink.local_tee(alloc_scratch_local);
                    sink.i32_const(bump_size as i32);
                    sink.i32_add();
                    sink.local_tee(alloc_size_local);
                    sink.i32_const(na.top_addr as i32);
                    sink.i32_load(MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    });
                    sink.i32_gt_u();
                    sink.if_(BlockType::Result(ValType::I64));
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                    sink.i32_const(alloc.headerless_fn_ptr as i32);
                    emit_push_site(&mut sink, &site_gcmap, op_idx);
                    sink.call_indirect(0, base + 1);
                    emit_reload_frame_if_necessary(
                        &mut sink,
                        residual_type_base,
                        ca.ca_reload_fn_ptr,
                        ca.jf_top_addr,
                    );
                    emit_reload_refs_from_homes(
                        &mut sink,
                        value_types,
                        site_homes.get(op_idx).map(Vec::as_slice).unwrap_or(&[]),
                        (!OpRef::raw_is_constant(vi)).then_some(vi),
                        frame,
                        &site_gcmap,
                        op_idx,
                    );
                    sink.else_();
                    sink.i32_const(na.free_addr as i32);
                    sink.local_get(alloc_size_local);
                    sink.i32_store(MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    });
                    sink.local_get(alloc_scratch_local);
                    sink.i64_extend_i32_u();
                    sink.end();
                } else {
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                    sink.i32_const(alloc.headerless_fn_ptr as i32);
                    emit_push_site(&mut sink, &site_gcmap, op_idx);
                    sink.call_indirect(0, base + 1);
                }
                if !OpRef::raw_is_constant(vi) {
                    sink.local_set(value_types.local(vi));
                } else {
                    sink.drop();
                }
                emit_memory_error_check(
                    &mut sink,
                    constants,
                    value_types,
                    op.pos().get(),
                    residual_type_base,
                    ca.ca_reload_fn_ptr,
                    ca.jf_top_addr,
                    ca.attached.propagate_exception_descr,
                );
                if !inlined {
                    let skip = (!OpRef::raw_is_constant(vi)).then_some(vi);
                    emit_reload_frame_if_necessary(
                        &mut sink,
                        residual_type_base,
                        ca.ca_reload_fn_ptr,
                        ca.jf_top_addr,
                    );
                    emit_reload_refs_from_homes(
                        &mut sink,
                        value_types,
                        site_homes.get(op_idx).map(Vec::as_slice).unwrap_or(&[]),
                        skip,
                        frame,
                        &site_gcmap,
                        op_idx,
                    );
                }
            }
            // Wasm does not install a headerless allocator.
            OpCode::CallMallocNurseryVarsizeHeaderless => {
                return Err(BackendError::Unsupported(format!(
                    "wasm codegen: unhandled opcode {:?}",
                    op.opcode
                )));
            }
            OpCode::CallMallocNurseryVarsize => {
                // x86 `malloc_cond_varsize`: bump `nursery_free` by the
                // 8-aligned `length * itemsize + basesize + header` when the
                // length is young-sized, write tid, and return the payload.
                // Oversize / a full nursery falls through to the arity-5
                // array helper (which may return old-gen).
                let vi = op.pos().get().raw();
                let Some(base) = residual_type_base else {
                    return Err(BackendError::Unsupported(
                        "wasm codegen: CallMallocNurseryVarsize needs a residual alloc helper"
                            .into(),
                    ));
                };
                let descr = op.getdescr().ok_or_else(|| {
                    BackendError::Unsupported(
                        "wasm codegen: CallMallocNurseryVarsize is missing its ArrayDescr".into(),
                    )
                })?;
                let ad = descr.as_array_descr().ok_or_else(|| {
                    BackendError::Unsupported(
                        "wasm codegen: CallMallocNurseryVarsize descr is not an ArrayDescr".into(),
                    )
                })?;
                let len_offset = ad.len_descr().map_or(0i64, |ld| ld.offset() as i64);
                let type_id = ad.type_id() as i64;
                let base_size = ad.base_size() as i64;
                let itemsize = op.arg(1).to_opref();
                let length = op.arg(2).to_opref();
                let header = GcHeader::SIZE as i64;
                let word = std::mem::size_of::<usize>() as i64;
                let inlined = nursery.is_some();
                if let Some(na) = nursery {
                    let max_length = na.large_threshold.saturating_sub(2 * word as usize) as i64;
                    emit_resolve(&mut sink, constants, value_types, length);
                    sink.i64_const(max_length);
                    sink.i64_gt_u();
                    sink.if_(BlockType::Result(ValType::I64));
                    emit_alloc_array_helper(
                        &mut sink,
                        constants,
                        value_types,
                        type_id,
                        base_size,
                        itemsize,
                        length,
                        len_offset,
                        alloc.new_array_fn_ptr,
                        base,
                        &site_gcmap,
                        op_idx,
                    );
                    emit_reload_frame_if_necessary(
                        &mut sink,
                        residual_type_base,
                        ca.ca_reload_fn_ptr,
                        ca.jf_top_addr,
                    );
                    emit_reload_refs_from_homes(
                        &mut sink,
                        value_types,
                        site_homes.get(op_idx).map(Vec::as_slice).unwrap_or(&[]),
                        (!OpRef::raw_is_constant(vi)).then_some(vi),
                        frame,
                        &site_gcmap,
                        op_idx,
                    );
                    sink.else_();
                    sink.i32_const(na.free_addr as i32);
                    sink.i32_load(MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    });
                    sink.local_tee(alloc_scratch_local);
                    emit_resolve(&mut sink, constants, value_types, length);
                    emit_resolve(&mut sink, constants, value_types, itemsize);
                    sink.i64_mul();
                    sink.i64_const(base_size + header + 7);
                    sink.i64_add();
                    sink.i64_const(!7i64);
                    sink.i64_and();
                    sink.i32_wrap_i64();
                    sink.i32_add();
                    sink.local_tee(alloc_size_local);
                    sink.i32_const(na.top_addr as i32);
                    sink.i32_load(MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    });
                    sink.i32_gt_u();
                    sink.if_(BlockType::Result(ValType::I64));
                    emit_alloc_array_helper(
                        &mut sink,
                        constants,
                        value_types,
                        type_id,
                        base_size,
                        itemsize,
                        length,
                        len_offset,
                        alloc.new_array_fn_ptr,
                        base,
                        &site_gcmap,
                        op_idx,
                    );
                    emit_reload_frame_if_necessary(
                        &mut sink,
                        residual_type_base,
                        ca.ca_reload_fn_ptr,
                        ca.jf_top_addr,
                    );
                    emit_reload_refs_from_homes(
                        &mut sink,
                        value_types,
                        site_homes.get(op_idx).map(Vec::as_slice).unwrap_or(&[]),
                        (!OpRef::raw_is_constant(vi)).then_some(vi),
                        frame,
                        &site_gcmap,
                        op_idx,
                    );
                    sink.else_();
                    sink.i32_const(na.free_addr as i32);
                    sink.local_get(alloc_size_local);
                    sink.i32_store(MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    });
                    // Fast path writes tid (`malloc_cond_varsize`); rewrite
                    // does not emit `gen_initialize_tid` after a varsize bump.
                    sink.local_get(alloc_scratch_local);
                    sink.i64_const(type_id);
                    sink.i64_store(MemArg {
                        offset: 0,
                        align: 3,
                        memory_index: 0,
                    });
                    sink.local_get(alloc_scratch_local);
                    sink.i32_const(GcHeader::SIZE as i32);
                    sink.i32_add();
                    sink.i64_extend_i32_u();
                    sink.end();
                    sink.end();
                } else {
                    emit_alloc_array_helper(
                        &mut sink,
                        constants,
                        value_types,
                        type_id,
                        base_size,
                        itemsize,
                        length,
                        len_offset,
                        alloc.new_array_fn_ptr,
                        base,
                        &site_gcmap,
                        op_idx,
                    );
                }
                if !OpRef::raw_is_constant(vi) {
                    sink.local_set(value_types.local(vi));
                } else {
                    sink.drop();
                }
                emit_memory_error_check(
                    &mut sink,
                    constants,
                    value_types,
                    op.pos().get(),
                    residual_type_base,
                    ca.ca_reload_fn_ptr,
                    ca.jf_top_addr,
                    ca.attached.propagate_exception_descr,
                );
                if !inlined {
                    let skip = (!OpRef::raw_is_constant(vi)).then_some(vi);
                    emit_reload_frame_if_necessary(
                        &mut sink,
                        residual_type_base,
                        ca.ca_reload_fn_ptr,
                        ca.jf_top_addr,
                    );
                    emit_reload_refs_from_homes(
                        &mut sink,
                        value_types,
                        site_homes.get(op_idx).map(Vec::as_slice).unwrap_or(&[]),
                        skip,
                        frame,
                        &site_gcmap,
                        op_idx,
                    );
                }
            }
            OpCode::CallMallocNurseryVarsizeFrame => {
                let vi = op.pos().get().raw();
                let size_const = const_operand_value(constants, op.arg(0).to_opref());
                let bump_size = size_const.and_then(aligned_varsize_frame_bump);
                let payload =
                    bump_size.map(|size| i64::from(size.saturating_sub(GcHeader::SIZE as u32)));
                let Some(base) = residual_type_base else {
                    return Err(BackendError::Unsupported(
                        "wasm codegen: CallMallocNurseryVarsizeFrame needs a residual alloc helper"
                            .into(),
                    ));
                };
                let inlined = nursery.is_some();
                if let (Some(na), Some(bump_size), Some(payload)) = (nursery, bump_size, payload) {
                    sink.i32_const(na.free_addr as i32);
                    sink.i32_load(MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    });
                    sink.local_tee(alloc_scratch_local);
                    sink.i32_const(bump_size as i32);
                    sink.i32_add();
                    sink.local_tee(alloc_size_local);
                    sink.i32_const(na.top_addr as i32);
                    sink.i32_load(MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    });
                    sink.i32_gt_u();
                    sink.if_(BlockType::Result(ValType::I64));
                    sink.i64_const(0);
                    sink.i64_const(payload);
                    sink.i32_const(alloc.new_fn_ptr as i32);
                    emit_push_site(&mut sink, &site_gcmap, op_idx);
                    sink.call_indirect(0, base + 2);
                    emit_reload_frame_if_necessary(
                        &mut sink,
                        residual_type_base,
                        ca.ca_reload_fn_ptr,
                        ca.jf_top_addr,
                    );
                    emit_reload_refs_from_homes(
                        &mut sink,
                        value_types,
                        site_homes.get(op_idx).map(Vec::as_slice).unwrap_or(&[]),
                        (!OpRef::raw_is_constant(vi)).then_some(vi),
                        frame,
                        &site_gcmap,
                        op_idx,
                    );
                    sink.else_();
                    sink.i32_const(na.free_addr as i32);
                    sink.local_get(alloc_size_local);
                    sink.i32_store(MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    });
                    // Header word only: `CallMallocNurseryVarsizeFrame`
                    // `mov QWORD [rcx], 0`. Payload stays dirty.
                    sink.local_get(alloc_scratch_local);
                    sink.i64_const(0);
                    sink.i64_store(MemArg {
                        offset: 0,
                        align: 3,
                        memory_index: 0,
                    });
                    sink.local_get(alloc_scratch_local);
                    sink.i32_const(GcHeader::SIZE as i32);
                    sink.i32_add();
                    sink.i64_extend_i32_u();
                    sink.end();
                } else if let Some(na) = nursery {
                    // `jfi_frame_size` is not a rewrite-time constant.
                    // `malloc_cond_varsize_frame` bumps that runtime total;
                    // a total at or above `large_threshold` takes the helper
                    // (`can_use_nursery_malloc`'s exclusive bound).
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                    sink.i64_const(7);
                    sink.i64_add();
                    sink.i64_const(!7i64);
                    sink.i64_and();
                    sink.i64_const(na.large_threshold as i64);
                    sink.i64_ge_u();
                    sink.if_(BlockType::Result(ValType::I64));
                    sink.i64_const(0);
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                    sink.i64_const(7);
                    sink.i64_add();
                    sink.i64_const(!7i64);
                    sink.i64_and();
                    sink.i64_const(GcHeader::SIZE as i64);
                    sink.i64_sub();
                    sink.i32_const(alloc.new_fn_ptr as i32);
                    emit_push_site(&mut sink, &site_gcmap, op_idx);
                    sink.call_indirect(0, base + 2);
                    emit_reload_frame_if_necessary(
                        &mut sink,
                        residual_type_base,
                        ca.ca_reload_fn_ptr,
                        ca.jf_top_addr,
                    );
                    emit_reload_refs_from_homes(
                        &mut sink,
                        value_types,
                        site_homes.get(op_idx).map(Vec::as_slice).unwrap_or(&[]),
                        (!OpRef::raw_is_constant(vi)).then_some(vi),
                        frame,
                        &site_gcmap,
                        op_idx,
                    );
                    sink.else_();
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                    sink.i64_const(7);
                    sink.i64_add();
                    sink.i64_const(!7i64);
                    sink.i64_and();
                    sink.i32_wrap_i64();
                    sink.local_set(alloc_size_local);
                    sink.i32_const(na.free_addr as i32);
                    sink.i32_load(MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    });
                    sink.local_tee(alloc_scratch_local);
                    sink.local_get(alloc_size_local);
                    sink.i32_add();
                    sink.i32_const(na.top_addr as i32);
                    sink.i32_load(MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    });
                    sink.i32_gt_u();
                    sink.if_(BlockType::Result(ValType::I64));
                    sink.i64_const(0);
                    sink.local_get(alloc_size_local);
                    sink.i64_extend_i32_u();
                    sink.i64_const(GcHeader::SIZE as i64);
                    sink.i64_sub();
                    sink.i32_const(alloc.new_fn_ptr as i32);
                    emit_push_site(&mut sink, &site_gcmap, op_idx);
                    sink.call_indirect(0, base + 2);
                    emit_reload_frame_if_necessary(
                        &mut sink,
                        residual_type_base,
                        ca.ca_reload_fn_ptr,
                        ca.jf_top_addr,
                    );
                    emit_reload_refs_from_homes(
                        &mut sink,
                        value_types,
                        site_homes.get(op_idx).map(Vec::as_slice).unwrap_or(&[]),
                        (!OpRef::raw_is_constant(vi)).then_some(vi),
                        frame,
                        &site_gcmap,
                        op_idx,
                    );
                    sink.else_();
                    sink.i32_const(na.free_addr as i32);
                    sink.local_get(alloc_scratch_local);
                    sink.local_get(alloc_size_local);
                    sink.i32_add();
                    sink.i32_store(MemArg {
                        offset: 0,
                        align: 2,
                        memory_index: 0,
                    });
                    sink.local_get(alloc_scratch_local);
                    sink.i64_const(0);
                    sink.i64_store(MemArg {
                        offset: 0,
                        align: 3,
                        memory_index: 0,
                    });
                    sink.local_get(alloc_scratch_local);
                    sink.i32_const(GcHeader::SIZE as i32);
                    sink.i32_add();
                    sink.i64_extend_i32_u();
                    sink.end();
                    sink.end();
                } else {
                    sink.i64_const(0);
                    if let Some(payload) = payload {
                        sink.i64_const(payload);
                    } else {
                        emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                        sink.i64_const(GcHeader::SIZE as i64);
                        sink.i64_sub();
                    }
                    sink.i32_const(alloc.new_fn_ptr as i32);
                    emit_push_site(&mut sink, &site_gcmap, op_idx);
                    sink.call_indirect(0, base + 2);
                }
                if !OpRef::raw_is_constant(vi) {
                    sink.local_set(value_types.local(vi));
                } else {
                    sink.drop();
                }
                emit_memory_error_check(
                    &mut sink,
                    constants,
                    value_types,
                    op.pos().get(),
                    residual_type_base,
                    ca.ca_reload_fn_ptr,
                    ca.jf_top_addr,
                    ca.attached.propagate_exception_descr,
                );
                if !inlined {
                    let skip = (!OpRef::raw_is_constant(vi)).then_some(vi);
                    emit_reload_frame_if_necessary(
                        &mut sink,
                        residual_type_base,
                        ca.ca_reload_fn_ptr,
                        ca.jf_top_addr,
                    );
                    emit_reload_refs_from_homes(
                        &mut sink,
                        value_types,
                        site_homes.get(op_idx).map(Vec::as_slice).unwrap_or(&[]),
                        skip,
                        frame,
                        &site_gcmap,
                        op_idx,
                    );
                }
                // `wasm_jit_alloc` can return old-gen when the nursery cannot
                // hold the frame.
            }
            // `GcRewriterImpl::_gen_call_malloc_gc` emits this after a residual
            // malloc. Address zero is valid linear memory, so omitting the
            // check would turn OOM into silent heap corruption.
            OpCode::CheckMemoryError => {
                emit_memory_error_check(
                    &mut sink,
                    constants,
                    value_types,
                    op.arg(0).to_opref(),
                    residual_type_base,
                    ca.ca_reload_fn_ptr,
                    ca.jf_top_addr,
                    ca.attached.propagate_exception_descr,
                );
            }
            OpCode::ZeroArray => {
                if op.num_args() < 5 {
                    return Err(BackendError::Unsupported(
                        "wasm codegen: ZeroArray expects \
                         [base, start, size, scale_start, scale_size]"
                            .into(),
                    ));
                }
                let descr = op.getdescr().ok_or_else(|| {
                    BackendError::Unsupported(
                        "wasm codegen: ZeroArray is missing its ArrayDescr".into(),
                    )
                })?;
                let ad = descr.as_array_descr().ok_or_else(|| {
                    BackendError::Unsupported(
                        "wasm codegen: ZeroArray descr is not an ArrayDescr".into(),
                    )
                })?;
                let scale_start =
                    const_operand_value(constants, op.arg(3).to_opref()).ok_or_else(|| {
                        BackendError::Unsupported(
                            "wasm codegen: ZeroArray scale_start is not constant".into(),
                        )
                    })?;
                let scale_size =
                    const_operand_value(constants, op.arg(4).to_opref()).ok_or_else(|| {
                        BackendError::Unsupported(
                            "wasm codegen: ZeroArray scale_size is not constant".into(),
                        )
                    })?;
                let scale_start = u64::try_from(scale_start).map_err(|_| {
                    BackendError::Unsupported(
                        "wasm codegen: ZeroArray has a negative start scale".into(),
                    )
                })?;
                let scale_size = u64::try_from(scale_size).map_err(|_| {
                    BackendError::Unsupported(
                        "wasm codegen: ZeroArray has a negative size scale".into(),
                    )
                })?;

                emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                sink.i32_wrap_i64();
                emit_resolve(&mut sink, constants, value_types, op.arg(1).to_opref());
                sink.i32_wrap_i64();
                emit_scale_index(&mut sink, scale_start);
                sink.i32_add();
                sink.i32_const(ad.base_size() as i32);
                sink.i32_add();
                sink.i32_const(0);
                emit_resolve(&mut sink, constants, value_types, op.arg(2).to_opref());
                sink.i32_wrap_i64();
                emit_scale_index(&mut sink, scale_size);
                sink.memory_fill(0);
            }
            OpCode::LoadFromGcTable => {
                // `assembler.py` `genop_load_from_gc_table`: the arg is a
                // `ConstInt(index)` into the per-loop `GcTable`
                // (`remove_ref_constants`, rewrite.py `remove_constptr`)
                // whose base is baked absolute. The table is a plain guest heap
                // allocation, so `base + index*WORD` is an ordinary linear-memory
                // address; the collector forwards the slot in place, so the load
                // reads the reference at its current address.
                let vi = op.pos().get().raw();
                if !OpRef::raw_is_constant(vi) {
                    let index = resolve_const_bits(constants, op.arg(0).to_opref());
                    let base = gc_table_bases.get(&vi).copied().unwrap_or(gc_table_base);
                    emit_gc_table_load(&mut sink, base, index);
                    sink.local_set(value_types.local(vi));
                }
            }
            OpCode::ThreadlocalrefGet => {
                let vi = op.pos().get().raw();
                if !OpRef::raw_is_constant(vi) {
                    let Some(base) = residual_type_base.filter(|_| alloc.threadlocal_fn_ptr != 0)
                    else {
                        return Err(BackendError::Unsupported(
                            "wasm backend: ThreadlocalrefGet needs a residual TLS helper".into(),
                        ));
                    };
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                    sink.i32_const(alloc.threadlocal_fn_ptr as i32);
                    emit_push_site(&mut sink, &site_gcmap, op_idx);
                    sink.call_indirect(0, base + 1);
                    sink.local_set(value_types.local(vi));
                }
            }
            // resoperation.py `LoadEffectiveAddress`:
            // base + (index << shift) + base_offset. Keep the calculation in
            // the IR's i64 address carrier; memory ops wrap only when loading.
            OpCode::LoadEffectiveAddress => {
                let vi = op.pos().get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve(&mut sink, constants, value_types, op.arg(1).to_opref());
                    emit_resolve(&mut sink, constants, value_types, op.arg(3).to_opref());
                    sink.i64_shl();
                    emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                    sink.i64_add();
                    emit_resolve(&mut sink, constants, value_types, op.arg(2).to_opref());
                    sink.i64_add();
                    sink.local_set(value_types.local(vi));
                }
            }

            // ── CALL_ASSEMBLER ──
            // Lower the call into an in-module `call_indirect` into its compiled
            // callee loop instead of a
            // host round-trip. A fresh callee frame is allocated as a real
            // GC-managed nursery `JitFrame` (push_jf-rooted on the jitframe
            // shadow stack; traced by its OWN per-frame gcmap covering
            // its input + home Ref slots), the descriptor inputs are written to its
            // input slots, the loop runs on it (recursing through this same arm
            // for deeper levels), then the result Ref is read back from output
            // slot 0. `compile_loop` and `compile_bridge` validate the descriptor
            // and target metadata before enabling this arm. The callee
            // `call_indirect` runs a full compiled loop, which allocates and
            // collects; each live callee frame is
            // self-described by its gcmap so a collection forwards its Refs (no
            // shared-arena single-stride walker). This bridge's own wasm-local
            // Refs still hold pre-call (from-space) addresses on return, so
            // reload them from the (forwarded) homes after the call.
            opcode if opcode.is_call_assembler() && ca.emit_ca => {
                emit_force_bracket_before_call(
                    &mut sink,
                    constants,
                    value_types,
                    ref_homes,
                    frame,
                    ops,
                    op_idx,
                    guard_idx,
                    guard_dispatch,
                );
                let vi = op.pos().get().raw();
                let descr = op
                    .getdescr()
                    .expect("CALL_ASSEMBLER op must carry a descriptor");
                let op_token = descr
                    .as_call_descr()
                    .and_then(|descr| descr.call_target_token())
                    .expect("CALL_ASSEMBLER op must carry a callee token");
                let tgt = ca
                    .targets
                    .get(&op_token)
                    .expect("CA op target must be registered");
                let dispatch_entry = tgt.dispatch_entry as i32;
                sink.i32_const(dispatch_entry);
                sink.i32_load(mem32(crate::failguard::WASM_CA_DISPATCH_TARGET_PTR_OFS));
                sink.local_tee(ca_target_local);
                sink.i32_eqz();
                sink.if_(BlockType::Empty);
                sink.unreachable();
                sink.end();

                // A terminally-declined target cannot be restarted from the
                // CALL_ASSEMBLER reds: these are loop-header live-ins, not a
                // function-entry PyFrame or necessarily the function's call
                // arguments.  Continue through the orthodox CA frame path
                // below instead.  It marshals every live-in into a callee
                // JitFrame and its non-finish path blackhole-resumes the
                // callee correctly.  This is temporarily more expensive in
                // the bounded caller-invalidation window; deopting the outer
                // trace at the Python CALL needs resume metadata that a
                // CALL_ASSEMBLER op does not currently carry.

                // The rewriter's `CallMallocNurseryVarsizeFrame` allocated the
                // frame and `GcStore`d each input at `_ll_initial_locs`.
                // Arg 0 is that object base. A virtualizable, when present,
                // is arg 1 and was also stored into its frame slot.
                if op.num_args() == 0 {
                    return Err(BackendError::Unsupported(
                        "wasm backend: CALL_ASSEMBLER is missing its frame argument".into(),
                    ));
                }
                emit_resolve(&mut sink, constants, value_types, op.arg(0).to_opref());
                sink.i32_wrap_i64();
                sink.local_tee(ca_cfp_local);
                emit_memory_error_if_i32_zero(
                    &mut sink,
                    residual_type_base,
                    ca.ca_reload_fn_ptr,
                    ca.jf_top_addr,
                    ca.attached.propagate_exception_descr,
                );
                // Recycled nursery bytes. The entry publish unions with the
                // live map, so a fresh frame must start with a null map or
                // that union walks a garbage length.
                sink.local_get(ca_cfp_local);
                sink.i64_const(0);
                emit_word_store(&mut sink, majit_backend::jitframe::JF_GCMAP_OFS as u64);
                // Used homes are a prefix of `home_slots`. A later redirect
                // marks further reserved homes in the same frame, so the
                // whole reserved region has to be null before the push.
                // Bytes outside it are not in the static home gcmap.
                emit_clear_reserved_homes(
                    &mut sink,
                    ca_cfp_local,
                    ca_target_local,
                    alloc_scratch_local,
                );
                emit_push_site(&mut sink, &site_gcmap, op_idx);
                emit_ca_push_frame(
                    &mut sink,
                    ca.inline.as_ref(),
                    residual_type_base,
                    ca.ca_push_fn_ptr,
                    jit_call_idx,
                    ca_cfp_local,
                    alloc_scratch_local,
                )?;
                sink.local_get(ca_cfp_local);
                sink.i32_const(majit_backend::jitframe::FIRST_ITEM_OFFSET as i32);
                sink.i32_add();
                sink.local_set(ca_cfp_local);
                // dispatch key = 0: run the loop from its entry (preamble), not a
                // LABEL resume — this is a fresh call. The offset is loaded
                // from the dispatch entry because redirect can move it.
                sink.local_get(ca_cfp_local);
                sink.local_get(ca_target_local);
                sink.i32_load(mem32(crate::failguard::WASM_CA_TARGET_DISPATCH_KEY_OFS_OFS));
                sink.i32_add();
                sink.i64_const(0);
                sink.i64_store(mem64(0));
                // Run the callee loop on F'; discard the returned frame_ptr.
                sink.local_get(ca_cfp_local);
                // The immutable target snapshot was loaded before allocating
                // F', so this function is exactly the one whose geometry and
                // gcmap initialized that frame.
                sink.local_get(ca_target_local);
                sink.i32_load(mem32(crate::failguard::WASM_CA_TARGET_FUNC_HANDLE_OFS));
                sink.local_tee(ca_fi_local);
                sink.i32_eqz();
                sink.if_(BlockType::Empty);
                sink.unreachable();
                sink.end();
                sink.local_get(ca_fi_local);
                emit_push_site(&mut sink, &site_gcmap, op_idx);
                sink.call_indirect(0, 0);
                sink.drop();
                // The recursive call may minor-collect and move this nursery
                // callee frame. Deeper levels have already popped, so the
                // jitframe shadow-stack top is this level's frame; reload its
                // ITEMS base before reading F'[0] or F'[1].
                if let (Some(_base), Some(inline)) = (residual_type_base, ca.inline) {
                    emit_ca_reload_top(&mut sink, inline.jf_top_addr);
                    sink.i64_extend_i32_u();
                } else if let Some(base) = residual_type_base {
                    sink.i32_const(ca.ca_reload_fn_ptr as i32);
                    emit_push_site(&mut sink, &site_gcmap, op_idx);
                    sink.call_indirect(0, base);
                } else {
                    let jit_call =
                        jit_call_idx.expect("CA arm needs jit_call for the frame trampolines");
                    emit_call_area_addr(&mut sink);
                    sink.i64_const(ca.ca_reload_fn_ptr);
                    sink.i64_store(mem64(STATIC_CALL_FUNC_OFS));
                    emit_call_area_addr(&mut sink);
                    sink.i64_const(0);
                    sink.i64_store(mem64(STATIC_CALL_NARGS_OFS));
                    emit_store_call_result_facts(&mut sink, 8);
                    emit_push_site(&mut sink, &site_gcmap, op_idx);
                    emit_jit_call(&mut sink, jit_call);
                    emit_call_area_addr(&mut sink);
                    sink.i64_load(mem64(STATIC_CALL_RESULT_OFS));
                }
                sink.i32_wrap_i64();
                sink.local_set(ca_cfp_local);
                // `jf_descr` is the callee's exit descr cell. A clean finish
                // writes the one `done_with_this_frame` cell for this result
                // kind (`_call_assembler_check_descr`). The result is already
                // in F'[1]. Any other cell is a guard deopt or a raising
                // finish; `wasm_ca_resume_deopt` blackhole-resumes it.
                let finish_ptr = ca
                    .attached
                    .done_with_this_frame_descr_ptr_for_type(op.opcode.result_type());
                sink.local_get(ca_cfp_local);
                sink.i32_const(majit_backend::jitframe::FIRST_ITEM_OFFSET as i32);
                sink.i32_sub();
                sink.i32_load(memarg(majit_backend::jitframe::JF_DESCR_OFS as u64, 2));
                // `0` is the unattached answer (`AttachedDescrPtrs`). It must
                // not compare equal to an unset `jf_descr`.
                if finish_ptr == 0 {
                    sink.drop();
                    sink.i32_const(0);
                } else {
                    sink.i32_const(finish_ptr as i32);
                    sink.i32_eq();
                }
                sink.if_(BlockType::Result(ValType::I64));
                // clean finish: result Ref = F'[1] (output slot 0).
                sink.local_get(ca_cfp_local);
                sink.i64_load(mem64(FRAME_SLOT_BASE));
                sink.else_();
                // deopt: wasm_ca_resume_deopt(frame_ptr: i64, compiled_ptr: i64).
                sink.local_get(ca_cfp_local);
                sink.i64_extend_i32_u();
                sink.local_get(ca_target_local);
                sink.i64_load32_u(memarg(crate::failguard::WASM_CA_TARGET_COMPILED_PTR_OFS, 2));
                sink.i32_const(ca.deopt_helper_slot as i32);
                // call_indirect(table_index, type_index): the shared table is 0.
                emit_push_site(&mut sink, &site_gcmap, op_idx);
                sink.call_indirect(0, ca_helper_type_idx);
                sink.end();
                // The recursive call or deopt helper may have collected and
                // moved this invocation's own frame. Reload it before the pop
                // trampoline and post-call home loads address local 0. As above,
                // the trampoline-only configuration retains its earlier stale-
                // local-0 limitation because its scratch writes cannot reload it
                // safely.
                if let (Some(_base), Some(inline)) = (residual_type_base, ca.inline) {
                    emit_ca_reload_caller(&mut sink, inline.jf_top_addr);
                    sink.local_set(0);
                    emit_frame_write_barrier(&mut sink);
                } else if let Some(base) = residual_type_base {
                    sink.i32_const(ca.ca_reload_caller_fn_ptr as i32);
                    emit_push_site(&mut sink, &site_gcmap, op_idx);
                    sink.call_indirect(0, base);
                    sink.i32_wrap_i64();
                    sink.local_set(0);
                    sink.sync_gcmap_frame();
                    emit_frame_write_barrier(&mut sink);
                }
                // The frame ABI carries every scalar result as i64 bits. Ref
                // and Int use those bits directly; Float crosses the local
                // boundary with a reinterpret; Void discards the placeholder
                // produced by the common clean/deopt expression.
                match op.opcode.result_type() {
                    Type::Float if !OpRef::raw_is_constant(vi) => {
                        sink.f64_reinterpret_i64();
                        sink.local_set(value_types.local(vi));
                    }
                    Type::Int | Type::Ref if !OpRef::raw_is_constant(vi) => {
                        sink.local_set(value_types.local(vi));
                    }
                    Type::Void | Type::Int | Type::Ref | Type::Float => {
                        sink.drop();
                    }
                }
                // Pop the callee frame off the jitframe shadow stack (strict
                // LIFO).  `assembler.py` `_call_footer_shadowstack` is
                // `SUB [rootstacktop], 2*WORD`; the helper pops only when
                // the inline shadow-stack cells are not published.
                if let Some(inline) = ca.inline {
                    emit_ca_pop_footer(&mut sink, inline, alloc_scratch_local, dispatch_entry);
                } else if let Some(base) = residual_type_base {
                    sink.local_get(ca_cfp_local);
                    sink.i64_extend_i32_u();
                    sink.i32_const(ca.ca_pop_fn_ptr as i32);
                    emit_push_site(&mut sink, &site_gcmap, op_idx);
                    sink.call_indirect(0, base + 1);
                    sink.drop(); // returns 0; ignored
                } else {
                    let jit_call =
                        jit_call_idx.expect("CA arm needs jit_call for the frame trampolines");
                    emit_call_area_addr(&mut sink);
                    sink.i64_const(ca.ca_pop_fn_ptr);
                    sink.i64_store(mem64(STATIC_CALL_FUNC_OFS));
                    emit_call_area_addr(&mut sink);
                    sink.i64_const(1);
                    sink.i64_store(mem64(STATIC_CALL_NARGS_OFS));
                    emit_call_area_addr(&mut sink);
                    sink.local_get(ca_cfp_local);
                    sink.i64_extend_i32_u();
                    sink.i64_store(mem64(STATIC_CALL_ARGS_OFS));
                    emit_store_call_result_facts(&mut sink, 8);
                    emit_push_site(&mut sink, &site_gcmap, op_idx);
                    emit_jit_call(&mut sink, jit_call);
                }
                // The callee recursion minor-collected; this bridge's other live
                // Ref locals are now stale. Reload them from the forwarded homes.
                // Skip the result `vi`: its local holds the just-read callee output
                // and its home is not written until the store-on-def below, so a
                // reload would clobber it with the home's pre-call (stale) value.
                let skip = (!OpRef::raw_is_constant(vi)).then_some(vi);
                // Simple `_call_footer_shadowstack` does not collect, and
                // local 0 already holds the caller from the post-call reload.
                // The helper path can collect; that is a property of the
                // callee snapshot, not of this module's ops.
                if ca.inline.is_none() {
                    emit_reload_ca_frame_if_necessary(
                        &mut sink,
                        residual_type_base,
                        ca.ca_reload_fn_ptr,
                        ca.inline,
                    );
                } else {
                    sink.i32_const(dispatch_entry);
                    sink.i32_load(mem32(crate::failguard::WASM_CA_DISPATCH_HAS_GNF2_OFS));
                    sink.if_(BlockType::Empty);
                    emit_reload_ca_frame_if_necessary(
                        &mut sink,
                        residual_type_base,
                        ca.ca_reload_fn_ptr,
                        ca.inline,
                    );
                    sink.end();
                }
                emit_reload_refs_from_homes(
                    &mut sink,
                    value_types,
                    site_homes.get(op_idx).map(Vec::as_slice).unwrap_or(&[]),
                    skip,
                    frame,
                    &site_gcmap,
                    op_idx,
                );
            }

            // ── CALL operations (via trampoline) ──
            OpCode::CallI
            | OpCode::CallR
            | OpCode::CallN
            | OpCode::CallF
            | OpCode::CallPureI
            | OpCode::CallPureR
            | OpCode::CallPureN
            | OpCode::CallMayForceI
            | OpCode::CallMayForceR
            | OpCode::CallMayForceN
            | OpCode::CallAssemblerI
            | OpCode::CallAssemblerR
            | OpCode::CallAssemblerN
            | OpCode::CallReleaseGilI
            | OpCode::CallReleaseGilN
            | OpCode::CallLoopinvariantI
            | OpCode::CallLoopinvariantR
            | OpCode::CallLoopinvariantN
            | OpCode::CallLoopinvariantF
            | OpCode::CallPureF
            | OpCode::CallMayForceF
            | OpCode::CallAssemblerF
            | OpCode::CallReleaseGilF => {
                // llgraph `runner.py` `_do_math_sqrt` / x86 `genop_math_sqrt`:
                // OS_MATH_SQRT is `f64.sqrt`, not a residual call.
                if op.with_call_descr(|cd| cd.get_extra_info().oopspecindex)
                    == Some(majit_ir::OopSpecIndex::MathSqrt)
                {
                    let vi = op.pos().get().raw();
                    if !OpRef::raw_is_constant(vi) {
                        emit_resolve_f64(&mut sink, constants, value_types, op.arg(1).to_opref());
                        sink.f64_sqrt();
                        sink.local_set(value_types.local(vi));
                    }
                    continue;
                }
                // `[savebox, funcbox] + argboxes` for CALL_RELEASE_GIL: its
                // `save_err` owns an errno save/restore around the raw call
                // even though there is no GIL to release on wasm.
                let save_err = if residual_func_ofs(op.opcode) == 1 {
                    let Some(save_err) = const_operand_value(constants, op.arg(0).to_opref())
                    else {
                        return Err(BackendError::Unsupported(
                            "wasm backend: CALL_RELEASE_GIL save_err is not a constant".into(),
                        ));
                    };
                    save_err
                } else {
                    0
                };
                let errno_helpers = if save_err == 0 {
                    None
                } else {
                    let Some(base) = residual_type_base.filter(|_| {
                        alloc.write_real_errno_fn_ptr != 0 && alloc.read_real_errno_fn_ptr != 0
                    }) else {
                        return Err(BackendError::Unsupported(
                            "wasm backend: CALL_RELEASE_GIL save_err needs the errno helpers"
                                .into(),
                        ));
                    };
                    Some(base)
                };
                emit_force_bracket_before_call(
                    &mut sink,
                    constants,
                    value_types,
                    ref_homes,
                    frame,
                    ops,
                    op_idx,
                    guard_idx,
                    guard_dispatch,
                );
                let vi = op.pos().get().raw();
                let can_collect = call_can_collect(op);

                // pyjitpl.py `direct_call_release_gil` records CALL_RELEASE_GIL_*
                // as `[savebox, funcbox] + argboxes[1:]`, so its callee is arg 1
                // and its own arguments start at 2. Every other CALL keeps the
                // callee at arg 0.
                let func_ofs = usize::from(matches!(
                    op.opcode,
                    OpCode::CallReleaseGilI | OpCode::CallReleaseGilF | OpCode::CallReleaseGilN
                ));
                let func_ptr_ref = op.arg(func_ofs).to_opref();

                // llsupport/callbuilder.py `emit_call_release_gil`:
                // write_real_errno(); emit_raw_call(); read_real_errno(),
                // with the read ahead of any frame reload in each arm below.
                if let Some(base) = errno_helpers {
                    emit_errno_helper_call(
                        &mut sink,
                        base,
                        save_err,
                        alloc.write_real_errno_fn_ptr,
                    );
                }
                let read_real_errno = |sink: &mut PeepSink<'_, '_>| {
                    if let Some(base) = errno_helpers {
                        emit_errno_helper_call(sink, base, save_err, alloc.read_real_errno_fn_ptr);
                    }
                };

                // Direct in-module residual call: skip the `jit_call` host hop and
                // `call_indirect` the callee's table slot with a static
                // `(i64×n)->i64` type. The residual ABI is uniformly i64 for
                // Int/Ref args+result, so args/result move on the wasm stack with
                // no marshalling and no call-area traffic. A direct target may
                // collect or force, so reload local 0 and its live Ref homes on
                // return. Falls back below when ineligible.
                if let (Some(base), Some(nargs)) =
                    (residual_type_base, residual_call_i64_arity(op, constants))
                {
                    let call_args = &op.getarglist()[func_ofs + 1..];
                    for arg in call_args {
                        emit_resolve(&mut sink, constants, value_types, arg.to_opref());
                    }
                    // func_ptr (arg 0) is the table slot — wrap to i32 index.
                    emit_resolve(&mut sink, constants, value_types, func_ptr_ref);
                    sink.i32_wrap_i64();
                    // call_indirect(table_index, type_index): table 0, type for arity n.
                    emit_push_site(&mut sink, &site_gcmap, op_idx);
                    sink.call_indirect(0, base + nargs as u32);
                    if !OpRef::raw_is_constant(vi) {
                        sink.local_set(value_types.local(vi));
                    } else {
                        sink.drop(); // value-producing call whose result is unused
                    }
                    read_real_errno(&mut sink);
                    if can_collect {
                        emit_reload_frame_if_necessary(
                            &mut sink,
                            residual_type_base,
                            ca.ca_reload_fn_ptr,
                            ca.jf_top_addr,
                        );
                        emit_reload_refs_from_homes(
                            &mut sink,
                            value_types,
                            site_homes.get(op_idx).map(Vec::as_slice).unwrap_or(&[]),
                            (!OpRef::raw_is_constant(vi)).then_some(vi),
                            frame,
                            &site_gcmap,
                            op_idx,
                        );
                    }
                    // store-on-def (end of loop) homes a Ref result, so the
                    // direct path must NOT `continue` past it.
                } else if let (Some(base), Some(nargs)) = (
                    residual_type_base,
                    residual_call_void_word_arity(op, constants),
                ) {
                    // Direct in-module word-ABI void residual call: the callee
                    // really is `(i64×n)->i64` (descr result_size == 8), so use
                    // the i64 family and drop the dummy result.
                    let call_args = &op.getarglist()[func_ofs + 1..];
                    for arg in call_args {
                        emit_resolve(&mut sink, constants, value_types, arg.to_opref());
                    }
                    emit_resolve(&mut sink, constants, value_types, func_ptr_ref);
                    sink.i32_wrap_i64();
                    emit_push_site(&mut sink, &site_gcmap, op_idx);
                    sink.call_indirect(0, base + nargs as u32);
                    sink.drop();
                    read_real_errno(&mut sink);
                    if can_collect {
                        emit_reload_frame_if_necessary(
                            &mut sink,
                            residual_type_base,
                            ca.ca_reload_fn_ptr,
                            ca.jf_top_addr,
                        );
                        emit_reload_refs_from_homes(
                            &mut sink,
                            value_types,
                            site_homes.get(op_idx).map(Vec::as_slice).unwrap_or(&[]),
                            None,
                            frame,
                            &site_gcmap,
                            op_idx,
                        );
                    }
                } else if let Some((sig, &type_idx)) = residual_call_typed_sig(op, constants)
                    .and_then(|sig| {
                        typed_residual_type_indices
                            .get(&sig)
                            .map(|type_idx| (sig, type_idx))
                    })
                {
                    // Direct in-module typed residual call: descr-derived
                    // mixed `(i64/f64…) -> i64/f64`, or the oracle's i32-ABI
                    // twin (`i32.wrap_i64` / `i64.extend_i32_u`).
                    let (params, result_ty) = &sig;
                    let call_args = &op.getarglist()[func_ofs + 1..];
                    debug_assert_eq!(call_args.len(), params.len());
                    emit_typed_residual_call(
                        &mut sink,
                        constants,
                        value_types,
                        call_args,
                        params,
                        func_ptr_ref,
                        &site_gcmap,
                        op_idx,
                        type_idx,
                    );
                    let is_void_op = matches!(
                        op.opcode,
                        OpCode::CallN
                            | OpCode::CallPureN
                            | OpCode::CallMayForceN
                            | OpCode::CallAssemblerN
                            | OpCode::CallReleaseGilN
                            | OpCode::CallLoopinvariantN
                    );
                    // A void callee leaves nothing on the stack, so there is
                    // neither a local to home it in nor a value to drop.
                    let homed = if result_ty.is_none() {
                        None
                    } else if is_void_op {
                        sink.drop();
                        None
                    } else if !OpRef::raw_is_constant(vi) {
                        widen_direct_call_result(&mut sink, op, *result_ty);
                        sink.local_set(value_types.local(vi));
                        Some(vi)
                    } else {
                        sink.drop();
                        None
                    };
                    read_real_errno(&mut sink);
                    if can_collect {
                        emit_reload_frame_if_necessary(
                            &mut sink,
                            residual_type_base,
                            ca.ca_reload_fn_ptr,
                            ca.jf_top_addr,
                        );
                        emit_reload_refs_from_homes(
                            &mut sink,
                            value_types,
                            site_homes.get(op_idx).map(Vec::as_slice).unwrap_or(&[]),
                            homed,
                            frame,
                            &site_gcmap,
                            op_idx,
                        );
                    }
                } else if let (Some(base), Some(nargs)) = (
                    true_void_residual_type_base,
                    residual_call_void_true_arity(op, constants),
                ) {
                    // Direct in-module true-void residual call: the callee is
                    // `(i64×n)->()` (descr result_size == 0), so the call has no
                    // result to drop.
                    let call_args = &op.getarglist()[func_ofs + 1..];
                    for arg in call_args {
                        emit_resolve(&mut sink, constants, value_types, arg.to_opref());
                    }
                    emit_resolve(&mut sink, constants, value_types, func_ptr_ref);
                    sink.i32_wrap_i64();
                    emit_push_site(&mut sink, &site_gcmap, op_idx);
                    sink.call_indirect(0, base + nargs as u32);
                    read_real_errno(&mut sink);
                    if can_collect {
                        emit_reload_frame_if_necessary(
                            &mut sink,
                            residual_type_base,
                            ca.ca_reload_fn_ptr,
                            ca.jf_top_addr,
                        );
                        emit_reload_refs_from_homes(
                            &mut sink,
                            value_types,
                            site_homes.get(op_idx).map(Vec::as_slice).unwrap_or(&[]),
                            None,
                            frame,
                            &site_gcmap,
                            op_idx,
                        );
                    }
                } else {
                    let jit_call = jit_call_idx.expect("CALL op present but jit_call not imported");
                    let call_args = &op.getarglist()[func_ofs + 1..];
                    let arg_refs: Vec<OpRef> = call_args.iter().map(|arg| arg.to_opref()).collect();
                    let is_void = matches!(
                        op.opcode,
                        OpCode::CallN
                            | OpCode::CallPureN
                            | OpCode::CallMayForceN
                            | OpCode::CallAssemblerN
                            | OpCode::CallReleaseGilN
                            | OpCode::CallLoopinvariantN
                    );
                    let home = (!OpRef::raw_is_constant(vi) && !is_void).then_some(vi);
                    emit_residual_trampoline_call(
                        &mut sink,
                        constants,
                        value_types,
                        jit_call,
                        func_ptr_ref,
                        &arg_refs,
                        &site_gcmap,
                        op_idx,
                        op,
                        home,
                    )?;
                    // Mirror the direct path: a trampoline residual call may force and collect.
                    read_real_errno(&mut sink);
                    if can_collect {
                        emit_reload_frame_if_necessary(
                            &mut sink,
                            residual_type_base,
                            ca.ca_reload_fn_ptr,
                            ca.jf_top_addr,
                        );
                        emit_reload_refs_from_homes(
                            &mut sink,
                            value_types,
                            site_homes.get(op_idx).map(Vec::as_slice).unwrap_or(&[]),
                            (!is_void && !OpRef::raw_is_constant(vi)).then_some(vi),
                            frame,
                            &site_gcmap,
                            op_idx,
                        );
                    }
                }
            }

            // ── Allocation (via trampoline — treated as CALL) ──
            // rewrite_ops_for_gc lowers every New / NewWithVtable /
            // NewArray / NewArrayClear before this match. Production
            // compile_loop / compile_bridge always run that rewrite.
            OpCode::New | OpCode::NewWithVtable | OpCode::NewArray | OpCode::NewArrayClear => {
                panic!(
                    "wasm codegen: {:?} must have been lowered by rewrite_ops_for_gc",
                    op.opcode
                );
            }
            // ── Misc ──
            OpCode::ForceToken => {
                let vi = op.pos().get().raw();
                if !OpRef::raw_is_constant(vi) {
                    // `FORCE_TOKEN/0/r` — "nowadays, returns the jitframe".
                    // The token is what the SETFIELD_GC that follows parks in
                    // the virtualizable's `vable_token`, and what
                    // `Backend::force` is handed to rebuild a deadframe from,
                    // so it has to NAME this frame. A zero here reads as "no
                    // JIT frame is holding this virtualizable", which makes
                    // `force_virtualizable_if_necessary` skip the force and
                    // leaves an `f_locals` read to answer out of whatever the
                    // frame's own array last received.
                    //
                    // Answer the `JitFrame` BASE, not the items base local 0
                    // holds: the result of this op is Ref-typed, so it takes a
                    // Ref home slot, and both `build_home_gcmap` and
                    // `build_callee_gcmap` mark those slots for the collector.
                    // An items base is an interior pointer; traced as an object
                    // it reads its type id out of the frame's `jf_forward` word.
                    // The object base is a real GCREF, so a CA callee frame that
                    // moves out of the nursery under the very call this token
                    // brackets is forwarded here like any other reference.
                    sink.local_get(0);
                    sink.i32_const(majit_backend::jitframe::FIRST_ITEM_OFFSET as i32);
                    sink.i32_sub();
                    sink.i64_extend_i32_u();
                    sink.local_set(value_types.local(vi));
                }
            }

            // Float operations
            OpCode::FloatAdd | OpCode::FloatSub | OpCode::FloatMul | OpCode::FloatTrueDiv => {
                let vi = op.pos().get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve_f64(&mut sink, constants, value_types, op.arg(0).to_opref());
                    emit_resolve_f64(&mut sink, constants, value_types, op.arg(1).to_opref());
                    match op.opcode {
                        OpCode::FloatAdd => {
                            sink.f64_add();
                        }
                        OpCode::FloatSub => {
                            sink.f64_sub();
                        }
                        OpCode::FloatMul => {
                            sink.f64_mul();
                        }
                        OpCode::FloatTrueDiv => {
                            sink.f64_div();
                        }
                        _ => unreachable!(),
                    }
                    sink.local_set(value_types.local(vi));
                }
            }
            OpCode::FloatNeg => {
                let vi = op.pos().get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve_f64(&mut sink, constants, value_types, op.arg(0).to_opref());
                    sink.f64_neg();
                    sink.local_set(value_types.local(vi));
                }
            }
            OpCode::FloatAbs => {
                let vi = op.pos().get().raw();
                if !OpRef::raw_is_constant(vi) {
                    emit_resolve_f64(&mut sink, constants, value_types, op.arg(0).to_opref());
                    sink.f64_abs();
                    sink.local_set(value_types.local(vi));
                }
            }

            // Debug / metadata / no-op
            OpCode::DebugMergePoint
            | OpCode::JitDebug
            | OpCode::IncrementDebugCounter
            | OpCode::EnterPortalFrame
            | OpCode::LeavePortalFrame
            | OpCode::ForceSpill
            | OpCode::Keepalive => {
                // `JitDebug` realizes its effect in the recorded trace, not in
                // emitted code: `consider_jit_debug` is `pass`.
            }

            _ => {
                // An opcode with no codegen arm declines the whole trace and
                // lets the metainterp fall back to the interpreter (correct,
                // unaccelerated). That covers a void opcode too: its `pos` is
                // `OpRef::NONE`, whose raw is `u32::MAX`, and
                // `raw_is_constant` rejects the sentinel range. So the only op
                // that reaches here and emits nothing is one the optimizer
                // folded into the constant namespace, which by then is pure and
                // has no side effect left to drop.
                //
                // An opcode that must emit nothing belongs in one of the no-op
                // arms above, where the reason it owes no code is written down;
                // a side-effecting op put there is dropped in silence, which is
                // what `CondCallN` was.
                let vi = op.pos().get().raw();
                if !OpRef::raw_is_constant(vi) {
                    return Err(BackendError::Unsupported(format!(
                        "wasm codegen: unhandled opcode {:?}",
                        op.opcode
                    )));
                }
            }
        }

        // store-on-def: mirror a freshly-defined Ref result into its home slot
        // so a (future) collecting allocation can forward it. The local
        // The mapped value local holds the value the matched arm just set; `ref_homes` only
        // keys Ref-typed value ids, so non-Ref / void / constant ops are
        // skipped. Each value-producing arm is operand-stack-neutral, so this
        // appended store is balanced.
        if let Some(id) = result_value_raw(op)
            && let Some(h) = ref_homes.home_id(id)
        {
            sink.local_get(0);
            sink.local_get(value_types.local(id));
            sink.i64_store(mem64(frame.home_ofs(h as u64)));
        }
    }

    if in_loop_body {
        sink.end(); // end loop
    }
    if resume_dispatch {
        sink.end(); // end R $resume
    }
    // A well-formed trace exits through a guard or Finish. Preserve the old
    // malformed/natural-fallthrough behavior without reaching bridge dispatch
    // with a stale frame fail index.
    sink.local_get(0);
    sink.return_();
    sink.end(); // end A $hot_exit

    // Frame-entry bridge dispatch for exits that branch out of the hot exit
    // block. Parameter-entry bridges tail-call from their own guard arm: that
    // arm knows the fixed failure arity and therefore the fixed wasm type.
    // The shared epilogue remains only for the established frame-entry form.
    if bridge_dispatch && bridge_param_type_indices.is_empty() {
        // slot = *(bridge_slot_local), where the local holds a cell address.
        // Address 0 is a guard with no cell; do not load guest address 0.
        sink.local_get(bridge_slot_local);
        sink.i32_eqz();
        sink.if_(BlockType::Result(ValType::I32));
        sink.i32_const(0);
        sink.else_();
        sink.local_get(bridge_slot_local);
        sink.i32_load(memarg(0, 2));
        sink.end();
        sink.local_tee(bridge_slot_local);
        sink.if_(BlockType::Empty);
        sink.local_get(0); // frame_ptr argument to the bridge
        sink.local_get(bridge_slot_local); // table slot
        sink.return_call_indirect(0, 0); // tail call, table 0, type 0: (i32) -> i32
        sink.end();
    }

    sink.local_get(0);
    sink.end(); // end function
    sink.flush();
    drop(sink);

    Ok(func)
}

// ── Helpers ──

/// A peeled loop — real work (the unrolled first iteration = preamble) precedes
/// the loop-header LABEL — whether it carries one LABEL or several. `loop` is
/// emitted at the JUMP's target label, so `build_function` wraps the trace in
/// the resume-at-LABEL entry `br_table` (keyed on the frame dispatch-key slot,
/// key = label ordinal + 1) and a loop-closing bridge re-enters at any of the
/// loop's labels up to and including the header, in-module
/// (`resumable_label_count`). `build_function` keys its wrapper on this
/// predicate; `compile_loop` records it on `CompiledWasmLoop` as
/// `has_preamble`. `compile_bridge` accepts a loop-closing bridge only when
/// its JUMP's descr identifies one of the source loop's OWN labels
/// (`label_descrs`) with matching arity and a resume-safe live set.
pub fn is_resumable_peeled(ops: &[Op]) -> bool {
    let Some(loop_label) = find_loop_label_index(ops) else {
        return false;
    };
    ops[..loop_label]
        .iter()
        .any(|op| op.opcode != OpCode::Label)
}

/// How many of a peeled loop's LABELs the entry `br_table` can re-enter at:
/// those at or before the loop header. Each resume point costs a
/// (past_loader, loader) block pair opened before the wasm `loop`, so a pair
/// belonging to a label INSIDE the loop body would have to close inside the
/// loop — which structured control flow forbids. Such a label stays an in-body
/// marker: it emits nothing and is not published as a target.
pub fn resumable_label_count(ops: &[Op]) -> usize {
    let Some(loop_label) = find_loop_label_index(ops) else {
        return 0;
    };
    ops[..=loop_label]
        .iter()
        .filter(|op| op.opcode == OpCode::Label)
        .count()
}

/// Number of entry-dispatch keys an armed trace module can observe. Ordinary
/// traces have only the fresh-entry bucket; a resumable peeled loop has key 0
/// plus one bucket for each `br_table` resume arm.
pub fn entry_dispatch_key_count(ops: &[Op]) -> usize {
    if is_resumable_peeled(ops) {
        resumable_label_count(ops) + 1
    } else {
        1
    }
}

/// `counter += 1; if counter == threshold { trip(pending_slot) }`, at the entry
/// of an out-of-line bridge whose merge into its owner is waiting on this
/// count. Equality rather than `>=` so the callback fires exactly once.
///
/// The source guard's cell is left alone. `assembler.py`
/// `patch_jump_for_descr` rewrites the guard's jump in place and never
/// leaves it without a target; zeroing the cell here would make the next
/// failure leave the guest while the old bridge is still the valid
/// continuation. The host installs the merged owner after this compiled
/// run returns (`execute_assembler`), and until that publish the cell keeps
/// dispatching to the bridge.
///
/// Operand-stack-neutral, and it reads no frame slot.
fn emit_inline_trip_probe(sink: &mut PeepSink<'_, '_>, probe: InlineTripProbe, type_idx: u32) {
    sink.i32_const(probe.counter_addr as i32);
    sink.i32_const(probe.counter_addr as i32);
    sink.i64_load(mem64(0));
    sink.i64_const(1);
    sink.i64_add();
    sink.i64_store(mem64(0));
    sink.i32_const(probe.counter_addr as i32);
    sink.i64_load(mem64(0));
    sink.i64_const(probe.threshold as i64);
    sink.i64_eq();
    sink.if_(BlockType::Empty);
    // Install from here. The probe runs in the bridge module; the parent
    // stays on the stack and is not re-entered. The parent's next back-edge
    // reads the resume cell and tail-calls the replacement.
    sink.i64_const(probe.pending_slot);
    sink.i32_const(probe.trip_fn_ptr as i32);
    sink.call_indirect(0, type_idx);
    sink.drop(); // returns 0; ignored
    sink.end();
}

/// Increment one module's guest-memory entry counter. `key_local` holds the
/// i32 value consumed by the entry `br_table`; out-of-range values retain that
/// table's normal default-to-fresh-entry behaviour but do not index beyond the
/// fixed counter array.
fn emit_trace_entry_census(
    sink: &mut PeepSink<'_, '_>,
    census: crate::TraceEntryCensusStorage,
    scratch_local: u32,
    key_local: Option<u32>,
) {
    if let Some(key_local) = key_local {
        sink.local_get(key_local);
        sink.i32_const(census.key_count as i32);
        sink.i32_lt_u();
        sink.if_(BlockType::Empty);
        sink.i32_const(census.base as i32);
        sink.local_get(key_local);
        sink.i32_const(std::mem::size_of::<u64>() as i32);
        sink.i32_mul();
        sink.i32_add();
        sink.local_set(scratch_local);
        sink.local_get(scratch_local);
        sink.local_get(scratch_local);
        sink.i64_load(mem64(0));
        sink.i64_const(1);
        sink.i64_add();
        sink.i64_store(mem64(0));
        sink.end();
    } else {
        sink.i32_const(census.base as i32);
        sink.local_set(scratch_local);
        sink.local_get(scratch_local);
        sink.local_get(scratch_local);
        sink.i64_load(mem64(0));
        sink.i64_const(1);
        sink.i64_add();
        sink.i64_store(mem64(0));
    }
}

/// The single-label subset of `is_resumable_peeled`: exactly one LABEL.
/// No longer consulted by the bridge accept-condition (which resolves the
/// JUMP's target label by descr identity uniformly); kept as a shape
/// predicate for tests.
pub fn is_single_label_peeled(ops: &[Op]) -> bool {
    let label_count = ops.iter().filter(|op| op.opcode == OpCode::Label).count();
    is_resumable_peeled(ops) && label_count == 1
}

/// Argument count of each `LABEL`, in ordinal order (the same ordinals
/// `compile_loop` stamps via `set_label_block_id`). `compile_bridge` declines
/// a loop-closing bridge whose JUMP arity differs from its target label's
/// count, since the resume loader reads exactly that many positional frame
/// slots.
pub fn label_arg_counts(ops: &[Op]) -> Vec<usize> {
    ops.iter()
        .filter(|op| op.opcode == OpCode::Label)
        .map(|op| op.num_args())
        .collect()
}

pub fn has_label_param_entry(
    inputargs: &[InputArgRc],
    ops: &[Op],
    frame: FrameGeometry,
    bridge_entry_arity: Option<usize>,
) -> bool {
    if bridge_entry_arity.is_some() || !is_resumable_peeled(ops) {
        return false;
    }
    let resumable = resumable_label_count(ops);
    let labels_fit = label_arg_counts(ops)
        .into_iter()
        .take(resumable)
        .all(|arity| arity <= crate::FROZEN_LABEL_PARAM_ARITY);
    labels_fit
        && inputargs.len() <= crate::FROZEN_LABEL_PARAM_ARITY
        // The shim loads from `FRAME_SLOT_BASE`, so its `FROZEN_LABEL_PARAM_ARITY`
        // reads occupy slots 1..=FROZEN_LABEL_PARAM_ARITY — slot 0 is the
        // dispatch key. A frame with exactly that many slots would let the last
        // load run off the end.
        && frame.value_slots >= crate::FROZEN_LABEL_PARAM_ARITY + 1
}

/// Per-label `(resume_safe, requires_own_frame)` metadata in ordinal order.
/// Missing pre-LABEL live-ins are safe when the frozen geometry contains the
/// capture plan. Such a plan is tied to the physical frame on which the owning
/// loop populated it; a sibling specialization may share the same geometry
/// but not those values, so bridge chaining must then stay on the owner.
pub fn label_resume_info(
    inputargs: &[InputArgRc],
    ops: &[Op],
    frame: FrameGeometry,
) -> Vec<(bool, bool)> {
    let resume = LabelResumeData::collect(inputargs, ops);
    let storage_supported = resume.supported_by(frame);
    resume
        .per_label
        .iter()
        .enumerate()
        .map(|(j, missing)| {
            (
                !resume.uncapturable[j] && (missing.is_empty() || storage_supported),
                !missing.is_empty(),
            )
        })
        .collect()
}

fn find_jump_target_label_index(ops: &[Op], jump: &Op) -> Option<usize> {
    let target = jump.getdescr()?;
    ops.iter().position(|op| {
        op.opcode == OpCode::Label
            && op
                .getdescr()
                .is_some_and(|descr| std::sync::Arc::ptr_eq(&descr, &target))
    })
}

pub(crate) fn find_loop_label_index(ops: &[Op]) -> Option<usize> {
    // The FIRST JUMP, not the last: a merged stream appends each inlined region
    // after the owner's ops, so the owner's terminal JUMP — the one that
    // defines the loop — precedes every region's. Reading the last would let a
    // region closing at an earlier LABEL move the `loop` onto that label.
    match ops.iter().find(|op| op.opcode == OpCode::Jump) {
        // x86/assembler.py:2463 `if target_token in
        // self.target_tokens_currently_compiling` — the TOKEN decides. A JUMP
        // that names a token this compilation does not define is upstream's
        // `else` arm at :2467 (`JMP(imm(target))`, an absolute jump into
        // another trace), even when this trace defines labels of its own.
        // That is exactly a `jump_to_preamble` retrace: compile.py:381 keeps
        // the retrace's own label_op in the middle while unroll.py:238-242
        // retargets the JUMP at the ORIGINAL loop's start descr. Answering
        // with the trailing label here would turn that JUMP into a back-edge
        // to a label it does not name.
        Some(jump) if jump.has_descr() => find_jump_target_label_index(ops, jump),
        // No descr to decide with (legacy IR whose JUMP carries none), or no
        // JUMP at all: keep the historical last-LABEL answer.
        _ => ops.iter().rposition(|op| op.opcode == OpCode::Label),
    }
}

/// Ordinal of the resumable LABEL a JUMP names, when that label is not the loop
/// header. `None` for the header, for a label past the resumable prefix, and
/// for a JUMP naming no local label. An inlined region with `Some(j)` has no
/// `br` target: the `loop` opens at the header, so branching there would skip
/// the segment between label `j` and the header.
fn jump_resume_ordinal(ops: &[Op], jump: &Op, num_labels: usize) -> Option<usize> {
    let ordinal = jump_label_ordinal(ops, jump)?;
    (ordinal + 1 < num_labels).then_some(ordinal)
}

/// Ordinal of the LABEL a JUMP names among this stream's LABELs. `None` when
/// the JUMP names no local label at all.
fn jump_label_ordinal(ops: &[Op], jump: &Op) -> Option<usize> {
    let label_idx = find_jump_target_label_index(ops, jump)?;
    Some(
        ops[..label_idx]
            .iter()
            .filter(|op| op.opcode == OpCode::Label)
            .count(),
    )
}

/// JUMP args that can occupy their LABEL-arg wasm local and Ref home.
///
/// x86 `RegisterManager` colors a loop-closing def into the LABEL
/// register; wasm otherwise mints a fresh local, then parallel-moves
/// and re-homes at every back-edge. When the def and the LABEL slot
/// do not interfere (`HomeLiveness::live_across`), share the location
/// so the JUMP is an identity self-move.
fn jump_phi_coalesce_pairs(ops: &[Op]) -> Vec<(u32, u32)> {
    let liveness = HomeLiveness::collect_with_regions(&[], ops, &[]);
    let mut pairs = Vec::new();
    let mut taken_j = Vec::new();
    let mut taken_l = Vec::new();
    for jump in ops.iter().filter(|op| op.opcode == OpCode::Jump) {
        if find_jump_target_label_index(ops, jump).is_none() {
            continue;
        }
        let label_args = find_label_args(ops, jump);
        let jump_args = jump.getarglist();
        let n = jump_args.len().min(label_args.len());
        // LABEL args are simultaneous phis. `live_across` dates each at
        // the LABEL (or at -1 for an inputarg), so it does not see two
        // phis as interfering. A swap `JUMP(p1, p0)` / rotate then
        // accepts every pair; ValueLocals walks the alias cycle and
        // falls back to distinct locals, but RefHomes remaps sequentially
        // and both values keep one home. Refuse a JUMP arg that is
        // itself a target LABEL arg.
        let label_arg_ids: Vec<u32> = label_args
            .iter()
            .copied()
            .filter(|a| *a != OpRef::NONE && !a.is_constant())
            .map(OpRef::raw)
            .collect();
        for i in 0..n {
            let jarg = jump_args[i].to_opref();
            let larg = label_args[i];
            if jarg.is_constant()
                || larg.is_constant()
                || jarg == OpRef::NONE
                || larg == OpRef::NONE
                || jarg.raw() == larg.raw()
                || jarg.ty() != larg.ty()
            {
                continue;
            }
            let jid = jarg.raw();
            let lid = larg.raw();
            if taken_j.contains(&jid) || taken_l.contains(&lid) {
                continue;
            }
            if label_arg_ids.contains(&jid) {
                continue;
            }
            let def_j = liveness.defined_at(jid);
            if def_j == i32::MAX || def_j < 0 {
                continue;
            }
            if liveness.live_across(lid, def_j as usize) {
                continue;
            }
            pairs.push((jid, lid));
            taken_j.push(jid);
            taken_l.push(lid);
        }
    }
    pairs
}

fn find_label_args(ops: &[Op], jump: &Op) -> Vec<OpRef> {
    // A multi-label trace's JUMP does not necessarily target its last label.
    // LABEL and JUMP share the loop-target descr, so resolve the target by Arc
    // identity just like compile_bridge's external-JUMP path. Falling back to
    // the last label preserves the historical behavior for legacy IR whose
    // JUMP carries no descr.
    if let Some(label_idx) = find_jump_target_label_index(ops, jump) {
        return ops[label_idx]
            .getarglist()
            .iter()
            .map(|arg| arg.to_opref())
            .collect();
    }
    for op in ops.iter().rev() {
        if op.opcode == OpCode::Label {
            return op.getarglist().iter().map(|a| a.to_opref()).collect();
        }
    }
    Vec::new()
}

/// A legacy pool-indexed const that is absent from the constants pool at emit
/// time is an optimizer-seeding invariant violation — panic loudly, matching
/// `collect_constants_from_ops`' `missing_legacy_const`, instead of emitting a
/// silent `0`. On native a null Ref traps on the first dereference; wasm's
/// offset 0 is valid linear memory, so a silent `0` is read as garbage and
/// miscompiles quietly rather than crashing.
#[cold]
#[inline(never)]
fn missing_emit_const(opref: OpRef) -> ! {
    panic!(
        "wasm emit_resolve: legacy pool-indexed const OpRef (raw={}) is absent \
         from the constants pool — the optimizer producer must seed it (or mint \
         an inline Const) instead of emitting a silent 0.",
        opref.raw()
    );
}

/// A memory-access or allocation op reached codegen without the layout descr it
/// must carry (Field/Array/Size). Emitting a default offset/size/type_id would
/// silently miscompile — on wasm, offset 0 is valid linear memory, so a bogus
/// address reads/writes garbage instead of trapping. Fail loud instead. Dead on
/// valid traces: every such op carries its descr (RPython invariant), and the
/// native x86 backend defaults identically without ever hitting the default.
#[cold]
#[inline(never)]
fn missing_layout_descr(what: &str, op: &Op) -> ! {
    panic!(
        "wasm codegen: {what} is absent for {:?} — a memory-access/allocation op \
         must carry its layout descr; a default offset/size/type_id would \
         silently miscompile.",
        op.opcode
    );
}

/// Resolve a constant operand's i64 bits: the inline `Const` value if the
/// variant carries one (`history.py:227/268/314`), else the legacy pool entry.
/// A pool miss panics via [`missing_emit_const`] rather than falling back to a
/// silent `0`.
fn resolve_const_bits(constants: &indexmap::IndexMap<u32, i64>, opref: OpRef) -> i64 {
    opref.inline_const_bits().unwrap_or_else(|| {
        constants
            .get(&opref.raw())
            .copied()
            .unwrap_or_else(|| missing_emit_const(opref))
    })
}

/// Reload the backend-only live-ins captured at `label` into their locals and
/// refresh their Ref homes, the ordinary homes the resumed body's
/// collecting-call reload path reads. Both resume paths need it: the entry
/// `br_table` arrives with every local zero-initialised, and an inlined region
/// arrives after the loop's own back edge may have rebound one of these locals
/// since the peeled pass wrote it.
fn emit_label_capture_restore(
    sink: &mut PeepSink<'_, '_>,
    label_resume: &LabelResumeData,
    value_types: &ValueLocals,
    ref_homes: &RefHomes,
    frame: FrameGeometry,
    label: usize,
) {
    for &r in &label_resume.per_label[label] {
        let storage = label_resume
            .storage(r)
            .expect("LABEL live-in has assigned capture storage");
        sink.local_get(0);
        sink.i64_load(mem64(label_resume.frame_offset(storage, frame)));
        if value_types.ty(r.raw()) == ValType::F64 {
            sink.f64_reinterpret_i64();
        }
        sink.local_set(value_types.local(r.raw()));
        if let Some(h) = ref_homes.home(r) {
            sink.local_get(0);
            sink.local_get(value_types.local(r.raw()));
            sink.i64_store(mem64(frame.home_ofs(h as u64)));
        }
    }
}

fn emit_gc_table_load(sink: &mut PeepSink<'_, '_>, base: u32, index: i64) {
    let slot = base as u64 + index as u64 * std::mem::size_of::<majit_ir::GcRef>() as u64;
    sink.i32_const(slot as i32);
    sink.i64_load32_u(memarg(0, 2));
}

/// Failarg counterpart of cranelift `resolve_failarg_opref`: rematerialize a
/// preamble `LoadFromGcTable` (or SameAs of one) instead of spilling the
/// local the back edge does not refresh. A Ref with a home is loaded from
/// that slot — the same contract `emit_force_arm` uses — because the local
/// can be a stale from-space pointer after a collecting call, and
/// `build_home_gcmap` only traces homes.
fn emit_resolve_failarg(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    opref: OpRef,
    gc_table_slots: &HashMap<u32, (u32, i64)>,
    ref_homes: &RefHomes,
    frame: FrameGeometry,
    const_tables: &ConstPtrTables,
    const_table_base: u32,
) {
    // rewrite.py leaves a ConstPtr failarg as a constant. Load it from
    // the table on this path only — the collector forwards the slot —
    // rather than baking the compile-time address as `i64.const`.
    if let Some(g) = opref.as_const_ptr()
        && !g.is_null()
        && let Some((base, index)) = const_tables.slot(const_table_base, g.0)
    {
        emit_gc_table_load(sink, base, i64::from(index));
        return;
    }
    if !opref.is_none()
        && !opref.is_constant()
        && let Some(&(base, index)) = gc_table_slots.get(&opref.raw())
    {
        emit_gc_table_load(sink, base, index);
        return;
    }
    if !opref.is_none()
        && !opref.is_constant()
        && let Some(home) = ref_homes.home(opref)
    {
        sink.local_get(0);
        sink.i64_load(mem64(frame.home_ofs(home as u64)));
        return;
    }
    emit_resolve(sink, constants, value_types, opref);
}

fn gc_table_failarg_slots(
    ops: &[Op],
    constants: &indexmap::IndexMap<u32, i64>,
    gc_table_base: u32,
    gc_table_bases: &HashMap<u32, u32>,
) -> HashMap<u32, (u32, i64)> {
    let mut slots = HashMap::new();
    for op in ops {
        match op.opcode {
            OpCode::LoadFromGcTable => {
                let vi = op.pos().get().raw();
                if !OpRef::raw_is_constant(vi) {
                    let index = resolve_const_bits(constants, op.arg(0).to_opref());
                    let base = gc_table_bases.get(&vi).copied().unwrap_or(gc_table_base);
                    slots.insert(vi, (base, index));
                }
            }
            OpCode::SameAsI | OpCode::SameAsR | OpCode::CastOpaquePtr => {
                let vi = op.pos().get().raw();
                let src = op.arg(0).to_opref();
                if !OpRef::raw_is_constant(vi)
                    && !src.is_none()
                    && !src.is_constant()
                    && let Some(&slot) = slots.get(&src.raw())
                {
                    slots.insert(vi, slot);
                }
            }
            _ => {}
        }
    }
    slots
}

fn emit_resolve(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    opref: OpRef,
) {
    if opref.is_constant() {
        let val = resolve_const_bits(constants, opref);
        sink.i64_const(val);
    } else if opref.is_none() {
        // A `NONE` fail-arg is a dead deopt slot: the optimizer numbered no
        // value for it, so the blackhole never reads it back (its resume data
        // carries the live values). Spill a zero placeholder — matching the
        // native backends, whose deadframe slot for an unmapped fail-arg is
        // never consumed. Resolving it as a local would index `value_types`
        // out of bounds (`raw() == u32::MAX`).
        sink.i64_const(0);
    } else {
        sink.local_get(value_types.local(opref.raw()));
        if value_types.ty(opref.raw()) == ValType::F64 {
            sink.i64_reinterpret_f64();
        }
    }
}

/// Resolve a Float operand as f64. Constants retain their i64 bit encoding in
/// the constant pool and are converted at the local boundary.
fn emit_resolve_f64(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    opref: OpRef,
) {
    if opref.is_constant() {
        let val = resolve_const_bits(constants, opref);
        sink.i64_const(val);
        sink.f64_reinterpret_i64();
    } else {
        debug_assert_eq!(value_types.ty(opref.raw()), ValType::F64);
        sink.local_get(value_types.local(opref.raw()));
    }
}

/// Values the optimizer left as plain (non-`Const`) OpRefs whose only
/// definition is a constant-pool entry, paired with that entry's raw bits.
///
/// Constant folding and the short preamble both hand the backend a folded
/// value under its original op position, with no producing op left in the
/// trace. `RegisterManager::loc` (dynasm `regalloc.rs`) covers that case with
/// a constants-map fallback taken once no register and no frame binding is
/// found. wasm materializes every value in a local instead of a location, so
/// the equivalent binding is a prologue store: without it the never-written
/// local reads as the zero wasm initializes it to, silently substituting 0
/// for the folded constant at every use.
///
/// Only positions actually read as a plain OpRef are returned, so a trace
/// whose pool holds no such value emits no extra prologue instruction.
/// Scalar bits a folded producer still carries after it left the compiled
/// stream. `RegisterManager::loc` (dynasm `regalloc.rs`) recovers the same
/// payload from the box; flattening to `IntOp(pos)` and looking only at the
/// backend pool drops `_resint` / `_forwarded` (`history.py *FrontendOp`).
fn folded_scalar_bits(arg: &Operand) -> Option<i64> {
    // `get_value()` is the tracing observation. Seed only a proven fold:
    // `Forwarded::Const`, or `get_box_replacement` landing on an inline
    // constant / the constants map.
    let value = match arg.get_forwarded() {
        Forwarded::Const(c) => c.get(),
        _ => {
            let replaced = arg.get_box_replacement(false);
            if !replaced.is_constant() {
                return None;
            }
            replaced.const_value()?
        }
    };
    match value {
        Value::Ref(_) => None,
        value => Some(value.as_raw_i64()),
    }
}

/// Whether this JUMP lands on a LABEL in the same stream.
///
/// A leftover InputArg on a self-loop is dummy plumbing (dynasm parks it in
/// a frame slot). The same leftover on a JUMP whose descr is not a local
/// LABEL is a live transfer into another trace; synthesizing 0 there would
/// hand the target a null/zero instead of the runtime value.
fn jump_targets_local_label(ops: &[Op], jump: &Op) -> bool {
    let Some(descr) = jump.getdescr() else {
        return true;
    };
    ops.iter().any(|op| {
        op.opcode.is_label()
            && op
                .getdescr()
                .is_some_and(|label| std::sync::Arc::ptr_eq(&label, &descr))
    })
}

fn unbound_pool_const_seeds(
    inputargs: &[InputArgRc],
    ops: &[Op],
    constants: &indexmap::IndexMap<u32, i64>,
    num_vars: u32,
) -> Result<Vec<(u32, i64)>, BackendError> {
    use std::collections::HashSet;
    let mut defined: HashSet<u32> = inputargs.iter().map(|ia| ia.index).collect();
    for op in ops {
        if let Some(id) = result_value_raw(op) {
            defined.insert(id);
        }
        // `consider_label` / `LabelResumeData`: LABEL args are block
        // parameters. A peeled header carries loop live-ins as InputArgs
        // that are not portal inputargs and have no producing op in the
        // stream — they are defined at the LABEL, not missing. Declining
        // them made every peeled Python loop (`fib_loop`) fall back.
        //
        // A folded constant under the same position is different: it has
        // no producer *and* a constants-map entry. Marking it defined
        // skipped the prologue seed, so the local stayed the zero wasm
        // initializes it to. Seed those; only treat a LABEL arg as
        // defined when the pool has nothing to materialize.
        //
        // A producerless LABEL *RefOp* is a residual virtualizable
        // slot, not a phi. wasm locals start at zero; treating it as
        // defined compiles a null that failarg writeback stores as
        // bytecode state. Leave it unresolved so the trace declines.
        // A peeled InputArgRef live-in is a real LABEL parameter
        // (`consider_label`) and must stay defined.
        if op.opcode == OpCode::Label {
            for a in op.getarglist() {
                let r = a.to_opref();
                if let Some(id) = value_box_raw(r)
                    && !constants.contains_key(&id)
                    && !matches!(r, OpRef::RefOp(_))
                {
                    defined.insert(id);
                }
            }
        }
    }
    let mut seeds: Vec<(u32, i64)> = Vec::new();
    let mut unresolved: Vec<u32> = Vec::new();
    let mut readers: Vec<String> = Vec::new();
    let mut seen: HashSet<u32> = HashSet::new();
    // A 0-seeded Ref in a guard snapshot is a null identity / pycode at
    // deopt (`consume_vable_info` / `BytecodeCorruption`). LABEL/JUMP can
    // still carry a dummy leftover; failargs cannot.
    let mut failarg_stray_refs: HashSet<u32> = HashSet::new();
    for op in ops {
        if let Some(fa) = op.getfailargs() {
            // Only the live arguments `emit_guard_fail_args_spill` stores
            // are reads. Recording a logical hole here would make a later
            // JUMP leftover of the same id look unresolved.
            let live = live_fail_arg_mask(op.getdescr().as_ref(), fa.len());
            let extent = live_fail_arg_extent(op.getdescr().as_ref(), fa.len());
            for (i, a) in fa.iter().take(extent).enumerate() {
                if !live.get(i).copied().unwrap_or(true) {
                    continue;
                }
                if a.is_constant() {
                    continue;
                }
                let opref = a.to_opref();
                if opref == OpRef::NONE || opref.is_constant() {
                    continue;
                }
                if !matches!(opref, OpRef::InputArgRef(_)) && opref.ty() != Some(Type::Ref) {
                    continue;
                }
                let raw = opref.raw();
                if defined.contains(&raw) || inputargs.iter().any(|ia| ia.index == raw) {
                    continue;
                }
                failarg_stray_refs.insert(raw);
            }
        }
    }
    let mut consider =
        |op: &Op, slot: &str, a: &Operand, seeds: &mut Vec<(u32, i64)>, seen: &mut HashSet<u32>| {
            if a.is_constant() {
                return;
            }
            let opref = a.to_opref();
            if opref == OpRef::NONE || opref.is_constant() {
                return;
            }
            let raw = opref.raw();
            if raw >= num_vars || defined.contains(&raw) {
                return;
            }
            if !seen.insert(raw) {
                if unresolved.contains(&raw) {
                    readers.push(format!("{:?}.{slot} {opref:?}", op.opcode));
                }
                return;
            }
            if let Some(&bits) = constants.get(&raw) {
                seeds.push((raw, bits));
                return;
            }
            if let Some(bits) = folded_scalar_bits(a) {
                seeds.push((raw, bits));
                return;
            }
            // A peeled-loop fallthrough scan can append a resume live-in
            // (`assemble_peeled_trace_with_jump_args`) that is an InputArg
            // not in the token's input list and has no producer — leftover
            // of a residualized interior slot (`FrameLocalsRoot` keeps that
            // address out of compiled Ref homes). Dynasm `RegisterManager.loc`
            // allocates a dummy frame slot; seed a JUMP leftover with 0 so
            // the module is well-formed. LABEL args are skipped above
            // (`consider_label`). A body read of the same hole is a real
            // unbound operand and must decline. A Ref that also sits in
            // failargs is the virtualizable identity / pycode: compiling
            // null there panics at deopt, so decline and let the interpreter
            // run the loop.
            if a.is_inputarg() && !inputargs.iter().any(|ia| ia.index == raw) {
                if failarg_stray_refs.contains(&raw) {
                    unresolved.push(raw);
                    readers.push(format!("{:?}.{slot} {opref:?}", op.opcode));
                    return;
                }
                if op.opcode == OpCode::Jump && jump_targets_local_label(ops, op) {
                    seeds.push((raw, 0));
                    return;
                }
            }
            // No producer, no pool entry, no leftover box value: the local
            // would read as the zero wasm initializes it to. Decline the
            // trace (the interpreter runs it correctly, unaccelerated).
            unresolved.push(raw);
            readers.push(format!("{:?}.{slot} {opref:?}", op.opcode));
        };
    for op in ops {
        // rewrite.py `keep` — JIT_DEBUG / DebugMergePoint keep their
        // constants inline and never execute as values. LABEL args are
        // phi destinations (`consider_label` / `LabelResumeData`), not
        // reads. An unbound remint sitting only on those ops must not
        // decline the trace. A later real operand of the same id is
        // still considered.
        if op.opcode.is_label() || op.opcode.is_jit_debug() {
            continue;
        }
        for (i, a) in op.getarglist().iter().enumerate() {
            consider(op, &format!("arg{i}"), a, &mut seeds, &mut seen);
        }
        // Only the live arguments `emit_guard_fail_args_spill` stores are
        // reads. Logical holes, including trailing ones, own no spill slot.
        let fail_args = exit_fail_args(op);
        let live = live_fail_arg_mask(op.getdescr().as_ref(), fail_args.len());
        let extent = live_fail_arg_extent(op.getdescr().as_ref(), fail_args.len());
        if let Some(fa) = op.getfailargs() {
            for (i, a) in fa.iter().take(extent).enumerate() {
                if live.get(i).copied().unwrap_or(true) {
                    consider(op, &format!("fail{i}"), a, &mut seeds, &mut seen);
                }
            }
        }
    }
    if !unresolved.is_empty() {
        let labels: Vec<Vec<OpRef>> = ops
            .iter()
            .filter(|op| op.opcode == OpCode::Label)
            .map(|op| op.getarglist().iter().map(|a| a.to_opref()).collect())
            .collect();
        let sameas: Vec<OpRef> = ops
            .iter()
            .filter(|op| {
                matches!(
                    op.opcode,
                    OpCode::SameAsI | OpCode::SameAsR | OpCode::SameAsF
                )
            })
            .map(|op| op.pos().get())
            .collect();
        let in_idx: Vec<u32> = inputargs.iter().map(|ia| ia.index).collect();
        return Err(BackendError::Unsupported(format!(
            "wasm codegen: value{unresolved:?} read with no producing op and no \
             constant-pool entry; readers=[{}]; inputargs={in_idx:?} \
             labels={labels:?} sameas={sameas:?}",
            readers.join(" | "),
        )));
    }
    Ok(seeds)
}

/// Compile-time value of a constant operand (what `emit_resolve` would push
/// as `i64.const`), or `None` for a runtime value.
fn const_operand_value(constants: &indexmap::IndexMap<u32, i64>, opref: OpRef) -> Option<i64> {
    opref
        .is_constant()
        .then(|| resolve_const_bits(constants, opref))
}

/// `llsupport/regalloc.py valid_addressing_size`: the scales x86 SIB (and a
/// wasm `i32.shl`) can form without a multiply.
fn valid_addressing_size(size: u64) -> bool {
    matches!(size, 1 | 2 | 4 | 8)
}

/// `llsupport/regalloc.py get_scale`: 1,2,4,8 → shift 0,1,2,3.
fn get_scale(size: u64) -> u32 {
    debug_assert!(valid_addressing_size(size));
    if size < 4 {
        (size as u32) - 1
    } else {
        (size as u32 >> 2) + 1
    }
}

/// Scale the i32 index already on the stack by `item_size`.
///
/// `x86/assembler.py` getarrayitem skips the scale when `itemsize == 1`.
/// `valid_addressing_size` / `get_scale` turn 2/4/8 into `i32.shl`; every
/// other stride keeps `i32.mul`, matching the IMUL fallback of
/// `_imul_const_scaled`.
fn emit_scale_index(sink: &mut PeepSink<'_, '_>, item_size: u64) {
    if item_size == 1 {
        return;
    }
    if valid_addressing_size(item_size) {
        sink.i32_const(get_scale(item_size) as i32);
        sink.i32_shl();
    } else {
        sink.i32_const(item_size as i32);
        sink.i32_mul();
    }
}

/// Leave `base + index * item_size` on the wasm stack as an i32 address, and
/// return the remaining displacement (`extra_offset`).
///
/// A ConstInt index is folded into that displacement — `rewrite.py
/// emit_gc_load_or_indexed` picks the non-indexed `GC_LOAD` arm for the
/// same case. The header / interior-field offset stays in the access's own
/// MemArg, the same place the `getfield` arms put `field_offset_from_descr`.
fn emit_scaled_index_addr(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    base: OpRef,
    index: OpRef,
    item_size: u64,
    extra_offset: u64,
) -> u64 {
    emit_resolve(sink, constants, value_types, base);
    sink.i32_wrap_i64();
    // Only a constant that really lands inside the addressable range folds: a
    // MemArg displacement is unsigned and traps past the end of memory, so a
    // negative or overflowing index has to keep the run-time `i32` arithmetic,
    // which wraps instead.
    if let Some(idx) = const_operand_value(constants, index)
        && let Some(offset) = u64::try_from(idx)
            .ok()
            .and_then(|idx| idx.checked_mul(item_size))
            .and_then(|scaled| scaled.checked_add(extra_offset))
            .filter(|offset| u32::try_from(*offset).is_ok())
    {
        return offset;
    }
    emit_resolve(sink, constants, value_types, index);
    sink.i32_wrap_i64();
    emit_scale_index(sink, item_size);
    sink.i32_add();
    extra_offset
}

/// Leave `base + index * item_size` on the wasm stack as an i32 address, and
/// return the `base_size` displacement the access still owes.
fn emit_array_addr(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    op: &Op,
) -> u64 {
    let (base_size, item_size) = op
        .with_array_descr(|ad| (ad.base_size() as u64, ad.item_size() as u64))
        .unwrap_or_else(|| missing_layout_descr("array descr (base/item size)", op));
    emit_scaled_index_addr(
        sink,
        constants,
        value_types,
        op.arg(0).to_opref(),
        op.arg(1).to_opref(),
        item_size,
        base_size,
    )
}

/// The width of a GC pointer in the wasm32 guest heap.
///
/// A descriptor mints its width from the compiling target, so for a pyre descr
/// this is already what `field_size` / `item_size` says. It stays as its own
/// constant for the descr that does not: a pointer is four bytes here whatever
/// the descr claims, and a reading arm and its writing arm must take the width
/// from ONE place — the two spellings agreeing today is not the same as them
/// being one rule.
const GUEST_PTR_SIZE: usize = 4;

/// The width an access to one array item moves, and how a read of it extends.
/// Every remaining `GETARRAYITEM_RAW_R` arm reads it from here.
///
/// The address stride still comes from the descriptor's own `item_size`
/// (`emit_array_addr`), which is what the allocation laid the array out with.
fn array_item_access_size_sign(op: &Op) -> (usize, bool) {
    op.with_array_descr(|ad| {
        if ad.is_array_of_pointers() {
            (GUEST_PTR_SIZE, false)
        } else {
            (ad.item_size(), ad.is_item_signed())
        }
    })
    .unwrap_or_else(|| missing_layout_descr("array descr (item size/sign)", op))
}

// ── Guard emission helpers ──

#[derive(Clone, Copy)]
struct InlineGuard<'a> {
    guard_idx: u32,
    inputargs: &'a [InputArgRc],
    /// Ordinal within this region's family, region 0 attached first. NOT a
    /// branch depth on its own: a family's blocks close one per region as the
    /// walk reaches each region's ops, so the depth of region N's block is this
    /// ordinal less however many of the family have already closed where the
    /// branching guard sits. `BridgeDispatch` carries that running count, and
    /// `emit_guard_exit` does the subtraction.
    region_ordinal: u32,
    outside_loop: bool,
}

#[derive(Clone, Copy)]
struct BridgeDispatch<'a> {
    cells_base: u32,
    /// Per-guard cell addresses, indexed by `guard_idx - fail_index_base`.
    /// A zero entry falls back to `cells_base + index * 4`.
    cell_addrs: &'a [u32],
    fail_index_base: u32,
    bridge_slot_local: u32,
    enabled: bool,
    /// `arity -> indirect-call type` for armed parameter dispatch. Every
    /// signature carries values as i64, including Float bit patterns.
    param_type_indices: &'a indexmap::IndexMap<usize, u32>,
    inline_guards: &'a [InlineGuard<'a>],
    /// Depth of the innermost still-open preamble-region block, at the
    /// statement level of the operation being emitted.
    outside_region_base: u32,
    /// Regions of each family whose block has already been closed at the
    /// operation being emitted — one closes at each region's first op. A guard
    /// in the owner's own stream sees zero of both; a guard nested inside
    /// region P sees P+1 of P's family.
    closed_body_regions: u32,
    closed_outside_regions: u32,
    ref_homes: &'a RefHomes,
    frame: FrameGeometry,
    /// The trace's one GUARD_VALUE counter slot (`counter_slot`), or `None`
    /// when no guard needs one.
    counter_slot: Option<u64>,
    /// Fail-argument count -> the module function that spills that many
    /// arguments, for the counts `spill_helper_arities` admitted. An exit whose
    /// count is absent writes its own stores.
    spill_helpers: &'a indexmap::IndexMap<usize, u32>,
    /// See [`CaParams::exit_table_base`].
    exit_table_base: u32,
    /// Preamble `LoadFromGcTable` results (and SameAs of them) that a
    /// later guard may spill. Keyed by value id; the pair is the baked
    /// table base and slot index.
    gc_table_slots: &'a HashMap<u32, (u32, i64)>,
    /// ConstPtr compile keys of every GC table this function emits, and the
    /// base of the table that owns the operation currently being emitted.
    const_tables: &'a ConstPtrTables,
    const_table_base: u32,
    /// See [`CaParams::attached`].
    attached: majit_backend::AttachedDescrPtrs,
    /// Outlined cold exit, when the module admitted one helper for every
    /// home-backed spill. `None` keeps each exit inline.
    guard_exit: Option<GuardExitOutline<'a>>,
}

fn emit_guard_true(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    guard_idx: u32,
    op: &Op,
    block_exit_depth: u32,
    dispatch: BridgeDispatch<'_>,
) {
    emit_resolve(sink, constants, value_types, op.arg(0).to_opref());
    sink.i64_eqz();
    emit_guard_if_exit(
        sink,
        constants,
        value_types,
        guard_idx,
        op,
        block_exit_depth,
        dispatch,
    );
}

fn emit_guard_false(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    guard_idx: u32,
    op: &Op,
    block_exit_depth: u32,
    dispatch: BridgeDispatch<'_>,
) {
    emit_resolve(sink, constants, value_types, op.arg(0).to_opref());
    sink.i64_const(0);
    sink.i64_ne();
    emit_guard_if_exit(
        sink,
        constants,
        value_types,
        guard_idx,
        op,
        block_exit_depth,
        dispatch,
    );
}

/// `llsupport/regalloc.py next_op_can_accept_cc` — the comparison at `i`
/// may hand its condition straight to the op at `i + 1` instead of
/// materialising a boolean, when that op is the condition's only reader. x86
/// leaves the condition in the flags (`x86/regalloc.py force_allocate_reg_or_cc
/// force_allocate_reg_or_cc`, ported to the dynasm sibling at
/// `next_op_can_accept_cc` in `majit-backend-dynasm/src/regalloc.rs`); wasm's
/// operand stack plays that role — [`push_cond`]'s i32 stays on the stack and
/// the guard's `if` tests it, so the `i64.extend_i32_u`/`local.set` and the
/// guard's own `local.get`/re-test disappear.
///
/// Matches the dynasm port: `GuardTrue`/`GuardFalse`/`GuardIsnull`/
/// `GuardNonnull`, whose wasm arms do nothing but re-test the boolean,
/// and `CondCallN`/`CondCallValue*`, whose predicate is the same word
/// and whose arm can consume the i32 directly.
fn next_op_can_accept_cc<'a>(
    ops: &'a [Op],
    i: usize,
    result: OpRef,
    liveness: &HomeLiveness,
    label_resume: &LabelResumeData,
    ref_homes: &RefHomes,
) -> Option<&'a Op> {
    if result == OpRef::NONE || result.is_constant() {
        return None;
    }
    let next_op = ops.get(i + 1)?;
    if !matches!(
        next_op.opcode,
        OpCode::GuardTrue
            | OpCode::VecGuardTrue
            | OpCode::GuardFalse
            | OpCode::VecGuardFalse
            | OpCode::GuardIsnull
            | OpCode::GuardNonnull
            | OpCode::CondCallN
            | OpCode::CondCallValueI
            | OpCode::CondCallValueR
    ) {
        return None;
    }
    // history.py `Const.is_constant()` — a Const operand is not an
    // op-result identity, so comparing raw positions against it is invalid.
    if next_op.num_args() == 0 || next_op.arg(0).is_constant() {
        return None;
    }
    if next_op.arg(0).to_opref().raw() != result.raw() {
        return None;
    }
    // BaseRegalloc.next_op_can_accept_cc: COND_CALL's callee and arguments
    // still read locals, even when their last use is this same operation.
    if matches!(
        next_op.opcode,
        OpCode::CondCallN | OpCode::CondCallValueI | OpCode::CondCallValueR
    ) && (1..next_op.num_args()).any(|arg| next_op.arg(arg).to_opref() == result)
    {
        return None;
    }
    // Any later reader (including this guard's own fail args, which
    // `HomeLiveness` records as uses at `i + 1`) needs the materialised local.
    if liveness.last_use(result.raw()) > i as i32 + 1 {
        return None;
    }
    if next_op
        .getfailargs()
        .is_some_and(|fa| fa.iter().any(|a| a.to_opref() == result))
    {
        return None;
    }
    // A LABEL resume loader restores its capture set from the frame, so a
    // captured value must have been bound; skipping the `local.set` would leave
    // wasm's zero-init in its place.
    if label_resume.storage(result).is_some() {
        return None;
    }
    // The store-on-def tail reads the result local for a Ref-homed value. A
    // comparison result is never a Ref, so this only pins the invariant.
    if ref_homes.home(result).is_some() {
        return None;
    }
    Some(next_op)
}

/// The overflow guard immediately following an overflow op. The flag is not an
/// SSA value — it has no local, no home slot, no LABEL capture, and can never be
/// a fail argument — so unlike `next_op_can_accept_cc` this needs no liveness
/// test: adjacency is the whole condition.
fn next_ovf_guard(ops: &[Op], i: usize) -> Option<&Op> {
    let next = ops.get(i + 1)?;
    matches!(next.opcode, OpCode::GuardNoOverflow | OpCode::GuardOverflow).then_some(next)
}

/// Common guard exit: condition is on stack (i32), spill and leave on failure.
///
/// The spill belongs in this arm rather than in one shared exit handler after
/// the trace. x86/assembler.py `write_pending_failure_recoveries` can
/// place its recovery stubs after the hot code because `GuardToken.fail_locs`
/// (llsupport/assembler.py) freezes the register or stack location the
/// allocator gave each fail argument *at the guard*, so a stub reads a fixed
/// home and nothing keeps the value live past its own guard. Wasm has no way
/// to record such a location: the allocator is the engine's, and it derives
/// liveness from where the emitted code reads a local. Routing every guard to
/// one handler block therefore makes it a join point whose live-in set is the
/// union of every guard's fail arguments, so each becomes live at every guard
/// in the trace, and the resulting long ranges spill in the hot body. Reading
/// them here ends each range at the guard that needs it.
///
/// `block_exit_depth` is the statement-level depth of the enclosing exit
/// `block` (preamble = 0, loop body = 1); the `+ 1` accounts for the `if`
/// this opens. The stores run only on the failing edge, so the fallthrough
/// carries no frame traffic. With bridge dispatch enabled, the failing arm
/// writes its fail index to `frame[0]`, records its constant bridge-cell
/// address in a local, and branches to the shared dispatch epilogue.
fn emit_guard_if_exit(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    guard_idx: u32,
    op: &Op,
    block_exit_depth: u32,
    dispatch: BridgeDispatch<'_>,
) {
    sink.if_(BlockType::Empty);
    emit_guard_exit(
        sink,
        constants,
        value_types,
        guard_idx,
        op,
        block_exit_depth + 1,
        dispatch,
        1,
    );
    sink.end();
}

/// Branch depth from the emitting instruction out to the `block` opened for
/// `inline`'s region.
///
/// A family's blocks close one per region as the walk reaches each region's
/// ops, so the ordinal counts from whichever of them are still open here.
/// `build_function` refuses any region whose source guard does not precede its
/// own ops, which is what keeps this subtraction from going negative.
///
/// `enclosing_frames` is what the caller opened between those blocks and this
/// instruction — one for the failing `if` of a conditional guard, none for an
/// exit emitted at statement level. Assuming the `if` unconditionally sends a
/// statement-level exit one frame too far out: for a region inside the header
/// `loop`, to the `loop` itself, which turns the exit into a back edge over the
/// owner's ops alone and leaves the region unreachable.
fn inline_region_br_depth(
    inline: &InlineGuard<'_>,
    dispatch: &BridgeDispatch<'_>,
    enclosing_frames: u32,
) -> u32 {
    let ordinal = if inline.outside_loop {
        inline.region_ordinal - dispatch.closed_outside_regions
    } else {
        inline.region_ordinal - dispatch.closed_body_regions
    };
    let depth = if inline.outside_loop {
        dispatch.outside_region_base + ordinal
    } else {
        ordinal
    };
    depth + enclosing_frames
}

fn emit_guard_exit(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    guard_idx: u32,
    op: &Op,
    block_exit_depth: u32,
    dispatch: BridgeDispatch<'_>,
    enclosing_frames: u32,
) {
    if let Some(inline) = dispatch
        .inline_guards
        .iter()
        .find(|g| g.guard_idx == guard_idx)
    {
        emit_guard_inline_bridge_move(
            sink,
            constants,
            value_types,
            dispatch.ref_homes,
            dispatch.frame,
            op,
            inline.inputargs,
            dispatch.gc_table_slots,
            dispatch.const_tables,
            dispatch.const_table_base,
        );
        sink.br(inline_region_br_depth(inline, &dispatch, enclosing_frames));
        return;
    }
    // A parameter signature does not arm a bridge: the dispatch cell array
    // must exist too. The native backend harness has signatures but no wasm
    // cells, and must use the ordinary guard recovery path.
    if !dispatch.enabled || dispatch.param_type_indices.is_empty() {
        emit_guard_spill(
            sink,
            constants,
            value_types,
            guard_idx,
            op,
            dispatch.counter_slot,
            dispatch.spill_helpers,
            dispatch.gc_table_slots,
            dispatch.ref_homes,
            dispatch.frame,
            dispatch.const_tables,
            dispatch.const_table_base,
            dispatch,
        );
        if dispatch.enabled {
            emit_guard_bridge_dispatch(sink, guard_idx, dispatch);
        }
    } else {
        emit_guard_param_tail_call(sink, constants, value_types, guard_idx, op, dispatch);
        // A missing cell keeps the historical recovery path. It is deliberately
        // after the cell test so a bridge crossing performs no frame spill.
        emit_guard_spill(
            sink,
            constants,
            value_types,
            guard_idx,
            op,
            dispatch.counter_slot,
            dispatch.spill_helpers,
            dispatch.gc_table_slots,
            dispatch.ref_homes,
            dispatch.frame,
            dispatch.const_tables,
            dispatch.const_table_base,
            dispatch,
        );
    }
    sink.br(block_exit_depth);
}

/// Tail-call this guard's bridge directly when its cell is armed. The guard's
/// failure list fixes both the values and the wasm function type, so this path
/// needs neither an arity tag nor staging locals.
/// A guard op's fail args restricted to the positions its bridge received.
///
/// Same rule as `live_fail_arg_mask`, read off the op's own descr.
fn live_fail_args_of(op: &Op) -> Vec<OpRef> {
    let all: Vec<OpRef> = op
        .getfailargs()
        .map(|args| args.iter().map(|arg| arg.to_opref()).collect::<Vec<_>>())
        .unwrap_or_else(|| op.getarglist().iter().map(|arg| arg.to_opref()).collect());
    let descr = op.getdescr();
    let mask = live_fail_arg_mask(descr.as_ref(), all.len());
    all.into_iter()
        .zip(mask)
        .filter_map(|(arg, live)| live.then_some(arg))
        .collect()
}

fn dispatch_cell_addr(dispatch: BridgeDispatch<'_>, guard_idx: u32) -> u32 {
    let index = guard_idx.wrapping_sub(dispatch.fail_index_base) as usize;
    if let Some(&addr) = dispatch.cell_addrs.get(index) {
        if addr != 0 {
            return addr;
        }
    }
    if dispatch.cells_base == 0 {
        return 0;
    }
    dispatch.cells_base + index as u32 * std::mem::size_of::<u32>() as u32
}

fn emit_guard_param_tail_call(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    guard_idx: u32,
    op: &Op,
    dispatch: BridgeDispatch<'_>,
) {
    let fail_args: Vec<OpRef> = live_fail_args_of(op);
    let arity = fail_args.len();
    let type_idx = *dispatch
        .param_type_indices
        .get(&arity)
        .expect("parameter dispatch type missing for guard fail arity");
    debug_assert!(dispatch.enabled);
    debug_assert!(guard_idx >= dispatch.fail_index_base);
    let cell_addr = dispatch_cell_addr(dispatch, guard_idx);
    if cell_addr == 0 {
        return;
    }
    sink.i32_const(cell_addr as i32);
    sink.i32_load(memarg(0, 2));
    sink.local_tee(dispatch.bridge_slot_local);
    sink.if_(BlockType::Empty);
    sink.local_get(0);
    for arg in fail_args {
        if arg.ty() == Some(Type::Float) {
            emit_resolve_f64(sink, constants, value_types, arg);
            sink.i64_reinterpret_f64();
        } else {
            emit_resolve_failarg(
                sink,
                constants,
                value_types,
                arg,
                dispatch.gc_table_slots,
                dispatch.ref_homes,
                dispatch.frame,
                dispatch.const_tables,
                dispatch.const_table_base,
            );
        }
    }
    sink.local_get(dispatch.bridge_slot_local);
    sink.return_call_indirect(0, type_idx);
    sink.end();
}

/// Transfer a failing guard directly into an inlined bridge.  All sources are
/// pushed before any destination local is written, preserving parallel-move
/// semantics when fail arguments overlap bridge input locals.
fn emit_guard_inline_bridge_move(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    ref_homes: &RefHomes,
    frame: FrameGeometry,
    op: &Op,
    inputargs: &[InputArgRc],
    gc_table_slots: &HashMap<u32, (u32, i64)>,
    const_tables: &ConstPtrTables,
    const_table_base: u32,
) {
    let fail_args: Vec<OpRef> = live_fail_args_of(op);
    assert_eq!(
        fail_args.len(),
        inputargs.len(),
        "guard and bridge input arity diverged"
    );
    for (arg, input) in fail_args.iter().zip(inputargs) {
        if value_types.ty(input.index) == ValType::F64 {
            emit_resolve_f64(sink, constants, value_types, *arg);
        } else {
            emit_resolve_failarg(
                sink,
                constants,
                value_types,
                *arg,
                gc_table_slots,
                ref_homes,
                frame,
                const_tables,
                const_table_base,
            );
        }
    }
    for input in inputargs.iter().rev() {
        sink.local_set(value_types.local(input.index));
    }
    for input in inputargs {
        if let Some(home) = ref_homes.home_id(input.index) {
            sink.local_get(0);
            sink.local_get(value_types.local(input.index));
            sink.i64_store(mem64(frame.home_ofs(home as u64)));
        }
    }
}

fn emit_guard_bridge_dispatch(
    sink: &mut PeepSink<'_, '_>,
    guard_idx: u32,
    dispatch: BridgeDispatch<'_>,
) {
    debug_assert!(guard_idx >= dispatch.fail_index_base);
    let cell_addr = dispatch_cell_addr(dispatch, guard_idx);
    if cell_addr == 0 {
        sink.i32_const(0);
        sink.local_set(dispatch.bridge_slot_local);
        return;
    }
    sink.i32_const(cell_addr as i32);
    sink.local_set(dispatch.bridge_slot_local);
}

/// Opcodes whose assembler publishes `jf_force_descr` before the call.
///
/// `aarch64/assembler.rs _store_force_index_if_next_guard` runs only for
/// `CallMayForce*`, `CallReleaseGil*`, and `CallAssembler*`. A plain `CallN`
/// followed by `GuardNotForced` does not arm the frame. Arming it anyway
/// makes a callee that reads `f_lineno` (`force` once `jf_force_descr` is
/// set) fail that guard on every iteration, and `is_guard_forced` never
/// bridges the exit.
fn call_publishes_force_descr(opcode: OpCode) -> bool {
    matches!(
        opcode,
        OpCode::CallMayForceI
            | OpCode::CallMayForceR
            | OpCode::CallMayForceF
            | OpCode::CallMayForceN
            | OpCode::CallReleaseGilI
            | OpCode::CallReleaseGilF
            | OpCode::CallReleaseGilN
            | OpCode::CallAssemblerI
            | OpCode::CallAssemblerR
            | OpCode::CallAssemblerF
            | OpCode::CallAssemblerN
    )
}

/// x86 `_store_force_index_if_next_guard`: a call that may force is bracketed
/// by the `GUARD_NOT_FORCED` immediately after it, so publish that guard's
/// coordinate before the call runs.
fn emit_force_bracket_before_call(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    ref_homes: &RefHomes,
    frame: FrameGeometry,
    ops: &[Op],
    op_idx: usize,
    guard_idx: u32,
    dispatch: BridgeDispatch<'_>,
) {
    if !call_publishes_force_descr(ops[op_idx].opcode) {
        return;
    }
    let Some(next_op) = ops.get(op_idx + 1) else {
        return;
    };
    if !matches!(
        next_op.opcode,
        OpCode::GuardNotForced | OpCode::GuardNotForced2
    ) {
        return;
    }
    // Everything the guard names is defined by an op at or before the call --
    // except the call's own result, whose local still holds the PREVIOUS
    // iteration's value here.
    emit_force_arm(
        sink,
        constants,
        value_types,
        ref_homes,
        frame,
        next_op,
        exit_index(next_op, guard_idx, dispatch.attached),
        Some(ops[op_idx].pos().get().raw()),
        dispatch.const_tables,
        dispatch.const_table_base,
        dispatch.exit_table_base,
        guard_idx.wrapping_sub(dispatch.fail_index_base),
    );
}

/// x86 `store_force_descr` / `_store_force_index`: publish where a force that
/// lands while this frame is still reachable reads its state from — upstream
/// writes the guard's descr into `jf_force_descr` and its fail arguments into
/// the frame. Publish the same coordinate here: the guard's exit index plus
/// one in `jf_force_descr`, and its fail arguments in force-only locations.
/// This is written unconditionally, not on a failure branch, because the reader
/// runs while the bracketed call is still on the stack.
///
/// A Ref argument is published as its **home slot offset**, tagged
/// `offset * 2 + 1`, rather than as its value. The force slots are not in
/// `build_home_gcmap`'s traced set — that set is type-precise, and blanket
/// marking a slot that holds a scalar would offer the collector an integer to
/// mistake for a nursery address — so a Ref value copied here would not be
/// forwarded by a collection the bracketed call performs, and
/// `dead_frame_from_forced_frame` would read a from-space address. The home
/// slot IS traced and holds the same value, so naming it survives the
/// collection. Ref pointers are 8-aligned, which is what makes the low tag bit
/// free to tell an offset from a value.
///
/// A non-null `ConstPtr` has no home. Publishing its compile-time address
/// into the untraced force slot goes stale if the bracketed call collects.
/// Homes use
/// `offset * 2 + 1` (bit 0 set; bit 1 is clear because `offset` is
/// 8-aligned). A table slot is published as `abs_addr | 3` so consume
/// reloads the forwarded table entry after the collection. `undefined`
/// and a null constant still publish a literal 0.
#[allow(clippy::too_many_arguments)]
fn emit_force_arm(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    ref_homes: &RefHomes,
    frame: FrameGeometry,
    guard_op: &Op,
    exit_idx: u32,
    undefined: Option<u32>,
    const_tables: &ConstPtrTables,
    const_table_base: u32,
    exit_table_base: u32,
    exit_local: u32,
) {
    // `counter_value_spill` answers `None` for anything but a GUARD_VALUE, so
    // the counter slot has nothing to contribute to a force bracket.
    //
    // The force reader follows the same per-descriptor locations as normal
    // deopt; logical holes must not consume physical force slots either.
    let force_args = live_exit_fail_args(guard_op);
    for (i, &arg_ref) in force_args.iter().enumerate() {
        sink.local_get(0);
        // Inline-Const failargs have no `raw()` index (`ConstPtr` panics).
        if !arg_ref.is_constant() && undefined == Some(arg_ref.raw()) {
            sink.i64_const(0);
        } else if let Some(home) = ref_homes.home(arg_ref) {
            let ofs = frame.home_ofs(home as u64);
            sink.i64_const((ofs as i64) * 2 + 1);
        } else if let Some(g) = arg_ref.as_const_ptr() {
            if g.is_null() {
                sink.i64_const(0);
            } else if let Some((base, index)) = const_tables.slot(const_table_base, g.0) {
                // Tag the GC-table slot so `dead_frame_from_forced_frame`
                // reloads after a collection inside the bracketed call.
                let addr = i64::from(base)
                    + i64::from(index) * std::mem::size_of::<majit_ir::GcRef>() as i64;
                sink.i64_const(addr | 3);
            } else {
                emit_resolve(sink, constants, value_types, arg_ref);
            }
        } else {
            emit_resolve(sink, constants, value_types, arg_ref);
        }
        sink.i64_store(mem64(frame.force_slot_ofs(i as u64)));
    }
    // x86 `store_force_descr`: the guard's descr cell, separate from `jf_descr`.
    // Zero remains the unarmed sentinel. `force` copies this word into `jf_descr`.
    if exit_table_base == 0 {
        sink.local_get(0);
        sink.i32_const(majit_backend::jitframe::FIRST_ITEM_OFFSET as i32);
        sink.i32_sub();
        sink.i32_const(exit_idx.wrapping_add(1) as i32);
        sink.i32_store(memarg(
            majit_backend::jitframe::JF_FORCE_DESCR_OFS as u64,
            2,
        ));
        sink.local_get(0);
        sink.i64_const(exit_idx as i64);
        sink.i64_store(mem64(0));
    } else {
        let _ = exit_idx;
        emit_store_loaded_header(
            sink,
            majit_backend::jitframe::JF_FORCE_DESCR_OFS as u64,
            exit_table_base,
            exit_local as usize,
            0,
        );
    }
}

fn emit_outlined_guard_exit(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    op: &Op,
    guard_idx: u32,
    dispatch: BridgeDispatch<'_>,
) -> bool {
    let Some(outline) = dispatch.guard_exit else {
        return false;
    };
    let local = guard_idx.wrapping_sub(dispatch.fail_index_base) as usize;
    let Some(&addr) = outline.addrs.get(local) else {
        return false;
    };
    if addr == 0 {
        return false;
    }
    sink.local_get(0);
    sink.i32_const(addr as i32);
    if let Some(operand) = counter_value_spill(op, &exit_fail_args(op))
        .zip(dispatch.counter_slot)
        .map(|(operand, _)| operand)
    {
        emit_resolve_failarg(
            sink,
            constants,
            value_types,
            operand,
            dispatch.gc_table_slots,
            dispatch.ref_homes,
            dispatch.frame,
            dispatch.const_tables,
            dispatch.const_table_base,
        );
    } else {
        sink.i64_const(0);
    }
    sink.call(outline.func);
    true
}

fn emit_guard_spill(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    guard_idx: u32,
    op: &Op,
    counter_slot: Option<u64>,
    spill_helpers: &indexmap::IndexMap<usize, u32>,
    gc_table_slots: &HashMap<u32, (u32, i64)>,
    ref_homes: &RefHomes,
    frame: FrameGeometry,
    const_tables: &ConstPtrTables,
    const_table_base: u32,
    dispatch: BridgeDispatch<'_>,
) {
    if emit_outlined_guard_exit(sink, constants, value_types, op, guard_idx, dispatch) {
        return;
    }
    emit_guard_fail_args_spill(
        sink,
        constants,
        value_types,
        op,
        counter_slot,
        spill_helpers,
        gc_table_slots,
        ref_homes,
        frame,
        const_tables,
        const_table_base,
    );
    emit_guard_fail_index_store(sink, dispatch, op, guard_idx);
    // assembler.py `_build_failure_recovery` (exc=True): stage pos_exc_value
    // into jf_guard_exc and clear pos_exception / pos_exc_value. Only the
    // failing arm reaches here; a bridge tail-call returns before this spill.
    emit_store_guard_exc(sink, op);
}

/// `llsupport/assembler.py` `must_save_exception`: GUARD_EXCEPTION,
/// GUARD_NO_EXCEPTION, GUARD_NOT_FORCED. `grab_exc_value` reads `jf_guard_exc`.
fn emit_store_guard_exc(sink: &mut PeepSink<'_, '_>, op: &Op) {
    if !matches!(
        op.opcode,
        OpCode::GuardNoException | OpCode::GuardException | OpCode::GuardNotForced
    ) {
        return;
    }
    emit_store_guard_exc_raw(sink);
}

/// `llsupport/assembler.py` `must_save_exception` body. The outlined helper
/// calls this when the record's exc flag is set; the opcode test stays at
/// the inline site.
fn emit_store_guard_exc_raw(sink: &mut PeepSink<'_, '_>) {
    let exc_value_addr = runtime_addr(crate::jit_exc_value_addr);
    let exc_type_addr = runtime_addr(crate::jit_exc_type_addr);
    sink.local_get(0);
    sink.i32_const(majit_backend::jitframe::FIRST_ITEM_OFFSET as i32);
    sink.i32_sub();
    sink.i32_const(exc_value_addr);
    sink.i64_load(mem64(0));
    sink.i32_wrap_i64();
    sink.i32_store(memarg(majit_backend::jitframe::JF_GUARD_EXC_OFS as u64, 2));
    sink.i32_const(exc_value_addr);
    sink.i64_const(0);
    sink.i64_store(mem64(0));
    sink.i32_const(exc_type_addr);
    sink.i64_const(0);
    sink.i64_store(mem64(0));
}

/// The exit a failing `op` writes into `frame[0]`.
///
/// `compile_done_with_this_frame` / `compile_exit_frame_with_exception` stamp
/// the FINISH with the singleton the cpu was handed, so every trace that
/// finishes the same way names the same exit and `_call_assembler_check_descr`
/// can recognise it with one compare. Guards, and the N-ary finishes pyre adds
/// on top of the `_DoneWithThisFrameDescr` family's 0/1-result classes, have no
/// shared identity and keep their own exit.
fn exit_index(op: &Op, guard_idx: u32, attached: majit_backend::AttachedDescrPtrs) -> u32 {
    if op.opcode != OpCode::Finish {
        return guard_idx;
    }
    crate::failguard::attached_finish_exit_index(&attached, &op.getdescr()).unwrap_or(guard_idx)
}

fn emit_guard_fail_args_spill(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    op: &Op,
    counter_slot: Option<u64>,
    spill_helpers: &indexmap::IndexMap<usize, u32>,
    gc_table_slots: &HashMap<u32, (u32, i64)>,
    ref_homes: &RefHomes,
    frame: FrameGeometry,
    const_tables: &ConstPtrTables,
    const_table_base: u32,
) {
    // Store the live values at their descriptor's physical locations. A hole
    // in ResumeDataLoopMemo's numbering is not a frame location.
    let fail_args = live_exit_fail_args(op);

    // The shared function writes the same slots in the same order; the call
    // site pushes the frame pointer once and each value once. The counter slot
    // below is per-trace rather than positional, so it stays inline.
    if let Some(&helper) = spill_helpers.get(&fail_args.len()) {
        sink.local_get(0);
        for &arg_ref in &fail_args {
            emit_resolve_failarg(
                sink,
                constants,
                value_types,
                arg_ref,
                gc_table_slots,
                ref_homes,
                frame,
                const_tables,
                const_table_base,
            );
        }
        sink.call(helper);
    } else {
        for (i, &arg_ref) in fail_args.iter().enumerate() {
            let offset = frame.spill_slot_ofs(i as u64);
            sink.local_get(0);
            emit_resolve_failarg(
                sink,
                constants,
                value_types,
                arg_ref,
                gc_table_slots,
                ref_homes,
                frame,
                const_tables,
                const_table_base,
            );
            sink.i64_store(mem64(offset));
        }
    }
    if let Some((operand, slot)) = counter_value_spill(op, &exit_fail_args(op)).zip(counter_slot) {
        let offset = frame.spill_slot_ofs(slot);
        sink.local_get(0);
        emit_resolve_failarg(
            sink,
            constants,
            value_types,
            operand,
            gc_table_slots,
            ref_homes,
            frame,
            const_tables,
            const_table_base,
        );
        sink.i64_store(mem64(offset));
    }
}

/// The GUARD_VALUE operand `op` must park in the trace's counter slot so that
/// `make_a_counter_per_value`'s index is readable, or `None` when the operand
/// already occupies a live fail-argument slot, or is a constant the optimizer has
/// already decided the guard on.
///
/// `regalloc.py prepare_op_guard_value` hands `cpu.all_reg_indexes[arg.value]`
/// — the operand's index in the register save area every guard exit writes, so
/// upstream always has a slot for a box the guard does not carry among its
/// fail args. An exit here writes its fail args and nothing else, so the
/// operand needs a slot of its own.
///
/// That slot is ONE per trace (`counter_slot`), past every exit's fail args
/// and past the inputargs. A per-guard "one past MY fail args" index would
/// land inside a wider guard's fail-arg range, where the parked word would be
/// read back as that guard's fail argument.
fn counter_value_spill(op: &Op, fail_args: &[OpRef]) -> Option<OpRef> {
    if op.opcode != OpCode::GuardValue {
        return None;
    }
    let arg0 = op.arg(0).to_opref();
    if arg0 == OpRef::NONE
        || arg0.is_constant()
        || live_fail_arg_position(op, fail_args, arg0).is_some()
    {
        return None;
    }
    Some(arg0)
}

/// Find a readable logical position, skipping holes even when they still
/// name the same box. Like x86 regalloc.consider_guard_value, the counter
/// must name a location whose exit actually saves the compared value.
fn live_fail_arg_position(op: &Op, fail_args: &[OpRef], value: OpRef) -> Option<usize> {
    let live = live_fail_arg_mask(op.getdescr().as_ref(), fail_args.len());
    fail_args
        .iter()
        .zip(live)
        .position(|(&arg, live)| live && arg == value)
}

/// This op's fail arguments as the exit writes them, in slot order.
fn exit_fail_args(op: &Op) -> Vec<OpRef> {
    op.getfailargs()
        .map(|fa| fa.iter().map(|a| a.to_opref()).collect())
        .unwrap_or_else(|| op.getarglist().iter().map(|a| a.to_opref()).collect())
}

/// x86/assembler.py `genop_discard_check_memory_error`: the NULL test
/// `rewrite.py` `_gen_call_malloc_gc` attaches to every collecting malloc.
/// Address 0 is ordinary linear memory here, so stores that follow an
/// allocation would corrupt it silently rather than fault.
///
/// The failing arm is `_build_propagate_exception_path`:
/// `_store_and_reset_exception` moves the `MemoryError` the allocation
/// helper published (`lib.rs` `oom_signal_if_zero`) into `jf_guard_exc`,
/// `jf_descr` becomes `propagate_exception_descr`, and the function returns
/// its frame pointer. The metainterp reader runs
/// `PropagateExceptionDescr.handle_fail` (null cell → `memory_error`).
/// It returns rather than branching to the hot-exit block because the
/// epilogue there dispatches on a per-guard bridge cell, and this exit
/// belongs to no guard.
///
/// A collecting helper can move the frame before it fails, so local 0 is
/// reloaded on this arm; the reload reads the shadow-stack top, so a caller
/// that already reloaded pays nothing for the second one.
///
/// A cpu that was never handed `propagate_exception_descr` traps, the same
/// choice the cranelift sibling's `emit_memory_error_check` makes for an
/// unattached descr.
fn emit_memory_error_check(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    value: OpRef,
    residual_type_base: Option<u32>,
    ca_reload_fn_ptr: i64,
    jf_top_addr: Option<u32>,
    propagate_exception_descr: usize,
) {
    emit_resolve(sink, constants, value_types, value);
    sink.i64_eqz();
    emit_memory_error_on_truthy(
        sink,
        residual_type_base,
        ca_reload_fn_ptr,
        jf_top_addr,
        propagate_exception_descr,
    );
}

fn emit_memory_error_if_i32_zero(
    sink: &mut PeepSink<'_, '_>,
    residual_type_base: Option<u32>,
    ca_reload_fn_ptr: i64,
    jf_top_addr: Option<u32>,
    propagate_exception_descr: usize,
) {
    sink.i32_eqz();
    emit_memory_error_on_truthy(
        sink,
        residual_type_base,
        ca_reload_fn_ptr,
        jf_top_addr,
        propagate_exception_descr,
    );
}

fn emit_memory_error_on_truthy(
    sink: &mut PeepSink<'_, '_>,
    residual_type_base: Option<u32>,
    ca_reload_fn_ptr: i64,
    jf_top_addr: Option<u32>,
    propagate_exception_descr: usize,
) {
    sink.if_(BlockType::Empty);
    if propagate_exception_descr != 0 {
        emit_reload_frame_if_necessary(sink, residual_type_base, ca_reload_fn_ptr, jf_top_addr);
        // `_store_and_reset_exception`: JIT_EXC_VALUE → jf_guard_exc, then
        // clear both globals. `grab_exc_value` reads jf_guard_exc.
        sink.local_get(0);
        sink.i32_const(majit_backend::jitframe::FIRST_ITEM_OFFSET as i32);
        sink.i32_sub();
        sink.i32_const(runtime_addr(crate::jit_exc_value_addr));
        sink.i64_load(mem64(0));
        sink.i32_wrap_i64();
        sink.i32_store(memarg(majit_backend::jitframe::JF_GUARD_EXC_OFS as u64, 2));
        sink.i32_const(runtime_addr(crate::jit_exc_value_addr));
        sink.i64_const(0);
        sink.i64_store(mem64(0));
        sink.i32_const(runtime_addr(crate::jit_exc_type_addr));
        sink.i64_const(0);
        sink.i64_store(mem64(0));
        emit_store_header_word(
            sink,
            majit_backend::jitframe::JF_DESCR_OFS as u64,
            propagate_exception_descr,
        );
        sink.local_get(0);
        sink.return_();
    } else {
        sink.unreachable();
    }
    sink.end();
}

/// `store_info_on_descr`: a fail location is the physical item the guest
/// stored, so `FRAME_SLOT_BASE + loc * 8` reloads it. A geometry with no
/// tail leaves the compact index unchanged.
fn record_physical_fail_locs(guards: &mut [GuardExit], frame: FrameGeometry) {
    if !frame.has_tail() {
        return;
    }
    for guard in guards {
        for loc in &mut guard.fail_locs {
            if let Some(slot) = loc.as_mut() {
                *slot = frame.spill_slot_index(*slot as u64) as usize;
            }
        }
    }
}

fn physical_fail_slot(guard: &GuardExit, index: usize) -> Option<usize> {
    if guard.fail_locs.is_empty() {
        return (index < guard.fail_arg_types.len()).then_some(index);
    }
    guard.fail_locs.get(index).copied().flatten()
}

/// Gcmap of this entry's Ref inputs at `spill_slot_ofs(k)`, parked on the
/// module's `LoopAsmResources`. Zero when no input is a Ref.
fn realloc_entry_gcmap(frame: FrameGeometry, inputargs: &[InputArgRc], sink: usize) -> i64 {
    let sign = std::mem::size_of::<isize>();
    let mut indices = Vec::new();
    for (k, ia) in inputargs.iter().enumerate() {
        if ia.tp.get() != Type::Ref {
            continue;
        }
        let slot = frame.spill_slot_index(k as u64) as usize;
        indices.push(((FRAME_SLOT_BASE as usize + slot * 8) / sign) as u32);
    }
    if indices.is_empty() {
        return 0;
    }
    crate::release::park_gcmap_raw(sink, gcmap_for_item_indices(&indices)) as i64
}

fn ref_spill_item_indices(guard: &GuardExit, sign: usize) -> Vec<u32> {
    let mut indices = Vec::new();
    for (i, ty) in guard.fail_arg_types.iter().enumerate() {
        if *ty != Type::Ref {
            continue;
        }
        let Some(slot) = physical_fail_slot(guard, i) else {
            continue;
        };
        indices.push(((FRAME_SLOT_BASE as usize + slot * 8) / sign) as u32);
    }
    indices
}

fn gcmap_for_item_indices(indices: &[u32]) -> Box<[usize]> {
    let bits_per_word = usize::BITS as usize;
    let num_words = indices
        .iter()
        .copied()
        .max()
        .map_or(1, |last| last as usize / bits_per_word + 1);
    let mut gcmap = vec![0usize; 1 + num_words];
    gcmap[0] = num_words;
    for &index in indices {
        let index = index as usize;
        gcmap[1 + index / bits_per_word] |= 1usize << (index % bits_per_word);
    }
    gcmap.into_boxed_slice()
}

/// `local 0` is the items base. Header pointer fields are wasm32 `i32`s:
/// the module always runs as wasm32, where a `GcRef` and a cell address
/// are four bytes (`emit_force_arm`'s `i32.store` of `jf_force_descr`).
fn emit_header_base(sink: &mut PeepSink<'_, '_>) {
    sink.local_get(0);
    sink.i32_const(majit_backend::jitframe::FIRST_ITEM_OFFSET as i32);
    sink.i32_sub();
}

fn emit_store_header_word(sink: &mut PeepSink<'_, '_>, offset: u64, value: usize) {
    emit_header_base(sink);
    sink.i32_const(value as i32);
    sink.i32_store(memarg(offset, 2));
}

/// Pair `local` of the exit table: word 0 is the descr cell, word 1 the gcmap.
/// Each word is a wasm32 pointer.
fn emit_load_exit_word(sink: &mut PeepSink<'_, '_>, table_base: u32, local: usize, word: usize) {
    let byte = (local * 2 + word) * 4;
    sink.i32_const(table_base as i32);
    if byte != 0 {
        sink.i32_const(byte as i32);
        sink.i32_add();
    }
    sink.i32_load(memarg(0, 2));
}

fn emit_store_loaded_header(
    sink: &mut PeepSink<'_, '_>,
    offset: u64,
    table_base: u32,
    local: usize,
    word: usize,
) {
    emit_header_base(sink);
    emit_load_exit_word(sink, table_base, local, word);
    sink.i32_store(memarg(offset, 2));
}

fn emit_guard_fail_index_store(
    sink: &mut PeepSink<'_, '_>,
    dispatch: BridgeDispatch<'_>,
    op: &Op,
    guard_idx: u32,
) {
    let local = (guard_idx - dispatch.fail_index_base) as usize;
    // Direct codegen tests pass frame address 0 and read the exit index at
    // offset 0. A real compile sets `exit_table_base` and stores the descr
    // cell in `jf_descr` plus the guard gcmap in `jf_gcmap`
    // (`_build_failure_recovery`).
    if dispatch.exit_table_base == 0 {
        sink.local_get(0);
        sink.i64_const(exit_index(op, guard_idx, dispatch.attached) as i64);
        sink.i64_store(mem64(0));
        return;
    }
    if op.opcode == OpCode::Finish
        && let Some(index) =
            crate::failguard::attached_finish_exit_index(&dispatch.attached, &op.getdescr())
    {
        emit_store_header_word(
            sink,
            majit_backend::jitframe::JF_DESCR_OFS as u64,
            crate::failguard::finish_cell_ptr(&dispatch.attached, index),
        );
    } else if dispatch.exit_table_base == 0 {
        emit_store_header_word(sink, majit_backend::jitframe::JF_DESCR_OFS as u64, 0);
    } else {
        emit_store_loaded_header(
            sink,
            majit_backend::jitframe::JF_DESCR_OFS as u64,
            dispatch.exit_table_base,
            local,
            0,
        );
    }
    if dispatch.exit_table_base == 0 {
        // A direct codegen test has no table. Storing 0 keeps the exit from
        // loading address 0 when the module is executed.
        emit_store_header_word(sink, majit_backend::jitframe::JF_GCMAP_OFS as u64, 0);
    } else {
        emit_store_loaded_header(
            sink,
            majit_backend::jitframe::JF_GCMAP_OFS as u64,
            dispatch.exit_table_base,
            local,
            1,
        );
    }
}

// ── Binary ops ──

#[derive(Clone, Copy, Debug)]
enum BinOp {
    I64Add,
    I64Sub,
    I64Mul,
    I64DivS,
    I64RemS,
    I64And,
    I64Or,
    I64Xor,
    I64Shl,
    I64ShrS,
    I64ShrU,
}

fn apply_binop(sink: &mut PeepSink<'_, '_>, op: BinOp) {
    match op {
        BinOp::I64Add => {
            sink.i64_add();
        }
        BinOp::I64Sub => {
            sink.i64_sub();
        }
        BinOp::I64Mul => {
            sink.i64_mul();
        }
        BinOp::I64DivS => {
            sink.i64_div_s();
        }
        BinOp::I64RemS => {
            sink.i64_rem_s();
        }
        BinOp::I64And => {
            sink.i64_and();
        }
        BinOp::I64Or => {
            sink.i64_or();
        }
        BinOp::I64Xor => {
            sink.i64_xor();
        }
        BinOp::I64Shl => {
            sink.i64_shl();
        }
        BinOp::I64ShrS => {
            sink.i64_shr_s();
        }
        BinOp::I64ShrU => {
            sink.i64_shr_u();
        }
    }
}

fn emit_binop(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    op: &Op,
    binop: BinOp,
) {
    let vi = op.pos().get().raw();
    if OpRef::raw_is_constant(vi) {
        return;
    }
    emit_resolve(sink, constants, value_types, op.arg(0).to_opref());
    emit_resolve(sink, constants, value_types, op.arg(1).to_opref());
    apply_binop(sink, binop);
    sink.local_set(value_types.local(vi));
}

/// `UintMulHigh`: high 64 bits of the unsigned 64×64→128 product. Wasm has
/// only `i64.mul` (low 64 bits), so compute via the classic 32-bit split:
/// a = ah·2³²+al, b = bh·2³²+bl, with carry-safe intermediates
///   mid1 = ah·bl + (al·bl >> 32)
///   high = ah·bh + (mid1 >> 32) + ((al·bh + (mid1 & 0xFFFFFFFF)) >> 32)
/// Uses the five scratch locals reserved after the dense value-local range.
fn emit_umulhi(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    op: &Op,
    value_local_count: u32,
) {
    let vi = op.pos().get().raw();
    if OpRef::raw_is_constant(vi) {
        return;
    }
    emit_umulhi_to_local(
        sink,
        constants,
        value_types,
        op,
        value_local_count,
        value_types.local(vi),
    );
}

fn emit_umulhi_to_local(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    op: &Op,
    value_local_count: u32,
    output_local: u32,
) {
    const MASK32: i64 = 0xFFFF_FFFF;
    let al = value_local_count + 1;
    let ah = value_local_count + 2;
    let bl = value_local_count + 3;
    let bh = value_local_count + 4;
    let mid1 = value_local_count + 5;

    // al = a & 0xFFFFFFFF
    emit_resolve(sink, constants, value_types, op.arg(0).to_opref());
    sink.i64_const(MASK32);
    sink.i64_and();
    sink.local_set(al);
    // ah = a >>u 32
    emit_resolve(sink, constants, value_types, op.arg(0).to_opref());
    sink.i64_const(32);
    sink.i64_shr_u();
    sink.local_set(ah);
    // bl = b & 0xFFFFFFFF
    emit_resolve(sink, constants, value_types, op.arg(1).to_opref());
    sink.i64_const(MASK32);
    sink.i64_and();
    sink.local_set(bl);
    // bh = b >>u 32
    emit_resolve(sink, constants, value_types, op.arg(1).to_opref());
    sink.i64_const(32);
    sink.i64_shr_u();
    sink.local_set(bh);

    // mid1 = ah*bl + ((al*bl) >>u 32)
    sink.local_get(al);
    sink.local_get(bl);
    sink.i64_mul();
    sink.i64_const(32);
    sink.i64_shr_u();
    sink.local_get(ah);
    sink.local_get(bl);
    sink.i64_mul();
    sink.i64_add();
    sink.local_set(mid1);

    // high = ah*bh + (mid1 >>u 32) + ((al*bh + (mid1 & MASK32)) >>u 32)
    sink.local_get(ah);
    sink.local_get(bh);
    sink.i64_mul();
    sink.local_get(mid1);
    sink.i64_const(32);
    sink.i64_shr_u();
    sink.i64_add();
    sink.local_get(al);
    sink.local_get(bh);
    sink.i64_mul();
    sink.local_get(mid1);
    sink.i64_const(MASK32);
    sink.i64_and();
    sink.i64_add();
    sink.i64_const(32);
    sink.i64_shr_u();
    sink.i64_add();

    sink.local_set(output_local);
}

/// The overflow condition for `a + c` / `a - c` against a constant `c`, as
/// `(limit, greater_than)`: the operation overflows exactly when `a > limit`
/// (`greater_than`) or when `a < limit`. `None` means it cannot overflow.
///
/// The general form needs both operands and the result to compare sign bits.
/// Against a constant the same predicate is one comparison against a bound
/// folded here, which also drops the dependency on the result. Each bound is
/// taken from the opposite extreme, so none of them can itself overflow:
/// `MAX - c` only for `c > 0`, `MIN - c` only for `c < 0`, and so on.
fn ovf_const_bound(binop: BinOp, c: i64) -> Option<(i64, bool)> {
    use std::cmp::Ordering;
    match binop {
        BinOp::I64Add => match c.cmp(&0) {
            Ordering::Greater => Some((i64::MAX - c, true)),
            Ordering::Less => Some((i64::MIN - c, false)),
            Ordering::Equal => None,
        },
        BinOp::I64Sub => match c.cmp(&0) {
            Ordering::Greater => Some((i64::MIN + c, false)),
            Ordering::Less => Some((i64::MAX + c, true)),
            Ordering::Equal => None,
        },
        _ => None,
    }
}

/// The variable operand and constant operand of an add/sub whose overflow can
/// take the [`ovf_const_bound`] test. Addition is commutative, so either side
/// may supply the constant; for subtraction only the subtrahend does, since
/// `c - a` has a different bound shape and keeps the general form.
fn ovf_const_operand(
    constants: &indexmap::IndexMap<u32, i64>,
    op: &Op,
    binop: BinOp,
) -> Option<(OpRef, i64)> {
    let (a, b) = (op.arg(0).to_opref(), op.arg(1).to_opref());
    match binop {
        BinOp::I64Add if a.is_constant() && !b.is_constant() => {
            Some((b, resolve_const_bits(constants, a)))
        }
        BinOp::I64Add | BinOp::I64Sub if !a.is_constant() && b.is_constant() => {
            Some((a, resolve_const_bits(constants, b)))
        }
        _ => None,
    }
}

/// What an overflow op left for the guard that follows it.
enum OvfFlag {
    /// The op was constant-folded and emitted nothing; there is no flag.
    Absent,
    /// `ovf_flag_local` holds the flag — nonzero means the op overflowed.
    InLocal,
    /// The following guard's failure condition is on the stack as an i32,
    /// ready for `emit_guard_if_exit`.
    FusedCond,
}

/// The comparison a fused guard makes on the overflow predicate.
/// `GuardNoOverflow` exits when the op overflowed and so takes the predicate as
/// it stands; `GuardOverflow` exits when it did not, and flipping the
/// comparison costs nothing where negating its result would cost an
/// instruction.
fn overflow_failure_cmp(cmp: CmpOp, fused_guard: OpCode) -> CmpOp {
    match fused_guard {
        OpCode::GuardNoOverflow => cmp,
        OpCode::GuardOverflow => match cmp {
            CmpOp::I64GtS => CmpOp::I64LeS,
            CmpOp::I64LtS => CmpOp::I64GeS,
            CmpOp::I64Ne => CmpOp::I64Eq,
            _ => unreachable!(
                "overflow comparison must be a constant bound or a sign-word inequality"
            ),
        },
        _ => unreachable!("overflow fusion requires an overflow guard"),
    }
}

/// Overflow binary op: stores the wrapping result in pos and either leaves the
/// overflow flag for a following guard or writes it to the scratch local.
fn emit_ovf_binop(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    op: &Op,
    binop: BinOp,
    value_local_count: u32,
    ovf_flag_local: u32,
    fused_guard: Option<OpCode>,
) -> OvfFlag {
    let vi = op.pos().get().raw();
    if OpRef::raw_is_constant(vi) {
        return OvfFlag::Absent;
    }
    let result_local = value_types.local(vi);

    emit_resolve(sink, constants, value_types, op.arg(0).to_opref());
    emit_resolve(sink, constants, value_types, op.arg(1).to_opref());
    apply_binop(sink, binop);
    sink.local_set(result_local);

    if let Some((var, c)) = ovf_const_operand(constants, op, binop) {
        match ovf_const_bound(binop, c) {
            Some((limit, greater_than)) => {
                let cmp = if greater_than {
                    CmpOp::I64GtS
                } else {
                    CmpOp::I64LtS
                };
                emit_resolve(sink, constants, value_types, var);
                sink.i64_const(limit);
                if let Some(guard) = fused_guard {
                    apply_cmp(sink, overflow_failure_cmp(cmp, guard));
                    return OvfFlag::FusedCond;
                }
                apply_cmp(sink, cmp);
                sink.i64_extend_i32_u();
            }
            // Adding or subtracting zero: the flag stays live so the paired
            // guard still finds it, and folds against a constant zero. There
            // is no predicate to hand a fused guard, so this answers in the
            // local whether or not one was offered.
            None => {
                sink.i64_const(0);
            }
        }
        sink.local_set(ovf_flag_local);
        return OvfFlag::InLocal;
    }

    match binop {
        BinOp::I64Add => {
            // (a ^ result) & (b ^ result) — negative exactly when it overflowed
            emit_resolve(sink, constants, value_types, op.arg(0).to_opref());
            sink.local_get(result_local);
            sink.i64_xor();
            emit_resolve(sink, constants, value_types, op.arg(1).to_opref());
            sink.local_get(result_local);
            sink.i64_xor();
            sink.i64_and();
        }
        BinOp::I64Sub => {
            // (a ^ b) & (a ^ result) — negative exactly when it overflowed
            emit_resolve(sink, constants, value_types, op.arg(0).to_opref());
            emit_resolve(sink, constants, value_types, op.arg(1).to_opref());
            sink.i64_xor();
            emit_resolve(sink, constants, value_types, op.arg(0).to_opref());
            sink.local_get(result_local);
            sink.i64_xor();
            sink.i64_and();
        }
        BinOp::I64Mul => {
            // Multiplying two signed-32-bit integers cannot overflow i64: the
            // largest magnitude is 2^62. This is the common Python-loop shape
            // (e.g. nested_loop's 0..19999 counters), and avoids expanding
            // every multiplication into a software 64x64->128 product. The
            // exact sign-extension checks preserve the full-width slow path
            // for every value outside that proven-safe domain.
            // Resolving an operand is the same single `local.get` that reading
            // a scratch copy of it would be, so the check reads the operands
            // and the umulhi bank stays the slow arm's alone.
            emit_resolve(sink, constants, value_types, op.arg(0).to_opref());
            sink.i64_extend32_s();
            emit_resolve(sink, constants, value_types, op.arg(0).to_opref());
            sink.i64_eq();
            emit_resolve(sink, constants, value_types, op.arg(1).to_opref());
            sink.i64_extend32_s();
            emit_resolve(sink, constants, value_types, op.arg(1).to_opref());
            sink.i64_eq();
            sink.i32_and();
            // Both arms answer the same one-bit predicate, so the block
            // yields it rather than each arm storing it: an adjacent guard
            // reads it off the stack the way the add and sub forms do, and
            // only an unpaired op pays for the flag local. With no paired
            // guard the predicate to yield is `GuardNoOverflow`'s -- "it
            // overflowed" -- which is what that local is defined to hold.
            let failure_of = fused_guard.unwrap_or(OpCode::GuardNoOverflow);
            sink.if_(BlockType::Result(ValType::I32));
            sink.i32_const(i32::from(matches!(failure_of, OpCode::GuardOverflow)));
            sink.else_();

            // Convert the unsigned high word to the signed high word:
            // smulhi = umulhi - ((a >>s 63) & b) - ((b >>s 63) & a).
            let high_local = value_local_count + 1;
            emit_umulhi_to_local(
                sink,
                constants,
                value_types,
                op,
                value_local_count,
                high_local,
            );
            sink.local_get(high_local);
            emit_resolve(sink, constants, value_types, op.arg(0).to_opref());
            sink.i64_const(63);
            sink.i64_shr_s();
            emit_resolve(sink, constants, value_types, op.arg(1).to_opref());
            sink.i64_and();
            sink.i64_sub();
            emit_resolve(sink, constants, value_types, op.arg(1).to_opref());
            sink.i64_const(63);
            sink.i64_shr_s();
            emit_resolve(sink, constants, value_types, op.arg(0).to_opref());
            sink.i64_and();
            sink.i64_sub();
            sink.local_get(result_local);
            sink.i64_const(63);
            sink.i64_shr_s();
            apply_cmp(sink, overflow_failure_cmp(CmpOp::I64Ne, failure_of));
            sink.end();
            if fused_guard.is_some() {
                return OvfFlag::FusedCond;
            }
            sink.i64_extend_i32_u();
            sink.local_set(ovf_flag_local);
            return OvfFlag::InLocal;
        }
        _ => unreachable!("overflow emitter requires add, sub, or mul"),
    }

    // The sign bit of the word both arms left on the stack is the answer, so a
    // fused guard reads it with the comparison it was going to make anyway.
    if let Some(guard) = fused_guard {
        sink.i64_const(0);
        apply_cmp(sink, overflow_failure_cmp(CmpOp::I64LtS, guard));
        OvfFlag::FusedCond
    } else {
        sink.i64_const(63);
        sink.i64_shr_s();
        sink.local_set(ovf_flag_local);
        OvfFlag::InLocal
    }
}

// ── Comparison ops ──

#[derive(Clone, Copy)]
enum CmpOp {
    I64LtS,
    I64LeS,
    I64Eq,
    I64Ne,
    I64GtS,
    I64GeS,
    I64LtU,
    I64LeU,
    I64GtU,
    I64GeU,
}

fn apply_cmp(sink: &mut PeepSink<'_, '_>, op: CmpOp) {
    match op {
        CmpOp::I64LtS => {
            sink.i64_lt_s();
        }
        CmpOp::I64LeS => {
            sink.i64_le_s();
        }
        CmpOp::I64Eq => {
            sink.i64_eq();
        }
        CmpOp::I64Ne => {
            sink.i64_ne();
        }
        CmpOp::I64GtS => {
            sink.i64_gt_s();
        }
        CmpOp::I64GeS => {
            sink.i64_ge_s();
        }
        CmpOp::I64LtU => {
            sink.i64_lt_u();
        }
        CmpOp::I64LeU => {
            sink.i64_le_u();
        }
        CmpOp::I64GtU => {
            sink.i64_gt_u();
        }
        CmpOp::I64GeU => {
            sink.i64_ge_u();
        }
    }
}

// ── Float comparison helper ──

#[derive(Clone, Copy)]
enum FloatCmp {
    Lt,
    Le,
    Eq,
    Ne,
    Gt,
    Ge,
}

/// An op whose result is a 0/1 boolean produced by a single wasm comparison.
/// [`push_cond`] leaves that comparison's i32 on the operand stack; [`emit_cond`]
/// is the ordinary spelling that widens and binds it to the result local.
#[derive(Clone, Copy)]
enum CondKind {
    Int(CmpOp),
    Float(FloatCmp),
    IsTrue,
    IsZero,
}

fn cond_kind_of(opcode: OpCode) -> Option<CondKind> {
    Some(match opcode {
        // ── Integer comparisons (signed) ──
        OpCode::IntLt => CondKind::Int(CmpOp::I64LtS),
        OpCode::IntLe => CondKind::Int(CmpOp::I64LeS),
        OpCode::IntEq => CondKind::Int(CmpOp::I64Eq),
        OpCode::IntNe => CondKind::Int(CmpOp::I64Ne),
        OpCode::IntGt => CondKind::Int(CmpOp::I64GtS),
        OpCode::IntGe => CondKind::Int(CmpOp::I64GeS),
        // ── Integer comparisons (unsigned) ──
        OpCode::UintLt => CondKind::Int(CmpOp::I64LtU),
        OpCode::UintLe => CondKind::Int(CmpOp::I64LeU),
        OpCode::UintGt => CondKind::Int(CmpOp::I64GtU),
        OpCode::UintGe => CondKind::Int(CmpOp::I64GeU),
        // ── Pointer comparisons ──
        OpCode::PtrEq | OpCode::InstancePtrEq => CondKind::Int(CmpOp::I64Eq),
        OpCode::PtrNe | OpCode::InstancePtrNe => CondKind::Int(CmpOp::I64Ne),
        // ── Float comparisons ──
        OpCode::FloatLt => CondKind::Float(FloatCmp::Lt),
        OpCode::FloatLe => CondKind::Float(FloatCmp::Le),
        OpCode::FloatEq => CondKind::Float(FloatCmp::Eq),
        OpCode::FloatNe => CondKind::Float(FloatCmp::Ne),
        OpCode::FloatGt => CondKind::Float(FloatCmp::Gt),
        OpCode::FloatGe => CondKind::Float(FloatCmp::Ge),
        // ── Truth tests ──
        OpCode::IntIsTrue => CondKind::IsTrue,
        OpCode::IntIsZero => CondKind::IsZero,
        _ => return None,
    })
}

/// Push the comparison's i32 result (0 or 1) onto the operand stack.
fn push_cond(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    op: &Op,
    kind: CondKind,
) {
    match kind {
        CondKind::Int(cmpop) => {
            emit_resolve(sink, constants, value_types, op.arg(0).to_opref());
            emit_resolve(sink, constants, value_types, op.arg(1).to_opref());
            apply_cmp(sink, cmpop);
        }
        CondKind::Float(cmp) => {
            emit_resolve_f64(sink, constants, value_types, op.arg(0).to_opref());
            emit_resolve_f64(sink, constants, value_types, op.arg(1).to_opref());
            match cmp {
                FloatCmp::Lt => {
                    sink.f64_lt();
                }
                FloatCmp::Le => {
                    sink.f64_le();
                }
                FloatCmp::Eq => {
                    sink.f64_eq();
                }
                FloatCmp::Ne => {
                    sink.f64_ne();
                }
                FloatCmp::Gt => {
                    sink.f64_gt();
                }
                FloatCmp::Ge => {
                    sink.f64_ge();
                }
            }
        }
        CondKind::IsTrue => {
            emit_resolve(sink, constants, value_types, op.arg(0).to_opref());
            sink.i64_const(0);
            sink.i64_ne();
        }
        CondKind::IsZero => {
            emit_resolve(sink, constants, value_types, op.arg(0).to_opref());
            sink.i64_eqz();
        }
    }
}

/// Push whether a fused GuardTrue/GuardFalse fails. Native backends invert the
/// integer condition code in place (`x86/assembler.py genop_guard_guard_true`); spelling the
/// inverse Wasm comparison directly avoids materialising `cmp; i32.eqz` at the
/// hot guard site. Float ordered comparisons deliberately keep `i32.eqz`:
/// their apparent inverse is not equivalent for NaN/unordered operands.
fn push_guard_failure_cond(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    op: &Op,
    kind: CondKind,
    guard_opcode: OpCode,
) {
    if matches!(
        guard_opcode,
        OpCode::GuardFalse | OpCode::VecGuardFalse | OpCode::GuardIsnull
    ) {
        push_cond(sink, constants, value_types, op, kind);
        return;
    }
    debug_assert!(
        matches!(
            guard_opcode,
            OpCode::GuardTrue | OpCode::VecGuardTrue | OpCode::GuardNonnull
        ),
        "fused guard must be a boolean or nullness test"
    );
    let inverse = match kind {
        CondKind::Int(cmp) => Some(CondKind::Int(match cmp {
            CmpOp::I64LtS => CmpOp::I64GeS,
            CmpOp::I64LeS => CmpOp::I64GtS,
            CmpOp::I64Eq => CmpOp::I64Ne,
            CmpOp::I64Ne => CmpOp::I64Eq,
            CmpOp::I64GtS => CmpOp::I64LeS,
            CmpOp::I64GeS => CmpOp::I64LtS,
            CmpOp::I64LtU => CmpOp::I64GeU,
            CmpOp::I64LeU => CmpOp::I64GtU,
            CmpOp::I64GtU => CmpOp::I64LeU,
            CmpOp::I64GeU => CmpOp::I64LtU,
        })),
        CondKind::IsTrue => Some(CondKind::IsZero),
        CondKind::IsZero => Some(CondKind::IsTrue),
        CondKind::Float(_) => None,
    };
    if let Some(inverse) = inverse {
        push_cond(sink, constants, value_types, op, inverse);
    } else {
        push_cond(sink, constants, value_types, op, kind);
        sink.i32_eqz();
    }
}

fn emit_cond(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    op: &Op,
    kind: CondKind,
) {
    let vi = op.pos().get().raw();
    if OpRef::raw_is_constant(vi) {
        return;
    }
    push_cond(sink, constants, value_types, op, kind);
    sink.i64_extend_i32_u();
    sink.local_set(value_types.local(vi));
}

// ── Unary op helper ──

fn emit_unary_vi(
    sink: &mut PeepSink<'_, '_>,
    constants: &indexmap::IndexMap<u32, i64>,
    value_types: &ValueLocals,
    op: &Op,
    prefix: impl FnOnce(&mut PeepSink<'_, '_>),
    suffix: impl FnOnce(&mut PeepSink<'_, '_>),
) {
    let vi = op.pos().get().raw();
    if !OpRef::raw_is_constant(vi) {
        prefix(sink);
        emit_resolve(sink, constants, value_types, op.arg(0).to_opref());
        suffix(sink);
        sink.local_set(value_types.local(vi));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu() -> crate::failguard::CpuTestGuard {
        crate::failguard::lock_cpu()
    }

    #[test]
    fn aligned_varsize_frame_bump_rejects_u32_overflow() {
        let _cpu = cpu();
        assert_eq!(aligned_varsize_frame_bump(20), Some(24));
        assert_eq!(aligned_varsize_frame_bump(0xffff_fffc), None);
    }

    #[test]
    fn value_id_space_skips_void_count() {
        // Value boxes are opencoder `_index`; voids are `_count` and must
        // not widen the wasm local namespace to the all-ops sequence.
        let _cpu = cpu();
        use majit_ir::forwarding::bound_operand_from_opref as rb;
        let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
        let guard = Op::new(OpCode::GuardTrue, &[rb(OpRef::input_arg_int(0))]);
        guard.pos().set(OpRef::void_op(1));
        let add = Op::new(
            OpCode::IntAdd,
            &[rb(OpRef::input_arg_int(0)), rb(OpRef::const_int(1))],
        );
        add.pos().set(OpRef::int_op(1));
        let finish = Op::new(OpCode::Finish, &[rb(OpRef::int_op(1))]);
        finish.pos().set(OpRef::void_op(3));
        let ops = vec![guard, add, finish];
        assert_eq!(value_id_end(&inputargs, &ops), 2);
        assert_eq!(collect_guards_and_vars(&inputargs, &ops).1, 2);
    }

    #[test]
    fn void_count_does_not_claim_a_colliding_float_local() {
        // A later VoidOp(_count) must not retag the FloatOp(_index) that
        // already owns that payload; the wasm local stays f64.
        let _cpu = cpu();
        use majit_ir::forwarding::bound_operand_from_opref as rb;
        let inputargs = vec![InputArg::from_type_rc(Type::Float, 0)];
        let add = Op::new(
            OpCode::FloatAdd,
            &[rb(OpRef::input_arg_float(0)), rb(OpRef::input_arg_float(0))],
        );
        add.pos().set(OpRef::float_op(1));
        let guard = Op::new(OpCode::GuardTrue, &[rb(OpRef::const_int(1))]);
        guard.pos().set(OpRef::void_op(1));
        let finish = Op::new(OpCode::Finish, &[rb(OpRef::float_op(1))]);
        finish.pos().set(OpRef::void_op(3));
        let ops = vec![add, guard, finish];
        let num_vars = collect_guards_and_vars(&inputargs, &ops).1;
        let locals = ValueLocals::collect(&inputargs, &ops, num_vars, 1);
        assert_eq!(locals.ty(1), ValType::F64);
        assert_eq!(num_vars, 2);
    }

    #[test]
    fn jump_phi_coalesces_new_onto_dead_label_slot() {
        let _cpu = cpu();
        use majit_ir::descr::SimpleSizeDescr;
        use majit_ir::forwarding::bound_operand_from_opref as rb;
        let descr: majit_ir::DescrRef = std::sync::Arc::new(SimpleSizeDescr::new(0, 24, 1));
        let p0 = OpRef::input_arg_ref(0);
        let label = Op::new(OpCode::Label, &[rb(p0)]);
        label.setdescr(descr.clone());
        let new = Op::new(OpCode::NewWithVtable, &[]);
        new.pos().set(OpRef::ref_op(10));
        new.setdescr(descr.clone());
        let jump = Op::new(OpCode::Jump, &[rb(OpRef::ref_op(10))]);
        jump.setdescr(descr);
        let ops = vec![label, new, jump];
        assert_eq!(jump_phi_coalesce_pairs(&ops), vec![(10, 0)]);
        let inputargs = vec![InputArg::from_type_rc(Type::Ref, 0)];
        let locals = ValueLocals::collect(&inputargs, &ops, 16, 1);
        assert_eq!(
            locals.local(10),
            locals.local(0),
            "New that only closes the JUMP must share the LABEL local"
        );
    }

    #[test]
    fn loop_invariant_keeps_its_local_across_the_backedge() {
        let _cpu = cpu();
        use majit_ir::descr::SimpleSizeDescr;
        use majit_ir::forwarding::bound_operand_from_opref as rb;
        let descr: majit_ir::DescrRef = std::sync::Arc::new(SimpleSizeDescr::new(0, 24, 1));
        let label = Op::new(OpCode::Label, &[]);
        label.setdescr(descr.clone());
        let use_inv = Op::new(
            OpCode::IntAdd,
            &[rb(OpRef::input_arg_int(0)), rb(OpRef::const_int(1))],
        );
        use_inv.pos().set(OpRef::int_op(2));
        let later = Op::new(
            OpCode::IntAdd,
            &[rb(OpRef::int_op(2)), rb(OpRef::const_int(1))],
        );
        later.pos().set(OpRef::int_op(3));
        let jump = Op::new(OpCode::Jump, &[]);
        jump.setdescr(descr);
        let ops = vec![label, use_inv, later, jump];
        let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
        let locals = ValueLocals::collect(&inputargs, &ops, 16, 1);
        assert_ne!(
            locals.local(0),
            locals.local(3),
            "a body result must not reuse a loop invariant's local"
        );
    }

    #[test]
    fn jump_phi_does_not_coalesce_when_label_slot_is_still_live() {
        let _cpu = cpu();
        use majit_ir::descr::SimpleSizeDescr;
        use majit_ir::forwarding::bound_operand_from_opref as rb;
        let descr: majit_ir::DescrRef = std::sync::Arc::new(SimpleSizeDescr::new(0, 24, 1));
        let p0 = OpRef::input_arg_ref(0);
        let label = Op::new(OpCode::Label, &[rb(p0)]);
        label.setdescr(descr.clone());
        let new = Op::new(OpCode::NewWithVtable, &[]);
        new.pos().set(OpRef::ref_op(10));
        new.setdescr(descr.clone());
        let keep = Op::new(OpCode::GuardNonnull, &[rb(p0)]);
        let jump = Op::new(OpCode::Jump, &[rb(OpRef::ref_op(10))]);
        jump.setdescr(descr);
        let ops = vec![label, new, keep, jump];
        assert!(
            jump_phi_coalesce_pairs(&ops).is_empty(),
            "LABEL slot still read after the New must keep its own local"
        );
    }

    #[test]
    fn jump_phi_does_not_coalesce_swapped_label_args() {
        let _cpu = cpu();
        use majit_ir::descr::SimpleSizeDescr;
        use majit_ir::forwarding::bound_operand_from_opref as rb;
        let descr: majit_ir::DescrRef = std::sync::Arc::new(SimpleSizeDescr::new(0, 24, 1));
        let p0 = OpRef::input_arg_ref(0);
        let p1 = OpRef::input_arg_ref(1);
        let label = Op::new(OpCode::Label, &[rb(p0), rb(p1)]);
        label.setdescr(descr.clone());
        let jump = Op::new(OpCode::Jump, &[rb(p1), rb(p0)]);
        jump.setdescr(descr);
        let ops = vec![label, jump];
        assert!(
            jump_phi_coalesce_pairs(&ops).is_empty(),
            "JUMP(p1, p0) onto LABEL(p0, p1) must not alias the two phis"
        );
    }

    #[test]
    fn jump_phi_does_not_coalesce_rotated_label_args() {
        let _cpu = cpu();
        use majit_ir::descr::SimpleSizeDescr;
        use majit_ir::forwarding::bound_operand_from_opref as rb;
        let descr: majit_ir::DescrRef = std::sync::Arc::new(SimpleSizeDescr::new(0, 24, 1));
        let p0 = OpRef::input_arg_ref(0);
        let p1 = OpRef::input_arg_ref(1);
        let p2 = OpRef::input_arg_ref(2);
        let label = Op::new(OpCode::Label, &[rb(p0), rb(p1), rb(p2)]);
        label.setdescr(descr.clone());
        let jump = Op::new(OpCode::Jump, &[rb(p1), rb(p2), rb(p0)]);
        jump.setdescr(descr);
        let ops = vec![label, jump];
        assert!(
            jump_phi_coalesce_pairs(&ops).is_empty(),
            "JUMP(p1, p2, p0) onto LABEL(p0, p1, p2) must not alias the rotate"
        );
    }

    #[test]
    fn ref_homes_keep_distinct_slots_for_swapped_label_args() {
        let _cpu = cpu();
        use majit_ir::descr::SimpleSizeDescr;
        use majit_ir::forwarding::bound_operand_from_opref as rb;
        let descr: majit_ir::DescrRef = std::sync::Arc::new(SimpleSizeDescr::new(0, 24, 1));
        let p0 = OpRef::input_arg_ref(0);
        let p1 = OpRef::input_arg_ref(1);
        let label = Op::new(OpCode::Label, &[rb(p0), rb(p1)]);
        label.setdescr(descr.clone());
        let new = Op::new(OpCode::NewWithVtable, &[]);
        new.pos().set(OpRef::ref_op(10));
        new.setdescr(descr.clone());
        let jump = Op::new(OpCode::Jump, &[rb(p1), rb(p0)]);
        jump.setdescr(descr);
        let ops = vec![label, new, jump];
        let inputargs = vec![
            InputArg::from_type_rc(Type::Ref, 0),
            InputArg::from_type_rc(Type::Ref, 1),
        ];
        let homes = RefHomes::collect(&inputargs, &ops, true, &[], &[]);
        let h0 = homes.home(p0).expect("p0 lives across NewWithVtable");
        let h1 = homes.home(p1).expect("p1 lives across NewWithVtable");
        assert_ne!(h0, h1, "swapped LABEL refs must keep distinct GC homes");
    }

    #[test]
    fn label_arg_that_is_also_a_later_read_is_still_seeded() {
        let _cpu = cpu();
        use majit_ir::forwarding::bound_operand_from_opref as rb;
        let inputargs = vec![InputArg::from_type_rc(Type::Int, 0)];
        let mut constants = indexmap::IndexMap::new();
        constants.insert(50, 42);
        let add = Op::new(
            OpCode::IntAdd,
            &[rb(OpRef::int_op(50)), rb(OpRef::const_int(1))],
        );
        add.pos().set(OpRef::int_op(200));
        let ops = vec![
            Op::new(
                OpCode::Label,
                &[rb(OpRef::input_arg_int(0)), rb(OpRef::int_op(50))],
            ),
            add,
        ];
        let seeds = unbound_pool_const_seeds(&inputargs, &ops, &constants, 256)
            .expect("dual-use LABEL id must not decline");
        assert!(
            seeds.contains(&(50, 42)),
            "later IntAdd of a producer-less LABEL arg must seed from the pool, got {seeds:?}"
        );
    }

    #[test]
    fn label_arg_import_hole_is_not_an_unbound_read() {
        let _cpu = cpu();
        use majit_ir::forwarding::bound_operand_from_opref as rb;
        let inputargs = vec![
            InputArg::from_type_rc(Type::Int, 0),
            InputArg::from_type_rc(Type::Int, 1),
        ];
        let constants = indexmap::IndexMap::new();
        let ops = vec![
            Op::new(
                OpCode::Label,
                &[rb(OpRef::input_arg_int(0)), rb(OpRef::input_arg_ref(101))],
            ),
            Op::new(
                OpCode::IntAdd,
                &[rb(OpRef::input_arg_int(0)), rb(OpRef::const_int(1))],
            ),
        ];
        unbound_pool_const_seeds(&inputargs, &ops, &constants, 256)
            .expect("stale LABEL import hole must not decline");
    }

    #[test]
    fn debug_merge_point_import_hole_is_not_an_unbound_read() {
        let _cpu = cpu();
        use majit_ir::forwarding::bound_operand_from_opref as rb;
        let inputargs = vec![
            InputArg::from_type_rc(Type::Int, 0),
            InputArg::from_type_rc(Type::Int, 1),
        ];
        let constants = indexmap::IndexMap::new();
        let ops = vec![
            Op::new(
                OpCode::Label,
                &[rb(OpRef::input_arg_int(0)), rb(OpRef::input_arg_ref(101))],
            ),
            Op::new(OpCode::DebugMergePoint, &[rb(OpRef::input_arg_ref(101))]),
            Op::new(
                OpCode::IntAdd,
                &[rb(OpRef::input_arg_int(0)), rb(OpRef::const_int(1))],
            ),
        ];
        unbound_pool_const_seeds(&inputargs, &ops, &constants, 256)
            .expect("stale debug_merge_point import hole must not decline");
    }

    #[test]
    fn guard_failarg_import_hole_is_not_an_unbound_read() {
        let _cpu = cpu();
        use majit_ir::forwarding::bound_operand_from_opref as rb;
        let inputargs = vec![
            InputArg::from_type_rc(Type::Int, 0),
            InputArg::from_type_rc(Type::Int, 1),
        ];
        let constants = indexmap::IndexMap::new();
        let guard = Op::new(OpCode::GuardNotInvalidated, &[]);
        guard.setfailargs(vec![rb(OpRef::input_arg_ref(101))].into());
        let ops = vec![
            Op::new(
                OpCode::Label,
                &[rb(OpRef::input_arg_int(0)), rb(OpRef::input_arg_ref(101))],
            ),
            guard,
            Op::new(
                OpCode::IntAdd,
                &[rb(OpRef::input_arg_int(0)), rb(OpRef::const_int(1))],
            ),
        ];
        unbound_pool_const_seeds(&inputargs, &ops, &constants, 256)
            .expect("stale guard failarg import hole must not decline");
    }

    #[test]
    fn peep_sink_applies_all_local_folds() {
        let _cpu = cpu();
        let mut bytes = Vec::new();
        {
            let mut raw_sink = InstructionSink::new(&mut bytes);
            let mut sink = PeepSink::new(&mut raw_sink);

            sink.local_set(1)
                .local_get(1)
                .i64_const(0)
                .i32_wrap_i64()
                .i32_const(4)
                .i32_mul()
                .i32_add();
            sink.flush();
        }

        assert_eq!(bytes, [0x22, 0x01]);
    }

    #[test]
    fn peep_sink_folds_i64_constants_and_right_identities() {
        let _cpu = cpu();
        let mut bytes = Vec::new();
        {
            let mut raw_sink = InstructionSink::new(&mut bytes);
            let mut sink = PeepSink::new(&mut raw_sink);

            sink.i64_const(6)
                .i64_const(7)
                .i64_add()
                .i64_const(1)
                .i64_mul()
                .i64_const(0)
                .i64_xor()
                .local_get(3)
                .i64_const(0)
                .i64_add();
            sink.flush();
        }

        assert_eq!(bytes, [0x42, 0x0d, 0x20, 0x03]);
    }

    #[test]
    fn peep_sink_folds_i32_constants_and_right_identities() {
        let _cpu = cpu();
        let mut bytes = Vec::new();
        {
            let mut raw_sink = InstructionSink::new(&mut bytes);
            let mut sink = PeepSink::new(&mut raw_sink);

            sink.i32_const(6)
                .i32_const(7)
                .i32_add()
                .i32_const(1)
                .i32_mul()
                .i32_const(0)
                .i32_or()
                .local_get(3)
                .i32_const(0)
                .i32_shl();
            sink.flush();
        }

        assert_eq!(bytes, [0x41, 0x0d, 0x20, 0x03]);
    }

    /// A guard exit that jumps into a merged region counts only the frames its
    /// own emission opened. A conditional guard branches from inside its
    /// failing `if`; `GUARD_ALWAYS_FAILS` and `FINISH` branch at statement
    /// level, and charging them for an `if` they never open would send the
    /// branch to the enclosing `loop` instead of the region's block.
    #[test]
    fn inline_region_br_depth_counts_only_the_frames_the_caller_opened() {
        let _cpu = cpu();
        let ref_homes = RefHomes {
            by_id: Vec::new(),
            len: 0,
        };
        let param_type_indices = indexmap::IndexMap::new();
        let spill_helpers = indexmap::IndexMap::new();
        let const_tables = ConstPtrTables {
            entries: Vec::new(),
        };
        let inline = InlineGuard {
            guard_idx: 0,
            inputargs: &[],
            region_ordinal: 0,
            outside_loop: false,
        };
        let dispatch = BridgeDispatch {
            cells_base: 0,
            cell_addrs: &[],
            fail_index_base: 0,
            bridge_slot_local: 0,
            enabled: false,
            param_type_indices: &param_type_indices,
            inline_guards: std::slice::from_ref(&inline),
            outside_region_base: 4,
            closed_body_regions: 0,
            closed_outside_regions: 0,
            ref_homes: &ref_homes,
            frame: FrameGeometry::compact(1, 0, 0),
            counter_slot: None,
            spill_helpers: &spill_helpers,
            exit_table_base: 0,
            gc_table_slots: &HashMap::new(),
            const_tables: &const_tables,
            const_table_base: 0,
            attached: majit_backend::AttachedDescrPtrs::default(),
            guard_exit: None,
        };

        assert_eq!(inline_region_br_depth(&inline, &dispatch, 0), 0);
        assert_eq!(inline_region_br_depth(&inline, &dispatch, 1), 1);

        // A preamble region is reached from the outside-loop base instead.
        let outside = InlineGuard {
            outside_loop: true,
            ..inline
        };
        assert_eq!(inline_region_br_depth(&outside, &dispatch, 0), 4);
        assert_eq!(inline_region_br_depth(&outside, &dispatch, 1), 5);
    }

    #[test]
    fn extended_geometry_keeps_source_offsets_and_tails_the_overflow() {
        let source = FrameGeometry::compact(8, 4, 1);
        assert_eq!(source.extend(8, source.ordinary_home_slots()), source);
        let ext = source.extend(11, source.ordinary_home_slots() + 2);
        assert!(ext.has_tail());
        assert_eq!(ext.dispatch_key_ofs, source.dispatch_key_ofs);
        assert_eq!(ext.home_slot_base, source.home_slot_base);
        assert_eq!(ext.force_slot_base, source.force_slot_base);
        assert_eq!(ext.call_result_ofs, source.call_result_ofs);
        assert_eq!(ext.spill_slot_ofs(0), FRAME_SLOT_BASE);
        assert_eq!(ext.spill_slot_ofs(7), FRAME_SLOT_BASE + 7 * SLOT_SIZE);
        assert_eq!(ext.spill_slot_ofs(8), source.frame_bytes as u64);
        let prefix_homes = source.ordinary_home_slots() as u64;
        assert_eq!(ext.home_ofs(0), source.home_slot_base);
        assert_eq!(
            ext.home_ofs(prefix_homes - 1),
            source.home_slot_base + (prefix_homes - 1) * SLOT_SIZE
        );
        let tail_base = source.frame_bytes as u64;
        assert_eq!(ext.spill_slot_ofs(8), tail_base);
        assert_eq!(ext.spill_slot_ofs(9), tail_base + 3 * SLOT_SIZE);
        assert_eq!(ext.spill_slot_ofs(10), tail_base + 6 * SLOT_SIZE);
        assert_eq!(ext.force_slot_ofs(8), tail_base + SLOT_SIZE);
        assert_eq!(ext.home_ofs(prefix_homes), tail_base + 2 * SLOT_SIZE);
        assert_eq!(ext.home_ofs(prefix_homes + 1), tail_base + 5 * SLOT_SIZE);
        let tail_home = ext.home_ofs(prefix_homes);
        assert!(tail_home >= source.frame_bytes as u64);
        let map = build_home_gcmap(ext, source.ordinary_home_slots() + 2, 1);
        let sign = std::mem::size_of::<isize>();
        let bits = std::mem::size_of::<usize>() * 8;
        let marked = |offset: usize| {
            let index = offset / sign;
            map[1 + index / bits] & (1usize << (index % bits)) != 0
        };
        assert!(marked(tail_home as usize), "gcmap marks the tail Ref home");
        assert!(marked(ext.home_ofs(prefix_homes + 1) as usize));
        assert!(marked(source.home_slot_base as usize));
    }

    /// A second extend keeps every offset the first tail published. Tail
    /// value, force, and home words of the grown geometry do not alias.
    #[test]
    fn ca_frame_depth_covers_tail_initial_locs() {
        let source = FrameGeometry::compact(8, 4, 1);
        assert_eq!(
            source.ca_frame_depth(),
            source.ca_frame_bytes as usize / std::mem::size_of::<isize>()
        );
        let tailed = source.extend(11, source.ordinary_home_slots());
        assert!(tailed.has_tail());
        let items_bytes = (tailed.ca_frame_depth() * std::mem::size_of::<isize>()) as u64;
        for k in 0..tailed.value_slots as u64 {
            assert!(
                tailed.spill_slot_ofs(k) + SLOT_SIZE <= items_bytes,
                "initial loc {k} past the CALL_ASSEMBLER frame"
            );
        }
    }

    #[test]
    fn second_extend_keeps_published_tail_offsets() {
        let source = FrameGeometry::compact(8, 4, 1);
        let first = source.extend(11, source.ordinary_home_slots() + 2);
        let second = first.extend(14, source.ordinary_home_slots() + 5);
        assert_eq!(second.tail_base, first.tail_base);
        assert_eq!(second.prefix_value_slots, first.prefix_value_slots);
        assert_eq!(second.prefix_ordinary_homes, first.prefix_ordinary_homes);
        for k in 0..first.value_slots as u64 {
            assert_eq!(
                second.spill_slot_ofs(k),
                first.spill_slot_ofs(k),
                "value {k}"
            );
            assert_eq!(
                second.force_slot_ofs(k),
                first.force_slot_ofs(k),
                "force {k}"
            );
        }
        for h in 0..first.addressable_ordinary_homes() as u64 {
            assert_eq!(second.home_ofs(h), first.home_ofs(h), "home {h}");
        }
        let mut offs = Vec::new();
        let mut push = |ofs: u64| {
            assert!(!offs.contains(&ofs), "tail offset {ofs:#x} is shared");
            offs.push(ofs);
        };
        let prefix_v = second.prefix_value_slots as u64;
        let prefix_h = second.prefix_ordinary_homes as u64;
        for k in prefix_v..second.value_slots as u64 {
            push(second.spill_slot_ofs(k));
            push(second.force_slot_ofs(k));
        }
        for h in prefix_h..second.addressable_ordinary_homes() as u64 {
            push(second.home_ofs(h));
        }
        let rows = (second.value_slots - second.prefix_value_slots).max(second.extra_ordinary_homes)
            as u64;
        assert_eq!(
            second.frame_bytes as u64,
            second.tail_base + 3 * rows * SLOT_SIZE
        );
    }

    #[test]
    fn compact_geometry_keeps_tail_call_area_out_of_ca_prefix() {
        let _cpu = cpu();
        let frame = FrameGeometry::compact(32, 16, 0);
        assert_eq!(frame.dispatch_key_ofs, 32 * SLOT_SIZE);
        assert_eq!(frame.home_slot_base, 33 * SLOT_SIZE);
        assert_eq!(frame.force_slot_base, 49 * SLOT_SIZE);
        assert_eq!(frame.ca_frame_bytes, 648);
        assert_eq!(frame.call_result_ofs, frame.ca_frame_bytes as u64);
        assert_eq!(frame.call_args_ofs, 672);
        assert_eq!(
            frame.frame_bytes as u64,
            frame.call_result_ofs + FrameGeometry::CALL_AREA_SLOTS as u64 * SLOT_SIZE
        );
        assert_eq!(frame.frame_bytes, 816);
    }

    /// The constant-operand bound must answer exactly what the wrapping
    /// arithmetic does, including at the extremes where the bound itself is
    /// closest to overflowing (`c` = `MIN` makes `MAX + c` and `MIN - c` the
    /// interesting cases).
    #[test]
    fn ovf_const_bound_agrees_with_checked_arithmetic() {
        let _cpu = cpu();
        let edges = [
            i64::MIN,
            i64::MIN + 1,
            -3,
            -1,
            0,
            1,
            3,
            i64::MAX - 1,
            i64::MAX,
        ];
        for &c in &edges {
            for &a in &edges {
                for (binop, expected) in [
                    (BinOp::I64Add, a.checked_add(c).is_none()),
                    (BinOp::I64Sub, a.checked_sub(c).is_none()),
                ] {
                    let got = match ovf_const_bound(binop, c) {
                        None => false,
                        Some((limit, true)) => a > limit,
                        Some((limit, false)) => a < limit,
                    };
                    assert_eq!(got, expected, "{binop:?}: a={a} c={c}");
                }
            }
        }
    }
}

/// `(field_size, is_signed)` from an op's FieldDescr. A field op always carries
/// a FieldDescr; a missing one is an invariant violation, so panic rather than
/// emit a silently-wrong width.
fn field_size_sign_from_descr(op: &Op) -> (usize, bool) {
    let descr = op.getdescr();
    if let Some(fd) = descr.as_ref().and_then(|d| d.as_field_descr()) {
        return (fd.field_size(), fd.is_field_signed());
    }
    missing_layout_descr("field descr (size/sign)", op)
}

/// Store width for a `SetfieldGc`/`SetfieldRaw`. A pointer (`Type::Ref`) field
/// is stored at machine-word width regardless of the descr's recorded size: a
/// pointer is 4 bytes on wasm32, so a fixed 8-byte store would clobber the
/// adjacent field. There is no `SetfieldGcR` opcode, so the field type is the
/// only signal — mirroring the `GetfieldGcR` read, which always loads pointers
/// at i32 width. Non-pointer fields use the descr's true field width.
fn setfield_store_size_from_descr(op: &Op) -> usize {
    let descr = op.getdescr();
    if let Some(fd) = descr.as_ref().and_then(|d| d.as_field_descr()) {
        if fd.is_pointer_field() {
            return std::mem::size_of::<usize>();
        }
        return fd.field_size();
    }
    missing_layout_descr("field descr (store size)", op)
}

fn field_is_float_from_descr(op: &Op) -> bool {
    let descr = op.getdescr();
    match descr.as_ref().and_then(|d| d.as_field_descr()) {
        Some(fd) => fd.is_float_field(),
        None => missing_layout_descr("field descr (is_float)", op),
    }
}

fn emit_float_load(
    sink: &mut PeepSink<'_, '_>,
    offset: u64,
    size: usize,
) -> Result<(), BackendError> {
    // `get_type_flag`: only `lltype.Float` is FLAG_FLOAT, and it is 8 bytes.
    // A singlefloat load is an integer load, not `f32.load` + `f64.promote`.
    if size != 8 {
        return Err(BackendError::Unsupported(format!(
            "wasm codegen: float load has size {size} (get_type_flag: lltype.Float is 8 bytes)"
        )));
    }
    sink.f64_load(mem64(offset));
    Ok(())
}

fn array_item_is_float_from_descr(op: &Op) -> bool {
    op.with_array_descr(|ad| ad.item_type() == Type::Float)
        .unwrap_or_else(|| missing_layout_descr("array descr (item is_float)", op))
}

/// Argument index of the stored value for a GC ref-storing op. `SetfieldRaw` /
/// `SetarrayitemRaw` store into non-GC memory and never need a write barrier,
/// so only the `*Gc` variants are listed (rewrite.py only routes `SETFIELD_GC`
/// / `SETARRAYITEM_GC` / `SETINTERIORFIELD_GC` through the barrier).
fn ref_store_value_arg(op: &Op) -> Option<usize> {
    match op.opcode {
        OpCode::SetfieldGc => Some(1),
        OpCode::SetarrayitemGc | OpCode::SetinteriorfieldGc => Some(2),
        _ => None,
    }
}

/// Extract field offset from op's descr (FieldDescr).
fn field_offset_from_descr(op: &Op) -> u64 {
    let __descr_arc_descr = op.getdescr();
    if let Some(descr) = __descr_arc_descr.as_ref()
        && let Some(fd) = descr.as_field_descr()
    {
        return fd.offset() as u64;
    }
    missing_layout_descr("field descr (offset)", op)
}

/// `(length-field offset, length-field size)` from an op's ArrayDescr length
/// descriptor, mirroring `bh_arraylen_gc`, which reads the length at
/// `len_descr().offset()` at machine-word width. The offset is taken from the
/// registered descr (not hardcoded) so it tracks the real per-target layout,
/// and the size lets the caller load at the field's true width — a word-sized
/// length is 4 bytes on wasm32, so a fixed 8-byte read would pull the adjacent
/// field into the high half. Falls back to the conventional offset / word
/// width when no length descr is registered.
fn array_len_layout_from_descr(op: &Op) -> (u64, usize) {
    op.with_array_descr(|ad| {
        ad.len_descr()
            .map(|ld| (ld.offset() as u64, ld.field_size()))
    })
    .flatten()
    .unwrap_or_else(|| missing_layout_descr("array descr (len layout)", op))
}
