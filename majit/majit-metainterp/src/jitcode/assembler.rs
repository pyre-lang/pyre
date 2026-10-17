/// JitCodeBuilder — bytecode assembler for JitCode construction.
///
/// RPython codewriter/assembler.py: assembler that emits bytecodes into a
/// JitCode object. This remains in metainterp only as transitional pyre ABI
/// glue until callers consume `majit_jitcode::codewriter::assembler::Assembler`.
use indexmap::{IndexMap, IndexSet};
use std::cmp::max;

use majit_backend::JitCellToken;
use majit_ir::OpCode;
use majit_jitcode::jitcode::{BhFieldSpec, BhSizeSpec, HEADERLESS_SIZE_OWNER_MARKER};
use majit_jitcode::rclass::ImmutableRank;

use crate::jitcode;

use super::{
    CanonicalBhDescr, JitArgKind, JitCallArg, JitCallAssemblerTarget, JitCallTarget, JitCode,
    RuntimeBhDescr,
};

/// The `EffectInfo` a jitcode-lowered `#[jit_may_force]` helper carries.
///
/// `EF_RANDOM_EFFECTS`, for the reason `pyre-jit`'s `CallFlavor::MayForce`
/// resolves to the same constant: `call.py getcalldescr`'s `if
/// virtualizable_analyzer.analyze(op):` reaches
/// `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE` only through
/// `effectinfo_from_writeanalyze(self.readwrite_analyzer.analyze(op))`
/// (`call.py getcalldescr`), which names every field and array the callee
/// writes — `self.stack[*]` for `tl.py Stack.roll`. The helper reached from here is a
/// Rust `extern "C"` fn with no graph for that analyzer to walk, so an empty
/// write set asserts something it has not proved, and
/// `heap.py force_from_effectinfo` believes it.
///
/// What makes the difference observable on this leg: a jitcode-lowered
/// may-force call belongs to a `#[jit_interp(state_fields = ...)]` machine,
/// which is `VirtualizableInfo::without_vable_token`. Nothing clears a token,
/// so `vable_after_residual_call` never aborts and the trace runs on past the
/// call and reads the state back through
/// `TraceCtx::reload_tokenless_virtualizable_after_residual_call` — a read the
/// empty write set lets the heap cache fold to the value stored before the
/// call.
///
/// The promotion costs no may-force machinery: `check_forces_virtual_or_
/// virtualizable()` reads `extraeffect >= EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`
/// (`effectinfo.py`) and RandomEffects is above it, so
/// `do_residual_call` still selects `CALL_MAY_FORCE_*` with its vable/vref
/// preparation.
fn jitcode_may_force_effect_info() -> majit_ir::effectinfo::EffectInfo {
    crate::call_descr::default_effect_info()
}

/// Lead operand of `rewrite_call(..., force_ir=True)`: the condition /
/// known-result / value register that sits before the funcptr.
enum LeadReg {
    Int(u16),
    Ref(u16),
}

/// Byte width of one scalar of `ty` in memory — a struct field or an array
/// item.
///
/// Only `Ref` follows the target word: its storage IS a pointer, so on
/// wasm32 it is 4 and a literal 8 would make the backend pick a load width
/// (`majit-backend-wasm codegen.rs emit_sized_int_load`) wide enough to fold
/// in the following slot's bytes.
///
/// `Int` does NOT follow the word.  It names the interpreter's integer
/// *bank*, whose values are `i64` (`majit_ir::value::Type` doc), and the
/// storage a descr describes is the source declaration, not the bank: a
/// `jit_interp` `[int; virt]` `VirtArray<i64>` stores `i64` items, and an
/// `int` struct field is an `i64`
/// (`majit-metainterp/tests/jit_interp_inline_helper_typed_return.rs`'s
/// `InlineTypedNode`).
/// Narrowing those to 4 on wasm32 strides past half of every item and
/// truncates the value.  `Float` is `f64` on every target for the same
/// reason.
///
/// This crate is compiled into the wasm guest, so `size_of::<usize>()` is
/// the target word here; build-time host code in `majit-translate` must use
/// `layout::target_word_size()` instead.
pub(crate) fn scalar_size(ty: majit_ir::value::Type) -> usize {
    match ty {
        majit_ir::value::Type::Ref => std::mem::size_of::<usize>(),
        _ => std::mem::size_of::<i64>(),
    }
}

/// Which slot of `fields` a field descr occupies, keyed the way
/// `heaptracker.py get_fielddescr_index_in(STRUCT, fieldname)` keys it.
///
/// Upstream has only the name, because upstream always has one. Every emit site
/// here carries one too — `jitcode_lower` passes the same `stringify!(#member)`
/// it hands `register_struct_layout`, and `newlist_clear` passes the `"length"`
/// / `"items"` it registered — so the name arm is the normal path. The byte
/// offset stands in for the residue: a name the layout *this* site saw does not
/// list, and the empty name a caller outside those paths would supply.
///
/// The offset arm is deliberately narrow. Offset is NOT an identity: measured on
/// this tree, 7 of 1714 submitted field specs sit at an offset another field of
/// the same parent also occupies (a flattened inline aggregate and its first
/// leaf share an address, `heaptracker.py`). Resolving against an
/// ambiguous offset would silently name a sibling, so this returns `None` and
/// leaves the caller's number alone — the same refusal `descr.rs
/// find_index_in_parent` documents for the runtime lookup, for the same reason.
fn field_slot_in(fields: &[BhFieldSpec], name: &str, offset: usize) -> Option<usize> {
    if !name.is_empty()
        && let Some(idx) = fields.iter().position(|fd| fd.name == name)
    {
        return Some(idx);
    }
    let mut at_offset = fields
        .iter()
        .enumerate()
        .filter(|(_, fd)| fd.offset == offset);
    let (idx, _) = at_offset.next()?;
    at_offset.next().is_none().then_some(idx)
}

/// One integer operand in assembler.py's bytecode namespace. `Register`
/// also covers an int constants-pool slot (`argcode == 'i'`), while
/// `ShortConst` is the inline signed byte selected by `USE_C_FORM`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JitCodeIntOperand {
    Register(u16),
    ShortConst(i8),
}

#[derive(Default)]
pub struct JitCodeBuilder {
    /// RPython `jitcode.py` `self.name = name`. Propagated to the
    /// finished `JitCode` by `finish()`; empty by default until a
    /// caller provides the source function name via `set_name`.
    name: String,
    code: Vec<u8>,
    /// RPython `jitcode.py` `self._startpoints = startpoints` —
    /// populated as a sibling of every opcode-byte push so
    /// `JitCode.get_live_vars_info` (RPython `jitcode.py`) can
    /// fire its non-translated `assert pc in self._startpoints` check
    /// against runtime-emitted bytecode.  Recording happens through
    /// `start_instr` / `write_insn`; every helper that pushes an opcode
    /// byte goes through one of those.
    startpoints: indexmap::IndexSet<usize>,
    /// RPython `assembler.py` `self.alllabels.add(len(self.code))` —
    /// every TLabel emit records the bytecode offset of the 2-byte
    /// label slot so `JitCode.follow_jump` (RPython `jitcode.py`)
    /// can fire its non-translated `assert position in self._alllabels`
    /// debug check.  Populated by `push_label_ref`.
    alllabels: indexmap::IndexSet<usize>,
    num_regs_i: u16,
    num_regs_r: u16,
    num_regs_f: u16,
    constants_i: Vec<i64>,
    constants_r: Vec<i64>,
    constants_f: Vec<i64>,
    labels: Vec<Option<usize>>,
    patches: Vec<(usize, usize)>,
    /// RPython `assembler.py` `emit_const` encodes a constant
    /// operand as `count_regs[kind] + len(constants) - 1` — i.e. a
    /// register-space index that points into the constants suffix of
    /// the register file. pyre's builder cannot resolve that final
    /// index at emit time (num_regs_X grows as more touch_reg calls
    /// happen); instead each const-source operand writes a placeholder
    /// 2-byte slot and records `(offset, kind, pool_idx)` here. The
    /// `finish()` pass rewrites each placeholder to
    /// `num_regs_X + pool_idx` once the per-kind register count is final.
    const_patches: Vec<(usize, ConstKind, u16)>,
    /// Same role as `const_patches`, but for 1-byte placeholder slots.
    /// Used by `loop_header` / `jit_merge_point` jdindex emission, where
    /// the operand byte must hold a single `num_regs_X + pool_idx`
    /// register-file index (upstream `@arguments("i")` —
    /// `blackhole.py`). `patch_const_u8_refs()` rewrites each
    /// placeholder once `num_regs_X` is final and asserts the slot fits
    /// in u8.
    const_patches_u8: Vec<(usize, ConstKind, u16)>,
    /// Runtime descriptor pool emitted into `JitCodeExecState.descrs`
    /// on `finish()`. Every `BC_INLINE_CALL` / `BC_CALL_*` /
    /// `BC_RESIDUAL_CALL_*` operand is a 2-byte index into this pool
    /// (RPython `j`/`d` argcode → `descrs[idx]` dispatch).
    descrs: Vec<RuntimeBhDescr>,
    /// Switch descriptors emitted before their target labels are known.
    /// RPython `flatten.py:283-308` stores labels on `SwitchDictDescr`,
    /// then `assembler.py:201-204` attaches concrete positions after
    /// label resolution.  Keep the same two-stage flow for runtime-built
    /// jitcodes.
    pending_switches: Vec<(u16, Vec<(i64, u16)>)>,
    /// `descr.py:108-120` `gccache._cache_size[STRUCT]` analog: the
    /// per-struct `BhSizeSpec` (size + full `all_fielddescrs`) registered
    /// when a `new` for that struct type is emitted, so a following
    /// `setfield_gc_*` can attach the parent SizeDescr plus the field's
    /// `index_in_parent` to its FieldDescr (`descr.py`
    /// `fielddescr.parent_descr = get_size_descr(...)`).  Keyed by the
    /// `path_hash` `type_id` the same way `LLType::Struct(type_id)` keys
    /// the runtime `DescrCache`.  Without it the optimizer's
    /// `optimize_setfield_gc` (`optimizeopt/virtualize.rs`) panics on
    /// the `get_parent_descr()` of a virtualized struct field.
    struct_size_specs: rustc_hash::FxHashMap<u64, BhSizeSpec>,
    /// Pyre-only bridge for canonical `residual_call_*_v`: the bytecode
    /// itself keeps the RPython shape (`i` funcptr operand + `d`
    /// calldescr operand), while the runtime trace path still needs the
    /// separate `{trace_ptr, concrete_ptr}` pair. Keying by the `d`
    /// operand keeps the bridge per callsite and avoids collapsing
    /// different trace wrappers that share a concrete function pointer.
    /// Drained into `JitCodeExecState.call_descr_to_call_target` at
    /// `finish()`.
    call_descr_to_call_target: indexmap::IndexMap<u16, JitCallTarget>,
    /// RPython `jitcode.py JitCode.setup self._resulttypes = resulttypes` —
    /// per-instruction result-kind char keyed by end-of-instruction
    /// position (`assembler.py:217-219`).  Consumed by
    /// `pyjitpl.py make_result_of_lastop` as a non-translated
    /// debug-only type check; pyre fires the same assertion in
    /// `MIFrame::make_result_of_lastop` (`frame.rs`).
    ///
    /// Populated as the LAST step of every typed-result emit helper
    /// — pyre's encoding writes operands AFTER the opcode byte, so
    /// `self.code.len()` after the last `push_u*` call equals the
    /// end-of-instruction position the reader sees as `frame.pc`.
    resulttypes: indexmap::IndexMap<usize, char>,
    /// Pending result-kind for a generic `write_insn("...>X")` call.
    /// RPython records the kind after all operands have been emitted
    /// (`assembler.py:217-219`).  In this builder the opcode helper
    /// runs before operands are pushed, so we defer recording until the
    /// next instruction starts or `finish()` seals the code.
    pending_resulttype: Option<char>,
    has_abort: bool,
    /// Set while emitting when an operand cannot fit the canonical
    /// single-byte register/constant namespace.  `try_finish()` declines
    /// the JitCode before any such bytecode can be installed.
    encoding_overflow: bool,
    /// Bytecode offset of the `BC_JIT_MERGE_POINT(_C)` opcode byte —
    /// captured by `jit_merge_point()` immediately before
    /// `write_insn` pushes the opcode.  Propagated to
    /// `JitCodeExecState::jit_merge_point_offset` at `finish()`.
    /// `None` until `jit_merge_point()` is called; later calls emit their
    /// markers while leaving this first offset unchanged. The LLBC sibling
    /// enforces `jtransform.py`'s single-marker portal-graph contract.
    jit_merge_point_offset: Option<usize>,
    /// Latched by `load_state_field` / `store_state_field` and the ref and
    /// float forms. `finish` copies it to
    /// `JitCodeExecState::reads_identity_slots`.
    reads_identity_slots: bool,
    /// RPython `jitcode.py` `self.fnaddr = fnaddr`. RPython hands
    /// `fnaddr` to `JitCode.__init__` *before* the assembler fills the
    /// body; pyre stages it here through `set_fnaddr` so `finish()` can
    /// commit it alongside the body in a single object construction step.
    fnaddr: i64,
    /// RPython `jitcode.py` `self.calldescr = calldescr`. RPython
    /// hands the calldescr to `JitCode.__init__` *before* the assembler
    /// fills the body, so the field is committed alongside the
    /// `setup()` body.  Callers stage the value here through
    /// `set_calldescr` so `finish()` can stamp it into the body atomically;
    /// the post-assemble `body_mut().calldescr = ...` write the previous
    /// implementation used violated the upstream order
    /// (`call.py` constructs `JitCode(name, fnaddr, calldescr)`
    /// before `assembler.assemble`).
    calldescr: majit_jitcode::jitcode::BhCallDescr,
    /// Set by [`Self::start_instr`] when the body emits an opcode whose answer
    /// comes from the blackhole's own context rather than from memory, so
    /// `finish()` can drop a staged native entry rather than let
    /// `bhimpl_inline_call_*` read a stale heap.
    ///
    /// There is no upstream counterpart because upstream has no such opcode in
    /// a callee: `bhimpl_inline_call_*` (`blackhole.py:1278-1319`) calls
    /// `jitcode.fnaddr` unconditionally, and RPython reaches the
    /// virtualizable's fields through `virtualizable_boxes`, which only the
    /// portal frame carries. pyre's `#[jit_interp]` state-field and vable
    /// bytecodes read the interpreter's registers, so a body carrying one is
    /// not the same program as the Rust function it was lowered from — see
    /// [`Self::native_entry_denied`] for the list.
    native_entry_blocked: bool,
    /// Whether `num_regs_{i,r,f}` have been frozen at the regalloc-
    /// final value. While set, `touch_reg`/`touch_ref_reg`/
    /// `touch_float_reg` skip the `max` update so subsequent operand
    /// emission cannot extend the register file past what regalloc
    /// computed.
    ///
    /// This is the precondition for routing residual_call `Const*`
    /// args through the constants pool: the encoded byte
    /// `num_regs_kind + pool_idx` must remain stable from the moment
    /// `add_const_*` returns through `finish()`'s constants-suffix
    /// placement (`init_register_files_from_runtime_jitcode`,
    /// `blackhole.rs`). RPython's assembler is naturally in
    /// this regime — `assembler.py` runs after regalloc has fixed
    /// `num_regs[kind]` — so freeze restores parity with the upstream
    /// invariant.
    num_regs_frozen: bool,
    /// Deferred-patch table populated by
    /// `live_placeholder_with_triple`. Each entry pairs a `live/<offset>`
    /// patch site (returned by `live_placeholder`) with the per-marker
    /// `(live_i, live_r, live_f)` triple computed by the macro lowerer's
    /// liveness walker (`compute_per_marker_liveness`).
    ///
    /// `finalize_liveness(asm)` walks this table once after body
    /// emission, registers each triple via
    /// `Assembler::_register_liveness_offset` (which dedupes against
    /// `all_liveness_positions`), and rewrites the BC_LIVE 2-byte slot
    /// via `patch_live_offset` so each marker points at its specific
    /// entry instead of the canonical "everything-alive" entry at
    /// offset 0.
    ///
    /// Mirrors the in-line `live` path (`assembler.py:146-158`) for
    /// callers that cannot supply an `&mut Assembler` at body-emit time.
    /// The macro-generated dispatch JitCode and embedded sub-JitCodes use
    /// this deferred path, then finalize against the driver-shared
    /// `Assembler` during install/prebuild.
    #[expect(
        clippy::type_complexity,
        reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
    )]
    pending_live_triples: Vec<(usize, Vec<u8>, Vec<u8>, Vec<u8>)>,
    /// Positions of leading-dummy `BC_LIVE` slots
    /// (`live_placeholder` without an explicit triple) whose 2-byte offset
    /// must be back-patched to the assembler's canonical "all-live" entry
    /// during `finalize_liveness`.
    ///
    /// Routing the leading dummy through deferred patching keeps the
    /// canonical entry from being pre-seeded at `all_liveness` offset 0
    /// before any real `-live-` marker is registered.  Per-marker triples
    /// (registered via `liveness_prebuild_tokens`'s direct
    /// `_register_liveness_offset` calls) get the IR-walk-ordered offsets
    /// at the head of `all_liveness`, matching `assembler.assemble`'s
    /// shape; the canonical entry lands at the tail (or wherever
    /// `ensure_canonical_liveness_offset` first registers it).  The
    /// leading dummy then back-patches its 2-byte slot to that tail
    /// offset rather than assuming offset 0.
    pending_canonical_patches: Vec<usize>,
}

/// Register-file kind for const_patches entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConstKind {
    Int,
    Ref,
    Float,
}

impl JitCodeBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// RPython `jitcode.py` `__init__(name, ...)`: set the symbolic
    /// name used by `dump()` / `Display`. Callers that know the source
    /// function name should call this before `finish()`.
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = name.into();
    }

    /// Stage `fnaddr` (RPython `jitcode.py:16`) so `finish()` can
    /// commit it onto the constructed `JitCode` alongside the body —
    /// matching `JitCode(name, fnaddr, calldescr)` constructor order
    /// (`call.py:167-169`).
    pub fn set_fnaddr(&mut self, fnaddr: i64) {
        self.fnaddr = fnaddr;
    }

    /// Stage the calldescr returned by `get_jitcode_calldescr`
    /// (`call.py`) so `finish()` can stamp it into the body alongside
    /// the bytecode/constants pool, matching the upstream
    /// `JitCode(name, fnaddr, calldescr)` constructor order.
    pub fn set_calldescr(&mut self, calldescr: majit_jitcode::jitcode::BhCallDescr) {
        self.calldescr = calldescr;
    }

    /// Stage the pair `call.py:167-169` passes to `JitCode(name, fnaddr,
    /// calldescr)` for a jitcode whose source function survives compilation as
    /// a callable entry point.
    ///
    /// This is what makes `bhimpl_inline_call_*` (`blackhole.py:1278-1319`)
    /// reachable.  Those handlers do not interpret the callee: they run
    /// `cpu.bh_call_X(adr2int(jitcode.fnaddr), args_i, args_r, args_f,
    /// jitcode.calldescr)`, so the whole callee subtree executes as compiled
    /// code in one call.  A jitcode with no entry — a `match`-arm fragment,
    /// which is a shape upstream does not have — leaves `fnaddr` at 0 and the
    /// blackhole falls back to executing its bytes.
    ///
    /// `arg_classes` is in source parameter order, one char per parameter, and
    /// must name the ABI of `fnaddr` itself: `collect_call_args` walks it to
    /// place the per-kind argument lists back into declaration positions.
    pub fn set_native_entry(&mut self, fnaddr: i64, arg_classes: &str, result_type: char) {
        self.fnaddr = fnaddr;
        self.calldescr = majit_jitcode::jitcode::BhCallDescr::from_arg_classes(
            arg_classes.to_string(),
            result_type,
            majit_ir::descr::EffectInfo::MOST_GENERAL,
        );
    }

    /// Whether `abort()` was called on this builder.
    ///
    /// pyre-specific: when true the caller should treat the resulting
    /// JitCode as non-executable and fall back to the interpreter.
    /// RPython has no equivalent because a translator that emits a bad
    /// bytecode crashes the build via AssertionError instead. Keeping
    /// this flag on the builder (not on `JitCode` itself) lets the
    /// metainterp-side `JitCode` stay aligned with RPython jitcode.py.
    pub fn has_abort_flag(&self) -> bool {
        self.has_abort
    }

    /// Pyre-specific: force-set the abort flag from an outer pipeline
    /// step (e.g. liveness overflow after `finish()` has already packed
    /// the JitCode). Updates the builder state but since finish() has
    /// consumed self, callers use this before finish() or on a new
    /// tracking variable.
    pub fn set_abort_flag(&mut self, v: bool) {
        self.has_abort = v;
    }

    /// Current bytecode emission position.
    pub fn current_pos(&self) -> usize {
        self.code.len()
    }

    pub fn add_const_i(&mut self, value: i64) -> u16 {
        if let Some(index) = self
            .constants_i
            .iter()
            .position(|&existing| existing == value)
        {
            return index as u16;
        }
        let index = self.constants_i.len() as u16;
        self.constants_i.push(value);
        index
    }

    /// Add a ref constant to the constant pool. Returns pool index.
    ///
    /// RPython `assembler.py emit_const` dedups every kind via the
    /// shared `constants_dict` keyed by `(kind, Constant(value_key))`, where
    /// the ref `value_key` is `None` for nullptr and `value._obj.container`
    /// otherwise. Pyre's pool stores raw `i64` pointers; identical raw values
    /// represent the same logical container, so a value-equality lookup
    /// matches upstream's container-identity dedup for all live cases
    /// (PY_NULL collapses against itself; pointer constants collapse when the
    /// caller emits the same address twice).
    pub fn add_const_r(&mut self, value: i64) -> u16 {
        if let Some(index) = self
            .constants_r
            .iter()
            .position(|&existing| existing == value)
        {
            return index as u16;
        }
        let index = self.constants_r.len() as u16;
        self.constants_r.push(value);
        index
    }

    /// Add a float constant (bits as i64) to the constant pool. Returns pool index.
    pub fn add_const_f(&mut self, value: i64) -> u16 {
        if let Some(index) = self
            .constants_f
            .iter()
            .position(|&existing| existing == value)
        {
            return index as u16;
        }
        let index = self.constants_f.len() as u16;
        self.constants_f.push(value);
        index
    }

    /// Current num_regs_i (for computing constant register indices).
    pub fn num_regs_i(&self) -> u16 {
        self.num_regs_i
    }

    /// Current num_regs_r (for computing constant register indices).
    pub fn num_regs_r(&self) -> u16 {
        self.num_regs_r
    }

    /// Current num_regs_f. Callers that need the full per-kind register
    /// ceiling (e.g. `SSAReprEmitter::finish_with`) read all three.
    pub fn num_regs_f(&self) -> u16 {
        self.num_regs_f
    }

    /// `assembler.py` parity: size of the int constant pool (used by
    /// assemble-time bounds checks that allow constant virtual-register
    /// indices `num_regs_i .. num_regs_i + num_consts_i()`).
    pub fn num_consts_i(&self) -> u16 {
        self.constants_i.len() as u16
    }

    /// Same as `num_consts_i` for the ref constant pool.
    pub fn num_consts_r(&self) -> u16 {
        self.constants_r.len() as u16
    }

    /// Same as `num_consts_i` for the float constant pool.
    pub fn num_consts_f(&self) -> u16 {
        self.constants_f.len() as u16
    }

    /// Materialise an integer constant into `dst`.
    ///
    /// `assembler.py` sets `allow_short = (insn[0] in USE_C_FORM)` and
    /// `int_copy` is in `USE_C_FORM` (`assembler.py`), so a value that
    /// fits a signed byte is written inline as `int_copy/c>i`
    /// (`assembler.py:99-107`) and costs no `constants_i` slot.  Registers
    /// and constants of one kind share a single 256-wide index space, so
    /// pooling every small constant instead pushes `try_finish` over the
    /// `total_i > 256` ceiling and declines JitCodes the build-time
    /// assembler — which does emit the short form — would have accepted.
    pub fn load_const_i_value(&mut self, dst: u16, value: i64) {
        if let Ok(short) = i8::try_from(value) {
            self.touch_reg(dst);
            self.write_insn("int_copy/c>i");
            self.push_u8(short as u8);
            self.push_reg_u8(dst, "int_copy/c>i dst");
            return;
        }
        let const_idx = self.add_const_i(value);
        self.load_const_i(dst, const_idx);
    }

    /// Lower to RPython `int_copy/i>i` reading from the constants
    /// window of the register file (`assembler.py` `emit_const`
    /// produces the same byte sequence — a register-space index that
    /// points into the post-regs constants suffix). The src operand
    /// is written as a placeholder; `finish()` patches it to
    /// `num_regs_i + const_idx` once the per-kind register count is
    /// final.
    pub fn load_const_i(&mut self, dst: u16, const_idx: u16) {
        self.touch_reg(dst);
        self.write_insn("int_copy/i>i");
        let src_offset = self.code.len();
        self.push_u8(0);
        self.const_patches_u8
            .push((src_offset, ConstKind::Int, const_idx));
        self.push_u8(dst as u8);
    }

    // ── State field access (register/tape machines) ──

    /// Load a scalar state field value into an int register.
    /// `assembler.py:165-167` `i` argcode emits 1-byte register index;
    /// `assembler.py:197-207` `d` argcode emits 2-byte descr index.
    pub fn load_state_field(&mut self, field_idx: u16, dest: u16) {
        self.reads_identity_slots = true;
        self.touch_reg(dest);
        self.write_insn("load_state_field/di");
        self.push_u16(field_idx);
        self.push_u8(dest as u8);
    }

    /// Store an int register value into a scalar state field.
    pub fn store_state_field(&mut self, field_idx: u16, src: u16) {
        self.reads_identity_slots = true;
        self.touch_reg(src);
        self.write_insn("store_state_field/di");
        self.push_u16(field_idx);
        self.push_u8(src as u8);
    }

    /// Load a ref-typed scalar state field into a ref register. The field
    /// lives in the ref register bank (`registers_r`) so a subsequent
    /// `getfield_gc_*` reads its struct base from a real ref value.
    /// `d` = u16 field index, `r` = ref register.
    pub fn load_state_field_ref(&mut self, field_idx: u16, dest: u16) {
        self.reads_identity_slots = true;
        self.touch_ref_reg(dest);
        self.write_insn("load_state_field_ref/dr");
        self.push_u16(field_idx);
        self.push_u8(dest as u8);
    }

    /// Store a ref register value into a ref-typed scalar state field.
    pub fn store_state_field_ref(&mut self, field_idx: u16, src: u16) {
        self.reads_identity_slots = true;
        self.touch_ref_reg(src);
        self.write_insn("store_state_field_ref/dr");
        self.push_u16(field_idx);
        self.push_u8(src as u8);
    }

    /// Load a float-typed scalar state field into a float register. The
    /// concrete shadow is carried as raw f64 bits in the float register bank.
    /// `d` = u16 field index, `f` = float register.
    pub fn load_state_field_float(&mut self, field_idx: u16, dest: u16) {
        self.reads_identity_slots = true;
        self.touch_float_reg(dest);
        self.write_insn("load_state_field_float/df");
        self.push_u16(field_idx);
        self.push_u8(dest as u8);
    }

    /// Store a float register value into a float-typed scalar state field.
    pub fn store_state_field_float(&mut self, field_idx: u16, src: u16) {
        self.reads_identity_slots = true;
        self.touch_float_reg(src);
        self.write_insn("store_state_field_float/df");
        self.push_u16(field_idx);
        self.push_u8(src as u8);
    }

    // ── First-class virtualizable access (getfield_vable_*) ──
    //
    // Canonical-only API: every emit threads a `vable_reg` operand for
    // the live struct pointer (`pyjitpl.py:1166 _opimpl_setfield_vable_*`
    // `r` argcode), and the descr operand resolves through the runtime
    // descr pool (`assembler.py:165-167` + `:197-207`). The pre-orthodox
    // helper-style legacy methods that omitted `vable_reg` and inlined
    // the field index were retired in Stage 3c-2 once `jit_interp` DSL
    // and pyre-jit both committed to the canonical encoding.

    /// Emit `new_with_vtable/d>r` (`blackhole.py bhimpl_new_with_vtable`):
    /// allocate a zeroed struct of `size` bytes and write `vtable` at the
    /// CPU's vtable offset; ref result in `dest`. The size descr is transient
    /// (no struct context, empty field layout) — sufficient for the blackhole
    /// allocation; the tracer records `OpCode::NewWithVtable` carrying it.
    pub fn new_with_vtable(&mut self, dest: u16, size: usize, type_id: u64, vtable: usize) {
        self.touch_ref_reg(dest);
        let descr = self.add_bh_descr(CanonicalBhDescr::Size {
            size,
            type_id,
            vtable: vtable as u64,
            owner: String::new(),
            all_fielddescrs: Vec::new(),
            is_gc_managed: true,
        });
        self.write_insn("new_with_vtable/d>r");
        self.push_u16(descr);
        self.push_reg_u8(dest, "new_with_vtable result");
    }

    /// Emit `new/d>r` (`blackhole.py bhimpl_new`): allocate a zeroed
    /// struct of `size` bytes; ref result in `dest`. No vtable write.
    ///
    /// `fields` is the struct's full `(offset, is_ref, name)` layout in
    /// `index_in_parent` order (`descr.py` `init_size_descr`
    /// populates `SizeDescr.all_fielddescrs` from
    /// `heaptracker.all_fielddescrs(STRUCT)`).  It is recorded into the
    /// `Size` descr's `all_fielddescrs` and cached under `type_id` so the
    /// matching `setfield_gc_*` can resolve the parent SizeDescr and the
    /// field index; pass an empty slice for a transient allocation with no
    /// field stores (the optimizer never virtualizes a fieldless New).
    pub fn new_struct(
        &mut self,
        dest: u16,
        size: usize,
        type_id: u64,
        headerless: bool,
        fields: &[(usize, bool, &str, usize, bool, bool)],
        immutable_fields: &str,
    ) {
        self.touch_ref_reg(dest);
        // descr.py get_size_descr + init_size_descr: cache the
        // full per-struct layout so the matching setfield_gc_* resolves
        // the parent SizeDescr and field index (descr.py).  Even a
        // headerless JIT `new` keeps `is_gc_managed = true`: the result
        // ref is still rooted by the reg/frame gcmap walk and collector
        // traced.  It just carries no GcHeader, so it must never be
        // subject to GUARD_GC_TYPE (which reads at `ref - GcHeader::SIZE`).
        // `StructPtrInfo.make_guards` gates that guard on `!sd.headerless()`.
        self.register_struct_layout(size, type_id, true, headerless, fields, immutable_fields);
        let all_fielddescrs = self
            .struct_size_specs
            .get(&type_id)
            .expect("register_struct_layout just inserted this type_id")
            .all_fielddescrs
            .clone();
        let descr = self.add_bh_descr(CanonicalBhDescr::Size {
            size,
            type_id,
            vtable: 0,
            owner: if headerless {
                HEADERLESS_SIZE_OWNER_MARKER.to_string()
            } else {
                String::new()
            },
            all_fielddescrs,
            is_gc_managed: true,
        });
        self.write_insn("new/d>r");
        self.push_u16(descr);
        self.push_reg_u8(dest, "new result");
    }

    /// `new_with_vtable/d>r` with the same field layout as [`Self::new_struct`].
    ///
    /// `jtransform.py rewrite_op_malloc` emits this when the struct has a
    /// vtable (`heaptracker.get_vtable_for_gcstruct`). The type word is not a
    /// field (`heaptracker.all_fielddescrs` drops `typeptr`); `vtable` is the
    /// class address stored at offset 0 by the backend.
    pub fn new_with_vtable_struct(
        &mut self,
        dest: u16,
        size: usize,
        type_id: u64,
        vtable: usize,
        headerless: bool,
        is_gc_managed: bool,
        fields: &[(usize, bool, &str, usize, bool, bool)],
        immutable_fields: &str,
    ) {
        self.touch_ref_reg(dest);
        self.register_struct_layout(
            size,
            type_id,
            is_gc_managed,
            headerless,
            fields,
            immutable_fields,
        );
        let all_fielddescrs = self
            .struct_size_specs
            .get(&type_id)
            .expect("register_struct_layout just inserted this type_id")
            .all_fielddescrs
            .clone();
        let descr = self.add_bh_descr(CanonicalBhDescr::Size {
            size,
            type_id,
            vtable: vtable as u64,
            owner: if headerless {
                HEADERLESS_SIZE_OWNER_MARKER.to_string()
            } else {
                String::new()
            },
            all_fielddescrs,
            is_gc_managed,
        });
        self.write_insn("new_with_vtable/d>r");
        self.push_u16(descr);
        self.push_reg_u8(dest, "new_with_vtable result");
    }

    /// Register a struct's `(offset, is_ref, name)` layout under `type_id`
    /// WITHOUT emitting a `new/d>r` allocation op.  Used for a struct that
    /// is allocated natively (outside the JIT) but whose fields are still
    /// read/written through `getfield_gc_*` / `setfield_gc_*`: the layout
    /// must be cached so `add_struct_field_descr` can resolve the parent
    /// SizeDescr + `index_in_parent`, giving read and write the same interned
    /// `Field` descr (the heapcache aliases get against set by that identity).
    /// Idempotent: re-registering the same `type_id` overwrites with an
    /// identical layout.
    ///
    /// `is_gc_managed` records whether the struct carries a GC header
    /// (`ref - 8` type-id word): `false` for a raw native struct (the
    /// caller above the JIT owns the memory, no header), `true` when a
    /// JIT-GC `new_struct` reuses this for layout caching.  Threaded to
    /// `SimpleSizeDescr.is_gc_managed` so `StructPtrInfo.make_guards`
    /// gates `GUARD_GC_TYPE`: a header-less raw struct must not be
    /// runtime-type-pinned.
    pub fn register_struct_layout(
        &mut self,
        size: usize,
        type_id: u64,
        is_gc_managed: bool,
        headerless: bool,
        fields: &[(usize, bool, &str, usize, bool, bool)],
        immutable_fields: &str,
    ) {
        crate::note_struct_layout_registration(fields.len());
        // `rclass.py _parse_field_list` splits the declaration once per class,
        // not once per field.  Parsing here lets the merge path and the insert
        // path below read the same ranks.
        let ranks = Self::immutable_ranks(immutable_fields);
        // Every emit site re-registers the layout it accesses, so the same few
        // type_ids arrive hundreds of times.  When the spec already lists an
        // offset for each incoming field the merge below pushes nothing, and
        // the re-sort/re-index that follow it are no-ops on an already sorted,
        // already indexed vector — the branch never touches `size`,
        // `is_gc_managed` or `headerless`, so returning here is exactly
        // equivalent and skips building `new_fields` (one owned String per
        // field) only to discard it.
        //
        // The same walk answers whether the offset that made a field redundant
        // was really *this* field's, so `Self::record_layout_conflicts` runs
        // off it rather than repeating it.
        if let Some(existing) = self.struct_size_specs.get_mut(&type_id) {
            // Apply the declaration before the early return decides there is
            // nothing to do.  A struct's layout accumulates across emit sites
            // in whatever order they lower, and not every site carries the
            // declaration — a `new_struct` for a transient allocation names no
            // struct to read it off.  Deciding on offsets alone would keep
            // whichever site happened to arrive first.
            Self::apply_immutable_ranks(&mut existing.all_fielddescrs, &ranks);
            let mut every_offset_known = true;
            for &field in fields {
                if !Self::record_layout_conflicts(&*existing, type_id, field) {
                    every_offset_known = false;
                }
            }
            if every_offset_known {
                // `descr.py get_size_descr` is one ranking per STRUCT. A
                // first registration may have ranked by offset before the
                // parent SizeDescr was cached; re-rank now so
                // `index_in_parent` follows `all_fielddescrs`.
                Self::rank_field_specs(type_id, &mut existing.all_fielddescrs);
                return;
            }
        }
        let new_fields = Self::field_specs_from_layout(type_id, fields, &ranks);
        // Merge into existing spec if present — each getfield/setfield
        // site registers only the field it accesses, so the complete
        // layout accumulates across multiple register_struct_layout
        // calls for the same type_id.  Without merging, a later call
        // would overwrite the spec and lose earlier fields, causing
        // `field_descr_ref_from_bh` to produce incomplete layouts whose
        // FieldDescr.get_parent_descr() Weak reference dangles.
        if let Some(existing) = self.struct_size_specs.get_mut(&type_id) {
            for nf in new_fields {
                if !existing
                    .all_fielddescrs
                    .iter()
                    .any(|ef| ef.offset == nf.offset)
                {
                    existing.all_fielddescrs.push(nf);
                }
            }
            // `descr.py get_size_descr` is one ranking per STRUCT. Reuse
            // the published SizeDescr's `all_fielddescrs` order when it
            // is already in `gc_cache`; otherwise rank by offset so a
            // struct with no published group stays deterministic.
            Self::rank_field_specs(type_id, &mut existing.all_fielddescrs);
        } else {
            self.struct_size_specs.insert(
                type_id,
                BhSizeSpec {
                    size,
                    type_id,
                    vtable: 0,
                    owner: String::new(),
                    is_gc_managed,
                    headerless,
                    all_fielddescrs: new_fields,
                },
            );
        }
    }

    /// Whether `existing` already lists `field`'s offset, recording what the
    /// offset-keyed dedup would discard if it does so for the wrong reason.
    ///
    /// The dedup key is the byte offset, but an offset does not identify a
    /// field: a flattened inline aggregate and its first leaf sit at one
    /// address, and the corpus really contains such pairs. So a hit at a known
    /// offset splits three ways.
    ///
    /// * The same name, described the same way — the redundant re-registration
    ///   every emit site performs, and what the dedup exists for.
    /// * The same name, described differently — one site calls the word a GC
    ///   pointer and another a scalar, so emit order decides what the collector
    ///   traces.
    /// * A different name — a real sibling, which the merge then drops. The
    ///   spec is left short an entry that `add_struct_field_descr` resolves
    ///   `index_in_parent` and `name` against.
    ///
    /// Only the first is silent. The other two are recorded and the caller
    /// still merges as before: this reports the disagreement, it does not
    /// arbitrate it.
    fn record_layout_conflicts(
        existing: &BhSizeSpec,
        type_id: u64,
        (offset, is_ref, name, decl_size, decl_signed, is_float): (
            usize,
            bool,
            &str,
            usize,
            bool,
            bool,
        ),
    ) -> bool {
        let mut at_offset = existing
            .all_fielddescrs
            .iter()
            .filter(|ef| ef.offset == offset)
            .peekable();
        if at_offset.peek().is_none() {
            return false;
        }
        crate::note_struct_layout_comparison();
        // Both sides are compared in the form the spec RETAINS, not the form
        // the caller submitted: `field_specs_from_layout` discards a ref
        // field's declared width and sign for the pointer word's own, so a
        // recorded ref never remembers what it was handed. Normalising the
        // incoming field the same way is what keeps two agreeing ref
        // registrations from reading as a disagreement.
        let incoming = crate::StructLayoutField {
            name: name.to_string(),
            is_ref,
            size: if is_ref {
                scalar_size(majit_ir::value::Type::Ref)
            } else if is_float {
                scalar_size(majit_ir::value::Type::Float)
            } else {
                decl_size
            },
            signed: !is_ref && !is_float && decl_signed,
        };
        let describe = |ef: &BhFieldSpec| crate::StructLayoutField {
            name: ef.name.clone(),
            is_ref: matches!(ef.field_type, majit_ir::value::Type::Ref),
            size: ef.field_size,
            signed: ef.is_field_signed,
        };
        let mut same_name = at_offset.clone().filter(|ef| ef.name == name).peekable();
        let (kept, kind) = match same_name.peek() {
            Some(&ef) => (describe(ef), crate::StructLayoutConflictKind::Redescribed),
            // Every field at this offset is named something else, so this one is
            // a sibling the merge has no slot for. Report it against the entry
            // the dedup will keep: the lowest-ranked one there.
            None => {
                let ef = at_offset.next().expect("peeked Some above");
                (
                    describe(ef),
                    crate::StructLayoutConflictKind::DroppedSibling,
                )
            }
        };
        if kept != incoming {
            crate::record_struct_layout_conflict(crate::StructLayoutConflict {
                type_id,
                offset,
                kind,
                kept,
                dropped: incoming,
            });
        }
        true
    }

    /// Build `Vec<BhFieldSpec>` from a `(offset, is_ref, name)` layout,
    /// mirroring `descr.py FieldDescr(name, offset, size, flag,
    /// index_in_parent, is_pure)` for each field.  Scalar fields are one
    /// machine word (`symbolic.py WORD = sizeof(lltype.Signed)`, so 4
    /// on wasm32); the flag/sign follow the field kind (pointer vs signed
    /// int).
    ///
    /// `index_in_parent` is the field's rank by byte offset, not its order
    /// in the incoming slice.  `heaptracker.get_fielddescr_index_in`
    /// (`descr.py`) derives the index from the STRUCT's own field list,
    /// a stable per-type property; the caller's slice comes from a Rust
    /// struct literal whose source order may differ from (and vary between
    /// literals of) the declared layout.  Sorting by offset gives every
    /// literal of the same struct an identical, deterministic
    /// `all_fielddescrs` order so the `type_id`-keyed cache is not polluted
    /// and the optimizer's `FieldDescr.n()` indexing stays consistent.
    /// The merged layout recorded for `type_id`, or `None` when no emit site
    /// has registered one.
    ///
    /// A struct's layout accumulates across every site that touches it, so this
    /// answers what the *last* registration left, not what any single site
    /// declared.
    pub fn struct_size_spec(&self, type_id: u64) -> Option<&BhSizeSpec> {
        self.struct_size_specs.get(&type_id)
    }

    /// The declared `(field name, rank)` pairs.
    ///
    /// The declaration travels from `#[jit_immutable_fields]` to here verbatim
    /// and is split in exactly one place: `ImmutableRank::parse`, the port of
    /// `rclass.py _parse_field_list`'s suffix grammar.
    fn immutable_ranks(immutable_fields: &str) -> Vec<(String, ImmutableRank)> {
        immutable_fields
            .split(',')
            .map(str::trim)
            .filter(|entry| !entry.is_empty())
            .map(ImmutableRank::parse)
            .collect()
    }

    /// The rank declared for `name`, or `None` when the declaration omits it.
    fn rank_of(ranks: &[(String, ImmutableRank)], name: &str) -> Option<ImmutableRank> {
        ranks
            .iter()
            .find(|(declared, _)| declared == name)
            .map(|&(_, rank)| rank)
    }

    /// Raise already-recorded fields to a declaration that arrived with a
    /// later emit site.
    ///
    /// It only raises.  A field the declaration omits keeps what it has: an
    /// omission means this site named no declaration, not that the field is
    /// mutable — and the site that did name one may already have run.
    fn apply_immutable_ranks(fields: &mut [BhFieldSpec], ranks: &[(String, ImmutableRank)]) {
        for field in fields.iter_mut() {
            let Some(rank) = Self::rank_of(ranks, &field.name) else {
                continue;
            };
            field.is_immutable |= rank.is_immutable();
            field.is_quasi_immutable |= rank.is_quasi_immutable();
        }
    }

    fn field_specs_from_layout(
        type_id: u64,
        fields: &[(usize, bool, &str, usize, bool, bool)],
        ranks: &[(String, ImmutableRank)],
    ) -> Vec<BhFieldSpec> {
        let mut specs: Vec<BhFieldSpec> = fields
            .iter()
            .map(
                |&(offset, is_ref, name, decl_size, decl_signed, is_float)| {
                    // descr.py `get_field_descr` reads FIELDTYPE once and
                    // `get_type_flag` picks the flag: pointer, FLOAT, signed, or
                    // unsigned. The emit site supplies that kind; nothing retags
                    // the descr afterwards.
                    let (field_type, field_size, field_flag, is_field_signed) = if is_ref {
                        (
                            majit_ir::value::Type::Ref,
                            scalar_size(majit_ir::value::Type::Ref),
                            majit_ir::descr::ArrayFlag::Pointer,
                            false,
                        )
                    } else if is_float {
                        (
                            majit_ir::value::Type::Float,
                            scalar_size(majit_ir::value::Type::Float),
                            majit_ir::descr::ArrayFlag::Float,
                            false,
                        )
                    } else {
                        let flag = if decl_signed {
                            majit_ir::descr::ArrayFlag::Signed
                        } else {
                            majit_ir::descr::ArrayFlag::Unsigned
                        };
                        (majit_ir::value::Type::Int, decl_size, flag, decl_signed)
                    };
                    BhFieldSpec {
                        // descr.py:656 SimpleFieldDescr id is unset until the
                        // runtime DescrCache mints one; index_in_parent carries
                        // the structural slot the optimizer indexes by.
                        index: u32::MAX,
                        field_key: name.to_string(),
                        name: name.to_string(),
                        offset,
                        field_size,
                        field_type,
                        field_flag,
                        is_field_signed,
                        is_immutable: Self::rank_of(ranks, name)
                            .is_some_and(ImmutableRank::is_immutable),
                        is_quasi_immutable: Self::rank_of(ranks, name)
                            .is_some_and(ImmutableRank::is_quasi_immutable),
                        index_in_parent: 0,
                        // The emit-site layout table names fields but declares no
                        // header row, so the rebuilding side falls back to the
                        // name.
                        is_class_word: None,
                    }
                },
            )
            .collect();
        Self::rank_field_specs(type_id, &mut specs);
        specs
    }

    /// Position of `spec` in the published SizeDescr list.
    ///
    /// `heaptracker.get_fielddescr_index_in(STRUCT, fieldname)` keys on the
    /// name. An offset is only used when it identifies one cached field: a
    /// flattened aggregate and its first leaf share an address, so matching
    /// `key == field_key || offset == spec.offset` would give both the
    /// aggregate's index.
    fn rank_of_cached_field(ranks: &[(String, usize)], spec: &BhFieldSpec) -> usize {
        if let Some(idx) = ranks.iter().position(|(key, _)| key == &spec.field_key) {
            return idx;
        }
        let mut found = None;
        for (i, (_, offset)) in ranks.iter().enumerate() {
            if *offset == spec.offset {
                if found.is_some() {
                    return ranks.len();
                }
                found = Some(i);
            }
        }
        found.unwrap_or(ranks.len())
    }

    /// Rank `specs` with the published SizeDescr's `all_fielddescrs` order
    /// (`descr.py get_size_descr` / `heaptracker.all_fielddescrs`) when that
    /// STRUCT is already in `gc_cache`. Otherwise rank by offset, the
    /// deterministic merge used when no group has published yet.
    fn rank_field_specs(type_id: u64, specs: &mut [BhFieldSpec]) {
        if let Some(ranks) = Self::cached_parent_ranks(type_id) {
            for spec in specs.iter_mut() {
                spec.index_in_parent = Self::rank_of_cached_field(&ranks, spec);
            }
            specs.sort_by_key(|spec| (spec.index_in_parent, spec.offset));
            let n_cached = ranks.len();
            let mut next_extra = n_cached;
            for spec in specs.iter_mut() {
                if spec.index_in_parent >= n_cached {
                    spec.index_in_parent = next_extra;
                    next_extra += 1;
                }
            }
        } else {
            specs.sort_by_key(|spec| spec.offset);
            for (idx, spec) in specs.iter_mut().enumerate() {
                spec.index_in_parent = idx;
            }
        }
    }

    fn cached_parent_ranks(type_id: u64) -> Option<Vec<(String, usize)>> {
        if type_id == 0 {
            return None;
        }
        let gc = majit_ir::descr::gc_cache().lock();
        let descr = gc
            ._cache_size
            .get(&majit_ir::descr::LLType::Struct(type_id))?;
        let size = descr.as_size_descr()?;
        let fields = size.all_fielddescrs();
        if fields.is_empty() {
            return None;
        }
        Some(
            fields
                .iter()
                .map(|field| (field.field_key().to_string(), field.offset()))
                .collect(),
        )
    }

    /// Build a scalar `Field` descr (one machine word) for a plain
    /// getfield_gc/setfield_gc operand.  `blackhole.py bhimpl_getfield_gc_i`
    /// resolves the `d` argcode to a FieldDescr carrying the byte
    /// offset; the flag/sign follow the field kind.
    fn add_scalar_field_descr(&mut self, offset: usize, field_type: majit_ir::value::Type) -> u16 {
        self.add_scalar_field_descr_with_immutability(offset, field_type, false)
    }

    /// [`Self::add_scalar_field_descr`] for a field the codewriter declared in
    /// `_immutable_fields_`, so `SimpleFieldDescr::is_always_pure` answers
    /// true and the read is foldable.
    fn add_scalar_field_descr_with_immutability(
        &mut self,
        offset: usize,
        field_type: majit_ir::value::Type,
        is_immutable: bool,
    ) -> u16 {
        let (field_flag, is_field_signed) = match field_type {
            majit_ir::value::Type::Ref => (majit_ir::descr::ArrayFlag::Pointer, false),
            majit_ir::value::Type::Float => (majit_ir::descr::ArrayFlag::Float, false),
            _ => (majit_ir::descr::ArrayFlag::Signed, true),
        };
        self.add_bh_descr(CanonicalBhDescr::Field {
            offset,
            field_size: scalar_size(field_type),
            field_type,
            field_flag,
            is_field_signed,
            is_immutable,
            is_quasi_immutable: false,
            // Same as the vable synthesizers: no parent, so no slot claim.
            index_in_parent: None,
            parent: None,
            name: String::new(),
            owner: String::new(),
        })
    }

    /// Build a `Field` descr for a `setfield_gc_*` on a struct whose
    /// `new` already registered its layout under `type_id`.  Mirrors
    /// `descr.py get_field_descr`: the descr carries
    /// `index_in_parent` (`descr.py`) and `parent_descr =
    /// get_size_descr(STRUCT)` (`descr.py`).  The optimizer's
    /// `optimize_setfield_gc` (`optimizeopt/virtualize.rs`) requires
    /// the parent SizeDescr to virtualize the field; a missing
    /// registration falls back to the placeholder builder (an escaping
    /// allocation that is never virtualized).
    fn add_struct_field_descr(
        &mut self,
        offset: usize,
        field_type: majit_ir::value::Type,
        type_id: u64,
        field_name: &str,
    ) -> u16 {
        let Some(parent_spec) = self.struct_size_specs.get(&type_id) else {
            return self.add_scalar_field_descr(offset, field_type);
        };
        let (field_flag, is_field_signed) = match field_type {
            majit_ir::value::Type::Ref => (majit_ir::descr::ArrayFlag::Pointer, false),
            majit_ir::value::Type::Float => (majit_ir::descr::ArrayFlag::Float, false),
            _ => (majit_ir::descr::ArrayFlag::Signed, true),
        };
        // descr.py:228 index = heaptracker.get_fielddescr_index_in(STRUCT,
        // fieldname): resolve the field's structural slot against the cached
        // (offset-ordered) layout `new_struct` registered, not the caller's
        // store order.  descr.py:227 name = '%s.%s' % (STRUCT._name,
        // fieldname): take it from the same layout so the repr matches.
        //
        // `field_name` is the `fieldname` argument itself, carried from the
        // emit site — `jitcode_lower` already hands `register_struct_layout`
        // the same `stringify!(#member)` in the same expansion, so read and
        // write agree by construction.  It is what makes this a real
        // `get_fielddescr_index_in(STRUCT, fieldname)` rather than an
        // offset lookup wearing its name.
        //
        // The offset remains the fallback for a caller that has no name (`""`),
        // and it is not an identity: a flattened layout puts an inline
        // aggregate and its first leaf at one address (`heaptracker.py`).
        // Where the offset names two fields, take the unresolved fallback
        // rather than the first of them.  A guessed NAME is the damaging half:
        // it becomes the descr's `_cache_field` key downstream, so naming the
        // aggregate hands the leaf's access the aggregate's descr.
        let slot = field_slot_in(&parent_spec.all_fielddescrs, field_name, offset);
        // The miss stays `None`. Collapsing it to `0` here is the same defect
        // `fielddescrof` carried: slot 0 of the parent's list is some other
        // field, and once the descr is minted nothing downstream can tell a
        // failed lookup from a field genuinely at slot 0.
        let (index_in_parent, name) = slot
            .map(|idx| (Some(idx), parent_spec.all_fielddescrs[idx].name.clone()))
            .unwrap_or((None, String::new()));
        // The registered layout is the record of what `descr.py get_field_descr
        // get_field_descr` derives from FIELDTYPE, so a field it names supplies
        // its own width and signedness.  The IR bank the access lands in
        // cannot: every integer read arrives here as `Type::Int` whether the
        // field is a byte or a word.
        //
        // Immutability rides the same lookup.  `rclass.py` attaches the
        // `_immutable_fields_` rank to the field, so the registered layout is
        // where it lives and this is the only place the emitted access can
        // read it back.  A miss stays mutable: a field the layout does not
        // name has no declaration to honour.
        let (field_size, field_flag, is_field_signed, is_immutable, is_quasi_immutable) = slot
            .map(|idx| {
                let f = &parent_spec.all_fielddescrs[idx];
                (
                    f.field_size,
                    f.field_flag,
                    f.is_field_signed,
                    f.is_immutable,
                    f.is_quasi_immutable,
                )
            })
            .unwrap_or((
                scalar_size(field_type),
                field_flag,
                is_field_signed,
                false,
                false,
            ));
        // Carry the scalars only.  `patch_field_descr_parents`, called
        // unconditionally from `try_finish` after the decline early-return,
        // replaces this snapshot with `struct_size_specs`' final merged spec
        // for the same `type_id` — and entries there are only inserted or
        // merged, never removed, so a type_id present now is present then.
        // `type_id` is therefore the only load-bearing part; deep-copying
        // `all_fielddescrs` (a String per field) once per field-descr mint
        // would build a table that is overwritten before anything reads it.
        let parent = Some(BhSizeSpec {
            size: parent_spec.size,
            type_id: parent_spec.type_id,
            vtable: parent_spec.vtable,
            owner: parent_spec.owner.clone(),
            is_gc_managed: parent_spec.is_gc_managed,
            headerless: parent_spec.headerless,
            all_fielddescrs: Vec::new(),
        });
        self.add_bh_descr(CanonicalBhDescr::Field {
            offset,
            field_size,
            field_type,
            field_flag,
            is_field_signed,
            is_immutable,
            is_quasi_immutable,
            index_in_parent,
            parent: parent.map(std::sync::Arc::new),
            name,
            owner: String::new(),
        })
    }

    /// Emit `setfield_gc_i/rid` (`blackhole.py bhimpl_setfield_gc_i`):
    /// store the int in `value_reg` into `struct_reg`'s field at `offset`.
    /// `type_id` + `offset` identify the field against the layout
    /// `new_struct` registered, so the optimizer can virtualize the store.
    pub fn setfield_gc_i(
        &mut self,
        struct_reg: u16,
        value_reg: u16,
        offset: usize,
        type_id: u64,
        field_name: &str,
    ) {
        self.touch_ref_reg(struct_reg);
        self.touch_reg(value_reg);
        let descr =
            self.add_struct_field_descr(offset, majit_ir::value::Type::Int, type_id, field_name);
        self.write_insn("setfield_gc_i/rid");
        self.push_reg_u8(struct_reg, "setfield_gc_i struct");
        self.push_reg_u8(value_reg, "setfield_gc_i value");
        self.push_u16(descr);
    }

    /// Emit `setfield_gc_r/rrd` (`blackhole.py bhimpl_setfield_gc_r`):
    /// store the ref in `value_reg` into `struct_reg`'s field at `offset`.
    /// `type_id` + `offset` identify the field against the layout
    /// `new_struct` registered, so the optimizer can virtualize the store.
    pub fn setfield_gc_r(
        &mut self,
        struct_reg: u16,
        value_reg: u16,
        offset: usize,
        type_id: u64,
        field_name: &str,
    ) {
        self.touch_ref_reg(struct_reg);
        self.touch_ref_reg(value_reg);
        let descr =
            self.add_struct_field_descr(offset, majit_ir::value::Type::Ref, type_id, field_name);
        self.write_insn("setfield_gc_r/rrd");
        self.push_reg_u8(struct_reg, "setfield_gc_r struct");
        self.push_reg_u8(value_reg, "setfield_gc_r value");
        self.push_u16(descr);
    }

    /// Emit `setfield_gc_f/rfd` (`jtransform.py` `rewrite_op_setfield`:
    /// `kind = getkind(RESULT)[0]` is `f`, so the op is `setfield_gc_f`).
    ///
    /// The field's `FLOAT` flag is fixed when `register_struct_layout` builds
    /// the descr (`descr.py` `get_field_descr` / `get_type_flag`).
    pub fn setfield_gc_f(
        &mut self,
        struct_reg: u16,
        value_reg: u16,
        offset: usize,
        type_id: u64,
        field_name: &str,
    ) {
        self.touch_ref_reg(struct_reg);
        self.touch_float_reg(value_reg);
        let descr =
            self.add_struct_field_descr(offset, majit_ir::value::Type::Float, type_id, field_name);
        self.write_insn("setfield_gc_f/rfd");
        self.push_reg_u8(struct_reg, "setfield_gc_f struct");
        self.push_reg_u8(value_reg, "setfield_gc_f value");
        self.push_u16(descr);
    }

    /// `c`-argcode form of [`Self::setfield_gc_i`] —
    /// `assembler.py emit_const(allow_short=True)` writes a small
    /// ConstInt value (-128..127) inline as one signed byte (`setfield_gc_i`
    /// is in `USE_C_FORM`, `assembler.py`).
    ///
    /// Encoding: `[BC_SETFIELD_GC_I_C][struct_reg u8][value i8]
    ///             [descr_idx lo u8][descr_idx hi u8]`.
    pub fn setfield_gc_i_c(
        &mut self,
        struct_reg: u16,
        value: i8,
        offset: usize,
        type_id: u64,
        field_name: &str,
    ) {
        self.touch_ref_reg(struct_reg);
        let descr =
            self.add_struct_field_descr(offset, majit_ir::value::Type::Int, type_id, field_name);
        self.write_insn("setfield_gc_i/rcd");
        self.push_reg_u8(struct_reg, "setfield_gc_i_c struct");
        self.push_u8(value as u8);
        self.push_u16(descr);
    }

    /// Emit `getfield_gc_i/rd>i` (`blackhole.py bhimpl_getfield_gc_i`):
    /// load `struct_reg`'s int field at `offset` into `dest`.
    ///
    /// `type_id` resolves the parent-carrying struct field descr (the same
    /// `add_struct_field_descr` the matching `setfield_gc_i` uses), so a
    /// getfield and a setfield on the same `(type_id, offset)` intern the
    /// same `Field` descr and the heapcache can alias read against write.
    /// A `type_id` whose layout was never registered degrades to a parentless
    /// scalar descr (`add_struct_field_descr` returns the scalar form), which
    /// keeps existing callers correct.
    /// `getfield_gc_i/rd>i` for a field descr the flattener already built.
    ///
    /// `PyFrame.flags` is one byte. Rebuilding the descr from `Type::Int`
    /// would store a machine word and overrun the next field.
    pub fn getfield_gc_i_bh(&mut self, dest: u16, struct_reg: u16, field: CanonicalBhDescr) {
        self.touch_ref_reg(struct_reg);
        self.touch_reg(dest);
        let descr = self.add_bh_descr(field);
        self.write_insn("getfield_gc_i/rd>i");
        self.push_reg_u8(struct_reg, "getfield_gc_i struct");
        self.push_u16(descr);
        self.push_reg_u8(dest, "getfield_gc_i result");
    }

    /// `setfield_gc_i/rid` for a field descr the flattener already built.
    /// Same width constraint as [`Self::getfield_gc_i_bh`].
    pub fn setfield_gc_i_bh(&mut self, struct_reg: u16, value_reg: u16, field: CanonicalBhDescr) {
        self.touch_ref_reg(struct_reg);
        self.touch_reg(value_reg);
        let descr = self.add_bh_descr(field);
        self.write_insn("setfield_gc_i/rid");
        self.push_reg_u8(struct_reg, "setfield_gc_i struct");
        self.push_reg_u8(value_reg, "setfield_gc_i value");
        self.push_u16(descr);
    }

    pub fn getfield_gc_i(
        &mut self,
        dest: u16,
        struct_reg: u16,
        offset: usize,
        type_id: u64,
        field_name: &str,
    ) {
        self.touch_ref_reg(struct_reg);
        self.touch_reg(dest);
        let descr =
            self.add_struct_field_descr(offset, majit_ir::value::Type::Int, type_id, field_name);
        self.write_insn("getfield_gc_i/rd>i");
        self.push_reg_u8(struct_reg, "getfield_gc_i struct");
        self.push_u16(descr);
        self.push_reg_u8(dest, "getfield_gc_i result");
    }

    /// Emit `getfield_gc_i_pure/rd>i` (`blackhole.py
    /// bhimpl_getfield_gc_i_pure`): the same load as
    /// [`Self::getfield_gc_i`], off a field declared immutable. The bytecode
    /// pair carries no separate op-kind — the recorded `GetfieldGcI` carries
    /// the immutable descr, and that descr is what licenses the fold.
    pub fn getfield_gc_i_pure(&mut self, dest: u16, struct_reg: u16, offset: usize) {
        self.touch_ref_reg(struct_reg);
        self.touch_reg(dest);
        let descr =
            self.add_scalar_field_descr_with_immutability(offset, majit_ir::value::Type::Int, true);
        self.write_insn("getfield_gc_i_pure/rd>i");
        self.push_reg_u8(struct_reg, "getfield_gc_i_pure struct");
        self.push_u16(descr);
        self.push_reg_u8(dest, "getfield_gc_i_pure result");
    }

    /// Emit `getfield_gc_r/rd>r` (`blackhole.py bhimpl_getfield_gc_r`):
    /// load `struct_reg`'s ref field at `offset` into `dest`.
    ///
    /// See [`Self::getfield_gc_i`] for the `type_id` parent-descr contract.
    pub fn getfield_gc_r(
        &mut self,
        dest: u16,
        struct_reg: u16,
        offset: usize,
        type_id: u64,
        field_name: &str,
    ) {
        self.touch_ref_reg(struct_reg);
        self.touch_ref_reg(dest);
        let descr =
            self.add_struct_field_descr(offset, majit_ir::value::Type::Ref, type_id, field_name);
        self.write_insn("getfield_gc_r/rd>r");
        self.push_reg_u8(struct_reg, "getfield_gc_r struct");
        self.push_u16(descr);
        self.push_reg_u8(dest, "getfield_gc_r result");
    }

    /// Emit `getfield_gc_f/rd>f` (`jtransform.py` `rewrite_op_getfield`:
    /// `getkind` of a `lltype.Float` field is `f`).
    ///
    /// Pair of [`Self::setfield_gc_f`]. Both intern the field descr as
    /// `Type::Float` (`descr.py` `get_type_flag` `FLAG_FLOAT`).
    pub fn getfield_gc_f(
        &mut self,
        dest: u16,
        struct_reg: u16,
        offset: usize,
        type_id: u64,
        field_name: &str,
    ) {
        self.touch_ref_reg(struct_reg);
        self.touch_float_reg(dest);
        let descr =
            self.add_struct_field_descr(offset, majit_ir::value::Type::Float, type_id, field_name);
        self.write_insn("getfield_gc_f/rd>f");
        self.push_reg_u8(struct_reg, "getfield_gc_f struct");
        self.push_u16(descr);
        self.push_reg_u8(dest, "getfield_gc_f result");
    }

    pub fn vable_getfield_int_with_base(&mut self, dest: u16, vable_reg: u16, field_idx: u16) {
        self.touch_ref_reg(vable_reg);
        self.touch_reg(dest);
        let field_descr = self.add_vable_field_descr(field_idx);
        self.write_insn("getfield_vable_i/rd>i");
        self.push_reg_u8(vable_reg, "getfield_vable_i base");
        self.push_u16(field_descr);
        self.push_reg_u8(dest, "getfield_vable_i result");
    }

    pub fn vable_getfield_ref_with_base(&mut self, dest: u16, vable_reg: u16, field_idx: u16) {
        self.touch_ref_reg(vable_reg);
        self.touch_ref_reg(dest);
        let field_descr = self.add_vable_field_descr(field_idx);
        self.write_insn("getfield_vable_r/rd>r");
        self.push_reg_u8(vable_reg, "getfield_vable_r base");
        self.push_u16(field_descr);
        self.push_reg_u8(dest, "getfield_vable_r result");
    }

    pub fn vable_getfield_float_with_base(&mut self, dest: u16, vable_reg: u16, field_idx: u16) {
        self.touch_ref_reg(vable_reg);
        self.touch_float_reg(dest);
        let field_descr = self.add_vable_field_descr(field_idx);
        self.write_insn("getfield_vable_f/rd>f");
        self.push_reg_u8(vable_reg, "getfield_vable_f base");
        self.push_u16(field_descr);
        self.push_reg_u8(dest, "getfield_vable_f result");
    }

    pub fn vable_setfield_int_with_base(&mut self, vable_reg: u16, field_idx: u16, src: u16) {
        self.touch_ref_reg(vable_reg);
        self.touch_reg(src);
        let field_descr = self.add_vable_field_descr(field_idx);
        self.write_insn("setfield_vable_i/rid");
        self.push_reg_u8(vable_reg, "setfield_vable_i base");
        self.push_reg_u8(src, "setfield_vable_i value");
        self.push_u16(field_descr);
    }

    pub fn vable_setfield_ref_with_base(&mut self, vable_reg: u16, field_idx: u16, src: u16) {
        self.touch_ref_reg(vable_reg);
        self.touch_ref_reg(src);
        let field_descr = self.add_vable_field_descr(field_idx);
        self.write_insn("setfield_vable_r/rrd");
        self.push_reg_u8(vable_reg, "setfield_vable_r base");
        self.push_reg_u8(src, "setfield_vable_r value");
        self.push_u16(field_descr);
    }

    pub fn vable_setfield_ref_const_value_with_base(
        &mut self,
        vable_reg: u16,
        field_idx: u16,
        value: i64,
    ) {
        let const_idx = self.add_const_r(value);
        self.touch_ref_reg(vable_reg);
        let field_descr = self.add_vable_field_descr(field_idx);
        self.write_insn("setfield_vable_r/rrd");
        self.push_reg_u8(vable_reg, "setfield_vable_r base");
        let src_offset = self.code.len();
        self.push_u8(0);
        self.const_patches_u8
            .push((src_offset, ConstKind::Ref, const_idx));
        self.push_u16(field_descr);
    }

    /// pyjitpl.py `_opimpl_setfield_vable` accepts any box for
    /// `valuebox` (i/r/f variants share the same generic body). The
    /// optimizer may fold a register source down to `ConstInt`; the
    /// assembler must accept that as a constant-pool patch matching
    /// the `_r` variant's pattern.
    pub fn vable_setfield_int_const_value_with_base(
        &mut self,
        vable_reg: u16,
        field_idx: u16,
        value: i64,
    ) {
        // `setfield_vable_i` is not in `USE_C_FORM` (`assembler.py`), so a
        // constant value would take a `constants_i` slot. One slot per
        // Python instruction (`dispatch_bytecode` `last_instr`) overflows
        // `check_result`'s 256-wide int index space. The immediate form
        // writes the u32 inline and shares the `VableField` descr with
        // `setfield_vable_i/rid`. Blackhole `handler_setfield_vable_i_imm`
        // and the walker's `setfield_vable_i_imm/rddd` arm execute it as
        // `_opimpl_setfield_vable`.
        let imm = u32::try_from(value)
            .unwrap_or_else(|_| panic!("setfield_vable_i immediate {value} does not fit in u32"));
        self.touch_ref_reg(vable_reg);
        let field_descr = self.add_vable_field_descr(field_idx);
        self.write_insn("setfield_vable_i_imm/rddd");
        self.push_reg_u8(vable_reg, "setfield_vable_i base");
        self.push_u16(imm as u16);
        self.push_u16((imm >> 16) as u16);
        self.push_u16(field_descr);
    }

    /// pyjitpl.py:1188-1199 — float counterpart to
    /// `vable_setfield_int_const_value_with_base`.
    pub fn vable_setfield_float_const_value_with_base(
        &mut self,
        vable_reg: u16,
        field_idx: u16,
        value: i64,
    ) {
        let const_idx = self.add_const_f(value);
        self.touch_ref_reg(vable_reg);
        let field_descr = self.add_vable_field_descr(field_idx);
        self.write_insn("setfield_vable_f/rfd");
        self.push_reg_u8(vable_reg, "setfield_vable_f base");
        let src_offset = self.code.len();
        self.push_u8(0);
        self.const_patches_u8
            .push((src_offset, ConstKind::Float, const_idx));
        self.push_u16(field_descr);
    }

    pub fn vable_setfield_float_with_base(&mut self, vable_reg: u16, field_idx: u16, src: u16) {
        self.touch_ref_reg(vable_reg);
        self.touch_float_reg(src);
        let field_descr = self.add_vable_field_descr(field_idx);
        self.write_insn("setfield_vable_f/rfd");
        self.push_reg_u8(vable_reg, "setfield_vable_f base");
        self.push_reg_u8(src, "setfield_vable_f value");
        self.push_u16(field_descr);
    }

    pub fn vable_getarrayitem_int_with_base(
        &mut self,
        dest: u16,
        vable_reg: u16,
        array_idx: u16,
        index_reg: u16,
    ) {
        self.touch_ref_reg(vable_reg);
        self.touch_reg(index_reg);
        self.touch_reg(dest);
        let field_descr = self.add_vable_array_field_descr(array_idx);
        let array_descr = self.add_vable_array_descr(majit_ir::value::Type::Int, true);
        self.write_insn("getarrayitem_vable_i/ridd>i");
        self.push_reg_u8(vable_reg, "getarrayitem_vable_i base");
        self.push_reg_u8(index_reg, "getarrayitem_vable_i index");
        self.push_u16(field_descr);
        self.push_u16(array_descr);
        self.push_reg_u8(dest, "getarrayitem_vable_i result");
    }

    pub fn vable_getarrayitem_ref_with_base(
        &mut self,
        dest: u16,
        vable_reg: u16,
        array_idx: u16,
        index_reg: u16,
    ) {
        self.touch_ref_reg(vable_reg);
        self.touch_reg(index_reg);
        self.touch_ref_reg(dest);
        let field_descr = self.add_vable_array_field_descr(array_idx);
        let array_descr = self.add_vable_array_descr(majit_ir::value::Type::Ref, false);
        self.write_insn("getarrayitem_vable_r/ridd>r");
        self.push_reg_u8(vable_reg, "getarrayitem_vable_r base");
        self.push_reg_u8(index_reg, "getarrayitem_vable_r index");
        self.push_u16(field_descr);
        self.push_u16(array_descr);
        self.push_reg_u8(dest, "getarrayitem_vable_r result");
    }

    /// `getarrayitem_vable_r` with a ConstInt index.  Mirrors
    /// `jtransform.py do_fixed_list_getitem` (vable branch)
    /// when the index is folded to a `ConstInt` by the optimizer.
    /// Patches the index slot via the constants pool.
    pub fn vable_getarrayitem_ref_const_idx_with_base(
        &mut self,
        dest: u16,
        vable_reg: u16,
        array_idx: u16,
        index_value: i64,
    ) {
        let const_idx = self.add_const_i(index_value);
        self.touch_ref_reg(vable_reg);
        self.touch_ref_reg(dest);
        let field_descr = self.add_vable_array_field_descr(array_idx);
        let array_descr = self.add_vable_array_descr(majit_ir::value::Type::Ref, false);
        self.write_insn("getarrayitem_vable_r/ridd>r");
        self.push_reg_u8(vable_reg, "getarrayitem_vable_r base");
        let idx_offset = self.code.len();
        self.push_u8(0);
        self.const_patches_u8
            .push((idx_offset, ConstKind::Int, const_idx));
        self.push_u16(field_descr);
        self.push_u16(array_descr);
        self.push_reg_u8(dest, "getarrayitem_vable_r result");
    }

    /// `getarrayitem_vable_i` with a ConstInt index.  Parity with
    /// `vable_getarrayitem_ref_const_idx_with_base` —
    /// `rpython/jit/codewriter/assembler.py:162` accepts Constant
    /// operands generically across all element types.
    pub fn vable_getarrayitem_int_const_idx_with_base(
        &mut self,
        dest: u16,
        vable_reg: u16,
        array_idx: u16,
        index_value: i64,
    ) {
        let const_idx = self.add_const_i(index_value);
        self.touch_ref_reg(vable_reg);
        self.touch_reg(dest);
        let field_descr = self.add_vable_array_field_descr(array_idx);
        let array_descr = self.add_vable_array_descr(majit_ir::value::Type::Int, true);
        self.write_insn("getarrayitem_vable_i/ridd>i");
        self.push_reg_u8(vable_reg, "getarrayitem_vable_i base");
        let idx_offset = self.code.len();
        self.push_u8(0);
        self.const_patches_u8
            .push((idx_offset, ConstKind::Int, const_idx));
        self.push_u16(field_descr);
        self.push_u16(array_descr);
        self.push_reg_u8(dest, "getarrayitem_vable_i result");
    }

    pub fn vable_getarrayitem_float_with_base(
        &mut self,
        dest: u16,
        vable_reg: u16,
        array_idx: u16,
        index_reg: u16,
    ) {
        self.touch_ref_reg(vable_reg);
        self.touch_reg(index_reg);
        self.touch_float_reg(dest);
        let field_descr = self.add_vable_array_field_descr(array_idx);
        let array_descr = self.add_vable_array_descr(majit_ir::value::Type::Float, false);
        self.write_insn("getarrayitem_vable_f/ridd>f");
        self.push_reg_u8(vable_reg, "getarrayitem_vable_f base");
        self.push_reg_u8(index_reg, "getarrayitem_vable_f index");
        self.push_u16(field_descr);
        self.push_u16(array_descr);
        self.push_reg_u8(dest, "getarrayitem_vable_f result");
    }

    /// `getarrayitem_vable_f` with a ConstInt index.  Parity with
    /// `vable_getarrayitem_ref_const_idx_with_base` per
    /// `rpython/jit/codewriter/assembler.py:162`.
    pub fn vable_getarrayitem_float_const_idx_with_base(
        &mut self,
        dest: u16,
        vable_reg: u16,
        array_idx: u16,
        index_value: i64,
    ) {
        let const_idx = self.add_const_i(index_value);
        self.touch_ref_reg(vable_reg);
        self.touch_float_reg(dest);
        let field_descr = self.add_vable_array_field_descr(array_idx);
        let array_descr = self.add_vable_array_descr(majit_ir::value::Type::Float, false);
        self.write_insn("getarrayitem_vable_f/ridd>f");
        self.push_reg_u8(vable_reg, "getarrayitem_vable_f base");
        let idx_offset = self.code.len();
        self.push_u8(0);
        self.const_patches_u8
            .push((idx_offset, ConstKind::Int, const_idx));
        self.push_u16(field_descr);
        self.push_u16(array_descr);
        self.push_reg_u8(dest, "getarrayitem_vable_f result");
    }

    pub fn vable_setarrayitem_int_with_base(
        &mut self,
        vable_reg: u16,
        array_idx: u16,
        index_reg: u16,
        src: u16,
    ) {
        self.touch_ref_reg(vable_reg);
        self.touch_reg(index_reg);
        self.touch_reg(src);
        let field_descr = self.add_vable_array_field_descr(array_idx);
        let array_descr = self.add_vable_array_descr(majit_ir::value::Type::Int, true);
        self.write_insn("setarrayitem_vable_i/riidd");
        self.push_reg_u8(vable_reg, "setarrayitem_vable_i base");
        self.push_reg_u8(index_reg, "setarrayitem_vable_i index");
        self.push_reg_u8(src, "setarrayitem_vable_i value");
        self.push_u16(field_descr);
        self.push_u16(array_descr);
    }

    pub fn vable_setarrayitem_ref_with_base(
        &mut self,
        vable_reg: u16,
        array_idx: u16,
        index_reg: u16,
        src: u16,
    ) {
        self.touch_ref_reg(vable_reg);
        self.touch_reg(index_reg);
        self.touch_ref_reg(src);
        let field_descr = self.add_vable_array_field_descr(array_idx);
        let array_descr = self.add_vable_array_descr(majit_ir::value::Type::Ref, false);
        self.write_insn("setarrayitem_vable_r/rirdd");
        self.push_reg_u8(vable_reg, "setarrayitem_vable_r base");
        self.push_reg_u8(index_reg, "setarrayitem_vable_r index");
        self.push_reg_u8(src, "setarrayitem_vable_r value");
        self.push_u16(field_descr);
        self.push_u16(array_descr);
    }

    pub fn vable_setarrayitem_ref_const_value_with_base(
        &mut self,
        vable_reg: u16,
        array_idx: u16,
        index_reg: u16,
        value: i64,
    ) {
        let const_idx = self.add_const_r(value);
        self.touch_ref_reg(vable_reg);
        self.touch_reg(index_reg);
        let field_descr = self.add_vable_array_field_descr(array_idx);
        let array_descr = self.add_vable_array_descr(majit_ir::value::Type::Ref, false);
        self.write_insn("setarrayitem_vable_r/rirdd");
        self.push_reg_u8(vable_reg, "setarrayitem_vable_r base");
        self.push_reg_u8(index_reg, "setarrayitem_vable_r index");
        let src_offset = self.code.len();
        self.push_u8(0);
        self.const_patches_u8
            .push((src_offset, ConstKind::Ref, const_idx));
        self.push_u16(field_descr);
        self.push_u16(array_descr);
    }

    /// `setarrayitem_vable_r` with a ConstInt-sourced index.  Mirrors
    /// `jtransform.py do_fixed_list_setitem` (vable branch) when
    /// the index is folded to a `ConstInt` by the optimizer.  Patches
    /// the index slot via the constants pool so the runtime
    /// `BC_SETARRAYITEM_VABLE_R` reads the index from the unified
    /// register space's constants suffix.
    pub fn vable_setarrayitem_ref_const_idx_with_base(
        &mut self,
        vable_reg: u16,
        array_idx: u16,
        index_value: i64,
        src: u16,
    ) {
        let const_idx = self.add_const_i(index_value);
        self.touch_ref_reg(vable_reg);
        self.touch_ref_reg(src);
        let field_descr = self.add_vable_array_field_descr(array_idx);
        let array_descr = self.add_vable_array_descr(majit_ir::value::Type::Ref, false);
        self.write_insn("setarrayitem_vable_r/rirdd");
        self.push_reg_u8(vable_reg, "setarrayitem_vable_r base");
        let idx_offset = self.code.len();
        self.push_u8(0);
        self.const_patches_u8
            .push((idx_offset, ConstKind::Int, const_idx));
        self.push_reg_u8(src, "setarrayitem_vable_r value");
        self.push_u16(field_descr);
        self.push_u16(array_descr);
    }

    /// `setarrayitem_vable_r` with both ConstInt index AND ConstRef
    /// source — the all-constant variant produced when both index and
    /// value are folded.  Patches both slots via constants pool.
    pub fn vable_setarrayitem_ref_const_idx_const_value_with_base(
        &mut self,
        vable_reg: u16,
        array_idx: u16,
        index_value: i64,
        src_value: i64,
    ) {
        let const_idx_int = self.add_const_i(index_value);
        let const_idx_ref = self.add_const_r(src_value);
        self.touch_ref_reg(vable_reg);
        let field_descr = self.add_vable_array_field_descr(array_idx);
        let array_descr = self.add_vable_array_descr(majit_ir::value::Type::Ref, false);
        self.write_insn("setarrayitem_vable_r/rirdd");
        self.push_reg_u8(vable_reg, "setarrayitem_vable_r base");
        let idx_offset = self.code.len();
        self.push_u8(0);
        self.const_patches_u8
            .push((idx_offset, ConstKind::Int, const_idx_int));
        let src_offset = self.code.len();
        self.push_u8(0);
        self.const_patches_u8
            .push((src_offset, ConstKind::Ref, const_idx_ref));
        self.push_u16(field_descr);
        self.push_u16(array_descr);
    }

    /// pyjitpl.py `_opimpl_setarrayitem_vable` accepts any
    /// box for `valuebox`; the optimizer may fold the source down to
    /// `ConstInt`.  See `vable_setfield_int_const_value_with_base`
    /// for the parity rationale.
    pub fn vable_setarrayitem_int_const_value_with_base(
        &mut self,
        vable_reg: u16,
        array_idx: u16,
        index_reg: u16,
        value: i64,
    ) {
        let const_idx = self.add_const_i(value);
        self.touch_ref_reg(vable_reg);
        self.touch_reg(index_reg);
        let field_descr = self.add_vable_array_field_descr(array_idx);
        let array_descr = self.add_vable_array_descr(majit_ir::value::Type::Int, true);
        self.write_insn("setarrayitem_vable_i/riidd");
        self.push_reg_u8(vable_reg, "setarrayitem_vable_i base");
        self.push_reg_u8(index_reg, "setarrayitem_vable_i index");
        let src_offset = self.code.len();
        self.push_u8(0);
        self.const_patches_u8
            .push((src_offset, ConstKind::Int, const_idx));
        self.push_u16(field_descr);
        self.push_u16(array_descr);
    }

    /// `setarrayitem_vable_i` with a ConstInt-sourced index.  Parity
    /// with `vable_setarrayitem_ref_const_idx_with_base` per
    /// `rpython/jit/codewriter/assembler.py:162`.
    pub fn vable_setarrayitem_int_const_idx_with_base(
        &mut self,
        vable_reg: u16,
        array_idx: u16,
        index_value: i64,
        src: u16,
    ) {
        let const_idx = self.add_const_i(index_value);
        self.touch_ref_reg(vable_reg);
        self.touch_reg(src);
        let field_descr = self.add_vable_array_field_descr(array_idx);
        let array_descr = self.add_vable_array_descr(majit_ir::value::Type::Int, true);
        self.write_insn("setarrayitem_vable_i/riidd");
        self.push_reg_u8(vable_reg, "setarrayitem_vable_i base");
        let idx_offset = self.code.len();
        self.push_u8(0);
        self.const_patches_u8
            .push((idx_offset, ConstKind::Int, const_idx));
        self.push_reg_u8(src, "setarrayitem_vable_i value");
        self.push_u16(field_descr);
        self.push_u16(array_descr);
    }

    /// `setarrayitem_vable_i` with both ConstInt index AND ConstInt
    /// source — the all-constant variant produced when both index and
    /// value are folded.  Parity with
    /// `vable_setarrayitem_ref_const_idx_const_value_with_base` per
    /// `rpython/jit/codewriter/assembler.py:162` (each operand slot
    /// accepts Register or Constant independently).
    pub fn vable_setarrayitem_int_const_idx_const_value_with_base(
        &mut self,
        vable_reg: u16,
        array_idx: u16,
        index_value: i64,
        src_value: i64,
    ) {
        let const_idx_int = self.add_const_i(index_value);
        let const_src_int = self.add_const_i(src_value);
        self.touch_ref_reg(vable_reg);
        let field_descr = self.add_vable_array_field_descr(array_idx);
        let array_descr = self.add_vable_array_descr(majit_ir::value::Type::Int, true);
        self.write_insn("setarrayitem_vable_i/riidd");
        self.push_reg_u8(vable_reg, "setarrayitem_vable_i base");
        let idx_offset = self.code.len();
        self.push_u8(0);
        self.const_patches_u8
            .push((idx_offset, ConstKind::Int, const_idx_int));
        let src_offset = self.code.len();
        self.push_u8(0);
        self.const_patches_u8
            .push((src_offset, ConstKind::Int, const_src_int));
        self.push_u16(field_descr);
        self.push_u16(array_descr);
    }

    /// pyjitpl.py:1236-1247 — float counterpart to
    /// `vable_setarrayitem_int_const_value_with_base`.
    pub fn vable_setarrayitem_float_const_value_with_base(
        &mut self,
        vable_reg: u16,
        array_idx: u16,
        index_reg: u16,
        value: i64,
    ) {
        let const_idx = self.add_const_f(value);
        self.touch_ref_reg(vable_reg);
        self.touch_reg(index_reg);
        let field_descr = self.add_vable_array_field_descr(array_idx);
        let array_descr = self.add_vable_array_descr(majit_ir::value::Type::Float, false);
        self.write_insn("setarrayitem_vable_f/rifdd");
        self.push_reg_u8(vable_reg, "setarrayitem_vable_f base");
        self.push_reg_u8(index_reg, "setarrayitem_vable_f index");
        let src_offset = self.code.len();
        self.push_u8(0);
        self.const_patches_u8
            .push((src_offset, ConstKind::Float, const_idx));
        self.push_u16(field_descr);
        self.push_u16(array_descr);
    }

    pub fn vable_setarrayitem_float_with_base(
        &mut self,
        vable_reg: u16,
        array_idx: u16,
        index_reg: u16,
        src: u16,
    ) {
        self.touch_ref_reg(vable_reg);
        self.touch_reg(index_reg);
        self.touch_float_reg(src);
        let field_descr = self.add_vable_array_field_descr(array_idx);
        let array_descr = self.add_vable_array_descr(majit_ir::value::Type::Float, false);
        self.write_insn("setarrayitem_vable_f/rifdd");
        self.push_reg_u8(vable_reg, "setarrayitem_vable_f base");
        self.push_reg_u8(index_reg, "setarrayitem_vable_f index");
        self.push_reg_u8(src, "setarrayitem_vable_f value");
        self.push_u16(field_descr);
        self.push_u16(array_descr);
    }

    /// `setarrayitem_vable_f` with a ConstInt-sourced index.  Parity
    /// with `vable_setarrayitem_ref_const_idx_with_base` per
    /// `rpython/jit/codewriter/assembler.py:162`.
    pub fn vable_setarrayitem_float_const_idx_with_base(
        &mut self,
        vable_reg: u16,
        array_idx: u16,
        index_value: i64,
        src: u16,
    ) {
        let const_idx = self.add_const_i(index_value);
        self.touch_ref_reg(vable_reg);
        self.touch_float_reg(src);
        let field_descr = self.add_vable_array_field_descr(array_idx);
        let array_descr = self.add_vable_array_descr(majit_ir::value::Type::Float, false);
        self.write_insn("setarrayitem_vable_f/rifdd");
        self.push_reg_u8(vable_reg, "setarrayitem_vable_f base");
        let idx_offset = self.code.len();
        self.push_u8(0);
        self.const_patches_u8
            .push((idx_offset, ConstKind::Int, const_idx));
        self.push_reg_u8(src, "setarrayitem_vable_f value");
        self.push_u16(field_descr);
        self.push_u16(array_descr);
    }

    /// `setarrayitem_vable_f` with both ConstInt index AND ConstFloat
    /// source — the all-constant variant produced when both index and
    /// value are folded.  Parity with
    /// `vable_setarrayitem_ref_const_idx_const_value_with_base` per
    /// `rpython/jit/codewriter/assembler.py:162` (each operand slot
    /// accepts Register or Constant independently).
    pub fn vable_setarrayitem_float_const_idx_const_value_with_base(
        &mut self,
        vable_reg: u16,
        array_idx: u16,
        index_value: i64,
        src_value: i64,
    ) {
        let const_idx_int = self.add_const_i(index_value);
        let const_src_float = self.add_const_f(src_value);
        self.touch_ref_reg(vable_reg);
        let field_descr = self.add_vable_array_field_descr(array_idx);
        let array_descr = self.add_vable_array_descr(majit_ir::value::Type::Float, false);
        self.write_insn("setarrayitem_vable_f/rifdd");
        self.push_reg_u8(vable_reg, "setarrayitem_vable_f base");
        let idx_offset = self.code.len();
        self.push_u8(0);
        self.const_patches_u8
            .push((idx_offset, ConstKind::Int, const_idx_int));
        let src_offset = self.code.len();
        self.push_u8(0);
        self.const_patches_u8
            .push((src_offset, ConstKind::Float, const_src_float));
        self.push_u16(field_descr);
        self.push_u16(array_descr);
    }

    pub fn vable_arraylen_with_base(&mut self, dest: u16, vable_reg: u16, array_idx: u16) {
        self.touch_ref_reg(vable_reg);
        self.touch_reg(dest);
        let field_descr = self.add_vable_array_field_descr(array_idx);
        let array_descr = self.add_vable_array_descr(majit_ir::value::Type::Ref, false);
        self.write_insn("arraylen_vable/rdd>i");
        self.push_reg_u8(vable_reg, "arraylen_vable base");
        self.push_u16(field_descr);
        self.push_u16(array_descr);
        self.push_reg_u8(dest, "arraylen_vable result");
    }

    /// Address of item 0 of virtualizable array `array_idx`, as an int.
    ///
    /// Operand layout is byte-for-byte the `arraylen_vable` triple above, so
    /// both decode through `read_vable_arraylen`.
    ///
    /// Emitting this is an escape: the address outlives the op and lets a
    /// callee mutate items represented by the trace's SSA values. Callers must
    /// pair it with the `TraceCtx` raw-base escape signal — the macro's
    /// lowering is the only caller, and it does.
    pub fn vable_arraybase_with_base(&mut self, dest: u16, vable_reg: u16, array_idx: u16) {
        self.touch_ref_reg(vable_reg);
        self.touch_reg(dest);
        let field_descr = self.add_vable_array_field_descr(array_idx);
        let array_descr = self.add_vable_array_descr(majit_ir::value::Type::Ref, false);
        self.write_insn("arraybase_vable/rdd>i");
        self.push_reg_u8(vable_reg, "arraybase_vable base");
        self.push_u16(field_descr);
        self.push_u16(array_descr);
        self.push_reg_u8(dest, "arraybase_vable result");
    }

    pub fn vable_force_with_base(&mut self, vable_reg: u16) {
        self.touch_ref_reg(vable_reg);
        self.write_insn("hint_force_virtualizable/r");
        self.push_reg_u8(vable_reg, "hint_force_virtualizable base");
    }

    /// Load an integer element from a GC-managed array.
    ///
    /// blackhole.py `bhimpl_getarrayitem_gc_i @arguments("r","i","d",returns="i")`:
    /// reads `registers_r[array_reg]` as the array pointer,
    /// `registers_i[index_reg]` as the element index, and `descrs[descr_idx]`
    /// as the array descriptor; writes the result into `registers_i[dst]`.
    ///
    /// Encoding: `[BC_GETARRAYITEM_GC_I][array_reg u8][index_reg u8]
    ///             [descr_idx lo u8][descr_idx hi u8][dst u8]`.
    ///
    /// Used by the dispatch JitCode body to encode `let opcode = program[pc]`
    /// (pyopcode.py `ord(co_code[next_instr])`).
    /// `blackhole.py` `bhimpl_strlen`: `[string_reg][dst]`, `strlen/r>i`.
    ///
    /// Records `OpCode::Strlen` on `rstr.STR` (hash, length, chars). No descr
    /// byte: `inject_builtin_string_descrs` stamps `str_descr` at compile.
    pub fn strlen(&mut self, dst: u16, string_reg: u16) {
        self.touch_ref_reg(string_reg);
        self.touch_reg(dst);
        self.write_insn("strlen/r>i");
        self.push_reg_u8(string_reg, "strlen string");
        self.push_reg_u8(dst, "strlen dst");
    }

    /// `blackhole.py` `bhimpl_strgetitem`: `[string_reg][index_reg][dst]`,
    /// `strgetitem/ri>i`.
    pub fn strgetitem(&mut self, dst: u16, string_reg: u16, index_reg: u16) {
        self.touch_ref_reg(string_reg);
        self.touch_reg(index_reg);
        self.touch_reg(dst);
        self.write_insn("strgetitem/ri>i");
        self.push_reg_u8(string_reg, "strgetitem string");
        self.push_reg_u8(index_reg, "strgetitem index");
        self.push_reg_u8(dst, "strgetitem dst");
    }

    pub fn getarrayitem_gc_i(&mut self, dst: u16, array_reg: u16, index_reg: u16, descr_idx: u16) {
        self.touch_ref_reg(array_reg);
        self.touch_reg(index_reg);
        self.touch_reg(dst);
        self.write_insn("getarrayitem_gc_i/rid>i");
        self.push_reg_u8(array_reg, "getarrayitem_gc_i array");
        self.push_reg_u8(index_reg, "getarrayitem_gc_i index");
        self.push_u16(descr_idx);
        self.push_reg_u8(dst, "getarrayitem_gc_i dst");
    }

    /// `blackhole.py` `bhimpl_getarrayitem_raw_i`: `[array_reg][index_reg][descr][dst]`,
    /// `getarrayitem_raw_i/iid>i`. Int-bank base (raw address).
    /// `jtransform.py` `rewrite_op_getarrayitem` raw arm.
    pub fn getarrayitem_raw_i(&mut self, dst: u16, array_reg: u16, index_reg: u16, descr_idx: u16) {
        self.touch_reg(array_reg);
        self.touch_reg(index_reg);
        self.touch_reg(dst);
        self.write_insn("getarrayitem_raw_i/iid>i");
        self.push_reg_u8(array_reg, "getarrayitem_raw_i array");
        self.push_reg_u8(index_reg, "getarrayitem_raw_i index");
        self.push_u16(descr_idx);
        self.push_reg_u8(dst, "getarrayitem_raw_i dst");
    }

    /// `blackhole.py` `bhimpl_setarrayitem_raw_i`: `[array_reg][index_reg][value_reg][descr]`,
    /// `setarrayitem_raw_i/iiid`. Int-bank base (raw address).
    /// `jtransform.py` `rewrite_op_setarrayitem` raw arm.
    pub fn setarrayitem_raw_i(
        &mut self,
        array_reg: u16,
        index_reg: u16,
        value_reg: u16,
        descr_idx: u16,
    ) {
        self.touch_reg(array_reg);
        self.touch_reg(index_reg);
        self.touch_reg(value_reg);
        self.write_insn("setarrayitem_raw_i/iiid");
        self.push_reg_u8(array_reg, "setarrayitem_raw_i array");
        self.push_reg_u8(index_reg, "setarrayitem_raw_i index");
        self.push_reg_u8(value_reg, "setarrayitem_raw_i value");
        self.push_u16(descr_idx);
    }

    /// `blackhole.py` `bhimpl_getarrayitem_raw_f`: `[array_reg][index_reg][descr][dst]`,
    /// `getarrayitem_raw_f/iid>f`. Int-bank base (raw address), float dest.
    /// `jtransform.py` `rewrite_op_getarrayitem` raw arm when
    /// `getkind(ARRAY.OF)` is float. The descr is
    /// [`Self::add_raw_float_array_descr`].
    pub fn getarrayitem_raw_f(&mut self, dst: u16, array_reg: u16, index_reg: u16, descr_idx: u16) {
        self.touch_reg(array_reg);
        self.touch_reg(index_reg);
        self.touch_float_reg(dst);
        self.write_insn("getarrayitem_raw_f/iid>f");
        self.push_reg_u8(array_reg, "getarrayitem_raw_f array");
        self.push_reg_u8(index_reg, "getarrayitem_raw_f index");
        self.push_u16(descr_idx);
        self.push_reg_u8(dst, "getarrayitem_raw_f dst");
    }

    /// `blackhole.py` `bhimpl_setarrayitem_raw_f`: `[array_reg][index_reg][value_reg][descr]`,
    /// `setarrayitem_raw_f/iifd`. Int-bank base (raw address), float value.
    /// `jtransform.py` `rewrite_op_setarrayitem` raw arm when the stored
    /// value's kind is float.
    pub fn setarrayitem_raw_f(
        &mut self,
        array_reg: u16,
        index_reg: u16,
        value_reg: u16,
        descr_idx: u16,
    ) {
        self.touch_reg(array_reg);
        self.touch_reg(index_reg);
        self.touch_float_reg(value_reg);
        self.write_insn("setarrayitem_raw_f/iifd");
        self.push_reg_u8(array_reg, "setarrayitem_raw_f array");
        self.push_reg_u8(index_reg, "setarrayitem_raw_f index");
        self.push_reg_u8(value_reg, "setarrayitem_raw_f value");
        self.push_u16(descr_idx);
    }

    /// The always-pure spelling of [`Self::getarrayitem_gc_i`], for an array
    /// whose elements never change after it is built.
    ///
    /// `blackhole.py` aliases `bhimpl_getarrayitem_gc_i_pure` onto the plain
    /// impl, so the two run the same handler; what differs is the opcode the
    /// tracer records. `GetarrayitemGcPureI` is inside the always-pure range
    /// that `OpHelpers.is_pure_with_descr` admits, which is what licenses the
    /// record-time constant fold of a read at a green index — the same fold
    /// `rlist.py ll_getitem_foldable_nonneg` earns by spelling the read this
    /// way. Emitting the plain opcode and folding it anyway would fold on the
    /// accident of who the caller happens to be.
    ///
    /// Encoding is [`Self::getarrayitem_gc_i`]'s.
    pub fn getarrayitem_gc_i_pure(
        &mut self,
        dst: u16,
        array_reg: u16,
        index_reg: u16,
        descr_idx: u16,
    ) {
        self.touch_ref_reg(array_reg);
        self.touch_reg(index_reg);
        self.touch_reg(dst);
        self.write_insn("getarrayitem_gc_i_pure/rid>i");
        self.push_reg_u8(array_reg, "getarrayitem_gc_i_pure array");
        self.push_reg_u8(index_reg, "getarrayitem_gc_i_pure index");
        self.push_u16(descr_idx);
        self.push_reg_u8(dst, "getarrayitem_gc_i_pure dst");
    }

    /// Ref-result array read: `dst = array[index]` where the element is a
    /// GC pointer (8 bytes). Mirrors [`Self::getarrayitem_gc_i`] but routes
    /// `dst` through the ref register bank and uses the `/rid>r` mnemonic.
    /// Used to re-derive a pointer-array element (a `pools[selected]`-shaped
    /// read) as a re-producible heap read instead of an opaque residual call.
    pub fn getarrayitem_gc_r(&mut self, dst: u16, array_reg: u16, index_reg: u16, descr_idx: u16) {
        self.touch_ref_reg(array_reg);
        self.touch_reg(index_reg);
        self.touch_ref_reg(dst);
        self.write_insn("getarrayitem_gc_r/rid>r");
        self.push_reg_u8(array_reg, "getarrayitem_gc_r array");
        self.push_reg_u8(index_reg, "getarrayitem_gc_r index");
        self.push_u16(descr_idx);
        self.push_reg_u8(dst, "getarrayitem_gc_r dst");
    }

    /// Load an f64 element (`blackhole.py` `bhimpl_getarrayitem_gc_f`).
    ///
    /// `jtransform.py` `rewrite_op_getarrayitem` picks `getarrayitem_gc_f`
    /// when the array's item type is `lltype.Float`. The descr is
    /// [`Self::add_raw_float_array_descr`].
    pub fn getarrayitem_gc_f(&mut self, dst: u16, array_reg: u16, index_reg: u16, descr_idx: u16) {
        self.touch_ref_reg(array_reg);
        self.touch_reg(index_reg);
        self.touch_float_reg(dst);
        self.write_insn("getarrayitem_gc_f/rid>f");
        self.push_reg_u8(array_reg, "getarrayitem_gc_f array");
        self.push_reg_u8(index_reg, "getarrayitem_gc_f index");
        self.push_u16(descr_idx);
        self.push_reg_u8(dst, "getarrayitem_gc_f dst");
    }

    /// Always-pure spelling of [`Self::getarrayitem_gc_f`].
    ///
    /// `jtransform.py` `rewrite_op_getarrayitem` appends `_pure` when
    /// `ARRAY._immutable_field(None)`, before choosing the kind, so an
    /// immutable float array is `getarrayitem_gc_f_pure`.
    pub fn getarrayitem_gc_f_pure(
        &mut self,
        dst: u16,
        array_reg: u16,
        index_reg: u16,
        descr_idx: u16,
    ) {
        self.touch_ref_reg(array_reg);
        self.touch_reg(index_reg);
        self.touch_float_reg(dst);
        self.write_insn("getarrayitem_gc_f_pure/rid>f");
        self.push_reg_u8(array_reg, "getarrayitem_gc_f_pure array");
        self.push_reg_u8(index_reg, "getarrayitem_gc_f_pure index");
        self.push_u16(descr_idx);
        self.push_reg_u8(dst, "getarrayitem_gc_f_pure dst");
    }

    /// Always-pure spelling of [`Self::getarrayitem_gc_r`].
    ///
    /// `jtransform.py` `rewrite_op_getarrayitem` appends `_pure` when
    /// `ARRAY._immutable_field(None)` (`_immutable_fields_ = ['x[*]']`).
    /// `blackhole.py` aliases the pure handler onto the plain read; the
    /// recorded opcode is `GetarrayitemGcPureR`, which
    /// `OpHelpers.is_always_pure` admits.
    pub fn getarrayitem_gc_r_pure(
        &mut self,
        dst: u16,
        array_reg: u16,
        index_reg: u16,
        descr_idx: u16,
    ) {
        self.touch_ref_reg(array_reg);
        self.touch_reg(index_reg);
        self.touch_ref_reg(dst);
        self.write_insn("getarrayitem_gc_r_pure/rid>r");
        self.push_reg_u8(array_reg, "getarrayitem_gc_r_pure array");
        self.push_reg_u8(index_reg, "getarrayitem_gc_r_pure index");
        self.push_u16(descr_idx);
        self.push_reg_u8(dst, "getarrayitem_gc_r_pure dst");
    }

    /// Add the array descriptor for a byte-element array to the descrs pool.
    ///
    /// Returns the descr index to pass as `descr_idx` to `getarrayitem_gc_i`.
    /// The descriptor models the `program: &[u8]` bytecode slice used by the
    /// dispatch JitCode body.
    ///
    /// Deduped: multiple calls with the same shape return the same index.
    ///
    /// `itemsize=1` mirrors RPython `pypy/interpreter/pyopcode.py`
    /// `ord(co_code[next_instr])` byte-load semantics. `is_item_signed=false`
    /// because `ord()` yields a non-negative `0..=255` integer (zero-extend
    /// to `i64`, never sign-extend); the `u8` SUB-INTERVAL property must
    /// not be widened to a signed `i64` at the descr boundary or the
    /// backend would emit `movsx` on byte loads ≥ `0x80` and corrupt
    /// opcode dispatch.
    pub fn add_gc_byte_array_descr(&mut self) -> u16 {
        self.add_env_int_array_descr(1, false)
    }

    /// Add the array descriptor for an integer-element `env` array whose items
    /// are `item_size` bytes wide (`is_item_signed` selects sign- vs
    /// zero-extension on sub-word loads).  Generalizes
    /// [`Self::add_gc_byte_array_descr`] so a wider env element type (`&[i64]`,
    /// `item_size = 8`) reads the element at byte offset `item_size * index`
    /// instead of the raw byte at `index`.
    ///
    /// The array is NOT GC-managed and carries no length word: a Rust slice
    /// (`&[T]`) data pointer (`codegen_trace.rs`'s `generate_trace_fn`, emitting
    /// `*const #env_type as *const ()`) points directly at `items[0]`, and the
    /// length travels in the
    /// fat pointer's other half, which never reaches the trace.  This is the
    /// `arraydescrof(rffi.CArray(T))` shape, so it is
    /// [`Self::add_raw_int_array_descr_signed`] — see the note there for what
    /// the GC-managed flag would cost.
    pub fn add_env_int_array_descr(&mut self, item_size: usize, is_item_signed: bool) -> u16 {
        self.add_raw_int_array_descr_signed(item_size, is_item_signed)
    }

    /// Add a descriptor for a `raw_store_i` / `raw_load_i` against raw
    /// (non-GC-managed) native memory whose items are `item_size` bytes
    /// wide.  Mirrors `jtransform.py rewrite_op_raw_store`'s
    /// `self.cpu.arraydescrof(rffi.CArray(T))`: a flat C array with no
    /// length header and no GC type tag.  `base_size = 0` and
    /// `len_offset = None` because a raw address points directly at the
    /// target byte (the effective address is passed as a byte offset, not
    /// a scaled index); `is_gc_managed = false` so `make_guards` never
    /// emits `GUARD_GC_TYPE` against a raw pointer that has no GC header.
    /// Deduped structurally by `add_bh_descr`.
    pub fn add_raw_int_array_descr(&mut self, item_size: usize) -> u16 {
        self.add_raw_int_array_descr_signed(item_size, false)
    }

    /// Add a descriptor for a raw native integer array, selecting sign- vs
    /// zero-extension for sub-word `raw_load_i` results.
    ///
    /// Also the descr for a header-less buffer reached through a bare `*mut T`
    /// field, even though those elements are read with `getarrayitem_gc_i`
    /// rather than `raw_load_i`: what the flag governs is the object header,
    /// and such a buffer has none.  `ArrayPtrInfo::make_guards`
    /// (`optimizeopt/info.rs`'s `make_guards`, mirroring `info.py`) emits
    /// `GUARD_GC_TYPE` for a GC-managed array descr, reading a type-id word at
    /// `ptr - GcHeader::SIZE`; ahead of such a buffer that word belongs to
    /// whatever precedes the allocation, so once the base is red the guard
    /// fails or faults on short-preamble re-entry.  The `&[u8]` `env` array
    /// is the same descr: a Rust slice data pointer has no header either, and
    /// the optimizer therefore gives it `PtrInfo::Constant`, never this arm.
    pub fn add_raw_int_array_descr_signed(
        &mut self,
        item_size: usize,
        is_item_signed: bool,
    ) -> u16 {
        self.add_bh_descr(CanonicalBhDescr::Array {
            base_size: 0,
            itemsize: item_size,
            len_offset: None,
            type_id: 0,
            gc_type_id: 0,
            item_type: majit_ir::value::Type::Int,
            is_array_of_pointers: false,
            is_array_of_structs: false,
            is_item_signed,
            ei_index: u32::MAX,
            array_type_id: None,
            interior_fields: Vec::new(),
            is_gc_managed: false,
        })
    }

    /// Add a descriptor for a raw native f64 array. The runtime raw-load-f
    /// path carries the loaded f64 through `registers_f` as raw bits.
    pub fn add_raw_float_array_descr(&mut self) -> u16 {
        self.add_bh_descr(CanonicalBhDescr::Array {
            base_size: 0,
            // f64 items are 8 bytes on every target — the one scalar width
            // that does not follow the word.
            itemsize: std::mem::size_of::<f64>(),
            len_offset: None,
            type_id: 0,
            gc_type_id: 0,
            item_type: majit_ir::value::Type::Float,
            is_array_of_pointers: false,
            is_array_of_structs: false,
            is_item_signed: false,
            ei_index: u32::MAX,
            array_type_id: None,
            interior_fields: Vec::new(),
            is_gc_managed: false,
        })
    }

    /// Emit `raw_load_i/iid>i` (`blackhole.py bhimpl_raw_load_i`):
    /// read an int from raw memory at `registers_i[base_reg] +
    /// registers_i[ea_reg]` (byte offset) into `dst`.  `jtransform.py rewrite_op_raw_load
    /// rewrite_op_raw_load` lowers a `raw_storage_getitem` to this op with an
    /// `arraydescrof(rffi.CArray(T))` descr (`add_raw_int_array_descr`).
    ///
    /// Encoding: `[BC_RAW_LOAD_I][base_reg u8][ea_reg u8][descr_idx lo u8]
    ///             [descr_idx hi u8][dst u8]`.
    pub fn raw_load_i(&mut self, dst: u16, base_reg: u16, ea_reg: u16, descr_idx: u16) {
        self.touch_reg(base_reg);
        self.touch_reg(ea_reg);
        self.touch_reg(dst);
        self.write_insn("raw_load_i/iid>i");
        self.push_reg_u8(base_reg, "raw_load_i base");
        self.push_reg_u8(ea_reg, "raw_load_i ea");
        self.push_u16(descr_idx);
        self.push_reg_u8(dst, "raw_load_i dst");
    }

    /// Emit `raw_load_f/iid>f`: read an f64 from raw memory at
    /// `registers_i[base_reg] + registers_i[ea_reg]` into float register
    /// `dst`, preserving the exact bit pattern in the float bank.
    pub fn raw_load_f(&mut self, dst: u16, base_reg: u16, ea_reg: u16, descr_idx: u16) {
        self.touch_reg(base_reg);
        self.touch_reg(ea_reg);
        self.touch_float_reg(dst);
        self.write_insn("raw_load_f/iid>f");
        self.push_reg_u8(base_reg, "raw_load_f base");
        self.push_reg_u8(ea_reg, "raw_load_f ea");
        self.push_u16(descr_idx);
        self.push_reg_u8(dst, "raw_load_f dst");
    }

    /// Emit `raw_store_i/iiid` (`blackhole.py bhimpl_raw_store_i`):
    /// store the int in `value_reg` into raw memory at
    /// `registers_i[base_reg] + registers_i[ea_reg]` (byte offset).
    /// `jtransform.py rewrite_op_raw_store` lowers a Python-level
    /// `raw_storage_setitem` to this op with an `arraydescrof(rffi.CArray(T))`
    /// descr (`add_raw_int_array_descr`, `item_size` bytes).
    ///
    /// Encoding: `[BC_RAW_STORE_I][base_reg u8][ea_reg u8][value_reg u8]
    ///             [descr_idx lo u8][descr_idx hi u8]`.
    pub fn raw_store_i(&mut self, base_reg: u16, ea_reg: u16, value_reg: u16, descr_idx: u16) {
        self.touch_reg(base_reg);
        self.touch_reg(ea_reg);
        self.touch_reg(value_reg);
        self.write_insn("raw_store_i/iiid");
        self.push_reg_u8(base_reg, "raw_store_i base");
        self.push_reg_u8(ea_reg, "raw_store_i ea");
        self.push_reg_u8(value_reg, "raw_store_i value");
        self.push_u16(descr_idx);
    }

    /// Add a GC-array descriptor for a raw-pointer-element array (one-word
    /// `*mut T` items) to the descrs pool; returns the descr index for
    /// `getarrayitem_gc_r`.  Models `[*mut T; N]` living inside a struct the
    /// caller owns: `base_size` is the byte offset of the first item from the
    /// pointer the read indexes off, and `len_offset` names a length word
    /// ahead of it when the caller keeps one.
    ///
    /// Both are the caller's to state because both are its layout.  RPython
    /// derives them from the lltype (`descr.py get_array_descr(gccache,
    /// ARRAY)`); a hand-written consumer struct has no lltype, so the values
    /// arrive from the declaration or they are guessed.
    ///
    /// `len_offset = None` is a supported shape, not a degraded one: a plain
    /// fixed-size inline array has no length in memory, and
    /// `ArrayPtrInfo.make_guards` skips the short preamble's `ARRAYLEN_GC` for
    /// a descr without a lendescr (`optimizeopt/info.rs`) rather than emitting
    /// an op the mandatory GC rewrite would then unwrap.  With a length word
    /// declared, the read keeps the `len > index` bound the optimizer narrows
    /// on a constant index (`heap.py:676 getlenbound().make_gt_const(index)`).
    ///
    /// Deduped structurally by `add_bh_descr`.
    pub fn add_ptr_array_descr(&mut self, base_size: usize, len_offset: Option<usize>) -> u16 {
        self.add_array_descr(CanonicalBhDescr::Array {
            base_size,
            itemsize: scalar_size(majit_ir::value::Type::Ref),
            len_offset,
            type_id: 0,
            gc_type_id: 0,
            item_type: majit_ir::value::Type::Ref,
            is_array_of_pointers: true,
            is_array_of_structs: false,
            is_item_signed: false,
            // Header-less raw native pointer-array (the `pool_arrays`
            // base): no GC header, so `make_guards` must NOT emit
            // GUARD_GC_TYPE against `ptr - GcHeader::SIZE`.
            is_gc_managed: false,
            ei_index: u32::MAX,
            array_type_id: None,
            interior_fields: Vec::new(),
        })
    }

    /// Allocate a zero-initialised GC array.
    ///
    /// blackhole.py `bhimpl_new_array_clear @arguments("cpu","i","d",
    /// returns="r")`: reads `registers_i[length_reg]` (or a const-pool
    /// slot) as the element count and `descrs[descr_idx]` as the array
    /// descriptor; writes the new array into `registers_r[dst]`.
    ///
    /// Encoding: `[BC_NEW_ARRAY_CLEAR][length_reg u8][descr_idx lo u8]
    ///             [descr_idx hi u8][dst u8]`.
    pub fn new_array_clear(&mut self, dst: u16, length_reg: u16, descr_idx: u16) {
        self.touch_int_reg_or_pool_slot(length_reg);
        self.touch_ref_reg(dst);
        self.write_insn("new_array_clear/id>r");
        self.push_reg_u8(length_reg, "new_array_clear length");
        self.push_u16(descr_idx);
        self.push_reg_u8(dst, "new_array_clear dst");
    }

    /// `c`-argcode form of [`Self::new_array_clear`] —
    /// `assembler.py emit_const(allow_short=True)` writes a
    /// small ConstInt length (-128..127) inline as one signed byte
    /// (`new_array_clear` is in `USE_C_FORM`, `assembler.py`).
    ///
    /// Encoding: `[BC_NEW_ARRAY_CLEAR_C][length i8][descr_idx lo u8]
    ///             [descr_idx hi u8][dst u8]`.
    pub fn new_array_clear_c(&mut self, dst: u16, length: i8, descr_idx: u16) {
        self.touch_ref_reg(dst);
        self.write_insn("new_array_clear/cd>r");
        self.push_u8(length as u8);
        self.push_u16(descr_idx);
        self.push_reg_u8(dst, "new_array_clear dst");
    }

    /// `new_array/id>r` (`blackhole.py` `bhimpl_new_array`).
    ///
    /// `jtransform.py` `rewrite_op_malloc_varsize` emits this when the
    /// element is not a pointer or struct and the malloc is not `zero`.
    /// Encoding matches [`Self::new_array_clear`]: length register, array
    /// descr, destination ref.
    pub fn new_array(&mut self, dst: u16, length_reg: u16, descr_idx: u16) {
        self.touch_int_reg_or_pool_slot(length_reg);
        self.touch_ref_reg(dst);
        self.write_insn("new_array/id>r");
        self.push_reg_u8(length_reg, "new_array length");
        self.push_u16(descr_idx);
        self.push_reg_u8(dst, "new_array dst");
    }

    /// `arraylen_gc/rd>i` (`blackhole.py` `bhimpl_arraylen_gc`).
    ///
    /// The length word is the descr's `lendescr` (`descr.py`
    /// `get_field_arraylen_descr`), not a separate struct field.
    pub fn arraylen_gc(&mut self, dst: u16, array_reg: u16, descr_idx: u16) {
        self.touch_ref_reg(array_reg);
        self.touch_reg(dst);
        self.write_insn("arraylen_gc/rd>i");
        self.push_reg_u8(array_reg, "arraylen_gc array");
        self.push_u16(descr_idx);
        self.push_reg_u8(dst, "arraylen_gc dst");
    }

    /// Array descr for `malloc_varsize` of a `GcArray`, including one
    /// inlined after a header (`descr.py` `get_array_descr`).
    ///
    /// `base_size` is the offset of element 0 (`offset_of!(Header, items)`
    /// for the `Elem in Header` shape). `len_offset` is the length word
    /// (`get_field_arraylen_descr`). `is_gc_managed` stays false: the
    /// pointer addresses the payload, and a `GUARD_GC_TYPE` would read a
    /// header the concrete block does not carry in front of that pointer.
    /// `type_id` is the lltype cache key (`path_hash`). `gc_type_id` is
    /// `init_array_descr` `descr.tid` — the collector's type-registry slot
    /// (`layoutbuilder.get_type_id(ARRAY)`). `rewrite.py`
    /// `gen_malloc_nursery_varsize` stamps that tid into the header.
    #[allow(clippy::too_many_arguments)]
    pub fn add_gc_varsize_array_descr(
        &mut self,
        base_size: usize,
        len_offset: usize,
        itemsize: usize,
        is_array_of_pointers: bool,
        is_item_signed: bool,
        type_id: u64,
        gc_type_id: u32,
    ) -> u16 {
        let item_type = if is_array_of_pointers {
            majit_ir::value::Type::Ref
        } else {
            majit_ir::value::Type::Int
        };
        self.add_array_descr(CanonicalBhDescr::Array {
            base_size,
            itemsize,
            len_offset: Some(len_offset),
            type_id,
            gc_type_id,
            item_type,
            is_array_of_pointers,
            is_array_of_structs: false,
            is_item_signed,
            is_gc_managed: false,
            ei_index: u32::MAX,
            array_type_id: None,
            interior_fields: Vec::new(),
        })
    }

    /// Float-item form of [`Self::add_gc_varsize_array_descr`].
    ///
    /// `jtransform.py` `rewrite_op_getarrayitem` / `getkind(ARRAY.OF)` selects
    /// the float opcode for an `f64` element in a header. `itemsize` is
    /// `sizeof(f64)` (`symbolic.py` `get_array_token`).
    pub fn add_gc_varsize_float_array_descr(
        &mut self,
        base_size: usize,
        len_offset: usize,
        type_id: u64,
        gc_type_id: u32,
    ) -> u16 {
        self.add_array_descr(CanonicalBhDescr::Array {
            base_size,
            itemsize: std::mem::size_of::<f64>(),
            len_offset: Some(len_offset),
            type_id,
            gc_type_id,
            item_type: majit_ir::value::Type::Float,
            is_array_of_pointers: false,
            is_array_of_structs: false,
            is_item_signed: false,
            is_gc_managed: false,
            ei_index: u32::MAX,
            array_type_id: None,
            interior_fields: Vec::new(),
        })
    }

    /// Emit `newlist_clear/idddd>r` (`handler_newlist_clear`,
    /// `blackhole.py:1173-1180`): the compound resizable-list allocation
    /// `do_resizable_newlist_clear` emits.  ONE opcode carrying the
    /// element count (int register) plus FOUR descriptors in the exact
    /// read order the handler and trace dispatch agree on —
    /// structdescr (`list` header Size), lengthdescr (`list.length`
    /// FieldDescr), itemsdescr (`list.items` FieldDescr), arraydescr
    /// (the length-prefixed items block).  The list header layout is
    /// `{ length: Signed @ length_offset, items: Ptr(GcArray) @
    /// items_offset }`; the items block is `arraydescrof(item_type,
    /// len_offset = Some(0))`.  Registers the header layout under
    /// `struct_type_id` so the two setfield FieldDescrs carry a parent
    /// SizeDescr (`descr.py`).
    #[allow(clippy::too_many_arguments)]
    pub fn newlist_clear(
        &mut self,
        dest: u16,
        length_reg: u16,
        struct_type_id: u64,
        struct_size: usize,
        length_offset: usize,
        items_offset: usize,
        array_type_id: u64,
        item_type: majit_ir::value::Type,
    ) {
        self.touch_int_reg_or_pool_slot(length_reg);
        self.touch_ref_reg(dest);
        // Register the `list` header layout so the length/items field
        // descrs resolve a parent SizeDescr + index_in_parent.
        self.register_struct_layout(
            struct_size,
            struct_type_id,
            true,
            false,
            &[
                (
                    length_offset,
                    false,
                    "length",
                    scalar_size(majit_ir::value::Type::Int),
                    true,
                    false,
                ),
                (
                    items_offset,
                    true,
                    "items",
                    scalar_size(majit_ir::value::Type::Ref),
                    false,
                    false,
                ),
            ],
            // The JIT synthesizes this header; there is no declaring struct to
            // read a `_immutable_fields_` list off.
            "",
        );
        let all_fielddescrs = self
            .struct_size_specs
            .get(&struct_type_id)
            .expect("register_struct_layout just inserted this type_id")
            .all_fielddescrs
            .clone();
        // structdescr: the `list` header Size descr (resizable list, no vtable).
        let struct_descr = self.add_bh_descr(CanonicalBhDescr::Size {
            size: struct_size,
            type_id: struct_type_id,
            vtable: 0,
            owner: String::new(),
            all_fielddescrs,
            is_gc_managed: true,
        });
        // lengthdescr / itemsdescr: parent-carrying field descrs.
        let length_descr = self.add_struct_field_descr(
            length_offset,
            majit_ir::value::Type::Int,
            struct_type_id,
            "length",
        );
        let items_descr = self.add_struct_field_descr(
            items_offset,
            majit_ir::value::Type::Ref,
            struct_type_id,
            "items",
        );
        // arraydescr: length-prefixed items block (length word at offset 0).
        let is_item_signed = !matches!(
            item_type,
            majit_ir::value::Type::Ref | majit_ir::value::Type::Float
        );
        let array_descr = self.add_array_descr(CanonicalBhDescr::Array {
            base_size: std::mem::size_of::<usize>(),
            itemsize: scalar_size(item_type),
            len_offset: Some(0),
            type_id: array_type_id,
            gc_type_id: 0,
            item_type,
            is_array_of_pointers: matches!(item_type, majit_ir::value::Type::Ref),
            is_array_of_structs: false,
            is_item_signed,
            ei_index: u32::MAX,
            array_type_id: None,
            interior_fields: Vec::new(),
            is_gc_managed: true,
        });
        self.write_insn("newlist_clear/idddd>r");
        self.push_reg_u8(length_reg, "newlist_clear length");
        self.push_u16(struct_descr);
        self.push_u16(length_descr);
        self.push_u16(items_descr);
        self.push_u16(array_descr);
        self.push_reg_u8(dest, "newlist_clear dst");
    }

    /// Store an integer element into a GC-managed array.
    ///
    /// blackhole.py `bhimpl_setarrayitem_gc_i @arguments("cpu","r","i",
    /// "i","d")`: reads `registers_r[array_reg]` as the array pointer,
    /// `registers_i[index_reg]` (or a const-pool slot) as the element index,
    /// `registers_i[value_reg]` (or a const-pool slot) as the new value, and
    /// `descrs[descr_idx]` as the array descriptor.
    ///
    /// Encoding: `[BC_SETARRAYITEM_GC_I][array_reg u8][index_reg u8]
    ///             [value_reg u8][descr_idx lo u8][descr_idx hi u8]`.
    ///
    /// Store counterpart of [`Self::getarrayitem_gc_i`]. Both take the
    /// element descr from [`Self::add_env_int_array_descr`], so a store
    /// invalidates the heapcache entry the matching load populated.
    pub fn setarrayitem_gc_i(
        &mut self,
        array_reg: u16,
        index_reg: u16,
        value_reg: u16,
        descr_idx: u16,
    ) {
        self.touch_ref_reg(array_reg);
        self.touch_int_reg_or_pool_slot(index_reg);
        self.touch_int_reg_or_pool_slot(value_reg);
        self.write_insn("setarrayitem_gc_i/riid");
        self.push_reg_u8(array_reg, "setarrayitem_gc_i array");
        self.push_reg_u8(index_reg, "setarrayitem_gc_i index");
        self.push_reg_u8(value_reg, "setarrayitem_gc_i value");
        self.push_u16(descr_idx);
    }

    /// `c`-argcode form of [`Self::setarrayitem_gc_i`] —
    /// `assembler.py emit_const(allow_short=True)` writes a small
    /// ConstInt VALUE (-128..127) inline as one signed byte
    /// (`setarrayitem_gc_i` is in `USE_C_FORM`, `assembler.py`).
    ///
    /// Note the constant operand differs from
    /// [`Self::setarrayitem_gc_r_c`], where it is the INDEX: here the index
    /// stays a register and the stored value is inline, matching the
    /// `/ricd` argcode string the blackhole registers.
    ///
    /// Encoding: `[BC_SETARRAYITEM_GC_I_C][array_reg u8][index_reg u8]
    ///             [value i8][descr_idx lo u8][descr_idx hi u8]`.
    pub fn setarrayitem_gc_i_c(
        &mut self,
        array_reg: u16,
        index_reg: u16,
        value: i8,
        descr_idx: u16,
    ) {
        self.touch_ref_reg(array_reg);
        self.touch_int_reg_or_pool_slot(index_reg);
        self.write_insn("setarrayitem_gc_i/ricd");
        self.push_reg_u8(array_reg, "setarrayitem_gc_i array");
        self.push_reg_u8(index_reg, "setarrayitem_gc_i index");
        self.push_u8(value as u8);
        self.push_u16(descr_idx);
    }

    /// Store a Ref element into a GC-managed array.
    ///
    /// blackhole.py `bhimpl_setarrayitem_gc_r @arguments("cpu","r","i",
    /// "r","d")`: reads `registers_r[array_reg]` as the array pointer,
    /// `registers_i[index_reg]` (or a const-pool slot) as the element
    /// index, `registers_r[value_reg]` (or a const-pool slot) as the new
    /// value, and `descrs[descr_idx]` as the array descriptor.
    ///
    /// Encoding: `[BC_SETARRAYITEM_GC_R][array_reg u8][index_reg u8]
    ///             [value_reg u8][descr_idx lo u8][descr_idx hi u8]`.
    pub fn setarrayitem_gc_r(
        &mut self,
        array_reg: u16,
        index_reg: u16,
        value_reg: u16,
        descr_idx: u16,
    ) {
        self.touch_ref_reg(array_reg);
        self.touch_int_reg_or_pool_slot(index_reg);
        self.touch_ref_reg_or_pool_slot(value_reg);
        self.write_insn("setarrayitem_gc_r/rird");
        self.push_reg_u8(array_reg, "setarrayitem_gc_r array");
        self.push_reg_u8(index_reg, "setarrayitem_gc_r index");
        self.push_reg_u8(value_reg, "setarrayitem_gc_r value");
        self.push_u16(descr_idx);
    }

    /// Store an f64 element (`blackhole.py` `bhimpl_setarrayitem_gc_f`).
    ///
    /// Store counterpart of [`Self::getarrayitem_gc_f`]. `jtransform.py`
    /// `rewrite_op_setarrayitem` emits this when the item type is
    /// `lltype.Float`.
    pub fn setarrayitem_gc_f(
        &mut self,
        array_reg: u16,
        index_reg: u16,
        value_reg: u16,
        descr_idx: u16,
    ) {
        self.touch_ref_reg(array_reg);
        self.touch_int_reg_or_pool_slot(index_reg);
        self.touch_float_reg(value_reg);
        self.write_insn("setarrayitem_gc_f/rifd");
        self.push_reg_u8(array_reg, "setarrayitem_gc_f array");
        self.push_reg_u8(index_reg, "setarrayitem_gc_f index");
        self.push_reg_u8(value_reg, "setarrayitem_gc_f value");
        self.push_u16(descr_idx);
    }

    /// `c`-argcode form of [`Self::setarrayitem_gc_r`] —
    /// `assembler.py emit_const(allow_short=True)` writes a
    /// small ConstInt index (-128..127) inline as one signed byte
    /// (`setarrayitem_gc_r` is in `USE_C_FORM`, `assembler.py`).
    ///
    /// Encoding: `[BC_SETARRAYITEM_GC_R_C][array_reg u8][index i8]
    ///             [value_reg u8][descr_idx lo u8][descr_idx hi u8]`.
    pub fn setarrayitem_gc_r_c(
        &mut self,
        array_reg: u16,
        index: i8,
        value_reg: u16,
        descr_idx: u16,
    ) {
        self.touch_ref_reg(array_reg);
        self.touch_ref_reg_or_pool_slot(value_reg);
        self.write_insn("setarrayitem_gc_r/rcrd");
        self.push_reg_u8(array_reg, "setarrayitem_gc_r array");
        self.push_u8(index as u8);
        self.push_reg_u8(value_reg, "setarrayitem_gc_r value");
        self.push_u16(descr_idx);
    }

    /// RPython `blackhole.py bhimpl_int_add` `bhimpl_int_*` per-opname handlers:
    /// each primitive has its own insn_id in `BlackholeInterpBuilder.insns`
    /// (`blackhole.py setup_insns`). Emits via `write_insn` with
    /// the canonical `opname/ii>i` key so the opcode byte comes from the
    /// shared insns table rather than a hand-assigned `BC_*` constant.
    ///
    /// `OpCode::IntFloorDiv` / `OpCode::IntMod` are intentionally absent:
    /// `jtransform.py:575-577` rewrites both to
    /// `direct_call(ll_int_py_*)` before jitcode emission, so neither
    /// `bhimpl_*` nor the corresponding `int_(floordiv|mod)/ii>i` insns
    /// key exists upstream.  Pyre's β' redirect at
    /// `pyre-jit-trace/src/jitcode_dispatch/specialize.rs::walker_emit_int_py_div_or_mod`
    /// covers the runtime trace path.
    pub fn record_binop_i(&mut self, dst: u16, opcode: OpCode, lhs: u16, rhs: u16) {
        let key = Self::binop_i_key(opcode);
        self.touch_reg(dst);
        self.touch_reg(lhs);
        self.touch_reg(rhs);
        self.write_insn(key);
        // `assembler.py write_insn` argcode `ii>i` byte order:
        // `[lhs][rhs][dst]`, matching the canonical `bhhandler_ii_i!`
        // decoder.
        self.push_u8(lhs as u8);
        self.push_u8(rhs as u8);
        self.push_u8(dst as u8);
    }

    /// [`Self::record_binop_i`] whose right operand is a `Constant`.
    /// `assembler.py` `emit_const` encodes it as the register-space index
    /// `num_regs_i + const_idx` of the constants suffix, so the operation
    /// keeps its `ii>i` key and no `int_copy` loads the value first.
    pub fn record_binop_i_const(&mut self, dst: u16, opcode: OpCode, lhs: u16, rhs: i64) {
        let key = Self::binop_i_key(opcode);
        self.touch_reg(dst);
        self.touch_reg(lhs);
        self.write_insn(key);
        self.push_u8(lhs as u8);
        self.push_const_i_operand(rhs);
        self.push_u8(dst as u8);
    }

    /// `assembler.py` `emit_const` for an int `Constant` operand: a
    /// placeholder byte that `finish()` patches to `num_regs_i + const_idx`.
    fn push_const_i_operand(&mut self, value: i64) {
        let const_idx = self.add_const_i(value);
        let offset = self.code.len();
        self.push_u8(0);
        self.const_patches_u8
            .push((offset, ConstKind::Int, const_idx));
    }

    fn binop_i_key(opcode: OpCode) -> &'static str {
        match opcode {
            OpCode::IntAdd => "int_add/ii>i",
            OpCode::IntSub => "int_sub/ii>i",
            OpCode::IntMul => "int_mul/ii>i",
            OpCode::IntAnd => "int_and/ii>i",
            OpCode::IntSignext => "int_signext/ii>i",
            OpCode::IntOr => "int_or/ii>i",
            OpCode::IntXor => "int_xor/ii>i",
            OpCode::IntLshift => "int_lshift/ii>i",
            OpCode::IntRshift => "int_rshift/ii>i",
            OpCode::IntEq => "int_eq/ii>i",
            OpCode::IntNe => "int_ne/ii>i",
            OpCode::IntLt => "int_lt/ii>i",
            OpCode::IntLe => "int_le/ii>i",
            OpCode::IntGt => "int_gt/ii>i",
            OpCode::IntGe => "int_ge/ii>i",
            // Unsigned integer primitives — RPython `blackhole.py bhimpl_uint_mul_high`.
            OpCode::UintRshift => "uint_rshift/ii>i",
            OpCode::UintMulHigh => "uint_mul_high/ii>i",
            OpCode::UintLt => "uint_lt/ii>i",
            OpCode::UintLe => "uint_le/ii>i",
            OpCode::UintGt => "uint_gt/ii>i",
            OpCode::UintGe => "uint_ge/ii>i",
            other => panic!("record_binop_i: unsupported opcode {other:?}"),
        }
    }

    /// Emit a dynamically-numbered `USE_C_FORM` integer operation. PyPy's
    /// assembler allocates the `(opname, argcodes)` byte only when that shape
    /// appears; the runtime assembler supplies that byte here.
    pub fn record_binop_i_operands(
        &mut self,
        dst: u16,
        opcode_byte: u8,
        lhs: JitCodeIntOperand,
        rhs: JitCodeIntOperand,
    ) {
        self.touch_reg(dst);
        for operand in [lhs, rhs] {
            if let JitCodeIntOperand::Register(reg) = operand {
                self.touch_int_reg_or_pool_slot(reg);
            }
        }
        self.start_instr(opcode_byte);
        self.pending_resulttype = Some('i');
        for operand in [lhs, rhs] {
            match operand {
                JitCodeIntOperand::Register(reg) => self.push_reg_u8(reg, "integer operand"),
                JitCodeIntOperand::ShortConst(value) => self.push_u8(value as u8),
            }
        }
        self.push_reg_u8(dst, "integer result");
    }

    /// Fused integer comparison branch with independently short-constant
    /// operands (`goto_if_not_int_*/{ci,ic,cc}L`).
    pub fn goto_if_not_int_operands(
        &mut self,
        opcode_byte: u8,
        lhs: JitCodeIntOperand,
        rhs: JitCodeIntOperand,
        label: u16,
    ) {
        for operand in [lhs, rhs] {
            if let JitCodeIntOperand::Register(reg) = operand {
                self.touch_int_reg_or_pool_slot(reg);
            }
        }
        self.start_instr(opcode_byte);
        for operand in [lhs, rhs] {
            match operand {
                JitCodeIntOperand::Register(reg) => self.push_reg_u8(reg, "integer branch operand"),
                JitCodeIntOperand::ShortConst(value) => self.push_u8(value as u8),
            }
        }
        self.push_label_ref(label);
    }

    /// `blackhole.py:478-497` `bhimpl_int_{add,sub,mul}_jump_if_ovf(label,
    /// a, b)`: compute the wrapping result and, on signed overflow, branch to
    /// `label` instead of writing the result register.  The metatracer records
    /// these as `IntAddOvf`/`IntSubOvf`/`IntMulOvf` + `GuardNoOverflow`
    /// (`pyjitpl/dispatch.rs`); on guard failure the blackhole resumes at this
    /// op and performs the overflow branch itself.  Byte layout follows argcode
    /// `Lii>i` (`assembler.py:165-174`): `[target:u16][lhs][rhs][dst]`, matching
    /// the `bhhandler_ovf_jump_ii!` decoder.  A caller-emitted `-live-` must
    /// precede the op (like every guard/branch) so the guard snapshot decodes.
    pub fn int_add_jump_if_ovf(&mut self, dst: u16, lhs: u16, rhs: u16, label: u16) {
        self.record_int_binop_jump_if_ovf("int_add_jump_if_ovf/Lii>i", dst, lhs, rhs, label);
    }

    pub fn int_sub_jump_if_ovf(&mut self, dst: u16, lhs: u16, rhs: u16, label: u16) {
        self.record_int_binop_jump_if_ovf("int_sub_jump_if_ovf/Lii>i", dst, lhs, rhs, label);
    }

    pub fn int_mul_jump_if_ovf(&mut self, dst: u16, lhs: u16, rhs: u16, label: u16) {
        self.record_int_binop_jump_if_ovf("int_mul_jump_if_ovf/Lii>i", dst, lhs, rhs, label);
    }

    fn record_int_binop_jump_if_ovf(
        &mut self,
        key: &'static str,
        dst: u16,
        lhs: u16,
        rhs: u16,
        label: u16,
    ) {
        self.touch_reg(lhs);
        self.touch_reg(rhs);
        self.touch_reg(dst);
        self.write_insn(key);
        self.push_label_ref(label);
        self.push_u8(lhs as u8);
        self.push_u8(rhs as u8);
        self.push_u8(dst as u8);
    }

    /// `support.py` `_ll_2_int_floordiv`, inlined because `int_floordiv` is
    /// in `inline_calls_to`. The body is Python-floor `int.py_div` plus the
    /// branch-free truncation adjustment:
    /// `r + (((x ^ y) >> (LONG_BIT - 1)) & (p != x))` with `p = r * y`.
    /// A zero divisor still panics inside `ll_int_py_div`, the same
    /// precondition as `blackhole::_ll_2_int_floordiv`.
    pub fn record_int_floordiv(&mut self, dst: u16, lhs: u16, rhs: u16) {
        self.touch_reg(lhs);
        self.touch_reg(rhs);
        self.touch_reg(dst);
        // `support.py` `_ll_2_int_floordiv`: LONG_BIT - 1 on a 64-bit Signed.
        let kshift = 63i64;
        let Some(base) = self.alloc_int_temps(6) else {
            return;
        };
        let r = base;
        let p = base + 1;
        let xor = base + 2;
        let shifted = base + 3;
        let cmp = base + 4;
        let k = base + 5;
        self.load_const_i_value(k, kshift);
        self.record_int_py_div(r, lhs, rhs);
        self.record_binop_i(p, OpCode::IntMul, r, rhs);
        self.record_binop_i(xor, OpCode::IntXor, lhs, rhs);
        self.record_binop_i(shifted, OpCode::IntRshift, xor, k);
        self.record_binop_i(cmp, OpCode::IntNe, p, lhs);
        self.record_binop_i(shifted, OpCode::IntAnd, shifted, cmp);
        self.record_binop_i(dst, OpCode::IntAdd, r, shifted);
    }

    /// `support.py` `_ll_2_int_mod`, inlined because `int_mod` is in
    /// `inline_calls_to`:
    /// `r -= y & (((x ^ y) & (r | -r)) >> (LONG_BIT - 1))`
    /// with `r = int.py_mod(x, y)`. See [`Self::record_int_floordiv`].
    pub fn record_int_mod(&mut self, dst: u16, lhs: u16, rhs: u16) {
        self.touch_reg(lhs);
        self.touch_reg(rhs);
        self.touch_reg(dst);
        let kshift = 63i64;
        let Some(base) = self.alloc_int_temps(5) else {
            return;
        };
        let r = base;
        let folded = base + 1;
        let xor = base + 2;
        let shifted = base + 3;
        let k = base + 4;
        self.load_const_i_value(k, kshift);
        self.record_int_py_mod(r, lhs, rhs);
        self.record_unary_i(folded, OpCode::IntNeg, r);
        self.record_binop_i(folded, OpCode::IntOr, r, folded);
        self.record_binop_i(xor, OpCode::IntXor, lhs, rhs);
        self.record_binop_i(shifted, OpCode::IntAnd, xor, folded);
        self.record_binop_i(shifted, OpCode::IntRshift, shifted, k);
        self.record_binop_i(shifted, OpCode::IntAnd, rhs, shifted);
        self.record_binop_i(dst, OpCode::IntSub, r, shifted);
    }

    /// Fresh int registers above every register already touched, including
    /// the operands of the instruction about to expand. Dead once that
    /// instruction's result is written; later ops may reuse the numbers.
    fn alloc_int_temps(&mut self, n: u16) -> Option<u16> {
        let base = self.num_regs_i;
        // Bytecode register indices are one byte (`assembler.py` `chr`).
        // A temp past that cannot be emitted; latch and decline the JitCode.
        if u32::from(base) + u32::from(n) > 256 {
            self.encoding_overflow = true;
            return None;
        }
        for i in 0..n {
            self.touch_reg(base.saturating_add(i));
        }
        Some(base)
    }

    /// `jtransform.py` `_handle_int_special` `int.py_div` → residual
    /// `ll_int_py_div` (`EF_ELIDABLE_CANNOT_RAISE`). Python-floor `//`,
    /// not Rust `/`.
    pub fn record_int_py_div(&mut self, dst: u16, lhs: u16, rhs: u16) {
        self.record_int_py_helper(
            dst,
            lhs,
            rhs,
            crate::blackhole::ll_int_py_div as *const (),
            crate::call_descr::INT_PY_DIV_EFFECT_INFO,
        );
    }

    pub fn record_int_py_mod(&mut self, dst: u16, lhs: u16, rhs: u16) {
        self.record_int_py_helper(
            dst,
            lhs,
            rhs,
            crate::blackhole::ll_int_py_mod as *const (),
            crate::call_descr::INT_PY_MOD_EFFECT_INFO,
        );
    }

    /// Unsigned `/` — the `int.udiv` oopspec residual call
    /// (`rint.py ll_uint_py_div`).  RPython has no unsigned division
    /// resop at all: `UINT_FLOORDIV` was deleted in 2016 in favour of this
    /// oopspec call, so unlike the unsigned comparisons (which do have
    /// `uint_lt`/`uint_le` primitives reachable through `record_binop_i`)
    /// there is nothing here to record but the call.  Same
    /// `rhs != 0` precondition as [`Self::record_int_py_div`]; unsigned has no
    /// `INT_MIN / -1` corner.
    pub fn record_uint_py_div(&mut self, dst: u16, lhs: u16, rhs: u16) {
        self.record_int_py_helper(
            dst,
            lhs,
            rhs,
            crate::blackhole::ll_uint_py_div as *const (),
            crate::call_descr::UINT_PY_DIV_EFFECT_INFO,
        );
    }

    /// Unsigned `%` — the `int.umod` oopspec residual call
    /// (`rint.py ll_uint_py_mod`).  See [`Self::record_uint_py_div`].
    pub fn record_uint_py_mod(&mut self, dst: u16, lhs: u16, rhs: u16) {
        self.record_int_py_helper(
            dst,
            lhs,
            rhs,
            crate::blackhole::ll_uint_py_mod as *const (),
            crate::call_descr::UINT_PY_MOD_EFFECT_INFO,
        );
    }

    fn record_int_py_helper(
        &mut self,
        dst: u16,
        lhs: u16,
        rhs: u16,
        funcptr: *const (),
        effect_info: majit_ir::descr::EffectInfo,
    ) {
        let fn_ptr_idx = self.add_call_target(funcptr, funcptr);
        self.residual_call_int_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            &[JitCallArg::int(lhs), JitCallArg::int(rhs)],
            dst,
            effect_info,
        );
    }

    /// `blackhole.py` `bhimpl_int_{neg,invert,is_zero,is_true,force_ge_zero}`
    /// per-opname handlers. See `record_binop_i` for the keying rationale.
    /// Byte layout follows `assembler.py`: each `Register` is
    /// emitted in argcode order, so `int_neg/i>i` stores `[src][dst]`
    /// matching the canonical `bhhandler_i_i!` decoder.
    pub fn record_unary_i(&mut self, dst: u16, opcode: OpCode, src: u16) {
        let key = match opcode {
            OpCode::IntNeg => "int_neg/i>i",
            OpCode::IntInvert => "int_invert/i>i",
            OpCode::IntIsZero => "int_is_zero/i>i",
            OpCode::IntIsTrue => "int_is_true/i>i",
            OpCode::IntForceGeZero => "int_force_ge_zero/i>i",
            other => panic!("record_unary_i: unsupported opcode {other:?}"),
        };
        self.touch_reg(dst);
        self.touch_reg(src);
        self.write_insn(key);
        self.push_u8(src as u8);
        self.push_u8(dst as u8);
    }

    /// RPython `blackhole.py:584-610` ref comparisons returning int:
    /// `ptr_eq`, `ptr_ne`, `instance_ptr_eq`, `instance_ptr_ne`.
    /// Byte layout follows `assembler.py`: `[lhs][rhs][dst]`
    /// matching the canonical `bhhandler_rr_i!` decoder.
    ///
    /// Either comparand may be a constant — `x is None` compares against the
    /// `None` singleton — and `assembler.py emit_const` gives such an
    /// argument a constant-pool slot in place of a register index, so both
    /// operands take the pool-aware toucher/pusher rather than the
    /// real-register-only pair.
    pub fn record_binop_r(&mut self, dst: u16, opcode: OpCode, lhs: u16, rhs: u16) {
        let key = match opcode {
            OpCode::PtrEq => "ptr_eq/rr>i",
            OpCode::PtrNe => "ptr_ne/rr>i",
            OpCode::InstancePtrEq => "instance_ptr_eq/rr>i",
            OpCode::InstancePtrNe => "instance_ptr_ne/rr>i",
            other => panic!("record_binop_r: unsupported opcode {other:?}"),
        };
        self.touch_reg(dst);
        self.touch_ref_reg_or_pool_slot(lhs);
        self.touch_ref_reg_or_pool_slot(rhs);
        self.write_insn(key);
        self.push_reg_u8(lhs, "ptr binop lhs");
        self.push_reg_u8(rhs, "ptr binop rhs");
        self.push_u8(dst as u8);
    }

    /// RPython `blackhole.py bhimpl_ptr_iszero` unary ptr nullity checks returning int.
    /// Byte layout `[src][dst]` per canonical `bhhandler_r_i!` decoder.
    pub fn ptr_iszero(&mut self, dst: u16, src: u16) {
        self.touch_reg(dst);
        self.touch_ref_reg(src);
        self.write_insn("ptr_iszero/r>i");
        self.push_u8(src as u8);
        self.push_u8(dst as u8);
    }

    pub fn ptr_nonzero(&mut self, dst: u16, src: u16) {
        self.touch_reg(dst);
        self.touch_ref_reg(src);
        self.write_insn("ptr_nonzero/r>i");
        self.push_u8(src as u8);
        self.push_u8(dst as u8);
    }

    pub fn new_label(&mut self) -> u16 {
        // A label id addresses `self.labels` and is patched into a two-byte
        // jump-target slot (`assembler.py fix_labels`), so the id
        // space ends at `u16::MAX`.  Past it the `as u16` truncation would
        // alias a fresh label onto an earlier one and silently patch the
        // wrong target; record the overflow so `try_finish` declines the
        // whole jitcode instead.
        let label = self.labels.len();
        self.encoding_overflow |= label > u16::MAX as usize;
        self.labels.push(None);
        label as u16
    }

    pub fn mark_label(&mut self, label: u16) {
        let slot = self
            .labels
            .get_mut(label as usize)
            .expect("jitcode label out of bounds");
        *slot = Some(self.code.len());
    }

    /// RPython `flatten.py` emits a plain `Bool` exitswitch as opname
    /// `goto_if_not`. The operand is already 0 or 1; the tracer may rebind
    /// it to `CONST_1`.
    pub fn goto_if_not(&mut self, reg: u16, label: u16) {
        self.touch_reg(reg);
        self.write_insn("goto_if_not/iL");
        self.push_u8(reg as u8);
        self.push_label_ref(label);
    }

    /// `jtransform.py optimize_goto_if_not` folds `int_is_true` into the
    /// exitswitch; `flatten.py` then emits `goto_if_not_int_is_true`.
    /// The tracer re-executes `INT_IS_TRUE` and branches with `replace=False`,
    /// so a general integer (`if n != 0`) must not go through `goto_if_not`.
    pub fn goto_if_not_int_is_true(&mut self, reg: u16, label: u16) {
        self.touch_reg(reg);
        self.write_insn("goto_if_not_int_is_true/iL");
        self.push_u8(reg as u8);
        self.push_label_ref(label);
    }

    // jtransform.py `optimize_goto_if_not` folds
    // `v = int_lt(a, b); exitswitch = v` into a single jitcode op
    // emitted by flatten.py:247-250 as `goto_if_not_int_lt/iiL`.
    // blackhole.py:864-911 semantics: take branch iff comparison is
    // false, i.e. `int_lt(a, b) == False` → `position = target`.
    // Canonical layout (`assembler.py:165-174` argcode `iiL`):
    // [a:u8][b:u8][target:u16]. Float and ref variants share the same
    // shape with their respective reg classes.
    pub fn goto_if_not_int_lt(&mut self, a: u16, b: u16, label: u16) {
        self.touch_reg(a);
        self.touch_reg(b);
        self.write_insn("goto_if_not_int_lt/iiL");
        self.push_u8(a as u8);
        self.push_u8(b as u8);
        self.push_label_ref(label);
    }

    pub fn goto_if_not_int_le(&mut self, a: u16, b: u16, label: u16) {
        self.touch_reg(a);
        self.touch_reg(b);
        self.write_insn("goto_if_not_int_le/iiL");
        self.push_u8(a as u8);
        self.push_u8(b as u8);
        self.push_label_ref(label);
    }

    /// `goto_if_not_int_<cmp>/iiL` whose second operand is a `Constant`,
    /// encoded by `assembler.py` `emit_const` as a constants-suffix
    /// register index (see [`Self::record_binop_i_const`]).
    pub fn goto_if_not_int_const(&mut self, opcode: OpCode, a: u16, b: i64, label: u16) {
        let key = match opcode {
            OpCode::IntLt => "goto_if_not_int_lt/iiL",
            OpCode::IntLe => "goto_if_not_int_le/iiL",
            OpCode::IntEq => "goto_if_not_int_eq/iiL",
            OpCode::IntNe => "goto_if_not_int_ne/iiL",
            OpCode::IntGt => "goto_if_not_int_gt/iiL",
            OpCode::IntGe => "goto_if_not_int_ge/iiL",
            other => panic!("goto_if_not_int_const: unsupported opcode {other:?}"),
        };
        self.touch_reg(a);
        self.write_insn(key);
        self.push_u8(a as u8);
        self.push_const_i_operand(b);
        self.push_label_ref(label);
    }

    pub fn goto_if_not_int_eq(&mut self, a: u16, b: u16, label: u16) {
        self.touch_reg(a);
        self.touch_reg(b);
        self.write_insn("goto_if_not_int_eq/iiL");
        self.push_u8(a as u8);
        self.push_u8(b as u8);
        self.push_label_ref(label);
    }

    /// Emit `switch/id`: one int register plus a 2-byte SwitchDictDescr
    /// index.  RPython `flatten.py:270-308` emits the default path as the
    /// fall-through after the switch; matching blackhole.py:954-960, a
    /// missing key leaves the pc at the next instruction.
    pub fn switch(&mut self, value: u16, cases: &[(i64, u16)]) {
        self.touch_reg(value);
        self.write_insn("switch/id");
        self.push_reg_u8(value, "switch/id value");
        let const_keys_in_order: Vec<i64> = cases.iter().map(|(key, _)| *key).collect();
        let descr_idx = self.add_switch_descr(const_keys_in_order);
        self.push_u16(descr_idx);
        self.pending_switches.push((descr_idx, cases.to_vec()));
    }

    pub fn goto_if_not_int_ne(&mut self, a: u16, b: u16, label: u16) {
        self.touch_reg(a);
        self.touch_reg(b);
        self.write_insn("goto_if_not_int_ne/iiL");
        self.push_u8(a as u8);
        self.push_u8(b as u8);
        self.push_label_ref(label);
    }

    pub fn goto_if_not_int_gt(&mut self, a: u16, b: u16, label: u16) {
        self.touch_reg(a);
        self.touch_reg(b);
        self.write_insn("goto_if_not_int_gt/iiL");
        self.push_u8(a as u8);
        self.push_u8(b as u8);
        self.push_label_ref(label);
    }

    pub fn goto_if_not_int_ge(&mut self, a: u16, b: u16, label: u16) {
        self.touch_reg(a);
        self.touch_reg(b);
        self.write_insn("goto_if_not_int_ge/iiL");
        self.push_u8(a as u8);
        self.push_u8(b as u8);
        self.push_label_ref(label);
    }

    // blackhole.py bhimpl_goto_if_not_float_lt float variants — same semantics, float regs.
    pub fn goto_if_not_float_lt(&mut self, a: u16, b: u16, label: u16) {
        self.touch_float_reg(a);
        self.touch_float_reg(b);
        self.write_insn("goto_if_not_float_lt/ffL");
        self.push_u8(a as u8);
        self.push_u8(b as u8);
        self.push_label_ref(label);
    }

    pub fn goto_if_not_float_le(&mut self, a: u16, b: u16, label: u16) {
        self.touch_float_reg(a);
        self.touch_float_reg(b);
        self.write_insn("goto_if_not_float_le/ffL");
        self.push_u8(a as u8);
        self.push_u8(b as u8);
        self.push_label_ref(label);
    }

    pub fn goto_if_not_float_eq(&mut self, a: u16, b: u16, label: u16) {
        self.touch_float_reg(a);
        self.touch_float_reg(b);
        self.write_insn("goto_if_not_float_eq/ffL");
        self.push_u8(a as u8);
        self.push_u8(b as u8);
        self.push_label_ref(label);
    }

    pub fn goto_if_not_float_ne(&mut self, a: u16, b: u16, label: u16) {
        self.touch_float_reg(a);
        self.touch_float_reg(b);
        self.write_insn("goto_if_not_float_ne/ffL");
        self.push_u8(a as u8);
        self.push_u8(b as u8);
        self.push_label_ref(label);
    }

    pub fn goto_if_not_float_gt(&mut self, a: u16, b: u16, label: u16) {
        self.touch_float_reg(a);
        self.touch_float_reg(b);
        self.write_insn("goto_if_not_float_gt/ffL");
        self.push_u8(a as u8);
        self.push_u8(b as u8);
        self.push_label_ref(label);
    }

    pub fn goto_if_not_float_ge(&mut self, a: u16, b: u16, label: u16) {
        self.touch_float_reg(a);
        self.touch_float_reg(b);
        self.write_insn("goto_if_not_float_ge/ffL");
        self.push_u8(a as u8);
        self.push_u8(b as u8);
        self.push_label_ref(label);
    }

    // blackhole.py:922-936 ptr variants — ref regs.
    pub fn goto_if_not_ptr_eq(&mut self, a: u16, b: u16, label: u16) {
        self.touch_ref_reg(a);
        self.touch_ref_reg(b);
        self.write_insn("goto_if_not_ptr_eq/rrL");
        self.push_u8(a as u8);
        self.push_u8(b as u8);
        self.push_label_ref(label);
    }

    pub fn goto_if_not_ptr_ne(&mut self, a: u16, b: u16, label: u16) {
        self.touch_ref_reg(a);
        self.touch_ref_reg(b);
        self.write_insn("goto_if_not_ptr_ne/rrL");
        self.push_u8(a as u8);
        self.push_u8(b as u8);
        self.push_label_ref(label);
    }

    pub fn goto_if_not_ptr_iszero(&mut self, a: u16, label: u16) {
        self.touch_ref_reg(a);
        self.write_insn("goto_if_not_ptr_iszero/rL");
        self.push_u8(a as u8);
        self.push_label_ref(label);
    }

    pub fn goto_if_not_ptr_nonzero(&mut self, a: u16, label: u16) {
        self.touch_ref_reg(a);
        self.write_insn("goto_if_not_ptr_nonzero/rL");
        self.push_u8(a as u8);
        self.push_label_ref(label);
    }

    // blackhole.py `bhimpl_goto_if_not_int_is_zero(a, target, pc)`:
    // fall through iff `not a` (a == 0), else take the target. jtransform.py _rewrite_equality
    // `_rewrite_equality` rewrites `int_eq(x, 0)` → `int_is_zero(x)` so
    // flatten.py:247 specialises the bool exitswitch into this unary form.
    pub fn goto_if_not_int_is_zero(&mut self, a: u16, label: u16) {
        self.touch_reg(a);
        self.write_insn("goto_if_not_int_is_zero/iL");
        self.push_u8(a as u8);
        self.push_label_ref(label);
    }

    pub fn jump(&mut self, label: u16) {
        self.write_insn("goto/L");
        self.push_label_ref(label);
    }

    /// RPython jtransform.py handle_jit_marker__loop_header emits
    /// `SpaceOperation('loop_header', [c_index], None)` with
    /// `Constant(jd.index, lltype.Signed)`. blackhole.py bhimpl_loop_header
    /// bhimpl_loop_header(jdindex) is a no-op; pyjitpl.py opimpl_loop_header
    /// opimpl_loop_header records the jitdriver index for the trace.
    ///
    /// `@arguments("i")` (blackhole.py) parity: jdindex is encoded
    /// as a single register-index byte pointing into the post-regs
    /// constants suffix of `registers_i`. `loop_header` is not in
    /// `assembler.py USE_C_FORM`, so the only valid argcode is
    /// `i` (constants-pool slot) — the short-form `c` is never emitted
    /// regardless of jdindex magnitude. `add_const_i` registers the
    /// value; the placeholder is patched at `finish()` once `num_regs_i`
    /// is final.
    pub fn loop_header(&mut self, jdindex: i64) {
        self.write_insn("loop_header/i");
        let const_idx = self.add_const_i(jdindex);
        let offset = self.code.len();
        self.push_u8(0);
        self.const_patches_u8
            .push((offset, ConstKind::Int, const_idx));
    }

    /// RPython `assembler.py:146-158` `Register('-live-', ...)` arm in
    /// `write_insn` — emit the `live/` opcode followed by the 2-byte
    /// offset returned from `Assembler._encode_liveness`.
    ///
    /// ```python
    /// elif insn[i].is_live():
    ///     self.code.append(chr(self.insns['live/']))   # 148
    ///     live_i = self.get_liveness_info(insn[i:], 'int')
    ///     live_r = self.get_liveness_info(insn[i:], 'ref')
    ///     live_f = self.get_liveness_info(insn[i:], 'float')
    ///     self._encode_liveness(live_i, live_r, live_f)  # 158
    /// ```
    ///
    /// Mirrors `Assembler::_encode_liveness` (assembler.py): the
    /// cache key is built from the set-equivalent (sorted, deduplicated)
    /// view of each `live_*` slice, so callers may pass arbitrary order.
    ///
    /// Pyre keeps `live_placeholder` + `patch_live_offset` below for
    /// deferred-patch callers that must emit the LIVE op before the
    /// canonical liveness entry is known (e.g., the macro state-field
    /// jitcode build registers its canonical entry once per process,
    /// long after individual jitcode bytes are written).
    pub fn live(
        &mut self,
        asm: &mut majit_jitcode::codewriter::assembler::Assembler,
        live_i: &[u8],
        live_r: &[u8],
        live_f: &[u8],
    ) {
        // assembler.py:148 `self.code.append(chr(self.insns['live/']))`
        self.write_insn("live/");
        // assembler.py `self._encode_liveness(live_i, live_r, live_f)`
        asm._encode_liveness(live_i, live_r, live_f, &mut self.code);
    }

    /// Emit the raw `BC_LIVE` opcode + 2-byte zero offset, returning the
    /// operand offset so a caller can record the patch for one of the
    /// deferred-patch lists.
    fn write_live_placeholder_bytes(&mut self) -> usize {
        self.write_insn("live/");
        let patch_offset = self.code.len();
        self.push_u16(0);
        patch_offset
    }

    /// RPython assembler.py: emit `live/` followed by a 2-byte offset into
    /// the shared all_liveness byte string.  Used for the leading dummy
    /// `BC_LIVE` slot at the start of every per-pc JitCode (which has no
    /// per-marker triple of its own — it points at the canonical "all
    /// live" entry).  The 2-byte slot is written as `0x0000` here and
    /// back-patched during [`Self::finalize_liveness`] via
    /// `Assembler::ensure_canonical_liveness_offset`.  Returns the operand
    /// offset so callers may chain into a custom patcher if needed.
    pub fn live_placeholder(&mut self) -> usize {
        let patch_offset = self.write_live_placeholder_bytes();
        self.pending_canonical_patches.push(patch_offset);
        patch_offset
    }

    pub fn patch_live_offset(&mut self, patch_offset: usize, offset: u16) {
        let bytes = offset.to_le_bytes();
        self.code[patch_offset] = bytes[0];
        self.code[patch_offset + 1] = bytes[1];
    }

    /// Deferred-patch entry point: emit a `live/`
    /// opcode followed by a 2-byte placeholder offset (mirroring
    /// [`Self::live_placeholder`]) and record the per-marker
    /// `(live_i, live_r, live_f)` triple in `pending_live_triples` so
    /// [`Self::finalize_liveness`] can later resolve and patch the offset.
    ///
    /// Each `live_*` slice must be a sorted+dedup register-set view
    /// matching the macro lowerer's
    /// `compute_per_marker_liveness` output (`liveness.py:33-79`).  Pyre
    /// renormalises in `_register_liveness_offset`, so callers may pass
    /// arbitrary order; passing the lowerer's already-sorted output
    /// matches RPython's `liveness.py encode_liveness live = sorted(live)` shape.
    pub fn live_placeholder_with_triple(
        &mut self,
        live_i: &[u8],
        live_r: &[u8],
        live_f: &[u8],
    ) -> usize {
        let patch_offset = self.write_live_placeholder_bytes();
        self.pending_live_triples.push((
            patch_offset,
            live_i.to_vec(),
            live_r.to_vec(),
            live_f.to_vec(),
        ));
        patch_offset
    }

    /// Finalisation step: register every pending
    /// per-marker liveness triple into `asm` (deduplicating against the
    /// shared `all_liveness_positions`) and rewrite each corresponding
    /// `live/<offset>` BC_LIVE slot via [`Self::patch_live_offset`].
    ///
    /// Mirrors `assembler.py:146-158`'s per-marker
    /// `_encode_liveness(live_i, live_r, live_f) → encode_offset(pos)`
    /// pair, deferred to a single post-emit pass for callers that can only
    /// acquire the driver-shared `&mut Assembler` after a JitCode body is
    /// built.  The `#[jit_interp]` dispatch singleton and its embedded
    /// sub-JitCodes use this path during install/prebuild.
    ///
    /// Idempotent: drains `pending_live_triples` and
    /// `pending_canonical_patches` so a second call is a no-op.
    ///
    /// Within a single call, canonical patches are processed before
    /// per-marker triples; this lets `ensure_canonical_liveness_offset`
    /// short-circuit to the cached offset on every subsequent invocation.
    /// At the macro level the per-pc liveness *prebuild*
    /// (`liveness_prebuild_tokens` in `jitcode_lower.rs`) directly calls
    /// `_register_liveness_offset` for every walker-emitted marker before
    /// any `finalize_liveness` runs, so by the time the first leading
    /// dummy fires (at trace time) per-marker triples already populate
    /// `all_liveness` and the safety-net
    /// `ensure_canonical_liveness_offset` call in
    /// `install_canonical_liveness` has appended canonical at the end.
    /// This matches RPython `assembler.assemble`'s shape: real `-live-`
    /// markers occupy the IR-walk-ordered positions; the canonical
    /// "all-live" entry exists only as a pyre-side affordance for the
    /// leading-dummy assertion.
    ///
    /// If `pending_canonical_patches` is non-empty,
    /// `Assembler::ensure_canonical_liveness_offset` must succeed (i.e.
    /// `set_canonical_liveness_triple` must have been called earlier);
    /// the test paths that emit a leading dummy without ever calling
    /// `finalize_liveness` keep the original `0x0000` placeholder bytes
    /// in `code` and never observe this assertion.
    pub fn finalize_liveness(&mut self, asm: &mut majit_jitcode::codewriter::assembler::Assembler) {
        let canonical_patches = std::mem::take(&mut self.pending_canonical_patches);
        if !canonical_patches.is_empty() {
            let canonical_off = asm.ensure_canonical_liveness_offset();
            assert!(
                canonical_off < (1 << 16),
                "canonical all_liveness offset {} exceeds 2-byte encoding",
                canonical_off
            );
            let canonical_off_u16 = canonical_off as u16;
            for patch_offset in canonical_patches {
                self.patch_live_offset(patch_offset, canonical_off_u16);
            }
        }
        for (patch_offset, live_i, live_r, live_f) in std::mem::take(&mut self.pending_live_triples)
        {
            let pos = asm._register_liveness_offset(&live_i, &live_r, &live_f);
            // assembler.py `encode_offset(pos, self.code)` — pyre
            // patches the already-emitted 2-byte slot in place rather
            // than appending; the bit pattern is identical.
            assert!(
                pos < (1 << 16),
                "all_liveness offset {} exceeds 2-byte encoding",
                pos
            );
            self.patch_live_offset(patch_offset, pos as u16);
        }
    }

    /// RPython blackhole.py `catch_exception/L`.
    pub fn catch_exception(&mut self, label: u16) {
        self.write_insn("catch_exception/L");
        self.push_label_ref(label);
    }

    /// RPython blackhole.py:993 `last_exc_value/>r`.
    pub fn last_exc_value(&mut self, dst: u16) {
        self.touch_ref_reg(dst);
        self.write_insn("last_exc_value/>r");
        self.push_u8(dst as u8);
    }

    /// RPython blackhole.py:987 `last_exception/>i`.
    pub fn last_exception(&mut self, dst: u16) {
        self.touch_reg(dst);
        self.write_insn("last_exception/>i");
        self.push_u8(dst as u8);
    }

    /// RPython blackhole.py:976-985 `goto_if_exception_mismatch/iL`.
    ///
    /// `vtable` is an int operand that may be either a real Int
    /// register or a const-pool slot synthesized as `num_regs_i +
    /// pool_idx` by `pyre-jit/src/jit/assembler.rs::expect_int_reg_or_pool`.
    /// Validated via the union helper so the operand byte's
    /// `[0, num_regs_i + num_consts_i)` range is enforced before the
    /// `as u8` push.
    pub fn goto_if_exception_mismatch(&mut self, vtable: u16, label: u16) {
        self.touch_int_reg_or_pool_slot(vtable);
        self.write_insn("goto_if_exception_mismatch/iL");
        self.push_u8(vtable as u8);
        self.push_label_ref(label);
    }

    /// blackhole.py bhimpl_jit_merge_point: portal merge point.
    ///
    /// assembler.py:181-196 parity: encodes jdindex + 6 typed register
    /// lists (greens_i, greens_r, greens_f, reds_i, reds_r, reds_f).
    /// Each list is `[length:u8][reg_indices:u8...]`.
    ///
    /// jdindex is emitted per assembler.py USE_C_FORM rules —
    /// `'c'` (raw signed byte) when fitting in `i8`, otherwise `'i'`
    /// (constants-pool slot). The blackhole `@arguments("i", ...)`
    /// decoder (blackhole.py) interprets the byte per the
    /// runtime argcode.
    ///
    /// jtransform.py:1704 emits `Constant(self.portal_jd.index,
    /// lltype.Signed)` so the assembler-side jdindex is signed; the
    /// USE_C_FORM short-form check is `-128 <= value <= 127`
    /// (assembler.py:99-107).
    ///
    /// interp_jit.py:64 portal contract for pyre's current
    /// `pypyjitdriver`:
    ///   greens = ['next_instr', 'is_being_profiled', 'pycode']
    ///   reds = ['frame', 'ec']
    /// — so `greens_f`, `reds_i`, `reds_f` arrive empty today. The
    /// helper does not assume that: a future jitdriver with float
    /// greens or red ints flows through the same code path with
    /// non-empty lists.
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython metainterpreter routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and frame ownership"
    )]
    pub fn jit_merge_point(
        &mut self,
        jdindex: i64,
        greens_i: &[u8],
        greens_r: &[u8],
        greens_f: &[u8],
        reds_i: &[u8],
        reds_r: &[u8],
        reds_f: &[u8],
    ) {
        // Capture the bytecode offset of the first OPCODE byte (before
        // `write_insn` pushes it).  PyPy's portal dispatch loop executes
        // `jit_merge_point()` at every bytecode dispatch, so a lowered
        // portal jitcode can contain more than one merge point when pyre
        // materializes several Python loop headers in one dispatch body.
        // The offset is only used by `register_dispatch_jitcode` for
        // schema validation, so keeping the first one preserves that
        // validation without rejecting later structurally identical merge
        // points.
        if self.jit_merge_point_offset.is_none() {
            self.jit_merge_point_offset = Some(self.code.len());
        }
        if (-128..=127).contains(&jdindex) {
            self.write_insn("jit_merge_point/cIRFIRF");
            self.push_u8((jdindex & 0xFF) as u8);
        } else {
            self.write_insn("jit_merge_point/iIRFIRF");
            let jdindex_const = self.add_const_i(jdindex);
            let jdindex_offset = self.code.len();
            self.push_u8(0);
            self.const_patches_u8
                .push((jdindex_offset, ConstKind::Int, jdindex_const));
        }
        for list in [greens_i, greens_r, greens_f, reds_i, reds_r, reds_f] {
            // RPython `assembler.py` encodes list lengths as a single byte
            // (`chr(len(lst))`); going past 255 silently wraps in Rust and
            // mis-encodes the per-list count.  Strict assert so the wrap
            // surfaces as a hard fail.
            let len = list.len();
            assert!(
                len < 256,
                "jit_merge_point list length {len} exceeds u8 byte encoding"
            );
            self.push_u8(len as u8);
            for &idx in list {
                self.push_u8(idx);
            }
        }
    }

    pub fn abort(&mut self) {
        self.write_insn("abort/");
        self.has_abort = true;
    }

    /// RPython `blackhole.py:841-862` typed return opcodes.
    /// The return value is in register `src`; `void_return` has no operand.
    pub fn int_return(&mut self, src: u16) {
        self.touch_reg(src);
        self.write_insn("int_return/i");
        self.push_u8(src as u8);
    }

    pub fn ref_return(&mut self, src: u16) {
        self.touch_ref_reg(src);
        self.write_insn("ref_return/r");
        self.push_u8(src as u8);
    }

    /// Emit a recursive portal call returning an int.
    ///
    /// `jtransform.py handle_recursive_call` + `assembler.py` encode
    /// `recursive_call_i/iIRFIRF>i`: jdindex as an `i` register (constant
    /// pool), then the six kind lists, then the dest byte so
    /// `blackhole.py _setup_return_value_i` can read `code[position-1]`.
    pub fn recursive_call_int(
        &mut self,
        jd_index: u16,
        result_dst: u16,
        greens: &[(JitArgKind, u16)],
        reds: &[(JitArgKind, u16)],
    ) {
        self.touch_reg(result_dst);
        self.emit_recursive_call(
            jitcode::insns::BC_RECURSIVE_CALL_INT,
            jd_index,
            Some(result_dst),
            greens,
            reds,
            Some('i'),
        );
    }

    /// Ref-result sibling of [`Self::recursive_call_int`]
    /// (`blackhole.py bhimpl_recursive_call_r`).
    pub fn recursive_call_ref(
        &mut self,
        jd_index: u16,
        result_dst: u16,
        greens: &[(JitArgKind, u16)],
        reds: &[(JitArgKind, u16)],
    ) {
        self.touch_ref_reg(result_dst);
        self.emit_recursive_call(
            jitcode::insns::BC_RECURSIVE_CALL_REF,
            jd_index,
            Some(result_dst),
            greens,
            reds,
            Some('r'),
        );
    }

    /// Float-result sibling of [`Self::recursive_call_int`]
    /// (`blackhole.py bhimpl_recursive_call_f`).
    pub fn recursive_call_float(
        &mut self,
        jd_index: u16,
        result_dst: u16,
        greens: &[(JitArgKind, u16)],
        reds: &[(JitArgKind, u16)],
    ) {
        self.touch_float_reg(result_dst);
        self.emit_recursive_call(
            jitcode::insns::BC_RECURSIVE_CALL_FLOAT,
            jd_index,
            Some(result_dst),
            greens,
            reds,
            Some('f'),
        );
    }

    /// Void-result sibling of [`Self::recursive_call_int`]
    /// (`blackhole.py bhimpl_recursive_call_v`).  No dest byte: the
    /// assembler omits `emit_call_result_arg` when the result is void.
    pub fn recursive_call_void(
        &mut self,
        jd_index: u16,
        greens: &[(JitArgKind, u16)],
        reds: &[(JitArgKind, u16)],
    ) {
        self.emit_recursive_call(
            jitcode::insns::BC_RECURSIVE_CALL_VOID,
            jd_index,
            None,
            greens,
            reds,
            None,
        );
    }

    /// `jtransform.py handle_recursive_call` payload:
    /// `recursive_call_{kind}(jd_index, G_I, G_R, G_F, R_I, R_R, R_F)`
    /// then the dest register for a typed result.
    fn emit_recursive_call(
        &mut self,
        opcode: u8,
        jd_index: u16,
        result_dst: Option<u16>,
        greens: &[(JitArgKind, u16)],
        reds: &[(JitArgKind, u16)],
        resulttype: Option<char>,
    ) {
        for &(kind, src) in greens.iter().chain(reds.iter()) {
            self.touch_call_arg(JitCallArg { kind, reg: src });
        }
        self.start_instr(opcode);
        // `blackhole.py bhimpl_recursive_call_*` takes jdindex as
        // `@arguments("self", "i", ...)` — a register, not a literal u16.
        // Same const-pool patch `jit_merge_point/iIRFIRF` uses.
        let jdindex_const = self.add_const_i(i64::from(jd_index));
        let jdindex_offset = self.code.len();
        self.push_u8(0);
        self.const_patches_u8
            .push((jdindex_offset, ConstKind::Int, jdindex_const));
        self.emit_kind_reg_list(greens, JitArgKind::Int);
        self.emit_kind_reg_list(greens, JitArgKind::Ref);
        self.emit_kind_reg_list(greens, JitArgKind::Float);
        self.emit_kind_reg_list(reds, JitArgKind::Int);
        self.emit_kind_reg_list(reds, JitArgKind::Ref);
        self.emit_kind_reg_list(reds, JitArgKind::Float);
        if let Some(dst) = result_dst {
            self.push_reg_u8(dst, "recursive_call result");
        }
        if let Some(kind) = resulttype {
            self.record_resulttype(kind);
        }
    }

    fn emit_kind_reg_list(&mut self, regs: &[(JitArgKind, u16)], want: JitArgKind) {
        let selected: Vec<u16> = regs
            .iter()
            .filter(|(kind, _)| *kind == want)
            .map(|(_, src)| *src)
            .collect();
        assert!(
            selected.len() < 256,
            "recursive_call list length {} exceeds u8 byte encoding",
            selected.len()
        );
        self.push_u8(selected.len() as u8);
        for src in selected {
            self.push_u8(src as u8);
        }
    }

    /// Constant-operand form of `ref_return`. `flatten.py make_return
    /// make_return` emits `ref_return` with `getcolor(v)`, and
    /// `flatten.py getcolor` passes a `Constant` through
    /// unchanged; `assembler.py emit_const` then encodes it as
    /// the high register-index byte `num_regs_r + pool_idx`, which the
    /// `r` argcode reader resolves transparently from the ref constant
    /// window. See `load_const_r` for the shared const-patch mechanism.
    pub fn ref_return_const(&mut self, value: i64) {
        let const_idx = self.add_const_r(value);
        self.write_insn("ref_return/r");
        let off = self.code.len();
        self.push_u8(0);
        self.const_patches_u8.push((off, ConstKind::Ref, const_idx));
    }

    pub fn float_return(&mut self, src: u16) {
        self.touch_float_reg(src);
        self.write_insn("float_return/f");
        self.push_u8(src as u8);
    }

    pub fn void_return(&mut self) {
        self.write_insn("void_return/");
    }

    pub fn abort_permanent(&mut self) {
        self.write_insn("abort_permanent/");
    }

    pub fn unreachable(&mut self) {
        self.write_insn("unreachable/");
    }

    /// blackhole.py bhimpl_raise(excvalue): raise exception from register.
    pub fn emit_raise(&mut self, src: u16) {
        self.touch_ref_reg(src);
        self.write_insn("raise/r");
        self.push_u8(src as u8);
    }

    /// Constant-operand form of `emit_raise`. `flatten.py make_link
    /// make_exception_link` / `make_return` reraise paths emit `raise`
    /// with `getcolor(exc_value)`, which is a `Constant` for a fixed
    /// exception instance (e.g. the overflow reraise). Encodes it via
    /// the high register-index ref constant window like
    /// `ref_return_const`.
    pub fn emit_raise_const(&mut self, value: i64) {
        let const_idx = self.add_const_r(value);
        self.write_insn("raise/r");
        let off = self.code.len();
        self.push_u8(0);
        self.const_patches_u8.push((off, ConstKind::Ref, const_idx));
    }

    /// blackhole.py bhimpl_reraise(): re-raise exception_last_value.
    pub fn emit_reraise(&mut self) {
        self.write_insn("reraise/");
    }

    /// pyjitpl.py opimpl_int_guard_value: promote int register to constant.
    ///
    /// Blackhole: no-op (value passes through).
    /// Tracing: emits GUARD_VALUE to specialize the trace on this value.
    pub fn int_guard_value(&mut self, src: u16) {
        self.touch_reg(src);
        self.write_insn("int_guard_value/i");
        self.push_u8(src as u8);
    }

    /// pyjitpl.py opimpl_assert_not_none: record that `src` is
    /// non-null.
    ///
    /// Blackhole: asserts the concrete ref is non-null
    /// (`blackhole.rs`'s `handler_assert_not_none`) and advances past
    /// the operand byte.  Tracing dispatcher routes
    /// through `TraceCtx::trace_assert_not_none(opref)` which gates on
    /// `heap_cache.is_nullity_known` and bumps `HEAPCACHED_OPS` on
    /// cache hit per pyjitpl.py.
    pub fn assert_not_none(&mut self, src: u16) {
        self.write_insn("assert_not_none/r");
        self.push_reg_u8(src, "assert_not_none");
    }

    /// pyjitpl.py opimpl_record_exact_class: record that
    /// `src`'s class is exactly `cls` (vtable pointer in the int bank).
    ///
    /// Blackhole: no-op (`blackhole.rs`'s `handler_record_exact_class`
    /// advances past the two operand bytes).  Tracing dispatcher routes
    /// through `TraceCtx::trace_record_exact_class(opref, cls_const)`
    /// which gates on `heap_cache.is_class_known` and bumps
    /// `HEAPCACHED_OPS` on cache hit per pyjitpl.py.  Argcodes
    /// `ri` mirrors `blackhole.py @arguments("r", "i")`.
    pub fn record_exact_class(&mut self, src: u16, cls: u16) {
        self.write_insn("record_exact_class/ri");
        self.push_reg_u8(src, "record_exact_class src");
        self.push_reg_u8(cls, "record_exact_class cls");
    }

    /// pyjitpl.py opimpl_ref_guard_value: promote ref register to constant.
    pub fn ref_guard_value(&mut self, src: u16) {
        self.touch_ref_reg(src);
        self.write_insn("ref_guard_value/r");
        self.push_u8(src as u8);
    }

    /// pyjitpl.py opimpl_float_guard_value: promote float register to constant.
    pub fn float_guard_value(&mut self, src: u16) {
        self.touch_float_reg(src);
        self.write_insn("float_guard_value/f");
        self.push_u8(src as u8);
    }

    pub fn inline_call(&mut self, sub_jitcode_idx: u16) {
        self.inline_call_r_v(sub_jitcode_idx, &[], None);
    }

    /// Emit RPython's canonical `inline_call_{r,ir,irf}_{i,r,f,v}`
    /// bytecode shape.
    ///
    /// This is deliberately separate from [`Self::inline_call_r_r`] and its
    /// siblings.  Those methods predate the codewriter port and emit pyre's
    /// opaque `inline_call_nested_ext/P` payload, including explicit callee
    /// destination registers.  `jtransform.py handle_regular_call` instead
    /// emits a JitCode descr followed by kind-separated varlists; the callee's
    /// `MIFrame.setup_call` places each list at the start of its matching
    /// register bank.  `pyre-jit`'s SSA assembler uses this method for that
    /// orthodox stream.
    pub fn canonical_inline_call(
        &mut self,
        key: &'static str,
        sub_jitcode_idx: u16,
        args_i: Option<&[u16]>,
        args_r: Option<&[u16]>,
        args_f: Option<&[u16]>,
        result: Option<(JitArgKind, u16)>,
    ) {
        // The callee places each varlist at the start of its matching register
        // bank (`MIFrame.setup_call`), so an argument's index within its kind
        // is the callee register it lands in.  The triples are grouped in the
        // same int/ref/float order `inline_call_grouped` uses, so a caller
        // whose kinds disagree with the callee's declared signature is
        // reported identically on both emit paths.
        let typed_args: Vec<(JitArgKind, u16, u16)> = [
            (JitArgKind::Int, args_i),
            (JitArgKind::Ref, args_r),
            (JitArgKind::Float, args_f),
        ]
        .into_iter()
        .flat_map(|(kind, args)| {
            args.unwrap_or(&[])
                .iter()
                .enumerate()
                .map(move |(callee_dst, &caller_src)| (kind, caller_src, callee_dst as u16))
        })
        .collect();
        self.check_inline_call_arg_classes(sub_jitcode_idx, &typed_args);
        self.write_insn(key);
        self.push_u16(sub_jitcode_idx);
        if let Some(args) = args_i {
            self.push_canonical_inline_varlist(args, JitArgKind::Int);
        }
        if let Some(args) = args_r {
            self.push_canonical_inline_varlist(args, JitArgKind::Ref);
        }
        if let Some(args) = args_f {
            self.push_canonical_inline_varlist(args, JitArgKind::Float);
        }
        if let Some((kind, dst)) = result {
            match kind {
                JitArgKind::Int => self.touch_reg(dst),
                JitArgKind::Ref => self.touch_ref_reg(dst),
                JitArgKind::Float => self.touch_float_reg(dst),
            }
            self.push_reg_u8(dst, "canonical inline_call return");
        }
    }

    fn push_canonical_inline_varlist(&mut self, args: &[u16], kind: JitArgKind) {
        if args.len() > u8::MAX as usize {
            self.encoding_overflow = true;
            self.push_u8(0);
            return;
        }
        self.push_u8(args.len() as u8);
        for &src in args {
            // `assembler.py write_insn` ListOfKind items: `emit_reg` for a
            // Register, `emit_const` for a Constant.  pyre fuses those at
            // this surface — `expect_list_regs_or_pool` numbers a ConstInt
            // `num_regs_i + pool_idx`, which is outside the frozen real
            // register window `touch_reg` accepts.
            match kind {
                JitArgKind::Int => self.touch_int_reg_or_pool_slot(src),
                JitArgKind::Ref => self.touch_ref_reg_or_pool_slot(src),
                JitArgKind::Float => self.touch_float_reg_or_pool_slot(src),
            }
            self.push_reg_u8(src, "canonical inline_call argument");
        }
    }

    pub fn inline_call_r_i(
        &mut self,
        sub_jitcode_idx: u16,
        args_r: &[(u16, u16)],
        return_i: Option<u16>,
    ) {
        self.inline_call_grouped(
            sub_jitcode_idx,
            false,
            false,
            &[],
            args_r,
            &[],
            return_i,
            None,
            None,
        );
    }

    pub fn inline_call_r_r(
        &mut self,
        sub_jitcode_idx: u16,
        args_r: &[(u16, u16)],
        return_r: Option<u16>,
    ) {
        self.inline_call_grouped(
            sub_jitcode_idx,
            false,
            false,
            &[],
            args_r,
            &[],
            None,
            return_r,
            None,
        );
    }

    pub fn inline_call_r_v(
        &mut self,
        sub_jitcode_idx: u16,
        args_r: &[(u16, u16)],
        _return_v: Option<u16>,
    ) {
        self.inline_call_grouped(
            sub_jitcode_idx,
            false,
            false,
            &[],
            args_r,
            &[],
            None,
            None,
            None,
        );
    }

    pub fn inline_call_ir_i(
        &mut self,
        sub_jitcode_idx: u16,
        args_i: &[(u16, u16)],
        args_r: &[(u16, u16)],
        return_i: Option<u16>,
    ) {
        self.inline_call_grouped(
            sub_jitcode_idx,
            true,
            false,
            args_i,
            args_r,
            &[],
            return_i,
            None,
            None,
        );
    }

    pub fn inline_call_ir_r(
        &mut self,
        sub_jitcode_idx: u16,
        args_i: &[(u16, u16)],
        args_r: &[(u16, u16)],
        return_r: Option<u16>,
    ) {
        self.inline_call_grouped(
            sub_jitcode_idx,
            true,
            false,
            args_i,
            args_r,
            &[],
            None,
            return_r,
            None,
        );
    }

    pub fn inline_call_ir_v(
        &mut self,
        sub_jitcode_idx: u16,
        args_i: &[(u16, u16)],
        args_r: &[(u16, u16)],
        _return_v: Option<u16>,
    ) {
        self.inline_call_grouped(
            sub_jitcode_idx,
            true,
            false,
            args_i,
            args_r,
            &[],
            None,
            None,
            None,
        );
    }

    pub fn inline_call_irf_i(
        &mut self,
        sub_jitcode_idx: u16,
        args_i: &[(u16, u16)],
        args_r: &[(u16, u16)],
        args_f: &[(u16, u16)],
        return_i: Option<u16>,
    ) {
        self.inline_call_grouped(
            sub_jitcode_idx,
            true,
            true,
            args_i,
            args_r,
            args_f,
            return_i,
            None,
            None,
        );
    }

    pub fn inline_call_irf_r(
        &mut self,
        sub_jitcode_idx: u16,
        args_i: &[(u16, u16)],
        args_r: &[(u16, u16)],
        args_f: &[(u16, u16)],
        return_r: Option<u16>,
    ) {
        self.inline_call_grouped(
            sub_jitcode_idx,
            true,
            true,
            args_i,
            args_r,
            args_f,
            None,
            return_r,
            None,
        );
    }

    pub fn inline_call_irf_f(
        &mut self,
        sub_jitcode_idx: u16,
        args_i: &[(u16, u16)],
        args_r: &[(u16, u16)],
        args_f: &[(u16, u16)],
        return_f: Option<u16>,
    ) {
        self.inline_call_grouped(
            sub_jitcode_idx,
            true,
            true,
            args_i,
            args_r,
            args_f,
            None,
            None,
            return_f,
        );
    }

    pub fn inline_call_irf_v(
        &mut self,
        sub_jitcode_idx: u16,
        args_i: &[(u16, u16)],
        args_r: &[(u16, u16)],
        args_f: &[(u16, u16)],
        _return_v: Option<u16>,
    ) {
        self.inline_call_grouped(
            sub_jitcode_idx,
            true,
            true,
            args_i,
            args_r,
            args_f,
            None,
            None,
            None,
        );
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython metainterpreter routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and frame ownership"
    )]
    fn inline_call_grouped(
        &mut self,
        sub_jitcode_idx: u16,
        has_i_list: bool,
        has_f_list: bool,
        args_i: &[(u16, u16)],
        args_r: &[(u16, u16)],
        args_f: &[(u16, u16)],
        return_i: Option<u16>,
        return_r: Option<u16>,
        return_f: Option<u16>,
    ) {
        let mut typed_args = Vec::with_capacity(args_i.len() + args_r.len() + args_f.len());
        typed_args.extend(
            args_i
                .iter()
                .map(|&(caller_src, callee_dst)| (JitArgKind::Int, caller_src, callee_dst)),
        );
        typed_args.extend(
            args_r
                .iter()
                .map(|&(caller_src, callee_dst)| (JitArgKind::Ref, caller_src, callee_dst)),
        );
        typed_args.extend(
            args_f
                .iter()
                .map(|&(caller_src, callee_dst)| (JitArgKind::Float, caller_src, callee_dst)),
        );
        self.inline_call_typed(
            sub_jitcode_idx,
            has_i_list,
            has_f_list,
            &typed_args,
            return_i,
            return_r,
            return_f,
        );
    }

    /// Reject an inline call whose argument kinds disagree with the callee's
    /// own declared signature.
    ///
    /// `BC_INLINE_CALL` moves each argument into the callee frame's register
    /// file *for that argument's kind*: an `Int` argument lands among the
    /// callee's integer registers, a `Ref` argument among its reference
    /// registers.  Nothing downstream re-checks that the register the callee
    /// goes on to read is one this call actually wrote, so a caller passing an
    /// integer where the callee declares a reference leaves the callee reading
    /// whatever its reference register happened to hold.  That is not a crash
    /// but a wrong value, and it appears only once the frame is compiled —
    /// the interpreter runs the host function directly and never decodes this
    /// bytecode, so the two disagree with no diagnostic in between.
    ///
    /// The kinds are compared as per-kind counts rather than position by
    /// position.  `arg_classes` is in source parameter order while a caller
    /// groups its arguments by kind, so a `(i, r, i)` signature legitimately
    /// arrives as `i, i, r`; each argument already names the callee register
    /// that places it, so the order within a kind is not this check's business.
    ///
    /// Only a callee that declares a signature is checked.  A jitcode this
    /// assembler builds carries `BhCallDescr::default()` — an empty
    /// `arg_classes` — because no caller stages one through `set_calldescr`,
    /// so an empty declaration reads as "not stated" rather than "takes
    /// nothing", and a genuinely nullary callee is indistinguishable from it.
    fn check_inline_call_arg_classes(&self, sub_jitcode_idx: u16, args: &[(JitArgKind, u16, u16)]) {
        // `as_jitcode_owned`, so a recursive helper's back edge is checked
        // like any other callee once it can be — and `let ... else` rather than
        // an unwrap because it cannot be during the helper's own assembly: the
        // `Weak` handed out by `Arc::new_cyclic` does not upgrade until the
        // value is published.  Declining there costs nothing; the callee is
        // this same jitcode and its own emit sites are checked.
        let Some(callee) = self
            .descrs
            .get(sub_jitcode_idx as usize)
            .and_then(RuntimeBhDescr::as_jitcode_owned)
        else {
            return;
        };
        let Some(body) = callee.try_body() else {
            return;
        };
        let declared = &body.calldescr.arg_classes;
        if declared.is_empty() {
            return;
        }
        let declares = |want: char| declared.chars().filter(|&class| class == want).count();
        let passes = |want: JitArgKind| args.iter().filter(|&&(kind, ..)| kind == want).count();
        assert!(
            declares('i') == passes(JitArgKind::Int)
                && declares('r') == passes(JitArgKind::Ref)
                && declares('f') == passes(JitArgKind::Float),
            "inline call into jitcode {:?} passes {} int / {} ref / {} float argument(s), \
             but that jitcode's signature declares {declared:?}",
            callee.name,
            passes(JitArgKind::Int),
            passes(JitArgKind::Ref),
            passes(JitArgKind::Float),
        );
    }

    fn inline_call_typed(
        &mut self,
        sub_jitcode_idx: u16,
        has_i_list: bool,
        has_f_list: bool,
        args: &[(JitArgKind, u16, u16)],
        return_i: Option<u16>,
        return_r: Option<u16>,
        return_f: Option<u16>,
    ) {
        self.check_inline_call_arg_classes(sub_jitcode_idx, args);
        for &(kind, caller_src, _) in args {
            match kind {
                JitArgKind::Int => self.touch_reg(caller_src),
                JitArgKind::Ref => self.touch_ref_reg(caller_src),
                JitArgKind::Float => self.touch_float_reg(caller_src),
            }
        }
        if let Some(caller_dst) = return_i {
            self.touch_reg(caller_dst);
        }
        if let Some(caller_dst) = return_r {
            self.touch_ref_reg(caller_dst);
        }
        if let Some(caller_dst) = return_f {
            self.touch_float_reg(caller_dst);
        }
        let opcode = match (has_i_list, has_f_list, return_i, return_r, return_f) {
            (false, false, Some(_), None, None) => jitcode::insns::BC_INLINE_CALL_R_I,
            (false, false, None, Some(_), None) => jitcode::insns::BC_INLINE_CALL_R_R,
            (false, false, None, None, None) => jitcode::insns::BC_INLINE_CALL_R_V,
            (true, false, Some(_), None, None) => jitcode::insns::BC_INLINE_CALL_IR_I,
            (true, false, None, Some(_), None) => jitcode::insns::BC_INLINE_CALL_IR_R,
            (true, false, None, None, None) => jitcode::insns::BC_INLINE_CALL_IR_V,
            (true, true, Some(_), None, None) => jitcode::insns::BC_INLINE_CALL_IRF_I,
            (true, true, None, Some(_), None) => jitcode::insns::BC_INLINE_CALL_IRF_R,
            (true, true, None, None, Some(_)) => jitcode::insns::BC_INLINE_CALL_IRF_F,
            (true, true, None, None, None) => jitcode::insns::BC_INLINE_CALL_IRF_V,
            slots => panic!(
                "inline_call_typed: unsupported shape has_i={has_i_list} has_f={has_f_list} \
                 returns {slots:?}"
            ),
        };
        // `bhimpl_inline_call_*` / `exec_typed_inline_call`: descr + I/R/F
        // lists. Callee register i of each bank is list position i, so a
        // non-positional callee_dst would be dropped.
        let mut args_i = Vec::new();
        let mut args_r = Vec::new();
        let mut args_f = Vec::new();
        for &(kind, caller_src, callee_dst) in args {
            match kind {
                JitArgKind::Int => {
                    assert_eq!(
                        callee_dst as usize,
                        args_i.len(),
                        "inline_call I-list is positional like bhimpl_inline_call_*"
                    );
                    args_i.push(caller_src);
                }
                JitArgKind::Ref => {
                    assert_eq!(
                        callee_dst as usize,
                        args_r.len(),
                        "inline_call R-list is positional like bhimpl_inline_call_*"
                    );
                    args_r.push(caller_src);
                }
                JitArgKind::Float => {
                    assert_eq!(
                        callee_dst as usize,
                        args_f.len(),
                        "inline_call F-list is positional like bhimpl_inline_call_*"
                    );
                    args_f.push(caller_src);
                }
            }
        }
        self.start_instr(opcode);
        self.push_u16(sub_jitcode_idx);
        if has_i_list {
            self.push_reg_list(&args_i);
        }
        self.push_reg_list(&args_r);
        if has_f_list {
            self.push_reg_list(&args_f);
        }
        // Void `bhimpl_inline_call_*` has no result byte. Typed results
        // write the destination last so `make_result_of_lastop` can
        // recover it as `code[pc-1]`.
        match (return_i, return_r, return_f) {
            (Some(dst), None, None) => {
                self.push_reg_u8(dst, "inline_call return");
                self.record_resulttype('i');
            }
            (None, Some(dst), None) => {
                self.push_reg_u8(dst, "inline_call return");
                self.record_resulttype('r');
            }
            (None, None, Some(dst)) => {
                self.push_reg_u8(dst, "inline_call return");
                self.record_resulttype('f');
            }
            (None, None, None) => {}
            slots => {
                panic!("inline_call_typed: at most one return slot may be Some, got {slots:?}")
            }
        }
    }

    fn push_reg_list(&mut self, regs: &[u16]) {
        // `assembler.py write_insn` `assert len(lst) <= 255, "list too long!"`
        // for `ListOfKind`. A silent `as u8` wrap turns 256 args into a
        // zero-length list and leaves the register bytes as later opcodes.
        assert!(
            regs.len() <= 255,
            "inline_call ListOfKind length {} exceeds 255",
            regs.len()
        );
        self.push_u8(regs.len() as u8);
        for &reg in regs {
            self.push_reg_u8(reg, "inline_call list");
        }
    }

    /// adapter — bridges `fn_ptr_idx`-using emit sites to the
    /// canonical `residual_call_*_v` byte layout. Looks up the
    /// `JitCallTarget` at `descrs[fn_ptr_idx]`, materializes
    /// `concrete_ptr` in the int constants pool, derives a `BhCallDescr`
    /// from `arg_regs` kinds (result `Void`, default `EffectInfo`), and
    /// records the pyre-only `(calldescr_idx → JitCallTarget)` bridge for
    /// trace-time pointer selection.
    ///
    /// The 7 production emit sites referenced in
    /// ()
    /// route through this adapter so callers do not have to thread
    /// `concrete_ptr` and `BhCallDescr` separately.
    /// `call.py getcalldescr`'s per-callee `extraeffect` for the
    /// target at `fn_ptr_idx`.
    ///
    /// The producer already stamped its classification onto the
    /// `JitCallTarget` (`add_call_target_with_save_err` /
    /// `add_fn_ptr_with_slot`). `cond_call` / `record_known_result` read
    /// that slot here. An unanalyzed `residual_call_*` does not: see
    /// [`Self::residual_effect_info_for_target`]. A `fn_ptr_idx` that is
    /// not a call target is left to the `_with_effect_info` sibling's
    /// own panic.
    fn effect_info_for_target(&self, fn_ptr_idx: u16) -> majit_ir::descr::EffectInfo {
        match self.descrs.get(fn_ptr_idx as usize) {
            Some(RuntimeBhDescr::Call(target)) => {
                crate::call_descr::effect_info_for_slot(target.effect_info_slot)
            }
            _ => crate::call_descr::default_effect_info(),
        }
    }

    /// Effect info for `residual_call_*_canonical_via_target`.
    ///
    /// A macro helper has no graph. `graphanalyze.py`
    /// `GraphAnalyzer.analyze` returns the top result when the callee has
    /// no `graph` (`AttributeError` on `funcobj.graph`). `CanRaise` is that
    /// top (`EffectInfo::MOST_GENERAL`). A precise slot stays on
    /// `effect_info_for_slot`. Callers that already pass an effect info
    /// (`residual_writes`, `nursery_alloc`) use the `_with_effect_info`
    /// siblings and never reach this helper. `cond_call` keeps
    /// [`Self::effect_info_for_target`]: `MOST_GENERAL` forces virtuals,
    /// which `jtransform.py _rewrite_op_cond_call` rejects.
    fn residual_effect_info_for_target(&self, fn_ptr_idx: u16) -> majit_ir::descr::EffectInfo {
        match self.descrs.get(fn_ptr_idx as usize) {
            Some(RuntimeBhDescr::Call(target)) => match target.effect_info_slot {
                crate::call_descr::EffectInfoSlot::CanRaise
                | crate::call_descr::EffectInfoSlot::Unanalyzed => {
                    crate::call_descr::default_effect_info()
                }
                slot => crate::call_descr::effect_info_for_slot(slot),
            },
            _ => crate::call_descr::default_effect_info(),
        }
    }

    pub fn residual_call_void_canonical_via_target(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
    ) {
        // pyjitpl.py do_residual_call invalidates the heapcache from
        // descriptor effects before recording the call. An unanalyzed
        // residual is `EF_RANDOM_EFFECTS` (`residual_effect_info_for_target`);
        // `check_can_raise()` holds for that row, so the walker keeps
        // emitting `GUARD_NO_EXCEPTION`.
        let effect_info = self.residual_effect_info_for_target(fn_ptr_idx);
        self.residual_call_void_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            arg_regs,
            effect_info,
        );
    }

    pub fn residual_call_void_canonical_via_target_with_effect_info(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        effect_info: majit_ir::descr::EffectInfo,
    ) {
        self.emit_canonical_call_void_via_target(
            (
                jitcode::insns::BC_RESIDUAL_CALL_R_V,
                jitcode::insns::BC_RESIDUAL_CALL_IR_V,
                jitcode::insns::BC_RESIDUAL_CALL_IRF_V,
            ),
            fn_ptr_idx,
            arg_regs,
            effect_info,
            "residual_call_void_canonical_via_target",
        );
    }

    /// Emit a void residual call whose `void_word_abi` flag retains an ignored
    /// machine-word result. `descr.py get_call_descr` gives `lltype.Void` descriptors
    /// `result_size = 0`, so it has no counterpart for a semantically void
    /// helper that physically returns a word. `pyre-jit/src/jit/codewriter.rs`
    /// (`CodeWriter::intern_call_descr_stub`) sets it for
    /// `RuntimeHelperKind::ListAppendValue`.
    pub fn residual_call_void_canonical_via_target_with_effect_info_and_word_abi(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        effect_info: majit_ir::descr::EffectInfo,
        void_word_abi: bool,
    ) {
        self.emit_canonical_call_void_via_target_with_word_abi(
            (
                jitcode::insns::BC_RESIDUAL_CALL_R_V,
                jitcode::insns::BC_RESIDUAL_CALL_IR_V,
                jitcode::insns::BC_RESIDUAL_CALL_IRF_V,
            ),
            fn_ptr_idx,
            arg_regs,
            effect_info,
            void_word_abi,
            "residual_call_void_canonical_via_target",
        );
    }

    /// Emit the canonical `residual_call_{r,ir,irf}_v` opname/argcodes byte
    /// layout.
    ///
    /// Policy-specific void calls use this same bytecode family; their
    /// behaviour is carried by `calldescr.extra_info`, matching RPython
    /// `pyjitpl.py` `do_residual_call`.
    pub fn residual_call_void_canonical_typed_args(
        &mut self,
        funcptr: i64,
        args: &[JitCallArg],
        calldescr: majit_jitcode::codewriter::jitcode::BhCallDescr,
    ) {
        let calldescr_idx = self.emit_canonical_call_void(
            (
                jitcode::insns::BC_RESIDUAL_CALL_R_V,
                jitcode::insns::BC_RESIDUAL_CALL_IR_V,
                jitcode::insns::BC_RESIDUAL_CALL_IRF_V,
            ),
            funcptr,
            args,
            calldescr,
        );
        self.call_descr_to_call_target
            .insert(calldescr_idx, JitCallTarget::from_fnaddr(funcptr));
    }

    /// Generic canonical `*_v` emission body shared by `residual_call`,
    /// `call_may_force`, `call_release_gil`, `call_loopinvariant`.
    ///
    /// `funcptr` is the raw concrete C-ABI pointer that
    /// `cpu.bh_call_v` (cranelift `compiler.rs`, dynasm
    /// `runner.rs` override) will eventually transmute into a
    /// typed `extern "C" fn(...)`. The pointer is stashed in the int
    /// constants pool — RPython `assembler.py emit_const`
    /// projects const-pool slot N into the post-regs window of
    /// `bh.registers_i` so the canonical handler at
    /// `blackhole.rs`'s `handler_residual_call_*` family can resolve `bh.registers_i[
    /// code[pos]]` uniformly. `finish()` patches the placeholder byte
    /// to `num_regs_i + funcptr_const_idx` (1-byte ceiling enforced by
    /// `patch_const_u8_refs`).
    ///
    /// `args` is bucket-sorted by kind into the int / ref / float lists
    /// the canonical handler reads in order. Sub-form auto-selection:
    ///   - any float arg → `opcode_irf_v` (`iIRFd`)
    ///   - else any int arg → `opcode_ir_v` (`iIRd`)
    ///   - else → `opcode_r_v` (`iRd`)
    ///     (R variant cannot hold int / float bytes — the handler reads only
    ///     the funcptr_reg byte + the R list + the descr.)
    fn emit_canonical_call_void(
        &mut self,
        opcodes: (u8, u8, u8),
        funcptr: i64,
        args: &[JitCallArg],
        calldescr: majit_jitcode::codewriter::jitcode::BhCallDescr,
    ) -> u16 {
        let (opcode_r_v, opcode_ir_v, opcode_irf_v) = opcodes;
        let mut int_regs: Vec<u16> = Vec::new();
        let mut ref_regs: Vec<u16> = Vec::new();
        let mut float_regs: Vec<u16> = Vec::new();
        for &arg in args {
            self.touch_call_arg(arg);
            match arg.kind {
                JitArgKind::Int => int_regs.push(arg.reg),
                JitArgKind::Ref => ref_regs.push(arg.reg),
                JitArgKind::Float => float_regs.push(arg.reg),
            }
        }

        let has_float = !float_regs.is_empty();
        let has_int = !int_regs.is_empty();
        let opcode = if has_float {
            opcode_irf_v
        } else if has_int {
            opcode_ir_v
        } else {
            opcode_r_v
        };

        let funcptr_const_idx = self.add_const_i(funcptr);
        let calldescr_idx = self.add_call_descr(calldescr);

        self.start_instr(opcode);
        // funcptr_reg: 1-byte placeholder, finish() patches to
        // `num_regs_i + funcptr_const_idx` once per-kind register counts
        // freeze (RPython `assembler.py:127-138` const-pool projection).
        let funcptr_offset = self.code.len();
        self.push_u8(0);
        self.const_patches_u8
            .push((funcptr_offset, ConstKind::Int, funcptr_const_idx));

        // IR / IRF carry the int list before the ref list.
        if has_float || has_int {
            assert!(
                int_regs.len() <= u8::MAX as usize,
                "canonical call_*_v int arg count {} overflows u8",
                int_regs.len()
            );
            self.push_u8(int_regs.len() as u8);
            for reg in int_regs {
                self.push_reg_u8(reg, "canonical call_*_v int arg");
            }
        }
        // R / IR / IRF all carry the ref list.
        assert!(
            ref_regs.len() <= u8::MAX as usize,
            "canonical call_*_v ref arg count {} overflows u8",
            ref_regs.len()
        );
        self.push_u8(ref_regs.len() as u8);
        for reg in ref_regs {
            self.push_reg_u8(reg, "canonical call_*_v ref arg");
        }
        // IRF carries the float list.
        if has_float {
            assert!(
                float_regs.len() <= u8::MAX as usize,
                "canonical call_*_v float arg count {} overflows u8",
                float_regs.len()
            );
            self.push_u8(float_regs.len() as u8);
            for reg in float_regs {
                self.push_reg_u8(reg, "canonical call_*_v float arg");
            }
        }
        // calldescr is encoded as a 2-byte runtime descrs index — read by
        // `blackhole.rs`'s `read_descr`.
        self.push_u16(calldescr_idx);
        calldescr_idx
    }

    /// Emit a canonical `residual_call_*_v` whose calldescr carries
    /// `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`.
    pub fn call_may_force_void_canonical_via_target(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
    ) {
        self.emit_canonical_call_void_via_target(
            (
                jitcode::insns::BC_RESIDUAL_CALL_R_V,
                jitcode::insns::BC_RESIDUAL_CALL_IR_V,
                jitcode::insns::BC_RESIDUAL_CALL_IRF_V,
            ),
            fn_ptr_idx,
            arg_regs,
            jitcode_may_force_effect_info(),
            "call_may_force_void_canonical_via_target",
        );
    }

    /// Emit a canonical `residual_call_*_v` whose calldescr carries
    /// `EF_LOOPINVARIANT`.
    pub fn call_loopinvariant_void_canonical_via_target(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
    ) {
        // RPython `codewriter/call.py getcalldescr`:
        //   if loopinvariant:
        //       assert not NON_VOID_ARGS, ("arguments not supported for "
        //                                  "loop-invariant function!")
        // Loop-invariant direct_call must take no non-void arguments.
        assert!(
            arg_regs.is_empty(),
            "arguments not supported for loop-invariant function!",
        );
        self.emit_canonical_call_void_via_target(
            (
                jitcode::insns::BC_RESIDUAL_CALL_R_V,
                jitcode::insns::BC_RESIDUAL_CALL_IR_V,
                jitcode::insns::BC_RESIDUAL_CALL_IRF_V,
            ),
            fn_ptr_idx,
            arg_regs,
            // RPython `effectinfo.py effectinfo_from_writeanalyze`:
            // EF_LOOPINVARIANT clears `_write_descrs_*`. Empty bitsets
            // here are intentional, not the unknown-callee fallback used
            // by may_force / release_gil.
            crate::call_descr::LOOPINVARIANT_EFFECT_INFO,
            "call_loopinvariant_void_canonical_via_target",
        );
    }

    /// Generic via_target body shared by Slices 1 and 2: resolves the
    /// `JitCallTarget` at `descrs[fn_ptr_idx]`, materializes
    /// `concrete_ptr` in the int constants pool, derives a void
    /// `BhCallDescr` from arg kinds and effect policy, and records the
    /// pyre trace/concrete pointer bridge against the emitted `d`
    /// operand.
    fn emit_canonical_call_void_via_target(
        &mut self,
        opcodes: (u8, u8, u8),
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        effect_info: majit_ir::descr::EffectInfo,
        helper_name: &'static str,
    ) {
        self.emit_canonical_call_void_via_target_with_word_abi(
            opcodes,
            fn_ptr_idx,
            arg_regs,
            effect_info,
            false,
            helper_name,
        );
    }

    fn emit_canonical_call_void_via_target_with_word_abi(
        &mut self,
        opcodes: (u8, u8, u8),
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        effect_info: majit_ir::descr::EffectInfo,
        void_word_abi: bool,
        helper_name: &'static str,
    ) {
        let target = match self.descrs.get(fn_ptr_idx as usize) {
            Some(RuntimeBhDescr::Call(target)) => *target.as_ref(),
            other => panic!(
                "{helper_name}: descrs[{fn_ptr_idx}] expected \
                 RuntimeBhDescr::Call, got {other:?}"
            ),
        };
        let concrete_ptr = target.concrete_ptr as i64;

        let arg_classes: String = arg_regs
            .iter()
            .map(|a| match a.kind {
                JitArgKind::Int => 'i',
                JitArgKind::Ref => 'r',
                JitArgKind::Float => 'f',
            })
            .collect();
        let calldescr = majit_jitcode::codewriter::jitcode::BhCallDescr::from_signature(
            arg_classes,
            majit_ir::value::Type::Void,
            effect_info,
        );
        let calldescr = if void_word_abi {
            calldescr.with_void_word_abi()
        } else {
            calldescr
        };
        let calldescr_idx =
            self.emit_canonical_call_void(opcodes, concrete_ptr, arg_regs, calldescr);
        self.call_descr_to_call_target.insert(calldescr_idx, target);
    }

    /// Sibling of `emit_canonical_call_void` for the non-void result
    /// shapes. RPython `blackhole.py bhimpl_residual_call_r_r`
    /// `bhimpl_residual_call_{r,ir,irf}_{i,r,f}`. The encoding mirrors
    /// the void form with one trailing `dst:u8` byte appended after
    /// the calldescr operand (consumed by handlers
    /// `handler_residual_call_*_{i,r,f}` in `blackhole.rs` via
    /// `code[p]`).
    ///
    /// The `residual_call_*_canonical_*` wrappers below are its first
    /// non-dormant callers.
    fn emit_canonical_call_typed(
        &mut self,
        opcodes: (u8, u8, u8),
        funcptr: i64,
        args: &[JitCallArg],
        calldescr: majit_jitcode::codewriter::jitcode::BhCallDescr,
        dst: u16,
        dst_kind: JitArgKind,
    ) -> u16 {
        let calldescr_idx = self.emit_canonical_call_void(opcodes, funcptr, args, calldescr);
        // Result-bank touch + trailing dst byte (`>i` / `>r` / `>f`
        // suffix in `wellknown_bh_insns` keys).
        match dst_kind {
            JitArgKind::Int => self.touch_reg(dst),
            JitArgKind::Ref => self.touch_ref_reg(dst),
            JitArgKind::Float => self.touch_float_reg(dst),
        }
        self.push_reg_u8(dst, "canonical residual_call_*_{i,r,f} dst");
        // RPython `assembler.py:217-219` records `_resulttypes[len(self.code)]
        // = op.result.kind` for any typed-result op.  Mirror here so the
        // canonical residual_call_*_{i,r,f} family populates the same map
        // the legacy `call_*_like` siblings already do — `MIFrame::
        // make_result_of_lastop` (`pyjitpl.py`) reads it back when
        // a caller materialises the typed result.
        self.record_resulttype(match dst_kind {
            JitArgKind::Int => 'i',
            JitArgKind::Ref => 'r',
            JitArgKind::Float => 'f',
        });
        calldescr_idx
    }

    /// Generic typed via_target body — sibling of
    /// [`Self::emit_canonical_call_void_via_target`] threading `dst` /
    /// `dst_kind` for the trailing result-bank byte.  Resolves the
    /// `JitCallTarget` at `descrs[fn_ptr_idx]`, materialises
    /// `concrete_ptr` in the int constants pool, derives a typed
    /// `BhCallDescr` from arg kinds + `result_type` + `effect_info`,
    /// and records the pyre trace/concrete pointer bridge against the
    /// emitted `d` operand.
    #[allow(dead_code)]
    #[expect(
        clippy::too_many_arguments,
        reason = "The parameter order mirrors the corresponding RPython metainterpreter routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and frame ownership"
    )]
    fn emit_canonical_call_typed_via_target(
        &mut self,
        opcodes: (u8, u8, u8),
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        result_type: majit_ir::value::Type,
        effect_info: majit_ir::descr::EffectInfo,
        dst: u16,
        dst_kind: JitArgKind,
        helper_name: &'static str,
    ) {
        let target = match self.descrs.get(fn_ptr_idx as usize) {
            Some(RuntimeBhDescr::Call(target)) => *target.as_ref(),
            other => panic!(
                "{helper_name}: descrs[{fn_ptr_idx}] expected \
                 RuntimeBhDescr::Call, got {other:?}"
            ),
        };
        let concrete_ptr = target.concrete_ptr as i64;
        let arg_classes: String = arg_regs
            .iter()
            .map(|a| match a.kind {
                JitArgKind::Int => 'i',
                JitArgKind::Ref => 'r',
                JitArgKind::Float => 'f',
            })
            .collect();
        let calldescr = majit_jitcode::codewriter::jitcode::BhCallDescr::from_signature(
            arg_classes,
            result_type,
            effect_info,
        );
        let calldescr_idx = self.emit_canonical_call_typed(
            opcodes,
            concrete_ptr,
            arg_regs,
            calldescr,
            dst,
            dst_kind,
        );
        self.call_descr_to_call_target.insert(calldescr_idx, target);
    }

    /// Int-result sibling of
    /// [`Self::residual_call_void_canonical_via_target`].
    /// `bhimpl_residual_call_{r,ir,irf}_i` (`blackhole.py bhimpl_residual_call_r_i`)
    /// dispatches via `cpu.bh_call_i` and writes the result into
    /// `bh.registers_i[dst]`.
    #[allow(dead_code)]
    pub fn residual_call_int_canonical_via_target(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        let effect_info = self.residual_effect_info_for_target(fn_ptr_idx);
        self.residual_call_int_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            arg_regs,
            dst,
            effect_info,
        );
    }

    #[allow(dead_code)]
    pub fn residual_call_int_canonical_via_target_with_effect_info(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
        effect_info: majit_ir::descr::EffectInfo,
    ) {
        self.emit_canonical_call_typed_via_target(
            (
                jitcode::insns::BC_RESIDUAL_CALL_R_I,
                jitcode::insns::BC_RESIDUAL_CALL_IR_I,
                jitcode::insns::BC_RESIDUAL_CALL_IRF_I,
            ),
            fn_ptr_idx,
            arg_regs,
            majit_ir::value::Type::Int,
            effect_info,
            dst,
            JitArgKind::Int,
            "residual_call_int_canonical_via_target",
        );
    }

    /// ref-result sibling.  `bhimpl_residual_call_{r,ir,irf}_r`
    /// (`blackhole.py bhimpl_residual_call_r_r`) dispatches via `cpu.bh_call_r`.
    #[allow(dead_code)]
    pub fn residual_call_ref_canonical_via_target(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        let effect_info = self.residual_effect_info_for_target(fn_ptr_idx);
        self.residual_call_ref_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            arg_regs,
            dst,
            effect_info,
        );
    }

    #[allow(dead_code)]
    pub fn residual_call_ref_canonical_via_target_with_effect_info(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
        effect_info: majit_ir::descr::EffectInfo,
    ) {
        self.emit_canonical_call_typed_via_target(
            (
                jitcode::insns::BC_RESIDUAL_CALL_R_R,
                jitcode::insns::BC_RESIDUAL_CALL_IR_R,
                jitcode::insns::BC_RESIDUAL_CALL_IRF_R,
            ),
            fn_ptr_idx,
            arg_regs,
            majit_ir::value::Type::Ref,
            effect_info,
            dst,
            JitArgKind::Ref,
            "residual_call_ref_canonical_via_target",
        );
    }

    /// Float-result emission body.  `bhimpl_residual_call_irf_f` is the
    /// only float-result variant per `resoperation.py call_release_gil_for_descr`, so
    /// the opcode is fixed — but the handler at
    /// `handler_residual_call_irf_f` (`blackhole.rs`) always reads
    /// `read_list_f` after `read_list_r`, so the layout MUST emit an
    /// (empty-or-not) float list even when the call has no float args.
    /// This differs from `emit_canonical_call_void`, which skips the
    /// float-list bytes whenever `has_float == false` (a valid
    /// optimisation for the `R_V` / `IR_V` variants but illegal for
    /// `IRF_F`).
    fn emit_canonical_call_typed_irf_f(
        &mut self,
        funcptr: i64,
        args: &[JitCallArg],
        calldescr: majit_jitcode::codewriter::jitcode::BhCallDescr,
        dst: u16,
    ) -> u16 {
        let mut int_regs: Vec<u16> = Vec::new();
        let mut ref_regs: Vec<u16> = Vec::new();
        let mut float_regs: Vec<u16> = Vec::new();
        for &arg in args {
            self.touch_call_arg(arg);
            match arg.kind {
                JitArgKind::Int => int_regs.push(arg.reg),
                JitArgKind::Ref => ref_regs.push(arg.reg),
                JitArgKind::Float => float_regs.push(arg.reg),
            }
        }
        let funcptr_const_idx = self.add_const_i(funcptr);
        let calldescr_idx = self.add_call_descr(calldescr);

        self.start_instr(jitcode::insns::BC_RESIDUAL_CALL_IRF_F);
        let funcptr_offset = self.code.len();
        self.push_u8(0);
        self.const_patches_u8
            .push((funcptr_offset, ConstKind::Int, funcptr_const_idx));

        // IRF mandates all three (count, regs) pairs, even if a list
        // is empty.  Layout matches `read_list_i` / `_r` / `_f` in the
        // canonical handler.
        for (regs, label) in [
            (&int_regs, "canonical residual_call_irf_f int arg"),
            (&ref_regs, "canonical residual_call_irf_f ref arg"),
            (&float_regs, "canonical residual_call_irf_f float arg"),
        ] {
            assert!(
                regs.len() <= u8::MAX as usize,
                "canonical residual_call_irf_f arg count {} overflows u8",
                regs.len()
            );
            self.push_u8(regs.len() as u8);
            for &reg in regs {
                self.push_reg_u8(reg, label);
            }
        }
        self.push_u16(calldescr_idx);
        self.touch_float_reg(dst);
        self.push_reg_u8(dst, "canonical residual_call_irf_f dst");
        // RPython `assembler.py:217-219` records `_resulttypes[len(self.code)]
        // = op.result.kind` for any typed-result op.  Sibling
        // `emit_canonical_call_typed` (line 1951) already records this for
        // `_i` / `_r` / `_f` (non-IRF) result variants — float-on-IRF must
        // do the same so `MIFrame::make_result_of_lastop`
        // (`pyjitpl.py`) reads the correct kind back.
        self.record_resulttype('f');
        calldescr_idx
    }

    /// float-result sibling.  Always uses `IRF_F` per
    /// `resoperation.py call_release_gil_for_descr` ("no such thing" `R_F` / `IR_F`)
    /// and goes through `Self::emit_canonical_call_typed_irf_f` so
    /// the F list count byte is always present.
    #[allow(dead_code)]
    pub fn residual_call_float_canonical_via_target(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        let effect_info = self.residual_effect_info_for_target(fn_ptr_idx);
        self.residual_call_float_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            arg_regs,
            dst,
            effect_info,
        );
    }

    #[allow(dead_code)]
    pub fn residual_call_float_canonical_via_target_with_effect_info(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
        effect_info: majit_ir::descr::EffectInfo,
    ) {
        let target = match self.descrs.get(fn_ptr_idx as usize) {
            Some(RuntimeBhDescr::Call(target)) => *target.as_ref(),
            other => panic!(
                "residual_call_float_canonical_via_target: descrs[{fn_ptr_idx}] \
                 expected RuntimeBhDescr::Call, got {other:?}"
            ),
        };
        // The funcbox baked below is read back by `executor.rs`'s `Type::Float`
        // arm through an `extern "C" fn(..) -> f64` ABI, i.e. out of the
        // floating-point return register.  That is right for a raw callee
        // address — `add_fn_ptr(ptr)` is `add_call_target(ptr, ptr)` — and
        // wrong for a `_concrete` wrapper, which the helper policy attributes
        // give an `-> i64` signature carrying `f64::to_bits`
        // (`emit_helper_call_target_fn`, `majit-macros/src/lib.rs`).
        // Only the `*_float_wrapped` call policies mint that divergence, and
        // no crate declares one; assert the invariant here so the first declaration
        // trips a build rather than silently stamping a float read out of the
        // integer return register.
        debug_assert_eq!(
            target.trace_ptr, target.concrete_ptr,
            "float residual call bakes concrete_ptr as its funcbox; a distinct \
             concrete target is the i64-packing `_concrete` wrapper and would be \
             read from the wrong return register"
        );
        let concrete_ptr = target.concrete_ptr as i64;
        let arg_classes: String = arg_regs
            .iter()
            .map(|a| match a.kind {
                JitArgKind::Int => 'i',
                JitArgKind::Ref => 'r',
                JitArgKind::Float => 'f',
            })
            .collect();
        let calldescr = majit_jitcode::codewriter::jitcode::BhCallDescr::from_signature(
            arg_classes,
            majit_ir::value::Type::Float,
            effect_info,
        );
        let calldescr_idx =
            self.emit_canonical_call_typed_irf_f(concrete_ptr, arg_regs, calldescr, dst);
        self.call_descr_to_call_target.insert(calldescr_idx, target);
    }

    // ── typed policy variants (dormant) ──
    //
    // Mirrors the void-family policy wrappers at lines 1772-1878.  Each
    // call_<policy>_<result>_canonical_via_target seeds the appropriate
    // EffectInfo and routes through the typed `_with_effect_info` body
    // (which selects the right `R_X / IR_X / IRF_X` opcode triple per
    // result kind).  RPython has no separate bytecode for these
    // policies — the policy is carried by `calldescr.extra_info`,
    // matching `pyjitpl.py do_residual_call`.

    /// Emit a canonical `residual_call_*_i` whose calldescr carries
    /// `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`.
    #[allow(dead_code)]
    pub fn call_may_force_int_canonical_via_target(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.residual_call_int_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            arg_regs,
            dst,
            jitcode_may_force_effect_info(),
        );
    }

    /// Emit a canonical `residual_call_*_i` whose calldescr carries
    /// `EF_LOOPINVARIANT`.  RPython `codewriter/call.py:249-251
    /// getcalldescr` rejects loop-invariant calls with non-void args.
    #[allow(dead_code)]
    pub fn call_loopinvariant_int_canonical_via_target(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        assert!(
            arg_regs.is_empty(),
            "arguments not supported for loop-invariant function!",
        );
        self.residual_call_int_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            arg_regs,
            dst,
            majit_ir::descr::EffectInfo {
                extraeffect: majit_ir::descr::ExtraEffect::LoopInvariant,
                ..majit_ir::descr::EffectInfo::default()
            },
        );
    }

    /// Ref-result sibling of [`Self::call_may_force_int_canonical_via_target`].
    #[allow(dead_code)]
    pub fn call_may_force_ref_canonical_via_target(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.residual_call_ref_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            arg_regs,
            dst,
            jitcode_may_force_effect_info(),
        );
    }

    /// Ref-result sibling of [`Self::call_loopinvariant_int_canonical_via_target`].
    #[allow(dead_code)]
    pub fn call_loopinvariant_ref_canonical_via_target(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        assert!(
            arg_regs.is_empty(),
            "arguments not supported for loop-invariant function!",
        );
        self.residual_call_ref_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            arg_regs,
            dst,
            majit_ir::descr::EffectInfo {
                extraeffect: majit_ir::descr::ExtraEffect::LoopInvariant,
                ..majit_ir::descr::EffectInfo::default()
            },
        );
    }

    /// Float-result sibling of [`Self::call_may_force_int_canonical_via_target`].
    #[allow(dead_code)]
    pub fn call_may_force_float_canonical_via_target(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.residual_call_float_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            arg_regs,
            dst,
            jitcode_may_force_effect_info(),
        );
    }

    /// Float-result sibling of [`Self::call_loopinvariant_int_canonical_via_target`].
    #[allow(dead_code)]
    pub fn call_loopinvariant_float_canonical_via_target(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        assert!(
            arg_regs.is_empty(),
            "arguments not supported for loop-invariant function!",
        );
        self.residual_call_float_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            arg_regs,
            dst,
            majit_ir::descr::EffectInfo {
                extraeffect: majit_ir::descr::ExtraEffect::LoopInvariant,
                ..majit_ir::descr::EffectInfo::default()
            },
        );
    }

    /// low-level int-result direct entry — sibling of
    /// [`Self::residual_call_void_canonical_typed_args`].  Skips the
    /// `JitCallTarget` resolution and registers a self-bridging
    /// `(funcptr, funcptr)` pair (mirrors void at line 1645).
    #[allow(dead_code)]
    pub fn residual_call_int_canonical_typed_args(
        &mut self,
        funcptr: i64,
        args: &[JitCallArg],
        calldescr: majit_jitcode::codewriter::jitcode::BhCallDescr,
        dst: u16,
    ) {
        let calldescr_idx = self.emit_canonical_call_typed(
            (
                jitcode::insns::BC_RESIDUAL_CALL_R_I,
                jitcode::insns::BC_RESIDUAL_CALL_IR_I,
                jitcode::insns::BC_RESIDUAL_CALL_IRF_I,
            ),
            funcptr,
            args,
            calldescr,
            dst,
            JitArgKind::Int,
        );
        self.call_descr_to_call_target
            .insert(calldescr_idx, JitCallTarget::from_fnaddr(funcptr));
    }

    #[allow(dead_code)]
    pub fn residual_call_ref_canonical_typed_args(
        &mut self,
        funcptr: i64,
        args: &[JitCallArg],
        calldescr: majit_jitcode::codewriter::jitcode::BhCallDescr,
        dst: u16,
    ) {
        let calldescr_idx = self.emit_canonical_call_typed(
            (
                jitcode::insns::BC_RESIDUAL_CALL_R_R,
                jitcode::insns::BC_RESIDUAL_CALL_IR_R,
                jitcode::insns::BC_RESIDUAL_CALL_IRF_R,
            ),
            funcptr,
            args,
            calldescr,
            dst,
            JitArgKind::Ref,
        );
        self.call_descr_to_call_target
            .insert(calldescr_idx, JitCallTarget::from_fnaddr(funcptr));
    }

    #[allow(dead_code)]
    pub fn residual_call_float_canonical_typed_args(
        &mut self,
        funcptr: i64,
        args: &[JitCallArg],
        calldescr: majit_jitcode::codewriter::jitcode::BhCallDescr,
        dst: u16,
    ) {
        let calldescr_idx = self.emit_canonical_call_typed_irf_f(funcptr, args, calldescr, dst);
        self.call_descr_to_call_target
            .insert(calldescr_idx, JitCallTarget::from_fnaddr(funcptr));
    }

    // `CALL_ASSEMBLER` is a trace IR op (`history.rs`, pyjitpl
    // `do_recursive_call(assembler_call=True)`). Jitcode emits
    // `recursive_call_*` (`jtransform.handle_recursive_call`).

    /// Pure sibling of
    /// [`Self::residual_call_int_canonical_via_target`].  Emits the
    /// canonical `BC_RESIDUAL_CALL_{R,IR,IRF}_I` opcode with the
    /// calldescr's `extra_info` set to `ELIDABLE_CAN_RAISE`
    /// (`call_descr::ELIDABLE_EFFECT_INFO`).  The canonical walker
    /// (`majit-metainterp/src/pyjitpl/dispatch.rs`) reads
    /// `effectinfo.check_is_elidable()` and routes the result through
    /// `record_result_of_call_pure` mirroring `pyjitpl.py`,
    /// retiring the legacy `BC_CALL_PURE_INT`-specific code path.
    ///
    /// `_can_raise` is the conservative default — emits a trailing
    /// `GUARD_NO_EXCEPTION` because `effectinfo.check_can_raise(False)`
    /// is true for `extraeffect ≥ 3`. Callers that have classified
    /// the callee per `call.py _canraise(op)` should prefer
    /// [`Self::call_pure_int_canonical_via_target_cannot_raise`] (no
    /// guard) or
    /// [`Self::call_pure_int_canonical_via_target_or_memerror`] (guard
    /// retained, distinguished metadata).
    pub fn call_pure_int_canonical_via_target(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.residual_call_int_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            arg_regs,
            dst,
            crate::call_descr::ELIDABLE_EFFECT_INFO,
        );
    }

    /// `EF_ELIDABLE_CANNOT_RAISE` sibling — `call.py getcalldescr`'s
    /// `else` branch (`_canraise(op) == False`). The calldescr's
    /// `extraeffect == 0` makes `effectinfo.check_can_raise(False)`
    /// false, so the canonical walker records `CALL_PURE_*` *without*
    /// the trailing `GUARD_NO_EXCEPTION` (`pyjitpl.py:2126`).
    pub fn call_pure_int_canonical_via_target_cannot_raise(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.residual_call_int_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            arg_regs,
            dst,
            crate::call_descr::ELIDABLE_CANNOT_RAISE_EFFECT_INFO,
        );
    }

    /// `EF_ELIDABLE_OR_MEMORYERROR` sibling — `call.py getcalldescr`'s
    /// `cr == "mem"` branch. Same dispatch as `_can_raise` (`extraeffect
    /// == 3` clears `check_can_raise(False)`'s gate at the boundary)
    /// but distinguishes memory-only failure modes for the optimizer.
    pub fn call_pure_int_canonical_via_target_or_memerror(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.residual_call_int_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            arg_regs,
            dst,
            crate::call_descr::ELIDABLE_OR_MEMERROR_EFFECT_INFO,
        );
    }

    // ── conditional_call / record_known_result (jtransform.py _rewrite_op_cond_call, 292-313) ──
    //
    // `rewrite_call(..., force_ir=True)` always emits `iiIRd` / `riIRd`
    // (`condition`/`value` + funcptr + I list + R list + calldescr). The
    // helper-side `fn_ptr_idx` is resolved here the same way
    // `residual_call_*_canonical_via_target` resolves it: concrete_ptr
    // goes in the int constants pool, and the `d` operand is a
    // `BhCallDescr`.

    /// RPython: `conditional_call_ir_v(condition, funcptr, calldescr, [i], [r])`
    pub fn conditional_call_ir_v_typed_args(
        &mut self,
        fn_ptr_idx: u16,
        cond_reg: u16,
        typed_args: &[JitCallArg],
    ) {
        self.emit_canonical_cond_or_record(
            "conditional_call_ir_v/iiIRd",
            LeadReg::Int(cond_reg),
            fn_ptr_idx,
            typed_args,
            majit_ir::value::Type::Void,
            None,
        );
    }

    /// RPython: `conditional_call_value_ir_i(value, funcptr, calldescr, [i], [r])`
    pub fn conditional_call_value_ir_i_typed_args(
        &mut self,
        fn_ptr_idx: u16,
        value_reg: u16,
        typed_args: &[JitCallArg],
        dst: u16,
    ) {
        self.emit_canonical_cond_or_record(
            "conditional_call_value_ir_i/iiIRd>i",
            LeadReg::Int(value_reg),
            fn_ptr_idx,
            typed_args,
            majit_ir::value::Type::Int,
            Some((dst, 'i')),
        );
    }

    /// RPython: `conditional_call_value_ir_r`
    pub fn conditional_call_value_ir_r_typed_args(
        &mut self,
        fn_ptr_idx: u16,
        value_reg: u16,
        typed_args: &[JitCallArg],
        dst: u16,
    ) {
        self.emit_canonical_cond_or_record(
            "conditional_call_value_ir_r/riIRd>r",
            LeadReg::Ref(value_reg),
            fn_ptr_idx,
            typed_args,
            majit_ir::value::Type::Ref,
            Some((dst, 'r')),
        );
    }

    /// RPython: `record_known_result_i_ir_v(result, funcptr, calldescr, [i], [r])`
    pub fn record_known_result_i_ir_v_typed_args(
        &mut self,
        fn_ptr_idx: u16,
        result_reg: u16,
        typed_args: &[JitCallArg],
    ) {
        self.emit_canonical_cond_or_record(
            "record_known_result_i_ir_v/iiIRd",
            LeadReg::Int(result_reg),
            fn_ptr_idx,
            typed_args,
            majit_ir::value::Type::Int,
            None,
        );
    }

    /// RPython: `record_known_result_r_ir_v`
    pub fn record_known_result_r_ir_v_typed_args(
        &mut self,
        fn_ptr_idx: u16,
        result_reg: u16,
        typed_args: &[JitCallArg],
    ) {
        self.emit_canonical_cond_or_record(
            "record_known_result_r_ir_v/riIRd",
            LeadReg::Ref(result_reg),
            fn_ptr_idx,
            typed_args,
            majit_ir::value::Type::Ref,
            None,
        );
    }

    fn emit_canonical_cond_or_record(
        &mut self,
        key: &'static str,
        lead: LeadReg,
        fn_ptr_idx: u16,
        typed_args: &[JitCallArg],
        result_type: majit_ir::value::Type,
        dst: Option<(u16, char)>,
    ) {
        let target = match self.descrs.get(fn_ptr_idx as usize) {
            Some(RuntimeBhDescr::Call(target)) => *target.as_ref(),
            other => {
                panic!("{key}: descrs[{fn_ptr_idx}] expected RuntimeBhDescr::Call, got {other:?}")
            }
        };
        assert!(
            typed_args.iter().all(|arg| arg.kind != JitArgKind::Float),
            "{key}: jtransform._rewrite_op_cond_call / record_known_result \
             reject float arguments"
        );
        match lead {
            LeadReg::Int(reg) => self.touch_reg(reg),
            LeadReg::Ref(reg) => self.touch_ref_reg(reg),
        }
        let mut int_regs = Vec::new();
        let mut ref_regs = Vec::new();
        let mut arg_classes = String::new();
        for &arg in typed_args {
            self.touch_call_arg(arg);
            match arg.kind {
                JitArgKind::Int => {
                    int_regs.push(arg.reg);
                    arg_classes.push('i');
                }
                JitArgKind::Ref => {
                    ref_regs.push(arg.reg);
                    arg_classes.push('r');
                }
                JitArgKind::Float => unreachable!("float args rejected above"),
            }
        }
        if let Some((dst, _)) = dst {
            match result_type {
                majit_ir::value::Type::Int => self.touch_reg(dst),
                majit_ir::value::Type::Ref => self.touch_ref_reg(dst),
                majit_ir::value::Type::Float | majit_ir::value::Type::Void => {
                    panic!("{key}: dst requires an int or ref result")
                }
            }
        }

        let effect_info = self.effect_info_for_target(fn_ptr_idx);
        let calldescr = majit_jitcode::codewriter::jitcode::BhCallDescr::from_signature(
            arg_classes,
            result_type,
            effect_info,
        );
        let funcptr_const_idx = self.add_const_i(target.concrete_ptr as i64);
        let calldescr_idx = self.add_call_descr(calldescr);

        self.write_insn(key);
        match lead {
            LeadReg::Int(reg) => self.push_reg_u8(reg, "cond/record lead int"),
            LeadReg::Ref(reg) => self.push_reg_u8(reg, "cond/record lead ref"),
        }
        let funcptr_offset = self.code.len();
        self.push_u8(0);
        self.const_patches_u8
            .push((funcptr_offset, ConstKind::Int, funcptr_const_idx));
        assert!(
            int_regs.len() <= u8::MAX as usize,
            "{key}: int arg count {} overflows u8",
            int_regs.len()
        );
        self.push_u8(int_regs.len() as u8);
        for reg in int_regs {
            self.push_reg_u8(reg, "cond/record int arg");
        }
        assert!(
            ref_regs.len() <= u8::MAX as usize,
            "{key}: ref arg count {} overflows u8",
            ref_regs.len()
        );
        self.push_u8(ref_regs.len() as u8);
        for reg in ref_regs {
            self.push_reg_u8(reg, "cond/record ref arg");
        }
        self.push_u16(calldescr_idx);
        if let Some((dst, kind)) = dst {
            self.push_reg_u8(dst, "cond/record result");
            self.record_resulttype(kind);
        }
        self.call_descr_to_call_target.insert(calldescr_idx, target);
    }

    /// RPython `blackhole.py` `bhimpl_int_copy(a) returns=i`.
    /// Byte layout follows `assembler.py`: each `Register` is
    /// emitted in argcode order, so `int_copy/i>i` stores `[src][dst]`
    /// and the `>i` result byte is the last operand.
    pub fn move_i(&mut self, dst: u16, src: u16) {
        self.touch_reg(dst);
        self.touch_reg(src);
        self.write_insn("int_copy/i>i");
        self.push_u8(src as u8);
        self.push_u8(dst as u8);
    }

    /// `x as usize` / `x as isize` on the word-sized int bank.
    /// `jtransform.py` `rewrite_op_cast_int_to_uint` /
    /// `rewrite_op_cast_uint_to_int` rename on a 64-bit word, so only a
    /// distinct target costs a copy. A 32-bit word narrows as
    /// `_int_to_int_cast` does: `int_signext` to 4 bytes, or `int_and` with
    /// the 32-bit mask.
    pub fn cast_int_to_word(&mut self, dst: u16, src: u16, signed: bool) {
        if std::mem::size_of::<usize>() >= 8 {
            if dst != src {
                self.move_i(dst, src);
            }
        } else if signed {
            self.record_binop_i_const(dst, OpCode::IntSignext, src, 4);
        } else {
            self.record_binop_i_const(dst, OpCode::IntAnd, src, 0xFFFF_FFFF);
        }
    }

    /// `flatten.py` `self.emitline('int_push', v)` / `blackhole.py bhimpl_int_push`
    /// `bhimpl_int_push(a)` — save `src` into the int-kind scratch slot.
    pub fn push_i(&mut self, src: u16) {
        self.touch_reg(src);
        self.write_insn("int_push/i");
        self.push_u8(src as u8);
    }

    /// `flatten.py` `self.emitline('int_pop', "->", w)` / `blackhole.py bhimpl_int_pop`
    /// `bhimpl_int_pop()` — load `dst` from the int-kind scratch slot.
    pub fn pop_i(&mut self, dst: u16) {
        self.touch_reg(dst);
        self.write_insn("int_pop/>i");
        self.push_u8(dst as u8);
    }

    pub fn ensure_i_regs(&mut self, count: u16) {
        self.num_regs_i = max(self.num_regs_i, count);
    }

    pub fn ensure_r_regs(&mut self, count: u16) {
        self.num_regs_r = max(self.num_regs_r, count);
    }

    pub fn ensure_f_regs(&mut self, count: u16) {
        self.num_regs_f = max(self.num_regs_f, count);
    }

    /// Lock `num_regs_{i,r,f}` at their current values so subsequent
    /// `touch_*_reg` calls cannot grow them. Callers that route
    /// `Const*` args through the constants pool depend on this — see
    /// the `num_regs_frozen` field comment for the invariant.
    pub fn freeze_num_regs(&mut self) {
        self.num_regs_frozen = true;
    }

    pub fn load_const_r_value(&mut self, dst: u16, value: i64) {
        let const_idx = self.add_const_r(value);
        self.load_const_r(dst, const_idx);
    }

    /// Lower to RPython `ref_copy/r>r` reading from the constants
    /// window of the ref register file. See `load_const_i` for the
    /// const-patch mechanism.
    pub fn load_const_r(&mut self, dst: u16, const_idx: u16) {
        self.touch_ref_reg(dst);
        self.write_insn("ref_copy/r>r");
        let src_offset = self.code.len();
        self.push_u8(0);
        self.const_patches_u8
            .push((src_offset, ConstKind::Ref, const_idx));
        self.push_u8(dst as u8);
    }

    /// RPython `blackhole.py` `bhimpl_ref_copy(a) returns=r`.
    /// See `move_i` for the `[src][dst]` argcode-order layout.
    pub fn move_r(&mut self, dst: u16, src: u16) {
        self.touch_ref_reg(dst);
        self.touch_ref_reg(src);
        self.write_insn("ref_copy/r>r");
        self.push_u8(src as u8);
        self.push_u8(dst as u8);
    }

    /// `flatten.py` `self.emitline('ref_push', v)` / `blackhole.py bhimpl_ref_push`
    /// `bhimpl_ref_push(a)` — save `src` into the ref-kind scratch slot.
    pub fn push_r(&mut self, src: u16) {
        self.touch_ref_reg(src);
        self.write_insn("ref_push/r");
        self.push_u8(src as u8);
    }

    /// `flatten.py` `self.emitline('ref_pop', "->", w)` / `blackhole.py bhimpl_ref_pop`
    /// `bhimpl_ref_pop()` — load `dst` from the ref-kind scratch slot.
    pub fn pop_r(&mut self, dst: u16) {
        self.touch_ref_reg(dst);
        self.write_insn("ref_pop/>r");
        self.push_u8(dst as u8);
    }

    /// See
    /// [`Self::call_pure_int_canonical_via_target`] for the rationale.
    pub fn call_pure_ref_canonical_via_target(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.residual_call_ref_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            arg_regs,
            dst,
            crate::call_descr::ELIDABLE_EFFECT_INFO,
        );
    }

    /// `EF_ELIDABLE_CANNOT_RAISE` sibling — see
    /// [`Self::call_pure_int_canonical_via_target_cannot_raise`].
    pub fn call_pure_ref_canonical_via_target_cannot_raise(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.residual_call_ref_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            arg_regs,
            dst,
            crate::call_descr::ELIDABLE_CANNOT_RAISE_EFFECT_INFO,
        );
    }

    /// `EF_ELIDABLE_OR_MEMORYERROR` sibling — see
    /// [`Self::call_pure_int_canonical_via_target_or_memerror`].
    pub fn call_pure_ref_canonical_via_target_or_memerror(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.residual_call_ref_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            arg_regs,
            dst,
            crate::call_descr::ELIDABLE_OR_MEMERROR_EFFECT_INFO,
        );
    }

    // call_release_gil_ref / _typed intentionally absent:
    // resoperation.py (`# no such thing`) excludes
    // CALL_RELEASE_GIL_R, so emitting BC_CALL_RELEASE_GIL_REF would
    // record an IR op the optimizer/backend cannot consume.

    pub fn load_const_f_value(&mut self, dst: u16, value: i64) {
        let const_idx = self.add_const_f(value);
        self.load_const_f(dst, const_idx);
    }

    /// Lower to RPython `float_copy/f>f` reading from the constants
    /// window of the float register file. See `load_const_i` for the
    /// const-patch mechanism.
    pub fn load_const_f(&mut self, dst: u16, const_idx: u16) {
        self.touch_float_reg(dst);
        self.write_insn("float_copy/f>f");
        let src_offset = self.code.len();
        self.push_u8(0);
        self.const_patches_u8
            .push((src_offset, ConstKind::Float, const_idx));
        self.push_u8(dst as u8);
    }

    /// RPython `blackhole.py` `bhimpl_float_copy(a) returns=f`.
    /// See `move_i` for the `[src][dst]` argcode-order layout.
    pub fn move_f(&mut self, dst: u16, src: u16) {
        self.touch_float_reg(dst);
        self.touch_float_reg(src);
        self.write_insn("float_copy/f>f");
        self.push_u8(src as u8);
        self.push_u8(dst as u8);
    }

    /// `flatten.py` `self.emitline('float_push', v)` / `blackhole.py bhimpl_float_push`
    /// `bhimpl_float_push(a)` — save `src` into the float-kind scratch slot.
    pub fn push_f(&mut self, src: u16) {
        self.touch_float_reg(src);
        self.write_insn("float_push/f");
        self.push_u8(src as u8);
    }

    /// `flatten.py` `self.emitline('float_pop', "->", w)` / `blackhole.py bhimpl_float_pop`
    /// `bhimpl_float_pop()` — load `dst` from the float-kind scratch slot.
    pub fn pop_f(&mut self, dst: u16) {
        self.touch_float_reg(dst);
        self.write_insn("float_pop/>f");
        self.push_u8(dst as u8);
    }

    /// See
    /// [`Self::call_pure_int_canonical_via_target`] for the rationale.
    pub fn call_pure_float_canonical_via_target(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.residual_call_float_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            arg_regs,
            dst,
            crate::call_descr::ELIDABLE_EFFECT_INFO,
        );
    }

    /// `EF_ELIDABLE_CANNOT_RAISE` sibling — see
    /// [`Self::call_pure_int_canonical_via_target_cannot_raise`].
    pub fn call_pure_float_canonical_via_target_cannot_raise(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.residual_call_float_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            arg_regs,
            dst,
            crate::call_descr::ELIDABLE_CANNOT_RAISE_EFFECT_INFO,
        );
    }

    /// `EF_ELIDABLE_OR_MEMORYERROR` sibling — see
    /// [`Self::call_pure_int_canonical_via_target_or_memerror`].
    pub fn call_pure_float_canonical_via_target_or_memerror(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.residual_call_float_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            arg_regs,
            dst,
            crate::call_descr::ELIDABLE_OR_MEMERROR_EFFECT_INFO,
        );
    }

    /// RPython `blackhole.py bhimpl_float_add` `bhimpl_float_{add,sub,mul,truediv}`
    /// per-opname handlers. `float_floordiv` / `float_mod` have no direct
    /// RPython `bhimpl_*` — those lower to a residual call at the
    /// codewriter layer, never reaching a jitcode bytecode.
    pub fn record_binop_f(&mut self, dst: u16, opcode: OpCode, lhs: u16, rhs: u16) {
        let key = match opcode {
            OpCode::FloatAdd => "float_add/ff>f",
            OpCode::FloatSub => "float_sub/ff>f",
            OpCode::FloatMul => "float_mul/ff>f",
            OpCode::FloatTrueDiv => "float_truediv/ff>f",
            other => panic!("record_binop_f: unsupported opcode {other:?}"),
        };
        self.touch_float_reg(dst);
        self.touch_float_reg(lhs);
        self.touch_float_reg(rhs);
        self.write_insn(key);
        // `assembler.py:165-174` argcode `ff>f` byte order `[lhs][rhs][dst]`,
        // matching the canonical `bhhandler_ff_f!` decoder.
        self.push_u8(lhs as u8);
        self.push_u8(rhs as u8);
        self.push_u8(dst as u8);
    }

    /// Float comparison producing a boolean int result — `float_lt/ff>i`
    /// etc. The two operands are read from the float bank; the `0`/`1`
    /// result lands in the int bank (`bhhandler_ff_i!` decode order
    /// `[lhs][rhs][dst]`). `optimize_goto_if_not` only fuses a float
    /// compare into a guard when it feeds a branch directly; a compare
    /// whose result is consumed as a value (e.g. an `&&` operand) stays
    /// in this value form.
    pub fn record_compare_f(&mut self, dst: u16, opcode: OpCode, lhs: u16, rhs: u16) {
        let key = match opcode {
            OpCode::FloatLt => "float_lt/ff>i",
            OpCode::FloatLe => "float_le/ff>i",
            OpCode::FloatEq => "float_eq/ff>i",
            OpCode::FloatNe => "float_ne/ff>i",
            OpCode::FloatGt => "float_gt/ff>i",
            OpCode::FloatGe => "float_ge/ff>i",
            other => panic!("record_compare_f: unsupported opcode {other:?}"),
        };
        self.touch_reg(dst);
        self.touch_float_reg(lhs);
        self.touch_float_reg(rhs);
        self.write_insn(key);
        self.push_u8(lhs as u8);
        self.push_u8(rhs as u8);
        self.push_u8(dst as u8);
    }

    /// RPython `blackhole.py:689-695` `bhimpl_float_{neg,abs}` per-opname handlers.
    /// `[src][dst]` byte layout per the canonical `bhhandler_f_f!` decoder.
    pub fn record_unary_f(&mut self, dst: u16, opcode: OpCode, src: u16) {
        let key = match opcode {
            OpCode::FloatNeg => "float_neg/f>f",
            OpCode::FloatAbs => "float_abs/f>f",
            other => panic!("record_unary_f: unsupported opcode {other:?}"),
        };
        self.touch_float_reg(dst);
        self.touch_float_reg(src);
        self.write_insn(key);
        self.push_u8(src as u8);
        self.push_u8(dst as u8);
    }

    /// Widen an int-bank value to the float bank — RPython `cast_int_to_float`
    /// (`blackhole.py bhimpl_cast_int_to_float`). The `i>f` operand
    /// crosses banks: an int source, a float result. `[src][dst]` byte layout
    /// per the `bhhandler_i_f!` decoder.
    pub fn record_cast_int_to_float(&mut self, dst: u16, src: u16) {
        self.touch_float_reg(dst);
        self.touch_reg(src);
        self.write_insn("cast_int_to_float/i>f");
        self.push_u8(src as u8);
        self.push_u8(dst as u8);
    }

    /// Narrow a float-bank value to the int bank — RPython `cast_float_to_int`
    /// (`blackhole.py bhimpl_cast_float_to_int`). The inverse of
    /// [`Self::record_cast_int_to_float`]: a VALUE cast that truncates toward
    /// zero, NOT the `convert_float_bytes_to_longlong` bitcast below. Same
    /// `f>i` operand crossing — float source, int result, `[src][dst]`.
    pub fn record_cast_float_to_int(&mut self, dst: u16, src: u16) {
        self.touch_reg(dst);
        self.touch_float_reg(src);
        self.write_insn("cast_float_to_int/f>i");
        self.push_u8(src as u8);
        self.push_u8(dst as u8);
    }

    /// `cast_ptr_to_int/r>i`. Same bits, ref bank to int bank.
    pub fn record_cast_ptr_to_int(&mut self, dst: u16, src: u16) {
        self.touch_reg(dst);
        self.touch_ref_reg(src);
        self.write_insn("cast_ptr_to_int/r>i");
        self.push_u8(src as u8);
        self.push_u8(dst as u8);
    }

    /// `cast_int_to_ptr/i>r`. Same bits, int bank to ref bank.
    pub fn record_cast_int_to_ptr(&mut self, dst: u16, src: u16) {
        self.touch_ref_reg(dst);
        self.touch_reg(src);
        self.write_insn("cast_int_to_ptr/i>r");
        self.push_u8(src as u8);
        self.push_u8(dst as u8);
    }

    /// `jtransform.py` `rewrite_op_cast_uint_to_float = _do_builtin_call`.
    pub fn record_cast_uint_to_float(&mut self, dst: u16, src: u16) {
        let fn_ptr_idx = self.add_call_target(
            crate::blackhole::_ll_1_cast_uint_to_float as *const (),
            crate::blackhole::_ll_1_cast_uint_to_float as *const (),
        );
        self.residual_call_float_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            &[JitCallArg::int(src)],
            dst,
            crate::call_descr::cannot_raise_effect_info(),
        );
    }

    /// `jtransform.py` `rewrite_op_cast_float_to_uint = _do_builtin_call`.
    pub fn record_cast_float_to_uint(&mut self, dst: u16, src: u16) {
        let fn_ptr_idx = self.add_call_target(
            crate::blackhole::_ll_1_cast_float_to_uint as *const (),
            crate::blackhole::_ll_1_cast_float_to_uint as *const (),
        );
        self.residual_call_int_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            &[JitCallArg::float(src)],
            dst,
            crate::call_descr::cannot_raise_effect_info(),
        );
    }

    /// Reinterpret a float's 64-bit pattern as an int — RPython
    /// `convert_float_bytes_to_longlong` (`blackhole.py`). A bitcast
    /// (`f>i`), NOT a value cast: float source, int result, `[src][dst]` layout.
    pub fn record_convert_float_bytes_to_longlong(&mut self, dst: u16, src: u16) {
        self.touch_reg(dst);
        self.touch_float_reg(src);
        self.write_insn("convert_float_bytes_to_longlong/f>i");
        self.push_u8(src as u8);
        self.push_u8(dst as u8);
    }

    /// Reinterpret an int's 64-bit pattern as a float — RPython
    /// `convert_longlong_bytes_to_float` (`blackhole.py`). The inverse
    /// bitcast (`i>f`): int source, float result, `[src][dst]` layout.
    pub fn record_convert_longlong_bytes_to_float(&mut self, dst: u16, src: u16) {
        self.touch_float_reg(dst);
        self.touch_reg(src);
        self.write_insn("convert_longlong_bytes_to_float/i>f");
        self.push_u8(src as u8);
        self.push_u8(dst as u8);
    }

    /// Append a sub-JitCode descriptor and return its runtime
    /// `descrs` index. Mirrors the RPython build-time flow where
    /// `Assembler._encode_descr(jitcode)` adds the callee `JitCode` to
    /// the shared descrs list and returns the 2-byte index that
    /// `bhimpl_inline_call_*` later resolves via `self.descrs[idx]`
    /// (`blackhole.py:150-157`).
    pub fn add_sub_jitcode(&mut self, jitcode: JitCode) -> u16 {
        self.add_sub_jitcode_arc(std::sync::Arc::new(jitcode))
    }

    /// Append a descr-pool entry and return the two-byte operand index that
    /// refers to it.
    ///
    /// `assembler.py assert 0 <= num <= 0xFFFF, "too many AbstractDescrs!"`
    /// bounds the same index; upstream aborts translation, and the
    /// build-time builder does too.  This builder runs at runtime and cannot
    /// panic, so an index past the operand width latches the overflow flag
    /// and `try_finish` declines the JitCode, the same way `push_reg_u8`
    /// handles a register index that outgrows its byte.
    fn push_descr_entry(&mut self, entry: RuntimeBhDescr) -> u16 {
        let idx = self.descrs.len();
        self.descrs.push(entry);
        if idx > u16::MAX as usize {
            self.encoding_overflow = true;
            return 0;
        }
        idx as u16
    }

    fn assert_sub_jitcode_has_no_identity_slots(jitcode: &JitCode) {
        assert!(
            !jitcode.reads_identity_slots(),
            "sub-JitCode {:?} emits load/store_state_field*; only the portal JitCode may",
            jitcode.name()
        );
    }

    /// Variant accepting an already-shared `Arc<JitCode>` for callers
    /// that already hold a shared handle (e.g. a re-export from
    /// `MetaInterpStaticData::indirectcalltargets`).
    pub fn add_sub_jitcode_arc(&mut self, jitcode: std::sync::Arc<JitCode>) -> u16 {
        Self::assert_sub_jitcode_has_no_identity_slots(&jitcode);
        // `assembler.py Assembler._encode_descr` memoises every descr through
        // `_descr_dict`, and upstream's `JitCode` IS an `AbstractDescr`, so a
        // jitcode operand goes through that memo like any other.  This was the
        // one descr kind here that never got it: every other `add_*` on this
        // builder already dedups by a linear scan.  It matters now because a
        // helper that names itself twice would otherwise claim two slots for
        // one callee.
        if let Some(existing) = self.descrs.iter().position(|entry| {
            matches!(entry, RuntimeBhDescr::JitCode(arc) if std::sync::Arc::ptr_eq(arc, &jitcode))
        }) {
            return existing as u16;
        }
        self.push_descr_entry(RuntimeBhDescr::JitCode(jitcode))
    }

    /// Record whichever `j` edge an inline call site resolved.
    ///
    /// A back edge is deduped by the address it names rather than by `Arc`
    /// identity: two `Weak`s cloned from one `Arc::new_cyclic` are distinct
    /// values naming one allocation, and `Weak::ptr_eq` on an
    /// under-construction target is the only comparison available — the
    /// upgrade that `Arc::ptr_eq` would need returns `None` until assembly
    /// finishes.
    pub fn add_sub_jitcode_ref(&mut self, edge: crate::jitcode::InlineJitCodeRef) -> u16 {
        match edge {
            crate::jitcode::InlineJitCodeRef::Strong(arc) => self.add_sub_jitcode_arc(arc),
            crate::jitcode::InlineJitCodeRef::BackEdge(weak) => {
                if let Some(jc) = weak.upgrade() {
                    Self::assert_sub_jitcode_has_no_identity_slots(&jc);
                }
                let target = std::sync::Weak::as_ptr(&weak);
                if let Some(existing) = self.descrs.iter().position(|entry| {
                    matches!(entry, RuntimeBhDescr::JitCodeBackEdge(seen)
                             if std::sync::Weak::as_ptr(seen) == target)
                }) {
                    return existing as u16;
                }
                self.push_descr_entry(RuntimeBhDescr::JitCodeBackEdge(weak))
            }
        }
    }

    pub fn add_fn_ptr(&mut self, ptr: *const ()) -> u16 {
        self.add_call_target(ptr, ptr)
    }

    /// [`Self::add_fn_ptr`] for a target the producer spells in machine words.
    /// The call descr carries that signature.
    pub fn add_word_abi_fn_ptr(&mut self, ptr: *const ()) -> u16 {
        self.add_fn_ptr(ptr)
    }

    /// [`Self::add_word_abi_fn_ptr`] for a target that returns nothing.
    pub fn add_word_abi_fn_ptr_void(&mut self, ptr: *const ()) -> u16 {
        self.add_fn_ptr(ptr)
    }

    /// [`Self::add_word_abi_fn_ptr`] carrying an
    /// [`crate::call_descr::EffectInfoSlot`] classification.
    pub fn add_word_abi_fn_ptr_with_slot(
        &mut self,
        ptr: *const (),
        slot: crate::call_descr::EffectInfoSlot,
    ) -> u16 {
        self.add_fn_ptr_with_slot(ptr, slot)
    }

    /// `add_fn_ptr` variant carrying a per-callee
    /// [`crate::call_descr::EffectInfoSlot`] classification
    /// (`call.py getcalldescr`'s `extraeffect` selection).
    /// Producers that statically know the helper's
    /// `_canraise` / `_elidable_function_` / `_jit_loop_invariant_`
    /// flags pick the matching slot; the dispatcher then threads the
    /// slot into `make_call_descr_from_target_slot` so the recorded
    /// trace descr carries the right `EffectInfo`.
    pub fn add_fn_ptr_with_slot(
        &mut self,
        ptr: *const (),
        slot: crate::call_descr::EffectInfoSlot,
    ) -> u16 {
        self.add_call_target_with_slot(ptr, ptr, slot)
    }

    /// Append a function target descriptor and return its runtime
    /// `descrs` index. Mirrors RPython `Assembler._encode_descr(calldescr)`
    /// (assembler.py:140) where the 2-byte operand downstream resolves
    /// to `self.descrs[idx]` at dispatch time. pyre dedups identical
    /// `(trace_ptr, concrete_ptr)` pairs to match
    /// `Assembler._encode_descr` memoisation.
    pub fn add_call_target(&mut self, trace_ptr: *const (), concrete_ptr: *const ()) -> u16 {
        self.add_call_target_with_slot(
            trace_ptr,
            concrete_ptr,
            crate::call_descr::EffectInfoSlot::CanRaise,
        )
    }

    /// `add_call_target` variant carrying a per-callee
    /// [`crate::call_descr::EffectInfoSlot`] classification.  Dedup
    /// matches the full `(trace_ptr, concrete_ptr, slot)` triple so the
    /// same fn pointer registered with two different slots produces two
    /// distinct entries; in practice the same helper is registered with
    /// a single classification and the dedup matches the `add_call_target`
    /// path verbatim.  `save_err` defaults to `0` (`RFFI_ERR_NONE`,
    /// `rffi.py`); release-gil callees use [`Self::add_call_target_with_save_err`]
    /// to thread the wrapper's `_call_aroundstate_target_[1]`
    /// (`rffi.py`) into the dedup key.
    pub fn add_call_target_with_slot(
        &mut self,
        trace_ptr: *const (),
        concrete_ptr: *const (),
        slot: crate::call_descr::EffectInfoSlot,
    ) -> u16 {
        self.add_call_target_with_save_err(trace_ptr, concrete_ptr, slot, 0)
    }

    /// `add_call_target_with_slot` variant for release-gil callees:
    /// records the wrapper callable's
    /// `_call_aroundstate_target_ = (funcptr, save_err)` decoration
    /// (`rffi.py`).  The `(trace_ptr, concrete_ptr, slot,
    /// save_err)` tuple is the dedup key — same callee registered with
    /// two different `save_err` values produces two distinct entries
    /// because the recorded `EffectInfo.call_release_gil_target`
    /// differs.
    pub fn add_call_target_with_save_err(
        &mut self,
        trace_ptr: *const (),
        concrete_ptr: *const (),
        slot: crate::call_descr::EffectInfoSlot,
        save_err: i32,
    ) -> u16 {
        let target = JitCallTarget::with_save_err(trace_ptr, concrete_ptr, slot, save_err);
        for (idx, entry) in self.descrs.iter().enumerate() {
            if let RuntimeBhDescr::Call(existing) = entry
                && existing.as_ref() == &target
            {
                return idx as u16;
            }
        }
        self.push_descr_entry(RuntimeBhDescr::Call(Box::new(target)))
    }

    /// [`Self::add_call_target_with_save_err`] for a pair of targets the
    /// producer spells in machine words. The call descr carries that signature.
    pub fn add_word_abi_call_target_with_save_err(
        &mut self,
        trace_ptr: *const (),
        concrete_ptr: *const (),
        slot: crate::call_descr::EffectInfoSlot,
        save_err: i32,
    ) -> u16 {
        self.add_call_target_with_save_err(trace_ptr, concrete_ptr, slot, save_err)
    }

    pub fn add_call_assembler_target(
        &mut self,
        target: &std::sync::Arc<JitCellToken>,
        concrete_ptr: *const (),
    ) -> u16 {
        let target = JitCallAssemblerTarget::new(target, concrete_ptr);
        for (idx, entry) in self.descrs.iter().enumerate() {
            if let RuntimeBhDescr::AssemblerToken(existing) = entry
                && *existing == target
            {
                return idx as u16;
            }
        }
        self.push_descr_entry(RuntimeBhDescr::AssemblerToken(target))
    }

    fn add_vable_field_descr(&mut self, field_idx: u16) -> u16 {
        self.add_bh_descr(CanonicalBhDescr::VableField {
            index: field_idx as usize,
        })
    }

    fn add_vable_array_field_descr(&mut self, array_idx: u16) -> u16 {
        self.add_bh_descr(CanonicalBhDescr::VableArray {
            index: array_idx as usize,
        })
    }

    fn add_vable_array_descr(
        &mut self,
        item_type: majit_ir::value::Type,
        is_item_signed: bool,
    ) -> u16 {
        self.add_bh_descr(CanonicalBhDescr::Array {
            // `Ptr(GcArray)` after `make_sure_not_resized`: length at 0,
            // items at the payload offset. Ref items sit at the target
            // word (`FixedObjectArray`). Int/float items sit at
            // `VirtArray::<i64>::ITEMS_OFFSET` (element-aligned past the
            // length word — 8 on every target, including wasm32).
            base_size: match item_type {
                majit_ir::value::Type::Ref => std::mem::size_of::<usize>(),
                _ => crate::virt_array::VirtArray::<i64>::ITEMS_OFFSET,
            },
            // A `[ref; virt]` array's runtime twin reads its item size from
            // `pyre_object::ITEMS_BLOCK_TOKEN`
            // (`pyre-jit-trace virtualizable_gen.rs`), which is
            // `size_of::<PyObjectRef>()`; a literal 8 makes the two descr
            // universes disagree on wasm32.  A `[int; virt]` array keeps 8 —
            // the interpreter integer bank is `i64` on every target.
            itemsize: scalar_size(item_type),
            len_offset: Some(0),
            type_id: 0,
            gc_type_id: 0,
            item_type,
            is_array_of_pointers: matches!(item_type, majit_ir::value::Type::Ref),
            is_array_of_structs: false,
            is_item_signed,
            ei_index: u32::MAX,
            // vable array slots are per-vinfo with no source-level
            // ARRAY type spelling; distinct slots are disambiguated by
            // the parent `VableArray { index }` variant.
            array_type_id: None,
            interior_fields: Vec::new(),
            // Virtualizable frame array slots keep the existing
            // GUARD_GC_TYPE behavior.
            is_gc_managed: true,
        })
    }

    /// Append a `CanonicalBhDescr::Call { calldescr }` entry to the descrs
    /// pool and return its index. Used by canonical `residual_call_*` /
    /// `call_*` emit paths that need a `d` argcode descriptor (RPython
    /// `assembler.py:197-207` `_encode_descr(calldescr)`).
    ///
    /// TODO: dedup is intentionally NOT done for the
    /// `Call` variant. RPython's `descr.py:660-668 _key_for_caching`
    /// dedups calldescrs on `(arg_classes, RESULT_ERASED, ffi_flags,
    /// extrainfo)` because the funcptr lives in `op.args[0]` separately
    /// from the descr. Pyre's adapter records a sidetable
    /// `JitCodeExecState.call_descr_to_call_target` keyed by the descr
    /// slot, so dedup'ing two distinct callees that share a signature
    /// would cause the second emit's `(trace_ptr, concrete_ptr)` pair to
    /// silently overwrite the first's. The convergence path is to lift
    /// the sidetable onto the funcptr int-const slot once trace_ptr and
    /// concrete_ptr unify; until then each emit gets a fresh descr slot.
    pub fn add_call_descr(
        &mut self,
        calldescr: majit_jitcode::codewriter::jitcode::BhCallDescr,
    ) -> u16 {
        self.add_bh_descr(CanonicalBhDescr::Call { calldescr })
    }

    /// Append an array descriptor to the descrs pool and return its
    /// index (deduped).  RPython `assembler.py:197-207 _encode_descr`
    /// accepts any `AbstractDescr`; pyre keeps the pool entry points
    /// typed per variant (`add_call_descr` for calls, this for arrays)
    /// so a mis-shaped descr is rejected at emit time instead of at
    /// blackhole decode time.
    pub fn add_array_descr(&mut self, descr: CanonicalBhDescr) -> u16 {
        assert!(
            matches!(descr, CanonicalBhDescr::Array { .. }),
            "add_array_descr expects CanonicalBhDescr::Array, got {descr:?}"
        );
        self.add_bh_descr(descr)
    }

    fn add_bh_descr(&mut self, descr: CanonicalBhDescr) -> u16 {
        for (idx, entry) in self.descrs.iter().enumerate() {
            if let RuntimeBhDescr::Descr(existing) = entry
                && canonical_bh_descr_eq(existing, &descr)
            {
                return idx as u16;
            }
        }
        self.push_descr_entry(RuntimeBhDescr::Descr(Box::new(descr)))
    }

    fn add_switch_descr(&mut self, mut const_keys_in_order: Vec<i64>) -> u16 {
        const_keys_in_order.sort();
        // RPython creates one SwitchDictDescr object per switch site
        // (`flatten.py:282-285`), so do not deduplicate.
        self.push_descr_entry(RuntimeBhDescr::Descr(Box::new(CanonicalBhDescr::Switch {
            dict: std::collections::HashMap::new(),
            const_keys_in_order,
        })))
    }

    pub fn try_finish(mut self) -> Option<JitCode> {
        self.flush_pending_resulttype();
        let total_i = self.num_regs_i as usize + self.constants_i.len();
        let total_r = self.num_regs_r as usize + self.constants_r.len();
        let total_f = self.num_regs_f as usize + self.constants_f.len();
        if self.encoding_overflow
            || self.num_regs_i >= 256
            || self.num_regs_r >= 256
            || self.num_regs_f >= 256
            || total_i > 256
            || total_r > 256
            || total_f > 256
            // `assembler.py` `fix_labels`: `assert 0 <= target <= 0xFFFF`.
            // A position past that unsigned 16-bit range cannot be named as
            // a jump target or as a resume pc. Decline it like the
            // register/const ceilings above; the interpreter keeps running it.
            || self.code.len() > u16::MAX as usize
        {
            return None;
        }
        self.patch_labels();
        self.patch_switch_descrs();
        self.patch_const_refs();
        self.patch_field_descr_parents();
        if crate::jit_strict_mode()
            && let Some(disagreement) = self.field_descr_position_disagreement()
        {
            panic!("{disagreement}");
        }
        self.descrs = std::mem::take(&mut self.descrs)
            .into_iter()
            .map(RuntimeBhDescr::into_resolved)
            .collect();
        self.patch_const_u8_refs();
        // RPython `jitcode.py JitCode.setup self._resulttypes = resulttypes`.
        // Upstream `assembler.py:217-219` records the result-kind
        // char at the end-of-instruction position (`len(self.code)`
        // after operands, before the next instruction's opcode) for
        // every instruction whose argcodes contain `>X`.  Consumed
        // by `pyjitpl.py make_result_of_lastop` in non-translated
        // builds as a debug-only type check:
        //
        // ```python
        // assert typeof[self.jitcode._resulttypes[self.pc]] == got_type
        // ```
        //
        // pyre fires the same assertion in
        // `MIFrame::make_result_of_lastop` (`frame.rs`).  Each
        // typed-result emit helper (`call_*_like`,
        // `call_assembler_*_like`, `inline_call_typed`) records the
        // kind via `self.record_resulttype(...)` as its LAST step.
        // Generic `write_insn("...>X")` stores `X` in
        // `pending_resulttype` and `start_instr` / `finish` flush it
        // after the operand bytes have been pushed.  The map handed
        // off to `JitCodeBody::resulttypes` is therefore keyed by
        // every typed-result instruction's end-of-instruction PC.
        let resulttypes = Some(self.resulttypes);
        // RPython enforces two distinct ceilings:
        //   * `jitcode.py` keeps `num_regs_*` in a single char — the
        //     stored `c_num_regs_*` is a single char.
        //   * `assembler.py` encodes `count_regs[kind] +
        //     len(constants) - 1` — the last
        //     constant-slot index in the unified register-plus-const
        //     namespace must fit in one byte, so the total count
        //     `num_regs + len(constants) <= 256` (last index 255).
        // `try_finish` checks both ceilings before patching operands so an
        // unencodable source graph is declined instead of being installed.
        if crate::majit_log_enabled() {
            eprintln!(
                "[bcenc-audit] {:?} regs i={} r={} f={} consts i={} r={} f={} total i={} r={} f={}",
                self.name,
                self.num_regs_i,
                self.num_regs_r,
                self.num_regs_f,
                self.constants_i.len(),
                self.constants_r.len(),
                self.constants_f.len(),
                total_i,
                total_r,
                total_f,
            );
        }
        // `jitcode.py` ceiling: `num_regs_X < 256` per kind.
        debug_assert!(self.num_regs_i < 256 && self.num_regs_r < 256 && self.num_regs_f < 256);
        // `assembler.py` ceiling: last unified slot
        // index `num_regs + len(consts) - 1 < 256`, i.e. `total <= 256`.
        debug_assert!(total_i <= 256 && total_r <= 256 && total_f <= 256);
        let body = majit_jitcode::jitcode::JitCodeBody {
            // RPython `jitcode.py JitCode.__init__ self.calldescr = calldescr` — the
            // value was stored on the builder via `set_calldescr` (the
            // analog of upstream's constructor argument).  Without an
            // explicit stage call the field remains at the
            // `BhCallDescr::default()` zero, matching the pre-set state
            // RPython's constructor shows when `calldescr=None`.
            calldescr: self.calldescr,
            // Recorded on the body as well as on `exec` below, so a body that
            // makes the round trip through serialization still says where its
            // marker is. The two routes into `JitCodeBody` now carry the same
            // field, and `JitCode::from_canonical` reads it back.
            jit_merge_point_offset: self.jit_merge_point_offset,
            code: self.code,
            constants_i: self.constants_i,
            constants_r: self.constants_r.into_iter().map(Into::into).collect(),
            constants_f: self.constants_f,
            // This builder constructs bytecode directly and never emits
            // prebuilt-string or unit-variant ref constants, so both
            // descriptor banks are empty.
            str_consts: Vec::new(),
            unit_variant_consts: Vec::new(),
            exc_instance_consts: Vec::new(),
            type_static_consts: Vec::new(),
            reloc_consts_i: Vec::new(),
            reloc_consts_r: Vec::new(),
            empty_array_consts: Vec::new(),
            prebuilt_instance_consts: Vec::new(),
            // `jitcode.py JitCode.setup`: the three register counts are
            // stored as one `chr` apiece.  The gate above proves these casts
            // lossless; keeping the builder counters wide is useful only
            // while detecting an unencodable source body.
            c_num_regs_i: self.num_regs_i as u8,
            c_num_regs_r: self.num_regs_r as u8,
            c_num_regs_f: self.num_regs_f as u8,
            // RPython `assembler.py make_jitcode(startpoints=
            // self.startpoints, alllabels=self.alllabels, ...)` —
            // assembled jitcodes always carry the recorded set, never
            // `None`. Wrap in `Some(...)` so the upstream None sentinel
            // is reserved for hand-built helper jitcodes that bypass the
            // builder (matching `JitCode.setup(..., startpoints=None,
            // alllabels=None)` defaults at jitcode.py).
            startpoints: Some(self.startpoints),
            alllabels: Some(self.alllabels),
            resulttypes,
            _ssarepr: None,
        };
        let mut jc = JitCode::new(self.name);
        // RPython `JitCode(name, fnaddr, calldescr)` (`call.py:167-169`)
        // writes `fnaddr` at construction time before the assembler fills
        // the body. Stage it on the builder via `set_fnaddr` and stamp it
        // here so callers do not need a post-`set_body` mutation.
        // A body that reads the interpreter's own context is not the function
        // it was lowered from, and neither is one whose callee is such a body:
        // running this jitcode natively runs that callee's Rust source too.
        // `descrs` holds the sub-jitcodes this body can enter, and each was
        // finished before this one, so their verdicts are already in.
        //
        // A `Weak` that does not upgrade is this jitcode's own back edge —
        // `Arc::new_cyclic` has not published it yet — and its verdict is the
        // one being computed here, so it is not a separate refusal.
        let callee_blocked = self.descrs.iter().any(|descr| {
            descr
                .as_jitcode_owned()
                .is_some_and(|callee| callee.fnaddr == 0)
        });
        jc.fnaddr = if self.native_entry_blocked || callee_blocked {
            0
        } else {
            self.fnaddr
        };
        jc.set_body(body);
        jc.exec = super::JitCodeExecState {
            descrs: self.descrs,
            call_descr_to_call_target: self.call_descr_to_call_target,
            // Propagate the captured `BC_JIT_MERGE_POINT(_C)` opcode
            // offset so `register_dispatch_jitcode` validates the payload
            // by direct seek instead of byte-stream scan
            // (blackhole.py handler argcode-based decode parity).
            jit_merge_point_offset: self.jit_merge_point_offset,
            reads_identity_slots: self.reads_identity_slots,
        };
        // codewriter.py `jitcode.index = index` — back-stamped by
        // `state::jitcode_for` at registration time. JitCode::new
        // leaves the OnceLock unset; runtime call sites that
        // previously expected `index = 0` from the flat-struct
        // Default consume `try_index()` instead, which correctly
        // returns `None` until the back-stamp lands.
        Some(jc)
    }

    pub fn finish(self) -> JitCode {
        self.try_finish()
            .expect("JitCodeBuilder::finish called for an unencodable JitCode")
    }

    /// Latch the single-byte-operand overflow flag so `try_finish` declines
    /// the JitCode.  Used by front-ends that compute a pool-backed operand
    /// index outside the builder's `push_reg_u8` seam and would otherwise have
    /// to panic when the index exceeds a single byte.
    pub fn note_encoding_overflow(&mut self) {
        self.encoding_overflow = true;
    }

    fn push_u8(&mut self, value: u8) {
        self.code.push(value);
    }

    /// Emit a single register operand ([`jitcode::JitcodeReg`], one byte).
    /// A register index that does not fit a byte latches the overflow flag so
    /// `try_finish` declines the JitCode. See [`jitcode::JitcodeReg`] — the
    /// register width is deliberately fixed at `u8`.
    fn push_reg_u8(&mut self, reg: u16, _context: &'static str) {
        if reg > jitcode::JitcodeReg::MAX as u16 {
            // The placeholder exists only in the transient builder;
            // `try_finish` observes the latch and declines the JitCode.
            self.encoding_overflow = true;
            self.push_u8(0);
        } else {
            self.push_u8(reg as jitcode::JitcodeReg);
        }
    }

    fn push_u16(&mut self, value: u16) {
        self.code.extend_from_slice(&value.to_le_bytes());
    }

    /// RPython `assembler.py` `write_insn` opcode-byte lane:
    /// looks up `opname/argcodes` in the shared insns table and emits
    /// the assigned byte. Operand emission is still done by the
    /// surrounding method because pyre's 2-byte register operands and
    /// per-BC operand layouts do not yet match the 1-byte `emit_reg` /
    /// `emit_const` encoding on the RPython side.
    fn write_insn(&mut self, key: &'static str) {
        self.start_instr(jitcode::insn_byte(key));
        self.pending_resulttype = key
            .split_once('>')
            .and_then(|(_, suffix)| suffix.chars().next())
            .filter(|kind| matches!(kind, 'i' | 'r' | 'f'));
    }

    /// Record the current bytecode offset as an instruction start
    /// (RPython `assembler.py:200-208` writes `self.startpoints.add(pos)`
    /// just before each opcode-byte push) and emit the opcode byte.
    /// Every helper that pushes a `BC_*` opcode goes through this so
    /// `JitCode.get_live_vars_info` (RPython `jitcode.py`) can
    /// fire its non-translated `assert pc in self._startpoints` check.
    fn start_instr(&mut self, opcode: u8) {
        self.native_entry_blocked |= Self::native_entry_denied(opcode);
        self.flush_pending_resulttype();
        self.startpoints.insert(self.code.len());
        self.push_u8(opcode);
    }

    /// Whether this opcode makes the body something other than the function it
    /// was lowered from, so a native entry for it would answer differently.
    ///
    /// The list is exactly the opcodes a native entry would get WRONG — either
    /// because the operand lives in the blackhole rather than in the heap, or
    /// because the opcode carries an obligation beyond the value it returns:
    ///
    /// * the `#[jit_interp]` state fields, which live in the interpreter's
    ///   register banks between merge points and reach memory only at a
    ///   writeback, so a native call would read the pre-writeback value;
    /// * the virtualizable getters/setters, which carry an obligation the
    ///   function they were lowered from does not: every one of them begins
    ///   `fielddescr.get_vinfo().clear_vable_token(vable)`
    ///   (`blackhole.py bhimpl_getarrayitem_vable_i bhimpl_getarrayitem_vable_*`, mirrored by
    ///   `vable_clear_token_and_get_vinfo`), and `clear_vable_token`
    ///   (`virtualizable.py`) FORCES a virtualizable that still holds
    ///   a token. Both blackholes then read the heap, so the field value would
    ///   agree — it is the forcing that a native entry would skip, leaving the
    ///   token standing. (Serving them out of `virtualizable_boxes` is the
    ///   METAINTERP's behaviour, `pyjitpl.py opimpl_getfield_vable_i
    ///   opimpl_getfield_vable_*`, not the blackhole's.);
    /// * `jit_merge_point` and `call_assembler`/`recursive_call`, which do not
    ///   return to the caller but raise a JitException the C-level call in
    ///   `bhimpl_inline_call_*` propagates and a Rust return value cannot; and
    /// * `loop_header`, which is a tracing hook, not a computation.
    ///
    /// Everything else — arithmetic, control flow, `getfield`/`setfield` on
    /// real objects, residual calls, nested inline calls — computes the same
    /// answer either way, because the heap is the heap. That is the whole
    /// premise of `bhimpl_inline_call_*`.
    const fn native_entry_denied(opcode: u8) -> bool {
        use majit_jitcode::insns as i;
        matches!(
            opcode,
            i::BC_LOAD_STATE_FIELD
                | i::BC_STORE_STATE_FIELD
                | i::BC_LOAD_STATE_FIELD_REF
                | i::BC_STORE_STATE_FIELD_REF
                | i::BC_LOAD_STATE_FIELD_FLOAT
                | i::BC_STORE_STATE_FIELD_FLOAT
                | i::BC_GETFIELD_VABLE_I
                | i::BC_GETFIELD_VABLE_R
                | i::BC_GETFIELD_VABLE_F
                | i::BC_SETFIELD_VABLE_I
                | i::BC_SETFIELD_VABLE_I_IMM
                | i::BC_SETFIELD_VABLE_R
                | i::BC_SETFIELD_VABLE_F
                | i::BC_GETARRAYITEM_VABLE_I
                | i::BC_GETARRAYITEM_VABLE_R
                | i::BC_GETARRAYITEM_VABLE_F
                | i::BC_SETARRAYITEM_VABLE_I
                | i::BC_SETARRAYITEM_VABLE_R
                | i::BC_SETARRAYITEM_VABLE_F
                | i::BC_ARRAYLEN_VABLE
                | i::BC_ARRAYBASE_VABLE
                | i::BC_HINT_FORCE_VIRTUALIZABLE
                | i::BC_JIT_MERGE_POINT
                | i::BC_JIT_MERGE_POINT_C
                | i::BC_RECURSIVE_CALL_INT
                | i::BC_RECURSIVE_CALL_REF
                | i::BC_RECURSIVE_CALL_FLOAT
                | i::BC_RECURSIVE_CALL_VOID
                | i::BC_LOOP_HEADER
        )
    }

    /// RPython `assembler.py:217-219`:
    ///
    /// ```python
    /// if '>' in argcodes:
    ///     assert argcodes.index('>') == len(argcodes) - 2
    ///     self.resulttypes[len(self.code)] = argcodes[-1]
    /// ```
    ///
    /// Called by typed-result emit helpers as their LAST step (after
    /// every operand byte has been pushed) so `self.code.len()`
    /// matches the upstream `len(self.code)` after-operands-before-
    /// next-instruction value.  Consumed by
    /// `MIFrame::make_result_of_lastop` (RPython `pyjitpl.py`)
    /// where `frame.pc` has already advanced past the instruction.
    fn record_resulttype(&mut self, kind: char) {
        self.resulttypes.insert(self.code.len(), kind);
    }

    fn flush_pending_resulttype(&mut self) {
        if let Some(kind) = self.pending_resulttype.take() {
            self.record_resulttype(kind);
        }
    }

    #[allow(dead_code)]
    fn call_int_like(&mut self, opcode: u8, fn_ptr_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.touch_reg(dst);
        for &arg in arg_regs {
            self.touch_call_arg(arg);
        }
        self.start_instr(opcode);
        self.push_u16(fn_ptr_idx);
        self.push_u16(dst);
        self.push_u16(arg_regs.len() as u16);
        for &arg in arg_regs {
            self.push_u8(arg.kind.encode());
            self.push_u16(arg.reg);
        }
        self.record_resulttype('i');
    }

    fn push_label_ref(&mut self, label: u16) {
        // RPython `assembler.py:176` records the offset of the 2-byte
        // label slot in `alllabels` just before the temp bytes go in.
        let patch_offset = self.code.len();
        self.alllabels.insert(patch_offset);
        self.push_u16(0);
        self.patches.push((label as usize, patch_offset));
    }

    /// `assembler.py:72-74` `emit_reg(reg)` parity: real-register emit
    /// path.  Asserts `reg.index < count_regs[kind]` when frozen so
    /// out-of-band register indices fail-fast instead of silently
    /// truncating via `as u8` at the matching push site.  Const-pool
    /// slots route through `touch_int_reg_or_pool_slot` instead — they
    /// live in `[num_regs_kind, num_regs_kind + num_consts_kind)`,
    /// outside the real-register window.
    fn touch_reg(&mut self, reg: u16) {
        if self.num_regs_frozen {
            assert!(
                reg < self.num_regs_i,
                "int reg {reg} out of bounds {} (assembler.py:72-74 emit_reg parity)",
                self.num_regs_i
            );
            self.encoding_overflow |= reg >= 256;
            return;
        }
        assert!(
            reg < 256,
            "int reg {reg} out of u8 range (assembler.py:72 chr() parity)"
        );
        self.num_regs_i = max(self.num_regs_i, reg.saturating_add(1));
    }

    fn touch_ref_reg(&mut self, reg: u16) {
        if self.num_regs_frozen {
            assert!(
                reg < self.num_regs_r,
                "ref reg {reg} out of bounds {} (assembler.py:72-74 emit_reg parity)",
                self.num_regs_r
            );
            self.encoding_overflow |= reg >= 256;
            return;
        }
        assert!(
            reg < 256,
            "ref reg {reg} out of u8 range (assembler.py:72 chr() parity)"
        );
        self.num_regs_r = max(self.num_regs_r, reg.saturating_add(1));
    }

    fn touch_float_reg(&mut self, reg: u16) {
        if self.num_regs_frozen {
            assert!(
                reg < self.num_regs_f,
                "float reg {reg} out of bounds {} (assembler.py:72-74 emit_reg parity)",
                self.num_regs_f
            );
            self.encoding_overflow |= reg >= 256;
            return;
        }
        assert!(
            reg < 256,
            "float reg {reg} out of u8 range (assembler.py:72 chr() parity)"
        );
        self.num_regs_f = max(self.num_regs_f, reg.saturating_add(1));
    }

    /// `assembler.py` `emit_const(...)` parity: the bytecode slot
    /// for a constant operand is `count_regs[kind] + const_index`.
    /// Real-register emit (`touch_reg`) and constant emit are two
    /// distinct upstream paths; pyre fuses them at the call-arg
    /// surface because the synthesized `[num_regs_kind, num_regs_kind +
    /// num_consts_kind)` slot indices flow through `JitCallArg` from
    /// `pyre-jit/src/jit/assembler.rs::expect_*_reg_or_pool`.  This
    /// union helper validates against the combined range so const-pool
    /// slots are accepted alongside real-register indices.
    fn touch_int_reg_or_pool_slot(&mut self, reg: u16) {
        let limit = self
            .num_regs_i
            .saturating_add(self.constants_i.len() as u16);
        if self.num_regs_frozen {
            assert!(
                reg < limit,
                "int reg-or-pool {reg} out of bounds {limit} \
                 (assembler.py:72-74 emit_reg / assembler.py:126 emit_const parity)"
            );
            self.encoding_overflow |= reg >= 256;
            return;
        }
        assert!(
            reg < 256,
            "int reg {reg} out of u8 range (assembler.py:72 chr() parity)"
        );
        self.num_regs_i = max(self.num_regs_i, reg.saturating_add(1));
    }

    fn touch_ref_reg_or_pool_slot(&mut self, reg: u16) {
        let limit = self
            .num_regs_r
            .saturating_add(self.constants_r.len() as u16);
        if self.num_regs_frozen {
            assert!(
                reg < limit,
                "ref reg-or-pool {reg} out of bounds {limit} \
                 (assembler.py:72-74 emit_reg / assembler.py:126 emit_const parity)"
            );
            self.encoding_overflow |= reg >= 256;
            return;
        }
        assert!(
            reg < 256,
            "ref reg {reg} out of u8 range (assembler.py:72 chr() parity)"
        );
        self.num_regs_r = max(self.num_regs_r, reg.saturating_add(1));
    }

    fn touch_float_reg_or_pool_slot(&mut self, reg: u16) {
        let limit = self
            .num_regs_f
            .saturating_add(self.constants_f.len() as u16);
        if self.num_regs_frozen {
            assert!(
                reg < limit,
                "float reg-or-pool {reg} out of bounds {limit} \
                 (assembler.py:72-74 emit_reg / assembler.py:126 emit_const parity)"
            );
            self.encoding_overflow |= reg >= 256;
            return;
        }
        assert!(
            reg < 256,
            "float reg {reg} out of u8 range (assembler.py:72 chr() parity)"
        );
        self.num_regs_f = max(self.num_regs_f, reg.saturating_add(1));
    }

    #[allow(dead_code)]
    fn call_ref_like(&mut self, opcode: u8, fn_ptr_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.touch_ref_reg(dst);
        for &arg in arg_regs {
            self.touch_call_arg(arg);
        }
        self.start_instr(opcode);
        self.push_u16(fn_ptr_idx);
        self.push_u16(dst);
        self.push_u16(arg_regs.len() as u16);
        for &arg in arg_regs {
            self.push_u8(arg.kind.encode());
            self.push_u16(arg.reg);
        }
        self.record_resulttype('r');
    }

    #[allow(dead_code)]
    fn call_float_like(&mut self, opcode: u8, fn_ptr_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.touch_float_reg(dst);
        for &arg in arg_regs {
            self.touch_call_arg(arg);
        }
        self.start_instr(opcode);
        self.push_u16(fn_ptr_idx);
        self.push_u16(dst);
        self.push_u16(arg_regs.len() as u16);
        for &arg in arg_regs {
            self.push_u8(arg.kind.encode());
            self.push_u16(arg.reg);
        }
        self.record_resulttype('f');
    }

    fn touch_call_arg(&mut self, arg: JitCallArg) {
        // Call-arg slot indices are produced by
        // `pyre-jit/src/jit/assembler.rs::expect_*_reg_or_pool` which
        // synthesizes `num_regs_kind + pool_idx` for constant operands,
        // so the validated range is the kind+consts union (RPython
        // `emit_const` slot range, `assembler.py`).
        match arg.kind {
            JitArgKind::Int => self.touch_int_reg_or_pool_slot(arg.reg),
            JitArgKind::Ref => self.touch_ref_reg_or_pool_slot(arg.reg),
            JitArgKind::Float => self.touch_float_reg_or_pool_slot(arg.reg),
        }
    }

    fn patch_labels(&mut self) {
        for &(label_idx, patch_offset) in &self.patches {
            let target = match self.labels.get(label_idx).copied().flatten() {
                Some(target) => target as u16,
                None => {
                    // A label was *referenced* (a goto / switch / forwarder operand was
                    // emitted at these code offsets) but never *marked* (its
                    // target block's `Label` was never emitted).  The orthodox
                    // cause is an un-simplified graph reaching flatten — a
                    // dead/trivial forwarder or a constant-folded switch arm
                    // whose target block produced no `Label`.  Report the
                    // offending label index, every referencing code offset, and
                    // the marked/total label counts so the origin can be traced
                    // before the panic, then point at the `simplify_graph`
                    // passes that remove the shape. The walker-safe subset wired ahead of flatten
                    // (`eliminate_empty_blocks` + `constfold_exitswitch` +
                    // `remove_trivial_links`, see `pyre-jit/src/jit/simplify.rs`
                    // module doc) already removes every shape that produces an
                    // unmarked label, so this panic is a fail-loud backstop, not
                    // a live failure mode.
                    let marked = self.labels.iter().filter(|l| l.is_some()).count();
                    let total = self.labels.len();
                    let referencing_offsets: Vec<usize> = self
                        .patches
                        .iter()
                        .filter(|(idx, _)| *idx == label_idx)
                        .map(|(_, off)| *off)
                        .collect();
                    panic!(
                        "jitcode label {label_idx} was never marked \
                         (referenced at code offsets {referencing_offsets:?}; \
                         {marked}/{total} labels marked). A goto/switch/forwarder \
                         targets a block whose Label \
                         was never emitted — typically an un-simplified dead/trivial \
                         forwarder or dead switch arm reaching flatten. The orthodox \
                         fix is graph normalization via simplify_graph \
                         (eliminate_empty_blocks / remove_trivial_links / \
                         constfold_exitswitch)."
                    );
                }
            };
            let bytes = target.to_le_bytes();
            self.code[patch_offset] = bytes[0];
            self.code[patch_offset + 1] = bytes[1];
        }
    }

    fn patch_switch_descrs(&mut self) {
        for (descr_idx, cases) in std::mem::take(&mut self.pending_switches) {
            let dict = cases
                .into_iter()
                .map(|(key, label_idx)| {
                    let target = self
                        .labels
                        .get(label_idx as usize)
                        .copied()
                        .flatten()
                        .unwrap_or_else(|| {
                            panic!(
                                "jitcode switch label {label_idx} was never marked for key {key}"
                            )
                        });
                    (key, target)
                })
                .collect();
            match self.descrs.get_mut(descr_idx as usize) {
                Some(RuntimeBhDescr::Descr(descr)) => {
                    let CanonicalBhDescr::Switch { dict: slot, .. } = descr.as_mut() else {
                        panic!("pending switch descr {descr_idx} is not Switch: {descr:?}");
                    };
                    *slot = dict;
                }
                other => panic!("pending switch descr {descr_idx} is not Switch: {other:?}"),
            }
        }
    }

    /// descr.py parity: `get_field_descr(STRUCT, fieldname)` sets
    /// `fielddescr.parent_descr = get_size_descr(STRUCT)` — the parent
    /// always carries the COMPLETE struct layout known at that point.
    ///
    /// During emission each `add_struct_field_descr` snapshots the
    /// `struct_size_specs` entry as `parent`.  Since `register_struct_layout`
    /// accumulates fields incrementally (each getfield/setfield site
    /// registers only the field it accesses), an early snapshot may contain
    /// fewer fields than the final merged spec.  This matters at runtime:
    /// `field_descr_ref_from_bh` builds a `SizeDescr` from the parent's
    /// `all_fielddescrs`, and the optimizer's `StructPtrInfo.init_fields`
    /// allocates one slot per field.  A partial parent → too few slots →
    /// cross-type forward when different-typed fields land in the same
    /// slot, or a dangling Weak when a later call creates a fuller
    /// SizeDescr that the cache rejects.
    ///
    /// Fix: after all emit is done, walk every `BhDescr::Field` whose
    /// `parent` carries a type_id present in `struct_size_specs` and
    /// replace the parent snapshot with the final (fully merged) spec.
    /// This is the pyre analogue of PyPy always calling `get_size_descr`
    /// at descr-creation time (which returns the single canonical
    /// SizeDescr with all fields populated by `heaptracker.all_fielddescrs`).
    ///
    /// `index_in_parent` and `name` are re-resolved against that same final
    /// spec, because `add_struct_field_descr` derived both from the snapshot
    /// this pass is replacing.  Swapping only the parent leaves the two halves
    /// of one descr disagreeing: the field is declared to sit at slot `i` of a
    /// list whose slot `i` is a different offset.
    ///
    /// The key is the FIELD NAME, `heaptracker.py
    /// get_fielddescr_index_in(STRUCT, fieldname)`, whenever the mint recorded
    /// one.  Offset is not an identity: measured on this tree, 7 of 1714
    /// submitted field specs sit at an offset another field of the same parent
    /// also occupies, so `position(|fd| fd.offset == offset)` can name a
    /// sibling.  It is used only for the descrs whose recorded name the final
    /// spec does not list — every emit site carries the declared member name
    /// into `add_struct_field_descr`, so what reaches the offset arm is a field
    /// the layout never enumerated, not a field that arrived anonymous — and
    /// then only when the offset is UNAMBIGUOUS.  Where it is not, this leaves
    /// the descr alone: `descr.rs find_index_in_parent` refuses to invent a
    /// position for exactly this reason, and a pass that guesses is worse than
    /// one that declines.
    ///
    /// A field still absent from the final spec keeps what the mint left it —
    /// `add_struct_field_descr`'s `unwrap_or((0, String::new()))`.  Those are
    /// the inline aggregates the flattened layout only covers through their
    /// leaves (`ob_header`, `int_items`, an enum's `__pos_0`), which
    /// `heaptracker.py` mints no descr for at all.
    fn patch_field_descr_parents(&mut self) {
        for entry in &mut self.descrs {
            let RuntimeBhDescr::Descr(descr) = entry else {
                continue;
            };
            let CanonicalBhDescr::Field {
                offset,
                index_in_parent,
                parent,
                name,
                ..
            } = descr.as_mut()
            else {
                continue;
            };
            let Some(p) = parent else { continue };
            let Some(final_spec) = self.struct_size_specs.get(&p.type_id) else {
                continue;
            };
            *p = std::sync::Arc::new(final_spec.clone());
            let Some(idx) = field_slot_in(&p.all_fielddescrs, name, *offset) else {
                continue;
            };
            *index_in_parent = Some(idx);
            name.clone_from(&p.all_fielddescrs[idx].name);
        }
    }

    /// The postcondition of [`patch_field_descr_parents`], checked rather than
    /// assumed: every emitted `Field` descr agrees with the parent attached to
    /// it about which slot the field occupies.
    ///
    /// This exists because the corpus cannot be the detector. The defect the
    /// pass fixes is *input-order dependent* — it needs a field minted before a
    /// lower-offset sibling registers — so a program that happens to register in
    /// offset order exercises the producer without exercising the bug. Measured:
    /// with the two re-resolution lines removed, `field_pos_rederived`,
    /// `field_pos_spec_misplaced`, `field_pos_attached_misplaced` and
    /// `positional_misplaced` all still read 0 over the whole corpus. A census
    /// counts what ran; only a construction-site check covers what the producer
    /// can emit.
    ///
    /// Three shapes, all of them real states this pass can leave behind:
    ///
    /// * the field resolves by [`field_slot_in`] — index and name must be that
    ///   slot's. This is the re-resolution above, so it guards future edits
    ///   rather than today's tree.
    /// * it does not resolve — the descr must still carry
    ///   `add_struct_field_descr`'s `(0, String::new())` fallback untouched.
    ///   Those are the inline aggregates (`ob_header`, `int_items`, an enum's
    ///   `__pos_0`) the flattened layout represents only through their leaves.
    /// * the parent's `type_id` is absent from `struct_size_specs` — the
    ///   `continue` above, which leaves the mint-time snapshot in place. Nothing
    ///   patched it, so nothing has established the invariant for it either.
    ///
    /// The claim is stated over [`field_slot_in`], not over a bare
    /// `position(offset)`: the tree has parents with two fields at one offset,
    /// so an offset-keyed claim would panic on a descr that names its field
    /// correctly and merely shares an address with a sibling.
    ///
    /// Returns the first disagreement as a message, so the caller's panic says
    /// which descr and both numbers. Gated at the call site on
    /// [`jit_strict_mode`](crate::jit_strict_mode) rather than wrapped in a
    /// `debug_assert!`: the message needs the same walk the predicate does and
    /// `debug_assert!` would run it twice, and `cfg!(debug_assertions)` alone
    /// would leave the check inert for the release corpus, which is where the
    /// producers this covers are exercised most widely.
    ///
    /// [`patch_field_descr_parents`]: Self::patch_field_descr_parents
    fn field_descr_position_disagreement(&self) -> Option<String> {
        for entry in &self.descrs {
            let RuntimeBhDescr::Descr(descr) = entry else {
                continue;
            };
            let CanonicalBhDescr::Field {
                offset,
                index_in_parent,
                parent: Some(p),
                name,
                ..
            } = descr.as_ref()
            else {
                continue;
            };
            match field_slot_in(&p.all_fielddescrs, name, *offset) {
                Some(idx) => {
                    let slot = &p.all_fielddescrs[idx];
                    if *index_in_parent != Some(idx) || *name != slot.name {
                        return Some(format!(
                            "field descr at offset {offset} of type_id {:#x} claims slot \
                             {index_in_parent:?} named {name:?}, but that offset is slot {idx} \
                             named {:?}",
                            p.type_id, slot.name,
                        ));
                    }
                }
                // The unresolved fallback is now a type, not a convention:
                // `None` beside an empty name.
                None if index_in_parent.is_some() || !name.is_empty() => {
                    return Some(format!(
                        "field descr at offset {offset} does not resolve in type_id {:#x}'s \
                         layout yet carries slot {index_in_parent:?} named {name:?} instead of \
                         the unresolved fallback",
                        p.type_id,
                    ));
                }
                None => {}
            }
        }
        None
    }

    /// RPython `assembler.py:131-138` resolves a const-source operand
    /// to `count_regs[kind] + len(constants) - 1`. pyre performs the
    /// same resolution as a post-emission pass once per-kind
    /// `num_regs_X` is final: `num_regs_X + pool_idx` is the slot the
    /// register file-backed `{int,ref,float}_copy` reads from.
    fn patch_const_refs(&mut self) {
        for &(offset, kind, pool_idx) in &self.const_patches {
            let base = match kind {
                ConstKind::Int => self.num_regs_i,
                ConstKind::Ref => self.num_regs_r,
                ConstKind::Float => self.num_regs_f,
            };
            let slot = base + pool_idx;
            let bytes = slot.to_le_bytes();
            self.code[offset] = bytes[0];
            self.code[offset + 1] = bytes[1];
        }
    }

    /// 1-byte counterpart of `patch_const_refs`. `loop_header/i` and
    /// `jit_merge_point/iIRFIRF` jdindex bytes (`@arguments("i")`,
    /// blackhole.py) carry a single register-index byte.
    /// Asserts the resolved slot fits in u8 — overflow means the portal
    /// has more than 255 int registers + constants combined, which
    /// would also break the broader 1-byte register encoding.
    fn patch_const_u8_refs(&mut self) {
        for &(offset, kind, pool_idx) in &self.const_patches_u8 {
            let base = match kind {
                ConstKind::Int => self.num_regs_i,
                ConstKind::Ref => self.num_regs_r,
                ConstKind::Float => self.num_regs_f,
            };
            let slot = base + pool_idx;
            assert!(
                slot <= jitcode::JitcodeReg::MAX as u16,
                "patch_const_u8_refs: slot {slot} (base={base} + pool_idx={pool_idx}) overflows u8"
            );
            self.code[offset] = slot as jitcode::JitcodeReg;
        }
    }
}

fn canonical_bh_descr_eq(lhs: &CanonicalBhDescr, rhs: &CanonicalBhDescr) -> bool {
    match (lhs, rhs) {
        (
            CanonicalBhDescr::VableField { index: lhs },
            CanonicalBhDescr::VableField { index: rhs },
        ) => lhs == rhs,
        (
            CanonicalBhDescr::VableArray { index: lhs },
            CanonicalBhDescr::VableArray { index: rhs },
        ) => lhs == rhs,
        (CanonicalBhDescr::Switch { .. }, CanonicalBhDescr::Switch { .. }) => false,
        (
            CanonicalBhDescr::Array {
                base_size: lhs_base_size,
                itemsize: lhs_itemsize,
                len_offset: lhs_len_offset,
                type_id: lhs_type_id,
                gc_type_id: lhs_gc_type_id,
                item_type: lhs_item_type,
                is_array_of_pointers: lhs_is_array_of_pointers,
                is_array_of_structs: lhs_is_array_of_structs,
                is_item_signed: lhs_is_item_signed,
                ei_index: _,
                array_type_id: lhs_array_type_id,
                interior_fields: lhs_interior_fields,
                is_gc_managed: lhs_is_gc_managed,
            },
            CanonicalBhDescr::Array {
                base_size: rhs_base_size,
                itemsize: rhs_itemsize,
                len_offset: rhs_len_offset,
                type_id: rhs_type_id,
                gc_type_id: rhs_gc_type_id,
                item_type: rhs_item_type,
                is_array_of_pointers: rhs_is_array_of_pointers,
                is_array_of_structs: rhs_is_array_of_structs,
                is_item_signed: rhs_is_item_signed,
                ei_index: _,
                array_type_id: rhs_array_type_id,
                interior_fields: rhs_interior_fields,
                is_gc_managed: rhs_is_gc_managed,
            },
        ) => {
            // `ei_index` is intentionally NOT part of the identity
            // tuple — upstream `gccache._cache_array[ARRAY_OR_STRUCT]`
            // (`descr.py get_array_descr`) keys on the lltype itself, and
            // `compute_bitstrings` (`effectinfo.py`) later assigns
            // the index slot as a derived attribute that multiple
            // descrs are free to share.
            //
            // `array_type_id` joins the identity tuple as the
            // codewriter lltype-identity proxy so two ARRAY entries
            // that disagree on the Rust type spelling stay on distinct
            // canonical slots even when their numeric `type_id`
            // collides (default `0`).
            //
            // `gc_type_id` is `init_array_descr` `descr.tid`. Two ARRAY
            // lltypes that share a layout still have distinct tids.
            lhs_base_size == rhs_base_size
                && lhs_itemsize == rhs_itemsize
                && lhs_len_offset == rhs_len_offset
                && lhs_type_id == rhs_type_id
                && lhs_gc_type_id == rhs_gc_type_id
                && lhs_item_type == rhs_item_type
                && lhs_is_array_of_pointers == rhs_is_array_of_pointers
                && lhs_is_array_of_structs == rhs_is_array_of_structs
                && lhs_is_item_signed == rhs_is_item_signed
                && lhs_array_type_id == rhs_array_type_id
                && lhs_interior_fields == rhs_interior_fields
                && lhs_is_gc_managed == rhs_is_gc_managed
        }
        // TODO: `Call` variant intentionally falls
        // through `_ => false`. See `add_call_descr`'s docstring — pyre's
        // per-callsite `JitCodeExecState.call_descr_to_call_target`
        // sidetable is keyed by descr slot, so dedup'ing two distinct
        // callees that share a signature would clobber the sidetable
        // entry. The convergence path requires lifting the sidetable
        // onto the funcptr int-const slot first.
        _ => false,
    }
}

/// RPython `assembler.py get_liveness_info(insn, kind)` adapted
/// for the flat-state JIT: every state_field slot is permanently live,
/// so the canonical `(live_i, live_r, live_f)` triple just enumerates
/// the int register file from `int_scalar_base..int_scalar_base +
/// total_slots`.
///
/// state-field JIT enforces `Type::Int` on every slot at macro
/// expansion (`codegen_state.rs`'s `generate_state_fields_jit_state`), so
/// `live_r` and `live_f` are
/// empty.  `total_slots` matches `JitCodeSym::total_slots`:
/// `num_scalars + sum(array_lens) + num_vable_identity_slots`. Callers
/// pass ptr then len (two Int slots) per plain `[int]` as `num_scalars`
/// with empty `array_lens`. The last term is 1 when the state has a virtualizable
/// and 0 otherwise —
/// `pyjitpl.py:2984-2989` carries the virtualizable once regardless of how
/// many `[.. ; virt]` arrays it holds, and `virtualizable.py:150-153` reads
/// every array length off the live object rather than boxing it.
///
/// The returned `Vec<u8>`s are caller-owned and can be passed directly
/// into `JitCodeBuilder::live` / `Assembler::_encode_liveness`.
///
/// # Panics
///
/// Panics if either bank's highest slot index would exceed 255, i.e.
/// `base + count > 256` (the ranges are end-exclusive, so an end of 256
/// fills indices 0..=255 and is legal).  RPython jitcode register indices
/// are `chr(...)`-encoded `u8`s (`assembler.py:241`, `liveness.py:148-159`);
/// a 257th slot would overflow that encoding.
#[expect(
    clippy::too_many_arguments,
    reason = "The parameter order mirrors the corresponding RPython metainterpreter routine; grouping arguments into a Rust-only context object would obscure line-by-line parity and frame ownership"
)]
pub fn live_slots_for_state_field_jit(
    num_scalars: usize,
    array_lens: &[usize],
    num_vable_identity_slots: usize,
    num_ref_scalars: usize,
    ref_scalar_base: usize,
    num_float_scalars: usize,
    float_scalar_base: usize,
    int_scalar_base: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let total_slots: usize = num_scalars + array_lens.iter().sum::<usize>();
    // Int identity slots start past the dispatch JitCode's int-bank
    // argument registers (`pc` at i0). The vable identity is a Ref in
    // the ref bank / `virtualizable_boxes[-1]`, not an int slot.
    let int_scalar_end = int_scalar_base + total_slots;
    // End-exclusive: `int_scalar_base..int_scalar_end` fills indices up to
    // `int_scalar_end - 1`, so an end of exactly 256 (highest index 255) is
    // legal; only an end past 256 overflows the u8 register index.
    assert!(
        int_scalar_end <= 256,
        "live_slots_for_state_field_jit: int_scalar_base={int_scalar_base} + \
         total_slots={total_slots} exceeds RPython jitcode u8 register-index \
         limit (assembler.py:241, liveness.py:148-159)",
    );
    // Ref-typed scalars occupy the ref register bank, densely packed at
    // ref_scalar_base..ref_scalar_base+num_ref_scalars. The base skips the
    // dispatch JitCode's ref-bank argument registers (`program` at r0, the
    // virtualizable identity at r1 when present): the blackhole re-executes
    // ops that read those argument registers, so an identity slot aliasing
    // one would clobber it at resume-decode time.
    let ref_scalar_end = ref_scalar_base + num_ref_scalars;
    assert!(
        ref_scalar_end <= 256,
        "live_slots_for_state_field_jit: ref_scalar_base={ref_scalar_base} + \
         num_ref_scalars={num_ref_scalars} exceeds the u8 register-index limit",
    );
    let float_scalar_end = float_scalar_base + num_float_scalars;
    assert!(
        float_scalar_end <= 256,
        "live_slots_for_state_field_jit: float_scalar_base={float_scalar_base} + \
         num_float_scalars={num_float_scalars} exceeds the u8 register-index limit",
    );
    let live_i: Vec<u8> = (int_scalar_base as u32..int_scalar_end as u32)
        .map(|i| i as u8)
        .collect();
    // Vable identity is the portal ref argument immediately before the
    // ref-scalar identity prefix (`ref_scalar_base - 1`).
    let live_r_start = if num_vable_identity_slots > 0 {
        ref_scalar_base.saturating_sub(1)
    } else {
        ref_scalar_base
    };
    let live_r: Vec<u8> = (live_r_start as u32..ref_scalar_end as u32)
        .map(|i| i as u8)
        .collect();
    let live_f: Vec<u8> = (float_scalar_base as u32..float_scalar_end as u32)
        .map(|i| i as u8)
        .collect();
    (live_i, live_r, live_f)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blackhole::{BlackholeInterpBuilder, wire_bhimpl_handlers};
    use majit_jitcode::codewriter::assembler::Assembler;

    fn assert_resulttype_after(emit: impl FnOnce(&mut JitCodeBuilder), kind: char) {
        let mut builder = JitCodeBuilder::new();
        emit(&mut builder);
        let pc = builder.current_pos();
        let jitcode = builder.finish();
        assert_eq!(
            jitcode
                .body()
                .resulttypes
                .as_ref()
                .and_then(|resulttypes| resulttypes.get(&pc).copied()),
            Some(kind)
        );
    }

    /// `inline_call_ending_at` has to answer with exactly the operands the
    /// emitter wrote, for every argument count and every return bank.
    ///
    /// It is the only reader of a `BC_INLINE_CALL`'s operands that does not
    /// start at the instruction — a frame rebuilt from resume data knows only
    /// where the instruction ENDED — so nothing else would catch it drifting
    /// from the encoder. Both directions are graded: the slots it recovers,
    /// and that it declines a position that is not the far side of one.
    #[test]
    fn inline_call_ending_at_recovers_what_the_emitter_wrote() {
        use crate::jitcode::JitArgKind;

        let args: [&[(u16, u16)]; 3] = [&[], &[(1, 0)], &[(1, 0), (2, 1), (3, 2)]];
        for args_r in args {
            for (bank, emit) in [
                (Some(JitArgKind::Int), 0usize),
                (Some(JitArgKind::Ref), 1),
                (None, 2),
            ] {
                let mut builder = JitCodeBuilder::new();
                // A leading instruction, so the call never starts at 0 and a
                // decoder that ignored the opcode byte could not accidentally
                // be right.
                builder.int_guard_value(4);
                let start = builder.current_pos();
                match emit {
                    0 => builder.inline_call_r_i(7, args_r, Some(9)),
                    1 => builder.inline_call_r_r(7, args_r, Some(9)),
                    _ => builder.inline_call_r_v(7, args_r, None),
                }
                let end = builder.current_pos();
                let jitcode = builder.finish();

                let site = jitcode
                    .inline_call_ending_at(end)
                    .expect("the call ends here, so it must decode");
                assert_eq!(site.sub_idx, 7, "sub-JitCode index");
                assert_eq!(
                    site.result_slot(),
                    bank.map(|kind| (kind, 9usize)),
                    "return slot for {args_r:?} args, bank {bank:?}",
                );
                // Canonical `inline_call_r_*`: opcode + descr u16 + R-list
                // + dest byte for a typed result.
                let dest_bytes = usize::from(bank.is_some());
                assert_eq!(
                    end - start,
                    1 + 2 + (1 + args_r.len()) + dest_bytes,
                    "encoded width"
                );
                // A position one byte off is not the far side of this call,
                // and answering there would be a wrong frame rebuild rather
                // than a decline.
                assert_eq!(jitcode.inline_call_ending_at(end - 1), None);
                assert_eq!(jitcode.inline_call_ending_at(start), None);
            }
        }
    }

    fn assert_no_resulttype_after(emit: impl FnOnce(&mut JitCodeBuilder)) {
        let mut builder = JitCodeBuilder::new();
        emit(&mut builder);
        let pc = builder.current_pos();
        let jitcode = builder.finish();
        assert_eq!(
            jitcode
                .body()
                .resulttypes
                .as_ref()
                .and_then(|resulttypes| resulttypes.get(&pc).copied()),
            None
        );
    }

    /// `add_struct_field_descr` resolves `index_in_parent` / `name` against
    /// the layout accumulated SO FAR, and `register_struct_layout` re-indexes
    /// every entry on each merge, so a field minted before a lower-offset
    /// sibling is registered carries a rank the merge then invalidates.
    ///
    /// `patch_field_descr_parents` installs the final merged spec as the
    /// parent, so a stale rank left beside it declares the field at a slot the
    /// parent fills with a different offset — `descr.py:228`'s index and
    /// `descr.py:238`'s parent_descr describing two different fields.
    ///
    /// Emission order here is high-offset first, which is what makes the
    /// prefix rank (0) differ from the final rank (1).
    #[test]
    fn field_descr_index_follows_the_final_layout_not_the_mint_time_prefix() {
        const TID: u64 = 0x5747_5F49_4458;
        let mut builder = JitCodeBuilder::new();
        // Site 1 registers only the HIGH offset, so the mint ranks it 0.
        builder.register_struct_layout(
            24,
            TID,
            false,
            false,
            &[(16, false, "hi", 8, true, false)],
            "",
        );
        builder.getfield_gc_i(0, 1, 16, TID, "hi");
        // Site 2 registers the LOW offset; the merge re-indexes to {8→0, 16→1}.
        builder.register_struct_layout(
            24,
            TID,
            false,
            false,
            &[(8, false, "lo", 8, true, false)],
            "",
        );
        builder.getfield_gc_i(2, 1, 8, TID, "lo");
        let jitcode = builder.finish();

        let fields: Vec<_> = jitcode
            .exec
            .descrs
            .iter()
            .filter_map(|entry| match entry.as_bh_descr() {
                Some(CanonicalBhDescr::Field {
                    offset,
                    index_in_parent,
                    parent,
                    name,
                    ..
                }) => Some((*offset, *index_in_parent, name.clone(), parent.clone())),
                _ => None,
            })
            .collect();
        assert_eq!(fields.len(), 2, "one Field descr per getfield site");
        for (offset, index_in_parent, name, parent) in fields {
            let parent = parent.expect("a registered type_id gives the field a parent");
            assert_eq!(
                parent.all_fielddescrs.len(),
                2,
                "the parent is the final merged spec, not the mint-time prefix",
            );
            let (expected_index, expected_name) = match offset {
                8 => (Some(0), "lo"),
                16 => (Some(1), "hi"),
                other => panic!("unexpected field offset {other}"),
            };
            assert_eq!(
                index_in_parent, expected_index,
                "field at offset {offset} must rank by the final layout",
            );
            assert_eq!(name, expected_name, "and name from that same layout");
            assert_eq!(
                parent.all_fielddescrs[index_in_parent.expect("resolved above")].offset,
                offset,
                "`all_fielddescrs[index_in_parent]` must be this very field \
                 (`heaptracker.py:60-72` / `:96-112` share one walker upstream)",
            );
        }
    }

    #[test]
    fn field_descr_runtime_entry_reuses_one_optimizer_identity() {
        const TID: u64 = 0x5254_4649_454c_44;
        let mut builder = JitCodeBuilder::new();
        builder.register_struct_layout(
            16,
            TID,
            false,
            false,
            &[(8, false, "value", 8, true, false)],
            "",
        );
        builder.getfield_gc_i(0, 1, 8, TID, "value");
        let jitcode = builder.finish();
        let entry = jitcode
            .exec
            .descrs
            .iter()
            .find(|entry| matches!(entry.as_bh_descr(), Some(CanonicalBhDescr::Field { .. })))
            .expect("getfield descriptor");

        let first = entry
            .as_optimizer_descr()
            .expect("field pool entry is resolved once")
            .clone();
        let second = entry
            .as_optimizer_descr()
            .expect("the resolved identity remains on the pool entry")
            .clone();
        assert!(std::sync::Arc::ptr_eq(&first, &second));
    }

    /// Two fields at one offset — a flattened inline aggregate and its first
    /// leaf — must not be arbitrated by offset. Neither the re-resolution nor
    /// the debug postcondition may name one of them.
    ///
    /// Measured on the real corpus: 7 of 1714 submitted field specs share an
    /// offset with a sibling, so this is a reachable state and not a
    /// hypothetical. An offset-keyed pass would rewrite `index_in_parent` to
    /// whichever of the two happens to sort first, and the postcondition would
    /// then panic on any descr correctly naming the other.
    ///
    /// The named access resolves through it; the unnamed one declines. Both
    /// halves matter: `heaptracker.get_fielddescr_index_in(STRUCT, fieldname)`
    /// keys on the name, and the offset is only what is left when a caller has
    /// none.
    #[test]
    fn an_ambiguous_offset_is_arbitrated_by_name_and_otherwise_declined() {
        const TID: u64 = 0x414D_4249_4755;
        let field_of = |name: &str| {
            let mut builder = JitCodeBuilder::new();
            // One `register_struct_layout` call, so the merge's offset dedup
            // does not apply and both fields survive at offset 8.
            builder.register_struct_layout(
                24,
                TID,
                false,
                false,
                &[
                    (8, false, "agg", 8, true, false),
                    (8, true, "leaf", 8, false, false),
                ],
                "",
            );
            builder.getfield_gc_i(0, 1, 8, TID, name);
            // `finish` runs the postcondition under `jit_strict_mode`; a
            // resolution that guessed would trip it before this returns.
            let jitcode = builder.finish();
            let (index_in_parent, name, parent) = jitcode
                .exec
                .descrs
                .iter()
                .find_map(|entry| match entry.as_bh_descr() {
                    Some(CanonicalBhDescr::Field {
                        index_in_parent,
                        parent,
                        name,
                        ..
                    }) => Some((*index_in_parent, name.clone(), parent.clone())),
                    _ => None,
                })
                .expect("one Field descr for the getfield site");
            let parent = parent.expect("a registered type_id gives the field a parent");
            assert_eq!(parent.all_fielddescrs.len(), 2, "both fields are listed");
            assert_eq!(
                parent.all_fielddescrs[0].offset, parent.all_fielddescrs[1].offset,
                "the fixture's point is that one offset names two fields",
            );
            (index_in_parent, name)
        };

        let (idx, name) = field_of("leaf");
        assert_eq!(
            (idx, name.as_str()),
            (Some(1), "leaf"),
            "the emit site's `fieldname` names the field the shared offset cannot",
        );
        let (idx, name) = field_of("");
        assert_eq!(
            (idx, name.as_str()),
            (None, ""),
            "without a name the ambiguous offset resolves to nothing, so the \
             mint's `unwrap_or((None, String::new()))` fallback stands — and \
             `None` rather than `0` is the point: the miss must not spell \
             itself as a claim on the parent's first slot",
        );
    }

    fn publish_parent_layout(type_id: u64, size: usize, fields: &[(&str, usize)]) {
        let specs: Vec<_> = fields
            .iter()
            .enumerate()
            .map(
                |(i, (name, offset))| majit_ir::descr::SimpleFieldDescrSpec {
                    index: i as u32,
                    field_key: (*name).to_string(),
                    name: (*name).to_string(),
                    offset: *offset,
                    field_size: 8,
                    field_type: majit_ir::value::Type::Int,
                    is_immutable: false,
                    is_quasi_immutable: false,
                    flag: majit_ir::descr::ArrayFlag::Signed,
                    virtualizable: false,
                    index_in_parent: i,
                    is_class_word: Some(false),
                },
            )
            .collect();
        let _ = majit_ir::descr::make_simple_descr_group_keyed_with_headerless(
            u32::MAX,
            size,
            0,
            type_id,
            0,
            true,
            false,
            &specs,
            &[],
        );
    }

    #[test]
    fn rank_field_specs_matches_exact_key_before_shared_offset() {
        const TID: u64 = 0x4B45_5946_4952_5354;
        publish_parent_layout(TID, 24, &[("agg", 8), ("leaf", 8)]);
        let fields = [
            (8, false, "agg", 8, true, false),
            (8, true, "leaf", 8, false, false),
        ];
        let specs = JitCodeBuilder::field_specs_from_layout(TID, &fields, &[]);
        assert_eq!(
            specs
                .iter()
                .map(|spec| (spec.field_key.as_str(), spec.index_in_parent))
                .collect::<Vec<_>>(),
            vec![("agg", 0), ("leaf", 1)],
            "exact field_key must win over a shared offset"
        );
    }

    #[test]
    fn register_struct_layout_reranks_known_offsets_after_cache_publication() {
        const TID: u64 = 0x5245_5241_4E4B_504F;
        let mut builder = JitCodeBuilder::new();
        builder.register_struct_layout(
            24,
            TID,
            false,
            false,
            &[
                (8, false, "lo", 8, true, false),
                (16, false, "hi", 8, true, false),
            ],
            "",
        );
        let before: Vec<_> = builder
            .struct_size_specs
            .get(&TID)
            .unwrap()
            .all_fielddescrs
            .iter()
            .map(|spec| (spec.field_key.clone(), spec.index_in_parent))
            .collect();
        assert_eq!(before, vec![("lo".into(), 0), ("hi".into(), 1)]);
        publish_parent_layout(TID, 24, &[("hi", 16), ("lo", 8)]);
        builder.register_struct_layout(
            24,
            TID,
            false,
            false,
            &[
                (8, false, "lo", 8, true, false),
                (16, false, "hi", 8, true, false),
            ],
            "",
        );
        let after: Vec<_> = builder
            .struct_size_specs
            .get(&TID)
            .unwrap()
            .all_fielddescrs
            .iter()
            .map(|spec| (spec.field_key.clone(), spec.index_in_parent))
            .collect();
        assert_eq!(
            after,
            vec![("hi".into(), 0), ("lo".into(), 1)],
            "re-registration after cache publication must adopt all_fielddescrs order"
        );
    }

    /// …but a name always wins, even when the offset it sits at is ambiguous.
    /// That is `heaptracker.py get_fielddescr_index_in(STRUCT, fieldname)`
    /// — the offset is only a stand-in for the mint sites that carry no name.
    #[test]
    fn a_named_field_resolves_by_name_through_an_ambiguous_offset() {
        let fields = [
            (0, false, "head", 8, false, false),
            (8, false, "agg", 8, true, false),
            (8, true, "leaf", 8, false, false),
        ];
        assert_eq!(
            super::field_slot_in(
                &JitCodeBuilder::field_specs_from_layout(0, &fields, &[]),
                "leaf",
                8,
            ),
            Some(2),
            "the name names the field; the shared offset does not",
        );
        assert_eq!(
            super::field_slot_in(
                &JitCodeBuilder::field_specs_from_layout(0, &fields, &[]),
                "",
                8
            ),
            None,
            "without a name the shared offset arbitrates nothing",
        );
        assert_eq!(
            super::field_slot_in(
                &JitCodeBuilder::field_specs_from_layout(0, &fields, &[]),
                "",
                0
            ),
            Some(0),
            "an unambiguous offset still resolves",
        );
    }

    #[test]
    fn typed_vable_helpers_record_resulttypes_at_end_pc() {
        // RPython assembler.py records `argcodes[-1]` at
        // `len(code)` after all operands are emitted. These helper-side
        // adapters are not canonical argcode layouts, but their result
        // byte is still the last operand consumed by make_result_of_lastop.
        let vr = 0;
        assert_resulttype_after(|b| b.vable_getfield_int_with_base(0, vr, 1), 'i');
        assert_resulttype_after(|b| b.vable_getfield_ref_with_base(0, vr, 1), 'r');
        assert_resulttype_after(|b| b.vable_getfield_float_with_base(0, vr, 1), 'f');
        assert_resulttype_after(|b| b.vable_getarrayitem_int_with_base(0, vr, 1, 2), 'i');
        assert_resulttype_after(|b| b.vable_getarrayitem_ref_with_base(0, vr, 1, 2), 'r');
        assert_resulttype_after(|b| b.vable_getarrayitem_float_with_base(0, vr, 1, 2), 'f');
        assert_resulttype_after(|b| b.vable_arraylen_with_base(0, vr, 1), 'i');
    }

    #[test]
    fn canonical_vable_field_emit_uses_base_reg_and_descr_pool() {
        let mut builder = JitCodeBuilder::new();
        builder.vable_getfield_ref_with_base(2, 1, 3);
        let jitcode = builder.finish();
        let opcode = jitcode::insn_byte("getfield_vable_r/rd>r");
        assert_eq!(jitcode.code, vec![opcode, 1, 0, 0, 2]);
        assert!(matches!(
            jitcode.exec.descrs[0].as_bh_descr(),
            Some(CanonicalBhDescr::VableField { index: 3 })
        ));
        assert_eq!(
            jitcode
                .body()
                .resulttypes
                .as_ref()
                .and_then(|resulttypes| resulttypes.get(&5).copied()),
            Some('r')
        );
    }

    #[test]
    fn canonical_new_with_vtable_emit_uses_size_descr_and_ref_result() {
        let mut builder = JitCodeBuilder::new();
        builder.new_with_vtable(2, 16, 0xAB, 0x1234);
        let jitcode = builder.finish();
        let opcode = jitcode::insn_byte("new_with_vtable/d>r");
        // [opcode, descr_idx_lo, descr_idx_hi, dest]
        assert_eq!(jitcode.code, vec![opcode, 0, 0, 2]);
        assert!(matches!(
            jitcode.exec.descrs[0].as_bh_descr(),
            Some(CanonicalBhDescr::Size {
                size: 16,
                type_id: 0xAB,
                vtable: 0x1234,
                ..
            })
        ));
        assert_eq!(
            jitcode
                .body()
                .resulttypes
                .as_ref()
                .and_then(|resulttypes| resulttypes.get(&4).copied()),
            Some('r')
        );
    }

    #[test]
    fn canonical_new_struct_emit_uses_size_descr_without_vtable() {
        let mut builder = JitCodeBuilder::new();
        builder.new_struct(3, 16, 0xCD, false, &[], "");
        let jitcode = builder.finish();
        let opcode = jitcode::insn_byte("new/d>r");
        assert_eq!(jitcode.code, vec![opcode, 0, 0, 3]);
        assert!(matches!(
            jitcode.exec.descrs[0].as_bh_descr(),
            Some(CanonicalBhDescr::Size {
                size: 16,
                type_id: 0xCD,
                vtable: 0,
                ..
            })
        ));
    }

    fn switch_return_jitcode() -> JitCode {
        let mut builder = JitCodeBuilder::new();
        let case_a = builder.new_label();
        let case_b = builder.new_label();
        let case_c = builder.new_label();
        builder.switch(0, &[(-3, case_a), (7, case_b), (42, case_c)]);
        builder.load_const_i_value(1, 10);
        builder.int_return(1);
        builder.mark_label(case_a);
        builder.load_const_i_value(1, 20);
        builder.int_return(1);
        builder.mark_label(case_b);
        builder.load_const_i_value(1, 30);
        builder.int_return(1);
        builder.mark_label(case_c);
        builder.load_const_i_value(1, 40);
        builder.int_return(1);
        builder.finish()
    }

    fn run_switch_blackhole(input: i64) -> i64 {
        let mut entries: indexmap::IndexMap<String, u8> = indexmap::IndexMap::new();
        entries.insert("switch/id".to_string(), jitcode::insns::BC_SWITCH);
        entries.insert("int_copy/i>i".to_string(), jitcode::insns::BC_MOVE_I);
        // `load_const_i_value` emits the `USE_C_FORM` short encoding for
        // byte-sized constants, so wire both `int_copy` variants.
        entries.insert("int_copy/c>i".to_string(), jitcode::insns::BC_MOVE_I_C);
        entries.insert("int_return/i".to_string(), jitcode::insns::BC_INT_RETURN);
        let mut builder = BlackholeInterpBuilder::new();
        builder.setup_insns(&entries);
        wire_bhimpl_handlers(&mut builder);
        let mut bh = builder.acquire_interp();
        bh.setposition(std::sync::Arc::new(switch_return_jitcode()), 0);
        bh.registers_i[0] = input;
        let _ = bh.run();
        bh.tmpreg_i
    }

    #[test]
    fn switch_builder_patches_descr_labels_for_blackhole() {
        // RPython `flatten.py:270-308` + `blackhole.py:954-960` parity:
        // matching keys jump to attached label positions; misses fall
        // through to the default path immediately after the switch.
        assert_eq!(run_switch_blackhole(-3), 20);
        assert_eq!(run_switch_blackhole(7), 30);
        assert_eq!(run_switch_blackhole(42), 40);
        assert_eq!(run_switch_blackhole(99), 10);
    }

    #[test]
    fn canonical_vable_array_emit_uses_two_descrs() {
        let mut builder = JitCodeBuilder::new();
        builder.vable_getarrayitem_ref_with_base(4, 1, 7, 2);
        let jitcode = builder.finish();
        let opcode = jitcode::insn_byte("getarrayitem_vable_r/ridd>r");
        assert_eq!(jitcode.code, vec![opcode, 1, 2, 0, 0, 1, 0, 4]);
        assert!(matches!(
            jitcode.exec.descrs[0].as_bh_descr(),
            Some(CanonicalBhDescr::VableArray { index: 7 })
        ));
        assert!(matches!(
            jitcode.exec.descrs[1].as_bh_descr(),
            Some(CanonicalBhDescr::Array {
                item_type: majit_ir::value::Type::Ref,
                ..
            })
        ));
    }

    #[test]
    fn add_call_descr_pushes_canonical_call_entry() {
        // of — verify
        // that `add_call_descr` puts a `BhDescr::Call { calldescr }` at the
        // returned pool index so canonical `residual_call_*_v` handlers
        // (`blackhole.rs`) can reach it via `read_descr` →
        // `as_calldescr()` (`majit-jitcode/src/codewriter/jitcode.rs`).
        let mut builder = JitCodeBuilder::new();
        let calldescr = majit_jitcode::codewriter::jitcode::BhCallDescr::from_signature(
            "i".to_string(),
            majit_ir::value::Type::Void,
            majit_ir::descr::EffectInfo::MOST_GENERAL,
        );
        let idx = builder.add_call_descr(calldescr);
        assert_eq!(idx, 0);
        let jitcode = builder.finish();
        let entry = &jitcode.exec.descrs[idx as usize];
        let bh_descr = entry.as_bh_descr().expect("call descr must be BhDescr");
        let cd = bh_descr.as_calldescr();
        assert_eq!(cd.arg_classes, "i");
        assert_eq!(cd.result_type, 'v');
    }

    /// A callee that declares its argument kinds, the way one assembled from a
    /// source function does.
    fn callee_declaring(arg_classes: &str) -> JitCode {
        let mut builder = JitCodeBuilder::new();
        builder.set_calldescr(
            majit_jitcode::codewriter::jitcode::BhCallDescr::from_signature(
                arg_classes.to_string(),
                majit_ir::value::Type::Void,
                majit_ir::descr::EffectInfo::MOST_GENERAL,
            ),
        );
        builder.finish()
    }

    #[test]
    #[should_panic(expected = "passes 1 int / 0 ref / 0 float argument(s)")]
    fn an_inline_call_passing_the_wrong_argument_kind_is_rejected() {
        // The callee takes one reference; the caller hands it an integer. The
        // callee would read its reference register 0, which this call never
        // wrote, and the frame would carry a stale value forward with nothing
        // reporting it.
        let mut builder = JitCodeBuilder::new();
        let idx = builder.add_sub_jitcode(callee_declaring("r"));
        builder.inline_call_ir_v(idx, &[(0, 0)], &[], None);
    }

    #[test]
    fn an_inline_call_matching_the_callees_signature_is_emitted() {
        let mut builder = JitCodeBuilder::new();
        let idx = builder.add_sub_jitcode(callee_declaring("r"));
        let start = builder.current_pos();
        builder.inline_call_ir_v(idx, &[], &[(0, 0)], None);
        assert!(builder.current_pos() > start);
    }

    #[test]
    #[should_panic(expected = "passes 1 int / 0 ref / 0 float argument(s)")]
    fn a_canonical_inline_call_passing_the_wrong_argument_kind_is_rejected() {
        // The canonical varlist shape reaches the callee's register banks the
        // same way the grouped shape does, so it is held to the same
        // declaration: a callee taking one reference, handed one integer.
        let mut builder = JitCodeBuilder::new();
        let idx = builder.add_sub_jitcode(callee_declaring("r"));
        builder.canonical_inline_call(
            "inline_call_ir_v/dIR",
            idx,
            Some(&[0]),
            Some(&[]),
            None,
            None,
        );
    }

    #[test]
    fn a_canonical_inline_call_may_reorder_kinds_against_the_signature() {
        // `inline_call_ir_r` passes the integers first and the references
        // second, while the callee declares them in source order `(r, r, i)`.
        // The counts per kind agree, which is all the callee's register banks
        // require, so this emits.
        let mut builder = JitCodeBuilder::new();
        let idx = builder.add_sub_jitcode(callee_declaring("rri"));
        let start = builder.current_pos();
        builder.canonical_inline_call(
            "inline_call_ir_r/dIR>r",
            idx,
            Some(&[0]),
            Some(&[0, 1]),
            None,
            Some((JitArgKind::Ref, 0)),
        );
        assert!(builder.current_pos() > start);
    }

    #[test]
    fn an_inline_call_into_a_callee_that_states_no_signature_is_not_checked() {
        // A jitcode this assembler builds carries the default calldescr, so
        // its empty `arg_classes` says nothing about what it accepts and the
        // check has to stand aside rather than read it as nullary.
        let mut builder = JitCodeBuilder::new();
        let idx = builder.add_sub_jitcode(JitCodeBuilder::new().finish());
        let start = builder.current_pos();
        builder.inline_call_ir_v(idx, &[(0, 0)], &[], None);
        assert!(builder.current_pos() > start);
    }

    #[test]
    fn residual_call_void_canonical_emits_irf_for_mixed_kinds() {
        // — `residual_call_void_canonical_typed_args` writes the
        // canonical byte layout that `blackhole.rs`'s `handler_residual_call_irf_v`
        // reads: 1B funcptr_reg + (countI:1 + regI×N) + (countR:1 + regR×M)
        // + (countF:1 + regF×K) + descr:2.
        let mut builder = JitCodeBuilder::new();
        // Reserve some live registers per kind so num_regs_X > 0 at finish().
        builder.touch_call_arg(JitCallArg::int(2));
        builder.touch_call_arg(JitCallArg::reference(3));
        builder.touch_call_arg(JitCallArg::float(1));
        let calldescr = majit_jitcode::codewriter::jitcode::BhCallDescr::from_signature(
            "irf".to_string(),
            majit_ir::value::Type::Void,
            majit_ir::descr::EffectInfo::MOST_GENERAL,
        );
        let funcptr = 0xDEAD_BEEFi64;
        let args = [
            JitCallArg::int(2),
            JitCallArg::reference(3),
            JitCallArg::float(1),
        ];
        let start = builder.current_pos();
        builder.residual_call_void_canonical_typed_args(funcptr, &args, calldescr);
        let jitcode = builder.finish();
        // Opcode byte (start_instr) + funcptr_reg(1) + countI(1)+regI(1)
        // + countR(1)+regR(1) + countF(1)+regF(1) + descr(2) = 10 bytes.
        let bytes = &jitcode.code[start..start + 10];
        assert_eq!(bytes[0], jitcode::insns::BC_RESIDUAL_CALL_IRF_V);
        // funcptr_reg patched to num_regs_i + funcptr_const_idx (=0).
        assert_eq!(bytes[1], jitcode.num_regs_i() as u8);
        assert_eq!(bytes[2], 1); // countI
        assert_eq!(bytes[3], 2); // regI
        assert_eq!(bytes[4], 1); // countR
        assert_eq!(bytes[5], 3); // regR
        assert_eq!(bytes[6], 1); // countF
        assert_eq!(bytes[7], 1); // regF
        // descr u16 LE
        let descr_idx = (bytes[8] as u16) | ((bytes[9] as u16) << 8);
        let entry = &jitcode.exec.descrs[descr_idx as usize];
        let cd = entry.as_bh_descr().unwrap().as_calldescr();
        assert_eq!(cd.arg_classes, "irf");
    }

    #[test]
    fn residual_call_void_canonical_uses_r_variant_for_ref_only() {
        // No int / float args → BC_RESIDUAL_CALL_R_V; layout is
        // funcptr_reg(1) + countR(1) + regR×M + descr(2).
        let mut builder = JitCodeBuilder::new();
        builder.touch_call_arg(JitCallArg::reference(5));
        let calldescr = majit_jitcode::codewriter::jitcode::BhCallDescr::from_signature(
            "r".to_string(),
            majit_ir::value::Type::Void,
            majit_ir::descr::EffectInfo::MOST_GENERAL,
        );
        let start = builder.current_pos();
        builder.residual_call_void_canonical_typed_args(
            0x1234_5678,
            &[JitCallArg::reference(5)],
            calldescr,
        );
        let jitcode = builder.finish();
        let bytes = &jitcode.code[start..start + 6];
        assert_eq!(bytes[0], jitcode::insns::BC_RESIDUAL_CALL_R_V);
        // funcptr_reg post-regs slot.
        assert_eq!(bytes[1], jitcode.num_regs_i() as u8);
        assert_eq!(bytes[2], 1); // countR
        assert_eq!(bytes[3], 5); // regR
        // No int / float lists — descr immediately follows.
        let descr_idx = (bytes[4] as u16) | ((bytes[5] as u16) << 8);
        let entry = &jitcode.exec.descrs[descr_idx as usize];
        assert_eq!(entry.as_bh_descr().unwrap().as_calldescr().arg_classes, "r");
    }

    #[test]
    fn residual_call_void_canonical_uses_ir_variant_when_no_floats() {
        let mut builder = JitCodeBuilder::new();
        builder.touch_call_arg(JitCallArg::int(2));
        builder.touch_call_arg(JitCallArg::reference(3));
        let calldescr = majit_jitcode::codewriter::jitcode::BhCallDescr::from_signature(
            "ir".to_string(),
            majit_ir::value::Type::Void,
            majit_ir::descr::EffectInfo::MOST_GENERAL,
        );
        let start = builder.current_pos();
        builder.residual_call_void_canonical_typed_args(
            0xCAFEi64,
            &[JitCallArg::int(2), JitCallArg::reference(3)],
            calldescr,
        );
        let jitcode = builder.finish();
        let bytes = &jitcode.code[start..start + 8];
        assert_eq!(bytes[0], jitcode::insns::BC_RESIDUAL_CALL_IR_V);
        assert_eq!(bytes[1], jitcode.num_regs_i() as u8);
        assert_eq!(bytes[2], 1); // countI
        assert_eq!(bytes[3], 2); // regI
        assert_eq!(bytes[4], 1); // countR
        assert_eq!(bytes[5], 3); // regR
        // No float list. descr 2 bytes.
        let descr_idx = (bytes[6] as u16) | ((bytes[7] as u16) << 8);
        let entry = &jitcode.exec.descrs[descr_idx as usize];
        assert_eq!(
            entry.as_bh_descr().unwrap().as_calldescr().arg_classes,
            "ir"
        );
    }

    #[test]
    fn residual_call_int_canonical_emits_irf_with_dst_for_mixed_kinds() {
        // — `residual_call_int_canonical_typed_args`
        // writes the same byte layout as the void IRF case, plus a
        // trailing `dst:u8` byte (`>i` suffix in `wellknown_bh_insns`).
        let mut builder = JitCodeBuilder::new();
        builder.touch_call_arg(JitCallArg::int(2));
        builder.touch_call_arg(JitCallArg::reference(3));
        builder.touch_call_arg(JitCallArg::float(1));
        builder.touch_reg(7); // dst slot
        let calldescr = majit_jitcode::codewriter::jitcode::BhCallDescr::from_signature(
            "irf".to_string(),
            majit_ir::value::Type::Int,
            majit_ir::descr::EffectInfo::MOST_GENERAL,
        );
        let start = builder.current_pos();
        builder.residual_call_int_canonical_typed_args(
            0xDEAD_BEEFi64,
            &[
                JitCallArg::int(2),
                JitCallArg::reference(3),
                JitCallArg::float(1),
            ],
            calldescr,
            7,
        );
        let jitcode = builder.finish();
        // Opcode + funcptr_reg + countI + regI + countR + regR + countF
        // + regF + descr×2 + dst = 11 bytes.
        let bytes = &jitcode.code[start..start + 11];
        assert_eq!(bytes[0], jitcode::insns::BC_RESIDUAL_CALL_IRF_I);
        assert_eq!(bytes[1], jitcode.num_regs_i() as u8);
        assert_eq!(bytes[2], 1); // countI
        assert_eq!(bytes[3], 2); // regI
        assert_eq!(bytes[4], 1); // countR
        assert_eq!(bytes[5], 3); // regR
        assert_eq!(bytes[6], 1); // countF
        assert_eq!(bytes[7], 1); // regF
        let descr_idx = (bytes[8] as u16) | ((bytes[9] as u16) << 8);
        assert_eq!(bytes[10], 7); // dst
        let entry = &jitcode.exec.descrs[descr_idx as usize];
        let cd = entry.as_bh_descr().unwrap().as_calldescr();
        assert_eq!(cd.arg_classes, "irf");
        assert_eq!(cd.result_type, 'i');
    }

    #[test]
    fn residual_call_float_canonical_always_uses_irf_even_for_int_only_args() {
        // Per `resoperation.py call_release_gil_for_descr`, float-result residual_calls
        // only have an IRF form — the R / IR shapes are "no such thing".
        // `emit_canonical_call_typed_irf_f` therefore always emits all
        // three (count, regs) pairs, even when a list is empty, so the
        // canonical handler `blackhole.rs`'s
        // `handler_residual_call_irf_f` can read `read_list_i` /
        // `read_list_r` / `read_list_f` in sequence.
        let mut builder = JitCodeBuilder::new();
        builder.touch_call_arg(JitCallArg::int(2));
        builder.touch_float_reg(4); // dst float slot
        let calldescr = majit_jitcode::codewriter::jitcode::BhCallDescr::from_signature(
            "i".to_string(),
            majit_ir::value::Type::Float,
            majit_ir::descr::EffectInfo::MOST_GENERAL,
        );
        let start = builder.current_pos();
        builder.residual_call_float_canonical_typed_args(
            0xCAFE_FACEi64,
            &[JitCallArg::int(2)],
            calldescr,
            4,
        );
        let jitcode = builder.finish();
        // Opcode(1) + funcptr_reg(1) + countI(1) + regI(1) + countR(1)
        // + countF(1) + descr(2) + dst(1) = 9 bytes for an int-only
        // float-returning call.
        let bytes = &jitcode.code[start..start + 9];
        assert_eq!(bytes[0], jitcode::insns::BC_RESIDUAL_CALL_IRF_F);
        assert_eq!(bytes[1], jitcode.num_regs_i() as u8);
        assert_eq!(bytes[2], 1); // countI
        assert_eq!(bytes[3], 2); // regI
        assert_eq!(bytes[4], 0); // countR — no ref args
        assert_eq!(bytes[5], 0); // countF — no float args
        let descr_idx = (bytes[6] as u16) | ((bytes[7] as u16) << 8);
        assert_eq!(bytes[8], 4); // dst float reg
        let entry = &jitcode.exec.descrs[descr_idx as usize];
        let cd = entry.as_bh_descr().unwrap().as_calldescr();
        assert_eq!(cd.arg_classes, "i");
        assert_eq!(cd.result_type, 'f');
    }

    #[test]
    fn residual_call_void_via_target_keeps_source_arg_classes() {
        let mut builder = JitCodeBuilder::new();
        let trace_ptr = 0x1111usize as *const ();
        let concrete_ptr = 0x2222usize as *const ();
        let fn_idx = builder.add_call_target(trace_ptr, concrete_ptr);
        let start = builder.current_pos();
        builder.residual_call_void_canonical_via_target(
            fn_idx,
            &[JitCallArg::reference(1), JitCallArg::int(2)],
        );

        let jitcode = builder.finish();
        let bytes = &jitcode.code[start..start + 8];
        assert_eq!(bytes[0], jitcode::insns::BC_RESIDUAL_CALL_IR_V);
        assert_eq!(bytes[2], 1); // countI
        assert_eq!(bytes[3], 2); // regI: grouped before refs
        assert_eq!(bytes[4], 1); // countR
        assert_eq!(bytes[5], 1); // regR
        let descr_idx = (bytes[6] as u16) | ((bytes[7] as u16) << 8);
        let entry = &jitcode.exec.descrs[descr_idx as usize];
        assert_eq!(
            entry.as_bh_descr().unwrap().as_calldescr().arg_classes,
            "ri"
        );
        assert_eq!(
            jitcode.exec.call_descr_to_call_target.get(&descr_idx),
            Some(&JitCallTarget::new(trace_ptr, concrete_ptr))
        );
    }

    #[test]
    fn residual_call_target_bridge_is_keyed_per_calldescr() {
        let mut builder = JitCodeBuilder::new();
        let concrete_ptr = 0x3333usize as *const ();
        let first_idx = builder.add_call_target(0x4444usize as *const (), concrete_ptr);
        let second_idx = builder.add_call_target(0x5555usize as *const (), concrete_ptr);
        let first_start = builder.current_pos();
        builder.residual_call_void_canonical_via_target(first_idx, &[]);
        let second_start = builder.current_pos();
        builder.residual_call_void_canonical_via_target(second_idx, &[]);

        let jitcode = builder.finish();
        let first_descr =
            (jitcode.code[first_start + 3] as u16) | ((jitcode.code[first_start + 4] as u16) << 8);
        let second_descr = (jitcode.code[second_start + 3] as u16)
            | ((jitcode.code[second_start + 4] as u16) << 8);
        assert_ne!(first_descr, second_descr);
        assert_eq!(
            jitcode
                .exec
                .call_descr_to_call_target
                .get(&first_descr)
                .unwrap()
                .trace_ptr,
            0x4444usize as *const ()
        );
        assert_eq!(
            jitcode
                .exec
                .call_descr_to_call_target
                .get(&second_descr)
                .unwrap()
                .trace_ptr,
            0x5555usize as *const ()
        );
    }

    #[test]
    fn typed_call_adapters_record_resulttypes_at_end_pc() {
        assert_resulttype_after(
            |b| {
                let idx = b.add_fn_ptr(std::ptr::null());
                b.conditional_call_value_ir_i_typed_args(idx, 1, &[], 2);
            },
            'i',
        );
        assert_resulttype_after(
            |b| {
                let idx = b.add_fn_ptr(std::ptr::null());
                b.conditional_call_value_ir_r_typed_args(idx, 1, &[], 2);
            },
            'r',
        );
    }

    #[test]
    fn conditional_call_ir_v_emits_canonical_iiird_bytes() {
        let mut builder = JitCodeBuilder::new();
        let idx = builder.add_fn_ptr(0x1111 as *const ());
        builder.conditional_call_ir_v_typed_args(
            idx,
            0,
            &[JitCallArg::int(1), JitCallArg::reference(0)],
        );
        let jitcode = builder.finish();
        assert_eq!(
            jitcode.code[0],
            jitcode::insns::BC_CONDITIONAL_CALL_IR_V,
            "helper-side cond_call must emit conditional_call_ir_v/iiIRd"
        );
        assert_ne!(jitcode.code[0], jitcode::insns::BC_COND_CALL_VOID);
        // lead cond, funcptr-placeholder, I count=1, i1, R count=1, r0, descr u16
        assert_eq!(jitcode.code[1], 0);
        assert_eq!(jitcode.code[3], 1);
        assert_eq!(jitcode.code[4], 1);
        assert_eq!(jitcode.code[5], 1);
        assert_eq!(jitcode.code[6], 0);
    }

    #[test]
    fn typed_void_call_adapters_do_not_record_resulttypes() {
        assert_no_resulttype_after(|b| {
            let idx = b.add_fn_ptr(std::ptr::null());
            b.residual_call_void_canonical_via_target(idx, &[]);
        });
    }

    #[test]
    fn live_writes_opcode_byte_then_two_offset_bytes() {
        // RPython assembler.py:146-158 — `live/` opcode followed by the
        // 2-byte offset returned by `_encode_liveness`.  The first call
        // for a never-seen liveness key must produce offset 0; the next
        // call with the same key must reuse offset 0 (dedup).
        let mut asm = Assembler::new();
        let mut builder = JitCodeBuilder::new();

        builder.live(&mut asm, &[0, 1], &[2], &[]);
        assert_eq!(builder.code.len(), 1 + 2, "opcode + u16 offset");
        // u16-LE 0 means the canonical entry for this key sits at the
        // start of `Assembler::all_liveness`.
        assert_eq!(builder.code[1], 0);
        assert_eq!(builder.code[2], 0);

        builder.live(&mut asm, &[0, 1], &[2], &[]);
        assert_eq!(
            builder.code.len(),
            (1 + 2) * 2,
            "second call appends another opcode + offset, no all_liveness growth"
        );
        // assembler.py:236-238 dedup: identical key reuses the same
        // 0-offset entry, so the second pair of offset bytes is also 0.
        assert_eq!(builder.code[4], 0);
        assert_eq!(builder.code[5], 0);

        // assembler.py `self.all_liveness = []` — only one canonical
        // payload exists despite two LIVE op emissions.
        let three_header_bytes = 3;
        let live_i_payload = 1; // [0, 1] fits in one bitset byte
        let live_r_payload = 1; // [2] fits in one bitset byte
        let live_f_payload = 0; // empty bitset → 0 bytes
        assert_eq!(
            asm.all_liveness().len(),
            three_header_bytes + live_i_payload + live_r_payload + live_f_payload,
        );
    }

    #[test]
    fn live_distinct_keys_advance_offset() {
        let mut asm = Assembler::new();
        let mut builder = JitCodeBuilder::new();

        builder.live(&mut asm, &[0], &[], &[]);
        let first_offset = u16::from_le_bytes([builder.code[1], builder.code[2]]);
        assert_eq!(first_offset, 0);

        builder.live(&mut asm, &[5], &[], &[]);
        let second_offset = u16::from_le_bytes([builder.code[4], builder.code[5]]);
        assert!(
            second_offset > 0,
            "distinct live_i key must advance into a fresh `all_liveness` entry"
        );
        assert_eq!(
            second_offset as usize,
            3 + 1,
            "first entry occupies 3 header + 1 payload byte"
        );
    }

    #[test]
    fn state_field_canonical_slots_empty_state() {
        // No scalars, no arrays, no virt arrays — empty triple.
        let (live_i, live_r, live_f) =
            super::live_slots_for_state_field_jit(0, &[], 0, 0, 0, 0, 0, 0);
        assert!(live_i.is_empty());
        assert!(live_r.is_empty());
        assert!(live_f.is_empty());
    }

    #[test]
    fn state_field_canonical_slots_ref_scalars_populate_live_r() {
        // 2 int scalars + 2 ref scalars at base 0: live_i = [0,1] (int
        // bank), live_r = [0,1] (ref bank), live_f empty.
        let (live_i, live_r, live_f) =
            super::live_slots_for_state_field_jit(2, &[], 0, 2, 0, 0, 0, 0);
        assert_eq!(live_i, vec![0u8, 1]);
        assert_eq!(live_r, vec![0u8, 1]);
        assert!(live_f.is_empty());
    }

    #[test]
    fn state_field_canonical_slots_ref_scalars_offset_by_base() {
        // ref_scalar_base shifts the ref bank past the dispatch JitCode's
        // ref arguments (program at r0): 2 ref scalars at base 1 occupy
        // r1..r3 while the int bank stays 0-based.
        let (live_i, live_r, live_f) =
            super::live_slots_for_state_field_jit(2, &[], 0, 2, 1, 0, 0, 0);
        assert_eq!(live_i, vec![0u8, 1]);
        assert_eq!(live_r, vec![1u8, 2]);
        assert!(live_f.is_empty());
    }

    #[test]
    fn state_field_canonical_slots_int_scalars_offset_by_base() {
        // int_scalar_base shifts the int bank past the dispatch JitCode's
        // int arguments (pc at i0): 2 int scalars at base 1 occupy
        // i1..i3 while the ref bank stays at its own base.
        let (live_i, live_r, live_f) =
            super::live_slots_for_state_field_jit(2, &[], 0, 2, 1, 0, 0, 1);
        assert_eq!(live_i, vec![1u8, 2]);
        assert_eq!(live_r, vec![1u8, 2]);
        assert!(live_f.is_empty());
    }

    #[test]
    fn state_field_canonical_slots_mixed_layout() {
        // 1 scalar + a 3-element flattened array in the int bank, plus a
        // virtualizable identity in the ref bank (portal r1, with
        // `ref_scalar_base` = 2 past `program` at r0 and the identity).
        let (live_i, live_r, live_f) =
            super::live_slots_for_state_field_jit(1, &[3], 1, 0, 2, 0, 0, 0);
        assert_eq!(live_i, vec![0u8, 1, 2, 3]);
        assert_eq!(live_r, vec![1u8]);
        assert!(live_f.is_empty());
    }

    #[test]
    fn state_field_canonical_slots_fills_max_index_255() {
        // End-exclusive boundary: base 1 + 255 scalars → int_scalar_end = 256,
        // filling indices i1..=i255 (the highest u8). This must NOT panic.
        let (live_i, live_r, live_f) =
            super::live_slots_for_state_field_jit(255, &[], 0, 0, 0, 0, 0, 1);
        assert_eq!(live_i.len(), 255);
        assert_eq!(*live_i.last().unwrap(), 255u8);
        assert!(live_r.is_empty());
        assert!(live_f.is_empty());
    }

    #[test]
    fn state_field_canonical_slots_feeds_assembler_encode() {
        // The triple plumbs straight into `Assembler::_encode_liveness`
        // / `JitCodeBuilder::live` without further reshaping.  Two
        // distinct shapes must dedup or advance through `all_liveness`
        // exactly as `live_writes_opcode_byte_then_two_offset_bytes`
        // / `live_distinct_keys_advance_offset` cover for hand-built
        // triples.
        let mut asm = Assembler::new();
        let mut builder = JitCodeBuilder::new();
        let (li, lr, lf) = super::live_slots_for_state_field_jit(2, &[1], 0, 0, 0, 0, 0, 0);
        builder.live(&mut asm, &li, &lr, &lf);
        // Header bytes = 3 (len_i, len_r, len_f).  live_i has 3 indices
        // (0, 1, 2) all in the first bitset byte.  live_r/live_f empty.
        assert_eq!(asm.all_liveness().len(), 3 + 1);
        assert_eq!(asm.all_liveness()[0], 3, "len(live_i) header byte");
        assert_eq!(asm.all_liveness()[1], 0, "len(live_r) header byte");
        assert_eq!(asm.all_liveness()[2], 0, "len(live_f) header byte");
    }

    #[test]
    #[should_panic(expected = "exceeds RPython jitcode u8 register-index limit")]
    fn state_field_canonical_slots_panics_on_overflow() {
        // 257 slots needs index 256, past the u8 register-index ceiling the
        // jitcode bytes are written through (assembler.py:241); 256 slots
        // (end-exclusive, indices 0..=255) is the valid maximum.
        let _ = super::live_slots_for_state_field_jit(257, &[], 0, 0, 0, 0, 0, 0);
    }

    #[test]
    fn live_placeholder_with_triple_records_then_finalize_patches() {
        // B.3-B.4 deferred-patch round-trip: `live_placeholder_with_triple`
        // emits the same `live/<00 00>` shape as `live_placeholder` but
        // additionally captures the triple in `pending_live_triples`.
        // `finalize_liveness` then registers each triple via
        // `_register_liveness_offset` and rewrites the BC_LIVE 2-byte slot
        // to point at the dedup'd entry — equivalent to the in-line
        // `live(asm, ...)` encoding shape, deferred.
        let mut asm = Assembler::new();
        let mut builder = JitCodeBuilder::new();

        let _ = builder.live_placeholder_with_triple(&[0, 1], &[2], &[]);
        // Pre-finalize: BC_LIVE byte + zero placeholder offset.
        assert_eq!(builder.code.len(), 1 + 2);
        assert_eq!(builder.code[1], 0);
        assert_eq!(builder.code[2], 0);
        assert_eq!(builder.pending_live_triples.len(), 1);

        builder.finalize_liveness(&mut asm);
        // Post-finalize: drained pending list, BC_LIVE slot patched to
        // point at the freshly-registered entry (offset 0 because it's
        // the first entry registered).
        assert!(builder.pending_live_triples.is_empty());
        assert_eq!(u16::from_le_bytes([builder.code[1], builder.code[2]]), 0);

        // A `live(asm, ...)` call with the same triple must dedup to the
        // same offset — no additional `all_liveness` growth.
        let pre_len = asm.all_liveness().len();
        let mut builder2 = JitCodeBuilder::new();
        builder2.live(&mut asm, &[0, 1], &[2], &[]);
        assert_eq!(asm.all_liveness().len(), pre_len);
        assert_eq!(u16::from_le_bytes([builder2.code[1], builder2.code[2]]), 0);
    }

    #[test]
    fn finalize_liveness_dedups_distinct_then_repeated_triples() {
        // Multiple pending triples: first distinct entries grow
        // all_liveness; repeats reuse offsets (assembler.py:235-238 dedup).
        let mut asm = Assembler::new();
        let mut builder = JitCodeBuilder::new();

        let pi0 = builder.live_placeholder_with_triple(&[0], &[], &[]);
        let pi1 = builder.live_placeholder_with_triple(&[5], &[], &[]);
        let pi2 = builder.live_placeholder_with_triple(&[0], &[], &[]); // repeat of pi0

        assert_eq!(builder.pending_live_triples.len(), 3);
        builder.finalize_liveness(&mut asm);

        let off0 = u16::from_le_bytes([builder.code[pi0], builder.code[pi0 + 1]]);
        let off1 = u16::from_le_bytes([builder.code[pi1], builder.code[pi1 + 1]]);
        let off2 = u16::from_le_bytes([builder.code[pi2], builder.code[pi2 + 1]]);

        assert_eq!(off0, 0, "first distinct entry sits at offset 0");
        assert_eq!(off1, 4, "second entry follows 3-header + 1 payload");
        assert_eq!(off2, off0, "repeat triple dedups to first entry");
    }

    #[test]
    fn finalize_liveness_is_idempotent_after_drain() {
        // Calling `finalize_liveness` twice must not double-patch (the
        // pending list is drained on first call; the second call is a
        // no-op).  Guards against accidental re-entry from caller flows
        // that loop or chain finalisation.
        let mut asm = Assembler::new();
        let mut builder = JitCodeBuilder::new();
        let patch = builder.live_placeholder_with_triple(&[3], &[], &[]);
        builder.finalize_liveness(&mut asm);
        let snapshot = builder.code.clone();
        let liveness_snapshot_len = asm.all_liveness().len();

        builder.finalize_liveness(&mut asm);
        assert_eq!(builder.code, snapshot, "second finalize is a no-op");
        assert_eq!(
            asm.all_liveness().len(),
            liveness_snapshot_len,
            "second finalize does not re-register"
        );
        // Sanity: patched offset is preserved.
        let off = u16::from_le_bytes([builder.code[patch], builder.code[patch + 1]]);
        assert_eq!(off, 0);
    }

    #[test]
    fn ref_return_const_encodes_constant_in_ref_window() {
        // `flatten.py make_return` emits `ref_return getcolor(v)`
        // where `getcolor` (`flatten.py`) passes a Constant
        // through unchanged; `assembler.py emit_const` encodes it
        // as the high register-index byte `num_regs_r + pool_idx`.  With
        // no ref registers touched, `num_regs_r == 0` and the first const
        // lands at pool slot 0, so the operand byte is 0.
        let mut builder = JitCodeBuilder::new();
        builder.ref_return_const(0x1234);
        let jitcode = builder.finish();
        let opcode = jitcode::insn_byte("ref_return/r");
        assert_eq!(jitcode.code, vec![opcode, 0]);
    }

    #[test]
    fn emit_raise_const_encodes_constant_in_ref_window() {
        // `make_exception_link` / `make_return` reraise paths emit
        // `raise getcolor(exc_value)` with a Constant exception instance;
        // the operand byte resolves to `num_regs_r + pool_idx` (0 here).
        let mut builder = JitCodeBuilder::new();
        builder.emit_raise_const(0x5678);
        let jitcode = builder.finish();
        let opcode = jitcode::insn_byte("raise/r");
        assert_eq!(jitcode.code, vec![opcode, 0]);
    }

    #[test]
    fn setarrayitem_gc_i_encodes_the_riid_operand_order() {
        // `handler_setarrayitem_gc_i` (blackhole.rs) reads
        // `code[position]` as the array ref reg, `+1` as the index int
        // reg, `+2` as the value int reg, then the descr — the `riid`
        // argcode string it is registered under. Pin that order here so
        // the emit side cannot drift from the handler that decodes it.
        let mut builder = JitCodeBuilder::new();
        let descr_idx = builder.add_env_int_array_descr(8, true);
        builder.setarrayitem_gc_i(1, 2, 3, descr_idx);
        let jitcode = builder.finish();
        let opcode = jitcode::insn_byte("setarrayitem_gc_i/riid");
        assert_eq!(
            jitcode.code,
            vec![
                opcode,
                1,
                2,
                3,
                (descr_idx & 0xff) as u8,
                (descr_idx >> 8) as u8
            ]
        );
        // The array operand is a REF reg and the other two are INT regs,
        // so the banks must be sized independently.
        assert_eq!(jitcode.c_num_regs_r, 2);
        assert_eq!(jitcode.c_num_regs_i, 4);
    }

    #[test]
    fn setarrayitem_gc_i_c_encodes_the_value_inline_not_the_index() {
        // The `_c` form's constant operand is the stored VALUE, not the
        // index — `setarrayitem_gc_i/ricd`, unlike `setarrayitem_gc_r`'s
        // `/rcrd` where it is the index. `handler_setarrayitem_gc_i_c`
        // reads `code[position + 1]` as an index REGISTER and
        // `code[position + 2]` as a signed inline byte.
        let mut builder = JitCodeBuilder::new();
        let descr_idx = builder.add_env_int_array_descr(8, true);
        builder.setarrayitem_gc_i_c(1, 2, -5, descr_idx);
        let jitcode = builder.finish();
        let opcode = jitcode::insn_byte("setarrayitem_gc_i/ricd");
        assert_eq!(
            jitcode.code,
            vec![
                opcode,
                1,
                2,
                (-5i8) as u8,
                (descr_idx & 0xff) as u8,
                (descr_idx >> 8) as u8
            ]
        );
        // Only the index is an int reg here; the value never touches the
        // int bank.
        assert_eq!(jitcode.c_num_regs_i, 3);
    }

    #[test]
    fn try_finish_accepts_last_single_byte_constant_slot() {
        let mut builder = JitCodeBuilder::new();
        builder.ensure_i_regs(1);
        for value in 0..255 {
            builder.add_const_i(value);
        }
        assert!(builder.try_finish().is_some());
    }

    #[test]
    fn try_finish_declines_register_constant_overflow() {
        let mut builder = JitCodeBuilder::new();
        builder.ensure_i_regs(1);
        for value in 0..256 {
            builder.add_const_i(value);
        }
        assert!(builder.try_finish().is_none());
    }

    /// `int_copy` is in `USE_C_FORM` (`assembler.py`), so a byte-sized
    /// constant is written inline rather than into `constants_i`.
    #[test]
    fn load_const_i_value_uses_the_short_form_for_byte_sized_values() {
        let mut builder = JitCodeBuilder::new();
        builder.ensure_i_regs(1);
        builder.load_const_i_value(0, -5);
        let jitcode = builder.try_finish().expect("short form must assemble");
        assert_eq!(
            jitcode.code,
            vec![
                majit_jitcode::codewriter::insns::insn_byte("int_copy/c>i"),
                (-5i8) as u8,
                0
            ]
        );
    }

    /// A value that does not fit a signed byte still goes through the pool
    /// as `int_copy/i>i`.
    #[test]
    fn load_const_i_value_pools_values_wider_than_a_byte() {
        let mut builder = JitCodeBuilder::new();
        builder.ensure_i_regs(1);
        builder.load_const_i_value(0, 1000);
        let jitcode = builder.try_finish().expect("pooled form must assemble");
        assert_eq!(
            jitcode.code[0],
            majit_jitcode::codewriter::insns::insn_byte("int_copy/i>i")
        );
    }

    /// A `Constant` operand is the register-space index of its constants
    /// slot, `num_regs_i + const_idx`, and a repeated value reuses the slot.
    #[test]
    fn a_word_cast_onto_its_own_register_emits_nothing_on_a_64_bit_word() {
        let mut builder = JitCodeBuilder::new();
        builder.ensure_i_regs(2);
        builder.cast_int_to_word(1, 1, false);
        builder.cast_int_to_word(1, 1, true);
        let empty = builder.code.is_empty();
        builder.cast_int_to_word(0, 1, false);
        let jitcode = builder.finish();
        if std::mem::size_of::<usize>() >= 8 {
            assert!(empty);
            let int_copy = crate::jitcode::wellknown_bh_insns()
                .into_iter()
                .find(|(name, _)| *name == "int_copy/i>i")
                .expect("int_copy/i>i is a wellknown insn")
                .1;
            assert_eq!(jitcode.code, vec![int_copy, 1, 0]);
            assert!(jitcode.constants_i.is_empty());
        } else {
            assert_eq!(jitcode.constants_i, vec![4, 0xFFFF_FFFF]);
        }
    }

    #[test]
    fn a_constant_operand_is_patched_to_its_constants_slot() {
        let mut builder = JitCodeBuilder::new();
        builder.ensure_i_regs(3);
        builder.record_binop_i_const(2, OpCode::IntSub, 1, 7);
        let label = builder.new_label();
        builder.goto_if_not_int_const(OpCode::IntEq, 2, 7, label);
        builder.mark_label(label);
        let jitcode = builder.try_finish().expect("must assemble");
        assert_eq!(jitcode.constants_i, vec![7]);
        assert_eq!(
            &jitcode.code[..4],
            &[
                majit_jitcode::codewriter::insns::insn_byte("int_sub/ii>i"),
                1,
                3,
                2
            ]
        );
        assert_eq!(
            &jitcode.code[4..7],
            &[
                majit_jitcode::codewriter::insns::insn_byte("goto_if_not_int_eq/iiL"),
                2,
                3
            ]
        );
    }

    /// Registers and constants of one kind share a single 256-wide index
    /// space.  Pooling every small constant pushes `total_i` past it and
    /// declines the JitCode; the short form keeps them out of the pool
    /// entirely, so a full byte range of distinct constants still assembles.
    #[test]
    fn a_full_byte_range_of_small_constants_still_assembles() {
        let mut builder = JitCodeBuilder::new();
        builder.ensure_i_regs(1);
        for value in -128..=127i64 {
            builder.load_const_i_value(0, value);
        }
        assert!(builder.try_finish().is_some());
    }

    fn fill_descr_pool(entries: usize) -> JitCodeBuilder {
        let mut builder = JitCodeBuilder::new();
        for _ in 0..entries {
            builder.push_descr_entry(RuntimeBhDescr::Call(Box::new(JitCallTarget::new(
                std::ptr::null(),
                std::ptr::null(),
            ))));
        }
        builder
    }

    /// A descr operand is two bytes wide, so `0xFFFF` is the last index a
    /// bytecode can name (`assembler.py:204`).
    #[test]
    fn try_finish_accepts_last_two_byte_descr_slot() {
        let builder = fill_descr_pool(u16::MAX as usize + 1);
        assert!(builder.try_finish().is_some());
    }

    /// `fix_labels` names a target with two unsigned bytes
    /// (`assert 0 <= target <= 0xFFFF`). A body longer than `i16::MAX`
    /// is still addressable.
    #[test]
    fn try_finish_accepts_code_longer_than_i16() {
        let mut builder = JitCodeBuilder::new();
        builder.code.resize(i16::MAX as usize + 1, 0);
        assert!(builder.try_finish().is_some());
    }

    /// One byte past the unsigned label target. `fix_labels` cannot name it.
    #[test]
    fn try_finish_declines_code_past_u16_label_target() {
        let mut builder = JitCodeBuilder::new();
        builder.code.resize(u16::MAX as usize + 1, 0);
        assert!(builder.try_finish().is_none());
    }

    /// One entry past the operand width. Upstream aborts translation with
    /// `assert 0 <= num <= 0xFFFF, "too many AbstractDescrs!"`; this builder
    /// runs at runtime, so it latches the overflow flag and declines the
    /// JitCode instead of wrapping the index with `as u16`.
    #[test]
    fn try_finish_declines_descr_pool_overflow() {
        let builder = fill_descr_pool(u16::MAX as usize + 2);
        assert!(builder.try_finish().is_none());
    }

    /// Temps past the one-byte register namespace latch `encoding_overflow`
    /// and emit nothing.
    #[test]
    fn alloc_int_temps_past_256_latches_overflow_and_skips_the_emit() {
        let mut floordiv = JitCodeBuilder::new();
        floordiv.touch_reg(251);
        let before = floordiv.code.len();
        floordiv.record_int_floordiv(0, 1, 2);
        assert!(floordiv.encoding_overflow);
        assert_eq!(floordiv.code.len(), before);

        let mut modulo = JitCodeBuilder::new();
        modulo.touch_reg(252);
        let before = modulo.code.len();
        modulo.record_int_mod(0, 1, 2);
        assert!(modulo.encoding_overflow);
        assert_eq!(modulo.code.len(), before);
    }

    /// `init_array_descr` `descr.tid` is the 7th argument, not a truncated
    /// `path_hash` left in `type_id`.
    #[test]
    fn add_gc_varsize_array_descr_stores_the_gc_type_id() {
        let mut builder = JitCodeBuilder::new();
        let idx = builder.add_gc_varsize_array_descr(8, 0, 8, false, true, 0xABCD, 22);
        match builder.finish().exec.descrs[idx as usize].as_bh_descr() {
            Some(CanonicalBhDescr::Array {
                gc_type_id,
                type_id,
                ..
            }) => {
                assert_eq!(*gc_type_id, 22);
                assert_eq!(*type_id, 0xABCD);
            }
            other => panic!("expected Array descr, got {other:?}"),
        }
    }

    #[test]
    fn add_gc_varsize_float_array_descr_stores_the_gc_type_id() {
        let mut builder = JitCodeBuilder::new();
        let idx = builder.add_gc_varsize_float_array_descr(8, 0, 0xBEEF, 23);
        match builder.finish().exec.descrs[idx as usize].as_bh_descr() {
            Some(CanonicalBhDescr::Array { gc_type_id, .. }) => {
                assert_eq!(*gc_type_id, 23);
            }
            other => panic!("expected Array descr, got {other:?}"),
        }
    }

    /// Two ARRAY types that share a layout still have distinct tids
    /// (`get_array_descr` keys on the lltype).
    #[test]
    fn array_descrs_with_distinct_gc_type_ids_do_not_share_a_slot() {
        let mut builder = JitCodeBuilder::new();
        let a = builder.add_gc_varsize_array_descr(8, 0, 8, false, true, 1, 22);
        let b = builder.add_gc_varsize_array_descr(8, 0, 8, false, true, 1, 23);
        assert_ne!(a, b);
    }
}
