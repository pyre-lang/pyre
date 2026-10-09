//! Line-by-line port of `rpython/jit/codewriter/assembler.py`.
//!
//! This module mirrors the high-level `Assembler` flow from RPython:
//! `setup()` → `write_insn()` → `fix_labels()` → `check_result()` →
//! `make_jitcode()`. pyre still targets `majit_metainterp::jitcode::JitCode`
//! and its fixed `BC_*` builder API, so regular-op dispatch lowers the
//! textual `SSARepr` opnames into `JitCodeBuilder` calls instead of growing
//! `self.insns` into a runtime opcode table.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, LazyLock};

use indexmap::IndexMap;
use majit_ir::IndexMapExt;
use majit_metainterp::jitcode::{JitCallArg, JitCode, JitCodeBuilder, JitCodeIntOperand};
use vecset::VecSet;

use super::flatten::{DescrOperand, Insn, Kind, ListOfKind, Operand, Register, SSARepr, TLabel};

/// `assembler.py` `count_regs = dict.fromkeys(KINDS, 0)` override.
#[derive(Debug, Clone, Copy, Default)]
pub struct NumRegs {
    pub int: u16,
    pub ref_: u16,
    pub float: u16,
}

/// `assembler.py` `class Assembler(object)`.
///
/// Writer-side state, per-instance, matching `Assembler.__init__`
/// (rpython/jit/codewriter/assembler.py) line-by-line. A fresh
/// `Assembler::new()` produces independent `all_liveness` /
/// `all_liveness_positions` / `num_liveness_ops` /
/// `indirectcalltargets`; upstream does the same — each `CodeWriter`
/// owns one `Assembler` and carries its own dedup state.
///
/// The reader side (`blackhole` / `trace_opcode`, both in the
/// lower `pyre_jit_trace` crate) still needs to read `all_liveness`
/// at runtime. pyre-jit-trace cannot depend on pyre-jit, so we
/// publish the latest `all_liveness` snapshot to
/// `pyre_jit_trace::assembler::ASSEMBLER_STATE` after every write
/// — Note for the crate layering, not a state
/// relocation.
#[derive(Debug, Default)]
pub struct Assembler {
    /// `assembler.py` `self.insns = {}`.
    ///
    /// RPython grows this dict in `write_insn()` with one opcode id per shape.
    /// Pyre reuses majit's reserved byte for a serialized well-known shape and
    /// lazily allocates an unreserved byte for a runtime-only shape such as a
    /// `USE_C_FORM` combination.  In both cases only shapes actually emitted
    /// by this assembler enter the map consumed by the lazy `finish_setup`
    /// cache refresh in `pyre_jit_trace::state`.
    insns: IndexMap<String, u8>,
    /// `assembler.py` `self.all_liveness = []`.
    all_liveness: Vec<u8>,
    /// `assembler.py` `self.all_liveness_length = 0`.
    all_liveness_length: usize,
    /// `assembler.py` `self.all_liveness_positions = {}`.
    all_liveness_positions: HashMap<(VecSet<u8>, VecSet<u8>, VecSet<u8>), u16>,
    /// `assembler.py` `self.num_liveness_ops = 0`.
    num_liveness_ops: usize,
    /// `assembler.py` `self.indirectcalltargets = set()    # set of JitCodes`.
    /// Accumulated across every `assemble()` call; `pyjitpl.py:2262`
    /// `self.setup_indirectcalltargets(asm.indirectcalltargets)` pipes
    /// this set to `MetaInterpStaticData.setup_indirectcalltargets`.
    pub indirectcalltargets: HashSet<ArcByPtr>,
    /// This assembler's identity for
    /// `pyre_jit_trace::assembler::publish_state_from`; `0` until the first
    /// publish assigns one.
    publish_token: u64,
}

/// Identity-keyed wrapper around `Arc<JitCode>` so `HashSet<ArcByPtr>`
/// matches RPython's `set of JitCodes` (Python set dedupes by object
/// identity, not value equality).  Used as the element type of
/// `Assembler::indirectcalltargets`.
///
/// The stored `JitCode` is still pyre's runtime adapter
/// `majit_metainterp::jitcode::JitCode`; once the runtime path consumes
/// the canonical codewriter `JitCode` directly, this wrapper should move
/// with that type rather than preserving a second object graph.
#[derive(Debug, Clone)]
pub struct ArcByPtr(pub Arc<JitCode>);

impl ArcByPtr {
    pub fn new(jc: Arc<JitCode>) -> Self {
        Self(jc)
    }

    pub fn into_inner(self) -> Arc<JitCode> {
        self.0
    }
}

impl PartialEq for ArcByPtr {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for ArcByPtr {}

impl std::hash::Hash for ArcByPtr {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (Arc::as_ptr(&self.0) as *const () as usize).hash(state);
    }
}

/// Builder-local state created by `setup()` for one `assemble()` call.
///
/// RPython keeps most of these on `self`; pyre keeps them in a per-call
/// struct because the runtime `JitCodeBuilder` is consumed by `finish()`.
struct AssemblyState {
    builder: JitCodeBuilder,
    /// `assembler.py` `self.label_positions = {}`.
    label_positions: IndexMap<String, usize>,
    /// Builder adapter for `Label/TLabel` name → builder label id.
    /// RPython stores bytecode positions directly in `label_positions`; this
    /// extra map exists only because `JitCodeBuilder` patches jumps by
    /// symbolic label id rather than by rewriting raw bytes in `fix_labels()`.
    builder_labels: IndexMap<String, u16>,
}

impl Assembler {
    pub fn new() -> Self {
        Self::default()
    }

    /// `assembler.py:19-32` `Assembler()` as `CodeWriter.__init__` builds it
    /// (codewriter.py), for a codewriter that continues the build-time
    /// drain rather than starting a fresh one.
    ///
    /// Upstream has one `Assembler` per program and `codewriter.py
    /// make_jitcodes` runs every pending graph through it, so `all_liveness`
    /// is a single offset space. pyre assembles the extracted interpreter
    /// graphs in `build.rs` and the Python-bytecode graphs at runtime; the
    /// build-time offsets are baked into those jitcodes' `-live-` operands,
    /// and a resume decodes them against `metainterp_sd.liveness_info`
    /// (`resume.py`). Seeding the buffer with the build-time bytes is
    /// what keeps that one offset space: the baked offsets stay addressable
    /// and `_encode_liveness` hands out positions strictly above them.
    ///
    /// The reader-side mirror seeds identically
    /// (`pyre_jit_trace::assembler::AssemblerState::new`), so the wholesale
    /// replace in `publish_state` never rewinds past this prefix.
    /// `assembler.py Assembler.__init__ self.insns = {}` is the other half of the same
    /// continuation, and it is load-bearing for the same reason. The build-time
    /// jitcodes' `-live-` markers are written with the canonical opcode byte,
    /// and `blackhole.py BlackholeInterpBuilder.__init__` recovers it as
    /// `asm.insns['live/']` — one dict, because upstream has one assembler.
    /// A runtime `insns` that starts empty leaves
    /// `MetaInterpStaticData.op_live` at its unset sentinel until some Python
    /// bytecode graph happens to assemble a `-live-`, and until then
    /// `JitCode::can_decode_live_vars` looks for the sentinel as a marker byte,
    /// finds none, and declines every resume against a build-time jitcode.
    /// That decline is silent and lossy: a jd1 drain guard that has already run
    /// its `self.next(w_iterator)` cannot be re-executed, so the fetched item
    /// is dropped instead of appended.
    ///
    /// Seeding is sound because the bytes are not allocated here — they come
    /// from the fixed `wellknown_bh_insns` / `extension_insns` tables, so
    /// the build-time and runtime spellings of an opname already agree on a
    /// byte; the map only records which opnames exist.
    pub fn resuming_build_time_liveness() -> Self {
        let all_liveness = pyre_jit_trace::jitcode_runtime::all_liveness().to_vec();
        let all_liveness_length = all_liveness.len();
        let insns = pyre_jit_trace::jitcode_runtime::insns_opname_to_byte().clone();
        // `assembler.py:31 self.all_liveness_positions` is the third half of
        // the same continuation. Seeding the bytes without it leaves the
        // resumed assembler blind to every record the prefix already holds, so
        // the first runtime `_encode_liveness` of a triple the build-time drain
        // already wrote appends a second copy instead of returning its offset.
        // The `-live-` operand is 2 bytes, so those duplicates come out of one
        // shared 64 KiB budget.
        let all_liveness_positions =
            majit_jitcode::liveness::decode_liveness_records(&all_liveness)
                .into_iter()
                .filter_map(|(live_i, live_r, live_f, offset)| {
                    let offset = u16::try_from(offset).ok()?;
                    Some((
                        (
                            live_i.into_iter().collect::<VecSet<u8>>(),
                            live_r.into_iter().collect::<VecSet<u8>>(),
                            live_f.into_iter().collect::<VecSet<u8>>(),
                        ),
                        offset,
                    ))
                })
                .collect();
        Self {
            insns,
            all_liveness,
            all_liveness_length,
            all_liveness_positions,
            ..Self::default()
        }
    }

    /// A process-unique non-zero token for this assembler, assigned on first
    /// use. The reader-side mirror appends only the new tail of a snapshot
    /// from the writer it last synced with, and this is how it tells writers
    /// apart.
    fn publish_token(&mut self) -> u64 {
        static NEXT_PUBLISH_TOKEN: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(1);
        if self.publish_token == 0 {
            self.publish_token =
                NEXT_PUBLISH_TOKEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        self.publish_token
    }

    /// `assembler.py:29` accessor.
    pub fn all_liveness(&self) -> &[u8] {
        &self.all_liveness
    }

    /// `assembler.py:30` accessor.
    pub fn all_liveness_length(&self) -> usize {
        self.all_liveness_length
    }

    /// Snapshot of the emitted well-known `opname/argcodes` keys.
    pub fn insns_snapshot(&self) -> IndexMap<String, u8> {
        self.insns.clone()
    }

    /// `assembler.py:32` accessor.
    pub fn num_liveness_ops(&self) -> usize {
        self.num_liveness_ops
    }

    /// Return the accumulated indirect-call target set as a deduped
    /// `Vec<Arc<JitCode>>`.  Matches the current runtime-adapter shape
    /// expected by `MetaInterpStaticData::setup_indirectcalltargets`;
    /// upstream parity for the dict-building semantics lives below that
    /// storage type, in `pyjitpl`'s fnaddr→jitcode helper.
    pub fn indirectcalltargets_vec(&self) -> Vec<Arc<JitCode>> {
        self.indirectcalltargets
            .iter()
            .map(|a| a.0.clone())
            .collect()
    }

    /// `assembler.py` `finished(self, callinfocollection)`.
    ///
    /// ```python
    /// def finished(self, callinfocollection):
    ///     # Helper called at the end of assembling.  Registers the extra
    ///     # functions shown in _callinfo_for_oopspec.
    ///     for func in callinfocollection.all_function_addresses_as_int():
    ///         func = int2adr(func)
    ///         self.see_raw_object(func.ptr)
    /// ```
    ///
    /// Called from `codewriter.py` after the `enum_pending_graphs`
    /// drain loop completes; registers raw helper-function addresses
    /// into `list_of_addr2name` for `MetaInterpStaticData`'s
    /// debug-symbol map.
    ///
    /// Note: pyre has no `oopspec` system, so the
    /// callinfocollection is empty (see
    /// [`super::call::CallInfoCollection`]) and the iteration is a
    /// no-op. The slot is preserved so `make_jitcodes` calls through
    /// at exactly the same point as `codewriter.py`.
    pub fn finished(&mut self, callinfocollection: &super::call::CallInfoCollection) {
        let _ = callinfocollection;
        pyre_jit_trace::assembler::publish_state_from(
            self.publish_token(),
            &self.insns,
            &self.all_liveness,
            self.all_liveness_length,
            self.num_liveness_ops,
        );
    }

    /// `assembler.py` `assemble(self, ssarepr, jitcode=None, num_regs=None)`.
    pub fn assemble(
        &mut self,
        ssarepr: &mut SSARepr,
        mut builder: JitCodeBuilder,
        num_regs: Option<NumRegs>,
    ) -> JitCode {
        self.try_assemble(ssarepr, builder, num_regs)
            .expect("Assembler::assemble called for an unencodable JitCode")
    }

    /// Checked assembly used by runtime graph construction.  The canonical
    /// `assembler.py` format has one-byte register/constant operands; `None`
    /// declines a graph whose per-kind namespace cannot be represented.
    pub fn try_assemble(
        &mut self,
        ssarepr: &mut SSARepr,
        mut builder: JitCodeBuilder,
        num_regs: Option<NumRegs>,
    ) -> Option<JitCode> {
        if let Some(nr) = num_regs {
            if nr.int >= 256 || nr.ref_ >= 256 || nr.float >= 256 {
                return None;
            }
            builder.ensure_i_regs(nr.int);
            builder.ensure_r_regs(nr.ref_);
            builder.ensure_f_regs(nr.float);
        }
        // Freeze num_regs at the regalloc-determined values: subsequent
        // operand emission must not grow the register file (every real
        // register slot is already inside `[0, num_regs_kind)` after
        // regalloc, and `Const*` args inside `ListOfKind` route through
        // the per-kind constants pool — see
        // `expect_int_reg_or_pool` and `dispatch_residual_call` for the
        // pool-byte synthesis that depends on stable `num_regs_kind`).
        builder.freeze_num_regs();
        builder.set_name(ssarepr.name.clone());

        let mut state = AssemblyState {
            builder,
            label_positions: IndexMap::new(),
            builder_labels: IndexMap::new(),
        };

        ssarepr.insns_pos = Some(Vec::with_capacity(ssarepr.insns.len()));
        for insn in &ssarepr.insns {
            ssarepr
                .insns_pos
                .as_mut()
                .expect("insns_pos initialized")
                .push(state.builder.current_pos());
            self.write_insn(&mut state, insn);
        }

        self.fix_labels(&mut state);

        let jitcode = state.builder.try_finish()?;
        dump_assembled_ssarepr(ssarepr, &jitcode);
        // `assembler.py` checks the completed register/constant namespace
        // before making the JitCode. Runtime-generated graphs use the checked
        // seam above; this assertion remains an invariant check on success.
        self.check_result(&jitcode);
        Some(jitcode)
    }

    /// `assembler.py` `write_insn(insn)`.
    fn write_insn(&mut self, state: &mut AssemblyState, insn: &Insn) {
        match insn {
            Insn::Unreachable => {}
            Insn::Label(label) => {
                mark_label_first_wins(state, &label.name);
            }
            Insn::Op {
                opname,
                args,
                result,
            } => {
                // `assembler.py` `-live-` branch. `liveness.py:5-12`
                // uses `insn[0] == '-live-'` as the discriminator; pyre
                // matches on `opname` to keep the tuple-shape parity.
                if opname == super::flatten::OPNAME_LIVE {
                    // `assembler.py:149` increments `self.num_liveness_ops`.
                    // pyre's counter lives on `ASSEMBLER_STATE`;
                    // `intern_liveness` below takes care of the `+= 1`, so
                    // there is no local pre-increment here (matching
                    // upstream: the bump happens inside `_encode_liveness`).
                    let live_i = get_liveness_info(args, Kind::Int);
                    let live_r = get_liveness_info(args, Kind::Ref);
                    let live_f = get_liveness_info(args, Kind::Float);
                    let patch_offset = state.builder.live_placeholder();
                    let offset = self.encode_liveness_info(&live_i, &live_r, &live_f);
                    state.builder.patch_live_offset(patch_offset, offset);
                    self.record_insn_key(opname, args, None);
                    return;
                }
                // `assembler.py:208-209`:
                //   elif isinstance(x, IndirectCallTargets):
                //       self.indirectcalltargets.update(x.lst)
                //
                // RPython's `write_insn` iterates every operand and dispatches
                // by Python type.  pyre's `dispatch_op` is keyed on `opname`
                // and matches no operand type at all, so operand iteration
                // is split: `IndirectCallTargets` is collected here before the
                // opcode-specific lowering runs.  The operand is purely
                // metadata for the call — it is not written into the jitcode
                // byte stream (RPython emits nothing for this branch either).
                for arg in args {
                    if let Operand::IndirectCallTargets(targets) = arg {
                        for jc in &targets.lst {
                            self.indirectcalltargets.insert(ArcByPtr::new(jc.clone()));
                        }
                    }
                }
                let opcode_byte = self.record_insn_key(opname, args, result.as_ref());
                // `assembler.py:197-206` registers descrs inline during op
                // encoding into `Assembler.descrs`. pyre's runtime
                // codewriter has no descr-consuming ops today, so no
                // registration path runs here — once a descr-consuming op
                // lands it must register onto the shared
                // `BlackholeInterpBuilder.descrs` pool (`blackhole.py:
                // 102-103 setup_descrs`), not a per-jitcode duplicate.
                CURRENT_DISPATCH_OP.with(|cur| cur.borrow_mut().replace_range(.., opname));
                dispatch_op(state, opname, args, result.as_ref(), opcode_byte);
            }
        }
    }

    /// `assembler.py` `fix_labels()`.
    ///
    /// Note: pyre's runtime `JitCodeBuilder` patches
    /// jumps by symbolic label id rather than by rewriting raw bytes, so
    /// the label-position loop is folded into the builder. The per-descr
    /// switch patching that upstream runs here is handled through the
    /// shared `BlackholeInterpBuilder.descrs` pool (`blackhole.py`
    /// `setup_descrs`), not per-jitcode — pyre's runtime codewriter has
    /// no descr-consuming ops yet, so this is currently a no-op.
    fn fix_labels(&mut self, _state: &mut AssemblyState) {}

    /// `assembler.py` `check_result()`.
    ///
    /// RPython enforces a 256-entry cap per kind because the assembled
    /// bytecode stream treats register-or-constant operands as one-byte
    /// indices. The surrounding pyre pipeline still relies on the same
    /// invariant for liveness and jit-merge-point operand encoding, so we
    /// keep the orthodox translator-time assertion here instead of
    /// widening the limit to match internal `u16` builder fields.
    ///
    /// Upstream can assert because its jitcodes come from RPython functions,
    /// where a violation is a translation-time bug. Pyre assembles arbitrary
    /// user bytecode at runtime, so the enforcing copy of this rule is
    /// `JitCodeBuilder::try_finish`, which tests the same three inequalities
    /// and returns `None` — the caller above declines the graph and leaves the
    /// function to the interpreter. That runs first, so by the time a
    /// `JitCode` reaches here the three asserts are already satisfied: this is
    /// a post-condition restating the rule at its orthodox position, not the
    /// gate that keeps an unencodable frame out.
    fn check_result(&self, jitcode: &JitCode) {
        assert!(
            (jitcode.num_regs_i() as usize) + jitcode.constants_i.len() <= 256,
            "too many int registers/constants"
        );
        assert!(
            (jitcode.num_regs_r() as usize) + jitcode.constants_r.len() <= 256,
            "too many ref registers/constants"
        );
        assert!(
            (jitcode.num_regs_f() as usize) + jitcode.constants_f.len() <= 256,
            "too many float registers/constants"
        );
    }

    /// `assembler.py` `_encode_liveness(...)`.
    ///
    /// Line-by-line port of RPython's `_encode_liveness`:
    ///   - Dedup key is `(frozenset(live_i), frozenset(live_r),
    ///     frozenset(live_f))`. Pyre carries each kind as a `VecSet<u8>`
    ///     (Vec-backed sorted set), and the cache uses the same
    ///     `(VecSet, VecSet, VecSet)` triple as its `HashMap` key.
    ///   - On miss: append the three u8 length bytes + the encoded
    ///     bitsets, and advance `all_liveness_length`.
    ///   - `num_liveness_ops` bumps once per `-live-` write regardless of
    ///     dedup (`assembler.py` does it in `write_insn`, before this
    ///     helper is reached). Matches upstream.
    ///
    /// After the per-instance update we publish the latest buffer to
    /// the `pyre_jit_trace::assembler::ASSEMBLER_STATE` thread-local so
    /// the blackhole reader (a lower-crate consumer that cannot depend
    /// on `pyre_jit`) sees the same bytes. Note for
    /// the crate layering, not a relocation of the RPython state.
    fn encode_liveness_info(
        &mut self,
        live_i: &VecSet<u8>,
        live_r: &VecSet<u8>,
        live_f: &VecSet<u8>,
    ) -> u16 {
        use majit_jitcode::liveness::encode_liveness;

        self.num_liveness_ops += 1;
        let key = (live_i.clone(), live_r.clone(), live_f.clone());
        let pos = if let Some(&cached) = self.all_liveness_positions.get(&key) {
            cached
        } else {
            // The 2-byte `-live-` operand caps the buffer at 64 KiB, upstream
            // included. pyre spends the low end of that budget on the
            // build-time drain's bytes (`resuming_build_time_liveness`), so
            // name both halves — an overflow here is "the shared pool filled
            // up", not "the runtime codewriter alone ran away".
            assert!(
                self.all_liveness_length <= u16::MAX as usize,
                "all_liveness offset overflow at {} bytes ({} of them the \
                 build-time drain's prefix); a `-live-` operand is 2 bytes",
                self.all_liveness_length,
                pyre_jit_trace::jitcode_runtime::all_liveness().len(),
            );
            assert!(
                live_i.len() <= u8::MAX as usize
                    && live_r.len() <= u8::MAX as usize
                    && live_f.len() <= u8::MAX as usize,
                "too many live registers in one slot"
            );
            let pos = self.all_liveness_length as u16;
            self.all_liveness.push(live_i.len() as u8);
            self.all_liveness.push(live_r.len() as u8);
            self.all_liveness.push(live_f.len() as u8);
            self.all_liveness_length += 3;
            for live in [live_i.as_slice(), live_r.as_slice(), live_f.as_slice()] {
                let encoded = encode_liveness(live);
                self.all_liveness.extend_from_slice(&encoded);
                self.all_liveness_length += encoded.len();
            }
            self.all_liveness_positions.insert(key, pos);
            pos
        };
        pyre_jit_trace::assembler::publish_state_from(
            self.publish_token(),
            &self.insns,
            &self.all_liveness,
            self.all_liveness_length,
            self.num_liveness_ops,
        );
        pos
    }

    fn record_insn_key(
        &mut self,
        opname: &str,
        args: &[Operand],
        result: Option<&Register>,
    ) -> Option<u8> {
        let key = if opname == super::flatten::OPNAME_LIVE {
            "live/".to_string()
        } else {
            insn_key(opname, args, result)
        };
        if let Some(&opcode) = self.insns.get(&key) {
            return Some(opcode);
        }
        let opcode = WELLKNOWN_BH_INSNS
            .get(key.as_str())
            .copied()
            .unwrap_or_else(|| {
                // assembler.py `self.insns.setdefault(key, len(self.insns))`,
                // adjusted only for pyre's serialized fixed-byte reservation.
                // Allocate an opcode for the shape that actually appeared; do not
                // reserve the Cartesian product of every USE_C_FORM variant.
                (0u16..=u8::MAX as u16)
                    .map(|value| value as u8)
                    .find(|value| {
                        !majit_jitcode::insns::is_reserved_opcode_byte(*value)
                            && !self.insns.values().any(|used| used == value)
                    })
                    .unwrap_or_else(|| panic!("runtime assembler opcode space exhausted for {key}"))
            });
        // The byte has to reach the trace decoder as well: `decode_op_at`
        // resolves every opcode through the byte -> opname map, and a byte only
        // this table knows makes the body undecodable, which its callers read
        // as "assume the worst".
        pyre_jit_trace::jitcode_runtime::publish_runtime_insn(&key, opcode);
        self.insns.insert(key, opcode);
        Some(opcode)
    }
}

static WELLKNOWN_BH_INSNS: LazyLock<indexmap::IndexMap<&'static str, u8>> =
    LazyLock::new(majit_metainterp::jitcode::wellknown_bh_insns);

fn insn_key(opname: &str, args: &[Operand], result: Option<&Register>) -> String {
    // `jtransform.rewrite_call(..., force_ir=True)` always emits these
    // keys. Helper-side SSA still spells the callee as a ConstInt
    // `fn_ptr_idx`; the JitCodeBuilder projects that to the canonical
    // `iiIRd` / `riIRd` byte layout.
    match opname {
        "conditional_call_ir_v" => return "conditional_call_ir_v/iiIRd".to_string(),
        "conditional_call_value_ir_i" => {
            return "conditional_call_value_ir_i/iiIRd>i".to_string();
        }
        "conditional_call_value_ir_r" => {
            return "conditional_call_value_ir_r/riIRd>r".to_string();
        }
        "record_known_result_i_ir_v" => return "record_known_result_i_ir_v/iiIRd".to_string(),
        "record_known_result_r_ir_v" => return "record_known_result_r_ir_v/riIRd".to_string(),
        _ => {}
    }
    if opname == "int_copy" {
        let src_code = match args.first() {
            Some(Operand::ConstInt(value)) if (-128..=127).contains(value) => 'c',
            Some(Operand::ConstInt(value)) if u16::try_from(*value).is_ok() => 'd',
            Some(Operand::ConstInt(_)) => 'i',
            Some(Operand::Register(reg)) if reg.kind == Kind::Int => 'i',
            other => panic!("int_copy: expected Int register or ConstInt, got {other:?}"),
        };
        return format!("int_copy/{src_code}>i");
    }
    if opname == "jit_merge_point" {
        let jdindex_argcode = match args {
            [Operand::ConstInt(value), ..] if (-128..=127).contains(value) => 'c',
            [Operand::ConstInt(_), ..] => 'i',
            _ => panic!(
                "jit_merge_point: expected 7-arg shape with ConstInt jd_index at arg[0]; \
                 jtransform.py:1690-1712 always emits the canonical \
                 [jd_index, greens_i, greens_r, greens_f, reds_i, reds_r, reds_f] form"
            ),
        };
        return format!("jit_merge_point/{jdindex_argcode}IRFIRF");
    }

    let mut argcodes = String::new();
    let allow_short = use_c_form(opname);
    for arg in args {
        match arg {
            Operand::Register(reg) => argcodes.push(kind_char(reg.kind)),
            Operand::ConstInt(value) => {
                let code = if allow_short && (-128..=127).contains(value) {
                    'c'
                } else {
                    'i'
                };
                argcodes.push(code);
            }
            Operand::ConstRef(_) => argcodes.push('r'),
            Operand::ConstFloat(_) => argcodes.push('f'),
            Operand::TLabel(_) => argcodes.push('L'),
            Operand::ListOfKind(list) => argcodes.push(kind_char(list.kind).to_ascii_uppercase()),
            Operand::Descr(_) => argcodes.push('d'),
            Operand::IndirectCallTargets(_) => {}
        }
    }
    if let Some(result) = result {
        argcodes.push('>');
        argcodes.push(kind_char(result.kind));
    }
    format!("{opname}/{argcodes}")
}

fn kind_char(kind: Kind) -> char {
    match kind {
        Kind::Int => 'i',
        Kind::Ref => 'r',
        Kind::Float => 'f',
    }
}

fn use_c_form(opname: &str) -> bool {
    matches!(
        opname,
        "copystrcontent"
            | "getarrayitem_gc_pure_i"
            | "getarrayitem_gc_pure_r"
            | "getarrayitem_gc_i"
            | "getarrayitem_gc_r"
            | "goto_if_not_int_eq"
            | "goto_if_not_int_ge"
            | "goto_if_not_int_gt"
            | "goto_if_not_int_le"
            | "goto_if_not_int_lt"
            | "goto_if_not_int_ne"
            | "int_add"
            | "int_and"
            | "int_copy"
            | "int_eq"
            | "int_ge"
            | "int_gt"
            | "int_le"
            | "int_lt"
            | "int_ne"
            | "int_return"
            | "int_sub"
            | "jit_merge_point"
            | "new_array"
            | "new_array_clear"
            | "newstr"
            | "setarrayitem_gc_i"
            | "setarrayitem_gc_r"
            | "setfield_gc_i"
            | "strgetitem"
            | "strsetitem"
            | "foobar"
            | "baz"
    )
}

/// Test-only convenience wrapper — creates a throwaway `Assembler` per
/// call. **Do not wire this into production paths.** RPython's
/// `CodeWriter` holds a single `Assembler` whose `all_liveness`,
/// `all_liveness_positions`, and `num_liveness_ops` accumulate across
/// every JitCode compiled in the session (`codewriter.py` /
/// `pyjitpl.py:2264`). A fresh `Assembler` per call splits the
/// global liveness table and breaks that invariant.
///
/// Production callers must construct one `Assembler` up front and call
/// `assembler.assemble(...)` repeatedly on it.
#[cfg(test)]
fn assemble(ssarepr: &mut SSARepr, builder: JitCodeBuilder, num_regs: Option<NumRegs>) -> JitCode {
    let mut assembler = Assembler::new();
    assembler.assemble(ssarepr, builder, num_regs)
}

/// Print the assembled instruction stream, byte position first, for graphs
/// whose name matches `MAJIT_DUMP_SSAREPR`.
///
/// A blackhole failure reports a raw `(jitcode, position)` pair; without the
/// stream there is no way back from that byte offset to the op that wrote it
/// or to the register operands it reads.  The env lookup is cached because
/// `try_assemble` runs per graph on the tracing path.
fn dump_assembled_ssarepr(ssarepr: &SSARepr, jitcode: &JitCode) {
    static WANT: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let want = WANT.get_or_init(|| {
        std::env::var_os("MAJIT_DUMP_SSAREPR").map(|v| v.to_string_lossy().into_owned())
    });
    let Some(want) = want else { return };
    if &ssarepr.name != want {
        return;
    }
    let positions = ssarepr
        .insns_pos
        .as_ref()
        .expect("insns_pos populated by the assembly loop");
    eprintln!(
        "-- ssarepr {} : {} insns, {} bytes, num_regs i={} r={} f={} --",
        ssarepr.name,
        ssarepr.insns.len(),
        jitcode.code.len(),
        jitcode.num_regs_i(),
        jitcode.num_regs_r(),
        jitcode.num_regs_f(),
    );
    for (idx, insn) in ssarepr.insns.iter().enumerate() {
        eprintln!("[{:5}] {:5}: {insn:?}", positions[idx], idx);
    }
}

fn builder_label(state: &mut AssemblyState, name: &str) -> u16 {
    if let Some(&label) = state.builder_labels.get(name) {
        return label;
    }
    let label = state.builder.new_label();
    state.builder_labels.insert(name.to_owned(), label);
    label
}

/// First-wins record of an `Insn::Label` byte position.  Duplicate
/// `Label(block)` emissions can survive a supersede re-walk under a
/// widened framestate when the walker's first emit for a block
/// becomes dead but its label still appears in the drained SSARepr;
/// first-wins keeps the dispatch byte position anchored to that
/// original emit so the assembler label slot agrees with downstream
/// resolvers.  Without this, `mark_label` would last-wins overwrite
/// the slot (`majit-metainterp/jitcode/assembler.rs::mark_label`
/// blindly sets `*slot = Some(code.len())`).
fn mark_label_first_wins(state: &mut AssemblyState, name: &str) {
    let label_id = builder_label(state, name);
    if state.label_positions.contains_key(name) {
        return;
    }
    state
        .label_positions
        .insert(name.to_owned(), state.builder.current_pos());
    state.builder.mark_label(label_id);
}

fn get_liveness_info(args: &[Operand], kind: Kind) -> VecSet<u8> {
    let mut lives = VecSet::new();
    for arg in args {
        if let Operand::Register(Register {
            kind: reg_kind,
            index,
        }) = arg
        {
            if *reg_kind == kind {
                lives.insert(u8::try_from(*index).expect("liveness register index exceeds u8"));
            }
        }
    }
    lives
}

fn dispatch_op(
    state: &mut AssemblyState,
    opname: &str,
    args: &[Operand],
    result: Option<&Register>,
    opcode_byte: Option<u8>,
) {
    match opname {
        "goto" => {
            let label = expect_tlabel(&args[0]);
            let label_id = builder_label(state, &label.name);
            state.builder.jump(label_id);
        }
        "switch" => {
            // `flatten.py` `GraphFlattener.insert_exits` `switch/id`: int
            // register plus SwitchDictDescr.  Missing keys fall through
            // (`blackhole.py` `bhimpl_switch`).
            assert_eq!(args.len(), 2, "switch expects [value, SwitchDictDescr]");
            let value = expect_reg(&args[0], Kind::Int);
            let cases: Vec<(i64, u16)> = match &args[1] {
                Operand::Descr(descr) => match &**descr {
                    DescrOperand::SwitchDict(switch) => switch
                        .labels
                        .iter()
                        .map(|(key, tlabel)| (*key, builder_label(state, &tlabel.name)))
                        .collect(),
                    other => panic!("switch descr must be SwitchDictDescr, got {other:?}"),
                },
                other => panic!("switch second arg must be Descr, got {other:?}"),
            };
            state.builder.switch(value, &cases);
        }
        "goto_if_not" => {
            let cond = expect_reg(&args[0], Kind::Int);
            let label = expect_tlabel(&args[1]);
            let label_id = builder_label(state, &label.name);
            state.builder.goto_if_not(cond, label_id);
        }
        "goto_if_not_int_is_true" => {
            let cond = expect_reg(&args[0], Kind::Int);
            let label = expect_tlabel(&args[1]);
            let label_id = builder_label(state, &label.name);
            state.builder.goto_if_not_int_is_true(cond, label_id);
        }
        "goto_if_not_int_eq" => {
            let lhs = expect_jitcode_int_operand(state, &args[0], true);
            let rhs = expect_jitcode_int_operand(state, &args[1], true);
            let label = expect_tlabel(&args[2]);
            let label_id = builder_label(state, &label.name);
            match (lhs, rhs) {
                (JitCodeIntOperand::Register(lhs), JitCodeIntOperand::Register(rhs))
                    if both_int_operands_are_registers(&args[0], &args[1]) =>
                {
                    state.builder.goto_if_not_int_eq(lhs, rhs, label_id)
                }
                _ => state.builder.goto_if_not_int_operands(
                    opcode_byte.expect("short goto_if_not_int_eq opcode"),
                    lhs,
                    rhs,
                    label_id,
                ),
            }
        }
        "goto_if_not_int_ne" => {
            let lhs = expect_jitcode_int_operand(state, &args[0], true);
            let rhs = expect_jitcode_int_operand(state, &args[1], true);
            let label = expect_tlabel(&args[2]);
            let label_id = builder_label(state, &label.name);
            match (lhs, rhs) {
                (JitCodeIntOperand::Register(lhs), JitCodeIntOperand::Register(rhs))
                    if both_int_operands_are_registers(&args[0], &args[1]) =>
                {
                    state.builder.goto_if_not_int_ne(lhs, rhs, label_id)
                }
                _ => state.builder.goto_if_not_int_operands(
                    opcode_byte.expect("short goto_if_not_int_ne opcode"),
                    lhs,
                    rhs,
                    label_id,
                ),
            }
        }
        "goto_if_not_int_lt" => {
            let lhs = expect_jitcode_int_operand(state, &args[0], true);
            let rhs = expect_jitcode_int_operand(state, &args[1], true);
            let label = expect_tlabel(&args[2]);
            let label_id = builder_label(state, &label.name);
            match (lhs, rhs) {
                (JitCodeIntOperand::Register(lhs), JitCodeIntOperand::Register(rhs))
                    if both_int_operands_are_registers(&args[0], &args[1]) =>
                {
                    state.builder.goto_if_not_int_lt(lhs, rhs, label_id)
                }
                _ => state.builder.goto_if_not_int_operands(
                    opcode_byte.expect("short goto_if_not_int_lt opcode"),
                    lhs,
                    rhs,
                    label_id,
                ),
            }
        }
        "goto_if_not_int_le" => {
            let lhs = expect_jitcode_int_operand(state, &args[0], true);
            let rhs = expect_jitcode_int_operand(state, &args[1], true);
            let label = expect_tlabel(&args[2]);
            let label_id = builder_label(state, &label.name);
            match (lhs, rhs) {
                (JitCodeIntOperand::Register(lhs), JitCodeIntOperand::Register(rhs))
                    if both_int_operands_are_registers(&args[0], &args[1]) =>
                {
                    state.builder.goto_if_not_int_le(lhs, rhs, label_id)
                }
                _ => state.builder.goto_if_not_int_operands(
                    opcode_byte.expect("short goto_if_not_int_le opcode"),
                    lhs,
                    rhs,
                    label_id,
                ),
            }
        }
        "goto_if_not_int_gt" => {
            let lhs = expect_jitcode_int_operand(state, &args[0], true);
            let rhs = expect_jitcode_int_operand(state, &args[1], true);
            let label = expect_tlabel(&args[2]);
            let label_id = builder_label(state, &label.name);
            match (lhs, rhs) {
                (JitCodeIntOperand::Register(lhs), JitCodeIntOperand::Register(rhs))
                    if both_int_operands_are_registers(&args[0], &args[1]) =>
                {
                    state.builder.goto_if_not_int_gt(lhs, rhs, label_id)
                }
                _ => state.builder.goto_if_not_int_operands(
                    opcode_byte.expect("short goto_if_not_int_gt opcode"),
                    lhs,
                    rhs,
                    label_id,
                ),
            }
        }
        "goto_if_not_int_ge" => {
            let lhs = expect_jitcode_int_operand(state, &args[0], true);
            let rhs = expect_jitcode_int_operand(state, &args[1], true);
            let label = expect_tlabel(&args[2]);
            let label_id = builder_label(state, &label.name);
            match (lhs, rhs) {
                (JitCodeIntOperand::Register(lhs), JitCodeIntOperand::Register(rhs))
                    if both_int_operands_are_registers(&args[0], &args[1]) =>
                {
                    state.builder.goto_if_not_int_ge(lhs, rhs, label_id)
                }
                _ => state.builder.goto_if_not_int_operands(
                    opcode_byte.expect("short goto_if_not_int_ge opcode"),
                    lhs,
                    rhs,
                    label_id,
                ),
            }
        }
        "goto_if_not_int_is_zero" => {
            let cond = expect_reg(&args[0], Kind::Int);
            let label = expect_tlabel(&args[1]);
            let label_id = builder_label(state, &label.name);
            state.builder.goto_if_not_int_is_zero(cond, label_id);
        }
        "goto_if_not_float_lt" => {
            let lhs = expect_reg(&args[0], Kind::Float);
            let rhs = expect_reg(&args[1], Kind::Float);
            let label = expect_tlabel(&args[2]);
            let label_id = builder_label(state, &label.name);
            state.builder.goto_if_not_float_lt(lhs, rhs, label_id);
        }
        "goto_if_not_float_le" => {
            let lhs = expect_reg(&args[0], Kind::Float);
            let rhs = expect_reg(&args[1], Kind::Float);
            let label = expect_tlabel(&args[2]);
            let label_id = builder_label(state, &label.name);
            state.builder.goto_if_not_float_le(lhs, rhs, label_id);
        }
        "goto_if_not_float_eq" => {
            let lhs = expect_reg(&args[0], Kind::Float);
            let rhs = expect_reg(&args[1], Kind::Float);
            let label = expect_tlabel(&args[2]);
            let label_id = builder_label(state, &label.name);
            state.builder.goto_if_not_float_eq(lhs, rhs, label_id);
        }
        "goto_if_not_float_ne" => {
            let lhs = expect_reg(&args[0], Kind::Float);
            let rhs = expect_reg(&args[1], Kind::Float);
            let label = expect_tlabel(&args[2]);
            let label_id = builder_label(state, &label.name);
            state.builder.goto_if_not_float_ne(lhs, rhs, label_id);
        }
        "goto_if_not_float_gt" => {
            let lhs = expect_reg(&args[0], Kind::Float);
            let rhs = expect_reg(&args[1], Kind::Float);
            let label = expect_tlabel(&args[2]);
            let label_id = builder_label(state, &label.name);
            state.builder.goto_if_not_float_gt(lhs, rhs, label_id);
        }
        "goto_if_not_float_ge" => {
            let lhs = expect_reg(&args[0], Kind::Float);
            let rhs = expect_reg(&args[1], Kind::Float);
            let label = expect_tlabel(&args[2]);
            let label_id = builder_label(state, &label.name);
            state.builder.goto_if_not_float_ge(lhs, rhs, label_id);
        }
        "goto_if_not_ptr_eq" => {
            let lhs = expect_reg(&args[0], Kind::Ref);
            let rhs = expect_reg(&args[1], Kind::Ref);
            let label = expect_tlabel(&args[2]);
            let label_id = builder_label(state, &label.name);
            state.builder.goto_if_not_ptr_eq(lhs, rhs, label_id);
        }
        "goto_if_not_ptr_ne" => {
            let lhs = expect_reg(&args[0], Kind::Ref);
            let rhs = expect_reg(&args[1], Kind::Ref);
            let label = expect_tlabel(&args[2]);
            let label_id = builder_label(state, &label.name);
            state.builder.goto_if_not_ptr_ne(lhs, rhs, label_id);
        }
        "goto_if_not_ptr_iszero" => {
            let cond = expect_reg(&args[0], Kind::Ref);
            let label = expect_tlabel(&args[1]);
            let label_id = builder_label(state, &label.name);
            state.builder.goto_if_not_ptr_iszero(cond, label_id);
        }
        "goto_if_not_ptr_nonzero" => {
            let cond = expect_reg(&args[0], Kind::Ref);
            let label = expect_tlabel(&args[1]);
            let label_id = builder_label(state, &label.name);
            state.builder.goto_if_not_ptr_nonzero(cond, label_id);
        }
        "catch_exception" => {
            let label = expect_tlabel(&args[0]);
            let label_id = builder_label(state, &label.name);
            state.builder.catch_exception(label_id);
        }
        "goto_if_exception_mismatch" => {
            let vtable = expect_int_reg_or_pool(state, &args[0]);
            let label = expect_tlabel(&args[1]);
            let label_id = builder_label(state, &label.name);
            state.builder.goto_if_exception_mismatch(vtable, label_id);
        }
        "jit_merge_point" => {
            // Upstream-orthodox 7-arg shape (jtransform.py handle_jit_marker__jit_merge_point +
            // jtransform.py make_three_lists):
            //   [jd_index_const, greens_i, greens_r, greens_f,
            //    reds_i, reds_r, reds_f]
            // Produced by `codewriter.rs::portal_jit_merge_point_graph_args`.
            // The builder lowers to bytecode that carries
            // jdindex + the six typed register lists generally; the
            // helper writes the same byte stream regardless of which
            // lists are empty for the current jitdriver.
            let [
                jdindex_arg,
                greens_i_arg,
                greens_r_arg,
                greens_f_arg,
                reds_i_arg,
                reds_r_arg,
                reds_f_arg,
            ] = args
            else {
                panic!(
                    "jit_merge_point: expected 7 upstream-orthodox args, got {}",
                    args.len()
                );
            };
            // jtransform.py:1704 emits `Constant(self.portal_jd.index,
            // lltype.Signed)`, so the assembler-side jdindex flows
            // through as a signed integer. The builder selects the
            // USE_C_FORM short byte vs the constants-pool `i` slot per
            // `assembler.py` (`-128 <= value <= 127`); no
            // pre-builder narrowing here.
            let jdindex = match jdindex_arg {
                Operand::ConstInt(v) => *v,
                other => {
                    panic!("jit_merge_point expects ConstInt jd_index at arg[0], got {other:?}")
                }
            };
            let greens_i = expect_list_regs_or_pool(state, greens_i_arg, Kind::Int);
            let greens_r = expect_list_regs_or_pool(state, greens_r_arg, Kind::Ref);
            let greens_f = expect_list_regs_or_pool(state, greens_f_arg, Kind::Float);
            let reds_i = expect_list_regs_or_pool(state, reds_i_arg, Kind::Int);
            let reds_r = expect_list_regs_or_pool(state, reds_r_arg, Kind::Ref);
            let reds_f = expect_list_regs_or_pool(state, reds_f_arg, Kind::Float);
            state.builder.jit_merge_point(
                jdindex, &greens_i, &greens_r, &greens_f, &reds_i, &reds_r, &reds_f,
            );
        }
        "loop_header" => {
            // RPython jtransform.py handle_jit_marker__loop_header
            // emits SpaceOperation('loop_header', [c_index], None) with
            // `Constant(jd.index, lltype.Signed)`. `loop_header` is not
            // in `assembler.py USE_C_FORM`, so the builder
            // always emits the constants-pool `i` form regardless of
            // jdindex magnitude — pass the signed value through.
            let jdindex = match &args[0] {
                Operand::ConstInt(v) => *v,
                other => panic!("loop_header expects ConstInt jdindex, got {:?}", other),
            };
            state.builder.loop_header(jdindex);
        }
        // `flatten.py make_return` / `:382-384 getcolor` pass a
        // `Constant` operand through unchanged; the const arm encodes it
        // into the ref constant window (`assembler.py emit_const`).
        "ref_return" => {
            if let Some(Operand::ConstRef(value)) = args.first() {
                state
                    .builder
                    .ref_return_const(untag_const_ref_value(*value));
            } else {
                let src = expect_result_or_first_reg(args, result, Kind::Ref);
                state.builder.ref_return(src);
            }
        }
        "raise" => match &args[0] {
            Operand::ConstRef(value) => state
                .builder
                .emit_raise_const(untag_const_ref_value(*value)),
            _ => {
                let src = expect_reg(&args[0], Kind::Ref);
                state.builder.emit_raise(src);
            }
        },
        "reraise" => state.builder.emit_reraise(),
        "last_exception" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Int);
            state.builder.last_exception(dst);
        }
        "last_exc_value" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Ref);
            state.builder.last_exc_value(dst);
        }
        "abort" => state.builder.abort(),
        "abort_permanent" => state.builder.abort_permanent(),
        "unreachable" => state.builder.unreachable(),
        // `flatten.py` `self.emitline('%s_copy' % kind, v, "->", w)`.
        // `v` is either a `Register` or a `Constant`. Register source
        // lowers to the primitive `move_{i,r,f}` builder method;
        // Constant source lowers via the builder's constant-pool helper,
        // matching the two argcode variants emitted by
        // `assembler.py:162-174` (`i` for Register, `c` for Constant).
        "int_copy" => match source_operand(args, result) {
            MoveSource::Reg(src) => {
                let dst = expect_result_or_first_reg(args, result, Kind::Int);
                let src = expect_reg(src, Kind::Int);
                state.builder.move_i(dst, src);
            }
            MoveSource::ConstInt(value) => {
                let dst = expect_result_or_first_reg(args, result, Kind::Int);
                state.builder.load_const_i_value(dst, value);
            }
            other => panic!("int_copy expects Register or ConstInt, got {:?}", other),
        },
        "ref_copy" => match source_operand(args, result) {
            MoveSource::Reg(src) => {
                let dst = expect_result_or_first_reg(args, result, Kind::Ref);
                let src = expect_reg(src, Kind::Ref);
                state.builder.move_r(dst, src);
            }
            MoveSource::ConstRef(value) => {
                let dst = expect_result_or_first_reg(args, result, Kind::Ref);
                state
                    .builder
                    .load_const_r_value(dst, untag_const_ref_value(value));
            }
            other => panic!("ref_copy expects Register or ConstRef, got {:?}", other),
        },
        "float_copy" => match source_operand(args, result) {
            MoveSource::Reg(src) => {
                let dst = expect_result_or_first_reg(args, result, Kind::Float);
                let src = expect_reg(src, Kind::Float);
                state.builder.move_f(dst, src);
            }
            MoveSource::ConstFloat(value) => {
                let dst = expect_result_or_first_reg(args, result, Kind::Float);
                state.builder.load_const_f_value(dst, value);
            }
            other => panic!("float_copy expects Register or ConstFloat, got {:?}", other),
        },
        // `flatten.py` `self.emitline('%s_push' % kind, v)` — cycle-break
        // save into the kind-typed scratch slot.
        "int_push" => {
            let src = expect_reg(&args[0], Kind::Int);
            state.builder.push_i(src);
        }
        "ref_push" => {
            let src = expect_reg(&args[0], Kind::Ref);
            state.builder.push_r(src);
        }
        "float_push" => {
            let src = expect_reg(&args[0], Kind::Float);
            state.builder.push_f(src);
        }
        // `flatten.py` `self.emitline('%s_pop' % kind, "->", w)` —
        // cycle-break load from the kind-typed scratch slot.
        "int_pop" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Int);
            state.builder.pop_i(dst);
        }
        "ref_pop" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Ref);
            state.builder.pop_r(dst);
        }
        "float_pop" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Float);
            state.builder.pop_f(dst);
        }
        // `load_state_*` / `store_state_*` arms removed: pyre-specific
        // opnames with no SSARepr emitter — the pyre value-stack runtime
        // ops are emitted directly through the JitCodeBuilder rather than
        // routed through the SSARepr dispatch table.
        //
        // RPython opname parity (jtransform.py:844-927, blackhole.py bhimpl_getfield_vable_i):  allow-line-citation
        // `kind = getkind(...)[0]` in jtransform.py forces single-char
        // suffix everywhere — `getfield_vable_i` / `_r` / `_f` and
        // `setfield_vable_i` / `_r` / `_f`. pyre's JitCodeBuilder methods
        // retain the `vable_*` prefix as a Note so the
        // rename is scoped to the Insn::Op key.  Long-form `_int` /
        // `_ref` / `_float` aliases retired — no SSARepr emitter writes
        // them (codewriter.rs `emit_vable_*` macros use the short form).
        "getfield_vable_i" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Int);
            let (vable_reg, field_idx) = expect_vable_field_read_args(args);
            state
                .builder
                .vable_getfield_int_with_base(dst, vable_reg, field_idx);
        }
        "getfield_vable_r" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Ref);
            let (vable_reg, field_idx) = expect_vable_field_read_args(args);
            state
                .builder
                .vable_getfield_ref_with_base(dst, vable_reg, field_idx);
        }
        "getfield_vable_f" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Float);
            let (vable_reg, field_idx) = expect_vable_field_read_args(args);
            state
                .builder
                .vable_getfield_float_with_base(dst, vable_reg, field_idx);
        }
        "getfield_gc_i" => {
            let dst = expect_result_reg(result, Kind::Int, "getfield_gc_i needs result");
            assert_eq!(args.len(), 2, "getfield_gc_i expects [struct, fielddescr]");
            let struct_reg = expect_reg(&args[0], Kind::Ref);
            let field = expect_field_bh_descr(&args[1], "getfield_gc_i");
            state.builder.getfield_gc_i_bh(dst, struct_reg, field);
        }
        "setfield_gc_i" => {
            assert_eq!(
                args.len(),
                3,
                "setfield_gc_i expects [struct, value, fielddescr]"
            );
            let struct_reg = expect_reg(&args[0], Kind::Ref);
            let value_reg = expect_reg(&args[1], Kind::Int);
            let field = expect_field_bh_descr(&args[2], "setfield_gc_i");
            state.builder.setfield_gc_i_bh(struct_reg, value_reg, field);
        }
        "setfield_vable_i" => {
            // pyjitpl.py `_opimpl_setfield_vable` is generic
            // across i/r/f — `valuebox` may be a ConstInt after optimizer
            // folding.  Mirror the `setfield_vable_r` ConstRef branch.
            assert_eq!(
                args.len(),
                3,
                "setfield_vable_i expects [vable, value, fielddescr]"
            );
            let vable_reg = expect_reg(&args[0], Kind::Ref);
            let field_idx = expect_descr_vable_static_field(&args[2]);
            match &args[1] {
                Operand::Register(r) if r.kind == Kind::Int => {
                    state
                        .builder
                        .vable_setfield_int_with_base(vable_reg, field_idx, r.index);
                }
                Operand::ConstInt(value) => {
                    state
                        .builder
                        .vable_setfield_int_const_value_with_base(vable_reg, field_idx, *value);
                }
                other => panic!(
                    "setfield_vable_i expects Register(Int) or ConstInt, got {:?}",
                    other,
                ),
            }
        }
        "setfield_vable_r" => {
            assert_eq!(
                args.len(),
                3,
                "setfield_vable_r expects [vable, value, fielddescr]"
            );
            let vable_reg = expect_reg(&args[0], Kind::Ref);
            let field_idx = expect_descr_vable_static_field(&args[2]);
            match &args[1] {
                Operand::Register(r) if r.kind == Kind::Ref => {
                    state
                        .builder
                        .vable_setfield_ref_with_base(vable_reg, field_idx, r.index);
                }
                Operand::ConstRef(value) => {
                    state.builder.vable_setfield_ref_const_value_with_base(
                        vable_reg,
                        field_idx,
                        untag_const_ref_value(*value),
                    );
                }
                other => panic!(
                    "setfield_vable_r expects Register(Ref) or ConstRef, got {:?}",
                    other,
                ),
            }
        }
        "setfield_vable_f" => {
            // pyjitpl.py:1188-1199 — float counterpart to setfield_vable_i
            // ConstInt branch.
            assert_eq!(
                args.len(),
                3,
                "setfield_vable_f expects [vable, value, fielddescr]"
            );
            let vable_reg = expect_reg(&args[0], Kind::Ref);
            let field_idx = expect_descr_vable_static_field(&args[2]);
            match &args[1] {
                Operand::Register(r) if r.kind == Kind::Float => {
                    state
                        .builder
                        .vable_setfield_float_with_base(vable_reg, field_idx, r.index);
                }
                Operand::ConstFloat(value) => {
                    state
                        .builder
                        .vable_setfield_float_const_value_with_base(vable_reg, field_idx, *value);
                }
                other => panic!(
                    "setfield_vable_f expects Register(Float) or ConstFloat, got {:?}",
                    other,
                ),
            }
        }
        "getarrayitem_vable_i" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Int);
            let (vable_reg, array_idx) = expect_vable_array_read_prefix(args);
            match &args[1] {
                Operand::Register(r) if r.kind == Kind::Int => {
                    state
                        .builder
                        .vable_getarrayitem_int_with_base(dst, vable_reg, array_idx, r.index);
                }
                Operand::ConstInt(value) => {
                    state.builder.vable_getarrayitem_int_const_idx_with_base(
                        dst, vable_reg, array_idx, *value,
                    );
                }
                other => panic!(
                    "getarrayitem_vable_i expects Register(Int) or ConstInt index, got {:?}",
                    other,
                ),
            }
        }
        "getarrayitem_vable_r" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Ref);
            let (vable_reg, array_idx) = expect_vable_array_read_prefix(args);
            match &args[1] {
                Operand::Register(r) if r.kind == Kind::Int => {
                    state
                        .builder
                        .vable_getarrayitem_ref_with_base(dst, vable_reg, array_idx, r.index);
                }
                Operand::ConstInt(value) => {
                    state.builder.vable_getarrayitem_ref_const_idx_with_base(
                        dst, vable_reg, array_idx, *value,
                    );
                }
                other => panic!(
                    "getarrayitem_vable_r expects Register(Int) or ConstInt index, got {:?}",
                    other,
                ),
            }
        }
        "getarrayitem_vable_f" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Float);
            let (vable_reg, array_idx) = expect_vable_array_read_prefix(args);
            match &args[1] {
                Operand::Register(r) if r.kind == Kind::Int => {
                    state
                        .builder
                        .vable_getarrayitem_float_with_base(dst, vable_reg, array_idx, r.index);
                }
                Operand::ConstInt(value) => {
                    state.builder.vable_getarrayitem_float_const_idx_with_base(
                        dst, vable_reg, array_idx, *value,
                    );
                }
                other => panic!(
                    "getarrayitem_vable_f expects Register(Int) or ConstInt index, got {:?}",
                    other,
                ),
            }
        }
        "setarrayitem_vable_i" => {
            // pyjitpl.py:1236-1247 — see setfield_vable_i ConstInt rationale.
            // ConstInt index is parity with `setarrayitem_vable_r` per
            // `rpython/jit/codewriter/assembler.py:162`.
            let (vable_reg, array_idx) = expect_vable_array_write_prefix_no_idx(args);
            match (&args[1], &args[2]) {
                (Operand::Register(idx), Operand::Register(src))
                    if idx.kind == Kind::Int && src.kind == Kind::Int =>
                {
                    state.builder.vable_setarrayitem_int_with_base(
                        vable_reg, array_idx, idx.index, src.index,
                    );
                }
                (Operand::Register(idx), Operand::ConstInt(value)) if idx.kind == Kind::Int => {
                    state.builder.vable_setarrayitem_int_const_value_with_base(
                        vable_reg, array_idx, idx.index, *value,
                    );
                }
                (Operand::ConstInt(idx_value), Operand::Register(src)) if src.kind == Kind::Int => {
                    state.builder.vable_setarrayitem_int_const_idx_with_base(
                        vable_reg, array_idx, *idx_value, src.index,
                    );
                }
                (Operand::ConstInt(idx_value), Operand::ConstInt(src_value)) => {
                    state
                        .builder
                        .vable_setarrayitem_int_const_idx_const_value_with_base(
                            vable_reg, array_idx, *idx_value, *src_value,
                        );
                }
                (idx, src) => panic!(
                    "setarrayitem_vable_i expects (Register(Int) | ConstInt) index and \
                     (Register(Int) | ConstInt) value, got idx={:?} src={:?}",
                    idx, src,
                ),
            }
        }
        "setarrayitem_vable_r" => {
            let (vable_reg, array_idx) = expect_vable_array_write_prefix_no_idx(args);
            match (&args[1], &args[2]) {
                (Operand::Register(idx), Operand::Register(src))
                    if idx.kind == Kind::Int && src.kind == Kind::Ref =>
                {
                    state.builder.vable_setarrayitem_ref_with_base(
                        vable_reg, array_idx, idx.index, src.index,
                    );
                }
                (Operand::Register(idx), Operand::ConstRef(value)) if idx.kind == Kind::Int => {
                    state.builder.vable_setarrayitem_ref_const_value_with_base(
                        vable_reg,
                        array_idx,
                        idx.index,
                        untag_const_ref_value(*value),
                    );
                }
                (Operand::ConstInt(idx_value), Operand::Register(src)) if src.kind == Kind::Ref => {
                    state.builder.vable_setarrayitem_ref_const_idx_with_base(
                        vable_reg, array_idx, *idx_value, src.index,
                    );
                }
                (Operand::ConstInt(idx_value), Operand::ConstRef(src_value)) => {
                    state
                        .builder
                        .vable_setarrayitem_ref_const_idx_const_value_with_base(
                            vable_reg,
                            array_idx,
                            *idx_value,
                            untag_const_ref_value(*src_value),
                        );
                }
                (idx, src) => panic!(
                    "setarrayitem_vable_r expects (Register(Int) | ConstInt) index and \
                     (Register(Ref) | ConstRef) value, got idx={:?} src={:?}",
                    idx, src,
                ),
            }
        }
        "setarrayitem_vable_f" => {
            // pyjitpl.py:1236-1247 — see setfield_vable_i ConstInt rationale.
            // ConstInt index is parity with `setarrayitem_vable_r` per
            // `rpython/jit/codewriter/assembler.py:162`.
            let (vable_reg, array_idx) = expect_vable_array_write_prefix_no_idx(args);
            match (&args[1], &args[2]) {
                (Operand::Register(idx), Operand::Register(src))
                    if idx.kind == Kind::Int && src.kind == Kind::Float =>
                {
                    state.builder.vable_setarrayitem_float_with_base(
                        vable_reg, array_idx, idx.index, src.index,
                    );
                }
                (Operand::Register(idx), Operand::ConstFloat(value)) if idx.kind == Kind::Int => {
                    state
                        .builder
                        .vable_setarrayitem_float_const_value_with_base(
                            vable_reg, array_idx, idx.index, *value,
                        );
                }
                (Operand::ConstInt(idx_value), Operand::Register(src))
                    if src.kind == Kind::Float =>
                {
                    state.builder.vable_setarrayitem_float_const_idx_with_base(
                        vable_reg, array_idx, *idx_value, src.index,
                    );
                }
                (Operand::ConstInt(idx_value), Operand::ConstFloat(src_value)) => {
                    state
                        .builder
                        .vable_setarrayitem_float_const_idx_const_value_with_base(
                            vable_reg, array_idx, *idx_value, *src_value,
                        );
                }
                (idx, src) => panic!(
                    "setarrayitem_vable_f expects (Register(Int) | ConstInt) index and \
                     (Register(Float) | ConstFloat) value, got idx={:?} src={:?}",
                    idx, src,
                ),
            }
        }
        "arraylen_vable" => {
            let dst = expect_result_or_first_reg(args, result, Kind::Int);
            let (vable_reg, array_idx) = expect_vable_arraylen_args(args);
            state
                .builder
                .vable_arraylen_with_base(dst, vable_reg, array_idx);
        }
        // `blackhole.py bhimpl_raw_load_i` / `jtransform.py rewrite_op_raw_load`.
        // Interned as a raw (non-GC) array descr so CSE cannot treat the
        // load as a pure GC-array read — the back-edge poll of the eval
        // breaker word depends on that.
        "raw_load_i" => {
            assert_eq!(
                args.len(),
                3,
                "raw_load_i expects [base, offset, arraydescr]"
            );
            let dst = expect_result_reg(result, Kind::Int, "raw_load_i needs result");
            let base = expect_int_reg_or_pool(state, &args[0]);
            let offset = expect_int_reg_or_pool(state, &args[1]);
            let descr_idx = match &args[2] {
                Operand::Descr(d) => match &**d {
                    DescrOperand::Bh(majit_jitcode::jitcode::BhDescr::Array {
                        itemsize,
                        is_item_signed,
                        ..
                    }) => state
                        .builder
                        .add_raw_int_array_descr_signed(*itemsize, *is_item_signed),
                    _ => expect_array_bh_descr(state, &args[2], "raw_load_i"),
                },
                _ => expect_array_bh_descr(state, &args[2], "raw_load_i"),
            };
            state.builder.raw_load_i(dst, base, offset, descr_idx);
        }
        // Heap GC-array primitives, canonical shapes
        // `new_array_clear/id>r` (blackhole.py `bhimpl_new_array_clear`)
        // and `setarrayitem_gc_r/rird` (`bhimpl_setarrayitem_gc_r`),
        // plus their `c`-argcode forms `cd>r`/`rcrd` for small ConstInt
        // operands (USE_C_FORM).
        // The descr operand arrives as an already-materialised
        // `DescrOperand::Bh(BhDescr::Array { .. })` and registers onto
        // the builder's shared descrs pool (`assembler.py:197-207
        // _encode_descr`).
        "new_array_clear" => {
            assert_eq!(
                args.len(),
                2,
                "new_array_clear expects [length, arraydescr]"
            );
            let dst = expect_result_reg(result, Kind::Ref, "new_array_clear needs result");
            let descr_idx = expect_array_bh_descr(state, &args[1], "new_array_clear");
            // `assembler.py emit_const(allow_short=True)` —
            // `new_array_clear` is in `USE_C_FORM` (`assembler.py`),
            // so a small ConstInt length is encoded inline as the `c`
            // argcode (`new_array_clear/cd>r`).
            match &args[0] {
                Operand::ConstInt(v) if (-128..=127).contains(v) => {
                    state.builder.new_array_clear_c(dst, *v as i8, descr_idx);
                }
                other => {
                    let length = expect_int_reg_or_pool(state, other);
                    state.builder.new_array_clear(dst, length, descr_idx);
                }
            }
        }
        "setarrayitem_gc_r" => {
            assert_eq!(
                args.len(),
                4,
                "setarrayitem_gc_r expects [array, index, value, arraydescr]"
            );
            let array = expect_reg(&args[0], Kind::Ref);
            let value = expect_ref_reg_or_pool(state, &args[2]);
            let descr_idx = expect_array_bh_descr(state, &args[3], "setarrayitem_gc_r");
            // `setarrayitem_gc_r` is also in `USE_C_FORM`: a small
            // ConstInt index takes the `c` form (`setarrayitem_gc_r/rcrd`).
            match &args[1] {
                Operand::ConstInt(v) if (-128..=127).contains(v) => {
                    state
                        .builder
                        .setarrayitem_gc_r_c(array, *v as i8, value, descr_idx);
                }
                other => {
                    let index = expect_int_reg_or_pool(state, other);
                    state
                        .builder
                        .setarrayitem_gc_r(array, index, value, descr_idx);
                }
            }
        }
        "hint_force_virtualizable" => {
            assert_eq!(args.len(), 1, "hint_force_virtualizable expects [vable]");
            let vable_reg = expect_reg(&args[0], Kind::Ref);
            state.builder.vable_force_with_base(vable_reg);
        }
        // `int_floordiv` / `int_mod` are intentionally absent from the
        // canonical-binop arm below.  `jtransform.py:575-577`
        // (`rewrite_op_int_floordiv = _do_builtin_call`,
        // `rewrite_op_int_mod = _do_builtin_call`) replaces both
        // primitives with a `direct_call(prepare_builtin_call(...))` to
        // `ll_int_py_div` / `ll_int_py_mod` (`rint.py/496`); the
        // bare opname never reaches the SSARepr emitter upstream.  In
        // pyre the runtime trace path goes through the β' redirect at
        // `pyre-jit-trace/src/jitcode_dispatch/specialize.rs::walker_emit_int_py_div_or_mod`,
        // which records `CallI` on `ll_int_py_div` / `ll_int_py_mod`
        // through `call_typed_with_effect_pure` with
        // `INT_PY_DIV_EFFECT_INFO`/`INT_PY_MOD_EFFECT_INFO`, preceded by
        // `walker_emit_int_div_domain_guards`' zero and `INT_MIN / -1`
        // `guard_false` pair from the `_ovf_zer` wrapper.
        // Per-OpCode opname dispatch for integer / float primitives —
        // RPython `rpython/jit/metainterp/blackhole.py bhimpl_int_add` defines
        // one `bhimpl_*` per opname; `assembler.py:162-222` routes each
        // via its own insns key. The SSARepr layer carries the
        // canonical opname so pyre dispatch picks the matching
        // `record_binop_*` / `record_unary_*` helper without a generic
        // OpCode-bearing wrapper insn. No `generic record_*` SSA
        // opname exists; only the canonical RPython names.
        "int_add" | "int_sub" | "int_mul" | "int_and" | "int_or" | "int_xor" | "int_lshift"
        | "int_rshift" | "int_eq" | "int_ne" | "int_lt" | "int_le" | "int_gt" | "int_ge"
        | "uint_rshift" | "uint_mul_high" | "uint_lt" | "uint_le" | "uint_gt" | "uint_ge" => {
            let dst = expect_result_reg(result, Kind::Int, "int binop needs result");
            let opcode = match opname {
                "int_add" => majit_ir::OpCode::IntAdd,
                "int_sub" => majit_ir::OpCode::IntSub,
                "int_mul" => majit_ir::OpCode::IntMul,
                "int_and" => majit_ir::OpCode::IntAnd,
                "int_or" => majit_ir::OpCode::IntOr,
                "int_xor" => majit_ir::OpCode::IntXor,
                "int_lshift" => majit_ir::OpCode::IntLshift,
                "int_rshift" => majit_ir::OpCode::IntRshift,
                "int_eq" => majit_ir::OpCode::IntEq,
                "int_ne" => majit_ir::OpCode::IntNe,
                "int_lt" => majit_ir::OpCode::IntLt,
                "int_le" => majit_ir::OpCode::IntLe,
                "int_gt" => majit_ir::OpCode::IntGt,
                "int_ge" => majit_ir::OpCode::IntGe,
                "uint_rshift" => majit_ir::OpCode::UintRshift,
                "uint_mul_high" => majit_ir::OpCode::UintMulHigh,
                "uint_lt" => majit_ir::OpCode::UintLt,
                "uint_le" => majit_ir::OpCode::UintLe,
                "uint_gt" => majit_ir::OpCode::UintGt,
                "uint_ge" => majit_ir::OpCode::UintGe,
                _ => unreachable!(),
            };
            let lhs = expect_jitcode_int_operand(state, &args[0], use_c_form(opname));
            let rhs = expect_jitcode_int_operand(state, &args[1], use_c_form(opname));
            match (lhs, rhs) {
                (JitCodeIntOperand::Register(lhs), JitCodeIntOperand::Register(rhs))
                    if both_int_operands_are_registers(&args[0], &args[1]) =>
                {
                    state.builder.record_binop_i(dst, opcode, lhs, rhs)
                }
                _ => state.builder.record_binop_i_operands(
                    dst,
                    opcode_byte.expect("short integer binop opcode"),
                    lhs,
                    rhs,
                ),
            }
        }
        "int_neg" | "int_invert" => {
            let dst = expect_result_reg(result, Kind::Int, "int unary needs result");
            let opcode = match opname {
                "int_neg" => majit_ir::OpCode::IntNeg,
                "int_invert" => majit_ir::OpCode::IntInvert,
                _ => unreachable!(),
            };
            state
                .builder
                .record_unary_i(dst, opcode, expect_reg(&args[0], Kind::Int));
        }
        "ptr_eq" | "ptr_ne" | "instance_ptr_eq" | "instance_ptr_ne" => {
            let dst = expect_result_reg(result, Kind::Int, "ptr binop needs result");
            let opcode = match opname {
                "ptr_eq" => majit_ir::OpCode::PtrEq,
                "ptr_ne" => majit_ir::OpCode::PtrNe,
                "instance_ptr_eq" => majit_ir::OpCode::InstancePtrEq,
                "instance_ptr_ne" => majit_ir::OpCode::InstancePtrNe,
                _ => unreachable!(),
            };
            // Either side may be a constant — `x is None` compares against the
            // `None` singleton — so route a `ConstRef` through the jitcode
            // `constants_r` pool the same way `setarrayitem_gc_r` does for its
            // value operand, rather than demanding a materialized register.
            let lhs = expect_ref_reg_or_pool(state, &args[0]);
            let rhs = expect_ref_reg_or_pool(state, &args[1]);
            state.builder.record_binop_r(dst, opcode, lhs, rhs);
        }
        "ptr_iszero" => {
            let dst = expect_result_reg(result, Kind::Int, "ptr_iszero needs result");
            state
                .builder
                .ptr_iszero(dst, expect_reg(&args[0], Kind::Ref));
        }
        "ptr_nonzero" => {
            let dst = expect_result_reg(result, Kind::Int, "ptr_nonzero needs result");
            state
                .builder
                .ptr_nonzero(dst, expect_reg(&args[0], Kind::Ref));
        }
        "float_add" | "float_sub" | "float_mul" | "float_truediv" => {
            let dst = expect_result_reg(result, Kind::Float, "float binop needs result");
            let opcode = match opname {
                "float_add" => majit_ir::OpCode::FloatAdd,
                "float_sub" => majit_ir::OpCode::FloatSub,
                "float_mul" => majit_ir::OpCode::FloatMul,
                "float_truediv" => majit_ir::OpCode::FloatTrueDiv,
                _ => unreachable!(),
            };
            state.builder.record_binop_f(
                dst,
                opcode,
                expect_reg(&args[0], Kind::Float),
                expect_reg(&args[1], Kind::Float),
            );
        }
        "float_neg" | "float_abs" => {
            let dst = expect_result_reg(result, Kind::Float, "float unary needs result");
            let opcode = match opname {
                "float_neg" => majit_ir::OpCode::FloatNeg,
                "float_abs" => majit_ir::OpCode::FloatAbs,
                _ => unreachable!(),
            };
            state
                .builder
                .record_unary_f(dst, opcode, expect_reg(&args[0], Kind::Float));
        }
        "conditional_call_ir_v" => {
            let fn_idx = expect_small_u16(&args[0]);
            let cond_reg = expect_reg(&args[1], Kind::Int);
            let call_args = expect_call_args(&args[2..]);
            state
                .builder
                .conditional_call_ir_v_typed_args(fn_idx, cond_reg, &call_args);
        }
        "conditional_call_value_ir_i" => {
            let fn_idx = expect_small_u16(&args[0]);
            let value_reg = expect_reg(&args[1], Kind::Int);
            let dst = expect_result_reg(
                result,
                Kind::Int,
                "conditional_call_value_ir_i needs result",
            );
            let call_args = expect_call_args(&args[2..]);
            state
                .builder
                .conditional_call_value_ir_i_typed_args(fn_idx, value_reg, &call_args, dst);
        }
        "conditional_call_value_ir_r" => {
            let fn_idx = expect_small_u16(&args[0]);
            let value_reg = expect_reg(&args[1], Kind::Ref);
            let dst = expect_result_reg(
                result,
                Kind::Ref,
                "conditional_call_value_ir_r needs result",
            );
            let call_args = expect_call_args(&args[2..]);
            state
                .builder
                .conditional_call_value_ir_r_typed_args(fn_idx, value_reg, &call_args, dst);
        }
        "record_known_result_i_ir_v" => {
            let fn_idx = expect_small_u16(&args[0]);
            let result_reg = expect_reg(&args[1], Kind::Int);
            let call_args = expect_call_args(&args[2..]);
            state
                .builder
                .record_known_result_i_ir_v_typed_args(fn_idx, result_reg, &call_args);
        }
        "record_known_result_r_ir_v" => {
            let fn_idx = expect_small_u16(&args[0]);
            let result_reg = expect_reg(&args[1], Kind::Ref);
            let call_args = expect_call_args(&args[2..]);
            state
                .builder
                .record_known_result_r_ir_v_typed_args(fn_idx, result_reg, &call_args);
        }
        // `rpython/jit/codewriter/jtransform.py rewrite_call` shape.
        // 12 opname combinations: residual_call_{r,ir,irf}_{i,r,f,v}. Only
        // the subset the walker currently emits is routed; unused arms
        // panic with a pointer to emit_residual_call.
        opname if opname.starts_with("residual_call_") => {
            dispatch_residual_call(state, opname, args, result);
        }
        // `jtransform.py handle_regular_call` shape: leading JitCode descr,
        // then the non-empty kind lists selected by `rewrite_call`, and an
        // optional typed result.  Unlike the older pyre-only nested-call
        // adapter, this emits the canonical dR/dIR/dIRF byte layout.
        opname if opname.starts_with("inline_call_") => {
            dispatch_inline_call(state, opname, args, result);
        }
        other => panic!(
            "assemble(): unimplemented opname {:?} — add a builder mapping in jit/assembler.rs",
            other
        ),
    }
}

fn dispatch_inline_call(
    state: &mut AssemblyState,
    opname: &str,
    args: &[Operand],
    result: Option<&Register>,
) {
    let tail = &opname["inline_call_".len()..];
    let (kinds, reskind) = tail
        .rsplit_once('_')
        .unwrap_or_else(|| panic!("malformed inline_call opname: {opname:?}"));
    assert!(matches!(kinds, "r" | "ir" | "irf"));
    assert!(matches!(reskind, "i" | "r" | "f" | "v"));

    let target = match args.first() {
        Some(Operand::Descr(descr)) => match &**descr {
            DescrOperand::JitCode(jitcode) => Arc::clone(jitcode),
            other => panic!("inline_call expects a JitCode descr, got {other:?}"),
        },
        other => panic!("inline_call expects a leading Descr operand, got {other:?}"),
    };
    let target_idx = state.builder.add_sub_jitcode_arc(target);

    let mut cursor = 1usize;
    let mut args_i = None;
    let mut args_r = None;
    let mut args_f = None;
    if kinds.contains('i') {
        args_i = Some(
            expect_list_regs_or_pool(state, &args[cursor], Kind::Int)
                .into_iter()
                .map(u16::from)
                .collect::<Vec<_>>(),
        );
        cursor += 1;
    }
    if kinds.contains('r') {
        args_r = Some(
            expect_list_regs_or_pool(state, &args[cursor], Kind::Ref)
                .into_iter()
                .map(u16::from)
                .collect::<Vec<_>>(),
        );
        cursor += 1;
    }
    if kinds.contains('f') {
        args_f = Some(
            expect_list_regs_or_pool(state, &args[cursor], Kind::Float)
                .into_iter()
                .map(u16::from)
                .collect::<Vec<_>>(),
        );
        cursor += 1;
    }
    assert_eq!(cursor, args.len(), "inline_call has trailing operands");

    let result = match reskind {
        "v" => {
            assert!(result.is_none(), "void inline_call cannot have a result");
            None
        }
        "i" => Some((
            majit_metainterp::jitcode::JitArgKind::Int,
            expect_result_reg(result, Kind::Int, "inline_call_i needs result"),
        )),
        "r" => Some((
            majit_metainterp::jitcode::JitArgKind::Ref,
            expect_result_reg(result, Kind::Ref, "inline_call_r needs result"),
        )),
        "f" => Some((
            majit_metainterp::jitcode::JitArgKind::Float,
            expect_result_reg(result, Kind::Float, "inline_call_f needs result"),
        )),
        _ => unreachable!(),
    };
    let key: &'static str = match (kinds, reskind) {
        ("r", "i") => "inline_call_r_i/dR>i",
        ("r", "r") => "inline_call_r_r/dR>r",
        ("r", "v") => "inline_call_r_v/dR",
        ("ir", "i") => "inline_call_ir_i/dIR>i",
        ("ir", "r") => "inline_call_ir_r/dIR>r",
        ("ir", "v") => "inline_call_ir_v/dIR",
        ("irf", "i") => "inline_call_irf_i/dIRF>i",
        ("irf", "r") => "inline_call_irf_r/dIRF>r",
        ("irf", "f") => "inline_call_irf_f/dIRF>f",
        ("irf", "v") => "inline_call_irf_v/dIRF",
        _ => panic!("unsupported canonical inline_call shape: {opname}"),
    };
    state.builder.canonical_inline_call(
        key,
        target_idx,
        args_i.as_deref(),
        args_r.as_deref(),
        args_f.as_deref(),
        result,
    );
}

/// Consumer for the `residual_call_{kinds}_{reskind}` shape
/// emitted by `flatten`'s `build_*_residual_call_*_insn` family.
///
/// SSARepr arg layout (as produced by those builders):
/// ```text
/// [Const(fn_idx),
///  ListOfKind(int,   [...])?,   # present iff 'i' in kinds
///  ListOfKind(ref,   [...])?,   # present iff 'r' in kinds
///  ListOfKind(float, [...])?,   # present iff 'f' in kinds
///  Descr(CallDescrStub{effect_info, arg_kinds})]
/// ```
///
/// pyre's `JitCodeBuilder::call_*_typed` takes a single flat
/// `&[JitCallArg]` in C-function parameter order. `arg_kinds` carries
/// that order; this helper reassembles by pulling one entry at a time
/// from the appropriate per-kind sublist in upstream's `bh_call_*`
/// pool layout.
fn dispatch_residual_call(
    state: &mut AssemblyState,
    opname: &str,
    args: &[Operand],
    result: Option<&Register>,
) {
    // opname suffix = `{kinds}_{reskind}`, kinds ∈ {r, ir, irf}.
    let tail = &opname["residual_call_".len()..];
    let (kinds, reskind_ch) = tail
        .rsplit_once('_')
        .unwrap_or_else(|| panic!("malformed residual_call opname: {opname:?}"));

    let fn_idx = expect_small_u16(&args[0]);

    // Walk `args[1..]`: per-kind ListOfKind sublists come first in
    // (int, ref, float) order per the `kinds` string, then the final
    // Descr operand carrying `CallDescrStub`.
    let mut cursor = 1usize;
    let mut int_list: &[Operand] = &[];
    let mut ref_list: &[Operand] = &[];
    let mut float_list: &[Operand] = &[];
    if kinds.contains('i') {
        match &args[cursor] {
            Operand::ListOfKind(ListOfKind {
                kind: Kind::Int,
                content,
            }) => int_list = content,
            other => {
                panic!("residual_call: expected ListOfKind(int) at index {cursor}, got {other:?}")
            }
        }
        cursor += 1;
    }
    if kinds.contains('r') {
        match &args[cursor] {
            Operand::ListOfKind(ListOfKind {
                kind: Kind::Ref,
                content,
            }) => ref_list = content,
            other => {
                panic!("residual_call: expected ListOfKind(ref) at index {cursor}, got {other:?}")
            }
        }
        cursor += 1;
    }
    if kinds.contains('f') {
        match &args[cursor] {
            Operand::ListOfKind(ListOfKind {
                kind: Kind::Float,
                content,
            }) => float_list = content,
            other => {
                panic!("residual_call: expected ListOfKind(float) at index {cursor}, got {other:?}")
            }
        }
        cursor += 1;
    }
    // CallDescrStub slot.
    let stub = match &args[cursor] {
        Operand::Descr(rc) => match &**rc {
            DescrOperand::CallDescrStub(stub) => stub,
            other => panic!(
                "residual_call: expected CallDescrStub descr at index {cursor}, got {other:?}"
            ),
        },
        other => panic!("residual_call: expected Descr operand at index {cursor}, got {other:?}"),
    };

    // Reassemble flat `&[JitCallArg]` in C-function parameter order.
    // `ListOfKind` slots admit `Variable | Constant`
    // (`rpython/jit/codewriter/jtransform.py` `make_three_lists`);
    // route `Const*` through the per-kind constant pool to synthesize the
    // `num_regs_kind + pool_idx` pseudo-register byte (mirrors
    // `assembler.py emit_const` and the `expect_list_regs_or_pool`
    // helper used by `jit_merge_point` lowering).
    let mut call_args: Vec<JitCallArg> = Vec::with_capacity(stub.arg_kinds.len());
    let mut i_cursor = 0usize;
    let mut r_cursor = 0usize;
    let mut f_cursor = 0usize;
    for &ak in &stub.arg_kinds {
        match ak {
            Kind::Int => {
                let reg = expect_int_reg_or_pool(state, &int_list[i_cursor]);
                call_args.push(JitCallArg::int(reg));
                i_cursor += 1;
            }
            Kind::Ref => {
                let reg = expect_ref_reg_or_pool(state, &ref_list[r_cursor]);
                call_args.push(JitCallArg::reference(reg));
                r_cursor += 1;
            }
            Kind::Float => {
                let reg = expect_float_reg_or_pool(state, &float_list[f_cursor]);
                call_args.push(JitCallArg::float(reg));
                f_cursor += 1;
            }
        }
    }

    use super::flatten::{CallFlavor, ResKind, dispatch_kind_for_effect_info};
    let reskind = match reskind_ch {
        "i" => ResKind::Int,
        "r" => ResKind::Ref,
        "f" => ResKind::Float,
        "v" => ResKind::Void,
        other => panic!("residual_call: unknown reskind {other:?}"),
    };

    // `descr.py:665 getCallDescr` keys on `result_type` AND `descr.py:670-674`
    // stores it on the constructed `CallDescr`; the dispatcher reads
    // `descr.result_type` and the opname suffix as a redundant pair
    // that MUST agree per `descr.create_call_stub` round-trip. Pyre
    // mirrors that invariant: the opname tail decodes to `reskind`,
    // `CallDescrStub.result_kind` carries the descr-side answer, and
    // they cross-validate here. Use `assert_eq!` (not
    // `debug_assert_eq!`) so the invariant survives release builds —
    // RPython's invariants are runtime-enforced, and a silent release
    // bypass would let a producer's bad opname/result_kind pair reach
    // the dispatch branch undetected.
    assert_eq!(
        stub.result_kind,
        reskind.to_kind(),
        "residual_call: descr result_kind {:?} disagrees with opname-tail reskind {:?}",
        stub.result_kind,
        reskind,
    );

    // Derive the dispatch branch
    // from `stub.effect_info`. Mirrors `pyjitpl.py do_residual_call
    // do_residual_call`'s precedence — the optimizer upstream picks
    // `call_may_force` / `call_release_gil` / `call_loopinvariant` /
    // `call_pure` from the same effectinfo bits, so pyre's pre-baked
    // codewriter branch resolves to the same builder method through the
    // predicates instead of a producer-side enum.
    let dispatch_kind = dispatch_kind_for_effect_info(&stub.effect_info);
    if reskind == ResKind::Void {
        if dispatch_kind.is_pure() {
            panic!("dispatch_residual_call: pure void call is not a valid rewrite_call shape");
        }
        state
            .builder
            .residual_call_void_canonical_via_target_with_effect_info_and_word_abi(
                fn_idx,
                &call_args,
                stub.effect_info.clone(),
                stub.void_word_abi,
            );
        return;
    }

    // The int / ref / float arms route through the
    // canonical `residual_call_<kind>_canonical_via_target_with_effect_info`
    // wrappers (jitcode/assembler.rs), threading
    // `stub.effect_info` end-to-end exactly like the
    // `residual_call_void_canonical_via_target_with_effect_info_and_word_abi`
    // arm above.  RPython `pyjitpl.py do_residual_call` carries
    // the same calldescr across every result kind via
    // `record_nospec(opnum, ..., calldescr)`; pyre now matches that for
    // i / r / f as well — the policy is encoded inside
    // `stub.effect_info` (already shaped by `effect_info_for_call_flavor`
    // in `flatten.rs`), and the canonical typed recording arms in
    // pyjitpl/dispatch.rs read it back via
    // `effectinfo.is_call_release_gil()` /
    // `check_forces_virtual_or_virtualizable()` /
    // `extraeffect == LoopInvariant`.
    //
    // `CallFlavor::Pure*` flows through
    // the canonical `residual_call_*_canonical_via_target_with_effect_info`
    // catchalls below.  `effect_info_for_call_flavor(Pure*)`
    // (`flatten.rs`) returns the matching `EF_ELIDABLE_*` per
    // `call.py:292-299`'s 3-way pick, which lands on the
    // calldescr's `extra_info`; the canonical
    // `BC_RESIDUAL_CALL_*_{I,R,F}` walker
    // (`majit-metainterp/src/pyjitpl/dispatch.rs`) reads
    // `effectinfo.check_is_elidable()` and routes the result through
    // `record_result_of_call_pure` mirroring `pyjitpl.py`.
    // ReleaseGil + Ref still panics per `resoperation.py # no
    // such thing`.
    match (dispatch_kind, reskind) {
        (CallFlavor::ReleaseGil, ResKind::Ref) => {
            // RPython `resoperation.py call_release_gil_for_descr`
            // explicitly skips the `tp == 'r'` arm with the comment
            // `# no such thing` (`:1243-1244`):
            //
            //     # no such thing
            //     #elif tp == 'r':
            //     #    return rop.CALL_RELEASE_GIL_R
            //
            // and `:1462 is_call_release_gil` excludes
            // `CALL_RELEASE_GIL_R` from the predicate.  No upstream
            // dispatcher / optimizer / backend can consume a
            // `CALL_RELEASE_GIL_R` op; emitting one is a deviation
            // surface that pyre rejects at the codewriter layer.
            panic!(
                "dispatch_residual_call: ReleaseGil + ResKind::Ref has no upstream \
                 counterpart (resoperation.py:1243-1244 `# no such thing`); pyre \
                 must not emit CALL_RELEASE_GIL_R via the residual_call shape."
            );
        }
        (_, ResKind::Int) => {
            let dst = expect_result_reg(result, Kind::Int, "residual_call int needs result");
            state
                .builder
                .residual_call_int_canonical_via_target_with_effect_info(
                    fn_idx,
                    &call_args,
                    dst,
                    stub.effect_info.clone(),
                );
        }
        (_, ResKind::Ref) => {
            let dst = expect_result_reg(result, Kind::Ref, "residual_call ref needs result");
            state
                .builder
                .residual_call_ref_canonical_via_target_with_effect_info(
                    fn_idx,
                    &call_args,
                    dst,
                    stub.effect_info.clone(),
                );
        }
        (_, ResKind::Float) => {
            let dst = expect_result_reg(result, Kind::Float, "residual_call float needs result");
            state
                .builder
                .residual_call_float_canonical_via_target_with_effect_info(
                    fn_idx,
                    &call_args,
                    dst,
                    stub.effect_info.clone(),
                );
        }
        (_, ResKind::Void) => unreachable!("void residual calls returned before policy dispatch"),
    }
}

fn expect_result_or_first_reg(args: &[Operand], result: Option<&Register>, kind: Kind) -> u16 {
    match result {
        Some(reg) if reg.kind == kind => reg.index,
        Some(reg) => panic!("expected result register of kind {:?}, got {:?}", kind, reg),
        None => expect_reg(&args[0], kind),
    }
}

fn expect_result_reg(result: Option<&Register>, kind: Kind, msg: &str) -> u16 {
    match result {
        Some(reg) if reg.kind == kind => reg.index,
        Some(reg) => panic!("expected result register of kind {:?}, got {:?}", kind, reg),
        None => panic!("{}", msg),
    }
}

thread_local! {
    /// Opname of the `Insn::Op` currently being lowered by `dispatch_op`,
    /// so operand-shape panics (`expect_reg` etc.) can name the opcode
    /// whose operand list violated the assembler's expectation instead
    /// of forcing the reader to grep the SSARepr for the bad operand.
    static CURRENT_DISPATCH_OP: std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) };
}

fn current_dispatch_op() -> String {
    CURRENT_DISPATCH_OP.with(|op| op.borrow().clone())
}

fn expect_reg(op: &Operand, expected: Kind) -> u16 {
    match op {
        Operand::Register(Register { kind, index }) if *kind == expected => *index,
        _ => panic!(
            "expected Register({:?}, _), got {:?} (op={})",
            expected,
            op,
            current_dispatch_op()
        ),
    }
}

fn expect_tlabel(op: &Operand) -> &TLabel {
    match op {
        Operand::TLabel(label) => label,
        _ => panic!("expected TLabel, got {:?}", op),
    }
}

fn expect_small_u16(op: &Operand) -> u16 {
    match op {
        Operand::ConstInt(value) => u16::try_from(*value).expect("expected u16-sized ConstInt"),
        _ => panic!("expected ConstInt(u16), got {:?}", op),
    }
}

/// `rpython/jit/metainterp/virtualizable.py:73 array_field_descrs[i]`:
/// extract the per-array index `i` carried by an
/// `Operand::Descr(DescrOperand::VableArrayField(i))` operand at the
/// trailing position of `getarrayitem_vable_X` / `setarrayitem_vable_X`
/// / `arraylen_vable` SSA ops (`jtransform.py:1880-1906`,
/// `:1865-1873`). Used by the assembler dispatch to populate the
/// canonical `BhDescr::VableArray` descriptor.
fn expect_descr_vable_array_field(op: &Operand) -> u16 {
    match op {
        Operand::Descr(rc) => match &**rc {
            DescrOperand::VableArrayField(idx) => *idx,
            other => panic!("expected DescrOperand::VableArrayField, got {:?}", other),
        },
        _ => panic!("expected Operand::Descr(VableArrayField), got {:?}", op),
    }
}

/// `rpython/jit/metainterp/virtualizable.py:58 array_descrs[i]`:
/// counterpart of `expect_descr_vable_array_field` for the second
/// trailing descr operand (the GcArray layout descr).  Always paired
/// with a `VableArrayField(i)` at the preceding position; the
/// dispatch arms cross-check the indices match.
fn expect_descr_vable_array(op: &Operand) -> u16 {
    match op {
        Operand::Descr(rc) => match &**rc {
            DescrOperand::VableArray(idx) => *idx,
            other => panic!("expected DescrOperand::VableArray, got {:?}", other),
        },
        _ => panic!("expected Operand::Descr(VableArray), got {:?}", op),
    }
}

/// `rpython/jit/metainterp/virtualizable.py:71 static_field_descrs[i]`:
/// extract the per-field index `i` carried by an
/// `Operand::Descr(DescrOperand::VableStaticField(i))` operand at the
/// trailing position of `getfield_vable_<kind>` / `setfield_vable_<kind>`
/// SSA ops (`jtransform.py:846-927`).  Used by the assembler dispatch
/// to populate the canonical `BhDescr::VableField` descriptor.
fn expect_descr_vable_static_field(op: &Operand) -> u16 {
    match op {
        Operand::Descr(rc) => match &**rc {
            DescrOperand::VableStaticField(idx) => *idx,
            other => panic!("expected DescrOperand::VableStaticField, got {:?}", other),
        },
        _ => panic!("expected Operand::Descr(VableStaticField), got {:?}", op),
    }
}

fn expect_vable_field_read_args(args: &[Operand]) -> (u16, u16) {
    assert_eq!(
        args.len(),
        2,
        "getfield_vable_* expects [vable, fielddescr]"
    );
    let vable_reg = expect_reg(&args[0], Kind::Ref);
    let field_idx = expect_descr_vable_static_field(&args[1]);
    (vable_reg, field_idx)
}

#[allow(dead_code)]
fn expect_vable_field_write_args(args: &[Operand], value_kind: Kind) -> (u16, u16, u16) {
    assert_eq!(
        args.len(),
        3,
        "setfield_vable_* expects [vable, value, fielddescr]"
    );
    let vable_reg = expect_reg(&args[0], Kind::Ref);
    let value_reg = expect_reg(&args[1], value_kind);
    let field_idx = expect_descr_vable_static_field(&args[2]);
    (vable_reg, value_reg, field_idx)
}

fn expect_matching_vable_array_descrs(field: &Operand, array: &Operand, opname: &str) -> u16 {
    let array_idx = expect_descr_vable_array_field(field);
    let array_descr_idx = expect_descr_vable_array(array);
    assert_eq!(
        array_idx, array_descr_idx,
        "{opname}: VableArrayField({array_idx}) and VableArray({array_descr_idx}) \
         descrs must reference the same array field",
    );
    array_idx
}

#[allow(dead_code)]
fn expect_vable_array_read_args(args: &[Operand]) -> (u16, u16, u16) {
    assert_eq!(
        args.len(),
        4,
        "getarrayitem_vable_* expects [vable, index, arrayfielddescr, arraydescr]"
    );
    let vable_reg = expect_reg(&args[0], Kind::Ref);
    let index_reg = expect_reg(&args[1], Kind::Int);
    let array_idx = expect_matching_vable_array_descrs(&args[2], &args[3], "getarrayitem_vable_*");
    (vable_reg, index_reg, array_idx)
}

/// Read prefix for getarrayitem_vable_*: returns (vable_reg, array_idx)
/// without dispatching on the index slot, leaving the caller to decide
/// between Register and ConstInt index.
fn expect_vable_array_read_prefix(args: &[Operand]) -> (u16, u16) {
    assert_eq!(
        args.len(),
        4,
        "getarrayitem_vable_* expects [vable, index, arrayfielddescr, arraydescr]"
    );
    let vable_reg = expect_reg(&args[0], Kind::Ref);
    let array_idx = expect_matching_vable_array_descrs(&args[2], &args[3], "getarrayitem_vable_*");
    (vable_reg, array_idx)
}

#[allow(dead_code)]
fn expect_vable_array_write_prefix(args: &[Operand]) -> (u16, u16, u16) {
    assert_eq!(
        args.len(),
        5,
        "setarrayitem_vable_* expects [vable, index, value, arrayfielddescr, arraydescr]"
    );
    let vable_reg = expect_reg(&args[0], Kind::Ref);
    let index_reg = expect_reg(&args[1], Kind::Int);
    let array_idx = expect_matching_vable_array_descrs(&args[3], &args[4], "setarrayitem_vable_*");
    (vable_reg, index_reg, array_idx)
}

/// Write prefix for setarrayitem_vable_* WITHOUT dispatching on the
/// index slot — returns (vable_reg, array_idx) only.  Caller dispatches
/// on args[1] (index) and args[2] (value) shape combinations.
fn expect_vable_array_write_prefix_no_idx(args: &[Operand]) -> (u16, u16) {
    assert_eq!(
        args.len(),
        5,
        "setarrayitem_vable_* expects [vable, index, value, arrayfielddescr, arraydescr]"
    );
    let vable_reg = expect_reg(&args[0], Kind::Ref);
    let array_idx = expect_matching_vable_array_descrs(&args[3], &args[4], "setarrayitem_vable_*");
    (vable_reg, array_idx)
}

fn expect_vable_arraylen_args(args: &[Operand]) -> (u16, u16) {
    assert_eq!(
        args.len(),
        3,
        "arraylen_vable expects [vable, arrayfielddescr, arraydescr]"
    );
    let vable_reg = expect_reg(&args[0], Kind::Ref);
    let array_idx = expect_matching_vable_array_descrs(&args[1], &args[2], "arraylen_vable");
    (vable_reg, array_idx)
}

/// Resolve a `DescrOperand::Bh(BhDescr::Array { .. })` operand onto the
/// builder's descrs pool and return the pool index.  RPython
/// `assembler.py _encode_descr` — the descr operand of a heap
/// array op is appended to `Assembler.descrs` and encoded as a 2-byte
/// index.
fn expect_field_bh_descr(op: &Operand, ctx: &'static str) -> majit_jitcode::jitcode::BhDescr {
    match op {
        Operand::Descr(d) => match &**d {
            DescrOperand::Bh(field @ majit_jitcode::jitcode::BhDescr::Field { .. }) => {
                field.clone()
            }
            other => panic!("{ctx} expects DescrOperand::Bh(BhDescr::Field), got {other:?}"),
        },
        other => panic!("{ctx} expects a descr operand, got {other:?}"),
    }
}

fn expect_array_bh_descr(state: &mut AssemblyState, op: &Operand, ctx: &'static str) -> u16 {
    match op {
        Operand::Descr(d) => match &**d {
            DescrOperand::Bh(bh @ majit_jitcode::jitcode::BhDescr::Array { .. }) => {
                state.builder.add_array_descr(bh.clone())
            }
            other => panic!("{ctx} expects DescrOperand::Bh(BhDescr::Array), got {other:?}"),
        },
        other => panic!("{ctx} expects a descr operand, got {other:?}"),
    }
}

fn expect_int_reg_or_pool(state: &mut AssemblyState, op: &Operand) -> u16 {
    match op {
        Operand::Register(Register {
            kind: Kind::Int,
            index,
        }) => *index,
        Operand::ConstInt(value) => {
            let idx = state.builder.add_const_i(*value);
            state.builder.num_regs_i() + idx
        }
        other => panic!(
            "expected Int register or ConstInt routable through constant pool, got {:?}",
            other
        ),
    }
}

/// True when both integer positions are real registers, so the fixed
/// `ii>i`/`iiL` emitters' `touch_reg` bound (`reg < num_regs_i`) holds.
///
/// A `ConstInt` becomes either a `ShortConst` or — when it does not fit the
/// signed byte — a constants-pool slot, which
/// [`expect_int_reg_or_pool`] numbers `num_regs_i + idx`.  `JitCodeIntOperand`
/// spells that slot `Register` because `assembler.py` gives it the same `i`
/// argcode (`blackhole.py` `registers_i` spans registers then constants), but
/// only the `*_operands` emitters account for the wider namespace with
/// `touch_int_reg_or_pool_slot`.  Routing a pool slot to the fixed emitter
/// trips that assert once register counts are frozen, and silently inflates
/// `num_regs_i` — shifting every later pool slot — before they are.
fn both_int_operands_are_registers(lhs: &Operand, rhs: &Operand) -> bool {
    matches!(lhs, Operand::Register(_)) && matches!(rhs, Operand::Register(_))
}

fn expect_jitcode_int_operand(
    state: &mut AssemblyState,
    op: &Operand,
    allow_short: bool,
) -> JitCodeIntOperand {
    match op {
        Operand::ConstInt(value) if allow_short && (-128..=127).contains(value) => {
            JitCodeIntOperand::ShortConst(*value as i8)
        }
        _ => JitCodeIntOperand::Register(expect_int_reg_or_pool(state, op)),
    }
}

/// Normalize a `ConstRef` value before it enters the jitcode `constants_r`
/// pool.  A tagged small-int immediate would be baked verbatim as a
/// `ConstPtr(GcRef(tagged))` that an in-trace `GuardClass` derefs as a heap
/// `W_IntObject` (fannkuch `sign=-1`), and the pool is GC-root-walked
/// (`walk_jitcode_constants_refs`, at every collection) so a tagged slot is
/// also forwarded as a moving-GC pointer.  Re-realize it as a distinct heap box so the pool holds
/// only real gcrefs — byte-identical to the flag-false bake (`w_int_new` of a
/// fresh heap `W_IntObject`).  `w_int_new_unique` allocates regardless of
/// `CAN_BE_TAGGED` (never re-tags).  Gated so flag-false emission is
/// byte-identical (a non-tagged real pointer is word-aligned, low bit clear,
/// so the guard is inert).
fn untag_const_ref_value(value: i64) -> i64 {
    if pyre_object::tagged_int::CAN_BE_TAGGED
        && pyre_object::tagged_int::is_tagged_int(value as pyre_object::PyObjectRef)
    {
        pyre_object::intobject::w_int_new_unique(pyre_object::tagged_int::untag_int(
            value as pyre_object::PyObjectRef,
        )) as i64
    } else {
        value
    }
}

fn expect_ref_reg_or_pool(state: &mut AssemblyState, op: &Operand) -> u16 {
    match op {
        Operand::Register(Register {
            kind: Kind::Ref,
            index,
        }) => *index,
        Operand::ConstRef(value) => {
            let idx = state.builder.add_const_r(untag_const_ref_value(*value));
            state.builder.num_regs_r() + idx
        }
        other => panic!(
            "expected Ref register or ConstRef routable through constant pool, got {:?}",
            other
        ),
    }
}

fn expect_float_reg_or_pool(state: &mut AssemblyState, op: &Operand) -> u16 {
    match op {
        Operand::Register(Register {
            kind: Kind::Float,
            index,
        }) => *index,
        Operand::ConstFloat(value) => {
            let idx = state.builder.add_const_f(*value);
            state.builder.num_regs_f() + idx
        }
        other => panic!(
            "expected Float register or ConstFloat routable through constant pool, got {:?}",
            other
        ),
    }
}

#[allow(dead_code)]
fn expect_list_regs(op: &Operand, expected: Kind) -> Vec<u8> {
    match op {
        Operand::ListOfKind(ListOfKind { kind, content }) if *kind == expected => content
            .iter()
            .map(|item| {
                u8::try_from(expect_reg(item, expected)).expect("register index exceeds u8")
            })
            .collect(),
        _ => panic!("expected ListOfKind({:?}), got {:?}", expected, op),
    }
}

/// Like `expect_list_regs`, but also accepts `Operand::ConstInt` /
/// `Operand::ConstRef` entries inside the `ListOfKind` and routes them
/// through the runtime constant pool (`builder.add_const_i` /
/// `builder.add_const_r`), returning the pool-backed register index
/// (`num_regs_kind + pool_idx`).
///
/// Matches upstream `rpython/jit/codewriter/assembler.py:157-175`
/// behavior where the assembler handles both pre-colored `Register`s
/// and raw `Constant`s inside `ListOfKind` by emitting distinct
/// argcodes ('i'/'r'/'f' for Register, 'c' for Constant) and routing
/// Constants through the per-kind constant table at write time.
///
/// Pyre's production jit_merge_point historically emitted
/// pre-pool-routed `Register` operands (via
/// `ssa_emitter.rs::emit_portal_jit_merge_point`), so this function
/// keeps that path working.  It also makes the 7-arg
/// upstream-orthodox shape produced by
/// `codewriter.rs::portal_jit_merge_point_graph_args` +
/// `GraphFlattener` (which emits raw `ConstInt`/`ConstRef` inside
/// `ListOfKind` because pyre's flow-graph constants are not
/// pool-routed upstream) work end-to-end.
fn expect_list_regs_or_pool(state: &mut AssemblyState, op: &Operand, expected: Kind) -> Vec<u8> {
    let ListOfKind { kind, content } = match op {
        Operand::ListOfKind(list) => list,
        _ => panic!("expected ListOfKind({:?}), got {:?}", expected, op),
    };
    assert_eq!(
        *kind, expected,
        "list kind mismatch: expected {expected:?}, got {:?}",
        kind
    );
    content
        .iter()
        .map(|item| {
            let reg_idx = match item {
                Operand::Register(Register { kind, index }) => {
                    assert_eq!(
                        *kind, expected,
                        "register kind mismatch inside list: expected {expected:?}, got {:?}",
                        kind
                    );
                    *index
                }
                Operand::ConstInt(value) => {
                    assert_eq!(
                        expected,
                        Kind::Int,
                        "ConstInt found inside non-int ListOfKind({expected:?})"
                    );
                    let idx = state.builder.add_const_i(*value);
                    state.builder.num_regs_i() + idx
                }
                Operand::ConstRef(value) => {
                    assert_eq!(
                        expected,
                        Kind::Ref,
                        "ConstRef found inside non-ref ListOfKind({expected:?})"
                    );
                    let idx = state.builder.add_const_r(untag_const_ref_value(*value));
                    state.builder.num_regs_r() + idx
                }
                other => panic!(
                    "expected Register/ConstInt/ConstRef inside ListOfKind({expected:?}), \
                     got {other:?}"
                ),
            };
            // A pool-backed index (`num_regs_kind + pool_idx`) can exceed a
            // single operand byte even though `num_regs_kind` itself stays
            // under the ceiling.  Latch the builder's overflow flag and emit a
            // placeholder rather than panicking; `try_finish` then declines the
            // JitCode and the interpreter keeps running the trace's function.
            match u8::try_from(reg_idx) {
                Ok(byte) => byte,
                Err(_) => {
                    state.builder.note_encoding_overflow();
                    0
                }
            }
        })
        .collect()
}

fn expect_call_args(args: &[Operand]) -> Vec<JitCallArg> {
    args.iter().map(expect_call_arg).collect()
}

fn expect_call_arg(op: &Operand) -> JitCallArg {
    match op {
        Operand::Register(Register { kind, index }) => match kind {
            Kind::Int => JitCallArg::int(*index),
            Kind::Ref => JitCallArg::reference(*index),
            Kind::Float => JitCallArg::float(*index),
        },
        _ => panic!("expected typed call register, got {:?}", op),
    }
}

/// `flatten.py:333` source operand for `{kind}_copy`. The source may
/// be either a Register (argcode `'i'`/`'r'`/`'f'`) or a Constant
/// (argcode `'c'` in short form). `MoveSource` captures the shape so
/// the dispatch arm can lower to the right `JitCodeBuilder` method.
#[derive(Debug)]
enum MoveSource<'a> {
    Reg(&'a Operand),
    ConstInt(i64),
    ConstRef(i64),
    ConstFloat(i64),
}

fn source_operand<'a>(args: &'a [Operand], result: Option<&Register>) -> MoveSource<'a> {
    let src_operand = if result.is_some() { &args[0] } else { &args[1] };
    match src_operand {
        Operand::Register(_) => MoveSource::Reg(src_operand),
        Operand::ConstInt(v) => MoveSource::ConstInt(*v),
        Operand::ConstRef(v) => MoveSource::ConstRef(*v),
        Operand::ConstFloat(v) => MoveSource::ConstFloat(*v),
        _ => panic!(
            "source_operand: unexpected source operand {:?}",
            src_operand
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::super::flatten::{CallDescrStub, CallFlavor, DescrOperand, Label, SwitchDictDescr};
    use super::*;

    fn r(kind: Kind, index: u16) -> Register {
        Register::new(kind, index)
    }

    #[test]
    fn assemble_empty_ssarepr_produces_valid_jitcode() {
        let mut ssarepr = SSARepr::new("empty");
        let jitcode = assemble(&mut ssarepr, JitCodeBuilder::default(), None);
        assert_eq!(jitcode.name, "empty");
        assert!(jitcode.code.is_empty(), "empty SSARepr -> empty code");
        assert_eq!(ssarepr.insns_pos, Some(Vec::new()));
    }

    #[test]
    fn try_assemble_declines_single_byte_register_overflow() {
        let mut ssarepr = SSARepr::new("overflow");
        let mut assembler = Assembler::new();
        let jitcode = assembler.try_assemble(
            &mut ssarepr,
            JitCodeBuilder::default(),
            Some(NumRegs {
                ref_: 256,
                ..NumRegs::default()
            }),
        );
        assert!(jitcode.is_none());
    }

    #[test]
    fn try_assemble_declines_constant_pool_byte_overflow() {
        // The register file alone fits (`num_regs.ref_ < 256`), so the
        // up-front gate passes. Constants routed through the per-kind pool
        // occupy the SAME one-byte namespace above `count_regs[ref]`, and the
        // byte is synthesized as they are appended (`assembler.py:131-137`), so
        // enough of them push a `jit_merge_point` red-list operand past 255 at
        // the `assembler.py` `assert 0 <= val < 256`. The runtime seam must
        // decline there rather than abort while emitting the operand.
        const NUM_REGS_R: u16 = 200;
        let reds_r: Vec<Operand> = (0..100).map(|i| Operand::ConstRef(i * 8)).collect();
        let empty = |kind| {
            Operand::ListOfKind(ListOfKind {
                kind,
                content: Vec::new(),
            })
        };

        let mut ssarepr = SSARepr::new("pool_overflow");
        ssarepr.insns.push(Insn::op(
            "jit_merge_point",
            vec![
                Operand::ConstInt(0),
                empty(Kind::Int),
                empty(Kind::Ref),
                empty(Kind::Float),
                empty(Kind::Int),
                Operand::ListOfKind(ListOfKind {
                    kind: Kind::Ref,
                    content: reds_r,
                }),
                empty(Kind::Float),
            ],
        ));

        let mut assembler = Assembler::new();
        let jitcode = assembler.try_assemble(
            &mut ssarepr,
            JitCodeBuilder::default(),
            Some(NumRegs {
                ref_: NUM_REGS_R,
                ..NumRegs::default()
            }),
        );
        assert!(
            jitcode.is_none(),
            "200 ref registers + 100 pooled ref constants exceeds the one-byte \
             operand namespace and must decline"
        );
    }

    #[test]
    fn assemble_records_insn_positions_and_liveness() {
        let mut ssarepr = SSARepr::new("live");
        ssarepr
            .insns
            .push(Insn::live(vec![Operand::Register(r(Kind::Ref, 0))]));
        ssarepr.insns.push(Insn::op(
            "ref_return",
            vec![Operand::Register(r(Kind::Ref, 0))],
        ));

        let mut assembler = Assembler::new();
        let before_ops = assembler.num_liveness_ops();
        let jitcode = assembler.assemble(
            &mut ssarepr,
            JitCodeBuilder::default(),
            Some(NumRegs {
                ref_: 1,
                ..NumRegs::default()
            }),
        );

        // `num_liveness_ops` and `all_liveness` live on the thread-local
        // `ASSEMBLER_STATE` (the single RPython-`Assembler` analog); other
        // tests running earlier on the same thread may already have
        // bumped the counter, so assert on the delta and the non-empty
        // buffer rather than a raw == 1.
        assert_eq!(assembler.num_liveness_ops() - before_ops, 1);
        assert!(!assembler.all_liveness().is_empty());
        assert_eq!(ssarepr.insns_pos.as_ref().map(Vec::len), Some(2));
        assert!(!jitcode.code.is_empty());
    }

    #[test]
    fn assemble_accumulates_used_wellknown_insns() {
        let mut assembler = Assembler::new();

        let mut first = SSARepr::new("first");
        first
            .insns
            .push(Insn::live(vec![Operand::Register(r(Kind::Ref, 0))]));
        first.insns.push(Insn::op(
            "ref_return",
            vec![Operand::Register(r(Kind::Ref, 0))],
        ));
        assembler.assemble(
            &mut first,
            JitCodeBuilder::default(),
            Some(NumRegs {
                ref_: 1,
                ..NumRegs::default()
            }),
        );

        let mut second = SSARepr::new("second");
        second
            .insns
            .push(Insn::op("goto", vec![Operand::TLabel(TLabel::new("L1"))]));
        second.insns.push(Insn::Label(Label::new("L1")));
        assembler.assemble(&mut second, JitCodeBuilder::default(), None);

        let insns = assembler.insns_snapshot();
        let wellknown = majit_metainterp::jitcode::wellknown_bh_insns();
        assert_eq!(insns.get("live/"), wellknown.get("live/"));
        assert_eq!(insns.get("ref_return/r"), wellknown.get("ref_return/r"));
        assert_eq!(insns.get("goto/L"), wellknown.get("goto/L"));
    }

    #[test]
    fn assemble_allocates_only_the_short_constant_shapes_that_appear() {
        let mut assembler = Assembler::new();
        let mut ssarepr = SSARepr::new("short_constants");
        ssarepr.insns.push(Insn::op_with_result(
            "int_sub",
            vec![Operand::ConstInt(48), Operand::Register(r(Kind::Int, 1))],
            r(Kind::Int, 2),
        ));
        ssarepr.insns.push(Insn::op(
            "goto_if_not_int_gt",
            vec![
                Operand::Register(r(Kind::Int, 2)),
                Operand::ConstInt(2),
                Operand::TLabel(TLabel::new("done")),
            ],
        ));
        ssarepr.insns.push(Insn::Label(Label::new("done")));
        ssarepr.insns.push(Insn::op(
            "ref_return",
            vec![Operand::Register(r(Kind::Ref, 0))],
        ));

        let jitcode = assembler.assemble(
            &mut ssarepr,
            JitCodeBuilder::default(),
            Some(NumRegs {
                int: 3,
                ref_: 1,
                ..NumRegs::default()
            }),
        );
        let insns = assembler.insns_snapshot();
        let sub = *insns.get("int_sub/ci>i").expect("short lhs key");
        let branch = *insns
            .get("goto_if_not_int_gt/icL")
            .expect("short rhs branch key");
        assert_ne!(sub, branch);
        assert!(!majit_jitcode::insns::is_reserved_opcode_byte(sub));
        assert!(!majit_jitcode::insns::is_reserved_opcode_byte(branch));
        assert_eq!(jitcode.code[0], sub);
        assert_eq!(jitcode.code[4], branch);
    }

    #[test]
    fn assemble_accumulates_canonical_last_exc_value_and_jit_merge_point_keys() {
        let mut assembler = Assembler::new();

        // Portal A — small jdindex ConstInt(0) lowers to `c` argcode
        // (insn_key arg[0] match: -128..=127 → 'c'). One
        // `jit_merge_point` per portal jitcode mirrors
        // `jtransform.py handle_jit_marker__jit_merge_point` (which emits exactly one
        // `jit_merge_point` per portal); the assembler accumulates
        // both shapes across separate `assemble` calls.
        let mut portal_small = SSARepr::new("portal_small");
        portal_small.insns.push(Insn::op(
            "jit_merge_point",
            vec![
                Operand::ConstInt(0),
                Operand::ListOfKind(ListOfKind::new(
                    Kind::Int,
                    vec![
                        Operand::Register(Register::new(Kind::Int, 0)),
                        Operand::Register(Register::new(Kind::Int, 1)),
                    ],
                )),
                Operand::ListOfKind(ListOfKind::new(
                    Kind::Ref,
                    vec![Operand::Register(Register::new(Kind::Ref, 0))],
                )),
                Operand::ListOfKind(ListOfKind::new(Kind::Float, Vec::new())),
                Operand::ListOfKind(ListOfKind::new(Kind::Int, Vec::new())),
                Operand::ListOfKind(ListOfKind::new(
                    Kind::Ref,
                    vec![
                        Operand::Register(Register::new(Kind::Ref, 1)),
                        Operand::Register(Register::new(Kind::Ref, 2)),
                    ],
                )),
                Operand::ListOfKind(ListOfKind::new(Kind::Float, Vec::new())),
            ],
        ));
        portal_small.insns.push(Insn::op_with_result(
            "last_exc_value",
            Vec::new(),
            Register::new(Kind::Ref, 3),
        ));
        assembler.assemble(
            &mut portal_small,
            JitCodeBuilder::default(),
            Some(NumRegs {
                int: 2,
                ref_: 4,
                ..NumRegs::default()
            }),
        );

        // Portal B — large jdindex ConstInt(200) lowers to `i` argcode
        // (out of -128..=127), exercising the const-patches path.
        let mut portal_large = SSARepr::new("portal_large");
        portal_large.insns.push(Insn::op(
            "jit_merge_point",
            vec![
                Operand::ConstInt(200),
                Operand::ListOfKind(ListOfKind::new(Kind::Int, Vec::new())),
                Operand::ListOfKind(ListOfKind::new(Kind::Ref, Vec::new())),
                Operand::ListOfKind(ListOfKind::new(Kind::Float, Vec::new())),
                Operand::ListOfKind(ListOfKind::new(Kind::Int, Vec::new())),
                Operand::ListOfKind(ListOfKind::new(Kind::Ref, Vec::new())),
                Operand::ListOfKind(ListOfKind::new(Kind::Float, Vec::new())),
            ],
        ));
        assembler.assemble(&mut portal_large, JitCodeBuilder::default(), None);

        let insns = assembler.insns_snapshot();
        let wellknown = majit_metainterp::jitcode::wellknown_bh_insns();
        assert_eq!(
            insns.get("jit_merge_point/cIRFIRF"),
            wellknown.get("jit_merge_point/cIRFIRF")
        );
        assert_eq!(
            insns.get("jit_merge_point/iIRFIRF"),
            wellknown.get("jit_merge_point/iIRFIRF")
        );
        assert_eq!(
            insns.get("last_exc_value/>r"),
            wellknown.get("last_exc_value/>r")
        );
    }

    #[test]
    fn assemble_accumulates_canonical_goto_if_exception_mismatch_key() {
        let mut assembler = Assembler::new();

        let mut ssarepr = SSARepr::new("exc_match");
        ssarepr.insns.push(Insn::op(
            "goto_if_exception_mismatch",
            vec![Operand::ConstInt(7), Operand::TLabel(TLabel::new("L1"))],
        ));
        ssarepr.insns.push(Insn::Label(Label::new("L1")));
        ssarepr.insns.push(Insn::op(
            "ref_return",
            vec![Operand::Register(r(Kind::Ref, 0))],
        ));
        assembler.assemble(
            &mut ssarepr,
            JitCodeBuilder::default(),
            Some(NumRegs {
                ref_: 1,
                ..NumRegs::default()
            }),
        );

        let insns = assembler.insns_snapshot();
        let wellknown = majit_metainterp::jitcode::wellknown_bh_insns();
        assert_eq!(
            insns.get("goto_if_exception_mismatch/iL"),
            wellknown.get("goto_if_exception_mismatch/iL")
        );
    }

    #[test]
    fn assemble_patches_jumps_through_labels() {
        let mut ssarepr = SSARepr::new("jump");
        ssarepr
            .insns
            .push(Insn::op("goto", vec![Operand::TLabel(TLabel::new("L1"))]));
        ssarepr.insns.push(Insn::Label(Label::new("L1")));
        ssarepr.insns.push(Insn::op(
            "ref_return",
            vec![Operand::Register(r(Kind::Ref, 0))],
        ));

        let jitcode = assemble(
            &mut ssarepr,
            JitCodeBuilder::default(),
            Some(NumRegs {
                ref_: 1,
                ..NumRegs::default()
            }),
        );

        assert_eq!(jitcode.follow_jump(3), 3);
    }

    #[test]
    fn assemble_int_copy_from_constant_uses_u16_immediate() {
        let mut ssarepr = SSARepr::new("const_copy");
        // Past the signed-byte short form, inside the u16 jump-target
        // space: inline immediate, no `constants_i` slot.
        ssarepr.insns.push(Insn::op_with_result(
            "int_copy",
            vec![Operand::ConstInt(4242)],
            Register::new(Kind::Int, 0),
        ));

        let jitcode = assemble(
            &mut ssarepr,
            JitCodeBuilder::default(),
            Some(NumRegs {
                int: 1,
                ..NumRegs::default()
            }),
        );

        let copy_opcode = *majit_metainterp::jitcode::wellknown_bh_insns()
            .get("int_copy/d>i")
            .expect("int_copy/d>i must be registered in wellknown insns");
        assert!(jitcode.constants_i.is_empty());
        assert_eq!(jitcode.code[0], copy_opcode);
        assert_eq!(jitcode.code[1], (4242u16 & 0xff) as u8);
        assert_eq!(jitcode.code[2], (4242u16 >> 8) as u8);
        assert_eq!(jitcode.code[3], 0);
    }

    #[test]
    fn assemble_int_copy_from_constant_uses_copy_opcode_and_constant_pool() {
        let mut ssarepr = SSARepr::new("const_copy_wide");
        // Outside the u16 immediate range, so the value takes a
        // `constants_i` slot and the `i` argcode.
        ssarepr.insns.push(Insn::op_with_result(
            "int_copy",
            vec![Operand::ConstInt(70_000)],
            Register::new(Kind::Int, 0),
        ));

        let jitcode = assemble(
            &mut ssarepr,
            JitCodeBuilder::default(),
            Some(NumRegs {
                int: 1,
                ..NumRegs::default()
            }),
        );

        let copy_opcode = *majit_metainterp::jitcode::wellknown_bh_insns()
            .get("int_copy/i>i")
            .expect("int_copy must be registered in wellknown insns");
        assert_eq!(jitcode.constants_i, vec![70_000]);
        assert_eq!(jitcode.code[0], copy_opcode);
        assert_eq!(jitcode.code[1], 1);
        assert_eq!(jitcode.code[2], 0);
    }

    #[test]
    fn assemble_int_copy_from_small_constant_uses_the_short_form() {
        let mut ssarepr = SSARepr::new("small_const_copy");
        ssarepr.insns.push(Insn::op_with_result(
            "int_copy",
            vec![Operand::ConstInt(42)],
            Register::new(Kind::Int, 0),
        ));

        let jitcode = assemble(
            &mut ssarepr,
            JitCodeBuilder::default(),
            Some(NumRegs {
                int: 1,
                ..NumRegs::default()
            }),
        );

        // `int_copy` is in `USE_C_FORM` (`assembler.py`), so
        // `assembler.py:99-107` writes a byte-sized value inline and
        // `assembler.py:172` spells the argcode `c` instead of `i`.
        let copy_opcode = *majit_metainterp::jitcode::wellknown_bh_insns()
            .get("int_copy/c>i")
            .expect("int_copy/c>i must be registered in wellknown insns");
        assert!(jitcode.constants_i.is_empty());
        assert_eq!(jitcode.code[0], copy_opcode);
        assert_eq!(jitcode.code[1], 42);
        assert_eq!(jitcode.code[2], 0);
    }

    #[test]
    fn assemble_setfield_vable_r_accepts_const_ref_source() {
        let mut ssarepr = SSARepr::new("vable_setfield_const_ref");
        ssarepr.insns.push(Insn::op(
            "setfield_vable_r",
            vec![
                Operand::Register(r(Kind::Ref, 0)),
                Operand::ConstRef(0),
                Operand::descr_vable_static_field(3),
            ],
        ));

        let jitcode = assemble(
            &mut ssarepr,
            JitCodeBuilder::default(),
            Some(NumRegs {
                ref_: 1,
                ..NumRegs::default()
            }),
        );

        let opcode = *majit_metainterp::jitcode::wellknown_bh_insns()
            .get("setfield_vable_r/rrd")
            .expect("setfield_vable_r must be registered in wellknown insns");
        assert_eq!(jitcode.code, vec![opcode, 0, 1, 0, 0]);
        assert_eq!(jitcode.constants_r, vec![0]);
        assert!(matches!(
            &jitcode.exec.descrs[0],
            majit_metainterp::jitcode::RuntimeBhDescr::Descr(descr)
                if matches!(
                    descr.as_ref(),
                    majit_metainterp::blackhole::BhDescr::VableField { index: 3 }
                )
        ));
    }

    #[test]
    fn assemble_ref_return_accepts_const_ref() {
        let mut ssarepr = SSARepr::new("ref_return_const");
        ssarepr
            .insns
            .push(Insn::op("ref_return", vec![Operand::ConstRef(42)]));

        let jitcode = assemble(
            &mut ssarepr,
            JitCodeBuilder::default(),
            Some(NumRegs {
                ref_: 2,
                ..NumRegs::default()
            }),
        );

        let opcode = *majit_metainterp::jitcode::wellknown_bh_insns()
            .get("ref_return/r")
            .expect("ref_return must be registered in wellknown insns");
        // The constant routes into the ref constant pool; the operand byte
        // resolves to `num_regs_r + pool_idx` (2 + 0) in the const window.
        assert_eq!(jitcode.code, vec![opcode, 2]);
        assert_eq!(jitcode.constants_r, vec![42]);
    }

    #[test]
    fn assemble_ref_return_preserves_result_form() {
        let mut ssarepr = SSARepr::new("ref_return_result_form");
        ssarepr.insns.push(Insn::op_with_result(
            "ref_return",
            vec![],
            Register::new(Kind::Ref, 0),
        ));

        let jitcode = assemble(
            &mut ssarepr,
            JitCodeBuilder::default(),
            Some(NumRegs {
                ref_: 1,
                ..NumRegs::default()
            }),
        );

        let opcode = *majit_metainterp::jitcode::wellknown_bh_insns()
            .get("ref_return/r")
            .expect("ref_return must be registered in wellknown insns");
        assert_eq!(jitcode.code, vec![opcode, 0]);
    }

    #[test]
    fn assemble_raise_accepts_const_ref() {
        let mut ssarepr = SSARepr::new("raise_const");
        // A `ConstRef` sentinel models a real gcref: word-aligned, low bit
        // clear. An odd value would read as a tagged int immediate and be
        // re-realized into a fresh heap box by `untag_const_ref_value`.
        ssarepr
            .insns
            .push(Insn::op("raise", vec![Operand::ConstRef(8)]));

        let jitcode = assemble(
            &mut ssarepr,
            JitCodeBuilder::default(),
            Some(NumRegs {
                ref_: 2,
                ..NumRegs::default()
            }),
        );

        let opcode = *majit_metainterp::jitcode::wellknown_bh_insns()
            .get("raise/r")
            .expect("raise must be registered in wellknown insns");
        assert_eq!(jitcode.code, vec![opcode, 2]);
        assert_eq!(jitcode.constants_r, vec![8]);
    }

    #[test]
    fn assemble_ptr_nonzero_uses_canonical_ref_nullity_opcode() {
        let mut ssarepr = SSARepr::new("ptr_nonzero");
        ssarepr.insns.push(Insn::op_with_result(
            "ptr_nonzero",
            vec![Operand::Register(r(Kind::Ref, 0))],
            Register::new(Kind::Int, 0),
        ));

        let jitcode = assemble(
            &mut ssarepr,
            JitCodeBuilder::default(),
            Some(NumRegs {
                int: 1,
                ref_: 1,
                ..NumRegs::default()
            }),
        );

        let opcode = *majit_metainterp::jitcode::wellknown_bh_insns()
            .get("ptr_nonzero/r>i")
            .expect("ptr_nonzero must be registered in wellknown insns");
        // Canonical `r>i` 1-byte encoding (`assembler.py:165-174`):
        // [opcode][src_reg][dst_reg].
        assert_eq!(jitcode.code[0], opcode);
        assert_eq!(jitcode.code[1], 0);
        assert_eq!(jitcode.code[2], 0);
    }

    #[test]
    fn assemble_goto_if_not_ptr_nonzero_uses_canonical_branch_opcode() {
        let mut ssarepr = SSARepr::new("ptr_branch");
        ssarepr.insns.push(Insn::op(
            "goto_if_not_ptr_nonzero",
            vec![
                Operand::Register(r(Kind::Ref, 0)),
                Operand::TLabel(TLabel::new("L1")),
            ],
        ));
        ssarepr.insns.push(Insn::Label(Label::new("L1")));
        // Append a trailing return so `L1` is not at end-of-code.
        // RPython `jitcode.py follow_jump` asserts
        // `labelvalue < len(code)` strictly, so the label must not
        // resolve to `len(code)` itself.
        ssarepr.insns.push(Insn::op(
            "ref_return",
            vec![Operand::Register(r(Kind::Ref, 0))],
        ));

        let jitcode = assemble(
            &mut ssarepr,
            JitCodeBuilder::default(),
            Some(NumRegs {
                ref_: 1,
                ..NumRegs::default()
            }),
        );

        let opcode = *majit_metainterp::jitcode::wellknown_bh_insns()
            .get("goto_if_not_ptr_nonzero/rL")
            .expect("ptr_nonzero branch must be registered in wellknown insns");
        assert_eq!(jitcode.code[0], opcode);
        // Canonical `rL` encoding: [opcode:1][src:u8][target:u16] = 4
        // bytes; L1 lands at position 4 (the ref_return start).
        assert_eq!(jitcode.follow_jump(4), 4);
    }

    #[test]
    fn assemble_conditional_call_value_ir_r_keeps_ref_register_bank() {
        let mut builder = JitCodeBuilder::default();
        let fn_idx = builder.add_fn_ptr(0x7777usize as *const ());
        let mut ssarepr = SSARepr::new("cond_call_value_ir_r");
        ssarepr.insns.push(Insn::op_with_result(
            "conditional_call_value_ir_r",
            vec![
                Operand::ConstInt(i64::from(fn_idx)),
                Operand::Register(Register::new(Kind::Ref, 1)),
                Operand::Register(Register::new(Kind::Int, 0)),
                Operand::Register(Register::new(Kind::Ref, 2)),
            ],
            Register::new(Kind::Ref, 4),
        ));

        let jitcode = assemble(
            &mut ssarepr,
            builder,
            Some(NumRegs {
                int: 1,
                ref_: 5,
                ..NumRegs::default()
            }),
        );

        assert_eq!(jitcode.num_regs_i(), 1);
        assert_eq!(jitcode.num_regs_r(), 5);
    }

    #[test]
    fn assemble_record_known_result_r_ir_v_keeps_ref_register_bank() {
        let mut builder = JitCodeBuilder::default();
        let fn_idx = builder.add_fn_ptr(0x7777usize as *const ());
        let mut ssarepr = SSARepr::new("record_known_result_r_ir_v");
        ssarepr.insns.push(Insn::op(
            "record_known_result_r_ir_v",
            vec![
                Operand::ConstInt(i64::from(fn_idx)),
                Operand::Register(Register::new(Kind::Ref, 2)),
                Operand::Register(Register::new(Kind::Int, 0)),
                Operand::Register(Register::new(Kind::Ref, 1)),
            ],
        ));

        let jitcode = assemble(
            &mut ssarepr,
            builder,
            Some(NumRegs {
                int: 1,
                ref_: 3,
                ..NumRegs::default()
            }),
        );

        assert_eq!(jitcode.num_regs_i(), 1);
        assert_eq!(jitcode.num_regs_r(), 3);
    }

    #[test]
    fn assemble_canonical_helper_call_family_publishes_iiird_keys() {
        let mut assembler = Assembler::new();
        let mut builder = JitCodeBuilder::default();
        let fn_idx = builder.add_fn_ptr(0x7777usize as *const ());

        let mut ssarepr = SSARepr::new("canonical_call_family");
        ssarepr.insns.push(Insn::op(
            "conditional_call_ir_v",
            vec![
                Operand::ConstInt(i64::from(fn_idx)),
                Operand::Register(Register::new(Kind::Int, 0)),
                Operand::Register(Register::new(Kind::Int, 1)),
                Operand::Register(Register::new(Kind::Ref, 0)),
            ],
        ));
        ssarepr.insns.push(Insn::op_with_result(
            "conditional_call_value_ir_r",
            vec![
                Operand::ConstInt(i64::from(fn_idx)),
                Operand::Register(Register::new(Kind::Ref, 1)),
                Operand::Register(Register::new(Kind::Int, 1)),
                Operand::Register(Register::new(Kind::Ref, 0)),
            ],
            Register::new(Kind::Ref, 2),
        ));
        ssarepr.insns.push(Insn::op(
            "record_known_result_r_ir_v",
            vec![
                Operand::ConstInt(i64::from(fn_idx)),
                Operand::Register(Register::new(Kind::Ref, 2)),
                Operand::Register(Register::new(Kind::Int, 1)),
                Operand::Register(Register::new(Kind::Ref, 0)),
            ],
        ));
        assembler.assemble(
            &mut ssarepr,
            builder,
            Some(NumRegs {
                int: 2,
                ref_: 3,
                ..NumRegs::default()
            }),
        );

        let insns = assembler.insns_snapshot();
        assert_eq!(
            insns.get("conditional_call_ir_v/iiIRd").copied(),
            Some(majit_jitcode::insns::BC_CONDITIONAL_CALL_IR_V),
        );
        assert_eq!(
            insns.get("conditional_call_value_ir_r/riIRd>r").copied(),
            Some(majit_jitcode::insns::BC_CONDITIONAL_CALL_VALUE_IR_R),
        );
        assert_eq!(
            insns.get("record_known_result_r_ir_v/riIRd").copied(),
            Some(majit_jitcode::insns::BC_RECORD_KNOWN_RESULT_R_IR_V),
        );
    }

    #[test]
    #[should_panic(expected = "unimplemented opname")]
    fn assemble_legacy_conditional_call_alias_is_rejected() {
        let mut ssarepr = SSARepr::new("legacy_conditional_call_alias");
        ssarepr.insns.push(Insn::op(
            "conditional_call_void",
            vec![
                Operand::ConstInt(7),
                Operand::Register(Register::new(Kind::Int, 0)),
                Operand::Register(Register::new(Kind::Int, 1)),
                Operand::Register(Register::new(Kind::Ref, 0)),
            ],
        ));

        let _ = assemble(
            &mut ssarepr,
            JitCodeBuilder::default(),
            Some(NumRegs {
                int: 2,
                ref_: 1,
                ..NumRegs::default()
            }),
        );
    }

    #[test]
    #[should_panic(expected = "unimplemented opname")]
    fn assemble_legacy_record_known_result_alias_is_rejected() {
        let mut ssarepr = SSARepr::new("legacy_record_known_result_alias");
        ssarepr.insns.push(Insn::op(
            "record_known_result_ref",
            vec![
                Operand::ConstInt(7),
                Operand::Register(Register::new(Kind::Ref, 2)),
                Operand::Register(Register::new(Kind::Int, 0)),
                Operand::Register(Register::new(Kind::Ref, 1)),
            ],
        ));

        let _ = assemble(
            &mut ssarepr,
            JitCodeBuilder::default(),
            Some(NumRegs {
                int: 1,
                ref_: 3,
                ..NumRegs::default()
            }),
        );
    }

    #[test]
    fn assemble_residual_call_irf_f_supports_pure_float_flavor() {
        // CallFlavor::Pure routes through
        // the canonical `BC_RESIDUAL_CALL_IRF_F` (=161) with the
        // calldescr carrying ElidableCanRaise.  The walker
        // (`pyjitpl/dispatch.rs`) reads
        // `effectinfo.check_is_elidable()` and runs
        // `record_result_of_call_pure` mirroring `pyjitpl.py`.
        let mut ssarepr = SSARepr::new("residual_call_irf_f");
        let mut builder = JitCodeBuilder::default();
        let fn_idx = builder.add_fn_ptr(0x7777usize as *const ());
        ssarepr.insns.push(Insn::op_with_result(
            "residual_call_irf_f",
            vec![
                Operand::ConstInt(fn_idx as i64),
                Operand::ListOfKind(ListOfKind::new(
                    Kind::Int,
                    vec![Operand::Register(Register::new(Kind::Int, 0))],
                )),
                Operand::ListOfKind(ListOfKind::new(
                    Kind::Ref,
                    vec![Operand::Register(Register::new(Kind::Ref, 0))],
                )),
                Operand::ListOfKind(ListOfKind::new(
                    Kind::Float,
                    vec![Operand::Register(Register::new(Kind::Float, 0))],
                )),
                Operand::descr(DescrOperand::CallDescrStub(CallDescrStub {
                    effect_info: super::super::flatten::effect_info_for_call_flavor(
                        CallFlavor::PureCanRaise,
                    ),
                    arg_kinds: vec![Kind::Int, Kind::Ref, Kind::Float],
                    result_kind: Some(Kind::Float),
                    void_word_abi: false,
                })),
            ],
            Register::new(Kind::Float, 1),
        ));

        let jitcode = assemble(
            &mut ssarepr,
            builder,
            Some(NumRegs {
                int: 1,
                ref_: 1,
                float: 2,
            }),
        );

        // majit_jitcode::insns::BC_RESIDUAL_CALL_IRF_F = 168.
        assert_eq!(jitcode.code[0], 168);
    }

    #[test]
    fn assemble_residual_call_ir_v_supports_release_gil_void_flavor() {
        let mut ssarepr = SSARepr::new("residual_call_ir_v");
        let mut builder = JitCodeBuilder::default();
        let fn_idx = builder.add_fn_ptr(0x7777usize as *const ());
        ssarepr.insns.push(Insn::op(
            "residual_call_ir_v",
            vec![
                Operand::ConstInt(fn_idx as i64),
                Operand::ListOfKind(ListOfKind::new(
                    Kind::Int,
                    vec![Operand::Register(Register::new(Kind::Int, 0))],
                )),
                Operand::ListOfKind(ListOfKind::new(
                    Kind::Ref,
                    vec![Operand::Register(Register::new(Kind::Ref, 0))],
                )),
                Operand::descr(DescrOperand::CallDescrStub(CallDescrStub {
                    // `call.py` `getcalldescr` fills the target from
                    // `_call_aroundstate_target_`.
                    effect_info: majit_ir::EffectInfo {
                        call_release_gil_target: (0x7777, 0),
                        ..majit_ir::EffectInfo::MOST_GENERAL.clone()
                    },
                    arg_kinds: vec![Kind::Int, Kind::Ref],
                    result_kind: None,
                    void_word_abi: false,
                })),
            ],
        ));

        let jitcode = assemble(
            &mut ssarepr,
            builder,
            Some(NumRegs {
                int: 1,
                ref_: 1,
                ..NumRegs::default()
            }),
        );

        let expected = majit_metainterp::jitcode::wellknown_bh_insns()
            .get("residual_call_ir_v/iIRd")
            .copied()
            .unwrap();
        assert_eq!(jitcode.code[0], expected);
        let descr_idx = (jitcode.code[6] as u16) | ((jitcode.code[7] as u16) << 8);
        let descr = jitcode.exec.descrs[descr_idx as usize]
            .as_bh_descr()
            .unwrap()
            .as_calldescr();
        assert!(descr.extra_info.is_call_release_gil());
    }

    #[test]
    fn assemble_inline_call_r_r_emits_the_canonical_descr_varlist_shape() {
        let target = Arc::new(JitCodeBuilder::default().finish());
        let mut ssarepr = SSARepr::new("inline_call_r_r");
        ssarepr.insns.push(Insn::op_with_result(
            "inline_call_r_r",
            vec![
                Operand::descr(DescrOperand::JitCode(Arc::clone(&target))),
                Operand::ListOfKind(ListOfKind::new(
                    Kind::Ref,
                    vec![Operand::Register(Register::new(Kind::Ref, 0))],
                )),
            ],
            Register::new(Kind::Ref, 1),
        ));

        let jitcode = assemble(
            &mut ssarepr,
            JitCodeBuilder::default(),
            Some(NumRegs {
                ref_: 2,
                ..NumRegs::default()
            }),
        );

        assert_eq!(
            jitcode.code,
            vec![majit_jitcode::insns::BC_INLINE_CALL_R_R, 0, 0, 1, 0, 1,]
        );
        match &jitcode.exec.descrs[0] {
            majit_metainterp::jitcode::RuntimeBhDescr::JitCode(stored) => {
                assert!(Arc::ptr_eq(stored, &target));
            }
            other => panic!("inline_call descr slot is not a JitCode: {other:?}"),
        }
        assert_eq!(
            jitcode
                .resulttypes
                .as_ref()
                .and_then(|resulttypes| resulttypes.get(&jitcode.code.len())),
            Some(&'r')
        );
    }

    #[test]
    fn assemble_inline_call_ir_r_accepts_constint_in_ilist() {
        // `jtransform.py handle_regular_call` + `assembler.py write_insn`
        // ListOfKind: a Constant in the I-list is `emit_const`, encoded as
        // `count_regs['int'] + pool_idx`.  BINARY/COMPARE flatten emit
        // `ListI([ConstInt(tag)])`; the frozen-regalloc path must not
        // treat that tag as a real int register (`touch_reg`).
        let target = Arc::new(JitCodeBuilder::default().finish());
        let mut ssarepr = SSARepr::new("inline_call_ir_r");
        ssarepr.insns.push(Insn::op_with_result(
            "inline_call_ir_r",
            vec![
                Operand::descr(DescrOperand::JitCode(Arc::clone(&target))),
                Operand::ListOfKind(ListOfKind::new(Kind::Int, vec![Operand::ConstInt(2)])),
                Operand::ListOfKind(ListOfKind::new(
                    Kind::Ref,
                    vec![
                        Operand::Register(Register::new(Kind::Ref, 0)),
                        Operand::Register(Register::new(Kind::Ref, 1)),
                    ],
                )),
            ],
            Register::new(Kind::Ref, 2),
        ));

        let jitcode = assemble(
            &mut ssarepr,
            JitCodeBuilder::default(),
            Some(NumRegs {
                int: 1,
                ref_: 3,
                ..NumRegs::default()
            }),
        );

        // I-list item is the first int-pool slot: num_regs_i(1) + 0.
        assert_eq!(
            jitcode.code,
            vec![
                majit_jitcode::insns::BC_INLINE_CALL_IR_R,
                0,
                0,
                1,
                1,
                2,
                0,
                1,
                2,
            ]
        );
        assert_eq!(jitcode.constants_i.as_slice(), &[2]);
    }

    #[test]
    fn residual_call_r_v_preserves_word_returning_and_genuine_void_abis() {
        let mut word_builder = JitCodeBuilder::default();
        let word_fn_idx = word_builder.add_fn_ptr(0x7777usize as *const ());
        let word_insn = super::super::flatten::build_store_subscr_fn_residual_call_r_v_insn(
            word_fn_idx,
            0,
            1,
            2,
        );
        let mut word_ssarepr = SSARepr::new("residual_call_r_v_word_abi");
        word_ssarepr.insns.push(word_insn);
        let word_jitcode = assemble(
            &mut word_ssarepr,
            word_builder,
            Some(NumRegs {
                ref_: 3,
                ..NumRegs::default()
            }),
        );
        let word_bh = word_jitcode
            .exec
            .descrs
            .iter()
            .filter_map(|descr| descr.as_bh_descr())
            .find_map(|descr| match descr {
                majit_jitcode::jitcode::BhDescr::Call { calldescr } => Some(calldescr),
                _ => None,
            })
            .expect("word-returning residual_call_r_v must emit a BhCallDescr");
        let word_descr = pyre_jit_trace::descr::make_call_descr_from_bh(word_bh);
        let word_call = word_descr
            .as_call_descr()
            .expect("word-returning residual_call_r_v must yield a call descr");
        assert_eq!(word_call.result_type(), majit_ir::Type::Void);
        assert_eq!(word_call.result_size(), 8);

        let mut void_builder = JitCodeBuilder::default();
        let void_fn_idx = void_builder.add_fn_ptr(0x8888usize as *const ());
        let void_insn =
            super::super::flatten::build_set_current_exception_fn_residual_call_r_v_insn(
                void_fn_idx,
                0,
            );
        let mut void_ssarepr = SSARepr::new("residual_call_r_v_genuine_void");
        void_ssarepr.insns.push(void_insn);
        let void_jitcode = assemble(
            &mut void_ssarepr,
            void_builder,
            Some(NumRegs {
                ref_: 1,
                ..NumRegs::default()
            }),
        );
        let void_bh = void_jitcode
            .exec
            .descrs
            .iter()
            .filter_map(|descr| descr.as_bh_descr())
            .find_map(|descr| match descr {
                majit_jitcode::jitcode::BhDescr::Call { calldescr } => Some(calldescr),
                _ => None,
            })
            .expect("set_current_exception residual_call_r_v must emit a BhCallDescr");
        let void_descr = pyre_jit_trace::descr::make_call_descr_from_bh(void_bh);
        let void_call = void_descr
            .as_call_descr()
            .expect("set_current_exception residual_call_r_v must yield a call descr");
        assert_eq!(void_call.result_type(), majit_ir::Type::Void);
        assert_eq!(void_call.result_size(), 0);
    }

    /// `assembler.py:197-206` parity: a `Descr` operand attached to an op
    /// that does not consume a descr in its emitted bytecode MUST NOT
    /// leak into any descr pool. Registration happens inline at 'd'
    /// argcode emission; pyre's runtime codewriter has no
    /// descr-consuming ops today so every `assemble()` is a no-op for
    /// descr registration. This test keeps the guarantee explicit so a
    /// future descr-consumer addition is forced to ship the
    /// corresponding registration onto `BlackholeInterpBuilder.descrs`
    /// (`blackhole.py setup_descrs`), not per-jitcode.
    #[test]
    fn assemble_does_not_register_unconsumed_descr() {
        let mut switch = SwitchDictDescr::new();
        switch.labels.push((4, TLabel::new("L2")));

        let mut ssarepr = SSARepr::new("switch");
        ssarepr.insns.push(Insn::Label(Label::new("L2")));
        // `abort_permanent` does not consume a descr in its bytecode (it
        // emits a single BC_ABORT_PERMANENT byte), so the attached
        // Descr operand is ignored by the assembler.
        ssarepr.insns.push(Insn::op(
            "abort_permanent",
            vec![Operand::descr(DescrOperand::SwitchDict(switch))],
        ));

        // Assemble succeeds without panic — the assertion is that no
        // code path tries to allocate a per-jitcode descr slot.
        let _jitcode = assemble(&mut ssarepr, JitCodeBuilder::default(), None);
    }

    // `switch` is now a descr-consuming op (`dispatch_op` `"switch"`);
    // `portal_we_are_jitted_is_shared_residual_block` checks the
    // `switch/id` round-trip.  `getfield_gc_d` is still unported.

    /// `assembler.py:208-209` parity:
    /// ```python
    /// elif isinstance(x, IndirectCallTargets):
    ///     self.indirectcalltargets.update(x.lst)
    /// ```
    /// An `Operand::IndirectCallTargets(...)` attached to any op must be
    /// folded into `self.indirectcalltargets` with identity-dedup (RPython
    /// uses a Python `set` which dedupes by object identity, not value).
    fn make_test_jitcode(fnaddr: usize) -> Arc<JitCode> {
        let mut jc = JitCodeBuilder::default().finish();
        jc.fnaddr = fnaddr as i64;
        Arc::new(jc)
    }

    #[test]
    fn assemble_collects_indirectcalltargets_from_ops() {
        use super::super::flatten::IndirectCallTargets;

        let jc_a = make_test_jitcode(0x1000);
        let jc_b = make_test_jitcode(0x2000);

        let mut ssarepr = SSARepr::new("indirect");
        // `abort_permanent` is a zero-descr op in pyre's dispatch table, so
        // it tolerates the attached `IndirectCallTargets` operand without
        // needing a live dispatch arm.  The parity-relevant behaviour —
        // `write_insn` scanning the arg list and updating
        // `self.indirectcalltargets` — does not depend on the op kind.
        ssarepr.insns.push(Insn::op(
            "abort_permanent",
            vec![Operand::IndirectCallTargets(IndirectCallTargets {
                lst: vec![jc_a.clone(), jc_b.clone(), jc_a.clone()],
            })],
        ));

        let mut assembler = Assembler::new();
        assembler.assemble(&mut ssarepr, JitCodeBuilder::default(), None);

        // Identity-dedup: jc_a appears twice in the list, but the set keeps
        // a single entry (Python `set.update` on JitCode identity).
        assert_eq!(assembler.indirectcalltargets.len(), 2);
        let vec = assembler.indirectcalltargets_vec();
        assert_eq!(vec.len(), 2);
        assert!(vec.iter().any(|a| Arc::ptr_eq(a, &jc_a)));
        assert!(vec.iter().any(|a| Arc::ptr_eq(a, &jc_b)));
    }

    #[test]
    fn assemble_accumulates_indirectcalltargets_across_ops() {
        use super::super::flatten::IndirectCallTargets;

        let jc_a = make_test_jitcode(0x1000);
        let jc_b = make_test_jitcode(0x2000);
        let jc_c = make_test_jitcode(0x3000);

        let mut assembler = Assembler::new();

        // First assemble() registers jc_a and jc_b.
        let mut r1 = SSARepr::new("first");
        r1.insns.push(Insn::op(
            "abort_permanent",
            vec![Operand::IndirectCallTargets(IndirectCallTargets {
                lst: vec![jc_a.clone(), jc_b.clone()],
            })],
        ));
        assembler.assemble(&mut r1, JitCodeBuilder::default(), None);

        // Second assemble() registers jc_b (already present) and jc_c.
        let mut r2 = SSARepr::new("second");
        r2.insns.push(Insn::op(
            "abort_permanent",
            vec![Operand::IndirectCallTargets(IndirectCallTargets {
                lst: vec![jc_b.clone(), jc_c.clone()],
            })],
        ));
        assembler.assemble(&mut r2, JitCodeBuilder::default(), None);

        // `assembler.py:24` persists `indirectcalltargets` across every
        // `assemble()` call — the CodeWriter holds a single Assembler and
        // the set accumulates for the whole build (codewriter.py).
        assert_eq!(assembler.indirectcalltargets.len(), 3);
    }
}
