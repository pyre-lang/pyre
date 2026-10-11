//! JitState implementation for pyre.
//!
//! `PyreJitState` bridges the interpreter's `PyFrame` with majit's JIT
//! infrastructure. It extracts live values from the frame, restores them
//! after compiled code runs, and provides the meta/sym types for tracing.

use crate::pypyjit_driver_layout;
use majit_backend::Backend;
use majit_ir::{DescrRef, OpCode, OpRef, Type, Value};
use majit_metainterp::virtualizable::VirtualizableInfo;
use majit_metainterp::{
    BridgeInlineCarrier, BridgeVirtualCache, JitDriverStaticData, JitState, PendingRefArrayWrite,
    ReconstructRecipe, TraceAction, TraceCtx, decode_fieldnum, materialize_bridge_virtual,
};
use pyre_interpreter::{locals_w, locals_w_mut};

use pyre_interpreter::bytecode::{CodeObject, ComparisonOperator, Instruction};
use pyre_interpreter::pyframe::PyFrame;
use pyre_object::PyObjectRef;
use pyre_object::boolobject::w_bool_get_value;
use pyre_object::pyobject::{
    FLOAT_TYPE, INT_TYPE, get_instantiate, is_bool, is_dict, is_float, is_int, py_type_check,
};
use pyre_object::{PY_NULL, w_float_get_value, w_int_get_value, w_int_new};

/// jitcode.py / codewriter.py:68: JitCode — compiled bytecode unit.
///
/// MetaInterpStaticData-side wrapper around the shared
/// [`crate::PyJitCode`] payload. The payload `Arc` is the same heap
/// allocation that `pyre_jit::jit::call::CallControl.jitcodes` holds
/// for the same CodeObject — RPython's `MetaInterpStaticData.jitcodes`
/// list and `CallControl.jitcodes` dict reference identical
/// `JitCode` Python objects through Python's refcount semantics, and
/// pyre mirrors that with a shared `Arc`. The wrapper keeps only
/// `index` (the SD-local `jitcode.index = len(all_jitcodes)` from
/// codewriter.py:68) on the SD side; `code_for_jitcode_index` recovers
/// the wrapping `PyObjectRef` from the payload's `code_ptr` through the
/// live-wrapper registry, so the wrapper no longer stores a `code`
/// field.
// SAFETY: JitCode is only written once (during creation) and then
// read-only. The code pointer is stable for the program lifetime.
unsafe impl Sync for JitCode {}

pub(crate) struct JitCode {
    /// codewriter.py:68: jitcode.index = len(all_jitcodes).
    pub index: i32,
    /// Shared `PyJitCode` payload. Same `Arc` instance also lives in
    /// `CallControl.jitcodes`. A rebuild (a later primary compile
    /// re-queuing an inlined callee graph) replaces this `Arc` so cached
    /// `*const JitCode` pointers see the refreshed payload on the next
    /// field access.
    pub payload: std::sync::Arc<crate::PyJitCode>,
}

impl JitCode {
    /// Extract raw CodeObject from this JitCode's payload.
    #[inline]
    /// # Safety
    /// The caller must uphold every validity, runtime-type, aliasing, and lifetime
    /// invariant required by the object and pointer arguments for the entire call.
    pub unsafe fn raw_code(&self) -> *const CodeObject {
        self.payload.code_ptr
    }
}

/// warmspot.py:148-282: MetaInterpStaticData — per-driver compile-time data.
///
/// RPython: created by WarmRunnerDesc, holds jitcodes list populated
/// by codewriter.make_jitcodes(). Accessed as MetaInterp.staticdata.
///
/// pyre: per-thread equivalent (no-GIL runtime). Populated from the
/// authoritative `CodeWriter.make_jitcodes()` result before tracing.
///
/// TODO: the RPython-orthodox
/// `MetaInterpStaticData` lives in
/// `majit_metainterp::pyjitpl::MetaInterpStaticData`.  pyre embeds it
/// as the `canonical` field below and delegates every RPython method
/// (`setup_indirectcalltargets`, `bytecode_for_address`, …) through
/// it, so there is exactly one port of each of those methods.  The
/// pyre-local fields (`jitcodes`, `finish_setup_done`, `op_*`,
/// `liveness_info`) stay here because the surrounding runtime still
/// stores Python-code metadata outside the canonical staticdata.
struct MetaInterpStaticData {
    /// warmspot.py:282: self.metainterp_sd.jitcodes = jitcodes.
    /// Box<JitCode> for address stability across vec growth.
    jitcodes: Vec<Box<JitCode>>,
    /// `incminimark.py old_objects_pointing_to_young` for the off-GC
    /// `JitCode.constants_r` arrays above. RPython's `JitCode` is a GC
    /// object, so publishing its freshly-filled `constants_r` field takes the
    /// normal write barrier and puts that one object on the remembered set.
    /// Pyre's body lives in an `Arc`, outside the managed heap; record the
    /// corresponding jitcode index at the same publication boundary. The next
    /// minor walk consumes the entries after forwarding their nursery
    /// referents, while a major walk still visits every jitcode.
    ///
    /// `RefCell` supplies the same interior mutability as the collector-owned
    /// AddressStack upstream: a stop-the-world root walk reaches this
    /// thread-local state through a shared opaque area pointer. Publication
    /// and collection cannot overlap on the owning mutator.
    jitcodes_with_young_constants: RefCell<Vec<usize>>,
    /// pyjitpl.py `self.liveness_info = "".join(asm.all_liveness)`.
    /// The assembler keeps appending, so each publish is [`SharedLiveness::set`]
    /// on this lock. The tracing `MetaInterp` adopts the same lock
    /// (`liveness_handle`) before it clones `staticdata`; capture then
    /// indexes the bytes the jitcode's `-live-` offset names.
    liveness_info: majit_metainterp::SharedLiveness,
    /// pyjitpl.py `finish_setup` is per MetaInterpStaticData instance.
    /// `METAINTERP_SD` is thread-local in pyre, so this guard must live on
    /// the thread-local object rather than in a process-global `Once`.
    ///
    /// Unlike RPython, pyre assembles lazily and the writer-side
    /// `Assembler.insns` table grows over time. The guard therefore marks
    /// "initial finish_setup done" only; later snapshots still refresh the
    /// cached opcode ids and liveness bytes in place.
    finish_setup_done: bool,

    /// `Assembler.insns` length the cached `op_*` ids above were read off.
    /// Paired with `liveness_info.len()` it decides whether a refresh would
    /// observe anything new — both writer-side buffers are append-only
    /// (`assembler.py:29-31`), so equal lengths mean equal contents.
    insns_len: usize,

    // pyjitpl.py opcode number cache filled by `setup_insns`.
    // RPython stores every field even when the runtime currently does
    // not read them, so the structural parity is preserved. Sentinel
    // `u8::MAX` matches `insns.get('…', -1)` for lookups that happen
    // before `setup_insns` runs (e.g. early fallback paths).
    /// pyjitpl.py `self.op_live = insns.get('live/', -1)`.
    op_live: u8,
    /// pyjitpl.py `self.op_goto = insns.get('goto/L', -1)`.
    op_goto: u8,
    /// pyjitpl.py `self.op_catch_exception = insns.get('catch_exception/L', -1)`.
    op_catch_exception: u8,
    /// pyjitpl.py `self.op_rvmprof_code = insns.get('rvmprof_code/ii', -1)`.
    op_rvmprof_code: u8,
    /// pyjitpl.py `self.op_int_return = insns.get('int_return/i', -1)`.
    op_int_return: u8,
    /// pyjitpl.py `self.op_ref_return = insns.get('ref_return/r', -1)`.
    op_ref_return: u8,
    /// pyjitpl.py `self.op_float_return = insns.get('float_return/f', -1)`.
    op_float_return: u8,
    /// pyjitpl.py `self.op_void_return = insns.get('void_return/', -1)`.
    op_void_return: u8,

    /// pyjitpl.py `class MetaInterpStaticData` — the canonical,
    /// RPython-orthodox port.  Owns `indirectcalltargets`,
    /// `globaldata.indirectcall_dict`, and every RPython staticdata
    /// method (`setup_indirectcalltargets`, `bytecode_for_address`,
    /// `setup_descrs`, `setup_list_of_addr2name`, …).  Reached from
    /// pyre's module-level wrappers (`state::setup_indirectcalltargets`,
    /// `state::bytecode_for_address`) and from future callers that
    /// hold a `&mut MetaInterpStaticData` directly.
    canonical: majit_metainterp::MetaInterpStaticData,
    /// Translated-mode `pyjitpl.py bytecode_for_address indirectcall_dict` values for
    /// frozen source-translation targets. The artifact-level dictionary maps
    /// fnaddr to dense index without decoding bodies; this owner memoizes only
    /// bodies actually requested, preserving upstream's stable JitCode object
    /// identity within this interpreter/JIT state.
    ///
    /// This memo is load-bearing rather than a convenience cache:
    /// `jitcode_runtime::indirectcalltarget_by_index` rebuilds a fresh
    /// `Arc<JitCode>` from the canonical body on every call, so without a memo
    /// two lookups of one fnaddr would hand out two distinct objects and
    /// `indirectcall_dict`'s identity contract would be lost.
    ///
    /// It must also stay on the thread-local state and must not be hoisted to a
    /// process-wide registry. The whole jitcode table it derives from is
    /// deliberately per-thread — `load_jitcode_cells` leaks a fresh cell slice
    /// per thread, on the stated ground that the runtime is single-threaded by
    /// construction and a per-thread cache is what matches the module-level
    /// dict semantics. A shared map over per-thread arenas would hand one
    /// thread a body minted from another thread's family.
    frozen_indirectcall_dict:
        indexmap::IndexMap<usize, std::sync::Arc<majit_metainterp::jitcode::JitCode>>,
}

#[allow(dead_code)]
impl MetaInterpStaticData {
    fn new() -> Self {
        Self {
            jitcodes: Vec::new(),
            jitcodes_with_young_constants: RefCell::new(Vec::new()),
            liveness_info: majit_metainterp::SharedLiveness::default(),
            finish_setup_done: false,
            insns_len: 0,
            op_live: u8::MAX,
            op_goto: u8::MAX,
            op_catch_exception: u8::MAX,
            op_rvmprof_code: u8::MAX,
            op_int_return: u8::MAX,
            op_ref_return: u8::MAX,
            op_float_return: u8::MAX,
            op_void_return: u8::MAX,
            canonical: majit_metainterp::MetaInterpStaticData::new(),
            frozen_indirectcall_dict: indexmap::IndexMap::new(),
        }
    }

    /// pyjitpl.py `setup_indirectcalltargets(indirectcalltargets)`.
    /// Thin delegate to the canonical port; exists solely so callers
    /// that hold `&mut state::MetaInterpStaticData` can reach the
    /// method without constructing a reference to the canonical
    /// field themselves.
    fn setup_indirectcalltargets(
        &mut self,
        targets: Vec<std::sync::Arc<majit_metainterp::jitcode::JitCode>>,
    ) {
        self.canonical.setup_indirectcalltargets(targets);
    }

    fn frozen_bytecode_for_address(
        &mut self,
        fnaddress: usize,
    ) -> Option<std::sync::Arc<majit_metainterp::jitcode::JitCode>> {
        if let Some(jitcode) = self.frozen_indirectcall_dict.get(&fnaddress) {
            return Some(std::sync::Arc::clone(jitcode));
        }
        let index = crate::jitcode_runtime::indirectcalltarget_index_for_address(fnaddress)?;
        let jitcode = crate::jitcode_runtime::indirectcalltarget_by_index(index)?;
        self.frozen_indirectcall_dict
            .insert(fnaddress, std::sync::Arc::clone(&jitcode));
        Some(jitcode)
    }

    /// pyjitpl.py `bytecode_for_address(fnaddress)`.
    /// Thin delegate to the canonical port.
    fn bytecode_for_address(
        &mut self,
        fnaddress: usize,
    ) -> Option<std::sync::Arc<majit_metainterp::jitcode::JitCode>> {
        self.canonical.bytecode_for_address(fnaddress)
    }
}

impl MetaInterpStaticData {
    /// pyjitpl.py `MetaInterpStaticData.setup_insns(self, insns)`:
    /// copy opcode numbers for the well-known bytecodes out of the
    /// assembler's `insns` dict in the same order as upstream.
    fn setup_insns(&mut self, insns: &indexmap::IndexMap<String, u8>) {
        self.op_live = insns.get("live/").copied().unwrap_or(u8::MAX);
        self.op_goto = insns.get("goto/L").copied().unwrap_or(u8::MAX);
        self.op_catch_exception = insns.get("catch_exception/L").copied().unwrap_or(u8::MAX);
        self.op_rvmprof_code = insns.get("rvmprof_code/ii").copied().unwrap_or(u8::MAX);
        self.op_int_return = insns.get("int_return/i").copied().unwrap_or(u8::MAX);
        self.op_ref_return = insns.get("ref_return/r").copied().unwrap_or(u8::MAX);
        self.op_float_return = insns.get("float_return/f").copied().unwrap_or(u8::MAX);
        self.op_void_return = insns.get("void_return/").copied().unwrap_or(u8::MAX);
        self.insns_len = insns.len();
    }

    /// Whether a `finish_setup_if_needed` call with these writer-side buffer
    /// lengths would republish anything. `ensure_finish_setup` runs on every
    /// `jitcode_for`, and building its arguments copies both buffers whole;
    /// `assembler.py:29-31` seeds them from the build-time tables, so that
    /// copy is proportional to the whole opcode and liveness universe rather
    /// than to what the call would change.
    fn finish_setup_would_republish(&self, insns_len: usize, all_liveness_len: usize) -> bool {
        !self.finish_setup_done
            || insns_len != self.insns_len
            || all_liveness_len != self.liveness_info.len()
    }

    /// pyjitpl.py `finish_setup`: wire the assembler's opcode table
    /// into this staticdata object and snapshot the current `all_liveness`.
    fn finish_setup_if_needed(
        &mut self,
        insns: &indexmap::IndexMap<String, u8>,
        all_liveness: Vec<u8>,
    ) {
        let was_done = self.finish_setup_done;
        self.setup_insns(insns);
        self.liveness_info.set(all_liveness);
        self.finish_setup_done = true;
        // pyjitpl.py `finish_setup_descrs`: PyPy invokes this
        // immediately after `finish_setup(codewriter)` from
        // `warmspot.py:289`. Pyre's `ensure_finish_setup` is the lazy
        // first-trace gate, so the descr bitstring compaction runs
        // exactly once on the very first JIT entry — the
        // `finish_setup_done` flag above this block keeps subsequent
        // calls idempotent. Bitstring writes land directly on each
        // descr's `ei_index` slot via `effectinfo::compute_bitstrings`
        // (`effectinfo.py descr.ei_index = …`); no process-global
        // side table.
        //
        // This staticdata is not the one the tracing `MetaInterp` owns —
        // pyre carries two — so the `all_descrs` list this publishes has to
        // be the process-wide one `descr_index` is stamped against
        // (`MetaInterpStaticData::all_descrs`), or `bridgeopt.py:155
        // metainterp_sd.all_descrs[descr_index]` indexes an empty list.
        if !was_done {
            self.canonical.finish_setup_descrs_frozen();
        }
    }

    /// pyjitpl.py `self.liveness_info = "".join(asm.all_liveness)` —
    /// refreshes the staticdata mirror after each writer-side append.
    pub(crate) fn set_liveness_info(&mut self, bytes: Vec<u8>) {
        self.liveness_info.set(bytes);
    }
}

/// Crate-local helper so sibling modules can refresh
/// `MetaInterpStaticData.liveness_info` without leaking the private
/// type through public API signatures.
pub(crate) fn publish_liveness_info(bytes: Vec<u8>) {
    METAINTERP_SD.with(|r| r.borrow_mut().set_liveness_info(bytes));
}

impl MetaInterpStaticData {
    #[inline]
    fn canonical_code_key_opt(code: *const ()) -> Option<usize> {
        if code.is_null() {
            return None;
        }
        // A raw `CodeObject*` IS the canonical key (jitcodes compare
        // `raw_code()` identity).  Post-#16 callers pass the raw code
        // identity directly (`ReconstructRecipe.code_ptr`); recognize it via
        // the live-wrapper registry — keyed by raw code identity — before
        // dereferencing `code` as a `PyCode` wrapper, which would read a
        // garbage field out of a raw `CodeObject`.
        if !pyre_interpreter::live_code_wrapper(code).is_null() {
            return Some(code as usize);
        }
        let raw = unsafe { pyre_interpreter::w_code_get_ptr(code as pyre_object::PyObjectRef) };
        if raw.is_null() {
            None
        } else {
            Some(raw as usize)
        }
    }

    /// codewriter.py:67-68 parity — stamp the SD-local `idx` onto the
    /// shared `Arc<majit_metainterp::jitcode::JitCode>` carried by the
    /// payload.  Upstream RPython unconditionally writes
    /// `jitcode.index = index`; a mismatch between an existing
    /// canonical stamp and the SD-local slot is a parity violation, so
    /// rely on `JitCode::set_index`'s own assert (same value: OK,
    /// different value: panic) instead of silently skipping.
    fn stamp_payload_index(idx: i32, payload: &std::sync::Arc<crate::PyJitCode>) {
        payload.jitcode.set_index(idx as usize);
    }

    /// Write-barrier half of [`Self::jitcodes_with_young_constants`].
    ///
    /// The pool is immutable after publication except for GC forwarding, so
    /// one pending entry is sufficient until the next minor collection. The
    /// linear duplicate check matches incminimark's flag-gated remembered-set
    /// insertion and stays over the small population of newly-published
    /// jitcodes, not the thousands of frozen placeholder slots.
    fn remember_jitcode_constants_write(&self, index: usize) {
        let Some(jitcode) = self.jitcodes.get(index) else {
            return;
        };
        let Some(body) = jitcode.payload.jitcode.try_body() else {
            return;
        };
        if body.constants_r.is_empty() {
            return;
        }
        let mut pending = self.jitcodes_with_young_constants.borrow_mut();
        if !pending.contains(&index) {
            pending.push(index);
        }
    }

    /// warmspot.py:281-282:
    ///
    /// ```python
    /// jitcodes = self.codewriter.make_jitcodes(verbose=verbose)
    /// self.metainterp_sd.jitcodes = jitcodes
    /// ```
    ///
    /// RPython runs `make_jitcodes()` once before tracing, so existing
    /// frame pointers and jitcode indices never move. Pyre still reaches
    /// this boundary from lazy portal-entry setup, so merge new payloads
    /// into the existing trace-side list instead of replacing the list.
    /// This preserves RPython's stable `metainterp_sd.jitcodes` invariant
    /// for already-captured resume data.
    fn set_jitcodes_from_make_result(&mut self, payloads: Vec<std::sync::Arc<crate::PyJitCode>>) {
        self.reserve_build_time_index_space();
        for payload in payloads {
            assert!(
                !payload.code_ptr.is_null(),
                "make_jitcodes returned a JitCode without PyCode identity"
            );
            assert!(
                !payload.is_skeleton(),
                "make_jitcodes returned an unpopulated JitCode skeleton"
            );
            let raw_key = payload.code_ptr as usize;
            let existing_pos = self.installed_jitcode_pos_for_raw_key(raw_key);
            match existing_pos {
                Some(pos) if Self::slot_accepts_payload(&self.jitcodes[pos], &payload) => {
                    let index = self.jitcodes[pos].index;
                    Self::stamp_payload_index(index, &payload);
                    self.jitcodes[pos].payload = payload;
                    self.remember_jitcode_constants_write(pos);
                }
                _ => {
                    let index = self.jitcodes.len() as i32;
                    Self::stamp_payload_index(index, &payload);
                    self.jitcodes.push(Box::new(JitCode { index, payload }));
                    self.remember_jitcode_constants_write(index as usize);
                }
            }
        }
    }

    /// May `slot`'s payload be replaced by `payload` in place (keeping
    /// the slot's jitcode index)?
    ///
    /// RPython's `metainterp_sd.jitcodes[i]` entries are immutable after
    /// `warmspot.py:282`: every `rd_numb` frame records `(jitcode_index,
    /// pc)` and the blackhole decoder re-derives the per-frame value
    /// count from that index's liveness tables, so a populated
    /// entry must never change shape under an index that live resume
    /// data already references. Pyre's runtime codewriter can re-splice
    /// a graph (when a later primary compile re-queues an inlined
    /// callee's graph at the drain
    /// boundary); such a rebuild must take a FRESH index, leaving the
    /// old entry intact for old resume data. Only not-yet-populated skeletons
    /// (with no liveness any rd_numb could reference) are filled in place, matching RPython's
    /// "same JitCode object is filled later" setup-time flow.
    fn slot_accepts_payload(slot: &JitCode, payload: &std::sync::Arc<crate::PyJitCode>) -> bool {
        !slot.payload.is_populated() || std::sync::Arc::ptr_eq(&slot.payload, payload)
    }

    fn installed_jitcode_pos_for_raw_key(&self, raw_key: usize) -> Option<usize> {
        // A re-spliced graph appends a fresh entry (see
        // `slot_accepts_payload`); new frames must bind the newest build,
        // while old indices keep resolving to their original entries.
        self.jitcodes
            .iter()
            .rposition(|jitcode| unsafe { jitcode.raw_code() as usize } == raw_key)
    }

    /// codewriter.py:67-68 / call.py get_jitcode adapter: install or return the  allow-line-citation
    /// trace-side wrapper for `code`. When `supplied` is present it is the
    /// populated PyJitCode Arc from CodeWriter's pending-graph drain; otherwise
    /// this creates the same empty skeleton shape as CallControl.get_jitcode()
    /// before the drain fills it.
    fn jitcode_for(
        &mut self,
        code: *const (),
        supplied: Option<std::sync::Arc<crate::PyJitCode>>,
    ) -> *const JitCode {
        self.reserve_build_time_index_space();
        let raw_key = Self::canonical_code_key_opt(code).unwrap_or(0);
        if let Some(pos) = self.installed_jitcode_pos_for_raw_key(raw_key) {
            match supplied {
                Some(payload) if Self::slot_accepts_payload(&self.jitcodes[pos], &payload) => {
                    let index = self.jitcodes[pos].index;
                    Self::stamp_payload_index(index, &payload);
                    self.jitcodes[pos].payload = payload;
                    self.remember_jitcode_constants_write(pos);
                }
                Some(payload) => {
                    // Re-spliced build of an already-populated entry:
                    // append under a fresh index (see `slot_accepts_payload`).
                    let index = self.jitcodes.len() as i32;
                    Self::stamp_payload_index(index, &payload);
                    self.jitcodes.push(Box::new(JitCode { index, payload }));
                    let pos = self.jitcodes.len() - 1;
                    self.remember_jitcode_constants_write(pos);
                    return &*self.jitcodes[pos] as *const JitCode;
                }
                None => {}
            }
            return &*self.jitcodes[pos] as *const JitCode;
        }

        let payload = supplied.unwrap_or_else(|| {
            let raw_code = if raw_key == 0 {
                std::ptr::null()
            } else {
                raw_key as *const CodeObject
            };
            std::sync::Arc::new(crate::PyJitCode::skeleton(raw_code))
        });
        let index = self.jitcodes.len() as i32;
        Self::stamp_payload_index(index, &payload);
        let jitcode = Box::new(JitCode { index, payload });
        let ptr = &*jitcode as *const JitCode;
        self.jitcodes.push(jitcode);
        self.remember_jitcode_constants_write(index as usize);
        ptr
    }

    /// Keep the leading `jitcodes` slots that a build-time jitcode's baked
    /// `JitCode::index` can name out of the runtime append space.
    ///
    /// The build-time table is dense (`get_jitcode_by_index(i).index == i`), so its
    /// length is exactly that range, and
    /// [`install_build_time_jitcode_at`] writes into it verbatim — a resume
    /// frame records the baked index and resolves it as `sd.jitcodes[index]`.
    /// Runtime PyCode entries append at `len()`, so without the reservation
    /// the first one takes slot 0, the build-time install overwrites it, and
    /// [`Self::compiled_jitcode_lookup`] answers `None` for a code the
    /// codewriter still reports as drained.
    ///
    /// Idempotent: a no-op once the leading slots exist.
    fn reserve_build_time_index_space(&mut self) {
        // Snapshot the build -> runtime address correspondences here, at the
        // instant the whole-table loader used to take them, rather than
        // wherever the first lazy body decode lands.
        crate::runtime_fnaddr_patch::prime_address_correspondences();
        let reserved = crate::jitcode_runtime::jitcode_count();
        // Pyre has two JitCode producers (the frozen source-translation table
        // and the lazy per-PyCode writer), so it must reserve the frozen
        // table's dense index range before appending runtime entries. RPython
        // needs no placeholder objects: its one codewriter has already
        // produced the complete list at `warmspot.py:281-282`. These slots
        // therefore share one inert null skeleton until an actual frozen body
        // is installed, instead of manufacturing thousands of heavyweight
        // skeletons solely to occupy indices.
        let placeholder = std::sync::Arc::new(crate::PyJitCode::skeleton(std::ptr::null()));
        while self.jitcodes.len() < reserved {
            let index = self.jitcodes.len() as i32;
            self.jitcodes.push(Box::new(JitCode {
                index,
                payload: std::sync::Arc::clone(&placeholder),
            }));
        }
    }

    /// Return the installed SD entry for a `PyCode`.
    /// RPython's runtime lookup never compiles and never creates a
    /// skeleton here: every entry must already have arrived through
    /// `make_jitcodes()` and `warmspot.py:282`.
    fn compiled_jitcode_lookup(&self, code: *const ()) -> Option<*const JitCode> {
        let key = Self::canonical_code_key_opt(code)?;
        self.jitcodes
            .iter()
            .rev()
            .find(|jitcode| unsafe { jitcode.raw_code() as usize } == key)
            .filter(|jitcode| !jitcode.payload.is_skeleton())
            .map(|jitcode| &**jitcode as *const JitCode)
    }
}

/// RPython assembler.py `Assembler._encode_liveness` parity:
/// intern one `[live_i, live_r, live_f]` triple in the assembler's
/// `all_liveness` buffer and return its 2-byte offset.
///
/// Writes land on `AssemblerState` (the writer side); `MetaInterpStaticData`
/// receives a fresh snapshot via the final `METAINTERP_SD.liveness_info`
/// assignment — matching `pyjitpl.py`'s
/// `self.liveness_info = "".join(asm.all_liveness)` after each append.
pub fn intern_liveness(live_i: &[u8], live_r: &[u8], live_f: &[u8]) -> Option<u16> {
    use crate::assembler::ASSEMBLER_STATE;
    use majit_jitcode::liveness::encode_liveness;

    ensure_finish_setup();

    let snapshot = ASSEMBLER_STATE.with(|r| -> Option<(u16, Vec<u8>)> {
        let mut asm = r.borrow_mut();
        // assembler.py `self.num_liveness_ops += 1` — counted once per
        // `-live-` instruction in `write_insn`, before the dedup lookup
        // inside `_encode_liveness`. The counter measures write-insn call
        // frequency, not unique-entry count.
        asm.num_liveness_ops += 1;
        // `assembler.py:238` keys on the three frozensets. `encode_liveness`
        // sorts and dedups before writing, so two argument orderings of one
        // set encode to identical bytes; keying on the raw slice order would
        // miss the dedup and append a second copy of a record already in the
        // buffer. Canonicalise to the same sorted/deduped form the writer side
        // (`VecSet`) and `decode_liveness_records` produce.
        let canonical = |live: &[u8]| {
            let mut v = live.to_vec();
            v.sort_unstable();
            v.dedup();
            v
        };
        let key = (canonical(live_i), canonical(live_r), canonical(live_f));
        // The dict is derived from the buffer on first use after each
        // `publish_state`, which installs bytes without one. See
        // `assembler::liveness_positions_of`.
        if asm.all_liveness_positions.is_none() {
            let positions = crate::assembler::liveness_positions_of(&asm.all_liveness);
            asm.all_liveness_positions = Some(positions);
        }
        let positions = asm
            .all_liveness_positions
            .as_ref()
            .expect("derived just above");
        if let Some(&pos) = positions.get(&key) {
            return Some((pos, asm.all_liveness.clone()));
        }
        let pos = asm.all_liveness_length;
        // `assembler.py` counts `len(live_i)` on the set
        // `get_liveness_info` returns (`assembler.py`), so the count byte
        // is the same cardinality `encode_liveness` packs into the bitset.
        // Take both the counts and the bytes off the canonical form for that
        // reason: `LivenessIterator` is driven by the count, so a raw slice
        // length inflated by a repeated index would read more bits than this
        // record holds, run into the next record's bytes, and desync every
        // record after it. The sibling producers get the invariant from their
        // argument type (`_register_liveness_offset` in
        // `codewriter/assembler.rs` counts a `VecSet`); this one
        // takes plain slices, so it establishes the set here.
        let (len_i, len_r, len_f) = (key.0.len(), key.1.len(), key.2.len());
        let encoded_i = encode_liveness(&key.0);
        let encoded_r = encode_liveness(&key.1);
        let encoded_f = encode_liveness(&key.2);
        if len_i > u8::MAX as usize
            || len_r > u8::MAX as usize
            || len_f > u8::MAX as usize
            || pos > u16::MAX as usize
        {
            return None;
        }
        let pos_u16 = pos as u16;
        asm.all_liveness_positions
            .as_mut()
            .expect("derived above")
            .insert(key, pos_u16);
        asm.all_liveness.push(len_i as u8);
        asm.all_liveness.push(len_r as u8);
        asm.all_liveness.push(len_f as u8);
        asm.all_liveness.extend(encoded_i);
        asm.all_liveness.extend(encoded_r);
        asm.all_liveness.extend(encoded_f);
        asm.all_liveness_length = asm.all_liveness.len();
        // The mirror now holds a record no writer published, so the next
        // `publish_state_from` must replace it wholesale.
        asm.synced_with = None;
        Some((pos_u16, asm.all_liveness.clone()))
    })?;

    let (pos, all_liveness) = snapshot;
    METAINTERP_SD.with(|r| {
        r.borrow_mut().liveness_info.set(all_liveness);
    });
    Some(pos)
}

/// `resume.py` `ResumeDataDirectReader` reads `metainterp_sd.liveness_info`;
/// this is a snapshot of those bytes
/// at this moment, for a reader that cannot hold the lock
/// (`ResumeDataDirectReader`).
///
/// The `Arc<[u8]>` is a copy. Holders of [`liveness_handle`] see later
/// `intern_liveness` publishes without taking another snapshot.
pub fn liveness_info_snapshot() -> std::sync::Arc<[u8]> {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| r.borrow().liveness_info.snapshot_arc())
}

/// The lock [`intern_liveness`] publishes into. The tracing `MetaInterp`
/// adopts it once, before `staticdata` is cloned, so capture and the
/// jitcode's `-live-` offsets share one table (`resume.py`
/// `metainterp.staticdata`).
pub fn liveness_handle() -> majit_metainterp::SharedLiveness {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| r.borrow().liveness_info.clone())
}

/// pyjitpl.py:2236 parity: expose the staticdata `live/` opcode for callers
/// that need to decode inline liveness offsets from a JitCode.
pub fn op_live() -> u8 {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| r.borrow().op_live)
}

/// blackhole.py:72-74 cached control opcodes consumed by pyre's
/// blackhole-resume adapters. Returns the same `insns.get(..., -1)`
/// values RPython would have stored on `BlackholeInterpBuilder`.
pub fn blackhole_control_opcodes() -> (i32, i32, i32) {
    use crate::assembler::ASSEMBLER_STATE;

    let needs_refresh = METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        let missing_live = sd.op_live == u8::MAX;
        let missing_catch_exception = sd.op_catch_exception == u8::MAX;
        let missing_rvmprof_code = sd.op_rvmprof_code == u8::MAX;
        if !sd.finish_setup_done {
            return true;
        }
        if !(missing_live || missing_catch_exception || missing_rvmprof_code) {
            return false;
        }
        ASSEMBLER_STATE.with(|a| {
            let asm = a.borrow();
            (missing_live && asm.insns.contains_key("live/"))
                || (missing_catch_exception && asm.insns.contains_key("catch_exception/L"))
                || (missing_rvmprof_code && asm.insns.contains_key("rvmprof_code/ii"))
        })
    });
    if needs_refresh {
        ensure_finish_setup();
    }
    METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        let decode = |opcode: u8| -> i32 { if opcode == u8::MAX { -1 } else { opcode as i32 } };
        (
            decode(sd.op_live),
            decode(sd.op_catch_exception),
            decode(sd.op_rvmprof_code),
        )
    })
}

/// pyjitpl.py setup_indirectcalltargets module-level entry point for
/// `MetaInterpStaticData::setup_indirectcalltargets`.
///
/// RPython sets this at `pyjitpl.py` during
/// `finish_setup(codewriter, optimizer)` by piping
/// `codewriter.assembler.indirectcalltargets` straight through.  pyre's
/// codewriter driver calls this after every `Assembler::assemble`
/// session so the staticdata reflects the assembler's latest target
/// set.  Matching shape: `Vec<Arc<JitCode>>` comes from
/// `Assembler::indirectcalltargets_vec` in `pyre-jit`.
pub fn setup_indirectcalltargets(targets: Vec<std::sync::Arc<majit_metainterp::jitcode::JitCode>>) {
    ensure_finish_setup();
    // Frozen source-translation targets stay in the artifact-backed lazy
    // dictionary used by `bytecode_for_address`; only runtime-writer targets
    // need concrete objects in the canonical staticdata list.
    METAINTERP_SD.with(|r| r.borrow_mut().setup_indirectcalltargets(targets));
}

/// pyjitpl.py bytecode_for_address module-level entry point for
/// `MetaInterpStaticData::bytecode_for_address`.
///
/// RPython calls this from `MIFrame.do_residual_or_indirect_call`
/// (`pyjitpl.py`) to check whether a `funcbox.getaddr()`
/// Const corresponds to a known indirect-call target.  pyre's tracer
/// consumer routes through here.
pub fn bytecode_for_address(
    fnaddress: usize,
) -> Option<std::sync::Arc<majit_metainterp::jitcode::JitCode>> {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| {
        let mut sd = r.borrow_mut();
        sd.frozen_bytecode_for_address(fnaddress)
            .or_else(|| sd.bytecode_for_address(fnaddress))
    })
}

use std::cell::RefCell;

thread_local! {
    /// warmspot.py:282: MetaInterp.staticdata (per-thread for no-GIL).
    pub(crate) static METAINTERP_SD: RefCell<MetaInterpStaticData> =
        RefCell::new(MetaInterpStaticData::new());

    /// Set when a guard's frame reports a resume coordinate the jitcode
    /// resume markers cannot resolve — the cross-frame snapshot gap (an inlined
    /// callee + exception-resume shape whose parent resume pc was never
    /// recorded; #124/#130).  `get_list_of_active_boxes` raises it instead of
    /// panicking; `metainterp::interpret` polls it each step and aborts the
    /// trace.  The trace is discarded before any code is installed, so the
    /// abort is side-effect free and the location interprets rather than the
    /// process crashing.
    static TRACE_ABORT_REQUESTED: std::cell::Cell<bool> = std::cell::Cell::new(false);
}

/// Request that the in-progress trait-tracer trace abort at the next step
/// boundary.  See [`TRACE_ABORT_REQUESTED`].
pub fn request_trace_abort() {
    TRACE_ABORT_REQUESTED.with(|c| c.set(true));
}

/// Read-and-clear the trace-abort request.  `metainterp::interpret` calls
/// this after every step; `trace_bytecode` calls it once at entry to drop a
/// flag a prior aborted trace may have left set.
pub fn take_trace_abort_requested() -> bool {
    TRACE_ABORT_REQUESTED.with(|c| c.replace(false))
}

/// pyjitpl.py `MetaInterpStaticData.finish_setup` parity entry point.
///
/// RPython runs `finish_setup` once per `MetaInterpStaticData` object. Pyre's
/// `METAINTERP_SD` is thread-local, so the setup guard also lives on
/// `MetaInterpStaticData`; only the unrelated callback registration below is
/// process-global.
fn ensure_finish_setup() {
    use crate::assembler::ASSEMBLER_STATE;
    use std::sync::Once;
    static FRAME_VALUE_COUNT_INIT: Once = Once::new();
    FRAME_VALUE_COUNT_INIT.call_once(|| {
        majit_ir::resumedata::set_frame_value_count_fn(frame_value_count_at);
    });
    let snapshot = ASSEMBLER_STATE.with(|a| {
        let asm = a.borrow();
        let republishes = METAINTERP_SD.with(|r| {
            r.borrow()
                .finish_setup_would_republish(asm.insns.len(), asm.all_liveness.len())
        });
        republishes.then(|| (asm.insns.clone(), asm.all_liveness.clone()))
    });
    let Some((insns, all_liveness)) = snapshot else {
        return;
    };
    crate::jitcode_runtime::rehydrate_build_descr_raw_sets();
    METAINTERP_SD.with(|r| {
        r.borrow_mut().finish_setup_if_needed(&insns, all_liveness);
    });
}

/// pyjitpl.py setup: frame.jitcode — resolve the JitCode for the frame's code
/// object through the writer-side CallControl.get_jitcode path.
pub(crate) fn jitcode_for(code: *const ()) -> *const JitCode {
    ensure_finish_setup();
    if let Some(existing) = METAINTERP_SD.with(|r| r.borrow().compiled_jitcode_lookup(code)) {
        return existing;
    }
    if code.is_null() {
        return METAINTERP_SD.with(|r| r.borrow_mut().jitcode_for(code, None));
    }
    let raw_code = unsafe {
        pyre_interpreter::w_code_get_ptr(code as pyre_object::PyObjectRef)
            as *const pyre_interpreter::CodeObject
    };
    if !raw_code.is_null() {
        let callbacks = crate::callbacks::try_get().unwrap_or_else(|| {
            panic!(
                "CallJitCallbacks not initialized while resolving JitCode for PyCode {:p}",
                code
            )
        });
        if !(callbacks.ensure_majit_jitcode)(raw_code, code) {
            return std::ptr::null();
        }
        if let Some(existing) = METAINTERP_SD.with(|r| r.borrow().compiled_jitcode_lookup(code)) {
            return existing;
        }
        return std::ptr::null();
    }
    METAINTERP_SD.with(|r| r.borrow_mut().jitcode_for(code, None))
}

/// Install one CodeWriter-owned PyJitCode payload into trace-side
/// MetaInterpStaticData. Used by the lazy CallControl.get_jitcode drain path
/// to publish the same Arc that CallControl.jitcodes stores.
pub fn install_jitcode_for(
    code: *const (),
    payload: std::sync::Arc<crate::PyJitCode>,
) -> *const () {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| r.borrow_mut().jitcode_for(code, Some(payload)) as *const ())
}

/// Install a build-time (code_ptr-less) portal jitcode into the thread-local
/// `MetaInterpStaticData.jitcodes` at a FIXED absolute `index`, growing the
/// store with skeleton placeholders as needed and overwriting any occupant of
/// `index`.
///
/// Unlike [`install_jitcode_for`] (which appends at `len()` and stamps a fresh
/// slot), this honours the jitcode's baked absolute `JitCode::index` — a
/// build-time interpreter portal (jd1's `unpackiterable_portal`,
/// index 0) carries that index in its serialized `OnceLock`, and a resume frame
/// records it verbatim (`dispatch.rs` `frame.jitcode.try_index()`).  For the
/// resume decoders to resolve `sd.jitcodes[index]` to the portal, it must sit
/// at exactly that slot.
///
/// The two index spaces no longer overlap: the leading
/// `jitcode_count()` slots are reserved for build-time indices
/// (`MetaInterpStaticData::reserve_build_time_index_space`) and runtime
/// PyCode entries append above them, so the write below can only replace a
/// reserved skeleton or an earlier install of the same portal.
///
/// Dropping the install is NOT an available shortcut, measured: without it the
/// jd1 live-enter blackhole resume resolves slot 0 to the user jitcode that
/// happens to own it and decodes garbage —
/// `synth/slots_class_var_conflict` SIGSEGVs in
/// `blackhole_resume_via_rd_numb` on dynasm and cranelift, and passes again
/// under `PYRE_NO_JD1=1` / `PYRE_JD1_NO_ENTER=1`.  The resume path does not
/// reach the build-time store on its own; only the index-space unification
/// removes the collision.
pub fn install_build_time_jitcode_at(index: usize, payload: std::sync::Arc<crate::PyJitCode>) {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| {
        let mut sd = r.borrow_mut();
        sd.reserve_build_time_index_space();
        // Idempotent same-value stamp: the portal core already carries `index`.
        payload.jitcode.set_index(index);
        let slot = Box::new(JitCode {
            index: index as i32,
            payload,
        });
        if index < sd.jitcodes.len() {
            debug_assert!(
                sd.jitcodes[index].payload.is_skeleton()
                    || unsafe { sd.jitcodes[index].raw_code() }.is_null(),
                "build-time slot {index} is occupied by a runtime PyCode entry",
            );
            sd.jitcodes[index] = slot;
        } else {
            sd.jitcodes.push(slot);
        }
        sd.remember_jitcode_constants_write(index);
    });
}

/// Publish one frozen source-translation JitCode into the unified runtime
/// registry on demand. RPython's `warmspot.py` installs the complete
/// `CodeWriter.make_jitcodes()` list, so an `MIFrame` created for a translated
/// helper resolves through the same absolute index as a portal frame. Pyre
/// keeps the frozen table lazy; materialize the selected entry without
/// changing that ownership or numbering shape.
pub fn ensure_build_time_jitcode_at(index: usize) -> Option<std::sync::Arc<crate::PyJitCode>> {
    ensure_finish_setup();
    if let Some(payload) = METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        sd.jitcodes
            .get(index)
            .filter(|entry| !entry.payload.is_skeleton())
            .map(|entry| std::sync::Arc::clone(&entry.payload))
    }) {
        return Some(payload);
    }

    let canonical = crate::jitcode_runtime::get_jitcode_by_index(index)?;
    let runtime = majit_metainterp::JitCode::from_canonical((*canonical).clone());
    let payload = std::sync::Arc::new(crate::PyJitCode::from_core_degenerate(
        std::sync::Arc::new(runtime),
        std::ptr::null(),
        false,
    ));
    install_build_time_jitcode_at(index, std::sync::Arc::clone(&payload));
    Some(payload)
}

/// Append a jitcode that has no Python `CodeObject` and hand back its index.
///
/// `warmspot.py:282` installs `CodeWriter.make_jitcodes()` wholesale, so every
/// graph upstream traces — including the interpreter's own, such as
/// `typeobject.py descr_call` — already has a slot and an absolute index. Pyre
/// builds its table from Python `CodeObject`s plus whatever the build-time BFS
/// from the portal reaches, and that BFS stops short of the type-call spine
/// (`call_function_impl_raw` is reached, `call_function_impl_result` is not),
/// so a graph pyre needs as a resume level has to be added here instead.
///
/// The slot appends past [`reserve_build_time_index_space`], the same place a
/// runtime `PyCode` entry lands, so no baked build-time index is displaced.
/// The payload carries a null `code_ptr`: `raw_code_for_jitcode_index` already
/// reports that as "no raw code" for the jd1 drain portal, and the consumers
/// that key a level by its Python code object read it through that accessor.
pub fn install_codeless_jitcode(payload: std::sync::Arc<crate::PyJitCode>) -> i32 {
    debug_assert!(
        payload.code_ptr.is_null(),
        "install_codeless_jitcode is for a jitcode with no Python CodeObject",
    );
    ensure_finish_setup();
    METAINTERP_SD.with(|r| {
        let mut sd = r.borrow_mut();
        sd.reserve_build_time_index_space();
        let index = sd.jitcodes.len() as i32;
        MetaInterpStaticData::stamp_payload_index(index, &payload);
        sd.jitcodes.push(Box::new(JitCode { index, payload }));
        sd.remember_jitcode_constants_write(index as usize);
        index
    })
}

/// `framework.py root_walker.walk_roots` parity for the persistent
/// `MetaInterpStaticData.jitcodes` list (warmspot.py:282
/// `self.metainterp_sd.jitcodes = jitcodes`).  Each entry's PyCode
/// wrapper is recovered from its `payload.code_ptr` via the live-wrapper
/// registry.
///
/// PyCode wrappers are ordinary movable GC objects. The visitor therefore
/// forwards each `LIVE_CODE_WRAPPERS` value in place on minor collections as
/// well as marking it on majors. Keeping this walk on the same per-mutator
/// area as the jitcode constants preserves the owner: RPython reaches both
/// through `MetaInterpStaticData.jitcodes`.
pub fn walk_jitcode_code_roots(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    METAINTERP_SD.with(|state| {
        walk_jitcode_code_roots_in(&state.borrow(), visitor);
    });
}

fn walk_jitcode_code_roots_in(
    sd: &MetaInterpStaticData,
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    use std::sync::atomic::Ordering::Relaxed;
    let probe = probe14_enabled();
    for (jitcode_index, jc) in sd.jitcodes.iter().enumerate() {
        pyre_interpreter::pycode::walk_live_code_wrapper(
            jc.payload.code_ptr as *const (),
            &mut |wrapper| {
                let old = *wrapper as usize;
                if old == 0 {
                    return;
                }
                if probe {
                    PROBE14_CODE_SLOTS.fetch_add(1, Relaxed);
                    probe14_note_nursery(
                        "CODE WRAPPER",
                        old,
                        jc.payload.jitcode.name(),
                        jitcode_index,
                    );
                }
                let root = unsafe {
                    &mut *(wrapper as *mut pyre_object::PyObjectRef as *mut majit_ir::GcRef)
                };
                visitor(root);
                if probe && root.0 != old {
                    let n = PROBE14_RELOCATED.fetch_add(1, Relaxed) + 1;
                    if n <= 20 {
                        eprintln!(
                            "[probe14] RELOCATION KEPT #{n}: code wrapper 0x{old:x} -> 0x{:x} \
                             (jitcode {jitcode_index}/{})",
                            root.0,
                            sd.jitcodes.len(),
                        );
                    }
                }
            },
        );
    }
}

/// Report whether either JitCode-owned population reaches the nursery. This is
/// expected for newly-created movable code wrappers and runtime constants; the
/// counters verify that the corresponding root walk sees them before reset.
fn probe14_note_nursery(kind: &str, addr: usize, name: &str, jitcode_index: usize) {
    use std::sync::atomic::Ordering::Relaxed;
    if !majit_gc::gc_is_nursery_object(addr) {
        return;
    }
    let n = PROBE14_NURSERY.fetch_add(1, Relaxed) + 1;
    if n <= 20 {
        // Address and header word only.  Reading the object's TYPE here is not
        // sound: measured, on the run that aborts the slot is already invalid
        // at walk time, and `type_name_of` returned a multi-kilobyte garbage
        // string from unmapped bytes.  The header word says the same thing
        // safely — a plausible type id means the corpse is still ahead.
        let header = unsafe { (addr as *const usize).offset(-1).read_unaligned() };
        eprintln!(
            "[probe14] NURSERY ROOT #{n}: {kind} 0x{addr:x} header={header:#x} in jitcode \
             {jitcode_index} {name:?}"
        );
    }
}

/// # Safety
/// `data` must come from [`capture_jitcode_constants_root_area`], and the
/// owning thread must be quiesced.
pub unsafe fn walk_jitcode_code_roots_area(
    data: *const (),
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    let state = unsafe { &*(data as *const RefCell<MetaInterpStaticData>) };
    // A collection may be re-entrant while the owner has a shared borrow on
    // the collecting thread. The interrupted owner is quiesced for the walk;
    // reading through RefCell::as_ptr avoids dropping all jitcode roots merely
    // because that suspended read borrow is live.
    let sd = unsafe { &*state.as_ptr() };
    walk_jitcode_code_roots_in(sd, visitor);
}

/// Install the complete `CodeWriter.make_jitcodes()` result into
/// `MetaInterpStaticData.jitcodes`. Setup-time bulk publish only.
///
/// RPython warmspot.py:281-282 stores the list returned by
/// `codewriter.make_jitcodes()` directly on `metainterp_sd.jitcodes`.
/// This function is the pyre-side analog and is invoked exclusively
/// from `register_portal_jitdriver` after the JitDriver-rooted
/// `make_jitcodes()` drain. Pyre may call that boundary more than once
/// while it still has lazy portal-entry setup, so the implementation
/// merges payloads without moving existing SD entries.
pub fn install_jitcodes(jitcodes: Vec<std::sync::Arc<crate::PyJitCode>>) {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| {
        r.borrow_mut().set_jitcodes_from_make_result(jitcodes);
    });
}

/// Return the SD-local `jitcode.index` for this `PyCode`, ensuring
/// the entry through the same `jitcode_for()` / CallControl path used by
/// trace frame setup.
pub fn ensure_jitcode_index(code: *const ()) -> Option<i32> {
    if code.is_null() {
        return None;
    }
    let jitcode = jitcode_for(code);
    if jitcode.is_null() {
        return None;
    }
    Some(unsafe { (*jitcode).index })
}

/// Return the `JitCode*` for this `PyCode` as an opaque pointer,
/// ensuring the entry through the same path as `ensure_jitcode_index`.
#[doc(hidden)]
pub fn ensure_jitcode_ptr(code: *const ()) -> Option<*const ()> {
    if code.is_null() {
        return None;
    }
    let jitcode = jitcode_for(code);
    (!jitcode.is_null()).then_some(jitcode as *const ())
}

#[doc(hidden)]
pub fn frame_locals_cells_stack_array_ref(ctx: &mut TraceCtx, frame: OpRef) -> OpRef {
    frame_locals_cells_stack_array(ctx, frame)
}

/// One index space. `CodeWriter.make_jitcodes` numbers each body, `warmspot`
/// stores that list on `pyjitpl.py` `MetaInterpStaticData.jitcodes`, and
/// `resume.py` `rebuild_from_resumedata` / `blackhole.py` `resume_in_blackhole`
/// index only that list.
///
/// A build-time index (`index < jitcode_runtime::jitcode_count()`) is a vacant
/// skeleton until this materializes the frozen body into `sd.jitcodes[index]`
/// via [`ensure_build_time_jitcode_at`]. That helper borrows `METAINTERP_SD`
/// itself, so the check drops its borrow before the call. Runtime indices stay
/// `>= jitcode_count()` and take the straight `sd.jitcodes` read; the default
/// non-interpret path only carries those.
///
/// Returns the payload only when the slot is occupied by a real body. A shared
/// null skeleton is a vacant marker and is never stamped.
pub fn jitcode_payload_at(jitcode_index: i32) -> Option<std::sync::Arc<crate::PyJitCode>> {
    if jitcode_index < 0 {
        return None;
    }
    let index = jitcode_index as usize;
    ensure_finish_setup();
    let build_time_len = crate::jitcode_runtime::jitcode_count();
    let needs_reserve = METAINTERP_SD.with(|r| r.borrow().jitcodes.len() < build_time_len);
    if needs_reserve {
        METAINTERP_SD.with(|r| r.borrow_mut().reserve_build_time_index_space());
    }
    if index < build_time_len {
        let vacant = METAINTERP_SD.with(|r| {
            let sd = r.borrow();
            sd.jitcodes
                .get(index)
                .map(|entry| entry.payload.is_skeleton())
                .unwrap_or(true)
        });
        if vacant {
            ensure_build_time_jitcode_at(index);
        }
    }
    METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        sd.jitcodes.get(index).and_then(|entry| {
            if entry.payload.is_skeleton() {
                None
            } else {
                Some(std::sync::Arc::clone(&entry.payload))
            }
        })
    })
}

/// `MetaInterpStaticData.jitcodes[jitcode_index]`: resolve the snapshot index
/// to the corresponding CodeObject pointer.
pub fn code_for_jitcode_index(jitcode_index: i32) -> Option<*const ()> {
    let payload = jitcode_payload_at(jitcode_index)?;
    Some(pyre_interpreter::live_code_wrapper(payload.code_ptr as *const ()) as *const ())
}

/// `MetaInterpStaticData.jitcodes[jitcode_index]` helper: resolve the indexed
/// entry to its canonical raw `CodeObject*`.
///
/// Unlike [`code_for_jitcode_index`], this strips the wrapper round-trip
/// up front for callers that only need the graph identity to re-enter
/// `CallControl.jitcodes`.
pub fn raw_code_for_jitcode_index(jitcode_index: i32) -> Option<*const CodeObject> {
    let payload = jitcode_payload_at(jitcode_index)?;
    // A novable drain portal (the jd1 unpackiterable driver) is a native
    // function with no Python `CodeObject`; its degenerate PyJitCode carries
    // a null `code_ptr`. Report that as "no raw code" so the instruction-
    // decoding consumers (bare-reraise probe, traceback lineno) skip it
    // instead of dereferencing null.
    let raw = payload.code_ptr;
    (!raw.is_null()).then_some(raw)
}

/// Resolve `MetaInterpStaticData.jitcodes[jitcode_index]` to the same
/// PyJitCode payload the trace-side frame used. This keeps blackhole /
/// resume consumers on the single-store path (`resume.py`
/// `rebuild_from_resumedata`) instead of re-looking-up through a second table.
pub fn pyjitcode_for_jitcode_index(jitcode_index: i32) -> Option<std::sync::Arc<crate::PyJitCode>> {
    jitcode_payload_at(jitcode_index)
}

/// `PyFrame.frame_finished_execution = True` for a blackhole level that has
/// left its caller.
///
/// `pyopcode.py RETURN_VALUE` stores the bit in the jitcode, and the blackhole
/// replays that `setfield_gc` before `*_return`, so a normal return arrives
/// here already finished. `pyopcode.py handle_operation_error` performs the
/// same store on a no-handler propagation, which has no jitcode op, so this
/// leave hook is what finishes those frames. A second store on the return
/// path is the same bit. Without the propagation store a frame that outlives
/// the call — reached through `sys._getframe()` or a retained traceback —
/// reads back as still executing and refuses `frame.clear()`.
///
/// A zero pointer is the inlined level whose frame was never materialized, the
/// state upstream reaches by keeping the MIFrame virtual; there is no object to
/// mark and no observer that could ask.
pub fn finish_blackhole_level_frame(frame_ptr: i64) {
    if frame_ptr == 0 {
        return;
    }
    // SAFETY: the caller resolved this from the blackhole level's own
    // `virtualizable_ptr`, which `convert_and_run_from_pyjitpl` fills from the
    // per-level concrete frame list, and the level is being torn down here so
    // no other borrow of it is live.
    unsafe {
        (*(frame_ptr as *mut pyre_interpreter::PyFrame)).set_frame_finished_execution(true);
    }
}

/// `walk_bh_regs` calls this before visiting a blackhole register bank.
///
/// `bhimpl_live` only records the marker pc. Precise liveness is a
/// collection-time `gcmap` (`jitframe` + `gcmap` in
/// `rpython/jit/backend/llsupport`), not a per-instruction clear.
/// `cleanup_registers` (`blackhole.py`) still runs at `release_interp`.
/// Between those points a loop-only colour is never redefined — this
/// codewriter colours one function per jitcode, not one
/// `dispatch_bytecode` graph — so `walk_bh_regs` would keep that object
/// alive until frame exit. The marker's Ref set (`liveness.py`) is the
/// bound: a register missing from it is rewritten before any read.
///
/// No marker, a marker that does not decode, or an empty Ref set roots
/// the whole bank. The clear stops at `num_regs_r()`; the slots above it
/// are the constants window.
///
/// # Safety
/// `ctx` is a live `BlackholeInterpreter` registered with the bank.
/// `regs` is that interpreter's `registers_r` for `len` slots.
pub unsafe fn retain_live_ref_registers(ctx: *const (), regs: *mut i64, len: usize) {
    let bh = unsafe { &*(ctx as *const majit_metainterp::blackhole::BlackholeInterpreter) };
    let Some(marker_pc) = bh.last_live_marker_pc() else {
        return;
    };
    let regs = unsafe { std::slice::from_raw_parts_mut(regs, len) };
    let num_regs_r = bh.jitcode.num_regs_r().min(regs.len());
    if num_regs_r == 0 {
        return;
    }
    if !bh.jitcode.can_decode_live_vars(marker_pc, bh.op_live) {
        return;
    }
    let info = bh.jitcode.get_live_vars_info(marker_pc, bh.op_live);
    METAINTERP_SD.with(|r| {
        let Ok(sd) = r.try_borrow() else {
            return;
        };
        let liveness_bytes = sd.liveness_info.snapshot_arc();
        let all_liveness: &[u8] = liveness_bytes.as_ref();
        if info + 3 > all_liveness.len() {
            return;
        }
        let length_r = all_liveness[info + 1] as usize;
        if length_r > num_regs_r || length_r == 0 {
            return;
        }
        let mut live_r: [u64; 4] = [0; 4];
        majit_jitcode::codewriter::jitcode::enumerate_vars(
            info,
            all_liveness,
            |_| {},
            |index| {
                let index = index as usize;
                if index < 256 {
                    live_r[index / 64] |= 1u64 << (index % 64);
                }
            },
            |_| {},
        );
        for index in 0..num_regs_r.min(256) {
            if live_r[index / 64] & (1u64 << (index % 64)) == 0 {
                regs[index] = 0;
            }
        }
    });
}

/// Whether an exception exit emitted for the Python instruction at `py_pc`
/// re-raises a value that already carries this frame's traceback node, so
/// `record_application_traceback` must be skipped.
///
/// `RAISE_VARARGS 0` and `RERAISE` both use RaiseWithExplicitTraceback.
/// FOR_ITER's exception-match mismatch arm re-raises the value its own
/// `catch_exception` caught — `pyopcode.py` re-raises `e` untouched. It
/// keeps this frame's existing node only when the `space.next` residual
/// attached that node; an inlined `__next__` instead leaves the callee's node
/// at the chain head.
pub fn raise_at_py_pc_keeps_existing_traceback(raw_code: &CodeObject, py_pc: usize) -> bool {
    match unsafe { pyre_interpreter::decode_instruction_at(raw_code, py_pc) } {
        Some((Instruction::RaiseVarargs { .. }, op_arg)) => u32::from(op_arg) == 0,
        Some((Instruction::Reraise { .. }, _)) | Some((Instruction::ForIter { .. }, _)) => true,
        _ => false,
    }
}

/// [`raise_at_py_pc_keeps_existing_traceback`] resolved from a jitcode
/// coordinate.
pub fn jitcode_pc_raise_keeps_existing_traceback(jitcode_index: i32, offset: i32) -> bool {
    let Some(raw_code) = raw_code_for_jitcode_index(jitcode_index) else {
        return false;
    };
    let Some(py_pc) = crate::py_coord::exact_py_pc_for_jitcode_pc_public(jitcode_index, offset)
        .or_else(|| crate::py_coord::containing_py_pc_for_jitcode_pc_public(jitcode_index, offset))
    else {
        return false;
    };
    raise_at_py_pc_keeps_existing_traceback(unsafe { &*raw_code }, py_pc as usize)
}

/// Whether the source function behind `jitcode_index` carries a Python
/// `try`/`except` handler, read from `co_exceptiontable`.
pub fn jitcode_source_has_exception_handler(jitcode_index: i32) -> Option<bool> {
    let raw_code = raw_code_for_jitcode_index(jitcode_index)?;
    Some(!unsafe { &*raw_code }.exceptiontable.is_empty())
}

/// `framework.py` `root_walker.walk_roots` hook for the boxed `Ref`
/// constants embedded in live jitcodes' `constants_r` pools.
///
/// RPython keeps these alive implicitly: a `JitCode` is a GC object and
/// its `constants_ptr` array is traced through the object graph, so a
/// boxed constant reachable only from a jitcode survives collection.
/// pyre's jitcodes live in Rust `Arc` memory (the append-only
/// `MetaInterpStaticData.jitcodes` store, warmspot.py:282), so the
/// boxed-ref constant slots need an explicit walker — the same shape
/// `MetaInterp::walk_rd_consts_refs` uses for resume-data consts.
///
/// `blackhole_from_resumedata` seeds the blackhole register file directly
/// from `jitcode.constants_r` (`init_register_files_from_runtime_jitcode`),
/// so a constant boxed object reachable only from a jitcode and swept by a
/// major collection between trace executions leaves the next guard-failure
/// resume reading a freed pointer.
///
/// The pool's VALUES are immutable after build; their ADDRESSES are not.  A
/// `residual_call` whose ref argument is a constant bakes whatever object the
/// tracer read, and that object is non-moving only when it happens to be an
/// old-generation number or a `malloc_typed`-immortal build-time constant, so
/// this walk writes the visitor's answer back into the slot instead of marking
/// a copy of it. Major collections walk every pool; minor collections consume
/// only [`MetaInterpStaticData::jitcodes_with_young_constants`], the off-GC
/// counterpart of incminimark's `old_objects_pointing_to_young`.
pub fn walk_jitcode_constants_refs(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    METAINTERP_SD.with(|state| {
        walk_jitcode_constants_refs_in(&state.borrow(), visitor);
    });
}

pub fn capture_jitcode_constants_root_area() -> *const () {
    METAINTERP_SD.with(|state| state as *const _ as *const ())
}

/// # Safety
/// `data` must come from [`capture_jitcode_constants_root_area`], and the
/// owning thread must be quiesced.
pub unsafe fn walk_jitcode_constants_refs_area(
    data: *const (),
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    let state = unsafe { &*(data as *const RefCell<MetaInterpStaticData>) };
    // See `walk_jitcode_code_roots_area`: the owning mutator is quiesced even
    // when the interrupted operation retains a RefCell borrow.
    let sd = unsafe { &*state.as_ptr() };
    walk_jitcode_constants_refs_in(sd, visitor);
}

/// Forwarding probe for [`walk_jitcode_constants_refs`] above.
/// `drag_out_root` relocates a root only when it is a nursery object start, so
/// a slot whose value the visitor CHANGES is one where a movable object entered
/// the pool and the walk had to write the new address back.  Before that
/// write-back existed the same count named relocations the walk DISCARDED.
///
/// `WALKS`/`SLOTS` are the arming witnesses: `RELOCATED == 0` means nothing
/// moved only when `SLOTS > 0`.  With `SLOTS == 0` the walker observed
/// nothing and the run says nothing either way.
///
/// Gated on `PYRE_PROBE14=1`; reports on stderr as it happens because the
/// fault this chases is a non-unwinding abort that never reaches shutdown.
pub static PROBE14_WALKS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static PROBE14_SLOTS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static PROBE14_RELOCATED: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
/// Companion witnesses for the OTHER half of this area, the `code_ptr ->
/// wrapper` roots, and for the invariant a relocation count cannot test.
///
/// A relocation counter alone cannot refute that the remembered-set boundary
/// missed a movable object: a missed pool is never offered to the visitor.
/// Membership of the nursery in every pool that is offered can, so the probe
/// records that independently.
pub static PROBE14_CODE_SLOTS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
pub static PROBE14_NURSERY: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn probe14_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("PYRE_PROBE14").as_deref() == Ok("1"))
}

fn walk_jitcode_constant_pool(
    sd: &MetaInterpStaticData,
    jitcode_index: usize,
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
    probe: bool,
) {
    use std::sync::atomic::Ordering::Relaxed;
    let Some(jc) = sd.jitcodes.get(jitcode_index) else {
        return;
    };
    let pool_len = jc.payload.jitcode.constants_r.len();
    // The pool cell, not a copy of it. Upstream reaches a run-time write into
    // a prebuilt structure through the write barrier —
    // `_remember_young_pointer_inlined` puts the written-to object on
    // `old_objects_pointing_to_young`, and the remembered set is scanned at
    // the next minor, so the entry is kept and forwarded.
    //
    // The store goes through `ConstSlotR::set`, whose receiver is `&self`.
    // By the time the walk runs the body is published behind an `Arc` and
    // there is no `&mut` route to it, so a `*mut` cast out of
    // `Vec::as_ptr()` would be a write through shared-reference provenance —
    // undefined, and free to be compiled as if the slot still held the
    // pre-write value, which is the stale GC reference this walker exists to
    // prevent. Interior mutability makes the store well-defined instead of
    // relying on it happening to survive optimisation.
    //
    // `METAINTERP_SD` is thread-local while `Arc<RuntimeJitCode>` is not, so
    // two threads' areas can present the same slot; the second visit reads an
    // address that is no longer a nursery start and is a no-op. That is also
    // why the slot is an `AtomicI64` and not a `Cell`.
    //
    // The pool also carries words that are not gcrefs at all — patched
    // host-static addresses and pre-patch STR sentinels. `drag_out_root` and
    // `seed_major_root` reject them before any deref, which is what makes
    // handing them the whole pool safe; a fast path that pre-filtered the
    // slots would have to reproduce those gates.
    let pool = jc.payload.jitcode.constants_r.as_slice();
    for pool_slot in 0..pool_len {
        let cell = &pool[pool_slot];
        let mut gcref = majit_ir::GcRef(cell.get() as usize);
        let before = gcref.0;
        if probe {
            probe14_note_nursery("CONSTANT", before, jc.payload.jitcode.name(), jitcode_index);
        }
        visitor(&mut gcref);
        if gcref.0 != before {
            cell.set(gcref.0 as i64);
        }
        if probe {
            PROBE14_SLOTS.fetch_add(1, Relaxed);
            if gcref.0 != before {
                let n = PROBE14_RELOCATED.fetch_add(1, Relaxed) + 1;
                if n <= 20 {
                    eprintln!(
                        "[probe14] RELOCATION APPLIED #{n}: constants_r slot \
                         0x{before:x} -> 0x{:x} (walk {}, cum_slot {}, \
                         jitcode {}/{}, pool_slot {}/{})",
                        gcref.0,
                        PROBE14_WALKS.load(Relaxed),
                        PROBE14_SLOTS.load(Relaxed),
                        jitcode_index,
                        sd.jitcodes.len(),
                        pool_slot,
                        pool_len,
                    );
                }
            }
        }
    }
}

fn walk_jitcode_constants_refs_in(
    sd: &MetaInterpStaticData,
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    use std::sync::atomic::Ordering::Relaxed;
    let probe = probe14_enabled();
    if probe {
        PROBE14_WALKS.fetch_add(1, Relaxed);
    }
    let is_minor = majit_gc::shadow_stack::extra_root_walk_kind()
        == majit_gc::shadow_stack::ExtraRootWalkKind::Minor;
    if is_minor {
        // `incminimark.py _minor_collection` consumes
        // `old_objects_pointing_to_young`: every nursery referent exposed by
        // these pools is promoted by the visitor below, and no later mutator
        // store can create another young edge because a published JitCode body
        // is immutable. A later body publication re-adds its slot through
        // `remember_jitcode_constants_write`.
        let pending = std::mem::take(&mut *sd.jitcodes_with_young_constants.borrow_mut());
        for jitcode_index in pending {
            walk_jitcode_constant_pool(sd, jitcode_index, visitor, probe);
        }
    } else {
        // `incminimark.py collect_nonstack_roots` / `visit_all_objects`:
        // major marking follows every live JitCode field, irrespective of its
        // remembered-set state.
        for jitcode_index in 0..sd.jitcodes.len() {
            walk_jitcode_constant_pool(sd, jitcode_index, visitor, probe);
        }
    }
    // ARMING WITNESS.  The relocation report above prints only when a slot
    // MOVES, so on a run with no relocations it emits nothing at all — and
    // "nothing" is indistinguishable from "this walker never ran" or "the pool
    // was empty".  Those are the REFUTES and NOT-EXERCISED branches of this
    // probe's pre-registration, and they must not share a rendering.  Emit the
    // observed population unconditionally so a silent run is always readable as
    // one or the other.  Printed as it happens, not at shutdown: the fault this
    // chases is a non-unwinding abort that never reaches shutdown.
    if probe {
        let walks = PROBE14_WALKS.load(Relaxed);
        // Window widened after the first run: `walks <= 3 || walks % 200 == 0`
        // sampled ONLY the walks before any jitcode was installed (jitcodes=0,
        // slots=0) and then nothing until walk 200, but the abort lands around
        // walk 5.  The arming witness therefore reported slots=0 -- "NOT
        // EXERCISED" -- for runs whose own relocation reports proved the
        // opposite.  An arming witness whose sampling window misses the armed
        // regime is worse than none: it manufactures the null it exists to rule
        // out.  Cover every early walk, then thin out.
        if walks <= 50 || walks % 50 == 0 {
            eprintln!(
                "[probe14] ARMED walk={} slots_cumulative={} code_slots_cumulative={} \
                 relocated_cumulative={} nursery_cumulative={} jitcodes={}",
                walks,
                PROBE14_SLOTS.load(Relaxed),
                PROBE14_CODE_SLOTS.load(Relaxed),
                PROBE14_RELOCATED.load(Relaxed),
                PROBE14_NURSERY.load(Relaxed),
                sd.jitcodes.len(),
            );
        }
    }
}

/// Resolve by PyCode wrapper through the trace-side
/// MetaInterpStaticData store. Used by blackhole paths that must see
/// CodeWriter-drained entries.
pub fn pyjitcode_for_code(code: *const ()) -> Option<std::sync::Arc<crate::PyJitCode>> {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        let ptr = sd.compiled_jitcode_lookup(code)?;
        Some(std::sync::Arc::clone(unsafe { &(*ptr).payload }))
    })
}

/// Build a `SubJitCodeBody` view over the callee per-fn JitCode for a
/// PyCode, building+installing it on demand (`jitcode_for`) when the
/// lazy per-fn build has not run yet.  Used by full-body-walk call inlining
/// to obtain a sub-walk body for a runtime callable's code.
///
/// The `Arc<PyJitCode>` is held for the program lifetime in the append-only
/// `MetaInterpStaticData.jitcodes` store (warmspot.py:282) and its bytecode /
/// constant pools are immutable after build, so extending those slices to
/// `'static` is sound — the same justification as the per-fn arm-entry borrow
/// extension at `trace.rs`'s `dispatch_perfn_frame` / `trace_opcode.rs`.
/// Returns `None` when the code is null or the on-demand build did not
/// install a payload.
pub(crate) fn sub_jitcode_body_for_code(
    code: *const (),
) -> Option<crate::jitcode_dispatch::SubJitCodeBody> {
    if code.is_null() {
        return None;
    }
    if jitcode_for(code).is_null() {
        return None;
    }
    let pjc = pyjitcode_for_code(code)?;
    sub_jitcode_body_from_payload(&pjc)
}

fn sub_jitcode_body_from_payload(
    pjc: &crate::PyJitCode,
) -> Option<crate::jitcode_dispatch::SubJitCodeBody> {
    let jc = &pjc.jitcode;
    // SAFETY: the payload lives for the program in the append-only jitcodes
    // store (a second Arc keeps the allocation alive after this local clone
    // drops); the pools are immutable post-build.
    Some(unsafe {
        crate::jitcode_dispatch::SubJitCodeBody {
            code: &*(jc.code.as_slice() as *const [u8]),
            num_regs_r: jc.num_regs_r() as usize,
            num_regs_i: jc.num_regs_i() as usize,
            num_regs_f: jc.num_regs_f() as usize,
            constants_i: &*(jc.constants_i.as_slice() as *const [i64]),
            constants_r: &*(jc.constants_r.as_slice()
                as *const [majit_jitcode::codewriter::jitcode::ConstSlotR]),
            constants_f: &*(jc.constants_f.as_slice() as *const [i64]),
        }
    })
}

/// Per-PC `(color, slot)` entries the resume map records at the callee entry
/// PC (`metadata.pcdep_color_slots[0]`).  Used by full-body-walk call inlining
/// to seed positional args at the registers the callee body actually reads its
/// params from — the canonical splice regalloc does not pin local-i inputargs
/// to identity colors, so `r0..nparams` seeding reads the wrong bank slots.
/// At entry the operand stack is empty, so every entry names a local slot;
/// a param dead at entry has no entry (the body never reads it).  `None` when
/// no payload is installed or the jitcode was never colored (empty
/// `pcdep_color_slots`, a portal/skeleton install whose colors are
/// slot-identity).
pub(crate) fn sub_jitcode_entry_param_colors(code: *const ()) -> Option<Vec<(u8, u16, u16)>> {
    if code.is_null() {
        return None;
    }
    let pjc = pyjitcode_for_code(code)?;
    let legacy = pjc.pcdep_for_jitcode_pc(0);
    legacy
}

pub(crate) type SubDescrPool = (
    &'static [DescrRef],
    &'static [majit_metainterp::jitcode::RuntimeBhDescr],
    &'static crate::jitcode_dispatch::SubJitCodeLookup,
);

/// Build (memoized on the payload itself) the per-fn descr pool a callee body
/// needs when inlined by full-body-walk call inlining: its OWN adapted
/// `descr_refs` + raw `RuntimeBhDescr` slice (for `RawDescrPool::PerFn`) +
/// `sub_jitcode_lookup`.  Mirror of the top-level diagnostic walk's per-fn
/// descr-pool construction (`dispatch_perfn_frame` in `trace.rs`): a callee
/// body's `d`/`j` descr operands index its OWN `exec.descrs`, not the
/// caller's pool.
///
/// The pool lives on `PyJitCodePayload.sub_descr_pool`, not in a side table
/// keyed by object identity: RPython has no such table — the active MIFrame's
/// JitCode carries its descriptors directly — and an identity-keyed cache
/// would survive `PyJitCode::replace_with`, which refills the body in place
/// under the same outer allocation (an in-place refill overwrites the inner
/// `exec.descrs` the pool borrows). Carried on the payload, the pool is
/// dropped by the same `replace_with` that invalidates its borrows.
///
/// The returned slices are `'static` because the payload lives for the
/// program in the append-only `MetaInterpStaticData.jitcodes` store
/// (warmspot.py:282) — the same justification as `sub_jitcode_body_for_code`
/// below; the adapted `descr_refs` Vec and the lookup closure are leaked once
/// per distinct callee body (bounded by the program's callable count ×
/// refinements, not per trace attempt). `'static` coerces to the walker's
/// `'static_a`.
pub(crate) fn sub_jitcode_descr_pool_for_code(code: *const ()) -> Option<SubDescrPool> {
    if code.is_null() || jitcode_for(code).is_null() {
        return None;
    }
    let pjc = pyjitcode_for_code(code)?;
    Some(sub_descr_pool_for_payload(&pjc))
}

fn sub_descr_pool_for_payload(pjc: &crate::PyJitCode) -> SubDescrPool {
    use majit_metainterp::jitcode::RuntimeBhDescr;
    *pjc.sub_descr_pool.get_or_init(|| {
        // SAFETY: the borrow is valid for `'static` because the payload is
        // retained for the program lifetime by the append-only
        // `MetaInterpStaticData.jitcodes` store, and `exec.descrs` is
        // immutable post-build. The one mutation path, `replace_with`,
        // replaces this `OnceCell` together with the body, so the pool can
        // never be returned for a body other than the one it was built from.
        let perfn_descrs: &'static [RuntimeBhDescr] =
            unsafe { &*(pjc.jitcode.exec.descrs.as_slice() as *const [RuntimeBhDescr]) };
        let descr_refs: Vec<DescrRef> = perfn_descrs
            .iter()
            .enumerate()
            .map(|(i, d)| match d {
                RuntimeBhDescr::Descr(bh) => crate::descr::make_descr_from_bh(bh),
                RuntimeBhDescr::ResolvedDescr { .. } => d
                    .as_optimizer_descr()
                    .expect("resolved field descriptor")
                    .clone(),
                RuntimeBhDescr::JitCode(_)
                | RuntimeBhDescr::JitCodeBackEdge(_)
                | RuntimeBhDescr::Call(_)
                | RuntimeBhDescr::AssemblerToken(_) => crate::descr::make_jitcode_descr(i),
            })
            .collect();
        let descr_refs: &'static [DescrRef] = Box::leak(descr_refs.into_boxed_slice());
        let lookup: Box<crate::jitcode_dispatch::SubJitCodeLookup> = Box::new(move |idx: usize| {
            perfn_descrs
                .get(idx)
                .and_then(|d| d.as_jitcode())
                .map(|jc| crate::jitcode_dispatch::SubJitCodeBody {
                    code: jc.code.as_slice(),
                    num_regs_r: jc.num_regs_r() as usize,
                    num_regs_i: jc.num_regs_i() as usize,
                    num_regs_f: jc.num_regs_f() as usize,
                    constants_i: jc.constants_i.as_slice(),
                    constants_r: jc.constants_r.as_slice(),
                    constants_f: jc.constants_f.as_slice(),
                })
        });
        let lookup: &'static crate::jitcode_dispatch::SubJitCodeLookup = Box::leak(lookup);
        (descr_refs, perfn_descrs, lookup)
    })
}

/// `resume.py` `consume_one_section` → `enumerate_vars` parity:
/// return the number of tagged values encoded for a frame at
/// (jitcode_index, pc).
///
/// Decode the `-live-` offset from the one `MetaInterpStaticData.jitcodes`
/// entry at `jitcode.get_live_vars_info(pc, op_live)`, then read the
/// three-byte `[len_i][len_r][len_f]` header in `liveness_info`. Total live
/// value count = `len_i + len_r + len_f`. [`jitcode_payload_at`] materializes
/// a build-time index into that list before the decode.
pub fn frame_value_count_at(jitcode_index: i32, pc: i32) -> usize {
    // Materialize before borrowing `op_live` / `liveness_info`. A published
    // coordinate the single list cannot name is a capture bug.
    let Some(payload) = jitcode_payload_at(jitcode_index) else {
        panic!(
            "frame_value_count_at: no jitcode at jitcode_index={jitcode_index} pc={pc} \
             — `MetaInterpStaticData.jitcodes` (`resume.py` `rebuild_from_resumedata`) \
             has no body for a published resume coordinate"
        )
    };
    let (op_live, liveness) = METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        (sd.op_live, sd.liveness_info.snapshot_arc())
    });
    if let Some(count) = decode_live_var_count(&payload.jitcode, pc, op_live, &liveness) {
        return count;
    }
    panic!(
        "frame_value_count_at: no decodable `-live-` for jitcode_index={jitcode_index} pc={pc} \
         (n_py_instrs={}, all_liveness.len={}). A published resume coordinate must decode \
         from `MetaInterpStaticData.jitcodes`.",
        payload.metadata.n_py_instrs,
        liveness.len(),
    );
}

fn decode_live_var_count(
    jitcode: &majit_jitcode::jitcode::JitCode,
    pc: i32,
    op_live: u8,
    all_liveness: &[u8],
) -> Option<usize> {
    if pc < 0 || !jitcode.can_decode_live_vars(pc as usize, op_live) {
        return None;
    }
    let off = jitcode.get_live_vars_info(pc as usize, op_live);
    if off + 2 >= all_liveness.len() {
        return None;
    }
    let length_i = all_liveness[off] as usize;
    let length_r = all_liveness[off + 1] as usize;
    let length_f = all_liveness[off + 2] as usize;
    Some(length_i + length_r + length_f)
}

/// Same decode as [`frame_value_count_at`]. jd1's
/// `JitDriverStaticData::frame_value_count_fn` still names this function.
/// Build-time and runtime indices share `MetaInterpStaticData.jitcodes`
/// (`CodeWriter.make_jitcodes` / `warmspot`); the `-live-` bytes are the
/// prefix of that staticdata's `liveness_info`
/// (`Assembler::resuming_build_time_liveness`).
pub fn build_time_frame_value_count_at(jitcode_index: i32, pc: i32) -> usize {
    frame_value_count_at(jitcode_index, pc)
}

/// Resolve the JitCode byte offset the full-body walk should RESUME at for a
/// bridge whose guard carried `carried_jitcode_pc`. Agrees with where the
/// blackhole resumes (`call_jit.rs` `resolve_jitcode`): a kept-stack branch
/// guard resumes at its OWN mid-opcode jitcode offset (the `goto_if_not`), not
/// the opcode-entry marker — re-executing the whole opcode from entry would
/// read abstract-register colors dead at the guard. The carried direct
/// coordinate IS that offset, so no Python pc is consulted here. Returns `None`
/// when the carried word does not name a decodable startpoint, which declines
/// the bridge at the caller.
pub fn resolve_bridge_walk_entry_at(jitcode_index: i32, carried_jitcode_pc: i32) -> Option<usize> {
    let payload = jitcode_payload_at(jitcode_index)?;
    let op_live = METAINTERP_SD.with(|r| r.borrow().op_live);
    payload.resolve_bridge_walk_entry_pc(carried_jitcode_pc, op_live)
}

/// virtualizable.py `read_boxes` parity: assemble the
/// `virtualizable_boxes` layout the tracing-time vable mirror expects
/// and hand it to `TraceCtx::init_virtualizable_boxes`. Used by both
/// the root portal seed (`initialize_sym`) and the bridge entry rebuild
/// (`setup_bridge_sym`). Matches RPython `virtualizable.py
/// load_list_of_boxes`, producing `[static_fields..., array_items...,
/// vable_box]` (the trailing `vable_box` is appended by
/// `init_virtualizable_boxes`).
///
/// * `scalar_oprefs` — the NUM_VABLE_SCALARS static field OpRefs in
///   declaration order (last_instr, pycode, valuestackdepth, debugdata,
///   w_globals). Excludes both the frame-identity slot and
///   any non-vable extra reds (e.g. `ec`); virtualizable_boxes only
///   carries the vable static fields plus array items.
/// * `array_items` — pre-resolved OpRefs for the heap-side
///   `locals_cells_stack_w` array. Entries past `array_len` are ignored;
///   short lists are padded with a shared const-NULL Ref so the vable
///   mirror covers every interpreter-visible slot (virtualizable.py
///   `assert len(boxes) == i + 1`).
/// * `array_len` — the runtime PyFrame's
///   `locals_cells_stack_w.len()`; `init_virtualizable_boxes` stores this
///   as the sole entry in `virtualizable_array_lengths`.
pub(crate) fn seed_virtualizable_boxes(
    ctx: &mut TraceCtx,
    vable_ref: OpRef,
    vable_ref_value: majit_ir::Value,
    scalar_oprefs: &[OpRef],
    array_items: &[OpRef],
    array_len: usize,
    input_values: &[majit_ir::Value],
    heap_ptr: *const u8,
) {
    let info = crate::frame_layout::build_pyframe_virtualizable_info();
    ctx.install_virtualizable_info(info.clone());
    let expected_scalars = crate::virtualizable_gen::NUM_VABLE_SCALARS;
    assert_eq!(
        scalar_oprefs.len(),
        expected_scalars,
        "seed_virtualizable_boxes: scalar_oprefs.len() must equal NUM_VABLE_SCALARS",
    );
    let mut input_oprefs: Vec<OpRef> = Vec::with_capacity(expected_scalars + array_len);
    input_oprefs.extend_from_slice(scalar_oprefs);
    let taken = array_items.len().min(array_len);
    input_oprefs.extend_from_slice(&array_items[..taken]);
    if taken < array_len {
        let null_ref = ctx.const_ref(pyre_object::PY_NULL as i64);
        for _ in taken..array_len {
            input_oprefs.push(null_ref);
        }
    }
    let array_lengths = vec![array_len];
    // `virtualizable.py load_list_of_boxes`: each box carries its concrete
    // (heap read for portal entry / resume-data stream for bridge entry).
    // Callers pass an empty slice to leave InputArg/`*FrontendOp` boxes
    // unstamped (unit-test / init-before-run path).
    ctx.init_virtualizable_boxes(
        &info,
        vable_ref,
        vable_ref_value,
        &input_oprefs,
        input_values,
        &array_lengths,
    );
    // pyjitpl.py synchronize_virtualizable parity: cache the live heap
    // pointer on TraceCtx so subsequent vable setfield / setarrayitem calls
    // can mirror their shadow updates into the live virtualizable.  Pass
    // a null pointer to disable (unit-test / init-before-run path).
    ctx.set_virtualizable_heap_ptr(heap_ptr);
}

/// Decode a raw `vinfo.read_boxes` entry into the typed Value a
/// virtualizable shadow slot expects.  Local helper that mirrors
/// `majit-metainterp::pyjitpl::heap_value_for` so both seed sites (root
/// portal in `initialize_virtualizable` and bridge entry here) use the
/// same raw-bit → typed-Value rule.
fn value_for_slot(ty: Type, bits: i64) -> majit_ir::Value {
    match ty {
        Type::Int => majit_ir::Value::Int(bits),
        Type::Float => majit_ir::Value::Float(f64::from_bits(bits as u64)),
        Type::Ref => majit_ir::Value::Ref(majit_ir::GcRef(bits as usize)),
        Type::Void => majit_ir::Value::Void,
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FrameLivenessRegIndices {
    pub int: Vec<u32>,
    pub ref_: Vec<u32>,
    pub float: Vec<u32>,
}

impl FrameLivenessRegIndices {
    pub fn total_len(&self) -> usize {
        self.int.len() + self.ref_.len() + self.float.len()
    }

    pub fn flattened(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.total_len());
        out.extend_from_slice(&self.int);
        out.extend_from_slice(&self.ref_);
        out.extend_from_slice(&self.float);
        out
    }
}

/// Test/diagnostic-only direct JitCode `-live-` query: return live register
/// indices split by the three liveness banks.
/// RPython writes decoded values through `_callback_i/_callback_r/_callback_f`
/// into `registers_i/r/f[index]`; keeping the banks separate prevents Ref-only
/// semantic-slot remapping from swallowing Int/Float slots.
pub fn frame_liveness_reg_indices_by_bank_at(
    jitcode_index: i32,
    pc: i32,
) -> FrameLivenessRegIndices {
    frame_liveness_reg_indices_by_bank_from_pc(jitcode_index, pc)
}

/// `#124` Approach B encoder/decoder liveness query: identical to
/// [`frame_liveness_reg_indices_by_bank_at`] but resolves the JitCode
/// coordinate through [`PyJitCode::resolve_resume_pc_with_jitcode_pc`]. The
/// snapshot encoder (`collect_outer_active_boxes`) and the bridge/inline
/// decoders (`setup_bridge_sym`, `rebuild_inline_callee`) call this with
/// the SAME carried word so their register color sets agree.
pub fn frame_liveness_reg_indices_by_bank_at_with_jitcode_pc(
    jitcode_index: i32,
    carried_jitcode_pc: i32,
) -> FrameLivenessRegIndices {
    try_frame_liveness_reg_indices_by_bank_at_with_jitcode_pc(jitcode_index, carried_jitcode_pc)
        .unwrap_or_default()
}

/// The decline-reporting form of
/// [`frame_liveness_reg_indices_by_bank_at_with_jitcode_pc`].
///
/// The three ways the decode can fail — unknown jitcode index, a coordinate
/// that is not a `-live-` anchored startpoint, and a liveness offset past the
/// end of the table — are indistinguishable from a genuinely empty live set in
/// the `FrameLivenessRegIndices` return. A caller that materializes registers
/// from the result (rather than merely reading them) needs to tell the two
/// apart, or it builds a frame whose every register is unset and hands it on as
/// if it were complete.
pub fn try_frame_liveness_reg_indices_by_bank_at_with_jitcode_pc(
    jitcode_index: i32,
    carried_jitcode_pc: i32,
) -> Option<FrameLivenessRegIndices> {
    let payload = jitcode_payload_at(jitcode_index)?;
    METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        // No Python-pc reconstruction is available here: an absent carried
        // coordinate declines.
        let jit_pc: usize =
            payload.resolve_resume_pc_with_jitcode_pc(carried_jitcode_pc, sd.op_live)?;
        let off = payload.jitcode.get_live_vars_info(jit_pc, sd.op_live);
        let liveness_bytes = sd.liveness_info.snapshot_arc();
        let all_liveness = liveness_bytes.as_ref();
        if off + 2 >= all_liveness.len() {
            return None;
        }
        let length_i = all_liveness[off] as u32;
        let length_r = all_liveness[off + 1] as u32;
        let length_f = all_liveness[off + 2] as u32;
        // Diagnostic removed — use post-decode logging instead.
        let mut cursor = off + 3;
        use majit_jitcode::liveness::LivenessIterator;

        fn read_bank(cursor: &mut usize, length: u32, all_liveness: &[u8]) -> Vec<u32> {
            if length == 0 {
                return Vec::new();
            }
            let mut it = LivenessIterator::new(*cursor, length, all_liveness);
            let mut out = Vec::with_capacity(length as usize);
            while let Some(reg_idx) = it.next() {
                out.push(reg_idx);
            }
            *cursor = it.offset;
            out
        }

        let int = read_bank(&mut cursor, length_i, all_liveness);
        let ref_ = read_bank(&mut cursor, length_r, all_liveness);
        let float = read_bank(&mut cursor, length_f, all_liveness);
        Some(FrameLivenessRegIndices { int, ref_, float })
    })
}

/// Post-flip decode entry: `pc` is already the JitCode byte offset (the flip
/// stores the resolved offset in the frame pc word), so resolve the liveness
/// coordinate from `pc` itself — no carried twin needed. Byte-identical to the
/// `_with_jitcode_pc` form for every frame whose stored `pc` equals the twin
/// offset (all corpus frames).
pub(crate) fn frame_liveness_reg_indices_by_bank_from_pc(
    jitcode_index: i32,
    pc: i32,
) -> FrameLivenessRegIndices {
    frame_liveness_reg_indices_by_bank_at_with_jitcode_pc(jitcode_index, pc)
}

/// Whether the resume frame's pc word names a resolved JitCode `-live-` offset
/// (the flip stores a decodable offset when the guard's resume marker resolves,
/// else the raw Python pc). Post-flip this reproduces the retired
/// `frame.jitcode_pc != NO_JITCODE_PC` twin test: the twin is non-sentinel iff a
/// marker resolved iff `frame.pc` is a decodable offset.
pub(crate) fn frame_pc_is_resolved_offset_at(jitcode_index: i32, pc: i32) -> bool {
    let Some(payload) = jitcode_payload_at(jitcode_index) else {
        return false;
    };
    let op_live = METAINTERP_SD.with(|r| r.borrow().op_live);
    pc >= 0 && payload.jitcode.can_decode_live_vars(pc as usize, op_live)
}

pub fn frame_liveness_reg_indices_at(jitcode_index: i32, pc: i32) -> Vec<u32> {
    frame_liveness_reg_indices_by_bank_at(jitcode_index, pc).flattened()
}

/// Test-only fixture seeder for `_with_compiled_trace_jitcode` unit tests.
/// Gated behind `#[cfg(any(test, feature = "test-support"))]` so production
/// builds cannot reach the synthetic register-bank fill — RPython has no
/// counterpart and live holes get filled with `0xfeed` / `0` dummy consts
/// that would corrupt a real trace if invoked outside fixtures.
///
/// `pyjitpl.py get_list_of_active_boxes` reads each live register
/// directly from its kind-specific bank (`registers_i[reg]` /
/// `registers_r[reg]` / `registers_f[reg]`). There is deliberately no
/// `registers_r_semantic` fallback. Production tracers fill the kind banks via
/// `bcd_op` dispatch as the trace is recorded, but unit-test fixtures
/// build a `PyreSym` directly through `from_test_state` and never run the
/// dispatch loop — without seeding, `current_fail_args` returns a vector
/// full of `OpRef::NONE` holes.
///
/// This helper mirrors what production code does:
///   1. `setup_kind_register_banks` to size the three banks +
///      copy-constants per `pyjitpl.py`.
///   2. Place each `(stack_depth, opref)` at the semantic frame-mirror
///      slot `registers_r[nlocals + depth]`. Production guard capture
///      then materializes the color-indexed Ref bank from the same
///      semantic/vable source before reading liveness.
///   3. Fill any still-`OpRef::NONE` Int/Float bank slot listed in
///      `live_pc`'s bank-split liveness with a typed dummy constant.
///      Ref slots are provided by the caller-provided stack slot map above.
#[cfg(any(test, feature = "test-support"))]
pub fn seed_compiled_trace_jitcode_test_state(
    sym: &mut PyreSym,
    ctx: &mut TraceCtx,
    jitcode_index: i32,
    live_jit_pc: i32,
    stack_slots: &[(usize, OpRef)],
) {
    if !sym.jitcode.is_null() {
        sym.setup_kind_register_banks(ctx);
    }

    for &(depth, opref) in stack_slots {
        // Match production stack writes: keep `registers_r` as a
        // semantic frame mirror. The encoder builds the color-indexed
        // bank from this mirror/vable view at snapshot time.
        let reg_idx = crate::trace_opcode::stack_slot_reg_idx(sym, depth);
        if reg_idx >= sym.registers_r.len() {
            sym.registers_r.resize(reg_idx + 1, OpRef::NONE);
        }
        sym.registers_r.set(reg_idx, opref);
    }

    let banks = frame_liveness_reg_indices_by_bank_from_pc(jitcode_index, live_jit_pc);
    for &reg in &banks.int {
        let r = reg as usize;
        if r >= sym.registers_i.len() {
            sym.registers_i.resize(r + 1, OpRef::NONE);
        }
        if sym.registers_i[r].is_none() {
            sym.registers_i[r] = ctx.const_int(0xfeed);
        }
    }
    for &reg in &banks.float {
        let r = reg as usize;
        if r >= sym.registers_f.len() {
            sym.registers_f.resize(r + 1, OpRef::NONE);
        }
        if sym.registers_f[r].is_none() {
            sym.registers_f[r] = ctx.const_float(0);
        }
    }
}

/// Trivia-aware per-PC `(color, semantic_slot)` resume entries keyed directly
/// by a JitCode byte offset. `None` when the index, offset, or trivia twin is
/// unavailable; callers that preserve the legacy empty-result behavior may use
/// `unwrap_or_default()`.
pub fn pcdep_trivia_at(jitcode_index: i32, jit_pc: i32) -> Option<Vec<(u8, u16, u16)>> {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        sd.jitcodes
            .get(jitcode_index as usize)?
            .payload
            .pcdep_trivia_for_jitcode_pc(usize::try_from(jit_pc).ok()?)
            .map(ToOwned::to_owned)
    })
}

/// Depth-based `valuestackdepth` for `w_code` at `py_pc`:
/// `nlocals + ncells + depth_at_py_pc[py_pc]`.  Mirrors the encoder's
/// published vsd (the `jitcode_dispatch` valuestackdepth publish).  The
/// stack base is `varnames + ncells` (`pyframe.py:111 valuestackdepth =
/// co_nlocals + ncellvars + nfreevars`), not `varnames` alone, so a code
/// object with cells/freevars (a closure/nested function) is not
/// under-counted — see `concrete_nlocals`.
///
/// A multi-frame (inlined-callee) guard restores the whole virtualizable
/// positionally via `write_from_resume_data_partial`, which writes the
/// CHAIN frame's `valuestackdepth` — the OUTER section's depth.  When the
/// resume pc is then overridden to the innermost section's `py_pc`, the
/// physical frame's vsd must be corrected to that section's depth, else
/// the interpreter resumes at the inner pc carrying the outer depth (an
/// over-count that materializes a stray operand slot).  Returns `None`
/// when the code or the liveness entry for `py_pc` is missing, or when the
/// forward analysis never reached `py_pc`.
pub fn depth_based_vsd_for_wcode(w_code: usize, py_pc: usize) -> Option<usize> {
    if w_code == 0 {
        return None;
    }
    let raw_code = unsafe {
        pyre_interpreter::w_code_get_ptr(w_code as PyObjectRef)
            as *const pyre_interpreter::CodeObject
    };
    if raw_code.is_null() {
        return None;
    }
    let code = unsafe { &*raw_code };
    let stack_base = code.varnames.len() + pyre_interpreter::pyframe::ncells(code);
    // `depth_at_py_pc` collapses the forward pass's `usize::MAX` unreachable
    // sentinel to `0`, which answers `stack_base` — a confidently empty operand
    // stack — for a pc the analysis never reached.  `stack_depth_at` keeps the
    // sentinel distinguishable; decline there instead, which every caller reads
    // as "leave the frame's own `valuestackdepth` alone".  That table is one
    // longer than the per-PC one, so keep the instruction-count bound.
    if py_pc >= code.instructions.len() {
        return None;
    }
    let depth = crate::liveness::liveness_for(raw_code).stack_depth_at(py_pc)?;
    Some(stack_base + depth)
}

/// Inputs `setup_bridge_sym` needs to rebuild the slot-indexed semantic
/// register file from the color-indexed resume decode of the parent
/// jitcode at the guard-failure PC. Mirrors the input block
/// `collect_outer_active_boxes` (jitcode_dispatch.rs) builds for the
/// encode side, but keyed by `jitcode_index` rather than a live
/// `sym.jitcode`, so the bridge decoder can invert each live Ref color to
/// its `locals_cells_stack_w` slot via `semantic_ref_slot_for_reg_color`.
///
/// `has_color_map` is `false`
/// for a jitcode the codewriter never colored (empty `pcdep_color_slots`) —
/// byte-identical over the corpus to the retired flat-map identity test
/// (`local_color_map.is_empty() && stack_color_map.is_empty()`), which a
/// zero-local frame owning a freely-colored operand stack correctly failed
/// (its `stack_color_map` was non-empty → its `pcdep_color_slots` is too).
pub(crate) struct BridgeSemanticMaps {
    /// `true` when the codewriter colored this jitcode (non-empty
    /// `pcdep_color_slots`); drives the per-slot pcdep inversion. `false` for
    /// skeletons whose color bank is the slot mirror.
    pub has_color_map: bool,
    pub stack_depth_at_pc: usize,
    /// #348 Part (2): per-PC `(color, slot)` entries at the resume PC.
    /// Non-empty only for gated jitcodes; when present it is the
    /// authoritative color→slot inversion (the same per-program-point color
    /// space the `-live-` markers and encode side use), superseding the flat
    /// `local_color_map` / `stack_color_map` for this bridge.
    /// Tuples: `(bank, color, slot)` where bank = Kind::index()
    /// (0=Int, 1=Ref, 2=Float).
    pub pcdep_entries: Vec<(u8, u16, u16)>,
}

/// Counters for the `via_py_pc` fallback of
/// [`bridge_semantic_maps_at_with_jitcode_pc`], which indexes the
/// Python-PC-keyed `LiveVars::depth_at_py_pc`. One instance per *leg* that
/// reaches the fallback — the `site=` field of the printed line names the leg,
/// not the caller.
///
/// The partition that decides whether a Python coordinate is being read at a
/// point the caller did not mean: an offset past the end of the table reads
/// `None` and the fallback answers 0, but one that lands inside returns a real
/// depth belonging to a different instruction.
///
/// Which bucket a caller can produce is fixed by the argument it passes, so the
/// two ends must not be welded together:
/// * `setup_bridge_sym` is the one consumer that reads `stack_depth_at_pc`
///   outside a `pcdep_entries` loop (`stack_only.max(..)` →
///   `semantic_prefix_len`), and it passes `None` — so `no_py_pc` is the only
///   bucket reachable from there and `in_range_*` is structurally unreachable.
/// * `in_range_nonzero` can only come from the one caller that supplies a
///   `Some` coordinate, `resume_snapshot.rs`
///   `walker_capture_snapshot_for_last_guard_impl`'s `liveness_py_pc`, whose
///   depth feeds that function's own `semantic_limit`. That call is dominated
///   by `resolve_resume_pc_with_jitcode_pc` on the same payload, `op_live` and
///   coordinate, and that helper returns `Some` only under
///   `can_decode_live_vars` — the very predicate selecting the twin arm here.
///   So `site=non_decodable` is unreachable from that caller: only
///   `EMPTY_TWIN_MISS` can fire there. A printed `site=non_decodable` line
///   therefore never belongs to the one `Some` caller, which is what excludes
///   it from every fire the corpus has produced so far.
///
/// That caller's coordinate is a Python PC by construction, so a nonzero depth
/// there is the designed result of the leg, not an anomaly — the bucket records
/// which leg ran, and does not by itself witness a coordinate mixup.
///
/// The counters are per leg, not per process: with one shared counter only the
/// leg that fires first is ever named, so a run in which one leg never executed
/// is indistinguishable from one in which it executed second.
///
/// Pure telemetry — with the variable unset every counter path is skipped.
struct EmptyTwinSite {
    name: &'static str,
    hits: std::sync::atomic::AtomicU64,
    no_py_pc: std::sync::atomic::AtomicU64,
    null_code: std::sync::atomic::AtomicU64,
    out_of_range: std::sync::atomic::AtomicU64,
    in_range_zero: std::sync::atomic::AtomicU64,
    in_range_nonzero: std::sync::atomic::AtomicU64,
}

impl EmptyTwinSite {
    const fn new(name: &'static str) -> Self {
        const fn zero() -> std::sync::atomic::AtomicU64 {
            std::sync::atomic::AtomicU64::new(0)
        }
        Self {
            name,
            hits: zero(),
            no_py_pc: zero(),
            null_code: zero(),
            out_of_range: zero(),
            in_range_zero: zero(),
            in_range_nonzero: zero(),
        }
    }
}

/// The carried jitcode coordinate decoded, but the twin tables had no entry.
static EMPTY_TWIN_MISS: EmptyTwinSite = EmptyTwinSite::new("twin_miss");
/// No usable jitcode coordinate: either none was carried (`NO_JITCODE_PC`) or
/// the carried offset does not decode as a `-live-` anchored startpoint.
static EMPTY_TWIN_NON_DECODABLE: EmptyTwinSite = EmptyTwinSite::new("non_decodable");

/// `jit_pc` / `twin` report the coordinate that reached the fallback and the
/// jitcode-keyed containing depth used there. Before that twin became the
/// production fallback, the 399-file synth corpus showed all 20 executions at
/// `jit_pc >= 0` with `twin = Some(3..=5)` while the function returned 0.
///
/// Read the `Some`-ness of that pair narrowly. `twin` is a FLOOR lookup over
/// `depth_containing_by_jit_pc`, whose keys are those of a table the codewriter
/// pads with key 0 (`py_floor_by_jit_pc`'s `insert(0, (0, 0))`), so once the
/// twin is populated at all it answers `Some` at every non-negative offset. A
/// `None` here therefore records an empty twin, not an out-of-range offset;
/// only the values carry per-coordinate information.
///
/// The counters remain per leg, not per call site, so a fire cannot be
/// attributed to one of the callers. `setup_bridge_sym` is the consumer whose
/// `stack_depth_at_pc` read is unconditional.
fn empty_twin_census(
    site: &EmptyTwinSite,
    rp: Option<usize>,
    table_len: Option<usize>,
    depth: u16,
    jit_pc: i32,
    twin: Option<u16>,
) {
    // Shares `PYRE_M73_EMPTYTWIN_CENSUS` with `py_coord::note_empty_twin_fallback`,
    // so one run reports both. That one prints `[m73-emptytwin]`; this one prints
    // `[m73-bridge-maps]`, distinct under a case-insensitive filter too.
    if !crate::py_coord::emptytwin_census_enabled() {
        return;
    }
    use std::sync::atomic::Ordering;
    let name = site.name;
    let hits = site.hits.fetch_add(1, Ordering::Relaxed) + 1;
    // `rp >= len` must be tested before the `depth == 0` arm: `table.get(rp)`
    // already collapses an out-of-range read to 0, so the zero arm would
    // otherwise swallow it and the two would be indistinguishable.
    let (bucket, counter, live) = match (rp, table_len) {
        (None, _) => ("no_py_pc", &site.no_py_pc, false),
        (Some(_), None) => ("null_code", &site.null_code, false),
        (Some(rp), Some(len)) if rp >= len => ("out_of_range", &site.out_of_range, false),
        (Some(_), Some(_)) if depth == 0 => ("in_range_zero", &site.in_range_zero, false),
        (Some(_), Some(_)) => ("in_range_nonzero", &site.in_range_nonzero, true),
    };
    let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
    // Every bucket announces its first witness, so the buckets that do fire are
    // the positive control for the ones that do not: a bucket that stays silent
    // on a stream carrying other buckets' lines is empty, not unprinted.
    // `in_range_nonzero` gets 20 rather than one because it is the only bucket
    // whose value varies — the rest answer 0 by construction, so one witness
    // says everything about them.
    if n == 1 || (live && n <= 20) {
        eprintln!(
            "[m73-bridge-maps] site={name} bucket={bucket} n={n} rp={rp:?} len={table_len:?} depth={depth} jit_pc={jit_pc} twin={twin:?}"
        );
    }
    if hits % 100_000 == 0 {
        eprintln!(
            "[m73-bridge-maps] totals site={name} hits={hits} no_py_pc={} null_code={} out_of_range={} in_range_zero={} in_range_nonzero={}",
            site.no_py_pc.load(Ordering::Relaxed),
            site.null_code.load(Ordering::Relaxed),
            site.out_of_range.load(Ordering::Relaxed),
            site.in_range_zero.load(Ordering::Relaxed),
            site.in_range_nonzero.load(Ordering::Relaxed),
        );
    }
}

/// When a kept-stack branch guard carries the guard's own jitcode
/// coordinate (`jitcode_pc != NO_JITCODE_PC`), the pcdep/depth tables
/// must be keyed at the GUARD's Python PC — the encode side
/// (`collect_outer_active_boxes`) already keys its pcdep by
/// `liveness_py_pc = guard_py_pc`, and the liveness decode
/// (`frame_liveness_reg_indices_by_bank_at_with_jitcode_pc`) resolves
/// through the carried `jitcode_pc`. Without this the decode-side
/// color→slot inversion reads the merge-target PC's pcdep (where the
/// computed kept temp is dead), leaving the temp as `OpRef::NONE`.
/// `py_pc` is the caller's OWN Python coordinate, or `None` when it holds
/// only a JitCode coordinate. It is not interchangeable with `jitcode_pc`:
/// it indexes the Python-PC-keyed static liveness, and `py_coord.rs`'s
/// module invariant is that "runtime consumers never project a JitCode PC
/// through a Python-keyed map". `None` is what makes that statable —
/// passing the JitCode word here reads a foreign instruction's depth
/// whenever the offset happens to fall inside the table.
pub(crate) fn bridge_semantic_maps_at_with_jitcode_pc(
    jitcode_index: i32,
    py_pc: Option<i32>,
    jitcode_pc: i32,
) -> BridgeSemanticMaps {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        let Some(jc) = sd.jitcodes.get(jitcode_index as usize) else {
            return BridgeSemanticMaps {
                has_color_map: false,
                stack_depth_at_pc: 0,
                pcdep_entries: Vec::new(),
            };
        };
        let payload = &jc.payload;
        // Expand a tagged branch `orgpc` word to the block-head marker
        // BEFORE the `>= 0` / offset uses below — a tagged NEGATIVE word cast to
        // usize would otherwise be a huge OOB index. No-op for offsets /
        // NO_JITCODE_PC (flip-off), so byte-identical when off.
        let jitcode_pc = crate::jitcode_dispatch::expand_branch_carried(payload, jitcode_pc);
        // A `Some` py_pc is already the published resume coordinate, so the
        // py_pc-keyed liveness/depth table fallback reads it directly.
        //
        // When the carried `jitcode_pc` is set (kept-stack branch
        // guard), resolve the guard's Python PC from the jitcode coordinate
        // and use it for the pcdep/depth lookup, matching the encode side's
        // `liveness_py_pc = guard_py_pc` keying. The guard PC is where the
        // computed kept operand-stack temps are live; at the merge-target PC
        // they've been consumed and carry no pcdep entry.
        //
        // A caller holding no Python coordinate reads the floor-only static
        // depth twin.  Frame width is a containing-operation property and does
        // not require decoding the register-value stream at `jitcode_pc`.
        //
        // Jitcode-keyed spellings of this same static table exist:
        // `depth_containing_for_jitcode_pc` is built (`finalize_jitcode`, the
        // `depth_containing_by_jit_pc` block) to reproduce
        // `depth_at_py_pc[containing_py]` for every offset, and
        // `depth_trivia_for_jitcode_pc` reads it at the trivia-skipped py.
        // `collect_outer_active_boxes`, the encode half this mirrors, already
        // sources its own `stack_depth_at_pc` from the JitCode-keyed depth twin
        // with no Python PC in hand. The census below measured what it answers:
        // over the 399-file synth corpus every one of the 20 executions of this
        // leg carried a non-negative offset at which the containing twin
        // answers 3, 4 or 5. Returning 0 truncated the reconstructed frame's
        // semantic prefix and dropped exactly those operand-stack slots.
        //
        // What this leg must not do, and did before the `Option`, is index the
        // Python-keyed table with the JitCode word.
        let via_py_pc = |rp: Option<i32>, site: &'static EmptyTwinSite| -> usize {
            // The JitCode-keyed static twin is the production fallback when no
            // sanctioned Python coordinate is carried.  It is built from the
            // same `depth_at_py_pc` table as the raw read below and therefore
            // preserves frame width without projecting a JitCode byte offset
            // through a Python-keyed map.
            let twin = (jitcode_pc >= 0)
                .then(|| payload.depth_containing_for_jitcode_pc(jitcode_pc as usize))
                .flatten();
            let Some(rp) = rp.and_then(|p| usize::try_from(p).ok()) else {
                let depth = twin.unwrap_or(0);
                empty_twin_census(site, None, None, depth, jitcode_pc, twin);
                return depth as usize;
            };
            if payload.code_ptr.is_null() {
                empty_twin_census(site, Some(rp), None, 0, jitcode_pc, twin);
                return 0;
            }
            let table = crate::liveness::liveness_for(payload.code_ptr).depth_at_py_pc();
            let depth = table.get(rp).copied().unwrap_or(0);
            empty_twin_census(site, Some(rp), Some(table.len()), depth, jitcode_pc, twin);
            depth as usize
        };
        let (stack_depth_at_pc, pcdep_entries) = if jitcode_pc >= 0
            && payload
                .jitcode
                .can_decode_live_vars(jitcode_pc as usize, sd.op_live)
        {
            let jp = jitcode_pc as usize;
            // Decode-identity: source depth/pcdep from the carried genuine
            // `jitcode_pc` via the predecessor-keyed twins.  Once liveness is
            // decodable, the two sidecars are independently optional: a miss
            // in one must not discard the other.
            let pcdep = payload.pcdep_for_jitcode_pc(jp).unwrap_or_default();
            let depth = payload
                .depth_for_jitcode_pc_pred(jp)
                .map(usize::from)
                .unwrap_or_else(|| via_py_pc(py_pc, &EMPTY_TWIN_MISS));
            (depth, pcdep)
        } else {
            // RPython resume.py rebuild_from_resumedata does
            // setup_resume_at_op(pc) and then passes that frame's
            // get_current_position_info() to consume_boxes.  The latter calls
            // get_live_vars_info(pc), so a position without a live-anchored
            // register stream cannot supply a color-to-slot pcdep map.  Keep
            // the independently useful static frame width, but do not combine
            // it with pcdep from a coordinate whose values cannot be decoded.
            (via_py_pc(py_pc, &EMPTY_TWIN_NON_DECODABLE), Vec::new())
        };
        BridgeSemanticMaps {
            // #73: the codewriter colored this jitcode iff `pcdep_color_slots`
            // is non-empty — the field-free replacement for the retired flat
            // `local_color_map.is_empty() && stack_color_map.is_empty()` test.
            has_color_map: payload.metadata.has_color_map,
            stack_depth_at_pc,
            pcdep_entries,
        }
    })
}

/// For a caller that passes a JitCode coordinate and no Python one. Named for
/// that coordinate so the word cannot be re-routed into the Python-PC slot.
///
/// Three callers genuinely hold nothing else (`bridge_subwalk.rs`
/// `recipe.jitcode_pc`, `residual_call.rs` `op_pc`, `resume_snapshot.rs`
/// `callee_jitcode_pc`). The two `RebuiltFrame` callers (`state.rs`
/// `reconstruct_inline_recipe` and `setup_bridge_sym`) hold `jitcode_index`
/// and `pc`; `resume_py_pc_for_jitcode_word` recovers the Python coordinate.
/// This function uses the JitCode-keyed containing depth twin and does not
/// route the JitCode word through a Python-keyed map.
pub(crate) fn bridge_semantic_maps_from_jitcode_pc(
    jitcode_index: i32,
    jitcode_pc: i32,
) -> BridgeSemanticMaps {
    bridge_semantic_maps_at_with_jitcode_pc(jitcode_index, None, jitcode_pc)
}

/// Operand-stack Ref constants (`(semantic_slot, raw_ref)`) at the resume PC of
/// a jitcode. The pcdep color map records live Variables only;
/// `reconstruct_inline_recipe` uses this to refill the registerless constant
/// slots an inlined-callee guard resume leaves empty after the color→slot
/// inversion. Sources the slots from the predecessor-keyed
/// `const_ref_slots_by_jit_pc` twin so the constant lookup uses the same
/// coordinate as `pcdep_entries` and the liveness decode.
pub(crate) fn const_ref_slots_at_pc_at(jitcode_index: i32, jitcode_pc: i32) -> Vec<(u16, i64)> {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        let Some(jc) = sd.jitcodes.get(jitcode_index as usize) else {
            return Vec::new();
        };
        // Expand a tagged branch `orgpc` word to the block-head marker
        // before the `>= 0` / offset uses below. No-op for offsets /
        // NO_JITCODE_PC (flip-off), so byte-identical when off.
        let jitcode_pc = crate::jitcode_dispatch::expand_branch_carried(&jc.payload, jitcode_pc);
        if jitcode_pc != majit_ir::resumedata::NO_JITCODE_PC && jitcode_pc >= 0 {
            let jp = jitcode_pc as usize;
            if jc.payload.jitcode.can_decode_live_vars(jp, sd.op_live) {
                // Source the const slots directly from the carried
                // genuine `jitcode_pc` via the predecessor-keyed twin — the same
                // decode-identity shape the pcdep/depth twins use at
                // `bridge_semantic_maps_at_with_jitcode_pc`. A colored jitcode
                // returns `Some` (an empty slot list is a legitimate hit).
                if let Some(slots) = jc.payload.const_ref_slots_for_jitcode_pc(jp) {
                    return slots;
                }
            }
        }
        Vec::new()
    })
}

pub(crate) fn const_ref_slots_from_pc(jitcode_index: i32, pc: i32) -> Vec<(u16, i64)> {
    const_ref_slots_at_pc_at(jitcode_index, pc)
}

/// Return the post-regalloc Ref-bank colors of the portal red args
/// (`pypy/module/pypyjit/interp_jit.py reds = ['frame', 'ec']`) for
/// the registered jitcode at `jitcode_index`. Both are `u16::MAX` for skeletons.
///
/// Used by per-code dispatch and reconstructed inline callees to seed the
/// frame/ec colors from the current MIFrame's own red inputs. Root bridge
/// setup keeps `ec` in its second root inputarg, outside the resumed
/// MIFrame's semantic register bank.
pub fn portal_red_regs_at(jitcode_index: i32) -> (u16, u16) {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        sd.jitcodes
            .get(jitcode_index as usize)
            .map(|jc| {
                (
                    jc.payload.metadata.portal_frame_reg,
                    jc.payload.metadata.portal_ec_reg,
                )
            })
            .unwrap_or((u16::MAX, u16::MAX))
    })
}

/// One `(frame_ptr, stack_base)` per `framestack` level, outermost first.
///
/// `blackhole.py convert_and_run_from_pyjitpl` copies each MIFrame
/// independently; pyre's extra `virtualizable_ptr` field has to be
/// rebound to that level's own red frame so traceback recording and
/// `executioncontext.py enter`/`leave` do not share the portal frame.
/// A helper jitcode with no portal frame red, or an unstamped register,
/// yields `(0, 0)` for that level.
pub fn per_frame_vables_from_framestack(
    framestack: &majit_metainterp::MIFrameStack,
) -> Vec<(i64, usize)> {
    framestack
        .frames
        .iter()
        .map(|frame| {
            let Ok(jitcode_index) = i32::try_from(frame.jitcode.index()) else {
                return (0, 0);
            };
            let frame_reg = portal_red_regs_at(jitcode_index).0;
            if frame_reg == u16::MAX {
                return (0, 0);
            }
            let Some(frame_ptr) = frame.ref_value_for_blackhole(frame_reg as usize) else {
                return (0, 0);
            };
            if frame_ptr == 0 {
                return (0, 0);
            }
            let stack_base = concrete_nlocals(frame_ptr as usize).unwrap_or(0);
            (frame_ptr, stack_base)
        })
        .collect()
}

pub fn built_as_portal_at(jitcode_index: i32) -> bool {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        sd.jitcodes
            .get(jitcode_index as usize)
            .map(|jc| jc.payload.metadata.built_as_portal)
            .unwrap_or(false)
    })
}

/// Map a post-regalloc Ref-bank color back to the semantic
/// `locals_cells_stack_w` slot it denotes at the current PC.
///
/// **Decoder-only**: the encoder
/// (`get_list_of_active_boxes`) reads `registers_r[color]`
/// directly and derives `semantic_idx` inline.  This function is
/// still used by `restore_guard_failure_values` (the decoder) which
/// needs the semantic slot index to call `set_local_at` /
/// `set_stack_at` on the concrete PyFrame.
///
/// After stack-slot pinning removal, stack slots are no longer forced to
/// occupy colors `nlocals + d`; the reverse lookup consults the per-PC
/// `pcdep_color_slots` entries, bounded to the LIVE stack prefix at the
/// current PC. Only if no live stack slot owns the color can the color
/// resolve to a local slot.
pub(crate) fn semantic_ref_slot_for_reg_color(
    nlocals: usize,
    stack_only: usize,
    pcdep_entries: &[(u8, u16, u16)],
    reg: usize,
) -> Option<usize> {
    semantic_slot_for_reg_color(nlocals, stack_only, pcdep_entries, 1, reg)
}

/// Every live semantic Ref slot owned by one post-regalloc color at this PC.
/// A COPY/short-circuit shape may deliberately map the same MIFrame register
/// to more than one semantic stack slot; rebuilding pyre's slot-indexed frame
/// mirror must fan that single register value out to all of them.
pub(crate) fn semantic_ref_slots_for_reg_color(
    nlocals: usize,
    stack_only: usize,
    pcdep_entries: &[(u8, u16, u16)],
    reg: usize,
) -> Vec<usize> {
    let mut slots = Vec::new();
    for &(bank, color, slot) in pcdep_entries {
        if bank != 1 || color as usize != reg {
            continue;
        }
        let slot = slot as usize;
        let live = slot < nlocals || slot - nlocals < stack_only;
        if live && slots.last().copied() != Some(slot) {
            slots.push(slot);
        }
    }
    slots
}

/// Per-PC `(bank, color, slot)` map inversion: given a register bank and
/// color, return the semantic `locals_cells_stack_w` slot it maps to at the
/// current PC. Prefer stack slots over locals (stack_match.or(local_match)).
pub(crate) fn semantic_slot_for_reg_color(
    nlocals: usize,
    stack_only: usize,
    pcdep_entries: &[(u8, u16, u16)],
    bank: u8,
    reg: usize,
) -> Option<usize> {
    // The per-PC `(bank, color, slot)` map is the authoritative color→slot
    // inversion at this resume PC — each slot's TRUE per-program-point
    // color rather than a flat one-color-per-slot label. Prefer the
    // smallest operand-stack slot carrying this color, else the smallest
    // local slot (entries are sorted by `(bank, color, slot)`, so locals
    // precede stack within a color). A stack slot counts only while it is
    // below the live stack depth (`stack_only`) at THIS resume — the
    // per-PC entries are gated by the compile-time depth, which can
    // exceed the runtime `stack_only` at a residual-call fallthrough.
    //
    // An empty map yields None (no live color owns this slot at this pc).
    let mut local_match: Option<usize> = None;
    let mut stack_match: Option<usize> = None;
    for &(b, color, slot) in pcdep_entries {
        if b != bank || color as usize != reg {
            continue;
        }
        let s = slot as usize;
        if s >= nlocals {
            if s - nlocals < stack_only && stack_match.is_none() {
                stack_match = Some(s);
            }
        } else if local_match.is_none() {
            local_match = Some(s);
        }
    }
    stack_match.or(local_match)
}

/// Inverse of [`semantic_slot_for_reg_color`] for the Ref bank: given a
/// semantic `locals_cells_stack_w` slot, return the Ref-bank color that owns
/// it at this PC per the `(bank, color, slot)` map. Used by the deep-kept
/// operand-stack recovery in `walker_capture_snapshot_for_last_guard_impl` to
/// name the guard-PC register (`registers_r[color]`) holding a kept operand-
/// stack slot the walk mirror lost. Returns the smallest matching color;
/// `None` when no live Ref color owns the slot at this PC.
pub(crate) fn semantic_slot_color_for_ref_slot(
    pcdep_entries: &[(u8, u16, u16)],
    slot: usize,
) -> Option<usize> {
    let mut best: Option<usize> = None;
    for &(b, color, s) in pcdep_entries {
        if b != 1 || s as usize != slot {
            continue;
        }
        let c = color as usize;
        best = Some(best.map_or(c, |cur: usize| cur.min(c)));
    }
    best
}

/// Resolve a resumed semantic Ref slot to its live register color.  Empty
/// metadata is retained for skeleton/fixture compatibility; a partially
/// populated per-PC map is authoritative, so an absent slot has no register.
fn reconstructed_ref_slot_color(pcdep_entries: &[(u8, u16, u16)], slot: usize) -> Option<usize> {
    semantic_slot_color_for_ref_slot(pcdep_entries, slot)
        .or_else(|| pcdep_entries.is_empty().then_some(slot))
}

// Sentinel null JitCode for uninitialized PyreSym.
//
// Cannot be `static` because `Arc::new` is not const; use a thread_local
// LazyCell so the initialiser runs once per thread and the resulting
// reference stays valid for the thread's lifetime.
thread_local! {
    static NULL_JITCODE_CELL: std::cell::OnceCell<JitCode> = const { std::cell::OnceCell::new() };
}

fn null_jitcode() -> &'static JitCode {
    NULL_JITCODE_CELL.with(|cell| {
        let r = cell.get_or_init(|| JitCode {
            index: -1,
            payload: std::sync::Arc::new(crate::PyJitCode::skeleton(std::ptr::null())),
        });
        // SAFETY: per-thread `OnceCell` initialises once; the
        // resulting reference lives for the thread's lifetime.
        unsafe { &*(r as *const JitCode) }
    })
}

/// Traced value — RPython `FrontendOp(position, _resint/_resref/_resfloat)` parity.
///
/// Carries both the symbolic IR reference (OpRef) and the concrete
/// execution value (ConcreteValue). Created by opcode handlers that
/// compute concrete results alongside IR recording.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrontendOp {
    pub opref: OpRef,
    pub concrete: ConcreteValue,
}

impl FrontendOp {
    pub fn new(opref: OpRef, concrete: ConcreteValue) -> Self {
        Self { opref, concrete }
    }

    /// `history.py` `FrontendOp(pos)` parity — the `type='v'`
    /// (void) variant carries only a recorder position with no value
    /// attribute. Pyre folds RPython's class hierarchy
    /// (`IntFrontendOp`/`RefFrontendOp`/`FloatFrontendOp`/bare
    /// `FrontendOp`) into `ConcreteValue` variants, so the void case is
    /// `concrete: Null`.
    pub fn void(opref: OpRef) -> Self {
        Self {
            opref,
            concrete: ConcreteValue::Null,
        }
    }
}

/// Typed concrete value — RPython `FrontendOp._resint/_resref/_resfloat` parity.
///
/// Python bytecode uses untyped locals, so we use a tagged enum instead of
/// RPython's separate `registers_i/r/f` arrays. Each variant corresponds to
/// one of RPython's Box types: `BoxInt`, `BoxPtr`, `BoxFloat`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConcreteValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    Ref(PyObjectRef),
    Null,
}

/// Convert a frame slot value to ConcreteValue, preserving null pointers
/// as Ref(PY_NULL) instead of ConcreteValue::Null. Frame slots always
/// contain known values — null means "uninitialized local", not "untracked".
pub(crate) fn concrete_value_from_slot(obj: PyObjectRef) -> ConcreteValue {
    if obj.is_null() {
        return ConcreteValue::Ref(pyre_object::PY_NULL);
    }
    ConcreteValue::from_pyobj(obj)
}

fn concrete_value_from_ir_value(value: majit_ir::Value) -> ConcreteValue {
    match value {
        Value::Ref(gc_ref) => concrete_value_from_slot(gc_ref.0 as PyObjectRef),
        Value::Int(n) => ConcreteValue::Int(n),
        Value::Float(n) => ConcreteValue::Float(n),
        Value::Void => ConcreteValue::Null,
    }
}

impl ConcreteValue {
    /// Convert from PyObjectRef (unbox if possible).
    /// Null pointers become ConcreteValue::Null ("untracked").
    pub fn from_pyobj(obj: PyObjectRef) -> Self {
        if obj.is_null() {
            return ConcreteValue::Null;
        }
        unsafe {
            if is_bool(obj) {
                ConcreteValue::Bool(w_bool_get_value(obj))
            } else if is_trace_plain_int(obj) {
                ConcreteValue::Int(w_int_get_value(obj))
            } else if is_trace_plain_float(obj) {
                ConcreteValue::Float(w_float_get_value(obj))
            } else {
                ConcreteValue::Ref(obj)
            }
        }
    }

    /// Convert to PyObjectRef (box if needed).
    pub fn to_pyobj(self) -> PyObjectRef {
        match self {
            ConcreteValue::Int(v) => w_int_new(v),
            ConcreteValue::Float(v) => pyre_object::w_float_new(v),
            ConcreteValue::Bool(v) => pyre_object::w_bool_from(v),
            ConcreteValue::Ref(obj) => obj,
            ConcreteValue::Null => PY_NULL,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, ConcreteValue::Null)
    }

    /// RPython box.getint() parity.
    pub fn getint(&self) -> Option<i64> {
        match self {
            ConcreteValue::Int(v) => Some(*v),
            ConcreteValue::Bool(v) => Some(*v as i64),
            _ => None,
        }
    }

    /// RPython box.getfloatstorage() parity.
    pub fn getfloat(&self) -> Option<f64> {
        match self {
            ConcreteValue::Float(v) => Some(*v),
            ConcreteValue::Int(v) => Some(*v as f64),
            ConcreteValue::Bool(v) => Some(*v as i64 as f64),
            _ => None,
        }
    }

    /// RPython box.getref_base() parity.
    pub fn getref(&self) -> PyObjectRef {
        self.to_pyobj()
    }

    /// Convert to majit IR Type.
    pub fn ir_type(&self) -> Type {
        match self {
            ConcreteValue::Int(_) | ConcreteValue::Bool(_) => Type::Int,
            ConcreteValue::Float(_) => Type::Float,
            ConcreteValue::Ref(_) => Type::Ref,
            ConcreteValue::Null => Type::Ref,
        }
    }

    /// Convert to majit IR `Value` for a Ref-typed virtualizable slot, or
    /// `None` when there is no valid heap pointer to record.
    ///
    /// `locals_cells_stack_w` is declared as a W_Root array
    /// (virtualizable.py read_boxes), so slots mirror RPython `Box(W_Root)` —
    /// `read_boxes` / `write_boxes` always see real boxed W_Root values.
    /// Pyre's lazy boxing means `wrapint` / `wrapfloat` emit a
    /// `NewWithVtable` OpRef without eagerly allocating a `W_IntObject` /
    /// `W_FloatObject`, so there is no heap pointer to flush back through
    /// `synchronize_virtualizable`. Returning `None` lets callers skip the
    /// concrete half of the shadow and update only the OpRef via
    /// `set_virtualizable_box_at`, preserving whatever valid W_Root the
    /// slot previously held instead of corrupting the PyFrame with an
    /// invalid pointer.
    pub fn to_ir_ref_value(&self) -> Option<majit_ir::Value> {
        match self {
            ConcreteValue::Ref(obj) => Some(majit_ir::Value::Ref(majit_ir::GcRef(*obj as usize))),
            ConcreteValue::Null => Some(majit_ir::Value::Ref(majit_ir::GcRef(0))),
            ConcreteValue::Int(_) | ConcreteValue::Float(_) | ConcreteValue::Bool(_) => None,
        }
    }
}

#[inline]
unsafe fn is_trace_plain_int(obj: PyObjectRef) -> bool {
    // A tagged immediate is always an exact `int` (never a subclass), so it
    // is a plain int without the `w_class` deref below. Gated on
    // `CAN_BE_TAGGED` (default false).
    if pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(obj) {
        return true;
    }
    // An int subclass instance carries `INT_USER_TYPE`, so the `INT_TYPE`
    // typeptr alone is `type(obj) is int`.
    unsafe { py_type_check(obj, &INT_TYPE) }
}

#[inline]
unsafe fn is_trace_plain_float(obj: PyObjectRef) -> bool {
    if !unsafe { py_type_check(obj, &FLOAT_TYPE) } {
        return false;
    }
    let float_typeobj = get_instantiate(&FLOAT_TYPE);
    if float_typeobj.is_null() {
        return unsafe { (*obj).w_class.is_null() };
    }
    let w_class = unsafe { (*obj).w_class };
    w_class.is_null() || std::ptr::eq(w_class, float_typeobj)
}

use crate::descr::{
    PY_OBJECT_ARRAY_GC_TYPE_ID, float_floatval_descr, int_intval_descr, make_array_descr_with_type,
    w_float_size_descr, w_int_size_descr,
};
use crate::frame_layout::{
    PYFRAME_DEBUGDATA_OFFSET, PYFRAME_LOCALS_CELLS_STACK_OFFSET, PYFRAME_PYCODE_OFFSET,
    PYFRAME_VALUESTACKDEPTH_OFFSET,
};
use crate::helpers::emit_box_float_inline;

// Re-export liveness items so downstream `pyre_jit_trace::state::*` keeps working.
pub use crate::liveness::{LiveVars, liveness_for};

/// Interpreter state exposed to the JIT framework.
///
/// Built from `PyFrame` before calling `back_edge`, and synced back
/// after compiled code runs.
/// Heap is the single source of truth (RPython parity).
/// next_instr / valuestackdepth live on the PyFrame heap object
/// and are accessed via read_frame_usize / write_frame_usize.
#[derive(majit_macros::VirtualizableState)]
pub struct PyreJitState {
    #[vable(frame)]
    pub frame: usize,
    /// PyPy portal's second red (`interp_jit.py` `reds = ['frame', 'ec']`).
    /// This belongs to the execution-context activation, not to `PyFrame`.
    pub execution_context: usize,
}

/// Meta information for a trace — describes the shape of the code being traced.
#[derive(Clone, majit_macros::VirtualizableMeta)]
pub struct PyreMeta {
    #[vable(num_locals)]
    pub num_locals: usize,
    pub ns_len: usize,
    pub namespace_dependent: bool,
    #[vable(valuestackdepth)]
    pub valuestackdepth: usize,
    /// Full `locals_cells_stack_w` length on the heap object
    /// (`co_nlocals + ncellvars + nfreevars + co_stacksize`).
    /// This stays separate from `valuestackdepth`, which is the live depth
    /// (`pyframe.py:111`) in the RPython model.
    pub array_capacity: usize,
    /// Extra portal reds between the frame red and the virtualizable
    /// payload. The Python portal carries `ec` here (`interp_jit.py`
    /// `reds = ['frame', 'ec']`). Guard-resume rebuilds the same
    /// contract in `rebuild_from_resumedata`.
    pub trace_extra_reds: usize,
    pub has_virtualizable: bool,
    #[vable(slot_types)]
    pub slot_types: Vec<Type>,
}

/// Symbolic state during tracing.
///
/// `frame` maps to a live IR `OpRef`. Symbolic frame field tracking
/// (locals, stack, valuestackdepth, next_instr) persists across instructions.
/// Locals and stack are virtualized (carried through JUMP args);
/// only next_instr and valuestackdepth are synced before guards / loop close.
#[derive(Clone, majit_macros::VirtualizableSym)]
pub struct PyreSym {
    /// OpRef for the owning PyFrame pointer.
    #[vable(frame)]
    pub frame: OpRef,
    /// PyPy portal second red: `ec` from `interp_jit.py:67`.
    ///
    /// This is not part of the virtualizable payload; it lives between the
    /// frame red and the virtualizable expansion when running under the
    /// canonical PyPy driver descriptor.
    pub(crate) execution_context: OpRef,
    // ── Persistent symbolic frame field tracking ──
    // The Python stack (`locals_cells_stack_w[nlocals..]`)
    // lives in the tail of `registers_r`. The macro's `collect_stack`
    // emits `registers_r[nlocals..nlocals + stack_only_depth]` so JUMP /
    // GUARD args carry locals followed by stack in one contiguous window.
    // This matches RPython's MIFrame register file (`pyjitpl.py`),
    // which treats locals and the stack as a single abstract register
    // vector.
    #[vable(local_types)]
    pub(crate) symbolic_local_types: Vec<Type>,
    #[vable(stack_types)]
    pub symbolic_stack_types: Vec<Type>,
    pub pending_next_instr: Option<usize>,
    pub(crate) locals_cells_stack_array_ref: OpRef,
    #[vable(valuestackdepth)]
    pub(crate) valuestackdepth: usize,
    #[vable(nlocals)]
    pub(crate) nlocals: usize,
    /// Bridge-specific override for the locals slice of registers_r.
    /// resume.py rebuild_from_resumedata parity: when set, init_symbolic uses these OpRefs
    /// (mapped from RebuiltValue::Box(n) in rebuild_from_resumedata) instead
    /// of the vable_array_base-based layout. This ensures bridge traces see
    /// frame locals as symbolic InputArgs, not concrete values.
    pub(crate) bridge_local_oprefs: crate::jitcode_dispatch::RegisterList,
    /// Bridge-specific override for the kept operand-stack slice of
    /// registers_r ([nlocals..nlocals+stack_only], semantic-slot == color
    /// in that prefix). resume.py rebuild_from_resumedata parity: setup_bridge_sym resolves
    /// the live operand-stack temps from the guard's resume data; this
    /// preserves them so init_symbolic (which runs AFTER setup_bridge_sym
    /// in pyre's bridge launcher) does not clobber the rebuilt stack tail
    /// back to NONE, and so the full-body-walk argbox seed can recover the
    /// kept conditional-expression / short-circuit value (#124).
    pub(crate) bridge_stack_oprefs: crate::jitcode_dispatch::RegisterList,
    /// Kept-stack branch-guard resume coordinate for the full-body walk: the
    /// guard's OWN jitcode byte offset (`frame0.jitcode_pc`, the mid-opcode
    /// `goto_if_not`), resolved the same way the blackhole resolves its
    /// `setposition` (`resolve_resume_pc_with_jitcode_pc`).  A chained-compare /
    /// short-circuit guard resumes mid-opcode: the opcode-entry marker
    /// for `py_pc` re-executes the whole comparison from the top, reading
    /// abstract-register colors that were live at entry but dead (recolored /
    /// consumed) at the guard — colors the guard snapshot never preserved.  The
    /// walk must resume where the blackhole does (the guard offset), so the
    /// re-executed suffix only reads colors the resume data actually carries.
    /// `None` for a non-branch-guard resume, where the walk
    /// keeps the opcode-entry offset for `py_pc`.
    pub(crate) bridge_walk_entry_pc: Option<usize>,
    /// JitCode body that owns `bridge_walk_entry_pc`. `resume.py:1049-1056`
    /// constructs the resumed frame and applies its PC from the same resume
    /// section; retain that body identity so the walker never interprets the
    /// offset in another installed body for this code object.
    pub(crate) bridge_walk_entry_jitcode_index: i32,
    /// The color-indexed Ref register bank as `consume_boxes`
    /// (resume.py) fills `f.registers_r` — one box per abstract
    /// register color the guard's resume numbering named. This is the
    /// authoritative `_get_list_of_active_boxes` (pyjitpl.py:216-233)
    /// source: it reads `registers_r[color]` by color at snapshot time.
    ///
    /// `sym.registers_r` diverges from this on the Ref bank: for the
    /// tracer's own LOAD_FAST/STORE_FAST reads pyre needs a SEMANTIC
    /// slot-indexed mirror (`[locals.., stack_tail..]`) because the
    /// per-CodeObject regalloc colors a slot at a color other than its
    /// slot index. `setup_bridge_sym` therefore overwrites
    /// `sym.registers_r` with that mirror and `init_symbolic` rebuilds it
    /// again from `bridge_local_oprefs`/`bridge_stack_oprefs`, so the
    /// color-indexed decode is lost from `sym.registers_r`. The int/float
    /// banks keep their color decode (init_symbolic rebuilds only the Ref
    /// bank), so only Ref needs this side field. A cross-frame bridge
    /// resume snapshot (`compute_bridge_root_parent_frame`) reads this to
    /// recover an operand live across a resumed call (e.g. `t1` in
    /// `return fib(n-1)+fib(n-2)`) that the semantic mirror does not carry
    /// by color.
    pub(crate) bridge_registers_r: crate::jitcode_dispatch::RegisterList,
    /// Bridge-specific override for symbolic_local_types.
    /// virtualizable.py:44 + interp_jit.py:25-30: locals_cells_stack_w[*]
    /// is a W_Root array → all items are Type::Ref. setup_bridge_sym
    /// populates this with all-Ref; downstream unboxing happens in
    /// opcode handlers via guard_class + getfield_gc_pure_i/_f, not at
    /// the virtualizable slot level.
    pub(crate) bridge_local_types: Option<Vec<Type>>,
    // virtualizable.py read_boxes: ALL static fields in declared order.
    // RPython's unroll_static_fields includes every field from
    // _virtualizable_; ALL must be inputarg (not info_only). The `type`
    // tag below mirrors pyframe.py's static-field declared types so the
    // macro mints `OpRef::input_arg_int/ref` matching the RPython
    // `InputArgInt`/`InputArgRef` class for that slot
    // (resoperation.py/739).
    #[vable(inputarg, type = int)]
    pub(crate) vable_last_instr: OpRef,
    #[vable(inputarg, type = ref)]
    pub(crate) vable_pycode: OpRef,
    #[vable(inputarg, type = int)]
    pub(crate) vable_valuestackdepth: OpRef,
    #[vable(inputarg, type = ref)]
    pub(crate) vable_debugdata: OpRef,
    /// Semantic namespace for this MIFrame. PyPy derives it from the frame's
    /// own pycode/debugdata; it is deliberately not a virtualizable scalar.
    pub(crate) frame_w_globals: OpRef,
    #[vable(array_base)]
    pub(crate) vable_array_base: Option<u32>,
    /// True when this frame's `locals_cells_stack_w` array IS the active
    /// virtualizable shadow for the current trace — i.e. when reads /
    /// writes against this frame's locals must consult / mutate
    /// `TraceCtx::virtualizable_boxes` (RPython `metainterp
    /// .virtualizable_boxes`, pyjitpl.py:1230) instead of going through
    /// regular MIFrame `registers_X`.
    ///
    /// Invariant: `is_active_vable_owner == vable_array_base.is_some()`.
    /// The boolean predicate (this field) and the u32 OpRef-offset (the
    /// `Option<u32>` value of `vable_array_base`) are split because their
    /// semantic roles differ: the predicate decides "consult vable
    /// shadow vs. registers", while the offset is only used as a
    /// fallback OpRef synthesizer at `trace_opcode.rs` when the
    /// metainterp-scope shadow is not yet populated. RPython has neither
    /// per-frame state — the codewriter emits `getarrayitem_vable`
    /// opcodes only on the toplevel frame's bytecode. Pyre dispatches
    /// the same `load_local_value` for all frames, so we encode the
    /// "active vable owner" identity per-frame.
    ///
    /// Helpers `become_active_vable_owner` / `clear_active_vable`
    /// maintain the invariant; do not write `vable_array_base` directly
    /// outside of those (or the macro-generated `init_vable_indices`
    /// invoked through `become_active_vable_owner`).
    pub(crate) is_active_vable_owner: bool,
    // ── MIFrame concrete Box tracking (RPython registers_i/r/f parity) ──
    // Concrete Python object values for locals and stack, tracked in
    // parallel with `registers_r`. Each opcode handler updates these
    // alongside the symbolic OpRefs so that guard decisions, branch
    // directions, and call results use internally tracked values
    // instead of reading from an external PyFrame snapshot.
    pub(crate) concrete_locals: Vec<ConcreteValue>,
    pub concrete_stack: Vec<ConcreteValue>,
    /// pyjitpl.py setup: frame.jitcode — JitCode reference.
    /// Provides both .code (CodeObject*) and .index (snapshot encoding).
    pub(crate) jitcode: *const JitCode,
    /// Namespace for global lookups (W_DictObject / W_ModuleDictObject).
    pub(crate) concrete_namespace: PyObjectRef,
    /// Execution context pointer (for creating callee frames).
    pub(crate) concrete_execution_context: *const pyre_interpreter::PyExecutionContext,
    /// Virtualizable object pointer (PyFrame).
    /// RPython MetaInterp stores the virtualizable separately from MIFrame.
    pub(crate) concrete_vable_ptr: *mut u8,
    /// Live (interpreter-owned) virtualizable `PyFrame` behind the tracing
    /// snapshot, or 0 when tracing runs without one (tests).
    /// `concrete_vable_ptr` points at the `snapshot_for_tracing` copy whose
    /// `debugdata` is an owned clone freed when the snapshot drops;
    /// vable-statics capture (`flush_to_frame`) reads that pointer-valued
    /// field from this frame so the trace's resume data never carries a
    /// snapshot-owned pointer. RPython has no snapshot —
    /// `read_boxes` (virtualizable.py) always reads the live
    /// virtualizable, which is what this field restores.
    pub(crate) live_vable_frame_addr: usize,
    /// Function-entry traces use typed locals (RPython MIFrame parity).
    pub(crate) is_function_entry_trace: bool,
    /// RPython MetaInterp.last_exc_value (pyjitpl.py): concrete
    /// exception object pending during tracing. Set by execute_ll_raised
    /// (raise_varargs), consumed by handle_possible_exception.
    pub(crate) last_exc_value: std::rc::Rc<std::cell::Cell<pyre_object::PyObjectRef>>,
    /// RPython MetaInterp.class_of_last_exc_is_const (pyjitpl.py):
    /// True after GUARD_EXCEPTION or GUARD_CLASS on the exception.
    pub(crate) class_of_last_exc_is_const: bool,
    /// RPython MetaInterp.last_exc_box (pyjitpl.py, 3386): symbolic
    /// OpRef for the exception value. Set directly by `opimpl_raise`, or
    /// by handle_possible_exception after GUARD_EXCEPTION, then consumed
    /// by finishframe_exception for stack push.
    pub(crate) last_exc_box: OpRef,
    /// Symbolic mirror of executioncontext.current_exception/sys_exc_info.
    /// Used by PUSH_EXC_INFO / POP_EXCEPT to preserve nested handler state.
    pub(crate) current_exc_value: std::rc::Rc<std::cell::Cell<pyre_object::PyObjectRef>>,
    pub(crate) current_exc_box: OpRef,
    // ── RPython MIFrame.registers_{i,r,f} port (pyjitpl.py) ──
    //
    // RPython reference (target shape):
    //   self.registers_i = [history.CONST_NULL] * jitcode.num_regs_and_consts_i()
    //   self.registers_r = [history.CONST_NULL] * jitcode.num_regs_and_consts_r()
    //   self.registers_f = [history.CONST_NULL] * jitcode.num_regs_and_consts_f()
    //
    // Each bank is sized to `num_regs_X + len(constants_X)` and indexed
    // by post-regalloc-color: `[0, num_regs_X)` are register slots
    // initialised to `CONST_NULL`, `[num_regs_X, ...)` are the constant
    // pool entries copied from `jitcode.constants_X`.
    //
    // SSA-authoritative live_r layout:
    //   - `setup_kind_register_banks` sizes all three banks to
    //     `num_regs_and_consts_X` when the owning JitCode is bound.
    //   - `registers_i` / `registers_f` are indexed by post-regalloc
    //     color. The encoder (`get_list_of_active_boxes`) reads them
    //     directly via the bank clone.
    //   - `registers_r` remains pyre's semantic frame mirror for
    //     `locals_cells_stack_w` because stack colors may coalesce with
    //     local colors. The encoder materializes a temporary
    //     color-indexed Ref-bank snapshot before reading liveness.
    pub(crate) registers_i: Vec<OpRef>,
    #[vable(locals)]
    pub(crate) registers_r: crate::jitcode_dispatch::RegisterList,
    pub(crate) registers_f: Vec<OpRef>,
}

pub trait WalkSym {
    fn frame(&self) -> OpRef;
    fn execution_context(&self) -> OpRef;
    fn set_execution_context(&mut self, value: OpRef);
    fn registers_i(&self) -> &[OpRef];
    fn registers_i_mut(&mut self) -> &mut Vec<OpRef>;
    fn registers_r(&self) -> &crate::jitcode_dispatch::RegisterList;
    fn registers_r_mut(&mut self) -> &crate::jitcode_dispatch::RegisterList;
    fn registers_f(&self) -> &[OpRef];
    fn registers_f_mut(&mut self) -> &mut Vec<OpRef>;
    fn nlocals(&self) -> usize;
    fn valuestackdepth(&self) -> usize;
    fn set_valuestackdepth(&mut self, value: usize);
    fn jitcode(&self) -> *const JitCode;
    fn concrete_execution_context(&self) -> *const pyre_interpreter::PyExecutionContext;
    /// Interpreter-frame snapshot the authoritative walker steps concretely.
    fn tracing_vable_frame_addr(&self) -> usize;
    fn live_vable_frame_addr(&self) -> usize;
    fn set_live_vable_frame_addr(&mut self, value: usize);
    fn bridge_walk_entry_pc(&self) -> Option<usize>;
    fn bridge_walk_entry_jitcode_index(&self) -> i32;
    fn bridge_registers_r(&self) -> Option<&crate::jitcode_dispatch::RegisterList>;
    fn bridge_registers_r_mut(&mut self) -> &crate::jitcode_dispatch::RegisterList;
    fn bridge_stack_oprefs(&self) -> Option<&crate::jitcode_dispatch::RegisterList>;
    fn bridge_stack_oprefs_mut(&mut self) -> &crate::jitcode_dispatch::RegisterList;
    fn bridge_local_oprefs_mut(&mut self) -> &crate::jitcode_dispatch::RegisterList;
    fn vable_array_base(&self) -> Option<u32>;
    fn vable_last_instr(&self) -> OpRef;
    fn vable_valuestackdepth(&self) -> OpRef;
    fn frame_w_globals(&self) -> OpRef;
    fn last_exc_value(&self) -> pyre_object::PyObjectRef;
    fn set_last_exc_value(&mut self, value: pyre_object::PyObjectRef);
    fn last_exc_box(&self) -> OpRef;
    fn set_last_exc_box(&mut self, value: OpRef);
    fn class_of_last_exc_is_const(&self) -> bool;
    fn set_class_of_last_exc_is_const(&mut self, value: bool);
    fn set_current_exc_value(&mut self, value: pyre_object::PyObjectRef);
    fn set_current_exc_box(&mut self, value: OpRef);
    fn owns_virtualizable_shadow(&self) -> bool;
    /// Publish independently owned tracing roots for the duration of this
    /// attempt, as translated roots keep the MIFrame and its lists alive.
    fn enter_trace_roots(&self) -> Option<crate::trace::TraceRoots> {
        None
    }
    fn init_symbolic(&mut self, ctx: &mut TraceCtx, concrete_frame: usize);
    fn close_loop_args_at(
        &mut self,
        ctx: &mut TraceCtx,
        concrete_frame: usize,
        orgpc: usize,
        target_pc: Option<usize>,
        header_marker_jit_pc: Option<usize>,
    ) -> Vec<OpRef>;
}

impl WalkSym for PyreSym {
    #[inline]
    fn frame(&self) -> OpRef {
        self.frame
    }

    #[inline]
    fn execution_context(&self) -> OpRef {
        self.execution_context
    }

    #[inline]
    fn set_execution_context(&mut self, value: OpRef) {
        self.execution_context = value;
    }

    #[inline]
    fn registers_i(&self) -> &[OpRef] {
        &self.registers_i
    }

    #[inline]
    fn registers_i_mut(&mut self) -> &mut Vec<OpRef> {
        &mut self.registers_i
    }

    #[inline]
    fn registers_r(&self) -> &crate::jitcode_dispatch::RegisterList {
        &self.registers_r
    }

    #[inline]
    fn registers_r_mut(&mut self) -> &crate::jitcode_dispatch::RegisterList {
        &self.registers_r
    }

    #[inline]
    fn registers_f(&self) -> &[OpRef] {
        &self.registers_f
    }

    #[inline]
    fn registers_f_mut(&mut self) -> &mut Vec<OpRef> {
        &mut self.registers_f
    }

    #[inline]
    fn nlocals(&self) -> usize {
        self.nlocals
    }

    #[inline]
    fn valuestackdepth(&self) -> usize {
        self.valuestackdepth
    }

    #[inline]
    fn set_valuestackdepth(&mut self, value: usize) {
        self.valuestackdepth = value;
    }

    #[inline]
    fn jitcode(&self) -> *const JitCode {
        self.jitcode
    }

    #[inline]
    fn concrete_execution_context(&self) -> *const pyre_interpreter::PyExecutionContext {
        self.concrete_execution_context
    }

    #[inline]
    fn tracing_vable_frame_addr(&self) -> usize {
        self.concrete_vable_ptr as usize
    }

    #[inline]
    fn live_vable_frame_addr(&self) -> usize {
        self.live_vable_frame_addr
    }

    #[inline]
    fn set_live_vable_frame_addr(&mut self, value: usize) {
        self.live_vable_frame_addr = value;
    }

    #[inline]
    fn bridge_walk_entry_pc(&self) -> Option<usize> {
        self.bridge_walk_entry_pc
    }

    #[inline]
    fn bridge_walk_entry_jitcode_index(&self) -> i32 {
        self.bridge_walk_entry_jitcode_index
    }

    #[inline]
    fn bridge_registers_r(&self) -> Option<&crate::jitcode_dispatch::RegisterList> {
        self.bridge_registers_r.as_ref()
    }

    #[inline]
    fn bridge_registers_r_mut(&mut self) -> &crate::jitcode_dispatch::RegisterList {
        &mut self.bridge_registers_r
    }

    #[inline]
    fn bridge_stack_oprefs(&self) -> Option<&crate::jitcode_dispatch::RegisterList> {
        self.bridge_stack_oprefs.as_ref()
    }

    #[inline]
    fn bridge_stack_oprefs_mut(&mut self) -> &crate::jitcode_dispatch::RegisterList {
        &mut self.bridge_stack_oprefs
    }

    #[inline]
    fn bridge_local_oprefs_mut(&mut self) -> &crate::jitcode_dispatch::RegisterList {
        &mut self.bridge_local_oprefs
    }

    #[inline]
    fn vable_array_base(&self) -> Option<u32> {
        self.vable_array_base
    }

    #[inline]
    fn vable_last_instr(&self) -> OpRef {
        self.vable_last_instr
    }

    #[inline]
    fn vable_valuestackdepth(&self) -> OpRef {
        self.vable_valuestackdepth
    }

    #[inline]
    fn frame_w_globals(&self) -> OpRef {
        self.frame_w_globals
    }

    #[inline]
    fn last_exc_value(&self) -> pyre_object::PyObjectRef {
        self.last_exc_value.get()
    }

    #[inline]
    fn set_last_exc_value(&mut self, value: pyre_object::PyObjectRef) {
        self.last_exc_value.set(value);
    }

    #[inline]
    fn last_exc_box(&self) -> OpRef {
        self.last_exc_box
    }

    #[inline]
    fn set_last_exc_box(&mut self, value: OpRef) {
        self.last_exc_box = value;
    }

    #[inline]
    fn class_of_last_exc_is_const(&self) -> bool {
        self.class_of_last_exc_is_const
    }

    #[inline]
    fn set_class_of_last_exc_is_const(&mut self, value: bool) {
        self.class_of_last_exc_is_const = value;
    }

    #[inline]
    fn set_current_exc_value(&mut self, value: pyre_object::PyObjectRef) {
        self.current_exc_value.set(value);
    }

    #[inline]
    fn set_current_exc_box(&mut self, value: OpRef) {
        self.current_exc_box = value;
    }

    #[inline]
    fn owns_virtualizable_shadow(&self) -> bool {
        PyreSym::owns_virtualizable_shadow(self)
    }

    #[inline]
    fn enter_trace_roots(&self) -> Option<crate::trace::TraceRoots> {
        Some(crate::trace::TraceRoots::enter(self))
    }

    #[inline]
    fn init_symbolic(&mut self, ctx: &mut TraceCtx, concrete_frame: usize) {
        PyreSym::init_symbolic(self, ctx, concrete_frame);
    }

    #[inline]
    fn close_loop_args_at(
        &mut self,
        ctx: &mut TraceCtx,
        concrete_frame: usize,
        orgpc: usize,
        target_pc: Option<usize>,
        header_marker_jit_pc: Option<usize>,
    ) -> Vec<OpRef> {
        let mut frame = MIFrame::from_sym(ctx, self, concrete_frame, orgpc, orgpc);
        frame.close_loop_args_at(ctx, target_pc, header_marker_jit_pc)
    }
}

#[doc(hidden)]
pub struct TestSymState {
    pub frame: OpRef,
    pub jitcode: *const (),
    pub nlocals: usize,
    pub valuestackdepth: usize,
    pub locals_cells_stack_array_ref: OpRef,
    pub symbolic_local_types: Vec<Type>,
    pub symbolic_stack_types: Vec<Type>,
    pub registers_r: Vec<OpRef>,
    pub concrete_stack: Vec<ConcreteValue>,
    pub concrete_namespace: PyObjectRef,
    pub vable_last_instr: OpRef,
    pub vable_pycode: OpRef,
    pub vable_valuestackdepth: OpRef,
    pub vable_debugdata: OpRef,
    pub frame_w_globals: OpRef,
}

/// Trace-time view over the virtualizable `PyFrame`.
///
/// Per-instruction wrapper that borrows persistent symbolic state from
/// `PyreSym` via raw pointer. The symbolic tracking (locals, stack,
/// valuestackdepth, next_instr) lives in PyreSym and survives across
/// instructions; this struct provides the per-instruction context
/// (ctx, fallthrough_pc).
pub struct MIFrame {
    pub(crate) ctx: *mut TraceCtx,
    pub(crate) sym: *mut PyreSym,
    pub(crate) fallthrough_pc: usize,
    /// Concrete PyFrame address for exception table lookup.
    pub(crate) concrete_frame_addr: usize,
    /// RPython pyjitpl.py orgpc parity: the PC at the START of the current
    /// opcode. All guards within one opcode capture this as their resume PC
    /// so that guard failure re-executes the opcode from the beginning.
    pub(crate) orgpc: usize,
    /// RPython `capture_resumedata(resumepc=orgpc)`
    /// Opcode-start snapshot of the unified `registers_r` file used by
    /// guard/resumedata capture for this one opcode. When `None`, guard
    /// capture reads the live register file directly. Shadow-owner snapshots
    /// are semantic frame prefixes sourced from `virtualizable_boxes`;
    /// non-owner snapshots clone the semantic `registers_r` mirror.
    pub(crate) pre_opcode_registers_r: Option<Vec<OpRef>>,
    /// Semantic frame prefix length at opcode start. This is distinct from
    /// `pre_opcode_registers_r.len()` for non-owner traces because the Ref
    /// bank may include post-regalloc color slots above the semantic
    /// locals+stack prefix.
    pub(crate) pre_opcode_semantic_depth: Option<usize>,
    /// `pyjitpl.py:181-193` `_result_argcode` analogue for non-top-frame
    /// snapshotting. When present, `get_list_of_active_boxes(in_a_call=True)`
    /// overwrites this caller stack slot with a zero/null placeholder before
    /// liveness encoding.
    pub pending_result_stack_idx: Option<usize>,
    pub pending_result_type: Option<Type>,
    pub pending_inline_frame: Option<PendingInlineFrame>,
    /// For an `in_a_call` parent frame snapshotted by
    /// `get_list_of_active_boxes`, the Python pc of the CALL whose
    /// post-residual-call `-live-`/`catch_exception` this frame must read
    /// liveness at (so encode and the marker-routed blackhole resume share
    /// the one `-live-` the way `pyjitpl.py pc=self.pc` does).
    /// `None` keeps the legacy `fallthrough_pc` liveness.
    pub residual_call_pc: Option<usize>,
    /// Resume-marker twin for the loop-close guards' snapshot word.
    /// `close_loop_args_at` sets and clears it around the back-edge
    /// eval-breaker poll and `GuardFutureCondition` emits; it is `None`
    /// outside that window.
    pub(crate) loop_close_marker_jit_pc: Option<usize>,
    /// Operand-stack depth of the merge point the closing JUMP targets,
    /// when it exceeds the depth the concrete `PyFrame` still advertises.
    /// `close_loop_args_at` sets and clears it around the JUMP-arg
    /// derivation; `concrete_valuestackdepth` returns it while set, so the
    /// flush and the arg derivation share one depth. `None` outside that
    /// window and whenever the concrete frame already covers the merge
    /// point — see `close_loop_args_at` for why a walk-driven close can
    /// read a depth belonging to a different bytecode offset.
    pub(crate) close_merge_point_vsd: Option<usize>,
}

#[allow(dead_code)]
pub(crate) fn instruction_consumes_comparison_truth(instruction: Instruction) -> bool {
    matches!(
        instruction,
        Instruction::PopJumpIfFalse { .. } | Instruction::PopJumpIfTrue { .. }
    )
}

#[allow(dead_code)]
pub(crate) fn instruction_is_trivia_between_compare_and_branch(instruction: Instruction) -> bool {
    matches!(
        instruction,
        Instruction::ExtendedArg
            | Instruction::Resume { .. }
            | Instruction::Nop
            | Instruction::Cache
            | Instruction::NotTaken
            | Instruction::ToBool
    )
}

#[allow(dead_code)]
pub(crate) fn instruction_needs_pre_opcode_snapshot(instruction: Instruction) -> bool {
    // Only keep the opcode-start snapshot for bytecodes that can emit a
    // guard after mutating the logical stack/register state. A larger
    // "may raise" set still needs GUARD_{NO_,}EXCEPTION handling, but
    // opcodes like GET_ITER / FOR_ITER / GET_LEN / IMPORT_FROM inspect
    // the current stack via peek-at-TOS and do not need a pre-pop
    // resumestate. Unsupported tracer opcodes also stay out of this set
    // until they gain a real guard-producing lowering.
    matches!(
        instruction,
        Instruction::Call { .. }
            | Instruction::CallKw { .. }
            | Instruction::StoreSubscr
            | Instruction::BinaryOp { .. }
            | Instruction::CompareOp { .. }
            | Instruction::UnaryNegative
            | Instruction::UnaryNot
            | Instruction::UnaryInvert
            | Instruction::RaiseVarargs { .. }
            // Both pop the operand stack (BUILD_TUPLE pop_n, UNPACK_SEQUENCE
            // pop_value) before emitting a guard: the specialised tuple path
            // emits w_class guards on the popped items, and the unpack path
            // emits sequence class / length guards on the popped sequence.
            // Without the opcode-start snapshot the guard's
            // resume state reflects the post-pop stack, so resuming at the
            // opcode start restores the consumed operands as null / mismatched.
            | Instruction::BuildTuple { .. }
            | Instruction::UnpackSequence { .. }
            // LOAD_ATTR can run load_method, which pop_value's the receiver
            // first, then emits a non-residual
            // receiver class guard (guard_class) plus a version_tag guard_value
            // — resume_pc=orgpc, so the consumed receiver must be in the
            // opcode-start snapshot.
            | Instruction::LoadAttr { .. }
            // The `list_append` hook pop_value's
            // the appended value, then records the generic `jit_list_append`
            // residual over the peeked list and popped value at
            // resume_pc=orgpc, so both consumed operands must be in the
            // opcode-start snapshot. The sibling comprehension helpers
            // (SET_ADD / MAP_ADD / LIST_EXTEND / …) have no tracer override
            // and abort before popping, so only LIST_APPEND qualifies.
            | Instruction::ListAppend { .. }
    )
}

/// RPython exc=True parity: instructions that correspond to JitCode ops
/// with exc=True. Only external calls and operations that invoke arbitrary
/// Python code need GUARD_NO_EXCEPTION. Arithmetic, comparisons, and
/// local variable access are lowered to primitive IR ops (exc=False) in
/// RPython and protected by type-specific guards instead.
#[allow(dead_code)]
pub(crate) fn instruction_may_raise(instruction: Instruction) -> bool {
    matches!(
        instruction,
        // RPython exc=True: external calls and attribute access that
        // may invoke arbitrary Python code (__getattr__, descriptors).
        // Unsupported tracer opcodes stay out of this set until they gain a
        // real lowering; otherwise the default "not implemented" error is
        // mis-recorded as a traced GUARD_EXCEPTION.
        Instruction::Call { .. }
            | Instruction::CallKw { .. }
            | Instruction::StoreAttr { .. }
            | Instruction::StoreSubscr
            | Instruction::ImportFrom { .. } // RPython raise/reraise are dedicated opimpls, not generic
                                             // exc=True execute_varargs sites. They unwind directly instead
                                             // of going through handle_possible_exception().
    )
}

/// Environment context — currently unused.
pub struct PyreEnv;

/// Descriptor for raw `PyObjectRef` item pointers that already address
/// `items[0]` (no length prefix to skip). Used after an explicit
/// `IntAdd(block, ITEMS_BLOCK_ITEMS_OFFSET)` converts a
/// `*mut ItemsBlock` block-base pointer into the items-base pointer.
/// `GETARRAYITEM_GC_R(ptr, i)` lands on `ptr + i * item_size`.
///
/// For callers that hold the block-base pointer directly (i.e.
/// `*mut ItemsBlock` for `W_ListObject.items` / tuple backing storage,
/// or `*mut FixedObjectArray` for `PyFrame.locals_cells_stack_w`), use
/// [`pyobject_gcarray_descr`] instead — its `base_size` is the block's
/// items offset, so the descriptor itself skips the length prefix.
#[allow(dead_code)]
pub(crate) fn pyobject_array_descr() -> DescrRef {
    // `nolength=True` shape (descr.py:359-360): items start at offset 0,
    // no length header — `GETARRAYITEM_GC_R(ptr, i)` lands on
    // `ptr + i * item_size`.  Stable identity carrier
    // `"pyre::pyobject_array_nolength"` so every `pyobject_array_descr()`
    // call returns the same Arc per PyPy `cpu.arraydescrof(ARRAY)`
    // singleton — `gc_cache._cache_array[LLType::Array(path_hash(...))]`
    // canonicalizes across analyzer / runtime consumers.
    crate::descr::make_array_descr_with_full_id(
        0,
        std::mem::size_of::<usize>(),
        0,
        None,
        Type::Ref,
        false,
        Some("pyre::pyobject_array_nolength".to_string()),
    )
}

/// Descriptor for RPython `Ptr(GcArray(PyObjectRef))` containers —
/// `[len][items...]` layout where the pointer addresses the length
/// header. Used by virtual array materialization (`NewArray` +
/// `SetarrayitemGc` in `decode_virtual_info`, resume.py:653-670) and
/// by the virtualizable-frame array field (`locals_cells_stack_w` via
/// the autogenerated `frame_locals_cells_stack_descr()` in
/// `virtualizable_gen.rs`). `base_size` is the block's items offset, so it
/// skips the length prefix and `GETARRAYITEM_GC_R(array_ptr, i)` lands on
/// items[i] directly.
pub fn pyobject_gcarray_descr() -> DescrRef {
    // type_id = PY_OBJECT_ARRAY_GC_TYPE_ID so the GC tracer walks each
    // item slot as a Ref. Without this, gen_initialize_tid stamps
    // OBJECT_GC_TYPE_ID into the GC header, the tracer treats the
    // allocation as a 16-byte PyObject, and the variable-part items
    // (the inline-emitted PyFrame.locals_cells_stack_w) survive the
    // first young collection without forwarding — manifesting as
    // wrong fib values on dynasm and SIGSEGV on cranelift once the
    // recursion depth fills the nursery (rgc parity:
    // gctypelayout.py T_IS_VARSIZE / T_IS_GCARRAY_OF_GCPTR).
    // Length-prefixed shape (descr.py:362): the array_ptr addresses the
    // length header, items start at the block's items offset. Both the
    // `ItemsBlock` list/tuple blocks and the `FixedObjectArray` frame-locals /
    // mro blocks carry this tid, and a `const` assertion in `object_array.rs`
    // holds their two tokens equal.
    let token = &pyre_object::ITEMS_BLOCK_TOKEN;
    let descr = make_array_descr_with_type(
        token.base_size,
        token.item_size,
        PY_OBJECT_ARRAY_GC_TYPE_ID,
        Some(token.len_offset),
        Type::Ref,
        false,
    );
    // Keep the mint above as `make_array_descr_with_type` (cache_key =
    // tid 9).  Frame locals and list items share this singleton; routing
    // the mint through `make_array_descr_with_full_id` let an analyzer
    // type_id=0 Arc win the slot and `fib_recursive` SIGSEGV'd.
    // Also publish the same Arc under the codewriter ARRAY identity so
    // `resolve_array_tid(path_hash(OBJECT_REF_GCARRAY_TYPE_ID))` finds
    // tid 9 (`cpu.arraydescrof(ARRAY)` / `descr.py get_array_descr`).
    // Force-insert: `register_keyed_array` is first-wins, and an earlier
    // analyzer mint under this key would otherwise keep a different tid.
    majit_ir::descr_registry::force_register_keyed_array(
        majit_ir::descr::LLType::Array(majit_ir::descr::path_hash(
            majit_jitcode::codewriter::jtransform::OBJECT_REF_GCARRAY_TYPE_ID,
        )),
        descr.clone(),
    );
    descr
}

/// The runtime's own descr for a list backing block or a `DICTENTRYARRAY`
/// named by its ARRAY identity.  The build-time descr pool consults this
/// before minting, so the GC tid stamped here reaches the shared
/// `_cache_array` slot first.
pub(crate) fn runtime_gcarray_descr(array_type_id: &str) -> Option<DescrRef> {
    use majit_jitcode::codewriter::jtransform as jt;
    let canonical = jt::canonical_array_type_id(array_type_id);
    match canonical.as_ref() {
        jt::LIST_INT_ITEMS_ARRAY => Some(int_gcarray_descr()),
        jt::LIST_FLOAT_ITEMS_ARRAY => Some(float_gcarray_descr()),
        jt::LIST_OBJ_ITEMS_ARRAY => Some(pyobject_gcarray_descr()),
        other => entries_gcarray_descr_for_type_id(other),
    }
}

fn entries_array_leaf(array_type_id: &str) -> &str {
    let inner = array_type_id
        .strip_prefix("GcArray<")
        .and_then(|s| s.strip_suffix('>'))
        .unwrap_or(array_type_id);
    inner.rsplit("::").next().unwrap_or(inner)
}

fn is_entry_kv(array_type_id: &str, key: &str, value_is_pyobject: bool) -> bool {
    let leaf = entries_array_leaf(array_type_id);
    if !leaf.starts_with("Entry<") {
        return false;
    }
    let has_key = leaf.contains(key);
    let has_pyobject = leaf.contains("PyObject") || leaf.contains("pyobject");
    has_key && has_pyobject == value_is_pyobject
}

/// `cpu.arraydescrof(ENTRIES)` for every `entries_gc_type_id!` pair.
fn entries_gcarray_descr_for_type_id(array_type_id: &str) -> Option<DescrRef> {
    type V = pyre_object::PyObjectRef;
    if is_entry_kv(array_type_id, "i64", true) {
        return Some(entries_gcarray_descr::<i64, V>(array_type_id));
    }
    if is_entry_kv(array_type_id, "ObjectKey", true) {
        return Some(entries_gcarray_descr::<
            pyre_object::dictmultiobject::ObjectKey,
            V,
        >(array_type_id));
    }
    if is_entry_kv(array_type_id, "BytesKey", true) {
        return Some(entries_gcarray_descr::<
            pyre_object::dictmultiobject::BytesKey,
            V,
        >(array_type_id));
    }
    if is_entry_kv(array_type_id, "IdentityKey", true) {
        return Some(entries_gcarray_descr::<
            pyre_object::identitydict::IdentityKey,
            V,
        >(array_type_id));
    }
    if is_entry_kv(array_type_id, "StrKey", true) {
        return Some(entries_gcarray_descr::<pyre_object::celldict::StrKey, V>(
            array_type_id,
        ));
    }
    if is_entry_kv(array_type_id, "ObjectKey", false) {
        return Some(entries_gcarray_descr::<
            pyre_object::dictmultiobject::ObjectKey,
            (),
        >(array_type_id));
    }
    None
}

/// `cpu.arraydescrof(ENTRIES)` for one `GcEntries<K, V>` ARRAY identity.
fn entries_gcarray_descr<K, V>(array_type_id: &str) -> DescrRef
where
    (K, V): pyre_object::rordereddict::GcEntriesType,
{
    use majit_ir::descr::{ArrayFlag, LLType};
    use majit_ir::{ArrayDescr, Descr};
    let base_size = std::mem::offset_of!(pyre_object::rordereddict::GcEntries<K, V>, items);
    let item_size = std::mem::size_of::<pyre_object::rordereddict::Entry<K, V>>();
    let len_offset = std::mem::offset_of!(pyre_object::rordereddict::GcEntries<K, V>, length);
    let tid = <(K, V) as pyre_object::rordereddict::GcEntriesType>::entries_gc_type_id();
    let key = LLType::Array(majit_ir::descr::path_hash(array_type_id));
    let descr = {
        let mut cache = majit_ir::descr::gc_cache().lock();
        cache.get_array_descr(
            key,
            base_size,
            item_size,
            ArrayFlag::Struct,
            Type::Ref,
            false,
            len_offset,
            false,
            '\x00',
        )
    };
    if !majit_ir::descr::array_tid_is_unresolved(tid) {
        if let Some(ad) = descr.as_array_descr() {
            if majit_ir::descr::array_tid_is_unresolved(ad.type_id()) || ad.type_id() != tid {
                ad.set_type_id(tid);
            }
        }
    }
    descr
}

/// `Ptr(GcArray(Signed))` — the `IntegerListStrategy` backing block
/// (`erase([int])`). Length-prefixed `[capacity][i64...]`: `base_size` skips
/// the capacity header so `GetarrayitemGcI(block, i)` lands on items[i], and
/// the op routes through the `(array, descr, index)` heap cache (CSE), unlike
/// the raw `int_array_descr`.
pub(crate) fn int_gcarray_descr() -> DescrRef {
    let token = &pyre_object::TYPED_ITEMS_BLOCK_INT_TOKEN;
    let live = pyre_object::gc_int_array_gc_type_id();
    let descr = crate::descr::make_array_descr_with_full_id(
        token.base_size,
        token.item_size,
        live,
        Some(token.len_offset),
        Type::Int,
        true,
        Some(majit_jitcode::codewriter::jtransform::LIST_INT_ITEMS_ARRAY.to_string()),
    );
    // Same publication as `pyobject_gcarray_descr`: `register_keyed_array`
    // is first-wins, and `opimpl_new_array` / `bh_new_array` allocate from
    // `BhDescr::resolve_gc_tid` *before* `dispatch_array_descr_ref` would
    // mint this descr. A cache miss truncates `path_hash(LIST_INT_ITEMS_ARRAY)`
    // into a fake tid (`resolved_gc_tid_checked`). Force-insert the live
    // collector tid under the codewriter ARRAY identity
    // (`GcLLDescr_framework.init_array_descr` / `cpu.arraydescrof`).
    if live != majit_rlib::lltypesystem::rlist::UNSET_GC_TYPE_ID
        && let Some(ad) = descr.as_array_descr()
        && ad.type_id() != live
    {
        ad.set_type_id(live);
    }
    majit_ir::descr_registry::force_register_keyed_array(
        majit_ir::descr::LLType::Array(majit_ir::descr::path_hash(
            majit_jitcode::codewriter::jtransform::LIST_INT_ITEMS_ARRAY,
        )),
        descr.clone(),
    );
    descr
}

/// `Ptr(GcArray(OBJECTPTR))` — `W_ObjectObject.storage`, the mapdict
/// attribute-value block (`mapdict.py _mapdict_init_empty` `self.storage`).
///
/// Structurally the same `ItemsBlock` as [`pyobject_gcarray_descr`], with its
/// own `W_MAPDICT_STORAGE_GC_TYPE_ID` and one property the allocation shape
/// turns on: `non_moving`. Raw `*mut ItemsBlock` copies outlive allocation
/// points in Rust locals and in `MapdictCarrier`, and the collector cannot
/// rewrite those, so the block must never move.
///
/// Every slot is a reference — a boxed attribute value, an unboxed attribute's
/// longlong GcArray, or NULL — so the tid is an ordinary array of pointers and
/// the barrier the GC rewrite emits for a `SetarrayitemGc` on this descr
/// records a real edge. That matches `alloc_mapdict_storage_block`
/// (`object_array.rs`), the allocator the interpreter itself uses.
pub(crate) fn mapdict_storage_gcarray_descr() -> DescrRef {
    let token = &pyre_object::ITEMS_BLOCK_TOKEN;
    let descr = crate::descr::make_array_descr_with_type(
        token.base_size,
        token.item_size,
        pyre_object::object_array::W_MAPDICT_STORAGE_GC_TYPE_ID,
        Some(token.len_offset),
        Type::Ref,
        false,
    );
    if let Some(simple) = descr
        .as_any()
        .and_then(|any| any.downcast_ref::<majit_ir::descr::SimpleArrayDescr>())
    {
        simple.set_non_moving(true);
    }
    descr
}

/// `Ptr(GcArray(Float))` — the `FloatListStrategy` backing block
/// (`erase([float])`). See [`int_gcarray_descr`].
pub(crate) fn float_gcarray_descr() -> DescrRef {
    let token = &pyre_object::TYPED_ITEMS_BLOCK_FLOAT_TOKEN;
    let live = pyre_object::gc_float_array_gc_type_id();
    let descr = crate::descr::make_array_descr_with_full_id(
        token.base_size,
        token.item_size,
        live,
        Some(token.len_offset),
        Type::Float,
        false,
        Some(majit_jitcode::codewriter::jtransform::LIST_FLOAT_ITEMS_ARRAY.to_string()),
    );
    if live != majit_rlib::lltypesystem::rlist::UNSET_GC_TYPE_ID
        && let Some(ad) = descr.as_array_descr()
        && ad.type_id() != live
    {
        ad.set_type_id(live);
    }
    majit_ir::descr_registry::force_register_keyed_array(
        majit_ir::descr::LLType::Array(majit_ir::descr::path_hash(
            majit_jitcode::codewriter::jtransform::LIST_FLOAT_ITEMS_ARRAY,
        )),
        descr.clone(),
    );
    descr
}

/// `GcLLDescr_framework.init_array_descr`: after the host has declared the
/// collector tids for `GcArray(Signed)` / `GcArray(Float)`, publish the
/// runtime ArrayDescrs under `LIST_INT_ITEMS_ARRAY` / `LIST_FLOAT_ITEMS_ARRAY`
/// so the first tracing `new_array` sees a real tid.
pub fn publish_typed_items_gcarray_descrs() {
    let _ = int_gcarray_descr();
    let _ = float_gcarray_descr();
}

/// `descr.py SizeDescr` for the host `PyFrame` virtualizable struct.
///
/// All `PyFrame` field descriptors point at this SizeDescr via
/// `FieldDescr.parent_descr` so the optimizer's `ensure_ptr_info_arg0`
/// (`optimizer.py`) can dispatch the GETFIELD/SETFIELD branch
/// to `InstancePtrInfo` / `StructPtrInfo`. Also handed to
/// `VirtualizableInfo::set_parent_descr` so virtualizable field
/// descriptors share the same parent.
pub fn pyframe_size_descr() -> DescrRef {
    crate::descr::pyframe_size_descr()
}

pub(crate) fn frame_locals_cells_stack_descr() -> DescrRef {
    crate::descr::pyframe_locals_cells_stack_descr()
}

pub(crate) fn wrapint(ctx: &mut TraceCtx, value: OpRef) -> OpRef {
    let boxed =
        crate::helpers::emit_box_int_inline(ctx, value, w_int_size_descr(), int_intval_descr());
    boxed
}

/// `blackhole_if_trace_too_long`'s warm-state bookkeeping (pyjitpl.py),
/// callable from the walker, which holds `&mut TraceCtx` and so cannot take
/// `MetaInterp` to run the method itself.
///
/// The two arms upstream keeps apart:
///
/// * `find_biggest_function()` named an inlined callee — the trace is long
///   because of THAT function, so it alone is disabled and the root is merely
///   asked to retrace (pyjitpl.py).  The root is NOT force-finished
///   and NOT marked dont-trace-here, and no bridge segmenting is enabled: the
///   next attempt is expected to fit once the callee stops being inlined.
/// * no callee named — `prepare_trace_segmenting` (pyjitpl.py): the
///   root itself is too big, so it gets all three warm-state marks and, for a
///   bridge, the source token gets `FORCE_BRIDGE_SEGMENTING`.
///
/// Both arguments are read straight off the walk's `TraceCtx`:
///
/// * `merge_key` — `if self.current_merge_points:` (pyjitpl.py, 2839). A
///   loop's outermost green key; `None` while tracing a bridge, which has no
///   merge point to boost.
/// * `source_token` — `if not isinstance(self.resumekey, ResumeFromInterpDescr):`
///   (pyjitpl.py). A bridge has no room in its `ResumeGuardDescr` for the
///   segmenting bit, so upstream sets it on the source `JitCellToken` instead,
///   where it applies to every bridge off that token. Marking the root green key
///   for a bridge would leave the real culprit untouched and let the same
///   oversized bridge re-record on each failure of that guard.
pub(crate) fn note_root_trace_too_long(
    merge_key: Option<majit_metainterp::PortalGreenKey>,
    source_token: Option<std::sync::Arc<majit_backend::JitCellToken>>,
) {
    // `try_driver_pair`, not `driver_pair`: a skeleton walk drives
    // `jitcode_dispatch` with no eval behind it, so there is no `MetaInterp` to
    // read `find_biggest_function` off or to mark.  It then names no huge
    // callee — the same answer the empty `portal_trace_positions` log gives —
    // and the token half below is reached, which is all a driverless walk could
    // ever do here.
    let huge_fn = crate::driver::try_driver_pair().and_then(|(driver, _)| {
        let meta = driver.meta_interp_mut();
        // pyjitpl.py `raise SwitchToBlackhole(ABORT_TOO_LONG)`: the caller's
        // DispatchError::TraceTooLong becomes TraceAction::SwitchToBlackhole.
        // The reason travels with the unwind, never in MetaInterp staging.
        // pyjitpl.py `jd_sd, greenkey_of_huge_function = self.find_biggest_function()`.
        let huge_fn = meta.find_biggest_function();
        // pyjitpl.py `self.portal_trace_positions = None` — the log's `_pos`
        // cursors index the recorder this abort is discarding.
        meta.retire_portal_trace_positions();
        if let Some((jd_no, huge_key)) = huge_fn.clone() {
            // pyjitpl.py `blackhole_if_trace_too_long`:
            // `jd_sd.warmstate.disable_noninlinable_function(
            // greenkey_of_huge_function)` disables through the warmstate of
            // the driver that owns the oversized frame (`huge_fn`'s `jd_no`).
            // Upstream's `dont_trace_here` (warmstate.py) is handed the greens
            // and reaches its cell through `ensure_jit_cell_at_key`, so take
            // the typed door when the log carried one — the hash form files a
            // comparekey-less cell that no later typed lookup can match.
            match huge_key.1.as_ref() {
                Some(key) => meta
                    .warm_state_for_driver(jd_no)
                    .disable_noninlinable_function_for_key(key),
                None => meta
                    .warm_state_for_driver(jd_no)
                    .disable_noninlinable_function(huge_key.0),
            }
            // pyjitpl.py, read by `aborted_tracing`'s `on_trace_abort`.
            meta.aborted_tracing_jitdriver = Some(jd_no);
            meta.aborted_tracing_greenkey = Some(huge_key);
            // pyjitpl.py `blackhole_if_trace_too_long`:
            // `warmrunnerstate = self.jitdriver_sd.warmstate` then
            // `JitCell.trace_next_iteration` on the outermost merge key.
            // Compiling `jitdriver_sd` is `active_jitdriver_sd.unwrap_or(0)`.
            // `trace_next_iteration` moves a counter and creates no cell
            // (warmstate.py `def trace_next_iteration_hash(hash)`), so
            // the hash is the whole identity it needs.
            if let Some((merge_key, _)) = merge_key.as_ref() {
                let compiling_jd = meta.active_jitdriver_sd.unwrap_or(0);
                meta.warm_state_for_driver(compiling_jd)
                    .trace_next_iteration(*merge_key);
            }
        } else if let Some((merge_key, merge_key_typed)) = merge_key.as_ref() {
            // pyjitpl.py `prepare_trace_segmenting`:
            // `warmrunnerstate = self.jitdriver_sd.warmstate` then
            // `JitCell.trace_next_iteration` / `mark_force_finish_tracing` /
            // `dont_trace_here` on the compiling `active_jitdriver_sd`.
            let compiling_jd = meta.active_jitdriver_sd.unwrap_or(0);
            let mut warm_state = meta.warm_state_for_driver(compiling_jd);
            // pyjitpl.py `warmrunnerstate.JitCell.trace_next_iteration(
            // greenkey)`, again counter-only.
            warm_state.trace_next_iteration(*merge_key);
            // pyjitpl.py `jd_sd.warmstate.mark_force_finish_tracing(
            // greenkey)`. `JC_FORCE_FINISH` is never cleared, so setting it on
            // the wrong cell of a bucket is permanent.
            match merge_key_typed.as_ref() {
                Some(key) => warm_state.mark_force_finish_tracing_for_key(key),
                None => warm_state.mark_force_finish_tracing(*merge_key),
            }
            // pyjitpl.py `warmstate.dont_trace_here(greenkey)`, the third
            // call of the same arm.  Without it the two above ask for the loop
            // to be re-traced and force-finished while its callers may still
            // inline it, so the next attempt can rebuild the very trace that
            // overflowed.
            match merge_key_typed.as_ref() {
                Some(key) => warm_state.disable_noninlinable_function_for_key(key),
                None => warm_state.disable_noninlinable_function(*merge_key),
            }
        }
        huge_fn
    });
    if huge_fn.is_none()
        && let Some(source_jct) = source_token.as_ref()
    {
        // pyjitpl.py:2857 `loop_token.retraced_count |= FORCE_BRIDGE_SEGMENTING`.
        let cur = source_jct.retraced_count.get();
        source_jct
            .retraced_count
            .set(cur | majit_backend::JitCellToken::FORCE_BRIDGE_SEGMENTING);
    }
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][trace-too-long] merge_key={merge_key:?} huge_fn={huge_fn:?} \
             bridge_segmenting={}",
            source_token.is_some() && huge_fn.is_none()
        );
    }
}

/// pyjitpl.py `_opimpl_recursive_call` → `perform_call` → `newframe` for a
/// callee the FBW walker is about to inline.
///
/// Upstream fills `portal_trace_positions` from `newframe`, reached through
/// `_opimpl_recursive_call`'s greenkey-bearing `perform_call`. The walker
/// inlines a Python callee by walking its body directly and never builds an
/// `MIFrame`, so that `newframe` never ran and `find_biggest_function` read
/// an empty log. This runs the same `newframe`; `newframe` itself appends
/// `(jitcode.jitdriver_sd, greenkey, self.history.get_trace_position())`
/// from `tracing.or(compile_tracing)`, the one `TraceCtx` the walker holds a
/// `&mut` into, so the walker does not stamp or append anything of its own.
///
/// Returns the owning driver when this is a recursive portal activation.
/// A normal exit pairs with [`note_inline_subwalk_end`]. `TraceTooLong` does
/// not: `popframe` would record `LEAVE_PORTAL_FRAME` after
/// `SwitchToBlackhole`, and the next trace's `initialize_state_from_start`
/// clears the framestack.
pub(crate) fn note_inline_subwalk_start(
    green_key: majit_metainterp::PortalGreenKey,
    portal_index: Option<usize>,
) -> Option<usize> {
    let (driver, _) = crate::driver::try_driver_pair()?;
    // `None` is the Python portal (driver 0). A recursive call passes the
    // target driver's main jitcode so `newframe` does not stamp driver 0.
    let index = match portal_index {
        Some(index) => index,
        None => crate::jitcode_runtime::portal_jitcode()?.index(),
    };
    let canonical = crate::jitcode_runtime::get_jitcode_by_index(index)?;
    let meta = driver.meta_interp_mut();
    if !meta.is_main_jitcode(&canonical) {
        return None;
    }
    let jd_no = canonical.jitdriver_sd()?;
    // pyjitpl.py `newframe(portal_code, greenkey)` — one owner for
    // portal_call_depth, call_ids, ENTER_PORTAL_FRAME, and the log.
    let jitcode = crate::jitcode_runtime::get_runtime_jitcode_by_index(index)?;
    meta.newframe(jitcode, Some(green_key));
    majit_metainterp::mc_diag_bump(58);
    Some(jd_no)
}

/// pyjitpl.py `popframe(leave_portal_frame=True)` — close the frame
/// [`note_inline_subwalk_start`] opened. `popframe` appends the closing
/// `(jitdriver_sd, None, get_trace_position())` row itself.
pub(crate) fn note_inline_subwalk_end() {
    let Some((driver, _)) = crate::driver::try_driver_pair() else {
        return;
    };
    majit_metainterp::mc_diag_bump(59);
    driver.meta_interp_mut().popframe(true);
}

/// Stage `reason` as the abort the walker is returning, so the single
/// `aborted_tracing` that follows counts it under that counter instead of the
/// `Generic` catch-all.
///
/// Naming the reason is pure accounting: the abort itself travels in the
/// `DispatchError` the caller returns and does not consult this slot. A
/// skeleton walk has no driver to stage onto, so it skips the note instead of
/// panicking in `callbacks::get`.
fn stage_walker_abort_reason(reason: i32) {
    let Some((driver, _)) = crate::driver::try_driver_pair() else {
        return;
    };
    driver.meta_interp_mut().stage_abort_reason(reason);
}

/// Name `Counters.ABORT_FORCE_QUASIIMMUT` as the reason for the abort the
/// walker is returning (`pyjitpl.py`), so the single `aborted_tracing`
/// that follows counts it under `profiler.abort_force_quasiimmut` instead of
/// the `Generic` catch-all.
pub(crate) fn note_force_quasi_immut_abort() {
    stage_walker_abort_reason(majit_metainterp::counters::ABORT_FORCE_QUASIIMMUT);
}

/// Name `Counters.ABORT_ESCAPE` as the reason for the abort the walker returns
/// when `tracing_after_residual_call` reports the virtualizable was forced
/// (`pyjitpl.py:3389-3390` raises `SwitchToBlackhole(Counters.ABORT_ESCAPE)`
/// there).  Without this the reason ladder falls through to
/// `AbortReason::Generic`, whose `as_int()` is `ABORT_BRIDGE` — so the escape
/// lands in the bucket `jitprof.rs ABORT_COUNTER_KINDS` labels
/// `bridge_or_generic`, and the tally reads as bridge activity that never
/// happened.
///
/// `raising_exception` does not travel this channel: pyre computes it from the
/// residual's own `exec_result` (`residual_call.rs`), where the escape's `Err`
/// arm already sets it, so upstream's hardcoded `True` needs no counterpart
/// here.
pub(crate) fn note_vable_escape_abort() {
    stage_walker_abort_reason(majit_metainterp::counters::ABORT_ESCAPE);
}

pub(crate) fn wrapfloat(ctx: &mut TraceCtx, value: OpRef) -> OpRef {
    emit_box_float_inline(ctx, value, w_float_size_descr(), float_floatval_descr())
}

// RPython parity note: pyjitpl.py (tracer) records GETFIELD_GC ops WITHOUT
// any constant folding. Folding happens exclusively in the optimizer's
// `optimize_GETFIELD_GC_I` (heap.py), which delegates to
// `optimizer.constant_fold(op)` → `_execute_arglist` → `do_getfield_gc_*`.
// pyre's `OptContext::constant_fold` in optimizeopt/mod.rs is the exact
// port of that path — it handles Int/Float/Ref via `execute_nonspec_const`
// dispatched on `field_type()` and `field_size()`.
//
// The previous tracer-level `try_trace_const_pure_int_field` helper was a
// pyre-specific pre-optimization that duplicated (and mistyped) the
// optimizer logic. It has been removed for structural parity with RPython.

#[allow(dead_code)]
pub(crate) fn try_trace_const_boxed_int(
    ctx: &mut TraceCtx,
    value: OpRef,
    concrete_value: PyObjectRef,
) -> Option<OpRef> {
    if ctx.const_value(value) != Some(concrete_value as i64) {
        return None;
    }
    unsafe {
        if is_bool(concrete_value) {
            return Some(ctx.const_int(if w_bool_get_value(concrete_value) {
                1
            } else {
                0
            }));
        }
        if is_trace_plain_int(concrete_value) {
            return Some(ctx.const_int(w_int_get_value(concrete_value)));
        }
    }
    None
}

/// pyjitpl.py:750-758: read container length.
///
/// RPython's `arraylen_gc` reads the GC array header — there is exactly one
/// length per array, so RPython keeps a per-box `heapc_deps[0]` slot. pyre
/// stores list/bytes/tuple lengths as plain struct fields, so the cached
/// value lives in `HeapCache.heap_cache[descr] -> CacheEntry`
/// (heapcache.py).  `opimpl_getfield_gc_i` already does that lookup,
/// so this helper is now just a thin alias kept for source-stability
/// with the call sites.
#[allow(dead_code)]
pub(crate) fn trace_arraylen_gc(ctx: &mut TraceCtx, obj: OpRef, descr: DescrRef) -> OpRef {
    opimpl_getfield_gc_i(ctx, obj, descr)
}

/// pyjitpl.py `opimpl_arraylen_gc`. Emits the actual
/// `ArraylenGc` op against the GcArray header (rlist.py:251
/// `len(l.items)` reads the array's length-prefix). Caller is the
/// items_block Ref (output of `opimpl_getfield_gc_r` on
/// `tuple_wrappeditems_descr` / `list_items_descr`); `descr` is the
/// matching array descr (e.g. `pyobject_gcarray_descr`).
///
/// Distinct from `trace_arraylen_gc` above — that helper reads
/// pyre-specific length FIELDS off the host wrapper struct via
/// getfield (TODO: bring to parity) and is reserved for callers
/// that still go through that path (`str_len`, `dict_len`,
/// `list_length`, typed list `int_items.len`).
pub(crate) fn opimpl_arraylen_gc(ctx: &mut TraceCtx, array: OpRef, descr: DescrRef) -> OpRef {
    if let Some(cached) = ctx.heap_cache().arraylen(array) {
        // pyjitpl.py opimpl_getinteriorfield_gc_i `_opimpl_arraylen_gc` cache hit:
        //     if length is not None:
        //         self.metainterp.staticdata.profiler.count_ops(rop.ARRAYLEN_GC, Counters.HEAPCACHED_OPS)
        //         return lengthbox
        ctx.profiler().count_ops(
            OpCode::ArraylenGc,
            majit_metainterp::counters::HEAPCACHED_OPS,
        );
        return cached;
    }
    // The cache miss routes through `execute_and_record`, which counts
    // the base OPS bucket at entry — before the `_all_constants` fold
    // check — so the fold and record paths below both count OPS.
    ctx.profiler()
        .count_ops(OpCode::ArraylenGc, majit_metainterp::counters::OPS);
    // ARRAYLEN_GC is ALWAYS_PURE, so `execute_and_record` folds a `Const`
    // array operand to the loaded length without recording
    // (`_all_constants` — a GcArray block's length is fixed at
    // allocation).  The folded box still feeds `arraylen_now_known`,
    // matching `opimpl_arraylen_gc`'s unconditional cache write.
    //
    if array.is_constant() {
        if let Some(majit_ir::Value::Ref(struct_ref)) = ctx.box_value(array) {
            let struct_ptr = struct_ref.0 as i64;
            if struct_ptr != 0 && struct_ptr != usize::MAX as i64 {
                if let Some(majit_ir::Value::Int(len)) =
                    ctx.arraylen_sanity_load(struct_ptr, &descr)
                {
                    let folded = ctx.const_int(len);
                    ctx.heap_cache_mut().arraylen_now_known(array, folded);
                    return folded;
                }
            }
        }
    }
    // `_record_helper` adds RECORDED_OPS exactly when the op is recorded.
    ctx.profiler()
        .count_ops(OpCode::ArraylenGc, majit_metainterp::counters::RECORDED_OPS);
    let result = ctx.record_op_with_descr(OpCode::ArraylenGc, &[array], descr.clone());
    // Box(value) parity: executor.py:188 do_arraylen_gc returns
    // BoxInt(cpu.bh_arraylen_gc(array, arraydescr)). Stamp the result
    // OpRef with the live length so downstream box_value(result) sees
    // the same value (matches RPython BoxInt(length) carrier).
    if let Some(majit_ir::Value::Ref(struct_ref)) = ctx.box_value(array) {
        let struct_ptr = struct_ref.0 as i64;
        if struct_ptr != usize::MAX as i64 && struct_ptr != 0 {
            if let Some(live) = ctx.arraylen_sanity_load(struct_ptr, &descr) {
                ctx.set_opref_concrete(result, live);
            }
        }
    }
    ctx.heap_cache_mut().arraylen_now_known(array, result);
    result
}

pub(crate) fn opimpl_getfield_gc_i(ctx: &mut TraceCtx, obj: OpRef, descr: DescrRef) -> OpRef {
    // `_opimpl_getfield_gc_any_pureornot`: a `ConstPtr` receiver through an
    // always-pure descr bypasses the heapcache completely — execute the
    // load now and substitute the value as a `Const`, recording no op (the
    // jitcode-dispatch twin `getfield_gc_via_heapcache` carries the same
    // fast path).  For a non-pure descr the tracer only records; folding a
    // constant object's mutable field read is the optimizer's job
    // (heap.py:optimize_GETFIELD_GC_I → optimizer.constant_fold, ported in
    // OptContext::execute_nonspec_const).
    if obj.is_constant() && descr.is_always_pure() {
        if let Some(majit_ir::Value::Ref(struct_ref)) = ctx.box_value(obj) {
            let struct_ptr = struct_ref.0 as i64;
            if struct_ptr != 0 && struct_ptr != usize::MAX as i64 {
                if let Some(majit_ir::Value::Int(n)) =
                    ctx.field_sanity_load(struct_ptr, &descr, majit_ir::Type::Int)
                {
                    return ctx.const_int(n);
                }
            }
        }
    }
    // heapcache.py: check if this field was already read/written in this trace
    let field_index = descr.index();
    let field_name = descr_field_name(&descr);
    if let Some(cached) = ctx.heapcache_getfield_cached(obj, field_index) {
        // pyjitpl.py:934-945 cache-hit sanity check (int arm). The
        // line-by-line port runs `executor.execute(cpu, mi, opnum,
        // fielddescr, box)` and asserts `resvalue ==
        // upd.currfieldbox.getint()`.  The cached Box's intrinsic
        // value is fetched via `box_value(cached)` — covering const
        // pool, standard-virtualizable shadow, and the frontend
        // object's `value` field (RPython `currfieldbox.getint()`
        // dispatch parity).
        let expected_int = match ctx.box_value(cached) {
            Some(majit_ir::Value::Int(n)) => Some(n),
            _ => None,
        };
        if let Some(cached_int) = expected_int {
            if let Some(struct_ptr) = concrete_gc_ptr(ctx, obj) {
                if let Some(majit_ir::Value::Int(loaded)) =
                    ctx.field_sanity_load(struct_ptr, &descr, majit_ir::Type::Int)
                {
                    if loaded != cached_int && crate::probe_subscr_enabled() {
                        eprintln!(
                            "[MAJIT_PROBE_SUBSCR] sanity-int-mismatch obj={:?} field_index={} struct_ptr={:#x} descr_pure={} cached={} loaded={}",
                            obj,
                            field_index,
                            struct_ptr,
                            descr.is_always_pure(),
                            cached_int,
                            loaded
                        );
                    }
                    assert_eq!(
                        loaded, cached_int,
                        "_opimpl_getfield_gc_any_pureornot sanity \
                             check (int): loaded {loaded} != cached \
                             {cached_int} (field_index={field_index}, \
                             field={field_name}, struct_ptr={struct_ptr:#x})"
                    );
                }
            }
        }
        // pyjitpl.py:946 cache-hit accounting:
        //   self.metainterp.staticdata.profiler.count_ops(rop.GETFIELD_GC_I, Counters.HEAPCACHED_OPS)
        //   return upd.currfieldbox
        ctx.profiler().count_ops(
            OpCode::GetfieldGcI,
            majit_metainterp::counters::HEAPCACHED_OPS,
        );
        return cached;
    }
    // pyjitpl.py opimpl_record_quasiimmut_field: quasi-immutable field handling.
    // Record the field as quasi-immut known so subsequent reads skip
    // the QUASIIMMUT_FIELD op. Emit GUARD_NOT_INVALIDATED if needed.
    // NOTE: GuardNotInvalidated is NOT emitted here — it requires
    // PyreSym.generate_guard for proper snapshot/fail_args (pyjitpl.py
    // generate_guard parity). Instead, set a flag on ctx so the caller
    // (PyreSym with_ctx block) can emit it with full resume data.
    if descr.is_quasi_immutable() {
        if ctx.heap_cache().is_quasi_immut_known(field_index, obj) {
            // pyjitpl.py:1077-1080 cache hit:
            //   if heapcache.is_quasi_immut_known(fielddescr, box):
            //       profiler.count_ops(rop.QUASIIMMUT_FIELD, HEAPCACHED_OPS)
            //       return
            ctx.profiler().count_ops(
                OpCode::QuasiimmutField,
                majit_metainterp::counters::HEAPCACHED_OPS,
            );
        } else {
            let qmutdescr = quasi_immut_descr(ctx, obj, &descr);
            ctx.heap_cache_mut().quasi_immut_now_known(field_index, obj);
            ctx.record_op_with_descr(
                OpCode::QuasiimmutField,
                &[obj],
                qmutdescr.unwrap_or_else(|| descr.clone()),
            );
            if ctx.heap_cache_mut().check_and_clear_guard_not_invalidated() {
                ctx.set_pending_guard_not_invalidated(Some(ctx.last_traced_pc));
            }
        }
    }
    // `opimpl_getfield_gc_i` records the plain GETFIELD_GC_I opnum
    // regardless of `descr.is_always_pure()`; purity is re-derived from
    // the descr by OptHeap (const fold + invalidation-exempt field
    // cache). There is no pure getfield opnum.
    ctx.profiler()
        .count_ops(OpCode::GetfieldGcI, majit_metainterp::counters::OPS);
    ctx.profiler().count_ops(
        OpCode::GetfieldGcI,
        majit_metainterp::counters::RECORDED_OPS,
    );
    let result = ctx.record_op_with_descr(OpCode::GetfieldGcI, &[obj], descr.clone());
    // pyjitpl.py `resbox = execute_with_descr(...); upd.getfield_now_known(resbox)`.
    // `resbox` carries the loaded value; pair the recorded opref with
    // the live int from `field_sanity_load` so subsequent
    // `box_value(result)` mirrors RPython's executor-returned Box.
    let live_value = concrete_gc_ptr(ctx, obj)
        .and_then(|struct_ptr| ctx.field_sanity_load(struct_ptr, &descr, majit_ir::Type::Int));
    if let Some(live_value) = live_value {
        ctx.set_opref_concrete(result, live_value);
    }
    ctx.heapcache_getfield_now_known(obj, field_index, result);
    result
}

/// pyjitpl.py `opimpl_getfield_gc_r`. Same shape as `_i`
/// modulo the rop variant: the `ConstPtr` + always-pure bypass folds to
/// a `Const` without recording, and for non-pure descrs folding lives in
/// the optimizer (`optimize_GETFIELD_GC_R = optimize_GETFIELD_GC_I` per
/// RPython's alias).
pub(crate) fn opimpl_getfield_gc_r(ctx: &mut TraceCtx, obj: OpRef, descr: DescrRef) -> OpRef {
    if obj.is_constant() && descr.is_always_pure() {
        if let Some(majit_ir::Value::Ref(struct_ref)) = ctx.box_value(obj) {
            let struct_ptr = struct_ref.0 as i64;
            if struct_ptr != 0 && struct_ptr != usize::MAX as i64 {
                if let Some(majit_ir::Value::Ref(loaded)) =
                    ctx.field_sanity_load(struct_ptr, &descr, majit_ir::Type::Ref)
                {
                    return ctx.const_ref(loaded.0 as i64);
                }
            }
        }
    }
    let field_index = descr.index();
    let field_name = descr_field_name(&descr);
    if let Some(cached) = ctx.heapcache_getfield_cached(obj, field_index) {
        // `_opimpl_getfield_gc_any_pureornot` cache-hit sanity check (ref arm).
        // `box_value(cached)` resolves the upstream
        // `currfieldbox.getref_base()` payload through the full chain
        // (const pool, standard-virtualizable shadow, the frontend
        // object's `value` field).
        let expected_ref = match ctx.box_value(cached) {
            Some(majit_ir::Value::Ref(r)) => Some(r),
            _ => None,
        };
        if let Some(cached_ref) = expected_ref {
            if cached_ref != majit_ir::GcRef::NO_CONCRETE {
                if let Some(struct_ptr) = concrete_gc_ptr(ctx, obj) {
                    let struct_ptr =
                        majit_gc::gc_current_object_address(struct_ptr as usize) as i64;
                    if let Some(majit_ir::Value::Ref(loaded)) =
                        ctx.field_sanity_load(struct_ptr, &descr, majit_ir::Type::Ref)
                    {
                        // `getref_base()` is a GC pointer; a nursery
                        // collection forwards it. The heapcache word is
                        // a raw copy and is not rewritten, so compare
                        // after `gc_current_object_address`.
                        let loaded_now =
                            majit_ir::GcRef(majit_gc::gc_current_object_address(loaded.0));
                        let cached_now =
                            majit_ir::GcRef(majit_gc::gc_current_object_address(cached_ref.0));
                        assert_eq!(
                            loaded_now, cached_now,
                            "_opimpl_getfield_gc_any_pureornot sanity \
                             check (ref): loaded {:#x} != cached {:#x} \
                             (field_index={field_index}, field={field_name}, struct_ptr=\
                             {struct_ptr:#x})",
                            loaded_now.0, cached_now.0,
                        );
                    }
                }
            }
        }
        // `_opimpl_getfield_gc_any_pureornot` hardcodes `GETFIELD_GC_I` regardless
        // of the rop variant (`_i` / `_r` / `_f`); pyre matches.
        ctx.profiler().count_ops(
            OpCode::GetfieldGcI,
            majit_metainterp::counters::HEAPCACHED_OPS,
        );
        return cached;
    }
    if descr.is_quasi_immutable() {
        if ctx.heap_cache().is_quasi_immut_known(field_index, obj) {
            // pyjitpl.py cache hit (see opimpl_getfield_gc_i above).
            ctx.profiler().count_ops(
                OpCode::QuasiimmutField,
                majit_metainterp::counters::HEAPCACHED_OPS,
            );
        } else {
            let qmutdescr = quasi_immut_descr(ctx, obj, &descr);
            ctx.heap_cache_mut().quasi_immut_now_known(field_index, obj);
            ctx.record_op_with_descr(
                OpCode::QuasiimmutField,
                &[obj],
                qmutdescr.unwrap_or_else(|| descr.clone()),
            );
            if ctx.heap_cache_mut().check_and_clear_guard_not_invalidated() {
                ctx.set_pending_guard_not_invalidated(Some(ctx.last_traced_pc));
            }
        }
    }
    // `opimpl_getfield_gc_r` records the plain GETFIELD_GC_R opnum
    // regardless of `descr.is_always_pure()`; purity is re-derived from
    // the descr by OptHeap. There is no pure getfield opnum.
    ctx.profiler()
        .count_ops(OpCode::GetfieldGcR, majit_metainterp::counters::OPS);
    ctx.profiler().count_ops(
        OpCode::GetfieldGcR,
        majit_metainterp::counters::RECORDED_OPS,
    );
    let result = ctx.record_op_with_descr(OpCode::GetfieldGcR, &[obj], descr.clone());
    // pyjitpl.py `resbox = execute_with_descr(...); upd.getfield_now_known(resbox)`.
    // Pair the recorded opref with the live ref so subsequent
    // `box_value(result)` mirrors RPython's executor-returned Box.
    let live_value = concrete_gc_ptr(ctx, obj)
        .and_then(|struct_ptr| ctx.field_sanity_load(struct_ptr, &descr, majit_ir::Type::Ref));
    if let Some(live_value) = live_value {
        ctx.set_opref_concrete(result, live_value);
    }
    ctx.heapcache_getfield_now_known(obj, field_index, result);
    result
}

fn descr_field_name(descr: &DescrRef) -> &str {
    descr
        .as_field_descr()
        .map(|field| field.field_name())
        .filter(|name| !name.is_empty())
        .unwrap_or("<unnamed>")
}

fn concrete_gc_ptr(ctx: &TraceCtx, obj: OpRef) -> Option<i64> {
    let Some(majit_ir::Value::Ref(struct_ref)) = ctx.box_value(obj) else {
        return None;
    };
    if struct_ref == majit_ir::GcRef::NO_CONCRETE
        || struct_ref.as_usize() == 0
        || struct_ref.as_usize() == usize::MAX
    {
        return None;
    }
    Some(struct_ref.0 as i64)
}

// Note: pyre does not currently route GetfieldGcF through
// state.rs. Float field unboxing goes via the codewriter-generated
// `getfield_gc_f_pureornot` (majit-translate/src/codegen.rs),
// which — matching RPython's pyjitpl.py opimpl_getfield_gc_f — records
// the GC op without folding. The optimizer's `optimize_GETFIELD_GC_F`
// (= `optimize_GETFIELD_GC_I` via RPython's alias) handles folding.

/// Unbox int with proper GuardClass resume data via the frame impl's
/// `generate_guard`. Generic over `WalkerFrameOps` so both `MIFrame` and
/// `WalkContext` can invoke the same lowering.
#[allow(dead_code)]
pub(crate) fn trace_unbox_int_with_resume<F: crate::walker_frame_ops::WalkerFrameOps>(
    frame: &mut F,
    obj: OpRef,
    int_type_addr: i64,
) -> OpRef {
    trace_unbox_int_with_resume_descr(frame, obj, int_type_addr, crate::descr::int_intval_descr())
}

/// `(guard_class type addr, intval descr)` for unboxing a concrete int- or
/// bool-valued operand. `W_BoolObject` is a `W_IntObject` subclass sharing
/// the `intval` field, so a bool unboxes through the same `getfield` but
/// guards its own `&BOOL_TYPE` vtable and keys the heapcache by the bool
/// field descr.
pub(crate) fn int_or_bool_unbox_type_descr(
    concrete: pyre_object::PyObjectRef,
) -> (i64, majit_ir::DescrRef) {
    if pyre_object::tagged_int::CAN_BE_TAGGED
        && !concrete.is_null()
        && pyre_object::tagged_int::is_tagged_int(concrete)
    {
        return (
            &pyre_object::pyobject::INT_TYPE as *const _ as i64,
            crate::descr::int_intval_descr(),
        );
    }
    if !concrete.is_null() && unsafe { pyre_object::is_bool(concrete) } {
        (
            &pyre_object::pyobject::BOOL_TYPE as *const _ as i64,
            crate::descr::bool_intval_descr(),
        )
    } else {
        (
            &pyre_object::pyobject::INT_TYPE as *const _ as i64,
            crate::descr::int_intval_descr(),
        )
    }
}

#[allow(dead_code)]
pub(crate) fn trace_unbox_int_with_resume_descr<F: crate::walker_frame_ops::WalkerFrameOps>(
    frame: &mut F,
    obj: OpRef,
    type_addr: i64,
    intval_descr: majit_ir::DescrRef,
) -> OpRef {
    if pyre_object::tagged_int::CAN_BE_TAGGED {
        if let Some(majit_ir::Value::Ref(r)) = frame.ctx().concrete_of_opref(obj) {
            if r != majit_ir::GcRef::NO_CONCRETE {
                let o = r.as_usize() as pyre_object::PyObjectRef;
                if !o.is_null() {
                    let tagged = pyre_object::tagged_int::is_tagged_int(o);
                    // rtagged.py ll_unboxed_getclass tag discrimination:
                    // IntAnd(CastPtrToInt(obj), 1), guarded GUARD_TRUE (tagged
                    // leg) / GUARD_FALSE (boxed leg). Both the lowbit test and
                    // its guard derive purely from `obj`; when `obj` is a Const
                    // the metainterp folds the arithmetic to a ConstInt and the
                    // guard is elided (pyjitpl.py generate_guard "if
                    // isinstance(box, Const): return"). Emit the runtime tag
                    // discrimination only for a non-constant operand — a
                    // constant tag is statically known.
                    if !obj.is_constant() {
                        let lowbit =
                            crate::helpers::emit_tag_lowbit_test(frame.ctx_mut(), obj, tagged);
                        let guard = if tagged {
                            OpCode::GuardTrue
                        } else {
                            OpCode::GuardFalse
                        };
                        frame.generate_guard(guard, &[lowbit]);
                    }
                    if tagged {
                        // rtagged.py ll_unboxed_to_int: arithmetic
                        // IntRshift(CastPtrToInt(obj), 1), no guard.
                        return crate::helpers::emit_untag_int(
                            frame.ctx_mut(),
                            obj,
                            pyre_object::tagged_int::untag_int(o),
                        );
                    }
                }
            }
        }
    }
    // pyjitpl.py GUARD_CLASS(box, cls): guard takes object box directly,
    // backend loads typeptr at offset 0.
    if !frame.ctx().heap_cache().is_class_known(obj) {
        let type_const = frame.ctx_mut().const_int(type_addr);
        frame.generate_guard(OpCode::GuardClass, &[obj, type_const]);
        frame.ctx_mut().heap_cache_mut().class_now_known(obj);
    }
    crate::trace_unbox_int(frame.ctx_mut(), obj, type_addr, intval_descr)
}

#[allow(dead_code)]
pub(crate) unsafe fn objspace_compare_ints(
    lhs_obj: PyObjectRef,
    rhs_obj: PyObjectRef,
    op: ComparisonOperator,
) -> bool {
    unsafe {
        let lhs = w_int_get_value(lhs_obj);
        let rhs = w_int_get_value(rhs_obj);
        match op {
            ComparisonOperator::Less => lhs < rhs,
            ComparisonOperator::LessOrEqual => lhs <= rhs,
            ComparisonOperator::Greater => lhs > rhs,
            ComparisonOperator::GreaterOrEqual => lhs >= rhs,
            ComparisonOperator::Equal => lhs == rhs,
            ComparisonOperator::NotEqual => lhs != rhs,
        }
    }
}

/// baseobjspace as_float: coerce int|float → f64.
/// Called only for int/float operands in the tracing fast path.
/// Long operands are handled by residual fallback, not this function.
#[allow(dead_code)]
unsafe fn as_float_for_trace(obj: PyObjectRef) -> f64 {
    unsafe {
        if is_float(obj) {
            w_float_get_value(obj)
        } else if is_int(obj) {
            w_int_get_value(obj) as f64
        } else {
            0.0 // unreachable in trace fast path — long triggers residual
        }
    }
}

/// Compare two numeric values as floats. Handles float_pair (int+float)
/// via as_float coercion matching baseobjspace::float_lt/le/gt/ge/eq/ne.
/// Long operands don't reach here — they trigger residual fallback.
#[allow(dead_code)]
pub(crate) unsafe fn objspace_compare_floats(
    lhs_obj: PyObjectRef,
    rhs_obj: PyObjectRef,
    op: ComparisonOperator,
) -> bool {
    unsafe {
        let lhs = as_float_for_trace(lhs_obj);
        let rhs = as_float_for_trace(rhs_obj);
        match op {
            ComparisonOperator::Less => lhs < rhs,
            ComparisonOperator::LessOrEqual => lhs <= rhs,
            ComparisonOperator::Greater => lhs > rhs,
            ComparisonOperator::GreaterOrEqual => lhs >= rhs,
            ComparisonOperator::Equal => lhs == rhs,
            ComparisonOperator::NotEqual => lhs != rhs,
        }
    }
}

/// virtualizable.py `read_boxes`: `lst = getattr(virtualizable, fieldname)`
/// for `locals_cells_stack_w[*]`. That getattr is `opimpl_getfield_gc_r`
/// (`GETFIELD_GC_R`); `compile.py` emits the same `GETFIELD_GC_R` in the
/// array-field preamble. The result is a GCREF, so a minor collection
/// forwards it (`rewrite.py` GETFIELD_GC_R, `_trace_drag_out`).
pub(crate) fn frame_locals_cells_stack_array(ctx: &mut TraceCtx, frame: OpRef) -> OpRef {
    opimpl_getfield_gc_r(ctx, frame, frame_locals_cells_stack_descr())
}

/// Read from frame's `locals_cells_stack_w` array. Caller passes the
/// `*mut FixedObjectArray` block-base pointer (output of
/// `frame_locals_cells_stack_array`); [`pyobject_gcarray_descr`] sets
/// `base_size` to the block's items offset so the descriptor skips the
/// length prefix and `GETARRAYITEM_GC_R(array, i)` lands on `items[i]`.
///
/// Uses GcR (Ref-typed) to match RPython's GETARRAYITEM_GC_R, ensuring
/// the optimizer knows these are boxed pointers.
pub(crate) fn trace_array_getitem_value(ctx: &mut TraceCtx, array: OpRef, index: OpRef) -> OpRef {
    let descr = pyobject_gcarray_descr();
    let descr_idx = descr.index();
    if let Some(cached) = ctx.heapcache_getarrayitem(array, index, descr_idx) {
        // pyjitpl.py `_do_getarrayitem_gc_any` cache hit.
        ctx.profiler().count_ops(
            OpCode::GetarrayitemGcR,
            majit_metainterp::counters::HEAPCACHED_OPS,
        );
        return cached;
    }
    ctx.profiler()
        .count_ops(OpCode::GetarrayitemGcR, majit_metainterp::counters::OPS);
    ctx.profiler().count_ops(
        OpCode::GetarrayitemGcR,
        majit_metainterp::counters::RECORDED_OPS,
    );
    let result = ctx.record_op_with_descr(OpCode::GetarrayitemGcR, &[array, index], descr.clone());
    if let Some(live_value) = array_load_for_cache(ctx, array, index, &descr, majit_ir::Type::Ref) {
        ctx.set_opref_concrete(result, live_value);
    }
    ctx.heapcache_getarrayitem_now_known(array, index, descr_idx, result);
    result
}

/// Helper: project (array, index) operands and dispatch
/// `array_sanity_load` to recover the executor-returned payload for
/// the fresh GetarrayitemGc<I|R|F> result.  Mirrors
/// `rpython/jit/metainterp/executor.py do_getarrayitem_gc_i` `do_getarrayitem_gc_*`
/// which reads `arraybox.getref_base()` / `indexbox.getint()` from
/// the Box directly — pyre's `box_value` exposes the same chain (Const
/// pool / standard-virtualizable shadow / `opref_concrete` stamp) so
/// any operand whose Box.value is known unblocks the sanity load
/// (not gated on the Const-only fast path).
fn array_load_for_cache(
    ctx: &majit_metainterp::TraceCtx,
    array: OpRef,
    index: OpRef,
    descr: &DescrRef,
    kind: majit_ir::Type,
) -> Option<majit_ir::Value> {
    let Some(majit_ir::Value::Ref(array_ref)) = ctx.box_value(array) else {
        return None;
    };
    let array_ptr = array_ref.0 as i64;
    if array_ref == majit_ir::GcRef::NO_CONCRETE || array_ptr == usize::MAX as i64 || array_ptr == 0
    {
        return None;
    }
    let Some(majit_ir::Value::Int(index_value)) = ctx.box_value(index) else {
        return None;
    };
    ctx.array_sanity_load(array_ptr, index_value, descr, kind)
}

/// Read from frame's locals_cells_stack_w — namespace access path.
#[allow(dead_code)]
pub(crate) fn trace_raw_array_getitem_value(
    ctx: &mut TraceCtx,
    array: OpRef,
    index: OpRef,
) -> OpRef {
    let descr = pyobject_array_descr();
    let descr_idx = descr.index();
    if let Some(cached) = ctx.heapcache_getarrayitem(array, index, descr_idx) {
        return cached;
    }
    // pyjitpl.py `_do_getarrayitem_gc_any` record leg.
    ctx.profiler()
        .count_ops(OpCode::GetarrayitemGcR, majit_metainterp::counters::OPS);
    ctx.profiler().count_ops(
        OpCode::GetarrayitemGcR,
        majit_metainterp::counters::RECORDED_OPS,
    );
    let result = ctx.record_op_with_descr(OpCode::GetarrayitemGcR, &[array, index], descr.clone());
    if let Some(live_value) = array_load_for_cache(ctx, array, index, &descr, majit_ir::Type::Ref) {
        ctx.set_opref_concrete(result, live_value);
    }
    ctx.heapcache_getarrayitem_now_known(array, index, descr_idx, result);
    result
}

/// `pyjitpl.py` `arraybox = opimpl_getfield_gc_r(listbox, itemsdescr)`
/// followed by `getarrayitem_gc(arraybox, idx, arraydescr)`.
///
/// Caller passes the `items_block` Ref (output of
/// `opimpl_getfield_gc_r` against the `items` /  `wrappeditems` field)
/// directly. The `pyobject_gcarray_descr` here is
/// `Ptr(GcArray(OBJECTPTR))` with `base_size = ITEMS_BLOCK_ITEMS_OFFSET`
/// (= length-prefix size), `item_size = sizeof(Ptr)` (word-sized: 4 on
/// wasm32, 8 on 64-bit), `item_type = Ref`, matching
/// `rpython/rtyper/lltypesystem/rlist.py:84` `GcArray(OBJECTPTR)`.
///
/// Replaces the prior two-step `IntAdd(items_block, OFFSET) +
/// raw-array op` deviation with the upstream single-op shape.
pub(crate) fn trace_items_block_getitem_value(
    ctx: &mut TraceCtx,
    block: OpRef,
    index: OpRef,
) -> OpRef {
    let descr = pyobject_gcarray_descr();
    let descr_idx = descr.index();
    if let Some(cached) = ctx.heapcache_getarrayitem(block, index, descr_idx) {
        return cached;
    }
    ctx.profiler()
        .count_ops(OpCode::GetarrayitemGcR, majit_metainterp::counters::OPS);
    ctx.profiler().count_ops(
        OpCode::GetarrayitemGcR,
        majit_metainterp::counters::RECORDED_OPS,
    );
    let result = ctx.record_op_with_descr(OpCode::GetarrayitemGcR, &[block, index], descr.clone());
    if let Some(live_value) = array_load_for_cache(ctx, block, index, &descr, majit_ir::Type::Ref) {
        ctx.set_opref_concrete(result, live_value);
    }
    ctx.heapcache_getarrayitem_now_known(block, index, descr_idx, result);
    result
}

/// Pure variant of [`trace_items_block_getitem_value`] — emits
/// `getarrayitem_gc_pure_r(block, index)` against the SAME
/// `pyobject_gcarray_descr` (`Ptr(GcArray(OBJECTPTR))`) for an
/// IMMUTABLE backing array (`W_TupleObject.wrappeditems`,
/// `tupleobject.py` `_immutable_fields_ = ['wrappeditems[*]']`).
///
/// Purity is carried ONLY by selecting `GetarrayitemGcPureR`; the descr
/// is the unchanged shared singleton (the items/gcarray descr must NOT
/// be marked pure — that would make every container of this strategy
/// non-invalidatable). `OptPure` CSEs / const-folds the pure op and
/// never invalidates it on an intervening write, which is sound here
/// because the tuple body is immutable.
///
/// Recording the pure op directly is the walker-native analogue of the
/// codewriter, which reaches `getarrayitem_gc_*_pure` through the
/// oopspec lowering of an immutable/foldable read (`jtransform.py:1891`);
/// the opcode-level effect is identical.
pub(crate) fn trace_items_block_getitem_value_pure(
    ctx: &mut TraceCtx,
    block: OpRef,
    index: OpRef,
) -> OpRef {
    let descr = pyobject_gcarray_descr();
    // Pure jitcode spellings alias the plain opimpl for profiling.
    ctx.profiler()
        .count_ops(OpCode::GetarrayitemGcR, majit_metainterp::counters::OPS);
    ctx.profiler().count_ops(
        OpCode::GetarrayitemGcR,
        majit_metainterp::counters::RECORDED_OPS,
    );
    let result =
        ctx.record_op_with_descr(OpCode::GetarrayitemGcPureR, &[block, index], descr.clone());
    if let Some(live_value) = array_load_for_cache(ctx, block, index, &descr, majit_ir::Type::Ref) {
        ctx.set_opref_concrete(result, live_value);
    }
    result
}

/// Companion of [`trace_items_block_getitem_value`] — emits
/// `setarrayitem_gc(block, index, value)` against `pyobject_gcarray_descr`.
pub(crate) fn trace_items_block_setitem_value(
    ctx: &mut TraceCtx,
    block: OpRef,
    index: OpRef,
    value: OpRef,
) {
    let descr = pyobject_gcarray_descr();
    let descr_idx = descr.index();
    ctx.profiler()
        .count_ops(OpCode::SetarrayitemGc, majit_metainterp::counters::OPS);
    ctx.profiler().count_ops(
        OpCode::SetarrayitemGc,
        majit_metainterp::counters::RECORDED_OPS,
    );
    ctx.record_op_with_descr(OpCode::SetarrayitemGc, &[block, index, value], descr);
    // pyjitpl.py:980 `upd.setarrayitem(valuebox)` — cache stores the
    // Box identity (`value` OpRef); cache-hit readers resolve the
    // intrinsic value via `box_value(cached)` at hit time.
    ctx.heapcache_setarrayitem(block, index, descr_idx, value);
}

/// `mapdict.py _mapdict_read_storage` — the mapdict-storage twin of
/// [`trace_items_block_getitem_value`], against
/// [`mapdict_storage_gcarray_descr`].  Only ever issued for a slot the guarded
/// map proves is boxed.
pub(crate) fn trace_mapdict_storage_getitem(
    ctx: &mut TraceCtx,
    block: OpRef,
    index: OpRef,
) -> OpRef {
    let descr = mapdict_storage_gcarray_descr();
    let descr_idx = descr.index();
    if let Some(cached) = ctx.heapcache_getarrayitem(block, index, descr_idx) {
        return cached;
    }
    ctx.profiler()
        .count_ops(OpCode::GetarrayitemGcR, majit_metainterp::counters::OPS);
    ctx.profiler().count_ops(
        OpCode::GetarrayitemGcR,
        majit_metainterp::counters::RECORDED_OPS,
    );
    let result = ctx.record_op_with_descr(OpCode::GetarrayitemGcR, &[block, index], descr.clone());
    if let Some(live_value) = array_load_for_cache(ctx, block, index, &descr, majit_ir::Type::Ref) {
        ctx.set_opref_concrete(result, live_value);
    }
    ctx.heapcache_getarrayitem_now_known(block, index, descr_idx, result);
    result
}

/// `mapdict.py _mapdict_write_storage` — companion of
/// [`trace_mapdict_storage_getitem`].
pub(crate) fn trace_mapdict_storage_setitem(
    ctx: &mut TraceCtx,
    block: OpRef,
    index: OpRef,
    value: OpRef,
) {
    let descr = mapdict_storage_gcarray_descr();
    let descr_idx = descr.index();
    ctx.profiler()
        .count_ops(OpCode::SetarrayitemGc, majit_metainterp::counters::OPS);
    ctx.profiler().count_ops(
        OpCode::SetarrayitemGc,
        majit_metainterp::counters::RECORDED_OPS,
    );
    ctx.record_op_with_descr(OpCode::SetarrayitemGc, &[block, index, value], descr);
    ctx.heapcache_setarrayitem(block, index, descr_idx, value);
}

/// Write to frame's locals_cells_stack_w array.
/// Uses Gc (GC-typed) to match RPython's SETARRAYITEM_GC.
#[allow(dead_code)]
pub(crate) fn trace_raw_array_setitem_value(
    ctx: &mut TraceCtx,
    array: OpRef,
    index: OpRef,
    value: OpRef,
) {
    let descr = pyobject_array_descr();
    let descr_idx = descr.index();
    ctx.profiler()
        .count_ops(OpCode::SetarrayitemGc, majit_metainterp::counters::OPS);
    ctx.profiler().count_ops(
        OpCode::SetarrayitemGc,
        majit_metainterp::counters::RECORDED_OPS,
    );
    ctx.record_op_with_descr(OpCode::SetarrayitemGc, &[array, index, value], descr);
    // pyjitpl.py:980 `upd.setarrayitem(valuebox)` parity — cache
    // stores the Box identity (`value` OpRef); cache-hit readers
    // resolve the intrinsic value via `box_value(cached)` at hit
    // time.
    ctx.heapcache_setarrayitem(array, index, descr_idx, value);
}

/// `GetarrayitemGcI(block, index)` against `int_gcarray_descr` — the
/// `IntegerListStrategy` GC-array read. Mirrors
/// [`trace_items_block_getitem_value`] (the Object-strategy GcArray(OBJECTPTR)
/// read) but with an `Int` item type, so the read routes through the heap cache.
pub(crate) fn trace_int_block_getitem_value(
    ctx: &mut TraceCtx,
    block: OpRef,
    index: OpRef,
) -> OpRef {
    let descr = int_gcarray_descr();
    let descr_idx = descr.index();
    if let Some(cached) = ctx.heapcache_getarrayitem(block, index, descr_idx) {
        return cached;
    }
    ctx.profiler()
        .count_ops(OpCode::GetarrayitemGcI, majit_metainterp::counters::OPS);
    ctx.profiler().count_ops(
        OpCode::GetarrayitemGcI,
        majit_metainterp::counters::RECORDED_OPS,
    );
    let result = ctx.record_op_with_descr(OpCode::GetarrayitemGcI, &[block, index], descr.clone());
    if let Some(live_value) = array_load_for_cache(ctx, block, index, &descr, majit_ir::Type::Int) {
        ctx.set_opref_concrete(result, live_value);
    }
    ctx.heapcache_getarrayitem_now_known(block, index, descr_idx, result);
    result
}

/// Companion of [`trace_int_block_getitem_value`] — `SetarrayitemGc(block,
/// index, value)` against `int_gcarray_descr` (int items carry no pointer, so
/// no write barrier is emitted).
pub(crate) fn trace_int_block_setitem_value(
    ctx: &mut TraceCtx,
    block: OpRef,
    index: OpRef,
    value: OpRef,
) {
    let descr = int_gcarray_descr();
    let descr_idx = descr.index();
    ctx.profiler()
        .count_ops(OpCode::SetarrayitemGc, majit_metainterp::counters::OPS);
    ctx.profiler().count_ops(
        OpCode::SetarrayitemGc,
        majit_metainterp::counters::RECORDED_OPS,
    );
    ctx.record_op_with_descr(OpCode::SetarrayitemGc, &[block, index, value], descr);
    ctx.heapcache_setarrayitem(block, index, descr_idx, value);
}

/// `GetarrayitemGcF(block, index)` against `float_gcarray_descr` — the
/// `FloatListStrategy` GC-array read (heap-cached). See
/// [`trace_int_block_getitem_value`].
pub(crate) fn trace_float_block_getitem_value(
    ctx: &mut TraceCtx,
    block: OpRef,
    index: OpRef,
) -> OpRef {
    let descr = float_gcarray_descr();
    let descr_idx = descr.index();
    if let Some(cached) = ctx.heapcache_getarrayitem(block, index, descr_idx) {
        return cached;
    }
    ctx.profiler()
        .count_ops(OpCode::GetarrayitemGcF, majit_metainterp::counters::OPS);
    ctx.profiler().count_ops(
        OpCode::GetarrayitemGcF,
        majit_metainterp::counters::RECORDED_OPS,
    );
    let result = ctx.record_op_with_descr(OpCode::GetarrayitemGcF, &[block, index], descr.clone());
    if let Some(live_value) = array_load_for_cache(ctx, block, index, &descr, majit_ir::Type::Float)
    {
        ctx.set_opref_concrete(result, live_value);
    }
    ctx.heapcache_getarrayitem_now_known(block, index, descr_idx, result);
    result
}

/// Companion of [`trace_float_block_getitem_value`] — `SetarrayitemGc(block,
/// index, value)` against `float_gcarray_descr`.
pub(crate) fn trace_float_block_setitem_value(
    ctx: &mut TraceCtx,
    block: OpRef,
    index: OpRef,
    value: OpRef,
) {
    let descr = float_gcarray_descr();
    let descr_idx = descr.index();
    ctx.profiler()
        .count_ops(OpCode::SetarrayitemGc, majit_metainterp::counters::OPS);
    ctx.profiler().count_ops(
        OpCode::SetarrayitemGc,
        majit_metainterp::counters::RECORDED_OPS,
    );
    ctx.record_op_with_descr(OpCode::SetarrayitemGc, &[block, index, value], descr);
    ctx.heapcache_setarrayitem(block, index, descr_idx, value);
}

/// `PyFrame.get_w_globals` common path: read the frame's own pycode, then its
/// first-seen globals. Inline callers only use this after proving that the
/// callee function globals is that code-global identity; the uncommon
/// debugdata override stays on the residual path.
pub(crate) fn frame_get_globals_obj(ctx: &mut TraceCtx, frame: OpRef) -> OpRef {
    let pycode = ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[frame],
        crate::descr::pyframe_code_descr(),
    );
    ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[pycode],
        crate::descr::pycode_w_globals_descr(),
    )
}

/// Read a value from the unified `locals_cells_stack_w` at the given absolute index.
pub fn concrete_stack_value(frame: usize, abs_idx: usize) -> Option<PyObjectRef> {
    let frame_ptr = (frame != 0).then_some(frame as *const u8)?;
    let arr_ptr = unsafe {
        *(frame_ptr.add(PYFRAME_LOCALS_CELLS_STACK_OFFSET)
            as *const *const pyre_object::FixedObjectArray)
    };
    let arr = unsafe { &*arr_ptr };
    arr.as_slice().get(abs_idx).copied()
}

/// Read up to `max_len` slots of the GC-rooted live virtualizable frame's
/// `locals_cells_stack_w` as concrete `Ref` values.  The live frame
/// (`sym.concrete_vable_ptr`) sits on the CURRENT_FRAME chain, so
/// `walk_pyframe_roots` forwards its slots on every collection — reading
/// them here is GC-safe.  This is the source the bridge concrete shadow
/// must use instead of the off-heap resume-decoded array, whose `Ref`
/// entries dangle once a minor collection (residual virtual materialization
/// during bridge setup) moves the referent but cannot forward the decode
/// `Vec`.  Falls back to `fallback` (the decoded values) when no live
/// frame/array is bound (unit-test / init-before-run path, no GC hazard).
fn live_frame_array_values(
    vable_ptr: usize,
    max_len: usize,
    fallback: &[majit_ir::Value],
) -> Vec<majit_ir::Value> {
    if vable_ptr == 0 {
        return fallback.to_vec();
    }
    let f = unsafe { &*(vable_ptr as *const pyre_interpreter::pyframe::PyFrame) };
    let lp = f.locals_cells_stack_w;
    if lp.is_null() {
        return fallback.to_vec();
    }
    let arr = unsafe { &*lp };
    let base = arr.items_ptr() as *const pyre_object::PyObjectRef;
    let n = max_len.min(arr.len());
    (0..n)
        .map(|i| majit_ir::Value::Ref(majit_ir::GcRef(unsafe { *base.add(i) } as usize)))
        .collect()
}

/// Re-arm the write barrier on a frame + its `locals_cells_stack_w` array
/// after a JIT-side flush stored refs into them raw. The stored values may
/// be nursery-young while the frame/array are old-gen, and after the flush
/// the frame can run detached from the walked frame chain (virtualizable),
/// so no minor re-traces the items unless an owner sits in the remembered
/// set (virtualizable.py `write_boxes` stores run under the translated
/// write barrier; `remember_frame_locals_array` is the creation-time twin).
/// A GC-owned array re-traces its own items; a stationary `std::alloc`
/// array is only re-walked through its owning frame's custom trace — arm
/// whichever side the GC owns.
pub(crate) fn frame_array_write_barrier(
    frame: *mut u8,
    arr_ptr: *mut pyre_object::FixedObjectArray,
) {
    if pyre_object::gc_hook::try_gc_owns_object(arr_ptr as pyre_object::gc_hook::GCREF) {
        pyre_object::gc_hook::try_gc_write_barrier(arr_ptr as pyre_object::gc_hook::GCREF);
    }
    if pyre_object::gc_hook::try_gc_owns_object(frame as pyre_object::gc_hook::GCREF) {
        pyre_object::gc_hook::try_gc_write_barrier(frame as pyre_object::gc_hook::GCREF);
    }
}

/// Write one decoded Ref back into the live frame's locals_cells_stack_w
/// (the GC-rooted virtualizable array), re-asserting the resume-decoded
/// vable image after the guard-failure vsd correction cleared root slots
/// in callee coordinates.  A no-op for a null frame/array, an out-of-range
/// slot, or a non-Ref value.
pub(crate) fn store_live_frame_array_slot(vable_ptr: usize, slot: usize, value: majit_ir::Value) {
    let majit_ir::Value::Ref(r) = value else {
        return;
    };
    if vable_ptr == 0 {
        return;
    }
    let f = unsafe { &*(vable_ptr as *const pyre_interpreter::pyframe::PyFrame) };
    let lp = f.locals_cells_stack_w;
    if lp.is_null() {
        return;
    }
    let arr = unsafe { &mut *lp };
    if slot >= arr.len() {
        return;
    }
    // `setarrayitem_gc` on the GcArray of GCREF: barrier the array
    // (`write_barrier_from_array`), not a raw slot write.
    arr.set_ref(slot, r.as_usize() as pyre_object::PyObjectRef);
}

/// Keep the scalar half of an inlined frame's red virtualizable coherent with
/// its MIFrame walk.  Static-field indices are the `PyFrame`
/// `VirtualizableInfo` order used by the codewriter: 0 = `last_instr`, 2 =
/// `valuestackdepth`.  Other fields are immutable frame identity/state and are
/// deliberately not mirrored here.
pub(crate) fn store_live_frame_static_int(vable_ptr: usize, field_index: usize, value: i64) {
    if vable_ptr == 0 {
        return;
    }
    match field_index {
        0 => unsafe {
            *((vable_ptr + crate::frame_layout::PYFRAME_LAST_INSTR_OFFSET) as *mut isize) =
                value as isize;
        },
        2 if value >= 0 => {
            let depth = value as usize;
            if concrete_frame_array_len(vable_ptr).is_some_and(|len| depth <= len) {
                set_concrete_stack_depth(vable_ptr, depth);
            }
        }
        _ => {}
    }
}

/// pyframe.py:107-110: `locals_cells_stack_w` length =
/// `co_nlocals + ncellvars + nfreevars + co_stacksize`. Returns the
/// full heap-side array length (matching `virtualizable.py read_boxes
/// read_boxes` which iterates `len(lst)` over the full array).
pub(crate) fn concrete_frame_array_len(frame: usize) -> Option<usize> {
    let frame_ptr = (frame != 0).then_some(frame as *const u8)?;
    let arr_ptr = unsafe {
        *(frame_ptr.add(PYFRAME_LOCALS_CELLS_STACK_OFFSET)
            as *const *const pyre_object::FixedObjectArray)
    };
    if arr_ptr.is_null() {
        return None;
    }
    Some(unsafe { &*arr_ptr }.as_slice().len())
}

/// pyframe.py:111: valuestackdepth = co_nlocals + ncellvars + nfreevars.
/// Returns the stack base index (nlocals + ncells) for the given frame.
/// This is the number of non-stack slots in the unified locals_cells_stack_w
/// array: local variables + cell/free variable slots.
pub(crate) fn concrete_nlocals(frame: usize) -> Option<usize> {
    let frame_ptr = (frame != 0).then_some(frame as *const u8)?;
    let w_code =
        unsafe { *(frame_ptr.add(crate::frame_layout::PYFRAME_PYCODE_OFFSET) as *const *const ()) };
    if w_code.is_null() {
        return None;
    }
    let raw_code = unsafe {
        pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
            as *const pyre_interpreter::CodeObject
    };
    // A non-null `w_code` does not make the inner pointer non-null; every other
    // caller of `w_code_get_ptr` in the tree checks the result before
    // dereferencing it.
    if raw_code.is_null() {
        return None;
    }
    let code = unsafe { &*raw_code };
    let nlocals = code.varnames.len();
    let ncells = pyre_interpreter::pyframe::ncells(code);
    Some(nlocals + ncells)
}

/// Static operand-stack depth at `target_pc` when it differs from the depth
/// `frame` still advertises. A closing JUMP retargeted to a different bytecode
/// offset must publish the merge point's own live depth.
///
/// `None` means "keep reading the frame": the frame or its code object cannot
/// answer, the code has no liveness entry for `target_pc`, the depth would
/// overrun `locals_cells_stack_w`, or the frame is already at the merge
/// point's depth.
pub(crate) fn merge_point_stack_depth_to_recover(frame: usize, target_pc: usize) -> Option<usize> {
    let concrete = concrete_stack_depth(frame)?;
    let w_code = unsafe {
        *((frame as *const u8).add(crate::frame_layout::PYFRAME_PYCODE_OFFSET) as *const usize)
    };
    let depth = depth_based_vsd_for_wcode(w_code, target_pc)?;
    // `store_live_frame_static_int`'s bound: a depth past the array is not a
    // depth this frame can describe.
    if concrete_frame_array_len(frame).is_none_or(|len| depth > len) {
        return None;
    }
    (depth != concrete).then_some(depth)
}

/// Return the absolute valuestackdepth.
pub(crate) fn concrete_stack_depth(frame: usize) -> Option<usize> {
    let frame_ptr = (frame != 0).then_some(frame as *const u8)?;
    Some(unsafe { *(frame_ptr.add(PYFRAME_VALUESTACKDEPTH_OFFSET) as *const usize) })
}

/// Write the absolute valuestackdepth into the concrete `PyFrame` at
/// `frame`. The trait `interpret()` leg traces on a heap snapshot that
/// is never concretely stepped (`trace.rs` KNOWN DIVERGENCE), so a
/// multi-frame bridge's snapshot keeps the `valuestackdepth` it was
/// reconstructed with (the guard-failure resume depth, mid-iteration).
/// The symbolic walk DOES track the live depth (`push_typed_value` /
/// `pop_value`). At the loop-header close the back-edge depth is the
/// loop-header invariant — the live symbolic value — so syncing the
/// snapshot here lets `concrete_valuestackdepth()` (the reader
/// `close_loop_args_at` trusts, matching RPython reading the single real
/// frame) report the merge-point depth instead of the stale seed.
pub(crate) fn set_concrete_stack_depth(frame: usize, depth: usize) {
    if frame != 0 {
        unsafe {
            *((frame as *mut u8).add(PYFRAME_VALUESTACKDEPTH_OFFSET) as *mut usize) = depth;
        }
    }
}

/// Derive `(num_locals, num_locals + max_stackdepth)` from a `CodeObject`.
///
/// Mirrors the `(callee_nlocals, callee_vsd)` pair the trace-side reads
/// from `driver.get_compiled_meta(callee_key)` for CALL_ASSEMBLER
/// emission (`trace_opcode.rs`). The second value is sized to
/// match `PyFrame::__init__` (`alloc_fixed_array_with_header(num_locals +
/// num_cells + max_stack, ...)`) — i.e. heap capacity rather than live
/// depth. Used as the fallback shape when no `compiled_meta` exists yet
/// (e.g. tmp_callback target where `compile_tmp_callback` produced a
/// JCT but no compiled-loop metadata).
#[allow(dead_code)]
pub(crate) fn callee_layout_for_call_assembler(
    code: &pyre_interpreter::CodeObject,
) -> (usize, usize) {
    let nlocals = code.varnames.len() + pyre_interpreter::pyframe::ncells(code);
    let stack_only = code.max_stackdepth as usize;
    (nlocals, nlocals + stack_only)
}

/// `celldict.py getdictvalue_no_unwrapping` slot lookup against the
/// module dict's `ModuleDictStorage` (the cell store). `None` for non-module dicts and after
/// `switch_to_object_strategy`.  Used by the cell fast path to derive the
/// elidable lookup key.
pub(crate) fn module_dict_cell_slot_direct(obj: PyObjectRef, name: &str) -> Option<usize> {
    unsafe { pyre_object::dictmultiobject::module_dict_cell_slot_of(obj, name) }
}

/// `celldict.py _getdictvalue_no_unwrapping_pure` — the raw stored
/// value-or-cell at `slot` (not unwrapped), read at trace time to
/// classify the cell (`ObjectMutableCell` → live `GetfieldGcR`, raw value
/// → const fold, `IntMutableCell` → name-based fallback).
pub(crate) fn module_dict_cell_value_direct(obj: PyObjectRef, slot: usize) -> Option<PyObjectRef> {
    unsafe { pyre_object::dictmultiobject::module_dict_cell_at(obj, slot) }
}

/// pyjitpl.py `opimpl_record_quasiimmut_field`: record the
/// dependency marker and arm the pending GUARD_NOT_INVALIDATED once per
/// heapcache epoch.
///
/// The heapcache keys on `descr.index()` and the receiver, the same way
/// [`opimpl_getfield_gc_i`] does. A cache hit counts and returns. The
/// following getfield owns the ordinary field cache
/// (`_opimpl_getfield_gc_any_pureornot`).
pub(crate) fn record_quasiimmut_field(ctx: &mut TraceCtx, obj: OpRef, descr: DescrRef) {
    let field_index = descr.index();
    if ctx.heap_cache().is_quasi_immut_known(field_index, obj) {
        ctx.profiler().count_ops(
            OpCode::QuasiimmutField,
            majit_metainterp::counters::HEAPCACHED_OPS,
        );
        return;
    }
    // quasiimmut.py `self.qmut = get_current_qmut_instance(cpu, struct,
    // mutatefielddescr)` — the half that makes the value captured below
    // answerable later.  The hidden `mutate_<name>` field is null until
    // something installs it, and while it is null a write to the
    // quasi-immutable field takes the no-watchers early return: the value moves
    // and nothing is forced.  Installing at the record makes every write from
    // here on run the invalidation, which is what both the tracer's own force
    // check and the optimizer's revalidation read.
    //
    // Ordered before the value read for the reason upstream orders `__init__`
    // that way: a write landing between the two must be one the watcher sees.
    // Reading first leaves a window where the field moves with no watcher
    // installed, so nothing invalidates and nothing bumps the force counter,
    // and the trace keeps a value that is already stale.  Without a GIL that
    // window is a real interleaving, not a theoretical one.
    let qmutdescr = quasi_immut_descr(ctx, obj, &descr);
    // `quasiimmut.py QuasiImmutDescr.__init__` captures
    // `constantfieldbox` on the descr; `pyjitpl.py
    // opimpl_record_quasiimmut_field` then `record1(QUASIIMMUT_FIELD,
    // box, descr=qmutdescr)`.
    let constantfieldbox = current_quasiimmut_field_value(ctx, obj, &descr);
    ctx.heap_cache_mut().quasi_immut_now_known(field_index, obj);
    let descr = match qmutdescr.as_ref().and_then(|d| d.as_quasi_immut_descr()) {
        Some(qd) => std::sync::Arc::new(majit_ir::QuasiImmutDescr::new(
            qd.fielddescr().clone(),
            qd.struct_ptr(),
            qd.qmut().clone(),
            constantfieldbox.and_then(|r| r.inline_const_to_value()),
        )) as majit_ir::DescrRef,
        None => {
            if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
                let name = descr
                    .as_field_descr()
                    .map(|field| field.field_name())
                    .unwrap_or("");
                eprintln!(
                    "[qmut-miss] index={:#x} name={name} quasi={} offset={}",
                    descr.index(),
                    descr.is_quasi_immutable(),
                    descr
                        .as_field_descr()
                        .map(|field| field.offset())
                        .unwrap_or(0),
                );
            }
            descr
        }
    };
    ctx.record_op_with_descr(OpCode::QuasiimmutField, &[obj], descr);
    if ctx.heap_cache_mut().check_and_clear_guard_not_invalidated() {
        ctx.set_pending_guard_not_invalidated(Some(ctx.last_traced_pc));
    }
}

/// `quasiimmut.py QuasiImmut` published to the optimizer and the
/// compiler as a [`majit_ir::QuasiImmutHandle`].
///
/// The interpreter's own type cannot carry the impl: `pyre-object` does not
/// depend on `majit-ir`, and which JIT questions an instance answers is not a
/// property of the object model.
struct RecordedQuasiImmut(std::sync::Arc<pyre_object::quasiimmut::QuasiImmut>);

impl std::fmt::Debug for RecordedQuasiImmut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "QuasiImmut({:p})", std::sync::Arc::as_ptr(&self.0))
    }
}

impl majit_ir::QuasiImmutHandle for RecordedQuasiImmut {
    fn is_current(&self) -> bool {
        self.0.is_current()
    }

    fn register_loop_token(&self, token: &std::sync::Arc<dyn majit_ir::QuasiImmutLoopToken>) {
        self.0.register_loop_token(token);
    }

    fn instance_identity(&self) -> usize {
        std::sync::Arc::as_ptr(&self.0) as usize
    }
}

/// `pycode.py` `"w_globals?"`. Analyzer `fielddescrof` stamps
/// `PyCode.w_globals`; the reserved quasi descr uses the same offset and a
/// `Struct.w_globals` name.
/// A constant globals operand is its own namespace. A non-constant one is
/// the recorded `PyCode.w_globals` read only when that field's
/// quasi-immutable dependency is already known: the marker keeps the dict
/// equal to the value observed while recording. A mutable read, including
/// `debugdata.w_globals`, is not that field.
pub(crate) fn globals_read_keeps_recorded_namespace(ctx: &TraceCtx, globals: OpRef) -> bool {
    if globals.is_constant() {
        return true;
    }
    let Some((descr, obj)) = ctx.ref_getfield_gc_r(globals) else {
        return false;
    };
    is_pycode_w_globals_descr(&descr) && ctx.heap_cache().is_quasi_immut_known(descr.index(), obj)
}

fn is_pycode_w_globals_descr(descr: &DescrRef) -> bool {
    if !descr.is_quasi_immutable() {
        return false;
    }
    let Some(field) = descr.as_field_descr() else {
        return false;
    };
    if field.offset() != pyre_interpreter::pycode::CODE_W_GLOBALS_OFFSET {
        return false;
    }
    let name = field.field_name();
    name == "w_globals" || name.ends_with(".w_globals")
}

/// Analyzer `fielddescrof` for `typeobject.py` `_version_tag?` mints its own
/// index. The reserved [`crate::descr::type_version_tag_descr`] is the same
/// slot: `W_TypeObject.version_tag`.
fn is_type_version_tag_descr(descr: &DescrRef) -> bool {
    if !descr.is_quasi_immutable() {
        return false;
    }
    let Some(field) = descr.as_field_descr() else {
        return false;
    };
    if field.offset() != core::mem::offset_of!(pyre_object::typeobject::W_TypeObject, version_tag) {
        return false;
    }
    let name = field.field_name();
    name == "version_tag" || name.ends_with(".version_tag")
}

/// `pyjitpl.py:1081 QuasiImmutDescr(cpu, structbox.getref_base(), fielddescr,
/// mutatefielddescr)` — the descr a recorded `QUASIIMMUT_FIELD` carries.
///
/// Building it is what resolves the instance (`quasiimmut.py
/// get_current_qmut_instance`), so it exists for the rest of the recording,
/// which is what arms `opimpl_jit_force_quasi_immutable`'s `mutatebox.nonnull()`
/// (`pyjitpl.py`) for a write reached later in that same trace. The
/// instance rides on the descr from here to `heap.py OptHeap.optimize_QUASIIMMUT_FIELD is_still_valid_for`
/// and `compile.py register_loop_token`, so neither has to walk from
/// the struct back to the hidden `mutate_*` slot a second time.
///
/// `None` where upstream cannot fail: an operand that is not a concrete
/// pointer, or a descr index no arm claims. The caller records the plain field
/// descr, and the optimizer then refuses the loop rather than standing a fold
/// with no watcher behind it — `heap.py:814`'s assert, read as a verdict.
fn quasi_immut_descr(ctx: &mut TraceCtx, obj: OpRef, descr: &DescrRef) -> Option<DescrRef> {
    let Some(majit_ir::Value::Ref(struct_ref)) = ctx.box_value(obj) else {
        return None;
    };
    let struct_ptr = struct_ref.0 as i64;
    if struct_ptr == 0 || struct_ptr == usize::MAX as i64 {
        return None;
    }
    let index = descr.index();
    // The index decides which type `struct_ptr` is cast to, so an unrecognised
    // one must fail loudly rather than reinterpret a headerless map-node
    // allocation as a `W_TypeObject`.  The arms below are every quasi-immutable
    // descr this binary mints: the hand-minted singletons, including
    // `GilReadyState.gil_ready`, `W_TypeObject.name?`, the nine `Function`
    // fields `function.py` declares (`function_quasi_immut_slot`), and
    // `PyCode.w_globals`. Analyzer `fielddescrof` stamps that field from
    // `pycode.py` `"w_globals?"`; the name plus `CODE_W_GLOBALS_OFFSET` is
    // the same slot as the reserved `pycode_w_globals_quasi_descr` index.
    let qmut = unsafe {
        if index == crate::descr::module_dict_version_descr().index() {
            pyre_object::dictmultiobject::module_dict_strategy_current_version_qmut(
                struct_ptr as *mut pyre_object::celldict::ModuleDictStrategy,
            )
        } else if index == crate::descr::type_version_tag_descr().index()
            || is_type_version_tag_descr(descr)
        {
            pyre_object::typeobject::w_type_current_qmut_instance(
                struct_ptr as pyre_object::PyObjectRef,
            )
        } else if index == crate::descr::type_name_obj_descr().index() {
            pyre_object::typeobject::w_type_current_w_name_qmut(
                struct_ptr as pyre_object::PyObjectRef,
            )
        } else if index == crate::descr::terminator_allow_unboxing_descr().index() {
            pyre_interpreter::objspace::std::mapdict::terminator_current_allow_unboxing_qmut(
                struct_ptr as *const _,
            )
        } else if index == crate::descr::plain_attribute_ever_mutated_descr().index() {
            pyre_interpreter::objspace::std::mapdict::plain_attribute_current_ever_mutated_qmut(
                struct_ptr as *const _,
            )
        } else if index == crate::descr::cell_family_ever_mutated_descr().index() {
            Some(
                (*(struct_ptr as *const pyre_object::nestedscope::CellFamily))
                    .current_ever_mutated_qmut(),
            )
        } else if index == crate::descr::holder_attr_descr().index() {
            pyre_interpreter::objspace::std::mapdict::holder_current_attr_qmut(
                struct_ptr as *const _,
            )
        } else if index == crate::descr::holder_typ_descr().index() {
            pyre_interpreter::objspace::std::mapdict::holder_current_typ_qmut(
                struct_ptr as *const _,
            )
        } else if index == crate::descr::audit_holder_hooks_descr().index() {
            pyre_interpreter::module::sys::vm::audit_holder_current_hooks_qmut(
                struct_ptr as *const _,
            )
        } else if index == crate::descr::property_fget_descr().index() {
            pyre_object::descriptor::w_property_current_fget_qmut(
                struct_ptr as pyre_object::PyObjectRef,
            )
        } else if index == crate::descr::property_fset_descr().index() {
            pyre_object::descriptor::w_property_current_fset_qmut(
                struct_ptr as pyre_object::PyObjectRef,
            )
        } else if index == crate::descr::staticmethod_w_function_quasi_descr().index() {
            pyre_object::function::w_staticmethod_current_w_function_qmut(
                struct_ptr as pyre_object::PyObjectRef,
            )
        } else if index == crate::descr::classmethod_w_function_quasi_descr().index() {
            pyre_object::function::w_classmethod_current_w_function_qmut(
                struct_ptr as pyre_object::PyObjectRef,
            )
        } else if index == crate::descr::pycode_w_globals_quasi_descr().index()
            || is_pycode_w_globals_descr(descr)
        {
            pyre_interpreter::pycode::w_code_current_w_globals_qmut(
                struct_ptr as pyre_object::PyObjectRef,
            )
        } else if index == crate::descr::ec_w_tracefunc_descr().index() {
            // Ahead of `function_quasi_immut_slot`.  The reserved index is
            // the owner identity; this order keeps an ordinal slip from
            // casting an EC as a Function.
            pyre_interpreter::executioncontext::ec_current_w_tracefunc_qmut(
                struct_ptr as *const pyre_interpreter::executioncontext::ExecutionContext,
            )
        } else if index == crate::descr::ec_profilefunc_descr().index() {
            pyre_interpreter::executioncontext::ec_current_profilefunc_qmut(
                struct_ptr as *const pyre_interpreter::executioncontext::ExecutionContext,
            )
        } else if index == crate::descr::gil_ready_descr().index() {
            // `gil.py` `GILThreadLocals.gil_ready?` — one process-global
            // watcher. `struct_ptr` is the static's address; every reader
            // shares this invalidation.
            Some(pyre_object::gil_ready::gil_ready_current_qmut())
        } else if index == crate::descr::w_class_descr().index() {
            // `PyObject.w_class?` — one process-global watcher; the
            // struct_ptr is unused (every instance shares the invalidation).
            Some(pyre_object::w_class_current_qmut())
        } else if descr.is_quasi_immutable()
            && descr.as_field_descr().is_some_and(|field| {
                let name = field.field_name();
                name == "size" || name.ends_with(".size")
            })
        {
            // `W_CType.size?` is the only quasi-immutable `size`. The watcher
            // lives on the ctype; the module hook reads it.
            pyre_interpreter::importing::ctype_size_qmut(struct_ptr as *const u8)
        } else if let Some(slot) = crate::descr::function_quasi_immut_slot(index) {
            pyre_interpreter::function::function_current_qmut_instance(
                struct_ptr as pyre_object::PyObjectRef,
                slot,
            )
        } else {
            debug_assert!(
                false,
                "unrecognised quasi-immutable field descr index {index:#x}"
            );
            None
        }
    }?;
    Some(std::sync::Arc::new(majit_ir::QuasiImmutDescr::new(
        descr.clone(),
        struct_ptr as u64,
        std::sync::Arc::new(RecordedQuasiImmut(qmut)),
        None,
    )))
}

/// quasiimmut.py `QuasiImmutDescr.get_current_constant_fieldvalue`.
///
/// ```text
///  def get_current_constant_fieldvalue(self):
///      struct = self.struct
///      fielddescr = self.fielddescr
///      if self.fielddescr.is_pointer_field():
///          return ConstPtr(self.cpu.bh_getfield_gc_r(struct, fielddescr))
///      elif self.fielddescr.is_float_field():
///          return ConstFloat(self.cpu.bh_getfield_gc_f(struct, fielddescr))
///      else:
///          return ConstInt(self.cpu.bh_getfield_gc_i(struct, fielddescr))
/// ```
///
/// `field_sanity_load` is the `cpu.bh_getfield_gc_*` triple behind one
/// `field_type()` dispatch.  `None` when the receiver is not a concrete
/// pointer or the descr is not a field descr — the optimizer then skips the
/// revalidation, matching heap.py:808-810, which ignores a QUASIIMMUT_FIELD
/// whose struct did not fold to a constant.
fn current_quasiimmut_field_value(
    ctx: &mut TraceCtx,
    obj: OpRef,
    descr: &DescrRef,
) -> Option<OpRef> {
    // quasiimmut.py `get_current_constant_fieldvalue` reads the
    // concrete struct through `cpu.bh_getfield_gc_*`. A heap-cache
    // hit is not that read: `quasi_immut_descr` already requires
    // `ctx.box_value(obj)` to be a live reference.
    let field_type = descr.as_field_descr()?.field_type();
    let majit_ir::Value::Ref(struct_ref) = ctx.box_value(obj)? else {
        return None;
    };
    let struct_ptr = struct_ref.0 as i64;
    if struct_ptr == 0 || struct_ptr == usize::MAX as i64 {
        return None;
    }
    match ctx.field_sanity_load(struct_ptr, descr, field_type)? {
        majit_ir::Value::Int(n) => Some(ctx.const_int(n)),
        majit_ir::Value::Ref(gcref) => Some(ctx.const_ref(gcref.0 as i64)),
        majit_ir::Value::Float(f) => Some(ctx.const_float(f.to_bits() as i64)),
        majit_ir::Value::Void => None,
    }
}

/// virtualizable.py:44 + interp_jit.py:25-30 —
/// `locals_cells_stack_w[*]` is declared as a W_Root array, so every
/// item's JIT type is GCREF (Type::Ref). W_IntObject/W_FloatObject are
/// stored as Ref pointers; unboxing happens inside trace opcode handlers
/// via `guard_class + getfield_gc_pure_i` / `_f`, never at the
/// virtualizable slot level.
pub(crate) fn concrete_virtualizable_slot_type(_value: PyObjectRef) -> Type {
    Type::Ref
}

/// pyjitpl.py `raise_continue_running_normally` parity for the
/// authoritative full-body walk.  The walk concretely executed the traced
/// region's residual calls, so a walk end that returns control to the
/// interpreter must hand back the END-of-walk frame state — the same
/// contract the guard-failure blackhole writeback satisfies
/// (`handle_jitexception` re-enters `eval_loop_jit` from the frame's own
/// `last_instr`).  Without this the interpreter (or a freshly entered
/// compiled loop, whose preamble reloads vable fields from the heap)
/// re-runs the walked region from its START state, re-applying every
/// concretely executed side effect.
///
/// Writes every live `locals_cells_stack_w` slot from the virtualizable
/// shadow's concrete half (`virtualizable_entry_at` — the value half of
/// the Box pair the authoritative walk maintains), `valuestackdepth` =
/// the `LiveVars` forward-stack-analysis depth at the merge point (the
/// cached `depth_at_py_pc` derivation, and `last_instr = resume_py_pc - 1` so
/// `next_instr()` re-enters at the merge point.  A bridge walk enters at
/// a guard pc whose stack depth differs from the merge point's, so the
/// depth must come from the analysis, not from the frame's entry value.
///
/// All-or-nothing: returns false (frame untouched) when any live slot
/// lacks a shadow entry or when the depth analysis has no entry for the
/// merge pc. The caller then keeps the legacy replay-from-start behavior.
pub(crate) fn flush_walk_end_state_to_frame(
    ctx: &TraceCtx,
    frame: usize,
    resume_py_pc: usize,
) -> bool {
    flush_walk_end_state_to_frame_inner(ctx, frame, resume_py_pc, None, &[], None, false)
}

/// Merge-point handoff for `raise_continue_running_normally`.
///
/// A handler loop can carry a null saved-exception sentinel as a real live
/// operand.  Accept it only when the recorded box independently resolves to
/// that same concrete null; abort-point handoffs keep rejecting null shadow
/// slots because their mid-expression operands may be unavailable.
pub(crate) fn flush_walk_loop_end_state_to_frame(
    ctx: &TraceCtx,
    frame: usize,
    resume_py_pc: usize,
) -> bool {
    flush_walk_end_state_to_frame_inner(ctx, frame, resume_py_pc, None, &[], None, true)
}

/// `flush_walk_end_state_to_frame` plus an optional in-flight FOR_ITER item
/// delivery (#32 S2).  When `push` is `Some((item, body_pc))` and the flush at
/// `resume_py_pc` (a FOR_ITER header) commits, the consumed `item` is pushed
/// one slot above the flushed operand stack and `last_instr` is repositioned to
/// `body_pc - 1` so `next_instr()` re-enters the FOR_ITER body — delivering the
/// already-advanced iteration exactly once (the FOR_ITER itself is NOT re-run).
/// `push = None` is byte-identical to the plain flush.
#[allow(dead_code)]
pub(crate) fn flush_walk_end_state_to_frame_with_item(
    ctx: &TraceCtx,
    frame: usize,
    resume_py_pc: usize,
    push: Option<(PyObjectRef, usize)>,
) -> bool {
    // This handoff is selected only for a proven FOR_ITER header.  Like the
    // ordinary loop-end handoff above, its kept stack can legitimately carry
    // a PUSH_NULL call sentinel from an enclosing expression; accept it when
    // the recorded box independently proves the same concrete null.
    flush_walk_end_state_to_frame_inner(ctx, frame, resume_py_pc, push, &[], None, true)
}

/// `flush_walk_end_state_to_frame` with the COMPLETE operand stack supplied by
/// the caller as a contiguous slice for absolute slots
/// `nlocals..nlocals + stack.len()`.  The mid-expression escape flush resolves
/// the walk's operand-stack mirror to concrete refs, which the vable shadow
/// cannot provide outside a merge point (its stack region reads NULL there).
/// Declines (frame untouched) when the supplied height disagrees with the
/// forward analysis's depth at `resume_py_pc`.
///
/// `stack` values are re-read per slot during the commit loop, whose Int/Float
/// local boxing can trigger a minor collection — the caller must keep the
/// slice registered as GC roots (`push_resume_ref_roots`) across the call so
/// the slots are forwarded in place.
pub(crate) fn flush_walk_end_state_to_frame_with_full_stack(
    ctx: &TraceCtx,
    frame: usize,
    resume_py_pc: usize,
    stack: &[PyObjectRef],
) -> bool {
    flush_walk_end_state_to_frame_inner(ctx, frame, resume_py_pc, None, &[], Some(stack), false)
}

fn flush_walk_end_state_to_frame_inner(
    ctx: &TraceCtx,
    frame: usize,
    resume_py_pc: usize,
    push: Option<(PyObjectRef, usize)>,
    stack_overrides: &[(usize, PyObjectRef)],
    full_stack: Option<&[PyObjectRef]>,
    allow_known_null_stack: bool,
) -> bool {
    let decline = |why: &str| {
        if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
            eprintln!("[fbw-flush] DECLINE at py_pc={resume_py_pc}: {why}");
        }
        false
    };
    if frame == 0 {
        return decline("frame==0");
    }
    let Some(nlocals) = concrete_nlocals(frame) else {
        return decline("no concrete_nlocals");
    };
    let Some(info) = ctx.virtualizable_info() else {
        return decline("no virtualizable_info");
    };
    // Stack depth at the merge point from the cached forward analysis:
    // absolute vsd = stack_base + depth, stack_base = nlocals + ncells
    // = `concrete_nlocals`.
    let frame_ptr = frame as *const u8;
    let w_code =
        unsafe { *(frame_ptr.add(crate::frame_layout::PYFRAME_PYCODE_OFFSET) as *const *const ()) };
    if w_code.is_null() {
        return decline("null w_code");
    }
    let raw_code = unsafe {
        pyre_interpreter::w_code_get_ptr(w_code as PyObjectRef)
            as *const pyre_interpreter::CodeObject
    };
    let Some(depth) = crate::liveness::liveness_for(raw_code)
        .depth_at_py_pc()
        .get(resume_py_pc)
        .copied()
    else {
        return decline("no depth_at_py_pc");
    };
    if let Some(full) = full_stack
        && full.len() != depth as usize
    {
        return decline("full-stack height != analysis depth");
    }
    let end_vsd = nlocals + depth as usize;
    let live = end_vsd.max(nlocals);
    let base = info.num_static_extra_boxes;
    // Validation pass first: it allocates nothing, so entry presence
    // cannot change under it.  Commit only when every live slot resolves.
    let stack_override_at = |abs: usize| -> Option<PyObjectRef> {
        if let Some(full) = full_stack {
            return full.get(abs - nlocals).copied();
        }
        stack_overrides
            .iter()
            .find_map(|&(slot, value)| (slot == abs).then_some(value))
    };
    let have_overrides = full_stack.is_some() || !stack_overrides.is_empty();
    for abs in 0..live {
        if abs >= nlocals && have_overrides {
            // An override is taken as authoritative, INCLUDING a null one — the
            // NULL-operand-slot refusal below cannot run here, because a
            // `CALL`'s `null_or_self` is a live slot whose correct value is
            // exactly null.  Keeping a null out of every OTHER slot is
            // therefore the producer's obligation:
            // `collect_call_stack_overrides` resolves each slot through
            // sources that report a null as UNRESOLVED and injects the one
            // sentinel by name, so a slot it could not resolve arrives absent
            // and declines here rather than arriving as a null.
            if stack_override_at(abs).is_none() {
                return decline("stack override missing for a live slot");
            }
            continue;
        }
        let Some((opref, value)) = ctx.virtualizable_entry_at(base + abs) else {
            return decline("no shadow entry for a live slot (validation)");
        };
        // An operand-STACK slot (`abs >= nlocals`) that resolves to a NULL Ref
        // is normally an UNPOPULATED shadow slot, not a live value: the
        // virtualizable shadow tracks locals/cells faithfully but its stack
        // region is only generally valid at a merge point where the walk's
        // stack effects have settled.  A merge-point handoff can also carry a
        // real null sentinel when its recorded box proves the same concrete
        // value.  At an arbitrary mid-opcode resume pc (an
        // `abort_permanent` marker reached partway through an opcode's stack
        // build — e.g. a MAKE_FUNCTION whose LOAD_CONST'd code object the walk
        // tracked symbolically, never writing it to the shadow) those slots
        // read back NULL.  Writing NULL into a live operand-stack slot and
        // resuming there faults the interpreter (it pops NULL where a real
        // object is expected).  Decline so the legacy replay reconstructs the
        // frame from its start state instead.  A local slot may legitimately
        // be NULL (an unbound local), so this only guards the stack region.
        if abs >= nlocals && matches!(value, Value::Ref(r) if r.0 == 0) {
            let recorded_null = matches!(
                ctx.concrete_of_opref(opref),
                Some(Value::Ref(r)) if r.0 == 0
            ) || ctx.virtualizable_slot_stored_live_null(base + abs)
                || crate::jitcode_dispatch::null_ref_is_a_value(opref);
            if !allow_known_null_stack || !recorded_null {
                return decline(&format!(
                    "NULL operand-stack shadow slot (mid-expression) at abs={abs} \
                     nlocals={nlocals} depth={depth} box={opref:?}"
                ));
            }
        }
    }
    let arr_ptr = unsafe {
        *(frame_ptr.add(PYFRAME_LOCALS_CELLS_STACK_OFFSET)
            as *const *mut pyre_object::FixedObjectArray)
    };
    // A pushed in-flight item (#32 S2) needs one slot above the `live` flushed
    // slots; require the capacity up front so a decline happens BEFORE any
    // frame mutation (all-or-nothing).
    let need = if push.is_some() { live + 1 } else { live };
    if arr_ptr.is_null() || unsafe { &*arr_ptr }.as_slice().len() < need {
        return decline("frame stack array too small / null");
    }
    // Commit one slot at a time, re-reading the shadow entry per slot:
    // boxing an Int/Float slot allocates and may trigger a minor
    // collection.  A nursery Ref already written into the detached frame
    // array is forwarded only while the array is in the remembered set,
    // and each minor consumes that entry, so re-arm the barrier after
    // every store — before the next iteration's boxing can allocate.
    for abs in 0..live {
        let boxed = if abs >= nlocals && have_overrides {
            // The override is the authoritative caller CALL stack.  Its
            // mid-opcode slots need not exist in the virtualizable shadow.
            stack_override_at(abs).expect("validated stack override")
        } else {
            let Some((_opref, value)) = ctx.virtualizable_entry_at(base + abs) else {
                return decline("no shadow entry for a live slot (commit)");
            };
            boxed_slot_value_for_type(Type::Ref, &value)
        };
        unsafe {
            (*arr_ptr).as_mut_slice()[abs] = boxed;
        }
        frame_array_write_barrier(frame as *mut u8, arr_ptr);
    }
    unsafe {
        let pf = &mut *(frame as *mut PyFrame);
        pf.valuestackdepth = end_vsd;
        pf.last_instr = resume_py_pc as isize - 1;
    }
    // #32: deliver the in-flight FOR_ITER item.  The flush wrote the
    // FOR_ITER-header operand stack (the iterator on TOS) into slots
    // `0..end_vsd`; the continue arm keeps the iterator and pushes the
    // consumed item above it (codewriter FOR_ITER continue arm,
    // `opcode_for_iter` never pops the iterator), so a single write at
    // `end_vsd` lands the item where the body's first opcode expects TOS.
    // `body_pc` is the FOR_ITER `orgpc + 1`; resume there so the FOR_ITER is
    // not re-executed (which would re-advance the iterator).  The slot at
    // `end_vsd` is within capacity (the early `need = live + 1` validation).
    if let Some((item, body_pc)) = push {
        unsafe {
            (*arr_ptr).as_mut_slice()[end_vsd] = item;
            let pf = &mut *(frame as *mut PyFrame);
            pf.valuestackdepth = end_vsd + 1;
            pf.last_instr = body_pc as isize - 1;
        }
    }
    frame_array_write_barrier(frame as *mut u8, arr_ptr);
    true
}

/// Write back ONLY the locals/cells region (absolute slots `0..nlocals`) from
/// the virtualizable shadow, leaving the operand-stack region,
/// `valuestackdepth` and `last_instr` untouched.
///
/// `virtualizable.py write_boxes` writes the whole array on every
/// force, with no way to decline: a callee handed the frame reads its
/// fastlocals straight out of the array (`pyframe.py fast2locals`), and
/// an unforced array of nulls renders as an EMPTY `f_locals` mapping — a wrong
/// answer, not a stale one.  The merge-point flush cannot carry that on its
/// own because it is all-or-nothing across the operand stack too, and a
/// mid-expression stack slot reads NULL from the shadow, declining the whole
/// write.  The locals region has no such hazard: a NULL local is a legitimate
/// unbound local.
///
/// Resume-pc authority stays with the full flush — this writes state only, so
/// the caller keeps the escape-flush undo armed and a legacy replay from the
/// trace entry still re-enters the pre-flush frame.
/// Report-only: the locals slots a shadow->frame publication is about to give a
/// different value.
///
/// A residual that writes fastlocals writes the frame the interpreter holds,
/// while the walk reads its own copy of the virtualizable, so a slot listed
/// here names a write this publication DESTROYS.  Empty unless
/// `PYRE_FBW_DEBUG_ABORT` is set, so a production build pays one bool.
fn clobbered_locals_slots(
    ctx: &TraceCtx,
    arr_ptr: *const pyre_object::FixedObjectArray,
    base: usize,
    nlocals: usize,
) -> Vec<usize> {
    if !crate::jitcode_dispatch::fbw_debug_abort_enabled() {
        return Vec::new();
    }
    (0..nlocals)
        .filter(|&abs| {
            ctx.virtualizable_entry_at(base + abs)
                .is_some_and(|(_, v)| {
                    let live = unsafe { (*arr_ptr).as_slice()[abs] };
                    live != boxed_slot_value_for_type(Type::Ref, &v)
                })
        })
        .collect()
}

pub(crate) fn flush_locals_region_to_frame(ctx: &TraceCtx, mut frame: usize) -> bool {
    if frame == 0 {
        return false;
    }
    let Some(nlocals) = concrete_nlocals(frame) else {
        return false;
    };
    let Some(info) = ctx.virtualizable_info() else {
        return false;
    };
    let base = info.num_static_extra_boxes;
    // Validation pass first: it allocates nothing, so a missing entry leaves
    // the frame untouched (same all-or-nothing discipline as the full flush).
    // `Value::Void` is the shadow's "no concrete half" sentinel, not a NULL
    // local: writing it back would box to `PY_NULL` and DESTROY the slot the
    // walk is holding in a register.  Decline instead.
    //
    // `Ref(NO_CONCRETE)` is the same answer in the other representation — "the
    // shadow has no concrete pointer for this slot" — and it is worse to write
    // back, because `boxed_slot_value_for_type` maps `Ref(r)` straight to
    // `r.as_usize()`: the slot would receive `usize::MAX - 1` as if it were an
    // object pointer.
    for abs in 0..nlocals {
        match ctx.virtualizable_entry_at(base + abs) {
            Some((_, Value::Void)) | None => return false,
            Some((_, Value::Ref(gc))) if gc == majit_ir::GcRef::NO_CONCRETE => return false,
            Some(_) => {}
        }
    }
    let frame_ptr = frame as *const u8;
    let arr_ptr = unsafe {
        *(frame_ptr.add(PYFRAME_LOCALS_CELLS_STACK_OFFSET)
            as *const *mut pyre_object::FixedObjectArray)
    };
    if arr_ptr.is_null() || unsafe { &*arr_ptr }.as_slice().len() < nlocals {
        return false;
    }
    let clobbered = clobbered_locals_slots(ctx, arr_ptr, base, nlocals);
    for abs in 0..nlocals {
        let Some((_opref, value)) = ctx.virtualizable_entry_at(base + abs) else {
            return false;
        };
        let boxed = boxed_slot_value_for_type(Type::Ref, &value);
        // Boxing an Int/Float slot allocates. A JIT-created frame can be
        // nursery-resident, so a minor collection here moves it; store
        // through the frame's current address and its current array.
        // The forwarding word is readable only until the next minor
        // collection, so carry the current address into the next slot.
        frame = pyre_object::gc_hook::try_gc_current_object_address(
            frame as pyre_object::gc_hook::GCREF,
        ) as usize;
        let arr_ptr = unsafe {
            *((frame as *const u8).add(PYFRAME_LOCALS_CELLS_STACK_OFFSET)
                as *const *mut pyre_object::FixedObjectArray)
        };
        unsafe {
            (*arr_ptr).as_mut_slice()[abs] = boxed;
        }
        // Each minor collection consumes the array's remembered-set entry,
        // so re-arm per store.
        frame_array_write_barrier(frame as *mut u8, arr_ptr);
    }
    if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
        eprintln!("[fbw-flush] locals-region written: nlocals={nlocals} clobbered={clobbered:?}");
    }
    true
}

/// Write one local when its concrete value is a real object.
///
/// `Value::Void` and `Ref(NO_CONCRETE)` are skipped: boxing either would
/// store a null or a sentinel over the slot.  Returns the frame's current
/// address after the store (see [`store_boxed_frame_local`]); a caller that
/// touches the frame again must use it instead of `frame`.
pub(crate) fn store_frame_local_value(frame: usize, abs: usize, value: &Value) -> Option<usize> {
    if frame == 0 || matches!(value, Value::Void) {
        return None;
    }
    if matches!(value, Value::Ref(gc) if *gc == majit_ir::GcRef::NO_CONCRETE) {
        return None;
    }
    let nlocals = concrete_nlocals(frame)?;
    if abs >= nlocals {
        return None;
    }
    let frame_ptr = frame as *const u8;
    let arr_ptr = unsafe {
        *(frame_ptr.add(PYFRAME_LOCALS_CELLS_STACK_OFFSET)
            as *const *mut pyre_object::FixedObjectArray)
    };
    if arr_ptr.is_null() || unsafe { &*arr_ptr }.as_slice().len() <= abs {
        return None;
    }
    store_boxed_frame_local(frame, abs, value)
}

/// Boxing an Int/Float local allocates. A nursery-resident frame moves in
/// that minor collection, so the pre-allocation locals array and frame are
/// both stale. Re-read the forwarded frame, then store and barrier that
/// address. Returns the frame's current address: the forwarding word is only
/// readable until the next minor collection reuses the old slot, so a caller
/// that stores again must pass this address, not the one it started with.
fn store_boxed_frame_local(frame: usize, abs: usize, value: &Value) -> Option<usize> {
    let boxed = boxed_slot_value_for_type(Type::Ref, value);
    let frame_now =
        pyre_object::gc_hook::try_gc_current_object_address(frame as pyre_object::gc_hook::GCREF)
            as usize;
    // A reused nursery slot's forwarding word is the debug fill. That
    // address is not a live object; writing through it faults.
    if frame_now == 0
        || !pyre_object::gc_hook::try_gc_owns_object(frame_now as pyre_object::gc_hook::GCREF)
    {
        return None;
    }
    let arr_ptr = unsafe {
        *((frame_now as *const u8).add(PYFRAME_LOCALS_CELLS_STACK_OFFSET)
            as *const *mut pyre_object::FixedObjectArray)
    };
    if arr_ptr.is_null() || unsafe { &*arr_ptr }.as_slice().len() <= abs {
        return None;
    }
    unsafe {
        (*arr_ptr).as_mut_slice()[abs] = boxed;
    }
    frame_array_write_barrier(frame_now as *mut u8, arr_ptr);
    Some(frame_now)
}

/// How many operands the CALL at `call_py_pc` sits on top of: the caller's
/// static stack height there minus the call's own `[callable, null_or_self,
/// args...]`.  Zero for a statement-position call, positive for an
/// expression-position one (`total += f(x)` keeps `total` below the call).
/// `None` when the two disagree, i.e. the residual's operand list does not
/// model this call shape.
pub(crate) fn outer_call_operands_below(
    frame: usize,
    call_py_pc: usize,
    post_call_py_pc: usize,
    call_stack_len: usize,
) -> Option<usize> {
    // `try_commit_midbody_abort_inner` calls this before its own
    // `cf_addr == 0` decline, so guard here rather than relying on callers to
    // order their checks — matching `can_flush_walk_end_state_after_outer_call`.
    if frame == 0 {
        return None;
    }
    let frame_ptr = frame as *const u8;
    let w_code =
        unsafe { *(frame_ptr.add(crate::frame_layout::PYFRAME_PYCODE_OFFSET) as *const *const ()) };
    if w_code.is_null() {
        return None;
    }
    let raw_code = unsafe {
        pyre_interpreter::w_code_get_ptr(w_code as PyObjectRef)
            as *const pyre_interpreter::CodeObject
    };
    if raw_code.is_null() {
        return None;
    }
    let depths = crate::liveness::liveness_for(raw_code).depth_at_py_pc();
    let at_call = usize::from(depths.get(call_py_pc).copied()?);
    let after_call = usize::from(depths.get(post_call_py_pc).copied()?);
    let below = at_call.checked_sub(call_stack_len)?;
    // The call leaves exactly one value where its operands were.
    (after_call == below + 1).then_some(below)
}

/// Allocation-free preflight for the post-CALL delivery. Every gate is checked
/// before the rebuilt callee runs, so a later decline cannot replay effects the
/// plain interpreter already committed.
///
/// `below` is [`outer_call_operands_below`]'s slice of the caller's operand
/// stack — empty for a statement-position call.  Leg 4's payload records only
/// the call's own operand count, so an expression-position call declines
/// and WalkAbort continues the live framestack.
pub(crate) fn can_flush_walk_end_state_after_outer_call(
    ctx: &TraceCtx,
    frame: usize,
    call_py_pc: usize,
    post_call_py_pc: usize,
    call_stack_len: usize,
    below: &[PyObjectRef],
) -> bool {
    if frame == 0 {
        return false;
    }
    let Some(nlocals) = concrete_nlocals(frame) else {
        return false;
    };
    let Some(info) = ctx.virtualizable_info() else {
        return false;
    };
    let frame_ptr = frame as *const u8;
    let w_code =
        unsafe { *(frame_ptr.add(crate::frame_layout::PYFRAME_PYCODE_OFFSET) as *const *const ()) };
    if w_code.is_null() {
        return false;
    }
    let raw_code = unsafe {
        pyre_interpreter::w_code_get_ptr(w_code as PyObjectRef)
            as *const pyre_interpreter::CodeObject
    };
    if raw_code.is_null() {
        return false;
    }
    let Some(want_below) =
        outer_call_operands_below(frame, call_py_pc, post_call_py_pc, call_stack_len)
    else {
        return false;
    };
    if want_below != below.len() {
        return false;
    }
    // `below` is not restricted to ordinary Python object pointers.  CPython
    // 3.14 LOAD_SPECIAL leaves `[exit_func, NULL]` below the body of a `with`
    // statement, and a nested CALL therefore legitimately carries that NULL
    // through this post-CALL handoff.  The carrier has already proved every
    // slot before it becomes a raw `PyObjectRef`: NULL is admitted only from
    // an inline `ConstPtr(NULL)` (RPython history.py `CONST_NULL`) or the
    // CALL layout's named null-or-self slot.  Rejecting it here collapses that
    // distinction again and forces the effectful callee prologue to replay.
    let base = info.num_static_extra_boxes;
    if let Some(abs) = (0..nlocals).find(|&abs| ctx.virtualizable_entry_at(base + abs).is_none()) {
        if crate::jitcode_dispatch::fbw_debug_abort_enabled() {
            eprintln!("[after-call] outer local {abs} has no shadow entry");
        }
        return false;
    }
    let arr_ptr = unsafe {
        *(frame_ptr.add(PYFRAME_LOCALS_CELLS_STACK_OFFSET)
            as *const *mut pyre_object::FixedObjectArray)
    };
    !arr_ptr.is_null() && unsafe { &*arr_ptr }.as_slice().len() >= nlocals + below.len() + 1
}

/// Take a copy of `frame`'s locals so a caller that publishes over them can put
/// the frame back exactly as it found it.  Returned as raw words: boxing a slot
/// during the publish can collect, so the copy has to be registered as resume
/// roots (`push_resume_ref_roots`) for the duration.
///
/// Restoring the FRAME is not the same as undoing the drive.  This and
/// [`capture_frame_scalars`] together cover the whole mutable virtualizable
/// surface — the only vable fields with a production emit site are `last_instr`
/// (index 0) and `valuestackdepth` (2), plus this array; `setfield_vable_r`/`_f`
/// have dispatch arms and no emitter.  A decline taken after
/// `drive_single_frame_blackhole` still leaves every heap effect the region's
/// residual calls made, and the replay repeats them — measured on
/// `getframe_root_loop_force_blackhole_crn_nonidempotent` as one extra list
/// entry per declined drive while the locals came back correct.  So this is an
/// undo for PRE-drive failures; it does not license a post-drive one.
///
/// Upstream needs no counterpart. `resume.py`'s virtualizable write-back
/// (`VirtualizableInfo.write_from_resume_data`) runs on a per-call `MIFrame`
/// whose values RPython's GC sees through ordinary object references, so the
/// question of publishing over a live frame and taking it back never arises;
/// carrying the old slots as raw words, and rooting them by hand, is what
/// replaces that ownership.
pub(crate) fn capture_frame_locals(frame: usize) -> Option<Vec<i64>> {
    if frame == 0 {
        return None;
    }
    let nlocals = concrete_nlocals(frame)?;
    let arr_ptr = unsafe {
        *((frame as *const u8).add(PYFRAME_LOCALS_CELLS_STACK_OFFSET)
            as *const *mut pyre_object::FixedObjectArray)
    };
    if arr_ptr.is_null() {
        return None;
    }
    let slots = unsafe { &*arr_ptr }.as_slice();
    if slots.len() < nlocals {
        return None;
    }
    Some(slots[..nlocals].iter().map(|&o| o as i64).collect())
}

/// Put back what [`capture_frame_locals`] took.
pub(crate) fn restore_frame_locals(frame: usize, saved: &[i64]) {
    if frame == 0 {
        return;
    }
    let arr_ptr = unsafe {
        *((frame as *const u8).add(PYFRAME_LOCALS_CELLS_STACK_OFFSET)
            as *const *mut pyre_object::FixedObjectArray)
    };
    if arr_ptr.is_null() {
        return;
    }
    let len = unsafe { &*arr_ptr }.as_slice().len();
    for (abs, &value) in saved.iter().enumerate().take(len) {
        unsafe {
            (*arr_ptr).as_mut_slice()[abs] = value as usize as PyObjectRef;
        }
    }
    frame_array_write_barrier(frame as *mut u8, arr_ptr);
}

/// The active operand-stack half of a tracer snapshot that must be published
/// to its live virtualizable before an overlong trace is continued in the
/// blackhole.
///
/// RPython has no separate copy step here: `convert_and_run_from_pyjitpl`
/// copies the already-advanced MIFrame, whose red virtualizable is the one
/// frame object carrying that call's locals and stack.  Pyre concretely steps
/// a detached `snapshot_for_tracing`, so the corresponding state has to cross
/// back to that same live frame explicitly.  The raw Ref words are exposed as
/// a mutable slice because publishing locals can allocate; callers root the
/// slice so nursery forwarding is reflected before the stack is stored.
pub(crate) struct CapturedFrameStack {
    stack_base: usize,
    scalars: FrameScalars,
    roots: Vec<i64>,
}

impl CapturedFrameStack {
    pub(crate) fn roots_mut(&mut self) -> &mut [i64] {
        self.roots.as_mut_slice()
    }

    /// Diagnostic copy of the captured slots.
    pub(crate) fn roots_snapshot(&self) -> Vec<i64> {
        self.roots.clone()
    }
}

fn frame_stack_transfer_shape(
    snapshot: usize,
    live_frame: usize,
) -> Option<(
    usize,
    usize,
    *const pyre_object::FixedObjectArray,
    *mut pyre_object::FixedObjectArray,
)> {
    if snapshot == 0 || live_frame == 0 {
        return None;
    }
    let snapshot_base = concrete_nlocals(snapshot)?;
    let live_base = concrete_nlocals(live_frame)?;
    if snapshot_base != live_base {
        return None;
    }
    let snapshot_depth = concrete_stack_depth(snapshot)?;
    if snapshot_depth < snapshot_base {
        return None;
    }
    let snapshot_arr = unsafe {
        *((snapshot as *const u8).add(PYFRAME_LOCALS_CELLS_STACK_OFFSET)
            as *const *const pyre_object::FixedObjectArray)
    };
    let live_arr = unsafe {
        *((live_frame as *const u8).add(PYFRAME_LOCALS_CELLS_STACK_OFFSET)
            as *const *mut pyre_object::FixedObjectArray)
    };
    if snapshot_arr.is_null()
        || live_arr.is_null()
        || unsafe { &*snapshot_arr }.as_slice().len() < snapshot_depth
        || unsafe { &*live_arr }.as_slice().len() < snapshot_depth
    {
        return None;
    }
    Some((snapshot_base, snapshot_depth, snapshot_arr, live_arr))
}

/// Allocation-free preflight for [`capture_frame_stack_for_publish`].
pub(crate) fn can_publish_frame_stack(snapshot: usize, live_frame: usize) -> bool {
    frame_stack_transfer_shape(snapshot, live_frame).is_some()
}

/// Copy the tracer snapshot's active operand stack into a rootable carrier.
/// The destination is checked at capture time and checked again at commit, so
/// no partial stack is ever published if frame identity or layout changes.
pub(crate) fn capture_frame_stack_for_publish(
    snapshot: usize,
    live_frame: usize,
) -> Option<CapturedFrameStack> {
    let (stack_base, stack_depth, snapshot_arr, _) =
        frame_stack_transfer_shape(snapshot, live_frame)?;
    let slots = unsafe { &*snapshot_arr }.as_slice();
    Some(CapturedFrameStack {
        stack_base,
        scalars: capture_frame_scalars(snapshot)?,
        roots: slots[stack_base..stack_depth]
            .iter()
            .map(|&value| value as i64)
            .collect(),
    })
}

/// Build the same carrier from the walker's operand-stack MIRROR instead of
/// the tracing snapshot's slot array.
///
/// [`capture_frame_stack_for_publish`] reads the detached snapshot frame, which
/// only describes the walk for a coordinate the walk's vable writes reached:
/// `setarrayitem_vable_via_metainterp` mirrors into a concrete frame through
/// `current_inline_vable_target`, i.e. only for the frame an INLINE level owns.
/// A root walk keeps its virtualizable symbolic, so that array still holds the
/// pre-walk stack, and publishing it resumes the blackhole on a stack from
/// before the walk ran.  Locals are unaffected — `write_back_outer_locals`
/// takes those from the virtualizable shadow.
///
/// The mirror reflects the depth ON ENTRY to the opcode at `py_pc`, so it pairs
/// with `last_instr = py_pc - 1`, the same coordinate
/// [`flush_walk_end_state_to_frame_with_full_stack`] stores for its own
/// full-stack override.  The height is checked against the forward analysis for
/// the same reason that override is: a mirror describing another coordinate
/// must decline, not publish a shifted stack.
pub(crate) fn capture_frame_stack_from_mirror(
    live_frame: usize,
    py_pc: usize,
    stack: &[PyObjectRef],
) -> Option<CapturedFrameStack> {
    let stack_base = concrete_nlocals(live_frame)?;
    let valuestackdepth = stack_base.checked_add(stack.len())?;
    let live_arr = unsafe {
        *((live_frame as *const u8).add(PYFRAME_LOCALS_CELLS_STACK_OFFSET)
            as *const *const pyre_object::FixedObjectArray)
    };
    if live_arr.is_null() || unsafe { &*live_arr }.as_slice().len() < valuestackdepth {
        return None;
    }
    let w_code = unsafe {
        *((live_frame as *const u8).add(crate::frame_layout::PYFRAME_PYCODE_OFFSET)
            as *const *const ())
    };
    if w_code.is_null() {
        return None;
    }
    let raw_code = unsafe {
        pyre_interpreter::w_code_get_ptr(w_code as PyObjectRef)
            as *const pyre_interpreter::CodeObject
    };
    let depth = crate::liveness::liveness_for(raw_code)
        .depth_at_py_pc()
        .get(py_pc)
        .copied()?;
    if depth as usize != stack.len() {
        return None;
    }
    Some(CapturedFrameStack {
        stack_base,
        scalars: FrameScalars {
            last_instr: py_pc as isize - 1,
            valuestackdepth,
        },
        roots: stack.iter().map(|&value| value as i64).collect(),
    })
}

/// Commit a captured post-step operand stack and its Python-frame coordinate
/// to the live red frame.  This performs no allocation: once the caller has
/// published locals and reached this point, the blackhole handoff is
/// irrevocable.
pub(crate) fn publish_captured_frame_stack(
    live_frame: usize,
    captured: &CapturedFrameStack,
) -> bool {
    let Some(live_base) = concrete_nlocals(live_frame) else {
        return false;
    };
    if live_base != captured.stack_base
        || captured.scalars.valuestackdepth != live_base + captured.roots.len()
    {
        return false;
    }
    let live_arr = unsafe {
        *((live_frame as *const u8).add(PYFRAME_LOCALS_CELLS_STACK_OFFSET)
            as *const *mut pyre_object::FixedObjectArray)
    };
    if live_arr.is_null()
        || unsafe { &*live_arr }.as_slice().len() < captured.scalars.valuestackdepth
    {
        return false;
    }
    unsafe {
        let slots = (*live_arr).as_mut_slice();
        for (offset, &value) in captured.roots.iter().enumerate() {
            slots[live_base + offset] = value as usize as PyObjectRef;
        }
    }
    restore_frame_scalars(live_frame, captured.scalars);
    frame_array_write_barrier(live_frame as *mut u8, live_arr);
    true
}

/// The scalar half of the frame image [`capture_frame_locals`] covers.
#[derive(Clone, Copy)]
pub(crate) struct FrameScalars {
    last_instr: isize,
    valuestackdepth: usize,
}

/// Take the two frame words a blackhole drive moves as it replays.  It reaches
/// them through `setfield_vable_i` against the frame register, which is the
/// live frame, so a caller that may still hand that frame back to the
/// interpreter has to take these beside the slots: the interpreter derives its
/// next opcode from `last_instr + 1`, and reads the operand stack at
/// `valuestackdepth`.  Restoring only the locals leaves the two disagreeing.
pub(crate) fn capture_frame_scalars(frame: usize) -> Option<FrameScalars> {
    if frame == 0 {
        return None;
    }
    Some(FrameScalars {
        last_instr: unsafe {
            *((frame + crate::frame_layout::PYFRAME_LAST_INSTR_OFFSET) as *const isize)
        },
        valuestackdepth: concrete_stack_depth(frame)?,
    })
}

/// Put back what [`capture_frame_scalars`] took.
pub(crate) fn restore_frame_scalars(frame: usize, saved: FrameScalars) {
    if frame == 0 {
        return;
    }
    unsafe {
        *((frame + crate::frame_layout::PYFRAME_LAST_INSTR_OFFSET) as *mut isize) =
            saved.last_instr;
    }
    set_concrete_stack_depth(frame, saved.valuestackdepth);
}

/// Materialize the outer frame's locals from the virtualizable shadow.
/// Callers must run their complete preflight before executing a rebuilt
/// callee; after that point a failure would make replay unsafe.
fn outer_locals_publish_target(
    ctx: &TraceCtx,
    frame: usize,
) -> Option<(usize, *mut pyre_object::FixedObjectArray, usize)> {
    if frame == 0 {
        return None;
    }
    let Some(nlocals) = concrete_nlocals(frame) else {
        return None;
    };
    let Some(info) = ctx.virtualizable_info() else {
        return None;
    };
    let frame_ptr = frame as *mut u8;
    let arr_ptr = unsafe {
        *(frame_ptr.add(PYFRAME_LOCALS_CELLS_STACK_OFFSET)
            as *const *mut pyre_object::FixedObjectArray)
    };
    if arr_ptr.is_null() || unsafe { &*arr_ptr }.as_slice().len() < nlocals {
        return None;
    }
    let base = info.num_static_extra_boxes;
    // Every slot has to resolve before the first store, the way the merge-point
    // flush validates ahead of its commit loop: bailing partway through leaves
    // the frame carrying a mix of walk-current and pre-walk locals, which is
    // neither of the two states a caller can recover from.
    if (0..nlocals).any(|abs| ctx.virtualizable_entry_at(base + abs).is_none()) {
        return None;
    }
    Some((nlocals, arr_ptr, base))
}

/// Read-only half of [`write_back_outer_locals`].
///
/// `ABORT_TOO_LONG` has already executed the current opcode when it decides to
/// stop tracing. It uses this census before publishing a blackhole image so a
/// later adoption cannot decline into trace-entry replay after those effects.
pub(crate) fn can_write_back_outer_locals(ctx: &TraceCtx, frame: usize) -> bool {
    outer_locals_publish_target(ctx, frame).is_some()
}

pub(crate) fn write_back_outer_locals(ctx: &TraceCtx, frame: usize) -> bool {
    let Some((nlocals, _arr_ptr, base)) = outer_locals_publish_target(ctx, frame) else {
        return false;
    };
    // Boxing an Int/Float slot allocates; the detached frame array is
    // forwarded only while it is in the remembered set, and each minor
    // consumes that entry, so re-arm the barrier after every store.
    let mut frame = frame;
    for abs in 0..nlocals {
        let (_opref, value) = ctx
            .virtualizable_entry_at(base + abs)
            .expect("outer-locals target was completely preflighted");
        let Some(frame_now) = store_boxed_frame_local(frame, abs, &value) else {
            return false;
        };
        frame = frame_now;
    }
    let arr_ptr = unsafe {
        *((frame as *const u8).add(PYFRAME_LOCALS_CELLS_STACK_OFFSET)
            as *const *mut pyre_object::FixedObjectArray)
    };
    frame_array_write_barrier(frame as *mut u8, arr_ptr);
    true
}

/// `_setup_return_value_r` parity for the outer half of a two-frame abort:
/// install the rebuilt callee's return value and resume after the CALL.
pub(crate) fn flush_walk_end_state_after_outer_call(
    ctx: &TraceCtx,
    frame: usize,
    call_py_pc: usize,
    post_call_py_pc: usize,
    call_stack_len: usize,
    below: &[PyObjectRef],
    retval: PyObjectRef,
) -> bool {
    if !can_flush_walk_end_state_after_outer_call(
        ctx,
        frame,
        call_py_pc,
        post_call_py_pc,
        call_stack_len,
        below,
    ) {
        return false;
    }
    let Some(nlocals) = concrete_nlocals(frame) else {
        return false;
    };
    let frame_ptr = frame as *mut u8;
    let arr_ptr = unsafe {
        *(frame_ptr.add(PYFRAME_LOCALS_CELLS_STACK_OFFSET)
            as *const *mut pyre_object::FixedObjectArray)
    };
    // Restore the operands the CALL sat on top of, then land the
    // nursery-resident result above them, arming the barrier before
    // `write_back_outer_locals` boxes locals (an allocation that can
    // collect the just-stored refs if the array is not remembered).
    unsafe {
        let slots = (*arr_ptr).as_mut_slice();
        if slots.len() < nlocals + below.len() + 1 {
            return false;
        }
        slots[nlocals..nlocals + below.len()].copy_from_slice(below);
        slots[nlocals + below.len()] = retval;
    }
    frame_array_write_barrier(frame as *mut u8, arr_ptr);
    if !write_back_outer_locals(ctx, frame) {
        return false;
    }
    unsafe {
        let pf = &mut *(frame as *mut PyFrame);
        pf.valuestackdepth = nlocals + below.len() + 1;
        pf.last_instr = post_call_py_pc as isize - 1;
    }
    frame_array_write_barrier(frame as *mut u8, arr_ptr);
    true
}

pub(crate) fn looks_like_heap_ref(value: PyObjectRef) -> bool {
    let addr = value as usize;
    let word_align = std::mem::align_of::<usize>() - 1;
    addr >= 0x1_0000 && addr < ((1u64 << 56) as usize) && (addr & word_align) == 0
}

pub(crate) fn extract_concrete_typed_value(slot_type: Type, value: PyObjectRef) -> Value {
    match slot_type {
        Type::Int => {
            if value.is_null() {
                Value::Int(0)
            } else if looks_like_heap_ref(value) && unsafe { is_int(value) } {
                Value::Int(unsafe { w_int_get_value(value) })
            } else {
                Value::Int(value as i64)
            }
        }
        Type::Float => {
            if value.is_null() {
                Value::Float(0.0)
            } else if looks_like_heap_ref(value) && unsafe { is_float(value) } {
                Value::Float(unsafe { pyre_object::floatobject::w_float_get_value(value) })
            } else {
                Value::Float(f64::from_bits(value as u64))
            }
        }
        Type::Ref | Type::Void => Value::Ref(majit_ir::GcRef(value as usize)),
    }
}

pub(crate) fn concrete_slot_types(
    frame: usize,
    num_locals: usize,
    valuestackdepth: usize,
) -> Vec<Type> {
    let stack_only = valuestackdepth.saturating_sub(num_locals);
    let mut types = Vec::with_capacity(num_locals + stack_only);
    for idx in 0..num_locals {
        types.push(
            concrete_stack_value(frame, idx)
                .map(concrete_virtualizable_slot_type)
                .unwrap_or(Type::Ref),
        );
    }
    for stack_idx in 0..stack_only {
        types.push(
            concrete_stack_value(frame, num_locals + stack_idx)
                .map(concrete_virtualizable_slot_type)
                .unwrap_or(Type::Ref),
        );
    }
    types
}

impl PyreMeta {
    /// Heap capacity of `locals_cells_stack_w`, distinct from the live
    /// `valuestackdepth`.
    pub fn array_stack_only_depth(&self) -> usize {
        self.array_capacity.saturating_sub(self.num_locals)
    }
}

pub(crate) fn boxed_slot_i64_for_type(slot_type: Type, raw: i64) -> PyObjectRef {
    match slot_type {
        Type::Int => w_int_new(raw),
        Type::Float => pyre_object::floatobject::w_float_new(f64::from_bits(raw as u64)),
        Type::Ref | Type::Void => raw as PyObjectRef,
    }
}

/// virtualizable.py `lst[j] = reader.load_next_value_of_type(ARRAYITEMTYPE)`:
/// pyre's `locals_cells_stack_w` array item type is GCREF, so every write to
/// a frame slot must produce a boxed PyObjectRef regardless of the label's
/// argument type (a Body-Label Int OpRef still stores a W_IntObject in the
/// array). `slot_type` is retained for call-site symmetry but no longer
/// gates the boxing decision.
pub(crate) fn boxed_slot_value_for_type(_slot_type: Type, value: &Value) -> PyObjectRef {
    match value {
        Value::Int(v) => w_int_new(*v),
        Value::Float(v) => pyre_object::floatobject::w_float_new(*v),
        Value::Ref(r) => r.as_usize() as PyObjectRef,
        Value::Void => PY_NULL,
    }
}

pub(crate) fn fail_arg_opref_for_typed_value(ctx: &mut TraceCtx, value: Value) -> OpRef {
    match value {
        Value::Int(v) => ctx.const_int(v),
        Value::Float(v) => ctx.const_int(v.to_bits() as i64),
        Value::Ref(r) => ctx.const_ref(r.as_usize() as i64),
        Value::Void => ctx.const_ref(PY_NULL as i64),
    }
}

#[allow(dead_code)]
pub(crate) fn frame_callable_arg_types(nargs: usize) -> Vec<Type> {
    let mut types = Vec::with_capacity(2 + nargs);
    types.push(Type::Ref);
    types.push(Type::Ref);
    for _ in 0..nargs {
        types.push(Type::Ref);
    }
    types
}

#[allow(dead_code)]
pub(crate) fn one_arg_callee_frame_helper(
    arg_type: Type,
    is_self_recursive: bool,
) -> (*const (), Vec<Type>) {
    match (is_self_recursive, arg_type) {
        (true, Type::Int) => (
            crate::callbacks::get().jit_create_self_recursive_callee_frame_1_raw_int,
            vec![Type::Ref, Type::Int],
        ),
        (true, _) => (
            crate::callbacks::get().jit_create_self_recursive_callee_frame_1,
            vec![Type::Ref, Type::Ref],
        ),
        (false, Type::Int) => (
            crate::callbacks::get().jit_create_callee_frame_1_raw_int,
            vec![Type::Ref, Type::Ref, Type::Int],
        ),
        (false, _) => (
            crate::callbacks::get().jit_create_callee_frame_1,
            vec![Type::Ref, Type::Ref, Type::Ref],
        ),
    }
}

#[allow(dead_code)]
pub(crate) fn fail_arg_types_for_virtualizable_state(len: usize) -> Vec<Type> {
    let n = crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
    crate::virtualizable_gen::virt_live_value_types(len.saturating_sub(n))
}

/// `pyjitpl.py MIFrame.copy_constants` line-by-line port.
///
/// ```python
/// def copy_constants(self, registers, constants, targetindex, ConstClass):
///     num_regs_and_consts = targetindex + len(constants)
///     if registers is None or len(registers) < num_regs_and_consts:
///         registers = [missing] * num_regs_and_consts
///     elif not we_are_translated():
///         for i in range(len(registers)):
///             registers[i] = missing
///     for i in range(len(constants)):
///         registers[targetindex] = ConstClass(constants[i])
///         targetindex += 1
///     return registers
/// ```
///
/// `ConstClass(constants[i])` is supplied as the closure `mint`. The
/// `elif not we_are_translated()` debug wipe is upstream-only
/// (translation-time fill of stale slots with `missing`); pyre always
/// runs the translated path. `MIFrame.setup` can be re-invoked on
/// `free_frames_list` reused frames (`pyjitpl.py`), and each call
/// re-mints `ConstClass(constants[i])` unconditionally — the fill loop
/// here matches that.
///
/// TODOs (pre-main divergences kept for now):
///
/// 1. Grow-in-place via `resize(num_regs_and_consts, NONE)` instead of
///    full replacement `[missing] * num_regs_and_consts` (the
///    `if registers is None or len(registers) < ...` arm at
///    `pyjitpl.py`). Convergence is blocked by Adaptation 2 below:
///    callers like `trace_opcode.rs` push callee args into
///    `sym.registers_r[0..args.len()]` BEFORE invoking this helper, and
///    a full replacement would zero those slots. Migrating to
///    full-replace requires reordering the trace_opcode callee setup
///    so the regalloc-color slot writes happen after
///    `setup_kind_register_banks`, plus matching opcode-dispatch reads
///    against the post-color slots rather than sequential
///    `[0..nlocals)`. That is a multi-file restructuring outside the
///    scope of this helper.
///
/// 2. `registers_r` carries both pyre's semantic locals/stack mirror
///    AND the post-regalloc-color Ref bank (see the wider doc on
///    `setup_kind_register_banks` below). RPython's `registers_r` is a
///    pure kind-specific Ref bank — semantic locals go through
///    opcode-dispatch reads of `registers_r[color]`. Pyre's dual-role
///    is what makes Adaptation 1 load-bearing; retiring one entails
///    retiring both.
///
/// 3. Ref constants land via `ctx.const_ref(val)` — a plain Ref
///    constant — instead of upstream's `ConstPtrJitCode` subclass
///    (`history.py`) that carries an `opencoder_index = -1`
///    cache field. Upstream uses that field to fast-path the trace
///    encoder's pointer dedup at `opencoder.py _cached_const_ptr
///    _cached_const_ptr` (over its `_refs_dict`). Pyre lacks the
///    per-instance cache: every `ctx.const_ref(val)` mints a fresh
///    inline constant OpRef (`TraceCtx::const_ref` returns an
///    `OpRef::const_ptr` — no HashMap-keyed dedup at this
///    layer). The two upstream dedup layers exist independently in
///    pyre:
///      - **Trace encoding** (port of `opencoder.py
///        _cached_const_ptr`): `opencoder.rs`'s `_encode_ptr` dedups
///        by address via `_refs_dict: HashMap<u64, u32>`, mirroring
///        upstream's `_refs_dict` lookup. The
///        `ConstPtrJitCode.opencoder_index` cache that bypasses the
///        dict has no pyre counterpart, but the underlying dedup is
///        preserved.
///      - **Resume-data numbering** (port of `resume.py:148-181
///        ResumeDataLoopMemo.large_ints / .refs`):
///        `resume.rs::ResumeDataLoopMemo`'s `large_ints` / `refs`
///        memo separately dedups constants when assigning
///        resume numbering tags.
trait ConstantRegisters {
    fn len(&self) -> usize;
    fn resize(&mut self, len: usize, value: OpRef);
    fn set(&mut self, index: usize, value: OpRef);
}

impl ConstantRegisters for Vec<OpRef> {
    fn len(&self) -> usize {
        Vec::len(self)
    }
    fn resize(&mut self, len: usize, value: OpRef) {
        Vec::resize(self, len, value);
    }
    fn set(&mut self, index: usize, value: OpRef) {
        self[index] = value;
    }
}

impl ConstantRegisters for crate::jitcode_dispatch::RegisterList {
    fn len(&self) -> usize {
        Self::len(self)
    }
    fn resize(&mut self, len: usize, value: OpRef) {
        Self::resize(self, len, value);
    }
    fn set(&mut self, index: usize, value: OpRef) {
        Self::set(self, index, value);
    }
}

fn copy_constants<F>(
    registers: &mut impl ConstantRegisters,
    constants: &[i64],
    targetindex: usize,
    mut mint: F,
) where
    F: FnMut(i64) -> OpRef,
{
    let num_regs_and_consts = targetindex + constants.len();
    if registers.len() < num_regs_and_consts {
        registers.resize(num_regs_and_consts, OpRef::NONE);
    }
    for (i, &val) in constants.iter().enumerate() {
        registers.set(targetindex + i, mint(val));
    }
}

impl PyreSym {
    pub(crate) fn new_uninit(frame: OpRef) -> Self {
        Self {
            frame,
            execution_context: OpRef::NONE,
            symbolic_local_types: Vec::new(),
            symbolic_stack_types: Vec::new(),
            pending_next_instr: None,
            locals_cells_stack_array_ref: OpRef::NONE,
            valuestackdepth: 0,
            nlocals: 0,
            bridge_local_oprefs: Default::default(),
            bridge_stack_oprefs: Default::default(),
            bridge_walk_entry_pc: None,
            bridge_walk_entry_jitcode_index: -1,
            bridge_registers_r: Default::default(),
            bridge_local_types: None,
            vable_last_instr: OpRef::NONE,
            vable_pycode: OpRef::NONE,
            vable_valuestackdepth: OpRef::NONE,
            vable_debugdata: OpRef::NONE,
            frame_w_globals: OpRef::NONE,
            vable_array_base: None,
            is_active_vable_owner: false,
            concrete_locals: Vec::new(),
            concrete_stack: Vec::new(),
            // jitcode and concrete_namespace initialized below
            jitcode: null_jitcode() as *const JitCode,
            concrete_namespace: std::ptr::null_mut(),
            is_function_entry_trace: false,
            concrete_execution_context: std::ptr::null(),
            concrete_vable_ptr: std::ptr::null_mut(),
            live_vable_frame_addr: 0,
            last_exc_value: Default::default(),
            class_of_last_exc_is_const: false,
            last_exc_box: OpRef::NONE,
            current_exc_value: std::rc::Rc::new(std::cell::Cell::new(
                pyre_interpreter::eval::get_current_exception(),
            )),
            current_exc_box: OpRef::NONE,
            // RPython pyjitpl.py setup init: registers_X[i] = CONST_NULL for
            // i in num_regs. Sized lazily here — `setup_kind_register_banks`
            // resizes `registers_i` / `registers_f` once the owning JitCode is
            // bound. `registers_r` continues to be driven by the existing
            // semantic-slot logic; the encoder is not yet rewired to
            // per-bank reads for `registers_r`.
            registers_i: Vec::new(),
            registers_r: Default::default(),
            registers_f: Vec::new(),
        }
    }

    /// `pyjitpl.py MIFrame.setup` parity for the per-kind register
    /// files. Sizes `registers_i` and `registers_f` to
    /// `num_regs_and_consts_X`, matching the RPython MIFrame layout where
    /// each bank holds the post-regalloc-color register slots followed by
    /// the constant pool entries. The leading `num_regs_X` slots stay
    /// `OpRef::NONE` (RPython's `CONST_NULL` placeholder); the trailing
    /// `[num_regs_X..num_regs_and_consts_X)` slots are filled with the
    /// constant-pool OpRefs in `pyjitpl.py copy_constants` order:
    ///   - `registers_i[num_regs_i + i]` ← `ctx.const_int(constants_i[i])`
    ///   - `registers_r[num_regs_r + i]` ← `ctx.const_ref(constants_r[i])`
    ///   - `registers_f[num_regs_f + i]` ← `ctx.const_float(constants_f[i])`
    /// `TraceCtx::const_int` / `const_ref` / `const_float` all mint a
    /// fresh inline constant OpRef per call (`OpRef::const_int` /
    /// `const_ptr` / `const_float`), matching RPython
    /// `ConstClass(constants[i])`'s fresh-Box allocation
    /// (`history.py:220/261/307`). Per-value dedup, where it exists
    /// upstream, lives in the resume memo
    /// (`resume.rs::ResumeDataLoopMemo`'s `large_ints` / `refs` fields,
    /// per `resume.py`), not in the constant pool. Re-entering
    /// this helper therefore overwrites the trailing slots with freshly
    /// minted OpRefs across all three banks, matching `copy_constants`
    /// overwrite semantics line-by-line.
    ///
    /// `registers_r` carries the unified locals + stack-tail abstract
    /// register file. The SSA-authoritative live_r work extends the resize to
    /// `registers_r` so its size matches the post-regalloc-color shape
    /// `num_regs_and_consts_r` already in use for `registers_i` /
    /// `registers_f`.
    ///
    /// This helper ports the full upstream `pyjitpl.py
    /// MIFrame.setup` body (resize + `copy_constants`).  Pyre still keeps
    /// `registers_r` as the semantic frame mirror for stack/local writes;
    /// guard capture materializes the post-regalloc-color Ref bank from
    /// that mirror or the virtualizable shadow.  The trailing constant
    /// slots `[num_regs_X, num_regs_and_consts_X)` are filled by this
    /// helper; no production reader consumes them yet (pending the
    /// encoder's constant-bank read path).
    ///
    /// Safe to call when `self.jitcode` points at the thread-local
    /// `null_jitcode()` placeholder — the skeleton's
    /// `num_regs_and_consts_X` values are zero and the constant pools
    /// are empty, which makes both the resize and the constant fill a
    /// no-op.
    pub(crate) fn setup_kind_register_banks(&mut self, ctx: &mut TraceCtx) {
        debug_assert!(!self.jitcode.is_null());
        let (num_regs_i, num_regs_r, num_regs_f, constants_i, constants_r, constants_f) = {
            let jc = unsafe { &*self.jitcode };
            let runtime_jc = &jc.payload.jitcode;
            (
                runtime_jc.num_regs_i() as usize,
                runtime_jc.num_regs_r() as usize,
                runtime_jc.num_regs_f() as usize,
                runtime_jc.constants_i.clone(),
                // Snapshot the pool's current values: `copy_constants`
                // seeds a register file and never writes back, so it wants
                // plain words, not the live interior-mutable slots.
                runtime_jc
                    .constants_r
                    .iter()
                    .map(|slot| slot.get())
                    .collect::<Vec<i64>>(),
                runtime_jc.constants_f.clone(),
            )
        };
        // pyjitpl.py `MIFrame.copy_constants` line-by-line:
        //   num_regs_and_consts = targetindex + len(constants)
        //   if registers is None or len(registers) < num_regs_and_consts:
        //       registers = [missing] * num_regs_and_consts
        //   for i in range(len(constants)):
        //       registers[targetindex] = ConstClass(constants[i])
        //       targetindex += 1
        // The "translation-time-only debug wipe" arm
        // (`elif not we_are_translated()`) has no pyre analogue.
        copy_constants(&mut self.registers_i, &constants_i, num_regs_i, |v| {
            ctx.const_int(v)
        });
        copy_constants(&mut self.registers_r, &constants_r, num_regs_r, |v| {
            ctx.const_ref(v)
        });
        copy_constants(&mut self.registers_f, &constants_f, num_regs_f, |v| {
            ctx.const_float(v)
        });
    }

    /// True when this frame is allowed to mirror writes into the
    /// metainterp-scope `TraceCtx::virtualizable_boxes` cache (RPython
    /// `metainterp.virtualizable_boxes`, pyjitpl.py:1230). Two disjoint
    /// states satisfy the predicate:
    ///
    ///   1. **portal entry** — `is_active_vable_owner == true` after
    ///      `become_active_vable_owner` (which wraps the macro-generated
    ///      `init_vable_indices`). The frame's `locals_cells_stack_w`
    ///      array IS the active virtualizable.
    ///   2. **bridge entry** — `bridge_local_oprefs == Some(...)` after
    ///      `setup_bridge_sym` calls `seed_virtualizable_boxes` to
    ///      repopulate the shadow from resume data. `is_active_vable_owner`
    ///      is cleared (`clear_active_vable`) because the bridge's
    ///      inputarg layout lacks the `[frame, last_instr, pycode,
    ///      valuestackdepth, debugdata]` scalar
    ///      header that `init_vable_indices` assumes (see
    ///      `virtualizable_spec.rs::PYFRAME_VABLE_FIELDS` for the
    ///      four concrete scalar fields selected from
    ///      `interp_jit.py PyFrame._virtualizable_`); the
    ///      frame still owns the shadow
    ///      semantically though.
    ///
    /// Callee inline frames (the retired inline-call path allocated a fresh
    /// `PyreSym::new_uninit`) keep both fields at their defaults and
    /// must NOT mirror into the caller's shadow — their
    /// `nlocals + stack_idx` space is the callee's own, not the
    /// caller's, so writing to `NUM_VABLE_SCALARS + reg_idx` in the
    /// shared `TraceCtx` shadow would corrupt the caller's portal
    /// layout. opencoder.py `_list_of_boxes_virtualizable` treats
    /// `self.virtualizable_boxes` as the single source of truth; this
    /// predicate names the set of syms that are allowed to update it.
    pub(crate) fn owns_virtualizable_shadow(&self) -> bool {
        self.is_active_vable_owner || self.bridge_local_oprefs.is_some()
    }

    /// Promote this frame to active virtualizable owner. Wraps the
    /// macro-generated `init_vable_indices` so the per-frame
    /// `is_active_vable_owner` boolean and the u32 OpRef-offset stay in
    /// lock-step. Call this everywhere `init_vable_indices()` was
    /// previously invoked directly.
    pub(crate) fn become_active_vable_owner(&mut self) {
        self.init_vable_indices(crate::virtualizable_gen::FIRST_VABLE_SCALAR_IDX);
        self.is_active_vable_owner = true;
        debug_assert!(
            self.vable_array_base.is_some(),
            "init_vable_indices must seed vable_array_base = Some(...)"
        );
    }

    /// Demote this frame from active virtualizable owner. Used at bridge
    /// setup (`setup_bridge_sym`) where the bridge's inputarg layout
    /// does not have the `[frame, last_instr, pycode, valuestackdepth,
    /// debugdata]` scalar header that the loop-portal
    /// `init_vable_indices` assumes (canonical four-scalar
    /// layout in `virtualizable_spec.rs::PYFRAME_VABLE_FIELDS`);
    /// subsequent reads consult `bridge_local_oprefs` or fall through
    /// to the heap array via `locals_cells_stack_array_ref`.
    pub(crate) fn clear_active_vable(&mut self) {
        self.vable_array_base = None;
        self.is_active_vable_owner = false;
    }

    #[doc(hidden)]
    pub fn from_test_state(state: TestSymState) -> Self {
        let mut sym = Self::new_uninit(state.frame);
        sym.jitcode = state.jitcode as *const JitCode;
        sym.nlocals = state.nlocals;
        sym.valuestackdepth = state.valuestackdepth;
        sym.locals_cells_stack_array_ref = state.locals_cells_stack_array_ref;
        sym.symbolic_local_types = state.symbolic_local_types;
        sym.symbolic_stack_types = state.symbolic_stack_types;
        sym.registers_r.replace(state.registers_r);
        sym.concrete_stack = state.concrete_stack;
        sym.concrete_namespace = state.concrete_namespace;
        sym.vable_last_instr = state.vable_last_instr;
        sym.vable_pycode = state.vable_pycode;
        sym.vable_valuestackdepth = state.vable_valuestackdepth;
        sym.vable_debugdata = state.vable_debugdata;
        sym.frame_w_globals = state.frame_w_globals;
        sym
    }

    #[doc(hidden)]
    pub fn set_test_execution_context(&mut self, execution_context: OpRef) {
        self.execution_context = execution_context;
    }

    /// Initialize symbolic tracking state. Called once when the owning
    /// MetaInterpFrame is pushed (trace.rs for root frame). Callee (inline)
    /// frames set symbolic state manually — in the walker through
    /// `inline_call.rs`'s `setup_call` port, in the metainterp through
    /// `pyjitpl.rs` `perform_call` — and do NOT call this.
    pub(crate) fn init_symbolic(&mut self, ctx: &mut TraceCtx, concrete_frame: usize) {
        self.is_function_entry_trace = ctx.header_pc == 0;
        let nlocals = concrete_nlocals(concrete_frame).unwrap_or(0);
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit][init-sym] concrete_frame={:#x} nlocals={} vable_base={:?} header_pc={} func_entry={}",
                concrete_frame,
                nlocals,
                self.vable_array_base,
                ctx.header_pc,
                self.is_function_entry_trace
            );
        }
        let valuestackdepth = if self.bridge_stack_oprefs.is_some() {
            // Bridge resumes keep the resume-decoded root depth. Multi-frame
            // deopt later rewrites the live frame's valuestackdepth to the
            // innermost callee depth (`decode_and_restore_guard_failure` in
            // pyre-jit's `eval.rs`), while the resume payload still carries the
            // root depth; heap state is rebuilt from
            // resume boxes, never the reverse (rebuild_state_after_failure,
            // pyjitpl.py).
            self.valuestackdepth
        } else {
            concrete_stack_depth(concrete_frame).unwrap_or(nlocals)
        };
        let stack_only_depth = valuestackdepth.saturating_sub(nlocals);
        self.nlocals = nlocals;
        self.locals_cells_stack_array_ref = if self.is_active_vable_owner {
            OpRef::NONE
        } else {
            frame_locals_cells_stack_array(ctx, self.frame)
        };
        // RPython pyjitpl.py setup init analogue for the local-slot view
        // of registers_r. The bridge override / vable inputarg / NONE
        // shape is the per-trace seed; subsequent load_local_value /
        // store_local_value updates the per-color slot directly.
        self.registers_r.replace(
            if let Some(ref overrides) = self.bridge_local_oprefs.as_ref() {
                // resume.py `rebuild_from_resumedata`: bridge trace uses OpRefs derived from
                // rebuild_from_resumedata (Box(n) → bridge InputArg OpRef::from_raw(n)).
                let mut locals = overrides.to_vec();
                locals.resize(nlocals, OpRef::NONE);
                locals
            } else if let Some(base) = self.vable_array_base {
                (0..nlocals)
                    .map(|i| OpRef::input_arg_ref(base + i as u32))
                    .collect()
            } else {
                vec![OpRef::NONE; nlocals]
            },
        );
        // RPython resume.py rebuild_from_resumedata parity: bridge traces enter with the
        // failing guard's saved boxes, NOT with the loop's full
        // virtualizable inputarg layout. Each `bridge_local_oprefs[i]`
        // points at the bridge inputarg slot the rebuilt frame placed in
        // local i, so the type for local i must come from
        // `inputarg_types[bridge_local_oprefs[i].0]` instead of the
        // loop's `vable_array_base + i` indexing (which only applies to
        // loops where every local sits in a fixed virtualizable slot).
        //
        // Loop / function-entry traces still use vable_array_base because
        // their inputarg list is the full vable layout.
        let inputarg_slot_types = if let Some(ref overrides) = self.bridge_local_oprefs.as_ref() {
            let inputarg_types = ctx.inputarg_types();
            let locals: Vec<Type> = (0..nlocals)
                .map(|i| {
                    overrides
                        .get(i)
                        .and_then(|opref| {
                            if opref.is_none() || opref.is_constant() {
                                None
                            } else {
                                inputarg_types.get(opref.raw() as usize).copied()
                            }
                        })
                        .unwrap_or(Type::Ref)
                })
                .collect();
            // resume.py consume_boxes + pyjitpl.py:2899 parity:
            // guard-failure resume reconstructs the CURRENT frame state,
            // including the live Python stack. `bridge_local_oprefs`
            // overrides only the locals slice; stack slots still follow
            // the virtualizable Ref contract until setup_bridge_sym
            // overwrites the rebuilt tail with the precise resume-data
            // OpRefs.
            Some((locals, vec![Type::Ref; stack_only_depth]))
        } else {
            self.vable_array_base.map(|base| {
                let inputarg_types = ctx.inputarg_types();
                let locals: Vec<Type> = (0..nlocals)
                    .map(|i| {
                        inputarg_types
                            .get(base as usize + i)
                            .copied()
                            .unwrap_or(Type::Ref)
                    })
                    .collect();
                let stack: Vec<Type> = (0..stack_only_depth)
                    .map(|i| {
                        inputarg_types
                            .get(base as usize + nlocals + i)
                            .copied()
                            .unwrap_or(Type::Ref)
                    })
                    .collect();
                (locals, stack)
            })
        };
        if self.is_function_entry_trace {
            // virtualizable.py read_boxes() parity: all array items
            // are GC pointers → Ref. No pre-unboxing at function entry.
            self.symbolic_local_types = vec![Type::Ref; nlocals];
        } else if let Some(ref overrides) = self.bridge_local_types {
            // virtualizable.py:44: bridge locals keep the Ref contract of
            // the virtualizable array. `bridge_local_types` is always
            // all-Ref (populated by setup_bridge_sym); the override is
            // retained only for length parity with bridge_local_oprefs.
            let mut types = overrides.clone();
            types.resize(nlocals, Type::Ref);
            self.symbolic_local_types = types;
        } else if let Some((ref local_types, _)) = inputarg_slot_types {
            // warmstate.py wrap(): inputargs resolved from the JIT
            // contract retain whatever declared type the input signature
            // records. For virtualizable array inputargs this is Ref.
            self.symbolic_local_types = local_types.clone();
        } else if self.symbolic_local_types.len() != nlocals {
            self.symbolic_local_types = concrete_slot_types(concrete_frame, nlocals, nlocals);
        }
        // Seed the stack portion of `registers_r`
        // directly. `registers_r` is the unified abstract register
        // file — locals occupy `[..nlocals]` and stack slots occupy
        // `[nlocals..nlocals + stack_only_depth]` (RPython
        // `pyjitpl.py` MIFrame parity).
        let stack_seed: Vec<OpRef> = if let Some(base) = self.vable_array_base {
            let stack_base = base + nlocals as u32;
            (0..stack_only_depth)
                .map(|i| OpRef::input_arg_ref(stack_base + i as u32))
                .collect()
        } else if let Some(ref bridge_stack) = self.bridge_stack_oprefs.as_ref() {
            // #124: a bridge resumes from a guard whose live operand stack
            // setup_bridge_sym already resolved into these OpRefs. The local
            // override (above) plus the comment at the bridge_local_oprefs
            // branch assume setup_bridge_sym overwrites the rebuilt tail,
            // but pyre's launcher runs setup_bridge_sym BEFORE this; so
            // preserve the kept temps here rather than reset them to NONE.
            let mut seed = bridge_stack.to_vec();
            seed.resize(stack_only_depth, OpRef::NONE);
            seed
        } else {
            vec![OpRef::NONE; stack_only_depth]
        };
        self.registers_r.extend(stack_seed.iter().copied());
        if let Some((_, ref stack_types)) = inputarg_slot_types {
            self.symbolic_stack_types = stack_types.clone();
        } else if self.symbolic_stack_types.len() != stack_only_depth {
            self.symbolic_stack_types =
                concrete_slot_types(concrete_frame, nlocals, valuestackdepth)
                    .into_iter()
                    .skip(nlocals)
                    .collect();
        }
        self.pending_next_instr = None;
        self.valuestackdepth = valuestackdepth;
        // MIFrame concrete Box tracking: populate concrete value arrays
        // from the concrete frame snapshot (RPython MIFrame.setup_call parity).
        // Use concrete_value_from_slot to distinguish "real null pointer"
        // (Ref(PY_NULL)) from "untracked" (ConcreteValue::Null).
        self.concrete_locals = (0..nlocals)
            .map(|i| {
                let obj = concrete_stack_value(concrete_frame, i).unwrap_or(PY_NULL);
                concrete_value_from_slot(obj)
            })
            .collect();
        self.concrete_stack = (0..stack_only_depth)
            .map(|i| {
                let obj = concrete_stack_value(concrete_frame, nlocals + i).unwrap_or(PY_NULL);
                concrete_value_from_slot(obj)
            })
            .collect();
        // Extract frame metadata pointers for use without concrete_frame
        if concrete_frame != 0 {
            let frame = unsafe { &*(concrete_frame as *const pyre_interpreter::pyframe::PyFrame) };
            self.jitcode = jitcode_for(frame.pycode);
            self.concrete_namespace = frame.get_w_globals();
            self.frame_w_globals = ctx.const_ref(self.concrete_namespace as usize as i64);
            if self.concrete_execution_context.is_null() {
                self.concrete_execution_context =
                    pyre_interpreter::call::getexecutioncontext() as *const _;
            }
            // `PyPyJitDriver.reds = ['frame', 'ec']`: an MIFrame register
            // contains a Box together with its recording-time value.  The
            // virtualizable seeding below supplies that value for `frame` and
            // every vable field, but `ec` is the one extra red outside the
            // virtualizable layout.  Stamp it explicitly so blackhole.py's
            // `_copy_data_from_miframe` analogue can copy the symbolic red
            // without turning it back into a thread-specific ConstPtr.
            if !self.execution_context.is_none() && !self.concrete_execution_context.is_null() {
                ctx.try_set_opref_concrete(
                    self.execution_context,
                    majit_ir::Value::Ref(majit_ir::GcRef(self.concrete_execution_context as usize)),
                );
            }
            self.concrete_vable_ptr = concrete_frame as *mut u8;
            // pyjitpl.py MIFrame.setup parity for the per-kind banks
            // (including pyjitpl.py copy_constants).
            self.setup_kind_register_banks(ctx);
        }
        // pyjitpl.py:3458-3462 / virtualizable.py read_boxes parity:
        // seed the tracing-time `virtualizable_boxes` cache with the
        // InputArg OpRefs that correspond to the portal's virtualizable
        // layout (scalar fields followed by array items, with the
        // virtualizable identity appended at `boxes[-1]`). Without this
        // seeding, `vable_setarrayitem_indexed` / `vable_getarrayitem_*`
        // fall through to raw SetarrayitemGc / GetarrayitemGc ops, and
        // `close_loop_args_at`'s `set_virtualizable_box_at` mirror
        // becomes a no-op — the very reason the retired MIFrame local-store
        // hook could not route through the standard vable path.
        if let Some(base) = self.vable_array_base {
            // virtualizable.py read_boxes iterates `len(lst)` — the
            // heap-side `locals_cells_stack_w` length, which is
            // `num_locals + num_cells + max_stack` (pyframe.py:107-110).
            // Match that shape so every interpreter-visible slot has a
            // tracing-time mirror. The live prefix [0, nlocals+stack_only_depth)
            // references the recorder's InputArg stream; reserved stack slots
            // beyond the current stack pointer are NULL on the PyFrame heap
            // (alloc_fixed_array_with_header(..., PY_NULL)) and the helper pads
            // them with a shared const-NULL OpRef.
            let num_vable_scalars = crate::virtualizable_gen::NUM_VABLE_SCALARS;
            let live_prefix = nlocals + stack_only_depth;
            // virtualizable.py read_boxes iterates `len(lst)` — the heap
            // array, not the live valuestackdepth window. Prefer the live
            // interpreter frame (the compiled loop's object), then the
            // tracing snapshot, then jitcode metadata. Never shrink below
            // an already-seeded `initialize_virtualizable` layout: that
            // length is the LABEL `inputarg_types`, and a shorter re-seed
            // made the walker JUMP one box short of it.
            let mut array_len = concrete_frame_array_len(self.live_vable_frame_addr)
                .or_else(|| concrete_frame_array_len(concrete_frame))
                .or_else(|| {
                    (!self.jitcode.is_null()).then(|| {
                        let jc = unsafe { &*self.jitcode };
                        jc.payload.metadata.stack_base + jc.payload.metadata.max_stackdepth
                    })
                })
                .unwrap_or(live_prefix);
            if let Some(existing) = ctx.virtualizable_array_lengths() {
                let existing_array: usize = existing.iter().copied().sum();
                if existing_array > array_len {
                    array_len = existing_array;
                }
            }
            // pyjitpl.py initialize_virtualizable parity: the concrete
            // half of virtualizable_boxes at portal entry comes from a live
            // heap read (vinfo.read_boxes(cpu, virtualizable, 0)). There is
            // no resume-data stream at root-trace start.
            let info = crate::frame_layout::build_pyframe_virtualizable_info();
            // Static fields inputargs at FIRST_VABLE_SCALAR_IDX..+NUM_VABLE_SCALARS.
            // virtualizable_boxes carries vable static fields only — non-vable
            // extra reds (e.g. `ec`) sit between frame and vable scalars in the
            // inputarg space (`pyjitpl.py:2957 redboxes` then `:2964
            // + virtualizable_boxes`) and never enter the shadow. Per-slot
            // type follows `info.static_fields[i].field_type` so the OpRef
            // variant matches RPython's BoxInt/BoxRef inputarg classes.
            let first = crate::virtualizable_gen::FIRST_VABLE_SCALAR_IDX;
            let scalar_oprefs: Vec<OpRef> = (0..num_vable_scalars)
                .map(|i| {
                    let pos = first + i as u32;
                    let tp = info.static_fields[i].field_type;
                    OpRef::input_arg_typed(pos, tp)
                })
                .collect();
            // Array items inputargs at base..base + live_prefix. Items are
            // W_Root (Ref) per `array_item_type = Ref` in virtualizable_gen.
            let array_items: Vec<OpRef> = (0..live_prefix)
                .map(|i| OpRef::input_arg_ref(base + i as u32))
                .collect();
            // SYM_FRAME_IDX is the virtualizable identity (Ref).
            let vable_ref = OpRef::input_arg_ref(crate::virtualizable_gen::SYM_FRAME_IDX);
            let array_lengths = [array_len];
            let (input_values, vable_ref_value) = if concrete_frame != 0 {
                let boxes = unsafe { info.read_boxes(concrete_frame as *const u8, &array_lengths) };
                let (static_boxes, mut items_left) = boxes.split_at(info.static_fields.len());
                let mut values = Vec::with_capacity(num_vable_scalars + array_len);
                for (field, bits) in info.static_fields.iter().zip(static_boxes) {
                    values.push(value_for_slot(field.field_type, *bits));
                }
                for (array, &length) in info.array_fields.iter().zip(&array_lengths) {
                    let (items, rest) = items_left.split_at(length);
                    items_left = rest;
                    let item_ty = array.item_type;
                    for (item_idx, bits) in items.iter().enumerate() {
                        // Seed a tagged-immediate local as the heap `W_IntObject`
                        // it stands for, so the recorded body reads it through the
                        // flag-false `GuardClass`+immutable `GetfieldGc` arm rather than
                        // the tagged `CastPtrToInt`+`IntAnd` arm. The runtime
                        // (`execute_assembler`) converts the live frame's local
                        // slots to heap boxes before the compiled loop reads them;
                        // the tracer works on a `snapshot_for_tracing` copy whose
                        // slots are still tagged, so the same conversion must run
                        // here at record time. Only the locals region is converted
                        // (cells/free vars and stack temps stay untouched).
                        if pyre_object::tagged_int::CAN_BE_TAGGED
                            && item_ty == Type::Ref
                            && item_idx < nlocals
                        {
                            let obj = *bits as pyre_object::PyObjectRef;
                            if !obj.is_null() && pyre_object::tagged_int::is_tagged_int(obj) {
                                let value = pyre_object::tagged_int::untag_int(obj);
                                let heap = pyre_object::intobject::w_int_new_unique(value);
                                values.push(majit_ir::Value::Ref(majit_ir::GcRef(heap as usize)));
                                continue;
                            }
                        }
                        values.push(value_for_slot(item_ty, *bits));
                    }
                }
                while values.len() < num_vable_scalars + array_len {
                    values.push(majit_ir::Value::Ref(majit_ir::GcRef::NULL));
                }
                // Bake the virtualizable identity against the
                // LIVE interpreter frame (the frame the compiled loop runs on),
                // not the discarded `snapshot_for_tracing` copy.  At root entry
                // the snapshot is a fresh copy so its field VALUES equal the
                // live frame's (read_boxes above); only the identity ADDRESS
                // must be the live one, matching the SYM_FRAME_IDX input arg's
                // runtime value (the live frame supplied by extract_live_values).
                // Falls back to the snapshot address when no live frame was
                // threaded (unit-test / init-before-run path).
                let vable_identity_addr = if self.live_vable_frame_addr != 0 {
                    self.live_vable_frame_addr
                } else {
                    concrete_frame
                };
                (
                    values,
                    majit_ir::Value::Ref(majit_ir::GcRef(vable_identity_addr)),
                )
            } else {
                (Vec::new(), majit_ir::Value::Void)
            };
            crate::state::seed_virtualizable_boxes(
                ctx,
                vable_ref,
                vable_ref_value,
                &scalar_oprefs,
                &array_items,
                array_len,
                &input_values,
                concrete_frame as *const u8,
            );
            // history.py `*FrontendOp._resref` parity: give every ref
            // input arg its runtime value on the recorder value channel (shared
            // across the outer walk and every inlined sub-walk), not only in the
            // per-frame register shadow / vable cache.  An inlined callee that
            // reads a loop-invariant OUTER inputarg — e.g. `getfield_gc_r` off a
            // portal-frame input reached through `sys._getframe` — records the
            // load with `box_value(inputarg)` resolvable, so the load result is
            // stamped and a blackhole-resume image can recover it.  Ref-only:
            // int/float locals reconstruct through their own concrete shadows,
            // and input args are never `is_constant()`, so folding gates are
            // untouched.
            for (i, opref) in scalar_oprefs.iter().enumerate() {
                if let Some(majit_ir::Value::Ref(_)) = input_values.get(i).copied() {
                    ctx.try_set_opref_concrete(*opref, input_values[i]);
                }
            }
            for (j, opref) in array_items.iter().enumerate() {
                if let Some(v @ majit_ir::Value::Ref(_)) =
                    input_values.get(num_vable_scalars + j).copied()
                {
                    ctx.try_set_opref_concrete(*opref, v);
                }
            }
            if let majit_ir::Value::Ref(_) = vable_ref_value {
                ctx.try_set_opref_concrete(vable_ref, vable_ref_value);
            }
        }
    }

    /// Read a concrete value from the Box arrays using an absolute
    /// unified-array index (0..nlocals = locals, nlocals.. = stack).
    pub(crate) fn concrete_value_at_opt(&self, abs_idx: usize) -> Option<ConcreteValue> {
        if abs_idx < self.nlocals {
            self.concrete_locals.get(abs_idx).copied()
        } else {
            let stack_idx = abs_idx - self.nlocals;
            self.concrete_stack.get(stack_idx).copied()
        }
    }

    /// Read a concrete value from the Box arrays using an absolute
    /// unified-array index (0..nlocals = locals, nlocals.. = stack).
    pub(crate) fn concrete_value_at(&self, abs_idx: usize) -> ConcreteValue {
        self.concrete_value_at_opt(abs_idx)
            .unwrap_or(ConcreteValue::Null)
    }

    /// #143 frame-advance: number of local slots whose post-trace concrete
    /// state is tracked (`min(concrete_locals.len(), nlocals)`).
    pub fn tracked_local_count(&self) -> usize {
        self.concrete_locals.len().min(self.nlocals)
    }

    /// #143 frame-advance: materialize local slot `i`'s post-trace concrete
    /// value as a boxed `PyObjectRef` (lazy `Int`/`Float` allocated via
    /// `w_int_new`/`w_float_new`; `Ref` returned as-is). `None` for a `Null`
    /// (dead/uninitialized) slot. The caller must root the returned box
    /// before the next allocation.
    pub fn materialize_local(&self, i: usize) -> Option<PyObjectRef> {
        let cv = *self.concrete_locals.get(i)?;
        if cv.is_null() {
            None
        } else {
            Some(cv.to_pyobj())
        }
    }
}

impl PyreJitState {
    /// Slot of the `pypyjit` portal driver in `jitdrivers_sd`; the first
    /// driver `warmspot.py` registers.
    pub const PYPYJIT_JD_INDEX: usize = 0;

    /// Canonical PyPy portal driver layout from `interp_jit.py:67-74`.
    ///
    /// Single source of truth for the portal greens/reds/virtualizable shape:
    /// used both trace-locally (`driver_descriptor` for merge-point payload
    /// validation) and by the production driver registration in
    /// `build_jit_driver_pair`, so the `portal_calldescr` ABI stays consistent
    /// with the descriptor the walker validates against.
    pub fn pypyjit_driver_descriptor() -> JitDriverStaticData {
        let mut descriptor = JitDriverStaticData::with_virtualizable(
            pypyjit_driver_layout::PYPYJIT_GREEN_VARS.to_vec(),
            pypyjit_driver_layout::PYPYJIT_RED_VARS.to_vec(),
            Some(pypyjit_driver_layout::PYPYJIT_VIRTUALIZABLE),
        );
        // interp_jit.py pypyjitdriver = PyPyJitDriver(name='pypyjit', ...)
        descriptor.name = "pypyjit".into();
        descriptor.is_recursive = true;
        // The portal's frames are numbered in the CodeObject-keyed runtime
        // store this crate grows, so name that decoder on the driver rather
        // than leaning on the process-global slot. The global has one writer
        // per store and no arbitration between them
        // (`ensure_finish_setup` here, `install_state_field_fvc` in
        // majit-metainterp), so a driver that leaves this `None` decodes
        // against whichever store registered last — and the wrong store
        // returns a mistyped count instead of failing (see the field's doc).
        // jd1 names its build-time store the same way
        // (`unpackiterable_driver_descriptor`).
        descriptor.frame_value_count_fn = Some(frame_value_count_at);
        descriptor
    }

    /// `generator.py` `generatorentry_driver`: greens `pycode`, reds `gen`
    /// and `w_arg`, no virtualizable. No `can_enter_jit` in the source, so
    /// `no_loop_header` stays set and the enter lives at the portal entry.
    pub fn generatorentry_driver_descriptor() -> JitDriverStaticData {
        let mut descriptor = JitDriverStaticData::new(
            vec![("pycode", Type::Ref)],
            vec![("gen", Type::Ref), ("w_arg", Type::Ref)],
        );
        descriptor.name = "generatorentry".into();
        descriptor.no_loop_header = true;
        descriptor.frame_value_count_fn = Some(build_time_frame_value_count_at);
        descriptor
    }

    fn execution_context_as_usize(&self) -> usize {
        self.execution_context
    }

    fn expanded_virtualizable_live_values_with_extra_reds(
        &self,
        meta: &PyreMeta,
        extra_reds: &[Value],
    ) -> Vec<Value> {
        let base = crate::virtualizable_gen::virt_extract_live_values(
            self.frame,
            self.execution_context_as_usize(),
            self.last_instr_as_usize(),
            self.pycode_as_usize(),
            self.valuestackdepth(),
            self.debugdata_as_usize(),
            meta.num_locals,
            meta.valuestackdepth,
            |i| self.local_at(i).unwrap_or(PY_NULL) as usize,
            |i| self.stack_at(i).unwrap_or(PY_NULL) as usize,
        );
        let Some((&frame, tail)) = base.split_first() else {
            return extra_reds.to_vec();
        };
        let mut values = Vec::with_capacity(base.len() + extra_reds.len());
        values.push(frame);
        values.extend_from_slice(extra_reds);
        values.extend_from_slice(tail);
        values
    }

    fn expanded_virtualizable_live_value_types_with_extra_reds(
        meta: &PyreMeta,
        extra_red_types: &[Type],
    ) -> Vec<Type> {
        let base = crate::virtualizable_gen::virt_live_value_types(meta.slot_types.len());
        let Some((&frame_tp, tail)) = base.split_first() else {
            return extra_red_types.to_vec();
        };
        let mut types = Vec::with_capacity(base.len() + extra_red_types.len());
        types.push(frame_tp);
        types.extend_from_slice(extra_red_types);
        types.extend_from_slice(tail);
        types
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn pypyjit_live_values_with_ec(&self, meta: &PyreMeta) -> Vec<Value> {
        let _ = meta;
        vec![
            Value::Ref(majit_ir::GcRef(self.frame)),
            Value::Ref(majit_ir::GcRef(self.execution_context_as_usize())),
        ]
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn pypyjit_live_value_types_with_ec(meta: &PyreMeta) -> Vec<Type> {
        let _ = meta;
        vec![Type::Ref, Type::Ref]
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn pypyjit_collect_jump_args(sym: &PyreSym) -> Vec<OpRef> {
        let base = sym.vable_collect_jump_args();
        let Some((&frame, tail)) = base.split_first() else {
            unreachable!(
                "vable_collect_jump_args returned empty — virtualizable macro \
                 did not emit frame_field push (see majit-macros/.../derive.rs:381)"
            );
        };
        let mut args = Vec::with_capacity(base.len() + 1);
        args.push(frame);
        args.push(sym.execution_context);
        args.extend_from_slice(tail);
        args
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn pypyjit_collect_typed_jump_args(sym: &PyreSym) -> Vec<(OpRef, Type)> {
        let base = sym.vable_collect_typed_jump_args();
        let Some(&(frame, frame_tp)) = base.first() else {
            unreachable!(
                "vable_collect_typed_jump_args returned empty — virtualizable \
                 macro did not emit frame_field push \
                 (see majit-macros/.../derive.rs:389)"
            );
        };
        let mut args = Vec::with_capacity(base.len() + 1);
        args.push((frame, frame_tp));
        args.push((sym.execution_context, Type::Ref));
        args.extend_from_slice(&base[1..]);
        args
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn pypyjit_create_sym(meta: &PyreMeta, _header_pc: usize) -> PyreSym {
        let mut sym = PyreSym::new_uninit(OpRef::input_arg_typed(0, Type::Ref));
        sym.execution_context = OpRef::input_arg_typed(1, Type::Ref);
        // `become_active_vable_owner` calls `init_vable_indices(FIRST_VABLE_SCALAR_IDX)`
        // where `FIRST_VABLE_SCALAR_IDX = 1 + NUM_EXTRA_REDS` already accounts for
        // the `ec` extra red. Do not re-shift the slot indices here — doing so
        // would double-count, leaving e.g. `vable_pycode` at the valuestackdepth
        // slot (Type::Int) instead of the pycode slot (Type::Ref) and tripping
        // the resume-time type assertion in `value_to_static_vable_bits`.
        sym.become_active_vable_owner();
        sym.nlocals = meta.num_locals;
        sym.valuestackdepth = meta.valuestackdepth;
        sym.symbolic_local_types = vec![Type::Ref; meta.num_locals];
        sym.symbolic_stack_types = vec![Type::Ref; meta.vable_stack_only_depth()];
        let stack_only = meta.vable_stack_only_depth();
        sym.concrete_stack = vec![ConcreteValue::Null; stack_only];
        sym
    }

    fn restore_expanded_virtualizable_with_extra_reds(
        &mut self,
        meta: &PyreMeta,
        values: &[Value],
        extra_reds: usize,
    ) {
        let Some(frame) = values.first() else {
            return;
        };
        self.frame = value_to_usize(frame);
        if values.len() <= 1 + extra_reds {
            return;
        }

        if meta.has_virtualizable {
            // next_instr is already synced to the PyFrame heap by the
            // compiled code's virtualizable sync before JUMP.
            self.set_valuestackdepth(meta.valuestackdepth);
            let nlocals = self.local_count();
            let stack_only = meta.valuestackdepth.saturating_sub(nlocals);
            let _ = extra_reds; // already folded into NUM_SCALAR_INPUTARGS
            let mut idx = crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
            for local_idx in 0..nlocals {
                if let Some(value) = values.get(idx) {
                    let slot_type = meta.slot_types.get(local_idx).copied().unwrap_or(Type::Ref);
                    let _ =
                        self.set_local_at(local_idx, boxed_slot_value_for_type(slot_type, value));
                }
                idx += 1;
            }
            for i in 0..stack_only {
                if let Some(value) = values.get(idx) {
                    let slot_type = meta
                        .slot_types
                        .get(nlocals + i)
                        .copied()
                        .unwrap_or(Type::Ref);
                    let _ = self.set_stack_at(i, boxed_slot_value_for_type(slot_type, value));
                }
                idx += 1;
            }
        } else {
            let nlocals = self.local_count();
            let stack_only_depth = meta.valuestackdepth.saturating_sub(nlocals);
            let mut idx = 1 + extra_reds;
            for local_idx in 0..nlocals {
                let slot_type = meta.slot_types.get(local_idx).copied().unwrap_or(Type::Ref);
                let _ = self.set_local_at(
                    local_idx,
                    boxed_slot_value_for_type(slot_type, &values[idx]),
                );
                idx += 1;
            }
            for i in 0..stack_only_depth {
                let slot_type = meta
                    .slot_types
                    .get(nlocals + i)
                    .copied()
                    .unwrap_or(Type::Ref);
                let _ = self.set_stack_at(i, boxed_slot_value_for_type(slot_type, &values[idx]));
                idx += 1;
            }
            self.set_valuestackdepth(meta.valuestackdepth);
        }
    }

    /// virtualizable.py write_from_resume_data_partial parity.
    ///
    /// Restore virtualizable frame state from raw output buffer values.
    /// interp_jit.py:25: locals_cells_stack_w ARRAYITEMTYPE = GCREF.
    /// All raw values are PyObjectRef pointers — write directly.
    pub fn restore_virtualizable_from_raw(&mut self, raw_values: &[i64]) -> bool {
        if raw_values.is_empty() {
            return false;
        }
        let mut idx = crate::virtualizable_gen::virt_restore_scalars_raw(self, raw_values);

        let nlocals = self.local_count();
        let stack_only = self.valuestackdepth().saturating_sub(nlocals);
        for local_idx in 0..nlocals {
            if idx < raw_values.len() {
                let _ = self.set_local_at(local_idx, raw_values[idx] as PyObjectRef);
            }
            idx += 1;
        }
        for stack_idx in 0..stack_only {
            if idx < raw_values.len() {
                let _ = self.set_stack_at(stack_idx, raw_values[idx] as PyObjectRef);
            }
            idx += 1;
        }
        true
    }

    /// Returns true if the optimizer virtualizable mechanism is active.
    fn has_virtualizable_info(&self) -> bool {
        // pyre always uses virtualizable (JitDriverStaticData::with_virtualizable)
        true
    }

    fn frame_ptr(&self) -> Option<*mut u8> {
        (self.frame != 0).then_some(self.frame as *mut u8)
    }

    fn frame_array(&self, offset: usize) -> Option<&pyre_object::FixedObjectArray> {
        let frame_ptr = self.frame_ptr()?;
        let arr_ptr =
            unsafe { *(frame_ptr.add(offset) as *const *const pyre_object::FixedObjectArray) };
        Some(unsafe { &*arr_ptr })
    }

    fn frame_array_mut(&mut self, offset: usize) -> Option<&mut pyre_object::FixedObjectArray> {
        let frame_ptr = self.frame_ptr()?;
        let arr_ptr =
            unsafe { *(frame_ptr.add(offset) as *const *mut pyre_object::FixedObjectArray) };
        Some(unsafe { &mut *arr_ptr })
    }

    fn read_frame_usize(&self, offset: usize) -> Option<usize> {
        let frame_ptr = self.frame_ptr()?;
        Some(unsafe { *(frame_ptr.add(offset) as *const usize) })
    }

    fn write_frame_usize(&mut self, offset: usize, value: usize) -> bool {
        let Some(frame_ptr) = self.frame_ptr() else {
            return false;
        };
        unsafe {
            *(frame_ptr.add(offset) as *mut usize) = value;
        }
        true
    }

    fn locals_cells_stack_array(&self) -> Option<&pyre_object::FixedObjectArray> {
        self.frame_array(PYFRAME_LOCALS_CELLS_STACK_OFFSET)
    }

    fn locals_cells_stack_array_mut(&mut self) -> Option<&mut pyre_object::FixedObjectArray> {
        self.frame_array_mut(PYFRAME_LOCALS_CELLS_STACK_OFFSET)
    }

    /// Return the namespace length for module dictionaries and exact plain
    /// dictionaries. Other mappings use a different object layout, so their
    /// length is unknown rather than zero: namespace-dependent traces must not
    /// treat two unsafe-to-inspect mappings as compatible.
    fn namespace_len(&self) -> Option<usize> {
        let Some(frame_ptr) = self.frame_ptr() else {
            return None;
        };
        let w_globals =
            unsafe { (&*(frame_ptr as *const pyre_interpreter::pyframe::PyFrame)).get_w_globals() };
        if w_globals.is_null() {
            return None;
        }
        // `dictmultiobject.py W_DictMultiObject.length`. The common
        // module-dict case reads `ModuleDictStorage` directly (O(1)) — this guard is on the
        // per-portal-entry path. Exact plain dict globals (exec/eval) fall back
        // to the polymorphic strategy length; dict subclasses and other
        // mappings do not carry that layout and remain unknown.
        if let Some(len) =
            unsafe { pyre_object::dictmultiobject::module_dict_storage_len(w_globals) }
        {
            Some(len)
        } else if unsafe { is_dict(w_globals) } {
            Some(unsafe { pyre_object::dictmultiobject::w_dict_len(w_globals) })
        } else {
            None
        }
    }

    fn restore_single_frame(&mut self, meta: &PyreMeta, values: &[i64]) {
        let Some(&frame) = values.first() else {
            return;
        };
        self.frame = frame as usize;
        if values.len() == 1 {
            return;
        }
        if meta.has_virtualizable {
            self.restore_virtualizable_i64(values);
        } else {
            let nlocals = self.local_count();
            let stack_only = self.valuestackdepth().saturating_sub(nlocals);
            let mut idx = 1;
            for local_idx in 0..nlocals {
                if idx < values.len() {
                    let slot_type = meta.slot_types.get(local_idx).copied().unwrap_or(Type::Ref);
                    let _ = self
                        .set_local_at(local_idx, boxed_slot_i64_for_type(slot_type, values[idx]));
                }
                idx += 1;
            }
            for i in 0..stack_only {
                if idx < values.len() {
                    let slot_type = meta
                        .slot_types
                        .get(nlocals + i)
                        .copied()
                        .unwrap_or(Type::Ref);
                    let _ = self.set_stack_at(i, boxed_slot_i64_for_type(slot_type, values[idx]));
                }
                idx += 1;
            }
        }
    }

    pub fn local_at(&self, idx: usize) -> Option<PyObjectRef> {
        self.locals_cells_stack_array()
            .and_then(|arr| arr.as_slice().get(idx).copied())
    }

    /// Number of local variable slots.
    pub fn local_count(&self) -> usize {
        concrete_nlocals(self.frame).unwrap_or(0)
    }

    /// Store one absolute slot of `locals_cells_stack_w` and re-arm the write
    /// barrier, as `store_live_frame_array_slot` and `write_back_outer_locals`
    /// do for the same array: the callers box Int/Float slots in a loop, so the
    /// next iteration can allocate, and each minor consumes the array's
    /// remembered-set entry.
    fn set_array_slot_at(&mut self, abs: usize, value: PyObjectRef) -> bool {
        let Some(frame) = self.frame_ptr() else {
            return false;
        };
        let Some(arr) = self.locals_cells_stack_array_mut() else {
            return false;
        };
        let arr_ptr = arr as *mut pyre_object::FixedObjectArray;
        let Some(slot) = arr.as_mut_slice().get_mut(abs) else {
            return false;
        };
        *slot = value;
        frame_array_write_barrier(frame, arr_ptr);
        true
    }

    pub fn set_local_at(&mut self, idx: usize, value: PyObjectRef) -> bool {
        self.set_array_slot_at(idx, value)
    }

    /// Read a stack value at stack-relative index `idx` (0-based from stack bottom).
    pub fn stack_at(&self, idx: usize) -> Option<PyObjectRef> {
        let nlocals = self.local_count();
        self.locals_cells_stack_array()
            .and_then(|arr| arr.as_slice().get(nlocals + idx).copied())
    }

    /// Total capacity of the unified array.
    pub fn array_capacity(&self) -> usize {
        self.locals_cells_stack_array()
            .map(pyre_object::FixedObjectArray::len)
            .unwrap_or(0)
    }

    /// Set a stack value at stack-relative index `idx`.
    pub fn set_stack_at(&mut self, idx: usize, value: PyObjectRef) -> bool {
        let nlocals = self.local_count();
        self.set_array_slot_at(nlocals + idx, value)
    }

    // ── Heap accessors: single source of truth (RPython parity) ──
    // RPython's virtualizable IS the heap object — getattr/setattr go
    // directly to the heap.  These accessors do the same via frame_ptr.

    pub fn last_instr_as_usize(&self) -> usize {
        let frame_ptr = self
            .frame_ptr()
            .expect("PyreJitState.frame must point to a valid PyFrame");
        unsafe { (*(frame_ptr as *const PyFrame)).last_instr as usize }
    }

    pub fn set_last_instr(&mut self, value: usize) {
        let frame_ptr = self
            .frame_ptr()
            .expect("PyreJitState.frame must point to a valid PyFrame");
        unsafe {
            (*(frame_ptr as *mut PyFrame)).last_instr = value as isize;
        }
    }

    pub fn next_instr(&self) -> usize {
        let frame_ptr = self
            .frame_ptr()
            .expect("PyreJitState.frame must point to a valid PyFrame");
        unsafe { (&*(frame_ptr as *const PyFrame)).next_instr() }
    }

    pub fn set_next_instr(&mut self, value: usize) {
        let frame_ptr = self
            .frame_ptr()
            .expect("PyreJitState.frame must point to a valid PyFrame");
        unsafe {
            (&mut *(frame_ptr as *mut PyFrame)).set_last_instr_from_next_instr(value);
        }
    }

    pub fn valuestackdepth(&self) -> usize {
        self.read_frame_usize(PYFRAME_VALUESTACKDEPTH_OFFSET)
            .expect("PyreJitState.frame must point to a valid PyFrame")
    }

    pub fn set_valuestackdepth(&mut self, value: usize) {
        assert!(
            self.write_frame_usize(PYFRAME_VALUESTACKDEPTH_OFFSET, value),
            "PyreJitState.frame must point to a valid PyFrame"
        );
    }

    /// Null the locals_cells_stack slots at and above `depth`, the
    /// fresh-frame parity clear (`write_from_resume_data_partial` does not
    /// trim).  Used after a vsd correction so a GC scan before the next
    /// push does not observe a stale operand pointer above the live depth.
    pub fn clear_stack_above(&mut self, depth: usize) {
        if let Some(frame) = self.frame_ptr() {
            unsafe { (*(frame as *mut PyFrame)).clear_stack_above(depth) };
        }
    }

    /// Read the code pointer (pycode) from the heap frame.
    pub fn pycode_as_usize(&self) -> usize {
        self.read_frame_usize(PYFRAME_PYCODE_OFFSET)
            .expect("PyreJitState.frame must point to a valid PyFrame")
    }

    /// Read the execution context red independently of the virtualizable frame.
    ///
    /// `interp_jit.py reds = ['frame', 'ec']`: ec is a non-vable red
    /// inputarg in RPython. It therefore lives on `PyreJitState`, beside the
    /// frame red, and never in the `PyFrame` heap layout.
    pub fn ec_as_usize(&self) -> usize {
        self.execution_context
    }

    /// Restore the independent execution-context red from resume data.
    pub fn set_ec(&mut self, value: usize) {
        self.execution_context = value;
    }

    /// Read the code pointer (pycode) from the heap frame.
    pub fn code_as_usize(&self) -> usize {
        self.pycode_as_usize()
    }

    /// Read the namespace pointer from the heap frame.
    pub fn namespace_as_usize(&self) -> usize {
        let frame_ptr = self
            .frame_ptr()
            .expect("PyreJitState.frame must point to a valid PyFrame");
        unsafe {
            (&*(frame_ptr as *const pyre_interpreter::pyframe::PyFrame)).get_w_globals() as usize
        }
    }

    /// Write the pycode pointer to the heap frame.
    /// virtualizable.py write_boxes: ALL static fields written.
    pub fn set_pycode(&mut self, value: usize) {
        assert!(
            self.write_frame_usize(PYFRAME_PYCODE_OFFSET, value),
            "PyreJitState.frame must point to a valid PyFrame"
        );
    }

    /// Compatibility wrapper for older callers that still speak in
    /// terms of `code` / `namespace`.
    pub fn set_code(&mut self, value: usize) {
        self.set_pycode(value);
    }

    pub fn set_namespace(&mut self, value: usize) {
        let frame_ptr = self
            .frame_ptr()
            .expect("PyreJitState.frame must point to a valid PyFrame");
        unsafe {
            (&mut *(frame_ptr as *mut pyre_interpreter::pyframe::PyFrame))
                .set_w_globals(value as pyre_object::PyObjectRef);
        }
    }

    /// pyframe.py:82 debugdata — read from heap frame.
    pub fn debugdata_as_usize(&self) -> usize {
        self.read_frame_usize(PYFRAME_DEBUGDATA_OFFSET).unwrap_or(0)
    }

    /// pyframe.py:82 debugdata — write to heap frame.
    pub fn set_debugdata(&mut self, value: usize) {
        let _ = self.write_frame_usize(PYFRAME_DEBUGDATA_OFFSET, value);
    }

    /// Validate that the frame pointer is usable (fields readable, array present).
    fn validate_frame(&self) -> bool {
        self.frame_ptr().is_some()
            && self
                .read_frame_usize(PYFRAME_VALUESTACKDEPTH_OFFSET)
                .is_some()
            && self.locals_cells_stack_array().is_some()
    }

    /// virtualizable.py write_from_resume_data_partial parity.
    ///
    /// Restores virtualizable array slots from the fail_args layout:
    ///   [frame, scalars..., active_locals..., active_stack...]
    ///
    /// interp_jit.py:25 declares locals_cells_stack_w as a single
    /// virtualizable array with uniform ARRAYITEMTYPE = GCREF.
    /// resume.py:1408 calls write_from_resume_data_partial which loops
    /// over the array calling reader.load_next_value_of_type(GCREF),
    /// i.e. reader.next_ref() — every slot is a GCREF pointer.
    ///
    /// The raw i64 values here are already PyObjectRef pointers from
    /// the backend's Ref register bank. Write them directly without
    /// per-slot type dispatch.
    fn restore_virtualizable_i64(&mut self, values: &[i64]) {
        let mut idx = crate::virtualizable_gen::virt_restore_scalars_raw(self, values);

        // virtualizable.py:134-137:
        //   for ARRAYITEMTYPE, fieldname in unroll_array_fields:
        //       lst = getattr(virtualizable, fieldname)
        //       for j in range(len(lst)):
        //           lst[j] = reader.load_next_value_of_type(ARRAYITEMTYPE)
        // ARRAYITEMTYPE is always GCREF for locals_cells_stack_w.
        let nlocals = self.local_count();
        for i in 0..nlocals {
            if idx < values.len() {
                let _ = self.set_local_at(i, values[idx] as PyObjectRef);
            }
            idx += 1;
        }

        let stack_only = self.valuestackdepth().saturating_sub(nlocals);
        for i in 0..stack_only {
            if idx < values.len() {
                let _ = self.set_stack_at(i, values[idx] as PyObjectRef);
            }
            idx += 1;
        }
    }

    fn import_virtualizable_state(&mut self, boxes: &[i64], array_lengths: &[usize]) -> bool {
        // virtualizable.py write_from_resume_data_partial parity:
        // write ALL static fields to heap via VirtualizableInfo.
        let info = crate::frame_layout::build_pyframe_virtualizable_info();
        let Some((static_boxes, items)) = boxes.split_at_checked(info.static_fields.len()) else {
            return false;
        };
        if !self.virt_import_static_boxes(&info, static_boxes) {
            return false;
        }

        // virtualizable.py:134-137: write array items to heap.
        // Validate array structure matches VirtualizableInfo.
        if array_lengths.len() != info.array_fields.len() {
            return false;
        }
        let Some(&unified_len) = array_lengths.first() else {
            return info.array_fields.is_empty();
        };
        let Some(unified) = items.get(..unified_len) else {
            return false;
        };
        let Some(frame_arr) = self.locals_cells_stack_array_mut() else {
            return false;
        };
        if frame_arr.len() != unified.len() {
            return false;
        }
        for (dst, &src) in frame_arr.as_mut_slice().iter_mut().zip(unified) {
            *dst = src as PyObjectRef;
        }
        true
    }

    fn export_virtualizable_state(&self) -> (Vec<i64>, Vec<usize>) {
        let info = crate::frame_layout::build_pyframe_virtualizable_info();
        self.virt_export_all(&info)
    }

    pub fn sync_from_virtualizable(&mut self, info: &VirtualizableInfo) -> bool {
        let _ = info;
        // Heap IS the source of truth. Just validate the frame is usable.
        self.validate_frame()
    }

    pub fn sync_to_virtualizable(&self, info: &VirtualizableInfo) -> bool {
        let Some(frame_ptr) = self.frame_ptr() else {
            return false;
        };
        // Heap is the single source of truth — no state-backed fields to
        // flush.  Only the vable_token needs resetting (virtualizable.py clear_vable_token
        // force_now: set vable_token to TOKEN_NONE).
        unsafe {
            info.reset_vable_token(frame_ptr);
        }
        true
    }
}

/// resume.py unified decode_box parity.
/// Returns `(OpRef, Value)`: symbolic OpRef for trace recording, concrete
/// Value for shadow slots / continue_tracing. Replaces the separate
/// `resolve()` + `decode_concrete()` closures so both paths always execute
/// together — no drift between symbolic and concrete materialization.
/// Decode one inlined-callee resume frame
/// (`resume_data.frames[i]`, `i >= 1`) into a [`ReconstructRecipe`] for the
/// multi-frame bridge carrier. Mirrors the per-bank `consume_boxes` decode
/// `setup_bridge_sym` runs for the portal frame (resume.py) but writes a
/// fresh recipe instead of the root `sym`, and emits NO trace IR (the
/// forward-call guard_value / callee-frame helper / no-exception guard belong
/// to a call with a LIVE caller; a reconstructed suspended frame has none).
///
/// The decoded bank vectors are indexed by pyre's semantic register index
/// (Python-bytecode `locals_cells_stack_w` position, not an RPython color),
/// so they align with `write_stack_slot`'s `nlocals + stack_idx`. The
/// assembly into a `PyFrame` + `PyreSym` is deferred to `trace_bytecode`
/// (the carrier drain), where the root concrete frame's globals/EC are
/// available and the rebuilt locals are GC-rooted immediately.
///
/// Returns `None` (→ abort the multi-frame path, fall back to the
/// single-frame bridge) when the callee cannot be faithfully rebuilt:
///   - `pc` is the no-snapshot sentinel (`< 0`),
///   - `jitcode_index` does not resolve to a code object,
///   - the callee has fresh cell variables (the forward inline constructor
///     does not yet allocate them either),
///   - the liveness enumeration count disagrees with the encoded section.
/// `rd_virtuals[vidx]` with negative-index resolution already applied by the
/// caller. Returns the `RdVirtualInfo` behind the `Rc`, or `None` when the
/// index is out of range or no virtuals were decoded.
fn rd_virtual_at(
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    vidx: usize,
) -> Option<&majit_ir::RdVirtualInfo> {
    rd_virtuals.and_then(|v| v.get(vidx)).map(|rc| &**rc)
}

/// One decoded Ref-array write from `rd_pendingfields`.
///
/// Rebuild the semantic Ref slots of a forced inline frame from its own
/// `locals_cells_stack_w` array.
///
/// The returned OpRefs are heap reads rooted at `frame_box`, not constants
/// made from the current concrete frame.  A compiled bridge therefore reloads
/// the frame image on every guard hit (resume.py consume_boxes).
///
/// `stream_covered[k]` marks a slot the caller fills from the resume stream
/// instead; the image is not read for it (the stream value is authoritative,
/// see `reconstruct_inline_recipe`), and the slot is left NONE/Void here.
fn reconstruct_materialized_frame_slots(
    ctx: &mut majit_metainterp::TraceCtx,
    frame_box: OpRef,
    frame_ref: majit_ir::GcRef,
    valuestackdepth: usize,
    pending_result_abs_slot: Option<usize>,
    stream_covered: &[bool],
    pending_ref_array_writes: &[PendingRefArrayWrite],
) -> Option<(Vec<OpRef>, Vec<majit_ir::Value>)> {
    if frame_ref.is_null() || frame_ref == majit_ir::GcRef::NO_CONCRETE {
        return None;
    }
    let concrete_frame =
        unsafe { &*(frame_ref.as_usize() as *const pyre_interpreter::pyframe::PyFrame) };
    let arr_ptr = concrete_frame.locals_cells_stack_w;
    if arr_ptr.is_null() {
        return None;
    }
    let arr = unsafe { &*arr_ptr };
    if valuestackdepth > arr.len() {
        return None;
    }

    // ResumeDataBoxReader applies pending writes before consuming frame boxes.
    // Pyre's portal-shaped frame section rediscovers locals through
    // `frame.locals_cells_stack_w`; retain the pending target's box identity
    // when it names that exact array. This is the decoded pending stream, not
    // optimizer state or a per-box side table.
    let pending_array_box = pending_ref_array_writes.iter().find_map(|write| {
        matches!(
            ctx.box_value(write.target),
            Some(majit_ir::Value::Ref(concrete))
                if concrete.as_usize() == arr_ptr as usize
        )
        .then_some(write.target)
    });
    let array_box =
        pending_array_box.unwrap_or_else(|| frame_locals_cells_stack_array(ctx, frame_box));
    // `frame_locals_cells_stack_array` is `GETFIELD_GC_R` of the array
    // field (`virtualizable.py` `read_boxes` getattr). Stamp the live
    // array GCREF so the following `GETARRAYITEM_GC_R` loads see it.
    ctx.try_set_opref_concrete(
        array_box,
        majit_ir::Value::Ref(majit_ir::GcRef(arr_ptr as usize)),
    );

    let mut registers_r = vec![OpRef::NONE; valuestackdepth];
    let mut concrete_r = vec![majit_ir::Value::Void; valuestackdepth];
    for k in 0..valuestackdepth {
        if Some(k) == pending_result_abs_slot || stream_covered.get(k).copied().unwrap_or(false) {
            continue;
        }
        // `_prepare_pendingfields` precedes `consume_boxes` in resume.py. If
        // this exact materialized locals array was just written, consume the
        // decoded value box directly. The pending descriptor may be the
        // generic virtualizable-array descriptor while the PyFrame reload uses
        // its concrete GC-array descriptor, so routing through a fresh GET
        // would hide the write from heap optimization despite identical boxes.
        if let Some(write) = pending_ref_array_writes.iter().rev().find(|write| {
            write.item_index == k
                && matches!(
                    ctx.box_value(write.target),
                    Some(majit_ir::Value::Ref(concrete))
                        if concrete.as_usize() == arr_ptr as usize
                )
        }) {
            registers_r[k] = write.value;
            concrete_r[k] = ctx.box_value(write.value).unwrap_or_else(|| {
                majit_ir::Value::Ref(majit_ir::GcRef(arr.as_slice()[k] as usize))
            });
            continue;
        }
        let index = ctx.const_int(k as i64);
        let value = trace_array_getitem_value(ctx, array_box, index);
        registers_r[k] = value;
        concrete_r[k] = ctx
            .box_value(value)
            .unwrap_or_else(|| majit_ir::Value::Ref(majit_ir::GcRef(arr.as_slice()[k] as usize)));
    }
    Some((registers_r, concrete_r))
}

/// resume.py `consume_boxes`: fill the semantic slots the resume stream
/// names straight from that stream, overriding whatever the paused frame's
/// image holds for them.
///
/// `stream_slots` is `(index into the frame's decoded value section, semantic
/// `locals_cells_stack_w` slot)`, built by inverting color -> slot with the
/// per-pc `pcdep` map the encoder used.
#[allow(clippy::too_many_arguments)]
fn overlay_stream_ref_slots(
    ctx: &mut majit_metainterp::TraceCtx,
    stream_slots: &[(usize, usize)],
    values: &[&majit_ir::resumedata::RebuiltValue],
    registers_r: &mut [OpRef],
    concrete_r: &mut [majit_ir::Value],
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    resume_data: &majit_metainterp::ResumeDataResult,
    fail_values: &[i64],
    fail_types: &[Type],
    backend: &dyn majit_backend::Backend,
    cache: &mut BridgeVirtualCache<'_>,
) {
    for &(pos, slot) in stream_slots {
        let (box_ref, value) = bridge_decode_box(
            ctx,
            values[pos],
            Type::Ref,
            rd_virtuals,
            resume_data,
            fail_values,
            fail_types,
            backend,
            cache,
        );
        ctx.try_set_opref_concrete(box_ref, value);
        registers_r[slot] = box_ref;
        concrete_r[slot] = value;
    }
}

/// Decode one suspended inline-callee frame's resume section into a
/// [`ReconstructRecipe`], or `None` to decline the multi-frame inline rebuild.
///
/// `resume.py rebuild_from_resumedata` (`newframe(jitcode)` +
/// `setup_resume_at_op` + `consume_boxes` per suspended frame) ALWAYS
/// allocates and refills every inlined frame — its snapshot writer guarantees
/// the liveness invariants the reader consumes. Pyre's per-function portal
/// model + super-instruction bytecode cannot guarantee those invariants for
/// every reconstructed callee, so this returns `None` exactly where a rebuild
/// would be unsound: a no-snapshot pc, an unresolved jitcode, a pure cellvar
/// band, unrecoverable callee globals, an `enumerate_vars` count that disagrees
/// with the encoded section (`consume_boxes`), or an int/float-bank register
/// with no boxed-Ref source. Existing freevar cell objects, and a cellvar that
/// overlaps a varname, are ordinary entries in the frame red's
/// `locals_cells_stack_w` array and are rebuilt with that frame, exactly as
/// `resume.py consume_boxes` rebuilds every MIFrame slot.
///
/// The `None` fallback is semantically equivalent in program result: the caller
/// declines the multi-frame inline reconstruction and routes to the
/// conservative single-frame bridge / blackhole resume, which re-enters the
/// interpreter and resumes correctly — only the inline-frame reconstruction
/// optimization is forfeited, never correctness. The forward inline capture
/// declines on the same conditions.
fn reconstruct_inline_recipe(
    ctx: &mut majit_metainterp::TraceCtx,
    frame: &majit_ir::resumedata::RebuiltFrame,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    resume_data: &majit_metainterp::ResumeDataResult,
    fail_values: &[i64],
    fail_types: &[Type],
    backend: &dyn majit_backend::Backend,
    cache: &mut BridgeVirtualCache<'_>,
    in_a_call: bool,
    pending_ref_array_writes: &[PendingRefArrayWrite],
) -> Option<ReconstructRecipe> {
    // Each decline below ends the whole multi-frame path (`recipes.clear()` in
    // the caller), so name every class: an uncensused decline is invisible in
    // the corpus and shows up only as the caller's NoRecipes abort.
    macro_rules! decline {
        ($class:literal) => {{
            crate::jitcode_dispatch::census_record(concat!("P2Recipe::", $class));
            return None;
        }};
    }
    if frame.pc < 0 {
        decline!("NoSnapshotPc");
    }
    // `descr_call`'s tail reconstructs no frame.  Its jitcode has no code
    // object, so every check below would decline it, and the whole chain with
    // it; but its JIT-visible body is only the discard of `__init__`'s result
    // and `ref_return` of the instance, which the drain can carry out by
    // substituting the box on the way out.  Decode that one box here.
    if crate::ctor_continuation::is_installed_level(frame.jitcode_index) {
        let [instance_value] = frame.values.as_slice() else {
            decline!("CtorTailArity");
        };
        let (instance, _) = bridge_decode_box(
            ctx,
            instance_value,
            Type::Ref,
            rd_virtuals,
            resume_data,
            fail_values,
            fail_types,
            backend,
            cache,
        );
        if instance.is_none() {
            decline!("CtorTailBox");
        }
        return Some(ReconstructRecipe {
            code_ptr: std::ptr::null(),
            jitcode_index: frame.jitcode_index,
            jitcode_pc: frame.pc,
            nlocals: 0,
            valuestackdepth: 0,
            registers_i: Vec::new(),
            registers_r: Vec::new(),
            registers_f: Vec::new(),
            concrete_r: Vec::new(),
            frame: OpRef::NONE,
            nargs: 0,
            return_substitute: Some(instance),
            len_tail: false,
            rebuilt_frame: None,
            rebuilt_ec: None,
        });
    }
    // An operator's tail (`crate::operator_continuation`) reconstructs no
    // frame either, and unlike `descr_call`'s it cannot be carried out by
    // substituting a box: its JIT-visible body is a CALL that has to run, over
    // a value the callee has not returned yet.  The conservative blackhole
    // resume runs it correctly, so decline the whole chain here -- named,
    // rather than as the `NoCodeForJitcodeIndex` every check below would give
    // it anyway.
    // `capture_resumedata` keeps `operation.py len` on the framestack. This
    // level is that tail: no code object, applied after `__len__` returns.
    if crate::operator_continuation::is_installed_level(frame.jitcode_index) {
        return Some(ReconstructRecipe {
            code_ptr: std::ptr::null(),
            jitcode_index: frame.jitcode_index,
            jitcode_pc: frame.pc,
            nlocals: 0,
            valuestackdepth: 0,
            registers_i: Vec::new(),
            registers_r: Vec::new(),
            registers_f: Vec::new(),
            concrete_r: Vec::new(),
            frame: OpRef::NONE,
            nargs: 0,
            return_substitute: None,
            len_tail: true,
            rebuilt_frame: None,
            rebuilt_ec: None,
        });
    }
    let py_pc =
        crate::py_coord::resume_py_pc_for_jitcode_word(frame.jitcode_index, frame.pc) as usize;
    let Some(w_code) = code_for_jitcode_index(frame.jitcode_index) else {
        decline!("NoCodeForJitcodeIndex");
    };
    if w_code.is_null() {
        decline!("NullWCode");
    }
    let raw_code = unsafe {
        pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
            as *const pyre_interpreter::CodeObject
    };
    if raw_code.is_null() {
        decline!("NullRawCode");
    }
    let code_ref = unsafe { &*raw_code };
    // Forward inlining already accepts freevars by threading the existing
    // closure cell objects into the callee's own frame, and a cellvar that
    // also names a parameter lives in that varname's slot
    // (`pyframe::npure_cellvars` excludes it). `MAKE_CELL` wraps the slot in
    // the callee prologue, so by any body pc the forward inline reaches the
    // cell object is already the local `try_walker_inline_resolved_user_call`
    // stored, and both reconstructions below copy that slot. A pure cellvar
    // lives in its own band, which this resume does not allocate, and stays
    // on the conservative path.
    if pyre_interpreter::pyframe::npure_cellvars(code_ref) > 0 {
        decline!("FreshCellvars");
    }
    // pyframe.py get_w_globals get_w_globals_storage(): the reconstructed callee frame's
    // globals come from its own pycode (`assemble_bridge_inline_pending`
    // resolves them the same way via `recover_inline_callee_globals`). If the
    // callee code never ran under known globals (no live wrapper recovers a
    // namespace), there is nothing to restore, so abort to the single-frame
    // bridge — the forward inline path declines the same way.
    if recover_inline_callee_globals(raw_code as *const ()).is_null() {
        decline!("NoCalleeGlobals");
    }
    let frame_nlocals = code_ref.varnames.len();
    // PyPy's MIFrame register prefix and PyFrame's physical stack base include
    // cell/freevar slots (`pyframe.py`). All semantic slot/color maps
    // below are keyed to that full prefix, so treat it as the recipe's
    // historical `nlocals` value. This is what keeps LOAD_DEREF attached to
    // the reconstructed callee frame rather than collapsing onto its caller.
    let nlocals = frame_nlocals + pyre_interpreter::pyframe::ncells(code_ref);

    let liveness = crate::liveness::liveness_for(raw_code);
    let stack_only = match liveness.stack_depth_at(py_pc) {
        Some(d) => d,
        None => decline!("NoStackDepthAtPc"),
    };
    let pending_result_abs_slot = if in_a_call && stack_only > 0 {
        Some(nlocals + stack_only - 1)
    } else {
        None
    };
    // `stack_depth_at(pc)` is the depth BEFORE executing `pc` (`[0]` is the
    // entry depth and `[pc + 1]` is `pc`'s fallthrough depth), so at the call
    // site an `in_a_call` frame is paused on, it still counts the whole operand
    // region the call consumes — the callable and every argument.  Resuming
    // once the callee returns collapses that region to one result, which the
    // framestack walk delivers through `make_result_of_lastop` rather than from
    // resumedata, so no callee ever flushed those operands to
    // `locals_cells_stack_w` and none of them is reconstructable.  Exempt the
    // whole region from the mandatory-operand demand below, not just its
    // topmost slot: with one argument or more the top slot alone left the
    // callable and the arguments demanded, so every argument-taking inlined
    // call declined and its guard was blacklisted.
    let unreconstructable_operand_floor = if in_a_call {
        liveness
            .stack_depth_at(py_pc + 1)
            .map(|post| nlocals + post.saturating_sub(1))
    } else {
        None
    };
    let pending_result_color = if in_a_call {
        let Some(pc) = usize::try_from(frame.pc).ok() else {
            decline!("NoPendingResultColor");
        };
        let Some(color) = pyjitcode_for_jitcode_index(frame.jitcode_index).and_then(|p| {
            p.result_color_trivia_for_jitcode_pc(pc)
                .map(|c| c as usize)
                .filter(|&c| c != u16::MAX as usize)
        }) else {
            decline!("NoPendingResultColor");
        };
        Some(color)
    } else {
        None
    };
    let mut reg_indices = frame_liveness_reg_indices_by_bank_from_pc(frame.jitcode_index, frame.pc);
    // The encoder enumerates live boxes in bank order [int | ref | float] and
    // `frame.values` mirrors that order. A pending call-result color (the dst of
    // an in-flight residual call) is removed from the ref bank below because the
    // framestack walk delivers that value via `make_result_of_lastop`, not from
    // resumedata. Drop its encoded value from the SAME position so the liveness
    // enumeration and the consumed value section stay aligned
    // (resume.py consume_boxes).
    let mut pending_value_index: Option<usize> = None;
    if let Some(color) = pending_result_color {
        if reg_indices.int.iter().any(|&c| c as usize == color)
            || reg_indices.float.iter().any(|&c| c as usize == color)
        {
            decline!("PendingResultNotRef");
        }
        if let Some(ref_pos) = reg_indices.ref_.iter().position(|&c| c as usize == color) {
            pending_value_index = Some(reg_indices.int.len() + ref_pos);
            reg_indices.ref_.remove(ref_pos);
        }
    }
    let values: Vec<&majit_ir::resumedata::RebuiltValue> = match pending_value_index {
        Some(idx) => frame
            .values
            .iter()
            .enumerate()
            .filter_map(|(i, v)| if i == idx { None } else { Some(v) })
            .collect(),
        None => frame.values.iter().collect(),
    };
    // resume.py consume_boxes: the liveness enumeration count must match
    // the (pending-excluded) encoded frame section exactly.
    if reg_indices.total_len() != values.len() {
        decline!("LivenessValueCountMismatch");
    }
    // resume.py rebuild_from_resumedata hands `consume_boxes` all three banks
    // (`f.registers_i, f.registers_r, f.registers_f`), so decode them all here.
    //
    // The int/float banks decode COLOR-indexed and STAY that way, unlike the
    // Ref bank which is inverted to `locals_cells_stack_w` slots below.  That
    // array is a W_Root array (virtualizable.py read_boxes), so an unboxed temp has
    // no slot to invert a color into; it has no home outside the register file
    // in the first place, which is exactly how `registers_i` holds it upstream.
    // `PyreSym::registers_i` is likewise read by color (trace_opcode.rs), and
    // `assemble_bridge_inline_pending` copies the bank across unchanged.
    //
    // Decoded before the portal-callee branch below so both its returns carry
    // the banks, and so the section offsets are computed once: the encoder
    // enumerates live boxes in bank order and `values` mirrors it, so the Ref
    // section starts at `int_len` (`pending_value_index` above already removed
    // the pending call result from within that section).
    let int_len = reg_indices.int.len();
    let mut registers_i: Vec<OpRef> = Vec::new();
    let mut registers_f: Vec<OpRef> = Vec::new();
    for (pos, &reg_idx) in reg_indices.int.iter().enumerate() {
        let (op, val) = bridge_decode_box(
            ctx,
            values[pos],
            Type::Int,
            rd_virtuals,
            resume_data,
            fail_values,
            fail_types,
            backend,
            cache,
        );
        // The failarg's value at this guard, threaded onto the OpRef-keyed
        // shadow the same way the frame red's is: without it the re-executed
        // callee reads the register as symbolic-only and stops at the first
        // condition it has to fold (`GotoIfNotValueNotConcrete`).
        ctx.try_set_opref_concrete(op, val);
        let reg_idx = reg_idx as usize;
        if reg_idx >= registers_i.len() {
            registers_i.resize(reg_idx + 1, OpRef::NONE);
        }
        registers_i[reg_idx] = op;
    }
    let float_base = int_len + reg_indices.ref_.len();
    for (pos, &reg_idx) in reg_indices.float.iter().enumerate() {
        let (op, val) = bridge_decode_box(
            ctx,
            values[float_base + pos],
            Type::Float,
            rd_virtuals,
            resume_data,
            fail_values,
            fail_types,
            backend,
            cache,
        );
        ctx.try_set_opref_concrete(op, val);
        let reg_idx = reg_idx as usize;
        if reg_idx >= registers_f.len() {
            registers_f.resize(reg_idx + 1, OpRef::NONE);
        }
        registers_f[reg_idx] = op;
    }
    // The Ref section alone, so every `reg_indices.ref_` position below indexes
    // it directly instead of carrying the `int_len` offset to each use site.
    let ref_values: Vec<&majit_ir::resumedata::RebuiltValue> = values[int_len..float_base].to_vec();

    // Virtualizable-callee shape (pyre's "every function is its own portal"
    // model): the callee's live ref registers are the portal reds [frame, ec]
    // (and, at a mid-expression pause, operands kept live across a CALL), while
    // its locals live in the `frame` red's own
    // `locals_cells_stack_w` virtual array — the same place the ROOT frame keeps
    // its locals — NOT in the register section. Recover them into REGISTER-
    // SECTION slots so the bridge rebuilds the callee as a plain MIFrame
    // (resume.py newframe + reload, reading `registers_r`), convergent
    // with the RPython frame shape and creating NO new vable. The color==
    // semantic identity check below only accepts a register-section callee
    // (locals at colors 0..nlocals), which this shape never satisfies.
    //
    // An `in_a_call` callee suspended at one of its own CALL sites carries a
    // pending result whose register was already dropped from the ref bank above
    // (`pending_value_index`). When that pending-result color coincides with the
    // `ec` portal red, the removal leaves only `[frame]` live — a one-ref
    // reduced form of the SAME portal-callee shape. `pec_reg` is consumed only
    // by this classifier gate; the reconstruction body threads
    // `pframe_reg` -> `frame_vidx` -> the `locals_cells_stack_w` array and never
    // reads `ec`, so `[frame]` alone is still faithfully reconstructable (the
    // pending operand slot is delivered by `make_result_of_lastop`, not from
    // resumedata, and is skipped by `pending_result_abs_slot` below).
    let (pframe_reg, pec_reg) = portal_red_regs_at(frame.jitcode_index);
    let (pframe_reg, pec_reg) = (pframe_reg as u32, pec_reg as u32);
    let has_frame = reg_indices.ref_.contains(&pframe_reg);
    let maps = bridge_semantic_maps_from_jitcode_pc(frame.jitcode_index, frame.pc);
    // resume.py `rebuild_from_resumedata` / `consume_boxes` rebuilds
    // every paused frame and fills EVERY live register from the resume stream
    // with no shape classifier. Pyre recovers a portal callee's locals from the
    // `frame` red's own `locals_cells_stack_w` virtual array (below) rather than
    // the register section, so accept any paused portal callee whose every live
    // non-red color names a semantic slot at this pc (the `pcdep` inversion the
    // encoder used): the reds are recovered through the `frame`, and each live
    // operand kept across a CALL — e.g. `result1` in `f(a) + g(b)` at the second
    // CALL — is decoded semantic-ordered from the flushed array. This single
    // predicate subsumes the canonical `[frame, ec]` and the reduced `[frame]`
    // in-a-call shapes; a slot the pause-time flush did not cover still declines
    // at the mandatory-operand check below.
    let portal_callee = has_frame
        && reg_indices.ref_.iter().all(|&c| {
            c == pframe_reg
                || c == pec_reg
                || semantic_ref_slot_for_reg_color(
                    nlocals,
                    stack_only,
                    &maps.pcdep_entries,
                    c as usize,
                )
                .is_some()
        });
    if pframe_reg != u32::from(u16::MAX) && pec_reg != u32::from(u16::MAX) && portal_callee {
        // Distinguish the sibling-operand admission (a live color beyond the
        // portal reds) from the canonical reds-only shape so the corpus shows
        // what exercises the widened acceptance.
        if reg_indices
            .ref_
            .iter()
            .any(|&c| c != pframe_reg && c != pec_reg)
        {
            crate::jitcode_dispatch::census_record("P2Recipe::PortalSiblingAdmit");
        }
        use majit_ir::resumedata::{RebuiltValue, TAGVIRTUAL, UNINITIALIZED_TAG, untag};
        let Some(frame_pos) = reg_indices.ref_.iter().position(|&c| c == pframe_reg) else {
            decline!("NoFrameRedPosition");
        };
        let valuestackdepth = nlocals + stack_only;
        // resume.py `consume_boxes` refills a paused frame from the
        // resume stream alone.  Both frame images read below are pyre's
        // stand-in for the slots that stream never names — a portal callee
        // keeps its locals in `locals_cells_stack_w` rather than the register
        // section — but wherever the stream DOES name a slot it is
        // authoritative: that slot lives in a machine register neither image
        // has been written back from.  A residual call's result at the guard is
        // exactly that shape, and reading the image there yields NULL (the
        // stale pre-call array entry for a forced frame, the never-stored
        // `NewArrayClear` entry for a virtual one), so the next residual
        // consuming it folds its Ref arg to concrete NULL
        // (`walker_abort_if_mayforce_null_ref_arg`) and no bridge is ever
        // built.  Invert color -> slot with the same `pcdep` map the encoder
        // used and take those slots from the stream instead.
        let mut stream_slots: Vec<(usize, usize)> = Vec::new();
        let mut stream_covered = vec![false; valuestackdepth];
        for (pos, &color) in reg_indices.ref_.iter().enumerate() {
            let slots = semantic_ref_slots_for_reg_color(
                nlocals,
                stack_only,
                &maps.pcdep_entries,
                color as usize,
            );
            if slots.is_empty() {
                // A portal red is a scratch value only when this per-PC map
                // gives its color no semantic owner.  The entry-time frame/ec
                // colors may be reused for live operands at arbitrary resume
                // PCs; in that case the resume stream is authoritative and
                // must overlay the callee's own frame image below.
                continue;
            }
            for slot in slots {
                // The pending call result is delivered by
                // `make_result_of_lastop`, not from resumedata, so its slot
                // stays unwritten here.
                if slot >= valuestackdepth
                    || Some(slot) == pending_result_abs_slot
                    || stream_covered[slot]
                {
                    continue;
                }
                stream_covered[slot] = true;
                stream_slots.push((pos, slot));
            }
        }
        if !stream_slots.is_empty() {
            crate::jitcode_dispatch::census_record("P2Recipe::StreamSlotOverImage");
        }
        // resume.py rebuild_from_resumedata consumes the saved boxes
        // for every frame without requiring the frame object itself to remain
        // virtual.  A Pyre callee frame can be forced before the guard (for
        // example Random.shuffle's loop frame), in which case the frame red is
        // a Box failarg rather than TAGVIRTUAL.  Its semantic locals/stack are
        // still recoverable from that frame's own locals_cells_stack_w array.
        //
        // Load them symbolically from the failarg-owned frame instead of
        // baking the first guard hit's concrete values as constants.  The
        // reconstructed bridge therefore reads the current frame image on
        // every execution, exactly as consume_boxes rebuilds a fresh MIFrame
        // from the current deadframe.
        if !matches!(ref_values[frame_pos], RebuiltValue::Virtual(_)) {
            let (frame_box, frame_value) = bridge_decode_box(
                ctx,
                ref_values[frame_pos],
                Type::Ref,
                rd_virtuals,
                resume_data,
                fail_values,
                fail_types,
                backend,
                cache,
            );
            let majit_ir::Value::Ref(frame_ref) = frame_value else {
                return None;
            };
            ctx.try_set_opref_concrete(frame_box, frame_value);

            let (mut registers_r, mut concrete_r) = reconstruct_materialized_frame_slots(
                ctx,
                frame_box,
                frame_ref,
                valuestackdepth,
                pending_result_abs_slot,
                &stream_covered,
                pending_ref_array_writes,
            )?;
            overlay_stream_ref_slots(
                ctx,
                &stream_slots,
                &ref_values,
                &mut registers_r,
                &mut concrete_r,
                rd_virtuals,
                resume_data,
                fail_values,
                fail_types,
                backend,
                cache,
            );
            crate::jitcode_dispatch::census_record("P2Recipe::MaterializedFrameAdmit");
            return Some(ReconstructRecipe {
                code_ptr: raw_code as *const (),
                jitcode_index: frame.jitcode_index,
                jitcode_pc: frame.pc,
                nlocals,
                valuestackdepth,
                registers_i,
                registers_r,
                registers_f,
                concrete_r,
                frame: frame_box,
                nargs: frame_nlocals,
                return_substitute: None,
                len_tail: false,
                rebuilt_frame: None,
                rebuilt_ec: None,
            });
        }
        let RebuiltValue::Virtual(frame_vidx) = ref_values[frame_pos] else {
            decline!("FrameRedNotVirtual");
        };
        // The `frame` red virtual is a PyFrame VirtualInfo; its
        // `locals_cells_stack_w` array field is at PYFRAME_LOCALS_CELLS_STACK_OFFSET.
        let array_vidx = {
            let Some(frame_virtual) = rd_virtual_at(rd_virtuals, *frame_vidx) else {
                decline!("FrameVirtualMissing");
            };
            let majit_ir::RdVirtualInfo::VirtualInfo {
                fieldnums,
                fielddescrs,
                ..
            } = frame_virtual
            else {
                decline!("FrameVirtualNotStruct");
            };
            let Some(arr_field_idx) = fielddescrs
                .iter()
                .position(|fd| fd.offset == crate::frame_layout::PYFRAME_LOCALS_CELLS_STACK_OFFSET)
            else {
                decline!("NoLocalsArrayField");
            };
            let Some(&arr_fieldnum) = fieldnums.get(arr_field_idx) else {
                decline!("LocalsArrayFieldnumMissing");
            };
            let (av, tb) = untag(arr_fieldnum);
            if tb != TAGVIRTUAL {
                decline!("LocalsArrayNotVirtual");
            }
            if av < 0 {
                (rd_virtuals.map_or(0, |v| v.len()) as i32 + av) as usize
            } else {
                av as usize
            }
        };
        let Some(array_virtual) = rd_virtual_at(rd_virtuals, array_vidx) else {
            decline!("LocalsArrayVirtualMissing");
        };
        let arr: Vec<i16> = match array_virtual {
            majit_ir::RdVirtualInfo::VArrayInfoClear { fieldnums, .. }
            | majit_ir::RdVirtualInfo::VArrayInfoNotClear { fieldnums, .. } => fieldnums.clone(),
            _ => decline!("LocalsArrayNotArrayInfo"),
        };
        if valuestackdepth > arr.len() {
            decline!("LocalsArrayTooShort");
        }
        let callinfocollection = ctx.callinfocollection.clone();
        let mut registers_r = vec![OpRef::NONE; valuestackdepth];
        let mut concrete_r = vec![majit_ir::Value::Void; valuestackdepth];
        // locals_cells_stack_w is SEMANTIC-slot-ordered ([locals | stack]) and a
        // W_Root array — every live slot is Ref. An UNINITIALIZED local stays
        // NULL in the rebuilt frame; a missing operand-stack slot is
        // unreconstructable and declines below.
        for k in 0..valuestackdepth {
            if Some(k) == pending_result_abs_slot || stream_covered[k] {
                continue;
            }
            let tag = arr[k];
            if tag == UNINITIALIZED_TAG {
                continue;
            }
            registers_r[k] = decode_fieldnum(ctx, tag, rd_virtuals, resume_data, cache);
            let bits = decode_tagged_concrete(
                tag,
                Type::Ref,
                rd_virtuals,
                fail_values,
                resume_data.num_failargs,
                resume_data.storage.as_ref(),
                backend,
                callinfocollection.as_ref(),
                cache,
            );
            concrete_r[k] = value_for_slot(Type::Ref, bits);
        }
        // Runs before the mandatory-operand check: a stack slot the virtual
        // array never stored is `UNINITIALIZED`/clear-NULL there but present in
        // the stream, so the overlay is what makes it reconstructable.
        overlay_stream_ref_slots(
            ctx,
            &stream_slots,
            &ref_values,
            &mut registers_r,
            &mut concrete_r,
            rd_virtuals,
            resume_data,
            fail_values,
            fail_types,
            backend,
            cache,
        );
        for s in nlocals..valuestackdepth {
            if Some(s) == pending_result_abs_slot
                || unreconstructable_operand_floor.is_some_and(|floor| s >= floor)
            {
                continue;
            }
            if registers_r[s] == OpRef::NONE {
                if crate::jitcode_dispatch::p2_diag_enabled() {
                    let sem: Vec<(u32, Option<usize>)> = reg_indices
                        .ref_
                        .iter()
                        .map(|&c| {
                            (
                                c,
                                semantic_ref_slot_for_reg_color(
                                    nlocals,
                                    stack_only,
                                    &maps.pcdep_entries,
                                    c as usize,
                                ),
                            )
                        })
                        .collect();
                    eprintln!(
                        "[p2-recipe] jitcode_index={} pc={} py_pc={py_pc} in_a_call={in_a_call} \
                         nlocals={nlocals} stack_only={stack_only} vsd={valuestackdepth} \
                         missing={s} arr={arr:?} pframe={pframe_reg} pec={pec_reg} live_ref={sem:?}",
                        frame.jitcode_index, frame.pc
                    );
                }
                crate::jitcode_dispatch::census_record("P2Recipe::MandatoryOperandMissing");
                return None;
            }
        }
        // The frame red itself.  A scope the parent trace still had open has
        // already materialized this virtual (the `virtualref_boxes` decode), so
        // the cache answers with that same box.
        let (frame_red, frame_red_value) = bridge_decode_box(
            ctx,
            ref_values[frame_pos],
            Type::Ref,
            rd_virtuals,
            resume_data,
            fail_values,
            fail_types,
            backend,
            cache,
        );
        ctx.try_set_opref_concrete(frame_red, frame_red_value);
        return Some(ReconstructRecipe {
            code_ptr: raw_code as *const (),
            jitcode_index: frame.jitcode_index,
            jitcode_pc: frame.pc,
            nlocals,
            valuestackdepth,
            registers_i,
            registers_r,
            registers_f,
            concrete_r,
            frame: frame_red,
            nargs: frame_nlocals,
            return_substitute: None,
            len_tail: false,
            rebuilt_frame: None,
            rebuilt_ec: None,
        });
    }

    // The recipe banks are written and read by the register COLOR reported in
    // the liveness stream (`registers_r[reg_idx]` below), but every consumer
    // indexes by the SEMANTIC `locals_cells_stack_w` slot: the retired MIFrame's
    // `load_local_value` reads `registers_r[local_idx]` (trace_opcode.rs)
    // and `stack_slot_reg_idx` reads `registers_r[nlocals + stack_idx]`
    // (trace_opcode.rs), and `assemble_bridge_inline_pending` reads
    // `concrete_r[k]` by semantic slot k. This recipe is therefore only valid
    // when each live color equals the semantic slot it denotes (color ==
    // semantic). Regalloc pins that identity at call-boundary resume pcs (a
    // straight-line callee resumes after its CALL with locals at colors
    // `0..nlocals`), but a MID-BODY resume — e.g. a guard fired inside a
    // callee branch (`goto_if_not` target) — coalesces colors so the live
    // stack value sits at a renamed color. #73: the per-PC `pcdep_entries`
    // color→slot map is the authoritative inversion; when it is empty at this
    // resume pc there is no per-pc map to faithfully rebuild the frame, so
    // decline to the single-frame bridge (whose vable payload IS semantic-
    // ordered) rather than rebuild the frame with mis-slotted boxes.
    if maps.pcdep_entries.is_empty() {
        crate::jitcode_dispatch::census_record("P2Recipe::PortalShapeRejected");
        return None;
    }
    for &color in &reg_indices.ref_ {
        match semantic_ref_slot_for_reg_color(
            nlocals,
            stack_only,
            &maps.pcdep_entries,
            color as usize,
        ) {
            // color == semantic slot: the recipe's color-indexed fill is
            // also the correct semantic fill.
            Some(semantic_idx) if semantic_idx == color as usize => {}
            // A live color whose semantic slot differs (or is unmappable):
            // the color-indexed recipe would mis-slot it. Decline.
            _ => {
                crate::jitcode_dispatch::census_record("P2Recipe::PortalShapeRejected");
                return None;
            }
        }
    }

    // The Ref bank decodes COLOR-indexed (the liveness register `reg_idx` is the
    // post-regalloc color). The reconstructed inline frame's `registers_r`/
    // `concrete_r` are read SLOT-indexed (`assemble_bridge_inline_pending` seeds
    // `locals_cells_stack_w[k]` and the resumed tracer reads LOAD_FAST's
    // `nlocals + stack_idx`), so invert color→slot below — for a borrowed local
    // pushed on the stack the color ≠ slot, and a color-indexed seed lands the
    // value at the wrong slot.  The int/float banks were decoded above and need
    // no inversion.
    let mut by_color_r: Vec<OpRef> = Vec::new();
    let mut by_color_c: Vec<majit_ir::Value> = Vec::new();
    let mut value_cursor = int_len;
    for &reg_idx in &reg_indices.ref_ {
        let (op, val) = bridge_decode_box(
            ctx,
            values[value_cursor],
            Type::Ref,
            rd_virtuals,
            resume_data,
            fail_values,
            fail_types,
            backend,
            cache,
        );
        let reg_idx = reg_idx as usize;
        if reg_idx >= by_color_r.len() {
            by_color_r.resize(reg_idx + 1, OpRef::NONE);
            by_color_c.resize(reg_idx + 1, majit_ir::Value::Void);
        }
        by_color_r[reg_idx] = op;
        by_color_c[reg_idx] = val;
        value_cursor += 1;
    }

    // `pyframe.py valuestackdepth`: locals + cells + stack. Pure cellvars
    // are gated out above; a cellvar that overlaps a varname is that
    // local's slot.
    // The semantic `valuestackdepth` is `stack_base() + operand_depth`, where
    // `operand_depth` is the logical stack height the codewriter's forward
    // dataflow computes for this pc (`LiveVars::stack_depth_at`). The portal
    // frame reads the equivalent figure from the encoded vable
    // `valuestackdepth` scalar (`state.rs`); an inline frame has no such
    // scalar, so derive it from the bytecode here. An unreachable resume pc
    // aborts the multi-frame path.
    let valuestackdepth = nlocals + stack_only;

    // Invert the COLOR-indexed decode into SLOT-indexed `registers_r`/
    // `concrete_r`, mirroring the root frame's `setup_bridge_sym` color→slot
    // mirror. `maps.pcdep_entries` is the per-PC `(color, slot)` map (non-empty
    // — the identity gate above declined otherwise): each decoded color is
    // placed at the semantic slot it denotes at this resume pc, so a borrowed
    // local on the operand stack lands at its stack slot, not its color.
    let mut registers_r = vec![OpRef::NONE; valuestackdepth];
    let mut concrete_r = vec![majit_ir::Value::Void; valuestackdepth];
    for &(bank, color, slot) in &maps.pcdep_entries {
        if bank != 1 {
            continue;
        } // Ref bank only
        let s = slot as usize;
        let c = color as usize;
        if s < valuestackdepth && c < by_color_r.len() && by_color_r[c] != OpRef::NONE {
            registers_r[s] = by_color_r[c];
            concrete_r[s] = by_color_c[c];
        }
    }

    // Refill registerless operand-stack constants the color map omits (an
    // inlined callee has no value-stack resumedata to rematerialize them from).
    for (slot, raw) in const_ref_slots_from_pc(frame.jitcode_index, frame.pc) {
        let s = slot as usize;
        if Some(s) == pending_result_abs_slot {
            continue;
        }
        if s < valuestackdepth {
            registers_r[s] = ctx.const_ref(raw);
            concrete_r[s] = value_for_slot(Type::Ref, raw);
        }
    }

    // Every operand-stack slot is live (the logical stack is dense). If one is
    // still unfilled after the color→slot inversion and the constant refill,
    // the slot holds a value neither path can reconstruct (an int/float operand
    // constant, or an unsupported shape) — decline to the single-frame bridge
    // rather than seed a NULL operand the re-executed bridge would deref.
    // This path rebuilds from resumedata colors exactly as the virtual-array
    // one above does, so it owes the operands a paused call consumed the same
    // exemption: no callee ever flushed them and the resume collapses them to
    // the result `make_result_of_lastop` delivers.
    for s in nlocals..valuestackdepth {
        if Some(s) == pending_result_abs_slot
            || unreconstructable_operand_floor.is_some_and(|floor| s >= floor)
        {
            continue;
        }
        if registers_r[s] == OpRef::NONE {
            decline!("MandatoryOperandMissing");
        }
    }

    Some(ReconstructRecipe {
        code_ptr: raw_code as *const (),
        jitcode_index: frame.jitcode_index,
        jitcode_pc: frame.pc,
        nlocals,
        valuestackdepth,
        registers_i,
        registers_r,
        registers_f,
        concrete_r,
        frame: OpRef::NONE,
        nargs: frame_nlocals,
        return_substitute: None,
        len_tail: false,
        rebuilt_frame: None,
        rebuilt_ec: None,
    })
}

fn bridge_decode_box(
    ctx: &mut majit_metainterp::TraceCtx,
    v: &majit_ir::resumedata::RebuiltValue,
    expected_kind: Type,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    resume_data: &majit_metainterp::ResumeDataResult,
    fail_values: &[i64],
    fail_types: &[Type],
    backend: &dyn majit_backend::Backend,
    cache: &mut BridgeVirtualCache<'_>,
) -> (OpRef, majit_ir::Value) {
    use majit_ir::resumedata::RebuiltValue;
    let callinfocollection = ctx.callinfocollection.clone();
    match v {
        RebuiltValue::Box(n, tp) => {
            let opref = OpRef::input_arg_typed(*n as u32, *tp);
            let bits = fail_values[*n];
            let effective_tp = fail_types.get(*n).copied().unwrap_or(*tp);
            // resume.py:1264 assert box.type == kind
            assert!(
                effective_tp == expected_kind,
                "bridge_decode_box: Box({n}) type {effective_tp:?} != expected {expected_kind:?}"
            );
            (opref, value_for_slot(effective_tp, bits))
        }
        RebuiltValue::Const(c) => {
            let opref = match c.get_type() {
                Type::Ref => ctx.const_ref(c.getref_base().as_usize() as i64),
                Type::Float => ctx.const_float(c.getfloatstorage()),
                _ => ctx.const_int(c.getint()),
            };
            // resume.py:1264 assert box.type == kind
            assert!(
                c.get_type() == expected_kind,
                "bridge_decode_box: Const type {:?} != expected {:?}",
                c.get_type(),
                expected_kind,
            );
            (opref, value_for_slot(c.get_type(), c.as_raw_i64()))
        }
        RebuiltValue::Virtual(vidx) => {
            // resume.py `getvirtual_ptr`: a cache hit returns the box
            // `allocate()` already filled. Capture that before this call
            // writes the slot.
            let already = cache.get_any(*vidx).is_some() || ctx.bridge_virtual_op(*vidx).is_some();
            let opref = materialize_bridge_virtual(ctx, *vidx, rd_virtuals, resume_data, cache);
            // The applying reader allocated the object at the `NEW` it
            // recorded and stamped it there; that object is the box's value.
            // Allocating another here would give the walk a second object
            // for one box.
            if let Some(value) = ctx.concrete_of_opref(opref)
                && value.get_type() == expected_kind
            {
                return (opref, value);
            }
            if expected_kind != Type::Int
                && let Some(gcref) = cache.get_concrete_ptr(*vidx)
            {
                ctx.try_set_opref_concrete(opref, majit_ir::Value::Ref(gcref));
                return (opref, majit_ir::Value::Ref(gcref));
            }
            if already {
                // Later TAGVIRTUAL: `virtuals_cache.get_ptr` hit. `allocate()`
                // already ran setfields; a second concrete SETFIELD writes
                // another `wrappeditems` array against the live tuple.
                // A recording-only hit still has to return the DirectReader
                // Box (`resume.py` `getvirtual_ptr`); Void is only the case
                // with no object on either cache.
                return (opref, majit_ir::Value::Void);
            }
            if expected_kind == Type::Int {
                let value = materialize_concrete_virtual_int(
                    *vidx,
                    rd_virtuals,
                    fail_values,
                    resume_data.num_failargs,
                    resume_data.storage.as_ref(),
                    backend,
                    callinfocollection.as_ref(),
                    cache,
                );
                (opref, majit_ir::Value::Int(value))
            } else {
                let gcref = materialize_concrete_virtual_ptr(
                    *vidx,
                    rd_virtuals,
                    fail_values,
                    resume_data.num_failargs,
                    resume_data.storage.as_ref(),
                    backend,
                    callinfocollection.as_ref(),
                    cache,
                );
                (opref, majit_ir::Value::Ref(gcref))
            }
        }
        RebuiltValue::Unassigned => (OpRef::NONE, majit_ir::Value::Void),
    }
}

/// resume.py `AbstractResumeDataReader._prepare`: virtual cache then
/// `_prepare_pendingfields`. Called before `newframe` / `consume_boxes`.
fn apply_bridge_prepare(
    sym: &mut PyreSym,
    ctx: &mut majit_metainterp::TraceCtx,
    resume_data: &majit_metainterp::ResumeDataResult,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    fail_values: &[i64],
    fail_types: &[Type],
    reader: &mut BridgeVirtualCache<'_>,
) {
    let (driver, _) = crate::driver::driver_pair();
    let backend = driver.meta_interp().backend();
    // resume.py `_prepare_virtuals`: the reader's cache starts with unset
    // holes (`virtual_ptr_default`, matching `rd_virtual is None`).
    // `getvirtual_ptr` lazily `allocate()`s on the first TAGVIRTUAL /
    // pending-field decode. `force_all_virtuals` is `force_from_resumedata`
    // only (direct/executing reader), not this BoxReader `_prepare`.
    prepare_bridge_pending_fields(
        sym,
        ctx,
        resume_data,
        rd_virtuals,
        fail_values,
        fail_types,
        backend,
        reader,
    );
}

/// Prepare the guard-owned pending field stream for bridge tracing.
/// `resume.py:_prepare_pendingfields` decodes these tagged values only after
/// virtual preparation; the exception-channel entry seeds the bridge walk's
/// current exception, while ordinary entries are re-emitted as trace ops
/// instead of being applied directly to the live execution context.
#[allow(clippy::too_many_arguments)]
fn prepare_bridge_pending_fields(
    sym: &mut PyreSym,
    ctx: &mut majit_metainterp::TraceCtx,
    resume_data: &majit_metainterp::ResumeDataResult,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    fail_values: &[i64],
    fail_types: &[Type],
    backend: &dyn majit_backend::Backend,
    cache: &mut BridgeVirtualCache<'_>,
) {
    let Some(storage) = resume_data.storage.as_ref() else {
        return;
    };
    let target_descr = crate::descr::ec_sys_exc_value_descr();
    for pending in storage.rd_pendingfields() {
        let Some(descr) = pending.descr.as_ref() else {
            continue;
        };
        let is_exc_channel = pending.item_index < 0 && std::sync::Arc::ptr_eq(descr, &target_descr);

        // `resume.py` `_prepare_pendingfields`: both operands use the same tagged decoder as
        // frame boxes.  Decode the target as well as the fieldbox so virtual
        // preparation and malformed-tag checks stay aligned with deopt.
        let rd_consts = storage.rd_consts();
        let target = majit_ir::resumedata::decode_tagged_value(
            pending.target_tagged,
            resume_data.num_failargs,
            rd_consts,
            &resume_data.fail_arg_types,
            storage.rd_virtuals.len(),
        );
        let value = majit_ir::resumedata::decode_tagged_value(
            pending.value_tagged,
            resume_data.num_failargs,
            rd_consts,
            &resume_data.fail_arg_types,
            storage.rd_virtuals.len(),
        );

        if is_exc_channel {
            let _ = bridge_decode_box(
                ctx,
                &target,
                Type::Ref,
                rd_virtuals,
                resume_data,
                fail_values,
                fail_types,
                backend,
                cache,
            );
            let (value_box, value_concrete) = bridge_decode_box(
                ctx,
                &value,
                Type::Ref,
                rd_virtuals,
                resume_data,
                fail_values,
                fail_types,
                backend,
                cache,
            );
            let majit_ir::Value::Ref(value_ref) = value_concrete else {
                continue;
            };
            sym.last_exc_box = value_box;
            sym.last_exc_value
                .set(value_ref.as_usize() as pyre_object::PyObjectRef);
            sym.class_of_last_exc_is_const = false;
        } else {
            // resume.py `_prepare_pendingfields`: replay ordinary
            // pending writes as bridge-entry SETFIELD_GC / SETARRAYITEM_GC ops.
            let (target_op, target_concrete) = bridge_decode_box(
                ctx,
                &target,
                Type::Ref,
                rd_virtuals,
                resume_data,
                fail_values,
                fail_types,
                backend,
                cache,
            );
            // history.py FrontendOp parity: a decoded live target carries its
            // runtime value on the box before decoding the value can allocate
            // and trigger a moving collection.
            ctx.try_set_opref_concrete(target_op, target_concrete);
            let value_kind = if pending.item_index < 0 {
                descr
                    .as_field_descr()
                    .map(|fd| fd.field_type())
                    .unwrap_or(Type::Ref)
            } else {
                descr
                    .as_array_descr()
                    .map(|ad| ad.item_type())
                    .unwrap_or(Type::Ref)
            };
            let (value_op, value_concrete) = bridge_decode_box(
                ctx,
                &value,
                value_kind,
                rd_virtuals,
                resume_data,
                fail_values,
                fail_types,
                backend,
                cache,
            );
            // The value box has the same FrontendOp value contract.  The target
            // stamp above also lets the later portal-frame consume recover this
            // exact pending-target box after a moving GC.
            ctx.try_set_opref_concrete(value_op, value_concrete);
            // ResumeDataBoxReader._prepare_pendingfields dispatches through
            // `execute_setfield_gc` / `execute_setarrayitem_gc`: execute the
            // write on the recording-time heap as well as recording it.  The
            // concrete execution is observable immediately when the following
            // frame-section consume reads a materialized PyFrame's locals
            // array; recording alone would leave that consume one write stale.
            // A target, descriptor or value shape this backend cannot execute
            // still has to be RECORDED below: `_prepare_pendingfields` applies
            // every decoded write, so leaving the loop here would drop the write
            // from bridge entry entirely.  Leave the execution block instead.
            'execute: {
                let target_ptr = match ctx.box_value(target_op).unwrap_or(target_concrete) {
                    majit_ir::Value::Ref(ptr) if !ptr.is_null() => ptr.as_usize() as i64,
                    _ => break 'execute,
                };
                if pending.item_index < 0 {
                    let Some(fielddescr) = descr.as_field_descr() else {
                        break 'execute;
                    };
                    let bh_descr = majit_jitcode::jitcode::BhDescr::from_field_descr(fielddescr);
                    match (fielddescr.field_type(), value_concrete) {
                        (Type::Ref, majit_ir::Value::Ref(value)) => {
                            backend.bh_setfield_gc_r(target_ptr, value, &bh_descr)
                        }
                        (Type::Float, majit_ir::Value::Float(value)) => {
                            backend.bh_setfield_gc_f(target_ptr, value, &bh_descr)
                        }
                        (Type::Int, majit_ir::Value::Int(value)) => {
                            backend.bh_setfield_gc_i(target_ptr, value, &bh_descr)
                        }
                        _ => break 'execute,
                    }
                } else {
                    let Some(arraydescr) = descr.as_array_descr() else {
                        break 'execute;
                    };
                    let bh_descr = majit_jitcode::jitcode::BhDescr::from_array_descr(arraydescr);
                    let index = i64::from(pending.item_index);
                    match (arraydescr.item_type(), value_concrete) {
                        (Type::Ref, majit_ir::Value::Ref(value)) => {
                            backend.bh_setarrayitem_gc_r(target_ptr, index, value, &bh_descr);
                            cache.push_pending_ref_array_write(PendingRefArrayWrite {
                                target: target_op,
                                item_index: pending.item_index as usize,
                                value: value_op,
                            });
                        }
                        (Type::Float, majit_ir::Value::Float(value)) => {
                            backend.bh_setarrayitem_gc_f(target_ptr, index, value, &bh_descr)
                        }
                        (Type::Int, majit_ir::Value::Int(value)) => {
                            backend.bh_setarrayitem_gc_i(target_ptr, index, value, &bh_descr)
                        }
                        _ => break 'execute,
                    }
                }
            }
            // resume.py _prepare_pendingfields op-emission: replay the
            // decoded write as a bridge-entry SETFIELD_GC / SETARRAYITEM_GC and
            // seed the heapcache after recording so a later same-slot get folds
            // against it. The pending target is always non-virtual (resume.py
            // asserts only the fieldbox is virtual), so the bridge-body read hits
            // the plain get heapcache path with no on-hit sanity load, and the
            // replayed store runs at bridge entry before any body op.
            // The direct reader that preceded this entry applied these writes
            // already, so this cache is the recording-only one and the call
            // records without storing.
            majit_metainterp::emit_pending_field_op(
                ctx,
                target_op,
                value_op,
                pending.item_index,
                descr,
                cache,
            );
        }
    }
}

/// resume.py decode_box concrete parity for tagged fieldnums.
/// Converts a tagged i16 (from `rd_virtuals[*].fieldnums`) into raw i64
/// bits suitable for backend concrete setters/calls.
fn decode_tagged_concrete(
    tagged: i16,
    expected_kind: Type,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    fail_values: &[i64],
    num_failargs: i32,
    storage: Option<&std::sync::Arc<majit_metainterp::resume::ResumeStorage>>,
    backend: &dyn majit_backend::Backend,
    callinfocollection: Option<&std::sync::Arc<majit_ir::CallInfoCollection>>,
    cache: &mut BridgeVirtualCache<'_>,
) -> i64 {
    use majit_ir::resumedata::{
        NULLREF, TAG_CONST_OFFSET, TAGBOX, TAGCONST, TAGINT, TAGVIRTUAL, UNINITIALIZED_TAG, untag,
    };
    if tagged == UNINITIALIZED_TAG {
        return 0;
    }
    let (val, tagbits) = untag(tagged);
    match tagbits {
        TAGBOX => {
            let idx = if val < 0 {
                (val + num_failargs) as usize
            } else {
                val as usize
            };
            // resume.py:1261 fail-loud: direct indexing, not silent fallback
            fail_values[idx]
        }
        TAGINT => val as i64,
        TAGCONST => {
            if tagged == NULLREF {
                return 0;
            }
            let ci = (val - TAG_CONST_OFFSET) as usize;
            // resume.py:1251 fail-loud: direct indexing
            let storage = storage.expect("decode_tagged_concrete: TAGCONST requires storage");
            storage.rd_consts()[ci].as_raw_i64()
        }
        TAGVIRTUAL => {
            // resume.py assign_number_to_virtual nested virtuals are numbered negatively;
            // getvirtual resolves them via Python negative list indexing
            // into rd_virtuals (resume.py:951-954).
            let vidx = if val < 0 {
                (rd_virtuals.map_or(0, |v| v.len()) as i32 + val) as usize
            } else {
                val as usize
            };
            if expected_kind == Type::Int {
                materialize_concrete_virtual_int(
                    vidx,
                    rd_virtuals,
                    fail_values,
                    num_failargs,
                    storage,
                    backend,
                    callinfocollection,
                    cache,
                )
            } else {
                let gcref = materialize_concrete_virtual_ptr(
                    vidx,
                    rd_virtuals,
                    fail_values,
                    num_failargs,
                    storage,
                    backend,
                    callinfocollection,
                    cache,
                );
                gcref.0 as i64
            }
        }
        _ => 0,
    }
}

fn bh_field_descr_from_info(fd: &majit_ir::FieldDescrInfo) -> majit_jitcode::jitcode::BhDescr {
    majit_jitcode::jitcode::BhDescr::Field {
        offset: fd.offset,
        field_size: fd.field_size,
        field_type: fd.field_type,
        field_flag: majit_ir::ArrayFlag::from_field_type(fd.field_type),
        is_field_signed: matches!(fd.field_type, Type::Int),
        is_immutable: false,
        is_quasi_immutable: false,
        // A live `FieldDescrInfo` carries a resolved index.
        index_in_parent: Some(fd.index as usize),
        parent: None,
        name: String::new(),
        owner: String::new(),
    }
}

fn bh_size_descr_from_size_descr(
    size_descr: &dyn majit_ir::descr::SizeDescr,
    vtable: usize,
) -> majit_jitcode::jitcode::BhDescr {
    majit_jitcode::jitcode::BhDescr::Size {
        size: size_descr.size(),
        // descr.py type_id is the dense GC tid for alloc_nursery_typed,
        // not the cache_key structural identity
        type_id: size_descr.type_id() as u64,
        vtable: vtable as u64,
        owner: String::new(),
        all_fielddescrs: majit_jitcode::jitcode::bh_field_specs_from_size_descr(size_descr),
        // Round-trip the GC-header flag off the descr.
        is_gc_managed: size_descr.is_gc_managed(),
    }
}

fn bh_array_descr_from_descr(descr: &majit_ir::DescrRef) -> majit_jitcode::jitcode::BhDescr {
    let ad = descr
        .as_array_descr()
        .expect("resume.py: allocate_array requires ArrayDescr");
    majit_jitcode::jitcode::BhDescr::from_array_descr(ad)
}

fn decode_tagged_for_kind(
    tagged: i16,
    kind: Type,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    fail_values: &[i64],
    num_failargs: i32,
    storage: Option<&std::sync::Arc<majit_metainterp::resume::ResumeStorage>>,
    backend: &dyn majit_backend::Backend,
    callinfocollection: Option<&std::sync::Arc<majit_ir::CallInfoCollection>>,
    cache: &mut BridgeVirtualCache<'_>,
) -> i64 {
    decode_tagged_concrete(
        tagged,
        kind,
        rd_virtuals,
        fail_values,
        num_failargs,
        storage,
        backend,
        callinfocollection,
        cache,
    )
}

fn live_concrete_addr(cache: &BridgeVirtualCache<'_>, vidx: usize, fallback: i64) -> i64 {
    cache
        .get_concrete_ptr(vidx)
        .map(|gcref| gcref.0 as i64)
        .unwrap_or(fallback)
}

fn setfield_concrete_from_tagged(
    backend: &dyn majit_backend::Backend,
    struct_ptr: i64,
    vidx: usize,
    fd: &majit_ir::FieldDescrInfo,
    fieldnum: i16,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    fail_values: &[i64],
    num_failargs: i32,
    storage: Option<&std::sync::Arc<majit_metainterp::resume::ResumeStorage>>,
    callinfocollection: Option<&std::sync::Arc<majit_ir::CallInfoCollection>>,
    cache: &mut BridgeVirtualCache<'_>,
) {
    let descr = bh_field_descr_from_info(fd);
    match fd.field_type {
        Type::Ref => {
            let value = decode_tagged_for_kind(
                fieldnum,
                Type::Ref,
                rd_virtuals,
                fail_values,
                num_failargs,
                storage,
                backend,
                callinfocollection,
                cache,
            );
            let struct_ptr = live_concrete_addr(cache, vidx, struct_ptr);
            backend.bh_setfield_gc_r(struct_ptr, majit_ir::GcRef(value as usize), &descr);
        }
        Type::Float => {
            let value = decode_tagged_for_kind(
                fieldnum,
                Type::Float,
                rd_virtuals,
                fail_values,
                num_failargs,
                storage,
                backend,
                callinfocollection,
                cache,
            );
            let struct_ptr = live_concrete_addr(cache, vidx, struct_ptr);
            backend.bh_setfield_gc_f(struct_ptr, f64::from_bits(value as u64), &descr);
        }
        _ => {
            let value = decode_tagged_for_kind(
                fieldnum,
                Type::Int,
                rd_virtuals,
                fail_values,
                num_failargs,
                storage,
                backend,
                callinfocollection,
                cache,
            );
            let struct_ptr = live_concrete_addr(cache, vidx, struct_ptr);
            backend.bh_setfield_gc_i(struct_ptr, value, &descr);
        }
    }
}

fn setarrayitem_concrete_from_tagged(
    backend: &dyn majit_backend::Backend,
    array_ptr: i64,
    vidx: usize,
    index: usize,
    arraydescr: &dyn majit_ir::descr::ArrayDescr,
    bh_descr: &majit_jitcode::jitcode::BhDescr,
    fieldnum: i16,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    fail_values: &[i64],
    num_failargs: i32,
    storage: Option<&std::sync::Arc<majit_metainterp::resume::ResumeStorage>>,
    callinfocollection: Option<&std::sync::Arc<majit_ir::CallInfoCollection>>,
    cache: &mut BridgeVirtualCache<'_>,
) {
    if arraydescr.is_array_of_pointers() {
        let value = decode_tagged_for_kind(
            fieldnum,
            Type::Ref,
            rd_virtuals,
            fail_values,
            num_failargs,
            storage,
            backend,
            callinfocollection,
            cache,
        );
        let array_ptr = live_concrete_addr(cache, vidx, array_ptr);
        backend.bh_setarrayitem_gc_r(
            array_ptr,
            index as i64,
            majit_ir::GcRef(value as usize),
            bh_descr,
        );
    } else if arraydescr.is_array_of_floats() {
        let value = decode_tagged_for_kind(
            fieldnum,
            Type::Float,
            rd_virtuals,
            fail_values,
            num_failargs,
            storage,
            backend,
            callinfocollection,
            cache,
        );
        let array_ptr = live_concrete_addr(cache, vidx, array_ptr);
        backend.bh_setarrayitem_gc_f(
            array_ptr,
            index as i64,
            f64::from_bits(value as u64),
            bh_descr,
        );
    } else {
        let value = decode_tagged_for_kind(
            fieldnum,
            Type::Int,
            rd_virtuals,
            fail_values,
            num_failargs,
            storage,
            backend,
            callinfocollection,
            cache,
        );
        let array_ptr = live_concrete_addr(cache, vidx, array_ptr);
        backend.bh_setarrayitem_gc_i(array_ptr, index as i64, value, bh_descr);
    }
}

/// resume.py setrawbuffer_item BlackholeResumeDataReader.setinteriorfield_{int,
/// ref,float}: dispatch on the `InteriorFieldDescr.fielddescr` type to
/// `cpu.bh_setinteriorfield_gc_{i,r,f}`.
fn setinteriorfield_concrete_from_tagged(
    backend: &dyn majit_backend::Backend,
    array_ptr: i64,
    vidx: usize,
    index: usize,
    interior_descr: &dyn majit_ir::descr::Descr,
    fieldnum: i16,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    fail_values: &[i64],
    num_failargs: i32,
    storage: Option<&std::sync::Arc<majit_metainterp::resume::ResumeStorage>>,
    callinfocollection: Option<&std::sync::Arc<majit_ir::CallInfoCollection>>,
    cache: &mut BridgeVirtualCache<'_>,
) {
    let ifd = interior_descr
        .as_interior_field_descr()
        .expect("VArrayStructInfo: fielddescr is not an InteriorFieldDescr");
    let bh = majit_jitcode::jitcode::BhDescr::from_interior_field_descr(ifd);
    match ifd.field_descr().field_type() {
        Type::Ref => {
            let value = decode_tagged_for_kind(
                fieldnum,
                Type::Ref,
                rd_virtuals,
                fail_values,
                num_failargs,
                storage,
                backend,
                callinfocollection,
                cache,
            );
            let array_ptr = live_concrete_addr(cache, vidx, array_ptr);
            backend.bh_setinteriorfield_gc_r(
                array_ptr,
                index as i64,
                majit_ir::GcRef(value as usize),
                &bh,
            );
        }
        Type::Float => {
            let value = decode_tagged_for_kind(
                fieldnum,
                Type::Float,
                rd_virtuals,
                fail_values,
                num_failargs,
                storage,
                backend,
                callinfocollection,
                cache,
            );
            let array_ptr = live_concrete_addr(cache, vidx, array_ptr);
            backend.bh_setinteriorfield_gc_f(
                array_ptr,
                index as i64,
                f64::from_bits(value as u64),
                &bh,
            );
        }
        _ => {
            let value = decode_tagged_for_kind(
                fieldnum,
                Type::Int,
                rd_virtuals,
                fail_values,
                num_failargs,
                storage,
                backend,
                callinfocollection,
                cache,
            );
            let array_ptr = live_concrete_addr(cache, vidx, array_ptr);
            backend.bh_setinteriorfield_gc_i(array_ptr, index as i64, value, &bh);
        }
    }
}

fn bh_call_r_for_oopspec(
    backend: &dyn majit_backend::Backend,
    callinfocollection: Option<&std::sync::Arc<majit_ir::CallInfoCollection>>,
    oopspec: majit_ir::OopSpecIndex,
    args_i: Option<&[i64]>,
    args_r: Option<&[i64]>,
) -> majit_ir::GcRef {
    let cic = callinfocollection.expect(
        "TraceCtx.callinfocollection missing — concrete VStr/VUni materialization \
         requires resume.py DirectReader funcptr_for_oopspec parity",
    );
    let (calldescr, func) = cic.callinfo_for_oopspec(oopspec);
    let calldescr = calldescr
        .expect("callinfo_for_oopspec missing entry for concrete VStr/VUni materialization");
    let cd = calldescr
        .as_call_descr()
        .expect("VStr/VUni oopspec calldescr must be CallDescr");
    let bh_calldescr = majit_jitcode::jitcode::BhCallDescr::from_call_descr(cd);
    backend.bh_call_r(func as i64, args_i, args_r, None, &bh_calldescr)
}

/// The block a resumed `PyFrame` virtual is materialized into while a bridge
/// is traced.
///
/// The bridge walk executes that frame as the interpreter would: the
/// reconstructed callee runs on it (`setup_reconstructed_callee_frame`) and
/// holds its address across allocations, so it needs the placement
/// `FrameBox::new` gives an executing frame — a non-moving young block — rather
/// than the nursery block `bh_new_with_vtable` answers with.  `None` for any
/// other type, or before the collector hook is installed, leaves the ordinary
/// allocation in place.
fn materialize_resumed_pyframe_block(
    size_descr: &dyn majit_ir::descr::SizeDescr,
    vtable: usize,
) -> Option<i64> {
    if size_descr.type_id() != crate::descr::PYFRAME_GC_TYPE_ID {
        return None;
    }
    let size = size_descr.size();
    let raw = pyre_object::gc_hook::try_gc_alloc_young_nonmoving_raw(
        crate::descr::PYFRAME_GC_TYPE_ID,
        size,
    );
    if raw.is_null() {
        return None;
    }
    unsafe {
        std::ptr::write_bytes(raw, 0, size);
        // `llmodel.py bh_new_with_vtable`: the vtable word at
        // `vtable_offset`, which pyre's backends set to `OB_TYPE_OFFSET`.
        *(raw.add(pyre_object::pyobject::OB_TYPE_OFFSET) as *mut usize) = vtable;
    }
    Some(raw as i64)
}

/// resume.py getvirtual_ptr concrete parity.
/// Lazily allocate a concrete object for virtual index `vidx`, caching in
/// `BridgeVirtualCache.concrete_ptr_cache` so shared/recursive virtuals
/// allocate exactly once.
///
/// resume.py polymorphic dispatch by RdVirtualInfo variant.
/// BlackholeResumeDataReader (resume.py) routes each kind
/// through the CPU backend: bh_new_with_vtable, bh_new, bh_new_array,
/// bh_newstr, bh_newunicode, bh_call_i/r.
fn materialize_concrete_virtual_ptr(
    vidx: usize,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    fail_values: &[i64],
    num_failargs: i32,
    storage: Option<&std::sync::Arc<majit_metainterp::resume::ResumeStorage>>,
    backend: &dyn majit_backend::Backend,
    callinfocollection: Option<&std::sync::Arc<majit_ir::CallInfoCollection>>,
    cache: &mut BridgeVirtualCache<'_>,
) -> majit_ir::GcRef {
    if let Some(cached) = cache.get_concrete_ptr(vidx) {
        return cached;
    }
    // resume.py:953 assert self.rd_virtuals is not None
    let virtuals = rd_virtuals.expect("materialize_concrete_virtual_ptr: rd_virtuals is None");
    // resume.py self.rd_virtuals[index].allocate(self, index) — direct indexing
    let entry = &virtuals[vidx];
    match entry.as_ref() {
        // resume.py VirtualInfo.allocate
        majit_ir::RdVirtualInfo::VirtualInfo {
            descr,
            fielddescrs,
            fieldnums,
            ..
        } => {
            let descr = descr.as_ref().expect("VirtualInfo: descr is None");
            let size_descr = descr.as_size_descr().expect("VirtualInfo: not a SizeDescr");
            let vtable = size_descr.vtable();
            // resume.py allocate_with_vtable(descr) → cpu.bh_new_with_vtable(descr)
            let bh_descr = bh_size_descr_from_size_descr(size_descr, vtable);
            let ptr = match materialize_resumed_pyframe_block(size_descr, vtable) {
                Some(ptr) => ptr,
                None => backend.bh_new_with_vtable(&bh_descr),
            };
            if ptr == 0 {
                return majit_ir::GcRef::NULL;
            }
            // Pyre adaptation: bh_new_with_vtable writes vtable at
            // vtable_offset but PyObject.w_class needs separate init
            // (`PyObject.w_class` in pyobject.rs). Matches
            // materialize_virtual_object.
            //
            // `w_class_obj()` — not `get_instantiate(vtable)` — is the source:
            // a vtable word is only a `PyType` pointer for a pyre object
            // descr.  `JitVirtualRef` carries the `jit_virtual_ref_vtable`
            // type-id constant there (`virtualref.py:21-23`) and its offset-8
            // slot is `virtual_token`, not `w_class`, so it returns None and
            // this seeding is skipped.
            let gcref = majit_ir::GcRef(ptr as usize);
            // VirtualInfo.allocate stores the struct in virtuals_cache before setfields so a circular reference resolves to it
            cache.set_concrete_ptr(vidx, gcref);
            let ptr = live_concrete_addr(cache, vidx, ptr);
            if let Some(w_class) = size_descr.w_class_obj() {
                unsafe {
                    let pyobj = ptr as *mut pyre_object::PyObject;
                    (*pyobj).w_class = w_class as pyre_object::pyobject::PyObjectRef;
                }
            }
            // resume.py setfields — range(len(fielddescrs)), index
            // fieldnums[i]. The len-equality assert (resume.py) is in
            // debug_prints, not this allocate path: a short fieldnums raises
            // IndexError here, a longer one is ignored.
            for i in 0..fielddescrs.len() {
                let fd = &fielddescrs[i];
                let fnum = fieldnums[i];
                if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                    continue;
                }
                setfield_concrete_from_tagged(
                    backend,
                    ptr,
                    vidx,
                    fd,
                    fnum,
                    rd_virtuals,
                    fail_values,
                    num_failargs,
                    storage,
                    callinfocollection,
                    cache,
                );
            }
            cache.get_concrete_ptr(vidx).unwrap_or(gcref)
        }
        // resume.py VStructInfo.allocate — no vtable
        majit_ir::RdVirtualInfo::VStructInfo {
            typedescr,
            fielddescrs,
            fieldnums,
            ..
        } => {
            let descr = typedescr.as_ref().expect("VStructInfo: typedescr is None");
            let size_descr = descr.as_size_descr().expect("VStructInfo: not a SizeDescr");
            // resume.py allocate_struct(typedescr) → cpu.bh_new(typedescr)
            let bh_descr = bh_size_descr_from_size_descr(size_descr, 0);
            let ptr = backend.bh_new(&bh_descr);
            if ptr == 0 {
                return majit_ir::GcRef::NULL;
            }
            let gcref = majit_ir::GcRef(ptr as usize);
            cache.set_concrete_ptr(vidx, gcref);
            // resume.py setfields — range(len(fielddescrs)), index
            // fieldnums[i]. The len-equality assert (resume.py) is in
            // debug_prints, not this allocate path: a short fieldnums raises
            // IndexError here, a longer one is ignored.
            for i in 0..fielddescrs.len() {
                let fd = &fielddescrs[i];
                let fnum = fieldnums[i];
                if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                    continue;
                }
                setfield_concrete_from_tagged(
                    backend,
                    ptr,
                    vidx,
                    fd,
                    fnum,
                    rd_virtuals,
                    fail_values,
                    num_failargs,
                    storage,
                    callinfocollection,
                    cache,
                );
            }
            cache.get_concrete_ptr(vidx).unwrap_or(gcref)
        }
        // resume.py VArrayInfo.allocate
        majit_ir::RdVirtualInfo::VArrayInfoClear {
            arraydescr,
            fieldnums,
            ..
        }
        | majit_ir::RdVirtualInfo::VArrayInfoNotClear {
            arraydescr,
            fieldnums,
            ..
        } => {
            let length = fieldnums.len();
            let is_clear = matches!(
                entry.as_ref(),
                majit_ir::RdVirtualInfo::VArrayInfoClear { .. }
            );
            // resume.py allocate_array(length, arraydescr, clear)
            // → cpu.bh_new_array_clear / bh_new_array
            let descr = arraydescr.as_ref().expect("VArrayInfo: arraydescr is None");
            let ad = descr
                .as_array_descr()
                .expect("VArrayInfo: not an ArrayDescr");
            let bh_descr = bh_array_descr_from_descr(descr);
            let ptr = if is_clear {
                backend.bh_new_array_clear(length as i64, &bh_descr)
            } else {
                backend.bh_new_array(length as i64, &bh_descr)
            };
            if ptr == 0 {
                return majit_ir::GcRef::NULL;
            }
            let gcref = majit_ir::GcRef(ptr as usize);
            cache.set_concrete_ptr(vidx, gcref);
            // resume.py setarrayitem per element
            for (i, &fnum) in fieldnums.iter().enumerate() {
                if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                    continue;
                }
                setarrayitem_concrete_from_tagged(
                    backend,
                    ptr,
                    vidx,
                    i,
                    ad,
                    &bh_descr,
                    fnum,
                    rd_virtuals,
                    fail_values,
                    num_failargs,
                    storage,
                    callinfocollection,
                    cache,
                );
            }
            cache.get_concrete_ptr(vidx).unwrap_or(gcref)
        }
        // resume.py VArrayStructInfo.allocate
        majit_ir::RdVirtualInfo::VArrayStructInfo {
            arraydescr,
            fielddescrs,
            size,
            fieldnums,
            ..
        } => {
            // resume.py array = decoder.allocate_array(self.size,
            // self.arraydescr, clear=True) — uses the live `self.arraydescr`
            // directly.
            let array_descr = arraydescr
                .as_ref()
                .expect("VArrayStructInfo: arraydescr is None");
            let bh_array = bh_array_descr_from_descr(array_descr);
            // resume.py:749 clear=True → bh_new_array_clear
            let ptr = backend.bh_new_array_clear(*size as i64, &bh_array);
            if ptr == 0 {
                return majit_ir::GcRef::NULL;
            }
            let gcref = majit_ir::GcRef(ptr as usize);
            cache.set_concrete_ptr(vidx, gcref);
            // resume.py:752-759:
            //   p = 0
            //   for i in range(self.size):
            //       for j in range(len(self.fielddescrs)):
            //           num = self.fieldnums[p]
            //           if not tagged_eq(num, UNINITIALIZED):
            //               decoder.setinteriorfield(i, array, num,
            //                   self.fielddescrs[j])
            //           p += 1
            let num_fields = fielddescrs.len();
            // resume.py:752-759 reads exactly size × len(fielddescrs) entries
            // via self.fieldnums[p] with no length-equality check: a short
            // fieldnums is an out-of-bounds error here (IndexError parity), a
            // longer one leaves its tail unread.
            let mut p = 0;
            for i in 0..*size {
                for j in 0..num_fields {
                    let fnum = fieldnums[p];
                    p += 1;
                    if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                        continue;
                    }
                    setinteriorfield_concrete_from_tagged(
                        backend,
                        ptr,
                        vidx,
                        i,
                        fielddescrs[j].as_ref(),
                        fnum,
                        rd_virtuals,
                        fail_values,
                        num_failargs,
                        storage,
                        callinfocollection,
                        cache,
                    );
                }
            }
            cache.get_concrete_ptr(vidx).unwrap_or(gcref)
        }
        // resume.py VStrPlainInfo.allocate
        majit_ir::RdVirtualInfo::VStrPlainInfo { fieldnums } => {
            let length = fieldnums.len();
            // resume.py allocate_string(length) → cpu.bh_newstr(length)
            let ptr = backend.bh_newstr(length as i64);
            if ptr == 0 {
                return majit_ir::GcRef::NULL;
            }
            let gcref = majit_ir::GcRef(ptr as usize);
            cache.set_concrete_ptr(vidx, gcref);
            // resume.py string_setitem → cpu.bh_strsetitem
            for (i, &fnum) in fieldnums.iter().enumerate() {
                if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                    continue;
                }
                let value = decode_tagged_concrete(
                    fnum,
                    Type::Int,
                    rd_virtuals,
                    fail_values,
                    num_failargs,
                    storage,
                    backend,
                    callinfocollection,
                    cache,
                );
                let ptr = live_concrete_addr(cache, vidx, ptr);
                backend.bh_strsetitem(ptr, i as i64, value);
            }
            cache.get_concrete_ptr(vidx).unwrap_or(gcref)
        }
        // resume.py VUniPlainInfo.allocate
        majit_ir::RdVirtualInfo::VUniPlainInfo { fieldnums } => {
            let length = fieldnums.len();
            // resume.py allocate_unicode(length) → cpu.bh_newunicode(length)
            let ptr = backend.bh_newunicode(length as i64);
            if ptr == 0 {
                return majit_ir::GcRef::NULL;
            }
            let gcref = majit_ir::GcRef(ptr as usize);
            cache.set_concrete_ptr(vidx, gcref);
            // resume.py unicode_setitem → cpu.bh_unicodesetitem
            for (i, &fnum) in fieldnums.iter().enumerate() {
                if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                    continue;
                }
                let value = decode_tagged_concrete(
                    fnum,
                    Type::Int,
                    rd_virtuals,
                    fail_values,
                    num_failargs,
                    storage,
                    backend,
                    callinfocollection,
                    cache,
                );
                let ptr = live_concrete_addr(cache, vidx, ptr);
                backend.bh_unicodesetitem(ptr, i as i64, value);
            }
            cache.get_concrete_ptr(vidx).unwrap_or(gcref)
        }
        // resume.py VStrConcatInfo / VUniConcatInfo
        majit_ir::RdVirtualInfo::VStrConcatInfo { fieldnums }
        | majit_ir::RdVirtualInfo::VUniConcatInfo { fieldnums } => {
            assert!(fieldnums.len() == 2, "concat: expected 2 fieldnums");
            let str1 = decode_tagged_concrete(
                fieldnums[0],
                Type::Ref,
                rd_virtuals,
                fail_values,
                num_failargs,
                storage,
                backend,
                callinfocollection,
                cache,
            );
            let str2 = decode_tagged_concrete(
                fieldnums[1],
                Type::Ref,
                rd_virtuals,
                fail_values,
                num_failargs,
                storage,
                backend,
                callinfocollection,
                cache,
            );
            let oopspec = if matches!(
                entry.as_ref(),
                majit_ir::RdVirtualInfo::VStrConcatInfo { .. }
            ) {
                majit_ir::descr::OopSpecIndex::StrConcat
            } else {
                majit_ir::descr::OopSpecIndex::UniConcat
            };
            // resume.py concat_strings DirectReader concat_* resolves funcptr
            // via callinfocollection.funcptr_for_oopspec, not a placeholder.
            let gcref = bh_call_r_for_oopspec(
                backend,
                callinfocollection,
                oopspec,
                None,
                Some(&[str1, str2]),
            );
            cache.set_concrete_ptr(vidx, gcref);
            gcref
        }
        // resume.py VStrSliceInfo / VUniSliceInfo
        majit_ir::RdVirtualInfo::VStrSliceInfo { fieldnums }
        | majit_ir::RdVirtualInfo::VUniSliceInfo { fieldnums } => {
            assert!(fieldnums.len() == 3, "slice: expected 3 fieldnums");
            let strbox = decode_tagged_concrete(
                fieldnums[0],
                Type::Ref,
                rd_virtuals,
                fail_values,
                num_failargs,
                storage,
                backend,
                callinfocollection,
                cache,
            );
            let start = decode_tagged_concrete(
                fieldnums[1],
                Type::Int,
                rd_virtuals,
                fail_values,
                num_failargs,
                storage,
                backend,
                callinfocollection,
                cache,
            );
            let length = decode_tagged_concrete(
                fieldnums[2],
                Type::Int,
                rd_virtuals,
                fail_values,
                num_failargs,
                storage,
                backend,
                callinfocollection,
                cache,
            );
            let stop = start + length;
            let oopspec = if matches!(
                entry.as_ref(),
                majit_ir::RdVirtualInfo::VStrSliceInfo { .. }
            ) {
                majit_ir::descr::OopSpecIndex::StrSlice
            } else {
                majit_ir::descr::OopSpecIndex::UniSlice
            };
            // resume.py slice_string DirectReader slice_* passes stop =
            // start + length to the funcptr resolved from callinfocollection.
            let gcref = bh_call_r_for_oopspec(
                backend,
                callinfocollection,
                oopspec,
                Some(&[start, stop]),
                Some(&[strbox]),
            );
            cache.set_concrete_ptr(vidx, gcref);
            gcref
        }
        // Raw virtuals are INT-kind — should not appear in getvirtual_ptr
        majit_ir::RdVirtualInfo::VRawBufferInfo { .. }
        | majit_ir::RdVirtualInfo::VRawSliceInfo { .. } => {
            panic!(
                "materialize_concrete_virtual_ptr: raw virtual at vidx={vidx} is INT-kind, not PTR"
            )
        }
        // resume.py getvirtual_ptr calls rd_virtuals[index].allocate()
        // directly; a None/Empty slot is never referenced by a TAGVIRTUAL tag
        // in a well-formed resume stream, so reaching it is a corrupt-stream
        // bug, not a NULL fallback (mirrors materialize_concrete_virtual_int
        // and eval.rs materialize_virtual, which already panic on Empty).
        majit_ir::RdVirtualInfo::Empty => panic!(
            "materialize_concrete_virtual_ptr: null rd_virtuals[{vidx}] (resume.py: null rd_virtuals[index])"
        ),
    }
}

/// resume.py getvirtual_int concrete parity.  Only raw-buffer
/// virtuals are INT virtuals (`VAbstractRawInfo.kind = INT`).
fn materialize_concrete_virtual_int(
    vidx: usize,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    fail_values: &[i64],
    num_failargs: i32,
    storage: Option<&std::sync::Arc<majit_metainterp::resume::ResumeStorage>>,
    backend: &dyn majit_backend::Backend,
    callinfocollection: Option<&std::sync::Arc<majit_ir::CallInfoCollection>>,
    cache: &mut BridgeVirtualCache<'_>,
) -> i64 {
    if let Some(cached) = cache.get_concrete_int(vidx) {
        return cached;
    }
    // resume.py:959 assert self.virtuals_cache is not None (cache is not None by construction)
    // resume.py:962 self.rd_virtuals[index] — direct indexing, not silent fallback
    let virtuals = rd_virtuals.expect("materialize_concrete_virtual_int: rd_virtuals is None");
    let entry = &virtuals[vidx];
    // resume.py assert v.is_about_raw and isinstance(v, VAbstractRawInfo)
    match entry.as_ref() {
        // resume.py VRawBufferInfo.allocate_int
        majit_ir::RdVirtualInfo::VRawBufferInfo {
            func,
            size,
            offsets,
            descrs,
            fieldnums,
        } => {
            // resume.py BlackholeResumeDataReader.allocate_raw_buffer:
            //   cic = self.callinfocollection
            //   calldescr, _ = cic.callinfo_for_oopspec(OS_RAW_MALLOC_VARSIZE_CHAR)
            //   return self.cpu.bh_call_i(func, [size], None, None, calldescr)
            // No fallback upstream — a missing CIC entry is a metainterp-setup
            // bug, surfaced fail-loud rather than papered over with a synthetic
            // calldescr.
            let cic = callinfocollection.expect("allocate_raw_buffer: callinfocollection is None");
            // resume.py:1455: calldescr, _ = cic.callinfo_for_oopspec(
            //   OS_RAW_MALLOC_VARSIZE_CHAR). callinfo_for_oopspec returns
            // (None, 0) on a missing entry (effectinfo.py) — no
            // lookup-time check; the calldescr is used directly.
            let (calldescr, _) =
                cic.callinfo_for_oopspec(majit_ir::descr::OopSpecIndex::RawMallocVarsizeChar);
            // resume.py:1456: self.cpu.bh_call_i(func, [size], None, None,
            //   calldescr). A missing entry surfaces here, as the calldescr is
            // consumed by the backend call, not as a separate lookup assertion.
            let descr_ref =
                calldescr.expect("OS_RAW_MALLOC_VARSIZE_CHAR calldescr (callinfocollection)");
            let cd = descr_ref
                .as_call_descr()
                .expect("OS_RAW_MALLOC_VARSIZE_CHAR: not a CallDescr");
            let bh = majit_jitcode::jitcode::BhCallDescr::from_arg_classes(
                cd.arg_classes(),
                cd.result_class(),
                cd.get_extra_info().clone(),
            );
            let buffer = backend.bh_call_i(*func, Some(&[*size as i64]), None, None, &bh);
            // resume.py: decoder.virtuals_cache.set_int(index, buffer)
            cache.set_concrete_int(vidx, buffer);
            // resume.py:705-708 iterate by len(self.offsets) unconditionally
            // (no buffer == 0 guard), indexing self.descrs[i] and
            // self.fieldnums[i] by the same i — a short descrs/fieldnums raises
            // IndexError here (encoder bug), a longer one is ignored. No
            // len-equality assert (VRawBufferInfo has none).
            for i in 0..offsets.len() {
                let off = offsets[i];
                let fnum = fieldnums[i];
                if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                    continue;
                }
                // resume.py setrawbuffer_item assert not descr.is_array_of_pointers()
                assert!(
                    descrs[i].item_type != 0,
                    "setrawbuffer_item: pointer arraydescr in raw buffer"
                );
                // resume.py setrawbuffer_item: the arraydescr drives
                // the store width via the backend's write_{int,float}_at_mem,
                // so the value is stored through bh_raw_store_{f,i} with the
                // real descr — no caller-side field-size truncation.
                let store_descr =
                    majit_jitcode::jitcode::BhDescr::from_array_descr_info(&descrs[i]);
                if descrs[i].item_type == 2 {
                    let value = decode_tagged_concrete(
                        fnum,
                        Type::Float,
                        rd_virtuals,
                        fail_values,
                        num_failargs,
                        storage,
                        backend,
                        callinfocollection,
                        cache,
                    );
                    // resume.py:1548-1549 bh_raw_store_f(ptr, offset, floatval, descr)
                    backend.bh_raw_store_f(
                        buffer,
                        off as i64,
                        f64::from_bits(value as u64),
                        &store_descr,
                    );
                } else {
                    let value = decode_tagged_concrete(
                        fnum,
                        Type::Int,
                        rd_virtuals,
                        fail_values,
                        num_failargs,
                        storage,
                        backend,
                        callinfocollection,
                        cache,
                    );
                    // resume.py:1550-1552 bh_raw_store_i(ptr, offset, intval, descr)
                    backend.bh_raw_store_i(buffer, off as i64, value, &store_descr);
                }
            }
            buffer
        }
        // resume.py VRawSliceInfo.allocate_int
        majit_ir::RdVirtualInfo::VRawSliceInfo { offset, fieldnums } => {
            assert!(
                fieldnums.len() == 1,
                "VRawSliceInfo must have exactly 1 fieldnum"
            );
            let base = decode_tagged_concrete(
                fieldnums[0],
                Type::Int,
                rd_virtuals,
                fail_values,
                num_failargs,
                storage,
                backend,
                callinfocollection,
                cache,
            );
            let buffer = base + *offset;
            cache.set_concrete_int(vidx, buffer);
            buffer
        }
        // resume.py getvirtual_int asserts is_about_raw
        other => panic!(
            "materialize_concrete_virtual_int: non-raw virtual kind at vidx={vidx}: {other:?}"
        ),
    }
}

/// ResumeGuardExcDescr analog for bridge walks.
///
/// `setup_bridge_sym` runs after guard-failure resume has rebuilt the live
/// frame. Per `pyjitpl.py _prepare_exception_resumption`, only a
/// ResumeGuardExcDescr/ResumeGuardCopiedExcDescr source guard carries the
/// pending exception into bridge tracing. Non-exception guards do not seed
/// standing exception state, and operand-stack values are never scanned to
/// infer one.
fn seed_bridge_standing_exception_from_current(
    sym: &mut PyreSym,
    ctx: &mut majit_metainterp::TraceCtx,
) {
    if !ctx.bridge_source_is_exception_guard() {
        return;
    }
    if !sym.last_exc_box.is_none() {
        return;
    }

    let mut exc = sym.current_exc_value.get();
    let exc_box = sym.current_exc_box;
    if exc.is_null() || !unsafe { pyre_object::is_exception(exc) } {
        let current = pyre_interpreter::eval::get_current_exception();
        if !current.is_null() && unsafe { pyre_object::is_exception(current) } {
            exc = current;
        } else {
            sym.current_exc_value.set(std::ptr::null_mut());
            sym.current_exc_box = OpRef::NONE;
            sym.last_exc_value.set(std::ptr::null_mut());
            sym.last_exc_box = OpRef::NONE;
            sym.class_of_last_exc_is_const = false;
            return;
        }
    }

    let exc_box = if exc_box.is_none() {
        ctx.const_ref(exc as i64)
    } else {
        exc_box
    };
    sym.current_exc_value.set(exc);
    sym.current_exc_box = exc_box;
    sym.last_exc_value.set(exc);
    sym.last_exc_box = exc_box;
    sym.class_of_last_exc_is_const = true;
}

impl JitState for PyreJitState {
    /// The unified runtime registry seats runtime Python bodies and
    /// materializes frozen bodies on demand (`ensure_build_time_jitcode_at`);
    /// the driver's own `jitcodes` vector holds a skeleton at those indices.
    fn resolve_resume_jitcode(
        index: usize,
    ) -> Option<std::sync::Arc<majit_metainterp::jitcode::JitCode>> {
        ensure_build_time_jitcode_at(index).map(|payload| std::sync::Arc::clone(&payload.jitcode))
    }

    type Meta = PyreMeta;
    type Sym = PyreSym;
    type Env = PyreEnv;

    /// `sync_virtualizable_after_guard_failure` has already run by the time
    /// bridge setup is reached: guard-failure recovery selects it through
    /// `ResumeVableMode::GuardFailureSync`, and it writes each slot through
    /// `value_to_static_vable_bits` / `value_to_vable_array_item_bits`, which
    /// box an unwrapped `Value::Int` / `Value::Float` back into an object
    /// before it lands in a slot the frame reads as one.
    const SYNCHRONIZES_VIRTUALIZABLE_AFTER_GUARD_FAILURE: bool = true;

    fn build_meta(&self, _header_pc: usize, _env: &Self::Env) -> Self::Meta {
        let num_locals = self.local_count();
        let vsd = self.valuestackdepth();
        let slot_types = concrete_slot_types(self.frame, num_locals, vsd);
        // `capacity` is the frame's real heap array capacity, not the
        // valuestackdepth: the meta describes the allocated slot count that
        // vable array accesses are bounds-checked against.
        let capacity = self.array_capacity();
        PyreMeta {
            num_locals,
            ns_len: self.namespace_len().unwrap_or(0),
            // Provisional seed only.  build_meta runs at trace START, before the
            // walk records any LOAD_GLOBAL/LOAD_NAME, so the true value does not
            // exist yet.  finish_trace_namespace_dependency overwrites this from
            // TraceCtx.reads_module_global on every trace_bytecode return path,
            // and the entry-bridge fold ORs the live flag mid-walk (`false` is
            // the OR identity).  Both invariants must hold for this seed to stay
            // safe to drop.
            namespace_dependent: false,
            valuestackdepth: vsd,
            array_capacity: capacity,
            // `virtualizable_gen.rs` wires `extra_reds = { ec: Ref }` per
            // interp_jit.py `reds = ['frame', 'ec']`, so the per-Sym
            // helpers (vable_collect_jump_args, NUM_EXTRA_REDS) already
            // thread the ec slot. The runtime flag here remains 0 because
            // pypy/module/pypyjit/interp_jit.py PyPyJitDriver reds=
            // ['frame', 'ec']. With vable heap-writeback landed
            // (vable heap-writeback active behind the reduced-LABEL
            // gate), the descriptor activation chain is unblocked.
            // Cluster 2 flip — pairs with `driver_descriptor() = Some(...)`
            // below.
            trace_extra_reds: 1,
            has_virtualizable: self.has_virtualizable_info(),
            slot_types,
        }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        self.extract_live_values(_meta)
            .into_iter()
            .map(|value| match value {
                Value::Int(v) => v,
                Value::Float(v) => v.to_bits() as i64,
                Value::Ref(r) => r.as_usize() as i64,
                Value::Void => 0,
            })
            .collect()
    }

    fn extract_live_values(&self, meta: &Self::Meta) -> Vec<Value> {
        if meta.trace_extra_reds == 1 {
            self.pypyjit_live_values_with_ec(meta)
        } else {
            self.expanded_virtualizable_live_values_with_extra_reds(meta, &[])
        }
    }

    fn extract_live_values_into(
        &self,
        meta: &Self::Meta,
        out: &mut Vec<Value>,
        raw: &mut Vec<i64>,
        types: &mut Vec<Type>,
    ) {
        let _ = (raw, types);
        if meta.trace_extra_reds == 1 {
            // `warmstate.py WarmEnterState.make_entry_point` specializes the
            // red-argument walk and hands the already-separated `frame` and
            // `ec` values to `EnterJitAssembler`; it does not allocate a list
            // and copy it into another entry buffer. The owning method above
            // remains the cold-path API, while the warm entry refills the
            // driver's retained buffer directly.
            out.push(Value::Ref(majit_ir::GcRef(self.frame)));
            out.push(Value::Ref(majit_ir::GcRef(
                self.execution_context_as_usize(),
            )));
        } else {
            out.extend(self.expanded_virtualizable_live_values_with_extra_reds(meta, &[]));
        }
    }

    fn close_loop_live_values(
        ctx: &majit_metainterp::TraceCtx,
        sym: &Self::Sym,
        _meta: &Self::Meta,
        live_arg_boxes: &[OpRef],
    ) -> Option<Vec<Value>> {
        let mut values = Vec::with_capacity(live_arg_boxes.len());
        let num_scalars = crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
        let num_vable_scalars = crate::virtualizable_gen::NUM_VABLE_SCALARS;

        let frame_addr = if sym.live_vable_frame_addr != 0 {
            sym.live_vable_frame_addr
        } else {
            sym.concrete_vable_ptr as usize
        };
        let frame_value = if frame_addr != 0 {
            Value::Ref(majit_ir::GcRef(frame_addr))
        } else {
            ctx.box_value(*live_arg_boxes.first()?)?
        };
        values.push(frame_value);

        if crate::virtualizable_gen::NUM_EXTRA_REDS == 1 {
            let ec_addr = sym.concrete_execution_context as usize;
            let ec_value = if ec_addr != 0 {
                Value::Ref(majit_ir::GcRef(ec_addr))
            } else {
                ctx.box_value(*live_arg_boxes.get(1)?)?
            };
            values.push(ec_value);
        }

        let vable_start = 1 + crate::virtualizable_gen::NUM_EXTRA_REDS;
        for i in 0..num_vable_scalars {
            let slot = vable_start + i;
            let value = ctx
                .virtualizable_entry_at(i)
                .map(|(_, value)| value)
                .or_else(|| {
                    live_arg_boxes
                        .get(slot)
                        .and_then(|opref| ctx.box_value(*opref))
                })?;
            values.push(value);
        }

        // Array slots are `virtualizable_boxes` (`box.getref_base()`). The
        // pyre `concrete_locals` / `concrete_stack` mirror is filled at seed
        // and is not updated by `setarrayitem_vable`, so it lags the traced
        // body (e.g. `i` still the header value after `i = i + 1`).
        let array_slots = live_arg_boxes.len().saturating_sub(num_scalars);
        for slot in 0..array_slots {
            let vable_idx = num_vable_scalars + slot;
            let value = ctx
                .virtualizable_entry_at(vable_idx)
                .map(|(_, value)| value)
                .or_else(|| {
                    live_arg_boxes
                        .get(num_scalars + slot)
                        .and_then(|opref| ctx.box_value(*opref))
                })
                .unwrap_or_else(|| {
                    let concrete = if slot < sym.nlocals {
                        sym.concrete_locals
                            .get(slot)
                            .copied()
                            .unwrap_or(ConcreteValue::Ref(PY_NULL))
                    } else {
                        let stack_idx = slot - sym.nlocals;
                        let live_stack = sym.valuestackdepth.saturating_sub(sym.nlocals);
                        if stack_idx < live_stack {
                            sym.concrete_stack
                                .get(stack_idx)
                                .copied()
                                .unwrap_or(ConcreteValue::Ref(PY_NULL))
                        } else {
                            ConcreteValue::Ref(PY_NULL)
                        }
                    };
                    Value::Ref(majit_ir::GcRef(concrete.to_pyobj() as usize))
                });
            values.push(value);
        }

        Some(values)
    }

    // virtualizable.py read_boxes() + warmstate.py wrap() parity:
    // Array items (locals_cells_stack_w) are GC pointers → RefFrontendOp.
    // No pre-unboxing at function entry. Unboxing happens during tracing
    // via guard_class + getfield_gc_i when arithmetic/compare handlers
    // encounter Ref-typed operands.

    fn live_value_types(&self, meta: &Self::Meta) -> Vec<Type> {
        if meta.trace_extra_reds == 1 {
            Self::pypyjit_live_value_types_with_ec(meta)
        } else {
            Self::expanded_virtualizable_live_value_types_with_extra_reds(meta, &[])
        }
    }

    fn live_value_types_into(&self, meta: &Self::Meta, out: &mut Vec<Type>) {
        if meta.trace_extra_reds == 1 {
            out.push(Type::Ref);
            out.push(Type::Ref);
        } else {
            out.extend(Self::expanded_virtualizable_live_value_types_with_extra_reds(meta, &[]));
        }
    }

    fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
        if _meta.trace_extra_reds == 1 {
            Self::pypyjit_create_sym(_meta, _header_pc)
        } else {
            let mut sym = PyreSym::new_uninit(OpRef::input_arg_typed(0, Type::Ref));
            // `extra_reds = { ec: Ref }` in virtualizable_gen.rs places ec
            // at OpRef::from_raw(1). `init_vable_indices` shifts the vable static
            // field + array_base OpRefs but does not own the extra_red
            // sym storage; initialize it explicitly. NUM_EXTRA_REDS == 1 is
            // a crate-level const-assertion (see `lib.rs`).
            sym.execution_context = OpRef::input_arg_typed(1, Type::Ref);
            sym.become_active_vable_owner();
            sym.nlocals = _meta.num_locals;
            sym.valuestackdepth = _meta.valuestackdepth;
            // virtualizable.py:44 + interp_jit.py:25-30: all locals_cells_stack_w
            // items are W_Root → Type::Ref. Unboxing happens inside trace opcode
            // handlers (guard_class + getfield_gc_pure_i/_f), not at slot setup.
            sym.symbolic_local_types =
                vec![Type::Ref; _meta.num_locals.min(_meta.slot_types.len())];
            sym.symbolic_stack_types =
                vec![Type::Ref; _meta.slot_types.len().saturating_sub(_meta.num_locals)];
            let stack_only = _meta.vable_stack_only_depth();
            sym.concrete_stack = vec![ConcreteValue::Null; stack_only];
            sym
        }
    }

    fn initialize_sym(&self, sym: &mut Self::Sym, _meta: &Self::Meta) {
        // `JitDriver::merge_point` sequencing parity: `start_bridge_tracing`
        // calls `create_sym` → `initialize_sym` → `setup_bridge_sym` BEFORE
        // `trace_bytecode` (which is where `init_symbolic` would otherwise
        // populate `concrete_vable_ptr`). `setup_bridge_sym` reads
        // `sym.concrete_vable_ptr` to size the vable shadow via
        // `concrete_frame_array_len`; without this seed the pointer is
        // null, the helper falls back to `nlocals`, and stack pushes
        // beyond the first array slot panic at `set_virtualizable_entry_at`
        // on recursive benchmarks (fib_recursive). Mirror PyreJitState's
        // live frame pointer so the bridge seed sees the real
        // `locals_cells_stack_w` length.
        sym.concrete_vable_ptr = self.frame as *mut u8;
        sym.concrete_execution_context =
            self.execution_context as *const pyre_interpreter::PyExecutionContext;
    }

    fn driver_descriptor(&self, _meta: &Self::Meta) -> Option<JitDriverStaticData> {
        // pypy/module/pypyjit/interp_jit.py PyPyJitDriver:
        //   reds = ['frame', 'ec']
        //   greens = ['next_instr', 'is_being_profiled', 'pycode']
        //   virtualizables = ['frame']
        //
        // History (now landed): the atomic flip needed a vable
        // heap-writeback pass that pyre's tracer originally did not
        // emit. Concretely,
        // `patch_new_loop_to_load_virtualizable_fields` (compile.py)
        // collapses the patched LABEL to `[reds]` and prepends a
        // GETFIELD_GC + GETARRAYITEM_GC preamble. The body then reads
        // every vable static field and locals_cells_stack_w slot from
        // the frame heap object on each iteration. pyre's tracer
        // updates symbolic state (`sym.vable_*`, `sym.registers_r[i]`)
        // but never emits the matching `SetfieldGc(frame, value, descr)`
        // or `SetarrayitemGc(array_ref, idx, value)` — only the
        // freshly-allocated W_IntObject.intval setfields appear in
        // trace dumps. With descriptor=None, this is fine because the
        // closing JUMP carries the post-loop state via expanded inputargs;
        // the patched-LABEL contract makes the heap the source of truth
        // and breaks that invariant.
        //
        // Prerequisites for the flip (all atomic with descriptor=Some):
        //   (a) `trace_extra_reds=1` (build_meta:3970) — emit
        //       `live_values=[frame, ec]` matching descriptor reds. The
        //       legacy `extract_live_values` shape relies on
        //       `descriptor=None`; flipping (a) alone breaks dynasm
        //       nested_loop / fannkuch / nbody (live_values consumers
        //       expect the expanded shape) and panics on
        //       `live_values[index>1]` access without (b).
        //   (b) `initialize_virtualizable` short-live_values gate
        //       (`pyjitpl.rs`): allow the heap-read branch when
        //       `vable_ptr` is non-null. Required to consume (a)'s
        //       reds-only live_values.
        //   (c1) `pending_frontend_boxes.clone()` (`pyjitpl.rs`):
        //        the second `compile_bridge` call needs the same stash
        //        as the first; current `take()` empties on the first
        //        call and the second hits `frontend_boxes.len()=0 vs
        //        liveboxes.len()=N`.
        //   (c2) Bridge JUMP arity vs patched LABEL: source emits the
        //        live-window shape (`num_scalars + valuestackdepth`) but
        //        the patched LABEL grows to full vable capacity. Const
        //        padding SSA-forwards `PY_NULL` into outer-loop locals
        //        on cranelift; heap-read padding via
        //        `trace_array_getitem_value` against
        //        `locals_cells_stack_array_ref` is the correct shape.
        //   (d) **Vable heap-writeback infrastructure** — *blocking*. With
        //       (a)+(b)+(c1) applied, cranelift nested_loop dispatches
        //       the bridge 5622+ times with identical inputs `[fr, ec,
        //       500, 2507475000]` because the patched parent reloads
        //       stale state from heap each iteration; the bridge body
        //       has no `SetfieldGc`/`SetarrayitemGc` to advance the
        //       heap. RPython's OptVirtualize emits these via its
        //       `force_at_end_of_preamble` pass; pyre's
        //       `OptVirtualize::force_virtualizable` (virtualize.rs)
        //       has the SETFIELD_RAW emission machinery but is not
        //       wired to fire at JUMP. An eager
        //       `gen_writeback_vable_to_heap` helper at
        //       `close_loop_args_at` / the intermediate merge point was
        //       tried, then removed once the hot-loop LABEL carried the
        //       live vable values as inputargs — the patched parent no
        //       longer reloads stale heap state each iteration, and a
        //       guard failure rebuilds the heap frame from resume-data
        //       (`consume_vref_and_vable_boxes` → `write_boxes`),
        //       matching RPython's no-eager-writeback model.
        //   (e) dynasm recursive CA frame contract — *blocking*
        //       for dynasm SIGSEGV at fib(24).
        //
        // Status: (a)+(b)+(c1)+(c2) have landed and descriptor is
        // active; (d)'s eager writeback was removed in favor of the
        // no-eager-writeback model (live values carried as inputargs,
        // heap frame rebuilt from resume-data). (e) dynasm recursive CA
        // frame contract remains a separate open item driving
        // fib_recursive on dynasm.
        Some(Self::pypyjit_driver_descriptor())
    }

    fn is_compatible(&self, meta: &Self::Meta) -> bool {
        // warmstate.py:503-511: RPython enters assembler unconditionally
        // when procedure_token exists. No next_instr check —
        // the compiled code's preamble handles entry from any PC.
        // Shape checks ensure the frame layout matches.  Namespace length is
        // load-bearing only for traces that read module globals: pure-compute
        // and builtin-only loops do not depend on later top-level binds, while
        // module-global reads still need this conservative gate because
        // same-key value rebinds are not value-guarded yet.
        self.local_count() == meta.num_locals
            && (!meta.namespace_dependent
                || self.namespace_len().is_some_and(|len| len == meta.ns_len))
    }

    fn mint_bridge_array_descr(
        base_size: usize,
        item_size: usize,
        len_offset: Option<usize>,
        item_type: Type,
        signed: bool,
    ) -> majit_ir::DescrRef {
        crate::descr::make_array_descr(base_size, item_size, len_offset, item_type, signed)
    }

    fn prepare_bridge_resume(
        sym: &mut Self::Sym,
        ctx: &mut majit_metainterp::TraceCtx,
        resume_data: &majit_metainterp::ResumeDataResult,
        rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
        fail_values: &[i64],
        fail_types: &[Type],
        reader: &mut BridgeVirtualCache<'_>,
    ) {
        apply_bridge_prepare(
            sym,
            ctx,
            resume_data,
            rd_virtuals,
            fail_values,
            fail_types,
            reader,
        );
    }

    fn consume_vref_and_vable_boxes(
        _sym: &mut Self::Sym,
        ctx: &mut majit_metainterp::TraceCtx,
        resume_data: &majit_metainterp::ResumeDataResult,
        rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
        fail_values: &[i64],
        fail_types: &[Type],
        virtuals_cache: &mut BridgeVirtualCache<'_>,
    ) -> majit_metainterp::VrefVableBoxes {
        let (driver, _) = crate::driver::driver_pair();
        let backend = driver.meta_interp().backend();
        let mut boxes = majit_metainterp::VrefVableBoxes::default();
        // `consume_virtualizable_boxes` — virtualizable.py `load_list_of_boxes`
        // layout `[vable_ptr, static_fields..., array_items...]`. resume.py
        // `assert box.type == kind`: the payload is NOT uniformly Ref — the
        // static fields carry their declared kinds
        // (`PyFrame.virtualizable_fields`: last_instr/valuestackdepth are
        // Int). `virt_live_value_types` yields the full live layout WITH the
        // extra reds ([frame, <NUM_EXTRA_REDS>, <NUM_VABLE_SCALARS>,
        // array...]); the vvals stream omits the extra reds, so strip them to
        // recover the per-slot kind for each payload position.
        let vvals = &resume_data.virtualizable_boxes;
        let array_item_count = vvals
            .len()
            .saturating_sub(1 + crate::virtualizable_gen::NUM_VABLE_SCALARS);
        let full_types = crate::virtualizable_gen::virt_live_value_types(array_item_count);
        let nreds = crate::virtualizable_gen::NUM_EXTRA_REDS;
        let mut vvals_types: Vec<Type> = Vec::with_capacity(vvals.len());
        vvals_types.push(full_types.first().copied().unwrap_or(Type::Ref));
        vvals_types.extend_from_slice(&full_types[(1 + nreds).min(full_types.len())..]);
        for (idx, v) in vvals.iter().enumerate() {
            let expected_kind = vvals_types.get(idx).copied().unwrap_or(Type::Ref);
            boxes.virtualizable_boxes.push(bridge_decode_box(
                ctx,
                v,
                expected_kind,
                rd_virtuals,
                resume_data,
                fail_values,
                fail_types,
                backend,
                virtuals_cache,
            ));
        }
        // `consume_virtualref_boxes` decodes exactly `size * 2` entries; a
        // malformed odd-length stream is a bug.
        let vref_values = &resume_data.virtualref_values;
        assert!(
            vref_values.len() % 2 == 0,
            "virtualref_values must contain an even number of entries (got {})",
            vref_values.len(),
        );
        for v in vref_values {
            boxes.virtualref_boxes.push(bridge_decode_box(
                ctx,
                v,
                Type::Ref,
                rd_virtuals,
                resume_data,
                fail_values,
                fail_types,
                backend,
                virtuals_cache,
            ));
        }
        boxes
    }

    fn setup_bridge_sym(
        sym: &mut Self::Sym,
        ctx: &mut majit_metainterp::TraceCtx,
        resume_data: &majit_metainterp::ResumeDataResult,
        rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
        fail_values: &[i64],
        fail_types: &[Type],
        virtuals_cache: &mut BridgeVirtualCache<'_>,
        boxes: &majit_metainterp::VrefVableBoxes,
    ) {
        // rebuild_state_after_failure still consumes virtualizable and
        // virtualref boxes when the framestack stays empty. Return only
        // when those streams are empty too.
        let no_frame = resume_data.frames.is_empty();
        let no_vable = resume_data.virtualizable_boxes.is_empty();
        let no_vref = resume_data.virtualref_values.is_empty();
        let no_pending = resume_data
            .storage
            .as_ref()
            .is_none_or(|storage| storage.rd_pendingfields.is_empty());
        if no_frame && no_vable && no_vref && no_pending {
            return;
        }
        // `_prepare_pendingfields` applies SETFIELD_GC / SETARRAYITEM_GC
        // through `prepare_bridge_pending_fields` immediately after the
        // virtual cache is created (`execute_setfield_gc` /
        // `execute_setarrayitem_gc`). A nonempty pending stream is not a
        // reason to mark the replay incomplete.

        // virtualizable.py load_list_of_boxes parity: decode each
        // RebuiltValue in the resume stream into a typed Value. The type
        // is the fixed Box kind the encoder recorded at numbering time
        // (fail_arg_types[idx] for Box variants), matching RPython's
        // immutable Box.type invariant. No heap read.

        // resume.py VirtualCache parity: single cache indexed by
        // virtual number, holding both symbolic OpRef (for trace ops) and
        // concrete GcRef (for shadow values / continue_tracing). RPython's
        // VirtualCache stores both in one object; pyre unifies them here.
        //
        // `ResumeDataBoxReader.allocate_with_vtable` is
        // `execute_new_with_vtable`: the `NEW` this walk records carries the
        // object it allocates, so the box and the object this walk executes
        // on are one pair. `executing: Some` is that applying reader
        // (`jit_state.rs` `setup_bridge_sym`); `None` records only because
        // a direct reader already applied this guard's writes.
        let (driver, _) = crate::driver::driver_pair();
        let backend = driver.meta_interp().backend();
        // resume.py one `virtuals_cache` from `_prepare_virtuals`; a later
        // TAGVIRTUAL is `getvirtual_ptr` on that cache, not a second
        // `allocate()` / SETFIELD of `wrappeditems`.

        // `_prepare` ran once, in `prepare_bridge_resume`; `consume_boxes`
        // never touches the pending fields again. The reader keeps the
        // ref-array writes `_prepare_pendingfields` applied.
        let pending_ref_array_writes = virtuals_cache.pending_ref_array_writes().to_vec();

        // resume.py decode_box parity — unified via bridge_decode_box.
        // Each call returns (OpRef, Value), eliminating the separate
        // resolve()/decode_concrete() paths so symbolic and concrete
        // materialization are always invoked together.

        let nlocals = sym.nlocals;
        // virtualizable.py:44 + interp_jit.py:25-30: locals_cells_stack_w[*]
        // items are declared Ref (W_Root array). Bridge resume slots stay
        // Ref at the virtualizable contract; any Int/Float unboxing must
        // happen inside trace opcode handlers, not at the inputarg level.
        let bridge_local_types = vec![Type::Ref; nlocals];

        // rebuild_state_after_failure (pyjitpl.py) keeps three
        // streams apart:
        //   virtualizable_boxes ← consume_virtualizable_boxes (vable mirror)
        //   registers_r/_i/_f    ← consume_boxes(f.get_current_position_info(),
        //                                        f.registers_i/_r/_f)
        //   virtualref_boxes     ← consume_virtualref_boxes
        // The majit decoder already splits rd_numb into the same three
        // streams (`resume_data.virtualizable_boxes`,
        // `resume_data.frames[*].values`, `resume_data.virtualref_values`).
        // This function consumes them in the same order and purpose.
        //
        // Part 1 — virtualizable payload (consume_virtualizable_boxes):
        // decode scalar header into sym.vable_* and capture the array-item
        // prefix for init_virtualizable_boxes below. Matches
        // virtualizable.py `read_boxes` layout
        //   [vable_ptr, static_fields..., array_items...].
        if !(no_frame && no_vable) {
            let vvals = &resume_data.virtualizable_boxes;
            // Resume virtualizable payload mirrors RPython
            // opencoder.py _list_of_boxes_virtualizable + virtualizable.py load_list_of_boxes:
            //   [vable, vable_static_fields..., array_items...]
            // Non-vable extra reds (e.g. `ec`) are root inputargs, not part of
            // this payload, so the boundary is NUM_VABLE_SCALARS, not
            // NUM_SCALAR_INPUTARGS.
            let first_vable_scalar_idx = 1usize;
            let vable_array_start =
                first_vable_scalar_idx + crate::virtualizable_gen::NUM_VABLE_SCALARS;
            let mut oprefs: Vec<OpRef> = Vec::with_capacity(vvals.len());
            let mut concrete_values: Vec<majit_ir::Value> = Vec::with_capacity(vvals.len());
            // The boxes `consume_vref_and_vable_boxes` decoded, one per
            // payload position.
            assert_eq!(
                boxes.virtualizable_boxes.len(),
                vvals.len(),
                "consume_vref_and_vable_boxes decoded the virtualizable stream"
            );
            for (idx, &(op, val)) in boxes.virtualizable_boxes.iter().enumerate() {
                if idx >= vable_array_start {
                    store_live_frame_array_slot(
                        sym.concrete_vable_ptr as usize,
                        idx - vable_array_start,
                        val,
                    );
                }
                oprefs.push(op);
                concrete_values.push(val);
            }
            sym.restore_inputarg_oprefs(&oprefs, first_vable_scalar_idx);
            let vable_ref_value = concrete_values
                .first()
                .copied()
                .unwrap_or(majit_ir::Value::Void);
            let vable_scalar_values: Vec<majit_ir::Value> = concrete_values
                .iter()
                .skip(first_vable_scalar_idx)
                .take(crate::virtualizable_gen::NUM_VABLE_SCALARS)
                .copied()
                .collect();
            let vable_array_items: Vec<OpRef> =
                oprefs.iter().skip(vable_array_start).copied().collect();
            let vable_array_values: Vec<majit_ir::Value> = concrete_values
                .iter()
                .skip(vable_array_start)
                .copied()
                .collect();
            let bridge_valuestackdepth = concrete_values
                // virtualizable_boxes has no ec red: [vable, last_instr,
                // pycode, valuestackdepth, debugdata, ...].
                .get(first_vable_scalar_idx + 2)
                .map(value_to_usize)
                .unwrap_or(sym.valuestackdepth)
                .max(nlocals);
            // `rebuild_state_after_failure` writes the virtualizable,
            // including its stack depth, before it walks frames. An empty
            // framestack still leaves that depth in place.
            sym.valuestackdepth = bridge_valuestackdepth;

            // Part 2 — frame registers (consume_boxes): walk the frame section
            // in liveness enumeration order ([int..., ref..., float...]), keep
            // each bank's register indices separate via
            // `frame_liveness_reg_indices_by_bank_at`, and write each decoded
            // value into the corresponding MIFrame register bank. This mirrors
            // resume.py `consume_boxes`
            // (`_callback_i/_r/_f(register_index)` writing to
            // `f.registers_i/_r/_f[index]` at the exact slot liveness
            // declared, not at an enumerate-order position). RPython indexes
            // a single `registers_X` vector by abstract register color —
            // there is no `idx < nlocals` decode.
            if !no_frame {
                let frame0 = &resume_data.frames[0];
                let reg_indices = crate::state::frame_liveness_reg_indices_by_bank_from_pc(
                    frame0.jitcode_index,
                    frame0.pc,
                );
                // For a kept-stack branch guard, the vable's
                // `valuestackdepth` may reflect the merge-target depth (consumed
                // stack) rather than the guard's deeper live depth. The guard PC's
                // pcdep `stack_depth_at_pc` (the `depth_pred_by_jit_pc` twin, keyed by
                // the carried `jitcode_pc`) IS the guard-time depth. Use the larger
                // of the two so the color→slot inversion covers the kept temps.
                // This is deferred until after `maps` is read (below) via a
                // re-adjustment of the semantic mirror length.
                let stack_only = bridge_valuestackdepth.saturating_sub(nlocals);
                let resume_maps = crate::state::bridge_semantic_maps_from_jitcode_pc(
                    frame0.jitcode_index,
                    frame0.pc,
                );
                let bridge_reg_len = nlocals + stack_only;
                let mut bridge_registers_r = vec![OpRef::NONE; bridge_reg_len];
                // RPython parity: after A.1 the guard-recovery path calls
                // `synchronize_virtualizable()` / `write_boxes()`
                // (pyjitpl.py) before `start_bridge_tracing`, so the
                // physical vable image the tracer is about to read is already
                // resume-data-complete. The
                // bridge register file is therefore expected to be fully
                // populated by the liveness-driven zip below; any remaining
                // OpRef::NONE signals a liveness-coverage gap (the tracer keeps a
                // local live past the `-live-` marker) and must be surfaced by
                // the assert rather than papered over with a vable-mirror read.
                assert!(
                    reg_indices.total_len() == frame0.values.len(),
                    "setup_bridge_sym: reg_indices len={} != frame.values len={} at pc={}",
                    reg_indices.total_len(),
                    frame0.values.len(),
                    frame0.pc,
                );
                // GotoIfNotValueNotConcrete (bridge sub-class): the resume
                // data carries the concrete runtime value for every live frame
                // register, but the consume_boxes loops below keep only the OpRef
                // and DROP the concrete — so a bridge resume leaves loop-carried
                // locals symbolic and a data-dependent branch derived from one
                // (`(i%7) and ...` → TO_BOOL → goto_if_not) can't fold its
                // direction, aborting the walk.  Stamp each decoded
                // concrete onto its OpRef so the symbolic walk folds the branch
                // per the actual failing-iteration path (the iteration the
                // compiled trace re-runs), emitting a real
                // GuardTrue/GuardFalse — orthodox meta-tracing ("trace the
                // concrete path, guard it"; the IR keeps the symbolic InputArg,
                // the concrete is a trace-time shadow only, so the optimizer does
                // NOT const-fold the loop-variant value).
                // `seed_deferred_to_overlay` is Ref-specific.  The Ref overlay below
                // rebuilds a SLOT-indexed mirror from the color-indexed resume decode
                // (via `pcdep_entries`).  At a kept-stack branch guard the body-internal
                // marker's Ref colors do not 1:1-correspond to semantic slots — a color
                // that lives a temp at the marker may name a different slot at the guard
                // PC — so seeding the Ref concrete at the color-indexed consume stage
                // stamps the wrong value; defer it to the post-overlay stage where the
                // mirror is slot-indexed and authoritative.  The Int and Float banks
                // (`sym.registers_i` / `sym.registers_f`) are pure scalar register reds
                // with no operand-stack slot mirror, so they reconstruct concrete at the
                // consume stage directly — matching `resume.py
                // rebuild_from_resumedata` → `consume_boxes(f.get_current_position_info(),
                // registers_i, registers_r, registers_f)`, which fills all three banks
                // uniformly at the guard's resume position.  A resolved `-live-`
                // offset alone does not identify that branch shape: after-residual
                // guards carry one too.  The branch is the shape whose guard-time
                // pcdep depth is deeper than the resumed virtualizable stack; when the
                // depths agree, the frame stream is authoritative and must be stamped
                // directly even if the frame-array image is still pre-call.
                let seed_deferred_to_overlay = resume_maps.stack_depth_at_pc > stack_only;
                let mut bridge_stamp_orphans = seed_deferred_to_overlay.then(Vec::new);
                let mut value_cursor = 0usize;
                for &reg_idx in &reg_indices.int {
                    let value = &frame0.values[value_cursor];
                    let (resolved, concrete_val) = bridge_decode_box(
                        ctx,
                        value,
                        Type::Int,
                        rd_virtuals,
                        resume_data,
                        fail_values,
                        fail_types,
                        backend,
                        virtuals_cache,
                    );
                    // The Int bank is a scalar register red: the decoded value is paired
                    // with `reg_indices.int` by `value_cursor` and read from this guard's
                    // `fail_values`, so it is guard-accurate.  Stamp it as the trace-time
                    // concrete unconditionally (not gated on `seed_deferred_to_overlay`,
                    // which is Ref-only) so a loop-carried Int red feeding an
                    // `(i % k) == 0` kept-stack branch guard folds its `goto_if_not`
                    // instead of declining the bridge with `GotoIfNotValueNotConcrete`.
                    // Mirrors `consume_boxes(..., f.registers_i, ...)` filling the Int
                    // bank at the guard's resume position.
                    if !matches!(concrete_val, majit_ir::Value::Void) {
                        ctx.try_set_opref_concrete(resolved, concrete_val);
                    }
                    let reg_idx = reg_idx as usize;
                    if reg_idx >= sym.registers_i.len() {
                        sym.registers_i.resize(reg_idx + 1, OpRef::NONE);
                    }
                    sym.registers_i[reg_idx] = resolved;
                    value_cursor += 1;
                }
                for &reg_idx in &reg_indices.ref_ {
                    let value = &frame0.values[value_cursor];
                    let (resolved, concrete_val) = bridge_decode_box(
                        ctx,
                        value,
                        Type::Ref,
                        rd_virtuals,
                        resume_data,
                        fail_values,
                        fail_types,
                        backend,
                        virtuals_cache,
                    );
                    if !seed_deferred_to_overlay && !matches!(concrete_val, majit_ir::Value::Void) {
                        ctx.try_set_opref_concrete(resolved, concrete_val);
                    }
                    if let Some(orphan_stamps) = bridge_stamp_orphans.as_mut() {
                        orphan_stamps.push((resolved, concrete_val));
                    }
                    let reg_idx = reg_idx as usize;
                    if reg_idx >= bridge_registers_r.len() {
                        bridge_registers_r.resize(reg_idx + 1, OpRef::NONE);
                    }
                    bridge_registers_r[reg_idx] = resolved;
                    value_cursor += 1;
                }
                for &reg_idx in &reg_indices.float {
                    let value = &frame0.values[value_cursor];
                    let (resolved, concrete_val) = bridge_decode_box(
                        ctx,
                        value,
                        Type::Float,
                        rd_virtuals,
                        resume_data,
                        fail_values,
                        fail_types,
                        backend,
                        virtuals_cache,
                    );
                    // Float bank: a scalar register red, guard-accurate like the Int
                    // bank above; stamp its concrete unconditionally (the Ref-only
                    // `seed_deferred_to_overlay` deferral does not apply), mirroring
                    // `consume_boxes(..., f.registers_f)`.
                    if !matches!(concrete_val, majit_ir::Value::Void) {
                        ctx.try_set_opref_concrete(resolved, concrete_val);
                    }
                    let reg_idx = reg_idx as usize;
                    if reg_idx >= sym.registers_f.len() {
                        sym.registers_f.resize(reg_idx + 1, OpRef::NONE);
                    }
                    sym.registers_f[reg_idx] = resolved;
                    value_cursor += 1;
                }
                // `interp_jit.py PyPyJitDriver.reds = ['frame', 'ec']`: codewriter gives both
                // portal reds dedicated Ref colors and force-keeps them live at every
                // guard. `consume_boxes` above therefore rebuilt the bridge's own EC
                // inputarg at `portal_ec_reg`; retain that OpRef before converting the
                // color-indexed register bank into the semantic locals/stack mirror.
                //
                // Two shapes leave nothing to retain: a skeleton jitcode carries
                // `u16::MAX` for both portal red colors, and a resumed register can be
                // empty. Either way the first consumer reaches
                // `MIFrame::ensure_execution_context`, which reads the thread's own
                // context instead.
                let (_, portal_ec_reg) = portal_red_regs_at(frame0.jitcode_index);
                let bridge_execution_context = (portal_ec_reg != u16::MAX)
                    .then(|| {
                        bridge_registers_r
                            .get(portal_ec_reg as usize)
                            .copied()
                            .unwrap_or(OpRef::NONE)
                    })
                    .unwrap_or(OpRef::NONE);
                // Reconstruct the slot-indexed semantic register file
                // (`[locals.., stack_tail..]`) from the color-indexed resume decode.
                // The decode just filled `bridge_registers_r` by abstract-register
                // color (`reg_indices.ref_`); the bridge trace, however, reads
                // `sym.registers_r` and the kept-stack/local oprefs by SEMANTIC slot
                // (LOAD_FAST `registers_r[var_num]`, stack `nlocals + depth`). When
                // the per-CodeObject regalloc colors a local/stack slot at a color
                // other than its slot index, a slot-indexed read of the color bank
                // returns a foreign value (a dead temp, a portal red, or a constant)
                // — a corruption the codewriter used to mask by pinning the
                // `[0,nlocals)` prefix to identity colors (now retired). Invert each
                // live color to its slot via `semantic_ref_slot_for_reg_color` so the
                // mirror is correct under freely-colored locals.
                let maps = resume_maps;
                // For a kept-stack branch guard, the vable's runtime
                // `valuestackdepth` reflects the merge-target depth (post
                // consumption) rather than the guard's deeper live depth. The
                // guard-PC pcdep `stack_depth_at_pc` IS the guard-time depth
                // (with kept temps live). Widen `semantic_prefix_len` to the
                // pcdep depth so the color→slot inversion covers kept temps.
                let stack_only = stack_only.max(maps.stack_depth_at_pc);
                let semantic_prefix_len = nlocals + stack_only;
                // Extend bridge_registers_r to cover the wider depth.
                if bridge_registers_r.len() < semantic_prefix_len {
                    bridge_registers_r.resize(semantic_prefix_len, OpRef::NONE);
                }
                if majit_metainterp::majit_log_enabled()
                    && crate::state::frame_pc_is_resolved_offset_at(frame0.jitcode_index, frame0.pc)
                {
                    let old_so = bridge_valuestackdepth.saturating_sub(nlocals);
                    eprintln!(
                        "[jit][kept-stack-bridge] jitcode_pc={} pc={} pcdep={:?} \
                     depth_at_guard={} vsd={} stack_only={}→{}",
                        frame0.pc,
                        frame0.pc,
                        maps.pcdep_entries,
                        maps.stack_depth_at_pc,
                        bridge_valuestackdepth,
                        old_so,
                        stack_only,
                    );
                }
                // virtualizable.py read_boxes + pyjitpl.py synchronize_virtualizable:
                // the frame's `locals_cells_stack_w` array (the vable image) is the
                // authoritative post-guard source for the frame's locals. At an
                // arbitrary interior resume pc a local slot's jitcode color may hold
                // a dead temp that decodes to a NULL constant (`ConstPtr(GcRef(0))`);
                // such a dead-temp NULL must not shadow the vable's live local, so for
                // local slots prefer the vable array item over a NONE/null-const value.
                // Stack slots keep the NONE-only fallback (a real stack value must not
                // be overwritten).
                //
                // GC-rooted concrete source for the per-local stamp below: read the
                // live virtualizable frame's slots directly (post the ref/int/float
                // decode loops above, which allocate and may trigger a minor GC) so
                // the stamp never points at a stale off-heap `vable_array_values`
                // copy whose `Ref`s a collection has since moved.  See
                // `live_frame_array_values`.
                let live_local_values = live_frame_array_values(
                    sym.concrete_vable_ptr as usize,
                    usize::MAX,
                    &vable_array_values,
                );
                let mut overlay_local = |slot: &mut OpRef, s: usize| {
                    let slot_is_null_const = matches!(*slot, OpRef::ConstPtr(v) if v.0 == 0);
                    if slot.is_none() || slot_is_null_const {
                        if let Some(v) = vable_array_items.get(s).copied() {
                            if !v.is_none() {
                                *slot = v;
                                // A local resolved from the vable image: stamp its
                                // concrete from the GC-rooted live frame slot
                                // (`live_local_values`, not the off-heap decoded
                                // array) so the seeded bridge walk can fold a branch
                                // derived from it without risking a moved-pointer
                                // stamp (bridge sub-class; see seed note above).
                                // Skip a NULL (`GcRef(0)`) source, matching the
                                // deferred-overlay seed below: a loop-carried local
                                // held in a register at an interior guard reads NULL
                                // from `locals_cells_stack_w` because it was never
                                // written back.  Stamping that hole poisons the real
                                // vable box with concrete NULL, folding a later
                                // residual's Ref arg to NULL →
                                // `MayForceNullRefArgUnsupported`.  Leaving the box
                                // unstamped keeps it symbolic so the residual reads
                                // the runtime value.
                                if let Some(&cv) = live_local_values.get(s) {
                                    if !matches!(
                                        cv,
                                        majit_ir::Value::Void
                                            | majit_ir::Value::Ref(majit_ir::GcRef(0))
                                    ) {
                                        ctx.try_set_opref_concrete(v, cv);
                                    }
                                }
                            } else if slot.is_none() {
                                *slot = OpRef::NONE;
                            }
                        }
                    }
                };
                let semantic_mirror: Vec<OpRef> = if !maps.has_color_map {
                    // No per-CodeObject regalloc: colors are slot-identity, so the
                    // color bank IS the slot mirror over the semantic prefix. Keep the
                    // in-place identity overlay, taking the vable image for a slot the
                    // color bank left NONE or decoded to a NULL constant.
                    // `!has_color_map` (empty `pcdep_color_slots`) is the field-free
                    // successor to the flat `local/stack_color_map.is_empty()` guard, so
                    // a zero-local frame that still owns a freely-colored operand stack
                    // (non-empty `pcdep_color_slots`) falls to the else branch (per-slot
                    // inversion) instead of reading the color-indexed bank as if it were
                    // slot-indexed.
                    for (idx, slot) in bridge_registers_r
                        .iter_mut()
                        .enumerate()
                        .take(semantic_prefix_len)
                    {
                        let slot_is_null_const = matches!(*slot, OpRef::ConstPtr(v) if v.0 == 0);
                        let want_vable = slot.is_none() || slot_is_null_const;
                        if want_vable {
                            if let Some(v) = vable_array_items.get(idx).copied() {
                                if !v.is_none() {
                                    *slot = v;
                                } else if slot.is_none() {
                                    *slot = OpRef::NONE;
                                }
                            }
                        }
                    }
                    bridge_registers_r
                        .iter()
                        .take(semantic_prefix_len)
                        .copied()
                        .collect()
                } else {
                    // Per-CodeObject: fill each live local/stack slot from its color.
                    let mut mirror = vec![OpRef::NONE; semantic_prefix_len];
                    // #348 Part (2): per-PC slot fill. `pcdep_entries` maps each live
                    // slot to its TRUE per-program-point color, so drive the fill by
                    // SLOT (not color): write `mirror[slot] = bridge_registers_r[color]`
                    // for every entry. A color shared by multiple slots — an aliased
                    // `DUP_TOP` / `ROT_THREE` operand-stack pair, or a local aliased
                    // onto the stack — writes its single value into EVERY slot it
                    // covers. A prior color→slot inversion kept only one slot per
                    // color (stack-first tie-break), leaving the sibling aliased slot
                    // `OpRef::NONE`; the vable image is also NULL for pure trace temps
                    // (`kept_stack_branch_depths` `0 < a < b < 9` keeps two copies of
                    // the same compare operand across the guard), so that slot folded
                    // to concrete `GcRef(0)` and the residual declined. Out-of-prefix
                    // slots are dropped by the `s >= semantic_prefix_len` guard.  #73:
                    // pcdep is the SOLE color→slot source here; the flat
                    // `local_color_map` / `stack_color_map` fallback is drained.
                    for &(bank, color, slot) in &maps.pcdep_entries {
                        // Only Ref-bank colors map to bridge_registers_r slots.
                        // Int/Float bank entries are structurally recorded but
                        // currently unreachable (operand stack is always Ref).
                        if bank != 1 {
                            continue;
                        }
                        let s = slot as usize;
                        if s >= semantic_prefix_len {
                            continue;
                        }
                        let col = color as usize;
                        if col < bridge_registers_r.len() {
                            mirror[s] = bridge_registers_r[col];
                        }
                    }
                    // pcdep-totality guard (#73): the drained flat-else previously
                    // inverted live operand-stack slots from `stack_color_map` when
                    // `pcdep_entries` was empty.  The corpus proof
                    // (`validate_pcdep_color_map`, injective + total) shows an empty
                    // `pcdep_entries` here carries no live operand stack — locals
                    // still refill from the vable image via `overlay_local` below.
                    // Fail loud under `PYRE_PCDEP_VALIDATE` if a live stack slot ever
                    // reaches here uncovered (a totality regression).
                    if maps.pcdep_entries.is_empty()
                        && std::env::var_os("PYRE_PCDEP_VALIDATE").is_some()
                    {
                        // `stack_depth_at_pc` never exceeds `max_stackdepth` (the retired
                        // `stack_color_map.len()` clamp), so the runtime `stack_only`
                        // bound is the only one that matters.
                        let live_stack = maps.stack_depth_at_pc.min(stack_only);
                        if live_stack > 0 {
                            eprintln!(
                                "PCDEP-TOTALITY-VIOLATION: empty pcdep_entries with \
                             {live_stack} live stack slot(s) at bridge resume \
                             (jitcode_index={}, pc={})",
                                frame0.jitcode_index, frame0.pc
                            );
                        }
                    }
                    for s in 0..nlocals {
                        overlay_local(&mut mirror[s], s);
                    }
                    // Pcdep-live kept-stack colors absent from the body marker's
                    // liveness leave their stack slots NONE after the color→slot
                    // inversion above, and a color the inversion DID cover can still
                    // decode to a dead temp's NULL constant.  The vable image
                    // (`locals_cells_stack_w`) is authoritative post-guard, so take it
                    // for either — the same null-const rule `overlay_local` applies to
                    // locals.  A NULL here does not mean the slot is empty: an empty
                    // slot reads NULL from the vable image too, so the fill is a no-op
                    // there, while a live slot whose color decoded NULL would otherwise
                    // carry that NULL into the closing JUMP's label argument and hand
                    // it back to the interpreter on the next guard failure.
                    for s in nlocals..semantic_prefix_len.min(mirror.len()) {
                        let slot_is_null_const =
                            matches!(mirror[s], OpRef::ConstPtr(v) if v.0 == 0);
                        if mirror[s].is_none() || slot_is_null_const {
                            if let Some(v) = vable_array_items.get(s).copied() {
                                if !v.is_none() {
                                    mirror[s] = v;
                                    if let Some(&cv) = live_local_values.get(s) {
                                        // Skip a NULL source for the same reason
                                        // `overlay_local` does: an empty stack slot
                                        // reads NULL here, and stamping that hole onto
                                        // the box would fold a later residual's Ref
                                        // arg to NULL.
                                        if !matches!(
                                            cv,
                                            majit_ir::Value::Void
                                                | majit_ir::Value::Ref(majit_ir::GcRef(0))
                                        ) {
                                            ctx.try_set_opref_concrete(v, cv);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    mirror
                };
                let bridge_locals: Vec<OpRef> =
                    semantic_mirror.iter().take(nlocals).copied().collect();
                // #124 kept operand-stack temps: the stack tail of the same
                // slot-indexed mirror (`[nlocals, nlocals + stack_only)`), so a
                // resumed conditional-expression / short-circuit value survives
                // init_symbolic's later NONE reset regardless of its abstract color.
                let bridge_stack: Vec<OpRef> = semantic_mirror
                    .iter()
                    .skip(nlocals)
                    .take(stack_only)
                    .copied()
                    .collect();

                if majit_metainterp::majit_log_enabled() {
                    eprintln!(
                        "[jit][bridge-sym] frames[0].values={} reg_indices={:?} \
                     bridge_locals={:?} vable_array_items={:?}",
                        frame0.values.len(),
                        reg_indices,
                        bridge_locals,
                        vable_array_items,
                    );
                }
                // Override sym.registers_r so subsequent LOAD_FAST sees the
                // bridge inputarg OpRefs, not the parent's vable_array_base+i
                // OpRefs that init_symbolic seeded before setup_bridge_sym ran.
                //
                // pyre's start_bridge_tracing calls initialize_sym() (which runs
                // init_symbolic) BEFORE setup_bridge_sym, so init_symbolic sees
                // bridge_local_oprefs == None and falls into the vable_array_base
                // branch (init_vable_indices derives vable_array_base = 6 for
                // pyre's `[frame, ec] + four scalars` portal header). That branch produces
                // OpRef::from_raw(base+i) values from the PARENT trace's namespace, leaving
                // stale parent OpRefs in registers_r after we set
                // bridge_local_oprefs here.
                //
                // `VirtualizableInfo.load_list_of_boxes` and `PyFrame.virtualizable_fields`: array item types are
                // all Ref; RETURN_VALUE / arithmetic paths unbox via
                // `trace_guarded_int_payload` (guard_class + getfield_gc_pure_i),
                // matching the RPython unbox-at-consumer model. The slot-level
                // type override is NOT how RPython avoids the guarded path.
                //
                // `semantic_mirror` is slot-indexed (`[locals.., stack_tail..]`),
                // matching what the trace-time mirror reads expect — NOT the
                // color-indexed `bridge_registers_r`. Portal reds (frame/ec) live in
                // their dedicated `sym` fields, so they are absent here by design.
                //
                // Deferred concrete seeding for kept-stack branch guards.
                // At the consume_boxes stage the register bank is color-indexed,
                // and the body marker's colors may not correspond 1:1 to semantic
                // slots — seeding there stamps the wrong value.  After the
                // overlay the mirror is slot-indexed and authoritative, so seed
                // each non-NONE slot from the GC-rooted live frame values.
                if seed_deferred_to_overlay {
                    for (s, opref) in semantic_mirror.iter().enumerate() {
                        if !opref.is_none() {
                            if let Some(&cv) = live_local_values.get(s) {
                                // Skip a NULL (`GcRef(0)`) source: an operand-stack slot
                                // above the frame's materialized `valuestackdepth` reads
                                // NULL from `locals_cells_stack_w` because the kept temp
                                // lives in the guard's register file / resume data, not
                                // the frame array. When a color aliases two slots
                                // (`DUP_TOP` / `ROT_THREE`), one slot may carry the real
                                // value and its sibling the NULL hole; both stamp the
                                // SAME opref, so a NULL stamp would clobber the real one
                                // (last write wins) and fold the residual's Ref arg to
                                // concrete NULL → `MayForceNullRefArgUnsupported`. A real
                                // frame Ref is never NULL here, so skipping NULL only
                                // drops the unmaterialized-hole case.
                                if !matches!(
                                    cv,
                                    majit_ir::Value::Void
                                        | majit_ir::Value::Ref(majit_ir::GcRef(0))
                                ) {
                                    ctx.try_set_opref_concrete(*opref, cv);
                                }
                            }
                        }
                    }
                }
                // Orphaned colors have decoded values but no semantic-mirror slot to
                // re-stamp; seed them so bridge-walk residuals can resolve arguments.
                if let Some(orphan_stamps) = bridge_stamp_orphans.as_ref() {
                    for (opref, cv) in orphan_stamps {
                        if opref.is_none() || semantic_mirror.contains(opref) {
                            continue;
                        }
                        if !matches!(
                            cv,
                            majit_ir::Value::Void | majit_ir::Value::Ref(majit_ir::GcRef(0))
                        ) {
                            ctx.try_set_opref_concrete(*opref, *cv);
                        }
                    }
                }
                // Seed operand-stack slots (kept temps) whose concrete the frame-array
                // overlay above could not source.  A kept operand's authoritative
                // post-guard value is the RESUME DATA (`fail_values`, decoded into
                // `bridge_stamp_orphans`), NOT the frame array — `locals_cells_stack_w`
                // holds a bare `PY_NULL` for an operand-stack temp above the
                // materialized `valuestackdepth`.  A CALL's `self_or_null` sentinel is
                // one such genuine `PY_NULL`: its box decodes to `Ref(GcRef(0))`, which
                // BOTH the frame-array seed and the orphan seed drop as a "hole",
                // leaving `box_value == None`.  The residual then declines that arg as
                // *unavailable* (its `null_or_self` exemption accepts a KNOWN null but
                // not an unavailable box), the effect goes unjournaled, and the legacy
                // replay drops the outer iteration.  Seed each still-unstamped operand
                // opref from its own resume-data decode — INCLUDING a genuine null — so
                // the residual sees the real (null) sentinel and executes.  Scoped to
                // operand slots (`>= nlocals`); locals keep the frame-array overlay,
                // and a slot already carrying a real concrete is left untouched.
                if let Some(orphan_stamps) = bridge_stamp_orphans.as_ref() {
                    for s in nlocals..semantic_prefix_len {
                        let Some(&opref) = semantic_mirror.get(s) else {
                            continue;
                        };
                        if opref.is_none() || ctx.box_value(opref).is_some() {
                            continue;
                        }
                        if let Some((_, cv)) = orphan_stamps.iter().find(|(o, _)| o == &opref) {
                            if !matches!(cv, majit_ir::Value::Void) {
                                ctx.try_set_opref_concrete(opref, *cv);
                            }
                        }
                    }
                }
                seed_bridge_standing_exception_from_current(sym, ctx);
                sym.registers_r.replace(semantic_mirror);
                sym.symbolic_local_types = {
                    let mut types = bridge_local_types.clone();
                    types.resize(sym.nlocals, Type::Ref);
                    types
                };
                // The bridge inputs do NOT have the six-slot portal header that
                // init_vable_indices assumes. Demote this frame from active
                // virtualizable owner so any later LOAD_FAST falling through to
                // the vable_array_base branch uses the heap-array path instead
                // of synthesizing parent OpRefs.
                sym.clear_active_vable();
                // `liveness.compute_liveness`: the explicit `-live-` args keep both
                // portal reds in every guard's frame-register section.  They are not
                // semantic PyFrame slots, so the color->slot inversion above correctly
                // leaves them out of `sym.registers_r`.  A bridge's inputargs come from
                // the failing guard's failargs, so the root trace's historical
                // `InputArgRef(1)` cannot be reused; `bridge_execution_context`, read
                // off the dedicated red color before the inversion, is the correctly
                // renumbered bridge input OpRef.
                sym.execution_context = bridge_execution_context;
                // Both outcomes compile, so only the tally separates the bridge that
                // carries the live red from the one whose first `ec` consumer re-derives
                // it from the thread.
                crate::trace::fbw_diag::bump(if sym.execution_context.is_none() {
                    crate::trace::fbw_diag::BRIDGE_EC_MISSING
                } else {
                    crate::trace::fbw_diag::BRIDGE_EC_FROM_PORTAL_RED
                });
                // pyjitpl.py `rebuild_state_after_failure` parity: after
                // a guard failure the tracing-time `virtualizable_boxes` mirror
                // must be rebuilt from the resume data so subsequent vable
                // ops see OpRefs drawn from the bridge's inputarg stream, not
                // the parent loop's vable_array_base+i indices that
                // init_symbolic seeded before setup_bridge_sym ran.
                //
                // Layout mirrors virtualizable.py read_boxes():
                //   boxes[0..NUM_SCALARS-1] = scalar fields 1..NUM_SCALARS
                //     (vable_last_instr, vable_pycode, vable_valuestackdepth,
                //      vable_debugdata)
                //   boxes[NUM_SCALARS-1..NUM_SCALARS-1+array_len] = array items
                //     (bridge_locals followed by reserved stack slots)
                //   boxes[-1] = vable identity (sym.frame)
                // `PyFrame.locals_cells_stack_w` length =
                // `nlocals + ncells + max_stack`. Pad beyond the bridge's live
                // local prefix with a shared const-NULL OpRef so every
                // interpreter-visible slot has a tracing-time mirror (matches
                // the portal path above).
                // pyframe.py:107-110 + pyjitpl.py: the virtualizable shape  allow-line-citation
                // committed at portal-entry time (`nlocals + ncells +
                // max_stackdepth` array slots) does not change at guard-failure
                // resume; `rebuild_state_after_failure` writes the resume blob
                // into the same `virtualizable_boxes` layout the portal seeded.
                // pyre's root portal seeds via `initialize_virtualizable` with
                // exactly that full layout, so the bridge entry must match it
                // here. Earlier `unwrap_or(nlocals)` undersized the shadow when
                // `concrete_frame_array_len` returned None (e.g. when
                // `sym.concrete_vable_ptr` had not yet been bound at
                // setup_bridge_sym time), causing pushes past `nlocals`/the
                // local prefix to panic at `set_virtualizable_entry_at: index N
                // out of range for N slots`. A probe captured the
                // mismatch directly. With the current four-scalar layout a root whose
                // array has 18 slots needs `vable_boxes_len=23` (= 4 + 18 + 1);
                // falling back to a shorter live prefix would make a later push run
                // past that shadow. Fall back to the metadata-derived
                // size — `metadata.stack_base + metadata.max_stackdepth` is the
                // same `nlocals + ncells + max_stackdepth` the codewriter
                // committed to and `PyFrame::__init__` allocates.
                let bridge_array_len = concrete_frame_array_len(sym.concrete_vable_ptr as usize)
                    .or_else(|| {
                        METAINTERP_SD.with(|r| {
                            let sd = r.borrow();
                            sd.jitcodes.get(frame0.jitcode_index as usize).map(|jc| {
                                jc.payload.metadata.stack_base + jc.payload.metadata.max_stackdepth
                            })
                        })
                    })
                    .unwrap_or(nlocals);
                let scalar_oprefs = [
                    sym.vable_last_instr,
                    sym.vable_pycode,
                    sym.vable_valuestackdepth,
                    sym.vable_debugdata,
                ];
                // virtualizable.py load_list_of_boxes parity: the OpRef half of
                // virtualizable_boxes comes from the resume-data stream
                // (`bridge_decode_box`). The CONCRETE array shadow, however, is read
                // from the restored live virtualizable — NOT from the resume-decoded
                // `vable_array_values`. Those decoded values are off-heap copies of
                // young GC pointers captured at guard-fail time; a minor collection
                // during bridge setup (residual virtual materialization / op recording
                // allocates) moves the referenced objects but cannot forward the
                // off-heap decode Vec, leaving dangling pointers that crash the walk's
                // getarrayitem_vable. The live frame (`sym.concrete_vable_ptr`,
                // restored by `decode_and_restore_guard_failure`) sits on the
                // CURRENT_FRAME chain, so `walk_pyframe_roots` forwards its
                // `locals_cells_stack_w` items on every collection — its slots are
                // always live. Read them directly, mirroring the root-trace seed
                // (`read_boxes` from the rooted portal frame). Falls back to the
                // decoded values only when no live pointer is bound (unit-test /
                // init-before-run). The seed helper pads short arrays with const-NULL
                // OpRef; match that here by padding concrete values with
                // Value::Ref(GcRef::NULL) to the same length.
                let live_array_values = live_frame_array_values(
                    sym.concrete_vable_ptr as usize,
                    bridge_array_len,
                    &vable_array_values,
                );
                // resume.py `rebuild_from_resumedata` fills the live MIFrame
                // register banks with `consume_boxes`; operand-stack boxes are the
                // authoritative values at an after-residual guard.  The separately
                // decoded virtualizable array can still contain the pre-call operand
                // because the residual result has not passed through a frame-array
                // write.  Locals and cells keep the vable-image authority established
                // above: a register color at an interior resume point can hold a dead
                // or stale local value.  Kept-stack branch guards retain their existing
                // post-overlay path: their deeper pcdep stack is not the resumed
                // virtualizable depth.
                let mut bridge_array_items = vable_array_items.clone();
                let mut bridge_array_values = live_array_values.clone();
                bridge_array_values.resize(
                    bridge_array_len,
                    majit_ir::Value::Ref(majit_ir::GcRef::NULL),
                );
                if !seed_deferred_to_overlay {
                    let semantic_array_len = sym.registers_r.len().min(bridge_array_len);
                    if bridge_array_items.len() < semantic_array_len {
                        let null_ref = ctx.const_ref(pyre_object::PY_NULL as i64);
                        bridge_array_items.resize(semantic_array_len, null_ref);
                    }
                    for (slot, opref) in sym
                        .registers_r
                        .iter()
                        .take(semantic_array_len)
                        .enumerate()
                        .skip(nlocals)
                    {
                        if opref.is_none() {
                            continue;
                        }
                        bridge_array_items[slot] = opref;
                        if let Some(value) = ctx.box_value(opref) {
                            if !matches!(value, majit_ir::Value::Void) {
                                bridge_array_values[slot] = value;
                                store_live_frame_array_slot(
                                    sym.concrete_vable_ptr as usize,
                                    slot,
                                    value,
                                );
                            }
                        }
                    }
                }
                sym.concrete_locals = (0..nlocals)
                    .map(|i| {
                        bridge_array_values
                            .get(i)
                            .copied()
                            .map(concrete_value_from_ir_value)
                            .unwrap_or(ConcreteValue::Ref(pyre_object::PY_NULL))
                    })
                    .collect();
                sym.concrete_stack = (0..stack_only)
                    .map(|i| {
                        bridge_array_values
                            .get(nlocals + i)
                            .copied()
                            .map(concrete_value_from_ir_value)
                            .unwrap_or(ConcreteValue::Ref(pyre_object::PY_NULL))
                    })
                    .collect();
                let mut concrete_values =
                    Vec::with_capacity(vable_scalar_values.len() + bridge_array_len);
                concrete_values.extend_from_slice(&vable_scalar_values);
                concrete_values.extend_from_slice(&bridge_array_values);
                crate::state::seed_virtualizable_boxes(
                    ctx,
                    sym.frame,
                    vable_ref_value,
                    &scalar_oprefs,
                    &bridge_array_items,
                    bridge_array_len,
                    &concrete_values,
                    sym.concrete_vable_ptr as *const u8,
                );
                // resume.py `rebuild_from_resumedata` parity: bridge
                // tracing resumes from the full restored frame state via the
                // vable scalar reads + consume_boxes — `valuestackdepth` is
                // recovered from the decoded virtualizable resume payload
                // (`bridge_valuestackdepth` derived from vable.valuestackdepth), not
                // hardcoded to `nlocals`. Keep the stack tail in the unified
                // register file and expose Ref-typed virtualizable slots to
                // subsequent LOAD_FAST / close_loop_args_at calls.
                sym.symbolic_stack_types = vec![Type::Ref; stack_only];
                sym.valuestackdepth = bridge_valuestackdepth;
                sym.bridge_local_oprefs.replace(bridge_locals);
                // #124: preserve the resolved kept operand-stack temps. `bridge_stack`
                // is the stack tail of the slot-indexed `semantic_mirror` (computed
                // above by inverting each live color to its slot), so it is correct
                // even when the stack slot's abstract color is not `nlocals + depth`.
                // init_symbolic runs AFTER setup_bridge_sym in pyre's bridge launcher
                // and would otherwise reset this tail to NONE; keeping it lets both the
                // rebuilt registers_r and the full-body-walk argbox seed recover the
                // kept conditional-expression / short-circuit value.
                sym.bridge_stack_oprefs.replace(bridge_stack);
                // Kept-stack branch guards resume the full-body walk at the guard's OWN
                // mid-opcode jitcode offset — the same resolved coordinate stored in
                // the frame pc — instead of the opcode-entry marker for `py_pc`.
                // `None` leaves the walk on the opcode-entry marker.
                sym.bridge_walk_entry_pc =
                    crate::state::frame_pc_is_resolved_offset_at(frame0.jitcode_index, frame0.pc)
                        .then_some(frame0.pc as usize);
                sym.bridge_walk_entry_jitcode_index = frame0.jitcode_index;
                sym.bridge_local_types = Some(bridge_local_types);
                // consume_boxes (resume.py) fills `f.registers_r` by abstract
                // register color; keep that color-indexed decode so a cross-frame
                // bridge resume snapshot can read `registers_r[color]`
                // (`_get_list_of_active_boxes`). `sym.registers_r`
                // is about to be overwritten with the slot-indexed semantic mirror and
                // then rebuilt by init_symbolic, losing this color decode.
                sym.bridge_registers_r.replace(bridge_registers_r.clone());
            }
        }
        // pyjitpl.py `rebuild_state_after_failure` tail —
        // `consume_virtualref_boxes` (resume.py):
        //   for i in range(0, len(virtualref_boxes), 2):
        //       virtual_box = virtualref_boxes[i]
        //       vref_box    = virtualref_boxes[i + 1]
        //   self.virtualref_boxes += [virtual_box, vref_box]
        // The parent guard already encoded the resumed pair sequence into
        // `rd_numb`'s vref section (resume.py `consume_vref_and_vable`
        // → `consume_virtualref_info`), and `rebuild_from_numbering`
        // (majit/majit-ir/src/resumedata.rs) decoded it into
        // `resume_data.virtualref_values`. Materialize the OpRef + concrete
        // pointer for each pair through the same `resolve` / `decode_concrete`
        // callbacks used for `frames[0].values`, then hand the pairs to the
        // trace's own `virtualref_boxes` — the single store, matching
        // upstream's one `MetaInterp.virtualref_boxes` — so an inlined call's
        // `virtual_ref_finish` and the residual-call bracket both observe the
        // parent's still-open virtualref scope at bridge entry.
        //
        // For traces with no active `virtual_ref`, `virtualref_values` is
        // empty and this loop is a no-op.
        let vref_values = &resume_data.virtualref_values;
        // resume.py:1397 `vrefinfo.continue_tracing(vref, virtual_val)` —
        // restore `JitVirtualRef.forced = virtual` and clear
        // `virtual_token` for every still-active pair. The runtime
        // mutation lives on the live JitVirtualRef heap object, so the
        // call is gated on a non-null concrete `vref_ptr`. Pyre's
        // `VirtualRefInfo::new()` (majit-metainterp/src/virtualref.rs)
        // is a cheap struct-only constructor — instantiate once outside
        // the loop instead of per-pair.
        let vrefinfo = majit_metainterp::virtualref::VirtualRefInfo::new();
        assert_eq!(
            boxes.virtualref_boxes.len(),
            vref_values.len(),
            "consume_vref_and_vable_boxes decoded the virtualref stream"
        );
        let mut restored_virtualref_boxes: Vec<(OpRef, usize)> =
            Vec::with_capacity(vref_values.len());
        for pair in boxes.virtualref_boxes.chunks_exact(2) {
            let (virt_opref, virt_val) = pair[0];
            let (vref_opref, vref_val) = pair[1];
            let virt_ptr = value_to_usize(&virt_val);
            let vref_ptr = value_to_usize(&vref_val);
            restored_virtualref_boxes.push((virt_opref, virt_ptr));
            restored_virtualref_boxes.push((vref_opref, vref_ptr));
            // pyjitpl.py:3438 / resume.py:1397: continue_tracing is called
            // unconditionally for every (vref, real_object) pair. The
            // is_virtual_ref(vref) guard (virtualref.py) and the
            // `assert real_object` invariant (virtualref.py, ported as
            // debug_assert!) both live inside continue_tracing itself
            // (in `virtualref.rs`) — the outer virt_ptr guard masked
            // exactly the case RPython asserts on, so do not pre-gate here.
            unsafe {
                vrefinfo.continue_tracing(vref_ptr as *mut u8, virt_ptr as *mut u8);
            }
        }
        // `ExecutionContext.topframeref` is a live heap field, so unlike the
        // RPython locals `rebuild_state_after_failure` reconstructs, whatever
        // the dead trace last stored there is still in it.  A trace that
        // abandoned an inlined callee level at this guard left that level's
        // vref published — and a compiled-trace vref carries a live FORCE_TOKEN
        // with a null `forced` (`virtualize.py optimize_VIRTUAL_REF`), naming a
        // JIT frame that has now exited.  Forcing it reaches `cpu.force` with
        // no armed `jf_force_descr`.
        //
        // The resume data says what the slot should hold: the innermost scope
        // the guard still had open, whose vref the `continue_tracing` loop
        // above has just repaired, or — with no scope open — the resumed frame
        // itself.  Rewrite only when the slot holds some *other* vref; an
        // already-correct slot (every non-abandoning exit) is left alone.
        let restored_top = restored_virtualref_boxes
            .last()
            .map(|&(_, vref_ptr)| vref_ptr)
            .unwrap_or(sym.concrete_vable_ptr as usize);
        let live_ec = sym.concrete_execution_context as *mut pyre_interpreter::PyExecutionContext;
        if !live_ec.is_null() && restored_top != 0 {
            let published = unsafe { (*live_ec).topframeref };
            // A scope the guard still had open must be republished whatever the
            // slot currently holds: `execute_assembler` restores `topframeref`
            // to the frame it saved before the run, so after an inlined-callee
            // guard the slot names the portal frame — not a vref — and leaving
            // it there would expose the portal to a residual `sys._getframe()`
            // and close the wrong concrete chain on leave.  With no scope open
            // `restored_top` is the resumed frame itself, and only a stale vref
            // in the slot is worth overwriting.
            let stale_scope = !restored_virtualref_boxes.is_empty()
                || unsafe {
                    majit_metainterp::virtualref::ptr_is_virtual_ref(published as *const u8)
                };
            if published as usize != restored_top && stale_scope {
                unsafe {
                    (*live_ec).topframeref = restored_top as *mut pyre_interpreter::PyFrame;
                }
            }
        }
        // `pyjitpl.py self.virtualref_boxes = virtualref_boxes`.
        ctx.restore_virtualref_boxes(restored_virtualref_boxes);

        // `sync_virtualizable_after_guard_failure` runs before bridge setup,
        // but on the multi-frame inlined-callee path its resume-decoded array
        // image is then clobbered by the vsd-correction `clear_stack_above`
        // (`decode_and_restore_guard_failure` in pyre-jit's `eval.rs`), which
        // applies a callee-coordinate depth to the root frame before
        // `setup_bridge_sym` reads it. The per-item write in the vvals loop
        // above re-asserts that array image immediately after
        // each decode; the statics retain the deliberate PR#569 override.

        // Multi-frame bridge. The body above reconstructed
        // the portal (`frames[0]`) into the caller-visible root `sym`. When
        // the guard fired inside inlined callees, `frames[1..]` (OUTERMOST-
        // FIRST) must also be reconstructed and pushed so the framestack
        // matches the inline depth (`rebuild_from_resumedata` resume.py-
        // 1056: every frame via `newframe`+`consume_boxes`, then tracing
        // continues at `framestack[-1]`). Decode each into a lightweight
        // recipe here (while `resume_data` / `virtuals_cache` are in scope);
        // `trace_bytecode` assembles+pushes them right before `interpret()`.
        // Any callee that cannot be faithfully rebuilt aborts the whole
        // multi-frame path back to the single-frame bridge.
        // The outermost (root) frame resumes at its OWN pc once the
        // reconstructed callees return — NOT at the innermost pc that
        // `decode_and_restore_guard_failure` returns as the trace start.
        // Thread `frames[0].pc` so `trace_bytecode` can root the outermost
        // frame at its own pc while the carrier callees resume at their
        // frames[i].pc. A negative (no-snapshot) root pc aborts the
        // multi-frame path.
        if std::env::var_os("PYRE_MFRAME_DIAG").is_some() {
            let pcs: Vec<i32> = resume_data.frames.iter().map(|f| f.pc).collect();
            eprintln!(
                "[mframe] resume frames.len()={} pcs={pcs:?}",
                resume_data.frames.len()
            );
        }
        if resume_data.frames.len() > 1 {
            let root_jitcode_index = resume_data.frames[0].jitcode_index;
            let root_pc_valid = resume_data.frames[0].pc >= 0;
            let root_pc = if root_pc_valid {
                resume_data.frames[0].pc as usize
            } else {
                0
            };
            let mut recipes: Vec<ReconstructRecipe> =
                Vec::with_capacity(resume_data.frames.len() - 1);
            let mut ok = root_pc_valid;
            if root_pc_valid {
                let inline_frames = &resume_data.frames[1..];
                for (idx, frame) in inline_frames.iter().enumerate() {
                    let in_a_call = idx + 1 < inline_frames.len();
                    match reconstruct_inline_recipe(
                        ctx,
                        frame,
                        rd_virtuals,
                        resume_data,
                        fail_values,
                        fail_types,
                        backend,
                        virtuals_cache,
                        in_a_call,
                        &pending_ref_array_writes,
                    ) {
                        Some(recipe) => recipes.push(recipe),
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
            }
            if !ok {
                // A multi-frame resume whose callee chain cannot be
                // reconstructed must still be MARKED as multi-frame: without a
                // carrier the trace-start routing treats the resume as a
                // single-frame bridge and walks the ROOT frame's code at the
                // INNERMOST frame's pc, compiling a degenerate bridge that
                // finishes with a root resume slot as the call result
                // (dropping the whole in-flight callee continuation — wrong
                // values, or a SEGV when the slot is not an int box).  An
                // empty-recipe carrier routes the walk to its NoRecipes abort
                // instead, degrading to the blackhole re-interpret.
                recipes.clear();
            } else if ctx.bridge_source_is_exception_guard() {
                // `prepare_resume_from_failure` records the exception guard
                // after this returns. Those frame ops have to already be in
                // the trace (`pyjitpl.py` `rebuild_from_resumedata`, then
                // `prepare_resume_from_failure`). A non-exception bridge
                // records no guard there; the walk emits its frames.
                record_rebuilt_callee_frames(ctx, sym.execution_context, &mut recipes);
            }
            ctx.set_bridge_inline_carrier(BridgeInlineCarrier {
                root_pc,
                root_jitcode_index,
                recipes,
            });
        }
    }

    /// resume.py rebuild_from_resumedata parity.
    ///
    /// Decodes rd_numb via `majit_ir::resumedata::rebuild_from_numbering`.
    /// Frame box counts come from jitcode liveness (jitcode.position_info)
    /// at the frame's resume pc — the same data the encoder uses via
    /// `get_list_of_active_boxes`.
    fn rebuild_from_resumedata(
        _meta: &mut Self::Meta,
        fail_arg_types: &[Type],
        storage: Option<&std::sync::Arc<majit_metainterp::resume::ResumeStorage>>,
    ) -> Option<majit_metainterp::ResumeDataResult> {
        use majit_ir::resumedata::rebuild_from_numbering;

        // interp_jit.py PyPyJitDriver portal reds are `[frame, ec]`;
        // virtualizable boxes are appended later by
        // MetaInterp::initialize_virtualizable(), not carried as reds.
        _meta.trace_extra_reds = 1;
        let storage = storage?;
        let rd_numb = storage.rd_numb.as_ref();
        // resume.py `self.consts = storage.rd_consts` — borrow
        // the shared pool; `ResumeDataResult` carries the Arc handle
        // so downstream virtual materialization reads the same pool
        // the GC walker updates.
        let rd_consts = storage.rd_consts();

        // resume.py parity: consume_boxes(f.get_current_position_info())
        // RPython uses jitcode liveness via get_current_position_info; majit
        // routes the same lookup through `frame_value_count_at`.
        let cb = crate::state::frame_value_count_at;
        let (num_failargs, vable_values, vref_values, frames) = rebuild_from_numbering(
            rd_numb,
            rd_consts,
            fail_arg_types,
            Some(&cb),
            storage.rd_virtuals.len(),
        );

        // `resume.py` `rebuild_from_resumedata` still returns the box lists
        // when the frame-section loop does not run. `rebuild_state_after_failure`
        // then keeps tracing with that empty framestack.
        Some(majit_metainterp::ResumeDataResult {
            frames,
            virtualizable_boxes: vable_values,
            virtualref_values: vref_values,
            storage: Some(storage.clone()),
            // resume.py rebuild_from_resumedata num_failargs from rd_numb header. Used by
            // bridge virtual materialization (resume.py decode_box
            // negative-index normalization: `num + len(liveboxes)`).
            num_failargs,
            // compile.py ResumeGuardDescr.fail_arg_types — kept so
            // `materialize_bridge_virtual::decode_fieldnum` can mint typed
            // `OpRef::input_arg_typed(idx, fail_arg_types[idx])` per
            // resume.py:1247-1264 (`return self.liveboxes[num]` whose Box
            // carries history.py:220 `box.type`).
            fail_arg_types: fail_arg_types.to_vec(),
        })
    }

    /// pyjitpl.py get_procedure_token: compute green key for a PC.
    fn green_key_for_pc(&self, pc: usize) -> Option<u64> {
        let frame_ptr = self.frame as *const pyre_interpreter::pyframe::PyFrame;
        if frame_ptr.is_null() {
            return None;
        }
        let frame = unsafe { &*frame_ptr };
        Some(crate::driver::make_green_key(
            frame.pycode,
            pc,
            frame.get_is_being_profiled(),
        ))
    }

    fn code_ptr(&self) -> usize {
        let frame_ptr = self.frame as *const pyre_interpreter::pyframe::PyFrame;
        if frame_ptr.is_null() {
            return 0;
        }
        unsafe { (*frame_ptr).pycode as usize }
    }

    fn update_meta_for_cut(meta: &mut Self::Meta, _header_pc: usize, original_box_types: &[Type]) {
        // Current pyre cut traces still recover only the expanded array-item
        // count from the merge-point box vector. Preserve the provisional live
        // depth and update just the heap-array capacity until the later
        // red-only/virtualizable-box migration lands.
        // `NUM_SCALAR_INPUTARGS` already counts `NUM_EXTRA_REDS` (frame +
        // extra_reds + vable scalars), so do NOT add `trace_extra_reds`.
        use crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
        let _ = meta.trace_extra_reds; // staging copy retained for activation gates only
        if original_box_types.len() >= NUM_SCALAR_INPUTARGS {
            let new_capacity = original_box_types
                .len()
                .saturating_sub(NUM_SCALAR_INPUTARGS);
            if meta.slot_types.len() > new_capacity {
                meta.slot_types.truncate(new_capacity);
            } else if meta.slot_types.len() < new_capacity {
                meta.slot_types.resize(new_capacity, Type::Ref);
            }
            meta.array_capacity = new_capacity;
        }
    }

    fn build_meta_from_merge_point(
        provisional: &PyreMeta,
        _header_pc: usize,
        original_boxes: &[majit_metainterp::GreenBox],
    ) -> PyreMeta {
        let original_box_count = original_boxes.len();
        // `NUM_SCALAR_INPUTARGS` already counts `NUM_EXTRA_REDS` (frame +
        // extra_reds + vable scalars), so do NOT add `trace_extra_reds`.
        use crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
        let _ = provisional.trace_extra_reds; // staging copy only
        // RPython parity: Python locals/stack are always Ref.
        let slot_types = if original_box_count >= NUM_SCALAR_INPUTARGS {
            vec![Type::Ref; original_box_count.saturating_sub(NUM_SCALAR_INPUTARGS)]
        } else {
            Vec::new()
        };
        let array_capacity = if original_box_count >= NUM_SCALAR_INPUTARGS {
            original_box_count.saturating_sub(NUM_SCALAR_INPUTARGS)
        } else {
            provisional.array_capacity
        };
        PyreMeta {
            num_locals: provisional.num_locals,
            ns_len: provisional.ns_len,
            namespace_dependent: provisional.namespace_dependent,
            valuestackdepth: provisional.valuestackdepth,
            array_capacity,
            trace_extra_reds: provisional.trace_extra_reds,
            has_virtualizable: provisional.has_virtualizable,
            slot_types,
        }
    }

    fn restore(&mut self, meta: &Self::Meta, values: &[i64]) {
        if values.is_empty() {
            return;
        }

        // Multi-frame format: [num_frames, size_0, data_0..., size_1, data_1...]
        // Detect: values[0] is a small number (1-10) = frame count
        // Legacy: values[0] is a large pointer
        let first = values[0];
        if first >= 1 && first <= 10 && values.len() > 2 {
            let _num_frames = first as usize;
            let outer_size = values[1] as usize;
            if outer_size > 0 && 2 + outer_size <= values.len() {
                // Restore outermost frame only.
                // Inner frame guard failure → interpreter re-executes callee call.
                self.restore_single_frame(meta, &values[2..2 + outer_size]);
                return;
            }
        }

        // Legacy single-frame format
        self.restore_single_frame(meta, values);
    }

    fn restore_values(&mut self, meta: &Self::Meta, values: &[Value]) {
        let Some(_frame) = values.first() else {
            return;
        };
        if majit_metainterp::majit_log_enabled() {
            let arg0 = self.local_at(0).and_then(|value| {
                if value.is_null() || !unsafe { pyre_object::pyobject::is_int(value) } {
                    return None;
                }
                Some(unsafe { pyre_object::intobject::w_int_get_value(value) })
            });
            eprintln!(
                "[jit][restore_values] before arg0={:?} meta.vsd={} has_vable={} values={:?}",
                arg0, meta.valuestackdepth, meta.has_virtualizable, values
            );
        }
        self.restore_expanded_virtualizable_with_extra_reds(meta, values, meta.trace_extra_reds);
        if majit_metainterp::majit_log_enabled() {
            let arg0 = self.local_at(0).and_then(|value| {
                if value.is_null() || !unsafe { pyre_object::pyobject::is_int(value) } {
                    return None;
                }
                Some(unsafe { pyre_object::intobject::w_int_get_value(value) })
            });
            eprintln!(
                "[jit][restore_values] after arg0={:?} ni={} vsd={}",
                arg0,
                self.next_instr(),
                self.valuestackdepth()
            );
        }
    }

    fn restore_guard_failure_values(
        &mut self,
        meta: &Self::Meta,
        values: &[Value],
        _exception: &majit_metainterp::blackhole::ExceptionState,
    ) -> bool {
        if !meta.has_virtualizable {
            self.restore_values(meta, values);
            return true;
        }

        let Some(frame) = values.first() else {
            return false;
        };
        self.frame = value_to_usize(frame);
        if values.len() == 1 {
            return self.validate_frame();
        }

        // The rd_numb deopt path restores the whole virtualizable
        // positionally before this runs (`resume.rs consume_vable_info` →
        // `write_from_resume_data_partial`, resume.py).  The
        // per-frame liveness section fills the blackhole REGISTERS and never
        // touches the vable (blackhole.py:1376+), so the positional restore
        // is authoritative for the frame.  Validate it and clear stale slots
        // beyond valuestackdepth (blackhole fresh-frame parity), which the
        // vable section restore does not perform.
        if !self.validate_frame() {
            return false;
        }
        let vsd = self.valuestackdepth();
        if let Some(arr) = self.locals_cells_stack_array_mut() {
            for i in vsd..arr.len() {
                arr[i] = pyre_object::PY_NULL;
            }
        }
        true
    }

    /// resume.py consume_boxes(info, boxes_i, boxes_r, boxes_f) parity:
    /// Return the type of each slot in the resumed frame section.
    /// In pyre, all frame slots are PyObjectRef (GCREF), so every slot
    /// is Ref. RPython uses typed registers (boxes_i/r/f) but pyre's
    /// virtualizable array is uniformly Ref.
    fn reconstructed_frame_value_types(
        &self,
        meta: &Self::Meta,
        _frame_index: usize,
        _total_frames: usize,
        _frame_pc: u64,
    ) -> Option<Vec<Type>> {
        // resume.py: consume_boxes fills boxes_i/boxes_r/boxes_f.
        // pyre frame slots (locals_cells_stack_w) are all GCREF (Ref).
        let nlocals = meta.num_locals;
        let stack_only = self.valuestackdepth().saturating_sub(nlocals);
        // Header [frame_ptr=Ref, ni=Int, code=Ref, vsd=Int, ns=Ref] + all locals/stack as Ref.
        Some(crate::virtualizable_gen::virt_live_value_types(
            nlocals + stack_only,
        ))
    }

    /// resume.py:1049 parity: restore frame register state from decoded values.
    /// resume.py consume_boxes → _prepare_next_section → enumerate_vars:
    /// each callback_r writes a ref value to the register at the given index.
    /// In pyre, this writes values to the PyFrame's locals/stack via the
    /// virtualizable mechanism (restore_virtualizable_state handles the
    /// full [frame, ni, code, vsd, ns, locals..., stack...] layout).
    fn restore_reconstructed_frame_values(
        &mut self,
        meta: &Self::Meta,
        _frame_index: usize,
        _total_frames: usize,
        _frame_pc: u64,
        values: &[Value],
        exception: &majit_metainterp::blackhole::ExceptionState,
    ) -> bool {
        // resume.py consume_boxes parity: write values to the frame.
        self.restore_guard_failure_values(meta, values, exception)
    }

    /// blackhole.py:1800 parity: multi-frame support.
    fn supports_multi_frame_restore(&self) -> bool {
        true
    }

    /// blackhole.py bhimpl_getarrayitem_gc_r parity: push outer frame for chain.
    /// Multi-frame recovery handled by blackhole chain in call_jit.rs
    /// which receives all frame sections in the typed vector.
    fn push_caller_frame(
        &mut self,
        _meta: &Self::Meta,
        _frame_index: usize,
        _total_frames: usize,
        _values: &[Value],
        _pc: u64,
        _jitcode_index: i32,
    ) -> bool {
        true
    }

    /// blackhole.py:1760 parity: frame transition via chain.
    fn pop_to_caller_frame(&mut self, _meta: &Self::Meta) -> bool {
        false // Blackhole chain handles this directly.
    }

    fn virtualizable_heap_ptr(
        &self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        _info: &VirtualizableInfo,
    ) -> Option<*mut u8> {
        crate::virtualizable_gen::virt_heap_ptr(self, _virtualizable)
    }

    fn virtualizable_array_lengths(
        &self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        info: &VirtualizableInfo,
    ) -> Option<Vec<usize>> {
        if info.array_fields.is_empty() {
            return Some(Vec::new());
        }
        // virtualizable.py read_boxes parity: full array length read from the
        // live heap object (`lst = getattr(virtualizable, fieldname);
        // append(len(lst))`). Upstream has no fallback — if the heap
        // isn't readable here, return `None` so the caller skips.
        if info.can_read_all_array_lengths_from_heap() {
            if let Some(frame_ptr) = self.frame_ptr() {
                let lens = unsafe { info.read_array_lengths_from_heap(frame_ptr) };
                return Some(lens);
            }
        }
        None
    }

    fn sync_virtualizable_before_jit(
        &mut self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        info: &VirtualizableInfo,
    ) -> bool {
        // Heap is source of truth — just validate the frame is usable.
        if !self.validate_frame() {
            return false;
        }
        // virtualizable.py:170 force_token_before_residual_call parity:
        // clear vable_token so the JIT knows the virtualizable is synced.
        if let Some(frame_ptr) = self.frame_ptr() {
            unsafe { info.reset_vable_token(frame_ptr) };
        }
        true
    }

    fn sync_virtualizable_after_jit(
        &mut self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        info: &VirtualizableInfo,
    ) {
        let Some(frame_ptr) = self.frame_ptr() else {
            return;
        };
        // Heap is source of truth — nothing to sync. Just reset token.
        unsafe {
            info.reset_vable_token(frame_ptr);
        }
    }

    fn import_virtualizable_boxes(
        &mut self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        _info: &VirtualizableInfo,
        boxes: &[i64],
        array_lengths: &[usize],
    ) -> bool {
        self.import_virtualizable_state(boxes, array_lengths)
    }

    fn export_virtualizable_boxes(
        &self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        _info: &VirtualizableInfo,
    ) -> Option<(Vec<i64>, Vec<usize>)> {
        Some(self.export_virtualizable_state())
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        Self::pypyjit_collect_jump_args(sym)
    }

    fn collect_jump_args_with_boxes(sym: &Self::Sym, boxes: &[(OpRef, Type)]) -> Vec<OpRef> {
        // pyjitpl.py reached_loop_header:
        //
        //   live_arg_boxes = greenboxes + redboxes
        //   live_arg_boxes += self.virtualizable_boxes
        //   live_arg_boxes.pop()
        //
        // Not reached in production, and the reason matters to anyone
        // reading this as PyFrame's close: this hook is called from the
        // metainterp's `TraceAction::CloseLoop` arm, and nothing under
        // `pyre/` ever produces that action — the portal closes through
        // `CloseLoopWithArgs`, whose args the tracer builds itself in
        // `jitcode_dispatch::append_virtualizable_boxes`. That function
        // splices the same shadow by the same formula, so the two agree;
        // this one exists so the metainterp arm is also correct for PyFrame
        // rather than falling back to the register mirror.
        //
        // PyFrame's reds are [frame, ec].  The TraceCtx shadow is the
        // authoritative `virtualizable_boxes` list in upstream order
        // [static fields..., locals_cells_stack_w..., frame], with the
        // standard-virtualizable identity last.  Portal local/stack writes
        // update that shadow directly; `sym.registers_r` is only pyre's
        // semantic/color-bank adaptation and may still contain the loop-entry
        // boxes.  Splice the live shadow exactly as RPython does instead of
        // delegating to the no-context register-mirror fallback.
        let mut args = Vec::with_capacity(2 + boxes.len().saturating_sub(1));
        args.push(sym.frame);
        args.push(sym.execution_context);
        args.extend(
            boxes
                .iter()
                .take(boxes.len().saturating_sub(1))
                .map(|&(opref, _)| opref),
        );
        args
    }

    fn collect_typed_jump_args(sym: &Self::Sym) -> Vec<(OpRef, Type)> {
        Self::pypyjit_collect_typed_jump_args(sym)
    }

    fn validate_close(sym: &Self::Sym, meta: &Self::Meta) -> bool {
        let _ = (sym, meta);
        true
    }

    fn validate_close_with_jump_args(
        _sym: &Self::Sym,
        meta: &Self::Meta,
        jump_args: &[OpRef],
    ) -> bool {
        let _ = meta;
        // pyre does not close loops by re-reading symbolic stack state from
        // the trace header.  Instead it materializes explicit jump args from
        // the concrete frame-backed virtualizable state at the merge point:
        //   [frame, next_instr, valuestackdepth, locals..., stack...]
        //
        // Retraces/bridges may start from a transient stackful state
        // (e.g. direct-call trace-through in progress) and still legally
        // jump back to a target loop whose merge-point stack is smaller.
        // RPython closes those traces against the target token contract, not
        // the retrace entry state's stack depth.  So for pyre's explicit
        // jump-arg model, the trace-start `meta.valuestackdepth` is not a
        // sound validator here.
        jump_args.len() >= crate::virtualizable_gen::NUM_SCALAR_INPUTARGS
    }

    /// RPython resume.py: materialize a virtual object from resume data.
    ///
    /// Called during guard failure recovery when the optimizer kept an
    /// object virtual (New + SetfieldGc eliminated). The resume mechanism
    /// reconstructs the object so the interpreter can use it.
    fn materialize_virtual_ref(
        &mut self,
        _meta: &Self::Meta,
        _virtual_index: usize,
        materialized: &majit_metainterp::resume::MaterializedVirtual,
    ) -> Option<majit_ir::GcRef> {
        self.materialize_virtual_ref_from_layout(materialized, &[])
    }

    fn materialize_virtual_ref_with_refs(
        &mut self,
        _meta: &Self::Meta,
        _virtual_index: usize,
        materialized: &majit_metainterp::resume::MaterializedVirtual,
        materialized_refs: &[Option<majit_ir::GcRef>],
    ) -> Option<majit_ir::GcRef> {
        self.materialize_virtual_ref_from_layout(materialized, materialized_refs)
    }
}

impl PyreJitState {
    fn materialize_virtual_ref_from_layout(
        &mut self,
        materialized: &majit_metainterp::resume::MaterializedVirtual,
        materialized_refs: &[Option<majit_ir::GcRef>],
    ) -> Option<majit_ir::GcRef> {
        use majit_metainterp::resume::MaterializedVirtual;

        match materialized {
            // resume.py VirtualInfo.allocate — `allocate_with_vtable(descr=self.descr)`
            // followed by `setfields`. descr carries both vtable and obj_size.
            MaterializedVirtual::Obj {
                descr: Some(descr),
                fields,
                ..
            } => materialize_virtual_object(descr, fields, materialized_refs),
            // resume.py VStructInfo.allocate — `allocate_struct(self.typedescr)`
            // + `setfields`. No vtable write.
            MaterializedVirtual::Struct {
                descr: Some(descr),
                fields,
                ..
            } => materialize_virtual_struct(descr, fields, materialized_refs),
            MaterializedVirtual::RawBuffer {
                func,
                size,
                offsets,
                descrs,
                values,
            } => materialize_virtual_raw_buffer(
                *func,
                *size,
                offsets,
                descrs,
                values,
                materialized_refs,
            ),
            _ => None,
        }
    }
}

/// resume.py VirtualInfo.allocate parity — `allocate_with_vtable(descr)`
/// then `setfields`.
///
/// Allocates `descr.size()` bytes aligned to 8, seeds the PyObject header
/// (`ob_type = descr.vtable()`, `w_class = get_instantiate(vtable)`), then
/// replays each traced field using the FieldDescr's byte offset/size/type
/// from the SizeDescr's `all_fielddescrs()` table (resume.py
/// setfields loop — `decoder.setfield(struct, num, descr)`).
///
/// The vtable source is the descr itself — there is no `type_id` special-casing,
/// so bool (type_id=0, vtable=&BOOL_TYPE), range-iter (type_id=0,
/// vtable=&RANGE_ITER_TYPE) and custom classes all dispatch uniformly.
fn materialize_virtual_object(
    descr: &DescrRef,
    fields: &[(u32, majit_metainterp::resume::MaterializedValue)],
    materialized_refs: &[Option<majit_ir::GcRef>],
) -> Option<majit_ir::GcRef> {
    use pyre_object::pyobject::{OB_TYPE_OFFSET, PyObject, PyType, W_CLASS_OFFSET};

    let size_descr = descr.as_size_descr()?;
    let vtable = size_descr.vtable();
    let obj_size = size_descr.size();
    if vtable == 0 || obj_size < std::mem::size_of::<PyObject>() {
        return None;
    }

    // resume.py allocate_with_vtable — raw heap allocation that
    // matches `Box::leak(Box::new(...))` for pyre's existing W_*Object
    // builders. 8-byte alignment matches the natural alignment of
    // `#[repr(C)]` structs whose first field is an 8-byte pointer.
    let layout = std::alloc::Layout::from_size_align(obj_size, 8).ok()?;
    let raw = unsafe { std::alloc::alloc_zeroed(layout) };
    if raw.is_null() {
        return None;
    }

    unsafe {
        let ptr = raw as *mut PyObject;
        (*ptr).ob_type = vtable as *const PyType;
        // Seed `w_class` from the type's cached instantiate pointer.  This
        // has no upstream counterpart: `OBJECT` carries the single field
        // `typeptr` (rclass.py) and `new_instance` stores only that
        // one word (rclass.py), while `instantiate` is a slot in
        // `OBJECT_VTABLE` (rclass.py), i.e. per class, never per
        // instance.  `heaptracker.py:66-69` skips exactly the name
        // `typeptr` and recurses into every other field of a nested header
        // struct, so upstream a second header word would be an ordinary
        // traced field — as the Python-level class is for user subclasses
        // (`mapdict.py getclass`, reached through the ordinary `map` field).
        // Tracing may therefore overwrite this via an explicit
        // `SetfieldGc(w_class)`, and the field replay below takes precedence.
        //
        // `w_class_obj()` is None when the vtable word is not a `PyType`
        // pointer — `JitVirtualRef` stores the `jit_virtual_ref_vtable`
        // type-id constant at offset 0 (`virtualref.py:21-23`) and keeps
        // `virtual_token` where `w_class` would sit.
        if let Some(w_class) = size_descr.w_class_obj() {
            (*ptr).w_class = w_class as *mut PyObject;
        }
    }

    // resume.py setfields parity: for each traced field,
    // look up the FieldDescr from the SizeDescr table and write the
    // value at descr.offset() with descr.field_size() bytes.
    for (field_idx, value) in fields {
        let Some(field_descr) = size_descr
            .all_fielddescrs()
            .iter()
            .find(|fd| fd.index() == *field_idx)
        else {
            // Field not in the SizeDescr table — skip (ob_type handled
            // above; anything else is a tracer bug we just log-silently
            // drop rather than corrupt memory).
            continue;
        };
        let offset = field_descr.offset();
        let field_size = field_descr.field_size();
        if offset == OB_TYPE_OFFSET && field_size == std::mem::size_of::<*const PyType>() {
            // ob_type already written from vtable; honour explicit override.
            let concrete = value.resolve_with_refs(materialized_refs)?;
            unsafe {
                (raw.add(offset) as *mut usize).write(concrete as usize);
            }
            continue;
        }
        if offset == W_CLASS_OFFSET && field_size == std::mem::size_of::<*mut PyObject>() {
            let concrete = value.resolve_with_refs(materialized_refs)?;
            unsafe {
                (raw.add(offset) as *mut usize).write(concrete as usize);
            }
            continue;
        }
        let concrete = value.resolve_with_refs(materialized_refs)?;
        unsafe {
            write_field_bytes(raw, offset, field_size, concrete);
        }
    }

    Some(majit_ir::GcRef(raw as usize))
}

/// resume.py VStructInfo.allocate parity — `allocate_struct(typedescr)`
/// (no vtable) + `setfields`.
fn materialize_virtual_struct(
    descr: &DescrRef,
    fields: &[(u32, majit_metainterp::resume::MaterializedValue)],
    materialized_refs: &[Option<majit_ir::GcRef>],
) -> Option<majit_ir::GcRef> {
    let size_descr = descr.as_size_descr()?;
    let obj_size = size_descr.size();
    if obj_size == 0 {
        return None;
    }
    let layout = std::alloc::Layout::from_size_align(obj_size, 8).ok()?;
    let raw = unsafe { std::alloc::alloc_zeroed(layout) };
    if raw.is_null() {
        return None;
    }
    for (field_idx, value) in fields {
        let Some(field_descr) = size_descr
            .all_fielddescrs()
            .iter()
            .find(|fd| fd.index() == *field_idx)
        else {
            continue;
        };
        let offset = field_descr.offset();
        let field_size = field_descr.field_size();
        let concrete = value.resolve_with_refs(materialized_refs)?;
        unsafe {
            write_field_bytes(raw, offset, field_size, concrete);
        }
    }
    Some(majit_ir::GcRef(raw as usize))
}

/// Write an i64 value into a byte-size field at `offset` inside `raw`.
/// Supports 1/2/4/8 byte widths. 8-byte write also handles Ref/Float
/// (Float arrives as `value.to_bits() as i64`; raw bytes reinterpret).
///
/// # Safety
/// Caller guarantees `raw + offset + field_size` is within the object allocation.
unsafe fn write_field_bytes(raw: *mut u8, offset: usize, field_size: usize, value: i64) {
    unsafe {
        let dst = raw.add(offset);
        match field_size {
            1 => (dst as *mut u8).write(value as u8),
            2 => (dst as *mut u16).write(value as u16),
            4 => (dst as *mut u32).write(value as u32),
            8 => (dst as *mut u64).write(value as u64),
            _ => {
                // Fallback: write as raw bytes LE, truncated.
                let bytes = value.to_le_bytes();
                for i in 0..field_size.min(bytes.len()) {
                    dst.add(i).write(bytes[i]);
                }
            }
        }
    }
}

fn materialize_virtual_raw_buffer(
    func: i64,
    size: usize,
    offsets: &[i64],
    descrs: &[majit_ir::ArrayDescrInfo],
    values: &[majit_metainterp::resume::MaterializedValue],
    materialized_refs: &[Option<majit_ir::GcRef>],
) -> Option<majit_ir::GcRef> {
    // resume.py VRawBufferInfo.allocate_int iterates len(self.offsets)
    // unconditionally, indexing self.descrs[i]/self.fieldnums[i] by the same i.
    // No len-equality assert (VRawBufferInfo has none); a short descrs/values
    // raises IndexError here (encoder bug), a longer one leaves its tail unread.
    let (driver, _) = crate::driver::driver_pair();
    // resume.py allocate_raw_buffer:
    //   cic = self.callinfocollection
    //   calldescr, _ = cic.callinfo_for_oopspec(OS_RAW_MALLOC_VARSIZE_CHAR)
    //   return self.cpu.bh_call_i(func, [size], None, None, calldescr)
    // The calldescr comes from the shared callinfocollection, not a freshly
    // minted MOST_GENERAL descr.  func stays the VRawBufferInfo.func (resume.py
    // discards the callinfo's func as `_`).
    let cic = driver
        .meta_interp()
        .callinfocollection()
        .expect(
            "materialize_virtual_raw_buffer: MetaInterp.callinfocollection \
             required for VRawBufferInfo recovery (resume.py:1453)",
        )
        .clone();
    // resume.py:1455: calldescr, _ = cic.callinfo_for_oopspec(...).
    // callinfo_for_oopspec returns (None, 0) on a miss (effectinfo.py)
    // — no lookup-time guard; the calldescr is consumed directly below.
    let (calldescr, _) =
        cic.callinfo_for_oopspec(majit_ir::descr::OopSpecIndex::RawMallocVarsizeChar);
    let descr_ref = calldescr.expect("OS_RAW_MALLOC_VARSIZE_CHAR calldescr (callinfocollection)");
    let cd = descr_ref
        .as_call_descr()
        .expect("OS_RAW_MALLOC_VARSIZE_CHAR: not a CallDescr");
    let bh_calldescr = majit_jitcode::jitcode::BhCallDescr::from_arg_classes(
        cd.arg_classes(),
        cd.result_class(),
        cd.get_extra_info().clone(),
    );
    // resume.py:703: buffer = self.cpu.bh_call_i(func, [size], None, None, calldescr)
    let buffer = driver.meta_interp().backend().bh_call_i(
        func,
        Some(&[size as i64]),
        None,
        None,
        &bh_calldescr,
    );

    let backend = driver.meta_interp().backend();
    // resume.py:705-708: per-item bh_raw_store_i/f, run unconditionally (no
    // buffer == 0 guard).
    for i in 0..offsets.len() {
        let concrete = values[i].resolve_with_refs(materialized_refs)?;
        let di = &descrs[i];
        let bh_descr = majit_jitcode::jitcode::BhDescr::from_array_descr_info(di);
        // resume.py:1544: assert not descr.is_array_of_pointers()
        assert!(
            !bh_descr.is_array_of_pointers(),
            "raw buffer entry must not be pointer type"
        );
        let offset = offsets[i];
        if di.item_type == 2 {
            // resume.py:1545-1547: descr.is_array_of_floats() → bh_raw_store_f
            backend.bh_raw_store_f(buffer, offset, f64::from_bits(concrete as u64), &bh_descr);
        } else {
            // resume.py:1548-1550: else → bh_raw_store_i
            backend.bh_raw_store_i(buffer, offset, concrete, &bh_descr);
        }
    }

    Some(majit_ir::GcRef(buffer as usize))
}

fn value_to_usize(value: &Value) -> usize {
    match value {
        Value::Ref(gc_ref) => gc_ref.0,
        Value::Int(n) => *n as usize,
        _ => 0,
    }
}

/// Process-wide callback table for trace tests that reach `driver_pair`.
///
/// `callbacks::init` accepts one table pointer for the process. Every test
/// that needs a driver goes through this function so a second leaked table
/// cannot trip that assert.
#[cfg(test)]
pub(crate) fn ensure_trace_test_driver() -> &'static mut crate::driver::JitDriverPair {
    use std::cell::UnsafeCell;
    static INIT: std::sync::Once = std::sync::Once::new();
    thread_local! {
        static TEST_JIT_DRIVER: UnsafeCell<crate::driver::JitDriverPair> = UnsafeCell::new({
            let info = crate::frame_layout::build_pyframe_virtualizable_info();
            let mut driver = majit_metainterp::JitDriver::new(1);
            driver.set_virtualizable_info(info.clone());
            (driver, info)
        });
    }
    INIT.call_once(|| {
        let cb = Box::leak(Box::new(crate::callbacks::CallJitCallbacks {
            callee_frame_helper: |_| None,
            recursive_force_cache_safe: |_| false,
            jit_drop_callee_frame: std::ptr::null(),
            jit_force_callee_frame: std::ptr::null(),
            jit_force_recursive_call_1: std::ptr::null(),
            jit_force_recursive_call_argraw_boxed_1: std::ptr::null(),
            jit_force_self_recursive_call_argraw_boxed_1: std::ptr::null(),
            jit_create_callee_frame_1: std::ptr::null(),
            jit_create_callee_frame_1_raw_int: std::ptr::null(),
            jit_create_self_recursive_callee_frame_1: std::ptr::null(),
            jit_create_self_recursive_callee_frame_1_raw_int: std::ptr::null(),
            driver_pair: || TEST_JIT_DRIVER.with(|cell| cell.get() as *mut u8),
            ensure_majit_jitcode: |_, _| false,
            drain_backend_jit_exc: || {},
        }));
        crate::callbacks::init(cb);
    });
    TEST_JIT_DRIVER.with(|cell| unsafe { &mut *cell.get() })
}

#[cfg(test)]
mod tests {
    //! Most tests in this module are original Rust regressions.
    //!
    //! PyPy/Pyre upstream tests usually assert behavior at the meta-interpreter
    //! level, not at the typed-raw tracer seam exercised here. These tests keep
    //! local descriptor choices, virtualizable restoration, and raw-int/raw-f64
    //! fast paths from drifting silently. Where a test is tied directly to an
    //! upstream parity point, that reference is called out inline.

    use super::*;
    use majit_metainterp::JitState;
    use majit_metainterp::resume::{MaterializedValue, MaterializedVirtual};
    use pyre_interpreter::bytecode::{CodeObject, ConstantData, Instruction};
    use pyre_interpreter::pyopcode::decode_instruction_at;
    use pyre_interpreter::{Mode, compile_exec, compile_source};
    use pyre_object::floatobject::w_float_get_value;
    use pyre_object::listobject::w_list_getitem;
    use pyre_object::pyobject::{INT_TYPE, PyType, is_list};

    #[test]
    fn reconstructed_ref_slot_does_not_invent_color_in_mapped_frame() {
        let pcdep = [(1, 4, 0), (1, 5, 2), (1, 6, 3)];

        assert_eq!(reconstructed_ref_slot_color(&pcdep, 0), Some(4));
        assert_eq!(reconstructed_ref_slot_color(&pcdep, 1), None);
        assert_eq!(reconstructed_ref_slot_color(&[], 1), Some(1));
    }

    #[test]
    fn materialized_inline_frame_slots_are_symbolic_heap_reads() {
        let code = compile_function_body("def f(a, b, c):\n    return a if b is None else c\n");
        let mut frame = pyre_interpreter::pyframe::PyFrame::new(code);
        let values = [w_int_new(11), pyre_object::w_none(), w_int_new(33)];
        for (index, value) in values.iter().copied().enumerate() {
            locals_w_mut!(frame)[index] = value;
        }
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut pyre_interpreter::pyframe::PyFrame as usize;

        let mut ctx = TraceCtx::for_test_types(&[Type::Ref]);
        let frame_box = OpRef::input_arg_ref(0);
        ctx.try_set_opref_concrete(frame_box, Value::Ref(majit_ir::GcRef(frame_ptr)));

        let (registers_r, concrete_r) = reconstruct_materialized_frame_slots(
            &mut ctx,
            frame_box,
            majit_ir::GcRef(frame_ptr),
            values.len(),
            Some(1),
            &[],
            &[],
        )
        .expect("forced frame slots should be reconstructable");

        assert!(!registers_r[0].is_none());
        assert!(registers_r[1].is_none());
        assert!(!registers_r[2].is_none());
        assert!(
            !matches!(registers_r[0], OpRef::ConstPtr(_))
                && !matches!(registers_r[2], OpRef::ConstPtr(_)),
            "bridge slots must reload from the current frame, not bake constants"
        );
        assert_eq!(
            concrete_r[0],
            Value::Ref(majit_ir::GcRef(values[0] as usize))
        );
        assert_eq!(concrete_r[1], Value::Void);
        assert_eq!(
            concrete_r[2],
            Value::Ref(majit_ir::GcRef(values[2] as usize))
        );

        let tree_loop = ctx.into_tree_loop();
        assert_eq!(
            tree_loop
                .ops
                .iter()
                .filter(|op| op.opcode == OpCode::GetarrayitemGcR)
                .count(),
            2
        );
    }

    #[test]
    fn materialized_inline_frame_consumes_pending_ref_array_write() {
        let code = compile_function_body("def f(a, b):\n    return a\n");
        let mut frame = pyre_interpreter::pyframe::PyFrame::new(code);
        let values = [w_int_new(11), w_int_new(22)];
        for (index, value) in values.iter().copied().enumerate() {
            locals_w_mut!(frame)[index] = value;
        }
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut pyre_interpreter::pyframe::PyFrame as usize;
        let array_ptr = frame.locals_cells_stack_w as usize;

        let mut ctx = TraceCtx::for_test_types(&[Type::Ref, Type::Ref, Type::Ref]);
        let frame_box = OpRef::input_arg_ref(0);
        let target = OpRef::input_arg_ref(1);
        let value = OpRef::input_arg_ref(2);
        ctx.try_set_opref_concrete(frame_box, Value::Ref(majit_ir::GcRef(frame_ptr)));
        ctx.try_set_opref_concrete(target, Value::Ref(majit_ir::GcRef(array_ptr)));
        ctx.try_set_opref_concrete(value, Value::Ref(majit_ir::GcRef(values[1] as usize)));

        let writes = [PendingRefArrayWrite {
            target,
            item_index: 1,
            value,
        }];
        let (registers_r, concrete_r) = reconstruct_materialized_frame_slots(
            &mut ctx,
            frame_box,
            majit_ir::GcRef(frame_ptr),
            values.len(),
            None,
            &[],
            &writes,
        )
        .expect("forced frame slots should consume the pending write stream");

        assert_eq!(registers_r[1], value);
        assert_eq!(
            concrete_r[1],
            Value::Ref(majit_ir::GcRef(values[1] as usize))
        );
    }

    #[test]
    fn materialized_inline_frame_slots_skip_stream_covered_slots() {
        // A slot the resume stream names is authoritative over the frame image
        // (resume.py consume_boxes), so the image must not be read
        // for it: the caller's `overlay_stream_ref_slots` owns that slot and
        // would otherwise be overwriting a stale array entry.
        let code = compile_function_body("def f(a, b, c):\n    return a if b is None else c\n");
        let mut frame = pyre_interpreter::pyframe::PyFrame::new(code);
        let values = [w_int_new(11), w_int_new(22), w_int_new(33)];
        for (index, value) in values.iter().copied().enumerate() {
            locals_w_mut!(frame)[index] = value;
        }
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut pyre_interpreter::pyframe::PyFrame as usize;

        let mut ctx = TraceCtx::for_test_types(&[Type::Ref]);
        let frame_box = OpRef::input_arg_ref(0);
        ctx.try_set_opref_concrete(frame_box, Value::Ref(majit_ir::GcRef(frame_ptr)));

        let (registers_r, concrete_r) = reconstruct_materialized_frame_slots(
            &mut ctx,
            frame_box,
            majit_ir::GcRef(frame_ptr),
            values.len(),
            None,
            &[false, true, false],
            &[],
        )
        .expect("forced frame slots should be reconstructable");

        assert!(registers_r[1].is_none(), "stream-covered slot stays unread");
        assert_eq!(concrete_r[1], Value::Void);
        assert!(!registers_r[0].is_none());
        assert!(!registers_r[2].is_none());
        assert_eq!(
            concrete_r[0],
            Value::Ref(majit_ir::GcRef(values[0] as usize))
        );
        assert_eq!(
            concrete_r[2],
            Value::Ref(majit_ir::GcRef(values[2] as usize))
        );

        let tree_loop = ctx.into_tree_loop();
        assert_eq!(
            tree_loop
                .ops
                .iter()
                .filter(|op| op.opcode == OpCode::GetarrayitemGcR)
                .count(),
            2
        );
    }

    #[test]
    fn concrete_value_preserves_int_subclass_identity() {
        pyre_interpreter::typedef::init_typeobjects();
        let obj = pyre_object::intobject::w_int_subclass_new(7);

        assert_eq!(ConcreteValue::from_pyobj(obj), ConcreteValue::Ref(obj));
    }

    #[test]
    fn concrete_value_still_unboxes_exact_int_and_bool() {
        pyre_interpreter::typedef::init_typeobjects();
        let int_obj = pyre_object::w_int_new(11);
        let true_obj = pyre_object::w_bool_from(true);
        let float_obj = pyre_object::w_float_new(3.25);

        assert_eq!(ConcreteValue::from_pyobj(int_obj), ConcreteValue::Int(11));
        assert_eq!(
            ConcreteValue::from_pyobj(true_obj),
            ConcreteValue::Bool(true)
        );
        assert_eq!(
            ConcreteValue::from_pyobj(float_obj),
            ConcreteValue::Float(3.25)
        );
    }

    #[test]
    fn concrete_value_preserves_float_subclass_identity() {
        pyre_interpreter::typedef::init_typeobjects();
        let obj = pyre_object::w_float_new(2.5);
        unsafe {
            (*obj).w_class = pyre_object::w_none();
        }

        assert_eq!(ConcreteValue::from_pyobj(obj), ConcreteValue::Ref(obj));
    }

    /// Minimal Backend for `field_sanity_load` / `arraylen_sanity_load`:
    /// the default `bh_getfield_gc_*` trait methods read raw memory, and
    /// `bh_arraylen_gc` mirrors the production overrides (the trait
    /// default is a stub returning 0).  Compilation/execution entry
    /// points are unreachable in these tests.
    struct FieldLoadTestCpu;
    impl majit_backend::Backend for FieldLoadTestCpu {
        fn compile_loop(
            &mut self,
            _inputargs: &[majit_ir::InputArgRc],
            _ops: &[majit_ir::OpRc],
            _token: &majit_backend::JitCellToken,
        ) -> Result<majit_backend::AsmInfo, majit_backend::BackendError> {
            unimplemented!("FieldLoadTestCpu::compile_loop")
        }
        fn compile_bridge(
            &mut self,
            _fail_descr: &dyn majit_ir::FailDescr,
            _inputargs: &[majit_ir::InputArgRc],
            _ops: &[majit_ir::OpRc],
            _original_token: &majit_backend::JitCellToken,
            _previous_tokens: &[std::sync::Arc<majit_backend::JitCellToken>],
            _caller_recovery_layout: Option<&majit_backend::ExitRecoveryLayout>,
        ) -> Result<majit_backend::AsmInfo, majit_backend::BackendError> {
            unimplemented!("FieldLoadTestCpu::compile_bridge")
        }
        fn execute_token(
            &self,
            _token: &majit_backend::JitCellToken,
            _args: &[majit_ir::Value],
        ) -> majit_backend::DeadFrame {
            unimplemented!("FieldLoadTestCpu::execute_token")
        }
        fn get_latest_descr<'a>(
            &'a self,
            _frame: &'a majit_backend::DeadFrame,
        ) -> &'a dyn majit_ir::FailDescr {
            unimplemented!("FieldLoadTestCpu::get_latest_descr")
        }
        fn get_latest_descr_arc(
            &self,
            _frame: &majit_backend::DeadFrame,
        ) -> std::sync::Arc<dyn majit_ir::descr::Descr> {
            unimplemented!("FieldLoadTestCpu::get_latest_descr_arc")
        }
        fn get_int_value(&self, _frame: &majit_backend::DeadFrame, _index: usize) -> i64 {
            unimplemented!("FieldLoadTestCpu::get_int_value")
        }
        fn get_value_direct(&self, frame: &majit_backend::DeadFrame, slot: usize) -> i64 {
            // FieldLoadTestCpu's slot space is the dense fail-value vector.
            self.get_int_value(frame, slot)
        }
        fn get_float_value(&self, _frame: &majit_backend::DeadFrame, _index: usize) -> f64 {
            unimplemented!("FieldLoadTestCpu::get_float_value")
        }
        fn get_ref_value(
            &self,
            _frame: &majit_backend::DeadFrame,
            _index: usize,
        ) -> majit_ir::GcRef {
            unimplemented!("FieldLoadTestCpu::get_ref_value")
        }
        fn invalidate_loop(&self, _token: &majit_backend::JitCellToken) {
            unimplemented!("FieldLoadTestCpu::invalidate_loop")
        }
        fn bh_arraylen_gc(
            &self,
            array_ptr: i64,
            arraydescr: &majit_jitcode::jitcode::BhDescr,
        ) -> i64 {
            let ofs = arraydescr
                .array_len_offset()
                .expect("bh_arraylen_gc requires ArrayDescr.lendescr");
            unsafe { *((array_ptr as *const u8).add(ofs) as *const usize) as i64 }
        }
    }

    /// `_opimpl_getfield_gc_any_pureornot`: a `ConstPtr` receiver through
    /// an always-pure descr bypasses the heapcache and folds to the loaded
    /// value with no recorded op; a non-pure descr still records.
    #[test]
    fn getfield_gc_const_receiver_through_pure_descr_folds_without_recording() {
        let cpu = FieldLoadTestCpu;
        let mut ctx = TraceCtx::for_test(0);
        ctx.set_cpu(Some(&cpu));
        // Fake two-word struct: the field at offset 8 holds 77.
        let storage: Box<[i64; 2]> = Box::new([0, 77]);
        let ptr = storage.as_ref().as_ptr() as i64;
        let obj = ctx.const_ref(ptr);

        let pure_descr = crate::descr::make_immutable_field_descr(8, 8, majit_ir::Type::Int, true);
        let ops_before = ctx.num_ops();
        let result = opimpl_getfield_gc_i(&mut ctx, obj, pure_descr);
        assert_eq!(
            ctx.num_ops(),
            ops_before,
            "const receiver + always-pure descr records nothing"
        );
        assert_eq!(
            result.inline_const_to_value(),
            Some(majit_ir::Value::Int(77)),
            "the fold substitutes the loaded value as a Const"
        );
        let prof_folded = ctx.profiler().snapshot();
        assert_eq!(prof_folded.ops, 0);
        assert_eq!(prof_folded.recorded_ops, 0);

        // Control: a non-pure descr on the same const receiver records.
        let mutable_descr = crate::descr::make_field_descr(8, 8, majit_ir::Type::Int, true);
        let recorded = opimpl_getfield_gc_i(&mut ctx, obj, mutable_descr);
        assert_eq!(
            ctx.num_ops(),
            ops_before + 1,
            "a non-pure field read on a const receiver still records"
        );
        assert!(!recorded.is_constant());
        let prof_recorded = ctx.profiler().snapshot();
        assert_eq!(prof_recorded.ops, 1);
        assert_eq!(prof_recorded.recorded_ops, 1);
    }

    /// `opimpl_arraylen_gc` + `execute_and_record`: ARRAYLEN_GC is
    /// ALWAYS_PURE, so a `Const` array operand folds to the loaded length
    /// without recording.
    #[test]
    fn arraylen_gc_const_array_folds_to_the_loaded_length_without_recording() {
        let cpu = FieldLoadTestCpu;
        let mut ctx = TraceCtx::for_test(0);
        ctx.set_cpu(Some(&cpu));
        // Fake GcArray block: the length prefix at offset 0 holds 5.
        let storage: Box<[usize; 2]> = Box::new([5, 0]);
        let ptr = storage.as_ref().as_ptr() as i64;
        let array = ctx.const_ref(ptr);
        let descr = crate::descr::make_array_descr(8, 8, Some(0), majit_ir::Type::Int, true);

        let ops_before = ctx.num_ops();
        let prof_before = ctx.profiler().snapshot();
        let result = opimpl_arraylen_gc(&mut ctx, array, descr.clone());
        assert_eq!(
            ctx.num_ops(),
            ops_before,
            "a const array operand folds without recording an ArraylenGc"
        );
        assert_eq!(
            result.inline_const_to_value(),
            Some(majit_ir::Value::Int(5)),
            "the fold substitutes the loaded length as a ConstInt"
        );
        // execute_and_record counts OPS at entry even when `_all_constants`
        // folds; RECORDED_OPS only accompanies an actual record.
        let prof_folded = ctx.profiler().snapshot();
        assert_eq!(prof_folded.ops, prof_before.ops + 1);
        assert_eq!(prof_folded.recorded_ops, prof_before.recorded_ops);

        // The heapcache skips Const keys both ways (`arraylen` /
        // `arraylen_now_known`), so a repeat call re-folds: OPS counts
        // again, still nothing recorded or heapcached.
        let refolded = opimpl_arraylen_gc(&mut ctx, array, descr);
        assert_eq!(refolded, result);
        let prof_refolded = ctx.profiler().snapshot();
        assert_eq!(prof_refolded.ops, prof_folded.ops + 1);
        assert_eq!(prof_refolded.recorded_ops, prof_folded.recorded_ops);
        assert_eq!(prof_refolded.heapcached_ops, prof_folded.heapcached_ops);
    }

    #[test]
    fn arraylen_gc_nonconst_array_records_and_counts_ops() {
        let mut ctx = TraceCtx::for_test_types(&[majit_ir::Type::Ref]);
        let array = OpRef::input_arg_ref(0);
        let descr = crate::descr::make_array_descr(8, 8, Some(0), majit_ir::Type::Int, true);

        let ops_before = ctx.num_ops();
        let prof_before = ctx.profiler().snapshot();
        let result = opimpl_arraylen_gc(&mut ctx, array, descr);
        assert!(!result.is_constant());
        assert_eq!(
            ctx.num_ops(),
            ops_before + 1,
            "a non-const array operand records an ArraylenGc"
        );
        let prof_after = ctx.profiler().snapshot();
        assert_eq!(prof_after.ops, prof_before.ops + 1);
        assert_eq!(prof_after.recorded_ops, prof_before.recorded_ops + 1);
    }

    fn ensure_test_callbacks() {
        let _ = super::ensure_trace_test_driver();
    }

    /// pyjitpl.py `blackhole_if_trace_too_long` disables through
    /// `jd_sd.warmstate.disable_noninlinable_function` on the huge
    /// function's own driver, not portal slot 0.
    ///
    /// The shipped FBW door is `note_root_trace_too_long`. The inlined
    /// callee is the biggest function and the portal merge key is present,
    /// so `prepare_trace_segmenting` must not run: the callee cell is
    /// `JC_DONT_TRACE_HERE` and the portal cell is not.
    #[test]
    fn note_root_trace_too_long_disables_the_huge_function_driver() {
        use majit_ir::{GreenKey, Type};
        use majit_metainterp::BackEdgeAction;
        use majit_metainterp::warmstate::JcFlags;

        let pair = super::ensure_trace_test_driver();
        extern "C" fn portal_runner_helper() -> i64 {
            0
        }
        let portal = GreenKey::with_types(vec![7, 0, 0xC0DE], [Type::Int, Type::Int, Type::Ref]);
        let callee = GreenKey::with_types(vec![0, 0, 0xCA11], [Type::Int, Type::Int, Type::Ref]);
        let portal_hash = portal.get_uhash();
        let callee_hash = callee.get_uhash();
        assert_ne!(
            portal_hash, callee_hash,
            "fixture: portal and callee must not share a cell key"
        );
        let idx1;
        {
            let meta = pair.0.meta_interp_mut();
            let mut first = JitDriverStaticData::new(vec![], vec![]);
            first.portal_runner_adr = portal_runner_helper as *const () as i64;
            first.index = Some(0);
            let mut second = JitDriverStaticData::new(vec![], vec![]);
            second.portal_runner_adr = portal_runner_helper as *const () as i64;
            second.index = Some(1);
            second.is_recursive = true;
            let idx0 = meta.register_jitdriver_sd(first);
            idx1 = meta.register_jitdriver_sd(second.clone());
            assert_eq!(idx0, 0);
            assert_eq!(idx1, 1);
            assert_eq!(second.index, Some(1));
            meta.finish_setup_descrs_for_jitdrivers();

            let action = meta.bound_reached(
                portal_hash,
                (0xC0DE, 7),
                Some(portal.clone()),
                Some(second),
                &[],
            );
            assert!(matches!(action, BackEdgeAction::StartedTracing));
            assert_eq!(meta.active_jitdriver_sd, Some(idx1));
            assert!(
                meta.portal_trace_positions.is_some(),
                "bound_reached arms portal_trace_positions"
            );

            fn pos(n: usize) -> majit_metainterp::recorder::TracePosition {
                majit_metainterp::recorder::TracePosition {
                    _pos: n,
                    _count: 0,
                    _index: 0,
                    snapshot_data_len: 0,
                    snapshot_array_data_len: 0,
                }
            }
            meta.push_portal_trace_position(
                idx1,
                Some((callee_hash, Some(callee.clone()))),
                pos(0),
            );
            meta.push_portal_trace_position(idx1, None, pos(10));
            assert_eq!(
                meta.find_biggest_function(),
                Some((idx1, (callee_hash, Some(callee.clone())))),
                "find_biggest_function names the inlined callee, not the portal"
            );
        }

        note_root_trace_too_long(Some((portal_hash, Some(portal.clone()))), None);

        {
            let meta = pair.0.meta_interp_mut();
            let callee_cell = meta
                .warm_state_for_driver(idx1)
                .lookup_chain_with_key(&callee)
                .expect("disable_noninlinable_function_for_key installs the callee cell");
            assert!(
                callee_cell
                    .flags
                    .get()
                    .contains(JcFlags::JC_DONT_TRACE_HERE),
                "the shipped door must disable the oversized inlined frame"
            );
            let portal_cell = meta
                .warm_state_for_driver(idx1)
                .lookup_chain_with_key(&portal)
                .expect("bound_reached installed the portal cell");
            assert!(
                !portal_cell
                    .flags
                    .get()
                    .contains(JcFlags::JC_DONT_TRACE_HERE),
                "the portal cell must not receive JC_DONT_TRACE_HERE"
            );
            assert!(
                !portal_cell.flags.get().contains(JcFlags::JC_FORCE_FINISH),
                "prepare_trace_segmenting must not run when an inlined frame caused the bloat"
            );
            assert!(
                meta.warm_state_for_driver(PyreJitState::PYPYJIT_JD_INDEX)
                    .can_inline_callable(callee_hash),
                "portal 0 stays inlinable"
            );
            meta.abort_trace_live(false);
        }
    }

    /// `names()`-shaped FBW abort: the portal is already the compiling
    /// driver-0 cell and the inlined callee is the biggest function.
    /// `note_root_trace_too_long` reads `portal_trace_positions` via
    /// `find_biggest_function`; `maybe_compile_and_run` must still enter
    /// the portal cell.
    #[test]
    fn note_root_trace_too_long_names_shaped_abort_keeps_portal_compilable() {
        use majit_ir::{GreenKey, Type};
        use majit_metainterp::BackEdgeAction;

        let pair = super::ensure_trace_test_driver();
        extern "C" fn portal_runner_helper() -> i64 {
            0
        }
        let portal = GreenKey::with_types(vec![7, 0, 0xC0DE], [Type::Int, Type::Int, Type::Ref]);
        let names = GreenKey::with_types(vec![0, 0, 0x4E41], [Type::Int, Type::Int, Type::Ref]);
        let portal_hash = portal.get_uhash();
        let names_hash = names.get_uhash();
        assert_ne!(
            portal_hash, names_hash,
            "fixture: portal and names() must not share a cell key"
        );
        {
            let meta = pair.0.meta_interp_mut();
            let mut first = JitDriverStaticData::new(vec![], vec![]);
            first.portal_runner_adr = portal_runner_helper as *const () as i64;
            first.index = Some(0);
            first.is_recursive = true;
            let idx0 = meta.register_jitdriver_sd(first);
            assert_eq!(idx0, 0);
            meta.finish_setup_descrs_for_jitdrivers();

            let action =
                meta.bound_reached(portal_hash, (0xC0DE, 7), Some(portal.clone()), None, &[]);
            assert!(matches!(action, BackEdgeAction::StartedTracing));
            assert_eq!(meta.active_jitdriver_sd, Some(idx0));

            let token = std::sync::Arc::new(majit_backend::JitCellToken::new(11));
            token.set_compiled(Box::new(()));
            meta.warm_state_for_driver(idx0)
                .memory_manager
                .keep_loop_alive(&token);
            meta.warm_state_for_driver(idx0)
                .attach_procedure_to_interp_for_key(&portal, std::sync::Arc::clone(&token));

            fn pos(n: usize) -> majit_metainterp::recorder::TracePosition {
                majit_metainterp::recorder::TracePosition {
                    _pos: n,
                    _count: 0,
                    _index: 0,
                    snapshot_data_len: 0,
                    snapshot_array_data_len: 0,
                }
            }
            meta.push_portal_trace_position(idx0, Some((names_hash, Some(names.clone()))), pos(0));
            meta.push_portal_trace_position(idx0, None, pos(10));
            assert_eq!(
                meta.find_biggest_function(),
                Some((idx0, (names_hash, Some(names.clone())))),
                "find_biggest_function names the inlined names() frame"
            );
        }

        note_root_trace_too_long(Some((portal_hash, Some(portal.clone()))), None);

        {
            let meta = pair.0.meta_interp_mut();
            meta.abort_trace_live(false);
            let (step, _) = meta.maybe_compile_and_run_step(portal_hash, (0xC0DE, 7));
            assert!(
                !matches!(step, majit_metainterp::warmstate::HotResult::NotHot),
                "maybe_compile_and_run still finds the already-compiled portal cell"
            );
        }
    }

    /// Shipped FBW abort: `note_inline_subwalk_start` is `newframe`, then
    /// `note_root_trace_too_long` is `blackhole_if_trace_too_long`. The log
    /// must name `names()` so the portal cell is not `JC_DONT_TRACE_HERE`.
    #[test]
    fn note_inline_subwalk_start_then_too_long_keeps_portal_compilable() {
        use majit_ir::{GreenKey, Type};
        use majit_metainterp::BackEdgeAction;
        use majit_metainterp::warmstate::JcFlags;

        let pair = super::ensure_trace_test_driver();
        extern "C" fn portal_runner_helper() -> i64 {
            0
        }
        let portal = GreenKey::with_types(vec![7, 0, 0xC0DE], [Type::Int, Type::Int, Type::Ref]);
        let names = GreenKey::with_types(vec![0, 0, 0x4E41], [Type::Int, Type::Int, Type::Ref]);
        let portal_hash = portal.get_uhash();
        let names_hash = names.get_uhash();
        assert_ne!(
            portal_hash, names_hash,
            "fixture: portal and names() must not share a cell key"
        );
        {
            let meta = pair.0.meta_interp_mut();
            let mut first = JitDriverStaticData::new(vec![], vec![]);
            first.portal_runner_adr = portal_runner_helper as *const () as i64;
            first.index = Some(0);
            first.is_recursive = true;
            let idx0 = meta.register_jitdriver_sd(first);
            assert_eq!(idx0, 0);
            meta.finish_setup_descrs_for_jitdrivers();

            let action =
                meta.bound_reached(portal_hash, (0xC0DE, 7), Some(portal.clone()), None, &[]);
            assert!(matches!(action, BackEdgeAction::StartedTracing));
            assert_eq!(meta.active_jitdriver_sd, Some(idx0));
            assert!(
                meta.portal_trace_positions.is_some(),
                "bound_reached arms portal_trace_positions"
            );

            let token = std::sync::Arc::new(majit_backend::JitCellToken::new(11));
            token.set_compiled(Box::new(()));
            meta.warm_state_for_driver(idx0)
                .memory_manager
                .keep_loop_alive(&token);
            meta.warm_state_for_driver(idx0)
                .attach_procedure_to_interp_for_key(&portal, std::sync::Arc::clone(&token));
        }
        let started = note_inline_subwalk_start((names_hash, Some(names.clone())), None);
        assert!(
            started.is_some(),
            "note_inline_subwalk_start logs names() through the portal jitcode"
        );
        {
            let meta = pair.0.meta_interp_mut();
            // pyjitpl.py `find_biggest_function`: an open frame is sized
            // `self.history.length() - pos`, so the inlined body records
            // one op after `newframe` logged its start.
            meta.trace_ctx()
                .expect("bound_reached opened the history")
                .record_same_as(majit_ir::OpRef::const_int(1), Type::Int);
            assert_eq!(
                meta.find_biggest_function(),
                Some((0, (names_hash, Some(names.clone())))),
                "shipped newframe log names the inlined names() frame"
            );
        }

        note_root_trace_too_long(Some((portal_hash, Some(portal.clone()))), None);

        {
            let meta = pair.0.meta_interp_mut();
            let names_cell = meta
                .warm_state_for_driver(0)
                .lookup_chain_with_key(&names)
                .expect("disable_noninlinable_function_for_key installs the names() cell");
            assert!(
                names_cell.flags.get().contains(JcFlags::JC_DONT_TRACE_HERE),
                "the shipped abort disables the oversized inlined names() frame"
            );
            let portal_cell = meta
                .warm_state_for_driver(0)
                .lookup_chain_with_key(&portal)
                .expect("bound_reached installed the portal cell");
            assert!(
                !portal_cell
                    .flags
                    .get()
                    .contains(JcFlags::JC_DONT_TRACE_HERE),
                "the portal cell must not receive JC_DONT_TRACE_HERE"
            );
            assert!(
                !portal_cell.flags.get().contains(JcFlags::JC_FORCE_FINISH),
                "prepare_trace_segmenting must not run when names() caused the bloat"
            );
            let (step, _) = meta.maybe_compile_and_run_step(portal_hash, (0xC0DE, 7));
            assert!(
                !matches!(step, majit_metainterp::warmstate::HotResult::NotHot),
                "maybe_compile_and_run still finds the already-compiled portal cell"
            );
        }
    }

    #[test]
    fn callback_table_is_visible_to_new_threads() {
        ensure_test_callbacks();
        assert!(
            std::thread::spawn(|| crate::callbacks::try_get().is_some())
                .join()
                .unwrap(),
            "process-global MetaInterpStaticData callbacks disappeared on a new thread"
        );
    }

    /// Install the real `hash_w` and `hash_str` on this test thread so
    /// object/str-keyed dicts built while running interpreted Python bucket
    /// through the single hash path (`baseobjspace.py is_none`).  These tests
    /// don't go through `init_jit_hooks`, which installs the hooks at boot for
    /// the pyrex binary; the thread-local cell must be set on the test thread.
    #[allow(dead_code)]
    fn install_test_hash_hook() {
        unsafe fn test_hash_w(obj: pyre_object::PyObjectRef) -> i64 {
            match pyre_interpreter::builtins::try_hash_value(obj) {
                Ok(h) => h,
                Err(e) => {
                    pyre_interpreter::baseobjspace::set_pending_dict_hash_error(e);
                    pyre_object::dict_eq_hook::signal_hash_error(obj);
                    0
                }
            }
        }
        unsafe fn test_hash_str(ptr: *const u8, len: usize) -> i64 {
            pyre_interpreter::builtins::hash_str_bytes(unsafe {
                std::slice::from_raw_parts(ptr, len)
            })
        }
        pyre_object::dict_eq_hook::register_hash_w_hook(test_hash_w);
        pyre_object::dict_eq_hook::register_hash_str_hook(test_hash_str);
    }

    /// Install a populated `PyJitCode` for `code_ref` so a subsequent
    /// `jitcode_for(code_ref)` returns a valid SD entry. The trace-side
    /// `MetaInterpStaticData.jitcodes` list is now populated only by
    /// `CodeWriter.make_jitcodes()` (warmspot.py:281-282); tests that
    /// exercise tracer logic directly must still arrive at that
    /// post-`make_jitcodes` state, so we install a minimal assembled
    /// jitcode with a `live/` anchor at offset 0 — every can-raise
    /// CALL_* in the tracer now emits an inline `GuardNoException`
    /// (pyjitpl.py) whose resumedata snapshot routes through
    /// `get_list_of_active_boxes` → `JitCode::get_live_vars_info`
    /// (jitcode.py), so the test fixture must satisfy that
    /// liveness lookup.
    fn install_test_jitcode(code: &CodeObject, code_ref: *const ()) {
        install_test_jitcode_with_liveness(code, code_ref, &[0, 0, 0], 1);
    }

    /// Register a skeleton jitcode for `code_ref` in `METAINTERP_SD` (keyed
    /// on the PyCode wrapper), publishing `all_liveness` so the
    /// resume/blackhole decoder (`restore_guard_failure_values`) reads the
    /// caller's `[length_i, length_r, length_f, <colors…>]` buffer at offset
    /// 0 instead of the empty `[0, 0, 0]` default. The single `live/` op the
    /// builder emits patches to liveness offset 0.
    fn install_test_jitcode_with_liveness(
        code: &CodeObject,
        code_ref: *const (),
        all_liveness: &[u8],
        num_liveness_ops: usize,
    ) {
        let raw_code = unsafe {
            pyre_interpreter::w_code_get_ptr(code_ref as pyre_object::PyObjectRef)
                as *const CodeObject
        };
        let mut builder = majit_metainterp::JitCodeBuilder::default();
        let live_patch = builder.live_placeholder();
        builder.patch_live_offset(live_patch, 0);
        let mut insns = indexmap::IndexMap::new();
        insns.insert(
            "live/".to_string(),
            majit_metainterp::jitcode::insns::BC_LIVE,
        );
        crate::assembler::publish_state(&insns, all_liveness, all_liveness.len(), num_liveness_ops);
        let mut pyjit = crate::PyJitCode::skeleton(raw_code);
        pyjit.jitcode = std::sync::Arc::new(builder.finish());
        pyjit.metadata.is_drained = true;
        METAINTERP_SD.with(|r| {
            r.borrow_mut()
                .set_jitcodes_from_make_result(vec![std::sync::Arc::new(pyjit)]);
        });
    }

    /// Bind `sym.jitcode` to a populated `JitCode` whose `raw_code()`
    /// resolves to a real `PyCode` (derived from `w_code`), so the
    /// methods that hard-deref `(*sym.jitcode).raw_code()` — notably the
    /// `close_loop_args_at` snapshot encoder (trace_opcode.rs) — do not
    /// null-deref. Mirrors the
    /// `get_list_of_active_boxes_reads_kind_specific_register_banks`
    /// harness (trace_opcode.rs) but with a non-null `JitCode.code`.
    ///
    /// `w_code` must be a real `PyCode` wrapper (e.g. `frame.pycode`,
    /// or `w_code_new(Box::into_raw(Box::new(code.clone())) as *const ())`
    /// for the no-frame case). `live_{i,r,f}` are the live color indices
    /// per kind bank; they seed the published `BC_LIVE` buffer so the
    /// snapshot/active-box readers find the expected registers.
    ///
    /// Returns the leaked `JitCode` pointer; the caller must reclaim it via
    /// `unsafe { Box::from_raw(ptr) }` after the test to avoid the leak.
    fn bind_real_jitcode(
        sym: &mut PyreSym,
        w_code: *const (),
        live_i: &[u8],
        live_r: &[u8],
        live_f: &[u8],
    ) -> *mut JitCode {
        use majit_jitcode::liveness::encode_liveness;
        let raw_code = unsafe {
            pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
                as *const CodeObject
        };
        let mut all_liveness = vec![live_i.len() as u8, live_r.len() as u8, live_f.len() as u8];
        all_liveness.extend(encode_liveness(live_i));
        all_liveness.extend(encode_liveness(live_r));
        all_liveness.extend(encode_liveness(live_f));
        let mut insns: indexmap::IndexMap<String, u8> = indexmap::IndexMap::new();
        insns.insert(
            "live/".to_string(),
            majit_metainterp::jitcode::insns::BC_LIVE,
        );
        crate::assembler::publish_state(&insns, &all_liveness, all_liveness.len(), 1);

        let num_regs = |live: &[u8]| {
            live.iter()
                .copied()
                .max()
                .map_or(4u8, |m| m.saturating_add(1).max(4))
        };
        let runtime_jc = {
            let inner = majit_metainterp::jitcode::JitCode::new("close_loop_args_test");
            inner.set_body(majit_jitcode::jitcode::JitCodeBody {
                code: vec![majit_metainterp::jitcode::insns::BC_LIVE, 0, 0],
                c_num_regs_i: num_regs(live_i),
                c_num_regs_r: num_regs(live_r),
                c_num_regs_f: num_regs(live_f),
                startpoints: Some([0_usize].into_iter().collect()),
                ..Default::default()
            });
            inner
        };
        let mut pyjit = crate::PyJitCode::skeleton(raw_code);
        pyjit.jitcode = std::sync::Arc::new(runtime_jc);
        pyjit.metadata.is_drained = true;
        let inner_jc = JitCode {
            index: 0,
            payload: std::sync::Arc::new(pyjit),
        };
        let inner_jc_ptr = Box::into_raw(Box::new(inner_jc));
        sym.jitcode = inner_jc_ptr;
        inner_jc_ptr
    }

    #[test]
    fn semantic_ref_slot_prefers_live_stack_color_reuse() {
        // Color 0 is owned by both local slot 0 and live operand-stack slot 0
        // (abs slot nlocals+0 = 2). With the stack slot in the live window
        // (stack_only=1) the inverse prefers the stack slot.
        assert_eq!(
            semantic_ref_slot_for_reg_color(2, 1, &[(1, 0, 0), (1, 0, 2), (1, 1, 1)], 0),
            Some(2),
        );
    }

    #[test]
    fn semantic_ref_slots_fan_out_a_copied_live_stack_color() {
        // JUMP_IF_FALSE_OR_POP/COPY can keep the same MIFrame register in two
        // semantic stack positions.  A slot-indexed reconstructed frame must
        // restore both positions from that one resume-stream value.
        assert_eq!(
            semantic_ref_slots_for_reg_color(
                2,
                2,
                &[(1, 0, 0), (1, 4, 2), (1, 4, 3), (1, 5, 4)],
                4,
            ),
            vec![2, 3],
        );
    }

    #[test]
    fn semantic_ref_slot_falls_back_to_local_color_map() {
        // Color 1 is owned only by local slot 1 (the sole live operand-stack
        // slot carries color 3), so the inverse falls through to the local.
        assert_eq!(
            semantic_ref_slot_for_reg_color(2, 1, &[(1, 1, 1), (1, 3, 2), (1, 4, 0)], 1),
            Some(1),
        );
    }

    #[test]
    fn semantic_ref_slot_ignores_dead_local_color_reuse() {
        // A dead local is simply absent from the per-PC entries (they record
        // only live slots), so color 0 has no live owner here -> None.
        assert_eq!(semantic_ref_slot_for_reg_color(2, 0, &[(1, 1, 1)], 0), None,);
    }

    #[test]
    fn semantic_ref_slot_none_for_beyond_window_stack_color() {
        // At BUILD_LIST entry (pc=40) the runtime stack is 3 deep
        // (stack_only=3), but the compile-time entries also carry color 5 at
        // operand-stack slot 3 (abs slot nlocals+3 = 5), live only at the
        // depth-4 sibling PC.  The inverse must classify color 5 as dead-here
        // (None) — its stack slot sits past the live window (3 >= stack_only)
        // — so `collect_outer_active_boxes` substitutes a CONST_NULL
        // placeholder rather than reading an unpopulated register.
        let pcdep = [
            (1u8, 2u16, 0u16),
            (1, 2, 1),
            (1, 2, 2),
            (1, 3, 3),
            (1, 4, 4),
            (1, 5, 5),
        ];
        assert_eq!(semantic_ref_slot_for_reg_color(2, 3, &pcdep, 5), None,);
        assert!(pcdep.iter().any(|&(_, c, _)| c == 5));
    }

    /// Encode<->decode resume-symmetry round trip over a NON-IDENTITY,
    /// coalesced color map shaped exactly as the codewriter publishes it as
    /// per-PC `(color, slot)` entries. The forward map assigns one
    /// post-regalloc Ref color per semantic slot; chordal coalescing
    /// legitimately reuses a color across slots that are never simultaneously
    /// live, so the shared inverse `semantic_ref_slot_for_reg_color` — called
    /// by the decode side (restore_guard_failure_values) — must disambiguate
    /// by the live window: the live stack prefix first, then the live locals.
    ///
    /// Layout: nlocals=2, max_stackdepth=3, live stack depth 2 (stack_only=2).
    ///   local 0 -> color 5, local 1 -> color 6
    ///   stack 0 -> color 6 (shared with live local 1)
    ///   stack 1 -> color 7
    ///   stack 2 -> color 5 (shared with live local 0; DEAD, index >= stack_only)
    ///
    /// Expected inverses are DERIVED from the published entries, so this
    /// asserts a true publish<->inverse identity rather than ad-hoc literals.
    /// The per-PC entries carry every live local plus the full compile-time
    /// stack depth; the inverse clamps out-of-window stack slots by stack_only.
    #[test]
    fn resume_symmetry_roundtrip_coalesced_color_map() {
        let nlocals = 2usize;
        // live stack depth = valuestackdepth - nlocals; the published stack map
        // is the FULL max_stackdepth (3), but only the live prefix is in window.
        let stack_only = 2usize;
        let local_map = [5u16, 6u16];
        let stack_map = [6u16, 7u16, 5u16];

        // Build the per-PC (color, slot) entries the codewriter publishes:
        // one entry per live local (slot = local index) plus one per
        // compile-time operand-stack slot (slot = nlocals + depth). Sorted by
        // (color, slot) so locals precede stack within a shared color.
        let mut pcdep: Vec<(u8, u16, u16)> = Vec::new();
        for (i, &c) in local_map.iter().enumerate() {
            pcdep.push((1, c, i as u16));
        }
        for (d, &c) in stack_map.iter().enumerate() {
            pcdep.push((1, c, (nlocals + d) as u16));
        }
        pcdep.sort();

        let invert =
            |reg: u16| semantic_ref_slot_for_reg_color(nlocals, stack_only, &pcdep, reg as usize);

        // Round-trip closure: every LIVE stack slot's published color inverts
        // back to its own semantic slot (nlocals + d) — the stack map is its
        // own inverse over the live prefix.
        for d in 0..stack_only {
            assert_eq!(invert(stack_map[d]), Some(nlocals + d));
        }

        // Color 6 is shared by live local 1 and live stack slot 0. The decoder
        // scans the live stack prefix first, so it MUST resolve to the stack
        // slot (Some(2)), never the local (Some(1)). This is the kept-stack /
        // local coalescing case the guard-failure deopt depends on.
        assert_eq!(invert(stack_map[0]), Some(2));
        assert_eq!(invert(local_map[1]), Some(2));

        // Color 5 is shared by live local 0 and the DEAD stack slot 2 (index
        // >= stack_only, outside the live window). It must route to the live
        // local, not the dead stack slot.
        assert_eq!(invert(local_map[0]), Some(0));

        // The dead stack slot's color must never recover as that slot.
        assert_ne!(invert(stack_map[2]), Some(nlocals + 2));
    }

    fn empty_meta() -> PyreMeta {
        PyreMeta {
            num_locals: 0,
            ns_len: 0,
            namespace_dependent: false,
            valuestackdepth: 0,
            array_capacity: 0,
            trace_extra_reds: 0,
            slot_types: Vec::new(),
            has_virtualizable: false,
        }
    }

    /// `resume.py` `rebuild_from_resumedata` returns the live box lists when
    /// `done_reading` is already true, so an empty frame section is a result.
    #[test]
    fn rebuild_from_resumedata_keeps_an_empty_frame_section() {
        let mut writer = majit_ir::resumecode::Writer::new(4);
        writer.append_int(0);
        writer.append_int(0);
        writer.append_int(0);
        writer.append_int(0);
        writer.patch_current_size(0);
        let storage = majit_metainterp::resume::ResumeStorage::new(
            writer.create_numbering(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );
        let mut meta = empty_meta();
        let result = PyreJitState::rebuild_from_resumedata(&mut meta, &[], Some(&storage))
            .expect("empty frame section still returns resume data");
        assert!(result.frames.is_empty());
        assert!(result.virtualizable_boxes.is_empty());
        assert!(result.virtualref_values.is_empty());
        assert_eq!(result.num_failargs, 0);
        assert_eq!(meta.trace_extra_reds, 1);
    }

    fn empty_state() -> PyreJitState {
        PyreJitState {
            frame: 0,
            execution_context: 0,
        }
    }

    fn compile_function_body(src: &str) -> CodeObject {
        let module = compile_source(src, Mode::Exec).expect("compile should succeed");
        module
            .constants
            .iter()
            .find_map(|c| match c {
                ConstantData::Code { code } => Some((**code).clone()),
                _ => None,
            })
            .expect("source should contain a function body")
    }

    #[test]
    fn trace_too_long_publishes_forwarded_for_iter_stack_and_coordinate() {
        use pyre_interpreter::pyframe::PyFrame;

        let code = compile_function_body("def f(xs):\n    for x in xs:\n        pass\n");
        let mut live = PyFrame::new(code);
        let mut snapshot = live.snapshot_for_tracing();
        let stack_base = snapshot.stack_base();
        let iterator = pyre_object::w_none();
        let item = pyre_object::w_bool_from(true);
        snapshot.push(iterator);
        snapshot.push(item);
        snapshot.last_instr = 37;

        let snapshot_addr = (&mut *snapshot) as *mut PyFrame as usize;
        let live_addr = (&mut *live) as *mut PyFrame as usize;
        assert_eq!(live.valuestackdepth, stack_base);
        assert!(can_publish_frame_stack(snapshot_addr, live_addr));

        let mut captured = capture_frame_stack_for_publish(snapshot_addr, live_addr)
            .expect("matching snapshot/live frame shapes must be capturable");
        assert_eq!(captured.stack_base, stack_base);
        assert_eq!(captured.scalars.valuestackdepth, stack_base + 2);
        assert_eq!(captured.scalars.last_instr, 37);
        assert_eq!(captured.roots, vec![iterator as i64, item as i64]);

        // The adoption path roots this slice while publishing locals. Model a
        // moving collection forwarding both active FOR_ITER stack values and
        // prove that publication reads the updated roots, not the stale raw
        // words still held by the detached tracing snapshot.
        let forwarded_iterator = pyre_object::w_bool_from(false);
        let forwarded_item = pyre_object::w_none();
        captured.roots_mut()[0] = forwarded_iterator as i64;
        captured.roots_mut()[1] = forwarded_item as i64;

        assert!(publish_captured_frame_stack(live_addr, &captured));
        assert_eq!(live.valuestackdepth, stack_base + 2);
        assert_eq!(live.last_instr, 37);
        assert_eq!(locals_w!(live)[stack_base], forwarded_iterator);
        assert_eq!(locals_w!(live)[stack_base + 1], forwarded_item);
        assert_eq!(locals_w!(snapshot)[stack_base], iterator);
        assert_eq!(locals_w!(snapshot)[stack_base + 1], item);
    }

    fn contains_instruction(code: &CodeObject, predicate: impl Fn(Instruction) -> bool) -> bool {
        (0..code.instructions.len()).any(|pc| {
            decode_instruction_at(code, pc)
                .map(|(instruction, _)| predicate(instruction))
                .unwrap_or(false)
        })
    }

    #[test]
    fn is_compatible_ignores_namespace_length_when_not_namespace_dependent() {
        // A pure-compute / builtin-only trace does not fold a module global, so
        // its compiled code is independent of the globals dict: a later
        // top-level bind (namespace growth) must not refuse re-entry.
        let state = empty_state();
        let mut meta = empty_meta();
        meta.num_locals = state.local_count();
        meta.namespace_dependent = false;
        meta.ns_len = state.namespace_len().unwrap_or(0) + 5;
        assert!(<PyreJitState as JitState>::is_compatible(&state, &meta));
    }

    #[test]
    fn is_compatible_enforces_namespace_length_when_namespace_dependent() {
        // A trace that read a module global keeps the conservative length gate,
        // since same-key value rebinds are not value-guarded: a namespace-length
        // mismatch must refuse re-entry.
        let state = empty_state();
        let mut meta = empty_meta();
        meta.num_locals = state.local_count();
        meta.namespace_dependent = true;
        meta.ns_len = state.namespace_len().unwrap_or(0) + 5;
        assert!(!<PyreJitState as JitState>::is_compatible(&state, &meta));
    }

    #[test]
    fn test_pre_opcode_snapshot_gate_skips_peek_only_and_no_guard_opcodes() {
        let iter_code =
            compile_function_body("def f(xs):\n    for x in xs:\n        return len(xs)\n");
        assert!(contains_instruction(&iter_code, |instruction| {
            matches!(instruction, Instruction::GetIter)
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));
        assert!(contains_instruction(&iter_code, |instruction| {
            matches!(instruction, Instruction::ForIter { .. })
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let match_code = compile_function_body(
            "def f(xs):\n    match xs:\n        case [a, b]:\n            return 1\n    return 0\n",
        );
        assert!(contains_instruction(&match_code, |instruction| {
            matches!(instruction, Instruction::GetLen)
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let import_code =
            compile_function_body("def f(pkg):\n    from pkg import name\n    return name\n");
        assert!(contains_instruction(&import_code, |instruction| {
            matches!(instruction, Instruction::ImportFrom { .. })
                && instruction_may_raise(instruction)
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let except_code = compile_function_body(
            "def f():\n    try:\n        raise ValueError('x')\n    except ValueError:\n        return 0\n",
        );
        assert!(contains_instruction(&except_code, |instruction| {
            matches!(instruction, Instruction::CheckExcMatch)
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let none_branch_code = compile_function_body(
            "def f(x):\n    if x is not None:\n        return 1\n    return 0\n",
        );
        assert!(contains_instruction(&none_branch_code, |instruction| {
            matches!(instruction, Instruction::PopJumpIfNone { .. })
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let not_none_branch_code =
            compile_function_body("def f(x):\n    if x is None:\n        return 1\n    return 0\n");
        assert!(contains_instruction(&not_none_branch_code, |instruction| {
            matches!(instruction, Instruction::PopJumpIfNotNone { .. })
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let truth_branch_code =
            compile_function_body("def f(x):\n    if x:\n        return 1\n    return 0\n");
        assert!(contains_instruction(&truth_branch_code, |instruction| {
            matches!(instruction, Instruction::PopJumpIfFalse { .. })
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let not_truth_branch_code =
            compile_function_body("def f(x):\n    if not x:\n        return 1\n    return 0\n");
        assert!(contains_instruction(
            &not_truth_branch_code,
            |instruction| {
                matches!(instruction, Instruction::PopJumpIfTrue { .. })
                    && !instruction_needs_pre_opcode_snapshot(instruction)
            }
        ));

        let contains_code = compile_function_body("def f(a, b):\n    return a in b\n");
        assert!(contains_instruction(&contains_code, |instruction| {
            matches!(instruction, Instruction::ContainsOp { .. })
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let with_except_code = compile_function_body(
            "def f(cm):\n    try:\n        with cm:\n            return 1\n    except Exception:\n        return 0\n",
        );
        assert!(contains_instruction(&with_except_code, |instruction| {
            matches!(instruction, Instruction::WithExceptStart)
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let import_name_code =
            compile_function_body("def f():\n    import math\n    return math\n");
        assert!(contains_instruction(&import_name_code, |instruction| {
            matches!(instruction, Instruction::ImportName { .. })
                && !instruction_may_raise(instruction)
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let call_code = compile_function_body("def f(g, x):\n    return g(x)\n");
        let call_kw_instruction = (0..call_code.instructions.len())
            .find_map(|pc| match decode_instruction_at(&call_code, pc) {
                Some((Instruction::Call { argc }, _)) => Some(Instruction::CallKw { argc }),
                _ => None,
            })
            .expect("source should contain a Call instruction to reuse argc shape");
        assert!(instruction_may_raise(call_kw_instruction));
        assert!(instruction_needs_pre_opcode_snapshot(call_kw_instruction));

        assert!(!instruction_may_raise(Instruction::CallFunctionEx));
        assert!(!instruction_needs_pre_opcode_snapshot(
            Instruction::CallFunctionEx
        ));

        let delete_attr_code = compile_function_body("def f(obj):\n    del obj.x\n");
        assert!(contains_instruction(&delete_attr_code, |instruction| {
            matches!(instruction, Instruction::DeleteAttr { .. })
                && !instruction_may_raise(instruction)
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let delete_subscr_code = compile_function_body("def f(a, i):\n    del a[i]\n");
        assert!(contains_instruction(&delete_subscr_code, |instruction| {
            matches!(instruction, Instruction::DeleteSubscr)
                && !instruction_may_raise(instruction)
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let store_attr_code = compile_function_body("def f(obj, v):\n    obj.x = v\n");
        assert!(contains_instruction(&store_attr_code, |instruction| {
            matches!(instruction, Instruction::StoreAttr { .. })
                && instruction_may_raise(instruction)
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let dict_update_code = compile_function_body("def f(a, b):\n    return {**a, **b}\n");
        assert!(contains_instruction(&dict_update_code, |instruction| {
            matches!(instruction, Instruction::DictUpdate { .. })
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let binary_slice_code = compile_function_body("def f(a, b, c):\n    return a[b:c]\n");
        assert!(contains_instruction(&binary_slice_code, |instruction| {
            matches!(instruction, Instruction::BinarySlice)
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let store_slice_code =
            compile_function_body("def f(a, b, c, v):\n    a[b:c] = v\n    return a\n");
        assert!(contains_instruction(&store_slice_code, |instruction| {
            matches!(instruction, Instruction::StoreSlice)
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let list_extend_code = compile_function_body("def f(a, b):\n    return [*a, *b]\n");
        assert!(contains_instruction(&list_extend_code, |instruction| {
            matches!(instruction, Instruction::ListExtend { .. })
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let set_update_code = compile_function_body("def f(a, b):\n    return {*a, *b}\n");
        assert!(contains_instruction(&set_update_code, |instruction| {
            matches!(instruction, Instruction::SetUpdate { .. })
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let map_add_code = compile_function_body("def f(xs):\n    return {x: x for x in xs}\n");
        assert!(contains_instruction(&map_add_code, |instruction| {
            matches!(instruction, Instruction::MapAdd { .. })
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let set_add_code = compile_function_body("def f(xs):\n    return {x for x in xs}\n");
        assert!(contains_instruction(&set_add_code, |instruction| {
            matches!(instruction, Instruction::SetAdd { .. })
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let reraise_code = compile_function_body(
            "def f():\n    try:\n        raise ValueError(1)\n    except Exception:\n        raise\n",
        );
        assert!(contains_instruction(&reraise_code, |instruction| {
            matches!(instruction, Instruction::Reraise { .. })
                && !instruction_may_raise(instruction)
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));
    }

    #[test]
    fn test_pre_opcode_snapshot_gate_keeps_pop_before_guard_opcodes() {
        let call_code = compile_function_body("def f(g, x):\n    return g(x)\n");
        assert!(contains_instruction(&call_code, |instruction| {
            matches!(instruction, Instruction::Call { .. })
                && instruction_may_raise(instruction)
                && instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let unary_code = compile_function_body("def f(x):\n    return -x\n");
        assert!(contains_instruction(&unary_code, |instruction| {
            matches!(instruction, Instruction::UnaryNegative)
                && instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let store_subscr_code =
            compile_function_body("def f(a, i, v):\n    a[i] = v\n    return a\n");
        assert!(contains_instruction(&store_subscr_code, |instruction| {
            matches!(instruction, Instruction::StoreSubscr)
                && instruction_may_raise(instruction)
                && instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let raise_code = compile_function_body(
            "def f():\n    try:\n        raise ValueError(1)\n    except Exception:\n        raise\n",
        );
        assert!(contains_instruction(&raise_code, |instruction| {
            matches!(instruction, Instruction::RaiseVarargs { .. })
                && !instruction_may_raise(instruction)
                && instruction_needs_pre_opcode_snapshot(instruction)
        }));

        // BUILD_TUPLE pops its operands before the specialised tuple path
        // emits the w_class guards.
        let build_tuple_code = compile_function_body("def f(a, b):\n    return (a, b)\n");
        assert!(contains_instruction(&build_tuple_code, |instruction| {
            matches!(instruction, Instruction::BuildTuple { .. })
                && instruction_needs_pre_opcode_snapshot(instruction)
        }));

        // UNPACK_SEQUENCE pops the sequence before the unpack path emits the
        // sequence class / length guards.
        let unpack_code = compile_function_body("def f(s):\n    a, b = s\n    return a + b\n");
        assert!(contains_instruction(&unpack_code, |instruction| {
            matches!(instruction, Instruction::UnpackSequence { .. })
                && instruction_needs_pre_opcode_snapshot(instruction)
        }));

        // LOAD_ATTR's foldable list-method form (lst.append) reaches the trait
        // leg load_method, which pop_value's the receiver before emitting the
        // receiver class / version_tag guards.
        let load_attr_code = compile_function_body("def f(lst, x):\n    return lst.append(x)\n");
        assert!(contains_instruction(&load_attr_code, |instruction| {
            matches!(instruction, Instruction::LoadAttr { .. })
                && instruction_needs_pre_opcode_snapshot(instruction)
        }));

        // LIST_APPEND (list comprehension) pops the appended value before the
        // strategy fast path emits its list class / strategy / unbox guards.
        let list_append_code = compile_function_body("def f(xs):\n    return [x for x in xs]\n");
        assert!(contains_instruction(&list_append_code, |instruction| {
            matches!(instruction, Instruction::ListAppend { .. })
                && instruction_needs_pre_opcode_snapshot(instruction)
        }));
    }

    #[test]
    fn test_pypyjit_driver_descriptor_matches_interp_jit_layout() {
        let descriptor = PyreJitState::pypyjit_driver_descriptor();

        assert!(descriptor.is_recursive);
        assert_eq!(descriptor.num_greens(), 3);
        assert_eq!(descriptor.num_reds(), 2);
        assert_eq!(descriptor.virtualizable.as_deref(), Some("frame"));
        assert_eq!(descriptor.index_of_virtualizable, 0);

        let greens = descriptor.greens();
        assert_eq!(greens[0].name, "next_instr");
        assert_eq!(greens[0].tp, Type::Int);
        assert_eq!(greens[1].name, "is_being_profiled");
        assert_eq!(greens[1].tp, Type::Int);
        assert_eq!(greens[2].name, "pycode");
        assert_eq!(greens[2].tp, Type::Ref);

        let reds = descriptor.reds();
        assert_eq!(reds[0].name, "frame");
        assert_eq!(reds[0].tp, Type::Ref);
        assert_eq!(reds[1].name, "ec");
        assert_eq!(reds[1].tp, Type::Ref);
    }

    #[test]
    fn test_pypyjit_live_values_with_ec_insert_execution_context_after_frame() {
        use pyre_interpreter::pyframe::PyFrame;

        let code = compile_exec("x = 1").expect("test code should compile");
        let mut frame = PyFrame::new(code);
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut state = empty_state();
        state.frame = frame_ptr;
        state.execution_context = pyre_interpreter::call::getexecutioncontext() as usize;
        let meta = state.build_meta(0, &PyreEnv);
        let with_ec = state.extract_live_values(&meta);

        assert_eq!(with_ec.len(), 2);
        assert_eq!(with_ec[0], Value::Ref(majit_ir::GcRef(frame_ptr)));
        assert_eq!(
            with_ec[1],
            Value::Ref(majit_ir::GcRef(state.execution_context))
        );

        let mut reused = Vec::with_capacity(2);
        let allocation = reused.as_ptr();
        let mut raw = Vec::new();
        let mut types = Vec::new();
        state.extract_live_values_into(&meta, &mut reused, &mut raw, &mut types);
        assert_eq!(reused, with_ec);
        assert_eq!(reused.as_ptr(), allocation);
        assert!(raw.is_empty());
        assert!(types.is_empty());
    }

    #[test]
    fn test_pypyjit_live_value_types_with_ec_match_descriptor_red_prefix() {
        let meta = PyreMeta {
            num_locals: 2,
            ns_len: 0,
            namespace_dependent: false,
            valuestackdepth: 3,
            array_capacity: 4,
            trace_extra_reds: 1,
            has_virtualizable: true,
            slot_types: vec![Type::Ref; 4],
        };

        let descriptor = PyreJitState::pypyjit_driver_descriptor();
        let with_ec = <PyreJitState as JitState>::live_value_types(&empty_state(), &meta);

        assert_eq!(
            with_ec,
            vec![descriptor.reds()[0].tp, descriptor.reds()[1].tp]
        );

        let mut reused = Vec::with_capacity(2);
        let allocation = reused.as_ptr();
        <PyreJitState as JitState>::live_value_types_into(&empty_state(), &meta, &mut reused);
        assert_eq!(reused, with_ec);
        assert_eq!(reused.as_ptr(), allocation);
    }

    #[test]
    fn collect_jump_args_rejects_locals_beyond_register_storage() {
        let mut ctx = crate::trace_ctx_for_test(0);
        let mut sym = PyreSym::new_uninit(ctx.const_ref(0));
        sym.nlocals = 1;
        sym.valuestackdepth = 1;
        // Value-only iteration must retain the old slice's bounds check,
        // including when the stack window itself is empty.
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                sym.vable_collect_jump_args()
            }))
            .is_err()
        );
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                sym.vable_collect_typed_jump_args()
            }))
            .is_err()
        );
    }

    #[test]
    fn test_pypyjit_collect_jump_args_inserts_ec_after_frame() {
        let mut ctx = crate::trace_ctx_for_test(0);
        let frame_ref = ctx.const_ref(0x1000);
        let ec_ref = ctx.const_ref(0x7000);
        let code_ref = ctx.const_ref(0x2000);
        let namespace_ref = ctx.const_ref(0x3000);
        let local0 = ctx.const_ref(0x4000);
        let stack0 = ctx.const_ref(0x5000);
        let stack1 = ctx.const_ref(0x6000);

        let mut sym = PyreSym::new_uninit(frame_ref);
        sym.nlocals = 1;
        sym.valuestackdepth = 3;
        sym.vable_last_instr = ctx.const_int(33);
        sym.vable_pycode = code_ref;
        sym.vable_valuestackdepth = ctx.const_int(3);
        sym.vable_debugdata = ctx.const_ref(0);
        sym.frame_w_globals = namespace_ref;
        sym.execution_context = ec_ref;
        sym.registers_r.replace(vec![local0, stack0, stack1]);
        sym.symbolic_local_types = vec![Type::Ref];
        sym.symbolic_stack_types = vec![Type::Ref, Type::Ref];

        let base = sym.vable_collect_jump_args();
        let with_ec = PyreJitState::pypyjit_collect_jump_args(&sym);
        let typed_with_ec = PyreJitState::pypyjit_collect_typed_jump_args(&sym);

        assert_eq!(with_ec.len(), base.len() + 1);
        assert_eq!(with_ec[0], frame_ref);
        assert_eq!(with_ec[1], ec_ref);
        assert_eq!(&with_ec[2..], &base[1..]);
        assert_eq!(typed_with_ec[0], (frame_ref, Type::Ref));
        assert_eq!(typed_with_ec[1], (ec_ref, Type::Ref));
    }

    #[test]
    fn test_pypyjit_collect_jump_args_with_boxes_uses_live_vable_shadow() {
        let mut ctx = crate::trace_ctx_for_test(0);
        let frame_ref = ctx.const_ref(0x1000);
        let ec_ref = ctx.const_ref(0x2000);
        let stale_local = ctx.const_ref(0x3000);
        let live_last_instr = ctx.const_int(41);
        let live_pycode = ctx.const_ref(0x4000);
        let live_vsd = ctx.const_int(2);
        let live_debugdata = ctx.const_ref(0x5000);
        let live_local = ctx.const_ref(0x8000);
        let live_stack = ctx.const_ref(0x9000);

        let mut sym = PyreSym::new_uninit(frame_ref);
        sym.execution_context = ec_ref;
        sym.registers_r.replace(vec![stale_local]);

        // TraceCtx::collect_virtualizable_typed_boxes order:
        // static fields, array items, trailing standard-vable identity.
        let boxes = vec![
            (live_last_instr, Type::Int),
            (live_pycode, Type::Ref),
            (live_vsd, Type::Int),
            (live_debugdata, Type::Ref),
            (live_local, Type::Ref),
            (live_stack, Type::Ref),
            (frame_ref, Type::Ref),
        ];

        let args = <PyreJitState as JitState>::collect_jump_args_with_boxes(&sym, &boxes);

        assert_eq!(
            args,
            vec![
                frame_ref,
                ec_ref,
                live_last_instr,
                live_pycode,
                live_vsd,
                live_debugdata,
                live_local,
                live_stack,
            ]
        );
        assert!(!args.contains(&stale_local));
    }

    #[test]
    fn test_pypyjit_create_sym_shifts_virtualizable_indices_after_ec() {
        let meta = PyreMeta {
            num_locals: 2,
            ns_len: 0,
            namespace_dependent: false,
            valuestackdepth: 4,
            array_capacity: 5,
            trace_extra_reds: 1,
            has_virtualizable: true,
            slot_types: vec![Type::Ref; 5],
        };

        let sym = PyreJitState::pypyjit_create_sym(&meta, 0);

        // virtualizable derive `init_vable_indices` mints typed
        // `OpRef::input_arg_*` variants per `#[vable(inputarg, type = ...)]`
        // (`virtualizable_gen.rs`'s `inputargs`). The expected slot variants
        // must match those exact types so variant-aware OpRef Eq (`OpRef`'s
        // `PartialEq`) compares correctly against later optimizer/heap-cache keys.
        assert_eq!(sym.frame, OpRef::input_arg_ref(0));
        assert_eq!(sym.execution_context, OpRef::input_arg_ref(1));
        assert_eq!(sym.vable_last_instr, OpRef::input_arg_int(2));
        assert_eq!(sym.vable_pycode, OpRef::input_arg_ref(3));
        assert_eq!(sym.vable_valuestackdepth, OpRef::input_arg_int(4));
        assert_eq!(sym.vable_debugdata, OpRef::input_arg_ref(5));
        assert!(sym.frame_w_globals.is_none());
        assert_eq!(sym.vable_array_base, Some(6));
        assert_eq!(sym.symbolic_local_types.len(), 2);
        assert_eq!(sym.symbolic_stack_types.len(), 2);
    }

    #[test]
    fn test_restore_expanded_virtualizable_with_extra_reds_skips_ec_slot() {
        use majit_ir::GcRef;
        use pyre_interpreter::pyframe::PyFrame;
        use pyre_interpreter::{ConstantData, compile_exec};

        let module = compile_exec("def f(a, b, c):\n    i = 0\n    return i\nf(1, 2, 3)\n")
            .expect("test code should compile");
        let code = module
            .constants
            .iter()
            .find_map(|constant| match constant {
                ConstantData::Code { code } if code.obj_name.as_str() == "f" => {
                    Some((**code).clone())
                }
                _ => None,
            })
            .expect("test source should contain function code");

        let mut frame = PyFrame::new(code);
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let execution_context = pyre_interpreter::call::getexecutioncontext() as usize;
        let mut state = PyreJitState {
            frame: frame_ptr,
            execution_context,
        };
        state.set_next_instr(0);
        state.set_valuestackdepth(4);
        let meta = PyreMeta {
            num_locals: 4,
            ns_len: 0,
            namespace_dependent: false,
            valuestackdepth: 4,
            array_capacity: 4,
            trace_extra_reds: 1,
            has_virtualizable: true,
            slot_types: vec![Type::Ref, Type::Ref, Type::Ref, Type::Ref],
        };
        let values = vec![
            Value::Ref(GcRef(frame_ptr)),             // frame
            Value::Ref(GcRef(execution_context)),     // ec
            Value::Int(8),                            // last_instr
            Value::Ref(GcRef(frame.pycode as usize)), // pycode
            Value::Int(4),                            // valuestackdepth
            Value::Ref(GcRef(0)),                     // debugdata
            Value::Ref(GcRef(w_int_new(1) as usize)), // local a
            Value::Ref(GcRef(w_int_new(2) as usize)), // local b
            Value::Ref(GcRef(w_int_new(3) as usize)), // local c
            Value::Int(7),                            // local i
        ];

        state.restore_expanded_virtualizable_with_extra_reds(&meta, &values, 1);

        assert_eq!(state.valuestackdepth(), 4);
        let restored_i = state.local_at(3).expect("local i should be restored");
        assert!(unsafe { is_int(restored_i) });
        assert_eq!(unsafe { w_int_get_value(restored_i) }, 7);
    }

    #[test]
    #[ignore = "PyreSym::new_uninit hits the Phase X-1 skeleton-panic since the \
                debug-only fallback was removed; needs a populated-jitcode harness."]
    fn test_guard_class_uses_guard_nonnull_class() {
        let mut ctx = crate::trace_ctx_for_test(1);
        // registers_r[i] tracks locals_cells_stack_w[*] — W_Root array, Type::Ref.
        let obj = OpRef::input_arg_ref(0);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.registers_r.replace(vec![obj]);
        sym.symbolic_local_types = vec![Type::Ref];
        sym.nlocals = 1;

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            residual_call_pc: None,
            loop_close_marker_jit_pc: None,
            close_merge_point_vsd: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        state.with_ctx(|this, ctx| {
            this.guard_class(ctx, obj, &INT_TYPE as *const PyType);
        });

        let tree_loop = ctx.into_tree_loop();
        let op = tree_loop.ops.last().expect("guard op should be present");
        assert_eq!(op.opcode, OpCode::GuardClass);
        assert_eq!(op.arg(0).to_opref(), obj);
    }

    #[test]
    #[ignore = "PyreSym::new_uninit hits the Phase X-1 skeleton-panic since the \
                debug-only fallback was removed; needs a populated-jitcode harness."]
    fn test_trace_guarded_int_payload_uses_guard_nonnull_class_and_pure_payload() {
        // value_type is read from the recorder's inputarg type (Phase α/β: Box.type
        // intrinsic parity, history.py:220) so the inputarg must be Ref for
        // trace_guarded_int_payload to take the fast path rather than short-circuit.
        let mut ctx = TraceCtx::for_test_types(&[Type::Ref]);
        // value_type is Ref per the comment above; registers_r[0] = inputarg slot 0.
        let int_obj = OpRef::input_arg_ref(0);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.registers_r.replace(vec![int_obj]);
        sym.symbolic_local_types = vec![Type::Ref];
        sym.nlocals = 1;

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            residual_call_pc: None,
            loop_close_marker_jit_pc: None,
            close_merge_point_vsd: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        let _ = state.with_ctx(|this, ctx| this.trace_guarded_int_payload(ctx, int_obj));

        let recorder = ctx.into_recorder();
        let mut saw_guard_nonnull_class = false;
        let mut saw_pure_payload = false;
        for pos in 1..(1 + recorder.num_ops() as u32) {
            let Some(op) = recorder.get_op_by_raw_pos(pos) else {
                continue;
            };
            if op.opcode == OpCode::GuardClass {
                saw_guard_nonnull_class = true;
            }
            if op.opcode == OpCode::GetfieldGcI
                && op
                    .getarglist()
                    .iter()
                    .map(|a| a.to_opref())
                    .collect::<Vec<_>>()
                    == vec![int_obj]
            {
                saw_pure_payload = true;
            }
        }
        assert!(
            saw_guard_nonnull_class,
            "int payload fast path should guard object class via GuardClass"
        );
        assert!(
            saw_pure_payload,
            "int payload fast path should read the immutable payload with GetfieldGcI"
        );
    }

    #[test]
    fn test_trace_unbox_int_with_resume_skips_guard_for_constant_object() {
        let mut ctx = crate::trace_ctx_for_test(0);
        let int_obj = ctx.const_ref(w_int_new(7) as i64);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);

        let payload = {
            let mut state = MIFrame {
                ctx: &mut ctx,
                sym: &mut sym,
                fallthrough_pc: 0,
                pending_result_stack_idx: None,
                pending_result_type: None,
                pending_inline_frame: None,
                residual_call_pc: None,
                loop_close_marker_jit_pc: None,
                close_merge_point_vsd: None,
                orgpc: 0,
                concrete_frame_addr: 0,
                pre_opcode_registers_r: None,
                pre_opcode_semantic_depth: None,
            };
            trace_unbox_int_with_resume(&mut state, int_obj, &INT_TYPE as *const PyType as i64)
        };

        let recorder = ctx.into_recorder();
        // A boxed W_Int reads its immutable payload with the plain
        // GetfieldGcI (purity re-derived from the descr by OptHeap). The
        // lowbit tag-discrimination guard (rtagged.py ll_unboxed_getclass) is elided because
        // `obj` is a Const (pyjitpl.py generate_guard const-skip), so no
        // guard is emitted at all.
        let payload_op = recorder
            .get_op_by_pos(payload)
            .expect("payload op should be present");
        assert_eq!(payload_op.opcode, OpCode::GetfieldGcI);
        for pos in 1..(1 + recorder.num_ops() as u32) {
            let Some(op) = recorder.get_op_by_raw_pos(pos) else {
                continue;
            };
            assert!(
                !op.opcode.is_guard(),
                "constant unbox must not emit any guard (got {:?})",
                op.opcode,
            );
        }
    }

    #[test]
    fn test_init_symbolic_skips_heap_array_read_for_standard_virtualizable() {
        use pyre_interpreter::pyframe::PyFrame;

        let code = compile_exec("1 + 2").expect("test code should compile");
        let mut frame = PyFrame::new(code.clone());
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;
        install_test_jitcode(&code, frame.pycode);
        let mut ctx = crate::trace_ctx_for_test(1);
        let mut sym = PyreSym::new_uninit(OpRef::input_arg_ref(0));
        sym.become_active_vable_owner();

        sym.init_symbolic(&mut ctx, frame_ptr);

        assert_eq!(sym.locals_cells_stack_array_ref, OpRef::NONE);
        let recorder = ctx.into_recorder();
        for pos in 1..(1 + recorder.num_ops() as u32) {
            let Some(op) = recorder.get_op_by_raw_pos(pos) else {
                continue;
            };
            assert_ne!(
                op.opcode,
                OpCode::GetfieldRawI,
                "standard virtualizable init should not read locals array from heap"
            );
        }
    }

    #[test]
    fn test_materialize_virtual_ref_reconstructs_float_object() {
        let mut state = empty_state();
        let meta = empty_meta();
        let value = 3.25f64;
        let descr = crate::descr::w_float_size_descr();
        let materialized = MaterializedVirtual::Obj {
            descr: Some(descr.clone()),
            type_id: crate::descr::W_FLOAT_GC_TYPE_ID,
            fields: vec![(
                crate::descr::float_floatval_descr().index(),
                MaterializedValue::Value(value.to_bits() as i64),
            )],
        };

        let ptr = <PyreJitState as JitState>::materialize_virtual_ref_with_refs(
            &mut state,
            &meta,
            0,
            &materialized,
            &[],
        )
        .expect("float virtual should materialize");

        unsafe {
            assert!(is_float(ptr.0 as PyObjectRef));
            assert_eq!(w_float_get_value(ptr.0 as PyObjectRef), value);
        }
    }

    /// W_BoolObject has a dedicated GC type id (`W_BOOL_GC_TYPE_ID = 5`)
    /// but shares `W_IntObject`'s layout, distinguished by the
    /// `&BOOL_TYPE` vtable. This test passes `type_id: 0` to confirm the
    /// orthodox materializer dispatches on `descr.vtable()` rather than
    /// `type_id`.
    #[test]
    fn test_materialize_virtual_ref_reconstructs_bool_object() {
        use pyre_object::boolobject::w_bool_get_value;
        let mut state = empty_state();
        let meta = empty_meta();
        let descr = crate::descr::w_bool_size_descr();
        let materialized = MaterializedVirtual::Obj {
            descr: Some(descr.clone()),
            type_id: 0,
            fields: vec![(
                crate::descr::bool_intval_descr().index(),
                MaterializedValue::Value(1),
            )],
        };

        let ptr = <PyreJitState as JitState>::materialize_virtual_ref_with_refs(
            &mut state,
            &meta,
            0,
            &materialized,
            &[],
        )
        .expect("bool virtual should materialize");

        unsafe {
            assert!(is_bool(ptr.0 as PyObjectRef));
            assert!(w_bool_get_value(ptr.0 as PyObjectRef));
        }
    }

    /// Second type_id=0 case: W_IntRangeIterator has three i64 fields.
    /// Verifies generic field replay at different offsets (no
    /// hard-coded PAYLOAD_0/PAYLOAD_1 dispatch).
    #[test]
    fn test_materialize_virtual_ref_reconstructs_range_iterator() {
        use pyre_object::functional::W_IntRangeIterator;
        let mut state = empty_state();
        let meta = empty_meta();
        let descr = crate::descr::w_range_iter_size_descr();
        let materialized = MaterializedVirtual::Obj {
            descr: Some(descr.clone()),
            type_id: 0,
            fields: vec![
                (
                    crate::descr::range_iter_current_descr().index(),
                    MaterializedValue::Value(7),
                ),
                (
                    crate::descr::range_iter_remaining_descr().index(),
                    MaterializedValue::Value(42),
                ),
                (
                    crate::descr::range_iter_step_descr().index(),
                    MaterializedValue::Value(3),
                ),
            ],
        };

        let ptr = <PyreJitState as JitState>::materialize_virtual_ref_with_refs(
            &mut state,
            &meta,
            0,
            &materialized,
            &[],
        )
        .expect("range-iter virtual should materialize");

        unsafe {
            let iter = &*(ptr.0 as *const W_IntRangeIterator);
            assert_eq!(iter.current, 7);
            assert_eq!(iter.remaining, 42);
            assert_eq!(iter.step, 3);
        }
    }

    // Needs a real raw-buffer allocator (`func` passed as a valid function
    // pointer into `bh_call_i`) and a callinfocollection carrying
    // OS_RAW_MALLOC_VARSIZE_CHAR. The current test supplies `func: 0` with no
    // registered calldescr, so `materialize_virtual_raw_buffer` cannot run the
    // resume.py:1456 bh_call_i; wiring up the backend allocator is a follow-up.
    #[ignore = "VRawBufferInfo recovery requires MetaInterp.callinfocollection with OS_RAW_MALLOC_VARSIZE_CHAR; current test has func=0/no calldescr"]
    #[test]
    fn test_materialize_virtual_ref_reconstructs_list_from_raw_buffer_ref() {
        ensure_test_callbacks();
        let mut state = empty_state();
        let meta = empty_meta();
        let first = w_int_new(2);
        let second = w_int_new(4);

        let raw_items = MaterializedVirtual::RawBuffer {
            func: 0,
            size: 16,
            offsets: vec![0, 8],
            descrs: vec![
                majit_ir::ArrayDescrInfo {
                    index: 0,
                    base_size: 0,
                    item_size: 8,
                    item_type: 1,
                    is_signed: false,
                    len_offset: None,
                },
                majit_ir::ArrayDescrInfo {
                    index: 0,
                    base_size: 0,
                    item_size: 8,
                    item_type: 1,
                    is_signed: false,
                    len_offset: None,
                },
            ],
            values: vec![
                MaterializedValue::Value(first as i64),
                MaterializedValue::Value(second as i64),
            ],
        };
        let raw_ptr = <PyreJitState as JitState>::materialize_virtual_ref_with_refs(
            &mut state,
            &meta,
            0,
            &raw_items,
            &[],
        )
        .expect("raw items buffer should materialize");

        let list_virtual = MaterializedVirtual::Obj {
            descr: None,
            type_id: 0,
            fields: vec![
                (
                    crate::descr::list_length_descr().index(),
                    MaterializedValue::Value(2),
                ),
                (
                    crate::descr::list_items_descr().index(),
                    MaterializedValue::VirtualRef(0),
                ),
            ],
        };

        let list_ptr = <PyreJitState as JitState>::materialize_virtual_ref_with_refs(
            &mut state,
            &meta,
            1,
            &list_virtual,
            &[Some(raw_ptr)],
        )
        .expect("list virtual should materialize");

        unsafe {
            assert!(is_list(list_ptr.0 as PyObjectRef));
            assert_eq!(
                w_int_get_value(w_list_getitem(list_ptr.0 as PyObjectRef, 0).unwrap()),
                2
            );
            assert_eq!(
                w_int_get_value(w_list_getitem(list_ptr.0 as PyObjectRef, 1).unwrap()),
                4
            );
        }
    }

    #[test]
    fn test_virtualizable_array_lengths_uses_full_array() {
        use pyre_interpreter::pyframe::PyFrame;

        let code = compile_exec("x = 1").expect("test code should compile");
        let mut frame = PyFrame::new(code);
        frame.fix_array_ptrs();
        let full_len = locals_w!(frame).len();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut state = empty_state();
        state.frame = frame_ptr;
        state.set_valuestackdepth(2);
        let info = crate::frame_layout::build_pyframe_virtualizable_info();

        // virtualizable.py read_boxes parity: full array length, not valuestackdepth.
        assert_eq!(
            <PyreJitState as JitState>::virtualizable_array_lengths(
                &state,
                &empty_meta(),
                "frame",
                &info,
            ),
            Some(vec![full_len])
        );
    }

    // the preserves-comparison-truth-across-extended-arg-trivia test was
    // removed: the last_comparison_truth cache no longer exists. Trivia
    // skipping between COMPARE_OP and POP_JUMP_IF* is now verified through
    // the fused-dispatch path (try_fused_compare_goto_if_not), which uses
    // semantic_fallthrough_pc to locate the branch across trivia.

    // Tests for concrete_popped_value, concrete_binary_operands,
    // concrete_store_subscr_operands removed: these stack-based concrete
    // read methods were replaced by direct FrontendOp.concrete parameter passing.

    // test_concrete_branch_truth_reads_last_popped_slot removed:
    // concrete_branch_truth now requires explicit concrete parameter.

    // test_concrete_branch_truth_uses_cached_comparison_truth_without_stack_value
    // and test_truth_value_direct_caches_concrete_truth_for_raw_int_branch_consumer
    // (consolidated on main as test_branch_truth_concrete_cache_paths) removed:
    // all three exercised the last_comparison_truth / last_comparison_concrete_truth
    // cache fields. Those fields were eliminated when the
    // try_fused_compare_goto_if_not dispatcher consumed the fused
    // COMPARE_OP + POP_JUMP_IF* pair directly (pyjitpl.py:541-556 parity).

    // test_close_loop_args_at_target_pc_preserves_virtualizable_stack moved
    // to `pyre-jit` so close_loop_args_at runs with a real compiled jitcode.

    /// `pyjitpl.py MIFrame.setup` parity: after
    /// `setup_kind_register_banks` runs, `registers_i` / `registers_r` /
    /// `registers_f` are sized to `num_regs_X + len(constants_X)` and
    /// the trailing `[num_regs_X..)` slots hold the constant-pool
    /// OpRefs from `pyjitpl.py copy_constants`
    /// (`ctx.const_int(constants_i[i])` for the int bank, `ctx.const_ref`
    /// for the ref bank, `ctx.const_float` for the float bank). The
    /// leading `[..num_regs_X)` register slots stay `OpRef::NONE` (the
    /// `CONST_NULL`-shaped placeholder).
    #[test]
    fn test_setup_kind_register_banks_sizes_per_num_regs_and_consts() {
        let mut runtime_jc = majit_metainterp::jitcode::JitCode::default();
        runtime_jc.body_mut().c_num_regs_i = 3;
        runtime_jc.body_mut().c_num_regs_r = 4;
        runtime_jc.body_mut().c_num_regs_f = 2;
        runtime_jc.body_mut().constants_i = vec![100, 200];
        runtime_jc.body_mut().constants_r = vec![(0xAABB_CCDD_u64 as i64).into()];
        runtime_jc.body_mut().constants_f = vec![3.25_f64.to_bits() as i64];

        let mut pyjit = crate::PyJitCode::skeleton(std::ptr::null());
        pyjit.jitcode = std::sync::Arc::new(runtime_jc);
        let inner_jc = super::JitCode {
            index: -1,
            payload: std::sync::Arc::new(pyjit),
        };
        let inner_jc_ptr = Box::into_raw(Box::new(inner_jc));

        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.jitcode = inner_jc_ptr;

        assert_eq!(sym.registers_i.len(), 0);
        assert_eq!(sym.registers_r.len(), 0);
        assert_eq!(sym.registers_f.len(), 0);

        let mut ctx = crate::trace_ctx_for_test(1);
        sym.setup_kind_register_banks(&mut ctx);

        // bank size = num_regs_X + len(constants_X)
        assert_eq!(sym.registers_i.len(), 3 + 2, "registers_i sized to 5");
        assert_eq!(sym.registers_r.len(), 4 + 1, "registers_r sized to 5");
        assert_eq!(sym.registers_f.len(), 2 + 1, "registers_f sized to 3");

        // Leading register slots — CONST_NULL placeholder per
        // pyjitpl.py:86-90.
        for i in 0..3 {
            assert_eq!(sym.registers_i[i], OpRef::NONE, "registers_i[{i}] reg-slot");
        }
        for i in 0..4 {
            assert_eq!(
                sym.registers_r.get(i).expect("bound Ref register"),
                OpRef::NONE,
                "registers_r[{i}] reg-slot"
            );
        }
        for i in 0..2 {
            assert_eq!(sym.registers_f[i], OpRef::NONE, "registers_f[{i}] reg-slot");
        }
        // Trailing constant slots — copy_constants populated. The pool
        // dedups by (type, raw value) so each kind's trailing slot
        // resolves to a distinct constant OpRef.
        for i in 0..2 {
            let op = sym.registers_i[3 + i];
            assert_ne!(op, OpRef::NONE, "registers_i[{}] constant slot", 3 + i);
            let val = ctx
                .constants_get_value(op)
                .expect("constants pool resolves trailing int slot");
            assert_eq!(val, majit_ir::Value::Int(100 + 100 * i as i64));
        }
        let op_r = sym.registers_r.get(4).expect("bound Ref register");
        assert_ne!(op_r, OpRef::NONE);
        assert!(matches!(
            ctx.constants_get_value(op_r),
            Some(majit_ir::Value::Ref(_))
        ));
        let op_f = sym.registers_f[2];
        assert_ne!(op_f, OpRef::NONE);
        assert!(matches!(
            ctx.constants_get_value(op_f),
            Some(majit_ir::Value::Float(_))
        ));

        // Calling twice does not shrink the register banks. `history.py:220
        // ConstInt`, `:261 ConstFloat`, `:307 ConstPtr` are all fresh-alloc
        // per construction; `Const.same_constant` (history.py) is the
        // upstream value-equality predicate. Assert value-equality across
        // both calls via `constants_get_value` rather than OpRef identity,
        // independent of whether the pool internally dedups.
        let trailing_i_value_before = ctx
            .constants_get_value(sym.registers_i[3])
            .expect("first-call trailing int slot resolves to a constant");
        let trailing_r_value_before = ctx
            .constants_get_value(sym.registers_r.get(4).expect("bound Ref register"))
            .expect("first-call trailing ref slot resolves to a constant");
        let trailing_f_value_before = ctx
            .constants_get_value(sym.registers_f[2])
            .expect("first-call trailing float slot resolves to a constant");
        sym.setup_kind_register_banks(&mut ctx);
        assert_eq!(sym.registers_i.len(), 5);
        assert_eq!(sym.registers_r.len(), 5);
        assert_eq!(sym.registers_f.len(), 3);
        assert_eq!(
            ctx.constants_get_value(sym.registers_i[3]),
            Some(trailing_i_value_before),
        );
        assert_eq!(
            ctx.constants_get_value(sym.registers_r.get(4).expect("bound Ref register")),
            Some(trailing_r_value_before),
        );
        assert_eq!(
            ctx.constants_get_value(sym.registers_f[2]),
            Some(trailing_f_value_before),
        );

        // SAFETY: drop the boxed JitCode; sym.jitcode now dangles but goes
        // out of scope at the end of this test.
        unsafe {
            let _ = Box::from_raw(inner_jc_ptr);
        }
    }

    /// `setup_kind_register_banks` is safe to call when `self.jitcode`
    /// points at the thread-local `null_jitcode()` placeholder — the
    /// skeleton's `num_regs_and_consts_X` values are zero and the
    /// constant pools are empty so the resize and the constant fill are
    /// both no-ops.
    #[test]
    fn test_setup_kind_register_banks_is_no_op_for_null_jitcode_placeholder() {
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        // sym.jitcode was initialized by new_uninit to null_jitcode().
        let mut ctx = crate::trace_ctx_for_test(1);
        sym.setup_kind_register_banks(&mut ctx);
        assert_eq!(sym.registers_i.len(), 0);
        assert_eq!(sym.registers_f.len(), 0);
        assert_eq!(sym.registers_r.len(), 0);
    }

    #[test]
    fn test_setup_bridge_sym_preserves_vable_locals_and_resumed_stack() {
        use majit_ir::resumedata::{RebuiltFrame, RebuiltValue};
        use majit_metainterp::jitcode::JitCodeBuilder;
        use pyre_interpreter::pyframe::PyFrame;

        ensure_test_callbacks();

        let raw_code = compile_exec("len(x)").expect("test code should compile");
        let mut frame = PyFrame::new(raw_code);
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;
        let code_ref = frame.pycode as *const ();

        // Seed the live/ -> BC_LIVE opcode mapping so the jitcode decoder
        // recognises the pc=0 liveness marker. `op_live` defaults to
        // `u8::MAX` until published; without this the decode underflows
        // (`pc -= OFFSET_SIZE + 1` at jitcode.rs). `publish_state` clears
        // `all_liveness`, so it must run before `intern_liveness` below.
        {
            let mut insns = indexmap::IndexMap::new();
            insns.insert(
                "live/".to_string(),
                majit_metainterp::jitcode::insns::BC_LIVE,
            );
            crate::assembler::publish_state(&insns, &[], 0, 0);
        }
        // Three semantic frame colors followed by the two dedicated portal
        // reds. `PyPyJitDriver.reds = ['frame', 'ec']` keeps the latter in
        // the frame-register resume section even though they are not locals.
        let live_off = crate::state::intern_liveness(&[], &[0, 1, 2, 3, 4], &[])
            .expect("bridge liveness must fit in the shared buffer");
        let mut builder = JitCodeBuilder::new();
        let patch = builder.live_placeholder();
        builder.patch_live_offset(patch, live_off);
        let runtime_jitcode = std::sync::Arc::new(builder.finish());
        let pyjit = std::sync::Arc::new(crate::PyJitCode::from_parts(
            runtime_jitcode,
            crate::PyJitCodeMetadata {
                forward_py_pc_marker_by_jit_pc: Vec::new(),
                forward_py_pc_pred_by_jit_pc: Vec::new(),
                after_residual_call_resume_marker_by_jit_pc: Vec::new(),
                after_residual_call_resume_pred_by_jit_pc: Vec::new(),
                n_py_instrs: 0,
                block_head_py_by_jit_pc: vec![(0, 0)],
                py_floor_by_jit_pc: Vec::new(),
                py_exact_by_jit_pc: Vec::new(),
                abort_permanent_py_pc_by_jit_pc: Vec::new(),
                merge_entry_by_green: Vec::new(),
                pcdep_by_jit_pc: vec![(0, Vec::new())],
                depth_pred_by_jit_pc: vec![(0, 1)],
                depth_trivia_marker_by_jit_pc: vec![(0, Some(1))],
                depth_trivia_pred_by_jit_pc: vec![(0, Some(1))],
                depth_containing_by_jit_pc: Vec::new(),
                depth_block_head_by_jit_pc: Vec::new(),
                pcdep_trivia_marker_by_jit_pc: Vec::new(),
                pcdep_trivia_pred_by_jit_pc: Vec::new(),
                const_ref_trivia_marker_by_jit_pc: Vec::new(),
                const_ref_trivia_pred_by_jit_pc: Vec::new(),
                result_color_trivia_marker_by_jit_pc: Vec::new(),
                result_color_trivia_pred_by_jit_pc: Vec::new(),
                resume_marker_marker_by_jit_pc: Vec::new(),
                resume_marker_pred_by_jit_pc: vec![(0, Some(0))],
                after_residual_marker_marker_by_jit_pc: Vec::new(),
                after_residual_marker_pred_by_jit_pc: Vec::new(),
                result_color_after_residual_marker_by_jit_pc: Vec::new(),
                result_color_after_residual_pred_by_jit_pc: Vec::new(),
                depth_after_residual_marker_by_jit_pc: Vec::new(),
                depth_after_residual_pred_by_jit_pc: Vec::new(),
                after_residual_fallthrough_py_pc_marker_by_jit_pc: Vec::new(),
                after_residual_fallthrough_py_pc_pred_by_jit_pc: Vec::new(),
                has_color_map: false,
                portal_frame_reg: 3,
                portal_ec_reg: 4,
                built_as_portal: true,
                stack_base: 1,
                max_stackdepth: 0,
                const_ref_slots_by_jit_pc: Vec::new(),
                is_drained: true,
            },
            std::ptr::null(),
            false,
        ));
        let jitcode_index = METAINTERP_SD.with(|r| unsafe {
            let ptr = r.borrow_mut().jitcode_for(code_ref, Some(pyjit));
            (*ptr).index
        });

        let input_types = [
            Type::Ref, // frame
            Type::Ref, // ec
            Type::Int, // next_instr
            Type::Ref, // pycode
            Type::Int, // valuestackdepth
            Type::Ref, // debugdata
            Type::Ref, // local0
            Type::Ref, // stale cell
            Type::Ref, // stack0
            Type::Ref, // live cell from the vable image
        ];
        let mut ctx = TraceCtx::for_test_types(&input_types);
        // Slots 0 (frame) and 1 (ec) are both Ref-typed per `input_types`
        // — production `init_vable_indices` mints typed `InputArgRef`
        // variants here (resoperation.py, `virtualizable_gen.rs`'s
        // `inputargs`), so variant-aware Eq (`OpRef`'s `PartialEq`) requires
        // the matching `OpRef::input_arg_ref` shape.
        let mut sym = PyreSym::new_uninit(OpRef::input_arg_ref(0));
        sym.frame = OpRef::input_arg_ref(0);
        sym.execution_context = OpRef::input_arg_ref(1);
        sym.nlocals = 2;
        sym.valuestackdepth = 2;
        sym.concrete_vable_ptr = frame_ptr as *mut u8;

        let local0 = w_int_new(41) as i64;
        let stale_cell = w_int_new(42) as i64;
        let stack0 = w_int_new(43) as i64;
        let live_cell = w_int_new(45) as i64;
        let execution_context = w_int_new(40) as i64;
        let fail_values = [
            frame_ptr as i64,
            execution_context,
            77,
            code_ref as i64,
            3,
            0,
            local0,
            stale_cell,
            stack0,
            live_cell,
        ];
        let fail_types = [
            Type::Ref,
            Type::Ref,
            Type::Int,
            Type::Ref,
            Type::Int,
            Type::Ref,
            Type::Ref,
            Type::Ref,
            Type::Ref,
            Type::Ref,
        ];
        let resume_data = majit_metainterp::ResumeDataResult {
            frames: vec![RebuiltFrame {
                jitcode_index,
                pc: 0,
                values: vec![
                    RebuiltValue::Const(majit_ir::Const::Ref(majit_ir::GcRef::NULL)),
                    RebuiltValue::Box(7, Type::Ref),
                    RebuiltValue::Box(8, Type::Ref),
                    RebuiltValue::Box(0, Type::Ref),
                    RebuiltValue::Box(1, Type::Ref),
                ],
            }],
            // Exercise both authorities in one bridge setup.  The first local's
            // register is a dead null and the second local/cell's register is a
            // stale non-null object; both vable entries are live.  Conversely,
            // the stack entry in the vable image is stale while the frame
            // register contains the after-call result.
            virtualizable_boxes: vec![
                RebuiltValue::Box(0, Type::Ref),
                RebuiltValue::Box(2, Type::Int),
                RebuiltValue::Box(3, Type::Ref),
                RebuiltValue::Box(4, Type::Int),
                RebuiltValue::Box(5, Type::Ref),
                RebuiltValue::Box(6, Type::Ref),
                RebuiltValue::Box(9, Type::Ref),
                RebuiltValue::Box(6, Type::Ref),
            ],
            virtualref_values: Vec::new(),
            storage: None,
            num_failargs: fail_values.len() as i32,
            fail_arg_types: fail_types.to_vec(),
        };

        let mut reader = BridgeVirtualCache::new(4, crate::descr::make_array_descr);
        let boxes = <PyreJitState as majit_metainterp::JitState>::consume_vref_and_vable_boxes(
            &mut sym,
            &mut ctx,
            &resume_data,
            None,
            &fail_values,
            &fail_types,
            &mut reader,
        );
        <PyreJitState as majit_metainterp::JitState>::setup_bridge_sym(
            &mut sym,
            &mut ctx,
            &resume_data,
            None,
            &fail_values,
            &fail_types,
            &mut reader,
            &boxes,
        );

        assert_eq!(sym.valuestackdepth, 3);
        // setup_bridge_sym now restores typed `InputArg*` OpRefs from
        // `RebuiltValue::Box(idx, tp)`. The resolve closure produces
        // `OpRef::input_arg_typed(idx, tp)`; expectations
        // must match so variant-aware Eq lines up with the resolved
        // bridge inputarg list.
        assert_eq!(
            sym.registers_r.to_vec(),
            vec![
                OpRef::input_arg_ref(6),
                OpRef::input_arg_ref(7),
                OpRef::input_arg_ref(8),
            ]
        );
        assert_eq!(sym.symbolic_local_types, vec![Type::Ref, Type::Ref]);
        assert_eq!(sym.symbolic_stack_types, vec![Type::Ref]);
        assert_eq!(sym.execution_context, OpRef::input_arg_ref(1));
        assert_eq!(
            sym.bridge_local_oprefs.as_ref().map(|regs| regs.to_vec()),
            Some(vec![OpRef::input_arg_ref(6), OpRef::input_arg_ref(7)])
        );
        assert_eq!(
            sym.concrete_locals,
            vec![ConcreteValue::Int(41), ConcreteValue::Int(45)]
        );
        let array_base = crate::virtualizable_gen::NUM_VABLE_SCALARS;
        assert_eq!(
            ctx.virtualizable_entry_at(array_base),
            Some((
                OpRef::input_arg_ref(6),
                majit_ir::Value::Ref(majit_ir::GcRef(local0 as usize)),
            )),
        );
        assert_eq!(
            ctx.virtualizable_entry_at(array_base + 1),
            Some((
                OpRef::input_arg_ref(9),
                majit_ir::Value::Ref(majit_ir::GcRef(live_cell as usize)),
            )),
        );
        assert_eq!(
            ctx.virtualizable_entry_at(array_base + 2),
            Some((
                OpRef::input_arg_ref(8),
                majit_ir::Value::Ref(majit_ir::GcRef(stack0 as usize)),
            )),
        );
        assert_eq!(
            ctx.box_value(OpRef::input_arg_ref(8)),
            Some(majit_ir::Value::Ref(majit_ir::GcRef(stack0 as usize))),
        );
    }

    /// `rebuild_state_after_failure` still runs `consume_virtualizable_boxes`
    /// when `framestack` stays empty. An empty frame list must not return
    /// before that payload is applied.
    #[test]
    fn setup_bridge_sym_applies_vable_payload_when_frames_are_empty() {
        use majit_ir::resumedata::RebuiltValue;
        use pyre_interpreter::pyframe::PyFrame;

        ensure_test_callbacks();
        let raw_code = compile_exec("None").expect("test code should compile");
        let mut frame = PyFrame::new(raw_code);
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;
        let code_ref = frame.pycode as *const ();
        let input_types = [
            Type::Ref,
            Type::Ref,
            Type::Int,
            Type::Ref,
            Type::Int,
            Type::Ref,
        ];
        let mut ctx = TraceCtx::for_test_types(&input_types);
        let mut sym = PyreSym::new_uninit(OpRef::input_arg_ref(0));
        sym.nlocals = 0;
        sym.valuestackdepth = 1;
        sym.concrete_vable_ptr = frame_ptr as *mut u8;
        let fail_values = [frame_ptr as i64, 0, 9, code_ref as i64, 4, 0];
        let fail_types = [
            Type::Ref,
            Type::Ref,
            Type::Int,
            Type::Ref,
            Type::Int,
            Type::Ref,
        ];
        let resume_data = majit_metainterp::ResumeDataResult {
            frames: Vec::new(),
            virtualizable_boxes: vec![
                RebuiltValue::Box(0, Type::Ref),
                RebuiltValue::Box(2, Type::Int),
                RebuiltValue::Box(3, Type::Ref),
                RebuiltValue::Box(4, Type::Int),
                RebuiltValue::Box(5, Type::Ref),
            ],
            virtualref_values: Vec::new(),
            storage: None,
            num_failargs: fail_values.len() as i32,
            fail_arg_types: fail_types.to_vec(),
        };
        let mut reader = BridgeVirtualCache::new(4, crate::descr::make_array_descr);
        let boxes = <PyreJitState as majit_metainterp::JitState>::consume_vref_and_vable_boxes(
            &mut sym,
            &mut ctx,
            &resume_data,
            None,
            &fail_values,
            &fail_types,
            &mut reader,
        );
        <PyreJitState as majit_metainterp::JitState>::setup_bridge_sym(
            &mut sym,
            &mut ctx,
            &resume_data,
            None,
            &fail_values,
            &fail_types,
            &mut reader,
            &boxes,
        );
        assert_eq!(sym.valuestackdepth, 4);
    }

    /// `resume.py` `_prepare` runs `_prepare_pendingfields` once, before
    /// `consume_boxes`; it applies SETFIELD_GC through
    /// `execute_setfield_gc`. An applying reader (`executing: Some`) with a
    /// nonempty pending stream must keep the bridge complete.
    #[test]
    fn prepare_bridge_resume_applies_pending_fields_without_marking_incomplete() {
        ensure_test_callbacks();
        #[repr(C)]
        struct FieldTarget {
            value: i64,
        }
        let mut field_target = FieldTarget { value: 1 };
        let field_descr: majit_ir::DescrRef = std::sync::Arc::new(
            majit_ir::descr::SimpleFieldDescr::new(1, 0, 8, Type::Int, false),
        );
        let tagged = |index| {
            majit_metainterp::resume::tag(index, majit_metainterp::resume::TAGBOX)
                .expect("small fail-argument index is taggable")
        };
        let pending = vec![majit_ir::GuardPendingFieldEntry {
            descr: Some(field_descr),
            item_index: -1,
            target: OpRef::input_arg_ref(0),
            value: OpRef::input_arg_int(1),
            target_tagged: tagged(0),
            value_tagged: tagged(1),
        }];
        let storage = majit_metainterp::resume::ResumeStorage::new(vec![], vec![], vec![], pending);
        let fail_values = [&mut field_target as *mut FieldTarget as i64, 9];
        let fail_types = [Type::Ref, Type::Int];
        let resume_data = majit_metainterp::ResumeDataResult {
            frames: Vec::new(),
            virtualizable_boxes: Vec::new(),
            virtualref_values: Vec::new(),
            storage: Some(storage),
            num_failargs: fail_values.len() as i32,
            fail_arg_types: fail_types.to_vec(),
        };
        let mut ctx = TraceCtx::for_test_types(&fail_types);
        let mut sym = PyreSym::new_uninit(OpRef::input_arg_ref(0));
        let allocator = majit_metainterp::resume::NullAllocator;
        let mut reader = BridgeVirtualCache::executing(
            0,
            crate::descr::make_array_descr,
            &allocator,
            &fail_values,
            &fail_types,
        );
        <PyreJitState as majit_metainterp::JitState>::prepare_bridge_resume(
            &mut sym,
            &mut ctx,
            &resume_data,
            None,
            &fail_values,
            &fail_types,
            &mut reader,
        );
        assert_eq!(field_target.value, 9, "_prepare applies the pending field");
        field_target.value = 1;
        let boxes = <PyreJitState as majit_metainterp::JitState>::consume_vref_and_vable_boxes(
            &mut sym,
            &mut ctx,
            &resume_data,
            None,
            &fail_values,
            &fail_types,
            &mut reader,
        );
        <PyreJitState as majit_metainterp::JitState>::setup_bridge_sym(
            &mut sym,
            &mut ctx,
            &resume_data,
            None,
            &fail_values,
            &fail_types,
            &mut reader,
            &boxes,
        );
        assert!(
            !ctx.bridge_replay_incomplete(),
            "ResumeDataBoxReader applies pending fields; the bridge stays complete"
        );
        assert_eq!(
            field_target.value, 1,
            "consume_boxes does not apply them again"
        );
    }

    /// BoxReader `_prepare` is `_prepare_virtuals` (cache only) then
    /// `_prepare_pendingfields`. `getvirtual_ptr` materializes the virtual a
    /// pending field names; Empty holes and unnamed nonempty `rd_virtuals`
    /// entries stay unset. `force_all_virtuals` is not this path.
    #[test]
    fn prepare_bridge_resume_materializes_only_named_virtuals() {
        ensure_test_callbacks();
        #[repr(C)]
        struct FieldTarget {
            value: i64,
        }
        let mut field_target = FieldTarget { value: 1 };
        let field_descr: majit_ir::DescrRef = std::sync::Arc::new(
            majit_ir::descr::SimpleFieldDescr::new(1, 0, 8, Type::Int, false),
        );
        let tagged_box = |index| {
            majit_metainterp::resume::tag(index, majit_metainterp::resume::TAGBOX)
                .expect("small fail-argument index is taggable")
        };
        let tagged_virtual = |index| {
            majit_metainterp::resume::tag(index, majit_metainterp::resume::TAGVIRTUAL)
                .expect("small virtual index is taggable")
        };
        let tag_int = |value: i32| {
            majit_metainterp::resume::tag(value, majit_metainterp::resume::TAGINT)
                .expect("small TAGINT is taggable")
        };
        let named = std::rc::Rc::new(majit_ir::RdVirtualInfo::VRawSliceInfo {
            offset: 7,
            fieldnums: vec![tag_int(5)],
        });
        let unnamed = std::rc::Rc::new(majit_ir::RdVirtualInfo::VRawSliceInfo {
            offset: 99,
            fieldnums: vec![tag_int(1)],
        });
        let virtuals = vec![
            named,
            std::rc::Rc::new(majit_ir::RdVirtualInfo::Empty),
            unnamed,
        ];
        let pending = vec![majit_ir::GuardPendingFieldEntry {
            descr: Some(field_descr),
            item_index: -1,
            target: OpRef::input_arg_ref(0),
            value: OpRef::NONE,
            target_tagged: tagged_box(0),
            value_tagged: tagged_virtual(0),
        }];
        let storage =
            majit_metainterp::resume::ResumeStorage::new(vec![], vec![], virtuals.clone(), pending);
        let fail_values = [&mut field_target as *mut FieldTarget as i64];
        let fail_types = [Type::Ref];
        let resume_data = majit_metainterp::ResumeDataResult {
            frames: Vec::new(),
            virtualizable_boxes: Vec::new(),
            virtualref_values: Vec::new(),
            storage: Some(storage),
            num_failargs: fail_values.len() as i32,
            fail_arg_types: fail_types.to_vec(),
        };
        let mut ctx = TraceCtx::for_test_types(&fail_types);
        let mut sym = PyreSym::new_uninit(OpRef::input_arg_ref(0));
        assert!(ctx.bridge_virtual_op(0).is_none());
        assert!(ctx.bridge_virtual_op(1).is_none());
        assert!(ctx.bridge_virtual_op(2).is_none());
        <PyreJitState as majit_metainterp::JitState>::prepare_bridge_resume(
            &mut sym,
            &mut ctx,
            &resume_data,
            Some(&virtuals),
            &fail_values,
            &fail_types,
            &mut BridgeVirtualCache::new(4, crate::descr::make_array_descr),
        );
        assert_eq!(ctx.bridge_virtual_op(0), Some(OpRef::const_int(12)));
        assert!(
            ctx.bridge_virtual_op(1).is_none(),
            "Empty hole stays unset; BoxReader `_prepare` never names it"
        );
        assert!(
            ctx.bridge_virtual_op(2).is_none(),
            "unnamed nonempty virtual stays unset until TAGVIRTUAL / pending decode"
        );
        assert_eq!(field_target.value, 12);
    }

    /// `ResumeDataBoxReader.virtuals_cache` is created in `_prepare_virtuals`
    /// and reused by a later TAGVIRTUAL `getvirtual_ptr`. `allocate()` runs
    /// setfields once; `setup_bridge_sym` must not SETFIELD a second
    /// `wrappeditems` array.
    #[test]
    fn setup_bridge_sym_reuses_prepare_virtuals_cache_on_later_tagvirtual() {
        use majit_ir::resumedata::RebuiltValue;

        ensure_test_callbacks();
        #[repr(C)]
        struct FieldTarget {
            value: i64,
        }
        let mut field_target = FieldTarget { value: 0 };
        let field: std::sync::Arc<dyn majit_ir::FieldDescr> = std::sync::Arc::new(
            majit_ir::descr::SimpleFieldDescr::new(0, 0, 8, Type::Ref, false),
        );
        let size_descr: majit_ir::DescrRef = std::sync::Arc::new(
            majit_ir::descr::SimpleSizeDescr::new(0, 16, 0)
                .with_all_fielddescrs(vec![field.clone()]),
        );
        let pending_descr: majit_ir::DescrRef = std::sync::Arc::new(
            majit_ir::descr::SimpleFieldDescr::new(1, 0, 8, Type::Ref, false),
        );
        let tagged_box = |index| {
            majit_metainterp::resume::tag(index, majit_metainterp::resume::TAGBOX)
                .expect("small fail-argument index is taggable")
        };
        let tagged_virtual = |index| {
            majit_metainterp::resume::tag(index, majit_metainterp::resume::TAGVIRTUAL)
                .expect("small virtual index is taggable")
        };
        let virtuals = vec![std::rc::Rc::new(majit_ir::RdVirtualInfo::VirtualInfo {
            descr: Some(size_descr),
            type_id: 0,
            known_class: None,
            fielddescrs: vec![majit_ir::FieldDescrInfo {
                index: 0,
                offset: 0,
                field_type: Type::Ref,
                field_size: 8,
            }],
            fieldnums: vec![majit_ir::resumedata::NULLREF],
            descr_size: 16,
        })];
        let pending = vec![majit_ir::GuardPendingFieldEntry {
            descr: Some(pending_descr),
            item_index: -1,
            target: OpRef::input_arg_ref(0),
            value: OpRef::NONE,
            target_tagged: tagged_box(0),
            value_tagged: tagged_virtual(0),
        }];
        let storage =
            majit_metainterp::resume::ResumeStorage::new(vec![], vec![], virtuals.clone(), pending);
        let fail_values = [&mut field_target as *mut FieldTarget as i64];
        let fail_types = [Type::Ref];
        let resume_data = majit_metainterp::ResumeDataResult {
            frames: Vec::new(),
            virtualizable_boxes: Vec::new(),
            virtualref_values: Vec::new(),
            storage: Some(storage),
            num_failargs: fail_values.len() as i32,
            fail_arg_types: fail_types.to_vec(),
        };
        let mut ctx = TraceCtx::for_test_types(&fail_types);
        let mut sym = PyreSym::new_uninit(OpRef::input_arg_ref(0));
        // One reader for `_prepare` and the later fill, as `start_bridge_tracing` builds it.
        let mut reader = BridgeVirtualCache::new(virtuals.len(), crate::descr::make_array_descr);
        <PyreJitState as majit_metainterp::JitState>::prepare_bridge_resume(
            &mut sym,
            &mut ctx,
            &resume_data,
            Some(&virtuals),
            &fail_values,
            &fail_types,
            &mut reader,
        );
        let first = ctx
            .bridge_virtual_op(0)
            .expect("pending TAGVIRTUAL allocate()s once");
        assert_eq!(reader.get_any(0), Some(first));
        let ops_after_prepare = ctx.num_ops();
        let mut resume_later = resume_data.clone();
        resume_later.virtualref_values =
            vec![RebuiltValue::Virtual(0), RebuiltValue::Box(0, Type::Ref)];
        let boxes = <PyreJitState as majit_metainterp::JitState>::consume_vref_and_vable_boxes(
            &mut sym,
            &mut ctx,
            &resume_later,
            Some(&virtuals),
            &fail_values,
            &fail_types,
            &mut reader,
        );
        <PyreJitState as majit_metainterp::JitState>::setup_bridge_sym(
            &mut sym,
            &mut ctx,
            &resume_later,
            Some(&virtuals),
            &fail_values,
            &fail_types,
            &mut reader,
            &boxes,
        );
        assert_eq!(ctx.bridge_virtual_op(0), Some(first));
        assert_eq!(
            ctx.num_ops(),
            ops_after_prepare,
            "later TAGVIRTUAL is a virtuals_cache hit; allocate() must not SETFIELD again"
        );
    }

    /// Recording-only `getvirtual_ptr` parks an `OpRef` without a concrete.
    /// The reader was constructed with the object the DirectReader
    /// allocated (`BridgeVirtualCache::recording`'s `all_virtuals`); a
    /// later TAGVIRTUAL must return that object, not `Value::Void`.
    #[test]
    fn setup_bridge_sym_recovers_direct_reader_object_on_recording_only_hit() {
        use majit_ir::resumedata::RebuiltValue;

        ensure_test_callbacks();
        #[repr(C)]
        struct FieldTarget {
            value: i64,
        }
        let mut field_target = FieldTarget { value: 0 };
        let field: std::sync::Arc<dyn majit_ir::FieldDescr> = std::sync::Arc::new(
            majit_ir::descr::SimpleFieldDescr::new(0, 0, 8, Type::Ref, false),
        );
        let size_descr: majit_ir::DescrRef = std::sync::Arc::new(
            majit_ir::descr::SimpleSizeDescr::new(0, 16, 0)
                .with_all_fielddescrs(vec![field.clone()]),
        );
        let pending_descr: majit_ir::DescrRef = std::sync::Arc::new(
            majit_ir::descr::SimpleFieldDescr::new(1, 0, 8, Type::Ref, false),
        );
        let tagged_box = |index| {
            majit_metainterp::resume::tag(index, majit_metainterp::resume::TAGBOX)
                .expect("small fail-argument index is taggable")
        };
        let tagged_virtual = |index| {
            majit_metainterp::resume::tag(index, majit_metainterp::resume::TAGVIRTUAL)
                .expect("small virtual index is taggable")
        };
        let virtuals = vec![std::rc::Rc::new(majit_ir::RdVirtualInfo::VirtualInfo {
            descr: Some(size_descr),
            type_id: 0,
            known_class: None,
            fielddescrs: vec![majit_ir::FieldDescrInfo {
                index: 0,
                offset: 0,
                field_type: Type::Ref,
                field_size: 8,
            }],
            fieldnums: vec![majit_ir::resumedata::NULLREF],
            descr_size: 16,
        })];
        let pending = vec![majit_ir::GuardPendingFieldEntry {
            descr: Some(pending_descr),
            item_index: -1,
            target: OpRef::input_arg_ref(0),
            value: OpRef::NONE,
            target_tagged: tagged_box(0),
            value_tagged: tagged_virtual(0),
        }];
        let storage =
            majit_metainterp::resume::ResumeStorage::new(vec![], vec![], virtuals.clone(), pending);
        let fail_values = [&mut field_target as *mut FieldTarget as i64];
        let fail_types = [Type::Ref];
        let resume_data = majit_metainterp::ResumeDataResult {
            frames: Vec::new(),
            virtualizable_boxes: Vec::new(),
            virtualref_values: Vec::new(),
            storage: Some(storage),
            num_failargs: fail_values.len() as i32,
            fail_arg_types: fail_types.to_vec(),
        };
        let mut ctx = TraceCtx::for_test_types(&fail_types);
        let mut sym = PyreSym::new_uninit(OpRef::input_arg_ref(0));
        let direct_object = majit_ir::GcRef(&mut field_target as *mut FieldTarget as usize);
        let mut reader = BridgeVirtualCache::recording(
            virtuals.len(),
            crate::descr::make_array_descr,
            &[Some(direct_object)],
        );
        <PyreJitState as majit_metainterp::JitState>::prepare_bridge_resume(
            &mut sym,
            &mut ctx,
            &resume_data,
            Some(&virtuals),
            &fail_values,
            &fail_types,
            &mut reader,
        );
        let first = ctx
            .bridge_virtual_op(0)
            .expect("pending TAGVIRTUAL allocate()s once");
        let ops_after_prepare = ctx.num_ops();
        let mut resume_later = resume_data.clone();
        resume_later.virtualref_values =
            vec![RebuiltValue::Virtual(0), RebuiltValue::Box(0, Type::Ref)];
        let boxes = <PyreJitState as majit_metainterp::JitState>::consume_vref_and_vable_boxes(
            &mut sym,
            &mut ctx,
            &resume_later,
            Some(&virtuals),
            &fail_values,
            &fail_types,
            &mut reader,
        );
        <PyreJitState as majit_metainterp::JitState>::setup_bridge_sym(
            &mut sym,
            &mut ctx,
            &resume_later,
            Some(&virtuals),
            &fail_values,
            &fail_types,
            &mut reader,
            &boxes,
        );
        assert_eq!(ctx.bridge_virtual_op(0), Some(first));
        assert_eq!(
            ctx.num_ops(),
            ops_after_prepare,
            "later TAGVIRTUAL is a virtuals_cache hit; allocate() must not SETFIELD again"
        );
        assert_eq!(
            reader.get_concrete_ptr(0),
            Some(direct_object),
            "getvirtual_ptr cache holds the DirectReader Box"
        );
    }

    #[test]
    fn bridge_semantic_maps_keep_width_but_drop_pcdep_at_non_live_jitcode_offset() {
        use pyre_interpreter::pyframe::PyFrame;

        ensure_test_callbacks();

        let raw_code = compile_exec("x = 1").expect("test code should compile");
        let frame = PyFrame::new(raw_code);
        let code_ref = frame.pycode as *const ();

        // A CALL/residual operation start can be a valid JitCode coordinate
        // without being a `-live-`-anchored decode point.  The codewriter still
        // publishes containing-depth and pcdep sidecars for that coordinate.
        // The static width remains usable, but pcdep describes the register
        // stream consumed by get_current_position_info; without a live marker
        // at pc=9 that stream cannot be decoded and the map must be dropped.
        let mut metadata = crate::PyJitCodeMetadata::degenerate();
        metadata.depth_containing_by_jit_pc = vec![(0, 4)];
        metadata.pcdep_by_jit_pc = vec![(0, vec![(1, 7, 3)])];
        metadata.has_color_map = true;
        metadata.is_drained = true;
        let pyjit = std::sync::Arc::new(crate::PyJitCode::from_parts(
            std::sync::Arc::new(majit_metainterp::jitcode::JitCode::default()),
            metadata,
            std::ptr::null(),
            false,
        ));
        let jitcode_index = METAINTERP_SD.with(|r| unsafe {
            let ptr = r.borrow_mut().jitcode_for(code_ref, Some(pyjit));
            (*ptr).index
        });

        let maps = bridge_semantic_maps_from_jitcode_pc(jitcode_index, 9);
        assert_eq!(maps.stack_depth_at_pc, 4);
        assert!(maps.pcdep_entries.is_empty());
    }

    #[test]
    fn test_close_loop_args_preserves_ec_between_frame_and_virtualizable_header() {
        ensure_test_callbacks();
        let input_types = [
            Type::Ref, // frame
            Type::Ref, // ec
            Type::Int, // next_instr
            Type::Ref, // pycode
            Type::Int, // valuestackdepth
            Type::Ref, // debugdata
            Type::Ref, // local0
            Type::Ref, // stack0
            Type::Ref, // stack1
        ];
        let mut ctx = TraceCtx::for_test_types(&input_types);

        // The vable static-field types come from `virtualizable_gen.rs`'s
        // `inputargs` annotations: int/ref/int/ref.
        // Mint typed `OpRef::input_arg_*` variants matching
        // those tags so variant-aware Eq (`OpRef`'s `PartialEq`) lines up
        // with what the production `init_vable_indices` produces.
        let mut sym = PyreSym::new_uninit(OpRef::input_arg_ref(0));
        sym.execution_context = OpRef::input_arg_ref(1);
        sym.nlocals = 1;
        sym.valuestackdepth = 3;
        sym.vable_last_instr = OpRef::input_arg_int(2);
        sym.vable_pycode = OpRef::input_arg_ref(3);
        sym.vable_valuestackdepth = OpRef::input_arg_int(4);
        sym.vable_debugdata = OpRef::input_arg_ref(5);
        // local0 / stack0 / stack1 are Ref-typed per `symbolic_local_types`
        // / `symbolic_stack_types` below — the macro mints the matching
        // `InputArgRef` variant.
        sym.registers_r.replace(vec![
            OpRef::input_arg_ref(6),
            OpRef::input_arg_ref(7),
            OpRef::input_arg_ref(8),
        ]);
        sym.symbolic_local_types = vec![Type::Ref];
        sym.symbolic_stack_types = vec![Type::Ref, Type::Ref];
        sym.concrete_stack = vec![ConcreteValue::Null, ConcreteValue::Null];
        // Bind a populated jitcode with a real PyCode so the
        // close_loop_args_at snapshot encoder (trace_opcode.rs) reads
        // code.varnames/ncells/max_stackdepth instead of null-derefing. No
        // frame here, so build a standalone wrapper (pattern B). The three
        // live Ref colors map to local0/stack0/stack1 in registers_r.
        let code = compile_exec("len(x)").expect("test code should compile");
        let w_code = pyre_interpreter::w_code_new(Box::into_raw(Box::new(code.clone())) as *const ())
            as *const ();
        let jc_ptr = bind_real_jitcode(&mut sym, w_code, &[], &[0, 1, 2], &[]);

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            residual_call_pc: None,
            loop_close_marker_jit_pc: None,
            close_merge_point_vsd: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        let jump_args = state.with_ctx(|this, ctx| this.close_loop_args_at(ctx, None, None));

        assert_eq!(jump_args.len(), 9);
        assert_eq!(jump_args[0], OpRef::input_arg_ref(0));
        assert_eq!(jump_args[1], OpRef::input_arg_ref(1));
        assert_eq!(
            &jump_args[6..],
            &[
                OpRef::input_arg_ref(6),
                OpRef::input_arg_ref(7),
                OpRef::input_arg_ref(8)
            ]
        );
        assert_eq!(state.sym().execution_context, OpRef::input_arg_ref(1));

        unsafe {
            let _ = Box::from_raw(jc_ptr);
        }
    }

    #[test]
    fn test_close_loop_args_uses_full_root_virtualizable_array_capacity() {
        use pyre_interpreter::pyframe::PyFrame;

        ensure_test_callbacks();
        let code = compile_exec("len(x)").expect("test code should compile");
        let mut frame = PyFrame::new(code);
        locals_w_mut!(frame)[0] = w_int_new(41);
        locals_w_mut!(frame)[1] = w_int_new(7);
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;
        let array_len =
            concrete_frame_array_len(frame_ptr).expect("frame should expose locals_cells_stack_w");
        assert!(
            array_len >= 2,
            "test frame must have stack capacity beyond nlocals"
        );

        let n = crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
        let mut input_types = vec![
            Type::Ref, // frame
            Type::Ref, // ec
            Type::Int, // next_instr
            Type::Ref, // pycode
            Type::Int, // valuestackdepth
            Type::Ref, // debugdata
        ];
        input_types.extend(std::iter::repeat(Type::Ref).take(array_len));
        let mut ctx = TraceCtx::for_test_types(&input_types);

        // Mint typed `OpRef::input_arg_*` matching each `input_types`
        // slot — production `init_vable_indices` (over
        // `virtualizable_gen.rs`'s `inputargs`) always selects `InputArgInt`
        // / `InputArgRef` (resoperation.py/739) per the static-field
        // tag, so variant-aware Eq (`OpRef`'s `PartialEq`) requires the
        // matching variant here too.
        let mut sym = PyreSym::new_uninit(OpRef::input_arg_ref(0));
        sym.execution_context = OpRef::input_arg_ref(1);
        sym.nlocals = 1;
        sym.valuestackdepth = 1;
        sym.vable_last_instr = OpRef::input_arg_int(2);
        sym.vable_pycode = OpRef::input_arg_ref(3);
        sym.vable_valuestackdepth = OpRef::input_arg_int(4);
        sym.vable_debugdata = OpRef::input_arg_ref(5);
        sym.frame_w_globals = ctx.const_ref(frame.get_w_globals() as usize as i64);
        sym.registers_r.replace(vec![OpRef::input_arg_ref(6)]);
        sym.symbolic_local_types = vec![Type::Ref];
        sym.symbolic_stack_types = Vec::new();
        sym.concrete_vable_ptr = frame_ptr as *mut u8;
        // Bind a populated jitcode (real PyCode = frame.pycode) so
        // any sym.jitcode/payload read is well-formed; the single live Ref
        // color 0 maps to local0 in registers_r.
        let jc_ptr = bind_real_jitcode(&mut sym, frame.pycode, &[], &[0], &[]);
        // Seed ctx.virtualizable_array_lengths with the full root array
        // capacity so close_loop_args' target_array_capacity equals the
        // concrete array length (the nlocals+stack_only fallback would only
        // yield 1). Empty input_values leaves InputArg boxes unstamped; the
        // non-owner path then reads local0 from registers_r and PY_NULLs the
        // dead stack tail (trace_opcode.rs).
        seed_virtualizable_boxes(
            &mut ctx,
            OpRef::input_arg_ref(0),
            majit_ir::Value::Ref(majit_ir::GcRef(frame_ptr)),
            &[
                OpRef::input_arg_int(2),
                OpRef::input_arg_ref(3),
                OpRef::input_arg_int(4),
                OpRef::input_arg_ref(5),
            ],
            &[OpRef::input_arg_ref(6)],
            array_len,
            &[],
            std::ptr::null(),
        );

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            residual_call_pc: None,
            loop_close_marker_jit_pc: None,
            close_merge_point_vsd: None,
            orgpc: 0,
            concrete_frame_addr: frame_ptr,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        let jump_args = state.with_ctx(|this, ctx| this.close_loop_args_at(ctx, None, None));

        assert_eq!(
            jump_args.len(),
            n + array_len,
            "root close_loop_args must carry full virtualizable array capacity"
        );
        assert!(
            jump_args[n..].iter().all(|arg| !arg.is_none()),
            "full-capacity root jump args must materialize missing stack slots"
        );

        unsafe {
            let _ = Box::from_raw(jc_ptr);
        }
    }

    /// `init_symbolic` must not replace `initialize_virtualizable`'s
    /// `read_boxes` layout with the live valuestackdepth prefix. That shrink
    /// left `append_virtualizable_boxes` one slot short of the LABEL
    /// registered from `inputarg_types`.
    #[test]
    fn init_symbolic_does_not_shrink_an_already_seeded_virtualizable_array() {
        ensure_test_callbacks();
        // Seed a full `read_boxes` layout, then call init_symbolic with no
        // concrete frame so the live_prefix fallback would be 0.
        let array_len = 8;
        let mut input_types = vec![
            Type::Ref,
            Type::Ref,
            Type::Int,
            Type::Ref,
            Type::Int,
            Type::Ref,
        ];
        input_types.extend(std::iter::repeat(Type::Ref).take(array_len));
        let mut ctx = TraceCtx::for_test_types(&input_types);
        let mut sym = PyreSym::new_uninit(OpRef::input_arg_ref(0));
        sym.become_active_vable_owner();
        let array_items: Vec<OpRef> = (0..array_len)
            .map(|i| OpRef::input_arg_ref(6 + i as u32))
            .collect();
        seed_virtualizable_boxes(
            &mut ctx,
            OpRef::input_arg_ref(0),
            majit_ir::Value::Ref(majit_ir::GcRef(0x1000)),
            &[
                OpRef::input_arg_int(2),
                OpRef::input_arg_ref(3),
                OpRef::input_arg_int(4),
                OpRef::input_arg_ref(5),
            ],
            &array_items,
            array_len,
            &[],
            std::ptr::null(),
        );
        let seeded = ctx.virtualizable_boxes_len();
        // concrete_frame=0: array_len would fall back to live_prefix=0
        // without keeping the already-seeded layout.
        sym.init_symbolic(&mut ctx, 0);
        assert_eq!(
            ctx.virtualizable_boxes_len(),
            seeded,
            "init_symbolic must not shrink initialize_virtualizable's read_boxes layout"
        );
        assert_eq!(
            ctx.virtualizable_array_lengths()
                .map(|l| l.iter().copied().sum::<usize>()),
            Some(array_len),
        );
        assert_eq!(
            ctx.virtualizable_data_boxes().len(),
            crate::virtualizable_gen::NUM_VABLE_SCALARS + array_len,
        );
    }

    /// A walk that resumed from a guard leaves `PyFrame.valuestackdepth` at
    /// the resume PC's depth. Closing on a merge point at a different offset
    /// must recover that offset's own depth, or every operand-stack slot the
    /// header binds is force-nulled into the JUMP.
    #[test]
    fn merge_point_stack_depth_recovers_the_header_depth() {
        use pyre_interpreter::pyframe::PyFrame;

        ensure_test_callbacks();
        let code = compile_exec("for i in x:\n    pass\n").expect("test code should compile");
        let mut frame = PyFrame::new(code);
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;
        let w_code = frame.pycode as usize;
        let base = concrete_nlocals(frame_ptr).expect("frame should expose a stack base");
        let array_len =
            concrete_frame_array_len(frame_ptr).expect("frame should expose locals_cells_stack_w");

        // A `for` body puts the iterator on the stack, so some offset has a
        // deeper static depth than the empty-stack base.
        let (deep_pc, deep_vsd) = (0..array_len * 64)
            .find_map(|pc| {
                depth_based_vsd_for_wcode(w_code, pc)
                    .filter(|&vsd| vsd > base)
                    .map(|vsd| (pc, vsd))
            })
            .expect("a for-loop body must have an offset with a non-empty operand stack");
        let shallow_pc = (0..array_len * 64)
            .find(|&pc| depth_based_vsd_for_wcode(w_code, pc) == Some(base))
            .expect("a for-loop must have an offset with an empty operand stack");

        // Stale frame (resumed at an empty-stack offset), closing on the
        // deeper header: recover the header's depth.
        frame.valuestackdepth = base;
        assert_eq!(
            merge_point_stack_depth_to_recover(frame_ptr, deep_pc),
            Some(deep_vsd),
        );

        // Same frame, header at its own depth: nothing to recover.
        assert_eq!(
            merge_point_stack_depth_to_recover(frame_ptr, shallow_pc),
            None,
        );

        // Stale frame (resumed at a deeper offset), closing on the shallower
        // header: slots above the header's static depth are dead capacity.
        frame.valuestackdepth = deep_vsd + 1;
        assert_eq!(
            merge_point_stack_depth_to_recover(frame_ptr, deep_pc),
            Some(deep_vsd),
        );

        // An offset the code object has no liveness entry for keeps the frame.
        frame.valuestackdepth = base;
        assert_eq!(
            merge_point_stack_depth_to_recover(frame_ptr, usize::MAX),
            None,
        );

        // A null frame cannot answer either.
        assert_eq!(merge_point_stack_depth_to_recover(0, deep_pc), None);
    }
}

// ── Virtualizable configuration ──────────────────────────────────────
//
// PyPy's `pypy/interpreter/pyframe.py` declares:
//
//     _virtualizable_ = ['locals_stack_w[*]', 'valuestackdepth',
//                         'last_instr', ...]
//
// Our Rust equivalent uses explicit byte offsets instead of name-based
// introspection. The JIT optimizer's Virtualize pass uses this info
// to keep frame fields in CPU registers, eliminating heap accesses
// for LoadFast/StoreFast and stack push/pop during compiled code.
//
// The shared frame layout contract now also lives in `pyre-jit/src/frame_layout.rs`
// so the tracer can compute the same offsets without depending on
// `pyre-interpreter`. Driver registration still happens in `pyre-jit/src/eval.rs`.

/// TODO: deferred `MetaInterp.perform_call`
/// (`rpython/jit/metainterp/pyjitpl.py`) for pyre.  RPython constructs
/// and pushes the callee `MIFrame` directly inside `perform_call`; pyre
/// returns this struct from the trace step so the framestack mutation
/// happens after the trace handler releases its borrow on
/// `MetaInterpFrame`.  No upstream counterpart.
pub struct PendingInlineFrame {
    pub sym: PyreSym,
    pub green_key: u64,
    /// Raw `(code_ptr, target_pc)` greenkey components for element-
    /// wise recursion-depth comparison. `green_key` above is the u64
    /// hash derived from this pair and stays the identity key for
    /// HashMap lookups; `green_key_raw` element-wise equality matches
    /// rpython/jit/metainterp/pyjitpl.py:1396-1401 `for i in
    /// range(len(gk)): if not gk[i].same_constant(greenboxes[i])`
    /// without the hash-collision risk a u64-only comparison carries.
    pub green_key_raw: (usize, usize),
    /// opencoder.py capture_resumedata: accumulated parent frame chain.
    pub parent_frames: Vec<ResumeFrameState>,
    pub nargs: usize,
    pub caller_result_stack_idx: Option<usize>,
    pub caller_result_type: Option<Type>,
    /// Symbolic callable + arg OpRefs of the CALL that pushed this frame.
    /// Consumed by the inline back-edge CALL_ASSEMBLER path
    /// (`do_recursive_call`) to shape the GuardNotForced /
    /// GuardNoException resume via `push_call_replay_stack` — on guard
    /// failure the parent re-executes the CALL, mirroring the residual
    /// call capture. `OpRef::NONE` when the path is unavailable
    /// (bridge-reconstructed frames).
    pub replay_callable: OpRef,
    pub replay_args: Vec<OpRef>,
}

/// Reify one Ref-bank recipe slot as the boxed `W_Root` pointer that
/// `locals_cells_stack_w` (a W_Root array, virtualizable.py read_boxes) must
/// hold. `reconstruct_inline_recipe` gates int/float banks out of multi-
/// frame recipes, so a live slot is always `Value::Ref` (the decoded box,
/// possibly null) and a dead slot is `Value::Void`; the int/float arms only
/// exist defensively (box back to a heap object) and never fire in practice.
fn recipe_slot_to_pyobj(v: majit_ir::Value) -> PyObjectRef {
    match v {
        majit_ir::Value::Ref(gc) => gc.0 as PyObjectRef,
        majit_ir::Value::Void => pyre_object::PY_NULL,
        majit_ir::Value::Int(i) => w_int_new(i),
        majit_ir::Value::Float(f) => pyre_object::w_float_new(f),
    }
}

/// Recover the callee module's globals OBJECT for a reconstructed inline frame.
///
/// The globals-stamped `PyCode` wrapper is recovered from the raw code pointer
/// through the `code_ptr → live wrapper` registry. Returns `PY_NULL` when no
/// live wrapper is registered or it carries no globals yet — the callers
/// (`reconstruct_inline_recipe` and `assemble_bridge_inline_pending`) treat a
/// null result as "decline the multi-frame path".
///
/// Reading the globals off the code object is `PyFrame.get_w_globals`
/// (pyframe.py): with no `FrameDebugData` override it returns
/// `jit.promote(self.pycode).w_globals`, the namespace `frame_stores_global`
/// (pycode.py) recorded first. The override case never reaches here: the
/// forward inline declines a callee whose function globals differ from that
/// first-seen namespace (`w_code_frame_stores_global` in `inline_call.rs`),
/// so every callee a bridge can reconstruct carries the code's own globals.
/// Recover the callee's `W_Code` OBJECT for a reconstructed inline frame.
///
/// Sibling of [`recover_inline_callee_globals`], reading the same
/// `code_ptr → live wrapper` registry.  A reconstructed frame's `pycode` red
/// has to be this object, not the raw code pointer the recipe carries: every
/// consumer treats it as a `W_Code` (constant pool reads, `PyTraceback` node
/// construction).  Returns `PY_NULL` when no live wrapper is registered.
pub(crate) fn recover_inline_callee_code(code_ptr: *const ()) -> pyre_object::PyObjectRef {
    let live = pyre_interpreter::live_code_wrapper(code_ptr);
    if live.is_null() {
        return pyre_object::PY_NULL;
    }
    live
}

pub(crate) fn recover_inline_callee_globals(code_ptr: *const ()) -> pyre_object::PyObjectRef {
    let live = pyre_interpreter::live_code_wrapper(code_ptr);
    if !live.is_null() {
        let globals = unsafe { pyre_interpreter::w_code_get_w_globals(live) };
        if !globals.is_null() {
            return globals;
        }
    }
    pyre_object::PY_NULL
}

/// Assemble one decoded inline-callee [`ReconstructRecipe`]
/// into a [`PendingInlineFrame`] (concrete `PyFrame` + symbolic `PyreSym`)
/// for `trace_bytecode` to push onto the bridge framestack.
///
/// Mirrors the retired inline-frame builder's IR-FREE FAST branch: set
/// `jitcode` FIRST (the
/// `setup_kind_register_banks` debug_assert requires a non-null jitcode),
/// then fill the per-bank register files + the lazy-boxing concrete shadow.
/// It emits NO trace IR — the forward FAST branch's guard_value /
/// callee-frame helper / GUARD_NO_EXCEPTION belong to a LIVE caller making a
/// call; a reconstructed suspended frame is simply pushed.
///
/// The callee frame's globals come from its OWN pycode (`recipe.code_ptr`),
/// matching `pyframe.py get_w_globals_storage()` where a frame's globals
/// derive from its promoted pycode rather than the caller; a cross-module
/// inlined callee's LOAD_GLOBAL then resolves against the callee module's
/// namespace. `execution_context` is the per-thread singleton and comes from
/// the bridge's root concrete frame.
///
/// `parent_frames` is the OUTER chain (immediate parent first) the drain
/// builds from the framestack.  The caller result slot travels in this
/// struct's own `caller_result_stack_idx` / `caller_result_type`; nothing
/// stamps `parent_frames.first().pending_result_*`, which stays `None`.
pub(crate) fn assemble_bridge_inline_pending(
    ctx: &mut TraceCtx,
    is_being_profiled: bool,
    recipe: &ReconstructRecipe,
    execution_context: *const pyre_interpreter::PyExecutionContext,
    parent_frames: Vec<ResumeFrameState>,
) -> PendingInlineFrame {
    let nlocals = recipe.nlocals;
    let valuestackdepth = recipe.valuestackdepth;

    // Recover the callee's globals-stamped PyCode wrapper from the
    // `code_ptr -> live-wrapper` registry. It is the SAME wrapper the forward
    // inline push obtains via `getcode(callable)` (raw code identity is 1:1
    // with its wrapper), so keying the reconstructed frame's greenkey by it
    // preserves the recursion-depth / inline-position identity.
    let w_code = pyre_interpreter::live_code_wrapper(recipe.code_ptr) as *const ();

    // pyframe.py get_w_globals get_w_globals_storage(): a frame's globals come from its OWN
    // pycode (`jit.promote(self.pycode).w_globals`), not the caller. Resolve
    // the callee's globals OBJECT for `recipe.code_ptr` — the same
    // `pycode.w_globals` the callee module exposes through its function's
    // `w_func_globals_obj` — so a cross-module inlined callee's LOAD_GLOBAL
    // sees the callee module's namespace. `reconstruct_inline_recipe` aborts
    // the multi-frame path when the callee code has no resolved globals
    // object, so this is non-null here.
    let w_globals = recover_inline_callee_globals(recipe.code_ptr);

    // resume.py rebuild_from_resumedata newframe + reload: the decoded boxes are reloaded
    // into the symbolic frame below. `setup_reconstructed_callee_frame`
    // supplies the matching recording-time PyFrame for residual execution.
    // Symbolic side: mirror the FAST branch field-for-field.
    let mut sym = PyreSym::new_uninit(OpRef::NONE);
    sym.nlocals = nlocals;
    sym.valuestackdepth = valuestackdepth;
    // jitcode FIRST — setup_kind_register_banks debug_asserts non-null.
    sym.jitcode = jitcode_for(w_code);
    // The Ref bank IS the unified locals+stack register file decoded by
    // `reconstruct_inline_recipe`; int/float banks are empty (gated out).
    sym.registers_i = recipe.registers_i.clone();
    sym.registers_r.replace(recipe.registers_r.clone());
    sym.registers_f = recipe.registers_f.clone();
    // locals_cells_stack_w is a W_Root array — every live slot is Ref.
    sym.symbolic_local_types = vec![Type::Ref; nlocals];
    sym.symbolic_stack_types = vec![Type::Ref; valuestackdepth - nlocals];
    // Box-identity shadow: unbox per the FAST branch, preserving PY_NULL
    // frame slots as Ref(PY_NULL) so uninitialized locals stay distinct from
    // untracked values.
    sym.concrete_locals = (0..nlocals)
        .map(|k| concrete_value_from_slot(recipe_slot_to_pyobj(recipe.concrete_r[k])))
        .collect();
    sym.concrete_stack = (nlocals..valuestackdepth)
        .map(|k| concrete_value_from_slot(recipe_slot_to_pyobj(recipe.concrete_r[k])))
        .collect();
    sym.concrete_namespace = w_globals;
    sym.frame_w_globals = ctx.const_ref(w_globals as usize as i64);
    sym.concrete_execution_context = execution_context;
    // perform_call threads the caller's `ec` down to every inlined callee
    // (reds=['frame','ec'], interp_jit.py:67), so the callee shares the
    // caller's ExecutionContext OpRef. Seed `sym.execution_context` from the
    // immediate parent's symbolic ec (the root caller's `setup_bridge_sym`
    // resolved it from `portal_ec_reg`) so an in-callee
    // `ensure_execution_context` returns it directly. `sym.frame` is NONE for
    // a reconstructed inline frame, so without this seed the recovery path
    // would emit `GetfieldGcR(NONE)` (an unbacked VoidOp arg) and fail
    // bridge regalloc.
    if let Some(parent) = parent_frames.first() {
        let parent_ec = unsafe { (*parent.sym).execution_context };
        if !parent_ec.is_none() {
            sym.execution_context = parent_ec;
        }
    }
    // pyjitpl.py MIFrame.setup: size the per-kind banks + copy_constants.
    // The constant tail lands at `[num_regs_X..]`, beyond the live
    // valuestackdepth prefix (`num_regs_r` is the full Ref register file),
    // so the live slots set above are preserved. `locals_cells_stack_array_
    // ref` stays NONE: dead (NONE) slots are never read — the liveness stream
    // only carries live values, so no heap fall-through is needed (and
    // `s.frame` is NONE, so emitting the array getfield would be invalid).
    sym.setup_kind_register_banks(ctx);

    PendingInlineFrame {
        sym,
        // The reconstructed frame represents the same inlined call the
        // forward trace pushed at function entry; match its (code, 0)
        // greenkey identity for recursion-depth + inline-position tracking.
        green_key: crate::driver::make_green_key(w_code, 0, is_being_profiled),
        green_key_raw: (w_code as usize, 0),
        parent_frames,
        nargs: recipe.nargs,
        caller_result_stack_idx: None,
        caller_result_type: Some(Type::Ref),
        // Reconstructed frames carry no CALL-site OpRefs, and the inline
        // back-edge CALL_ASSEMBLER path is gated out for them anyway.
        replay_callable: OpRef::NONE,
        replay_args: Vec::new(),
    }
}

/// Build the per-frame setup for ONE reconstructed bridge-carrier callee so
/// the full-body walker can drive it (issue #215 item 2):
///
///   - emit the frame vable seeded with the recipe's LOCALS (slots
///     `0..nlocals`); the live operand-stack temps stay in the abstract
///     register file, seeded into `argboxes_r` below.  `valuestackdepth`
///     passed to the vable builder is `nlocals`, matching the forward-inline
///     callee (the stack is rebuilt in registers as the body runs);
///   - assemble the symbolic + concrete frame
///     (`assemble_bridge_inline_pending`) and bind `sym.frame` to the emitted
///     vable so the portal body reads its locals through `getarrayitem_vable`
///     (a NONE `sym.frame` would take the nonstandard-vable Void leg);
///   - seed the walk's initial register file: frame/ec at their portal-red
///     colors + the live stack temps at `[nlocals..valuestackdepth)` (the same
///     semantic-prefix convention the root bridge seeding uses, trace.rs).
///
/// Returns `(pending, argboxes_r)`; `None` when the callee body/layout is
/// unavailable. The frame's trace ops were recorded by
/// [`record_rebuilt_callee_frames`] before `prepare_resume_from_failure`
/// (`rebuild_from_resumedata`, then the guard). This binds that box and
/// the concrete `PyFrame`. Emitting another allocation after the exception
/// guard would name an op the guard's snapshot has not recorded yet.
/// Recipe checks [`setup_reconstructed_callee_frame`] applies before it
/// allocates a concrete `PyFrame`. The carrier drain uses this so a
/// catching middle that cannot be reconstructed declines before any
/// `carrier_ec_leave` shortens `topframeref`.
fn recipe_records_callee_frame(recipe: &ReconstructRecipe) -> bool {
    !recipe.len_tail
        && recipe.return_substitute.is_none()
        && reconstructed_callee_recipe_is_portable(recipe)
}

/// `registers_r` are the paused root's OpRefs, whose trace concrete can still
/// read as a stale NULL once reused as the reconstructed callee's locals,
/// while `concrete_r[k]` holds the value captured at guard failure. A
/// residual call consuming such a local folds its Ref arg to concrete NULL
/// and the walk stops on `MayForceNullRefArgUnsupported`, so no bridge is
/// ever built and the guard re-deopts forever. Restamp each local from its
/// captured value; a NULL/Void capture is an unmaterialized hole and is left
/// alone, matching how `setup_bridge_sym` seeds the root frame's slots.
fn restamp_reconstructed_callee_prefix(
    ctx: &mut TraceCtx,
    recipe: &ReconstructRecipe,
    stack_base: usize,
) {
    for (k, &opref) in recipe.registers_r[..stack_base].iter().enumerate() {
        if opref.is_none() {
            continue;
        }
        let Some(&captured) = recipe.concrete_r.get(k) else {
            continue;
        };
        if matches!(
            captured,
            majit_ir::Value::Void | majit_ir::Value::Ref(majit_ir::GcRef(0))
        ) {
            continue;
        }
        ctx.try_set_opref_concrete(opref, captured);
    }
}

fn emit_reconstructed_callee_pyframe(
    ctx: &mut TraceCtx,
    recipe: &ReconstructRecipe,
) -> Option<OpRef> {
    if !reconstructed_callee_recipe_is_portable(recipe) {
        return None;
    }
    let raw_code = recipe.code_ptr as *const pyre_interpreter::CodeObject;
    let code_ref = unsafe { &*raw_code };
    let (stack_base, frame_array_size) = callee_layout_for_call_assembler(code_ref);
    let nlocals = code_ref.varnames.len();
    let w_code = pyre_interpreter::live_code_wrapper(recipe.code_ptr) as *const ();
    let w_globals = recover_inline_callee_globals(recipe.code_ptr);
    let pycode_const = ctx.const_ref(w_code as i64);
    let w_globals_const = ctx.const_ref(w_globals as i64);
    let locals_boxes: Vec<OpRef> = recipe.registers_r[..nlocals].to_vec();
    let freevar_cells: Vec<OpRef> = recipe.registers_r[nlocals..stack_base].to_vec();
    // The operands live at the guard, at their `locals_cells_stack_w` slots:
    // the walk resumes mid-body and reads the frame image there (a callee
    // loop header's FOR_ITER reads its iterator from the stack slot).
    let stack_items: Vec<OpRef> = recipe.registers_r[stack_base..recipe.valuestackdepth].to_vec();
    restamp_reconstructed_callee_prefix(ctx, recipe, stack_base);
    crate::helpers::emit_new_pyframe_inline_with_params(
        ctx,
        &locals_boxes,
        &freevar_cells,
        &stack_items,
        nlocals,
        frame_array_size,
        stack_base,
        pycode_const,
        w_globals_const,
    )
}

/// Record one `PyFrame` per rebuilt callee of an exception-guard bridge,
/// before `prepare_resume_from_failure` records that guard. `len` tails
/// and `descr_call`'s return substitute build no frame.
fn record_rebuilt_callee_frames(
    ctx: &mut TraceCtx,
    ec_box: OpRef,
    recipes: &mut [ReconstructRecipe],
) {
    if !recipes.iter().any(recipe_records_callee_frame) {
        return;
    }
    let mut ec_seed = if !ec_box.is_none() {
        Some(ec_box)
    } else {
        None
    };
    for recipe in recipes.iter_mut() {
        if !recipe_records_callee_frame(recipe) {
            continue;
        }
        let Some(frame) = emit_reconstructed_callee_pyframe(ctx, recipe) else {
            continue;
        };
        let ec = match ec_seed {
            Some(ec) => ec,
            None => {
                let ec = crate::helpers::emit_current_execution_context(
                    ctx,
                    "ExecutionContext::ReconstructedCallee",
                );
                ec_seed = Some(ec);
                ec
            }
        };
        recipe.rebuilt_frame = Some(frame);
        recipe.rebuilt_ec = Some(ec);
    }
}

pub(crate) fn reconstructed_callee_recipe_is_portable(recipe: &ReconstructRecipe) -> bool {
    let raw_code = recipe.code_ptr as *const pyre_interpreter::CodeObject;
    if raw_code.is_null() {
        return false;
    }
    let code_ref = unsafe { &*raw_code };
    let (stack_base, _frame_array_size) = callee_layout_for_call_assembler(code_ref);
    let valuestackdepth = recipe.valuestackdepth;
    if recipe.nlocals != stack_base
        || recipe.registers_r.len() < valuestackdepth
        || stack_base > valuestackdepth
    {
        return false;
    }
    let (frame_reg, ec_reg) = portal_red_regs_at(recipe.jitcode_index);
    frame_reg != u16::MAX && ec_reg != u16::MAX
}

/// The resumed callee's own frame, when its resume section carried one the
/// walk can run on.
///
/// `resume.py consume_boxes` refills every register of the rebuilt `MIFrame`,
/// the `frame` red included, so the callee keeps running on the frame the
/// parent trace built: the one its `virtual_ref` scope names and its callees'
/// `f_backref` reach.  A second frame built here would be the one a traceback
/// records while `ExecutionContext.topframeref` still names the first, so the
/// traceback frame's `f_back` reads as `None` and a callee's `f_back` names the
/// other copy.
///
/// The walk runs the frame the way [`setup_reconstructed_callee_frame`]'s own
/// constructor would, so it is left in the same state that constructor leaves
/// its frame in: the locals/cells prefix stored through the heap cache, with
/// `valuestackdepth` at the recipe's captured depth. Operand-stack temps stay
/// as the deopt restore left them.  `None` — the caller builds a frame — when
/// the level carried no frame red, or when that frame is not a block the walk
/// may hold across an allocation (a nursery object moves; see `FrameBox::new`).
fn resume_reconstructed_callee_frame(
    ctx: &mut TraceCtx,
    recipe: &ReconstructRecipe,
    w_code: *const (),
    stack_base: usize,
) -> Option<(OpRef, *mut pyre_interpreter::PyFrame)> {
    let frame = recipe.frame;
    if frame.is_none() {
        return None;
    }
    let Some(majit_ir::Value::Ref(frame_ref)) = ctx.concrete_of_opref(frame) else {
        return None;
    };
    if frame_ref.is_null() || frame_ref == majit_ir::GcRef::NO_CONCRETE {
        return None;
    }
    if majit_gc::gc_is_nursery_object(frame_ref.as_usize()) {
        return None;
    }
    let concrete_frame = frame_ref.as_usize() as *mut pyre_interpreter::PyFrame;
    let arr_ptr = unsafe {
        if (*concrete_frame).pycode != w_code {
            return None;
        }
        (*concrete_frame).locals_cells_stack_w
    };
    if arr_ptr.is_null() || unsafe { (*arr_ptr).len() } < stack_base {
        return None;
    }
    store_reconstructed_callee_array_image(
        ctx,
        frame,
        concrete_frame,
        recipe,
        &recipe.concrete_r,
        stack_base,
    );
    Some((frame, concrete_frame))
}

/// Store the locals/cells prefix in `concrete_r` into a concrete `PyFrame`
/// the walk is about to run. Operand-stack temps stay as the deopt restore
/// left them: a mid-CALL `-live-` drops them from the recipe, so writing that
/// image would publish `ConstPtr(0)` into callable/arg slots. `valuestackdepth`
/// stays the recipe's captured depth so a blackhole that resumes the CALL
/// opcode still sees those restored operands.
///
/// `concrete_r` is the image to write now: a minor collection between capture
/// and this store can move young objects, so the caller passes the current
/// words (re-read from roots taken before the allocation) rather than the
/// addresses captured in `recipe.concrete_r`.
fn store_reconstructed_callee_array_image(
    ctx: &mut TraceCtx,
    frame: OpRef,
    concrete_frame: *mut pyre_interpreter::PyFrame,
    recipe: &ReconstructRecipe,
    concrete_r: &[majit_ir::Value],
    stack_base: usize,
) {
    let arr_ptr = unsafe { (*concrete_frame).locals_cells_stack_w };
    if arr_ptr.is_null() {
        return;
    }
    let arr_len = unsafe { (*arr_ptr).len() };
    let store_end = stack_base.min(arr_len).min(recipe.registers_r.len());
    let locals_array = frame_locals_cells_stack_array(ctx, frame);
    ctx.try_set_opref_concrete(
        locals_array,
        majit_ir::Value::Ref(majit_ir::GcRef(arr_ptr as usize)),
    );
    // Keyed like `emit_new_pyframe_inline_with_params`: the walk reads the
    // locals through `getarrayitem_vable`, whose descr is the virtualizable
    // info's array item descr.
    let array_descr = pyobject_gcarray_descr();
    let heapcache_item_descr_index = ctx
        .virtualizable_info()
        .map(|info| info.array_item_descr(0).index())
        .unwrap_or_else(|| array_descr.index());
    for k in 0..store_end {
        let value = recipe.registers_r[k];
        if !value.is_none() {
            let idx = ctx.const_int(k as i64);
            ctx.record_op_with_descr(
                OpCode::SetarrayitemGc,
                &[locals_array, idx, value],
                array_descr.clone(),
            );
            ctx.heapcache_setarrayitem(locals_array, idx, heapcache_item_descr_index, value);
        }
        if let Some(&majit_ir::Value::Ref(gc)) = concrete_r.get(k)
            && !gc.is_null()
            && gc != majit_ir::GcRef::NO_CONCRETE
        {
            unsafe { (*arr_ptr).set_ref(k, gc.as_usize() as pyre_object::PyObjectRef) };
        }
    }
    // The frame may already be old, and the stores above can hand it young
    // references.
    frame_array_write_barrier(concrete_frame as *mut u8, arr_ptr);
    let published_vsd = recipe
        .valuestackdepth
        .min(arr_len)
        .max(stack_base.min(arr_len));
    let vsd = ctx.const_int(published_vsd as i64);
    let vsd_descr = crate::descr::pyframe_stack_depth_descr();
    let vsd_idx = vsd_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[frame, vsd], vsd_descr);
    ctx.heapcache_setfield_cached(frame, vsd_idx, vsd);
    unsafe { (*concrete_frame).valuestackdepth = published_vsd };
}

pub(crate) fn setup_reconstructed_callee_frame(
    ctx: &mut TraceCtx,
    is_being_profiled: bool,
    recipe: &ReconstructRecipe,
    execution_context: *const pyre_interpreter::PyExecutionContext,
    ec_box: OpRef,
    parent_frames: Vec<ResumeFrameState>,
) -> Option<(PendingInlineFrame, Vec<OpRef>)> {
    if !reconstructed_callee_recipe_is_portable(recipe) {
        return None;
    }
    let raw_code = recipe.code_ptr as *const pyre_interpreter::CodeObject;
    let code_ref = unsafe { &*raw_code };
    let (stack_base, _frame_array_size) = callee_layout_for_call_assembler(code_ref);
    let nlocals = code_ref.varnames.len();
    let valuestackdepth = recipe.valuestackdepth;
    let (frame_reg, ec_reg) = portal_red_regs_at(recipe.jitcode_index);

    let w_code = pyre_interpreter::live_code_wrapper(recipe.code_ptr) as *const ();
    let w_globals = recover_inline_callee_globals(recipe.code_ptr);
    restamp_reconstructed_callee_prefix(ctx, recipe, stack_base);
    // The resume section's own frame, when it is a block the walk can keep.
    // Building a second frame would split the traceback `f_back` chain.
    let resumed = if w_code.is_null() {
        None
    } else {
        resume_reconstructed_callee_frame(ctx, recipe, w_code, stack_base)
    };
    // The box recorded before `prepare_resume_from_failure`. A missing box
    // after that guard must not be allocated here: the snapshot would name it.
    let frame_vable = if let Some((frame, _)) = resumed {
        frame
    } else if let Some(frame) = recipe.rebuilt_frame {
        frame
    } else if ctx.bridge_exception_resume_prepared() {
        return None;
    } else {
        emit_reconstructed_callee_pyframe(ctx, recipe)?
    };
    // `PyPyJitDriver.reds = ['frame', 'ec']`: `perform_call` gives an inlined
    // callee the caller's own `ec` Box. Prefer the box paired with the
    // pre-guard frame. A bridge whose resume data held no value at the portal
    // `ec` color arrives with an empty `ec_box` (`bridge_ec_missing`). Read
    // the thread's own context instead — a ConstPtr built from the concrete
    // pointer would bake the recording thread's ExecutionContext into every
    // bridge compiled from this loop.
    let ec_seed = if let Some(ec) = recipe.rebuilt_ec {
        ec
    } else if !ec_box.is_none() {
        ec_box
    } else if ctx.bridge_exception_resume_prepared() {
        return None;
    } else {
        crate::helpers::emit_current_execution_context(ctx, "ExecutionContext::ReconstructedCallee")
    };

    // A callee whose code object has no live wrapper cannot receive a
    // concrete frame; its residual calls and the blackhole preflight would
    // both decline on the unset frame pointer, so decline here.
    if w_code.is_null() {
        return None;
    }
    let mut live_stack_concrete: Option<Vec<majit_ir::Value>> = None;
    let concrete_frame_ptr = if let Some((_, ptr)) = resumed {
        ptr
    } else {
        // `perform_call` (`pyjitpl.py`) is three lines — `newframe` +
        // `setup_call` + `raise ChangeFrame` — and `newframe` builds an
        // `MIFrame` and nothing else: upstream has no recording-time app-level
        // frame to hand out here, because the app-level frame appears only as
        // ordinary traced code when PyPy traces the interpreter's own frame
        // construction.  pyre does not trace that construction, so the concrete
        // `PyFrame` an inlined callee needs has to be built explicitly.
        // The forward-inline callee does so with a GC-managed `FrameBox`
        // whose pointer is stamped onto the emitted vable
        // (`try_walker_inline_resolved_user_call_inner` in `inline_call.rs`).  The
        // reconstructed carrier callee needs the same object: its residual calls
        // run through `execute_inline_residual_call(frame, nargs)`, and the abort
        // image names
        // this framestack level by its frame pointer.  A vable with no concrete
        // makes both decline — the residual call on its frame argument, and the
        // blackhole preflight on the unset frame pointer — so the drain aborts
        // after its sub-walk already executed a side effect and the rollback to the
        // guard then replays it.
        // `perform_call` (pyjitpl.py) and a bridge resume
        // (resume.py) both allocate the callee MIFrame and then fill its
        // boxes — `newframe(jitcode)` + `setup_call` / `consume_boxes`. Neither
        // allocates an app-level frame; upstream never needs to, because there the
        // app-level frame is built by traced interpreter code. pyre has to supply
        // it. The emitted `frame_vable` above is the runtime half; give
        // it the matching recording-time PyFrame so residuals that consume the
        // frame red can execute while tracing. Without this value the very first
        // such residual sees a symbolic-only Ref and the carrier sub-walk aborts.
        //
        // Root the code, globals, and every reconstructed prefix slot before the
        // closure tuple/frame allocations. This is the same shadow-stack shape as
        // the forward-inline constructor in `inline_call.rs`; the resulting
        // GC-managed FrameBox relinquishes only its host handle after the frontend
        // op becomes its trace root.
        let concrete_roots = pyre_object::gc_roots::push_roots();
        let root_base = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_code as pyre_object::PyObjectRef);
        let _ = pyre_object::gc_roots::pin_root(w_globals as pyre_object::PyObjectRef);
        for (slot, &captured) in recipe.concrete_r[..stack_base].iter().enumerate() {
            let value = match captured {
                majit_ir::Value::Ref(gc) if gc != majit_ir::GcRef::NO_CONCRETE => {
                    if slot >= nlocals && slot < stack_base && gc.is_null() {
                        return None;
                    }
                    gc.as_usize() as pyre_object::PyObjectRef
                }
                majit_ir::Value::Void => pyre_object::PY_NULL,
                _ => return None,
            };
            let _ = pyre_object::gc_roots::pin_root(value);
        }
        let closure = if stack_base == nlocals {
            pyre_object::PY_NULL
        } else {
            let cells = (nlocals..stack_base)
                .map(|i| pyre_object::gc_roots::shadow_stack_get(root_base + 2 + i))
                .collect();
            let closure = pyre_object::w_tuple_new(cells);
            let closure = pyre_object::gc_roots::pin_root(closure);
            closure
        };
        let closure_root = (stack_base != nlocals).then_some(root_base + 2 + stack_base);
        let concrete_locals: Vec<pyre_object::PyObjectRef> = (0..nlocals)
            .map(|i| pyre_object::gc_roots::shadow_stack_get(root_base + 2 + i))
            .collect();
        let current_code = pyre_object::gc_roots::shadow_stack_get(root_base) as *const ();
        let current_globals = pyre_object::gc_roots::shadow_stack_get(root_base + 1);
        let current_closure = closure_root
            .map(pyre_object::gc_roots::shadow_stack_get)
            .unwrap_or(closure);
        for k in stack_base..valuestackdepth.min(recipe.concrete_r.len()) {
            let value = match recipe.concrete_r[k] {
                majit_ir::Value::Ref(gc) if gc != majit_ir::GcRef::NO_CONCRETE => {
                    gc.as_usize() as pyre_object::PyObjectRef
                }
                _ => pyre_object::PY_NULL,
            };
            let _ = pyre_object::gc_roots::pin_root(value);
        }
        let mut concrete_frame = pyre_interpreter::pyframe::FrameBox::new(
            pyre_interpreter::pyframe::PyFrame::new_for_call_with_closure_and_globals_obj(
                current_code,
                &concrete_locals,
                current_globals,
                execution_context,
                current_closure,
                pyre_interpreter::pyframe::FrameLocalsArrayAllocation::NurseryGc,
            ),
        );
        // The `drop(concrete_frame)` below relinquishes only the host handle for a
        // GC-owned frame. `FrameBox::new` falls back to `new_boxed` whenever the
        // stable allocator answers null, and that box's `drop` FREES the frame —
        // which would leave the `pending.sym.concrete_vable_ptr` published further
        // down dangling for the rest of the recording. Decline rather than publish
        // a pointer this scope is about to invalidate; the `FrameBox` then frees
        // itself on the way out, as it should.
        if !concrete_frame.is_gc_owned() {
            return None;
        }
        let concrete_frame_ptr = concrete_frame.as_mut_ptr();
        ctx.set_opref_concrete(
            frame_vable,
            majit_ir::Value::Ref(majit_ir::GcRef(concrete_frame_ptr as usize)),
        );
        // FrameBox::new (and w_tuple_new) may have moved young objects; the
        // pinned slots hold the forwarded addresses, recipe.concrete_r does not.
        let live_prefix: Vec<majit_ir::Value> = (0..stack_base)
            .map(|k| {
                majit_ir::Value::Ref(majit_ir::GcRef(pyre_object::gc_roots::shadow_stack_get(
                    root_base + 2 + k,
                ) as usize))
            })
            .collect();
        let stack_temp_root = closure_root
            .map(|i| i + 1)
            .unwrap_or(root_base + 2 + stack_base);
        let mut live_stack = vec![majit_ir::Value::Void; valuestackdepth];
        for k in stack_base..valuestackdepth.min(recipe.concrete_r.len()) {
            live_stack[k] = match recipe.concrete_r[k] {
                majit_ir::Value::Ref(_) => majit_ir::Value::Ref(majit_ir::GcRef(
                    pyre_object::gc_roots::shadow_stack_get(stack_temp_root + (k - stack_base))
                        as usize,
                )),
                other => other,
            };
        }
        store_reconstructed_callee_array_image(
            ctx,
            frame_vable,
            concrete_frame_ptr,
            recipe,
            &live_prefix,
            stack_base,
        );
        live_stack_concrete = Some(live_stack);
        drop(concrete_frame);
        drop(concrete_roots);
        concrete_frame_ptr
    };

    let mut pending = assemble_bridge_inline_pending(
        ctx,
        is_being_profiled,
        recipe,
        execution_context,
        parent_frames,
    );
    pending.sym.frame = frame_vable;
    pending.sym.concrete_vable_ptr = concrete_frame_ptr as *mut u8;
    // The portal reds are [frame, ec], force-alive at every pc; a guard snapshot
    // (`collect_outer_active_boxes`) reads ec at portal_ec_reg via
    // `sym.execution_context`.  `assemble_bridge_inline_pending` only seeds it
    // from `parent_frames.first()`, so seed it directly here for the
    // walker-driven callee (an unset ec surfaces as a liveness-active NONE
    // panic at the first in-callee guard).
    if pending.sym.execution_context.is_none() {
        pending.sym.execution_context = ec_seed;
    }

    let max_reg = valuestackdepth
        .max(frame_reg as usize + 1)
        .max(ec_reg as usize + 1);
    let mut argboxes_r: Vec<OpRef> = vec![OpRef::NONE; max_reg];
    argboxes_r[frame_reg as usize] = frame_vable;
    argboxes_r[ec_reg as usize] = ec_seed;
    // The recipe's `registers_r` is SEMANTIC-slot-indexed (filled from the
    // frame vable's `locals_cells_stack_w` array), but the re-executed callee
    // reads its registers by post-rename COLOR. After stack-slot-pinning
    // removal a live operand-stack slot's color is per-program-point
    // (`pcdep_color_slots`) and need not equal `nlocals + d` — the encoder
    // (`get_list_of_active_boxes`) maps color→slot via
    // `semantic_ref_slot_for_reg_color`, so invert it here to land each stack
    // value at the color the dispatcher will touch. Placing it at the raw slot
    // index (the old identity assumption) lands it under the wrong register and
    // leaves the true color still holding the portal-red seed (frame/ec) whose
    // color the register allocator reused for this stack slot. Runs AFTER the
    // frame/ec seeding so a reused color resolves to the live stack value.
    // Falls back to identity when no live color owns the slot (empty map /
    // non-diverging coloring).
    let pcdep = pcdep_trivia_at(recipe.jitcode_index, recipe.jitcode_pc).unwrap_or_default();
    // The allocate path re-reads Refs from roots taken before FrameBox::new;
    // the resume path captured with no allocation in between, so
    // recipe.concrete_r is still current.
    let slot_concrete = live_stack_concrete.as_deref().unwrap_or(&recipe.concrete_r);
    for k in stack_base..valuestackdepth {
        let opref = recipe.registers_r[k];
        if opref.is_none() {
            continue;
        }
        let color = match reconstructed_ref_slot_color(&pcdep, k) {
            Some(color) => color,
            // An entirely uncolored skeleton/fixture retains the historical
            // identity layout.  Once a per-PC color map exists, however, an
            // unmapped semantic slot has no register at this coordinate (for
            // example PUSH_NULL's call sentinel).  Treating its slot number as
            // a color can overwrite a different mapped value: `_richcmp`'s
            // `[op, NULL, lhs, rhs]` resume otherwise writes the NULL sentinel
            // over `op`'s color and the bridge calls a null callable.
            None => continue,
        };
        if color >= argboxes_r.len() {
            argboxes_r.resize(color + 1, OpRef::NONE);
        }
        argboxes_r[color] = opref;
        // `box.value` parity at the frame boundary (mirror `ref_return/r`):
        // thread the reconstructed slot's concrete onto the OpRef-keyed shadow
        // so the re-executed callee's speculation gates
        // see the live value's class. The
        // emitted specialization stays runtime-correct (guard_class + int_op on
        // the unboxed register); the concrete is only the tracing shadow, as for
        // any resumed operand. Skips constants (`constants.get_value` is
        // authoritative) and non-value (`Void`) slots.
        if !opref.is_constant() {
            if let Some(v @ majit_ir::Value::Int(_)) = slot_concrete.get(k).copied() {
                ctx.set_opref_concrete(opref, v);
            } else if let Some(v @ majit_ir::Value::Ref(gc)) = slot_concrete.get(k).copied()
                && !gc.is_null()
                && gc != majit_ir::GcRef::NO_CONCRETE
            {
                // A NULL recipe concrete is an unmaterialized frame-image
                // hole, not evidence that the symbolic box is Python NULL.
                // Keep any stronger fact already attached to the OpRef (for
                // example the non-null callable pinned by GUARD_VALUE), just
                // as the locals-prefix restamp above does.
                ctx.set_opref_concrete(opref, v);
            }
        }
    }

    Some((pending, argboxes_r))
}

pub fn execute_inline_residual_call(
    frame: &mut pyre_interpreter::pyframe::PyFrame,
    nargs: usize,
) -> Result<(), pyre_interpreter::PyError> {
    let required = nargs + 2; // callable + null/self + args
    if frame.valuestackdepth < frame.stack_base() + required {
        return Err(pyre_interpreter::PyError::type_error(
            "inline residual call stack underflow",
        ));
    }

    let mut args = Vec::with_capacity(nargs);
    for _ in 0..nargs {
        args.push(frame.pop());
    }
    args.reverse();
    let _null_or_self = frame.pop();
    let callable = frame.pop();
    let result = pyre_interpreter::call::call_callable_inline_residual(frame, callable, &args)?;
    frame.push(result);
    Ok(())
}

// inline_trace_and_execute / trace_through_callee removed — the trait
// meta-interpreter that replaced them (PyreMetaInterp.interpret() +
// push_inline_frame) is itself retired (#203); the FBW walker
// handles both root and inline frames.

/// `pypy/objspace/std/listobject.py is_plain_int1` parity.
///
/// IntegerListStrategy stores raw i64 and so cannot preserve the
/// W_*Object pointer identity of its elements. PyPy therefore demotes
/// to object strategy on insertion of any value whose exact type is
/// not `W_IntObject` — bools, int subclasses (PyPy `W_IntObjectUser`
/// from `interpreter/typedef.py subcls`), and `W_LongObject` whose
/// value doesn't fit in a machine int.
///
/// Delegates to the single `pyre_object::is_plain_int1` helper that
/// already implements the full upstream predicate including the
/// int-subclass rejection (`listobject.rs`).
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn int_strategy_preserves_identity(value: pyre_object::PyObjectRef) -> bool {
    unsafe { pyre_object::is_plain_int1(value) }
}

/// Test-only RAII guard: swap a `MetaInterpStaticData` into the
/// thread-local `METAINTERP_SD` and restore the previous value on drop,
/// so tests sharing a thread (`--test-threads=1`) do not observe each
/// other's registrations.
#[cfg(test)]
pub(crate) struct MetainterpSdGuard {
    prev: Option<MetaInterpStaticData>,
}

#[cfg(test)]
impl MetainterpSdGuard {
    pub(crate) fn swap(sd: MetaInterpStaticData) -> Self {
        let prev = METAINTERP_SD.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), sd));
        Self { prev: Some(prev) }
    }
}

#[cfg(test)]
impl Drop for MetainterpSdGuard {
    fn drop(&mut self) {
        if let Some(prev) = self.prev.take() {
            METAINTERP_SD.with(|slot| *slot.borrow_mut() = prev);
        }
    }
}

#[cfg(test)]
mod indirectcalltargets_tests {
    //! Line-by-line parity tests for `pyjitpl.py setup_indirectcalltargets` and
    //! `pyjitpl.py bytecode_for_address`.  Tests exercise the local
    //! `MetaInterpStaticData` methods directly — independent of the
    //! thread-local `METAINTERP_SD` singleton so concurrent callers
    //! (and unrelated tests that use the thread-local) do not alias.
    use super::{
        MetaInterpStaticData, MetainterpSdGuard, pyjitcode_for_jitcode_index,
        raw_code_for_jitcode_index, walk_jitcode_constants_refs_in,
    };
    use majit_metainterp::jitcode::{JitCode, JitCodeBuilder};
    use pyre_interpreter::bytecode::CodeObject;
    use std::sync::Arc;

    fn make_jitcode_with_fnaddr(fnaddr: usize) -> Arc<JitCode> {
        let mut jc = JitCodeBuilder::default().finish();
        jc.fnaddr = fnaddr as i64;
        Arc::new(jc)
    }

    /// Freshly-constructed staticdata has no indirect-call targets.
    /// Delegates through `canonical`; matches the behaviour of
    /// `majit_metainterp::MetaInterpStaticData::new()` where the
    /// `MetaInterpGlobalData` inside starts with every lazy cache
    /// (`indirectcall_dict`, `addr2name`) set to `None`.
    #[test]
    fn bytecode_for_address_none_when_empty() {
        let mut sd = MetaInterpStaticData::new();
        assert!(sd.bytecode_for_address(0xdeadbeef).is_none());
    }

    /// `pyjitpl.py bytecode_for_address` hit path: registered fnaddrs resolve to
    /// their JitCode.
    #[test]
    fn bytecode_for_address_returns_jitcode_when_registered() {
        let mut sd = MetaInterpStaticData::new();
        let j100 = make_jitcode_with_fnaddr(0x100);
        let j200 = make_jitcode_with_fnaddr(0x200);
        let j300 = make_jitcode_with_fnaddr(0x300);
        sd.setup_indirectcalltargets(vec![j100.clone(), j200.clone(), j300.clone()]);

        assert!(Arc::ptr_eq(&sd.bytecode_for_address(0x100).unwrap(), &j100));
        assert!(Arc::ptr_eq(&sd.bytecode_for_address(0x200).unwrap(), &j200));
        assert!(Arc::ptr_eq(&sd.bytecode_for_address(0x300).unwrap(), &j300));
        assert!(sd.bytecode_for_address(0x400).is_none());
    }

    /// `pyjitpl.py` `setup_indirectcalltargets` parity:
    /// every call replaces the targets list and invalidates the lazy
    /// dict so the next lookup rebuilds from the new list.
    #[test]
    fn setup_indirectcalltargets_invalidates_cache() {
        let mut sd = MetaInterpStaticData::new();
        sd.setup_indirectcalltargets(vec![make_jitcode_with_fnaddr(0x100)]);
        assert!(sd.bytecode_for_address(0x100).is_some());
        sd.setup_indirectcalltargets(vec![
            make_jitcode_with_fnaddr(0x200),
            make_jitcode_with_fnaddr(0x300),
        ]);
        assert!(sd.bytecode_for_address(0x100).is_none());
        assert!(sd.bytecode_for_address(0x200).is_some());
        assert!(sd.bytecode_for_address(0x300).is_some());
    }

    fn make_code(source: &str) -> (*const (), *const CodeObject) {
        let raw_code = pyre_interpreter::compile_exec(source).expect("source must compile");
        let code = pyre_interpreter::w_code_new(Box::into_raw(Box::new(raw_code)) as *const ())
            as *const ();
        let raw_code = unsafe {
            pyre_interpreter::w_code_get_ptr(code as pyre_object::PyObjectRef) as *const CodeObject
        };
        (code, raw_code)
    }

    fn populated_pyjit(raw_code: *const CodeObject) -> Arc<crate::PyJitCode> {
        let mut pyjit = crate::PyJitCode::skeleton(raw_code);
        // A drained PerCodeObject install has non-empty `code` and
        // `is_drained` set; give the fixture both so it is not classified
        // as a skeleton (`is_skeleton()` now tests `code.is_empty()`).
        let runtime_jc = majit_metainterp::jitcode::JitCode::new("populated_pyjit_test");
        runtime_jc.set_body(majit_jitcode::jitcode::JitCodeBody {
            code: vec![majit_metainterp::jitcode::insns::BC_LIVE, 0, 0],
            startpoints: Some([0_usize].into_iter().collect()),
            ..Default::default()
        });
        pyjit.jitcode = Arc::new(runtime_jc);
        pyjit.metadata.is_drained = true;
        Arc::new(pyjit)
    }

    fn pyjit_with_ref_constant(value: usize) -> Arc<crate::PyJitCode> {
        let runtime_jc = majit_metainterp::jitcode::JitCode::new("ref_constant_test");
        runtime_jc.set_body(majit_jitcode::jitcode::JitCodeBody {
            code: vec![majit_metainterp::jitcode::insns::BC_LIVE, 0, 0],
            constants_r: vec![(value as i64).into()],
            startpoints: Some([0_usize].into_iter().collect()),
            ..Default::default()
        });
        let mut pyjit = crate::PyJitCode::skeleton(std::ptr::null());
        pyjit.jitcode = Arc::new(runtime_jc);
        Arc::new(pyjit)
    }

    /// `incminimark.py old_objects_pointing_to_young`: publishing a JitCode
    /// remembers its off-GC constant pool for exactly the next minor. A major
    /// still follows the complete `MetaInterpStaticData.jitcodes` graph.
    #[test]
    fn jitcode_constants_minor_walk_consumes_publication_remembered_set() {
        use majit_gc::shadow_stack::{ExtraRootWalkKind, set_extra_root_walk_kind};

        let mut sd = MetaInterpStaticData::new();
        sd.jitcodes.push(Box::new(super::JitCode {
            index: 0,
            payload: pyjit_with_ref_constant(0x1000),
        }));
        sd.remember_jitcode_constants_write(0);

        set_extra_root_walk_kind(ExtraRootWalkKind::Minor);
        let mut minor_seen = 0;
        walk_jitcode_constants_refs_in(&sd, &mut |slot| {
            minor_seen += 1;
            slot.0 = 0x2000;
        });
        assert_eq!(minor_seen, 1);
        assert_eq!(sd.jitcodes[0].payload.jitcode.constants_r[0].get(), 0x2000);

        walk_jitcode_constants_refs_in(&sd, &mut |_| minor_seen += 1);
        assert_eq!(minor_seen, 1, "clean pool is absent from the next minor");

        set_extra_root_walk_kind(ExtraRootWalkKind::Major);
        let mut major_seen = 0;
        walk_jitcode_constants_refs_in(&sd, &mut |_| major_seen += 1);
        assert_eq!(major_seen, 1, "major walks every live JitCode pool");
    }

    #[should_panic(expected = "make_jitcodes returned an unpopulated JitCode skeleton")]
    #[test]
    fn install_jitcodes_rejects_skeleton_payload() {
        let mut sd = MetaInterpStaticData::new();
        let (_code, raw_code) = make_code("x = 1\n");
        let skeleton = Arc::new(crate::PyJitCode::skeleton(raw_code));
        sd.set_jitcodes_from_make_result(vec![skeleton]);
    }

    #[test]
    fn compiled_jitcode_lookup_returns_populated_entry() {
        let mut sd = MetaInterpStaticData::new();
        let (code, raw_code) = make_code("x = 1\n");
        sd.set_jitcodes_from_make_result(vec![populated_pyjit(raw_code)]);

        let hit = sd
            .compiled_jitcode_lookup(code)
            .expect("populated payload should be installed by make_jitcodes");
        // Runtime entries append above the reserved build-time index space, so
        // read the slot back through the entry's own index.
        let index = unsafe { (*hit).index } as usize;
        assert!(std::ptr::eq(sd.jitcodes[index].as_ref(), hit));
    }

    #[test]
    fn compiled_jitcode_lookup_scans_by_raw_code_identity() {
        let mut sd = MetaInterpStaticData::new();
        let (code, raw_code) = make_code("x = 1\n");
        sd.set_jitcodes_from_make_result(vec![populated_pyjit(raw_code)]);

        let hit = sd
            .compiled_jitcode_lookup(code)
            .expect("wrapper pointer should resolve through raw CodeObject identity");
        assert_eq!(unsafe { (*hit).raw_code() }, raw_code);
    }

    #[test]
    fn raw_code_for_jitcode_index_returns_canonical_graph_pointer() {
        let mut sd = MetaInterpStaticData::new();
        let (_code, expected_raw) = make_code("x = 1\n");
        sd.set_jitcodes_from_make_result(vec![populated_pyjit(expected_raw)]);
        // The install appends above the reserved build-time index space.
        let index = sd.jitcodes.len() as i32 - 1;
        let _sd_guard = MetainterpSdGuard::swap(sd);

        let hit = raw_code_for_jitcode_index(index).expect("installed jitcode index must resolve");
        assert_eq!(hit, expected_raw);
    }
}

#[derive(Clone, Copy)]
pub struct ResumeFrameState {
    pub sym: *mut PyreSym,
    pub concrete_frame_addr: usize,
    pub resume_pc: usize,
    /// Jitcode-space twin of the pc word this frame will encode (gh#369).
    pub resume_marker_jit_pc: Option<usize>,
    /// Python pc of the CALL whose residual call this frame is the caller
    /// of (the call still on this frame's stack when the callee was
    /// inlined).  When that call sits in a try-block the jitcode emits a
    /// post-call `-live-`/`catch_exception` keyed by this pc
    /// (the post-call catch-marker twin); on a guard that deopts mid-callee
    /// the blackhole must resume this frame AT that catch
    /// (`blackhole.py handle_exception_in_frame`,
    /// `pyjitpl.py:2601-2602`).  `None` for frames whose call has no catch
    /// marker.
    pub call_pc: Option<usize>,
    /// pyjitpl.py `get_list_of_active_boxes(in_a_call=True)`.
    /// Non-top frames clear the caller's pending result slot before
    /// snapshotting liveness so the undefined call result does not leak
    /// stale boxes into guard fail_args.
    pub pending_result_stack_idx: Option<usize>,
    pub pending_result_type: Option<Type>,
}

#[cfg(test)]
mod finish_setup_tests {
    use super::{MetaInterpStaticData, MetainterpSdGuard, blackhole_control_opcodes};
    use crate::assembler::publish_state;

    #[test]
    fn finish_setup_refreshes_opcode_cache_after_initial_empty_snapshot() {
        let mut sd = MetaInterpStaticData::new();
        let empty: indexmap::IndexMap<String, u8> = indexmap::IndexMap::new();
        sd.finish_setup_if_needed(&empty, Vec::new());
        assert_eq!(sd.op_live, u8::MAX);
        assert_eq!(sd.op_goto, u8::MAX);

        let mut insns = indexmap::IndexMap::new();
        insns.insert("live/".to_string(), 88u8);
        insns.insert("goto/L".to_string(), 16u8);
        insns.insert("catch_exception/L".to_string(), 89u8);
        insns.insert("rvmprof_code/ii".to_string(), 91u8);
        insns.insert("int_return/i".to_string(), 148u8);
        insns.insert("ref_return/r".to_string(), 76u8);
        insns.insert("float_return/f".to_string(), 149u8);
        insns.insert("void_return/".to_string(), 150u8);

        sd.finish_setup_if_needed(&insns, vec![1, 2, 3]);
        assert_eq!(sd.op_live, 88);
        assert_eq!(sd.op_goto, 16);
        assert_eq!(sd.op_catch_exception, 89);
        assert_eq!(sd.op_rvmprof_code, 91);
        assert_eq!(sd.op_int_return, 148);
        assert_eq!(sd.op_ref_return, 76);
        assert_eq!(sd.op_float_return, 149);
        assert_eq!(sd.op_void_return, 150);
        assert_eq!(sd.liveness_info.snapshot_arc().as_ref(), &[1u8, 2, 3][..]);
        assert!(sd.finish_setup_done);
    }

    #[test]
    fn blackhole_control_opcodes_reflect_finish_setup_cache() {
        let mut sd = MetaInterpStaticData::new();
        let mut insns = indexmap::IndexMap::new();
        insns.insert("live/".to_string(), 88u8);
        insns.insert("catch_exception/L".to_string(), 89u8);
        insns.insert("rvmprof_code/ii".to_string(), 91u8);
        sd.finish_setup_if_needed(&insns, Vec::new());
        let _sd_guard = MetainterpSdGuard::swap(sd);

        assert_eq!(blackhole_control_opcodes(), (88, 89, 91));
    }

    #[test]
    fn blackhole_control_opcodes_refresh_after_initial_empty_snapshot() {
        let mut sd = MetaInterpStaticData::new();
        let empty: indexmap::IndexMap<String, u8> = indexmap::IndexMap::new();
        sd.finish_setup_if_needed(&empty, Vec::new());
        let _sd_guard = MetainterpSdGuard::swap(sd);

        let mut insns = indexmap::IndexMap::new();
        insns.insert("live/".to_string(), 88u8);
        insns.insert("catch_exception/L".to_string(), 89u8);
        insns.insert("rvmprof_code/ii".to_string(), 91u8);
        publish_state(&insns, &[], 0, 0);

        assert_eq!(blackhole_control_opcodes(), (88, 89, 91));
    }
}
