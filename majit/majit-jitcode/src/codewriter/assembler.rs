//! `Assembler` state that the JIT runtime keeps: the opcode table, the
//! shared liveness table, the descr pool and the inline-jitcode cache.
//!
//! RPython equivalent: `rpython/jit/codewriter/assembler.py` class
//! `Assembler`. The methods that read a flattened graph (`assemble` and
//! its helpers) stay in `majit-translate` as `AssemblerExt`.

use std::fmt;

use vecset::VecSet;

use super::flatten::Label;
use majit_ir::CallInfoCollection;

/// Non-canonical tag marking a deferred prebuilt-string slot in
/// `constants_r`.  x86-64 user addresses occupy `0..2^48`, so this high-word
/// pattern can never alias a real GCREF / host-static address; the low 48
/// bits carry the [`super::jitcode::StrConstDescriptor`] ordinal.  The runtime load pass
/// overwrites every such slot with a live immortal STR address before the
/// jitcode is used, so the sentinel is never dereferenced (a non-canonical
/// deref would fault, surfacing any missed patch immediately).
pub const STR_CONST_SENTINEL_BASE: i64 = 0x7E57_0000_0000_0000u64 as i64;

/// Non-canonical tag marking a deferred unit-variant singleton slot in
/// `constants_r`, disjoint from [`STR_CONST_SENTINEL_BASE`] in the same
/// non-canonical high-word space; the low 48 bits carry the
/// [`super::jitcode::UnitVariantConstDescriptor`] ordinal.
pub const UNIT_VARIANT_CONST_SENTINEL_BASE: i64 = 0x7E58_0000_0000_0000u64 as i64;

/// Non-canonical tag marking a deferred prebuilt exception-instance slot
/// in `constants_r`, disjoint from the string and unit-variant sentinels.
/// The low 48 bits carry the descriptor ordinal.
pub const EXC_INSTANCE_CONST_SENTINEL_BASE: i64 = 0x7E59_0000_0000_0000u64 as i64;

/// Non-canonical tag marking a deferred type-static slot in
/// `constants_r`, disjoint from the string, unit-variant, and
/// exception-instance bases. The low 48 bits carry the
/// [`super::jitcode::TypeStaticConstDescriptor`] ordinal.
///
/// A `new_with_vtable` size descr stores the same tag in
/// `BhDescr::Size.vtable` (ordinal into [`Assembler::intern_type_static_addrs`])
/// and the shared name in `owner`. The runtime descr load replaces the tag
/// with the address that name publishes. The base stays clear of the
/// exception-instance sentinel.
pub const TYPE_STATIC_CONST_SENTINEL_BASE: i64 = 0x7E5A_0000_0000_0000u64 as i64;

/// `TYPE_STATIC_CONST_SENTINEL_BASE | ordinal`.
pub fn type_static_const_sentinel(ordinal: usize) -> i64 {
    debug_assert!(
        (ordinal as u64) < (1u64 << 48),
        "too many type-static constants"
    );
    TYPE_STATIC_CONST_SENTINEL_BASE | ordinal as i64
}

/// Whether `bits` is a [`type_static_const_sentinel`] rather than an address.
///
/// User addresses occupy `0..2^48`, so the high half-word of the base never
/// aliases one.
pub fn is_type_static_const_sentinel(bits: u64) -> bool {
    const HIGH: u64 = 0xFFFF_0000_0000_0000;
    bits & HIGH == (TYPE_STATIC_CONST_SENTINEL_BASE as u64) & HIGH
}

/// Non-canonical tag marking a deferred zero-length array constant in
/// `constants_r`, disjoint from the type-static base. The low 48 bits
/// carry the [`super::jitcode::EmptyArrayConstDescriptor`] ordinal.
///
/// `Assembler.emit_const` puts the `lltype` pointer of a prebuilt
/// zero-length `GcArray` in `constants_r` (`ListRepr.convert_const` /
/// `_ll_prebuilt_empty_array`). The translator cannot allocate that
/// block, so the slot holds this sentinel until the load pass writes
/// the immortal empty array.
pub const EMPTY_ARRAY_CONST_SENTINEL_BASE: i64 = 0x7E5B_0000_0000_0000u64 as i64;

/// `EMPTY_ARRAY_CONST_SENTINEL_BASE | ordinal`.
pub fn empty_array_const_sentinel(ordinal: usize) -> i64 {
    debug_assert!(
        (ordinal as u64) < (1u64 << 48),
        "too many empty-array constants"
    );
    EMPTY_ARRAY_CONST_SENTINEL_BASE | ordinal as i64
}

/// Whether `bits` is an [`empty_array_const_sentinel`] rather than an address.
pub fn is_empty_array_const_sentinel(bits: u64) -> bool {
    const HIGH: u64 = 0xFFFF_0000_0000_0000;
    bits & HIGH == (EMPTY_ARRAY_CONST_SENTINEL_BASE as u64) & HIGH
}

/// RPython `class AssemblerError(Exception)` (assembler.py).
///
/// Upstream raises this for unsupported constant kinds while assembling
/// SSARepr (`assembler.py:124-126`). Most Rust assembler paths currently
/// fail through panics because they are internal translation invariants,
/// but this carrier keeps the public codewriter surface aligned for
/// call sites that need a typed assembler diagnostic.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct AssemblerError(pub String);

impl AssemblerError {
    pub fn message<S: Into<String>>(message: S) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for AssemblerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for AssemblerError {}

/// Assembler — converts SSARepr to JitCode.
///
/// RPython: `assembler.py::Assembler`.
///
/// The assembler maintains state across multiple JitCode assemblies
/// (shared descriptor table, liveness encoding, etc.)
pub struct Assembler {
    /// RPython: Assembler.insns — map {opcode_key: opcode_number}
    pub insns: indexmap::IndexMap<String, u8>,
    /// Next candidate for the translator-only `setdefault` fallback
    /// (`assembler.py:220`). RPython grows `self.insns` densely from
    /// zero; pyre keeps canonical / extension `BC_*` bytes reserved for
    /// build/runtime stability, so this cursor scans upward from zero
    /// and skips only those reserved bytes plus already-assigned
    /// translator-only bytes.
    pub dynamic_byte_cursor: u16,
    /// RPython: Assembler.descrs — list of descriptors. Inline-call
    /// descriptors keep the callee JitCode object until the final
    /// snapshot, where `jitcode.index` is guaranteed to be assigned.
    pub descrs: Vec<AssemblerDescr>,
    /// RPython: Assembler._descr_dict — descriptor to descrs[] index.
    /// Upstream `assembler.py:26` + `:197-203` keeps a Python dict to
    /// deduplicate AbstractDescr objects before emitting the two-byte 'd'
    /// operand; the no-HashMap house rule replaces the dict with a
    /// IndexMap linear-scan lookup.
    pub descr_dict: indexmap::IndexMap<AssemblerDescrKey, usize>,
    /// RPython: `Assembler.indirectcalltargets` — merged `IndirectCallTargets`
    /// sidecars from every `residual_call` emitted during assembly
    /// (`assembler.py:208-209`).  RPython stores `JitCode` objects; we
    /// store their jitcode indices because codewriter owns the
    /// jitcode-index allocator.
    /// RPython `assembler.py` `self.indirectcalltargets.update(x.lst)`:
    /// a `set` of JitCode objects (Python identity dedup). pyre uses
    /// `JitCodeHandle` as the identity-keyed wrapper around
    /// `Arc<JitCode>` so the same shells handed out by
    /// `CallControl::get_jitcode` survive into the metainterp side
    /// without copying.
    pub indirectcalltargets: std::collections::HashSet<crate::jitcode::JitCodeHandle>,
    /// RPython: Assembler.list_of_addr2name — (addr, name) pairs for debugging.
    /// In majit: (target_path, name) string pairs since we don't have raw addresses.
    pub list_of_addr2name: Vec<(String, String)>,
    /// RPython: Assembler._count_jitcodes
    pub count_jitcodes: usize,
    /// RPython: Assembler._seen_raw_objects — dedup set for see_raw_object.
    pub seen_raw_objects: std::collections::HashSet<String>,
    /// RPython: Assembler.all_liveness — shared liveness table.
    /// Encoded as bytes: [count_i, count_r, count_f, reg_indices...].
    /// Deduplicated across all JitCodes via all_liveness_positions.
    pub all_liveness: Vec<u8>,
    /// RPython: Assembler.all_liveness_length (assembler.py).
    pub all_liveness_length: usize,
    /// RPython: Assembler.all_liveness_positions — dedup cache.
    /// Maps (live_i set, live_r set, live_f set) → offset in all_liveness.
    #[expect(
        clippy::type_complexity,
        reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
    )]
    pub all_liveness_positions: indexmap::IndexMap<(VecSet<u8>, VecSet<u8>, VecSet<u8>), usize>,
    /// Length of the build-time liveness prefix installed by an embedded
    /// JitCode table; zero until one is installed.
    pub embedded_liveness_prefix_len: usize,
    /// RPython: Assembler.num_liveness_ops (assembler.py).
    pub num_liveness_ops: usize,
    /// State-field JIT canonical "all-live" liveness triple, set once at
    /// `__JitMeta_<fn>::install_canonical_liveness` time (RPython
    /// `assembler.py get_liveness_info` flat-state adaptation).
    /// `JitCodeBuilder::live_placeholder` defers patching of the leading
    /// `BC_LIVE` slot at the start of every per-opcode JitCode until
    /// `finalize_liveness` runs, at which point this triple is registered
    /// via `_register_liveness_offset` (the result is cached in
    /// `canonical_liveness_offset`).
    ///
    /// RPython `assembler.assemble` itself has no concept of a canonical
    /// entry — it only emits `-live-` markers as it walks the IR.  The
    /// canonical entry exists in pyre because per-opcode JitCodes need a
    /// leading `BC_LIVE` to satisfy `code[orgpc - SIZE_LIVE_OP] == op_live`
    /// at JitCode entry; lazy registration via `live_placeholder` keeps
    /// the `all_liveness` order encounter-driven (matching RPython's
    /// IR-walk order) instead of pre-seeding canonical at offset 0.
    pub canonical_liveness_triple: Option<(Vec<u8>, Vec<u8>, Vec<u8>)>,
    /// Cached offset returned by the first `_register_liveness_offset`
    /// call against `canonical_liveness_triple`.
    pub canonical_liveness_offset: Option<usize>,
    /// Name of the graph currently being assembled, threaded through so
    /// diagnostic panics (e.g. missing regalloc coloring) can cite the
    /// exact function.  RPython tracks this via `self.jitcode.name`
    /// captured at `assembler.py:56 self.setup(ssarepr.name)`.
    pub current_graph_name: Option<String>,
    /// Pretty-printed FlatOp currently being encoded, only used by
    /// the `MAJIT_COVERAGE_PANIC=1` diagnostic so the missing-coloring
    /// panic can cite the offending op.
    pub current_flatop_debug: Option<String>,
    /// `call.py CallControl.jitcodes` — the per-graph cache `get_jitcode`
    /// consults before assembling anything.
    ///
    /// Upstream mints an empty JitCode and registers it under the graph
    /// BEFORE the body is written, so a graph that calls itself links to the
    /// object it is already registered under instead of assembling a second
    /// copy.  This is the same cache, keyed by a per-helper identity address,
    /// and it is what stops a self-recursive `#[jit_inline]` helper from
    /// recursing forever while its jitcode is being built.
    ///
    /// Type-erased because `majit-metainterp` depends on this crate and not the
    /// other way round; every value is an
    /// `Arc<majit_metainterp::jitcode::InlineJitCodeSlot>`.  It sits beside
    /// `indirectcalltargets`, which upstream also keeps as a JitCode-identity
    /// set on this object.
    pub inline_jitcodes: indexmap::IndexMap<usize, std::sync::Arc<dyn std::any::Any + Send + Sync>>,
    /// `call.py CallControl.unfinished_graphs` membership, for the install-time
    /// liveness prebuild walk.
    ///
    /// That walk is a second recursion with no `JitCodeBuilder` anywhere in
    /// scope: a helper's prebuild body calls its callees' prebuilds directly,
    /// so a recursive helper overflows there too, earlier and with nothing to
    /// catch it.  Membership here is what makes the walk visit each helper once.
    pub inline_prebuild_seen: indexmap::IndexSet<usize>,
    /// `(address, name)` rows for host `PyType` singletons.
    /// `assembler.py` `Assembler` keeps `constants_r` on the assembler;
    /// these rows name the sentinel that pool writes into that vector.
    type_static_by_addr: Vec<(i64, String)>,
    /// `(address, name)` rows for host-static refs (`HostStaticAddrs.refs`).
    /// Parallel to [`Self::type_static_by_addr`]: these rows name the
    /// `reloc_consts_r` descriptor the pool writes for a prebuilt GCREF.
    static_ref_by_addr: Vec<(i64, String)>,
}

impl Assembler {
    /// RPython: `Assembler.__init__()` (assembler.py:21-32).
    pub fn new() -> Self {
        Self {
            insns: indexmap::IndexMap::new(),
            dynamic_byte_cursor: 0,
            descrs: Vec::new(),
            descr_dict: indexmap::IndexMap::new(),
            indirectcalltargets: std::collections::HashSet::new(),
            list_of_addr2name: Vec::new(),
            count_jitcodes: 0,
            seen_raw_objects: std::collections::HashSet::new(),
            all_liveness: Vec::new(),
            all_liveness_length: 0,
            all_liveness_positions: indexmap::IndexMap::new(),
            embedded_liveness_prefix_len: 0,
            num_liveness_ops: 0,
            canonical_liveness_triple: None,
            canonical_liveness_offset: None,
            current_graph_name: None,
            current_flatop_debug: None,
            inline_jitcodes: indexmap::IndexMap::new(),
            inline_prebuild_seen: indexmap::IndexSet::new(),
            type_static_by_addr: Vec::new(),
            static_ref_by_addr: Vec::new(),
        }
    }

    /// Record host `PyType` singleton addresses so `emit_const_r` can emit a
    /// named sentinel instead of a translator-local pointer.  Duplicate
    /// addresses keep the first name.
    pub fn intern_type_static_addrs(&mut self, rows: &[(&str, i64)]) {
        for (name, addr) in rows {
            if *addr == 0 {
                continue;
            }
            if !self
                .type_static_by_addr
                .iter()
                .any(|(existing, _)| *existing == *addr)
            {
                self.type_static_by_addr.push((*addr, (*name).to_string()));
            }
        }
    }

    /// The name [`Self::intern_type_static_addrs`] recorded for `addr`.
    pub fn type_static_const_by_addr(&self, addr: i64) -> Option<&str> {
        if addr == 0 {
            return None;
        }
        self.type_static_by_addr
            .iter()
            .find(|(existing, _)| *existing == addr)
            .map(|(_, name)| name.as_str())
    }

    /// Record host-static ref addresses so `emit_const_r` can emit a
    /// named reloc descriptor instead of an untagged translator-local
    /// pointer. Duplicate addresses keep the first name.
    pub fn intern_static_ref_addrs(&mut self, rows: &[(&str, i64)]) {
        for (name, addr) in rows {
            if *addr == 0 {
                continue;
            }
            if !self
                .static_ref_by_addr
                .iter()
                .any(|(existing, _)| *existing == *addr)
            {
                self.static_ref_by_addr.push((*addr, (*name).to_string()));
            }
        }
    }

    /// The name [`Self::intern_static_ref_addrs`] recorded for `addr`.
    pub fn static_ref_by_addr(&self, addr: i64) -> Option<&str> {
        if addr == 0 {
            return None;
        }
        self.static_ref_by_addr
            .iter()
            .find(|(existing, _)| *existing == addr)
            .map(|(_, name)| name.as_str())
    }

    /// Wire spelling of a `new_with_vtable` type word.
    ///
    /// `Some((sentinel, name))` when `addr` was recorded by
    /// [`Self::intern_type_static_addrs`]. The sentinel is
    /// [`type_static_const_sentinel`] of that row's ordinal — the same
    /// encoding a `constants_r` type-static slot uses — and `name` is the
    /// key the runtime map resolves. `None` when `addr` is not a registered
    /// type static; the caller keeps the raw word.
    pub fn type_static_size_words(&self, addr: i64) -> Option<(u64, String)> {
        if addr == 0 {
            return None;
        }
        let (ordinal, name) = self
            .type_static_by_addr
            .iter()
            .enumerate()
            .find(|(_, (existing, _))| *existing == addr)
            .map(|(ordinal, (_, name))| (ordinal, name.clone()))?;
        Some((type_static_const_sentinel(ordinal) as u64, name))
    }

    /// `call.py CallControl.jitcodes.get(graph)` — the slot registered for
    /// `key`, or `None` if this helper has not been entered yet.
    ///
    /// The caller downcasts; this object cannot name the slot type without
    /// depending on the crate above it.
    pub fn inline_jitcode_slot(
        &self,
        key: usize,
    ) -> Option<&std::sync::Arc<dyn std::any::Any + Send + Sync>> {
        self.inline_jitcodes.get(&key)
    }

    /// `call.py CallControl.jitcodes[graph] = jitcode`.
    ///
    /// Called twice per helper: once with the under-construction slot before
    /// the body is assembled, once with the finished one after.
    pub fn inline_jitcode_insert(
        &mut self,
        key: usize,
        slot: std::sync::Arc<dyn std::any::Any + Send + Sync>,
    ) {
        self.inline_jitcodes.insert(key, slot);
    }

    /// Whether this is the first time the liveness prebuild walk has reached
    /// `key`.  `false` means the walk is already inside this helper and must
    /// not descend again.
    pub fn enter_inline_prebuild(&mut self, key: usize) -> bool {
        self.inline_prebuild_seen.insert(key)
    }

    /// Stage the state-field JIT canonical "all-live" triple for lazy
    /// registration by `ensure_canonical_liveness_offset`.  Called once
    /// per `__JitMeta_<fn>::install_canonical_liveness` invocation, before any
    /// per-pc JitCode is built.
    pub fn set_canonical_liveness_triple(
        &mut self,
        live_i: Vec<u8>,
        live_r: Vec<u8>,
        live_f: Vec<u8>,
    ) {
        self.canonical_liveness_triple = Some((live_i, live_r, live_f));
    }

    /// Lazily register the canonical triple via
    /// `_register_liveness_offset` (deduplicating against
    /// `all_liveness_positions`) and cache the resulting offset.  Subsequent
    /// calls return the cached offset.  Panics if the triple has not been
    /// staged via `set_canonical_liveness_triple`.
    pub fn ensure_canonical_liveness_offset(&mut self) -> usize {
        if let Some(off) = self.canonical_liveness_offset {
            return off;
        }
        let (li, lr, lf) = self
            .canonical_liveness_triple
            .clone()
            .expect("canonical_liveness_triple not staged before ensure_canonical_liveness_offset");
        let off = self._register_liveness_offset(&li, &lr, &lf);
        self.canonical_liveness_offset = Some(off);
        off
    }

    /// RPython: `Assembler.assemble` descriptor operand path
    /// (`assembler.py:197-207`).
    ///
    /// A descriptor is inserted into `descrs` only once and every later bytecode
    /// operand reuses the same two-byte index from `_descr_dict`.
    pub fn emit_descr(&mut self, descr: AssemblerDescr) -> usize {
        let key = AssemblerDescrKey::from_descr(&descr);
        if let Some(index) = self.descr_dict.get(&key) {
            return *index;
        }
        let index = self.descrs.len();
        assert!(index <= 0xFFFF, "too many AbstractDescrs!");
        self.descrs.push(descr);
        self.descr_dict.insert(key, index);
        index
    }

    pub fn emit_ready_descr(&mut self, descr: crate::jitcode::BhDescr) -> usize {
        self.emit_descr(AssemblerDescr::Ready(Box::new(descr)))
    }

    pub fn emit_pending_jitcode_descr(&mut self, jitcode: crate::jitcode::JitCodeHandle) -> usize {
        self.emit_descr(AssemblerDescr::PendingJitCode { jitcode })
    }

    pub fn emit_pending_switch_descr(&mut self, cases: Vec<(i64, Label)>) -> usize {
        let index = self.descrs.len();
        assert!(index <= 0xFFFF, "too many AbstractDescrs!");
        // RPython creates a fresh SwitchDictDescr per switch site; do
        // not route through `_descr_dict`, because labels are local to
        // the currently assembled JitCode.
        self.descrs.push(AssemblerDescr::PendingSwitch { cases });
        index
    }

    /// RPython `assembler.py` `_encode_liveness(live_i, live_r,
    /// live_f)` — register a `(live_i, live_r, live_f)` triple in the
    /// shared `all_liveness` table (deduplicating against
    /// `all_liveness_positions`) and append the 2-byte offset of the
    /// canonical entry into `code`.
    ///
    /// Mirrors RPython `assembler.py:235`
    /// `key = (frozenset(live_i), frozenset(live_r), frozenset(live_f))`:
    /// the cache key is set-valued, so callers may pass arbitrary-order
    /// or duplicated slices.  Each kind's effective payload is the
    /// sorted, deduplicated set, exactly as `liveness.py:148` `live =
    /// sorted(live)` produces during inner encoding.
    ///
    /// On a cache miss we append three header bytes
    /// (`len(live_i)`, `len(live_r)`, `len(live_f)`) followed by
    /// `encode_liveness` of each kind, exactly mirroring upstream
    /// `assembler.py:241-247` byte order.  The returned offset is
    /// finally written via `liveness::encode_offset` (parity with
    /// `liveness.py`).
    pub fn _encode_liveness(
        &mut self,
        live_i: &[u8],
        live_r: &[u8],
        live_f: &[u8],
        code: &mut Vec<u8>,
    ) {
        let pos = self._register_liveness_offset(live_i, live_r, live_f);
        // assembler.py `encode_offset(pos, self.code)`.
        crate::codewriter::liveness::encode_offset(pos, code);
    }

    /// Registration-only sibling of [`_encode_liveness`]: deduplicate the
    /// `(live_i, live_r, live_f)` triple into the shared `all_liveness`
    /// table and return the entry's offset, without writing the 2-byte
    /// `encode_offset` bytes anywhere.
    ///
    /// The `live/<offset>` 2-byte slot in a JitCode is patched by the
    /// caller via `JitCodeBuilder::patch_live_offset` once the offset is
    /// known.  Used by the deferred-patch path in
    /// `JitCodeBuilder::finalize_liveness` where
    /// the lowerer collects per-marker triples first, then registers and
    /// patches them in a single post-emission pass.
    pub fn _register_liveness_offset(
        &mut self,
        live_i: &[u8],
        live_r: &[u8],
        live_f: &[u8],
    ) -> usize {
        // frozenset(live_f))`.  `VecSet` is a Vec-backed sorted set, so
        // collecting the input into one yields the same canonical form
        // `frozenset` would have produced.
        let key = (
            live_i.iter().copied().collect::<VecSet<u8>>(),
            live_r.iter().copied().collect::<VecSet<u8>>(),
            live_f.iter().copied().collect::<VecSet<u8>>(),
        );
        if let Some(&cached) = self.all_liveness_positions.get(&key) {
            return cached;
        }
        let pos = self.all_liveness.len();
        // assembler.py `chr(len(live_i)) + chr(len(live_r)) + chr(len(live_f))`.
        // RPython `chr(N)` raises `ValueError` for N >= 256; Rust `as u8`
        // silently wraps. Strict assert mirrors the RPython failure mode
        // (`assembler.py:265` constants+regs <= 256 bound) so a regression
        // that emits a 256+-element bank surfaces here instead of being
        // mis-encoded into a low byte the decoder later misreads.
        let len_i = key.0.len();
        let len_r = key.1.len();
        let len_f = key.2.len();
        assert!(
            len_i < 256,
            "live_i length {len_i} exceeds u8; assembler.py:241 chr() would ValueError"
        );
        assert!(
            len_r < 256,
            "live_r length {len_r} exceeds u8; assembler.py:241 chr() would ValueError"
        );
        assert!(
            len_f < 256,
            "live_f length {len_f} exceeds u8; assembler.py:241 chr() would ValueError"
        );
        self.all_liveness.push(len_i as u8);
        self.all_liveness.push(len_r as u8);
        self.all_liveness.push(len_f as u8);
        // assembler.py:243-247 `for live in live_i, live_r, live_f:
        // liveness = encode_liveness(live); …`
        for live in [key.0.as_slice(), key.1.as_slice(), key.2.as_slice()] {
            let encoded = crate::codewriter::liveness::encode_liveness(live);
            self.all_liveness.extend_from_slice(&encoded);
        }
        self.all_liveness_length = self.all_liveness.len();
        self.all_liveness_positions.insert(key, pos);
        pos
    }

    /// RPython: opcode key → opcode number.
    /// RPython `assembler.py:220-222`:
    /// ```text
    /// key = opname + '/' + ''.join(argcodes)
    /// num = self.insns.setdefault(key, len(self.insns))
    /// ```
    ///
    /// RPython parity: `assembler.py:220
    /// self.insns.setdefault(key, len(self.insns))`.  Each opname/
    /// argcodes key gets a stable opcode byte recorded into
    /// `self.insns`; subsequent emissions of the same key reuse the
    /// recorded byte.
    ///
    /// Pyre serialises `insns.bin` at build time and the runtime
    /// decoder reads those bytes verbatim, so canonical/extension keys
    /// pin a reserved `BC_*` (`crate::insns::wellknown_bh_insns` /
    /// `extension_insns`, merged through
    /// [`crate::insns::insn_byte_opt`]) — this preserves byte stability
    /// across builds for keys that the runtime walker dispatches.
    /// Translator-only keys (transient codewriter helpers, test
    /// fixtures) follow the upstream `setdefault` shape as closely as
    /// pyre's fixed-byte adaptation allows: scan upward from zero and
    /// allocate the lowest byte that is neither reserved by a
    /// canonical/extension key nor already used by another
    /// translator-only key.  Their byte landing in `self.insns` flows
    /// verbatim into the serialized pipeline.insns blob the runtime
    /// decoder reads.
    ///
    /// TODO: byte-stability vs. dynamic-range
    /// trade-off).  Upstream `assembler.py:221 setdefault(key,
    /// len(self.insns))` allocates densely from 0 — every emitted key
    /// consumes one of the full 256 byte slots, no reservation.  Pyre
    /// pins canonical/extension keys at fixed `BC_*` so build-time
    /// `pipeline.insns` and runtime `wellknown_bh_insns()` can decode
    /// the same byte to the same opname; the cost is that
    /// translator-only keys must avoid reserved bytes.  Earlier pyre
    /// builds allocated only above `canonical_byte_high_water()`, which
    /// made every gap below the high-water unusable.  The scanner below
    /// preserves fixed canonical bytes while recovering those gaps,
    /// leaving only actually reserved bytes unavailable.  The panic
    /// surfaces exhaustion at the offending registration site instead
    /// of silently wrapping.
    pub fn get_opnum(&mut self, key: &str) -> u8 {
        if let Some(&existing) = self.insns.get(&key.to_string()) {
            return existing;
        }
        if let Some(num) = crate::insns::insn_byte_opt(key) {
            debug_assert!(
                crate::insns::is_reserved_opcode_byte(num),
                "insn_byte_opt({key:?}) returned {num} which is not reserved — \
                 wellknown/extension tables out of sync with is_reserved_opcode_byte",
            );
            self.insns.insert(key.to_string(), num);
            return num;
        }
        let num = self.next_dynamic_opnum(key);
        self.insns.insert(key.to_string(), num);
        num
    }

    pub fn next_dynamic_opnum(&mut self, key: &str) -> u8 {
        let mut candidate = self.dynamic_byte_cursor;
        while candidate <= u8::MAX as u16 {
            let byte = candidate as u8;
            let is_available = !crate::insns::is_reserved_opcode_byte(byte)
                && !self.insns.values().any(|&used| used == byte);
            if is_available {
                self.dynamic_byte_cursor = candidate + 1;
                return byte;
            }
            candidate += 1;
        }
        panic!(
            "Assembler::get_opnum: opcode byte exhausted while assigning \
             translator-only key {key:?}; all non-reserved u8 opcode bytes \
             are already assigned"
        );
    }
}

impl Assembler {
    /// RPython: `Assembler.see_raw_object(value)` (assembler.py).
    ///
    /// Registers a function/vtable name for debugging.
    /// RPython stores `(addr, name)` pairs; majit stores `(path, name)`.
    pub fn see_raw_object(&mut self, path: &str, name: &str) {
        if self.seen_raw_objects.insert(path.to_string()) {
            self.list_of_addr2name
                .push((path.to_string(), name.to_string()));
        }
    }

    /// RPython: `Assembler.finished(callinfocollection)` (assembler.py).
    ///
    /// ```python
    /// def finished(self, callinfocollection):
    ///     for func in callinfocollection.all_function_addresses_as_int():
    ///         func = int2adr(func)
    ///         self.see_raw_object(func.ptr)
    /// ```
    ///
    /// RPython's `see_raw_object` extracts `func.ptr._obj._name` to build
    /// `list_of_addr2name`. In majit, names are registered at `add()` time
    /// via `register_func_name()`.
    /// RPython: Assembler.insns — the opcode table. Needed by
    /// BlackholeInterpBuilder::setup_insns() to build the dispatch table.
    pub fn insns(&self) -> &indexmap::IndexMap<String, u8> {
        &self.insns
    }

    /// Register an `(opname/argcodes, opnum)` pair into `self.insns`.
    ///
    /// RPython `assembler.py:222 self.insns[key] = opnum` records every
    /// opcode the assembler emits during `assemble()`.  Pyre's
    /// state-field-JIT macro path skips `assemble()` entirely (the
    /// `JitCodeBuilder` emits BC_* directly), so the canonical entries
    /// — `live/`, `catch_exception/L`, `*_return/*` — are populated
    /// here at install time so `MetaInterpStaticData::setup_insns`
    /// (`pyjitpl.py`) can do the dynamic
    /// `insns.get(name)` lookup instead of a parallel hardcoded
    /// `BC_*` seeding block.
    pub fn register_insn(&mut self, name: &str, opnum: u8) {
        self.insns.insert(name.to_string(), opnum);
    }

    /// RPython `assembler.py Assembler.__init__ self.all_liveness = []` — the shared
    /// liveness byte stream populated by `_encode_liveness`.  Returned
    /// as a contiguous `&[u8]` view so consumers (notably
    /// `MetaInterpStaticData::finish_setup` per `pyjitpl.py`) can
    /// take a snapshot without depending on the dedup cache or
    /// position table.
    pub fn all_liveness(&self) -> &[u8] {
        &self.all_liveness
    }

    /// Place an embedded codewriter's liveness table before entries already
    /// registered by the runtime JitCode builder.
    ///
    /// The embedded JitCodes already contain offsets relative to byte zero of
    /// `prefix`, so that table cannot be appended. Entries registered in this
    /// Assembler have not yet been patched into runtime-built JitCodes; moving
    /// their cached offsets by the prefix length keeps both producers in one
    /// `metainterp_sd.liveness_info` stream. Repeated installation of the same
    /// prefix is idempotent.
    pub fn prepend_embedded_liveness(&mut self, prefix: &[u8]) {
        if prefix.is_empty() {
            return;
        }
        if self.embedded_liveness_prefix_len != 0 {
            assert_eq!(
                self.embedded_liveness_prefix_len,
                prefix.len(),
                "a different embedded liveness prefix was already installed"
            );
            assert_eq!(
                &self.all_liveness[..prefix.len()],
                prefix,
                "the installed embedded liveness prefix changed"
            );
            return;
        }

        let shift = prefix.len();
        let mut combined = Vec::with_capacity(shift + self.all_liveness.len());
        combined.extend_from_slice(prefix);
        combined.extend_from_slice(&self.all_liveness);
        self.all_liveness = combined;
        self.embedded_liveness_prefix_len = shift;
        self.all_liveness_length = self.all_liveness.len();
        for offset in self.all_liveness_positions.values_mut() {
            *offset += shift;
        }
        if let Some(offset) = self.canonical_liveness_offset.as_mut() {
            *offset += shift;
        }
    }

    /// Snapshot the descriptor table after all jitcodes have been fully
    /// assembled. Pending inline-call descriptors are lowered here to the
    /// final `(jitcode_index, fnaddr, calldescr)` form that runtime
    /// consumers expect.
    pub fn snapshot_descrs(&self) -> Vec<crate::jitcode::BhDescr> {
        self.descrs
            .iter()
            .map(|descr| match descr {
                AssemblerDescr::Ready(descr) => descr.as_ref().clone(),
                AssemblerDescr::PendingJitCode { jitcode } => crate::jitcode::BhDescr::JitCode {
                    jitcode_index: jitcode.index(),
                    fnaddr: jitcode.fnaddr,
                    calldescr: jitcode.calldescr().clone(),
                },
                AssemblerDescr::PendingSwitch { .. } => {
                    panic!("snapshot_descrs called before switch descriptors were resolved")
                }
            })
            .collect()
    }

    /// descriptor census: how much of the descr pool is content-duplicate.
    ///
    /// `emit_descr` dedups on [`AssemblerDescrKey`], whose `Call` arms carry an
    /// [`EffectInfoKey`] keyed on `Arc::as_ptr` ptr-ids. Upstream can key on
    /// `id(descr)` (`effectinfo.py:152-164`) because one gccache per process
    /// canonicalises every descr, so `id()` *is* content identity. Pyre can
    /// mint the same logical descr more than once, so two EffectInfos that
    /// agree on content can disagree on ptr-id and take separate pool slots.
    ///
    /// **That over-split is a real unsoundness but it is NOT the cause of
    /// descriptor census's byte non-determinism, and this census is what refuted it.**
    /// Measured across two generations: `structurally_distinct` equals
    /// `comparable` (229 == 229), so every effect-keyed entry is already
    /// pairwise distinct under the structural key — re-keying on
    /// `descr_set_keys` would yield the same entries. And `total` was
    /// identical (4537) in both generations while `descrs.bin` still moved
    /// −2,226 bytes. The pool population does not move; entry *lengths* do.
    /// See [`crate::codewriter::jitcode::descr_pool_content`], which measures
    /// the channel that does.
    ///
    /// Retained because a *stable* over-split is still worth knowing about,
    /// and because this function is the control proving the population is not
    /// the mover. Measurable in ONE run.
    pub fn descr_pool_duplication(&self) -> DescrPoolDuplication {
        let mut out = DescrPoolDuplication {
            total: self.descrs.len(),
            ..Default::default()
        };
        let mut seen: std::collections::HashSet<(String, EffectInfoStructuralKey)> =
            std::collections::HashSet::new();
        for descr in &self.descrs {
            let (shape, effect) = match descr {
                AssemblerDescr::Ready(descr) => match descr.as_ref() {
                    crate::jitcode::BhDescr::Call { calldescr } => (
                        format!(
                            "call|{}|{}|{}|{}|{:?}",
                            calldescr.arg_classes,
                            calldescr.result_type,
                            calldescr.result_signed,
                            calldescr.result_size,
                            calldescr.result_erased
                        ),
                        &calldescr.extra_info,
                    ),
                    crate::jitcode::BhDescr::JitCode {
                        jitcode_index,
                        fnaddr,
                        calldescr,
                    } => (
                        format!(
                            "jitcode|{jitcode_index}|{fnaddr}|{}|{}|{}|{}|{:?}",
                            calldescr.arg_classes,
                            calldescr.result_type,
                            calldescr.result_signed,
                            calldescr.result_size,
                            calldescr.result_erased
                        ),
                        &calldescr.extra_info,
                    ),
                    _ => continue,
                },
                _ => continue,
            };
            out.effect_keyed += 1;
            match DescrSetShape::of(effect) {
                DescrSetShape::InvariantViolation => {
                    out.invariant_violations += 1;
                    continue;
                }
                DescrSetShape::Concrete => out.concrete += 1,
                DescrSetShape::Wildcard => out.wildcard += 1,
            }
            out.comparable += 1;
            seen.insert((shape, EffectInfoStructuralKey::from_effect_info(effect)));
        }
        out.structurally_distinct = seen.len();
        out
    }

    pub fn finished(&mut self, callinfocollection: &CallInfoCollection) {
        for func_addr in callinfocollection.all_function_addresses_as_int() {
            // RPython: see_raw_object(func.ptr)
            // → name = value._obj._name (for FuncType)
            // → self.list_of_addr2name.append((addr, name))
            let name = callinfocollection.func_name(func_addr).unwrap_or("?");
            let addr_key = format!("{func_addr:#x}");
            self.see_raw_object(&addr_key, name);
        }
    }

    /// Number of JitCodes assembled so far.
    pub fn count_jitcodes(&self) -> usize {
        self.count_jitcodes
    }
}

/// `effectinfo.py` `EffectInfo._cache` cache key parity.
///
/// PyPy keys the EI factory cache on the raw `frozenset[Descr]`
/// readonly/write sets, NOT on the `bitstring_*` fields.  The
/// bitstrings are setup-time derived state (`compute_bitstrings`
/// at `effectinfo.py`), so the same logical EI must hit the
/// same cache slot before AND after compaction.  Pyre's lift
/// projects the `Vec<DescrRef>` raw sets to `Arc::as_ptr` ptr-id
/// `Vec<usize>` for `Hash`/`Eq` — direct lift of PyPy's
/// `frozenset[id(descr)]` cache key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EffectInfoKey {
    extraeffect: majit_ir::descr::ExtraEffect,
    oopspecindex: majit_ir::descr::OopSpecIndex,
    readonly_descrs_fields: Option<Vec<usize>>,
    write_descrs_fields: Option<Vec<usize>>,
    readonly_descrs_arrays: Option<Vec<usize>>,
    write_descrs_arrays: Option<Vec<usize>>,
    readonly_descrs_interiorfields: Option<Vec<usize>>,
    write_descrs_interiorfields: Option<Vec<usize>>,
    can_invalidate: bool,
    can_collect: bool,
    call_release_gil_target: (u64, i32),
}

/// [`EffectInfoKey`] re-spelled structurally, for the descriptor census census only.
///
/// Identical except that the six raw descr sets are read from
/// `EffectInfo::descr_set_keys` — the `DescrSetMember` projection that already
/// crosses the build/runtime boundary — instead of `Arc::as_ptr` ptr-ids.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EffectInfoStructuralKey {
    extraeffect: majit_ir::descr::ExtraEffect,
    oopspecindex: majit_ir::descr::OopSpecIndex,
    descr_set_keys: Option<majit_ir::effectinfo::DescrSetKeys>,
    can_invalidate: bool,
    can_collect: bool,
    call_release_gil_target: (u64, i32),
}

impl EffectInfoStructuralKey {
    fn from_effect_info(effect: &majit_ir::descr::EffectInfo) -> Self {
        Self {
            extraeffect: effect.extraeffect,
            oopspecindex: effect.oopspecindex,
            descr_set_keys: effect.descr_set_keys.as_deref().cloned(),
            can_invalidate: effect.can_invalidate,
            can_collect: effect.can_collect,
            call_release_gil_target: effect.call_release_gil_target,
        }
    }
}

/// Which shape an `EffectInfo`'s `descr_set_keys` takes (descriptor census).
///
/// `effectinfo.rs`'s `analyze_external_call` states the rule and names its
/// upstream source:
/// `effectinfo.py` makes the six raw sets `None` **iff** the EI is
/// `EF_RANDOM_EFFECTS`. So the population partitions in two, and the third
/// class below is unrepresentable rather than merely rare.
///
/// An earlier version of this file folded [`Self::Wildcard`] and
/// [`Self::Concrete`] into one `is_informative` predicate and excluded only
/// [`Self::InvariantViolation`]. Because those two disjuncts *partition* the
/// population, that predicate was identically `true` and its exclusion count
/// identically zero — a counter that cannot fire, reported as if it were the
/// census's control. Splitting the two shapes is what makes the control real:
/// the census is only trustworthy if it can be shown to have seen both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DescrSetShape {
    /// `descr_set_keys: Some(_)` — the structural projection is present. The
    /// only shape where two `Arc`s for one logical descr can split a key.
    Concrete,
    /// `descr_set_keys: None` with all six raw sets `None` — the
    /// `EF_RANDOM_EFFECTS` wildcard. Its ptr-id projection is `None` too, so
    /// it is already canonical under the ptr-keyed dict and cannot move.
    Wildcard,
    /// Keys absent while a raw set is present. **The invariant above forbids
    /// this**; counted so the census *asserts* the invariant rather than
    /// assuming it. Expected zero — and a zero here means only that, not that
    /// the census works.
    InvariantViolation,
}

impl DescrSetShape {
    fn of(effect: &majit_ir::descr::EffectInfo) -> Self {
        if effect.descr_set_keys.is_some() {
            return Self::Concrete;
        }
        let raw_sets_absent = effect._readonly_descrs_fields.is_none()
            && effect._write_descrs_fields.is_none()
            && effect._readonly_descrs_arrays.is_none()
            && effect._write_descrs_arrays.is_none()
            && effect._readonly_descrs_interiorfields.is_none()
            && effect._write_descrs_interiorfields.is_none();
        if raw_sets_absent {
            Self::Wildcard
        } else {
            Self::InvariantViolation
        }
    }
}

/// Output of [`Assembler::descr_pool_duplication`] (descriptor census).
#[derive(Debug, Default)]
pub struct DescrPoolDuplication {
    /// `descrs.len()` — the bincode `Vec` length that opens `descrs.bin`.
    pub total: usize,
    /// Entries whose `_descr_dict` key carries an `EffectInfoKey`, i.e. the
    /// ones keyed partly on `Arc::as_ptr`.
    pub effect_keyed: usize,
    /// Of those, [`DescrSetShape::Concrete`] — the only sub-population the
    /// ptr-keyed dict can over-split, and the only one the fix can move.
    pub concrete: usize,
    /// Of those, [`DescrSetShape::Wildcard`] — already canonical under both
    /// keys, so inert for the defect signal and load-bearing as the control.
    pub wildcard: usize,
    /// `concrete + wildcard`. Entries carrying a well-defined structural key.
    pub comparable: usize,
    /// Distinct structural keys among `comparable`. **Lower than `comparable`
    /// means the ptr-keyed dict over-split: the pool holds entries that agree
    /// on content and disagree only on which `Arc` instance they reached.**
    pub structurally_distinct: usize,
    /// [`DescrSetShape::InvariantViolation`] — expected zero *by construction*.
    /// It asserts the `effectinfo.py:149-162` invariant; it does **not**
    /// certify the census.
    pub invariant_violations: usize,
}

impl DescrPoolDuplication {
    /// Whether the census observed both shapes, so `structurally_distinct` can
    /// be read at all.
    ///
    /// This is the control the earlier `ambiguous == 0` was mistaken for. A
    /// census that saw no `Concrete` entry cannot have seen an over-split
    /// whatever it reports, and one that saw no `Wildcard` entry is walking a
    /// population that does not match the enumerated construction sites.
    pub fn saw_both_shapes(&self) -> bool {
        self.concrete > 0 && self.wildcard > 0
    }

    /// The partition identity. False means the classification lost entries and
    /// every other field is suspect.
    pub fn counts_reconcile(&self) -> bool {
        self.concrete + self.wildcard + self.invariant_violations == self.effect_keyed
            && self.comparable == self.concrete + self.wildcard
    }
}

impl EffectInfoKey {
    fn from_effect_info(effect: &majit_ir::descr::EffectInfo) -> Self {
        Self {
            extraeffect: effect.extraeffect,
            oopspecindex: effect.oopspecindex,
            // `effectinfo.py:152-164` cache key: raw `_*_descrs_*`
            // sets (frozenset[Descr] lift), projected to
            // `Arc::as_ptr` ptr-ids.  NOT the lazily-published
            // `bitstring_*` fields.
            readonly_descrs_fields: majit_ir::effectinfo::descr_set_to_ptr_set_pub(
                &effect._readonly_descrs_fields,
            ),
            write_descrs_fields: majit_ir::effectinfo::descr_set_to_ptr_set_pub(
                &effect._write_descrs_fields,
            ),
            readonly_descrs_arrays: majit_ir::effectinfo::descr_set_to_ptr_set_pub(
                &effect._readonly_descrs_arrays,
            ),
            write_descrs_arrays: majit_ir::effectinfo::descr_set_to_ptr_set_pub(
                &effect._write_descrs_arrays,
            ),
            readonly_descrs_interiorfields: majit_ir::effectinfo::descr_set_to_ptr_set_pub(
                &effect._readonly_descrs_interiorfields,
            ),
            write_descrs_interiorfields: majit_ir::effectinfo::descr_set_to_ptr_set_pub(
                &effect._write_descrs_interiorfields,
            ),
            can_invalidate: effect.can_invalidate,
            can_collect: effect.can_collect,
            call_release_gil_target: effect.call_release_gil_target,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AssemblerDescrKey {
    Field {
        offset: usize,
        field_size: usize,
        field_type: majit_ir::value::Type,
        field_flag: majit_ir::descr::ArrayFlag,
        is_field_signed: bool,
        is_immutable: bool,
        is_quasi_immutable: bool,
        /// `Option`, matching `BhDescr::Field`, so `None` and `Some(0)` keep
        /// separate pool slots. Two mints that agree on every other component
        /// while one resolved a slot and the other did not are not the same
        /// descr, and collapsing them here would hand the runtime whichever
        /// provenance happened to be minted first.
        index_in_parent: Option<usize>,
        parent: Option<std::sync::Arc<crate::jitcode::BhSizeSpec>>,
        name: String,
        owner: String,
    },
    Array {
        base_size: usize,
        itemsize: usize,
        len_offset: Option<usize>,
        /// u64 cache-key surrogate matching `BhDescr::Array.type_id`.
        type_id: u64,
        item_type: majit_ir::value::Type,
        is_array_of_pointers: bool,
        is_array_of_structs: bool,
        is_item_signed: bool,
        // `ei_index` deliberately omitted from the identity tuple —
        // upstream `gccache._cache_array[ARRAY_OR_STRUCT]`
        // (`descr.py get_array_descr`) keys on the lltype itself, and
        // `compute_bitstrings` (`effectinfo.py`) later assigns the
        // index slot as a derived attribute that multiple descrs are
        // free to share.
        //
        // `array_type_id` joins the identity tuple as the codewriter
        // lltype-identity proxy so two ARRAYs that disagree only on
        // the Rust type string (e.g. `Vec<Foo>` vs `Vec<Bar>` with
        // both at `type_id == 0`) keep distinct slots in the
        // assembler's `_descr_dict`, mirroring upstream's per-lltype
        // cache identity.
        array_type_id: Option<String>,
        interior_fields: Vec<crate::jitcode::BhInteriorFieldSpec>,
    },
    Size {
        size: usize,
        /// u64 cache-key surrogate matching `BhDescr::Size.type_id`.
        type_id: u64,
        vtable: u64,
        owner: String,
        all_fielddescrs: Vec<crate::jitcode::BhFieldSpec>,
    },
    Call {
        arg_classes: String,
        result_type: char,
        result_signed: bool,
        result_size: usize,
        result_erased: crate::jitcode::CallResultErasedKey,
        void_word_abi: bool,
        effect: EffectInfoKey,
    },
    /// RPython uses the JitCode object itself as an AbstractDescr for
    /// inline_call. The Rust key is therefore the identity-keyed handle, not
    /// the callsite-local `BhCallDescr`.
    JitCode(crate::jitcode::JitCodeHandle),
    SnapshotJitCode {
        jitcode_index: usize,
        fnaddr: i64,
        arg_classes: String,
        result_type: char,
        result_signed: bool,
        result_size: usize,
        result_erased: crate::jitcode::CallResultErasedKey,
        void_word_abi: bool,
        effect: EffectInfoKey,
    },
    Switch(Vec<(i64, usize)>),
    VableField {
        index: usize,
    },
    VableArray {
        index: usize,
    },
    VtableMethod {
        trait_root: String,
        method_name: String,
    },
    /// `descr.py InteriorFieldDescr(arraydescr, fielddescr)` identity
    /// is the composition of the array and field descriptor keys.
    InteriorField {
        array: Box<AssemblerDescrKey>,
        field: Box<AssemblerDescrKey>,
    },
}

impl AssemblerDescrKey {
    fn from_descr(descr: &AssemblerDescr) -> Self {
        match descr {
            AssemblerDescr::Ready(descr) => Self::from_ready(descr),
            AssemblerDescr::PendingJitCode { jitcode } => Self::JitCode(jitcode.clone()),
            AssemblerDescr::PendingSwitch { .. } => {
                unreachable!("switch descriptors bypass `_descr_dict`")
            }
        }
    }

    fn from_ready(descr: &crate::jitcode::BhDescr) -> Self {
        match descr {
            crate::jitcode::BhDescr::Field {
                offset,
                field_size,
                field_type,
                field_flag,
                is_field_signed,
                is_immutable,
                is_quasi_immutable,
                index_in_parent,
                parent,
                name,
                owner,
            } => Self::Field {
                offset: *offset,
                field_size: *field_size,
                field_type: *field_type,
                field_flag: *field_flag,
                is_field_signed: *is_field_signed,
                is_immutable: *is_immutable,
                is_quasi_immutable: *is_quasi_immutable,
                index_in_parent: *index_in_parent,
                parent: parent.clone(),
                name: name.clone(),
                owner: owner.clone(),
            },
            crate::jitcode::BhDescr::Array {
                base_size,
                itemsize,
                len_offset,
                type_id,
                gc_type_id: _,
                item_type,
                is_array_of_pointers,
                is_array_of_structs,
                is_item_signed,
                // `ei_index` intentionally not part of the identity
                // tuple — see `AssemblerDescrKey::Array` comment.
                ei_index: _,
                // Functionally determined by the array shape (`type_id` /
                // raw-vs-GC), so not part of the dedup-key identity —
                // same treatment as the `Size` arm.
                is_gc_managed: _,
                array_type_id,
                interior_fields,
            } => Self::Array {
                base_size: *base_size,
                itemsize: *itemsize,
                len_offset: *len_offset,
                type_id: *type_id,
                item_type: *item_type,
                is_array_of_pointers: *is_array_of_pointers,
                is_array_of_structs: *is_array_of_structs,
                is_item_signed: *is_item_signed,
                array_type_id: array_type_id.clone(),
                interior_fields: interior_fields.clone(),
            },
            crate::jitcode::BhDescr::Size {
                size,
                type_id,
                vtable,
                owner,
                all_fielddescrs,
                // Functionally determined by `type_id` (a struct is GC or
                // raw, not both), so not part of the dedup-key identity.
                is_gc_managed: _,
            } => Self::Size {
                size: *size,
                type_id: *type_id,
                vtable: *vtable,
                owner: owner.clone(),
                all_fielddescrs: all_fielddescrs.clone(),
            },
            crate::jitcode::BhDescr::Call { calldescr } => Self::Call {
                arg_classes: calldescr.arg_classes.clone(),
                result_type: calldescr.result_type,
                result_signed: calldescr.result_signed,
                result_size: calldescr.result_size,
                result_erased: calldescr.result_erased,
                void_word_abi: calldescr.void_word_abi,
                effect: EffectInfoKey::from_effect_info(&calldescr.extra_info),
            },
            crate::jitcode::BhDescr::JitCode {
                jitcode_index,
                fnaddr,
                calldescr,
            } => Self::SnapshotJitCode {
                jitcode_index: *jitcode_index,
                fnaddr: *fnaddr,
                arg_classes: calldescr.arg_classes.clone(),
                result_type: calldescr.result_type,
                result_signed: calldescr.result_signed,
                result_size: calldescr.result_size,
                result_erased: calldescr.result_erased,
                void_word_abi: calldescr.void_word_abi,
                effect: EffectInfoKey::from_effect_info(&calldescr.extra_info),
            },
            crate::jitcode::BhDescr::Switch { dict, .. } => {
                let mut items: Vec<_> = dict.iter().map(|(key, value)| (*key, *value)).collect();
                items.sort_unstable_by_key(|(key, _)| *key);
                Self::Switch(items)
            }
            crate::jitcode::BhDescr::VableField { index } => Self::VableField { index: *index },
            crate::jitcode::BhDescr::VableArray { index } => Self::VableArray { index: *index },
            crate::jitcode::BhDescr::VtableMethod {
                trait_root,
                method_name,
            } => Self::VtableMethod {
                trait_root: trait_root.clone(),
                method_name: method_name.clone(),
            },
            crate::jitcode::BhDescr::InteriorField { array, field } => Self::InteriorField {
                array: Box::new(Self::from_ready(array)),
                field: Box::new(Self::from_ready(field)),
            },
        }
    }
}

#[derive(Debug, Clone)]
pub enum AssemblerDescr {
    Ready(Box<crate::jitcode::BhDescr>),
    PendingJitCode {
        jitcode: crate::jitcode::JitCodeHandle,
    },
    PendingSwitch {
        cases: Vec<(i64, Label)>,
    },
}

impl Default for Assembler {
    fn default() -> Self {
        Self::new()
    }
}
