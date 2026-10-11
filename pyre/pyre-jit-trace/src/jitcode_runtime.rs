//! Runtime access to the build-time `pipeline.jitcodes` table.
//!
//! RPython: `MetaInterpStaticData.jitcodes` in `warmspot.py` — the list
//! of `JitCode` objects produced by `CodeWriter.make_jitcodes()`
//! (codewriter.py). In RPython this list is passed by reference from
//! `CallControl.jitcodes` directly into `MetaInterpStaticData`; the two
//! stores reference the same Python objects.
//!
//! majit's build-time side lives in `majit_jitcode::jitcode::JitCode`
//! (serde-serializable, emitted by `build.rs` into `$OUT_DIR/jitcodes.bin`).
//! This module uses the separately encoded name/offset index to deserialize
//! each `Arc<JitCode>` on first access. The configured portal index is read
//! from the separately serialized generic `CompiledJitDriver` metadata.
//!
//! No opcode-body side table is serialized: `pipeline.jitcodes`, in allocation
//! order, remains the single executable store matching RPython's model
//! (`feedback_single_jitcodes_store`). Driver metadata only identifies indices
//! in that store; it does not duplicate JitCode bodies.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::{Arc, LazyLock, Once, OnceLock};

use majit_ir::DescrRef;
use majit_jitcode::artifacts::CompiledJitDriver;
use majit_jitcode::jitcode::{BhDescr, DescrTable, JitCode};

use majit_jitcode::artifacts::JitCodeIndex;

/// Runtime-frozen canonical JitCode.
///
/// The serialized body skips `_ssarepr`, the sole path from this type into
/// translation-time `Rc<RefCell<Variable>>` graphs. `load_jitcode` performs
/// every address patch while its Arc is still unique, then this wrapper
/// publishes the immutable result. Its remaining interior cells are
/// `std::sync::OnceLock` caches. Keeping this proof on the runtime wrapper
/// avoids falsely declaring the translation-time `JitCode` Sync.
struct FrozenJitCode(Arc<JitCode>);

// SAFETY: see the publication/frozen-payload invariant above.
unsafe impl Send for FrozenJitCode {}
unsafe impl Sync for FrozenJitCode {}

struct FrozenJitCodeSlice(Box<[Arc<JitCode>]>);

// SAFETY: every Arc in this slice comes from a published `FrozenJitCode`.
unsafe impl Send for FrozenJitCodeSlice {}
unsafe impl Sync for FrozenJitCodeSlice {}

/// RPython `MetaInterpStaticData.jitcodes`: one process-global table shared by
/// every execution context. Runtime patching finishes before each cell is
/// published.
static JITCODE_CELLS: LazyLock<Vec<OnceLock<FrozenJitCode>>> =
    LazyLock::new(|| (0..jitcode_count()).map(|_| OnceLock::new()).collect());

/// Decoded names and byte offsets; loading this never decodes a body.
static JITCODE_INDEX: LazyLock<JitCodeIndex> = LazyLock::new(load_jitcode_index);

/// Compatibility accessor storage for callers requesting the complete table.
static FORCED_ALL_JITCODES: LazyLock<FrozenJitCodeSlice> = LazyLock::new(|| {
    FrozenJitCodeSlice(
        (0..jitcode_count())
            .map(|index| get_jitcode_by_index(index).unwrap())
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    )
});

/// Explicit build-time JIT-driver metadata for every configured driver.
static COMPILED_JIT_DRIVERS: LazyLock<Vec<CompiledJitDriver>> =
    LazyLock::new(load_compiled_jit_drivers);

/// Frozen source-translation indirect-call-target family.
static INDIRECTCALLTARGETS: LazyLock<Box<[Arc<majit_metainterp::jitcode::JitCode>]>> =
    LazyLock::new(build_indirectcalltargets);

fn load_jitcode_index() -> JitCodeIndex {
    const INDEX_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/jitcodes_index.bin"));
    const BODY_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/jitcodes.bin"));
    JitCodeIndex::decode(INDEX_BYTES, BODY_BYTES).expect("pyre JitCode index")
}

fn jitcode_index() -> &'static JitCodeIndex {
    &JITCODE_INDEX
}

pub fn jitcode_count() -> usize {
    jitcode_index().offsets.len() - 1
}

fn load_jitcode(index: usize) -> Arc<JitCode> {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/jitcodes.bin"));
    let mut jitcode = jitcode_index().load(BYTES, index).unwrap_or_else(|e| {
        panic!("pyre-jit-trace: failed to deserialize jitcodes.bin entry {index}: {e}")
    });
    // RPython's translator AOT-compiles every helper into the same binary so
    // `JitCode.fnaddr` / `constants_i` funcptrs are linker-resolved.  Pyre's
    // codewriter ran in `build.rs` (a separate process) and captured stale
    // build-time addresses; patch each Arc<JitCode> in place — refcount is
    // still 1 here, no consumer has cloned yet — using
    // `pyre_interpreter::jit_trace_fnaddrs()`'s runtime values.
    crate::runtime_fnaddr_patch::patch_constants_i_fnaddrs(std::slice::from_mut(&mut jitcode));
    // Host-static *data* addresses the assembler tagged on `reloc_consts_i`
    // (`ConstIRelocKind::StaticAddr`; exception-class llexitcase) and
    // `reloc_consts_r` (prebuilt `HostStaticAddrs.refs`). Re-pair those
    // with the runtime addresses while refcount is still 1.
    crate::runtime_fnaddr_patch::patch_static_addr_constants(std::slice::from_mut(&mut jitcode));
    // Deferred prebuilt-string constants the codewriter could not allocate
    // at build time (separate process) carry a non-canonical sentinel in
    // `constants_r`; materialize their immortal STR blocks and overwrite the
    // sentinels here, while refcount is still 1 — before any consumer can
    // observe a sentinel as a forged GCREF.
    crate::runtime_fnaddr_patch::materialize_str_consts(std::slice::from_mut(&mut jitcode));
    // Deferred unit-variant singleton constants follow the same contract:
    // materialize each variant's immortal discriminant cell and overwrite
    // the sentinel while refcount is still 1.
    crate::runtime_fnaddr_patch::materialize_unit_variant_consts(std::slice::from_mut(
        &mut jitcode,
    ));
    crate::runtime_fnaddr_patch::materialize_exc_instance_consts(std::slice::from_mut(
        &mut jitcode,
    ));
    crate::runtime_fnaddr_patch::materialize_type_static_consts(std::slice::from_mut(&mut jitcode));
    // RPython codewriter.py:80: `all_jitcodes[jitcode.index] is jitcode`.
    // Check per entry so any regression in
    // `collect_jitcodes_in_alloc_order` is caught immediately.
    assert_eq!(
        jitcode.index(),
        index,
        "pyre-jit-trace: jitcode[{index}].index = {} (expected {index}); \
         RPython invariant `all_jitcodes[i].index == i` broken",
        jitcode.index(),
    );
    jitcode
}

pub(crate) fn get_jitcode_ref_by_index(index: usize) -> Option<&'static Arc<JitCode>> {
    let cell = JITCODE_CELLS.get(index)?;
    Some(&cell.get_or_init(|| FrozenJitCode(load_jitcode(index))).0)
}

fn load_compiled_jit_drivers() -> Vec<CompiledJitDriver> {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/jit_drivers.bin"));
    let vec: Vec<CompiledJitDriver> = bincode::deserialize(BYTES).unwrap_or_else(|e| {
        panic!(
            "pyre-jit-trace: failed to deserialize jit_drivers.bin \
             ({} bytes): {e}",
            BYTES.len(),
        )
    });
    vec
}

/// RPython: `metainterp_sd.jitcodes` — explicitly force the full table.
pub fn all_jitcodes() -> &'static [Arc<JitCode>] {
    &FORCED_ALL_JITCODES.0
}

/// RPython: `metainterp_sd.jitcodes[index]` where `index == jitcode.index`.
///
/// The dense invariant (`all_jitcodes[i].index == i`) is asserted when each
/// entry is decoded, so direct indexed lookup is correct.
pub fn get_jitcode_by_index(index: usize) -> Option<Arc<JitCode>> {
    get_jitcode_ref_by_index(index).cloned()
}

/// Runtime wrappers for canonical source-translator JitCodes embedded as
/// `AbstractDescr` operands in per-Python-function bytecode.  RPython owns
/// these in the process-wide `CallControl.jitcodes[graph]` map, so this cache
/// is process-wide too rather than another copy in each thread.  The slots are
/// index-aligned with the canonical JitCode table.
static RUNTIME_JITCODE_CELLS: std::sync::OnceLock<
    Box<[std::sync::OnceLock<Arc<majit_metainterp::jitcode::JitCode>>]>,
> = std::sync::OnceLock::new();

fn runtime_jitcode_cells() -> &'static [std::sync::OnceLock<Arc<majit_metainterp::jitcode::JitCode>>]
{
    RUNTIME_JITCODE_CELLS.get_or_init(|| {
        (0..jitcode_count())
            .map(|_| std::sync::OnceLock::new())
            .collect::<Vec<_>>()
            .into_boxed_slice()
    })
}

fn seeded_reachable_symbolic_residuals(
    index: usize,
) -> majit_metainterp::jitcode::ReachableSymbolicResiduals {
    let (values, visited) = jitcode_index()
        .reachable_symbolic_residuals_at(index)
        .unwrap_or_else(|| panic!("jitcode {index} has no reachable symbolic residual row"));
    // Constants this process rewrote are no longer symbolic in the patched
    // body. The table still names the build-time hashes.
    let targets = values
        .iter()
        .copied()
        .filter(|value| !crate::runtime_fnaddr_patch::is_bound_symbolic_fnaddr(*value))
        .collect();
    majit_metainterp::jitcode::ReachableSymbolicResiduals {
        targets,
        visited_jitcodes: visited as usize,
    }
}

pub fn get_runtime_jitcode_by_index(
    index: usize,
) -> Option<Arc<majit_metainterp::jitcode::JitCode>> {
    let cell = runtime_jitcode_cells().get(index)?;
    Some(
        cell.get_or_init(|| {
            let canonical = get_jitcode_by_index(index)
                .expect("runtime JitCode cell index must resolve canonically");
            Arc::new(
                majit_metainterp::jitcode::JitCode::from_canonical((*canonical).clone())
                    .with_reachable_symbolic_residuals(seeded_reachable_symbolic_residuals(index)),
            )
        })
        .clone(),
    )
}

/// The source translator's exact `Assembler.indirectcalltargets` set as
/// references into `all_jitcodes`.
///
/// `assembler.indirectcalltargets` is filled once while the graphs are being
/// assembled and holds the same `JitCode` objects the codewriter already
/// handed to `metainterp_sd.jitcodes` — one object per graph for the whole
/// process. The wrapper conversion below still has to own its core, so the
/// family is materialized once for the process and handed out by reference
/// afterwards; every publisher sees the same objects.
pub fn indirectcalltargets() -> &'static [Arc<majit_metainterp::jitcode::JitCode>] {
    &INDIRECTCALLTARGETS
}

fn build_indirectcalltargets() -> Box<[Arc<majit_metainterp::jitcode::JitCode>]> {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/indirectcalltargets.bin"));
    let entries: Vec<(usize, i64)> = bincode::deserialize(BYTES).unwrap_or_else(|e| {
        panic!(
            "pyre-jit-trace: failed to deserialize indirectcalltargets.bin \
             ({} bytes): {e}",
            BYTES.len(),
        )
    });
    let vec: Vec<Arc<majit_metainterp::jitcode::JitCode>> = entries
        .into_iter()
        .map(|(index, _)| {
            get_runtime_jitcode_by_index(index).unwrap_or_else(|| {
                panic!(
                    "pyre-jit-trace: indirect-call target index {index} is \
                     outside all_jitcodes (len={})",
                    jitcode_count()
                )
            })
        })
        .collect();
    vec.into_boxed_slice()
}

/// RPython `bytecode_for_address`'s translated-mode dictionary, represented
/// without eagerly materializing its AOT JitCode values.
///
/// In a translated PyPy binary the JitCode shells already exist as static
/// objects and the first lookup only builds `fnaddr -> object`.  Pyre's shells
/// are serialized build artifacts, so deserializing all ~1,500 bodies to make
/// that dictionary turns static binary data into a large first-loop RSS tax.
/// Preserve the same dict lookup while keeping `index` as the value; only the
/// matched JitCode body is decoded.
static INDIRECTCALLTARGET_BY_FNADDR: LazyLock<indexmap::IndexMap<usize, usize>> =
    LazyLock::new(|| {
        const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/indirectcalltargets.bin"));
        let entries: Vec<(usize, i64)> = bincode::deserialize(BYTES).unwrap_or_else(|e| {
            panic!(
                "pyre-jit-trace: failed to deserialize indirectcalltargets.bin \
             ({} bytes): {e}",
                BYTES.len(),
            )
        });
        let mut by_fnaddr = indexmap::IndexMap::with_capacity(entries.len());
        for (index, build_fnaddr) in entries {
            let fnaddr = crate::runtime_fnaddr_patch::runtime_fnaddr(build_fnaddr) as usize;
            // Identical-code folding can map several build-time addresses to
            // one runtime address; retain the first target for that address.
            //
            // `pyjitpl.py bytecode_for_address` asserts the address is
            // absent instead, because RPython's translation never folds two
            // graphs onto one entry point. Keeping the first target is the
            // only choice available here, but a wrong pick would otherwise be
            // invisible, so name every dropped target once at startup.
            match by_fnaddr.entry(fnaddr) {
                indexmap::map::Entry::Vacant(slot) => {
                    slot.insert(index);
                }
                indexmap::map::Entry::Occupied(kept) => majit_ir::debug::log_one(
                    "jit-indirectcalltargets",
                    &format!(
                        "runtime fnaddr {fnaddr:#x} already resolves to target {}; \
                         dropping target {index} (build fnaddr {build_fnaddr:#x})",
                        kept.get()
                    ),
                ),
            }
        }
        by_fnaddr
    });

pub(crate) fn indirectcalltarget_index_for_address(fnaddress: usize) -> Option<usize> {
    INDIRECTCALLTARGET_BY_FNADDR.get(&fnaddress).copied()
}

pub(crate) fn indirectcalltarget_by_index(
    index: usize,
) -> Option<Arc<majit_metainterp::jitcode::JitCode>> {
    get_runtime_jitcode_by_index(index)
}

// Cached index of the build-time portal jitcode within `ALL_JITCODES`.
//
// RPython `warmspot.py:281-282` + `call.py:147-148`:
// `jd.mainjitcode = self.get_jitcode(jd.portal_graph)` followed by
// `jd.mainjitcode.jitdriver_sd = jd`. The build artifact serializes that
// index directly in `CompiledJitDriver`; runtime validates the referenced
// JitCode still carries the driver marker instead of rediscovering portal
// identity from a name or flag scan.
/// Cached process-global portal JitCode index. The driver metadata and JitCode
/// table it resolves are process-global too.
static PORTAL_JITCODE_INDEX: OnceLock<Option<usize>> = OnceLock::new();

fn compute_portal_jitcode_index_for_key(key: &str) -> Option<usize> {
    let driver = COMPILED_JIT_DRIVERS
        .iter()
        .find(|driver| driver.portal.canonical_key() == key)?;
    let index = driver.main_jitcode_index;
    let jitcode = get_jitcode_ref_by_index(index).unwrap_or_else(|| {
        panic!("configured portal `{key}` refers to missing JitCode index {index}",)
    });
    assert!(
        jitcode.jitdriver_sd().is_some(),
        "configured portal `{}` does not carry its JitDriverStaticData marker",
        driver.portal.canonical_key(),
    );
    Some(index)
}

fn compute_portal_jitcode_index() -> Option<usize> {
    let mut matching = COMPILED_JIT_DRIVERS.iter().filter(|driver| {
        get_jitcode_ref_by_index(driver.main_jitcode_index)
            .is_some_and(|jitcode| jitcode.jitdriver_sd() == Some(0))
    });
    let driver = matching.next()?;
    assert!(
        matching.next().is_none(),
        "multiple compiled JIT drivers carry JitDriverStaticData index 0"
    );
    Some(driver.main_jitcode_index)
}

/// RPython: `metainterp_sd.jitcodes[jitdriver_sd.mainjitcode.index]`
/// (warmspot.py:281-282 + call.py:147-148) — the main
/// portal jitcode for JitDriverStaticData index 0, which
/// `find_all_graphs(portal, policy)` seeds the jitcode closure from. For a
/// per-driver path resolver, use [`portal_jitcode_for_key`]. Returns `None`
/// when driver index 0 is absent from the compiled metadata.
///
/// Trace-side user-function calls (`callee_frame_helper`,
/// `jit_create_callee_frame_*`, `jit_force_callee_frame`) route through
/// this accessor instead of emitting runtime per-CodeObject jitcodes.
/// The orthodox model treats every user CodeObject as the portal's
/// `pycode` input argument and reuses the single portal JitCode for
/// every call — see RPython `pypy/module/pypyjit/interp_jit.py
/// portal_runner` and `rpython/jit/codewriter/jtransform.py handle_regular_call`
/// `inline_call_*` emit.
pub fn portal_jitcode() -> Option<Arc<JitCode>> {
    let idx = (*PORTAL_JITCODE_INDEX.get_or_init(compute_portal_jitcode_index))?;
    get_jitcode_by_index(idx)
}

/// `newframe` takes the metainterp wrapper around the same portal body.
pub fn portal_metainterp_jitcode() -> Option<Arc<majit_metainterp::jitcode::JitCode>> {
    static PORTAL_META: OnceLock<Option<Arc<majit_metainterp::jitcode::JitCode>>> = OnceLock::new();
    PORTAL_META
        .get_or_init(|| {
            let canonical = portal_jitcode()?;
            let index = canonical.index();
            Some(Arc::new(
                majit_metainterp::jitcode::JitCode::from_canonical((*canonical).clone())
                    .with_reachable_symbolic_residuals(seeded_reachable_symbolic_residuals(index)),
            ))
        })
        .clone()
}

/// Resolve the portal `JitCode` for the configured driver whose portal
/// graph has canonical key `key` (e.g. a secondary driver's
/// `baseobjspace::unpackiterable_portal`). Per-driver analogue of
/// [`portal_jitcode`] — `warmspot.py:281-282`
/// `jd.mainjitcode = self.get_jitcode(jd.portal_graph)`.
pub fn portal_jitcode_for_key(key: &str) -> Option<Arc<JitCode>> {
    let idx = compute_portal_jitcode_index_for_key(key)?;
    get_jitcode_by_index(idx)
}

// Cached index of the charon-extracted `w_list_append` body within
// `ALL_JITCODES`.
//
// Unlike the portal (resolved by explicit `CompiledJitDriver` metadata),
// the list-op helper bodies carry no runtime marker — they are ordinary
// `make_jitcodes` function entries
// whose only stable handle is their `name`, which `get_jitcode` sets to
// the graph's last path segment (`CallControl::get_jitcode` in
// `codewriter/call.rs`). The positional index
// shifts whenever the jitcode set changes, so it must be resolved by
// name and never hardcoded.
//
// This by-name lookup is the foundation the list-append inline needs to descend
// the FBW walker into the orthodox charon list-append body (issue #62/#23): the
// dynamic `lst.append` recognition arm resolves the body here, then
// builds a by-index sub-walk from `get_jitcode_by_index`.
/// Cached `ALL_JITCODES` index of `w_list_append`, resolved by name because
/// jitcode indices are build-dependent.
static LIST_APPEND_JITCODE_INDEX: OnceLock<Option<usize>> = OnceLock::new();
/// Cached `ALL_JITCODES` index of `w_list_pop_end_inner`, resolved by name for
/// the same reason.
static LIST_POP_END_JITCODE_INDEX: OnceLock<Option<usize>> = OnceLock::new();
/// Cached `ALL_JITCODES` index of `write_cell` (`typeobject.py write_cell`).
static WRITE_CELL_JITCODE_INDEX: OnceLock<Option<usize>> = OnceLock::new();
/// Cached `ALL_JITCODES` index of `unwrap_cell` (`typeobject.py unwrap_cell`).
static UNWRAP_CELL_JITCODE_INDEX: OnceLock<Option<usize>> = OnceLock::new();
/// Cached `ALL_JITCODES` index of `w_list_setitem_inner`, resolved by name for
/// the same reason.
static LIST_SETITEM_JITCODE_INDEX: OnceLock<Option<usize>> = OnceLock::new();
/// Cached `ALL_JITCODES` index of `w_list_getitem_inner`, resolved by name for
/// the same reason.
static LIST_GETITEM_JITCODE_INDEX: OnceLock<Option<usize>> = OnceLock::new();
/// Cached `ALL_JITCODES` index of `w_tuple_getitem`, resolved by graph key
/// rather than by name -- see [`compute_pathed_jitcode_index`].
static TUPLE_GETITEM_JITCODE_INDEX: OnceLock<Option<usize>> = OnceLock::new();
/// `intobject.py descr_str` / `descr_repr`.
static INT_DESCR_STR_JITCODE_INDEX: OnceLock<Option<usize>> = OnceLock::new();
/// `baseobjspace::getitem_str` (`unicodeobject.py descr_getitem`).
static STR_GETITEM_JITCODE_INDEX: OnceLock<Option<usize>> = OnceLock::new();
/// `objspace.py StdObjSpace.newutf8` / `W_UnicodeObject.__init__`.
static NEWUTF8_JITCODE_INDEX: OnceLock<Option<usize>> = OnceLock::new();
/// Cached `ALL_JITCODES` index of the interpreter-source
/// `load_super_attr_value_w` body. The fused opcode reaches this ordinary
/// graph so `_super_check` and `W_Super.getattribute` are traced from their
/// generated JitCodes, as PyPy traces their RPython owners.
static LOAD_SUPER_ATTR_VALUE_JITCODE_INDEX: OnceLock<Option<usize>> = OnceLock::new();

/// Scan the build-time names index for the unique entry equal to `name`.
///
/// Returns `None` when no entry matches (e.g. compact test inputs that
/// omit the helper). Panics on a duplicate leaf name — the by-name
/// resolution model assumes the charon helper's path-last-segment name
/// is unique within the pipeline, so a collision signals a structural
/// regression that must surface immediately rather than silently pick
/// the wrong body.
fn compute_named_jitcode_index(name: &str) -> Option<usize> {
    let mut hits = jitcode_index()
        .names
        .iter()
        .enumerate()
        .filter(|(_, jitcode_name)| *jitcode_name == name)
        .map(|(i, _)| i);
    let first = hits.next();
    if hits.next().is_some() {
        panic!(
            "pyre-jit-trace: build-time pipeline has more than one jitcode \
             named `{name}`; the list-op by-name resolver assumes the \
             charon helper leaf name is unique"
        );
    }
    first
}

/// Resolve a JitCode by the graph key the codewriter allocated it under,
/// spelled as `CallPath::canonical_key` — the `::`-joined segments, e.g.
/// `pyre_interpreter::opcode_ops::compare_value`.
///
/// `CallControl::get_jitcode` keys `self.jitcodes` by `CallPath`, mirroring
/// `call.py get_jitcode`'s graph-keyed dict, and takes the JitCode's `name`
/// from the path's last segment only so dumps stay readable. Two graphs may
/// therefore share a name — 238 groups do — and the by-name resolver has no
/// way to tell them apart, so it panics rather than pick a body. Resolving by
/// the allocation key has no such ambiguity: `IndexMap<CallPath, _>` makes it
/// unique by construction.
///
/// `None` when the pipeline holds no such graph, so a caller can decline
/// rather than assume the helper was compiled.
pub(crate) fn compute_pathed_jitcode_index(canonical_path: &str) -> Option<usize> {
    if canonical_path.is_empty() {
        return None;
    }
    PATHED_JITCODE_INDEX.get(canonical_path).copied()
}

/// Process-wide `CallControl.jitcodes` keyed by `CallPath::canonical_key`.
///
/// RPython keeps one dict on `CallControl` (`call.py get_jitcode`); pyre's
/// frozen `jitcode_index().paths` is that dict's key column, allocation-aligned
/// with `ALL_JITCODES`. This map is the reverse lookup, built once, shared by
/// every thread. Empty paths are omitted: those jitcodes were minted without a
/// graph key. A duplicate path is a broken `IndexMap<CallPath, _>` invariant.
static PATHED_JITCODE_INDEX: LazyLock<indexmap::IndexMap<&'static str, usize>> =
    LazyLock::new(|| {
        let paths = &jitcode_index().paths;
        let mut by_path = indexmap::IndexMap::with_capacity(paths.len());
        for (index, path) in paths.iter().enumerate() {
            if path.is_empty() {
                continue;
            }
            match by_path.entry(path.as_str()) {
                indexmap::map::Entry::Vacant(slot) => {
                    slot.insert(index);
                }
                indexmap::map::Entry::Occupied(_) => panic!(
                    "pyre-jit-trace: build-time pipeline has more than one jitcode \
                     under path `{path}`; CallControl.jitcodes keys each graph once"
                ),
            }
        }
        by_path
    });

/// The JitCode allocated under `canonical_path`, or `None` when the build-time
/// pipeline does not contain that graph. See [`compute_pathed_jitcode_index`].
pub(crate) fn pathed_jitcode(canonical_path: &str) -> Option<Arc<JitCode>> {
    get_jitcode_by_index(compute_pathed_jitcode_index(canonical_path)?)
}

/// [`pathed_jitcode`] for a descent's fixed target.  The descents this
/// serves are rows of a table (`specialize.rs UnaryDescent`); the path
/// is a literal and [`PATHED_JITCODE_INDEX`] is the process-wide dict.
pub(crate) fn pathed_jitcode_cached(canonical_path: &str) -> Option<Arc<JitCode>> {
    pathed_jitcode(canonical_path)
}

/// Runtime-wrapper sibling of [`pathed_jitcode_cached`], for a codewriter
/// `inline_call_*` operand.  RPython's JitCode is both the body and its
/// `AbstractDescr`; pyre keeps the serializable canonical body separate from
/// the runtime descr pool, so this is the single identity-preserving join.
pub fn pathed_runtime_jitcode_cached(
    canonical_path: &str,
) -> Option<Arc<majit_metainterp::jitcode::JitCode>> {
    get_runtime_jitcode_by_index(compute_pathed_jitcode_index(canonical_path)?)
}

/// Resolve an ordinary portal-closure JitCode by its unique graph leaf name.
/// Prefer stable graph paths at build time; this runtime helper exists for
/// diagnostics and tests whose serialized artifact stores names only.
#[allow(dead_code)]
pub(crate) fn named_jitcode(name: &str) -> Option<Arc<JitCode>> {
    get_jitcode_by_index(compute_named_jitcode_index(name)?)
}

/// The charon `w_list_append` body in `ALL_JITCODES`, resolved by name
/// and cached process-wide. `None` if the helper is absent from the
/// build-time pipeline. See `LIST_APPEND_JITCODE_INDEX`.
pub fn list_append_jitcode() -> Option<Arc<JitCode>> {
    let idx = (*LIST_APPEND_JITCODE_INDEX
        .get_or_init(|| compute_named_jitcode_index("w_list_append_inner")))?;
    get_jitcode_by_index(idx)
}

/// The lock-guard-free charon `w_list_pop_end_inner` body in `ALL_JITCODES`,
/// resolved by name and cached process-wide. "Lock" is the whole of it: the body
/// holds no `w_list_lock` pair, which is what would decline the fold's
/// sub-walk (`listobject.rs` `w_list_pop_end`). It says nothing about trace
/// guards — for those see `try_walker_orthodox_list_pop`, whose soundness
/// turns on where they land relative to the body's first store.
pub fn list_pop_end_jitcode() -> Option<Arc<JitCode>> {
    let idx = (*LIST_POP_END_JITCODE_INDEX
        .get_or_init(|| compute_named_jitcode_index("w_list_pop_end_inner")))?;
    get_jitcode_by_index(idx)
}

/// Generated `typeobject.py write_cell` body.  Leaf name is unique in the
/// pipeline (one graph).  `None` when the helper is absent (compact tests).
pub fn write_cell_jitcode() -> Option<Arc<JitCode>> {
    let idx =
        (*WRITE_CELL_JITCODE_INDEX.get_or_init(|| compute_named_jitcode_index("write_cell")))?;
    get_jitcode_by_index(idx)
}

/// Generated `typeobject.py unwrap_cell` body.  Leaf name is unique.
pub fn unwrap_cell_jitcode() -> Option<Arc<JitCode>> {
    let idx =
        (*UNWRAP_CELL_JITCODE_INDEX.get_or_init(|| compute_named_jitcode_index("unwrap_cell")))?;
    get_jitcode_by_index(idx)
}

/// The lock-guard-free charon `w_list_setitem_inner` body in `ALL_JITCODES`,
/// resolved by name and cached process-wide. Same lock-vs-trace-guard
/// split as [`list_pop_end_jitcode`]: the body holds no `w_list_lock` pair.
pub fn list_setitem_jitcode() -> Option<Arc<JitCode>> {
    let idx = (*LIST_SETITEM_JITCODE_INDEX
        .get_or_init(|| compute_named_jitcode_index("w_list_setitem_inner")))?;
    get_jitcode_by_index(idx)
}

/// The lock-guard-free charon `w_list_getitem_inner` body in `ALL_JITCODES`.
pub fn list_getitem_jitcode() -> Option<Arc<JitCode>> {
    let idx = (*LIST_GETITEM_JITCODE_INDEX
        .get_or_init(|| compute_named_jitcode_index("w_list_getitem_inner")))?;
    get_jitcode_by_index(idx)
}

/// The wrapped-name interpreter-source value half of `LOAD_SUPER_ATTR`, resolved by the
/// graph key the codewriter allocated it under and cached process-wide.
pub fn load_super_attr_value_jitcode() -> Option<Arc<JitCode>> {
    let idx = (*LOAD_SUPER_ATTR_VALUE_JITCODE_INDEX.get_or_init(|| {
        compute_pathed_jitcode_index("pyre_interpreter::eval::load_super_attr_value_w")
    }))?;
    get_jitcode_by_index(idx)
}

/// The charon `w_tuple_getitem` body in `ALL_JITCODES`, resolved by the graph
/// key the codewriter allocated it under and cached process-wide. `None` if the
/// helper is absent from the build-time pipeline.
///
/// This is the reader behind every tuple subscript. It normalises a negative
/// index against the length, range-checks it, and dispatches the four
/// `W_AbstractTupleObject` layouts (`tupleobject.rs`) -- the canonical items
/// block and the three arity-2 specialisations -- applying each
/// specialisation's `wraps[i]` (`specialisedtupleobject.py getitem`).
///
/// Its `Option<PyObjectRef>` result lowers to a plain nullable ref
/// (`result_type: 'r'`), so an out-of-range index comes back null rather than
/// through a second result channel. Descending it rather than the
/// `_known` reader is what keeps the length test in the trace: the caller's
/// trace-time range check proves only that THIS receiver is long enough, while
/// the recorded `arraylen` guard is what holds for the next one.
pub fn tuple_getitem_jitcode() -> Option<Arc<JitCode>> {
    let idx = (*TUPLE_GETITEM_JITCODE_INDEX.get_or_init(|| {
        compute_pathed_jitcode_index("pyre_object::tupleobject::w_tuple_getitem")
    }))?;
    get_jitcode_by_index(idx)
}

/// `intobject.py descr_str` / `descr_repr` (`ll_int2dec` + `newutf8`).
pub fn int_descr_str_jitcode() -> Option<Arc<JitCode>> {
    let idx = (*INT_DESCR_STR_JITCODE_INDEX
        .get_or_init(|| compute_pathed_jitcode_index("pyre_object::intobject::descr_str")))?;
    get_jitcode_by_index(idx)
}

/// `baseobjspace::getitem_str` — `descr_getitem` after `getindex_w`.
/// `w_str_getitem` is the inlined reader inside this graph; the
/// rtyper does not keep a separate jitcode for that small helper.
pub fn str_getitem_jitcode() -> Option<Arc<JitCode>> {
    let idx = (*STR_GETITEM_JITCODE_INDEX.get_or_init(|| {
        compute_pathed_jitcode_index("pyre_interpreter::baseobjspace::getitem_str")
    }))?;
    get_jitcode_by_index(idx)
}

/// `space.newutf8` — wrap a `STR` payload with an explicit code-point count.
pub fn newutf8_jitcode() -> Option<Arc<JitCode>> {
    let idx = (*NEWUTF8_JITCODE_INDEX.get_or_init(|| {
        compute_pathed_jitcode_index("pyre_object::unicodeobject::w_str_from_storage_and_length")
    }))?;
    get_jitcode_by_index(idx)
}

/// Deserialized `pipeline.insns` overlaid with `extension_insns()`
/// — the build-observed `Assembler::write_insn` emit set plus the
/// `_pyre/P` adapter keys that pyre's production blackhole builder
/// registers via `setup_insns(...)` at runtime.  Both sources must
/// contribute to this static map so that `decode_op_at`
/// (`opcode_byte → opname` lookup) and the production dispatch path
/// (`build_inline_call_only_bh_builder`) agree on which bytes are
/// known.
///
/// `Assembler::get_opnum` mirrors RPython
/// `assembler.py setdefault(key, len(self.insns))`: keys present
/// in `majit_jitcode::insns::{wellknown_bh_insns, extension_insns}`
/// reuse their reserved `BC_*` byte for build/runtime stability, and
/// translator-only keys outside the canonical universe get the lowest
/// available non-reserved dynamic byte.  Both kinds land here verbatim,
/// so a key in this map is NOT a guarantee of canonical pinning — only
/// the byte the build observed gets serialised.  `JitCode.code[i]`
/// bytes can be mapped back to opnames through the inverted view
/// exposed by `opname_for_byte`.  Matches RPython `setup_insns(insns)`
/// consumption at `pyjitpl.py`.
///
/// Static justification: this is not a process-global mutable cache.
/// `insns.bin` is a frozen build artifact emitted alongside the
/// jitcodes whose byte streams it decodes, so every runtime frame in this
/// binary must see the same immutable opname -> byte table.  RPython keeps
/// the equivalent `Assembler.insns` object on the translated staticdata /
/// blackhole-builder path; pyre's `LazyLock` is the binary-embedded form
/// of that same single translated table.
/// Build-time `pipeline.insns` exactly as the assembler serialised it —
/// the opnames `make_jitcodes` actually emitted into this binary's
/// jitcodes, with no canonical-universe overlay.
///
/// This is the direct analogue of upstream's `asm.insns`, which is what
/// `BlackholeInterpBuilder.__init__` hands to `setup_insns`
/// (`blackhole.py`): upstream registers exactly what was emitted,
/// so its dispatch table covers the whole reachable bytecode universe by
/// construction.  Any byte in this map that the production blackhole
/// builder leaves unregistered is a byte a real jitcode can carry and the
/// blackhole cannot execute — see
/// `production_bh_builder_covers_every_build_emitted_opname`.
static BUILD_EMITTED_INSNS: LazyLock<indexmap::IndexMap<String, u8>> = LazyLock::new(|| {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/insns.bin"));
    bincode::deserialize(BYTES).unwrap_or_else(|e| {
        panic!(
            "pyre-jit-trace: failed to deserialize insns.bin ({} bytes): {e}",
            BYTES.len(),
        )
    })
});

/// The build-emitted opname → byte table (upstream `asm.insns`).
pub fn build_emitted_insns() -> &'static indexmap::IndexMap<String, u8> {
    &BUILD_EMITTED_INSNS
}

static INSNS_OPNAME_TO_BYTE: LazyLock<indexmap::IndexMap<String, u8>> = LazyLock::new(|| {
    let mut table = BUILD_EMITTED_INSNS.clone();
    // Overlay the canonical `wellknown_bh_insns` and pyre-only
    // `extension_insns` keys so the runtime table covers every
    // opname a `BlackholeInterpBuilder` could be asked to dispatch.
    //
    // RPython parity: `blackhole.py BlackholeInterpBuilder.__init__`
    // populates `self.insns = asm.insns` and then `wire_bhimpl_handlers`
    // (`:152-179`) iterates that map to bind every bhimpl handler.
    // Pyre's build-time `pipeline.insns` only records opnames the
    // assembler actually emitted during `make_jitcodes`; canonical
    // opnames the analyzed source set did not exercise (e.g.
    // `ref_guard_value/r`) are absent.  Overlaying
    // both `wellknown_bh_insns` and `extension_insns` restores
    // the closed key universe RPython's runtime sees, so callers
    // (`build_default_bh_builder_with_unwired_report`, dispatch
    // tests) treat opname coverage as a property of the codebase,
    // not of which paths the build observed.
    fn overlay_insns(
        table: &mut indexmap::IndexMap<String, u8>,
        source: &indexmap::IndexMap<&'static str, u8>,
    ) {
        for (key, byte) in source.iter() {
            let owned = (*key).to_string();
            if let Some(&prev) = table.get(&owned) {
                assert_eq!(
                    prev, *byte,
                    "insns overlay: opname {key:?} disagrees with build-time \
                         pipeline.insns (build={prev}, overlay={byte})",
                );
            } else {
                table.insert(owned, *byte);
            }
        }
    }
    overlay_insns(&mut table, &majit_jitcode::insns::wellknown_bh_insns());
    overlay_insns(&mut table, &majit_jitcode::insns::extension_insns());
    table
});

/// Inverted view: `u8` opcode byte → opname string.  Built lazily on
/// first access from `INSNS_OPNAME_TO_BYTE`.  Upstream `assembler.py:
/// 220` assigns a fresh byte to every distinct key (`setdefault(key,
/// len(self.insns))`), so the reverse map is one-to-one and panics on
/// any duplicate-byte collision.
static INSNS_BYTE_TO_OPNAME: LazyLock<HashMap<u8, String>> = LazyLock::new(|| {
    let mut map: HashMap<u8, String> = HashMap::with_capacity(INSNS_OPNAME_TO_BYTE.len());
    for (name, &byte) in INSNS_OPNAME_TO_BYTE.iter() {
        if let Some(existing) = map.insert(byte, name.clone()) {
            panic!(
                "INSNS_BYTE_TO_OPNAME: duplicate byte {byte} maps to both {existing:?} and \
                 {name:?} (upstream Assembler.insns is 1:1; if both spellings need to \
                 dispatch to the same handler, allocate distinct bytes per assembler.py:220)",
            );
        }
    }
    map
});

/// What a runtime-allocated opcode byte decodes to, if anything.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RuntimeInsn {
    /// Every publication of this byte named the same key.
    Unique(&'static str),
    /// Two `Assembler`s numbered the same byte differently, so the byte alone
    /// does not identify an instruction — see [`RUNTIME_INSNS_BYTE_TO_OPNAME`].
    Ambiguous,
}

/// Opcode bytes the runtime assembler allocated after `INSNS_BYTE_TO_OPNAME`
/// was fixed.
///
/// `assembler.py` grows `Assembler.insns` with `setdefault(key,
/// len(self.insns))`, so upstream's table is whatever the run has emitted so
/// far.  Pyre freezes the build-time half into `insns.bin` and lets
/// `record_insn_key` allocate the rest on demand — the `USE_C_FORM` shapes
/// (`int_sub/ci>i`, `goto_if_not_int_gt/icL`, ...) have no fixed `BC_*` byte
/// and only appear once a graph uses them.  Those bytes must reach
/// [`decode_op_at`] too: it resolves every opcode through the byte -> opname
/// map, and an unknown byte makes the whole body undecodable. Callers treat
/// that as an unreadable instruction.
///
/// **The numbering is per-`Assembler`, not global.** `record_insn_key` picks
/// the lowest unreserved byte this assembler has not used, so first-appearance
/// order decides, and two assemblers that met different shapes first disagree.
/// `codewriter.py:21-23` keeps ONE `Assembler` for the session and pyre's
/// `CodeWriter` mirrors that with a single `RefCell<Assembler>`, so the
/// production numbering is in fact global — but this table cannot assume it.
/// A byte published under two keys becomes [`RuntimeInsn::Ambiguous`] and the
/// decoder goes back to reporting it as undecodable, which is what it did for
/// every runtime byte before this table existed.  Guessing one of the two would
/// hand a caller the wrong instruction, which "assume the worst" never does.
///
/// Keys are leaked so the decoder can keep handing out `&'static str` exactly
/// as it does for the build-time table.  The set is small and bounded by the
/// unreserved opcode bytes.
static RUNTIME_INSNS_BYTE_TO_OPNAME: LazyLock<std::sync::RwLock<HashMap<u8, RuntimeInsn>>> =
    LazyLock::new(|| std::sync::RwLock::new(HashMap::new()));

/// Publish one runtime-allocated `(opname/argcodes, byte)` pair to the decoder.
///
/// Idempotent.  A byte that ends up naming two different keys — including one
/// the build-time table already owns — is recorded as ambiguous instead.
pub fn publish_runtime_insn(key: &str, byte: u8) {
    let build_time = INSNS_BYTE_TO_OPNAME.get(&byte).map(String::as_str);
    if build_time == Some(key) {
        // Already resolvable, and to this very key.
        return;
    }
    let mut table = RUNTIME_INSNS_BYTE_TO_OPNAME
        .write()
        .expect("runtime insn table poisoned");
    let changed = match table.entry(byte) {
        std::collections::hash_map::Entry::Vacant(vacant) => {
            let insn = match build_time {
                // The build-time table claims this byte for something else, so no
                // single key can answer for it.
                Some(_) => RuntimeInsn::Ambiguous,
                None => RuntimeInsn::Unique(Box::leak(key.to_string().into_boxed_str())),
            };
            vacant.insert(insn);
            true
        }
        std::collections::hash_map::Entry::Occupied(mut occupied) => match *occupied.get() {
            RuntimeInsn::Ambiguous => false,
            RuntimeInsn::Unique(recorded) if recorded == key => false,
            RuntimeInsn::Unique(_) => {
                *occupied.get_mut() = RuntimeInsn::Ambiguous;
                true
            }
        },
    };
    drop(table);
    if changed {
        rebuild_decode_snap();
    }
}

/// Pre-split `opname/argcodes` for one opcode byte. `blackhole.py`
/// `setup_insns` builds the dispatch table once; the walker indexes it.
#[derive(Clone, Copy)]
struct DecodeInsn {
    key: &'static str,
    opname: &'static str,
    argcodes: &'static str,
}

struct DecodeSnap {
    by_byte: [Option<DecodeInsn>; 256],
}

/// Snapshot of the byte → insn table. Rebuilt when `publish_runtime_insn`
/// grows `Assembler.insns`; the walker loads it with Acquire and never
/// takes the runtime `RwLock`.
static DECODE_SNAP: AtomicPtr<DecodeSnap> = AtomicPtr::new(std::ptr::null_mut());

fn decode_insn_from_key(key: &'static str) -> DecodeInsn {
    let (opname, argcodes) = split_key(key);
    DecodeInsn {
        key,
        opname,
        argcodes,
    }
}

fn rebuild_decode_snap() {
    let runtime = RUNTIME_INSNS_BYTE_TO_OPNAME
        .read()
        .expect("runtime insn table poisoned");
    let mut by_byte = [None; 256];
    for (byte, key) in INSNS_BYTE_TO_OPNAME.iter() {
        by_byte[*byte as usize] = Some(decode_insn_from_key(key.as_str()));
    }
    for (byte, insn) in runtime.iter() {
        match insn {
            RuntimeInsn::Unique(key) => {
                by_byte[*byte as usize] = Some(decode_insn_from_key(key));
            }
            RuntimeInsn::Ambiguous => {
                by_byte[*byte as usize] = None;
            }
        }
    }
    let snap = Box::new(DecodeSnap { by_byte });
    // Walkers may still be reading the previous table; leak it. Rebuilds
    // happen only when `publish_runtime_insn` grows `Assembler.insns`.
    let _old = DECODE_SNAP.swap(Box::into_raw(snap), Ordering::Release);
}

fn insn_for_opcode_byte(byte: u8) -> Option<DecodeInsn> {
    let mut ptr = DECODE_SNAP.load(Ordering::Acquire);
    if ptr.is_null() {
        rebuild_decode_snap();
        ptr = DECODE_SNAP.load(Ordering::Acquire);
        if ptr.is_null() {
            return None;
        }
    }
    unsafe { (*ptr).by_byte[byte as usize] }
}

/// RPython `setup_insns(insns)` — full opname → opcode-byte table.
pub fn insns_opname_to_byte() -> &'static indexmap::IndexMap<String, u8> {
    &INSNS_OPNAME_TO_BYTE
}

/// Inverse lookup: `JitCode.code[i]` byte → opname. None for unknown
/// bytes (e.g. operand bytes, not opcode bytes).
pub fn opname_for_byte(byte: u8) -> Option<&'static str> {
    INSNS_BYTE_TO_OPNAME.get(&byte).map(String::as_str)
}

/// Inverse of `insns_opname_to_byte()` — full `u8 -> opname/argcodes` table.
pub fn insns_byte_to_opname() -> &'static HashMap<u8, String> {
    &INSNS_BYTE_TO_OPNAME
}

/// Indexed `pipeline.descrs` — RPython `Assembler.descrs`
/// (assembler.py:23). Handed to `BlackholeInterpBuilder.setup_descrs`
/// at builder construction (blackhole.py `self.setup_descrs(asm.descrs)`,
/// :102-103 `def setup_descrs(self, descrs): self.descrs = descrs`).
///
/// Each 'd'/'j' argcode in a `JitCode.code` byte stream is a 2-byte
/// little-endian index into this pool. The resolved `BhDescr` is what
/// every `bhimpl_*` handler reads for field offsets, call descriptors,
/// sub-JitCodes, and switch dicts.
struct DescrIndex {
    offsets: Box<[u32]>,
    /// `0` for descriptors owned by `GcCache`, `1` for Call/JitCode, and
    /// `2` for opcode-only descriptors such as `SwitchDictDescr`.
    /// Mirrors the groups consumed by `rehydrate_build_descr_raw_sets`.
    kinds: Box<[u8]>,
    /// Dense reference into `descr_layouts.bin`, or `u32::MAX` for a
    /// descriptor without a parent.  This is the frozen-image spelling of
    /// PyPy's `FieldDescr.parent_descr` reference to the one SizeDescr owned
    /// by `GcCache`; it is not a runtime semantic side table.
    parent_layouts: Box<[u32]>,
    /// `BhDescr::Size.type_id` for a parentless size slot, else 0.
    /// `descr.py` `get_size_descr` returns the cached row, so a size whose
    /// STRUCT the parent-layout walk already numbered is not parsed again.
    size_type_ids: Box<[u64]>,
}

fn load_descr_index() -> DescrIndex {
    const INDEX_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/descrs_index.bin"));
    const BODY_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/descrs.bin"));
    let (offsets, kinds, parent_layouts, size_type_ids): (Vec<u32>, Vec<u8>, Vec<u32>, Vec<u64>) =
        bincode::deserialize(INDEX_BYTES).unwrap_or_else(|e| {
            panic!(
                "pyre-jit-trace: failed to deserialize descrs_index.bin \
             ({} bytes): {e}",
                INDEX_BYTES.len(),
            )
        });
    assert!(!offsets.is_empty());
    assert_eq!(offsets.first().copied(), Some(0));
    assert_eq!(offsets.last().copied(), Some(BODY_BYTES.len() as u32));
    assert!(offsets.windows(2).all(|pair| pair[0] <= pair[1]));
    assert_eq!(kinds.len() + 1, offsets.len());
    assert_eq!(parent_layouts.len(), kinds.len());
    assert_eq!(size_type_ids.len(), kinds.len());
    assert!(kinds.iter().all(|kind| matches!(kind, 0 | 1 | 2)));
    DescrIndex {
        offsets: offsets.into_boxed_slice(),
        kinds: kinds.into_boxed_slice(),
        parent_layouts: parent_layouts.into_boxed_slice(),
        size_type_ids: size_type_ids.into_boxed_slice(),
    }
}

fn descrs_index() -> &'static DescrIndex {
    static INDEX: OnceLock<DescrIndex> = OnceLock::new();
    INDEX.get_or_init(load_descr_index)
}

fn descr_count() -> usize {
    descrs_index().kinds.len()
}

fn descr_layout_offsets() -> &'static [u32] {
    static OFFSETS: OnceLock<Box<[u32]>> = OnceLock::new();
    OFFSETS.get_or_init(|| {
        const INDEX_BYTES: &[u8] =
            include_bytes!(concat!(env!("OUT_DIR"), "/descr_layouts_index.bin"));
        const BODY_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/descr_layouts.bin"));
        let offsets: Vec<u32> = bincode::deserialize(INDEX_BYTES).unwrap_or_else(|e| {
            panic!(
                "pyre-jit-trace: failed to deserialize descr_layouts_index.bin \
                 ({} bytes): {e}",
                INDEX_BYTES.len(),
            )
        });
        assert!(!offsets.is_empty());
        assert_eq!(offsets.first().copied(), Some(0));
        assert_eq!(offsets.last().copied(), Some(BODY_BYTES.len() as u32));
        assert!(offsets.windows(2).all(|pair| pair[0] <= pair[1]));
        offsets.into_boxed_slice()
    })
}

fn descr_layout_at(index: usize) -> std::sync::Arc<majit_jitcode::jitcode::BhSizeSpec> {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/descr_layouts.bin"));
    let offsets = descr_layout_offsets();
    let start = offsets[index] as usize;
    let end = offsets[index + 1] as usize;
    // PyPy retains the canonical `SizeDescr` in `GcCache._cache_size`, not
    // the codewriter-to-runtime transport object that described it. Keep a
    // layout only when the lazy opcode `BhDescr::Field` itself keeps the Arc;
    // the eager GcCache replay drops this wire value after publication.
    //
    // Deliberately NOT interned behind a per-index `OnceLock`: two fields
    // naming one layout index do each decode their own copy, but those copies
    // are transport and die with the field that holds them.  A cache would
    // instead pin every layout it ever decoded for the life of the process,
    // trading a bounded transient for permanent retention — the opposite of
    // what the canonical-owner rule above is for, and against the startup-RSS
    // work that motivated it.  Share the canonical object through
    // `_cache_size`, not through this decoder.
    let (spec, consumed) = majit_jitcode::jitcode::BhSizeSpec::unpack_from(&BYTES[start..end]);
    assert_eq!(
        consumed,
        end - start,
        "descr_layouts.bin entry {index} did not fill {start}..{end}"
    );
    std::sync::Arc::new(spec)
}

/// Parent layouts translation already stored, in table order.
///
/// `warmspot.py` `WarmRunnerDesc.finish` calls `pyjitpl.py`
/// `finish_setup_descrs` on the rows it built. Walking this table publishes
/// those rows. [`descr_layout_at`] stays for a later Field slot that still
/// needs one parent; kind-0 startup does not call it.
fn for_each_packed_parent_layout(mut visit: impl FnMut(majit_jitcode::jitcode::BhSizeSpec)) {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/descr_layouts.bin"));
    let mut rest = BYTES;
    while !rest.is_empty() {
        let (spec, consumed) = majit_jitcode::jitcode::BhSizeSpec::unpack_from(rest);
        rest = &rest[consumed..];
        visit(spec);
    }
}

/// Publish each packed parent once.
///
/// `descr.py` `get_field_descr` already filled a declared STRUCT. When that
/// size lists the same fields, skip the name bytes (`pyjitpl.py`
/// `finish_setup_descrs` numbers stored rows; it does not rebuild them).
fn publish_packed_parent_layouts() {
    const BYTES: &'static [u8] = include_bytes!(concat!(env!("OUT_DIR"), "/descr_layouts.bin"));
    // `descr.py` `get_size_descr` sees groups translation already built.
    crate::descr::force_all_declared_groups();
    // Rows `get_size_descr` already returned. Later records of the same
    // layout compare this table instead of locking `GcCache` again.
    let mut known = crate::descr::kind0_vtable_field_counts();
    // Same answer as `kind0_struct_absent`, without a lock per record.
    // A publish below inserts the id so a later record of that STRUCT
    // takes `publish_kind0_parent_layout`, matching the cache after the mint.
    let mut present = crate::descr::kind0_present_struct_ids();
    let mut rest: &'static [u8] = BYTES;
    while !rest.is_empty() {
        let (type_id, nfields) = majit_jitcode::jitcode::BhSizeSpec::peek_header(rest);
        // `kind0_vtable_field_counts` is the one `get_size_descr` pass.
        // `finish_setup_descrs` does not lock again to rediscover a vtable
        // row. A packed record whose cached size has no vtable is not a hit:
        // `kind0_layout_already_published` returns false for `vtable == 0`.
        let known_hit = known
            .binary_search_by_key(&type_id, |row| row.0)
            .is_ok_and(|index| known[index].1 == nfields);
        if type_id == 0 || known_hit {
            let consumed = majit_jitcode::jitcode::BhSizeSpec::skip_record(rest);
            rest = &rest[consumed..];
            continue;
        }
        let gained_vtable = if present.binary_search(&type_id).is_err() {
            let (layout, consumed) = majit_jitcode::jitcode::BhSizeSpec::read_static(rest);
            rest = &rest[consumed..];
            let gained_vtable = layout.vtable != 0;
            crate::descr::publish_borrowed_parent_layout(layout);
            gained_vtable
        } else {
            let (spec, consumed) = majit_jitcode::jitcode::BhSizeSpec::unpack_from(rest);
            rest = &rest[consumed..];
            let gained_vtable = spec.vtable != 0;
            crate::descr::publish_kind0_parent_layout(spec);
            gained_vtable
        };
        if let Err(index) = present.binary_search(&type_id) {
            present.insert(index, type_id);
        }
        // A vtable-less publish cannot enter the skip table. Locking
        // `get_size_descr` to confirm that is the per-row cost this pass
        // already refused.
        if gained_vtable && crate::descr::kind0_layout_already_published(type_id, nfields) {
            match known.binary_search_by_key(&type_id, |row| row.0) {
                Ok(index) => known[index].1 = nfields,
                Err(index) => known.insert(index, (type_id, nfields)),
            }
        }
    }
}

fn load_descr_uncached(index: usize) -> BhDescr {
    load_descr_with_parent(index, descr_layout_at)
}

/// Decode one `descrs.bin` slot.
///
/// A parentless `BhDescr::Size` whose `type_id` is nonzero is the STRUCT row
/// `descr.py` `get_size_descr` returns. `finish_setup_descrs` already numbered
/// that row, so the bytes are a `BhSizeSpec` record (`pack_into`), not a
/// bincode enum. Every other slot stays a bincode `BhDescr`.
fn descr_from_indexed_bytes(bytes: &[u8], size_type_id: u64) -> BhDescr {
    if size_type_id != 0 {
        let (spec, consumed) = majit_jitcode::jitcode::BhSizeSpec::unpack_from(bytes);
        assert_eq!(
            consumed,
            bytes.len(),
            "packed size record did not fill its descrs.bin slot"
        );
        assert_eq!(spec.type_id, size_type_id);
        let owner = if spec.headerless && spec.owner.is_empty() {
            majit_jitcode::jitcode::HEADERLESS_SIZE_OWNER_MARKER.to_string()
        } else {
            spec.owner
        };
        return BhDescr::Size {
            size: spec.size,
            type_id: spec.type_id,
            vtable: spec.vtable,
            owner,
            all_fielddescrs: spec.all_fielddescrs,
            is_gc_managed: spec.is_gc_managed,
        };
    }
    bincode::deserialize(bytes).unwrap_or_else(|e| {
        panic!(
            "pyre-jit-trace: failed to deserialize descrs.bin entry ({} bytes): {e}",
            bytes.len(),
        )
    })
}

/// Decode descriptor slot `index`, taking a Field's parent layout from
/// `parent_layout_at` instead of decoding it anew.
fn load_descr_with_parent(
    index: usize,
    parent_layout_at: impl FnOnce(usize) -> std::sync::Arc<majit_jitcode::jitcode::BhSizeSpec>,
) -> BhDescr {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/descrs.bin"));
    let offsets = &descrs_index().offsets;
    let start = offsets[index] as usize;
    let end = offsets[index + 1] as usize;
    let mut descr =
        descr_from_indexed_bytes(&BYTES[start..end], descrs_index().size_type_ids[index]);
    let parent_layout = descrs_index().parent_layouts[index];
    if parent_layout != u32::MAX {
        let BhDescr::Field { parent, .. } = &mut descr else {
            panic!("descriptor slot {index} carries a parent layout but is not a Field");
        };
        assert!(
            parent.is_none(),
            "wire Field unexpectedly embeds its parent layout"
        );
        *parent = Some(parent_layout_at(parent_layout as usize));
    }
    // A `new_with_vtable` size descr carries a type-static sentinel in
    // `vtable` and the static's name in `owner`. Every consumer of the
    // loaded descr — tracing `BC_NEW_WITH_VTABLE`, `bh_new_with_vtable` —
    // reads `get_vtable()`, so the runtime address has to be in place
    // before the cell is published.
    crate::runtime_fnaddr_patch::rebind_type_static_size_vtable(&mut descr);
    descr
}

fn descr_cells() -> &'static [OnceLock<&'static BhDescr>] {
    static CELLS: OnceLock<&'static [OnceLock<&'static BhDescr>]> = OnceLock::new();
    CELLS.get_or_init(|| {
        let cells: Vec<OnceLock<&'static BhDescr>> =
            (0..descr_count()).map(|_| OnceLock::new()).collect();
        let cells = Box::leak(cells.into_boxed_slice());
        assert_eq!(
            cells.len() + 1,
            descrs_index().offsets.len(),
            "pyre-jit-trace: descr cell count must match the indexed entry count",
        );
        cells
    })
}

pub fn get_descr_by_index(index: usize) -> Option<&'static BhDescr> {
    let cell = descr_cells().get(index)?;
    Some(*cell.get_or_init(|| Box::leak(Box::new(load_descr_uncached(index)))))
}

struct LazyDescrTable;

impl DescrTable for LazyDescrTable {
    fn get(&'static self, index: usize) -> Option<&'static BhDescr> {
        get_descr_by_index(index)
    }

    fn len(&self) -> usize {
        descr_count()
    }
}

static LAZY_DESCR_TABLE: LazyDescrTable = LazyDescrTable;

pub fn descr_table() -> &'static dyn DescrTable {
    &LAZY_DESCR_TABLE
}

/// RPython: `metainterp_sd.opcode_descrs` (`pyjitpl.py setup_descrs`) — the
/// bytecode constant pool, not `metainterp_sd.all_descrs`.
///
/// `all_descrs` upstream is `cpu.setup_descrs()` (`pyjitpl.py`), the full
/// gccache walk of `descr.py`; pyre's counterpart of *that* is
/// `MetaInterpStaticData::finish_setup_descrs`, which enumerates the live
/// `descr_registry`. The gap between the two tables is what
/// [`load_ei_descr_mints`] carries.
#[cfg(test)]
pub fn all_descrs() -> &'static [BhDescr] {
    static MATERIALIZED: OnceLock<&'static [BhDescr]> = OnceLock::new();
    MATERIALIZED.get_or_init(|| {
        Box::leak(
            (0..descr_count())
                .map(load_descr_uncached)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        )
    })
}

/// Canonical translated EffectInfo objects. RPython embeds each object once
/// and lets CallDescrs retain references to it; the indexed wire table carries
/// that identity across pyre's build-script/runtime process boundary.
fn effect_info_offsets() -> &'static [u32] {
    static OFFSETS: OnceLock<Box<[u32]>> = OnceLock::new();
    OFFSETS.get_or_init(|| {
        const INDEX_BYTES: &[u8] =
            include_bytes!(concat!(env!("OUT_DIR"), "/effect_infos_index.bin"));
        const BODY_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/effect_infos.bin"));
        let offsets: Vec<u32> = bincode::deserialize(INDEX_BYTES).unwrap_or_else(|e| {
            panic!(
                "pyre-jit-trace: failed to deserialize effect_infos_index.bin \
                 ({} bytes): {e}",
                INDEX_BYTES.len(),
            )
        });
        assert!(!offsets.is_empty());
        assert_eq!(offsets.first().copied(), Some(0));
        assert_eq!(offsets.last().copied(), Some(BODY_BYTES.len() as u32));
        assert!(offsets.windows(2).all(|pair| pair[0] <= pair[1]));
        offsets.into_boxed_slice()
    })
}

fn effect_info_count() -> usize {
    effect_info_offsets().len() - 1
}

fn load_effect_info(index: usize) -> (u32, majit_ir::EffectInfo) {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/effect_infos.bin"));
    let offsets = effect_info_offsets();
    let start = offsets[index] as usize;
    let end = offsets[index + 1] as usize;
    bincode::deserialize(&BYTES[start..end]).unwrap_or_else(|e| {
        panic!(
            "pyre-jit-trace: failed to deserialize effect_infos.bin entry {index} \
             ({start}..{end} of {} bytes): {e}",
            BYTES.len(),
        )
    })
}

/// Indexed `pipeline.ei_descr_mints` — gccache slots named only by an
/// EffectInfo raw set, paired with the arguments their mint took.
///
/// See `descr::publish_effect_info_descr_mints` for why the opcode table alone
/// leaves these slots empty on this side of the build/runtime split.  Entries
/// are decoded independently so setup never retains all 2,000+ string-heavy
/// mint specs at once.
fn ei_descr_mint_offsets() -> &'static [u32] {
    static OFFSETS: OnceLock<Box<[u32]>> = OnceLock::new();
    OFFSETS.get_or_init(|| {
        const INDEX_BYTES: &[u8] =
            include_bytes!(concat!(env!("OUT_DIR"), "/ei_descr_mints_index.bin"));
        const BODY_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ei_descr_mints.bin"));
        let offsets: Vec<u32> = bincode::deserialize(INDEX_BYTES).unwrap_or_else(|e| {
            panic!(
                "pyre-jit-trace: failed to deserialize ei_descr_mints_index.bin \
                 ({} bytes): {e}",
                INDEX_BYTES.len(),
            )
        });
        assert!(!offsets.is_empty());
        assert_eq!(offsets.first().copied(), Some(0));
        assert_eq!(offsets.last().copied(), Some(BODY_BYTES.len() as u32));
        assert!(offsets.windows(2).all(|pair| pair[0] <= pair[1]));
        offsets.into_boxed_slice()
    })
}

fn ei_descr_mint_count() -> usize {
    ei_descr_mint_offsets().len() - 1
}

fn load_ei_descr_mint(index: usize) -> majit_ir::effectinfo::DescrMintEntry {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ei_descr_mints.bin"));
    let offsets = ei_descr_mint_offsets();
    let start = offsets[index] as usize;
    let end = offsets[index + 1] as usize;
    bincode::deserialize(&BYTES[start..end]).unwrap_or_else(|e| {
        panic!(
            "pyre-jit-trace: failed to deserialize ei_descr_mints.bin entry {index} \
             ({start}..{end} of {} bytes): {e}",
            BYTES.len(),
        )
    })
}

fn ei_descr_stamp_offsets() -> &'static [u32] {
    static OFFSETS: OnceLock<Box<[u32]>> = OnceLock::new();
    OFFSETS.get_or_init(|| {
        const INDEX_BYTES: &[u8] =
            include_bytes!(concat!(env!("OUT_DIR"), "/ei_descr_stamps_index.bin"));
        const BODY_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ei_descr_stamps.bin"));
        let offsets: Vec<u32> = bincode::deserialize(INDEX_BYTES).unwrap_or_else(|e| {
            panic!(
                "pyre-jit-trace: failed to deserialize ei_descr_stamps_index.bin \
                 ({} bytes): {e}",
                INDEX_BYTES.len(),
            )
        });
        assert!(!offsets.is_empty());
        assert_eq!(offsets.first().copied(), Some(0));
        assert_eq!(offsets.last().copied(), Some(BODY_BYTES.len() as u32));
        assert!(offsets.windows(2).all(|pair| pair[0] <= pair[1]));
        offsets.into_boxed_slice()
    })
}

fn ei_descr_stamp_count() -> usize {
    ei_descr_stamp_offsets().len() - 1
}

fn load_ei_descr_stamp(index: usize) -> (majit_ir::effectinfo::DescrSetMember, u32) {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ei_descr_stamps.bin"));
    let offsets = ei_descr_stamp_offsets();
    let start = offsets[index] as usize;
    let end = offsets[index + 1] as usize;
    bincode::deserialize(&BYTES[start..end]).unwrap_or_else(|e| {
        panic!(
            "pyre-jit-trace: failed to deserialize ei_descr_stamps.bin entry {index} \
             ({start}..{end} of {} bytes): {e}",
            BYTES.len(),
        )
    })
}

/// Analyzer-side release census captured at the end of the build-script
/// translation. The live process adds its own counters before formatting the
/// existing field-position stats line.
static BUILD_FIELD_MINT_CENSUS: LazyLock<majit_ir::descr::FieldMintCensus> = LazyLock::new(|| {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/field_mint_census.bin"));
    bincode::deserialize(BYTES).unwrap_or_else(|e| {
        panic!(
            "pyre-jit-trace: failed to deserialize field_mint_census.bin \
                 ({} bytes): {e}",
            BYTES.len(),
        )
    })
});

/// Deserialized `pipeline.all_liveness` — RPython `Assembler.all_liveness`
/// (assembler.py), the target of `pyjitpl.py:2264 self.liveness_info =
/// "".join(asm.all_liveness)`.
///
/// The build-time codewriter dedups every `(live_i, live_r, live_f)` triple
/// into this single byte stream and bakes each `BC_LIVE` op's 2-byte offset
/// into `JitCode.code`.  A runtime consumer re-tracing a build-time jitcode
/// (jd1's `_unpackiterable_unknown_length` merge-point walk) resolves those
/// baked offsets by installing this table into `metainterp_sd.liveness_info`.
static ALL_LIVENESS: LazyLock<Vec<u8>> = LazyLock::new(|| {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/liveness.bin"));
    bincode::deserialize(BYTES).unwrap_or_else(|e| {
        panic!(
            "pyre-jit-trace: failed to deserialize liveness.bin \
             ({} bytes): {e}",
            BYTES.len(),
        )
    })
});

/// RPython: `metainterp_sd.liveness_info` — full shared `all_liveness`
/// byte stream (see [`ALL_LIVENESS`]).
pub fn all_liveness() -> &'static [u8] {
    &ALL_LIVENESS
}

fn call_descr_arg_types(arg_classes: &str) -> Vec<majit_ir::Type> {
    arg_classes
        .chars()
        .filter_map(|c| match c {
            'i' | 'S' => Some(majit_ir::Type::Int),
            'r' => Some(majit_ir::Type::Ref),
            'f' | 'L' => Some(majit_ir::Type::Float),
            _ => None,
        })
        .collect()
}

fn call_descr_result_type(result_type: char) -> majit_ir::Type {
    match result_type {
        'i' | 'S' => majit_ir::Type::Int,
        'r' => majit_ir::Type::Ref,
        'f' | 'L' => majit_ir::Type::Float,
        _ => majit_ir::Type::Void,
    }
}

fn rehydrated_call_descr_ref(bh: majit_jitcode::jitcode::BhCallDescr) -> majit_ir::DescrRef {
    let arg_types = call_descr_arg_types(&bh.arg_classes);
    let result_type = call_descr_result_type(bh.result_type);
    // This BhCallDescr was decoded solely for setup.  Move its EffectInfo
    // into the canonical runtime CallDescr instead of cloning the complete
    // six-set serialization payload and then dropping the source copy.
    majit_metainterp::make_call_descr_sized_with_translated_effect(
        &bh.arg_classes,
        &arg_types,
        result_type,
        // `descr.py get_result_type()` keeps the raw char, so a
        // rehydrated descr reports `'S'`/`'L'` rather than the class its
        // normalised `result_type` derives.
        bh.result_type,
        bh.result_signed,
        bh.result_size,
        bh.translated_effect_info_id
            .expect("translated BhCallDescr is missing its EffectInfo identity"),
    )
}

/// Publish the GcCache-owned (non-call) opcode descrs and stamp synthetic
/// struct collector tids before a trace walk.
///
/// `GcLLDescr_framework.init_size_descr` asks `TypeLayoutBuilder.get_type_id`
/// for those ids during translation, against Size objects already in
/// `GcCache`. pyre cannot embed the collector ids in the executable.
/// `publish_kind0_descrs_before_trace` calls this before
/// `force_start_tracing` / `bound_reached`. Publishing from inside the
/// walk is too late: Windows `frame_chain` then allocates about 2.2 TiB
/// (exit 3221226505). A process that never traces never calls this.
/// Field minting publishes the parent Size `init_size_descr` reads.
/// `set_type_registry_close_hook` runs this before `freeze_types`, so the
/// registry is still open when the tids are registered. CallDescr
/// restoration stays on the first slot lookup.
pub fn materialize_gccache_owned_descrs() {
    // The close hook and a trace that starts without `init_jit_hooks` can
    // already be inside recursive `CALL_ASSEMBLER`. That stack must not
    // run the bincode.
    materialize_gccache_owned_descrs_with(false);
}

/// Same publication as [`materialize_gccache_owned_descrs`], decoded on
/// this thread.
///
/// The `Once` is shared with the helper-thread path: whichever caller
/// runs first chooses the stack, and the other call only registers tids.
pub fn materialize_gccache_owned_descrs_on_caller_stack() {
    materialize_gccache_owned_descrs_with(true);
}

fn materialize_gccache_owned_descrs_with(on_caller_stack: bool) {
    static ONCE: Once = Once::new();
    // `heaptracker.setup_cache_gcstruct2vtable` before `get_size_descr`.
    // Kind-0 decode mints analyzer field descrs with `vtable=0`; the
    // table has to be populated first so those SizeDescrs are born
    // `is_object()`.
    crate::descr::publish_pyre_class_vtables();
    ONCE.call_once(|| {
        if on_caller_stack {
            decode_kind0_descrs();
        } else {
            decode_kind0_descrs_off_caller_stack();
        }
    });
    // The decode `Once` may have run before a collector existed. Register
    // once the live collector is installed; a second call is a no-op.
    register_synthetic_struct_tids();
    // Collector tids may have been `UNASSIGNED` on the first walk.
    crate::descr::refresh_pyre_class_vtable_tids();
}

/// Kind-0 bincode runs on a fresh 8 MiB stack.
///
/// `frame_chain` closes the type registry from inside a recursive
/// `CALL_ASSEMBLER` chain. Decoding the table on that stack makes Windows
/// exit `3221226505` (`STATUS_STACK_BUFFER_OVERRUN`). A spawned thread's
/// stack is empty, which is the same room a Linux main thread has at boot.
/// `clone` is refused after seccomp; that path decodes on the caller.
///
/// Collector tids stay on this thread. `gc_op` is the GIL-backed borrow
/// of `malloc_fast` (`framework.py`); the helper thread does not hold it.
fn decode_kind0_descrs_off_caller_stack() {
    let spawned = std::thread::Builder::new()
        .name("gccache-kind0".into())
        .stack_size(8 * 1024 * 1024)
        .spawn(decode_kind0_descrs);
    match spawned {
        Ok(handle) => handle.join().expect("kind-0 descr decode"),
        Err(_) => decode_kind0_descrs(),
    }
}

fn decode_kind0_descrs() {
    let index = descrs_index();
    // `descr.py` `GcCache.setup_descrs` numbers rows already stored by
    // `get_field_descr`. `pyjitpl.py` `finish_setup_descrs` does that at
    // translation (`warmspot.py` `WarmRunnerDesc.finish`). The packed parent
    // table is those rows: publish each once, without `descr_layout_at`.
    // Size, Array, and InteriorField slots have no parent layout and still
    // decode.
    publish_packed_parent_layouts();
    // `descr.py` `get_size_descr` / `pyjitpl.py` `finish_setup_descrs`.
    // One snapshot, then a parentless size whose row is already numbered
    // is not locked or decoded again.
    let mut published = crate::descr::kind0_published_size_ids();
    for (i, kind) in index.kinds.iter().copied().enumerate() {
        if kind != 0 || index.parent_layouts[i] != u32::MAX {
            continue;
        }
        let type_id = index.size_type_ids[i];
        if type_id != 0 && published.binary_search(&type_id).is_ok() {
            continue;
        }
        let bh = load_descr_with_parent(i, descr_layout_at);
        debug_assert!(!matches!(
            bh,
            BhDescr::Call { .. } | BhDescr::JitCode { .. }
        ));
        crate::descr::make_descr_from_bh(&bh);
        if type_id != 0 && crate::descr::kind0_size_already_published(type_id) {
            if let Err(pos) = published.binary_search(&type_id) {
                published.insert(pos, type_id);
            }
        }
    }
}

/// Stamp collector tids onto synthetic struct `SizeDescr`s.
///
/// No-op until `gc_sync` has a collector: unit tests rehydrate the descr
/// table with no `MiniMarkGC`. Once a collector exists the registry must
/// still be open — `set_type_registry_close_hook` guarantees that in
/// production by running [`materialize_gccache_owned_descrs`] before
/// `freeze_types`.
fn register_synthetic_struct_tids() {
    if !majit_gc::gc_sync::is_initialized() {
        return;
    }
    // One registration per collector. `reset_gc_fresh_for_test` installs a
    // new `MiniMarkGC`; a process-wide `Once` would leave that registry
    // without the tids already stamped on the shared size descriptors.
    static LAST_GC: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    majit_gc::gc_sync::gc_op(|gc| {
        use majit_gc::GcAllocator;
        use std::sync::atomic::Ordering;
        let addr = gc as *mut majit_gc::collector::MiniMarkGC as usize;
        let previous = LAST_GC.swap(addr, Ordering::AcqRel);
        if previous == addr && gc.types_frozen() {
            return;
        }
        assert!(
            !gc.types_frozen(),
            "materialize_gccache_owned_descrs: synthetic struct tids must be \
             registered while the type registry is still open; \
             set_type_registry_close_hook runs this before freeze_types"
        );
        let mut cache = majit_ir::descr::gc_cache().lock();
        if previous == addr || previous == 0 {
            cache.register_unresolved_struct_tids(|size, offsets| {
                gc.register_type(majit_gc::trace::TypeInfo::with_gc_ptrs(size, offsets))
            });
            cache.register_unresolved_array_tids(|base, item, len, offs, is_ptr| {
                register_array_layout(gc, base, item, len, offs, is_ptr)
            });
            return;
        }
        cache.replay_synthetic_struct_tids(|size, offsets| {
            gc.register_type(majit_gc::trace::TypeInfo::with_gc_ptrs(size, offsets))
        });
        cache.replay_synthetic_array_tids(|base, item, len, offs, is_ptr| {
            register_array_layout(gc, base, item, len, offs, is_ptr)
        });
    });
}

/// `TypeLayoutBuilder.get_type_id(A)` then `encode_type_shape`. Host
/// `register_dict_entries` / int/float item arrays already occupy a
/// `TypeInfo` with the same `ofstovar` / `varitemsize` / `ofstolength` /
/// `varofstoinnergcptrs`. Two distinct `LLType` keys that denote that
/// same lltype therefore reuse the existing tid via
/// `TypeRegistry::find_varsize_type` instead of appending a second one.
fn register_array_layout(
    gc: &mut majit_gc::collector::MiniMarkGC,
    base_size: usize,
    item_size: usize,
    len_offset: usize,
    item_gc_offsets: Vec<usize>,
    is_ptr: bool,
) -> u32 {
    use majit_gc::GcAllocator;
    let var_offsets = if !item_gc_offsets.is_empty() {
        item_gc_offsets
    } else if is_ptr {
        vec![0]
    } else {
        Vec::new()
    };
    if let Some(tid) = gc
        .types
        .find_varsize_type(base_size, item_size, len_offset, &var_offsets)
    {
        return tid;
    }
    gc.register_type(majit_gc::trace::TypeInfo::varsize_with_gc_ptr_offsets(
        base_size,
        item_size,
        len_offset,
        var_offsets,
        Vec::new(),
    ))
}

/// Rehydrate build-time EffectInfo raw descr sets before
/// `finish_setup_descrs`. `finish_setup_done` is per-thread, but the
/// rehydrated raw sets live in the process-global `GcCache`, so this guard is
/// process-global.
///
/// # Why this stays eager, and what is actually known about its cost
///
/// It is the obvious candidate for "make the first JIT stop rebuilding the
/// whole descriptor universe", and it decodes a lot: 2,364 non-call slots,
/// 6,884 mint specs, 185 EffectInfos and 2,534 call slots — 11,967 bincode
/// decodes off 2.6 MB of embedded records.  Two things to know before taking
/// that on.
///
/// **This rehydrate is a first-JIT cost, not a startup one.**  It is reachable
/// only through `ensure_finish_setup`, and every caller of that sits on a JIT
/// path — [`crate::state::intern_liveness`],
/// [`crate::state::liveness_info_snapshot`], [`crate::state::op_live`],
/// [`crate::state::blackhole_control_opcodes`],
/// [`crate::state::setup_indirectcalltargets`],
/// [`crate::state::bytecode_for_address`].  An interpreter that never traces
/// never runs it.  Kind-0 minting is not in that set:
/// [`materialize_gccache_owned_descrs`] runs from
/// `publish_kind0_descrs_before_trace` before the portal walk, and from the
/// type-registry close hook. A process that never traces does not decode
/// the table. The call below is the `Once` hitting an already-published
/// table, then the EffectInfo and Call passes.
///
/// **Its size is not currently measured, and two obvious instruments cannot
/// measure it.**  Allocation here is mmap-backed, so `malloc_history` / `heap`
/// attribution does not see the object heap at all; and subtracting two peak
/// RSS readings does not isolate it either, because the JIT changes what the
/// workload itself allocates.  Measured on darwin arm64 against a 63.9 MB
/// never-traced baseline, a 200,000-iteration integer loop grows peak RSS by
/// 5.8 MB with the JIT on and 23.8 MB with `PYRE_JIT=off` — the JIT arm nets
/// activation against the boxing the interpreter avoided.  Sizing this needs
/// an instrument that attributes the arena.
///
/// What settles the question is soundness rather than size, and the dense-index
/// halves of the lazy plan already ship: `descrs_index.bin` /
/// `ei_descr_mints_index.bin` / `effect_infos_index.bin` give per-index
/// locatability, [`descr_cells`] / [`descr_ref_cells`] are the `OnceLock`
/// arrays, and `descr::force_declared_group` is the exact-key, on-demand
/// container-group publish.
///
/// What is left eager here is eager because it must be.
/// `descr::descr_from_set_member` is lookup-only by construction, and
/// `descr::prepare_frozen_effect_info` resolves each EffectInfo's six raw sets
/// exactly once and degrades the whole EffectInfo to `RandomEffects` when any
/// member is unresolvable — a container published afterwards cannot repair it.
/// Deferring pass 2 or pass 3 therefore buys an unmeasured amount of memory
/// against a silent optimizer downgrade that surfaces only as
/// `descr_set_absent` rising off zero.
/// Type id of the packed parent at `index` in `descr_layouts.bin`.
fn descr_layout_type_id(index: usize) -> u64 {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/descr_layouts.bin"));
    let start = descr_layout_offsets()[index] as usize;
    majit_jitcode::jitcode::BhSizeSpec::peek_header(&BYTES[start..]).0
}

/// `descr.py` `get_field_descr`: mint the opcode Field into
/// `_cache_field[STRUCT][fieldname]` with `parent_descr = get_size_descr`.
/// Packed parent `all_fielddescrs` already holds `heaptracker.all_fielddescrs`.
fn publish_kind0_field(struct_id: u64, field_name: &str) {
    let index = descrs_index();
    for (slot, &layout_index) in index.parent_layouts.iter().enumerate() {
        if index.kinds[slot] != 0 || layout_index == u32::MAX {
            continue;
        }
        if descr_layout_type_id(layout_index as usize) != struct_id {
            continue;
        }
        let bh = load_descr_with_parent(slot, descr_layout_at);
        let field_key = match &bh {
            majit_jitcode::jitcode::BhDescr::Field { owner, name, .. } => {
                crate::descr::bh_field_cache_key(owner, name)
            }
            _ => continue,
        };
        if field_key != field_name {
            continue;
        }
        let _ = crate::descr::make_descr_from_bh(&bh);
    }
}

pub fn rehydrate_build_descr_raw_sets() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        // `descr.py::GcCache.setup_descrs` group order — every non-call slot
        // first. Each `Size` / `Field` entry publishes its parent's FULL
        // `heaptracker.all_fielddescrs(STRUCT)` list into the gccache, and
        // `descr_from_set_member` is lookup-only, so the raw-set members
        // below can only land on slots that already carry their complete
        // layout.  Resolving in the other order would leave every member
        // whose struct has not been published yet unresolvable.
        //
        // `materialize_gccache_owned_descrs` publishes the Size objects
        // `init_size_descr` walks and stamps their collector tids. EffectInfo
        // and Call restoration stay in this pass, off the startup path.
        materialize_gccache_owned_descrs();
        // Members already published by the opcode-descr pass need only the
        // `compute_bitstrings` partition stamp. PyPy mutates that one cached
        // descriptor object in place; replaying the complete mint spec would
        // re-run a cache-miss recipe against a slot that is already full.
        for i in 0..ei_descr_stamp_count() {
            let (member, ei_index) = load_ei_descr_stamp(i);
            if let Some((struct_id, field_name)) = crate::descr::ambiguous_field_struct(&member) {
                publish_kind0_field(struct_id, field_name);
            }
            crate::descr::stamp_effect_info_descr(&member, ei_index);
        }
        // Last, and only into what is still empty: the slots no opcode names,
        // which `descrs.bin` — RPython's `opcode_descrs`, not its `all_descrs`
        // — therefore does not carry.  Without them a raw-set member whose
        // container appears in no bytecode reads as `AbsentContainer` even
        // though the analyzer minted it.
        //
        // After the loop above rather than before it, because both are
        // build-time producers that disagree on `index_in_parent` for a shared
        // field, and going first would mean pre-filling a slot the loop then
        // asks for with its own numbering.  Publishing what the established
        // producers left over keeps their answers untouched.
        // These mint specs are a one-shot wire format, not runtime state.
        // Upstream's descriptors remain in GcCache; the temporary arguments
        // passed to get_*_descr do not.  Drop the decoded strings/specs as
        // soon as their canonical runtime descriptors have been published.
        for i in 0..ei_descr_mint_count() {
            let entry = load_ei_descr_mint(i);
            crate::descr::publish_effect_info_descr_mints(std::slice::from_ref(&entry));
        }
        // `effectinfo.compute_bitstrings` has already run in the translator.
        // Resolve the structural spellings once per canonical EffectInfo,
        // retain only its compact bitstrings/single-write descriptor, and
        // publish the object table before any CallDescr asks for an id.
        for i in 0..effect_info_count() {
            let (translated_id, mut effect_info) = load_effect_info(i);
            crate::runtime_fnaddr_patch::rewrite_call_release_gil_target(&mut effect_info);
            crate::descr::prepare_frozen_effect_info(&mut effect_info);
            majit_ir::effectinfo::intern_translated_effect_info(translated_id, effect_info);
        }
        // Call/JitCode slots stay lazy across the Rust build/runtime-image
        // boundary. PyPy's `descr.py get_call_descr` caches these during
        // translation, and `GcCache.setup_descrs` only enumerates them.
        // [`descr_ref_at`] restores a requested slot through
        // `rehydrated_call_descr_ref` after this pass publishes its frozen
        // EffectInfo identity, preserving the canonical call-cache key.
        report_descr_spelling_gate();
        // EffectInfo mints can create size descriptors after the first
        // registration. Stamp those onto the still-open collector.
        register_synthetic_struct_tids();
    });
}

/// Name every member the ledger could not join, under
/// `PYRE_DESCR_SPELLING_GATE=1`.
///
/// The counts themselves ride the `[jit-stats]` line
/// ([`descr_set_jit_stats`]), which is what gates in CI. This is the detail
/// behind them — the counterpart of `effectinfo.py:474-499`, where
/// `compute_bitstrings` logs its per-key descr tallies through
/// `policy.log` while expressing the impossible cases as bare `assert`s.
///
/// The whole serialized raw-set universe is resolved inside the `Once` above
/// and nowhere else, so this runs exactly once with the ledger complete.
fn report_descr_spelling_gate() {
    if std::env::var_os("PYRE_DESCR_SPELLING_GATE").is_none() {
        return;
    }
    let ledger = crate::descr::set_member_ledger();
    for label in &ledger.ambiguous {
        eprintln!("[descr-spelling-gate] ambiguous {label}");
    }
    for label in &ledger.absent {
        eprintln!("[descr-spelling-gate] absent {label}");
    }
}

/// The descr-universe invariants, as `[jit-stats]` key/value tokens.
///
/// All three of `absent`, `ambiguous` and `stale_absent` are answers upstream
/// cannot reach — `effectinfo.py:492-494` builds its sets out of descr objects
/// `cpu.*descrof` just minted, so no member of a raw set can fail to resolve —
/// so `check.py` lists them in `JITSTATS_BADNESS_FIELDS` and fails the run if
/// any rises off zero. That is pyre's form of the bare `assert` upstream states
/// this class of condition with (`descr.py:47`, `effectinfo.py:486`, `:525`),
/// which the build-time/runtime split makes reachable and therefore
/// unassertable.
///
/// `resolved` is the denominator, and rides along so a run that resolves
/// nothing cannot look identical to one that resolves everything.
///
/// `stale_absent` re-asks the `AbsentContainer` question against the universe
/// as it stands now, so it must be read at process exit, after the run has
/// published everything it is going to.
pub fn descr_set_jit_stats() -> String {
    let DescrSetCounts {
        resolved,
        absent,
        ambiguous,
        stale_absent,
    } = descr_set_counts();
    format!(
        "descr_set_resolved={resolved} descr_set_absent={absent} \
         descr_set_ambiguous={ambiguous} descr_set_stale_absent={stale_absent}"
    )
}

/// The field-position census, as `[jit-stats]` key/value tokens.
///
/// This sits next to `descr_set_*` because it answers the question those
/// counters only appear to answer. `descr_set_absent` asks whether a raw-set
/// member resolved to SOME descr; it reads 0 while a field is bound under a
/// parent that numbers its `all_fielddescrs` by a different convention, because
/// resolving and resolving CORRECTLY are different questions. Upstream cannot
/// tell them apart either — but it does not have to, since `heaptracker.py
/// get_fielddescr_index_in` and `:96-112 all_fielddescrs` are one walker sharing
/// one skip set, so `all_fielddescrs(S)[i].get_index() == i` holds by
/// construction and there is nothing to count.
///
/// `attached_misplaced` and `spec_misplaced` are the two to watch. Both count a
/// producer-supplied `index_in_parent` that disagrees with where the field
/// actually sits, read before anything downstream can normalise it, and between
/// them they cover the two shapes a producer can emit: `spec_misplaced` a
/// parent's own positional list, `attached_misplaced` a standalone field descr
/// pointing into one. The second is not implied by the first — a producer builds
/// a list by enumerating what it just sorted, so the list is self-consistent by
/// construction while a descr minted against an earlier state of it is not.
///
/// `rederived` looks like it asks that question and does not. `derive_index_in_parent`
/// is the judge and the repairman: it replaces the caller's number with the
/// parent's in the same expression, so a nonzero reading is a log of repairs
/// already applied, never a defect still present. Worse, it counts only the
/// mints it reached — `parent_absent` is the ones where it never asked, and that
/// is the *majority* of them, because
/// `make_simple_descr_group_keyed_with_headerless` mints every field before
/// `register_keyed_size` publishes the parent those fields will be indexed
/// against. `field_pos_rederived=0` therefore states nothing about the
/// producers; it has to be read as a fraction of `parent_absent`, and a
/// producer defect can sit at zero forever. `spec_misplaced` is the same defect
/// measured where no reader has had the chance to repair it.
///
/// The defect being measured: `all_fielddescrs()[index_in_parent]` is a
/// load-bearing lookup (`optimizeopt/info.rs force_box`), so an index naming a
/// different slot than the field occupies either runs off the end or emits the
/// store against a DIFFERENT field.
///
/// `positional_misplaced` is the output-side companion — the same predicate on
/// the published list, after `get_field_descr` has reconciled it. The gap
/// between the two is exactly how much the reader is absorbing.
///
/// `size_shell_*` is the producer-side companion. `parent_empty` counts mint
/// attempts that found a fieldless parent, so one shell hit by many fields reads
/// as many; `size_shell_shadowing` counts the shells themselves, and only those
/// for which this same cache already holds fields under the key. That is the
/// state `descr.py` cannot reach: `get_size_descr` returns on cache hit, so a
/// shell published first outranks the real layout for the rest of the run.
pub fn field_position_jit_stats() -> String {
    // Through `field_position_counts` rather than the censuses directly, so
    // this line and the wasm exports cannot describe different populations.
    let FieldPositionCounts {
        parent_absent,
        parent_empty,
        rederived,
        unresolved,
        spec_checked,
        spec_misplaced,
        attached_checked,
        attached_misplaced,
    } = field_position_counts();
    let mint = *BUILD_FIELD_MINT_CENSUS + majit_ir::descr::field_mint_census_snapshot();
    let mint_fields = mint
        .fields()
        .into_iter()
        .map(|(name, value)| {
            let report_name = if let Some(suffix) = name.strip_prefix("cache_hit_") {
                format!("field_cache_hit_{suffix}")
            } else if let Some(suffix) = name.strip_prefix("offset_") {
                format!("field_offset_{suffix}")
            } else if let Some(suffix) = name.strip_prefix("struct_size_") {
                format!("compute_struct_size_{suffix}")
            } else if let Some(suffix) = name.strip_prefix("ei_") {
                format!("ei_descr_mint_{suffix}")
            } else if name == "owner_id_registry_miss" {
                "field_owner_id_registry_miss".to_owned()
            } else {
                name.to_owned()
            };
            format!("{report_name}={value}")
        })
        .collect::<Vec<_>>()
        .join(" ");
    let (
        [published, fieldless, shadowing, aliased, aliased_multi],
        [slots, misplaced],
        [key_compared, key_conflicting],
        sample,
    ) = {
        let gc = majit_ir::descr::gc_cache();
        let guard = gc.lock();
        let sample = if std::env::var_os("MAJIT_SIZE_SHELL_OWNERS").is_some() {
            let lines = guard.size_shell_owner_sample(24);
            format!("\n[jit-stats] {}", lines.join("\n[jit-stats] "))
        } else {
            String::new()
        };
        (
            guard.size_shell_census(),
            guard.positional_invariant_census(),
            guard.identity_collision_census(),
            sample,
        )
    };
    format!(
        "field_pos_parent_absent={parent_absent} field_pos_parent_empty={parent_empty} \
         field_pos_rederived={rederived} field_pos_unresolved={unresolved} \
         field_pos_spec_checked={spec_checked} field_pos_spec_misplaced={spec_misplaced} \
         field_pos_attached_checked={attached_checked} \
         field_pos_attached_misplaced={attached_misplaced} {mint_fields} \
         size_shell_published={published} size_shell_fieldless={fieldless} \
         size_shell_shadowing={shadowing} size_shell_aliased={aliased} \
         size_shell_aliased_multi={aliased_multi} \
         positional_slots={slots} positional_misplaced={misplaced} \
         key_compared={key_compared} key_conflicting={key_conflicting}{sample}",
    )
}

/// The same four numbers [`descr_set_jit_stats`] formats, as numbers.
///
/// Backends that cannot print a line reach the counters through here rather than
/// re-deriving them, so the gated values cannot drift from the printed ones. The
/// wasm guest has no stderr and exports these individually
/// (`pyre_jit_descr_set_*` in `pyre-wasm`), which the runner prints on its
/// behalf.
pub fn descr_set_counts() -> DescrSetCounts {
    let ledger = crate::descr::set_member_ledger();
    DescrSetCounts {
        resolved: ledger.resolved as u64,
        absent: ledger.absent.len() as u64,
        ambiguous: ledger.ambiguous.len() as u64,
        stale_absent: crate::descr::stale_absent_containers().len() as u64,
    }
}

/// The descr-universe invariants as counts. `resolved` is the denominator; the
/// other three are `JITSTATS_BADNESS_FIELDS` members and healthy only at zero.
pub struct DescrSetCounts {
    pub resolved: u64,
    pub absent: u64,
    pub ambiguous: u64,
    pub stale_absent: u64,
}

/// Name the `field_pos_unresolved` mints, under `MAJIT_FIELD_POS_UNRESOLVED=1`.
///
/// Same split as `report_descr_spelling_gate`: the count rides the gated
/// `[jit-stats]` line and this is the detail behind it. The count says the slot
/// hazard was reached; only the names say by whom, and that decides whether the
/// fix belongs at one producer or at the resolution layer.
///
/// `all_descrs` is the caller's — `metainterp_sd.all_descrs`, the table
/// `pyrex` already reads for its own diag line, not this module's
/// [`all_descrs`] opcode pool. It rides along as the denominator that says the
/// descr universe was loaded at all: an empty table under a nonzero
/// `all_descrs` is a clean tree, and an empty table under a zero one is a run
/// that never got far enough to have an answer.
///
/// Empty when the knob is unset — a bare zero is never printed for a census
/// that did not run. How many rows it prints is the knob's own value
/// (`MAJIT_FIELD_POS_UNRESOLVED=<n>`, `1` for all of them).
pub fn field_position_unresolved_report(all_descrs: usize) -> Vec<String> {
    if !majit_ir::descr::field_position_unresolved_naming_enabled() {
        return Vec::new();
    }
    majit_ir::descr::GcCache::field_position_unresolved_sample(all_descrs)
}

/// The two producer-side field-position invariants as numbers, for the same
/// reason [`descr_set_counts`] exists: the wasm guest has no stderr, so it
/// exports them individually (`pyre_jit_field_pos_*` in `pyre-wasm`) and the
/// runner prints the line.
///
/// This matters more on wasm than the name suggests. The invariant is stated in
/// terms of BYTE OFFSETS — `index_in_parent` must name the slot the field's
/// offset occupies — and wasm32's word is 4 bytes (`symbolic.py WORD =
/// sizeof(lltype.Signed)`), so every struct is laid out differently there. A
/// producer that ranks correctly on a 64-bit host is not thereby correct on
/// wasm, and without these exports a wasm-only rise reads as absent-and-
/// therefore-zero, i.e. healthy.
pub fn field_position_counts() -> FieldPositionCounts {
    let [parent_absent, parent_empty, rederived, unresolved] =
        majit_ir::descr::GcCache::field_position_census();
    let [spec_checked, spec_misplaced] = majit_ir::descr::GcCache::spec_position_census();
    let [attached_checked, attached_misplaced] =
        majit_ir::descr::GcCache::attached_position_census();
    FieldPositionCounts {
        parent_absent: parent_absent as u64,
        parent_empty: parent_empty as u64,
        rederived: rederived as u64,
        unresolved: unresolved as u64,
        spec_checked: spec_checked as u64,
        spec_misplaced: spec_misplaced as u64,
        attached_checked: attached_checked as u64,
        attached_misplaced: attached_misplaced as u64,
    }
}

/// `*_checked` are the denominators — reported so a run that checked nothing
/// cannot read the same as one that checked everything, but host-dependent and
/// therefore not in `JITSTATS_SNAPSHOT_FIELDS`. The two `*_misplaced` are
/// `JITSTATS_BADNESS_FIELDS` members and healthy only at zero.
///
/// The four census members are the `derive_index_in_parent` dispositions, and
/// they are here rather than in a second struct so that the printed line and
/// the wasm exports cannot describe different populations: `field_position_jit_stats`
/// now formats THIS value instead of re-reading the censuses itself. Two
/// independent readers of the same three counters could only ever agree by
/// convention, and the convention is what rots — the printed line is what a
/// human reads and the exports are what the gate reads, so a drift between them
/// is invisible from either side.
pub struct FieldPositionCounts {
    pub parent_absent: u64,
    pub parent_empty: u64,
    pub rederived: u64,
    pub unresolved: u64,
    pub spec_checked: u64,
    pub spec_misplaced: u64,
    pub attached_checked: u64,
    pub attached_misplaced: u64,
}

/// Name every member whose container has been registered since its raw set was
/// frozen, under `PYRE_DESCR_SPELLING_GATE=1`.
///
/// Each one is a live instance of the gap documented on
/// `SetMemberLookup::AbsentContainer`: an `EffectInfo` frozen while its
/// container was unregistered still claims the callee does not touch it. The
/// count is gated through [`descr_set_jit_stats`]; this is the detail.
pub fn descr_spelling_gate_recheck_now() {
    if std::env::var_os("PYRE_DESCR_SPELLING_GATE").is_none() {
        return;
    }
    for label in &crate::descr::stale_absent_containers() {
        eprintln!("[descr-spelling-gate] stale_absent {label}");
    }
}

/// Lazy pool of `DescrRef`s indexed alongside [`descr_table`] so the
/// trace-side jitcode walker
/// ([`crate::jitcode_dispatch::dispatch_via_miframe`]) can resolve each
/// `d`/`j` argcode operand to a real `Arc<dyn Descr>`. Every entry runs
/// through [`crate::descr::make_descr_from_bh`] — the per-variant
/// adapter that maps each `BhDescr` shape to its RPython-orthodox
/// counterpart on the metainterp side:
///
/// - `Field` → `make_field_descr` / immutable variants with full
///   offset, size, type, signedness, and purity flags.
/// - `Array` → `make_array_descr` with full base size, item size,
///   type id, item type, signedness, and struct-array classification.
/// - `Size` → `make_size_descr_with_type_and_vtable`.
/// - `Call` → `make_call_descr_with_effect` — `arg_classes` /
///   `result_type` are reshaped into typed inputs/output AND
///   `extra_info` is threaded through so RPython
///   `call.py:320 effectinfo_from_writeanalyze` parity is preserved
///   (descr cache key + can_raise / oopspec / read-write descrs all
///   match the codewriter's stamp).
/// - `JitCode` → `make_jitcode_descr` (the existing adapter the
///   walker's `inline_call_*` recursion exercises).
///
/// `Switch` / `VableField` / `VableArray` / `VtableMethod` also get
/// concrete trace-side descriptor adapters so the pool shape stays
/// equivalent to `Assembler.descrs`.
///
/// Identity (`Arc::ptr_eq`) currently matches only for the `JitCode`
/// slot — content-derived adapters
/// for `Field` / `Array` / `Size` / `Call` produce fresh `Arc`s
/// per-resolution, so record sites that need the same
/// `Arc` instance still build their own at the call site
/// until the by-index identity factories land.
fn descr_ref_cells() -> &'static [OnceLock<DescrRef>] {
    static CELLS: OnceLock<&'static [OnceLock<DescrRef>]> = OnceLock::new();
    CELLS.get_or_init(|| {
        Box::leak(
            (0..descr_count())
                .map(|_| OnceLock::new())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        )
    })
}

/// Resolve one trace-side descriptor without forcing any other pool entry.
///
/// `rehydrate_build_descr_raw_sets` must remain first: its non-call pass
/// publishes every container group into the gccache and freezes EffectInfo
/// ids before a CallDescr can ask for one.  Once that ordering has
/// completed, Size/Field/Array reuse the cached `Arc` and Call/JitCode
/// mint on this first request (`descr.py get_call_descr`).
pub fn descr_ref_at(index: usize) -> Option<DescrRef> {
    rehydrate_build_descr_raw_sets();
    let cell = descr_ref_cells().get(index)?;
    Some(
        cell.get_or_init(|| {
            let bh = load_descr_uncached(index);
            match bh {
                BhDescr::Call { calldescr } => rehydrated_call_descr_ref(calldescr),
                // RPython `Transformer.handle_regular_call` uses the
                // callee's JitCode itself as the inline_call descriptor,
                // not its residual-call signature. Preserve its body index.
                other => crate::descr::make_descr_from_bh(&other),
            }
        })
        .clone(),
    )
}

struct LazyDescrRefTable;

impl crate::jitcode_dispatch::DescrRefTable for LazyDescrRefTable {
    fn at(&self, index: usize) -> Option<DescrRef> {
        descr_ref_at(index)
    }

    fn len(&self) -> usize {
        descr_count()
    }
}

static LAZY_DESCR_REF_TABLE: LazyDescrRefTable = LazyDescrRefTable;

pub fn descr_ref_table() -> &'static dyn crate::jitcode_dispatch::DescrRefTable {
    &LAZY_DESCR_REF_TABLE
}

/// S4c prerequisite measurement — how many build-time `BhDescr::Field` slots
/// resolve to the SAME `Arc` the runtime `descr.py get_field_descr`
/// cache holds for their `(STRUCT, fieldname)` key.
///
/// `effectinfo.py compute_bitstrings` partitions descrs by object
/// identity, so an `EffectInfo` raw set rehydrated from `descrs.bin` is only
/// meaningful if each member lands on the descr the trace itself caches.  A
/// slot that mints a fresh `Arc` instead is a silent mis-partition, which is
/// strictly worse than the missing raw set it would replace — hence this runs
/// before the format change, not after.
pub fn field_descr_identity_census_now() {
    rehydrate_build_descr_raw_sets();
    field_descr_identity_census();
}

/// Same-Arc test for two `DescrRef`s.  `Arc::ptr_eq` on `Arc<dyn Descr>`
/// compares the fat pointer (data + vtable); two upcasts of the same
/// allocation through the same concrete type agree on both halves, and an
/// upcast of a *different* concrete type must not compare equal anyway.
fn same_arc(a: &DescrRef, b: &DescrRef) -> bool {
    std::sync::Arc::as_ptr(a) as *const () == std::sync::Arc::as_ptr(b) as *const ()
}

/// Why a build-time `Field` slot fails to land on the canonical
/// `_cache_field[STRUCT][fieldname]` Arc.  `descr.py get_field_descr`
/// admits exactly one outcome — cache hit or cache-miss mint — so every
/// class below except `Converged` marks a place where pyre mints a second
/// FieldDescr for a `(STRUCT, fieldname)` PyPy keeps single.
#[derive(PartialEq, Eq, Hash, Clone, Copy, PartialOrd, Ord)]
enum FieldIdentityClass {
    Converged,
    /// `_cache_size[STRUCT]` empty — the parent was never published, so
    /// `get_field_descr` could not have set `parent_descr` either.
    NoParentSlot,
    /// Parent published, but `_cache_field` has no inner map for it:
    /// `heaptracker.all_fielddescrs` never ran through `get_field_descr`.
    NoFieldMap,
    /// Inner map exists but not under this spelling — pyre's cache key and
    /// its lookup key disagree (PyPy keys on `fieldname`, displays
    /// `'%s.%s' % (STRUCT._name, fieldname)`; pyre conflates the two).
    NameMiss,
    /// Entry exists and is a different Arc: two mint points for one field.
    DoubleMint,
}

impl FieldIdentityClass {
    fn label(self) -> &'static str {
        match self {
            Self::Converged => "converged",
            Self::NoParentSlot => "no _cache_size[STRUCT]",
            Self::NoFieldMap => "no _cache_field[STRUCT]",
            Self::NameMiss => "name not in _cache_field[STRUCT]",
            Self::DoubleMint => "different Arc (double mint)",
        }
    }
}

fn field_descr_identity_census() {
    use std::collections::BTreeMap;

    let (mut fields, mut keyed, mut parentless) = (0usize, 0usize, 0usize);
    // Pool-side (`field_descr_from_bh_field`) vs `_cache_field`.
    let mut pool_classes: BTreeMap<FieldIdentityClass, usize> = BTreeMap::new();
    // Walker-side (`field_descr_ref_from_bh`, the Arc actually baked into
    // recorded getfield/setfield ops) vs `_cache_field`.
    let mut walker_classes: BTreeMap<FieldIdentityClass, usize> = BTreeMap::new();
    // Pool-side vs walker-side: the split that makes an `EffectInfo` raw set
    // rehydrated from `descrs.bin` unable to reach the recorded op's descr.
    let mut pool_vs_walker_same = 0usize;
    // How often the pool Arc is one of the parent SizeDescr's
    // `all_fielddescrs()` (the second mint point).
    let mut pool_from_all_fielddescrs = 0usize;
    let mut samples: Vec<String> = Vec::new();

    for i in 0..descr_count() {
        let bh = load_descr_uncached(i);
        let BhDescr::Field {
            parent,
            name,
            owner,
            index_in_parent,
            ..
        } = &bh
        else {
            continue;
        };
        fields += 1;
        let Some(parent) = parent.as_ref().filter(|p| p.type_id != 0) else {
            parentless += 1;
            continue;
        };
        keyed += 1;
        let key = majit_ir::descr::LLType::Struct(parent.type_id);
        let (parent_size, cached, field_keys) = {
            let gc = majit_ir::descr::gc_cache().lock();
            let inner = gc._cache_field.get(&key);
            (
                gc._cache_size.get(&key).cloned(),
                inner.and_then(|m| m.get(name.as_str())).cloned(),
                inner.map(|m| {
                    let mut ks: Vec<String> = m.keys().cloned().collect();
                    ks.sort();
                    ks
                }),
            )
        };
        let cached_ref: Option<DescrRef> = cached.map(|fd| fd as DescrRef);
        let classify = |candidate: &DescrRef| -> FieldIdentityClass {
            match (&parent_size, &cached_ref, &field_keys) {
                (None, _, _) => FieldIdentityClass::NoParentSlot,
                (_, Some(c), _) if same_arc(c, candidate) => FieldIdentityClass::Converged,
                (_, Some(_), _) => FieldIdentityClass::DoubleMint,
                (_, None, None) => FieldIdentityClass::NoFieldMap,
                (_, None, Some(_)) => FieldIdentityClass::NameMiss,
            }
        };

        let pool = crate::descr::make_descr_from_bh(&bh);
        let (_, walker) = majit_metainterp::field_descr_ref_from_bh(&bh);
        let pool_class = classify(&pool);
        let walker_class = classify(&walker);
        *pool_classes.entry(pool_class).or_default() += 1;
        *walker_classes.entry(walker_class).or_default() += 1;
        if same_arc(&pool, &walker) {
            pool_vs_walker_same += 1;
        }
        if let Some(sd) = parent_size.as_ref().and_then(|p| p.as_size_descr()) {
            if sd
                .all_fielddescrs()
                .iter()
                .any(|fd| same_arc(&(fd.clone() as DescrRef), &pool))
            {
                pool_from_all_fielddescrs += 1;
            }
        }

        if pool_class != FieldIdentityClass::Converged && samples.len() < 25 {
            let n_all = parent_size
                .as_ref()
                .and_then(|p| p.as_size_descr())
                .map(|sd| sd.all_fielddescrs().len());
            samples.push(format!(
                "{owner}.{name}[{index_in_parent:?}] T{:#x} pool={} walker={} \
                 all_fielddescrs={n_all:?} _cache_field keys={:?}",
                parent.type_id,
                pool_class.label(),
                walker_class.label(),
                field_keys.as_deref().unwrap_or(&[]),
            ));
        }
    }

    eprintln!("[field-identity] {fields} Field slots: {keyed} keyed, {parentless} parentless");
    for (label, classes) in [("pool", &pool_classes), ("walker", &walker_classes)] {
        let rendered: Vec<String> = classes
            .iter()
            .map(|(c, n)| format!("{}={n}", c.label()))
            .collect();
        eprintln!(
            "[field-identity] {label} vs _cache_field: {}",
            rendered.join(", ")
        );
    }
    eprintln!(
        "[field-identity] pool==walker: {pool_vs_walker_same}/{keyed}; \
         pool Arc came from parent.all_fielddescrs(): {pool_from_all_fielddescrs}/{keyed}"
    );
    for s in &samples {
        eprintln!("[field-identity]   {s}");
    }
}

/// Test-only materialized view for fixtures that inspect the complete pool.
#[cfg(test)]
pub fn all_descr_refs() -> &'static [DescrRef] {
    static MATERIALIZED: OnceLock<&'static [DescrRef]> = OnceLock::new();
    MATERIALIZED.get_or_init(|| {
        Box::leak(
            (0..descr_count())
                .map(|i| descr_ref_at(i).unwrap())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        )
    })
}

/// Distinct dense descriptor indices a run actually resolves, recorded when
/// `PYRE_DESCR_DEMAND` is set.
///
/// Entries materialize only when resolved. Gated behind a `OnceLock` env read
/// so the resolve path pays nothing when the probe is off.
static DESCR_DEMAND: LazyLock<Mutex<std::collections::HashSet<usize>>> =
    LazyLock::new(|| Mutex::new(std::collections::HashSet::new()));

fn descr_demand_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("PYRE_DESCR_DEMAND").is_some())
}

pub fn record_descr_demand(index: usize) {
    if descr_demand_enabled() {
        DESCR_DEMAND.lock().insert(index);
    }
}

/// `(distinct indices resolved, pool size)`; `(0, _)` when the probe is off.
pub fn descr_demand_summary() -> (usize, usize) {
    let touched = DESCR_DEMAND.lock().len();
    (touched, descr_count())
}

/// Byte offset the build-time descr pool records for `{owner}.{name}`, or
/// `None` when no `Field` slot names that field.
///
/// [`all_descrs`] is serialised by `build.rs` from `majit-translate`'s layout
/// model, whose word size and Charon-resolved field offsets both follow the
/// target being compiled. A consumer that resolves `d` operands through this
/// pool therefore reads the same bytes `offset_of!` names — the
/// `build_time_offset_matches_runtime_layout` test pins that agreement for the
/// structs the orthodox sub-walk reads.
pub fn build_time_field_offset(owner: &str, name: &str) -> Option<usize> {
    (0..descr_count()).find_map(|i| {
        let descr = load_descr_uncached(i);
        match descr {
            BhDescr::Field {
                owner: o,
                name: n,
                offset,
                ..
            } if o == owner && n == name => Some(offset),
            _ => None,
        }
    })
}

/// Install the metainterp-side lazy `RuntimeBhDescr` view over the shared
/// build-time descriptor table as the
/// process-global build-time descr pool (`JitCode::descr_at`'s fallback for
/// LLBC-extracted jitcodes, whose per-jitcode `exec.descrs` is empty).
/// Mirrors RPython's single shared `Assembler.descrs`: build-time jitcodes
/// resolve every `d`/`j` argcode through this one pool.
///
/// Each `BhDescr::JitCode { jitcode_index, .. }` becomes a
/// `RuntimeBhDescr::JitCode` wrapping `ALL_JITCODES[jitcode_index]` via
/// `from_canonical` — the callee itself carries an empty per-jitcode pool
/// and so resolves its own descrs through this same global pool (no
/// circularity: the wrapper embeds no descrs). Every other `BhDescr` variant
/// is carried verbatim as `RuntimeBhDescr::Descr`, which is exactly what the
/// residual-call / getfield / setfield / switch dispatch arms read back via
/// `as_bh_descr()`.
///
/// Idempotent: the metainterp `OnceLock` keeps the first table, so repeated
/// calls (harness + production init) are safe. Individual entries are decoded
/// and retained only when `JitCode::descr_at` names their index.
struct RuntimeDescrCells(&'static [OnceLock<majit_metainterp::RuntimeBhDescr>]);

// SAFETY: the cells are write-once and this table only constructs `Descr` and
// `JitCode` entries. It never constructs the raw-pointer-bearing `Call` or
// `AssemblerToken` variants; this is the same invariant the former frozen
// `GlobalDescrPool` wrapper enforced.
unsafe impl Send for RuntimeDescrCells {}
unsafe impl Sync for RuntimeDescrCells {}

fn runtime_descr_cells() -> &'static [OnceLock<majit_metainterp::RuntimeBhDescr>] {
    static CELLS: OnceLock<RuntimeDescrCells> = OnceLock::new();
    CELLS
        .get_or_init(|| {
            RuntimeDescrCells(Box::leak(
                (0..descr_count())
                    .map(|_| OnceLock::new())
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            ))
        })
        .0
}

/// Serialized `BhCallDescr.extra_info` is an inert `MOST_GENERAL`
/// placeholder (`prepass.rs` after `translated_effect_info_id` is
/// assigned). The real EffectInfo lives in the interned table — the
/// same join `descr_ref_at` / `rehydrated_call_descr_ref` already
/// performs for the FBW walker. Restore it so `JitCode::descr_at` →
/// `as_calldescr().extra_info` (the interpret residual path) sees the
/// extraeffect the assembler used when deciding whether to emit `-live-`.
fn restore_translated_call_extra_info(bh: &mut BhDescr) {
    let calldescr = match bh {
        BhDescr::Call { calldescr } | BhDescr::JitCode { calldescr, .. } => calldescr,
        _ => return,
    };
    let Some(id) = calldescr.translated_effect_info_id else {
        return;
    };
    let cell = majit_ir::effectinfo::translated_effect_info(id).unwrap_or_else(|| {
        panic!("translated EffectInfo {id} was not published before its CallDescr")
    });
    calldescr.extra_info = cell.get().clone();
}

fn load_runtime_descr(index: usize) -> majit_metainterp::RuntimeBhDescr {
    use majit_metainterp::RuntimeBhDescr;
    // Same ordering as `descr_ref_at`: freeze EffectInfo identities
    // before a CallDescr can ask for one.
    rehydrate_build_descr_raw_sets();
    let mut bh = load_descr_uncached(index);
    restore_translated_call_extra_info(&mut bh);
    match bh {
        BhDescr::JitCode { jitcode_index, .. } => match get_runtime_jitcode_by_index(jitcode_index)
        {
            Some(runtime) => RuntimeBhDescr::JitCode(runtime),
            None => RuntimeBhDescr::Descr(Box::new(bh)),
        },
        other => RuntimeBhDescr::Descr(Box::new(other)),
    }
}

struct LazyRuntimeDescrTable;

impl majit_metainterp::RuntimeDescrTable for LazyRuntimeDescrTable {
    fn get(&self, index: usize) -> Option<&'static majit_metainterp::RuntimeBhDescr> {
        let cell = runtime_descr_cells().get(index)?;
        Some(cell.get_or_init(|| load_runtime_descr(index)))
    }

    fn len(&self) -> usize {
        descr_count()
    }

    fn jitcode_at(
        &self,
        index: usize,
    ) -> Option<std::sync::Arc<majit_metainterp::jitcode::JitCode>> {
        crate::state::ensure_build_time_jitcode_at(index)
            .map(|payload| std::sync::Arc::clone(&payload.jitcode))
    }
}

static LAZY_RUNTIME_DESCR_TABLE: LazyRuntimeDescrTable = LazyRuntimeDescrTable;

pub fn install_global_build_descr_pool() {
    majit_metainterp::init_global_build_descr_pool(&LAZY_RUNTIME_DESCR_TABLE);
}

/// Build a `BlackholeInterpBuilder` pre-configured for this binary's
/// jitcodes, paired with the list of `insns` opnames that
/// `wire_bhimpl_handlers` did not assign a handler to.
///
/// RPython parity: `BlackholeInterpBuilder.__init__` (blackhole.py)
/// runs `setup_insns(asm.insns)` + `setup_descrs(asm.descrs)` and
/// `setup_insns` (blackhole.py) resolves each opname via
/// `_get_method` eagerly, raising `AttributeError` if any `bhimpl_*`
/// is missing. The Rust port mirrors the `setup_insns` + `setup_descrs`
/// + `wire_bhimpl_handlers` sequence, but surfaces the unwired list as
/// a return value instead of panicking.
///
/// TODO(diagnostic). RPython has no equivalent
/// of this Result-returning shape because upstream's `setup_insns`
/// has total opname coverage at startup. The strict variant
/// [`build_default_bh_builder`] is the production path; this helper is
/// retained for tests that need to inspect the coverage set directly.
pub fn build_default_bh_builder_with_unwired_report() -> (
    majit_metainterp::blackhole::BlackholeInterpBuilder,
    Vec<String>,
) {
    let mut builder = majit_metainterp::blackhole::BlackholeInterpBuilder::new();
    // blackhole.py order: setup_insns, then setup_descrs.
    builder.setup_insns(insns_opname_to_byte());
    builder.setup_descrs(descr_table());
    majit_metainterp::blackhole::wire_bhimpl_handlers(&mut builder);
    let unwired: Vec<String> = builder
        .unwired_opnames()
        .into_iter()
        .map(|s| s.to_string())
        .collect();
    (builder, unwired)
}

/// Production-side strict blackhole builder for pyre's guard-failure
/// resume path.  It delegates to `build_inline_call_only_bh_builder`,
/// which installs the audited pyre dispatch surface: byte-identical
/// canonical keys, pyre-u16 register-width adapters, residual_call/vable
/// families, state-field adapters, and the pyre nested inline-call
/// handler.
///
/// Why not `build_default_bh_builder()`: pyre runtime bytecode still
/// contains helper-side layouts that are not canonical RPython argcodes.
/// This builder therefore registers only shapes with an explicit handler
/// contract.  Any emitted byte outside that setup surface now reaches
/// `dispatch_step`'s unwired-opcode panic; there is no legacy fallback.
/// Leftover `cond_call_*_ext` / `record_known_result_*_ext` bytes stay
/// unwired; canonical `conditional_call_*` / `record_known_result_*`
/// keys are registered by `build_inline_call_only_bh_builder`.
pub fn build_pyre_production_bh_builder() -> majit_metainterp::blackhole::BlackholeInterpBuilder {
    install_py_container_ctors();
    // `setup_insns(asm.insns)`: dynamically numbered recursive_call_* keys
    // take the byte this build's assembler gave them.
    // `handle_recursive_call` emits `recursive_call_{i,r,f,v}/iIRFIRF{>X,}`
    // (`blackhole.py bhimpl_recursive_call_{i,r,f,v}`); only the variants
    // this build actually emitted need a slot. `wire_bhimpl_handlers`
    // already binds each key; without the slot, `setup_insns` never
    // records the byte and a guard-failure resume panics here.
    // A backend that did not emit `getarrayitem_raw_i` (wasm jitcodes)
    // omits that key. `setarrayitem_raw_i/iiid` is the store twin
    // (`blackhole.py bhimpl_setarrayitem_raw_i`); looking inside
    // `copy_object_slice_range_into_vec` emits it when the dest is a
    // residual `ll_vec_alloc_and_set_r` Int header.
    const DYNAMIC_INSN_KEYS: &[&str] = &[
        "recursive_call_i/iIRFIRF>i",
        "recursive_call_r/iIRFIRF>r",
        "recursive_call_f/iIRFIRF>f",
        "recursive_call_v/iIRFIRF",
        "getarrayitem_raw_i/iid>i",
        "setarrayitem_raw_i/iiid",
        "int_isvirtual/i>i",
    ];
    let dynamic: Vec<(&str, u8)> = DYNAMIC_INSN_KEYS
        .iter()
        .filter_map(|&key| build_emitted_insns().get(key).map(|&byte| (key, byte)))
        .collect();
    let mut builder = majit_metainterp::blackhole::build_inline_call_only_bh_builder(&dynamic);
    // Legacy `NewList` / `NewTuple` that the rtyper did not rewrite still
    // reach the assembler as `newlist/>r` and `newtuple/rr>r`. Their bytes
    // are assigned by `get_opnum` while emitting jitcodes, so the slot is
    // whatever this build recorded.
    for key in ["newlist/>r", "newtuple/rr>r"] {
        let Some(&byte) = build_emitted_insns().get(key) else {
            continue;
        };
        let slot = &mut builder._insns[byte as usize];
        if slot.is_empty() {
            *slot = key.to_string();
        }
        let handler: majit_metainterp::blackhole::BhOpcodeHandler = match key {
            "newlist/>r" => handler_newlist_refs,
            "newtuple/rr>r" => handler_newtuple_refs,
            _ => unreachable!(),
        };
        assert!(
            builder.wire_handler(key, handler),
            "emitted {key} at byte {byte} did not land in the blackhole table"
        );
    }
    assert_production_builder_spans_the_emitted_universe(&builder);
    builder
}

fn install_py_container_ctors() {
    let _ = PY_LIST_CTOR.set(py_list_from_words);
    let _ = PY_TUPLE_CTOR.set(py_tuple_from_words);
}

fn py_list_from_words(words: &[i64]) -> i64 {
    let items = words_as_refs(words);
    pyre_interpreter::runtime_ops::build_list_from_refs(&items) as usize as i64
}

fn py_tuple_from_words(words: &[i64]) -> i64 {
    let items = words_as_refs(words);
    pyre_interpreter::runtime_ops::build_tuple_from_refs(&items) as usize as i64
}

fn words_as_refs(words: &[i64]) -> Vec<pyre_object::PyObjectRef> {
    words
        .iter()
        .map(|word| *word as usize as pyre_object::PyObjectRef)
        .collect()
}

static PY_LIST_CTOR: std::sync::OnceLock<fn(&[i64]) -> i64> = std::sync::OnceLock::new();
static PY_TUPLE_CTOR: std::sync::OnceLock<fn(&[i64]) -> i64> = std::sync::OnceLock::new();

fn handler_newlist_refs(
    bh: &mut majit_metainterp::blackhole::BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, majit_metainterp::blackhole::DispatchError> {
    // `newlist/>r`: no item registers, one result register.
    let dst = code[p] as usize;
    let ctor = PY_LIST_CTOR.get().expect("list constructor");
    bh.registers_r[dst] = ctor(&[]);
    Ok(p + 1)
}

fn handler_newtuple_refs(
    bh: &mut majit_metainterp::blackhole::BlackholeInterpreter,
    code: &[u8],
    p: usize,
) -> Result<usize, majit_metainterp::blackhole::DispatchError> {
    // `newtuple/rr>r`: two ref inputs, then the result register.
    let a = bh.registers_r[code[p] as usize];
    let b = bh.registers_r[code[p + 1] as usize];
    let dst = code[p + 2] as usize;
    let ctor = PY_TUPLE_CTOR.get().expect("tuple constructor");
    bh.registers_r[dst] = ctor(&[a, b]);
    Ok(p + 3)
}

/// The two coverage properties `blackhole.py setup_insns` holds by
/// construction, checked here because pyre's production builder cannot get
/// them that way.
///
/// Upstream calls `setup_insns(asm.insns)`, so its dispatch table spans exactly
/// the opnames the assembler emitted, each resolved through `_get_method`.
/// `build_inline_call_only_bh_builder` instead hand-curates its registered set
/// family by family and spells each byte as a `BC_*` constant, which admits two
/// drifts upstream cannot have:
///
/// 1. an emitted opname it never registered — reached only when a forward
///    resume happens to land on that byte, as `dispatch_step`'s unwired-opcode
///    panic;
/// 2. a key bound to a byte the canonical table gives to a *different* opname,
///    which decodes every occurrence as the wrong instruction.  `setup_insns`
///    rejects two keys claiming one byte within this builder and nothing
///    checks the byte against the generated table, so (2) is silent.
///
/// Both were pinned by tests alone — `production_bh_builder_covers_every_build_emitted_opname`
/// and `production_bh_builder_bytes_match_canonical_table` —
/// which read the generated tables of whatever tree ran them; the failure they
/// describe belongs to the binary that dispatches, so it is raised where that
/// binary assembles its builder.  Both production call sites build once per
/// thread behind a `thread_local!`, so this walks a few hundred keys per
/// thread, not per resume.
///
/// The universe is `build_emitted_insns()` — pyre's `asm.insns`, the serialised
/// build-time `pipeline.insns` before the canonical overlay — and deliberately
/// not `insns_opname_to_byte()`, which also carries `wellknown_bh_insns` /
/// `extension_insns` keys this build never emitted.  Their bytes cannot appear
/// in any jitcode here, so leaving them unregistered is the state upstream is
/// in too.
fn assert_production_builder_spans_the_emitted_universe(
    builder: &majit_metainterp::blackhole::BlackholeInterpBuilder,
) {
    let mut unregistered: Vec<&str> = build_emitted_insns()
        .iter()
        .filter(|(_key, byte)| {
            builder
                ._insns
                .get(**byte as usize)
                .is_none_or(|name| name.is_empty())
        })
        .map(|(key, _)| key.as_str())
        .collect();
    unregistered.sort_unstable();
    assert!(
        unregistered.is_empty(),
        "the production blackhole cannot dispatch {} opname(s) this build's \
         jitcodes can carry: {unregistered:?} — register them in \
         `build_inline_call_only_bh_builder`, confirming the emit-side operand \
         shape matches the wired handler's decoder",
        unregistered.len(),
    );

    let mut mismatched: Vec<String> = Vec::new();
    for (byte, key) in builder._insns.iter().enumerate() {
        if key.is_empty() {
            continue;
        }
        match insns_opname_to_byte().get(key) {
            Some(&canonical) if canonical as usize == byte => {}
            Some(&canonical) => mismatched.push(format!(
                "{key:?} registered at byte {byte}, canonical byte is {canonical}"
            )),
            None => mismatched.push(format!(
                "{key:?} registered at byte {byte} but is not a canonical opname"
            )),
        }
    }
    mismatched.sort();
    assert!(
        mismatched.is_empty(),
        "the production blackhole's hand-spelled bytes disagree with the \
         canonical opname table, so each listed byte decodes as the wrong \
         instruction: {mismatched:?}",
    );
}

/// Strict-coverage variant of [`build_default_bh_builder_with_unwired_report`]
/// — panics when any `insns` opname lacks a `bhimpl_*` handler.
///
/// RPython parity: matches the `AttributeError` raised by upstream's
/// `setup_insns` (blackhole.py) when `_get_method(name, argcodes)`
/// fails. Use this in any code path that must run real production
/// jitcodes; tests that inspect the raw coverage list can use
/// [`build_default_bh_builder_with_unwired_report`].
pub fn build_default_bh_builder() -> majit_metainterp::blackhole::BlackholeInterpBuilder {
    let (builder, unwired) = build_default_bh_builder_with_unwired_report();
    if !unwired.is_empty() {
        panic!(
            "build_default_bh_builder: {} insns opnames have no bhimpl_* \
             handler (RPython blackhole.py:66 raises AttributeError here): \
             {:?}",
            unwired.len(),
            unwired,
        );
    }
    builder
}

/// Decoded one jitcode instruction. Mirrors the static slice that RPython
/// `BlackholeInterpBuilder._get_method` would walk over, without any
/// execution of `bhimpl_*`. Lifetime is tied to the `insns` table, so the
/// opname stays valid while the runtime is alive (`'static`).
///
/// RPython parity: `blackhole.py` `_get_method.handler` consumes
/// operand bytes per `argcodes` char; this struct captures the same byte
/// layout without executing.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DecodedOp {
    /// The `opname/argcodes` key from the `insns` table.
    pub key: &'static str,
    /// `opname` part of `key` (before the `/`).
    pub opname: &'static str,
    /// `argcodes` part of `key` (after the `/`). Empty for `live/`.
    pub argcodes: &'static str,
    /// Position of the opcode byte in the jitcode.
    pub pc: usize,
    /// Position of the first byte *after* this instruction. `None` if the
    /// instruction reads a variable-length operand (`I`/`R`/`F`) that
    /// overflowed the code slice.
    pub next_pc: usize,
}

/// Decode the byte length of a `/P` adapter payload starting at
/// `cursor`.  Each `_pyre/P` opname has its own pyre helper-side flat
/// payload shape; the `P` pseudo-argcode is just an opt-out from the
/// canonical RPython argcode alphabet.  Producers and shapes:
///
/// | opname                              | producer (`assembler.rs`)             | payload bytes |
/// |-------------------------------------|---------------------------------------|---------------|
/// | `inline_call_nested_ext`           | `:nested_inline_call_*_typed_args`    | `4 + num_args*3 + 3`  (`sub_idx u16 + num_args u16 + num_args × (kind u8, caller_src u8, callee_dst u8) + return_i u8 + return_r u8 + return_f u8`) |
/// | `call_assembler_int_ext`           | `:3429 call_assembler_int_like`       | `5 + num_args*2`  (`target_idx u16 + dst u8 + num_args u16 + num_args × (kind u8, reg u8)`) |
/// | `call_assembler_ref_ext`           | `:3451 call_assembler_ref_like`       | same as int |
/// | `call_assembler_float_ext`         | `:3489 call_assembler_float_like`     | same as int |
/// | `call_assembler_void_ext`          | `:3370 call_assembler_void_like`      | `4 + num_args*2`  (omits `dst u8`) |
/// | `cond_call_void_ext`               | `:2642 call_cond_like`                | `4 + arg_count*2`  (`first_reg u8 + fn_ptr_idx u16 + arg_count u8 + arg_count × (kind u8) + arg_count × (reg u8)`) |
/// | `record_known_result_int_ext`      | `:2618 call_cond_like`                | same as cond_call |
/// | `record_known_result_ref_ext`      | `:2630 call_cond_like`                | same as cond_call |
/// | `cond_call_value_int_ext`          | `:2660 call_cond_value_like`          | `5 + arg_count*2`  (cond_call layout + trailing `dst u8`) |
/// | `cond_call_value_ref_ext`          | `:2660 call_cond_value_like`          | same as int variant |
fn pyre_p_payload_len(opname: &str, code: &[u8], cursor: usize) -> Option<usize> {
    match opname {
        "inline_call_nested_ext" => {
            // sub_idx u16 + num_args u16 + num_args × (kind u8, caller_src
            // u8, callee_dst u8) + return_i u8 + return_r u8 + return_f u8
            let num_args =
                u16::from_le_bytes([*code.get(cursor + 2)?, *code.get(cursor + 3)?]) as usize;
            Some(4 + num_args * 3 + 3)
        }
        "call_assembler_int_ext" | "call_assembler_ref_ext" | "call_assembler_float_ext" => {
            // target_idx u16 + dst u8 + num_args u16 + num_args × (kind u8, reg u8)
            let num_args =
                u16::from_le_bytes([*code.get(cursor + 3)?, *code.get(cursor + 4)?]) as usize;
            Some(5 + num_args * 2)
        }
        "call_assembler_void_ext" => {
            // target_idx u16 + num_args u16 + num_args × (kind u8, reg u8)
            let num_args =
                u16::from_le_bytes([*code.get(cursor + 2)?, *code.get(cursor + 3)?]) as usize;
            Some(4 + num_args * 2)
        }
        "cond_call_void_ext" | "record_known_result_int_ext" | "record_known_result_ref_ext" => {
            // first_reg u8 + fn_ptr_idx u16 + arg_count u8 + arg_count × (kind u8) + arg_count × (reg u8)
            let arg_count = *code.get(cursor + 3)? as usize;
            Some(4 + arg_count * 2)
        }
        "cond_call_value_int_ext" | "cond_call_value_ref_ext" => {
            // cond_call shape + trailing dst u8
            let arg_count = *code.get(cursor + 3)? as usize;
            Some(5 + arg_count * 2)
        }
        _ => None,
    }
}

/// Statically walk one jitcode instruction starting at `pc`. Returns
/// `None` when `pc` is beyond `code.len()` or when the opcode byte at
/// `pc` is unknown to the `insns` table.
///
/// The `argcodes` char-byte mapping follows `blackhole.py:112-157`:
///
/// ```text
///   'i'|'c'|'r'|'f'  -> 1 byte (register index or signed const byte)
///   'L'              -> 2 bytes (unsigned label)
///   'd'|'j'          -> 2 bytes (descr index)
///   'I'|'R'|'F'      -> 1 + N bytes (N = first byte = list length)
///   '>' + ('i'|'r'|'f') -> 1 byte (result destination register)
/// ```
///
/// `live/` is special-cased to advance by `liveness::OFFSET_SIZE` per
/// `blackhole.py` (`bhimpl_live(pc): return pc + OFFSET_SIZE`).
pub fn decode_op_at(code: &[u8], pc: usize) -> Option<DecodedOp> {
    let opcode_byte = *code.get(pc)?;
    let insn = insn_for_opcode_byte(opcode_byte)?;
    let key = insn.key;
    let opname = insn.opname;
    let argcodes = insn.argcodes;

    let mut cursor = pc + 1;
    if opname == "live" {
        // blackhole.py bhimpl_live(pc): position += OFFSET_SIZE.
        // The `live/` key has empty argcodes so the generic walker would
        // advance 0 bytes, but dispatch skips 2 bytes of liveness offset.
        cursor += majit_jitcode::liveness::OFFSET_SIZE;
        if cursor > code.len() {
            return None;
        }
        return Some(DecodedOp {
            key,
            opname,
            argcodes,
            pc,
            next_pc: cursor,
        });
    }

    let mut chars = argcodes.chars();
    while let Some(c) = chars.next() {
        match c {
            'i' | 'c' | 'r' | 'f' => cursor += 1,
            'L' | 'd' | 'j' => cursor += 2,
            'I' | 'R' | 'F' => {
                // blackhole.py:139-143: varlist opens with a 1-byte length,
                // followed by that many 1-byte register indices.
                let list_len = *code.get(cursor)? as usize;
                cursor += 1 + list_len;
            }
            '>' => {
                // blackhole.py:185-209: result destination is 1 byte,
                // following `>i`, `>r`, or `>f`.
                let rt = chars.next()?;
                if !matches!(rt, 'i' | 'r' | 'f') {
                    return None;
                }
                cursor += 1;
            }
            // TODO: pseudo-argcode for pyre-only
            // helper-side opnames.  Each `*_pyre/P` opname carries its
            // own flat payload shape; see `pyre_p_payload_len` for the
            // table.
            'P' => {
                cursor += pyre_p_payload_len(opname, code, cursor)?;
            }
            _ => return None,
        }
    }

    if cursor > code.len() {
        return None;
    }
    Some(DecodedOp {
        key,
        opname,
        argcodes,
        pc,
        next_pc: cursor,
    })
}

/// Iterator over every instruction in a jitcode `code` slice. Yields
/// `DecodedOp` in linear order (fallthrough layout — branch targets in
/// `L`-typed operands are not followed). Stops on the first decode
/// failure, which surfaces either end-of-code or an insns-table miss.
pub fn decoded_ops(code: &[u8]) -> impl Iterator<Item = DecodedOp> + '_ {
    let mut pc = 0;
    std::iter::from_fn(move || {
        let op = decode_op_at(code, pc)?;
        pc = op.next_pc;
        Some(op)
    })
}

fn split_key(key: &str) -> (&str, &str) {
    match key.split_once('/') {
        Some((name, codes)) => (name, codes),
        None => (key, ""),
    }
}

/// Where a resolved operand came from and the value read at that slot.
///
/// RPython `blackhole.py:112-157` argcodes consume a register index byte
/// or a small-constant byte and produce the value the `bhimpl_*` method
/// receives. This enum captures both the source byte(s) and the resolved
/// value, so diagnostics and shadow-execution paths can surface either
/// without re-walking the code.
#[derive(Debug, Clone, PartialEq)]
pub enum ResolvedOperand {
    /// `i`: `self.registers_i[code[pc]]` (blackhole.py:120).
    IntReg { reg: u8, value: i64 },
    /// `c`: signed byte treated as a small constant (blackhole.py:121-123,
    /// `signedord`).
    ConstByte { byte: i8 },
    /// `r`: `self.registers_r[code[pc]]` (blackhole.py:124-126).
    RefReg { reg: u8, value: i64 },
    /// `f`: `self.registers_f[code[pc]]` (blackhole.py:127-129).
    FloatReg { reg: u8, value: i64 },
    /// `L`: 2-byte little-endian unsigned label (blackhole.py:133-138).
    Label { target: u16 },
    /// `d`|`j`: 2-byte little-endian descr index into
    /// `metainterp_sd.descrs` (blackhole.py:150-157). `j` carries the
    /// additional RPython assertion `isinstance(value, JitCode)`; the
    /// resolver here records the index only.
    DescrIdx { index: u16, is_jitcode: bool },
    /// `I`: `[registers_i[idx] for idx in list]` (blackhole.py:139-143 via
    /// `_get_list_of_values`).
    IntList(Vec<(u8, i64)>),
    /// `R`: ref-list variant of the above.
    RefList(Vec<(u8, i64)>),
    /// `F`: float-list variant of the above.
    FloatList(Vec<(u8, i64)>),
}

/// Where a `bhimpl_*` result would be written back.
///
/// RPython `blackhole.py:185-223` handles `>i`, `>r`, `>f` result slots
/// (and the `iL` split for `goto_if_*` which the resolver treats as
/// `Int` here — the shadow layer can interpret the `L`-branch later).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedResult {
    Int { reg: u8 },
    Ref { reg: u8 },
    Float { reg: u8 },
}

/// One `DecodedOp` enriched with resolved operand values and the result
/// destination (if any). Construction is pure — walking the byte stream
/// with a read-only register file.
///
/// RPython parity: `_get_method.handler` up to the `unboundmethod(*args)`
/// call. The bhimpl dispatch itself is intentionally left out; this
/// struct is the data the shadow-record layer or a diff-only
/// analyzer can inspect without executing any side effect.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedOp {
    pub decoded: DecodedOp,
    pub operands: Vec<ResolvedOperand>,
    pub result: Option<ResolvedResult>,
}

/// Read-only view over the three register files a `bhimpl_*` would
/// consume. Empty slots read as 0 (RPython has no corresponding case —
/// in-range reads are a translation-time invariant). The resolver here
/// returns `None` for out-of-range indices so the caller can treat an
/// oversized jitcode as a decode failure rather than silently misread.
#[derive(Debug, Clone, Copy)]
pub struct RegisterFileView<'a> {
    pub registers_i: &'a [i64],
    pub registers_r: &'a [i64],
    pub registers_f: &'a [i64],
}

impl<'a> RegisterFileView<'a> {
    pub fn empty() -> Self {
        Self {
            registers_i: &[],
            registers_r: &[],
            registers_f: &[],
        }
    }
}

/// Port of `_get_method.handler` operand-resolution phase (everything up
/// to the `unboundmethod(*args)` call at blackhole.py).
///
/// Decodes the op at `pc`, then walks `argcodes` once more — this time
/// reading each operand byte *and* resolving it via the register files.
/// Returns `None` on any of:
///   - `pc` beyond `code.len()` or unknown opcode byte (same as
///     `decode_op_at`).
///   - register index out of range for the active file.
///   - unrecognized argcode char.
///
/// The function is intentionally stateless; the caller supplies a
/// `RegisterFileView` borrowing from whatever concrete register storage
/// the shadow layer uses.
pub fn resolve_op_at(code: &[u8], pc: usize, regs: RegisterFileView<'_>) -> Option<ResolvedOp> {
    let decoded = decode_op_at(code, pc)?;
    if decoded.opname == "live" {
        // bhimpl_live consumes OFFSET_SIZE operand bytes but produces no
        // resolved operands — decode_op_at already advanced past them.
        return Some(ResolvedOp {
            decoded,
            operands: Vec::new(),
            result: None,
        });
    }

    let mut cursor = pc + 1;
    let mut operands: Vec<ResolvedOperand> = Vec::new();
    let mut result: Option<ResolvedResult> = None;
    let mut chars = decoded.argcodes.chars();
    while let Some(c) = chars.next() {
        match c {
            'i' => {
                let reg = *code.get(cursor)?;
                cursor += 1;
                let value = *regs.registers_i.get(reg as usize)?;
                operands.push(ResolvedOperand::IntReg { reg, value });
            }
            'c' => {
                // blackhole.py `signedord`: signed byte constant.
                let byte = *code.get(cursor)? as i8;
                cursor += 1;
                operands.push(ResolvedOperand::ConstByte { byte });
            }
            'r' => {
                let reg = *code.get(cursor)?;
                cursor += 1;
                let value = *regs.registers_r.get(reg as usize)?;
                operands.push(ResolvedOperand::RefReg { reg, value });
            }
            'f' => {
                let reg = *code.get(cursor)?;
                cursor += 1;
                let value = *regs.registers_f.get(reg as usize)?;
                operands.push(ResolvedOperand::FloatReg { reg, value });
            }
            'L' => {
                let lo = *code.get(cursor)? as u16;
                let hi = *code.get(cursor + 1)? as u16;
                cursor += 2;
                operands.push(ResolvedOperand::Label {
                    target: lo | (hi << 8),
                });
            }
            'd' | 'j' => {
                let lo = *code.get(cursor)? as u16;
                let hi = *code.get(cursor + 1)? as u16;
                cursor += 2;
                operands.push(ResolvedOperand::DescrIdx {
                    index: lo | (hi << 8),
                    is_jitcode: c == 'j',
                });
            }
            'I' | 'R' | 'F' => {
                let list_len = *code.get(cursor)? as usize;
                cursor += 1;
                let mut entries = Vec::with_capacity(list_len);
                let file: &[i64] = match c {
                    'I' => regs.registers_i,
                    'R' => regs.registers_r,
                    'F' => regs.registers_f,
                    _ => unreachable!(),
                };
                for _ in 0..list_len {
                    let reg = *code.get(cursor)?;
                    cursor += 1;
                    let value = *file.get(reg as usize)?;
                    entries.push((reg, value));
                }
                operands.push(match c {
                    'I' => ResolvedOperand::IntList(entries),
                    'R' => ResolvedOperand::RefList(entries),
                    'F' => ResolvedOperand::FloatList(entries),
                    _ => unreachable!(),
                });
            }
            '>' => {
                let rt = chars.next()?;
                let reg = *code.get(cursor)?;
                cursor += 1;
                result = Some(match rt {
                    'i' => ResolvedResult::Int { reg },
                    'r' => ResolvedResult::Ref { reg },
                    'f' => ResolvedResult::Float { reg },
                    _ => return None,
                });
            }
            // TODO: pyre `*_pyre/P` payloads are
            // opaque to the canonical operand-resolver — no matching
            // `ResolvedOperand` variant exists.  Advance the cursor by
            // the per-opname computed length so the trailing
            // `cursor == decoded.next_pc` invariant holds; consumers
            // dispatch on `decoded.opname` for payload-aware handling.
            'P' => {
                cursor += pyre_p_payload_len(decoded.opname, code, cursor)?;
            }
            _ => return None,
        }
    }

    // Sanity: the walker here must land on the same next_pc decode_op_at
    // computed. If not, our argcodes handling disagrees with decode_op_at
    // and something silently miscounted operand bytes.
    debug_assert_eq!(
        cursor, decoded.next_pc,
        "resolve_op_at cursor {cursor} != decode_op_at next_pc {} for key {}",
        decoded.next_pc, decoded.key,
    );
    Some(ResolvedOp {
        decoded,
        operands,
        result,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `descr.py` `get_size_descr` returns a STRUCT that already lists fields.
    /// The parentless size slot must not bincode that list again, even when
    /// the published vtable word is 0. An empty shell still decodes. A declared
    /// STRUCT's field stays the group's descr when kind-0 does not re-decode
    /// the slot.
    #[test]
    fn kind0_size_and_materialize() {
        {
            // kind0_size_with_fields_skips_the_parentless_slot
            let published = 0xF1E1_E201;
            assert!(!crate::descr::kind0_size_already_published(published));
            majit_ir::descr::publish_borrowed_struct_layout(
                u32::MAX,
                24,
                1,
                published,
                0,
                true,
                false,
                &[],
                vec![majit_ir::descr::BorrowedField {
                    index: 0,
                    name: std::borrow::Cow::Borrowed("Parentless.slot"),
                    field_key: "slot",
                    offset: 16,
                    field_size: 8,
                    field_type: majit_ir::value::Type::Int,
                    flag: majit_ir::descr::ArrayFlag::Signed,
                    is_immutable: false,
                    is_quasi_immutable: false,
                    index_in_parent: 0,
                    is_class_word: Some(false),
                }],
            );
            assert!(crate::descr::kind0_size_already_published(published));
            assert!(
                crate::descr::kind0_published_size_ids()
                    .binary_search(&published)
                    .is_ok()
            );
            let shell = 0xF1E1_E202;
            majit_ir::descr::publish_borrowed_struct_layout(
                u32::MAX,
                16,
                1,
                shell,
                0,
                true,
                false,
                &[],
                Vec::new(),
            );
            assert!(!crate::descr::kind0_size_already_published(shell));
            assert!(
                crate::descr::kind0_published_size_ids()
                    .binary_search(&shell)
                    .is_err()
            );
            assert!(!crate::descr::kind0_size_already_published(0));
        }
        {
            // kind0_materialize_publishes_declared_int_field
            materialize_gccache_owned_descrs();
            let key = majit_ir::descr::LLType::Struct(majit_ir::descr::path_hash(
                "intobject::W_IntObject",
            ));
            let cached = majit_ir::descr::gc_cache()
                .lock()
                ._cache_field
                .get(&key)
                .and_then(|fields| fields.get("intval"))
                .cloned()
                .expect("W_IntObject.intval published before user code");
            let cached = cached as majit_ir::DescrRef;
            assert!(std::sync::Arc::ptr_eq(
                &cached,
                &crate::descr::int_intval_descr()
            ));
        }
    }

    /// `descr.py` `get_size_descr` reads the STRUCT row. A nonzero `type_id`
    /// slot is that packed row, not a bincode `BhDescr` enum.
    #[test]
    fn packed_size_slot_and_parent_layouts() {
        {
            // packed_size_slot_decodes_without_bincode
            use majit_jitcode::jitcode::{BhFieldSpec, BhSizeSpec};
            let mut spec = BhSizeSpec {
                size: 24,
                type_id: 0xA11C_E501,
                vtable: 0,
                owner: String::new(),
                is_gc_managed: true,
                headerless: false,
                all_fielddescrs: vec![BhFieldSpec {
                    index: 0,
                    field_key: "slot".to_string(),
                    name: "Packed.slot".to_string(),
                    offset: 16,
                    field_size: 8,
                    field_type: majit_ir::value::Type::Int,
                    field_flag: majit_ir::descr::ArrayFlag::Signed,
                    is_field_signed: true,
                    is_immutable: false,
                    is_quasi_immutable: false,
                    index_in_parent: 0,
                    is_class_word: Some(false),
                }],
            };
            let mut bytes = Vec::new();
            spec.pack_into(&mut bytes);
            let BhDescr::Size {
                type_id,
                all_fielddescrs,
                owner,
                ..
            } = descr_from_indexed_bytes(&bytes, spec.type_id)
            else {
                panic!("packed size slot must be a Size");
            };
            assert_eq!(type_id, spec.type_id);
            assert!(owner.is_empty());
            assert_eq!(all_fielddescrs.len(), 1);
            assert_eq!(all_fielddescrs[0].field_key(), "slot");

            spec.owner = "INT_TYPE".into();
            spec.vtable = 0x7fff_ff00;
            let mut named = Vec::new();
            spec.pack_into(&mut named);
            let BhDescr::Size {
                owner: named_owner,
                vtable: named_vtable,
                ..
            } = descr_from_indexed_bytes(&named, spec.type_id)
            else {
                panic!("packed size slot must keep owner");
            };
            assert_eq!(named_owner, "INT_TYPE");
            assert_eq!(named_vtable, 0x7fff_ff00);

            let raw = bincode::serialize(&BhDescr::VableField { index: 3 }).unwrap();
            assert!(matches!(
                descr_from_indexed_bytes(&raw, 0),
                BhDescr::VableField { index: 3 }
            ));
        }
        {
            // packed_parent_layouts_match_descr_layout_at
            let mut count = 0usize;
            for_each_packed_parent_layout(|spec| {
                assert_eq!(*descr_layout_at(count), spec);
                count += 1;
            });
            assert_eq!(count, descr_layout_offsets().len() - 1);
            assert!(count > 0, "translation stored no parent layout");
        }
    }

    #[test]
    fn descr_index_kind_matches_each_serialized_entry() {
        let index = descrs_index();
        assert_eq!(index.kinds.len(), descr_count());
        for (i, kind) in index.kinds.iter().copied().enumerate() {
            let expected = match load_descr_uncached(i) {
                BhDescr::Call { .. } | BhDescr::JitCode { .. } => 1,
                BhDescr::Field { .. }
                | BhDescr::Array { .. }
                | BhDescr::InteriorField { .. }
                | BhDescr::Size { .. } => 0,
                BhDescr::Switch { .. }
                | BhDescr::VableField { .. }
                | BhDescr::VableArray { .. }
                | BhDescr::VtableMethod { .. } => 2,
            };
            assert_eq!(kind, expected, "descriptor index {i}");
        }
    }

    #[test]
    fn frozen_field_parents_decode_the_canonical_layout_value() {
        let index = descrs_index();
        let layout_count = descr_layout_offsets().len() - 1;
        assert_eq!(index.parent_layouts.len(), descr_count());

        let mut first_parent_by_layout: std::collections::BTreeMap<
            u32,
            std::sync::Arc<majit_jitcode::jitcode::BhSizeSpec>,
        > = std::collections::BTreeMap::new();
        let mut parent_references = 0usize;
        for (slot, &layout_index) in index.parent_layouts.iter().enumerate() {
            let descr = load_descr_uncached(slot);
            if layout_index == u32::MAX {
                if let BhDescr::Field { parent, .. } = descr {
                    assert!(parent.is_none(), "parentless Field slot {slot}");
                }
                continue;
            }

            assert!((layout_index as usize) < layout_count);
            let BhDescr::Field {
                parent: Some(parent),
                ..
            } = descr
            else {
                panic!("layout-bearing descriptor slot {slot} is not a parented Field");
            };
            parent_references += 1;
            if let Some(first) = first_parent_by_layout.get(&layout_index) {
                assert_eq!(
                    first.as_ref(),
                    parent.as_ref(),
                    "layout {layout_index} decoded to different transport values"
                );
            } else {
                first_parent_by_layout.insert(layout_index, parent);
            }
        }

        assert_eq!(first_parent_by_layout.len(), layout_count);
        assert!(
            parent_references > layout_count,
            "fixture must exercise shared parent layouts"
        );
    }

    #[test]
    fn effect_info_mint_index_bounds_each_serialized_entry() {
        let offsets = ei_descr_mint_offsets();
        assert_eq!(offsets.len(), ei_descr_mint_count() + 1);
        for i in 0..ei_descr_mint_count() {
            let entry = load_ei_descr_mint(i);
            assert_ne!(
                entry.ei_index,
                u32::MAX,
                "mint entry {i} is absent from every frozen raw set"
            );
            assert!(offsets[i] < offsets[i + 1], "mint entry {i} is empty");
        }
    }

    #[test]
    fn effect_info_stamp_index_bounds_each_serialized_entry() {
        let offsets = ei_descr_stamp_offsets();
        assert_eq!(offsets.len(), ei_descr_stamp_count() + 1);
        for i in 0..ei_descr_stamp_count() {
            let (_member, ei_index) = load_ei_descr_stamp(i);
            assert_ne!(ei_index, u32::MAX, "stamp entry {i} has no partition");
            assert!(offsets[i] < offsets[i + 1], "stamp entry {i} is empty");
        }
    }

    #[test]
    fn rehydrate_effect_info_consumes_serialized_set_keys() {
        let mut effect_info = majit_ir::EffectInfo::default();
        assert!(effect_info.descr_set_keys.is_some());

        crate::descr::rehydrate_effect_info(&mut effect_info);

        assert!(effect_info.descr_set_keys.is_none());
        assert!(
            effect_info
                ._readonly_descrs_fields
                .as_ref()
                .is_some_and(Vec::is_empty)
        );
        assert!(
            effect_info
                ._write_descrs_fields
                .as_ref()
                .is_some_and(Vec::is_empty)
        );
        assert!(
            effect_info
                ._readonly_descrs_arrays
                .as_ref()
                .is_some_and(Vec::is_empty)
        );
        assert!(
            effect_info
                ._write_descrs_arrays
                .as_ref()
                .is_some_and(Vec::is_empty)
        );
        assert!(
            effect_info
                ._readonly_descrs_interiorfields
                .as_ref()
                .is_some_and(Vec::is_empty)
        );
        assert!(
            effect_info
                ._write_descrs_interiorfields
                .as_ref()
                .is_some_and(Vec::is_empty)
        );
    }

    /// The wire `BhCallDescr.extra_info` is `MOST_GENERAL`; interpret
    /// reads that field after `descr_at`. The loader must join the
    /// interned EffectInfo so extraeffect matches the assembler.
    #[test]
    fn load_runtime_descr_joins_translated_effect_info() {
        rehydrate_build_descr_raw_sets();
        let mut joined = 0usize;
        let mut restored_non_random = 0usize;
        for i in 0..descr_count() {
            let descr = load_runtime_descr(i);
            let Some(bh) = descr.as_bh_descr() else {
                continue;
            };
            let calldescr = match bh {
                BhDescr::Call { calldescr } | BhDescr::JitCode { calldescr, .. } => calldescr,
                _ => continue,
            };
            let Some(id) = calldescr.translated_effect_info_id else {
                continue;
            };
            let expected = majit_ir::effectinfo::translated_effect_info(id)
                .unwrap_or_else(|| panic!("missing interned EffectInfo {id} for descr {i}"));
            assert_eq!(
                calldescr.extra_info.extraeffect,
                expected.get().extraeffect,
                "descr {i} extraeffect did not join translated EffectInfo {id}",
            );
            joined += 1;
            if calldescr.extra_info.extraeffect != majit_ir::descr::ExtraEffect::RandomEffects {
                restored_non_random += 1;
            }
        }
        assert!(joined > 0, "fixture must contain translated CallDescrs");
        assert!(
            restored_non_random > 0,
            "fixture must restore at least one non-RandomEffects extraeffect; \
             otherwise interpret would still treat every residual as may_force"
        );
    }

    /// Splits the first-JIT descr cost into its stages, because the fix for
    /// each is different: deserialization is paid by the wire format,
    /// materialization by how much of the pool a run actually names.
    ///
    /// Run alone so nothing else in the process moves the number:
    /// `cargo test -p pyre-jit-trace --no-default-features --features dynasm \
    ///  descr_startup_rss_decomposition -- --exact --nocapture --test-threads=1`
    ///
    /// Ignored by default: it is a measurement, not an assertion, and the
    /// stages are process-global `Once`/`LazyLock` state that any other test
    /// in the binary can force first.
    #[test]
    #[ignore = "measurement; run alone with --exact --nocapture"]
    fn descr_startup_rss_decomposition() {
        // `ps` rather than `getrusage` because this crate has no libc
        // dependency and adding one would move its LLBC fingerprint. RSS only
        // grows across these stages, so the sample is the stage peak.
        fn rss_kb() -> i64 {
            let out = std::process::Command::new("ps")
                .args(["-o", "rss=", "-p"])
                .arg(std::process::id().to_string())
                .output()
                .expect("ps");
            String::from_utf8_lossy(&out.stdout)
                .trim()
                .parse()
                .unwrap_or(-1)
        }

        // Warm `Command`/`ps` bookkeeping before the baseline. On macOS the
        // first sample itself leaves about 1 MB of process-library state
        // resident, which otherwise gets charged to `deserialize_descrs`.
        let _ = rss_kb();
        // Likewise warm the exact index-deserializer monomorph. The real
        // `descrs_index.bin` is still decoded after the baseline; this keeps
        // cold code pages out of the retained-heap stage they would obscure.
        let _: Vec<u32> = bincode::deserialize(&[0_u8; 8]).unwrap();
        let base = rss_kb();
        let n_descrs = descr_count();
        let after_descrs = rss_kb();
        let n_mints = ei_descr_mint_count();
        let after_mints = rss_kb();
        rehydrate_build_descr_raw_sets();
        let after_rehydrate = rss_kb();
        let n_refs = descr_ref_table().len();
        let _ = descr_ref_cells();
        let after_refs = rss_kb();

        eprintln!("[rss-decomp] base={base} KB");
        eprintln!(
            "[rss-decomp] descr_index          n={n_descrs} delta={} KB",
            after_descrs - base
        );
        eprintln!(
            "[rss-decomp] ei_mint_index        n={n_mints} delta={} KB",
            after_mints - after_descrs
        );
        eprintln!(
            "[rss-decomp] rehydrate            delta={} KB",
            after_rehydrate - after_mints
        );
        eprintln!(
            "[rss-decomp] descr_ref_cells      n={n_refs} delta={} KB",
            after_refs - after_rehydrate
        );
        eprintln!("[rss-decomp] TOTAL={} KB", after_refs - base);

        // The pool census runs LAST and streams: it is the reason the stages
        // above are worth measuring, but a full walk allocates and frees every
        // entry, and sitting between two samples it charged that churn to
        // whichever stage followed it.
        {
            use majit_jitcode::jitcode::BhDescr;
            let mut variants: std::collections::BTreeMap<&str, usize> = Default::default();
            let mut specs = 0usize;
            let mut distinct_layouts: std::collections::HashSet<u64> = Default::default();
            let mut strings = 0usize;
            let mut string_bytes = 0usize;
            let note = |sp: &majit_jitcode::jitcode::BhSizeSpec,
                        specs: &mut usize,
                        strings: &mut usize,
                        string_bytes: &mut usize,
                        distinct: &mut std::collections::HashSet<u64>| {
                *specs += sp.all_fielddescrs.len();
                distinct.insert(sp.type_id);
                for f in &sp.all_fielddescrs {
                    *strings += 2;
                    *string_bytes += f.name.len() + f.field_key.len();
                }
            };
            for i in 0..n_descrs {
                let d = load_descr_uncached(i);
                let k = match &d {
                    BhDescr::Field { parent, .. } => {
                        if let Some(p) = parent {
                            note(
                                p,
                                &mut specs,
                                &mut strings,
                                &mut string_bytes,
                                &mut distinct_layouts,
                            );
                        }
                        "Field"
                    }
                    BhDescr::Array { .. } => "Array",
                    BhDescr::InteriorField { .. } => "InteriorField",
                    BhDescr::Size { .. } => "Size",
                    BhDescr::Call { .. } => "Call",
                    BhDescr::JitCode { .. } => "JitCode",
                    BhDescr::Switch { .. } => "Switch",
                    _ => "other",
                };
                *variants.entry(k).or_default() += 1;
            }
            eprintln!(
                "[rss-decomp] size_of::<BhDescr>()={} B  -> Vec body {} KB",
                std::mem::size_of::<BhDescr>(),
                std::mem::size_of::<BhDescr>() * n_descrs / 1024,
            );
            eprintln!("[rss-decomp] variants {variants:?}");
            eprintln!(
                "[rss-decomp] embedded parent field specs={specs} over {} distinct type_ids; \
                 strings={strings} ({string_bytes} B of text)",
                distinct_layouts.len(),
            );
        }
    }

    #[test]
    #[ignore = "measurement; run alone with --exact --nocapture"]
    fn descr_rehydrate_phase_rss_decomposition() {
        fn rss_kb() -> i64 {
            let out = std::process::Command::new("ps")
                .args(["-o", "rss=", "-p"])
                .arg(std::process::id().to_string())
                .output()
                .expect("ps");
            String::from_utf8_lossy(&out.stdout)
                .trim()
                .parse()
                .unwrap_or(-1)
        }

        let _ = rss_kb();
        let _: Vec<u32> = bincode::deserialize(&[0_u8; 8]).unwrap();
        let base = rss_kb();
        let heap_base = crate::allocation_counter::live_bytes();
        let index = descrs_index();
        let after_index = rss_kb();
        let heap_after_index = crate::allocation_counter::live_bytes();

        let layout_count = descr_layout_offsets().len() - 1;
        for i in 0..layout_count {
            drop(descr_layout_at(i));
        }
        let after_layouts = rss_kb();
        let heap_after_layouts = crate::allocation_counter::live_bytes();

        for (i, kind) in index.kinds.iter().copied().enumerate() {
            if kind == 0 {
                let bh = load_descr_uncached(i);
                crate::descr::make_descr_from_bh(&bh);
            }
        }
        let after_noncall = rss_kb();
        let heap_after_noncall = crate::allocation_counter::live_bytes();
        let (category_counts, cache_lens, cache_caps, field_inner_cap, field_key_bytes) = {
            let gc = majit_ir::descr::gc_cache().lock();
            (
                gc.category_counts(),
                (
                    gc._cache_size.len(),
                    gc._cache_field
                        .values()
                        .map(|inner| inner.len())
                        .sum::<usize>(),
                    gc._cache_array.len(),
                    gc._cache_arraylen.len(),
                    gc._cache_call.len(),
                    gc._cache_interiorfield.len(),
                ),
                (
                    gc._cache_size.capacity(),
                    gc._cache_field.capacity(),
                    gc._cache_array.capacity(),
                    gc._cache_arraylen.capacity(),
                    gc._cache_call.capacity(),
                    gc._cache_interiorfield.capacity(),
                ),
                gc._cache_field
                    .values()
                    .map(|inner| inner.capacity())
                    .sum::<usize>(),
                gc._cache_field
                    .values()
                    .flat_map(|inner| inner.keys())
                    .map(String::len)
                    .sum::<usize>(),
            )
        };

        for i in 0..ei_descr_stamp_count() {
            let (member, ei_index) = load_ei_descr_stamp(i);
            if let Some((struct_id, field_name)) = crate::descr::ambiguous_field_struct(&member) {
                publish_kind0_field(struct_id, field_name);
            }
            crate::descr::stamp_effect_info_descr(&member, ei_index);
        }
        let after_stamps = rss_kb();
        let heap_after_stamps = crate::allocation_counter::live_bytes();

        for i in 0..ei_descr_mint_count() {
            let entry = load_ei_descr_mint(i);
            crate::descr::publish_effect_info_descr_mints(std::slice::from_ref(&entry));
        }
        let after_mints = rss_kb();
        let heap_after_mints = crate::allocation_counter::live_bytes();

        for i in 0..effect_info_count() {
            let (translated_id, mut effect_info) = load_effect_info(i);
            crate::runtime_fnaddr_patch::rewrite_call_release_gil_target(&mut effect_info);
            crate::descr::prepare_frozen_effect_info(&mut effect_info);
            majit_ir::effectinfo::intern_translated_effect_info(translated_id, effect_info);
        }
        let after_effect_infos = rss_kb();
        let heap_after_effect_infos = crate::allocation_counter::live_bytes();
        let frozen_effect_info_count = effect_info_count();

        for (i, kind) in index.kinds.iter().copied().enumerate() {
            if kind == 1 {
                let bh = load_descr_uncached(i);
                let calldescr = match bh {
                    BhDescr::Call { calldescr } | BhDescr::JitCode { calldescr, .. } => calldescr,
                    _ => unreachable!(),
                };
                rehydrated_call_descr_ref(calldescr);
            }
        }
        let after_calls = rss_kb();
        let heap_after_calls = crate::allocation_counter::live_bytes();

        eprintln!("[rss-phases] base={base} KB");
        eprintln!(
            "[rss-phases] index          rss={} KB heap={} B",
            after_index - base,
            heap_after_index - heap_base,
        );
        eprintln!(
            "[rss-phases] wire_layouts   rss={} KB heap={} B",
            after_layouts - after_index,
            heap_after_layouts - heap_after_index,
        );
        eprintln!(
            "[rss-phases] noncall_descrs rss={} KB heap={} B",
            after_noncall - after_layouts,
            heap_after_noncall - heap_after_layouts,
        );
        eprintln!(
            "[rss-phases] gccache counts={category_counts:?} lens={cache_lens:?} \
             caps={cache_caps:?} \
             field_inner_cap={field_inner_cap} field_key_bytes={field_key_bytes} \
             sizeof(field,size,array)=({},{},{})",
            std::mem::size_of::<majit_ir::descr::SimpleFieldDescr>(),
            std::mem::size_of::<majit_ir::descr::SimpleSizeDescr>(),
            std::mem::size_of::<majit_ir::descr::SimpleArrayDescr>(),
        );
        eprintln!(
            "[rss-phases] compact_stamps rss={} KB heap={} B",
            after_stamps - after_noncall,
            heap_after_stamps - heap_after_noncall,
        );
        eprintln!(
            "[rss-phases] raw_set_mints  rss={} KB heap={} B",
            after_mints - after_stamps,
            heap_after_mints - heap_after_stamps,
        );
        eprintln!(
            "[rss-phases] effect_infos   rss={} KB heap={} B",
            after_effect_infos - after_mints,
            heap_after_effect_infos - heap_after_mints,
        );
        eprintln!(
            "[rss-phases] effect_info_count={frozen_effect_info_count} sizeof(EffectInfo)={} B",
            std::mem::size_of::<majit_ir::EffectInfo>(),
        );
        eprintln!(
            "[rss-phases] call_descrs    rss={} KB heap={} B",
            after_calls - after_effect_infos,
            heap_after_calls - heap_after_effect_infos,
        );
        eprintln!(
            "[rss-phases] TOTAL={} KB heap={} B",
            after_calls - base,
            heap_after_calls - heap_base,
        );
    }

    /// The build-time descr pool must name the same bytes `offset_of!` does.
    ///
    /// The pool is serialised by `build.rs` from `majit-translate`'s layout
    /// model, which sizes a pointer from `CARGO_CFG_TARGET_POINTER_WIDTH` and
    /// takes exact offsets from the Charon layout Charon resolved for the
    /// build's `TARGET`.  Both used to be fixed to the extraction host's
    /// 64-bit layout, so on a 32-bit target every offset past the first
    /// pointer field named the wrong bytes: the orthodox `w_list_append`
    /// sub-walk, the one production consumer that resolves its `d` operands
    /// through this pool, folded its subclass test on garbage, walked the
    /// dead `switch_to_object_strategy` leg, executed that leg's residuals
    /// against the live list, and aborted — dropping the in-flight `FOR_ITER`
    /// item with it (`bench/synth/delete_negative_open_slice_hot.py` then
    /// ran one loop iteration short).
    ///
    /// These two fields are exactly the ones that sub-walk reads, and both
    /// sit past a pointer, so they move between 64- and 32-bit targets.
    ///
    /// The owners are spelled with their crate-stripped module path, which is
    /// how the mint records them: a field descriptor's owner goes through
    /// `canonical_struct_name`, so two structs sharing a leaf but declared in
    /// different modules stay apart in the pool. Passing the bare leaf here
    /// matches nothing at all rather than matching loosely.
    #[test]
    fn build_time_offset_matches_runtime_layout() {
        for (owner, name, runtime) in [
            (
                "pyobject::PyObject",
                "w_class",
                std::mem::offset_of!(pyre_object::pyobject::PyObject, w_class),
            ),
            (
                "pyobject::PyType",
                "instantiate",
                std::mem::offset_of!(pyre_object::pyobject::PyType, instantiate),
            ),
        ] {
            let Some(built) = build_time_field_offset(owner, name) else {
                panic!("build-time descr pool carries no `{owner}.{name}` field slot");
            };
            assert_eq!(
                built, runtime,
                "build-time descr pool places `{owner}.{name}` at {built}, \
                 but this target's layout puts it at {runtime}",
            );
        }
    }

    /// Every key in build-observed `pipeline.insns` that ALSO appears in
    /// the canonical universe (`majit_jitcode::insns::
    /// {wellknown_bh_insns, extension_insns}`) must carry the
    /// matching reserved byte.  Translator-only keys allocated by
    /// `Assembler::get_opnum`'s `setdefault` fallback (`assembler.py`
    /// parity) may live in any non-reserved byte, including gaps below
    /// the canonical high-water byte, and are permitted in
    /// pipeline.insns without a canonical entry.
    /// This regression test guards against build-time/runtime drift on
    /// the bytes the runtime walker actually dispatches (canonical
    /// keys), without locking out the upstream-shaped dynamic-byte
    /// allocator for transient codewriter helpers.
    #[test]
    fn pipeline_insns_canonical_keys_match_canonical_bytes() {
        let observed = insns_opname_to_byte();
        let mut canonical: HashMap<String, u8> = majit_jitcode::insns::wellknown_bh_insns()
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        for (k, v) in majit_jitcode::insns::extension_insns() {
            assert!(
                canonical.insert(k.to_string(), v).is_none(),
                "duplicate opname {k:?} between wellknown_bh_insns() and \
                 extension_insns()",
            );
        }
        for (key, &observed_byte) in observed.iter() {
            if let Some(canonical_byte) = canonical.get(key).copied() {
                assert_eq!(
                    canonical_byte, observed_byte,
                    "pipeline.insns assigned byte {observed_byte} to \
                     canonical key {key:?} but the canonical table has \
                     byte {canonical_byte} — the build-time and runtime \
                     byte tables have drifted",
                );
            } else {
                assert!(
                    !majit_jitcode::insns::is_reserved_opcode_byte(observed_byte),
                    "pipeline.insns key {key:?} is absent from \
                     wellknown_bh_insns() ∪ extension_insns() but \
                     was assigned reserved byte {observed_byte}; \
                     translator-only dynamic bytes must not collide with \
                     canonical/extension opcodes",
                );
            }
        }
    }

    #[test]
    fn index_queries_do_not_decode_jitcode_bodies() {
        let count = jitcode_count();
        assert!(compute_named_jitcode_index("__missing_jitcode_name__").is_none());
        assert_eq!(jitcode_count(), count);
        assert!(get_jitcode_by_index(0).is_some());
        assert!(JITCODE_CELLS[0].get().is_some());
    }

    #[test]
    fn build_time_jitcode_identity_is_shared_across_threads() {
        let here = Arc::as_ptr(get_jitcode_ref_by_index(0).unwrap()) as usize;
        let there =
            std::thread::spawn(|| Arc::as_ptr(get_jitcode_ref_by_index(0).unwrap()) as usize)
                .join()
                .unwrap();
        assert_eq!(there, here);
    }

    #[test]
    fn deserializes_jitcodes_without_error() {
        let jitcodes = all_jitcodes();
        assert!(!jitcodes.is_empty(), "expected at least one jitcode");
    }

    #[test]
    fn canonical_runtime_wrapper_identity_is_shared_across_threads() {
        let first = get_runtime_jitcode_by_index(0).expect("first canonical runtime JitCode");
        let second = std::thread::spawn(|| {
            get_runtime_jitcode_by_index(0).expect("first canonical runtime JitCode on child")
        })
        .join()
        .unwrap();
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn pathed_lookup_is_the_process_wide_callcontrol_dict() {
        let path: &'static str = jitcode_index()
            .paths
            .iter()
            .find(|path| !path.is_empty())
            .expect("at least one graph-backed JitCode")
            .as_str();
        let first = pathed_runtime_jitcode_cached(path).expect("canonical path");
        let again = pathed_runtime_jitcode_cached(path).expect("same path again");
        let other = std::thread::spawn(move || {
            pathed_runtime_jitcode_cached(path).expect("canonical path on another thread")
        })
        .join()
        .unwrap();
        assert!(Arc::ptr_eq(&first, &again));
        assert!(Arc::ptr_eq(&first, &other));
        assert_eq!(
            compute_pathed_jitcode_index(path),
            jitcode_index()
                .paths
                .iter()
                .position(|candidate| candidate == path)
        );
        assert!(pathed_runtime_jitcode_cached("__missing_jitcode_path__").is_none());
        assert!(compute_pathed_jitcode_index("").is_none());
    }

    #[test]
    fn indirect_target_lookup_decodes_only_the_matched_jitcode() {
        let (&fnaddr, &index) = INDIRECTCALLTARGET_BY_FNADDR
            .iter()
            .next()
            .expect("expected at least one frozen indirect-call target");
        assert_eq!(indirectcalltarget_index_for_address(fnaddr), Some(index));
        let jitcode = indirectcalltarget_by_index(index)
            .expect("frozen indirect-call target index must resolve");
        // The map is keyed by runtime address, while `JitCode.fnaddr`
        // stays the build-time one; translate before comparing.
        assert_eq!(
            crate::runtime_fnaddr_patch::runtime_fnaddr(jitcode.fnaddr) as usize,
            fnaddr
        );
        assert!(JITCODE_CELLS[index].get().is_some());
    }

    #[test]
    fn portal_jitcode_resolves_to_unique_jitdriver_entry() {
        // Verify the portal accessor returns the build-time JitCode for the
        // unique driver carrying JitDriverStaticData index 0 (RPython
        // call.py `jd.mainjitcode = self.get_jitcode(jd.portal_graph)`).
        let portal = portal_jitcode().expect("build-time pipeline must register a portal jitcode");
        assert_eq!(portal.jitdriver_sd(), Some(0));
        let main_drivers = COMPILED_JIT_DRIVERS
            .iter()
            .filter(|driver| {
                get_jitcode_ref_by_index(driver.main_jitcode_index)
                    .is_some_and(|jitcode| jitcode.jitdriver_sd() == Some(0))
            })
            .collect::<Vec<_>>();
        assert_eq!(
            main_drivers.len(),
            1,
            "JitDriverStaticData index 0 must resolve to exactly one compiled JIT driver"
        );
        assert_eq!(portal.index(), main_drivers[0].main_jitcode_index);
    }

    #[test]
    fn named_jitcode_resolves_charon_body() {
        let cases = [
            ("list_append", list_append_jitcode(), "w_list_append_inner"),
            (
                "list_setitem",
                list_setitem_jitcode(),
                "w_list_setitem_inner",
            ),
            (
                "list_getitem",
                list_getitem_jitcode(),
                "w_list_getitem_inner",
            ),
            (
                "list_pop_end",
                list_pop_end_jitcode(),
                "w_list_pop_end_inner",
            ),
            ("write_cell", write_cell_jitcode(), "write_cell"),
            ("unwrap_cell", unwrap_cell_jitcode(), "unwrap_cell"),
        ];
        for (name, jc, expect) in cases {
            let jc = jc.unwrap_or_else(|| panic!("case {name}"));
            assert_eq!(jc.name, expect, "case {name}");
            assert!(!jc.code.is_empty(), "case {name}");
        }
    }

    #[test]
    fn opname_round_trips_through_byte() {
        // RPython assembler.py keys are `opname/argcode` (the argcode is
        // appended during `write_insn`). `live/` — the canonical BC_LIVE
        // marker emitted by liveness insertion — is the stablest key to
        // assert on, since every flattened graph touches it at least
        // once. If this test fails the assembler stopped emitting BC_LIVE,
        // which is a bigger parity break.
        let byte = *insns_opname_to_byte()
            .get("live/")
            .expect("`live/` opcode must be in the insns table");
        assert_eq!(opname_for_byte(byte), Some("live/"));
    }

    #[test]
    fn decode_live_skips_offset_size() {
        // `live/` is special-cased: empty argcodes but +OFFSET_SIZE (2)
        // bytes per blackhole.py:1603-1605.
        let live_byte = *insns_opname_to_byte()
            .get("live/")
            .expect("`live/` must be in insns table");
        let code = [live_byte, 0x00, 0x00];
        let op = decode_op_at(&code, 0).expect("live/ must decode");
        assert_eq!(op.opname, "live");
        assert_eq!(op.argcodes, "");
        assert_eq!(op.pc, 0);
        assert_eq!(
            op.next_pc,
            1 + majit_jitcode::liveness::OFFSET_SIZE,
            "live/ must advance by OFFSET_SIZE past the opcode byte",
        );
    }

    #[test]
    fn decode_int_add_reads_ii_operands_and_one_result_byte() {
        // `int_add/ii>i` — 1+1 operand bytes + 1 result byte = 3 bytes
        // after the opcode.
        let op_byte = *insns_opname_to_byte()
            .get("int_add/ii>i")
            .expect("int_add/ii>i must be in insns table");
        let code = [op_byte, 0x01, 0x02, 0x03];
        let op = decode_op_at(&code, 0).expect("int_add must decode");
        assert_eq!(op.opname, "int_add");
        assert_eq!(op.argcodes, "ii>i");
        assert_eq!(op.next_pc, 4);
    }

    #[test]
    fn decode_portal_jitcode_walks_to_end() {
        // Walking the configured portal with `decoded_ops`
        // must reach exactly code.len() if every byte decodes cleanly.
        let jc = portal_jitcode().expect("configured portal must resolve to a jitcode");
        let mut last_next = 0;
        let mut step_count = 0;
        for op in decoded_ops(&jc.code) {
            last_next = op.next_pc;
            step_count += 1;
        }
        assert!(step_count > 0, "should decode at least one op");
        assert_eq!(
            last_next,
            jc.code.len(),
            "decoded stream must end exactly at code.len() for the portal \
             (stopped at {last_next} after {step_count} ops, code.len()={})",
            jc.code.len(),
        );
    }

    #[test]
    fn decode_varlist_reads_length_byte_plus_items() {
        // Synthetic: inline_call_ir_r/dIR>r — d(2) + I(1+N) + R(1+M) + r(1).
        let op_byte = *insns_opname_to_byte()
            .get("inline_call_ir_r/dIR>r")
            .expect("inline_call_ir_r/dIR>r must be in insns table");
        // opcode, d=0x0001, I-list len=2 + 2 regs, R-list len=1 + 1 reg, dst
        // = 1 + 2 + (1+2) + (1+1) + 1 = 9 bytes
        let code = [
            op_byte, 0x01, 0x00, // d
            0x02, 0x00, 0x01, // I: len=2, [0,1]
            0x01, 0x00, // R: len=1, [0]
            0x03, // >r: dst=3
        ];
        let op = decode_op_at(&code, 0).expect("inline_call_ir_r must decode");
        assert_eq!(op.opname, "inline_call_ir_r");
        assert_eq!(op.next_pc, 9);
    }

    #[test]
    fn decode_unknown_opcode_returns_none() {
        // Byte 0xFF should not be a valid opcode — 21 entries go 0..=20.
        let code = [0xFF];
        assert!(
            decode_op_at(&code, 0).is_none(),
            "unknown opcode byte must yield None",
        );
    }

    #[test]
    fn resolve_int_add_reads_both_register_values() {
        // `int_add/ii>i`: canonical — both operands read from int-regs,
        // result written to int-reg. RPython
        // `blackhole.py:@arguments("i", "i", returns="i")`.
        let op_byte = *insns_opname_to_byte()
            .get("int_add/ii>i")
            .expect("int_add/ii>i must be in insns table");
        // code: [opcode, i_idx=2, i_idx=1, dst=0]
        let code = [op_byte, 0x02, 0x01, 0x00];
        let regs = RegisterFileView {
            registers_i: &[0, 7, 42, 0],
            registers_r: &[],
            registers_f: &[],
        };
        let op = resolve_op_at(&code, 0, regs).expect("int_add must resolve");
        assert_eq!(op.decoded.opname, "int_add");
        assert_eq!(op.operands.len(), 2);
        assert_eq!(
            op.operands[0],
            ResolvedOperand::IntReg { reg: 2, value: 42 }
        );
        assert_eq!(op.operands[1], ResolvedOperand::IntReg { reg: 1, value: 7 });
        assert_eq!(op.result, Some(ResolvedResult::Int { reg: 0 }));
    }

    #[test]
    fn resolve_live_yields_no_operands() {
        // `live/`: empty argcodes, OFFSET_SIZE skipped; no operands, no
        // result.
        let live_byte = *insns_opname_to_byte()
            .get("live/")
            .expect("live/ must be in insns table");
        let code = [live_byte, 0x00, 0x00];
        let op = resolve_op_at(&code, 0, RegisterFileView::empty()).expect("live/ must resolve");
        assert!(op.operands.is_empty());
        assert!(op.result.is_none());
        assert_eq!(op.decoded.opname, "live");
    }

    #[test]
    fn resolve_out_of_range_int_reg_returns_none() {
        // int_add/ii>i: opcode reads registers_i[5], but registers_i is
        // only 2 wide. Must surface as decode failure, not a silent 0.
        let op_byte = *insns_opname_to_byte()
            .get("int_add/ii>i")
            .expect("int_add/ii>i must be in insns table");
        let code = [op_byte, 0x00, 0x05, 0x00];
        let regs = RegisterFileView {
            registers_i: &[10, 20],
            registers_r: &[],
            registers_f: &[],
        };
        assert!(resolve_op_at(&code, 0, regs).is_none());
    }

    #[test]
    fn resolve_varlist_reads_each_member() {
        // inline_call_ir_r/dIR>r: d(2) + I(len+items) + R(len+items) + >r(1).
        let op_byte = *insns_opname_to_byte()
            .get("inline_call_ir_r/dIR>r")
            .expect("inline_call_ir_r/dIR>r must be in insns table");
        // descr=0x0102, I=[reg1, reg2], R=[reg0], dst=4
        let code = [
            op_byte, 0x02, 0x01, //
            0x02, 0x01, 0x02, //
            0x01, 0x00, //
            0x04,
        ];
        let regs = RegisterFileView {
            registers_i: &[0, 111, 222],
            registers_r: &[333],
            registers_f: &[],
        };
        let op = resolve_op_at(&code, 0, regs).expect("inline_call_ir_r must resolve");
        assert_eq!(op.operands.len(), 3);
        assert_eq!(
            op.operands[0],
            ResolvedOperand::DescrIdx {
                index: 0x0102,
                is_jitcode: false,
            },
        );
        assert_eq!(
            op.operands[1],
            ResolvedOperand::IntList(vec![(1, 111), (2, 222)]),
        );
        assert_eq!(op.operands[2], ResolvedOperand::RefList(vec![(0, 333)]));
        assert_eq!(op.result, Some(ResolvedResult::Ref { reg: 4 }));
    }

    // Closure check: the strict builder covers every opname in
    // the generated insns table.  Earlier revisions kept this ignored
    // while kind-flow bugs emitted pyre-only mixed signatures such as
    // `int_ge/ir>i` and a late `int_mod/ii>i` leak; those are now fixed
    // at source typing / jtransform emission rather than by adding
    // blackhole aliases.
    #[test]
    fn build_default_bh_builder_matches_insns_table() {
        // The runtime-side `BlackholeInterpBuilder` is reachable
        // from pyre-jit-trace. After `setup_insns + wire_bhimpl_handlers`
        // it must carry the same byte<->opname mapping as the build-time
        // insns bincode, and it must resolve the three well-known
        // opcodes (`live/`, `catch_exception/L`, `rvmprof_code/ii`) when
        // they appear in the table.
        let (builder, _unwired) = build_default_bh_builder_with_unwired_report();
        let expected_live = insns_opname_to_byte().get("live/").copied();
        assert_eq!(Some(builder.op_live), expected_live);
        // Reverse mapping parity: for every opname in the build-time
        // forward table, `builder._insns[byte]` must hold an opname
        // that round-trips to the same byte.  RPython
        // `assembler.py:220` allocates a fresh byte per distinct key
        // (`setdefault(key, len(self.insns))`), so the forward map is
        // injective and the reverse map is naturally 1:1.  Python
        // class-attribute aliases on `BlackholeInterpreter`
        // (`blackhole.py bhimpl_goto_if_not_int_is_true =
        // bhimpl_goto_if_not`) share the handler function under two
        // attribute names but never register two opnames at the same
        // byte in `Assembler.insns`.
        for (key, &byte) in insns_opname_to_byte() {
            let inverse = &builder._insns[byte as usize];
            let inverse_byte = insns_opname_to_byte()
                .get(inverse)
                .copied()
                .unwrap_or_else(|| {
                    panic!(
                        "builder._insns[{byte}] = {inverse:?} but that opname is not in the \
                     forward table"
                    )
                });
            assert_eq!(
                inverse_byte, byte,
                "builder._insns[{byte}] = {inverse:?} maps to byte {inverse_byte}, expected {byte} \
                 (for forward key {key:?})",
            );
        }
    }

    #[test]
    fn build_default_bh_builder_shares_descr_table() {
        let (mut builder, _) = build_default_bh_builder_with_unwired_report();
        assert!(!builder.descrs.is_empty(), "shared table must not be empty");
        assert_eq!(builder.descrs.len(), descr_table().len());
        assert!(
            std::ptr::eq(
                std::ptr::from_ref(builder.descrs).cast::<()>(),
                std::ptr::from_ref(descr_table()).cast::<()>(),
            ),
            "builder must ALIAS the process table, not copy it"
        );
        let bh = builder.acquire_interp();
        assert!(
            std::ptr::eq(
                std::ptr::from_ref(bh.descrs).cast::<()>(),
                std::ptr::from_ref(descr_table()).cast::<()>(),
            ),
            "acquire_interp must alias, not copy (blackhole.py:288)"
        );
    }

    /// Coverage of the *production* builder, which is not the default one.
    ///
    /// `build_pyre_production_bh_builder` delegates to
    /// `build_inline_call_only_bh_builder`, which hand-curates its registered
    /// set family by family, so an emitted byte outside that surface reaches
    /// `dispatch_step`'s unwired-opcode panic — and only once a forward resume
    /// happens to land on it.  Upstream cannot have this failure mode:
    /// `setup_insns(asm.insns)` (`blackhole.py`, :66) resolves every
    /// opname the assembler emitted, so its dispatch table spans the reachable
    /// bytecode universe by construction.
    ///
    /// This is the structural form of that guarantee: `build_emitted_insns()`
    /// IS pyre's `asm.insns` — the serialised build-time `pipeline.insns`,
    /// before the canonical-universe overlay — so every key in it names a byte
    /// a real jitcode in this binary can carry.  Each must be dispatchable.
    ///
    /// Deliberately NOT asserted over `insns_opname_to_byte()`: that map
    /// additionally carries the `wellknown_bh_insns` / `extension_insns`
    /// overlay, i.e. canonical opnames this build never emitted. Their bytes
    /// cannot appear in any jitcode here, so leaving them unregistered is the
    /// same state upstream is in — its `asm.insns` would not list them either.
    #[test]
    fn production_bh_builder_covers_every_build_emitted_opname() {
        let builder = build_pyre_production_bh_builder();
        let mut unregistered: Vec<String> = build_emitted_insns()
            .iter()
            .filter(|(_key, byte)| {
                builder
                    ._insns
                    .get(**byte as usize)
                    .is_none_or(|name| name.is_empty())
            })
            .map(|(key, _)| key.clone())
            .collect();
        unregistered.sort();
        assert_eq!(
            unregistered,
            Vec::<String>::new(),
            "the production blackhole cannot dispatch an opname this build's \
             jitcodes actually contain; register it in \
             `build_inline_call_only_bh_builder` (and confirm the emit-side \
             operand shape matches the wired handler's decoder) rather than \
             widening this assertion",
        );
        let mut unwired: Vec<String> = builder
            .unwired_opnames()
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        unwired.sort();
        assert_eq!(
            unwired,
            Vec::<String>::new(),
            "production builder registered an opname without a handler",
        );
        // The curated list spells each byte as a `BC_*` constant by hand, so it
        // can also drift the other way: a key bound to a byte the canonical
        // table gives to a different opname would decode every occurrence as
        // the wrong instruction.  `setup_insns` only rejects two keys claiming
        // one byte *within* this builder, so nothing else catches it.
        let mut mismatched: Vec<String> = Vec::new();
        for (byte, key) in builder._insns.iter().enumerate() {
            if key.is_empty() {
                continue;
            }
            match insns_opname_to_byte().get(key) {
                Some(&canonical) if canonical as usize == byte => {}
                Some(&canonical) => mismatched.push(format!(
                    "{key:?} registered at byte {byte}, canonical byte is {canonical}"
                )),
                None => mismatched.push(format!(
                    "{key:?} registered at byte {byte} but is not a canonical opname"
                )),
            }
        }
        mismatched.sort();
        assert_eq!(
            mismatched,
            Vec::<String>::new(),
            "production builder's hand-spelled bytes disagree with the canonical \
             opname table",
        );
    }

    /// The overlay-only remainder: canonical opnames that exist in the
    /// blackhole's key universe but that this build never emitted.
    ///
    /// Every entry has a `bhimpl_*` handler — `build_default_bh_builder` wires
    /// all of them and its unwired set is empty (see the snapshot test below)
    /// — so this is a registration gap, not an implementation gap, and it is
    /// unreachable: no jitcode in this binary carries these bytes.  It is
    /// pinned so that an opname MOVING out of this list (because the codewriter
    /// started emitting it) shows up as a test failure to be classified,
    /// instead of silently becoming a live `dispatch_step` panic.
    ///
    /// `vtable_method_ptr/rd>i` left the list exactly that way: routing
    /// `dyn Trait` calls through `CallTarget::Indirect` became the default, so
    /// the codewriter emits it and it is now registered in
    /// `build_inline_call_only_bh_builder` instead.
    ///
    /// `new/d>r` left it the same way: lowering a rebuilt `Result` shell as an
    /// allocation, and a `#[jit_interp]` frontend's declared `struct_allocs`
    /// literal, each make `OpKind::New` reachable from the codewriter. Its
    /// operand shape is `new_with_vtable/d>r`'s — a 2-byte little-endian descr
    /// index then a 1-byte ref register — and `handler_new` decodes exactly
    /// that.
    ///
    /// `new_array/id>r` left with it: the builder reached both with a wired
    /// handler but no opname→byte entry, so the blackhole took
    /// `dispatch_step: unwired opcode=0xd9` on the abort path.  Both are now
    /// registered.
    ///
    /// The two float<->int bitcasts, `convert_float_bytes_to_longlong/f>i`
    /// and `convert_longlong_bytes_to_float/i>f`, left the same way: a
    /// `#[jit_interp]` arm that carries a float across a branch as its bits
    /// emits them, so `build_inline_call_only_bh_builder` registers both and a
    /// resume through such an arm dispatches them instead of panicking
    /// `unwired opcode`.
    ///
    /// `cast_int_to_ptr/i>r` left with them: `rewrite_op_cast_int_to_ptr`
    /// keeps the unary llop, so assembled JitCodes carry the key and the
    /// production overlay registers it next to `cast_ptr_to_int/r>i`.
    ///
    /// The three `getinteriorfield_gc_*` loads left without the emitted set
    /// growing: they are registered pre-emptively, not in response to an
    /// observed panic naming one of them.  Their
    /// `bhimpl_getinteriorfield_gc_*` handlers already existed and only the
    /// opname→byte entry was missing — the same shape `new_array/id>r` was in
    /// when it did panic, and the shape that surfaces as `dispatch_step:
    /// unwired opcode` rather than a decode.  `BUILD_EMITTED_INSNS` records
    /// only what the assembler emitted while analyzing this build's source
    /// set, so absence from it is not by itself proof that no jitcode carries
    /// the byte.
    ///
    /// `record_quasiimmut_field/rdd` is registered.
    /// `jtransform.py rewrite_op_getfield` emits it ahead of a
    /// quasi-immutable getfield, and `handler_record_quasiimmut_field` is
    /// the no-op step.
    #[test]
    fn production_bh_builder_overlay_only_gap_snapshot() {
        let builder = build_pyre_production_bh_builder();
        let mut gap: Vec<String> = insns_opname_to_byte()
            .iter()
            .filter(|(key, byte)| {
                !build_emitted_insns().contains_key(*key)
                    && builder
                        ._insns
                        .get(**byte as usize)
                        .is_none_or(|name| name.is_empty())
            })
            .map(|(key, _)| key.clone())
            .collect();
        gap.sort();
        // blackhole.py has no `bhimpl_call_assembler` and no leftover
        // `cond_call_*_ext` / `record_known_result_*_ext`. Those
        // `extension_insns()` bytes stay reserved for tracing
        // `opimpl_call_assembler_*` / `opimpl_cond_call_void`.
        let expected = [
            "assert_not_none/r",
            "call_assembler_float_ext/P",
            "call_assembler_int_ext/P",
            "call_assembler_ref_ext/P",
            "call_assembler_void_ext/P",
            "check_neg_index/rid>i",
            "cond_call_value_int_ext/P",
            "cond_call_value_ref_ext/P",
            "cond_call_void_ext/P",
            "gc_load_indexed_f/riiii>f",
            "gc_load_indexed_i/riiii>i",
            "getlistitem_gc_f/ridd>f",
            "getlistitem_gc_i/ridd>i",
            "getlistitem_gc_r/ridd>r",
            "guard_class/r>i",
            "newlist/idddd>r",
            "newlist_clear/idddd>r",
            "newlist_hint/idddd>r",
            "record_exact_class/ri",
            "record_known_result_int_ext/P",
            "record_known_result_ref_ext/P",
            "rvmprof_code/ii",
        ];
        assert_eq!(
            gap, expected,
            "the unreachable production-blackhole registration gap drifted; if \
             an opname LEFT this list the codewriter now emits it, so it must \
             be registered in `build_inline_call_only_bh_builder`",
        );
    }

    #[test]
    fn default_bh_builder_unwired_set_matches_task_85_snapshot() {
        // Lock-in: every generated opname must be wired by
        // `wire_bhimpl_handlers`.  Any entry here means codewriter /
        // regalloc emitted a kind shape that no RPython blackhole handler
        // has — fix at upstream emission, do NOT add a `*_r>i` /
        // `*_ir>i` alias.
        let (_builder, mut unwired) = build_default_bh_builder_with_unwired_report();
        unwired.sort();
        // blackhole.py has no `bhimpl_call_assembler` and no leftover
        // `cond_call_*_ext` / `record_known_result_*_ext`. Those
        // `extension_insns()` bytes stay reserved for tracing
        // `opimpl_call_assembler_*` / `opimpl_cond_call_void`.
        let expected = [
            "call_assembler_float_ext/P",
            "call_assembler_int_ext/P",
            "call_assembler_ref_ext/P",
            "call_assembler_void_ext/P",
            "cond_call_value_int_ext/P",
            "cond_call_value_ref_ext/P",
            "cond_call_void_ext/P",
            "record_known_result_int_ext/P",
            "record_known_result_ref_ext/P",
        ]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
        assert_eq!(
            unwired, expected,
            "Unwired-opname snapshot drifted. If a new entry \
             appeared, find the new kind-flow bug upstream instead of \
             adding a bhhandler alias.  Existing entries document a \
             pending upstream rewrite — see the `expected` literal.",
        );
    }

    #[test]
    fn default_bh_builder_handler_coverage_report() {
        // Diagnostic: surface the opnames in the real insns table that
        // `wire_bhimpl_handlers` did NOT override.  These fall back to
        // the `setup_insns` placeholder and would panic on dispatch.
        //
        // `BlackholeInterpBuilder::unwired_opnames()` is the accessor
        // that returns the gap.  The current expected unwired set is
        // documented in
        // `default_bh_builder_unwired_set_matches_task_85_snapshot`.
        //
        // The test does NOT fail on unwired opnames — it just reports
        // them.  Gating turns on later once the documented unwired
        // entries are closed at upstream emission.
        let (builder, _) = build_default_bh_builder_with_unwired_report();
        let total = insns_opname_to_byte().len();
        let mut unwired: Vec<&str> = builder.unwired_opnames();
        unwired.sort_unstable();
        let wired = total - unwired.len();
        eprintln!(
            "[jitcode_runtime] coverage: {wired}/{total} opnames wired; \
             {} unwired: {:?}",
            unwired.len(),
            unwired,
        );
        // Sanity: `live/` must be present in the insns table and wired.
        // Any binary that lacks it is structurally broken.
        assert!(
            insns_opname_to_byte().contains_key("live/"),
            "live/ missing from real insns table — broken build",
        );
        assert!(
            !unwired.iter().any(|k| *k == "live/"),
            "live/ must be wired by wire_bhimpl_handlers",
        );
    }

    #[test]
    fn portal_jitcode_is_complete_in_canonical_store() {
        // RPython parity target: the deserialized `ALL_JITCODES` entries
        // are themselves the canonical objects produced by
        // `CodeWriter.make_jitcodes()`. Avoid the transitional
        // build→runtime `From` adapter here and assert directly on the
        // canonical object that build.rs persisted.
        let bt_jc = portal_jitcode().expect("configured portal must resolve to a jitcode");
        assert!(!bt_jc.code.is_empty());
        // `warmspot.py split_graph_and_record_jitdriver` registers the copy
        // cut at `jit_merge_point`.
        let eval_driver = COMPILED_JIT_DRIVERS
            .iter()
            .find(|driver| driver.portal.canonical_key() == "eval::eval_loop_jit_portal")
            .expect("compiled drivers must contain the split eval portal");
        assert_eq!(eval_driver.main_jitcode_index, bt_jc.index());
        assert_eq!(
            bt_jc.num_regs_and_consts_i(),
            bt_jc.num_regs_i() + bt_jc.constants_i.len()
        );
        assert_eq!(
            bt_jc.num_regs_and_consts_r(),
            bt_jc.num_regs_r() + bt_jc.constants_r.len()
        );
        assert_eq!(
            bt_jc.num_regs_and_consts_f(),
            bt_jc.num_regs_f() + bt_jc.constants_f.len()
        );
    }

    #[test]
    fn dispatch_loop_executes_int_add_via_real_insns_table() {
        // Confirm the build-time `pipeline.insns` byte
        // assignments resolve to the real `wire_bhimpl_handlers`
        // dispatch entries — a hand-assembled bytecode using those
        // bytes runs end-to-end through
        // `BlackholeInterpBuilder::dispatch_loop` and lands the
        // expected `bhimpl_int_add` result.
        //
        // RPython parity: same shape as `setup_insns + dispatch_loop +
        // bhimpl_int_add` (blackhole.py + 452-460), but driven
        // by the artifact this binary actually loads — not a synthetic
        // 3-entry insns dict like the analogous test inside
        // majit-metainterp. Closes the build-artifact → runtime →
        // BlackholeInterpBuilder round trip end-to-end.
        //
        // Use the reporting helper here because this test inspects a tiny
        // synthetic bytecode slice and does not need the strict coverage
        // assertion exercised by `build_default_bh_builder()`.
        let (mut builder, _unwired) = build_default_bh_builder_with_unwired_report();

        let table = insns_opname_to_byte();
        let live_byte = *table.get("live/").expect("`live/` must be in insns");
        let int_add_byte = *table
            .get("int_add/ii>i")
            .expect("`int_add/ii>i` must be in insns");
        let int_return_byte = *table
            .get("int_return/i")
            .expect("`int_return/i` must be in insns");

        // live + int_add(r0, r1) → r2 + int_return(r2). The two zero
        // bytes after `live/` are the OFFSET_SIZE liveness offset that
        // `bhimpl_live` skips (blackhole.py).
        let code: Vec<u8> = vec![
            live_byte,
            0x00,
            0x00, //
            int_add_byte,
            0x00,
            0x01,
            0x02, //
            int_return_byte,
            0x02,
        ];

        let mut bh = builder.acquire_interp();
        bh.registers_i = vec![0i64; 3];
        bh.registers_i[0] = 10;
        bh.registers_i[1] = 32;

        let result = builder.dispatch_loop(&mut bh, &code, 0);
        assert!(
            matches!(
                result,
                Err(majit_metainterp::blackhole::DispatchError::LeaveFrame)
            ),
            "dispatch_loop must terminate with LeaveFrame after \
             int_return; got {result:?}",
        );
        assert_eq!(bh.tmpreg_i, 42, "int_add(10, 32) must produce 42");
    }

    #[test]
    fn dispatch_loop_chains_int_add_then_int_sub_via_real_insns_table() {
        // Chain two binops + a label-free linear control
        // flow through `dispatch_loop`. Validates that the second
        // binop reads the register the first wrote (multi-step value
        // flow through the register file) and that two distinct wired
        // bhimpl handlers (`bhimpl_int_add`, `bhimpl_int_sub`) advance
        // `position` correctly back to back.
        //
        // RPython parity: blackhole.py `bhimpl_int_add` +
        // :462-464 `bhimpl_int_sub` chained with the same register
        // file, identical to RPython's per-op `_get_method.handler`
        // dispatch.
        let (mut builder, _unwired) = build_default_bh_builder_with_unwired_report();

        let table = insns_opname_to_byte();
        let live_byte = *table.get("live/").expect("`live/` must be in insns");
        let int_add_byte = *table
            .get("int_add/ii>i")
            .expect("`int_add/ii>i` must be in insns");
        let int_sub_byte = *table
            .get("int_sub/ii>i")
            .expect("`int_sub/ii>i` must be in insns");
        let int_return_byte = *table
            .get("int_return/i")
            .expect("`int_return/i` must be in insns");

        // live + int_add(r0=10, r1=32) → r2 (=42)
        //      + int_sub(r2, r0)        → r3 (=32)
        //      + int_return(r3)
        let code: Vec<u8> = vec![
            live_byte,
            0x00,
            0x00, //
            int_add_byte,
            0x00,
            0x01,
            0x02, //
            int_sub_byte,
            0x02,
            0x00,
            0x03, //
            int_return_byte,
            0x03,
        ];

        let mut bh = builder.acquire_interp();
        bh.registers_i = vec![0i64; 4];
        bh.registers_i[0] = 10;
        bh.registers_i[1] = 32;

        let result = builder.dispatch_loop(&mut bh, &code, 0);
        assert!(
            matches!(
                result,
                Err(majit_metainterp::blackhole::DispatchError::LeaveFrame)
            ),
            "dispatch_loop must terminate with LeaveFrame after \
             int_return; got {result:?}",
        );
        assert_eq!(bh.registers_i[2], 42, "first int_add must store 42 into r2",);
        assert_eq!(
            bh.registers_i[3], 32,
            "second int_sub must store (42-10)=32 into r3",
        );
        assert_eq!(bh.tmpreg_i, 32, "int_return must place r3 into tmpreg_i",);
    }

    #[test]
    fn dispatch_loop_executes_count_up_loop_via_real_insns_table() {
        // Closed loop via `int_lt/ii>i` + `goto_if_not/iL`
        // + `goto/L` — exercises the dispatch_loop's absolute-target
        // label semantics on both backward and forward jumps.
        //
        // pyre's build-time assembler does not currently emit the
        // fused `goto_if_not_int_*` family; the orthodox unfused
        // shape (`int_lt` produces a 0/1 register, `goto_if_not`
        // dispatches on it) is what `pipeline.insns` actually
        // contains, so the loop is wired against the unfused pair.
        //
        // Loop body (count r0 from 0 up to r1=5, step r2=1):
        //
        //   PC=0:  live/
        //   PC=3:  LOOP: int_lt r0, r1 → r3
        //   PC=7:        goto_if_not r3, END   (forward jump)
        //   PC=11:       int_add r0, r2 → r0
        //   PC=15:       goto LOOP=3            (backward jump)
        //   PC=18: END:  int_return r0
        //
        // RPython parity: blackhole.py `bhimpl_goto_if_not`
        // — target is an absolute byte offset into the jitcode
        // `code` array.
        let (mut builder, _unwired) = build_default_bh_builder_with_unwired_report();

        let table = insns_opname_to_byte();
        let live_byte = *table.get("live/").expect("`live/` must be in insns");
        let int_lt_byte = *table
            .get("int_lt/ii>i")
            .expect("`int_lt/ii>i` must be in insns");
        let goto_if_not_byte = *table
            .get("goto_if_not/iL")
            .expect("`goto_if_not/iL` must be in insns");
        let int_add_byte = *table
            .get("int_add/ii>i")
            .expect("`int_add/ii>i` must be in insns");
        let goto_byte = *table.get("goto/L").expect("`goto/L` must be in insns");
        let int_return_byte = *table
            .get("int_return/i")
            .expect("`int_return/i` must be in insns");

        // PC offsets (must match the layout above): LOOP_HEAD = 3, END = 18.
        const LOOP_HEAD: u8 = 3;
        const END: u8 = 18;
        let code: Vec<u8> = vec![
            // PC=0:  live/  (1 + OFFSET_SIZE = 3 bytes)
            live_byte,
            0x00,
            0x00, //
            // PC=3:  int_lt r0, r1 → r3  (1 + 3 = 4 bytes)
            int_lt_byte,
            0x00,
            0x01,
            0x03, //
            // PC=7:  goto_if_not r3, END  (1 + 1 + 2 = 4 bytes)
            goto_if_not_byte,
            0x03,
            END,
            0x00, //
            // PC=11: int_add r0, r2 → r0  (1 + 3 = 4 bytes)
            int_add_byte,
            0x00,
            0x02,
            0x00, //
            // PC=15: goto LOOP_HEAD  (1 + 2 = 3 bytes)
            goto_byte,
            LOOP_HEAD,
            0x00, //
            // PC=18: int_return r0  (1 + 1 = 2 bytes)
            int_return_byte,
            0x00,
        ];
        assert_eq!(code.len(), 20, "loop bytecode must be exactly 20 bytes");

        let mut bh = builder.acquire_interp();
        bh.registers_i = vec![0i64; 4];
        bh.registers_i[0] = 0; // counter
        bh.registers_i[1] = 5; // limit
        bh.registers_i[2] = 1; // step
        // r3 is the int_lt result slot — left zero, written each loop iter.

        let result = builder.dispatch_loop(&mut bh, &code, 0);
        assert!(
            matches!(
                result,
                Err(majit_metainterp::blackhole::DispatchError::LeaveFrame)
            ),
            "dispatch_loop must terminate with LeaveFrame after the \
             loop's int_return; got {result:?}",
        );
        assert_eq!(
            bh.registers_i[0], 5,
            "counter r0 must reach the limit (5) before the loop \
             exits via int_lt(5,5)=0 → goto_if_not jumps to END",
        );
        assert_eq!(bh.tmpreg_i, 5, "int_return must place r0 into tmpreg_i");
    }

    #[test]
    fn dispatch_loop_executes_float_add_via_real_insns_table() {
        // Float register file + ff>f decode/encode +
        // void_return termination.
        //
        //   PC=0:  live/
        //   PC=3:  float_add r0, r1 → r2
        //   PC=7:  void_return
        //
        // RPython parity: blackhole.py `bhimpl_float_add` +
        // :859-862 `bhimpl_void_return`. Pyre's `registers_f` stores
        // `f64::to_bits() as i64`; the `bhhandler_ff_f!` macro decodes
        // via `f64::from_bits` on read and `to_bits()` on write, so
        // the test setup mirrors that encoding.
        //
        // `float_return/f` is wired in `wire_bhimpl_handlers` but is
        // NOT in the build-time `pipeline.insns` (no jitcode emitted
        // by the production assembler currently returns a float — all
        // float ops feed back into either box_float or another float
        // op). The test validates the float register file + binop
        // decode end-to-end and inspects `registers_f[2]` directly.
        let (mut builder, _unwired) = build_default_bh_builder_with_unwired_report();

        let table = insns_opname_to_byte();
        let live_byte = *table.get("live/").expect("`live/` must be in insns");
        let float_add_byte = *table
            .get("float_add/ff>f")
            .expect("`float_add/ff>f` must be in insns");
        let void_return_byte = *table
            .get("void_return/")
            .expect("`void_return/` must be in insns");

        let code: Vec<u8> = vec![
            live_byte,
            0x00,
            0x00, //
            float_add_byte,
            0x00,
            0x01,
            0x02, //
            void_return_byte,
        ];

        let mut bh = builder.acquire_interp();
        bh.registers_f = vec![0i64; 3];
        bh.registers_f[0] = (1.5_f64).to_bits() as i64;
        bh.registers_f[1] = (2.5_f64).to_bits() as i64;

        let result = builder.dispatch_loop(&mut bh, &code, 0);
        assert!(
            matches!(
                result,
                Err(majit_metainterp::blackhole::DispatchError::LeaveFrame)
            ),
            "dispatch_loop must terminate with LeaveFrame after \
             void_return; got {result:?}",
        );
        assert_eq!(
            f64::from_bits(bh.registers_f[2] as u64),
            4.0,
            "float_add(1.5, 2.5) must store 4.0 (bits) into r2",
        );
        assert_eq!(
            bh.return_type,
            majit_metainterp::blackhole::BhReturnType::Void,
            "void_return must set return_type = Void",
        );
    }

    #[test]
    fn dispatch_loop_loads_constant_via_setposition_lifecycle() {
        // Full RPython-shape lifecycle —
        // `acquire_interp` + `setposition` + `dispatch_loop`. Earlier
        // dispatch_loop tests bypassed `setposition` by hand-setting
        // `bh.registers_i = vec![...]`; here we construct a real
        // runtime `JitCode` with `c_num_regs_i = 1` and
        // `constants_i = [42]` so `setposition` allocates a register
        // file of `num_regs_and_consts_i() = 2` slots and copies the
        // constant into slot 1 (RPython
        // `blackhole.py setposition` parity).
        //
        //   slot 0 = scratch dst
        //   slot 1 = constant 42 (preloaded by setposition)
        //
        //   PC=0:  live/
        //   PC=3:  int_copy r1 → r0    (r0 := constant)
        //   PC=6:  int_return r0
        //
        // RPython parity: `bhimpl_int_copy` (blackhole.py)
        // reads from `registers_i[code[pc]]` and writes
        // `registers_i[code[pc+1]]`, which validates that the
        // constants area is reachable through the same register-index
        // protocol the bhimpl handlers consume.
        let (mut builder, _unwired) = build_default_bh_builder_with_unwired_report();

        let table = insns_opname_to_byte();
        let live_byte = *table.get("live/").expect("`live/` must be in insns");
        let int_copy_byte = *table
            .get("int_copy/i>i")
            .expect("`int_copy/i>i` must be in insns");
        let int_return_byte = *table
            .get("int_return/i")
            .expect("`int_return/i` must be in insns");

        let code: Vec<u8> = vec![
            live_byte,
            0x00,
            0x00, //
            int_copy_byte,
            0x01,
            0x00, //   r1 (= constant) → r0
            int_return_byte,
            0x00,
        ];

        let jc = {
            let inner = majit_metainterp::jitcode::JitCode::new("test_setposition");
            inner.set_body(majit_jitcode::jitcode::JitCodeBody {
                c_num_regs_i: 1,
                constants_i: vec![42],
                code: code.clone(),
                ..Default::default()
            });
            std::sync::Arc::new(inner)
        };

        let mut bh = builder.acquire_interp();
        bh.setposition(jc.clone(), 0);

        // setposition must have allocated num_regs_i + constants_i.len() slots
        // and copied the constant into the upper half.
        assert_eq!(
            bh.registers_i.len(),
            2,
            "setposition must size registers_i to num_regs_and_consts_i = 2",
        );
        assert_eq!(
            bh.registers_i[1], 42,
            "setposition must copy constants_i[0]=42 into slot num_regs_i = 1",
        );

        let result = builder.dispatch_loop(&mut bh, &jc.code, 0);
        assert!(
            matches!(
                result,
                Err(majit_metainterp::blackhole::DispatchError::LeaveFrame)
            ),
            "dispatch_loop must terminate with LeaveFrame after \
             int_return; got {result:?}",
        );
        assert_eq!(
            bh.registers_i[0], 42,
            "int_copy r1→r0 must move the preloaded constant 42 into r0",
        );
        assert_eq!(bh.tmpreg_i, 42, "int_return must surface 42 via tmpreg_i");
    }

    #[test]
    fn dispatch_loop_executes_ref_return_via_real_insns_table() {
        // Ref register file + ref_return r-typed
        // termination — fills the third register-file dimension that
        // the earlier dispatch_loop tests did not touch.
        //
        // The build-time insns table has only `ref_return/r` from the
        // `ref_*` family (no `ref_copy/r>r`, no `ref_push/r`, no
        // `ref_pop/>r`) — pyre's production assembler does not emit
        // those today. So the smallest ref-typed round-trip uses
        // `ref_return/r` as the sole ref-side opcode and validates
        // that `registers_r[k]` is reachable through the standard
        // `r`-argcode protocol that `bhhandler_*` macros consume.
        //
        //   PC=0: live/
        //   PC=3: ref_return r0
        //
        // RPython parity: blackhole.py `bhimpl_ref_return`.
        // `registers_r` and `tmpreg_r` store ref pointers as raw `i64`
        // bits; the test uses an arbitrary nonzero pattern to verify
        // the read is byte-for-byte without dereferencing.
        let (mut builder, _unwired) = build_default_bh_builder_with_unwired_report();

        let table = insns_opname_to_byte();
        let live_byte = *table.get("live/").expect("`live/` must be in insns");
        let ref_return_byte = *table
            .get("ref_return/r")
            .expect("`ref_return/r` must be in insns");

        let code: Vec<u8> = vec![
            live_byte,
            0x00,
            0x00, //
            ref_return_byte,
            0x00,
        ];

        let mut bh = builder.acquire_interp();
        bh.registers_r = vec![0i64; 1];
        // Arbitrary nonzero pattern. Treated as a raw ref pointer; the
        // test does not dereference it.
        let probe: i64 = 0x1234_5678_9abc_def0_u64 as i64;
        bh.registers_r[0] = probe;

        let result = builder.dispatch_loop(&mut bh, &code, 0);
        assert!(
            matches!(
                result,
                Err(majit_metainterp::blackhole::DispatchError::LeaveFrame)
            ),
            "dispatch_loop must terminate with LeaveFrame after \
             ref_return; got {result:?}",
        );
        assert_eq!(
            bh.tmpreg_r, probe,
            "ref_return must place r0's bits into tmpreg_r byte-for-byte",
        );
        assert_eq!(
            bh.return_type,
            majit_metainterp::blackhole::BhReturnType::Ref,
            "ref_return must set return_type = Ref",
        );
    }

    #[test]
    fn dispatch_loop_with_probe_captures_opcode_sequence_and_preserves_result() {
        // Probe-hook variant of dispatch_loop — first
        // shadow-execution scaffold. Each dispatched opcode invokes
        // the probe BEFORE the handler runs, so a shadow caller
        // (MIFrame side) can capture the jitcode op
        // sequence and compare it against the trace IR emitted by the
        // trait-based `execute_opcode_step`. The probe must NOT
        // change the dispatch result — running the same int_add
        // bytecode through `dispatch_loop_with_probe` must produce
        // the same `tmpreg_i==42` + LeaveFrame as the bare
        // `dispatch_loop`.
        let (mut builder, _unwired) = build_default_bh_builder_with_unwired_report();

        let table = insns_opname_to_byte();
        let live_byte = *table.get("live/").expect("`live/` must be in insns");
        let int_add_byte = *table
            .get("int_add/ii>i")
            .expect("`int_add/ii>i` must be in insns");
        let int_return_byte = *table
            .get("int_return/i")
            .expect("`int_return/i` must be in insns");

        // Same shape as the bare int_add dispatch test:
        //   PC=0: live/                3 bytes
        //   PC=3: int_add r0, r1 → r2  4 bytes
        //   PC=7: int_return r2        2 bytes
        let code: Vec<u8> = vec![
            live_byte,
            0x00,
            0x00, //
            int_add_byte,
            0x00,
            0x01,
            0x02, //
            int_return_byte,
            0x02,
        ];

        let mut bh = builder.acquire_interp();
        bh.registers_i = vec![0i64; 3];
        bh.registers_i[0] = 10;
        bh.registers_i[1] = 32;

        let mut captured: Vec<(usize, u8, String)> = Vec::new();
        let result =
            builder.dispatch_loop_with_probe(&mut bh, &code, 0, |_bh, pc, opcode, opname| {
                captured.push((pc, opcode, opname.to_string()));
            });

        assert!(
            matches!(
                result,
                Err(majit_metainterp::blackhole::DispatchError::LeaveFrame)
            ),
            "dispatch_loop_with_probe must terminate with LeaveFrame after \
             int_return; got {result:?}",
        );
        assert_eq!(
            bh.tmpreg_i, 42,
            "probe must not perturb dispatch — int_add(10,32)→42 still \
             surfaces via tmpreg_i",
        );

        // Probe fired exactly once per opcode at the opcode-byte
        // position (not after operand decode).
        assert_eq!(
            captured.len(),
            3,
            "probe must fire exactly once per opcode (live + int_add + \
             int_return); got {captured:?}",
        );
        assert_eq!(captured[0], (0, live_byte, "live/".to_string()));
        assert_eq!(captured[1], (3, int_add_byte, "int_add/ii>i".to_string()));
        assert_eq!(
            captured[2],
            (7, int_return_byte, "int_return/i".to_string())
        );
    }

    #[test]
    fn dispatch_loop_probe_observes_register_state_at_each_op() {
        // Probe receives `&BlackholeInterpreter` at each
        // firing — the second piece of shadow-execution scaffolding.
        // The closure can read register values BEFORE the upcoming
        // handler runs, capturing the input data flow needed to
        // compare against the trace IR emitted by the trait-based
        // `execute_opcode_step`.
        //
        // Bytecode: live + int_add(r0,r1)→r2 + int_sub(r2,r0)→r3
        //         + int_return(r3)  (same shape as the chained
        // int_add/int_sub dispatch test).
        //
        // Probe captures `registers_i[0..4]` at every firing. The
        // sequence must be:
        //
        //   probe[0] (live/):       [10, 32, 0, 0]   (initial state)
        //   probe[1] (int_add):     [10, 32, 0, 0]   (operands visible, dst still 0)
        //   probe[2] (int_sub):     [10, 32, 42, 0]  (int_add's effect now visible)
        //   probe[3] (int_return):  [10, 32, 42, 32] (int_sub's effect now visible)
        //
        // The non-trivial validation: at probe[2], `registers_i[2] == 42`
        // — proving the probe sees the live data flow exactly as the
        // chained handlers produce it. Shadow-execute logic can use
        // this to read input register values per op without re-running
        // the handler chain.
        let (mut builder, _unwired) = build_default_bh_builder_with_unwired_report();

        let table = insns_opname_to_byte();
        let live_byte = *table.get("live/").expect("`live/` must be in insns");
        let int_add_byte = *table
            .get("int_add/ii>i")
            .expect("`int_add/ii>i` must be in insns");
        let int_sub_byte = *table
            .get("int_sub/ii>i")
            .expect("`int_sub/ii>i` must be in insns");
        let int_return_byte = *table
            .get("int_return/i")
            .expect("`int_return/i` must be in insns");

        let code: Vec<u8> = vec![
            live_byte,
            0x00,
            0x00, //
            int_add_byte,
            0x00,
            0x01,
            0x02, //
            int_sub_byte,
            0x02,
            0x00,
            0x03, //
            int_return_byte,
            0x03,
        ];

        let mut bh = builder.acquire_interp();
        bh.registers_i = vec![0i64; 4];
        bh.registers_i[0] = 10;
        bh.registers_i[1] = 32;

        let mut snapshots: Vec<(String, [i64; 4])> = Vec::new();
        let result =
            builder.dispatch_loop_with_probe(&mut bh, &code, 0, |bh_view, _pc, _opcode, opname| {
                snapshots.push((
                    opname.to_string(),
                    [
                        bh_view.registers_i[0],
                        bh_view.registers_i[1],
                        bh_view.registers_i[2],
                        bh_view.registers_i[3],
                    ],
                ));
            });

        assert!(
            matches!(
                result,
                Err(majit_metainterp::blackhole::DispatchError::LeaveFrame)
            ),
            "dispatch_loop_with_probe must terminate with LeaveFrame; \
             got {result:?}",
        );
        assert_eq!(bh.tmpreg_i, 32, "int_return must place r3=32 into tmpreg_i");
        assert_eq!(
            snapshots.len(),
            4,
            "probe must fire 4 times (live + int_add + int_sub + int_return)",
        );
        // probe[0] live/: pre-everything snapshot.
        assert_eq!(snapshots[0].0, "live/");
        assert_eq!(snapshots[0].1, [10, 32, 0, 0]);
        // probe[1] int_add: about to compute r2 = r0 + r1; r2 still 0.
        assert_eq!(snapshots[1].0, "int_add/ii>i");
        assert_eq!(snapshots[1].1, [10, 32, 0, 0]);
        // probe[2] int_sub: int_add's effect (r2=42) now visible; r3 still 0.
        assert_eq!(snapshots[2].0, "int_sub/ii>i");
        assert_eq!(snapshots[2].1, [10, 32, 42, 0]);
        // probe[3] int_return: int_sub's effect (r3=32) now visible.
        assert_eq!(snapshots[3].0, "int_return/i");
        assert_eq!(snapshots[3].1, [10, 32, 42, 32]);
    }
}
