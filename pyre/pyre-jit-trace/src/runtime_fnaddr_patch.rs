//! Runtime patcher for stale build-time fnaddrs in deserialized JitCodes.
//!
//! RPython's translator AOT-compiles every helper into the same C binary as
//! the runtime metainterp, so `JitCode.fnaddr` and the funcptr entries the
//! codewriter materializes into `JitCode.constants_i` (`jtransform.py:455-471`
//! `handle_residual_call` + `:614-623 direct_funcptr_value`) are linker-
//! resolved C addresses that the runtime executes via `cpu.bh_call_*`
//! without further bookkeeping.
//!
//! Pyre's `majit-translate` runs in `pyre-jit-trace/build.rs` — a separate
//! cargo build-script process from `pyre-dynasm` (and any other pyre runtime
//! binary).  The fnaddrs the codewriter captured therefore reflect the
//! build-script process's `pyre_interpreter::jit_trace_fnaddrs()` snapshot,
//! whose addresses are invalidated by ASLR (per-process random slide) and by
//! the divergent executable layouts (the build-script binary embeds a
//! subset of the runtime's symbols).  A walker that follows
//! `execute_residual_call`'s elidable-EI branch
//! (`jitcode_dispatch::residual_call`'s `try_fold_pure_call_via_executor`)
//! into one of those stale addresses dereferences arbitrary memory →
//! SEGV.
//!
//! This module bridges that gap.  At build time
//! `pyre-jit-trace/build.rs` serialises the `(path, build_fnaddr)` table
//! that `pyre_interpreter::jit_trace_fnaddrs()` returned for the
//! codewriter.  At runtime [`patch_constants_i_fnaddrs`] re-queries
//! `jit_trace_fnaddrs()` (now reading the runtime process's addresses)
//! and rewrites only slots that carry a provenance record: `JitCode.fnaddr`
//! via `fnaddr_reloc`, and `constants_i` via `reloc_consts_i`
//! (`assembler.py emit_const` stores the symbolic object itself;
//! `call.py get_jitcode` stores `getfunctionptr(graph)`).  After the
//! patch the walker's `call_int_function(funcptr, args)` invokes the
//! correct runtime entry point, matching the upstream linker-resolved
//! invariant.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use majit_jitcode::jitcode::{
    ConstIRelocKind, JitCode, StrConstDescriptor, StrConstKind, unicode_const_cells,
};

/// Path recorded for a `symbolic_fnaddr_for_path` / `symbolic_fnaddr_for_target`
/// hash. The codewriter's registry lives in the build-script process; this is
/// that snapshot (`pipeline.symbolic_fnaddr_paths`).
pub fn symbolic_fnaddr_path(fnaddr: i64) -> Option<&'static str> {
    static TABLE: LazyLock<Vec<(i64, String)>> = LazyLock::new(|| {
        const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/symbolic_fnaddr_paths.bin"));
        let mut entries: Vec<(i64, String)> = bincode::deserialize(BYTES).unwrap_or_else(|e| {
            panic!(
                "pyre-jit-trace: failed to deserialize symbolic_fnaddr_paths.bin ({} bytes): {e}",
                BYTES.len(),
            )
        });
        entries.sort_by_key(|entry| entry.0);
        entries
    });
    let table: &'static [(i64, String)] = &TABLE;
    table
        .binary_search_by_key(&fnaddr, |entry| entry.0)
        .ok()
        .map(|index| table[index].1.as_str())
}

/// Build-time `(path, build_fnaddr)` snapshot — bincoded by
/// `pyre-jit-trace/build.rs` from
/// `pyre_interpreter::jit_trace_fnaddrs()` immediately before the
/// codewriter consumes it.  Each entry shares its `path` with the
/// runtime call to `jit_trace_fnaddrs()` below; only the `i64` address
/// differs across processes.
fn build_time_fnaddr_bindings() -> Vec<(String, i64)> {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fnaddr_bindings.bin"));
    bincode::deserialize(BYTES).unwrap_or_else(|e| {
        panic!(
            "pyre-jit-trace: failed to deserialize fnaddr_bindings.bin \
             ({} bytes): {e}",
            BYTES.len(),
        )
    })
}

/// Apply the build → runtime fnaddr correspondence to every JitCode the
/// caller just deserialised.  Mutates each Arc in place — refcount must
/// be 1 on entry (the per-index loader satisfies this because
/// `bincode::deserialize` produces a fresh `Arc::new(...)` shell before any
/// consumer can clone it).
///
/// `JitCode.fnaddr` carries the shell-level fnaddr the codewriter
/// recorded in `CallControl::get_jitcode` (`codewriter/call.rs`); the per-
/// instruction funcptr operands the residual_call dispatcher reads
/// land in `JitCodeBody.constants_i` via the assembler's
/// `emit_const_i_reloc` path (`codewriter/assembler.rs`). Both are
/// rewritten only through the `{ path, symbolic }` record written at
/// that decision — matching `assembler.py emit_const` storing the
/// symbolic object itself rather than inferring a funcptr from integer
/// bits.
pub fn patch_constants_i_fnaddrs(jitcodes: &mut [Arc<JitCode>]) {
    let has_fnaddr_reloc = jitcodes.iter().any(|jc| {
        matches!(jc.fnaddr_reloc, Some(ConstIRelocKind::FnAddr { .. }))
            || jc.try_body().is_some_and(|body| {
                body.reloc_consts_i
                    .iter()
                    .any(|d| matches!(d.kind, ConstIRelocKind::FnAddr { .. }))
            })
    });
    if !has_fnaddr_reloc {
        return;
    }

    for arc in jitcodes.iter_mut() {
        let jc = Arc::get_mut(arc).expect(
            "patch_constants_i_fnaddrs: Arc<JitCode> already shared before patch — \
             every caller must run this before publishing the table to consumers",
        );
        if let Some(ConstIRelocKind::FnAddr { path, .. }) = &jc.fnaddr_reloc
            && let Some(runtime) = runtime_fnaddr_by_path(path)
        {
            jc.fnaddr = runtime;
        }
        // Some shells reach the persisted table without a committed body
        // (e.g. `Default::default()` placeholders kept for `Arc<JitCode>::
        // default()` consumers in `BlackholeInterpreter::new`); they carry
        // empty `constants_i` so skipping the body-mut access is safe.
        if jc.try_body().is_some() {
            let body = jc.body_mut();
            let updates: Vec<(usize, i64)> = body
                .reloc_consts_i
                .iter()
                .filter_map(|desc| {
                    let ConstIRelocKind::FnAddr { path, .. } = &desc.kind else {
                        return None;
                    };
                    runtime_fnaddr_for_reloc_path(path)
                        .map(|runtime| (desc.constants_i_index, runtime))
                })
                .collect();
            for (index, runtime) in updates {
                body.constants_i[index] = runtime;
            }
        }
    }
}

/// Resolve a reloc-descriptor path to this process's function address.
///
/// The descriptor stores the `jit_trace_fnaddrs` key the codewriter
/// bound (`assembler.py emit_const` carrying the symbolic object).
/// Exact match only: a suffix/`crate::` fallback can bind a different
/// function that shares a leaf.
fn runtime_fnaddr_for_reloc_path(path: &str) -> Option<i64> {
    runtime_fnaddr_by_path(path)
}

fn runtime_static_addr_by_name(name: &str) -> Option<i64> {
    static MAP: LazyLock<HashMap<&'static str, i64>> = LazyLock::new(|| {
        let mut runtime_map: HashMap<&'static str, i64> = HashMap::new();
        runtime_map.extend(pyre_interpreter::jit_static_pytype_addrs());
        runtime_map.extend(pyre_interpreter::jit_static_ref_addrs());
        runtime_map
    });
    MAP.get(name).copied()
}

/// The runtime address published for `path` in `jit_trace_fnaddrs`, or
/// `None` when the path is not published.  A walker fold that recognises a
/// residual by its callee compares the call's funcbox against this.
///
/// Tagged reloc lookup uses this same set as the old value-based
/// correspondence / disarm: `fnaddr_bindings.bin` serialises exactly
/// these names, so every bound slot has a runtime address.
pub fn runtime_fnaddr_by_path(path: &str) -> Option<i64> {
    static RUNTIME_FNADDRS: LazyLock<HashMap<&'static str, i64>> =
        LazyLock::new(|| pyre_interpreter::jit_trace_fnaddrs().into_iter().collect());
    RUNTIME_FNADDRS.get(path).copied()
}

static FNADDR_CORRESPONDENCE: LazyLock<HashMap<i64, i64>> = LazyLock::new(|| {
    let build_bindings = build_time_fnaddr_bindings();
    let runtime_bindings = pyre_interpreter::jit_trace_fnaddrs();
    let runtime_map: HashMap<&'static str, i64> = runtime_bindings.into_iter().collect();

    // `correspondence[build_fnaddr] = runtime_fnaddr` — only entries
    // whose runtime lookup actually disagrees with the build value get
    // patched; identical entries are dropped so the constants_i scan
    // can early-exit on a `HashMap::get` miss without comparing.
    //
    // The key is an address from the build process and the value one from
    // this one, so a build address can stand for two registered paths.  Both
    // ways that happens are benign, and last write wins in each:
    //
    //   * Several spellings of one function are registered on purpose
    //     (`jit_fnaddr.rs` lists both
    //     `pyre_object::listobject::jit_list_reverse` and
    //     `pyre_object::jit_list_reverse`); they resolve to the same runtime
    //     address, so the repeated insert is idempotent.
    //   * The build binary's identical-COMDAT folding (MSVC `/OPT:ICF`, or
    //     LLVM `MergeFunctions`) merges two byte-identical bodies onto one
    //     address.  With `#[inline(never)]` confined to the release-GIL
    //     surface the tracing-policy helpers inline their callees, and several
    //     distinct registered residuals then compile to the same code:
    //     `label_arg_to_usize` / `load_fast_var_num_to_index` are both
    //     `arg.get(op_arg).as_usize()`, `convert_value_arg` /
    //     `special_method_arg` are both `arg.get(op_arg)`, and
    //     `hash_str_hooked_bytes` decomposes its slice to the `(ptr, len)`
    //     `hash_str_hooked` already takes.  The build binary folds each pair;
    //     the runtime binary, built at a different optimization level, need
    //     not, so the two runtime addresses differ.  Folding merges only
    //     identical machine code, so a residual call patched to either twin
    //     runs the same body.
    //
    // A genuinely wrong runtime target could only come from a path bound to
    // the wrong `fn`, and that binding runs identically in both processes, so
    // the build and runtime addresses would agree rather than collide.  The
    // `jit_fnaddr` registry test guards against new same-address pairs but
    // observes only this process, where nothing folds, so it cannot see a
    // build-script fold.  There is therefore no collision this map must
    // reject; picking any of the folded twins' runtime addresses is correct.
    let mut correspondence: HashMap<i64, i64> = HashMap::new();
    for (path, build_fnaddr) in &build_bindings {
        let Some(&runtime_fnaddr) = runtime_map.get(path.as_str()) else {
            continue;
        };
        if *build_fnaddr != runtime_fnaddr {
            correspondence.insert(*build_fnaddr, runtime_fnaddr);
        }
    }

    correspondence
});

/// Rewrite `EffectInfo.call_release_gil_target.0` from the build-script
/// address to this process's address.
///
/// `call.py` `getcalldescr` stores `llmemory.cast_ptr_to_adr(tgt_func)`,
/// the raw funcptr, not the `ccall_*` wrapper. A translate-time miss is
/// `symbolic_fnaddr_for_path` of that funcptr. `0` is left alone.
///
/// A funcptr whose path this process does not link (a binary built without
/// the module that declares it, e.g. the core or wasm runner) maps to
/// [`call_release_gil_target_not_linked`]: its `ccall_*` wrapper is not
/// linked either, so no jitcode here can reach the call, and a call that
/// does reach it aborts.  An address with no recorded path at all is a
/// translator/runtime mismatch and panics.
pub fn rewrite_call_release_gil_target(effect_info: &mut majit_ir::EffectInfo) {
    let addr = effect_info.call_release_gil_target.0;
    if addr == 0 {
        return;
    }
    effect_info.call_release_gil_target.0 = release_gil_runtime_addr(addr);
}

fn release_gil_runtime_addr(addr: u64) -> u64 {
    let addr_i = addr as i64;
    if majit_jitcode::codewriter::call::is_symbolic_fnaddr(addr_i) {
        let path = symbolic_fnaddr_path(addr_i).unwrap_or_else(|| {
            panic!("call_release_gil_target {addr:#x} is a symbolic fnaddr with no recorded path")
        });
        return runtime_addr_for_path(path).map_or_else(not_linked_addr, |addr| addr as u64);
    }
    if let Some(&runtime) = FNADDR_CORRESPONDENCE.get(&addr_i) {
        return runtime as u64;
    }
    let bindings = build_time_fnaddr_bindings();
    let Some(path) = bindings
        .iter()
        .find(|(_, build)| *build == addr_i)
        .map(|(path, _)| path.clone())
    else {
        panic!("call_release_gil_target address {addr:#x} has no jit_trace_fnaddrs path");
    };
    runtime_addr_for_path(&path).map_or_else(not_linked_addr, |addr| addr as u64)
}

fn not_linked_addr() -> u64 {
    pyre_interpreter::residual_word_addr!(0, call_release_gil_target_not_linked) as usize as u64
}

/// Stand-in target for a `CALL_RELEASE_GIL` funcptr that this binary does
/// not link.
extern "C" fn call_release_gil_target_not_linked() {
    eprintln!("CALL_RELEASE_GIL reached a funcptr that is not linked into this binary");
    std::process::abort();
}

fn runtime_addr_for_path(path: &str) -> Option<i64> {
    if let Some(addr) = runtime_fnaddr_by_path(path) {
        return Some(addr);
    }
    // The recorded path is crate-stripped; the runtime table spells the
    // crate.  Accept the suffix match only when it names one function.
    let suffix = format!("::{path}");
    let mut matches = pyre_interpreter::jit_trace_fnaddrs()
        .into_iter()
        .filter(|(registered, _)| registered.ends_with(suffix.as_str()));
    let (first, addr) = matches.next()?;
    if let Some((second, _)) = matches.next() {
        panic!("call_release_gil_target path {path} is ambiguous: {first}, {second}");
    }
    Some(addr)
}

/// Build-time `(name, build_addr)` snapshot for the host `PyType` singleton
/// pointers the codewriter baked into `constants_i` (supplied through
/// `HostStaticAddrs.pytypes`). Same ASLR hazard + bincode round-trip as
/// [`build_time_fnaddr_bindings`].
#[cfg(test)]
fn build_time_pytype_bindings() -> Vec<(String, i64)> {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/static_pytype_bindings.bin"));
    bincode::deserialize(BYTES).unwrap_or_else(|e| {
        panic!(
            "pyre-jit-trace: failed to deserialize static_pytype_bindings.bin ({} bytes): {e}",
            BYTES.len(),
        )
    })
}

/// Build-time `(name, build_addr)` snapshot for the prebuilt ref singletons
/// (`HostStaticAddrs.refs`).
#[cfg(test)]
fn build_time_ref_bindings() -> Vec<(String, i64)> {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/static_ref_bindings.bin"));
    bincode::deserialize(BYTES).unwrap_or_else(|e| {
        panic!(
            "pyre-jit-trace: failed to deserialize static_ref_bindings.bin ({} bytes): {e}",
            BYTES.len(),
        )
    })
}

/// Rewrite stale build-time host-static addresses (`PyType` singletons and
/// prebuilt refs) the codewriter baked into the constant pools.  Mirrors
/// [`patch_constants_i_fnaddrs`] for the `HostStaticAddrs` *data* the body
/// references directly — e.g. `is_int`'s `ptr::eq((*obj).ob_type, &INT_TYPE)`
/// inlined into the `w_list_append` body, whose `&INT_TYPE` const was captured
/// in the build-script process and ASLR-invalidated at runtime.  Re-pairs each
/// named slot with the runtime address from `jit_static_pytype_addrs` /
/// `jit_static_ref_addrs`.  A host static consumed as an integer lands in
/// `constants_i` (`reloc_consts_i` StaticAddr); one used as a pointer-`eq`
/// operand materializes as a `GcRef` in `constants_r` (`reloc_consts_r`).
/// `JitCode.fnaddr` is left untouched (these are data, not call targets).
pub fn patch_static_addr_constants(jitcodes: &mut [Arc<JitCode>]) {
    disarm_unpaired_build_addrs(jitcodes);

    let has_static_reloc = jitcodes.iter().any(|jc| {
        jc.try_body().is_some_and(|body| {
            body.reloc_consts_i
                .iter()
                .any(|d| matches!(d.kind, ConstIRelocKind::StaticAddr { .. }))
                || body
                    .reloc_consts_r
                    .iter()
                    .any(|d| matches!(d.kind, ConstIRelocKind::StaticAddr { .. }))
        })
    });
    if has_static_reloc {
        for arc in jitcodes.iter_mut() {
            let jc = Arc::get_mut(arc).expect(
                "patch_static_addr_constants: Arc<JitCode> already shared before patch — \
                 every caller must run this before publishing the table to consumers",
            );
            if jc.try_body().is_some() {
                let body = jc.body_mut();
                let updates_i: Vec<(usize, i64)> = body
                    .reloc_consts_i
                    .iter()
                    .filter_map(|desc| {
                        let ConstIRelocKind::StaticAddr { name } = &desc.kind else {
                            return None;
                        };
                        runtime_static_addr_by_name(name)
                            .map(|runtime| (desc.constants_i_index, runtime))
                    })
                    .collect();
                for (index, runtime) in updates_i {
                    body.constants_i[index] = runtime;
                }
                let updates_r: Vec<(usize, i64)> = body
                    .reloc_consts_r
                    .iter()
                    .filter_map(|desc| {
                        let ConstIRelocKind::StaticAddr { name } = &desc.kind else {
                            return None;
                        };
                        runtime_static_addr_by_name(name)
                            .map(|runtime| (desc.constants_r_index, runtime))
                    })
                    .collect();
                for (index, runtime) in updates_r {
                    *body.constants_r[index].get_mut() = runtime;
                }
            }
        }
    }
}

#[cfg(test)]
static STATIC_ADDR_CORRESPONDENCE: LazyLock<HashMap<i64, i64>> = LazyLock::new(|| {
    let mut runtime_map: HashMap<&'static str, i64> = HashMap::new();
    runtime_map.extend(pyre_interpreter::jit_static_pytype_addrs());
    runtime_map.extend(pyre_interpreter::jit_static_ref_addrs());

    let mut correspondence: HashMap<i64, i64> = HashMap::new();
    for (name, build_addr) in build_time_pytype_bindings()
        .into_iter()
        .chain(build_time_ref_bindings())
    {
        if let Some(&runtime_addr) = runtime_map.get(name.as_str()) {
            if build_addr != runtime_addr {
                correspondence.insert(build_addr, runtime_addr);
            }
        }
    }

    correspondence
});

/// Build addresses whose name the runtime pool cannot re-pair, keyed by the
/// address, valued by the name it was bound under.
///
/// Both correspondences above rewrite only names present in the build pool and
/// the runtime one, so an unmatched name leaves the build process's address
/// where the codewriter put it.  The two pools come from two compilations of
/// the same list, and they answer a `cfg` differently whenever the build
/// script's copy is configured differently from the crate it feeds: the script
/// is built for the host, so a module behind `target_arch`, `unix` or
/// `windows` registers its accessors and its `#[pyre_class]` statics there and
/// not in a cross-target runtime, and it is a build-dependency with its own
/// feature resolution, so a row behind a feature the binary turns off is
/// bound here and absent there.
///
/// Binding such a name costs nothing while no jitcode carries its address, so
/// the address in a constant pool is what this names, not the binding.
///
/// An address the correspondences do know is excluded: the build binary's
/// identical-code folding can put a re-pairable name and an unmatched one on
/// one address, and the constant then belongs to the name that re-pairs.
#[cfg(test)]
static UNPAIRED_BUILD_ADDRS: LazyLock<HashMap<i64, String>> = LazyLock::new(|| {
    let mut runtime_names: std::collections::HashSet<&'static str> =
        pyre_interpreter::jit_trace_fnaddrs()
            .into_iter()
            .map(|(name, _)| name)
            .collect();
    runtime_names.extend(
        pyre_interpreter::jit_static_pytype_addrs()
            .into_iter()
            .map(|(name, _)| name),
    );
    runtime_names.extend(
        pyre_interpreter::jit_static_ref_addrs()
            .into_iter()
            .map(|(name, _)| name),
    );

    let mut paired: std::collections::HashSet<i64> = std::collections::HashSet::new();
    let mut unpaired: HashMap<i64, String> = HashMap::new();
    for (name, build_addr) in build_time_fnaddr_bindings()
        .into_iter()
        .chain(build_time_pytype_bindings())
        .chain(build_time_ref_bindings())
    {
        if runtime_names.contains(name.as_str()) {
            paired.insert(build_addr);
        } else {
            unpaired.insert(build_addr, name);
        }
    }
    unpaired.retain(|addr, _| !paired.contains(addr));
    unpaired
});

/// Take out the addresses no runtime name re-pairs, before anything reads them.
///
/// Such a value is a pointer into the build-script process, and every use the
/// pools have for one is wrong with it: as a residual call target it enters
/// whatever this process holds at that address, and as a `PyType` operand it is
/// dereferenced by a type test or compared against every object's type.
///
/// Zero is not a chosen poison, it is the spelling this already has for "no
/// address" — `jitcode.py JitCode.__init__ fnaddr=None` — and both consumers
/// test for it before they branch.  A call target goes through
/// `is_callable_fnaddr`, so the blackhole's `residual_call_*` and
/// `inline_call_*` handlers decline it and hand the continuation back to the
/// interpreter, and the walker's residual path declines the same value as
/// `ResidualDecline::Symbolic`.  A type operand is only ever compared, and a
/// comparison against zero matches no object, so the guard reading it fails
/// and the path leaves the JIT before anything can dereference it.  That is
/// what makes zero an answer here rather than a trap laid for later.
///
/// `constants_i` / `constants_r` / `jc.fnaddr` are provenance-based: only
/// slots whose recorded name is unpaired become 0, and only when the
/// descriptor recorded a build address. A tagged slot that stored a
/// symbolic hash keeps it (the old value-based scan never saw that hash
/// in the unpaired-address set). Untagged words are never touched, so an
/// ordinary integer that collides with an unpaired build address stays.
///
/// That pool already carries words that are not gcrefs — patched host
/// statics and pre-patch string sentinels — and the collector's
/// `drag_out_root` / `seed_major_root` gates reject a non-object word
/// before any deref, so a zero there is read the same way the address it
/// replaces was.
fn disarm_unpaired_build_addrs(jitcodes: &mut [Arc<JitCode>]) {
    for arc in jitcodes.iter_mut() {
        let jc = Arc::get_mut(arc).expect(
            "disarm_unpaired_build_addrs: Arc<JitCode> already shared before patch — \
             every caller must run this before publishing the table to consumers",
        );
        if let Some(ConstIRelocKind::FnAddr { path, symbolic }) = &jc.fnaddr_reloc
            && !*symbolic
            && runtime_fnaddr_by_path(path).is_none()
        {
            jc.fnaddr = 0;
        }
        if jc.try_body().is_some() {
            let body = jc.body_mut();
            let zeros_i: Vec<usize> = body
                .reloc_consts_i
                .iter()
                .filter_map(|desc| {
                    reloc_kind_is_unpaired_build(&desc.kind).then_some(desc.constants_i_index)
                })
                .collect();
            for idx in zeros_i {
                body.constants_i[idx] = 0;
            }
            let zeros_r: Vec<usize> = body
                .reloc_consts_r
                .iter()
                .filter_map(|desc| {
                    reloc_kind_is_unpaired_build(&desc.kind).then_some(desc.constants_r_index)
                })
                .collect();
            for idx in zeros_r {
                *body.constants_r[idx].get_mut() = 0;
            }
        }
    }
}

fn reloc_kind_is_unpaired_build(kind: &ConstIRelocKind) -> bool {
    match kind {
        ConstIRelocKind::FnAddr { path, symbolic } => {
            !*symbolic && runtime_fnaddr_by_path(path).is_none()
        }
        ConstIRelocKind::StaticAddr { name } => runtime_static_addr_by_name(name).is_none(),
    }
}

/// Take the build -> runtime address snapshots now.
///
/// Each map is a `LazyLock`, so without this it would be built at whatever
/// moment the first jitcode happens to be decoded.  Once jitcode bodies decode
/// lazily that moment is mid-trace, long after `init_typeobjects` has retagged
/// the `PyType` singletons — and the map, frozen there, would then patch every
/// later-decoded body's `constants_r` with addresses that no longer name what
/// the runtime holds.  Those constants reach a compiled loop's gc_table, so the
/// collector drags a bad root out of it (`GC BUG: invalid type_id`, site
/// `minor_root_target`).
///
/// Calling this from `ensure_finish_setup` restores the snapshot instant the
/// whole-table loader had, when the first `all_jitcodes()` decoded everything
/// against one consistent view.
pub fn prime_address_correspondences() {
    // `FNADDR_CORRESPONDENCE` still feeds `release_gil_runtime_addr` and
    // the lazy indirect-call-target index (`runtime_fnaddr`); force it
    // at the same instant the whole-table loader used to.
    LazyLock::force(&FNADDR_CORRESPONDENCE);
    LazyLock::force(&TYPE_STATIC_RUNTIME_MAP);
}

/// Name → runtime address of every host `PyType` static.
///
/// The same three publishers [`materialize_type_static_consts`] consults.
/// A `new_with_vtable` size descr stores one of these names beside a
/// type-static sentinel; [`rebind_type_static_size_vtable`] reads the
/// address back out of this map.
pub fn type_static_runtime_map() -> &'static HashMap<&'static str, i64> {
    &TYPE_STATIC_RUNTIME_MAP
}

static TYPE_STATIC_RUNTIME_MAP: LazyLock<HashMap<&'static str, i64>> = LazyLock::new(|| {
    let mut runtime_map: HashMap<&'static str, i64> = HashMap::new();
    runtime_map.extend(pyre_interpreter::jit_static_pytype_addrs());
    runtime_map.extend(pyre_interpreter::pyre_class_pytype_addrs());
    runtime_map.extend(pyre_interpreter::pyre_class_pytype_by_struct_addrs());
    runtime_map
});

/// Replace a type-static sentinel in `BhDescr::Size.vtable` with the
/// runtime address of the name stored in `owner`, then clear `owner`.
///
/// A descr whose vtable is already an address is left untouched, including
/// its `owner` (`STRUCT._name` or the headerless marker).
pub fn rebind_type_static_size_vtable(descr: &mut majit_jitcode::jitcode::BhDescr) {
    use majit_jitcode::codewriter::assembler::is_type_static_const_sentinel;
    use majit_jitcode::jitcode::BhDescr;
    let BhDescr::Size { vtable, owner, .. } = descr else {
        return;
    };
    if !is_type_static_const_sentinel(*vtable) {
        return;
    }
    let addr = *type_static_runtime_map()
        .get(owner.as_str())
        .unwrap_or_else(|| {
            panic!("type-static size descr '{owner}' has no published runtime address")
        });
    *vtable = addr as u64;
    owner.clear();
}

/// Whether [`patch_constants_i_fnaddrs`] rewrites `fnaddr`.
///
/// The correspondence maps a build-process address onto this process's
/// address, and only when the two differ. A symbolic residual constant that
/// is a key was replaced before any wrapper observed the body.
pub fn is_bound_symbolic_fnaddr(fnaddr: i64) -> bool {
    FNADDR_CORRESPONDENCE.contains_key(&fnaddr)
}

/// Resolve one build-process function address to this process's address.
/// Used by the lazy indirect-call-target index, which needs the shell's
/// `fnaddr` but deliberately does not deserialize the shell's JitCode body.
pub(crate) fn runtime_fnaddr(build_fnaddr: i64) -> i64 {
    FNADDR_CORRESPONDENCE
        .get(&build_fnaddr)
        .copied()
        .unwrap_or(build_fnaddr)
}

/// High 16 bits of a deferred prebuilt-string sentinel (see
/// [`majit_jitcode::codewriter::assembler::STR_CONST_SENTINEL_BASE`]).  x86-64 user
/// addresses occupy `0..2^48`, so a real GCREF / host-static address always
/// has these bits clear, while every sentinel has them set to the base
/// pattern.
const SENTINEL_HIGH_MASK: u64 = 0xFFFF_0000_0000_0000;

/// Materialize one immortal string constant.
///
/// [`StrConstKind::UnicodeObject`] is the `box_str_constant` result — a
/// `W_UnicodeObject` the residual `w_name` ABI passes as one Ref.
/// [`StrConstKind::Str`] is rstr `Ptr(STR)` (`StringRepr.convert_const`):
/// `bh_strlen` / `bh_strgetitem` read that payload.
/// [`StrConstKind::Unicode`] is rstr `Ptr(UNICODE)` (`UnicodeRepr.lowleveltype`):
/// `unicodelen` / `unicodegetitem` / `ll_find_char_unicode` read UCS-4 cells
/// through `unicode_as_unichars`.
fn materialize_prebuilt_str(desc: &StrConstDescriptor) -> i64 {
    match desc.kind {
        StrConstKind::Unicode => materialize_prebuilt_unicode(&desc.bytes, desc.precomputed_hash),
        StrConstKind::Str => {
            // `BaseLLStringRepr.convert_const` copies each `Char` into STR
            // storage. Without `value._obj.container` identity the block is
            // the `CONST_STR_CACHE` entry for these chars
            // ([`pyre_object::unicodeobject::prebuilt_str_storage_bytes`]);
            // a descriptor with identity gets its own block. Either way the
            // hash is `ll_strhash` over the chars.
            if desc.container_index.is_none() {
                pyre_object::unicodeobject::prebuilt_str_storage_bytes(&desc.bytes) as i64
            } else {
                let payload = pyre_object::unicodeobject::alloc_utf8_payload(&desc.bytes, false);
                // `p.hash = 0; self.ll.ll_strhash(p)`: the payload comes back with a
                // zero hash and `ll_strhash` precomputes it over the chars.
                let _ = pyre_object::lowlevel_string::ll_strhash(payload);
                payload as i64
            }
        }
        StrConstKind::UnicodeObject => {
            let wtf8 = rustpython_wtf8::Wtf8::from_bytes(&desc.bytes)
                .expect("prebuilt UnicodeObject constant bytes are not valid WTF-8");
            if desc.container_index.is_some() {
                pyre_object::unicodeobject::w_str_from_wtf8_immortal(wtf8.to_owned()) as i64
            } else {
                pyre_object::unicodeobject::box_str_constant(wtf8) as i64
            }
        }
    }
}

/// Immortal rstr `Ptr(UNICODE)` (`bh_newunicode` / `unicode_as_unichars`
/// layout: `hash` @0, `length` @one word, UCS-4 `chars` after the header).
/// Uses `alloc_varsize_with_gc_header_immortal`, the same prebuilt route
/// as STR (`alloc_raw_utf8_payload`) and empty-class instances.
fn materialize_prebuilt_unicode(bytes: &[u8], precomputed_hash: i64) -> i64 {
    let cells = unicode_const_cells(bytes);
    let n = cells.len();
    let items = n
        .checked_mul(std::mem::size_of::<u32>())
        .expect("Ptr(UNICODE) chars overflow");
    let total = pyre_object::lowlevel_string::LOWLEVEL_UNICODE_BASE_SIZE
        .checked_add(items)
        .expect("Ptr(UNICODE) block overflow");
    let tid = pyre_object::lowlevel_string::lowlevel_unicode_gc_type_id();
    let ptr = majit_gc::header::alloc_varsize_with_gc_header_immortal(total, tid);
    assert!(
        !ptr.is_null(),
        "materialize_prebuilt_unicode: immortal allocation failed"
    );
    unsafe {
        ptr.cast::<isize>().write(precomputed_hash as isize);
        ptr.add(pyre_object::lowlevel_string::LOWLEVEL_STRING_LEN_OFFSET)
            .cast::<usize>()
            .write(n);
        std::ptr::copy_nonoverlapping(
            cells.as_ptr(),
            ptr.add(pyre_object::lowlevel_string::LOWLEVEL_STRING_CHARS_OFFSET)
                .cast::<u32>(),
            n,
        );
    }
    ptr as i64
}

/// Intern table for one assembly's jitcode set (`assembler.py`
/// `constants_dict` / `constants_r` retain the container; the runtime
/// maps that assembly's dense ordinal onto the materialized block).
/// Owned by the loader of one jitcode table, not process-wide: two
/// independent assemblies may reuse ordinal 0 for different contents.
#[derive(Default)]
pub struct AssemblyConstIntern {
    str_by_index: Vec<Option<i64>>,
    instance_by_index: Vec<Option<i64>>,
}

impl AssemblyConstIntern {
    pub fn new() -> Self {
        Self::default()
    }

    fn str_addr(&self, index: u32) -> Option<i64> {
        self.str_by_index.get(index as usize).copied().flatten()
    }

    fn insert_str(&mut self, index: u32, addr: i64) -> i64 {
        let i = index as usize;
        if self.str_by_index.len() <= i {
            self.str_by_index.resize(i + 1, None);
        }
        if let Some(existing) = self.str_by_index[i] {
            existing
        } else {
            self.str_by_index[i] = Some(addr);
            addr
        }
    }

    fn instance_addr(&self, index: u32) -> Option<i64> {
        self.instance_by_index
            .get(index as usize)
            .copied()
            .flatten()
    }

    fn insert_instance(&mut self, index: u32, addr: i64) -> i64 {
        let i = index as usize;
        if self.instance_by_index.len() <= i {
            self.instance_by_index.resize(i + 1, None);
        }
        if let Some(existing) = self.instance_by_index[i] {
            existing
        } else {
            self.instance_by_index[i] = Some(addr);
            addr
        }
    }
}

/// Bytes-only intern key (`ConstInternedStr`): share by `(bytes, kind)`,
/// including the interpreter's interned literal storage.
#[derive(Hash, Eq, PartialEq)]
struct StrConstBytesKey {
    bytes: Vec<u8>,
    kind: StrConstKind,
}

/// Materialize every deferred prebuilt-string constant the codewriter
/// recorded (`JitCodeBody.str_consts`, [`patch_constants_i_fnaddrs`]'s
/// sibling).  The build-time translator could not allocate a runtime STR
/// block, so it pooled a non-canonical sentinel in the slot named by each
/// descriptor's `constants_r_index`; here we allocate the immortal block and
/// overwrite the sentinel with its live address. Runs before the per-index
/// `OnceCell` publishes the entry — refcount must be 1 (`Arc::get_mut`), so no
/// consumer can observe the sentinel as a forged GCREF.
///
/// Identical literals share one immortal block across the whole table, the
/// runtime analog of the assembler's per-jitcode dedup. `StrConstKind::Str`
/// without identity is `CONST_STR_CACHE` (`BaseLLStringRepr.convert_const`);
/// `StrConstKind::UnicodeObject` without identity is `box_str_constant`'s
/// process-wide `WEAK_INTERN`. Container-backed descriptors intern through
/// `intern` (one table per assembly); bytes-only descriptors keep
/// process-wide `(bytes, kind)` sharing.
pub fn materialize_str_consts(jitcodes: &mut [Arc<JitCode>]) {
    let mut intern = AssemblyConstIntern::new();
    materialize_str_consts_in(jitcodes, &mut intern);
}

pub fn materialize_str_consts_in(jitcodes: &mut [Arc<JitCode>], intern: &mut AssemblyConstIntern) {
    static BYTES_INTERNED: LazyLock<std::sync::Mutex<HashMap<StrConstBytesKey, i64>>> =
        LazyLock::new(|| std::sync::Mutex::new(HashMap::new()));
    for arc in jitcodes.iter_mut() {
        // Body-less placeholder shells, and bodies with no deferred strings
        // (the common case — only cutover string literals record any), need
        // no work and must not trip `Arc::get_mut` for nothing.
        if arc.try_body().is_none_or(|b| b.str_consts.is_empty()) {
            continue;
        }
        let jc = Arc::get_mut(arc).expect(
            "materialize_str_consts: Arc<JitCode> already shared before patch — \
             every caller must run this before publishing the table to consumers",
        );
        let body = jc.body_mut();
        for i in 0..body.str_consts.len() {
            let idx = body.str_consts[i].constants_r_index;
            let addr = {
                let desc = &body.str_consts[i];
                if let Some(container_index) = desc.container_index {
                    if let Some(a) = intern.str_addr(container_index) {
                        a
                    } else {
                        let a = materialize_prebuilt_str(desc);
                        intern.insert_str(container_index, a)
                    }
                } else {
                    let key = StrConstBytesKey {
                        bytes: desc.bytes.clone(),
                        kind: desc.kind,
                    };
                    let mut interned = BYTES_INTERNED.lock().unwrap();
                    if let Some(&a) = interned.get(&key) {
                        a
                    } else {
                        let a = materialize_prebuilt_str(desc);
                        interned.insert(key, a);
                        a
                    }
                }
            };
            // The slot must still hold its non-canonical sentinel — never a
            // real address (which has the high bits clear).
            assert_eq!(
                (body.constants_r[idx].get() as u64) & SENTINEL_HIGH_MASK,
                (majit_jitcode::codewriter::assembler::STR_CONST_SENTINEL_BASE as u64)
                    & SENTINEL_HIGH_MASK,
                "constants_r[{idx}] did not hold a prebuilt-string sentinel",
            );
            body.constants_r[idx] = addr.into();
        }
    }
}

/// Materialize every deferred unit-variant singleton constant
/// ([`materialize_str_consts`]' sibling).  Each descriptor names a
/// `constants_r` slot holding a non-canonical sentinel; the cell minted
/// here is one immortal leaked `i64` holding the variant's declaration
/// index — the entire layout of a payload-less variant class is its
/// `__discriminant` word at offset 0, so `getfield_gc_i(_pure)` reads
/// on the singleton resolve against real memory.  The cell never enters
/// the GC heap: `walk_jitcode_constants_refs`' visitor relocates only
/// nursery object starts, so the address passes through every
/// collection unchanged.  One cell per qualname process-wide, mirroring
/// the build-side interner's identity sharing.
pub fn materialize_unit_variant_consts(jitcodes: &mut [Arc<JitCode>]) {
    static CELLS: LazyLock<std::sync::Mutex<Vec<(String, i64)>>> =
        LazyLock::new(|| std::sync::Mutex::new(Vec::new()));
    for arc in jitcodes.iter_mut() {
        if arc
            .try_body()
            .is_none_or(|b| b.unit_variant_consts.is_empty())
        {
            continue;
        }
        let jc = Arc::get_mut(arc).expect(
            "materialize_unit_variant_consts: Arc<JitCode> already shared before patch — \
             every caller must run this before publishing the table to consumers",
        );
        let body = jc.body_mut();
        for i in 0..body.unit_variant_consts.len() {
            let idx = body.unit_variant_consts[i].constants_r_index;
            let qualname = body.unit_variant_consts[i].qualname.clone();
            let tag = body.unit_variant_consts[i].tag;
            let addr = {
                let mut cells = CELLS.lock().unwrap();
                if let Some((_, a)) = cells.iter().find(|(q, _)| *q == qualname) {
                    // The cell holds the tag the first jitcode carried for
                    // this qualname.  Two bodies assembled from different
                    // graphs must agree on it: a silent disagreement would
                    // share one cell and answer the wrong discriminant.
                    debug_assert_eq!(
                        unsafe { *(*a as *const i64) },
                        tag,
                        "unit-variant {qualname} carries two tags",
                    );
                    *a
                } else {
                    let a = Box::leak(Box::new(tag)) as *const i64 as i64;
                    cells.push((qualname, a));
                    a
                }
            };
            // The slot must still hold its non-canonical sentinel — never
            // a real address (which has the high bits clear).
            assert_eq!(
                (body.constants_r[idx].get() as u64) & SENTINEL_HIGH_MASK,
                (majit_jitcode::codewriter::assembler::UNIT_VARIANT_CONST_SENTINEL_BASE as u64)
                    & SENTINEL_HIGH_MASK,
                "constants_r[{idx}] did not hold a unit-variant sentinel",
            );
            body.constants_r[idx] = addr.into();
        }
    }
}

/// Materialize every deferred prebuilt exception instance.
///
/// Each descriptor names a class and an optional message. The load pass
/// allocates one immortal instance and overwrites the sentinel. Identical
/// `(class, message)` pairs share one object.
pub fn materialize_exc_instance_consts(jitcodes: &mut [Arc<JitCode>]) {
    for arc in jitcodes.iter_mut() {
        if arc
            .try_body()
            .is_none_or(|b| b.exc_instance_consts.is_empty())
        {
            continue;
        }
        let jc = Arc::get_mut(arc).expect(
            "materialize_exc_instance_consts: Arc<JitCode> already shared before patch — \
             every caller must run this before publishing the table to consumers",
        );
        let body = jc.body_mut();
        for i in 0..body.exc_instance_consts.len() {
            let idx = body.exc_instance_consts[i].constants_r_index;
            let class_name = body.exc_instance_consts[i].class_name.clone();
            let message = body.exc_instance_consts[i].message.clone();
            let kind = pyre_object::interp_exceptions::exc_kind_from_name(&class_name)
                .unwrap_or_else(|| {
                    panic!("prebuilt exception instance has no ExcKind for class {class_name}")
                });
            let addr = match &message {
                None => pyre_object::interp_exceptions::standard_exc_instance(kind) as usize as i64,
                Some(bytes) => {
                    let text = std::str::from_utf8(bytes)
                        .unwrap_or_else(|_| panic!("prebuilt {class_name} message is not UTF-8"));
                    pyre_object::interp_exceptions::prebuilt_exception_with_message(kind, text)
                        as usize as i64
                }
            };
            assert_eq!(
                (body.constants_r[idx].get() as u64) & SENTINEL_HIGH_MASK,
                (majit_jitcode::codewriter::assembler::EXC_INSTANCE_CONST_SENTINEL_BASE as u64)
                    & SENTINEL_HIGH_MASK,
                "constants_r[{idx}] did not hold an exception-instance sentinel",
            );
            body.constants_r[idx] = addr.into();
        }
    }
}

/// Materialize every deferred type-static constant
/// ([`materialize_str_consts`]' sibling for `PyType` singletons).
/// Each descriptor names a `constants_r` slot holding a non-canonical
/// sentinel; overwrite it with the live `&INT_TYPE` (etc.) from
/// `jit_static_pytype_addrs`, keyed by the shared name.
pub fn materialize_type_static_consts(jitcodes: &mut [Arc<JitCode>]) {
    let runtime_map = type_static_runtime_map();

    for arc in jitcodes.iter_mut() {
        if arc
            .try_body()
            .is_none_or(|b| b.type_static_consts.is_empty())
        {
            continue;
        }
        let jc = Arc::get_mut(arc).expect(
            "materialize_type_static_consts: Arc<JitCode> already shared before patch — \
             every caller must run this before publishing the table to consumers",
        );
        let body = jc.body_mut();
        for i in 0..body.type_static_consts.len() {
            let idx = body.type_static_consts[i].constants_r_index;
            let name = body.type_static_consts[i].name.as_str();
            assert_eq!(
                (body.constants_r[idx].get() as u64) & SENTINEL_HIGH_MASK,
                (majit_jitcode::codewriter::assembler::TYPE_STATIC_CONST_SENTINEL_BASE as u64)
                    & SENTINEL_HIGH_MASK,
                "constants_r[{idx}] did not hold a type-static sentinel",
            );
            let addr = *runtime_map.get(name).unwrap_or_else(|| {
                panic!("type-static constant {name} has no published runtime address")
            });
            body.constants_r[idx] = addr.into();
        }
    }
}

/// Materialize every deferred zero-length array constant
/// ([`materialize_str_consts`]' sibling for `_ll_prebuilt_empty_array`).
/// Each descriptor names a `constants_r` slot holding a non-canonical
/// sentinel; overwrite it with one immortal empty `GcArray` per item type.
pub fn materialize_empty_array_consts(jitcodes: &mut [Arc<JitCode>]) {
    static CELLS: LazyLock<std::sync::Mutex<Vec<(String, i64)>>> =
        LazyLock::new(|| std::sync::Mutex::new(Vec::new()));
    for arc in jitcodes.iter_mut() {
        if arc
            .try_body()
            .is_none_or(|b| b.empty_array_consts.is_empty())
        {
            continue;
        }
        let jc = Arc::get_mut(arc).expect(
            "materialize_empty_array_consts: Arc<JitCode> already shared before patch — \
             every caller must run this before publishing the table to consumers",
        );
        let body = jc.body_mut();
        for i in 0..body.empty_array_consts.len() {
            let idx = body.empty_array_consts[i].constants_r_index;
            let type_name = body.empty_array_consts[i].type_name.clone();
            let addr = {
                let mut cells = CELLS.lock().unwrap();
                if let Some((_, a)) = cells.iter().find(|(q, _)| *q == type_name) {
                    *a
                } else {
                    let a = prebuilt_empty_array_addr(&type_name);
                    cells.push((type_name, a));
                    a
                }
            };
            assert_eq!(
                (body.constants_r[idx].get() as u64) & SENTINEL_HIGH_MASK,
                (majit_jitcode::codewriter::assembler::EMPTY_ARRAY_CONST_SENTINEL_BASE as u64)
                    & SENTINEL_HIGH_MASK,
                "constants_r[{idx}] did not hold an empty-array sentinel",
            );
            body.constants_r[idx] = addr.into();
        }
    }
}

/// Materialize every deferred prebuilt empty-class instance
/// ([`materialize_str_consts`]' sibling for `InstanceRepr.convert_const` /
/// `convert_const_exact`). Each descriptor names a `constants_r` slot
/// holding a non-canonical sentinel; overwrite it with the immortal
/// instance `create_instance` would malloc. Intern one instance per
/// assembly ordinal (`iprebuiltinstances[value]`, `emit_const`'s
/// `value._obj.container`).
pub fn materialize_prebuilt_instance_consts(jitcodes: &mut [Arc<JitCode>]) {
    let mut intern = AssemblyConstIntern::new();
    materialize_prebuilt_instance_consts_in(jitcodes, &mut intern);
}

pub fn materialize_prebuilt_instance_consts_in(
    jitcodes: &mut [Arc<JitCode>],
    intern: &mut AssemblyConstIntern,
) {
    for arc in jitcodes.iter_mut() {
        if arc
            .try_body()
            .is_none_or(|b| b.prebuilt_instance_consts.is_empty())
        {
            continue;
        }
        let jc = Arc::get_mut(arc).expect(
            "materialize_prebuilt_instance_consts: Arc<JitCode> already shared before patch — \
             every caller must run this before publishing the table to consumers",
        );
        let body = jc.body_mut();
        for i in 0..body.prebuilt_instance_consts.len() {
            let desc = body.prebuilt_instance_consts[i].clone();
            let idx = desc.constants_r_index;
            let container_index = desc.container_index;
            let addr = if let Some(a) = intern.instance_addr(container_index) {
                a
            } else {
                let a = prebuilt_empty_class_instance_addr(&desc);
                intern.insert_instance(container_index, a)
            };
            assert_eq!(
                (body.constants_r[idx].get() as u64) & SENTINEL_HIGH_MASK,
                (majit_jitcode::codewriter::assembler::PREBUILT_INSTANCE_CONST_SENTINEL_BASE
                    as u64)
                    & SENTINEL_HIGH_MASK,
                "constants_r[{idx}] did not hold a prebuilt-instance sentinel",
            );
            body.constants_r[idx] = addr.into();
        }
    }
}

/// Immortal rclass instance. `InstanceRepr.create_instance` mallocs
/// `self.object_type` immortal with `flavor=self.gcflavor`;
/// `initialize_prebuilt_instance` writes `ClassRepr.getvtable` into
/// typeptr when the STRUCT has one.
///
/// `TypeLayoutBuilder.consider_constant` installs a GC header and type
/// id only for GC containers, and only when a framework GC is present
/// (`database.py getcontainernode` calls it only if the transformer
/// defines it; `BoehmGCTransformer` / `NoneGcPolicy` do not). A
/// GC-managed non-headerless instance therefore takes
/// `alloc_varsize_with_gc_header_immortal_zeroed` only while
/// `gc_sync::is_initialized()`. With no collector, `bh_alloc_struct`'s
/// no-collector tail is a zeroed malloc with no header and no tid.
///
/// `rewrite_op_malloc` is `new_with_vtable` iff `has_gcstruct_a_vtable`;
/// `get_vtable_for_gcstruct` only reads `_cache_gcstruct2vtable` (filled
/// by `publish_pyre_class_vtables`) and never mints a vtable. A typeptr
/// header with no registered row is a registration gap.
fn prebuilt_empty_class_instance_addr(
    desc: &majit_jitcode::jitcode::PrebuiltInstanceConstDescriptor,
) -> i64 {
    let vtable = prebuilt_instance_vtable(desc);
    let ptr = if desc.is_gc_managed && !desc.headerless && majit_gc::gc_sync::is_initialized() {
        let tid = prebuilt_instance_tid(desc);
        majit_gc::header::alloc_varsize_with_gc_header_immortal_zeroed(desc.size, tid) as usize
    } else {
        let layout =
            std::alloc::Layout::from_size_align(desc.size.max(1), std::mem::align_of::<usize>())
                .expect("prebuilt raw instance layout");
        let raw = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!raw.is_null(), "prebuilt instance allocation failed");
        raw as usize
    };
    assert!(ptr != 0, "prebuilt instance allocation failed");
    if let Some(vtable) = vtable {
        assert!(
            desc.size >= std::mem::size_of::<usize>(),
            "prebuilt instance typeptr header needs a word, size={}",
            desc.size
        );
        unsafe {
            *(ptr as *mut usize) = vtable;
        }
    }
    ptr as i64
}

/// Size descr `OpKind::New` interns. Empty-class `all_fielddescrs` is
/// empty; vtable is 0. The typeptr word is `ClassRepr.getvtable` from
/// `_cache_gcstruct2vtable`, never this field.
///
/// `is_gc_managed` is `InstanceRepr.create_instance`'s `self.gcflavor`.
/// `TypeLayoutBuilder.consider_constant` headers GC containers only;
/// a headerless Size stamps `HEADERLESS_SIZE_OWNER_MARKER` in `owner`.
fn prebuilt_instance_size_descr(
    desc: &majit_jitcode::jitcode::PrebuiltInstanceConstDescriptor,
) -> majit_jitcode::jitcode::BhDescr {
    majit_jitcode::jitcode::BhDescr::Size {
        size: desc.size,
        type_id: desc.type_id,
        vtable: 0,
        owner: if desc.headerless {
            majit_jitcode::jitcode::HEADERLESS_SIZE_OWNER_MARKER.to_string()
        } else {
            String::new()
        },
        all_fielddescrs: Vec::new(),
        is_gc_managed: desc.is_gc_managed,
    }
}

/// Typeptr-header: the `_cache_gcstruct2vtable` row
/// `heaptracker.get_vtable_for_gcstruct` reads. Never minted.
fn prebuilt_instance_vtable(
    desc: &majit_jitcode::jitcode::PrebuiltInstanceConstDescriptor,
) -> Option<usize> {
    if desc.has_vtable {
        crate::descr::publish_pyre_class_vtables();
        let found = {
            let gc = majit_ir::descr::gc_cache().lock();
            gc._cache_gcstruct2vtable.get(&desc.type_id).copied()
        };
        match found {
            Some((vt, _)) if vt != 0 => Some(vt),
            _ => panic!(
                "prebuilt instance: typeptr-header struct type_id={:#x} has no registered vtable \
                 (heaptracker.get_vtable_for_gcstruct reads _cache_gcstruct2vtable; it never mints one)",
                desc.type_id
            ),
        }
    } else {
        None
    }
}

/// Collector tid for a GC-container prebuilt. Reached only while
/// `gc_sync::is_initialized()` (`prebuilt_empty_class_instance_addr`).
/// Header tid is `SizeDescr.type_id` on the descr `new` interns
/// (`descr.py get_size_descr` via `make_descr_from_bh` of this STRUCT's
/// Size). `TypeLayoutBuilder.consider_constant` interned that Size into
/// kind-0 so `GcLLDescr_framework.init_size_descr` already stamped it.
fn prebuilt_instance_tid(desc: &majit_jitcode::jitcode::PrebuiltInstanceConstDescriptor) -> u32 {
    if let Some((_, cached_size)) = majit_ir::descr::gc_cache()
        .lock()
        .resolve_struct_layout(desc.type_id)
        && cached_size != desc.size
    {
        panic!(
            "prebuilt instance: BhDescr GC identity resolves to a foreign allocation layout \
             (descr.py get_size_descr is one SizeDescr per STRUCT)"
        );
    }
    // `GCTransformer.finish_tables` stamps every TYPE
    // `TypeLayoutBuilder.consider_constant` registered before any
    // code uses a prebuilt constant. Decode kind-0 and
    // `register_synthetic_struct_tids` are that happens-before.
    crate::jitcode_runtime::materialize_gccache_owned_descrs();
    let descr = crate::descr::make_descr_from_bh(&prebuilt_instance_size_descr(desc));
    let (cache_key, tid) = {
        let sd = descr.as_size_descr().expect("new of the same Size descr");
        (sd.cache_key(), sd.type_id())
    };
    // This function is only reached with a collector, so
    // `GCTransformer.finish_tables` / `TypeLayoutBuilder.consider_constant`
    // must already have stamped the STRUCT. An unresolved tid is a
    // happens-before violation.
    if majit_ir::descr::struct_tid_is_unresolved(cache_key, tid) || (cache_key != 0 && tid == 0) {
        panic!(
            "prebuilt instance: SizeDescr collector tid for cache_key={cache_key:#x} is unresolved \
             after TypeLayoutBuilder.consider_constant / kind-0"
        );
    }
    tid
}

/// One immortal empty `GcArray` for `type_name` (`Array<T;0>`).
fn prebuilt_empty_array_addr(type_name: &str) -> i64 {
    let item = empty_array_item(type_name)
        .unwrap_or_else(|| panic!("empty-array constant {type_name} is not Array<T;0>"));
    if item_is_object_pointer(item) {
        pyre_object::prebuilt_empty_object_array() as i64
    } else {
        // Header-only empty GcArray (`prepare_const(0)`), including
        // `Array<u8;0>` (`_ll_prebuilt_empty_array` of `GcArray(Char)`, not
        // a `STR`). Length is 0, so the item stride is unused; one block
        // per `type_name` via the caller cache.
        unsafe { pyre_object::alloc_mro_block_gc(&[]) as i64 }
    }
}

fn empty_array_item(type_name: &str) -> Option<&str> {
    let inner = type_name.strip_prefix("Array<")?.strip_suffix('>')?;
    let (item, len) = inner.rsplit_once(';')?;
    (len.trim() == "0").then_some(item.trim())
}

fn item_is_object_pointer(item: &str) -> bool {
    let item = item.trim();
    if item == "PyObjectRef" || item.ends_with("::PyObjectRef") {
        return true;
    }
    let Some(pointee) = item
        .strip_prefix("*mut ")
        .or_else(|| item.strip_prefix("*const "))
    else {
        return false;
    };
    let pointee = pointee.trim();
    pointee == "PyObject" || pointee.ends_with("::PyObject")
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_jitcode::codewriter::assembler::STR_CONST_SENTINEL_BASE;
    use majit_jitcode::jitcode::{
        ConstIRelocKind, JitCode, JitCodeBody, StrConstDescriptor, StrConstKind,
        unicode_const_bytes,
    };

    fn sentinel(ordinal: i64) -> i64 {
        STR_CONST_SENTINEL_BASE | ordinal
    }

    /// Build a fresh `Arc<JitCode>` whose body carries `descs` plus a
    /// `constants_r` pre-seeded with the matching sentinels, mirroring the
    /// assembler's emit (ordinal == position in `str_consts`).
    fn jitcode_with_str_consts(descs: Vec<StrConstDescriptor>) -> Arc<JitCode> {
        let len = descs
            .iter()
            .map(|d| d.constants_r_index + 1)
            .max()
            .unwrap_or(0);
        let mut constants_r = vec![0_i64; len];
        for (ordinal, d) in descs.iter().enumerate() {
            constants_r[d.constants_r_index] = sentinel(ordinal as i64);
        }
        let jc = JitCode::new("test");
        jc.set_body(JitCodeBody {
            str_consts: descs,
            constants_r: constants_r.into_iter().map(Into::into).collect(),
            ..Default::default()
        });
        Arc::new(jc)
    }

    #[test]
    fn patch_static_addr_constants_keeps_sentinels_and_runtime_statics() {
        let known = pyre_interpreter::jit_static_ref_addrs()
            .into_iter()
            .find(|(_, addr)| *addr != 0)
            .map(|(_, addr)| addr)
            .expect("jit_static_ref_addrs");
        let exc = majit_jitcode::codewriter::assembler::EXC_INSTANCE_CONST_SENTINEL_BASE | 2;
        let jc = JitCode::new("listed-ref");
        jc.set_body(JitCodeBody {
            constants_r: vec![sentinel(1).into(), known.into(), exc.into()],
            ..Default::default()
        });
        let mut jcs = vec![Arc::new(jc)];
        patch_static_addr_constants(&mut jcs);
        let body = jcs[0].body();
        assert_eq!(body.constants_r[0].get(), sentinel(1));
        assert_eq!(body.constants_r[1].get(), known);
        assert_eq!(body.constants_r[2].get(), exc);
    }

    #[test]
    fn materialize_str_consts_overwrites_sentinel_with_str_object() {
        use majit_ir::GcRef;
        use majit_metainterp::cpu::Cpu;

        let descs = vec![StrConstDescriptor {
            constants_r_index: 0,
            bytes: b"hello".to_vec(),
            precomputed_hash: 0x1234_5678,
            kind: StrConstKind::Str,
            container_index: None,
        }];
        let mut jcs = vec![jitcode_with_str_consts(descs)];
        materialize_str_consts(&mut jcs);

        let addr = jcs[0].body().constants_r[0].get();
        assert_ne!(addr, sentinel(0), "sentinel must be overwritten");
        assert_eq!(
            (addr as u64) & SENTINEL_HIGH_MASK,
            0,
            "a real STR payload address must have the sentinel high bits clear",
        );
        // The slot holds `_utf8` (`StringRepr.convert_const` → `Ptr(STR)`).
        let cpu = crate::pyre_cpu::PyreCpu::new();
        assert_eq!(cpu.bh_strlen(GcRef(addr as usize)), Some(5));
        let got: Vec<u8> = (0..5)
            .map(|i| cpu.bh_strgetitem(GcRef(addr as usize), i).unwrap() as u8)
            .collect();
        assert_eq!(got, b"hello");
    }

    /// `BaseLLStringRepr.convert_const` stores arbitrary `Char` cells; a
    /// lone `0xff` is a valid STR payload with or without a container id, and
    /// its hash is `ll_strhash` over those chars.
    #[test]
    fn materialize_str_consts_str_kind_preserves_arbitrary_char_bytes() {
        use majit_ir::GcRef;
        use majit_metainterp::cpu::Cpu;

        let bytes = vec![b'a', 0xff, b'b'];
        let hash = 0x55aa_i64;
        let expected_hash = {
            let p = pyre_object::unicodeobject::alloc_utf8_payload(&bytes, false);
            pyre_object::lowlevel_string::ll_strhash(p) as i64
        };
        for container_index in [None, Some(0_u32)] {
            let descs = vec![StrConstDescriptor {
                constants_r_index: 0,
                bytes: bytes.clone(),
                precomputed_hash: hash,
                kind: StrConstKind::Str,
                container_index,
            }];
            let mut jcs = vec![jitcode_with_str_consts(descs)];
            materialize_str_consts(&mut jcs);
            let addr = jcs[0].body().constants_r[0].get();
            let cpu = crate::pyre_cpu::PyreCpu::new();
            assert_eq!(cpu.bh_strlen(GcRef(addr as usize)), Some(3));
            let got: Vec<u8> = (0..3)
                .map(|i| cpu.bh_strgetitem(GcRef(addr as usize), i).unwrap() as u8)
                .collect();
            assert_eq!(got, bytes);
            let got_hash = unsafe { *(addr as *const isize) } as i64;
            assert_eq!(
                got_hash, expected_hash,
                "STR constant hash is ll_strhash over its chars"
            );
        }
    }

    #[test]
    fn materialize_str_consts_interns_identical_bytes_across_jitcodes() {
        let desc = || StrConstDescriptor {
            constants_r_index: 0,
            bytes: b"x".to_vec(),
            precomputed_hash: 7,
            kind: StrConstKind::Str,
            container_index: None,
        };
        let mut jcs = vec![
            jitcode_with_str_consts(vec![desc()]),
            jitcode_with_str_consts(vec![desc()]),
        ];
        materialize_str_consts(&mut jcs);
        let a0 = jcs[0].body().constants_r[0].get();
        let a1 = jcs[1].body().constants_r[0].get();
        assert_eq!(
            a0, a1,
            "identical literals must share one immortal STR payload",
        );
        assert_ne!(a0, sentinel(0));
    }

    /// The lazy per-index loader materializes one entry per call, so the
    /// call-local `interned` map cannot be what shares the block between two
    /// jitcodes — `CONST_STR_CACHE` (`BaseLLStringRepr.convert_const`) is.
    /// Pins that the one-block identity survives the split into separate calls.
    #[test]
    fn materialize_str_consts_interns_across_separate_calls() {
        std::thread::spawn(|| {
            let desc = || StrConstDescriptor {
                constants_r_index: 0,
                bytes: b"y".to_vec(),
                precomputed_hash: 9,
                kind: StrConstKind::Str,
                container_index: None,
            };
            let mut first = vec![jitcode_with_str_consts(vec![desc()])];
            let mut second = vec![jitcode_with_str_consts(vec![desc()])];
            materialize_str_consts(&mut first);
            materialize_str_consts(&mut second);
            assert_eq!(
                first[0].body().constants_r[0],
                second[0].body().constants_r[0],
                "identical literals must share one immortal CONST_STR_CACHE STR \
                 even when materialized one jitcode at a time",
            );
        })
        .join()
        .unwrap();
    }

    fn unit_variant_sentinel(ordinal: i64) -> i64 {
        majit_jitcode::codewriter::assembler::UNIT_VARIANT_CONST_SENTINEL_BASE | ordinal
    }

    /// Mirror of [`jitcode_with_str_consts`] for the unit-variant bank: the
    /// assembler seeds `constants_r` with one sentinel per descriptor, in
    /// emit order.
    fn jitcode_with_unit_variant_consts(
        descs: Vec<majit_jitcode::jitcode::UnitVariantConstDescriptor>,
    ) -> Arc<JitCode> {
        let len = descs
            .iter()
            .map(|d| d.constants_r_index + 1)
            .max()
            .unwrap_or(0);
        let mut constants_r = vec![0_i64; len];
        for (ordinal, d) in descs.iter().enumerate() {
            constants_r[d.constants_r_index] = unit_variant_sentinel(ordinal as i64);
        }
        let jc = JitCode::new("test");
        jc.set_body(JitCodeBody {
            unit_variant_consts: descs,
            constants_r: constants_r.into_iter().map(Into::into).collect(),
            ..Default::default()
        });
        Arc::new(jc)
    }

    fn unit_variant_desc(
        qualname: &str,
        tag: i64,
    ) -> majit_jitcode::jitcode::UnitVariantConstDescriptor {
        majit_jitcode::jitcode::UnitVariantConstDescriptor {
            constants_r_index: 0,
            qualname: qualname.to_owned(),
            tag,
        }
    }

    #[test]
    fn materialize_unit_variant_consts_overwrites_sentinel_with_a_cell_holding_the_tag() {
        let mut jcs = vec![jitcode_with_unit_variant_consts(vec![unit_variant_desc(
            "StepResult::Continue",
            0,
        )])];
        materialize_unit_variant_consts(&mut jcs);

        let addr = jcs[0].body().constants_r[0].get();
        assert_ne!(
            addr,
            unit_variant_sentinel(0),
            "sentinel must be overwritten"
        );
        assert_eq!(
            (addr as u64) & SENTINEL_HIGH_MASK,
            0,
            "a real cell address must have the sentinel high bits clear",
        );
        // The discriminant read the portal switch performs: one word at
        // offset 0 of the cell.
        assert_eq!(unsafe { *(addr as *const i64) }, 0);
    }

    #[test]
    fn materialize_unit_variant_consts_interns_one_cell_per_qualname() {
        let mut jcs = vec![
            jitcode_with_unit_variant_consts(vec![unit_variant_desc("JitAction::Return", 1)]),
            jitcode_with_unit_variant_consts(vec![unit_variant_desc("JitAction::Return", 1)]),
        ];
        materialize_unit_variant_consts(&mut jcs);
        let a0 = jcs[0].body().constants_r[0].get();
        let a1 = jcs[1].body().constants_r[0].get();
        assert_eq!(a0, a1, "one qualname must resolve to one immortal cell");
        assert_ne!(a0, unit_variant_sentinel(0));
        assert_eq!(unsafe { *(a0 as *const i64) }, 1);
    }

    /// The interner is process-wide, not per call: two separate
    /// materializations of the same qualname share the cell, which is what
    /// keeps a discriminant comparison between two jitcodes meaningful.
    #[test]
    fn materialize_unit_variant_consts_interns_across_separate_calls() {
        let mut first = vec![jitcode_with_unit_variant_consts(vec![unit_variant_desc(
            "LoopResult::Done",
            2,
        )])];
        let mut second = vec![jitcode_with_unit_variant_consts(vec![unit_variant_desc(
            "LoopResult::Done",
            2,
        )])];
        materialize_unit_variant_consts(&mut first);
        materialize_unit_variant_consts(&mut second);
        assert_eq!(
            first[0].body().constants_r[0],
            second[0].body().constants_r[0],
            "one qualname must resolve to one cell even when materialized \
             one jitcode at a time",
        );
    }

    /// A body carrying no unit-variant descriptors is skipped whole, so its
    /// `constants_r` is left exactly as the assembler wrote it.
    #[test]
    fn materialize_unit_variant_consts_leaves_a_body_without_descriptors_alone() {
        let mut jcs = vec![jitcode_with_unit_variant_consts(Vec::new())];
        materialize_unit_variant_consts(&mut jcs);
        assert!(jcs[0].body().constants_r.is_empty());
    }

    #[test]
    fn materialize_str_consts_empty_string() {
        use majit_ir::GcRef;
        use majit_metainterp::cpu::Cpu;

        let descs = vec![StrConstDescriptor {
            constants_r_index: 0,
            bytes: Vec::new(),
            precomputed_hash: -1,
            kind: StrConstKind::Str,
            container_index: None,
        }];
        let mut jcs = vec![jitcode_with_str_consts(descs)];
        materialize_str_consts(&mut jcs);
        let addr = jcs[0].body().constants_r[0].get();
        let cpu = crate::pyre_cpu::PyreCpu::new();
        assert_eq!(cpu.bh_strlen(GcRef(addr as usize)), Some(0));
    }

    /// `unicode_as_unichars` layout: `hash` @0, `length` @one word, UCS-4
    /// `chars` after the header (`rstr.py UNICODE`, `bh_newunicode`).
    fn unicode_as_unichars_cells(addr: i64) -> &'static [u32] {
        let len = unsafe {
            *((addr as usize + pyre_object::lowlevel_string::LOWLEVEL_STRING_LEN_OFFSET)
                as *const usize)
        };
        unsafe {
            std::slice::from_raw_parts(
                (addr as usize + pyre_object::lowlevel_string::LOWLEVEL_STRING_CHARS_OFFSET)
                    as *const u32,
                len,
            )
        }
    }

    #[test]
    fn materialize_str_consts_unicode_u0100_and_u1f600() {
        let cells = [0x0100u32, 0x1F600];
        let hash = 0x1111_2222_3333_i64;
        let descs = vec![StrConstDescriptor {
            constants_r_index: 0,
            bytes: unicode_const_bytes(&cells),
            precomputed_hash: hash,
            kind: StrConstKind::Unicode,
            container_index: None,
        }];
        let mut jcs = vec![jitcode_with_str_consts(descs)];
        materialize_str_consts(&mut jcs);
        let addr = jcs[0].body().constants_r[0].get();
        assert_ne!(addr, sentinel(0), "sentinel must be overwritten");
        assert_eq!((addr as u64) & SENTINEL_HIGH_MASK, 0);
        assert_eq!(unicode_as_unichars_cells(addr), &cells);
        let got_hash = unsafe { *(addr as *const isize) } as i64;
        assert_eq!(got_hash, hash);
    }

    #[test]
    fn materialize_str_consts_keys_by_container_identity() {
        let bytes = b"same-bytes".to_vec();
        let desc = |constants_r_index, container_index| StrConstDescriptor {
            constants_r_index,
            bytes: bytes.clone(),
            precomputed_hash: 11,
            kind: StrConstKind::Str,
            container_index,
        };
        let mut jcs = vec![jitcode_with_str_consts(vec![
            desc(0, Some(0)),
            desc(1, Some(1)),
            desc(2, Some(0)),
        ])];
        materialize_str_consts(&mut jcs);
        let a0 = jcs[0].body().constants_r[0].get();
        let a1 = jcs[0].body().constants_r[1].get();
        let a2 = jcs[0].body().constants_r[2].get();
        assert_ne!(
            a0, a1,
            "equal bytes with different container ordinals must be distinct blocks"
        );
        assert_eq!(a0, a2, "equal container ordinals must share one block");
        assert_ne!(a0, sentinel(0));
        assert_ne!(a1, sentinel(1));
    }

    #[test]
    fn materialize_str_consts_independent_assemblies_same_ordinal_stay_distinct() {
        let desc = |bytes: &[u8]| StrConstDescriptor {
            constants_r_index: 0,
            bytes: bytes.to_vec(),
            precomputed_hash: 13,
            kind: StrConstKind::Str,
            container_index: Some(0),
        };
        let mut first = vec![jitcode_with_str_consts(vec![desc(b"assembly-a")])];
        let mut second = vec![jitcode_with_str_consts(vec![desc(b"assembly-b")])];
        materialize_str_consts(&mut first);
        materialize_str_consts(&mut second);
        let a0 = first[0].body().constants_r[0].get();
        let a1 = second[0].body().constants_r[0].get();
        assert_ne!(
            a0, a1,
            "independent jitcode sets must not share a container ordinal"
        );
        assert_ne!(a0, sentinel(0));
        assert_ne!(a1, sentinel(0));
    }

    #[test]
    fn materialize_exc_instance_consts_overwrites_sentinel_with_assertion_error() {
        use majit_jitcode::codewriter::assembler::EXC_INSTANCE_CONST_SENTINEL_BASE;
        use majit_jitcode::jitcode::ExcInstanceConstDescriptor;

        let desc = ExcInstanceConstDescriptor {
            constants_r_index: 0,
            class_name: "AssertionError".into(),
            message: Some(b"implicit AssertionError shouldn't occur".to_vec()),
        };
        let jc = JitCode::new("raise_assert");
        jc.set_body(JitCodeBody {
            exc_instance_consts: vec![desc],
            constants_r: vec![EXC_INSTANCE_CONST_SENTINEL_BASE.into()],
            ..Default::default()
        });
        let mut jcs = vec![Arc::new(jc)];
        materialize_exc_instance_consts(&mut jcs);
        let addr = jcs[0].body().constants_r[0].get();
        assert_ne!(addr, EXC_INSTANCE_CONST_SENTINEL_BASE);
        assert_eq!((addr as u64) & SENTINEL_HIGH_MASK, 0);
        let kind = unsafe {
            pyre_object::interp_exceptions::w_exception_get_kind(addr as pyre_object::PyObjectRef)
        };
        assert_eq!(
            kind,
            pyre_object::interp_exceptions::ExcKind::AssertionError
        );
        let again = pyre_object::interp_exceptions::prebuilt_exception_with_message(
            pyre_object::interp_exceptions::ExcKind::AssertionError,
            "implicit AssertionError shouldn't occur",
        );
        assert_eq!(addr, again as usize as i64);
    }

    #[test]
    fn materialize_empty_array_consts_overwrites_sentinel_with_zero_length_gcarray() {
        use majit_jitcode::codewriter::assembler::EMPTY_ARRAY_CONST_SENTINEL_BASE;
        use majit_jitcode::jitcode::EmptyArrayConstDescriptor;

        let desc = EmptyArrayConstDescriptor {
            constants_r_index: 0,
            type_name: "Array<PyObjectRef;0>".into(),
        };
        let jc = JitCode::new("empty_args_w");
        jc.set_body(JitCodeBody {
            empty_array_consts: vec![desc],
            constants_r: vec![EMPTY_ARRAY_CONST_SENTINEL_BASE.into()],
            ..Default::default()
        });
        let mut jcs = vec![Arc::new(jc)];
        materialize_empty_array_consts(&mut jcs);
        let addr = jcs[0].body().constants_r[0].get();
        assert_ne!(addr, EMPTY_ARRAY_CONST_SENTINEL_BASE);
        assert_eq!((addr as u64) & SENTINEL_HIGH_MASK, 0);
        assert_ne!(addr, 0);
        assert_eq!(
            pyre_object::gcarray_len(addr as *const pyre_object::GcTypedArray),
            0
        );
        let again = pyre_object::prebuilt_empty_object_array() as i64;
        assert_eq!(addr, again);
    }

    #[test]
    fn materialize_empty_u8_array_consts_overwrites_sentinel() {
        use majit_jitcode::codewriter::assembler::EMPTY_ARRAY_CONST_SENTINEL_BASE;
        use majit_jitcode::jitcode::EmptyArrayConstDescriptor;

        let desc = EmptyArrayConstDescriptor {
            constants_r_index: 0,
            type_name: "Array<u8;0>".into(),
        };
        let jc = JitCode::new("empty_bytes");
        jc.set_body(JitCodeBody {
            empty_array_consts: vec![desc],
            constants_r: vec![EMPTY_ARRAY_CONST_SENTINEL_BASE.into()],
            ..Default::default()
        });
        let mut jcs = vec![Arc::new(jc)];
        materialize_empty_array_consts(&mut jcs);
        let addr = jcs[0].body().constants_r[0].get();
        assert_ne!(addr, 0);
        assert_eq!((addr as u64) & SENTINEL_HIGH_MASK, 0);
        assert_eq!(
            pyre_object::gcarray_len(addr as *const pyre_object::GcTypedArray),
            0
        );
    }

    fn prebuilt_instance_desc(
        container_index: u32,
        size: usize,
        type_id: u64,
        has_vtable: bool,
    ) -> majit_jitcode::jitcode::PrebuiltInstanceConstDescriptor {
        majit_jitcode::jitcode::PrebuiltInstanceConstDescriptor {
            constants_r_index: 0,
            container_index,
            size,
            type_id,
            has_vtable,
            is_gc_managed: true,
            headerless: false,
        }
    }

    fn jitcode_with_prebuilt(
        name: &str,
        desc: majit_jitcode::jitcode::PrebuiltInstanceConstDescriptor,
    ) -> Arc<JitCode> {
        use majit_jitcode::codewriter::assembler::PREBUILT_INSTANCE_CONST_SENTINEL_BASE;
        let jc = JitCode::new(name);
        jc.set_body(JitCodeBody {
            prebuilt_instance_consts: vec![desc],
            constants_r: vec![PREBUILT_INSTANCE_CONST_SENTINEL_BASE.into()],
            ..Default::default()
        });
        Arc::new(jc)
    }

    /// Kind-0 decode then `gc.py init_size_descr` analog: publish the Size
    /// `new` interns, then stamp that descr on a local `MiniMarkGC`
    /// (`TypeRegistry::register`). Slot 0 is OBJECT. Does not install a
    /// process-global collector (`gc_sync::store_singleton`) and does not
    /// walk `_cache_size` (`register_unresolved_struct_tids`).
    fn publish_kind0_size_then_stamp(
        desc: &majit_jitcode::jitcode::PrebuiltInstanceConstDescriptor,
    ) -> u32 {
        use majit_ir::descr::{Descr, FieldDescr, SimpleSizeDescr, SizeDescr};
        let descr = crate::descr::make_descr_from_bh(&prebuilt_instance_size_descr(desc));
        let sd = descr
            .as_any()
            .and_then(|descr| descr.downcast_ref::<SimpleSizeDescr>())
            .expect("new of the same Size descr");
        let mut gc = majit_gc::collector::MiniMarkGC::new();
        let object_tid = gc.register_type(majit_gc::TypeInfo::simple(16));
        assert_eq!(object_tid, 0);
        let ref_offsets: Vec<usize> = sd
            .gc_fielddescrs()
            .iter()
            .map(|field| field.offset())
            .collect();
        let tid = gc.register_type(majit_gc::trace::TypeInfo::with_gc_ptrs(
            sd.size(),
            ref_offsets,
        ));
        sd.set_type_id(tid);
        tid
    }

    fn published_size_descr(
        desc: &majit_jitcode::jitcode::PrebuiltInstanceConstDescriptor,
    ) -> majit_ir::descr::DescrRef {
        // descr.py get_size_descr cache[STRUCT].
        majit_ir::descr::gc_cache().lock().get_size_descr(
            majit_ir::descr::LLType::Struct(desc.type_id),
            desc.size,
            0,
            false,
        )
    }

    fn assert_header_tid_is_stamped(
        header_tid: u32,
        desc: &majit_jitcode::jitcode::PrebuiltInstanceConstDescriptor,
    ) {
        let size_descr = published_size_descr(desc);
        let sd = size_descr
            .as_size_descr()
            .expect("new of the same Size descr");
        assert_eq!(
            header_tid,
            sd.type_id(),
            "prebuilt header tid is the collector id a new of this STRUCT gets"
        );
        assert_ne!(
            header_tid, 0,
            "tid 0 is OBJECT, not an unstamped placeholder"
        );
        assert!(
            u64::from(header_tid) <= majit_gc::header::TYPE_ID_MASK,
            "header tid {header_tid} does not fit TYPE_ID_BITS"
        );
        assert!(
            !majit_ir::descr::struct_tid_is_unresolved(sd.cache_key(), header_tid),
            "header tid is the collector id init_size_descr / get_type_id stamps"
        );
    }

    /// Collector-less materialize takes `bh_alloc_struct`'s no-collector
    /// tail: immortal zeroed block, no GC header, no tid. `vtable` is the
    /// typeptr word `initialize_prebuilt_instance` writes when the STRUCT
    /// has one.
    fn assert_raw_prebuilt_payload(
        addr: i64,
        desc: &majit_jitcode::jitcode::PrebuiltInstanceConstDescriptor,
        vtable: Option<usize>,
    ) {
        use majit_jitcode::codewriter::assembler::PREBUILT_INSTANCE_CONST_SENTINEL_BASE;
        assert!(
            !majit_gc::gc_sync::is_initialized(),
            "these tests do not install a process-global collector"
        );
        assert_ne!(addr, PREBUILT_INSTANCE_CONST_SENTINEL_BASE);
        assert_eq!((addr as u64) & SENTINEL_HIGH_MASK, 0);
        assert_ne!(addr, 0);
        let nbytes = desc.size.max(1);
        let bytes = unsafe { std::slice::from_raw_parts(addr as *const u8, nbytes) };
        if let Some(vtable) = vtable {
            assert!(
                desc.size >= std::mem::size_of::<usize>(),
                "typeptr-header prebuilt needs a word, size={}",
                desc.size
            );
            assert_eq!(
                unsafe { *(addr as *const usize) },
                vtable,
                "typeptr is ClassRepr.getvtable, the registered STRUCT vtable"
            );
            assert!(
                bytes[std::mem::size_of::<usize>()..]
                    .iter()
                    .all(|&b| b == 0),
                "collector-less prebuilt payload after the typeptr word must be zeroed"
            );
        } else {
            assert!(
                bytes.iter().all(|&b| b == 0),
                "collector-less prebuilt payload must be zeroed"
            );
        }
    }

    #[test]
    fn materialize_prebuilt_instance_consts_overwrites_sentinel_with_header_only_instance() {
        let type_id = majit_ir::descr::path_hash("c1i::HeaderOnlyEmpty");
        let desc = prebuilt_instance_desc(0, 0, type_id, false);
        publish_kind0_size_then_stamp(&desc);
        let tid = prebuilt_instance_tid(&desc);
        assert_header_tid_is_stamped(tid, &desc);
        let mut jcs = vec![jitcode_with_prebuilt("dict_key_error", desc.clone())];
        materialize_prebuilt_instance_consts(&mut jcs);
        let addr = jcs[0].body().constants_r[0].get();
        assert_raw_prebuilt_payload(addr, &desc, None);
        let registered = majit_ir::descr::gc_cache()
            .lock()
            ._cache_gcstruct2vtable
            .get(&type_id)
            .copied();
        assert!(
            registered.is_none(),
            "header-only new does not mint a vtable (get_vtable_for_gcstruct only reads)"
        );
    }

    #[test]
    fn materialize_prebuilt_instance_consts_unstamped_header_tid_matches_new() {
        let type_id = majit_ir::descr::path_hash("c1i::UnstampedHeaderOnly");
        assert!(
            majit_ir::descr::gc_cache()
                .lock()
                .resolve_struct_layout(type_id)
                .is_none(),
            "size descr must not be pre-stamped"
        );
        let desc = prebuilt_instance_desc(0, 0, type_id, false);
        let stamped_tid = publish_kind0_size_then_stamp(&desc);
        let tid = prebuilt_instance_tid(&desc);
        assert_header_tid_is_stamped(tid, &desc);
        assert_eq!(
            tid, stamped_tid,
            "header tid is the tid the local init_size_descr analog returned"
        );
        let mut jcs = vec![jitcode_with_prebuilt("unstamped_header", desc.clone())];
        materialize_prebuilt_instance_consts(&mut jcs);
        let addr = jcs[0].body().constants_r[0].get();
        assert_raw_prebuilt_payload(addr, &desc, None);
    }

    #[test]
    fn materialize_prebuilt_instance_consts_interns_one_instance_per_identity() {
        let type_id = majit_ir::descr::path_hash("c1i::InternByIdentity");
        let same = prebuilt_instance_desc(0, 0, type_id, false);
        publish_kind0_size_then_stamp(&same);
        let tid = prebuilt_instance_tid(&same);
        assert_header_tid_is_stamped(tid, &same);
        let mut intern = AssemblyConstIntern::new();
        let mut first = vec![jitcode_with_prebuilt("a", same.clone())];
        let mut second = vec![jitcode_with_prebuilt("b", same.clone())];
        materialize_prebuilt_instance_consts_in(&mut first, &mut intern);
        materialize_prebuilt_instance_consts_in(&mut second, &mut intern);
        assert_eq!(
            first[0].body().constants_r[0],
            second[0].body().constants_r[0],
            "one HostObject ordinal must resolve to one immortal instance"
        );
        let addr_a = first[0].body().constants_r[0].get();
        assert_raw_prebuilt_payload(addr_a, &same, None);
        let other = {
            let desc = prebuilt_instance_desc(1, 0, type_id, false);
            let mut jcs = vec![jitcode_with_prebuilt("other", desc)];
            materialize_prebuilt_instance_consts_in(&mut jcs, &mut intern);
            jcs[0].body().constants_r[0].get()
        };
        assert_ne!(
            addr_a, other,
            "distinct HostObjects keep distinct instances"
        );
    }

    #[test]
    fn materialize_prebuilt_instance_consts_two_identities_of_one_class_are_distinct() {
        let type_id = majit_ir::descr::path_hash("c1i::TwoRuntimeHosts");
        let a = prebuilt_instance_desc(0, 0, type_id, false);
        let b = prebuilt_instance_desc(1, 0, type_id, false);
        publish_kind0_size_then_stamp(&a);
        let tid = prebuilt_instance_tid(&a);
        assert_header_tid_is_stamped(tid, &a);
        let mut jcs = vec![
            jitcode_with_prebuilt("host_a", a.clone()),
            jitcode_with_prebuilt("host_b", b),
        ];
        materialize_prebuilt_instance_consts(&mut jcs);
        let addr_a = jcs[0].body().constants_r[0].get();
        let addr_b = jcs[1].body().constants_r[0].get();
        assert_ne!(
            addr_a, addr_b,
            "two HostObjects of one empty class are two runtime objects"
        );
        assert_raw_prebuilt_payload(addr_a, &a, None);
    }

    #[test]
    fn materialize_prebuilt_instance_consts_uses_registered_class_vtable() {
        let type_id = majit_ir::descr::path_hash("c1i::RegisteredVtableEmpty");
        let real_vtable = Box::leak(Box::new(0xC1B2_0000_0001usize)) as *const usize as usize;
        let real_tid = 4242u32;
        {
            let mut gc = majit_ir::descr::gc_cache().lock();
            gc.register_gcstruct_vtable(type_id, real_vtable, real_tid);
        }

        let desc = prebuilt_instance_desc(0, 8, type_id, true);
        // heaptracker.setup_cache_gcstruct2vtable before get_size_descr.
        crate::jitcode_runtime::materialize_gccache_owned_descrs();
        let mut jcs = vec![jitcode_with_prebuilt("registered_vtable", desc.clone())];
        materialize_prebuilt_instance_consts(&mut jcs);
        let addr = jcs[0].body().constants_r[0].get();
        assert_raw_prebuilt_payload(addr, &desc, Some(real_vtable));
        let registered = majit_ir::descr::gc_cache()
            .lock()
            ._cache_gcstruct2vtable
            .get(&type_id)
            .copied();
        assert_eq!(registered, Some((real_vtable, real_tid)));
    }

    #[test]
    #[should_panic(expected = "no registered vtable")]
    fn materialize_prebuilt_instance_consts_panics_when_typeptr_header_has_no_vtable() {
        let type_id = majit_ir::descr::path_hash("c1i::MissingVtableEmpty");
        let desc = prebuilt_instance_desc(0, 8, type_id, true);
        let mut jcs = vec![jitcode_with_prebuilt("missing_vtable", desc)];
        materialize_prebuilt_instance_consts(&mut jcs);
    }

    #[test]
    #[should_panic(expected = "foreign allocation layout")]
    fn materialize_prebuilt_instance_consts_panics_on_foreign_allocation_layout() {
        let type_id = majit_ir::descr::path_hash("c1i::ForeignLayoutEmpty");
        let _ = majit_ir::descr::gc_cache().lock().get_size_descr(
            majit_ir::descr::LLType::Struct(type_id),
            16,
            0,
            false,
        );
        let desc = prebuilt_instance_desc(0, 0, type_id, false);
        // Collector-less materialize takes the raw branch and does not call
        // `prebuilt_instance_tid`. The layout check still lives there and
        // is reached with a collector (`prebuilt_empty_class_instance_addr`).
        let _ = prebuilt_instance_tid(&desc);
    }

    #[test]
    fn materialize_prebuilt_instance_consts_unpublished_struct_without_collector_matches_new() {
        let type_id = majit_ir::descr::path_hash("c1i::UnpublishedPrebuilt");
        let desc = prebuilt_instance_desc(0, 0, type_id, false);
        let mut jcs = vec![jitcode_with_prebuilt("unpublished", desc.clone())];
        materialize_prebuilt_instance_consts(&mut jcs);
        let addr = jcs[0].body().constants_r[0].get();
        assert_raw_prebuilt_payload(addr, &desc, None);
    }

    #[test]
    fn materialize_prebuilt_instance_consts_vtable_class_first_keeps_vtable_on_new() {
        let type_id = majit_ir::descr::path_hash("c1i::VtableClassBeforeNew");
        let real_vtable = Box::leak(Box::new(0xC1B2_0000_0002usize)) as *const usize as usize;
        let real_tid = 4343u32;
        {
            let mut gc = majit_ir::descr::gc_cache().lock();
            gc.register_gcstruct_vtable(type_id, real_vtable, real_tid);
        }

        let desc = prebuilt_instance_desc(0, 8, type_id, true);
        // heaptracker.setup_cache_gcstruct2vtable before get_size_descr.
        crate::jitcode_runtime::materialize_gccache_owned_descrs();
        let mut jcs = vec![jitcode_with_prebuilt("vtable_before_new", desc.clone())];
        materialize_prebuilt_instance_consts(&mut jcs);
        let addr = jcs[0].body().constants_r[0].get();
        assert_raw_prebuilt_payload(addr, &desc, Some(real_vtable));
        let size_descr = published_size_descr(&desc);
        let sd = size_descr
            .as_size_descr()
            .expect("new of the same Size descr");
        assert_eq!(
            sd.vtable(),
            real_vtable,
            "new_with_vtable keeps ClassRepr.getvtable"
        );
        assert_ne!(sd.vtable(), 0);
    }

    #[test]
    fn materialize_prebuilt_instance_consts_raw_flavor_zeroed_without_gc_header() {
        use majit_jitcode::codewriter::assembler::PREBUILT_INSTANCE_CONST_SENTINEL_BASE;
        use majit_jitcode::jitcode::BhDescr;

        let type_id = majit_ir::descr::path_hash("c1i::RawFlavorEmpty");
        let desc = majit_jitcode::jitcode::PrebuiltInstanceConstDescriptor {
            constants_r_index: 0,
            container_index: 0,
            size: 16,
            type_id,
            has_vtable: false,
            is_gc_managed: false,
            headerless: false,
        };
        match prebuilt_instance_size_descr(&desc) {
            BhDescr::Size { is_gc_managed, .. } => {
                assert!(!is_gc_managed, "raw flavor Size is not GC-managed")
            }
            other => panic!("expected Size descr, got {other:?}"),
        }
        let mut jcs = vec![jitcode_with_prebuilt("raw_flavor", desc.clone())];
        materialize_prebuilt_instance_consts(&mut jcs);
        let addr = jcs[0].body().constants_r[0].get();
        assert_ne!(addr, PREBUILT_INSTANCE_CONST_SENTINEL_BASE);
        assert_eq!((addr as u64) & SENTINEL_HIGH_MASK, 0);
        assert_ne!(addr, 0);
        let bytes = unsafe { std::slice::from_raw_parts(addr as *const u8, desc.size) };
        assert!(
            bytes.iter().all(|&b| b == 0),
            "raw immortal payload must be zeroed"
        );
    }

    #[test]
    fn materialize_prebuilt_instance_consts_truncated_path_hash_without_collector_is_raw() {
        let type_id = majit_ir::descr::path_hash("c1i::TruncatedPathHashPrebuilt");
        assert!(
            type_id > u32::MAX as u64,
            "path_hash must truncate when stored as SizeDescr type_id"
        );
        let real_vtable = Box::leak(Box::new(0xC1B2_0000_0003usize)) as *const usize as usize;
        {
            let mut gc = majit_ir::descr::gc_cache().lock();
            gc.register_gcstruct_vtable(type_id, real_vtable, 1);
        }
        let desc = prebuilt_instance_desc(0, 8, type_id, true);
        let mut jcs = vec![jitcode_with_prebuilt("truncated_path_hash", desc.clone())];
        materialize_prebuilt_instance_consts(&mut jcs);
        let addr = jcs[0].body().constants_r[0].get();
        assert_raw_prebuilt_payload(addr, &desc, Some(real_vtable));
    }

    /// Walker folds that recognise a residual by callee compare against this
    /// map. An unpublished path would silently disable the fold on wasm32
    /// (and, without the raw-cast fallback, on native too).
    #[test]
    fn patch_constants_i_rewrites_only_provenance_tagged_slots() {
        use majit_jitcode::jitcode::{ConstIRelocDescriptor, ConstIRelocKind};

        let (path, build_fnaddr) = build_time_fnaddr_bindings()
            .into_iter()
            .find(|(p, build)| runtime_fnaddr_by_path(p).is_some_and(|runtime| runtime != *build))
            .expect(
                "need a registered fnaddr whose runtime address differs from the build snapshot",
            );
        let runtime = runtime_fnaddr_by_path(&path).expect("path published at runtime");
        let symbolic = {
            const BYTES: &[u8] =
                include_bytes!(concat!(env!("OUT_DIR"), "/symbolic_fnaddr_paths.bin"));
            let entries: Vec<(i64, String)> = bincode::deserialize(BYTES).unwrap();
            entries
                .into_iter()
                .map(|(hash, _)| hash)
                .find(|hash| *hash != build_fnaddr)
                .expect("need a registered symbolic fnaddr hash")
        };

        let jc = JitCode::new("ordinary_int_alias");
        jc.set_body(JitCodeBody {
            constants_i: vec![build_fnaddr, symbolic, build_fnaddr],
            reloc_consts_i: vec![ConstIRelocDescriptor {
                constants_i_index: 2,
                kind: ConstIRelocKind::FnAddr {
                    path: path.clone(),
                    symbolic: false,
                },
            }],
            ..Default::default()
        });
        let mut jcs = vec![Arc::new(jc)];
        patch_constants_i_fnaddrs(&mut jcs);
        patch_static_addr_constants(&mut jcs);
        let body = jcs[0].body();
        assert_eq!(
            body.constants_i[0], build_fnaddr,
            "ordinary integer equal to a registered build fnaddr must be kept"
        );
        assert_eq!(
            body.constants_i[1], symbolic,
            "ordinary integer equal to a registered symbolic hash must be kept"
        );
        assert_eq!(
            body.constants_i[2], runtime,
            "provenance-tagged slot must be rewritten to the runtime address"
        );
    }

    #[test]
    fn patch_static_addr_constants_rewrites_only_provenance_tagged_slots() {
        use majit_jitcode::jitcode::{ConstIRelocDescriptor, ConstIRelocKind};

        let (name, build_addr) = build_time_pytype_bindings()
            .into_iter()
            .find(|(n, build)| runtime_static_addr_by_name(n).is_some_and(|runtime| runtime != *build))
            .expect(
                "need a registered static addr whose runtime address differs from the build snapshot",
            );
        let runtime = runtime_static_addr_by_name(&name).expect("name published at runtime");

        let jc = JitCode::new("ordinary_static_alias");
        jc.set_body(JitCodeBody {
            constants_i: vec![build_addr, build_addr],
            reloc_consts_i: vec![ConstIRelocDescriptor {
                constants_i_index: 1,
                kind: ConstIRelocKind::StaticAddr { name: name.clone() },
            }],
            ..Default::default()
        });
        let mut jcs = vec![Arc::new(jc)];
        patch_static_addr_constants(&mut jcs);
        let body = jcs[0].body();
        assert_eq!(
            body.constants_i[0], build_addr,
            "ordinary integer equal to a registered build static addr must be kept"
        );
        assert_eq!(
            body.constants_i[1], runtime,
            "provenance-tagged static-addr slot must be rewritten to the runtime address"
        );
    }

    #[test]
    fn patch_static_addr_constants_rewrites_only_provenance_tagged_constants_r() {
        use majit_jitcode::jitcode::{ConstIRelocKind, ConstRRelocDescriptor};

        let (name, build_addr) = build_time_ref_bindings()
            .into_iter()
            .find(|(n, build)| runtime_static_addr_by_name(n).is_some_and(|runtime| runtime != *build))
            .expect(
                "need a registered static ref whose runtime address differs from the build snapshot",
            );
        let runtime = runtime_static_addr_by_name(&name).expect("name published at runtime");

        let jc = JitCode::new("ordinary_ref_alias");
        jc.set_body(JitCodeBody {
            constants_r: vec![build_addr.into(), build_addr.into()],
            reloc_consts_r: vec![ConstRRelocDescriptor {
                constants_r_index: 1,
                kind: ConstIRelocKind::StaticAddr { name: name.clone() },
            }],
            ..Default::default()
        });
        let mut jcs = vec![Arc::new(jc)];
        patch_static_addr_constants(&mut jcs);
        let body = jcs[0].body();
        assert_eq!(
            body.constants_r[0].get(),
            build_addr,
            "ordinary ref equal to a registered build static addr must be kept"
        );
        assert_eq!(
            body.constants_r[1].get(),
            runtime,
            "provenance-tagged constants_r slot must be rewritten to the runtime address"
        );
    }

    #[test]
    fn recorded_nonsymbolic_fnaddr_resolves_or_is_disarmed() {
        let mut jcs = unpatched_production_jitcodes();
        new_provenance_patch_pipeline(&mut jcs);

        let mut live_unresolved = Vec::new();
        for jc in &jcs {
            let Some(ConstIRelocKind::FnAddr { path, symbolic }) = &jc.fnaddr_reloc else {
                continue;
            };
            if *symbolic {
                continue;
            }
            match runtime_fnaddr_by_path(path) {
                Some(runtime) => assert_eq!(
                    jc.fnaddr, runtime,
                    "non-symbolic fnaddr_reloc {path} in {} must resolve via runtime_fnaddr_by_path",
                    jc.name
                ),
                None if jc.fnaddr != 0 => {
                    live_unresolved.push(format!("{path} in {} val={:#x}", jc.name, jc.fnaddr));
                }
                None => {}
            }
        }
        live_unresolved.sort();
        live_unresolved.dedup();
        assert!(
            live_unresolved.is_empty(),
            "non-symbolic fnaddr_reloc did not resolve and was not disarmed: {live_unresolved:?}"
        );
    }

    #[test]
    fn runtime_fnaddr_by_path_resolves_walker_residual_fold_callees() {
        assert!(
            runtime_fnaddr_by_path("pyre_interpreter::call::take_last_exec_ctx").is_some(),
            "pyre_interpreter::call::take_last_exec_ctx must be published in jit_trace_fnaddrs"
        );
    }

    /// An untagged `constants_i` word that equals a build-process fnaddr
    /// used to be rewritten by value. After provenance tagging, only
    /// `reloc_consts_i` slots move; an untagged hit is a missed
    /// `ConstFnAddr` / `emit_const_i_reloc` site (ordinary ints colliding
    /// with a 64-bit ASLR address are not realistic).
    #[test]
    fn untagged_constants_i_do_not_equal_a_build_fnaddr() {
        use majit_jitcode::jitcode::ConstIRelocKind;
        use std::collections::HashSet;

        let build_fnaddrs: HashSet<i64> = build_time_fnaddr_bindings()
            .into_iter()
            .map(|(_, addr)| addr)
            .filter(|addr| *addr != 0)
            .collect();
        let build_statics: HashSet<i64> = build_time_pytype_bindings()
            .into_iter()
            .chain(build_time_ref_bindings())
            .map(|(_, addr)| addr)
            .filter(|addr| *addr != 0)
            .collect();

        let mut hits = Vec::new();
        for jc in crate::jitcode_runtime::all_jitcodes() {
            let Some(body) = jc.try_body() else {
                continue;
            };
            let tagged: HashSet<usize> = body
                .reloc_consts_i
                .iter()
                .filter(|d| {
                    matches!(
                        d.kind,
                        ConstIRelocKind::FnAddr { .. } | ConstIRelocKind::StaticAddr { .. }
                    )
                })
                .map(|d| d.constants_i_index)
                .collect();
            for (index, &c) in body.constants_i.iter().enumerate() {
                if tagged.contains(&index) {
                    continue;
                }
                if build_fnaddrs.contains(&c) {
                    hits.push(format!("fnaddr {c:#x} at {} constants_i[{index}]", jc.name));
                }
                if build_statics.contains(&c) {
                    hits.push(format!("static {c:#x} at {} constants_i[{index}]", jc.name));
                }
            }
        }
        hits.sort();
        hits.dedup();
        assert!(
            hits.is_empty(),
            "untagged constants_i equal a build address: {hits:?}"
        );

        let mut r_hits = Vec::new();
        for jc in crate::jitcode_runtime::all_jitcodes() {
            let Some(body) = jc.try_body() else {
                continue;
            };
            let tagged: HashSet<usize> = body
                .reloc_consts_r
                .iter()
                .map(|d| d.constants_r_index)
                .collect();
            for (index, slot) in body.constants_r.iter().enumerate() {
                if tagged.contains(&index) {
                    continue;
                }
                let c = slot.get();
                if build_statics.contains(&c) {
                    r_hits.push(format!("static {c:#x} at {} constants_r[{index}]", jc.name));
                }
            }
        }
        r_hits.sort();
        r_hits.dedup();
        assert!(
            r_hits.is_empty(),
            "untagged constants_r equal a build static addr: {r_hits:?}"
        );
    }

    /// An untagged `constants_i` word equal to an unpaired build address
    /// used to be zeroed by the value scan. Provenance disarm leaves it.
    #[test]
    fn untagged_constants_i_equal_to_unpaired_build_addr_survives_disarm() {
        // Every build binding is paired today, so the set is normally
        // empty; a word no binding names is then as untagged as any.
        let unpaired_addr = UNPAIRED_BUILD_ADDRS
            .keys()
            .copied()
            .find(|addr| *addr != 0)
            .unwrap_or(0x5eed_0000_1000);

        let jc = JitCode::new("untagged_unpaired_collision");
        jc.set_body(JitCodeBody {
            constants_i: vec![unpaired_addr],
            ..Default::default()
        });
        let mut jcs = vec![Arc::new(jc)];
        disarm_unpaired_build_addrs(&mut jcs);
        assert_eq!(
            jcs[0].body().constants_i[0],
            unpaired_addr,
            "untagged integer equal to an unpaired build address must survive disarm"
        );
    }

    /// A tagged FnAddr whose path does not exact-match a published
    /// `jit_trace_fnaddrs` key must not keep a build-process pointer.
    #[test]
    fn unresolved_tagged_fnaddr_slots_are_not_live_pointers() {
        use majit_jitcode::jitcode::ConstIRelocKind;

        let mut live = Vec::new();
        for jc in crate::jitcode_runtime::all_jitcodes() {
            let Some(body) = jc.try_body() else {
                continue;
            };
            for desc in &body.reloc_consts_i {
                let ConstIRelocKind::FnAddr { path, symbolic } = &desc.kind else {
                    continue;
                };
                if runtime_fnaddr_for_reloc_path(path).is_some() {
                    continue;
                }
                let val = body.constants_i[desc.constants_i_index];
                if val != 0 && !*symbolic {
                    live.push(format!("{path} in {} val={val:#x}", jc.name));
                }
            }
        }
        live.sort();
        live.dedup();
        assert!(
            live.is_empty(),
            "unresolved tagged fnaddr slots still hold a pointer: {live:?}"
        );
    }

    /// Oracle only: the descriptor flag the codewriter recorded must equal
    /// `is_symbolic_fnaddr` of the stored word. Production never consults
    /// the bits of a tagged slot.
    fn assert_tagged_fnaddr_symbolic_flag_matches_bit_test(jcs: &[Arc<JitCode>]) {
        use majit_jitcode::codewriter::call::is_symbolic_fnaddr;

        let mut mismatches = Vec::new();
        for jc in jcs {
            let Some(body) = jc.try_body() else {
                continue;
            };
            for desc in &body.reloc_consts_i {
                let ConstIRelocKind::FnAddr { path, symbolic } = &desc.kind else {
                    continue;
                };
                let value = body.constants_i[desc.constants_i_index];
                let bits = is_symbolic_fnaddr(value);
                if *symbolic != bits {
                    mismatches.push(format!(
                        "{path} in {} constants_i[{}] val={value:#x} flag={symbolic} bits={bits}",
                        jc.name, desc.constants_i_index
                    ));
                }
            }
        }
        mismatches.sort();
        mismatches.dedup();
        assert!(
            mismatches.is_empty(),
            "tagged FnAddr symbolic flag disagrees with the bit test: {mismatches:?}"
        );
    }

    fn unpatched_production_jitcodes() -> Vec<Arc<JitCode>> {
        use majit_jitcode::artifacts::JitCodeIndex;
        const INDEX_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/jitcodes_index.bin"));
        const BODY_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/jitcodes.bin"));
        let index = JitCodeIndex::decode(INDEX_BYTES, BODY_BYTES).expect("JitCode index");
        (0..index.offsets.len().saturating_sub(1))
            .map(|i| {
                index
                    .load(BODY_BYTES, i)
                    .unwrap_or_else(|e| panic!("deserialize jitcodes.bin entry {i}: {e}"))
            })
            .collect()
    }

    /// Test-only oracle: HEAD's value-based `patch_constants_i_fnaddrs`.
    /// Rewrites every `constants_i` word (and the shell `fnaddr`) whose bits
    /// appear in `FNADDR_CORRESPONDENCE`, then resolves leftover symbolic
    /// hashes through `runtime_fnaddr_by_path`.
    fn old_value_based_patch_constants_i_fnaddrs(jitcodes: &mut [Arc<JitCode>]) {
        let correspondence = &*FNADDR_CORRESPONDENCE;
        for arc in jitcodes.iter_mut() {
            let jc = Arc::get_mut(arc).expect("old-value oracle: Arc already shared");
            if let Some(&runtime) = correspondence.get(&jc.fnaddr) {
                jc.fnaddr = runtime;
            } else if majit_jitcode::codewriter::call::is_symbolic_fnaddr(jc.fnaddr)
                && let Some(path) = symbolic_fnaddr_path(jc.fnaddr)
                && let Some(runtime) = runtime_fnaddr_by_path(path)
            {
                jc.fnaddr = runtime;
            }
            if jc.try_body().is_some() {
                for c in jc.body_mut().constants_i.iter_mut() {
                    if let Some(&runtime) = correspondence.get(c) {
                        *c = runtime;
                    } else if majit_jitcode::codewriter::call::is_symbolic_fnaddr(*c)
                        && let Some(path) = symbolic_fnaddr_path(*c)
                        && let Some(runtime) = runtime_fnaddr_by_path(path)
                    {
                        *c = runtime;
                    }
                }
            }
        }
    }

    /// Test-only oracle: HEAD's value-based `disarm_unpaired_build_addrs`.
    /// Zeros every `constants_i` / `constants_r` word whose bits are an
    /// unpaired build address, tagged or not.
    fn old_value_based_disarm_unpaired_build_addrs(jitcodes: &mut [Arc<JitCode>]) {
        let unpaired = &*UNPAIRED_BUILD_ADDRS;
        if unpaired.is_empty() {
            return;
        }
        for arc in jitcodes.iter_mut() {
            let jc = Arc::get_mut(arc).expect("old-value oracle: Arc already shared");
            if unpaired.contains_key(&jc.fnaddr) {
                jc.fnaddr = 0;
            }
            if jc.try_body().is_some() {
                let body = jc.body_mut();
                for c in body
                    .constants_i
                    .iter_mut()
                    .chain(body.constants_r.iter_mut().map(|slot| slot.get_mut()))
                {
                    if unpaired.contains_key(c) {
                        *c = 0;
                    }
                }
            }
        }
    }

    /// Test-only oracle: HEAD's `patch_static_addr_constants`. Disarm first,
    /// then rewrite every remaining word in both pools by
    /// `STATIC_ADDR_CORRESPONDENCE`.
    fn old_value_based_patch_static_addr_constants(jitcodes: &mut [Arc<JitCode>]) {
        let correspondence = &*STATIC_ADDR_CORRESPONDENCE;
        old_value_based_disarm_unpaired_build_addrs(jitcodes);
        if correspondence.is_empty() {
            return;
        }
        for arc in jitcodes.iter_mut() {
            let jc = Arc::get_mut(arc).expect("old-value oracle: Arc already shared");
            if jc.try_body().is_some() {
                let body = jc.body_mut();
                for c in body
                    .constants_i
                    .iter_mut()
                    .chain(body.constants_r.iter_mut().map(|slot| slot.get_mut()))
                {
                    if let Some(&runtime) = correspondence.get(c) {
                        *c = runtime;
                    }
                }
            }
        }
    }

    fn old_value_based_patch_pipeline(jitcodes: &mut [Arc<JitCode>]) {
        old_value_based_patch_constants_i_fnaddrs(jitcodes);
        old_value_based_patch_static_addr_constants(jitcodes);
    }

    fn new_provenance_patch_pipeline(jitcodes: &mut [Arc<JitCode>]) {
        patch_constants_i_fnaddrs(jitcodes);
        patch_static_addr_constants(jitcodes);
    }

    /// Build words bound to more than one distinct path/name in the
    /// build-time binding tables. Optimized MSVC links pass `/OPT:REF,ICF`
    /// to link.exe, so identical-COMDAT folding can give several distinct
    /// functions one build address; the value-based oracle then maps that
    /// word to whichever binding it last inserted.
    fn colliding_build_words_from<S: AsRef<str>>(
        bindings: impl IntoIterator<Item = (S, i64)>,
    ) -> std::collections::HashSet<i64> {
        use std::collections::{HashMap, HashSet};
        let mut names_by_word: HashMap<i64, HashSet<String>> = HashMap::new();
        for (name, word) in bindings {
            names_by_word
                .entry(word)
                .or_default()
                .insert(name.as_ref().to_owned());
        }
        names_by_word
            .into_iter()
            .filter(|(_, names)| names.len() > 1)
            .map(|(word, _)| word)
            .collect()
    }

    fn production_colliding_build_words() -> std::collections::HashSet<i64> {
        colliding_build_words_from(
            build_time_fnaddr_bindings()
                .into_iter()
                .chain(build_time_pytype_bindings())
                .chain(build_time_ref_bindings()),
        )
    }

    /// Same lookup the provenance pipeline uses for a tagged slot.
    fn runtime_lookup_for_reloc_kind(kind: &ConstIRelocKind) -> Option<i64> {
        match kind {
            ConstIRelocKind::FnAddr { path, .. } => runtime_fnaddr_for_reloc_path(path),
            ConstIRelocKind::StaticAddr { name } => runtime_static_addr_by_name(name),
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum TaggedRelocMismatchClass {
        /// `build` is bound to more than one path/name, and `new` is the
        /// runtime address of the slot's own recorded path.
        IcfCollision,
        Unexpected,
    }

    fn classify_tagged_reloc_mismatch(
        build: i64,
        new: i64,
        slot_runtime: Option<i64>,
        colliding: &std::collections::HashSet<i64>,
    ) -> TaggedRelocMismatchClass {
        if colliding.contains(&build) && slot_runtime == Some(new) {
            TaggedRelocMismatchClass::IcfCollision
        } else {
            TaggedRelocMismatchClass::Unexpected
        }
    }

    fn note_tagged_reloc_mismatch(
        tag: String,
        location: String,
        build: i64,
        old: i64,
        new: i64,
        slot_runtime: Option<i64>,
        colliding: &std::collections::HashSet<i64>,
        unexpected: &mut Vec<String>,
    ) {
        match classify_tagged_reloc_mismatch(build, new, slot_runtime, colliding) {
            TaggedRelocMismatchClass::IcfCollision => {
                assert!(
                    colliding.contains(&build),
                    "collision branch taken for a build word bound to one path/name: \
                     {tag} in {location} build={build:#x}"
                );
                assert_eq!(
                    Some(new),
                    slot_runtime,
                    "provenance pipeline must write the runtime address of the slot's \
                     own recorded path: {tag} in {location} new={new:#x}"
                );
            }
            TaggedRelocMismatchClass::Unexpected => {
                unexpected.push(format!(
                    "{tag} in {location} build={build:#x} old={old:#x} new={new:#x}"
                ));
            }
        }
    }

    /// Every tagged FnAddr / StaticAddr slot's final value after the
    /// provenance pipeline must equal the final value HEAD's value-based
    /// pipeline writes for the same build word — including slots that
    /// pipeline zeroed.
    ///
    /// Optimized MSVC links pass `/OPT:REF,ICF` to link.exe, so identical-
    /// COMDAT folding can give several distinct functions one build address
    /// in `fnaddr_bindings.bin` (and the static-address tables). The
    /// value-based oracle maps that word to whichever binding it last
    /// inserted; the provenance pipeline relocates by the `{ path, symbolic }`
    /// record. A mismatch is allowed only when the build word is bound to
    /// more than one distinct path/name and the new pipeline wrote the
    /// runtime address of the slot's own recorded path.
    #[test]
    fn tagged_reloc_path_matches_old_value_correspondence() {
        let orig = unpatched_production_jitcodes();
        assert_tagged_fnaddr_symbolic_flag_matches_bit_test(&orig);
        let mut old_jcs = unpatched_production_jitcodes();
        let mut new_jcs = unpatched_production_jitcodes();
        old_value_based_patch_pipeline(&mut old_jcs);
        new_provenance_patch_pipeline(&mut new_jcs);

        let colliding = production_colliding_build_words();
        let mut unexpected = Vec::new();
        for ((orig_jc, old_jc), new_jc) in orig.iter().zip(&old_jcs).zip(&new_jcs) {
            let Some(orig_body) = orig_jc.try_body() else {
                continue;
            };
            let old_body = old_jc.try_body().expect("old oracle kept the body");
            let new_body = new_jc.try_body().expect("new pipeline kept the body");
            if let Some(kind) = &orig_jc.fnaddr_reloc
                && old_jc.fnaddr != new_jc.fnaddr
            {
                let tag = match kind {
                    ConstIRelocKind::FnAddr { path, .. } => format!("FnAddr {path}"),
                    ConstIRelocKind::StaticAddr { name } => format!("StaticAddr {name}"),
                };
                note_tagged_reloc_mismatch(
                    tag,
                    format!("fnaddr in {}", orig_jc.name),
                    orig_jc.fnaddr,
                    old_jc.fnaddr,
                    new_jc.fnaddr,
                    runtime_lookup_for_reloc_kind(kind),
                    &colliding,
                    &mut unexpected,
                );
            }
            for desc in &orig_body.reloc_consts_i {
                let idx = desc.constants_i_index;
                let build = orig_body.constants_i[idx];
                let old = old_body.constants_i[idx];
                let new = new_body.constants_i[idx];
                if old == new {
                    continue;
                }
                let tag = match &desc.kind {
                    ConstIRelocKind::FnAddr { path, .. } => format!("FnAddr {path}"),
                    ConstIRelocKind::StaticAddr { name } => format!("StaticAddr {name}"),
                };
                note_tagged_reloc_mismatch(
                    tag,
                    format!("{} constants_i[{idx}]", orig_jc.name),
                    build,
                    old,
                    new,
                    runtime_lookup_for_reloc_kind(&desc.kind),
                    &colliding,
                    &mut unexpected,
                );
            }
            for desc in &orig_body.reloc_consts_r {
                let idx = desc.constants_r_index;
                let build = orig_body.constants_r[idx].get();
                let old = old_body.constants_r[idx].get();
                let new = new_body.constants_r[idx].get();
                if old == new {
                    continue;
                }
                let tag = match &desc.kind {
                    ConstIRelocKind::FnAddr { path, .. } => format!("FnAddr {path}"),
                    ConstIRelocKind::StaticAddr { name } => format!("StaticAddr {name}"),
                };
                note_tagged_reloc_mismatch(
                    tag,
                    format!("{} constants_r[{idx}]", orig_jc.name),
                    build,
                    old,
                    new,
                    runtime_lookup_for_reloc_kind(&desc.kind),
                    &colliding,
                    &mut unexpected,
                );
            }
        }
        unexpected.sort();
        unexpected.dedup();
        assert!(
            unexpected.is_empty(),
            "tagged reloc final value disagrees with the old value-based pipeline: {unexpected:?}"
        );
    }

    /// Two registered paths can share one build word when the host link
    /// folds identical COMDATs (`/OPT:REF,ICF`). The value-based oracle then
    /// relocates that word to whichever binding it last inserted; the
    /// provenance pipeline uses the slot's own path.
    #[test]
    fn tagged_reloc_icf_collision_classifies_oracle_disagreement() {
        let build = 0x1000;
        let runtime_a = 0x2000;
        let runtime_b = 0x3000;
        let unique_build = 0x4000;
        let unique_runtime = 0x5000;
        let colliding = colliding_build_words_from([
            ("path_a", build),
            ("path_b", build),
            ("path_c", unique_build),
        ]);
        assert!(colliding.contains(&build));
        assert!(
            !colliding.contains(&unique_build),
            "a word bound to one path is not an ICF collision"
        );

        let lookup = |path: &str| -> Option<i64> {
            match path {
                "path_a" => Some(runtime_a),
                "path_b" => Some(runtime_b),
                "path_c" => Some(unique_runtime),
                _ => None,
            }
        };
        let slot_path = "path_a";
        let slot_runtime = lookup(slot_path);
        assert_eq!(
            slot_runtime,
            Some(runtime_a),
            "provenance picks the slot's own path"
        );

        // Value-based oracle last-wrote path_b, so old != new.
        let old = runtime_b;
        let new = slot_runtime.expect("slot path published");
        assert_ne!(old, new);
        assert_eq!(
            classify_tagged_reloc_mismatch(build, new, slot_runtime, &colliding),
            TaggedRelocMismatchClass::IcfCollision,
            "value-based disagreement on an ICF-folded word is a collision"
        );

        assert_eq!(
            classify_tagged_reloc_mismatch(
                unique_build,
                unique_runtime.wrapping_add(1),
                Some(unique_runtime),
                &colliding,
            ),
            TaggedRelocMismatchClass::Unexpected,
            "a non-colliding disagreement is reported"
        );
    }

    /// Every bound tagged FnAddr must exact-match a published
    /// `jit_trace_fnaddrs` key, and no residual-call target may be null:
    /// neither as built nor after the runtime patch. A null target is a
    /// build-time binding with no runtime address, which the walker can
    /// only abort on.
    #[test]
    fn tagged_live_fnaddr_paths_resolve_exactly() {
        let orig = unpatched_production_jitcodes();
        let mut old_jcs = unpatched_production_jitcodes();
        old_value_based_patch_pipeline(&mut old_jcs);

        let mut unresolved_live = Vec::new();
        for (orig_jc, old_jc) in orig.iter().zip(&old_jcs) {
            let Some(orig_body) = orig_jc.try_body() else {
                continue;
            };
            let old_body = old_jc.try_body().expect("old oracle kept the body");
            for desc in &orig_body.reloc_consts_i {
                let ConstIRelocKind::FnAddr { path, symbolic } = &desc.kind else {
                    continue;
                };
                let idx = desc.constants_i_index;
                let value = orig_body.constants_i[idx];
                if *symbolic {
                    continue;
                }
                if value == 0 || old_body.constants_i[idx] == 0 {
                    unresolved_live.push(format!(
                        "{path} in {} constants_i[{idx}] is null (built {value:#x})",
                        orig_jc.name
                    ));
                    continue;
                }
                if runtime_fnaddr_for_reloc_path(path).is_none() {
                    unresolved_live.push(format!(
                        "{path} in {} constants_i[{idx}] val={value:#x}",
                        orig_jc.name
                    ));
                }
            }
        }
        for jc in crate::jitcode_runtime::all_jitcodes() {
            let Some(body) = jc.try_body() else {
                continue;
            };
            for desc in &body.reloc_consts_i {
                let ConstIRelocKind::FnAddr { path, .. } = &desc.kind else {
                    continue;
                };
                let idx = desc.constants_i_index;
                if body.constants_i[idx] == 0 {
                    unresolved_live.push(format!(
                        "{path} in {} constants_i[{idx}] is null after the runtime patch",
                        jc.name
                    ));
                }
            }
        }
        unresolved_live.sort();
        unresolved_live.dedup();
        assert!(
            unresolved_live.is_empty(),
            "tagged fnaddr path is null or not a published registry key: {unresolved_live:?}"
        );
    }

    /// A type static a traced body compares against (`is_int` reads
    /// `INT_USER_TYPE`) must reach the constant pool as its runtime address.
    /// One missing from `jit_static_pytype_addrs` stays a symbolic hash, and
    /// every descent through the body that names it is refused.
    #[test]
    fn no_jitcode_constant_is_an_unbound_type_static() {
        let mut unbound = Vec::new();
        for jc in crate::jitcode_runtime::all_jitcodes() {
            let Some(body) = jc.try_body() else {
                continue;
            };
            let constants = body
                .constants_i
                .iter()
                .copied()
                .chain(body.constants_r.iter().map(|slot| slot.get()));
            for c in constants {
                if let Some(path) = symbolic_fnaddr_path(c)
                    && path.ends_with("_TYPE")
                {
                    unbound.push(format!("{} in {}", path, jc.name));
                }
            }
        }
        unbound.sort();
        unbound.dedup();
        assert!(
            unbound.is_empty(),
            "type statics without a runtime address: {unbound:?}"
        );
    }
}
