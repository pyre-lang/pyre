//! The names this crate matches by spelling, in one place.
//!
//! `majit-translate` is a line-by-line port of `rpython/{flowspace,annotator,
//! rtyper,translator}` plus a Rust front end, and pyre is the only consumer
//! that drives it. Two families of name follow from that: the intrinsic shims
//! pyre exports for the flowspace HOST_ENV, and the crate, module and symbol
//! spellings its extracted LLBC carries.
//!
//! `scripts/check-majit-boundary.py` keeps runtime-owned *identifiers* out of
//! this subtree, but it reads Rust tokens and deliberately skips string
//! literals — so the names below are exactly the ones it cannot see. That is
//! the failure this module is for: a shim's name is a contract with whoever
//! exports it, and a contract written out at every site is one that can be
//! half-changed. `__cast_instance_intrinsic` is registered in the HOST_ENV,
//! given an annotator, matched by the rtyper and asserted in a dozen fixtures,
//! and a rename that misses one of those still builds — the lookup simply
//! misses and the operation falls back, which is the failure mode nothing
//! counts.
//!
//! So the names live here and the rest of the crate refers to them. That does
//! not decouple anything, and it is not meant to: [`crate::front`] says
//! outright that it is the pyre-specific layer. A second consumer would seed
//! these the way [`crate::local_crates`] already seeds crate roots from the
//! loaded LLBC set — there is no second consumer today, so that machinery is
//! not written, and the list below is what it would take.
//!
//! # What deliberately stays outside this module
//!
//! The crate still says "pyre" elsewhere, in three kinds. None of them stays
//! because the name is upstream: a case-insensitive `pyre` over an RPython
//! checkout matches nothing at all. Each kind stays for its own reason.
//!
//! **Path matchers under [`crate::front`].** Thirty literals matching a
//! `pyre_object::` / `pyre_interpreter::` MIR path directly. That module's doc
//! states that every file under it is Rust-specific lowering with no RPython
//! structural match, and its maintenance rule is to justify the deviation
//! rather than avoid it. Naming such a path is what the layer is for; a
//! `front/` that named none would not be doing its job.
//!
//! **Diagnostic text.** The error and panic messages that spell a shim or a
//! crate, plus two skip labels (`callee-pyre-class-ctor`,
//! `skip-pyre-class-allocate-ctor`). These are not the failure this module
//! exists to prevent. The lookup sites moved because a lookup that misses is
//! SILENT — the operation falls back and nothing counts it. A message is read
//! by whoever hits it, so a stale one announces itself; routing it through a
//! constant would buy consistency, not detection.
//!
//! **Environment gate names.** `PYRE_MIR_FRONTEND_DEBUG`, read by
//! [`crate::decline`]. This kind must not move, for a mechanical reason rather
//! than a stylistic one. `gate_triage_complete`'s `NAMESPACES` maps the
//! `PYRE_` prefix to `pyre/gate-triage.md` and checks both directions by
//! scanning source for the name as a quoted literal inside the `env::var` call
//! itself; its `gates_read_by` says outright that a gate held in a Rust const
//! does not count as live. Routing one through a constant here would not
//! rename anything — it would drop the gate out of the completeness check
//! while leaving it live — the same silent direction the shim names in this
//! module exist to escape.
//!
//! Rust item names are not on that list because this crate does not police
//! them: `scripts/check-majit-boundary.py` rejects a `pyre`-spelled identifier
//! or path component anywhere under `majit/`. That is a check rather than a
//! convention, and it reaches further than this module could — the jitcode keys
//! are spelled `*_ext/P` on both sides of the wire format because of it.

/// LLBC artefacts the fixtures load.
///
/// Built with `concat!` at the constant rather than at each site: `concat!`
/// takes literals, so a site that wants the path cannot compose it from a
/// `const &str` and would have to spell the tail again.
///
/// These are `build/` outputs, so a fixture naming one is `#[ignore]`d — it
/// asserts against an artefact that a plain `cargo test` has no reason to have
/// produced. Nothing outside a test names one, which is what the gate records:
/// a path into `build/` appearing in a non-test build would be this crate
/// reaching for an artefact at run time, and it does not.
#[cfg(test)]
pub(crate) mod artifacts {
    pub(crate) const INTERPRETER_ULLBC: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../build/llbc/pyre-interpreter.ullbc"
    );
    pub(crate) const OBJECT_ULLBC: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../build/llbc/pyre-object.ullbc"
    );
    pub(crate) const MODULE_ULLBC: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../build/llbc/pyre-module.ullbc"
    );
    pub(crate) const JIT_ULLBC: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../build/llbc/pyre-jit.ullbc"
    );
    pub(crate) const MAJIT_RLIB_ULLBC: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../build/llbc/majit-rlib.ullbc"
    );
    pub(crate) const CHARON_CORPUS_ULLBC: &str =
        concat!(env!("CARGO_MANIFEST_DIR"), "/../charon-corpus/corpus.ullbc");
}

/// Host adapter spellings — callables the consumer exports for the flowspace
/// HOST_ENV when Rust source has no directly importable RPython callable.
/// Most are translation-only shims. `LL_ARRAYMOVE` is the one exception: it
/// names a callable RPython does have (`rpython/rtyper/rlist.py`,
/// `ll_arraymove`, wrapping `rgc.ll_arraymove`), reached here through a
/// proven adapter boundary because neither `rlist.ll_arraymove` nor its
/// caller `ll_delitem_nonneg` is built yet. The restoration reaches the
/// rtyper, oopspec, and codewriter boundary: the minted helper is marked
/// `list.ll_arraymove`, and `_handle_list_call` emits OS_ARRAYMOVE (9) while
/// retaining this adapter only as the source-level restoration seam.
///
/// Registered in `flowspace::model`, given an annotator in
/// `annotator::builtin`, and resolved in `translator::rtyper`. All three must
/// agree, which is the whole reason these are constants.
pub(crate) mod shims {
    pub(crate) const CAST_INSTANCE: &str = "__cast_instance_intrinsic";
    pub(crate) const CAST_ADDRESS: &str = "__cast_address_intrinsic";
    pub(crate) const RANGE: &str = "__majit_range";
    /// `enumerate(lst)`: the front respells `lst.iter().enumerate()` as
    /// `iter(__majit_enumerate(lst))`.
    pub(crate) const ENUMERATE: &str = "__majit_enumerate";
    pub(crate) const STRINGBUILDER_NEW: &str = "__majit_stringbuilder_new";
    pub(crate) const STRINGBUILDER_APPEND: &str = "__majit_stringbuilder_append";
    pub(crate) const STRINGBUILDER_BUILD: &str = "__majit_stringbuilder_build";
    /// Rust `ptr::copy` at a proven list-storage adapter boundary, restored
    /// to RPython `rgc.ll_arraymove(array, source_start, dest_start, length)`.
    pub(crate) const LL_ARRAYMOVE: &str = "__majit_ll_arraymove";
    /// Prefix, not a whole name: the wrapping arithmetic shims are
    /// `__majit_wrap_<op>`, matched by prefix where the op is not known.
    pub(crate) const WRAP_PREFIX: &str = "__majit_wrap_";
}

/// Dotted module paths the HOST_ENV registers the runtime's lltype surface
/// under.
pub(crate) mod modules {
    /// The consumer's object-model module. Its leaf spells the object model
    /// itself, which `scripts/check-majit-boundary.py` rejects as a Rust
    /// identifier anywhere under `majit/`, so the leaf lives only in the
    /// value.
    pub(crate) const OBJECT_MODEL: &str = "pyre_object.pyobject";
    pub(crate) const OBJECT_LLTYPE: &str = "pyre_object.lltype";
    pub(crate) const MALLOC: &str = "pyre_object.lltype.malloc";
    pub(crate) const MALLOC_TYPED: &str = "pyre_object.lltype.malloc_typed";
    pub(crate) const MALLOC_TYPED_MANAGED: &str = "pyre_object.lltype.malloc_typed_managed";
    pub(crate) const MALLOC_TYPED_STABLE: &str = "pyre_object.lltype.malloc_typed_stable";
    pub(crate) const MALLOC_RAW: &str = "pyre_object.lltype.malloc_raw";
    /// `majit_gc::header` varsize immortal helpers (`init_gc_object_immortal`
    /// / `malloc_varsize` analogue). Concrete, so they need the same HOST_ENV
    /// residual as `malloc_typed` rather than a look-inside graph.
    pub(crate) const GC_HEADER: &str = "majit_gc.header";
    pub(crate) const ALLOC_VARSIZE_GC_HEADER_IMMORTAL: &str =
        "majit_gc.header.alloc_varsize_with_gc_header_immortal";
    pub(crate) const ALLOC_VARSIZE_GC_HEADER_IMMORTAL_ZEROED: &str =
        "majit_gc.header.alloc_varsize_with_gc_header_immortal_zeroed";
    pub(crate) const DEALLOC_VARSIZE_GC_HEADER: &str =
        "majit_gc.header.dealloc_varsize_with_gc_header";
}

/// The crate names the consumer's LLBC set extracts under.
///
/// [`crate::local_crates`] seeds its roots from the loaded set's own
/// `crate_name()`s, so these are not a fallback list; they are the spellings
/// this crate matches against directly, in fixtures and in the portal path.
pub(crate) mod crates {
    pub(crate) const OBJECT: &str = "pyre_object";
    pub(crate) const INTERPRETER: &str = "pyre_interpreter";
    /// Test-gated: the portal graph resolves against the loaded LLBC set's
    /// own crate names, so only fixtures that build a portal path by hand
    /// spell this root.
    #[cfg(test)]
    pub(crate) const JIT: &str = "jit_artifact";
}

/// Fully-qualified runtime symbols this crate matches on by name.
pub(crate) mod symbols {
    /// The frame constructor the GC transform recognises to place a stack root
    /// map. Spelled with the `<Impl>` the extractor emits, not a normalised
    /// form — this is matched against Charon output verbatim.
    ///
    /// Test-gated because only the transform's fixtures name it: the
    /// production path matches a graph it was handed rather than a spelling.
    #[cfg(test)]
    pub(crate) const PYFRAME_NEW: &str = "pyre::pyframe::<Impl>::new";
    /// The wrapping-add helper `rpbc` registers a graph for. Test-gated for
    /// the same reason as [`PYFRAME_NEW`].
    #[cfg(test)]
    pub(crate) const WRAP_ADD: &str = "__majit_wrap_add";
    /// Dotted spelling of the frame-block class, as the annotator's class
    /// registry keys it. Test-gated: the production lookup goes through the
    /// registry rather than this spelling, and only the fixtures name it.
    #[cfg(test)]
    pub(crate) const FRAMEBLOCK: &str = "pyre_interpreter.pyframe.FrameBlock";
}
