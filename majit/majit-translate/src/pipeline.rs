//! Pipeline result types.
//!
//! Data carriers consumed by codegen + downstream tooling. The
//! production producer is `lib::analyze_pipeline_from_module_paths` (which
//! resolves explicitly configured JIT drivers and compiles the graph closure
//! reachable from their portals).

use serde::{Deserialize, Serialize};

use crate::flatten::SSARepr;
use crate::jtransform::{GraphTransformConfig, GraphTransformNote};
use crate::parse::CallPath;

/// JitDriver portal binding.
///
/// RPython equivalent: `JitDriverStaticData.portal_graph` + the driver's
/// `greens=[...]`/`reds=[...]`/`virtualizables=[...]` declarations
/// (`rlib/jit.py::JitDriver`).
/// `CallControl.setup_jitdriver` consumes this to register the portal
/// entry point and its green/red layout.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JitDriverSpec {
    /// Exact, qualified graph identity of the portal entry point.
    ///
    /// Resolution is deliberately exact: leaf-name fallback makes portal
    /// selection depend on unrelated functions and aliases in the input set.
    pub portal: CallPath,
    /// Exact identity of the synthetic portal runner function.
    ///
    /// RPython: `JitDriverStaticData.portal_runner_ptr`.  This is distinct
    /// from both `portal_graph` and `_jit_merge_point_in`: only direct calls
    /// to the runner classify as recursive in `CallControl.guess_call_kind`.
    #[serde(default)]
    pub portal_runner: Option<CallPath>,
    /// Interpreter-side portal entry (`eval_current_frame_raw` inlines to
    /// this). Same recursive identity as `portal_runner_ptr` after the
    /// crate split; `rewrite_jit_merge_point` still calls `portal_runner`.
    #[serde(default)]
    pub portal_enter: Option<CallPath>,
    pub greens: Vec<String>,
    pub reds: Vec<String>,
    /// Optional operand-kind declarations parallel to `greens`.
    ///
    /// Upstream derives the portal argument kinds from `PORTALFUNC.ARGS` in
    /// warmspot.py make_args_specification. This layer does not carry that portal signature,
    /// so consumers may declare the positional marker kinds and the codewriter
    /// checks them. This is the positional-argument counterpart to the keyword
    /// validation in rlib/jit.py:892-901.
    #[serde(default)]
    pub green_kinds: Vec<majit_ir::Type>,
    /// Optional operand-kind declarations parallel to `reds`. Empty leaves
    /// the check disabled, which keeps auto-red drivers and legacy tests
    /// compatible.
    #[serde(default)]
    pub red_kinds: Vec<majit_ir::Type>,
    /// RPython: `jitdriver.autoreds` — true when the driver declared
    /// `reds='auto'` and the portal marker reds must be discovered by
    /// `autodetect_jit_markers_redvars`.
    #[serde(default)]
    pub autoreds: bool,
    /// Optional explicit virtualizable red names. Empty means no
    /// virtualizable, which matches the common non-pyre case.
    #[serde(default)]
    pub virtualizables: Vec<String>,
    /// Optional red-type identities parallel to `reds`, mirroring the
    /// `_JIT_ENTER_FUNCTYPE.ARGS` information warmspot uses upstream.
    #[serde(default)]
    pub red_types: Vec<String>,
    /// Register the driver against a `warmspot.py
    /// split_graph_and_record_jitdriver` copy of the portal graph, split
    /// before its `jit_merge_point`, instead of against the graph that
    /// contains the marker.
    ///
    /// Default true for a declared-reds portal: `register_configured_jitdrivers`
    /// copies the graph and splits it before `jit_merge_point`. Autoreds
    /// drivers run `autodetect_jit_markers_redvars` first so the split has
    /// a fixed `numreds`, matching `warmspot.py find_portals`.
    #[serde(default)]
    pub split_portal: bool,
}

/// Configuration for the full analysis pipeline.
///
/// RPython: implicit in `CodeWriter.__init__` + `CallControl.__init__`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// jtransform configuration (virtualizable fields, call classification).
    pub transform: GraphTransformConfig,
    /// Explicit JIT-driver bindings consumed by
    /// `CallControl::setup_jitdriver`.
    ///
    /// RPython supports multiple JitDrivers, so the public configuration is
    /// plural even though current Pyre and Boa integrations each register one.
    /// Every public analyzer produces JitCodes, so callers must provide at
    /// least one driver explicitly; there is no consumer-neutral default
    /// portal.
    pub jit_drivers: Vec<JitDriverSpec>,
    /// Opt-in receiver-driven method-dispatch families (issue #346):
    /// qualified `name_path()`s of `>=2`-impl traits whose `dyn Trait`
    /// receivers should annotate to a base `ClassDef` linking the impl
    /// subclasses (so a method getattr resolves the impl `MethodDesc`
    /// family via the attrfamily merge).  Empty for pyre production — its
    /// multi-impl traits keep their classdef-less / fail-loud
    /// disposition; a consumer opts its own traits in by name.
    #[serde(default)]
    pub register_trait_families: Vec<String>,
    /// Graphs seeded into the `find_all_graphs` BFS beside the portals.
    ///
    /// `call.py` seeds `support.inline_calls_to`: helper graphs the
    /// portal never calls in source because the codewriter lowers an
    /// operation straight to a residual call of that helper.  A host that
    /// lowers an opcode to a residual the same way names the residual's
    /// body here, so the graph exists for a trace to descend
    /// (`CallControl::register_helper_graph`).
    #[serde(default)]
    pub helper_graphs: Vec<CallPath>,
    /// Canonical `support.py` `builtin_func_for_spec` impl name → the host
    /// graph that implements it.
    ///
    /// `call.py` `find_all_graphs` walks `support.inline_calls_to` and seeds
    /// `c_func.value._obj.graph` from `builtin_func_for_spec`.
    /// `setup_extra_builtin` names that graph `_ll_<n>_<oopspec>`, the
    /// single-segment path the seed looks up. A host whose graph is
    /// registered under another path lists `(name, source)` here so the
    /// pipeline registers the graph found at `source` under `name`.
    #[serde(default)]
    pub builtin_graphs: Vec<(String, CallPath)>,
}

/// Result of running the full pipeline on a single function.
///
/// RPython: the result of `transform_graph_to_jitcode()` — one per function.
#[derive(Debug, Clone, Serialize)]
pub struct PipelineResult {
    pub name: String,
    pub original_blocks: usize,
    pub annotations_count: usize,
    pub vable_rewrites: usize,
    pub transform_notes: Vec<GraphTransformNote>,
    /// RPython: the SSARepr produced by flatten_graph().
    ///
    /// This stays in-memory only. Build artifacts persist the assembled
    /// JitCodes and driver metadata, not the debug SSA dump.
    #[serde(skip, default)]
    pub flattened: SSARepr,
}

pub use majit_jitcode::artifacts::CompiledJitDriver;

/// Result of running the pipeline on a full program.
#[derive(Debug, Clone, Serialize)]
pub struct ProgramPipelineResult {
    pub functions: Vec<PipelineResult>,
    /// Explicit driver-to-main-JitCode mapping. Consumers should use this
    /// instead of rediscovering portals by scanning JitCode flags or names.
    pub jit_drivers: Vec<CompiledJitDriver>,
    /// RPython: all_jitcodes returned by CodeWriter.make_jitcodes() (codewriter.py).
    /// Assembled JitCode bytecode for each transformed graph. `Arc` so the
    /// shells handed out earlier (e.g. into
    /// `JitDriverStaticData.mainjitcode` or `IndirectCallTargets`) share
    /// identity with the values appearing here.
    pub jitcodes: Vec<std::sync::Arc<crate::jitcode::JitCode>>,
    /// Every symbolic fnaddr placeholder minted while codewriting, paired with
    /// a description of the callee it stands for.  Lets a consumer resolve a
    /// symbolic funcbox constant back to the function it replaced.
    pub symbolic_fnaddr_paths: Vec<(i64, String)>,
    /// RPython `Assembler.indirectcalltargets`, encoded by identity as dense
    /// indices into `jitcodes`.
    #[serde(default)]
    pub indirectcalltarget_indices: Vec<usize>,
    /// RPython: `rpython/jit/codewriter/call.py:87 self.jitcodes`
    /// (graph-keyed dict). Pyre uses `CallPath` as graph identity at the
    /// module boundary. Paired with `jitcodes` (which mirrors
    /// `self.all_jitcodes` from `call.py:88`) so consumers can look up a
    /// JitCode either by alloc-order index or by graph key.
    ///
    /// Skipped by serde because serde_json cannot serialize a map
    /// keyed by a struct (`CallPath` is not a `String`). The
    /// `jit_metadata.json` round-trip used by `pyre-jit-trace/build.rs`
    /// does not need this view — it reads the alloc-ordered `jitcodes`
    /// vector directly. Consumers that require `by_path` read it from
    /// the live in-memory `ProgramPipelineResult`, not from the JSON
    /// artifact.
    #[serde(skip)]
    pub jitcodes_by_path:
        indexmap::IndexMap<crate::parse::CallPath, std::sync::Arc<crate::jitcode::JitCode>>,
    /// RPython: `Assembler.insns` (assembler.py:?). The opcode-key → u8
    /// table grown on-demand by `write_insn`. Persisted alongside the
    /// jitcodes so the runtime can map bytecode bytes back to opnames —
    /// without it, the u8 opcodes embedded in `JitCode.code` are opaque
    /// (the mapping is local to the build-time assembler instance).
    /// Consumed by `BlackholeInterpBuilder::setup_insns` at runtime.
    #[serde(default)]
    pub insns: indexmap::IndexMap<String, u8>,
    /// RPython: `Assembler.descrs` (assembler.py), consumed by
    /// `BlackholeInterpBuilder.setup_descrs(asm.descrs)`
    /// (blackhole.py, 102-103). Each 'd'/'j' argcode in a
    /// `JitCode.code` byte stream indexes into this shared descr pool
    /// to read field offsets / call descrs / sub-JitCodes.
    ///
    /// Persisted alongside `insns` so `BlackholeInterpBuilder` at
    /// runtime can call `setup_descrs(descrs)` and dispatch any 'd'/'j'
    /// argcode opname through the shared pool — matches RPython's
    /// single-store descr model.
    #[serde(default)]
    pub descrs: Vec<crate::jitcode::BhDescr>,
    /// RPython: `Assembler.all_liveness` (assembler.py) — the shared,
    /// deduplicated `(live_i, live_r, live_f)` byte stream that every
    /// `BC_LIVE` op's baked 2-byte offset (in `JitCode.code`) indexes into.
    ///
    /// Persisted alongside `insns`/`descrs` so a runtime consumer re-tracing
    /// a build-time jitcode (whose `BC_LIVE` liveness offsets are baked at
    /// build time) can resolve those offsets through
    /// `metainterp_sd.liveness_info` — the target of `pyjitpl.py:2264
    /// self.liveness_info = "".join(asm.all_liveness)`.
    #[serde(default)]
    pub all_liveness: Vec<u8>,
    /// `CallInfoCollection` rows exported after `make_jitcodes`:
    /// `(oopspecindex, Assembler.descrs index, build-time func address)`.
    #[serde(default)]
    pub callinfo_rows: Vec<majit_ir::effectinfo::CallInfoRow>,
    /// The `(gccache slot key, mint arguments)` pairs behind every descr an
    /// `EffectInfo` raw set names — see `majit_ir::descr::ei_descr_mints_snapshot`.
    ///
    /// `descrs` above is upstream's `opcode_descrs` (`pyjitpl.py
    /// setup_descrs(asm.descrs)`), not its `all_descrs` (`pyjitpl.py
    /// self.cpu.setup_descrs()`, the full gccache walk at `descr.py`).
    /// A descr the analyzer minted only to fill a raw set is named by no
    /// opcode, so it is absent from `descrs` and would have no slot to resolve
    /// against once the process boundary is crossed. Upstream never needs this
    /// because there is one gccache in one process.
    #[serde(default)]
    pub ei_descr_mints: Vec<majit_ir::effectinfo::DescrMintEntry>,
    /// Policy-declined residual callees that still need a callable address.
    /// Consumed by `pyre-jit-trace/build.rs` to emit generated residual shims.
    #[serde(default)]
    pub residual_shim_targets: Vec<crate::residual_shim::ResidualShimTarget>,
    pub total_blocks: usize,
    pub total_ops: usize,
    pub total_vable_rewrites: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::flatten::{FlatOp, SSARepr};
    use crate::flowspace::model::ConstValue;
    use crate::jitcode::JitCode;

    #[test]
    fn serialized_program_pipeline_skips_flattened_ssa_consts() {
        let flattened = SSARepr {
            name: "consts".into(),
            insns: vec![FlatOp::RefReturn(crate::flatten::RegOrConst::Const(
                crate::flowspace::model::Constant::new(ConstValue::byte_str("hello")),
            ))],
            num_blocks: 1,
            insns_pos: None,
        };
        let program = ProgramPipelineResult {
            functions: vec![PipelineResult {
                name: "consts".into(),
                original_blocks: 1,
                annotations_count: 0,
                vable_rewrites: 0,
                transform_notes: Vec::new(),
                flattened: flattened.clone(),
            }],
            jit_drivers: vec![CompiledJitDriver {
                portal: CallPath::from_segments(["eval", "mainloop"]),
                portal_runner: None,
                main_jitcode_index: 0,
                greens: Vec::new(),
                reds: Vec::new(),
                green_args_spec: Vec::new(),
                red_args_types: Vec::new(),
                virtualizables: Vec::new(),
            }],
            jitcodes: vec![Arc::new(JitCode::new("consts"))],
            symbolic_fnaddr_paths: Vec::new(),
            indirectcalltarget_indices: Vec::new(),
            jitcodes_by_path: indexmap::IndexMap::new(),
            insns: indexmap::IndexMap::new(),
            descrs: Vec::new(),
            all_liveness: Vec::new(),
            callinfo_rows: Vec::new(),
            ei_descr_mints: Vec::new(),
            residual_shim_targets: Vec::new(),
            total_blocks: 1,
            total_ops: 1,
            total_vable_rewrites: 0,
        };

        let json = serde_json::to_string(&program).expect("program pipeline should serialize");
        assert!(
            !json.contains("flattened"),
            "serialized artifact should not persist debug SSA payloads"
        );
        assert_eq!(
            serde_json::to_value(&program).unwrap()["jit_drivers"][0]["main_jitcode_index"],
            0,
        );
    }

    #[test]
    fn serialized_pipeline_config_requires_explicit_jit_drivers() {
        let config = PipelineConfig {
            transform: GraphTransformConfig::default(),
            helper_graphs: Vec::new(),
            builtin_graphs: Vec::new(),
            jit_drivers: vec![JitDriverSpec {
                portal: CallPath::from_segments(["engine", "mainloop"]),
                portal_runner: None,
                portal_enter: None,
                greens: Vec::new(),
                reds: Vec::new(),
                green_kinds: Vec::new(),
                red_kinds: Vec::new(),
                autoreds: false,
                virtualizables: Vec::new(),
                red_types: Vec::new(),
                split_portal: false,
            }],
            register_trait_families: Vec::new(),
        };
        let mut value = serde_json::to_value(config).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .remove("jit_drivers")
            .unwrap();

        let error = serde_json::from_value::<PipelineConfig>(value).unwrap_err();
        assert!(
            error.to_string().contains("missing field `jit_drivers`"),
            "missing driver configuration must fail during deserialization: {error}",
        );
    }
}
