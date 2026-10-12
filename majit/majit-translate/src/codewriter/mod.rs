//! Ports of `rpython/jit/codewriter/`.
//!
//! Contains the low-level codewriter stage that converts an rtyped graph
//! into `JitCode` consumed by `majit-metainterp`. Sibling modules in this
//! directory that correspond to upstream files keep the upstream stem
//! (`assembler.py` -> `assembler.rs`, `jtransform.py` -> `jtransform.rs`,
//! and so on).
//!
//! A few files are local Rust boundaries, not missing upstream ports:
//! `annotation_state`, `insns`, `jtransform_opname`,
//! `jtransform_shadow`, `transform_profile`, and `type_state`.
//! Each of those modules documents the upstream surface it adapts or the
//! diagnostic role it owns. Do not treat them as candidates for blind
//! deletion solely because `rpython/jit/codewriter/` has no same-named
//! Python file.

// Local Rust boundary: `ValueType` to `SomeValue` projection used while
// the real annotator/rtyper cutover still bridges legacy graphs.
pub(crate) mod annotation_state;
pub mod assembler;
pub mod call;
#[expect(
    clippy::module_inception,
    reason = "module names mirror the corresponding RPython package and file layout for line-by-line parity"
)]
pub mod codewriter;
pub use codewriter::{AllJitCodes, CodeWriter};
pub mod effectinfo;
// Local Rust boundary: the error carrier's exception edges converted to the
// runtime exception value, after annotation (see the module doc).
pub mod error_carrier_edges;
pub mod flatten;
pub mod format;
pub mod heaptracker;
// Local Rust boundary for the stable byte table derived from
// `assembler.py:Assembler.insns`; pyre serializes bytecode across build
// and runtime, so the dynamic upstream table is materialized here.
pub use majit_jitcode::codewriter::insns;
// `[v; N]` for the rich-`OpKind` spine — mints `ll_fixed_alloc_and_set`.
pub mod alloc_and_set;
// `l[start:]` for the rich-`OpKind` spine — mints `ll_listslice_startonly`.
pub mod getslice;
// `core::cmp::{min,max}` for the rich-`OpKind` spine — mints `ll_min` / `ll_max`.
pub mod iter_lower;
pub use majit_jitcode::codewriter::jitcode;
pub mod jtransform;
pub mod longlong;
pub mod minmax;
// Opname-dispatch transducer ("Spine B"): lowers rtyper low-level helper
// graphs (opname `SpaceOperation`s) to rich-`OpKind` graphs that re-enter the
// shared flatten/regalloc/assembler tail. Port of `jtransform.py`'s
// `_rewrite_ops[op.opname]` dispatch; see the module docs.
pub(crate) mod jtransform_opname;
// No upstream sibling: an inert, env-gated diagnostic that gauges how much
// of the rtyped flowspace graph an opname-dispatching jtransform would
// already accept. Never on the production path; see the module docs.
pub(crate) mod jtransform_shadow;
pub mod liveness;
pub mod policy;
pub mod regalloc;
pub mod support;
// Local env-gated profiler for the drain pipeline, with no upstream
// sibling and no effect unless `MAJIT_PROFILE_DRAIN` is set.
pub(crate) mod transform_profile;
// Local Rust boundary for concretetype projection and temporary import
// compatibility while concretetype data migrates onto Variables.
pub(crate) mod type_state;
