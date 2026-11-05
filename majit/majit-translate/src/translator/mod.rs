//! `translator/` — RPython-orthodox port files that mirror upstream
//! `rpython/rtyper/` and `rpython/annotator/`.
//!
//! Files in this tree must use RPython-orthodox structure (file
//! names, function names, control flow). The legacy `translate_legacy/`
//! subtree it once paired with was deleted; the residual
//! adapter walker that still drives the dual-gate Skip arm now lives
//! in [`rtyper::legacy_annotator`] / [`rtyper::legacy_resolve`]
//! inside this same tree.  Both `majit-annotator` and `majit-rtyper`
//! standalone crates remain on the roadmap.

pub mod backend;
pub mod backendopt;
pub mod driver;
pub mod gensupp;
pub mod goal;
pub mod interactive;
pub use majit_eci::platform;
pub mod revdb;
pub mod rtyper;
pub mod sandbox;
pub mod simplify;
pub mod targetspec;
pub mod timing;
pub mod tool;
pub mod transform;
#[expect(
    clippy::module_inception,
    reason = "module names mirror the corresponding RPython package and file layout for line-by-line parity"
)]
pub mod translator;
pub mod unsimplify;
