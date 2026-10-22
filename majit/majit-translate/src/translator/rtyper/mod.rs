//! `translator/rtyper/` — ports of `rpython/rtyper/`.
//!
//! Same-stem modules mirror upstream `rpython/rtyper/*.py` by name and
//! structure (`rclass.py` -> `rclass.rs`, `rpbc.py` -> `rpbc.rs`).
//! Local Rust boundaries are kept only where pyre has a second
//! graph model or a Rust-only frontend surface:
//!
//! * `flowspace_adapter`, `cutover`, `legacy_annotator`, and
//!   `legacy_resolve` are transitional bridges between pyre's legacy
//!   `model::FunctionGraph` and the orthodox
//!   `flowspace::FunctionGraph` / `RPythonTyper` path.
//! * `call_registry` owns symbolic `FunctionPath` -> synthetic
//!   `HostObject` / `FunctionDesc` registration, because pyre has no
//!   CPython callable object identity to key `Bookkeeper.descs`.
//! * `pairtype` centralizes rtyper-side `class __extend__(pairtype(...))`
//!   blocks that Python's metaclass machinery wires implicitly upstream;
//!   the actual `rpython/tool/pairtype.py` port is
//!   [`crate::tool::pairtype`].
//! * `unit_variant_fold` pre-folds Rust unit-variant constructors into
//!   prebuilt PBC-like constants, matching the effect of upstream
//!   frozen-PBC/instance-repr lowering before `jtransform`.
//! * `str_const_fold` resolves synthetic string-literal calls to the
//!   prebuilt-string constant shape produced by `StringRepr.convert_const`.
//! * `box_str_const_fold` removes redundant boxing calls whose string view is
//!   proven to come from that prebuilt-string constant.
//!
//! `lltypesystem::ll2ctypes`, `lltypesystem::llarena`, and
//! `tool::rffi_platform` are intentionally absent; their module roots
//! document why those C/backend-GC probing or simulation layers are
//! permanently unused in pyre.

pub mod annlowlevel;
pub(crate) mod box_str_const_fold;
pub(crate) mod call_registry;
pub mod callparse;
pub mod controllerentry;
pub(crate) mod cutover;
pub mod debug;
pub mod error;
pub mod exceptiondata;
pub mod extfunc;
pub mod extfuncregistry;
pub mod extregistry;
pub(crate) mod flowspace_adapter;
pub mod legacy_annotator;
pub mod legacy_resolve;
pub mod llannotation;
pub mod llinterp;
pub mod lltypesystem;
pub mod normalizecalls;
pub(crate) mod pairtype;
pub mod raddress;
pub mod rbool;
pub mod rbuilder;
pub mod rbuiltin;
pub mod rbytearray;
pub mod rclass;
pub mod rdict;
pub mod rfloat;
pub mod rint;
pub mod rlist;
pub mod rmodel;
pub mod rnone;
pub mod rpbc;
pub mod rptr;
pub mod rrange;
pub mod rrustvec;
pub mod rstr;
pub mod rtuple;
#[expect(
    clippy::module_inception,
    reason = "module names mirror the corresponding RPython package and file layout for line-by-line parity"
)]
pub mod rtyper;
pub mod rvirtualizable;
pub mod rweakref;
pub(crate) mod str_const_fold;
pub mod tool;
pub(crate) mod unit_variant_fold;
