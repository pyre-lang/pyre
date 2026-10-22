//! `rpython/rtyper/lltypesystem` — the low-level forms an RPython value is
//! lowered to. Only the bodies this crate's own modules allocate live here; the
//! rest of the lltypesystem stays with the translator, except the C
//! `llexternal`s of [`module`], whose addresses the JIT calls at run time.

pub mod llmemory;
pub mod module;
pub mod rffi;
pub mod rlist;
pub mod rstr;
pub mod rvec;
