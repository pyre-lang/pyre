//! `rpython/rlib` — the runtime support library RPython programs are written
//! against, as opposed to the translator that compiles them (`majit-translate`)
//! or the collector that runs under them (`majit-gc`).
//!
//! A module here is a port of an upstream file and is named after it, so
//! `rpython/rlib/rbigint.py` is [`rbigint`]. The one exception is
//! [`lltypesystem`], which carries the `rpython/rtyper/lltypesystem` bodies that
//! an `rlib` module's own storage is lowered to: `rbigint._digits` is a list
//! whose low-level form is a `Ptr(GcArray(Signed))`, and that body cannot live
//! in a crate above the module that allocates it. The module path spells the
//! upstream package it came from.

// `#[look_inside_iff]` expansions name `majit_rlib::jit`.
extern crate self as majit_rlib;

#[cfg(any(unix, windows))]
#[allow(non_snake_case)]
pub mod _rsocket_rffi;

pub mod cache;
#[cfg(any(unix, windows))]
pub mod clibffi;
pub mod debug;
pub mod jit;
pub mod jit_libffi;
pub mod lltypesystem;
pub mod nonconst;
pub mod rarithmetic;
pub mod rbigint;
#[cfg(any(unix, windows))]
pub mod rdynload;
pub mod rffi;
#[allow(non_snake_case)]
pub mod rlocale;
#[cfg(unix)]
pub mod rmmap;
#[cfg(unix)]
pub mod rpoll;
pub mod rposix;
#[cfg(unix)]
pub mod rposix_environ;
#[cfg(unix)]
pub mod rposix_stat;
#[cfg(unix)]
pub mod rsignal;
#[cfg(any(unix, windows))]
pub mod rsocket;
pub mod rstring;
#[cfg(unix)]
pub mod rtermios;
pub mod rthread;
#[cfg(unix)]
pub mod rtime;
#[cfg(unix)]
pub mod rurandom;
pub mod rvmprof;
#[cfg(target_os = "windows")]
#[allow(non_snake_case)]
pub mod rwin32;
#[cfg(target_os = "windows")]
#[allow(non_snake_case)]
pub mod rwin32file;

/// `lltype.malloc(T, flavor='raw')` (`rpython/rtyper/lltypesystem/lltype.py`) —
/// an allocation the collector neither traces nor reclaims.
///
/// Kept here rather than in a `lltypesystem::lltype` module because this is the
/// whole of that file this crate uses: the raw-flavor allocations rbigint falls
/// back to when no GC owns the heap.
pub fn malloc_raw<T>(value: T) -> *mut T {
    Box::into_raw(Box::new(value))
}
