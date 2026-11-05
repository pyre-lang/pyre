//! `translator/rtyper/tool/` — RPython-orthodox
//! `rpython/rtyper/tool/` counterparts.

pub mod gcstat;
pub mod mkrffi;
// `rffi_platform` is not a translator module: the translator reads Rust
// types and Charon-extracted layouts, and asks no C compiler. Build
// scripts ask theirs through `majit_eci::rffi_platform`.
pub mod rfficache;
