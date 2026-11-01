//! `tool` — Rust port of `rpython/tool/`.
//!
//! Upstream `tool/` collects small, reusable helpers used across
//! `rpython/annotator/`, `rpython/rtyper/`, and `rpython/translator/`.
//! Only the subset required by the downstream ports lands; new
//! submodules appear as their dependencies are pulled in.
//!
//! | majit                 | rpython/tool/                 |
//! |-----------------------|-------------------------------|
//! | `algo::unionfind`     | `algo/unionfind.py`           |
//! | `ansi_print`          | `ansi_print.py`               |
//! | `error`               | `error.py`                    |
//! | `flattenrec`          | `flattenrec.py`               |
//! | `gcc_cache`           | `gcc_cache.py`                |
//! | `pairtype`            | `pairtype.py`                 |
//! | `sourcetools`         | `sourcetools.py`              |
//! | `udir`                | `udir.py`                     |

pub mod algo;
pub mod ansi_print;
pub mod error;
pub mod flattenrec;
pub mod gcc_cache;
pub mod pairtype;
pub mod sourcetools;
pub mod udir;
