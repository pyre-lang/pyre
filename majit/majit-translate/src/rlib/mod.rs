//! `rlib` — Rust port of `rpython/rlib/` helpers.
//!
//! Only the subset required by downstream annotator / rtyper ports
//! lands; new submodules appear as their dependencies are pulled in.
//!
//! | majit                 | rpython/rlib/                 |
//! |-----------------------|-------------------------------|
//! | `entrypoint`          | `entrypoint.py`               |
//! | `jit`                 | `jit.py` (marker/extregistry half; the user hint API lives at `majit_metainterp::jit`) |
//! | `rarithmetic`         | `rarithmetic.py`              |
//! | `rstring`             | `rstring.py`                  |
//! | `rutf8`               | `rutf8.py`                    |

pub mod entrypoint;
pub mod jit;
pub mod rarithmetic;
pub mod rstring;
pub mod rutf8;
