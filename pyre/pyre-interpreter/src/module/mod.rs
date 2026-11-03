//! Builtin module registry — PyPy equivalent: pypy/module/
//!
//! Each subdirectory corresponds to a PyPy module package
//! (e.g. `math/` ↔ `pypy/module/math/`).

pub mod __builtin__;
pub mod __pypy__;
pub mod _abc;
pub mod _ast;
pub mod _codecs;
pub mod _collections;
pub mod _contextvars;
pub mod _functools;
pub mod _immutables_map;
pub mod _io;
pub mod _locale;
pub mod _opcode;
pub mod _pypy_generic_alias;
pub mod _sre;
pub mod _stat;
pub mod _structseq;
pub mod _suggestions;
pub mod _symtable;
pub mod _template;
pub mod _types;
pub mod _typing;
pub mod _warnings;
pub mod _weakref;
#[cfg(windows)]
pub mod _winreg;
pub mod atexit;
pub mod errno;
#[cfg(all(not(target_arch = "wasm32"), not(feature = "sandbox")))]
pub mod faulthandler;
pub mod imp;
pub mod importlib;
pub mod itertools;
pub mod marshal;
pub mod operator;
pub mod posix;
#[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
pub mod pwd;
pub mod pypyjit;
pub mod signal;
pub mod r#struct;
pub mod sys;
pub mod thread;
pub mod time;
