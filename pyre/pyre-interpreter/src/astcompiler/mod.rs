//! `pypy/interpreter/astcompiler/` — the passes that stand between an AST and
//! a code object.  Parsing and code generation come from the RustPython
//! compiler crates; what lives here is what has no counterpart there.

pub mod consts;
pub mod validate;
