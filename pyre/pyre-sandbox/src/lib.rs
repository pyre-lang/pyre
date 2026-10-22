//! RPython-style sandbox for pyre.
//!
//! This crate is the single live home of the sandbox protocol shared by the
//! untrusted client (compiled into `pyre --features sandbox`) and the trusted
//! controller (`pyre interact`). It ports, from the RPython/PyPy source tree:
//!
//!   - `rpython/rlib/rmarshal.py` + `rpython/translator/sandbox/_marshal.py`
//!     -> `majit_rlib::rmarshal` (the byte-exact wire codec),
//!   - `rpython/translator/sandbox/rsandbox.py` (runtime half)
//!     -> `majit_rlib::rsandbox` (the trampoline body),
//!   - `rpython/translator/sandbox/vfs.py` -> [`vfs`],
//!   - `rpython/translator/sandbox/sandlib.py` -> [`sandlib`],
//!   - `pypy/sandbox/pypy_interact.py` -> [`controller`].
//!
//! The translator-side shells under `majit-translate/.../sandbox/` stay as inert
//! RPython structural-parity mirrors; this crate is the runtime implementation.
//!
//! # Structural constraint — the guarantee model without a genc backend
//!
//! RPython's `--sandbox` earns a *whole-program, compiler-derived* proof that no
//! syscall instruction survives in the shipped binary: genc emits the entire
//! interpreter + runtime + GC to C from one exhaustive database traversal and,
//! in that same pass, replaces every external funcptr's graph with a fd1/fd0
//! marshal stub (`rpython/translator/c/node.py:new_funcnode`,
//! `rffi.py:llexternal(sandboxsafe=...)`). The property is total *by
//! construction* — an external call is an explicit, `sandboxsafe`-tagged graph
//! node, and nothing reachable escapes the traversal.
//!
//! pyre has no genc and cannot reproduce that mechanism: the shipped binary is
//! built by rustc/cargo, `majit-translate` is a JIT that only compiles hot
//! *guest* bytecode traces (never the interpreter itself), and an external call
//! is an ordinary `direct_call` funcptr with no `sandboxsafe` tag for a pass to
//! intercept. The translator-side `sandbox/` shells stay inert precisely because
//! they presume that missing backend. So the "no syscall in the binary" property
//! here is **not derived from translation**; it is reconstructed from two layers
//! of a different shape, and the gap between them and genc *is* the boundary to
//! keep in mind:
//!
//!   - **Compile time — selective, not total.** The `host_seam` choke-point, the
//!     fails-closed `host_seam::sys` facade, and the CI clippy fence make a stray
//!     `libc::*` / `std::{fs,env,io,net,process}` call a compile/CI error. This
//!     is the closest analog to genc, but it is only as complete as the seam and
//!     the fence's *enumerated* surface: a host-access path that names neither
//!     `libc` nor a fenced `std` item — a new dependency's own FFI, a raw
//!     `syscall!`, an un-rerouted call site — still compiles. genc's proof is
//!     exhaustive; this one rests on the fence staying complete as code grows.
//!
//!   - **Runtime — total, but platform-bound.** [`seccomp`] is the only
//!     whole-program guarantee pyre actually holds: the kernel refuses every
//!     non-allowlisted syscall regardless of what is in the binary, so it also
//!     covers what the source layer cannot (the linked `host_env` crate, Rust
//!     std, an un-rerouted site, a syscall reached by a memory-safety exploit).
//!     But it is Linux-only (`#![cfg(target_os = "linux")]`), rides the opt-in
//!     `sandbox` feature, can be disabled with `PYRE_SANDBOX_NO_SECCOMP`, and the
//!     trusted `interact` controller is exempt by design.
//!
//! **Threat-model bottom line.** On Linux with seccomp active, the child has a
//! genuine kernel-enforced whole-program boundary *despite* the missing genc —
//! it is not "unsafe without genc". Off Linux, or with seccomp disabled, there
//! is no kernel backstop and containment is only as strong as the compile-time
//! seam + fence coverage; the residual exposure (a syscall reached outside the
//! seam, or via unsafe memory corruption) is exactly what genc's whole-program C
//! emission would have closed *structurally*, and it cannot be closed here
//! without re-architecting pyre into an ahead-of-time translator. Widening the
//! child's trusted surface (new host APIs, new dependencies compiled into it)
//! must be weighed against the fence and seccomp allowlist — never assumed away
//! by "it's translated".

// The controller, the policy/dispatch in sandlib, and the virtual filesystem
// drive a fork/fd-based sandbox via unix-only syscalls (kill/SIGKILL, arg0,
// O_ACCMODE, pointer-sized read/write); they are unix-only. The wire codec
// and trampoline client live in `majit_rlib::{rmarshal,rsandbox}` under the
// same unix gate.
#[cfg(all(unix, test))]
mod client;
#[cfg(unix)]
pub mod controller;
#[cfg(unix)]
pub mod sandlib;
// OS-level seccomp backstop for the sandboxed child (Linux only).
#[cfg(target_os = "linux")]
pub mod seccomp;
#[cfg(unix)]
pub mod vfs;
