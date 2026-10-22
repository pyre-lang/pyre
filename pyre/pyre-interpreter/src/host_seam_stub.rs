//! Shared `host_seam::stub` / `catch_sandbox_stub` bodies.
//!
//! Both the unix `host_seam` module and the non-unix inline `host_seam`
//! re-export these so each name is defined once.

/// The not-implemented stub for OS surface that the sandbox controller does not
/// service (signal/socket/dup/ftruncate/…). Port of `rsandbox.py`'s
/// `get_sandbox_stub`/`not_implemented_stub`: raise `RuntimeError` rather than
/// touch the OS (`not_implemented_stub` does `raise RuntimeError(msg)`).
#[cfg(any(not(feature = "host_env"), feature = "sandbox"))]
pub fn stub(fnname: &str) -> crate::PyError {
    crate::PyError::runtime_error(format!(
        "Not implemented: sandboxing for external function '{fnname}'"
    ))
}

/// Catch a rustc-path [`majit_rlib::rffi::SandboxStub`] panic and turn it into
/// [`stub`]. Default (host_env, not sandbox) is a plain call-through.
#[cfg(any(not(feature = "host_env"), feature = "sandbox"))]
pub fn catch_sandbox_stub<F>(f: F) -> Result<pyre_object::PyObjectRef, crate::PyError>
where
    F: FnOnce() -> Result<pyre_object::PyObjectRef, crate::PyError>,
{
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(result) => result,
        Err(payload) => match payload.downcast::<majit_rlib::rffi::SandboxStub>() {
            Ok(payload) => Err(stub(payload.fnname)),
            Err(payload) => {
                #[cfg(all(unix, feature = "sandbox"))]
                {
                    match payload.downcast::<majit_rlib::rsandbox::SandboxError>() {
                        Ok(err) => {
                            return Err(crate::host_seam::seam_os_err((*err).into(), ""));
                        }
                        Err(payload) => std::panic::resume_unwind(payload),
                    }
                }
                #[cfg(not(all(unix, feature = "sandbox")))]
                {
                    std::panic::resume_unwind(payload)
                }
            }
        },
    }
}

#[cfg(all(feature = "host_env", not(feature = "sandbox")))]
#[inline(always)]
pub fn catch_sandbox_stub<F>(f: F) -> Result<pyre_object::PyObjectRef, crate::PyError>
where
    F: FnOnce() -> Result<pyre_object::PyObjectRef, crate::PyError>,
{
    f()
}
