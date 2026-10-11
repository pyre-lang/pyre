//! _posixshmem module — PyPy: `lib_pypy/_posixshmem.py`.
//!
//! Backs `multiprocessing.shared_memory` on POSIX.  Entire surface is
//! gated on `cfg(feature = "host_env")`; `host_env = off` builds expose
//! an empty module so `import _posixshmem` still succeeds (matching
//! PyPy's mixedmodule behaviour when the conditional `interpleveldefs`
//! entry is absent).
//!
//! `host_env` calls `c_shm_open` / `c_shm_unlink`.

use pyre_object::*;

/// `lib_pypy/_posixshmem_build.py` includes and libc calls:
/// `includes=['sys/mman.h', 'sys/stat.h', 'fcntl.h']`, C names `shm_open` /
/// `shm_unlink`, `releasegil=False`, no `save_err`. Darwin links nothing;
/// every other POSIX target links `rt` (`libraries`).
#[cfg(feature = "host_env")]
mod ll {
    use majit_rlib::rffi::{CCHARP, INT, UINT};

    #[cfg(target_vendor = "apple")]
    majit_rlib::rffi::external_compilation_info! {
        const ECI = {
            includes: ["sys/mman.h", "sys/stat.h", "fcntl.h"],
        };
    }
    #[cfg(not(target_vendor = "apple"))]
    majit_rlib::rffi::external_compilation_info! {
        const ECI = {
            includes: ["sys/mman.h", "sys/stat.h", "fcntl.h"],
            libraries: ["rt"],
        };
    }

    majit_rlib::rffi::llexternal!(
        pub(super) c_shm_open = "shm_open",
        [CCHARP, INT, UINT],
        INT,
        compilation_info = ECI,
        releasegil = false
    );
    majit_rlib::rffi::llexternal!(
        pub(super) c_shm_unlink = "shm_unlink",
        [CCHARP],
        INT,
        compilation_info = ECI,
        releasegil = false
    );
}

#[cfg(feature = "host_env")]
fn shm_open(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    if !(2..=3).contains(&args.len()) {
        return Err(pyre_interpreter::PyError::type_error(
            "shm_open() requires (path, flags[, mode])",
        ));
    }
    let mut w_path = args[0];
    let mut w_flags = args[1];
    let mut w_mode = args.get(2).copied().unwrap_or(PY_NULL);
    let name = unsafe {
        if !is_str(w_path) {
            return Err(pyre_interpreter::PyError::type_error(
                "shm_open: path must be a string",
            ));
        }
        pyre_object::with_roots!(w_path, w_flags, w_mode =>
            pyre_interpreter::baseobjspace::str_utf8_w(w_path)
        )?
        .to_string()
    };
    let flags = (unsafe { w_int_get_value(w_flags) }) as libc::c_int;
    let mode = if args.len() >= 3 {
        (unsafe { w_int_get_value(w_mode) }) as libc::c_uint
    } else {
        0o600
    };
    let c_name = std::ffi::CString::new(name.as_bytes())
        .map_err(|_| pyre_interpreter::PyError::value_error("embedded null character"))?;
    // `lib_pypy/_posixshmem.py shm_open` retries on EINTR.
    let fd = loop {
        let fd = pyre_object::with_roots!(w_path, w_flags, w_mode => unsafe {
            ll::c_shm_open(c_name.as_ptr() as majit_rlib::rffi::CCHARP, flags, mode)
        });
        if fd < 0 {
            let errno = majit_rlib::rposix::_get_errno();
            if errno == libc::EINTR {
                continue;
            }
            let e = std::io::Error::from_raw_os_error(errno);
            return Err(pyre_interpreter::PyError::os_error_with_errno(
                errno,
                format!("shm_open: {e}"),
            ));
        }
        break fd;
    };
    Ok(w_int_new(fd as i64))
}

#[cfg(feature = "host_env")]
fn shm_unlink(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    if args.is_empty() {
        return Err(pyre_interpreter::PyError::type_error(
            "shm_unlink() needs path",
        ));
    }
    let mut w_path = args[0];
    let name = unsafe {
        if !is_str(w_path) {
            return Err(pyre_interpreter::PyError::type_error(
                "shm_unlink: path must be a string",
            ));
        }
        pyre_object::with_roots!(w_path => pyre_interpreter::baseobjspace::str_utf8_w(w_path))?
            .to_string()
    };
    let c_name = std::ffi::CString::new(name.as_bytes())
        .map_err(|_| pyre_interpreter::PyError::value_error("embedded null character"))?;
    // `lib_pypy/_posixshmem.py shm_unlink` retries on EINTR.
    loop {
        let rv = pyre_object::with_roots!(w_path => unsafe {
            ll::c_shm_unlink(c_name.as_ptr() as majit_rlib::rffi::CCHARP)
        });
        if rv < 0 {
            let errno = majit_rlib::rposix::_get_errno();
            if errno == libc::EINTR {
                continue;
            }
            let e = std::io::Error::from_raw_os_error(errno);
            return Err(pyre_interpreter::PyError::os_error_with_errno(
                errno,
                format!("shm_unlink: {e}"),
            ));
        }
        break;
    }
    Ok(w_none())
}

pyre_interpreter::py_module! {
    "_posixshmem",
    extra_init: |ns| {
        #[cfg(feature = "host_env")]
        {
            pyre_interpreter::__pyre_store!(ns, "shm_open", pyre_interpreter::make_builtin_function("shm_open", shm_open));
            pyre_interpreter::__pyre_store!(ns, "shm_unlink", pyre_interpreter::make_builtin_function_with_arity("shm_unlink", shm_unlink, 1));
        }
        #[cfg(not(feature = "host_env"))]
        let _ = ns;
    }
}
