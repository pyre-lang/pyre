//! errno module — PyPy: `pypy/module/errno/`.
//!
//! Numerics differ per OS (e.g. `EAGAIN` is 11 on Linux but 35 on
//! macOS), so when `host_env` is enabled every constant resolves
//! through `rustpython_host_env::errno::errors` (a `pub use libc::*`
//! re-export).  The `host_env = off` build keeps a darwin/BSD-flavoured
//! fallback so pyre-wasm preserves its previous behaviour.
//!
//! Which names exist is the other half of the contract: `interp_errno.py`
//! declares one list for every platform and lets `DefinedConstantInteger`
//! drop the entries the host's `errno.h` does not define, so a name absent
//! here is a name the module claims no platform has.  A Rust port cannot
//! probe, so the same partition is spelled with target gates — anything
//! `libc` (or, on windows, the `WSA` re-exports) supplies everywhere goes in
//! the common list, and the Darwin-only Mach/RPC block sits behind
//! `target_vendor = "apple"`.

pyre_interpreter::py_module! {
    "errno",
    extra_init: |ns| {
        // `interp_errno.py` builds `errorcode = {code: name, ...}`
        // alongside each exported constant.  We populate it incrementally
        // as we register the constants below.
        //
        // A `dict` header moves, and every registration allocates: the
        // constant's `int`, the name `str`, the key string each namespace
        // store builds, and the dict's own storage as it grows.  A minor
        // collection anywhere in that run relocates the dict — being reachable
        // from the module namespace keeps it alive but does not hold it still
        // — so the word lives in a root slot and is read back for every
        // insertion.  `ns` is a module dict, which is allocated stable, so it
        // stays valid across the same run.
        let roots = pyre_object::gc_roots::push_roots();
        let errorcode_slot = roots.base();
        let _ = roots.pin_root(pyre_object::w_dict_new());
        crate::__pyre_store!(ns, "errorcode", roots.get(errorcode_slot));
        let mut store = |name: &str, value: i64| {
            crate::__pyre_store!(ns, name, pyre_object::w_int_new(value));
            // Call arguments evaluate left to right, so the key and the value
            // are built first: reading the dict slot inline with them would
            // read it before those allocations and hand over a pre-move word.
            // Each newborn is pinned before the next allocation, then the
            // store reads both from their slots (`RootScope::pin_root`).
            let _pair_roots = pyre_object::gc_roots::push_roots();
            let code_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(pyre_object::w_int_new(value));
            let name_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_new(name));
            unsafe {
                pyre_object::w_dict_store(
                    roots.get(errorcode_slot),
                    pyre_object::gc_roots::shadow_stack_get(code_slot),
                    pyre_object::gc_roots::shadow_stack_get(name_slot),
                );
            }
        };
        #[cfg(all(feature = "host_env", not(target_arch = "wasm32")))]
        {
            use rustpython_host_env::errno::errors as host_errno;
            let entries: &[(&str, i32)] = &[
                ("EPERM", host_errno::EPERM),
                ("ENOENT", host_errno::ENOENT),
                ("ESRCH", host_errno::ESRCH),
                ("EINTR", host_errno::EINTR),
                ("EIO", host_errno::EIO),
                ("ENXIO", host_errno::ENXIO),
                ("E2BIG", host_errno::E2BIG),
                ("ENOEXEC", host_errno::ENOEXEC),
                ("EBADF", host_errno::EBADF),
                ("ECHILD", host_errno::ECHILD),
                ("EAGAIN", host_errno::EAGAIN),
                ("EWOULDBLOCK", host_errno::EWOULDBLOCK),
                ("ENOMEM", host_errno::ENOMEM),
                ("EACCES", host_errno::EACCES),
                ("EFAULT", host_errno::EFAULT),
                ("EBUSY", host_errno::EBUSY),
                ("EEXIST", host_errno::EEXIST),
                ("EXDEV", host_errno::EXDEV),
                ("ENODEV", host_errno::ENODEV),
                ("ENOTDIR", host_errno::ENOTDIR),
                ("EISDIR", host_errno::EISDIR),
                ("EINVAL", host_errno::EINVAL),
                ("ENFILE", host_errno::ENFILE),
                ("EMFILE", host_errno::EMFILE),
                ("ENOTTY", host_errno::ENOTTY),
                ("EFBIG", host_errno::EFBIG),
                ("ENOSPC", host_errno::ENOSPC),
                ("ESPIPE", host_errno::ESPIPE),
                ("EROFS", host_errno::EROFS),
                ("EMLINK", host_errno::EMLINK),
                ("EPIPE", host_errno::EPIPE),
                ("EDOM", host_errno::EDOM),
                ("ERANGE", host_errno::ERANGE),
                ("EDEADLK", host_errno::EDEADLK),
                ("ENAMETOOLONG", host_errno::ENAMETOOLONG),
                ("ENOLCK", host_errno::ENOLCK),
                ("ENOSYS", host_errno::ENOSYS),
                ("ENOTEMPTY", host_errno::ENOTEMPTY),
                ("ELOOP", host_errno::ELOOP),
                ("EOVERFLOW", host_errno::EOVERFLOW),
                ("EPROTO", host_errno::EPROTO),
                ("EDESTADDRREQ", host_errno::EDESTADDRREQ),
                ("EAFNOSUPPORT", host_errno::EAFNOSUPPORT),
                ("EALREADY", host_errno::EALREADY),
                ("EDQUOT", host_errno::EDQUOT),
                // socket / network errnos (used by ftplib, ssl, socket,
                // asyncio, logging.handlers, …)
                ("EINPROGRESS", host_errno::EINPROGRESS),
                ("ENOTSOCK", host_errno::ENOTSOCK),
                ("EMSGSIZE", host_errno::EMSGSIZE),
                ("EPROTOTYPE", host_errno::EPROTOTYPE),
                ("ENOPROTOOPT", host_errno::ENOPROTOOPT),
                ("EPROTONOSUPPORT", host_errno::EPROTONOSUPPORT),
                ("EOPNOTSUPP", host_errno::EOPNOTSUPP),
                ("EADDRINUSE", host_errno::EADDRINUSE),
                ("EADDRNOTAVAIL", host_errno::EADDRNOTAVAIL),
                ("ENETDOWN", host_errno::ENETDOWN),
                ("ENETUNREACH", host_errno::ENETUNREACH),
                ("ENETRESET", host_errno::ENETRESET),
                ("ECONNABORTED", host_errno::ECONNABORTED),
                ("ECONNRESET", host_errno::ECONNRESET),
                ("ENOBUFS", host_errno::ENOBUFS),
                ("EISCONN", host_errno::EISCONN),
                ("ENOTCONN", host_errno::ENOTCONN),
                ("ESHUTDOWN", host_errno::ESHUTDOWN),
                ("ETIMEDOUT", host_errno::ETIMEDOUT),
                ("ECONNREFUSED", host_errno::ECONNREFUSED),
                ("EHOSTDOWN", host_errno::EHOSTDOWN),
                ("EHOSTUNREACH", host_errno::EHOSTUNREACH),
                // `interp_errno.py` declares these in the same list, and
                // every platform pyre builds for defines them in its own
                // `errno.h`.
                ("EILSEQ", host_errno::EILSEQ),
                ("ECANCELED", host_errno::ECANCELED),
                ("ENOTSUP", host_errno::ENOTSUP),
                ("EOWNERDEAD", host_errno::EOWNERDEAD),
                ("ENOTRECOVERABLE", host_errno::ENOTRECOVERABLE),
            ];
            for (name, value) in entries {
                store(name, *value as i64);
            }
            // The MSVC runtime carries these under the same names, so they
            // are not the Unix-only set they were grouped with.
            #[cfg(any(unix, windows))]
            {
                let posix_entries: &[(&str, i32)] = &[
                    ("ETXTBSY", host_errno::ETXTBSY),
                    ("ENOMSG", host_errno::ENOMSG),
                    ("EIDRM", host_errno::EIDRM),
                    ("EBADMSG", host_errno::EBADMSG),
                    ("ENODATA", host_errno::ENODATA),
                    ("ENOLINK", host_errno::ENOLINK),
                    ("ENOSR", host_errno::ENOSR),
                    ("ENOSTR", host_errno::ENOSTR),
                    ("ETIME", host_errno::ETIME),
                ];
                for (name, value) in posix_entries {
                    store(name, *value as i64);
                }
            }
            #[cfg(unix)]
            {
                let unix_entries: &[(&str, i32)] = &[
                    // Declared in the same list, and supplied by `libc` here.
                    // Windows reaches the first six through `win_errors`
                    // (`interp_errno.py`), which registers the `WSA`
                    // spelling and adds the stripped name as a second binding
                    // -- the windows block below stores them in that order, so
                    // the reverse mapping names the `WSA` one there.
                    ("ESTALE", host_errno::ESTALE),
                    ("EUSERS", host_errno::EUSERS),
                    ("EREMOTE", host_errno::EREMOTE),
                    ("ETOOMANYREFS", host_errno::ETOOMANYREFS),
                    ("EPFNOSUPPORT", host_errno::EPFNOSUPPORT),
                    ("ESOCKTNOSUPPORT", host_errno::ESOCKTNOSUPPORT),
                    ("ENOTBLK", host_errno::ENOTBLK),
                    ("EMULTIHOP", host_errno::EMULTIHOP),
                ];
                for (name, value) in unix_entries {
                    store(name, *value as i64);
                }
            }
            // `interp_errno.py`'s "MacOSX specific errnos" block, plus
            // `EQFULL`, which that list omits.  Nothing here writes a number
            // down -- `libc` supplies each one, and `errno.EQFULL` read 106
            // under 3.14.6 on darwin when that omission was checked.
            // `extra_tests/snippets/errno_platform_names.py` asserts that
            // every name in this block is exported and reaches `errorcode`,
            // on the reference interpreter as well as on pyre.
            // `DefinedConstantInteger` drops each of these on a platform whose
            // `errno.h` lacks it; the equivalent here is the target gate,
            // since `libc` declares them for apple targets only.
            #[cfg(target_vendor = "apple")]
            {
                let apple_entries: &[(&str, i32)] = &[
                    ("ENOATTR", host_errno::ENOATTR),
                    ("EAUTH", host_errno::EAUTH),
                    ("EBADARCH", host_errno::EBADARCH),
                    ("EBADEXEC", host_errno::EBADEXEC),
                    ("EBADMACHO", host_errno::EBADMACHO),
                    ("EBADRPC", host_errno::EBADRPC),
                    ("EDEVERR", host_errno::EDEVERR),
                    ("EFTYPE", host_errno::EFTYPE),
                    ("ENEEDAUTH", host_errno::ENEEDAUTH),
                    ("ENOPOLICY", host_errno::ENOPOLICY),
                    ("EPROCLIM", host_errno::EPROCLIM),
                    ("EPROCUNAVAIL", host_errno::EPROCUNAVAIL),
                    ("EPROGMISMATCH", host_errno::EPROGMISMATCH),
                    ("EPROGUNAVAIL", host_errno::EPROGUNAVAIL),
                    ("EPWROFF", host_errno::EPWROFF),
                    ("EQFULL", host_errno::EQFULL),
                    ("ERPCMISMATCH", host_errno::ERPCMISMATCH),
                    ("ESHLIBVERS", host_errno::ESHLIBVERS),
                ];
                for (name, value) in apple_entries {
                    store(name, *value as i64);
                }
            }
            // The rest of the runtime's own set, then the Winsock codes.
            //
            // A socket reports the Winsock code, so the classic socket names
            // above already resolve to those and the `WSAE*` spellings share
            // their values.  They are stored second on purpose: `errorcode`
            // is keyed by the number, and the name it answers with for a
            // shared one is the Winsock spelling.
            #[cfg(windows)]
            {
                let windows_entries: &[(&str, i32)] = &[
                    ("EDEADLOCK", host_errno::EDEADLOCK),
                    ("ESOCKTNOSUPPORT", host_errno::ESOCKTNOSUPPORT),
                    ("EPFNOSUPPORT", host_errno::EPFNOSUPPORT),
                    ("ETOOMANYREFS", host_errno::ETOOMANYREFS),
                    ("EUSERS", host_errno::EUSERS),
                    ("ESTALE", host_errno::ESTALE),
                    ("EREMOTE", host_errno::EREMOTE),
                    ("WSABASEERR", host_errno::WSABASEERR),
                    ("WSAEINTR", host_errno::WSAEINTR),
                    ("WSAEBADF", host_errno::WSAEBADF),
                    ("WSAEACCES", host_errno::WSAEACCES),
                    ("WSAEFAULT", host_errno::WSAEFAULT),
                    ("WSAEINVAL", host_errno::WSAEINVAL),
                    ("WSAEMFILE", host_errno::WSAEMFILE),
                    ("WSAEWOULDBLOCK", host_errno::WSAEWOULDBLOCK),
                    ("WSAEINPROGRESS", host_errno::WSAEINPROGRESS),
                    ("WSAEALREADY", host_errno::WSAEALREADY),
                    ("WSAENOTSOCK", host_errno::WSAENOTSOCK),
                    ("WSAEDESTADDRREQ", host_errno::WSAEDESTADDRREQ),
                    ("WSAEMSGSIZE", host_errno::WSAEMSGSIZE),
                    ("WSAEPROTOTYPE", host_errno::WSAEPROTOTYPE),
                    ("WSAENOPROTOOPT", host_errno::WSAENOPROTOOPT),
                    ("WSAEPROTONOSUPPORT", host_errno::WSAEPROTONOSUPPORT),
                    ("WSAESOCKTNOSUPPORT", host_errno::WSAESOCKTNOSUPPORT),
                    ("WSAEOPNOTSUPP", host_errno::WSAEOPNOTSUPP),
                    ("WSAEPFNOSUPPORT", host_errno::WSAEPFNOSUPPORT),
                    ("WSAEAFNOSUPPORT", host_errno::WSAEAFNOSUPPORT),
                    ("WSAEADDRINUSE", host_errno::WSAEADDRINUSE),
                    ("WSAEADDRNOTAVAIL", host_errno::WSAEADDRNOTAVAIL),
                    ("WSAENETDOWN", host_errno::WSAENETDOWN),
                    ("WSAENETUNREACH", host_errno::WSAENETUNREACH),
                    ("WSAENETRESET", host_errno::WSAENETRESET),
                    ("WSAECONNABORTED", host_errno::WSAECONNABORTED),
                    ("WSAECONNRESET", host_errno::WSAECONNRESET),
                    ("WSAENOBUFS", host_errno::WSAENOBUFS),
                    ("WSAEISCONN", host_errno::WSAEISCONN),
                    ("WSAENOTCONN", host_errno::WSAENOTCONN),
                    ("WSAESHUTDOWN", host_errno::WSAESHUTDOWN),
                    ("WSAETOOMANYREFS", host_errno::WSAETOOMANYREFS),
                    ("WSAETIMEDOUT", host_errno::WSAETIMEDOUT),
                    ("WSAECONNREFUSED", host_errno::WSAECONNREFUSED),
                    ("WSAELOOP", host_errno::WSAELOOP),
                    ("WSAENAMETOOLONG", host_errno::WSAENAMETOOLONG),
                    ("WSAEHOSTDOWN", host_errno::WSAEHOSTDOWN),
                    ("WSAEHOSTUNREACH", host_errno::WSAEHOSTUNREACH),
                    ("WSAENOTEMPTY", host_errno::WSAENOTEMPTY),
                    ("WSAEPROCLIM", host_errno::WSAEPROCLIM),
                    ("WSAEUSERS", host_errno::WSAEUSERS),
                    ("WSAEDQUOT", host_errno::WSAEDQUOT),
                    ("WSAESTALE", host_errno::WSAESTALE),
                    ("WSAEREMOTE", host_errno::WSAEREMOTE),
                    ("WSASYSNOTREADY", host_errno::WSASYSNOTREADY),
                    ("WSAVERNOTSUPPORTED", host_errno::WSAVERNOTSUPPORTED),
                    ("WSANOTINITIALISED", host_errno::WSANOTINITIALISED),
                    ("WSAEDISCON", host_errno::WSAEDISCON),
                ];
                for (name, value) in windows_entries {
                    store(name, *value as i64);
                }
            }
        }
        #[cfg(any(not(feature = "host_env"), target_arch = "wasm32"))]
        {
            // darwin/BSD fallback so pyre-wasm keeps the same numeric
            // surface as before host_env existed.
            let entries: &[(&str, i64)] = &[
                ("EPERM", 1),
                ("ENOENT", 2),
                ("ESRCH", 3),
                ("EINTR", 4),
                ("EIO", 5),
                ("ENXIO", 6),
                ("E2BIG", 7),
                ("ENOEXEC", 8),
                ("EBADF", 9),
                ("ECHILD", 10),
                ("EAGAIN", 35),
                ("EWOULDBLOCK", 35),
                ("ENOMEM", 12),
                ("EACCES", 13),
                ("EFAULT", 14),
                ("ENOTBLK", 15),
                ("EBUSY", 16),
                ("EEXIST", 17),
                ("EXDEV", 18),
                ("ENODEV", 19),
                ("ENOTDIR", 20),
                ("EISDIR", 21),
                ("EINVAL", 22),
                ("ENFILE", 23),
                ("EMFILE", 24),
                ("ENOTTY", 25),
                ("ETXTBSY", 26),
                ("EFBIG", 27),
                ("ENOSPC", 28),
                ("ESPIPE", 29),
                ("EROFS", 30),
                ("EMLINK", 31),
                ("EPIPE", 32),
                ("EDOM", 33),
                ("ERANGE", 34),
                ("EDEADLK", 11),
                ("ENAMETOOLONG", 63),
                ("ENOLCK", 77),
                ("ENOSYS", 78),
                ("ENOTEMPTY", 66),
                ("ELOOP", 62),
                ("ENOMSG", 91),
                ("EIDRM", 90),
                ("EBADMSG", 94),
                ("EMULTIHOP", 95),
                ("ENODATA", 96),
                ("ENOLINK", 97),
                ("ENOSR", 98),
                ("ENOSTR", 99),
                ("EOVERFLOW", 84),
                ("EPROTO", 100),
                ("ETIME", 101),
                ("EDESTADDRREQ", 39),
                ("EAFNOSUPPORT", 47),
                ("EALREADY", 37),
                ("EDQUOT", 69),
                // socket / network errnos (darwin/BSD numerics)
                ("EINPROGRESS", 36),
                ("ENOTSOCK", 38),
                ("EMSGSIZE", 40),
                ("EPROTOTYPE", 41),
                ("ENOPROTOOPT", 42),
                ("EPROTONOSUPPORT", 43),
                ("EOPNOTSUPP", 45),
                ("EADDRINUSE", 48),
                ("EADDRNOTAVAIL", 49),
                ("ENETDOWN", 50),
                ("ENETUNREACH", 51),
                ("ENETRESET", 52),
                ("ECONNABORTED", 53),
                ("ECONNRESET", 54),
                ("ENOBUFS", 55),
                ("EISCONN", 56),
                ("ENOTCONN", 57),
                ("ESHUTDOWN", 58),
                ("ETIMEDOUT", 60),
                ("ECONNREFUSED", 61),
                ("EHOSTDOWN", 64),
                ("EHOSTUNREACH", 65),
                // The portable tail of `interp_errno.py`'s list, in the same
                // darwin/BSD numbering the rest of this table uses.  The
                // Darwin-only Mach/RPC names are deliberately absent: this
                // arm also serves wasm, where they describe nothing.
                ("ESOCKTNOSUPPORT", 44),
                ("ENOTSUP", 45),
                ("EPFNOSUPPORT", 46),
                ("ETOOMANYREFS", 59),
                ("EUSERS", 68),
                ("ESTALE", 70),
                ("EREMOTE", 71),
                ("ECANCELED", 89),
                ("EILSEQ", 92),
                ("ENOTRECOVERABLE", 104),
                ("EOWNERDEAD", 105),
            ];
            for (name, value) in entries {
                store(name, *value);
            }
        }
    }
}
