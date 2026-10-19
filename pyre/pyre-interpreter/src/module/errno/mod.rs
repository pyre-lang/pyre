//! errno module — PyPy: `pypy/module/errno/`.
//!
//! Numerics differ per OS (e.g. `EAGAIN` is 11 on Linux but 35 on
//! macOS), so when `host_env` is enabled every constant resolves
//! through `libc` on Unix and through `rustpython_host_env::errno::errors`
//! on Windows (a `pub use libc::*` re-export plus Winsock names).  The
//! `host_env = off` build keeps a darwin/BSD-flavoured fallback so
//! pyre-wasm preserves its previous behaviour.
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
            #[cfg(unix)]
            use libc as host_errno;
            #[cfg(windows)]
            use rustpython_host_env::errno::errors as host_errno;
            // `interp_errno.py` stores `errors`, then `more_errors`, in list
            // order, and a later name takes `errorcode[code]` over an earlier
            // one with the same value: `EAGAIN` over `EWOULDBLOCK`, `EDEADLOCK`
            // over `EDEADLK`, `ENOTSUP` over `EOPNOTSUPP`.  `more_errors` names
            // some codes twice; storing one again changes nothing.
            // `DefinedConstantInteger` drops a name the host `errno.h` lacks;
            // the equivalent here is the target gate, since `libc` declares
            // the Linux-only and Apple-only names for those targets alone.
            // The Solaris pair (`ELOCKUNMAPPED`, `ENOTACTIVE`) is on no target
            // pyre builds for.  `EQFULL` comes last: `interp_errno.py`'s
            // "MacOSX specific errnos" omit it, and `errno.EQFULL` read 106
            // under 3.14.6 on darwin when that omission was checked.
            // `extra_tests/snippets/errno_platform_names.py` asserts that every
            // Apple name here is exported and reaches `errorcode`, on the
            // reference interpreter as well as on pyre.
            #[cfg(unix)]
            {
                let entries: &[(&str, i32)] = &[
                    ("ENODEV", host_errno::ENODEV),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("ENOCSI", host_errno::ENOCSI),
                    ("EHOSTUNREACH", host_errno::EHOSTUNREACH),
                    ("ENOMSG", host_errno::ENOMSG),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("EUCLEAN", host_errno::EUCLEAN),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("EL2NSYNC", host_errno::EL2NSYNC),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("EL2HLT", host_errno::EL2HLT),
                    ("ENODATA", host_errno::ENODATA),
                    ("ENOTBLK", host_errno::ENOTBLK),
                    ("ENOSYS", host_errno::ENOSYS),
                    ("EPIPE", host_errno::EPIPE),
                    ("EINVAL", host_errno::EINVAL),
                    ("EOVERFLOW", host_errno::EOVERFLOW),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("EADV", host_errno::EADV),
                    ("EINTR", host_errno::EINTR),
                    ("EUSERS", host_errno::EUSERS),
                    ("ENOTEMPTY", host_errno::ENOTEMPTY),
                    ("ENOBUFS", host_errno::ENOBUFS),
                    ("EPROTO", host_errno::EPROTO),
                    ("EREMOTE", host_errno::EREMOTE),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("ENAVAIL", host_errno::ENAVAIL),
                    ("ECHILD", host_errno::ECHILD),
                    ("ELOOP", host_errno::ELOOP),
                    ("EXDEV", host_errno::EXDEV),
                    ("E2BIG", host_errno::E2BIG),
                    ("ESRCH", host_errno::ESRCH),
                    ("EMSGSIZE", host_errno::EMSGSIZE),
                    ("EAFNOSUPPORT", host_errno::EAFNOSUPPORT),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("EBADR", host_errno::EBADR),
                    ("EHOSTDOWN", host_errno::EHOSTDOWN),
                    ("EPFNOSUPPORT", host_errno::EPFNOSUPPORT),
                    ("ENOPROTOOPT", host_errno::ENOPROTOOPT),
                    ("EBUSY", host_errno::EBUSY),
                    ("EWOULDBLOCK", host_errno::EWOULDBLOCK),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("EBADFD", host_errno::EBADFD),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("EDOTDOT", host_errno::EDOTDOT),
                    ("EISCONN", host_errno::EISCONN),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("ENOANO", host_errno::ENOANO),
                    ("ESHUTDOWN", host_errno::ESHUTDOWN),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("ECHRNG", host_errno::ECHRNG),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("ELIBBAD", host_errno::ELIBBAD),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("ENONET", host_errno::ENONET),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("EBADE", host_errno::EBADE),
                    ("EBADF", host_errno::EBADF),
                    ("EMULTIHOP", host_errno::EMULTIHOP),
                    ("EIO", host_errno::EIO),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("EUNATCH", host_errno::EUNATCH),
                    ("EPROTOTYPE", host_errno::EPROTOTYPE),
                    ("ENOSPC", host_errno::ENOSPC),
                    ("ENOEXEC", host_errno::ENOEXEC),
                    ("EALREADY", host_errno::EALREADY),
                    ("ENETDOWN", host_errno::ENETDOWN),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("ENOTNAM", host_errno::ENOTNAM),
                    ("EACCES", host_errno::EACCES),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("ELNRNG", host_errno::ELNRNG),
                    ("EILSEQ", host_errno::EILSEQ),
                    ("ENOTDIR", host_errno::ENOTDIR),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("ENOTUNIQ", host_errno::ENOTUNIQ),
                    ("EPERM", host_errno::EPERM),
                    ("EDOM", host_errno::EDOM),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("EXFULL", host_errno::EXFULL),
                    ("ECONNREFUSED", host_errno::ECONNREFUSED),
                    ("EISDIR", host_errno::EISDIR),
                    ("EPROTONOSUPPORT", host_errno::EPROTONOSUPPORT),
                    ("EROFS", host_errno::EROFS),
                    ("EADDRNOTAVAIL", host_errno::EADDRNOTAVAIL),
                    ("EIDRM", host_errno::EIDRM),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("ECOMM", host_errno::ECOMM),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("ESRMNT", host_errno::ESRMNT),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("EREMOTEIO", host_errno::EREMOTEIO),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("EL3RST", host_errno::EL3RST),
                    ("EBADMSG", host_errno::EBADMSG),
                    ("ENFILE", host_errno::ENFILE),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("ELIBMAX", host_errno::ELIBMAX),
                    ("ESPIPE", host_errno::ESPIPE),
                    ("ENOLINK", host_errno::ENOLINK),
                    ("ENETRESET", host_errno::ENETRESET),
                    ("ETIMEDOUT", host_errno::ETIMEDOUT),
                    ("ENOENT", host_errno::ENOENT),
                    ("EEXIST", host_errno::EEXIST),
                    ("EDQUOT", host_errno::EDQUOT),
                    ("ENOSTR", host_errno::ENOSTR),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("EBADSLT", host_errno::EBADSLT),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("EBADRQC", host_errno::EBADRQC),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("ELIBACC", host_errno::ELIBACC),
                    ("EFAULT", host_errno::EFAULT),
                    ("EFBIG", host_errno::EFBIG),
                    ("EDEADLK", host_errno::EDEADLK),
                    ("ENOTCONN", host_errno::ENOTCONN),
                    ("EDESTADDRREQ", host_errno::EDESTADDRREQ),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("ELIBSCN", host_errno::ELIBSCN),
                    ("ENOLCK", host_errno::ENOLCK),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("EISNAM", host_errno::EISNAM),
                    ("ECONNABORTED", host_errno::ECONNABORTED),
                    ("ENETUNREACH", host_errno::ENETUNREACH),
                    ("ESTALE", host_errno::ESTALE),
                    ("ENOSR", host_errno::ENOSR),
                    ("ENOMEM", host_errno::ENOMEM),
                    ("ENOTSOCK", host_errno::ENOTSOCK),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("ESTRPIPE", host_errno::ESTRPIPE),
                    ("EMLINK", host_errno::EMLINK),
                    ("ERANGE", host_errno::ERANGE),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("ELIBEXEC", host_errno::ELIBEXEC),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("EL3HLT", host_errno::EL3HLT),
                    ("ECONNRESET", host_errno::ECONNRESET),
                    ("EADDRINUSE", host_errno::EADDRINUSE),
                    ("EOPNOTSUPP", host_errno::EOPNOTSUPP),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("EREMCHG", host_errno::EREMCHG),
                    ("EAGAIN", host_errno::EAGAIN),
                    ("ENAMETOOLONG", host_errno::ENAMETOOLONG),
                    ("ENOTTY", host_errno::ENOTTY),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("ERESTART", host_errno::ERESTART),
                    ("ESOCKTNOSUPPORT", host_errno::ESOCKTNOSUPPORT),
                    ("ETIME", host_errno::ETIME),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("EBFONT", host_errno::EBFONT),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("EDEADLOCK", host_errno::EDEADLOCK),
                    ("ETOOMANYREFS", host_errno::ETOOMANYREFS),
                    ("EMFILE", host_errno::EMFILE),
                    ("ETXTBSY", host_errno::ETXTBSY),
                    ("EINPROGRESS", host_errno::EINPROGRESS),
                    ("ENXIO", host_errno::ENXIO),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("ENOPKG", host_errno::ENOPKG),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("ENOMEDIUM", host_errno::ENOMEDIUM),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("EMEDIUMTYPE", host_errno::EMEDIUMTYPE),
                    ("ECANCELED", host_errno::ECANCELED),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("ENOKEY", host_errno::ENOKEY),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("EKEYEXPIRED", host_errno::EKEYEXPIRED),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("EKEYREVOKED", host_errno::EKEYREVOKED),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("EKEYREJECTED", host_errno::EKEYREJECTED),
                    ("EOWNERDEAD", host_errno::EOWNERDEAD),
                    ("ENOTRECOVERABLE", host_errno::ENOTRECOVERABLE),
                    #[cfg(any(target_os = "linux", target_os = "android"))]
                    ("ERFKILL", host_errno::ERFKILL),
                    ("ECANCELED", host_errno::ECANCELED),
                    ("ENOTSUP", host_errno::ENOTSUP),
                    ("EOWNERDEAD", host_errno::EOWNERDEAD),
                    ("ENOTRECOVERABLE", host_errno::ENOTRECOVERABLE),
                    #[cfg(target_vendor = "apple")]
                    ("EAUTH", host_errno::EAUTH),
                    #[cfg(target_vendor = "apple")]
                    ("EBADARCH", host_errno::EBADARCH),
                    #[cfg(target_vendor = "apple")]
                    ("EBADEXEC", host_errno::EBADEXEC),
                    #[cfg(target_vendor = "apple")]
                    ("EBADMACHO", host_errno::EBADMACHO),
                    #[cfg(target_vendor = "apple")]
                    ("EBADRPC", host_errno::EBADRPC),
                    #[cfg(target_vendor = "apple")]
                    ("EDEVERR", host_errno::EDEVERR),
                    #[cfg(target_vendor = "apple")]
                    ("EFTYPE", host_errno::EFTYPE),
                    #[cfg(target_vendor = "apple")]
                    ("ENEEDAUTH", host_errno::ENEEDAUTH),
                    #[cfg(target_vendor = "apple")]
                    ("ENOATTR", host_errno::ENOATTR),
                    #[cfg(target_vendor = "apple")]
                    ("ENOPOLICY", host_errno::ENOPOLICY),
                    #[cfg(target_vendor = "apple")]
                    ("EPROCLIM", host_errno::EPROCLIM),
                    #[cfg(target_vendor = "apple")]
                    ("EPROCUNAVAIL", host_errno::EPROCUNAVAIL),
                    #[cfg(target_vendor = "apple")]
                    ("EPROGMISMATCH", host_errno::EPROGMISMATCH),
                    #[cfg(target_vendor = "apple")]
                    ("EPROGUNAVAIL", host_errno::EPROGUNAVAIL),
                    #[cfg(target_vendor = "apple")]
                    ("EPWROFF", host_errno::EPWROFF),
                    #[cfg(target_vendor = "apple")]
                    ("ERPCMISMATCH", host_errno::ERPCMISMATCH),
                    #[cfg(target_vendor = "apple")]
                    ("ESHLIBVERS", host_errno::ESHLIBVERS),
                    #[cfg(target_vendor = "apple")]
                    ("EQFULL", host_errno::EQFULL),
                ];
                for (name, value) in entries {
                    store(name, *value as i64);
                }
            }
            // Windows: the runtime's own set, then the Winsock codes.
            //
            // A socket reports the Winsock code, so the classic socket names
            // already resolve to those and the `WSAE*` spellings share their
            // values.  They are stored second on purpose: `errorcode` is keyed
            // by the number, and the name it answers with for a shared one is
            // the Winsock spelling.
            #[cfg(windows)]
            {
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
                // The MSVC runtime carries these under the same names, so
                // they are not a Unix-only set.
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
