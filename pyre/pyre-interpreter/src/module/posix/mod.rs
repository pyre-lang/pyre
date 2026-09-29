//! posix module — PyPy: pypy/module/posix/
//!
//! Provides the minimal surface that os.py module init needs to succeed
//! plus the host_env-backed implementations of the calls pyre actually
//! exercises.  The shared `stat_result` builtin type lives here too, since
//! both arms build one: `interp_posix` wraps the host's syscalls, and
//! `interp_posix_wasm` answers from the import machinery's source seam on a
//! target that has none.

#[cfg(not(target_arch = "wasm32"))]
crate::pyre_module_init!(interp_posix);

/// `interp_nt.py`, the Windows helpers `interp_posix` calls.
#[cfg(all(windows, feature = "host_env"))]
mod interp_nt;

#[cfg(not(target_arch = "wasm32"))]
pub use interp_posix::{W_DirEntry, W_ScandirIterator};

#[cfg(unix)]
pub(crate) use interp_posix::reinit_fork_tables_after_fork;

// wasm32 has no operating system to wrap: the only filesystem the guest can
// see is the embedder's, reached through the import machinery's
// `SourceProvider`.  That leaves a `posix` too narrow to be a `cfg` pass over
// the syscall module, so it is its own arm.  The seam is what `host_env`
// builds, so without it there is nothing for this arm to answer from and the
// module is not registered at all.
#[cfg(all(target_arch = "wasm32", feature = "host_env"))]
crate::pyre_module_init!(interp_posix_wasm);

use pyre_object::PyObjectRef;
use rustpython_wtf8::Wtf8;

/// `interp_posix.urandom`, which `_random.W_Random.seed` calls directly.
#[cfg(not(target_arch = "wasm32"))]
pub use interp_posix::urandom;
#[cfg(all(target_arch = "wasm32", feature = "host_env"))]
pub use interp_posix_wasm::urandom;

/// A wasm32 build without `host_env` registers no posix module and has no
/// entropy source, which is the OSError `W_Random.seed` falls back on.
#[cfg(all(target_arch = "wasm32", not(feature = "host_env")))]
pub fn urandom(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Err(crate::PyError::os_error("urandom unavailable"))
}

/// `posix.stat_result` — a real structseq (tuple subclass) so `st[0]`,
/// `len(st)`, iteration and `isinstance(st, tuple)` all work, matching
/// `posixmodule.c` `stat_result_desc`.  The 10 sequence slots hold the
/// integer fields, with the integer-seconds times at 7..10 under the
/// hidden `_integer_atime`/`_integer_mtime`/`_integer_ctime` names; the
/// float `st_atime`/`st_mtime`/`st_ctime`, the `st_*_ns` integers, and the
/// `st_blksize`/`st_blocks`/`st_rdev` block-device fields are named-only
/// extras.
pub(crate) fn stat_result_seq_type() -> PyObjectRef {
    static STAT_RESULT_SEQ_TYPE: pyre_object::gc_roots::RootedOnceRef =
        pyre_object::gc_roots::RootedOnceRef::new();
    STAT_RESULT_SEQ_TYPE.get_or_init(|| {
        crate::_structseq::make_struct_seq_with_extra(
            // Dotted name → `__name__` "stat_result", repr "os.stat_result(...)".
            "os.stat_result",
            // `app_posix.py` `stat_result` — slots 7..10 are the hidden
            // integer timestamps; the float `st_atime`/`st_mtime`/`st_ctime`
            // are named-only extras, never indexable.
            &[
                "st_mode",
                "st_ino",
                "st_dev",
                "st_nlink",
                "st_uid",
                "st_gid",
                "st_size",
                "_integer_atime",
                "_integer_mtime",
                "_integer_ctime",
            ],
            // `app_posix.py` `stat_result` — named-only extras ordered by
            // their `structseqfield` index (11..13, 20..23, 40..42, 50..52).
            // `structseq_descr_new` fills surplus sequence items into
            // this list in order, so the list order must match PyPy's
            // index sort, not the build-time population order.
            &[
                // float times, indices 11..13.
                "st_atime",
                "st_mtime",
                "st_ctime",
                // `app_posix.py` `stat_result` — present where the
                // platform's `struct stat` carries them (every Unix
                // target), indices 20..23.
                #[cfg(unix)]
                "st_blksize",
                #[cfg(unix)]
                "st_blocks",
                #[cfg(unix)]
                "st_rdev",
                // `rposix_stat.py` exposes `st_flags` where the C
                // `struct stat` carries it (BSD family / macOS).
                #[cfg(target_os = "macos")]
                "st_flags",
                // `build_stat_result` (interp_posix.py) +
                // `rposix_stat.py STAT_FIELDS += ALL_STAT_FIELDS[-3:]`
                // — the sub-second nanosecond remainders, indices 40..42.
                "nsec_atime",
                "nsec_mtime",
                "nsec_ctime",
                // full nanosecond timestamps, indices 50..52.
                "st_atime_ns",
                "st_mtime_ns",
                "st_ctime_ns",
                // The Windows members of `stat_result_fields`, which sit
                // after every time and platform extra there too.
                #[cfg(windows)]
                "st_birthtime",
                #[cfg(windows)]
                "st_birthtime_ns",
                #[cfg(windows)]
                "st_file_attributes",
                #[cfg(windows)]
                "st_reparse_tag",
            ],
        )
    })
}

/// A filesystem name reported back to the caller: `bytes` when the path
/// argument was `bytes` (`posixmodule.c path_converter`), else `str`.
///
/// The name arrives as raw bytes because `readdir`'s `d_name` is handed
/// back unchanged. Both arms keep it openable: bytes mode returns it
/// verbatim, and the `str` arm takes the filesystem decoding
/// (`interp_posix.py space.newfilename(f)`), so a name like
/// `b"bad_\xff"` still names the file on disk instead of carrying U+FFFD.
pub(crate) fn fs_name_obj(bytes_mode: bool, name: &[u8]) -> PyObjectRef {
    if bytes_mode {
        pyre_object::bytesobject::w_bytes_from_bytes(&crate::gateway::fs_result_bytes(name))
    } else {
        crate::gateway::fsdecode_filename_bytes(name)
    }
}

/// `posix_fspath` / `PyOS_FSPath` — `str` and `bytes` pass through unchanged
/// (the protocol's identity case); any other object has `type(path).__fspath__`
/// bound before it is called.
pub(crate) fn fspath(
    arg: pyre_object::PyObjectRef,
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    // `str` and `bytes` only — a `bytearray` is a readable buffer and not a
    // path, so it goes on to be rejected below.
    unsafe {
        if pyre_object::is_str(arg) || pyre_object::bytesobject::is_bytes(arg) {
            return Ok(arg);
        }
    }
    let roots = pyre_object::gc_roots::push_roots();
    let arg_slot = pyre_object::gc_roots::shadow_stack_len();
    let arg = pyre_object::gc_roots::pin_root(arg);
    let path_type = crate::typedef::r#type(arg);
    if let Some(pt) = path_type
        && let Some(fspath_descr) = unsafe {
            crate::baseobjspace::lookup_in_type(
                pt.as_ptr(),
                pyre_object::unicodeobject::box_str_constant(Wtf8::new("__fspath__")),
            )
        }
    {
        let fspath_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(fspath_descr);
        // `interp_posix.py fspath` binds `__fspath__` before calling it
        // (`space.get_and_call_function`); a non-descriptor is its own value.
        let fspath_fn = unsafe {
            crate::baseobjspace::get(
                pyre_object::gc_roots::shadow_stack_get(fspath_slot),
                pyre_object::gc_roots::shadow_stack_get(arg_slot),
                pt.as_ptr(),
            )?
        }
        .unwrap_or_else(|| pyre_object::gc_roots::shadow_stack_get(fspath_slot));
        let arg = pyre_object::gc_roots::shadow_stack_get(arg_slot);
        // A `None` left on the type switches the protocol off the way
        // `__hash__ = None` does, so the object is turned away as not
        // path-like and named by its own type.
        //
        // [3.14-spec] `interp_posix.py fspath` only refuses a missing
        // `__fspath__` and calls the `None` it found, reporting `NoneType` as
        // not callable; `posixmodule.c PyOS_FSPath` tests `func == Py_None`
        // first. Evidence: `lib-python/3/test/test_os.py`
        // `test_fspath_set_to_None` asserts "expected str, bytes or
        // os.PathLike object, not Foo" (and `Baz`) for `__fspath__ = None`.
        if unsafe { pyre_object::is_none(fspath_fn) } {
            return Err(crate::PyError::type_error(format!(
                "expected str, bytes or os.PathLike object, not {}",
                crate::gateway::short_type_name(arg)
            )));
        }
        pyre_object::gc_roots::shadow_stack_set(fspath_slot, fspath_fn);
        let result = crate::call::call_function_impl_result(
            pyre_object::gc_roots::shadow_stack_get(fspath_slot),
            &[],
        )?;
        let result_slot = pyre_object::gc_roots::shadow_stack_len();
        let result = pyre_object::gc_roots::pin_root(result);
        // The protocol is only satisfied by what a path can be, so an answer
        // that is neither names the object that gave it and the type it gave.
        if unsafe { pyre_object::is_str(result) || pyre_object::bytesobject::is_bytes(result) } {
            return Ok(result);
        }
        return Err(crate::PyError::type_error(format!(
            "expected {}.__fspath__() to return str or bytes, not {}",
            crate::gateway::short_type_name(pyre_object::gc_roots::shadow_stack_get(arg_slot)),
            crate::gateway::short_type_name(result)
        )));
    }
    let arg = pyre_object::gc_roots::shadow_stack_get(arg_slot);
    let error = crate::PyError::type_error(format!(
        "expected str, bytes or os.PathLike object, not {}",
        crate::gateway::short_type_name(arg)
    ));
    drop(roots);
    Err(error)
}

/// `os.terminal_size` structseq — `(columns, lines)`.
pub(crate) fn terminal_size_seq_type() -> PyObjectRef {
    static T: pyre_object::gc_roots::RootedOnceRef = pyre_object::gc_roots::RootedOnceRef::new();
    T.get_or_init(|| crate::_structseq::make_struct_seq("os.terminal_size", &["columns", "lines"]))
}

/// `uname_result` structseq — `(sysname, nodename, release, version,
/// machine)`.  `uname_result_desc` names the type after the module it is
/// registered in, which is what pickle imports to resolve it: `posix` where
/// this module is `posix`, `nt` on Windows, which has the type even though it
/// has no `uname` to build one with.
pub(crate) fn uname_result_seq_type() -> PyObjectRef {
    static T: pyre_object::gc_roots::RootedOnceRef = pyre_object::gc_roots::RootedOnceRef::new();
    T.get_or_init(|| {
        crate::_structseq::make_struct_seq(
            if cfg!(windows) {
                "nt.uname_result"
            } else {
                "posix.uname_result"
            },
            &["sysname", "nodename", "release", "version", "machine"],
        )
    })
}

/// `os.statvfs_result` structseq — 10 sequence slots with `f_fsid` as an
/// extra named field (`n_sequence_fields=10`, `n_fields=11`).
pub(crate) fn statvfs_result_seq_type() -> PyObjectRef {
    static T: pyre_object::gc_roots::RootedOnceRef = pyre_object::gc_roots::RootedOnceRef::new();
    T.get_or_init(|| {
        crate::_structseq::make_struct_seq_with_extra(
            "os.statvfs_result",
            &[
                "f_bsize",
                "f_frsize",
                "f_blocks",
                "f_bfree",
                "f_bavail",
                "f_files",
                "f_ffree",
                "f_favail",
                "f_flag",
                "f_namemax",
            ],
            &["f_fsid"],
        )
    })
}

/// `os.times_result` structseq — `(user, system, children_user,
/// children_system, elapsed)`; repr renders "posix.times_result(...)", or
/// "nt.times_result(...)" on the host whose module is spelled that way.  The
/// name is the one `pickle` imports to resolve the type, so it has to be the
/// module the host actually has.
///
/// [3.14-spec] `app_posix.py times_result` fixes `name = "posix.times_result"`
/// on every host; `posixmodule.c` sets `times_result_desc.name = MODNAME
/// ".times_result"`, with `MODNAME` `"nt"` under `MS_WINDOWS`. Windows only;
/// evidence is the C source at v3.14.6.
pub(crate) fn times_result_seq_type() -> PyObjectRef {
    static T: pyre_object::gc_roots::RootedOnceRef = pyre_object::gc_roots::RootedOnceRef::new();
    T.get_or_init(|| {
        crate::_structseq::make_struct_seq(
            if cfg!(windows) {
                "nt.times_result"
            } else {
                "posix.times_result"
            },
            &[
                "user",
                "system",
                "children_user",
                "children_system",
                "elapsed",
            ],
        )
    })
}
