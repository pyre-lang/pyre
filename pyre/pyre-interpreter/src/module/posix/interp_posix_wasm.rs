//! `posix` on wasm32 — PyPy: pypy/module/posix/interp_posix.py
//!
//! wasm32 has no operating system underneath it, so nothing here wraps a
//! syscall.  What the guest does have is the embedder's filesystem, reached
//! through the import machinery's [`SourceProvider`]: the same seam `import`
//! resolves modules with and traceback rendering reads source lines with.
//! This module publishes that seam under the names `os.py` reads.
//!
//! The list is what `os.py`'s module body needs to run — `environ`,
//! `_have_functions`, `stat` and `cpu_count` — plus `lstat`, `listdir`,
//! `fspath`, `scandir`, the structseq types `posix` publishes on every
//! platform (`stat_result`, `statvfs_result`, `times_result`, `uname_result`,
//! `terminal_size`), and the three calls the stdlib reaches for
//! that the embedder can still answer: `getcwd`, `getcwdb` and `urandom`.
//! On top of that sits a descriptor half — `open`, `close`, `read`, `lseek`,
//! `fstat` — over a table this module keeps itself.  The seam answers a whole
//! path at a time, so a descriptor is the bytes a path resolved to plus a
//! position; the numbers are the guest's own and name nothing on the other
//! side of the host ABI.  The seam is a read-only mount, so every entry point
//! that would change it (`open` for write, `unlink`, `rmdir`) is published and
//! answers `EROFS`, which is what a read-only filesystem answers — a different
//! statement from the call not existing.
//!
//! Everything else `os` normally re-exports stays absent, so `os._exists`
//! answers False for it and `os` builds the same fallbacks it builds on a
//! platform whose C library lacks the call.
//!
//! [`SourceProvider`]: crate::importing::SourceProvider

use pyre_object::{PyObjectRef, is_none};

/// `S_IFREG` / `S_IFDIR` — the two file types the seam can tell apart.  It
/// answers `is_dir` and a byte length and nothing else, so a symlink, a device
/// or a socket is not distinguishable from the file it resolves to.
const S_IFDIR: i64 = 0o040000;
const S_IFREG: i64 = 0o100000;

/// `posix.stat(path, *, dir_fd=None, follow_symlinks=True)` /
/// `posix.lstat(path, *, dir_fd=None)` — build a `stat_result` from the two
/// facts the seam reports.
///
/// `st_mode` carries the file type and a read-only permission set, and
/// `st_size` the byte length; every other field is zero, including all three
/// timestamps, because no timestamp reaches the guest.  A constant mtime is
/// self-consistent rather than merely absent: `linecache.checkcache` compares
/// a cached entry against a fresh `stat` of the same file, so equal values
/// keep the entry valid instead of invalidating it on every check.
///
/// The seam takes a name and nothing else, so it is the platform without
/// `fstatat` and without `fstat`: `dir_fd` is refused the way
/// `_DirFD_Unavailable` refuses it, and a descriptor is not one of the types
/// the argument accepts.  `follow_symlinks` is accepted and has no effect —
/// the seam resolves a path without reporting whether it went through a
/// symlink, which is the same reason `lstat` can only answer what `stat`
/// answers.
fn stat(args: &[PyObjectRef], default_follow: bool) -> Result<PyObjectRef, crate::PyError> {
    let name = if default_follow { "stat" } else { "lstat" };
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let allowed: &[&str] = if default_follow {
        &["path", "dir_fd", "follow_symlinks"]
    } else {
        &["path", "dir_fd"]
    };
    crate::builtins::kwarg_reject_unknown(kwargs, allowed, name)?;
    if pos.len() > 1 {
        return Err(crate::PyError::type_error(format!(
            "{name}() takes at most 1 positional argument ({} given)",
            pos.len()
        )));
    }
    let Some(w_path) = crate::builtins::bind_pos_or_kw(pos, kwargs, 0, "path", name, 1)? else {
        return Err(crate::PyError::type_error(format!(
            "{name}() missing required argument 'path' (pos 1)"
        )));
    };
    // Unwrapped in signature order, because `__fspath__` for `path` and
    // `__index__` for `dir_fd` can both raise and both run user code.
    let resolved = crate::gateway::fsencode_path_or_fd_w(w_path, name, false)?;
    if crate::builtins::kwarg_get(kwargs, "dir_fd").is_some_and(|v| !unsafe { is_none(v) }) {
        return Err(crate::PyError::not_implemented(
            "dir_fd unavailable on this platform",
        ));
    }
    if let Some(v) = crate::builtins::kwarg_get(kwargs, "follow_symlinks") {
        crate::baseobjspace::is_true(v)?;
    }
    stat_resolved(&resolved)
}

/// The guest's working directory, once something has changed it.
///
/// The seam resolves a relative name against the embedder's directory, which is
/// what `getcwd` reports until `chdir` runs.  After that the guest owns the
/// answer, and a relative path is joined here before it crosses -- the same
/// place `getcwd` reads, so the two never disagree.
static WORKING_DIRECTORY: std::sync::Mutex<Option<Vec<u8>>> = std::sync::Mutex::new(None);

/// The directory a relative path is resolved against, as bytes.
fn cwd_bytes() -> Vec<u8> {
    if let Some(directory) = WORKING_DIRECTORY.lock().unwrap().clone() {
        return directory;
    }
    crate::importing::source_cwd().unwrap_or_default()
}

/// `path` as the seam should see it: an absolute name unchanged, a relative one
/// joined to the working directory.
///
/// An empty name stays empty so that the seam reports `ENOENT` for it, which is
/// what `open("")` owes its caller rather than the working directory's contents.
fn resolve_against_cwd(path: &[u8]) -> Vec<u8> {
    if path.is_empty() {
        return path.to_vec();
    }
    if path.starts_with(b"/") {
        return normalized(path);
    }
    let mut resolved = cwd_bytes();
    if resolved.is_empty() {
        return normalized(path);
    }
    if !resolved.ends_with(b"/") {
        resolved.push(b'/');
    }
    resolved.extend_from_slice(path);
    normalized(&resolved)
}

/// `path` with its `.` and `..` components resolved and its repeated slashes
/// collapsed, the way `posixpath.normpath` resolves them.
///
/// The seam is asked for a name, not walked component by component, and the
/// embedded filesystem answers an exact key: `lib-python/3/..` names a
/// directory that is there, and asking for it under that spelling finds
/// nothing.  Resolving is lexical, so a `..` above a symbolic link lands where
/// the spelling says rather than where a walk would -- which is what
/// `normpath` documents about itself, and the only rule available to a seam
/// that cannot walk.
fn normalized(path: &[u8]) -> Vec<u8> {
    let rooted = path.starts_with(b"/");
    let mut parts: Vec<&[u8]> = Vec::new();
    for part in path.split(|b| *b == b'/') {
        match part {
            b"" | b"." => {}
            b".." => match parts.last() {
                // A `..` above the root is the root, and one that has no
                // component to cancel is kept only where it can still mean
                // something -- which is nowhere below an absolute path.
                Some(&last) if last != b".." => {
                    parts.pop();
                }
                _ if rooted => {}
                _ => parts.push(part),
            },
            other => parts.push(other),
        }
    }
    let mut out = if rooted { vec![b'/'] } else { Vec::new() };
    for (index, part) in parts.iter().enumerate() {
        if index > 0 {
            out.push(b'/');
        }
        out.extend_from_slice(part);
    }
    // Everything cancelled: an absolute path is left as the root, a relative
    // one as the directory it started in.
    if out.is_empty() {
        out.push(b'.');
    }
    out
}

/// `posix.chdir(path)` — the directory later relative paths resolve against.
///
/// `fchdir` is what a descriptor argument would need, and the seam has no
/// directory descriptor to offer it, so a name is the only spelling accepted.
fn chdir(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // `path_arg` reads `dir_fd` because most of its callers take one; `chdir`
    // is spelled `chdir(path)` and `fchdir` is the call that would take the
    // other, so the keyword is not a name this entry point knows.
    let (_, kwargs) = crate::builtins::split_builtin_kwargs(args);
    crate::builtins::kwarg_reject_unknown(kwargs, &["path"], "chdir")?;
    let resolved = path_arg(args, "chdir")?;
    let target = resolve_against_cwd(&resolved.as_bytes);
    let path: std::path::PathBuf = crate::gateway::os_string_from_fs_bytes(&target).into();
    if !crate::importing::source_is_dir(&path) {
        // The two answers a failed `chdir` gives: the name is not there, or it
        // is there and is not a directory.
        let errno = match crate::importing::source_file_size(&path) {
            Ok(_) => crate::builtins::wasm_errno::ENOTDIR,
            Err(e) => crate::builtins::wasm_errno::seam_errno(&e),
        };
        return Err(crate::PyError::os_error_syscall(errno, resolved.w_path()));
    }
    *WORKING_DIRECTORY.lock().unwrap() = Some(target);
    Ok(pyre_object::w_none())
}

/// The seam takes a `&Path`, and on this target an `OsStr` is the byte string
/// itself, so the filesystem bytes cross it whole rather than through a text
/// spelling that a name without one would not survive: an entry `listdir`
/// reported is an entry `stat` can be called with again.
///
/// This is the one place a relative name becomes an absolute one, so every
/// entry point that takes a path resolves it against the same directory
/// `getcwd` reports.
pub(crate) fn seam_path(bytes: &[u8]) -> std::path::PathBuf {
    crate::gateway::os_string_from_fs_bytes(&resolve_against_cwd(bytes)).into()
}

/// The seam's `io::Error` as the OSError it stands for, named by the path
/// object the caller passed rather than by a re-decoded spelling of it.
fn seam_error(e: &std::io::Error, w_path: PyObjectRef) -> crate::PyError {
    let errno = crate::builtins::wasm_errno::seam_errno(e);
    crate::PyError::os_error_syscall(crate::builtins::io_error_posix_errno(e, errno), w_path)
}

/// The `stat_result` layout of [`super::stat_result_seq_type`] with the
/// platform extras that target carries — on wasm32 neither the unix
/// block-device fields nor the Windows ones, leaving the float times, the
/// sub-second remainders and the full nanosecond timestamps.
fn make_stat_result(mode: i64, size: i64) -> PyObjectRef {
    let _roots = pyre_object::gc_roots::push_roots();
    let mut extra_slots: Vec<(&str, usize)> = Vec::new();
    let mut put_extra = |name: &'static str, value: PyObjectRef| {
        extra_slots.push((name, pyre_object::gc_roots::shadow_stack_len()));
        let _ = pyre_object::gc_roots::pin_root(value);
    };
    put_extra("st_atime", pyre_object::w_float_new(0.0));
    put_extra("st_mtime", pyre_object::w_float_new(0.0));
    put_extra("st_ctime", pyre_object::w_float_new(0.0));
    put_extra("nsec_atime", pyre_object::w_int_new(0));
    put_extra("nsec_mtime", pyre_object::w_int_new(0));
    put_extra("nsec_ctime", pyre_object::w_int_new(0));
    put_extra("st_atime_ns", pyre_object::w_int_new(0));
    put_extra("st_mtime_ns", pyre_object::w_int_new(0));
    put_extra("st_ctime_ns", pyre_object::w_int_new(0));
    drop(put_extra);
    let mut fields = pyre_object::gc_roots::RootedItems::new();
    fields.push(pyre_object::w_int_new(mode));
    fields.push(pyre_object::w_int_new(0)); // st_ino
    fields.push(pyre_object::w_int_new(0)); // st_dev
    fields.push(pyre_object::w_int_new(1)); // st_nlink
    fields.push(pyre_object::w_int_new(0)); // st_uid
    fields.push(pyre_object::w_int_new(0)); // st_gid
    fields.push(pyre_object::w_int_new(size));
    fields.push(pyre_object::w_int_new(0)); // _integer_atime
    fields.push(pyre_object::w_int_new(0)); // _integer_mtime
    fields.push(pyre_object::w_int_new(0)); // _integer_ctime
    let extras: Vec<_> = extra_slots
        .into_iter()
        .map(|(name, slot)| (name, pyre_object::gc_roots::shadow_stack_get(slot)))
        .collect();
    crate::_structseq::new_instance_with_extra(super::stat_result_seq_type(), fields.take(), extras)
}

/// `posix.listdir(path=None)` — the entry names the seam reports, in its
/// order.
///
/// The omitted argument is the `None` the signature names, which resolves to
/// `"."`; the guest has no `getcwd` to spell that any other way, so what it
/// stands for is whatever directory the embedder started in.  A `bytes` path
/// asks for `bytes` names, as it does on every other target.
fn listdir(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    crate::builtins::kwarg_reject_unknown(kwargs, &["path"], "listdir")?;
    if pos.len() > 1 {
        return Err(crate::PyError::type_error(format!(
            "listdir() takes at most 1 argument ({} given)",
            pos.len()
        )));
    }
    let w_path = crate::builtins::bind_pos_or_kw(pos, kwargs, 0, "path", "listdir", 1)?
        .unwrap_or(pyre_object::w_none());
    // One resolution yields both the path and its bytes-ness, so `__fspath__`
    // runs exactly once.
    let resolved = crate::gateway::fsencode_path_or_fd_nullable_w(w_path, "listdir", false)?;
    let bytes_mode = unsafe { resolved.is_bytes() };
    let path = seam_path(&resolved.as_bytes);
    let entries =
        crate::importing::source_list_dir(&path).map_err(|e| seam_error(&e, resolved.w_path()))?;
    // Each name is freshly allocated and the next one allocates again, so they
    // are pinned as they arrive.
    let mut items = pyre_object::gc_roots::RootedItems::new();
    for name in &entries {
        items.push(super::fs_name_obj(bytes_mode, name));
    }
    Ok(pyre_object::w_list_new(items.take()))
}

/// One `stat_result` for a path the caller has already resolved, so that
/// `stat`, `lstat` and `DirEntry.stat` report one file the same way.
fn stat_resolved(resolved: &crate::gateway::FsEncodedPath) -> Result<PyObjectRef, crate::PyError> {
    let path = seam_path(&resolved.as_bytes);
    let (mode, size) = if crate::importing::source_is_dir(&path) {
        (S_IFDIR | 0o555, 0)
    } else {
        match crate::importing::source_file_size(&path) {
            Ok(size) => (S_IFREG | 0o444, size as i64),
            // The seam reports why it could not answer, and the path object the
            // caller passed is what names the failure.
            Err(e) => return Err(seam_error(&e, resolved.w_path())),
        }
    };
    Ok(make_stat_result(mode, size))
}

/// `join_path_filename` — the directory and the entry name with one separator
/// between them.  An empty directory leaves the name alone, which is what
/// `scandir("")` would otherwise turn into an absolute path.
fn join_entry_path(dir_bytes: &[u8], name: &[u8]) -> Vec<u8> {
    if dir_bytes.is_empty() {
        return name.to_vec();
    }
    let mut joined = dir_bytes.to_vec();
    if !joined.ends_with(b"/") {
        joined.push(b'/');
    }
    joined.extend_from_slice(name);
    joined
}

/// The path a `DirEntry` was built for, back in the form the seam takes.
///
/// It is read off the entry rather than kept beside it: `path` is the entry's
/// own public attribute, so one spelling answers both the caller and the seam.
fn entry_path(self_obj: PyObjectRef) -> Result<crate::gateway::FsEncodedPath, crate::PyError> {
    crate::gateway::fsencode_path_w(crate::baseobjspace::getattr_str(self_obj, "path")?)
}

/// The receiver of a `DirEntry` method, with `follow_symlinks` accepted and
/// without effect for the same reason `lstat` answers what `stat` answers.
fn entry_self(
    args: &[PyObjectRef],
    name: &'static str,
    follow: bool,
) -> Result<PyObjectRef, crate::PyError> {
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let allowed: &[&str] = if follow { &["follow_symlinks"] } else { &[] };
    crate::builtins::kwarg_reject_unknown(kwargs, allowed, name)?;
    if let Some(v) = crate::builtins::kwarg_get(kwargs, "follow_symlinks") {
        crate::baseobjspace::is_true(v)?;
    }
    pos.first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error(format!("{name}() requires self")))
}

/// `posix.DirEntry` — the record `scandir` yields for one name.
///
/// The seam reports a name, whether it is a directory, and a byte length, so
/// that is what an entry can answer.  It keeps its name and its path and asks
/// the seam again for the rest, which is the same on-demand `stat` the entry
/// performs anywhere its `readdir` did not carry a type.  `is_symlink` is
/// false for every entry, not as a claim that nothing is a link but because a
/// name and what it resolves to are the only two things this seam separates.
fn dir_entry_type() -> PyObjectRef {
    static DIR_ENTRY_TYPE: pyre_object::gc_roots::RootedOnceRef =
        pyre_object::gc_roots::RootedOnceRef::new();
    DIR_ENTRY_TYPE.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type("posix.DirEntry", init_dir_entry_type);
        crate::typedef::mark_cpython_heap_type(tp, false);
        unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
        tp
    })
}

fn init_dir_entry_type(ns: PyObjectRef) {
    let method = |name: &'static str, f: crate::gateway::BuiltinCodeFn| unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            name,
            crate::make_builtin_function(name, f),
        );
    };
    method("is_dir", |args| {
        let resolved = entry_path(entry_self(args, "is_dir", true)?)?;
        Ok(pyre_object::w_bool_from(crate::importing::source_is_dir(
            &seam_path(&resolved.as_bytes),
        )))
    });
    method("is_file", |args| {
        let resolved = entry_path(entry_self(args, "is_file", true)?)?;
        let path = seam_path(&resolved.as_bytes);
        Ok(pyre_object::w_bool_from(
            !crate::importing::source_is_dir(&path)
                && crate::importing::source_file_size(&path).is_ok(),
        ))
    });
    method("is_symlink", |args| {
        entry_self(args, "is_symlink", false)?;
        Ok(pyre_object::w_bool_from(false))
    });
    method("is_junction", |args| {
        entry_self(args, "is_junction", false)?;
        Ok(pyre_object::w_bool_from(false))
    });
    method("stat", |args| {
        stat_resolved(&entry_path(entry_self(args, "stat", true)?)?)
    });
    // No inode number reaches the guest, so every entry reports the zero
    // `st_ino` its `stat_result` carries: one file has one identity either way.
    method("inode", |args| {
        entry_self(args, "inode", false)?;
        Ok(pyre_object::w_int_new(0))
    });
    method("__fspath__", |args| {
        crate::baseobjspace::getattr_str(entry_self(args, "__fspath__", false)?, "path")
    });
    method("__repr__", |args| {
        let name = crate::baseobjspace::getattr_str(entry_self(args, "__repr__", false)?, "name")?;
        Ok(pyre_object::w_str_from_wtf8_managed(
            crate::display::wtf8_format!(
                "<DirEntry ",
                unsafe { crate::display::py_repr_wtf8(name)? },
                ">"
            ),
        ))
    });
}

/// The entry for one name inside the directory `scandir` was called on.
fn new_dir_entry(bytes_mode: bool, dir_bytes: &[u8], name: &[u8]) -> PyObjectRef {
    let _roots = pyre_object::gc_roots::push_roots();
    let entry_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_instance_new(dir_entry_type()));
    // Each value is built before the receiver is read back, because building
    // it allocates and the allocation may move the entry.
    let store = |field: &str, value: PyObjectRef| {
        crate::baseobjspace::setdictvalue_native(
            pyre_object::gc_roots::shadow_stack_get(entry_slot),
            field,
            value,
        );
    };
    store("name", super::fs_name_obj(bytes_mode, name));
    store(
        "path",
        super::fs_name_obj(bytes_mode, &join_entry_path(dir_bytes, name)),
    );
    pyre_object::gc_roots::shadow_stack_get(entry_slot)
}

/// `posix.ScandirIterator` — the iterator `scandir` returns, and the context
/// manager `os.walk` enters.
///
/// The seam answers a whole directory at a time, so the entries exist before
/// the first `__next__`; what the iterator holds is the position in them.
/// `close` drops that, which is what makes an exhausted or closed iterator
/// report the end rather than the directory again.
fn scandir_iterator_type() -> PyObjectRef {
    static SCANDIR_ITERATOR_TYPE: pyre_object::gc_roots::RootedOnceRef =
        pyre_object::gc_roots::RootedOnceRef::new();
    SCANDIR_ITERATOR_TYPE.get_or_init(|| {
        let tp =
            crate::typedef::make_builtin_type("posix.ScandirIterator", init_scandir_iterator_type);
        crate::typedef::mark_cpython_heap_type(tp, false);
        unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
        tp
    })
}

/// The iterator's own position, or `None` once it has been closed.
const SCANDIR_POSITION: &str = "__scandir_iter__";

fn init_scandir_iterator_type(ns: PyObjectRef) {
    let method = |name: &'static str, f: crate::gateway::BuiltinCodeFn| unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            name,
            crate::make_builtin_function(name, f),
        );
    };
    let close = |args: &[PyObjectRef]| -> Result<PyObjectRef, crate::PyError> {
        crate::baseobjspace::setdictvalue_native(args[0], SCANDIR_POSITION, pyre_object::w_none());
        Ok(pyre_object::w_none())
    };
    method("__iter__", |args| Ok(args[0]));
    method("__next__", |args| {
        let position = crate::baseobjspace::getattr_str(args[0], SCANDIR_POSITION)?;
        if unsafe { is_none(position) } {
            return Err(crate::PyError::stop_iteration());
        }
        crate::baseobjspace::next(position)
    });
    method("__enter__", |args| Ok(args[0]));
    method("__exit__", close);
    method("close", close);
}

/// `posix.scandir(path=".")` — the entries the seam reports, as the records
/// `os.walk`, `glob` and `shutil` read them through.
///
/// The omitted argument is the `"."` the signature names, and its entries
/// carry the `./name` paths that spelling joins to.  A `bytes` path asks for
/// `bytes` names, as it does for `listdir`.  `fd` is not one of the types the
/// argument accepts: the seam takes a name, so a descriptor names no directory
/// on the other side of it.
fn scandir(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    crate::builtins::kwarg_reject_unknown(kwargs, &["path"], "scandir")?;
    if pos.len() > 1 {
        return Err(crate::PyError::type_error(format!(
            "scandir() takes at most 1 argument ({} given)",
            pos.len()
        )));
    }
    let w_path = crate::builtins::bind_pos_or_kw(pos, kwargs, 0, "path", "scandir", 1)?
        .unwrap_or(pyre_object::w_none());
    let resolved = crate::gateway::fsencode_path_or_fd_nullable_w(w_path, "scandir", false)?;
    let bytes_mode = unsafe { resolved.is_bytes() };
    // An omitted path is the directory `listdir` reads for the same argument,
    // and the prefix its entries are named relative to.
    let dir_bytes: &[u8] = if resolved.as_bytes.is_empty() && unsafe { is_none(w_path) } {
        b"."
    } else {
        &resolved.as_bytes
    };
    let names = crate::importing::source_list_dir(&seam_path(&resolved.as_bytes))
        .map_err(|e| seam_error(&e, resolved.w_path()))?;
    let _roots = pyre_object::gc_roots::push_roots();
    let mut items = pyre_object::gc_roots::RootedItems::new();
    for name in &names {
        items.push(new_dir_entry(bytes_mode, dir_bytes, name));
    }
    let entries_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_list_new(items.take()));
    let iterator_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_instance_new(scandir_iterator_type()));
    let position =
        crate::baseobjspace::iter(pyre_object::gc_roots::shadow_stack_get(entries_slot))?;
    crate::baseobjspace::setdictvalue_native(
        pyre_object::gc_roots::shadow_stack_get(iterator_slot),
        SCANDIR_POSITION,
        position,
    );
    Ok(pyre_object::gc_roots::shadow_stack_get(iterator_slot))
}

/// `O_*` — the darwin/BSD numbering the guest's `errno` table already
/// follows, so a flag and the errno a call refuses it with are drawn from one
/// platform rather than two.
mod oflag {
    pub const RDONLY: i64 = 0x0000;
    pub const WRONLY: i64 = 0x0001;
    pub const RDWR: i64 = 0x0002;
    pub const ACCMODE: i64 = 0x0003;
    pub const NONBLOCK: i64 = 0x0004;
    pub const APPEND: i64 = 0x0008;
    pub const SYNC: i64 = 0x0080;
    pub const NOFOLLOW: i64 = 0x0100;
    pub const CREAT: i64 = 0x0200;
    pub const TRUNC: i64 = 0x0400;
    pub const EXCL: i64 = 0x0800;
    pub const DIRECTORY: i64 = 0x0010_0000;
    pub const CLOEXEC: i64 = 0x0100_0000;
}

/// An open descriptor: the bytes the path resolved to, and how far the guest
/// has read them.
///
/// A directory carries no bytes.  It is opened so that `O_DIRECTORY` can mean
/// what it means, and reading it is `EISDIR` — the same two answers a
/// directory descriptor gives anywhere else.
struct OpenFile {
    data: Vec<u8>,
    pos: u64,
    is_dir: bool,
    /// One of the standard streams 0, 1 and 2. Writes to 1 and 2 reach the
    /// embedder's output and error capture; 0 reads back empty.
    stdio: bool,
}

/// The standard streams hold 0, 1 and 2, so a descriptor this table hands out
/// starts above them.
const FIRST_FD: i32 = 3;
/// The table is the whole descriptor space on this target, so it is what runs
/// out: past this, `open` reports `EMFILE` rather than growing without bound.
const MAX_OPEN_FILES: i32 = 4096;

static OPEN_FILES: std::sync::LazyLock<
    std::sync::Mutex<std::collections::BTreeMap<i32, OpenFile>>,
> = std::sync::LazyLock::new(|| {
    let table = (0..FIRST_FD)
        .map(|fd| {
            let file = OpenFile {
                data: Vec::new(),
                pos: 0,
                is_dir: false,
                stdio: true,
            };
            (fd, file)
        })
        .collect();
    std::sync::Mutex::new(table)
});

/// Take the lowest free number, which is the number `open` is required to
/// return.
fn insert_open_file(file: OpenFile) -> Result<i32, crate::PyError> {
    let mut table = OPEN_FILES.lock().unwrap();
    let fd = (FIRST_FD..FIRST_FD + MAX_OPEN_FILES)
        .find(|n| !table.contains_key(n))
        .ok_or_else(|| {
            crate::PyError::os_error_syscall(
                crate::builtins::wasm_errno::EMFILE,
                pyre_object::w_none(),
            )
        })?;
    table.insert(fd, file);
    Ok(fd)
}

/// Run `f` over the descriptor numbered `fd`, or report the number as one no
/// descriptor is open on.
fn with_raw_fd<R>(
    fd: i32,
    f: impl FnOnce(&mut OpenFile) -> Result<R, crate::PyError>,
) -> Result<R, crate::PyError> {
    let mut table = OPEN_FILES.lock().unwrap();
    let Some(file) = table.get_mut(&fd) else {
        return Err(crate::PyError::os_error_syscall(
            crate::builtins::wasm_errno::EBADF,
            pyre_object::w_none(),
        ));
    };
    f(file)
}

/// [`with_raw_fd`] where the descriptor arrives as a Python argument.
fn with_open_file<R>(
    w_fd: PyObjectRef,
    f: impl FnOnce(&mut OpenFile) -> Result<R, crate::PyError>,
) -> Result<R, crate::PyError> {
    with_raw_fd(crate::baseobjspace::c_int_w(w_fd)?, f)
}

/// Take up to `n` bytes from the descriptor's position, which advances by what
/// was taken.  `None` takes everything left, which is what a sizeless `read()`
/// asks for.
///
/// The `_io` layer reaches its descriptors through here too, so a number
/// `posix.open` returned is a number `io.open(fd)` can read.
pub(crate) fn fd_take(fd: i32, n: Option<usize>) -> Result<Vec<u8>, crate::PyError> {
    with_raw_fd(fd, |file| {
        if file.is_dir {
            return Err(crate::PyError::os_error_syscall(
                crate::builtins::wasm_errno::EISDIR,
                pyre_object::w_none(),
            ));
        }
        let start = (file.pos as usize).min(file.data.len());
        let end = match n {
            Some(n) => start.saturating_add(n).min(file.data.len()),
            None => file.data.len(),
        };
        file.pos = end as u64;
        Ok(file.data[start..end].to_vec())
    })
}

/// [`fd_take`] into a caller's buffer, reporting how much of it was filled.
pub(crate) fn fd_read_into(fd: i32, target: &mut [u8]) -> Result<usize, crate::PyError> {
    let data = fd_take(fd, Some(target.len()))?;
    target[..data.len()].copy_from_slice(&data);
    Ok(data.len())
}

/// Move the descriptor's position and report where it landed.
pub(crate) fn fd_lseek(fd: i32, offset: i64, whence: i32) -> Result<i64, crate::PyError> {
    with_raw_fd(fd, |file| {
        let base = match whence {
            0 => 0,
            1 => file.pos as i64,
            2 => file.data.len() as i64,
            _ => {
                return Err(crate::PyError::os_error_with_errno(
                    crate::builtins::wasm_errno::EINVAL,
                    "lseek: unknown whence",
                ));
            }
        };
        let Some(pos) = base.checked_add(offset).filter(|p| *p >= 0) else {
            return Err(crate::PyError::os_error_with_errno(
                crate::builtins::wasm_errno::EINVAL,
                "lseek: position out of range",
            ));
        };
        file.pos = pos as u64;
        Ok(pos)
    })
}

/// Whether the descriptor is open on a directory, which is the one fact a
/// stream needs before it adopts a number: `EISDIR` is what `io.open(fd)` owes
/// a caller who hands it one.
pub(crate) fn fd_is_dir(fd: i32) -> Result<bool, crate::PyError> {
    with_raw_fd(fd, |file| Ok(file.is_dir))
}

/// Drop the descriptor and the bytes it held.
pub(crate) fn fd_close(fd: i32) -> Result<(), crate::PyError> {
    if OPEN_FILES.lock().unwrap().remove(&fd).is_none() {
        return Err(crate::PyError::os_error_syscall(
            crate::builtins::wasm_errno::EBADF,
            pyre_object::w_none(),
        ));
    }
    Ok(())
}

/// Write `data` to `fd`, the layer `os.write` and `FileIO.write` share.
///
/// 1 and 2 are the embedder's output and error buffers, which is where
/// `print` writes too, so the two stay in order. With no embedder listening
/// the bytes take the host descriptor itself, raw: this is not the layer that
/// rewrites newlines. 0 is not open for writing, and every other open
/// descriptor is the read-only mount: the seam has no writing half.
pub(crate) fn fd_write(fd: i32, data: &[u8]) -> Result<i64, crate::PyError> {
    let stdio = with_raw_fd(fd, |file| Ok(file.stdio))?;
    let errno = match (stdio, fd) {
        (true, 1 | 2) => {
            let delivered = if fd == 1 {
                crate::print_hook_emit_bytes(data)
            } else {
                crate::stderr_hook_emit(data)
            };
            if delivered {
                return Ok(data.len() as i64);
            }
            use std::io::Write;
            let written = if fd == 1 {
                std::io::stdout().write_all(data)
            } else {
                std::io::stderr().write_all(data)
            };
            match written {
                Ok(()) => return Ok(data.len() as i64),
                Err(error) => error
                    .raw_os_error()
                    .unwrap_or(crate::builtins::wasm_errno::EIO),
            }
        }
        // Every other descriptor in the table was opened read-only.
        _ => crate::builtins::wasm_errno::EBADF,
    };
    Err(crate::PyError::os_error_syscall(
        errno,
        pyre_object::w_none(),
    ))
}

/// The path argument of a one-path entry point, with `dir_fd` refused the way
/// `stat` refuses it: the seam takes a name and nothing else.
fn path_arg(
    args: &[PyObjectRef],
    name: &'static str,
) -> Result<crate::gateway::FsEncodedPath, crate::PyError> {
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    crate::builtins::kwarg_reject_unknown(kwargs, &["path", "dir_fd"], name)?;
    if pos.len() > 1 {
        return Err(crate::PyError::type_error(format!(
            "{name}() takes at most 1 positional argument ({} given)",
            pos.len()
        )));
    }
    let Some(w_path) = crate::builtins::bind_pos_or_kw(pos, kwargs, 0, "path", name, 1)? else {
        return Err(crate::PyError::type_error(format!(
            "{name}() missing required argument 'path' (pos 1)"
        )));
    };
    let resolved = crate::gateway::fsencode_path_or_fd_w(w_path, name, false)?;
    if crate::builtins::kwarg_get(kwargs, "dir_fd").is_some_and(|v| !unsafe { is_none(v) }) {
        return Err(crate::PyError::not_implemented(
            "dir_fd unavailable on this platform",
        ));
    }
    Ok(resolved)
}

/// `posix.open(path, flags, mode=0o777, *, dir_fd=None)` — a descriptor over
/// the bytes the seam reports for `path`.
///
/// The whole file is read here rather than at the first `read`: the seam has
/// no way to resume a path part-way, so the descriptor is the only place the
/// bytes can live.  `mode` is accepted and unused — it describes a file that
/// is about to be created, and this seam creates none.
fn open_file(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    crate::builtins::kwarg_reject_unknown(kwargs, &["path", "flags", "mode", "dir_fd"], "open")?;
    if pos.len() > 3 {
        return Err(crate::PyError::type_error(format!(
            "open() takes at most 3 positional arguments ({} given)",
            pos.len()
        )));
    }
    let Some(w_path) = crate::builtins::bind_pos_or_kw(pos, kwargs, 0, "path", "open", 3)? else {
        return Err(crate::PyError::type_error(
            "open() missing required argument 'path' (pos 1)",
        ));
    };
    let Some(w_flags) = crate::builtins::bind_pos_or_kw(pos, kwargs, 1, "flags", "open", 3)? else {
        return Err(crate::PyError::type_error(
            "open() missing required argument 'flags' (pos 2)",
        ));
    };
    let _ = crate::builtins::bind_pos_or_kw(pos, kwargs, 2, "mode", "open", 3)?;
    let resolved = crate::gateway::fsencode_path_or_fd_w(w_path, "open", false)?;
    if crate::builtins::kwarg_get(kwargs, "dir_fd").is_some_and(|v| !unsafe { is_none(v) }) {
        return Err(crate::PyError::not_implemented(
            "dir_fd unavailable on this platform",
        ));
    }
    let flags = crate::baseobjspace::int_w(w_flags)?;
    // Every bit that asks to change the mount, refused where the file is
    // named rather than at the write that would discover it.
    if flags & oflag::ACCMODE != oflag::RDONLY
        || flags & (oflag::CREAT | oflag::TRUNC | oflag::APPEND | oflag::EXCL) != 0
    {
        return Err(crate::PyError::os_error_syscall(
            crate::builtins::wasm_errno::EROFS,
            resolved.w_path(),
        ));
    }
    let path = seam_path(&resolved.as_bytes);
    let is_dir = crate::importing::source_is_dir(&path);
    if flags & oflag::DIRECTORY != 0 && !is_dir {
        return Err(crate::PyError::os_error_syscall(
            crate::builtins::wasm_errno::ENOTDIR,
            resolved.w_path(),
        ));
    }
    let data = if is_dir {
        Vec::new()
    } else {
        crate::importing::read_source_bytes(&path).map_err(|e| seam_error(&e, resolved.w_path()))?
    };
    let fd = insert_open_file(OpenFile {
        data,
        pos: 0,
        is_dir,
        stdio: false,
    })?;
    Ok(pyre_object::w_int_new(fd as i64))
}

/// `posix.close(fd)` — drop the descriptor and the bytes it held.
fn close_file(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    fd_close(crate::baseobjspace::c_int_w(args[0])?)?;
    Ok(pyre_object::w_none())
}

/// `posix.read(fd, length)` — up to `length` bytes from the position, which
/// advances by what was returned.  A short answer at the end is the end.
fn read_file(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let n_signed = crate::baseobjspace::int_w(args[1])?;
    // A negative size would wrap to a huge `usize` (and allocation);
    // os.read rejects it with EINVAL, matching the host read(2).
    if n_signed < 0 {
        return Err(crate::PyError::os_error_with_errno(
            crate::builtins::wasm_errno::EINVAL,
            "read: negative size",
        ));
    }
    let fd = crate::baseobjspace::c_int_w(args[0])?;
    let data = fd_take(fd, Some(n_signed as usize))?;
    Ok(pyre_object::w_bytes_from_bytes(&data))
}

/// `posix.lseek(fd, position, whence)` — the new position.
///
/// A position past the end is a position, not an error: it reads as empty and
/// is what `SEEK_END` with a positive offset names.
fn lseek_file(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let fd = crate::baseobjspace::c_int_w(args[0])?;
    let offset = crate::baseobjspace::int_w(args[1])?;
    let whence = crate::baseobjspace::int_w(args[2])?;
    Ok(pyre_object::w_int_new(fd_lseek(
        fd,
        offset,
        i32::try_from(whence).unwrap_or(i32::MAX),
    )?))
}

/// `posix.fstat(fd)` — the same two facts `stat` reports, taken from the
/// descriptor rather than from a name.
fn fstat_file(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    with_open_file(args[0], |file| {
        Ok(if file.is_dir {
            make_stat_result(S_IFDIR | 0o555, 0)
        } else {
            make_stat_result(S_IFREG | 0o444, file.data.len() as i64)
        })
    })
}

/// `posix.readlink(path, *, dir_fd=None)` — the seam resolves a name and does
/// not report whether it went through a link, so no path it answers for is a
/// symbolic link.  `EINVAL` is what `readlink` answers for one that is not.
fn readlink(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let resolved = path_arg(args, "readlink")?;
    let path = seam_path(&resolved.as_bytes);
    if !crate::importing::source_is_dir(&path)
        && let Err(e) = crate::importing::source_file_size(&path)
    {
        return Err(seam_error(&e, resolved.w_path()));
    }
    Err(crate::PyError::os_error_syscall(
        crate::builtins::wasm_errno::EINVAL,
        resolved.w_path(),
    ))
}

/// The guest's environment.
///
/// This is what a C library's `environ` array is: the store `putenv` and
/// `unsetenv` write to, and the one `posix.environ` is built from once.  `os.py`
/// wraps that dict rather than copying it, so `os.environ` tracks the dict and
/// these two track this map -- the same two-place split every platform has, and
/// the reason a bare `os.putenv` is not visible through `os.environ` there
/// either.  `_create_environ` is what reads this side back, which is how
/// `os.reload_environ()` sees what `putenv` wrote.
///
/// The guest is started with no environment: the runner forwards the few
/// variables the launcher options resolve against rather than an environment
/// block, so this begins empty and holds whatever the program puts in it.
static ENVIRONMENT: std::sync::LazyLock<
    std::sync::Mutex<std::collections::BTreeMap<Vec<u8>, Vec<u8>>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::BTreeMap::new()));

/// A fresh dict holding what the environment holds now.
///
/// Bytes on both sides: `os.py` encodes a key before it stores one and decodes
/// it on the way back, and `os.environb` hands the same dict out undecoded.
fn environ_snapshot() -> PyObjectRef {
    let entries: Vec<(Vec<u8>, Vec<u8>)> = ENVIRONMENT
        .lock()
        .unwrap()
        .iter()
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect();
    let _roots = pyre_object::gc_roots::push_roots();
    let dict_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_dict_new());
    for (name, value) in entries {
        // Key and value pins live across this one `setitem`. The dict stays
        // on the enclosing bracket.
        let _pyre_store_roots = pyre_object::gc_roots::push_roots();
        let key_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(pyre_object::w_bytes_from_bytes(&name));
        let value_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(pyre_object::w_bytes_from_bytes(&value));
        let _ = crate::baseobjspace::setitem(
            pyre_object::gc_roots::shadow_stack_get(dict_slot),
            pyre_object::gc_roots::shadow_stack_get(key_slot),
            pyre_object::gc_roots::shadow_stack_get(value_slot),
        );
    }
    pyre_object::gc_roots::shadow_stack_get(dict_slot)
}

/// The name or value argument of `putenv` / `unsetenv`, as the bytes the
/// environment holds.
///
/// `os.environ` hands these down already encoded, and `os.environb` hands down
/// bytes to begin with, so both spellings arrive here and both are kept as
/// bytes: a name that came from the embedder is a name the guest can set again,
/// whatever it spells.
///
/// The converter's own two refusals are the ones owed here: a TypeError for
/// something that is neither spelling, and a ValueError for a name carrying an
/// embedded NUL, which is the one `test_os` asks for by name.
fn env_bytes_arg(
    w_arg: PyObjectRef,
    name: &'static str,
) -> Result<crate::gateway::FsEncodedPath, crate::PyError> {
    crate::gateway::fsencode_path_or_fd_w(w_arg, name, false)
}

/// The names `setenv` and `unsetenv` refuse outright: an empty one names no
/// variable, and one carrying `=` could not be spelled back out of the
/// environment it went into.  Both report `EINVAL`.
fn check_env_name(name: &[u8]) -> Result<(), crate::PyError> {
    if name.is_empty() || name.contains(&b'=') {
        return Err(crate::PyError::os_error_syscall(
            crate::builtins::wasm_errno::EINVAL,
            pyre_object::w_none(),
        ));
    }
    Ok(())
}

/// `posix.putenv(name, value)` — add or replace one variable.
fn putenv(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    crate::builtins::kwarg_reject_unknown(kwargs, &[], "putenv")?;
    if pos.len() != 2 {
        return Err(crate::PyError::type_error(format!(
            "putenv expected 2 arguments, got {}",
            pos.len()
        )));
    }
    let name = env_bytes_arg(pos[0], "putenv")?;
    // `putenv` is defined over `name=value`, so a name carrying the separator
    // could not be spelled back out of the environment it went into.  That one
    // is a ValueError; an empty name is the syscall's own `EINVAL`.
    if name.as_bytes.contains(&b'=') {
        return Err(crate::PyError::value_error(
            "illegal environment variable name",
        ));
    }
    check_env_name(&name.as_bytes)?;
    let value = env_bytes_arg(pos[1], "putenv")?;
    ENVIRONMENT
        .lock()
        .unwrap()
        .insert(name.as_bytes.clone(), value.as_bytes.clone());
    Ok(pyre_object::w_none())
}

/// `posix.unsetenv(name)` — remove one variable, or leave an absent one absent.
fn unsetenv(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let name = env_bytes_arg(
        args.first()
            .copied()
            .ok_or_else(|| crate::PyError::type_error("unsetenv expected 1 argument, got 0"))?,
        "unsetenv",
    )?;
    // `unsetenv` has no separate report for a name it cannot spell: both the
    // empty one and one carrying `=` come back as `EINVAL`.
    check_env_name(&name.as_bytes)?;
    // A name nothing set is a name already unset, which is what the syscall
    // reports and what `os.environ.__delitem__` relies on so that it can raise
    // the KeyError itself, against the caller's own spelling of the key.
    ENVIRONMENT.lock().unwrap().remove(&name.as_bytes);
    Ok(pyre_object::w_none())
}

/// `posix.uname()` — the record wasi's own `uname` fills in, which names the
/// guest rather than whatever machine the embedder happens to be.
///
/// `platform.uname` reads all five, and `platform._node` takes `nodename`
/// through `socket.gethostname`, which wasi writes over this same call.
fn uname(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let fields = ["wasi", "(none)", "0.0.0", "0.0.0", "wasm32"];
    let mut seq = pyre_object::gc_roots::RootedItems::new();
    for field in fields {
        seq.push(pyre_object::w_str_new_managed(field));
    }
    Ok(crate::_structseq::new_instance(
        super::uname_result_seq_type(),
        seq.take(),
    ))
}

/// What one parameter of a refused entry point is, which decides the
/// conversion a caller's mistake trips before the mount refuses the call.
enum Param {
    /// A path, encoded the way every other entry point encodes one.
    Path(&'static str),
    /// An integer, at the width the entry point declares.
    Int(&'static str),
    /// A file offset: `r_longlong` in PyPy and `Py_off_t` in 3.14's clinic
    /// signature, rather than the narrower `c_int` used by modes.
    Offset(&'static str),
    /// The positional time pair `utime` validates before it reaches the
    /// filesystem operation.
    UTimePair(&'static str),
    /// A value this arm never reads because it has no effect on this target:
    /// the POSIX `symlink` implementation ignores `target_is_directory`.
    Unread(&'static str),
}

impl Param {
    fn name(&self) -> &'static str {
        match self {
            Param::Path(name)
            | Param::Int(name)
            | Param::Offset(name)
            | Param::UTimePair(name)
            | Param::Unread(name) => name,
        }
    }
}

/// Validate the time values `interp_posix.parse_utime_args` consumes before
/// making the mutating call.  A read-only result comes after these conversions:
/// malformed pairs, mutually exclusive spellings and values outside `time_t`
/// do not become EROFS merely because this particular mount cannot write.
fn validate_utime_args(
    w_times: Option<PyObjectRef>,
    w_ns: Option<PyObjectRef>,
) -> Result<(), crate::PyError> {
    let present =
        |value: Option<PyObjectRef>| value.filter(|w_value| !unsafe { is_none(*w_value) });
    let times = present(w_times);
    let ns = present(w_ns);
    if times.is_some() && ns.is_some() {
        return Err(crate::PyError::value_error(
            "utime: you may specify either 'times' or 'ns' but not both",
        ));
    }

    let unpack_two = |w_pair: PyObjectRef, what: &str| {
        if !unsafe { pyre_object::is_tuple(w_pair) }
            || unsafe { pyre_object::w_tuple_len(w_pair) } != 2
        {
            let shape = if what == "times" {
                "either a tuple of two ints or None"
            } else {
                "a tuple of two ints"
            };
            return Err(crate::PyError::type_error(format!(
                "utime: '{what}' must be {shape}"
            )));
        }
        Ok((
            unsafe { pyre_object::w_tuple_getitem(w_pair, 0) }.unwrap(),
            unsafe { pyre_object::w_tuple_getitem(w_pair, 1) }.unwrap(),
        ))
    };
    let time_t_overflow =
        || crate::PyError::overflow_error("timestamp out of range for platform time_t");
    let validate_seconds = |w_value: PyObjectRef| -> Result<(), crate::PyError> {
        if unsafe { pyre_object::is_int_or_long(w_value) } {
            crate::builtins::space_index_w(w_value).map_err(|_| time_t_overflow())?;
            return Ok(());
        }
        let w_float = crate::builtins::builtin_float(&[w_value]).map_err(|error| {
            if error.kind == crate::PyErrorKind::OverflowError {
                time_t_overflow()
            } else {
                crate::PyError::type_error(format!(
                    "argument must be int or float, not {}",
                    crate::type_methods::arg_type_name(w_value)
                ))
            }
        })?;
        let seconds = unsafe { pyre_object::w_float_get_value(w_float) };
        if seconds.is_nan() {
            return Err(crate::PyError::value_error(
                "Invalid value NaN (not a number)",
            ));
        }
        if !(seconds.floor() >= -(2f64.powi(63)) && seconds.floor() < 2f64.powi(63)) {
            return Err(time_t_overflow());
        }
        Ok(())
    };
    let validate_nanoseconds = |w_value: PyObjectRef| -> Result<(), crate::PyError> {
        let w_split = crate::baseobjspace::divmod(w_value, pyre_object::w_int_new(1_000_000_000))?;
        let (Some(w_seconds), Some(w_remainder)) = (unsafe {
            (
                pyre_object::w_tuple_getitem(w_split, 0),
                pyre_object::w_tuple_getitem(w_split, 1),
            )
        }) else {
            return Err(crate::PyError::type_error(
                "utime: divmod() returned a non-pair",
            ));
        };
        crate::builtins::space_index_w(w_seconds).map_err(|error| {
            if error.kind == crate::PyErrorKind::OverflowError {
                time_t_overflow()
            } else {
                error
            }
        })?;
        crate::builtins::space_index_w(w_remainder)?;
        Ok(())
    };

    if let Some(w_pair) = times {
        let (w_access, w_modified) = unpack_two(w_pair, "times")?;
        validate_seconds(w_access)?;
        validate_seconds(w_modified)?;
    } else if let Some(w_pair) = ns {
        let (w_access, w_modified) = unpack_two(w_pair, "ns")?;
        validate_nanoseconds(w_access)?;
        validate_nanoseconds(w_modified)?;
    }
    Ok(())
}

/// The signature a refused entry point still has to bind.
///
/// `EROFS` is the answer to a call this mount cannot carry out, but a call
/// that never named a valid argument list does not reach the mount at all:
/// it is the `TypeError` its binding raises on any other host.
struct WriteSig {
    /// The positional-or-keyword parameters, in order.  The first path among
    /// them is the one the `EROFS` is named by, and it is not called the same
    /// word for all of them -- the one-path calls name it `path` and the
    /// two-path calls name it `src`.
    params: &'static [Param],
    /// How many of `params` a call must supply.
    required: usize,
    /// The keyword-only parameters.  Every `dir_fd` among them has to be
    /// absent, since the seam is handed a name rather than a descriptor to
    /// resolve against, and `follow_symlinks` has to stay true, since nothing
    /// this seam reports is a symbolic link.
    kwonly: &'static [&'static str],
}

/// Reject a keyword-only argument this platform has no way to honour, using
/// the wording the argument clinic's own two refusals carry.
fn check_kwonly(
    kwargs: Option<PyObjectRef>,
    name: &'static str,
    kwonly: &'static [&'static str],
) -> Result<(), crate::PyError> {
    for key in kwonly {
        let Some(w_value) = crate::builtins::kwarg_get(kwargs, key) else {
            continue;
        };
        // `dir_fd_unavailable` names the parameter `dir_fd` whichever of the
        // three spellings the entry point gave it.
        let message = if key.ends_with("dir_fd") {
            (!unsafe { is_none(w_value) }).then(|| "dir_fd unavailable on this platform".to_owned())
        } else if *key == "follow_symlinks" {
            (!crate::baseobjspace::is_true(w_value)?)
                .then(|| format!("{name}: follow_symlinks unavailable on this platform"))
        } else {
            None
        };
        if let Some(message) = message {
            return Err(crate::PyError::not_implemented(message));
        }
    }
    Ok(())
}

/// Bind `sig` against a call and answer with the first path it named.
fn write_path_arg(
    args: &[PyObjectRef],
    name: &'static str,
    sig: &WriteSig,
) -> Result<crate::gateway::FsEncodedPath, crate::PyError> {
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let allowed: Vec<&str> = sig
        .params
        .iter()
        .map(Param::name)
        .chain(sig.kwonly.iter().copied())
        .collect();
    crate::builtins::kwarg_reject_unknown(kwargs, &allowed, name)?;
    if pos.len() > sig.params.len() {
        return Err(crate::PyError::type_error(format!(
            "{name}() takes at most {} positional arguments ({} given)",
            sig.params.len(),
            pos.len()
        )));
    }
    let mut first_path = None;
    let mut utime_times = None;
    let mut has_utime_pair = false;
    for (slot, param) in sig.params.iter().enumerate() {
        let bound =
            crate::builtins::bind_pos_or_kw(pos, kwargs, slot, param.name(), name, slot + 1)?;
        if matches!(param, Param::UTimePair(_)) {
            has_utime_pair = true;
            utime_times = bound;
        }
        let Some(w_value) = bound else {
            if slot < sig.required {
                return Err(crate::PyError::type_error(format!(
                    "{name}() missing required argument '{}' (pos {})",
                    param.name(),
                    slot + 1
                )));
            }
            continue;
        };
        match param {
            Param::Path(_) => {
                let resolved = crate::gateway::fsencode_path_or_fd_w(w_value, name, false)?;
                if first_path.is_none() {
                    first_path = Some(resolved);
                }
            }
            Param::Int(_) => {
                crate::baseobjspace::c_int_w(w_value)?;
            }
            Param::Offset(_) => {
                crate::builtins::space_index_w(w_value)?;
            }
            Param::UTimePair(_) => {}
            Param::Unread(_) => {}
        }
    }
    check_kwonly(kwargs, name, sig.kwonly)?;
    if has_utime_pair {
        validate_utime_args(utime_times, crate::builtins::kwarg_get(kwargs, "ns"))?;
    }
    first_path.ok_or_else(|| {
        crate::PyError::type_error(format!("{name}() missing required path argument"))
    })
}

/// The entry points a read-only mount publishes and refuses, named by the
/// first path they were given.
///
/// wasi carries all of these, so a program that asks whether they exist is
/// answered the way it would be on any wasi host; what it cannot do is write
/// through them, and `EROFS` is what a read-only mount reports for that.
fn refuse_write_path(
    args: &[PyObjectRef],
    name: &'static str,
    sig: &WriteSig,
) -> Result<PyObjectRef, crate::PyError> {
    let resolved = write_path_arg(args, name, sig)?;
    Err(crate::PyError::os_error_syscall(
        crate::builtins::wasm_errno::EROFS,
        resolved.w_path(),
    ))
}

/// `mkdir(path, mode=0o777, *, dir_fd=None)`.
static MKDIR_SIG: WriteSig = WriteSig {
    params: &[Param::Path("path"), Param::Int("mode")],
    required: 1,
    kwonly: &["dir_fd"],
};

/// `chmod(path, mode, *, dir_fd=None, follow_symlinks=True)`.
static CHMOD_SIG: WriteSig = WriteSig {
    params: &[Param::Path("path"), Param::Int("mode")],
    required: 2,
    kwonly: &["dir_fd", "follow_symlinks"],
};

/// `utime(path, times=None, *, ns=..., dir_fd=None, follow_symlinks=True)`.
static UTIME_SIG: WriteSig = WriteSig {
    params: &[Param::Path("path"), Param::UTimePair("times")],
    required: 1,
    kwonly: &["ns", "dir_fd", "follow_symlinks"],
};

/// `truncate(path, length)`.
static TRUNCATE_SIG: WriteSig = WriteSig {
    params: &[Param::Path("path"), Param::Offset("length")],
    required: 2,
    kwonly: &[],
};

/// `rename(src, dst, *, src_dir_fd=None, dst_dir_fd=None)`, which `replace`
/// repeats.
static RENAME_SIG: WriteSig = WriteSig {
    params: &[Param::Path("src"), Param::Path("dst")],
    required: 2,
    kwonly: &["src_dir_fd", "dst_dir_fd"],
};

/// `link(src, dst, *, src_dir_fd=None, dst_dir_fd=None, follow_symlinks=True)`.
static LINK_SIG: WriteSig = WriteSig {
    params: &[Param::Path("src"), Param::Path("dst")],
    required: 2,
    kwonly: &["src_dir_fd", "dst_dir_fd", "follow_symlinks"],
};

/// `symlink(src, dst, target_is_directory=False, *, dir_fd=None)`.
static SYMLINK_SIG: WriteSig = WriteSig {
    params: &[
        Param::Path("src"),
        Param::Path("dst"),
        Param::Unread("target_is_directory"),
    ],
    required: 2,
    kwonly: &["dir_fd"],
};

/// Two of the four `access` mode bits, which the constant table publishes
/// under the same names.
#[cfg(feature = "host_env")]
const X_OK: i64 = rustpython_host_env::os::X_OK as i64;
#[cfg(not(feature = "host_env"))]
const X_OK: i64 = 1;
#[cfg(feature = "host_env")]
const W_OK: i64 = rustpython_host_env::os::W_OK as i64;
#[cfg(not(feature = "host_env"))]
const W_OK: i64 = 2;
#[cfg(feature = "host_env")]
const F_OK: i64 = rustpython_host_env::os::F_OK as i64;
#[cfg(not(feature = "host_env"))]
const F_OK: i64 = 0;
#[cfg(feature = "host_env")]
const R_OK: i64 = rustpython_host_env::os::R_OK as i64;
#[cfg(not(feature = "host_env"))]
const R_OK: i64 = 4;

/// `posix.access(path, mode, ...)` — whether the name is there and the access
/// asked for is one this mount grants.
///
/// The mount is read-only, so `W_OK` is false for everything; `X_OK` is true
/// for a directory, which is the permission that makes one traversable, and
/// false for a file, since nothing here can be executed.
fn access(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    crate::builtins::kwarg_reject_unknown(
        kwargs,
        &["path", "mode", "dir_fd", "effective_ids", "follow_symlinks"],
        "access",
    )?;
    if pos.len() > 2 {
        return Err(crate::PyError::type_error(format!(
            "access() takes at most 2 positional arguments ({} given)",
            pos.len()
        )));
    }
    let Some(w_path) = crate::builtins::bind_pos_or_kw(pos, kwargs, 0, "path", "access", 2)? else {
        return Err(crate::PyError::type_error(
            "access() missing required argument 'path' (pos 1)",
        ));
    };
    let Some(w_mode) = crate::builtins::bind_pos_or_kw(pos, kwargs, 1, "mode", "access", 2)? else {
        return Err(crate::PyError::type_error(
            "access() missing required argument 'mode' (pos 2)",
        ));
    };
    let mode = i64::from(crate::baseobjspace::c_int_w(w_mode)?);
    let resolved = crate::gateway::fsencode_path_or_fd_w(w_path, "access", false)?;
    check_kwonly(kwargs, "access", &["dir_fd", "follow_symlinks"])?;
    // The seam answers for the caller, and there is no second identity here
    // to answer for instead.  Do not turn an exception from a caller's
    // `__bool__` into the default false spelling.
    if let Some(w_effective_ids) = crate::builtins::kwarg_get(kwargs, "effective_ids") {
        if crate::baseobjspace::is_true(w_effective_ids)? {
            return Err(crate::PyError::not_implemented(
                "access: effective_ids unavailable on this platform",
            ));
        }
    }
    let path = seam_path(&resolved.as_bytes);
    let is_dir = crate::importing::source_is_dir(&path);
    let exists = is_dir || crate::importing::source_file_size(&path).is_ok();
    let granted = exists && (mode & W_OK) == 0 && ((mode & X_OK) == 0 || is_dir);
    Ok(pyre_object::w_bool_from(granted))
}

/// `posix.write(fd, data)`; see [`fd_write`] for what each descriptor does
/// with the bytes.
fn write(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    crate::builtins::kwarg_reject_unknown(kwargs, &[], "write")?;
    if pos.len() != 2 {
        return Err(crate::PyError::type_error(format!(
            "write expected 2 arguments, got {}",
            pos.len()
        )));
    }
    let fd = crate::baseobjspace::c_int_w(pos[0])?;
    let data = unsafe {
        if !pyre_object::bytesobject::is_bytes_like(pos[1]) {
            return Err(crate::PyError::type_error(
                "a bytes-like object is required, not 'str'",
            ));
        }
        pyre_object::bytesobject::bytes_like_data(pos[1])
    };
    let n = fd_write(fd, data)?;
    Ok(pyre_object::w_int_new(n))
}

/// `posix.fsync(fd)` / `posix.fdatasync(fd)` — nothing to flush, but the
/// descriptor is still checked: a read-only file has no dirty pages, and a
/// number naming no descriptor is `EBADF` here as everywhere.
fn sync_fd(args: &[PyObjectRef], name: &'static str) -> Result<PyObjectRef, crate::PyError> {
    let Some(w_fd) = args.first().copied() else {
        return Err(crate::PyError::type_error(format!(
            "{name} expected 1 argument, got 0"
        )));
    };
    let fd = crate::baseobjspace::c_int_w(w_fd)?;
    with_raw_fd(fd, |_| Ok(()))?;
    Ok(pyre_object::w_none())
}

/// `posix.unlink(path, *, dir_fd=None)` / `posix.rmdir(path, *, dir_fd=None)`
/// — the entry points a read-only mount publishes and refuses.
fn refuse_write(args: &[PyObjectRef], name: &'static str) -> Result<PyObjectRef, crate::PyError> {
    let resolved = path_arg(args, name)?;
    Err(crate::PyError::os_error_syscall(
        crate::builtins::wasm_errno::EROFS,
        resolved.w_path(),
    ))
}

pub fn register_module(ns: PyObjectRef) -> Result<(), crate::PyError> {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    // `os._create_environ_mapping` binds this dict itself rather than copying
    // it, so `os.environ._data` is this object and `os.environ` is a live view
    // of it.  The guest is started with no environment — the runner hands the
    // interpreter the few variables it forwards, rather than an environment
    // block — so this begins empty and holds whatever the program puts in it.
    crate::__pyre_put_new!(ns_slot, "environ", environ_snapshot());
    // `os.py` builds `supports_dir_fd` and friends only when this exists, and
    // then reads `stat` out of its own namespace to seed `supports_fd`.  Empty:
    // no call here takes a directory or a descriptor argument.
    crate::__pyre_put_new!(ns_slot, "_have_functions", pyre_object::w_list_new(vec![]));
    crate::__pyre_put_new!(ns_slot, "stat_result", super::stat_result_seq_type());
    // `posixmodule_exec` publishes this type on every platform, and only the
    // `get_terminal_size` that fills one is guarded: `shutil.get_terminal_size`
    // catches the AttributeError from the missing call and builds its fallback
    // out of the type, so a target with neither raises from the handler.
    crate::__pyre_put_new!(ns_slot, "terminal_size", super::terminal_size_seq_type());
    // `moduledef.py` lists these three as appleveldefs with no platform test,
    // and `posixmodule_exec` creates them unconditionally too: the types exist
    // wherever `posix` does, even where the call that fills one is absent.
    crate::__pyre_put_new!(ns_slot, "statvfs_result", super::statvfs_result_seq_type());
    crate::__pyre_put_new!(ns_slot, "times_result", super::times_result_seq_type());
    crate::__pyre_put_new!(ns_slot, "uname_result", super::uname_result_seq_type());
    // `follow_symlinks` and `dir_fd` are keyword-only, so neither entry point
    // can take the fixed-arity carrier that rejects keywords.
    crate::__pyre_put_new!(
        ns_slot,
        "stat",
        crate::make_builtin_function("stat", |args| stat(args, true))
    );
    crate::__pyre_put_new!(
        ns_slot,
        "lstat",
        crate::make_builtin_function("lstat", |args| stat(args, false))
    );
    crate::__pyre_put_new!(
        ns_slot,
        "listdir",
        crate::make_builtin_function("listdir", listdir)
    );
    crate::__pyre_put_new!(
        ns_slot,
        "scandir",
        crate::make_builtin_function("scandir", scandir)
    );
    crate::__pyre_put_new!(
        ns_slot,
        "chdir",
        crate::make_builtin_function("chdir", chdir)
    );
    // `os.environ.__setitem__` calls `putenv` by name, so a target without it
    // cannot set a variable at all.
    crate::__pyre_put_new!(
        ns_slot,
        "putenv",
        crate::make_builtin_function("putenv", putenv)
    );
    // `os.reload_environ` exists only where this does, and it is the one reader
    // that can see what a bare `putenv` wrote.
    crate::__pyre_put_new!(
        ns_slot,
        "_create_environ",
        crate::make_builtin_function_with_arity(
            "_create_environ",
            |_args| Ok(environ_snapshot()),
            0,
        )
    );
    crate::__pyre_put_new!(
        ns_slot,
        "unsetenv",
        crate::make_builtin_function_with_arity("unsetenv", unsetenv, 1)
    );
    // One wasm instance, one process, and it is not one the embedder named.
    crate::__pyre_put_new!(
        ns_slot,
        "getpid",
        crate::make_builtin_function_with_arity("getpid", |_args| Ok(pyre_object::w_int_new(1)), 0,)
    );
    // No terminal is reachable through the seam, so no descriptor is one.  The
    // answer is False rather than an error for the same reason it is on every
    // other platform: `isatty` reports, it does not validate.
    crate::__pyre_put_new!(
        ns_slot,
        "isatty",
        crate::make_builtin_function_with_arity(
            "isatty",
            |_args| Ok(pyre_object::w_bool_from(false)),
            1,
        )
    );
    // `os.walk` catches `OSError` from the iterator and `shutil` calls
    // `isinstance(entry, os.DirEntry)`, so both the type and the entries it
    // stamps are published.
    crate::__pyre_put_new!(ns_slot, "DirEntry", dir_entry_type());
    // `_bootstrap_external` reads `_os.fspath`, not `os.fspath`, so the
    // pure-Python fallback `os.py` installs when `posix` lacks it does not
    // stand in for the import machinery.  The protocol itself touches no
    // filesystem, so both arms publish the one implementation.
    crate::__pyre_put_new!(
        ns_slot,
        "fspath",
        crate::make_builtin_function_with_arity(
            "fspath",
            |args| super::fspath(args.first().copied().unwrap_or(pyre_object::w_none())),
            1,
        )
    );
    // `os.process_cpu_count` is an alias for this one, and reads it at module
    // level, so its absence would stop `import os` outright.  One: the guest is
    // a single wasm instance.
    crate::__pyre_put_new!(
        ns_slot,
        "cpu_count",
        crate::make_builtin_function_with_arity(
            "cpu_count",
            |_args| Ok(pyre_object::w_int_new(1)),
            0,
        )
    );
    // The seam reports the embedder's working directory; without one there is
    // no directory to name, and `""` is what the syscall arm returns when the
    // host refuses too.  `posixpath.abspath` is this call, so every relative
    // path the stdlib resolves goes through it.
    crate::__pyre_put_new!(
        ns_slot,
        "getcwd",
        crate::make_builtin_function_with_arity(
            "getcwd",
            |_args| Ok(crate::gateway::fsdecode_filename_bytes(&cwd_bytes())),
            0,
        )
    );
    crate::__pyre_put_new!(
        ns_slot,
        "getcwdb",
        crate::make_builtin_function_with_arity(
            "getcwdb",
            |_args| Ok(pyre_object::w_bytes_from_bytes(&cwd_bytes())),
            0,
        )
    );
    crate::__pyre_put_new!(
        ns_slot,
        "urandom",
        crate::make_builtin_function_with_arity("urandom", urandom, 1)
    );
    // ── the descriptor half ──
    crate::__pyre_put_new!(
        ns_slot,
        "open",
        crate::make_builtin_function("open", open_file)
    );
    crate::__pyre_put_new!(
        ns_slot,
        "close",
        crate::make_builtin_function_with_arity("close", close_file, 1)
    );
    crate::__pyre_put_new!(
        ns_slot,
        "read",
        crate::make_builtin_function_with_arity("read", read_file, 2)
    );
    crate::__pyre_put_new!(
        ns_slot,
        "lseek",
        crate::make_builtin_function_with_arity("lseek", lseek_file, 3)
    );
    crate::__pyre_put_new!(
        ns_slot,
        "fstat",
        crate::make_builtin_function_with_arity("fstat", fstat_file, 1)
    );
    crate::__pyre_put_new!(
        ns_slot,
        "readlink",
        crate::make_builtin_function("readlink", readlink)
    );
    crate::__pyre_put_new!(
        ns_slot,
        "unlink",
        crate::make_builtin_function("unlink", |args| refuse_write(args, "unlink"))
    );
    // `os.remove` is the same call under its other name on every platform.
    crate::__pyre_put_new!(
        ns_slot,
        "remove",
        crate::make_builtin_function("remove", |args| refuse_write(args, "remove"))
    );
    crate::__pyre_put_new!(
        ns_slot,
        "rmdir",
        crate::make_builtin_function("rmdir", |args| refuse_write(args, "rmdir"))
    );
    // The rest of wasi's writing surface, published so that a program can ask
    // for it and refused because the mount is read-only.  `mkdir` and `chmod`
    // take a mode after the path, `utime` a pair of times, `truncate` a
    // length: each is converted before the mount's EROFS is reported, just as
    // the syscall-backed entry points convert them before making their call.
    crate::__pyre_put_new!(
        ns_slot,
        "mkdir",
        crate::make_builtin_function("mkdir", |args| {
            refuse_write_path(args, "mkdir", &MKDIR_SIG)
        })
    );
    crate::__pyre_put_new!(
        ns_slot,
        "chmod",
        crate::make_builtin_function("chmod", |args| {
            refuse_write_path(args, "chmod", &CHMOD_SIG)
        })
    );
    crate::__pyre_put_new!(
        ns_slot,
        "utime",
        crate::make_builtin_function("utime", |args| {
            refuse_write_path(args, "utime", &UTIME_SIG)
        })
    );
    crate::__pyre_put_new!(
        ns_slot,
        "truncate",
        crate::make_builtin_function("truncate", |args| {
            refuse_write_path(args, "truncate", &TRUNCATE_SIG)
        })
    );
    // The two-path calls, named by their source the way `rename` reports it.
    crate::__pyre_put_new!(
        ns_slot,
        "rename",
        crate::make_builtin_function("rename", |args| {
            refuse_write_path(args, "rename", &RENAME_SIG)
        })
    );
    crate::__pyre_put_new!(
        ns_slot,
        "replace",
        crate::make_builtin_function("replace", |args| {
            refuse_write_path(args, "replace", &RENAME_SIG)
        })
    );
    crate::__pyre_put_new!(
        ns_slot,
        "link",
        crate::make_builtin_function("link", |args| refuse_write_path(args, "link", &LINK_SIG))
    );
    crate::__pyre_put_new!(
        ns_slot,
        "symlink",
        crate::make_builtin_function("symlink", |args| {
            refuse_write_path(args, "symlink", &SYMLINK_SIG)
        })
    );
    crate::__pyre_put_new!(
        ns_slot,
        "ftruncate",
        crate::make_builtin_function_with_arity(
            "ftruncate",
            |args| {
                // `interp_posix.ftruncate` unwraps `fd=c_int,
                // length=r_longlong` before reaching `rposix.ftruncate`; the
                // descriptor lookup happens only after both conversions.
                let fd = crate::baseobjspace::c_int_w(args[0])?;
                crate::builtins::space_index_w(args[1])?;
                if fd != 1 && fd != 2 {
                    with_raw_fd(fd, |_| Ok(()))?;
                }
                Err(crate::PyError::os_error_syscall(
                    crate::builtins::wasm_errno::EROFS,
                    pyre_object::w_none(),
                ))
            },
            2,
        )
    );
    crate::__pyre_put_new!(
        ns_slot,
        "fsync",
        crate::make_builtin_function_with_arity("fsync", |args| sync_fd(args, "fsync"), 1)
    );
    crate::__pyre_put_new!(
        ns_slot,
        "fdatasync",
        crate::make_builtin_function_with_arity("fdatasync", |args| sync_fd(args, "fdatasync"), 1,)
    );
    crate::__pyre_put_new!(
        ns_slot,
        "access",
        crate::make_builtin_function("access", access)
    );
    crate::__pyre_put_new!(
        ns_slot,
        "write",
        crate::make_builtin_function("write", write)
    );
    crate::__pyre_put_new!(
        ns_slot,
        "uname",
        crate::make_builtin_function_with_arity("uname", uname, 0)
    );
    // `_io` asks this of every descriptor it wraps; no terminal is reachable,
    // and a descriptor that is not one has no encoding of its own.
    crate::__pyre_put_new!(
        ns_slot,
        "device_encoding",
        crate::make_builtin_function_with_arity(
            "device_encoding",
            |args| {
                // `interp_posix.device_encoding` takes `fd=c_int` even on a
                // platform where no descriptor can be a terminal.
                crate::baseobjspace::c_int_w(args[0])?;
                Ok(pyre_object::w_none())
            },
            1,
        )
    );
    // ── the constants those entry points are called with ──
    let constants: &[(&str, i64)] = &[
        ("O_RDONLY", oflag::RDONLY),
        ("O_WRONLY", oflag::WRONLY),
        ("O_RDWR", oflag::RDWR),
        ("O_ACCMODE", oflag::ACCMODE),
        ("O_NONBLOCK", oflag::NONBLOCK),
        ("O_APPEND", oflag::APPEND),
        ("O_SYNC", oflag::SYNC),
        ("O_NOFOLLOW", oflag::NOFOLLOW),
        ("O_CREAT", oflag::CREAT),
        ("O_TRUNC", oflag::TRUNC),
        ("O_EXCL", oflag::EXCL),
        ("O_DIRECTORY", oflag::DIRECTORY),
        ("O_CLOEXEC", oflag::CLOEXEC),
        ("F_OK", F_OK),
        ("X_OK", X_OK),
        ("W_OK", W_OK),
        ("R_OK", R_OK),
        ("SEEK_SET", 0),
        ("SEEK_CUR", 1),
        ("SEEK_END", 2),
    ];
    for (name, value) in constants {
        crate::__pyre_put_new!(ns_slot, name, pyre_object::w_int_new(*value));
    }
    Ok(())
}

/// `posix.urandom(n)` — the same entry point the syscall arm publishes, over
/// the entropy the embedder supplies.
///
/// `random.Random()` and `secrets` are both this call, so a guest without it
/// cannot import either; the bytes are the host's, not a stream the guest
/// generates, because that is what the callers are asking for.
fn urandom(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::gateway::check_declared_arity("urandom", 1, args.len())?;
    let n = crate::builtins::space_index_w(args[0])?;
    if n < 0 {
        return Err(crate::PyError::value_error("negative argument not allowed"));
    }
    let n =
        usize::try_from(n).map_err(|_| crate::PyError::overflow_error("argument out of range"))?;
    // Reporting an entropy failure rather than absorbing it: the alternative
    // is handing a caller who asked for unpredictable bytes a buffer of zeros.
    let buf = crate::importing::host::os::urandom(n).map_err(|e| {
        crate::PyError::os_error_syscall(
            crate::builtins::io_error_posix_errno(&e, crate::builtins::wasm_errno::ENOTSUP),
            pyre_object::w_none(),
        )
    })?;
    Ok(pyre_object::w_bytes_from_bytes(&buf))
}
