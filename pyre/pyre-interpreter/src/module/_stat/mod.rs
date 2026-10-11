//! `_stat` accelerator module — `Modules/_stat.c`.
//!
//! PyPy has no interpreter-level `_stat` at all: `lib-python/3/stat.py`
//! defines every constant and every `S_IS*` helper itself and only then does
//! `from _stat import *`.  So this module exists to replace those portable
//! defaults with the values the build target's `<sys/stat.h>` actually uses —
//! `S_IFWHT` and the BSD `chflags` masks on the Apple targets, the file
//! attribute bits on Windows.  Names it does not export keep the app-level
//! definition, which is how `stat.SF_RESTRICTED` survives.
//!
//! RustPython's corresponding owner is `crates/vm/src/stdlib/_stat.rs`, whose
//! `libc_const!` shape this file follows: `_stat.c` reads each constant out of
//! the platform header and compiles in a portable default only where the
//! header is silent, and [`libc_const`] is that `#ifndef` cascade — the host
//! value through `libc`, otherwise the literal `_stat.c` falls back to.

use pyre_object::*;

/// `mode_t`, the width `_PyLong_AsMode_t` narrows to.  Its size is
/// observable: 16-bit on the Apple targets, so a mode above `0xffff` is
/// rejected there and accepted on Linux.
#[cfg(unix)]
type Mode = libc::mode_t;
/// Windows has no `mode_t`; `_stat.c` typedefs it to `unsigned short`.
#[cfg(windows)]
type Mode = u16;
#[cfg(not(any(unix, windows)))]
type Mode = u32;

/// The host's definition of a `<sys/stat.h>` macro, or the value `_stat.c`
/// compiles in when the platform does not define it.
macro_rules! libc_const {
    ($cfg:meta, $name:ident, $fallback:expr) => {{
        #[cfg($cfg)]
        {
            libc::$name
        }
        #[cfg(not($cfg))]
        {
            $fallback
        }
    }};
}

// S_IFXXX constants (file types).  Only the names are defined by POSIX, not
// their values, but every platform pyre builds for agrees on the common ones.
const S_IFDIR: Mode = libc_const!(unix, S_IFDIR, 0o040000);
const S_IFCHR: Mode = libc_const!(unix, S_IFCHR, 0o020000);
const S_IFBLK: Mode = libc_const!(unix, S_IFBLK, 0o060000);
const S_IFREG: Mode = libc_const!(unix, S_IFREG, 0o100000);
const S_IFIFO: Mode = libc_const!(unix, S_IFIFO, 0o010000);
const S_IFLNK: Mode = libc_const!(unix, S_IFLNK, 0o120000);
const S_IFSOCK: Mode = libc_const!(unix, S_IFSOCK, 0o140000);
const S_IFMT: Mode = libc_const!(unix, S_IFMT, 0o170000);

/// A file type no platform pyre builds for names.  `_stat.c` defines both the
/// constant and its `S_IS*` macro as a literal `0`, which [`is_format`]
/// reproduces — comparing against `0` instead would answer "yes" for every
/// mode carrying no type bits.
const S_IFDOOR: Mode = 0;
const S_IFPORT: Mode = 0;

/// `libc` carries no `S_IFWHT`; the value is the one the Apple headers use.
const S_IFWHT: Mode = if cfg!(target_vendor = "apple") {
    0o160000
} else {
    0
};

// S_I* file permissions.  The permission bit values are defined by POSIX.
const S_ISUID: Mode = libc_const!(unix, S_ISUID, 0o4000);
const S_ISGID: Mode = libc_const!(unix, S_ISGID, 0o2000);
const S_ENFMT: Mode = S_ISGID;
const S_ISVTX: Mode = libc_const!(unix, S_ISVTX, 0o1000);
const S_IRWXU: Mode = libc_const!(unix, S_IRWXU, 0o0700);
const S_IRUSR: Mode = libc_const!(unix, S_IRUSR, 0o0400);
const S_IWUSR: Mode = libc_const!(unix, S_IWUSR, 0o0200);
const S_IXUSR: Mode = libc_const!(unix, S_IXUSR, 0o0100);
const S_IRWXG: Mode = libc_const!(unix, S_IRWXG, 0o0070);
const S_IRGRP: Mode = libc_const!(unix, S_IRGRP, 0o0040);
const S_IWGRP: Mode = libc_const!(unix, S_IWGRP, 0o0020);
const S_IXGRP: Mode = libc_const!(unix, S_IXGRP, 0o0010);
const S_IRWXO: Mode = libc_const!(unix, S_IRWXO, 0o0007);
const S_IROTH: Mode = libc_const!(unix, S_IROTH, 0o0004);
const S_IWOTH: Mode = libc_const!(unix, S_IWOTH, 0o0002);
const S_IXOTH: Mode = libc_const!(unix, S_IXOTH, 0o0001);

// The Unix V7 synonyms.  `libc` carries them for the BSD-derived targets
// only — glibc keeps them behind `__USE_MISC` — so Linux takes the same
// default `_stat.c` compiles in, which is the value glibc defines anyway.
const S_IREAD: Mode = libc_const!(target_vendor = "apple", S_IREAD, 0o0400);
const S_IWRITE: Mode = libc_const!(target_vendor = "apple", S_IWRITE, 0o0200);
const S_IEXEC: Mode = libc_const!(target_vendor = "apple", S_IEXEC, 0o0100);

// Names for file flags.  These are BSD `chflags` bits, so only the Apple
// targets have header definitions; everywhere else `_stat.c`'s defaults are
// all there is.
const UF_SETTABLE: u32 = libc_const!(target_vendor = "apple", UF_SETTABLE, 0x0000ffff);
const UF_NODUMP: u32 = libc_const!(target_vendor = "apple", UF_NODUMP, 0x00000001);
const UF_IMMUTABLE: u32 = libc_const!(target_vendor = "apple", UF_IMMUTABLE, 0x00000002);
const UF_APPEND: u32 = libc_const!(target_vendor = "apple", UF_APPEND, 0x00000004);
const UF_OPAQUE: u32 = libc_const!(target_vendor = "apple", UF_OPAQUE, 0x00000008);
const UF_COMPRESSED: u32 = libc_const!(target_vendor = "apple", UF_COMPRESSED, 0x00000020);
const UF_TRACKED: u32 = libc_const!(target_vendor = "apple", UF_TRACKED, 0x00000040);
const UF_HIDDEN: u32 = libc_const!(target_vendor = "apple", UF_HIDDEN, 0x00008000);
const SF_ARCHIVED: u32 = libc_const!(target_vendor = "apple", SF_ARCHIVED, 0x00010000);
const SF_IMMUTABLE: u32 = libc_const!(target_vendor = "apple", SF_IMMUTABLE, 0x00020000);
const SF_APPEND: u32 = libc_const!(target_vendor = "apple", SF_APPEND, 0x00040000);

// Flags `libc` does not carry for any target pyre builds for.
const UF_NOUNLINK: u32 = 0x00000010;
const UF_DATAVAULT: u32 = 0x00000080;
const SF_NOUNLINK: u32 = 0x00100000;
const SF_SNAPSHOT: u32 = 0x00200000;
const SF_FIRMLINK: u32 = 0x00800000;
const SF_DATALESS: u32 = 0x40000000;

/// The Apple headers reserve the top two flag bits for the synthetic flags,
/// so the super-user mask stops short of them.
///
/// `_stat.c` publishes each flag with `PyModule_AddIntMacro`, whose value
/// parameter is a C `long`.  `0xffff0000` does not fit the 32-bit `long` of an
/// LLP64 target, so Windows publishes these bits as `-65536` while an LP64
/// target publishes `4294901760`.
const SF_SETTABLE: std::ffi::c_long =
    libc_const!(target_vendor = "apple", SF_SETTABLE, 0xffff_0000u32) as std::ffi::c_long;

#[cfg(target_vendor = "apple")]
const SF_SUPPORTED: u32 = 0x009f0000;
#[cfg(target_vendor = "apple")]
const SF_SYNTHETIC: u32 = 0xc0000000;

/// The `st_*` field positions of the 10-tuple form of a stat result.
const ST_CONSTANTS: [&str; 10] = [
    "ST_MODE", "ST_INO", "ST_DEV", "ST_NLINK", "ST_UID", "ST_GID", "ST_SIZE", "ST_ATIME",
    "ST_MTIME", "ST_CTIME",
];

#[cfg(all(windows, feature = "host_env"))]
use rustpython_host_env::nt as host_nt;

#[cfg(all(windows, feature = "host_env"))]
const FILE_ATTRIBUTES: [(&str, i64); 17] = [
    (
        "FILE_ATTRIBUTE_ARCHIVE",
        host_nt::FILE_ATTRIBUTE_ARCHIVE as i64,
    ),
    (
        "FILE_ATTRIBUTE_COMPRESSED",
        host_nt::FILE_ATTRIBUTE_COMPRESSED as i64,
    ),
    (
        "FILE_ATTRIBUTE_DEVICE",
        host_nt::FILE_ATTRIBUTE_DEVICE as i64,
    ),
    (
        "FILE_ATTRIBUTE_DIRECTORY",
        host_nt::FILE_ATTRIBUTE_DIRECTORY as i64,
    ),
    (
        "FILE_ATTRIBUTE_ENCRYPTED",
        host_nt::FILE_ATTRIBUTE_ENCRYPTED as i64,
    ),
    (
        "FILE_ATTRIBUTE_HIDDEN",
        host_nt::FILE_ATTRIBUTE_HIDDEN as i64,
    ),
    (
        "FILE_ATTRIBUTE_INTEGRITY_STREAM",
        host_nt::FILE_ATTRIBUTE_INTEGRITY_STREAM as i64,
    ),
    (
        "FILE_ATTRIBUTE_NORMAL",
        host_nt::FILE_ATTRIBUTE_NORMAL as i64,
    ),
    (
        "FILE_ATTRIBUTE_NOT_CONTENT_INDEXED",
        host_nt::FILE_ATTRIBUTE_NOT_CONTENT_INDEXED as i64,
    ),
    (
        "FILE_ATTRIBUTE_NO_SCRUB_DATA",
        host_nt::FILE_ATTRIBUTE_NO_SCRUB_DATA as i64,
    ),
    (
        "FILE_ATTRIBUTE_OFFLINE",
        host_nt::FILE_ATTRIBUTE_OFFLINE as i64,
    ),
    (
        "FILE_ATTRIBUTE_READONLY",
        host_nt::FILE_ATTRIBUTE_READONLY as i64,
    ),
    (
        "FILE_ATTRIBUTE_REPARSE_POINT",
        host_nt::FILE_ATTRIBUTE_REPARSE_POINT as i64,
    ),
    (
        "FILE_ATTRIBUTE_SPARSE_FILE",
        host_nt::FILE_ATTRIBUTE_SPARSE_FILE as i64,
    ),
    (
        "FILE_ATTRIBUTE_SYSTEM",
        host_nt::FILE_ATTRIBUTE_SYSTEM as i64,
    ),
    (
        "FILE_ATTRIBUTE_TEMPORARY",
        host_nt::FILE_ATTRIBUTE_TEMPORARY as i64,
    ),
    (
        "FILE_ATTRIBUTE_VIRTUAL",
        host_nt::FILE_ATTRIBUTE_VIRTUAL as i64,
    ),
];

#[cfg(all(windows, feature = "host_env"))]
const IO_REPARSE_TAGS: [(&str, i64); 3] = [
    (
        "IO_REPARSE_TAG_SYMLINK",
        host_nt::IO_REPARSE_TAG_SYMLINK as i64,
    ),
    (
        "IO_REPARSE_TAG_MOUNT_POINT",
        host_nt::IO_REPARSE_TAG_MOUNT_POINT as i64,
    ),
    (
        "IO_REPARSE_TAG_APPEXECLINK",
        host_nt::IO_REPARSE_TAG_APPEXECLINK as i64,
    ),
];
#[cfg(all(windows, not(feature = "host_env")))]
const IO_REPARSE_TAGS: [(&str, i64); 3] = [
    ("IO_REPARSE_TAG_SYMLINK", 0xa000_000c),
    ("IO_REPARSE_TAG_MOUNT_POINT", 0xa000_0003),
    ("IO_REPARSE_TAG_APPEXECLINK", 0x8000_001b),
];

/// `_PyLong_AsMode_t` — take the `__index__` of the argument, refuse a
/// negative, and refuse a value the platform's `mode_t` cannot hold.
fn mode_t_w(value: PyObjectRef) -> Result<Mode, pyre_interpreter::PyError> {
    let index = pyre_interpreter::baseobjspace::space_index(value)?;
    let value = pyre_interpreter::baseobjspace::int_w(index)?;
    if value < 0 {
        return Err(pyre_interpreter::PyError::overflow_error(
            "can't convert negative value to unsigned int",
        ));
    }
    let mode = value as Mode;
    if i64::from(mode) != value {
        return Err(pyre_interpreter::PyError::overflow_error(
            "mode out of range",
        ));
    }
    Ok(mode)
}

fn argument_mode(args: &[PyObjectRef], function: &str) -> Result<Mode, pyre_interpreter::PyError> {
    let mode = args.first().copied().ok_or_else(|| {
        pyre_interpreter::PyError::type_error(format!(
            "{function}() takes exactly one argument (0 given)"
        ))
    })?;
    mode_t_w(mode)
}

/// The `S_ISXXX()` family.  A `format` of `0` is a file type this platform
/// does not name, whose macro `_stat.c` compiles as a constant `0`.
fn is_format(
    args: &[PyObjectRef],
    format: Mode,
    function: &str,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let mode = argument_mode(args, function)?;
    Ok(w_bool_from(format != 0 && mode & S_IFMT == format))
}

fn s_imode(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    /* From Python's stat.py */
    const S_IMODE: Mode = 0o7777;
    Ok(w_int_new(i64::from(
        argument_mode(args, "S_IMODE")? & S_IMODE,
    )))
}

fn s_ifmt(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    Ok(w_int_new(i64::from(
        argument_mode(args, "S_IFMT")? & S_IFMT,
    )))
}

/* file type chars according to
http://en.wikibooks.org/wiki/C_Programming/POSIX_Reference/sys/stat.h */
fn filetype(mode: Mode) -> u8 {
    let format = mode & S_IFMT;
    let is = |candidate: Mode| candidate != 0 && format == candidate;
    /* common cases first */
    if is(S_IFREG) {
        b'-'
    } else if is(S_IFDIR) {
        b'd'
    } else if is(S_IFLNK) {
        b'l'
    /* special files */
    } else if is(S_IFBLK) {
        b'b'
    } else if is(S_IFCHR) {
        b'c'
    } else if is(S_IFIFO) {
        b'p'
    } else if is(S_IFSOCK) {
        b's'
    /* non-standard types */
    } else if is(S_IFDOOR) {
        b'D'
    } else if is(S_IFPORT) {
        b'P'
    } else if is(S_IFWHT) {
        b'w'
    /* unknown */
    } else {
        b'?'
    }
}

fn fileperm(mode: Mode, buf: &mut [u8; 9]) {
    let bit = |mask: Mode, set: u8| if mode & mask != 0 { set } else { b'-' };
    // The set-user/group-ID and sticky bits reuse the execute column, in
    // upper case when the execute bit itself is clear.
    let special = |mask: Mode, execute: Mode, on: u8, off: u8| {
        if mode & mask != 0 {
            if mode & execute != 0 { on } else { off }
        } else {
            bit(execute, b'x')
        }
    };
    buf[0] = bit(S_IRUSR, b'r');
    buf[1] = bit(S_IWUSR, b'w');
    buf[2] = special(S_ISUID, S_IXUSR, b's', b'S');
    buf[3] = bit(S_IRGRP, b'r');
    buf[4] = bit(S_IWGRP, b'w');
    buf[5] = special(S_ISGID, S_IXGRP, b's', b'S');
    buf[6] = bit(S_IROTH, b'r');
    buf[7] = bit(S_IWOTH, b'w');
    buf[8] = special(S_ISVTX, S_IXOTH, b't', b'T');
}

fn filemode(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let mode = argument_mode(args, "filemode")?;
    let mut buf = [0u8; 10];
    buf[0] = filetype(mode);
    let (_, perm) = buf.split_at_mut(1);
    fileperm(mode, perm.try_into().expect("nine permission columns"));
    Ok(w_str_new_managed(
        std::str::from_utf8(&buf).expect("filemode writes ASCII only"),
    ))
}

pyre_interpreter::py_module! {
    "_stat",
    int_constants: {
        "S_IFDIR" => S_IFDIR,
        "S_IFCHR" => S_IFCHR,
        "S_IFBLK" => S_IFBLK,
        "S_IFREG" => S_IFREG,
        "S_IFIFO" => S_IFIFO,
        "S_IFLNK" => S_IFLNK,
        "S_IFSOCK" => S_IFSOCK,
        "S_IFDOOR" => S_IFDOOR,
        "S_IFPORT" => S_IFPORT,
        "S_IFWHT" => S_IFWHT,

        "S_ISUID" => S_ISUID,
        "S_ISGID" => S_ISGID,
        "S_ISVTX" => S_ISVTX,
        "S_ENFMT" => S_ENFMT,

        "S_IREAD" => S_IREAD,
        "S_IWRITE" => S_IWRITE,
        "S_IEXEC" => S_IEXEC,

        "S_IRWXU" => S_IRWXU,
        "S_IRUSR" => S_IRUSR,
        "S_IWUSR" => S_IWUSR,
        "S_IXUSR" => S_IXUSR,

        "S_IRWXG" => S_IRWXG,
        "S_IRGRP" => S_IRGRP,
        "S_IWGRP" => S_IWGRP,
        "S_IXGRP" => S_IXGRP,

        "S_IRWXO" => S_IRWXO,
        "S_IROTH" => S_IROTH,
        "S_IWOTH" => S_IWOTH,
        "S_IXOTH" => S_IXOTH,

        "UF_SETTABLE" => UF_SETTABLE,
        "UF_NODUMP" => UF_NODUMP,
        "UF_IMMUTABLE" => UF_IMMUTABLE,
        "UF_APPEND" => UF_APPEND,
        "UF_OPAQUE" => UF_OPAQUE,
        "UF_NOUNLINK" => UF_NOUNLINK,
        "UF_COMPRESSED" => UF_COMPRESSED,
        "UF_TRACKED" => UF_TRACKED,
        "UF_DATAVAULT" => UF_DATAVAULT,
        "UF_HIDDEN" => UF_HIDDEN,
        "SF_SETTABLE" => SF_SETTABLE,
        "SF_ARCHIVED" => SF_ARCHIVED,
        "SF_IMMUTABLE" => SF_IMMUTABLE,
        "SF_APPEND" => SF_APPEND,
        "SF_NOUNLINK" => SF_NOUNLINK,
        "SF_SNAPSHOT" => SF_SNAPSHOT,
        "SF_FIRMLINK" => SF_FIRMLINK,
        "SF_DATALESS" => SF_DATALESS,
    },
    functions: {
        "S_ISDIR"  / 1 = |args| is_format(args, S_IFDIR, "S_ISDIR"),
        "S_ISCHR"  / 1 = |args| is_format(args, S_IFCHR, "S_ISCHR"),
        "S_ISBLK"  / 1 = |args| is_format(args, S_IFBLK, "S_ISBLK"),
        "S_ISREG"  / 1 = |args| is_format(args, S_IFREG, "S_ISREG"),
        "S_ISFIFO" / 1 = |args| is_format(args, S_IFIFO, "S_ISFIFO"),
        "S_ISLNK"  / 1 = |args| is_format(args, S_IFLNK, "S_ISLNK"),
        "S_ISSOCK" / 1 = |args| is_format(args, S_IFSOCK, "S_ISSOCK"),
        "S_ISDOOR" / 1 = |args| is_format(args, S_IFDOOR, "S_ISDOOR"),
        "S_ISPORT" / 1 = |args| is_format(args, S_IFPORT, "S_ISPORT"),
        "S_ISWHT"  / 1 = |args| is_format(args, S_IFWHT, "S_ISWHT"),
        "S_IMODE"  / 1 = s_imode,
        "S_IFMT"   / 1 = s_ifmt,
        "filemode" / 1 = filemode,
    },
    extra_init: |ns| {
        for (position, name) in ST_CONSTANTS.iter().enumerate() {
            crate::__pyre_store!(ns, name, w_int_new(position as i64));
        }
        #[cfg(target_vendor = "apple")]
        {
            crate::__pyre_store!(ns, "SF_SUPPORTED", w_int_new(i64::from(SF_SUPPORTED)));
            crate::__pyre_store!(ns, "SF_SYNTHETIC", w_int_new(i64::from(SF_SYNTHETIC)));
        }
        #[cfg(all(windows, feature = "host_env"))]
        for (name, value) in FILE_ATTRIBUTES {
            crate::__pyre_store!(ns, name, w_int_new(value));
        }
        #[cfg(windows)]
        for (name, value) in IO_REPARSE_TAGS {
            crate::__pyre_store!(ns, name, w_int_new(value));
        }
    }
}
