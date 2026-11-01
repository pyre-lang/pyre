//! `rpython/rtyper/lltypesystem/rffi.py` — C types, `llexternal`, the errno
//! flags an external call saves, and the runtime string/buffer helpers.
//!
//! `ExternalCompilationInfo` is `rpython/translator/tool/cbuild.py`.

#![allow(non_camel_case_types)]

pub use majit_jitcode::rffi::{
    RFFI_ALT_ERRNO, RFFI_ERR_ALL, RFFI_ERR_NONE, RFFI_FULL_ERRNO, RFFI_FULL_ERRNO_ZERO,
    RFFI_FULL_LASTERROR, RFFI_READSAVED_ERRNO, RFFI_READSAVED_LASTERROR, RFFI_SAVE_ERRNO,
    RFFI_SAVE_LASTERROR, RFFI_SAVE_WSALASTERROR, RFFI_ZERO_ERRNO_BEFORE,
};
pub use majit_macros::{
    call_aroundstate_target, external_compilation_info, jit_close_stack, llexternal,
};

/// `rffi.CHAR` (`lltype.Char`, 0..=255; `size_and_sign` reports it unsigned).
pub type CHAR = u8;
/// `rffi.UCHAR`.
pub type UCHAR = std::ffi::c_uchar;
/// `rffi.SHORT`.
pub type SHORT = std::ffi::c_short;
/// `rffi.USHORT`.
pub type USHORT = std::ffi::c_ushort;
/// `rffi.INT`.
pub type INT = std::ffi::c_int;
/// `rffi.UINT`.
pub type UINT = std::ffi::c_uint;
/// `rffi.LONG`.
pub type LONG = std::ffi::c_long;
/// `rffi.ULONG`.
pub type ULONG = std::ffi::c_ulong;
/// `rffi.LONGLONG`.
pub type LONGLONG = std::ffi::c_longlong;
/// `rffi.ULONGLONG`.
pub type ULONGLONG = std::ffi::c_ulonglong;
/// `rffi.SIZE_T`.
pub type SIZE_T = usize;
/// `rffi.SSIZE_T`.
pub type SSIZE_T = isize;
/// `rffi.SIGNED` (`lltype.Signed`).
pub type SIGNED = isize;
/// `rffi.UNSIGNED` (`lltype.Unsigned`).
pub type UNSIGNED = usize;
/// `rffi.DOUBLE`.
pub type DOUBLE = std::ffi::c_double;
/// `rffi.FLOAT`.
pub type FLOAT = std::ffi::c_float;
/// `rffi.VOIDP`.
pub type VOIDP = *mut std::ffi::c_void;
/// `rffi.CONST_VOIDP`.
pub type CONST_VOIDP = *const std::ffi::c_void;
/// `rffi.CCHARP`.
pub type CCHARP = *mut CHAR;
/// `rffi.CONST_CCHARP`.
pub type CONST_CCHARP = *const CHAR;
/// `rffi.CCHARPP`.
pub type CCHARPP = *mut CCHARP;
/// `wchar_t` behind [`CWCHARP`]. wasm32-unknown-unknown has no `libc`.
#[cfg(not(target_arch = "wasm32"))]
pub(super) type Wchar = libc::wchar_t;
/// `wchar_t` behind [`CWCHARP`]. wasm32-unknown-unknown has no `libc`.
#[cfg(target_arch = "wasm32")]
pub(super) type Wchar = i32;
/// `rffi.CWCHARP` (`wchar_t *`).
pub type CWCHARP = *mut Wchar;
/// `rffi.CWCHARPP`.
pub type CWCHARPP = *mut CWCHARP;

mod buffer;
mod callback;
mod convert;

pub use crate::getintfield;
pub use crate::offsetof;
pub use crate::setintfield;
pub use buffer::*;
pub use callback::*;
pub use convert::*;

use std::sync::OnceLock;

/// Panic payload for a rustc-path sandbox stub (`rsandbox.py`
/// `not_implemented_stub`).
#[derive(Debug)]
pub struct SandboxStub {
    pub fnname: &'static str,
}

/// Panic with [`SandboxStub`] carrying `fnname` so the rustc-path catcher
/// can call `rsandbox.py not_implemented_stub`.
pub fn sandbox_stub_panic(fnname: &'static str) -> ! {
    std::panic::panic_any(SandboxStub { fnname })
}

type SandboxStubPublisher = fn(&'static str);

static SANDBOX_STUB_PUBLISHER: OnceLock<SandboxStubPublisher> = OnceLock::new();

/// Register the process-global publisher the JIT residual `ccall_` path uses.
pub fn register_sandbox_stub_publisher(publisher: SandboxStubPublisher) {
    let _ = SANDBOX_STUB_PUBLISHER.set(publisher);
}

/// Invoke the residual publisher with `fnname` so it can call
/// `rsandbox.py not_implemented_stub`.
pub fn sandbox_stub_publish(fnname: &'static str) {
    if let Some(publisher) = SANDBOX_STUB_PUBLISHER.get() {
        publisher(fnname);
    }
}

// `rffi.c_memcpy` / `rffi.c_memset`: `releasegil=False`, `calling_conv='c'`.
// Upstream declares `lltype.Void` and an `lltype.Signed` fill byte and the C
// compiler converts at the call through `<string.h>`'s prototype. A Rust
// `extern` block is that prototype, so it spells the C signature: `void *`
// result and an `int` fill byte.
llexternal!(
    pub c_memcpy = "memcpy",
    [VOIDP, CONST_VOIDP, SIZE_T],
    VOIDP,
    releasegil = false,
    calling_conv = "c",
);
llexternal!(
    pub c_memset = "memset",
    [VOIDP, INT, SIZE_T],
    VOIDP,
    releasegil = false,
    calling_conv = "c",
);

/// `ExternalCompilationInfo` (`cbuild.py`).
///
/// Slices stay `'static` so a declaration is a const.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExternalCompilationInfo {
    pub pre_include_bits: &'static [&'static str],
    pub includes: &'static [&'static str],
    pub include_dirs: &'static [&'static str],
    pub post_include_bits: &'static [&'static str],
    pub libraries: &'static [&'static str],
    pub library_dirs: &'static [&'static str],
    pub separate_module_sources: &'static [&'static str],
    pub separate_module_files: &'static [&'static str],
    pub compile_extra: &'static [&'static str],
    pub link_extra: &'static [&'static str],
    pub frameworks: &'static [&'static str],
    pub link_files: &'static [&'static str],
    pub testonly_libraries: &'static [&'static str],
    pub use_cpp_linker: bool,
}

impl ExternalCompilationInfo {
    pub const DEFAULT: Self = Self {
        pre_include_bits: &[],
        includes: &[],
        include_dirs: &[],
        post_include_bits: &[],
        libraries: &[],
        library_dirs: &[],
        separate_module_sources: &[],
        separate_module_files: &[],
        compile_extra: &[],
        link_extra: &[],
        frameworks: &[],
        link_files: &[],
        testonly_libraries: &[],
        use_cpp_linker: false,
    };

    pub const fn new() -> Self {
        Self::DEFAULT
    }
}

impl Default for ExternalCompilationInfo {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[cfg(all(test, feature = "host_env", not(feature = "sandbox")))]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    static GIL_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    static HELD_DURING: AtomicBool = AtomicBool::new(false);

    struct GilHold;
    impl Drop for GilHold {
        fn drop(&mut self) {
            if majit_gc::rgil::am_i_holding_the_gil() {
                majit_gc::rgil::release();
            }
        }
    }

    fn hold_gil() -> (std::sync::MutexGuard<'static, ()>, GilHold) {
        let guard = GIL_LOCK.lock().unwrap();
        majit_gc::rgil::allocate();
        if !majit_gc::rgil::am_i_holding_the_gil() {
            majit_gc::rgil::acquire();
        }
        (guard, GilHold)
    }

    #[unsafe(no_mangle)]
    unsafe extern "C" fn majit_rffi_t1_gil_probe() -> i32 {
        HELD_DURING.store(majit_gc::rgil::am_i_holding_the_gil(), Ordering::SeqCst);
        7
    }

    #[unsafe(no_mangle)]
    unsafe extern "C" fn majit_rffi_t1_set_errno(value: i32) -> i32 {
        #[cfg(any(target_os = "macos", target_os = "ios", target_os = "freebsd"))]
        unsafe {
            *libc::__error() = value;
        }
        #[cfg(any(target_os = "linux", target_os = "android"))]
        unsafe {
            *libc::__errno_location() = value;
        }
        #[cfg(windows)]
        unsafe {
            *_errno() = value;
        }
        value
    }

    #[cfg(windows)]
    unsafe extern "C" {
        fn _errno() -> *mut i32;
    }

    external_compilation_info! {
        const EMPTY_ECI = {};
    }

    #[cfg(unix)]
    external_compilation_info! {
        const LIBC_ECI = {
            libraries: ["c"],
            includes: ["errno.h"],
        };
    }

    llexternal!(
        gil_probe = "majit_rffi_t1_gil_probe",
        [],
        i32,
        compilation_info = EMPTY_ECI
    );
    llexternal!(
        gil_probe_direct = "majit_rffi_t1_gil_probe",
        [],
        i32,
        _nowrapper = true
    );
    llexternal!(
        set_errno_ext = "majit_rffi_t1_set_errno",
        [i32],
        i32,
        save_err = RFFI_SAVE_ERRNO
    );
    llexternal!(
        pub pub_gil_probe = "majit_rffi_t1_gil_probe",
        [],
        i32,
        sandboxsafe = true
    );
    #[cfg(unix)]
    llexternal!(
        fcntl_int = "fcntl",
        [INT, INT, INT],
        INT,
        natural_arity = 2,
        save_err = RFFI_SAVE_ERRNO
    );

    #[test]
    fn gil_is_released_during_the_call_and_held_after() {
        let (_lock, _hold) = hold_gil();
        assert!(majit_gc::rgil::am_i_holding_the_gil());
        let value = unsafe { gil_probe() };
        assert_eq!(value, 7);
        assert!(!HELD_DURING.load(Ordering::SeqCst));
        assert!(majit_gc::rgil::am_i_holding_the_gil());
    }

    #[test]
    fn nowrapper_calls_the_funcptr_without_releasing_the_gil() {
        let (_lock, _hold) = hold_gil();
        let value = unsafe { gil_probe_direct() };
        assert_eq!(value, 7);
        assert!(HELD_DURING.load(Ordering::SeqCst));
        assert!(majit_gc::rgil::am_i_holding_the_gil());
    }

    #[test]
    fn save_errno_is_visible_to_get_saved_errno() {
        let (_lock, _hold) = hold_gil();
        crate::rposix::set_saved_errno(0);
        let value = unsafe { set_errno_ext(123) };
        assert_eq!(value, 123);
        assert_eq!(crate::rposix::get_saved_errno(), 123);
        assert!(majit_gc::rgil::am_i_holding_the_gil());
    }

    #[cfg(unix)]
    #[test]
    fn variadic_fcntl_matches_libc_and_saves_ebadf() {
        let (_lock, _hold) = hold_gil();
        let mut fds = [0; 2];
        assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0);
        let expected = unsafe { libc::fcntl(fds[0], libc::F_GETFD) };
        assert!(expected >= 0);
        assert_eq!(unsafe { fcntl_int(fds[0], libc::F_GETFD, 0) }, expected);
        let with_cloexec = expected | libc::FD_CLOEXEC;
        assert_eq!(unsafe { fcntl_int(fds[0], libc::F_SETFD, with_cloexec) }, 0);
        assert_eq!(unsafe { libc::fcntl(fds[0], libc::F_GETFD) }, with_cloexec);
        crate::rposix::set_saved_errno(0);
        // -1 is never a live fd; concurrent tests can reuse a just-closed number.
        assert!(unsafe { fcntl_int(-1, libc::F_GETFD, 0) } < 0);
        assert_eq!(crate::rposix::get_saved_errno(), libc::EBADF);
        assert_eq!(unsafe { libc::close(fds[0]) }, 0);
        unsafe { libc::close(fds[1]) };
    }

    #[test]
    fn empty_compilation_info_starts_empty() {
        assert!(EMPTY_ECI.libraries.is_empty());
        assert!(!EMPTY_ECI.use_cpp_linker);
        assert_eq!(
            ExternalCompilationInfo::new(),
            ExternalCompilationInfo::DEFAULT
        );
        #[cfg(unix)]
        {
            assert_eq!(LIBC_ECI.libraries, ["c"]);
            assert_eq!(LIBC_ECI.includes, ["errno.h"]);
        }
    }

    #[test]
    fn str2charp_round_trip_and_embedded_nul() {
        let owned = str2charp(b"hello", true).unwrap();
        assert_eq!(unsafe { charp2str(owned) }.unwrap(), b"hello");
        unsafe { free_charp(owned, true) };

        let untracked = str2charp(b"x", false).unwrap();
        assert_eq!(unsafe { charp2str(untracked) }.unwrap(), b"x");
        unsafe { free_charp(untracked, false) };

        // The copy keeps the interior NUL; `charp2str` stops, `charpsize2str` does not.
        let raw = str2charp(b"ab\0cd", true).unwrap();
        assert_eq!(unsafe { charp2str(raw) }.unwrap(), b"ab");
        assert_eq!(unsafe { charpsize2str(raw, 5) }.unwrap(), b"ab\0cd");
        assert_eq!(unsafe { charp2strn(raw, 4) }.unwrap(), b"ab");
        assert_eq!(unsafe { charp2strn(raw, 1) }.unwrap(), b"a");
        let whole = str2charp(b"abcdef", true).unwrap();
        assert_eq!(unsafe { charp2strn(whole, 3) }.unwrap(), b"abc");
        assert_eq!(unsafe { charp2strn(whole, 100) }.unwrap(), b"abcdef");
        unsafe { free_charp(raw, true) };
        unsafe { free_charp(whole, true) };

        let cp = str2constcharp(b"hi", true).unwrap();
        assert_eq!(unsafe { constcharp2str(cp) }.unwrap(), b"hi");
        assert_eq!(unsafe { constcharpsize2str(cp, 2) }.unwrap(), b"hi");
        unsafe { free_charp(cast::<CCHARP>(cp), true) };
    }

    #[test]
    fn str2chararray_and_str2rawmem_copy_a_slice() {
        let (buf, gc, case) = alloc_buffer(8).unwrap();
        assert_eq!(case, 0);
        unsafe {
            c_memset(cast::<VOIDP>(buf), 0xFF, 8);
            assert_eq!(str2chararray(b"hello", buf, 3), 3);
            assert_eq!(charpsize2str(buf, 4).unwrap(), b"hel\xFF");
            str2rawmem(b"abcdef", buf, 2, 4);
            assert_eq!(charpsize2str(buf, 4).unwrap(), b"cdef");
            keep_buffer_alive_until_here(buf, gc, case);
            crate::lltypesystem::rstr::free_immortal_str(gc, 8);
        }
    }

    #[test]
    fn liststr2charpp_round_trip() {
        let pp = liststr2charpp(&[b"one", b"two", b""]).unwrap();
        let got = unsafe { charpp2liststr(pp) }.unwrap();
        assert_eq!(got, vec![b"one".to_vec(), b"two".to_vec(), b"".to_vec()]);
        unsafe { free_charpp(pp) };

        let empty = liststr2charpp(&[]).unwrap();
        assert!(unsafe { charpp2liststr(empty) }.unwrap().is_empty());
        unsafe { free_charpp(empty) };
    }

    #[test]
    fn alloc_buffer_str_from_buffer_and_scoped_drop() {
        let (raw, gc, case) = alloc_buffer(8).unwrap();
        unsafe {
            c_memset(cast::<VOIDP>(raw), 0x41, 8);
            let s = str_from_buffer(raw, gc, case, 8, 3).unwrap();
            assert_eq!(
                crate::lltypesystem::rstr::str_as_bytes(
                    s.0 as *const crate::lltypesystem::rstr::STR
                ),
                b"AAA"
            );
            keep_buffer_alive_until_here(raw, gc, case);
            crate::lltypesystem::rstr::free_immortal_str(s, 3);
            crate::lltypesystem::rstr::free_immortal_str(gc, 8);
        }

        let staged = scoped_alloc_buffer::new(8).unwrap();
        assert_eq!(staged.size, 8);
        assert_eq!(staged.case_num, 0);
        unsafe { c_memset(cast::<VOIDP>(staged.raw), 0x42, 8) };
        let s = staged.str(2).unwrap();
        unsafe {
            assert_eq!(
                crate::lltypesystem::rstr::str_as_bytes(
                    s.0 as *const crate::lltypesystem::rstr::STR
                ),
                b"BB"
            );
            crate::lltypesystem::rstr::free_immortal_str(s, 2);
        }

        // No collector: `rgc.can_move` is false, so the pointer is inside
        // the immortal STR (`\x04`) and `rgc.pin` is never asked.
        let view_str = crate::lltypesystem::rstr::alloc_immortal_str(b"ab\0c");
        assert!(!majit_gc::can_move(view_str));
        assert!(!majit_gc::pin(view_str));
        let view = scoped_view_charp::new(view_str).unwrap();
        assert_eq!(view.flag, 0x04);
        assert_eq!(unsafe { charp2str(view.buf) }.unwrap(), b"ab");
        assert_eq!(unsafe { charpsize2str(view.buf, 4) }.unwrap(), b"ab\0c");
        drop(view);
        unsafe { crate::lltypesystem::rstr::free_immortal_str(view_str, 4) };

        let moving_str = crate::lltypesystem::rstr::alloc_immortal_str(b"xyz");
        let moving = scoped_nonmovingbuffer::new(moving_str).unwrap();
        assert_eq!(moving.flag, 0x04);
        assert_eq!(unsafe { charpsize2str(moving.buf, 3) }.unwrap(), b"xyz");
        drop(moving);
        unsafe { crate::lltypesystem::rstr::free_immortal_str(moving_str, 3) };

        let none = scoped_str2charp::new(None).unwrap();
        assert!(none.buf.is_null());
        drop(none);
        let some = scoped_str2charp::new(Some(b"zz")).unwrap();
        assert_eq!(unsafe { charp2str(some.buf) }.unwrap(), b"zz");
        drop(some);
    }

    #[test]
    fn alloc_buffer_large_count_uses_raw_storage() {
        let allocated_size = 8;
        let needed_size = 5;
        let raw = super::buffer::raw_malloc_or_memory_error(allocated_size)
            .unwrap()
            .cast::<CHAR>();
        unsafe {
            c_memset(cast::<VOIDP>(raw), 0x58, allocated_size);
            for (i, byte) in b"hello".iter().enumerate() {
                *raw.add(i) = *byte;
            }
            let s = str_from_buffer(raw, majit_ir::GcRef::NULL, 3, allocated_size, needed_size)
                .unwrap();
            assert_eq!(
                crate::lltypesystem::rstr::str_as_bytes(
                    s.0 as *const crate::lltypesystem::rstr::STR
                ),
                b"hello"
            );
            keep_buffer_alive_until_here(raw, majit_ir::GcRef::NULL, 3);
            crate::lltypesystem::rstr::free_immortal_str(s, needed_size);
        }
    }

    #[test]
    fn cast_truncates_and_sign_extends() {
        assert_eq!(cast::<u8>(0x1_ABCDi32), 0xCDu8);
        assert_eq!(cast::<i8>(-1i32), -1i8);
        assert_eq!(cast::<i16>(0x1_0005i32), 5i16);
        assert_eq!(cast::<i64>(-1i8), -1i64);
        assert_eq!(cast::<u64>(-1i8), u64::MAX);
        assert_eq!(cast::<i32>(65535u16), 65535i32);

        let p = str2charp(b"z", true).unwrap();
        let addr = cast::<usize>(p);
        assert_eq!(cast::<CCHARP>(addr), p);
        unsafe { free_charp(p, true) };

        assert_eq!(size_and_sign::<INT>(), (4, false));
        assert_eq!(size_and_sign::<UINT>(), (4, true));
        assert_eq!(sizeof::<FLOAT>(), 4);
        assert_eq!(size_and_sign::<DOUBLE>(), (8, false));
        assert_eq!(size_and_sign::<CCHARP>().0, core::mem::size_of::<CCHARP>());
        assert!(!size_and_sign::<CCHARP>().1);

        #[repr(C)]
        struct OffSample {
            a: u8,
            b: u32,
        }
        assert_eq!(offsetof!(OffSample, b), core::mem::offset_of!(OffSample, b));

        struct Rec {
            v: i16,
        }
        let mut rec = Rec { v: 0 };
        setintfield!(&mut rec, v, -1i64);
        assert_eq!(rec.v, -1);
        assert_eq!(getintfield!(&rec, v), -1isize);
        setintfield!(&mut rec, v, 0x1_0005i32);
        assert_eq!(rec.v, 5);
        assert_eq!(getintfield!(&rec, v), 5isize);

        let (base, gc, case) = alloc_buffer(4).unwrap();
        unsafe {
            c_memset(cast::<VOIDP>(base), 0, 4);
            *base = 1;
            *ptradd(base, 2) = 3;
            assert_eq!(ptradd(ptradd(base, 2), -2), base);
            assert_eq!(charpsize2str(base, 4).unwrap(), vec![1, 0, 3, 0]);
            keep_buffer_alive_until_here(base, gc, case);
            crate::lltypesystem::rstr::free_immortal_str(gc, 4);
        }
    }

    #[test]
    fn utf8_wchar_round_trip_includes_non_bmp() {
        let text = "hé你😀";
        let bytes = text.as_bytes();
        let n = text.chars().count();
        let w = utf82wcharp(bytes, n, true).unwrap();
        let wide = unsafe { core::mem::size_of_val(&*w) } >= 4;
        let (back, count) = unsafe { wcharp2utf8(w).unwrap() };
        let (cut, cut_n) = unsafe { wcharp2utf8n(w, 3).unwrap() };
        let sized = unsafe { wcharpsize2utf8(w, n).unwrap() };
        unsafe { free_wcharp(w, false) };
        assert_eq!(count, n);
        assert_eq!(cut_n, 3);
        if wide {
            assert_eq!(back, bytes);
            assert_eq!(sized, bytes);
            assert_eq!(cut, "hé你".as_bytes());
        }

        let with_nul = b"a\0b";
        let w = utf82wcharp(with_nul, 3, true).unwrap();
        let (stopped, nzero) = unsafe { wcharp2utf8(w).unwrap() };
        let full = unsafe { wcharpsize2utf8(w, 3).unwrap() };
        unsafe { free_wcharp(w, true) };
        assert_eq!((stopped.as_slice(), nzero), (&b"a"[..], 1));
        assert_eq!(full, b"a\0b");

        if wide {
            let w = utf82wcharp(b"", 0, true).unwrap();
            unsafe { core::ptr::write(w.cast::<i32>(), 0x110000) };
            assert_eq!(unsafe { wcharpsize2utf8(w, 1) }.unwrap_err().code, 0x110000);
            unsafe { free_wcharp(w, true) };
        }

        let scoped = scoped_utf82wcharp::new(Some(bytes), -1).unwrap();
        let (again, again_n) = unsafe { wcharp2utf8(scoped.buf).unwrap() };
        assert_eq!(again_n, n);
        if wide {
            assert_eq!(again, bytes);
        }
        let absent = scoped_utf82wcharp::new(None, -1).unwrap();
        assert!(absent.buf.is_null());
    }

    #[test]
    fn c_memcpy_and_c_memset_fill_a_raw_buffer() {
        let (buf, gc, case) = alloc_buffer(8).unwrap();
        unsafe {
            c_memset(cast::<VOIDP>(buf), 0x11, 8);
            assert_eq!(charpsize2str(buf, 8).unwrap(), vec![0x11; 8]);
            let src = str2charp(b"abcd", true).unwrap();
            c_memcpy(cast::<VOIDP>(buf), cast::<CONST_VOIDP>(src), 4);
            assert_eq!(charpsize2str(buf, 8).unwrap(), b"abcd\x11\x11\x11\x11");
            free_charp(src, true);
            keep_buffer_alive_until_here(buf, gc, case);
            crate::lltypesystem::rstr::free_immortal_str(gc, 8);
        }
    }

    #[test]
    fn utf82wcharp_ex_writes_a_surrogate_pair_above_the_bmp() {
        let text = "a😀\u{ffff}";
        let w = utf82wcharp_ex(text.as_bytes(), 3, true).unwrap();
        let units: Vec<u32> = (0..5).map(|i| unsafe { *w.add(i) } as u32).collect();
        unsafe { free_wcharp(w, true) };
        assert_eq!(units, [0x61, 0xD83D, 0xDE00, 0xFFFF, 0]);
    }

    #[test]
    fn char_is_unsigned() {
        assert_eq!(size_and_sign::<CHAR>(), (1, true));
        assert_eq!(cast::<CHAR>(-1i32), 0xFF);
    }

    #[test]
    fn callback_through_releasegil_llexternal_runs_holding_the_gil() {
        crate::rffi::set_acquire_maybe_in_new_thread(majit_gc::gc_sync::enter_external_callback);
        crate::rffi::llexternal!(
            c_qsort = "qsort",
            [
                VOIDP,
                SIZE_T,
                SIZE_T,
                unsafe extern "C" fn(VOIDP, VOIDP) -> INT
            ],
            ()
        );

        static SAW_GIL: AtomicBool = AtomicBool::new(false);

        fn compare(a: VOIDP, b: VOIDP) -> INT {
            SAW_GIL.store(majit_gc::rgil::am_i_holding_the_gil(), Ordering::SeqCst);
            let left = unsafe { *(a as *const i32) };
            let right = unsafe { *(b as *const i32) };
            match left.cmp(&right) {
                std::cmp::Ordering::Less => -1,
                std::cmp::Ordering::Equal => 0,
                std::cmp::Ordering::Greater => 1,
            }
        }

        let (_lock, _hold) = hold_gil();
        let mut values = [3i32, 1, 2];
        unsafe {
            c_qsort(
                values.as_mut_ptr().cast(),
                3,
                std::mem::size_of::<i32>(),
                compare,
            );
        }
        assert!(SAW_GIL.load(Ordering::SeqCst));
        assert_eq!(values, [1, 2, 3]);

        unsafe extern "C" fn compare_llptr(a: VOIDP, b: VOIDP) -> INT {
            compare(a, b)
        }
        let mut llptr_values = [9i32, 7, 8];
        unsafe {
            c_qsort(
                llptr_values.as_mut_ptr().cast(),
                3,
                std::mem::size_of::<i32>(),
                compare_llptr as unsafe extern "C" fn(VOIDP, VOIDP) -> INT,
            );
        }
        assert_eq!(llptr_values, [7, 8, 9]);
    }
}

#[cfg(all(test, feature = "host_env", not(feature = "sandbox")))]
#[test]
fn pub_wrapper_is_visible_from_the_parent() {
    let _probe: unsafe fn() -> i32 = tests::pub_gil_probe;
}
