//! `pypy/module/faulthandler/dumper.py`.
//!
//! Everything here runs from signal handlers, the vectored exception handler
//! and the watchdog thread, so it allocates nothing: strings go out through
//! a stack buffer and every read is a plain field load off the frame chain.

use majit_rlib::rffi::INT;

use super::cintf::{pypy_faulthandler_write, pypy_faulthandler_write_uint};

/// `dumper.py MAX_STRING_LENGTH`.
const MAX_STRING_LENGTH: usize = 500;

/// `faulthandler.c MAX_FRAME_DEPTH`.
const MAX_FRAME_DEPTH: usize = 100;

/// `dumper.py _dump`.
///
/// [3.15-spec] `_dump` copies at most `MAX_STRING_LENGTH - 1` raw bytes;
/// `traceback.c _Py_DumpASCII` counts code points, writes each non-printable
/// one as `\xNN` / `\uNNNN` / `\UNNNNNNNN` and marks a cut with `...`.
/// Evidence: `lib-python/3/test/test_faulthandler.py` `test_truncate` expects
/// `'x' * 500 + '...'` for a 550-character function name. The escapes are
/// `_Py_DumpASCII` in `Python/traceback.c` read at v3.15.0, and the 3.15.0
/// build measured on Windows dumps a function named `f` + U+00E9 + U+4E2D
/// as `in f\xe9\u4e2d`.
fn _dump(fd: INT, s: &[u8]) {
    let mut buf = [0u8; 128];
    let mut used = 0usize;
    let flush = |buf: &[u8; 128], used: &mut usize| {
        pypy_faulthandler_write(fd, &buf[..*used]);
        *used = 0;
    };
    let mut count = 0usize;
    let mut i = 0usize;
    while i < s.len() {
        if count == MAX_STRING_LENGTH {
            break;
        }
        // WTF-8: a lone surrogate is a three-byte sequence like any other
        // BMP code point, so a plain UTF-8 decode covers it.
        let lead = s[i];
        let (ch, width) = if lead < 0x80 {
            (u32::from(lead), 1)
        } else if lead >= 0xf0 && i + 3 < s.len() {
            let ch = (u32::from(lead & 0x07) << 18)
                | (u32::from(s[i + 1] & 0x3f) << 12)
                | (u32::from(s[i + 2] & 0x3f) << 6)
                | u32::from(s[i + 3] & 0x3f);
            (ch, 4)
        } else if lead >= 0xe0 && i + 2 < s.len() {
            let ch = (u32::from(lead & 0x0f) << 12)
                | (u32::from(s[i + 1] & 0x3f) << 6)
                | u32::from(s[i + 2] & 0x3f);
            (ch, 3)
        } else if lead >= 0xc0 && i + 1 < s.len() {
            let ch = (u32::from(lead & 0x1f) << 6) | u32::from(s[i + 1] & 0x3f);
            (ch, 2)
        } else {
            (u32::from(lead), 1)
        };
        i += width;
        count += 1;
        if used + 10 > buf.len() {
            flush(&buf, &mut used);
        }
        if (0x20..=0x7e).contains(&ch) {
            buf[used] = ch as u8;
            used += 1;
            continue;
        }
        let (prefix, digits): (&[u8], u32) = if ch <= 0xff {
            (b"\\x", 2)
        } else if ch <= 0xffff {
            (b"\\u", 4)
        } else {
            (b"\\U", 8)
        };
        buf[used..used + 2].copy_from_slice(prefix);
        used += 2;
        for shift in (0..digits).rev() {
            buf[used] = b"0123456789abcdef"[((ch >> (shift * 4)) & 0xf) as usize];
            used += 1;
        }
    }
    flush(&buf, &mut used);
    if i < s.len() {
        pypy_faulthandler_write(fd, b"...");
    }
}

/// `dumper.py _dump_nonneg_int`.
fn _dump_nonneg_int(fd: INT, i: usize) {
    pypy_faulthandler_write_uint(fd, i as libc::c_ulong, 1);
}

/// `dumper.py dump_code`, given the frame the code runs in.
///
/// [3.15-spec] `dump_code` prints `co_firstlineno` and leaves the frame out
/// (the header says "approximate line numbers"); `traceback.c dump_frame`
/// prints the line of the frame's last instruction, or `???` when it has
/// none. Evidence: `lib-python/3/test/test_faulthandler.py`
/// `expected_traceback` and `check_dump_traceback` match the executing line
/// of each frame; the `???` is `dump_frame`'s `lineno >= 0` test.
///
/// # Safety
/// `frame` is a live, non-virtual frame.
unsafe fn dump_code(frame: *const pyre_interpreter::pyframe::PyFrame, fd: INT) {
    let pycode = unsafe { (*frame).pycode as *const pyre_interpreter::pycode::PyCode };
    if pycode.is_null() || unsafe { (*pycode).code_ptr.is_null() } {
        pypy_faulthandler_write(fd, b"  File ???\n");
        return;
    }
    let code = unsafe { &*((*pycode).code_ptr as *const pyre_interpreter::CodeObject) };
    let filename = if unsafe { (*pycode).filename_bytes.is_null() } {
        code.source_path.as_bytes()
    } else {
        unsafe { &*(*pycode).filename_bytes }.as_slice()
    };
    // `_dump` writes bytes as they are; `_Py_DumpASCII` escapes, so only the
    // two strings the frame names go through it.
    pypy_faulthandler_write(fd, b"  File \"");
    _dump(fd, filename);
    pypy_faulthandler_write(fd, b"\", line ");
    let lineno = unsafe { (*frame).get_lineno_at((*frame).last_instr) };
    if lineno >= 0 {
        _dump_nonneg_int(fd, lineno as usize);
    } else {
        pypy_faulthandler_write(fd, b"???");
    }
    pypy_faulthandler_write(fd, b" in ");
    let cached_name = unsafe { (*pycode).w_name };
    let name = if cached_name.is_null() {
        code.obj_name.as_bytes()
    } else {
        unsafe { pyre_object::w_str_get_wtf8(cached_name) }.as_bytes()
    };
    _dump(fd, name);
    pypy_faulthandler_write(fd, b"\n");
}

/// `dumper.py _dump_callback`, the `DUMP_CALLBACK` `pypy_faulthandler_setup`
/// installs.
///
/// PRE-EXISTING-ADAPTATION: upstream receives the `(array_p, array_length)`
/// pairs `vmprof_get_traceback` collected off the thread's `vmprof_tl_stack`
/// and resolves each code id with `traceback.walk_traceback`. pyre has no
/// rvmprof stack (`majit_rlib::rvmprof` is the disabled adapter), so the
/// callback is handed the thread's `ExecutionContext` and walks
/// `topframeref` / `f_backref` instead — the chain `sys._current_frames` and
/// `sys._getframe` read. Converging means porting `rvmprof`'s
/// `vmprof_execute_code` stack and its `enter_portal_frame` JIT ops.
///
/// A frame slot the tracer holds as a virtual ref cannot be forced here, so
/// the walk stops at it.
///
/// [3.15-spec] A `hidden_applevel` frame is left out and not counted.
/// Upstream's vmprof stack carries every `execute_frame`, the gateway's
/// app-level helpers included; CPython has no such frames, because those
/// helpers are C there. Evidence: `lib-python/3/test/test_faulthandler.py`
/// `check_dump_traceback_threads` expects `run` directly under
/// `_bootstrap_inner`, where pyre's `Context.run` is the app-level
/// `_contextvars_app.py`. `pytraceback.py record_application_traceback`
/// skips the same frames.
///
/// # Safety
/// `ec` is null or a registered `ExecutionContext`.
pub unsafe extern "C" fn _dump_callback(fd: INT, ec: *const pyre_interpreter::PyExecutionContext) {
    let mut frame = if ec.is_null() {
        std::ptr::null_mut()
    } else {
        unsafe { (*ec).topframeref }
    };
    // `traceback.c dump_traceback`.
    if frame.is_null() {
        pypy_faulthandler_write(fd, b"  <no Python frame>\n");
        return;
    }
    let mut depth = 0usize;
    while !frame.is_null() {
        if depth >= MAX_FRAME_DEPTH {
            break;
        }
        if unsafe { majit_metainterp::virtualref::ptr_is_virtual_ref(frame as *const u8) } {
            pypy_faulthandler_write(fd, b"  <invalid frame>\n");
            break;
        }
        let pycode = unsafe { (*frame).pycode as *const pyre_interpreter::pycode::PyCode };
        if pycode.is_null() || !unsafe { (*pycode).hidden_applevel } {
            unsafe { dump_code(frame, fd) };
            depth += 1;
        }
        frame = unsafe { (*frame).f_backref };
    }
}
