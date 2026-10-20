//! `pypy/module/_codecs/locale.py` — locale codecs via POSIX `mbstowcs` /
//! `mbrtowc`.
//!
//! `str_decode_locale_surrogateescape` is the decoder `error.py strerror`
//! uses. The C helper it wraps is `pypy_char2wchar` in `locale_codec.c`.

#[cfg(all(unix, not(target_arch = "wasm32")))]
use rustpython_wtf8::CodePoint;
use rustpython_wtf8::Wtf8Buf;

/// `locale.py str_decode_locale_surrogateescape`.
///
/// Returns `(utf8, codepoint_count)` the way `rawwcharp2utf8en` does.
pub fn str_decode_locale_surrogateescape(s: &[u8]) -> Result<(Wtf8Buf, usize), ()> {
    #[cfg(all(unix, not(target_arch = "wasm32")))]
    {
        let wide = pypy_char2wchar(s).ok_or(())?;
        Ok(rawwcharp2utf8en(&wide))
    }
    // Windows / wasm: no `mbstowcs` locale codec; decode UTF-8 with
    // surrogateescape (`typedef::charp2uni_wtf8`).
    #[cfg(not(all(unix, not(target_arch = "wasm32"))))]
    {
        let utf8 = crate::typedef::charp2uni_wtf8(s);
        let lgt = utf8.code_points().count();
        Ok((utf8, lgt))
    }
}

/// libc 0.2 does not export `mbstowcs` / `mbrtowc` / `mbstate_t` on Darwin
/// (they are not in its portable unix API). `locale_codec.c pypy_char2wchar`
/// calls the C library symbols; we declare the same.
#[cfg(all(unix, not(target_arch = "wasm32")))]
mod wchar_conv {
    /// Darwin `mbstate_t` is a 128-byte union; glibc's is smaller. A zeroed
    /// 128-byte buffer covers both ABIs.
    #[repr(C)]
    pub(super) struct MbState(pub [u8; 128]);

    unsafe extern "C" {
        pub(super) fn mbstowcs(
            dest: *mut libc::wchar_t,
            src: *const libc::c_char,
            n: libc::size_t,
        ) -> libc::size_t;
        pub(super) fn mbrtowc(
            pwc: *mut libc::wchar_t,
            s: *const libc::c_char,
            n: libc::size_t,
            ps: *mut MbState,
        ) -> libc::size_t;
    }
}

/// `locale.py rawwcharp2utf8en` — unix `wchar_t` is 32-bit, so
/// `_should_split_surrogates` is false and each value is one code point.
#[cfg(all(unix, not(target_arch = "wasm32")))]
fn rawwcharp2utf8en(wcp: &[libc::wchar_t]) -> (Wtf8Buf, usize) {
    let mut buf = Wtf8Buf::new();
    let mut lgt = 0usize;
    for &v in wcp {
        if v == 0 {
            break;
        }
        let v = v as u32;
        let cp = CodePoint::from_u32(v).unwrap_or(CodePoint::from_char('\u{FFFD}'));
        buf.push(cp);
        lgt += 1;
    }
    (buf, lgt)
}

/// `locale_codec.c pypy_char2wchar` — `mbstowcs` when the whole string
/// converts without surrogates, else `mbrtowc` with surrogateescape for
/// undecodable bytes.
#[cfg(all(unix, not(target_arch = "wasm32")))]
fn pypy_char2wchar(arg: &[u8]) -> Option<Vec<libc::wchar_t>> {
    let mut c_arg = Vec::with_capacity(arg.len() + 1);
    c_arg.extend_from_slice(arg);
    c_arg.push(0);
    let c_ptr = c_arg.as_ptr() as *const libc::c_char;
    unsafe {
        if let Some(wide) = mbstowcs_no_surrogates(c_ptr) {
            return Some(wide);
        }
        mbrtowc_surrogateescape(&c_arg)
    }
}

/// The `mbstowcs` success path of `pypy_char2wchar`: count, convert, keep
/// the result only when it contains no surrogate.
#[cfg(all(unix, not(target_arch = "wasm32")))]
unsafe fn mbstowcs_no_surrogates(arg: *const libc::c_char) -> Option<Vec<libc::wchar_t>> {
    unsafe {
        let argsize = wchar_conv::mbstowcs(std::ptr::null_mut(), arg, 0);
        if argsize == usize::MAX {
            return None;
        }
        let mut res = vec![0 as libc::wchar_t; argsize + 1];
        let count = wchar_conv::mbstowcs(res.as_mut_ptr(), arg, argsize + 1);
        if count == usize::MAX {
            return None;
        }
        if res[..count]
            .iter()
            .any(|&ch| (0xD800..=0xDFFF).contains(&(ch as u32)))
        {
            return None;
        }
        res.truncate(count);
        Some(res)
    }
}

/// The `HAVE_MBRTOWC` fallback of `pypy_char2wchar`.
#[cfg(all(unix, not(target_arch = "wasm32")))]
unsafe fn mbrtowc_surrogateescape(arg_with_nul: &[u8]) -> Option<Vec<libc::wchar_t>> {
    unsafe {
        let mut res = Vec::with_capacity(arg_with_nul.len());
        let mut mbs = wchar_conv::MbState([0; 128]);
        let mut in_ptr = arg_with_nul.as_ptr();
        let mut remaining = arg_with_nul.len();
        while remaining > 0 {
            let mut out: libc::wchar_t = 0;
            let converted =
                wchar_conv::mbrtowc(&mut out, in_ptr as *const libc::c_char, remaining, &mut mbs);
            if converted == 0 {
                break;
            }
            if converted == usize::MAX - 1 {
                // Incomplete character with the whole input supplied.
                return None;
            }
            if converted == usize::MAX {
                res.push(0xDC00 + *in_ptr as libc::wchar_t);
                in_ptr = in_ptr.add(1);
                remaining -= 1;
                mbs = wchar_conv::MbState([0; 128]);
                continue;
            }
            if (0xD800..=0xDFFF).contains(&(out as u32)) || (out as u32) > 0x10FFFF {
                remaining -= converted;
                for _ in 0..converted {
                    res.push(0xDC00 + *in_ptr as libc::wchar_t);
                    in_ptr = in_ptr.add(1);
                }
                continue;
            }
            res.push(out);
            in_ptr = in_ptr.add(converted);
            remaining -= converted;
        }
        Some(res)
    }
}

#[cfg(test)]
mod tests {
    use super::str_decode_locale_surrogateescape;

    fn assert_ascii_and_ff(expected_ff: u32) {
        let (ascii, lgt) = str_decode_locale_surrogateescape(b"hello").expect("ascii");
        assert_eq!(ascii.as_str().expect("ascii is utf-8"), "hello");
        assert_eq!(lgt, 5);

        let (decoded, lgt) = str_decode_locale_surrogateescape(&[0xFF]).expect("0xFF");
        assert_eq!(lgt, 1);
        let cp = decoded
            .code_points()
            .next()
            .expect("one code point")
            .to_u32();
        assert_eq!(cp, expected_ff);
    }

    /// `pypy_char2wchar` keeps the `mbstowcs` result when it succeeds without
    /// a surrogate, otherwise surrogateescapes the byte to U+DCFF. Darwin's
    /// C locale `mbstowcs` accepts 0x80-0xFF as Latin-1 (`force_ascii` is
    /// `#if !defined(__APPLE__)` in `locale_codec.c`).
    #[cfg(all(unix, not(target_arch = "wasm32")))]
    fn c_locale_ff_codepoint() -> u32 {
        let src = [0xFFu8, 0];
        unsafe {
            let n = super::wchar_conv::mbstowcs(
                std::ptr::null_mut(),
                src.as_ptr() as *const libc::c_char,
                0,
            );
            if n == usize::MAX {
                return 0xDCFF;
            }
            let mut buf = vec![0 as libc::wchar_t; n + 1];
            let count = super::wchar_conv::mbstowcs(
                buf.as_mut_ptr(),
                src.as_ptr() as *const libc::c_char,
                n + 1,
            );
            let ch = buf[0] as u32;
            if count == usize::MAX || (0xD800..=0xDFFF).contains(&ch) {
                0xDCFF
            } else {
                ch
            }
        }
    }

    #[test]
    fn ascii_and_undecodable_byte_under_c_locale() {
        #[cfg(all(unix, not(target_arch = "wasm32")))]
        {
            with_c_locale(|| assert_ascii_and_ff(c_locale_ff_codepoint()));
        }
        #[cfg(not(all(unix, not(target_arch = "wasm32"))))]
        {
            assert_ascii_and_ff(0xDCFF);
        }
    }

    #[cfg(all(unix, not(target_arch = "wasm32")))]
    fn with_c_locale<R>(f: impl FnOnce() -> R) -> R {
        use std::sync::Mutex;
        static LOCK: Mutex<()> = Mutex::new(());
        let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            let old = libc::setlocale(libc::LC_CTYPE, std::ptr::null());
            let old_owned = if old.is_null() {
                None
            } else {
                Some(std::ffi::CStr::from_ptr(old).to_owned())
            };
            let set = libc::setlocale(libc::LC_CTYPE, c"C".as_ptr());
            assert!(!set.is_null(), "setlocale(LC_CTYPE, C)");
            let result = f();
            if let Some(old) = old_owned {
                libc::setlocale(libc::LC_CTYPE, old.as_ptr());
            }
            result
        }
    }
}
