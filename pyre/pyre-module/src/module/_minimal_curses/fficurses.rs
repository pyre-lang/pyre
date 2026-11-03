//! `fficurses.py` — `setupterm`, `rpy_curses_tigetstr`, `rpy_curses_tparm`.
//!
//! `term.h` renames ordinary identifiers, so the bodies are the
//! `separate_module_sources` the build script compiles. `guess_eci` picks
//! the libraries; this ECI stays empty and does not emit a second `#[link]`.

use majit_rlib::rffi::{CCHARP, INT};

majit_rlib::rffi::external_compilation_info! {
    const ECI = {};
}

majit_rlib::rffi::llexternal!(
    pub(super) setupterm = "rpy_curses_setupterm",
    [CCHARP, INT, *mut INT],
    INT,
    compilation_info = ECI
);

majit_rlib::rffi::llexternal!(
    pub(super) rpy_curses_tigetstr = "rpy_curses_tigetstr",
    [CCHARP],
    CCHARP,
    compilation_info = ECI
);

majit_rlib::rffi::llexternal!(
    pub(super) rpy_curses_tparm = "rpy_curses_tparm",
    [CCHARP, INT, INT, INT, INT, INT, INT, INT, INT, INT],
    CCHARP,
    compilation_info = ECI
);

// Uppercase int constants `moduledef.py` copies from `_curses`.
// The build probe writes the C table these three functions read.
unsafe extern "C" {
    pub(super) fn rpy_curses_int_count() -> std::ffi::c_int;
    pub(super) fn rpy_curses_int_name(index: std::ffi::c_int) -> *const std::ffi::c_char;
    pub(super) fn rpy_curses_int_value(index: std::ffi::c_int) -> std::ffi::c_longlong;
}

#[cfg(test)]
mod tests {
    use super::{rpy_curses_int_count, rpy_curses_int_name, rpy_curses_int_value};

    fn value(name: &str) -> Option<i64> {
        let count = unsafe { rpy_curses_int_count() };
        for index in 0..count {
            let got = unsafe { std::ffi::CStr::from_ptr(rpy_curses_int_name(index)) };
            if got.to_bytes() == name.as_bytes() {
                return Some(unsafe { rpy_curses_int_value(index) } as i64);
            }
        }
        None
    }

    #[test]
    fn err_and_ok_match_the_curses_sentinels() {
        assert_eq!(value("ERR"), Some(-1));
        assert_eq!(value("OK"), Some(0));
    }

    #[test]
    fn netbsd_omits_keyname_constants() {
        if cfg!(target_os = "netbsd") {
            assert_eq!(value("KEY_UP"), None);
            assert_eq!(value("A_INVIS"), None);
            assert!(value("KEY_MIN").is_some());
            assert!(value("KEY_MAX").is_some());
        } else {
            assert!(value("KEY_UP").is_some());
            assert!(value("A_INVIS").is_some());
        }
    }
}
