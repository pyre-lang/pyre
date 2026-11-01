//! `rpython/rlib/rposix_stat.py` — `os.stat` / `os.lstat` and the
//! `STAT_FIELDS` result they return.

#[cfg(not(feature = "sandbox"))]
use crate::rposix::get_saved_errno;

// `rposix_stat.compilation_info`.
crate::rffi::external_compilation_info! {
    const STAT_ECI = {
        includes: ["sys/types.h", "sys/stat.h", "sys/statvfs.h", "unistd.h"],
    };
}

crate::rffi::llexternal!(
    pub c_stat = "stat",
    [*const libc::c_char, *mut libc::stat],
    crate::rffi::INT,
    compilation_info = STAT_ECI,
    save_err = crate::rffi::RFFI_SAVE_ERRNO,
    macro = true
);

crate::rffi::llexternal!(
    pub c_lstat = "lstat",
    [*const libc::c_char, *mut libc::stat],
    crate::rffi::INT,
    compilation_info = STAT_ECI,
    save_err = crate::rffi::RFFI_SAVE_ERRNO,
    macro = true
);

crate::rffi::llexternal!(
    pub c_fstat = "fstat",
    [crate::rffi::INT, *mut libc::stat],
    crate::rffi::INT,
    compilation_info = STAT_ECI,
    save_err = crate::rffi::RFFI_SAVE_ERRNO,
    macro = true
);

/// `rposix_stat.STAT_FIELDS` on unix: the portable 10, then `st_blksize` /
/// `st_blocks` / `st_rdev` (and `st_flags` where `struct stat` has it), then
/// the three `nsec_*` extras `build_stat_result` always appends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StatResult {
    pub st_mode: i64,
    pub st_ino: i64,
    pub st_dev: i64,
    pub st_nlink: i64,
    pub st_uid: i64,
    pub st_gid: i64,
    pub st_size: i64,
    pub st_atime: i64,
    pub st_mtime: i64,
    pub st_ctime: i64,
    pub st_blksize: i64,
    pub st_blocks: i64,
    pub st_rdev: i64,
    #[cfg(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "freebsd",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "dragonfly"
    ))]
    pub st_flags: i64,
    pub nsec_atime: i64,
    pub nsec_mtime: i64,
    pub nsec_ctime: i64,
}

/// `rposix_stat.build_stat_result`.
pub fn build_stat_result(st: &libc::stat) -> StatResult {
    StatResult {
        st_mode: st.st_mode as i64,
        st_ino: st.st_ino as i64,
        st_dev: st.st_dev as i64,
        st_nlink: st.st_nlink as i64,
        st_uid: st.st_uid as i64,
        st_gid: st.st_gid as i64,
        st_size: st.st_size as i64,
        st_atime: st.st_atime as i64,
        st_mtime: st.st_mtime as i64,
        st_ctime: st.st_ctime as i64,
        st_blksize: st.st_blksize as i64,
        st_blocks: st.st_blocks as i64,
        st_rdev: st.st_rdev as i64,
        #[cfg(any(
            target_os = "macos",
            target_os = "ios",
            target_os = "freebsd",
            target_os = "netbsd",
            target_os = "openbsd",
            target_os = "dragonfly"
        ))]
        st_flags: st.st_flags as i64,
        nsec_atime: st.st_atime_nsec as i64,
        nsec_mtime: st.st_mtime_nsec as i64,
        nsec_ctime: st.st_ctime_nsec as i64,
    }
}

#[cfg(not(feature = "sandbox"))]
fn finish_stat(
    ret: crate::rffi::INT,
    st: std::mem::MaybeUninit<libc::stat>,
) -> Result<StatResult, i32> {
    if ret < 0 {
        Err(get_saved_errno())
    } else {
        Ok(build_stat_result(&unsafe { st.assume_init() }))
    }
}

/// `rposix_stat.stat` (`replace_os_function('stat')`).
#[majit_macros::replace_os_function("stat")]
pub fn stat(path: &[u8]) -> Result<StatResult, i32> {
    let c_path = std::ffi::CString::new(path).map_err(|_| libc::EINVAL)?;
    let mut st = std::mem::MaybeUninit::<libc::stat>::uninit();
    let ret = unsafe { c_stat(c_path.as_ptr(), st.as_mut_ptr()) };
    finish_stat(ret, st)
}

/// `rposix_stat.lstat` (`replace_os_function('lstat')`).
#[majit_macros::replace_os_function("lstat")]
pub fn lstat(path: &[u8]) -> Result<StatResult, i32> {
    let c_path = std::ffi::CString::new(path).map_err(|_| libc::EINVAL)?;
    let mut st = std::mem::MaybeUninit::<libc::stat>::uninit();
    let ret = unsafe { c_lstat(c_path.as_ptr(), st.as_mut_ptr()) };
    finish_stat(ret, st)
}

/// `rposix_stat.fstat` (`replace_os_function('fstat')`).
#[majit_macros::replace_os_function("fstat")]
pub fn fstat(fd: i32) -> Result<StatResult, i32> {
    let mut st = std::mem::MaybeUninit::<libc::stat>::uninit();
    let ret = unsafe { c_fstat(fd, st.as_mut_ptr()) };
    finish_stat(ret, st)
}

#[cfg(all(test, feature = "host_env", not(feature = "sandbox")))]
mod tests {
    use super::{lstat, stat};
    use std::os::unix::ffi::OsStrExt;

    #[test]
    fn stat_and_lstat_on_a_file_and_symlink() {
        let dir =
            std::env::temp_dir().join(format!("pyre-rffi-rposix-stat-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("f");
        std::fs::write(&file, b"hi").unwrap();
        let link = dir.join("l");
        let _ = std::fs::remove_file(&link);
        std::os::unix::fs::symlink(&file, &link).unwrap();

        let st = stat(file.as_os_str().as_bytes()).expect("stat file");
        assert_eq!(st.st_size, 2);
        assert_eq!(st.st_mode as libc::mode_t & libc::S_IFMT, libc::S_IFREG);

        let lst = lstat(link.as_os_str().as_bytes()).expect("lstat link");
        assert_eq!(lst.st_mode as libc::mode_t & libc::S_IFMT, libc::S_IFLNK);
        let followed = stat(link.as_os_str().as_bytes()).expect("stat link");
        assert_eq!(followed.st_size, 2);

        assert_eq!(
            stat(b"/no/such/pyre-rffi-rposix-stat").unwrap_err(),
            libc::ENOENT
        );
        assert_eq!(stat(b"a\0b").unwrap_err(), libc::EINVAL);

        let _ = std::fs::remove_file(&link);
        let _ = std::fs::remove_file(&file);
        let _ = std::fs::remove_dir(&dir);
    }
}
