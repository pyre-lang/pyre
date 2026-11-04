//! Port of `rpython/translator/platform/arm.py` (`ARM(Linux)`).

use std::borrow::Cow;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::LazyLock;

use super::{HostFamily, LOG, Platform, linux, os_flags};

/// Scratchbox2 state for `ARM` (`SB2ARGS`, `available_librarydirs`,
/// `available_includedirs`). `None` on every other platform.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Sb2 {
    pub args: Vec<String>,
    pub available_librarydirs: Vec<OsString>,
    pub available_includedirs: Vec<OsString>,
}

/// `arm.py` module-level `SB2` / `SB2ARGS` / `available_*dirs`. Initialised
/// on first `arm_platform` call (the module import).
static SB2_MODULE: LazyLock<Sb2> =
    LazyLock::new(|| sb2_from(std::env::var_os("SB2"), std::env::var("SB2OPT").ok()));

/// `ARM(cc)` after the `arm.py` module-level `SB2` / `SB2OPT` setup.
pub(super) fn arm_platform(cc: Option<String>) -> Platform {
    let env_cc = {
        let _guard = super::CC_COMMAND_LOCK.lock();
        super::snapshot_cc_command()
    };
    arm_platform_with_sb2(&*SB2_MODULE, cc, env_cc)
}

/// Testable `ARM` constructor: `SB2` / `SB2OPT` / env `CC` are parameters
/// instead of process environment. Does not touch `SB2_MODULE`.
#[cfg(test)]
pub(super) fn arm_platform_from(
    sb2_root: Option<OsString>,
    sb2opt: Option<String>,
    cc: Option<String>,
    env_cc: Option<String>,
) -> Platform {
    // Each call models a fresh `arm.py` import for tests.
    arm_platform_with_sb2(Box::leak(Box::new(sb2_from(sb2_root, sb2opt))), cc, env_cc)
}

/// `arm.py` module-level `SB2` / `SB2OPT` / directory-list setup.
fn sb2_from(sb2_root: Option<OsString>, sb2opt: Option<String>) -> Sb2 {
    /// `arm.py` module-level `log.error` / `assert 0` when `SB2` is unset.
    const SB2_MISSING: &str =
        "SB2: Provide a path to the sb2 rootfs for the target in env variable SB2";
    /// `ARM.available_librarydirs` suffixes concatenated onto `SB2`.
    const AVAILABLE_LIBRARYDIR_SUFFIXES: &[&str] = &[
        "/lib/arm-linux-gnueabi/",
        "/lib/arm-linux-gnueabihf/",
        "/lib/aarch64-linux-gnu/",
        "/usr/lib/arm-linux-gnueabi/",
        "/usr/lib/arm-linux-gnueabihf/",
        "/usr/lib/aarch64-linux-gnu/",
    ];
    /// `ARM.available_includedirs` suffixes concatenated onto `SB2`.
    const AVAILABLE_INCLUDEDIR_SUFFIXES: &[&str] = &[
        "/usr/include/arm-linux-gnueabi/",
        "/usr/include/arm-linux-gnueabihf/",
        "/usr/include/aarch64-linux-gnu/",
    ];

    let Some(sb2_root) = sb2_root else {
        LOG.Error(SB2_MISSING);
        // arm.py module import `assert 0`
        panic!("{SB2_MISSING}");
    };

    let args = match sb2opt {
        None => {
            LOG.info("Pass additional options to sb2 in SB2OPT");
            Vec::new()
        }
        Some(params) => params.split(' ').map(str::to_string).collect(),
    };

    let available_librarydirs = AVAILABLE_LIBRARYDIR_SUFFIXES
        .iter()
        .map(|suffix| {
            let mut dir = sb2_root.clone();
            dir.push(suffix);
            dir
        })
        .collect();
    let available_includedirs = AVAILABLE_INCLUDEDIR_SUFFIXES
        .iter()
        .map(|suffix| {
            let mut dir = sb2_root.clone();
            dir.push(suffix);
            dir
        })
        .collect();

    Sb2 {
        args,
        available_librarydirs,
        available_includedirs,
    }
}

fn arm_platform_with_sb2(
    sb2: &'static Sb2,
    cc: Option<String>,
    env_cc: Option<String>,
) -> Platform {
    let mut plat = linux::linux_platform(super::host_triple(), cfg!(target_arch = "s390x"));
    plat.class_name = "ARM";
    plat.name = "arm".to_string();
    plat.shared_only = os_flags(["-fPIC"]);
    // `BasePosix.__init__`: `self.cc = cc or os.environ.get('CC', self.DEFAULT_CC)`.
    plat.cc_command = Some(match cc {
        Some(cc) if !cc.is_empty() => cc,
        _ => env_cc.unwrap_or_else(|| super::default_cc(HostFamily::Linux).to_string()),
    });
    plat.sb2 = Some(sb2);
    plat
}

/// `ARM._execute_c_compiler` command: `["sb2"] + SB2ARGS + [cc] + args`.
/// `cc` is the configured compiler/linker string, unsplit.
pub(super) fn sb2_c_compiler_command(sb2: &Sb2, cc: &str, args: &[OsString]) -> Command {
    let args_joined = args
        .iter()
        .map(|a| a.to_string_lossy())
        .collect::<Vec<_>>()
        .join(" ");
    let log_line = format!("sb2 {} {} {}", sb2.args.join(" "), cc, args_joined);
    LOG.method("execute", &log_line);

    let mut cmd = Command::new("sb2");
    cmd.args(&sb2.args);
    cmd.arg(cc);
    cmd.args(args);
    cmd
}

/// `ARM.include_dirs_for_libffi`.
pub(super) fn include_dirs_for_libffi(
    plat: &Platform,
) -> Result<Cow<'static, [OsString]>, super::posix::ValueError> {
    Ok(Cow::Borrowed(
        &plat
            .sb2
            .expect("ARM.include_dirs_for_libffi")
            .available_includedirs,
    ))
}

/// `ARM.library_dirs_for_libffi`.
pub(super) fn library_dirs_for_libffi(
    plat: &Platform,
) -> Result<Cow<'static, [OsString]>, super::posix::ValueError> {
    Ok(Cow::Borrowed(
        &plat
            .sb2
            .expect("ARM.library_dirs_for_libffi")
            .available_librarydirs,
    ))
}

impl Platform {
    /// `ARM._invent_new_name`. `copied_cache` is declared but never read upstream.
    pub(crate) fn _invent_new_name(&self, basepath: &Path, base: &str) -> PathBuf {
        let mut pth = invent_join(basepath, base);
        let mut num = 0;
        // `LocalPath.check()` with no keywords is `os.path.exists`.
        while pth.exists() {
            pth = invent_join(basepath, &format!("{base}_{num}"));
            num += 1;
        }
        fs::create_dir_all(&pth).expect("ARM._invent_new_name ensure(dir=1)");
        pth
    }
}

/// `py.path.local.join`: strip leading `/` from each argument and
/// `normpath` the result.
#[cfg_attr(not(test), expect(dead_code))] // rpython/translator/platform/arm.py ARM._invent_new_name
fn invent_join(basepath: &Path, name: &str) -> PathBuf {
    super::lexical_normpath(&basepath.join(name.trim_start_matches('/')))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn sample_arm() -> Platform {
        arm_platform_from(
            Some(OsString::from("/sb2root")),
            Some("a  b".to_string()),
            Some("gcc".to_string()),
            None,
        )
    }

    fn arm_with_cc(cc: Option<String>, env_cc: Option<String>) -> Platform {
        arm_platform_from(
            Some(OsString::from("/sb2root")),
            Some("a  b".to_string()),
            cc,
            env_cc,
        )
    }

    #[test]
    fn arm_class_name_and_name() {
        let plat = sample_arm();
        assert_eq!(plat.class_name, "ARM");
        assert_eq!(plat.name, "arm");
    }

    #[test]
    fn arm_sb2opt_split_keeps_empty_strings() {
        let plat = sample_arm();
        assert_eq!(
            plat.sb2.as_ref().expect("ARM stores Sb2").args,
            ["a", "", "b"]
        );
    }

    #[test]
    fn arm_preprocess_library_dirs_appends_available() {
        let plat = sample_arm();
        let dirs = plat._preprocess_library_dirs(&[PathBuf::from("/x")]);
        assert_eq!(
            dirs,
            [
                "/x",
                "/sb2root/lib/arm-linux-gnueabi/",
                "/sb2root/lib/arm-linux-gnueabihf/",
                "/sb2root/lib/aarch64-linux-gnu/",
                "/sb2root/usr/lib/arm-linux-gnueabi/",
                "/sb2root/usr/lib/arm-linux-gnueabihf/",
                "/sb2root/usr/lib/aarch64-linux-gnu/",
            ]
            .iter()
            .map(PathBuf::from)
            .collect::<Vec<_>>()
        );
    }

    #[test]
    fn arm_sb2_c_compiler_command_uses_configured_cc_unsplit() {
        let plat = sample_arm();
        let sb2 = plat.sb2.as_ref().expect("ARM stores Sb2");
        let cmd = sb2_c_compiler_command(
            sb2,
            "gcc -m32",
            &[OsString::from("-c"), OsString::from("foo.c")],
        );
        assert_eq!(cmd.get_program(), "sb2");
        let args: Vec<String> = cmd
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        assert_eq!(args, ["a", "", "b", "gcc -m32", "-c", "foo.c"]);
        assert!(cmd.get_current_dir().is_none());
        assert!(cmd.get_envs().next().is_none());
    }

    #[test]
    #[should_panic(expected = "SB2: Provide a path")]
    fn arm_absent_sb2_panics() {
        let _ = arm_platform_from(None, None, None, None);
    }

    #[test]
    fn arm_empty_cc_falls_back_to_env_cc() {
        let plat = arm_with_cc(Some(String::new()), Some("clang".to_string()));
        assert_eq!(plat.cc_command.as_deref(), Some("clang"));
    }

    #[test]
    fn arm_absent_cc_keeps_present_empty_env_cc() {
        let plat = arm_with_cc(None, Some(String::new()));
        assert_eq!(plat.cc_command.as_deref(), Some(""));
    }

    #[test]
    fn arm_absent_cc_and_env_cc_uses_default_cc() {
        let plat = arm_with_cc(None, None);
        assert_eq!(plat.cc_command.as_deref(), Some("gcc"));
    }

    #[test]
    fn arm_nonempty_cc_wins_over_env_cc() {
        let plat = arm_with_cc(Some("g++".to_string()), Some("clang".to_string()));
        assert_eq!(plat.cc_command.as_deref(), Some("g++"));
    }

    #[test]
    fn arm_key_uses_configured_cc_without_host_discovery() {
        let plat = arm_platform_from(
            Some(OsString::from("/sb2root")),
            Some("a  b".to_string()),
            Some("arm-none-gnueabi-gcc-nonexistent".to_string()),
            None,
        );
        let key = plat.key();
        assert!(
            key.contains("cc='arm-none-gnueabi-gcc-nonexistent'"),
            "{key}"
        );
        assert!(!key.contains("host="), "{key}");
        assert!(!key.contains("cc:<"), "{key}");
    }

    #[test]
    fn arm_invent_new_name_joins_leading_slash_onto_basepath() {
        let plat = sample_arm();
        let basepath = crate::tool::udir::udir().join(format!(
            "arm-invent-slash-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        let result = plat._invent_new_name(&basepath, "/child");
        assert_eq!(result, basepath.join("child"));
        assert!(result.is_dir());
    }

    #[test]
    fn arm_invent_new_name_skips_existing_paths() {
        let plat = sample_arm();
        let basepath = crate::tool::udir::udir().join(format!(
            "arm-invent-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        let first = plat._invent_new_name(&basepath, "slot");
        assert_eq!(first, basepath.join("slot"));
        assert!(first.is_dir());
        let second = plat._invent_new_name(&basepath, "slot");
        assert_eq!(second, basepath.join("slot_0"));
        assert!(second.is_dir());
        let file_base = basepath.join("filebase");
        fs::write(&file_base, b"").unwrap();
        let skipped = plat._invent_new_name(&basepath, "filebase");
        assert_eq!(skipped, basepath.join("filebase_0"));
        assert!(skipped.is_dir());
    }

    #[test]
    fn arm_libffi_dirs_ignore_pypy_localbase() {
        let plat = sample_arm();
        let localbase = Some(OsString::from("/foo/baz"));
        let sb2 = plat.sb2.expect("ARM stores Sb2");
        assert_eq!(
            &*plat
                .include_dirs_for_libffi_from(localbase.clone())
                .expect("ARM.include_dirs_for_libffi"),
            sb2.available_includedirs.as_slice()
        );
        assert_eq!(
            &*plat
                .library_dirs_for_libffi_from(localbase)
                .expect("ARM.library_dirs_for_libffi"),
            sb2.available_librarydirs.as_slice()
        );
    }

    #[test]
    fn arm_include_dirs_for_libffi_borrows_class_list() {
        let plat = arm_platform_from(Some(OsString::from("/sb2root")), None, None, None);
        let r = plat
            .include_dirs_for_libffi()
            .expect("ARM.include_dirs_for_libffi");
        assert!(matches!(r, Cow::Borrowed(_)));
        assert!(std::ptr::eq(
            r.as_ptr(),
            plat.sb2.unwrap().available_includedirs.as_ptr()
        ));
    }
}
