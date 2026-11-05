//! Port of `rpython/translator/platform/__init__.py`.
//!
//! Host construction follows `platform.host_factory` (`Linux` / `Darwin` /
//! `Freebsd` / `OpenBSD` / `Netbsd` / `MsvcPlatform` / `MingwPlatform`, else
//! Linux-like posix). `Platform.compile(cfiles, eci, outputfilename=None,
//! standalone=True)` runs `_compile_o_files` then `_finish_linking`.
//! `_handle_error` writes `outname.new(ext='errors')` and honours
//! `log_errors`. `Platform.execute` keeps the upstream copy-or-replace env /
//! library-path injection / CRLF-normalized `ExecutionResult` shape.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::ffi::{OsStr, OsString};
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use crate::tool::ansi_print::AnsiLogger;
use crate::translator::tool::cbuild::ExternalCompilationInfo;
use crate::translator::tool::taskengine::TaskError;
use parking_lot::{ReentrantMutex, RwLock};

mod arm;
mod bsd;
mod darwin;
mod freebsd;
mod linux;
mod netbsd;
mod openbsd;
mod posix;
mod windows;

use posix::py_bytes_repr;

/// `log = AnsiLogger("platform")` (`translator.platform`).
pub static LOG: AnsiLogger = AnsiLogger::new("platform");

/// Port of upstream `CompilationError`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompilationError {
    pub out: String,
    pub err: String,
}

impl CompilationError {
    pub fn new(out: Vec<u8>, err: Vec<u8>) -> Self {
        Self {
            out: normalize_newlines(String::from_utf8_lossy(&out).into_owned()),
            err: normalize_newlines(String::from_utf8_lossy(&err).into_owned()),
        }
    }
}

impl std::fmt::Display for CompilationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (attr, text) = if self.err.is_empty() {
            ("out", &self.out)
        } else {
            ("err", &self.err)
        };
        write!(
            f,
            "CompilationError({attr}=\"\"\"\n\t{}\"\"\")",
            text.replace('\n', "\n\t")
        )
    }
}

impl std::error::Error for CompilationError {}

/// RPython `Platform.execute` receives `env` as a Python dict and immediately
/// does `env.copy()`, so the Rust port uses the direct dict-shaped equivalent
/// here.
pub type EnvMapping = HashMap<String, String>;

/// Port of upstream `ExecutionResult`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionResult {
    pub returncode: i32,
    pub out: String,
    pub err: String,
}

impl ExecutionResult {
    pub fn new(returncode: i32, out: Vec<u8>, err: Vec<u8>) -> Self {
        Self {
            returncode,
            out: normalize_newlines(String::from_utf8_lossy(&out).into_owned()),
            err: normalize_newlines(String::from_utf8_lossy(&err).into_owned()),
        }
    }
}

/// `Popen.returncode` (`runsubprocess._run`): the exit code, or `-N` when
/// the child was terminated by signal `N`.
fn popen_returncode(status: std::process::ExitStatus) -> i32 {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(signal) = status.signal() {
            return -signal;
        }
    }
    status.code().unwrap_or(-1)
}

/// Host subclass chosen by `platform.host_factory`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HostFamily {
    Linux,
    Darwin,
    Freebsd,
    Openbsd,
    Netbsd,
    Msvc,
    Mingw,
}

/// Compiler command built by `_compile_c_file` / `_link` before
/// `_execute_c_compiler`.
#[derive(Debug, Clone)]
pub(crate) struct CompilerInvocation {
    pub program: PathBuf,
    pub args: Vec<OsString>,
    pub cwd: Option<PathBuf>,
}

/// Per-family fields for `Platform.from_spec`.
pub(super) struct FamilySpec {
    pub name: String,
    pub class_name: &'static str,
    pub family: HostFamily,
    pub relevant_environ: &'static [&'static str],
    pub cflags: Vec<OsString>,
    pub link_flags: Vec<OsString>,
    pub standalone_only: Vec<OsString>,
    pub shared_only: Vec<OsString>,
    pub extra_libs: Vec<OsString>,
    pub exe_ext: &'static str,
    pub so_ext: &'static str,
    pub link_path: PathBuf,
    pub masm_path: PathBuf,
}

/// Compiler + MSVC sidecars resolved on first `compile` / `key` / link.
#[derive(Debug, Clone)]
struct ResolvedTools {
    cc: cc::Tool,
    link_tool: Option<cc::Tool>,
    link_path: PathBuf,
    masm_tool: Option<cc::Tool>,
    masm_path: PathBuf,
}

/// Concrete host platform. Upstream's base class is abstract
/// (`Platform.__init__` raises `TypeError` when instantiated directly);
/// `host_factory()` builds the posix/darwin/linux/windows subclass.
#[derive(Debug)]
pub struct Platform {
    pub name: String,
    class_name: &'static str,
    family: HostFamily,
    /// `BasePosix.__init__`: `cc or os.environ.get('CC', self.DEFAULT_CC)`.
    /// `MsvcPlatform.__init__`: always `Some` after construction.
    /// `Platform.__init__(self, 'cl.exe')` when `cc is None`, else `self.cc = cc`.
    cc_command: Option<String>,
    /// Resolved on first `compile` / `key` / link (`try_get_compiler`).
    cc: OnceLock<Result<ResolvedTools, CompilationError>>,
    /// `MsvcPlatform.link` fallback (`link.exe`) before discovery.
    link_path: PathBuf,
    /// `MsvcPlatform.masm` fallback (`ml.exe` / `ml64.exe`) before discovery.
    masm_path: PathBuf,
    /// `MsvcPlatform.__init__`: `self.c_environ = os.environ.copy()` after
    /// `patch_os_env`. `None` on every non-MSVC platform (`Platform.c_environ
    /// = None`), MinGW included.
    pub(super) c_environ: Option<Vec<(OsString, OsString)>>,
    pub relevant_environ: &'static [&'static str],
    pub log_errors: AtomicBool,
    host_triple: String,
    /// `Platform.cflags` / `Linux.cflags` / `Darwin.cflags` / `MsvcPlatform.cflags`.
    cflags: Vec<OsString>,
    /// `Platform.link_flags` / `Linux.link_flags` / `Darwin.link_flags` /
    /// `MsvcPlatform.link_flags`.
    link_flags: Vec<OsString>,
    /// `Linux.standalone_only` / `Darwin.standalone_only` /
    /// `MsvcPlatform.standalone_only`.
    standalone_only: Vec<OsString>,
    /// `Linux.shared_only` / `Darwin.shared_only` / `MsvcPlatform.shared_only`.
    shared_only: Vec<OsString>,
    /// `Linux.extra_libs` (`-lrt`) / empty on Darwin and MSVC.
    extra_libs: Vec<OsString>,
    /// `BasePosix.exe_ext` / `MsvcPlatform.exe_ext`.
    exe_ext: &'static str,
    /// `Linux.so_ext` / `Darwin.so_ext` / `MsvcPlatform.so_ext`.
    so_ext: &'static str,
    /// `ARM` scratchbox2 state (`arm.py`); `None` on every other platform.
    sb2: Option<&'static arm::Sb2>,
}

impl Clone for Platform {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            class_name: self.class_name,
            family: self.family,
            cc_command: self.cc_command.clone(),
            cc: self.cc.clone(),
            link_path: self.link_path.clone(),
            masm_path: self.masm_path.clone(),
            c_environ: self.c_environ.clone(),
            relevant_environ: self.relevant_environ,
            log_errors: AtomicBool::new(self.log_errors.load(Ordering::Relaxed)),
            host_triple: self.host_triple.clone(),
            cflags: self.cflags.clone(),
            link_flags: self.link_flags.clone(),
            standalone_only: self.standalone_only.clone(),
            shared_only: self.shared_only.clone(),
            extra_libs: self.extra_libs.clone(),
            exe_ext: self.exe_ext,
            so_ext: self.so_ext,
            sb2: self.sb2.clone(),
        }
    }
}

/// `Platform.__eq__`: same class and construction-state fields. The lazily
/// filled `cc` cache is derived from `cc_command` and is not compared, so two
/// platforms built from the same state stay equal whether or not one compiled.
impl PartialEq for Platform {
    fn eq(&self, other: &Self) -> bool {
        self.class_name == other.class_name
            && self.name == other.name
            && self.family == other.family
            && self.cc_command == other.cc_command
            && self.link_path == other.link_path
            && self.masm_path == other.masm_path
            && c_environ_eq(&self.c_environ, &other.c_environ)
            && self.relevant_environ == other.relevant_environ
            && self.log_errors.load(Ordering::Relaxed) == other.log_errors.load(Ordering::Relaxed)
            && self.host_triple == other.host_triple
            && self.cflags == other.cflags
            && self.link_flags == other.link_flags
            && self.standalone_only == other.standalone_only
            && self.shared_only == other.shared_only
            && self.extra_libs == other.extra_libs
            && self.exe_ext == other.exe_ext
            && self.so_ext == other.so_ext
            && self.sb2 == other.sb2
    }
}

/// `Platform.__eq__` compares `__dict__`; `c_environ` is a dict, so equality
/// ignores insertion order. The field stays a `Vec` so `Command::env`
/// enumeration order is deterministic.
fn c_environ_eq(
    left: &Option<Vec<(OsString, OsString)>>,
    right: &Option<Vec<(OsString, OsString)>>,
) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => c_environ_as_map(left) == c_environ_as_map(right),
        _ => false,
    }
}

/// Windows `os.environ` upper-cases its keys, so `os.environ.copy()` snapshots
/// compare names case-insensitively there; POSIX names are case-sensitive.
fn c_environ_as_map(env: &[(OsString, OsString)]) -> BTreeMap<OsString, &OsString> {
    env.iter()
        .map(|(k, v)| {
            let key = if cfg!(windows) {
                k.to_ascii_uppercase()
            } else {
                k.clone()
            };
            (key, v)
        })
        .collect()
}

impl Eq for Platform {}

/// `Platform.__hash__`: only `class_name` participates.
impl std::hash::Hash for Platform {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::hash::Hash::hash(&self.class_name, state);
    }
}

impl Platform {
    pub(super) fn from_spec(host_triple: String, spec: FamilySpec) -> Self {
        Self {
            name: spec.name,
            class_name: spec.class_name,
            family: spec.family,
            cc_command: {
                let _guard = CC_COMMAND_LOCK.lock();
                snapshot_cc_command()
            },
            cc: OnceLock::new(),
            link_path: spec.link_path,
            masm_path: spec.masm_path,
            c_environ: None,
            relevant_environ: spec.relevant_environ,
            log_errors: AtomicBool::new(true),
            host_triple,
            cflags: spec.cflags,
            link_flags: spec.link_flags,
            standalone_only: spec.standalone_only,
            shared_only: spec.shared_only,
            extra_libs: spec.extra_libs,
            exe_ext: spec.exe_ext,
            so_ext: spec.so_ext,
            sb2: None,
        }
    }

    /// `platform.host_factory()` construction: records `CC` and never
    /// discovers a compiler (`BasePosix.__init__` / `Platform.__init__`).
    pub fn host() -> Self {
        host_platform()
    }

    pub(crate) fn family(&self) -> HostFamily {
        self.family
    }

    pub(crate) fn is_msvc(&self) -> bool {
        self.family() == HostFamily::Msvc
    }

    pub(crate) fn link_path(&self) -> &Path {
        match self.cc.get() {
            Some(Ok(tools)) => tools.link_path.as_path(),
            _ => self.link_path.as_path(),
        }
    }

    pub(crate) fn masm_path(&self) -> &Path {
        match self.cc.get() {
            Some(Ok(tools)) => tools.masm_path.as_path(),
            _ => self.masm_path.as_path(),
        }
    }

    /// Resolved `cc::Tool` (`self.cc` after discovery). Construction does
    /// not call this; `compile` / `key` / link do.
    pub fn cc(&self) -> Result<&cc::Tool, CompilationError> {
        self.resolve_cc().map(|tools| &tools.cc)
    }

    fn resolve_cc(&self) -> Result<&ResolvedTools, CompilationError> {
        self.cc
            .get_or_init(|| {
                discover_tools(
                    &self.host_triple,
                    self.cc_command.as_deref(),
                    self.family,
                    &self.link_path,
                    &self.masm_path,
                    self.c_environ.as_deref(),
                )
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    fn cc_path(&self) -> PathBuf {
        match self.cc.get() {
            Some(Ok(tools)) => tools.cc.path().to_path_buf(),
            // Cache miss / discovery error: stored `self.cc`
            // (`BasePosix.__init__` / `Platform.__init__(self, 'cl.exe')`).
            _ => PathBuf::from(
                self.cc_command
                    .as_deref()
                    .unwrap_or_else(|| default_cc(self.family)),
            ),
        }
    }

    /// `Platform.key`: class name, `cc` command including args/env,
    /// `relevant_environ` values, and `sys.maxint` word size.
    /// Discovery failure is encoded as `cc:<error>` and never panics.
    pub fn key(&self) -> String {
        let relevant = relevant_environ_values(self.relevant_environ);
        // `Platform.key` (`'cc=%r' % self.cc`): ARM compilation never
        // discovers a host compiler (`ARM._execute_c_compiler` passes the
        // configured `cc` string to `sb2`), so `key()` must not either.
        if self.sb2.is_some() {
            return format_platform_key_configured(
                self.class_name,
                self.cc_command
                    .as_deref()
                    .unwrap_or_else(|| default_cc(self.family)),
                &relevant,
            );
        }
        match self.resolve_cc() {
            Ok(tools) => format_platform_key(
                self.class_name,
                tools.cc.path(),
                tools.cc.args(),
                tools.cc.env(),
                &relevant,
                &self.host_triple,
            ),
            Err(err) => format_platform_key_unresolved(
                self.class_name,
                &discovery_error_tag(&err),
                &relevant,
                &self.host_triple,
            ),
        }
    }

    /// `Platform.compile(cfiles, eci, outputfilename=None, standalone=True)`.
    pub fn compile(
        &self,
        c_files: &[impl AsRef<Path>],
        eci: &ExternalCompilationInfo,
        outputfilename: Option<&str>,
        standalone: bool,
    ) -> Result<PathBuf, CompilationError> {
        // `ARM._execute_c_compiler` never discovers a host `cc`.
        if self.sb2.is_none() {
            self.resolve_cc()?;
        }
        let ofiles = self._compile_o_files(c_files, eci, standalone)?;
        self._finish_linking(&ofiles, eci, outputfilename, standalone)
    }

    /// `Platform._all_cfiles`.
    fn _all_cfiles(
        &self,
        c_files: &[impl AsRef<Path>],
        eci: &ExternalCompilationInfo,
    ) -> Vec<PathBuf> {
        let mut seen = Vec::new();
        let mut result = Vec::new();
        let extras = eci.separate_module_files.iter().map(PathBuf::as_path);
        for cfile in c_files.iter().map(AsRef::as_ref).chain(extras) {
            let cfile = abs_path(cfile);
            if !seen.contains(&cfile) {
                seen.push(cfile.clone());
                result.push(cfile);
            }
        }
        result
    }

    /// `Platform._compile_o_files`. `.asm` (case-insensitive) uses
    /// `MsvcPlatform.masm` with empty extra args.
    fn _compile_o_files(
        &self,
        c_files: &[impl AsRef<Path>],
        eci: &ExternalCompilationInfo,
        standalone: bool,
    ) -> Result<Vec<PathBuf>, CompilationError> {
        let cfiles = self._all_cfiles(c_files, eci);
        let compile_args = self._compile_args_from_eci(eci, standalone);
        let mut ofiles = Vec::with_capacity(cfiles.len());
        for cfile in cfiles {
            ofiles.push(self._compile_c_file(
                &self.compiler_for_source(&cfile),
                &cfile,
                if self.is_msvc() && is_asm_source(&cfile) {
                    &[]
                } else {
                    compile_args.as_slice()
                },
            )?);
        }
        Ok(ofiles)
    }

    /// Compiler used for `cfile`: `self.masm` for MSVC `.asm`, else `self.cc`.
    pub(crate) fn compiler_for_source(&self, cfile: &Path) -> PathBuf {
        if self.is_msvc() && is_asm_source(cfile) {
            self.masm_path().to_path_buf()
        } else {
            self.cc_path()
        }
    }

    /// `BasePosix._compile_c_file` / `MsvcPlatform._compile_c_file`.
    fn _compile_c_file(
        &self,
        cc: &Path,
        cfile: &Path,
        compile_args: &[OsString],
    ) -> Result<PathBuf, CompilationError> {
        let (oname, inv) = self.compile_c_file_invocation(cc, cfile, compile_args);
        self._execute_invocation(&inv, &oname)?;
        Ok(oname)
    }

    pub(crate) fn compile_c_file_invocation(
        &self,
        cc: &Path,
        cfile: &Path,
        compile_args: &[OsString],
    ) -> (PathBuf, CompilerInvocation) {
        let oname = self._make_o_file(cfile, if self.is_msvc() { "obj" } else { "o" });
        let inv = if self.is_msvc() {
            windows::msvc_compile_c_file_invocation(cc, cfile, compile_args, &oname)
        } else {
            posix::posix_compile_c_file_invocation(cc, cfile, compile_args, &oname)
        };
        (oname, inv)
    }

    /// `Platform._make_o_file`.
    fn _make_o_file(&self, cfile: &Path, ext: &str) -> PathBuf {
        let ofile = cfile.with_extension(ext);
        let udir = crate::tool::udir::udir();
        if ofile.starts_with(udir) {
            return ofile;
        }
        let root = rpythonroot();
        if let Ok(rel) = ofile.strip_prefix(&root) {
            let dest = udir.join(rel);
            if let Some(parent) = dest.parent() {
                let _ = fs::create_dir_all(parent);
            }
            return dest;
        }
        panic!("{ofile:?} should be relative to either {root:?} or {udir:?}");
    }

    /// `Platform.preprocess_include_dirs`.
    pub(crate) fn preprocess_include_dirs(&self, include_dirs: &[PathBuf]) -> Vec<PathBuf> {
        prepend_pypy_localbase(self._preprocess_include_dirs(include_dirs), "include")
    }

    /// `Platform._preprocess_include_dirs` / BSD / NetBSD overrides.
    fn _preprocess_include_dirs(&self, include_dirs: &[PathBuf]) -> Vec<PathBuf> {
        match self.family() {
            HostFamily::Freebsd | HostFamily::Openbsd => bsd::preprocess_include_dirs(include_dirs),
            HostFamily::Netbsd => netbsd::preprocess_include_dirs(include_dirs),
            _ => include_dirs.to_vec(),
        }
    }

    /// `Platform.preprocess_library_dirs`.
    pub(crate) fn preprocess_library_dirs(&self, library_dirs: &[PathBuf]) -> Vec<PathBuf> {
        prepend_pypy_localbase(self._preprocess_library_dirs(library_dirs), "lib")
    }

    /// `Platform._preprocess_library_dirs` / BSD / NetBSD / `ARM` overrides.
    fn _preprocess_library_dirs(&self, library_dirs: &[PathBuf]) -> Vec<PathBuf> {
        if let Some(sb2) = self.sb2 {
            let mut dirs = library_dirs.to_vec();
            dirs.extend(sb2.available_librarydirs.iter().cloned().map(PathBuf::from));
            return dirs;
        }
        match self.family() {
            HostFamily::Freebsd | HostFamily::Openbsd => bsd::preprocess_library_dirs(library_dirs),
            HostFamily::Netbsd => netbsd::preprocess_library_dirs(library_dirs),
            _ => library_dirs.to_vec(),
        }
    }

    /// `Platform.include_dirs_for_libffi`.
    pub fn include_dirs_for_libffi(&self) -> Result<Cow<'static, [OsString]>, posix::ValueError> {
        self.include_dirs_for_libffi_from(std::env::var_os("PYPY_LOCALBASE"))
    }

    fn include_dirs_for_libffi_from(
        &self,
        pypy_localbase: Option<OsString>,
    ) -> Result<Cow<'static, [OsString]>, posix::ValueError> {
        if self.sb2.is_some() {
            return arm::include_dirs_for_libffi(self);
        }
        Ok(Cow::Owned(prepend_pypy_localbase_strings(
            self._include_dirs_for_libffi()?,
            "include",
            pypy_localbase,
        )))
    }

    /// `Platform.library_dirs_for_libffi`.
    pub fn library_dirs_for_libffi(&self) -> Result<Cow<'static, [OsString]>, posix::ValueError> {
        self.library_dirs_for_libffi_from(std::env::var_os("PYPY_LOCALBASE"))
    }

    fn library_dirs_for_libffi_from(
        &self,
        pypy_localbase: Option<OsString>,
    ) -> Result<Cow<'static, [OsString]>, posix::ValueError> {
        if self.sb2.is_some() {
            return arm::library_dirs_for_libffi(self);
        }
        Ok(Cow::Owned(prepend_pypy_localbase_strings(
            self._library_dirs_for_libffi()?,
            "lib",
            pypy_localbase,
        )))
    }

    /// `Platform._include_dirs_for_libffi` / Linux / Darwin / BSD / NetBSD /
    /// `MingwPlatform` overrides.
    fn _include_dirs_for_libffi(&self) -> Result<Vec<OsString>, posix::ValueError> {
        match self.family() {
            HostFamily::Linux => linux::_include_dirs_for_libffi(self),
            HostFamily::Darwin => darwin::_include_dirs_for_libffi(self),
            HostFamily::Freebsd | HostFamily::Openbsd => bsd::_include_dirs_for_libffi(self),
            HostFamily::Netbsd => netbsd::_include_dirs_for_libffi(self),
            HostFamily::Mingw => windows::_include_dirs_for_libffi(self),
            HostFamily::Msvc => {
                // `Platform._include_dirs_for_libffi` raises `NotImplementedError`.
                panic!("Needs to be overwritten")
            }
        }
    }

    /// `Platform._library_dirs_for_libffi` / Linux / Darwin / BSD / NetBSD /
    /// `MingwPlatform` overrides.
    fn _library_dirs_for_libffi(&self) -> Result<Vec<OsString>, posix::ValueError> {
        match self.family() {
            HostFamily::Linux => linux::_library_dirs_for_libffi(self),
            HostFamily::Darwin => darwin::_library_dirs_for_libffi(self),
            HostFamily::Freebsd | HostFamily::Openbsd => bsd::_library_dirs_for_libffi(self),
            HostFamily::Netbsd => netbsd::_library_dirs_for_libffi(self),
            HostFamily::Mingw => windows::_library_dirs_for_libffi(self),
            HostFamily::Msvc => {
                // `Platform._library_dirs_for_libffi` raises `NotImplementedError`.
                panic!("Needs to be overwritten")
            }
        }
    }

    /// `BasePosix._pkg_config`.
    fn _pkg_config(
        &self,
        lib: &str,
        opt: &str,
        default: Vec<OsString>,
        check_result_dir: bool,
    ) -> Result<Vec<OsString>, posix::ValueError> {
        posix::_pkg_config(lib, opt, default, check_result_dir)
    }

    /// `BasePosix._includedirs` / `MsvcPlatform._includedirs`.
    fn _includedirs(&self, include_dirs: &[PathBuf]) -> Vec<OsString> {
        if self.is_msvc() {
            windows::msvc_includedirs(include_dirs)
        } else {
            posix::posix_includedirs(include_dirs)
        }
    }

    /// `BasePosix._libdirs` / `MsvcPlatform._libdirs`.
    fn _libdirs(&self, library_dirs: &[PathBuf]) -> Vec<OsString> {
        if self.is_msvc() {
            windows::msvc_libdirs(library_dirs)
        } else {
            posix::posix_libdirs(library_dirs)
        }
    }

    /// `BasePosix._libs` / `MsvcPlatform._libs` / `OpenBSD._libs`.
    fn _libs(&self, libraries: &[String]) -> Vec<OsString> {
        match self.family() {
            HostFamily::Msvc => windows::msvc_libs(libraries),
            HostFamily::Openbsd => openbsd::openbsd_libs(libraries),
            _ => posix::posix_libs(libraries),
        }
    }

    /// `Platform._linkfiles` / `BasePosix._linkfiles` / `MsvcPlatform._linkfiles`.
    fn _linkfiles(&self, link_files: &[PathBuf]) -> Vec<OsString> {
        link_files
            .iter()
            .map(|file| file.as_os_str().to_os_string())
            .collect()
    }

    /// `Platform.get_shared_only_compile_flags` / `BasePosix.get_shared_only_compile_flags`.
    fn get_shared_only_compile_flags(&self) -> Vec<OsString> {
        let mut flags = self.shared_only.clone();
        if !self.is_msvc() {
            flags.push(OsString::from("-fvisibility=hidden"));
        }
        flags
    }

    /// `Platform._compile_args_from_eci`.
    pub(crate) fn _compile_args_from_eci(
        &self,
        eci: &ExternalCompilationInfo,
        standalone: bool,
    ) -> Vec<OsString> {
        let include_dirs = self.preprocess_include_dirs(&eci.include_dirs);
        let args = self._includedirs(&include_dirs);
        let extra = if standalone {
            self.standalone_only.clone()
        } else {
            self.get_shared_only_compile_flags()
        };
        let mut cflags = self.cflags.clone();
        cflags.extend(extra);
        cflags.extend(eci.compile_extra.iter().map(OsString::from));
        cflags.extend(args);
        cflags
    }

    /// `Platform._exportsymbols_link_flags` / `BasePosix._exportsymbols_link_flags` /
    /// `Darwin._exportsymbols_link_flags`.
    fn _exportsymbols_link_flags(&self) -> Vec<OsString> {
        match self.family() {
            HostFamily::Msvc => windows::msvc_exportsymbols_link_flags(),
            HostFamily::Darwin => darwin::darwin_exportsymbols_link_flags(),
            HostFamily::Mingw => posix::posix_export_all_symbols(),
            _ => posix::posix_export_dynamic(),
        }
    }

    /// `Platform._link_args_from_eci` / `Darwin._link_args_from_eci` /
    /// `MsvcPlatform._link_args_from_eci`.
    pub(crate) fn _link_args_from_eci(
        &self,
        eci: &ExternalCompilationInfo,
        standalone: bool,
    ) -> Vec<OsString> {
        let _standalone = if self.is_msvc() { true } else { standalone };
        let library_dirs = self.preprocess_library_dirs(&eci.library_dirs);
        let mut args = self._libdirs(&library_dirs);
        args.extend(self.link_flags.iter().cloned());
        args.extend(self._exportsymbols_link_flags());
        args.extend(self._linkfiles(&eci.link_files));
        args.extend(eci.link_extra.iter().map(OsString::from));
        args.extend(self._libs(&eci.libraries));
        args.extend(self.extra_libs.iter().cloned());
        if self.family() == HostFamily::Darwin {
            args.extend(darwin::frameworks(&eci.frameworks));
            args.extend(self._includedirs(&eci.include_dirs));
        }
        let _ = _standalone;
        args
    }

    /// `g++` (`Platform._finish_linking`: `if eci.use_cpp_linker: cc_link = 'g++'      # XXX hard-coded so far`).
    pub(crate) fn cpp_linker_name(&self) -> &'static str {
        "g++"
    }

    /// Linker program for `_finish_linking` (`eci.use_cpp_linker` → `g++`;
    /// MSVC `_link` still runs `self.link`).
    pub(crate) fn cc_link_program(&self, eci: &ExternalCompilationInfo) -> PathBuf {
        if eci.use_cpp_linker {
            PathBuf::from(self.cpp_linker_name())
        } else {
            self.cc_path()
        }
    }

    /// `Platform._finish_linking`.
    fn _finish_linking(
        &self,
        ofiles: &[PathBuf],
        eci: &ExternalCompilationInfo,
        outputfilename: Option<&str>,
        standalone: bool,
    ) -> Result<PathBuf, CompilationError> {
        let outputfilename = match outputfilename {
            Some(name) => name.to_string(),
            None => ofiles[0]
                .file_stem()
                .unwrap_or_else(|| ofiles[0].as_os_str())
                .to_string_lossy()
                .into_owned(),
        };
        let dirname = if let Some(first) = ofiles.first() {
            first
                .parent()
                .filter(|path| !path.as_os_str().is_empty())
                .map(Path::to_path_buf)
                .unwrap_or_else(|| crate::tool::udir::udir().join("module_cache"))
        } else {
            crate::tool::udir::udir().join("module_cache")
        };
        let mut exe_name = dirname.join(&outputfilename);
        if standalone {
            if !self.exe_ext.is_empty() {
                let mut name = OsString::from(exe_name.as_os_str());
                name.push(".");
                name.push(self.exe_ext);
                exe_name = PathBuf::from(name);
            }
        } else {
            let mut name = OsString::from(exe_name.as_os_str());
            name.push(".");
            name.push(self.so_ext);
            exe_name = PathBuf::from(name);
        }
        let cc_link = self.cc_link_program(eci);
        let largs = self._link_args_from_eci(eci, standalone);
        self._link(
            &cc_link,
            ofiles,
            &largs,
            standalone,
            &exe_name,
            eci.use_cpp_linker,
        )?;
        Ok(exe_name)
    }

    /// `BasePosix._args_for_shared` / `Darwin._args_for_shared` /
    /// `MsvcPlatform._args_for_shared`.
    fn _args_for_shared(&self, args: Vec<OsString>, exe_name: &Path) -> Vec<OsString> {
        match self.family() {
            HostFamily::Msvc => windows::msvc_args_for_shared(args),
            HostFamily::Darwin => darwin::darwin_args_for_shared(&self.shared_only, args, exe_name),
            _ => posix::posix_args_for_shared(args),
        }
    }

    /// `BasePosix._link` / `MsvcPlatform._link`.
    fn _link(
        &self,
        cc: &Path,
        ofiles: &[PathBuf],
        link_args: &[OsString],
        standalone: bool,
        exe_name: &Path,
        use_cpp_linker: bool,
    ) -> Result<(), CompilationError> {
        let inv = self.link_invocation(cc, ofiles, link_args, standalone, exe_name);
        if let Some(sb2) = self.sb2 {
            let cc = if use_cpp_linker {
                self.cpp_linker_name()
            } else {
                self.cc_command
                    .as_deref()
                    .unwrap_or_else(|| default_cc(self.family))
            };
            return self._execute_arm_c_compiler(sb2, cc, &inv.args, exe_name);
        }
        // MSVC `_link` runs `self.link` and ignores `cc_link = 'g++'`.
        self._execute_command(
            self.link_command(&inv.program, use_cpp_linker && !self.is_msvc()),
            &inv,
            exe_name,
        )
    }

    pub(crate) fn link_invocation(
        &self,
        cc: &Path,
        ofiles: &[PathBuf],
        link_args: &[OsString],
        standalone: bool,
        exe_name: &Path,
    ) -> CompilerInvocation {
        if self.is_msvc() {
            return windows::msvc_link_invocation(
                self.link_path(),
                ofiles,
                link_args,
                standalone,
                exe_name,
            );
        }
        let mut args: Vec<OsString> = ofiles
            .iter()
            .map(|ofile| ofile.as_os_str().to_os_string())
            .collect();
        args.extend(link_args.iter().cloned());
        args.push(OsString::from("-o"));
        args.push(exe_name.as_os_str().to_os_string());
        if !standalone {
            args = self._args_for_shared(args, exe_name);
        }
        CompilerInvocation {
            program: cc.to_path_buf(),
            args,
            cwd: exe_name
                .parent()
                .filter(|path| !path.as_os_str().is_empty())
                .map(Path::to_path_buf),
        }
    }

    fn _execute_invocation(
        &self,
        inv: &CompilerInvocation,
        outname: &Path,
    ) -> Result<(), CompilationError> {
        if let Some(sb2) = self.sb2 {
            let cc = self
                .cc_command
                .as_deref()
                .unwrap_or_else(|| default_cc(self.family));
            return self._execute_arm_c_compiler(sb2, cc, &inv.args, outname);
        }
        self._execute_command(self.spawn_compiler(&inv.program), inv, outname)
    }

    fn _execute_arm_c_compiler(
        &self,
        sb2: &arm::Sb2,
        cc: &str,
        args: &[OsString],
        outname: &Path,
    ) -> Result<(), CompilationError> {
        let mut cmd = arm::sb2_c_compiler_command(sb2, cc, args);
        let output = cmd.output().map_err(|err| CompilationError {
            out: String::new(),
            err: format!("failed to spawn sb2: {err}"),
        })?;
        // ARM._execute_c_compiler: `_handle_error(returncode, stderr, stdout, outname)`.
        self._handle_error(
            popen_returncode(output.status),
            output.stderr,
            output.stdout,
            outname,
        )
    }

    fn _execute_command(
        &self,
        mut cmd: Command,
        inv: &CompilerInvocation,
        outname: &Path,
    ) -> Result<(), CompilationError> {
        if let Some(cwd) = &inv.cwd {
            cmd.current_dir(cwd);
        }
        cmd.args(&inv.args);
        let output = cmd.output().map_err(|err| CompilationError {
            out: String::new(),
            err: format!("failed to spawn {}: {err}", inv.program.display()),
        })?;
        self._handle_error(
            popen_returncode(output.status),
            output.stdout,
            output.stderr,
            outname,
        )
    }

    /// Linker `Command` for `_finish_linking`. `use_cpp_linker` is the
    /// `cc_link = 'g++'` path: extra words of `CC` are not carried.
    fn link_command(&self, program: &Path, use_cpp_linker: bool) -> Command {
        if use_cpp_linker {
            let mut cmd = Command::new(program);
            self.apply_c_environ(&mut cmd);
            cmd
        } else {
            self.spawn_compiler(program)
        }
    }

    fn spawn_compiler(&self, program: &Path) -> Command {
        let mut cmd = Command::new(program);
        match self.resolve_cc() {
            Ok(tools) => {
                let overlay_env = if program == tools.cc.path() {
                    cmd.args(tools.cc.args());
                    tools.cc.env()
                } else if let Some(tool) = &tools.link_tool
                    && program == tool.path()
                {
                    cmd.args(tool.args());
                    tool.env()
                } else if let Some(tool) = &tools.masm_tool
                    && program == tool.path()
                {
                    cmd.args(tool.args());
                    tool.env()
                } else {
                    tools.cc.env()
                };
                // All commands run with `env=self.c_environ`
                // (`_execute_c_compiler`, `MsvcPlatform._link` / masm).
                if let Some(environ) = &self.c_environ {
                    Self::apply_environ(&mut cmd, environ);
                } else {
                    for (key, value) in overlay_env {
                        cmd.env(key, value);
                    }
                }
            }
            Err(_) => {
                self.apply_c_environ(&mut cmd);
            }
        }
        cmd
    }

    fn apply_environ(cmd: &mut Command, environ: &[(OsString, OsString)]) {
        cmd.env_clear();
        for (name, value) in environ {
            cmd.env(name, value);
        }
    }

    fn apply_c_environ(&self, cmd: &mut Command) {
        if let Some(environ) = &self.c_environ {
            Self::apply_environ(cmd, environ);
        }
    }

    /// `Platform._handle_error` / `MingwPlatform._handle_error` /
    /// `MsvcPlatform._handle_error`.
    fn _handle_error(
        &self,
        returncode: i32,
        stdout: Vec<u8>,
        stderr: Vec<u8>,
        outname: &Path,
    ) -> Result<(), CompilationError> {
        if returncode != 0 {
            let (out, logged) = match self.family() {
                HostFamily::Msvc => {
                    let mut combined = stdout.clone();
                    combined.extend_from_slice(&stderr);
                    (stdout, combined)
                }
                HostFamily::Mingw => {
                    let mut combined = stderr;
                    combined.extend_from_slice(&stdout);
                    (Vec::new(), combined)
                }
                _ => (stdout, stderr),
            };
            let errorfile = outname.with_extension("errors");
            if let Err(io_err) = fs::write(&errorfile, &logged) {
                // `Platform._handle_error` (`errorfile.write`): an I/O
                // exception escapes uncaught and aborts the translation.
                panic!("{}: {io_err}", errorfile.display());
            }
            self.log_stderr_lines(&logged, false);
            Err(CompilationError::new(out, logged))
        } else {
            self.log_stderr_lines(&stderr, true);
            Ok(())
        }
    }

    fn log_stderr_lines(&self, stderr: &[u8], warning: bool) {
        if !self.log_errors.load(Ordering::Relaxed) {
            return;
        }
        let text = String::from_utf8_lossy(stderr);
        for line in text.lines() {
            if warning {
                LOG.WARNING(line);
            } else {
                LOG.Error(line);
            }
        }
    }

    /// Port of upstream `Platform.execute` / `ARM.execute`.
    pub fn execute(
        &self,
        executable: impl AsRef<Path>,
        args: Option<&[String]>,
        env: Option<&EnvMapping>,
        compilation_info: Option<&ExternalCompilationInfo>,
    ) -> Result<ExecutionResult, TaskError> {
        if let Some(sb2) = self.sb2 {
            let mut log_args = sb2.args.clone();
            log_args.push(executable.as_ref().to_string_lossy().into_owned());
            if let Some(args) = args {
                log_args.extend(args.iter().cloned());
            }
            LOG.method("message", &format!("executing sb2 {}", log_args.join(" ")));
            let mut command = Command::new("sb2");
            command.args(&sb2.args);
            command.arg(executable.as_ref());
            if let Some(args) = args {
                command.args(args);
            }
            if let Some(env) = env {
                command.env_clear();
                command.envs(env);
            }
            let output = command.output().map_err(|e| TaskError {
                message: format!("platform Platform.execute failed to run sb2: {e}"),
            })?;
            return Ok(ExecutionResult::new(
                popen_returncode(output.status),
                output.stdout,
                output.stderr,
            ));
        }
        execute(executable, args, env, compilation_info)
    }
}

/// Serializes `CC` snapshot during `host_factory` construction so tests can
/// point `CC` at a missing compiler without racing other `Platform::host()`
/// calls.
static CC_COMMAND_LOCK: ReentrantMutex<()> = ReentrantMutex::new(());

/// Module-global `platform` (`set_platform` / `pick_platform`). Lazily the
/// same `Arc` as `host` from `host_factory()`.
static PLATFORM: RwLock<Option<Arc<Platform>>> = RwLock::new(None);
/// Module-global `host` (`set_platform` rebinds it only when `new_platform ==
/// "host"`).
static HOST: RwLock<Option<Arc<Platform>>> = RwLock::new(None);

/// `platform = host = host_factory()`: first access of either global binds
/// both to the same `Arc`.
fn ensure_initialised() {
    let mut plat_slot = PLATFORM.write();
    let mut host_slot = HOST.write();
    if plat_slot.is_some() && host_slot.is_some() {
        return;
    }
    let plat = Arc::new(host_platform());
    if plat_slot.is_none() {
        *plat_slot = Some(Arc::clone(&plat));
    }
    if host_slot.is_none() {
        *host_slot = Some(plat);
    }
}

/// Module-global `platform = host = host_factory()`. Construction records
/// the `CC` command and does not discover a compiler.
pub(crate) fn platform() -> Arc<Platform> {
    ensure_initialised();
    Arc::clone(
        PLATFORM
            .read()
            .as_ref()
            .expect("ensure_initialised fills PLATFORM"),
    )
}

/// Module-global `host` (`translator.platform.host`).
pub(crate) fn host() -> Arc<Platform> {
    ensure_initialised();
    Arc::clone(HOST.read().as_ref().expect("ensure_initialised fills HOST"))
}

/// `is_host_build`: `host == platform` via `Platform.__eq__`.
#[cfg_attr(not(test), expect(dead_code))] // rpython/translator/platform/__init__.py is_host_build
pub(crate) fn is_host_build() -> bool {
    *host() == *platform()
}

/// `pick_platform(new_platform, cc)`. `"host"` is `host_factory(cc)`; `"arm"`
/// is `ARM(cc)`; anything else is `ValueError("platform = %s")`.
pub(crate) fn pick_platform(new_platform: &str, cc: Option<String>) -> Result<Platform, String> {
    if new_platform == "host" {
        Ok(match cc.as_deref() {
            None | Some("") => host_platform(),
            Some(_) => host_platform_with_cc(cc),
        })
    } else if new_platform == "arm" {
        Ok(arm::arm_platform(cc))
    } else {
        Err(format!("platform = {new_platform}"))
    }
}

/// `set_platform(new_platform, cc)`: rebind `platform`, and `host` when
/// `new_platform == "host"`.
#[cfg_attr(not(test), expect(dead_code))] // rpython/translator/platform/__init__.py set_platform
pub(crate) fn set_platform(new_platform: &str, cc: Option<String>) -> Result<(), String> {
    ensure_initialised();
    let plat = pick_platform(new_platform, cc.clone())?;
    let using_cc = plat
        .cc_command
        .clone()
        .unwrap_or_else(|| default_cc(plat.family).to_string());
    let arc = Arc::new(plat);
    *PLATFORM.write() = Some(Arc::clone(&arc));
    LOG.method(
        "msg",
        &format!(
            "Set platform with {} cc={}, using cc={}, version={}",
            py_bytes_repr(new_platform.as_bytes()),
            cc.as_deref().unwrap_or("None"),
            py_bytes_repr(using_cc.as_bytes()),
            py_bytes_repr(b"Unknown"),
        ),
    );
    if new_platform == "host" {
        *HOST.write() = Some(arc);
    }
    Ok(())
}

fn host_platform() -> Platform {
    let _guard = CC_COMMAND_LOCK.lock();
    host_platform_with_cc(snapshot_cc_command())
}

fn host_platform_with_cc(cc_command: Option<String>) -> Platform {
    let host_triple = host_triple();
    // `_get_compiler_type` must see the raw `CC` value (unset vs set) before
    // `BasePosix.__init__` applies `DEFAULT_CC`.
    let family = if std::env::consts::OS == "windows" {
        _get_compiler_type(cc_command.as_deref())
    } else {
        host_family()
    };
    let mut plat = if family == HostFamily::Msvc {
        // `_get_compiler_type`: `if not cc: MsvcPlatform(x64)` (`cc is None`).
        // `MsvcPlatform.__init__`: `if cc is None: find_msvc_env(...)`.
        let msvc_cc = match cc_command.as_deref() {
            None | Some("") => None,
            Some(_) => cc_command.clone(),
        };
        let is_64 = cfg!(target_pointer_width = "64");
        windows::msvc_platform(host_triple, is_64, msvc_cc)
    } else {
        platform_for_family(family, host_triple)
    };
    // Windows constructors already carry `MsvcPlatform` / `MingwPlatform`
    // `class_name` and `msvc` / `mingw32` `name`; do not overwrite them.
    if std::env::consts::OS != "windows" {
        plat.class_name = host_class_name();
        plat.name = host_platform_name();
    }
    plat.relevant_environ = match family {
        HostFamily::Msvc => windows::RELEVANT_ENVIRON,
        _ => posix::RELEVANT_ENVIRON,
    };
    if family != HostFamily::Msvc {
        // `BasePosix.__init__`: `cc or os.environ.get('CC', self.DEFAULT_CC)`.
        // A present `Some("")` is `os.environ.get` returning `''` and is kept.
        plat.cc_command = Some(cc_command.unwrap_or_else(|| default_cc(family).to_string()));
    }
    plat
}

fn platform_for_family(family: HostFamily, host_triple: String) -> Platform {
    let is_64 = cfg!(target_pointer_width = "64");
    match family {
        HostFamily::Linux => linux::linux_platform(host_triple, cfg!(target_arch = "s390x")),
        HostFamily::Darwin => darwin::darwin_platform(host_triple),
        HostFamily::Freebsd => freebsd::freebsd_platform(host_triple, is_64),
        HostFamily::Openbsd => openbsd::openbsd_platform(host_triple, is_64),
        HostFamily::Netbsd => netbsd::netbsd_platform(host_triple, is_64),
        HostFamily::Msvc => windows::msvc_platform(host_triple, is_64, None),
        HostFamily::Mingw => windows::mingw_platform(host_triple),
    }
}

fn snapshot_cc_command() -> Option<String> {
    std::env::var("CC").ok()
}

/// Upstream `DEFAULT_CC` (or the `__init__` default when the class has none):
/// `Linux` inherits `BasePosix.DEFAULT_CC` (`gcc`); `Darwin.DEFAULT_CC`
/// (`clang`); `Freebsd` inherits `BSD.DEFAULT_CC` (`clang`);
/// `OpenBSD.DEFAULT_CC` (`cc`); `Netbsd.__init__` `get_env("CC", "gcc")`;
/// `MingwPlatform.__init__` (`gcc`); `MsvcPlatform.__init__` passes
/// `'cl.exe'` to `Platform.__init__` after `find_msvc_env` when `cc is None`.
fn default_cc(family: HostFamily) -> &'static str {
    match family {
        HostFamily::Linux => "gcc",
        HostFamily::Darwin => "clang",
        HostFamily::Freebsd => "clang",
        HostFamily::Openbsd => "cc",
        HostFamily::Netbsd => "gcc",
        HostFamily::Mingw => "gcc",
        HostFamily::Msvc => "cl.exe",
    }
}

fn discover_tools(
    host_triple: &str,
    cc_command: Option<&str>,
    family: HostFamily,
    fallback_link: &Path,
    fallback_masm: &Path,
    c_environ: Option<&[(OsString, OsString)]>,
) -> Result<ResolvedTools, CompilationError> {
    let cc = host_c_compiler(host_triple, cc_command, c_environ)?;
    if family != HostFamily::Msvc {
        return Ok(ResolvedTools {
            cc,
            link_tool: None,
            link_path: PathBuf::new(),
            masm_tool: None,
            masm_path: PathBuf::new(),
        });
    }
    let link_name = fallback_link
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("link.exe");
    let masm_name = fallback_masm
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("ml.exe");
    let (link_tool, link_path) =
        windows::discover_msvc_sidecar(&cc, host_triple, link_name, c_environ);
    let (masm_tool, masm_path) =
        windows::discover_msvc_sidecar(&cc, host_triple, masm_name, c_environ);
    Ok(ResolvedTools {
        cc,
        link_tool,
        link_path,
        masm_tool,
        masm_path,
    })
}

/// `cc::Build` host-triple compiler selection. `cc_command` is the recorded
/// `CC` / `DEFAULT_CC` program (`BasePosix.__init__`); `None` is the
/// `MsvcPlatform.__init__` `find_msvc_env` analog and leaves discovery to `cc`.
pub(super) fn host_c_compiler(
    host_triple: &str,
    cc_command: Option<&str>,
    c_environ: Option<&[(OsString, OsString)]>,
) -> Result<cc::Tool, CompilationError> {
    let mut build = cc::Build::new();
    build
        .target(host_triple)
        .host(host_triple)
        .opt_level(0)
        .cargo_metadata(false)
        .warnings(false);
    if let Some(cc) = cc_command {
        let mut words = cc.split_whitespace();
        let program = words.next().unwrap_or(cc);
        let program_path = Path::new(program);
        let resolved = match resolve_compiler_on_path(program_path, c_environ) {
            Some(path) => path,
            None => {
                return Err(CompilationError {
                    out: String::new(),
                    err: format!("cannot find C compiler {program}"),
                });
            }
        };
        // `Build::env` is not visible to `get_env` / `which` /
        // `find_msvc_tools_find_tool` (those call `env::var_os`). When the
        // snapshot is present, pass the snapshot-PATH absolute path so `cc`
        // does not re-search the unpatched process PATH.
        if c_environ.is_some() {
            build.compiler(&resolved);
        } else {
            build.compiler(program);
        }
        // `Platform._execute_c_compiler`: `cclist = cc.split(); args = cclist[1:] + args`.
        // `Build::flag` lands in `Tool::args()` / `to_command()`.
        for extra in words {
            build.flag(extra);
        }
    }
    let tool = build.try_get_compiler().map_err(|err| CompilationError {
        out: String::new(),
        err: err.to_string(),
    })?;
    if !compiler_exists(tool.path(), c_environ) {
        return Err(CompilationError {
            out: String::new(),
            err: format!("cannot find C compiler {}", tool.path().display()),
        });
    }
    Ok(tool)
}

fn compiler_exists(path: &Path, c_environ: Option<&[(OsString, OsString)]>) -> bool {
    resolve_compiler_on_path(path, c_environ).is_some()
}

/// Snapshot keys match the OS-returned spelling. Windows env names are
/// case-insensitive; POSIX names are case-sensitive.
pub(super) fn env_name_eq(key: &OsString, name: &str) -> bool {
    if key == std::ffi::OsStr::new(name) {
        true
    } else {
        cfg!(windows) && key.to_str().is_some_and(|k| k.eq_ignore_ascii_case(name))
    }
}

/// POSIX `execvp` / `Platform._execute_c_compiler`: a name is looked up on
/// `PATH` only when it has no directory component. An absolute path or a
/// path with a separator is used as given.
fn path_has_directory_component(path: &Path) -> bool {
    path.is_absolute() || path.components().count() > 1
}

/// Windows command-shell search sequence, reached through `_run_subprocess`
/// (`runsubprocess.py` `_run`: `shell = shell_default`, `True` on win32) with
/// `env=self.c_environ`. A name with an extension is matched exactly; a name
/// without one gets each `PATHEXT` extension appended in order. A name with a
/// directory component is looked up only there; a bare name is looked up in
/// the current directory, then in each `PATH` entry (`;`-separated; a `"`
/// toggles quoting and is dropped, and a `;` inside quotes belongs to the entry).
fn shell_command_search(executable: &Path, path_var: &OsStr, pathext: &OsStr) -> Option<PathBuf> {
    let exts: Vec<&OsStr> = if executable.extension().is_some() {
        vec![OsStr::new("")]
    } else {
        pathext
            .as_encoded_bytes()
            .split(|&b| b == b';')
            .filter(|piece| !piece.is_empty())
            .map(|piece| {
                // SAFETY: splitting on an ASCII byte yields valid encoded-bytes
                // boundaries (documented on `OsStr::from_encoded_bytes_unchecked`).
                unsafe { OsStr::from_encoded_bytes_unchecked(piece) }
            })
            .collect()
    };

    let candidate_name = |ext: &OsStr| {
        let mut name = executable.as_os_str().to_os_string();
        name.push(ext);
        name
    };

    if path_has_directory_component(executable) {
        for ext in &exts {
            let candidate = PathBuf::from(candidate_name(ext));
            if candidate.is_file() {
                return Some(candidate);
            }
        }
        return None;
    }

    for ext in &exts {
        let candidate = PathBuf::from(candidate_name(ext));
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    for dir in shell_path_entries(path_var) {
        for ext in &exts {
            let candidate = Path::new(&dir).join(candidate_name(ext));
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

/// Windows command-shell `PATH` entries: a `"` toggles quoting and is dropped;
/// a `;` outside quotes ends the current entry, and a `;` inside quotes is kept.
/// Empty entries are skipped. Models the shell that `_run_subprocess` `_run`
/// reaches with `shell=True` on win32.
fn shell_path_entries(path_var: &OsStr) -> Vec<OsString> {
    let mut entries = Vec::new();
    let mut current = Vec::new();
    let mut in_quote = false;
    for &b in path_var.as_encoded_bytes() {
        if b == b'"' {
            in_quote = !in_quote;
        } else if b == b';' && !in_quote {
            if !current.is_empty() {
                // SAFETY: only ASCII bytes `"`/`;` are removed or split on, so
                // the remaining pieces are valid encoded bytes per
                // `OsStr::from_encoded_bytes_unchecked` docs.
                entries.push(unsafe {
                    OsString::from_encoded_bytes_unchecked(std::mem::take(&mut current))
                });
            }
        } else {
            current.push(b);
        }
    }
    if !current.is_empty() {
        // SAFETY: only ASCII bytes `"`/`;` are removed or split on, so the
        // remaining pieces are valid encoded bytes per
        // `OsStr::from_encoded_bytes_unchecked` docs.
        entries.push(unsafe { OsString::from_encoded_bytes_unchecked(current) });
    }
    entries
}

/// Locate `path` as a compiler. Windows is the command-shell search
/// sequence reached through `_run_subprocess` (`runsubprocess.py` `_run`:
/// `shell = shell_default`, `True` on win32) with `env=self.c_environ`.
/// POSIX (`execvp` / `Platform._execute_c_compiler`) returns a
/// path with a directory component only when it is a file, and searches
/// `PATH` for a bare name only. `PATH` is taken from `c_environ` when
/// given (`MsvcPlatform.__init__` snapshot after `patch_os_env`) and the
/// process `PATH` otherwise. Windows `PATHEXT` is taken from that snapshot
/// or the process environment, defaulting to `.COM;.EXE;.BAT;.CMD`.
pub(super) fn resolve_compiler_on_path(
    path: &Path,
    c_environ: Option<&[(OsString, OsString)]>,
) -> Option<PathBuf> {
    if cfg!(windows) {
        let path_var = match c_environ {
            Some(environ) => environ
                .iter()
                .find(|(key, _)| env_name_eq(key, "PATH"))
                .map(|(_, value)| value.clone())
                .unwrap_or_default(),
            None => std::env::var_os("PATH").unwrap_or_default(),
        };
        let pathext = match c_environ {
            Some(environ) => environ
                .iter()
                .find(|(key, _)| env_name_eq(key, "PATHEXT"))
                .map(|(_, value)| value.clone())
                .unwrap_or_else(|| OsString::from(".COM;.EXE;.BAT;.CMD")),
            None => {
                std::env::var_os("PATHEXT").unwrap_or_else(|| OsString::from(".COM;.EXE;.BAT;.CMD"))
            }
        };
        shell_command_search(path, &path_var, &pathext)
    } else {
        if path_has_directory_component(path) {
            return path.is_file().then(|| path.to_path_buf());
        }
        let name = path.file_name()?;
        let path_var = match c_environ {
            Some(environ) => environ
                .iter()
                .find(|(key, _)| env_name_eq(key, "PATH"))
                .map(|(_, value)| value.clone())?,
            None => std::env::var_os("PATH")?,
        };
        std::env::split_paths(&path_var).find_map(|dir| {
            let candidate = dir.join(name);
            candidate.is_file().then_some(candidate)
        })
    }
}

pub(crate) fn host_triple() -> String {
    std::env::var("HOST").unwrap_or_else(|_| env!("MAJIT_TRANSLATE_HOST").to_string())
}

/// `_get_compiler_type` (`windows.py`): empty/missing `cc` → `MsvcPlatform`;
/// `cc.startswith('mingw')` or `cc == 'gcc'` → `MingwPlatform`; else
/// `MsvcPlatform`.
fn _get_compiler_type(cc_command: Option<&str>) -> HostFamily {
    let cc = cc_command.unwrap_or("");
    if cc.is_empty() {
        HostFamily::Msvc
    } else if cc.starts_with("mingw") || cc == "gcc" {
        HostFamily::Mingw
    } else {
        HostFamily::Msvc
    }
}

/// `sys.platform` / `os.name` dispatch in `translator.platform`. On Windows
/// the family follows `_get_compiler_type` over the recorded `CC` snapshot.
pub(crate) fn host_family() -> HostFamily {
    match std::env::consts::OS {
        "linux" => HostFamily::Linux,
        "macos" => HostFamily::Darwin,
        "freebsd" => HostFamily::Freebsd,
        "openbsd" => HostFamily::Openbsd,
        "netbsd" => HostFamily::Netbsd,
        "windows" => {
            let _guard = CC_COMMAND_LOCK.lock();
            _get_compiler_type(snapshot_cc_command().as_deref())
        }
        _ => HostFamily::Linux,
    }
}

/// `name` of the object `host_factory()` returns (`Darwin_arm64.name` is
/// `darwin_arm64`; `MsvcPlatform.name` is `msvc`; `MingwPlatform.name` is
/// `mingw32`).
fn host_platform_name() -> String {
    match host_class_name() {
        "Darwin_arm64" => "darwin_arm64".to_string(),
        "Darwin_x86_64" => "darwin_x86_64".to_string(),
        "Darwin_i386" => "darwin_i386".to_string(),
        "Darwin_PowerPC" => "darwin_powerpc".to_string(),
        "Darwin" => "darwin".to_string(),
        "Linux" => "linux".to_string(),
        "MsvcPlatform" => "msvc".to_string(),
        "MingwPlatform" => "mingw32".to_string(),
        "Freebsd" | "Freebsd_64" => "freebsd".to_string(),
        "OpenBSD" | "OpenBSD_64" => "openbsd".to_string(),
        "Netbsd" | "Netbsd_64" => "netbsd".to_string(),
        _ => std::env::consts::OS.to_string(),
    }
}

/// Class of the object `host_factory()` returns. `Windows` / `Windows_x64`
/// are factory functions, so the Windows class is `MsvcPlatform` or
/// `MingwPlatform`. Darwin uses `Darwin_arm64` / `Darwin_x86_64` /
/// `Darwin_i386` / `Darwin_PowerPC` / `Darwin`.
fn host_class_name() -> &'static str {
    match host_family() {
        HostFamily::Linux => "Linux",
        HostFamily::Darwin => {
            if cfg!(target_arch = "aarch64") {
                "Darwin_arm64"
            } else if cfg!(target_arch = "x86_64") {
                "Darwin_x86_64"
            } else if cfg!(target_arch = "x86") {
                "Darwin_i386"
            } else if cfg!(any(target_arch = "powerpc", target_arch = "powerpc64")) {
                "Darwin_PowerPC"
            } else {
                "Darwin"
            }
        }
        HostFamily::Freebsd => {
            if cfg!(target_pointer_width = "64") {
                "Freebsd_64"
            } else {
                "Freebsd"
            }
        }
        HostFamily::Openbsd => {
            if cfg!(target_pointer_width = "64") {
                "OpenBSD_64"
            } else {
                "OpenBSD"
            }
        }
        HostFamily::Netbsd => {
            if cfg!(target_pointer_width = "64") {
                "Netbsd_64"
            } else {
                "Netbsd"
            }
        }
        HostFamily::Msvc => "MsvcPlatform",
        HostFamily::Mingw => "MingwPlatform",
    }
}

/// `BasePosix.relevant_environ` / `MsvcPlatform.relevant_environ`.
/// `MingwPlatform` inherits `BasePosix.relevant_environ`.
#[cfg(test)]
pub(crate) fn host_relevant_environ() -> &'static [&'static str] {
    match host_family() {
        HostFamily::Msvc => windows::RELEVANT_ENVIRON,
        _ => posix::RELEVANT_ENVIRON,
    }
}

pub(super) fn os_flags<const N: usize>(flags: [&str; N]) -> Vec<OsString> {
    flags.into_iter().map(OsString::from).collect()
}

pub(super) fn split_env_flags(name: &str) -> Vec<OsString> {
    std::env::var(name)
        .ok()
        .map(|value| value.split_whitespace().map(OsString::from).collect())
        .unwrap_or_default()
}

fn relevant_environ_values(names: &[&'static str]) -> Vec<(&'static str, Option<OsString>)> {
    names
        .iter()
        .copied()
        .map(|name| (name, std::env::var_os(name)))
        .collect()
}

/// String form of `Platform.key` with an injectable compiler identity.
pub(crate) fn format_platform_key(
    class_name: &str,
    cc_path: &Path,
    cc_args: &[OsString],
    cc_env: &[(OsString, OsString)],
    relevant_env: &[(&str, Option<OsString>)],
    host_triple: &str,
) -> String {
    let mut bits = Vec::new();
    bits.push(class_name.to_string());
    bits.push(format!("cc={cc_path:?}"));
    bits.push(format!("args={cc_args:?}"));
    bits.push(format!("env={cc_env:?}"));
    bits.push(format!("host={host_triple:?}"));
    for (name, value) in relevant_env {
        bits.push(format!("{name}={value:?}"));
    }
    bits.push(format!("sys.maxint={}", isize::MAX));
    bits.join(" ")
}

fn discovery_error_tag(err: &CompilationError) -> String {
    let text = if err.err.is_empty() {
        err.out.as_str()
    } else {
        err.err.as_str()
    };
    text.replace('\n', " ")
}

/// `Platform.key` when compiler discovery failed: `cc:<error>` is stable
/// and never panics.
pub(crate) fn format_platform_key_unresolved(
    class_name: &str,
    error_tag: &str,
    relevant_env: &[(&str, Option<OsString>)],
    host_triple: &str,
) -> String {
    let mut bits = Vec::new();
    bits.push(class_name.to_string());
    bits.push(format!("cc:<{error_tag}>"));
    bits.push(format!("host={host_triple:?}"));
    for (name, value) in relevant_env {
        bits.push(format!("{name}={value:?}"));
    }
    bits.push(format!("sys.maxint={}", isize::MAX));
    bits.join(" ")
}

/// `Platform.key` from the configured `cc` string (ARM never discovers a host
/// compiler, `ARM._execute_c_compiler`); bits and `%r` rendering as upstream.
pub(crate) fn format_platform_key_configured(
    class_name: &str,
    cc: &str,
    relevant_env: &[(&str, Option<OsString>)],
) -> String {
    let mut bits = Vec::new();
    bits.push(class_name.to_string());
    bits.push(format!("cc={}", py_bytes_repr(cc.as_bytes())));
    for (name, value) in relevant_env {
        bits.push(format!(
            "{name}={}",
            match value {
                Some(v) => py_bytes_repr(v.as_encoded_bytes()),
                None => "None".to_string(),
            }
        ));
    }
    bits.push(format!("sys.maxint={}", isize::MAX));
    bits.join(" ")
}

/// Free-function helper for call sites that do not need a platform value.
pub fn execute(
    executable: impl AsRef<Path>,
    args: Option<&[String]>,
    env: Option<&EnvMapping>,
    compilation_info: Option<&ExternalCompilationInfo>,
) -> Result<ExecutionResult, TaskError> {
    let mut command = Command::new(executable.as_ref());
    if let Some(args) = args {
        command.args(args);
    }

    // `env is None` means copy process environment; a provided env replaces
    // it. `Command` inherits by default, so only the replacement branch needs
    // `env_clear`.
    if let Some(env) = env {
        command.env_clear();
        command.envs(env);
        // Windows programs often require SystemRoot.
        #[cfg(windows)]
        if !env.contains_key("SystemRoot")
            && let Ok(system_root) = std::env::var("SystemRoot")
        {
            command.env("SystemRoot", system_root);
        }
    }

    // Inject loader search path for compiled artifacts.
    if cfg!(unix)
        && let Some(info) = compilation_info
    {
        let joined = info
            .library_dirs
            .iter()
            .map(|path| path.to_string_lossy())
            .collect::<Vec<_>>()
            .join(":");
        if cfg!(target_os = "macos") {
            command.env("DYLD_LIBRARY_PATH", joined);
        } else {
            command.env("LD_LIBRARY_PATH", joined);
        }
    }

    let output = command.output().map_err(|e| TaskError {
        message: format!(
            "platform Platform.execute failed to run {}: {e}",
            executable.as_ref().display()
        ),
    })?;
    Ok(ExecutionResult::new(
        popen_returncode(output.status),
        output.stdout,
        output.stderr,
    ))
}

fn abs_path(path: &Path) -> PathBuf {
    let joined = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
    };
    lexical_normpath(&joined)
}

/// `os.path.normpath` / `py.path.local`: collapse `.`, `..`, and duplicate
/// separators lexically, without resolving symlinks (`canonicalize`).
/// On POSIX, `posixpath.normpath` keeps a leading `//` (exactly two slashes)
/// and collapses three or more to one.
pub(super) fn lexical_normpath(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => match out.components().next_back() {
                Some(Component::Normal(_)) => {
                    out.pop();
                }
                Some(Component::Prefix(_) | Component::RootDir) => {}
                _ => out.push(component),
            },
            component => out.push(component),
        }
    }
    if out.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        #[cfg(unix)]
        {
            // `posixpath.normpath`: `initial_slashes = 2` when the path
            // starts with exactly `//` (not `///`). Rebuild on `OsStr` bytes.
            use std::os::unix::ffi::{OsStrExt, OsStringExt};
            let raw = path.as_os_str().as_bytes();
            if raw.starts_with(b"//") && !raw.starts_with(b"///") {
                let out_bytes = out.as_os_str().as_bytes();
                let rest = out_bytes.strip_prefix(b"/").unwrap_or(out_bytes);
                let mut bytes = Vec::with_capacity(2 + rest.len());
                bytes.extend_from_slice(b"//");
                bytes.extend_from_slice(rest);
                return PathBuf::from(OsString::from_vec(bytes));
            }
        }
        out
    }
}

pub(super) fn prefixed_path(prefix: &str, path: &Path) -> OsString {
    let mut flag = OsString::from(prefix);
    flag.push(path);
    flag
}

/// `str(cfile).lower().endswith('.asm')`.
pub(crate) fn is_asm_source(path: &Path) -> bool {
    path.to_string_lossy()
        .to_ascii_lowercase()
        .ends_with(".asm")
}

fn prepend_pypy_localbase(dirs: Vec<PathBuf>, suffix: &str) -> Vec<PathBuf> {
    prepend_pypy_localbase_from(dirs, suffix, std::env::var_os("PYPY_LOCALBASE"))
}

/// `'PYPY_LOCALBASE' in os.environ` — presence, including an empty value.
fn prepend_pypy_localbase_from(
    dirs: Vec<PathBuf>,
    suffix: &str,
    localbase: Option<OsString>,
) -> Vec<PathBuf> {
    match localbase {
        Some(base) => {
            let mut joined = base;
            joined.push("/");
            joined.push(suffix);
            let mut out = vec![PathBuf::from(joined)];
            out.extend(dirs);
            out
        }
        None => dirs,
    }
}

/// `os.environ['PYPY_LOCALBASE'] + '/include'` (resp. `'/lib'`).
fn prepend_pypy_localbase_strings(
    dirs: Vec<OsString>,
    suffix: &str,
    localbase: Option<OsString>,
) -> Vec<OsString> {
    match localbase {
        Some(localbase) => {
            let mut s = localbase.clone();
            s.push("/");
            s.push(suffix);
            let mut out = vec![s];
            out.extend(dirs);
            out
        }
        None => dirs,
    }
}

/// Parent of the `rpython` package; `rpython.tool.version.rpythonroot`.
pub(super) fn rpythonroot() -> PathBuf {
    let mut root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    root.pop();
    root.pop();
    root
}

fn normalize_newlines(s: String) -> String {
    s.replace("\r\n", "\n")
}

#[cfg(test)]
mod tests {
    use super::windows;
    use super::*;
    use std::fs;

    #[test]
    fn execution_result_normalizes_crlf() {
        let result = ExecutionResult::new(7, b"a\r\nb\r\n".to_vec(), b"e\r\n".to_vec());

        assert_eq!(result.returncode, 7);
        assert_eq!(result.out, "a\nb\n");
        assert_eq!(result.err, "e\n");
    }

    #[cfg(unix)]
    #[test]
    fn popen_returncode_reports_signal_as_negative() {
        use std::os::unix::process::ExitStatusExt;
        assert_eq!(
            popen_returncode(std::process::ExitStatus::from_raw(15)),
            -15
        );
        assert_eq!(
            popen_returncode(std::process::ExitStatus::from_raw(3 << 8)),
            3
        );
    }

    #[test]
    fn compilation_error_normalizes_crlf_and_prefers_err() {
        let err = CompilationError::new(b"out\r\n".to_vec(), b"err\r\n".to_vec());

        assert_eq!(err.out, "out\n");
        assert_eq!(err.err, "err\n");
        assert!(err.to_string().starts_with("CompilationError(err="));
    }

    #[cfg(unix)]
    #[test]
    fn execute_runs_program_and_captures_output() {
        let args = vec!["-c".to_string(), "printf 'ok\\n'".to_string()];
        let result = execute("/bin/sh", Some(&args), None, None).expect("execute");

        assert_eq!(result.returncode, 0);
        assert_eq!(result.out, "ok\n");
        assert_eq!(result.err, "");
    }

    #[cfg(unix)]
    #[test]
    fn execute_replaces_environment_when_env_is_provided() {
        let args = vec![
            "-c".to_string(),
            "printf '%s' \"$RPY_PLATFORM_TEST_VALUE\"".to_string(),
        ];
        let mut env = EnvMapping::new();
        env.insert("RPY_PLATFORM_TEST_VALUE".to_string(), "visible".to_string());

        let result = execute("/bin/sh", Some(&args), Some(&env), None).expect("execute");

        assert_eq!(result.returncode, 0);
        assert_eq!(result.out, "visible");
    }

    #[test]
    fn format_platform_key_configured_matches_upstream_repr() {
        let key = format_platform_key_configured(
            "ARM",
            "gcc -O2",
            &[("CC", None), ("CFLAGS", Some(OsString::from("it's")))],
        );
        let expected = format!(
            "ARM cc='gcc -O2' CC=None CFLAGS=\"it's\" sys.maxint={}",
            isize::MAX
        );
        assert_eq!(key, expected);
    }

    #[test]
    fn key_differs_when_compiler_args_differ() {
        let path = Path::new("cc");
        let triple = "x86_64-unknown-linux-gnu";
        let extra = [OsString::from("-m32")];
        assert_ne!(
            format_platform_key("Linux", path, &[], &[], &[], triple),
            format_platform_key("Linux", path, &extra, &[], &[], triple)
        );
    }

    #[test]
    fn key_differs_when_relevant_environ_differs() {
        let path = Path::new("cc");
        let triple = "x86_64-unknown-linux-gnu";
        let name = host_relevant_environ()[0];
        let unset: [(&str, Option<OsString>); 1] = [(name, None)];
        let set: [(&str, Option<OsString>); 1] = [(name, Some(OsString::from("/changed/include")))];
        assert_ne!(
            format_platform_key("Linux", path, &[], &[], &unset, triple),
            format_platform_key("Linux", path, &[], &[], &set, triple)
        );
    }

    #[test]
    fn relevant_environ_matches_host_family() {
        match host_family() {
            HostFamily::Msvc => {
                assert_eq!(host_relevant_environ(), &["PATH", "INCLUDE", "LIB"]);
            }
            _ => {
                assert_eq!(
                    host_relevant_environ(),
                    &["CPATH", "LIBRARY_PATH", "C_INCLUDE_PATH"]
                );
            }
        }
        let mingw = windows::mingw_platform(host_triple());
        assert_eq!(
            mingw.relevant_environ,
            &["CPATH", "LIBRARY_PATH", "C_INCLUDE_PATH"]
        );
    }

    #[test]
    fn host_family_is_not_linux_for_darwin() {
        #[cfg(target_os = "macos")]
        assert_eq!(host_family(), HostFamily::Darwin);
        #[cfg(target_os = "linux")]
        assert_eq!(host_family(), HostFamily::Linux);
        #[cfg(target_os = "freebsd")]
        assert_eq!(host_family(), HostFamily::Freebsd);
        #[cfg(windows)]
        assert!(matches!(
            host_family(),
            HostFamily::Msvc | HostFamily::Mingw
        ));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn darwin_host_does_not_use_linux_cflags() {
        let plat = Platform::host();
        let compile = plat._compile_args_from_eci(&ExternalCompilationInfo::default(), true);
        assert!(
            compile
                .iter()
                .any(|flag| flag == "-mmacosx-version-min=10.13")
        );
        assert!(compile.iter().all(|flag| flag != "-Wall"));
        assert!(compile.iter().any(|flag| flag == "-mdynamic-no-pic"));
        let link = plat._link_args_from_eci(&ExternalCompilationInfo::default(), true);
        assert!(link.iter().all(|flag| flag != "-lrt"));
    }

    static COMPILE_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    fn unique(label: &str) -> String {
        format!(
            "plat-{label}-{}-{}",
            std::process::id(),
            COMPILE_SEQ.fetch_add(1, Ordering::Relaxed)
        )
    }

    #[test]
    fn compiler_exists_reads_snapshot_path() {
        let dir = crate::tool::udir::udir().join(unique("cc-snap"));
        fs::create_dir_all(&dir).unwrap();
        let stem = unique("probe-cc");
        let exe_name = if cfg!(windows) {
            format!("{stem}.exe")
        } else {
            stem
        };
        let exe = dir.join(&exe_name);
        fs::write(&exe, b"").unwrap();
        let snapshot = [(OsString::from("PATH"), dir.as_os_str().to_os_string())];
        assert!(
            compiler_exists(Path::new(&exe_name), Some(&snapshot)),
            "snapshot PATH must find {}",
            exe.display()
        );
        let empty = [(OsString::from("PATH"), OsString::from(""))];
        assert!(
            !compiler_exists(Path::new(&exe_name), Some(&empty)),
            "empty snapshot PATH must not find {}",
            exe.display()
        );
    }

    #[test]
    fn resolve_compiler_on_path_does_not_search_path_for_dir_component() {
        let dir = crate::tool::udir::udir().join(unique("cc-dir-comp"));
        fs::create_dir_all(&dir).unwrap();
        let exe = dir.join("cc-name");
        fs::write(&exe, b"").unwrap();
        if cfg!(windows) {
            fs::write(dir.join("cc-name.exe"), b"").unwrap();
        }
        let snapshot = [(OsString::from("PATH"), dir.as_os_str().to_os_string())];
        assert_eq!(
            resolve_compiler_on_path(Path::new("missing-dir/cc-name"), Some(&snapshot)),
            None
        );
    }

    #[test]
    fn shell_command_search_keeps_exact_name() {
        let dir = crate::tool::udir::udir().join(unique("shell-exact"));
        fs::create_dir_all(&dir).unwrap();
        let cl = dir.join("cl.exe");
        fs::write(&cl, b"").unwrap();
        fs::write(dir.join("cl.exe.exe"), b"").unwrap();
        assert_eq!(
            shell_command_search(&cl, OsStr::new(""), OsStr::new(".COM;.EXE")),
            Some(cl)
        );
    }

    #[test]
    fn shell_command_search_appends_pathext_in_order() {
        let dir = crate::tool::udir::udir().join(unique("shell-pathext"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("tool.BAT"), b"").unwrap();
        fs::write(dir.join("tool.EXE"), b"").unwrap();
        let pathext = OsStr::new(".COM;.EXE;.BAT");
        assert_eq!(
            shell_command_search(&dir.join("tool"), OsStr::new(""), pathext),
            Some(dir.join("tool.EXE"))
        );

        let dir2 = crate::tool::udir::udir().join(unique("shell-pathext-bare"));
        fs::create_dir_all(&dir2).unwrap();
        fs::write(dir2.join("tool"), b"").unwrap();
        assert_eq!(
            shell_command_search(&dir2.join("tool"), OsStr::new(""), pathext),
            None
        );
    }

    #[test]
    fn shell_command_search_bare_name_on_path() {
        let dir = crate::tool::udir::udir().join(unique("shell-path"));
        fs::create_dir_all(&dir).unwrap();
        let exe = dir.join("cc-bare.exe");
        fs::write(&exe, b"").unwrap();
        let pathext = OsStr::new(".COM;.EXE");
        assert_eq!(
            shell_command_search(Path::new("cc-bare.exe"), dir.as_os_str(), pathext),
            Some(exe.clone())
        );

        let missing = dir.join("missing-path-entry");
        assert!(!missing.exists());
        let mut path_var = missing.into_os_string();
        path_var.push(";");
        path_var.push(dir.as_os_str());
        assert_eq!(
            shell_command_search(Path::new("cc-bare.exe"), &path_var, pathext),
            Some(exe)
        );
    }

    #[test]
    fn shell_command_search_dir_component_does_not_search_path() {
        let dir = crate::tool::udir::udir().join(unique("shell-dir-comp"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("cc-x.exe"), b"").unwrap();
        assert_eq!(
            shell_command_search(
                Path::new("missing-dir/cc-x.exe"),
                dir.as_os_str(),
                OsStr::new(".COM;.EXE")
            ),
            None
        );
    }

    #[test]
    fn shell_path_entries_drops_quotes_and_splits_outside() {
        assert_eq!(
            shell_path_entries(OsStr::new("\"C:\\tools\";C:\\Windows")),
            vec![OsString::from("C:\\tools"), OsString::from("C:\\Windows")]
        );
        assert_eq!(
            shell_path_entries(OsStr::new("\"a;b\";c;;")),
            vec![OsString::from("a;b"), OsString::from("c")]
        );
    }

    #[test]
    fn shell_command_search_quoted_path_entry() {
        let dir = crate::tool::udir::udir().join(unique("shell-quoted"));
        fs::create_dir_all(&dir).unwrap();
        let exe = dir.join("cc-q.exe");
        fs::write(&exe, b"").unwrap();
        let mut path_var = OsString::from("\"");
        path_var.push(dir.as_os_str());
        path_var.push("\"");
        assert_eq!(
            shell_command_search(Path::new("cc-q.exe"), &path_var, OsStr::new(".EXE")),
            Some(exe)
        );
    }

    fn write_udir(rel: &str, contents: &str) -> PathBuf {
        let path = crate::tool::udir::udir().join(rel);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&path, contents).unwrap();
        path
    }

    fn silent_host() -> Platform {
        let plat = Platform::host();
        plat.log_errors.store(false, Ordering::Relaxed);
        plat
    }

    #[test]
    fn compile_sources_in_different_directories() {
        let plat = silent_host();
        let prefix = unique("dirs");
        let main = write_udir(
            &format!("{prefix}/sub/main.c"),
            r#"
            int func(void);
            int main(void) {
                return func();
            }
            "#,
        );
        let other = write_udir(
            &format!("{prefix}/elsewhere/func.c"),
            r#"
            int func(void) {
                return 0;
            }
            "#,
        );
        let exe = plat
            .compile(
                &[main, other],
                &ExternalCompilationInfo::default(),
                None,
                true,
            )
            .expect("compile sources in different directories");
        let result = plat.execute(&exe, None, None, None).expect("execute");
        assert_eq!(
            result.returncode, 0,
            "out={} err={}",
            result.out, result.err
        );
    }

    #[test]
    fn compile_args_from_eci_emits_include_dirs() {
        let plat = silent_host();
        let inc = crate::tool::udir::udir().join(unique("inc"));
        let eci = ExternalCompilationInfo {
            include_dirs: vec![inc.clone()],
            ..ExternalCompilationInfo::default()
        };
        let args = plat._compile_args_from_eci(&eci, true);
        let expected = prefixed_path(if plat.is_msvc() { "/I" } else { "-I" }, &inc);
        assert!(
            args.iter().any(|arg| arg == &expected),
            "expected {expected:?} in {args:?}"
        );
    }

    #[test]
    fn compile_uses_eci_include_dirs() {
        let plat = silent_host();
        let prefix = unique("eci");
        let inc = crate::tool::udir::udir().join(format!("{prefix}-inc"));
        fs::create_dir_all(&inc).unwrap();
        fs::write(inc.join("needed.h"), "#define SOMEHASHDEFINE 42\n").unwrap();
        let cfile = write_udir(
            &format!("{prefix}-use.c"),
            r#"
            #include <stdio.h>
            #include "needed.h"
            int main(void) {
                printf("%d\n", SOMEHASHDEFINE);
                return 0;
            }
            "#,
        );
        let eci = ExternalCompilationInfo {
            include_dirs: vec![inc],
            ..ExternalCompilationInfo::default()
        };
        let exe = plat
            .compile(&[cfile], &eci, None, true)
            .expect("compile with include_dirs");
        let result = plat.execute(&exe, None, None, None).expect("execute");
        assert_eq!(result.returncode, 0);
        assert_eq!(result.out, "42\n");
    }

    #[test]
    fn compile_failure_writes_errors_file() {
        let plat = silent_host();
        let cfile = write_udir(
            &format!("{}.c", unique("errors")),
            "#error platform-compile-fail\n",
        );
        let err = plat
            .compile(
                std::slice::from_ref(&cfile),
                &ExternalCompilationInfo::default(),
                None,
                true,
            )
            .expect_err("compile must fail");
        let errorfile = cfile.with_extension("errors");
        assert!(
            errorfile.exists(),
            "expected {} from _handle_error",
            errorfile.display()
        );
        let written = fs::read_to_string(&errorfile).unwrap_or_default();
        // CompilationError.__init__ stores err.replace('\r\n', '\n').
        assert_eq!(written.replace("\r\n", "\n"), err.err);
        assert!(
            written.contains("platform-compile-fail") || err.out.contains("platform-compile-fail"),
            "compiler diagnostic missing: out={:?} err={:?}",
            err.out,
            err.err
        );
    }

    #[test]
    fn compile_honours_outputfilename_and_standalone() {
        let plat = silent_host();
        let cfile = write_udir(
            &format!("{}.c", unique("named")),
            "int main(void) { return 0; }\n",
        );
        let exe = plat
            .compile(
                std::slice::from_ref(&cfile),
                &ExternalCompilationInfo::default(),
                Some("custom_out"),
                true,
            )
            .expect("named standalone");
        assert_eq!(exe.file_stem().unwrap(), "custom_out");
        assert!(exe.exists(), "expected {}", exe.display());

        let lib_src = write_udir(
            &format!("{}.c", unique("shared")),
            "int func(void) { return 1; }\n",
        );
        let dylib = plat
            .compile(
                std::slice::from_ref(&lib_src),
                &ExternalCompilationInfo::default(),
                Some("custom_lib"),
                false,
            )
            .expect("named shared");
        assert_eq!(dylib.file_stem().unwrap(), "custom_lib");
        if !plat.so_ext.is_empty() {
            assert_eq!(dylib.extension().unwrap(), plat.so_ext);
        }
        assert!(dylib.exists(), "expected {}", dylib.display());
    }

    #[test]
    fn link_args_include_eci_link_files() {
        let plat = silent_host();
        let eci = ExternalCompilationInfo {
            link_files: vec![PathBuf::from("/foo/bar.a")],
            link_extra: vec!["-Wl,-dead_strip".to_string()],
            ..ExternalCompilationInfo::default()
        };
        let args = plat._link_args_from_eci(&eci, true);
        let rendered: Vec<String> = args
            .iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();
        assert!(
            rendered.iter().any(|arg| arg == "/foo/bar.a"),
            "link_files missing from {rendered:?}"
        );
        let link_file_at = rendered.iter().position(|arg| arg == "/foo/bar.a").unwrap();
        let extra_at = rendered
            .iter()
            .position(|arg| arg == "-Wl,-dead_strip")
            .unwrap();
        assert!(
            link_file_at < extra_at,
            "link_files before link_extra: {rendered:?}"
        );
    }

    #[test]
    fn cpp_link_drops_cc_extra_words() {
        let plat = host_platform_with_cc(Some("g++ -m32".into()));
        plat.log_errors.store(false, Ordering::Relaxed);
        if plat.is_msvc() {
            return;
        }
        let cc = plat.cc().expect("g++ must exist for this test");
        let compile_cmd = plat.spawn_compiler(cc.path());
        assert!(
            compile_cmd.get_args().any(|arg| arg == "-m32"),
            "compile keeps CC extra words: {:?}",
            compile_cmd.get_args().collect::<Vec<_>>()
        );
        let eci = ExternalCompilationInfo {
            use_cpp_linker: true,
            ..ExternalCompilationInfo::default()
        };
        let link_cmd = plat.link_command(&plat.cc_link_program(&eci), true);
        assert!(
            link_cmd.get_args().all(|arg| arg != "-m32"),
            "use_cpp_linker link drops CC extra words: {:?}",
            link_cmd.get_args().collect::<Vec<_>>()
        );
    }

    #[test]
    fn cpp_linker_is_gpp_on_every_family() {
        for family in [
            HostFamily::Linux,
            HostFamily::Darwin,
            HostFamily::Freebsd,
            HostFamily::Openbsd,
            HostFamily::Netbsd,
            HostFamily::Msvc,
            HostFamily::Mingw,
        ] {
            let plat = platform_for_family(family, host_triple());
            assert_eq!(plat.cpp_linker_name(), "g++", "{family:?}");
        }
        let plat = silent_host();
        let eci = ExternalCompilationInfo {
            use_cpp_linker: true,
            ..ExternalCompilationInfo::default()
        };
        if plat.is_msvc() {
            let inv = plat.link_invocation(
                &plat.cc_link_program(&eci),
                &[PathBuf::from("a.obj")],
                &[],
                true,
                Path::new("a.exe"),
            );
            assert_eq!(inv.program.file_name().unwrap(), "link.exe");
        } else {
            assert_eq!(
                plat.cc_link_program(&eci),
                PathBuf::from(plat.cpp_linker_name())
            );
        }
    }

    #[test]
    fn pypy_localbase_presence_includes_empty_value() {
        let dirs = vec![PathBuf::from("/app/include")];
        assert_eq!(
            prepend_pypy_localbase_from(dirs.clone(), "include", None),
            dirs
        );
        let empty = prepend_pypy_localbase_from(dirs.clone(), "include", Some(OsString::new()));
        assert_eq!(empty[0], PathBuf::from("/include"));
        assert_eq!(empty[1], PathBuf::from("/app/include"));
        let set =
            prepend_pypy_localbase_from(dirs.clone(), "lib", Some(OsString::from("/foo/baz")));
        assert_eq!(set[0], PathBuf::from("/foo/baz/lib"));
        assert_eq!(set[1], PathBuf::from("/app/include"));
    }

    #[test]
    fn is_asm_source_is_case_insensitive() {
        assert!(is_asm_source(Path::new("foo.asm")));
        assert!(is_asm_source(Path::new("foo.ASM")));
        assert!(is_asm_source(Path::new("foo.Asm")));
        assert!(!is_asm_source(Path::new("foo.c")));
        assert!(!is_asm_source(Path::new("foo.s")));
    }

    #[test]
    fn host_construction_does_not_require_a_compiler() {
        let plat = Platform::host();
        assert!(!plat.name.is_empty());
        let _ = plat.relevant_environ;
    }

    #[test]
    fn default_cc_matches_upstream_class_attribute() {
        assert_eq!(default_cc(HostFamily::Linux), "gcc");
        assert_eq!(default_cc(HostFamily::Darwin), "clang");
        assert_eq!(default_cc(HostFamily::Freebsd), "clang");
        assert_eq!(default_cc(HostFamily::Openbsd), "cc");
        assert_eq!(default_cc(HostFamily::Netbsd), "gcc");
        assert_eq!(default_cc(HostFamily::Mingw), "gcc");
        assert_eq!(default_cc(HostFamily::Msvc), "cl.exe");
    }

    #[cfg(not(windows))]
    #[test]
    fn host_platform_with_cc_none_applies_default_cc() {
        let plat = host_platform_with_cc(None);
        assert_eq!(plat.cc_command.as_deref(), Some(default_cc(host_family())));
    }

    #[test]
    fn host_platform_with_cc_keeps_present_empty_string() {
        let plat = host_platform_with_cc(Some(String::new()));
        if plat.is_msvc() {
            assert_eq!(plat.cc_command.as_deref(), Some("cl.exe"));
            assert_eq!(plat.family, HostFamily::Msvc);
        } else {
            assert_eq!(plat.cc_command.as_deref(), Some(""));
        }
    }

    #[test]
    fn pick_platform_empty_cc_matches_none() {
        let _guard = CC_COMMAND_LOCK.lock();
        let empty = pick_platform("host", Some(String::new())).unwrap();
        let none = pick_platform("host", None).unwrap();
        assert_eq!(empty.cc_command, none.cc_command);
        assert_eq!(empty, none);
    }

    #[test]
    fn snapshot_cc_command_keeps_empty_env() {
        let _guard = CC_COMMAND_LOCK.lock();
        let prior = std::env::var_os("CC");
        unsafe {
            std::env::set_var("CC", "");
        }
        let snap = snapshot_cc_command();
        let plat = host_platform();
        unsafe {
            match prior {
                Some(value) => std::env::set_var("CC", value),
                None => std::env::remove_var("CC"),
            }
        }
        assert_eq!(snap.as_deref(), Some(""));
        if plat.is_msvc() {
            assert_eq!(plat.cc_command.as_deref(), Some("cl.exe"));
            assert_eq!(plat.family, HostFamily::Msvc);
        } else {
            assert_eq!(plat.cc_command.as_deref(), Some(""));
        }
    }

    #[cfg(unix)]
    #[test]
    fn host_with_missing_cc_constructs_and_execute_works() {
        let plat = {
            let _guard = CC_COMMAND_LOCK.lock();
            let prior = std::env::var("CC").ok();
            unsafe {
                std::env::set_var("CC", "/no/such/majit-platform-cc");
            }
            let plat = Platform::host();
            unsafe {
                match prior {
                    Some(value) => std::env::set_var("CC", value),
                    None => std::env::remove_var("CC"),
                }
            }
            plat
        };
        let result = plat
            .execute("/bin/echo", Some(&["hello".to_string()]), None, None)
            .expect("execute does not need a C compiler");
        assert_eq!(result.returncode, 0);
        assert_eq!(result.out, "hello\n");

        let key = plat.key();
        assert!(
            key.contains("cc:<"),
            "discovery failure is encoded in key: {key}"
        );

        let src = write_udir(
            &format!("{}.c", unique("missing-cc")),
            "int main(void) { return 0; }\n",
        );
        let err = plat
            .compile(
                std::slice::from_ref(&src),
                &ExternalCompilationInfo::default(),
                None,
                true,
            )
            .expect_err("compile must surface discovery failure");
        assert!(
            !err.err.is_empty() || !err.out.is_empty(),
            "expected CompilationError, got {err:?}"
        );
    }

    #[test]
    fn platform_eq_ignores_lazy_cc_cache() {
        let a = Platform::host();
        let b = Platform::host();
        assert_eq!(a, b);
        let _ = a.key();
        assert_eq!(a, b);
        let c = host_platform_with_cc(Some("majit-platform-eq-other-cc".into()));
        assert_ne!(a, c);
    }

    #[test]
    fn platform_eq_c_environ_ignores_order() {
        let mut a = windows::msvc_platform(host_triple(), true, None);
        let mut b = windows::msvc_platform(host_triple(), true, None);
        a.c_environ = Some(vec![
            (OsString::from("PATH"), OsString::from("a")),
            (OsString::from("INCLUDE"), OsString::from("b")),
        ]);
        b.c_environ = Some(vec![
            (OsString::from("INCLUDE"), OsString::from("b")),
            (OsString::from("PATH"), OsString::from("a")),
        ]);
        assert_eq!(a, b);
        b.c_environ = Some(vec![
            (OsString::from("INCLUDE"), OsString::from("b")),
            (OsString::from("PATH"), OsString::from("other")),
        ]);
        assert_ne!(a, b);
    }

    #[test]
    fn c_environ_eq_windows_keys_are_case_insensitive() {
        let left = Some(vec![(OsString::from("Path"), OsString::from("x"))]);
        let right = Some(vec![(OsString::from("PATH"), OsString::from("x"))]);
        assert_eq!(c_environ_eq(&left, &right), cfg!(windows));
    }

    #[test]
    fn get_compiler_type_matches_windows() {
        assert_eq!(_get_compiler_type(None), HostFamily::Msvc);
        assert_eq!(_get_compiler_type(Some("")), HostFamily::Msvc);
        assert_eq!(_get_compiler_type(Some("gcc")), HostFamily::Mingw);
        assert_eq!(_get_compiler_type(Some("mingw32-gcc")), HostFamily::Mingw);
        assert_eq!(_get_compiler_type(Some("cl.exe")), HostFamily::Msvc);
        assert_eq!(_get_compiler_type(Some("gcc-12")), HostFamily::Msvc);
    }

    #[test]
    fn set_platform_host_binds_platform_and_host() {
        let _guard = CC_COMMAND_LOCK.lock();
        set_platform("host", None).expect("set_platform host");
        assert_eq!(*platform(), *host());
        assert!(is_host_build());
        set_platform("host", None).expect("restore host platform");
    }

    #[test]
    fn cc_command_extra_args_are_passed_to_compiler() {
        let host = silent_host();
        let Ok(cc) = host.cc() else {
            return;
        };
        let program = cc.path().display().to_string();
        // `_execute_c_compiler` splits `cc` on whitespace (`cc.split()`), so a
        // compiler path containing whitespace cannot be spelled in `cc`.
        if program.split_whitespace().nth(1).is_some() {
            return;
        }
        let plat = host_platform_with_cc(Some(format!("{program} -DPYRE_CC_ARG_TEST=1")));
        plat.log_errors.store(false, Ordering::Relaxed);
        let cfile = write_udir(
            &format!("{}.c", unique("cc-arg")),
            "#ifndef PYRE_CC_ARG_TEST\n#error missing\n#endif\nint main(void){return 0;}\n",
        );
        plat.compile(
            std::slice::from_ref(&cfile),
            &ExternalCompilationInfo::default(),
            None,
            true,
        )
        .expect("CC extra args must reach the compiler");
    }

    #[test]
    fn finish_linking_appends_so_ext_to_outputfilename() {
        let plat = silent_host();
        let cfile = write_udir(
            &format!("{}.c", unique("debug-so")),
            "int func(void) { return 1; }\n",
        );
        let dylib = plat
            .compile(
                std::slice::from_ref(&cfile),
                &ExternalCompilationInfo::default(),
                Some("probe.debug"),
                false,
            )
            .expect("shared probe.debug");
        let expected_suffix = format!("probe.debug.{}", plat.so_ext);
        let name = dylib.file_name().unwrap().to_string_lossy();
        assert!(
            name.ends_with(&expected_suffix),
            "expected name to end with {expected_suffix}, got {name}"
        );
        assert!(dylib.exists(), "expected {}", dylib.display());
    }

    #[cfg(not(windows))]
    #[test]
    fn lexical_normpath_keeps_exactly_two_leading_slashes() {
        assert_eq!(
            lexical_normpath(Path::new("//server/src/../a.c")),
            PathBuf::from("//server/a.c")
        );
        assert_eq!(
            lexical_normpath(Path::new("///x/./y")),
            PathBuf::from("/x/y")
        );
        assert_eq!(
            lexical_normpath(Path::new("/a/./b/../c")),
            PathBuf::from("/a/c")
        );
    }

    #[cfg(unix)]
    #[test]
    fn lexical_normpath_keeps_non_utf8_bytes_with_double_slash() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;
        assert_eq!(
            lexical_normpath(Path::new(OsStr::from_bytes(b"//srv/\xff/../a.c")))
                .as_os_str()
                .as_bytes(),
            b"//srv/a.c"
        );
        assert_eq!(
            lexical_normpath(Path::new(OsStr::from_bytes(b"//srv/\xff/a.c")))
                .as_os_str()
                .as_bytes(),
            b"//srv/\xff/a.c"
        );
    }

    #[test]
    fn all_cfiles_normalises_lexical_duplicates() {
        let plat = Platform::host();
        let files = plat._all_cfiles(
            &[
                Path::new("src.c"),
                Path::new("dir/../src.c"),
                Path::new("./src.c"),
            ],
            &ExternalCompilationInfo::default(),
        );
        assert_eq!(files.len(), 1);
    }

    #[test]
    fn mingw_handle_error_appends_stdout_after_stderr() {
        let plat = windows::mingw_platform(host_triple());
        let dir = crate::tool::udir::udir().join(unique("mingw-err"));
        fs::create_dir_all(&dir).unwrap();
        let outname = dir.join("out.o");
        let err = plat
            ._handle_error(1, b"OUT".to_vec(), b"ERR".to_vec(), &outname)
            .expect_err("nonzero returncode");
        assert_eq!(err.out, "");
        assert_eq!(err.err, "ERROUT");
        let written = fs::read_to_string(outname.with_extension("errors")).unwrap();
        assert_eq!(written, "ERROUT");
    }

    #[test]
    #[should_panic(expected = ".errors")]
    fn handle_error_write_failure_names_errors_file() {
        let plat = windows::mingw_platform(host_triple());
        let outname = crate::tool::udir::udir()
            .join(unique("missing-dir"))
            .join("nested")
            .join("out.o");
        let _ = plat._handle_error(1, b"OUT".to_vec(), b"ERR".to_vec(), &outname);
    }

    #[test]
    fn pick_platform_bogus_errors() {
        let err = pick_platform("bogus", None).expect_err("bogus platform");
        assert!(
            err.contains("platform = bogus"),
            "expected platform = bogus in {err}"
        );
    }
}
