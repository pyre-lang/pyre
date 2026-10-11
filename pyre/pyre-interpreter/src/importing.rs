//! Module importing — PyPy equivalent: pypy/module/imp/importing.py
//!
//! Implements the import machinery:
//! - `importhook()` — main entry point (called by IMPORT_NAME opcode)
//! - `find_module()` — locate a .py file on sys.path
//! - `load_source_module()` — compile and execute a .py file
//! - `check_sys_modules()` — consult the module cache
//! - `import_all_from()` — IMPORT_STAR handler

use parking_lot::Mutex;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{
    Arc, LazyLock, OnceLock,
    atomic::{AtomicBool, AtomicI64, AtomicUsize, Ordering},
};
// `Path` and `PathBuf` both appear in the host_env-independent module-search
// surface — `SYS_PATH`, `find_module`, `parent_package_path`, `load_part`,
// and `init_sys_path`, which every host calls — so both stay in scope
// unconditionally.
use std::path::Path;
// `normalize_lexically` walks a path a component at a time to collapse `.`
// and `..`; it is reached only from the executable-discovery arm.
#[cfg(all(
    feature = "host_env",
    not(feature = "sandbox"),
    not(target_arch = "wasm32")
))]
use std::path::Component;

use crate::PyExecutionContext;
use crate::{CodeObject, CompileOpts, Mode, PyFrame, compile_source_with_filename};
use pyre_object::*;
use rustpython_wtf8::{Wtf8, Wtf8Buf};

/// Module-local re-export of the host-OS surface.  Routes through
/// `rustpython_host_env` when the `host_env` feature is enabled; when
/// disabled the same names fall back to `std::*` shims so call sites
/// stay uniform.
#[cfg(feature = "host_env")]
pub mod host {
    #[cfg(not(target_arch = "wasm32"))]
    pub use rustpython_host_env::fs;
    pub mod os {
        pub use rustpython_host_env::os::*;
        // `replace` lives in the host layer's `posix`, which is only built for
        // unix, wasi and windows. Targets without it have no replace-specific
        // platform call, so the name degrades to `rename` exactly as the
        // unix-like body does.
        #[cfg(not(any(unix, windows, target_os = "wasi")))]
        pub use rustpython_host_env::os::rename as replace;
        #[cfg(any(unix, windows, target_os = "wasi"))]
        pub use rustpython_host_env::posix::replace;
    }
}
#[cfg(not(feature = "host_env"))]
pub mod host {
    pub mod fs {
        pub use std::fs::{metadata, read, read_dir, read_to_string, symlink_metadata};
    }
    pub mod os {
        pub fn current_dir() -> std::io::Result<std::path::PathBuf> {
            std::env::current_dir()
        }
        pub fn var(key: &str) -> Result<String, std::env::VarError> {
            std::env::var(key)
        }
        pub fn vars_os() -> std::env::VarsOs {
            std::env::vars_os()
        }
        pub fn process_id() -> u32 {
            std::process::id()
        }
        pub fn isatty(fd: i32) -> bool {
            // wasm32 has no descriptor table to ask, and no `libc::isatty` to
            // ask it with.
            #[cfg(target_arch = "wasm32")]
            {
                let _ = fd;
                return false;
            }
            #[cfg(not(target_arch = "wasm32"))]
            unsafe {
                libc::isatty(fd) != 0
            }
        }
        pub fn rename(
            from: impl AsRef<std::path::Path>,
            to: impl AsRef<std::path::Path>,
        ) -> std::io::Result<()> {
            std::fs::rename(from, to)
        }
        pub fn replace(
            from: impl AsRef<std::path::Path>,
            to: impl AsRef<std::path::Path>,
        ) -> std::io::Result<()> {
            std::fs::rename(from, to)
        }
        pub fn urandom(size: usize) -> std::io::Result<Vec<u8>> {
            use std::io::Read;
            let mut f = std::fs::File::open("/dev/urandom")?;
            let mut buf = vec![0u8; size];
            f.read_exact(&mut buf)?;
            Ok(buf)
        }
    }
}
#[cfg(all(not(target_arch = "wasm32"), not(feature = "sandbox")))]
use host::fs as host_fs;
use host::os as host_os;

// ── SourceProvider: the host-agnostic byte source for module loading ──
// PyPy/CPython read module source from a filesystem.  pyre routes the three
// FS touchpoints the import machinery actually exercises — the package/module
// `is_file` probes in `find_in_dirs` and the `read_to_string` in
// `load_source_module` — through one object, so the SAME import resolution
// runs over a real kernel FS (native, and the wasmtime runner via host
// imports) or an in-memory VFS (the browser/web build, populated from an
// embedded stdlib bundle).  The import machinery never branches per host;
// only the installed provider differs.
#[cfg(feature = "host_env")]
pub trait SourceProvider: Send + Sync {
    /// True when `path` names a readable regular file.
    fn is_file(&self, path: &Path) -> bool;
    /// True when `path` names a directory.
    fn is_dir(&self, path: &Path) -> bool;
    /// Read the whole file at `path` as raw bytes.  Source files carry a BOM
    /// or a PEP 263 cookie, so decoding belongs to the caller, not here.
    fn read_to_bytes(&self, path: &Path) -> std::io::Result<Vec<u8>>;
    /// Byte length of the regular file at `path`.
    ///
    /// Defaults to the length of a whole read, which every provider can
    /// answer; one that knows the size without transferring the bytes
    /// overrides it.
    fn file_size(&self, path: &Path) -> std::io::Result<u64> {
        Ok(self.read_to_bytes(path)?.len() as u64)
    }
    /// Entry names of the directory at `path`, in the order the seam reports
    /// them, each as the filesystem's own bytes.
    ///
    /// Names are bytes rather than text because a directory entry need not
    /// have a UTF-8 spelling and `listdir` still has to return it: the caller
    /// decides between `bytes` and a surrogateescape-decoded `str` from the
    /// type of the path it was given.
    ///
    /// Defaults to refusing: enumeration is the one filesystem question a
    /// provider built to answer `import`'s probes need not be able to answer,
    /// and `posix.listdir` turns the refusal into the OSError its caller
    /// expects rather than inventing an empty directory.
    fn list_dir(&self, path: &Path) -> std::io::Result<Vec<Vec<u8>>> {
        let _ = path;
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "this source provider cannot list a directory",
        ))
    }
    /// The directory a relative path is resolved against, as the filesystem's
    /// own bytes.
    ///
    /// Defaults to none: a provider serving an in-memory image has no working
    /// directory to report, and `posix.getcwd` says so rather than naming a
    /// directory that does not exist.
    fn cwd(&self) -> Option<Vec<u8>> {
        None
    }
    /// Whether the frozen marshal cache may be used for this provider.
    ///
    /// Only the host filesystem provider answers true: the cache file lives
    /// next to the stdlib sources that provider reads. A custom provider can
    /// serve different content at the same path, so the cache stays off.
    fn frozen_marshal_cache_ok(&self) -> bool {
        false
    }
}

#[cfg(feature = "host_env")]
static SOURCE_PROVIDER: LazyLock<Mutex<Option<Arc<dyn SourceProvider>>>> =
    LazyLock::new(|| Mutex::new(None));

/// Install the byte source the import machinery reads through.  The wasm
/// bootstrap installs a host-import-backed or in-memory-VFS provider before
/// the first import; native/pyrex leaves it unset and the default kernel-FS
/// provider answers every probe.
#[cfg(feature = "host_env")]
pub fn install_source_provider(provider: Arc<dyn SourceProvider>) {
    *SOURCE_PROVIDER.lock() = Some(provider);
}

/// Run `f` against the installed provider, lazily defaulting to the platform's
/// kernel-FS provider when none was installed. The `Arc` is cloned out before
/// `f` runs so the process-global lock is not held across the re-entrant import
/// path. PyPy owns this provider through its shared object space, not through
/// an execution context or OS thread.
#[cfg(feature = "host_env")]
pub(crate) fn with_source_provider<R>(f: impl FnOnce(&dyn SourceProvider) -> R) -> R {
    let provider = {
        let mut slot = SOURCE_PROVIDER.lock();
        slot.get_or_insert_with(default_source_provider).clone()
    };
    f(&*provider)
}

/// Read a source file through the installed [`SourceProvider`] — the
/// seam-mediated VFS under sandbox, the host FS otherwise. Traceback rendering
/// uses this so it honours the same jail as the import machinery instead of
/// reaching `std::fs` for a guest-controlled path.
#[cfg(feature = "host_env")]
pub fn read_source_bytes(path: &Path) -> std::io::Result<Vec<u8>> {
    with_source_provider(|p| p.read_to_bytes(path))
}

/// Whether the seam sees `path` as a directory, and the size it reports for a
/// regular file — the two probes `posix.stat` answers from on a target whose
/// only view of a filesystem is this seam.
#[cfg(feature = "host_env")]
pub fn source_is_dir(path: &Path) -> bool {
    with_source_provider(|p| p.is_dir(path))
}

#[cfg(feature = "host_env")]
pub fn source_file_size(path: &Path) -> std::io::Result<u64> {
    with_source_provider(|p| p.file_size(path))
}

#[cfg(feature = "host_env")]
pub fn source_list_dir(path: &Path) -> std::io::Result<Vec<Vec<u8>>> {
    with_source_provider(|p| p.list_dir(path))
}

#[cfg(feature = "host_env")]
pub fn source_cwd() -> Option<Vec<u8>> {
    with_source_provider(|p| p.cwd())
}

/// [`read_source_bytes`] decoded as plain UTF-8, for callers that hold a
/// text file rather than Python source.  Source readers decode through
/// `decode_source_bytes` instead so the BOM and the PEP 263 cookie apply.
#[cfg(feature = "host_env")]
pub fn read_source_to_string(path: &Path) -> std::io::Result<String> {
    let bytes = read_source_bytes(path)?;
    String::from_utf8(bytes).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "stream did not contain valid UTF-8",
        )
    })
}

#[cfg(all(
    feature = "host_env",
    not(target_arch = "wasm32"),
    not(feature = "sandbox")
))]
fn default_source_provider() -> Arc<dyn SourceProvider> {
    Arc::new(HostFsProvider)
}

#[cfg(all(feature = "host_env", not(target_arch = "wasm32"), feature = "sandbox"))]
fn default_source_provider() -> Arc<dyn SourceProvider> {
    Arc::new(SeamSourceProvider)
}

#[cfg(all(feature = "host_env", target_arch = "wasm32"))]
fn default_source_provider() -> Arc<dyn SourceProvider> {
    Arc::new(NullSourceProvider)
}

/// Kernel-filesystem provider — the default on native and the wasmtime
/// runner's real-FS path.  `is_file`/`is_dir` go straight to `std::fs::
/// metadata` via the `Path` methods (matching the historical `find_in_dirs`
/// probes); reads route through the host_env `fs` shim.
#[cfg(all(
    feature = "host_env",
    not(target_arch = "wasm32"),
    not(feature = "sandbox")
))]
struct HostFsProvider;

#[cfg(all(
    feature = "host_env",
    not(target_arch = "wasm32"),
    not(feature = "sandbox")
))]
impl SourceProvider for HostFsProvider {
    fn is_file(&self, path: &Path) -> bool {
        path.is_file()
    }
    fn is_dir(&self, path: &Path) -> bool {
        path.is_dir()
    }
    fn read_to_bytes(&self, path: &Path) -> std::io::Result<Vec<u8>> {
        host_fs::read(path)
    }
    fn file_size(&self, path: &Path) -> std::io::Result<u64> {
        host_fs::metadata(path).map(|meta| meta.len())
    }
    fn list_dir(&self, path: &Path) -> std::io::Result<Vec<Vec<u8>>> {
        host_fs::read_dir(path)?
            .map(|entry| entry.map(|entry| entry.file_name().as_encoded_bytes().to_vec()))
            .collect()
    }
    fn frozen_marshal_cache_ok(&self) -> bool {
        true
    }
}

/// Sandbox provider: every import probe and source read round-trips through the
/// host_seam trampoline to the trusted controller, which enforces the virtual
/// filesystem policy (read-only, path-jailed). Replaces `HostFsProvider` so the
/// import machinery cannot `std::fs` its way to an attacker-chosen host path
/// (e.g. `sys.path.append('/etc'); __import__('shadow')`).
#[cfg(all(feature = "host_env", feature = "sandbox"))]
struct SeamSourceProvider;

#[cfg(all(feature = "host_env", feature = "sandbox"))]
impl SeamSourceProvider {
    fn stat_mode(path: &Path) -> Option<u32> {
        use std::os::unix::ffi::OsStrExt;
        crate::host_seam::ops::stat(path.as_os_str().as_bytes())
            .ok()
            .map(|s| s.mode)
    }
}

#[cfg(all(feature = "host_env", feature = "sandbox"))]
impl SourceProvider for SeamSourceProvider {
    fn is_file(&self, path: &Path) -> bool {
        Self::stat_mode(path).is_some_and(|m| m & libc::S_IFMT as u32 == libc::S_IFREG as u32)
    }
    fn is_dir(&self, path: &Path) -> bool {
        Self::stat_mode(path).is_some_and(|m| m & libc::S_IFMT as u32 == libc::S_IFDIR as u32)
    }
    fn read_to_bytes(&self, path: &Path) -> std::io::Result<Vec<u8>> {
        use std::os::unix::ffi::OsStrExt;
        fn to_io(e: crate::host_seam::SeamError) -> std::io::Error {
            match e {
                crate::host_seam::SeamError::Os(errno) => std::io::Error::from_raw_os_error(errno),
                _ => std::io::Error::other("sandbox source read failed"),
            }
        }
        let bytes = path.as_os_str().as_bytes();
        let fd = crate::host_seam::ops::open(bytes, libc::O_RDONLY, 0).map_err(to_io)?;
        let mut data = Vec::new();
        loop {
            match crate::host_seam::ops::read(fd, 65536) {
                Ok(chunk) if chunk.is_empty() => break,
                Ok(chunk) => data.extend_from_slice(&chunk),
                Err(e) => {
                    let _ = crate::host_seam::ops::close(fd);
                    return Err(to_io(e));
                }
            }
        }
        let _ = crate::host_seam::ops::close(fd);
        Ok(data)
    }
}

/// Default provider on wasm before the bootstrap installs a real one: resolves
/// nothing, preserving the historical "builtins only" behaviour.
#[cfg(all(feature = "host_env", target_arch = "wasm32"))]
struct NullSourceProvider;

#[cfg(all(feature = "host_env", target_arch = "wasm32"))]
impl SourceProvider for NullSourceProvider {
    fn is_file(&self, _path: &Path) -> bool {
        false
    }
    fn is_dir(&self, _path: &Path) -> bool {
        false
    }
    fn read_to_bytes(&self, path: &Path) -> std::io::Result<Vec<u8>> {
        Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("no source provider installed: {}", path.display()),
        ))
    }
}

// ── embedded-stdlib VFS (wasm_vfs) ───────────────────────────────────
// The browser/web wasm target has no filesystem, so a playground-sized
// pure-Python stdlib closure (`re`, `json`, `datetime`, `pathlib`, …) is
// compiled into the binary (see pyre-native/build.rs `STDLIB_CLOSURE`)
// and served from this in-memory map.  Keys are `mount.join(<relpath>)`, so the
// SAME `find_in_dirs` probes (`<dir>/re/__init__.py`, `<dir>/enum.py`, …) that
// hit a real FS on native resolve here once `mount` is on sys.path.
#[cfg(feature = "wasm_vfs")]
pub static VFS_BLOB: &[u8] = pyre_native::vfs::STDLIB_BLOB;

#[cfg(feature = "wasm_vfs")]
enum VfsEntry {
    File(Arc<str>),
    Dir,
}

#[cfg(feature = "wasm_vfs")]
struct VfsProvider {
    map: HashMap<PathBuf, VfsEntry>,
}

#[cfg(feature = "wasm_vfs")]
impl VfsProvider {
    /// Decompress and parse the build-time blob into a `mount`-rooted map.
    /// Each embedded file becomes a `File` entry at `mount/<relpath>`, plus a
    /// synthetic `Dir` entry for every ancestor directory (so `is_dir` answers
    /// for `re/`, `collections/`, and the mount itself).
    fn from_blob(blob: &[u8], mount: &Path) -> Self {
        let files = pyre_native::vfs::unpack(blob).expect("wasm_vfs: corrupt embedded stdlib blob");
        let mut map: HashMap<PathBuf, VfsEntry> = HashMap::new();
        map.insert(mount.to_path_buf(), VfsEntry::Dir);

        for (name, source) in files {
            let full = mount.join(&name);
            // Register every ancestor directory under `mount` as a Dir.
            let mut ancestor = full.parent();
            while let Some(dir) = ancestor {
                if dir == mount || !dir.starts_with(mount) {
                    break;
                }
                map.entry(dir.to_path_buf()).or_insert(VfsEntry::Dir);
                ancestor = dir.parent();
            }
            map.insert(full, VfsEntry::File(Arc::from(source.as_str())));
        }
        VfsProvider { map }
    }
}

#[cfg(feature = "wasm_vfs")]
impl SourceProvider for VfsProvider {
    fn is_file(&self, path: &Path) -> bool {
        matches!(self.map.get(path), Some(VfsEntry::File(_)))
    }
    fn is_dir(&self, path: &Path) -> bool {
        matches!(self.map.get(path), Some(VfsEntry::Dir))
    }
    fn read_to_bytes(&self, path: &Path) -> std::io::Result<Vec<u8>> {
        match self.map.get(path) {
            Some(VfsEntry::File(src)) => Ok(src.as_bytes().to_vec()),
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("not in embedded stdlib: {}", path.display()),
            )),
        }
    }
}

/// Mount the embedded stdlib closure at `mount`, add `mount` to `sys.path`, and
/// install the VFS as the import source.  Called by the web wasm bootstrap
/// before the first import.
#[cfg(feature = "wasm_vfs")]
pub fn mount_embedded_stdlib(mount: &Path) {
    let provider = VfsProvider::from_blob(VFS_BLOB, mount);
    add_sys_path(mount);
    install_source_provider(Arc::new(provider));
}

// ── sys.modules cache ────────────────────────────────────────────────
// PyPy equivalent: `space.sys.get('modules')`, `space.sys.path`, and
// `space.builtin_modules` are object-space/process state, shared by every
// ExecutionContext.  Raw GC references use the established process-global
// `usize` representation; the mutex serializes semantic access and keeps
// foreign STW root walks well-defined.
static SYS_MODULES: LazyLock<Mutex<HashMap<String, usize>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
/// Every `Module` ever bound by [`set_sys_module`], keyed by address.
///
/// A builtin module is allocated immortal (`w_module_new_aliasing_dict` ->
/// `malloc_typed`), so the collector never traces it and its `w_dict` is
/// reachable only through [`walk_module_dicts_gc`].  Keying that walk on the
/// name→module map alone loses the dict the moment a second module takes the
/// name — `del sys.modules['_functools']` followed by a fresh `import`
/// mints a new module, and the collection after that frees the first
/// module's dict while Python still holds the module.  The module itself is
/// immortal and can never be freed, so its dict has to stay valid for as
/// long; entries are added and never removed.
static MODULE_DICT_ROOTS: LazyLock<Mutex<std::collections::HashSet<usize>>> =
    LazyLock::new(|| Mutex::new(std::collections::HashSet::new()));
static SYS_MODULES_DICT: AtomicUsize = AtomicUsize::new(0);
/// `baseobjspace.py:730` — `self.w_default_importlib_import =
/// frozen_importlib.w_import`: the interp-level `__import__` gateway, held on
/// the object space rather than read back out of a namespace, so a later
/// `builtins.__import__` rebind cannot reach the importer `_handle_fromlist`
/// is handed.  `moduledef.py:78-87 startup` puts the same object in builtins,
/// so this cell is filled where that copy is made.
///
/// The wrapper is allocated in the movable nursery like any other function
/// object. A mutator store into this cell takes `mark_prebuilt_roots_dirty`
/// (`incminimark.py remember_young_pointer_from_prebuilt`); the gated
/// `walk_process_import_roots` then forwards the raw copy.
static DEFAULT_IMPORTLIB_IMPORT: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "host_env")]
static SYS_PATH: LazyLock<Mutex<Vec<PathBuf>>> = LazyLock::new(|| Mutex::new(Vec::new()));
/// The directory prepended to `sys.path` at startup (`config->sys_path_0`):
/// the script's directory, or the cwd for `-c` / `-m`.  Captured once by
/// `init_sys_path`; read by the shadowing check, which must not see later
/// `sys.path` mutations.  PyPy records this on interpreter/import state, not
/// on an OS thread: import shadowing decisions made by free-threaded workers
/// must observe the startup path captured by the launcher.
static SYS_PATH_0: LazyLock<Mutex<Option<Wtf8Buf>>> = LazyLock::new(|| Mutex::new(None));
/// The literal `sys.path[0]` entry staged by `init_sys_path` and prepended
/// by `add_sys_path_0` once `site` has run (`pymain_sys_path_add_path0`):
/// `""` for `-c` / stdin / the REPL, the cwd for `-m`, the script's
/// directory for a script.  `None` under `-P`, and taken on first insert so
/// the `-i` REPL-after-script path does not prepend it twice.  Process-global
/// for the same reason as `SYS_PATH_0`: the launcher stages it and whichever
/// thread runs the insert must observe that staging.
static SYS_PATH_0_PENDING: LazyLock<Mutex<Option<std::ffi::OsString>>> =
    LazyLock::new(|| Mutex::new(None));
pub(crate) static BUILTIN_MODULES: LazyLock<Mutex<HashMap<&'static str, BuiltinModuleDef>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// The GC layout a module type declares for itself.
#[derive(Clone, Copy)]
pub enum ModuleGcLayout {
    /// The `#[pyre_class]` inline pointer offsets, optionally with the field
    /// that reports its native memory pressure.
    PyreClass {
        memory_pressure_offset: Option<usize>,
    },
    /// An `rclass.OBJECT` subclass whose payload holds no traced edge.
    Object,
    /// An `rclass.OBJECT` subclass walked by its own trace hook
    /// (`rgc.register_custom_trace_hook`).
    CustomTrace(majit_gc::trace::CustomTraceFn),
    /// Inline pointer offsets only. The type has no Python-visible vtable and
    /// no `pytype_to_tid` entry (`TypeInfo::with_gc_ptrs`).
    WithGcPtrs,
}

/// One GC type a module owns: its layout and, when it holds native state,
/// the sweep destructor that releases it (`_finalize_` / `__del__`).
#[derive(Clone, Copy)]
pub struct ModuleGcType {
    pub descriptor: &'static pyre_object::lltype::PyreClassDescriptor,
    pub layout: ModuleGcLayout,
    pub destructor: Option<majit_gc::trace::DestructorFn>,
}

/// One `ffi_type` of a `CIF_DESCRIPTION` as `jit_libffi.py` reads it:
/// `types.getkind`'s kind letter and the type's `size`.
#[derive(Clone, Copy)]
pub struct LibffiType {
    pub kind: u8,
    pub size: usize,
}

/// The call shape a `CIF_DESCRIPTION` block records: `rtype`, then each
/// `atypes[i]` with its `exchange_args[i]` offset.
pub struct LibffiCifShape {
    pub rtype: LibffiType,
    pub args: Vec<(LibffiType, usize)>,
}

/// Optional modules live in `pyre-module` so this crate does not depend on
/// them. The final binary links both and installs the hooks once before
/// [`install_builtin_modules`]. PyPy freezes the same surface at
/// translation (`ObjSpace.get_builtinmodule_to_install` /
/// `config.objspace.usemodules`); one write-once record is that boot
/// install, not four independent caches.
pub struct OptionalModuleHooks {
    pub install_modules: fn(),
    pub walk_global_roots: fn(&mut dyn FnMut(&mut majit_ir::GcRef)),
    pub walk_prebuilt_slots: fn(&mut dyn FnMut(&mut PyObjectRef)),
    /// Residual addresses whose functions live in `pyre-module` (`ll_math`
    /// hypot/atan2/…). `jit_trace_fnaddrs` appends these after the
    /// interpreter-owned table.
    pub publish_fnaddrs: fn(&mut Vec<(&'static str, i64)>),
    pub mini_buffer_params: fn(PyObjectRef) -> Option<(*mut u8, usize)>,
    /// `Some(calls_python)` when `obj` is a cdata flavor this path handles.
    pub cffi_finalizer_kind: fn(PyObjectRef) -> Option<bool>,
    /// `W_CData.call` receiver whose ctype is a function pointer.
    pub cdata_is_function_pointer: fn(PyObjectRef) -> bool,
    /// `W_CType.size?` watcher. `ptr` is the ctype. `None` without the module.
    pub ctype_size_qmut:
        fn(*const u8) -> Option<std::sync::Arc<pyre_object::quasiimmut::QuasiImmut>>,
    pub run_cffi_finalize: fn(PyObjectRef),
    pub close_cffi_fileobj: fn(PyObjectRef),
    pub load_cffi1_module: fn(&str, &std::path::Path, usize) -> Result<PyObjectRef, crate::PyError>,
    pub ctypes_buffer_view:
        fn(PyObjectRef) -> Option<(PyObjectRef, usize, usize, String, usize, Vec<usize>)>,
    pub ctypes_bytes_object: fn(PyObjectRef) -> Option<PyObjectRef>,
    pub ctypes_array_instance: fn(PyObjectRef) -> bool,
    pub ctypes_pointer_instance: fn(PyObjectRef) -> bool,
    /// The GC types the modules own, in registration order. `build_gc`
    /// numbers them from [`crate::MODULE_FIRST_TYPE_ID`], after every
    /// interpreter class.
    pub gc_types: fn() -> Vec<ModuleGcType>,
    /// Header-only `#[pyre_class]` types the modules allocate immortal; only
    /// the immortal-root walker learns their `w_class` edge.
    pub immortal_w_class_only_descriptors:
        fn() -> Vec<&'static pyre_object::lltype::PyreClassDescriptor>,
    /// Read the `CIF_DESCRIPTION` block at the given address; `None` when
    /// `_cffi_backend`, the only builder of such blocks, is absent.
    ///
    /// # Safety
    /// The address must be a block `_cffi_backend` built.
    pub libffi_cif_shape: unsafe fn(usize) -> Option<LibffiCifShape>,
    /// The name of the canonical `math` builtin a callable is, if any.
    pub math_builtin_name: fn(PyObjectRef) -> Option<&'static str>,
    /// `interp_gc.py`'s app-level hook object, installed once when the
    /// execution context binds the shared action flag.
    pub gc_initialize: fn(PyObjectRef, &mut (dyn crate::executioncontext::ActionFlagOps + 'static)),
    /// `interp_gc.py _run_finalizers`, with the enable/disable bracket the
    /// module keeps around `UserDelAction._run_finalizers`.
    pub gc_run_finalizers_now: fn(),
    /// `interp_array.py` type setup. `array.array` is installed only when
    /// this runs; without the module the type is absent.
    pub init_array_type: fn(PyObjectRef),
    /// `interp_array.py W_Array.w_getitem`: box one live array element.
    pub array_w_getitem: fn(PyObjectRef, usize, bool) -> crate::PyResult,
    /// `interp_array.py descr_repr`, shared with `display::py_repr_wtf8`.
    pub array_repr_wtf8: fn(PyObjectRef) -> Result<rustpython_wtf8::Wtf8Buf, crate::PyError>,
}

static OPTIONAL_MODULE_HOOKS: std::sync::OnceLock<OptionalModuleHooks> = std::sync::OnceLock::new();

/// Install the `pyre-module` hooks. Call once from the binary before
/// [`init_sys_path`] / [`install_builtin_modules`].
pub fn set_optional_module_hooks(hooks: OptionalModuleHooks) {
    let _ = OPTIONAL_MODULE_HOOKS.set(hooks);
}

pub fn optional_module_hooks() -> Option<&'static OptionalModuleHooks> {
    OPTIONAL_MODULE_HOOKS.get()
}

/// Function-pointer cdata, once `_cffi_backend` has installed its hook.
pub fn cdata_is_function_pointer(obj: PyObjectRef) -> bool {
    optional_module_hooks().is_some_and(|hooks| (hooks.cdata_is_function_pointer)(obj))
}

/// Current `W_CType.size?` watcher, once `_cffi_backend` has installed its hook.
pub fn ctype_size_qmut(
    ptr: *const u8,
) -> Option<std::sync::Arc<pyre_object::quasiimmut::QuasiImmut>> {
    optional_module_hooks().and_then(|hooks| (hooks.ctype_size_qmut)(ptr))
}

thread_local! {
    static IMPORT_ROOT_AREA: ImportRootArea = ImportRootArea {
        argv_pending: SYS_ARGV_PENDING.with(|p| p as *const _),
    };
}

#[derive(Clone, Copy)]
pub(crate) struct BuiltinModuleDef {
    init: fn(PyObjectRef) -> Result<(), crate::PyError>,
    startup: Option<fn(PyObjectRef, *const PyExecutionContext) -> Result<(), crate::PyError>>,
    /// The module follows ordinary `sys.modules` ownership and may be
    /// collected after a fresh import is removed from that cache. Most of
    /// pyre's legacy MixedModules still have JIT/native owners that require
    /// their historical immortal holder; opt in only after that module's
    /// state and cached references have been audited.
    collectible: bool,
    /// `space.builtin_modules[name]`: the module built the first time
    /// `getbuiltinmodule` names it, 0 before. A builtin module does not move,
    /// and `walk_process_import_roots` marks it.
    w_mod: usize,
}

struct ImportRootArea {
    /// The pending `sys.argv` list is reachable only from this cell between
    /// `set_sys_argv` and `take_pending_sys_argv`.
    argv_pending: *const std::cell::Cell<PyObjectRef>,
}

// ── builtin module registry ──────────────────────────────────────────
// PyPy equivalent: space.builtin_modules dict + MixedModule.interpleveldefs
//
// Lazy loading (MixedModule.buildloaders / getdictvalue,
// `mixedmodule.py`): PyPy defers two things — (1) creating a
// module's contents until the module is first imported, and (2)
// evaluating each interpleveldef/appleveldef until the corresponding
// attribute is first accessed.  Pyre achieves (1) directly: this
// registry stores `name → init` and `new_builtin_module` runs `init`
// on demand at first import, never at interpreter startup.  (2) has no
// counterpart and is deliberately not ported: pyre's interpleveldefs are
// compile-time `const` / function-pointer expressions (not interp-eval
// strings), so there is nothing expensive to defer per attribute — a
// per-attribute loader table would be a side-table with no upstream
// basis.  The import-triggered `init` IS the buildloaders equivalent.

/// Register a builtin module initializer.
///
/// PyPy equivalent: Module.install() → space.builtin_modules[name] = mod
pub fn register_builtin_module(
    name: &'static str,
    init: fn(PyObjectRef) -> Result<(), crate::PyError>,
) {
    BUILTIN_MODULES.lock().insert(
        name,
        BuiltinModuleDef {
            init,
            startup: None,
            collectible: false,
            w_mod: 0,
        },
    );
}

/// Register a builtin module whose holder has no native/JIT owner beyond the
/// ordinary Python object graph.
pub fn register_collectible_builtin_module(
    name: &'static str,
    init: fn(PyObjectRef) -> Result<(), crate::PyError>,
) {
    BUILTIN_MODULES.lock().insert(
        name,
        BuiltinModuleDef {
            init,
            startup: None,
            collectible: true,
            w_mod: 0,
        },
    );
}

/// Register a MixedModule initializer plus its `Module.startup` hook.
/// The hook runs after the new module is visible in `sys.modules`, matching
/// `getbuiltinmodule()` and allowing startup imports without a module cycle.
pub fn register_builtin_module_with_startup(
    name: &'static str,
    init: fn(PyObjectRef) -> Result<(), crate::PyError>,
    startup: fn(PyObjectRef, *const PyExecutionContext) -> Result<(), crate::PyError>,
) {
    BUILTIN_MODULES.lock().insert(
        name,
        BuiltinModuleDef {
            init,
            startup: Some(startup),
            collectible: false,
            w_mod: 0,
        },
    );
}

/// The registered builtin module names, for `sys.builtin_module_names`.
///
/// PyPy equivalent: `pypy/module/sys/state.py get_builtin_module_names`,
/// which likewise reads `space.builtin_modules` rather than a second list.
/// Reading the registry keeps the advertised set equal to the set `import`
/// can satisfy under every `cfg` combination, which a parallel list cannot:
/// the modules gated on `unix` / `wasm32` / `sandbox` differ per build.
/// Dotted keys (`importlib.machinery`, `__pypy__.builders`) are registry
/// entries for submodules, not top-level modules, so they are left out.
pub fn builtin_module_names() -> Vec<&'static str> {
    let mut names: Vec<&'static str> = BUILTIN_MODULES
        .lock()
        .keys()
        .copied()
        // Frozen importlib's BuiltinImporter consults this tuple before
        // invoking pyre's dotted-module hook.  Keep the PyPy package
        // submodules advertised so `from __pypy__ import bufferable` (and
        // the existing builders module) can reach their registered Mixed
        // Module initializers; unrelated dotted aliases remain internal.
        .filter(|name| !name.contains('.') || name.starts_with("__pypy__."))
        .collect();
    names.sort_unstable();
    names
}

/// Install all standard builtin modules.
///
/// Mirrors PyPy's `baseobjspace.make_builtins()` +
/// `install_mixedmodule()` walk of `objspace.usemodules`.  The
/// `pyre_install_module!` arms below give a per-line declarative shape:
///
/// * `name(module)`               — register `crate::module::module::init` under `"name"` (alias arm).
/// * `module`                     — `name` defaults to the module identifier.
/// * `name => path`               — explicit init function path.
///
/// This is an explicit hand-maintained list by design — the upstream
/// equivalent (`pypy/config/pypyoption.py` `essential_modules` /
/// `default_modules` / `working_modules`) is likewise an explicit set of
/// string literals with platform conditionals, not filesystem discovery.
/// Automatic discovery is intentionally not done: it could not express
/// the alias arms (`"builtins"` → `__builtin__`), explicit-path arms
/// (`importlib.machinery` → a non-default init fn), or the
/// `#[cfg(unix)]` gating that `resource` / `fcntl` / `syslog` require.
// PyPy `baseobjspace.py make_builtins` is `@not_rpython`: this table
// and each MixedModule constructor are assembled while the object space is
// initialised, before translated execution begins.
#[majit_macros::not_rpython]
pub fn install_builtin_modules() {
    macro_rules! pyre_install_module {
        // `module` — `register_builtin_module("module", crate::module::module::init)`.
        ($mod:ident) => {
            register_builtin_module(stringify!($mod), crate::module::$mod::init);
        };
        // `name(module)` — re-register `module::init` under a different name.
        ($name:literal ( $mod:ident )) => {
            register_builtin_module($name, crate::module::$mod::init);
        };
        // `name => path::to::fn` — explicit init fn.
        ($name:literal => $path:path) => {
            register_builtin_module($name, $path);
        };
    }

    // Core pyre modules backed by `interpleveldefs` tables.
    pyre_install_module!(time);
    pyre_install_module!(sys);
    // `moduledef.py applevel_name = '_operator'` — the interp-level table
    // is reachable only as `_operator`; `import operator` resolves to
    // `operator.py`, whose `from _operator import *` drops the underscore
    // names the table also carries.
    pyre_install_module!("_operator"(operator));
    pyre_install_module!("builtins"(__builtin__));
    pyre_install_module!(_io);
    pyre_install_module!(_sre);
    // `baseobjspace.py:finish` runs `atexit._run_exitfuncs` at shutdown;
    // `warnings` imports `_contextvars`, which imports `_immutables_map`;
    // `os` imports `errno`. `_opcode` is one of `essential_modules`, and a
    // JIT build always has `pypyjit`. The `BUILD_TEMPLATE` and
    // `BUILD_INTERPOLATION` opcodes import `_template`.
    pyre_install_module!(atexit);
    pyre_install_module!(_contextvars);
    pyre_install_module!(_immutables_map);
    pyre_install_module!(errno);
    pyre_install_module!(_opcode);
    pyre_install_module!(pypyjit);
    pyre_install_module!(_template);
    // `Modules/Setup.bootstrap`: the modules built into the interpreter.
    // Frozen importlib imports `_stat` while bootstrapping a sandbox that
    // mounts no stdlib files, so it is unconditional.
    pyre_install_module!(_abc);
    pyre_install_module!(_functools);
    pyre_install_module!(_stat);
    pyre_install_module!(_structseq);
    pyre_install_module!(_suggestions);
    pyre_install_module!(_symtable);
    pyre_install_module!(_typing);
    #[cfg(all(not(target_arch = "wasm32"), not(feature = "sandbox")))]
    pyre_install_module!(faulthandler);
    #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
    pyre_install_module!(pwd);

    // C-extension stubs required for stdlib import chains
    // (PyPy: pypy/module/* mixed modules).
    pyre_install_module!(_weakref);
    pyre_install_module!(_warnings);
    pyre_install_module!("_thread"(thread));
    pyre_install_module!(itertools);
    pyre_install_module!(_codecs);
    // PyPy `_codecs/moduledef.py Module.__init__` performs this beside
    // MixedModule installation, not inside the translated CodecState ctor.
    crate::module::_codecs::register_builtin_error_handlers();
    // moduledef.py: `applevel_name = os.name` installs the one posix module
    // under `os.name` — `"posix"` on a POSIX host, `"nt"` on Windows, where a
    // module literally named `posix` does not exist. os.py picks `os.name` and
    // the `path` module (posixpath vs ntpath) from which of the two names is in
    // `sys.builtin_module_names`.
    // wasm32's arm answers from the import source seam, which only `host_env`
    // builds; without one there is nothing for `posix` to report and the name
    // stays out of `sys.builtin_module_names` too.
    #[cfg(all(not(windows), any(not(target_arch = "wasm32"), feature = "host_env")))]
    pyre_install_module!(posix);
    #[cfg(windows)]
    pyre_install_module!("nt"(posix));
    // `_bootstrap_external` imports `winreg` unconditionally under
    // `sys.platform == "win32"`, so on Windows it belongs to the same
    // bootstrap set as `nt` and `_stat`: without it the import machinery — and
    // therefore `site` — never comes up, and the interpreter runs the whole
    // program through the native importer fallback instead. `winreg` is in
    // `sys.builtin_module_names` on both CPython 3.14 and PyPy 8.0.0.
    // `moduledef.py applevel_name = 'winreg'` over `pypy/module/_winreg`.
    #[cfg(windows)]
    pyre_install_module!("winreg"(_winreg));
    pyre_install_module!(_collections);
    pyre_install_module!(_ast);
    pyre_install_module!("_imp"(imp));

    // importlib package and its submodules load their real source from disk:
    // the package `__init__.py` binds `__import__`/`import_module`/… from the
    // frozen `_bootstrap`, and `machinery`/`abc`/`util` re-export the real
    // finders/loaders/spec classes out of `_frozen_importlib{,_external}`. The
    // frozen bootstrap modules already carry the full surface, so a native stub
    // would only inject placeholder `object` classes that shadow them.

    // __pypy__ package + builders submodule — the PyPy-only surface
    // pickle.py imports (identity_dict + builders.BytesBuilder).
    pyre_install_module!("__pypy__" => crate::module::__pypy__::init);
    pyre_install_module!("__pypy__.builders" => crate::module::__pypy__::builders::init);
    pyre_install_module!("__pypy__.bufferable" => crate::module::__pypy__::bufferable::init);

    // Host-access modules — arbitrary FFI (`_ctypes`), real signals.
    // `select`, `mmap`, `_socket`/`_ssl`, `pwd`/`grp`, `errno`, `_stat`,
    // `_pypy_generic_alias`, `_pickle`, `_random`, `_tokenize`, `array`, `gc`,
    // and the Windows host modules other than `winreg` live in `pyre-module`.
    // None of the host ones belong to the mediated ll_os/ll_time surface, so
    // the sandbox interpreter omits them entirely: `import _ctypes` then
    // raises ModuleNotFoundError, as in a build whose syscall code is absent.
    #[cfg(not(feature = "sandbox"))]
    {
        // `_signal` is a bootstrap module upstream: it is built on every
        // platform, and what a platform lacks it lacks entry point by entry
        // point.  A target with no kernel to route a number publishes the
        // handler table and `raise_signal`, and leaves out the itimers, the
        // sigset calls and `pause`.
        pyre_install_module!("_signal"(signal));
    }
    pyre_install_module!(_locale);
    register_collectible_builtin_module("_struct", crate::module::r#struct::init);
    pyre_install_module!(marshal);

    // Modules whose stdlib wrapper does `import X` + attribute access or
    // `from X import *` are deliberately NOT stubbed here: an empty stub
    // makes the `import` succeed and the later access raise AttributeError
    // (or silently bind nothing), which the wrapper's `try/except
    // ImportError` cannot recover from.  Leaving them unregistered lets the
    // pure-Python fallback take over: `_datetime` -> `_pydatetime`,
    // `_decimal` -> `_pydecimal`, `_asyncio` -> pure-Python asyncio.
    register_builtin_module("_types", crate::module::_types::init);
    register_builtin_module("_string", init_string_module);
    register_builtin_module("_tracemalloc", init_tracemalloc);
    register_builtin_module("_sysconfig", init_sysconfig_stub);
    // `_get_sysconfigdata` is reached only from `_init_posix`, which
    // `sysconfig` runs when `os.name == 'posix'`; a Windows run takes
    // `_init_non_posix` and never names this module.
    #[cfg(not(windows))]
    register_builtin_module("_sysconfigdata", init_sysconfigdata);
    if let Some(hooks) = optional_module_hooks() {
        (hooks.install_modules)();
    }
}

fn require_string_module_str(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let Some(&arg) = args.first() else {
        return Err(crate::PyError::type_error("expected str, got object"));
    };
    if !unsafe { pyre_object::is_str(arg) } {
        return Err(crate::PyError::type_error(format!(
            "expected str, got {}",
            crate::type_methods::arg_type_name(arg)
        )));
    }
    Ok(arg)
}

fn init_string_module(ns: PyObjectRef) -> Result<(), crate::PyError> {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    crate::__pyre_put_new!(
        ns_slot,
        "formatter_parser",
        crate::make_builtin_function("formatter_parser", |args| {
            use rustpython_common::format::{FormatPart, FormatString, FromTemplate};

            let arg = require_string_module_str(args)?;
            let body = unsafe { pyre_object::w_str_get_wtf8(arg) }.to_wtf8_buf();
            let parsed = FormatString::from_str(&body)
                .map_err(|_| crate::PyError::value_error("bad format string"))?;

            // Every tuple, and every string inside it, is a nursery object
            // and the next mint can collect, so they all sit on one bracket.
            let _roots = pyre_object::gc_roots::push_roots();
            let mut tuple_slots = Vec::new();
            let mut pending: Option<Wtf8Buf> = None;
            for part in parsed.format_parts {
                match part {
                    FormatPart::Literal(text) => pending = Some(text),
                    FormatPart::Field {
                        field_name,
                        conversion_spec,
                        format_spec,
                    } => {
                        let literal = pending.take().unwrap_or_default();
                        let lit = pyre_object::gc_roots::pin_root(
                            pyre_object::w_str_from_wtf8_managed(literal),
                        );
                        let field = pyre_object::gc_roots::pin_root(
                            pyre_object::w_str_from_wtf8_managed(field_name),
                        );
                        let spec = pyre_object::gc_roots::pin_root(
                            pyre_object::w_str_from_wtf8_managed(format_spec),
                        );
                        let conv = match conversion_spec {
                            Some(c) => pyre_object::gc_roots::pin_root(
                                pyre_object::w_str_new_managed(&c.to_char_lossy().to_string()),
                            ),
                            None => pyre_object::w_none(),
                        };
                        let slot = pyre_object::gc_roots::shadow_stack_len();
                        let _ = pyre_object::gc_roots::pin_root(pyre_object::w_tuple_new(vec![
                            lit, field, spec, conv,
                        ]));
                        tuple_slots.push(slot);
                    }
                }
            }
            if let Some(text) = pending {
                let lit =
                    pyre_object::gc_roots::pin_root(pyre_object::w_str_from_wtf8_managed(text));
                let slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(pyre_object::w_tuple_new(vec![
                    lit,
                    pyre_object::w_none(),
                    pyre_object::w_none(),
                    pyre_object::w_none(),
                ]));
                tuple_slots.push(slot);
            }
            Ok(pyre_object::w_list_new(
                tuple_slots
                    .into_iter()
                    .map(pyre_object::gc_roots::shadow_stack_get)
                    .collect(),
            ))
        })
    );
    crate::__pyre_put_new!(
        ns_slot,
        "formatter_field_name_split",
        crate::make_builtin_function("formatter_field_name_split", |args| {
            use rustpython_common::format::{FieldName, FieldNamePart, FieldType};

            let arg = require_string_module_str(args)?;
            let body = unsafe { pyre_object::w_str_get_wtf8(arg) }.to_wtf8_buf();
            let FieldName { field_type, parts } = FieldName::parse(&body)
                .map_err(|_| crate::PyError::value_error("bad field name"))?;

            let first = match field_type {
                FieldType::Auto => pyre_object::w_str_new(""),
                FieldType::Index(n) => pyre_object::w_int_new(n as i64),
                FieldType::Keyword(s) => pyre_object::w_str_from_wtf8_managed(s),
            };
            // `first` is a young managed str or int; pin it and every part
            // tuple on one bracket. Reload `first` for the result tuple.
            let _roots = pyre_object::gc_roots::push_roots();
            let first_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(first);
            let rest_list = {
                let mut rest_slots = Vec::new();
                for part in parts {
                    let (flag, value) = match part {
                        FieldNamePart::Attribute(s) => (
                            pyre_object::w_bool_from(true),
                            pyre_object::gc_roots::pin_root(pyre_object::w_str_from_wtf8_managed(
                                s,
                            )),
                        ),
                        FieldNamePart::Index(n) => (
                            pyre_object::w_bool_from(false),
                            pyre_object::gc_roots::pin_root(pyre_object::w_int_new(n as i64)),
                        ),
                        FieldNamePart::StringIndex(s) => (
                            pyre_object::w_bool_from(false),
                            pyre_object::gc_roots::pin_root(pyre_object::w_str_from_wtf8_managed(
                                s,
                            )),
                        ),
                    };
                    let slot = pyre_object::gc_roots::shadow_stack_len();
                    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_tuple_new(vec![
                        flag, value,
                    ]));
                    rest_slots.push(slot);
                }
                pyre_object::w_list_new(
                    rest_slots
                        .into_iter()
                        .map(pyre_object::gc_roots::shadow_stack_get)
                        .collect(),
                )
            };
            let rest_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(rest_list);
            Ok(pyre_object::w_tuple_new(vec![
                pyre_object::gc_roots::shadow_stack_get(first_slot),
                pyre_object::gc_roots::shadow_stack_get(rest_slot),
            ]))
        })
    );
    Ok(())
}

/// `_sysconfig` — `config_vars()`, the build variables that come from the
/// running binary rather than from a generated `_sysconfigdata_*` module.
///
/// `sysconfig._init_non_posix` SUBSCRIPTS `Py_GIL_DISABLED` and `Py_DEBUG` to
/// spell `ABIFLAGS`, so on Windows a missing key is a `KeyError` out of the
/// first `get_config_var` call rather than the `None` the `.get()` readers
/// take. Pyre runs its mutators without a global interpreter lock, so
/// `Py_GIL_DISABLED` is 1 and the derived `ABIFLAGS` is `t`; `Py_DEBUG` is 0.
///
/// `EXT_SUFFIX` and `SOABI` name pyre's own cpyext ABI, never CPython's ABI
/// tag. They are build metadata rather than a claim that an extension loader
/// exists: `packaging.tags` derives a wheel's ABI tag from `EXT_SUFFIX` and
/// fails outright when the key is absent, so every `pip install` needs it.
/// Both come from `extension_abi_suffix`, and the pair keeps the relation the
/// suffix is built from: `EXT_SUFFIX == "." + SOABI + shared-library
/// extension`. `_sysconfigdata` publishes the same keys for `_init_posix` but
/// spells `SOABI` shorter — see [`soabi_tag`].
fn init_sysconfig_stub(ns: PyObjectRef) -> Result<(), crate::PyError> {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ns = pyre_object::gc_roots::pin_root(ns);
    crate::__pyre_put_new!(
        ns_slot,
        "config_vars",
        crate::make_builtin_function("config_vars", |_| {
            // A `dict` header moves, and every store allocates the key, the
            // value, and the dict's own storage when it grows, so the word is
            // read back out of a root slot per store.  Call arguments evaluate
            // left to right, so the key and value are bound before that read.
            let roots = pyre_object::gc_roots::push_roots();
            let vars_slot = roots.base();
            let _ = roots.pin_root(pyre_object::w_dict_new());
            let so_ext = extension_abi_suffix();
            unsafe {
                for (name, value) in [("Py_DEBUG", 0), ("Py_GIL_DISABLED", 1)] {
                    // Key and value pins live across this one store. The dict
                    // stays on `roots`.
                    let _pyre_store_roots = pyre_object::gc_roots::push_roots();
                    let key_slot = pyre_object::gc_roots::shadow_stack_len();
                    let _ = roots.pin_root(pyre_object::w_str_new_managed(name));
                    let value_slot = pyre_object::gc_roots::shadow_stack_len();
                    let _ = roots.pin_root(pyre_object::w_int_new(value));
                    pyre_object::w_dict_store(
                        roots.get(vars_slot),
                        roots.get(key_slot),
                        roots.get(value_slot),
                    );
                }
                for (name, value) in [
                    ("SOABI", soabi_middle(&so_ext)),
                    ("EXT_SUFFIX", so_ext.clone()),
                ] {
                    let _pyre_store_roots = pyre_object::gc_roots::push_roots();
                    let key_slot = pyre_object::gc_roots::shadow_stack_len();
                    let _ = roots.pin_root(pyre_object::w_str_new_managed(name));
                    let val_slot = pyre_object::gc_roots::shadow_stack_len();
                    let _ = roots.pin_root(pyre_object::w_str_new_managed(&value));
                    pyre_object::w_dict_store(
                        roots.get(vars_slot),
                        roots.get(key_slot),
                        roots.get(val_slot),
                    );
                }
            }
            Ok(roots.get(vars_slot))
        })
    );
    Ok(())
}

/// The whole middle component of the extension suffix — PEP 3149's "ABI
/// tag"-"platform tag" together, the spelling `_sysconfig.config_vars()` uses
/// so that `EXT_SUFFIX` stays `"." + SOABI + shared-library extension`.
fn soabi_middle(so_ext: &str) -> String {
    so_ext.split('.').nth(1).unwrap_or_default().to_string()
}

/// The `SOABI` `_sysconfigdata` publishes: the ABI half of the extension
/// suffix alone, without the platform tag (`_sysconfigdata.py:19`).
///
/// wheel 0.34.2 depends on the shorter form, so do not widen it without
/// checking wheel — `pep425tags.get_abi_tag` special-cases CPython.
fn soabi_tag(so_ext: &str) -> String {
    so_ext
        .split('.')
        .nth(1)
        .unwrap_or_default()
        .split('-')
        .take(2)
        .collect::<Vec<_>>()
        .join("-")
}

/// The platform half of the extension ABI, empty where there is none.
/// `sys.implementation._multiarch` publishes the same string.
pub(crate) fn multiarch() -> &'static str {
    if cfg!(target_os = "macos") {
        "darwin"
    } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        "x86_64-linux-gnu"
    } else if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        "aarch64-linux-gnu"
    } else {
        ""
    }
}

/// Whether this build has an extension loader — the predicate `create_dynamic`
/// is gated on (`imp/interp_imp.py`), and therefore the one that decides
/// whether `_imp.extension_suffixes()` answers anything.  A build that cannot
/// load an extension must not name the files one would be compiled against.
pub(crate) const fn extension_loader_present() -> bool {
    cfg!(all(
        feature = "cpyext",
        not(feature = "sandbox"),
        any(target_os = "macos", target_os = "linux")
    ))
}

/// The name `EXT_SUFFIX` and `SO` publish.
///
/// A `cpyext` build has an extension loader, and the suffix it builds its
/// candidate filenames from is the authority. Without the loader there is no
/// loaded ABI to read, so the same name is spelled from the pieces the loader
/// would have used and a build that gains the loader keeps the metadata it
/// already published.
fn extension_abi_suffix() -> String {
    #[cfg(all(
        feature = "cpyext",
        not(feature = "sandbox"),
        any(target_os = "macos", target_os = "linux")
    ))]
    {
        crate::cpyext::extension_suffix().to_string()
    }
    #[cfg(not(all(
        feature = "cpyext",
        not(feature = "sandbox"),
        any(target_os = "macos", target_os = "linux")
    )))]
    {
        let platform_tag = match multiarch() {
            "" => String::new(),
            tag => format!("-{tag}"),
        };
        let shared_ext = if cfg!(windows) { ".pyd" } else { ".so" };
        format!(".pyre314{platform_tag}{shared_ext}")
    }
}

/// `shutil.which` for the compiler probe in `init_sysconfigdata`: the build
/// variables name `gcc` only when one is on `PATH`.
///
/// The entries are tested with [`exists_and_is_executable`], the same probe
/// `find_invoked_executable` walks `PATH` with, so both answer X_OK the way
/// `os.access` does rather than by reading mode bits.
#[cfg(all(
    feature = "host_env",
    not(feature = "sandbox"),
    not(target_arch = "wasm32")
))]
fn on_path(program: &str) -> bool {
    std::env::var_os("PATH").is_some_and(|path| {
        std::env::split_paths(&path).any(|directory| {
            !directory.as_os_str().is_empty() && exists_and_is_executable(&directory.join(program))
        })
    })
}

/// A sandboxed interpreter reaches no host `PATH` and a wasm guest has no host
/// process at all, so neither names a compiler.
#[cfg(not(all(
    feature = "host_env",
    not(feature = "sandbox"),
    not(target_arch = "wasm32")
)))]
fn on_path(_program: &str) -> bool {
    false
}

/// `sys.base_prefix` as a path, empty where the interpreter publishes `''`.
fn sysconfigdata_base_prefix() -> PathBuf {
    #[cfg(feature = "host_env")]
    {
        startup_path_config().base_prefix.clone()
    }
    #[cfg(not(feature = "host_env"))]
    {
        PathBuf::new()
    }
}

/// Spell `path` so that it survives as one word in `CFLAGS`.
///
/// Both readers of that variable split it the way a shell does:
/// `customize_compiler` hands `cc + ' ' + cflags` to `distutils.util.split_quoted`,
/// and a build backend that pastes the variable into a command line hands it to
/// the shell.  A prefix with a space in it would reach either of them as two
/// arguments.  Double quotes are the form they agree on; a path that needs none
/// is left alone so the variable reads the way it always has.
fn quote_for_cflags(path: &str) -> String {
    let bare = |b: u8| b.is_ascii_alphanumeric() || b"-_./+=:,@~".contains(&b);
    if path.bytes().all(bare) {
        return path.to_owned();
    }
    let mut quoted = String::with_capacity(path.len() + 2);
    quoted.push('"');
    for ch in path.chars() {
        // The two characters that would otherwise end the quoted region or
        // start an escape inside it.
        if matches!(ch, '"' | '\\') {
            quoted.push('\\');
        }
        quoted.push(ch);
    }
    quoted.push('"');
    quoted
}

/// `_sysconfigdata` — `build_time_vars`, the mapping `sysconfig._init_posix`
/// merges into the config vars.
///
/// CPython writes this module out of its `configure` run and installs it beside
/// the stdlib. Pyre has no configure step, so the values come from the running
/// build instead, and a builtin module keeps them where the rest of the build
/// metadata already lives: `_sysconfig.config_vars()` answers `_init_non_posix`
/// from the same source on Windows.
///
/// `sysconfig._get_sysconfigdata` reaches this module by name only after
/// `_PYTHON_SYSCONFIGDATA_NAME` and `_PYTHON_SYSCONFIGDATA_PATH` have had their
/// turn, so a cross-compilation snapshot still overrides it.
fn init_sysconfigdata(ns: PyObjectRef) -> Result<(), crate::PyError> {
    // These take the root slot rather than the dict itself: a word passed by
    // value goes stale the moment the key or the value allocates.
    fn store_str(vars_slot: usize, key: &str, value: &str) {
        unsafe {
            let key_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_new_managed(key));
            let val_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_new_managed(value));
            pyre_object::w_dict_store(
                pyre_object::gc_roots::shadow_stack_get(vars_slot),
                pyre_object::gc_roots::shadow_stack_get(key_slot),
                pyre_object::gc_roots::shadow_stack_get(val_slot),
            );
        }
    }
    fn store_int(vars_slot: usize, key: &str, value: i64) {
        unsafe {
            let key_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_new_managed(key));
            let w_value = pyre_object::w_int_new(value);
            pyre_object::w_dict_store(
                pyre_object::gc_roots::shadow_stack_get(vars_slot),
                pyre_object::gc_roots::shadow_stack_get(key_slot),
                w_value,
            );
        }
    }

    let so_ext = extension_abi_suffix();
    let soabi = soabi_tag(&so_ext);

    let gnuld = on_path("gcc");
    // Darwin names the architecture in CC and CXX, and `platform.machine()`
    // spells aarch64 `arm64` there.
    #[cfg(target_os = "macos")]
    let arch = if cfg!(target_arch = "aarch64") {
        "arm64"
    } else {
        "x86_64"
    };
    #[cfg(target_os = "macos")]
    let arch_flag = format!(" -arch {arch}");
    #[cfg(not(target_os = "macos"))]
    let arch_flag = "";

    let cc = format!(
        "{}{arch_flag}",
        if gnuld { "gcc -pthread" } else { "cc -pthread" }
    );
    let cxx = format!(
        "{}{arch_flag}",
        if gnuld && on_path("g++") {
            "g++ -pthread"
        } else {
            "c++ -pthread"
        }
    );

    #[cfg(not(target_os = "macos"))]
    let (ldflags, ldshared, ldcxxshared) = {
        let ldflags = String::from("-Wl,-Bsymbolic-functions");
        let ldshared = format!("{cc} -shared {ldflags}");
        (
            ldflags,
            ldshared,
            String::from("c++ -shared -Wl,-O1 -Wl,-Bsymbolic-functions"),
        )
    };
    #[cfg(target_os = "macos")]
    let (ldflags, ldshared, ldcxxshared) = (
        String::from("-undefined dynamic_lookup"),
        String::from("clang -bundle -undefined dynamic_lookup "),
        String::from("clang++ -bundle -undefined dynamic_lookup "),
    );

    let base_prefix = sysconfigdata_base_prefix();
    // `include/{implementation_lower}{py_version_short}{abi_thread}`, the
    // directory the posix_prefix scheme resolves `get_paths()['include']` to.
    // An interpreter that publishes no base prefix has nowhere to name.
    //
    // The predicate is the loader's own, the one `extension_abi_suffix` and
    // `has_so_extension` are built on rather than the `cpyext` feature alone:
    // under `sandbox`, or on a platform with no `create_dynamic`,
    // `_imp.extension_suffixes()` is empty, and naming a header directory
    // there would let a build backend compile an extension this interpreter
    // can never load.
    let include_py = if extension_loader_present() && !base_prefix.as_os_str().is_empty() {
        base_prefix
            .join("include")
            .join("pyre3.14t")
            .to_string_lossy()
            .into_owned()
    } else {
        String::new()
    };

    let _roots = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    let vars_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_dict_new());
    let prefix_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_from_wtf8_managed(
        crate::gateway::fsdecode_os_str_wtf8(base_prefix.as_os_str()),
    ));
    // `_init_non_posix` derives the same `t` from `Py_GIL_DISABLED` below.
    // The lower-case `abiflags` that forms the include and site-packages
    // directory names is a separate variable, read from `sys.abiflags`
    // (`sysconfig:545`), and stays empty: the release tree has no `t` suffix
    // on its stdlib directory for `site.py:409` to find.
    store_str(vars_slot, "ABIFLAGS", "t");
    store_str(vars_slot, "SOABI", &soabi);
    // Deprecated in Python 3, kept for backward compatibility.
    store_str(vars_slot, "SO", &so_ext);
    store_str(vars_slot, "MULTIARCH", multiarch());
    store_str(vars_slot, "CC", &cc);
    store_str(vars_slot, "CXX", &cxx);
    store_str(vars_slot, "OPT", "-DNDEBUG -O2");
    // `setuptools`' vendored `_distutils.sysconfig._get_python_inc_posix_prefix`
    // builds the include directory as `include/<impl><version><abiflags>` with
    // `<impl>` chosen by `IS_PYPY = '__pypy__' in sys.builtin_module_names`.
    // Pyre answers that test and spells its own tree `pyre3.14t`, so the `-I`
    // that `build_ext` appends names a directory pyre never installs.  `CFLAGS`
    // is the path that cannot be displaced: `configure_system` builds every
    // extension compile as `cc + ' ' + cflags`, and the unusable `-I` alongside
    // it is inert.
    let mut cflags = String::from("-DNDEBUG -O2");
    if !include_py.is_empty() {
        cflags.push_str(" -I");
        cflags.push_str(&quote_for_cflags(&include_py));
    }
    store_str(vars_slot, "CFLAGS", &cflags);
    store_str(vars_slot, "CCSHARED", "-fPIC");
    store_str(vars_slot, "LDFLAGS", &ldflags);
    store_str(vars_slot, "LDSHARED", &ldshared);
    store_str(vars_slot, "LDCXXSHARED", &ldcxxshared);
    store_str(vars_slot, "EXT_SUFFIX", &so_ext);
    store_str(vars_slot, "SHLIB_SUFFIX", ".so");
    store_str(vars_slot, "AR", "ar");
    store_str(vars_slot, "ARFLAGS", "rc");
    store_str(vars_slot, "EXE", "");
    store_str(vars_slot, "VERSION", "3.14");
    store_str(vars_slot, "LDVERSION", "3.14");
    // cpyext never uses Py_DEBUG.  Pyre runs its mutators without a global
    // interpreter lock.  Py_ENABLE_SHARED at 1 would add a python shared
    // object to link lines as `-lpython3.x`.
    store_int(vars_slot, "Py_DEBUG", 0);
    store_int(vars_slot, "Py_GIL_DISABLED", 1);
    store_int(vars_slot, "Py_ENABLE_SHARED", 0);
    // Publish an entropy capability only when it names the call this build
    // makes. Unsandboxed `host_env` reaches the host through `getrandom::fill`,
    // which picks its backend per target: `getrandom(2)` for the first set
    // below, `getentropy(2)` for the second. A target in neither set makes
    // neither call — the iOS family goes through `CCRandomGenerateBytes`,
    // NetBSD resolves `getrandom` at runtime and reads a sysctl when it is
    // absent, and Haiku, Redox, NTO and AIX open `/dev/urandom` — so it
    // publishes no name, and neither do the sandbox and no-`host_env` routes,
    // which open `/dev/urandom`, or Windows, which uses its cryptography
    // provider.
    #[cfg(all(
        any(
            target_os = "linux",
            target_os = "android",
            target_os = "freebsd",
            target_os = "dragonfly",
            target_os = "illumos",
            target_os = "solaris",
            target_os = "hurd",
        ),
        feature = "host_env",
        not(feature = "sandbox")
    ))]
    store_int(vars_slot, "HAVE_GETRANDOM_SYSCALL", 1);
    #[cfg(all(
        any(
            target_os = "macos",
            target_os = "openbsd",
            target_os = "vita",
            target_os = "emscripten",
        ),
        feature = "host_env",
        not(feature = "sandbox")
    ))]
    store_int(vars_slot, "HAVE_GETENTROPY", 1);
    // Pyre has no separately linkable runtime library.  Keep the build ABI
    // metadata above for wheel tags, but never invent files that are absent
    // from the installation.
    for key in [
        "LIBRARY",
        "LDLIBRARY",
        "LIBPYTHON",
        "INCLUDEPY",
        "CONFINCLUDEPY",
        "LIBDIR",
    ] {
        store_str(vars_slot, key, "");
    }
    // The headers are the one entry above a build can have: they exist in a
    // `cpyext` build and nowhere else, so the empty default stands until there
    // is a directory to name.  Build backends that do not go through
    // `_distutils` — meson-python, scikit-build, CMake's FindPython — read
    // `INCLUDEPY` directly.
    if !include_py.is_empty() {
        store_str(vars_slot, "INCLUDEPY", &include_py);
        store_str(vars_slot, "CONFINCLUDEPY", &include_py);
    }
    store_int(
        vars_slot,
        "SIZEOF_VOID_P",
        std::mem::size_of::<usize>() as i64,
    );
    if gnuld {
        store_str(vars_slot, "GNULD", "yes");
    }
    #[cfg(target_os = "macos")]
    {
        // scikit-build checks WITH_DYLD; it is left over from the NextStep rld
        // linker.  MACOSX_DEPLOYMENT_TARGET was added to solve problems that
        // may have been solved elsewhere — see cibuildwheel PR 185 and
        // pypa/wheel, and check the interaction with `build_cffi_imports.py`
        // before removing it.  Keep it in sync with DARWIN_VERSION_MIN in
        // `rpython/translator/platform/darwin.py` and `Lib/_osx_support.py`.
        store_int(vars_slot, "WITH_DYLD", 1);
        store_str(
            vars_slot,
            "MACOSX_DEPLOYMENT_TARGET",
            if arch == "arm64" { "11.0" } else { "10.15" },
        );
    }
    // Python 3.14's relocation check reads these from the generated data.
    unsafe {
        for key in ["prefix", "exec_prefix", "srcdir"] {
            // The key pin lives across this one store. `vars_slot` and
            // `prefix_slot` stay on the enclosing bracket.
            let _pyre_store_roots = pyre_object::gc_roots::push_roots();
            let key_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_new_managed(key));
            pyre_object::w_dict_store(
                pyre_object::gc_roots::shadow_stack_get(vars_slot),
                pyre_object::gc_roots::shadow_stack_get(key_slot),
                pyre_object::gc_roots::shadow_stack_get(prefix_slot),
            );
        }
    }
    // Keep PyPy's relocatable zoneinfo search rooted at `base_prefix`.  The
    // C-runtime path block above intentionally differs: PyPy ships libpypy
    // beside its binary, whereas pyre has no separate library to name.
    #[cfg(not(windows))]
    {
        let mut tzpaths = vec![
            base_prefix.join("share").join("zoneinfo"),
            base_prefix.join("lib").join("zoneinfo"),
            base_prefix.join("share").join("lib").join("zoneinfo"),
            base_prefix.join("..").join("etc").join("zoneinfo"),
        ];
        // Absolute system paths, unless `base_prefix` is `/usr` and the four
        // relative entries above already name them.
        if base_prefix != Path::new("/usr") {
            tzpaths.extend(
                [
                    "/usr/share/zoneinfo",
                    "/usr/lib/zoneinfo",
                    "/usr/share/lib/zoneinfo",
                    "/etc/zoneinfo",
                ]
                .map(PathBuf::from),
            );
        }
        let mut joined = std::ffi::OsString::new();
        for (index, path) in tzpaths.iter().enumerate() {
            if index > 0 {
                joined.push(":");
            }
            joined.push(path);
        }
        unsafe {
            let key_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_new("TZPATH"));
            let w_value =
                pyre_object::w_str_from_wtf8_managed(crate::gateway::fsdecode_os_str_wtf8(&joined));
            pyre_object::w_dict_store(
                pyre_object::gc_roots::shadow_stack_get(vars_slot),
                pyre_object::gc_roots::shadow_stack_get(key_slot),
                w_value,
            );
        }
    }
    crate::module_ns_store(
        pyre_object::gc_roots::shadow_stack_get(ns_slot),
        "build_time_vars",
        pyre_object::gc_roots::shadow_stack_get(vars_slot),
    );
    Ok(())
}

/// `_tracemalloc` stub — allocation tracking is not implemented, so the
/// tracing primitives are neutral no-ops that let `tracemalloc` import and
/// report an inactive tracer.
fn init_tracemalloc(mut ns: PyObjectRef) -> Result<(), crate::PyError> {
    let _root_scope = pyre_object::gc_roots::push_roots();
    crate::__pyre_store!(
        ns,
        "start",
        crate::make_builtin_function("start", |_| Ok(pyre_object::w_none()))
    );
    crate::__pyre_store!(
        ns,
        "stop",
        crate::make_builtin_function("stop", |_| Ok(pyre_object::w_none()))
    );
    crate::__pyre_store!(
        ns,
        "clear_traces",
        crate::make_builtin_function("clear_traces", |_| Ok(pyre_object::w_none()))
    );
    crate::__pyre_store!(
        ns,
        "reset_peak",
        crate::make_builtin_function("reset_peak", |_| Ok(pyre_object::w_none()))
    );
    crate::__pyre_store!(
        ns,
        "is_tracing",
        crate::make_builtin_function("is_tracing", |_| Ok(pyre_object::w_bool_from(false)))
    );
    crate::__pyre_store!(
        ns,
        "get_traceback_limit",
        crate::make_builtin_function("get_traceback_limit", |_| Ok(pyre_object::w_int_new(1)))
    );
    crate::__pyre_store!(
        ns,
        "get_tracemalloc_memory",
        crate::make_builtin_function("get_tracemalloc_memory", |_| Ok(pyre_object::w_int_new(0)))
    );
    crate::__pyre_store!(
        ns,
        "get_traced_memory",
        crate::make_builtin_function("get_traced_memory", |_| {
            let mut fields = pyre_object::gc_roots::RootedItems::new();
            fields.push(pyre_object::w_int_new(0));
            fields.push(pyre_object::w_int_new(0));
            Ok(pyre_object::w_tuple_new(fields.take()))
        })
    );
    crate::__pyre_store!(
        ns,
        "_get_traces",
        crate::make_builtin_function("_get_traces", |_| Ok(pyre_object::w_list_new(Vec::new())))
    );
    crate::__pyre_store!(
        ns,
        "_get_object_traceback",
        crate::make_builtin_function("_get_object_traceback", |_| Ok(pyre_object::w_none()))
    );
    Ok(())
}

/// `MixedModule._load_lazily` builds `BuiltinFunction(func)` for a function
/// that sits directly in the module. The carrier is a globals-less `Function`
/// with no instance dict — the `demote_module_function_to_builtin` condition —
/// or an object already tagged `BuiltinFunction`. A carrier the GC already
/// owns is traced from the module dict and stays in place.
///
/// # Safety
/// `value` must be a valid object pointer. Null fails the type check.
unsafe fn untraced_mixed_module_function(value: PyObjectRef) -> bool {
    let carrier = if unsafe { py_type_check(value, &crate::function::FUNCTION_TYPE) } {
        let func = unsafe { &*(value as *const crate::function::Function) };
        func.w_func_globals_obj.is_null() && func.w_func_dict.is_null()
    } else {
        unsafe { py_type_check(value, &crate::function::BUILTIN_FUNCTION_TYPE) }
    };
    carrier && !pyre_object::gc_hook::try_gc_owns_object(value as pyre_object::gc_hook::GCREF)
}

/// `baseobjspace.py getbuiltinmodule`.
///
/// `Ok(None)` is "no builtin by that name", which upstream raises as a
/// `SystemError`: pyre's importers screen the name against the registry
/// first, and `_imp.create_builtin` reports it as `None`.
pub fn getbuiltinmodule(
    name: &str,
    force_init: bool,
    reuse: bool,
    execution_context: *const PyExecutionContext,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};

    if !force_init {
        debug_assert!(reuse);
        if let Some(w_mod) = sys_modules_entry(name) {
            return Ok(Some(w_mod));
        }
    }
    // If the module is a builtin but not yet imported,
    // retrieve it and initialize it
    let Some(w_mod) = builtin_modules_get(name, execution_context)? else {
        return Ok(None);
    };
    let _roots = push_roots();
    let mod_slot = shadow_stack_len();
    let _ = pin_root(w_mod);
    // Add the module to sys.modules and initialize the module. The
    // order is important to avoid recursions.
    if !reuse && unsafe { pyre_object::w_module_startup_called(shadow_stack_get(mod_slot)) } {
        // create a copy of the module.  (see issue1514) eventlet
        // patcher relies on this behaviour.
        let copy_slot = shadow_stack_len();
        let _ = pin_root(pyre_object::w_module_new_managed(name));
        set_sys_module(name, shadow_stack_get(copy_slot));
        // A startup that raised left no saved dict; `update(None)` raises.
        let w_initialdict =
            unsafe { pyre_object::w_module_get_initialdict(shadow_stack_get(mod_slot)) };
        let w_initialdict = if w_initialdict.is_null() {
            pyre_object::w_none()
        } else {
            w_initialdict
        };
        crate::baseobjspace::call_method_result(
            unsafe { pyre_object::w_module_get_w_dict(shadow_stack_get(copy_slot)) },
            "update",
            &[w_initialdict],
        )?;
        return Ok(Some(shadow_stack_get(copy_slot)));
    }
    set_sys_module(name, shadow_stack_get(mod_slot));
    mixedmodule_init(name, shadow_stack_get(mod_slot), execution_context)?;
    Ok(Some(shadow_stack_get(mod_slot)))
}

/// `space.builtin_modules[name]`.
///
/// The `builtins` entry is `space.builtin`: every frame falls back to the
/// execution context's module for `LOAD_GLOBAL`, so `import builtins` binds
/// that one. Every other entry is built by [`new_builtin_module`] the first
/// time it is named and kept in [`BUILTIN_MODULES`].
fn builtin_modules_get(
    name: &str,
    execution_context: *const PyExecutionContext,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    if name == "builtins" && !execution_context.is_null() {
        return Ok(Some(unsafe { (*execution_context).get_builtin() }));
    }
    let (static_name, module_def) = {
        let table = BUILTIN_MODULES.lock();
        let Some((static_name, def)) = table.get_key_value(name) else {
            return Ok(None);
        };
        (*static_name, *def)
    };
    if module_def.w_mod != 0 {
        return Ok(Some(module_def.w_mod as PyObjectRef));
    }
    let w_mod = new_builtin_module(static_name, module_def)?;
    // An initializer that named this module again has installed one already,
    // and that one stays the entry.
    let mut table = BUILTIN_MODULES.lock();
    let entry = table
        .get_mut(static_name)
        .expect("a registered builtin module stays registered");
    if entry.w_mod == 0 {
        // Young module pointer into a prebuilt `space.builtin_modules` slot
        // (`incminimark.py remember_young_pointer_from_prebuilt`).
        entry.w_mod = w_mod as usize;
        pyre_object::gc_roots::mark_prebuilt_roots_dirty();
    }
    Ok(Some(entry.w_mod as PyObjectRef))
}

/// `mixedmodule.py MixedModule.init`: called each time the module is imported
/// or reloaded.
fn mixedmodule_init(
    name: &str,
    w_mod: PyObjectRef,
    execution_context: *const PyExecutionContext,
) -> Result<(), crate::PyError> {
    use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};

    let _roots = push_roots();
    let mod_slot = shadow_stack_len();
    let _ = pin_root(w_mod);
    let w_initialdict =
        unsafe { pyre_object::w_module_get_initialdict(shadow_stack_get(mod_slot)) };
    if !w_initialdict.is_null() {
        // the module was already imported.  Refresh its content with
        // the saved dict.
        crate::baseobjspace::call_method_result(
            unsafe { pyre_object::w_module_get_w_dict(shadow_stack_get(mod_slot)) },
            "update",
            &[w_initialdict],
        )?;
    }
    if unsafe { pyre_object::w_module_get_initialdict(shadow_stack_get(mod_slot)) }.is_null() {
        module_init(name, shadow_stack_get(mod_slot), execution_context)?;
        if unsafe { pyre_object::w_module_get_initialdict(shadow_stack_get(mod_slot)) }.is_null() {
            save_module_content_for_future_reload(shadow_stack_get(mod_slot))?;
        }
    }
    Ok(())
}

/// `module.py Module.init`: the first call runs `startup()`.
fn module_init(
    name: &str,
    w_mod: PyObjectRef,
    execution_context: *const PyExecutionContext,
) -> Result<(), crate::PyError> {
    if !unsafe { pyre_object::w_module_startup_called(w_mod) } {
        unsafe { pyre_object::w_module_set_startup_called(w_mod) };
        let startup = BUILTIN_MODULES.lock().get(name).and_then(|d| d.startup);
        if let Some(startup) = startup {
            startup(w_mod, execution_context)?;
        }
    }
    Ok(())
}

/// `mixedmodule.py save_module_content_for_future_reload`: save the current
/// dictionary in `w_initialdict`, for future reloads.
fn save_module_content_for_future_reload(w_mod: PyObjectRef) -> Result<(), crate::PyError> {
    use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};

    let _roots = push_roots();
    let mod_slot = shadow_stack_len();
    let _ = pin_root(w_mod);
    let w_initialdict = crate::baseobjspace::call_method_result(
        unsafe { pyre_object::w_module_get_w_dict(shadow_stack_get(mod_slot)) },
        "copy",
        &[],
    )?;
    unsafe { pyre_object::w_module_set_initialdict(shadow_stack_get(mod_slot), w_initialdict) };
    Ok(())
}

/// `module.py init_extra_module_attrs`.
fn init_extra_module_attrs(w_dict: PyObjectRef) {
    for extra in ["__package__", "__loader__", "__spec__"] {
        if unsafe { pyre_object::dictmultiobject::w_dict_getitem_str(w_dict, extra) }.is_none() {
            crate::module_ns_store(w_dict, extra, pyre_object::w_none());
        }
    }
}

/// `MixedModule.__init__` for a registered builtin, which `install()` puts in
/// `space.builtin_modules`. It runs the first time [`builtin_modules_get`]
/// names the module, not when the space is built. The initializer fills the
/// namespace the way forcing every interpleveldef loader does
/// (`MixedModule.getdict`), so the module comes back no longer `lazy`.
///
/// Every module gets a `W_ModuleDictObject`
/// (`dictmultiobject.py allocate_and_init_instance(module=True)`), and the
/// initializer writes directly into that rooted, non-moving dict.
///
/// An `Err` is an initializer that ran and raised.  A module whose namespace
/// comes from a bundled app-level source fails the way any module body can --
/// the source's own imports resolve through the running `sys.modules`, which
/// the program owns -- and that belongs to the import which asked for the
/// module, not to the process.
fn new_builtin_module(
    name: &'static str,
    module_def: BuiltinModuleDef,
) -> Result<PyObjectRef, crate::PyError> {
    let w_dict = pyre_object::dictmultiobject::w_module_dict_new();
    let _roots = pyre_object::gc_roots::push_roots();
    let save_point = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_dict);
    // Immortal builtin functions stamp this as `w_module`. A nursery
    // string in that slot is reached only by `walk_raw_function_roots`,
    // which clean minors skip, so intern it the way `w(modulename)` does.
    let name_obj = pyre_object::unicodeobject::intern_str_value(name);
    let _ = pyre_object::gc_roots::pin_root(name_obj);
    // Set __name__ (PyPy: Module.__init__ sets __name__)
    crate::module_ns_store(
        pyre_object::gc_roots::shadow_stack_get(save_point),
        "__name__",
        pyre_object::gc_roots::shadow_stack_get(save_point + 1),
    );
    init_extra_module_attrs(pyre_object::gc_roots::shadow_stack_get(save_point));
    // Run module-specific initializer (PyPy: interpleveldefs)
    (module_def.init)(pyre_object::gc_roots::shadow_stack_get(save_point))?;
    // One flag for the holder and the module object. Before `sys.modules`
    // exists, or for a legacy MixedModule, functions stay Box-immortal and
    // are retagged in place. A collectible module follows
    // `MixedModule._load_lazily`: each direct function becomes a fresh GC
    // `BuiltinFunction` (`BuiltinFunction.__init__`) so the managed module
    // name and the module object are traced from the module dict.
    let managed_holder = !sys_modules_dict().is_null() && module_def.collectible;
    // MixedModule parity: interp-level builtin functions carry the module
    // name as `__module__`, so `pickle` can save them by reference
    // (`save_global`) without guessing via `whichmodule`. Snapshot owned
    // keys, then reload each movable value and the rooted name afresh.
    let keys: Vec<String> = unsafe {
        pyre_object::dictmultiobject::w_dict_str_entries(pyre_object::gc_roots::shadow_stack_get(
            save_point,
        ))
    }
    .into_iter()
    .map(|(key, _)| key)
    .collect();
    for key in &keys {
        // A managed copy's pin stays through `module_ns_store`,
        // `builtin_function_set_module` and `with_module`, then drops.
        // `w_dict` and the module name stay on `_roots`.
        let _pyre_store_roots = pyre_object::gc_roots::push_roots();
        if let Some(value) = unsafe {
            pyre_object::dictmultiobject::w_dict_getitem_str(
                pyre_object::gc_roots::shadow_stack_get(save_point),
                key,
            )
        } {
            let value = if managed_holder && unsafe { untraced_mixed_module_function(value) } {
                // `BuiltinFunction(func)`. Pin the result before the dict
                // store: until that store it is reachable only from this
                // local, and `module_ns_store` can allocate. Reload it from
                // the shadow stack after that store.
                let created = unsafe { crate::function::builtin_function_new_managed(value) };
                let created_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(created);
                crate::module_ns_store(
                    pyre_object::gc_roots::shadow_stack_get(save_point),
                    key,
                    pyre_object::gc_roots::shadow_stack_get(created_slot),
                );
                pyre_object::gc_roots::shadow_stack_get(created_slot)
            } else {
                unsafe {
                    // MixedModule._load_lazily: every function directly in a
                    // mixed-module is a non-descriptor `BuiltinFunction` (no
                    // `__get__`), so storing it on a user class does not bind `self`.
                    crate::function::demote_module_function_to_builtin(value);
                }
                value
            };
            unsafe {
                crate::function::builtin_function_set_module(
                    value,
                    pyre_object::gc_roots::shadow_stack_get(save_point + 1),
                );
            }
            // The same name on the code object, where the error wordings read
            // it (`math.sqrt() takes exactly one argument`).  A module built by
            // a registration table already stamped its own functions, so this
            // only reaches the hand-built namespaces. A managed copy shares
            // `func.code`, so this stamps that code when init did not.
            // The registry key outlives the module, which is what lets this
            // hand the name to `BuiltinCode.module` without copying it.
            crate::gateway::with_module(name, value);
        }
    }
    // `mixedmodule.py:192-193` — a module def that names no `__doc__` still
    // publishes one, through `get__doc__`'s `newtext_or_none(cls.__doc__)`.
    // A Rust module def carries no class docstring, so the None arm is what
    // every def that stays silent resolves to; the key itself is not optional,
    // and `dir(<module>)` lists it.
    if unsafe {
        pyre_object::dictmultiobject::w_dict_getitem_str(
            pyre_object::gc_roots::shadow_stack_get(save_point),
            "__doc__",
        )
    }
    .is_none()
    {
        crate::module_ns_store(
            pyre_object::gc_roots::shadow_stack_get(save_point),
            "__doc__",
            pyre_object::w_none(),
        );
    }
    // Before `sys.modules` exists the native bootstrap registry is the only
    // owner and retains the legacy immortal module shape. Afterwards an
    // audited collectible module follows the PyPy `space.sys.modules` object
    // graph; legacy MixedModules retain the immortal holder until their
    // native/JIT caches have been migrated to traced owners.
    let module = if !managed_holder {
        pyre_object::w_module_new_aliasing_dict(
            name,
            pyre_object::gc_roots::shadow_stack_get(save_point),
        )
    } else {
        pyre_object::w_module_new_aliasing_dict_managed(
            name,
            pyre_object::gc_roots::shadow_stack_get(save_point),
        )
    };
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(module);
    // function.py BuiltinFunction.w_moduleobj — MixedModule binds
    // every interp-level function to the live defining module object.
    for key in &keys {
        if let Some(value) = unsafe {
            pyre_object::dictmultiobject::w_dict_getitem_str(
                pyre_object::gc_roots::shadow_stack_get(save_point),
                key,
            )
        } {
            unsafe {
                crate::function::builtin_function_set_module_obj(
                    value,
                    pyre_object::gc_roots::shadow_stack_get(module_slot),
                )
            };
        }
    }
    // `baseobjspace.py make_builtins` installs the self
    // reference `space.builtin.w_dict['__builtins__'] = space.builtin`
    // so user code can reach the builtins module through
    // `import builtins; builtins.__builtins__`.  The pyre split
    // between EC.builtins_module (used by LOAD_GLOBAL fallback) and
    // the import-time module (returned here) is a known pre-existing
    // adaptation; install the self-reference on the imported flavour
    // so `import builtins; builtins.__builtins__ is builtins` holds
    // for user code regardless of the split.
    if name == "builtins" {
        crate::module_ns_store(
            pyre_object::gc_roots::shadow_stack_get(save_point),
            "__builtins__",
            pyre_object::gc_roots::shadow_stack_get(module_slot),
        );
    }
    Ok(pyre_object::gc_roots::shadow_stack_get(module_slot))
}

/// Set a builtin module's `__spec__`/`__loader__`/`__package__` from the
/// app-level `BuiltinImporter`, matching `BuiltinImporter.exec_module` →
/// `_init_module_attrs`. Reachable only once `_frozen_importlib` is wired;
/// the handful of builtins imported before that are fixed up in bulk by
/// `_bootstrap._setup`'s sys.modules walk, so a no-op here is correct then.
#[cfg(feature = "host_env")]
fn set_builtin_module_spec(name: &str, mut module: PyObjectRef) -> Result<(), crate::PyError> {
    use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};

    let Some(bootstrap) = pyre_object::with_roots!(module => importlib_bootstrap_module()) else {
        return Ok(());
    };

    let _roots = push_roots();
    let mod_slot = shadow_stack_len();
    let _ = pin_root(module);
    let boot_slot = shadow_stack_len();
    let _ = pin_root(bootstrap);

    // Best-effort throughout: a builtin imported while `_bootstrap` is still
    // executing sees a partially-initialised module whose `BuiltinImporter` /
    // `_init_module_attrs` are not defined yet. Skip rather than break the
    // import — `_bootstrap._setup` fixes up any builtin missed here.
    let Ok(importer) =
        crate::baseobjspace::getattr_str(shadow_stack_get(boot_slot), "BuiltinImporter")
    else {
        return Ok(());
    };
    let importer_slot = shadow_stack_len();
    let _ = pin_root(importer);
    let Ok(find_spec) =
        crate::baseobjspace::getattr_str(shadow_stack_get(importer_slot), "find_spec")
    else {
        return Ok(());
    };
    let find_spec_slot = shadow_stack_len();
    let _ = pin_root(find_spec);
    let name_slot = shadow_stack_len();
    let _ = pin_root(pyre_object::w_str_new_managed(name));
    let Ok(spec) = crate::call::call_function_impl_result(
        shadow_stack_get(find_spec_slot),
        &[shadow_stack_get(name_slot)],
    ) else {
        return Ok(());
    };
    if unsafe { pyre_object::is_none(spec) } {
        return Ok(());
    }
    let spec_slot = shadow_stack_len();
    let _ = pin_root(spec);

    // _init_module_attrs(spec, module)
    let Ok(init) =
        crate::baseobjspace::getattr_str(shadow_stack_get(boot_slot), "_init_module_attrs")
    else {
        return Ok(());
    };
    let init_slot = shadow_stack_len();
    let _ = pin_root(init);
    let _ = crate::call::call_function_impl_result(
        shadow_stack_get(init_slot),
        &[shadow_stack_get(spec_slot), shadow_stack_get(mod_slot)],
    );
    Ok(())
}

/// Off-`host_env` builds have no app-level importlib to source specs from.
#[cfg(not(feature = "host_env"))]
fn set_builtin_module_spec(_name: &str, _module: PyObjectRef) -> Result<(), crate::PyError> {
    Ok(())
}

/// Set an extension module's `__spec__`/`__loader__`/`__package__`/`__file__`
/// from the app-level `_bootstrap_external.spec_from_file_location`, matching
/// what `ExtensionFileLoader` installs when the same file is imported through
/// importlib. `PyModule_Create2` supplies only the name, docstring and file, so
/// without this a natively-resolved extension carries less metadata than the
/// source module beside it — and the dict the extension cache snapshots is the
/// incomplete one.
///
/// Best-effort throughout, like `set_builtin_module_spec`: an extension
/// imported before the bootstrap is wired keeps what `PyModule_Create2` gave
/// it rather than failing the import.
#[cfg(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
))]
/// The `ModuleSpec` an extension file is loaded under, or `PY_NULL` when the
/// importlib bootstrap is not wired up yet.
///
/// Built before the extension's init function runs, because a PEP 489
/// `Py_mod_create` slot is handed the spec and reads its `name` and `origin`:
/// a `Py_mod_create` given NULL dereferences it.
///
/// The caller must root the result before anything else allocates.
#[cfg(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
))]
fn extension_module_spec(name: &str, pathname: &Path) -> PyObjectRef {
    use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};

    let Some(ext) = importlib_bootstrap_external_module() else {
        return pyre_object::PY_NULL;
    };
    let _roots = push_roots();
    let ext_slot = shadow_stack_len();
    let _ = pin_root(ext);
    let Ok(from_location) =
        crate::baseobjspace::getattr_str(shadow_stack_get(ext_slot), "spec_from_file_location")
    else {
        return pyre_object::PY_NULL;
    };
    let from_location_slot = shadow_stack_len();
    let _ = pin_root(from_location);
    let name_slot = shadow_stack_len();
    let _ = pin_root(pyre_object::w_str_new_managed(name));
    let w_path = crate::gateway::fsdecode_os_str(pathname.as_os_str());
    let path_slot = shadow_stack_len();
    let _ = pin_root(w_path);
    // A suffix `_get_supported_file_loaders` does not know yields `None`; that
    // is the same "no spec" answer as a missing bootstrap.
    let Ok(spec) = crate::call::call_function_impl_result(
        shadow_stack_get(from_location_slot),
        &[shadow_stack_get(name_slot), shadow_stack_get(path_slot)],
    ) else {
        return pyre_object::PY_NULL;
    };
    if unsafe { pyre_object::is_none(spec) } {
        return pyre_object::PY_NULL;
    }
    spec
}

#[allow(dead_code)]
fn set_extension_module_spec(
    name: &str,
    pathname: &Path,
    mut module: PyObjectRef,
) -> Result<(), crate::PyError> {
    use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};

    let bootstrap = pyre_object::with_roots!(module => importlib_bootstrap_module());
    let has_bootstrap = bootstrap.is_some();
    let mut bootstrap_root = bootstrap.unwrap_or(pyre_object::PY_NULL);
    let ext = pyre_object::with_roots!(
        module, bootstrap_root => importlib_bootstrap_external_module()
    );
    let bootstrap = has_bootstrap.then(|| bootstrap_root);
    let (Some(bootstrap), Some(ext)) = (bootstrap, ext) else {
        return Ok(());
    };

    let _roots = push_roots();
    let mod_slot = shadow_stack_len();
    let _ = pin_root(module);
    let boot_slot = shadow_stack_len();
    let _ = pin_root(bootstrap);
    let ext_slot = shadow_stack_len();
    let _ = pin_root(ext);

    let Ok(from_location) =
        crate::baseobjspace::getattr_str(shadow_stack_get(ext_slot), "spec_from_file_location")
    else {
        return Ok(());
    };
    let from_location_slot = shadow_stack_len();
    let _ = pin_root(from_location);
    let name_slot = shadow_stack_len();
    let _ = pin_root(pyre_object::w_str_new_managed(name));
    let w_path = crate::gateway::fsdecode_os_str(pathname.as_os_str());
    let path_slot = shadow_stack_len();
    let _ = pin_root(w_path);

    // A suffix `_get_supported_file_loaders` does not know yields `None`; that
    // is the same "leave it alone" answer as a missing bootstrap.
    let Ok(spec) = crate::call::call_function_impl_result(
        shadow_stack_get(from_location_slot),
        &[shadow_stack_get(name_slot), shadow_stack_get(path_slot)],
    ) else {
        return Ok(());
    };
    if unsafe { pyre_object::is_none(spec) } {
        return Ok(());
    }
    let spec_slot = shadow_stack_len();
    let _ = pin_root(spec);

    let Ok(init) =
        crate::baseobjspace::getattr_str(shadow_stack_get(boot_slot), "_init_module_attrs")
    else {
        return Ok(());
    };
    let init_slot = shadow_stack_len();
    let _ = pin_root(init);
    let _ = crate::call::call_function_impl_result(
        shadow_stack_get(init_slot),
        &[shadow_stack_get(spec_slot), shadow_stack_get(mod_slot)],
    );
    Ok(())
}

/// Set a source module's `__spec__`/`__loader__`/`__file__`/`__cached__` from
/// the app-level `_bootstrap_external._fix_up_module` — the helper
/// `PyImport_ExecCodeModuleObject` calls. Returns `false` when the importlib
/// bootstrap is not wired yet, so the caller can seed `None` instead.
///
/// `ns` (the module dict) is written in place; the caller keeps it pinned.
#[cfg(feature = "host_env")]
fn fix_up_source_module_spec(
    mut ns: PyObjectRef,
    pathname: &rustpython_wtf8::Wtf8,
    cpathname: Option<&rustpython_wtf8::Wtf8>,
) -> Result<bool, crate::PyError> {
    use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};

    let Some(ext) = pyre_object::with_roots!(ns => importlib_bootstrap_external_module()) else {
        return Ok(false);
    };
    let Some(w_name) = (unsafe { pyre_object::w_dict_getitem_str(ns, "__name__") }) else {
        return Ok(false);
    };

    let _roots = push_roots();
    let ns_slot = shadow_stack_len();
    let _ = pin_root(ns);
    let name_slot = shadow_stack_len();
    let _ = pin_root(w_name);
    let ext_slot = shadow_stack_len();
    let _ = pin_root(ext);
    // `newutf8` copies the path once into the STR payload.
    let w_path = pyre_object::w_str_from_wtf8_managed_borrowed(pathname);
    let path_slot = shadow_stack_len();
    let _ = pin_root(w_path);
    let w_cpath = match cpathname {
        Some(c) => pyre_object::w_str_from_wtf8_managed_borrowed(c),
        None => pyre_object::w_none(),
    };
    let cpath_slot = shadow_stack_len();
    let _ = pin_root(w_cpath);

    // Best-effort: `_bootstrap_external` itself is a source module, and its
    // spec is fixed up (during partial-init) before its body defines
    // `_fix_up_module` — the getattr then raises. Fall back to `None` seeding
    // for that (and any other partially-initialised) case rather than break
    // the import; the module's spec is corrected by later imports / `_setup`.
    let Ok(fix) = crate::baseobjspace::getattr_str(shadow_stack_get(ext_slot), "_fix_up_module")
    else {
        return Ok(false);
    };
    let fix_slot = shadow_stack_len();
    let _ = pin_root(fix);
    // An error raised by `_fix_up_module` itself propagates — the appexec
    // at importing.py:293-298 does not shield the call.
    crate::call::call_function_impl_result(
        shadow_stack_get(fix_slot),
        &[
            shadow_stack_get(ns_slot),
            shadow_stack_get(name_slot),
            shadow_stack_get(path_slot),
            shadow_stack_get(cpath_slot),
        ],
    )?;
    Ok(true)
}

/// Off-`host_env` builds have no app-level importlib to source specs from.
#[cfg(not(feature = "host_env"))]
fn fix_up_source_module_spec(
    _ns: PyObjectRef,
    _pathname: &rustpython_wtf8::Wtf8,
    _cpathname: Option<&rustpython_wtf8::Wtf8>,
) -> Result<bool, crate::PyError> {
    Ok(false)
}

/// Initialize sys.path with the directory containing the main script.
///
/// PyPy equivalent: sys.path is populated at startup with the script
/// directory, then PYTHONPATH entries, then the stdlib.
#[cfg(feature = "host_env")]
/// The canonical absolute form of a startup `sys.path[0]` directory, for the
/// shadowing check to compare against absolute module origins.  Under the
/// sandbox the path is left as given (a virtual path the controller mediates);
/// `canonicalize` would issue raw host syscalls past the seccomp lockdown.
/// The extended-length spelling `canonicalize` answers with on Windows, back
/// as the name everything else uses.
///
/// It resolves through `GetFinalPathNameByHandleW`, whose result carries the
/// `\\?\` prefix — `\\?\Z:\src` for `Z:\src`, `\\?\UNC\host\share` for
/// `\\host\share`.  A module's `__spec__.origin` is the plain name the
/// finder joined, so `is_possibly_shadowing` would be comparing two spellings
/// of the same directory and finding them different.  A prefix with no plain
/// spelling (`\\?\Volume{...}`) is left as it is.
#[cfg(all(windows, not(feature = "sandbox")))]
fn strip_verbatim_prefix(path: PathBuf) -> PathBuf {
    use std::os::windows::ffi::{OsStrExt, OsStringExt};

    const fn wide(s: &str) -> [u16; 4] {
        let b = s.as_bytes();
        [b[0] as u16, b[1] as u16, b[2] as u16, b[3] as u16]
    }
    const SEP: u16 = b'\\' as u16;
    let units: Vec<u16> = path.as_os_str().encode_wide().collect();
    let Some(rest) = units.strip_prefix(&wide(r"\\?\")) else {
        return path;
    };
    let plain = if let Some(unc) = rest.strip_prefix(&wide(r"UNC\")) {
        [&[SEP, SEP][..], unc].concat()
    } else if let [_, colon, SEP, ..] = rest
        && *colon == b':' as u16
    {
        rest.to_vec()
    } else {
        return path;
    };
    PathBuf::from(std::ffi::OsString::from_wide(&plain))
}

fn canonical_startup_dir(dir: &Path) -> Wtf8Buf {
    #[cfg(not(feature = "sandbox"))]
    if let Ok(abs) = std::path::absolute(dir) {
        let abs = abs.canonicalize().unwrap_or_else(|_| abs.clone());
        #[cfg(windows)]
        let abs = strip_verbatim_prefix(abs);
        return crate::gateway::fsdecode_os_str_wtf8(abs.as_os_str());
    }
    crate::gateway::fsdecode_os_str_wtf8(dir.as_os_str())
}

/// `script_dir` is the shadowing-check anchor (`config->sys_path_0`'s
/// directory); `path0` is the literal entry `add_sys_path_0` later prepends to
/// `sys.path` — `""` for `-c` / stdin / the REPL, the cwd for `-m`, the
/// script's directory for a script.
// Startup-only caller of `install_builtin_modules`, matching PyPy's
// `ObjSpace.make_builtins` / `setup_builtin_modules` host phase.  User code
// can mutate `sys.path` later, but never re-enters this process bootstrap.
#[majit_macros::not_rpython]
pub fn init_sys_path(script_dir: &Path, path0: &std::ffi::OsStr) {
    // Register builtin modules (PyPy: make_builtins / setup_builtin_modules)
    install_builtin_modules();

    // Record the startup `sys.path[0]` for the shadowing check before any user
    // code can mutate `sys.path`.  Module origins are absolute canonical paths,
    // so store the canonical absolute form here for the directory comparison to
    // hold for a relative script argument or a symlinked working directory.
    *SYS_PATH_0.lock() = Some(canonical_startup_dir(script_dir));

    // `pymain_run_python` prepends `sys.path[0]` only after `site` has run, so
    // `site.removeduppaths()` never absolutizes the `-c` / REPL empty entry into
    // the cwd.  Stage the entry here; `add_sys_path_0` performs the insert.
    // `-P` (safe_path) suppresses it entirely.
    *SYS_PATH_0_PENDING.lock() = (!safe_path_flag()).then(|| path0.to_os_string());

    // The search path is a filesystem's, so a build with no host filesystem
    // seeds none: `SYS_PATH` is `host_env`'s and so is every entry that could
    // go in it.
    #[cfg(feature = "host_env")]
    {
        let mut path = SYS_PATH.lock();
        path.clear();
        // PYTHONPATH entries head the seed and precede the stdlib
        // (pathconfig.c), split on the platform path-list separator.  Honoured
        // regardless of the safe-path flag, which only suppresses the
        // `sys.path[0]` entry, but skipped under `-E` / `-I` (ignore_environment),
        // which ignore every `PYTHON*` variable. The sandbox interpreter takes
        // its search path from the controller, so it does not read the host
        // environment here.
        #[cfg(not(feature = "sandbox"))]
        if !ignore_environment_flag()
            && let Ok(pythonpath) = host_os::var("PYTHONPATH")
        {
            let sep = if cfg!(windows) { ';' } else { ':' };
            // Empty components are preserved — an empty `sys.path` entry
            // denotes the current directory (app_main.setup_and_fix_paths
            // extends with the raw split).
            path.extend(pythonpath.split(sep).map(PathBuf::from));
        }
        // The stdlib entry is appended when the `sys` module is created —
        // `create_sys_path_list` forces `ensure_stdlib_path` before flushing
        // this seed into `sys.path`.
    }
}

/// Replace the pending startup `sys.path[0]` entry for an accepted directory
/// or zipfile run.  `app_main.py:1054` suppresses the ordinary script
/// directory under safe-path modes, but the package-run branch still prepends
/// the accepted run target so `runpy` can find `__main__`.
pub fn restage_sys_path_0(path0: &std::ffi::OsStr) {
    *SYS_PATH_0_PENDING.lock() = Some(path0.to_os_string());
}

/// Process-wide path configuration computed before `sys` exposes any path.
///
/// PyPy keeps this state on the one object space (`module/sys/state.py`) and
/// fills it from `initpath.py:find_executable` / `find_stdlib`.  It is not
/// execution-context or thread state: a worker importing `sys` must observe
/// the same executable, installation prefix and virtual environment as the
/// launcher thread.  Keep that ownership shape here instead of using TLS.
#[cfg(feature = "host_env")]
#[derive(Clone, Debug)]
pub(crate) struct StartupPathConfig {
    pub executable: PathBuf,
    pub base_executable: PathBuf,
    pub prefix: PathBuf,
    pub base_prefix: PathBuf,
    /// Bootstrap search entries in PyPy's order.  A source checkout keeps
    /// `lib_pypy` before `lib-python/3`; an installed tree has one merged
    /// `lib/pyre3.14t` entry.
    #[cfg(all(feature = "host_env", not(target_arch = "wasm32")))]
    pub stdlib_paths: Vec<PathBuf>,
    /// The language stdlib root (the entry containing `os.py` / `site.py`).
    /// This is exposed as `sys._stdlib_dir` and is deliberately distinct from
    /// the complete search list above.
    pub stdlib: Option<PathBuf>,
    /// The `._pth` beside the executable, when the deployment placed one.
    #[cfg(all(not(feature = "sandbox"), not(target_arch = "wasm32")))]
    pub pth: Option<PthConfig>,
}

#[cfg(feature = "host_env")]
static STARTUP_PATH_CONFIG: OnceLock<StartupPathConfig> = OnceLock::new();

#[cfg(all(
    feature = "host_env",
    not(feature = "sandbox"),
    not(target_arch = "wasm32")
))]
#[derive(Default, Debug, PartialEq, Eq)]
struct PyVenvConfig {
    home: Option<PathBuf>,
    executable: Option<PathBuf>,
}

/// Parse only the two path keys needed during bootstrap.  This deliberately
/// mirrors PyPy's tiny `find_pyvenv_cfg` parser rather than importing `configparser`
/// before the stdlib path exists.  CPython's `venv.create_configuration`
/// writes both keys as UTF-8, one `key = value` pair per line.
#[cfg(all(
    feature = "host_env",
    not(feature = "sandbox"),
    not(target_arch = "wasm32")
))]
fn parse_pyvenv_cfg(contents: &str) -> PyVenvConfig {
    let mut config = PyVenvConfig::default();
    for line in contents.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let value = value.trim();
        if value.is_empty() {
            continue;
        }
        match key.trim().to_ascii_lowercase().as_str() {
            "home" => config.home = Some(PathBuf::from(value)),
            "executable" => config.executable = Some(PathBuf::from(value)),
            _ => {}
        }
    }
    config
}

/// PyPy `initpath.py:_exists_and_is_executable`: executable discovery is a
/// launch decision, not merely a filesystem existence check.  In particular,
/// a non-executable file earlier on PATH must not shadow the real interpreter.
#[cfg(all(
    feature = "host_env",
    not(feature = "sandbox"),
    not(target_arch = "wasm32")
))]
fn exists_and_is_executable(path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let Ok(path) = std::ffi::CString::new(path.as_os_str().as_bytes()) else {
            return false;
        };
        // Like os.access(X_OK), libc::access checks the process's real uid and
        // gid.  Inspecting mode bits by hand would get ACLs and root wrong.
        unsafe { libc::access(path.as_ptr(), libc::X_OK) == 0 }
    }
    #[cfg(not(unix))]
    {
        true
    }
}

#[cfg(all(
    feature = "host_env",
    not(feature = "sandbox"),
    not(target_arch = "wasm32")
))]
fn absolute_from(path: PathBuf, cwd: &Path) -> PathBuf {
    let joined = if path.is_absolute() {
        path
    } else {
        cwd.join(path)
    };
    normalize_lexically(&joined)
}

/// Collapse `.` and `..` without touching the filesystem, the second half of
/// `posixpath.abspath` (`normpath(join(cwd, path))`).  `Path::join` alone
/// leaves the spelling the caller typed, so invoking `./venv/bin/pyre` would
/// carry the `.` into `sys.executable` and every prefix derived from it —
/// `site.py` then reports the venv prefix as unexpected because it compares
/// that string against the directory it found itself in.
///
/// Symlinks stay unresolved on purpose: `find_invoked_executable` needs the
/// venv spelling, and a lexical walk cannot follow a link anyway.  That is the
/// same division `initpath.py` draws between `abspath` and `resolvedirof`.
#[cfg(all(
    feature = "host_env",
    not(feature = "sandbox"),
    not(target_arch = "wasm32")
))]
fn normalize_lexically(path: &Path) -> PathBuf {
    // `\\.\` device names and `\\?\` literal paths are handed to the OS as
    // spelled and are returned unchanged (`_nt_rnormpath`): a `.` or `..`
    // inside one is an ordinary name, so collapsing it would rewrite which
    // object the path reaches.
    //
    // The exemption is read off the spelling rather than off the parsed
    // prefix, and the difference is observable: `//./name` parses as the same
    // device prefix, but `_nt_rnormpath` tests for the two literal prefixes
    // *before* it rewrites `/` into `\`, so a slash-spelled device path is
    // never exempt and goes on to be normalized like any other.  The whole
    // arm is Windows-only because `_posix_rnormpath` has no counterpart to
    // it, and on unix a backslash is an ordinary filename character.
    #[cfg(windows)]
    {
        let spelled = path.as_os_str().as_encoded_bytes();
        if spelled.starts_with(br"\\.\") || spelled.starts_with(br"\\?\") {
            return path.to_path_buf();
        }
    }
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => match out.components().next_back() {
                // Only a normal segment can be cancelled.
                Some(Component::Normal(_)) => {
                    out.pop();
                }
                // `/..` is `/`: nothing sits above a root, so the component
                // names nothing and is dropped rather than kept -- the
                // `i == 0 and prefix.endswith(sep)` arm of `_posix_rnormpath`,
                // and the same arm of `_nt_rnormpath` for the drive-rooted
                // spelling.
                Some(Component::RootDir) => {}
                // A relative path may open with `..`, `../..` keeps both, and
                // a drive-relative `C:..` keeps its own: popping a prefix
                // would rewrite which volume the path names.
                _ => out.push(component),
            },
            other => out.push(other),
        }
    }
    if out.as_os_str().is_empty() {
        out.push(Component::CurDir);
    }
    restore_double_root(path, out)
}

/// A path opening with exactly two slashes is reserved for the host to
/// interpret, so `//host/bin` keeps both while `///x` collapses to one
/// (the `initial_slashes == 2` arm of `_posix_rnormpath`).  `Path::components`
/// yields a single `RootDir` either
/// way, so the distinction has to be read off the original spelling and put
/// back afterwards.
#[cfg(all(
    feature = "host_env",
    not(feature = "sandbox"),
    not(target_arch = "wasm32"),
    unix
))]
fn restore_double_root(original: &Path, normalized: PathBuf) -> PathBuf {
    use std::os::unix::ffi::OsStrExt;

    let bytes = original.as_os_str().as_bytes();
    if !bytes.starts_with(b"//") || bytes.starts_with(b"///") {
        return normalized;
    }
    let mut doubled = std::ffi::OsString::from("/");
    doubled.push(normalized.as_os_str());
    PathBuf::from(doubled)
}

/// Windows carries its own leading separators in `Component::Prefix`, which
/// `Path::push` spells back out, so nothing is left to restore.
#[cfg(all(
    feature = "host_env",
    not(feature = "sandbox"),
    not(target_arch = "wasm32"),
    not(unix)
))]
fn restore_double_root(_original: &Path, normalized: PathBuf) -> PathBuf {
    normalized
}

/// Return the executable spelling used to invoke pyre, made absolute without
/// resolving symlinks.  `std::env::current_exe()` resolves the process image on
/// several hosts and would turn `venv/bin/pyre -> base/bin/pyre` back into the
/// base path before we get a chance to inspect the venv's `pyvenv.cfg`.
///
/// PyPy equivalent: `initpath.py:find_executable` searches PATH and applies
/// `abspath`, while `resolvedirof` follows links only for stdlib discovery.
#[cfg(all(
    feature = "host_env",
    not(feature = "sandbox"),
    not(target_arch = "wasm32")
))]
fn find_invoked_executable() -> PathBuf {
    let argv0 = SYS_ORIG_ARGV.lock().first().cloned();
    if let Some(argv0) = argv0 {
        let mut candidate = PathBuf::from(argv0);
        #[cfg(windows)]
        if candidate.extension().is_none() {
            candidate.set_extension("exe");
        }
        let cwd = host_os::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        if candidate.components().count() > 1 || candidate.is_absolute() {
            let absolute = absolute_from(candidate, &cwd);
            if exists_and_is_executable(&absolute) {
                return absolute;
            }
        } else if let Some(path) = std::env::var_os("PATH").and_then(|path| {
            std::env::split_paths(&path)
                .map(|dir| absolute_from(dir.join(&candidate), &cwd))
                .find(|path| exists_and_is_executable(path))
        }) {
            return path;
        }
        // PyPy deliberately exposes an empty sys.executable for an argv[0]
        // which cannot be resolved to an executable (CPython issue #7774).
        return PathBuf::new();
    }
    std::env::current_exe().unwrap_or_else(|_| PathBuf::from("pyre"))
}

/// Recognize both PyPy-style source trees (`lib-python/3`) and the packaged
/// Pyre layout (`lib/pyre3.14t`).  The latter is the installation shape pip and
/// venv will consume; accepting the former keeps the repository executable as
/// the untranslated/development oracle, matching PyPy's two
/// `compute_stdlib_path_*` arms.
#[cfg(all(
    feature = "host_env",
    not(feature = "sandbox"),
    not(target_arch = "wasm32")
))]
fn stdlib_at_prefix(prefix: &Path) -> Option<(Vec<PathBuf>, Option<PathBuf>)> {
    // `compute_stdlib_path_packaged` and `_sourcetree` both prefer the
    // versioned zip.  It belongs on sys.path, but it is not `sys._stdlib_dir`:
    // PyPy sets that attribute only for an entry containing a real `os.py`.
    let lib_pyzip = prefix.join("python314.zip");
    if lib_pyzip.is_file() {
        return Some((vec![lib_pyzip], None));
    }

    // pypy/module/sys/initpath.py `compute_stdlib_path_maybe_source` puts
    // `lib_pypy` on sys.path ahead of `lib-python/3`.  pyre does not: most of
    // `lib_pypy` is PyPy's cffi/pure-Python shims for modules pyre implements
    // natively or not at all (`_testcapi`, `_md5`, `_sha*`, `_sqlite3`, ...),
    // and the CPython suite branches on whether `import X` succeeds.  Exposing
    // a partial shim turns a clean skip into a run against a module that
    // cannot answer.  The supported surface is `lib-python/3` plus the builtin
    // modules: what `lib_pypy` still owns for PyPy is a builtin module here
    // (`_sysconfigdata`, `_sysconfig`, ...) rather than a Python file.
    let lib_python = prefix.join("lib-python").join("3");
    if lib_python.join("site.py").is_file() {
        let mut paths = vec![lib_python.clone()];
        append_macos_stdlib_paths(&mut paths, &lib_python);
        return Some((paths, Some(lib_python)));
    }

    // `compute_stdlib_path`: release packaging merges the two source trees
    // into one implementation-version directory.
    for packaged in [
        prefix.join("lib").join("pyre3.14t"),
        // cargo-dist's Homebrew formula installs non-binary archive contents
        // into `pkgshare` (`<keg>/share/pyrex`).  This is still a single
        // interpreter-owned prefix, analogous to PyPy's macOS bundle search
        // arm, and must not depend on a Homebrew-specific environment value.
        prefix
            .join("share")
            .join("pyrex")
            .join("lib")
            .join("pyre3.14t"),
    ] {
        if packaged.join("site.py").is_file() {
            let mut paths = vec![packaged.clone()];
            append_macos_stdlib_paths(&mut paths, &packaged);
            return Some((paths, Some(packaged)));
        }
    }
    #[cfg(windows)]
    {
        let candidate = prefix.join("Lib");
        if candidate.join("site.py").is_file() {
            return Some((vec![candidate.clone()], Some(candidate)));
        }
    }
    None
}

#[cfg(all(
    feature = "host_env",
    not(feature = "sandbox"),
    not(target_arch = "wasm32")
))]
fn append_macos_stdlib_paths(paths: &mut Vec<PathBuf>, stdlib: &Path) {
    // initpath.py appends both entries on macOS without probing them first.
    #[cfg(target_os = "macos")]
    {
        let plat_mac = stdlib.join("plat-mac");
        paths.push(plat_mac.clone());
        paths.push(plat_mac.join("lib-scriptpackages"));
    }
    #[cfg(not(target_os = "macos"))]
    let _ = (paths, stdlib);
}

/// Follow links whose final component is the executable while preserving the
/// spelling of its ancestor directories.  This is PyPy `resolvedirof`, not
/// `realpath`: canonicalizing the entire path would silently turn a macOS
/// `/var/...` venv into `/private/var/...` and make sys.prefix disagree with
/// pyvenv.cfg and site-packages paths.
#[cfg(all(
    feature = "host_env",
    not(feature = "sandbox"),
    not(target_arch = "wasm32")
))]
fn resolve_final_symlinks(path: &Path) -> PathBuf {
    let cwd = host_os::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut resolved = absolute_from(path.to_path_buf(), &cwd);
    loop {
        let Ok(metadata) = host_fs::symlink_metadata(&resolved) else {
            return resolved;
        };
        if !metadata.file_type().is_symlink() {
            return resolved;
        }
        let Ok(target) = std::fs::read_link(&resolved) else {
            return resolved;
        };
        resolved = if target.is_absolute() {
            target
        } else {
            resolved.parent().unwrap_or(Path::new("")).join(target)
        };
    }
}

/// Walk from an executable or `pyvenv.cfg` home directory to the first Pyre
/// stdlib and return both the stdlib and its owning installation prefix.
/// Symlinks are resolved for this search only; `sys.executable` retains the
/// invocation spelling above.
#[cfg(all(
    feature = "host_env",
    not(feature = "sandbox"),
    not(target_arch = "wasm32")
))]
fn find_stdlib_from(start: &Path) -> Option<(Vec<PathBuf>, Option<PathBuf>, PathBuf)> {
    if start.as_os_str().is_empty() {
        return None;
    }
    let resolved = resolve_final_symlinks(start);
    let mut dir = if resolved.is_dir() {
        Some(resolved.as_path())
    } else {
        resolved.parent()
    };
    while let Some(prefix) = dir {
        if let Some((paths, stdlib_dir)) = stdlib_at_prefix(prefix) {
            return Some((paths, stdlib_dir, prefix.to_path_buf()));
        }
        dir = prefix.parent();
    }
    None
}

#[cfg(all(
    feature = "host_env",
    not(feature = "sandbox"),
    not(target_arch = "wasm32")
))]
fn prefix_from_stdlib(stdlib: &Path) -> PathBuf {
    let parent = stdlib.parent();
    if stdlib
        .file_name()
        .is_some_and(|name| name == "python314.zip")
    {
        return parent.unwrap_or(stdlib).to_path_buf();
    }
    if stdlib.file_name().is_some_and(|name| name == "3")
        && parent
            .and_then(Path::file_name)
            .is_some_and(|name| name == "lib-python")
    {
        return parent
            .and_then(Path::parent)
            .unwrap_or(stdlib)
            .to_path_buf();
    }
    if parent
        .and_then(Path::file_name)
        .is_some_and(|name| name == "lib")
    {
        // cargo-dist Homebrew data lives at
        // `<keg>/share/pyrex/lib/pyre3.14t`, while the executable still lives
        // at `<keg>/bin/pyre`.  An explicit PYRE_STDLIB pointing at that data
        // must recover the keg, not claim `<keg>/share/pyrex` as sys.prefix.
        if let Some(pyre_share) = parent.and_then(Path::parent)
            && pyre_share.file_name().is_some_and(|name| name == "pyrex")
            && pyre_share
                .parent()
                .and_then(Path::file_name)
                .is_some_and(|name| name == "share")
        {
            return pyre_share
                .parent()
                .and_then(Path::parent)
                .unwrap_or(stdlib)
                .to_path_buf();
        }
        return parent
            .and_then(Path::parent)
            .unwrap_or(stdlib)
            .to_path_buf();
    }
    if stdlib.file_name().is_some_and(|name| name == "Lib") {
        return parent.unwrap_or(stdlib).to_path_buf();
    }
    parent.unwrap_or(stdlib).to_path_buf()
}

#[cfg(all(
    feature = "host_env",
    not(feature = "sandbox"),
    not(target_arch = "wasm32")
))]
#[derive(Clone, Debug)]
pub(crate) struct PthConfig {
    /// The directory holding the file.  Every relative line resolves against
    /// it and it is the prefix the interpreter goes on to report.
    pub dir: PathBuf,
    /// One entry per usable line, in the file's order; duplicates are kept
    /// because the file is the search path verbatim.
    pub entries: Vec<PathBuf>,
    /// Whether a line asked for `site` back.
    pub import_site: bool,
}

/// The name a `._pth` beside `executable` would have.
///
/// Windows drops an `exe` or `dll` extension first, so the file beside
/// `python.exe` is `python._pth`; every other platform appends to the whole
/// name, leaving `python3.14._pth`.
#[cfg(all(
    feature = "host_env",
    not(feature = "sandbox"),
    not(target_arch = "wasm32")
))]
fn pth_path_beside(executable: &Path) -> Option<PathBuf> {
    let parent = executable.parent()?;
    let mut name = executable.file_name()?.to_os_string();
    #[cfg(windows)]
    if executable
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("exe") || ext.eq_ignore_ascii_case("dll"))
        && let Some(stem) = executable.file_stem()
    {
        name = stem.to_os_string();
    }
    name.push("._pth");
    Some(parent.join(name))
}

/// The `._pth` beside the executable, parsed.
///
/// A `._pth` replaces the whole path computation: the file lists every
/// `sys.path` entry, the directory holding it is the prefix, the environment
/// is not consulted, no `sys.path[0]` is prepended, and `site` is skipped
/// unless a line reads `import site`.  A line is taken up to its first `#` and
/// then stripped; what is left empty is dropped.  `PYTHONHOME` does not
/// suppress the search — only an embedder calling `Py_SetPythonHome` does, and
/// there is no such caller here.
///
/// PyPy's `initpath` computes the search path from the executable alone and
/// has no `._pth` at all, so there is no shape to follow there.  3.14 has one
/// and `test_site._pthFileTests` asserts it, so it is 3.14's, measured against
/// 3.14.2 on this host over the eleven cases the parser distinguishes.
///
/// The search visits the interpreter library, the invoked executable and the
/// process image in that order.  This interpreter is one static executable —
/// which is what `sys.dllhandle` being 0 says, `GetModuleFileName(NULL)`
/// naming the executable itself — so the library and the first executable are
/// the same file, leaving two candidates.
#[cfg(all(
    feature = "host_env",
    not(feature = "sandbox"),
    not(target_arch = "wasm32")
))]
fn read_pth_config(executable: &Path) -> Option<PthConfig> {
    let image = std::env::current_exe().ok();
    for candidate in [Some(executable.to_path_buf()), image]
        .into_iter()
        .flatten()
    {
        let Some(path) = pth_path_beside(&candidate) else {
            continue;
        };
        let Ok(mut contents) = host_fs::read(&path) else {
            continue;
        };
        // The bounded bootstrap read `pyvenv.cfg` takes is 16 KiB.  A `._pth`
        // is deliberately longer — `test_site` writes one of about 32 KB,
        // which is the size gh-113628 was about — so the bound here is larger,
        // and is still a bound.
        contents.truncate(1024 * 1024);
        let dir = path.parent().unwrap_or_else(|| Path::new("")).to_path_buf();
        let dir = std::path::absolute(&dir).unwrap_or(dir);
        return Some(parse_pth(dir, &String::from_utf8_lossy(&contents)));
    }
    None
}

#[cfg(all(
    feature = "host_env",
    not(feature = "sandbox"),
    not(target_arch = "wasm32")
))]
fn parse_pth(dir: PathBuf, text: &str) -> PthConfig {
    let mut config = PthConfig {
        dir,
        entries: Vec::new(),
        import_site: false,
    };
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if line == "import site" {
            config.import_site = true;
        } else if line.starts_with("import ") {
            eprintln!("unsupported 'import' line in ._pth file");
        } else {
            // The entry is `abspath(join(pth_dir, line))`: an absolute line
            // replaces the directory, and either way the result is collapsed
            // lexically, so `..` walks up a level the file never has to spell
            // out.
            let entry = normalize_lexically(&config.dir.join(line));
            config.entries.push(entry);
        }
    }
    config
}

/// Fold a `._pth` beside the executable into the launcher options.
///
/// Its presence is a configuration in itself, and `sys.flags` has to report
/// that before the first `import sys`.  `-S` still wins over an `import site`
/// line: the option was named and the file only declines to remove `site`.
#[cfg(all(
    feature = "host_env",
    not(feature = "sandbox"),
    not(target_arch = "wasm32")
))]
pub fn apply_pth_config(flags: &mut crate::launch_env::LaunchFlags) {
    let Some(pth) = startup_path_config().pth.as_ref() else {
        return;
    };
    flags.isolated = true;
    flags.ignore_environment = true;
    flags.safe_path = true;
    flags.no_site = flags.no_site || !pth.import_site;
}

/// A build with no host filesystem has no file to find.
#[cfg(not(all(
    feature = "host_env",
    not(feature = "sandbox"),
    not(target_arch = "wasm32")
)))]
pub fn apply_pth_config(_flags: &mut crate::launch_env::LaunchFlags) {}

#[cfg(all(
    feature = "host_env",
    not(feature = "sandbox"),
    not(target_arch = "wasm32")
))]
fn read_pyvenv_config(executable: &Path) -> Option<(PathBuf, PyVenvConfig)> {
    let exe_dir = executable.parent()?;
    for candidate in [
        exe_dir.join("pyvenv.cfg"),
        exe_dir.parent()?.join("pyvenv.cfg"),
    ] {
        if let Ok(mut contents) = host_fs::read(&candidate) {
            // PyPy performs one bounded 16 KiB bootstrap read.  Do not let an
            // unbounded config file become startup input before stdlib setup.
            contents.truncate(16 * 1024);
            return Some((
                candidate,
                parse_pyvenv_cfg(&String::from_utf8_lossy(&contents)),
            ));
        }
    }
    None
}

#[cfg(all(
    feature = "host_env",
    not(feature = "sandbox"),
    not(target_arch = "wasm32")
))]
fn compute_startup_path_config_from(
    executable: PathBuf,
    explicit_stdlib: Option<PathBuf>,
) -> StartupPathConfig {
    // A `._pth` answers the whole question, so neither `pyvenv.cfg` nor the
    // landmark search below is consulted.  `stdlib_paths` is the file's list
    // verbatim: `ensure_stdlib_path` appends exactly that and nothing else,
    // and `apply_pth_config` has already suppressed the `PYTHONPATH` seed and
    // the `sys.path[0]` entry.
    if let Some(pth) = read_pth_config(&executable) {
        return StartupPathConfig {
            base_executable: executable.clone(),
            executable,
            prefix: pth.dir.clone(),
            base_prefix: pth.dir.clone(),
            stdlib_paths: pth.entries.clone(),
            // `sys._stdlib_dir` names one directory rather than the search
            // list, and it is the directory frozen modules read their sources
            // out of, so it has to be one that exists.  3.14 spells it
            // `<prefix>/Lib` without looking, which is a directory this layout
            // does not have under either name; the prefix probe is the same
            // question asked the way this tree answers it, and an entry
            // holding `site.py` answers a file that points elsewhere.
            stdlib: stdlib_at_prefix(&pth.dir)
                .and_then(|(_, stdlib_dir)| stdlib_dir)
                .or_else(|| {
                    pth.entries
                        .iter()
                        .find(|entry| entry.join("site.py").is_file())
                        .cloned()
                }),
            pth: Some(pth),
        };
    }

    let pyvenv = read_pyvenv_config(&executable);

    let base_executable = pyvenv
        .as_ref()
        .and_then(|(_, config)| config.executable.clone())
        .or_else(|| pyvenv.as_ref().map(|_| resolve_final_symlinks(&executable)))
        .unwrap_or_else(|| executable.clone());

    let (stdlib_paths, stdlib, base_prefix) =
        if let Some(stdlib_path) = explicit_stdlib.filter(|path| path.is_dir() || path.is_file()) {
            let prefix = prefix_from_stdlib(&stdlib_path);
            let stdlib_dir = stdlib_path.is_dir().then(|| stdlib_path.clone());
            // Naming the stdlib directory says where it is, not that it is the
            // only entry, so reproduce the shape `stdlib_at_prefix` builds.
            let mut paths = vec![stdlib_path];
            if let Some(stdlib_dir) = stdlib_dir.as_deref() {
                append_macos_stdlib_paths(&mut paths, stdlib_dir);
            }
            (paths, stdlib_dir, prefix)
        } else {
            // CPython 3.14's venv config records the exact base executable.  PyPy
            // records `home`.  Try both, in that order, then the invoked image;
            // `find_stdlib_from` resolves links only for this search.
            let mut starts = Vec::new();
            starts.push(base_executable.clone());
            if let Some(home) = pyvenv.as_ref().and_then(|(_, config)| config.home.clone())
                && !starts.contains(&home)
            {
                starts.push(home);
            }
            if !starts.contains(&executable) {
                starts.push(executable.clone());
            }
            // `find_invoked_executable` reports an unresolvable argv[0] as an
            // empty path, which is the right spelling for `sys.executable` but
            // says nothing about where the stdlib is.  The process image still
            // knows, so search from it before giving up: without this a caller
            // that passes a made-up argv[0] (the `-i` REPL spawned by
            // `test_inspect`, whose argv[0] is `<stdin>`) gets `sys.path ==
            // ['']` and cannot import anything at all.
            if let Ok(image) = std::env::current_exe()
                && !starts.contains(&image)
            {
                starts.push(image);
            }
            starts
                .iter()
                .find_map(|start| find_stdlib_from(start))
                .map(|(paths, stdlib_dir, prefix)| (paths, stdlib_dir, prefix))
                .unwrap_or_else(|| (Vec::new(), None, PathBuf::new()))
        };

    // `venv` writes pyvenv.cfg at the environment root (one directory above
    // `bin`).  CPython 3.14 path initialization sets the venv prefix before
    // importing site; site.py now only validates it and emits a warning on a
    // mismatch.  For the compatibility form beside the executable, the file's
    // own parent is likewise the environment prefix.
    let prefix = pyvenv
        .as_ref()
        .and_then(|(config_path, _)| config_path.parent())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| base_prefix.clone());

    StartupPathConfig {
        executable,
        base_executable,
        prefix,
        base_prefix,
        #[cfg(all(feature = "host_env", not(target_arch = "wasm32")))]
        stdlib_paths,
        stdlib,
        pth: None,
    }
}

#[cfg(all(
    feature = "host_env",
    not(feature = "sandbox"),
    not(target_arch = "wasm32")
))]
fn compute_startup_path_config() -> StartupPathConfig {
    compute_startup_path_config_from(
        find_invoked_executable(),
        std::env::var_os("PYRE_STDLIB").map(PathBuf::from),
    )
}

#[cfg(all(feature = "sandbox", not(target_arch = "wasm32")))]
fn compute_startup_path_config() -> StartupPathConfig {
    use std::os::unix::ffi::OsStrExt;
    let executable = PathBuf::from("/bin/pyre");
    let stdlib = crate::host_seam::ops::getenv(b"PYRE_STDLIB")
        .ok()
        .flatten()
        .map(|bytes| PathBuf::from(std::ffi::OsStr::from_bytes(&bytes)));
    StartupPathConfig {
        base_executable: executable.clone(),
        prefix: PathBuf::from("/bin"),
        base_prefix: PathBuf::from("/bin"),
        executable,
        #[cfg(all(feature = "host_env", not(target_arch = "wasm32")))]
        stdlib_paths: stdlib.iter().cloned().collect(),
        stdlib,
    }
}

#[cfg(all(feature = "host_env", target_arch = "wasm32"))]
fn compute_startup_path_config() -> StartupPathConfig {
    // The wasm runners install their VFS-backed path explicitly.  There is no
    // process executable or host prefix to discover inside the guest.
    StartupPathConfig {
        executable: PathBuf::new(),
        base_executable: PathBuf::new(),
        prefix: PathBuf::new(),
        base_prefix: PathBuf::new(),
        #[cfg(all(feature = "host_env", not(target_arch = "wasm32")))]
        stdlib_paths: Vec::new(),
        stdlib: None,
    }
}

#[cfg(feature = "host_env")]
pub(crate) fn startup_path_config() -> &'static StartupPathConfig {
    STARTUP_PATH_CONFIG.get_or_init(compute_startup_path_config)
}

/// Resolve the stdlib directory to add to `sys.path`.
///
/// Order: the `PYRE_STDLIB` override, then PyPy's source and packaged layouts
/// relative to the executable (or its venv base).  A host Python stdlib is
/// intentionally never used: its bytecode, `_sre`, sysconfig and platform
/// assumptions belong to a different interpreter.
///
/// PyPy equivalent: initpath.py scans for lib-python/X.Y at startup.
#[cfg(feature = "host_env")]
pub(crate) fn detect_stdlib_path() -> Option<PathBuf> {
    startup_path_config().stdlib.clone()
}

/// Add a directory to `sys.path`.
///
/// Before the `sys` module exists this stages the entry in the native
/// `SYS_PATH` seed, which is flushed into `sys.path` when `sys` is created.
/// Once `sys` exists the Python list is authoritative, so the entry is appended
/// to it in place (deduplicated) and the spent seed is left untouched. A
/// missing or non-list `sys.path` (e.g. after `del sys.path`) is respected.
#[cfg(feature = "host_env")]
pub fn add_sys_path(dir: &Path) {
    use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};

    let entry = crate::gateway::fsdecode_os_str_wtf8(dir.as_os_str());
    if get_interpreter_sys_module().is_none() {
        {
            let mut path = SYS_PATH.lock();
            let pb = dir.to_path_buf();
            if !path.contains(&pb) {
                path.push(pb);
            }
        }
        return;
    }
    // Pin the new entry before any further allocation (`get_sys_module` and the
    // dict lookup allocate) can relocate it. After the list is fetched below the
    // path is allocation-free until the pinned entry reaches `w_list_append`.
    let _roots = push_roots();
    let slot = shadow_stack_len();
    let _ = pin_root(pyre_object::w_str_from_wtf8_managed(entry.clone()));
    let Some(sys_mod) = get_interpreter_sys_module() else {
        return;
    };
    let w_dict = unsafe { pyre_object::w_module_get_w_dict(sys_mod) };
    if w_dict.is_null() {
        return;
    }
    let Some(w_path) = (unsafe { pyre_object::w_dict_getitem_str(w_dict, "path") }) else {
        return;
    };
    if !unsafe { pyre_object::is_list(w_path) } {
        return;
    }
    let n = unsafe { pyre_object::listobject::w_list_len(w_path) };
    for i in 0..n {
        if let Some(item) = unsafe { pyre_object::listobject::w_list_getitem(w_path, i as i64) } {
            // Both spellings can hold a surrogate escape, so the two are
            // compared as the WTF-8 they are rather than through `&str`.
            if unsafe { pyre_object::is_str(item) }
                && unsafe { pyre_object::w_str_get_wtf8(item) }.as_bytes() == entry.as_bytes()
            {
                return;
            }
        }
    }
    unsafe { pyre_object::listobject::w_list_append(w_path, shadow_stack_get(slot)) };
}

/// `pymain_sys_path_add_path0` — prepend the startup `sys.path[0]` entry staged
/// by `init_sys_path`.  Called after `site` has run so `removeduppaths` cannot
/// absolutize an empty entry into the cwd, and inserted unconditionally (no
/// dedup).  The entry is taken, so a second call is a no-op.
#[cfg(feature = "host_env")]
pub fn add_sys_path_0() {
    use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};

    let Some(entry) = SYS_PATH_0_PENDING.lock().take() else {
        return;
    };
    // No `sys` yet (an embedder that never imports `site`): stage at the front
    // of the seed instead, which `create_sys_path_list` flushes in order.
    if get_interpreter_sys_module().is_none() {
        SYS_PATH.lock().insert(0, PathBuf::from(entry));
        return;
    }
    // Pin the new entry before any further allocation (`get_sys_module` and the
    // dict lookup allocate) can relocate it — `add_sys_path` parity.
    let _roots = push_roots();
    let slot = shadow_stack_len();
    let _ = pin_root(crate::gateway::fsdecode_os_str(&entry));
    let Some(sys_mod) = get_interpreter_sys_module() else {
        return;
    };
    let w_dict = unsafe { pyre_object::w_module_get_w_dict(sys_mod) };
    if w_dict.is_null() {
        return;
    }
    let Some(w_path) = (unsafe { pyre_object::w_dict_getitem_str(w_dict, "path") }) else {
        return;
    };
    if !unsafe { pyre_object::is_list(w_path) } {
        return;
    }
    unsafe { pyre_object::listobject::w_list_insert(w_path, 0, shadow_stack_get(slot)) };
}

// ── check_sys_modules ────────────────────────────────────────────────
// PyPy equivalent: importing.py `check_sys_modules(space, w_modulename)`

/// PyPy `importing.py:check_sys_modules` is an ordinary traceable lookup.
/// The mutable `sys.modules` dictionary supplies the invalidation boundary;
/// hiding this whole function behind `dont_look_inside` turns the cached
/// import fast path into an `EF_RANDOM_EFFECTS` residual and prevents the
/// optimizer from seeing the dictionary read at all.
pub(crate) fn check_sys_modules(name: &str) -> Option<PyObjectRef> {
    // Once installed, the Python-visible dict is the sole semantic module
    // cache.  PyPy's `check_sys_modules` reads `space.sys.get('modules')` and
    // a deleted key is genuinely absent; the process-owned registry below is
    // only the bootstrap owner/root used before that dict exists, and the
    // `None` sentinel is something only a running program writes, so the
    // registry has none to drop.
    let dict = sys_modules_dict();
    if !dict.is_null() {
        // No `Option::filter`: the closure is an un-lowered synthetic ctor
        // and is the descent-scan blocker on `__import__`.
        let Some(m) = sys_modules_dict_entry(dict, name) else {
            return None;
        };
        if unsafe { pyre_object::is_none(m) } {
            return None;
        }
        return Some(m);
    }
    sys_modules_registry_get(name)
}

/// `PyImport_GetModule` — whatever `sys.modules` holds under `name`, sentinel
/// included.
///
/// A reader that is not importing owes the program the refusal the binding
/// itself states: `sys.displayhook` hands the entry to `setattr`, and a name
/// bound to `None` is refused there rather than treated as absent.
pub(crate) fn sys_modules_entry(name: &str) -> Option<PyObjectRef> {
    let dict = sys_modules_dict();
    if dict.is_null() {
        return sys_modules_registry_get(name);
    }
    sys_modules_dict_entry(dict, name)
}

/// The `sys.modules` dict read both readers above share: a missing key and a
/// null are the same absence.
fn sys_modules_dict_entry(dict: PyObjectRef, name: &str) -> Option<PyObjectRef> {
    // `interp_import.py _gcd_import` uses `space.finditem_str`.  Spell that
    // here so the jitted arm of `finditem_str_named` (no DictOperationGuard)
    // is the one the `__import__` walk records.  No `Option::filter`: the
    // closure is an un-lowered synthetic ctor the descent scan declines on.
    // The `Result` match on `finditem_str` residualises as
    // `from_exc_object` (`__majit_stringbuilder_new`).  Keep the
    // probe itself look-inside-adjacent: this wrapper is the one
    // residual, so the exception conversion is not on the
    // `__import__` look-inside graph.
    sys_modules_finditem_str(dict, name)
}

/// `finditem_str` plus the miss/`Err` swallow.  Hidden so
/// `from_exc_object` stays off the look-inside `__import__` body.
#[majit_macros::dont_look_inside]
pub(crate) fn sys_modules_finditem_str(dict: PyObjectRef, name: &str) -> Option<PyObjectRef> {
    match crate::baseobjspace::finditem_str(dict, name) {
        Ok(Some(m)) if !m.is_null() => Some(m),
        _ => None,
    }
}

/// Sentinel a 1-word lookup residual returns after
/// [`crate::runtime_ops::jit_publish_residual_error`].  A static byte
/// whose address is a real pointer (not the integer 1), compared with
/// `ptr::eq` so the look-inside graph never `cast_ptr_to_int`.
#[majit_macros::dont_look_inside_cannot_raise]
pub(crate) fn import_lookup_err_ptr() -> PyObjectRef {
    static SENTINEL: u8 = 0;
    std::ptr::from_ref(&SENTINEL) as *const u8 as PyObjectRef
}

fn import_lookup_is_err(p: PyObjectRef) -> bool {
    std::ptr::eq(p, import_lookup_err_ptr())
}

/// Object-keyed `finditem`.  Hidden so `from_exc_object` stays off the
/// look-inside `__import__` body.  Two object arguments and a nullable
/// object result: the walker can execute this residual.  A raise is
/// published and returned as [`import_lookup_err_ptr`], not swallowed.
#[majit_macros::dont_look_inside]
pub(crate) fn sys_modules_finditem_w(dict: PyObjectRef, w_name: PyObjectRef) -> PyObjectRef {
    match crate::baseobjspace::finditem(dict, w_name) {
        Ok(Some(m)) if !m.is_null() => m,
        Ok(_) => pyre_object::PY_NULL,
        Err(e) => {
            crate::runtime_ops::jit_publish_residual_error(e);
            import_lookup_err_ptr()
        }
    }
}

/// Exact-str `sys.modules` probe: precomputed-hash `getitem_str`, no
/// `DictOperationGuard`.  Cannot raise on an interned str key, so the
/// residual is `CallR` rather than `CallMayForceR`.
#[majit_macros::dont_look_inside_cannot_raise]
pub(crate) fn sys_modules_finditem_str_exact(
    dict: PyObjectRef,
    w_name: PyObjectRef,
) -> PyObjectRef {
    let Some(key) = (unsafe { pyre_object::w_str_get_value_opt(w_name) }) else {
        return pyre_object::PY_NULL;
    };
    let strategy = unsafe { pyre_object::dictmultiobject::w_module_dict_strategy_or_null(dict) };
    if !strategy.is_null() {
        return unsafe { (*strategy).getitem_str(dict, key) }.unwrap_or(pyre_object::PY_NULL);
    }
    let hash = unsafe { pyre_object::w_str_hash_memoized(w_name) };
    if majit_metainterp::jit::we_are_jitted() {
        return unsafe {
            pyre_object::dictmultiobject::w_dict_get_strategy(dict)
                .getitem_str_hashed(dict, key, hash)
        }
        .unwrap_or(pyre_object::PY_NULL);
    }
    match unsafe {
        pyre_object::dictmultiobject::w_dict_getitem_str_checked_hashed(dict, key, hash)
    } {
        Ok(Some(m)) if !m.is_null() => m,
        Ok(_) => pyre_object::PY_NULL,
        Err(_) => {
            crate::runtime_ops::jit_publish_residual_error(
                crate::baseobjspace::take_pending_dict_key_error(w_name),
            );
            import_lookup_err_ptr()
        }
    }
}

/// `check_sys_modules` for a caller that already holds the name object.
///
/// An exact str uses [`sys_modules_finditem_str_exact`]; any other key
/// keeps the generic [`sys_modules_finditem_w`] residual.
fn check_sys_modules_import(w_name: PyObjectRef) -> Result<Option<PyObjectRef>, crate::PyError> {
    let dict = sys_modules_dict();
    if !dict.is_null() {
        let m = if unsafe { pyre_object::is_str(w_name) } {
            sys_modules_finditem_str_exact(dict, w_name)
        } else {
            sys_modules_finditem_w(dict, w_name)
        };
        if import_lookup_is_err(m) {
            return Err(take_published_residual_error());
        }
        if m.is_null() || unsafe { pyre_object::is_none(m) } {
            return Ok(None);
        }
        return Ok(Some(m));
    }
    let Some(name) = (unsafe { pyre_object::w_str_get_value_opt(w_name) }) else {
        return Ok(None);
    };
    Ok(sys_modules_registry_get(name))
}

/// Look `name` up in `SYS_MODULES`, the process-owned name→module registry.
///
/// The single read seam every traced reader of `SYS_MODULES` goes through.
/// Entries are stamped at runtime by `set_sys_module` / `remove_sys_module`,
/// so the map is not a build-time constant and the JIT residualizes the read
/// instead of tracing into it (`@dont_look_inside`, the `sys_modules_dict`
/// shape).  The mutating and GC-walking users keep the raw static.
///
/// The read takes the guard unconditionally, following the
/// `get_interpreter_sys_module` spelling this replaces: `parking_lot` hands a
/// panicking thread's lock straight on, and a `HashMap<String, usize>` has no
/// invariant a panic mid-`insert`/`remove` could leave broken.
#[majit_macros::dont_look_inside]
/// `importing.py check_sys_modules(space, w_modulename)` — the object-keyed
/// half of the pair whose `&str` half is [`check_sys_modules`].  A module name
/// is WTF-8 and may hold a lone surrogate, which has no `&str` spelling at all,
/// so a caller that already holds the name object asks with it.
pub(crate) fn check_sys_modules_w(w_name: PyObjectRef) -> Option<PyObjectRef> {
    let dict = sys_modules_dict();
    if dict.is_null() {
        // Before the Python-visible dict exists the process registry is the
        // cache, and it is keyed by `&str`; a name with no UTF-8 spelling
        // cannot be in it.
        let name = unsafe { pyre_object::w_str_get_value_opt(w_name) }?;
        return sys_modules_registry_get(name);
    }
    unsafe { pyre_object::w_dict_lookup(dict, w_name) }
        .filter(|m| !m.is_null() && !unsafe { pyre_object::is_none(*m) })
}

/// Bootstrap name→module map.  PyPy has no such registry on the
/// `check_sys_modules` look-inside path; the Python-visible dict is the
/// cache.  Residualise the `HashMap` so a red `sys_modules_dict()` null
/// check does not drag lock/hash helpers into `__import__`.
#[majit_macros::dont_look_inside]
pub(crate) fn sys_modules_registry_get(name: &str) -> Option<PyObjectRef> {
    match SYS_MODULES.lock().get(name).copied() {
        Some(module) => Some(module as PyObjectRef),
        None => None,
    }
}

/// Whether `sys.modules[name]` is bound to `None`, the sentinel that marks
/// a name as blocked.
///
/// `_bootstrap._find_and_load` treats it as a cached "this import must not
/// succeed" and raises instead of searching, which is how code disables an
/// import for everything downstream of it.  `check_sys_modules` cannot
/// report it — `None` is not a module it can hand back, so it falls through
/// to the search — hence the separate lookup.
fn sys_modules_blocks(name: &str) -> bool {
    sys_modules_blocks_key(pyre_object::w_str_new_managed(name))
}

/// [`sys_modules_blocks`] for a name with no `&str` spelling.  A module name
/// is WTF-8 and may carry a lone surrogate, and `sys.modules` is free to hold
/// such a name as a key, so the block is asked for as an object.
fn sys_modules_blocks_wtf8(name: &Wtf8) -> bool {
    sys_modules_blocks_key(pyre_object::w_str_from_wtf8_managed(name.to_owned()))
}

fn sys_modules_blocks_key(key: PyObjectRef) -> bool {
    // The key is built before the dict is read, the order `check_sys_modules`
    // uses: reading the thread-local last keeps the borrow off the stack
    // across the allocation.
    let dict = sys_modules_dict();
    if dict.is_null() {
        return false;
    }
    match unsafe { pyre_object::w_dict_lookup(dict, key) } {
        Some(m) => !m.is_null() && unsafe { pyre_object::is_none(m) },
        None => false,
    }
}

/// Look up a loaded module by name through [`check_sys_modules`].
pub fn get_sys_module(name: &str) -> Option<PyObjectRef> {
    check_sys_modules(name)
}

/// Startup registers `_frozen_importlib`. A test that blocks that name and
/// loads the source copy leaves the same module at `importlib._bootstrap`.
pub fn importlib_bootstrap_module() -> Option<PyObjectRef> {
    get_sys_module("_frozen_importlib").or_else(|| get_sys_module("importlib._bootstrap"))
}

/// Same pair as [`importlib_bootstrap_module`] for the external bootstrap.
pub fn importlib_bootstrap_external_module() -> Option<PyObjectRef> {
    get_sys_module("_frozen_importlib_external")
        .or_else(|| get_sys_module("importlib._bootstrap_external"))
}

/// `space.getbuiltinmodule(name)` — the module object for a registered builtin,
/// minted on the first ask.
///
/// `sys.modules` holds what has been imported; `space.builtin_modules` holds
/// every builtin the space was built with, and `getbuiltinmodule` creates the
/// module the first time one is named.  An interpreter-level reader of pyre's
/// own builtin needs the second: `_io`'s newline decoder is an app-level class
/// that lives on the `_io` module object and nowhere else, so a program that
/// never imported `_io` would otherwise not be able to open a text file.
pub fn get_builtin_module(name: &str) -> Option<PyObjectRef> {
    // Minting runs the module's `startup` hook, which is handed the execution
    // context and may import through it — `array`'s registers the type with
    // `_collections_abc.MutableSequence`.  The live context is what
    // `getbuiltinmodule` would have run under anyway, so it is what this
    // reader hands over rather than a null nothing on that path may
    // dereference.
    getbuiltinmodule(name, false, true, crate::call::getexecutioncontext())
        .ok()
        .flatten()
}

/// Return the interpreter-owned `sys` module, bypassing the Python-visible
/// `sys.modules` mapping.
///
/// CPython's `_PySys_GetOptionalAttr*` reads `PyInterpreterState.sysdict`
/// directly, and PyPy reads `space.sys`; neither follows a replacement
/// `sys.modules["sys"]`.  `SYS_MODULES` is pyre's process/interpreter-owned
/// module registry and is independently walked as a GC root, so its original
/// `sys` entry is the corresponding owner.
pub fn get_interpreter_sys_module() -> Option<PyObjectRef> {
    sys_modules_registry_get("sys")
}

/// Return the interpreter-owned `builtins` module, bypassing the
/// Python-visible `sys.modules` mapping.
///
/// The other half of [`get_interpreter_sys_module`]: `PyInterpreterState`
/// carries `builtins` beside `sysdict`, and `finalize_modules_delete_special`
/// reads both from there rather than through a mapping the program owns.
/// The module this returns is the registry's, not
/// `ExecutionContext::get_builtin()`'s -- the two are the split described at
/// `builtin_modules_get`, and finalization wants the one whose dict
/// `sys.modules` published.
pub fn get_interpreter_builtins_module() -> Option<PyObjectRef> {
    sys_modules_registry_get("builtins")
}

/// The Python-visible `sys.modules` dict, or `PY_NULL` before it is
/// installed. Used by callers that need to iterate every loaded module
/// (e.g. pickle's `whichmodule` scan), and the single read seam every
/// traced reader of `SYS_MODULES_DICT` goes through.
///
/// The pointer is stamped at runtime by `set_sys_modules_dict`, so it is not
/// a build-time constant and the JIT residualizes the read instead of
/// tracing into it (`@dont_look_inside`, the `gc_interp::enabled` shape).
/// The `-> PyObjectRef` return fits a single word and it cannot raise.
#[majit_macros::dont_look_inside]
pub fn sys_modules_dict() -> PyObjectRef {
    SYS_MODULES_DICT.load(Ordering::Acquire) as PyObjectRef
}

pub fn set_sys_module(name: &str, module: PyObjectRef) {
    // Native-owned immortal modules need their managed dicts in
    // `walk_module_dicts_gc`; managed modules are instead reached through the
    // ordinary object graph rooted at `sys.modules`.  Test actual ownership,
    // not whether that dict currently exists: the immortal bootstrap `sys`
    // module installs `sys.modules` from inside its initializer, before this
    // function receives the finished module.
    pyre_object::gc_roots::mark_prebuilt_roots_dirty();
    if !majit_gc::gc_owns_object(module as usize) {
        MODULE_DICT_ROOTS.lock().insert(module as usize);
    }
    SYS_MODULES.lock().insert(name.to_string(), module as usize);
    // Keep the Python-visible sys.modules dict in sync.  The read is fresh —
    // the collector walks that cell — but the `dict` moves, and the key string
    // minted below can be the collection that moves it.
    let dict = sys_modules_dict();
    if !dict.is_null() {
        let roots = pyre_object::gc_roots::push_roots();
        let dict_slot = roots.base();
        let _ = roots.pin_root(dict);
        let key_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(pyre_object::w_str_new_managed(name));
        let module_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(module);
        unsafe {
            pyre_object::w_dict_store(
                roots.get(dict_slot),
                roots.get(key_slot),
                roots.get(module_slot),
            );
        }
    }
}

/// Remove a (partially initialised) module from `sys.modules`.
///
/// `importlib._bootstrap._load` deletes the module it pre-registered when
/// `exec_module` raises, so a retried import re-executes the body rather
/// than handing back a half-built module.  Without this a failed
/// `import ssl` (missing `_ssl`) leaves a broken `ssl` shell behind, and
/// the next `import ssl` succeeds with no `SSLWantReadError`, etc.
pub fn remove_sys_module(name: &str) {
    SYS_MODULES.lock().remove(name);
    let dict = sys_modules_dict();
    if !dict.is_null() {
        unsafe {
            pyre_object::w_dict_delitem_str(dict, name);
        }
    }
}

/// What [`release_sys_modules_for_shutdown`] detached from the import cache.
pub struct ReleasedSysModules {
    /// The real modules it found, newest last, each still referenced by the
    /// import cache's caller until it takes its own hold on them.
    pub modules: Vec<(Wtf8Buf, PyObjectRef)>,
}

/// `finalize_remove_modules`: snapshot real modules, then detach import-cache
/// entries so module/dict cycles can become unreachable during shutdown.
pub fn release_sys_modules_for_shutdown() -> ReleasedSysModules {
    let dict = sys_modules_dict();
    if dict.is_null() {
        return ReleasedSysModules {
            modules: Vec::new(),
        };
    }
    let entries = unsafe { pyre_object::w_dict_items(dict) };
    let mut modules = Vec::new();
    for (key, module) in entries {
        let name = if unsafe { pyre_object::is_str(key) } {
            Some(unsafe { pyre_object::w_str_get_wtf8(key) }.to_owned())
        } else {
            None
        };
        if !module.is_null() && unsafe { pyre_object::is_module(module) } {
            if let Some(name) = &name {
                modules.push((name.clone(), module));
            }
        }
        let keep_entry = name.as_deref().is_some_and(|name| {
            let bytes = name.as_bytes();
            bytes == b"sys" || bytes == b"builtins"
        });
        if !keep_entry {
            unsafe {
                pyre_object::w_dict_delitem(dict, key);
            }
        }
    }
    ReleasedSysModules { modules }
}

/// GC root walk over every bound module's wrapped name and dict storage.
///
/// The walk is keyed on [`MODULE_DICT_ROOTS`], not on the live name→module
/// map: a module that lost its name to a later import is still immortal and
/// still reachable from Python, so its dict must keep being marked.
///
/// Modules (`malloc_typed`) are Box-immortal, while their non-moving
/// `W_ModuleDictObject`s are GC-managed. Visit each `Module.w_name` and
/// `Module.w_dict` field first so their headers are marked and the dict's
/// custom trace can reach the
/// authoritative `dstorage` / `object_storage` / cell registry. A movable
/// value bound at module scope
/// — e.g. `gc.collect` reached through `gc.__dict__`, or any
/// module-level list / instance — would otherwise be read back stale
/// after a collection relocates it.  Treat each loaded module's dict as
/// a pinned root source so those slots stay forwarded.  This complements
/// the per-frame `w_globals` walk in `eval::walk_pyframe_roots`,
/// which additionally covers `exec`/`eval` globals dicts that are not
/// registered in `sys.modules`.
///
/// # Safety
/// `visitor` must tolerate being called on every movable module-dict
/// value slot reachable here.
pub unsafe fn walk_module_dicts_gc(visitor: &mut dyn FnMut(&mut PyObjectRef)) {
    unsafe { walk_bound_module_dicts(visitor) };
}

/// The shared body of the two module-dict root walks.
///
/// # Safety
/// `visitor` must tolerate being called on every movable module-dict value
/// slot reachable here.
unsafe fn walk_bound_module_dicts(visitor: &mut dyn FnMut(&mut PyObjectRef)) {
    for &module in MODULE_DICT_ROOTS.lock().iter() {
        let module = module as PyObjectRef;
        if module.is_null() || !unsafe { pyre_object::is_module(module) } {
            continue;
        }
        unsafe {
            let module = &mut *(module as *mut pyre_object::module::Module);
            visitor(&mut module.w_name);
            visitor(&mut module.w_dict);
            visitor(&mut module.w_initialdict);
            let w_dict = module.w_dict;
            pyre_object::dictmultiobject::w_module_dict_walk_gc_cells(w_dict, visitor);
        }
    }
}

/// `baseobjspace.py` — record the interp-level `__import__` as the space's
/// default importer.  Called where `builtins.__import__` is bound, which is
/// what `moduledef.py:87 startup` copies from.
///
/// The space captures one importer while it is being set up, so a namespace
/// built after that keeps the first one: `install_default_builtins` also runs
/// for a fresh builtins dict, and the module that dict belongs to is not the
/// one every frame resolves `__import__` through.
pub fn set_default_importlib_import(w_import: PyObjectRef) {
    if DEFAULT_IMPORTLIB_IMPORT
        .compare_exchange(0, w_import as usize, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
    {
        // Young `__import__` wrapper into a prebuilt space slot
        // (`incminimark.py remember_young_pointer_from_prebuilt`).
        pyre_object::gc_roots::mark_prebuilt_roots_dirty();
    }
}

/// `space.w_default_importlib_import`, or `None` before builtins is bound.
fn default_importlib_import() -> Option<PyObjectRef> {
    let w_import = DEFAULT_IMPORTLIB_IMPORT.load(Ordering::Acquire) as PyObjectRef;
    (!w_import.is_null()).then_some(w_import)
}

/// Word form of [`default_importlib_import`]. Null before builtins is bound.
/// An atomic load: no Python, no allocation.
#[majit_macros::dont_look_inside_cannot_raise]
pub(crate) fn default_importlib_import_word() -> PyObjectRef {
    DEFAULT_IMPORTLIB_IMPORT.load(Ordering::Acquire) as PyObjectRef
}

/// Forward the `sys.modules` dict pointer cached in `SYS_MODULES_DICT`.
///
/// The same dict object is also reachable as `sys.__dict__["modules"]`
/// (forwarded by [`walk_module_dicts_gc`]), but this fast-path cell holds an
/// independent raw copy.  `w_dict_new` allocates the dict in the movable
/// nursery, so a collection relocates it and leaves this cell pointing at the
/// vacated (reclaimed) slot; the next `check_sys_modules` would then run
/// `w_dict_lookup` against dead memory.  Forward the cell in place so it
/// tracks the relocation, mirroring the EC-slot forwarding in
/// `eval::walk_pyframe_roots`.
///
/// # Safety
/// `visitor` must tolerate a non-nursery or already-forwarded pointer.
pub unsafe fn walk_sys_modules_dict_gc(visitor: &mut dyn FnMut(&mut PyObjectRef)) {
    let mut dict = SYS_MODULES_DICT.load(Ordering::Acquire) as PyObjectRef;
    if dict.is_null() {
        return;
    }
    visitor(&mut dict);
    SYS_MODULES_DICT.store(dict as usize, Ordering::Release);
}

pub(crate) fn capture_import_root_area() -> *const () {
    IMPORT_ROOT_AREA.with(|area| area as *const _ as *const ())
}

/// # Safety
/// `data` must come from [`capture_import_root_area`], and the owning thread
/// must be quiesced.
pub(crate) unsafe fn walk_import_roots_area(
    data: *const (),
    visitor: &mut dyn FnMut(&mut PyObjectRef),
) {
    let area = unsafe { &*(data as *const ImportRootArea) };
    let argv_pending = unsafe { &*area.argv_pending };
    let mut argv = argv_pending.get();
    if !argv.is_null() {
        visitor(&mut argv);
        argv_pending.set(argv);
    }
}

/// The process-owned object pointers in the import state.
///
/// These slots are prebuilt (`space.sys` / `SysModuleState`). A mutator store
/// of a young pointer into them takes `mark_prebuilt_roots_dirty`
/// (`incminimark.py remember_young_pointer_from_prebuilt`); this walk is the
/// corresponding `prebuilt_root_objects` visit, reached only through the gated
/// [`walk_process_import_roots`]. Stores here rewrite forwarded addresses and
/// are not mutator stores.
unsafe fn walk_process_import_object_roots(visitor: &mut dyn FnMut(&mut PyObjectRef)) {
    let mut dict = SYS_MODULES_DICT.load(Ordering::Acquire) as PyObjectRef;
    if !dict.is_null() {
        visitor(&mut dict);
        SYS_MODULES_DICT.store(dict as usize, Ordering::Release);
    }
    let mut w_import = DEFAULT_IMPORTLIB_IMPORT.load(Ordering::Acquire) as PyObjectRef;
    if !w_import.is_null() {
        visitor(&mut w_import);
        DEFAULT_IMPORTLIB_IMPORT.store(w_import as usize, Ordering::Release);
    }
    // `space.builtin_modules`. An immortal module's fields are walked above,
    // since `getbuiltinmodule` binds it in `sys.modules` before anything
    // else; a managed one is traced through its own fields.
    for def in BUILTIN_MODULES.lock().values_mut() {
        if def.w_mod != 0 {
            let mut w_mod = def.w_mod as PyObjectRef;
            visitor(&mut w_mod);
            def.w_mod = w_mod as usize;
        }
    }
}

/// Walk the import state owned by PyPy's process/interpreter object space.
///
/// Unlike `SYS_ARGV_PENDING`, these slots are not thread-local.  Keeping this
/// walk separate lets the collector's `collect_nonstack_roots` parity rescan
/// them exactly once at the end of incremental marking.
pub(crate) unsafe fn walk_process_import_roots(visitor: &mut dyn FnMut(&mut PyObjectRef)) {
    // `space.sys.modules` is process-owned in PyPy.  STW has quiesced every
    // mutator, so the process-global cache cannot be semantically mutated
    // while this walk holds its native lock.
    unsafe { walk_bound_module_dicts(visitor) };
    walk_process_import_object_roots(visitor);
}

/// Set the Python-visible sys.modules dict reference. Called during sys
/// module initialization so subsequent set_sys_module calls keep it in sync.
/// Also copies all previously cached modules into the dict.
/// Set sys.argv from the arguments the host gave the process.
/// Must be called after the first `import sys` has run (e.g. after
/// `run_source` compiles the module-level code).
///
/// `targetpypystandalone.py:76-80` builds the list with `space.newfilename`,
/// which is `fsdecode(newbytes(s))`, so an argument the filesystem encoding
/// cannot spell arrives as the surrogate escape rather than being rejected or
/// replaced. The arguments stay in the host's own spelling until here for that
/// reason: a Rust `String` cannot hold the escape.
pub fn set_sys_argv(args: &[std::ffi::OsString]) {
    let items: Vec<pyre_object::PyObjectRef> = args
        .iter()
        .map(|s| crate::gateway::fsdecode_os_str(s))
        .collect();
    let argv = pyre_object::w_list_new(items);
    SYS_ARGV_PENDING.with(|p| p.set(argv));
}

thread_local! {
    static SYS_ARGV_PENDING: std::cell::Cell<pyre_object::PyObjectRef> =
        const { std::cell::Cell::new(pyre_object::PY_NULL) };
}

// The launcher flags belong to the interpreter, not to whichever thread
// happened to parse the command line: `space.sys.get_flag(...)` reads them
// off the one `space`.  Keeping them process-global means a reader on any
// thread — a codec lookup, `sys.flags`, an embedder calling in — observes
// the values the launcher recorded instead of the per-thread default.
static SYS_NO_SITE: AtomicBool = AtomicBool::new(false);
static SYS_QUIET: AtomicBool = AtomicBool::new(false);
static SYS_INSPECT: AtomicBool = AtomicBool::new(false);
static SYS_INTERACTIVE: AtomicBool = AtomicBool::new(false);
static SYS_NO_USER_SITE: AtomicBool = AtomicBool::new(false);
static SYS_IGNORE_ENVIRONMENT: AtomicBool = AtomicBool::new(false);
static SYS_ISOLATED: AtomicBool = AtomicBool::new(false);
static SYS_DEV_MODE: AtomicBool = AtomicBool::new(false);
static SYS_UTF8_MODE: AtomicI64 = AtomicI64::new(0);
static SYS_SAFE_PATH: AtomicBool = AtomicBool::new(false);
static SYS_OPTIMIZE: AtomicI64 = AtomicI64::new(0);
static SYS_VERBOSE: AtomicI64 = AtomicI64::new(0);
static SYS_DEBUG: AtomicI64 = AtomicI64::new(0);
static SYS_BYTES_WARNING: AtomicI64 = AtomicI64::new(0);
static SYS_DONT_WRITE_BYTECODE: AtomicBool = AtomicBool::new(false);
static SYS_FAULTHANDLER: AtomicBool = AtomicBool::new(false);
static SYS_UNBUFFERED: AtomicBool = AtomicBool::new(false);
static SYS_LEGACY_WINDOWS_STDIO: AtomicBool = AtomicBool::new(false);
// pypy/interpreter/app_main.py keeps the raw `-X` strings in
// `options['_xoptions']` (a list) until sys initialization builds the public
// dict.  Preserve that owner/storage shape rather than introducing a map here.
static SYS_XOPTIONS: LazyLock<Mutex<Vec<std::ffi::OsString>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));
static SYS_WARNOPTIONS: LazyLock<Mutex<Vec<std::ffi::OsString>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));
static SYS_ORIG_ARGV: LazyLock<Mutex<Vec<std::ffi::OsString>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));
static SYS_STDIO_ENCODING: LazyLock<Mutex<Option<String>>> = LazyLock::new(|| Mutex::new(None));
static SYS_PYCACHE_PREFIX: LazyLock<Mutex<Option<std::ffi::OsString>>> =
    LazyLock::new(|| Mutex::new(None));
static SYS_WARN_DEFAULT_ENCODING: AtomicBool = AtomicBool::new(false);
static SYS_CODE_DEBUG_RANGES: AtomicBool = AtomicBool::new(true);

/// Record whether the launcher was given `-S` (no `site` import), so the
/// `sys.flags.no_site` field built during sys module init reflects it. Set
/// before the first `import sys`.
pub fn set_no_site(no_site: bool) {
    SYS_NO_SITE.store(no_site, Ordering::Relaxed);
}

/// Read the `-S` flag for `sys.flags.no_site`.
pub fn no_site_flag() -> bool {
    SYS_NO_SITE.load(Ordering::Relaxed)
}

/// Record the launcher option block consumed by `app_main.py` before `sys` is
/// initialized.  The codec paths also read `dev_mode` directly, matching
/// `space.sys.get_flag('dev_mode')` in `unicodeobject.py`.  The block must have
/// been through `launch_env::finalize`, which is what fills `utf8_mode`.
pub fn set_runtime_flags(flags: &crate::launch_env::LaunchFlags) {
    SYS_QUIET.store(flags.quiet, Ordering::Relaxed);
    SYS_INSPECT.store(flags.inspect, Ordering::Relaxed);
    SYS_INTERACTIVE.store(flags.interactive, Ordering::Relaxed);
    SYS_NO_USER_SITE.store(flags.no_user_site, Ordering::Relaxed);
    SYS_IGNORE_ENVIRONMENT.store(flags.ignore_environment, Ordering::Relaxed);
    SYS_ISOLATED.store(flags.isolated, Ordering::Relaxed);
    SYS_DEV_MODE.store(flags.dev_mode, Ordering::Relaxed);
    SYS_WARN_DEFAULT_ENCODING.store(flags.warn_default_encoding, Ordering::Relaxed);
    // The filesystem codec is picked in `preconfig_read`, ahead of every
    // import, so `os`'s own `_fscodec` closes over the legacy pair rather than
    // the one `sys._enablelegacywindowsfsencoding` leaves it holding.
    #[cfg(windows)]
    crate::typedef::LEGACY_WINDOWS_FS_ENCODING
        .store(flags.legacy_windows_fs_encoding, Ordering::Relaxed);
    SYS_CODE_DEBUG_RANGES.store(!flags.no_debug_ranges, Ordering::Relaxed);
    SYS_UTF8_MODE.store(flags.utf8_mode.unwrap_or(0), Ordering::Relaxed);
    SYS_SAFE_PATH.store(flags.safe_path, Ordering::Relaxed);
    SYS_OPTIMIZE.store(flags.optimize, Ordering::Relaxed);
    SYS_VERBOSE.store(flags.verbose, Ordering::Relaxed);
    SYS_DEBUG.store(flags.debug, Ordering::Relaxed);
    SYS_BYTES_WARNING.store(flags.bytes_warning, Ordering::Relaxed);
    SYS_DONT_WRITE_BYTECODE.store(flags.dont_write_bytecode, Ordering::Relaxed);
    SYS_FAULTHANDLER.store(flags.faulthandler, Ordering::Relaxed);
    *SYS_PYCACHE_PREFIX.lock() = flags.pycache_prefix.clone();
    SYS_UNBUFFERED.store(flags.unbuffered, Ordering::Relaxed);
    SYS_LEGACY_WINDOWS_STDIO.store(flags.legacy_windows_stdio, Ordering::Relaxed);
    *SYS_XOPTIONS.lock() = flags.xoptions.clone();
    *SYS_WARNOPTIONS.lock() = flags.warnoptions.clone();
    *SYS_STDIO_ENCODING.lock() = flags.stdio_encoding.clone();
}

/// Whether the launcher resolved `-X faulthandler` / `-X dev` /
/// PYTHONFAULTHANDLER, which asks for the fatal-signal handlers to be in place
/// before user code runs.
pub fn faulthandler_flag() -> bool {
    SYS_FAULTHANDLER.load(Ordering::Relaxed)
}

/// The directory `-X pycache_prefix` / PYTHONPYCACHEPREFIX named, already
/// resolved to `None` for every spelling that names no directory.  Read back as
/// `sys.pycache_prefix`, which is what `_bootstrap_external.cache_from_source`
/// computes the bytecode path from.
pub fn pycache_prefix() -> Option<std::ffi::OsString> {
    SYS_PYCACHE_PREFIX.lock().clone()
}

/// Raw `-X` values recorded by the launcher, in command-line order.
pub fn xoptions() -> Vec<std::ffi::OsString> {
    SYS_XOPTIONS.lock().clone()
}

pub fn bytes_warning_flag() -> i64 {
    SYS_BYTES_WARNING.load(Ordering::Relaxed)
}

/// `-v` / PYTHONVERBOSE, read back as `sys.flags.verbose`.  The import trace
/// itself is emitted by `importlib._bootstrap`'s `_verbose_message`, which
/// compares this against the verbosity a message asks for.
pub fn verbose_flag() -> i64 {
    SYS_VERBOSE.load(Ordering::Relaxed)
}

/// `-d` / PYTHONDEBUG as the config counts it.  `sys.flags.debug` saturates
/// this to 0/1, because `parser_debug` is declared BOOL there.
pub fn debug_flag() -> i64 {
    SYS_DEBUG.load(Ordering::Relaxed)
}

/// CPython 3.14 `PyConfig.code_debug_ranges`, consumed by every compiler
/// entry point so command-line, import and builtin `compile()` agree.
pub fn code_debug_ranges_flag() -> bool {
    SYS_CODE_DEBUG_RANGES.load(Ordering::Relaxed)
}

/// PyPy `app_main.py:execfile` entry used when the command-line script name
/// ends in `.pyc`.  The actual header and marshal ownership remains in `_imp`,
/// alongside importlib's copy of the same protocol.
pub fn load_pyc_script(bytes: &[u8]) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    crate::module::imp::interp_imp::load_pyc_script(bytes)
}

pub fn unbuffered_flag() -> bool {
    SYS_UNBUFFERED.load(Ordering::Relaxed)
}

pub fn legacy_windows_stdio_flag() -> bool {
    SYS_LEGACY_WINDOWS_STDIO.load(Ordering::Relaxed)
}

pub fn warnoptions() -> Vec<std::ffi::OsString> {
    SYS_WARNOPTIONS.lock().clone()
}

pub fn stdio_encoding() -> Option<String> {
    SYS_STDIO_ENCODING.lock().clone()
}

/// `app_main.py sys.orig_argv[:] = [executable] + argv`: the launcher's
/// own arguments, before parsing rewrote them into the run mode and `sys.argv`.
/// Held in the host's spelling for the same reason `set_sys_argv` takes it —
/// an argument with no UTF-8 form has to survive to the decode.
pub fn set_sys_orig_argv(argv: Vec<std::ffi::OsString>) {
    *SYS_ORIG_ARGV.lock() = argv;
}

pub fn sys_orig_argv() -> Vec<std::ffi::OsString> {
    SYS_ORIG_ARGV.lock().clone()
}

pub fn quiet_flag() -> bool {
    SYS_QUIET.load(Ordering::Relaxed)
}

/// `-i` folded with PYTHONINSPECT, reported as `sys.flags.inspect`.
pub fn inspect_flag() -> bool {
    SYS_INSPECT.load(Ordering::Relaxed)
}

/// `-i` alone, reported as `sys.flags.interactive`.
///
/// The variable sets only `inspect`, so a run that owes the flag to the
/// environment reports `interactive` as false -- and needs a terminal on stdin
/// before `stdin_is_interactive` will open a prompt for it.
pub fn interactive_flag() -> bool {
    SYS_INTERACTIVE.load(Ordering::Relaxed)
}

pub fn no_user_site_flag() -> bool {
    SYS_NO_USER_SITE.load(Ordering::Relaxed)
}

pub fn ignore_environment_flag() -> bool {
    SYS_IGNORE_ENVIRONMENT.load(Ordering::Relaxed)
}

pub fn isolated_flag() -> bool {
    SYS_ISOLATED.load(Ordering::Relaxed)
}

pub fn dev_mode_flag() -> bool {
    SYS_DEV_MODE.load(Ordering::Relaxed)
}

pub fn utf8_mode_flag() -> i64 {
    SYS_UTF8_MODE.load(Ordering::Relaxed)
}

pub fn warn_default_encoding_flag() -> bool {
    SYS_WARN_DEFAULT_ENCODING.load(Ordering::Relaxed)
}

/// `#[dont_look_inside]`: reads the runtime-mutable `SYS_SAFE_PATH` global
/// (`-P` / `PYTHONSAFEPATH`), so the tracer residualises the read rather than
/// folding a build-time constant — the runtime-mutable-global accessor pattern.
#[majit_macros::dont_look_inside]
pub fn safe_path_flag() -> bool {
    SYS_SAFE_PATH.load(Ordering::Relaxed)
}

/// `-O` / `-OO` / PYTHONOPTIMIZE level for the default compile `optimize`
/// (stripped asserts, `__debug__` = False, and at level 2 discarded
/// docstrings), clamped into the compiler's byte-wide field.  Levels above 2
/// behave as 2.
pub fn optimize_flag() -> u8 {
    SYS_OPTIMIZE
        .load(Ordering::Relaxed)
        .clamp(0, i64::from(u8::MAX)) as u8
}

/// The raw optimization level for `sys.flags.optimize`, mirroring a large
/// `PYTHONOPTIMIZE` value verbatim; `optimize_flag` clamps this for the
/// compiler.
pub fn optimize_level() -> i64 {
    SYS_OPTIMIZE.load(Ordering::Relaxed)
}

/// `-B` / PYTHONDONTWRITEBYTECODE, driving `sys.flags.dont_write_bytecode` and
/// `sys.dont_write_bytecode`.
pub fn dont_write_bytecode_flag() -> bool {
    SYS_DONT_WRITE_BYTECODE.load(Ordering::Relaxed)
}

/// Called from sys module init to pick up any pending argv.
pub fn take_pending_sys_argv() -> pyre_object::PyObjectRef {
    SYS_ARGV_PENDING.with(|p| {
        let v = p.get();
        p.set(pyre_object::PY_NULL);
        v
    })
}

pub fn set_sys_modules_dict(dict: PyObjectRef) {
    // Young `sys.modules` dict into a prebuilt process slot
    // (`incminimark.py remember_young_pointer_from_prebuilt`).
    pyre_object::gc_roots::mark_prebuilt_roots_dirty();
    SYS_MODULES_DICT.store(dict as usize, Ordering::Release);
    // The cell just stored is walked; the parameter is a bare local the
    // collector never updates, and every iteration below mints a key string
    // and may grow the dict.
    let roots = pyre_object::gc_roots::push_roots();
    let dict_slot = roots.base();
    let _ = roots.pin_root(dict);
    // Populate with all modules already in the cache.
    for (name, &module) in SYS_MODULES.lock().iter() {
        // Key and module pins live across this one store. The modules dict
        // stays on `roots`.
        let _pyre_store_roots = pyre_object::gc_roots::push_roots();
        let module_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(module as PyObjectRef);
        let key_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(pyre_object::w_str_new_managed(name));
        unsafe {
            pyre_object::w_dict_store(
                roots.get(dict_slot),
                roots.get(key_slot),
                roots.get(module_slot),
            );
        }
    }
}

// ── find_module ──────────────────────────────────────────────────────
// PyPy equivalent: importing.py `find_module()`
// Searches sys.path for `<partname>.py` or `<partname>/__init__.py` (package).

#[derive(Debug)]
enum FindInfo {
    /// A .py source file was found.
    #[cfg(feature = "host_env")]
    SourceFile { pathname: PathBuf },
    /// A pyre-ABI C extension was found.
    #[cfg(all(
        feature = "cpyext",
        not(feature = "sandbox"),
        any(target_os = "macos", target_os = "linux")
    ))]
    ExtensionFile { pathname: PathBuf },
    /// A package whose initializer is a pyre-ABI C extension.
    #[cfg(all(
        feature = "cpyext",
        not(feature = "sandbox"),
        any(target_os = "macos", target_os = "linux")
    ))]
    ExtensionPackage { dirpath: PathBuf, pathname: PathBuf },
    /// A package directory with __init__.py was found.
    #[cfg(feature = "host_env")]
    Package { dirpath: PathBuf },
    /// PEP 420 namespace package: one or more matching directories that carry
    /// no `__init__.py`. The portions become the package's `__path__`.
    #[cfg(feature = "host_env")]
    Namespace { dirs: Vec<PathBuf> },
    /// A builtin (Rust-implemented) module was found.
    /// PyPy equivalent: C_BUILTIN modtype in find_module()
    Builtin,
}

#[cfg(all(feature = "host_env", not(target_arch = "wasm32")))]
fn find_module(
    partname: &str,
    parent_dirs: Option<&[PathBuf]>,
) -> Result<Option<FindInfo>, crate::PyError> {
    // Submodule import: search ONLY the parent package's `__path__`, never
    // sys.path or builtins by leaf name.  `_bootstrap._find_and_load` resolves
    // `pkg.sub` against `pkg.__path__`; routing through sys.path lets a
    // same-leaf module from an unrelated package on sys.path shadow it (e.g.
    // `concurrent.futures` resolving to `asyncio/futures.py`).
    if let Some(dirs) = parent_dirs {
        return Ok(find_in_dirs(partname, dirs));
    }

    // Check builtin modules first (PyPy: space.builtin_modules check in find_module)
    let is_builtin = BUILTIN_MODULES.lock().contains_key(partname);
    if is_builtin {
        return Ok(Some(FindInfo::Builtin));
    }

    // Try sys.path first
    if let Some(info) = find_in_sys_path(partname)? {
        return Ok(Some(info));
    }

    // Fallback stdlib detection. `create_sys_path_list` already forces this at
    // sys-module creation, so the `DONE` guard normally makes this a no-op; it
    // stays as a defensive retry for any miss reached before sys exists.
    ensure_stdlib_path();
    find_in_sys_path(partname)
}

// wasm has no current_exe / python3 spawn, so there is no `ensure_stdlib_path`
// lazy detection; `sys.path` is seeded by the wasm bootstrap (the embedded-VFS
// mount point, or the host stdlib root for the runner). Otherwise this mirrors
// the native `find_module`: submodule imports search the parent package's
// `__path__`, top-level names check builtins then sys.path — all FS probes go
// through the installed `SourceProvider`.
#[cfg(all(feature = "host_env", target_arch = "wasm32"))]
fn find_module(
    partname: &str,
    parent_dirs: Option<&[PathBuf]>,
) -> Result<Option<FindInfo>, crate::PyError> {
    if let Some(dirs) = parent_dirs {
        return Ok(find_in_dirs(partname, dirs));
    }
    let is_builtin = BUILTIN_MODULES.lock().contains_key(partname);
    if is_builtin {
        return Ok(Some(FindInfo::Builtin));
    }
    find_in_sys_path(partname)
}

#[cfg(not(feature = "host_env"))]
fn find_module(
    partname: &str,
    _parent_dirs: Option<&[PathBuf]>,
) -> Result<Option<FindInfo>, crate::PyError> {
    let is_builtin = BUILTIN_MODULES.lock().contains_key(partname);
    if is_builtin {
        return Ok(Some(FindInfo::Builtin));
    }
    Ok(None)
}

/// Install the PyPy-shaped bootstrap path exactly once for the process.
#[cfg(all(feature = "host_env", not(target_arch = "wasm32")))]
fn ensure_stdlib_path() {
    static DONE: AtomicBool = AtomicBool::new(false);
    if DONE.swap(true, Ordering::AcqRel) {
        return;
    }
    let config = startup_path_config();
    // A `._pth` is `sys.path` verbatim, down to a repeated line being a
    // repeated entry, so its list is staged as written rather than through
    // `add_sys_path`, which folds a duplicate away.  An empty file is likewise
    // the search path the deployment asked for, so the "keep the binary inside
    // its tree" warning below does not apply to it either.
    #[cfg(not(feature = "sandbox"))]
    if config.pth.is_some() {
        SYS_PATH.lock().extend(config.stdlib_paths.iter().cloned());
        return;
    }
    if config.stdlib_paths.is_empty() {
        let warning = format!(
            "debug: WARNING: Library path not found, using the existing sys.path;\n\
             debug: WARNING: sys.prefix = {:?}\n\
             debug: WARNING: Make sure the pyre binary is kept inside its tree of files.\n",
            config.prefix
        );
        // A sandbox build compiles this path too, and there the real stderr is
        // not ours to write; `host_seam::ops` is that build's stderr.  It is
        // also the only build that has it, so the ordinary one keeps `eprint!`.
        #[cfg(feature = "sandbox")]
        let _ = crate::host_seam::ops::write(2, warning.as_bytes());
        #[cfg(not(feature = "sandbox"))]
        eprint!("{warning}");
    }
    for path in &config.stdlib_paths {
        add_sys_path(path);
    }
}

#[cfg(feature = "host_env")]
fn find_in_dirs(partname: &str, dirs: &[PathBuf]) -> Option<FindInfo> {
    let mut namespace_dirs: Vec<PathBuf> = Vec::new();
    for dir in dirs {
        // Check for package: <dir>/<partname>/__init__.py
        let pkg_dir = dir.join(partname);
        #[cfg(all(
            feature = "cpyext",
            not(feature = "sandbox"),
            any(target_os = "macos", target_os = "linux")
        ))]
        {
            let init_extension =
                pkg_dir.join(format!("__init__{}", crate::cpyext::extension_suffix()));
            if with_source_provider(|p| p.is_file(&init_extension)) {
                return Some(FindInfo::ExtensionPackage {
                    dirpath: pkg_dir,
                    pathname: init_extension,
                });
            }
        }
        let init_file = pkg_dir.join("__init__.py");
        if with_source_provider(|p| p.is_file(&init_file)) {
            return Some(FindInfo::Package { dirpath: pkg_dir });
        }

        // FileFinder probes packages before files. Only pyre's own suffix is
        // accepted: advertising CPython's tag would falsely promise its object
        // layout ABI.
        #[cfg(all(
            feature = "cpyext",
            not(feature = "sandbox"),
            any(target_os = "macos", target_os = "linux")
        ))]
        {
            let extension = dir.join(format!("{partname}{}", crate::cpyext::extension_suffix()));
            if with_source_provider(|p| p.is_file(&extension)) {
                return Some(FindInfo::ExtensionFile {
                    pathname: extension,
                });
            }
        }

        // Check for source file: <dir>/<partname>.py
        let source_file = dir.join(format!("{partname}.py"));
        if with_source_provider(|p| p.is_file(&source_file)) {
            return Some(FindInfo::SourceFile {
                pathname: source_file,
            });
        }

        // PEP 420: a matching directory without `__init__.py` is a namespace
        // portion. Record it and keep scanning — a regular module or package
        // in a later directory still wins; only if no concrete match is found
        // do the recorded portions form a namespace package.
        if with_source_provider(|p| p.is_dir(&pkg_dir)) {
            namespace_dirs.push(pkg_dir);
        }
    }
    if !namespace_dirs.is_empty() {
        return Some(FindInfo::Namespace {
            dirs: namespace_dirs,
        });
    }
    None
}

/// Read the live Python `sys.path` list as filesystem directories, or an
/// empty vec before `sys` / its `path` list exists (the pre-`sync` bootstrap
/// window). Only `str` entries are collected — path hooks (zipimporter keys
/// and the like) are not resolvable by the native file search.
///
/// An entry the filesystem encoding cannot spell raises out of the import
/// rather than being passed over.
#[cfg(feature = "host_env")]
fn python_sys_path_dirs() -> Result<Option<Vec<PathBuf>>, crate::PyError> {
    // `None` means the `sys` module does not exist yet (the pre-`sys` bootstrap
    // window) — the caller falls back to the native seed. Once `sys` exists the
    // Python list is authoritative even when empty: a missing / non-list / empty
    // `sys.path` searches nothing, so `del sys.path` and `sys.path.clear()` break
    // imports exactly as they do under CPython, rather than resurrecting the seed.
    let Some(sys_mod) = get_interpreter_sys_module() else {
        return Ok(None);
    };
    let w_dict = unsafe { pyre_object::w_module_get_w_dict(sys_mod) };
    if w_dict.is_null() {
        return Ok(None);
    }
    // Copy the entries up front so no Python borrow is held across the
    // filesystem probes in `find_in_dirs` (which never invoke user code).
    let mut dirs = Vec::new();
    if let Some(mut w_path) = unsafe { pyre_object::w_dict_getitem_str(w_dict, "path") }
        && unsafe { pyre_object::is_list(w_path) }
    {
        let n = unsafe { pyre_object::listobject::w_list_len(w_path) };
        dirs.reserve(n);
        for i in 0..n {
            if let Some(item) = unsafe { pyre_object::listobject::w_list_getitem(w_path, i as i64) }
            {
                // Non-str entries are skipped: pyre's only path hook is the
                // native filesystem probe, and CPython also skips an entry no
                // hook accepts.
                if unsafe { pyre_object::is_str(item) } {
                    dirs.push(
                        pyre_object::with_roots!(w_path => crate::gateway::fspath_buf(item))?,
                    );
                }
            }
        }
    }
    Ok(Some(dirs))
}

/// Search the import path for a top-level `partname`.
///
/// The live Python `sys.path` list is authoritative — user code (and PYTHONPATH)
/// mutate it and imports must honor it, the same precedence `check_sys_modules`
/// gives the Python `sys.modules` dict. The native `SYS_PATH` is only the write
/// side of a pre-`sys` staging seed (flushed into `sys.path` at sys-module
/// creation) and is consulted here solely in that pre-`sys` window.
#[cfg(feature = "host_env")]
fn find_in_sys_path(partname: &str) -> Result<Option<FindInfo>, crate::PyError> {
    Ok(match python_sys_path_dirs()? {
        Some(dirs) => {
            let found = find_in_dirs(partname, &dirs);
            // Windows: pyre still registers the `posix` builtin (never `nt`),
            // so `os.path` is posixpath and `site.removeduppaths()` rewrites
            // every drive-letter `sys.path` entry into `<cwd>/D:\...` garbage
            // at startup. Until the `nt` registration lands, a live-list miss
            // falls back to the native seed so the stdlib stays importable.
            #[cfg(windows)]
            let found = found.or_else(|| {
                let path = SYS_PATH.lock();
                find_in_dirs(partname, &path)
            });
            found
        }
        None => {
            let path = SYS_PATH.lock();
            find_in_dirs(partname, &path)
        }
    })
}

/// Build the initial Python `sys.path` list from the native `SYS_PATH` seed.
/// Called once at sys-module creation so the Python list is populated the
/// instant `sys` exists; from then on the Python list is authoritative and the
/// seed is spent.
///
/// Stdlib detection is forced here (off-wasm) so the vendored stdlib is on
/// `sys.path` before any user code — including `python -S` / `-S -P` runs that
/// never import `site` — reads it, matching the unconditional detection the
/// removed `sync_python_sys_path` performed. `sys` is not yet in `sys.modules`
/// during its own creation, so `ensure_stdlib_path`'s `add_sys_path` stages the
/// stdlib in the seed, and the flush below picks it up.
#[cfg(feature = "host_env")]
pub(crate) fn create_sys_path_list() -> PyObjectRef {
    #[cfg(not(target_arch = "wasm32"))]
    ensure_stdlib_path();
    // Snapshot the paths first so the lock is not held across an allocation,
    // then pin each decoded string as it is produced -- a plain `Vec` is not a
    // root area (`build_list_storage`).
    let paths: Vec<std::ffi::OsString> = SYS_PATH
        .lock()
        .iter()
        .map(|d| d.as_os_str().to_os_string())
        .collect();
    let mut items = pyre_object::gc_roots::RootedItems::new();
    for d in &paths {
        items.push(crate::gateway::fsdecode_os_str(d.as_os_str()));
    }
    pyre_object::w_list_new(items.take())
}

/// Off-`host_env` builds have no native seed; `sys.path` starts empty.
#[cfg(not(feature = "host_env"))]
pub(crate) fn create_sys_path_list() -> PyObjectRef {
    pyre_object::w_list_new(vec![])
}

/// Extract a package module's `__path__` as filesystem directories.
///
/// Returns `None` when `__path__` is absent or is not a plain list of `str`.
/// `absolute_import` rejects the absent case up front ("'<parent>' is not a
/// package"), so a `None` reaching the caller means a package whose `__path__`
/// yielded no usable directories. An entry the filesystem encoding cannot spell
/// raises, the same as one on `sys.path`.
fn parent_package_path(parent: PyObjectRef) -> Result<Option<Vec<PathBuf>>, crate::PyError> {
    let w_dict = unsafe { pyre_object::w_module_get_w_dict(parent) };
    if w_dict.is_null() || !unsafe { pyre_object::is_dict(w_dict) } {
        return Ok(None);
    }
    let Some(mut path_obj) = (unsafe { pyre_object::w_dict_getitem_str(w_dict, "__path__") })
    else {
        return Ok(None);
    };
    if path_obj.is_null() || !unsafe { pyre_object::is_list(path_obj) } {
        return Ok(None);
    }
    let n = unsafe { pyre_object::listobject::w_list_len(path_obj) };
    let mut dirs = Vec::with_capacity(n);
    for i in 0..n {
        if let Some(item) = unsafe { pyre_object::listobject::w_list_getitem(path_obj, i as i64) }
            && unsafe { pyre_object::is_str(item) }
        {
            dirs.push(pyre_object::with_roots!(path_obj => crate::gateway::fspath_buf(item))?);
        }
    }
    Ok(Some(dirs))
}

// ── parse_source_module ──────────────────────────────────────────────
// PyPy equivalent: importing.py `parse_source_module(space, pathname, source)`

fn parse_source_module(
    pathname: &rustpython_wtf8::Wtf8,
    source: &str,
) -> Result<CodeObject, crate::syntax_warnings::SourceCompileError> {
    // PyPy `PythonAstCompiler.compile` runs the same string-literal warning
    // pass for imported source as for `compile()`.  RustPython exposes parsing
    // separately, so keep pyre's shared warning pass in front of this file
    // compiler too; otherwise an invalid escape warns under `-c` but silently
    // disappears when the identical source is loaded from a `.py` file.
    // `pathname` is the decoded filename, not the compiler's lossy UTF-8 path.
    crate::syntax_warnings::compile_with_codegen_warnings(
        source,
        Mode::Exec,
        pathname,
        CompileOpts {
            optimize: optimize_flag(),
            debug_ranges: code_debug_ranges_flag(),
            int_max_str_digits: crate::module::sys::state::int_max_str_digits().max(0) as usize,
            ..Default::default()
        },
    )
}

// ── exec_code_module ─────────────────────────────────────────────────
// PyPy equivalent: importing.py `exec_code_module(space, w_mod, code_w,
//                                  pathname, cpathname, write_paths=True)`
//
// Mirrors `pypy/module/imp/importing.py` line-by-line:
//   w_dict = space.getattr(w_mod, '__dict__')                       # ns
//   space.call_method(w_dict, 'setdefault',
//                     '__builtins__', space.builtin)
//   if write_paths:
//       space.setitem(w_dict, '__file__', w_pathname)
//       space.setitem(w_dict, '__cached__', w_cpathname)
//       _fix_up_module(d, name, pathname, cpathname)               # appexec
//   code_w.exec_code(space, w_dict, w_dict)
//
// `pathname` is `None` for callers that do not have a filesystem path
// (REPL `__main__`, builtin module bootstrap), matching PyPy's
// `write_paths=False` shape.  `cpathname` is `None` when no `.pyc` cache
// is available (pyre has no .pyc cache today, so all reachable callers
// pass `None` here — kept as a parameter so the signature mirrors PyPy
// instead of erasing the field).

fn exec_code_module(
    mut w_code: PyObjectRef,
    mut w_globals: pyre_object::PyObjectRef,
    execution_context: *const PyExecutionContext,
    pathname: Option<&rustpython_wtf8::Wtf8>,
    cpathname: Option<&rustpython_wtf8::Wtf8>,
    plain_dispatch: bool,
) -> Result<PyObjectRef, crate::PyError> {
    // importing.py:272-274 — setdefault('__builtins__', space.builtin).
    // `fresh_module_globals` already seeds `__builtins__` for module-shape
    // namespaces; the explicit setdefault here mirrors PyPy's defensive
    // call so callers that hand in a pre-built globals object (future
    // `_imp.exec_dynamic`-style entry) still inherit the builtins
    // pointer with no surprises.
    if unsafe { pyre_object::w_dict_getitem_str(w_globals, "__builtins__") }.is_none() {
        let ctx = unsafe { &*execution_context };
        let w_builtin = ctx.get_builtin();
        if !w_builtin.is_null() {
            unsafe {
                pyre_object::w_dict_setitem_str(w_globals, "__builtins__", w_builtin);
            }
        }
    }
    // importing.py:275-298 write_paths block.  Pyre callers always pass
    // `Some(pathname)` for source-file imports and `None` for the
    // `write_paths=False` shape (REPL, builtin bootstrap).
    if let Some(p) = pathname {
        // importing.py:284 setitem('__file__', w_pathname).
        // `importing.py` `space.newfilename` / `newutf8` copies the path once.
        let w_pathname = pyre_object::with_roots!(w_code, w_globals => {
            pyre_object::w_str_from_wtf8_managed_borrowed(p)
        });
        unsafe {
            pyre_object::w_dict_setitem_str(w_globals, "__file__", w_pathname);
        }
        // importing.py:285 setitem('__cached__', w_cpathname).  PyPy
        // surfaces `space.w_None` when `cpathname is None`, i.e. the
        // import was not satisfied from a `.pyc`.  Pyre has no .pyc
        // path today so reachable callers still hit the None arm.
        let w_cpathname = match cpathname {
            Some(c) => pyre_object::w_str_from_wtf8_managed_borrowed(c),
            None => pyre_object::w_none(),
        };
        unsafe {
            pyre_object::w_dict_setitem_str(w_globals, "__cached__", w_cpathname);
        }
        // importing.py:286-298 — `_fix_up_module(d, name, pathname,
        // cpathname)` sets `__spec__`/`__loader__`/`__file__`/`__cached__`
        // from the app-level `SourceFileLoader` + `spec_from_file_location`
        // helpers.  Reachable only once the importlib bootstrap is wired;
        // before that, seed `__loader__`/`__spec__` with `None` only when
        // missing — the `if not loader / if not spec` guards at
        // `_bootstrap_external.py:_fix_up_module`.
        if !pyre_object::with_roots!(w_code, w_globals => fix_up_source_module_spec(w_globals, p, cpathname))?
        {
            if unsafe { pyre_object::w_dict_getitem_str(w_globals, "__loader__") }.is_none() {
                unsafe {
                    pyre_object::w_dict_setitem_str(w_globals, "__loader__", pyre_object::w_none());
                }
            }
            if unsafe { pyre_object::w_dict_getitem_str(w_globals, "__spec__") }.is_none() {
                unsafe {
                    pyre_object::w_dict_setitem_str(w_globals, "__spec__", pyre_object::w_none());
                }
            }
        }
    }
    // importing.py:300 code_w.exec_code(space, w_dict, w_dict) → eval.py
    // Code.exec_code → space.createframe(...) + frame.run().  Surface
    // initialize_frame_scopes' freevar/closure mismatch (TypeError /
    // ValueError per pyframe.py) as PyError so the importer
    // reports it instead of panicking.  Route through run_with_jit so the
    // GENERATOR / COROUTINE / ASYNC_GENERATOR dispatch in
    // pyframe.py holds for the import path too, and so an imported
    // module's top-level hot loop reaches the JIT portal.
    // `moduledef.py` `_compile_bootstrap_module` is NOT_RPYTHON and
    // `startup_at_translation_time_only` does not re-exec those frames.
    // `zipimport` `Module.install` only inserts `zipimporter`. Those three
    // bodies run once, on `execute_frame`, the same plain dispatch
    // `appleveldef_install_seeded` uses for `@not_rpython` app files.
    let mut frame =
        crate::pyframe::createframe_obj(w_code as *const (), w_globals, execution_context, None)?;
    if plain_dispatch {
        frame.execute_frame(None, None)
    } else {
        frame.run_with_jit()
    }
}

// ── appleveldef_install ──────────────────────────────────────────────
// PyPy equivalent: `pypy/interpreter/mixedmodule.py MixedModule.get`
// resolves an `appleveldefs` entry by lazily executing the sibling
// `app_*.py` file into a per-mixedmodule namespace and reading the
// named attribute.  Pyre's macro form bundles all entries from one app
// file into a single install call; the source is included at
// compile time via `include_str!` so no filesystem read happens at
// module-init time.

/// Execute `source` (a Python module) into a fresh namespace and copy
/// each binding in `names` into the caller's module dict `ns`.
///
/// `filename` is used as the source path for tracebacks / co_filename
/// only; `modname` is the name the namespace answers `__name__` with, and so
/// the `__module__` of everything `source` defines.  Every function defined
/// in `source` retains the intermediate
/// namespace as its `__globals__`; once copied into the caller's module,
/// those functions keep the namespace transitively reachable.
#[doc(hidden)]
pub trait AppleveldefNamespace {
    fn store(&mut self, name: &str, value: PyObjectRef);
}

impl AppleveldefNamespace for PyObjectRef {
    fn store(&mut self, name: &str, value: PyObjectRef) {
        // Only a mixed-module dict is wrapped. `load_builtin_module` stamps
        // `__name__` before `init`. Callers that install into a fresh dict
        // (`app_functional`, `async_operation`, `reduce_protocol`) have no
        // module name there, and those values stay plain functions.
        let _roots = pyre_object::gc_roots::push_roots();
        let ns_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(*self);
        let value_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(value);
        let stored = applevel_module_value(
            pyre_object::gc_roots::shadow_stack_get(ns_slot),
            name,
            pyre_object::gc_roots::shadow_stack_get(value_slot),
        );
        let stored_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(stored);
        crate::module_ns_store(
            pyre_object::gc_roots::shadow_stack_get(ns_slot),
            name,
            pyre_object::gc_roots::shadow_stack_get(stored_slot),
        );
    }
}

/// `mixedmodule.py _load_lazily`: a plain `Function` placed directly in a
/// mixed-module dict becomes `BuiltinFunction` (no `__get__`). Classes and
/// every other object are stored as they are.
fn applevel_module_value(ns: PyObjectRef, name: &str, value: PyObjectRef) -> PyObjectRef {
    let Some(w_module) = (unsafe { pyre_object::w_dict_getitem_str(ns, "__name__") }) else {
        return value;
    };
    if !unsafe { pyre_object::py_type_check(value, &crate::function::FUNCTION_TYPE) } {
        return value;
    }
    unsafe { crate::function::builtin_function_from_function(value, name, w_module) }
}

pub fn appleveldef_install(
    ns: impl AppleveldefNamespace,
    source: &str,
    filename: &str,
    modname: &str,
    names: &[&str],
) -> Result<(), crate::PyError> {
    appleveldef_install_seeded(ns, source, filename, modname, names, &[])
}

/// [`appleveldef_install`] with `seed` bound into the app namespace before the
/// source runs.
///
/// `mixedmodule.py MixedModule.get` executes the sibling app file lazily, on
/// the first access to a name it defines, so by then the module is importable
/// and the file can reach its own interp-level types through a plain
/// `import _io` (`app_io.py`).  pyre installs app files eagerly from the
/// module initializer, before the module object exists, so a name the source
/// needs from its own module is bound up front instead.
pub fn appleveldef_install_seeded(
    mut ns: impl AppleveldefNamespace,
    source: &str,
    filename: &str,
    modname: &str,
    names: &[&str],
    seed: &[(&str, PyObjectRef)],
) -> Result<(), crate::PyError> {
    let ctx = crate::call::getexecutioncontext();
    if ctx.is_null() {
        panic!("appleveldef `{filename}`: no execution context at module init");
    }
    // `gateway.py ApplevelClass` / `mixedmodule.py MixedModule._cleanup_`
    // compile the app file at translation. The source is linked into this
    // binary; the marshalled image is keyed by the full source bytes
    // (`interp_imp.applevel_cache_load`), not a hash in the pyc mtime word.
    let cache_key = format!("applevel.{modname}.{filename}");
    let loaded = crate::module::imp::interp_imp::applevel_cache_load(&cache_key, source);
    let _root = pyre_object::gc_roots::push_roots();
    let code_slot = pyre_object::gc_roots::shadow_stack_len();
    if let Some(w_code) = loaded {
        unsafe { crate::pycode::set_hidden_applevel_unit(w_code) };
        let _ = pyre_object::gc_roots::pin_root(w_code);
    } else {
        let code = compile_source_with_filename(source, Mode::Exec, filename)
            .unwrap_or_else(|e| panic!("appleveldef `{filename}`: compile failed — {e}"));
        let w_code = crate::pycode::box_code_object_with_hidden_applevel(code, true);
        let _ = pyre_object::gc_roots::pin_root(w_code);
        crate::module::imp::interp_imp::applevel_cache_store(
            &cache_key,
            source,
            pyre_object::gc_roots::shadow_stack_get(code_slot),
        );
    }
    let w_app_globals = unsafe { (*ctx).fresh_module_globals() };
    let globals_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_app_globals);
    // `gateway.py build_applevel_dict` binds the owning module's name into the
    // namespace before the source runs, so a `def` or a `class` in it answers
    // `__module__` with the module it belongs to rather than `None` for a
    // function or `builtins` for a class.  `modname` is what
    // `mixedmodule.py MixedModule.get_applevel_name` names: the module's own
    // name, or `"<parent>.<child>"` for a submodule.  A source that must be
    // seen under a different name -- the public spelling of an accelerator it
    // backs -- rebinds `__name__` itself, and that assignment runs later.
    let name_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_new_managed(modname));
    unsafe {
        pyre_object::w_dict_setitem_str(
            pyre_object::gc_roots::shadow_stack_get(globals_slot),
            "__name__",
            pyre_object::gc_roots::shadow_stack_get(name_slot),
        )
    };
    for &(name, value) in seed {
        unsafe {
            pyre_object::w_dict_setitem_str(
                pyre_object::gc_roots::shadow_stack_get(globals_slot),
                name,
                value,
            )
        };
    }
    // `gateway.py ApplevelClass.hidden_applevel = True` is already on `w_code`
    // (compile path via `box_code_object_with_hidden_applevel`, cache path via
    // `set_hidden_applevel_unit`). Every source installed here is a native
    // module's body, so none of these frames are the running program's.
    let w_code = pyre_object::gc_roots::shadow_stack_get(code_slot);
    let w_app_globals = pyre_object::gc_roots::shadow_stack_get(globals_slot);
    let mut frame = crate::pyframe::createframe_obj(w_code as *const (), w_app_globals, ctx, None)
        .unwrap_or_else(|e| panic!("appleveldef `{filename}`: createframe — {e:?}"));
    // `mixedmodule.py` `buildloaders` is `@not_rpython` and `_cleanup_`
    // freezes the module, so these bodies are not traced. `gateway.py`
    // `ApplevelClass.hidden_applevel` marks the same frames. Run the
    // non-jitted `dispatch_bytecode` arm (`execute_frame`) instead of the
    // portal: the body runs once and has no back-edge for `can_enter_jit`.
    // What it raises is the program's to see: a `from _operator import eq`
    // resolves through `sys.modules`, which the program can have blocked.
    // Only a bundled file that does not compile, or a name it never binds,
    // stays a panic.
    frame.execute_frame(None, None)?;
    for &name in names {
        match unsafe {
            pyre_object::w_dict_getitem_str(
                pyre_object::gc_roots::shadow_stack_get(globals_slot),
                name,
            )
        } {
            Some(val) => ns.store(name, val),
            None => panic!("appleveldef `{filename}`: name `{name}` not bound by source"),
        }
    }
    Ok(())
}

// ── load_source_module ───────────────────────────────────────────────
// PyPy equivalent: importing.py `load_source_module()`
//
// Parse + execute a .py source file, producing a module object.

#[cfg(feature = "host_env")]
fn load_source_module(
    modulename: &str,
    pathname: &Path,
    package_dir: Option<&Path>,
    execution_context: *const PyExecutionContext,
) -> Result<PyObjectRef, crate::PyError> {
    // The name reaches the module as `__file__` and the code object as
    // `co_filename`, so it is carried in the two spellings `pycode.py:431
    // filename='fsencode'` keeps apart: the filesystem bytes it was named
    // with, and the UTF-8 spelling the compiler's own `source_path` is limited
    // to. `path_text` is the first, decoded, and is what application level
    // sees.
    let path_bytes = crate::gateway::fsencode_os_str(pathname.as_os_str());
    let path_text = crate::gateway::fsdecode_filename_wtf8(&path_bytes);
    let (_pathname_str, filename_bytes) =
        crate::pycode::split_code_filename_bytes(path_bytes, None);
    // The two importlib bootstrap sources and `zipimport` are imported by the
    // native importer before `SourceFileLoader` exists, so they never reach the
    // `.pyc` cache and otherwise recompile on every startup.
    // `Module._cached_compile` (which `zipimport`'s moduledef also goes
    // through): reload a marshalled, source-validated code object when the
    // cache holds one for this binary, recompiling only on a miss.
    // Startup loads the bootstrap sources under their frozen names; a source
    // copy still loads the public submodule names. Both share one cache entry.
    // `zipimport` stays keyed by its own name.  A hit still reads the source
    // so `Module._cached_compile` can compare it with the bytes stored in the
    // cache (`previous != source + '\x00'`).
    let cache_key = match modulename {
        "_frozen_importlib" | "importlib._bootstrap" => Some("importlib._bootstrap"),
        "_frozen_importlib_external" | "importlib._bootstrap_external" => {
            Some("importlib._bootstrap_external")
        }
        "zipimport" => Some(modulename),
        _ => None,
    };
    let cache_ok = with_source_provider(|p| p.frozen_marshal_cache_ok());
    let read_decoded_source = || -> Result<String, crate::PyError> {
        let bytes = with_source_provider(|p| p.read_to_bytes(pathname)).map_err(|e| {
            let mut message = rustpython_wtf8::Wtf8Buf::from_string("cannot read '".to_string());
            message.push_wtf8(&path_text);
            message.push_str(&format!("': {e}"));
            crate::PyError::new(crate::PyErrorKind::ImportError, message)
        })?;
        // A source file carries its own encoding in a BOM or a PEP 263 cookie; a
        // bad declaration is the tokenizer's SyntaxError, not an ImportError.
        // [3.14-spec] CPython 3.14 `SourceFileLoader.exec_module` rejects an
        // embedded NUL here with `source_as_string`'s unlocated "source code
        // string" error.  The real pypy3 loader accepts it; pyre takes the 3.14
        // observable while its command-line file path retains PyPy's located
        // tokenizer boundary through `decode_file_source_bytes`.
        crate::compile::decode_source_bytes(&bytes, &path_text, false)
    };
    let parse_decoded_source = |source: &str| -> Result<PyObjectRef, crate::PyError> {
        let code = parse_source_module(&path_text, source).map_err(|error| match error {
            crate::syntax_warnings::SourceCompileError::Compile(error) => {
                crate::compile_err_to_syntax_error(error, source, Mode::Exec)
            }
            crate::syntax_warnings::SourceCompileError::Warning(error) => error,
        })?;
        Ok(crate::box_code_object(code))
    };
    // `_bootstrap_external._validate_timestamp_pyc` / `check_compiled_module`:
    // a timestamp `.pyc` hit skips the parse; `__file__` stays the source path
    // and `__cached__` is the `.pyc` (`importing.py` `exec_code_module`).
    #[cfg(not(feature = "sandbox"))]
    let mut timestamp_cpathname: Option<rustpython_wtf8::Wtf8Buf> = None;
    let mut source = String::new();
    // `gctransform` (`framework.py` + `shadowstack.py` `gc_push_roots`)
    // keeps `code_w` in a shadow-stack root across `space.newfilename` and
    // `update_code_filenames` (`importing.py` `load_source_module`). Open
    // the bracket before the compile-or-cache branch so each arm can pin
    // the code object as soon as it exists; a timestamp `.pyc` hit has to
    // hold it across the allocating `newfilename` decode.
    let roots = pyre_object::gc_roots::push_roots();
    let code_slot = roots.base();
    let store = if cache_ok && let Some(key) = cache_key {
        source = read_decoded_source()?;
        match crate::module::imp::interp_imp::frozen_cache_load(key, source.as_bytes()) {
            Some(w_code) => {
                let _ = roots.pin_root(w_code);
                false
            }
            None => {
                let _ = roots.pin_root(parse_decoded_source(&source)?);
                true
            }
        }
    } else {
        #[cfg(not(feature = "sandbox"))]
        {
            if cache_ok
                && let Some((code, cpathname)) =
                    crate::module::imp::interp_imp::try_load_timestamp_pyc(pathname)?
            {
                // Pin before `space.newfilename(cpathname)`: filesystem-decoded
                // like `pathname`, so undecodable bytes round-trip, and the
                // decode can collect.
                let _ = roots.pin_root(code);
                timestamp_cpathname = Some(crate::gateway::fsdecode_filename_wtf8(
                    &crate::gateway::fsencode_os_str(cpathname.as_os_str()),
                ));
                false
            } else {
                source = read_decoded_source()?;
                let _ = roots.pin_root(parse_decoded_source(&source)?);
                false
            }
        }
        #[cfg(feature = "sandbox")]
        {
            source = read_decoded_source()?;
            let _ = roots.pin_root(parse_decoded_source(&source)?);
            false
        }
    };
    // The whole unit was named by this path, so recurse through the eager
    // nested PyCode constants like PyPy `update_code_filenames`.
    // `set_compilation_unit_filename_bytes` walks nested codes and may
    // allocate; read the code object back from the slot `gctransform`
    // keeps live across that call.
    unsafe {
        crate::pycode::set_compilation_unit_filename_bytes(roots.get(code_slot), filename_bytes)
    };
    if let (true, Some(key)) = (store, cache_key) {
        crate::module::imp::interp_imp::frozen_cache_store(
            key,
            source.as_bytes(),
            roots.get(code_slot),
        );
    }

    // Create a fresh namespace for the module, seeded with builtins.
    // PyPy equivalent: Module.__init__ creates w_dict = space.newdict()
    // then exec_code_module sets __builtins__ and runs code in w_dict.
    let ctx = unsafe { &*execution_context };
    let w_globals = ctx.fresh_module_globals();
    let globals_slot = code_slot + 1;
    let _ = roots.pin_root(w_globals);
    // PyPy `interpreter/module.py:Module.__init__` seeds `__name__` on
    // the module's w_dict.  `w_module_new_aliasing_dict` below does that
    // via `w_dict_setitem_str("__name__", ...)`, so an explicit store
    // here would be redundant.
    //
    // `__file__`/`__cached__` setting moved into `exec_code_module`
    // (`importing.py`) so the per-module attribute seeding
    // mirrors the PyPy call order.
    //
    // `__package__` is set by PyPy `interp_imp._prepare_module`
    // (`pypy/module/imp/interp_imp.py`); pyre has no `_prepare_module`
    // yet, so we still seed it here as a TODO until
    // the prepare-module path is ported.
    // A package's `__init__.py` is its own `__package__`; a plain module's
    // `__package__` is its containing package.
    // Frozen top-level modules have an empty `__package__`. The two bootstrap
    // names are loaded as top-level modules, not as `importlib` submodules.
    let pkg = if matches!(
        modulename,
        "_frozen_importlib" | "_frozen_importlib_external"
    ) {
        ""
    } else if package_dir.is_some() {
        modulename
    } else if let Some(dot) = modulename.rfind('.') {
        &modulename[..dot]
    } else {
        modulename
    };
    let package_name = pyre_object::w_str_new_managed(pkg);
    unsafe {
        pyre_object::w_dict_setitem_str(roots.get(globals_slot), "__package__", package_name);
    }

    // Seed `__path__` BEFORE executing the package body so relative imports
    // inside `__init__.py` (`from .sub import *`) resolve against the package
    // directory.  `_bootstrap` sets `__path__` on the module before
    // `exec_module`; setting it afterwards lets those imports fall through to
    // sys.path and pick up a same-leaf module from an unrelated package.
    if let Some(dir) = package_dir {
        let mut path_str = crate::gateway::fsdecode_os_str(dir.as_os_str());
        let path_list = pyre_object::with_roots!(path_str =>
            pyre_object::w_list_new(vec![path_str])
        );
        unsafe {
            pyre_object::w_dict_setitem_str(roots.get(globals_slot), "__path__", path_list);
        }
    }

    // Create the module object BEFORE execution and register in sys.modules.
    // PyPy: load_source_module → set_sys_modules BEFORE exec_code_module.
    // This prevents infinite recursion on circular imports.
    let canonical = roots.get(globals_slot);
    let mut module = pyre_object::w_module_new_aliasing_dict(modulename, canonical);
    pyre_object::with_roots!(module => set_sys_module(modulename, module));
    // Exec and the sys.modules reads below can collect. The identity compare
    // after exec has to see the same module the dict entry was updated to.
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(module);

    // A source copy of the public submodule (`from . import _bootstrap` once
    // `_frozen_importlib` is blocked) still execs here under the frozen
    // `__name__`, so classes capture that `__module__`, then restores the
    // public name. Startup does not: it loads the frozen name itself
    // (`moduledef.py Module.install`) and leaves `__name__` unchanged.
    let frozen_exec_name = match modulename {
        "importlib._bootstrap" => Some("_frozen_importlib"),
        "importlib._bootstrap_external" => Some("_frozen_importlib_external"),
        _ => None,
    };
    if let Some(frozen) = frozen_exec_name {
        unsafe {
            let frozen_name = pyre_object::w_str_new_managed(frozen);
            pyre_object::w_dict_setitem_str(roots.get(globals_slot), "__name__", frozen_name);
        }
    }

    // PyPy `importing.py:300` passes `pathname`/`cpathname` to
    // `exec_code_module`. A timestamp `.pyc` hit supplies that path so
    // `__cached__` is set; a source parse leaves it None.
    //
    // On exec failure drop the pre-registered module from sys.modules
    // (`_bootstrap._load`) so a retried import re-runs the body instead of
    // observing a half-built module.
    #[cfg(not(feature = "sandbox"))]
    let cpathname = timestamp_cpathname.as_deref();
    #[cfg(feature = "sandbox")]
    let cpathname: Option<&rustpython_wtf8::Wtf8> = None;
    if let Err(e) = exec_code_module(
        roots.get(code_slot),
        roots.get(globals_slot),
        execution_context,
        Some(&path_text),
        cpathname,
        cache_key.is_some(),
    ) {
        remove_sys_module(modulename);
        return Err(e);
    }

    // Restore the public name now that class bodies have captured the frozen
    // `__module__`; `module.__name__` resolves from this dict entry.
    if frozen_exec_name.is_some() {
        unsafe {
            let public_name = pyre_object::w_str_new_managed(modulename);
            pyre_object::w_dict_setitem_str(roots.get(globals_slot), "__name__", public_name);
        }
    }

    // Module-level code may have rewritten `sys.modules[name]` (the
    // `decimal` → `_pydecimal` pattern, or PyPy's `_cffi_backend` style
    // late rewiring). Honour that — PyPy: interp_import.importhook
    // reads sys.modules again after exec_code_module via importcache.
    if let Some(replaced) = check_sys_modules(modulename)
        && !std::ptr::eq(replaced, roots.get(module_slot))
    {
        return Ok(replaced);
    }

    // `_bootstrap` has no `sys` or `_imp` of its own until it is handed them,
    // so every entry point through it raises NameError until this runs. Read
    // the module back out of sys.modules rather than reusing the local: the
    // body just executed arbitrary code, and a collection in there relocates
    // a young module while only the dict entry is updated.
    if modulename == "_frozen_importlib"
        && importlib_bootstrap_needs_install()
        && let Some(loaded) = check_sys_modules(modulename)
        && let Err(e) = install_importlib_bootstrap(loaded, execution_context)
    {
        // Unwind the partial install: `dunder_import` routes through
        // `_bootstrap.__import__` whenever `_frozen_importlib` is
        // in `sys.modules`, and a half-installed bootstrap (module
        // registered, PathFinder missing — e.g. `_bootstrap_external`
        // needs the `nt` builtin on Windows) would then answer every
        // import with no file finder installed. Dropping the frozen
        // names keeps the native importer authoritative, the minimal-
        // importer role the boot sequence already documents.
        remove_sys_module("_frozen_importlib");
        remove_sys_module("_frozen_importlib_external");
        return Err(e);
    }

    Ok(roots.get(module_slot))
}

/// PyPy installs the importlib bootstrap once while constructing the object
/// space.  A later source-copy import (the 3.14 importlib tests deliberately
/// remove its module names) executes the source but must not append a second
/// BuiltinImporter/FrozenImporter/PathFinder set to the interpreter state.
#[cfg(feature = "host_env")]
fn importlib_bootstrap_needs_install() -> bool {
    let Some(sys) = get_interpreter_sys_module() else {
        return true;
    };
    !crate::baseobjspace::findattr_result(sys, "_pyre_importlib_bootstrap_installed")
        .ok()
        .flatten()
        .is_some_and(|value| unsafe {
            pyre_object::is_bool(value) && pyre_object::w_bool_get_value(value)
        })
}

/// Wire a freshly executed `_frozen_importlib` into this interpreter.
///
/// PyPy does the same from `_frozen_importlib`'s `install()`: `_install`
/// binds `sys` / `_imp` into the module globals and appends BuiltinImporter
/// and FrozenImporter to `sys.meta_path`, then
/// `_install_external_importers` imports `_frozen_importlib_external` —
/// the same top-level loader `absolute_import` uses for that frozen name —
/// and appends PathFinder.
///
/// PyPy runs it while building the space; doing it as the module finishes
/// loading keeps both bootstrap files off the startup path, since nothing
/// reads `sys.meta_path` before something has imported the machinery that
/// fills it.
#[cfg(feature = "host_env")]
fn install_importlib_bootstrap(
    module: PyObjectRef,
    execution_context: *const PyExecutionContext,
) -> Result<(), crate::PyError> {
    use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};

    let _roots = push_roots();
    let module_slot = shadow_stack_len();
    let _ = pin_root(module);

    // `moduledef.py` `startup_at_translation_time_only` calls
    // `_install(space.getbuiltinmodule('sys'), space.getbuiltinmodule('_imp'))`.
    // The seed already registered both; `absolute_import` would walk the
    // pre-bootstrap finder for the same objects.
    let w_sys = getbuiltinmodule("sys", false, true, execution_context)?.ok_or_else(|| {
        crate::PyError::module_not_found_with_name("No module named sys".to_string(), "sys")
    })?;
    let sys_slot = shadow_stack_len();
    let _ = pin_root(w_sys);

    let w_imp = getbuiltinmodule("_imp", false, true, execution_context)?.ok_or_else(|| {
        crate::PyError::module_not_found_with_name("No module named _imp".to_string(), "_imp")
    })?;
    let imp_slot = shadow_stack_len();
    let _ = pin_root(w_imp);

    // `_bootstrap._setup` walks `sys.modules` and, for a name `_imp.is_frozen`
    // accepts, calls `FrozenImporter._fix_up_module`. That helper reads the
    // public `__origname__` and rejects a `__file__` the source exec stored.
    // `set_frozen_alias_metadata` writes the same origname again afterwards.
    let origname = pyre_object::w_str_new_managed("importlib._bootstrap");
    let origname_slot = shadow_stack_len();
    let _ = pin_root(origname);
    let w_dict = unsafe { pyre_object::w_module_get_w_dict(shadow_stack_get(module_slot)) };
    if !w_dict.is_null() {
        unsafe {
            pyre_object::w_dict_delitem_str(w_dict, "__file__");
            pyre_object::w_dict_delitem_str(w_dict, "__cached__");
        }
    }
    let w_dict = unsafe { pyre_object::w_module_get_w_dict(shadow_stack_get(module_slot)) };
    if !w_dict.is_null() {
        unsafe {
            pyre_object::w_dict_setitem_str(
                w_dict,
                "__origname__",
                shadow_stack_get(origname_slot),
            );
        }
    }

    let install = crate::baseobjspace::getattr_str(shadow_stack_get(module_slot), "_install")?;
    let install_slot = shadow_stack_len();
    let _ = pin_root(install);
    let args = [shadow_stack_get(sys_slot), shadow_stack_get(imp_slot)];
    crate::call::call_function_impl_result(shadow_stack_get(install_slot), &args)?;

    let install_external = crate::baseobjspace::getattr_str(
        shadow_stack_get(module_slot),
        "_install_external_importers",
    )?;
    crate::call::call_function_impl_result(install_external, &[])?;

    if let Some(external) = check_sys_modules("_frozen_importlib_external") {
        set_frozen_alias_metadata(
            external,
            "_frozen_importlib_external",
            shadow_stack_get(module_slot),
        )?;
    }

    // `_install_external_importers` imports `_frozen_importlib_external`.
    // The bootstrap module is already registered under its frozen name;
    // re-registering after both installs have succeeded, then attaching
    // loader metadata, leaves neither frozen name behind when a body raised
    // (the caller unwinds them).
    set_sys_module("_frozen_importlib", shadow_stack_get(module_slot));
    set_frozen_alias_metadata(
        shadow_stack_get(module_slot),
        "_frozen_importlib",
        shadow_stack_get(module_slot),
    )?;

    // `sys.path_hooks.insert(0, zipimporter)` (zipimport moduledef startup /
    // pylifecycle.c init after the external importers) so zip archives on
    // `sys.path` are importable. `zipimport` is served from the frozen table
    // and its body imports `_frozen_importlib`, hence after the
    // `set_frozen_alias_metadata` call that registers that alias.
    // A failed import leaves the hook out — the tolerant `# can't import
    // zipimport` path — rather than failing the whole bootstrap.
    if let Ok(w_zipimport) = absolute_import(
        Wtf8::new("zipimport"),
        pyre_object::PY_NULL,
        execution_context,
    ) {
        let zipimport_slot = shadow_stack_len();
        let _ = pin_root(w_zipimport);
        let w_zipimporter =
            crate::baseobjspace::getattr_str(shadow_stack_get(zipimport_slot), "zipimporter")?;
        let zipimporter_slot = shadow_stack_len();
        let _ = pin_root(w_zipimporter);
        let w_path_hooks =
            crate::baseobjspace::getattr_str(shadow_stack_get(sys_slot), "path_hooks")?;
        unsafe {
            pyre_object::w_list_insert(w_path_hooks, 0, shadow_stack_get(zipimporter_slot));
        }
    }
    crate::baseobjspace::setattr_str(
        shadow_stack_get(sys_slot),
        "_pyre_importlib_bootstrap_installed",
        pyre_object::w_bool_from(true),
    )?;
    Ok(())
}

/// pylifecycle.c `init_importlib` / `init_importlib_external`: run the
/// bootstrap sequence so `sys.meta_path` / `sys.path_hooks` are populated and
/// `importlib.util.find_spec` works — which `runpy._get_module_details` (the
/// `-m` entry) requires. The native importer does not consult `sys.meta_path`,
/// so before this `_frozen_importlib` has neither `sys` nor `_imp` injected
/// and `meta_path` is empty.
///
/// Interpreter startup owns the step, not one launcher: a launcher that skips
/// it leaves `sys.modules["_frozen_importlib"]` absent, which is the one
/// condition under which [`dunder_import`] serves every import natively
/// instead of through `_bootstrap.__import__`.
#[cfg(feature = "host_env")]
pub fn init_importlib_bootstrap(
    canonical: PyObjectRef,
    execution_context: *const PyExecutionContext,
) -> Result<(), crate::PyError> {
    let bootstrapped = bootstrap_importlib_modules(canonical, execution_context);
    // pylifecycle.c init_sys_streams, which follows init_importlib: the
    // standard streams can reach a text codec from here on.  A failed
    // bootstrap leaves the native importer serving imports, so the codec is
    // still reachable and the streams still want it.
    // A bootstrap failure is the more fundamental of the two, so it wins;
    // otherwise a codec the streams could not build is reported rather than
    // leaving a stream that reports itself unreadable. The `Result` is
    // consumed first; a live handle is pinned across the codec init.
    let mut bootstrap_err = match bootstrapped {
        Ok(()) => None,
        Err(err) => Some(err),
    };
    if let Some(mut err) = bootstrap_err.take() {
        let _roots = pyre_object::gc_roots::push_roots();
        let mut err = err;
        let slot = err.pin(&_roots);
        let _stream_codecs = crate::module::sys::vm::init_stream_codecs();
        err.reload(&_roots, slot);
        return Err(err);
    }
    crate::module::sys::vm::init_stream_codecs()
}

/// Off-`host_env` builds reach no bootstrap sources, so the native importer is
/// the only importer and there is nothing to install.
#[cfg(not(feature = "host_env"))]
pub fn init_importlib_bootstrap(
    _canonical: PyObjectRef,
    _execution_context: *const PyExecutionContext,
) -> Result<(), crate::PyError> {
    Ok(())
}

/// `_bootstrap._setup` (importlib/_bootstrap.py) reads the bootstrap builtins
/// `_thread`/`_warnings`/`_weakref` from `sys.modules`, so import them first to
/// seed `sys.modules`; a name already present skips `_builtin_from_name` and
/// keeps the natively-registered module object authoritative.
#[cfg(feature = "host_env")]
fn seed_importlib_bootstrap_builtins(
    execution_context: *const PyExecutionContext,
) -> Result<(), crate::PyError> {
    // `moduledef.py` `startup_at_translation_time_only` passes
    // `space.getbuiltinmodule('sys')` and `getbuiltinmodule('_imp')`.
    // `_setup` reads `_thread` / `_warnings` / `_weakref` from `sys.modules`
    // and calls `_builtin_from_name` only on a miss. That lookup is
    // `getbuiltinmodule`, not `importhook`: `BuiltinImporter.find_spec` is
    // what `_setup` runs later, once, when it stamps the spec.
    for name in ["sys", "_imp", "_thread", "_warnings", "_weakref"] {
        if getbuiltinmodule(name, false, true, execution_context)?.is_none() {
            return Err(crate::PyError::module_not_found_with_name(
                format!("No module named {name}"),
                name,
            ));
        }
    }
    Ok(())
}

#[cfg(feature = "host_env")]
fn bootstrap_importlib_modules(
    canonical: PyObjectRef,
    execution_context: *const PyExecutionContext,
) -> Result<(), crate::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let canon_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(canonical);
    let import = |name: &str| {
        importhook(
            Wtf8::new(name),
            roots.get(canon_slot),
            pyre_object::PY_NULL,
            0,
            execution_context,
        )
    };

    for name in ["_thread", "_warnings", "_weakref"] {
        import(name)?;
    }
    import("sys")?;
    import("_imp")?;
    // Importing `_frozen_importlib` fires `install_importlib_bootstrap`
    // as its body finishes: `_install(sys, _imp)` and
    // `_install_external_importers()`, which links
    // `_frozen_importlib_external` and defines `ModuleSpec`.
    // A cached module skips the hook, so a second startup does not
    // append the importers again.
    import("_frozen_importlib")?;
    Ok(())
}

#[cfg(feature = "host_env")]
fn set_frozen_alias_metadata(
    module: PyObjectRef,
    name: &str,
    bootstrap: PyObjectRef,
) -> Result<(), crate::PyError> {
    use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};

    let _roots = push_roots();
    let module_slot = shadow_stack_len();
    let _ = pin_root(module);
    let bootstrap_slot = shadow_stack_len();
    let _ = pin_root(bootstrap);

    // `_bootstrap._setup` already stored a FrozenImporter spec, and
    // `_fix_up_module` popped `__origname__`. `moduledef.py`
    // `startup_at_translation_time_only` calls `_install` and does not call
    // `FrozenImporter.find_spec` again.
    let origname = match name {
        "_frozen_importlib" => "importlib._bootstrap",
        "_frozen_importlib_external" => "importlib._bootstrap_external",
        _ => name,
    };
    if spec_loader_is_frozen_importer(shadow_stack_get(module_slot)) {
        crate::baseobjspace::setattr_str(
            shadow_stack_get(module_slot),
            "__origname__",
            pyre_object::w_str_new_managed(origname),
        )?;
        return Ok(());
    }
    let loader =
        crate::baseobjspace::getattr_str(shadow_stack_get(bootstrap_slot), "FrozenImporter")?;
    let loader_slot = shadow_stack_len();
    let _ = pin_root(loader);
    let find_spec = crate::baseobjspace::getattr_str(shadow_stack_get(loader_slot), "find_spec")?;
    let find_spec_slot = shadow_stack_len();
    let _ = pin_root(find_spec);
    let name_slot = shadow_stack_len();
    let _ = pin_root(pyre_object::w_str_new_managed(name));
    let spec = crate::call::call_function_impl_result(
        shadow_stack_get(find_spec_slot),
        &[shadow_stack_get(name_slot)],
    )?;
    let spec_slot = shadow_stack_len();
    let _ = pin_root(spec);

    crate::baseobjspace::setattr_str(
        shadow_stack_get(module_slot),
        "__loader__",
        shadow_stack_get(loader_slot),
    )?;
    crate::baseobjspace::setattr_str(
        shadow_stack_get(module_slot),
        "__spec__",
        shadow_stack_get(spec_slot),
    )?;
    // `FrozenImporter._fix_up_module` consumes this when a fresh source copy
    // of importlib repairs the already-loaded essential frozen aliases.
    crate::baseobjspace::setattr_str(
        shadow_stack_get(module_slot),
        "__origname__",
        pyre_object::w_str_new_managed(origname),
    )?;
    Ok(())
}

/// `_setup` stamps `FrozenImporter` as `__spec__.loader` before this runs.
fn spec_loader_is_frozen_importer(module: PyObjectRef) -> bool {
    let Ok(Some(spec)) = crate::baseobjspace::findattr_result(module, "__spec__") else {
        return false;
    };
    if unsafe { pyre_object::is_none(spec) } {
        return false;
    }
    let Ok(loader) = crate::baseobjspace::getattr_str(spec, "loader") else {
        return false;
    };
    let Ok(loader_name) = crate::baseobjspace::getattr_str(loader, "__name__") else {
        return false;
    };
    unsafe {
        pyre_object::is_str(loader_name)
            && pyre_object::w_str_get_wtf8(loader_name) == "FrozenImporter"
    }
}

// ── load_package ─────────────────────────────────────────────────────
// PyPy equivalent: load_module with PKG_DIRECTORY modtype

#[cfg(feature = "host_env")]
fn load_package(
    modulename: &str,
    dirpath: &Path,
    execution_context: *const PyExecutionContext,
) -> Result<PyObjectRef, crate::PyError> {
    // `__path__` / `__package__` are seeded in `load_source_module` BEFORE
    // the body runs (relative imports in `__init__.py` need them in place),
    // and `__init__.py` may legitimately rewrite `__path__` (namespace
    // packages via `pkgutil.extend_path`), so they are not re-stamped here.
    let init_path = dirpath.join("__init__.py");
    load_source_module(modulename, &init_path, Some(dirpath), execution_context)
}

// ── load_namespace_package ───────────────────────────────────────────
// PEP 420: a package directory (or set of directories) with no `__init__.py`.

#[cfg(feature = "host_env")]
fn load_namespace_package(
    modulename: &str,
    dirs: &[PathBuf],
    execution_context: *const PyExecutionContext,
) -> Result<PyObjectRef, crate::PyError> {
    // A namespace package has no source to read or execute: it is a module
    // carrying `__path__` (the portions) and `__package__`, but no `__file__`.
    // Submodule imports resolve against `__path__` exactly as for a regular
    // package.
    let ctx = unsafe { &*execution_context };
    let w_globals = ctx.fresh_module_globals();
    let roots = pyre_object::gc_roots::push_roots();
    let globals_slot = roots.base();
    let _ = roots.pin_root(w_globals);
    let package_name = pyre_object::w_str_new_managed(modulename);
    unsafe {
        pyre_object::w_dict_setitem_str(roots.get(globals_slot), "__package__", package_name);
    }

    let mut path_items = pyre_object::gc_roots::RootedItems::new();
    for d in dirs {
        path_items.push(crate::gateway::fsdecode_os_str(d.as_os_str()));
    }
    unsafe {
        let path_list = pyre_object::w_list_new(path_items.take());
        pyre_object::w_dict_setitem_str(roots.get(globals_slot), "__path__", path_list);
    }

    let mut module = pyre_object::w_module_new_aliasing_dict(modulename, roots.get(globals_slot));
    pyre_object::with_roots!(module => set_sys_module(modulename, module));
    Ok(module)
}

// ── load_part ────────────────────────────────────────────────────────
// PyPy equivalent: importing.py `load_part()`

fn load_part(
    modulename: &Wtf8,
    partname: &Wtf8,
    parent_dirs: Option<&[PathBuf]>,
    execution_context: *const PyExecutionContext,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    // Builtin tables and filesystem paths are `&str`. A name that is not
    // UTF-8 is absent from both: answer the cache or the block, then miss.
    let (Some(modulename), Some(partname)) = (name_utf8(modulename), name_utf8(partname)) else {
        if modules_block(modulename) {
            return Err(import_halted(modulename));
        }
        if let Some(cached) = modules_cached(modulename) {
            return Ok(Some(cached));
        }
        return Ok(None);
    };
    // A blocked name is answered before any search, so it also applies to a
    // name that was never importable to begin with.
    if sys_modules_blocks(modulename) {
        return Err(crate::PyError::module_not_found_with_name(
            format!("import of {modulename} halted; None in sys.modules"),
            modulename,
        ));
    }

    // Check sys.modules cache first
    if let Some(cached) = check_sys_modules(modulename) {
        return Ok(Some(cached));
    }

    // Try a full-name builtin match first so dotted stubs like
    // `importlib.machinery` can override the filesystem search.
    // PyPy: interp_import.importhook consults sys.builtin_module_names by
    // the fully-qualified name.
    let full_is_builtin = BUILTIN_MODULES.lock().contains_key(modulename);
    if full_is_builtin {
        // `interp_imp.py create_builtin` for a name `sys.modules` does not
        // hold, then the spec `module_from_spec` stamps.
        let mut m =
            getbuiltinmodule(modulename, true, false, execution_context)?.ok_or_else(|| {
                crate::PyError::new(
                    crate::PyErrorKind::ImportError,
                    format!("builtin module '{modulename}' failed to initialize"),
                )
            })?;
        pyre_object::with_roots!(m => set_builtin_module_spec(modulename, m))?;
        // The spec runs app-level code, which can rebind the name; re-read
        // the live entry from sys.modules.
        let m = pyre_object::with_roots!(m => check_sys_modules(modulename)).unwrap_or(m);
        return Ok(Some(m));
    }

    // Find the module on disk
    let Some(info) = find_module(partname, parent_dirs)? else {
        return Ok(None);
    };

    let module = match info {
        #[cfg(all(
            feature = "cpyext",
            not(feature = "sandbox"),
            any(target_os = "macos", target_os = "linux")
        ))]
        FindInfo::ExtensionFile { pathname } => {
            let roots = pyre_object::gc_roots::push_roots();
            let spec_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = roots.pin_root(extension_module_spec(modulename, &pathname));
            let module = crate::cpyext::load_extension_module(
                modulename,
                &pathname,
                pyre_object::gc_roots::shadow_stack_get(spec_slot),
            )?;
            let module_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = roots.pin_root(module);
            set_extension_module_spec(
                modulename,
                &pathname,
                pyre_object::gc_roots::shadow_stack_get(module_slot),
            )?;
            // `_bootstrap._load` writes `sys.modules[spec.name]` between
            // `create_module` and `exec_module`, and the single-phase branch of
            // the load above has already registered its own module.  Registering
            // here is what lets a slot that imports its own module find it,
            // which is why the source loader does the same before executing.
            set_sys_module(
                modulename,
                pyre_object::gc_roots::shadow_stack_get(module_slot),
            );
            // `_imp.create_dynamic` leaves the second PEP 489 phase to
            // importlib's `_imp.exec_dynamic`; this importer runs both, so the
            // exec slots run here, after `__spec__` is in place.
            crate::cpyext::exec_dynamic(pyre_object::gc_roots::shadow_stack_get(module_slot))?;
            pyre_object::gc_roots::shadow_stack_get(module_slot)
        }
        #[cfg(all(
            feature = "cpyext",
            not(feature = "sandbox"),
            any(target_os = "macos", target_os = "linux")
        ))]
        FindInfo::ExtensionPackage { dirpath, pathname } => {
            let roots = pyre_object::gc_roots::push_roots();
            let spec_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = roots.pin_root(extension_module_spec(modulename, &pathname));
            let module = crate::cpyext::load_extension_module(
                modulename,
                &pathname,
                pyre_object::gc_roots::shadow_stack_get(spec_slot),
            )?;
            let module_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = roots.pin_root(module);
            set_extension_module_spec(
                modulename,
                &pathname,
                pyre_object::gc_roots::shadow_stack_get(module_slot),
            )?;
            let path = crate::gateway::fsdecode_os_str(dirpath.as_os_str());
            let path_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = roots.pin_root(path);
            // `__path__` is re-stamped after the spec: the loader derives it
            // from the file name, and the directory found here is the same
            // answer spelled without that dependency.
            // The store below allocates, so the fresh list is pinned rather
            // than held in a local across it.
            let list_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = roots.pin_root(pyre_object::w_list_new(vec![
                pyre_object::gc_roots::shadow_stack_get(path_slot),
            ]));
            let module = pyre_object::gc_roots::shadow_stack_get(module_slot);
            crate::module_ns_store(
                unsafe { pyre_object::w_module_get_w_dict(module) },
                "__path__",
                pyre_object::gc_roots::shadow_stack_get(list_slot),
            );
            // Registered before the exec phase, as in the file branch above.
            set_sys_module(
                modulename,
                pyre_object::gc_roots::shadow_stack_get(module_slot),
            );
            // The exec phase runs after `__spec__` and `__path__`, for the same
            // reason it does in the file branch above.
            crate::cpyext::exec_dynamic(pyre_object::gc_roots::shadow_stack_get(module_slot))?;
            pyre_object::gc_roots::shadow_stack_get(module_slot)
        }
        #[cfg(feature = "host_env")]
        FindInfo::SourceFile { pathname } => {
            match load_source_module(modulename, &pathname, None, execution_context) {
                Ok(m) => m,
                Err(e) => {
                    return Err(e);
                }
            }
        }
        #[cfg(feature = "host_env")]
        FindInfo::Package { dirpath } => load_package(modulename, &dirpath, execution_context)?,
        #[cfg(feature = "host_env")]
        FindInfo::Namespace { dirs } => {
            load_namespace_package(modulename, &dirs, execution_context)?
        }
        FindInfo::Builtin => {
            // Same `create_builtin` as the full_is_builtin branch above; a
            // builtin is found only for a top-level name, so `partname` is
            // `modulename`.
            let mut m =
                getbuiltinmodule(partname, true, false, execution_context)?.ok_or_else(|| {
                    crate::PyError::new(
                        crate::PyErrorKind::ImportError,
                        format!("builtin module '{modulename}' failed to initialize"),
                    )
                })?;
            pyre_object::with_roots!(m => set_builtin_module_spec(partname, m))?;
            // The spec runs app-level code, which can rebind the name; re-read
            // the live entry from sys.modules.
            pyre_object::with_roots!(m => check_sys_modules(modulename)).unwrap_or(m)
        }
    };

    Ok(Some(module))
}

// ── _absolute_import ─────────────────────────────────────────────────
// PyPy equivalent: importing.py `_absolute_import()`

/// The on-disk file a submodule load of `importlib._bootstrap` or
/// `importlib._bootstrap_external` would open. Locating the `importlib`
/// package directory does not execute `importlib/__init__.py`.
#[cfg(feature = "host_env")]
fn frozen_bootstrap_source(modulename: &str) -> Result<Option<PathBuf>, crate::PyError> {
    let leaf = match modulename {
        "_frozen_importlib" => "_bootstrap",
        "_frozen_importlib_external" => "_bootstrap_external",
        _ => return Ok(None),
    };
    let Some(info) = find_module("importlib", None)? else {
        return Ok(None);
    };
    let FindInfo::Package { dirpath } = info else {
        return Ok(None);
    };
    let pathname = dirpath.join(format!("{leaf}.py"));
    if with_source_provider(|p| p.is_file(&pathname)) {
        Ok(Some(pathname))
    } else {
        Ok(None)
    }
}

/// A WTF-8 module name that is also UTF-8. Surrogate bytes are the `ED A0`
/// pattern; everything else is viewed as `&str` without a `Result`.
fn name_utf8(name: &Wtf8) -> Option<&str> {
    let bytes = name.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == 0xED
            && index + 2 < bytes.len()
            && bytes[index + 1] >= 0xA0
            && bytes[index + 1] <= 0xBF
        {
            return None;
        }
        index += 1;
    }
    Some(unsafe { std::str::from_utf8_unchecked(bytes) })
}

/// `sys.modules` block for a module name carried as WTF-8.
fn modules_block(name: &Wtf8) -> bool {
    if let Some(utf8) = name_utf8(name) {
        sys_modules_blocks(utf8)
    } else {
        sys_modules_blocks_wtf8(name)
    }
}

/// `check_sys_modules` for a module name carried as WTF-8. A UTF-8 name
/// keeps the `&str` dict probe; a lone surrogate is looked up as an object.
fn modules_cached(name: &Wtf8) -> Option<PyObjectRef> {
    if let Some(utf8) = name_utf8(name) {
        check_sys_modules(utf8)
    } else {
        check_sys_modules_w(pyre_object::w_str_from_wtf8_managed(name.to_wtf8_buf()))
    }
}

fn module_not_found_name(name: &Wtf8) -> crate::PyError {
    let mut message = Wtf8Buf::new();
    message.push_str("No module named '");
    message.push_wtf8(name);
    message.push_str("'");
    crate::PyError::module_not_found_with_name_obj(
        message,
        pyre_object::w_str_from_wtf8_managed(name.to_wtf8_buf()),
    )
}

fn module_not_found_parent(full: &Wtf8, parent: &Wtf8) -> crate::PyError {
    let mut message = Wtf8Buf::new();
    message.push_str("No module named '");
    message.push_wtf8(full);
    message.push_str("'; '");
    message.push_wtf8(parent);
    message.push_str("' is not a package");
    crate::PyError::module_not_found_with_name_obj(
        message,
        pyre_object::w_str_from_wtf8_managed(full.to_wtf8_buf()),
    )
}

fn import_halted(name: &Wtf8) -> crate::PyError {
    let mut message = Wtf8Buf::new();
    message.push_str("import of ");
    message.push_wtf8(name);
    message.push_str(" halted; None in sys.modules");
    crate::PyError::module_not_found_with_name_obj(
        message,
        pyre_object::w_str_from_wtf8_managed(name.to_wtf8_buf()),
    )
}

/// Split a module name on `.`. The separator is ASCII, so it never falls
/// inside a WTF-8 sequence.
fn split_module_name(name: &Wtf8) -> Vec<&Wtf8> {
    let bytes = name.as_bytes();
    let mut parts = Vec::new();
    let mut start = 0;
    for (index, &byte) in bytes.iter().enumerate() {
        if byte == b'.' {
            parts.push(unsafe { Wtf8::from_bytes_unchecked(&bytes[start..index]) });
            start = index + 1;
        }
    }
    parts.push(unsafe { Wtf8::from_bytes_unchecked(&bytes[start..]) });
    parts
}

fn join_module_name(parts: &[&Wtf8]) -> Wtf8Buf {
    let mut joined = Wtf8Buf::new();
    for (index, part) in parts.iter().enumerate() {
        if index != 0 {
            joined.push_str(".");
        }
        joined.push_wtf8(part);
    }
    joined
}

fn absolute_import(
    modulename: &Wtf8,
    mut w_fromlist: PyObjectRef,
    execution_context: *const PyExecutionContext,
) -> Result<PyObjectRef, crate::PyError> {
    // `moduledef.py Module.install` execs `importlib/_bootstrap.py` and
    // `importlib/_bootstrap_external.py` into top-level modules registered
    // only under the frozen names, leaving `importlib/__init__.py` unexecuted.
    // The package's `else` arm publishes the public aliases later, when
    // something imports `importlib`.
    if let Some(frozen) = name_utf8(modulename)
        && matches!(frozen, "_frozen_importlib" | "_frozen_importlib_external")
    {
        // `sys.modules[name] = None` is an explicit import block.  In
        // particular, `test.support.import_helper.import_fresh_module` uses
        // it to force importlib's source bootstrap: the failed frozen import
        // is what selects the `except ImportError` arm that calls
        // `_bootstrap._setup(sys, _imp)`.  Do not turn that sentinel back
        // into the source module.
        if sys_modules_blocks(frozen) {
            return Err(import_halted(modulename));
        }
        if let Some(cached) = pyre_object::with_roots!(w_fromlist => check_sys_modules(frozen)) {
            return Ok(cached);
        }
        #[cfg(feature = "host_env")]
        if let Some(pathname) =
            pyre_object::with_roots!(w_fromlist => frozen_bootstrap_source(frozen))?
        {
            return load_source_module(frozen, &pathname, None, execution_context);
        }
    }

    let parts = split_module_name(modulename);
    // Each iteration's module is read by the next iteration's cache and
    // package checks, which can collect. `pin_root` publishes that module
    // (`framework.py` `push_roots` writes one livevar) and the next iteration
    // reloads the slot, so a collection forwards the pointer it reads.
    let loop_roots = pyre_object::gc_roots::push_roots();
    let mut first_slot: Option<usize> = None;
    let mut parent_slot: Option<usize> = None;
    let mut prefix: Vec<&Wtf8> = Vec::new();

    for (level, &part) in parts.iter().enumerate() {
        prefix.push(part);
        let full_name = join_module_name(&prefix);
        // A submodule is resolved against its parent package's `__path__`;
        // top-level names (level 0, no parent) search sys.path.
        //
        // `_bootstrap._find_and_load_unlocked` fixes the order: a sys.modules
        // hit for the full name wins first (`load_part` answers both the cached
        // entry and the blocked-name sentinel), and only then must the parent be
        // a package. Presence of `__path__` is the whole test — a PEP 420
        // `_NamespacePath` (or any other non-list value) still marks a package
        // even though `parent_package_path` cannot turn it into directories.
        // Without the check a dotted name whose parent is a plain module falls
        // back to the top-level search and can resolve to a same-leaf builtin.
        // `_bootstrap._find_and_load` returns a `sys.modules` hit before
        // `_find_and_load_unlocked` runs, so the parent binding below is the
        // business of a load this call actually performed.  Read before
        // `load_part`, which answers the cache and cannot be asked afterwards.
        let was_cached =
            pyre_object::with_roots!(w_fromlist => modules_cached(&full_name)).is_some();
        let parent_live = match parent_slot {
            Some(slot) => Some(loop_roots.get(slot)),
            None => None,
        };
        let parent_dirs = match parent_live {
            None => None,
            Some(_) if was_cached || modules_block(&full_name) => None,
            Some(mut parent_mod) => {
                if pyre_object::with_roots!(parent_mod, w_fromlist => crate::baseobjspace::findattr_result(parent_mod, "__path__"))?.is_none() {
                    let parent_name = join_module_name(&parts[..level]);
                    return Err(module_not_found_parent(&full_name, &parent_name));
                }
                pyre_object::with_roots!(parent_mod, w_fromlist => parent_package_path(parent_mod))?
            }
        };
        let w_mod = pyre_object::with_roots!(w_fromlist => load_part(&full_name, part, parent_dirs.as_deref(), execution_context))?;
        let Some(module) = w_mod else {
            // _bootstrap.py:1335 raises for the prefix that actually failed
            // (`name=name`): `import a.b.c` with `a.b` missing reports `a.b`.
            return Err(module_not_found_name(&full_name));
        };
        let module_slot = pyre_object::gc_roots::shadow_stack_len();
        let mut module = loop_roots.pin_root(module);
        // _bootstrap._find_and_load (_bootstrap.py): bind the
        // submodule as an attribute of its parent package so `import a.b`
        // makes `a.b` reachable. Only an AttributeError is swallowed (with an
        // ImportWarning); any other exception propagates.
        // A child whose name is not UTF-8 was answered from `sys.modules`
        // (`was_cached`) or not found; the attribute bind is a `&str` store.
        if !was_cached
            && let Some(mut parent_mod) = match parent_slot {
                Some(slot) => Some(loop_roots.get(slot)),
                None => None,
            }
            && let Some(part_utf8) = name_utf8(part)
            && let Err(err) = pyre_object::with_roots!(module, parent_mod, w_fromlist => {
                crate::setattr_str(parent_mod, part_utf8, module)
            })
        {
            if err.kind != crate::PyErrorKind::AttributeError {
                return Err(err);
            }
            let parent_name = join_module_name(&parts[..level]);
            if let Some(parent_utf8) = name_utf8(&parent_name) {
                let message = format!(
                    "Cannot set an attribute on '{parent_utf8}' for child module '{part_utf8}'"
                );
                pyre_object::with_roots!(module, w_fromlist => {
                    crate::warn::warn(&message, "ImportWarning");
                });
            }
        }
        if level == 0 {
            first_slot = Some(module_slot);
        }
        parent_slot = Some(module_slot);
    }
    let mut first = match first_slot {
        Some(slot) => Some(loop_roots.get(slot)),
        None => None,
    };
    drop(loop_roots);

    // PyPy: if w_fromlist is not None, return the leaf module.
    // Otherwise, return the first (top-level) module.
    if !w_fromlist.is_null() && !unsafe { is_none(w_fromlist) } {
        // `from X.Y import Z` → return the leaf module (Y)
        let roots = pyre_object::gc_roots::push_roots();
        let base = roots.pin_roots(&[first.unwrap_or(pyre_object::PY_NULL)]);
        let cached = modules_cached(modulename);
        let w = roots.get(base);
        first = if w.is_null() { None } else { Some(w) };
        drop(roots);
        if let Some(cached) = cached {
            return Ok(cached);
        }
    }

    // `import X.Y` → return the top-level module (X)
    first.ok_or_else(|| module_not_found_name(modulename))
}

// ── IMPORT_NAME ──────────────────────────────────────────────────────

/// PyPy equivalent: pyopcode.py `IMPORT_NAME`.
/// `pyopcode.py` `IMPORT_NAME`'s `self.get_builtin().getdictvalue(space,
/// '__import__')`, on its own so the interpreter and the JIT's `LoadImport`
/// residual read the importer the same way.  The `is_module` test has no
/// upstream counterpart: `get_builtin` can hand back a plain mapping, which
/// carries no module dict to look the name up in.
pub fn lookup_dunder_import(frame: &PyFrame) -> Result<PyObjectRef, crate::PyError> {
    let w_builtin = frame.get_builtin();
    let w_import = if !w_builtin.is_null() && unsafe { is_module(w_builtin) } {
        let w_dict = unsafe { pyre_object::w_module_get_w_dict(w_builtin) };
        if w_dict.is_null() {
            None
        } else {
            crate::baseobjspace::finditem_str(w_dict, "__import__")?
        }
    } else {
        None
    };
    w_import
        .ok_or_else(|| crate::PyError::new(crate::PyErrorKind::ImportError, "__import__ not found"))
}

/// IMPORT_NAME's third argument (`pyopcode.py`).  `getdebug()`
/// peeks: a frame that never materialized its locals mapping passes `None`
/// instead of creating one, which is what separates this from LOAD_LOCALS'
/// `getorcreatedebug()`.  Split out so the JIT's `LoadImportLocals` residual
/// and the interpreter answer from the same code.
pub fn import_locals(frame: &PyFrame) -> PyObjectRef {
    match frame.getdebug() {
        Some(d) if !d.w_locals.is_null() => d.w_locals,
        _ => pyre_object::w_none(),
    }
}

pub fn import_name(
    frame: &mut PyFrame,
    w_modulename: PyObjectRef,
    w_fromlist: PyObjectRef,
    w_flag: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    // `lookup_dunder_import` reads the builtin dict and can collect.
    // The three incoming GCREFs are native copies the collector does
    // not update, so they live on the shadow stack across that lookup
    // (`pyopcode.py IMPORT_NAME` livevars).
    let _import_name_roots = pyre_object::gc_roots::push_roots();
    let name_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_modulename);
    let fromlist_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_fromlist);
    let flag_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_flag);
    let w_import = lookup_dunder_import(frame)?;
    let import_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_import);
    let w_locals = import_locals(frame);
    let locals_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_locals);
    let w_globals = frame.get_w_globals();
    let globals_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_globals);
    crate::call::call_args_in_frame(
        frame,
        pyre_object::gc_roots::shadow_stack_get(import_slot),
        &[
            pyre_object::gc_roots::shadow_stack_get(name_slot),
            pyre_object::gc_roots::shadow_stack_get(globals_slot),
            pyre_object::gc_roots::shadow_stack_get(locals_slot),
            pyre_object::gc_roots::shadow_stack_get(fromlist_slot),
            pyre_object::gc_roots::shadow_stack_get(flag_slot),
        ],
    )
}

// ── __import__ ───────────────────────────────────────────────────────
// PyPy equivalent: _frozen_importlib/interp_import.py `interp___import__`

/// `_gcd_import` cache probe: `sys.modules` plus `__spec__._initializing`.
///
/// `interp_import.py _gcd_import` raises `FastPathGiveUp` on a miss, a
/// missing `__spec__`, or a truthy `_initializing`.  A `__spec__` without
/// `_initializing` counts as initialised (a builtin module).
enum GcdCache {
    Miss,
    Ready(PyObjectRef),
    Initializing(PyObjectRef),
}

fn gcd_import_cache_probe(name: &str) -> Result<GcdCache, crate::PyError> {
    // `interp_import.py _gcd_import` — `space.sys.get('modules')` then
    // `finditem_str`.  No shadow-stack pin: the probe must stay free of
    // effects so a look-inside of `__import__` can record the dict read
    // and the two getattrs.  A Python `None` sentinel is FastPathGiveUp
    // (`getattr(None, "__spec__")` is AttributeError) and the slow path
    // raises `import of {name} halted; None in sys.modules`.
    let Some(w_module) = check_sys_modules(name) else {
        return Ok(GcdCache::Miss);
    };
    gcd_import_cache_probe_after(w_module)
}

/// Object-keyed cache probe for the look-inside `__import__` body.
///
/// The wrapper already holds the name object; looking it up with
/// [`check_sys_modules_import`] keeps `&str` off the executed residual.
fn gcd_import_cache_probe_w(w_name: PyObjectRef) -> Result<GcdCache, crate::PyError> {
    let Some(w_module) = check_sys_modules_import(w_name)? else {
        return Ok(GcdCache::Miss);
    };
    gcd_import_cache_probe_after(w_module)
}

fn gcd_import_cache_probe_after(mut w_module: PyObjectRef) -> Result<GcdCache, crate::PyError> {
    // Exact module only — the same cut `sys_module_if_initialized` already
    // applies.  A non-module is FastPathGiveUp: `findattr` would enter
    // `getattr_str_impl` (`force` + `pin_root` + `w_str_new_managed`), and
    // after the red `sys.modules` pointer the descent scan cannot prove
    // `is_module`, so that arm would residualise the whole `__import__`.
    if !unsafe { pyre_object::is_module(w_module) } {
        return Ok(GcdCache::Miss);
    }
    let dict = unsafe { pyre_object::w_module_get_w_dict(w_module) };
    if dict.is_null() {
        return Ok(GcdCache::Miss);
    }
    let mut w_spec = pyre_object::with_roots!(w_module => module_dict_get_spec(dict));
    if import_lookup_is_err(w_spec) {
        return Err(take_published_residual_error());
    }
    if w_spec.is_null() {
        return Ok(GcdCache::Miss);
    }
    if unsafe { pyre_object::is_none(w_spec) } {
        return Ok(GcdCache::Ready(w_module));
    }
    // `space.getattr(w_spec, "_initializing")`.  AttributeError is
    // "initialized" (a builtin module).  The key literal stays inside the
    // residual so stringbuilder stays off the look-inside graph.  `1` is
    // initializing, `0` is ready, `-1` is `FastPathGiveUp` (`_gcd_import`'s
    // miss): the read would have run Python.
    let state = pyre_object::with_roots!(w_module, w_spec => module_spec_get_initializing(w_spec));
    if state < 0 {
        return Ok(GcdCache::Miss);
    }
    if state > 0 {
        return Ok(GcdCache::Initializing(w_module));
    }
    Ok(GcdCache::Ready(w_module))
}

/// `_gcd_import` fast path: the already-imported module for `name`, after
/// waiting for another thread to finish initialising it.
///
/// PyPy's `interp_import.py:_gcd_import` performs the same `sys.modules` /
/// `__spec__._initializing` inspection.  CPython 3.14's
/// `import_ensure_initialized` supplies the missing concurrent-import tail:
/// call `_lock_unlock_module`, whose deadlock handler deliberately accepts a
/// partially initialised module for a concurrent circular import.  Re-read
/// `sys.modules` afterwards, as `PyImport_ImportModuleLevelObject` does, so an
/// import that failed and removed or replaced its module is retried by the
/// slow path instead of returning a stale object.
///
/// `None` means the slow path must run — a missing `sys.modules` entry, a
/// missing `__spec__`, or an entry removed/replaced while we waited.  A
/// `__spec__` without `_initializing` counts as initialised (a builtin module).
fn gcd_import_fast(name: &str) -> Result<Option<PyObjectRef>, crate::PyError> {
    match gcd_import_cache_probe(name)? {
        GcdCache::Miss => Ok(None),
        GcdCache::Ready(w_module) => Ok(Some(w_module)),
        GcdCache::Initializing(w_module) => {
            // The wait is a Python call into `_lock_unlock_module`.  Keep it
            // off the look-inside graph so `test_import.test_import_in_function`
            // can see only the initialized-module arm (`guard_not_invalidated`).
            wait_initializing_module(name, w_module)
        }
    }
}

/// Look-inside `_gcd_import` fast path: same as [`gcd_import_fast`] but
/// the `sys.modules` probe is keyed by the name object.
fn gcd_import_fast_w(
    name: &str,
    w_name: PyObjectRef,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    match gcd_import_cache_probe_w(w_name)? {
        GcdCache::Miss => Ok(None),
        GcdCache::Ready(w_module) => Ok(Some(w_module)),
        GcdCache::Initializing(w_module) => wait_initializing_module(name, w_module),
    }
}

/// JIT residual view of `_gcd_import`: initialized `sys.modules` entry,
/// or `None` on every `FastPathGiveUp` case *and* on any shape whose
/// `__spec__` / `_initializing` read would run Python.
///
/// The residual is a non-forcing `CallR` so
/// `test_import.test_import_in_function` stays `guard_not_invalidated`
/// only.  A non-default module `__getattribute__`, a data descriptor, or a
/// non-bool `_initializing` declines; the original `IMPORT_NAME` then
/// runs `dunder_import` (including hooks and the 3.14 wait).
///
/// Exact `module` uses `Module.descr_getattribute`, not
/// `object.__getattribute__`.  The object-default check would decline every
/// real module and force the slow `CallMayForce` importer on every
/// `import math`.
pub fn sys_module_if_initialized(name: &str) -> Option<PyObjectRef> {
    if sys_modules_blocks_no_callback(name)? {
        return None;
    }
    let w_module = check_sys_modules_no_callback(name)??;
    if !unsafe { pyre_object::is_module(w_module) } {
        return None;
    }
    let w_type = unsafe { (*w_module).w_class };
    if unsafe { crate::baseobjspace::module_getattribute_if_not_from_default(w_type) }.is_some() {
        return None;
    }
    // `getattr` consults a data descriptor before the instance dict.
    if unsafe { crate::baseobjspace::type_lookup_is_data_descr(w_type, "__spec__") } {
        return None;
    }
    let dict = unsafe { pyre_object::w_module_get_w_dict(w_module) };
    if dict.is_null() {
        return None;
    }
    let w_spec = match dict_getitem_str_no_callback(dict, "__spec__")? {
        None => {
            // Dict miss is `getattr` `AttributeError` only when the type
            // does not still bind `__spec__`.
            if unsafe { crate::baseobjspace::lookup_in_type(w_type, "__spec__") }.is_some() {
                return None;
            }
            return None;
        }
        Some(spec) if spec.is_null() || unsafe { pyre_object::is_none(spec) } => {
            // `getattr(None, "_initializing")` is `AttributeError`;
            // `_gcd_import` then treats the module as initialized.
            return Some(w_module);
        }
        Some(spec) => spec,
    };
    let spec_type = unsafe { (*w_spec).w_class };
    if unsafe { crate::baseobjspace::getattribute_if_not_from_object(spec_type) }.is_some() {
        return None;
    }
    if unsafe { crate::baseobjspace::type_lookup_is_data_descr(spec_type, "_initializing") } {
        return None;
    }
    if !unsafe { crate::objspace::std::mapdict::has_mapdict_storage(w_spec) } {
        return None;
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let spec_slot = pyre_object::gc_roots::pin_roots(&[w_spec, w_module, spec_type]);
    let module_slot = spec_slot + 1;
    let type_slot = spec_slot + 2;
    let spec_dict = crate::objspace::std::mapdict::_obj_getdict(
        pyre_object::gc_roots::shadow_stack_get(spec_slot),
    );
    let w_module = pyre_object::gc_roots::shadow_stack_get(module_slot);
    let spec_type = pyre_object::gc_roots::shadow_stack_get(type_slot);
    if spec_dict.is_null() {
        return None;
    }
    match dict_getitem_str_no_callback(spec_dict, "_initializing")? {
        None => {
            // Dict miss is "initialized" only when getattr would not still
            // bind a type-level non-data descriptor.
            if unsafe { crate::baseobjspace::lookup_in_type(spec_type, "_initializing") }.is_some()
            {
                None
            } else {
                Some(w_module)
            }
        }
        Some(flag) if unsafe { pyre_object::is_bool(flag) } => {
            if unsafe { pyre_object::w_bool_get_value(flag) } {
                None
            } else {
                Some(w_module)
            }
        }
        Some(_) => None,
    }
}

/// `w_dict_getitem_str` without running a stored key's `__eq__`.
///
/// Object-strategy dicts can hold a non-string key whose hash collides with
/// `key`; the infallible getitem then calls that `__eq__` and swallows a
/// raise as a miss.  A broken callback-free probe is `None` so the residual
/// declines instead of executing the hook and letting `IMPORT_NAME` run it
/// again.
fn dict_getitem_str_no_callback(dict: PyObjectRef, key: &str) -> Option<Option<PyObjectRef>> {
    pyre_object::dict_eq_hook::begin_callback_free_probe();
    let hit = unsafe { pyre_object::w_dict_getitem_str(dict, key) };
    if pyre_object::dict_eq_hook::end_callback_free_probe() {
        None
    } else {
        Some(hit)
    }
}

/// `interp___import__` `space.findattr(w_mod, "__path__")` without hooks.
/// `None` is hooky (decline the residual). `Some(false)` is a genuine
/// attribute miss. `Some(true)` is a present `__path__`, including Python
/// `None`: `findattr` then returns `w_None`, which is not RPython `None`,
/// so `_handle_fromlist` still runs.
pub fn module_is_package_no_callback(w_module: PyObjectRef) -> Option<bool> {
    let w_type = unsafe { (*w_module).w_class };
    if unsafe { crate::baseobjspace::module_getattribute_if_not_from_default(w_type) }.is_some() {
        return None;
    }
    if unsafe { crate::baseobjspace::type_lookup_is_data_descr(w_type, "__path__") } {
        return None;
    }
    let dict = unsafe { pyre_object::w_module_get_w_dict(w_module) };
    if dict.is_null() {
        return None;
    }
    match dict_getitem_str_no_callback(dict, "__path__")? {
        None => {
            if unsafe { crate::baseobjspace::lookup_in_type(w_type, "__path__") }.is_some() {
                None
            } else {
                Some(false)
            }
        }
        Some(_) => Some(true),
    }
}

fn check_sys_modules_no_callback(name: &str) -> Option<Option<PyObjectRef>> {
    let dict = sys_modules_dict();
    if dict.is_null() {
        return Some(sys_modules_registry_get(name));
    }
    let entry = dict_getitem_str_no_callback(dict, name)?;
    Some(entry.filter(|m| !m.is_null() && !unsafe { pyre_object::is_none(*m) }))
}

fn sys_modules_blocks_no_callback(name: &str) -> Option<bool> {
    let dict = sys_modules_dict();
    if dict.is_null() {
        return Some(false);
    }
    let hit = dict_getitem_str_no_callback(dict, name)?;
    Some(hit.is_some_and(|m| !m.is_null() && unsafe { pyre_object::is_none(m) }))
}

/// Concurrent-import tail of `_gcd_import`: wait for `__spec__._initializing`
/// to clear, then re-read `sys.modules`.
///
/// `interp_import.py _gcd_import` raises `FastPathGiveUp` when `_initializing`
/// is true.  The 3.14 `import_ensure_initialized` wait lives here instead, as
/// a residual, so the initialized arm stays a small look-inside body.
#[majit_macros::dont_look_inside]
pub(crate) fn wait_initializing_module(
    name: &str,
    w_module: PyObjectRef,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};

    let _roots = push_roots();
    let mod_slot = shadow_stack_len();
    let _ = pin_root(w_module);
    let Some(w_bootstrap) = importlib_bootstrap_module() else {
        return Ok(None);
    };
    let bootstrap_slot = shadow_stack_len();
    let _ = pin_root(w_bootstrap);
    let Some(w_lock_unlock) = crate::baseobjspace::findattr_result(
        shadow_stack_get(bootstrap_slot),
        "_lock_unlock_module",
    )?
    else {
        return Ok(None);
    };
    let lock_unlock_slot = shadow_stack_len();
    let _ = pin_root(w_lock_unlock);
    let name_slot = shadow_stack_len();
    let _ = pin_root(pyre_object::w_str_new_managed(name));
    crate::call::call_function_impl_result(
        shadow_stack_get(lock_unlock_slot),
        &[shadow_stack_get(name_slot)],
    )?;

    let Some(w_current) = check_sys_modules(name) else {
        return Ok(None);
    };
    if w_current != shadow_stack_get(mod_slot) {
        return Ok(None);
    }
    Ok(Some(shadow_stack_get(mod_slot)))
}

/// `interp_import.py interp___import__` — the `except OperationError` arm:
///
/// ```python
/// e.remove_traceback_module_frames(
///       '<frozen importlib._bootstrap>',
///       '<frozen importlib._bootstrap_external>',
///       '<builtin>/frozen importlib._bootstrap_external')
/// raise
/// ```
///
/// Drops the leading traceback entries that belong to the importlib bootstrap
/// so an import error does not expose its internal `__import__` /
/// `_find_and_load` machinery.  pyre runs the bootstrap from the on-disk
/// `importlib/_bootstrap{,_external}.py` sources rather than frozen modules,
/// so those two path spellings are named alongside upstream's three.
fn strip_bootstrap_traceback_frames(mut err: crate::PyError) -> crate::PyError {
    err.remove_traceback_module_frames(&[
        "<frozen importlib._bootstrap>",
        "<frozen importlib._bootstrap_external>",
        "<builtin>/frozen importlib._bootstrap_external",
        "importlib/_bootstrap.py",
        "importlib/_bootstrap_external.py",
    ]);
    err
}

/// Native counterpart of RPython `str.find(char, start) -> Signed`.
///
/// `interp_import.py interp___import__` reads `name.find(".")` as a Signed
/// with `-1` for "no dot".  Rust's `str::find` returns `Option<usize>` and
/// takes no start, so the interpreter carries this exact compatibility
/// function and the translator emits the orthodox RPython string method
/// instead of tracing this body.  The only caller passes `0`, and an ASCII
/// dot is always a UTF-8 boundary.
fn rpython_str_find_char(value: &str, needle: char, start: i64) -> i64 {
    let Ok(start) = usize::try_from(start) else {
        return -1;
    };
    // No `.and_then` closures: each is an un-lowered synthetic ctor the
    // descent scan declines on after `sys_modules_dict`.
    let Some(tail) = value.get(start..) else {
        return -1;
    };
    let Some(offset) = tail.find(needle) else {
        return -1;
    };
    match i64::try_from(start + offset) {
        Ok(index) => index,
        Err(_) => -1,
    }
}

/// Native counterpart of RPython `s[:stop]`.
///
/// `interp_import.py` slices the dotted name as `name[:dotindex]`.  The
/// caller has just proved that `stop` is a non-negative byte index returned by
/// `rpython_str_find_char`, so translation replaces this exact helper with the
/// existing post-annotation `getslice(s, 0, stop)` marker.
fn rpython_str_slice_prefix(value: &str, stop: i64) -> &str {
    // No `expect`: the panic formatter is `__majit_stringbuilder_new`.
    let Ok(stop) = usize::try_from(stop) else {
        return value;
    };
    match value.get(..stop) {
        Some(prefix) => prefix,
        None => value,
    }
}

/// `space.getbuiltinmodule('_frozen_importlib')` then that module's
/// `_handle_fromlist`, or null when the binding is absent.
///
/// Null also when `__getattribute__` is not `Module.descr_getattribute`,
/// or when `_handle_fromlist` is a data descriptor on the type.
/// `descr_getattribute` runs `object_getattribute` before the instance
/// dict, so a raw dict read would skip that override. The caller stays
/// on [`dunder_import_package_fromlist`], which resolves the name with
/// `findattr_result` (`space.call_method`).
///
/// The name literals stay here, the same way [`module_dict_cell_get_path`]
/// keeps `"__path__"`. A comparison `begin_callback_free_probe` refuses
/// answers null.
#[majit_macros::dont_look_inside_cannot_raise]
pub(crate) fn bootstrap_handle_fromlist() -> PyObjectRef {
    pyre_object::dict_eq_hook::begin_callback_free_probe();
    let found = unsafe { bootstrap_handle_fromlist_lookup() };
    if pyre_object::dict_eq_hook::end_callback_free_probe() {
        pyre_object::PY_NULL
    } else {
        found
    }
}

unsafe fn bootstrap_handle_fromlist_lookup() -> PyObjectRef {
    let w_bootstrap = sys_modules_exact_str("_frozen_importlib");
    let mut w_bootstrap = if w_bootstrap.is_null() {
        sys_modules_exact_str("importlib._bootstrap")
    } else {
        w_bootstrap
    };
    if w_bootstrap.is_null() || !pyre_object::is_module(w_bootstrap) {
        return pyre_object::PY_NULL;
    }
    let mut w_type = (*w_bootstrap).w_class;
    if w_type.is_null()
        || pyre_object::with_roots!(w_bootstrap, w_type => {
            crate::baseobjspace::module_getattribute_if_not_from_default(w_type)
        })
        .is_some()
        || pyre_object::with_roots!(w_bootstrap => {
            crate::baseobjspace::type_lookup_is_data_descr(w_type, "_handle_fromlist")
        })
    {
        return pyre_object::PY_NULL;
    }
    let dict = pyre_object::w_module_get_w_dict(w_bootstrap);
    if dict.is_null() {
        return pyre_object::PY_NULL;
    }
    let strategy = pyre_object::dictmultiobject::w_module_dict_strategy_or_null(dict);
    if strategy.is_null() {
        return pyre_object::PY_NULL;
    }
    let raw = (*strategy)
        .getdictvalue_no_unwrapping(dict, "_handle_fromlist")
        .unwrap_or(pyre_object::PY_NULL);
    if raw.is_null() {
        return pyre_object::PY_NULL;
    }
    let value = pyre_object::celldict::unwrap_cell(raw);
    if value.is_null() {
        pyre_object::PY_NULL
    } else {
        value
    }
}

/// Exact-str `sys.modules` read used by [`bootstrap_handle_fromlist`].
/// Miss and a non-module dict are null. The checked raising arm of
/// [`sys_modules_finditem_str_exact`] stays out: this reader cannot raise.
unsafe fn sys_modules_exact_str(name: &str) -> PyObjectRef {
    let dict = SYS_MODULES_DICT.load(Ordering::Acquire) as PyObjectRef;
    if dict.is_null() {
        return pyre_object::PY_NULL;
    }
    let module_strategy = pyre_object::dictmultiobject::w_module_dict_strategy_or_null(dict);
    if !module_strategy.is_null() {
        return (*module_strategy)
            .getitem_str(dict, name)
            .unwrap_or(pyre_object::PY_NULL);
    }
    pyre_object::dictmultiobject::w_dict_get_strategy(dict)
        .getitem_str(dict, name)
        .unwrap_or(pyre_object::PY_NULL)
}

/// `space.call_method` of a three-argument callable, as five words.
///
/// `bh_call_fn_3` (`call_jit.rs`) is the same layout — callable,
/// null-or-self, then the positional arguments — and lives in `pyre-jit`,
/// which cannot be called from here. `null_or_self` null is a plain call.
/// A raise is published and answered as [`import_lookup_err_ptr`].
#[majit_macros::dont_look_inside]
pub extern "C" fn jit_portal_call_3(
    callable: PyObjectRef,
    null_or_self: PyObjectRef,
    a0: PyObjectRef,
    a1: PyObjectRef,
    a2: PyObjectRef,
) -> PyObjectRef {
    // `call_function_impl_result` runs Python and can collect. Pin every
    // word first, the same way `handle_fromlist_fast` does, and pass the
    // reloaded slots. A null `null_or_self` stays null.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[callable, null_or_self, a0, a1, a2]);
    let callable = pyre_object::gc_roots::shadow_stack_get(base);
    let null_or_self = pyre_object::gc_roots::shadow_stack_get(base + 1);
    let a0 = pyre_object::gc_roots::shadow_stack_get(base + 2);
    let a1 = pyre_object::gc_roots::shadow_stack_get(base + 3);
    let a2 = pyre_object::gc_roots::shadow_stack_get(base + 4);
    let result = if null_or_self.is_null() {
        crate::call::call_function_impl_result(callable, &[a0, a1, a2])
    } else {
        crate::call::call_function_impl_result(callable, &[null_or_self, a0, a1, a2])
    };
    match result {
        Ok(obj) => obj,
        Err(error) => {
            crate::runtime_ops::jit_publish_residual_error(error);
            import_lookup_err_ptr()
        }
    }
}

/// True when `addr` is the residual the codewriter emits for
/// [`jit_portal_call_3`]. The hand-listed row is the function itself; the
/// `dont_look_inside` trampoline is the address the macro publishes when
/// that row does not occupy the path.
pub fn residual_addr_is_jit_portal_call_3(addr: i64) -> bool {
    addr == jit_portal_call_3 as *const () as usize as i64
        || addr == __majit_call_target_jit_portal_call_3 as *const () as usize as i64
}

/// `interp___import__` — hand a cached package's fromlist to importlib.
///
/// PyPy's `interp___import__` reaches this as `space.call_method` on
/// `_handle_fromlist`, a residual Python call.  The same residual keeps the
/// package arm out of the look-inside graph so a cached non-package
/// (`from math import pi`) still folds.
///
/// `space.call_method(w_importlib, "_handle_fromlist", w_mod, w_fromlist,
/// space.w_default_importlib_import)`.  `_handle_fromlist` imports whichever
/// names the list adds, so the package case needs no `sys.meta_path` walk of
/// its own.  Answers `None` while the bootstrap is not installed, which leaves
/// the caller on the slow path.
#[majit_macros::dont_look_inside]
pub(crate) fn handle_fromlist_fast(
    w_mod: PyObjectRef,
    w_fromlist: PyObjectRef,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};

    let _roots = push_roots();
    let mod_slot = shadow_stack_len();
    let _ = pin_root(w_mod);
    let fromlist_slot = shadow_stack_len();
    let _ = pin_root(w_fromlist);
    let Some(w_bootstrap) = importlib_bootstrap_module() else {
        return Ok(None);
    };
    let bootstrap_slot = shadow_stack_len();
    let _ = pin_root(w_bootstrap);
    let Some(w_handle) =
        crate::baseobjspace::findattr_result(shadow_stack_get(bootstrap_slot), "_handle_fromlist")?
    else {
        return Ok(None);
    };
    let handle_slot = shadow_stack_len();
    let _ = pin_root(w_handle);
    // `space.w_default_importlib_import` is the importer `_handle_fromlist`
    // calls for a name the package does not already carry.  It is the space's
    // own captured copy, so rebinding `builtins.__import__` or
    // `_bootstrap.__import__` cannot redirect it.
    let Some(w_import) = default_importlib_import() else {
        return Ok(None);
    };
    let import_slot = shadow_stack_len();
    let _ = pin_root(w_import);
    match crate::call::call_function_impl_result(
        shadow_stack_get(handle_slot),
        &[
            shadow_stack_get(mod_slot),
            shadow_stack_get(fromlist_slot),
            shadow_stack_get(import_slot),
        ],
    ) {
        Ok(w_result) => Ok(Some(w_result)),
        Err(err) => Err(err),
    }
}

/// One-word residual-call ABI for [`handle_fromlist_fast`].
///
/// Three outcomes share the result register: `Ok(Some(p))` is `p`,
/// `Ok(None)` is null, and an error is published then answered as
/// [`import_lookup_err_ptr`].
pub extern "C" fn handle_fromlist_fast_jit_abi(
    w_mod: PyObjectRef,
    w_fromlist: PyObjectRef,
) -> PyObjectRef {
    match handle_fromlist_fast(w_mod, w_fromlist) {
        Ok(Some(result)) => result,
        Ok(None) => pyre_object::PY_NULL,
        Err(error) => {
            crate::runtime_ops::jit_publish_residual_error(error);
            import_lookup_err_ptr()
        }
    }
}

/// `builtins.__import__` — `interp___import__`: a fast path answering
/// absolute imports from initialised `sys.modules` entries, the app-level
/// `_bootstrap.__import__` (the full `sys.meta_path` / `sys.path_hooks`
/// protocol) otherwise.  While the importlib bootstrap is not installed —
/// during startup, or when no stdlib is reachable — the native `importhook`
/// stands in, the role of PyPy's minimal `importing.py` importer.
pub fn dunder_import(
    name: &str,
    w_globals: PyObjectRef,
    w_locals: PyObjectRef,
    w_fromlist: PyObjectRef,
    level: i64,
    execution_context: *const PyExecutionContext,
) -> Result<PyObjectRef, crate::PyError> {
    dunder_import_inner(
        name,
        pyre_object::PY_NULL,
        w_globals,
        w_locals,
        w_fromlist,
        level,
        execution_context,
    )
}

/// Look-inside `__import__` after the wrapper has the name object.
///
/// Same body as [`dunder_import`], but the cache probe is keyed by
/// `w_name` so the executed residual is [`sys_modules_finditem_w`].
pub(crate) fn dunder_import_w(
    name: &str,
    w_name: PyObjectRef,
    w_globals: PyObjectRef,
    w_locals: PyObjectRef,
    w_fromlist: PyObjectRef,
    level: i64,
    execution_context: *const PyExecutionContext,
) -> Result<PyObjectRef, crate::PyError> {
    dunder_import_inner(
        name,
        w_name,
        w_globals,
        w_locals,
        w_fromlist,
        level,
        execution_context,
    )
}

fn dunder_import_inner(
    name: &str,
    w_name: PyObjectRef,
    w_globals: PyObjectRef,
    w_locals: PyObjectRef,
    w_fromlist: PyObjectRef,
    level: i64,
    execution_context: *const PyExecutionContext,
) -> Result<PyObjectRef, crate::PyError> {
    // `name` is borrowed from a managed `str` payload (`w_str_get_value_opt`).
    // Pinning `w_name` forwards the header, not this slice. A minor collection
    // reuses the nursery under the slice, and the next `sys.modules` probe
    // hashes bytes that are no longer WTF-8. Copy before any collecting call.
    let name_owned = name.to_owned();
    let name = name_owned.as_str();
    // Captured before any Python can run below (`is_true` may call a
    // `__bool__`).  `gcd_import_fast` can collect; the caller's natives
    // (`w_fromlist` / `w_mod` / globals / locals / name) are not rewritten
    // by that callee.
    let fromlist_missing = w_fromlist.is_null() || unsafe { is_none(w_fromlist) };
    let _import_roots = pyre_object::gc_roots::push_roots();
    let import_base = pyre_object::gc_roots::pin_roots(&[w_name, w_globals, w_locals, w_fromlist]);
    let reload_import = |index: usize| pyre_object::gc_roots::shadow_stack_get(import_base + index);

    if level == 0 {
        // `interp_import.py:66-92` — the fast path is only for absolute
        // imports.  A fromlist is answered from the cache when the module is
        // not a package; a package hands its list to `_handle_fromlist`, which
        // imports whatever the list adds.  No import lock is taken here:
        // `interp_import.py:19` records that CPython's fast path does not
        // take one either.
        let w_mod = if reload_import(0).is_null() {
            gcd_import_fast(name)?
        } else {
            gcd_import_fast_w(name, reload_import(0))?
        };
        if let Some(w_mod) = w_mod {
            let w_mod_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(w_mod);
            let w_mod = || pyre_object::gc_roots::shadow_stack_get(w_mod_slot);
            // `interp_import.py interp___import__` — the list is tested once
            // the cache hit is in hand, and a dotted name's head is resolved
            // only inside the empty-list arm.  Resolving it ahead of the test
            // would run `gcd_import_fast` on the head, whose `__spec__` and
            // `_initializing` reads can run a module's own Python; neither
            // importer runs that for a non-empty list.
            let have_fromlist = !fromlist_missing && is_true_import(reload_import(3))?;
            if !have_fromlist {
                // `name.find(".")` / `name[:dotindex]` residualise as
                // `find` / `__getslice_rangeto`.  Keep that arm off the
                // look-inside graph so a cached `from math import pi`
                // (this branch is not taken) is not declined for helpers
                // only the empty-fromlist dotted name needs.
                return dunder_import_absolute_head(
                    name,
                    w_mod(),
                    reload_import(1),
                    reload_import(2),
                    execution_context,
                );
            }
            // `assert have_fromlist` — the package test and its two return
            // arms.  Exact-module `__path__` is the module-dict hit, not
            // `findattr` (`getattr_str_impl` starts with `force` + `pin_root`
            // and is the `__majit_stringbuilder_new` descent wall).  A
            // non-module with a fromlist is FastPathGiveUp, same cut as
            // `gcd_import_cache_probe`.
            if unsafe { pyre_object::is_module(w_mod()) } {
                let dict = unsafe { pyre_object::w_module_get_w_dict(w_mod()) };
                let w_path = if dict.is_null() {
                    pyre_object::PY_NULL
                } else {
                    module_dict_get_path(dict)
                };
                if import_lookup_is_err(w_path) {
                    return Err(take_published_residual_error());
                }
                if dict.is_null() || w_path.is_null() {
                    return Ok(w_mod());
                }
                // `interp___import__` `space.call_method(w_importlib,
                // "_handle_fromlist", w_mod, w_fromlist,
                // space.w_default_importlib_import)`.  The two lookups do
                // not call Python.  A miss keeps the pin and the native
                // stand-in inside `dunder_import_package_fromlist`, off
                // this graph: after the red `sys.modules` pointer the scan
                // cannot prove `__path__` is absent, and `push_roots` here
                // would decline a cached `from math import pi`.
                let w_handle = bootstrap_handle_fromlist();
                let w_import = default_importlib_import_word();
                if !w_handle.is_null() && !w_import.is_null() {
                    let w_result = jit_portal_call_3(
                        w_handle,
                        pyre_object::PY_NULL,
                        w_mod(),
                        reload_import(3),
                        w_import,
                    );
                    if import_lookup_is_err(w_result) {
                        return Err(take_published_residual_error());
                    }
                    return Ok(w_result);
                }
                return dunder_import_package_fromlist(
                    name,
                    w_mod(),
                    reload_import(1),
                    reload_import(2),
                    reload_import(3),
                    fromlist_missing,
                    level,
                    execution_context,
                );
            }
        }
    }

    // `interp_import.py interp___import__` hands the miss to frozen
    // importlib as one residual call.  Keep that graph behind
    // `dont_look_inside` so a cached `import math` does not drag the
    // bootstrap into the look-inside body.
    dunder_import_slow(
        name,
        reload_import(0),
        reload_import(1),
        reload_import(2),
        reload_import(3),
        fromlist_missing,
        level,
        execution_context,
    )
}

/// `space.getattr(w_spec, "_initializing")` as a tri-state that does not
/// call Python.
///
/// `1` — the instance flag is an exact `True`. `0` — exact `False`, or the
/// attribute is absent on both the map chain and the type MRO (AttributeError
/// is "initialized", `interp_import.py` `_gcd_import`). `-1` —
/// `FastPathGiveUp`: not mapdict, a devolved or unboxed slot, a non-bool, a
/// data descriptor, a type-level non-data binding with no instance value
/// (its `__get__` would run), or a comparison the callback-free probe refused.
/// The slow path then performs the real getattr.
///
/// The key literal stays in this body so `__majit_stringbuilder_new` does not
/// land on the look-inside graph.
#[majit_macros::dont_look_inside_cannot_raise]
pub(crate) fn module_spec_get_initializing(w_spec: PyObjectRef) -> i64 {
    if w_spec.is_null() {
        return -1;
    }
    // `dict_keys_equal` skips `space.eq_w` for the whole probe and sets the
    // broken flag instead.  A broken probe did not run Python; its answer is
    // discarded.
    pyre_object::dict_eq_hook::begin_callback_free_probe();
    let state = unsafe { module_spec_initializing_state(w_spec) };
    if pyre_object::dict_eq_hook::end_callback_free_probe() {
        -1
    } else {
        state
    }
}

enum MroName {
    NoMro,
    Miss,
    Hit(PyObjectRef),
}

unsafe fn mro_unwrapped(w_type: PyObjectRef, name: &Wtf8) -> MroName {
    if w_type.is_null() || !unsafe { pyre_object::is_type(w_type) } {
        return MroName::NoMro;
    }
    let mro = unsafe { pyre_object::w_type_get_mro(w_type) };
    if mro.is_null() {
        return MroName::NoMro;
    }
    for cls in unsafe { (*mro).as_slice() } {
        if cls.is_null() || !unsafe { pyre_object::is_type(*cls) } {
            continue;
        }
        let Some(raw) = crate::runtime_ops::type_dict_lookup_wtf8_no_unwrapping(*cls, name) else {
            continue;
        };
        let value = unsafe { pyre_object::celldict::unwrap_cell(raw) };
        if value.is_null() {
            continue;
        }
        return MroName::Hit(value);
    }
    MroName::Miss
}

unsafe fn spec_has_object_getattribute(w_type: PyObjectRef) -> bool {
    if unsafe { pyre_object::typeobject::w_type_get_uses_object_getattribute(w_type) } {
        return true;
    }
    // `getattribute_if_not_from_object` boxes the name under the JIT and
    // writes the memoized flag off it.  The raw MRO walk compares WTF-8
    // bytes (`type_dict_lookup_wtf8_no_unwrapping`) and leaves the flag alone.
    let name = Wtf8::new("__getattribute__");
    let MroName::Hit(descr) = (unsafe { mro_unwrapped(w_type, name) }) else {
        return false;
    };
    let mut descr = descr;
    let object_hit = pyre_object::with_roots!(descr => unsafe {
        mro_unwrapped(crate::typedef::w_object(), name)
    });
    let MroName::Hit(object_descr) = object_hit else {
        return false;
    };
    std::ptr::eq(descr, object_descr)
}

unsafe fn module_spec_initializing_state(mut w_spec: PyObjectRef) -> i64 {
    let mut w_type = unsafe { (*w_spec).w_class };
    if !pyre_object::with_roots!(w_spec, w_type => unsafe {
        spec_has_object_getattribute(w_type)
    }) {
        return -1;
    }
    let name = Wtf8::new("_initializing");
    let looked = pyre_object::with_roots!(w_spec => unsafe { mro_unwrapped(w_type, name) });
    let type_binds_non_data = match looked {
        MroName::NoMro => return -1,
        MroName::Hit(value) => {
            if pyre_object::with_roots!(w_spec => unsafe {
                crate::baseobjspace::is_data_descr(value)
            }) {
                return -1;
            }
            true
        }
        MroName::Miss => false,
    };
    match unsafe { crate::objspace::std::mapdict::mapdict_boxed_dict_attr(w_spec, name) } {
        Some(flag) if !flag.is_null() && unsafe { pyre_object::is_bool(flag) } => {
            if unsafe { pyre_object::w_bool_get_value(flag) } {
                1
            } else {
                0
            }
        }
        Some(flag) if flag.is_null() && !type_binds_non_data => 0,
        _ => -1,
    }
}

/// Exact-module `__spec__`: celldict `getdictvalue_no_unwrapping` then
/// `unwrap_cell`.  The literal stays inside the cannot-raise residual;
/// unwrap is look-inside so the trace records the cell field read.
fn module_dict_get_spec(dict: PyObjectRef) -> PyObjectRef {
    if unsafe { pyre_object::dictmultiobject::w_module_dict_strategy_or_null(dict) }.is_null() {
        return module_dict_finditem_spec(dict);
    }
    let raw = module_dict_cell_get_spec(dict);
    if raw.is_null() {
        return pyre_object::PY_NULL;
    }
    let v = unsafe { pyre_object::celldict::unwrap_cell(raw) };
    if v.is_null() { pyre_object::PY_NULL } else { v }
}

#[majit_macros::dont_look_inside_cannot_raise]
pub(crate) fn module_dict_cell_get_spec(dict: PyObjectRef) -> PyObjectRef {
    let strategy = unsafe { pyre_object::dictmultiobject::w_module_dict_strategy_or_null(dict) };
    if strategy.is_null() {
        return pyre_object::PY_NULL;
    }
    unsafe { (*strategy).getdictvalue_no_unwrapping(dict, "__spec__") }
        .unwrap_or(pyre_object::PY_NULL)
}

#[majit_macros::dont_look_inside]
pub(crate) fn module_dict_finditem_spec(dict: PyObjectRef) -> PyObjectRef {
    match crate::baseobjspace::finditem_str(dict, "__spec__") {
        Ok(Some(s)) if !s.is_null() => s,
        Ok(_) => pyre_object::PY_NULL,
        Err(e) => {
            crate::runtime_ops::jit_publish_residual_error(e);
            import_lookup_err_ptr()
        }
    }
}

/// Exact-module `__path__`, same split as [`module_dict_get_spec`].
fn module_dict_get_path(dict: PyObjectRef) -> PyObjectRef {
    if unsafe { pyre_object::dictmultiobject::w_module_dict_strategy_or_null(dict) }.is_null() {
        return module_dict_finditem_path(dict);
    }
    let raw = module_dict_cell_get_path(dict);
    if raw.is_null() {
        return pyre_object::PY_NULL;
    }
    let v = unsafe { pyre_object::celldict::unwrap_cell(raw) };
    if v.is_null() { pyre_object::PY_NULL } else { v }
}

#[majit_macros::dont_look_inside_cannot_raise]
pub(crate) fn module_dict_cell_get_path(dict: PyObjectRef) -> PyObjectRef {
    let strategy = unsafe { pyre_object::dictmultiobject::w_module_dict_strategy_or_null(dict) };
    if strategy.is_null() {
        return pyre_object::PY_NULL;
    }
    unsafe { (*strategy).getdictvalue_no_unwrapping(dict, "__path__") }
        .unwrap_or(pyre_object::PY_NULL)
}

#[majit_macros::dont_look_inside]
pub(crate) fn module_dict_finditem_path(dict: PyObjectRef) -> PyObjectRef {
    match crate::baseobjspace::finditem_str(dict, "__path__") {
        Ok(Some(s)) if !s.is_null() => s,
        Ok(_) => pyre_object::PY_NULL,
        Err(e) => {
            crate::runtime_ops::jit_publish_residual_error(e);
            import_lookup_err_ptr()
        }
    }
}

/// Look-inside `space.is_true` for `__import__`: exact tuple / bool /
/// None are answered here so a constant fromlist and an exact-bool
/// `_initializing` never call the hidden residual.  The generic arm
/// is [`is_true_after_modules`].
fn is_true_import(obj: PyObjectRef) -> Result<bool, crate::PyError> {
    unsafe {
        if pyre_object::is_tuple(obj) {
            return Ok(pyre_object::w_tuple_len(obj) != 0);
        }
        if pyre_object::is_bool(obj) {
            return Ok(std::ptr::eq(obj, pyre_object::w_bool_from(true)));
        }
        if pyre_object::is_none(obj) {
            return Ok(false);
        }
    }
    let flag = is_true_after_modules(obj);
    if flag < 0 {
        return Err(take_published_residual_error());
    }
    Ok(flag != 0)
}

/// Generic `space.is_true` after the red `sys.modules` read.  Hidden so
/// a red receiver does not drag `is_true_slot`'s layout predicates
/// into the look-inside body.  `1` / `0` / `-1` (raise published).
#[majit_macros::dont_look_inside]
pub(crate) fn is_true_after_modules(obj: PyObjectRef) -> i64 {
    match crate::baseobjspace::is_true(obj) {
        Ok(true) => 1,
        Ok(false) => 0,
        Err(e) => {
            crate::runtime_ops::jit_publish_residual_error(e);
            -1
        }
    }
}

/// Recover the exception a 1-word residual published through
/// [`crate::runtime_ops::jit_publish_residual_error`].  Does not
/// re-run the helper.  Hidden; the cached-import walk does not take
/// this arm.
#[majit_macros::dont_look_inside]
pub(crate) fn take_published_residual_error() -> crate::PyError {
    let obj = majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|cell| {
        let v = cell.get();
        cell.set(0);
        v
    });
    if obj == 0 {
        crate::PyError::type_error("residual published no exception")
    } else {
        unsafe { crate::PyError::from_exc_object(obj as PyObjectRef) }
    }
}

/// One-word residual-call ABI for [`take_published_residual_error`].
///
/// The helper consumes an already-published exception and rebuilds a
/// `PyError` for `return Err(...)`. A residual cannot hand that struct
/// back, so the bridge re-publishes it and returns 0; `GUARD_NO_EXCEPTION`
/// takes the error arm.
pub extern "C" fn take_published_residual_error_jit_abi() -> i64 {
    crate::runtime_ops::jit_publish_residual_error(take_published_residual_error())
}

/// `interp___import__` empty-fromlist arm: answer `name.partition('.')[0]`.
///
/// The `find` / slice helpers are un-lowered, so the whole arm — including
/// the undotted `import math` return — stays off the look-inside graph.
#[majit_macros::dont_look_inside]
pub(crate) fn dunder_import_absolute_head(
    name: &str,
    w_mod: PyObjectRef,
    mut w_globals: PyObjectRef,
    mut w_locals: PyObjectRef,
    execution_context: *const PyExecutionContext,
) -> Result<PyObjectRef, crate::PyError> {
    // `import a.b` answers `a`.
    let dotindex = rpython_str_find_char(name, '.', 0);
    if dotindex < 0 {
        return Ok(w_mod);
    }
    let head = rpython_str_slice_prefix(name, dotindex);
    if let Some(w_head) = pyre_object::with_roots!(w_globals, w_locals => gcd_import_fast(head))? {
        return Ok(w_head);
    }
    // An uncached head is what `_bootstrap.__import__`'s own
    // `if not fromlist:` arm imports — `return _gcd_import(
    // name.partition('.')[0])`.  Re-entering under the head name
    // with no list reaches that same import while leaving the
    // caller's list alone: the one truth test `__import__` owes a
    // stateful `__bool__` has already been made here.
    dunder_import(
        head,
        w_globals,
        w_locals,
        pyre_object::w_none(),
        0,
        execution_context,
    )
}

/// Cached-package fromlist arm of `interp___import__`: pin, then
/// `_handle_fromlist` (bootstrap) or the native stand-in.  Hidden so
/// `push_roots` / `pin_root` stay off the look-inside graph.
///
/// A broken installed bootstrap falls through to [`dunder_import_slow`]
/// with the same arguments the look-inside caller used to pass.
#[majit_macros::dont_look_inside]
pub(crate) fn dunder_import_package_fromlist(
    name: &str,
    w_mod: PyObjectRef,
    w_globals: PyObjectRef,
    w_locals: PyObjectRef,
    w_fromlist: PyObjectRef,
    fromlist_missing: bool,
    level: i64,
    execution_context: *const PyExecutionContext,
) -> Result<PyObjectRef, crate::PyError> {
    use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};

    let fromlist_is_null = w_fromlist.is_null();
    let _roots = push_roots();
    let mod_slot = shadow_stack_len();
    let _ = pin_root(w_mod);
    let fromlist_slot = shadow_stack_len();
    let _ = pin_root(if fromlist_is_null {
        pyre_object::w_none()
    } else {
        w_fromlist
    });
    let globals_slot = shadow_stack_len();
    let _ = pin_root(w_globals);
    let locals_slot = shadow_stack_len();
    let _ = pin_root(w_locals);
    if let Some(w_handled) =
        handle_fromlist_fast(shadow_stack_get(mod_slot), shadow_stack_get(fromlist_slot))?
    {
        return Ok(w_handled);
    } else if importlib_bootstrap_module().is_none() {
        // `interp_import.py interp___import__` returns from both
        // arms of the package test; the `_handle_fromlist` it
        // calls is always installed upstream.  While the
        // bootstrap is not installed at all, the native one
        // stands in here.  Falling through to the slow path
        // instead would resolve the same cached module a second
        // time and test the list a second time, and `__import__`
        // owes a stateful `__bool__` exactly one test.
        handle_fromlist(
            shadow_stack_get(mod_slot),
            shadow_stack_get(fromlist_slot),
            false,
            execution_context,
        )?;
        return Ok(shadow_stack_get(mod_slot));
    }
    // An installed bootstrap that cannot serve `_handle_fromlist`
    // is a broken bootstrap, not an absent one.
    // `_bootstrap.__import__` reaches the handler as a module
    // global and raises when it is gone, so fall through to the
    // slow path and let that error be the one `__import__`
    // reports rather than answering from the native handler.
    dunder_import_slow(
        name,
        pyre_object::PY_NULL,
        shadow_stack_get(globals_slot),
        shadow_stack_get(locals_slot),
        if fromlist_is_null {
            pyre_object::PY_NULL
        } else {
            shadow_stack_get(fromlist_slot)
        },
        fromlist_missing,
        level,
        execution_context,
    )
}

/// Miss / relative / bootstrap half of `interp___import__`.
///
/// PyPy's `space.call_function(FrozenCache.w_frozen_import, ...)` is one
/// residual.  The generated `__import__` wrapper must stay small enough for
/// `try_walker_inline_builtin_call` to descend: a look-inside of this body
/// is 300+ un-lowered helpers and the descent scan refuses the whole builtin.
#[majit_macros::dont_look_inside]
pub(crate) fn dunder_import_slow(
    name: &str,
    w_name: PyObjectRef,
    w_globals: PyObjectRef,
    w_locals: PyObjectRef,
    w_fromlist: PyObjectRef,
    fromlist_missing: bool,
    level: i64,
    execution_context: *const PyExecutionContext,
) -> Result<PyObjectRef, crate::PyError> {
    use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};

    let globals_is_null = w_globals.is_null();
    let fromlist_is_null = w_fromlist.is_null();
    let _roots = push_roots();
    let name_slot = shadow_stack_len();
    let _ = pin_root(w_name);
    let globals_slot = shadow_stack_len();
    let _ = pin_root(if globals_is_null {
        pyre_object::w_none()
    } else {
        w_globals
    });
    let locals_slot = shadow_stack_len();
    let _ = pin_root(if w_locals.is_null() {
        pyre_object::w_none()
    } else {
        w_locals
    });
    let fromlist_slot = shadow_stack_len();
    let _ = pin_root(if fromlist_is_null {
        pyre_object::w_none()
    } else {
        w_fromlist
    });

    // The frozen bootstrap modules stay on the native importer:
    // `_install_external_importers` imports `_frozen_importlib_external`
    // while installing PathFinder, so no finder can serve it yet —
    // `absolute_import` execs the on-disk bootstrap source under that name.
    if matches!(name, "_frozen_importlib" | "_frozen_importlib_external") {
        return importhook(
            Wtf8::new(name),
            if globals_is_null {
                pyre_object::PY_NULL
            } else {
                shadow_stack_get(globals_slot)
            },
            if fromlist_is_null {
                pyre_object::PY_NULL
            } else {
                shadow_stack_get(fromlist_slot)
            },
            level,
            execution_context,
        );
    }

    // Slow path: the app-level `_bootstrap.__import__`.
    if let Some(w_bootstrap) = importlib_bootstrap_module() {
        let bootstrap_slot = shadow_stack_len();
        let _ = pin_root(w_bootstrap);
        if let Some(w_import) =
            crate::baseobjspace::findattr_result(shadow_stack_get(bootstrap_slot), "__import__")?
        {
            let import_slot = shadow_stack_len();
            let _ = pin_root(w_import);
            // `PyImport_ImportModuleLevelObject`: a relative import with an
            // empty fromlist hands back the head package named by the *resolved
            // absolute name*, and the imported module itself when `name` carries
            // no dot.  `_bootstrap.__import__` reaches the same slice through
            // `module.__name__`, so a `sys.modules` entry that is not a module
            // raises AttributeError there instead of the KeyError the
            // interpreter entry point reports.
            if level > 0
                && !name.is_empty()
                && !(!fromlist_missing
                    && crate::baseobjspace::is_true(shadow_stack_get(fromlist_slot))?)
            {
                return relative_import_head(
                    shadow_stack_get(bootstrap_slot),
                    name,
                    shadow_stack_get(globals_slot),
                    level,
                );
            }
            // `interp_import.py` `interp___import__` passes the caller's
            // `w_modulename` into the frozen import. Allocate only when
            // this entry has no string object (a `&str` head name).
            let pinned_name = shadow_stack_get(name_slot);
            let w_name = if !pinned_name.is_null() && unsafe { pyre_object::is_str(pinned_name) } {
                pinned_name
            } else {
                pyre_object::w_str_new_managed(name)
            };
            return call_bootstrap_import(
                shadow_stack_get(import_slot),
                w_name,
                shadow_stack_get(globals_slot),
                shadow_stack_get(locals_slot),
                if fromlist_missing {
                    pyre_object::PY_NULL
                } else {
                    shadow_stack_get(fromlist_slot)
                },
                level,
            );
        }
    }
    importhook(
        Wtf8::new(name),
        if globals_is_null {
            pyre_object::PY_NULL
        } else {
            shadow_stack_get(globals_slot)
        },
        if fromlist_is_null {
            pyre_object::PY_NULL
        } else {
            shadow_stack_get(fromlist_slot)
        },
        level,
        execution_context,
    )
}

/// The `builtins.__import__` binding, or the error a missing one raises.
///
/// `space.getattr(space.builtin, '__import__')`, the way `_lookup_codec_loop`
/// reads it: off the space-wide builtins module, at the call, so whatever the
/// name holds by then is what runs.  [`lookup_dunder_import`] answers the same
/// question for IMPORT_NAME against the frame's own builtins, and raises the
/// same error, because a name that is not there is a missing importer and not
/// a licence to run the one the binding would have replaced.
fn dunder_import_binding(
    execution_context: *const PyExecutionContext,
) -> Result<PyObjectRef, crate::PyError> {
    use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};

    // `space.builtin` -- the one Module every frame uses for its LOAD_GLOBAL
    // fallback, and the object `import builtins` publishes.
    let w_builtin = if execution_context.is_null() {
        pyre_object::PY_NULL
    } else {
        unsafe { (*execution_context).get_builtin() }
    };
    let w_import = if !w_builtin.is_null() && unsafe { is_module(w_builtin) } {
        // Pinned before the read: it is a `getattr` on a module, which can run
        // a module-level `__getattr__` and collect underneath the receiver.
        let _roots = push_roots();
        let builtin_slot = shadow_stack_len();
        let _ = pin_root(w_builtin);
        crate::baseobjspace::findattr_result(shadow_stack_get(builtin_slot), "__import__")?
    } else {
        None
    };
    w_import
        .ok_or_else(|| crate::PyError::new(crate::PyErrorKind::ImportError, "__import__ not found"))
}

/// The binding called with the name alone, which is how `_lookup_codec_loop`
/// spells its two imports: `space.call_function(w_import, w_name)`.  A
/// replacement is user code with a signature of its own, so a call carrying
/// arguments the site upstream never passes is a call it can reject.
pub fn call_dunder_import_name_only(
    name: &str,
    execution_context: *const PyExecutionContext,
) -> Result<PyObjectRef, crate::PyError> {
    use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};

    let _roots = push_roots();
    let import_slot = shadow_stack_len();
    let _ = pin_root(dunder_import_binding(execution_context)?);
    // Minted before the slot is read back: the allocation can move what the
    // slot holds, and an argument read ahead of it would be a stale address.
    let name_slot = shadow_stack_len();
    let _ = pin_root(pyre_object::w_str_new_managed(name));
    crate::call::call_function_impl_result(
        shadow_stack_get(import_slot),
        &[shadow_stack_get(name_slot)],
    )
    .map_err(strip_bootstrap_traceback_frames)
}

/// `builtins.__import__` as the binding rather than as the function it starts
/// out holding, called the way an `import` statement calls it: with the name,
/// the two namespaces, the fromlist and the level, which is what IMPORT_NAME
/// passes.  A program that replaces `builtins.__import__` observes the imports
/// the interpreter makes on its behalf through here.
pub fn call_dunder_import(
    name: &str,
    w_globals: PyObjectRef,
    w_locals: PyObjectRef,
    w_fromlist: PyObjectRef,
    level: i64,
    execution_context: *const PyExecutionContext,
) -> Result<PyObjectRef, crate::PyError> {
    use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};

    let fromlist_is_null = w_fromlist.is_null();
    let _roots = push_roots();
    // Pinned before the binding is read: reading it is a `getattr` on a module,
    // which can run a module-level `__getattr__`, and the caller's raw
    // references do not survive a collection there.
    let globals_slot = shadow_stack_len();
    let _ = pin_root(if w_globals.is_null() {
        pyre_object::w_none()
    } else {
        w_globals
    });
    let locals_slot = shadow_stack_len();
    let _ = pin_root(if w_locals.is_null() {
        pyre_object::w_none()
    } else {
        w_locals
    });
    let fromlist_slot = shadow_stack_len();
    let _ = pin_root(if fromlist_is_null {
        pyre_object::w_none()
    } else {
        w_fromlist
    });

    let import_slot = shadow_stack_len();
    let _ = pin_root(dunder_import_binding(execution_context)?);
    // Minted before any slot is read back: the allocation can move what the
    // slots hold, and an argument read ahead of it would be a stale address.
    let w_name = pyre_object::w_str_new_managed(name);
    call_bootstrap_import(
        shadow_stack_get(import_slot),
        w_name,
        shadow_stack_get(globals_slot),
        shadow_stack_get(locals_slot),
        if fromlist_is_null {
            pyre_object::PY_NULL
        } else {
            shadow_stack_get(fromlist_slot)
        },
        level,
    )
}

/// Invoke an app-level `__import__` -- `_bootstrap`'s, or whatever the
/// `builtins.__import__` binding names.  A null `w_fromlist` means the
/// argument was omitted, which reaches `__import__` as its `()` default rather
/// than as `None` (interp_import.py `WrappedDefault(())`).
fn call_bootstrap_import(
    w_import: PyObjectRef,
    w_name: PyObjectRef,
    w_globals: PyObjectRef,
    w_locals: PyObjectRef,
    w_fromlist: PyObjectRef,
    level: i64,
) -> Result<PyObjectRef, crate::PyError> {
    use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};

    let _roots = push_roots();
    let import_slot = shadow_stack_len();
    let _ = pin_root(w_import);
    let name_slot = shadow_stack_len();
    let _ = pin_root(w_name);
    let globals_slot = shadow_stack_len();
    let _ = pin_root(w_globals);
    let locals_slot = shadow_stack_len();
    let _ = pin_root(w_locals);
    let fromlist_slot = shadow_stack_len();
    let _ = pin_root(if w_fromlist.is_null() {
        pyre_object::w_tuple_new(vec![])
    } else {
        w_fromlist
    });
    let level_slot = shadow_stack_len();
    let _ = pin_root(pyre_object::w_int_new(level));
    crate::call::call_function_impl_result(
        shadow_stack_get(import_slot),
        &[
            shadow_stack_get(name_slot),
            shadow_stack_get(globals_slot),
            shadow_stack_get(locals_slot),
            shadow_stack_get(fromlist_slot),
            shadow_stack_get(level_slot),
        ],
    )
    .map_err(strip_bootstrap_traceback_frames)
}

/// `__import__` for a name with no `&str` spelling.
///
/// Every native lookup — `sys.modules`, the builtin registry, the frozen table
/// and the path search — is `&str`-keyed and can hold no such name, so the
/// app-level `_bootstrap.__import__` runs it with the name object as given,
/// which keeps the dotted-name, `level` and `fromlist` handling intact.
/// Residual: a red `w_str_get_value_opt` must not drag this bootstrap
/// graph into the look-inside `__import__` body.
#[majit_macros::dont_look_inside]
pub fn dunder_import_name_obj(
    w_name: PyObjectRef,
    w_globals: PyObjectRef,
    w_locals: PyObjectRef,
    w_fromlist: PyObjectRef,
    level: i64,
) -> Result<PyObjectRef, crate::PyError> {
    use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};

    let _roots = push_roots();
    let name_slot = shadow_stack_len();
    let _ = pin_root(w_name);
    let globals_slot = shadow_stack_len();
    let _ = pin_root(if w_globals.is_null() {
        pyre_object::w_none()
    } else {
        w_globals
    });
    let locals_slot = shadow_stack_len();
    let _ = pin_root(if w_locals.is_null() {
        pyre_object::w_none()
    } else {
        w_locals
    });
    let fromlist_slot = shadow_stack_len();
    let w_fromlist = if w_fromlist.is_null() {
        pyre_object::w_none()
    } else {
        w_fromlist
    };
    let _ = pin_root(w_fromlist);

    let bootstrap = if let Some(w_bootstrap) = importlib_bootstrap_module() {
        let bootstrap_slot = shadow_stack_len();
        let _ = pin_root(w_bootstrap);
        crate::baseobjspace::findattr_result(shadow_stack_get(bootstrap_slot), "__import__")?
    } else {
        None
    };
    let Some(w_import) = bootstrap else {
        // No app-level importer yet. The native `importhook` carries the
        // name as WTF-8, including a lone surrogate.
        return importhook(
            unsafe { pyre_object::w_str_get_wtf8(shadow_stack_get(name_slot)) },
            shadow_stack_get(globals_slot),
            shadow_stack_get(fromlist_slot),
            level,
            crate::call::getexecutioncontext(),
        );
    };
    call_bootstrap_import(
        w_import,
        shadow_stack_get(name_slot),
        shadow_stack_get(globals_slot),
        shadow_stack_get(locals_slot),
        shadow_stack_get(fromlist_slot),
        level,
    )
}

/// One-word residual-call ABI for [`dunder_import_name_obj`].
pub extern "C" fn dunder_import_name_obj_jit_abi(
    w_name: PyObjectRef,
    w_globals: PyObjectRef,
    w_locals: PyObjectRef,
    w_fromlist: PyObjectRef,
    level: i64,
) -> PyObjectRef {
    match dunder_import_name_obj(w_name, w_globals, w_locals, w_fromlist, level) {
        Ok(result) => result,
        Err(error) => crate::runtime_ops::jit_publish_residual_error_ref(error),
    }
}

/// `PyImport_ImportModuleLevelObject`, the `level > 0` / empty-fromlist tail:
/// import `name` relative to `globals`, then return the head package named by
/// the absolute name the resolution produced.
///
/// `_bootstrap.__import__` spells the same slice as
/// `sys.modules[module.__name__[:len(module.__name__) - cut_off]]`, which
/// requires the loaded object to carry `__name__`.  A `sys.modules` entry that
/// is not a module has none, so the interpreter entry point slices the name it
/// resolved itself and reports a missing head as a KeyError.
fn relative_import_head(
    w_bootstrap: PyObjectRef,
    name: &str,
    w_globals: PyObjectRef,
    level: i64,
) -> Result<PyObjectRef, crate::PyError> {
    use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};

    let _roots = push_roots();
    let bootstrap_slot = shadow_stack_len();
    let _ = pin_root(w_bootstrap);
    // `_bootstrap.__import__` — `globals_ = globals if globals is not None else {}`.
    let globals_slot = shadow_stack_len();
    let _ = pin_root(if w_globals.is_null() || unsafe { is_none(w_globals) } {
        pyre_object::w_dict_new()
    } else {
        w_globals
    });
    let name_slot = shadow_stack_len();
    let _ = pin_root(pyre_object::w_str_new_managed(name));
    let level_slot = shadow_stack_len();
    let _ = pin_root(pyre_object::w_int_new(level));

    // `_bootstrap.__import__` — `package = _calc___package__(globals_)`.
    let w_calc =
        crate::baseobjspace::getattr_str(shadow_stack_get(bootstrap_slot), "_calc___package__")?;
    let calc_slot = shadow_stack_len();
    let _ = pin_root(w_calc);
    let w_package = crate::call::call_function_impl_result(
        shadow_stack_get(calc_slot),
        &[shadow_stack_get(globals_slot)],
    )
    .map_err(strip_bootstrap_traceback_frames)?;
    let package_slot = shadow_stack_len();
    let _ = pin_root(w_package);

    // `_bootstrap.__import__` — `module = _gcd_import(name, package, level)`.
    // `_sanity_check` / `_resolve_name` validate `package` and `level` in there,
    // so running it before the slice below keeps their diagnostics first.
    let w_gcd = crate::baseobjspace::getattr_str(shadow_stack_get(bootstrap_slot), "_gcd_import")?;
    let gcd_slot = shadow_stack_len();
    let _ = pin_root(w_gcd);
    let w_module = crate::call::call_function_impl_result(
        shadow_stack_get(gcd_slot),
        &[
            shadow_stack_get(name_slot),
            shadow_stack_get(package_slot),
            shadow_stack_get(level_slot),
        ],
    )
    .map_err(strip_bootstrap_traceback_frames)?;
    let module_slot = shadow_stack_len();
    let _ = pin_root(w_module);

    // No dot in `name`: the imported module is already the head.
    let Some(dot) = name.find('.') else {
        return Ok(shadow_stack_get(module_slot));
    };

    // `_resolve_name(name, package, level)` is the absolute name just imported;
    // it ends with `name`, so trimming from `name`'s first dot leaves the head.
    let w_resolve =
        crate::baseobjspace::getattr_str(shadow_stack_get(bootstrap_slot), "_resolve_name")?;
    let resolve_slot = shadow_stack_len();
    let _ = pin_root(w_resolve);
    let w_abs_name = crate::call::call_function_impl_result(
        shadow_stack_get(resolve_slot),
        &[
            shadow_stack_get(name_slot),
            shadow_stack_get(package_slot),
            shadow_stack_get(level_slot),
        ],
    )
    .map_err(strip_bootstrap_traceback_frames)?;
    let abs_slot = shadow_stack_len();
    let _ = pin_root(w_abs_name);
    // Owned: the slicing below outlives the allocations that follow it.
    let abs_name = crate::baseobjspace::utf8_w(shadow_stack_get(abs_slot))?.to_string();
    let cut_off = name.len() - dot;
    let head = abs_name
        .len()
        .checked_sub(cut_off)
        .and_then(|end| abs_name.get(..end))
        .unwrap_or(abs_name.as_str());

    let head_slot = shadow_stack_len();
    let _ = pin_root(pyre_object::w_str_new_managed(head));
    // The raw `sys.modules` entry, `None` sentinel included: only a *missing*
    // key is the KeyError.
    let w_modules = sys_modules_dict();
    let found = if w_modules.is_null() {
        check_sys_modules(head)
    } else {
        unsafe {
            pyre_object::dictmultiobject::w_dict_lookup(w_modules, shadow_stack_get(head_slot))
        }
        .filter(|w| !w.is_null())
    };
    if let Some(w_found) = found {
        return Ok(w_found);
    }
    let head_repr = unsafe { crate::display::py_repr_wtf8(shadow_stack_get(head_slot)) }?;
    Err(crate::PyError::key_error(crate::display::wtf8_format!(
        head_repr,
        " not in sys.modules as expected"
    )))
}

// ── importhook ───────────────────────────────────────────────────────
// PyPy equivalent: importing.py `importhook()`

pub fn importhook(
    name: &Wtf8,
    w_globals: PyObjectRef,
    w_fromlist: PyObjectRef,
    level: i64,
    execution_context: *const PyExecutionContext,
) -> Result<PyObjectRef, crate::PyError> {
    use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};

    // CPython 3.14 import.c / PyPy's import-level validation: negative
    // levels are rejected independently of the module name.  An empty name
    // is valid only for an actual relative import (`from . import x`).
    if level < 0 {
        return Err(crate::PyError::new(
            crate::PyErrorKind::ValueError,
            "level must be >= 0",
        ));
    }
    if name.is_empty() && level == 0 {
        return Err(crate::PyError::new(
            crate::PyErrorKind::ValueError,
            "Empty module name",
        ));
    }

    // The import below runs module bodies, so the caller's raw `w_fromlist`
    // is stale by the time `handle_fromlist` reads it back.
    let fromlist_missing = w_fromlist.is_null() || unsafe { is_none(w_fromlist) };
    let _roots = push_roots();
    let fromlist_slot = shadow_stack_len();
    let _ = pin_root(if w_fromlist.is_null() {
        pyre_object::w_none()
    } else {
        w_fromlist
    });

    let w_mod = if level > 0 {
        relative_import(name, w_globals, w_fromlist, level, execution_context)?
    } else {
        absolute_import(name, w_fromlist, execution_context)?
    };
    let mod_slot = shadow_stack_len();
    let _ = pin_root(w_mod);

    // `_bootstrap.py __import__` — `if not fromlist:` is a truth test, taken
    // after the import, and it answers the HEAD package rather than the leaf
    // `absolute_import` returns for any non-null list.  `[]` and `()` are
    // falsy and belong on this arm; only an omitted list skips the test.
    if fromlist_missing || !crate::baseobjspace::is_true(shadow_stack_get(fromlist_slot))? {
        return import_head(name, shadow_stack_get(mod_slot), level, execution_context);
    }

    // `elif hasattr(module, '__path__'): return _handle_fromlist(module,
    // fromlist, _gcd_import)` — a module that is not a package answers the
    // list from its own attributes and needs no further import.
    if crate::baseobjspace::findattr_result(shadow_stack_get(mod_slot), "__path__")?.is_some() {
        handle_fromlist(
            shadow_stack_get(mod_slot),
            shadow_stack_get(fromlist_slot),
            false,
            execution_context,
        )?;
    }
    Ok(shadow_stack_get(mod_slot))
}

// ── __import__'s empty-fromlist tail ─────────────────────────────────
// `_bootstrap.py __import__`

/// Answer the head package for an import whose fromlist is empty:
///
/// ```text
/// if level == 0:
///     return _gcd_import(name.partition('.')[0])
/// elif not name:
///     return module
/// else:
///     cut_off = len(name) - len(name.partition('.')[0])
///     return sys.modules[module.__name__[:len(module.__name__) - cut_off]]
/// ```
///
/// `dunder_import` answers this from the cache for the app-level importer; the
/// native one owes the same answer, and without it `import a.b` and
/// `__import__('a.b', ..., [], 0)` hand back the leaf.
///
/// The relative arm slices in bytes rather than code points: a relative import
/// resolves to `<base>.<name>`, so the trailing `cut_off` bytes of the resolved
/// name are the trailing bytes of `name` itself and the cut lands in the same
/// place either way.
fn import_head(
    name: &Wtf8,
    w_mod: PyObjectRef,
    level: i64,
    execution_context: *const PyExecutionContext,
) -> Result<PyObjectRef, crate::PyError> {
    use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};

    let dotindex = match name.as_bytes().iter().position(|&byte| byte == b'.') {
        Some(index) => index,
        None => {
            // `name.partition('.')[0]` is the whole name, so the head is the
            // module this call already resolved -- and so is `not name`.
            return Ok(w_mod);
        }
    };
    if level == 0 {
        // `_gcd_import(name.partition('.')[0])`, answered from `sys.modules`,
        // which the import that preceded this has already populated.
        let head = unsafe { Wtf8::from_bytes_unchecked(&name.as_bytes()[..dotindex]) };
        return absolute_import(head, pyre_object::PY_NULL, execution_context);
    }

    let _roots = push_roots();
    let mod_slot = shadow_stack_len();
    let _ = pin_root(w_mod);
    let cut_off = name.len() - dotindex;
    let w_name = crate::baseobjspace::getattr_str(shadow_stack_get(mod_slot), "__name__")?;
    let full = unsafe { pyre_object::w_str_get_wtf8(w_name) };
    if cut_off >= full.len() {
        return Ok(shadow_stack_get(mod_slot));
    }
    let head = unsafe { Wtf8::from_bytes_unchecked(&full.as_bytes()[..full.len() - cut_off]) };
    modules_cached(head).ok_or_else(|| {
        let mut message = Wtf8Buf::new();
        message.push_str("'");
        message.push_wtf8(head);
        message.push_str("' not in sys.modules as expected");
        crate::PyError::key_error(message)
    })
}

// ── _handle_fromlist ─────────────────────────────────────────────────
// `_bootstrap.py _handle_fromlist()`

/// Import the submodules a `from <package> import ...` names.
///
/// Every name the package does not already carry is imported as
/// `<__name__>.<name>`, and that import binds it as an attribute of the
/// package on the way out -- which is what makes `from package import
/// submodule` resolve to a submodule no attribute holds yet.  `IMPORT_FROM`
/// runs after this and only reads what it left behind.
///
/// The app-level bootstrap owns this wherever `_frozen_importlib` is
/// installed, reached through `handle_fromlist_fast` or
/// `call_bootstrap_import`.  The native importer stands in while it is not --
/// during startup, and for a guest carrying no importlib at all -- so it owes
/// the same work; `import_` is fixed to `absolute_import` there rather than
/// being passed in.
///
/// Returns unit rather than the module: the caller holds the pinned leaf, and
/// the imports below can move whatever this were to hand back.
///
/// Hidden: the native stand-in iterates, formats TypeErrors, and imports
/// children.  After the red `sys.modules` read the descent scan cannot
/// prove the bootstrap is installed, so inlining this body is the
/// `__majit_stringbuilder_new` / warning / compare wall on `__import__`.
#[majit_macros::dont_look_inside]
pub(crate) fn handle_fromlist(
    w_mod: PyObjectRef,
    w_fromlist: PyObjectRef,
    recursive: bool,
    execution_context: *const PyExecutionContext,
) -> Result<(), crate::PyError> {
    use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};

    let _roots = push_roots();
    let mod_slot = shadow_stack_len();
    let _ = pin_root(w_mod);
    let fromlist_slot = shadow_stack_len();
    let _ = pin_root(w_fromlist);

    // `for x in fromlist` — iterated lazily, one item fully processed before
    // the next is requested.  A custom iterable is observable here: draining it
    // up front lets a later `__next__` error preempt the TypeError an earlier
    // item already owes, and hides an import an earlier item performed from the
    // `__next__` that follows it.
    let w_iter = crate::baseobjspace::iter(shadow_stack_get(fromlist_slot))?;
    let iter_slot = shadow_stack_len();
    let _ = pin_root(w_iter);

    loop {
        let w_x = match crate::baseobjspace::next(shadow_stack_get(iter_slot)) {
            Ok(item) => item,
            Err(err) if err.kind == crate::PyErrorKind::StopIteration => break,
            Err(err) => return Err(err),
        };
        let _item_roots = push_roots();
        let x_slot = shadow_stack_len();
        let _ = pin_root(w_x);

        if !unsafe { pyre_object::is_str(shadow_stack_get(x_slot)) } {
            // The `__name__` read runs first: it can collect, and both it and
            // `type_name_of`'s view into the type object move under it.
            let mut message: Wtf8Buf = "Item in ".into();
            if recursive {
                let w_name =
                    crate::baseobjspace::getattr_str(shadow_stack_get(mod_slot), "__name__")?;
                message.push_wtf8(&unsafe { crate::display::py_str_wtf8(w_name) }?);
                message.push_str(".__all__");
            } else {
                message.push_str("``from list''");
            }
            message.push_str(" must be str, not ");
            message.push_str(&crate::error::type_name_of(shadow_stack_get(x_slot)));
            return Err(crate::PyError::type_error(message));
        }

        // `elif x == '*'` — a comparison the item answers, so a `str`
        // subclass whose `__eq__` denies `'*'` is an ordinary name however
        // its payload reads.  The literal is pinned before the call: `eq_w`
        // runs that `__eq__` and both operands move under it.
        //
        // `__all__` is expanded once; a name inside it that is itself `*` is an
        // ordinary name, not a second expansion.
        let star_slot = shadow_stack_len();
        let _ = pin_root(pyre_object::unicodeobject::intern_str_value("*"));
        if crate::baseobjspace::eq_w(shadow_stack_get(x_slot), shadow_stack_get(star_slot))? {
            if !recursive
                && let Some(w_all) =
                    crate::baseobjspace::findattr_result(shadow_stack_get(mod_slot), "__all__")?
            {
                handle_fromlist(shadow_stack_get(mod_slot), w_all, true, execution_context)?;
            }
            continue;
        }

        // `hasattr(module, x)` — keyed by the item object rather than a `&str`
        // spelling of it, so a name carrying a lone surrogate is looked up as
        // written instead of reaching an accessor that rejects it.
        match crate::baseobjspace::getattr(shadow_stack_get(mod_slot), shadow_stack_get(x_slot)) {
            Ok(_) => continue,
            Err(err) if err.kind == crate::PyErrorKind::AttributeError => {}
            Err(err) => return Err(err),
        }

        // `from_name = f'{module.__name__}.{x}'` — both operands are
        // formatted with an empty spec rather than read out of their storage,
        // so a `__format__` override on either runs and a `ValueError` it
        // raises propagates.  That also coerces a `__name__` that is not a
        // str: `sys.modules` can hold any object with a `__path__`, and the
        // f-string reaches it through `object.__format__` rather than a
        // str-only accessor.
        let w_name = crate::baseobjspace::getattr_str(shadow_stack_get(mod_slot), "__name__")?;
        let name_slot = shadow_stack_len();
        let _ = pin_root(w_name);
        let mut from_name =
            crate::type_methods::format_value_dispatch(shadow_stack_get(name_slot), Wtf8::new(""))?;
        from_name.push_str(".");
        from_name.push_wtf8(&crate::type_methods::format_value_dispatch(
            shadow_stack_get(x_slot),
            Wtf8::new(""),
        )?);
        if let Err(err) = absolute_import(&from_name, pyre_object::PY_NULL, execution_context) {
            // Backwards-compatibility dictates we ignore failed imports
            // triggered by fromlist for modules that don't exist:
            // `exc.name == from_name and sys.modules.get(from_name,
            // _NEEDS_LOADING) is not None`.  The absent key answers the
            // sentinel, so the only entry that re-raises is an explicit
            // `None` block.
            if err.kind != crate::PyErrorKind::ModuleNotFoundError
                || !error_names_module(&err, &from_name)
                || modules_block(&from_name)
            {
                return Err(err);
            }
        }
    }
    Ok(())
}

/// One-word residual-call ABI for [`handle_fromlist`].
///
/// `Ok(())` is 1; an error is published and answered as 0.
pub extern "C" fn handle_fromlist_jit_abi(
    w_mod: PyObjectRef,
    w_fromlist: PyObjectRef,
    recursive: i64,
    execution_context: *const PyExecutionContext,
) -> i64 {
    match handle_fromlist(w_mod, w_fromlist, recursive != 0, execution_context) {
        Ok(()) => 1,
        Err(error) => crate::runtime_ops::jit_publish_residual_error(error),
    }
}

/// Whether a ModuleNotFoundError's `name` is exactly `module_name` -- the
/// `exc.name == from_name` half of `_handle_fromlist`'s suppression test.  An
/// error raised for a *different* name propagates: `from a import b` where
/// `a.b` exists but its body raises `ModuleNotFoundError` for `c` is a real
/// failure.
fn error_names_module(err: &crate::PyError, module_name: &Wtf8) -> bool {
    if err.w_name_context.is_null() || !unsafe { pyre_object::is_str(err.w_name_context) } {
        return false;
    }
    // Compare the raw buffer. A lone surrogate has no `&str` spelling, and
    // `w_str_get_value` would panic on it.
    unsafe { pyre_object::w_str_get_wtf8(err.w_name_context) == module_name }
}

/// Relative import: `from .foo import bar` (level=1), `from ..foo import bar` (level=2).
///
/// PyPy: importing.py `_relative_import()`.
/// Resolves the package base from __package__ or __name__ in w_globals,
/// strips `level - 1` trailing components, then does absolute import.
fn relative_import(
    name: &Wtf8,
    w_globals: PyObjectRef,
    w_fromlist: PyObjectRef,
    level: i64,
    execution_context: *const PyExecutionContext,
) -> Result<PyObjectRef, crate::PyError> {
    use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};

    // `resolve_package_name` reads the caller's namespace, which runs a dict
    // subclass's `__getitem__` and `warnings.warn`; the list handed to
    // `absolute_import` below moves under that.  A null list and `None` are
    // the same argument to `absolute_import`, so the pin can stand in.
    let _roots = push_roots();
    let fromlist_slot = shadow_stack_len();
    let _ = pin_root(if w_fromlist.is_null() {
        pyre_object::w_none()
    } else {
        w_fromlist
    });

    // Get the package name from the calling module's globals.
    // PyPy: pkgname = globals.get('__package__') or globals.get('__name__')
    let package = resolve_package_name(w_globals)?.ok_or_else(|| {
        crate::PyError::new(
            crate::PyErrorKind::ImportError,
            "attempted relative import with no known parent package",
        )
    })?;
    // importlib._bootstrap._resolve_name: an empty fallback package (the
    // `__main__` case) has no parent to anchor even a level-1 import.
    if package.is_empty() {
        return Err(crate::PyError::new(
            crate::PyErrorKind::ImportError,
            "attempted relative import with no known parent package",
        ));
    }

    // Strip (level - 1) trailing components from package
    // PyPy: for dotted name "a.b.c" with level=2, strip "c" → "a.b", then strip "b" → "a"
    let mut parts = split_module_name(&package);
    let strips = (level - 1) as usize;
    if strips >= parts.len() {
        let mut message = Wtf8Buf::new();
        message.push_str("attempted relative import beyond top-level package (package='");
        message.push_wtf8(&package);
        message.push_str("', level=");
        message.push_str(&format!("{level}"));
        message.push_str(")");
        return Err(crate::PyError::new(
            crate::PyErrorKind::ImportError,
            message,
        ));
    }
    for _ in 0..strips {
        parts.pop();
    }
    let mut base = join_module_name(&parts);

    // Build the fully-qualified module name
    if !name.is_empty() {
        base.push_str(".");
        base.push_wtf8(name);
    }

    absolute_import(&base, shadow_stack_get(fromlist_slot), execution_context)
}

/// Extract the package name from the calling module's globals namespace.
///
/// PyPy: importing.py — checks __package__ first, falls back to __name__,
/// strips the last component if __name__ has dots (module in a package).
fn resolve_package_name(w_globals: PyObjectRef) -> Result<Option<Wtf8Buf>, crate::PyError> {
    use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};

    if w_globals.is_null() {
        return Ok(None);
    }

    // Every lookup below runs Python -- a dict subclass's `__getitem__`, a
    // `__spec__.parent` descriptor, `warnings.warn` -- so the namespace and
    // each entry read out of it move under the call that follows them.  That
    // includes the `&str` view into a str payload the `__name__` arm holds.
    let _roots = push_roots();
    let globals_slot = shadow_stack_len();
    let _ = pin_root(w_globals);

    // `space.finditem_str` (baseobjspace.py) maps only KeyError to a
    // missing entry; any other `__getitem__` error (a dict-subclass globals
    // raising) must propagate.  `?` re-raises it; `if let Some(..)` consumes
    // the present case.
    // Python 3.14 importlib._bootstrap._calc___package__: an explicit
    // non-None __package__ wins; otherwise __spec__.parent is authoritative.
    let package = crate::baseobjspace::finditem_str(shadow_stack_get(globals_slot), "__package__")?;
    let has_package = package.is_some();
    let package_slot = shadow_stack_len();
    let _ = pin_root(package.unwrap_or(pyre_object::w_none()));
    let spec = crate::baseobjspace::finditem_str(shadow_stack_get(globals_slot), "__spec__")?;
    let spec_slot = shadow_stack_len();
    let _ = pin_root(spec.unwrap_or(pyre_object::w_none()));

    if has_package {
        let pkg = shadow_stack_get(package_slot);
        if !unsafe { pyre_object::is_none(pkg) } {
            if !unsafe { pyre_object::is_str(pkg) } {
                return Err(crate::PyError::type_error(
                    "__package__ not set to a string",
                ));
            }
            let mut package_name = Wtf8Buf::new();
            package_name.push_wtf8(unsafe { pyre_object::w_str_get_wtf8(pkg) });
            return Ok(Some(package_name));
        }
    }
    if spec.is_some() {
        let w_spec = shadow_stack_get(spec_slot);
        if !unsafe { pyre_object::is_none(w_spec) } {
            let parent = crate::baseobjspace::getattr_str(w_spec, "parent")?;
            if !unsafe { pyre_object::is_str(parent) } {
                return Err(crate::PyError::type_error(
                    "__spec__.parent is not a string",
                ));
            }
            let mut parent_name = Wtf8Buf::new();
            parent_name.push_wtf8(unsafe { pyre_object::w_str_get_wtf8(parent) });
            return Ok(Some(parent_name));
        }
    }

    // _calc___package__ emits ImportWarning before the legacy __name__ /
    // __path__ fallback.  Route it through warnings.warn so assertWarns and
    // user warning filters see the event.
    crate::warn::warn_category(
        "can't resolve package from __spec__ or __package__, falling back on __name__ and __path__",
        "ImportWarning",
        2,
    )?;

    // Fallback: __name__ (for modules inside packages)
    if let Some(name_obj) =
        crate::baseobjspace::finditem_str(shadow_stack_get(globals_slot), "__name__")?
        && !name_obj.is_null()
        && unsafe { pyre_object::is_str(name_obj) }
    {
        let name_slot = shadow_stack_len();
        let _ = pin_root(name_obj);
        // If the module has a __path__, it's a package — use __name__ as-is
        let has_path =
            crate::baseobjspace::finditem_str(shadow_stack_get(globals_slot), "__path__")?
                .is_some();
        let name = unsafe { pyre_object::w_str_get_wtf8(shadow_stack_get(name_slot)) };
        let mut package_name = Wtf8Buf::new();
        if has_path {
            package_name.push_wtf8(name);
            return Ok(Some(package_name));
        }
        // Otherwise `rpartition('.')[0]` is also the empty string for a
        // top-level module such as __main__.
        let end = match name.as_bytes().iter().rposition(|&byte| byte == b'.') {
            Some(dot) => dot,
            None => 0,
        };
        package_name.push_wtf8(unsafe { Wtf8::from_bytes_unchecked(&name.as_bytes()[..end]) });
        return Ok(Some(package_name));
    }

    Ok(None)
}

// ── import_from ──────────────────────────────────────────────────────
// PyPy equivalent: pyopcode.py `IMPORT_FROM`
//
// Get an attribute from the module on TOS. Like `space.getattr(w_module, w_name)`.

/// `importing.py get_spec` — `space.getattr(w_module, '__spec__')`,
/// returning None when the module carries no `__spec__`.  Only a missing
/// attribute is suppressed; any other lookup error propagates.
pub(crate) fn get_spec(module: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    match crate::baseobjspace::getattr_str(module, "__spec__") {
        Ok(v) => Ok(v),
        Err(e) if e.kind == crate::PyErrorKind::AttributeError => Ok(pyre_object::w_none()),
        Err(e) => Err(e),
    }
}

/// `importing.py is_spec_initializing` — a spec whose `_initializing`
/// flag is truthy marks a module still executing, the circular-import signal.
/// A missing `_initializing` reads as not initializing; any other lookup error,
/// and the truth test itself, propagate.
pub(crate) fn is_spec_initializing(w_spec: PyObjectRef) -> Result<bool, crate::PyError> {
    if unsafe { pyre_object::is_none(w_spec) } {
        return Ok(false);
    }
    let w_initializing = match crate::baseobjspace::getattr_str(w_spec, "_initializing") {
        Ok(v) => v,
        Err(e) if e.kind == crate::PyErrorKind::AttributeError => return Ok(false),
        Err(e) => return Err(e),
    };
    crate::baseobjspace::is_true(w_initializing)
}

/// `importing.py is_spec_uninitialized_submodule` — the name appears in the
/// spec's `_uninitialized_submodules` list.  A missing attribute reads as not a
/// submodule; any other lookup error, and the containment test, propagate.
pub(crate) fn is_spec_uninitialized_submodule(
    w_spec: PyObjectRef,
    name: &str,
) -> Result<bool, crate::PyError> {
    if unsafe { pyre_object::is_none(w_spec) } {
        return Ok(false);
    }
    let w_value = match crate::baseobjspace::getattr_str(w_spec, "_uninitialized_submodules") {
        Ok(v) => v,
        Err(e) if e.kind == crate::PyErrorKind::AttributeError => return Ok(false),
        Err(e) => return Err(e),
    };
    let _roots = pyre_object::gc_roots::push_roots();
    let list_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_value);
    let name_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_new_managed(name));
    crate::baseobjspace::contains(
        pyre_object::gc_roots::shadow_stack_get(list_slot),
        pyre_object::gc_roots::shadow_stack_get(name_slot),
    )
}

/// Exact `importlib._bootstrap.ModuleSpec` (not a subclass).  That class's
/// `has_location` is the `_set_fileattr` field (`_bootstrap.py has_location`);
/// a subclass or a custom spec object must go through the public name.
fn is_exact_stdlib_module_spec(mut w_spec: PyObjectRef) -> bool {
    let Some(bootstrap) = pyre_object::with_roots!(w_spec => importlib_bootstrap_module()) else {
        return false;
    };
    let dict = unsafe { pyre_object::w_module_get_w_dict(bootstrap) };
    if dict.is_null() {
        return false;
    }
    let Some(Some(spec_type)) = dict_getitem_str_no_callback(dict, "ModuleSpec") else {
        return false;
    };
    if spec_type.is_null() {
        return false;
    }
    unsafe { (*w_spec).w_class == spec_type }
}

/// `_PyModuleSpec_GetFileOrigin` — the spec's file origin: its `origin` string
/// when `has_location` is truthy and `origin` is a string, otherwise None.  A
/// missing `has_location` / `origin`, or a falsey `has_location`, yields None;
/// other lookup errors and the truth test propagate.
pub(crate) fn spec_file_origin(
    mut w_spec: PyObjectRef,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    if unsafe { pyre_object::is_none(w_spec) } {
        return Ok(None);
    }
    // `_PyModuleSpec_GetFileOrigin` uses `GetOptionalAttr(has_location)`.
    // On the exact stdlib `ModuleSpec` that is the `_set_fileattr` field
    // (`_bootstrap.py has_location`), including `spec.has_location = False`.
    // Reading the property name on that class enters the one-line Python
    // getter on every failed `IMPORT_FROM`, and the getter compiles as a
    // map-specialized function-entry whose map guard fails across specs.
    // A custom spec or a `ModuleSpec` subclass that overrides the property
    // must still see the public name.
    let location_attr = if pyre_object::with_roots!(w_spec => is_exact_stdlib_module_spec(w_spec)) {
        "_set_fileattr"
    } else {
        "has_location"
    };
    let _scope = pyre_object::gc_roots::push_roots();
    let spec_slot = pyre_object::gc_roots::shadow_stack_len();
    let w_spec = pyre_object::gc_roots::pin_root(w_spec);
    let w_has_location = match crate::baseobjspace::getattr_str(w_spec, location_attr) {
        Ok(v) => v,
        Err(e) if e.kind == crate::PyErrorKind::AttributeError => return Ok(None),
        Err(e) => return Err(e),
    };
    if !crate::baseobjspace::is_true(w_has_location)? {
        return Ok(None);
    }
    let w_spec = pyre_object::gc_roots::shadow_stack_get(spec_slot);
    let w_origin = match crate::baseobjspace::getattr_str(w_spec, "origin") {
        Ok(v) => v,
        Err(e) if e.kind == crate::PyErrorKind::AttributeError => return Ok(None),
        Err(e) => return Err(e),
    };
    if !unsafe { pyre_object::is_str(w_origin) } {
        return Ok(None);
    }
    Ok(Some(w_origin))
}

/// `_PyModule_IsPossiblyShadowing` — whether a module loaded from `origin` could
/// be shadowing a same-named module later on the search path.  True when `-P`
/// is off and the file's directory equals the startup `sys.path[0]`; a package
/// `__init__.py` compares its parent directory instead.
pub(crate) fn is_possibly_shadowing(origin: &rustpython_wtf8::Wtf8) -> bool {
    if safe_path_flag() {
        return false;
    }
    let Some(sys_path_0) = SYS_PATH_0.lock().clone() else {
        return false;
    };
    let sep = std::path::MAIN_SEPARATOR as u8;
    // The separator and `__init__.py` are ASCII, and an ASCII byte never
    // occurs inside a multi-byte WTF-8 sequence, so the scan runs on bytes
    // while the name itself may hold a surrogate escape.
    // root = os.path.dirname(origin.removesuffix(os.sep + "__init__.py"))
    let mut root = origin.as_bytes();
    let Some(idx) = root.iter().rposition(|&b| b == sep) else {
        return false;
    };
    if &root[idx + 1..] == b"__init__.py" {
        root = &root[..idx];
        let Some(idx2) = root.iter().rposition(|&b| b == sep) else {
            return false;
        };
        root = &root[..idx2];
    } else {
        root = &root[..idx];
    }
    root == sys_path_0.as_bytes()
}

/// The shadowing classification for a module: its spec file origin (a path
/// string, when it has one), whether it is possibly shadowing a same-named
/// search-path module, and whether that shadowed module is a standard-library
/// one.  `w_name` is the module's `__name__` object, tested against
/// `sys.stdlib_module_names`.
pub(crate) fn module_shadow_info(
    w_spec: PyObjectRef,
    mut w_name: PyObjectRef,
) -> Result<(Option<Wtf8Buf>, bool, bool), crate::PyError> {
    let origin = match pyre_object::with_roots!(w_name => spec_file_origin(w_spec))? {
        // A `spec.origin` is a filename, so it can hold the surrogate escape
        // an undecodable path byte decodes to and has no `&str` spelling.
        Some(o) => unsafe { pyre_object::w_str_get_wtf8(o) }.to_wtf8_buf(),
        None => return Ok((None, false, false)),
    };
    if !is_possibly_shadowing(&origin) {
        return Ok((Some(origin), false, false));
    }
    // Shadowing a same-named search-path module; a standard-library name gets
    // the stronger hint.  `sys.stdlib_module_names` may be replaced or deleted
    // by user code — only a real set participates, and its membership test
    // (which hashes `__name__`) propagates.
    let mut shadowing_stdlib = false;
    if let Some(sys_mod) = get_interpreter_sys_module() {
        let _scope = pyre_object::gc_roots::push_roots();
        let name_slot = pyre_object::gc_roots::shadow_stack_len();
        let w_name = pyre_object::gc_roots::pin_root(w_name);
        let w_names = match crate::baseobjspace::getattr_str(sys_mod, "stdlib_module_names") {
            Ok(v) => v,
            Err(e) if e.kind == crate::PyErrorKind::AttributeError => pyre_object::w_none(),
            Err(e) => return Err(e),
        };
        if unsafe { pyre_object::is_set_or_frozenset(w_names) } {
            let w_name = pyre_object::gc_roots::shadow_stack_get(name_slot);
            shadowing_stdlib = crate::baseobjspace::contains(w_names, w_name)?;
        }
    }
    Ok((Some(origin), true, shadowing_stdlib))
}

pub fn import_from(mut module: PyObjectRef, name: &str) -> Result<PyObjectRef, crate::PyError> {
    // Module.descr_getattribute → object_getattribute → getdictvalue
    // (`module.py getdictvalue` is finditem_str(self.w_dict, attr)).
    // Spelled here so the walk does not enter getattr_str_impl: force()
    // and pin_root are effects, and every dont_look_inside after them is
    // after-effect.  A custom `__getattribute__`, a data descriptor, or a
    // dict miss stays on import_from_slow (full getattr + sys.modules).
    if unsafe { pyre_object::is_module(module) } {
        let w_type = unsafe { (*module).w_class };
        if unsafe { crate::baseobjspace::module_getattribute_if_not_from_default(w_type) }.is_none()
            && !unsafe { crate::baseobjspace::type_lookup_is_data_descr(w_type, name) }
        {
            let dict = unsafe { pyre_object::w_module_get_w_dict(module) };
            if !dict.is_null()
                && let Some(value) = pyre_object::with_roots!(module => crate::baseobjspace::finditem_str(dict, name))?
            {
                return Ok(value);
            }
        }
    }
    import_from_slow(module, name)
}

/// pyopcode.py import_from after the module-dict hit: full getattr,
/// then the sys.modules submodule fallback and ImportError.
#[majit_macros::dont_look_inside]
fn import_from_slow(mut module: PyObjectRef, name: &str) -> Result<PyObjectRef, crate::PyError> {
    // pyopcode.py import_from — first `space.getattr(w_module, w_name)`,
    // which honours the module attribute protocol (`__getattribute__` /
    // `__getattr__`).  Only an AttributeError falls through to the
    // `sys.modules` lookup below; any other error propagates.
    match pyre_object::with_roots!(module => crate::baseobjspace::getattr_str(module, name)) {
        Ok(value) => return Ok(value),
        Err(e) if e.kind == crate::PyErrorKind::AttributeError => {}
        Err(e) => return Err(e),
    }

    // pyopcode.py import_from — the submodule case is a plain
    // `space.getitem(space.sys.get('modules'), w_fullname)` lookup.  No import
    // runs here and nothing is stored back onto the parent: `__import__` has
    // already run `_handle_fromlist` (app-level through `handle_fromlist_fast`
    // / `call_bootstrap_import`, or `handle_fromlist` on the native importer),
    // so this lookup only covers what that binding missed -- a submodule whose
    // `sys.modules` entry predates the import, which `_find_and_load` answers
    // from the cache without touching the parent.
    //
    // import_from Issue #17636 — a submodule already bound in sys.modules under
    // `<__name__>.<name>` is returned even when the parent object rejected the
    // attribute (e.g. a non-module stand-in with restrictive slots that
    // `_handle_fromlist` could not setattr onto). Read `__name__` off the object
    // rather than requiring it to be a module.
    //
    // pyopcode.py import_from — `w_pkgname` starts at the default and is
    // overwritten by ONE `space.getattr(w_module, '__name__')`, whose result
    // both keys this lookup and names the package in the error below.  Reading
    // it twice runs a `__getattr__`- or descriptor-backed `__name__` twice,
    // which the single protected read upstream does not.
    let _roots = pyre_object::gc_roots::push_roots();
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(module);
    let pkgname_slot = pyre_object::gc_roots::shadow_stack_len();
    let (w_pkgname, named) = match crate::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(module_slot),
        "__name__",
    ) {
        Ok(v) if unsafe { pyre_object::is_str(v) } => (v, true),
        _ => (
            pyre_object::w_str_new_managed("<unknown module name>"),
            false,
        ),
    };
    // Pinned for the whole function: every lookup below can run a descriptor
    // or `__getattr__` and so collect, and this name is young and movable.
    let _ = pyre_object::gc_roots::pin_root(w_pkgname);
    if named {
        // A module name is WTF-8 and may carry a lone surrogate, so the key is
        // assembled as WTF-8 and asked for as an object.  Rendering it through
        // a `&str` accessor panics on exactly the names `sys.modules` is still
        // free to hold.
        let mut fullname = Wtf8Buf::new();
        fullname.push_wtf8(unsafe {
            pyre_object::w_str_get_wtf8(pyre_object::gc_roots::shadow_stack_get(pkgname_slot))
        });
        fullname.push_str(".");
        fullname.push_str(name);
        if let Some(submod) = check_sys_modules_w(pyre_object::w_str_from_wtf8_managed(fullname)) {
            return Ok(submod);
        }
    }

    // The name is neither a real attribute nor an importable submodule.
    // `error.py:new_import_error` — raise `ImportError(msg, name=pkgname,
    // path=pkgpath)`: `pkgname` is the package `__name__` (default "<unknown
    // module name>"), `pkgpath` its `__file__` (`get_path`, default "unknown
    // location").  pyopcode.py:1152 resolves the name via
    // `space.getattr(w_module, '__name__')` and importing.py `get_path`
    // via `space.getattr(w_module, '__file__')`, so a descriptor- or
    // `__getattr__`-supplied value and a non-module `from` target are honored;
    // a missing / None path takes the default.
    // pypy/module/imp/importing.py get_path — a non-str `__file__`
    // (including None) reports the location as unknown.
    let w_pkgpath = match crate::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(module_slot),
        "__file__",
    ) {
        Ok(v) if unsafe { pyre_object::is_str(v) } => v,
        Ok(_) => pyre_object::w_str_new_managed("unknown location"),
        Err(e) if e.kind == crate::PyErrorKind::AttributeError => {
            pyre_object::w_str_new_managed("unknown location")
        }
        Err(e) => return Err(e),
    };
    // Pin the path string across the spec lookups below (which allocate) too.
    let pkgpath_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_pkgpath);
    let w_pkgname = pyre_object::gc_roots::shadow_stack_get(pkgname_slot);
    // Own the name so the spec lookups below (which allocate) cannot dangle it.
    // Rendered lossily, the way `format!` already renders every field of these
    // messages except `origin`: a module name is WTF-8 and the `&str` accessor
    // panics on a lone surrogate. `w_pkgname` itself rides the exception
    // untouched, so `exc.name` is still exact.
    let pkgname = unsafe { pyre_object::w_str_get_wtf8(w_pkgname) }
        .to_string_lossy()
        .into_owned();
    // Classify the failure through `__spec__`: a same-named file shadowing a
    // search-path module is flagged first, then a module still executing
    // reports the circular-import cause, then an unset submodule slot.
    let w_spec = get_spec(pyre_object::gc_roots::shadow_stack_get(module_slot))?;
    let spec_slot = pyre_object::gc_roots::shadow_stack_len();
    let w_spec = pyre_object::gc_roots::pin_root(w_spec);
    let w_pkgname = pyre_object::gc_roots::shadow_stack_get(pkgname_slot);
    let (origin, is_shadowing, is_shadowing_stdlib) = module_shadow_info(w_spec, w_pkgname)?;
    let w_spec = pyre_object::gc_roots::shadow_stack_get(spec_slot);
    let initializing = is_spec_initializing(w_spec)?;
    let w_spec = pyre_object::gc_roots::shadow_stack_get(spec_slot);
    let uninit_submodule = !initializing && is_spec_uninitialized_submodule(w_spec, &pkgname)?;
    let pkgpath =
        crate::baseobjspace::utf8_w(pyre_object::gc_roots::shadow_stack_get(pkgpath_slot))?;
    let origin = origin.unwrap_or_default();
    // The origin is a filename and may hold a surrogate escape, so the two
    // messages that name it are assembled as WTF-8; `format!` would render it
    // through `Display` and substitute U+FFFD.
    let renaming_hint = |tail: String| {
        let mut msg = Wtf8Buf::from_string(format!(
            "cannot import name '{name}' from '{pkgname}' (consider renaming '"
        ));
        msg.push_wtf8(&origin);
        msg.push_str(&tail);
        msg
    };
    let msg: Wtf8Buf = if is_shadowing_stdlib {
        renaming_hint(format!(
            "' since it has the same name as the standard library module named \
             '{pkgname}' and prevents importing that standard library module)"
        ))
    } else if initializing {
        if is_shadowing {
            renaming_hint(
                "' if it has the same name as a library you intended to import)".to_string(),
            )
        } else {
            format!(
                "cannot import name '{name}' from partially initialized module \
                 '{pkgname}' (most likely due to a circular import) ({pkgpath})"
            )
            .into()
        }
    } else if uninit_submodule {
        format!(
            "cannot access submodule '{pkgname}' of module '{name}' \
             (most likely due to a circular import)"
        )
        .into()
    } else {
        format!("cannot import name '{name}' from '{pkgname}' ({pkgpath})").into()
    };
    // The name the `from` list asked for rides along as `.name_from`; it is
    // built before the three reads below because the allocation can move the
    // pinned package name and path.
    let name_from_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_new_managed(name));
    let w_pkgname = pyre_object::gc_roots::shadow_stack_get(pkgname_slot);
    let w_pkgpath = pyre_object::gc_roots::shadow_stack_get(pkgpath_slot);
    let w_name_from = pyre_object::gc_roots::shadow_stack_get(name_from_slot);
    Err(crate::PyError::new_import_error_name_from(
        msg,
        w_pkgname,
        w_pkgpath,
        w_name_from,
    ))
}

// ── import_all_from ──────────────────────────────────────────────────
// PyPy equivalent: pyopcode.py `import_all_from(module,
// into_locals)` (applevel function called by IMPORT_STAR).

fn type_name_for_err(w_obj: PyObjectRef) -> String {
    unsafe {
        match crate::typedef::r#type(w_obj) {
            Some(tp) => pyre_object::w_type_get_name(tp.as_ptr()).to_string(),
            None => (*(*w_obj).ob_type).name.to_string(),
        }
    }
}

/// pypy/interpreter/pyopcode.py `import_all_from` — applevel
/// driver.  Iterates `for name in all:` lazily via `space.iter` /
/// `space.next`, applies the per-name str check + leading-underscore
/// filter, and invokes `write` once per accepted name.  Used by the
/// `*mut DictStorage` and generic-mapping wrappers below.
///
/// ```python
/// try:
///     all = module.__all__
/// except AttributeError:
///     try:
///         dict = module.__dict__
///     except AttributeError:
///         raise ImportError("from-import-* object has no __dict__ "
///                           "and no __all__")
///     all = dict.keys()
///     skip_leading_underscores = True
/// else:
///     skip_leading_underscores = False
///
/// module_name = module.__name__
/// if not isinstance(module_name, str):
///     raise TypeError("module __name__ must be a string, not %s",
///                     type(module_name).__name__)
///
/// for name in all:
///     if not isinstance(name, str):
///         ...  # raise TypeError ("Item in <m>.__all__ ..." or
///              #                  "Key in <m>.__dict__ ...")
///     if skip_leading_underscores and name and name[0] == '_':
///         continue
///     into_locals[name] = getattr(module, name)
/// ```
fn import_all_from_each<F>(mut module: PyObjectRef, mut write: F) -> Result<(), crate::PyError>
where
    F: FnMut(&str, PyObjectRef) -> Result<(), crate::PyError>,
{
    let (mut w_iterable, skip_leading_underscores) = match pyre_object::with_roots!(module => crate::baseobjspace::getattr_str(module, "__all__"))
    {
        Ok(w_all) => (w_all, false),
        Err(e) if e.kind == crate::PyErrorKind::AttributeError => {
            // pyopcode.py:2225-2230 — `dict = module.__dict__; all = dict.keys()`.
            // `space.getattr(module, '__dict__')` so any object exposing
            // `__dict__` (Module, class, instance with `__dict__`,
            // bytes-keyed proxies, ...) participates.
            match pyre_object::with_roots!(module => crate::baseobjspace::getattr_str(module, "__dict__"))
            {
                Ok(w_dict) => {
                    let w_keys_method = pyre_object::with_roots!(module => crate::baseobjspace::getattr_str(w_dict, "keys"))?;
                    // pyopcode.py `all = dict.keys()` — pyre's
                    // `call_function` stashes errors as PY_NULL; use
                    // `call_and_check` so a misbehaving `keys()` (or
                    // `__getattr__`-installed override) raises here
                    // rather than handing a bogus iterable to
                    // `space.iter` below.
                    let w_keys = pyre_object::with_roots!(module => crate::builtins::call_and_check(w_keys_method, &[]))?;
                    (w_keys, true)
                }
                Err(e2) if e2.kind == crate::PyErrorKind::AttributeError => {
                    return Err(crate::PyError::new(
                        crate::PyErrorKind::ImportError,
                        "from-import-* object has no __dict__ and no __all__".to_string(),
                    ));
                }
                Err(e2) => return Err(e2),
            }
        }
        Err(e) => return Err(e),
    };

    // pyopcode.py — `module_name = module.__name__` with str check.
    let module_name_w = pyre_object::with_roots!(module, w_iterable => crate::baseobjspace::getattr_str(module, "__name__"))?;
    if !unsafe { is_str(module_name_w) } {
        return Err(crate::PyError::type_error(format!(
            "module __name__ must be a string, not {}",
            type_name_for_err(module_name_w),
        )));
    }
    // Diagnostic only (`pyopcode.py` interpolates it into the TypeError).
    // A lone surrogate must not fail the import of the names in `__all__`.
    let mut module_name = Wtf8Buf::new();
    module_name.push_wtf8(unsafe { pyre_object::w_str_get_wtf8(module_name_w) });

    // pyopcode.py — `for name in all:` lazy iteration.
    let mut w_iter = pyre_object::with_roots!(module => crate::baseobjspace::iter(w_iterable))?;
    loop {
        let next_name =
            pyre_object::with_roots!(module, w_iter => crate::baseobjspace::next(w_iter));
        let mut stop_err = None;
        let w_name = match next_name {
            Ok(v) => Some(v),
            Err(e) => {
                stop_err = Some(e);
                None
            }
        };
        if let Some(mut e) = stop_err {
            let (stop, e) = e.matches_stop_iteration_keep();
            if stop {
                break;
            }
            return Err(e);
        }
        let mut w_name = w_name.unwrap();
        // pyopcode.py — per-name str check.
        if !unsafe { is_str(w_name) } {
            let (container, accessor) = if skip_leading_underscores {
                ("__dict__", "Key")
            } else {
                ("__all__", "Item")
            };
            let mut msg = Wtf8Buf::new();
            msg.push_str(accessor);
            msg.push_str(" in ");
            msg.push_wtf8(&module_name);
            msg.push_str(".");
            msg.push_str(container);
            msg.push_str(" must be str, not ");
            msg.push_str(&type_name_for_err(w_name));
            return Err(crate::PyError::type_error(msg));
        }
        let name =
            pyre_object::with_roots!(module, w_iter, w_name => crate::baseobjspace::str_utf8_w(w_name))?
                .to_string();
        // pyopcode.py:2256-2257 — leading-underscore filter (only for
        // the `__dict__.keys()` fallback).
        if skip_leading_underscores && name.starts_with('_') {
            continue;
        }
        // pyopcode.py — `into_locals[name] = getattr(module, name)`.
        let value = pyre_object::with_roots!(module, w_iter => crate::baseobjspace::getattr_str(module, &name))?;
        write(&name, value)?;
    }
    Ok(())
}

/// pypy/interpreter/pyopcode.py `import_all_from` — applies each
/// public name to the locals mapping object via `space.setitem`.  Errors from
/// `__setitem__` propagate (a misbehaving mapping surfaces its TypeError /
/// KeyError to the caller).
pub fn import_all_from_w(
    module: PyObjectRef,
    into_locals: PyObjectRef,
) -> Result<(), crate::PyError> {
    import_all_from_each(module, |name, value| {
        let _name_roots = pyre_object::gc_roots::push_roots();
        let locals_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(into_locals);
        let name_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_new_managed(name));
        let value_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(value);
        crate::baseobjspace::setitem(
            pyre_object::gc_roots::shadow_stack_get(locals_slot),
            pyre_object::gc_roots::shadow_stack_get(name_slot),
            pyre_object::gc_roots::shadow_stack_get(value_slot),
        )?;
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_startup_body_and_alias_spec() {
        crate::test_hooks::install_hash_hook();
        crate::typedef::init_typeobjects();
        {
            // source_exec_stores_file_from_the_path_buffer
            let ec = std::rc::Rc::new(crate::PyExecutionContext::default());
            crate::call::set_last_exec_ctx(std::rc::Rc::as_ptr(&ec));
            let code = crate::compile::compile_source("x = 1\n", crate::compile::Mode::Exec)
                .expect("compile");
            let w_code = crate::pycode::w_code_new(Box::into_raw(Box::new(code)) as *const ());
            let w_globals = unsafe { &*std::rc::Rc::as_ptr(&ec) }.fresh_module_globals();
            let path = rustpython_wtf8::Wtf8Buf::from("/tmp/m.py");
            exec_code_module(
                w_code,
                w_globals,
                std::rc::Rc::as_ptr(&ec),
                Some(&path),
                None,
                false,
            )
            .expect("exec");
            let file = unsafe { pyre_object::w_dict_getitem_str(w_globals, "__file__") }
                .expect("__file__");
            unsafe {
                assert_eq!(pyre_object::w_str_len(file), 9);
                assert_eq!(pyre_object::w_str_get_wtf8(file), "/tmp/m.py");
            }
            let bound = unsafe { pyre_object::w_dict_getitem_str(w_globals, "x") }.expect("x");
            assert_eq!(unsafe { pyre_object::intobject::w_int_get_value(bound) }, 1);
        }
        {
            // frozen_startup_body_runs_on_the_plain_dispatch
            let ec = std::rc::Rc::new(crate::PyExecutionContext::default());
            crate::call::set_last_exec_ctx(std::rc::Rc::as_ptr(&ec));
            let code = crate::compile::compile_source("y = 2\n", crate::compile::Mode::Exec)
                .expect("compile");
            let w_code = crate::pycode::w_code_new(Box::into_raw(Box::new(code)) as *const ());
            let w_globals = unsafe { &*std::rc::Rc::as_ptr(&ec) }.fresh_module_globals();
            let path = rustpython_wtf8::Wtf8Buf::from("<frozen importlib._bootstrap>");
            exec_code_module(
                w_code,
                w_globals,
                std::rc::Rc::as_ptr(&ec),
                Some(&path),
                None,
                true,
            )
            .expect("plain exec");
            let bound = unsafe { pyre_object::w_dict_getitem_str(w_globals, "y") }.expect("y");
            assert_eq!(unsafe { pyre_object::intobject::w_int_get_value(bound) }, 2);
        }
        {
            // frozen_alias_keeps_an_installed_frozen_spec
            let loader = pyre_object::w_module_new("FrozenImporter");
            let spec = pyre_object::w_module_new("spec");
            let module = pyre_object::w_module_new("_frozen_importlib");
            unsafe {
                pyre_object::dictmultiobject::w_dict_setitem_str(
                    pyre_object::w_module_get_w_dict(spec),
                    "loader",
                    loader,
                );
                pyre_object::dictmultiobject::w_dict_setitem_str(
                    pyre_object::w_module_get_w_dict(module),
                    "__spec__",
                    spec,
                );
            }
            set_frozen_alias_metadata(module, "_frozen_importlib", module).expect("origname");
            let kept = unsafe {
                pyre_object::w_dict_getitem_str(
                    pyre_object::w_module_get_w_dict(module),
                    "__spec__",
                )
            }
            .expect("spec");
            assert!(std::ptr::eq(kept, spec));
            let origname = unsafe {
                pyre_object::w_dict_getitem_str(
                    pyre_object::w_module_get_w_dict(module),
                    "__origname__",
                )
            }
            .expect("origname");
            unsafe {
                assert_eq!(
                    pyre_object::w_str_get_wtf8(origname),
                    "importlib._bootstrap"
                );
            }
        }
    }

    #[test]
    fn bootstrap_seed_and_appleveldef_install() {
        crate::test_hooks::install_hash_hook();
        crate::typedef::init_typeobjects();
        {
            // appleveldef_install_binds_through_the_plain_dispatch
            let ec = std::rc::Rc::new(crate::PyExecutionContext::default());
            crate::call::set_last_exec_ctx(std::rc::Rc::as_ptr(&ec));
            let ns = pyre_object::dictmultiobject::w_dict_new();
            appleveldef_install(
                ns,
                "answer = 1\n",
                "app_answer.py",
                "app_answer",
                &["answer"],
            )
            .expect("applevel body binds answer");
            let bound = unsafe { pyre_object::w_dict_getitem_str(ns, "answer") };
            assert!(bound.is_some(), "answer is copied into the module dict");
        }
        #[cfg(feature = "host_env")]
        {
            // bootstrap_seed_registers_builtins_via_getbuiltinmodule
            install_builtin_modules();
            let ec = std::rc::Rc::new(crate::PyExecutionContext::default());
            crate::call::set_last_exec_ctx(std::rc::Rc::as_ptr(&ec));
            seed_importlib_bootstrap_builtins(std::rc::Rc::as_ptr(&ec)).expect("seed");
            for name in ["sys", "_imp", "_thread", "_warnings", "_weakref"] {
                assert!(
                    check_sys_modules(name).is_some(),
                    "{name} is in sys.modules"
                );
            }
        }
    }

    #[cfg(all(
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32")
    ))]
    #[test]
    fn pyvenv_cfg_bootstrap_parser_reads_only_path_keys() {
        let config = parse_pyvenv_cfg(
            "home = /opt/pyre/bin\n\
             include-system-site-packages = false\n\
             executable = /opt/pyre/bin/pyre\n\
             command = ignored\n",
        );
        assert_eq!(config.home, Some(PathBuf::from("/opt/pyre/bin")));
        assert_eq!(config.executable, Some(PathBuf::from("/opt/pyre/bin/pyre")));
    }

    #[cfg(all(
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32")
    ))]
    #[test]
    fn stdlib_prefix_recognizes_source_and_packaged_layouts() {
        assert_eq!(
            prefix_from_stdlib(Path::new("/src/pyre/lib-python/3")),
            PathBuf::from("/src/pyre")
        );
        assert_eq!(
            prefix_from_stdlib(Path::new("/opt/pyre/lib/pyre3.14t")),
            PathBuf::from("/opt/pyre")
        );
        assert_eq!(
            prefix_from_stdlib(Path::new(
                "/opt/homebrew/Cellar/pyrex/0.0.2/share/pyrex/lib/pyre3.14t"
            )),
            PathBuf::from("/opt/homebrew/Cellar/pyrex/0.0.2")
        );
        assert_eq!(
            prefix_from_stdlib(Path::new("/opt/pyre/python314.zip")),
            PathBuf::from("/opt/pyre")
        );
    }

    #[cfg(all(
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32")
    ))]
    #[test]
    fn stdlib_layout_prefers_the_versioned_zip_without_claiming_a_directory() {
        let tree = tempfile::tempdir().unwrap();
        let packaged = tree.path().join("lib/pyre3.14t");
        std::fs::create_dir_all(&packaged).unwrap();
        std::fs::write(packaged.join("site.py"), "").unwrap();
        let zip = tree.path().join("python314.zip");
        std::fs::write(&zip, "not opened during path discovery").unwrap();

        let (paths, stdlib_dir) = stdlib_at_prefix(tree.path()).unwrap();
        assert_eq!(paths, vec![zip]);
        assert_eq!(stdlib_dir, None);
    }

    #[cfg(all(
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32")
    ))]
    #[test]
    fn startup_config_keeps_venv_and_base_prefixes_distinct() {
        let tree = tempfile::tempdir().unwrap();
        let base = tree.path().join("base");
        let base_executable = base.join("bin/pyre");
        let base_stdlib = base.join("lib/pyre3.14t");
        std::fs::create_dir_all(base_executable.parent().unwrap()).unwrap();
        std::fs::create_dir_all(&base_stdlib).unwrap();
        std::fs::write(&base_executable, "").unwrap();
        std::fs::write(base_stdlib.join("site.py"), "").unwrap();

        let venv = tree.path().join("venv");
        let venv_executable = venv.join("bin/pyre");
        std::fs::create_dir_all(venv_executable.parent().unwrap()).unwrap();
        std::fs::write(&venv_executable, "").unwrap();
        std::fs::write(
            venv.join("pyvenv.cfg"),
            format!(
                "home = {}\nexecutable = {}\ninclude-system-site-packages = false\n",
                base_executable.parent().unwrap().display(),
                base_executable.display()
            ),
        )
        .unwrap();

        let config = compute_startup_path_config_from(venv_executable.clone(), None);
        assert_eq!(config.executable, venv_executable);
        assert_eq!(config.base_executable, base_executable);
        assert_eq!(config.prefix, venv);
        assert_eq!(config.base_prefix, base);
        let mut expected_paths = vec![base_stdlib.clone()];
        append_macos_stdlib_paths(&mut expected_paths, &base_stdlib);
        assert_eq!(config.stdlib_paths, expected_paths);
        assert_eq!(config.stdlib, Some(base_stdlib));
    }

    #[cfg(all(
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32"),
        unix
    ))]
    #[test]
    fn executable_probe_rejects_a_regular_file_without_execute_permission() {
        use std::os::unix::fs::PermissionsExt;

        let tree = tempfile::tempdir().unwrap();
        let executable = tree.path().join("pyre");
        std::fs::write(&executable, "").unwrap();
        std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o600)).unwrap();
        assert!(!exists_and_is_executable(&executable));
        std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
        assert!(exists_and_is_executable(&executable));
    }

    #[test]
    fn import_cache_probe_accepts_exact_module_with_spec() {
        crate::test_hooks::install_hash_hook();
        crate::typedef::init_typeobjects();
        let module = pyre_object::module::w_module_new("import_cache_probe_mod");
        let spec_cls = crate::typedef::make_builtin_type("ImportCacheProbeSpec", |_| {});
        unsafe { pyre_object::w_type_set_hasdict(spec_cls, true) };
        let spec = pyre_object::objectobject::w_instance_new(spec_cls);
        let dict = unsafe { pyre_object::w_module_get_w_dict(module) };
        unsafe { pyre_object::w_dict_setitem_str(dict, "__spec__", spec) };
        set_sys_module("import_cache_probe_mod", module);
        let w_type = unsafe { (*module).w_class };
        assert!(
            unsafe { crate::baseobjspace::module_getattribute_if_not_from_default(w_type) }
                .is_none(),
            "exact module keeps Module.descr_getattribute"
        );
        assert!(
            sys_module_if_initialized("import_cache_probe_mod").is_some(),
            "exact module with a dict-only spec must take the import-cache residual"
        );
    }

    #[test]
    fn import_cache_probe_declines_nondata_initializing_descriptor() {
        crate::test_hooks::install_hash_hook();
        crate::typedef::init_typeobjects();
        let module = pyre_object::module::w_module_new("import_cache_nondescript_mod");
        let spec_cls = crate::typedef::make_builtin_type("ImportCacheNonDataSpec", |_| {});
        unsafe { pyre_object::w_type_set_hasdict(spec_cls, true) };
        let descr = crate::gateway::make_builtin_function("_initializing", |_| {
            Ok(pyre_object::w_bool_from(true))
        });
        crate::type_dict_store(spec_cls, "_initializing", descr);
        unsafe { crate::baseobjspace::mutated(spec_cls, "_initializing") };
        let spec = pyre_object::objectobject::w_instance_new(spec_cls);
        let dict = unsafe { pyre_object::w_module_get_w_dict(module) };
        unsafe { pyre_object::w_dict_setitem_str(dict, "__spec__", spec) };
        set_sys_module("import_cache_nondescript_mod", module);
        unsafe {
            assert!(
                !crate::baseobjspace::type_lookup_is_data_descr(spec_cls, "_initializing"),
                "a function is a non-data descriptor"
            );
            assert!(
                crate::baseobjspace::lookup_in_type(spec_cls, "_initializing").is_some(),
                "the type still exposes _initializing after the dict miss"
            );
        }
        assert!(
            sys_module_if_initialized("import_cache_nondescript_mod").is_none(),
            "getattr would bind the type-level _initializing; the residual must decline"
        );
    }

    #[test]
    fn import_cache_probe_accepts_none_spec_as_initialized() {
        crate::test_hooks::install_hash_hook();
        crate::typedef::init_typeobjects();
        let module = pyre_object::module::w_module_new("import_cache_none_spec_mod");
        let dict = unsafe { pyre_object::w_module_get_w_dict(module) };
        unsafe { pyre_object::w_dict_setitem_str(dict, "__spec__", pyre_object::w_none()) };
        set_sys_module("import_cache_none_spec_mod", module);
        assert!(
            sys_module_if_initialized("import_cache_none_spec_mod").is_some(),
            "_gcd_import treats getattr(None, '_initializing') as initialized"
        );
    }

    #[test]
    fn import_cache_path_probe_sees_dict_path_as_package() {
        crate::test_hooks::install_hash_hook();
        crate::typedef::init_typeobjects();
        let module = pyre_object::module::w_module_new("import_cache_pkg_mod");
        let dict = unsafe { pyre_object::w_module_get_w_dict(module) };
        unsafe {
            pyre_object::w_dict_setitem_str(dict, "__path__", pyre_object::w_list_new_empty())
        };
        assert_eq!(module_is_package_no_callback(module), Some(true));
        unsafe { pyre_object::w_dict_setitem_str(dict, "__path__", pyre_object::w_none()) };
        assert_eq!(module_is_package_no_callback(module), Some(true));
        unsafe { pyre_object::w_dict_delitem_str(dict, "__path__") };
        assert_eq!(module_is_package_no_callback(module), Some(false));
    }

    #[test]
    fn import_from_reads_exact_module_dict() {
        crate::test_hooks::install_hash_hook();
        crate::typedef::init_typeobjects();
        let module = pyre_object::module::w_module_new("import_from_dict_mod");
        let dict = unsafe { pyre_object::w_module_get_w_dict(module) };
        let value = pyre_object::w_int_new(3);
        unsafe { pyre_object::w_dict_setitem_str(dict, "pi", value) };
        let got = import_from(module, "pi").unwrap();
        assert!(unsafe { pyre_object::is_int(got) });
        assert_eq!(unsafe { pyre_object::w_int_get_value(got) }, 3);
    }

    #[test]
    fn spec_file_origin_reads_public_has_location_on_custom_spec() {
        crate::test_hooks::install_hash_hook();
        crate::typedef::init_typeobjects();
        let spec_cls = crate::typedef::make_builtin_type("CustomFileOriginSpec", |_| {});
        unsafe { pyre_object::w_type_set_hasdict(spec_cls, true) };
        let spec = pyre_object::objectobject::w_instance_new(spec_cls);
        crate::baseobjspace::setattr_str(spec, "has_location", pyre_object::w_bool_from(true))
            .unwrap();
        crate::baseobjspace::setattr_str(spec, "origin", pyre_object::w_str_new("keyword.py"))
            .unwrap();
        let origin = spec_file_origin(spec)
            .unwrap()
            .expect("public has_location");
        assert!(unsafe { pyre_object::is_str(origin) });
        assert_eq!(unsafe { pyre_object::w_str_get_wtf8(origin) }, "keyword.py");
    }

    #[test]
    fn spec_file_origin_ignores_set_fileattr_on_custom_spec() {
        crate::test_hooks::install_hash_hook();
        crate::typedef::init_typeobjects();
        let spec_cls = crate::typedef::make_builtin_type("CustomSetFileattrSpec", |_| {});
        unsafe { pyre_object::w_type_set_hasdict(spec_cls, true) };
        let spec = pyre_object::objectobject::w_instance_new(spec_cls);
        crate::baseobjspace::setattr_str(spec, "_set_fileattr", pyre_object::w_bool_from(true))
            .unwrap();
        crate::baseobjspace::setattr_str(spec, "origin", pyre_object::w_str_new("keyword.py"))
            .unwrap();
        assert!(
            spec_file_origin(spec).unwrap().is_none(),
            "a non-ModuleSpec must not take the _set_fileattr shortcut"
        );
    }

    #[test]
    fn importhook_rejects_invalid_absolute_name_and_level() {
        crate::test_hooks::install_hash_hook();
        let empty = importhook(Wtf8::new(""), PY_NULL, PY_NULL, 0, std::ptr::null()).unwrap_err();
        assert_eq!(empty.kind, crate::PyErrorKind::ValueError);
        assert_eq!(empty.message_text(), "Empty module name");

        let negative =
            importhook(Wtf8::new("sys"), PY_NULL, PY_NULL, -1, std::ptr::null()).unwrap_err();
        assert_eq!(negative.kind, crate::PyErrorKind::ValueError);
        assert_eq!(negative.message_text(), "level must be >= 0");

        let globals = pyre_object::w_dict_new();
        unsafe {
            pyre_object::w_dict_setitem_str(globals, "__package__", pyre_object::w_str_new(""));
        }
        let no_parent =
            importhook(Wtf8::new(""), globals, PY_NULL, 1, std::ptr::null()).unwrap_err();
        assert_eq!(no_parent.kind, crate::PyErrorKind::ImportError);
        assert_eq!(
            no_parent.message_text(),
            "attempted relative import with no known parent package"
        );
    }

    #[test]
    fn test_sys_modules_cache() {
        crate::test_hooks::install_hash_hook();
        crate::typedef::init_typeobjects();
        // `check_sys_modules` is `finditem` plus PyPy's `if w_mod`: a stored
        // None is an import miss, whether or not a previous test installed
        // the Python-visible dict. A real module is the cache hit.
        let module = pyre_object::module::w_module_new("test_cached");
        set_sys_module("test_cached", module);
        assert_eq!(check_sys_modules("test_cached"), Some(module));
    }

    #[test]
    fn cflags_include_path_survives_word_splitting() {
        // An ordinary prefix keeps reading the way it always has.
        assert_eq!(
            quote_for_cflags("/opt/pyre/include/pyre3.14t"),
            "/opt/pyre/include/pyre3.14t"
        );
        // Anything a shell would split on, or that would end the region, is
        // spelled so `split_quoted` and a shell both return one word.
        assert_eq!(
            quote_for_cflags("/Volumes/My Disk/include"),
            "\"/Volumes/My Disk/include\""
        );
        assert_eq!(quote_for_cflags("/a\"b\\c"), "\"/a\\\"b\\\\c\"");
    }

    #[test]
    fn test_find_module_nonexistent() {
        // Should not find a module that doesn't exist
        let result = find_module("__nonexistent_pyre_test_module__", None);
        assert!(matches!(result, Ok(None)));
    }

    #[cfg(feature = "wasm_vfs")]
    #[test]
    fn test_embedded_vfs_round_trips() {
        let mount = Path::new("/stdlib");
        let vfs = VfsProvider::from_blob(VFS_BLOB, mount);

        // `re` is a package: its `__init__.py` is a file and `re/` is a dir.
        assert!(vfs.is_file(&mount.join("re/__init__.py")));
        assert!(vfs.is_dir(&mount.join("re")));
        assert!(vfs.is_dir(mount));

        // A top-level module the closure pulls in.
        assert!(vfs.is_file(&mount.join("enum.py")));

        // Source is readable and non-empty; misses report NotFound.
        let src =
            String::from_utf8(vfs.read_to_bytes(&mount.join("re/__init__.py")).unwrap()).unwrap();
        assert!(src.contains("def compile"));
        assert!(vfs.read_to_bytes(&mount.join("re/_nope.py")).is_err());
        assert!(!vfs.is_file(&mount.join("re/_nope.py")));
    }

    #[cfg(all(
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32")
    ))]
    #[test]
    fn a_pth_line_is_taken_up_to_its_first_hash_and_then_stripped() {
        let dir = PathBuf::from(if cfg!(windows) {
            r"C:\deploy"
        } else {
            "/deploy"
        });
        // Written with CRLF, which is what `test_site` writes on Windows and
        // what the file must not carry into an entry.
        let config = parse_pth(
            dir.clone(),
            "#skipme\r\n# also skipped\r\nkeep#trailing\r\n  padded  \r\n\r\n\
             same\r\nsame\r\n.\r\n..\r\nimport site\r\nimport os\r\n",
        );
        assert!(config.import_site);
        // A repeated line is a repeated entry: the file is `sys.path` verbatim.
        assert_eq!(
            config.entries,
            vec![
                dir.join("keep"),
                dir.join("padded"),
                dir.join("same"),
                dir.join("same"),
                dir.clone(),
                dir.parent().unwrap().to_path_buf(),
            ]
        );
    }

    #[test]
    fn signed_string_find_matches_the_rpython_operation() {
        assert_eq!(rpython_str_find_char("pkg.child.leaf", '.', 0), 3);
        assert_eq!(rpython_str_find_char("pkg.child.leaf", '.', 4), 9);
        assert_eq!(rpython_str_find_char("pkg.child.leaf", '.', 10), -1);
        assert_eq!(rpython_str_find_char("plain", '.', 0), -1);
        assert_eq!(rpython_str_slice_prefix("pkg.child", 3), "pkg");
    }

    #[cfg(all(
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32"),
        windows
    ))]
    #[test]
    fn only_a_literally_spelled_device_or_literal_path_skips_normalization() {
        // Spelled with the two backslashes the exemption is written in, so a
        // `..` inside names a file rather than a level to walk up.
        for spelled in [
            r"\\.\device\..\name",
            r"\\?\C:\dir\..\file",
            r"\\?\UNC\host\share\..\x",
        ] {
            assert_eq!(
                normalize_lexically(Path::new(spelled)),
                PathBuf::from(spelled)
            );
        }

        // The same device prefix spelled with slashes is not exempt:
        // `_nt_rnormpath` matches the literal prefixes before it rewrites `/`
        // into `\`, so this one is normalized. What is asserted is that the
        // `..` was resolved, not the separators it comes back with -- this
        // walk never rewrites `/` into `\`, for any path.
        let normalized = normalize_lexically(Path::new("//./device/../name"));
        assert!(
            !normalized
                .components()
                .any(|part| part == Component::ParentDir),
            "{normalized:?}"
        );
        assert!(normalized.ends_with("name"), "{normalized:?}");
        assert!(
            matches!(normalized.components().next(), Some(Component::Prefix(_))),
            "{normalized:?}"
        );
    }
}
