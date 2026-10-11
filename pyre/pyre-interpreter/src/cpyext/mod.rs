//! CPython C-API compatibility layer -- PyPy `pypy/module/cpyext/`.
//!
//! The C-visible object is deliberately *not* pyre's internal
//! [`pyre_object::PyObject`]: PyPy likewise uses a separate raw refcounted
//! mirror (`cpyext/pyobject.py`) and links it to the moving GC object.
//!
//! The file layout follows upstream's: [`pyobject`] is the mirror and its
//! links, [`pyerrors`] the C exception indicator, [`methodobject`] the
//! `PyCFunction` carrier, [`modsupport`] module creation and method-table
//! conversion, and the per-type modules the object constructors and accessors.

#![cfg(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
))]

mod address_table;
pub mod buffer;
pub mod bytearrayobject;
pub mod bytesobject;
pub mod capsule;
pub mod cdatetime;
pub mod classobject;
pub mod codecs;
pub mod complexobject;
pub mod contextvars;
pub mod csources;
pub mod dictobject;
pub mod eval;
pub mod exception;
pub mod floatobject;
pub mod frameobject;
pub mod funcobject;
pub mod gc;
pub mod genericaliasobject;
pub mod import_;
pub mod iterator;
pub mod listobject;
pub mod lock;
pub mod longobject;
pub mod mapping;
pub mod methodobject;
pub mod modsupport;
pub mod namespaceobject;
pub mod number;
pub mod object;
pub mod osmodule;
pub mod pyerrors;
pub mod pyfile;
pub mod pymem;
pub mod pyobject;
pub mod pystate;
pub mod pystrtod;
pub mod pythonrun;
pub mod pytime;
pub mod sequence;
pub mod setobject;
pub mod sliceobject;
pub mod structobject;
pub mod sysmodule;
pub mod tupleobject;
pub mod typeobject;
pub mod unicodeobject;
pub mod unicodewriter;
pub mod warnings;
pub mod weakrefobject;

/// Keep the weakref lifeline with a C mirror until its `tp_dealloc` has had
/// the `PyObject_ClearWeakRefs` opportunity PyPy gives the live `W_Root`.
pub(crate) fn remember_weakref_lifeline(w_obj: PyObjectRef, lifeline: PyObjectRef) {
    weakrefobject::remember_lifeline(w_obj, lifeline);
}

use parking_lot::ReentrantMutex;
use pyre_object::PyObjectRef;
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::ffi::c_int;
use std::hash::BuildHasherDefault;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicIsize, Ordering};

use pyobject::CPyObject;

struct ExtensionCacheEntry {
    /// Copy of the module dict, exactly PyPy `State.extensions[path]`.
    dict: usize,
    /// Keeps the library and its static module definition alive.
    _handle: usize,
}

/// A mutex whose lock word is rebuilt in the child of a `fork`, keeping the
/// payload it guards.
///
/// The data-less `ForkExtensionLoadLock` below cannot serve these tables: the
/// value lives *inside* the lock, so a fresh lock has to carry it over.  The
/// payload is what the child must keep — `fork` copies the address space, so
/// the loaded libraries, their static module definitions and every raw mirror
/// are still mapped, and dropping the census would leave `ob_pyre_link` slots
/// unvisited.  Only the lock word is stale, because the thread that held it
/// does not exist in the child.
///
/// `ptr::read` moves the payload out without acquiring, and `ptr::write` does
/// not drop the abandoned lock — the same reason the data-less wrappers write
/// over theirs rather than replacing them by assignment.
struct ForkMutex<T>(UnsafeCell<parking_lot::Mutex<T>>);
unsafe impl<T: Send> Sync for ForkMutex<T> {}

impl<T> ForkMutex<T> {
    const fn new(value: T) -> Self {
        Self(UnsafeCell::new(parking_lot::Mutex::new(value)))
    }

    fn lock(&self) -> parking_lot::MutexGuard<'_, T> {
        unsafe { &*self.0.get() }.lock()
    }

    unsafe fn reinit_after_fork(&self) {
        let value = unsafe { (*self.0.get()).data_ptr().read() };
        unsafe { self.0.get().write(parking_lot::Mutex::new(value)) };
    }
}

/// Keyed by path, so the hasher is never fed attacker-chosen keys; the default
/// one is spelled out only because it is the const-constructible form.
type ExtensionCache =
    HashMap<PathBuf, ExtensionCacheEntry, BuildHasherDefault<std::hash::DefaultHasher>>;

/// PyPy `State.extensions`: the upstream owner is a process/interpreter state
/// dictionary, so a `HashMap` is intentional here rather than a side-table
/// workaround.  The values are copied module dictionaries, not per-object
/// optimization metadata.
static EXTENSIONS: ForkMutex<ExtensionCache> =
    ForkMutex::new(HashMap::with_hasher(BuildHasherDefault::new()));
static EXTENSIONS_ACTIVE: AtomicBool = AtomicBool::new(false);

/// `State.package_context`, consumed by `PyModule_Create2` while `PyInit_*`
/// runs. Imports are serialized by the import lock, as in PyPy.
static PACKAGE_CONTEXT: ForkMutex<Option<(String, PathBuf)>> = ForkMutex::new(None);

/// PyPy performs cpyext initialization under the GIL. Pyre currently has no
/// process GIL, so this reentrant boundary preserves the same serialization
/// while still allowing an init function to import another extension.
struct ForkExtensionLoadLock(UnsafeCell<ReentrantMutex<()>>);
unsafe impl Sync for ForkExtensionLoadLock {}

impl ForkExtensionLoadLock {
    const fn new() -> Self {
        Self(UnsafeCell::new(ReentrantMutex::new(())))
    }

    fn get(&self) -> &ReentrantMutex<()> {
        unsafe { &*self.0.get() }
    }

    unsafe fn reinit_after_fork(&self) {
        unsafe { self.0.get().write(ReentrantMutex::new(())) };
    }
}

static EXTENSION_LOAD_LOCK: ForkExtensionLoadLock = ForkExtensionLoadLock::new();
static DLOPEN_FLAGS: AtomicIsize = AtomicIsize::new((libc::RTLD_NOW | libc::RTLD_LOCAL) as isize);

pub fn register_sys_dlopenflags(ns: PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    crate::__pyre_put_new!(
        ns_slot,
        "getdlopenflags",
        crate::make_builtin_function_with_arity(
            "getdlopenflags",
            |_| {
                Ok(pyre_object::w_int_new(
                    DLOPEN_FLAGS.load(Ordering::Relaxed) as i64
                ))
            },
            0,
        )
    );
    crate::__pyre_put_new!(
        ns_slot,
        "setdlopenflags",
        crate::make_builtin_function_with_arity(
            "setdlopenflags",
            |args| {
                DLOPEN_FLAGS.store(
                    crate::baseobjspace::gateway_int_w(args[0])? as isize,
                    Ordering::Relaxed,
                );
                Ok(pyre_object::w_none())
            },
            1,
        )
    );
}

type ExtensionLoadGuard = parking_lot::lock_api::ReentrantMutexGuard<
    'static,
    parking_lot::RawMutex,
    parking_lot::RawThreadId,
    (),
>;

#[majit_macros::dont_look_inside]
fn extension_load_lock() -> ExtensionLoadGuard {
    if let Some(guard) = EXTENSION_LOAD_LOCK.get().try_lock() {
        return guard;
    }
    let blocked = crate::module::thread::before_external_block();
    let guard = EXTENSION_LOAD_LOCK.get().lock();
    drop(blocked);
    guard
}

pub fn after_fork_child() {
    unsafe { EXTENSION_LOAD_LOCK.reinit_after_fork() };
    // Every one of these can be held by a thread the child does not have, so
    // the lock word is replaced before anything acquires it — an import or a
    // collection in the child would otherwise block forever.
    unsafe {
        EXTENSIONS.reinit_after_fork();
        PACKAGE_CONTEXT.reinit_after_fork();
        address_table::after_fork_child();
        pyobject::after_fork_child();
        typeobject::after_fork_child();
        modsupport::after_fork_child();
        methodobject::after_fork_child();
        unicodeobject::after_fork_child();
        frameobject::after_fork_child();
        pyerrors::after_fork_child();
        cdatetime::after_fork_child();
        gc::after_fork_child();
        buffer::after_fork_child();
        pystate::after_fork_child();
        dictobject::after_fork_child();
    }
    // `PyInit_*` cannot have been mid-flight in the child, and the parent's
    // half-finished import must not name the next module created here.
    *PACKAGE_CONTEXT.lock() = None;
}

pub const fn soabi() -> &'static str {
    if cfg!(target_os = "macos") {
        "pyre314-darwin"
    } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        "pyre314-x86_64-linux-gnu"
    } else if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        "pyre314-aarch64-linux-gnu"
    } else {
        "pyre314-linux-gnu"
    }
}

pub const fn extension_suffix() -> &'static str {
    if cfg!(target_os = "macos") {
        ".pyre314-darwin.so"
    } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        ".pyre314-x86_64-linux-gnu.so"
    } else if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        ".pyre314-aarch64-linux-gnu.so"
    } else {
        ".pyre314-linux-gnu.so"
    }
}

/// Forward/mark the cached module dictionaries.
///
/// These are interpreter-state roots just like PyPy's `State.extensions`.  The
/// mirror links are *not* here: a P-link is a root only while the C side holds
/// a reference, and the collector decides that itself
/// (`pyobject::init_rawrefcount`).
pub fn walk_gc_roots(visitor: &mut dyn FnMut(&mut PyObjectRef)) {
    if !EXTENSIONS_ACTIVE.load(Ordering::Acquire) {
        return;
    }
    for entry in EXTENSIONS.lock().values_mut() {
        let mut dict = entry.dict as PyObjectRef;
        if !dict.is_null() {
            visitor(&mut dict);
            entry.dict = dict as usize;
        }
    }
}

fn module_from_cached_dict(name: &str, dict: PyObjectRef) -> PyObjectRef {
    let roots = pyre_object::gc_roots::push_roots();
    let dict_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(dict);
    let module = pyre_object::w_module_new_managed(name);
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    let module = roots.pin_root(module);
    let module_dict = unsafe { pyre_object::w_module_get_w_dict(module) };
    crate::type_methods::dict_update1(
        module_dict,
        pyre_object::gc_roots::shadow_stack_get(dict_slot),
    )
    .expect("cpyext cached module dictionaries are ordinary dicts");
    pyre_object::gc_roots::shadow_stack_get(module_slot)
}

fn cached_extension(name: &str, path: &Path) -> Option<PyObjectRef> {
    let roots = pyre_object::gc_roots::push_roots();
    let dict_slot = {
        let cache = EXTENSIONS.lock();
        let dict = cache.get(path)?.dict as PyObjectRef;
        let slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(dict);
        slot
    };
    Some(module_from_cached_dict(
        name,
        pyre_object::gc_roots::shadow_stack_get(dict_slot),
    ))
}

fn cached_extension_handle(path: &Path) -> Option<usize> {
    EXTENSIONS.lock().get(path).map(|entry| entry._handle)
}

fn fixup_extension(module: PyObjectRef, name: &str, path: &Path, handle: usize) {
    let roots = pyre_object::gc_roots::push_roots();
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    let module = roots.pin_root(module);
    crate::importing::set_sys_module(name, module);
    let module = pyre_object::gc_roots::shadow_stack_get(module_slot);
    let dict = unsafe { pyre_object::w_module_get_w_dict(module) };
    let dict_copy = unsafe { pyre_object::dictmultiobject::w_dict_copy(dict) };
    EXTENSIONS.lock().insert(
        path.to_path_buf(),
        ExtensionCacheEntry {
            dict: dict_copy as usize,
            _handle: handle,
        },
    );
    EXTENSIONS_ACTIVE.store(true, Ordering::Release);
}

/// Which entry point an extension published, and under which protocol.
#[derive(Clone, Copy, PartialEq, Eq)]
enum InitProtocol {
    /// `PyModExport_*`, answering with a slot array.
    Export,
    /// `PyInit_*`, answering with a module or a module definition.
    Init,
}

impl InitProtocol {
    fn symbol(self, basename: &InitName) -> String {
        let prefix = match (self, basename.non_ascii) {
            (InitProtocol::Export, false) => "PyModExport_",
            (InitProtocol::Export, true) => "PyModExportU_",
            (InitProtocol::Init, false) => "PyInit_",
            (InitProtocol::Init, true) => "PyInitU_",
        };
        format!("{prefix}{}", basename.text)
    }
}

/// The variable part of a module's export symbol.
///
/// PEP 489: a name outside ASCII is spelled in punycode and looked up under
/// the `U` prefix instead, so the two namespaces cannot collide.
struct InitName {
    text: String,
    non_ascii: bool,
}

/// `importdl.c get_encoded_name`: the short name, encoded to ASCII where it
/// can be and to punycode where it cannot.  `-` is punycode's delimiter and is
/// not a C identifier character, so it becomes `_`.
fn init_basename(name: &str) -> Result<InitName, crate::PyError> {
    let basename = name.rsplit('.').next().unwrap_or(name);
    if basename.is_ascii() {
        return Ok(InitName {
            text: basename.to_string(),
            non_ascii: false,
        });
    }
    // `encode_object` reaches the `punycode` codec, which is written in
    // Python and allocates, so the name it reads is rooted across the call.
    let roots = pyre_object::gc_roots::push_roots();
    let basename_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(pyre_object::w_str_new(basename));
    let encoded = crate::type_methods::encode_object(
        pyre_object::gc_roots::shadow_stack_get(basename_slot),
        "punycode",
        "strict",
    )?;
    let text = String::from_utf8(encoded)
        .map_err(|_| {
            crate::PyError::new(
                crate::PyErrorKind::ImportError,
                format!("punycode encoding of {basename} is not text"),
            )
        })?
        .replace('-', "_");
    Ok(InitName {
        text,
        non_ascii: true,
    })
}

/// The entry point an open library publishes.
///
/// `PyModExport_*` is looked for first, so an extension that publishes both
/// is read under the protocol that describes itself rather than the one that
/// hands back a struct this interpreter would have to agree on the layout of.
fn lookup_init(handle: usize, name: &str) -> Result<Option<(InitProtocol, usize)>, crate::PyError> {
    let basename = init_basename(name)?;
    for protocol in [InitProtocol::Export, InitProtocol::Init] {
        if let Some(address) = lookup_init_address(handle, &protocol.symbol(&basename)) {
            return Ok(Some((protocol, address)));
        }
    }
    Ok(None)
}

/// PyPy `cpyext.api.load_extension_module` looks for the generated CFFI entry
/// point before either cpyext protocol.
fn lookup_cffi_init(handle: usize, name: &str) -> Option<usize> {
    let basename = name.rsplit('.').next().unwrap_or(name);
    lookup_init_address(handle, &format!("_cffi_pypyinit_{basename}"))
}

/// What the lookup reports when neither entry point is there.
fn missing_init_error(name: &str, path: &Path) -> crate::PyError {
    let symbol = match init_basename(name) {
        Ok(basename) => InitProtocol::Init.symbol(&basename),
        // The name could not be encoded at all, so name it as it was written.
        Err(_) => format!("PyInit_{}", name.rsplit('.').next().unwrap_or(name)),
    };
    extension_import_error(
        format!(
            "function {symbol} not found in library '{}'",
            path.display()
        ),
        name,
        path,
    )
}

fn extension_import_error(message: String, name: &str, path: &Path) -> crate::PyError {
    let roots = pyre_object::gc_roots::push_roots();
    let name_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(pyre_object::w_str_new_managed(name));
    let path_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(crate::gateway::fsdecode_os_str(path.as_os_str()));
    crate::PyError::new_import_error(
        message,
        pyre_object::gc_roots::shadow_stack_get(name_slot),
        pyre_object::gc_roots::shadow_stack_get(path_slot),
    )
}

/// Resolve an extension's init entry point, or `None` if the library has none.
///
/// `dlsym` reports a miss by returning NULL, and a resolver that itself
/// returns NULL leaves `dlerror` unset, so the lookup reports success with
/// address 0. `rdynload.dlsym` rejects that, and it must be rejected here too:
/// address 0 transmuted to the init signature is a call through a null pointer.
fn lookup_init_address(handle: usize, symbol: &str) -> Option<usize> {
    match rustpython_host_env::ctypes::lookup_function_symbol_addr(handle, symbol.as_bytes()) {
        Ok(0) | Err(_) => None,
        Ok(address) => Some(address),
    }
}

/// PyPy `cpyext.api.create_extension_module`, single-phase branch.
pub fn load_extension_module(
    name: &str,
    path: &Path,
    spec: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    let _load_guard = extension_load_lock();
    ensure_linked();
    initialize();
    // `host_env::ctypes` deduplicates libraries by the native handle without
    // maintaining an open count. A second open followed by drop would unload
    // the one library owned by `EXTENSIONS`, so resolve PyPy's cache before
    // opening on this host abstraction.
    if let Some(handle) = cached_extension_handle(path) {
        // `create_extension_module` takes the CFFI branch before it reaches the
        // extension cache, so a generated module is rebuilt on every import.
        // Serving one from the cache restores `name` alone and leaves the
        // `name.lib` entry the initializer also registers missing. The CFFI
        // branch runs only with the `pyre-module` hooks; without them the
        // `PyInit_*` module is the one cached.
        let cffi = lookup_cffi_init(handle, name).is_some()
            && crate::importing::optional_module_hooks().is_some();
        if !cffi && lookup_init(handle, name)?.is_none() {
            return Err(missing_init_error(name, path));
        }
        if !cffi && let Some(module) = cached_extension(name, path) {
            let roots = pyre_object::gc_roots::push_roots();
            let module_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = roots.pin_root(module);
            crate::importing::set_sys_module(
                name,
                pyre_object::gc_roots::shadow_stack_get(module_slot),
            );
            return Ok(pyre_object::gc_roots::shadow_stack_get(module_slot));
        }
    }
    let mut mode = DLOPEN_FLAGS.load(Ordering::Relaxed) as c_int;
    if mode & (libc::RTLD_LAZY | libc::RTLD_NOW) == 0 {
        mode |= libc::RTLD_NOW;
    }
    let handle =
        rustpython_host_env::ctypes::open_library_with_mode(path, mode).map_err(|error| {
            extension_import_error(
                format!(
                    "cannot load extension '{}': {}",
                    path.display(),
                    crate::with_causes(&error)
                ),
                name,
                path,
            )
        })?;

    // Every path from here leaves the library loaded, however it fails.  The
    // handle cache keeps one `dlopen` count per library and hands the same
    // handle back to a second open, so closing it here unloads the image
    // outright -- and a multi-phase module never records its path in
    // `EXTENSIONS`, so a later load of a *different* name out of the same file
    // reaches this code while the modules the earlier loads built are still
    // live.  Their types' `tp_traverse` is then a pointer into text nobody has
    // mapped, which the next minor collection walks.  Upstream keeps the
    // library on both of these failures.
    let cffi_address = lookup_cffi_init(handle, name);
    let found = lookup_init(handle, name)?;
    // `_cffi_backend` lives in `pyre-module`. Without its hooks an extension
    // that only exports the cffi init symbol is the same miss as one that
    // exports no init at all.
    if let (Some(address), Some(hooks)) = (cffi_address, crate::importing::optional_module_hooks())
    {
        let module = (hooks.load_cffi1_module)(name, path, address)?;
        fixup_extension(module, name, path, handle);
        return Ok(module);
    }
    let Some((protocol, address)) = found else {
        return Err(missing_init_error(name, path));
    };

    let old_context = PACKAGE_CONTEXT
        .lock()
        .replace((name.to_string(), path.to_path_buf()));
    let answer = match protocol {
        InitProtocol::Export => {
            let export: unsafe extern "C" fn() -> *mut typeobject::CPySlot =
                unsafe { std::mem::transmute(address) };
            let slots = unsafe { export() };
            // The export protocol names the module itself, so the context the
            // single-phase path reads is not this path's to leave set.
            *PACKAGE_CONTEXT.lock() = old_context;
            modsupport::create_module_from_export_slots(slots, spec, name, Some(path))
                .map(InitResult::MultiPhase)
        }
        InitProtocol::Init => {
            let init: unsafe extern "C" fn() -> *mut CPyObject =
                unsafe { std::mem::transmute(address) };
            let result = unsafe { init() };
            *PACKAGE_CONTEXT.lock() = old_context;
            finish_init(name, path, spec, result)
        }
    };

    let init_result = answer?;
    let roots = pyre_object::gc_roots::push_roots();
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(init_result.module());
    match init_result {
        // `create_cpyext_module` returns straight from the multi-phase branch:
        // the definition, not a copied dictionary, is what a later import
        // rebuilds the module from, and `sys.modules` is the importer's to
        // write -- a caller that only creates the module and executes it
        // itself must not find the name there.
        InitResult::MultiPhase(_) => {}
        InitResult::SinglePhase(_) => fixup_extension(
            pyre_object::gc_roots::shadow_stack_get(module_slot),
            name,
            path,
            handle,
        ),
    }
    Ok(pyre_object::gc_roots::shadow_stack_get(module_slot))
}

/// Which of the two initialization protocols `PyInit_*` used.
enum InitResult {
    SinglePhase(PyObjectRef),
    MultiPhase(PyObjectRef),
}

impl InitResult {
    fn module(&self) -> PyObjectRef {
        match *self {
            InitResult::SinglePhase(module) | InitResult::MultiPhase(module) => module,
        }
    }
}

/// Take the init function's result apart, single-phase branch.
///
/// A NULL result means the init function raised; the pending C exception is
/// the error to report, and its absence is upstream's
/// "failed without raising an exception" `SystemError`
/// (`cpyext/api.py:create_extension_module`).
fn finish_init(
    name: &str,
    path: &Path,
    spec: PyObjectRef,
    result: *mut CPyObject,
) -> Result<InitResult, crate::PyError> {
    if result.is_null() {
        return Err(pyerrors::take_pending_error().unwrap_or_else(|| {
            crate::PyError::new(
                crate::PyErrorKind::SystemError,
                format!("initialization of {name} failed without raising an exception"),
            )
        }));
    }
    // `importdl.c _Py_ext_module_loader_result_from_module`: this is checked
    // before the uninitialized-object one, so a module that both left an
    // exception set and came back untyped is reported as the raise.
    if let Some(pending) = pyerrors::take_pending_error() {
        return Err(crate::error::system_error_from_cause(
            format!("initialization of {name} raised unreported exception"),
            pending,
        ));
    }
    if unsafe { (*result).ob_type.is_null() } {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            format!("init function of {name} returned an uninitialized object"),
        ));
    }
    // PEP 489: `PyModuleDef_Init` hands back the definition itself, and the
    // module is created from it against the import spec instead.
    if typeobject::is_module_def(result) {
        // `PACKAGE_CONTEXT` is only consumed by the single-phase path; the
        // multi-phase one names the module from the spec, so clear it here.
        *PACKAGE_CONTEXT.lock() = None;
        // Upstream leaves `__file__` to importlib, which always runs for a
        // multi-phase module there. Pyre's own importer reaches this without
        // importlib, so the path the module was loaded from is recorded here.
        return Ok(InitResult::MultiPhase(
            modsupport::create_module_from_def_and_spec(
                unsafe { modsupport::module_def_of(result) },
                spec,
                name,
                Some(path),
            )?,
        ));
    }
    let module = unsafe { pyobject::from_ref(result) };
    if module.is_null() {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            format!("init function of {name} returned an unsupported object"),
        ));
    }
    // Transfer the init function's owned result to the interpreter object,
    // exactly `get_w_obj_and_decref` in PyPy.
    unsafe { pyobject::decref(result) };
    Ok(InitResult::SinglePhase(module))
}

pub fn create_dynamic(spec: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let spec_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(spec);
    let w_name = crate::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(spec_slot),
        "name",
    )?;
    let name_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_name);
    let w_origin = crate::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(spec_slot),
        "origin",
    )?;
    let origin_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_origin);
    let name =
        crate::baseobjspace::text0_wtf8_w(pyre_object::gc_roots::shadow_stack_get(name_slot))?
            .to_string();
    crate::baseobjspace::text0_wtf8_w(pyre_object::gc_roots::shadow_stack_get(origin_slot))?;
    let path = PathBuf::from(crate::gateway::os_string_from_fs_bytes(
        &crate::gateway::fsencode(pyre_object::gc_roots::shadow_stack_get(origin_slot))?,
    ));
    let path = if path.components().count() == 1 {
        PathBuf::from(".").join(path)
    } else {
        path
    };
    load_extension_module(
        &name,
        &path,
        pyre_object::gc_roots::shadow_stack_get(spec_slot),
    )
}

/// Call a C function through one of the `PyMethodDef` calling conventions --
/// PyPy `cpyext/api.py:generic_cpy_call`.
///
/// `dont_look_inside` for the same reason upstream marks it: the callee is
/// opaque native code, so the tracer has nothing to record past this boundary.
#[majit_macros::dont_look_inside]
pub(super) fn call_cfunction(
    function: *const std::ffi::c_void,
    flags: methodobject::MethFlags,
    w_self: PyObjectRef,
    positional: &[PyObjectRef],
    keywords: &[(String, PyObjectRef)],
) -> Result<PyObjectRef, crate::PyError> {
    call_cfunction_in_class(function, flags, w_self, None, positional, keywords)
}

/// [`call_cfunction`] naming the class a `METH_METHOD` definition was
/// declared in, which its signature takes between the receiver and the
/// arguments.
pub(super) fn call_cfunction_in_class(
    function: *const std::ffi::c_void,
    flags: methodobject::MethFlags,
    w_self: PyObjectRef,
    defining_class: Option<PyObjectRef>,
    positional: &[PyObjectRef],
    keywords: &[(String, PyObjectRef)],
) -> Result<PyObjectRef, crate::PyError> {
    use methodobject::MethFlags;

    if function.is_null() {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "cpyext method definition has no implementation",
        ));
    }
    // Everything the call needs is pinned before the first allocation below:
    // the argument tuple, the keyword dict and the key strings all collect.
    let roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_self);
    for &argument in positional {
        let _ = roots.pin_root(argument);
    }
    for (_, value) in keywords {
        let _ = roots.pin_root(*value);
    }
    let value_slot = |index: usize| pyre_object::gc_roots::shadow_stack_get(base + 1 + index);

    let fastcall = flags.contains(MethFlags::METH_FASTCALL);
    let mut arguments = std::ptr::null_mut();
    let mut keywords_arg = std::ptr::null_mut();
    let mut fastcall_slots: Vec<*mut CPyObject> = Vec::new();
    if fastcall {
        for index in 0..positional.len() + keywords.len() {
            fastcall_slots.push(pyobject::make_ref(value_slot(index)));
        }
        if flags.contains(MethFlags::METH_KEYWORDS) && !keywords.is_empty() {
            let mut name_slots = Vec::with_capacity(keywords.len());
            for (name, _) in keywords.iter() {
                let name_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = roots.pin_root(pyre_object::w_str_new_managed(name));
                name_slots.push(name_slot);
            }
            let names: Vec<PyObjectRef> = name_slots
                .iter()
                .map(|&slot| pyre_object::gc_roots::shadow_stack_get(slot))
                .collect();
            keywords_arg = pyobject::make_ref(pyre_object::tupleobject::w_tuple_new(names));
        }
    } else if !flags.intersects(MethFlags::METH_NOARGS | MethFlags::METH_O) {
        let items: Vec<PyObjectRef> = (0..positional.len()).map(value_slot).collect();
        let tuple = pyre_object::tupleobject::w_tuple_new(items);
        let tuple_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(tuple);
        if flags.contains(MethFlags::METH_KEYWORDS) && !keywords.is_empty() {
            let dict = pyre_object::dictmultiobject::w_dict_new();
            let dict_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = roots.pin_root(dict);
            for (index, (name, _)) in keywords.iter().enumerate() {
                unsafe {
                    pyre_object::dictmultiobject::w_dict_setitem_str(
                        pyre_object::gc_roots::shadow_stack_get(dict_slot),
                        name,
                        value_slot(positional.len() + index),
                    )
                };
            }
            keywords_arg = pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(dict_slot));
        }
        arguments = pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(tuple_slot));
    } else if flags.contains(MethFlags::METH_O) {
        arguments = pyobject::make_ref(value_slot(0));
    }

    let receiver = pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(base));
    let class = match defining_class {
        Some(class) => pyobject::make_ref(class),
        None => std::ptr::null_mut(),
    };
    let result = unsafe {
        if !class.is_null() {
            // `METH_METHOD | METH_FASTCALL | METH_KEYWORDS`, the only
            // combination the flag rides in: the fastcall-with-keywords
            // signature widened by the defining class.
            let call: unsafe extern "C" fn(
                *mut CPyObject,
                *mut CPyObject,
                *const *mut CPyObject,
                isize,
                *mut CPyObject,
            ) -> *mut CPyObject = std::mem::transmute(function);
            call(
                receiver,
                class,
                fastcall_slots.as_ptr(),
                positional.len() as isize,
                keywords_arg,
            )
        } else if fastcall && flags.contains(MethFlags::METH_KEYWORDS) {
            let call: unsafe extern "C" fn(
                *mut CPyObject,
                *const *mut CPyObject,
                isize,
                *mut CPyObject,
            ) -> *mut CPyObject = std::mem::transmute(function);
            call(
                receiver,
                fastcall_slots.as_ptr(),
                positional.len() as isize,
                keywords_arg,
            )
        } else if fastcall {
            let call: unsafe extern "C" fn(
                *mut CPyObject,
                *const *mut CPyObject,
                isize,
            ) -> *mut CPyObject = std::mem::transmute(function);
            call(receiver, fastcall_slots.as_ptr(), positional.len() as isize)
        } else if flags.contains(MethFlags::METH_KEYWORDS) {
            let call: unsafe extern "C" fn(
                *mut CPyObject,
                *mut CPyObject,
                *mut CPyObject,
            ) -> *mut CPyObject = std::mem::transmute(function);
            call(receiver, arguments, keywords_arg)
        } else {
            let call: unsafe extern "C" fn(*mut CPyObject, *mut CPyObject) -> *mut CPyObject =
                std::mem::transmute(function);
            call(receiver, arguments)
        }
    };
    unsafe {
        pyobject::decref(receiver);
        pyobject::decref(class);
        pyobject::decref(arguments);
        pyobject::decref(keywords_arg);
        for slot in fastcall_slots {
            pyobject::decref(slot);
        }
    }
    from_c_result(result)
}

/// `_Py_CheckFunctionResult`: a NULL result must come with an exception, and a
/// non-NULL one must not.
pub(super) fn from_c_result(result: *mut CPyObject) -> Result<PyObjectRef, crate::PyError> {
    if result.is_null() {
        return Err(pyerrors::take_pending_error().unwrap_or_else(|| {
            crate::PyError::new(
                crate::PyErrorKind::SystemError,
                "cpyext function returned NULL without setting an exception",
            )
        }));
    }
    if pyerrors::has_pending_error() {
        let _ = pyerrors::take_pending_error();
        unsafe { pyobject::decref(result) };
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "cpyext function returned a result with an exception set",
        ));
    }
    let value = unsafe { pyobject::from_ref(result) };
    unsafe { pyobject::decref(result) };
    Ok(value)
}

/// `_imp.exec_dynamic` — PyPy `cpyext/api.py:exec_extension_module`.
///
/// A module that already owns a state block has run its slots, so the second
/// call is the no-op that lets pyre's own importer and a later
/// `_imp.exec_dynamic` both name this entry point.
pub fn exec_dynamic(module: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let _load_guard = extension_load_lock();
    let roots = pyre_object::gc_roots::push_roots();
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    let module = roots.pin_root(module);
    if unsafe { pyre_object::module::is_module(module) }
        && !modsupport::has_module_state(pyre_object::gc_roots::shadow_stack_get(module_slot))
    {
        modsupport::exec_def(pyre_object::gc_roots::shadow_stack_get(module_slot))?;
    }
    Ok(pyre_object::w_none())
}

/// Bind the process-global mirrors an extension resolves against.
///
/// Upstream splits this over `State.build_api` and the `@bootstrap_function`
/// hooks each module registers; pyre has one call because the mirrors it
/// prepares are the singletons, the builtin types and the exception types.
fn initialize() {
    // First: every mirror below creates a P-link, and a link needs the
    // collector's rawrefcount state to exist.
    pyobject::init_rawrefcount();
    // Before the singletons: binding one resolves its `ob_type`, and a type
    // reached before its static is entered gets a synthesized block instead,
    // which is then the block `Py_TYPE(Py_True)` answers with forever.
    typeobject::init_type_mirrors();
    pyobject::init_singletons();
    pyerrors::init_exception_mirrors();
}

/// Force the linker to retain the public C entry points in every native pyre
/// executable. `pyrex/build.rs` additionally exports these names from the main
/// program so a bundle/shared object can resolve them at `dlopen` time.
pub fn ensure_linked() {
    std::hint::black_box(&raw const pyobject::_Py_NoneStruct);
    std::hint::black_box(&raw const pyobject::_Py_TrueStruct);
    std::hint::black_box(&raw const pyobject::_Py_FalseStruct);
    std::hint::black_box(&raw const pyobject::_Py_NotImplementedStruct);
    std::hint::black_box(&raw const pyobject::_Py_EllipsisObject);
    csources::ensure_linked();
    pyobject::ensure_linked();
    pystate::ensure_linked();
    pystrtod::ensure_linked();
    pythonrun::ensure_linked();
    pytime::ensure_linked();
    namespaceobject::ensure_linked();
    bytearrayobject::ensure_linked();
    complexobject::ensure_linked();
    cdatetime::ensure_linked();
    osmodule::ensure_linked();
    warnings::ensure_linked();
    exception::ensure_linked();
    weakrefobject::ensure_linked();
    gc::ensure_linked();
    genericaliasobject::ensure_linked();
    lock::ensure_linked();
    unicodewriter::ensure_linked();
    pyerrors::ensure_linked();
    pyfile::ensure_linked();
    pymem::ensure_linked();
    setobject::ensure_linked();
    sliceobject::ensure_linked();
    sysmodule::ensure_linked();
    modsupport::ensure_linked();
    methodobject::ensure_linked();
    object::ensure_linked();
    longobject::ensure_linked();
    floatobject::ensure_linked();
    unicodeobject::ensure_linked();
    bytesobject::ensure_linked();
    tupleobject::ensure_linked();
    typeobject::ensure_linked();
    listobject::ensure_linked();
    dictobject::ensure_linked();
    capsule::ensure_linked();
    classobject::ensure_linked();
    codecs::ensure_linked();
    import_::ensure_linked();
    number::ensure_linked();
    sequence::ensure_linked();
    mapping::ensure_linked();
    iterator::ensure_linked();
    buffer::ensure_linked();
    eval::ensure_linked();
    contextvars::ensure_linked();
}

#[cfg(test)]
mod tests {
    /// Every source file under `src/cpyext`, by name.
    fn sources() -> Vec<(String, String)> {
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/cpyext");
        let mut sources: Vec<(String, String)> = std::fs::read_dir(&directory)
            .expect("src/cpyext is readable")
            .map(|entry| entry.expect("a directory entry").path())
            .filter(|path| path.extension().is_some_and(|kind| kind == "rs"))
            .map(|path| {
                let name = path
                    .file_name()
                    .expect("a file name")
                    .to_string_lossy()
                    .into_owned();
                (name, std::fs::read_to_string(&path).expect("a source file"))
            })
            .collect();
        assert!(!sources.is_empty(), "no sources at {}", directory.display());
        sources.sort();
        sources
    }

    /// A lock a forked child inherits from a thread it does not have is reset
    /// before anything acquires it, and `reinit_after_fork` is the only way to
    /// do that.  One declared without a reset hangs the child on its first
    /// acquisition, which is a deadlock rather than an error.
    ///
    /// The `BoundFamily.taken` locks are a field of a table rather than a
    /// static of their own; `typeobject::after_fork_child` walks
    /// `BOUND_FAMILIES` to reach them.
    ///
    /// An [`address_table::AddressTable`] holds one of these locks, so its
    /// statics are checked under the same rule.
    #[test]
    fn every_fork_lock_is_reset_in_the_child() {
        let sources = sources();
        let whole: String = sources.iter().map(|(_, text)| text.as_str()).collect();
        let mut checked = 0;
        for (file, text) in &sources {
            for line in text.lines() {
                let Some(declaration) = line.trim_start().strip_prefix("static ") else {
                    continue;
                };
                if !declaration.contains("ForkMutex<")
                    && !declaration.contains("AddressTable<")
                    && !declaration.contains("ForkExtensionLoadLock")
                {
                    continue;
                }
                let (name, _) = declaration.split_once(':').expect("a typed static");
                assert!(
                    whole.contains(&format!("{name}.reinit_after_fork()")),
                    "{file} declares {name}, and nothing resets it after a fork",
                );
                checked += 1;
            }
        }
        assert!(checked > 0, "no fork locks found to check");
    }

    #[test]
    fn every_module_hook_is_called() {
        let sources = sources();
        let (_, root) = sources
            .iter()
            .find(|(name, _)| name == "mod.rs")
            .expect("cpyext/mod.rs");
        let cases = [
            (
                "after_fork_child",
                "pub fn after_fork_child() {",
                "pub(super) unsafe fn after_fork_child()",
                "after_fork_child()",
            ),
            (
                "ensure_linked",
                "pub fn ensure_linked() {",
                "pub(super) fn ensure_linked()",
                "ensure_linked()",
            ),
        ];
        for (hook, entry, decl, call) in cases {
            let body = root
                .split_once(entry)
                .unwrap_or_else(|| panic!("case {hook}: the entry point"))
                .1
                .split_once("\n}\n")
                .unwrap_or_else(|| panic!("case {hook}: its end"))
                .0;
            for (file, text) in &sources {
                if file == "mod.rs" || !text.contains(decl) {
                    continue;
                }
                let module = file.strip_suffix(".rs").expect("a Rust source");
                assert!(
                    body.contains(&format!("{module}::{call}")),
                    "case {hook}: {file} has a {hook} that cpyext::{hook} never calls",
                );
            }
        }
    }
}
