//! Optional builtin modules — the `pypy/module/` subset the interpreter never
//! reaches by name.
//!
//! A module belongs in `pyre-interpreter` when any of three things holds:
//!
//! - the interpreter reaches it by name — `import`, `absolute_import`,
//!   `import_module`, or upstream's `space.getbuiltinmodule`;
//! - PyPy's `essential_modules` names it (`_opcode`, `__pypy__`);
//! - CPython's `Modules/Setup.bootstrap` names it (`_abc`, `_functools`,
//!   `_stat`, `_symtable`, `_types`, `faulthandler`, `pwd`, `time`).
//!
//! Everything else belongs here, and the test runs in both directions: a
//! module sitting in `pyre-interpreter` that answers none of the three belongs
//! here instead.
//!
//! PyPy's `default_modules` tier is not one of the three. `pypyoption.py`
//! gives every module a `BoolOption(modname, default=modname in
//! default_modules)`, so that tier means "on by default, switchable off" —
//! which is what `pyrex`'s `default = [..., "pyre-module"]` already
//! expresses. `math` and `cmath` are `default_modules` and stay here.

// `(lo..hi).contains(&x)` is a `Range::contains` call the trace walker does
// not descend; a JIT-visible range check stays two comparisons.
#![allow(clippy::manual_range_contains)]

/// Adapt a `W_Root` sweep hook (`fn(PyObjectRef)`) to the collector's
/// address-taking `DestructorFn`.
macro_rules! gc_destructor {
    ($hook:path) => {{
        unsafe fn destructor(obj_addr: usize) {
            unsafe { $hook(obj_addr as pyre_object::PyObjectRef) }
        }
        destructor as majit_gc::trace::DestructorFn
    }};
}

/// Keep the `module::` path prefix so harvested hint paths and
/// `should_lower_module` stay `module::<name>` after the crate split.
pub mod module;

/// Register default/working modules into the interpreter builtin table.
///
/// The interpreter does not depend on this crate. The final binary calls
/// [`register`] before `install_builtin_modules`.
pub fn install_optional_modules() {
    pyre_interpreter::importing::register_builtin_module("_blake2", module::_blake2::init);
    #[cfg(all(
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32")
    ))]
    pyre_interpreter::importing::register_builtin_module(
        "_cffi_backend",
        module::_cffi_backend::init,
    );
    // `lib_pypy/_ctypes` sits on `_rawffi`, which needs the host's dynamic
    // linker and libffi; without them `import _ctypes` is an ImportError.
    #[cfg(all(any(unix, windows), feature = "host_env", not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("_ctypes", module::_ctypes::init);
    pyre_interpreter::importing::register_builtin_module("_bz2", module::bz2::init);
    pyre_interpreter::importing::register_builtin_module("_pickle", module::_pickle::init);
    pyre_interpreter::importing::register_builtin_module("_random", module::_random::init);
    pyre_interpreter::importing::register_builtin_module("_tokenize", module::_tokenize::init);
    pyre_interpreter::importing::register_builtin_module("_csv", module::_csv::init);
    pyre_interpreter::importing::register_builtin_module("_codecs_cn", module::_codecs_cn::init);
    pyre_interpreter::importing::register_builtin_module("_codecs_hk", module::_codecs_hk::init);
    pyre_interpreter::importing::register_builtin_module(
        "_codecs_iso2022",
        module::_codecs_iso2022::init,
    );
    pyre_interpreter::importing::register_builtin_module("_codecs_jp", module::_codecs_jp::init);
    pyre_interpreter::importing::register_builtin_module("_codecs_kr", module::_codecs_kr::init);
    pyre_interpreter::importing::register_builtin_module("_codecs_tw", module::_codecs_tw::init);
    pyre_interpreter::importing::register_builtin_module("_hashlib", module::_hashlib::init);
    pyre_interpreter::importing::register_builtin_module("_json", module::_json::init);
    pyre_interpreter::importing::register_builtin_module("_lsprof", module::_lsprof::init);
    pyre_interpreter::importing::register_builtin_module("_lzma", module::_lzma::init);
    // fficurses.py `guess_eci`: the module is present when the probe linked.
    #[cfg(all(
        unix,
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32"),
        pyre_minimal_curses
    ))]
    pyre_interpreter::importing::register_builtin_module(
        "_minimal_curses",
        module::_minimal_curses::moduledef::init,
    );
    pyre_interpreter::importing::register_builtin_module(
        "_multibytecodec",
        module::_multibytecodec::init,
    );
    #[cfg(not(feature = "sandbox"))]
    pyre_interpreter::importing::register_builtin_module(
        "_multiprocessing",
        module::_multiprocessing::init,
    );
    #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("_overlapped", module::_overlapped::init);
    #[cfg(all(unix, not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("_posixshmem", module::_posixshmem::init);
    #[cfg(all(unix, not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module(
        "_posixsubprocess",
        module::_posixsubprocess::init,
    );
    pyre_interpreter::importing::register_builtin_module(
        "_pypy_generic_alias",
        module::_pypy_generic_alias::init,
    );
    pyre_interpreter::importing::register_builtin_module("_queue", module::_queue::init);
    pyre_interpreter::importing::register_builtin_module("gc", module::gc::moduledef::init);
    pyre_interpreter::importing::register_builtin_module_with_startup(
        "array",
        module::array::init_array_module,
        module::array::startup_array_module,
    );
    #[cfg(all(not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("_socket", module::_socket::init);
    #[cfg(all(not(target_arch = "wasm32"), not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("_ssl", module::_ssl::init);
    pyre_interpreter::importing::register_builtin_module("_statistics", module::_statistics::init);
    #[cfg(windows)]
    pyre_interpreter::importing::register_builtin_module("_winapi", module::_winapi::init);
    #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("_wmi", module::_wmi::init);
    #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("_uuid", module::_uuid::init);
    #[cfg(target_os = "macos")]
    pyre_interpreter::importing::register_builtin_module("_scproxy", module::_scproxy::init);
    pyre_interpreter::importing::register_builtin_module("binascii", module::binascii::init);
    pyre_interpreter::importing::register_builtin_module("cmath", module::cmath::init);
    pyre_interpreter::importing::register_builtin_module("math", module::math::init);
    #[cfg(all(windows, feature = "host_env"))]
    pyre_interpreter::importing::register_builtin_module("msvcrt", module::msvcrt::init);
    #[cfg(all(
        not(target_arch = "wasm32"),
        feature = "host_env",
        not(feature = "sandbox")
    ))]
    pyre_interpreter::importing::register_builtin_module("mmap", module::mmap::init);
    #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("fcntl", module::fcntl::init);
    #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("grp", module::grp::init);
    pyre_interpreter::importing::register_builtin_module("pyexpat", module::pyexpat::init);
    #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("resource", module::resource::init);
    #[cfg(all(not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("select", module::select::init);
    #[cfg(all(unix, not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("syslog", module::syslog::init);
    #[cfg(all(unix, not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("termios", module::termios::init);
    #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
    pyre_interpreter::importing::register_builtin_module("winsound", module::winsound::init);
    pyre_interpreter::importing::register_builtin_module(
        "unicodedata",
        module::unicodedata::moduledef::init,
    );
    pyre_interpreter::importing::register_builtin_module("zlib", module::zlib::init);
}

/// Immortal `#[pyre_class]` types allocated through `allocate`.  The
/// collector never walks them, so `build_gc` only registers their
/// `w_class` offset.  `select` is compiled out of a sandbox build
/// (`module/mod.rs`'s `pub mod select`), so its descriptors carry that
/// gate too.
pub fn all_immortal_w_class_only_descriptors()
-> Vec<&'static pyre_object::lltype::PyreClassDescriptor> {
    #[allow(unused_imports)]
    use pyre_object::lltype::PyreClassPyTypeOf;
    vec![
        #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
        <module::select::interp_select::Poll as PyreClassPyTypeOf>::DESCRIPTOR,
        #[cfg(all(target_os = "macos", feature = "host_env", not(feature = "sandbox")))]
        <module::select::interp_kqueue::W_Kqueue as PyreClassPyTypeOf>::DESCRIPTOR,
        #[cfg(all(target_os = "macos", feature = "host_env", not(feature = "sandbox")))]
        <module::select::interp_kevent::W_Kevent as PyreClassPyTypeOf>::DESCRIPTOR,
    ]
}

/// Install [`install_optional_modules`] as the interpreter's optional-module hook.
///
/// Also run from a constructor so a binary that links this crate publishes
/// the rclass aliases before any `init_subclass_ranges` OnceLock in a
/// parallel test can freeze an incomplete census. PyPy has the same
/// modules in the translated program from process start.
#[::ctor::ctor(unsafe)]
fn register_on_load() {
    register();
}

fn hook_mini_buffer_params(obj: pyre_object::PyObjectRef) -> Option<(*mut u8, usize)> {
    #[cfg(all(
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32")
    ))]
    {
        module::_cffi_backend::cbuffer::mini_buffer_params(obj)
    }
    #[cfg(not(all(
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32")
    )))]
    {
        let _ = obj;
        None
    }
}

fn hook_ctype_size_qmut(
    ptr: pyre_object::gc_hook::GCREF,
) -> Option<std::sync::Arc<pyre_object::quasiimmut::QuasiImmut>> {
    #[cfg(all(
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32")
    ))]
    {
        Some(unsafe {
            (*(ptr as *const module::_cffi_backend::ctypeobj::W_CType))
                .mutate_size()
                .get_current_qmut_instance()
        })
    }
    #[cfg(not(all(
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32")
    )))]
    {
        let _ = ptr;
        None
    }
}

fn hook_cdata_is_function_pointer(obj: pyre_object::PyObjectRef) -> bool {
    #[cfg(all(
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32")
    ))]
    {
        module::_cffi_backend::cdataobj::is_function_pointer_cdata(obj)
    }
    #[cfg(not(all(
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32")
    )))]
    {
        let _ = obj;
        false
    }
}

fn hook_cffi_finalizer_kind(obj: pyre_object::PyObjectRef) -> Option<bool> {
    #[cfg(all(
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32")
    ))]
    {
        module::_cffi_backend::cdataobj::ec_finalizer_kind(obj)
    }
    #[cfg(not(all(
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32")
    )))]
    {
        let _ = obj;
        None
    }
}

fn hook_run_cffi_finalize(obj: pyre_object::PyObjectRef) {
    #[cfg(all(
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32")
    ))]
    {
        module::_cffi_backend::cdataobj::run_ec_finalize(obj);
    }
    #[cfg(not(all(
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32")
    )))]
    {
        let _ = obj;
    }
}

fn hook_close_cffi_fileobj(obj: pyre_object::PyObjectRef) {
    #[cfg(all(
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32")
    ))]
    {
        module::_cffi_backend::ctypeptr::close_cffi_fileobj(obj);
    }
    #[cfg(not(all(
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32")
    )))]
    {
        let _ = obj;
    }
}

fn hook_ctypes_buffer_view(
    obj: pyre_object::PyObjectRef,
) -> Option<(
    pyre_object::PyObjectRef,
    usize,
    usize,
    String,
    usize,
    Vec<usize>,
)> {
    #[cfg(all(any(unix, windows), feature = "host_env", not(feature = "sandbox")))]
    {
        module::_ctypes::cdata::cdata_buffer_view(obj)
    }
    #[cfg(not(all(any(unix, windows), feature = "host_env", not(feature = "sandbox"))))]
    {
        let _ = obj;
        None
    }
}

fn hook_ctypes_bytes_object(obj: pyre_object::PyObjectRef) -> Option<pyre_object::PyObjectRef> {
    #[cfg(all(any(unix, windows), feature = "host_env", not(feature = "sandbox")))]
    {
        module::_ctypes::cdata::cdata_bytes_object(obj)
    }
    #[cfg(not(all(any(unix, windows), feature = "host_env", not(feature = "sandbox"))))]
    {
        let _ = obj;
        None
    }
}

fn hook_ctypes_array_instance(obj: pyre_object::PyObjectRef) -> bool {
    #[cfg(all(any(unix, windows), feature = "host_env", not(feature = "sandbox")))]
    {
        module::_ctypes::metaclass::is_array_instance(obj)
    }
    #[cfg(not(all(any(unix, windows), feature = "host_env", not(feature = "sandbox"))))]
    {
        let _ = obj;
        false
    }
}

fn hook_ctypes_pointer_instance(obj: pyre_object::PyObjectRef) -> bool {
    #[cfg(all(any(unix, windows), feature = "host_env", not(feature = "sandbox")))]
    {
        module::_ctypes::metaclass::is_pointer_instance(obj)
    }
    #[cfg(not(all(any(unix, windows), feature = "host_env", not(feature = "sandbox"))))]
    {
        let _ = obj;
        false
    }
}

fn hook_load_cffi1_module(
    name: &str,
    path: &std::path::Path,
    init_address: usize,
) -> Result<pyre_object::PyObjectRef, pyre_interpreter::PyError> {
    #[cfg(all(
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32")
    ))]
    {
        module::_cffi_backend::cffi1_module::load_cffi1_module(name, path, init_address)
    }
    #[cfg(not(all(
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32")
    )))]
    {
        let _ = (name, path, init_address);
        Err(pyre_interpreter::PyError::system_error(
            "_cffi_backend is not available",
        ))
    }
}

pub fn register() {
    pyre_interpreter::importing::set_optional_module_hooks(
        pyre_interpreter::importing::OptionalModuleHooks {
            install_modules: install_optional_modules,
            walk_global_roots: walk_optional_global_roots,
            walk_prebuilt_slots: |fwd| {
                module::_csv::walk_csv_state_gc(fwd);
                // `_compat_pickle`'s fix_imports tables are a
                // `space.fromcache(State)` off-GC slot published lazily without
                // `mark_prebuilt_roots_dirty`, so its possibly young dicts must
                // be forwarded on the first collection.
                module::_pickle::walk_pickle_state_gc(fwd);
            },
            publish_fnaddrs: publish_optional_fnaddrs,
            mini_buffer_params: hook_mini_buffer_params,
            cffi_finalizer_kind: hook_cffi_finalizer_kind,
            cdata_is_function_pointer: hook_cdata_is_function_pointer,
            ctype_size_qmut: hook_ctype_size_qmut,
            run_cffi_finalize: hook_run_cffi_finalize,
            close_cffi_fileobj: hook_close_cffi_fileobj,
            load_cffi1_module: hook_load_cffi1_module,
            ctypes_buffer_view: hook_ctypes_buffer_view,
            ctypes_bytes_object: hook_ctypes_bytes_object,
            ctypes_array_instance: hook_ctypes_array_instance,
            ctypes_pointer_instance: hook_ctypes_pointer_instance,
            gc_types: module_gc_types,
            immortal_w_class_only_descriptors: all_immortal_w_class_only_descriptors,
            libffi_cif_shape: hook_libffi_cif_shape,
            gc_initialize: hook_gc_initialize,
            init_array_type: module::array::init_array_type,
            array_w_getitem: module::array::array_w_getitem,
            array_repr_wtf8: module::array::array_repr_wtf8,
        },
    );
}

/// `interp_gc.py`'s hook installation. The execution context only needs the
/// hooks bound to the shared action flag, not the object `initialize` returns.
fn hook_gc_initialize(
    space: pyre_object::PyObjectRef,
    actionflag: &mut (dyn pyre_interpreter::executioncontext::ActionFlagOps + 'static),
) {
    let _ = module::gc::hook::initialize(space, actionflag);
}

/// `jit_libffi.py`'s reading of a `CIF_DESCRIPTION` block for the tracer.
unsafe fn hook_libffi_cif_shape(
    cif_description: usize,
) -> Option<pyre_interpreter::importing::LibffiCifShape> {
    #[cfg(all(
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32")
    ))]
    {
        use majit_rlib::jit_libffi::{self, types};
        use pyre_interpreter::importing::{LibffiCifShape, LibffiType};
        // `getkind(0)` is `OTHER`; there is no record to size.
        let read = |ffi_type: usize| LibffiType {
            kind: unsafe { types::getkind(ffi_type) } as u8,
            size: if ffi_type == 0 {
                0
            } else {
                unsafe { types::getsize(ffi_type) }
            },
        };
        let nargs = unsafe { jit_libffi::nargs(cif_description) };
        Some(LibffiCifShape {
            rtype: read(unsafe { jit_libffi::rtype(cif_description) }),
            args: (0..nargs)
                .map(|i| {
                    (
                        read(unsafe { jit_libffi::atype(cif_description, i) }),
                        unsafe { jit_libffi::exchange_arg(cif_description, i) },
                    )
                })
                .collect(),
        })
    }
    #[cfg(not(all(
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32")
    )))]
    {
        let _ = cif_description;
        None
    }
}

/// The GC types this crate's modules own, in `build_gc` registration order.
/// `build_gc` numbers them consecutively from
/// `pyre_interpreter::MODULE_FIRST_TYPE_ID`.
fn module_gc_types() -> Vec<pyre_interpreter::importing::ModuleGcType> {
    let mut types = Vec::new();
    module::_tokenize::gc_types(&mut types);
    module::unicodedata::moduledef::gc_types(&mut types);
    module::_json::gc_types(&mut types);
    module::_hashlib::gc_types(&mut types);
    module::zlib::gc_types(&mut types);
    module::bz2::gc_types(&mut types);
    module::_lzma::gc_types(&mut types);
    module::_lsprof::gc_types(&mut types);
    module::_queue::gc_types(&mut types);
    module::gc::moduledef::gc_types(&mut types);
    module::_pickle::gc_types(&mut types);
    module::_random::gc_types(&mut types);
    #[cfg(all(not(target_arch = "wasm32"), not(feature = "sandbox")))]
    module::_ssl::gc_types(&mut types);
    #[cfg(all(
        not(target_arch = "wasm32"),
        feature = "host_env",
        not(feature = "sandbox")
    ))]
    module::mmap::gc_types(&mut types);
    #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
    module::_overlapped::gc_types(&mut types);
    #[cfg(windows)]
    module::_winapi::gc_types(&mut types);
    #[cfg(all(
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32")
    ))]
    module::_cffi_backend::gc_types(&mut types);
    module::_csv::gc_types(&mut types);
    types
}

/// Residual-call targets that only this crate's modules provide.
fn publish_optional_fnaddrs(entries: &mut Vec<(&'static str, i64)>) {
    fn single(entries: &mut Vec<(&'static str, i64)>, path: &'static str, fnptr: *const ()) {
        let addr = fnptr as usize as i64;
        if addr != 0 {
            entries.push((path, addr));
        }
    }

    // `pymath` is outside the extraction set, so every call of it reaches the
    // artifact as an un-lowerable target.  `ulp` takes and returns one float,
    // which the residual-call ABI carries, so binding its real address makes
    // the call executable; the rest of the family returns `Result<f64, _>`,
    // which is wider than a result slot, and stays unpublished.
    single(
        entries,
        "pymath::math::misc::ulp",
        pymath::math::ulp as *const (),
    );
    // `rpython/rlib/rrandom.py Random.genrand32` contains the Mersenne Twister
    // refill loops. `JitPolicy.look_inside_graph` rejects the loopy graph (it is
    // not `@jit.unroll_safe`), so `Random.random` keeps two ordinary residual
    // calls to the translated native helper. Publish that helper's address just
    // as RPython's source translation/link step does; otherwise the codewriter
    // can only emit a `symbolic_fnaddr_for_path` hash and an inline sub-walk
    // must abort before reaching the native residual.
    {
        let genrand32: fn(&mut module::_random::Random) -> u32 = module::_random::Random::genrand32;
        let addr = genrand32 as *const () as usize as i64;
        if addr != 0 {
            entries.push(("pyre_interpreter::module::_random::Random::genrand32", addr));
            entries.push(("module::_random::Random::genrand32", addr));
            entries.push(("pyre_module::module::_random::Random::genrand32", addr));
        }
    }
    // `gc.collect`'s finalizer drain, residual for the reason given at
    // `module::gc::interp_gc::run_finalizers_now`.
    {
        let addr = module::gc::interp_gc::run_finalizers_now as *const () as usize as i64;
        if addr != 0 {
            entries.push((
                "pyre_interpreter::module::gc::interp_gc::run_finalizers_now",
                addr,
            ));
            entries.push((
                "pyre_module::module::gc::interp_gc::run_finalizers_now",
                addr,
            ));
        }
    }
    #[cfg(all(
        not(target_arch = "wasm32"),
        feature = "host_env",
        not(feature = "sandbox")
    ))]
    {
        let mmap_type: fn() -> pyre_object::PyObjectRef = module::mmap::interp_mmap::mmap_type;
        let addr = mmap_type as *const () as usize as i64;
        if addr != 0 {
            entries.push((
                "pyre_interpreter::module::mmap::interp_mmap::mmap_type",
                addr,
            ));
            entries.push(("pyre_interpreter::mmap_type", addr));
            entries.push(("pyre_module::module::mmap::interp_mmap::mmap_type", addr));
        }
    }
    #[cfg(all(
        feature = "host_env",
        not(feature = "sandbox"),
        not(target_arch = "wasm32")
    ))]
    {
        use majit_rlib::jit_libffi;
        use module::_cffi_backend::{cdataobj, ctypefunc, ctypeprim, misc};
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::cdataobj::raw_malloc_varsize_char",
            cdataobj::raw_malloc_varsize_char as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::cdataobj::raw_malloc_varsize_char",
            cdataobj::raw_malloc_varsize_char as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::cdataobj::raw_malloc_varsize_zero",
            cdataobj::raw_malloc_varsize_zero as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::cdataobj::raw_malloc_varsize_zero",
            cdataobj::raw_malloc_varsize_zero as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::cdataobj::raw_free",
            cdataobj::raw_free as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::cdataobj::raw_free",
            cdataobj::raw_free as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::ctypefunc::get_mustfree_flag",
            ctypefunc::get_mustfree_flag as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::ctypefunc::get_mustfree_flag",
            ctypefunc::get_mustfree_flag as *const (),
        );
        // `jit_libffi.py` lives in `majit_rlib`; Charon records both the
        // `imp` bodies and the `pub use imp::*` re-exports.
        macro_rules! libffi_fnaddr {
            ($name:literal, $fn:expr) => {{
                let f = $fn as *const ();
                single(entries, concat!("majit_rlib::jit_libffi::imp::", $name), f);
                single(entries, concat!("majit_rlib::jit_libffi::", $name), f);
            }};
        }
        libffi_fnaddr!("exchange_size", jit_libffi::exchange_size);
        libffi_fnaddr!("exchange_result", jit_libffi::exchange_result);
        libffi_fnaddr!("exchange_arg", jit_libffi::exchange_arg);
        libffi_fnaddr!("rtype", jit_libffi::rtype);
        libffi_fnaddr!("nargs", jit_libffi::nargs);
        libffi_fnaddr!("types::getkind", jit_libffi::types::getkind);
        libffi_fnaddr!("types::getsize", jit_libffi::types::getsize);
        libffi_fnaddr!("jit_ffi_call_impl_int", jit_libffi::jit_ffi_call_impl_int);
        libffi_fnaddr!(
            "jit_ffi_call_impl_float",
            jit_libffi::jit_ffi_call_impl_float
        );
        libffi_fnaddr!(
            "jit_ffi_call_impl_longlong",
            jit_libffi::jit_ffi_call_impl_longlong
        );
        libffi_fnaddr!(
            "jit_ffi_call_impl_singlefloat",
            jit_libffi::jit_ffi_call_impl_singlefloat
        );
        libffi_fnaddr!(
            "do_ffi_call_singlefloat",
            jit_libffi::do_ffi_call_singlefloat
        );
        libffi_fnaddr!("jit_ffi_call_impl_void", jit_libffi::jit_ffi_call_impl_void);
        libffi_fnaddr!("jit_ffi_call_impl_any", jit_libffi::jit_ffi_call_impl_any);
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::cdataobj::raw_ptradd",
            cdataobj::raw_ptradd as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::cdataobj::raw_ptradd",
            cdataobj::raw_ptradd as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::cdataobj::raw_read_ptr",
            cdataobj::raw_read_ptr as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::cdataobj::raw_read_ptr",
            cdataobj::raw_read_ptr as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_read_i8",
            misc::raw_read_i8 as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::misc::raw_read_i8",
            misc::raw_read_i8 as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_read_u8",
            misc::raw_read_u8 as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::misc::raw_read_u8",
            misc::raw_read_u8 as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_write_i8",
            misc::raw_write_i8 as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::misc::raw_write_i8",
            misc::raw_write_i8 as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_write_u8",
            misc::raw_write_u8 as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::misc::raw_write_u8",
            misc::raw_write_u8 as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_read_i16",
            misc::raw_read_i16 as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::misc::raw_read_i16",
            misc::raw_read_i16 as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_read_u16",
            misc::raw_read_u16 as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::misc::raw_read_u16",
            misc::raw_read_u16 as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_write_i16",
            misc::raw_write_i16 as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::misc::raw_write_i16",
            misc::raw_write_i16 as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_write_u16",
            misc::raw_write_u16 as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::misc::raw_write_u16",
            misc::raw_write_u16 as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_read_i32",
            misc::raw_read_i32 as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::misc::raw_read_i32",
            misc::raw_read_i32 as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_read_u32",
            misc::raw_read_u32 as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::misc::raw_read_u32",
            misc::raw_read_u32 as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_write_i32",
            misc::raw_write_i32 as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::misc::raw_write_i32",
            misc::raw_write_i32 as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_write_u32",
            misc::raw_write_u32 as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::misc::raw_write_u32",
            misc::raw_write_u32 as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_read_i64",
            misc::raw_read_i64 as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::misc::raw_read_i64",
            misc::raw_read_i64 as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_read_u64",
            misc::raw_read_u64 as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::misc::raw_read_u64",
            misc::raw_read_u64 as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_write_i64",
            misc::raw_write_i64 as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::misc::raw_write_i64",
            misc::raw_write_i64 as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_write_u64",
            misc::raw_write_u64 as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::misc::raw_write_u64",
            misc::raw_write_u64 as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_read_f32",
            misc::raw_read_f32 as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::misc::raw_read_f32",
            misc::raw_read_f32 as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_write_f32",
            misc::raw_write_f32 as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::misc::raw_write_f32",
            misc::raw_write_f32 as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_read_f64",
            misc::raw_read_f64 as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::misc::raw_read_f64",
            misc::raw_read_f64 as *const (),
        );
        single(
            entries,
            "pyre_interpreter::module::_cffi_backend::misc::raw_write_f64",
            misc::raw_write_f64 as *const (),
        );
        single(
            entries,
            "pyre_module::module::_cffi_backend::misc::raw_write_f64",
            misc::raw_write_f64 as *const (),
        );
        // `@jit.dont_look_inside` leaves of the `_CDataBase` descent, and the
        // `raw_write_ptr` oopspec leaf.
        for (interp, module, fnptr) in [
            (
                "pyre_interpreter::module::_cffi_backend::ctypeprim::copy_longdouble",
                "pyre_module::module::_cffi_backend::ctypeprim::copy_longdouble",
                ctypeprim::copy_longdouble as *const (),
            ),
            (
                "pyre_interpreter::module::_cffi_backend::misc::raw_memcopy_opaque",
                "pyre_module::module::_cffi_backend::misc::raw_memcopy_opaque",
                misc::raw_memcopy_opaque as *const (),
            ),
            (
                "pyre_interpreter::module::_cffi_backend::cdataobj::raw_write_ptr",
                "pyre_module::module::_cffi_backend::cdataobj::raw_write_ptr",
                cdataobj::raw_write_ptr as *const (),
            ),
        ] {
            single(entries, interp, fnptr);
            single(entries, module, fnptr);
        }
    }
    #[cfg(all(any(unix, windows), feature = "host_env", not(feature = "sandbox")))]
    {
        let f = module::_ctypes::cdata::cdata_bytes_object as *const ();
        single(
            entries,
            "pyre_interpreter::module::_ctypes::cdata::cdata_bytes_object",
            f,
        );
        single(entries, "pyre_interpreter::cdata_bytes_object", f);
        single(
            entries,
            "pyre_module::module::_ctypes::cdata::cdata_bytes_object",
            f,
        );
    }
}

fn walk_optional_global_roots(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    module::gc::hook::walk_hook_roots(visitor);
    #[cfg(all(any(unix, windows), feature = "host_env", not(feature = "sandbox")))]
    module::_ctypes::cdata::walk_pyobj_container_roots(visitor);
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    /// `W_CTypePrimitiveLongDouble._copy_longdouble` is
    /// `@jit.dont_look_inside`. The helper moved here with `_cffi_backend`.
    #[cfg(all(
        not(target_arch = "wasm32"),
        feature = "host_env",
        not(feature = "sandbox")
    ))]
    #[test]
    fn jit_trace_fnaddrs_covers_copy_longdouble() {
        crate::register();
        let bindings: HashMap<&'static str, i64> =
            pyre_interpreter::jit_trace_fnaddrs().into_iter().collect();
        let expected =
            crate::module::_cffi_backend::ctypeprim::copy_longdouble as *const () as usize as i64;
        assert_eq!(
            bindings["pyre_module::module::_cffi_backend::ctypeprim::copy_longdouble"],
            expected
        );
        assert_eq!(
            bindings["pyre_interpreter::module::_cffi_backend::ctypeprim::copy_longdouble"],
            expected
        );
    }

    /// `misc.py _raw_memcopy_opaque` is `@jit.dont_look_inside`.
    #[cfg(all(
        not(target_arch = "wasm32"),
        feature = "host_env",
        not(feature = "sandbox")
    ))]
    #[test]
    fn jit_trace_fnaddrs_covers_raw_memcopy_opaque() {
        crate::register();
        let bindings: HashMap<&'static str, i64> =
            pyre_interpreter::jit_trace_fnaddrs().into_iter().collect();
        let expected =
            crate::module::_cffi_backend::misc::raw_memcopy_opaque as *const () as usize as i64;
        assert_eq!(
            bindings["pyre_module::module::_cffi_backend::misc::raw_memcopy_opaque"],
            expected
        );
        assert_eq!(
            bindings["pyre_interpreter::module::_cffi_backend::misc::raw_memcopy_opaque"],
            expected
        );
    }

    /// `rffi.cast(rffi.CCHARPP, data)[0] = value` is the `raw_write_ptr` oopspec.
    #[cfg(all(
        not(target_arch = "wasm32"),
        feature = "host_env",
        not(feature = "sandbox")
    ))]
    #[test]
    fn jit_trace_fnaddrs_covers_raw_write_ptr() {
        crate::register();
        let bindings: HashMap<&'static str, i64> =
            pyre_interpreter::jit_trace_fnaddrs().into_iter().collect();
        let expected =
            crate::module::_cffi_backend::cdataobj::raw_write_ptr as *const () as usize as i64;
        assert_eq!(
            bindings["pyre_module::module::_cffi_backend::cdataobj::raw_write_ptr"],
            expected
        );
        assert_eq!(
            bindings["pyre_interpreter::module::_cffi_backend::cdataobj::raw_write_ptr"],
            expected
        );
    }

    /// `_csv.Dialect`'s `type_object` accessor is `dont_look_inside`. It must
    /// publish a residual address, or a traced type lookup residualizes to a
    /// symbolic fnaddr and inline JIT descent aborts.
    #[test]
    fn jit_trace_fnaddrs_covers_csv_dialect_type_object() {
        let bindings: HashMap<&'static str, i64> =
            pyre_interpreter::jit_trace_fnaddrs().into_iter().collect();
        assert!(
            bindings.contains_key("pyre_module::module::_csv::type_object"),
            "_csv.Dialect type_object must publish a residual fnaddr",
        );
        assert!(
            bindings.contains_key("module::_csv::type_object"),
            "the crate-stripped alias must resolve too",
        );
    }

    /// `#[pyre_methods]` wrappers in this crate must appear in the
    /// process-global fnaddr table. The prepass reads the same table
    /// from a host copy of this crate; a missing row here is the same
    /// defect as a build script that forgot to link `pyre-module`.
    #[test]
    fn jit_trace_fnaddrs_covers_moved_hashlib_wrapper() {
        let bindings: HashMap<&'static str, i64> =
            pyre_interpreter::jit_trace_fnaddrs().into_iter().collect();
        assert!(
            bindings.contains_key(
                "pyre_module::module::_hashlib::hash_state_class::__majit_wrap___new__"
            ),
            "moved _hashlib #[pyre_methods] wrappers must publish residual fnaddrs",
        );
    }

    #[cfg(all(
        not(target_arch = "wasm32"),
        feature = "host_env",
        not(feature = "sandbox")
    ))]
    #[test]
    fn jit_trace_fnaddrs_covers_moved_mmap_type() {
        let bindings: HashMap<&'static str, i64> =
            pyre_interpreter::jit_trace_fnaddrs().into_iter().collect();
        assert!(
            bindings.contains_key("pyre_module::module::mmap::interp_mmap::mmap_type"),
            "moved mmap_type must publish a residual fnaddr",
        );
    }

    #[cfg(all(unix, not(feature = "sandbox")))]
    #[test]
    fn jit_trace_fnaddrs_covers_moved_select_wrapper() {
        let bindings: HashMap<&'static str, i64> =
            pyre_interpreter::jit_trace_fnaddrs().into_iter().collect();
        assert!(
            bindings
                .contains_key("pyre_module::module::select::interp_select::__majit_wrap_register"),
            "moved select #[pyre_methods] wrappers must publish residual fnaddrs",
        );
    }

    #[test]
    fn jit_trace_fnaddrs_covers_moved_lzma_wrapper() {
        let bindings: HashMap<&'static str, i64> =
            pyre_interpreter::jit_trace_fnaddrs().into_iter().collect();
        assert!(
            bindings.contains_key(
                "pyre_module::module::_lzma::compressor_methods::__majit_wrap___new__"
            ),
            "moved _lzma #[pyre_methods] wrappers must publish residual fnaddrs",
        );
    }

    #[test]
    fn jit_trace_fnaddrs_covers_moved_unicodedata_wrapper() {
        let bindings: HashMap<&'static str, i64> =
            pyre_interpreter::jit_trace_fnaddrs().into_iter().collect();
        assert!(
            bindings.contains_key(
                "pyre_module::module::unicodedata::interp_ucd::__majit_wrap_category"
            ),
            "moved unicodedata #[pyre_methods] wrappers must publish residual fnaddrs",
        );
    }

    #[test]
    fn jit_trace_fnaddrs_covers_moved_bz2_wrapper() {
        let bindings: HashMap<&'static str, i64> =
            pyre_interpreter::jit_trace_fnaddrs().into_iter().collect();
        assert!(
            bindings
                .contains_key("pyre_module::module::bz2::compressor_methods::__majit_wrap___new__"),
            "moved _bz2 #[pyre_methods] wrappers must publish residual fnaddrs",
        );
    }

    #[cfg(all(not(target_arch = "wasm32"), not(feature = "sandbox")))]
    #[test]
    fn jit_trace_fnaddrs_covers_moved_ssl_wrapper() {
        let bindings: HashMap<&'static str, i64> =
            pyre_interpreter::jit_trace_fnaddrs().into_iter().collect();
        assert!(
            bindings.contains_key(
                "pyre_module::module::_ssl::ssl_session_methods::__majit_wrap___new__"
            ),
            "moved _ssl #[pyre_methods] wrappers must publish residual fnaddrs",
        );
    }

    #[test]
    fn jit_trace_fnaddrs_covers_moved_lsprof_wrapper() {
        let bindings: HashMap<&'static str, i64> =
            pyre_interpreter::jit_trace_fnaddrs().into_iter().collect();
        assert!(
            bindings.contains_key(
                "pyre_module::module::_lsprof::profiler_methods::__majit_wrap___new__"
            ),
            "moved _lsprof #[pyre_methods] wrappers must publish residual fnaddrs",
        );
    }

    /// `Random.genrand32` holds the Mersenne Twister refill loops that
    /// `JitPolicy.look_inside_graph` rejects, so the codewriter needs its real
    /// address for the residual call. Every resolver spelling binds the helper.
    #[test]
    fn jit_trace_fnaddrs_covers_moved_random_genrand32() {
        crate::register();
        let bindings: HashMap<&'static str, i64> =
            pyre_interpreter::jit_trace_fnaddrs().into_iter().collect();
        let genrand32: fn(&mut crate::module::_random::Random) -> u32 =
            crate::module::_random::Random::genrand32;
        let expected = genrand32 as *const () as usize as i64;

        assert_eq!(
            bindings["pyre_module::module::_random::Random::genrand32"],
            expected
        );
        assert_eq!(
            bindings["pyre_interpreter::module::_random::Random::genrand32"],
            expected
        );
        assert_eq!(bindings["module::_random::Random::genrand32"], expected);
    }

    /// `gc.collect`'s drain. The interpreter reaches the same body through its
    /// own `dont_look_inside` wrapper, which publishes separately; this module's
    /// two spellings bind the body itself.
    #[test]
    fn jit_trace_fnaddrs_covers_moved_run_finalizers_now() {
        crate::register();
        let bindings: HashMap<&'static str, i64> =
            pyre_interpreter::jit_trace_fnaddrs().into_iter().collect();
        let expected =
            crate::module::gc::interp_gc::run_finalizers_now as *const () as usize as i64;

        assert_eq!(
            bindings["pyre_module::module::gc::interp_gc::run_finalizers_now"],
            expected
        );
        assert_eq!(
            bindings["pyre_interpreter::module::gc::interp_gc::run_finalizers_now"],
            expected
        );
    }
}
