//! Module definitions and method-table conversion -- PyPy
//! `cpyext/modsupport.py`.

use super::methodobject::{CPyMethodDef, MethFlags, new_pycfunction};
use super::pyerrors::{self, trap};
use super::pyobject::{self, CPyObject, REFCNT_IMMORTAL};
use pyre_object::{PY_NULL, PyObjectRef};
use std::ffi::{CStr, c_char, c_int, c_long, c_void};
use std::hash::BuildHasherDefault;
use std::sync::atomic::{AtomicIsize, Ordering};

/// The `slots_generated.h` identifiers a `PyModuleDef_Slot` array carries.
///
/// A slot names one identifier rather than a set of bits, so these stay
/// numbers.  Declaring each once mints the table
/// `every_module_slot_is_the_number_the_header_gives_it` walks, so a slot
/// added here is compared against the header without anyone remembering to
/// list it.  The values are the non-limited numbering (`_Py_SLOT_COMPAT_VALUE`
/// expands to NEW).
macro_rules! module_slots {
    ($($name:ident = $value:expr,)*) => {
        $(const $name: c_int = $value;)*

        #[cfg(test)]
        const ALL_MODULE_SLOTS: &[(&str, c_int)] = &[$((stringify!($name), $value),)*];
    };
}

module_slots! {
    PY_MOD_CREATE = 84,
    PY_MOD_EXEC = 85,
    PY_MOD_MULTIPLE_INTERPRETERS = 86,
    PY_MOD_GIL = 87,
    PY_MOD_ABI = 109,
    PY_MOD_TOKEN = 110,
}

/// Map a `PyModuleDef_Slot.slot` id onto the current numbering.
///
/// `_PySlot_resolve_mod_slot`: a pre-3.15 id 1..4 becomes the matching
/// `Py_mod_*`; an id that is already a current module slot is kept; anything
/// else is invalid.
fn resolve_mod_slot(id: c_int) -> Option<c_int> {
    match id {
        1 => Some(PY_MOD_CREATE),
        2 => Some(PY_MOD_EXEC),
        3 => Some(PY_MOD_MULTIPLE_INTERPRETERS),
        4 => Some(PY_MOD_GIL),
        PY_MOD_CREATE
        | PY_MOD_EXEC
        | PY_MOD_MULTIPLE_INTERPRETERS
        | PY_MOD_GIL
        | PY_MOD_ABI
        | PY_MOD_TOKEN => Some(id),
        _ => None,
    }
}

/// The two versions a module may declare: the full API version an extension
/// built against `Python.h` carries, and the stable-ABI one a limited-API
/// build carries instead.  Both must match `Python.h`.
const PYTHON_API_VERSION: c_int = 1013;
const PYTHON_ABI_VERSION: c_int = 3;

/// The `md_def`, `md_state` and token CPython keeps in the module object's
/// own C struct (`pycore_moduleobject.h` `PyModuleObject`).
///
/// Pyre's module has no such fields.  The module dictionary is not a place to
/// put them: it is the module's Python-visible namespace, so `vars(mod)` and
/// `mod.__dict__[...] = 0` both reach it, and `PyModule_GetState` would then
/// hand a C extension an address of the writer's choosing to dereference.  A
/// mirror is the C-side half of the module, never moves, and dies with the
/// module, so the pair is filed under the mirror's address instead.
#[derive(Clone, Copy, Default)]
struct ModuleFields {
    md_def: usize,
    md_state: usize,
    md_token: usize,
    md_token_is_def: bool,
}

type ModuleTable = super::address_table::HeldMap<ModuleFields>;
use super::address_table::{AddressTable, hold};

static MODULE_FIELDS: AddressTable<ModuleTable> = AddressTable::new(
    std::collections::HashMap::with_hasher(BuildHasherDefault::new()),
);

pub(super) unsafe fn after_fork_child() {
    unsafe { MODULE_FIELDS.reinit_after_fork() };
    unsafe { MODULES_BY_INDEX.reinit_after_fork() };
}

/// The pair a module has recorded, all zero for one that has recorded none.
///
/// The read never builds a mirror: a module with none has nothing filed.
fn fields(module: PyObjectRef) -> ModuleFields {
    let mirror = pyobject::as_pyobj(module) as usize;
    if mirror == 0 {
        return ModuleFields::default();
    }
    MODULE_FIELDS
        .lock()
        .get(&mirror)
        .copied()
        .unwrap_or_default()
}

/// Record one of the pair, building the module's mirror if it has none.
///
/// # Safety
/// The caller must be holding `module` rooted, as [`pyobject::borrow_mirror`]
/// requires.
fn set_field(module: PyObjectRef, update: impl FnOnce(&mut ModuleFields)) {
    // Outside the lock: building a mirror takes the census lock of its own.
    let mirror = pyobject::borrow_mirror(module) as usize;
    if mirror == 0 {
        return;
    }
    update(MODULE_FIELDS.lock().entry(hold(mirror)).or_default());
}

/// Drop what a dying module mirror recorded.
///
/// The state block itself is not freed: pyre has no module deallocation path,
/// and an extension may still hold the address.
pub(super) fn forget_module_fields(mirror: usize) {
    MODULE_FIELDS.take(mirror);
}

/// The single-phase modules `PyState_AddModule` has filed, one slot per module
/// index (`MODULES_BY_INDEX` in `import.c`).  Slot 0 is never used, because a
/// def carries index 0 until `PyModuleDef_Init` stamps it; an empty slot is
/// upstream's `None`.
///
/// A slot owns its reference, as upstream's list does, so a module filed here
/// outlives every other name for it.
static MODULES_BY_INDEX: super::ForkMutex<Vec<usize>> = super::ForkMutex::new(Vec::new());

/// The index a def is filed under (`_get_module_index_from_def`).
unsafe fn module_index(def: *mut CPyModuleDef) -> isize {
    unsafe { (*def).m_base.m_index }
}

/// `_modules_by_index_check` — why an index cannot be read or cleared, or
/// `None` when it can.
fn modules_by_index_check(table: &[usize], index: isize) -> Option<&'static str> {
    if index <= 0 {
        return Some("invalid module index");
    }
    if index as usize >= table.len() {
        return Some("Module index out of bounds.");
    }
    None
}

/// `PyState_FindModule(def)` — the module filed under a def's index, borrowed.
///
/// A def with slots is multi-phase and files nothing, so it answers NULL
/// rather than reading a slot another def could own.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyState_FindModule(def: *mut CPyModuleDef) -> *mut CPyObject {
    if def.is_null() {
        return std::ptr::null_mut();
    }
    if unsafe { !(*def).m_slots.is_null() } {
        return std::ptr::null_mut();
    }
    let index = unsafe { module_index(def) };
    let table = MODULES_BY_INDEX.lock();
    if modules_by_index_check(&table, index).is_some() {
        return std::ptr::null_mut();
    }
    table[index as usize] as *mut CPyObject
}

/// `PyState_AddModule(module, def)` — file a single-phase module under its
/// def's index.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyState_AddModule(
    module: *mut CPyObject,
    def: *mut CPyModuleDef,
) -> c_int {
    if def.is_null() {
        super::pyerrors::fatal_error(None, "module definition is NULL");
    }
    if unsafe { !(*def).m_slots.is_null() } {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "PyState_AddModule called on module with slots",
        ));
        return -1;
    }
    let index = unsafe { module_index(def) };
    let mut table = MODULES_BY_INDEX.lock();
    if index > 0 && (index as usize) < table.len() && table[index as usize] == module as usize {
        super::pyerrors::fatal_error(
            Some("PyState_AddModule"),
            &format!("module {module:p} already added"),
        );
    }
    // `_modules_by_index_set` asserts a positive index rather than refusing
    // one: a def reaching here with 0 has skipped `PyModuleDef_Init`, and the
    // slot it then writes is the one every reader rejects.
    debug_assert!(index > 0, "PyState_AddModule on an uninitialized def");
    if table.len() <= index as usize {
        table.resize(index as usize + 1, 0);
    }
    let previous = std::mem::replace(&mut table[index as usize], module as usize);
    drop(table);
    unsafe { super::pyobject::incref(module) };
    if previous != 0 {
        unsafe { super::pyobject::decref(previous as *mut CPyObject) };
    }
    0
}

/// `PyState_RemoveModule(def)` — empty the slot a def's index owns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyState_RemoveModule(def: *mut CPyModuleDef) -> c_int {
    if def.is_null() {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return -1;
    }
    if unsafe { !(*def).m_slots.is_null() } {
        super::pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "PyState_RemoveModule called on module with slots",
        ));
        return -1;
    }
    let index = unsafe { module_index(def) };
    let mut table = MODULES_BY_INDEX.lock();
    if let Some(err) = modules_by_index_check(&table, index) {
        drop(table);
        super::pyerrors::fatal_error(None, err);
    }
    let previous = std::mem::replace(&mut table[index as usize], 0);
    drop(table);
    if previous != 0 {
        unsafe { super::pyobject::decref(previous as *mut CPyObject) };
    }
    0
}

static NEXT_MODULE_INDEX: AtomicIsize = AtomicIsize::new(1);

#[repr(C)]
pub struct CPyModuleDefBase {
    pub ob_base: CPyObject,
    pub m_init: Option<unsafe extern "C" fn() -> *mut CPyObject>,
    pub m_index: isize,
    pub m_copy: *mut CPyObject,
}

#[repr(C)]
pub struct CPyModuleDefSlot {
    pub slot: c_int,
    pub value: *mut c_void,
}

#[repr(C)]
pub struct CPyModuleDef {
    pub m_base: CPyModuleDefBase,
    pub m_name: *const c_char,
    pub m_doc: *const c_char,
    pub m_size: isize,
    pub m_methods: *mut CPyMethodDef,
    pub m_slots: *mut CPyModuleDefSlot,
    pub m_traverse: *const c_void,
    pub m_clear: *const c_void,
    pub m_free: *const c_void,
}

/// `PyABIInfo` — `modsupport.h`.
#[repr(C)]
pub struct CPyABIInfo {
    pub abiinfo_major_version: u8,
    pub abiinfo_minor_version: u8,
    pub flags: u16,
    pub build_version: u32,
    pub abi_version: u32,
}

/// Packed the same way `patchlevel.h` `_Py_PACK_FULL_VERSION` packs
/// `PY_VERSION_HEX`.
const fn pack_full_version(x: u32, y: u32, z: u32, level: u32, serial: u32) -> u32 {
    ((x & 0xff) << 24)
        | ((y & 0xff) << 16)
        | ((z & 0xff) << 8)
        | ((level & 0xf) << 4)
        | (serial & 0xf)
}

const PY_VERSION_HEX: u32 = pack_full_version(3, 15, 0, 0xf, 0);
const PY_PACK_VERSION_3_2: u32 = pack_full_version(3, 2, 0, 0, 0);
const XY_MASK: u32 = 0xffff0000;

const PYABIINFO_STABLE: u16 = 0x0001;
const PYABIINFO_GIL: u16 = 0x0002;
const PYABIINFO_FREETHREADED: u16 = 0x0004;
const PYABIINFO_INTERNAL: u16 = 0x0008;

/// `_abiinfo_raise`: ImportError `"{name}: {msg}"` when a name is given.
fn abiinfo_raise(module_name: Option<&str>, msg: String) -> crate::PyError {
    let text = match module_name {
        Some(name) => format!("{name}: {msg}"),
        None => msg,
    };
    crate::PyError::new(crate::PyErrorKind::ImportError, text)
}

fn abi_xy(version: u32) -> String {
    format!("{}.{}", (version >> 24) & 0xff, (version >> 16) & 0xff)
}

/// `Python/modsupport.c` `PyABIInfo_Check`.  pyre's `pyconfig.h` does not
/// define `Py_GIL_DISABLED`, so the GIL-build branch applies.
fn abiinfo_check(info: *mut CPyABIInfo, module_name: Option<&str>) -> Result<(), crate::PyError> {
    let raise = |msg: String| Err(abiinfo_raise(module_name, msg));
    if info.is_null() {
        return raise("NULL PyABIInfo".to_owned());
    }
    let info = unsafe { &*info };
    if info.abiinfo_major_version == 0 {
        return Ok(());
    }
    if info.abiinfo_major_version > 1 {
        return raise("PyABIInfo version too high".to_owned());
    }
    if info.flags & PYABIINFO_INTERNAL != 0
        && info.abi_version != 0
        && info.abi_version != PY_VERSION_HEX
    {
        return raise(format!(
            "incompatible internal ABI (0x{:x} != 0x{:x})",
            info.abi_version, PY_VERSION_HEX
        ));
    }
    if info.flags & PYABIINFO_STABLE != 0 {
        if info.abi_version != 0 {
            if (info.abi_version & XY_MASK) > (PY_VERSION_HEX & XY_MASK) {
                return raise(format!(
                    "incompatible future stable ABI version ({})",
                    abi_xy(info.abi_version)
                ));
            }
            if info.abi_version < PY_PACK_VERSION_3_2 {
                return raise(format!(
                    "invalid stable ABI version ({})",
                    abi_xy(info.abi_version)
                ));
            }
        }
        if info.flags & PYABIINFO_INTERNAL != 0 {
            return raise("cannot use both internal and stable ABI".to_owned());
        }
    } else if info.abi_version != 0 && (info.abi_version & XY_MASK) != (PY_VERSION_HEX & XY_MASK) {
        return raise(format!(
            "incompatible ABI version ({})",
            abi_xy(info.abi_version)
        ));
    }
    let gilflags = info.flags & (PYABIINFO_GIL | PYABIINFO_FREETHREADED);
    if gilflags == PYABIINFO_FREETHREADED {
        return raise("only compatible with free-threaded CPython".to_owned());
    }
    Ok(())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyABIInfo_Check(
    info: *mut CPyABIInfo,
    module_name: *const c_char,
) -> c_int {
    let owned = if module_name.is_null() {
        None
    } else {
        Some(
            unsafe { CStr::from_ptr(module_name) }
                .to_string_lossy()
                .into_owned(),
        )
    };
    match abiinfo_check(info, owned.as_deref()) {
        Ok(()) => 0,
        Err(err) => {
            pyerrors::set_pending_error(err);
            -1
        }
    }
}

fn text_or_empty(pointer: *const c_char) -> String {
    if pointer.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(pointer) }
            .to_string_lossy()
            .into_owned()
    }
}

fn module_dict(module: PyObjectRef) -> PyObjectRef {
    unsafe { pyre_object::w_module_get_w_dict(module) }
}

fn store(module: PyObjectRef, name: &str, value: PyObjectRef) {
    crate::module_ns_store(module_dict(module), name, value);
}

/// `_add_methods_to_object` / `modsupport.py:157` — the same store against
/// whatever a create slot answered.
///
/// Only a module has the proxy-free dict [`store`] writes straight through;
/// anything else takes the ordinary attribute path, which is what upstream
/// uses for both.
fn store_on_object(
    target: PyObjectRef,
    name: &str,
    value: PyObjectRef,
) -> Result<(), crate::PyError> {
    if unsafe { pyre_object::module::is_module(target) } {
        store(target, name, value);
        return Ok(());
    }
    crate::baseobjspace::setattr_str(target, name, value).map(drop)
}

/// `modsupport.py:convert_method_defs`, module branch.
///
/// The table is NUL-name terminated.  `METH_CLASS` / `METH_STATIC` are
/// rejected here exactly as upstream rejects them for a module-level table;
/// their type-level meaning arrives with C-defined types.
///
/// `module` is whatever the create slot answered, so it need not be one.
fn convert_method_defs(
    module: PyObjectRef,
    methods: *mut CPyMethodDef,
    w_module_name: PyObjectRef,
) -> Result<(), crate::PyError> {
    if methods.is_null() {
        return Ok(());
    }
    let roots = pyre_object::gc_roots::push_roots();
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    let module = roots.pin_root(module);
    let name_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_module_name);
    let mut index = 0isize;
    loop {
        let method = unsafe { methods.offset(index) };
        if unsafe { (*method).ml_name.is_null() } {
            return Ok(());
        }
        index += 1;
        let name = text_or_empty(unsafe { (*method).ml_name });
        if unsafe { (*method).ml_flags }.intersects(MethFlags::METH_CLASS | MethFlags::METH_STATIC)
        {
            return Err(crate::PyError::value_error(
                "module functions cannot set METH_CLASS or METH_STATIC",
            ));
        }
        let function = new_pycfunction(
            method,
            pyre_object::gc_roots::shadow_stack_get(module_slot),
            pyre_object::gc_roots::shadow_stack_get(name_slot),
        )?;
        let function_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(function);
        store_on_object(
            pyre_object::gc_roots::shadow_stack_get(module_slot),
            &name,
            pyre_object::gc_roots::shadow_stack_get(function_slot),
        )?;
    }
}

/// Allocate the zero-filled per-module block `m_size` asks for.
///
/// A zero `m_size` still gets a block: [`exec_def`] uses the address as its
/// "already executed" marker, and address 0 cannot say that.
///
/// Never released: pyre has no module deallocation path yet, so the block
/// lives as long as the process — the same lifetime upstream's `md_state` has
/// in practice, since `module_dealloc` only runs when the module dies.
fn allocate_module_state(module: PyObjectRef, size: isize) -> Result<(), crate::PyError> {
    let block = unsafe { libc::calloc(1, size.max(1) as usize) };
    if block.is_null() {
        return Err(crate::PyError::new(
            crate::PyErrorKind::MemoryError,
            "cannot allocate cpyext module state",
        ));
    }
    set_field(module, |fields| fields.md_state = block as usize);
    Ok(())
}

/// Fill a fresh module from `def`: definition address, file, methods and doc.
///
/// The state block is not allocated here — `PyModule_Create2` allocates it for
/// a single-phase module and [`exec_def`] for a multi-phase one.
fn populate_module(
    module: PyObjectRef,
    def: *mut CPyModuleDef,
    name: &str,
    path: Option<&std::path::Path>,
) -> Result<PyObjectRef, crate::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(module);
    let reload = |slot| pyre_object::gc_roots::shadow_stack_get(slot);

    set_field(reload(module_slot), |fields| {
        fields.md_def = def as usize;
        fields.md_token = def as usize;
        fields.md_token_is_def = true;
    });

    if let Some(path) = path {
        let value = crate::gateway::fsdecode_os_str(path.as_os_str());
        let value_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(value);
        store(reload(module_slot), "__file__", reload(value_slot));
    }

    add_methods_and_doc(reload(module_slot), def, name)
}

/// The tail of `moduleobject.c PyModule_FromDefAndSpec2`: `m_methods` and
/// `m_doc` go on whatever the create slot answered, module or not.
///
/// `_testmultiphase`'s `nonmodule_with_methods` answers a `SimpleNamespace`
/// and still declares a method table, which is the case that separates this
/// from the module-only fields [`populate_module`] sets.
fn add_methods_and_doc(
    module: PyObjectRef,
    def: *mut CPyModuleDef,
    name: &str,
) -> Result<PyObjectRef, crate::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(module);
    let reload = |slot| pyre_object::gc_roots::shadow_stack_get(slot);

    let w_name = pyre_object::w_str_new(name);
    let name_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_name);
    convert_method_defs(
        reload(module_slot),
        unsafe { (*def).m_methods },
        reload(name_slot),
    )?;

    let doc = text_or_empty(unsafe { (*def).m_doc });
    if !doc.is_empty() {
        let value = pyre_object::w_str_new(&doc);
        let value_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(value);
        store_on_object(reload(module_slot), "__doc__", reload(value_slot))?;
    }

    Ok(reload(module_slot))
}

/// `PyModuleDef_Init` -- also the marker returned by PEP 489 init functions.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModuleDef_Init(def: *mut CPyModuleDef) -> *mut CPyObject {
    if def.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        if (*def).m_base.m_index == 0 {
            (*def).m_base.ob_base.ob_refcnt = REFCNT_IMMORTAL;
            (*def).m_base.ob_base.ob_pyre_link = PY_NULL;
            (*def).m_base.ob_base.ob_type = &raw mut super::typeobject::CPY_MODULE_DEF_TYPE;
            (*def).m_base.m_index = NEXT_MODULE_INDEX.fetch_add(1, Ordering::Relaxed);
        }
        &mut (*def).m_base.ob_base
    }
}

/// # Safety
/// `raw` must be a mirror [`is_module_def`] accepts.
pub(super) unsafe fn module_def_of(raw: *mut CPyObject) -> *mut CPyModuleDef {
    raw as *mut CPyModuleDef
}

/// `PyModule_Create2` -- PyPy `modsupport.py:PyModule_Create2`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_Create2(
    def: *mut CPyModuleDef,
    _api_version: c_int,
) -> *mut CPyObject {
    if def.is_null() || unsafe { (*def).m_name.is_null() } {
        unsafe { pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    if unsafe { !(*def).m_slots.is_null() } {
        pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            format!(
                "module {} has multi-phase initialization slots, \
                 but PyModule_Create was called",
                text_or_empty(unsafe { (*def).m_name })
            ),
        ));
        return std::ptr::null_mut();
    }

    let declared_name = text_or_empty(unsafe { (*def).m_name });
    // PyPy `PyModule_Create2` consumes package_context before allocating the
    // module. Releasing the mutex here also avoids holding it across GC.
    let context = super::PACKAGE_CONTEXT.lock().take();
    let (name, path) = context
        .as_ref()
        .map(|(name, path)| (name.as_str(), Some(path.as_path())))
        .unwrap_or((declared_name.as_str(), None));
    let module = pyre_object::w_module_new_managed(name);
    let roots = pyre_object::gc_roots::push_roots();
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    let module = roots.pin_root(module);
    let Some(module) = trap(populate_module(
        pyre_object::gc_roots::shadow_stack_get(module_slot),
        def,
        name,
        path,
    )) else {
        return std::ptr::null_mut();
    };
    if unsafe { (*def).m_size } > 0 {
        let module_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(module);
        if trap(allocate_module_state(
            pyre_object::gc_roots::shadow_stack_get(module_slot),
            unsafe { (*def).m_size },
        ))
        .is_none()
        {
            return std::ptr::null_mut();
        }
        return pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(module_slot));
    }
    pyobject::make_ref(module)
}

/// The `PySlot` identifiers a module export array carries.  The four that a
/// `PyModuleDef_Slot` array also names, plus `Py_mod_abi` / `Py_mod_token`,
/// reuse `PY_MOD_*` so both readers share one numbering.
mod export_id {
    use super::{
        PY_MOD_ABI, PY_MOD_CREATE, PY_MOD_EXEC, PY_MOD_GIL, PY_MOD_MULTIPLE_INTERPRETERS,
        PY_MOD_TOKEN,
    };

    pub const MOD_CREATE: u16 = PY_MOD_CREATE as u16;
    pub const MOD_EXEC: u16 = PY_MOD_EXEC as u16;
    pub const MOD_MULTIPLE_INTERPRETERS: u16 = PY_MOD_MULTIPLE_INTERPRETERS as u16;
    pub const MOD_GIL: u16 = PY_MOD_GIL as u16;
    pub const MOD_NAME: u16 = 100;
    pub const MOD_DOC: u16 = 101;
    pub const MOD_STATE_SIZE: u16 = 102;
    pub const MOD_METHODS: u16 = 103;
    pub const MOD_STATE_TRAVERSE: u16 = 104;
    pub const MOD_STATE_CLEAR: u16 = 105;
    pub const MOD_STATE_FREE: u16 = 106;
    pub const MOD_ABI: u16 = PY_MOD_ABI as u16;
    pub const MOD_TOKEN: u16 = PY_MOD_TOKEN as u16;
}

/// `PyModExport_*`'s answer, read into the definition the create phase takes.
///
/// The export protocol says the same things a `PyModuleDef` says, as an array
/// rather than as a struct: an extension that never names the struct cannot be
/// laid out against the wrong one.  The definition built here is this layer's
/// own, and lives as long as the module does.
pub(super) fn create_module_from_export_slots(
    slots: *mut super::typeobject::CPySlot,
    spec: PyObjectRef,
    name: &str,
    path: Option<&std::path::Path>,
) -> Result<PyObjectRef, crate::PyError> {
    let refuse = |message: String| {
        Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            message,
        ))
    };
    // `import_run_modexport`: a NULL answer keeps the hook's exception, or
    // reports that none was set; a non-NULL answer with an exception set
    // is an unreported raise.
    if slots.is_null() {
        if let Some(pending) = pyerrors::take_pending_error() {
            return Err(pending);
        }
        return refuse(format!(
            "module export hook for module '{name}' failed without setting an exception"
        ));
    }
    if let Some(pending) = pyerrors::take_pending_error() {
        return Err(crate::error::system_error_from_cause(
            format!("module export hook for module '{name}' raised unreported exception"),
            pending,
        ));
    }
    let mut def = Box::new(CPyModuleDef {
        m_base: CPyModuleDefBase {
            ob_base: CPyObject {
                ob_refcnt: 0,
                ob_pyre_link: PY_NULL,
                ob_pyre_pad: 0,
                ob_type: std::ptr::null_mut(),
            },
            m_init: None,
            m_index: 0,
            m_copy: std::ptr::null_mut(),
        },
        m_name: std::ptr::null(),
        m_doc: std::ptr::null(),
        m_size: 0,
        m_methods: std::ptr::null_mut(),
        m_slots: std::ptr::null_mut(),
        m_traverse: std::ptr::null(),
        m_clear: std::ptr::null(),
        m_free: std::ptr::null(),
    });
    let mut def_slots: Vec<CPyModuleDefSlot> = Vec::new();
    let mut saw_abi = false;
    let mut export_token: usize = 0;
    let mut entry = slots;
    loop {
        let slot = unsafe { &*entry };
        if slot.sl_id == 0 {
            break;
        }
        let value = slot.sl_value as *mut c_void;
        match slot.sl_id {
            export_id::MOD_NAME => def.m_name = value as *const c_char,
            export_id::MOD_DOC => def.m_doc = value as *const c_char,
            export_id::MOD_STATE_SIZE => def.m_size = slot.sl_value as isize,
            export_id::MOD_METHODS => def.m_methods = value as *mut CPyMethodDef,
            export_id::MOD_STATE_TRAVERSE => def.m_traverse = value as *const c_void,
            export_id::MOD_STATE_CLEAR => def.m_clear = value as *const c_void,
            export_id::MOD_STATE_FREE => def.m_free = value as *const c_void,
            export_id::MOD_ABI => {
                saw_abi = true;
                abiinfo_check(value as *mut CPyABIInfo, Some(name))?;
            }
            export_id::MOD_TOKEN => export_token = value as usize,
            export_id::MOD_CREATE => def_slots.push(CPyModuleDefSlot {
                slot: PY_MOD_CREATE,
                value,
            }),
            export_id::MOD_EXEC => def_slots.push(CPyModuleDefSlot {
                slot: PY_MOD_EXEC,
                value,
            }),
            export_id::MOD_MULTIPLE_INTERPRETERS => def_slots.push(CPyModuleDefSlot {
                slot: PY_MOD_MULTIPLE_INTERPRETERS,
                value,
            }),
            export_id::MOD_GIL => def_slots.push(CPyModuleDefSlot {
                slot: PY_MOD_GIL,
                value,
            }),
            unknown => {
                if !super::typeobject::SlotFlags::from_bits_retain(slot.sl_flags)
                    .contains(super::typeobject::SlotFlags::PYSLOT_OPTIONAL)
                {
                    return refuse(format!(
                        "module {name}: PyModExport_* returned an unrecognised slot {unknown}"
                    ));
                }
            }
        }
        entry = unsafe { entry.add(1) };
    }
    if !saw_abi {
        return refuse(format!(
            "module {name} does not define Py_mod_abi, which is mandatory for modules defined from slots only."
        ));
    }
    // The create phase reads the array until a zero, and takes both arrays by
    // pointer rather than copying them, so both outlive this call.
    def_slots.push(CPyModuleDefSlot {
        slot: 0,
        value: std::ptr::null_mut(),
    });
    def.m_slots = Box::leak(def_slots.into_boxed_slice()).as_mut_ptr();
    let def = Box::leak(def);
    // Stamps the header the create phase reads the definition back through.
    unsafe { PyModuleDef_Init(def) };
    // `original_def` is NULL for slots-only creation (`moduleobject.c`
    // `module_from_slots_and_spec`), so a create slot sees NULL rather than
    // the synthetic definition kept for exec/state.
    let created = create_module_from_def_like(def, spec, name, path, std::ptr::null_mut())?;
    if !unsafe { pyre_object::module::is_module(created) } {
        return Ok(created);
    }
    let roots = pyre_object::gc_roots::push_roots();
    let module = roots.pin_root(created);
    // `import_run_modexport`: if no `Py_mod_token` was given, the slots array
    // pointer is the token.
    let token = if export_token != 0 {
        export_token
    } else {
        slots as usize
    };
    set_field(module, |fields| {
        fields.md_token = token;
        fields.md_token_is_def = false;
    });
    Ok(module)
}

/// `modsupport.py:create_module_from_def_and_spec` — the PEP 489 create phase.
pub(super) fn create_module_from_def_and_spec(
    def: *mut CPyModuleDef,
    spec: PyObjectRef,
    name: &str,
    path: Option<&std::path::Path>,
) -> Result<PyObjectRef, crate::PyError> {
    create_module_from_def_like(def, spec, name, path, def)
}

/// `moduleobject.c` `module_from_slots_and_spec`.  `original_def` is the
/// pointer handed to a create slot and the token of a def-created module;
/// NULL for a slots-only export.
fn create_module_from_def_like(
    def: *mut CPyModuleDef,
    spec: PyObjectRef,
    name: &str,
    path: Option<&std::path::Path>,
    original_def: *mut CPyModuleDef,
) -> Result<PyObjectRef, crate::PyError> {
    // Every module definition the extension supplied names itself, so a NULL
    // here is not a definition read at the offsets it was written at.  It is
    // the one field that says so before anything else goes wrong: a definition
    // laid out against a `PyObject` header one word wider than this
    // interpreter's puts its `m_copy` where `m_name` is read, its `m_name`
    // where `m_doc` is, and its `m_methods` where `m_slots` is.  Reading it
    // that way is not an error anywhere: the module is created, `__doc__`
    // holds the name, no execution slot is found, and the module an extension
    // spent its whole init filling comes back empty.
    if !original_def.is_null() && unsafe { (*def).m_name.is_null() } {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            format!(
                "module {name}: its PyModuleDef has no m_name, so it was not \
                 laid out against this interpreter's headers -- an extension \
                 that declares CPython's structs itself reads every field of \
                 this one at the wrong offset, since a PyObject here is {} \
                 bytes",
                std::mem::size_of::<CPyObject>()
            ),
        ));
    }
    if unsafe { (*def).m_size } < 0 {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            format!("module {name}: m_size may not be negative for multi-phase initialization"),
        ));
    }
    let mut create: *mut c_void = std::ptr::null_mut();
    let mut has_execution_slots = false;
    // `moduleobject.c:322-343` records each of the two declarations once and
    // rejects a repeat.  What either one selects is guarded by
    // `!_Py_IsMainInterpreter` or by the per-interpreter GIL, neither of which
    // pyre has, so recording is the whole of it here.
    let mut declared_interpreters = false;
    let mut declared_gil = false;
    let mut slot = unsafe { (*def).m_slots };
    while !slot.is_null() && unsafe { (*slot).slot } != 0 {
        let raw = unsafe { (*slot).slot };
        let Some(id) = resolve_mod_slot(raw) else {
            return Err(crate::PyError::new(
                crate::PyErrorKind::SystemError,
                format!("module {name} uses unknown slot ID {raw}"),
            ));
        };
        let repeated = |what: &str| {
            Err(crate::PyError::new(
                crate::PyErrorKind::SystemError,
                format!("module {name} has multiple {what} slots"),
            ))
        };
        match id {
            PY_MOD_CREATE => {
                if !create.is_null() {
                    return repeated("create");
                }
                create = unsafe { (*slot).value };
            }
            PY_MOD_EXEC => has_execution_slots = true,
            PY_MOD_MULTIPLE_INTERPRETERS => {
                if declared_interpreters {
                    return repeated("Py_mod_multiple_interpreters");
                }
                declared_interpreters = true;
            }
            PY_MOD_GIL => {
                if declared_gil {
                    return repeated("Py_mod_gil");
                }
                declared_gil = true;
            }
            PY_MOD_ABI => {
                abiinfo_check(unsafe { (*slot).value as *mut CPyABIInfo }, Some(name))?;
            }
            PY_MOD_TOKEN => {
                if !original_def.is_null()
                    && unsafe { (*slot).value } != original_def as *mut c_void
                {
                    return Err(crate::PyError::new(
                        crate::PyErrorKind::SystemError,
                        format!(
                            "module {name}: arbitrary Py_mod_token not allowed with PyModuleDef"
                        ),
                    ));
                }
            }
            other => {
                return Err(crate::PyError::new(
                    crate::PyErrorKind::SystemError,
                    format!("module {name} uses unknown slot ID {other}"),
                ));
            }
        }
        slot = unsafe { slot.add(1) };
    }

    // The create slot is handed the spec as its first argument and reads
    // `spec.name` out of it, so a NULL is dereferenced inside the extension.
    // An import that reaches here before the importlib bootstrap can build a
    // spec has none to hand over.
    if !create.is_null() && spec.is_null() {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            format!("module {name}: the create slot needs a module spec"),
        ));
    }

    let roots = pyre_object::gc_roots::push_roots();
    let spec_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(spec);
    let module = if create.is_null() {
        pyre_object::w_module_new_managed(name)
    } else {
        let spec_ref = pyobject::make_ref(pyre_object::gc_roots::shadow_stack_get(spec_slot));
        let result = unsafe {
            let call: unsafe extern "C" fn(*mut CPyObject, *mut CPyModuleDef) -> *mut CPyObject =
                std::mem::transmute(create);
            call(spec_ref, original_def)
        };
        unsafe { pyobject::decref(spec_ref) };
        // `moduleobject.c PyModule_FromDefAndSpec2`: a create slot that answered
        // a module while leaving an exception set reported a success it did not
        // have.  `from_c_result` would report that as an anonymous
        // `SystemError`, so the module's name and the raise are kept here.
        if !result.is_null()
            && let Some(pending) = pyerrors::take_pending_error()
        {
            unsafe { pyobject::decref(result) };
            return Err(crate::error::system_error_from_cause(
                format!("creation of module {name} raised unreported exception"),
                pending,
            ));
        }
        super::from_c_result(result)?
    };
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    let module = roots.pin_root(module);
    if !unsafe { pyre_object::module::is_module(module) } {
        if unsafe { (*def).m_size } > 0
            || unsafe { !(*def).m_traverse.is_null() }
            || unsafe { !(*def).m_clear.is_null() }
            || unsafe { !(*def).m_free.is_null() }
        {
            return Err(crate::PyError::new(
                crate::PyErrorKind::SystemError,
                format!("module {name} is not a module object, but requests module state"),
            ));
        }
        if has_execution_slots {
            return Err(crate::PyError::new(
                crate::PyErrorKind::SystemError,
                format!(
                    "module {name} specifies execution slots, \
                     but did not create a ModuleType instance"
                ),
            ));
        }
        return add_methods_and_doc(
            pyre_object::gc_roots::shadow_stack_get(module_slot),
            def,
            name,
        );
    }
    populate_module(
        pyre_object::gc_roots::shadow_stack_get(module_slot),
        def,
        name,
        path,
    )
}

/// `true` once the module owns a state block, which is what
/// `exec_extension_module` reads to tell an executed module from a fresh one.
pub(super) fn has_module_state(module: PyObjectRef) -> bool {
    fields(module).md_state != 0
}

/// `modsupport.py:exec_def` — the PEP 489 exec phase, on the definition the
/// module recorded for itself.
pub(super) fn exec_def(module: PyObjectRef) -> Result<(), crate::PyError> {
    let address = fields(module).md_def;
    if address == 0 {
        return Ok(());
    }
    exec_def_of(module, address as *mut CPyModuleDef)
}

/// The exec phase against an explicitly named definition, which is the form
/// `PyModule_ExecDef` hands in.
fn exec_def_of(module: PyObjectRef, def: *mut CPyModuleDef) -> Result<(), crate::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    let module = roots.pin_root(module);
    // The state block is allocated before the first slot runs, both because
    // `PyModule_GetState` is what an exec slot is there to fill and because its
    // address is the marker that keeps a second exec from running.
    if unsafe { (*def).m_size } >= 0
        && !has_module_state(pyre_object::gc_roots::shadow_stack_get(module_slot))
    {
        allocate_module_state(
            pyre_object::gc_roots::shadow_stack_get(module_slot),
            unsafe { (*def).m_size },
        )?;
    }
    let mut slot = unsafe { (*def).m_slots };
    while !slot.is_null() && unsafe { (*slot).slot } != 0 {
        if resolve_mod_slot(unsafe { (*slot).slot }) == Some(PY_MOD_EXEC) {
            let module = pyre_object::gc_roots::shadow_stack_get(module_slot);
            let module_ref = pyobject::make_ref(module);
            let result = unsafe {
                let call: unsafe extern "C" fn(*mut CPyObject) -> c_int =
                    std::mem::transmute((*slot).value);
                call(module_ref)
            };
            unsafe { pyobject::decref(module_ref) };
            if result != 0 {
                return Err(pyerrors::take_pending_error().unwrap_or_else(|| {
                    crate::PyError::new(
                        crate::PyErrorKind::SystemError,
                        format!(
                            "execution of module {} failed without setting an exception",
                            text_or_empty(unsafe { (*def).m_name })
                        ),
                    )
                }));
            }
            if let Some(pending) = pyerrors::take_pending_error() {
                return Err(crate::error::system_error_from_cause(
                    format!(
                        "execution of module {} raised unreported exception",
                        text_or_empty(unsafe { (*def).m_name })
                    ),
                    pending,
                ));
            }
        }
        slot = unsafe { slot.add(1) };
    }
    Ok(())
}

// ── the module accessors ────────────────────────────────────────────────

fn module_argument(raw: *mut CPyObject, function: &str) -> Option<PyObjectRef> {
    let module = unsafe { pyobject::from_ref(raw) };
    if module.is_null() || !unsafe { pyre_object::module::is_module(module) } {
        pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            format!("{function}(): not a module"),
        ));
        return None;
    }
    Some(module)
}

fn as_module(raw: *mut CPyObject) -> Option<PyObjectRef> {
    let module = unsafe { pyobject::from_ref(raw) };
    if module.is_null() || !unsafe { pyre_object::module::is_module(module) } {
        None
    } else {
        Some(module)
    }
}

/// `_PyModule_GetToken` — the stored token, or NULL when `raw` is not a module.
pub(super) unsafe fn module_token(raw: *mut CPyObject) -> *mut c_void {
    match as_module(raw) {
        Some(module) => fields(module).md_token as *mut c_void,
        None => std::ptr::null_mut(),
    }
}

fn expected_module_got(raw: *mut CPyObject) -> String {
    if raw.is_null() {
        return "expected module, got NULL".to_owned();
    }
    format!(
        "expected module, got {}",
        crate::type_methods::arg_type_name(unsafe { pyobject::from_ref(raw) })
    )
}

/// Borrowed, exactly as upstream marks it `result_borrowed=True`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_GetDict(module: *mut CPyObject) -> *mut CPyObject {
    let Some(module) = module_argument(module, "PyModule_GetDict") else {
        return std::ptr::null_mut();
    };
    // Borrowed: the module's mirror owns the reference for as long as it
    // lives, which is what makes the result usable after this returns.
    pyobject::borrow_from(pyobject::as_pyobj(module), module_dict(module))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_GetState(module: *mut CPyObject) -> *mut c_void {
    let Some(module) = module_argument(module, "PyModule_GetState") else {
        return std::ptr::null_mut();
    };
    fields(module).md_state as *mut c_void
}

/// `PyModule_GetState_DuringGC` — no exception on a non-module.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_GetState_DuringGC(module: *mut CPyObject) -> *mut c_void {
    match as_module(module) {
        Some(module) => fields(module).md_state as *mut c_void,
        None => std::ptr::null_mut(),
    }
}

/// `_PyModule_GetDefOrNull`: the def only when the token *is* the def.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_GetDef(module: *mut CPyObject) -> *mut CPyModuleDef {
    let Some(module) = module_argument(module, "PyModule_GetDef") else {
        return std::ptr::null_mut();
    };
    let stored = fields(module);
    if stored.md_token_is_def {
        stored.md_def as *mut CPyModuleDef
    } else {
        std::ptr::null_mut()
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_GetToken_DuringGC(
    module: *mut CPyObject,
    token_p: *mut *mut c_void,
) -> c_int {
    unsafe { *token_p = std::ptr::null_mut() };
    let Some(module) = as_module(module) else {
        return -1;
    };
    unsafe { *token_p = fields(module).md_token as *mut c_void };
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_GetToken(
    module: *mut CPyObject,
    token_p: *mut *mut c_void,
) -> c_int {
    unsafe { *token_p = std::ptr::null_mut() };
    let Some(module) = as_module(module) else {
        pyerrors::set_pending_error(crate::PyError::type_error(expected_module_got(module)));
        return -1;
    };
    unsafe { *token_p = fields(module).md_token as *mut c_void };
    0
}

/// Steals a reference to `value` on success, which is `PyModule_AddObject`'s
/// documented and much-complained-about contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_AddObject(
    module: *mut CPyObject,
    name: *const c_char,
    value: *mut CPyObject,
) -> c_int {
    super::object::realize_all([module, value]);
    if unsafe { PyModule_AddObjectRef(module, name, value) } != 0 {
        return -1;
    }
    unsafe { pyobject::decref(value) };
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_AddObjectRef(
    module: *mut CPyObject,
    name: *const c_char,
    value: *mut CPyObject,
) -> c_int {
    super::object::realize_all([module, value]);
    // The two rejections are `src/modsupport.c PyModule_AddObjectRef`'s own,
    // rather than the `X(): not a module` the entry points implemented in
    // `modsupport.py` share.
    let w_module = unsafe { pyobject::from_ref(module) };
    if w_module.is_null() || !unsafe { pyre_object::module::is_module(w_module) } {
        pyerrors::set_pending_error(crate::PyError::new(
            crate::PyErrorKind::TypeError,
            "PyModule_AddObjectRef() first argument must be a module".to_owned(),
        ));
        return -1;
    }
    if value.is_null() {
        // A caller returning NULL is expected to say why, and that exception is
        // the one the failure carries; the message below stands in only when
        // there is none.
        if !pyerrors::has_pending_error() {
            pyerrors::set_pending_error(crate::PyError::new(
                crate::PyErrorKind::SystemError,
                "PyModule_AddObjectRef() must be called with an exception \
                 raised if value is NULL"
                    .to_owned(),
            ));
        }
        return -1;
    }
    if name.is_null() {
        unsafe { pyerrors::PyErr_BadInternalCall() };
        return -1;
    }
    let key = text_or_empty(name);
    let value = unsafe { pyobject::from_ref(value) };
    store(w_module, &key, value);
    0
}

/// The value is a C `long`, which is the width the reference declaration says
/// and narrower than a `Py_ssize_t` wherever the two differ.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_AddIntConstant(
    module: *mut CPyObject,
    name: *const c_char,
    value: c_long,
) -> c_int {
    let Some(module) = module_argument(module, "PyModule_AddIntConstant") else {
        return -1;
    };
    if name.is_null() {
        unsafe { pyerrors::PyErr_BadInternalCall() };
        return -1;
    }
    let key = text_or_empty(name);
    let roots = pyre_object::gc_roots::push_roots();
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(module);
    let value = pyre_object::w_int_new(value as i64);
    store(
        pyre_object::gc_roots::shadow_stack_get(module_slot),
        &key,
        value,
    );
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_AddStringConstant(
    module: *mut CPyObject,
    name: *const c_char,
    value: *const c_char,
) -> c_int {
    let Some(module) = module_argument(module, "PyModule_AddStringConstant") else {
        return -1;
    };
    if name.is_null() || value.is_null() {
        unsafe { pyerrors::PyErr_BadInternalCall() };
        return -1;
    }
    let key = text_or_empty(name);
    let text = text_or_empty(value);
    let roots = pyre_object::gc_roots::push_roots();
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(module);
    let value_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(pyre_object::w_str_new_managed(&text));
    store(
        pyre_object::gc_roots::shadow_stack_get(module_slot),
        &key,
        pyre_object::gc_roots::shadow_stack_get(value_slot),
    );
    0
}

/// `module.py:24` with the name object kept as it arrived: a module name can
/// carry a lone surrogate, so projecting it through a `&str` would lose the
/// buffer the caller handed over.
///
/// The four entries beyond `__name__` are the ones a module built this way is
/// documented to arrive with; `__file__` is deliberately not among them, and
/// is the caller's to add.
fn new_module(w_name: PyObjectRef) -> PyObjectRef {
    let roots = pyre_object::gc_roots::push_roots();
    let name_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_name);
    // The empty name is the anonymous-module sentinel, so the name is attached
    // afterwards rather than through the allocation.
    let module = pyre_object::w_module_new_managed("");
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(module);
    let reload = |slot| pyre_object::gc_roots::shadow_stack_get(slot);
    unsafe { pyre_object::w_module_set_name(reload(module_slot), reload(name_slot)) };
    store(reload(module_slot), "__name__", reload(name_slot));
    for key in ["__doc__", "__package__", "__loader__", "__spec__"] {
        store(reload(module_slot), key, pyre_object::w_none());
    }
    reload(module_slot)
}

/// `modsupport.py:PyModule_New` — only `__name__` is filled in; `__file__` is
/// the caller's to provide.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_New(name: *const c_char) -> *mut CPyObject {
    if name.is_null() {
        unsafe { pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    let name = text_or_empty(name);
    pyobject::make_ref(new_module(pyre_object::w_str_new_managed(&name)))
}

/// `modsupport.py:PyModule_NewObject`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_NewObject(name: *mut CPyObject) -> *mut CPyObject {
    let w_name = unsafe { pyobject::from_ref(name) };
    if w_name.is_null() {
        unsafe { pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    pyobject::make_ref(new_module(w_name))
}

/// The `__name__` a module reports.
///
/// Upstream reads `w_mod.w_name`, the name frozen at construction; the module
/// dictionary is read here instead so that a module renamed after import
/// reports the new name, and every module pyre builds seeds both from the same
/// object.
fn module_name_object(module: PyObjectRef, function: &str) -> Option<PyObjectRef> {
    let dict = module_dict(module);
    let name = if dict.is_null() {
        None
    } else {
        unsafe { pyre_object::dictmultiobject::w_dict_getitem_str(dict, "__name__") }
    };
    match name {
        Some(name) if unsafe { pyre_object::is_str(name) } => Some(name),
        _ => {
            pyerrors::set_pending_error(crate::PyError::new(
                crate::PyErrorKind::SystemError,
                format!("{function}(): nameless module"),
            ));
            None
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_GetNameObject(module: *mut CPyObject) -> *mut CPyObject {
    let Some(module) = module_argument(module, "PyModule_GetNameObject") else {
        return std::ptr::null_mut();
    };
    let Some(name) = module_name_object(module, "PyModule_GetNameObject") else {
        return std::ptr::null_mut();
    };
    pyobject::make_ref(name)
}

/// The buffer belongs to the name's mirror, which the module dictionary keeps
/// alive for as long as the module — the lifetime upstream relies on for the
/// same `char *`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_GetName(module: *mut CPyObject) -> *const c_char {
    let Some(module) = module_argument(module, "PyModule_GetName") else {
        return std::ptr::null();
    };
    let Some(name) = module_name_object(module, "PyModule_GetName") else {
        return std::ptr::null();
    };
    unsafe { super::unicodeobject::PyUnicode_AsUTF8(pyobject::borrow_mirror(name)) }
}

/// The `__file__` a module was loaded from.
///
/// A missing or non-`str` entry is the `SystemError` the entry point is
/// documented to raise, rather than the `KeyError` a plain `getitem` would
/// surface.
fn module_filename_object(module: PyObjectRef, function: &str) -> Option<PyObjectRef> {
    let dict = module_dict(module);
    let file = if dict.is_null() {
        None
    } else {
        unsafe { pyre_object::dictmultiobject::w_dict_getitem_str(dict, "__file__") }
    };
    match file {
        Some(file) if unsafe { pyre_object::is_str(file) } => Some(file),
        _ => {
            pyerrors::set_pending_error(crate::PyError::new(
                crate::PyErrorKind::SystemError,
                format!("{function}(): module filename missing"),
            ));
            None
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_GetFilenameObject(module: *mut CPyObject) -> *mut CPyObject {
    let Some(module) = module_argument(module, "PyModule_GetFilenameObject") else {
        return std::ptr::null_mut();
    };
    let Some(file) = module_filename_object(module, "PyModule_GetFilenameObject") else {
        return std::ptr::null_mut();
    };
    pyobject::make_ref(file)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_GetFilename(module: *mut CPyObject) -> *const c_char {
    let Some(module) = module_argument(module, "PyModule_GetFilename") else {
        return std::ptr::null();
    };
    let Some(file) = module_filename_object(module, "PyModule_GetFilename") else {
        return std::ptr::null();
    };
    unsafe { super::unicodeobject::PyUnicode_AsUTF8(pyobject::borrow_mirror(file)) }
}

/// Releases `value` whether the store succeeds or not, which is what makes
/// this the spelling that composes with a call producing a new reference.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_Add(
    module: *mut CPyObject,
    name: *const c_char,
    value: *mut CPyObject,
) -> c_int {
    super::object::realize_all([module, value]);
    let result = unsafe { PyModule_AddObjectRef(module, name, value) };
    if !value.is_null() {
        unsafe { pyobject::decref(value) };
    }
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_SetDocString(
    module: *mut CPyObject,
    doc: *const c_char,
) -> c_int {
    let Some(module) = module_argument(module, "PyModule_SetDocString") else {
        return -1;
    };
    if doc.is_null() {
        unsafe { pyerrors::PyErr_BadInternalCall() };
        return -1;
    }
    let text = text_or_empty(doc);
    let roots = pyre_object::gc_roots::push_roots();
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(module);
    let value_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(pyre_object::w_str_new_managed(&text));
    store(
        pyre_object::gc_roots::shadow_stack_get(module_slot),
        "__doc__",
        pyre_object::gc_roots::shadow_stack_get(value_slot),
    );
    0
}

/// `modsupport.py:PyModule_AddFunctions`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_AddFunctions(
    module: *mut CPyObject,
    methods: *mut CPyMethodDef,
) -> c_int {
    let Some(module) = module_argument(module, "PyModule_AddFunctions") else {
        return -1;
    };
    let roots = pyre_object::gc_roots::push_roots();
    let module_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(module);
    let Some(name) = module_name_object(
        pyre_object::gc_roots::shadow_stack_get(module_slot),
        "PyModule_AddFunctions",
    ) else {
        return -1;
    };
    let name_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(name);
    if trap(convert_method_defs(
        pyre_object::gc_roots::shadow_stack_get(module_slot),
        methods,
        pyre_object::gc_roots::shadow_stack_get(name_slot),
    ))
    .is_none()
    {
        return -1;
    }
    0
}

/// `modsupport.py:PyModule_ExecDef` — the exec phase driven by the caller
/// rather than by import, against the definition it names.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_ExecDef(module: *mut CPyObject, def: *mut CPyModuleDef) -> c_int {
    let Some(module) = module_argument(module, "PyModule_ExecDef") else {
        return -1;
    };
    if def.is_null() {
        unsafe { pyerrors::PyErr_BadInternalCall() };
        return -1;
    }
    if trap(exec_def_of(module, def)).is_none() {
        return -1;
    }
    0
}

/// `modsupport.py:PyModule_FromDefAndSpec2` — the PEP 489 create phase reached
/// from C instead of from import, so the module records no `__file__`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_FromDefAndSpec2(
    def: *mut CPyModuleDef,
    spec: *mut CPyObject,
    module_api_version: c_int,
) -> *mut CPyObject {
    if def.is_null() {
        unsafe { pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    let w_spec = if spec.is_null() {
        pyre_object::w_none()
    } else {
        unsafe { pyobject::from_ref(spec) }
    };
    let roots = pyre_object::gc_roots::push_roots();
    let spec_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_spec);
    let Some(name) = trap(
        crate::baseobjspace::getattr_str(
            pyre_object::gc_roots::shadow_stack_get(spec_slot),
            "name",
        )
        .and_then(|w_name| crate::baseobjspace::text_w(w_name).map(str::to_owned)),
    ) else {
        return std::ptr::null_mut();
    };
    if module_api_version != PYTHON_API_VERSION && module_api_version != PYTHON_ABI_VERSION {
        let truncated: String = name.chars().take(100).collect();
        let message = format!(
            "Python C API version mismatch for module {truncated}: \
             This Python has API version {PYTHON_API_VERSION}, \
             module {truncated} has version {module_api_version}."
        );
        if trap(crate::warn::warn_category(&message, "RuntimeWarning", 1)).is_none() {
            return std::ptr::null_mut();
        }
    }
    let Some(module) = trap(create_module_from_def_and_spec(
        def,
        pyre_object::gc_roots::shadow_stack_get(spec_slot),
        &name,
        None,
    )) else {
        return std::ptr::null_mut();
    };
    pyobject::make_ref(module)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_Check(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    (!object.is_null() && unsafe { pyre_object::module::is_module(object) }) as c_int
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyModule_CheckExact(object: *mut CPyObject) -> c_int {
    let object = unsafe { pyobject::from_ref(object) };
    (!object.is_null() && super::object::is_exactly(object, &pyre_object::MODULE_TYPE)) as c_int
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PyModuleDef_Init as *const ());
    std::hint::black_box(PyModule_Create2 as *const ());
    std::hint::black_box(PyModule_GetDict as *const ());
    std::hint::black_box(PyModule_GetState as *const ());
    std::hint::black_box(PyModule_GetState_DuringGC as *const ());
    std::hint::black_box(PyModule_GetToken as *const ());
    std::hint::black_box(PyModule_GetToken_DuringGC as *const ());
    std::hint::black_box(PyABIInfo_Check as *const ());
    std::hint::black_box(PyState_AddModule as *const ());
    std::hint::black_box(PyState_FindModule as *const ());
    std::hint::black_box(PyState_RemoveModule as *const ());
    std::hint::black_box(PyModule_GetDef as *const ());
    std::hint::black_box(PyModule_AddObject as *const ());
    std::hint::black_box(PyModule_AddObjectRef as *const ());
    std::hint::black_box(PyModule_AddIntConstant as *const ());
    std::hint::black_box(PyModule_AddStringConstant as *const ());
    std::hint::black_box(PyModule_New as *const ());
    std::hint::black_box(PyModule_NewObject as *const ());
    std::hint::black_box(PyModule_GetName as *const ());
    std::hint::black_box(PyModule_GetNameObject as *const ());
    std::hint::black_box(PyModule_GetFilename as *const ());
    std::hint::black_box(PyModule_GetFilenameObject as *const ());
    std::hint::black_box(PyModule_Add as *const ());
    std::hint::black_box(PyModule_SetDocString as *const ());
    std::hint::black_box(PyModule_AddFunctions as *const ());
    std::hint::black_box(PyModule_ExecDef as *const ());
    std::hint::black_box(PyModule_FromDefAndSpec2 as *const ());
    std::hint::black_box(PyModule_Check as *const ());
    std::hint::black_box(PyModule_CheckExact as *const ());
}

#[cfg(test)]
mod tests {
    /// A `PyModuleDef_Slot.slot` is the number an extension compiled against
    /// `include/pyre3.15t/slots_generated.h` wrote, so the identifiers are one
    /// table in two places.  This walks the header and rejects any identifier
    /// this file spells whose number differs.  A `_Py_SLOT_COMPAT_VALUE(OLD,
    /// NEW)` body is the non-limited expansion, NEW.
    #[test]
    fn every_module_slot_is_the_number_the_header_gives_it() {
        const HEADER: &str = include_str!("../../../../include/pyre3.15t/slots_generated.h");

        let mut header: Vec<(&str, std::ffi::c_int)> = Vec::new();
        for line in HEADER.lines() {
            let Some(rest) = line.strip_prefix("#define Py_mod_") else {
                continue;
            };
            let Some((name, body)) = rest.split_once(' ') else {
                continue;
            };
            let Some(value) = slot_define_value(body) else {
                continue;
            };
            header.push((name, value));
        }

        for &(known, ours) in super::ALL_MODULE_SLOTS {
            let name = known
                .strip_prefix("PY_MOD_")
                .expect("every module slot is declared under the header's own name")
                .to_ascii_lowercase();
            let found = header.iter().find(|(theirs, _)| *theirs == name);
            let Some((_, theirs)) = found else {
                panic!(
                    "this file declares {known} = {ours}, and slots_generated.h has no Py_mod_{name}"
                );
            };
            assert_eq!(
                ours, *theirs,
                "Py_mod_{name} is {theirs} in slots_generated.h and {ours} here"
            );
        }

        let new_of = |name: &str| {
            header
                .iter()
                .find(|(theirs, _)| *theirs == name)
                .map(|(_, value)| *value)
                .unwrap_or_else(|| panic!("slots_generated.h declares no Py_mod_{name}"))
        };
        assert_eq!(super::resolve_mod_slot(1), Some(new_of("create")));
        assert_eq!(super::resolve_mod_slot(2), Some(new_of("exec")));
        assert_eq!(
            super::resolve_mod_slot(3),
            Some(new_of("multiple_interpreters"))
        );
        assert_eq!(super::resolve_mod_slot(4), Some(new_of("gil")));
    }

    fn slot_define_value(body: &str) -> Option<std::ffi::c_int> {
        let body = body.trim();
        if let Some(args) = body.strip_prefix("_Py_SLOT_COMPAT_VALUE(") {
            let args = args.strip_suffix(')')?;
            let (_, new) = args.split_once(',')?;
            new.trim().parse().ok()
        } else {
            body.parse().ok()
        }
    }

    /// `PY_VERSION_HEX` is `patchlevel.h`'s `_Py_PACK_FULL_VERSION` of the
    /// five version macros, which is what `PyABIInfo_Check` compares against.
    #[test]
    fn py_version_hex_is_the_number_the_header_packs() {
        const HEADER: &str = include_str!("../../../../include/pyre3.15t/patchlevel.h");
        let define = |name: &str| -> u32 {
            HEADER
                .lines()
                .find_map(|line| {
                    line.strip_prefix(&format!("#define {name} "))
                        .and_then(|body| {
                            let token = body.split_whitespace().next()?;
                            if let Some(hex) = token.strip_prefix("0x") {
                                u32::from_str_radix(hex, 16).ok()
                            } else {
                                token.parse().ok()
                            }
                        })
                })
                .unwrap_or_else(|| panic!("patchlevel.h declares no {name}"))
        };
        let level_name = HEADER
            .lines()
            .find_map(|line| line.strip_prefix("#define PY_RELEASE_LEVEL "))
            .map(str::trim)
            .expect("patchlevel.h declares PY_RELEASE_LEVEL");
        let packed = super::pack_full_version(
            define("PY_MAJOR_VERSION"),
            define("PY_MINOR_VERSION"),
            define("PY_MICRO_VERSION"),
            define(level_name),
            define("PY_RELEASE_SERIAL"),
        );
        assert_eq!(
            super::PY_VERSION_HEX,
            packed,
            "PY_VERSION_HEX is {packed:#x} in patchlevel.h and {:#x} here",
            super::PY_VERSION_HEX
        );
    }

    /// The two versions `PyModule_Create2` accepts are `patchlevel.h`'s, which
    /// is what an extension's own `PyModule_Create` passes.
    #[test]
    fn both_module_versions_are_the_numbers_the_header_gives_them() {
        const HEADER: &str = include_str!("../../../../include/pyre3.15t/patchlevel.h");

        for (name, ours) in [
            ("PYTHON_API_VERSION", super::PYTHON_API_VERSION),
            ("PYTHON_ABI_VERSION", super::PYTHON_ABI_VERSION),
        ] {
            let theirs = HEADER
                .lines()
                .find_map(|line| line.strip_prefix(&format!("#define {name} ")))
                .and_then(|body| body.trim().parse::<std::ffi::c_int>().ok())
                .unwrap_or_else(|| panic!("patchlevel.h declares no {name}"));
            assert_eq!(
                ours, theirs,
                "{name} is {theirs} in patchlevel.h and {ours} here"
            );
        }
    }
}
