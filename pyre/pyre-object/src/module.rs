//! `pypy/interpreter/module.py` — Python `module` type.
//!
//! A module holds its wrapped name and backing dict objects.

#![allow(unsafe_op_in_unsafe_fn)]

use crate::pyobject::*;

/// Python module object.
///
/// Layout: `[ob_type | w_name | w_dict | w_initialdict | startup_called | lifeline]`
///
/// `w_dict` mirrors PyPy `module.py self.w_dict = w_dict` — every
/// Module owns a non-null `W_DictObject` (or dict subclass instance
/// for the user-supplied wrap case at `moduledef.py:102-103`).  For
/// ordinary Modules pyre constructs a `W_ModuleDictObject`, so
/// `getdict(space)` returns a stable identity across calls. For the user-supplied
/// case `w_dict` is the caller's object directly, preserving subclass
/// identity for `space.finditem_str` dispatch.
#[repr(C)]
pub struct Module {
    pub ob_header: PyObject,
    /// `module.py:22 self.w_name = w_name`. `PY_NULL` is the anonymous
    /// sentinel installed by `module.__new__` before `module.__init__` runs.
    pub w_name: PyObjectRef,
    /// Authoritative dict object (`PyPy module.w_dict`).  Always non-null
    /// after construction.
    pub w_dict: PyObjectRef,
    /// `mixedmodule.py MixedModule.w_initialdict`: `PY_NULL` as long as the
    /// builtin module has not been imported yet, and when it has been, its
    /// `__dict__.copy()` taken just after `startup()`. Stays `PY_NULL` on a
    /// module that is not a builtin.
    pub w_initialdict: PyObjectRef,
    /// `module.py Module.startup_called`.
    pub startup_called: bool,
    /// `typedef.py make_weakref_descr(Module)` `_lifeline_`.
    pub lifeline: PyObjectRef,
}

/// GC type id assigned to `Module` at JitDriver init time.
pub const W_MODULE_GC_TYPE_ID: u32 = 36;

/// Fixed payload size (`framework.py:811`).
pub const W_MODULE_OBJECT_SIZE: usize = std::mem::size_of::<Module>();

/// Byte offsets of the inline `PyObjectRef` slots the GC must trace.
///
/// `w_dict` — the aliased `W_DictObject` (`pypy/interpreter/module.py:22
/// self.w_dict = w_dict`) so a Module surviving a collection keeps its
/// dict alive.
///
/// `w_class` — the module's class. For a `types.ModuleType` subclass
/// instance this is a heap-allocated (GC-managed, collectible)
/// `W_TypeObject`; if the module were its only reference, an untraced
/// slot would let a major collection sweep the class and leave
/// `type(m)` / slot dispatch pointing at freed memory. `W_ObjectObject`
/// traces its `w_class` for the same reason (`object_object_custom_trace`).
///
/// `w_initialdict` — the saved builtin dict a later import copies from.
///
pub const W_MODULE_GC_PTR_OFFSETS: [usize; 5] = [
    std::mem::offset_of!(Module, ob_header.w_class),
    std::mem::offset_of!(Module, w_name),
    std::mem::offset_of!(Module, w_dict),
    std::mem::offset_of!(Module, w_initialdict),
    std::mem::offset_of!(Module, lifeline),
];

impl crate::lltype::GcType for Module {
    fn type_id() -> u32 {
        W_MODULE_GC_TYPE_ID
    }
    const SIZE: usize = W_MODULE_OBJECT_SIZE;
}

/// Allocate a new Module backed by a fresh `W_ModuleDictObject`. Use this
/// for `space.builtin`, freshly-imported modules, REPL `__main__`, and
/// other Modules whose authoritative namespace is their dict object.
///
/// `module.py` — `if w_name is not None: setitem(w_dict, '__name__',
/// w_name)`. Pyre seeds `__name__` through `w_dict_setitem_str`.
///
/// `name` — the module name (e.g. "math", "os.path"); empty string is
///   the anonymous-name sentinel for `pick_builtin`'s default Module
///   case (`moduledef.py:106-108`, PyPy `Module(space, None, ...)`)
///   in which `Module.__init__` skips the `__name__` setitem.
///
fn module_value(name: &str) -> Module {
    // `pypy/interpreter/module.py Module.__init__` opens
    // `w_dict = space.newdict(module=True)` per `dictmultiobject.py
    // _newdict(module=True)`, which lands on `W_ModuleDictObject`
    // (ModuleDictStrategy + cell-cache). Pyre routes through
    // `w_module_dict_new`; `pypy/objspace/std/celldict.py` strategy semantics
    // (`get_global_cache`, `invalidate_caches`,
    // `switch_to_object_strategy`) cover the module surface.
    let w_name = if name.is_empty() {
        PY_NULL
    } else {
        crate::w_str_new(name)
    };
    let w_dict = crate::dictmultiobject::w_module_dict_new();
    if !w_name.is_null() {
        unsafe {
            crate::dictmultiobject::w_dict_setitem_str(w_dict, "__name__", w_name);
        }
    }
    Module {
        ob_header: PyObject {
            ob_type: &MODULE_TYPE as *const PyType,
            w_class: get_instantiate(&MODULE_TYPE),
        },
        w_name,
        w_dict,
        w_initialdict: PY_NULL,
        startup_called: false,
        lifeline: PY_NULL,
    }
}

/// Bootstrap/import allocation. Native owners of these modules still keep
/// stable raw pointers, so retain the legacy immortal allocation until those
/// owners are migrated to ordinary GC roots.
pub fn w_module_new(name: &str) -> PyObjectRef {
    crate::lltype::malloc_typed(module_value(name)) as PyObjectRef
}

/// Python-visible module allocation (`types.ModuleType.__new__`).
///
/// The holder belongs to the collector, carries `W_MODULE_GC_TYPE_ID`, and is
/// traced through `W_MODULE_GC_PTR_OFFSETS`, so a minor collection forwards
/// `w_dict`. The allocation itself is stable (non-moving old gen): module
/// objects flow into JIT traces as promoted constants (globals lookups,
/// attribute caches), and a baked pointer must survive later collections.
pub fn w_module_new_managed(name: &str) -> PyObjectRef {
    crate::lltype::malloc_typed_stable(module_value(name)) as PyObjectRef
}

/// `typedef.py` `_getusercls(Module)`. The base payload stays at offset 0,
/// including the typed `w_dict`; `MapdictStorageMixin` contributes `map`
/// and `storage` for `__slots__`. `Module.typedef` is weakrefable, so the
/// weakref lifeline stays out of that storage.
#[repr(C)]
pub struct ModuleUser {
    pub base: Module,
    pub map: usize,
    pub storage: *mut crate::object_array::ItemsBlock,
}

const _: () = {
    assert!(
        std::mem::offset_of!(ModuleUser, storage)
            == std::mem::offset_of!(ModuleUser, map) + std::mem::size_of::<usize>()
    );
};

/// User-subclass module layout. Appended after the closed ids, ahead of
/// the target-gated tail (`build_gc`).
pub const W_MODULE_USER_GC_TYPE_ID: u32 = 203;
pub const W_MODULE_USER_OBJECT_SIZE: usize = std::mem::size_of::<ModuleUser>();

impl crate::lltype::GcType for ModuleUser {
    #[inline(always)]
    fn type_id() -> u32 {
        W_MODULE_USER_GC_TYPE_ID
    }
    const SIZE: usize = W_MODULE_USER_OBJECT_SIZE;
}

/// Subclass module (`module.py` `Module.descr_module__new__` →
/// `space.allocate_instance(Module, w_subtype)`). Stable, like
/// [`w_module_new_managed`]: the typed namespace is born with the instance
/// and a later minor collection has to forward it. The caller enqueues a
/// finalizer when the subtype defines `__del__`.
pub fn w_module_user_new(w_class: PyObjectRef) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let cls_slot = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(w_class);
    let mut value = module_value("");
    let w_class = crate::gc_roots::shadow_stack_get(cls_slot);
    value.ob_header = PyObject {
        ob_type: &MODULE_USER_TYPE as *const PyType,
        w_class,
    };
    crate::lltype::malloc_typed_stable(ModuleUser {
        base: value,
        map: 0,
        storage: std::ptr::null_mut(),
    }) as PyObjectRef
}

/// Allocate a `Module` aliasing a user-supplied `W_DictObject`.
/// Mirrors `pypy/module/__builtin__/moduledef.py
/// module.Module(space, None, w_builtin)`: the Module's dict identity
/// IS the user dict (PyPy `module.w_dict = w_builtin`).
///
/// This is the direct port of PyPy's `Module(space, None, w_builtin)` for dict subclasses:
/// `LOAD_GLOBAL` falls through to `space.finditem_str(module.w_dict,
/// name)` so subclass `__getitem__` overrides are not bypassed.
///
/// `name` seeding (`pypy/interpreter/module.py`): when `name` is a
/// non-empty string, set `w_dict["__name__"] = name` so
/// `module.__name__` resolves and `from module import *`,
/// `import_from` submodule fallback work.  PyPy's
/// `Module.__init__(space, w_name, w_dict)` does `space.setitem(w_dict,
/// space.newtext("__name__"), w_name)` when `w_name is not None`; pyre
/// honours the same contract here so every caller gets `__name__`
/// without duplicating the seeding step at each callsite.  When
/// `w_dict` is a non-`W_DictObject` (dict subclass instance), the
/// setitem is skipped — the subclass's own `__init__` is responsible
/// for seeding `__name__` (matching PyPy `moduledef.py:102-103
/// Module(space, None, w_builtin)` where `w_name=None`).
///
/// `#[dont_look_inside]` (`@jit.dont_look_inside`, `rlib/jit.py`):
/// the body performs a `lltype::malloc_typed` allocation (`Module`) that
/// `fuse_boxing_alloc` leaves unfused, so the JIT residualises the whole call
/// to a stable runtime fnaddr instead of tracing the allocation. The `-> PyObjectRef` result is a plain GCREF with no
/// discriminant to erase.
#[majit_macros::dont_look_inside]
pub fn w_module_new_aliasing_dict(name: &str, w_dict_object: PyObjectRef) -> PyObjectRef {
    let value = module_aliasing_dict_value(name, w_dict_object);
    crate::lltype::malloc_typed(value) as PyObjectRef
}

/// GC-managed counterpart of [`w_module_new_aliasing_dict`].
///
/// Once `sys.modules` exists it is the ordinary object-graph owner of imported
/// modules, exactly as `space.sys.modules` is in PyPy.  A builtin module made
/// after that point must therefore be collectible when its cache entry and
/// every application reference disappear; in particular its
/// module -> dict -> builtin function -> module cycle is not a process root.
/// Stable allocation preserves the address assumptions of promoted module
/// constants while still letting a major collection reclaim the cycle.
#[majit_macros::dont_look_inside]
pub fn w_module_new_aliasing_dict_managed(name: &str, w_dict_object: PyObjectRef) -> PyObjectRef {
    let value = module_aliasing_dict_value(name, w_dict_object);
    crate::lltype::malloc_typed_stable(value) as PyObjectRef
}

fn module_aliasing_dict_value(name: &str, w_dict_object: PyObjectRef) -> Module {
    let w_name = if name.is_empty() {
        PY_NULL
    } else {
        crate::w_str_new(name)
    };
    if !w_name.is_null() && !w_dict_object.is_null() && unsafe { crate::is_dict(w_dict_object) } {
        unsafe {
            crate::dictmultiobject::w_dict_setitem_str(w_dict_object, "__name__", w_name);
        }
    }
    Module {
        ob_header: PyObject {
            ob_type: &MODULE_TYPE as *const PyType,
            w_class: get_instantiate(&MODULE_TYPE),
        },
        w_name,
        w_dict: w_dict_object,
        w_initialdict: PY_NULL,
        startup_called: false,
        lifeline: PY_NULL,
    }
}

/// Get the module name.
///
/// # Safety
/// `obj` must point to a valid `Module`.
pub unsafe fn w_module_get_name(obj: PyObjectRef) -> PyObjectRef {
    let module = &*(obj as *const Module);
    module.w_name
}

/// Replace the wrapped module name (`module.py:22 self.w_name = w_name`). Used
/// by `module.__init__(name, doc)` after `module.__new__` allocates an
/// anonymous module. The holder may already be old, so publish the new edge
/// through the ordinary minimark write barrier.
///
/// # Safety
/// `obj` must point to a valid `Module`.
pub unsafe fn w_module_set_name(obj: PyObjectRef, w_name: PyObjectRef) {
    let module = &mut *(obj as *mut Module);
    module.w_name = w_name;
    crate::gc_hook::try_gc_write_barrier(obj as crate::gc_hook::GCREF);
}

/// `MixedModule.w_initialdict`.
///
/// # Safety
/// `obj` must point to a valid `Module`.
pub unsafe fn w_module_get_initialdict(obj: PyObjectRef) -> PyObjectRef {
    (*(obj as *const Module)).w_initialdict
}

/// `mixedmodule.py save_module_content_for_future_reload` stores the copy.
/// A managed holder records the edge through the minimark write barrier; an
/// immortal one is extra-rooted every minor by
/// `walk_process_import_object_roots` (`Module.w_initialdict`).
///
/// # Safety
/// `obj` must point to a valid `Module`.
pub unsafe fn w_module_set_initialdict(obj: PyObjectRef, w_initialdict: PyObjectRef) {
    (*(obj as *mut Module)).w_initialdict = w_initialdict;
    crate::gc_hook::try_gc_write_barrier(obj as crate::gc_hook::GCREF);
    crate::gc_roots::mark_prebuilt_roots_dirty();
}

/// `Module.startup_called`.
///
/// # Safety
/// `obj` must point to a valid `Module`.
pub unsafe fn w_module_startup_called(obj: PyObjectRef) -> bool {
    (*(obj as *const Module)).startup_called
}

/// `Module.init` sets `startup_called` before it runs `startup()`.
///
/// # Safety
/// `obj` must point to a valid `Module`.
pub unsafe fn w_module_set_startup_called(obj: PyObjectRef) {
    (*(obj as *mut Module)).startup_called = true;
}

/// Get the aliased `W_DictObject` (`PY_NULL` when storage-only).
///
/// # Safety
/// `obj` must point to a valid `Module`.
pub unsafe fn w_module_get_w_dict(obj: PyObjectRef) -> PyObjectRef {
    let module = &*(obj as *const Module);
    module.w_dict
}

/// pypy/interpreter/module.py:Module.getdictvalue —
/// `space.finditem_str(self.w_dict, attr)`.  When `w_dict` is a real
/// `W_DictObject` pyre routes through `w_dict_getitem_str` (which
/// honours the storage-proxy read-through, so storage-only Modules
/// surface storage entries via the same call).  When `w_dict` is a
/// dict subclass instance the caller must take the
/// `space.finditem_str` dispatch path itself (subclass `__getitem__`
/// override) — pyre-object can't reach the interpreter's dispatcher,
/// so we return `None` and rely on the storage fallback at the
/// caller (`eval.rs:load_global_value`).
///
/// # Safety
/// `obj` must point to a valid `Module`.
pub unsafe fn w_module_alias_getitem_str(obj: PyObjectRef, name: &str) -> Option<PyObjectRef> {
    let module = &*(obj as *const Module);
    if module.w_dict.is_null() {
        return None;
    }
    // `W_ModuleDictObject` (`module.py newdict(module=True)`) joins
    // `W_DictObject` here so `w_dict_getitem_str` (which dispatches via
    // the strategy slot) reaches both module-strategy and object-strategy
    // backings.  Subclass instances still fall through to None so the
    // caller (`eval.rs:load_global_value`) takes the
    // `space.finditem_str` dispatch path with the subclass's own
    // `__getitem__`.
    if !crate::is_dict(module.w_dict) && !crate::dictmultiobject::is_module_dict(module.w_dict) {
        return None;
    }
    crate::dictmultiobject::w_dict_getitem_str(module.w_dict, name)
}

/// Check if an object is a module.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_module(obj: PyObjectRef) -> bool {
    py_type_check(obj, &MODULE_TYPE) || py_type_check(obj, &MODULE_USER_TYPE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_create_and_check() {
        let obj = w_module_new("test_mod");
        unsafe {
            assert!(is_module(obj));
            assert!(!is_int(obj));
            let w_name = w_module_get_name(obj);
            assert_eq!(crate::w_str_get_wtf8(w_name), "test_mod");
        }
    }
}
