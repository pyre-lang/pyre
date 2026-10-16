//! `pypy/interpreter/typedef.py` descriptor payload parity port.
//!
//! PyPy stores `fget` / `fset` / `fdel` / `doc` / `reqcls` /
//! `use_closure` / `name` as instance fields on the GetSetProperty
//! object itself — `class GetSetProperty(W_Root): _immutable_fields_
//! = [...]` (typedef.py).  Pyre previously emulated this with
//! a process-global `RwLock<HashMap<usize, GetSetFields>>` keyed by
//! descriptor pointer; that side table was a pure adaptation with no
//! RPython justification (and quietly leaked entries when descriptors
//! were collected).
//!
//! This module replaces the side table with a real W_Root struct
//! whose layout mirrors PyPy's instance shape line-for-line — readers
//! reach the slots via `&*(obj as *const GetSetProperty)`, the GC
//! traces every `PyObjectRef`-shaped field, and there is no global
//! state to fall out of sync with the descriptor's actual lifetime.

use crate::pyobject::*;
use pyre_macros::pyre_class;

/// Host values accepted by TypeDef.rawdict. Keep Python host strings/None
/// distinct from W_Root values: StdObjSpace.wrap allocates the former but
/// calls spacebind on the latter. In particular a wrapped string is NOT the
/// host string TypeDef.__init__ accepts as its doc candidate.
pub enum TypeDefValue {
    Text(String),
    None,
    Root(std::cell::UnsafeCell<PyObjectRef>),
}

impl TypeDefValue {
    /// Wrap a W_Root for `TypeDef.rawdict`. The pointer lives in the host
    /// dict; `walk_typedef_roots` forwards it in place.
    /// # Safety
    /// `value` must be a live W_Root object, rooted across this call.
    pub unsafe fn root(value: PyObjectRef) -> Self {
        assert!(!value.is_null(), "use TypeDefValue::None for host None");
        crate::gc_roots::mark_prebuilt_roots_dirty();
        Self::Root(std::cell::UnsafeCell::new(value))
    }
}

/// `pypy/interpreter/typedef.py TypeDef`, shared with `typeobject.py Layout`.
///
/// This is the existing runtime metadata, not a second type-definition
/// registry. Derived layouts retain this object's identity. It lives in the
/// object crate because Layout cannot depend on the interpreter crate;
/// pyre-interpreter's typedef module re-exports this same type.
///
/// The current bootstrap supplies the layout-relevant subset of TypeDef's
/// fields. Full `TypeDef.__init__` (bases/rawdict/gateway metadata) and
/// TypeCache.build must converge on this owner, not allocate a parallel key.
pub struct TypeDef {
    /// Present for declaration-driven definitions; the legacy layout-only
    /// producers have not supplied declaration names yet.
    pub name: Option<String>,
    pub bases: Vec<*const TypeDef>,
    /// Host declarations, not the Function-valued namespace produced by
    /// TypeCache.build. IndexMap preserves Python dict insertion order.
    pub rawdict: indexmap::IndexMap<String, TypeDefValue>,
    pub doc: Option<String>,
    /// TypeDef.__init__: `self.text_signature = _text_signature_`. Applied to
    /// the W_TypeObject by TypeCache.build from the override declaration
    /// (`setup_builtin_type`).
    pub text_signature: Option<String>,
    pub weakrefable: bool,
    /// Existing allocation-vtable representation of the interpreter class;
    /// replace with the canonical class metadata when rpy_cls is connected.
    pub instance_type: *const PyType,
    acceptable_as_base_class: std::sync::atomic::AtomicBool,
    pub hasdict: bool,
    /// TypeDef.__init__: initialization-time declaration, not a flag inferred
    /// from the generated W_TypeObject's name or namespace.
    pub method_descriptor: bool,
    /// typeobject.py `typedef.flag_sequence_bug_compat`. Copied onto the
    /// W_TypeObject by TypeCache.build; not inherited by subclasses.
    pub flag_sequence_bug_compat: bool,
    /// TypeDef.__init__: `self.heaptype = False`. TypeCache.build passes
    /// `is_heaptype=overridetypedef.heaptype` into W_TypeObject.__init__.
    pub heaptype: bool,
    /// TypeDef.__init__: `self.applevel_subclasses_base = None`. TypeCache.build
    /// uses `applevel_subclasses_base.typedef` as `overridetypedef` so the
    /// derived declaration reuses that base's Layout.
    pub applevel_subclasses_base: *const TypeDef,
    /// TypeDef.__init__ `__buffer`.  `None` takes the first base that
    /// declared one.  cpyext `make_bf_getbuffer` reads only presence;
    /// the `'read'` / `'read-write'` vocabulary is never consulted.
    pub buffer: Option<TypeDefBuffer>,
}

/// TypeDef.__init__ `__buffer` — `'read'` or `'read-write'`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TypeDefBuffer {
    Read,
    ReadWrite,
}

impl TypeDef {
    /// Construct the metadata subset currently supplied by type bootstrap.
    /// This is not yet the full initialization-time TypeDef.__init__ API.
    pub fn new(
        instance_type: *const PyType,
        acceptable_as_base_class: bool,
        hasdict: bool,
    ) -> Self {
        Self {
            name: None,
            bases: Vec::new(),
            rawdict: indexmap::IndexMap::new(),
            doc: None,
            text_signature: None,
            weakrefable: false,
            instance_type,
            acceptable_as_base_class: std::sync::atomic::AtomicBool::new(acceptable_as_base_class),
            hasdict,
            method_descriptor: false,
            flag_sequence_bug_compat: false,
            heaptype: false,
            applevel_subclasses_base: std::ptr::null(),
            buffer: None,
        }
    }

    #[inline]
    pub fn acceptable_as_base_class(&self) -> bool {
        self.acceptable_as_base_class
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    #[inline]
    pub fn set_acceptable_as_base_class(&self, value: bool) {
        self.acceptable_as_base_class
            .store(value, std::sync::atomic::Ordering::Relaxed);
    }

    /// TypeDef._freeze_: track individual prebuilt definitions as PBCs.
    pub fn _freeze_(&self) -> bool {
        true
    }

    /// TypeDef.__init__'s declaration-derived metadata. The host dictionary
    /// must contain unbound gateways/properties, not materialized Functions.
    ///
    /// # Safety
    /// Every Root value must be a live W_Root and every base a live prebuilt
    /// TypeDef. Definitions and their native root registrations are immortal.
    #[majit_macros::not_rpython]
    pub unsafe fn from_rawdict(
        name: &str,
        bases: Vec<*const TypeDef>,
        rawdict: indexmap::IndexMap<String, TypeDefValue>,
        instance_type: *const PyType,
    ) -> *const Self {
        let mut definition = Self::new(
            instance_type,
            rawdict.contains_key("__new__"),
            rawdict.contains_key("__dict__"),
        );
        definition.name = Some(name.to_string());
        assert!(
            !rawdict.contains_key("__del__"),
            "TypeDef requires an RPython finalizer"
        );
        definition.weakrefable = rawdict.contains_key("__weakref__");
        definition.doc = match rawdict.get("__doc__") {
            Some(TypeDefValue::Text(doc)) => Some(doc.clone()),
            _ => None,
        };
        for &base in &bases {
            // SAFETY: the constructor contract requires live prebuilt bases.
            unsafe {
                definition.hasdict |= (*base).hasdict;
                definition.weakrefable |= (*base).weakrefable;
                if definition.buffer.is_none() {
                    definition.buffer = (*base).buffer;
                }
            }
        }
        definition.bases = bases;
        // Publish before add_entries: TypeDef.add_entries names GetSetProperty
        // with w_str_new, and collect_static_in_prebuilt_nongc only walks
        // TypeDefs already in the census.
        let ptr = publish_typedef(definition);
        unsafe { (*(ptr as *mut TypeDef)).add_entries(rawdict) };
        ptr
    }

    /// TypeDef.add_entries: install declarations, then name descriptors.
    /// Remember incoming keys in order, then extend so a published TypeDef's
    /// Root cells are in the census before `w_str_new` (`GetSetProperty.name`).
    /// Name each incoming key by lookup: a replaced key keeps its original
    /// index, so walking only `start..len` would skip it.
    /// # Safety
    /// Root entries must contain valid W_Root objects; mutate declarations
    /// only during host initialization. `self` must already be published.
    pub unsafe fn add_entries(&mut self, entries: indexmap::IndexMap<String, TypeDefValue>) {
        let keys: Vec<String> = entries.keys().cloned().collect();
        self.rawdict.extend(entries);
        for key in keys {
            let value_slot = match self.rawdict.get(&key) {
                Some(TypeDefValue::Root(slot)) => slot.get(),
                _ => continue,
            };
            let value = unsafe { *value_slot };
            if unsafe { crate::gateway::is_interp2app(value) } {
                let gateway = unsafe { &mut *(value as *mut crate::gateway::interp2app) };
                gateway.name = Box::leak(key.into_boxed_str());
                gateway._is_type_method = true;
            } else if unsafe { is_getset_property(value) } {
                let w_name = crate::w_str_new(&key);
                let value = unsafe { *value_slot };
                unsafe { w_getset_set_name(value, w_name) };
            }
        }
    }
}

/// TypeDef is `@not_rpython` (`typedef.py`). After translation,
/// `collect_roots_in_nursery` walks static pointers in prebuilt non-gc
/// structs (`collect_static_in_prebuilt_nongc`). Native bootstrap has no
/// PBC section, so the census is the TypeDefs themselves.
///
/// `*const TypeDef` is `!Send`; these pointers are process-lifetime host
/// structs, the same addresses a PBC static-root table would name.
#[derive(Clone, Copy)]
struct TypeDefPtr(*const TypeDef);
unsafe impl Send for TypeDefPtr {}
unsafe impl Sync for TypeDefPtr {}

static TYPEDEFS: parking_lot::Mutex<Vec<TypeDefPtr>> = parking_lot::Mutex::new(Vec::new());

fn publish_typedef(definition: TypeDef) -> *const TypeDef {
    let ptr = crate::lltype::malloc_raw(definition);
    TYPEDEFS.lock().push(TypeDefPtr(ptr));
    crate::gc_roots::mark_prebuilt_roots_dirty();
    ptr
}

/// Trace `TypeDef.rawdict` Root values in place.
/// The visitor must not allocate or construct another TypeDef.
pub fn walk_typedef_roots(forward: &mut dyn FnMut(&mut PyObjectRef)) {
    for &TypeDefPtr(typedef) in TYPEDEFS.lock().iter() {
        unsafe {
            for value in (*typedef).rawdict.values() {
                if let TypeDefValue::Root(slot) = value {
                    forward(&mut *slot.get());
                }
            }
        }
    }
}

pub fn test_declaration_roots_len() -> usize {
    TYPEDEFS.lock().len()
}

pub fn test_last_declaration_slot() -> *mut PyObjectRef {
    let typedefs = TYPEDEFS.lock();
    let mut last = None;
    for &TypeDefPtr(typedef) in typedefs.iter() {
        unsafe {
            for value in (*typedef).rawdict.values() {
                if let TypeDefValue::Root(slot) = value {
                    last = Some(slot.get());
                }
            }
        }
    }
    last.expect("no TypeDef.rawdict Root in the TypeDef census")
}

pub fn test_truncate_declaration_roots(len: usize) {
    TYPEDEFS.lock().truncate(len);
}

/// Existing process-lifetime allocation for module-level `W_X.typedef`
/// metadata. Keep the allocation identity while its bootstrap owner migrates
/// to TypeDef.__init__ / TypeCache.build.
pub fn leak_typedef(
    instance_type: *const PyType,
    acceptable_as_base_class: bool,
    hasdict: bool,
) -> *const TypeDef {
    publish_typedef(TypeDef::new(
        instance_type,
        acceptable_as_base_class,
        hasdict,
    ))
}

/// `pypy/interpreter/typedef.py class GetSetProperty(W_Root)`.
///
/// All `PyObjectRef`-shaped slots default to `PY_NULL` to mark
/// "absent" (PyPy uses `None`); `use_closure` is a `bool` mirroring
/// the eponymous PyPy field.
///
/// `pytype_static = "GETSET_DESCRIPTOR_TYPE"` keeps the PyType under
/// its existing public name (`typedef.py GetSetProperty.typedef =
/// TypeDef("getset_descriptor", ...)`) while the GC consts stay on
/// the `W_GETSET_PROPERTY_*` convention.
///
/// `GetSetProperty._immutable_fields_ = ["fget", "fset", "fdel"]`.
#[pyre_class(
    "getset_descriptor",
    type_id = 40,
    static_name = "GETSET_PROPERTY",
    pytype_static = "GETSET_DESCRIPTOR_TYPE"
)]
pub struct GetSetProperty {
    /// `typedef.py self.fget` — getter callable.
    pub fget: PyObjectRef,
    /// `typedef.py:340 self.fset` — setter callable.
    pub fset: PyObjectRef,
    /// `typedef.py:341 self.fdel` — deleter callable.
    pub fdel: PyObjectRef,
    /// `typedef.py:342 self.doc` — wrapped docstring.
    pub doc: PyObjectRef,
    /// `typedef.py:343 self.reqcls` — required receiver class for
    /// `descr_self_interp_w` mismatch checking.
    pub reqcls: PyObjectRef,
    /// `typedef.py:346 self.name` — descriptor name (defaults to
    /// `'<generic property>'` when the caller passes None).
    pub name: PyObjectRef,
    /// `typedef.py w_objclass = None` class default + per-instance
    /// override stamped by `copy_for_type` (typedef.py).  Read by
    /// `descr_get_objclass` (typedef.py) before falling back
    /// to `space.gettypeobject(self.reqcls.typedef)`.
    pub w_objclass: PyObjectRef,
    /// `typedef.py self.w_qualname = None` — lazy cache for
    /// `descr_get_qualname` (typedef.py); first reader stamps
    /// `"<class>.<name>"` (or `"?.<name>"` when `reqcls is None`).
    pub w_qualname: PyObjectRef,
    /// `typedef.py:345 self.use_closure` — passes `(self, space, obj)`
    /// vs `(space, obj)` to the wrapped callbacks.
    pub use_closure: bool,
}

/// ULLBC harvest of `GetSetProperty._immutable_fields_ = ["fget", "fset", "fdel"]`.
/// `#[jit_immutable_fields]` cannot wrap a `#[pyre_class]` struct (both
/// consume the item), so this is the marker const that macro emits:
/// `harvest_immutable_fields_from_llbcs` reads `_immutable_fields_<Struct>`,
/// and `jitcode_lower` reads `__MAJIT_IMMUTABLE_FIELDS`.
#[doc(hidden)]
#[allow(non_upper_case_globals, dead_code)]
pub const _immutable_fields_GetSetProperty: &str = "fget,fset,fdel";

impl GetSetProperty {
    #[doc(hidden)]
    #[allow(dead_code)]
    pub const __MAJIT_IMMUTABLE_FIELDS: &str = "fget,fset,fdel";
}

/// Allocate a `GetSetProperty` bound to `GETSET_DESCRIPTOR_TYPE`.
/// Mirrors `typedef.py _init` / `instantiate(GetSetProperty)` —
/// every slot is set in one shot so the descriptor is fully
/// initialised before the first reader.
///
/// Young-nonmoving (`malloc_fixedsize`): `copy_for_type` writes
/// `new.w_objclass = w_objclass` on this instance, and
/// `create_dict_slot` / `create_weakref_slot` store it in the
/// type dict. An old-gen copy enters the remembered set and
/// promotes the young type on every minor, so `add_subclass`
/// never reuses a dead `weak_subclasses` slot.
///
/// `name` may be `PY_NULL`, in which case the caller is responsible
/// for substituting `'<generic property>'` (matching `typedef.py
/// self.name = name if name is not None else '<generic property>'`);
/// pyre's call sites pass an already-resolved name to keep the
/// allocation hot path branchless.
pub fn w_getset_property_new(
    fget: PyObjectRef,
    fset: PyObjectRef,
    fdel: PyObjectRef,
    doc: PyObjectRef,
    reqcls: PyObjectRef,
    use_closure: bool,
    name: PyObjectRef,
) -> PyObjectRef {
    GetSetProperty::allocate(GetSetProperty {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        fget,
        fset,
        fdel,
        doc,
        reqcls,
        name,
        w_objclass: PY_NULL,
        w_qualname: PY_NULL,
        use_closure,
    })
}

/// `typedef.py GetSetProperty.copy_for_type`: `instantiate(GetSetProperty)`.
///
/// The TypeDef template (`w_getset_property_new`) stays on
/// `malloc_typed_stable` so its address can live in `TYPEDEF_VALUE_ROOTS`.
/// The copy is a regular nursery `W_Root` stored only in the heap type's
/// dict; born-old copies joined `old_objects_pointing_to_young` at the
/// creation barrier (`malloc_typed_stable`) and then received `w_objclass`
/// without a second barrier, so a discarded `type(name, (), {})` stayed
/// reachable through a dead GetSet at every minor (`W_TypeObject.add_subclass`
/// never reused the slot).
pub fn w_getset_property_instantiate(
    fget: PyObjectRef,
    fset: PyObjectRef,
    fdel: PyObjectRef,
    doc: PyObjectRef,
    reqcls: PyObjectRef,
    use_closure: bool,
    name: PyObjectRef,
    w_objclass: PyObjectRef,
) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    let save = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_roots(&[fget, fset, fdel, doc, reqcls, name, w_objclass]);
    let mut allocation_root: crate::gc_hook::GCREF = std::ptr::null_mut();
    let mut needs_write_barrier = true;
    let raw = crate::gc_hook::GcAllocOutcome::from_hook(unsafe {
        crate::gc_hook::try_gc_alloc_collecting_rooted(
            W_GETSET_PROPERTY_GC_TYPE_ID,
            W_GETSET_PROPERTY_OBJECT_SIZE,
            &mut allocation_root,
            &mut needs_write_barrier,
        )
    })
    .allocated_or_abort(W_GETSET_PROPERTY_OBJECT_SIZE)
    .unwrap_or(std::ptr::null_mut());
    let value = GetSetProperty {
        ob: PyObject {
            ob_type: &GETSET_DESCRIPTOR_TYPE as *const PyType,
            w_class: crate::pyobject::get_instantiate(&GETSET_DESCRIPTOR_TYPE),
        },
        fget: crate::gc_roots::shadow_stack_get(save),
        fset: crate::gc_roots::shadow_stack_get(save + 1),
        fdel: crate::gc_roots::shadow_stack_get(save + 2),
        doc: crate::gc_roots::shadow_stack_get(save + 3),
        reqcls: crate::gc_roots::shadow_stack_get(save + 4),
        name: crate::gc_roots::shadow_stack_get(save + 5),
        w_objclass: crate::gc_roots::shadow_stack_get(save + 6),
        w_qualname: PY_NULL,
        use_closure,
    };
    if raw.is_null() {
        return crate::lltype::malloc_typed(value) as PyObjectRef;
    }
    unsafe {
        std::ptr::write(raw as *mut GetSetProperty, value);
        if needs_write_barrier {
            crate::gc_hook::try_gc_write_barrier_managed(raw);
        }
    }
    raw as PyObjectRef
}

/// Test whether `obj` is a `GetSetProperty`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_getset_property(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &GETSET_DESCRIPTOR_TYPE) }
}

/// # Safety
/// `obj` must point to a valid `GetSetProperty`.
#[inline]
pub unsafe fn w_getset_get_fget(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const GetSetProperty)).fget }
}

/// # Safety
/// `obj` must point to a valid `GetSetProperty`.
#[inline]
pub unsafe fn w_getset_get_fset(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const GetSetProperty)).fset }
}

/// # Safety
/// `obj` must point to a valid `GetSetProperty`.
#[inline]
pub unsafe fn w_getset_get_fdel(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const GetSetProperty)).fdel }
}

/// # Safety
/// `obj` must point to a valid `GetSetProperty`.
#[inline]
pub unsafe fn w_getset_get_reqcls(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const GetSetProperty)).reqcls }
}

/// # Safety
/// `obj` must point to a valid `GetSetProperty`.
#[inline]
pub unsafe fn w_getset_get_name(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const GetSetProperty)).name }
}

/// `typedef.py add_entries` parity — overwrite the descriptor's
/// `name` slot with the dict-key it was registered under.  Used by
/// the post-init namespace walker so descriptors built without an
/// explicit name (most `make_getset_descriptor` callers) carry the
/// matching `__name__` instead of the `<generic property>` sentinel.
///
/// # Safety
/// `obj` must point to a valid `GetSetProperty`.
#[inline]
pub unsafe fn w_getset_set_name(obj: PyObjectRef, value: PyObjectRef) {
    unsafe { (*(obj as *mut GetSetProperty)).name = value }
}

/// `typedef.py self.reqcls = cls` — write the required-receiver
/// class slot.  Used by `patch_builtin_function_descriptors` to
/// install the BuiltinFunction class onto the shared
/// `__self__`/`__doc__` GetSetProperty descriptors after the
/// W_TypeObject for BuiltinFunction is materialised.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_getset_set_reqcls(obj: PyObjectRef, value: PyObjectRef) {
    unsafe { (*(obj as *mut GetSetProperty)).reqcls = value }
}

/// # Safety
/// `obj` must point to a valid `GetSetProperty`.
#[inline]
pub unsafe fn w_getset_get_doc(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const GetSetProperty)).doc }
}

/// `typedef.py / 348-356 copy_for_type` writes `new.w_objclass`.
/// Pyre keeps the slot directly on the struct so the descriptor's
/// `descr_get_objclass` reads it without any side-table.
///
/// # Safety
/// `obj` must point to a valid `GetSetProperty`.
#[inline]
pub unsafe fn w_getset_get_objclass(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const GetSetProperty)).w_objclass }
}

/// # Safety
/// `obj` must point to a valid `GetSetProperty`.
#[inline]
pub unsafe fn w_getset_set_objclass(obj: PyObjectRef, value: PyObjectRef) {
    unsafe { (*(obj as *mut GetSetProperty)).w_objclass = value }
    if crate::gc_hook::try_gc_owns_object(obj as crate::gc_hook::GCREF) {
        // GC-owned: traced through the type-id offsets. A young
        // descriptor ignores the barrier; an old-gen fallback joins
        // the remembered set so `w_objclass` is forwarded.
        crate::gc_hook::try_gc_write_barrier_managed(obj as crate::gc_hook::GCREF);
    } else {
        // Immortal template reached only by `walk_raw_getset_roots`,
        // skipped on a clean minor; record the store.
        crate::gc_roots::mark_prebuilt_roots_dirty();
    }
}

/// `typedef.py self.w_qualname = None` — lazy cache slot for
/// `descr_get_qualname` (typedef.py).
///
/// # Safety
/// `obj` must point to a valid `GetSetProperty`.
#[inline]
pub unsafe fn w_getset_get_qualname(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const GetSetProperty)).w_qualname }
}

/// # Safety
/// `obj` must point to a valid `GetSetProperty`.
#[inline]
pub unsafe fn w_getset_set_qualname(obj: PyObjectRef, value: PyObjectRef) {
    unsafe { (*(obj as *mut GetSetProperty)).w_qualname = value }
    if crate::gc_hook::try_gc_owns_object(obj as crate::gc_hook::GCREF) {
        crate::gc_hook::try_gc_write_barrier_managed(obj as crate::gc_hook::GCREF);
    } else {
        crate::gc_roots::mark_prebuilt_roots_dirty();
    }
}

/// `typedef.py:345 self.use_closure` — read-only accessor.
///
/// # Safety
/// `obj` must point to a valid `GetSetProperty`.
#[inline]
pub unsafe fn w_getset_get_use_closure(obj: PyObjectRef) -> bool {
    unsafe { (*(obj as *const GetSetProperty)).use_closure }
}

/// `pypy/interpreter/typedef.py Member` — slot descriptor
/// for `__slots__`.
///
/// A Member descriptor provides attribute access to a specific
/// `__slots__` entry. In PyPy, slots are stored at fixed offsets in
/// the object struct; in pyre, instance attributes are stored in a
/// dict, so the Member acts as a marker and accessor by name.
///
/// The macro skips the non-PyObjectRef `index` (u32) and `name`
/// (`*const String`) fields when emitting GC pointer offsets — only
/// `w_cls` is traced.
#[pyre_class("member_descriptor", type_id = 26, static_name = "MEMBER")]
pub struct W_MemberDescr {
    /// Slot index (base_nslots + position in newslotnames).
    pub index: u32,
    /// Slot name (owned, leaked).
    pub name: *const String,
    /// Owning type object (for typecheck).
    pub w_cls: PyObjectRef,
    /// `PyMemberDef.doc` for native member descriptors.  Like CPython's
    /// static C string, this is non-GC metadata; Python `__slots__` members
    /// leave it null.
    pub doc: *const String,
}

/// Python 3.14's function type exposes five direct `PyMemberDef` entries.
/// PyPy represents the same values with GetSetProperty, while ordinary PyPy
/// `Member` objects use `index` for `__slots__`.  Reserve the high bit so the
/// existing slot-index shape stays intact and the interpreter can distinguish
/// the 3.14 direct members without a side table.
pub const MEMBER_DIRECT_FLAG: u32 = 1 << 31;
pub const MEMBER_FUNCTION_CLOSURE: u32 = MEMBER_DIRECT_FLAG;
pub const MEMBER_FUNCTION_DOC: u32 = MEMBER_DIRECT_FLAG | 1;
pub const MEMBER_FUNCTION_GLOBALS: u32 = MEMBER_DIRECT_FLAG | 2;
pub const MEMBER_FUNCTION_MODULE: u32 = MEMBER_DIRECT_FLAG | 3;
pub const MEMBER_FUNCTION_BUILTINS: u32 = MEMBER_DIRECT_FLAG | 4;
/// CPython 3.14 `module_members`: the authoritative Module.w_dict field.
pub const MEMBER_MODULE_DICT: u32 = MEMBER_DIRECT_FLAG | 5;
/// CPython 3.14 `complex_members`: `Py_T_DOUBLE`, `Py_READONLY`.
pub const MEMBER_COMPLEX_REAL: u32 = MEMBER_DIRECT_FLAG | 6;
pub const MEMBER_COMPLEX_IMAG: u32 = MEMBER_DIRECT_FLAG | 7;
/// `descrobject.c descr_members`, shared by every descriptor type: the owning
/// class (`PyDescrObject.d_type`) and the attribute name (`d_name`), both
/// read-only.  PyPy publishes the same two values as GetSetProperty
/// (`typedef.py`, `:538-539`); the descriptor kind is the 3.14
/// difference.  The descriptor payloads here — GetSetProperty, Member and the
/// Function carrier — do not share a header, so the reader dispatches on the
/// receiver instead of reading one fixed offset.
pub const MEMBER_DESCR_OBJCLASS: u32 = MEMBER_DIRECT_FLAG | 8;
pub const MEMBER_DESCR_NAME: u32 = MEMBER_DIRECT_FLAG | 9;
/// CPython 3.14 `BaseExceptionGroup_members`: the immutable constructor
/// message and tuple of nested exceptions.
pub const MEMBER_EXCEPTION_GROUP_MESSAGE: u32 = MEMBER_DIRECT_FLAG | 10;
pub const MEMBER_EXCEPTION_GROUP_EXCEPTIONS: u32 = MEMBER_DIRECT_FLAG | 11;
/// CPython 3.14 `SyntaxError_members._metadata`, a writable private slot.
pub const MEMBER_SYNTAX_ERROR_METADATA: u32 = MEMBER_DIRECT_FLAG | 12;
/// CPython 3.14 `StopIteration.value`, backed by PyPy's `w_value` field.
pub const MEMBER_STOP_ITERATION_VALUE: u32 = MEMBER_DIRECT_FLAG | 13;
/// CPython 3.14 `BaseException.__suppress_context__` boolean member.
pub const MEMBER_EXCEPTION_SUPPRESS_CONTEXT: u32 = MEMBER_DIRECT_FLAG | 14;
/// CPython 3.14 `AttributeError_members.name`, backed by PyPy's `w_name`.
pub const MEMBER_ATTRIBUTE_ERROR_NAME: u32 = MEMBER_DIRECT_FLAG | 15;
/// CPython 3.14 `AttributeError_members.obj`, backed by PyPy's `w_obj`.
pub const MEMBER_ATTRIBUTE_ERROR_OBJ: u32 = MEMBER_DIRECT_FLAG | 16;
/// CPython 3.14 `NameError_members.name`, backed by PyPy's `w_name`.
pub const MEMBER_NAME_ERROR_NAME: u32 = MEMBER_DIRECT_FLAG | 17;
/// CPython 3.14 `ImportError_members.msg`, backed by PyPy's `w_msg`.
pub const MEMBER_IMPORT_ERROR_MSG: u32 = MEMBER_DIRECT_FLAG | 18;
/// CPython 3.14 `ImportError_members.name`, backed by PyPy's `w_name`.
pub const MEMBER_IMPORT_ERROR_NAME: u32 = MEMBER_DIRECT_FLAG | 19;
/// CPython 3.14 `ImportError_members.name_from`, a pyre-flattened PyPy field.
pub const MEMBER_IMPORT_ERROR_NAME_FROM: u32 = MEMBER_DIRECT_FLAG | 20;
/// CPython 3.14 `ImportError_members.path`, backed by PyPy's `w_path`.
pub const MEMBER_IMPORT_ERROR_PATH: u32 = MEMBER_DIRECT_FLAG | 21;
/// CPython 3.14 `OSError_members.errno`, backed by PyPy's `w_errno`.
pub const MEMBER_OS_ERROR_ERRNO: u32 = MEMBER_DIRECT_FLAG | 22;
/// CPython 3.14 `OSError_members.strerror`, backed by PyPy's `w_strerror`.
pub const MEMBER_OS_ERROR_STRERROR: u32 = MEMBER_DIRECT_FLAG | 23;
/// CPython 3.14 `OSError_members.filename`, backed by PyPy's `w_filename`.
pub const MEMBER_OS_ERROR_FILENAME: u32 = MEMBER_DIRECT_FLAG | 24;
/// CPython 3.14 `OSError_members.filename2`, backed by PyPy's `w_filename2`.
pub const MEMBER_OS_ERROR_FILENAME2: u32 = MEMBER_DIRECT_FLAG | 25;
/// CPython 3.14 `SystemExit_members.code`, backed by PyPy's `w_code`.
pub const MEMBER_SYSTEM_EXIT_CODE: u32 = MEMBER_DIRECT_FLAG | 26;
/// CPython 3.14 `SyntaxError_members.msg`, backed by PyPy's `w_msg`.
pub const MEMBER_SYNTAX_ERROR_MSG: u32 = MEMBER_DIRECT_FLAG | 27;
/// CPython 3.14 `SyntaxError_members.filename`, backed by PyPy's `w_filename`.
pub const MEMBER_SYNTAX_ERROR_FILENAME: u32 = MEMBER_DIRECT_FLAG | 28;
/// CPython 3.14 `SyntaxError_members.lineno`, backed by PyPy's `w_lineno`.
pub const MEMBER_SYNTAX_ERROR_LINENO: u32 = MEMBER_DIRECT_FLAG | 29;
/// CPython 3.14 `SyntaxError_members.offset`, backed by PyPy's `w_offset`.
pub const MEMBER_SYNTAX_ERROR_OFFSET: u32 = MEMBER_DIRECT_FLAG | 30;
/// CPython 3.14 `SyntaxError_members.text`, backed by PyPy's `w_text`.
pub const MEMBER_SYNTAX_ERROR_TEXT: u32 = MEMBER_DIRECT_FLAG | 31;
/// CPython 3.14 `SyntaxError_members.end_lineno`, backed by PyPy's field.
pub const MEMBER_SYNTAX_ERROR_END_LINENO: u32 = MEMBER_DIRECT_FLAG | 32;
/// CPython 3.14 `SyntaxError_members.end_offset`, backed by PyPy's field.
pub const MEMBER_SYNTAX_ERROR_END_OFFSET: u32 = MEMBER_DIRECT_FLAG | 33;
/// CPython 3.14 `SyntaxError_members.print_file_and_line`, backed by PyPy's field.
pub const MEMBER_SYNTAX_ERROR_PRINT_FILE_AND_LINE: u32 = MEMBER_DIRECT_FLAG | 34;
/// CPython 3.14 Unicode*Error `encoding`, backed by PyPy's `w_encoding`.
pub const MEMBER_UNICODE_ERROR_ENCODING: u32 = MEMBER_DIRECT_FLAG | 35;
/// CPython 3.14 Unicode*Error `object`, backed by PyPy's `w_object`.
pub const MEMBER_UNICODE_ERROR_OBJECT: u32 = MEMBER_DIRECT_FLAG | 36;
/// CPython 3.14 Unicode*Error `start`, backed by PyPy's `w_start`.
pub const MEMBER_UNICODE_ERROR_START: u32 = MEMBER_DIRECT_FLAG | 37;
/// CPython 3.14 Unicode*Error `end`, backed by PyPy's `w_end`.
pub const MEMBER_UNICODE_ERROR_END: u32 = MEMBER_DIRECT_FLAG | 38;
/// CPython 3.14 Unicode*Error `reason`, backed by PyPy's `w_reason`.
pub const MEMBER_UNICODE_ERROR_REASON: u32 = MEMBER_DIRECT_FLAG | 39;
/// CPython 3.14 `staticmethod_members`: both `__func__` and `__wrapped__`
/// expose PyPy's `StaticMethod.w_function` field read-only.
pub const MEMBER_STATICMETHOD_FUNCTION: u32 = MEMBER_DIRECT_FLAG | 40;
/// CPython 3.14 `classmethod_members`: both `__func__` and `__wrapped__`
/// expose PyPy's `ClassMethod.w_function` field read-only.
pub const MEMBER_CLASSMETHOD_FUNCTION: u32 = MEMBER_DIRECT_FLAG | 41;
/// CPython 3.14 `property_members.fget`, backed by PyPy's `w_fget` field.
pub const MEMBER_PROPERTY_FGET: u32 = MEMBER_DIRECT_FLAG | 42;
/// CPython 3.14 `property_members.fset`, backed by PyPy's `w_fset` field.
pub const MEMBER_PROPERTY_FSET: u32 = MEMBER_DIRECT_FLAG | 43;
/// CPython 3.14 `property_members.fdel`, backed by PyPy's `w_fdel` field.
pub const MEMBER_PROPERTY_FDEL: u32 = MEMBER_DIRECT_FLAG | 44;
/// CPython 3.14 `property_members.__doc__`, the writable `w_doc` field.
pub const MEMBER_PROPERTY_DOC: u32 = MEMBER_DIRECT_FLAG | 45;
/// CPython 3.14 `range_members.start`, backed by PyPy's `w_start` field.
pub const MEMBER_RANGE_START: u32 = MEMBER_DIRECT_FLAG | 46;
/// CPython 3.14 `range_members.stop`, backed by PyPy's `w_stop` field.
pub const MEMBER_RANGE_STOP: u32 = MEMBER_DIRECT_FLAG | 47;
/// CPython 3.14 `range_members.step`, backed by PyPy's `w_step` field.
pub const MEMBER_RANGE_STEP: u32 = MEMBER_DIRECT_FLAG | 48;
/// CPython 3.14 `slice_members.start`, backed by PyPy's `w_start` field.
pub const MEMBER_SLICE_START: u32 = MEMBER_DIRECT_FLAG | 49;
/// CPython 3.14 `slice_members.stop`, backed by PyPy's `w_stop` field.
pub const MEMBER_SLICE_STOP: u32 = MEMBER_DIRECT_FLAG | 50;
/// CPython 3.14 `slice_members.step`, backed by PyPy's `w_step` field.
pub const MEMBER_SLICE_STEP: u32 = MEMBER_DIRECT_FLAG | 51;
/// CPython 3.14 `super_members.__thisclass__`, backed by PyPy's `w_starttype`.
pub const MEMBER_SUPER_THISCLASS: u32 = MEMBER_DIRECT_FLAG | 52;
/// CPython 3.14 `super_members.__self__`, backed by PyPy's `w_self`.
pub const MEMBER_SUPER_SELF: u32 = MEMBER_DIRECT_FLAG | 53;
/// CPython 3.14 `super_members.__self_class__`, backed by PyPy's `w_objtype`.
pub const MEMBER_SUPER_SELF_CLASS: u32 = MEMBER_DIRECT_FLAG | 54;
/// CPython 3.14 `OSError_members.winerror`, backed by PyPy's `w_winerror`.
/// Declared on every platform, registered on `OSError` only where the
/// platform has Windows error codes (`interp_exceptions.py:723-728`).
pub const MEMBER_OS_ERROR_WINERROR: u32 = MEMBER_DIRECT_FLAG | 61;
/// CPython 3.14 `tb_memberlist.tb_frame`, backed by PyPy's
/// `interp_attrproperty_w('frame')`.  Read-only on both sides.
pub const MEMBER_TRACEBACK_FRAME: u32 = MEMBER_DIRECT_FLAG | 62;
/// CPython 3.14 `tb_memberlist.tb_lasti`, backed by PyPy's `lasti` field.
/// `Py_READONLY` there, writable through `descr_set_tb_lasti` in PyPy; the
/// setter is not wired here, for the reason `init_pytraceback_type` records.
pub const MEMBER_TRACEBACK_LASTI: u32 = MEMBER_DIRECT_FLAG | 63;
/// `instancemethod.__func__`, the callable stored by `InstanceMethod`.
pub const MEMBER_INSTANCEMETHOD_FUNCTION: u32 = MEMBER_DIRECT_FLAG | 64;

/// Create a new Member descriptor.
pub fn w_member_new(index: u32, name: String, w_cls: PyObjectRef) -> PyObjectRef {
    w_member_new_with_doc(index, name, None, w_cls)
}

/// Create a Member descriptor with CPython `PyMemberDef.doc` metadata.
pub fn w_member_new_with_doc(
    index: u32,
    name: String,
    doc: Option<String>,
    w_cls: PyObjectRef,
) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py`).
    let _roots = crate::gc_roots::push_roots();
    let root_base = crate::gc_roots::shadow_stack_len();
    let _ = crate::gc_roots::pin_root(w_cls);
    let name = crate::lltype::malloc_raw(name);
    let doc = doc.map_or(std::ptr::null(), |doc| {
        crate::lltype::malloc_raw(doc) as *const String
    });
    let w_cls = crate::gc_roots::shadow_stack_get(root_base);
    // Managed (`allocate_stable`), not the movable `malloc_typed` immortal a
    // bare `allocate` would give: a `__slots__` member outlives the statement
    // that created it (`d = C.x` keeps the descriptor after `C` is dropped),
    // and `w_cls` is then its only reference to the owning type.  An immortal
    // is outside the collector's sweep set, so the marker takes GCFLAG_VISITED
    // on it during the first major and never clears it again; from the second
    // major on it is skipped, `w_cls` is never re-marked, and the type is swept
    // while the descriptor still points at it.  A stable managed allocation
    // keeps the address fixed for the raw `*mut W_MemberDescr` accessors below
    // while putting the object under the ordinary mark-and-clear cycle.
    W_MemberDescr::allocate_stable(W_MemberDescr {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        index,
        name,
        doc,
        w_cls,
    })
}

/// Create one of Python 3.14's direct function member descriptors.
pub fn w_member_new_direct(kind: u32, name: String, w_cls: PyObjectRef) -> PyObjectRef {
    debug_assert_ne!(kind & MEMBER_DIRECT_FLAG, 0);
    w_member_new(kind, name, w_cls)
}

/// Create a Python 3.14 direct member descriptor with `PyMemberDef.doc`.
pub fn w_member_new_direct_with_doc(
    kind: u32,
    name: String,
    doc: String,
    w_cls: PyObjectRef,
) -> PyObjectRef {
    debug_assert_ne!(kind & MEMBER_DIRECT_FLAG, 0);
    w_member_new_with_doc(kind, name, Some(doc), w_cls)
}

/// Check if an object is a Member descriptor.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_member(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &MEMBER_TYPE) }
}

/// Get the Member's slot name.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_member_get_name(obj: PyObjectRef) -> &'static str {
    unsafe { &*(*(obj as *const W_MemberDescr)).name }
}

/// Get the optional CPython `PyMemberDef.doc`.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_member_get_doc(obj: PyObjectRef) -> Option<&'static str> {
    let doc = unsafe { (*(obj as *const W_MemberDescr)).doc };
    if doc.is_null() {
        None
    } else {
        Some(unsafe { &*doc })
    }
}

/// Get the Member's owning class.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_member_get_cls(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_MemberDescr)).w_cls }
}

/// Fill a descriptor owner after the built-in type registry is published.
///
/// Two remembered-set notifications, because the descriptor can be either
/// shape: the prebuilt-family bit covers a descriptor that predates the
/// managed-allocation hooks and so fell back to `malloc_typed`, and the write
/// barrier covers the ordinary `allocate_stable` case, where an old descriptor
/// gaining a young or unmarked `w_cls` must re-enter the collector's worklist.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_member_set_cls(obj: PyObjectRef, w_cls: PyObjectRef) {
    crate::gc_roots::mark_prebuilt_roots_dirty();
    unsafe { (*(obj as *mut W_MemberDescr)).w_cls = w_cls };
    crate::gc_hook::try_gc_write_barrier(obj as crate::gc_hook::GCREF);
}

/// `typedef.py Member.index` — the slot index (`base_nslots + position`),
/// used by the LOAD_ATTR/STORE_ATTR cache to form the `SLOTS_STARTING_FROM +
/// index` attrkind (mapdict.py).
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_member_get_index(obj: PyObjectRef) -> u32 {
    unsafe { (*(obj as *const W_MemberDescr)).index }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_member_is_direct(obj: PyObjectRef) -> bool {
    unsafe { w_member_get_index(obj) & MEMBER_DIRECT_FLAG != 0 }
}

#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_member_get_direct_kind(obj: PyObjectRef) -> u32 {
    let kind = unsafe { w_member_get_index(obj) };
    debug_assert_ne!(kind & MEMBER_DIRECT_FLAG, 0);
    kind
}

#[cfg(test)]
mod tests {
    #[test]
    fn declaration_metadata_distinguishes_host_text_and_wrapped_values() {
        use super::*;
        use indexmap::IndexMap;

        let _restore = RestoreTypedefCensus::enter();
        unsafe {
            let base = TypeDef::from_rawdict(
                "base",
                vec![],
                IndexMap::from([
                    ("__dict__".into(), TypeDefValue::None),
                    ("__weakref__".into(), TypeDefValue::None),
                    (
                        "__doc__".into(),
                        TypeDefValue::Text("host documentation".into()),
                    ),
                ]),
                &INSTANCE_TYPE,
            );
            assert_eq!((*base).doc.as_deref(), Some("host documentation"));
            assert!(!(*base).acceptable_as_base_class());
            let child = TypeDef::from_rawdict(
                "child",
                vec![base],
                IndexMap::from([
                    ("__new__".into(), TypeDefValue::None),
                    (
                        "__doc__".into(),
                        TypeDefValue::root(crate::w_str_new("wrapped")),
                    ),
                ]),
                &INSTANCE_TYPE,
            );
            assert_eq!((*child).name.as_deref(), Some("child"));
            assert_eq!((*child).bases, vec![base]);
            assert!((*child).hasdict && (*child).weakrefable);
            assert!((*child).acceptable_as_base_class());
            assert_eq!((*child).doc, None);
            assert_eq!(
                (*child)
                    .rawdict
                    .keys()
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
                vec!["__new__", "__doc__"]
            );
            // add_entries updates declarations; it does not recompute the
            // metadata derived during TypeDef.__init__.
            let child = &mut *(child as *mut TypeDef);
            child.add_entries(IndexMap::from([(
                "__doc__".into(),
                TypeDefValue::Text("later".into()),
            )]));
            assert_eq!(child.doc, None);
        }
    }

    #[test]
    fn add_entries_names_replaced_interp2app_and_getset() {
        use super::*;
        use indexmap::IndexMap;

        let _restore = RestoreTypedefCensus::enter();
        unsafe {
            let definition =
                TypeDef::from_rawdict("replace_probe", vec![], IndexMap::new(), &INSTANCE_TYPE);
            let definition = &mut *(definition as *mut TypeDef);

            let first_app = crate::gateway::interp2app::new(PY_NULL, "old_app");
            let first_gs =
                w_getset_property_new(PY_NULL, PY_NULL, PY_NULL, PY_NULL, PY_NULL, false, PY_NULL);
            definition.add_entries(IndexMap::from([
                ("method".into(), TypeDefValue::root(first_app)),
                ("prop".into(), TypeDefValue::root(first_gs)),
            ]));

            let second_app = crate::gateway::interp2app::new(PY_NULL, "other_app");
            let second_gs =
                w_getset_property_new(PY_NULL, PY_NULL, PY_NULL, PY_NULL, PY_NULL, false, PY_NULL);
            definition.add_entries(IndexMap::from([
                ("method".into(), TypeDefValue::root(second_app)),
                ("prop".into(), TypeDefValue::root(second_gs)),
            ]));

            let method = match definition.rawdict.get("method") {
                Some(TypeDefValue::Root(slot)) => *slot.get(),
                _ => panic!("expected Root slot method"),
            };
            assert_eq!(method, second_app);
            let gateway = &*(method as *const crate::gateway::interp2app);
            assert_eq!(gateway.name, "method");
            assert!(gateway._is_type_method);

            let prop = match definition.rawdict.get("prop") {
                Some(TypeDefValue::Root(slot)) => *slot.get(),
                _ => panic!("expected Root slot prop"),
            };
            assert_eq!(prop, second_gs);
            let name = w_getset_get_name(prop);
            assert_eq!(crate::w_str_get_value_opt(name), Some("prop"));
        }
    }

    #[test]
    fn method_descriptor_is_projected_from_the_selected_layout_owner() {
        use super::*;
        use crate::typeobject::*;

        let _restore = RestoreTypedefCensus::enter();
        let mut definition = TypeDef::new(&INSTANCE_TYPE, false, false);
        assert!(!definition.method_descriptor);
        definition.method_descriptor = true;
        let definition = crate::lltype::malloc_raw(definition);
        let layout = leak_layout(Layout {
            typedef: definition,
            nslots: 0,
            newslotnames: vec![],
            base_layout: std::ptr::null(),
            dict_data_slot: DICT_DATA_SLOT_UNRESOLVED,
        });
        let plain = leak_layout(Layout {
            typedef: leak_typedef(&INSTANCE_TYPE, true, false),
            nslots: 0,
            newslotnames: vec![],
            base_layout: std::ptr::null(),
            dict_data_slot: DICT_DATA_SLOT_UNRESOLVED,
        });
        unsafe {
            // Neither spelling nor the initial flag selects the behavior.
            let w_type = w_type_new("not_a_descriptor_name", PY_NULL, std::ptr::null_mut());
            assert!(!w_type_get_flag_method_descriptor(w_type));
            w_type_set_layout(w_type, layout);
            assert!(w_type_get_flag_method_descriptor(w_type));
            w_type_set_layout(w_type, plain);
            assert!(!w_type_get_flag_method_descriptor(w_type));
        }
    }

    #[test]
    fn typedef_identity_is_preserved_by_layout_and_freeze() {
        use super::{TypeDef, leak_typedef};
        use crate::typeobject::{DICT_DATA_SLOT_UNRESOLVED, Layout, leak_layout};

        let _restore = RestoreTypedefCensus::enter();
        let definition: *const TypeDef = leak_typedef(&crate::pyobject::INSTANCE_TYPE, true, true);
        let root = leak_layout(Layout {
            typedef: definition,
            nslots: 0,
            newslotnames: vec![],
            base_layout: std::ptr::null(),
            dict_data_slot: DICT_DATA_SLOT_UNRESOLVED,
        });
        let child = leak_layout(Layout {
            typedef: definition,
            nslots: 1,
            newslotnames: vec!["x".into()],
            base_layout: root,
            dict_data_slot: DICT_DATA_SLOT_UNRESOLVED,
        });
        unsafe {
            assert!(std::ptr::eq((*root).typedef, (*child).typedef));
            assert!((*definition)._freeze_());
            (*definition).set_acceptable_as_base_class(false);
            assert!(!(*(*root).typedef).acceptable_as_base_class());
            assert!(!(*(*child).typedef).acceptable_as_base_class());
            assert!((*(*child).typedef).hasdict);
            assert!((*child).issublayout(root));
        }
    }

    use super::*;

    #[test]
    fn w_member_gc_type_id_matches_descr() {
        assert_eq!(W_MEMBER_GC_TYPE_ID, 26);
        assert_eq!(
            <W_MemberDescr as crate::lltype::GcType>::type_id(),
            W_MEMBER_GC_TYPE_ID
        );
        assert_eq!(
            <W_MemberDescr as crate::lltype::GcType>::SIZE,
            W_MEMBER_OBJECT_SIZE
        );
    }

    static TYPEDEF_CENSUS_TEST_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

    struct RestoreTypedefCensus {
        _guard: parking_lot::MutexGuard<'static, ()>,
        len: usize,
    }

    impl RestoreTypedefCensus {
        fn enter() -> Self {
            let guard = TYPEDEF_CENSUS_TEST_LOCK.lock();
            let len = test_declaration_roots_len();
            Self { _guard: guard, len }
        }
    }

    impl Drop for RestoreTypedefCensus {
        fn drop(&mut self) {
            test_truncate_declaration_roots(self.len);
        }
    }

    #[test]
    fn leak_typedef_and_from_rawdict_register_the_host_struct() {
        use indexmap::IndexMap;

        let _restore = RestoreTypedefCensus::enter();
        let before = test_declaration_roots_len();
        let _layout_only = leak_typedef(&INSTANCE_TYPE, false, false);
        assert!(test_declaration_roots_len() >= before + 1);
        unsafe {
            let _declared =
                TypeDef::from_rawdict("census_probe", vec![], IndexMap::new(), &INSTANCE_TYPE);
        }
        assert!(test_declaration_roots_len() >= before + 2);
    }

    #[test]
    fn walk_typedef_roots_forwards_rawdict_in_place() {
        let _restore = RestoreTypedefCensus::enter();
        let marker = 0x0000_0000_0DEF_0009 as PyObjectRef;
        let forwarded = 0x0000_0000_0DEF_00F1 as PyObjectRef;
        let definition = leak_typedef(&INSTANCE_TYPE, false, false);
        unsafe {
            (*(definition as *mut TypeDef))
                .rawdict
                .insert("slot".into(), TypeDefValue::root(marker));
        }
        let mut seen = Vec::new();
        walk_typedef_roots(&mut |slot| seen.push(*slot as usize));
        assert!(
            seen.contains(&(marker as usize)),
            "walk_typedef_roots must visit TypeDef.rawdict Root values"
        );
        walk_typedef_roots(&mut |slot| {
            if *slot == marker {
                *slot = forwarded;
            }
        });
        unsafe {
            match (*definition).rawdict.get("slot") {
                Some(TypeDefValue::Root(slot)) => assert_eq!(*slot.get(), forwarded),
                _ => panic!("expected Root slot"),
            }
        }
    }
}
