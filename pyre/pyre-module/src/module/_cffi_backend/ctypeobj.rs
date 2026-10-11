//! `_cffi_backend.CType` — PyPy: `pypy/module/_cffi_backend/ctypeobj.py`
//! and every RPython subclass of `W_CType` (`ctypevoid`, `ctypeprim`,
//! `ctypeptr`, `ctypearray`).
//!
//! `W_CType.typedef.acceptable_as_base_class = False` and none of the
//! subclasses declares a typedef of its own, so the whole family is one
//! Python type.  What the RPython hierarchy expresses as a class, this
//! module expresses as [`W_CType::kind`]: the subclass is data, and the
//! methods it overrides are the `match` arms of the free functions below.

use pyre_interpreter::PyError;
use pyre_object::PyObjectRef;

use super::cdataobj::W_CData;

// ── the subclass discriminant ───────────────────────────────────────────

/// `ctypevoid.py W_CTypeVoid`.
pub const KIND_VOID: i64 = 0;
/// `ctypeprim.py W_CTypePrimitiveChar`.
pub const KIND_PRIM_CHAR: i64 = 1;
/// `ctypeprim.py W_CTypePrimitiveUniChar`.
pub const KIND_PRIM_UNICHAR: i64 = 2;
/// `ctypeprim.py W_CTypePrimitiveSigned`.
pub const KIND_PRIM_SIGNED: i64 = 3;
/// `ctypeprim.py W_CTypePrimitiveUnsigned`.
pub const KIND_PRIM_UNSIGNED: i64 = 4;
/// `ctypeprim.py W_CTypePrimitiveBool`.
pub const KIND_PRIM_BOOL: i64 = 5;
/// `ctypeprim.py W_CTypePrimitiveFloat`.
pub const KIND_PRIM_FLOAT: i64 = 6;
/// `ctypeprim.py W_CTypePrimitiveLongDouble`.
pub const KIND_PRIM_LONGDOUBLE: i64 = 7;
/// `ctypeprim.py W_CTypePrimitiveComplex`.
pub const KIND_PRIM_COMPLEX: i64 = 8;
/// `ctypeptr.py W_CTypePointer`.
pub const KIND_POINTER: i64 = 9;
/// `ctypearray.py W_CTypeArray`.
pub const KIND_ARRAY: i64 = 10;
/// `ctypestruct.py W_CTypeStruct`.
pub const KIND_STRUCT: i64 = 11;
/// `ctypestruct.py W_CTypeUnion`.
pub const KIND_UNION: i64 = 12;
/// `ctypefunc.py W_CTypeFunc`.
pub const KIND_FUNC: i64 = 13;

// ── the boolean class attributes ────────────────────────────────────────

bitflags::bitflags! {
    /// Boolean class attributes packed onto `W_CType.flags`.
    #[repr(transparent)]
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub struct CTypeFlags: i64 {
        const PRIMITIVE_INTEGER = 1 << 0;
        const NONFUNC_POINTER_OR_ARRAY = 1 << 1;
        const ACCEPT_STR = 1 << 2;
        const VOID_PTR = 1 << 3;
        const VOIDCHAR_PTR = 1 << 4;
        const ONEBYTE_PTR = 1 << 5;
        const FILE_PTR = 1 << 6;
        const VALUE_FITS_LONG = 1 << 7;
        const VALUE_SMALLER_THAN_LONG = 1 << 8;
        const VALUE_FITS_ULONG = 1 << 9;
        const SIGNED_WCHAR = 1 << 10;
        const ELLIPSIS = 1 << 11;
        const ENUM = 1 << 12;
        const CUSTOM_FIELD_POS = 1 << 13;
        const WITH_VAR_ARRAY = 1 << 14;
        const WITH_PACKED_CHANGE = 1 << 15;
    }
}

/// `ctypeobj.py W_CType` and the RPython subclasses that share its typedef.
///
/// `UniqueCache` keeps primitive/singleton ctypes strongly and reaches derived
/// ctypes through weak references (`W_CType._pointer_type`,
/// `W_CTypePointer._array_types`).  Thus a derived ctype dies with its last
/// user while a repeated constructor still returns the live memoized object.
#[pyre_interpreter::pyre_class("_cffi_backend.CType", weakrefable)]
// The `_immutable_fields_` the RPython hierarchy spreads over its subclasses,
// restricted to the ones this flattened struct still writes only while a ctype
// is being constructed: `W_CType` names `name_position`, `W_CTypePtrOrArray`
// names `ctitem` and `length`, `W_CTypeArray` names `ctptr`, and `W_CTypeFunc`
// names `fargs[*]`, `abi` and `cif_descr`.  The `cif_descr` declaration is what
// lets a call through a constant function type fold `exchange_size` /
// `exchange_args[i]` / `exchange_result` to trace constants.
//
// `kind` has no upstream entry to quote because the hierarchy spells the
// subclass identity as the class pointer, which is immutable by construction.
// It is this flattening's stand-in for that pointer, so declaring it is what
// turns a `match ct.kind` dispatcher on a promoted ctype back into the static
// overload resolution the subclasses get for free.
//
// `align` and `flags` stay mutable: completion writes both after the ctype is
// reachable, and `flags` carries `_custom_field_pos` and `_with_var_array`.
// `size` is `W_CType`'s `size?`: the same writes go through `set_size`, which
// invalidates loops that folded the old value.
#[majit_macros::jit_immutable_fields(
    "kind",
    "name_position",
    "ctitem",
    "ctptr",
    "length",
    "fargs[*]",
    "abi",
    "cif_descr",
    // `W_CTypePrimitiveSigned._immutable_fields_` and the unsigned twin.
    // Set once from `flags` in `new_ctype` and never stored again, so a
    // promoted ctype folds `convert_from_object` / `convert_to_object`.
    "value_fits_long",
    "value_smaller_than_long",
    "value_fits_ulong",
    "size?"
)]
pub struct W_CType {
    /// `W_CType.size` — the size of an instance, or -1 when unknown.
    /// Quasi-immutable (`size?`): [`set_size`] is the only store.
    pub size: i64,
    /// Hidden `mutate_size` for `size?`. Holds no GC pointers.
    pub size_watchers: pyre_object::quasiimmut::QuasiImmutField,
    /// `W_CType.name`.  Interpreter-level, as upstream has it: `cname`
    /// wraps it on each read.  Every ctype is memoised and rooted for the
    /// process, so its name is leaked on the same terms.
    pub name: &'static str,
    /// `W_CType.name_position` — where `insert_name` splices into `name`.
    pub name_position: i64,
    /// Which RPython subclass of `W_CType` this is; one of the `KIND_*`.
    pub kind: i64,
    /// `W_CTypePrimitive.align`, or -1 for a type of unknown alignment.
    pub align: i64,
    /// `W_CTypePtrOrArray.ctitem` — what a pointer points to, what an array
    /// holds.  `PY_NULL` on every other kind.
    pub ctitem: PyObjectRef,
    /// `W_CType._pointer_type` — the GC weakref box for the pointer *to* this
    /// type.  `PY_NULL` until one is asked for.
    pub pointer_type: PyObjectRef,
    /// `W_CTypePointer._array_types` — the pointer-owned weak-value mapping
    /// from lengths to array ctypes.  `PY_NULL` on non-pointers and before the
    /// first array type is requested.
    pub array_types: PyObjectRef,
    /// `W_CTypeArray.ctptr` — the pointer this array was built from, whose
    /// `ctitem` is the array's item type.  `PY_NULL` on every other kind.
    /// It is not `pointer_type`: the pointer to an `int[5]` is `int(*)[5]`,
    /// while its `ctptr` is `int *`.
    pub ctptr: PyObjectRef,
    /// `W_CTypeArray.length` — -1 for `int[]` and for anything not an array.
    pub length: i64,
    /// Packed [`CTypeFlags`].
    pub flags: i64,
    /// `W_CTypePrimitiveSigned.value_fits_long` /
    /// `W_CTypePrimitiveUnsigned.value_fits_long`.
    pub value_fits_long: bool,
    /// `W_CTypePrimitiveSigned.value_smaller_than_long`.
    pub value_smaller_than_long: bool,
    /// `W_CTypePrimitiveUnsigned.value_fits_ulong`.
    pub value_fits_ulong: bool,
    /// `W_CTypeFunc.fargs` — the argument ctypes, as a tuple
    /// (`_immutable_fields_ = ['fargs[*]']`).  `PY_NULL` on every other kind.
    pub fargs: PyObjectRef,
    /// `W_CTypeFunc.abi`, the `FFI_*` calling convention.
    pub abi: i64,
    /// `W_CTypeFunc.cif_descr` — the `ffi_cif` this function type was
    /// prepared with, built once here rather than once per call.  Zero for a
    /// variadic function, whose cif depends on the arguments actually passed.
    ///
    /// `CIF_DESCRIPTION` is a raw-flavour block, so `CIF_DESCRIPTION_P` is an
    /// integer-kind pointer (`getkind(Ptr(TO))` answers `int` when
    /// `TO._gckind` is `raw`) and the address belongs in the integer register
    /// bank, not among the references a collector traces and rewrites.
    pub cif_descr: usize,
    /// `W_CTypeStructOrUnion._fields_list` — a list of [`super::ctypestruct::W_CField`]
    /// in declaration order.  `PY_NULL` while the struct is opaque or lazy.
    pub fields_list: PyObjectRef,
    /// `W_CTypeStructOrUnion._fields_dict` — name to `W_CField`.
    pub fields_dict: PyObjectRef,
    /// `_Mixin_Enum.enumerators2values` — a dict of `str` to `int`.
    pub enumerators2values: PyObjectRef,
    /// `_Mixin_Enum.enumvalues2erators` — a dict of `int` to `str`.
    pub enumvalues2erators: PyObjectRef,
    /// `W_CTypeStructOrUnion._lazy_ffi`, retained until its fields are forced.
    pub lazy_ffi: PyObjectRef,
    /// `W_CTypeStructOrUnion._lazy_s` as an index in `ctx.c_struct_unions`.
    pub lazy_sindex: i64,
}

impl W_CType {
    /// `W_CType.name`.
    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn has(&self, flag: CTypeFlags) -> bool {
        self.flags & flag.bits() != 0
    }

    /// The ctype as the object it is; every ctype is allocated non-moving, so
    /// its address is stable for the process.
    pub fn as_object(&self) -> PyObjectRef {
        std::ptr::from_ref(self)
            .cast::<pyre_object::PyObject>()
            .cast_mut()
    }

    /// `isinstance(ct, W_CTypePrimitive)`.
    pub fn is_primitive(&self) -> bool {
        (KIND_PRIM_CHAR..=KIND_PRIM_COMPLEX).contains(&self.kind)
    }

    /// `isinstance(ct, W_CTypePrimitiveFloat)` — which
    /// `W_CTypePrimitiveLongDouble` also satisfies.
    pub fn is_float_family(&self) -> bool {
        self.kind == KIND_PRIM_FLOAT || self.kind == KIND_PRIM_LONGDOUBLE
    }

    /// `isinstance(ct, W_CTypePrimitiveCharOrUniChar)`.
    pub fn is_char_or_unichar(&self) -> bool {
        self.kind == KIND_PRIM_CHAR || self.kind == KIND_PRIM_UNICHAR
    }

    /// `isinstance(ct, W_CTypePtrOrArray)`.  A function ctype answers yes:
    /// `W_CTypeFunc` is a `W_CTypePtrBase`, which is a `W_CTypePtrOrArray`.
    /// The predicate that excludes it is `CTypeFlags::NONFUNC_POINTER_OR_ARRAY`.
    pub fn is_ptr_or_array(&self) -> bool {
        self.kind == KIND_POINTER || self.kind == KIND_ARRAY || self.kind == KIND_FUNC
    }

    /// `isinstance(ct, W_CTypeStructOrUnion)`.
    pub fn is_struct_or_union(&self) -> bool {
        self.kind == KIND_STRUCT || self.kind == KIND_UNION
    }

    /// `W_CTypeStructOrUnion.check_complete` — an opaque struct answers
    /// nothing about its contents.
    pub fn check_complete(&self, value_error: bool) -> Result<(), PyError> {
        if self.size >= 0 {
            return Ok(());
        }
        let msg = format!("'{}' is opaque or not completed yet", self.name());
        Err(if value_error {
            PyError::value_error(msg)
        } else {
            PyError::type_error(msg)
        })
    }

    /// `W_CTypeStructOrUnion.force_lazy_struct` — complain if we are opaque.
    ///
    pub fn force_lazy_struct(&self) -> Result<(), PyError> {
        if !self.lazy_ffi.is_null() {
            super::realize_c_type::do_realize_lazy_struct(self.as_object())?;
        }
        self.check_complete(false)
    }

    /// `W_CTypePtrOrArray.is_unichar_ptr_or_array`, which `W_CTypeFunc`
    /// overrides to false.
    pub fn is_unichar_ptr_or_array(&self) -> bool {
        self.has(CTypeFlags::NONFUNC_POINTER_OR_ARRAY)
            && ctype_at(self.ctitem).is_some_and(|it| it.kind == KIND_PRIM_UNICHAR)
    }

    /// `W_CTypePtrOrArray.is_char_or_unichar_ptr_or_array`, which
    /// `W_CTypeFunc` overrides to false.
    pub fn is_char_or_unichar_ptr_or_array(&self) -> bool {
        self.has(CTypeFlags::NONFUNC_POINTER_OR_ARRAY)
            && ctype_at(self.ctitem).is_some_and(|it| it.is_char_or_unichar())
    }

    /// `W_CType._within_bounds`.
    pub fn within_bounds(&self, actual_length: i64) -> bool {
        self.length < 0 || actual_length <= self.length
    }

    /// `W_CType.insert_name` — splice `extra` into `name` at the recorded
    /// position, and report where the spliced name's own position is.
    pub fn insert_name(&self, extra: &str, extra_position: i64) -> (String, i64) {
        let name = self.name();
        let at = self.name_position as usize;
        let mut spliced = String::with_capacity(name.len() + extra.len());
        spliced.push_str(&name[..at]);
        spliced.push_str(extra);
        spliced.push_str(&name[at..]);
        (spliced, self.name_position + extra_position)
    }

    /// `W_CType.alignof` — `_alignof` raises for a type that has none.
    pub fn alignof(&self) -> Result<i64, PyError> {
        match self.kind {
            // `W_CTypePtrBase._alignof` — every pointer aligns as one, and
            // a function type is one of them.
            KIND_POINTER | KIND_FUNC => Ok(align_of::<*const u8>() as i64),
            // `W_CTypeArray._alignof` — `self.ctitem.alignof()`.
            KIND_ARRAY => ctype_at(self.ctitem)
                .ok_or_else(|| self.unknown_alignment())?
                .alignof(),
            // `W_CTypeStructOrUnion._alignof` — completing the struct is what
            // computes it.
            KIND_STRUCT | KIND_UNION => {
                self.check_complete(true)?;
                self.force_lazy_struct()?;
                Ok(self.align)
            }
            _ if self.align >= 0 => Ok(self.align),
            _ => Err(self.unknown_alignment()),
        }
    }

    fn unknown_alignment(&self) -> PyError {
        PyError::value_error(format!("ctype '{}' is of unknown alignment", self.name()))
    }

    /// `W_CType.extra_repr` — what `<cdata '...' HERE>` shows.
    ///
    /// # Safety
    /// `cdata` must be readable for this ctype's size when it is primitive.
    pub unsafe fn extra_repr(&self, cdata: *const u8) -> Result<String, PyError> {
        if self.kind == KIND_PRIM_LONGDOUBLE {
            return Ok(unsafe { super::misc::longdouble2str(cdata) });
        }
        if self.has(CTypeFlags::ENUM) {
            return unsafe { super::ctypeenum::extra_repr(self, cdata) };
        }
        if self.is_primitive() {
            // `W_CTypePrimitive.extra_repr` — `repr(convert_to_object(cdata))`.
            let roots = pyre_object::gc_roots::push_roots();
            let ob_slot = roots.base();
            let _ = roots.pin_root(unsafe { convert_to_object(self, cdata as usize)? });
            let w_repr = pyre_interpreter::builtins::builtin_repr(&[roots.get(ob_slot)])?;
            return Ok(pyre_interpreter::baseobjspace::str_utf8_w(w_repr)?.to_string());
        }
        Ok(if cdata.is_null() {
            "NULL".to_string()
        } else {
            format!("0x{:x}", cdata as usize)
        })
    }
}

/// Borrow the `W_CType` a `PyObjectRef` names, or `None` when it is not one.
pub fn ctype_at(w_ctype: PyObjectRef) -> Option<&'static mut W_CType> {
    if w_ctype.is_null() {
        return None;
    }
    W_CType::from_obj(w_ctype)
}

/// The ctype a cdata carries.
pub fn ctype_of(cdata: &W_CData) -> Option<&'static mut W_CType> {
    ctype_at(cdata.ctype)
}

/// `@unwrap_spec(w_ctype=ctypeobj.W_CType)`.
pub fn ctype_arg(w_ctype: PyObjectRef) -> Result<&'static mut W_CType, PyError> {
    // A `match`, not `Option::ok_or_else`: the combinator's closure is a
    // callee a traced caller would stop at.
    match ctype_at(w_ctype) {
        Some(ct) => Ok(ct),
        None => Err(PyError::type_error(format!(
            "expected a ctype object, got '{}'",
            pyre_interpreter::type_methods::arg_type_name(w_ctype)
        ))),
    }
}

// ── construction ────────────────────────────────────────────────────────

/// Build a GC-managed ctype.  `UniqueCache` roots only its strong primitive
/// and singleton entries; derived caches retain weakref boxes instead.
pub fn new_ctype(
    kind: i64,
    size: i64,
    name: &str,
    name_position: i64,
    align: i64,
    ctitem: PyObjectRef,
    length: i64,
    flags: i64,
) -> PyObjectRef {
    let roots = pyre_object::gc_roots::push_roots();
    let ctitem_slot = roots.base();
    let _ = roots.pin_root(ctitem);
    let obj = W_CType::allocate_stable(W_CType {
        size,
        size_watchers: pyre_object::quasiimmut::QuasiImmutField::new(),
        name: String::leak(name.to_string()),
        name_position,
        kind,
        align,
        ctitem: roots.get(ctitem_slot),
        pointer_type: pyre_object::PY_NULL,
        array_types: pyre_object::PY_NULL,
        ctptr: pyre_object::PY_NULL,
        length,
        flags,
        value_fits_long: flags & CTypeFlags::VALUE_FITS_LONG.bits() != 0,
        value_smaller_than_long: flags & CTypeFlags::VALUE_SMALLER_THAN_LONG.bits() != 0,
        value_fits_ulong: flags & CTypeFlags::VALUE_FITS_ULONG.bits() != 0,
        ..Default::default()
    });
    obj
}

impl Default for W_CType {
    fn default() -> Self {
        Self {
            ob: pyre_object::PyObject {
                ob_type: std::ptr::null(),
                w_class: pyre_object::PY_NULL,
            },
            size: -1,
            size_watchers: pyre_object::quasiimmut::QuasiImmutField::new(),
            name: "",
            name_position: 0,
            kind: KIND_VOID,
            align: -1,
            ctitem: pyre_object::PY_NULL,
            pointer_type: pyre_object::PY_NULL,
            array_types: pyre_object::PY_NULL,
            ctptr: pyre_object::PY_NULL,
            length: -1,
            flags: 0,
            value_fits_long: false,
            value_smaller_than_long: false,
            value_fits_ulong: false,
            fargs: pyre_object::PY_NULL,
            abi: 0,
            cif_descr: 0,
            fields_list: pyre_object::PY_NULL,
            fields_dict: pyre_object::PY_NULL,
            enumerators2values: pyre_object::PY_NULL,
            enumvalues2erators: pyre_object::PY_NULL,
            lazy_ffi: pyre_object::PY_NULL,
            lazy_sindex: -1,
            lifeline: pyre_object::PY_NULL,
        }
    }
}

/// Strong roots owned by `UniqueCache`: primitive/singleton ctypes and the
/// weakref boxes held by process-global weak-value cache containers.  Ctypes
/// are born through `allocate_stable`, so cached addresses never relocate.
static ROOTED_CTYPES: std::sync::Mutex<Vec<Box<usize>>> = std::sync::Mutex::new(Vec::new());

/// `rclass.py hook_setfield` for `size?`: invalidate, then store.
pub fn set_size(ct: &mut W_CType, size: i64) {
    let ct_ptr = ct as *mut W_CType;
    unsafe {
        (*ct_ptr).size_watchers.invalidate_then_store(|| {
            (*ct_ptr).size = size;
        });
    }
}

/// Sweep half of `size?`. The collector runs no `Drop` on the payload.
///
/// # Safety
/// `obj` must be a GC-dead `W_CType`, and this must run once.
pub unsafe fn w_ctype_dealloc(obj: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_CType)).size_watchers.reclaim();
    }
}

pub fn root_forever(obj: PyObjectRef) {
    let _ = root_forever_slot(obj);
}

/// Register a process-lifetime root and return the stable address of its slot.
/// The collector rewrites the value in this slot when `obj` moves; native
/// cache containers must read through this address instead of retaining the
/// object's original address.
pub fn root_forever_slot(obj: PyObjectRef) -> *const usize {
    let mut slot = Box::new(obj as usize);
    let root_slot = (&raw mut *slot) as *mut pyre_object::gc_hook::GCREF;
    unsafe { pyre_object::gc_hook::try_gc_add_root(root_slot) };
    let stable_slot = (&raw const *slot) as *const usize;
    ROOTED_CTYPES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push(slot);
    stable_slot
}

// ── the dispatchers the RPython hierarchy spells as overrides ───────────

/// `W_CType.convert_to_object`.
///
/// # Safety
/// `cdata` must be readable for `ct.size` bytes.
pub unsafe fn convert_to_object(ct: &W_CType, cdata: usize) -> Result<PyObjectRef, PyError> {
    match ct.kind {
        // `W_CTypeFunc(W_CTypePtrBase)` inherits the same conversion.
        KIND_POINTER | KIND_FUNC => {
            Ok(unsafe { super::ctypeptr::pointer_convert_to_object(ct, cdata as *const u8) })
        }
        KIND_ARRAY => {
            Ok(unsafe { super::ctypeptr::array_convert_to_object(ct, cdata as *const u8) })
        }
        KIND_STRUCT | KIND_UNION => unsafe {
            super::ctypestruct::convert_to_object(ct, cdata as *const u8)
        },
        _ if ct.is_primitive() => unsafe { super::ctypeprim::convert_to_object(ct, cdata) },
        _ => Err(unsafe { PyError::from_exc_object(cannot_return_cdata(ct)) }),
    }
}

/// `W_CType.convert_to_object` — `oefmt("cannot return a cdata '%s'")`.
///
/// `error::oefmt` renders the message eagerly, so the rendering stays behind
/// `dont_look_inside` and the helper answers the exception instance: one
/// residual word, which the raise site wraps into the `PyError` the way
/// `OperationError` stores `w_value`. The other `_cffi_backend` `oefmt`
/// leaves take the same shape.
#[majit_macros::dont_look_inside]
pub(crate) fn cannot_return_cdata(ct: &W_CType) -> PyObjectRef {
    PyError::type_error(format!("cannot return a cdata '{}'", ct.name())).to_exc_object()
}

/// `W_CType.copy_and_convert_to_object` — `void` answers `None` rather than
/// reading anything.
///
/// # Safety
/// `cdata` must be readable for `ct.size` bytes unless the ctype is `void`.
pub unsafe fn copy_and_convert_to_object(
    ct: &W_CType,
    cdata: usize,
) -> Result<PyObjectRef, PyError> {
    if ct.kind == KIND_VOID {
        return Ok(pyre_object::w_none());
    }
    if ct.is_struct_or_union() {
        return unsafe { super::ctypestruct::copy_and_convert_to_object(ct, cdata as *const u8) };
    }
    unsafe { convert_to_object(ct, cdata) }
}

/// `W_CType.convert_from_object`.
///
/// # Safety
/// `cdata` must be writable for `ct.size` bytes.
pub unsafe fn convert_from_object(
    ct: &W_CType,
    cdata: usize,
    w_ob: PyObjectRef,
) -> Result<(), PyError> {
    // `self = jit.promote(self)` then a subclass method: `kind` is the
    // flattened class pointer (`jit_immutable_fields`), so promoting it
    // makes this match the static overload the hierarchy gets for free.
    let kind = majit_metainterp::jit::promote(ct.kind);
    match kind {
        // `W_CTypeFunc(W_CTypePtrBase)` inherits the same conversion.
        KIND_POINTER | KIND_FUNC => unsafe {
            super::ctypeptr::pointer_convert_from_object(ct, cdata as *mut u8, w_ob)
        },
        KIND_ARRAY => unsafe {
            super::ctypeptr::array_convert_from_object(ct, cdata as *mut u8, w_ob)
        },
        KIND_STRUCT | KIND_UNION => unsafe {
            super::ctypestruct::convert_from_object(ct, cdata as *mut u8, w_ob)
        },
        _ if ct.is_primitive() => unsafe { super::ctypeprim::convert_from_object(ct, cdata, w_ob) },
        _ => Err(unsafe { PyError::from_exc_object(cannot_initialize_cdata(ct)) }),
    }
}

/// `W_CType.convert_from_object` — `oefmt("cannot initialize cdata '%s'")`.
#[majit_macros::dont_look_inside]
pub(crate) fn cannot_initialize_cdata(ct: &W_CType) -> PyObjectRef {
    PyError::type_error(format!("cannot initialize cdata '{}'", ct.name())).to_exc_object()
}

/// `W_CType._convert_error` — the initializer-mismatch TypeError, with the
/// two special cases a same-named cdata gets.
///
/// Shaped as [`cannot_return_cdata`]. `expected` is one of the callers'
/// literals, passed through a `'static` reference because a `&str` is two
/// words and the residual ABI passes one per argument.
#[majit_macros::dont_look_inside]
pub(crate) fn convert_error(
    ct: &W_CType,
    expected: &'static &'static str,
    w_got: PyObjectRef,
) -> PyObjectRef {
    let name = ct.name();
    let mut error = if let Some(got) = W_CData::from_obj(w_got)
        && let Some(got_ctype) = ctype_at(got.ctype)
    {
        if name == got_ctype.name() {
            if std::ptr::eq(ct, got_ctype) {
                PyError::system_error(format!(
                    "initializer for ctype '{name}' is correct, but we get an internal mismatch--please report a bug"
                ))
            } else {
                PyError::type_error(format!(
                    "initializer for ctype '{name}' appears indeed to be '{}', but the types are different (check that you are not e.g. mixing up different ffi instances)",
                    got_ctype.name()
                ))
            }
        } else {
            PyError::type_error(format!(
                "initializer for ctype '{name}' must be a {expected}, not cdata '{}'",
                got_ctype.name()
            ))
        }
    } else {
        PyError::type_error(format!(
            "initializer for ctype '{name}' must be a {expected}, not {}",
            pyre_interpreter::type_methods::arg_type_name(w_got)
        ))
    };
    error.to_exc_object()
}

/// `W_CType.cast`.
pub fn cast(w_ctype: PyObjectRef, w_ob: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let ct = ctype_arg(w_ctype)?;
    match ct.kind {
        _ if ct.is_ptr_or_array() => super::ctypeptr::cast(w_ctype, w_ob),
        _ if ct.is_primitive() => super::ctypeprim::cast(w_ctype, w_ob),
        _ => Err(PyError::type_error(format!(
            "cannot cast to '{}'",
            ct.name()
        ))),
    }
}

/// `W_CType.newp`.
pub fn newp(w_ctype: PyObjectRef, w_init: PyObjectRef) -> Result<PyObjectRef, PyError> {
    newp_with_allocator(w_ctype, w_init, pyre_object::PY_NULL)
}

/// `W_CType.newp(w_init, allocator)`.
pub fn newp_with_allocator(
    w_ctype: PyObjectRef,
    w_init: PyObjectRef,
    w_allocator: PyObjectRef,
) -> Result<PyObjectRef, PyError> {
    let ct = ctype_arg(w_ctype)?;
    match ct.kind {
        KIND_POINTER => super::ctypeptr::pointer_newp(w_ctype, w_init, w_allocator),
        KIND_ARRAY => super::ctypeptr::array_newp(w_ctype, w_init, w_allocator),
        _ => Err(PyError::type_error(format!(
            "expected a pointer or array ctype, got '{}'",
            ct.name()
        ))),
    }
}

/// `W_CType.cast_to_int` — what `int(cdata)` reads.
///
/// # Safety
/// `cdata` must be readable for `ct.size` bytes.
pub unsafe fn cast_to_int(ct: &W_CType, cdata: *const u8) -> Result<PyObjectRef, PyError> {
    if ct.is_primitive() {
        return unsafe { super::ctypeprim::cast_to_int(ct, cdata) };
    }
    Err(PyError::type_error(format!(
        "int() not supported on cdata '{}'",
        ct.name()
    )))
}

/// `W_CType.float`.
///
/// # Safety
/// `cdata` must be readable for `ct.size` bytes.
pub unsafe fn float(ct: &W_CType, cdata: *const u8) -> Result<PyObjectRef, PyError> {
    if ct.is_float_family() {
        return unsafe { super::ctypeprim::float(ct, cdata) };
    }
    Err(PyError::type_error(format!(
        "float() not supported on cdata '{}'",
        ct.name()
    )))
}

/// `W_CType.complex`.
///
/// # Safety
/// `cdata` must be readable for `ct.size` bytes.
pub unsafe fn complex(ct: &W_CType, cdata: *const u8) -> Result<PyObjectRef, PyError> {
    if ct.kind == KIND_PRIM_COMPLEX {
        return unsafe { super::ctypeprim::convert_to_object(ct, cdata as usize) };
    }
    Err(PyError::type_error(format!(
        "complex() not supported on cdata '{}'",
        ct.name()
    )))
}

/// `W_CType.nonzero`.
///
/// # Safety
/// `cdata` must be readable for `ct.size` bytes.
pub unsafe fn nonzero(ct: &W_CType, cdata: *const u8) -> Result<bool, PyError> {
    if ct.is_primitive() {
        return unsafe { super::ctypeprim::nonzero(ct, cdata) };
    }
    Ok(!cdata.is_null())
}

/// `W_CType.add` — pointer arithmetic on a cdata.
pub fn add(w_ctype: PyObjectRef, cdata: *mut u8, i: i64) -> Result<PyObjectRef, PyError> {
    let ct = ctype_arg(w_ctype)?;
    match ct.kind {
        KIND_POINTER | KIND_ARRAY => super::ctypeptr::add(w_ctype, cdata, i),
        _ => Err(PyError::type_error(format!(
            "cannot add a cdata '{}' and a number",
            ct.name()
        ))),
    }
}

/// `W_CType.string`.
pub fn string(w_cdata: PyObjectRef, maxlen: i64) -> Result<PyObjectRef, PyError> {
    let cdata =
        W_CData::from_obj(w_cdata).ok_or_else(|| PyError::type_error("expected a cdata object"))?;
    let ct = ctype_of(cdata).ok_or_else(|| PyError::type_error("expected a cdata object"))?;
    match ct.kind {
        KIND_POINTER | KIND_ARRAY => super::ctypeptr::string(w_cdata, maxlen),
        _ if ct.has(CTypeFlags::ENUM) => unsafe {
            super::ctypeenum::string(ct, cdata.ptr as *const u8)
        },
        _ if ct.is_primitive() => super::ctypeprim::string(w_cdata, maxlen),
        _ => Err(unexpected_string_argument(ct)),
    }
}

/// `W_CType.convert_argument_from_object` — writes one call argument into the
/// exchange buffer and reports whether the pointer it left there has to be
/// freed after the call.
///
/// # Safety
/// `cdata` must be writable for `ct.size` bytes, and for the must-free flag
/// byte just before it when the ctype is a pointer.
pub unsafe fn convert_argument_from_object(
    ct: &W_CType,
    cdata: usize,
    w_ob: PyObjectRef,
) -> Result<bool, PyError> {
    if ct.kind == KIND_POINTER {
        return unsafe {
            super::ctypeptr::pointer_convert_argument_from_object(ct, cdata as *mut u8, w_ob)
        };
    }
    unsafe { convert_from_object(ct, cdata, w_ob)? };
    Ok(false)
}

/// `W_CType.getcfield` and the two overrides of it, with the mode
/// `W_CData.getcfield` names in the opaque-struct message.
pub fn getcfield(
    ct: &W_CType,
    attr: &str,
    mode: &str,
) -> Result<&'static mut super::ctypestruct::W_CField, PyError> {
    // `W_CTypePointer.getcfield` — a pointer to a struct reads its fields.
    let owner = if ct.kind == KIND_POINTER {
        ctype_at(ct.ctitem).filter(|it| it.is_struct_or_union())
    } else {
        None
    };
    let owner = match owner {
        Some(owner) => owner,
        None if ct.is_struct_or_union() => ct,
        // `W_CType.getcfield` — nothing else has fields at all.
        None => {
            return Err(PyError::attribute_error(format!(
                "cdata '{}' has no attribute '{attr}'",
                ct.name()
            )));
        }
    };
    if owner.fields_dict.is_null() {
        // `W_CTypeStructOrUnion.getcfield` returns None for an opaque struct,
        // which `W_CData.getcfield` turns into its own message.
        owner.check_complete(false).map_err(|_| {
            PyError::attribute_error(format!(
                "cdata '{}' points to an opaque type: cannot {mode} fields",
                ct.name()
            ))
        })?;
        owner.force_lazy_struct()?;
    }
    let w_field =
        unsafe { pyre_object::dictmultiobject::w_dict_getitem_str(owner.fields_dict, attr) };
    w_field
        .and_then(super::ctypestruct::W_CField::from_obj)
        .ok_or_else(|| {
            PyError::attribute_error(format!("cdata '{}' has no field '{attr}'", ct.name()))
        })
}

pub fn unexpected_string_argument(ct: &W_CType) -> PyError {
    PyError::type_error(format!(
        "string(): unexpected cdata '{}' argument",
        ct.name()
    ))
}

/// `W_CType.get_vararg_type` — the type an argument is promoted to when it
/// is passed through `...`.
pub fn get_vararg_type(w_ctype: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let ct = ctype_arg(w_ctype)?;
    match ct.kind {
        KIND_ARRAY => Ok(ct.ctptr),
        KIND_PRIM_CHAR | KIND_PRIM_UNICHAR => super::newtype::new_primitive_type("int"),
        KIND_PRIM_SIGNED | KIND_PRIM_UNSIGNED
            if ct.size < std::mem::size_of::<std::ffi::c_int>() as i64 =>
        {
            super::newtype::new_primitive_type("int")
        }
        _ => Ok(w_ctype),
    }
}

// ── the Python type ─────────────────────────────────────────────────────

static CTYPE_TYPE_OBJ: pyre_object::gc_roots::RootedOnceRef =
    pyre_object::gc_roots::RootedOnceRef::new();

/// `_cffi_backend.CType`.
pub fn ctype_type() -> PyObjectRef {
    CTYPE_TYPE_OBJ.get_or_init(|| {
        let tp = pyre_interpreter::typedef::make_builtin_type_with_layout(
            "_cffi_backend.CType",
            init_ctype_type,
            pyre_interpreter::typedef::w_object(),
            <W_CType as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE,
        );
        pyre_object::pyobject::set_instantiate(
            unsafe { &*<W_CType as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE },
            tp,
        );
        tp
    })
}

/// The names `W_CType.dir` reports — `typedef.rawdict` minus the dunders,
/// sorted, filtered by which of them the ctype actually answers.
const ATTRIBUTE_NAMES: [&str; 11] = [
    "abi",
    "args",
    "cname",
    "elements",
    "ellipsis",
    "fields",
    "item",
    "kind",
    "length",
    "relements",
    "result",
];

fn init_ctype_type(ns: PyObjectRef) {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    let store =
        |name: &str, value: PyObjectRef| pyre_interpreter::__pyre_put_new!(ns_slot, name, value);
    store(
        "__repr__",
        pyre_interpreter::make_builtin_function_with_arity("__repr__", ctype_repr, 1),
    );
    store(
        "__dir__",
        pyre_interpreter::make_builtin_function_with_arity("__dir__", ctype_dir, 1),
    );
    store(
        "__weakref__",
        pyre_interpreter::typedef::make_weakref_descr(pyre_object::gc_roots::shadow_stack_get(
            ns_slot,
        )),
    );
    for (name, doc, attrchar) in [
        ("kind", "kind", 'k'),
        ("cname", "C name", 'c'),
        ("item", "pointer to, or array of", 'i'),
        ("length", "array length or None", 'l'),
        ("fields", "struct or union fields", 'f'),
        ("args", "function argument types", 'a'),
        ("result", "function result type", 'r'),
        ("ellipsis", "function has '...'", 'E'),
        ("abi", "function ABI", 'A'),
        ("elements", "enum elements", 'e'),
        ("relements", "enum elements, reversed", 'R'),
    ] {
        let getter = make_fget(name, attrchar);
        store(
            name,
            pyre_interpreter::typedef::make_getset_property_named_doc(
                getter,
                pyre_object::PY_NULL,
                pyre_object::PY_NULL,
                doc,
                name,
            ),
        );
    }
}

/// One `fget_*` per attribute character.  `W_CType._fget` is a single method
/// switching on that character, so the wrappers only differ in which one they
/// pass; a builtin function carries no closure, hence the explicit table.
fn make_fget(name: &'static str, attrchar: char) -> PyObjectRef {
    macro_rules! fget {
        ($c:literal) => {
            pyre_interpreter::make_builtin_function_with_arity(
                name,
                |args| {
                    // `typedef.py self.fget(self, space, w_obj)` — the
                    // descriptor comes first and the instance second.
                    let w_self = args
                        .get(1)
                        .copied()
                        .ok_or_else(|| PyError::type_error("descriptor requires an instance"))?;
                    fget(w_self, $c)
                },
                2,
            )
        };
    }
    match attrchar {
        'k' => fget!('k'),
        'c' => fget!('c'),
        'i' => fget!('i'),
        'l' => fget!('l'),
        'f' => fget!('f'),
        'a' => fget!('a'),
        'r' => fget!('r'),
        'E' => fget!('E'),
        'A' => fget!('A'),
        'e' => fget!('e'),
        _ => fget!('R'),
    }
}

/// `W_CType._fget` and the subclass overrides of it.
fn fget(w_self: PyObjectRef, attrchar: char) -> Result<PyObjectRef, PyError> {
    let ct = ctype_arg(w_self)?;
    match attrchar {
        // `W_CType._fget('k')` — a class attribute in PyPy.
        'k' => Ok(pyre_object::w_str_new_managed(kind_name(ct))),
        'c' => Ok(pyre_object::w_str_new_managed(ct.name)),
        // `W_CTypePointer._fget('i')` and `W_CTypeArray._fget('i')`.  A
        // function type has none: `W_CTypeFunc` reaches `W_CType._fget`.
        'i' if ct.is_ptr_or_array() => Ok(ct.ctitem),
        // `W_CTypeArray._fget('l')`.
        'l' if ct.kind == KIND_ARRAY => Ok(if ct.length >= 0 {
            pyre_object::w_int_new(ct.length)
        } else {
            pyre_object::w_none()
        }),
        // `W_CTypeStructOrUnion._fget('f')`.
        'f' if ct.is_struct_or_union() => super::ctypestruct::fget_fields(ct),
        // `W_CTypeFunc._fget('a')` — a fresh tuple of the declared argument
        // ctypes (`ctypefunc.py` `_fget`). The stored `fargs` stays the
        // internal tuple; the attribute is not that object.
        'a' if ct.kind == KIND_FUNC => Ok(if ct.fargs.is_null() {
            pyre_object::tupleobject::w_tuple_new(Vec::new())
        } else {
            pyre_object::tupleobject::w_tuple_new(unsafe {
                pyre_object::tupleobject::w_tuple_items_copy_as_vec(ct.fargs)
            })
        }),
        'r' if ct.kind == KIND_FUNC => Ok(ct.ctitem),
        'E' if ct.kind == KIND_FUNC => Ok(pyre_object::boolobject::w_bool_from(
            ct.has(CTypeFlags::ELLIPSIS),
        )),
        'A' if ct.kind == KIND_FUNC => Ok(pyre_object::w_int_new(ct.abi)),
        // `_Mixin_Enum._fget` builds a fresh dict on each read, so the two
        // maps a ctype holds stay its own.
        'e' if ct.has(CTypeFlags::ENUM) => super::ctypeenum::copy_map(ct.enumvalues2erators),
        'R' if ct.has(CTypeFlags::ENUM) => super::ctypeenum::copy_map(ct.enumerators2values),
        _ => Err(PyError::attribute_error(format!(
            "ctype '{}' has no such attribute",
            ct.name()
        ))),
    }
}

/// `W_CType.kind`, the class attribute each subclass overrides.
pub fn kind_name(ct: &W_CType) -> &'static str {
    match ct.kind {
        KIND_VOID => "void",
        KIND_POINTER => "pointer",
        KIND_ARRAY => "array",
        KIND_STRUCT => "struct",
        KIND_UNION => "union",
        KIND_FUNC => "function",
        _ if ct.has(CTypeFlags::ENUM) => "enum",
        _ => "primitive",
    }
}

/// `W_CType.repr`.
fn ctype_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let ct = ctype_arg(args[0])?;
    Ok(pyre_object::w_str_new_managed(&format!(
        "<ctype '{}'>",
        ct.name()
    )))
}

/// `W_CType.dir` — every attribute above that this ctype answers.
fn ctype_dir(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_self = args[0];
    let roots = pyre_object::gc_roots::push_roots();
    let self_slot = roots.pin_roots(&[w_self]);
    let base = self_slot + 1;
    let mut count = 0;
    for name in ATTRIBUTE_NAMES {
        if pyre_interpreter::baseobjspace::getattr_str(roots.get(self_slot), name).is_ok() {
            let _ = roots.pin_root(pyre_object::w_str_new(name));
            count += 1;
        }
    }
    let items = (0..count).map(|i| roots.get(base + i)).collect();
    Ok(pyre_object::w_list_new(items))
}
