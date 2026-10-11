//! `sys` module VM helpers.
//!
//! PyPy equivalent: `pypy/module/sys/vm.py`.

use crate::executioncontext::ActionFlagOps;
use crate::{
    make_builtin_function, make_builtin_function_with_arity,
    make_builtin_function_with_arity_and_maybe_sig, module_ns_store,
};
use pyre_object::*;
use std::sync::OnceLock;

const GETSIZEOF_DOC: &str = "getsizeof(object [, default]) -> int\n\n\
Return the size of object in bytes.";

/// The ids `sys.monitoring.use_tool_id` accepts, one below the first id the
/// runtime reserves for itself (`PY_MONITORING_SYS_PROFILE_ID`).
const MONITORING_TOOL_COUNT: usize = 6;

/// `interp->monitoring_tool_names` — the name each claimed tool id was
/// registered under, or NULL while the id is free.
///
/// The event machinery below is inert, but *ownership* is not: `get_tool`
/// reports it, `bdb` consults it before installing its tracer, and it is what
/// makes a second `cProfile.Profile.enable()` fail instead of silently
/// displacing the first.  Held as raw addresses and forwarded by
/// [`walk_monitoring_tool_roots`], since a name is an ordinary movable str.
static MONITORING_TOOL_NAMES: [std::sync::atomic::AtomicUsize; MONITORING_TOOL_COUNT] =
    [const { std::sync::atomic::AtomicUsize::new(0) }; MONITORING_TOOL_COUNT];

/// Forward the claimed tool names for the process-global root walk.
pub fn walk_monitoring_tool_roots(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    for slot in MONITORING_TOOL_NAMES.iter() {
        let addr = slot.load(std::sync::atomic::Ordering::Relaxed);
        if addr == 0 {
            continue;
        }
        let mut root = majit_ir::GcRef(addr);
        visitor(&mut root);
        slot.store(root.0, std::sync::atomic::Ordering::Relaxed);
    }
}

/// `check_valid_tool` — reject an id outside the claimable range before it is
/// used to index the table.
fn monitoring_tool_id_w(w_tool_id: PyObjectRef) -> Result<usize, crate::PyError> {
    let tool_id = crate::builtins::space_index_w(w_tool_id)?;
    if !(0..MONITORING_TOOL_COUNT as i64).contains(&tool_id) {
        return Err(crate::PyError::value_error(format!(
            "invalid tool {tool_id} (must be between 0 and {})",
            MONITORING_TOOL_COUNT - 1
        )));
    }
    Ok(tool_id as usize)
}

fn monitoring_tool_name(tool_id: usize) -> PyObjectRef {
    MONITORING_TOOL_NAMES[tool_id].load(std::sync::atomic::Ordering::Relaxed) as PyObjectRef
}

fn monitoring_set_tool_name(tool_id: usize, name: PyObjectRef) {
    MONITORING_TOOL_NAMES[tool_id].store(name as usize, std::sync::atomic::Ordering::Relaxed);
}

/// `monitoring_use_tool_id_impl` — claim `tool_id` for `w_name`, refusing an
/// id someone else already holds.
pub fn monitoring_use_tool_id(tool_id: usize, w_name: PyObjectRef) -> Result<(), crate::PyError> {
    if !unsafe { pyre_object::is_str(w_name) } {
        return Err(crate::PyError::value_error("tool name must be a str"));
    }
    if !monitoring_tool_name(tool_id).is_null() {
        return Err(crate::PyError::value_error(format!(
            "tool {tool_id} is already in use"
        )));
    }
    monitoring_set_tool_name(tool_id, w_name);
    Ok(())
}

/// The id `_lsprof` claims, matching `sys.monitoring.PROFILER_ID`.
pub const MONITORING_PROFILER_ID: usize = 2;

/// Release `tool_id` unconditionally — `free_tool_id`'s interp-level spelling,
/// for callers that claimed the id from native code.
pub fn monitoring_free_tool_id(tool_id: usize) {
    monitoring_set_tool_name(tool_id, pyre_object::PY_NULL);
}

/// CPython 3.14 `_PySys_GetSizeOf`: look up `__sizeof__` on the type,
/// call the bound method, require a non-negative Py_ssize_t result, then add
/// the pre-header used by a heap type.  Pyre has no physical CPython object
/// pre-header, but exposes the 3.14 logical size required by `sys.getsizeof`.
///
/// Answering with a size is a deliberate divergence from this module's
/// counterpart. `pypy/module/sys/vm.py` returns `w_default` unchanged
/// and raises `TypeError` when no default was supplied, for the reason its own
/// `getsizeof_missing` text gives: maps are shared across instances, equal
/// strings may share their data, and some sequences materialise their items as
/// you read them, so a per-object number would be inconsistent with reality
/// there. That argument is about an object model where the storage backing a
/// value is shared and lazily created; pyre's objects have the layout the
/// question assumes, so the number is answerable and is answered.
///
/// The divergence is confined to producing a number, whose two terms come from
/// different places. `size` is whatever the object's own `__sizeof__` returns,
/// so it may read instance state — `list.__sizeof__` adds a word per allocated
/// slot. The pre-header term is the one read off the type rather than off the
/// instance's storage, which is what keeps that half from depending on the
/// allocation that produced the object (see the pre-header comment below).
fn get_sizeof(w_obj: PyObjectRef) -> crate::PyResult {
    let roots = pyre_object::gc_roots::push_roots();
    let obj_slot = roots.base();
    let _ = roots.pin_root(w_obj);
    let current = || roots.get(obj_slot);
    let method = unsafe { crate::baseobjspace::lookup_special(current(), "__sizeof__")? }
        .ok_or_else(|| {
            crate::PyError::type_error(format!(
                "Type {} doesn't define __sizeof__",
                crate::baseobjspace::object_functionstr_type_name(current()),
            ))
        })?;
    let w_size = crate::call::call_function_impl_result(method, &[])?;

    // PyLong_AsSsize_t accepts bool/int subclasses but performs no __index__
    // conversion on an arbitrary return value.
    let size = if unsafe { pyre_object::is_bool(w_size) } {
        unsafe { pyre_object::w_bool_get_value(w_size) as i64 }
    } else if unsafe { pyre_object::is_int(w_size) } {
        unsafe { pyre_object::w_int_get_value(w_size) }
    } else if unsafe { pyre_object::is_long(w_size) } {
        let value = unsafe { pyre_object::w_long_get_value(w_size) };
        if pyre_object::longobject::jit_bigint_to_i64_fits(value) == 0 {
            return Err(crate::PyError::overflow_error(
                "Python int too large to convert to C ssize_t",
            ));
        }
        pyre_object::longobject::jit_bigint_to_i64_value(value)
    } else {
        return Err(crate::PyError::type_error("an integer is required"));
    };
    if size < 0 {
        return Err(crate::PyError::value_error(
            "__sizeof__() should return >= 0",
        ));
    }

    // `_PyType_PreHeaderSize(Py_TYPE(o))` — the two-word managed dict/weakref
    // prefix an instance type requests. Its other term, a `PyGC_Head` ahead of
    // every tracked object, is compiled out of a build without a global
    // interpreter lock: the collector keeps its bits in the object header
    // there. That is why the four-word header those builds carry costs a
    // tracked type nothing over the two-word one, while an untracked leaf pays
    // the whole two words.
    //
    // The term is read off the type. Asking instead which heap the instance
    // landed in would make the answer depend on the allocation that produced
    // it: a `str` folded into a code constant sits outside the collector's
    // ranges and one built at run time does not, so the same value reported two
    // different sizes.
    let word = std::mem::size_of::<usize>() as u64;
    let pre_header = crate::typedef::r#type(current()).map_or(0, |tp| unsafe {
        if pyre_object::w_type_is_heaptype(tp.as_ptr())
            && (pyre_object::w_type_get_hasdict(tp.as_ptr())
                || pyre_object::w_type_get_weakrefable(tp.as_ptr()))
        {
            2 * word
        } else {
            0
        }
    });
    let total = (size as u64)
        .checked_add(pre_header)
        .expect("Py_ssize_t plus the fixed pre-header fits in size_t");
    if total <= i64::MAX as u64 {
        Ok(w_int_new(total as i64))
    } else {
        Ok(pyre_object::w_long_new(
            majit_rlib::rbigint::RBigInt::from_u128(total as u128),
        ))
    }
}

/// Shared stub type for `sys._getframe`, `sys.stdout` and other module-level
/// sys attributes that expose attribute bags.
///
/// `typedef::w_object()` (plain `object`) cannot store instance attributes —
/// its type flag `hasdict` is false, matching CPython where `object()`
/// instances reject `__setattr__` unless their subclass explicitly opts in.
/// PyPy's `sys` module exposes these as dedicated W_Root types with their
/// own typedefs, not as bare `object` instances. The Rust port mirrors that
/// by installing a single `sys.namespace` type with `__dict__` in its
/// typedef slots so every stub instance supports `setattr`.
fn sys_namespace_type() -> PyObjectRef {
    static TYPE: pyre_object::gc_roots::RootedOnceRef = pyre_object::gc_roots::RootedOnceRef::new();
    TYPE.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type("sys.namespace", |ns| {
            let _root_scope = pyre_object::gc_roots::push_roots();
            let ns_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(ns);
            unsafe {
                crate::__pyre_put_new!(
                    ns_slot,
                    "__init__",
                    crate::make_builtin_function("__init__", sys_namespace_init)
                )
            };
        });
        // The stubs want a per-instance mapdict store; a `__dict__`
        // rawdict key would instead claim the typedef manages the dict
        // (typedef.py:40) and suppress the mapdict one
        // (typeobject.py:253-257), so flip `hasdict` directly — the
        // `create_dict_slot` flag flip (typeobject.py).
        unsafe { w_type_set_hasdict(tp, true) };
        tp
    })
}

fn sys_namespace_init(args: &[PyObjectRef]) -> crate::PyResult {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let Some(&self_obj) = positional.first() else {
        return Err(crate::PyError::type_error(
            "__init__() missing 1 required positional argument: 'self'",
        ));
    };
    if positional.len() > 1 {
        return Err(crate::PyError::type_error(
            "types.SimpleNamespace() takes no positional arguments",
        ));
    }
    namespace_apply_kwargs(self_obj, kwargs)
}

/// Copy the keyword arguments into a namespace instance's dict, skipping the
/// `__pyre_kw__` marker. Shared by `sys.namespace` and `types.SimpleNamespace`
/// construction. `lib_pypy._structseq.SimpleNamespace.__init__` writes the
/// instance dict directly, so `setdictvalue` is used rather than `setattr` — a
/// subclass `__setattr__` is not consulted during construction.
fn namespace_apply_kwargs(self_obj: PyObjectRef, kwargs: Option<PyObjectRef>) -> crate::PyResult {
    if let Some(dict) = kwargs {
        namespace_update_dict(self_obj, dict, true)?;
    } else {
        // `self.__dict__.update(kwargs)` evaluates `self.__dict__` first, so
        // a receiver without an instance dict raises even for no keywords.
        crate::baseobjspace::getattr_str(self_obj, "__dict__")?;
    }
    Ok(w_none())
}

/// CPython 3.14 `PyDict_Update(ns->ns_dict, source)`, with the flat builtin
/// ABI's private `__pyre_kw__` marker optionally omitted.  Validate every key
/// before the first store, matching `PyArg_ValidateKeywordArguments`: a bad
/// mapping cannot partially update the namespace.  The destination is the
/// real instance dict, so subclass `__setattr__` is deliberately bypassed.
fn namespace_update_dict(
    self_obj: PyObjectRef,
    source: PyObjectRef,
    skip_kw_marker: bool,
) -> Result<(), crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let sp = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(self_obj);
    let _ = pyre_object::gc_roots::pin_root(source);
    let destination =
        crate::baseobjspace::getattr_str(pyre_object::gc_roots::shadow_stack_get(sp), "__dict__")?;
    let _ = pyre_object::gc_roots::pin_root(destination);
    let items =
        unsafe { pyre_object::w_dict_items(pyre_object::gc_roots::shadow_stack_get(sp + 1)) };
    let items_sp = pyre_object::gc_roots::shadow_stack_len();
    for &(key, value) in &items {
        let _ = pyre_object::gc_roots::pin_root(key);
        let _ = pyre_object::gc_roots::pin_root(value);
    }
    for i in 0..items.len() {
        let key = pyre_object::gc_roots::shadow_stack_get(items_sp + i * 2);
        if !unsafe { pyre_object::is_str(key) } {
            return Err(crate::PyError::type_error("keywords must be strings"));
        }
    }
    for i in 0..items.len() {
        let key = pyre_object::gc_roots::shadow_stack_get(items_sp + i * 2);
        let value = pyre_object::gc_roots::shadow_stack_get(items_sp + i * 2 + 1);
        if skip_kw_marker
            && unsafe { pyre_object::w_str_get_wtf8(key).as_str() == Ok("__pyre_kw__") }
        {
            continue;
        }
        crate::type_methods::dict_store_checked(
            pyre_object::gc_roots::shadow_stack_get(sp + 2),
            key,
            value,
        )?;
    }
    Ok(())
}

/// Allocate a fresh stub instance whose type supports `setattr`. Used for
/// all the CPython-style attribute bags surfaced by the sys module.
fn make_sys_namespace_instance() -> PyObjectRef {
    w_instance_new(sys_namespace_type())
}

/// Fresh `types.SimpleNamespace` instance — `type(sys.implementation)`, the
/// attribute bag `time.get_clock_info` fills in and returns.
pub fn new_simple_namespace_instance() -> crate::PyResult {
    simple_namespace_new(&[simple_namespace_type()])
}

/// CPython 3.14 `namespace_new`: allocate the instance and give it an
/// independent insertion-ordered dictionary before `namespace_init` sees any
/// arguments.
///
/// PyPy 3.11's app-level SimpleNamespace leaves its `__dict__` on
/// MapDictStrategy.  A delete on one instance can then reuse a reordered map
/// transition for a later instance, which is why PyPy sorts namespace reprs.
/// CPython 3.14 instead stores a fresh exact dict in `ns_dict`, and its repr
/// follows insertion order.  Use mapdict.py's own
/// `MapDictStrategy.switch_to_text_strategy` devolution path here: the
/// dictionary remains the instance's one authoritative `__dict__`, while its
/// storage becomes private and ordered like CPython's `ns_dict`.
///
/// The devolution is load-bearing, and an ordinary attribute-assigning class
/// does not show why — its instances keep insertion order across the same
/// delete. The case that needs it is the keyword constructor:
///
/// ```text
/// del SimpleNamespace(x=1, y=2, z=3).y      # reaches a shortened map
/// list(SimpleNamespace(x=1, y=2, z=3).__dict__)
/// # ['z', 'x', 'y'] on the map strategy, ['x', 'y', 'z'] devolved
/// ```
fn simple_namespace_new(args: &[PyObjectRef]) -> crate::PyResult {
    let Some(&subtype) = args.first() else {
        return Err(crate::PyError::type_error(
            "types.SimpleNamespace.__new__(): not enough arguments",
        ));
    };
    if !unsafe { pyre_object::is_type(subtype) } {
        return Err(crate::PyError::type_error(format!(
            "types.SimpleNamespace.__new__(X): X is not a type object ({})",
            crate::type_methods::arg_type_name(subtype),
        )));
    }
    // `tp_new_wrapper` checks the declared owner before namespace_new reaches
    // `type->tp_alloc`: calling SimpleNamespace.__new__(int) must not fall
    // through to object.__new__'s unrelated layout diagnostic.
    crate::typedef::check_user_subclass(simple_namespace_type(), subtype)?;
    // namespace_new ignores constructor arguments after the subtype; they
    // belong to namespace_init.  Passing only the subtype also mirrors
    // `type->tp_alloc(type, 0)` rather than object.__new__'s excess-args
    // validation.
    let object = crate::typedef::object_descr_new(&args[..1])?;
    let _roots = pyre_object::gc_roots::push_roots();
    let object_slot = pyre_object::gc_roots::pin_roots(&[object]);
    // `W_Root.getdict`, not the public attribute lookup: `namespace_new`
    // reaches the instance mapping directly, so a subclass that overrides
    // `__getattribute__` or shadows `__dict__` neither runs during
    // construction nor can make `S()` raise.
    let dict = crate::baseobjspace::getdict(pyre_object::gc_roots::shadow_stack_get(object_slot))?;
    let _ = pyre_object::gc_roots::pin_root(dict);
    let dict = pyre_object::gc_roots::shadow_stack_get(object_slot + 1);
    unsafe {
        use pyre_object::dictmultiobject::DictStrategy;
        if pyre_object::dictmultiobject::w_dict_get_strategy(dict).strategy_kind()
            == pyre_object::dictmultiobject::StrategyKind::Map
        {
            crate::objspace::std::mapdict::mapdict_switch_to_text_strategy(dict);
        }
    }
    Ok(pyre_object::gc_roots::shadow_stack_get(object_slot))
}

/// CPython 3.14 `namespace_init`: accept at most one positional mapping or
/// iterable of pairs, validate that its resulting dict has only string keys,
/// merge it into the instance, then overlay keyword arguments.  This is the
/// target-version delta from PyPy 3.11's keyword-only `_structseq.py` class.
fn simple_namespace_init(args: &[PyObjectRef]) -> crate::PyResult {
    let _roots = pyre_object::gc_roots::push_roots();
    let sp = _roots.pin_roots(args);
    let mut rooted = Vec::with_capacity(args.len());
    for i in 0..args.len() {
        rooted.push(_roots.get(sp + i));
    }
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(&rooted);
    let Some(&self_obj) = positional.first() else {
        return Err(crate::PyError::type_error(
            "__init__() missing 1 required positional argument: 'self'",
        ));
    };
    if positional.len() > 2 {
        return Err(crate::PyError::type_error(format!(
            "SimpleNamespace expected at most 1 argument, got {}",
            positional.len() - 1
        )));
    }
    // The kwargs carrier is not guaranteed to occupy the last raw ABI slot
    // relative to positional arguments.  Pin the parsed operands into a
    // canonical order before any allocation instead of deriving their slots
    // from the flat input layout.
    let mut operands = Vec::with_capacity(3);
    operands.push(self_obj);
    if positional.len() == 2 {
        operands.push(positional[1]);
    }
    if let Some(kwargs) = kwargs {
        operands.push(kwargs);
    }
    let operands_sp = _roots.pin_roots(&operands);
    let has_kwargs = kwargs.is_some();
    let kwargs_slot = _roots.pin_roots(&[kwargs.unwrap_or(pyre_object::PY_NULL)]);
    if positional.len() == 2 {
        let temporary = w_dict_new();
        let _ = pyre_object::gc_roots::pin_root(temporary);
        let temporary_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let temporary = pyre_object::gc_roots::shadow_stack_get(temporary_slot);
        crate::type_methods::dict_update1(
            temporary,
            pyre_object::gc_roots::shadow_stack_get(operands_sp + 1),
        )?;
        namespace_update_dict(
            pyre_object::gc_roots::shadow_stack_get(operands_sp),
            pyre_object::gc_roots::shadow_stack_get(temporary_slot),
            false,
        )?;
    }
    let kwargs = has_kwargs.then(|| _roots.get(kwargs_slot));
    let w_kwargs = if kwargs.is_some() {
        Some(pyre_object::gc_roots::shadow_stack_get(
            operands_sp + 1 + usize::from(positional.len() == 2),
        ))
    } else {
        None
    };
    namespace_apply_kwargs(
        pyre_object::gc_roots::shadow_stack_get(operands_sp),
        w_kwargs,
    )
}

fn simple_namespace_method(
    name: &'static str,
    function: fn(&[PyObjectRef]) -> crate::PyResult,
    arity: u16,
    doc: &'static str,
    text_signature: &'static str,
) -> PyObjectRef {
    let _roots = pyre_object::gc_roots::push_roots();
    let slot = pyre_object::gc_roots::shadow_stack_len();
    let method =
        crate::gateway::make_builtin_function_with_arity_and_doc(name, function, arity, doc);
    let _ = pyre_object::gc_roots::pin_root(method);
    let signature = w_str_new(text_signature);
    let method = pyre_object::gc_roots::shadow_stack_get(slot);
    unsafe { crate::function::fset_func_text_signature(method, signature) };
    method
}

fn simple_namespace_variadic_method(
    name: &'static str,
    function: fn(&[PyObjectRef]) -> crate::PyResult,
    doc: &'static str,
    text_signature: &'static str,
) -> PyObjectRef {
    let _roots = pyre_object::gc_roots::push_roots();
    let slot = pyre_object::gc_roots::shadow_stack_len();
    let method = crate::gateway::make_builtin_function_with_doc(name, function, doc);
    let _ = pyre_object::gc_roots::pin_root(method);
    let signature = w_str_new(text_signature);
    let method = pyre_object::gc_roots::shadow_stack_get(slot);
    unsafe { crate::function::fset_func_text_signature(method, signature) };
    method
}

fn simple_namespace_new_descr() -> PyObjectRef {
    let _roots = pyre_object::gc_roots::push_roots();
    let slot = pyre_object::gc_roots::shadow_stack_len();
    let descr = crate::typedef::make_new_descr_with_doc(
        simple_namespace_new,
        "Create and return a new object.  See help(type) for accurate signature.",
    );
    let _ = pyre_object::gc_roots::pin_root(descr);
    let signature = w_str_new("($type, *args, **kwargs)");
    let descr = pyre_object::gc_roots::shadow_stack_get(slot);
    unsafe { crate::function::fset_func_text_signature(descr, signature) };
    descr
}

/// Receiver predicate for the gateways materialised from PyPy's app-level
/// `SimpleNamespace` methods.  Its storage remains the ordinary instance/map
/// layout, so the descriptor owner must check Python type membership rather
/// than pretending that every object with that shared layout is a namespace.
#[majit_macros::dont_look_inside]
pub(crate) unsafe fn is_simple_namespace(obj: PyObjectRef) -> bool {
    if obj.is_null() {
        return false;
    }
    crate::typedef::r#type(obj).is_some_and(|obj_type| unsafe {
        crate::baseobjspace::issubtype_w(obj_type.as_ptr(), simple_namespace_type())
    })
}

/// `types.SimpleNamespace` — the attribute-bag type exposed as
/// `type(sys.implementation)` and re-published by `types.py`
/// (`SimpleNamespace = type(sys.implementation)`).
///
/// `lib_pypy._structseq.SimpleNamespace`, with CPython 3.14's newer constructor,
/// full rich-comparison surface, pickle reducer and `__replace__`.  Storage
/// remains PyPy-shaped: the values live in the instance dict, not a side
/// table or a second native mapping.
pub(crate) fn simple_namespace_type() -> PyObjectRef {
    static TYPE: pyre_object::gc_roots::RootedOnceRef = pyre_object::gc_roots::RootedOnceRef::new();
    TYPE.get_or_init(|| {
        // PyPy owns this as the app-level class
        // `lib_pypy._structseq.SimpleNamespace`, so it keeps `object`'s
        // TypeDef/Layout instead of introducing an interpreter TypeDef merely
        // because pyre materialises its methods in Rust.
        let object = crate::typedef::w_object();
        let object_layout = unsafe { pyre_object::w_type_get_layout_ptr(object) };
        let object_typedef = unsafe { (*object_layout).typedef };
        let tp = crate::typedef::make_builtin_type_with_overridetypedef(
            "types.SimpleNamespace",
            |ns| {
                let _roots = pyre_object::gc_roots::push_roots();
                let ns_slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(ns);

                macro_rules! install {
                    ($name:literal, $value:expr) => {{
                        // The value pin lives across this one setitem. `ns_slot`
                        // stays on `_roots`.
                        let _pyre_store_roots = pyre_object::gc_roots::push_roots();
                        let value = $value;
                        let value_slot = pyre_object::gc_roots::shadow_stack_len();
                        let _ = pyre_object::gc_roots::pin_root(value);
                        let ns = pyre_object::gc_roots::shadow_stack_get(ns_slot);
                        let value = pyre_object::gc_roots::shadow_stack_get(value_slot);
                        unsafe {
                            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                                ns, $name, value,
                            )
                        };
                    }};
                }

                // PyPy `lib_pypy._structseq.SimpleNamespace` is an app-level
                // class: the class-body rows lead the namespace in source
                // order.  CPython 3.14 contributes the additional observable
                // slots below, but does not replace that storage owner.
                // [3.14-spec] `PyNamespace_Type`'s docs/signatures and rich
                // slots ↔ `lib_pypy._structseq.SimpleNamespace`'s smaller
                // class body — callers observe the former metadata while the
                // instance dict and control-flow shape remain the latter's.
                install!("__module__", w_str_new("types"));
                install!("__doc__", w_str_new("A simple attribute-based namespace."));
                install!("__new__", simple_namespace_new_descr());
                install!(
                    "__init__",
                    simple_namespace_variadic_method(
                        "__init__",
                        simple_namespace_init,
                        "Initialize self.  See help(type(self)) for accurate signature.",
                        "($self, /, *args, **kwargs)",
                    )
                );
                install!(
                    "__repr__",
                    simple_namespace_method(
                        "__repr__",
                        simple_namespace_repr,
                        1,
                        "Return repr(self).",
                        "($self, /)",
                    )
                );
                install!(
                    "__eq__",
                    simple_namespace_method(
                        "__eq__",
                        simple_namespace_eq,
                        2,
                        "Return self==value.",
                        "($self, value, /)",
                    )
                );
                install!(
                    "__ne__",
                    simple_namespace_method(
                        "__ne__",
                        simple_namespace_ne,
                        2,
                        "Return self!=value.",
                        "($self, value, /)",
                    )
                );
                for (name, function, doc) in [
                    (
                        "__lt__",
                        simple_namespace_lt as fn(&[PyObjectRef]) -> crate::PyResult,
                        "Return self<value.",
                    ),
                    ("__le__", simple_namespace_le, "Return self<=value."),
                    ("__gt__", simple_namespace_gt, "Return self>value."),
                    ("__ge__", simple_namespace_ge, "Return self>=value."),
                ] {
                    // The method pin lives across this one setitem. `ns_slot`
                    // stays on `_roots`.
                    let _pyre_store_roots = pyre_object::gc_roots::push_roots();
                    let method = simple_namespace_method(
                        name,
                        function,
                        2,
                        doc,
                        "($self, value, /)",
                    );
                    let method_slot = pyre_object::gc_roots::shadow_stack_len();
                    let _ = pyre_object::gc_roots::pin_root(method);
                    let ns = pyre_object::gc_roots::shadow_stack_get(ns_slot);
                    let method = pyre_object::gc_roots::shadow_stack_get(method_slot);
                    unsafe {
                        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                            ns, name, method,
                        )
                    };
                }
                install!(
                    "__reduce__",
                    simple_namespace_method(
                        "__reduce__",
                        simple_namespace_reduce,
                        1,
                        "Return state information for pickling",
                        "($self, /)",
                    )
                );
                install!(
                    "__replace__",
                    simple_namespace_variadic_method(
                        "__replace__",
                        simple_namespace_replace,
                        "Return a copy of the namespace object with new values for the specified attributes.",
                        "($self, /, **changes)",
                    )
                );
                install!("__dict__", crate::typedef::dict_descr());
            },
            object,
            object_typedef,
        );
        unsafe {
            w_type_set_hasdict(tp, true);
            pyre_object::w_type_set_text_signature(tp, "(mapping_or_iterable=(), /, **kwargs)");
        }
        tp
    })
}

/// CPython 3.14 `namespace_repr`, layered over PyPy's recursion guard.  Exact
/// instances use `namespace`, subclasses use their concrete type name.  Walk
/// a snapshot of the insertion-ordered keys, then re-read each live value so
/// a re-entrant repr that mutates the dict has CPython's skip/update behavior.
fn simple_namespace_repr(args: &[PyObjectRef]) -> crate::PyResult {
    let Some(&self_obj) = args.first() else {
        return Err(crate::PyError::type_error(
            "__repr__() missing 1 required positional argument: 'self'",
        ));
    };
    let _roots = pyre_object::gc_roots::push_roots();
    let sp = pyre_object::gc_roots::shadow_stack_len();
    let self_obj = pyre_object::gc_roots::pin_root(self_obj);
    let actual_type = crate::typedef::r#type(self_obj)
        .map(|tp| tp.as_ptr())
        .unwrap_or(simple_namespace_type());
    let name = if std::ptr::eq(actual_type, simple_namespace_type()) {
        "namespace".to_string()
    } else {
        unsafe { w_type_get_name(actual_type) }.to_string()
    };
    let Some(_guard) = crate::display::ReprGuard::enter(self_obj) else {
        return Ok(w_str_new_managed(&format!("{name}(...)")));
    };
    let dict =
        crate::baseobjspace::getattr_str(pyre_object::gc_roots::shadow_stack_get(sp), "__dict__")?;
    let _ = pyre_object::gc_roots::pin_root(dict);
    let keys = unsafe {
        pyre_object::w_dict_items(pyre_object::gc_roots::shadow_stack_get(sp + 1))
            .into_iter()
            .map(|(key, _)| key)
            .collect::<Vec<_>>()
    };
    let keys_sp = pyre_object::gc_roots::shadow_stack_len();
    for &key in &keys {
        let _ = pyre_object::gc_roots::pin_root(key);
    }
    let mut parts = Vec::with_capacity(keys.len());
    for i in 0..keys.len() {
        let key = pyre_object::gc_roots::shadow_stack_get(keys_sp + i);
        if !unsafe { pyre_object::is_str(key) } || unsafe { pyre_object::w_str_len(key) == 0 } {
            continue;
        }
        let value = match crate::baseobjspace::getitem(
            pyre_object::gc_roots::shadow_stack_get(sp + 1),
            key,
        ) {
            Ok(value) => value,
            Err(err) if err.kind == crate::PyErrorKind::KeyError => continue,
            Err(err) => return Err(err),
        };
        let value_sp = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(value);
        // `getitem` above ran a lookup that can collect, so the key has to be
        // reread from its slot rather than reused from before the call.
        let key = pyre_object::gc_roots::shadow_stack_get(keys_sp + i);
        parts.push(crate::display::wtf8_format!(
            unsafe { crate::display::py_str_wtf8(key)? },
            "=",
            unsafe {
                crate::display::py_repr_wtf8(pyre_object::gc_roots::shadow_stack_get(value_sp))?
            }
        ));
    }
    let mut text = rustpython_wtf8::Wtf8Buf::from_string(format!("{name}("));
    for (index, part) in parts.iter().enumerate() {
        if index > 0 {
            text.push_str(", ");
        }
        text.push_wtf8(part);
    }
    text.push_str(")");
    Ok(pyre_object::w_str_from_wtf8_managed(text))
}

/// `lib_pypy._structseq.SimpleNamespace.__eq__` — structural over `__dict__`
/// when `other` is a namespace, NotImplemented otherwise.
fn simple_namespace_eq(args: &[PyObjectRef]) -> crate::PyResult {
    simple_namespace_richcompare(args, "__eq__", crate::baseobjspace::CompareOp::Eq)
}

/// `lib_pypy._structseq.SimpleNamespace.__ne__`.
fn simple_namespace_ne(args: &[PyObjectRef]) -> crate::PyResult {
    simple_namespace_richcompare(args, "__ne__", crate::baseobjspace::CompareOp::Ne)
}

fn simple_namespace_lt(args: &[PyObjectRef]) -> crate::PyResult {
    simple_namespace_richcompare(args, "__lt__", crate::baseobjspace::CompareOp::Lt)
}

fn simple_namespace_le(args: &[PyObjectRef]) -> crate::PyResult {
    simple_namespace_richcompare(args, "__le__", crate::baseobjspace::CompareOp::Le)
}

fn simple_namespace_gt(args: &[PyObjectRef]) -> crate::PyResult {
    simple_namespace_richcompare(args, "__gt__", crate::baseobjspace::CompareOp::Gt)
}

fn simple_namespace_ge(args: &[PyObjectRef]) -> crate::PyResult {
    simple_namespace_richcompare(args, "__ge__", crate::baseobjspace::CompareOp::Ge)
}

fn simple_namespace_richcompare(
    args: &[PyObjectRef],
    name: &str,
    op: crate::baseobjspace::CompareOp,
) -> crate::PyResult {
    // `def __eq__(self, other)` — a missing argument is an arity error, not a
    // NotImplemented result.
    let (Some(&self_obj), Some(&other)) = (args.first(), args.get(1)) else {
        return Err(crate::PyError::type_error(format!(
            "SimpleNamespace.{name}() missing 1 required positional argument: '{}'",
            if args.is_empty() { "self" } else { "other" }
        )));
    };
    let other_type = crate::typedef::r#type(other)
        .map(|tp| tp.as_ptr())
        .unwrap_or(PY_NULL);
    if !unsafe { crate::baseobjspace::issubtype_w(other_type, simple_namespace_type()) } {
        return Ok(w_not_implemented());
    }
    // CPython 3.14 forwards all six operations to the two namespace dicts.
    // In particular, ordering reaches dict's TypeError instead of returning
    // NotImplemented from the namespace type itself.
    //
    // A subclass `__getattribute__` runs on each lookup, so both operands and
    // the first dict have to survive it.
    let _roots = pyre_object::gc_roots::push_roots();
    let sp = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(self_obj);
    let _ = pyre_object::gc_roots::pin_root(other);
    let self_dict =
        crate::baseobjspace::getattr_str(pyre_object::gc_roots::shadow_stack_get(sp), "__dict__")?;
    let _ = pyre_object::gc_roots::pin_root(self_dict);
    let other_dict = crate::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(sp + 1),
        "__dict__",
    )?;
    crate::baseobjspace::compare(
        pyre_object::gc_roots::shadow_stack_get(sp + 2),
        other_dict,
        op,
    )
}

/// CPython 3.14 `namespace_reduce`: `(type(self), (), self.__dict__)`.
fn simple_namespace_reduce(args: &[PyObjectRef]) -> crate::PyResult {
    let Some(&self_obj) = args.first() else {
        return Err(crate::PyError::type_error(
            "__reduce__() missing 1 required positional argument: 'self'",
        ));
    };
    let _roots = pyre_object::gc_roots::push_roots();
    let sp = pyre_object::gc_roots::shadow_stack_len();
    let self_obj = pyre_object::gc_roots::pin_root(self_obj);
    let w_type = crate::typedef::r#type(self_obj)
        .map(|tp| tp.as_ptr())
        .unwrap_or(PY_NULL);
    let _ = pyre_object::gc_roots::pin_root(w_type);
    let w_args = w_tuple_new(Vec::new());
    let _ = pyre_object::gc_roots::pin_root(w_args);
    let w_dict =
        crate::baseobjspace::getattr_str(pyre_object::gc_roots::shadow_stack_get(sp), "__dict__")?;
    let _ = pyre_object::gc_roots::pin_root(w_dict);
    Ok(w_tuple_new(vec![
        pyre_object::gc_roots::shadow_stack_get(sp + 1),
        pyre_object::gc_roots::shadow_stack_get(sp + 2),
        pyre_object::gc_roots::shadow_stack_get(sp + 3),
    ]))
}

/// CPython 3.14 `namespace_replace`: construct `type(self)()` first, require
/// that its actual type remains a SimpleNamespace subtype, copy the source
/// dict, then overlay keyword changes.
fn simple_namespace_replace(args: &[PyObjectRef]) -> crate::PyResult {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let Some(&self_obj) = positional.first() else {
        return Err(crate::PyError::type_error(
            "__replace__() missing 1 required positional argument: 'self'",
        ));
    };
    if positional.len() != 1 {
        return Err(crate::PyError::type_error(
            "__replace__() takes no positional arguments",
        ));
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let sp = pyre_object::gc_roots::shadow_stack_len();
    let self_obj = pyre_object::gc_roots::pin_root(self_obj);
    // Only the presence of keyword changes is tested below; the dict itself
    // is read from its slot.
    let has_kwargs = kwargs.is_some();
    if let Some(kwargs) = kwargs {
        let _ = pyre_object::gc_roots::pin_root(kwargs);
    }
    let self_type = crate::typedef::r#type(self_obj)
        .map(|tp| tp.as_ptr())
        .unwrap_or(PY_NULL);
    // `namespace___replace__` is a method on the namespace type, so a foreign
    // receiver is rejected by the descriptor before anything else runs — in
    // particular before `type(self)()` could execute a foreign constructor.
    if !unsafe { crate::baseobjspace::issubtype_w(self_type, simple_namespace_type()) } {
        let received = if self_type.is_null() {
            rustpython_wtf8::Wtf8Buf::from("object")
        } else {
            unsafe { crate::baseobjspace::type_fully_qualified_name(self_type) }
        };
        return Err(crate::PyError::type_error(crate::display::wtf8_format!(
            "descriptor '__replace__' for 'types.SimpleNamespace' objects doesn't apply to a '",
            received,
            "' object"
        )));
    }
    let _ = pyre_object::gc_roots::pin_root(self_type);
    let result = crate::call::call_function_impl_result(
        pyre_object::gc_roots::shadow_stack_get(sp + 1 + usize::from(has_kwargs)),
        &[],
    )?;
    let _ = pyre_object::gc_roots::pin_root(result);
    // Reread from this slot at every use below: the `__dict__` lookup and each
    // `namespace_update_dict` can run Python and collect, so a local captured
    // here would name the pre-relocation address.
    let result_slot = sp + 2 + usize::from(has_kwargs);
    let result = pyre_object::gc_roots::shadow_stack_get(result_slot);
    let result_type = crate::typedef::r#type(result)
        .map(|tp| tp.as_ptr())
        .unwrap_or(PY_NULL);
    if !unsafe { crate::baseobjspace::issubtype_w(result_type, simple_namespace_type()) } {
        let constructed = unsafe {
            crate::baseobjspace::type_fully_qualified_name(pyre_object::gc_roots::shadow_stack_get(
                sp + 1 + usize::from(has_kwargs),
            ))
        };
        let returned = if result_type.is_null() {
            "object"
        } else {
            unsafe { w_type_get_name(result_type) }
        };
        return Err(crate::PyError::type_error(crate::display::wtf8_format!(
            "expect types.SimpleNamespace type, but ",
            constructed,
            format!("() returned '{returned}' object")
        )));
    }
    let source_dict =
        crate::baseobjspace::getattr_str(pyre_object::gc_roots::shadow_stack_get(sp), "__dict__")?;
    let _ = pyre_object::gc_roots::pin_root(source_dict);
    namespace_update_dict(
        pyre_object::gc_roots::shadow_stack_get(result_slot),
        pyre_object::gc_roots::shadow_stack_get(sp + 3 + usize::from(has_kwargs)),
        false,
    )?;
    if has_kwargs {
        namespace_update_dict(
            pyre_object::gc_roots::shadow_stack_get(result_slot),
            pyre_object::gc_roots::shadow_stack_get(sp + 1),
            true,
        )?;
    }
    Ok(pyre_object::gc_roots::shadow_stack_get(result_slot))
}

/// `pypy/module/sys/vm.py settrace space.getexecutioncontext()` access for
/// `sys.gettrace`/`settrace`/`getprofile`/`setprofile`.
///
/// Pyre's `crate::call::getexecutioncontext` returns the TLS-cached
/// active context (set on eval-loop entry); see the helper's doc for
/// the staleness gap relative to PyPy's `space.getexecutioncontext()`
/// which always queries the thread state.
/// `pypy/module/sys/vm.py getframe` — the non-hidden frame-chain walk that
/// `sys._getframe` hands out.
///
/// ```python
/// @jit.look_inside_iff(lambda space, depth: jit.isconstant(depth))
/// def getframe(space, depth):
///     ec = space.getexecutioncontext()
///     f = ec.gettopframe_nohidden()
///     while True:
///         if f is None:
///             raise oefmt(space.w_ValueError, "call stack is not deep enough")
///         if depth == 0:
///             f.mark_as_escaped()
///             audit(space, "sys._getframe", [f])
///             return f
///         depth -= 1
///         f = ec.getnextframe_nohidden(f)
/// ```
///
/// `hidden_applevel` gateway / bridge frames are skipped, matching `f_back`.
/// The `f is None` test runs at the START of every iteration including the
/// first, so a missing top frame raises rather than fabricating a stub, and a
/// negative depth walks the chain to its end and raises there.
///
/// The returned object is the live `PyFrame` itself (`FRAME_TYPE` typedef), not
/// a stub, so `frame.f_globals is globals()` holds and `f_back` chains lazily
/// through `pyframe.py GetSetProperty(PyFrame.fget_f_back)`.
///
/// The walk takes no virtualizable force.  This is load-bearing upstream
/// structure, not merely a fast path: `getframe` follows `f_backref`, marks the
/// result escaped and returns it.  `rvirtualizable.py hook_access_field` puts
/// `jit_force_virtualizable` at the later redirected FIELD access, and
/// `jtransform.py rewrite_op_jit_force_virtualizable` deletes it only from a
/// graph the codewriter looks inside.  Pyre mirrors that split in the manual
/// frame descriptor gateways (`pyframe.rs __majit_wrap_descr_typecheck_fget_*`): their
/// runtime bodies retain the marker, while every gateway is published through
/// `BUILTIN_WRAPPER_DESCRIPTORS` so the JIT sees and deletes it.
///
/// Putting the force here, before the consumer is known, is not an equivalent
/// relocation.  Inside a `CALL_MAY_FORCE` it clears `TOKEN_TRACING_RESCALL`, so
/// `vable_after_residual_call` aborts a trace even when the caller only asks for
/// frame identity or an invariant field whose gateway the JIT can see.
///
/// `try_walker_specialize_sys_getframe`
/// (`pyre-jit-trace/src/jitcode_dispatch/specialize.rs`) reproduces upstream's
/// traced-through depth-0 form for either the portal or an inline MIFrame.  It
/// also resolves a guarded positive-depth inline chain when the exact portal
/// result is immediately consumed by `f_locals`; `fast2locals` is
/// `@jit.unroll_safe`, so upstream reaches that mapping by reading the
/// virtualizable boxes instead of forcing, and the specialized `f_locals`
/// writes the locals region back out of the image in the dropped force's
/// place.
///
/// Every other shape still arrives here, and the force stays load-bearing for
/// it.  In particular, generic positive-depth consumers and inline
/// `_getframe().f_lineno` / `f_lasti` remain residual until each app-level frame
/// getter carries the same per-frame red identity and resume coordinate.  The
/// guarded positive-depth specialization admits only its completed immediate
/// `f_locals` slice, so it cannot bake a stale inline `last_instr` into those
/// other getters.
/// `pypy/module/sys/vm.py getframe` — `@jit.look_inside_iff(lambda space, depth: jit.isconstant(depth))`.
fn getframe_iff(depth: i64) -> bool {
    majit_rlib::jit::isconstant(&depth)
}

#[majit_macros::look_inside_iff(getframe_iff)]
pub fn getframe(depth: i64) -> crate::PyResult {
    let ec = current_execution_context();
    let mut current = if ec.is_null() {
        std::ptr::null_mut()
    } else {
        unsafe { (*ec).gettopframe_nohidden() }
    };
    let mut remaining = depth;
    loop {
        if current.is_null() {
            return Err(crate::PyError::value_error("call stack is not deep enough"));
        }
        if remaining == 0 {
            break;
        }
        remaining -= 1;
        current = crate::executioncontext::ExecutionContext::getnextframe_nohidden(current);
    }
    unsafe { (*current).mark_as_escaped() };
    // `vm.py audit(space, "sys._getframe", [f])`.  The event follows
    // `mark_as_escaped`; a hook that reads a redirected frame field reaches
    // that field's descriptor gateway and forces there.
    if audit_hooks_armed() {
        // The wrap of the event name and the hooks themselves are both
        // collection points, and the frame this is about to return reaches them
        // only as a copied pointer, which the collector does not rewrite.  Root
        // it across the emit and read the answer back out of its slot.
        let _roots = pyre_object::gc_roots::push_roots();
        let frame_slot = pyre_object::gc_roots::pin_roots(&[current as PyObjectRef]);
        audit("sys._getframe", &[current as PyObjectRef])?;
        current = pyre_object::gc_roots::shadow_stack_get(frame_slot) as *mut crate::PyFrame;
    }
    Ok(current as PyObjectRef)
}

/// `pypy/module/sys/vm.py _getframe` — `@unwrap_spec(depth=int)`, the
/// argument surface in front of [`getframe`].
///
/// A named `fn` rather than a closure so [`is_builtin_getframe_function`] can
/// compare builtin-code identity against it.
fn sys_getframe(args: &[PyObjectRef]) -> crate::PyResult {
    // `unwrap_spec` enforces a single optional int argument, so any extra
    // positional arg must surface as TypeError before the depth walk runs.
    if args.len() > 1 {
        return Err(crate::PyError::type_error(format!(
            "_getframe expected at most 1 argument, got {}",
            args.len()
        )));
    }
    let depth = if args.is_empty() {
        0i64
    } else if unsafe { pyre_object::is_int(args[0]) } {
        unsafe { pyre_object::w_int_get_value(args[0]) }
    } else {
        return Err(crate::PyError::type_error(
            "_getframe(): argument must be an int",
        ));
    };
    // `vm.py:37-38 if depth < 0: raise ... "frame index must not be negative"`
    // — the message differs from `getframe`'s exhausted-stack case.
    if depth < 0 {
        return Err(crate::PyError::value_error(
            "frame index must not be negative",
        ));
    }
    getframe(depth)
}

/// True iff `callable` is the canonical `sys._getframe` builtin.
///
/// `sys` is an ordinary mutable module, so the JIT walker has to key on
/// builtin-code identity rather than the global name before it may reproduce
/// [`getframe`] in a trace. Mirrors `builtins::is_builtin_len_function`.
pub fn is_builtin_getframe_function(callable: PyObjectRef) -> bool {
    unsafe {
        if callable.is_null() || !crate::is_function(callable) {
            return false;
        }
        let code = crate::function_get_code(callable) as PyObjectRef;
        if code.is_null() || !crate::gateway::is_builtin_code(code) {
            return false;
        }
        crate::gateway::builtin_code_fn_eq(
            crate::gateway::builtin_code_get(code),
            sys_getframe as crate::gateway::BuiltinCodeFn,
        )
    }
}

fn current_execution_context() -> *mut crate::PyExecutionContext {
    crate::call::getexecutioncontext() as *mut crate::PyExecutionContext
}

fn sys_gettrace_impl(_args: &[PyObjectRef]) -> crate::PyResult {
    let ec = current_execution_context();
    if ec.is_null() {
        return Ok(w_none());
    }
    let w_trace = unsafe { (*ec).gettrace() };
    Ok(if w_trace.is_null() { w_none() } else { w_trace })
}

fn sys_settrace_impl(args: &[PyObjectRef]) -> crate::PyResult {
    // pypy/module/sys/vm.py `def settrace(space, w_func)` — w_func is
    // a required positional. Calling `sys.settrace()` with no args raises
    // TypeError at the gateway layer in PyPy; reproduce that here.
    let w_func = *args.first().ok_or_else(|| {
        crate::PyError::type_error("settrace() missing 1 required positional argument: 'function'")
    })?;
    let ec = current_execution_context();
    if !ec.is_null() {
        unsafe { (*ec).settrace(w_func) };
    }
    Ok(w_none())
}

fn sys_getprofile_impl(_args: &[PyObjectRef]) -> crate::PyResult {
    let ec = current_execution_context();
    if ec.is_null() {
        return Ok(w_none());
    }
    let w_profile = unsafe { (*ec).getprofile() };
    Ok(if w_profile.is_null() {
        w_none()
    } else {
        w_profile
    })
}

fn sys_setprofile_impl(args: &[PyObjectRef]) -> crate::PyResult {
    // pypy/module/sys/vm.py `def setprofile(space, w_func)` — w_func
    // is a required positional. Calling `sys.setprofile()` with no args
    // raises TypeError at the gateway layer in PyPy.
    let w_func = *args.first().ok_or_else(|| {
        crate::PyError::type_error(
            "setprofile() missing 1 required positional argument: 'function'",
        )
    })?;
    let ec = current_execution_context();
    if !ec.is_null() {
        // executioncontext.py ValueError("Cannot call setllprofile
        // with real None") propagates via setprofile -> setllprofile.
        unsafe { (*ec).setprofile(w_func)? };
    }
    Ok(w_none())
}

fn sys_settraceallthreads_impl(args: &[PyObjectRef]) -> crate::PyResult {
    let w_func = *args.first().ok_or_else(|| {
        crate::PyError::type_error(
            "_settraceallthreads() missing 1 required positional argument: 'function'",
        )
    })?;
    crate::module::thread::set_trace_all_execution_contexts(w_func);
    Ok(w_none())
}

fn sys_setprofileallthreads_impl(args: &[PyObjectRef]) -> crate::PyResult {
    let w_func = *args.first().ok_or_else(|| {
        crate::PyError::type_error(
            "_setprofileallthreads() missing 1 required positional argument: 'function'",
        )
    })?;
    crate::module::thread::set_profile_all_execution_contexts(w_func)?;
    Ok(w_none())
}

fn sys_get_coroutine_origin_tracking_depth(_args: &[PyObjectRef]) -> crate::PyResult {
    let ec = current_execution_context();
    let depth = if ec.is_null() {
        0
    } else {
        unsafe { (*ec).coroutine_origin_tracking_depth }
    };
    Ok(pyre_object::w_int_new(depth))
}

fn sys_set_coroutine_origin_tracking_depth(args: &[PyObjectRef]) -> crate::PyResult {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    crate::builtins::kwarg_reject_unknown(
        kwargs,
        &["depth"],
        "set_coroutine_origin_tracking_depth",
    )?;
    if positional.len() > 1 {
        return Err(crate::PyError::type_error(format!(
            "set_coroutine_origin_tracking_depth() takes exactly one argument ({} given)",
            positional.len(),
        )));
    }
    let kw_depth = crate::builtins::kwarg_get(kwargs, "depth");
    if !positional.is_empty() && kw_depth.is_some() {
        return Err(crate::PyError::type_error(
            "set_coroutine_origin_tracking_depth() got multiple values for argument 'depth'",
        ));
    }
    let w_depth = positional.first().copied().or(kw_depth).ok_or_else(|| {
        crate::PyError::type_error(
            "set_coroutine_origin_tracking_depth() missing required argument 'depth'",
        )
    })?;
    let indexed = crate::baseobjspace::space_index(w_depth)?;
    let depth = crate::baseobjspace::int_w(indexed)?;
    if depth < 0 {
        return Err(crate::PyError::value_error("depth must be >= 0"));
    }
    let ec = current_execution_context();
    if !ec.is_null() {
        unsafe { (*ec).coroutine_origin_tracking_depth = depth };
    }
    Ok(w_none())
}

fn asyncgen_hooks_type() -> PyObjectRef {
    static TYPE: pyre_object::gc_roots::RootedOnceRef = pyre_object::gc_roots::RootedOnceRef::new();
    TYPE.get_or_init(|| {
        crate::_structseq::make_struct_seq("asyncgen_hooks", &["firstiter", "finalizer"])
    })
}

fn sys_get_asyncgen_hooks_impl(_args: &[PyObjectRef]) -> crate::PyResult {
    let ec = current_execution_context();
    let (firstiter, finalizer) = if ec.is_null() {
        (w_none(), w_none())
    } else {
        unsafe {
            (
                if (*ec).w_asyncgen_firstiter_fn.is_null() {
                    w_none()
                } else {
                    (*ec).w_asyncgen_firstiter_fn
                },
                if (*ec).w_asyncgen_finalizer_fn.is_null() {
                    w_none()
                } else {
                    (*ec).w_asyncgen_finalizer_fn
                },
            )
        }
    };
    let mut fields = pyre_object::gc_roots::RootedItems::new();
    fields.push(firstiter);
    fields.push(finalizer);
    Ok(crate::_structseq::new_instance(
        asyncgen_hooks_type(),
        fields.take(),
    ))
}

fn sys_set_asyncgen_hooks_impl(args: &[PyObjectRef]) -> crate::PyResult {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    crate::builtins::kwarg_reject_unknown(
        kwargs,
        &["firstiter", "finalizer"],
        "set_asyncgen_hooks",
    )?;
    if positional.len() > 2 {
        return Err(crate::PyError::type_error(format!(
            "set_asyncgen_hooks() takes at most 2 arguments ({} given)",
            positional.len()
        )));
    }
    let kw_firstiter = crate::builtins::kwarg_get(kwargs, "firstiter");
    let kw_finalizer = crate::builtins::kwarg_get(kwargs, "finalizer");
    if !positional.is_empty() && kw_firstiter.is_some() {
        return Err(crate::PyError::type_error(
            "set_asyncgen_hooks() got multiple values for argument 'firstiter'",
        ));
    }
    if positional.len() > 1 && kw_finalizer.is_some() {
        return Err(crate::PyError::type_error(
            "set_asyncgen_hooks() got multiple values for argument 'finalizer'",
        ));
    }
    let firstiter = positional.first().copied().or(kw_firstiter);
    let finalizer = positional.get(1).copied().or(kw_finalizer);
    let ec = current_execution_context();
    if !ec.is_null() {
        unsafe {
            // PyPy vm.py:set_asyncgen_hooks updates and validates finalizer
            // first, then firstiter.  Preserve that observable ordering when
            // the second argument is invalid.
            if let Some(value) = finalizer {
                if is_none(value) {
                    (*ec).w_asyncgen_finalizer_fn = PY_NULL;
                } else if crate::baseobjspace::callable_w(value) {
                    (*ec).w_asyncgen_finalizer_fn = value;
                } else {
                    return Err(crate::PyError::type_error(format!(
                        "callable finalizer expected, got {}",
                        crate::type_methods::arg_type_name(value)
                    )));
                }
            }
            if let Some(value) = firstiter {
                if is_none(value) {
                    (*ec).w_asyncgen_firstiter_fn = PY_NULL;
                } else if crate::baseobjspace::callable_w(value) {
                    (*ec).w_asyncgen_firstiter_fn = value;
                } else {
                    return Err(crate::PyError::type_error(format!(
                        "callable firstiter expected, got {}",
                        crate::type_methods::arg_type_name(value)
                    )));
                }
            }
        }
    }
    Ok(w_none())
}

/// `app.py breakpointhook` — the hook `breakpoint()` calls.
///
/// `$PYTHONBREAKPOINT` names the callable to run: unset or empty means
/// `pdb.set_trace`, `"0"` disables the hook outright, and any other value is
/// imported as a dotted path (a bare name resolves against `builtins`).  An
/// unimportable value warns and returns None rather than propagating, so a
/// stray environment variable cannot break an otherwise working program.
fn sys_breakpointhook(args: &[PyObjectRef]) -> crate::PyResult {
    // `args` is the gateway's native copy; importing the hook runs Python.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = _roots.pin_roots(args);
    let hookname = match crate::importing::host::os::var("PYTHONBREAKPOINT") {
        Ok(name) if name == "0" => return Ok(w_none()),
        Ok(name) if !name.is_empty() => name,
        _ => "pdb.set_trace".to_string(),
    };
    let (modname, funcname) = match hookname.rsplit_once('.') {
        Some((modname, funcname)) => (modname, funcname),
        None => ("builtins", hookname.as_str()),
    };

    // `dunder_import` returns the root package, so reach the leaf through
    // `sys.modules` before pulling the attribute off it.
    let hook = crate::importing::dunder_import(
        modname,
        pyre_object::PY_NULL,
        pyre_object::PY_NULL,
        pyre_object::PY_NULL,
        0,
        std::ptr::null(),
    )
    .ok()
    .and_then(|_| crate::importing::get_sys_module(modname))
    .and_then(|module| crate::baseobjspace::getattr_str(module, funcname).ok());
    let Some(hook) = hook else {
        crate::warn::warn_category(
            &format!("Ignoring unimportable $PYTHONBREAKPOINT: \"{hookname}\""),
            "RuntimeWarning",
            1,
        )?;
        return Ok(w_none());
    };
    // The hook "must accept whatever arguments are passed".
    let mut live_args = Vec::with_capacity(args.len());
    for i in 0..args.len() {
        live_args.push(_roots.get(base + i));
    }
    crate::builtins::call_forwarding_args(hook, &live_args)
}

/// `sys._baserepl` — `PyRun_AnyFileExFlags(stdin, "<stdin>", 0, &cf)`: read
/// stdin and execute what it holds in `__main__`'s namespace.
/// `_pyrepl.main.interactive_console` calls this whenever the enhanced REPL
/// cannot run, so a body that returns without reading anything ends the
/// session instead of starting it.
///
/// `code.interact` is that loop at applevel, over the same namespace.  pyre's
/// own prompt cannot answer here: `pyrex::repl::run_repl` builds a fresh
/// `PyExecutionContext` and a fresh `__main__`, which is a top-level driver
/// rather than something a running program can call into.
///
/// `_Py_FdIsInteractive` sends a terminal — and `-i` — down that prompt
/// loop, and everything else to `_PyRun_SimpleFileObject`, which compiles the
/// whole of stdin at once and reports a failure through `PyErr_Print`.  Only
/// the prompt loop is reproduced: `PyErr_Print` belongs to the driver that
/// owns process exit, which this crate sits below, so redirected stdin runs
/// here statement by statement with a prompt written between each.
fn sys_baserepl(_args: &[PyObjectRef]) -> crate::PyResult {
    crate::importing::dunder_import(
        "code",
        pyre_object::PY_NULL,
        pyre_object::PY_NULL,
        pyre_object::PY_NULL,
        0,
        std::ptr::null(),
    )?;
    let code_module = crate::importing::get_sys_module("code")
        .ok_or_else(|| crate::PyError::runtime_error("no module named 'code'"))?;
    // Every argument is pinned as it is produced and read back at the call:
    // the attribute lookups and string allocations between them can each run a
    // collection that forwards the slot without touching a local copy.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::shadow_stack_len();
    let _ =
        pyre_object::gc_roots::pin_root(crate::baseobjspace::getattr_str(code_module, "interact")?);
    // `interact(banner, readfunc, local, exitmsg)`.  Both messages are empty:
    // the console this stands in for prints neither a banner nor a farewell.
    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_new(""));
    let main_module = crate::importing::get_sys_module("__main__")
        .ok_or_else(|| crate::PyError::runtime_error("no module named '__main__'"))?;
    let _ =
        pyre_object::gc_roots::pin_root(crate::baseobjspace::getattr_str(main_module, "__dict__")?);
    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_new(""));
    let slot = pyre_object::gc_roots::shadow_stack_get;
    crate::call::call_function_impl_result(
        slot(base),
        &[slot(base + 1), w_none(), slot(base + 2), slot(base + 3)],
    )?;
    Ok(w_none())
}

fn sys_unraisablehook(args: &[PyObjectRef]) -> crate::PyResult {
    let Some(&(mut w_hookargs)) = args.first() else {
        return Err(crate::PyError::type_error(
            "unraisablehook() missing 1 required positional argument",
        ));
    };
    let mut w_type = pyre_object::with_roots!(w_hookargs => crate::baseobjspace::getattr_str(w_hookargs, "exc_type"))?;
    let mut w_value = pyre_object::with_roots!(w_hookargs, w_type => crate::baseobjspace::getattr_str(w_hookargs, "exc_value"))?;
    let mut w_tb = pyre_object::with_roots!(w_hookargs, w_type, w_value => crate::baseobjspace::getattr_str(w_hookargs, "exc_traceback"))?;
    let mut w_err_msg = pyre_object::with_roots!(w_hookargs, w_tb, w_type, w_value => crate::baseobjspace::getattr_str(w_hookargs, "err_msg"))?;
    let err_msg = if unsafe { pyre_object::is_none(w_err_msg) } {
        rustpython_wtf8::Wtf8Buf::new()
    } else if unsafe { pyre_object::is_str(w_err_msg) } {
        unsafe { pyre_object::w_str_get_wtf8(w_err_msg) }.to_wtf8_buf()
    } else {
        pyre_object::with_roots!(w_hookargs, w_tb, w_type, w_value, w_err_msg => unsafe { crate::display::py_str_wtf8(w_err_msg) })?
    };
    let w_object = pyre_object::with_roots!(w_tb, w_type, w_value => crate::baseobjspace::getattr_str(w_hookargs, "object"))?;
    crate::PyError::write_unraisable_default(
        w_none(),
        w_type,
        w_value,
        w_tb,
        &err_msg,
        w_object,
        "",
    );
    Ok(w_none())
}

/// Decode the next logical opcode after `pc`, skipping the inline-cache words
/// CPython 3.14 places after adaptive instructions.  PyPy's bytecode has no
/// cache words; this is the representation adapter around its direct `co_code`
/// indexing in `exc_info_direct`.
fn exc_info_next_instruction(
    code: &crate::CodeObject,
    pc: usize,
) -> Option<(usize, crate::bytecode::Instruction, crate::bytecode::OpArg)> {
    let pc = crate::pyopcode::skip_caches(&code.instructions, pc);
    let Ok((opcode_pc, instruction, op_arg)) =
        crate::pyopcode::decode_instruction_forward(code, pc)
    else {
        return None;
    };
    Some((opcode_pc, instruction.deoptimize(), op_arg))
}

/// Return the wrapped constant loaded by `instruction`, preserving the
/// enclosing PyCode's `co_consts_w` identity just like
/// `PyFrame.getconstant_w` in `pypy/interpreter/pyopcode.py`.
fn exc_info_load_const(
    frame: &crate::pyframe::PyFrame,
    instruction: crate::bytecode::Instruction,
    op_arg: crate::bytecode::OpArg,
) -> PyObjectRef {
    match instruction {
        crate::bytecode::Instruction::LoadConst { consti } => unsafe {
            crate::pycode::w_code_const(
                frame.pycode as PyObjectRef,
                usize::from(consti.get(op_arg)),
            )
        },
        // CPython 3.14's compiler uses this compact spelling for the small
        // non-negative indices PyPy spells as LOAD_CONST.  It denotes the
        // same immutable integer input to the look-ahead test.
        crate::bytecode::Instruction::LoadSmallInt { i } => {
            pyre_object::w_small_int_const(i.get(op_arg) as i64)
        }
        _ => pyre_object::PY_NULL,
    }
}

fn exc_info_is_binary_subscr(
    instruction: crate::bytecode::Instruction,
    op_arg: crate::bytecode::OpArg,
) -> bool {
    let crate::bytecode::Instruction::BinaryOp { op } = instruction else {
        return false;
    };
    crate::pyopcode::binary_op_arg(op, op_arg) == crate::bytecode::BinaryOperator::Subscr
}

/// `pypy/module/sys/vm.py exc_info_direct` bytecode look-ahead.
///
/// The traceback can stay unescaped when the caller immediately selects a
/// tuple component/slice that cannot reach slot 2.  This is the JIT-relevant
/// part of PyPy's implementation: returning None in slot 2 prevents the live
/// traceback/frame chain from forcing the virtual frame.
fn exc_info_result_needs_traceback_at(frame: &crate::pyframe::PyFrame, call_pc: usize) -> bool {
    use crate::bytecode::{BuildSliceArgCount, Instruction};

    if frame.hide() {
        return true;
    }
    let code = frame.getcode();
    let Some((call_instruction, _)) = crate::pyopcode::decode_instruction_at(code, call_pc) else {
        return true;
    };
    if !matches!(call_instruction.deoptimize(), Instruction::Call { .. }) {
        return true;
    }

    let Some((first_pc, first_instruction, first_arg)) =
        exc_info_next_instruction(code, call_pc + 1)
    else {
        return true;
    };
    let mut w_first = exc_info_load_const(frame, first_instruction, first_arg);
    if w_first.is_null() {
        return true;
    }

    let Some((second_pc, second_instruction, second_arg)) =
        exc_info_next_instruction(code, first_pc + 1)
    else {
        return true;
    };
    if exc_info_is_binary_subscr(second_instruction, second_arg) {
        // CPython 3.14 folds a two-argument BUILD_SLICE into one immutable
        // slice constant.  Apply PyPy's same start/stop proof to that compact
        // representation before handling the ordinary integer index.
        if unsafe { pyre_object::sliceobject::is_slice(w_first) } {
            let start = unsafe { pyre_object::sliceobject::w_slice_get_start(w_first) };
            let stop = unsafe { pyre_object::sliceobject::w_slice_get_stop(w_first) };
            let mut step = unsafe { pyre_object::sliceobject::w_slice_get_step(w_first) };
            let safe_start = unsafe { pyre_object::is_none(start) }
                || unsafe { crate::baseobjspace::isinstance_int_w(start) };
            let safe_stop = if unsafe { crate::baseobjspace::isinstance_int_w(stop) } {
                match pyre_object::with_roots!(step => crate::baseobjspace::int_w(stop)) {
                    Ok(value) => value <= 2,
                    Err(_) => false,
                }
            } else {
                false
            };
            return !(safe_start && safe_stop && unsafe { pyre_object::is_none(step) });
        }
        if unsafe { crate::baseobjspace::isinstance_int_w(w_first) }
            && let Ok(index) = crate::baseobjspace::int_w(w_first)
        {
            return !((-3..=1).contains(&index) && index != -1);
        }
        return true;
    }

    let w_second = pyre_object::with_roots!(w_first => exc_info_load_const(frame, second_instruction, second_arg));
    if w_second.is_null()
        || !(unsafe { pyre_object::is_none(w_first) }
            || unsafe { crate::baseobjspace::isinstance_int_w(w_first) })
    {
        return true;
    }
    let Ok(stop) = crate::baseobjspace::int_w(w_second) else {
        return true;
    };
    if stop > 2 {
        return true;
    }

    let Some((slice_pc, slice_instruction, slice_arg)) =
        exc_info_next_instruction(code, second_pc + 1)
    else {
        return true;
    };
    let Instruction::BuildSlice { argc } = slice_instruction else {
        return true;
    };
    if crate::pyopcode::build_slice_arg(argc, slice_arg) != BuildSliceArgCount::Two {
        return true;
    }
    let Some((_, subscr_instruction, subscr_arg)) = exc_info_next_instruction(code, slice_pc + 1)
    else {
        return true;
    };
    !exc_info_is_binary_subscr(subscr_instruction, subscr_arg)
}

/// Tracer-facing half of PyPy's `exc_info_direct` look-ahead.
///
/// The generated full-body walker owns the executing Python coordinate rather
/// than publishing it to `PyFrame.last_instr` before every source operation.
/// Let that walker ask the identical bytecode question with its live per-frame
/// coordinate; this is the `CallFn`-boundary equivalent of
/// `function.py funccall_valuestack` passing its live frame upstream.
pub fn exc_info_result_needs_traceback_for_call(
    frame: &crate::pyframe::PyFrame,
    call_pc: usize,
) -> bool {
    exc_info_result_needs_traceback_at(frame, call_pc)
}

fn exc_info_result_needs_traceback(frame: &crate::pyframe::PyFrame) -> bool {
    if frame.last_instr < 0 {
        return true;
    }
    exc_info_result_needs_traceback_at(frame, frame.last_instr as usize)
}

/// Return the active exception as `(type, value, traceback-or-None)`.
fn exc_info_tuple(include_traceback: bool) -> PyObjectRef {
    let exc = crate::eval::get_sys_exception();
    unsafe {
        if exc.is_null() {
            crate::runtime_ops::jit_build_tuple_3(w_none(), w_none(), w_none())
        } else {
            let exc_type = crate::baseobjspace::exception_getclass(exc);
            let exc_type = if exc_type.is_null() {
                w_none()
            } else {
                exc_type
            };
            let w_tb = if include_traceback {
                let tb = pyre_object::interp_exceptions::w_exception_get_traceback(exc);
                crate::pytraceback::mark_traceback_escaped(tb);
                if tb.is_null() { w_none() } else { tb }
            } else {
                w_none()
            };
            crate::runtime_ops::jit_build_tuple_3(exc_type, exc, w_tb)
        }
    }
}

/// pypy/module/sys/vm.py `exc_info_direct` — return the active exception
/// as a `(type, value, traceback)` tuple, omitting the traceback when the live
/// caller bytecode proves it cannot observe tuple slot 2.
///
/// Used by the JIT direct path in `function.funccall_valuestack`
/// (function.py). Splitting it out lets the JIT bypass invoke the same logic
/// without going through the regular builtin call dispatch.
pub fn exc_info_direct(frame: &mut crate::pyframe::PyFrame) -> PyObjectRef {
    let include_traceback = exc_info_result_needs_traceback(frame);
    exc_info_tuple(include_traceback)
}

/// The `BuiltinCode.func` behind `sys.exc_info`.
///
/// Named rather than a closure so it has a symbol path: source translation
/// discovers a `BuiltinCode.func` PBC member through
/// `BUILTIN_WRAPPER_DESCRIPTORS` and codewrites its graph, and
/// `jit_fnaddr` binds the same path to the runtime address.  A closure has
/// neither, so the tracer had no jitcode for the target and left the whole
/// call an opaque `CallMayForceR` -- the shape an unfolded builtin pays 700+
/// instructions per call for, on top of the tuple.
///
/// The gateway Signature binds arity 0, so `args` is empty on every call that
/// reaches here; it is threaded through to `exc_info_direct` unread, as
/// `__majit_wrap_builtin_len` threads its own.
pub fn __majit_wrap_exc_info(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    debug_assert!(args.is_empty(), "exc_info takes no arguments");
    let _ = args;
    Ok(exc_info_tuple(true))
}

/// True iff `callable` is the canonical `sys.exc_info` builtin.
///
/// `sys` is mutable, so the generated JIT must pin the function's builtin-code
/// identity before reproducing `function.py funccall_valuestack`'s direct
/// path. This is the `sys.exc_info` sibling of
/// [`is_builtin_getframe_function`].
pub fn is_builtin_exc_info_function(callable: PyObjectRef) -> bool {
    unsafe {
        if callable.is_null() || !crate::is_function(callable) {
            return false;
        }
        let code = crate::function_get_code(callable) as PyObjectRef;
        if code.is_null() || !crate::gateway::is_builtin_code(code) {
            return false;
        }
        crate::gateway::builtin_code_fn_eq(
            crate::gateway::builtin_code_get(code),
            __majit_wrap_exc_info as crate::gateway::BuiltinCodeFn,
        )
    }
}

crate::builtin_wrapper_descriptor!(__majit_wrap_exc_info_target, __majit_wrap_exc_info);

/// pypy/module/sys/vm.py `exception` — the exception instance currently being
/// handled, or None outside an `except` block: the value half of
/// `exc_info_direct`'s tuple.
pub fn sys_exception_direct() -> PyObjectRef {
    let exc = crate::eval::get_sys_exception();
    // `vm.py exception`: `sys_exc_info()` returns an internal exception
    // carrier or None, and `get_w_value()` returns its value without another
    // isinstance check. Pyre stores that value directly in `sys_exc_value`;
    // the slot's writers preserve the same exception-or-null invariant.
    if exc.is_null() { w_none() } else { exc }
}

/// The `BuiltinCode.func` behind `sys.exception`.
///
/// Named rather than a closure for the reason spelled out on
/// [`__majit_wrap_exc_info`]: a closure has no symbol path, so neither
/// `BUILTIN_WRAPPER_DESCRIPTORS` nor `jit_fnaddr` can name it and the tracer
/// has no jitcode for the call.
///
/// The gateway Signature binds arity 0, so `args` is empty on every call that
/// reaches here.
pub fn __majit_wrap_sys_exception(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    debug_assert!(args.is_empty(), "exception takes no arguments");
    let _ = args;
    Ok(sys_exception_direct())
}

crate::builtin_wrapper_descriptor!(
    __majit_wrap_sys_exception_target,
    __majit_wrap_sys_exception
);

pub fn register_module(ns: pyre_object::PyObjectRef) -> Result<(), crate::PyError> {
    let _root_scope = pyre_object::gc_roots::push_roots();
    let ns_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(ns);
    crate::__pyre_put_new!(ns_slot, "maxsize", w_int_new(i64::MAX));
    crate::__pyre_put_new!(ns_slot, "maxunicode", w_int_new(0x10FFFF));
    #[cfg(all(
        feature = "cpyext",
        not(feature = "sandbox"),
        any(target_os = "macos", target_os = "linux")
    ))]
    crate::cpyext::register_sys_dlopenflags(pyre_object::gc_roots::shadow_stack_get(ns_slot));
    // Each decoded argv string is pinned before the next decode, then the
    // list is built from those slots (`RootedItems`). `w_list_new` of a
    // `Vec` built first would leave every element unpublished. `ns` lives in
    // `ns_slot` across those allocations (`ShadowStackFrameworkGCTransformer.push_roots`).
    let mut ns = pyre_object::gc_roots::shadow_stack_get(ns_slot);
    let mut w_orig_argv = {
        let mut items = pyre_object::gc_roots::RootedItems::new();
        for arg in crate::importing::sys_orig_argv().iter() {
            items.push(crate::gateway::fsdecode_os_str(arg));
        }
        w_list_new(items.take())
    };
    crate::__pyre_store!(ns, "orig_argv", w_orig_argv);
    // pypy/interpreter/app_main.py:785-786:
    //   sys._xoptions = dict(x.split('=', 1) if '=' in x else (x, True)
    //                        for x in options['_xoptions'])
    // `fsdecode_os_str*` reaches `call_registered_decode_error_handler`, which
    // calls app-level Python, so a decode is a collection point.  The dict is
    // filled across every one of them and a dict moves, so it lives in a root
    // slot and each store reads it back from there rather than from the word
    // that was live before the decode.
    let roots = pyre_object::gc_roots::push_roots();
    // The module namespace stays live across every later allocation.
    let ns_slot = roots.base();
    let _ = roots.pin_root(ns);
    let xoptions_slot = ns_slot + 1;
    let _ = roots.pin_root(w_dict_new());
    for option in crate::importing::xoptions() {
        // `split('=', 1)` over a value that need not have a UTF-8 form. `=` is
        // ASCII, and `OsStr` documents its encoded form as splittable at an
        // ASCII byte, so the halves are whole `OsStr`s either side of it.
        let bytes = option.as_encoded_bytes();
        let (name, value) = match bytes.iter().position(|&b| b == b'=') {
            Some(eq) => {
                let (name, value) = bytes.split_at(eq);
                (name, Some(&value[1..]))
            }
            None => (bytes, None),
        };
        // The name decodes first: it answers a `Wtf8Buf`, which no collection
        // touches, whereas the decoded value is a `str` that moves.  Decoding
        // the value last leaves nothing movable live across a decode but the
        // rooted dict.
        let w_name = crate::gateway::fsdecode_os_str_wtf8(unsafe {
            std::ffi::OsStr::from_encoded_bytes_unchecked(name)
        });
        let value = match value {
            Some(value) => crate::gateway::fsdecode_os_str(unsafe {
                std::ffi::OsStr::from_encoded_bytes_unchecked(value)
            }),
            None => w_bool_from(true),
        };
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_wtf8_no_proxy(
                roots.get(xoptions_slot),
                &w_name,
                value,
            );
        }
    }
    module_ns_store(roots.get(ns_slot), "_xoptions", roots.get(xoptions_slot));
    // Format matches `platform._sys_version`'s CPython parser:
    // `version (buildinfo) [compiler]`.
    //
    // On Windows the compiler slot also names the bitness and the processor,
    // the way the `MSC v.1944 64 bit (AMD64)` token does. That token is the
    // only place the architecture is published: `sysconfig.get_platform`
    // reads it back out of `sys.version` to answer `win-amd64`/`win-arm64`,
    // and pip turns that answer into the wheel platform tag. Elsewhere the
    // compiler string carries no architecture, so neither does this one.
    //
    // The name itself is the compiler that built the C the interpreter
    // carries, which is what `compilerinfo.py:20-27` publishes: `MSC v.` and
    // `_MSC_VER` on an MSVC build, the compiler's own name elsewhere.
    // `ctypes.util._get_build_version` reads that number back out to decide
    // which C runtime `find_library("c")` may hand out, and with no token at
    // all it assumes MSVC 6 and answers `msvcrt.dll` — a runtime this build
    // links none of, and whose `errno` is therefore not the one `use_errno`
    // reports.
    let name = match option_env!("PYRE_MSC_VER") {
        Some(msc_ver) => format!("MSC v.{msc_ver}"),
        None => "Rust".to_string(),
    };
    let arch = if !cfg!(windows) {
        ""
    } else if cfg!(target_arch = "x86_64") {
        " 64 bit (AMD64)"
    } else if cfg!(target_arch = "aarch64") {
        " 64 bit (ARM64)"
    } else if cfg!(target_arch = "arm") {
        " 32 bit (ARM)"
    } else {
        " 32 bit (Intel)"
    };
    let compiler = format!("{name}{arch}");
    module_ns_store(
        roots.get(ns_slot),
        "version",
        w_str_new(&format!("3.14.6 (pyre 0.0.1) [{compiler}]")),
    );
    // The stdlib branches on this name to decide what a platform can do, and
    // the two spellings it knows for a WebAssembly guest are `emscripten` and
    // `wasi`.  A guest reaching its embedder through host imports is the
    // second of those: `emscripten` promises the JavaScript runtime behind it
    // — `platform._sys_version` reads `sys._emscripten_info` on that name and
    // has nothing to fall back to — while `wasi` promises only that there is
    // no process to fork, no home directory and no signal delivery, which is
    // what this target is.  Reported for both wasm hosts: what the guest can
    // do is the same on either side of the browser boundary.
    module_ns_store(
        roots.get(ns_slot),
        "platform",
        w_str_new(if cfg!(target_os = "macos") {
            "darwin"
        } else if cfg!(target_os = "linux") {
            "linux"
        } else if cfg!(target_os = "windows") {
            "win32"
        } else if cfg!(target_arch = "wasm32") {
            "wasi"
        } else {
            "unknown"
        }),
    );
    // sys.winver — the "major.minor" tag Windows uses for the per-user site
    // directory and the PythonCore registry keys. site.getusersitepackages
    // reads it to build USER_SITE.  A build without a global interpreter lock
    // carries the `t` here, which is how Windows spells what `sys.abiflags`
    // spells elsewhere: the tag follows the ABI the build publishes
    // (`Py_GIL_DISABLED` is 1 and `abi_thread` is `t`), not
    // `sys._is_gil_enabled()`, which reports whether the lock is on right now.
    // `venv.EnvBuilder.setup_python` reads the same config var to name
    // `python3.14t.exe`.
    #[cfg(windows)]
    module_ns_store(roots.get(ns_slot), "winver", w_str_new("3.14t"));
    // sys.dllhandle — the handle of the DLL exporting the Python C API,
    // published beside `winver` because both come from the same `MS_COREDLL`
    // block.  `ctypes/__init__.py` builds `pythonapi` out of it with no
    // import guard, so a missing attribute is an AttributeError out of `import
    // ctypes`.  `vm.py get_dllhandle` answers 0 for a build without
    // `cpyext`, and there is no cpyext here, so 0 is the whole answer: the
    // handle names an API this interpreter does not export, and reporting the
    // executable's own module would only make `pythonapi` fail one call later.
    #[cfg(windows)]
    module_ns_store(roots.get(ns_slot), "dllhandle", w_int_new(0));
    // sys._vpath — the build's relative path from the executable's directory to
    // the prefix. `sysconfig._init_config_vars` subscripts it under `os.name ==
    // 'nt'`, so it is an AttributeError out of the first `get_config_var` call
    // when absent; it is stored into `_CONFIG_VARS['VPATH']` and read nowhere
    // else, `sys._stdlib_dir` being what locates the stdlib here.
    #[cfg(windows)]
    module_ns_store(roots.get(ns_slot), "_vpath", w_str_new(r"..\.."));
    module_ns_store(
        roots.get(ns_slot),
        "byteorder",
        w_str_new(if cfg!(target_endian = "little") {
            "little"
        } else {
            "big"
        }),
    );
    // sys.version_info — structseq(major, minor, micro, releaselevel,
    // serial); a tuple subclass so `>= (3, 14)` / `[0]` and `.major` both work.
    {
        let version_info_type =
            crate::_structseq::disallow_instantiation(crate::_structseq::make_struct_seq(
                "sys.version_info",
                &["major", "minor", "micro", "releaselevel", "serial"],
            ));
        let mut fields = pyre_object::gc_roots::RootedItems::new();
        fields.push(w_int_new(3));
        fields.push(w_int_new(14));
        fields.push(w_int_new(6));
        fields.push(w_str_new("final"));
        fields.push(w_int_new(0));
        let vi = crate::_structseq::new_instance(version_info_type, fields.take());
        module_ns_store(roots.get(ns_slot), "version_info", vi);
    }
    // PyPy exposes its implementation release independently from the Python
    // language compatibility version.  pip's vendored packaging consumes the
    // named fields through sys.implementation.version; a plain tuple (or the
    // language's sys.version_info) is observably wrong.
    {
        let pyre_version_info_type = crate::_structseq::make_struct_seq(
            "sys.pyre_version_info",
            &["major", "minor", "micro", "releaselevel", "serial"],
        );
        let parse_component = |value: &str| value.parse::<i64>().unwrap_or(0);
        let mut fields = pyre_object::gc_roots::RootedItems::new();
        fields.push(w_int_new(parse_component(env!("CARGO_PKG_VERSION_MAJOR"))));
        fields.push(w_int_new(parse_component(env!("CARGO_PKG_VERSION_MINOR"))));
        fields.push(w_int_new(parse_component(env!("CARGO_PKG_VERSION_PATCH"))));
        fields.push(w_str_new("final"));
        fields.push(w_int_new(0));
        let vi = crate::_structseq::new_instance(pyre_version_info_type, fields.take());
        module_ns_store(roots.get(ns_slot), "pyre_version_info", vi);
    }
    // sys.modules — live dict synced with the import cache.
    let modules_dict = w_dict_new();
    crate::importing::set_sys_modules_dict(modules_dict);
    module_ns_store(roots.get(ns_slot), "modules", modules_dict);
    // sys.path — flush the native search-path seed into the authoritative list
    // the instant `sys` exists; from here on the Python list is the source of
    // truth and `add_sys_path` mutates it in place.
    let path_list = crate::importing::create_sys_path_list();
    module_ns_store(roots.get(ns_slot), "path", path_list);
    // sys.stdout/stderr/stdin — `_io.TextIOWrapper`-typed file-like objects.
    // Real CPython wires these through io.TextIOWrapper around the std fds;
    // pyre exposes objects of the same type with the minimum surface so
    // anything that writes status (unittest, traceback, warnings) keeps
    // working.  `sys.__stdout__ is sys.stdout` (a single object each).
    //
    // rustc `sanitize_standard_fds` reopens `/dev/null` onto EBADF 0/1/2
    // before `main`; `rposix::restore_closed_standard_fds` undoes the
    // occupy that kept the sanitizer from claiming a `posix_spawn`
    // `POSIX_SPAWN_CLOSE` slot, so `create_stdio` still sees EBADF.
    majit_rlib::rposix::restore_closed_standard_fds();
    let stdout = make_std_stream("<stdout>", 1);
    let stdout_slot = roots.pin_roots(&[stdout]);
    let stderr = make_std_stream("<stderr>", 2);
    let stderr_slot = roots.pin_roots(&[stderr]);
    let stdin = make_std_stream("<stdin>", 0);
    module_ns_store(roots.get(ns_slot), "stdout", roots.get(stdout_slot));
    module_ns_store(roots.get(ns_slot), "stderr", roots.get(stderr_slot));
    module_ns_store(roots.get(ns_slot), "stdin", stdin);
    module_ns_store(roots.get(ns_slot), "__stdout__", roots.get(stdout_slot));
    module_ns_store(roots.get(ns_slot), "__stderr__", roots.get(stderr_slot));
    module_ns_store(roots.get(ns_slot), "__stdin__", stdin);
    // `vm.py _getframe` (arguments) over `vm.py getframe` (the walk) —
    // see [`sys_getframe`] and [`getframe`], which keep upstream's split.
    module_ns_store(
        roots.get(ns_slot),
        "_getframe",
        crate::make_builtin_function("_getframe", sys_getframe),
    );
    // `sys._getframemodulename(depth=0)` — the module name of the frame
    // `depth` levels up, or None when the walk runs off the stack.  Used by
    // `warnings` and `functools` to attribute a call to its module without
    // materialising the frame.
    //
    // `sys__getframemodulename_impl` reads `PyFunction_GetModule(f->f_funcobj)`.
    // A pyre frame carries `pycode` and `w_globals` but no link back to the
    // function object (the JIT virtualizable layout is the hot path, and
    // `debugdata` is allocated lazily), so the module name comes from
    // `globals["__name__"]` — the value `__module__` is initialised from.  A
    // function whose `__module__` was reassigned after definition therefore
    // still reports its defining module.
    module_ns_store(
        roots.get(ns_slot),
        "_getframemodulename",
        crate::make_builtin_function("_getframemodulename", |args| {
            let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
            crate::builtins::kwarg_reject_unknown(kwargs, &["depth"], "_getframemodulename")?;
            // The arity is judged on the total argument count, so supplying
            // `depth` both ways is "takes at most 1 argument (2 given)" rather
            // than the duplicate-binding report.
            let supplied = positional.len() + crate::builtins::real_kwarg_count(kwargs);
            if supplied > 1 {
                return Err(crate::PyError::type_error(format!(
                    "_getframemodulename() takes at most 1 argument ({supplied} given)"
                )));
            }
            let depth = match crate::builtins::bind_pos_or_kw(
                positional,
                kwargs,
                0,
                "depth",
                "_getframemodulename",
                1,
            )? {
                Some(v) => crate::baseobjspace::int_w(crate::baseobjspace::space_index(v)?)?,
                None => 0,
            };
            let ec = current_execution_context();
            if ec.is_null() {
                return Ok(pyre_object::w_none());
            }
            let mut current = unsafe { (*ec).gettopframe_nohidden() };
            // `while (f && (_PyFrame_IsIncomplete(f) || depth-- > 0))` — the
            // post-decrement test fails immediately for a negative depth, so a
            // negative walks zero frames and reports the current module rather
            // than `None`.
            let mut remaining = depth.max(0);
            while !current.is_null() && remaining > 0 {
                current = crate::executioncontext::ExecutionContext::getnextframe_nohidden(current);
                remaining -= 1;
            }
            if current.is_null() {
                return Ok(pyre_object::w_none());
            }
            // `get_w_globals` reads `pycode`, which `interp_jit.py`
            // declares virtualizable, so the frame it is read off has to be
            // materialized first.  The force belongs HERE, at the consumer, and
            // not at the walk that reached the frame — see [`force_frame`]:
            // forcing a walk escapes the traced virtualizable and
            // `vable_after_residual_call` aborts the trace with ABORT_ESCAPE.
            let anchor = unsafe { crate::eval::FrameAnchor::from_raw(current) };
            crate::executioncontext::jit_force_virtualizable(anchor.live());
            let w_globals = unsafe { (*anchor.live()).get_w_globals() };
            if w_globals.is_null() {
                return Ok(pyre_object::w_none());
            }
            match crate::baseobjspace::finditem_str(w_globals, "__name__")? {
                Some(name) if !name.is_null() => Ok(name),
                _ => Ok(pyre_object::w_none()),
            }
        }),
    );
    // sys.exc_info() → (type, value, traceback)
    //
    // Tuple construction is shared with `exc_info_direct` (the JIT fast-path
    // entry registered below), so the regular call path and the JIT bypass
    // observe the same value.  The `BuiltinCode.func` is the named
    // `__majit_wrap_exc_info` rather than a closure so the tracer can descend
    // into it.
    let exc_info_fn = make_builtin_function_with_arity("exc_info", __majit_wrap_exc_info, 0);
    module_ns_store(roots.get(ns_slot), "exc_info", exc_info_fn);
    // baseobjspace.py: register `space._code_of_sys_exc_info` so
    // `function.funccall_valuestack` can take the JIT direct path
    // (function.py:146-150). The builtin code pointer lives on the
    // `BuiltinCode` object backing `exc_info_fn`; `getcode` returns it.
    let exc_info_code = unsafe { crate::getcode(exc_info_fn) };
    crate::function::register_sys_exc_info_path(exc_info_code, exc_info_direct);
    // sys.flags — pypy/module/sys/app.py `class sysflags` with
    // `__metaclass__ = structseqtype`: an immutable tuple subclass whose
    // first `n_sequence_fields` entries are also indexable, so
    // `sys.flags[3] is sys.flags.optimize` and `isinstance(sys.flags,
    // tuple)` both hold.  `gil`, `thread_inherit_context` and
    // `context_aware_warnings` sit past the sequence as named-only extras,
    // which is what makes `n_fields` (21) exceed `len()` (18).
    {
        let flags_type = crate::_structseq::disallow_instantiation(
            crate::_structseq::make_struct_seq_with_extra(
                "sys.flags",
                &[
                    "debug",
                    "inspect",
                    "interactive",
                    "optimize",
                    "dont_write_bytecode",
                    "no_user_site",
                    "no_site",
                    "ignore_environment",
                    "verbose",
                    "bytes_warning",
                    "quiet",
                    "hash_randomization",
                    "isolated",
                    "dev_mode",
                    "utf8_mode",
                    "warn_default_encoding",
                    "safe_path",
                    "int_max_str_digits",
                ],
                &["gil", "thread_inherit_context", "context_aware_warnings"],
            ),
        );
        let _flag_roots = pyre_object::gc_roots::push_roots();
        let gil_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_int_new(1));
        let inherit_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_int_new(0));
        let warn_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_int_new(0));
        let mut fields = pyre_object::gc_roots::RootedItems::new();
        // `initconfig.c PYCONFIG_SPEC` declares `parser_debug` BOOL
        // while `verbose` is UINT, so `-dd` reports 1 here and 2
        // there.  Both count in the config; only this one saturates.
        fields.push(w_int_new(i64::from(crate::importing::debug_flag() != 0)));
        // `-i` sets both.
        fields.push(w_int_new(i64::from(crate::importing::inspect_flag())));
        fields.push(w_int_new(i64::from(crate::importing::interactive_flag())));
        fields.push(w_int_new(crate::importing::optimize_level()));
        fields.push(w_int_new(i64::from(
            crate::importing::dont_write_bytecode_flag(),
        )));
        fields.push(w_int_new(i64::from(crate::importing::no_user_site_flag())));
        // `-S` (skip `import site`) is recorded by the launcher.
        fields.push(w_int_new(i64::from(crate::importing::no_site_flag())));
        fields.push(w_int_new(i64::from(
            crate::importing::ignore_environment_flag(),
        )));
        fields.push(w_int_new(crate::importing::verbose_flag()));
        fields.push(w_int_new(crate::importing::bytes_warning_flag()));
        fields.push(w_int_new(i64::from(crate::importing::quiet_flag())));
        fields.push(w_int_new(0)); // hash_randomization
        fields.push(w_int_new(i64::from(crate::importing::isolated_flag())));
        fields.push(w_bool_from(crate::importing::dev_mode_flag()));
        fields.push(w_int_new(crate::importing::utf8_mode_flag()));
        // sysmodule.c builds this one with `SetFlag` (PyLong_FromLong);
        // only `dev_mode` and `safe_path` are PyBool_FromLong.
        fields.push(w_int_new(i64::from(
            crate::importing::warn_default_encoding_flag(),
        )));
        fields.push(w_bool_from(crate::importing::safe_path_flag()));
        fields.push(w_int_new(
            crate::module::sys::state::int_max_str_digits() as i64
        ));
        let flags = crate::_structseq::new_instance_with_extra(
            flags_type,
            fields.take(),
            vec![
                // The interpreter holds a GIL, so `-X gil=0` is not available.
                ("gil", pyre_object::gc_roots::shadow_stack_get(gil_slot)),
                (
                    "thread_inherit_context",
                    pyre_object::gc_roots::shadow_stack_get(inherit_slot),
                ),
                (
                    "context_aware_warnings",
                    pyre_object::gc_roots::shadow_stack_get(warn_slot),
                ),
            ],
        );
        module_ns_store(roots.get(ns_slot), "flags", flags);
    }
    // sys.getdefaultencoding
    module_ns_store(
        roots.get(ns_slot),
        "getdefaultencoding",
        make_builtin_function_with_arity("getdefaultencoding", |_| Ok(w_str_new("utf-8")), 0),
    );
    // sys.getwindowsversion — a five-field sequence over `OSVERSIONINFOEXW`
    // with five named-only fields beyond it, the same shape `os.stat_result`
    // carries its `st_*_ns` extras in. `test.support.os_helper` reads
    // `.platform` at import, so every `test.support` consumer needs it.
    //
    // `major`/`minor`/`build` are `GetVersionEx`'s answer, which is the running
    // version because the executable carries `python.manifest`; without one the
    // call reports the version the process is shimmed to.  `platform_version`
    // is kernel32's own file version
    // (`_sys_getwindowsversion_from_kernel32`), which is a different number on
    // a release shipped as an enablement package over the previous build.
    #[cfg(windows)]
    module_ns_store(
        roots.get(ns_slot),
        "getwindowsversion",
        make_builtin_function_with_arity(
            "getwindowsversion",
            |args| {
                if !args.is_empty() {
                    return Err(crate::PyError::type_error(
                        "getwindowsversion() takes no arguments",
                    ));
                }
                let info = rustpython_host_env::windows::get_windows_version().map_err(|e| {
                    crate::PyError::os_error_win32_syscall2(
                        e.raw_os_error().unwrap_or(0),
                        pyre_object::PY_NULL,
                        pyre_object::PY_NULL,
                    )
                })?;
                // Built once, like [`stat_result_seq_type`]: the type is the
                // answer's identity, so making a fresh one per call would leave
                // `type(sys.getwindowsversion())` a different class each time.
                static SEQ_TYPE: pyre_object::gc_roots::RootedOnceRef =
                    pyre_object::gc_roots::RootedOnceRef::new();
                let cls = SEQ_TYPE.get_or_init(|| {
                    crate::_structseq::disallow_instantiation(
                        crate::_structseq::make_struct_seq_with_extra(
                            "sys.getwindowsversion",
                            &["major", "minor", "build", "platform", "service_pack"],
                            &[
                                "service_pack_major",
                                "service_pack_minor",
                                "suite_mask",
                                "product_type",
                                "platform_version",
                            ],
                        ),
                    )
                });
                let (major, minor, build) = rustpython_host_env::winapi::version_ex_triple()
                    .unwrap_or((info.major, info.minor, info.build));
                let _roots = pyre_object::gc_roots::push_roots();
                let mut extra_slots: Vec<(&str, usize)> = Vec::new();
                let mut put_extra = |name: &'static str, value: PyObjectRef| {
                    extra_slots.push((name, pyre_object::gc_roots::shadow_stack_len()));
                    let _ = pyre_object::gc_roots::pin_root(value);
                };
                put_extra(
                    "service_pack_major",
                    w_int_new(info.service_pack_major as i64),
                );
                put_extra(
                    "service_pack_minor",
                    w_int_new(info.service_pack_minor as i64),
                );
                put_extra("suite_mask", w_int_new(info.suite_mask as i64));
                put_extra("product_type", w_int_new(info.product_type as i64));
                let platform_version = {
                    let mut fields = pyre_object::gc_roots::RootedItems::new();
                    fields.push(w_int_new(info.major as i64));
                    fields.push(w_int_new(info.minor as i64));
                    fields.push(w_int_new(info.build as i64));
                    pyre_object::w_tuple_new(fields.take())
                };
                put_extra("platform_version", platform_version);
                drop(put_extra);
                let mut fields = pyre_object::gc_roots::RootedItems::new();
                fields.push(w_int_new(major as i64));
                fields.push(w_int_new(minor as i64));
                fields.push(w_int_new(build as i64));
                fields.push(w_int_new(info.platform as i64));
                fields.push(w_str_new_managed(&info.service_pack));
                let extras: Vec<_> = extra_slots
                    .into_iter()
                    .map(|(name, slot)| (name, pyre_object::gc_roots::shadow_stack_get(slot)))
                    .collect();
                Ok(crate::_structseq::new_instance_with_extra(
                    cls,
                    fields.take(),
                    extras,
                ))
            },
            0,
        ),
    );
    // sys._get_cpu_count_config() — the cpu count the interpreter was
    // configured with, or -1 where none was. `os.py:1180` asks before it
    // decides whether `process_cpu_count` counts the affinity mask or aliases
    // `cpu_count`, so a `posix` that publishes `sched_getaffinity` and a `sys`
    // that does not answer this makes `import os` raise.
    //
    // -1 is the answer here rather than a placeholder: the value is set by
    // `-X cpu_count` and `PYTHON_CPU_COUNT`, and this interpreter reads
    // neither, so there is no configured count to report.
    module_ns_store(
        roots.get(ns_slot),
        "_get_cpu_count_config",
        make_builtin_function_with_arity("_get_cpu_count_config", |_| Ok(w_int_new(-1)), 0),
    );
    // sys.getrecursionlimit / setrecursionlimit — pypy/module/sys/vm.py.
    // The runtime stack budget lives in `crate::stack_check`; both
    // helpers route through it so the interpreter, JIT prologue probe,
    // and blackhole resume see a consistent recursion budget.
    module_ns_store(
        roots.get(ns_slot),
        "getrecursionlimit",
        make_builtin_function_with_arity(
            "getrecursionlimit",
            |args| {
                // pypy/module/sys/vm.py:72 — no arguments.
                if !args.is_empty() {
                    return Err(crate::PyError::type_error(
                        "getrecursionlimit() takes no arguments",
                    ));
                }
                Ok(w_int_new(crate::stack_check::get_recursion_limit() as i64))
            },
            0,
        ),
    );
    module_ns_store(
        roots.get(ns_slot),
        "setrecursionlimit",
        make_builtin_function_with_arity(
            "setrecursionlimit",
            |args| {
                // pypy/module/sys/vm.py `@unwrap_spec(new_limit="c_int")`
                // — exactly one positional argument, coerced through
                // baseobjspace.c_int_w (gateway_int_w + 32-bit range
                // check). `c_int_w` accepts int subclasses and any object
                // implementing `__int__`, rejects floats, and surfaces
                // out-of-range values as OverflowError.
                if args.len() != 1 {
                    return Err(crate::PyError::type_error(
                        "setrecursionlimit() takes exactly one argument",
                    ));
                }
                // `sys_setrecursionlimit_impl`'s clinic `int new_limit`
                // converter reads `__index__`, so a non-index argument is
                // named in the error rather than reported as a bare
                // "expected integer".
                let new_limit = crate::builtins::space_index_w(args[0])?;
                if !(i32::MIN as i64..=i32::MAX as i64).contains(&new_limit) {
                    return Err(crate::PyError::overflow_error("expected a 32-bit integer"));
                }
                let new_limit = new_limit as i32;
                crate::stack_check::set_recursion_limit(new_limit)?;
                Ok(w_none())
            },
            1,
        ),
    );
    module_ns_store(
        roots.get(ns_slot),
        "getswitchinterval",
        make_builtin_function_with_arity(
            "getswitchinterval",
            |args| {
                if !args.is_empty() {
                    return Err(crate::PyError::type_error(
                        "getswitchinterval() takes no arguments",
                    ));
                }
                let ec = current_execution_context();
                let interval = if ec.is_null() {
                    0.005
                } else {
                    unsafe { (*ec).actionflag.getcheckinterval() as f64 / 2_000_000.0 }
                };
                Ok(w_float_new(interval))
            },
            0,
        ),
    );
    module_ns_store(
        roots.get(ns_slot),
        "setswitchinterval",
        make_builtin_function_with_arity(
            "setswitchinterval",
            |args| {
                if args.len() != 1 {
                    return Err(crate::PyError::type_error(
                        "setswitchinterval() takes exactly one argument",
                    ));
                }
                let interval = crate::baseobjspace::float_w(args[0])?;
                if interval <= 0.0 {
                    return Err(crate::PyError::value_error(
                        "switch interval must be strictly positive",
                    ));
                }
                let ec = current_execution_context();
                if !ec.is_null() {
                    unsafe {
                        (*(ec as *mut crate::PyExecutionContext))
                            .actionflag
                            .setcheckinterval((interval * 2_000_000.0) as usize)
                    };
                }
                Ok(w_none())
            },
            1,
        ),
    );
    module_ns_store(
        roots.get(ns_slot),
        "_current_frames",
        make_builtin_function_with_arity(
            "_current_frames",
            |args| {
                if !args.is_empty() {
                    return Err(crate::PyError::type_error(
                        "_current_frames() takes no arguments",
                    ));
                }
                Ok(crate::module::thread::current_frames())
            },
            0,
        ),
    );
    module_ns_store(
        roots.get(ns_slot),
        "_current_exceptions",
        make_builtin_function_with_arity(
            "_current_exceptions",
            |_| Ok(crate::module::thread::current_exceptions()),
            0,
        ),
    );
    // PyPy: pypy/module/sys/state.py:get_int_max_str_digits and
    // set_int_max_str_digits. The limit is object-space state, shared by
    // every caller rather than thread-local state.
    module_ns_store(
        roots.get(ns_slot),
        "get_int_max_str_digits",
        make_builtin_function_with_arity(
            "get_int_max_str_digits",
            |args| {
                // The fixed arity above is only a fast-dispatch hint; the
                // direct path still delivers whatever the caller passed.
                if !args.is_empty() {
                    return Err(crate::PyError::type_error(format!(
                        "get_int_max_str_digits() takes 0 positional arguments but {} {} given",
                        args.len(),
                        if args.len() == 1 { "was" } else { "were" },
                    )));
                }
                Ok(w_int_new(
                    crate::module::sys::state::int_max_str_digits() as i64
                ))
            },
            0,
        ),
    );
    module_ns_store(
        roots.get(ns_slot),
        "set_int_max_str_digits",
        make_builtin_function_with_arity_and_maybe_sig(
            "set_int_max_str_digits",
            |args| {
                if args.len() != 1 {
                    let message = if args.is_empty() {
                        "set_int_max_str_digits() missing 1 required positional argument: \
                         'maxdigits'"
                            .to_string()
                    } else {
                        format!(
                            "set_int_max_str_digits() takes 1 positional argument but {} were given",
                            args.len(),
                        )
                    };
                    return Err(crate::PyError::type_error(message));
                }
                let maxdigits = crate::baseobjspace::c_int_w(args[0])?;
                crate::module::sys::state::set_int_max_str_digits(maxdigits)?;
                Ok(w_none())
            },
            1,
            Some(crate::gateway::Signature::new(
                vec!["maxdigits"],
                None,
                None,
                0,
                0,
            )),
        ),
    );
    // sys.intern
    module_ns_store(
        roots.get(ns_slot),
        "intern",
        make_builtin_function_with_arity(
            "intern",
            |args| {
                let s = args[0];
                if !unsafe { pyre_object::is_exact_type(s, &pyre_object::STR_TYPE) } {
                    // The clinic converter takes a `unicode`, so anything that
                    // is not a `str` at all is refused by the argument and
                    // never reaches `sys_intern_impl`; only a `str` subclass
                    // does, and that is the refusal the body itself states.
                    let name = crate::type_methods::clinic_arg_type_name(s);
                    let message = if unsafe { is_str(s) } {
                        format!("can't intern {name}")
                    } else {
                        format!("intern() argument must be str, not {name}")
                    };
                    return Err(crate::PyError::type_error(message));
                }
                Ok(unsafe { pyre_object::unicodeobject::intern_exact_str(s) })
            },
            1,
        ),
    );
    module_ns_store(roots.get(ns_slot), "api_version", w_int_new(1013));
    module_ns_store(
        roots.get(ns_slot),
        "_git",
        w_tuple_new(vec![w_str_new("pyre"), w_str_new(""), w_str_new("")]),
    );
    module_ns_store(roots.get(ns_slot), "_home", w_none());
    // sys.implementation — PyPy app.py's implementation_dict, with the Pyre
    // implementation version rather than Python's language version.
    {
        let impl_obj = w_instance_new(simple_namespace_type());
        let impl_slot = roots.pin_roots(&[impl_obj]);
        let name_obj = w_str_new("pyre");
        crate::baseobjspace::setdictvalue_native(roots.get(impl_slot), "name", name_obj);
        let implementation_version =
            crate::runtime_ops::module_ns_get(roots.get(ns_slot), "pyre_version_info")
                .expect("sys.pyre_version_info was installed above");
        crate::baseobjspace::setdictvalue_native(
            roots.get(impl_slot),
            "version",
            implementation_version,
        );
        let version_major = env!("CARGO_PKG_VERSION_MAJOR").parse::<i64>().unwrap_or(0);
        let version_minor = env!("CARGO_PKG_VERSION_MINOR").parse::<i64>().unwrap_or(0);
        let version_micro = env!("CARGO_PKG_VERSION_PATCH").parse::<i64>().unwrap_or(0);
        let implementation_hexversion =
            (version_major << 24) | (version_minor << 16) | (version_micro << 8) | 0xf0;
        let hexversion = w_int_new(implementation_hexversion);
        crate::baseobjspace::setdictvalue_native(roots.get(impl_slot), "hexversion", hexversion);
        let cache_tag = w_str_new("pyre314");
        crate::baseobjspace::setdictvalue_native(roots.get(impl_slot), "cache_tag", cache_tag);
        // PEP 734. pyre runs one interpreter per process, so the answer is
        // false; `sys.implementation` carries the name either way from 3.14 on.
        let supports = w_bool_from(false);
        crate::baseobjspace::setdictvalue_native(
            roots.get(impl_slot),
            "supports_isolated_interpreters",
            supports,
        );
        let multiarch = w_str_new(crate::importing::multiarch());
        crate::baseobjspace::setdictvalue_native(roots.get(impl_slot), "_multiarch", multiarch);
        module_ns_store(roots.get(ns_slot), "implementation", roots.get(impl_slot));
    }
    // `system.py:get_hash_info|get_float_info|get_thread_info|get_int_info`:
    // these values are instances of their named structseq classes, not
    // namespace-shaped records. Tuple storage is observable through indexing,
    // unpacking, repr, pickling and the n_* class attributes.
    {
        let ty = crate::_structseq::make_struct_seq(
            "sys.hash_info",
            &[
                "width",
                "modulus",
                "inf",
                "nan",
                "imag",
                "algorithm",
                "hash_bits",
                "seed_bits",
                "cutoff",
            ],
        );
        let mut fields = pyre_object::gc_roots::RootedItems::new();
        fields.push(w_int_new(64));
        fields.push(w_int_new((1i64 << 61) - 1));
        fields.push(w_int_new(314159));
        fields.push(w_int_new(0));
        fields.push(w_int_new(1000003));
        fields.push(w_str_new("siphash13"));
        fields.push(w_int_new(64));
        fields.push(w_int_new(128));
        fields.push(w_int_new(0));
        let value = crate::_structseq::new_instance(ty, fields.take());
        module_ns_store(roots.get(ns_slot), "hash_info", value);
    }
    {
        let ty = crate::_structseq::make_struct_seq(
            "sys.float_info",
            &[
                "max",
                "max_exp",
                "max_10_exp",
                "min",
                "min_exp",
                "min_10_exp",
                "dig",
                "mant_dig",
                "epsilon",
                "radix",
                "rounds",
            ],
        );
        let mut fields = pyre_object::gc_roots::RootedItems::new();
        fields.push(w_float_new(f64::MAX));
        fields.push(w_int_new(1024));
        fields.push(w_int_new(308));
        fields.push(w_float_new(f64::MIN_POSITIVE));
        fields.push(w_int_new(-1021));
        fields.push(w_int_new(-307));
        fields.push(w_int_new(15));
        fields.push(w_int_new(53));
        fields.push(w_float_new(f64::EPSILON));
        fields.push(w_int_new(2));
        fields.push(w_int_new(1));
        let value = crate::_structseq::new_instance(ty, fields.take());
        module_ns_store(roots.get(ns_slot), "float_info", value);
    }
    // sysmodule.c — `sys.float_repr_style` is "short" wherever float repr
    // uses David Gay's shortest-round-trip algorithm (always, here).
    module_ns_store(roots.get(ns_slot), "float_repr_style", w_str_new("short"));
    {
        let ty =
            crate::_structseq::make_struct_seq("sys.thread_info", &["name", "lock", "version"]);
        let mut fields = pyre_object::gc_roots::RootedItems::new();
        fields.push(w_str_new(if cfg!(windows) { "nt" } else { "pthread" }));
        fields.push(if cfg!(windows) {
            w_none()
        } else {
            w_str_new("semaphore")
        });
        fields.push(w_none());
        let value = crate::_structseq::new_instance(ty, fields.take());
        module_ns_store(roots.get(ns_slot), "thread_info", value);
    }
    {
        let ty = crate::_structseq::make_struct_seq(
            "sys.int_info",
            &[
                "bits_per_digit",
                "sizeof_digit",
                "default_max_str_digits",
                "str_digits_check_threshold",
            ],
        );
        let mut fields = pyre_object::gc_roots::RootedItems::new();
        fields.push(w_int_new(30));
        fields.push(w_int_new(4));
        fields.push(w_int_new(4300));
        fields.push(w_int_new(640));
        let value = crate::_structseq::new_instance(ty, fields.take());
        module_ns_store(roots.get(ns_slot), "int_info", value);
    }
    module_ns_store(roots.get(ns_slot), "hexversion", w_int_new(0x030e06f0));
    #[cfg(feature = "host_env")]
    {
        // `app_main.setup_bootstrap_path` obtains all five values from one
        // initpath computation before site runs.  In particular,
        // `sys.executable` retains a venv symlink/copy while stdlib lookup
        // follows it to `base_prefix`; independent `current_exe` and stdlib
        // probes collapse that distinction and make pyvenv.cfg ineffective.
        let paths = crate::importing::startup_path_config();
        let fs_path = |path: &std::path::Path| {
            w_str_from_wtf8(crate::gateway::fsdecode_os_str_wtf8(path.as_os_str()))
        };
        let executable = fs_path(&paths.executable);
        module_ns_store(roots.get(ns_slot), "executable", executable);
        let base_executable = fs_path(&paths.base_executable);
        module_ns_store(roots.get(ns_slot), "_base_executable", base_executable);
        let prefix = fs_path(&paths.prefix);
        module_ns_store(roots.get(ns_slot), "prefix", prefix);
        let exec_prefix = fs_path(&paths.prefix);
        module_ns_store(roots.get(ns_slot), "exec_prefix", exec_prefix);
        let base_prefix = fs_path(&paths.base_prefix);
        module_ns_store(roots.get(ns_slot), "base_prefix", base_prefix);
        let base_exec_prefix = fs_path(&paths.base_prefix);
        module_ns_store(roots.get(ns_slot), "base_exec_prefix", base_exec_prefix);
        // FrozenImporter uses the resolved stdlib root to reconstruct source
        // filenames for frozen stdlib modules.
        let stdlib_dir = paths.stdlib.as_deref().map(fs_path).unwrap_or_else(w_none);
        module_ns_store(roots.get(ns_slot), "_stdlib_dir", stdlib_dir);
    }
    #[cfg(not(feature = "host_env"))]
    {
        module_ns_store(roots.get(ns_slot), "executable", w_str_new(""));
        module_ns_store(roots.get(ns_slot), "_base_executable", w_str_new(""));
        module_ns_store(roots.get(ns_slot), "prefix", w_str_new(""));
        module_ns_store(roots.get(ns_slot), "exec_prefix", w_str_new(""));
        module_ns_store(roots.get(ns_slot), "base_prefix", w_str_new(""));
        module_ns_store(roots.get(ns_slot), "base_exec_prefix", w_str_new(""));
        module_ns_store(roots.get(ns_slot), "_stdlib_dir", w_none());
    }
    // sys._framework — macOS framework name (empty string on non-framework builds)
    module_ns_store(roots.get(ns_slot), "_framework", w_str_new(""));
    // sys._jit — namespace with is_enabled/is_available methods.
    // Python 3.14+ introduced sys._jit for CPython tier-2 JIT support checks.
    {
        let jit = make_sys_namespace_instance();
        let jit_slot = roots.pin_roots(&[jit]);
        let is_enabled =
            make_builtin_function_with_arity("is_enabled", |_| Ok(w_bool_from(false)), 0);
        crate::baseobjspace::setdictvalue_native(roots.get(jit_slot), "is_enabled", is_enabled);
        let is_available =
            make_builtin_function_with_arity("is_available", |_| Ok(w_bool_from(false)), 0);
        crate::baseobjspace::setdictvalue_native(roots.get(jit_slot), "is_available", is_available);
        module_ns_store(roots.get(ns_slot), "_jit", roots.get(jit_slot));
    }
    // sys.monitoring — PEP 669 low-impact monitoring API. The runtime hooks
    // are stubbed (no events ever fire), but the namespace, tool-id
    // constants, sentinels, and `events` flags are present so importing
    // modules (bdb/pdb/cProfile/coverage tooling) succeed and can build
    // their tracer objects.
    {
        let mon = make_sys_namespace_instance();
        let mon_slot = roots.pin_roots(&[mon]);
        // Tool-id constants (Python/instrumentation.c).
        for (name, id) in [
            ("DEBUGGER_ID", 0),
            ("COVERAGE_ID", 1),
            ("PROFILER_ID", 2),
            ("OPTIMIZER_ID", 5),
        ] {
            let value = w_int_new(id);
            crate::baseobjspace::setdictvalue_native(roots.get(mon_slot), name, value);
        }
        // DISABLE / MISSING sentinels — distinct singleton objects compared
        // by identity (`callback() == DISABLE`, `assertIs(x, MISSING)`).
        let disable = make_sys_namespace_instance();
        crate::baseobjspace::setdictvalue_native(roots.get(mon_slot), "DISABLE", disable);
        let missing = make_sys_namespace_instance();
        crate::baseobjspace::setdictvalue_native(roots.get(mon_slot), "MISSING", missing);
        // events namespace — `1 << event_id` flags that OR together.
        {
            let events = make_sys_namespace_instance();
            let events_slot = roots.pin_roots(&[events]);
            let no_events = w_int_new(0);
            crate::baseobjspace::setdictvalue_native(
                roots.get(events_slot),
                "NO_EVENTS",
                no_events,
            );
            for (i, name) in [
                "PY_START",
                "PY_RESUME",
                "PY_RETURN",
                "PY_YIELD",
                "CALL",
                "LINE",
                "INSTRUCTION",
                "JUMP",
                "BRANCH_LEFT",
                "BRANCH_RIGHT",
                "STOP_ITERATION",
                "RAISE",
                "EXCEPTION_HANDLED",
                "PY_UNWIND",
                "PY_THROW",
                "RERAISE",
                "C_RETURN",
                "C_RAISE",
            ]
            .iter()
            .enumerate()
            {
                let flag = w_int_new(1i64 << i);
                crate::baseobjspace::setdictvalue_native(roots.get(events_slot), name, flag);
            }
            // BRANCH retained as an alias of BRANCH_LEFT for callers predating
            // the 3.14 left/right split.
            let branch = w_int_new(1i64 << 8);
            crate::baseobjspace::setdictvalue_native(roots.get(events_slot), "BRANCH", branch);
            crate::baseobjspace::setdictvalue_native(
                roots.get(mon_slot),
                "events",
                roots.get(events_slot),
            );
        }
        // Runtime hooks — no-op stubs returning sensible defaults.
        let store_fn =
            |mut obj, name: &'static str, f: crate::gateway::BuiltinCodeFn, arity: u16| {
                let func = pyre_object::with_roots!(obj => {
                    make_builtin_function_with_arity(name, f, arity)
                });
                crate::baseobjspace::setdictvalue_native(obj, name, func);
            };
        store_fn(
            roots.get(mon_slot),
            "use_tool_id",
            |args| {
                if args.len() != 2 {
                    return Err(crate::PyError::type_error(
                        "use_tool_id() takes exactly two arguments",
                    ));
                }
                // `args` is the gateway's native copy; `__index__` can collect.
                let mut w_name = args[1];
                let tool_id = pyre_object::with_roots!(w_name => monitoring_tool_id_w(args[0]))?;
                monitoring_use_tool_id(tool_id, w_name)?;
                Ok(w_none())
            },
            2,
        );
        store_fn(
            roots.get(mon_slot),
            "free_tool_id",
            |args| {
                if args.len() != 1 {
                    return Err(crate::PyError::type_error(
                        "free_tool_id() takes exactly one argument",
                    ));
                }
                monitoring_set_tool_name(monitoring_tool_id_w(args[0])?, pyre_object::PY_NULL);
                Ok(w_none())
            },
            1,
        );
        // `clear_tool_id` drops the tool's callbacks and events but keeps its
        // claim on the id; with the event machinery inert there is nothing to
        // clear, so only the id check is observable.
        store_fn(
            roots.get(mon_slot),
            "clear_tool_id",
            |args| {
                if args.len() != 1 {
                    return Err(crate::PyError::type_error(
                        "clear_tool_id() takes exactly one argument",
                    ));
                }
                monitoring_tool_id_w(args[0])?;
                Ok(w_none())
            },
            1,
        );
        store_fn(
            roots.get(mon_slot),
            "get_tool",
            |args| {
                if args.len() != 1 {
                    return Err(crate::PyError::type_error(
                        "get_tool() takes exactly one argument",
                    ));
                }
                let name = monitoring_tool_name(monitoring_tool_id_w(args[0])?);
                Ok(if name.is_null() { w_none() } else { name })
            },
            1,
        );
        store_fn(
            roots.get(mon_slot),
            "register_callback",
            |_| Ok(w_none()),
            3,
        );
        store_fn(roots.get(mon_slot), "set_events", |_| Ok(w_none()), 2);
        store_fn(roots.get(mon_slot), "get_events", |_| Ok(w_int_new(0)), 1);
        store_fn(roots.get(mon_slot), "set_local_events", |_| Ok(w_none()), 3);
        store_fn(
            roots.get(mon_slot),
            "get_local_events",
            |_| Ok(w_int_new(0)),
            2,
        );
        store_fn(roots.get(mon_slot), "restart_events", |_| Ok(w_none()), 0);
        module_ns_store(roots.get(ns_slot), "monitoring", roots.get(mon_slot));
    }
    // sys.platlibdir — the platform-specific library directory sysconfig and
    // `site.addsitepackages` build install paths from: "lib" on POSIX, and
    // "DLLs" on Windows, where the extension modules sit beside the interpreter
    // instead of under a versioned lib directory.
    module_ns_store(
        roots.get(ns_slot),
        "platlibdir",
        w_str_new(if cfg!(windows) { "DLLs" } else { "lib" }),
    );
    // `sys/app.py exit(exitcode=None)` — raise SystemExit(exitcode),
    // de-tupelizing a tuple argument so `exit((a, b))` becomes
    // `SystemExit(a, b)` (the extra de-tupelizing normalize_exception does
    // for `raise SystemExit, exitcode`).  A bare `exit()` defaults exitcode
    // to None, so the instance carries `code = None` / `args = (None,)`.
    // Interpreting the code (None → 0, int() coercion,
    // print-non-integral-and-exit-1) is the launcher's job
    // (`app_main.py handle_sys_exit`).
    module_ns_store(
        roots.get(ns_slot),
        "exit",
        crate::make_module_builtin_function("exit", |args| {
            // `exit(exitcode=None)` — resolve the single optional argument
            // like the app-level signature: strip the `__pyre_kw__` trailer,
            // reject unknown keywords, reproduce the normal function-call
            // arity diagnostics, and reject a positional/`exitcode=`
            // duplicate.
            let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
            crate::builtins::kwarg_reject_unknown(kwargs, &["exitcode"], "exit")?;
            if positional.len() > 1 {
                return Err(crate::PyError::type_error(format!(
                    "exit() takes from 0 to 1 positional arguments but {} were given",
                    positional.len()
                )));
            }
            let kw_exitcode = crate::builtins::kwarg_get(kwargs, "exitcode");
            if !positional.is_empty() && kw_exitcode.is_some() {
                return Err(crate::PyError::type_error(
                    "exit() got multiple values for argument 'exitcode'",
                ));
            }
            let exitcode = positional
                .first()
                .copied()
                .or(kw_exitcode)
                .unwrap_or_else(w_none);
            let cls = crate::builtins::lookup_exc_class("SystemExit")
                .ok_or_else(|| crate::PyError::runtime_error("SystemExit class missing"))?;
            let ctor_args = if unsafe { is_tuple(exitcode) } {
                unsafe { w_tuple_items_copy_as_vec(exitcode) }
            } else {
                vec![exitcode]
            };
            let exc = crate::call::call_function_impl_result(cls, &ctor_args)?;
            Err(unsafe { crate::PyError::from_exc_object(exc) })
        }),
    );
    // `init_sys_streams`: Windows has no `sys.abiflags`; its ABI tag
    // belongs to `sys.winver`.  Absence is observable through `hasattr` and is
    // relied on by the 3.14 Windows sys tests, so do not publish an empty
    // stand-in.
    #[cfg(not(windows))]
    module_ns_store(roots.get(ns_slot), "abiflags", w_str_new("t"));
    // sys.argv — pick up pending argv from set_sys_argv if available.
    let pending = crate::importing::take_pending_sys_argv();
    let argv = if pending.is_null() {
        w_list_new(vec![])
    } else {
        pending
    };
    module_ns_store(roots.get(ns_slot), "argv", argv);
    // sys.warnoptions — each decoded option is freshly allocated and the next
    // decode allocates again, so they are pinned as they arrive
    // (`build_list_storage`).
    {
        let mut warnoptions = pyre_object::gc_roots::RootedItems::new();
        for option in crate::importing::warnoptions() {
            warnoptions.push(crate::gateway::fsdecode_os_str(&option));
        }
        let warn_list = w_list_new(warnoptions.take());
        module_ns_store(roots.get(ns_slot), "warnoptions", warn_list);
    }
    // sys.builtin_module_names — tuple of names of modules compiled into
    // the interpreter. PyPy: pypy/module/sys/state.py get_builtin_module_names,
    // which reads the same registry `import` resolves against, so the
    // advertised set cannot drift from what is actually importable on a build.
    let builtin_names = crate::importing::builtin_module_names();
    module_ns_store(
        roots.get(ns_slot),
        "builtin_module_names",
        w_tuple_new(builtin_names.into_iter().map(w_str_new).collect()),
    );
    // sys.stdlib_module_names — frozenset of every top-level standard-library
    // module name (`Python/stdlib_module_names.h`).  Read by
    // `traceback.TracebackException` for "did you forget to import" hints and
    // by the module shadowing check for the stronger stdlib rename hint.  The
    // list is the full set regardless of platform, as upstream ships it.
    const STDLIB_MODULE_NAMES: &[&str] = &[
        "__future__",
        "_abc",
        "_aix_support",
        "_android_support",
        "_apple_support",
        "_ast",
        "_ast_unparse",
        "_asyncio",
        "_bisect",
        "_blake2",
        "_bz2",
        "_codecs",
        "_codecs_cn",
        "_codecs_hk",
        "_codecs_iso2022",
        "_codecs_jp",
        "_codecs_kr",
        "_codecs_tw",
        "_collections",
        "_collections_abc",
        "_colorize",
        "_compat_pickle",
        "_contextvars",
        "_csv",
        "_ctypes",
        "_curses",
        "_curses_panel",
        "_datetime",
        "_dbm",
        "_decimal",
        "_elementtree",
        "_frozen_importlib",
        "_frozen_importlib_external",
        "_functools",
        "_gdbm",
        "_hashlib",
        "_heapq",
        "_hmac",
        "_imp",
        "_interpchannels",
        "_interpqueues",
        "_interpreters",
        "_io",
        "_ios_support",
        "_json",
        "_locale",
        "_lsprof",
        "_lzma",
        "_markupbase",
        "_md5",
        "_multibytecodec",
        "_multiprocessing",
        "_opcode",
        "_opcode_metadata",
        "_operator",
        "_osx_support",
        "_overlapped",
        "_pickle",
        "_posixshmem",
        "_posixsubprocess",
        "_py_abc",
        "_py_warnings",
        "_pydatetime",
        "_pydecimal",
        "_pyio",
        "_pylong",
        "_pyrepl",
        "_queue",
        "_random",
        "_remote_debugging",
        "_scproxy",
        "_sha1",
        "_sha2",
        "_sha3",
        "_signal",
        "_sitebuiltins",
        "_socket",
        "_sqlite3",
        "_sre",
        "_ssl",
        "_stat",
        "_statistics",
        "_string",
        "_strptime",
        "_struct",
        "_suggestions",
        "_symtable",
        "_sysconfig",
        "_thread",
        "_threading_local",
        "_tkinter",
        "_tokenize",
        "_tracemalloc",
        "_types",
        "_typing",
        "_uuid",
        "_warnings",
        "_weakref",
        "_weakrefset",
        "_winapi",
        "_wmi",
        "_zoneinfo",
        "_zstd",
        "abc",
        "annotationlib",
        "antigravity",
        "argparse",
        "array",
        "ast",
        "asyncio",
        "atexit",
        "base64",
        "bdb",
        "binascii",
        "bisect",
        "builtins",
        "bz2",
        "cProfile",
        "calendar",
        "cmath",
        "cmd",
        "code",
        "codecs",
        "codeop",
        "collections",
        "colorsys",
        "compileall",
        "compression",
        "concurrent",
        "configparser",
        "contextlib",
        "contextvars",
        "copy",
        "copyreg",
        "csv",
        "ctypes",
        "curses",
        "dataclasses",
        "datetime",
        "dbm",
        "decimal",
        "difflib",
        "dis",
        "doctest",
        "email",
        "encodings",
        "ensurepip",
        "enum",
        "errno",
        "faulthandler",
        "fcntl",
        "filecmp",
        "fileinput",
        "fnmatch",
        "fractions",
        "ftplib",
        "functools",
        "gc",
        "genericpath",
        "getopt",
        "getpass",
        "gettext",
        "glob",
        "graphlib",
        "grp",
        "gzip",
        "hashlib",
        "heapq",
        "hmac",
        "html",
        "http",
        "idlelib",
        "imaplib",
        "importlib",
        "inspect",
        "io",
        "ipaddress",
        "itertools",
        "json",
        "keyword",
        "linecache",
        "locale",
        "logging",
        "lzma",
        "mailbox",
        "marshal",
        "math",
        "mimetypes",
        "mmap",
        "modulefinder",
        "msvcrt",
        "multiprocessing",
        "netrc",
        "nt",
        "ntpath",
        "nturl2path",
        "numbers",
        "opcode",
        "operator",
        "optparse",
        "os",
        "pathlib",
        "pdb",
        "pickle",
        "pickletools",
        "pkgutil",
        "platform",
        "plistlib",
        "poplib",
        "posix",
        "posixpath",
        "pprint",
        "profile",
        "pstats",
        "pty",
        "pwd",
        "py_compile",
        "pyclbr",
        "pydoc",
        "pydoc_data",
        "pyexpat",
        "queue",
        "quopri",
        "random",
        "re",
        "readline",
        "reprlib",
        "resource",
        "rlcompleter",
        "runpy",
        "sched",
        "secrets",
        "select",
        "selectors",
        "shelve",
        "shlex",
        "shutil",
        "signal",
        "site",
        "smtplib",
        "socket",
        "socketserver",
        "sqlite3",
        "sre_compile",
        "sre_constants",
        "sre_parse",
        "ssl",
        "stat",
        "statistics",
        "string",
        "stringprep",
        "struct",
        "subprocess",
        "symtable",
        "sys",
        "sysconfig",
        "syslog",
        "tabnanny",
        "tarfile",
        "tempfile",
        "termios",
        "textwrap",
        "this",
        "threading",
        "time",
        "timeit",
        "tkinter",
        "token",
        "tokenize",
        "tomllib",
        "trace",
        "traceback",
        "tracemalloc",
        "tty",
        "turtle",
        "turtledemo",
        "types",
        "typing",
        "unicodedata",
        "unittest",
        "urllib",
        "uuid",
        "venv",
        "warnings",
        "wave",
        "weakref",
        "webbrowser",
        "winreg",
        "winsound",
        "wsgiref",
        "xml",
        "xmlrpc",
        "zipapp",
        "zipfile",
        "zipimport",
        "zlib",
        "zoneinfo",
    ];
    module_ns_store(
        roots.get(ns_slot),
        "stdlib_module_names",
        pyre_object::setobject::w_frozenset_from_items(
            &STDLIB_MODULE_NAMES
                .iter()
                .map(|&n| w_str_new(n))
                .collect::<Vec<_>>(),
        ),
    );
    // sys.exception() — the value half of `sys.exc_info()`: the exception
    // instance currently being handled, or None outside an `except` block.
    module_ns_store(
        roots.get(ns_slot),
        "exception",
        make_builtin_function_with_arity("exception", __majit_wrap_sys_exception, 0),
    );
    module_ns_store(
        roots.get(ns_slot),
        "call_tracing",
        make_builtin_function_with_arity(
            "call_tracing",
            |args| {
                if !unsafe { pyre_object::is_tuple(args[1]) } {
                    return Err(crate::PyError::type_error(format!(
                        "call_tracing() argument 2 must be tuple, not {}",
                        crate::type_methods::clinic_arg_type_name(args[1])
                    )));
                }
                let ec = current_execution_context();
                if ec.is_null() {
                    return Err(crate::PyError::runtime_error(
                        "no current execution context",
                    ));
                }
                let result = unsafe { (*ec).call_tracing(args[0], args[1]) };
                if result.is_null() {
                    Err(crate::call::take_call_error()
                        .unwrap_or_else(|| crate::PyError::runtime_error("call_tracing failed")))
                } else {
                    Ok(result)
                }
            },
            2,
        ),
    );
    module_ns_store(
        roots.get(ns_slot),
        "_clear_internal_caches",
        make_builtin_function_with_arity(
            "_clear_internal_caches",
            |_| {
                crate::baseobjspace::clear_method_cache();
                crate::objspace::std::mapdict::clear_map_attr_cache();
                Ok(w_none())
            },
            0,
        ),
    );
    module_ns_store(
        roots.get(ns_slot),
        "_clear_type_cache",
        make_builtin_function_with_arity(
            "_clear_type_cache",
            |_| {
                crate::warn::warn_category(
                    "sys._clear_type_cache() is deprecated and scheduled for removal in a future version. Use sys._clear_internal_caches() instead.",
                    "DeprecationWarning",
                    1,
                )?;
                crate::baseobjspace::clear_method_cache();
                crate::objspace::std::mapdict::clear_map_attr_cache();
                Ok(w_none())
            },
            0,
        ),
    );
    module_ns_store(
        roots.get(ns_slot),
        "_debugmallocstats",
        make_builtin_function_with_arity("_debugmallocstats", |_| Ok(w_none()), 0),
    );
    module_ns_store(
        roots.get(ns_slot),
        "getallocatedblocks",
        make_builtin_function_with_arity(
            "getallocatedblocks",
            // MiniMark accounts bytes rather than allocator blocks.  Zero is
            // the documented answer for a build without pymalloc accounting.
            |_| Ok(w_int_new(0)),
            0,
        ),
    );
    module_ns_store(
        roots.get(ns_slot),
        "getunicodeinternedsize",
        crate::make_builtin_function("getunicodeinternedsize", |args| {
            let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
            crate::builtins::kwarg_reject_unknown(
                kwargs,
                &["_only_immortal"],
                "getunicodeinternedsize",
            )?;
            if !positional.is_empty() {
                return Err(crate::PyError::type_error(
                    "getunicodeinternedsize() takes no positional arguments",
                ));
            }
            let only_immortal = match crate::builtins::kwarg_get(kwargs, "_only_immortal") {
                Some(value) => crate::baseobjspace::is_true(value)?,
                None => false,
            };
            let size = if only_immortal {
                pyre_object::unicodeobject::interned_size_immortal()
            } else {
                pyre_object::unicodeobject::interned_size()
            };
            Ok(w_int_new(size as i64))
        }),
    );
    module_ns_store(
        roots.get(ns_slot),
        "_is_interned",
        make_builtin_function_with_arity(
            "_is_interned",
            |args| {
                if !unsafe { pyre_object::is_str(args[0]) } {
                    return Err(crate::PyError::type_error(format!(
                        "_is_interned() argument must be str, not {}",
                        crate::type_methods::clinic_arg_type_name(args[0]),
                    )));
                }
                let exact = unsafe { pyre_object::is_exact_type(args[0], &pyre_object::STR_TYPE) };
                Ok(w_bool_from(
                    exact && unsafe { pyre_object::unicodeobject::is_interned_exact_str(args[0]) },
                ))
            },
            1,
        ),
    );
    module_ns_store(
        roots.get(ns_slot),
        "_is_immortal",
        make_builtin_function_with_arity(
            "_is_immortal",
            |args| {
                Ok(w_bool_from(!pyre_object::gc_hook::try_gc_owns_object(
                    args[0].cast(),
                )))
            },
            1,
        ),
    );
    module_ns_store(
        roots.get(ns_slot),
        "_is_gil_enabled",
        // Whether the interpreter is *currently* running with the lock on,
        // which is a separate question from the `t` ABI `sysconfig` publishes.
        // pyre runs bytecode under `rgil` (majit-gc/src/rgil.rs:21-31): one
        // process-wide lock, taken when a thread starts and dropped only around
        // external calls, so a thread holds it across arbitrarily many
        // bytecodes.  That is a GIL, and it is never off.
        make_builtin_function_with_arity("_is_gil_enabled", |_| Ok(w_bool_from(true)), 0),
    );
    module_ns_store(
        roots.get(ns_slot),
        "activate_stack_trampoline",
        make_builtin_function_with_arity(
            "activate_stack_trampoline",
            |_| Err(crate::PyError::value_error("perf trampoline not available")),
            1,
        ),
    );
    module_ns_store(
        roots.get(ns_slot),
        "deactivate_stack_trampoline",
        make_builtin_function_with_arity("deactivate_stack_trampoline", |_| Ok(w_none()), 0),
    );
    module_ns_store(
        roots.get(ns_slot),
        "is_stack_trampoline_active",
        make_builtin_function_with_arity(
            "is_stack_trampoline_active",
            |_| Ok(w_bool_from(false)),
            0,
        ),
    );
    module_ns_store(
        roots.get(ns_slot),
        "_dump_tracelets",
        make_builtin_function_with_arity(
            "_dump_tracelets",
            |args| {
                let _ = crate::gateway::fsencode_bytes_w(args[0])?;
                Err(crate::PyError::not_implemented("No JIT available"))
            },
            1,
        ),
    );
    module_ns_store(
        roots.get(ns_slot),
        "_baserepl",
        make_builtin_function_with_arity("_baserepl", sys_baserepl, 0),
    );
    module_ns_store(
        roots.get(ns_slot),
        "remote_exec",
        make_builtin_function_with_arity(
            "remote_exec",
            |args| {
                // `args` is the gateway's native copy; `__index__` can collect.
                let mut w_script = args[1];
                let _pid = pyre_object::with_roots!(
                    w_script => crate::builtins::space_index_w(args[0])
                )?;
                let _script = crate::gateway::fsencode_bytes_w(w_script)?;
                Err(crate::PyError::runtime_error(
                    "remote debugging is not supported by this interpreter",
                ))
            },
            2,
        ),
    );
    // sys.is_remote_debug_enabled() — no remote-debug interface is wired,
    // so always False.
    module_ns_store(
        roots.get(ns_slot),
        "is_remote_debug_enabled",
        make_builtin_function_with_arity(
            "is_remote_debug_enabled",
            |_| Ok(pyre_object::w_bool_from(false)),
            0,
        ),
    );
    // sys.copyright — informational string consumed by `site` and `test`.
    module_ns_store(
        roots.get(ns_slot),
        "copyright",
        w_str_new("Copyright (c) 2001-2024 Python Software Foundation.\nAll Rights Reserved."),
    );
    module_ns_store(
        roots.get(ns_slot),
        "getsizeof",
        crate::gateway::make_builtin_function_with_doc(
            "getsizeof",
            |args| {
                let scope = crate::builtins::bind_builtin_kwargs(
                    args,
                    &["object", "default"],
                    &[true, false],
                    "getsizeof",
                )?;
                let w_obj = scope[0];
                let w_default = scope[1];
                let roots = pyre_object::gc_roots::push_roots();
                let obj_slot = roots.base();
                let _ = roots.pin_root(w_obj);
                let default_slot = obj_slot + 1;
                let w_default = roots.pin_root(w_default);
                match get_sizeof(roots.get(obj_slot)) {
                    Ok(size) => Ok(size),
                    Err(err)
                        if !w_default.is_null() && err.kind == crate::PyErrorKind::TypeError =>
                    {
                        Ok(roots.get(default_slot))
                    }
                    Err(err) => Err(err),
                }
            },
            GETSIZEOF_DOC,
        ),
    );
    // PyPy normally omits CPython's raw refcount API.  The shared ctypes
    // tests only require the strong-reference delta created by a c_char_p
    // `_objects` keepalive; bytes records that real ownership transition in
    // its object payload, while other tracing-GC objects report the stable
    // call/argument baseline.
    module_ns_store(
        roots.get(ns_slot),
        "getrefcount",
        make_builtin_function_with_arity(
            "getrefcount",
            |args| {
                let owned = if unsafe { pyre_object::is_bytes(args[0]) } {
                    unsafe { pyre_object::bytesobject::w_bytes_ctypes_keepalive_refs(args[0]) }
                } else {
                    0
                };
                Ok(pyre_object::w_int_new((2 + owned) as i64))
            },
            1,
        ),
    );
    // sys.gettrace / settrace
    module_ns_store(
        roots.get(ns_slot),
        "gettrace",
        make_builtin_function_with_arity("gettrace", sys_gettrace_impl, 0),
    );
    module_ns_store(
        roots.get(ns_slot),
        "settrace",
        make_builtin_function_with_arity("settrace", sys_settrace_impl, 1),
    );
    // sys.getprofile / setprofile
    module_ns_store(
        roots.get(ns_slot),
        "getprofile",
        make_builtin_function_with_arity("getprofile", sys_getprofile_impl, 0),
    );
    module_ns_store(
        roots.get(ns_slot),
        "setprofile",
        make_builtin_function_with_arity("setprofile", sys_setprofile_impl, 1),
    );
    module_ns_store(
        roots.get(ns_slot),
        "_settraceallthreads",
        make_builtin_function_with_arity("_settraceallthreads", sys_settraceallthreads_impl, 1),
    );
    module_ns_store(
        roots.get(ns_slot),
        "_setprofileallthreads",
        make_builtin_function_with_arity("_setprofileallthreads", sys_setprofileallthreads_impl, 1),
    );
    module_ns_store(
        roots.get(ns_slot),
        "get_coroutine_origin_tracking_depth",
        make_builtin_function_with_arity(
            "get_coroutine_origin_tracking_depth",
            sys_get_coroutine_origin_tracking_depth,
            0,
        ),
    );
    module_ns_store(
        roots.get(ns_slot),
        "set_coroutine_origin_tracking_depth",
        // `depth` is positional-or-keyword, so this cannot take the
        // fixed-arity carrier (which rejects keywords before the body runs).
        crate::make_builtin_function(
            "set_coroutine_origin_tracking_depth",
            sys_set_coroutine_origin_tracking_depth,
        ),
    );
    module_ns_store(
        roots.get(ns_slot),
        "get_asyncgen_hooks",
        make_builtin_function_with_arity("get_asyncgen_hooks", sys_get_asyncgen_hooks_impl, 0),
    );
    module_ns_store(
        roots.get(ns_slot),
        "set_asyncgen_hooks",
        crate::make_builtin_function("set_asyncgen_hooks", sys_set_asyncgen_hooks_impl),
    );
    // sys.getfilesystemencoding
    module_ns_store(
        roots.get(ns_slot),
        "getfilesystemencoding",
        make_builtin_function_with_arity(
            "getfilesystemencoding",
            |_| Ok(w_str_new(crate::typedef::fs_encoding())),
            0,
        ),
    );
    // `interp_encoding.py:8-12 base_error` — PEP 529 puts Windows on
    // `surrogatepass`, every other platform on `surrogateescape`. `os.fsencode`
    // reads this at import, so it decides the app-level conversion the same way
    // `gateway::fsencode` decides the interpreter's.
    module_ns_store(
        roots.get(ns_slot),
        "getfilesystemencodeerrors",
        make_builtin_function_with_arity(
            "getfilesystemencodeerrors",
            |_| Ok(w_str_new(crate::typedef::fs_errors())),
            0,
        ),
    );
    #[cfg(windows)]
    module_ns_store(
        roots.get(ns_slot),
        "_enablelegacywindowsfsencoding",
        make_builtin_function_with_arity(
            "_enablelegacywindowsfsencoding",
            |_| {
                crate::warn::warn_category(
                    "sys._enablelegacywindowsfsencoding() is deprecated and will be removed in Python 3.16. Use PYTHONLEGACYWINDOWSFSENCODING instead.",
                    "DeprecationWarning",
                    1,
                )?;
                // `_PyUnicode_EnableLegacyWindowsFSEncoding` re-runs
                // `init_fs_codec`, so the flag moves the two conversions the
                // codec names as well as the getters that report it:
                // `gateway::fs_arg_bytes` for a path spelled as `bytes` and
                // `gateway::fs_result_bytes` for a name reported back as one.
                // A `str` path is untouched, as `path_converter` keeps the
                // wide string it was given and asks no codec for it — measured
                // at 3.14: `os.stat` still finds a name the code page cannot
                // spell, while `os.listdir(b'.')` reports it as `b'?'`.
                crate::typedef::LEGACY_WINDOWS_FS_ENCODING
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                Ok(w_none())
            },
            0,
        ),
    );
    // `vm.py audit` — `@unwrap_spec(event="text")`, so the event name is
    // required and must be a str; everything after it is the argument tuple.
    module_ns_store(
        roots.get(ns_slot),
        "audit",
        crate::make_builtin_function("audit", sys_audit),
    );
    // sys._clear_type_descriptors(cls) — remove the descriptors owned by the
    // original class before `dataclasses._add_slots` copies its namespace into
    // the replacement slotted class.
    module_ns_store(
        roots.get(ns_slot),
        "_clear_type_descriptors",
        make_builtin_function_with_arity("_clear_type_descriptors", sys_clear_type_descriptors, 1),
    );
    // sys.is_finalizing
    module_ns_store(
        roots.get(ns_slot),
        "is_finalizing",
        make_builtin_function_with_arity(
            "is_finalizing",
            |_| Ok(w_bool_from(crate::module::thread::is_finalizing())),
            0,
        ),
    );
    // sys.displayhook / excepthook. The double-underscore names keep the
    // original hooks so code (e.g. doctest and exception tests) can save and
    // restore them.
    let displayhook_fn =
        make_builtin_function_with_arity("displayhook", crate::builtins::sys_displayhook, 1);
    module_ns_store(roots.get(ns_slot), "displayhook", displayhook_fn);
    module_ns_store(roots.get(ns_slot), "__displayhook__", displayhook_fn);
    let excepthook_fn =
        make_builtin_function_with_arity("excepthook", crate::builtins::sys_excepthook, 3);
    module_ns_store(roots.get(ns_slot), "excepthook", excepthook_fn);
    module_ns_store(roots.get(ns_slot), "__excepthook__", excepthook_fn);
    // sys.breakpointhook — `app.py breakpointhook`, called by `breakpoint()`.
    // `__breakpointhook__` keeps the original so code can restore it.
    let breakpointhook_fn = make_builtin_function("breakpointhook", sys_breakpointhook);
    module_ns_store(roots.get(ns_slot), "breakpointhook", breakpointhook_fn);
    module_ns_store(roots.get(ns_slot), "__breakpointhook__", breakpointhook_fn);
    // sys.unraisablehook(unraisable) — handles exceptions raised where they
    // cannot propagate (e.g. __del__).  Stored alongside the read-only
    // `__unraisablehook__` original so code can save and restore it.
    let unraisablehook_fn =
        make_builtin_function_with_arity("unraisablehook", sys_unraisablehook, 1);
    module_ns_store(roots.get(ns_slot), "unraisablehook", unraisablehook_fn);
    module_ns_store(roots.get(ns_slot), "__unraisablehook__", unraisablehook_fn);
    // sys.path_hooks / path_importer_cache
    let path_hooks = w_list_new(vec![]);
    module_ns_store(roots.get(ns_slot), "path_hooks", path_hooks);
    module_ns_store(roots.get(ns_slot), "path_importer_cache", w_dict_new());
    // sys.meta_path — empty
    let meta_path = w_list_new(vec![]);
    module_ns_store(roots.get(ns_slot), "meta_path", meta_path);
    // sys.dont_write_bytecode — mirrors `sys.flags.dont_write_bytecode`
    // (`-B` / PYTHONDONTWRITEBYTECODE). The flag is honoured, not merely
    // reported: `_bootstrap_external` writes `__pycache__/*.pyre314.pyc` next
    // to every source it imports unless it is set.
    module_ns_store(
        roots.get(ns_slot),
        "dont_write_bytecode",
        w_bool_from(crate::importing::dont_write_bytecode_flag()),
    );
    // sys.pycache_prefix — None unless -X pycache_prefix / PYTHONPYCACHEPREFIX.
    // `importlib._bootstrap_external.cache_from_source` reads it to compute the
    // bytecode path before `dont_write_bytecode` is consulted.  The path is
    // decoded the way `sys._xoptions` decodes the same bytes, so one the host
    // cannot spell in UTF-8 arrives with the surrogates that re-encode to it.
    module_ns_store(
        roots.get(ns_slot),
        "pycache_prefix",
        crate::importing::pycache_prefix()
            .map_or_else(w_none, |path| crate::gateway::fsdecode_os_str(&path)),
    );
    // `vm.py addaudithook`
    module_ns_store(
        roots.get(ns_slot),
        "addaudithook",
        make_builtin_function_with_arity("addaudithook", sys_addaudithook, 1),
    );
    // `space.fromcache(AuditHolder)` — see [`audit_holder`] for why the holder
    // is built here rather than on the first `addaudithook`.
    audit_holder();
    Ok(())
}

/// `pypy/module/sys/vm.py AuditHolder`, which upstream reaches through
/// `space.fromcache(AuditHolder)`.
///
/// pyre has no space object graph to hang a cache object off, so the holder is
/// a process-global — the shape `_codecs`' `CodecState` already uses for a
/// registry that belongs to the interpreter rather than to one mutator.  Its
/// Python objects are handed to the collector by [`walk_audit_hooks_gc`].
pub struct AuditHolder {
    /// `vm.py AuditHolder.__init__ self.hooks_w = None` projected onto a byte at a fixed offset:
    /// false while upstream's list is None, which is exactly the `vm.py`
    /// early-out.  The projection exists because the JIT reads this field
    /// through a descriptor, and `Box<[T]>` has no target-stable null spelling
    /// — a wasm32 fat pointer is two 4-byte halves, not one 8-byte word.
    pub hooks_armed: std::cell::Cell<bool>,
    /// The hidden watcher field implementing `vm.py:439 _immutable_fields_ =
    /// ['hooks_w?[:]']`.  The owner is a `Box::into_raw` leak that is never
    /// freed, so [`pyre_object::quasiimmut::QuasiImmutField`]'s `Drop` is
    /// unreachable and its inner box is reclaimed only through `invalidate`.
    pub hooks_watchers: pyre_object::quasiimmut::QuasiImmutField,
    /// A boxed slice rather than a `Vec` because `vm.py
    /// debug.make_sure_not_resized` is the other half of that declaration:
    /// [`sys_addaudithook`] replaces the list, never grows it in place.
    hooks_w: Box<[pyre_object::PyObjectRef]>,
}

static AUDIT_HOOKS: std::sync::atomic::AtomicPtr<AuditHolder> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

/// `space.fromcache(AuditHolder)`.
///
/// Created eagerly, from [`register_module`], rather than on the first
/// `addaudithook`: a trace recorded while no hook exists pins `hooks_w?` on
/// this object, so the object has to exist before it does.  `fromcache` gives
/// upstream the same guarantee for free.
fn audit_holder() -> *mut AuditHolder {
    loop {
        let holder = AUDIT_HOOKS.load(std::sync::atomic::Ordering::Acquire);
        if !holder.is_null() {
            return holder;
        }
        let created = Box::into_raw(Box::new(AuditHolder {
            hooks_armed: std::cell::Cell::new(false),
            hooks_watchers: pyre_object::quasiimmut::QuasiImmutField::new(),
            hooks_w: Box::new([]),
        }));
        if AUDIT_HOOKS
            .compare_exchange(
                std::ptr::null_mut(),
                created,
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Acquire,
            )
            .is_ok()
        {
            return created;
        }
        // Another thread published one first; drop ours and take theirs.
        drop(unsafe { Box::from_raw(created) });
    }
}

/// The holder a `QUASIIMMUT_FIELD` on `hooks_w?` hangs off, or null before
/// `sys` is registered — the compiler declines the fold rather than pinning a
/// field on nothing.
pub fn audit_holder_ptr() -> *const AuditHolder {
    AUDIT_HOOKS.load(std::sync::atomic::Ordering::Acquire)
}

/// `quasiimmut.py get_current_qmut_instance` for the holder's
/// `hooks_w?` field.
///
/// # Safety
/// `holder` must be null or a live [`AuditHolder`].
pub unsafe fn audit_holder_current_hooks_qmut(
    holder: *const AuditHolder,
) -> Option<std::sync::Arc<pyre_object::quasiimmut::QuasiImmut>> {
    if holder.is_null() {
        return None;
    }
    Some(unsafe { (*holder).hooks_watchers.get_current_qmut_instance() })
}

/// Forward the installed hooks.  Upstream reaches them through the space's
/// object graph; pyre holds them off-heap, so the collector has to be handed
/// them — and unconditionally, not behind the prebuilt-roots bit: a hook is
/// installed by running app code, so its callable is young when it lands.
pub(crate) fn walk_audit_hooks_gc(visitor: &mut dyn FnMut(&mut pyre_object::PyObjectRef)) {
    let holder = AUDIT_HOOKS.load(std::sync::atomic::Ordering::Acquire);
    if holder.is_null() {
        return;
    }
    for slot in unsafe { (*holder).hooks_w.iter_mut() } {
        visitor(slot);
    }
}

/// `vm.py AuditHolder.trigger_audit_events`.
fn trigger_audit_events(
    holder: &AuditHolder,
    w_event: pyre_object::PyObjectRef,
    args_w: &[pyre_object::PyObjectRef],
) -> Result<(), crate::PyError> {
    // `pin_root` copies the pointer into a shadow-stack slot and the collector
    // rewrites THAT slot, never the local it was copied from.  So every value
    // still needed after one of the app-level calls below is reached through
    // its slot index, the way `baseobjspace::isinstance` does it.
    //
    // The event, the arguments and the hook set are one livevar set spanning
    // three slices, so they are published together and normalized once
    // (`gc_roots::pin_roots`): a per-value pin queries the collector after the
    // first write, which would let a foreign collection run while the values
    // behind it were still invisible to it.
    //
    // A hook may install another hook, and upstream's list is replaced rather
    // than appended to, so an in-flight trigger keeps iterating the set it
    // started with.  The published slots ARE that snapshot — they keep every
    // callable forwarded across the calls below and are unaffected by a
    // replacement of `holder.hooks_w`.
    let _roots = pyre_object::gc_roots::push_roots();
    let event_slot = pyre_object::gc_roots::publish_roots(&[w_event]);
    let args_slot = pyre_object::gc_roots::publish_roots(args_w);
    let hooks_slot = pyre_object::gc_roots::publish_roots(&holder.hooks_w);
    let hook_count = holder.hooks_w.len();
    pyre_object::gc_roots::normalize_roots(event_slot, 1 + args_w.len() + hook_count);

    // From the published slots, not from the caller's slice: the tuple's own
    // allocation can collect, and so can everything reached before it.
    let items = (0..args_w.len())
        .map(|i| pyre_object::gc_roots::shadow_stack_get(args_slot + i))
        .collect();
    let args_tuple_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pyre_object::tupleobject::w_tuple_new(items));

    let ec = crate::call::getexecutioncontext() as *mut crate::executioncontext::ExecutionContext;
    // don't trace audithooks by default
    if !ec.is_null() {
        unsafe { (*ec).is_tracing += 1 };
    }
    let mut result = Ok(());
    for i in 0..hook_count {
        let w_hook = pyre_object::gc_roots::shadow_stack_get(hooks_slot + i);
        // `space.findattr` (`baseobjspace.py`) answers `None` for ANY
        // non-async error out of the lookup, so a hook whose `__cantrace__`
        // descriptor raises is simply treated as not having one.  The bare
        // `findattr` panics on those instead, and this argument is app code, so
        // it is the async arm alone (`error.py`; pyre carries SystemExit
        // of that pair today) that may travel out of here.
        let w_cantrace = match crate::baseobjspace::findattr_result(w_hook, "__cantrace__") {
            Ok(found) => found,
            Err(err) if err.kind == crate::PyErrorKind::SystemExit => {
                result = Err(err);
                break;
            }
            Err(_) => None,
        };
        let cantrace = match w_cantrace {
            None => false,
            Some(w_cantrace) => match crate::baseobjspace::is_true(w_cantrace) {
                Ok(cantrace) => cantrace,
                Err(err) => {
                    result = Err(err);
                    break;
                }
            },
        };
        if cantrace && !ec.is_null() {
            unsafe { (*ec).is_tracing -= 1 };
        }
        let w_result = crate::baseobjspace::call_function(
            pyre_object::gc_roots::shadow_stack_get(hooks_slot + i),
            &[
                pyre_object::gc_roots::shadow_stack_get(event_slot),
                pyre_object::gc_roots::shadow_stack_get(args_tuple_slot),
            ],
        );
        if cantrace && !ec.is_null() {
            unsafe { (*ec).is_tracing += 1 };
        }
        if w_result.is_null() {
            result = Err(crate::call::take_call_error()
                .unwrap_or_else(|| crate::PyError::runtime_error("audit hook call failed")));
            break;
        }
    }
    if !ec.is_null() {
        unsafe { (*ec).is_tracing -= 1 };
    }
    result
}

/// `vm.py audit` with the event already wrapped, for callers that hold a
/// `str` object rather than a Rust `&str`.
pub fn audit_w(
    w_event: pyre_object::PyObjectRef,
    args_w: &[pyre_object::PyObjectRef],
) -> Result<(), crate::PyError> {
    if !audit_hooks_armed() {
        return Ok(());
    }
    let holder = AUDIT_HOOKS.load(std::sync::atomic::Ordering::Acquire);
    trigger_audit_events(unsafe { &*holder }, w_event, args_w)
}

/// Traced `sys._getframe` event once hooks are armed.
///
/// `vm.py trigger_audit_events` is `@objectmodel.dont_inline`. The walker
/// keeps the frame walk and `mark_as_escaped` in the trace and residuals
/// only this call, so `getframe` itself does not force the portal
/// virtualizable. A raising hook publishes the exception and returns null.
#[majit_macros::dont_look_inside]
pub fn jit_audit_sys_getframe(frame: pyre_object::PyObjectRef) -> pyre_object::PyObjectRef {
    if frame.is_null() {
        return frame;
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let frame_slot = pyre_object::gc_roots::pin_roots(&[frame]);
    let frame = pyre_object::gc_roots::shadow_stack_get(frame_slot);
    match audit("sys._getframe", &[frame]) {
        Ok(()) => pyre_object::gc_roots::shadow_stack_get(frame_slot),
        Err(mut err) => {
            crate::eval::set_current_exception(err.to_exc_object());
            std::ptr::null_mut()
        }
    }
}

/// `vm.py audit` — `space.audit(event, args_w)` for interpreter-level
/// emitters.  Free when no hook is installed, which is what lets a caller on a
/// hot path emit unconditionally.
pub fn audit(event: &str, args_w: &[pyre_object::PyObjectRef]) -> Result<(), crate::PyError> {
    if !audit_hooks_armed() {
        return Ok(());
    }
    // The wrap is a collection point and the arguments reach it only as copied
    // pointers — `call_function_impl_result` reloads its own from the shadow
    // stack for exactly that reason — so they are rooted in front of it and the
    // emit runs off the reloaded values.
    let _roots = pyre_object::gc_roots::push_roots();
    let args_slot = pyre_object::gc_roots::pin_roots(args_w);
    let w_event = pyre_object::gc_roots::pin_root(w_str_new_managed(event));
    let args_w: Vec<pyre_object::PyObjectRef> = (0..args_w.len())
        .map(|i| pyre_object::gc_roots::shadow_stack_get(args_slot + i))
        .collect();
    audit_w(w_event, &args_w)
}

/// `vm.py:481 holder.hooks_w is None`, negated.  A JIT fold that answers a call
/// without running the body it would have traced through needs this to decide
/// whether the event it is eliding could be observed.
pub fn audit_hooks_armed() -> bool {
    let holder = AUDIT_HOOKS.load(std::sync::atomic::Ordering::Acquire);
    !holder.is_null() && unsafe { (*holder).hooks_armed.get() }
}

/// `vm.py audit(space, event, args_w)` under `@unwrap_spec(event="text")`.
///
/// The unwrap in front of the hook dispatch is observable on its own: the
/// parameters are positional-only, and the event name has to be a `str` with a
/// UTF-8 spelling, so a bad event name is reported at the call rather than
/// carried to whichever hook reads it.
///
/// The unwrap is `str_utf8_w` and not a surrogate-tolerant read because the
/// encode is there for the error it raises: an event name holding a lone
/// surrogate is a `UnicodeEncodeError` at the call.
fn sys_audit(args: &[pyre_object::PyObjectRef]) -> crate::PyResult {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    if crate::builtins::has_real_kwargs(kwargs) {
        return Err(crate::PyError::type_error(
            "sys.audit() takes no keyword arguments",
        ));
    }
    let Some(&w_event) = positional.first() else {
        return Err(crate::PyError::type_error(
            "audit expected at least 1 argument, got 0",
        ));
    };
    // `@unwrap_spec(event="text")`
    if !unsafe { pyre_object::is_str(w_event) } {
        // `_PyArg_BadArgument` names the `None` singleton itself rather than
        // its type.
        let given = if unsafe { pyre_object::is_none(w_event) } {
            "None".to_string()
        } else {
            crate::type_methods::arg_type_name(w_event)
        };
        return Err(crate::PyError::type_error(format!(
            "audit() argument 1 must be str, not {given}"
        )));
    }
    // The `@unwrap_spec` round trip is observable: the hooks are handed the
    // `str` the unwrapped name is re-wrapped as, so a `str` subclass reaches
    // them flattened to a plain one.  The arguments behind the event are
    // rooted in front of the unwrap and the re-wrap because they reach them
    // only as copied pointers — the same bracket [`audit`] takes around its
    // own — and the owned copy is taken so no borrow of the event outlives
    // the unwrap.
    let _roots = pyre_object::gc_roots::push_roots();
    let args_slot = _roots.pin_roots(positional);
    let event = crate::baseobjspace::str_utf8_w(_roots.get(args_slot))?.to_string();
    let w_text = _roots.pin_root(w_str_new_managed(&event));
    let mut args_w: Vec<pyre_object::PyObjectRef> = Vec::with_capacity(positional.len() - 1);
    for i in 1..positional.len() {
        args_w.push(_roots.get(args_slot + i));
    }
    audit_w(w_text, &args_w)?;
    Ok(w_none())
}

/// `vm.py addaudithook`.  The hooks already installed get a say: an
/// `Exception` out of the `sys.addaudithook` event means the set refused the
/// new hook, and it is dropped rather than added.  A `BaseException` outside
/// `Exception` propagates.  The new hook is not installed yet when the event
/// fires, so it never gets to refuse its own installation.
fn sys_addaudithook(args: &[pyre_object::PyObjectRef]) -> crate::PyResult {
    let Some(&w_hook) = args.first() else {
        return Err(crate::PyError::type_error(
            "addaudithook() missing 1 required positional argument",
        ));
    };
    // The event below runs the already-installed hooks, which is app code and
    // can collect, so the pin has to precede it — and the value stored after it
    // has to come back out of the slot, since a relocation rewrites the slot
    // and not this local.
    let _roots = pyre_object::gc_roots::push_roots();
    let hook_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_hook);
    if let Err(err) = audit("sys.addaudithook", &[]) {
        // `error_is_exception` runs `isinstance` when the carrier holds an
        // object. The handle returned to the caller is the slot word.
        let mut err = err;
        let err_slot = err.pin(&_roots);
        let is_exception = error_is_exception(&err);
        err.reload(&_roots, err_slot);
        if !is_exception {
            return Err(err);
        }
        return Ok(w_none());
    }
    let holder = audit_holder();
    unsafe {
        // `holder.hooks_w = holder.hooks_w + [w_hook]` — a fresh list, so a
        // `trigger_audit_events` already iterating the old one keeps the set it
        // started with.  The new slice is complete before the notification, so
        // no borrow of `holder.hooks_w` is live across it.
        let mut next = {
            let old: &[pyre_object::PyObjectRef] = &(*holder).hooks_w;
            let mut next = Vec::with_capacity(old.len() + 1);
            next.extend_from_slice(old);
            next
        };
        next.push(pyre_object::gc_roots::shadow_stack_get(hook_slot));
        let next = next.into_boxed_slice();
        // Notification precedes the store, as `rclass.py:1010-1012
        // hook_setfield` emits `jit_force_quasi_immutable` before `setfield`.
        // The `is_installed()` fast path is `quasiimmut.py invalidation`'s
        // null test, so an unwatched install stays lock-free.
        if (*holder).hooks_watchers.is_installed() {
            pyre_object::quasiimmut::sweep_quasi_immut_field(&(*holder).hooks_watchers);
        }
        (*holder).hooks_w = next;
        (*holder).hooks_armed.set(true);
    }
    Ok(w_none())
}

/// `e.match(space, space.w_Exception)` for a `PyError` that may carry either an
/// interpreter-level kind or a materialised exception object.
///
/// The C-level `PySys_AddAuditHook` reads a refusal as `RuntimeError`, but the
/// `sys.addaudithook` a Python caller reaches widens it to `Exception`.
/// Measured against 3.14 with a refusing hook already installed: `RuntimeError`,
/// `ValueError` and `Exception` all leave the call returning `None`, and only a
/// `BaseException` outside `Exception` — `KeyboardInterrupt`, bare
/// `BaseException` — comes back out.
fn error_is_exception(err: &crate::PyError) -> bool {
    // A hook raises a real exception object, so the class test is the live one;
    // the interpreter-level kind is the fallback for an error that never
    // materialised an object.
    let w_exception = pyre_object::interp_exceptions::lookup_exc_class_for_kind(
        pyre_object::interp_exceptions::ExcKind::Exception,
    );
    if !err.exc_object.is_null() && !w_exception.is_null() {
        let _roots = pyre_object::gc_roots::push_roots();
        let exc_object = pyre_object::gc_roots::pin_root(err.exc_object);
        let w_exception = pyre_object::gc_roots::pin_root(w_exception);
        return crate::baseobjspace::isinstance(exc_object, w_exception).unwrap_or(false);
    }
    // `exc_kind_matches(kind, "Exception")` spelled over `PyErrorKind`: every
    // variant descends from `Exception` except the two `BaseException` ones.
    !matches!(
        err.kind,
        crate::PyErrorKind::GeneratorExit | crate::PyErrorKind::SystemExit
    )
}

/// `sysmodule.c sys._clear_type_descriptors`: remove the instance-dict and weakref
/// descriptors while retaining their references until both dictionary
/// mutations are complete, then invalidate the type lookup caches once.
fn sys_clear_type_descriptors(args: &[PyObjectRef]) -> crate::PyResult {
    let w_type = args[0];
    if !unsafe { pyre_object::is_type(w_type) } {
        return Err(crate::PyError::type_error(
            "_clear_type_descriptors() argument must be a type",
        ));
    }
    if unsafe { pyre_object::w_type_is_cpython_immutabletype(w_type) } {
        return Err(crate::PyError::type_error("argument is immutable"));
    }

    let _roots = pyre_object::gc_roots::push_roots();
    if let Some(descr) = crate::type_dict_lookup(w_type, "__dict__") {
        let _ = pyre_object::gc_roots::pin_root(descr);
    }
    if let Some(descr) = crate::type_dict_lookup(w_type, "__weakref__") {
        let _ = pyre_object::gc_roots::pin_root(descr);
    }
    crate::type_dict_delete(w_type, "__dict__");
    crate::type_dict_delete(w_type, "__weakref__");
    unsafe { crate::baseobjspace::mutated_absent(w_type) };
    Ok(w_none())
}

/// Construct a stdio object whose type is `_io.TextIOWrapper` (so
/// `isinstance(sys.stdout, io.TextIOWrapper)` holds), exposing `write`,
/// `flush`, `isatty`, `fileno`, `reconfigure`, and `name`.  `fd` is the
/// descriptor it reports: 0 (stdin) / 1 (stdout) / 2 (stderr).  PyPy wires a
/// real W_File-backed `TextIOWrapper`; pyre routes writes through Rust's
/// stdout/stderr (the same sink as `print`) so output ordering is preserved,
/// storing the read/write surface as instance attributes.
/// `app_main.py initstdio`'s test for the default error handler: UTF-8 mode,
/// or a locale selected by the platform policy, asks for `surrogateescape`;
/// every other locale settles for the `strict` that
/// `TextIOWrapper(errors=None)` takes.
///
/// Windows answers yes outright. `initstdio` reaches that through UTF-8 mode,
/// which PyPy runs the whole platform in -- `sys.flags.utf8_mode` reads 1 there
/// -- and pyre does not, because 3.14 reports 0 unless `-X utf8` asked for it;
/// `config_get_stdio_errors` states the same answer as its own `MS_WINDOWS`
/// arm instead of deriving it. Measured, both interpreters answer
/// `surrogateescape` on this platform. The locale test could not stand in for
/// it either: the MSVC runtime's `setlocale(LC_CTYPE, "")` reads the user's
/// ANSI locale and ignores `LC_ALL`, so it never sees C there.
fn locale_asks_for_surrogateescape() -> bool {
    #[cfg(windows)]
    {
        true
    }
    #[cfg(not(windows))]
    {
        if crate::importing::utf8_mode_flag() != 0 {
            return true;
        }
        #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
        {
            // `initstdio` installs the locale the environment names before
            // reading it back, so what is tested is the locale the C library
            // took rather than the variables that asked for it.
            rustpython_host_env::locale::setlocale(
                rustpython_host_env::locale::LC_CTYPE,
                Some(c""),
            );
            let effective =
                rustpython_host_env::locale::setlocale(rustpython_host_env::locale::LC_CTYPE, None);
            // [3.14-spec] CPython `config_get_stdio_errors` also consults
            // `_Py_IsLocaleCoercionTarget`; its target table is C.UTF-8,
            // C.utf8 and UTF-8.  PyPy `app_main.initstdio` owns the control
            // flow here, while those observable stdio errors follow 3.14.
            matches!(
                effective.as_deref(),
                None | Some(b"C")
                    | Some(b"POSIX")
                    | Some(b"C.UTF-8")
                    | Some(b"C.utf8")
                    | Some(b"UTF-8")
            )
        }
        // No locale database at all -- wasm32, and the sandbox, whose
        // `_locale.setlocale` answers `C` for every category.
        #[cfg(not(all(unix, feature = "host_env", not(feature = "sandbox"))))]
        {
            true
        }
    }
}

/// `app_main.py initstdio`: the encoding and error handler the standard streams
/// open with.
///
/// PYTHONIOENCODING is `encoding[:errors]`, and which half the user filled in
/// decides what an empty handler half means. A non-empty encoding is explicit
/// and settles for `strict`; an omitted one leaves the handler to
/// [`locale_asks_for_surrogateescape`]. So `iso8859-1` and `iso8859-1:` both
/// answer `strict`, while `:`, the empty value and no value at all answer
/// whatever the locale asks for. stderr replaces the handler with
/// `backslashreplace` separately.
///
/// An unnamed encoding resolves through [`default_stdio_encoding`] rather than
/// `initstdio`'s `"locale"` string, so the answer is fixed here instead of
/// inside `TextIOWrapper`.
///
/// Resolved once: `initstdio` runs once, and a later `locale.setlocale` must
/// not change the answer a stream is already open with.
fn stdio_encoding_and_errors() -> (String, String) {
    static RESOLVED: OnceLock<(String, String)> = OnceLock::new();
    RESOLVED
        .get_or_init(|| {
            let raw = crate::importing::stdio_encoding();
            let (encoding, errors, user_set_encoding) = match raw.as_deref() {
                Some(value) if value.contains(':') => {
                    let (encoding, errors) = value.split_once(':').expect("value holds a ':'");
                    let user_set_encoding = !encoding.is_empty();
                    let errors = if !errors.is_empty() {
                        Some(errors)
                    } else if user_set_encoding {
                        Some("strict")
                    } else {
                        None
                    };
                    let encoding = if encoding.is_empty() {
                        default_stdio_encoding()
                    } else {
                        encoding.to_string()
                    };
                    (encoding, errors, user_set_encoding)
                }
                Some(value) if !value.is_empty() => (value.to_string(), None, true),
                _ => (default_stdio_encoding(), None, false),
            };
            let errors = match errors {
                Some(errors) => errors.to_string(),
                None if !user_set_encoding && locale_asks_for_surrogateescape() => {
                    "surrogateescape".to_string()
                }
                None => "strict".to_string(),
            };
            (encoding, errors)
        })
        .clone()
}

/// `config_get_locale_encoding` — the encoding a standard stream opens with
/// once PYTHONIOENCODING has named none.
///
/// utf-8 mode fixes every stream at utf-8; otherwise the locale's own codeset
/// is the answer, which on Windows is the ANSI code page rather than utf-8.  A
/// console stream overrides it back to utf-8 in [`make_std_stream`], the way
/// `create_stdio` does after `_io.open` has told it which raw layer it got.
///
/// Only the Windows arm resolves the locale so far: `_Py_GetLocaleEncoding`'s
/// POSIX answer is the `nl_langinfo` codeset, and reading it here would make a
/// C-locale host open its streams as `ascii` — the encoding CPython avoids
/// through the PEP 538 locale coercion pyre does not implement yet.
fn default_stdio_encoding() -> String {
    if crate::importing::utf8_mode_flag() != 0 {
        return "utf-8".to_string();
    }
    #[cfg(windows)]
    {
        crate::module::_locale::interp_locale::locale_encoding()
    }
    #[cfg(not(windows))]
    {
        "utf-8".to_string()
    }
}

/// `create_stdio`'s `PyObject_TypeCheck(raw, winconsoleio_type)`: a console
/// stream is always utf-8 encoded, whatever encoding the configuration named.
///
/// The test is against the layout type rather than the class object, for two
/// reasons: `init_sys_streams` runs before `_io` has finished registering its
/// types, so `get_instantiate` still answers a null class here, and a subclass
/// of a builtin keeps the builtin's `ob_type` — which is what makes an
/// `ob_type` comparison the subclass-accepting check rather than an exact one.
#[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
fn raw_is_windows_console(raw: PyObjectRef) -> bool {
    if raw.is_null() {
        return false;
    }
    let pytype: *const pyre_object::PyType =
        <crate::module::_io::W_WinConsoleIO as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE;
    std::ptr::eq(unsafe { (*raw).ob_type }, pytype)
}

/// No `_WindowsConsoleIO` exists to compare against, so no stream is one.
#[cfg(not(all(windows, feature = "host_env", not(feature = "sandbox"))))]
fn raw_is_windows_console(_raw: PyObjectRef) -> bool {
    false
}

fn live_stdio_encoding_errors(stream_name: &str, default_errors: &str) -> (String, String) {
    let defaults = stdio_encoding_and_errors();
    let Some(sys) = crate::importing::get_interpreter_sys_module() else {
        return (defaults.0, default_errors.to_string());
    };
    let Ok(stream) = crate::baseobjspace::getattr_str(sys, stream_name) else {
        return (defaults.0, default_errors.to_string());
    };
    let text_attr = |name: &str, default: &str| {
        crate::baseobjspace::getattr_str(stream, name)
            .ok()
            .filter(|value| unsafe { is_str(*value) })
            .and_then(|value| {
                crate::baseobjspace::str_utf8_w(value)
                    .ok()
                    .map(str::to_string)
            })
            .unwrap_or_else(|| default.to_string())
    };
    (
        text_attr("encoding", &defaults.0),
        text_attr("errors", default_errors),
    )
}

/// `pylifecycle.c init_sys_streams` builds the standard streams after the
/// import system, so a text codec is reachable by the time they need one.
/// Pyre builds them from `sys` module creation instead, where the codec lookup
/// would import `encodings` and re-enter that creation through its own
/// `import sys`.  They start without an encoder and decoder and pick them up
/// here, once the import system is up.
///
/// The `__std*__` aliases name the streams this built even after user code
/// rebinds `sys.stdout`; a rebound one is not this function's to reconfigure.
pub fn init_stream_codecs() -> Result<(), crate::PyError> {
    let Some(mut sys) = crate::importing::get_interpreter_sys_module() else {
        return Ok(());
    };
    for name in ["__stdout__", "__stderr__", "__stdin__"] {
        if let Ok(stream) =
            pyre_object::with_roots!(sys => crate::baseobjspace::getattr_str(sys, name))
        {
            if unsafe { pyre_object::is_none(stream) } {
                continue;
            }
            pyre_object::with_roots!(sys => crate::module::_io::W_TextIOWrapper::attach_stdio_codec(stream))?;
        }
    }
    Ok(())
}

/// `textio.c CHECK_CLOSED`, the sentence a `TextIOWrapper` method states.
const CLOSED_TEXT_LAYER: &str = "I/O operation on closed file.";

/// `fileio.c self_check_closed`, where a query delegated down the stack ends.
const CLOSED_RAW_LAYER: &str = "I/O operation on closed file";

/// The `CHECK_CLOSED` an instance override owes.
///
/// The type methods these stand in for all start at one, and these reach the
/// descriptor without going through the buffer at all, so a stream the program
/// has closed would keep answering.  `message` is the spelling of the layer
/// the override skipped: a `TextIOWrapper` method ends the sentence with a
/// period, and a query that would have been delegated down to the raw
/// descriptor does not.
///
/// The stream is named through the `__std*__` alias rather than `sys.stdout`
/// and its neighbours, because it is the stream `make_std_stream` built that
/// carries the override, and that is the one the alias keeps naming after user
/// code rebinds the plain name.  A missing or unreadable alias leaves the
/// override answering as it did before, which is what the seam does whenever
/// `sys` cannot state something -- it runs before `sys` exists and after it is
/// torn down.
fn stdio_check_closed(alias: &str, message: &'static str) -> Result<(), crate::PyError> {
    let closed = crate::importing::get_interpreter_sys_module()
        .and_then(|sys| crate::baseobjspace::getattr_str(sys, alias).ok())
        .and_then(|stream| crate::baseobjspace::getattr_str(stream, "closed").ok())
        .is_some_and(|closed| crate::baseobjspace::is_true(closed).unwrap_or(false));
    if closed {
        return Err(crate::PyError::value_error(message));
    }
    Ok(())
}

/// The body every `flush` override shares once its own stream has answered
/// `CHECK_CLOSED`.
fn flush_std_descriptors() -> crate::PyResult {
    // The sandbox path writes unbuffered ll_os_write requests, so there
    // is nothing to flush (and the real fds are the marshalling pipe).
    #[cfg(not(feature = "sandbox"))]
    {
        use std::io::Write;
        let _ = std::io::stdout().flush();
        let _ = std::io::stderr().flush();
    }
    Ok(w_none())
}

fn make_std_stream(name: &'static str, fd: i32) -> PyObjectRef {
    let writable = fd != 0;
    let to_stderr = fd == 2;
    let unbuffered = writable && crate::importing::unbuffered_flag();
    // `app_main.py create_stdio`: retain the FileIO-backed binary layer
    // as TextIOWrapper.buffer. libregrtest workers deliberately write invalid
    // byte sequences through this exact owner.
    //
    // `create_stdio` returns `None` when `_io.open` raises OSError EBADF, so
    // `initstdio` binds `sys.stdout` / `sys.stderr` / `sys.stdin` to `None`
    // for a closed standard fd. A wrapper that still names the fd number
    // would write into whichever file a later `open()` reused.
    //
    // Any other `_io.open` failure keeps the stream with its buffer absent:
    // the instance-override methods reach the descriptor without the buffer,
    // and the sandbox controller mounts no real files to open.
    #[cfg(all(unix, not(feature = "sandbox")))]
    {
        if unsafe { libc::fcntl(fd, libc::F_GETFD) } == -1 {
            return w_none();
        }
    }
    let allow_windows_console = !crate::importing::legacy_windows_stdio_flag();
    let buffer = crate::builtins::builtin_open_stdio(
        &[
            w_int_new(i64::from(fd)),
            w_str_new(if writable { "wb" } else { "rb" }),
            w_int_new(if unbuffered { 0 } else { -1 }),
            w_none(),
            w_none(),
            w_none(),
            w_bool_from(false),
        ],
        allow_windows_console,
    )
    .unwrap_or_else(|_| w_none());
    let _roots = pyre_object::gc_roots::push_roots();
    let buffer = pyre_object::gc_roots::pin_root(buffer);
    let buffer_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    // `app_main.py create_stdio` names the raw descriptor object rather
    // than the wrapper, so `repr(sys.stdin.buffer)` reads `name='<stdin>'`
    // instead of the bare descriptor number.  An unbuffered writable stream
    // has no buffered layer to unwrap and the descriptor object is the buffer
    // itself, which is the `raw = Py_NewRef(buf)` arm of `create_stdio`.
    let raw = {
        let buffer = pyre_object::gc_roots::shadow_stack_get(buffer_slot);
        crate::baseobjspace::getattr_str(buffer, "raw").unwrap_or(buffer)
    };
    let _ = pyre_object::gc_roots::pin_root(raw);
    let raw_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    {
        let w_name = w_str_new(name);
        let raw = pyre_object::gc_roots::shadow_stack_get(raw_slot);
        let _ = crate::baseobjspace::setattr_str(raw, "name", w_name);
    }
    let (encoding, configured_errors) = stdio_encoding_and_errors();
    // `create_stdio` reads the raw layer back after `_io.open` has chosen it:
    // a console stream is utf-8 encoded whatever the configuration named.
    let encoding = if raw_is_windows_console(pyre_object::gc_roots::shadow_stack_get(raw_slot)) {
        "utf-8".to_string()
    } else {
        encoding
    };
    let errors = if to_stderr {
        "backslashreplace"
    } else {
        configured_errors.as_str()
    };
    let stream = crate::module::_io::W_TextIOWrapper::allocate_stdio(
        name,
        pyre_object::gc_roots::shadow_stack_get(buffer_slot),
        &encoding,
        errors,
        // `create_stdio`: `line_buffering=unbuffered or fd == 2 or raw.isatty()`.
        // `FileIO.isatty` (`file_method_isatty`) asks `isatty` of the descriptor.
        unbuffered || to_stderr || crate::importing::host::os::isatty(fd),
        unbuffered,
    );
    let _ = pyre_object::gc_roots::pin_root(stream);
    let stream_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let w_name = w_str_new(name);
    crate::baseobjspace::setdictvalue_native(
        pyre_object::gc_roots::shadow_stack_get(stream_slot),
        "name",
        w_name,
    );
    // `pylifecycle.c init_set_builtins_open`/`init_sys_streams`: stderr uses the
    // `backslashreplace` handler so traceback printing never fails on a lone
    // surrogate; stdout/stdin default to `strict`.
    let w_mode = w_str_new(if writable { "w" } else { "r" });
    crate::baseobjspace::setdictvalue_native(
        pyre_object::gc_roots::shadow_stack_get(stream_slot),
        "mode",
        w_mode,
    );
    crate::baseobjspace::setdictvalue_native(
        pyre_object::gc_roots::shadow_stack_get(stream_slot),
        "closed",
        w_bool_from(false),
    );
    crate::baseobjspace::setdictvalue_native(
        pyre_object::gc_roots::shadow_stack_get(stream_slot),
        "buffer",
        pyre_object::gc_roots::shadow_stack_get(buffer_slot),
    );
    // Instance-stored builtin methods do not get `self` prepended (see
    // pyopcode load_method dispatch), so the first arg may be the string
    // directly. Pick whichever element is a real str.
    fn pick_str(args: &[PyObjectRef]) -> Option<PyObjectRef> {
        args.iter()
            .find(|&&a| !a.is_null() && unsafe { is_str(a) })
            .copied()
    }
    /// The error a `write` call that handed over no `str` owes.
    ///
    /// `W_TextIOWrapper.write_w` rejects a non-`str` argument, and code in the
    /// wild depends on it: `click._compat._is_binary_writer` decides whether a
    /// stream is the binary layer by calling `stream.write(b"")` and seeing
    /// whether it raises, then wraps `sys.stdout` in its own text layer when it
    /// does not — and everything written after that is lost.  Returning a
    /// zero-length count is indistinguishable from a successful empty write.
    ///
    /// The value can only be named positionally, for the same reason
    /// `pick_str` above scans: the receiver may or may not be prepended, and
    /// the text is the last element either way.  A call that passed nothing
    /// leaves only the receiver there, which is why an empty argument list and
    /// a bare receiver both fall to the arity message.
    fn reject_non_str(args: &[PyObjectRef]) -> crate::PyError {
        match args.last() {
            Some(&arg) if !arg.is_null() && unsafe { !is_str(arg) } => {
                crate::PyError::type_error(format!(
                    "unicode argument expected, got '{}'",
                    crate::type_methods::arg_type_name(arg)
                ))
            }
            _ => crate::PyError::type_error("write() takes exactly one argument (0 given)"),
        }
    }
    // `write` here is an instance builtin that encodes straight to the
    // descriptor instead of going through `TextIOWrapper.write`, so it
    // substitutes the line separator itself — on the text, before encoding,
    // because at the byte level a utf-16 stdio encoding would be corrupted.
    // The substitution is the one the stream is configured for, so a
    // `reconfigure(newline=...)` reaches this path too.  It is resolved by
    // name off `sys`, the way `live_stdio_encoding_errors` beside it resolves
    // the encoding and the error handler: the function sits in the stream's
    // instance dictionary rather than being bound to it, so it is handed the
    // text and nothing else.  The count the caller gets back stays the length
    // of what it passed in.
    fn encode_stdio_text(
        w_text: PyObjectRef,
        stream_name: &str,
        encoding: &str,
        errors: &str,
    ) -> Result<Vec<u8>, crate::PyError> {
        let text = unsafe { pyre_object::w_str_get_wtf8(w_text) };
        // The mode is read off the live stream, so the cheap test comes first:
        // text carrying no newline needs no mode and asks `sys` for nothing.
        let newline = text
            .as_bytes()
            .contains(&b'\n')
            .then(|| crate::stdio_newline(stream_name))
            .flatten()
            .filter(|nl| *nl != "\n");
        let Some(newline) = newline else {
            return crate::type_methods::encode_object(w_text, encoding, errors);
        };
        let mut out = rustpython_wtf8::Wtf8Buf::with_capacity(text.len() + 8);
        for point in text.code_points() {
            if point.to_u32() == u32::from(b'\n') {
                out.push_str(newline);
            } else {
                out.push(point);
            }
        }
        // `encode_object` runs a codec, so the substituted text is rooted
        // across it rather than left as a bare temporary.
        let _roots = pyre_object::gc_roots::push_roots();
        let slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_from_wtf8_managed(out));
        crate::type_methods::encode_object(
            pyre_object::gc_roots::shadow_stack_get(slot),
            encoding,
            errors,
        )
    }
    // POSIX EIO is 5 on every host ABI pyre targets. Keep the fallback value
    // target-neutral: wasm's `libc` intentionally exposes no `EIO` symbol,
    // while a real descriptor error still supplies its actual raw errno.
    const EIO_FALLBACK: i32 = 5;
    // Encode through `encode_object` with the stream's error handler so a lone
    // surrogate is routed there (stdout `strict` → UnicodeEncodeError; stderr
    // `backslashreplace` → escaped) instead of panicking in `w_str_get_value`.
    let write_fn = if to_stderr {
        crate::make_builtin_function("write", |args| {
            stdio_check_closed("__stderr__", CLOSED_TEXT_LAYER)?;
            if let Some(mut s_obj) = pick_str(args) {
                let (encoding, _) = pyre_object::with_roots!(s_obj => live_stdio_encoding_errors("stderr", "backslashreplace"));
                let bytes = pyre_object::with_roots!(s_obj => encode_stdio_text(s_obj, "stderr", &encoding, "backslashreplace"))?;
                // An embedder that installed a hook takes the bytes here;
                // otherwise fall through to the descriptor.
                if !crate::stderr_hook_emit(&bytes) {
                    // Under sandbox fd 1 is the marshalling pipe, so a raw write
                    // would corrupt the protocol: route through ll_os_write(2,…)
                    // and let the controller relay it to its own stderr.
                    #[cfg(not(feature = "sandbox"))]
                    {
                        use std::io::Write;
                        std::io::stderr().write_all(&bytes).map_err(|error| {
                            crate::PyError::os_error_with_errno(
                                error.raw_os_error().unwrap_or(EIO_FALLBACK),
                                error.to_string(),
                            )
                        })?;
                    }
                    #[cfg(feature = "sandbox")]
                    crate::host_seam::ops::write(2, &bytes)
                        .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                }
                return Ok(w_int_new(unsafe { w_str_len(s_obj) } as i64));
            }
            Err(reject_non_str(args))
        })
    } else {
        crate::make_builtin_function("write", |args| {
            stdio_check_closed("__stdout__", CLOSED_TEXT_LAYER)?;
            if let Some(mut s_obj) = pick_str(args) {
                let (encoding, errors) = pyre_object::with_roots!(s_obj => live_stdio_encoding_errors("stdout", "strict"));
                let bytes = pyre_object::with_roots!(s_obj => encode_stdio_text(s_obj, "stdout", &encoding, &errors))?;
                // Same seam `print` rides, so an embedder that captures stdout
                // sees `sys.stdout.write` too and the two stay in order.
                if !crate::print_hook_emit_bytes(&bytes) {
                    #[cfg(not(feature = "sandbox"))]
                    {
                        use std::io::Write;
                        std::io::stdout().write_all(&bytes).map_err(|error| {
                            crate::PyError::os_error_with_errno(
                                error.raw_os_error().unwrap_or(EIO_FALLBACK),
                                error.to_string(),
                            )
                        })?;
                    }
                    #[cfg(feature = "sandbox")]
                    crate::host_seam::ops::write(1, &bytes)
                        .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                    crate::host_seam::flush_stdout_when_unbuffered();
                }
                return Ok(w_int_new(unsafe { w_str_len(s_obj) } as i64));
            }
            Err(reject_non_str(args))
        })
    };
    let write_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(write_fn);
    crate::baseobjspace::setdictvalue_native(
        pyre_object::gc_roots::shadow_stack_get(stream_slot),
        "write",
        pyre_object::gc_roots::shadow_stack_get(write_slot),
    );
    // `flush` stands in for `TextIOWrapper.flush`, which starts at
    // `CHECK_CLOSED`, so the descriptor picks the stream to ask -- a bare `fn`
    // pointer carries no captures, the way `fileno` below selects one per
    // descriptor rather than closing over `fd`.
    let flush_fn = match fd {
        0 => crate::make_builtin_function("flush", |_| {
            stdio_check_closed("__stdin__", CLOSED_TEXT_LAYER)?;
            flush_std_descriptors()
        }),
        2 => crate::make_builtin_function("flush", |_| {
            stdio_check_closed("__stderr__", CLOSED_TEXT_LAYER)?;
            flush_std_descriptors()
        }),
        _ => crate::make_builtin_function("flush", |_| {
            stdio_check_closed("__stdout__", CLOSED_TEXT_LAYER)?;
            flush_std_descriptors()
        }),
    };
    let flush_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(flush_fn);
    crate::baseobjspace::setdictvalue_native(
        pyre_object::gc_roots::shadow_stack_get(stream_slot),
        "flush",
        pyre_object::gc_roots::shadow_stack_get(flush_slot),
    );
    // PyPy's W_TextIOWrapper.isatty_w delegates to its live buffer, which in
    // turn delegates to the raw descriptor.  Do not install an instance
    // override here: after forkpty changes fd 0 into the slave terminal, the
    // type method must observe that new descriptor state.
    // `BuiltinCodeFn` is a bare `fn` pointer (no captures), so select a
    // constant-returning function per descriptor rather than closing over `fd`.
    let fileno_fn = match fd {
        0 => crate::make_builtin_function("fileno", |_| {
            stdio_check_closed("__stdin__", CLOSED_RAW_LAYER)?;
            Ok(w_int_new(0))
        }),
        2 => crate::make_builtin_function("fileno", |_| {
            stdio_check_closed("__stderr__", CLOSED_RAW_LAYER)?;
            Ok(w_int_new(2))
        }),
        _ => crate::make_builtin_function("fileno", |_| {
            stdio_check_closed("__stdout__", CLOSED_RAW_LAYER)?;
            Ok(w_int_new(1))
        }),
    };
    let fileno_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(fileno_fn);
    crate::baseobjspace::setdictvalue_native(
        pyre_object::gc_roots::shadow_stack_get(stream_slot),
        "fileno",
        pyre_object::gc_roots::shadow_stack_get(fileno_slot),
    );
    // A buffered layer delegates the query that names its own direction down to
    // the raw descriptor, which refuses once the stream is closed, and answers
    // the opposite one with a constant `False` that closing does not turn into
    // an error.  Only the matching query takes the check.
    let writable_fn = match fd {
        0 => crate::make_builtin_function("writable", |_| Ok(w_bool_from(false))),
        2 => crate::make_builtin_function("writable", |_| {
            stdio_check_closed("__stderr__", CLOSED_RAW_LAYER)?;
            Ok(w_bool_from(true))
        }),
        _ => crate::make_builtin_function("writable", |_| {
            stdio_check_closed("__stdout__", CLOSED_RAW_LAYER)?;
            Ok(w_bool_from(true))
        }),
    };
    let writable_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(writable_fn);
    let readable_fn = match fd {
        0 => crate::make_builtin_function("readable", |_| {
            stdio_check_closed("__stdin__", CLOSED_RAW_LAYER)?;
            Ok(w_bool_from(true))
        }),
        2 => crate::make_builtin_function("readable", |_| Ok(w_bool_from(false))),
        _ => crate::make_builtin_function("readable", |_| Ok(w_bool_from(false))),
    };
    let readable_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(readable_fn);
    crate::baseobjspace::setdictvalue_native(
        pyre_object::gc_roots::shadow_stack_get(stream_slot),
        "writable",
        pyre_object::gc_roots::shadow_stack_get(writable_slot),
    );
    crate::baseobjspace::setdictvalue_native(
        pyre_object::gc_roots::shadow_stack_get(stream_slot),
        "readable",
        pyre_object::gc_roots::shadow_stack_get(readable_slot),
    );
    pyre_object::gc_roots::shadow_stack_get(stream_slot)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::Instruction;

    fn traceback_needed_after(source: &str) -> bool {
        crate::test_hooks::install_hash_hook();
        let code = crate::compile::compile_exec(source).expect("compile failed");
        let mut frame = crate::pyframe::PyFrame::new(code);
        let call_pc = frame
            .getcode()
            .instructions
            .iter()
            .enumerate()
            .find_map(|(pc, unit)| {
                matches!(unit.op.deoptimize(), Instruction::Call { .. }).then_some(pc)
            })
            .expect("source must contain CALL");
        frame.last_instr = call_pc as isize;
        exc_info_result_needs_traceback(&frame)
    }

    #[test]
    fn exc_info_direct_omits_traceback_only_for_pypy_safe_subscripts() {
        for source in [
            "import sys\nresult = sys.exc_info()[0]",
            "import sys\nresult = sys.exc_info()[1]",
            "import sys\nresult = sys.exc_info()[-2]",
            "import sys\nresult = sys.exc_info()[-3]",
            "import sys\nresult = sys.exc_info()[:2]",
            "import sys\nresult = sys.exc_info()[1:2]",
        ] {
            assert!(!traceback_needed_after(source), "{source}");
        }
        for source in [
            "import sys\nresult = sys.exc_info()",
            "import sys\nresult = sys.exc_info()[-1]",
            "import sys\nresult = sys.exc_info()[2]",
            "import sys\nresult = sys.exc_info()[:]",
            "import sys\nresult = sys.exc_info()[1:3]",
        ] {
            assert!(traceback_needed_after(source), "{source}");
        }
    }
}
