//! `_pickle` — interp-level accelerator for the `pickle` module.
//!
//! Port of `pypy/module/_pickle/interp_pickle.py` (`W_Pickler` /
//! `W_Unpickler`). Targets the CPython 3.14 wire format.
//!
//! Scope: protocols 0-5 — atoms (None / bool / int / float / str / bytes),
//! the memo (PUT/GET families), containers (tuple / list / dict with
//! APPENDS / SETITEMS batching + recursion), set / frozenset / bytearray,
//! the reduce protocol (save_reduce / save_global / find_class), the legacy
//! protocol-0/1 text opcodes, persistent_id, `_compat_pickle` fix_imports,
//! and the multi-frame framer. Out-of-band proto-5 buffers are deferred
//! (see the note at the pickler dispatch site).
//!
//! The module exports all nine names `pickle.py` imports — `Pickler`,
//! `Unpickler`, `dump`, `dumps`, `load`, `loads`, `PickleError`,
//! `PicklingError`, `UnpicklingError` — so `from _pickle import (...)`
//! resolves and the accelerated path engages.

use std::sync::atomic::{AtomicPtr, Ordering};

use majit_rlib::rbigint::{RBigInt as BigInt, RBigIntGcRoot, RBigIntSign as Sign};
use pyre_object::PyObjectRef;

use pyre_interpreter::PyError;

mod pickler;
mod unpickler;

pub use pickler::{PicklerMemoProxy, W_Pickler};
pub use unpickler::{UnpicklerMemoProxy, W_Unpickler};

pub(crate) const HIGHEST_PROTOCOL: i64 = 5;
// [3.14-spec] 5, where `interp_pickle.py DEFAULT_PROTOCOL` is 4. The protocol a
// `Pickler` picks when the caller passes none is observable in the stream it
// writes, and the pinned stdlib's `pickle.DEFAULT_PROTOCOL` is 5
// (`lib-python/3/pickle.py`); a real `pypy3` reports 4. No JIT hint reads this
// constant -- `interp_pickle.py` carries one only on `W_Pickler.write_binfloat`.
pub(crate) const DEFAULT_PROTOCOL: i64 = 5;
pub(crate) const FRAME_SIZE_MIN: usize = 4;
pub(crate) const FRAME_SIZE_TARGET: usize = 64 * 1024;

/// `interp_pickle.py Opcodes`.
pub(crate) mod op {
    pub const PROTO: u8 = 0x80;
    pub const FRAME: u8 = 0x95;
    pub const STOP: u8 = b'.';
    pub const NONE: u8 = b'N';
    pub const NEWTRUE: u8 = 0x88;
    pub const NEWFALSE: u8 = 0x89;
    pub const BININT: u8 = b'J';
    pub const BININT1: u8 = b'K';
    pub const BININT2: u8 = b'M';
    pub const LONG1: u8 = 0x8a;
    pub const LONG4: u8 = 0x8b;
    pub const BINFLOAT: u8 = b'G';
    pub const SHORT_BINUNICODE: u8 = 0x8c;
    pub const BINUNICODE: u8 = b'X';
    pub const BINUNICODE8: u8 = 0x8d;
    pub const SHORT_BINBYTES: u8 = b'C';
    pub const BINBYTES: u8 = b'B';
    pub const BINBYTES8: u8 = 0x8e;
    // memo
    pub const MEMOIZE: u8 = 0x94;
    pub const BINPUT: u8 = b'q';
    pub const LONG_BINPUT: u8 = b'r';
    pub const PUT: u8 = b'p';
    pub const GET: u8 = b'g';
    pub const BINGET: u8 = b'h';
    pub const LONG_BINGET: u8 = b'j';
    // stack
    pub const MARK: u8 = b'(';
    pub const POP: u8 = b'0';
    pub const POP_MARK: u8 = b'1';
    // tuple
    pub const EMPTY_TUPLE: u8 = b')';
    pub const TUPLE: u8 = b't';
    pub const TUPLE1: u8 = 0x85;
    pub const TUPLE2: u8 = 0x86;
    pub const TUPLE3: u8 = 0x87;
    // list
    pub const EMPTY_LIST: u8 = b']';
    pub const LIST: u8 = b'l';
    pub const APPEND: u8 = b'a';
    pub const APPENDS: u8 = b'e';
    // dict
    pub const EMPTY_DICT: u8 = b'}';
    pub const DICT: u8 = b'd';
    pub const SETITEM: u8 = b's';
    pub const SETITEMS: u8 = b'u';
    // set / frozenset
    pub const EMPTY_SET: u8 = 0x8f;
    pub const FROZENSET: u8 = 0x91;
    pub const ADDITEMS: u8 = 0x90;
    // bytearray
    pub const BYTEARRAY8: u8 = 0x96;
    // protocol 5 out-of-band buffers (see the deferral note in `pickler.rs`).
    pub const NEXT_BUFFER: u8 = 0x97;
    pub const READONLY_BUFFER: u8 = 0x98;
    // reduce / global
    pub const REDUCE: u8 = b'R';
    pub const BUILD: u8 = b'b';
    pub const GLOBAL: u8 = b'c';
    pub const STACK_GLOBAL: u8 = 0x93;
    pub const NEWOBJ: u8 = 0x81;
    pub const NEWOBJ_EX: u8 = 0x92;
    pub const EXT1: u8 = 0x82;
    pub const EXT2: u8 = 0x83;
    pub const EXT4: u8 = 0x84;
    // protocol 0 / 1 legacy text opcodes
    pub const INT: u8 = b'I';
    pub const LONG: u8 = b'L';
    pub const FLOAT: u8 = b'F';
    pub const STRING: u8 = b'S';
    pub const UNICODE: u8 = b'V';
    pub const BINSTRING: u8 = b'T';
    pub const SHORT_BINSTRING: u8 = b'U';
    pub const INST: u8 = b'i';
    pub const OBJ: u8 = b'o';
    pub const DUP: u8 = b'2';
    // persistent id
    pub const PERSID: u8 = b'P';
    pub const BINPERSID: u8 = b'Q';

    /// `_tuplesize2code` — TUPLE1/2/3 indexed by element count (1..=3).
    pub const TUPLESIZE2CODE: [u8; 4] = [EMPTY_TUPLE, TUPLE1, TUPLE2, TUPLE3];
}

/// `interp_pickle.py W_Pickler._BATCHSIZE`.
pub(crate) const BATCHSIZE: usize = 1000;

// ── shared call helpers ──────────────────────────────────────────────
// `call_function` / `call_method` return PY_NULL on failure and stash the
// error through `call::set_call_error`; surface it as a Rust `Result`.

pub(crate) fn call_fn(callable: PyObjectRef, args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let r = pyre_interpreter::baseobjspace::call_function(callable, args);
    if r.is_null() {
        return Err(pyre_interpreter::call::take_call_error()
            .unwrap_or_else(|| PyError::runtime_error("call failed")));
    }
    Ok(r)
}

fn call_fn_with_kwargs(
    callable: PyObjectRef,
    args: &[PyObjectRef],
    kwargs: &[(&str, PyObjectRef)],
) -> Result<PyObjectRef, PyError> {
    let ec = pyre_interpreter::call::getexecutioncontext();
    if ec.is_null() {
        return Err(PyError::runtime_error(
            "no execution context for _pickle keyword call",
        ));
    }
    let frame = unsafe { (*ec).gettopframe() };
    if frame.is_null() {
        return Err(PyError::runtime_error("no frame for _pickle keyword call"));
    }
    let kwargs = kwargs
        .iter()
        .map(|(name, value)| (rustpython_wtf8::Wtf8Buf::from(*name), *value))
        .collect::<Vec<_>>();
    pyre_interpreter::call::call_with_kwargs(unsafe { &mut *frame }, callable, args, &kwargs)
}

pub(crate) fn call_meth(
    obj: PyObjectRef,
    name: &str,
    args: &[PyObjectRef],
) -> Result<PyObjectRef, PyError> {
    let r = pyre_interpreter::baseobjspace::call_method(obj, name, args);
    if r.is_null() {
        return Err(pyre_interpreter::call::take_call_error()
            .unwrap_or_else(|| PyError::runtime_error("method call failed")));
    }
    Ok(r)
}

/// Build a `PyError` whose raised object is an instance of the named exception
/// class, with `msg` as the single argument. Falls back to a generic ValueError
/// carrying the same text if the class is somehow unavailable.
fn pickle_exc(class_name: &str, msg: rustpython_wtf8::Wtf8Buf) -> PyError {
    let Some(cls) = pyre_interpreter::builtins::lookup_exc_class(class_name) else {
        return PyError::value_error(msg);
    };
    let _roots = pyre_object::gc_roots::push_roots();
    let cls_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(cls);
    let w_value = pyre_object::w_str_from_wtf8_managed(msg);
    PyError::from_type_and_value(pyre_object::gc_roots::shadow_stack_get(cls_slot), w_value)
}

pub(crate) fn unpickling_error(msg: &str) -> PyError {
    pickle_exc("_pickle.UnpicklingError", msg.into())
}

pub(crate) fn pickling_error(msg: impl Into<rustpython_wtf8::Wtf8Buf>) -> PyError {
    pickle_exc("_pickle.PicklingError", msg.into())
}

#[allow(dead_code)]
pub(crate) fn eof_error(msg: &str) -> PyError {
    pickle_exc("EOFError", msg.into())
}

// ── import / dotted attribute resolution (save_global / find_class) ───

/// PyPy `W_Unpickler.find_class`: call builtin `__import__(name)` first, then
/// return the authoritative `sys.modules[name]` entry.
///
/// The import call is required even when the entry already exists:
/// `dunder_import` waits on `__spec__._initializing` via the module lock, so a
/// concurrent unpickler cannot observe a half-executed module.  Its initialized
/// fast path does not reload or rebind modules such as `builtins`.
pub(crate) fn import_module(name: &str) -> Result<PyObjectRef, PyError> {
    pyre_interpreter::importing::dunder_import(
        name,
        pyre_object::w_none(),
        pyre_object::w_none(),
        pyre_object::w_none(),
        0,
        pyre_interpreter::call::getexecutioncontext(),
    )?;
    pyre_interpreter::importing::get_sys_module(name)
        .ok_or_else(|| PyError::value_error(format!("Can't find module {name:?} in sys.modules")))
}

/// The live execution context reached through the thread-owned space state.
fn current_ec() -> Option<*const pyre_interpreter::PyExecutionContext> {
    let ec = pyre_interpreter::call::getexecutioncontext();
    if ec.is_null() { None } else { Some(ec) }
}

/// Resolve a name in the `builtins` module through the execution context's
/// `lookup_builtin` (the `LOAD_GLOBAL` fallback path). This bypasses the
/// module-object `getattr` wrapper, whose hash table does not see builtins
/// installed on the underlying storage.
pub(crate) fn lookup_builtin(name: &str) -> Option<PyObjectRef> {
    current_ec().and_then(|ec| unsafe { (*ec).lookup_builtin(name) })
}

// ── `_compat_pickle` fix_imports tables (space.fromcache(State)) ──────

/// `pypy/module/_pickle/state.py State` — the four `_compat_pickle` mapping
/// dicts, imported once and cached for the process instead of re-`getattr`'d on
/// every `compat_map` call. Held off the GC object graph; the collector reaches
/// the dicts through [`walk_pickle_state_gc`].
struct PickleState {
    w_name_mapping: PyObjectRef,
    w_import_mapping: PyObjectRef,
    w_reverse_name_mapping: PyObjectRef,
    w_reverse_import_mapping: PyObjectRef,
}

/// Published once and never replaced (the `space.fromcache` singleton), so the
/// collector reads the slot without coordinating with a mutator inside
/// `compat_map`.
static PICKLE_STATE: AtomicPtr<PickleState> = AtomicPtr::new(std::ptr::null_mut());

/// The cached `_compat_pickle` tables, built on first use. Returns `None` when
/// `_compat_pickle` or one of its four tables is unavailable, so `compat_map`
/// falls back to the identity mapping. Yields a raw `*const PickleState` (not a
/// shared reference): a reader may hold it across a collection that rewrites the
/// slots through `walk_pickle_state_gc`'s raw pointers, which a `&PickleState`
/// spanning that write would make undefined under Rust's aliasing model.
///
/// `State.startup` (state.py) imports `_compat_pickle` eagerly at space startup
/// and lets a failure propagate. This builds the same tables lazily on the first
/// `compat_map` and degrades to the identity mapping when the module is absent
/// rather than propagating: the eager-startup hook has no pyre counterpart, and
/// the fallback keeps protocol-<3 pickling usable on a minimal stdlib that omits
/// `_compat_pickle` (proto>=3 never reaches `compat_map`).
fn pickle_state() -> Option<*const PickleState> {
    let ptr = PICKLE_STATE.load(Ordering::Acquire);
    if !ptr.is_null() {
        return Some(ptr as *const PickleState);
    }
    // Pin the module and each table across the reads: a plain module `getattr`
    // stays native, but the defensive pin keeps the four dicts rooted even if a
    // lookup ever runs Python and relocates them before publication.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::shadow_stack_len();
    let compat = import_module("_compat_pickle").ok()?;
    let _ = pyre_object::gc_roots::pin_root(compat);
    for attr in [
        "NAME_MAPPING",
        "IMPORT_MAPPING",
        "REVERSE_NAME_MAPPING",
        "REVERSE_IMPORT_MAPPING",
    ] {
        let compat = pyre_object::gc_roots::shadow_stack_get(base);
        let table = pyre_interpreter::baseobjspace::getattr_str(compat, attr).ok()?;
        let _ = pyre_object::gc_roots::pin_root(table);
    }
    // Slots base+1..=base+4 hold the four tables (updated by any relocation
    // during the reads). No GC-allocating step runs between here and publication.
    let created = Box::into_raw(Box::new(PickleState {
        w_name_mapping: pyre_object::gc_roots::shadow_stack_get(base + 1),
        w_import_mapping: pyre_object::gc_roots::shadow_stack_get(base + 2),
        w_reverse_name_mapping: pyre_object::gc_roots::shadow_stack_get(base + 3),
        w_reverse_import_mapping: pyre_object::gc_roots::shadow_stack_get(base + 4),
    }));
    let published = match PICKLE_STATE.compare_exchange(
        std::ptr::null_mut(),
        created,
        Ordering::AcqRel,
        Ordering::Acquire,
    ) {
        Ok(_) => created,
        Err(existing) => {
            drop(unsafe { Box::from_raw(created) });
            existing
        }
    };
    Some(published as *const PickleState)
}

/// Forward the four cached `_compat_pickle` mapping dicts. Called on every
/// collection from `walk_global_prebuilt_roots`, so a minor move updates the
/// off-GC cached pointers. Upstream reaches the same dicts through the space's
/// object graph; pyre holds them off-heap and hands them to the collector.
pub(crate) fn walk_pickle_state_gc(visitor: &mut dyn FnMut(&mut PyObjectRef)) {
    let ptr = PICKLE_STATE.load(Ordering::Acquire);
    if ptr.is_null() {
        return;
    }
    // A collection can run mid-`compat_map`, which reads these same slots through
    // raw pointers (never a live `&PickleState`); write them through raw pointers
    // too so no `&`/`&mut` to the shared state is outstanding across the update.
    unsafe {
        visitor(&mut *std::ptr::addr_of_mut!((*ptr).w_name_mapping));
        visitor(&mut *std::ptr::addr_of_mut!((*ptr).w_import_mapping));
        visitor(&mut *std::ptr::addr_of_mut!((*ptr).w_reverse_name_mapping));
        visitor(&mut *std::ptr::addr_of_mut!(
            (*ptr).w_reverse_import_mapping
        ));
    }
}

/// `_compat_pickle` import/name compatibility mapping applied at protocol
/// < 3. `reverse` picks the py3 → py2 direction used when dumping (the
/// REVERSE_* tables); the forward direction (py2 → py3) is used when
/// loading. Returns the mapped `(module, name)`, unchanged when no entry
/// matches.
pub(crate) fn compat_map(
    module: &str,
    name: &str,
    reverse: bool,
) -> Result<(String, String), pyre_interpreter::PyError> {
    // `pickle_state()` yields a raw `*const PickleState`, not a shared reference:
    // the generic `finditem` below can run Python on a user-replaced mapping and
    // drive a collection, and `walk_pickle_state_gc` then rewrites the cached
    // slots through raw pointers. A `&PickleState` spanning that write would be
    // invalidated under Rust's aliasing model, so each slot is read through a raw
    // pointer here — the same discipline the collection walker uses.
    let Some(state) = pickle_state() else {
        return Ok((module.to_string(), name.to_string()));
    };
    // Build the (module, name) key before touching the cached slots, then read
    // each mapping slot immediately before its probe.
    let mut key_items = pyre_object::gc_roots::RootedItems::new();
    key_items.push(pyre_object::w_str_new_managed(module));
    key_items.push(pyre_object::w_str_new_managed(name));
    let key = pyre_object::tupleobject::w_tuple_new(key_items.take());
    // `finditem` below is generic: a replaced or non-dict `*_MAPPING` routes
    // through `getitem`, which can run Python and drive a collection. Pin the
    // freshly built key so a collection there cannot sweep it mid-lookup — the
    // Rust counterpart of the RPython GC transform auto-rooting `w_key` across
    // `space.finditem`. The tuple is nursery-allocated, so the local below is
    // the word the pin handed back rather than the one it was handed.
    // The cached real-dict tables take the native `w_dict_lookup_checked` path
    // and never collect, so this only guards a user-replaced mapping.
    let _key_root = pyre_object::gc_roots::push_roots();
    let key = pyre_object::gc_roots::pin_root(key);
    // (module, name) entry takes precedence over a bare module remap. Probe
    // through the generic `space.finditem` so a replaced or non-dict `*_MAPPING`
    // and a raising key comparison behave like PyPy's `space.finditem`.
    let w_name_map = unsafe {
        if reverse {
            std::ptr::addr_of!((*state).w_reverse_name_mapping).read()
        } else {
            std::ptr::addr_of!((*state).w_name_mapping).read()
        }
    };
    // `if w_1:` (find_class, interp_pickle.py) is an interp-level presence
    // test: `space.finditem` returns None when absent, and RPython `if w_1:` on a
    // possibly-None `W_Root` checks `is not None`, not Python truthiness. A
    // present entry commits to `w_module_name, w_name = space.listview(w_1)`,
    // which requires exactly two elements (ValueError otherwise) and whose
    // `text_w` conversions raise TypeError on a non-str element — the guards a
    // malformed value would otherwise hit in the downstream `__import__` /
    // `getattr`.
    if let Some(v) = pyre_interpreter::baseobjspace::finditem(w_name_map, key)? {
        let pair = pyre_interpreter::baseobjspace::fixedview(v, 2)?;
        return Ok((
            pyre_interpreter::baseobjspace::text_w(pair[0])?.to_string(),
            pyre_interpreter::baseobjspace::text_w(pair[1])?.to_string(),
        ));
    }
    let w_import_map = unsafe {
        if reverse {
            std::ptr::addr_of!((*state).w_reverse_import_mapping).read()
        } else {
            std::ptr::addr_of!((*state).w_import_mapping).read()
        }
    };
    // `elif w_2:` — presence of a bare-module remap, then `w_module_name = w_2`
    // with `name` unchanged; `text_w` raises TypeError on a non-str remap value.
    if let Some(v) = pyre_interpreter::baseobjspace::finditem_str(w_import_map, module)? {
        return Ok((
            pyre_interpreter::baseobjspace::text_w(v)?.to_string(),
            name.to_string(),
        ));
    }
    Ok((module.to_string(), name.to_string()))
}

/// `mangle_dotted_path`: walk a dotted qualname backwards and mangle each
/// private component against its parent (`PrivateNestedClasses.__Nested`
/// becomes `PrivateNestedClasses._PrivateNestedClasses__Nested`).
pub(crate) fn mangle_dotted_path(name: &str) -> String {
    if !name.contains(".__") {
        return name.to_string();
    }
    let mut parts: Vec<String> = name.split('.').map(str::to_string).collect();
    let n = parts.len();
    for i in (1..n).rev() {
        parts[i] = pyre_interpreter::call::mangle(&parts[i], &parts[i - 1]);
    }
    parts.join(".")
}

/// `interp_pickle.py _getattribute` — walk a dotted `qualname` from `obj`,
/// returning `(resolved, parent)`.
pub(crate) fn getattribute_dotted(
    obj: PyObjectRef,
    qualname: &str,
) -> Result<(PyObjectRef, PyObjectRef), PyError> {
    let mut cur = obj;
    let mut parent = obj;
    for sub in qualname.split('.') {
        if sub == "<locals>" {
            return Err(PyError::attribute_error(format!(
                "Can't get local attribute {qualname:?}"
            )));
        }
        parent = cur;
        cur = pyre_object::with_roots!(parent => pyre_interpreter::baseobjspace::getattr_str(cur, sub))?;
    }
    Ok((cur, parent))
}

/// Wrapped-string variant of `_getattribute`, used when a qualname contains a
/// lone surrogate and therefore has no Rust `&str` view.
pub(crate) fn getattribute_dotted_obj(
    obj: PyObjectRef,
    w_qualname: PyObjectRef,
) -> Result<(PyObjectRef, PyObjectRef), PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[obj, w_qualname]);
    let mut cur_slot = base;
    let qualname_slot = base + 1;
    let w_dot = pyre_object::unicodeobject::intern_str_value(".");
    let _ = pyre_object::gc_roots::pin_root(w_dot);
    let dot_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let w_parts = call_meth(
        pyre_object::gc_roots::shadow_stack_get(qualname_slot),
        "split",
        &[pyre_object::gc_roots::shadow_stack_get(dot_slot)],
    )?;
    let _ = pyre_object::gc_roots::pin_root(w_parts);
    let parts_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let n = unsafe {
        pyre_object::listobject::w_list_len(pyre_object::gc_roots::shadow_stack_get(parts_slot))
    };
    let mut parent_slot = cur_slot;
    for i in 0..n {
        let w_part = unsafe {
            pyre_object::listobject::w_list_getitem(
                pyre_object::gc_roots::shadow_stack_get(parts_slot),
                i as i64,
            )
            .unwrap()
        };
        if unsafe { pyre_object::w_str_get_wtf8(w_part) }.as_bytes() == b"<locals>" {
            let qualname_repr = unsafe {
                pyre_interpreter::display::py_repr_wtf8(pyre_object::gc_roots::shadow_stack_get(
                    qualname_slot,
                ))
            }
            .unwrap_or_else(|_| rustpython_wtf8::Wtf8Buf::from_string("<qualname>".to_string()));
            return Err(PyError::attribute_error(
                pyre_interpreter::display::wtf8_format!(
                    "Can't get local attribute ",
                    qualname_repr
                ),
            ));
        }
        parent_slot = cur_slot;
        let next = pyre_interpreter::baseobjspace::getattr(
            pyre_object::gc_roots::shadow_stack_get(cur_slot),
            w_part,
        )?;
        let _ = pyre_object::gc_roots::pin_root(next);
        cur_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    }
    Ok((
        pyre_object::gc_roots::shadow_stack_get(cur_slot),
        pyre_object::gc_roots::shadow_stack_get(parent_slot),
    ))
}

/// Resolve `module_name.name` to the live object, importing the module
/// first. `builtins` names resolve through the execution context (whose
/// `getattr` wrapper does not see builtins on the underlying storage); other
/// names resolve through `getattr`, component-by-component for a dotted name,
/// so a module-level dynamic attribute (`__getattr__`, PEP 562) resolves as it
/// does for the live object. Returns `Ok(None)` when the name is simply absent
/// (AttributeError) so the dump-time verification can report a precise
/// "not found" error; the load-side `find_class` maps `None` to an error. No
/// `_compat_pickle` remap — callers apply that first.
///
/// `allow_qualname` mirrors `getattribute(obj, name, allow_qualname)`: when set
/// (proto >= 4 STACK_GLOBAL qualnames and the dump-side `whichmodule`
/// verification) the name is walked as a qualname via `getattr` on each
/// component, like `_getattribute`; when clear (the unpickler at protocol < 4)
/// the name is a single `getattr`, so a dotted name resolves as one literal
/// attribute and fails, as in CPython.
#[allow(dead_code)]
pub(crate) fn try_resolve_global(
    module_name: &str,
    name: &str,
    allow_qualname: bool,
) -> Result<Option<PyObjectRef>, PyError> {
    if module_name == "builtins"
        && !name.contains('.')
        && let Some(obj) = lookup_builtin(name)
    {
        return Ok(Some(obj));
    }
    let module = import_module(module_name)?;
    if allow_qualname {
        // protocol >= 4: walk the qualname via `getattr` on each component (a
        // non-dotted name is a single `getattr`), mirroring `_getattribute`.
        return match getattribute_dotted(module, name) {
            Ok((obj, _)) => Ok(Some(obj)),
            Err(e) if e.kind == pyre_interpreter::PyErrorKind::AttributeError => Ok(None),
            Err(e) => Err(e),
        };
    }
    // protocol < 4: a single `getattr`, never a qualname walk.
    match pyre_interpreter::baseobjspace::getattr_str(module, name) {
        Ok(obj) => Ok(Some(obj)),
        Err(e) if e.kind == pyre_interpreter::PyErrorKind::AttributeError => Ok(None),
        Err(e) => Err(e),
    }
}

// ── encode_long / decode_long — two's-complement little-endian ───────
// `interp_pickle.py encode_long` / CPython `pickle.encode_long`.

pub(crate) fn encode_long(big: &BigInt) -> Vec<u8> {
    let sign = big.sign();
    if sign == Sign::NoSign {
        return Vec::new(); // 0 -> b''
    }
    // magnitude, little-endian
    let (_s, digits) = big.to_u32_digits();
    let mut mag: Vec<u8> = Vec::with_capacity(digits.len() * 4);
    for d in &digits {
        mag.extend_from_slice(&d.to_le_bytes());
    }
    while mag.len() > 1 && *mag.last().unwrap() == 0 {
        mag.pop();
    }
    // reserve a byte for the sign bit when the top magnitude bit is set
    if mag.last().is_none_or(|&b| b & 0x80 != 0) {
        mag.push(0x00);
    }
    if sign == Sign::Plus {
        return mag;
    }
    // negative: two's complement (invert + 1)
    let mut carry: u16 = 1;
    for b in mag.iter_mut() {
        let v = (!*b as u16) + carry;
        *b = (v & 0xff) as u8;
        carry = v >> 8;
    }
    // trim a redundant 0xff sign byte (encode_long minimal form)
    while mag.len() > 1 && mag[mag.len() - 1] == 0xff && (mag[mag.len() - 2] & 0x80) != 0 {
        mag.pop();
    }
    mag
}

pub(crate) fn decode_long(data: &[u8]) -> PyObjectRef {
    if data.is_empty() {
        return pyre_object::w_int_new(0);
    }
    let negative = data[data.len() - 1] & 0x80 != 0;
    let unsigned = RBigIntGcRoot::new(BigInt::from_bytes_le(Sign::Plus, data));
    let value = if negative {
        // subtract 2**(8*len): little-endian bytes are `len` zeros then 0x01.
        // `from_bytes_le` collects, so the subtrahend is built before
        // `unsigned` is read.
        let mut pow = vec![0u8; data.len()];
        pow.push(1);
        let modulus = BigInt::from_bytes_le(Sign::Plus, &pow);
        unsigned.sub(&modulus)
    } else {
        unsigned.translated_alias()
    };
    int_from_bigint(value)
}

/// Demote to a small int when it fits, mirroring the int/long unification.
pub(crate) fn int_from_bigint(value: BigInt) -> PyObjectRef {
    match i64::try_from(&value) {
        Ok(v) => pyre_object::w_int_new(v),
        Err(_) => pyre_object::w_long_new(value),
    }
}

/// Parse a decimal integer literal (INT / LONG text opcodes) into an int.
pub(crate) fn parse_int_text(s: &str) -> Result<PyObjectRef, PyError> {
    // interp_pickle.py:2201 calls the ordinary `int` constructor.  Reuse its
    // NumberStringParser consumer so the digit limit, MemoryError edge, and
    // literal diagnostics do not diverge for protocol-0 INT/LONG opcodes.
    pyre_interpreter::builtins::parse_int_from_str(pyre_object::w_str_new_managed(s), s, 10)
}

/// `calc_binsize`: unsigned little-endian size used by FRAME / BIN*8.
/// `None` when the value exceeds `isize::MAX` (`PY_SSIZE_T_MAX`).
pub(crate) fn calc_binsize(data: &[u8]) -> Option<usize> {
    let mut x: u64 = 0;
    for (i, &b) in data.iter().enumerate() {
        if i >= std::mem::size_of::<usize>() {
            if b != 0 {
                return None;
            }
            continue;
        }
        x |= (b as u64) << (8 * i);
    }
    if x > isize::MAX as u64 {
        None
    } else {
        Some(x as usize)
    }
}

pub(crate) fn str_from_utf8(data: &[u8]) -> Result<PyObjectRef, PyError> {
    // BINUNICODE is encoded with Python's UTF-8 `surrogatepass`: lone
    // surrogates are valid pickle payloads and are represented internally as
    // WTF-8, while malformed byte sequences must still be rejected.  The
    // rejection is `rutf8::wtf8_from_bytes`'s and not `Wtf8Buf::from_bytes`'s,
    // which admits sequences that hold no code point at all, and it is
    // reported at the position that validator stopped at.
    let s = pyre_object::rutf8::wtf8_from_bytes(data, true).map_err(|error| {
        pyre_interpreter::typedef::utf8_decode_error_from(data, error.pos as usize)
    })?;
    Ok(pyre_object::unicodeobject::w_str_from_wtf8_managed(
        s.to_owned(),
    ))
}

pyre_interpreter::py_module! {
    "_pickle",
    interpleveldefs: {
        "Pickler" => pickler::type_object(),
        "Unpickler" => unpickler::type_object(),
        // Shared singleton with `__pypy__.PickleBuffer`; `pickle.py` does
        // `from _pickle import PickleBuffer` to set `_HAVE_PICKLE_BUFFER`.
        "PickleBuffer" =>
            pyre_interpreter::module::__pypy__::interp_buffer::picklebuffer_type_object(),
    },
    exceptions: {
        "PickleError" => pyre_interpreter::builtins::lookup_exc_class("Exception")
            .expect("Exception must be installed before _pickle init"),
        "PicklingError" => pyre_interpreter::builtins::lookup_exc_class("_pickle.PickleError")
            .expect("_pickle.PickleError registered just above"),
        "UnpicklingError" => pyre_interpreter::builtins::lookup_exc_class("_pickle.PickleError")
            .expect("_pickle.PickleError registered just above"),
    },
    inline_functions: {
        // `pickle.dump` — write a pickle of `obj` to `file`.
        fn dump(
            obj: PyObjectRef,
            file: PyObjectRef,
            #[default(pyre_object::w_none())] protocol: PyObjectRef,
            #[kwonly]
            #[default(pyre_object::boolobject::w_bool_from(true))]
            fix_imports: PyObjectRef,
            #[kwonly]
            #[default(pyre_object::w_none())]
            buffer_callback: PyObjectRef,
        ) -> Result<PyObjectRef, PyError> {
            // Pin every input before any Python-visible work: `normalize_protocol`
            // (`__index__`) and `is_true` (`__bool__`) run user code and can
            // relocate objects under the moving GC.
            let _roots = pyre_object::gc_roots::push_roots();
            let _ = pyre_object::gc_roots::pin_root(obj);
            let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let _ = pyre_object::gc_roots::pin_root(buffer_callback);
            let bc_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let _ = pyre_object::gc_roots::pin_root(file);
            let file_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let _ = pyre_object::gc_roots::pin_root(protocol);
            let proto_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let _ = pyre_object::gc_roots::pin_root(fix_imports);
            let fix_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let proto = pickler::normalize_protocol(pyre_object::gc_roots::shadow_stack_get(
                proto_slot,
            ))?;
            pickler::check_buffer_callback(pyre_object::gc_roots::shadow_stack_get(bc_slot), proto)?;
            let fix =
                pyre_interpreter::baseobjspace::is_true(pyre_object::gc_roots::shadow_stack_get(fix_slot))?;
            // The dump-time `dispatch_table` (no per-pickler one here) — its
            // `copyreg` import can collect; pin it before the memo allocation.
            let dispatch_table = pickler::copyreg_dispatch_table();
            let _ = pyre_object::gc_roots::pin_root(dispatch_table);
            let dt_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let w_memo = pyre_object::listobject::w_list_new_empty();
            // Pass `file` so `pickle_core` streams the frames to it directly.
            pickler::pickle_core(
                pyre_object::gc_roots::shadow_stack_get(obj_slot),
                pyre_object::gc_roots::shadow_stack_get(file_slot),
                proto,
                proto >= 1,
                proto >= 4,
                fix,
                pyre_object::PY_NULL,
                pyre_object::gc_roots::shadow_stack_get(bc_slot),
                w_memo,
                false,
                pyre_object::gc_roots::shadow_stack_get(dt_slot),
                pyre_object::PY_NULL,
            )?;
            Ok(pyre_object::w_none())
        }

        // `pickle.dumps` — return a pickle of `obj` as `bytes`.
        fn dumps(
            obj: PyObjectRef,
            #[default(pyre_object::w_none())] protocol: PyObjectRef,
            #[kwonly]
            #[default(pyre_object::boolobject::w_bool_from(true))]
            fix_imports: PyObjectRef,
            #[kwonly]
            #[default(pyre_object::w_none())]
            buffer_callback: PyObjectRef,
        ) -> Result<PyObjectRef, PyError> {
            // Pin every input before any Python-visible work (`__index__` /
            // `__bool__` / `copyreg` import / memo alloc can relocate objects).
            let _roots = pyre_object::gc_roots::push_roots();
            let _ = pyre_object::gc_roots::pin_root(obj);
            let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let _ = pyre_object::gc_roots::pin_root(buffer_callback);
            let bc_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let _ = pyre_object::gc_roots::pin_root(protocol);
            let proto_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let _ = pyre_object::gc_roots::pin_root(fix_imports);
            let fix_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let proto = pickler::normalize_protocol(pyre_object::gc_roots::shadow_stack_get(
                proto_slot,
            ))?;
            pickler::check_buffer_callback(pyre_object::gc_roots::shadow_stack_get(bc_slot), proto)?;
            let fix =
                pyre_interpreter::baseobjspace::is_true(pyre_object::gc_roots::shadow_stack_get(fix_slot))?;
            let dispatch_table = pickler::copyreg_dispatch_table();
            let _ = pyre_object::gc_roots::pin_root(dispatch_table);
            let dt_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let w_memo = pyre_object::listobject::w_list_new_empty();
            // No file: `pickle_core` accumulates and returns the pickle bytes.
            pickler::pickle_core(
                pyre_object::gc_roots::shadow_stack_get(obj_slot),
                pyre_object::PY_NULL,
                proto,
                proto >= 1,
                proto >= 4,
                fix,
                pyre_object::PY_NULL,
                pyre_object::gc_roots::shadow_stack_get(bc_slot),
                w_memo,
                false,
                pyre_object::gc_roots::shadow_stack_get(dt_slot),
                pyre_object::PY_NULL,
            )
        }

        // `pickle.load` — read a pickle from `file`.
        fn load(
            file: PyObjectRef,
            #[kwonly]
            #[default(pyre_object::boolobject::w_bool_from(true))]
            fix_imports: PyObjectRef,
            #[kwonly] #[default(pyre_object::w_none())] encoding: PyObjectRef,
            #[kwonly] #[default(pyre_object::w_none())] errors: PyObjectRef,
            #[kwonly] #[default(pyre_object::w_none())] buffers: PyObjectRef,
        ) -> Result<PyObjectRef, PyError> {
            let unpickler = call_fn_with_kwargs(
                unpickler::type_object(),
                &[file],
                &[
                    ("fix_imports", fix_imports),
                    ("encoding", encoding),
                    ("errors", errors),
                    ("buffers", buffers),
                ],
            )?;
            call_meth(unpickler, "load", &[])
        }

        // `pickle.loads` — read a pickle from a `bytes` object.
        fn loads(
            data: PyObjectRef,
            #[kwonly]
            #[default(pyre_object::boolobject::w_bool_from(true))]
            fix_imports: PyObjectRef,
            #[kwonly] #[default(pyre_object::w_none())] encoding: PyObjectRef,
            #[kwonly] #[default(pyre_object::w_none())] errors: PyObjectRef,
            #[kwonly] #[default(pyre_object::w_none())] buffers: PyObjectRef,
        ) -> Result<PyObjectRef, PyError> {
            // Pin every argument that outlives the `BytesIO` construction;
            // a minor collection there can relocate them.
            let _roots = pyre_object::gc_roots::push_roots();
            let base = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(data);
            let _ = pyre_object::gc_roots::pin_root(fix_imports);
            let _ = pyre_object::gc_roots::pin_root(encoding);
            let _ = pyre_object::gc_roots::pin_root(errors);
            let _ = pyre_object::gc_roots::pin_root(buffers);
            let io = import_module("io")?;
            let bytesio_cls = pyre_interpreter::baseobjspace::getattr_str(io, "BytesIO")?;
            let file = call_fn(
                bytesio_cls,
                &[pyre_object::gc_roots::shadow_stack_get(base)],
            )?;
            let unpickler = call_fn_with_kwargs(
                unpickler::type_object(),
                &[file],
                &[
                    (
                        "fix_imports",
                        pyre_object::gc_roots::shadow_stack_get(base + 1),
                    ),
                    (
                        "encoding",
                        pyre_object::gc_roots::shadow_stack_get(base + 2),
                    ),
                    (
                        "errors",
                        pyre_object::gc_roots::shadow_stack_get(base + 3),
                    ),
                    (
                        "buffers",
                        pyre_object::gc_roots::shadow_stack_get(base + 4),
                    ),
                ],
            )?;
            call_meth(unpickler, "load", &[])
        }
    },
    extra_init: |ns| {
        // `dump`/`dumps` `fix_imports=True, buffer_callback=None` and
        // `load`/`loads` `fix_imports=True, encoding/errors/buffers=None`.
        // `_match_signature` requires `Function.w_kw_defs` for those
        // declared defaults (`interp2app._getdefaults`).
        let ns_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(ns);
        for name in ["dump", "dumps"] {
            if let Some(mut function) = unsafe {
                pyre_object::w_dict_getitem_str(
                    pyre_object::gc_roots::shadow_stack_get(ns_slot),
                    name,
                )
            } {
                function = pyre_interpreter::function_set_kwonly_default_pairs(
                    function,
                    &[
                        ("fix_imports", pyre_object::boolobject::w_bool_from(true)),
                        ("buffer_callback", pyre_object::w_none()),
                    ],
                );
                pyre_interpreter::__pyre_put_new!(ns_slot, name, function);
            }
        }
        for name in ["load", "loads"] {
            if let Some(mut function) = unsafe {
                pyre_object::w_dict_getitem_str(
                    pyre_object::gc_roots::shadow_stack_get(ns_slot),
                    name,
                )
            } {
                function = pyre_interpreter::function_set_kwonly_default_pairs(
                    function,
                    &[
                        ("fix_imports", pyre_object::boolobject::w_bool_from(true)),
                        ("encoding", pyre_object::w_none()),
                        ("errors", pyre_object::w_none()),
                        ("buffers", pyre_object::w_none()),
                    ],
                );
                pyre_interpreter::__pyre_put_new!(ns_slot, name, function);
            }
        }
    },
}

/// The GC types this module owns, in `build_gc` registration order.
pub(crate) fn gc_types(types: &mut Vec<pyre_interpreter::importing::ModuleGcType>) {
    use pyre_interpreter::importing::{ModuleGcLayout, ModuleGcType};
    use pyre_object::lltype::PyreClassPyTypeOf;
    // W_Pickler / W_Unpickler carry inline `PyObjectRef` fields — the pickler's
    // output file; the unpickler's read/readline callables, result stack and
    // active frame — that the collector must walk. A subclass is `typedef.py`
    // `_getusercls`: the user descriptor follows its base and reuses the base
    // destructor (`None`). Mapdict storage is walked by the user-layout trace.
    for descriptor in [
        <W_Pickler as PyreClassPyTypeOf>::DESCRIPTOR,
        &pickler::W_PICKLER_USER_PYRE_CLASS_DESCRIPTOR,
        <W_Unpickler as PyreClassPyTypeOf>::DESCRIPTOR,
        &unpickler::W_UNPICKLER_USER_PYRE_CLASS_DESCRIPTOR,
        // Each memo proxy holds one traced `PyObjectRef` back-reference to its
        // owning pickler/unpickler.
        <PicklerMemoProxy as PyreClassPyTypeOf>::DESCRIPTOR,
        <UnpicklerMemoProxy as PyreClassPyTypeOf>::DESCRIPTOR,
    ] {
        types.push(ModuleGcType {
            descriptor,
            layout: ModuleGcLayout::PyreClass {
                memory_pressure_offset: None,
            },
            destructor: None,
        });
    }
}
