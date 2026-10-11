//! `_pickle.Pickler` — `interp_pickle.py W_Pickler` (atom + container subset).

use std::collections::HashMap;

use majit_rlib::rbigint::RBigInt as BigInt;
use pyre_object::PyObjectRef;

use pyre_interpreter::PyError;

use super::{
    BATCHSIZE, DEFAULT_PROTOCOL, FRAME_SIZE_MIN, FRAME_SIZE_TARGET, HIGHEST_PROTOCOL, call_fn,
    call_meth, encode_long, getattribute_dotted, import_module, op, pickling_error,
};

/// Clear `running` on the relocated live pickler, not an interior pointer into
/// the pre-collection copy.  The owning shadow-stack scope outlives this guard.
struct RunningGuard {
    self_slot: usize,
}

impl Drop for RunningGuard {
    fn drop(&mut self) {
        let current = pyre_object::gc_roots::shadow_stack_get(self.self_slot);
        unsafe { (*(current as *mut W_Pickler)).running = false };
    }
}

#[inline]
fn cur_pickler(slot: usize) -> &'static mut W_Pickler {
    unsafe { &mut *(pyre_object::gc_roots::shadow_stack_get(slot) as *mut W_Pickler) }
}

/// A reference kept for the duration of `pickle_core_impl`. Movable values
/// live in a shadow-stack slot and are re-read on every access so relocation
/// is observed; `PY_NULL` and immovable singletons can be held verbatim.
struct PinnedRef {
    slot: Option<usize>,
    immovable: PyObjectRef,
}

impl PinnedRef {
    fn new(value: PyObjectRef, movable: bool) -> Self {
        let slot = if movable {
            let _ = pyre_object::gc_roots::pin_root(value);
            Some(pyre_object::gc_roots::shadow_stack_len() - 1)
        } else {
            None
        };
        Self {
            slot,
            immovable: value,
        }
    }

    fn get(&self) -> PyObjectRef {
        self.slot.map_or(self.immovable, |slot| {
            pyre_object::gc_roots::shadow_stack_get(slot)
        })
    }
}

/// Old-to-young / incremental-mark write barrier for a store into the
/// pickler's own GC-pointer fields (`w_memo` / `w_dispatch_table` /
/// `w_pers_func` / …). The GC transform emits `ll_writebarrier` after every
/// such store (`rpython/memory/gctransform/framework.py:1423`); the
/// hand-written port must call it, or an incremental major collection sweeps
/// a freshly stored (white) child that the already-marked pickler still
/// references — leaving a dangling field the next mark trips over.
#[inline]
fn pickler_write_barrier(obj: PyObjectRef) {
    pyre_object::gc_hook::try_gc_write_barrier(obj as pyre_object::gc_hook::GCREF);
}

// CPython 3.14 Modules/_pickle.c:pickle_exec CREATE_TYPEs pickler_type_spec;
// the spec is an immutable heap type.
#[pyre_interpreter::pyre_class("_pickle.Pickler", cpython_heaptype, user_layout)]
pub struct W_Pickler {
    /// Output file (has a `write` method).
    w_file: PyObjectRef,
    /// Bound `file.write`, resolved once by `__init__` and reused for every
    /// write. `interp_pickle.py` resolves it at `:555-560` only to validate,
    /// discards the result, and re-resolves per write in `_Framer.file_write`
    /// (`:353`); measured on 3.14.5, rebinding `file.write` after construction
    /// is *not* observed by a later `dump()`, and `pickle.py` captures the
    /// callable the same way. Reusing the resolution is what matches.
    w_write: PyObjectRef,
    proto: i64,
    bin: bool,
    framing: bool,
    /// Apply the `_compat_pickle` py3→py2 name remap at protocol < 3.
    fix_imports: bool,
    /// `buffer_callback` for proto-5 out-of-band buffers, or `None`.
    buffer_callback: PyObjectRef,
    /// Memo of saved objects — a Python `list` (GC-walked) persisted across
    /// `dump` calls until `clear_memo`, position = memo index.
    w_memo: PyObjectRef,
    /// `fast` mode — when non-zero, memoization is skipped. Stored as the
    /// assigned integer (`space.int_w`) so `Pickler.fast` round-trips.
    fast: i64,
    /// User-set `dispatch_table` mapping, or `PY_NULL` when unset; the dump
    /// path falls back to `copyreg.dispatch_table` when unset.
    w_dispatch_table: PyObjectRef,
    /// `persistent_id` callable set on the instance, or `PY_NULL` when unset
    /// (a subclass may instead override the `persistent_id` method).
    w_pers_func: PyObjectRef,
    /// CPython `_pickle.c:648` `running`: guards `__init__` and `dump`
    /// against re-entering the same pickler from a reducer or stream method.
    running: bool,
}

/// Per-`dump` pickling context.  The identity memo maps an already-saved
/// object to its memo index.  pyre's incminimark nursery relocates live
/// objects, so the memo cannot key on a raw address: the memoized objects
/// live in a pinned Python `list` (`memo_slot`) which the GC walks, so the
/// stored references follow every move, and `index` maps the move-stable
/// `gc_identity_hash` to the list positions sharing that hash, resolved by
/// pointer identity against a freshly-read list element.  The memo index
/// (the PUT/GET argument) is the object's position in that list.
///
/// That list must carry the Object strategy: an unboxing strategy keeps the
/// payload rather than the reference and wraps a fresh object per read, which
/// the pointer-identity resolution below reads as a different object.  A memo
/// whose entries were all `bytes` then missed on every lookup and wrote the
/// object again where CPython emits a GET.
struct PickleCtx {
    proto: i64,
    bin: bool,
    /// Apply the `_compat_pickle` py3→py2 name remap at protocol < 3.
    fix_imports: bool,
    /// Shadow-stack slot of the memo `list`; re-read on every access so a
    /// relocation of the list itself is observed.
    memo_slot: usize,
    /// `gc_identity_hash(obj)` → memo indices sharing that hash.
    index: HashMap<usize, Vec<usize>>,
    /// `str_memo` / `bytes_memo` (interp_pickle.py): value-keyed memos for the
    /// exact `str` and `bytes` types, consulted before the identity memo.  An
    /// unboxing list strategy stores the erased rpython string and wraps a
    /// fresh object per read, so repeated references to one logical value
    /// reach the identity memo as different objects and would each be written
    /// where the second belongs as a GET.
    str_memo: HashMap<Vec<u8>, usize>,
    bytes_memo: HashMap<Vec<u8>, usize>,
    /// `persistent_id` callable resolved off the pickler (subclass override
    /// or set attribute), or `PY_NULL` when not defined.
    pers_func: PinnedRef,
    /// `buffer_callback` for proto-5 out-of-band buffers, or `None`/`PY_NULL`.
    buffer_callback: PinnedRef,
    /// `fast` mode — when set, memoization is skipped (no PUT/GET); a
    /// shallow cyclic-object guard (`fast_nesting` / `fast_memo`) still fires
    /// past `FAST_NESTING_LIMIT`.
    fast: bool,
    /// Nested-container depth, for the fast-mode cyclic guard.
    fast_nesting: i64,
    /// `gc_identity_hash` → shadow-stack slots of the containers on the active
    /// save path sharing that hash, populated only past `FAST_NESTING_LIMIT`.
    /// A repeat — resolved by pointer identity against the pinned ancestors,
    /// like `memo_get`, since a shared hash is not move-stable-unique — means a
    /// cycle, so fast mode raises `ValueError` instead of recursing into a
    /// stack overflow.
    fast_memo: HashMap<usize, Vec<usize>>,
    /// Effective `dispatch_table` (the pickler's, else `copyreg.dispatch_table`)
    /// consulted by `type` for the reduce of an otherwise-unhandled object;
    /// `None`/`PY_NULL` when unavailable.
    dispatch_table: PinnedRef,
    /// `reducer_override` callable (a subclass hook) consulted for every
    /// object, or `PY_NULL` when not defined.
    reducer_override: PinnedRef,
}

impl PickleCtx {
    /// The memo `list`, re-read from its pinned slot (it may have moved).
    fn memo_list(&self) -> PyObjectRef {
        pyre_object::gc_roots::shadow_stack_get(self.memo_slot)
    }

    /// The `str_memo` / `bytes_memo` key for an exact `str` or `bytes`:
    /// `space.utf8_w(w_obj)` and `space.bytes_w(w_obj)` respectively.
    fn value_memo_key(w_obj: PyObjectRef) -> Option<(bool, Vec<u8>)> {
        unsafe {
            if pyre_object::is_exact_type(w_obj, &pyre_object::STR_TYPE) {
                return Some((
                    true,
                    pyre_object::unicodeobject::w_str_get_wtf8(w_obj)
                        .as_bytes()
                        .to_vec(),
                ));
            }
            if pyre_object::is_exact_type(w_obj, &pyre_object::bytesobject::BYTES_TYPE) {
                return Some((
                    false,
                    pyre_object::bytesobject::w_bytes_data(w_obj).to_vec(),
                ));
            }
        }
        None
    }

    /// `interp_pickle.py W_Pickler.save`'s value-based memo lookup, which
    /// precedes the identity memo.
    fn value_memo_get(&self, w_obj: PyObjectRef) -> Option<usize> {
        let (is_str, key) = Self::value_memo_key(w_obj)?;
        let memo = if is_str {
            &self.str_memo
        } else {
            &self.bytes_memo
        };
        memo.get(&key).copied()
    }

    fn memo_get(&self, w_obj: PyObjectRef) -> Option<usize> {
        let h = pyre_object::gc_hook::gc_identity_hash(w_obj as usize);
        let list = self.memo_list();
        for &idx in self.index.get(&h)? {
            let memoized =
                unsafe { pyre_object::listobject::w_list_getitem(list, idx as i64) }.unwrap();
            if memoized == w_obj {
                return Some(idx);
            }
        }
        None
    }
}

/// `pickle.py:_T` — the display name used by the pickler's PEP 678 notes.
///
/// Keep this as an attribute-based lookup rather than using the Rust type
/// descriptor's short name: local classes need their Python `__module__` and
/// `__qualname__` in notes such as
/// `when serializing test.pickletester.REX object`.
fn pickle_type_name(w_obj: PyObjectRef) -> Result<String, PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let _ = pyre_object::gc_roots::pin_root(w_obj);
    let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let w_type =
        pyre_interpreter::typedef::r#type(pyre_object::gc_roots::shadow_stack_get(obj_slot))
            .ok_or_else(|| PyError::type_error("object has no type"))?
            .as_ptr();
    let _ = pyre_object::gc_roots::pin_root(w_type);
    let type_slot = pyre_object::gc_roots::shadow_stack_len() - 1;

    let w_module = pyre_interpreter::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(type_slot),
        "__module__",
    )?;
    let module = if unsafe { pyre_object::is_none(w_module) } {
        None
    } else {
        Some(pyre_interpreter::baseobjspace::text_w(w_module)?.to_string())
    };
    let w_qualname = pyre_interpreter::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(type_slot),
        "__qualname__",
    )?;
    let qualname = pyre_interpreter::baseobjspace::text_w(w_qualname)?.to_string();
    match module.as_deref() {
        None | Some("builtins") | Some("__main__") => Ok(qualname),
        Some(module) => Ok(format!("{module}.{qualname}")),
    }
}

/// Attach one of pickle.py's contextual PEP 678 notes without losing the
/// original exception object. The carrier is pinned across the type-name
/// lookup and `add_note` call; `expand_pop_roots` reloads it before the
/// instance is written back.
fn add_pickle_object_note(
    err: PyError,
    w_obj: PyObjectRef,
    role: &rustpython_wtf8::Wtf8,
) -> PyError {
    let _roots = pyre_object::gc_roots::push_roots();
    let mut err = err;
    let err_slot = err.pin(&_roots);
    let _ = pyre_object::gc_roots::pin_root(w_obj);
    let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let w_exc = err.to_exc_object();
    let _ = pyre_object::gc_roots::pin_root(w_exc);
    let exc_slot = pyre_object::gc_roots::shadow_stack_len() - 1;

    if let Ok(type_name) = pickle_type_name(pyre_object::gc_roots::shadow_stack_get(obj_slot)) {
        let w_note = pyre_object::w_str_from_wtf8_managed(pyre_interpreter::display::wtf8_format!(
            format!("when serializing {type_name} "),
            role
        ));
        let _ = pyre_object::gc_roots::pin_root(w_note);
        let note_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        if let Ok(w_add_note) = pyre_interpreter::baseobjspace::getattr_str(
            pyre_object::gc_roots::shadow_stack_get(exc_slot),
            "add_note",
        ) {
            let _ = pyre_object::gc_roots::pin_root(w_add_note);
            let add_note_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            // The original exception remains authoritative. A pathological
            // overridden `add_note` must not turn serializer context handling
            // into a second, unrelated exception in the native pickler.
            let _ = call_fn(
                pyre_object::gc_roots::shadow_stack_get(add_note_slot),
                &[pyre_object::gc_roots::shadow_stack_get(note_slot)],
            );
        }
    }
    err.reload(&_roots, err_slot);
    err.set_exc_object(pyre_object::gc_roots::shadow_stack_get(exc_slot));
    err
}

fn add_reduce_note(
    err: PyError,
    w_obj_slot: Option<usize>,
    role: &rustpython_wtf8::Wtf8,
) -> PyError {
    match w_obj_slot {
        Some(slot) => {
            add_pickle_object_note(err, pyre_object::gc_roots::shadow_stack_get(slot), role)
        }
        None => err,
    }
}

/// `interp_pickle.py _Framer` — emits output as frames and streams them to the
/// file's `write` as they are produced. Bytes are appended to the active frame;
/// once a frame reaches `FRAME_SIZE_TARGET` it is committed (FRAME opcode +
/// 8-byte little-endian length + body when the body is at least
/// `FRAME_SIZE_MIN` bytes) into `pending` and `pending` is flushed to the file.
/// Large payloads bypass the frame and are written directly
/// (`write_large_bytes`). When framing is off (protocol < 4) the active frame
/// is `None` and bytes accumulate in `pending` until the final flush.
///
/// `push` / `extend_from_slice` mirror the `Vec<u8>` methods the save routines
/// call. They never touch Python, so the deep save tree stays GC-inert; the
/// only nursery-relocation points are the `commit_frame` at each `save`
/// boundary, `write_large_bytes`, and the end flush — and those callers pin the
/// objects they still need across the `file.write` (arbitrary Python).
///
/// `file_slot` is the shadow-stack slot of the destination file and
/// `write_slot` is the cached bound `file.write` (both pinned by the caller for
/// the whole dump). When `file_slot` is `None` (the `dumps` path) nothing is
/// flushed: `pending` accumulates the entire pickle and the caller takes it.
struct Framer {
    current_frame: Option<Vec<u8>>,
    pending: Vec<u8>,
    file_slot: Option<usize>,
    write_slot: Option<usize>,
}

impl Framer {
    fn new(file_slot: Option<usize>, write_slot: Option<usize>) -> Self {
        Framer {
            current_frame: None,
            pending: Vec::new(),
            file_slot,
            write_slot,
        }
    }

    /// `_Framer.write` (single byte).
    fn push(&mut self, byte: u8) {
        match &mut self.current_frame {
            Some(f) => f.push(byte),
            None => self.pending.push(byte),
        }
    }

    /// `_Framer.write` (slice).
    fn extend_from_slice(&mut self, data: &[u8]) {
        match &mut self.current_frame {
            Some(f) => f.extend_from_slice(data),
            None => self.pending.extend_from_slice(data),
        }
    }

    /// `_Framer.start_framing`.
    fn start_framing(&mut self) {
        self.current_frame = Some(Vec::new());
    }

    /// Write the buffered `pending` bytes to the file in a single `write`. A
    /// no-op when there is no file (`dumps`) or nothing is pending. The file is
    /// re-read from its pin because `write` (arbitrary Python) can relocate it.
    fn flush(&mut self) -> Result<(), PyError> {
        if let Some(slot) = self.file_slot
            && !self.pending.is_empty()
        {
            let w_bytes = pyre_object::w_bytes_from_bytes(&self.pending);
            // Pin the freshly-built bytes and re-read both it and the
            // cached callable from their roots before arbitrary Python.
            let _roots = pyre_object::gc_roots::push_roots();
            let _ = pyre_object::gc_roots::pin_root(w_bytes);
            let bytes_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            if let Some(write_slot) = self.write_slot {
                call_fn(
                    pyre_object::gc_roots::shadow_stack_get(write_slot),
                    &[pyre_object::gc_roots::shadow_stack_get(bytes_slot)],
                )?;
            } else {
                let w_file = pyre_object::gc_roots::shadow_stack_get(slot);
                call_meth(
                    w_file,
                    "write",
                    &[pyre_object::gc_roots::shadow_stack_get(bytes_slot)],
                )?;
            }
            self.pending.clear();
        }
        Ok(())
    }

    /// `_Framer.end_framing` — commit any remaining frame and stop framing.
    fn end_framing(&mut self) -> Result<(), PyError> {
        if matches!(&self.current_frame, Some(f) if !f.is_empty()) {
            self.commit_frame(true)?;
        }
        self.current_frame = None;
        Ok(())
    }

    /// `_Framer.commit_frame` — commit the active frame into `pending` when it
    /// has reached the target size (or `force`), then flush `pending` to the
    /// file. Only ever called at a `save` boundary or `end_framing`.
    fn commit_frame(&mut self, force: bool) -> Result<(), PyError> {
        let commit = match &self.current_frame {
            Some(f) => f.len() >= FRAME_SIZE_TARGET || force,
            None => false,
        };
        if !commit {
            return Ok(());
        }
        let data = std::mem::take(self.current_frame.as_mut().unwrap());
        if data.len() >= FRAME_SIZE_MIN {
            self.pending.push(op::FRAME);
            self.pending
                .extend_from_slice(&(data.len() as u64).to_le_bytes());
        }
        self.pending.extend_from_slice(&data);
        self.flush()
    }

    /// `_Framer.write_large_bytes` — terminate the active frame, then write the
    /// header (with the committed frame) and the payload as separate `write`s,
    /// avoiding buffering the large payload. The payload is copied to
    /// GC-stable storage first because the `write` calls below run arbitrary
    /// Python and can relocate the (managed) source the slice borrows from.
    fn write_large_bytes(&mut self, header: &[u8], payload: &[u8]) -> Result<(), PyError> {
        let owned = payload.to_vec();
        if matches!(&self.current_frame, Some(f) if !f.is_empty()) {
            self.commit_frame(true)?;
        }
        self.pending.extend_from_slice(header);
        match self.file_slot {
            Some(slot) => {
                self.flush()?;
                let w_payload = pyre_object::w_bytes_from_bytes(&owned);
                // Pin the freshly-built payload and re-read the cached
                // callable from its outer root before arbitrary Python.
                let _roots = pyre_object::gc_roots::push_roots();
                let _ = pyre_object::gc_roots::pin_root(w_payload);
                let payload_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
                if let Some(write_slot) = self.write_slot {
                    call_fn(
                        pyre_object::gc_roots::shadow_stack_get(write_slot),
                        &[pyre_object::gc_roots::shadow_stack_get(payload_slot)],
                    )?;
                } else {
                    let w_file = pyre_object::gc_roots::shadow_stack_get(slot);
                    call_meth(
                        w_file,
                        "write",
                        &[pyre_object::gc_roots::shadow_stack_get(payload_slot)],
                    )?;
                }
            }
            None => self.pending.extend_from_slice(&owned),
        }
        Ok(())
    }
}

#[pyre_interpreter::pyre_methods(
    doc = "Pickler(file, protocol=None) -> pickler writing to file.",
    _text_signature_ = "(file, protocol=None, fix_imports=True, buffer_callback=None)"
)]
impl W_Pickler {
    #[staticmethod]
    fn __new__(cls: PyObjectRef, _args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
        // Construction arguments are consumed/validated by `__init__`; accept
        // (and ignore) any positional or keyword args here via the whole-slice
        // catch-all so the ctor keyword parameters (protocol/fix_imports/
        // buffer_callback) do not trip an unknown-argument error in `__new__`.
        let _ = _args;
        // `objspace.py` `allocate_instance` validates the requested subtype.
        // The builtin is the base layout; a subclass is `typedef.py`
        // `_getusercls` (`interp_pickle.py` `descr__new__`).
        pyre_interpreter::typedef::check_user_subclass(type_object(), cls)?;
        let _roots = pyre_object::gc_roots::push_roots();
        let cls_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(cls);
        // Allocate the memo before the payload's other words are taken.
        // A subclass `allocate_instance` pins that list across its allocation.
        let w_memo = pyre_object::listobject::w_list_new_empty();
        Ok(W_Pickler::allocate_instance(
            W_Pickler {
                ob: pyre_object::PyObject {
                    ob_type: std::ptr::null(),
                    w_class: std::ptr::null_mut(),
                },
                w_file: pyre_object::w_none(),
                w_write: pyre_object::w_none(),
                proto: 0,
                bin: false,
                framing: false,
                fix_imports: true,
                buffer_callback: pyre_object::w_none(),
                w_memo,
                fast: 0,
                w_dispatch_table: pyre_object::PY_NULL,
                w_pers_func: pyre_object::PY_NULL,
                running: false,
            },
            pyre_object::gc_roots::shadow_stack_get(cls_slot),
        ))
    }

    fn __init__(
        &mut self,
        file: PyObjectRef,
        #[default(pyre_object::w_none())] protocol: PyObjectRef,
        #[default(pyre_object::boolobject::w_bool_from(true))] fix_imports: PyObjectRef,
        #[default(pyre_object::w_none())] buffer_callback: PyObjectRef,
    ) -> Result<(), PyError> {
        // `_pickle.c:711-721` BEGIN/END_USING_PICKLER.  In particular,
        // `file.write` lookup and protocol coercion may execute Python.
        let _roots = pyre_object::gc_roots::push_roots();
        let _ = pyre_object::gc_roots::pin_root(self as *mut W_Pickler as PyObjectRef);
        let self_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let _ = pyre_object::gc_roots::pin_root(file);
        let file_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let _ = pyre_object::gc_roots::pin_root(protocol);
        let protocol_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let _ = pyre_object::gc_roots::pin_root(fix_imports);
        let fix_imports_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let _ = pyre_object::gc_roots::pin_root(buffer_callback);
        let buffer_callback_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        if cur_pickler(self_slot).running {
            return Err(PyError::runtime_error("Pickler object is already used"));
        }
        cur_pickler(self_slot).running = true;
        let _running = RunningGuard { self_slot };
        // `fix_imports` gates the `_compat_pickle` py3→py2 name remap that the
        // protocol-< 3 save path would otherwise always apply.
        let proto = normalize_protocol(pyre_object::gc_roots::shadow_stack_get(protocol_slot))?;
        // `file must have a 'write' attribute` (interp_pickle.py). This
        // check precedes the `buffer_callback` one below; `descr__new__`
        // (`interp_pickle.py:1822`) orders them the other way. Measured on
        // 3.14.5, a call carrying both faults reports this TypeError.
        let Some(w_write) = pyre_interpreter::baseobjspace::findattr_result(
            pyre_object::gc_roots::shadow_stack_get(file_slot),
            "write",
        )?
        else {
            return Err(PyError::type_error("file must have a 'write' attribute"));
        };
        let _ = pyre_object::gc_roots::pin_root(w_write);
        let write_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        if !unsafe {
            pyre_object::is_none(pyre_object::gc_roots::shadow_stack_get(
                buffer_callback_slot,
            ))
        } && proto < 5
        {
            return Err(PyError::value_error("buffer_callback needs protocol >= 5"));
        }
        let fix_imports = pyre_interpreter::baseobjspace::is_true(
            pyre_object::gc_roots::shadow_stack_get(fix_imports_slot),
        )?;
        let memo = pyre_object::listobject::w_list_new_empty();
        let current = cur_pickler(self_slot);
        current.w_file = pyre_object::gc_roots::shadow_stack_get(file_slot);
        current.w_write = pyre_object::gc_roots::shadow_stack_get(write_slot);
        current.proto = proto;
        current.bin = proto >= 1;
        current.framing = proto >= 4;
        current.fix_imports = fix_imports;
        current.buffer_callback = pyre_object::gc_roots::shadow_stack_get(buffer_callback_slot);
        current.w_memo = memo;
        current.fast = 0;
        current.w_dispatch_table = pyre_object::PY_NULL;
        current.w_pers_func = pyre_object::PY_NULL;
        pyre_object::gc_hook::try_gc_write_barrier(pyre_object::gc_roots::shadow_stack_get(
            self_slot,
        ) as pyre_object::gc_hook::GCREF);
        Ok(())
    }

    /// `Pickler.clear_memo` — reset the memo so the next `dump` starts fresh.
    fn clear_memo(&mut self) {
        // `w_list_new` can collect, so re-read `self` from a pinned slot
        // afterwards rather than storing through the stale `&mut` — the twin
        // `UnpicklerMemoProxy::clear` already has this shape.
        let self_obj = self as *mut W_Pickler as PyObjectRef;
        let _roots = pyre_object::gc_roots::push_roots();
        let _ = pyre_object::gc_roots::pin_root(self_obj);
        let slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let empty = pyre_object::listobject::w_list_new_empty();
        let me = cur_pickler(slot);
        me.w_memo = empty;
        pickler_write_barrier(me as *mut W_Pickler as PyObjectRef);
    }

    fn dump(&mut self, w_obj: PyObjectRef) -> Result<(), PyError> {
        // `_pickle.c:4737` BEGIN_USING_PICKLER: reducers, persistent-id
        // hooks, and destination writes must not recursively mutate this
        // pickler's active memo/framer state.
        let self_ptr = self as *mut W_Pickler as PyObjectRef;
        let _roots = pyre_object::gc_roots::push_roots();
        let _ = pyre_object::gc_roots::pin_root(self_ptr);
        let self_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let (
            proto,
            bin,
            framing,
            fix_imports,
            fast,
            w_dispatch_table,
            w_file,
            w_write,
            buffer_callback,
            w_memo,
        ) = {
            let current = cur_pickler(self_slot);
            if current.running {
                return Err(PyError::runtime_error("Pickler object is already used"));
            }
            current.running = true;
            // Snapshot scalar fields and root every object field below before
            // an attribute lookup can execute Python and relocate the owner.
            (
                current.proto,
                current.bin,
                current.framing,
                current.fix_imports,
                current.fast,
                current.w_dispatch_table,
                current.w_file,
                current.w_write,
                current.buffer_callback,
                current.w_memo,
            )
        };
        // Declared after `_roots`, so Drop runs while `self_slot` is live.
        let _running = RunningGuard { self_slot };

        // CPython/PyPy reject a subclass instance whose `__init__` did not
        // initialise the output stream as a PicklingError, before attempting
        // to resolve or call `file.write`.
        if unsafe { pyre_object::is_none(w_file) } {
            return Err(pickling_error("Pickler.__init__() was not called"));
        }

        // Pin everything that must survive the `persistent_id` /
        // `reducer_override` / `dispatch_table` lookups below: each runs Python
        // (a property / `__getattr__` / `copyreg` import) and can relocate
        // objects under the moving GC.
        let _ = pyre_object::gc_roots::pin_root(w_obj);
        let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let _ = pyre_object::gc_roots::pin_root(w_file);
        let file_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let _ = pyre_object::gc_roots::pin_root(w_write);
        let write_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let _ = pyre_object::gc_roots::pin_root(w_memo);
        let memo_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let _ = pyre_object::gc_roots::pin_root(buffer_callback);
        let cb_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let _ = pyre_object::gc_roots::pin_root(w_dispatch_table);
        let configured_dt_slot = pyre_object::gc_roots::shadow_stack_len() - 1;

        // `persistent_id` resolves to a subclass method override or the
        // instance value set through the getter (its getter raises while
        // unset, so a base pickler resolves to `PY_NULL`); `reducer_override`
        // is a subclass hook only (absent on a base pickler).  An explicit
        // `persistent_id = None` / `reducer_override = None` is kept as the
        // hook: `dump` then calls `None(obj)` and raises `TypeError`, matching
        // `_pickle` (only deleting/leaving it unset disables the hook).
        // `findattr_result` propagates a hook property's own error instead of
        // panicking; each resolved hook is pinned before the next lookup.
        let pers_func = pyre_interpreter::baseobjspace::findattr_result(
            pyre_object::gc_roots::shadow_stack_get(self_slot),
            "persistent_id",
        )?
        .unwrap_or(pyre_object::PY_NULL);
        let _ = pyre_object::gc_roots::pin_root(pers_func);
        let pers_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let reducer_override = pyre_interpreter::baseobjspace::findattr_result(
            pyre_object::gc_roots::shadow_stack_get(self_slot),
            "reducer_override",
        )?
        .unwrap_or(pyre_object::PY_NULL);
        let _ = pyre_object::gc_roots::pin_root(reducer_override);
        let reducer_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        // `interp_pickle.py:686-690` — the internal `dispatch_table` field,
        // else a dynamically-resolved `dispatch_table` attribute (a subclass
        // class attr / property), else the global `copyreg.dispatch_table`.  An
        // explicit `dispatch_table = None` is honored as the table (later
        // subscripting `None[type]` raises `TypeError`); only an absent
        // attribute falls back to `copyreg`.
        let configured_dispatch_table = pyre_object::gc_roots::shadow_stack_get(configured_dt_slot);
        let dispatch_table = if !configured_dispatch_table.is_null() {
            configured_dispatch_table
        } else if let Some(dt) = pyre_interpreter::baseobjspace::findattr_result(
            pyre_object::gc_roots::shadow_stack_get(self_slot),
            "dispatch_table",
        )? {
            dt
        } else {
            copyreg_dispatch_table()
        };
        let _ = pyre_object::gc_roots::pin_root(dispatch_table);
        let dt_slot = pyre_object::gc_roots::shadow_stack_len() - 1;

        // Re-read every pinned value: the lookups / `copyreg` import above may
        // have collected. `pickle_core` streams the frames to `w_file`.
        let w_obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
        let w_memo = pyre_object::gc_roots::shadow_stack_get(memo_slot);
        let w_file = pyre_object::gc_roots::shadow_stack_get(file_slot);
        let w_write = pyre_object::gc_roots::shadow_stack_get(write_slot);
        let buffer_callback = pyre_object::gc_roots::shadow_stack_get(cb_slot);
        let pers_func = pyre_object::gc_roots::shadow_stack_get(pers_slot);
        let reducer_override = pyre_object::gc_roots::shadow_stack_get(reducer_slot);
        let dispatch_table = pyre_object::gc_roots::shadow_stack_get(dt_slot);
        pickle_core_impl(
            w_obj,
            w_file,
            w_write,
            proto,
            bin,
            framing,
            fix_imports,
            pers_func,
            buffer_callback,
            w_memo,
            fast != 0,
            dispatch_table,
            reducer_override,
        )?;
        Ok(())
    }

    /// `Pickler.bin` — whether binary opcodes are in use (protocol >= 1).
    #[getter]
    fn bin(&self) -> i64 {
        self.bin as i64
    }

    /// `Pickler.fast` — when non-zero, memoization is skipped.
    #[getter]
    fn fast(&self) -> i64 {
        self.fast
    }

    #[setter]
    fn set_fast(&mut self, w_value: PyObjectRef) -> Result<(), PyError> {
        // `interp_pickle.py` stores `space.int_w(w_val)`.
        self.fast = pyre_interpreter::baseobjspace::int_w(w_value)?;
        Ok(())
    }

    /// `Pickler.dispatch_table` — the per-pickler type→reduce mapping. Unset by
    /// default (reading it raises `AttributeError`); when unset the dump path
    /// falls back to `copyreg.dispatch_table`.
    #[getter]
    fn dispatch_table(&self) -> Result<PyObjectRef, PyError> {
        if self.w_dispatch_table.is_null() {
            return Err(PyError::attribute_error(
                "'_pickle.Pickler' object has no attribute 'dispatch_table'".to_string(),
            ));
        }
        Ok(self.w_dispatch_table)
    }

    #[setter]
    fn set_dispatch_table(&mut self, w_value: PyObjectRef) {
        self.w_dispatch_table = w_value;
        pickler_write_barrier(self as *mut W_Pickler as PyObjectRef);
    }

    /// `_pickle.c:persistent_id` — the base implementation deliberately
    /// returns None.  It lives in the type dict (and is therefore GC-rooted)
    /// so the `persistent_id` getset below can bind the same descriptor on
    /// every read, matching CPython's method identity/equality semantics
    /// without an off-object singleton or TLS root.
    fn __persistent_id_default(&self, _obj: PyObjectRef) -> PyObjectRef {
        pyre_object::w_none()
    }

    /// `Pickler.persistent_id` — CPython keeps a base method descriptor and a
    /// separate `persistent_id_attr` instance override.  PyPy's getset stores
    /// the override in `pers_func`.  Keep that owner shape: when unset, bind
    /// the GC-rooted base implementation above; assignment/deletion only
    /// changes this pickler's `w_pers_func`.
    #[getter]
    fn persistent_id(&self) -> Result<PyObjectRef, PyError> {
        if !self.w_pers_func.is_null() {
            return Ok(self.w_pers_func);
        }
        let _roots = pyre_object::gc_roots::push_roots();
        let _ = pyre_object::gc_roots::pin_root(self as *const W_Pickler as PyObjectRef);
        let self_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let cls = type_object();
        let _ = pyre_object::gc_roots::pin_root(cls);
        let cls_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let descr = pyre_interpreter::runtime_ops::type_dict_lookup(
            pyre_object::gc_roots::shadow_stack_get(cls_slot),
            "__persistent_id_default",
        )
        .ok_or_else(|| PyError::runtime_error("lost Pickler.persistent_id descriptor"))?;
        let _ = pyre_object::gc_roots::pin_root(descr);
        let descr_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        Ok(pyre_object::w_method_new(
            pyre_object::gc_roots::shadow_stack_get(descr_slot),
            pyre_object::gc_roots::shadow_stack_get(self_slot),
            pyre_object::gc_roots::shadow_stack_get(cls_slot),
        ))
    }

    #[setter]
    fn set_persistent_id(&mut self, w_value: PyObjectRef) {
        self.w_pers_func = w_value;
        pickler_write_barrier(self as *mut W_Pickler as PyObjectRef);
    }

    #[deleter("persistent_id")]
    fn del_persistent_id(&mut self) {
        self.w_pers_func = pyre_object::PY_NULL;
    }

    /// `Pickler.memo` — a fresh `PicklerMemoProxy` viewing this pickler's memo
    /// (CPython hands back a new proxy on each access).
    #[getter]
    fn memo(&self) -> PyObjectRef {
        let self_obj = self as *const W_Pickler as PyObjectRef;
        let _roots = pyre_object::gc_roots::push_roots();
        let _ = pyre_object::gc_roots::pin_root(self_obj);
        let slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        memo_proxy::type_object();
        let proxy = PicklerMemoProxy::allocate_stable(PicklerMemoProxy {
            ob: pyre_object::PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            w_pickler: pyre_object::PY_NULL,
        });
        // `allocate` may have relocated the pickler; wire the (young) proxy to
        // its post-collection address.
        if let Some(px) = PicklerMemoProxy::from_obj(proxy) {
            px.w_pickler = pyre_object::gc_roots::shadow_stack_get(slot);
            // `find_initializing_stores` would elide the barrier for a store
            // into a just-malloc'd young object, but `allocate_stable` fires
            // the managed barrier during construction and so consumes the
            // proxy's fresh TRACK_YOUNG_PTRS; the exemption's premise no
            // longer holds here.
            pickler_write_barrier(proxy);
        }
        proxy
    }

    /// `Pickler.memo` setter — replace the memo from a `PicklerMemoProxy` or an
    /// `{id: (index, obj)}` mapping; the object is restored at its `index`
    /// position (the position-indexed memo list, from which the dump rebuilds
    /// the identity index). Any other type is a `TypeError`.
    #[setter]
    fn set_memo(&mut self, w_value: PyObjectRef) -> Result<(), PyError> {
        let self_obj = self as *mut W_Pickler as PyObjectRef;
        let _roots = pyre_object::gc_roots::push_roots();
        let _ = pyre_object::gc_roots::pin_root(self_obj);
        let self_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let w_dict = if PicklerMemoProxy::from_obj(w_value).is_some() {
            call_meth(w_value, "copy", &[])?
        } else if unsafe { pyre_object::is_dict(w_value) } {
            unsafe { pyre_object::dictmultiobject::w_dict_copy(w_value) }
        } else {
            return Err(PyError::type_error(format!(
                "'memo' attribute must be a PicklerMemoProxy object or dict, not {}",
                pyre_interpreter::baseobjspace::object_functionstr_type_name(w_value),
            )));
        };
        let _ = pyre_object::gc_roots::pin_root(w_dict);
        let dict_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        // No GC between reading the items and `w_list_new` (which re-pins them).
        let items = unsafe {
            pyre_object::dictmultiobject::w_dict_items(pyre_object::gc_roots::shadow_stack_get(
                dict_slot,
            ))
        };
        let mut max_idx: i64 = -1;
        for (_, tup) in &items {
            if !(unsafe { pyre_object::is_tuple(*tup) }
                && unsafe { pyre_object::tupleobject::w_tuple_len(*tup) } == 2)
            {
                return Err(PyError::type_error("'memo' values must be 2-item tuples"));
            }
            let idx = unsafe { pyre_object::tupleobject::w_tuple_getitem(*tup, 0) }.unwrap();
            let i = pyre_interpreter::baseobjspace::int_w(idx)?;
            if i < 0 {
                // The memo is a position-indexed list, so a negative slot index
                // has no representation (and would emit a malformed back-reference).
                return Err(PyError::value_error("memo index must be non-negative"));
            }
            if i > max_idx {
                max_idx = i;
            }
        }
        // Gaps in a sparse memo hold `PY_NULL`, not `None`: a `None` slot would
        // be indistinguishable from a genuinely memoized `None`, which `copy()`
        // must still expose. `w_list_new_object` keeps the `PY_NULL` holes by
        // pointer (Object strategy), mirroring the unpickler memo.
        let mut slots: Vec<PyObjectRef> = vec![pyre_object::PY_NULL; (max_idx + 1) as usize];
        for (_, tup) in &items {
            let idx = unsafe { pyre_object::tupleobject::w_tuple_getitem(*tup, 0) }.unwrap();
            let i = pyre_interpreter::baseobjspace::int_w(idx)? as usize;
            slots[i] = unsafe { pyre_object::tupleobject::w_tuple_getitem(*tup, 1) }.unwrap();
        }
        let list = pyre_object::listobject::w_list_new_object(slots);
        let me =
            unsafe { &mut *(pyre_object::gc_roots::shadow_stack_get(self_slot) as *mut W_Pickler) };
        me.w_memo = list;
        pickler_write_barrier(me as *mut W_Pickler as PyObjectRef);
        Ok(())
    }

    /// `Pickler.memo` is not deletable.
    #[deleter("memo")]
    fn del_memo(&self) -> Result<(), PyError> {
        Err(PyError::type_error("attribute deletion is not supported"))
    }
}

/// `interp_pickle.py PicklerMemoProxy` — a live view of a pickler's identity
/// memo. `copy` snapshots it as `{id(obj): (memo_index, obj)}`; `clear` empties
/// it. The pickler stores the memo as a position-indexed Python `list` (see
/// `PickleCtx`), so `copy` derives the `id` keys from the listed objects.
///
/// Held in its own module so `#[pyre_methods]` emits a `type_object()` that
/// does not clash with `W_Pickler`'s (each impl emits a module-scoped one).
pub use memo_proxy::PicklerMemoProxy;

mod memo_proxy {
    use super::*;

    // Modules/_pickle.c:pickle_exec CREATE_TYPEs memoproxy_spec too.
    #[pyre_interpreter::pyre_class("_pickle.PicklerMemoProxy", cpython_heaptype)]
    pub struct PicklerMemoProxy {
        pub(super) w_pickler: PyObjectRef,
    }

    #[pyre_interpreter::pyre_methods(doc = "Proxy for a Pickler's memo.")]
    impl PicklerMemoProxy {
        /// `PicklerMemoProxy.copy` — `{id(obj): (memo_index, obj)}`.
        fn copy(&self) -> Result<PyObjectRef, PyError> {
            let w_memo = unsafe { &*(self.w_pickler as *const W_Pickler) }.w_memo;
            let _roots = pyre_object::gc_roots::push_roots();
            let _ = pyre_object::gc_roots::pin_root(w_memo);
            let memo_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let result = pyre_object::dictmultiobject::w_dict_new();
            let _ = pyre_object::gc_roots::pin_root(result);
            let res_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let n = unsafe {
                pyre_object::listobject::w_list_len(pyre_object::gc_roots::shadow_stack_get(
                    memo_slot,
                ))
            };
            for i in 0..n {
                let _it = pyre_object::gc_roots::push_roots();
                let obj = unsafe {
                    pyre_object::listobject::w_list_getitem(
                        pyre_object::gc_roots::shadow_stack_get(memo_slot),
                        i as i64,
                    )
                }
                .unwrap();
                // Gap slots from a sparse `set_memo` hold the `PY_NULL`
                // placeholder, not real memo entries; skip them so `copy()`
                // exposes only the memoized objects (an id-keyed, gap-free
                // snapshot) while still surfacing a genuinely memoized `None`.
                if obj.is_null() {
                    continue;
                }
                // `(index, obj)` — pin the boxed index before the tuple malloc.
                let mut fields = pyre_object::gc_roots::RootedItems::new();
                fields.push(pyre_object::w_int_new(i as i64));
                fields.push(obj);
                let tup = pyre_object::tupleobject::w_tuple_new(fields.take());
                let _ = pyre_object::gc_roots::pin_root(tup);
                let tup_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
                // id(obj) read from the (relocated) tuple element.
                let cur_obj = unsafe {
                    pyre_object::tupleobject::w_tuple_getitem(
                        pyre_object::gc_roots::shadow_stack_get(tup_slot),
                        1,
                    )
                }
                .unwrap();
                let key = pyre_object::w_int_new(cur_obj as i64);
                let _ = pyre_object::gc_roots::pin_root(key);
                let key_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
                // The IndexMap insert allocates via `std::alloc` (no collection), so
                // the freshly read addresses stay valid through it.
                unsafe {
                    pyre_object::dictmultiobject::w_dict_store(
                        pyre_object::gc_roots::shadow_stack_get(res_slot),
                        pyre_object::gc_roots::shadow_stack_get(key_slot),
                        pyre_object::gc_roots::shadow_stack_get(tup_slot),
                    );
                }
            }
            Ok(pyre_object::gc_roots::shadow_stack_get(res_slot))
        }

        /// `PicklerMemoProxy.clear` — empty the pickler's memo.
        fn clear(&self) {
            let w_pickler = self.w_pickler;
            let _roots = pyre_object::gc_roots::push_roots();
            let _ = pyre_object::gc_roots::pin_root(w_pickler);
            let slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let empty = pyre_object::listobject::w_list_new_empty();
            let p =
                unsafe { &mut *(pyre_object::gc_roots::shadow_stack_get(slot) as *mut W_Pickler) };
            p.w_memo = empty;
            pickler_write_barrier(p as *mut W_Pickler as PyObjectRef);
        }
    }
}

/// `copyreg.dispatch_table`, or `PY_NULL` when `copyreg` is unavailable.
pub(crate) fn copyreg_dispatch_table() -> PyObjectRef {
    match import_module("copyreg") {
        Ok(copyreg) => pyre_interpreter::baseobjspace::getattr_str(copyreg, "dispatch_table")
            .unwrap_or(pyre_object::PY_NULL),
        Err(_) => pyre_object::PY_NULL,
    }
}

/// `buffer_callback needs protocol >= 5` — reject a non-None callback under
/// an earlier protocol (interp_pickle.py:1818).
pub(crate) fn check_buffer_callback(
    buffer_callback: PyObjectRef,
    proto: i64,
) -> Result<(), PyError> {
    if !unsafe { pyre_object::is_none(buffer_callback) } && proto < 5 {
        return Err(PyError::value_error("buffer_callback needs protocol >= 5"));
    }
    Ok(())
}

/// `interp_pickle.py W_Pickler.__init__` protocol resolution: `None` →
/// `DEFAULT_PROTOCOL`, a negative value → `HIGHEST_PROTOCOL`, and anything
/// above `HIGHEST_PROTOCOL` is rejected.
pub(crate) fn normalize_protocol(w_protocol: PyObjectRef) -> Result<i64, PyError> {
    if unsafe { pyre_object::is_none(w_protocol) } {
        return Ok(DEFAULT_PROTOCOL);
    }
    let p = pyre_interpreter::baseobjspace::int_w(w_protocol)?;
    if p < 0 {
        Ok(HIGHEST_PROTOCOL)
    } else if p > HIGHEST_PROTOCOL {
        Err(PyError::value_error("pickle protocol must be <= 5"))
    } else {
        Ok(p)
    }
}

/// Pickle `w_obj`. When `w_file` is a file object the frames are streamed to
/// its `write` as they are produced and `PY_NULL` is returned; when `w_file` is
/// `PY_NULL` (the `dumps` path) the whole pickle is accumulated and returned as
/// a `bytes`. Shared by `W_Pickler.dump` and the module-level `dump` / `dumps`.
/// `pers_func` is the `persistent_id` callable or `PY_NULL`. PROTO is written
/// before framing begins (outside the frame); STOP is written while framing is
/// active (inside the last frame).
#[allow(clippy::too_many_arguments)]
pub(crate) fn pickle_core(
    w_obj: PyObjectRef,
    w_file: PyObjectRef,
    proto: i64,
    bin: bool,
    framing: bool,
    fix_imports: bool,
    pers_func: PyObjectRef,
    buffer_callback: PyObjectRef,
    w_memo: PyObjectRef,
    fast: bool,
    dispatch_table: PyObjectRef,
    reducer_override: PyObjectRef,
) -> Result<PyObjectRef, PyError> {
    pickle_core_impl(
        w_obj,
        w_file,
        pyre_object::PY_NULL,
        proto,
        bin,
        framing,
        fix_imports,
        pers_func,
        buffer_callback,
        w_memo,
        fast,
        dispatch_table,
        reducer_override,
    )
}

#[allow(clippy::too_many_arguments)]
fn pickle_core_impl(
    w_obj: PyObjectRef,
    w_file: PyObjectRef,
    w_write: PyObjectRef,
    proto: i64,
    bin: bool,
    framing: bool,
    fix_imports: bool,
    pers_func: PyObjectRef,
    buffer_callback: PyObjectRef,
    w_memo: PyObjectRef,
    fast: bool,
    dispatch_table: PyObjectRef,
    reducer_override: PyObjectRef,
) -> Result<PyObjectRef, PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let w_obj = pyre_object::gc_roots::pin_root(w_obj);
    let pers_func = PinnedRef::new(pers_func, !pers_func.is_null());
    let buffer_callback = PinnedRef::new(
        buffer_callback,
        !buffer_callback.is_null() && !unsafe { pyre_object::is_none(buffer_callback) },
    );
    let dispatch_table = PinnedRef::new(
        dispatch_table,
        !dispatch_table.is_null() && !unsafe { pyre_object::is_none(dispatch_table) },
    );
    let reducer_override = PinnedRef::new(reducer_override, !reducer_override.is_null());
    // Pin the memo list and index its existing entries (a reused `Pickler`
    // carries memo state across `dump` calls until `clear_memo`).
    let w_memo = pyre_object::gc_roots::pin_root(w_memo);
    let memo_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let mut index: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut str_memo: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut bytes_memo: HashMap<Vec<u8>, usize> = HashMap::new();
    let n = unsafe { pyre_object::listobject::w_list_len(w_memo) };
    for i in 0..n {
        let o = unsafe { pyre_object::listobject::w_list_getitem(w_memo, i as i64) }.unwrap();
        // Gap slots from a sparse `set_memo` hold `PY_NULL`; they are not real
        // memo entries, so leave them out of the identity index.
        if o.is_null() {
            continue;
        }
        index
            .entry(pyre_object::gc_hook::gc_identity_hash(o as usize))
            .or_default()
            .push(i);
        if let Some((is_str, key)) = PickleCtx::value_memo_key(o) {
            let memo = if is_str {
                &mut str_memo
            } else {
                &mut bytes_memo
            };
            memo.insert(key, i);
        }
    }

    let mut ctx = PickleCtx {
        proto,
        bin,
        fix_imports,
        memo_slot,
        index,
        str_memo,
        bytes_memo,
        pers_func,
        buffer_callback,
        fast,
        fast_nesting: 0,
        fast_memo: HashMap::new(),
        dispatch_table,
        reducer_override,
    };
    // Pin the destination file so the framer can re-read it across each
    // streaming `write` (which can relocate it). `None` selects accumulate mode.
    let file_slot = if w_file.is_null() {
        None
    } else {
        let _ = pyre_object::gc_roots::pin_root(w_file);
        Some(pyre_object::gc_roots::shadow_stack_len() - 1)
    };
    let write_slot = if w_write.is_null() {
        None
    } else {
        let _ = pyre_object::gc_roots::pin_root(w_write);
        Some(pyre_object::gc_roots::shadow_stack_len() - 1)
    };
    let mut fr = Framer::new(file_slot, write_slot);
    if proto >= 2 {
        fr.push(op::PROTO);
        fr.push(proto as u8);
    }
    if framing {
        fr.start_framing();
    }
    save(&mut ctx, &mut fr, w_obj)?;
    fr.push(op::STOP);
    fr.end_framing()?;
    // Flush the unframed residual (the whole pickle for protocol < 4, a no-op
    // after the last frame committed for protocol >= 4); a no-op without a file.
    fr.flush()?;

    if file_slot.is_some() {
        Ok(pyre_object::PY_NULL)
    } else {
        Ok(pyre_object::w_bytes_from_bytes(&fr.pending))
    }
}

/// `_pickle.c FAST_NESTING_LIMIT` — the nested-container depth past which fast
/// mode starts tracking active containers to detect a cycle.
const FAST_NESTING_LIMIT: i64 = 50;

/// `_pickle.c fast_save_enter`. Fast mode skips memoization, so a cyclic
/// container would recurse until the stack overflows. Past `FAST_NESTING_LIMIT`
/// nested containers, track the active container by identity and raise
/// `ValueError` on a repeat. Only the container saves (`save_list` / `save_dict`
/// / `save_set`) call this, so a reducer that re-enters the same custom object
/// keeps the normal recursion behaviour (a `RecursionError`) rather than being
/// misreported as a fast-mode container cycle. `obj_slot` is the container's
/// pinned shadow-stack slot; a shared `gc_identity_hash` is disambiguated by
/// pointer identity against the pinned ancestors, like `memo_get`. Returns the
/// bucket key to hand back to `fast_save_leave`, or `None` when nothing was
/// tracked.
fn fast_save_enter(ctx: &mut PickleCtx, obj_slot: usize) -> Result<Option<usize>, PyError> {
    if !ctx.fast {
        return Ok(None);
    }
    ctx.fast_nesting += 1;
    if ctx.fast_nesting < FAST_NESTING_LIMIT {
        return Ok(None);
    }
    let w_cur = pyre_object::gc_roots::shadow_stack_get(obj_slot);
    let h = pyre_object::gc_hook::gc_identity_hash(w_cur as usize);
    let is_cycle = ctx.fast_memo.get(&h).is_some_and(|slots| {
        slots
            .iter()
            .any(|&anc| pyre_object::gc_roots::shadow_stack_get(anc) == w_cur)
    });
    if is_cycle {
        ctx.fast_nesting -= 1;
        let type_name = pyre_interpreter::baseobjspace::object_functionstr_type_name(w_cur);
        let w_cur = pyre_object::gc_roots::shadow_stack_get(obj_slot);
        return Err(PyError::value_error(format!(
            "fast mode: can't pickle cyclic objects including object type {} at {}",
            type_name,
            pyre_interpreter::display::repr_gc_addr(w_cur),
        )));
    }
    ctx.fast_memo.entry(h).or_default().push(obj_slot);
    Ok(Some(h))
}

/// `_pickle.c fast_save_leave` — drop the active container from the path on the
/// way out. Must run for both an `Ok` and an `Err` member-save result so
/// `fast_nesting` and `fast_memo` stay balanced.
fn fast_save_leave(ctx: &mut PickleCtx, token: Option<usize>, obj_slot: usize) {
    if !ctx.fast {
        return;
    }
    if let Some(h) = token
        && let Some(slots) = ctx.fast_memo.get_mut(&h)
    {
        slots.retain(|&anc| anc != obj_slot);
        if slots.is_empty() {
            ctx.fast_memo.remove(&h);
        }
    }
    ctx.fast_nesting -= 1;
}

/// `interp_pickle.py W_Pickler.save` with the persistent-id hook: every
/// object is first offered to `persistent_id`; a non-None result is saved
/// as a persistent reference instead of by value.
fn save(ctx: &mut PickleCtx, buf: &mut Framer, w_obj: PyObjectRef) -> Result<(), PyError> {
    // A frame boundary can only fall at the start of a `save`, never inside an
    // object; commit the active frame once it has grown past the target.
    // Committing streams to the file (arbitrary Python), and the `persistent_id`
    // hook below is too, so pin `w_obj` across both and read it from the pin.
    let _roots = pyre_object::gc_roots::push_roots();
    let _ = pyre_object::gc_roots::pin_root(w_obj);
    let slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    buf.commit_frame(false)?;
    let pers_func = ctx.pers_func.get();
    if !pers_func.is_null() {
        let w_pid = call_fn(pers_func, &[pyre_object::gc_roots::shadow_stack_get(slot)])?;
        if !unsafe { pyre_object::is_none(w_pid) } {
            save_pers(ctx, buf, w_pid)
        } else {
            save_object(ctx, buf, pyre_object::gc_roots::shadow_stack_get(slot))
        }
    } else {
        save_object(ctx, buf, pyre_object::gc_roots::shadow_stack_get(slot))
    }
}

/// `interp_pickle.py save_pers` — emit a persistent reference. The
/// persistent id itself is saved by value (skipping the persistent-id
/// hook) in binary protocols, or as an ASCII line in protocol 0.
fn save_pers(ctx: &mut PickleCtx, buf: &mut Framer, w_pid: PyObjectRef) -> Result<(), PyError> {
    if ctx.bin {
        save_object(ctx, buf, w_pid)?;
        buf.push(op::BINPERSID);
        return Ok(());
    }
    let w_str = if unsafe { pyre_object::is_str(w_pid) } {
        w_pid
    } else {
        let str_fn = crate::module::_pickle::lookup_builtin("str")
            .ok_or_else(|| pickling_error("str builtin unavailable"))?;
        call_fn(str_fn, &[w_pid])?
    };
    let s = unsafe { pyre_object::unicodeobject::w_str_get_wtf8(w_str) }.as_bytes();
    if !s.is_ascii() {
        return Err(pickling_error(
            "persistent IDs in protocol 0 must be ASCII strings",
        ));
    }
    buf.push(op::PERSID);
    buf.extend_from_slice(s);
    buf.push(b'\n');
    Ok(())
}

/// `interp_pickle.py W_Pickler.save` body after the persistent-id hook: the
/// identity memo (atoms are never memoized), then the type dispatch in
/// [`dispatch_save`].  The memo is consulted before `reducer_override` (a
/// repeated reference becomes a GET, matching CPython 3.14: a second
/// occurrence of the same object never re-enters the hook).
fn save_object(ctx: &mut PickleCtx, buf: &mut Framer, w_obj: PyObjectRef) -> Result<(), PyError> {
    // Only EXACT atoms skip the memo; an `int`/`float` subclass is a regular
    // object (memoized, saved through the reduce protocol).
    let is_atom = unsafe {
        pyre_object::is_none(w_obj)
            || pyre_object::is_bool(w_obj)
            || pyre_object::is_exact_type(w_obj, &pyre_object::INT_TYPE)
            || pyre_object::is_exact_type(w_obj, &pyre_object::FLOAT_TYPE)
    };
    // Value memo, then identity memo — a repeated reference becomes a GET
    // back-reference.
    if !is_atom && let Some(idx) = ctx.value_memo_get(w_obj).or_else(|| ctx.memo_get(w_obj)) {
        write_get(ctx, buf, idx);
        return Ok(());
    }
    dispatch_save(ctx, buf, w_obj)
}

/// Exact-type dispatch via the `is_*` predicates (bool is checked before int
/// because a bool is not an int here, and `is_int_or_long` also covers big
/// integers). The memo has already been consulted by [`save_object`]. Classes
/// and functions are saved by reference; anything else goes through the
/// `dispatch_table` then the reduce protocol.
fn dispatch_save(ctx: &mut PickleCtx, buf: &mut Framer, w_obj: PyObjectRef) -> Result<(), PyError> {
    // Atoms — never memoized.
    if unsafe { pyre_object::is_none(w_obj) } {
        buf.push(op::NONE);
        return Ok(());
    }
    if unsafe { pyre_object::is_bool(w_obj) } {
        return save_bool(ctx, buf, w_obj);
    }
    // Built-in dispatch is by EXACT type (`type(obj) is X`): a subclass of a
    // built-in carries the same `ob_type`/layout but a retagged `w_class`, and
    // must fall through to `save_global_or_reduce` so its subclass identity is
    // preserved.  `is_exact_type` is correct for the specialised arity-2 tuples
    // (distinct `ob_type`, canonical `tuple` `w_class`) where the `ob_type`
    // predicate `is_exact_builtin_instance` is not.
    if unsafe { pyre_object::is_exact_type(w_obj, &pyre_object::INT_TYPE) } {
        save_long(ctx, buf, w_obj)?;
        return Ok(());
    }
    if unsafe { pyre_object::is_exact_type(w_obj, &pyre_object::FLOAT_TYPE) } {
        save_float(ctx, buf, w_obj)?;
        return Ok(());
    }
    if unsafe { pyre_object::is_exact_type(w_obj, &pyre_object::bytesobject::BYTES_TYPE) } {
        return save_bytes(ctx, buf, w_obj);
    }
    if unsafe { pyre_object::is_exact_type(w_obj, &pyre_object::STR_TYPE) } {
        return save_str(ctx, buf, w_obj);
    }
    if unsafe { pyre_object::is_exact_type(w_obj, &pyre_object::DICT_TYPE) } {
        return save_dict(ctx, buf, w_obj);
    }
    if unsafe { pyre_object::is_exact_type(w_obj, &pyre_object::setobject::SET_TYPE) } {
        return save_set(ctx, buf, w_obj);
    }
    if unsafe { pyre_object::is_exact_type(w_obj, &pyre_object::setobject::FROZENSET_TYPE) } {
        return save_frozenset(ctx, buf, w_obj);
    }
    if unsafe { pyre_object::is_exact_type(w_obj, &pyre_object::LIST_TYPE) } {
        return save_list(ctx, buf, w_obj);
    }
    if unsafe { pyre_object::is_exact_type(w_obj, &pyre_object::TUPLE_TYPE) } {
        return save_tuple(ctx, buf, w_obj);
    }
    if unsafe { pyre_object::is_exact_type(w_obj, &pyre_object::bytearrayobject::BYTEARRAY_TYPE) } {
        return save_bytearray(ctx, buf, w_obj);
    }
    if pyre_interpreter::module::__pypy__::W_PickleBuffer::from_obj(w_obj).is_some() {
        return save_picklebuffer(ctx, buf, w_obj);
    }

    // `reducer_override` (a Pickler subclass hook) is offered every object not
    // dispatched above as a built-in atom/container, before save_global and
    // the reduce protocol. A result other than NotImplemented replaces the
    // default reduction. (interp_pickle.py:619-625 calls it earlier; CPython
    // 3.14 — the behaviour target — dispatches built-in types first, so the
    // hook never sees a list/dict/str and a repeated object hits the memo.)
    let reducer_override = ctx.reducer_override.get();
    if !reducer_override.is_null() {
        let _roots = pyre_object::gc_roots::push_roots();
        let _ = pyre_object::gc_roots::pin_root(w_obj);
        let slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let w_rv = call_fn(
            reducer_override,
            &[pyre_object::gc_roots::shadow_stack_get(slot)],
        )?;
        if !unsafe { pyre_object::is_not_implemented(w_rv) } {
            return save_reduce_value(
                ctx,
                buf,
                pyre_object::gc_roots::shadow_stack_get(slot),
                w_rv,
            );
        }
        return save_global_or_reduce(ctx, buf, pyre_object::gc_roots::shadow_stack_get(slot));
    }
    save_global_or_reduce(ctx, buf, w_obj)
}

/// The tail of [`dispatch_save`] after built-in dispatch and the
/// `reducer_override` hook: classes/functions by reference, then the
/// per-pickler `dispatch_table`, then the `__reduce_ex__` / `__reduce__`
/// protocol.
fn save_global_or_reduce(
    ctx: &mut PickleCtx,
    buf: &mut Framer,
    w_obj: PyObjectRef,
) -> Result<(), PyError> {
    // The reduce protocol below runs Python — the `dispatch_table` lookup and
    // `__reduce_ex__` — and the object being pickled can be a `list` or `dict`
    // subclass, whose header moves.  Pin it and re-read it from the slot at the
    // two consumers that sit after one of those calls.
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    let w_obj = pyre_object::gc_roots::pin_root(w_obj);
    // CPython 3.14 pickle.py dispatch[type] = save_type: only a class whose
    // exact metaclass is `type` takes this built-in dispatch entry. A class
    // with a custom metaclass must consult dispatch_table first.
    let is_class = unsafe { pyre_object::typeobject::is_type(w_obj) };
    let has_exact_type_metaclass = is_class
        && pyre_interpreter::typedef::r#type(w_obj).is_some_and(|metaclass| {
            pyre_interpreter::baseobjspace::is_w(
                metaclass.as_ptr(),
                pyre_interpreter::typedef::w_type(),
            )
        });
    if has_exact_type_metaclass {
        return save_type(ctx, buf, w_obj);
    }
    // Functions are saved by reference.
    if unsafe { pyre_interpreter::function::is_function(w_obj) } {
        return save_global(ctx, buf, w_obj, None);
    }

    // A `dispatch_table` reduce function registered for this exact type takes
    // precedence over `__reduce_ex__`.
    if let Some(w_rv) = dispatch_table_reduce(ctx, w_obj)? {
        let w_obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
        return save_reduce_value(ctx, buf, w_obj, w_rv);
    }

    // pickle.py: a class with a custom metaclass and no registered reducer is
    // still a global, but only after the metaclass dispatch-table lookup.
    if is_class {
        return save_global(ctx, buf, w_obj, None);
    }

    // Everything else goes through the reduce protocol.
    let w_rv = match pyre_interpreter::baseobjspace::findattr_result(w_obj, "__reduce_ex__")? {
        Some(reduce_ex) => {
            // A bound method the lookup minted has no other referrer, and
            // boxing the protocol number allocates before the call reaches it.
            let reduce_ex = pyre_object::gc_roots::pin_root(reduce_ex);
            call_fn(reduce_ex, &[pyre_object::w_int_new(ctx.proto)])?
        }
        None => match pyre_interpreter::baseobjspace::findattr_result(
            pyre_object::gc_roots::shadow_stack_get(obj_slot),
            "__reduce__",
        )? {
            Some(reduce) => call_fn(reduce, &[])?,
            None => return Err(pickling_error("Can't pickle object: no __reduce_ex__")),
        },
    };
    save_reduce_value(
        ctx,
        buf,
        pyre_object::gc_roots::shadow_stack_get(obj_slot),
        w_rv,
    )
}

/// CPython 3.14 `pickle._Pickler.save_type`, line by line.
fn save_type(ctx: &mut PickleCtx, buf: &mut Framer, w_obj: PyObjectRef) -> Result<(), PyError> {
    let singleton = [
        pyre_object::w_none(),
        pyre_object::special::w_not_implemented(),
        pyre_object::special::w_ellipsis(),
    ]
    .into_iter()
    .find(|&value| {
        pyre_interpreter::typedef::r#type(value)
            .is_some_and(|w_type| pyre_interpreter::baseobjspace::is_w(w_obj, w_type.as_ptr()))
    });
    let Some(singleton) = singleton else {
        return save_global(ctx, buf, w_obj, None);
    };

    let _roots = pyre_object::gc_roots::push_roots();
    let _ = pyre_object::gc_roots::pin_root(w_obj);
    let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let w_type = builtin_attr("type")?;
    let _ = pyre_object::gc_roots::pin_root(w_type);
    let type_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let mut fields = pyre_object::gc_roots::RootedItems::new();
    fields.push(singleton);
    let w_args = pyre_object::tupleobject::w_tuple_new(fields.take());
    let _ = pyre_object::gc_roots::pin_root(w_args);
    let args_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    save_reduce(
        ctx,
        buf,
        &[
            pyre_object::gc_roots::shadow_stack_get(type_slot),
            pyre_object::gc_roots::shadow_stack_get(args_slot),
        ],
        Some(pyre_object::gc_roots::shadow_stack_get(obj_slot)),
    )
}

/// Save the result of a reduce hook (`reducer_override` / `dispatch_table` /
/// `__reduce_ex__`): a `str` saves the object by global reference, a 2-to-6
/// element tuple drives `save_reduce`.
fn save_reduce_value(
    ctx: &mut PickleCtx,
    buf: &mut Framer,
    w_obj: PyObjectRef,
    w_rv: PyObjectRef,
) -> Result<(), PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let _ = pyre_object::gc_roots::pin_root(w_obj);
    let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let _ = pyre_object::gc_roots::pin_root(w_rv);
    let rv_slot = pyre_object::gc_roots::shadow_stack_len() - 1;

    let w_rv = pyre_object::gc_roots::shadow_stack_get(rv_slot);
    if unsafe { pyre_object::is_str(w_rv) } {
        return save_global(
            ctx,
            buf,
            pyre_object::gc_roots::shadow_stack_get(obj_slot),
            Some(w_rv),
        );
    }
    let result = if unsafe { pyre_object::is_tuple(w_rv) } {
        let n = unsafe { pyre_object::tupleobject::w_tuple_len(w_rv) };
        if !(2..=6).contains(&n) {
            Err(pickling_error(
                "tuple returned by __reduce__ must contain 2 through 6 elements",
            ))
        } else {
            let rv: Vec<PyObjectRef> = (0..n)
                .map(|i| unsafe {
                    pyre_object::tupleobject::w_tuple_getitem(
                        pyre_object::gc_roots::shadow_stack_get(rv_slot),
                        i as i64,
                    )
                    .unwrap()
                })
                .collect();
            save_reduce(
                ctx,
                buf,
                &rv,
                Some(pyre_object::gc_roots::shadow_stack_get(obj_slot)),
            )
        }
    } else {
        Err(pickling_error(format!(
            "__reduce__ must return a string or tuple, not {}",
            pyre_interpreter::error::type_name_of(w_rv)
        )))
    };
    result.map_err(|err| {
        add_pickle_object_note(
            err,
            pyre_object::gc_roots::shadow_stack_get(obj_slot),
            rustpython_wtf8::Wtf8::new("object"),
        )
    })
}

/// Look up `type(w_obj)` in the effective `dispatch_table` and, if registered,
/// call the reduce function. Returns its result, or `None` when no table or
/// no matching entry applies. The `copyreg.dispatch_table` fallback is a dict;
/// a user-set `dispatch_table` may be any mapping, so a non-dict is consulted
/// via `__getitem__` with `KeyError` meaning "no entry".
fn dispatch_table_reduce(
    ctx: &PickleCtx,
    w_obj: PyObjectRef,
) -> Result<Option<PyObjectRef>, PyError> {
    let dt = ctx.dispatch_table.get();
    if dt.is_null() {
        return Ok(None);
    }
    if unsafe { pyre_object::is_none(dt) } {
        // An explicit `dispatch_table = None` is still consulted by subscript;
        // `None[type(obj)]` raises, so surface that rather than silently
        // falling back to `__reduce_ex__`.
        return Err(PyError::type_error(
            "'NoneType' object is not subscriptable",
        ));
    }
    // The common case is an empty `copyreg.dispatch_table`; skip the lookup.
    if unsafe { pyre_object::is_dict(dt) }
        && unsafe { pyre_object::dictmultiobject::w_dict_len(dt) } == 0
    {
        return Ok(None);
    }
    let type_fn = crate::module::_pickle::lookup_builtin("type")
        .ok_or_else(|| pickling_error("type builtin unavailable"))?;
    let _roots = pyre_object::gc_roots::push_roots();
    let _ = pyre_object::gc_roots::pin_root(w_obj);
    let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let _ = pyre_object::gc_roots::pin_root(dt);
    let dt_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let w_type = call_fn(
        type_fn,
        &[pyre_object::gc_roots::shadow_stack_get(obj_slot)],
    )?;
    let _ = pyre_object::gc_roots::pin_root(w_type);
    let type_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let dt = pyre_object::gc_roots::shadow_stack_get(dt_slot);
    let w_type = pyre_object::gc_roots::shadow_stack_get(type_slot);
    let reduce_fn = if unsafe { pyre_object::is_dict(dt) } {
        unsafe { pyre_object::w_dict_lookup(dt, w_type) }
    } else {
        match pyre_interpreter::baseobjspace::getitem(dt, w_type) {
            Ok(reduce_fn) => Some(reduce_fn),
            Err(e) if e.kind == pyre_interpreter::PyErrorKind::KeyError => None,
            Err(e) => return Err(e),
        }
    };
    match reduce_fn {
        Some(reduce_fn) => Ok(Some(call_fn(
            reduce_fn,
            &[pyre_object::gc_roots::shadow_stack_get(obj_slot)],
        )?)),
        None => Ok(None),
    }
}

fn save_bool(ctx: &PickleCtx, buf: &mut Framer, w_obj: PyObjectRef) -> Result<(), PyError> {
    let truthy = pyre_interpreter::baseobjspace::is_true(w_obj)?;
    if ctx.proto >= 2 {
        buf.push(if truthy { op::NEWTRUE } else { op::NEWFALSE });
    } else {
        // I00\n / I01\n
        buf.extend_from_slice(if truthy { b"I01\n" } else { b"I00\n" });
    }
    Ok(())
}

fn save_long(ctx: &PickleCtx, buf: &mut Framer, mut w_obj: PyObjectRef) -> Result<(), PyError> {
    let small =
        pyre_object::with_roots!(w_obj => pyre_interpreter::baseobjspace::int_w(w_obj)).ok();
    let to_big = |v: Option<i64>| match v {
        Some(v) => BigInt::from(v),
        None => unsafe { pyre_interpreter::builtins::obj_to_bigint(w_obj) },
    };
    if ctx.bin
        && let Some(v) = small
    {
        if v >= 0 {
            if v <= 0xff {
                buf.push(op::BININT1);
                buf.push(v as u8);
                return Ok(());
            }
            if v <= 0xffff {
                buf.push(op::BININT2);
                buf.extend_from_slice(&(v as u16).to_le_bytes());
                return Ok(());
            }
        }
        if (-0x8000_0000..=0x7fff_ffff).contains(&v) {
            buf.push(op::BININT);
            buf.extend_from_slice(&(v as i32).to_le_bytes());
            return Ok(());
        }
    }
    if ctx.proto >= 2 {
        write_long(buf, &encode_long(&to_big(small)));
        return Ok(());
    }
    // protocol 0 / 1 text: INT for a signed 4-byte value, else LONG.
    if let Some(v) = small
        && (-0x8000_0000..=0x7fff_ffff).contains(&v)
    {
        buf.push(op::INT);
        buf.extend_from_slice(v.to_string().as_bytes());
        buf.push(b'\n');
        return Ok(());
    }
    buf.push(op::LONG);
    buf.extend_from_slice(to_big(small).to_string().as_bytes());
    buf.extend_from_slice(b"L\n");
    Ok(())
}

fn write_long(buf: &mut Framer, enc: &[u8]) {
    let n = enc.len();
    if n < 256 {
        buf.push(op::LONG1);
        buf.push(n as u8);
    } else {
        buf.push(op::LONG4);
        buf.extend_from_slice(&(n as i32).to_le_bytes());
    }
    buf.extend_from_slice(enc);
}

/// `interp_pickle.py _Framer.write_binfloat` is `@jit.unroll_safe`.
/// The 8-byte walk is `f.to_be_bytes()` here; the hint matches the decorator.
#[majit_macros::unroll_safe]
fn save_float(ctx: &PickleCtx, buf: &mut Framer, w_obj: PyObjectRef) -> Result<(), PyError> {
    if ctx.bin {
        let f = pyre_interpreter::baseobjspace::float_w(w_obj)?;
        // BINFLOAT — 8-byte big-endian IEEE 754.
        buf.push(op::BINFLOAT);
        buf.extend_from_slice(&f.to_be_bytes());
    } else {
        // proto 0: FLOAT + repr(obj) + '\n' (shortest round-trip text).
        let f = pyre_interpreter::baseobjspace::float_w(w_obj)?;
        buf.push(op::FLOAT);
        buf.extend_from_slice(pyre_interpreter::display::format_float_repr(f).as_bytes());
        buf.push(b'\n');
    }
    Ok(())
}

fn save_bytes(ctx: &mut PickleCtx, buf: &mut Framer, w_obj: PyObjectRef) -> Result<(), PyError> {
    // proto < 3 emits a `codecs.encode(s, 'latin1')` / `bytes()` reduce
    // instead of a BINBYTES opcode (interp_pickle.py:1349).
    if ctx.proto < 3 {
        let _roots = pyre_object::gc_roots::push_roots();
        let obj_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_obj);
        let data = unsafe {
            pyre_object::bytesobject::w_bytes_data(pyre_object::gc_roots::shadow_stack_get(
                obj_slot,
            ))
        }
        .to_vec();
        if data.is_empty() {
            let w_bytes =
                pyre_interpreter::typedef::gettypeobject(&pyre_object::bytesobject::BYTES_TYPE);
            let w_args = pyre_object::tupleobject::w_tuple_new(Vec::new());
            return save_reduce(
                ctx,
                buf,
                &[w_bytes, w_args],
                Some(pyre_object::gc_roots::shadow_stack_get(obj_slot)),
            );
        }
        let codecs = import_module("codecs")?;
        let encode_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(pyre_interpreter::baseobjspace::getattr_str(
            codecs, "encode",
        )?);
        let enc_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_new("latin1"));
        let w_decoded = call_meth(
            pyre_object::gc_roots::shadow_stack_get(obj_slot),
            "decode",
            &[pyre_object::gc_roots::shadow_stack_get(enc_slot)],
        )?;
        let dec_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_decoded);
        let w_args = pyre_object::tupleobject::w_tuple_new(vec![
            pyre_object::gc_roots::shadow_stack_get(dec_slot),
            pyre_object::gc_roots::shadow_stack_get(enc_slot),
        ]);
        return save_reduce(
            ctx,
            buf,
            &[pyre_object::gc_roots::shadow_stack_get(encode_slot), w_args],
            Some(pyre_object::gc_roots::shadow_stack_get(obj_slot)),
        );
    }
    // A large payload streams via `file.write` (arbitrary Python); pin `w_obj`
    // so the trailing `memoize` reads it at its post-write address, and copy
    // the payload off it before later allocs.
    let _roots = pyre_object::gc_roots::push_roots();
    let _ = pyre_object::gc_roots::pin_root(w_obj);
    let slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let data = unsafe {
        pyre_object::bytesobject::w_bytes_data(pyre_object::gc_roots::shadow_stack_get(slot))
    }
    .to_vec();
    let n = data.len();
    if n <= 0xff {
        buf.push(op::SHORT_BINBYTES);
        buf.push(n as u8);
        buf.extend_from_slice(&data);
    } else if n > 0xffff_ffff && ctx.proto >= 4 {
        let mut header = vec![op::BINBYTES8];
        header.extend_from_slice(&(n as u64).to_le_bytes());
        buf.write_large_bytes(&header, &data)?;
    } else if n >= FRAME_SIZE_TARGET {
        let mut header = vec![op::BINBYTES];
        header.extend_from_slice(&(n as u32).to_le_bytes());
        buf.write_large_bytes(&header, &data)?;
    } else {
        buf.push(op::BINBYTES);
        buf.extend_from_slice(&(n as u32).to_le_bytes());
        buf.extend_from_slice(&data);
    }
    memoize(ctx, buf, pyre_object::gc_roots::shadow_stack_get(slot));
    Ok(())
}

fn save_str(ctx: &mut PickleCtx, buf: &mut Framer, w_obj: PyObjectRef) -> Result<(), PyError> {
    // A large payload streams via `file.write` (arbitrary Python); pin `w_obj`
    // so the trailing `memoize` reads it at its post-write address.
    let _roots = pyre_object::gc_roots::push_roots();
    let w_obj = pyre_object::gc_roots::pin_root(w_obj);
    let slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    if ctx.bin {
        // The binary opcodes carry the string's `surrogatepass` UTF-8, which is
        // exactly the backing WTF-8, so the raw buffer is written as-is and a
        // lone surrogate round-trips instead of demanding a `&str` view.
        let data = unsafe { pyre_object::unicodeobject::w_str_get_wtf8(w_obj) }.as_bytes();
        let n = data.len();
        if n <= 0xff && ctx.proto >= 4 {
            buf.push(op::SHORT_BINUNICODE);
            buf.push(n as u8);
            buf.extend_from_slice(data);
        } else if n > 0xffff_ffff && ctx.proto >= 4 {
            let mut header = vec![op::BINUNICODE8];
            header.extend_from_slice(&(n as u64).to_le_bytes());
            buf.write_large_bytes(&header, data)?;
        } else if n >= FRAME_SIZE_TARGET {
            let mut header = vec![op::BINUNICODE];
            header.extend_from_slice(&(n as u32).to_le_bytes());
            buf.write_large_bytes(&header, data)?;
        } else {
            buf.push(op::BINUNICODE);
            buf.extend_from_slice(&(n as u32).to_le_bytes());
            buf.extend_from_slice(data);
        }
    } else {
        // proto 0: UNICODE + raw-unicode-escape. The codec leaves
        // backslash / NUL / newline / CR / EOF-on-DOS literal, so escape
        // those first; the load side reverses with raw-unicode-escape.
        let mut w_tmp_slot = slot;
        for (from, to) in [
            ("\\", "\\u005c"),
            ("\0", "\\u0000"),
            ("\n", "\\u000a"),
            ("\r", "\\u000d"),
            ("\u{1a}", "\\u001a"),
        ] {
            let from_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_new_managed(from));
            let to_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_new_managed(to));
            let replaced = call_meth(
                pyre_object::gc_roots::shadow_stack_get(w_tmp_slot),
                "replace",
                &[
                    pyre_object::gc_roots::shadow_stack_get(from_slot),
                    pyre_object::gc_roots::shadow_stack_get(to_slot),
                ],
            )?;
            w_tmp_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(replaced);
        }
        let encoding_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ =
            pyre_object::gc_roots::pin_root(pyre_object::w_str_new_managed("raw-unicode-escape"));
        let w_enc = call_meth(
            pyre_object::gc_roots::shadow_stack_get(w_tmp_slot),
            "encode",
            &[pyre_object::gc_roots::shadow_stack_get(encoding_slot)],
        )?;
        let data = unsafe { pyre_object::bytesobject::w_bytes_data(w_enc) };
        buf.push(op::UNICODE);
        buf.extend_from_slice(data);
        buf.push(b'\n');
    }
    memoize(ctx, buf, pyre_object::gc_roots::shadow_stack_get(slot));
    Ok(())
}

/// `interp_pickle.py save_tuple`.
fn save_tuple(ctx: &mut PickleCtx, buf: &mut Framer, w_obj: PyObjectRef) -> Result<(), PyError> {
    let n = unsafe { pyre_object::tupleobject::w_tuple_len(w_obj) };
    if n == 0 {
        if ctx.bin {
            buf.push(op::EMPTY_TUPLE);
        } else {
            buf.push(op::MARK);
            buf.push(op::TUPLE);
        }
        return Ok(());
    }

    // Pin the tuple; a recursive save below can relocate the elements, so
    // re-read each one (and the tuple itself for the memo) from the
    // GC-walked tuple right before it is used.
    let _roots = pyre_object::gc_roots::push_roots();
    let _ = pyre_object::gc_roots::pin_root(w_obj);
    let slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let elem = |i: usize| unsafe {
        pyre_object::tupleobject::w_tuple_getitem(
            pyre_object::gc_roots::shadow_stack_get(slot),
            i as i64,
        )
        .unwrap()
    };

    if n <= 3 && ctx.proto >= 2 {
        for i in 0..n {
            if let Err(err) = save(ctx, buf, elem(i)) {
                return Err(add_pickle_object_note(
                    err,
                    pyre_object::gc_roots::shadow_stack_get(slot),
                    rustpython_wtf8::Wtf8::new(format!("item {i}").as_str()),
                ));
            }
        }
        // Subtle: saving the elements may have memoized this very tuple
        // (a recursive tuple). If so, discard the elements and GET it.
        if let Some(idx) = ctx.memo_get(pyre_object::gc_roots::shadow_stack_get(slot)) {
            for _ in 0..n {
                buf.push(op::POP);
            }
            write_get(ctx, buf, idx);
        } else {
            buf.push(op::TUPLESIZE2CODE[n]);
            memoize(ctx, buf, pyre_object::gc_roots::shadow_stack_get(slot));
        }
        return Ok(());
    }

    buf.push(op::MARK);
    for i in 0..n {
        if let Err(err) = save(ctx, buf, elem(i)) {
            return Err(add_pickle_object_note(
                err,
                pyre_object::gc_roots::shadow_stack_get(slot),
                rustpython_wtf8::Wtf8::new(format!("item {i}").as_str()),
            ));
        }
    }
    if let Some(idx) = ctx.memo_get(pyre_object::gc_roots::shadow_stack_get(slot)) {
        // Recursive tuple: throw away the stack contents and GET it.
        if ctx.bin {
            buf.push(op::POP_MARK);
        } else {
            for _ in 0..(n + 1) {
                buf.push(op::POP);
            }
        }
        write_get(ctx, buf, idx);
        return Ok(());
    }
    buf.push(op::TUPLE);
    memoize(ctx, buf, pyre_object::gc_roots::shadow_stack_get(slot));
    Ok(())
}

/// `interp_pickle.py save_list`. The PyPy ascii/bytes-list fast paths are
/// gated on `pypy_extensions` (off here) so the generic path is used,
/// matching CPython's wire format.
fn save_list(ctx: &mut PickleCtx, buf: &mut Framer, w_obj: PyObjectRef) -> Result<(), PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    // The list itself is a GC-walked Python `list`; pin it and append by
    // re-reading each element, so a relocation during a recursive save is
    // observed instead of dereferencing a stale snapshot.
    let _ = pyre_object::gc_roots::pin_root(w_obj);
    let slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let fast_token = fast_save_enter(ctx, slot)?;
    if ctx.bin {
        buf.push(op::EMPTY_LIST);
    } else {
        buf.push(op::MARK);
        buf.push(op::LIST);
    }
    memoize(ctx, buf, pyre_object::gc_roots::shadow_stack_get(slot));
    let result = batch_appends(ctx, buf, slot, Some(slot));
    fast_save_leave(ctx, fast_token, slot);
    result
}

/// `interp_pickle.py save_dict`.
fn save_dict(ctx: &mut PickleCtx, buf: &mut Framer, w_obj: PyObjectRef) -> Result<(), PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    // Pin the dict so `memoize` sees its current address. PyPy passes the live
    // `dict.items()` view to `_batch_setitems`; retaining that iterator shape
    // is what detects mutation from a persistent-id callback.
    let _ = pyre_object::gc_roots::pin_root(w_obj);
    let dict_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let fast_token = fast_save_enter(ctx, dict_slot)?;
    if ctx.bin {
        buf.push(op::EMPTY_DICT);
    } else {
        buf.push(op::MARK);
        buf.push(op::DICT);
    }
    memoize(ctx, buf, pyre_object::gc_roots::shadow_stack_get(dict_slot));
    let w_items = call_meth(
        pyre_object::gc_roots::shadow_stack_get(dict_slot),
        "items",
        &[],
    )?;
    let _ = pyre_object::gc_roots::pin_root(w_items);
    let items_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let result = batch_setitems(ctx, buf, items_slot, Some(dict_slot));
    fast_save_leave(ctx, fast_token, dict_slot);
    result
}

/// `interp_pickle.py save_set`. Sets are unordered, so the wire bytes are
/// not byte-identical to CPython, but the encoding round-trips. The
/// protocol < 4 reduce fallback arrives with `save_reduce`.
fn save_set(ctx: &mut PickleCtx, buf: &mut Framer, w_obj: PyObjectRef) -> Result<(), PyError> {
    if ctx.proto < 4 {
        // save_reduce(set, (list(obj),)).
        // `w_list_new` collects, so publish the set and every already-live
        // member first and rebuild the list from the forwarded slots.
        let _roots = pyre_object::gc_roots::push_roots();
        let items = unsafe { pyre_object::setobject::w_set_items(w_obj) };
        let mut live = Vec::with_capacity(items.len() + 1);
        live.push(w_obj);
        live.extend_from_slice(&items);
        let base = pyre_object::gc_roots::publish_roots(&live);
        pyre_object::gc_roots::normalize_roots(base, live.len());
        let reloaded: Vec<_> = (0..items.len())
            .map(|i| pyre_object::gc_roots::shadow_stack_get(base + 1 + i))
            .collect();
        let list_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(pyre_object::listobject::w_list_new(reloaded));
        let w_args =
            pyre_object::tupleobject::w_tuple_new(vec![pyre_object::gc_roots::shadow_stack_get(
                list_slot,
            )]);
        let w_set_type =
            pyre_interpreter::typedef::gettypeobject(&pyre_object::setobject::SET_TYPE);
        return save_reduce(
            ctx,
            buf,
            &[w_set_type, w_args],
            Some(pyre_object::gc_roots::shadow_stack_get(base)),
        );
    }
    buf.push(op::EMPTY_SET);
    // Pin the set so `memoize` records its current address, then snapshot its
    // members into a pinned Python `list` re-read per save (a recursive save
    // can relocate them).
    let _roots = pyre_object::gc_roots::push_roots();
    let _ = pyre_object::gc_roots::pin_root(w_obj);
    let set_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let fast_token = fast_save_enter(ctx, set_slot)?;
    memoize(ctx, buf, pyre_object::gc_roots::shadow_stack_get(set_slot));

    let items = unsafe {
        pyre_object::setobject::w_set_items(pyre_object::gc_roots::shadow_stack_get(set_slot))
    };
    let items_slot = pin_published_items(items);
    let result = save_set_items(ctx, buf, items_slot, set_slot);
    fast_save_leave(ctx, fast_token, set_slot);
    result
}

/// Emit the ADDITEMS batches for the members of a proto >= 4 set already opened
/// with EMPTY_SET. `items_slot` pins the member snapshot, re-read per save.
fn save_set_items(
    ctx: &mut PickleCtx,
    buf: &mut Framer,
    items_slot: usize,
    obj_slot: usize,
) -> Result<(), PyError> {
    let length = pinned_len(items_slot);
    if length == 0 {
        return Ok(());
    }
    buf.push(op::MARK);
    if let Err(err) = save(ctx, buf, pinned_get(items_slot, 0)) {
        return Err(add_pickle_object_note(
            err,
            pyre_object::gc_roots::shadow_stack_get(obj_slot),
            rustpython_wtf8::Wtf8::new("element"),
        ));
    }
    let mut i = 1;
    while i + 1 < length {
        if i % BATCHSIZE == 0 {
            buf.push(op::ADDITEMS);
            buf.push(op::MARK);
        }
        if let Err(err) = save(ctx, buf, pinned_get(items_slot, i)) {
            return Err(add_pickle_object_note(
                err,
                pyre_object::gc_roots::shadow_stack_get(obj_slot),
                rustpython_wtf8::Wtf8::new("element"),
            ));
        }
        i += 1;
    }
    if length > 1
        && let Err(err) = save(ctx, buf, pinned_get(items_slot, length - 1))
    {
        return Err(add_pickle_object_note(
            err,
            pyre_object::gc_roots::shadow_stack_get(obj_slot),
            rustpython_wtf8::Wtf8::new("element"),
        ));
    }
    buf.push(op::ADDITEMS);
    Ok(())
}

/// `interp_pickle.py save_frozenset`. Protocol < 4 reduces to
/// `frozenset(list(obj))`; protocol >= 4 uses the FROZENSET opcode.
/// Unordered, so not byte-identical to CPython.
fn save_frozenset(
    ctx: &mut PickleCtx,
    buf: &mut Framer,
    w_obj: PyObjectRef,
) -> Result<(), PyError> {
    if ctx.proto < 4 {
        // save_reduce(frozenset, (list(obj),)).
        let _roots = pyre_object::gc_roots::push_roots();
        let items = unsafe { pyre_object::setobject::w_set_items(w_obj) };
        let mut live = Vec::with_capacity(items.len() + 1);
        live.push(w_obj);
        live.extend_from_slice(&items);
        let base = pyre_object::gc_roots::publish_roots(&live);
        pyre_object::gc_roots::normalize_roots(base, live.len());
        let reloaded: Vec<_> = (0..items.len())
            .map(|i| pyre_object::gc_roots::shadow_stack_get(base + 1 + i))
            .collect();
        let list_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(pyre_object::listobject::w_list_new(reloaded));
        let w_args =
            pyre_object::tupleobject::w_tuple_new(vec![pyre_object::gc_roots::shadow_stack_get(
                list_slot,
            )]);
        let w_frozenset_type =
            pyre_interpreter::typedef::gettypeobject(&pyre_object::setobject::FROZENSET_TYPE);
        return save_reduce(
            ctx,
            buf,
            &[w_frozenset_type, w_args],
            Some(pyre_object::gc_roots::shadow_stack_get(base)),
        );
    }
    // Pin the frozenset and snapshot its members into a pinned Python `list`
    // re-read per save (a recursive save can relocate them); the frozenset
    // itself is re-read for the memo check after the saves.
    let _roots = pyre_object::gc_roots::push_roots();
    let _ = pyre_object::gc_roots::pin_root(w_obj);
    let fs_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let items = unsafe {
        pyre_object::setobject::w_set_items(pyre_object::gc_roots::shadow_stack_get(fs_slot))
    };
    let slot = pin_published_items(items);
    buf.push(op::MARK);
    let n = pinned_len(slot);
    for i in 0..n {
        if let Err(err) = save(ctx, buf, pinned_get(slot, i)) {
            return Err(add_pickle_object_note(
                err,
                pyre_object::gc_roots::shadow_stack_get(fs_slot),
                rustpython_wtf8::Wtf8::new("element"),
            ));
        }
    }
    if let Some(idx) = ctx.memo_get(pyre_object::gc_roots::shadow_stack_get(fs_slot)) {
        buf.push(op::POP);
        write_get(ctx, buf, idx);
    } else {
        buf.push(op::FROZENSET);
        memoize(ctx, buf, pyre_object::gc_roots::shadow_stack_get(fs_slot));
    }
    Ok(())
}

/// Copy a pinned bytearray's payload. `vec` allocation can collect, so the
/// slice is taken from `slot` again after the alloc.
fn copy_pinned_bytearray(slot: usize) -> Vec<u8> {
    let n = unsafe {
        pyre_object::bytearrayobject::w_bytearray_data(pyre_object::gc_roots::shadow_stack_get(
            slot,
        ))
        .len()
    };
    let mut owned = vec![0u8; n];
    owned.copy_from_slice(unsafe {
        pyre_object::bytearrayobject::w_bytearray_data(pyre_object::gc_roots::shadow_stack_get(
            slot,
        ))
    });
    owned
}

/// `interp_pickle.py save_bytearray` (proto >= 5 raw form; lower protocols
/// reach the generic reduce path).
fn save_bytearray(
    ctx: &mut PickleCtx,
    buf: &mut Framer,
    w_obj: PyObjectRef,
) -> Result<(), PyError> {
    if ctx.proto < 5 {
        // save_reduce(bytearray, ()) for empty, else save_reduce(bytearray, (bytes,)).
        // The bytearray is nursery-movable and `w_bytes_from_bytes` /
        // `w_tuple_new` collect, so pin it before those allocs and copy
        // the payload off the object before the bytes constructor runs.
        let _roots = pyre_object::gc_roots::push_roots();
        let obj_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_obj);
        let w_bytearray_type =
            pyre_interpreter::typedef::gettypeobject(&pyre_object::bytearrayobject::BYTEARRAY_TYPE);
        let owned = copy_pinned_bytearray(obj_slot);
        let w_args = if owned.is_empty() {
            pyre_object::tupleobject::w_tuple_new(Vec::new())
        } else {
            let bytes_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = pyre_object::gc_roots::pin_root(pyre_object::w_bytes_from_bytes(&owned));
            pyre_object::tupleobject::w_tuple_new(vec![pyre_object::gc_roots::shadow_stack_get(
                bytes_slot,
            )])
        };
        return save_reduce(
            ctx,
            buf,
            &[w_bytearray_type, w_args],
            Some(pyre_object::gc_roots::shadow_stack_get(obj_slot)),
        );
    }
    // interp_pickle.py save_bytearray: `view.as_str()` copies the payload
    // before BYTEARRAY8 / memoize. Holding `w_bytearray_data` across
    // `pin_root` leaves a nursery slice dangling after the object moves.
    let _roots = pyre_object::gc_roots::push_roots();
    let _ = pyre_object::gc_roots::pin_root(w_obj);
    let slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let owned = copy_pinned_bytearray(slot);
    save_raw_bytearray(buf, &owned)?;
    memoize(ctx, buf, pyre_object::gc_roots::shadow_stack_get(slot));
    Ok(())
}

/// `interp_pickle.py save_picklebuffer` — serialize a `PickleBuffer`. With
/// no `buffer_callback`, or a callback returning a true value, the contents
/// are written in-band (BINBYTES for a read-only buffer, BYTEARRAY8 for a
/// mutable one). A callback returning a false value writes the data
/// out-of-band: NEXT_BUFFER, plus READONLY_BUFFER for a read-only buffer.
fn save_picklebuffer(
    ctx: &mut PickleCtx,
    buf: &mut Framer,
    w_obj: PyObjectRef,
) -> Result<(), PyError> {
    if ctx.proto < 5 {
        return Err(pickling_error(
            "PickleBuffer can only be pickled with protocol >= 5",
        ));
    }
    // Read the wrapped object out of the buffer, then drop the borrow before
    // any allocation (the callback below) can relocate the wrapper.
    let wrapped = {
        let pb = pyre_interpreter::module::__pypy__::W_PickleBuffer::from_obj(w_obj)
            .ok_or_else(|| pickling_error("save_picklebuffer: not a PickleBuffer"))?;
        pb.wrapped()
    };
    if unsafe { pyre_object::is_none(wrapped) } {
        return Err(pickling_error(
            "PickleBuffer can not be pickled after release",
        ));
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[w_obj, wrapped]);
    let obj_slot = base;
    let wrapped_slot = base + 1;
    if !pyre_interpreter::module::__pypy__::interp_buffer::is_contiguous(
        pyre_object::gc_roots::shadow_stack_get(wrapped_slot),
    )? {
        return Err(pickling_error(
            "PickleBuffer can not be pickled when pointing to a non-contiguous buffer",
        ));
    }
    let (data, readonly) = pyre_interpreter::module::__pypy__::interp_buffer::buffer_view(
        pyre_object::gc_roots::shadow_stack_get(wrapped_slot),
    )?;
    let mut in_band = true;
    let buffer_callback = ctx.buffer_callback.get();
    if !unsafe { pyre_object::is_none(buffer_callback) } {
        let w_ret = call_fn(
            buffer_callback,
            &[pyre_object::gc_roots::shadow_stack_get(obj_slot)],
        )?;
        in_band = pyre_interpreter::baseobjspace::is_true(w_ret)?;
    }
    if in_band {
        // In-band buffers memoize the wrapper (`_save_bytes_data` /
        // `_save_bytearray_data`), so a repeated reference becomes a GET.
        if readonly {
            save_raw_bytes(ctx, buf, &data)?;
        } else {
            save_raw_bytearray(buf, &data)?;
        }
        memoize(ctx, buf, pyre_object::gc_roots::shadow_stack_get(obj_slot));
    } else {
        buf.push(op::NEXT_BUFFER);
        if readonly {
            buf.push(op::READONLY_BUFFER);
        }
    }
    Ok(())
}

/// `interp_pickle.py save_raw_bytes` — emit raw bytes with the size-appropriate
/// BINBYTES opcode (no memoization). `data` is owned storage, so the streaming
/// `file.write` in `write_large_bytes` cannot dangle it; the caller pins any
/// object it still needs afterwards.
fn save_raw_bytes(ctx: &PickleCtx, buf: &mut Framer, data: &[u8]) -> Result<(), PyError> {
    let n = data.len();
    if n <= 0xff {
        buf.push(op::SHORT_BINBYTES);
        buf.push(n as u8);
        buf.extend_from_slice(data);
    } else if n > 0xffff_ffff && ctx.proto >= 4 {
        let mut header = vec![op::BINBYTES8];
        header.extend_from_slice(&(n as u64).to_le_bytes());
        buf.write_large_bytes(&header, data)?;
    } else if n >= FRAME_SIZE_TARGET {
        let mut header = vec![op::BINBYTES];
        header.extend_from_slice(&(n as u32).to_le_bytes());
        buf.write_large_bytes(&header, data)?;
    } else {
        buf.push(op::BINBYTES);
        buf.extend_from_slice(&(n as u32).to_le_bytes());
        buf.extend_from_slice(data);
    }
    Ok(())
}

/// `interp_pickle.py save_raw_bytearray` — emit raw bytes with BYTEARRAY8
/// (no memoization).
fn save_raw_bytearray(buf: &mut Framer, data: &[u8]) -> Result<(), PyError> {
    let n = data.len();
    if n >= FRAME_SIZE_TARGET {
        let mut header = vec![op::BYTEARRAY8];
        header.extend_from_slice(&(n as u64).to_le_bytes());
        buf.write_large_bytes(&header, data)?;
    } else {
        buf.push(op::BYTEARRAY8);
        buf.extend_from_slice(&(n as u64).to_le_bytes());
        buf.extend_from_slice(data);
    }
    Ok(())
}

/// Build a Python `list` from `items` and pin it in the shadow stack,
/// returning its slot.  `w_list_new_object` pins each element across its own
/// allocation, so the snapshot is captured safely; thereafter the GC walks
/// the list and rewrites its entries, so `pinned_get` reads the relocated
/// element even after the recursive `save` calls below trigger collections.
///
/// The Object strategy is what makes `pinned_get` answer the SAME object it
/// was handed: an unboxing strategy stores the payload and wraps a fresh
/// object per read, and `memo_get` resolves its hash bucket by pointer
/// identity, so a container of unboxable elements would miss the memo and
/// write each element again instead of a GET.
fn pin_items(items: Vec<PyObjectRef>) -> usize {
    let w_list = pyre_object::listobject::w_list_new_object(items);
    let _ = pyre_object::gc_roots::pin_root(w_list);
    pyre_object::gc_roots::shadow_stack_len() - 1
}

/// Publish an already-live member slice, then [`pin_items`].  The list
/// constructor is a safepoint, so handing `pin_items` an unpublished
/// `Vec` would freeze forwarding stubs into the snapshot.
fn pin_published_items(items: Vec<PyObjectRef>) -> usize {
    let base = pyre_object::gc_roots::publish_roots(&items);
    pyre_object::gc_roots::normalize_roots(base, items.len());
    let reloaded: Vec<_> = (0..items.len())
        .map(|i| pyre_object::gc_roots::shadow_stack_get(base + i))
        .collect();
    pin_items(reloaded)
}

/// Length of the pinned list at `slot`.
fn pinned_len(slot: usize) -> usize {
    let list = pyre_object::gc_roots::shadow_stack_get(slot);
    unsafe { pyre_object::listobject::w_list_len(list) }
}

/// Element `i` of the pinned list at `slot`, re-read so a relocation of the
/// element (or the list) since the last access is observed.
fn pinned_get(slot: usize, i: usize) -> PyObjectRef {
    let list = pyre_object::gc_roots::shadow_stack_get(slot);
    unsafe { pyre_object::listobject::w_list_getitem(list, i as i64) }.unwrap()
}

/// Advance the iterator pinned at `iter_slot`. A yielded object is pinned
/// before returning because `_batch_appends` fetches its second item before
/// saving the first, exactly like `interp_pickle.py`; the first save
/// may run arbitrary Python and remove the second item from the source list.
fn pinned_iter_next(iter_slot: usize) -> Result<Option<usize>, PyError> {
    let next_item =
        pyre_interpreter::baseobjspace::next(pyre_object::gc_roots::shadow_stack_get(iter_slot));
    let err = match next_item {
        Ok(item) => {
            let _ = pyre_object::gc_roots::pin_root(item);
            return Ok(Some(pyre_object::gc_roots::shadow_stack_len() - 1));
        }
        Err(e) => e,
    };
    let (stop, err) = err.matches_stop_iteration_keep();
    if stop { Ok(None) } else { Err(err) }
}

/// Snapshot the iterable pinned at `source_slot` into a GC-walked list.
///
/// This is `space.listview(w_list)` from PyPy's protocol-0 `_batch_appends`
/// path. Building the snapshot before calling `save` is observable when a
/// persistent-id hook mutates the source collection.
fn snapshot_pinned_iterable(source_slot: usize) -> Result<usize, PyError> {
    let snapshot_slot = pin_items(Vec::new());
    let w_iter =
        pyre_interpreter::baseobjspace::iter(pyre_object::gc_roots::shadow_stack_get(source_slot))?;
    let _ = pyre_object::gc_roots::pin_root(w_iter);
    let iter_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    loop {
        let next_item = pyre_interpreter::baseobjspace::next(
            pyre_object::gc_roots::shadow_stack_get(iter_slot),
        );
        let mut stop_err = None;
        let item = match next_item {
            Ok(item) => Some(item),
            Err(e) => {
                stop_err = Some(e);
                None
            }
        };
        if let Some(e) = stop_err {
            let (stop, e) = e.matches_stop_iteration_keep();
            if stop {
                break;
            }
            return Err(e);
        }
        let item = item.unwrap();
        {
            let _item_root = pyre_object::gc_roots::push_roots();
            let _ = pyre_object::gc_roots::pin_root(item);
            let item_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            unsafe {
                pyre_object::listobject::w_list_append(
                    pyre_object::gc_roots::shadow_stack_get(snapshot_slot),
                    pyre_object::gc_roots::shadow_stack_get(item_slot),
                );
            }
        }
    }
    Ok(snapshot_slot)
}

/// `interp_pickle.py _batch_appends`.
///
/// Protocol 0 consumes `space.listview(w_list)`, a pre-save snapshot. Binary
/// protocols keep a live iterator over the original list, so removals can end
/// iteration and appends can extend it. Do not replace this with a fixed
/// length plus indexed reads: `persistent_id` may mutate the list while an
/// item is being saved (CPython gh-92930).
fn batch_appends(
    ctx: &mut PickleCtx,
    buf: &mut Framer,
    slot: usize,
    obj_slot: Option<usize>,
) -> Result<(), PyError> {
    if !ctx.bin {
        // proto 0 — no APPENDS, one APPEND per item. PyPy snapshots through
        // space.listview before the first recursive save.
        let snapshot_slot = snapshot_pinned_iterable(slot)?;
        let n = pinned_len(snapshot_slot);
        for i in 0..n {
            if let Err(err) = save(ctx, buf, pinned_get(snapshot_slot, i)) {
                return Err(add_reduce_note(
                    err,
                    obj_slot,
                    rustpython_wtf8::Wtf8::new(format!("item {i}").as_str()),
                ));
            }
            buf.push(op::APPEND);
        }
        return Ok(());
    }

    // `_batch_appends` keeps `w_it` as a local of this operation.
    // `ShadowStackFrameworkGCTransformer.push_roots` already covers the
    // caller's live set; pin the iterator there. The inner batch bracket
    // matches locals that die at the end of a batch.
    let w_iter =
        pyre_interpreter::baseobjspace::iter(pyre_object::gc_roots::shadow_stack_get(slot))?;
    let _ = pyre_object::gc_roots::pin_root(w_iter);
    let iter_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let mut index = 0usize;

    loop {
        // Keep at most one batch of yielded items pinned. The iterator itself
        // is below this guard and remains pinned for the whole function.
        let _batch_roots = pyre_object::gc_roots::push_roots();
        let first_slot = match pinned_iter_next(iter_slot)? {
            Some(slot) => slot,
            None => return Ok(()),
        };
        let mut item_slot = match pinned_iter_next(iter_slot)? {
            Some(slot) => slot,
            None => {
                if let Err(err) = save(
                    ctx,
                    buf,
                    pyre_object::gc_roots::shadow_stack_get(first_slot),
                ) {
                    return Err(add_reduce_note(
                        err,
                        obj_slot,
                        rustpython_wtf8::Wtf8::new(format!("item {index}").as_str()),
                    ));
                }
                buf.push(op::APPEND);
                return Ok(());
            }
        };

        buf.push(op::MARK);
        if let Err(err) = save(
            ctx,
            buf,
            pyre_object::gc_roots::shadow_stack_get(first_slot),
        ) {
            return Err(add_reduce_note(
                err,
                obj_slot,
                rustpython_wtf8::Wtf8::new(format!("item {index}").as_str()),
            ));
        }
        index += 1;
        let mut count = 1;
        loop {
            if let Err(err) = save(ctx, buf, pyre_object::gc_roots::shadow_stack_get(item_slot)) {
                return Err(add_reduce_note(
                    err,
                    obj_slot,
                    rustpython_wtf8::Wtf8::new(format!("item {index}").as_str()),
                ));
            }
            index += 1;
            count += 1;
            if count == BATCHSIZE {
                break;
            }
            item_slot = match pinned_iter_next(iter_slot)? {
                Some(slot) => slot,
                None => {
                    buf.push(op::APPENDS);
                    return Ok(());
                }
            };
        }
        buf.push(op::APPENDS);
    }
}

/// Advance a dict-items iterator and pin its unpacked `(key, value)` pair in a
/// two-element GC-walked list.
fn pinned_pair_next(iter_slot: usize) -> Result<Option<usize>, PyError> {
    let next_item =
        pyre_interpreter::baseobjspace::next(pyre_object::gc_roots::shadow_stack_get(iter_slot));
    let mut stop_err = None;
    let item = match next_item {
        Ok(item) => Some(item),
        Err(e) => {
            stop_err = Some(e);
            None
        }
    };
    if let Some(e) = stop_err {
        let (stop, e) = e.matches_stop_iteration_keep();
        return if stop { Ok(None) } else { Err(e) };
    }
    let item = item.unwrap();
    let _ = pyre_object::gc_roots::pin_root(item);
    let item_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let item = pyre_object::gc_roots::shadow_stack_get(item_slot);
    if !unsafe { pyre_object::is_tuple(item) }
        || unsafe { pyre_object::tupleobject::w_tuple_len(item) } != 2
    {
        return Err(PyError::type_error(
            "dict items iterator must return 2-tuples",
        ));
    }
    let pair = vec![
        unsafe { pyre_object::tupleobject::w_tuple_getitem(item, 0) }.unwrap(),
        unsafe { pyre_object::tupleobject::w_tuple_getitem(item, 1) }.unwrap(),
    ];
    Ok(Some(pin_items(pair)))
}

/// Save one `(key, value)` item and attach pickle.py's value context. Keys are
/// deliberately outside the note wrapper: `_batch_setitems` only annotates a
/// failure while serializing the value, using the already-saved key's repr.
fn save_pair(
    ctx: &mut PickleCtx,
    buf: &mut Framer,
    pair_slot: usize,
    obj_slot: Option<usize>,
) -> Result<(), PyError> {
    save(ctx, buf, pinned_get(pair_slot, 0))?;
    match save(ctx, buf, pinned_get(pair_slot, 1)) {
        Ok(()) => Ok(()),
        Err(err) => {
            // pickle.py only invokes the key's arbitrary __repr__ while
            // annotating a value-save failure. Successful dictionary saves
            // must not gain an observable repr call.
            let _roots = pyre_object::gc_roots::push_roots();
            let mut err = err;
            let slot = err.pin(&_roots);
            let key_repr =
                unsafe { pyre_interpreter::display::py_repr_wtf8(pinned_get(pair_slot, 0)) };
            err.reload(&_roots, slot);
            let key_repr = key_repr?;
            Err(add_reduce_note(
                err,
                obj_slot,
                &pyre_interpreter::display::wtf8_format!("item ", key_repr),
            ))
        }
    }
}

/// `interp_pickle.py _batch_setitems`.
///
/// The iterable remains live while its pairs are saved. In particular a
/// `dict_items` iterator raises its normal "dictionary changed size during
/// iteration" error if a persistent-id hook mutates the source dictionary.
fn batch_setitems(
    ctx: &mut PickleCtx,
    buf: &mut Framer,
    slot: usize,
    obj_slot: Option<usize>,
) -> Result<(), PyError> {
    // `_batch_setitems` keeps `w_it` as a local of this operation.
    // `ShadowStackFrameworkGCTransformer.push_roots` already covers the
    // caller's live set; pin the iterator there. The inner pair and batch
    // brackets match locals that die at the end of a batch.
    let w_iter =
        pyre_interpreter::baseobjspace::iter(pyre_object::gc_roots::shadow_stack_get(slot))?;
    let _ = pyre_object::gc_roots::pin_root(w_iter);
    let iter_slot = pyre_object::gc_roots::shadow_stack_len() - 1;

    if !ctx.bin {
        // proto 0 — no SETITEMS, one SETITEM per pair.
        loop {
            let _pair_roots = pyre_object::gc_roots::push_roots();
            let pair_slot = match pinned_pair_next(iter_slot)? {
                Some(slot) => slot,
                None => return Ok(()),
            };
            save_pair(ctx, buf, pair_slot, obj_slot)?;
            buf.push(op::SETITEM);
        }
    }

    loop {
        let _batch_roots = pyre_object::gc_roots::push_roots();
        let first_slot = match pinned_pair_next(iter_slot)? {
            Some(slot) => slot,
            None => return Ok(()),
        };
        let mut pair_slot = match pinned_pair_next(iter_slot)? {
            Some(slot) => slot,
            None => {
                save_pair(ctx, buf, first_slot, obj_slot)?;
                buf.push(op::SETITEM);
                return Ok(());
            }
        };

        buf.push(op::MARK);
        save_pair(ctx, buf, first_slot, obj_slot)?;
        let mut count = 1;
        loop {
            save_pair(ctx, buf, pair_slot, obj_slot)?;
            count += 1;
            if count == BATCHSIZE {
                break;
            }
            pair_slot = match pinned_pair_next(iter_slot)? {
                Some(slot) => slot,
                None => {
                    buf.push(op::SETITEMS);
                    return Ok(());
                }
            };
        }
        buf.push(op::SETITEMS);
    }
}

/// `interp_pickle.py W_Pickler.write_get` — emit a GET back-reference.
fn write_get(ctx: &PickleCtx, buf: &mut Framer, idx: usize) {
    if ctx.bin {
        if idx < 256 {
            buf.push(op::BINGET);
            buf.push(idx as u8);
        } else {
            buf.push(op::LONG_BINGET);
            buf.extend_from_slice(&(idx as u32).to_le_bytes());
        }
    } else {
        buf.push(op::GET);
        buf.extend_from_slice(format!("{idx}\n").as_bytes());
    }
}

/// `interp_pickle.py memoize` — record the object's identity and write the
/// put opcode. A no-op in `fast` mode (no PUT/GET, no recursion guard).
fn memoize(ctx: &mut PickleCtx, buf: &mut Framer, w_obj: PyObjectRef) {
    if ctx.fast {
        return;
    }
    let list = ctx.memo_list();
    let idx = unsafe { pyre_object::listobject::w_list_len(list) };
    // Compute the move-stable hash before the append, whose growth could
    // relocate `w_obj` and leave the local stale.
    let h = pyre_object::gc_hook::gc_identity_hash(w_obj as usize);
    let value_key = PickleCtx::value_memo_key(w_obj);
    unsafe { pyre_object::listobject::w_list_append(list, w_obj) };
    ctx.index.entry(h).or_default().push(idx);
    if let Some((is_str, key)) = value_key {
        let memo = if is_str {
            &mut ctx.str_memo
        } else {
            &mut ctx.bytes_memo
        };
        memo.insert(key, idx);
    }
    if ctx.proto >= 4 {
        buf.push(op::MEMOIZE);
    } else if ctx.bin {
        if idx < 256 {
            buf.push(op::BINPUT);
            buf.push(idx as u8);
        } else {
            buf.push(op::LONG_BINPUT);
            buf.extend_from_slice(&(idx as u32).to_le_bytes());
        }
    } else {
        buf.push(op::PUT);
        buf.extend_from_slice(format!("{idx}\n").as_bytes());
    }
}

// ── reduce / global ──────────────────────────────────────────────────

/// Build a PicklingError while preserving the intercepted import/attribute
/// exception as `__context__`, matching the native pickler's exception
/// chaining rather than flattening it to text.
fn pickling_error_with_context(
    message: impl Into<rustpython_wtf8::Wtf8Buf>,
    mut context: PyError,
) -> PyError {
    let _roots = pyre_object::gc_roots::push_roots();
    let w_context = context.to_exc_object();
    let _ = pyre_object::gc_roots::pin_root(w_context);
    let context_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let mut error = pickling_error(message);
    let w_error = error.to_exc_object();
    unsafe {
        pyre_object::interp_exceptions::w_exception_set_context(
            w_error,
            pyre_object::gc_roots::shadow_stack_get(context_slot),
        );
    }
    error
}

/// `whichmodule` — the module an object belongs to. `__module__` takes
/// precedence; otherwise scan `sys.modules` for the module that exposes
/// `name` resolving back to `w_obj`, skipping `__main__` / `__mp_main__` /
/// `None`, and default to `"__main__"`. Then verify (importing the module)
/// that `module_name.name` resolves back to this exact object, raising a
/// PicklingError otherwise — the dump-time check that the wire reference is
/// actually loadable.
enum ModuleName {
    Utf8(String),
    Surrogate(PyObjectRef),
}

fn whichmodule(w_obj: PyObjectRef, name: &str) -> Result<ModuleName, PyError> {
    // Every reporting path below renders `w_obj`'s repr, and the module scan
    // in between runs arbitrary Python; keep the object in a shadow-stack slot
    // so a relocation cannot leave those reads holding a stale address.
    let _roots = pyre_object::gc_roots::push_roots();
    let _ = pyre_object::gc_roots::pin_root(w_obj);
    let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    if name.split('.').any(|s| s == "<locals>") {
        let obj_repr = unsafe {
            pyre_interpreter::display::py_repr_wtf8(pyre_object::gc_roots::shadow_stack_get(
                obj_slot,
            ))?
        };
        return Err(pickling_error(pyre_interpreter::display::wtf8_format!(
            "Can't pickle local object ",
            obj_repr
        )));
    }
    // `interp_pickle.py whichmodule` returns any non-None
    // `__module__`. A non-string module name is invalid; reject it here with
    // `TypeError("module name must be a string")` — the same error it surfaces
    // once used, raised at resolution time rather than deferred to the import.
    let from_attr: Option<ModuleName> = match pyre_interpreter::baseobjspace::findattr_result(
        pyre_object::gc_roots::shadow_stack_get(obj_slot),
        "__module__",
    )? {
        Some(m) if !unsafe { pyre_object::is_none(m) } => {
            if !unsafe { pyre_object::is_str(m) } {
                return Err(PyError::type_error("module name must be a string"));
            }
            match unsafe { pyre_object::unicodeobject::w_str_get_value_opt(m) } {
                Some(module) => Some(ModuleName::Utf8(module.to_string())),
                None => Some(ModuleName::Surrogate(m)),
            }
        }
        _ => None,
    };
    let module_name = match from_attr {
        Some(ModuleName::Utf8(mn)) => mn,
        Some(ModuleName::Surrogate(w_module_name)) => {
            let _roots = pyre_object::gc_roots::push_roots();
            let _ =
                pyre_object::gc_roots::pin_root(pyre_object::gc_roots::shadow_stack_get(obj_slot));
            let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let _ = pyre_object::gc_roots::pin_root(w_module_name);
            let module_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let modules = pyre_interpreter::importing::sys_modules_dict();
            let module = unsafe {
                pyre_object::w_dict_lookup(
                    modules,
                    pyre_object::gc_roots::shadow_stack_get(module_slot),
                )
            }
            .ok_or_else(|| pickling_error("Can't pickle object: module is not in sys.modules"))?;
            let resolved = match getattribute_dotted(module, name) {
                Ok((value, _)) => Some(value),
                Err(error)
                    if matches!(error.kind, pyre_interpreter::PyErrorKind::AttributeError) =>
                {
                    None
                }
                Err(error) => return Err(error),
            };
            return match resolved {
                Some(resolved)
                    if pyre_interpreter::baseobjspace::is_w(
                        resolved,
                        pyre_object::gc_roots::shadow_stack_get(obj_slot),
                    ) =>
                {
                    Ok(ModuleName::Surrogate(
                        pyre_object::gc_roots::shadow_stack_get(module_slot),
                    ))
                }
                Some(_) => Err(pickling_error(
                    "Can't pickle object: global identity mismatch",
                )),
                None => Err(pickling_error("Can't pickle object: global is not found")),
            };
        }
        None => {
            // Scan sys.modules; a match here is already verified by identity.
            let modules = pyre_interpreter::importing::sys_modules_dict();
            let mut found: Option<String> = None;
            if !modules.is_null() {
                // `getattribute_dotted` runs Python and can relocate objects, so
                // pin `w_obj` and every candidate module up front (capturing each
                // name as an owned String — GC-independent), then scan via the
                // pinned slots. The snapshot loop itself triggers no collection.
                let _roots = pyre_object::gc_roots::push_roots();
                let _ = pyre_object::gc_roots::pin_root(pyre_object::gc_roots::shadow_stack_get(
                    obj_slot,
                ));
                let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
                let mut candidates: Vec<(String, usize)> = Vec::new();
                for (w_modname, w_module) in
                    unsafe { pyre_object::dictmultiobject::w_dict_items(modules) }
                {
                    if !unsafe { pyre_object::is_str(w_modname) }
                        || unsafe { pyre_object::is_none(w_module) }
                    {
                        continue;
                    }
                    let Some(modname) =
                        (unsafe { pyre_object::unicodeobject::w_str_get_value_opt(w_modname) })
                    else {
                        continue;
                    };
                    let modname = modname.to_string();
                    if modname == "__main__" || modname == "__mp_main__" {
                        continue;
                    }
                    let _ = pyre_object::gc_roots::pin_root(w_module);
                    candidates.push((modname, pyre_object::gc_roots::shadow_stack_len() - 1));
                }
                for (modname, mod_slot) in candidates {
                    let w_module = pyre_object::gc_roots::shadow_stack_get(mod_slot);
                    if let Ok((resolved, _)) = getattribute_dotted(w_module, name)
                        && pyre_interpreter::baseobjspace::is_w(
                            resolved,
                            pyre_object::gc_roots::shadow_stack_get(obj_slot),
                        )
                    {
                        found = Some(modname);
                        break;
                    }
                }
            }
            match found {
                Some(mn) => return Ok(ModuleName::Utf8(mn)),
                None => String::from("__main__"),
            }
        }
    };
    // Verify `module_name.name` resolves back to `w_obj` (qualnames walked).
    // CPython pickle.whichmodule converts import failures (including the
    // ModuleNotFoundError subclass), an invalid module name, and a missing
    // sys.modules entry into PicklingError.  Other exceptions raised by a
    // module's dynamic attribute machinery still propagate unchanged.
    let module = match import_module(&module_name) {
        Ok(module) => module,
        Err(error)
            if matches!(
                error.kind,
                pyre_interpreter::PyErrorKind::ImportError
                    | pyre_interpreter::PyErrorKind::ModuleNotFoundError
                    | pyre_interpreter::PyErrorKind::ValueError
                    | pyre_interpreter::PyErrorKind::KeyError
            ) =>
        {
            let _roots = pyre_object::gc_roots::push_roots();
            let mut error = error;
            let error_slot = error.pin(&_roots);
            let obj_repr = unsafe {
                pyre_interpreter::display::py_repr_wtf8(pyre_object::gc_roots::shadow_stack_get(
                    obj_slot,
                ))?
            };
            error.reload(&_roots, error_slot);
            let detail = error.message_wtf8();
            error.reload(&_roots, error_slot);
            return Err(pickling_error_with_context(
                pyre_interpreter::display::wtf8_format!("Can't pickle ", obj_repr, ": ", detail),
                error,
            ));
        }
        Err(error) => return Err(error),
    };
    // Attribute traversal is deliberately outside the import-error
    // conversion above.  A module's PEP 562 __getattr__ may raise KeyError,
    // and that user exception must propagate rather than become PicklingError.
    let resolved = match getattribute_dotted(module, name) {
        Ok((value, _)) => value,
        Err(error) if matches!(error.kind, pyre_interpreter::PyErrorKind::AttributeError) => {
            let _roots = pyre_object::gc_roots::push_roots();
            let mut error = error;
            let error_slot = error.pin(&_roots);
            let obj_repr = unsafe {
                pyre_interpreter::display::py_repr_wtf8(pyre_object::gc_roots::shadow_stack_get(
                    obj_slot,
                ))?
            };
            error.reload(&_roots, error_slot);
            return Err(pickling_error_with_context(
                pyre_interpreter::display::wtf8_format!(
                    "Can't pickle ",
                    obj_repr,
                    format!(": it's not found as {module_name}.{name}")
                ),
                error,
            ));
        }
        Err(error) => return Err(error),
    };
    if pyre_interpreter::baseobjspace::is_w(
        resolved,
        pyre_object::gc_roots::shadow_stack_get(obj_slot),
    ) {
        Ok(ModuleName::Utf8(module_name))
    } else {
        let obj_repr = unsafe {
            pyre_interpreter::display::py_repr_wtf8(pyre_object::gc_roots::shadow_stack_get(
                obj_slot,
            ))?
        };
        Err(pickling_error(pyre_interpreter::display::wtf8_format!(
            "Can't pickle ",
            obj_repr,
            format!(": it's not the same object as {module_name}.{name}")
        )))
    }
}

/// `interp_pickle.py save_global` / `save_global2` — save an object by
/// qualified reference. `w_name_opt` carries the name when a `__reduce__`
/// returned a string; otherwise it is derived from `__qualname__`.
fn save_global(
    ctx: &mut PickleCtx,
    buf: &mut Framer,
    mut w_obj: PyObjectRef,
    w_name_opt: Option<PyObjectRef>,
) -> Result<(), PyError> {
    let w_name = match w_name_opt {
        Some(n) => n,
        None => match pyre_object::with_roots!(w_obj => pyre_interpreter::baseobjspace::findattr_result(w_obj, "__qualname__"))? {
            Some(n) => n,
            None => pyre_object::with_roots!(w_obj => pyre_interpreter::baseobjspace::findattr_result(w_obj, "__name__"))?
                .ok_or_else(|| pickling_error("Can't pickle object: no __qualname__ / __name__"))?,
        },
    };
    // `whichmodule` imports the home module (to verify the reference), so pin
    // `w_obj` / `w_name` and re-read them afterwards.
    let _roots = pyre_object::gc_roots::push_roots();
    let _ = pyre_object::gc_roots::pin_root(w_obj);
    let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let _ = pyre_object::gc_roots::pin_root(w_name);
    let name_slot = pyre_object::gc_roots::shadow_stack_len() - 1;

    let w_name = pyre_object::gc_roots::shadow_stack_get(name_slot);
    if !unsafe { pyre_object::is_str(w_name) } {
        return Err(PyError::type_error("global name must be a string"));
    }
    let Some(name) =
        (unsafe { pyre_object::unicodeobject::w_str_get_value_opt(w_name) }).map(str::to_string)
    else {
        return save_global_surrogate_name(ctx, buf, obj_slot, name_slot);
    };
    let module_name = whichmodule(pyre_object::gc_roots::shadow_stack_get(obj_slot), &name)?;
    let module_name = match module_name {
        ModuleName::Utf8(module_name) => module_name,
        ModuleName::Surrogate(w_module_name) => {
            let _ = pyre_object::gc_roots::pin_root(w_module_name);
            let module_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            return save_global_surrogate_module(ctx, buf, obj_slot, name_slot, module_slot);
        }
    };

    // protocol >= 2: a `copyreg` extension code is emitted as EXT1/EXT2/EXT4
    // (and the object is not memoized — the reference is idempotent).
    if ctx.proto >= 2
        && let Some(code) = extension_code(&module_name, &name)
    {
        write_ext(buf, code)?;
        return Ok(());
    }

    if ctx.proto >= 4 {
        save(ctx, buf, pyre_object::w_str_new_managed(&module_name))?;
        save(ctx, buf, pyre_object::gc_roots::shadow_stack_get(name_slot))?;
        buf.push(op::STACK_GLOBAL);
    } else if name.contains('.') {
        // protocol < 4 nested: objects with a multi-part `__qualname__` are
        // represented as `getattr(getattr(..., attrname1), attrname2)`. The
        // top-level name is emitted by GLOBAL and each remaining dotted part
        // by one `getattr(<current>, attrname)` REDUCE. `getattr` and the
        // attrname strings memoize through `save`; the top-level GLOBAL and
        // the intermediate tuples are not memoized, and the final result is
        // memoized by the trailing `memoize(obj)`.
        let parts: Vec<&str> = name.split('.').collect();
        let rest = &parts[1..];
        for _ in rest {
            let w_getattr = builtin_attr("getattr")?;
            save(ctx, buf, w_getattr)?;
            if ctx.proto < 2 {
                buf.push(op::MARK);
            }
        }
        save_toplevel_by_name(ctx, buf, &module_name, parts[0])?;
        for attrname in rest {
            save(ctx, buf, pyre_object::w_str_new_managed(attrname))?;
            if ctx.proto < 2 {
                buf.push(op::TUPLE);
            } else {
                buf.push(op::TUPLE2);
            }
            buf.push(op::REDUCE);
        }
    } else {
        save_toplevel_by_name(ctx, buf, &module_name, &name)?;
    }
    memoize(ctx, buf, pyre_object::gc_roots::shadow_stack_get(obj_slot));
    Ok(())
}

fn save_global_surrogate_module(
    ctx: &mut PickleCtx,
    buf: &mut Framer,
    obj_slot: usize,
    name_slot: usize,
    module_slot: usize,
) -> Result<(), PyError> {
    if ctx.proto < 4 {
        let encoding = if ctx.proto < 3 { "ascii" } else { "utf-8" };
        return match pyre_interpreter::type_methods::encode_object(
            pyre_object::gc_roots::shadow_stack_get(module_slot),
            encoding,
            "strict",
        ) {
            Ok(_) => Err(pickling_error(
                "surrogate module identifier unexpectedly encoded",
            )),
            Err(error) => Err(identifier_encoding_error(
                error,
                pyre_object::gc_roots::shadow_stack_get(module_slot),
                "module",
                ctx.proto,
            )),
        };
    }
    save(
        ctx,
        buf,
        pyre_object::gc_roots::shadow_stack_get(module_slot),
    )?;
    save(ctx, buf, pyre_object::gc_roots::shadow_stack_get(name_slot))?;
    buf.push(op::STACK_GLOBAL);
    memoize(ctx, buf, pyre_object::gc_roots::shadow_stack_get(obj_slot));
    Ok(())
}

/// CPython 3.14 `save_global` path for a name containing a lone surrogate.
///
/// Attribute lookup keeps the name as a wrapped Python string, matching
/// `PyObject_GetOptionalAttr(module, name)`. Protocols before STACK_GLOBAL
/// then fail while strictly encoding the GLOBAL identifier; protocol >= 4
/// serializes the surrogatepass UTF-8 through `save_str`.
fn save_global_surrogate_name(
    ctx: &mut PickleCtx,
    buf: &mut Framer,
    obj_slot: usize,
    name_slot: usize,
) -> Result<(), PyError> {
    let w_obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
    let w_module_name = match pyre_interpreter::baseobjspace::findattr_result(w_obj, "__module__")?
    {
        Some(module) if !unsafe { pyre_object::is_none(module) } => {
            if !unsafe { pyre_object::is_str(module) } {
                return Err(PyError::type_error("module name must be a string"));
            }
            module
        }
        _ => pyre_object::w_str_new("__main__"),
    };
    let _ = pyre_object::gc_roots::pin_root(w_module_name);
    let module_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let module_name = unsafe {
        pyre_object::unicodeobject::w_str_get_value_opt(pyre_object::gc_roots::shadow_stack_get(
            module_slot,
        ))
    };
    let module = if let Some(module_name) = module_name {
        import_module(module_name)?
    } else {
        let modules = pyre_interpreter::importing::sys_modules_dict();
        unsafe {
            pyre_object::w_dict_lookup(
                modules,
                pyre_object::gc_roots::shadow_stack_get(module_slot),
            )
        }
        .ok_or_else(|| pickling_error("Can't pickle object: module is not in sys.modules"))?
    };
    let (resolved, _) = crate::module::_pickle::getattribute_dotted_obj(
        module,
        pyre_object::gc_roots::shadow_stack_get(name_slot),
    )?;
    if !pyre_interpreter::baseobjspace::is_w(
        resolved,
        pyre_object::gc_roots::shadow_stack_get(obj_slot),
    ) {
        return Err(pickling_error(
            "Can't pickle object: global identity mismatch",
        ));
    }

    if ctx.proto < 4 {
        let encoding = if ctx.proto < 3 { "ascii" } else { "utf-8" };
        if let Err(error) = pyre_interpreter::type_methods::encode_object(
            pyre_object::gc_roots::shadow_stack_get(module_slot),
            encoding,
            "strict",
        ) {
            return Err(identifier_encoding_error(
                error,
                pyre_object::gc_roots::shadow_stack_get(module_slot),
                "module",
                ctx.proto,
            ));
        }
        return match pyre_interpreter::type_methods::encode_object(
            pyre_object::gc_roots::shadow_stack_get(name_slot),
            encoding,
            "strict",
        ) {
            Ok(_) => Err(pickling_error(
                "surrogate global identifier unexpectedly encoded",
            )),
            Err(error) => Err(identifier_encoding_error(
                error,
                pyre_object::gc_roots::shadow_stack_get(name_slot),
                "global",
                ctx.proto,
            )),
        };
    }

    save(
        ctx,
        buf,
        pyre_object::gc_roots::shadow_stack_get(module_slot),
    )?;
    save(ctx, buf, pyre_object::gc_roots::shadow_stack_get(name_slot))?;
    buf.push(op::STACK_GLOBAL);
    memoize(ctx, buf, pyre_object::gc_roots::shadow_stack_get(obj_slot));
    Ok(())
}

fn identifier_encoding_error(
    mut context: PyError,
    w_identifier: PyObjectRef,
    kind: &str,
    proto: i64,
) -> PyError {
    let _roots = pyre_object::gc_roots::push_roots();
    let _ = pyre_object::gc_roots::pin_root(w_identifier);
    let identifier_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let context_obj = context.to_exc_object();
    let _ = pyre_object::gc_roots::pin_root(context_obj);
    let context_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let identifier_repr = unsafe {
        pyre_interpreter::display::py_repr_wtf8(pyre_object::gc_roots::shadow_stack_get(
            identifier_slot,
        ))
    }
    .unwrap_or_else(|_| rustpython_wtf8::Wtf8Buf::from_string("<identifier>".to_string()));
    let mut error = pickling_error(pyre_interpreter::display::wtf8_format!(
        format!("can't pickle {kind} identifier "),
        identifier_repr,
        format!(" using pickle protocol {proto}")
    ));
    let exc = error.to_exc_object();
    let _ = pyre_object::gc_roots::pin_root(exc);
    let exc_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    pyre_interpreter::error::chain_context(
        pyre_object::gc_roots::shadow_stack_get(exc_slot),
        pyre_object::gc_roots::shadow_stack_get(context_slot),
    );
    error.set_exc_object(pyre_object::gc_roots::shadow_stack_get(exc_slot));
    error
}

/// `_save_toplevel_by_name` — emit a GLOBAL opcode for a top-level name,
/// applying the protocol-< 3 `fix_imports` py3 → py2 reverse map.
fn save_toplevel_by_name(
    ctx: &PickleCtx,
    buf: &mut Framer,
    module_name: &str,
    name: &str,
) -> Result<(), PyError> {
    let (module_name, name) = if ctx.proto < 3 && ctx.fix_imports {
        crate::module::_pickle::compat_map(module_name, name, true)?
    } else {
        (module_name.to_string(), name.to_string())
    };
    let encoding = if ctx.proto < 3 { "ascii" } else { "utf-8" };
    let _roots = pyre_object::gc_roots::push_roots();
    let w_module_name = pyre_object::w_str_new_managed(&module_name);
    let _ = pyre_object::gc_roots::pin_root(w_module_name);
    let module_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let w_name = pyre_object::w_str_new_managed(&name);
    let _ = pyre_object::gc_roots::pin_root(w_name);
    let name_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let module_bytes = pyre_interpreter::type_methods::encode_object(
        pyre_object::gc_roots::shadow_stack_get(module_slot),
        encoding,
        "strict",
    )
    .map_err(|error| {
        identifier_encoding_error(
            error,
            pyre_object::gc_roots::shadow_stack_get(module_slot),
            "module",
            ctx.proto,
        )
    })?;
    let name_bytes = pyre_interpreter::type_methods::encode_object(
        pyre_object::gc_roots::shadow_stack_get(name_slot),
        encoding,
        "strict",
    )
    .map_err(|error| {
        identifier_encoding_error(
            error,
            pyre_object::gc_roots::shadow_stack_get(name_slot),
            "global",
            ctx.proto,
        )
    })?;
    buf.push(op::GLOBAL);
    buf.extend_from_slice(&module_bytes);
    buf.push(b'\n');
    buf.extend_from_slice(&name_bytes);
    buf.push(b'\n');
    Ok(())
}

/// The `copyreg._extension_registry` code for `(module_name, name)`, if any.
fn extension_code(module_name: &str, name: &str) -> Option<i64> {
    let copyreg = import_module("copyreg").ok()?;
    let registry =
        pyre_interpreter::baseobjspace::getattr_str(copyreg, "_extension_registry").ok()?;
    let mut key_items = pyre_object::gc_roots::RootedItems::new();
    key_items.push(pyre_object::w_str_new_managed(module_name));
    key_items.push(pyre_object::w_str_new_managed(name));
    let key = pyre_object::tupleobject::w_tuple_new(key_items.take());
    let code = unsafe { pyre_object::w_dict_lookup(registry, key) }?;
    pyre_interpreter::baseobjspace::int_w(code).ok()
}

/// Emit the size-appropriate EXT opcode for an extension code. `save_global`
/// requires `0 < code <= 0x7fffffff`; anything else is out of range.
fn write_ext(buf: &mut Framer, code: i64) -> Result<(), PyError> {
    if code <= 0 || code > 0x7fffffff {
        return Err(PyError::runtime_error(format!(
            "extension code {code} is out of range"
        )));
    }
    if code <= 0xff {
        buf.push(op::EXT1);
        buf.push(code as u8);
    } else if code <= 0xffff {
        buf.push(op::EXT2);
        buf.extend_from_slice(&(code as u16).to_le_bytes());
    } else {
        buf.push(op::EXT4);
        buf.extend_from_slice(&(code as i32).to_le_bytes());
    }
    Ok(())
}

/// `__builtins__.<name>` (e.g. `getattr`), via the execution context's
/// `lookup_builtin` (the `LOAD_GLOBAL` path). Only used on the rare nested
/// protocol < 4 path.
fn builtin_attr(name: &str) -> Result<PyObjectRef, PyError> {
    crate::module::_pickle::lookup_builtin(name)
        .ok_or_else(|| pickling_error(format!("Can't resolve builtin {name:?}")))
}

/// `interp_pickle.py save_reduce`. `rv` is the 2-to-6 element reduce tuple
/// `(func, args[, state[, listitems[, dictitems[, state_setter]]]])`.
fn save_reduce(
    ctx: &mut PickleCtx,
    buf: &mut Framer,
    rv: &[PyObjectRef],
    w_obj_opt: Option<PyObjectRef>,
) -> Result<(), PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    // `pin_items` allocates a list.  Publish the reduce object and every
    // reduce value first, then normalize once: a `pin_root` of `w_obj`
    // alone is a safepoint that would leave the still-unpublished `rv`
    // slice invisible to a foreign collection.
    let has_obj = w_obj_opt.is_some();
    let mut live = Vec::with_capacity(rv.len() + usize::from(has_obj));
    if let Some(o) = w_obj_opt {
        live.push(o);
    }
    live.extend_from_slice(rv);
    let base = pyre_object::gc_roots::publish_roots(&live);
    pyre_object::gc_roots::normalize_roots(base, live.len());
    let w_obj_slot = has_obj.then_some(base);
    let rv_off = usize::from(has_obj);
    let rv_reloaded: Vec<_> = (0..rv.len())
        .map(|i| pyre_object::gc_roots::shadow_stack_get(base + rv_off + i))
        .collect();
    // Recursive saves (and the reduce callbacks they invoke) relocate young
    // objects, so pin the reduce values in a GC-walked `list` and re-read each
    // one immediately before it is consumed.
    let rv_len = rv.len();
    let rv_slot = pin_items(rv_reloaded);
    let rv_get = |i: usize| pinned_get(rv_slot, i);
    let present = |i: usize| i < rv_len && !unsafe { pyre_object::is_none(pinned_get(rv_slot, i)) };

    if !pyre_interpreter::baseobjspace::callable_w(rv_get(0)) {
        return Err(pickling_error(format!(
            "first item of the tuple returned by __reduce__ must be callable, not {}",
            pyre_interpreter::error::type_name_of(rv_get(0))
        )));
    }
    if !unsafe { pyre_object::is_tuple(rv_get(1)) } {
        return Err(pickling_error(format!(
            "second item of the tuple returned by __reduce__ must be a tuple, not {}",
            pyre_interpreter::error::type_name_of(rv_get(1))
        )));
    }

    let has_state = present(2);
    let has_listitems = present(3);
    let has_dictitems = present(4);
    let has_state_setter = present(5);

    // CPython's native pickler accepts iterators here, not arbitrary
    // iterables. This is the C implementation's structural counterpart of
    // PyIter_Check in save_reduce; accepting a list silently changes when its
    // contents are observed and misses mutation errors.
    if has_listitems
        && pyre_interpreter::baseobjspace::findattr_result(rv_get(3), "__next__")?.is_none()
    {
        return Err(pickling_error(format!(
            "fourth item of the tuple returned by __reduce__ must be an iterator, not {}",
            pyre_interpreter::error::type_name_of(rv_get(3))
        )));
    }
    if has_dictitems
        && pyre_interpreter::baseobjspace::findattr_result(rv_get(4), "__next__")?.is_none()
    {
        return Err(pickling_error(format!(
            "fifth item of the tuple returned by __reduce__ must be an iterator, not {}",
            pyre_interpreter::error::type_name_of(rv_get(4))
        )));
    }
    if has_state_setter && !pyre_interpreter::baseobjspace::callable_w(rv_get(5)) {
        return Err(pickling_error(format!(
            "sixth item of the tuple returned by __reduce__ must be callable, not {}",
            pyre_interpreter::error::type_name_of(rv_get(5))
        )));
    }

    let func_name = func_name_str(rv_get(0))?;

    // Pin the args tuple; its elements are re-read per save.
    let _ = pyre_object::gc_roots::pin_root(rv_get(1));
    let args_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let args_get = |i: usize| unsafe {
        pyre_object::tupleobject::w_tuple_getitem(
            pyre_object::gc_roots::shadow_stack_get(args_slot),
            i as i64,
        )
        .unwrap()
    };
    let args_len = unsafe {
        pyre_object::tupleobject::w_tuple_len(pyre_object::gc_roots::shadow_stack_get(args_slot))
    };

    if ctx.proto >= 2 && func_name.as_deref() == Some("__newobj_ex__") {
        if args_len != 3 {
            return Err(pickling_error(format!(
                "__newobj_ex__ expected 3 arguments, got {args_len}"
            )));
        }
        if pyre_interpreter::baseobjspace::findattr_result(args_get(0), "__new__")?.is_none() {
            return Err(pickling_error(
                "first argument to __newobj_ex__() has no __new__",
            ));
        }
        if !unsafe { pyre_object::is_tuple(args_get(1)) } {
            return Err(pickling_error(format!(
                "second argument to __newobj_ex__() must be a tuple, not {}",
                pyre_interpreter::error::type_name_of(args_get(1))
            )));
        }
        if !unsafe { pyre_object::is_dict(args_get(2)) } {
            return Err(pickling_error(format!(
                "third argument to __newobj_ex__() must be a dict, not {}",
                pyre_interpreter::error::type_name_of(args_get(2))
            )));
        }
        if let Some(slot) = w_obj_slot {
            let w_class = pyre_interpreter::baseobjspace::getattr_str(
                pyre_object::gc_roots::shadow_stack_get(slot),
                "__class__",
            )?;
            let _ = pyre_object::gc_roots::pin_root(w_class);
            let class_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            if !pyre_interpreter::baseobjspace::is_w(
                args_get(0),
                pyre_object::gc_roots::shadow_stack_get(class_slot),
            ) {
                let cls_repr = unsafe { pyre_interpreter::display::py_repr_wtf8(args_get(0))? };
                let obj_class_repr = unsafe {
                    pyre_interpreter::display::py_repr_wtf8(
                        pyre_object::gc_roots::shadow_stack_get(class_slot),
                    )?
                };
                return Err(pickling_error(pyre_interpreter::display::wtf8_format!(
                    "first argument to __newobj_ex__() must be ",
                    obj_class_repr,
                    ", not ",
                    cls_repr
                )));
            }
        }
        if !unsafe { pyre_object::is_tuple(args_get(1)) } {
            return Err(pickling_error(format!(
                "second argument to __newobj_ex__() must be a tuple, not {}",
                pyre_interpreter::baseobjspace::object_functionstr_type_name(args_get(1)),
            )));
        }
        if !unsafe { pyre_object::is_dict(args_get(2)) } {
            return Err(pickling_error(format!(
                "third argument to __newobj_ex__() must be a dict, not {}",
                pyre_interpreter::baseobjspace::object_functionstr_type_name(args_get(2)),
            )));
        }
        if ctx.proto >= 4 {
            if let Err(err) = save(ctx, buf, args_get(0)) {
                return Err(add_reduce_note(
                    err,
                    w_obj_slot,
                    rustpython_wtf8::Wtf8::new("class"),
                ));
            }
            if let Err(err) = save(ctx, buf, args_get(1)) {
                return Err(add_reduce_note(
                    err,
                    w_obj_slot,
                    rustpython_wtf8::Wtf8::new("__new__ arguments"),
                ));
            }
            if let Err(err) = save(ctx, buf, args_get(2)) {
                return Err(add_reduce_note(
                    err,
                    w_obj_slot,
                    rustpython_wtf8::Wtf8::new("__new__ arguments"),
                ));
            }
            buf.push(op::NEWOBJ_EX);
        } else {
            // pickle.py. Protocol 2/3 has no NEWOBJ_EX opcode, so
            // construct `partial(cls.__new__, cls, *args, **kwargs)` and
            // reduce that zero-argument callable. Keeping `cls` and `args`
            // inside the partial is observable in both reconstruction and
            // the nested PEP 678 notes emitted while the partial is saved.
            let w_new = pyre_interpreter::baseobjspace::getattr_str(args_get(0), "__new__")?;
            let _ = pyre_object::gc_roots::pin_root(w_new);
            let new_slot = pyre_object::gc_roots::shadow_stack_len() - 1;

            let functools = import_module("functools")?;
            let w_partial = pyre_interpreter::baseobjspace::getattr_str(functools, "partial")?;
            let kw_items = unsafe { pyre_object::dictmultiobject::w_dict_items(args_get(2)) };
            let mut kwargs = Vec::with_capacity(kw_items.len());
            for (k, v) in kw_items {
                if !unsafe { pyre_object::is_str(k) } {
                    return Err(pickling_error("__newobj_ex__ kwargs keys must be strings"));
                }
                kwargs.push((
                    unsafe { pyre_object::unicodeobject::w_str_get_wtf8(k) }.to_owned(),
                    v,
                ));
            }
            let ec = pyre_interpreter::call::getexecutioncontext();
            if ec.is_null() {
                return Err(pickling_error("no execution context for __newobj_ex__"));
            }
            let frame = unsafe { (*ec).gettopframe() };
            if frame.is_null() {
                return Err(pickling_error("no frame for __newobj_ex__ at protocol < 4"));
            }
            // Re-read every positional input from its root after the imports
            // and kwargs conversion above, which may have relocated them.
            let mut partial_args = Vec::with_capacity(2 + tuple_items(args_get(1)).len());
            partial_args.push(pyre_object::gc_roots::shadow_stack_get(new_slot));
            partial_args.push(args_get(0));
            partial_args.extend(tuple_items(args_get(1)));
            let w_func = pyre_interpreter::call::call_with_kwargs(
                unsafe { &mut *frame },
                w_partial,
                &partial_args,
                &kwargs,
            )?;
            let _ = pyre_object::gc_roots::pin_root(w_func);
            let func_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            if let Err(err) = save(ctx, buf, pyre_object::gc_roots::shadow_stack_get(func_slot)) {
                return Err(add_reduce_note(
                    err,
                    w_obj_slot,
                    rustpython_wtf8::Wtf8::new("reconstructor"),
                ));
            }
            let empty_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ =
                pyre_object::gc_roots::pin_root(pyre_object::tupleobject::w_tuple_new(Vec::new()));
            save(
                ctx,
                buf,
                pyre_object::gc_roots::shadow_stack_get(empty_slot),
            )?;
            buf.push(op::REDUCE);
        }
    } else if ctx.proto >= 2 && func_name.as_deref() == Some("__newobj__") {
        if args_len == 0 {
            return Err(pickling_error(
                "__newobj__ expected at least 1 argument, got 0",
            ));
        }
        if pyre_interpreter::baseobjspace::findattr_result(args_get(0), "__new__")?.is_none() {
            return Err(pickling_error(
                "first argument to __newobj__() has no __new__",
            ));
        }
        if let Some(slot) = w_obj_slot {
            let w_class = pyre_interpreter::baseobjspace::getattr_str(
                pyre_object::gc_roots::shadow_stack_get(slot),
                "__class__",
            )?;
            let _ = pyre_object::gc_roots::pin_root(w_class);
            let class_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            if !pyre_interpreter::baseobjspace::is_w(
                args_get(0),
                pyre_object::gc_roots::shadow_stack_get(class_slot),
            ) {
                let cls_repr = unsafe { pyre_interpreter::display::py_repr_wtf8(args_get(0))? };
                let obj_class_repr = unsafe {
                    pyre_interpreter::display::py_repr_wtf8(
                        pyre_object::gc_roots::shadow_stack_get(class_slot),
                    )?
                };
                return Err(pickling_error(pyre_interpreter::display::wtf8_format!(
                    "first argument to __newobj__() must be ",
                    obj_class_repr,
                    ", not ",
                    cls_repr
                )));
            }
        }
        let w_newargs =
            pyre_object::tupleobject::w_tuple_new((1..args_len).map(&args_get).collect());
        let _ = pyre_object::gc_roots::pin_root(w_newargs);
        let newargs_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        if let Err(err) = save(ctx, buf, args_get(0)) {
            return Err(add_reduce_note(
                err,
                w_obj_slot,
                rustpython_wtf8::Wtf8::new("class"),
            ));
        }
        if let Err(err) = save(
            ctx,
            buf,
            pyre_object::gc_roots::shadow_stack_get(newargs_slot),
        ) {
            return Err(add_reduce_note(
                err,
                w_obj_slot,
                rustpython_wtf8::Wtf8::new("__new__ arguments"),
            ));
        }
        buf.push(op::NEWOBJ);
    } else {
        if let Err(err) = save(ctx, buf, rv_get(0)) {
            return Err(add_reduce_note(
                err,
                w_obj_slot,
                rustpython_wtf8::Wtf8::new("reconstructor"),
            ));
        }
        if let Err(err) = save(ctx, buf, rv_get(1)) {
            return Err(add_reduce_note(
                err,
                w_obj_slot,
                rustpython_wtf8::Wtf8::new("reconstructor arguments"),
            ));
        }
        buf.push(op::REDUCE);
    }

    if let Some(slot) = w_obj_slot {
        let w_obj = pyre_object::gc_roots::shadow_stack_get(slot);
        if let Some(idx) = ctx.memo_get(w_obj) {
            buf.push(op::POP);
            write_get(ctx, buf, idx);
        } else {
            memoize(ctx, buf, w_obj);
        }
    }

    if has_listitems {
        let _ = pyre_object::gc_roots::pin_root(rv_get(3));
        let items_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        batch_appends(ctx, buf, items_slot, w_obj_slot)?;
    }
    if has_dictitems {
        let _ = pyre_object::gc_roots::pin_root(rv_get(4));
        let pairs_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        batch_setitems(ctx, buf, pairs_slot, w_obj_slot)?;
    }
    if has_state {
        if has_state_setter {
            if let Err(err) = save(ctx, buf, rv_get(5)) {
                return Err(add_reduce_note(
                    err,
                    w_obj_slot,
                    rustpython_wtf8::Wtf8::new("state setter"),
                ));
            }
            let state_obj = w_obj_slot
                .map(pyre_object::gc_roots::shadow_stack_get)
                .unwrap_or_else(pyre_object::w_none);
            save(ctx, buf, state_obj)?;
            if let Err(err) = save(ctx, buf, rv_get(2)) {
                return Err(add_reduce_note(
                    err,
                    w_obj_slot,
                    rustpython_wtf8::Wtf8::new("state"),
                ));
            }
            buf.push(op::TUPLE2);
            buf.push(op::REDUCE);
            buf.push(op::POP);
        } else {
            if let Err(err) = save(ctx, buf, rv_get(2)) {
                return Err(add_reduce_note(
                    err,
                    w_obj_slot,
                    rustpython_wtf8::Wtf8::new("state"),
                ));
            }
            buf.push(op::BUILD);
        }
    }
    Ok(())
}

/// The `__name__` of a callable as an owned `String`, if it is a str.
fn func_name_str(w_func: PyObjectRef) -> Result<Option<String>, PyError> {
    let Some(w_name) = pyre_interpreter::baseobjspace::findattr_result(w_func, "__name__")? else {
        return Ok(None);
    };
    if unsafe { pyre_object::is_str(w_name) } {
        Ok(Some(
            pyre_interpreter::baseobjspace::str_utf8_w(w_name)?.to_string(),
        ))
    } else {
        Ok(None)
    }
}

fn tuple_items(w_tuple: PyObjectRef) -> Vec<PyObjectRef> {
    let n = unsafe { pyre_object::tupleobject::w_tuple_len(w_tuple) };
    (0..n)
        .map(|i| unsafe { pyre_object::tupleobject::w_tuple_getitem(w_tuple, i as i64).unwrap() })
        .collect()
}
