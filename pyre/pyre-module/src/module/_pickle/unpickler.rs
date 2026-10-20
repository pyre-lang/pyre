//! `_pickle.Unpickler` — `interp_pickle.py W_Unpickler` (atom + container subset).

use pyre_object::PyObjectRef;

use pyre_interpreter::PyError;

use super::{
    HIGHEST_PROTOCOL, call_fn, call_meth, decode_long, import_module, op, parse_int_text,
    read_int_le, str_from_utf8, unpickling_error,
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum PackedListKind {
    Ascii,
    Bytes,
}

fn packed_list_sentinel_called(_args: &[PyObjectRef]) -> pyre_interpreter::PyResult {
    Err(unpickling_error(
        "internal packed-list unpickler sentinel is not callable",
    ))
}

/// Process-global stand-ins for PyPy's two `_W_*ListUnpickler` instances.
///
/// The objects made by `make_builtin_function` are shared by every thread.
/// They never escape the unpickler stack: GLOBAL installs one and REDUCE
/// consumes it by identity.  The cache is a `RootedOnceRef` so a move
/// rewrites the slot.
fn packed_list_sentinel(kind: PackedListKind) -> PyObjectRef {
    static ASCII: pyre_object::gc_roots::RootedOnceRef =
        pyre_object::gc_roots::RootedOnceRef::new();
    static BYTES: pyre_object::gc_roots::RootedOnceRef =
        pyre_object::gc_roots::RootedOnceRef::new();
    let cell = match kind {
        PackedListKind::Ascii => &ASCII,
        PackedListKind::Bytes => &BYTES,
    };
    cell.get_or_init(|| {
        pyre_interpreter::make_builtin_function(
            match kind {
                PackedListKind::Ascii => "_ascii_list_unpickle",
                PackedListKind::Bytes => "_bytes_list_unpickle",
            },
            packed_list_sentinel_called,
        )
    })
}

/// Clear `running` through the live shadow-stack owner after a moving
/// collection.  Holding `&mut self.running` would target evacuated memory.
struct RunningGuard {
    self_slot: usize,
}

impl Drop for RunningGuard {
    fn drop(&mut self) {
        cur(self.self_slot).running = false;
    }
}

// CPython 3.14 Modules/_pickle.c:pickle_exec CREATE_TYPEs
// unpickler_type_spec; the spec is an immutable heap type.
#[pyre_interpreter::pyre_class("_pickle.Unpickler", cpython_heaptype, user_layout)]
pub struct W_Unpickler {
    w_file_read: PyObjectRef,
    w_file_readline: PyObjectRef,
    /// Result stack — a Python `list` (GC-managed across `read` allocs).
    /// Object strategy, like `w_memo`: a GET pushes the memoized object here
    /// and whatever pops it must be that same object, which an unboxing
    /// strategy breaks by wrapping a fresh one per read.
    w_stack: PyObjectRef,
    /// Saved stacks for the MARK machinery — a Python `list` of lists.
    w_metastack: PyObjectRef,
    /// Memo — a position-indexed Python `list` (`interp_pickle.py:2016`). Unset
    /// slots hold `PY_NULL`; the Object strategy keeps stored values by pointer
    /// identity (a GET returns the exact memoized object).
    w_memo: PyObjectRef,
    /// Sparse backup memo (`_Unpickler_MemoPut`); `None` while unset.
    w_memo_dict: PyObjectRef,
    /// Next free memo slot (`_memo_append` target). Dense PUTs keep this in
    /// step with `len(w_memo)`; a sparse PUT that lands in `w_memo_dict` can
    /// leave it ahead of the list length.
    memo_index: i64,
    /// Active frame bytes (`bytes`) or None.
    w_frame: PyObjectRef,
    frame_index: i64,
    proto: i64,
    /// Apply the `_compat_pickle` py2→py3 name remap at protocol < 3.
    fix_imports: bool,
    /// Encoding for the legacy STRING / BINSTRING / SHORT_BINSTRING decode
    /// (`"ASCII"` by default; `"bytes"` returns the raw bytes object).
    encoding: String,
    /// Decode error handler for the above (`"strict"` by default).
    errors: String,
    /// Out-of-band `buffers` iterator (proto 5), or None.
    w_buffers: PyObjectRef,
    /// `persistent_load` callable set on the instance, or `PY_NULL` when unset
    /// (a subclass may instead override the `persistent_load` method).
    w_persistent_load: PyObjectRef,
    /// CPython `_pickle.c:692` `running`: guards `__init__` and `load`
    /// against re-entering the same unpickler from a reducer or stream.
    running: bool,
}

#[pyre_interpreter::pyre_methods(
    doc = "Unpickler(file) -> unpickler reading from file.",
    _text_signature_ = "(file, *, fix_imports=True, encoding='ASCII', errors='strict', buffers=())"
)]
impl W_Unpickler {
    #[staticmethod]
    fn __new__(cls: PyObjectRef, _args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
        // Construction arguments are consumed/validated by `__init__`; accept
        // (and ignore) any positional or keyword args here via the whole-slice
        // catch-all so the ctor keyword parameters (fix_imports/encoding/errors/
        // buffers) do not trip an unknown-argument error in `__new__`.
        let _ = _args;
        // `objspace.py` `allocate_instance` validates the requested subtype.
        // The builtin is the base layout; a subclass is `typedef.py`
        // `_getusercls` (`interp_pickle.py` `descr__new__unpickler`).
        // The initial words are `None` or `PY_NULL`, so nothing young is
        // allocated ahead of the instance.
        pyre_interpreter::typedef::check_user_subclass(type_object(), cls)?;
        Ok(W_Unpickler::allocate_instance(
            W_Unpickler {
                ob: pyre_object::PyObject {
                    ob_type: std::ptr::null(),
                    w_class: std::ptr::null_mut(),
                },
                w_file_read: pyre_object::w_none(),
                w_file_readline: pyre_object::w_none(),
                w_stack: pyre_object::w_none(),
                w_metastack: pyre_object::w_none(),
                w_memo: pyre_object::w_none(),
                w_memo_dict: pyre_object::w_none(),
                memo_index: 0,
                w_frame: pyre_object::w_none(),
                frame_index: 0,
                proto: 0,
                fix_imports: true,
                encoding: String::from("ASCII"),
                errors: String::from("strict"),
                w_buffers: pyre_object::w_none(),
                w_persistent_load: pyre_object::PY_NULL,
                running: false,
            },
            cls,
        ))
    }

    fn __init__(
        &mut self,
        file: PyObjectRef,
        #[kwonly]
        #[default(pyre_object::boolobject::w_bool_from(true))]
        fix_imports: PyObjectRef,
        #[kwonly]
        #[default(pyre_object::w_none())]
        encoding: PyObjectRef,
        #[kwonly]
        #[default(pyre_object::w_none())]
        errors: PyObjectRef,
        #[kwonly]
        #[default(pyre_object::w_none())]
        buffers: PyObjectRef,
    ) -> Result<(), PyError> {
        // `_pickle.c:724-734` BEGIN/END_USING_UNPICKLER.  Attribute lookup
        // on the input stream may execute arbitrary Python.
        let _roots = pyre_object::gc_roots::push_roots();
        let _ = pyre_object::gc_roots::pin_root(self as *mut W_Unpickler as PyObjectRef);
        let self_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        for value in [file, fix_imports, encoding, errors, buffers] {
            let _ = pyre_object::gc_roots::pin_root(value);
        }
        let file_slot = self_slot + 1;
        let fix_imports_slot = self_slot + 2;
        let encoding_slot = self_slot + 3;
        let errors_slot = self_slot + 4;
        let buffers_slot = self_slot + 5;
        if cur(self_slot).running {
            return Err(PyError::runtime_error("Unpickler object is already used"));
        }
        cur(self_slot).running = true;
        let _running = RunningGuard { self_slot };
        // `encoding` / `errors` govern the legacy STRING / BINSTRING /
        // SHORT_BINSTRING decode (`_decode_string`); `None` falls back to the
        // `"ASCII"` / `"strict"` defaults. `fix_imports` gates the proto-< 3
        // py2→py3 name remap.
        // `interp_pickle.py:2858` declares `encoding="text"`, `errors="text"`:
        // a non-string (other than the `None` default) is a TypeError.
        let encoding = pyre_object::gc_roots::shadow_stack_get(encoding_slot);
        let encoding = if unsafe { pyre_object::is_none(encoding) } {
            String::from("ASCII")
        } else if unsafe { pyre_object::is_str(encoding) } {
            pyre_interpreter::baseobjspace::str_utf8_w(encoding)?.to_string()
        } else {
            return Err(PyError::type_error(format!(
                "Unpickler() argument 'encoding' must be str, not {}",
                pyre_interpreter::baseobjspace::object_functionstr_type_name(encoding)
            )));
        };
        let errors = pyre_object::gc_roots::shadow_stack_get(errors_slot);
        let errors = if unsafe { pyre_object::is_none(errors) } {
            String::from("strict")
        } else if unsafe { pyre_object::is_str(errors) } {
            pyre_interpreter::baseobjspace::str_utf8_w(errors)?.to_string()
        } else {
            return Err(PyError::type_error(format!(
                "Unpickler() argument 'errors' must be str, not {}",
                pyre_interpreter::baseobjspace::object_functionstr_type_name(errors)
            )));
        };
        let fix_imports = pyre_interpreter::baseobjspace::is_true(
            pyre_object::gc_roots::shadow_stack_get(fix_imports_slot),
        )?;
        // `file` must expose both `read` and `readline`; a missing either is a
        // TypeError, not the bare AttributeError a direct `getattr` surfaces
        // (`_Unpickler_SetInputStream`). Store `read` before resolving
        // `readline` so the first bound method is rooted across that allocation.
        let w_file_read = pyre_interpreter::baseobjspace::findattr_result(
            pyre_object::gc_roots::shadow_stack_get(file_slot),
            "read",
        )?
        .unwrap_or(pyre_object::PY_NULL);
        let _ = pyre_object::gc_roots::pin_root(w_file_read);
        let read_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let w_file_readline = pyre_interpreter::baseobjspace::findattr_result(
            pyre_object::gc_roots::shadow_stack_get(file_slot),
            "readline",
        )?
        .unwrap_or(pyre_object::PY_NULL);
        let _ = pyre_object::gc_roots::pin_root(w_file_readline);
        let readline_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        if pyre_object::gc_roots::shadow_stack_get(read_slot).is_null()
            || pyre_object::gc_roots::shadow_stack_get(readline_slot).is_null()
        {
            return Err(PyError::type_error(
                "file must have 'read' and 'readline' attributes",
            ));
        }
        // The memo persists across `load` calls (a multi-object stream may
        // back-reference an object memoized by an earlier load).
        let memo = pyre_object::listobject::w_list_new_empty();
        let _ = pyre_object::gc_roots::pin_root(memo);
        let memo_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        // A non-None `buffers` is consumed as an iterator by NEXT_BUFFER.
        let buffers = pyre_object::gc_roots::shadow_stack_get(buffers_slot);
        let w_buffers = if unsafe { pyre_object::is_none(buffers) } {
            pyre_object::w_none()
        } else {
            pyre_interpreter::baseobjspace::iter(buffers)?
        };
        let current = cur(self_slot);
        current.encoding = encoding;
        current.errors = errors;
        current.fix_imports = fix_imports;
        current.w_file_read = pyre_object::gc_roots::shadow_stack_get(read_slot);
        current.w_file_readline = pyre_object::gc_roots::shadow_stack_get(readline_slot);
        current.w_stack = pyre_object::w_none();
        current.w_metastack = pyre_object::w_none();
        current.w_memo = pyre_object::gc_roots::shadow_stack_get(memo_slot);
        // `_pickle_Unpickler___init___impl` calls `Unpickler_clear`, which
        // `Py_CLEAR`s `memo_dict` before `_Unpickler_MemoCleanup`.
        current.w_memo_dict = pyre_object::w_none();
        current.memo_index = 0;
        current.w_frame = pyre_object::w_none();
        current.frame_index = 0;
        current.proto = 0;
        current.w_buffers = w_buffers;
        current.w_persistent_load = pyre_object::PY_NULL;
        pyre_object::gc_hook::try_gc_write_barrier(pyre_object::gc_roots::shadow_stack_get(
            self_slot,
        ) as pyre_object::gc_hook::GCREF);
        Ok(())
    }

    fn load(&mut self) -> Result<PyObjectRef, PyError> {
        // `_pickle.c:7142` BEGIN_USING_UNPICKLER: reads and REDUCE callbacks
        // must not recursively consume or reset the active opcode stack.
        let self_ptr = self as *mut W_Unpickler as PyObjectRef;
        let _roots = pyre_object::gc_roots::push_roots();
        let _ = pyre_object::gc_roots::pin_root(self_ptr);
        let slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        if cur(slot).running {
            return Err(PyError::runtime_error("Unpickler object is already used"));
        }
        cur(slot).running = true;
        // Declared after `_roots`, so Drop can still reload `slot`.
        let _running = RunningGuard { self_slot: slot };
        if unsafe { pyre_object::is_none(cur(slot).w_file_read) } {
            // interp_pickle.py:2030-2039 — a subclass constructed without a
            // file may provide `read` and `readline` methods itself.
            let Some(w_read) = pyre_interpreter::baseobjspace::findattr_result(
                pyre_object::gc_roots::shadow_stack_get(slot),
                "read",
            )?
            else {
                return Err(unpickling_error("Unpickler.__init__() was not called"));
            };
            let _ = pyre_object::gc_roots::pin_root(w_read);
            let read_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let Some(w_readline) = pyre_interpreter::baseobjspace::findattr_result(
                pyre_object::gc_roots::shadow_stack_get(slot),
                "readline",
            )?
            else {
                return Err(unpickling_error("Unpickler.__init__() was not called"));
            };
            let _ = pyre_object::gc_roots::pin_root(w_readline);
            let readline_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let current = cur(slot);
            current.w_file_read = pyre_object::gc_roots::shadow_stack_get(read_slot);
            current.w_file_readline = pyre_object::gc_roots::shadow_stack_get(readline_slot);
            pyre_object::gc_hook::try_gc_write_barrier(
                pyre_object::gc_roots::shadow_stack_get(slot) as pyre_object::gc_hook::GCREF,
            );
        }

        // Fresh stack each load; the memo persists across `load` calls so a
        // later object can back-reference one memoized by an earlier load
        // (lazily created when the unpickler was built only via `__new__`).
        let w_stack = pyre_object::listobject::w_list_new_empty();
        let me = cur(slot);
        me.w_stack = w_stack;
        unpickler_write_barrier(me as *mut W_Unpickler as PyObjectRef);
        let w_metastack = pyre_object::listobject::w_list_new_empty();
        let me = cur(slot);
        me.w_metastack = w_metastack;
        unpickler_write_barrier(me as *mut W_Unpickler as PyObjectRef);
        if unsafe { pyre_object::is_none(cur(slot).w_memo) } {
            let w_memo = pyre_object::listobject::w_list_new_empty();
            let me = cur(slot);
            me.w_memo = w_memo;
            me.w_memo_dict = pyre_object::w_none();
            me.memo_index = 0;
            unpickler_write_barrier(me as *mut W_Unpickler as PyObjectRef);
        }
        let me = cur(slot);
        me.w_frame = pyre_object::w_none();
        me.frame_index = 0;
        me.proto = 0;

        loop {
            let opcode = read_opcode(slot)?;
            if opcode == op::STOP {
                let me = cur(slot);
                return unsafe { pyre_object::listobject::w_list_pop_end(me.w_stack) }
                    .ok_or_else(|| unpickling_error("STOP with empty stack"));
            }
            dispatch(slot, opcode)?;
        }
    }

    /// `Unpickler.find_class(module, name)` — import `module` and resolve
    /// `name` against it. A subclass may override this to control which
    /// globals the unpickler is allowed to import (the standard security
    /// hook). Emits the `pickle.find_class` audit event.
    fn find_class(
        &self,
        mut w_module: PyObjectRef,
        mut w_name: PyObjectRef,
    ) -> Result<PyObjectRef, PyError> {
        // `find_class` is public; reject non-str args before the unchecked
        // `w_str_get_value` reinterpret cast (a non-str would be UB).
        if !unsafe { pyre_object::is_str(w_module) } {
            return Err(PyError::type_error("module name must be a string"));
        }
        if !unsafe { pyre_object::is_str(w_name) } {
            return Err(PyError::type_error(format!(
                "attribute name must be string, not '{}'",
                pyre_interpreter::baseobjspace::object_functionstr_type_name(w_name)
            )));
        }
        let module = unsafe { pyre_object::unicodeobject::w_str_get_value_opt(w_module) };
        let name = unsafe { pyre_object::unicodeobject::w_str_get_value_opt(w_name) };
        let (Some(module), Some(name)) = (module, name) else {
            pyre_object::with_roots!(w_module, w_name => audit_find_class_objects(w_module, w_name))?;
            let module_obj = if let Some(module) = module {
                pyre_object::with_roots!(w_name => import_module(module))?
            } else {
                let modules = pyre_interpreter::importing::sys_modules_dict();
                unsafe { pyre_object::w_dict_lookup(modules, w_module) }.ok_or_else(|| {
                    PyError::new(
                        pyre_interpreter::PyErrorKind::ModuleNotFoundError,
                        "No module named by surrogate identifier",
                    )
                })?
            };
            let (resolved, _) =
                crate::module::_pickle::getattribute_dotted_obj(module_obj, w_name)?;
            return Ok(resolved);
        };
        let module = module.to_string();
        let name = name.to_string();
        audit_find_class(&module, &name)?;
        // protocol < 3 with `fix_imports` applies the py2 → py3 `_compat_pickle`
        // forward map before resolution; otherwise the name resolves literally.
        let (module, name) = if self.proto < 3 && self.fix_imports {
            crate::module::_pickle::compat_map(&module, &name, false)?
        } else {
            (module, name)
        };
        // `interp_pickle.py`: after importing, read the live module
        // and let its actual getattr failure propagate.  In particular,
        // protocol < 4 treats a dotted name as one literal attribute and
        // reports the module's normal AttributeError; protocol >= 4 walks the
        // qualname component by component.
        if module == "builtins"
            && !name.contains('.')
            && let Some(obj) = crate::module::_pickle::lookup_builtin(&name)
        {
            return Ok(obj);
        }
        let w_module = crate::module::_pickle::import_module(&module)?;
        if self.proto >= 4 {
            match resolve_dotted_find_class(w_module, &name) {
                Ok(value) => Ok(value),
                Err(error)
                    if error.kind == pyre_interpreter::PyErrorKind::AttributeError
                        && name.contains('.') =>
                {
                    Err(attribute_error_with_context(
                        format!(
                            "Can't resolve path '{}' on module '{}'",
                            name.replace('\\', "\\\\").replace('\'', "\\'"),
                            module.replace('\\', "\\\\").replace('\'', "\\'")
                        ),
                        error,
                    ))
                }
                Err(error) => Err(error),
            }
        } else {
            pyre_interpreter::baseobjspace::getattr_str(w_module, &name)
        }
    }

    /// `_pickle.c:_pickle_Unpickler_persistent_load_impl` — the base hook is a
    /// real method which reports that no resolver was supplied.  Keeping the
    /// descriptor in the type dict gives it the same GC owner as every other
    /// interp2app method.
    fn __persistent_load_default(&self, _pid: PyObjectRef) -> Result<PyObjectRef, PyError> {
        Err(unpickling_error(
            "A load persistent id instruction was encountered, but no persistent_load function was specified.",
        ))
    }

    /// `Unpickler.persistent_load` — bind the base method while the
    /// per-instance override slot is empty; assignment/deletion changes only
    /// that slot.  Subclass methods still win through normal MRO lookup.
    #[getter]
    fn persistent_load(&self) -> Result<PyObjectRef, PyError> {
        if !self.w_persistent_load.is_null() {
            return Ok(self.w_persistent_load);
        }
        let _roots = pyre_object::gc_roots::push_roots();
        let _ = pyre_object::gc_roots::pin_root(self as *const W_Unpickler as PyObjectRef);
        let self_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let cls = type_object();
        let _ = pyre_object::gc_roots::pin_root(cls);
        let cls_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let descr = pyre_interpreter::runtime_ops::type_dict_lookup(
            pyre_object::gc_roots::shadow_stack_get(cls_slot),
            "__persistent_load_default",
        )
        .ok_or_else(|| PyError::runtime_error("lost Unpickler.persistent_load descriptor"))?;
        let _ = pyre_object::gc_roots::pin_root(descr);
        let descr_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        Ok(pyre_object::w_method_new(
            pyre_object::gc_roots::shadow_stack_get(descr_slot),
            pyre_object::gc_roots::shadow_stack_get(self_slot),
            pyre_object::gc_roots::shadow_stack_get(cls_slot),
        ))
    }

    #[setter]
    fn set_persistent_load(&mut self, w_value: PyObjectRef) {
        self.w_persistent_load = w_value;
        unpickler_write_barrier(self as *mut W_Unpickler as PyObjectRef);
    }

    #[deleter("persistent_load")]
    fn del_persistent_load(&mut self) {
        self.w_persistent_load = pyre_object::PY_NULL;
    }

    /// `Unpickler.memo` — a fresh `UnpicklerMemoProxy` viewing this unpickler's
    /// memo (CPython hands back a new proxy on each access).
    #[getter]
    fn memo(&self) -> PyObjectRef {
        let self_obj = self as *const W_Unpickler as PyObjectRef;
        let _roots = pyre_object::gc_roots::push_roots();
        let _ = pyre_object::gc_roots::pin_root(self_obj);
        let slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        memo_proxy::type_object();
        let proxy = UnpicklerMemoProxy::allocate_stable(UnpicklerMemoProxy {
            ob: pyre_object::PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            w_unpickler: pyre_object::PY_NULL,
        });
        // `allocate` may have relocated the unpickler; wire the (young) proxy to
        // its post-collection address.
        if let Some(px) = UnpicklerMemoProxy::from_obj(proxy) {
            px.w_unpickler = pyre_object::gc_roots::shadow_stack_get(slot);
            // See the pickler twin: `allocate_stable` already consumed the
            // proxy's fresh TRACK_YOUNG_PTRS, so the initializing-store
            // exemption does not apply.
            unpickler_write_barrier(proxy);
        }
        proxy
    }

    /// `Unpickler.memo` setter — `Unpickler_set_memo`.
    ///
    /// A proxy copies the source's dense array only (`_Unpickler_NewMemo` +
    /// `Py_XNewRef` each `memo[i]`); neither side's `memo_dict` is copied or
    /// cleared. A dict allocates `_Unpickler_NewMemo(len(dict))`, then
    /// `_Unpickler_MemoPut(self, idx, value)` each pair against the *current*
    /// memo, then `_Unpickler_MemoCleanup` and installs that empty array — so
    /// sparse keys survive in `w_memo_dict` and dense keys leave with the old
    /// array.
    #[setter]
    fn set_memo(&mut self, w_value: PyObjectRef) -> Result<(), PyError> {
        let self_obj = self as *mut W_Unpickler as PyObjectRef;
        let _roots = pyre_object::gc_roots::push_roots();
        let _ = pyre_object::gc_roots::pin_root(self_obj);
        let self_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let _ = pyre_object::gc_roots::pin_root(w_value);
        let value_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let w_value = pyre_object::gc_roots::shadow_stack_get(value_slot);
        if let Some(px) = UnpicklerMemoProxy::from_obj(w_value) {
            let src = px.w_unpickler;
            let _ = pyre_object::gc_roots::pin_root(src);
            let src_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let new_memo = memo_copy_dense_list(src_slot);
            let _ = pyre_object::gc_roots::pin_root(new_memo);
            let new_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            memo_replace_dense(self_slot, pyre_object::gc_roots::shadow_stack_get(new_slot));
        } else if unsafe { pyre_object::is_dict(w_value) } {
            memo_assign_from_dict(
                self_slot,
                pyre_object::gc_roots::shadow_stack_get(value_slot),
            )?;
        } else {
            return Err(PyError::type_error(format!(
                "'memo' attribute must be an UnpicklerMemoProxy object or dict, not {}",
                pyre_interpreter::baseobjspace::object_functionstr_type_name(
                    pyre_object::gc_roots::shadow_stack_get(value_slot),
                ),
            )));
        }
        Ok(())
    }

    /// `Unpickler.memo` is not deletable.
    #[deleter("memo")]
    fn del_memo(&self) -> Result<(), PyError> {
        Err(PyError::type_error("attribute deletion is not supported"))
    }
}

/// The native CPython 3.14 `_pickle` path does not apply pickle.py's
/// `<locals>` pre-check while resolving a protocol-4 global: each component
/// reaches `getattr`, and the path-level error retains that failure as its
/// context.  Keep this load-only detail separate from the PyPy-derived
/// `getattribute_dotted`, which is also used by the dump-side verifier.
fn resolve_dotted_find_class(
    mut object: PyObjectRef,
    qualname: &str,
) -> Result<PyObjectRef, PyError> {
    for component in qualname.split('.') {
        object = pyre_interpreter::baseobjspace::getattr_str(object, component)?;
    }
    Ok(object)
}

/// CPython `_pickle.Unpickler.find_class` preserves the AttributeError raised
/// by an intermediate dotted-path lookup as the context of its path-level
/// AttributeError.  Keep both objects alive while linking the chain.
fn attribute_error_with_context(message: String, mut context: PyError) -> PyError {
    let _roots = pyre_object::gc_roots::push_roots();
    let w_context = context.to_exc_object();
    let _ = pyre_object::gc_roots::pin_root(w_context);
    let context_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let mut error = PyError::attribute_error(message);
    let w_error = error.to_exc_object();
    unsafe {
        pyre_object::interp_exceptions::w_exception_set_context(
            w_error,
            pyre_object::gc_roots::shadow_stack_get(context_slot),
        );
    }
    error
}

/// `interp_pickle.py UnpicklerMemoProxy` — a live view of an unpickler's
/// memo. `copy` / `clear` / assignment follow the 3.15 dense-array surface.
///
/// Held in its own module so `#[pyre_methods]` emits a `type_object()` that
/// does not clash with `W_Unpickler`'s (each impl emits a module-scoped one).
pub use memo_proxy::UnpicklerMemoProxy;

mod memo_proxy {
    use super::*;

    // Modules/_pickle.c:pickle_exec CREATE_TYPEs unpickler_memoproxy_spec too.
    #[pyre_interpreter::pyre_class("_pickle.UnpicklerMemoProxy", cpython_heaptype)]
    pub struct UnpicklerMemoProxy {
        pub(super) w_unpickler: PyObjectRef,
    }

    #[pyre_interpreter::pyre_methods(doc = "Proxy for an Unpickler's memo.")]
    impl UnpicklerMemoProxy {
        /// `_pickle_UnpicklerMemoProxy_copy_impl` — occupied dense-array slots
        /// only; `memo_dict` is not included.
        fn copy(&self) -> Result<PyObjectRef, PyError> {
            let _roots = pyre_object::gc_roots::push_roots();
            let _ = pyre_object::gc_roots::pin_root(self.w_unpickler);
            let unpickler_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let w_dict = pyre_object::dictmultiobject::w_dict_new();
            let _ = pyre_object::gc_roots::pin_root(w_dict);
            let dict_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let w_memo = unsafe {
                &*(pyre_object::gc_roots::shadow_stack_get(unpickler_slot) as *const W_Unpickler)
            }
            .w_memo;
            if !unsafe { pyre_object::is_none(w_memo) } {
                let _ = pyre_object::gc_roots::pin_root(w_memo);
                let memo_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
                let len = unsafe {
                    pyre_object::listobject::w_list_len(pyre_object::gc_roots::shadow_stack_get(
                        memo_slot,
                    ))
                } as i64;
                for i in 0..len {
                    let v = unsafe {
                        pyre_object::listobject::w_list_getitem(
                            pyre_object::gc_roots::shadow_stack_get(memo_slot),
                            i,
                        )
                    };
                    if let Some(v) = v
                        && !v.is_null()
                    {
                        // `w_dict_setitem` boxes the int key (`w_int_new`), which
                        // may collect and move `v`; pin it across the store.
                        let _r = pyre_object::gc_roots::push_roots();
                        let _ = pyre_object::gc_roots::pin_root(v);
                        let v_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
                        let w_dict = pyre_object::gc_roots::shadow_stack_get(dict_slot);
                        unsafe {
                            pyre_object::dictmultiobject::w_dict_setitem(
                                w_dict,
                                i,
                                pyre_object::gc_roots::shadow_stack_get(v_slot),
                            )
                        };
                    }
                }
            }
            Ok(pyre_object::gc_roots::shadow_stack_get(dict_slot))
        }

        /// `_pickle_UnpicklerMemoProxy_clear_impl` — `_Unpickler_MemoCleanup`
        /// then `_Unpickler_NewMemo` of the same dense size. `memo_dict` is
        /// not touched.
        fn clear(&self) {
            let _roots = pyre_object::gc_roots::push_roots();
            let _ = pyre_object::gc_roots::pin_root(self.w_unpickler);
            let slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let empty = pyre_object::listobject::w_list_new_empty();
            let _ = pyre_object::gc_roots::pin_root(empty);
            let empty_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            super::memo_replace_dense(slot, pyre_object::gc_roots::shadow_stack_get(empty_slot));
        }
    }
}

/// Re-read the (possibly relocated) unpickler from the pinned shadow slot.
#[inline]
fn cur(slot: usize) -> &'static mut W_Unpickler {
    unsafe { &mut *(pyre_object::gc_roots::shadow_stack_get(slot) as *mut W_Unpickler) }
}

/// Old-to-young / incremental-mark write barrier for a store into the
/// unpickler's own GC-pointer fields (`w_stack` / `w_metastack` / `w_memo` /
/// `w_frame` / …). RPython's GC transform emits `ll_writebarrier` after every
/// such store; the hand-written port must call it, or an incremental major
/// collection sweeps a freshly stored (white) child that the already-marked
/// unpickler still references — leaving a dangling field the next mark trips
/// over.
#[inline]
fn unpickler_write_barrier(obj: PyObjectRef) {
    pyre_object::gc_hook::try_gc_write_barrier(obj as pyre_object::gc_hook::GCREF);
}

// ── stack / metastack helpers ────────────────────────────────────────

fn push(slot: usize, obj: PyObjectRef) {
    let me = cur(slot);
    unsafe { pyre_object::listobject::w_list_append(me.w_stack, obj) };
}

/// `data_pop` — pop the top of the current stack.
fn pop(slot: usize) -> Result<PyObjectRef, PyError> {
    let me = cur(slot);
    unsafe { pyre_object::listobject::w_list_pop_end(me.w_stack) }
        .ok_or_else(|| unpickling_error("unpickling stack underflow"))
}

/// `_stack_top` — the top of the current stack without removing it.
fn top(slot: usize, opcode_name: &str) -> Result<PyObjectRef, PyError> {
    let me = cur(slot);
    let n = unsafe { pyre_object::listobject::w_list_len(me.w_stack) };
    if n < 1 {
        return Err(unpickling_error(&format!("stack empty in {opcode_name}")));
    }
    Ok(unsafe { pyre_object::listobject::w_list_getitem(me.w_stack, (n - 1) as i64).unwrap() })
}

/// `load_mark` — save the current stack and start a fresh one.
fn mark(slot: usize) {
    let me = cur(slot);
    unsafe { pyre_object::listobject::w_list_append(me.w_metastack, me.w_stack) };
    let new_stack = pyre_object::listobject::w_list_new_empty();
    let me = cur(slot);
    me.w_stack = new_stack;
    unpickler_write_barrier(me as *mut W_Unpickler as PyObjectRef);
}

/// `pop_mark` — return the items pushed since the last MARK and restore the
/// previous stack.
fn pop_mark(slot: usize) -> Result<PyObjectRef, PyError> {
    let me = cur(slot);
    let items = me.w_stack;
    let prev = unsafe { pyre_object::listobject::w_list_pop_end(me.w_metastack) }
        .ok_or_else(|| unpickling_error("no items on stack"))?;
    let me = cur(slot);
    me.w_stack = prev;
    unpickler_write_barrier(me as *mut W_Unpickler as PyObjectRef);
    Ok(items)
}

// ── out-of-band buffers ──────────────────────────────────────────────

/// `load_next_buffer` (NEXT_BUFFER) — push the next buffer from the
/// `buffers` iterator given at construction.
fn load_next_buffer(slot: usize) -> Result<(), PyError> {
    let w_buffers = cur(slot).w_buffers;
    if unsafe { pyre_object::is_none(w_buffers) } {
        return Err(unpickling_error(
            "pickle stream refers to out-of-band data but no *buffers* argument was given",
        ));
    }
    let next_buf = pyre_interpreter::baseobjspace::next(w_buffers);
    let w_buf = match next_buf {
        Ok(b) => b,
        Err(e) => {
            let (stop, e) = e.matches_stop_iteration_keep();
            return Err(if stop {
                unpickling_error("not enough out-of-band buffers")
            } else {
                e
            });
        }
    };
    push(slot, w_buf);
    Ok(())
}

/// `load_readonly_buffer` (READONLY_BUFFER) — replace the top buffer with a
/// read-only memoryview onto it.
fn load_readonly_buffer(slot: usize) -> Result<(), PyError> {
    let mut w_buf = top(slot, "READONLY_BUFFER")?;
    let w_memoryview = pyre_object::with_roots!(w_buf => memoryview_type())?;
    let w_mv = call_fn(w_memoryview, &[w_buf])?;
    let w_readonly = call_meth(w_mv, "toreadonly", &[])?;
    // Replace the top of the stack (`stack[-1] = w_readonly`).
    pop(slot)?;
    push(slot, w_readonly);
    Ok(())
}

/// The `memoryview` builtin type via the live execution context.
fn memoryview_type() -> Result<PyObjectRef, PyError> {
    let ec = pyre_interpreter::call::getexecutioncontext();
    if ec.is_null() {
        return Err(unpickling_error("memoryview type unavailable"));
    }
    unsafe { (*ec).lookup_builtin("memoryview") }
        .ok_or_else(|| unpickling_error("memoryview type unavailable"))
}

// ── memo helpers ─────────────────────────────────────────────────────

/// `_Unpickler_MemoPut` — store `w_val` at index `i`. A dense index grows the
/// list (NULL-filling any gap); a sparse index (`i >= len` and `i > len * 2`)
/// goes into `w_memo_dict` instead.
fn memo_put(slot: usize, i: i64, w_val: PyObjectRef) {
    let len = unsafe { pyre_object::listobject::w_list_len(cur(slot).w_memo) } as i64;
    if i >= len && i > len.saturating_mul(2) {
        let _roots = pyre_object::gc_roots::push_roots();
        let _ = pyre_object::gc_roots::pin_root(w_val);
        let val_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        if unsafe { pyre_object::is_none(cur(slot).w_memo_dict) } {
            let w_dict = pyre_object::dictmultiobject::w_dict_new();
            let me = cur(slot);
            me.w_memo_dict = w_dict;
            unpickler_write_barrier(me as *mut W_Unpickler as PyObjectRef);
        }
        let _ = pyre_object::gc_roots::pin_root(cur(slot).w_memo_dict);
        let dict_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem(
                pyre_object::gc_roots::shadow_stack_get(dict_slot),
                i,
                pyre_object::gc_roots::shadow_stack_get(val_slot),
            )
        };
    } else if i < len {
        // In-range overwrite: no allocation, so no relocation.
        unsafe { pyre_object::listobject::w_list_setitem(cur(slot).w_memo, i, w_val) };
    } else {
        // Grow. `w_list_append` may collect; pin the value and re-read self/list.
        let _roots = pyre_object::gc_roots::push_roots();
        let _ = pyre_object::gc_roots::pin_root(w_val);
        let val_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        // NULL-fill through index `i` (inclusive). Appending `PY_NULL` forces the
        // Object strategy, so memoized values keep pointer identity on GET.
        while (unsafe { pyre_object::listobject::w_list_len(cur(slot).w_memo) } as i64) <= i {
            unsafe {
                pyre_object::listobject::w_list_append(cur(slot).w_memo, pyre_object::PY_NULL)
            };
        }
        unsafe {
            pyre_object::listobject::w_list_setitem(
                cur(slot).w_memo,
                i,
                pyre_object::gc_roots::shadow_stack_get(val_slot),
            )
        };
    }
    let me = cur(slot);
    if i >= me.memo_index {
        me.memo_index = i + 1;
    }
}

/// `_memo_append` — store `w_val` at the next free slot.
fn memo_append(slot: usize, w_val: PyObjectRef) {
    let i = cur(slot).memo_index;
    memo_put(slot, i, w_val);
}

fn memo_get(slot: usize, i: i64) -> Result<PyObjectRef, PyError> {
    // A negative index must not wrap (the list would index from the end); treat
    // it as absent, matching the prior dict lookup.
    if i < 0 {
        return Err(unpickling_error(&format!(
            "Memo value not found at index {i}"
        )));
    }
    match unsafe { pyre_object::listobject::w_list_getitem(cur(slot).w_memo, i) } {
        Some(v) if !v.is_null() => return Ok(v),
        _ => {}
    }
    let w_memo_dict = cur(slot).w_memo_dict;
    if unsafe { pyre_object::is_none(w_memo_dict) } {
        return Err(unpickling_error(&format!(
            "Memo value not found at index {i}"
        )));
    }
    // `_Unpickler_MemoGet`: box the index, then either migrate a dict entry
    // into a live list slot or read it in place. `w_int_new` can collect.
    let _roots = pyre_object::gc_roots::push_roots();
    let _ = pyre_object::gc_roots::pin_root(w_memo_dict);
    let dict_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let w_key = pyre_object::w_int_new(i);
    let _ = pyre_object::gc_roots::pin_root(w_key);
    let key_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let len = unsafe { pyre_object::listobject::w_list_len(cur(slot).w_memo) } as i64;
    let w_memo_dict = pyre_object::gc_roots::shadow_stack_get(dict_slot);
    let w_key = pyre_object::gc_roots::shadow_stack_get(key_slot);
    let found = if i < len {
        match unsafe { pyre_object::dictmultiobject::w_dict_pop_checked(w_memo_dict, w_key) } {
            Ok(Some(v)) => {
                let _ = pyre_object::gc_roots::pin_root(v);
                let val_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
                unsafe {
                    pyre_object::listobject::w_list_setitem(
                        cur(slot).w_memo,
                        i,
                        pyre_object::gc_roots::shadow_stack_get(val_slot),
                    )
                };
                Some(pyre_object::gc_roots::shadow_stack_get(val_slot))
            }
            _ => None,
        }
    } else {
        unsafe { pyre_object::dictmultiobject::w_dict_lookup(w_memo_dict, w_key) }
    };
    match found {
        Some(v) if !v.is_null() => Ok(v),
        _ => Err(unpickling_error(&format!(
            "Memo value not found at index {i}"
        ))),
    }
}

/// Replace the dense memo list. `w_memo_dict` is left as-is
/// (`_Unpickler_MemoCleanup` does not `Py_CLEAR(memo_dict)`).
fn memo_replace_dense(slot: usize, w_list: PyObjectRef) {
    let me = cur(slot);
    me.w_memo = w_list;
    unpickler_write_barrier(me as *mut W_Unpickler as PyObjectRef);
}

/// Snapshot the source unpickler's dense list, NULL slots included
/// (`Unpickler_set_memo` proxy branch: `_Unpickler_NewMemo` + `Py_XNewRef`
/// each `memo[i]`).
fn memo_copy_dense_list(src_slot: usize) -> PyObjectRef {
    let _roots = pyre_object::gc_roots::push_roots();
    let w_src = cur(src_slot).w_memo;
    let _ = pyre_object::gc_roots::pin_root(w_src);
    let src_memo_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let dst = pyre_object::listobject::w_list_new_empty();
    let _ = pyre_object::gc_roots::pin_root(dst);
    let dst_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    if unsafe { pyre_object::is_none(pyre_object::gc_roots::shadow_stack_get(src_memo_slot)) } {
        return pyre_object::gc_roots::shadow_stack_get(dst_slot);
    }
    let len = unsafe {
        pyre_object::listobject::w_list_len(pyre_object::gc_roots::shadow_stack_get(src_memo_slot))
    } as i64;
    for i in 0..len {
        unsafe {
            pyre_object::listobject::w_list_append(
                pyre_object::gc_roots::shadow_stack_get(dst_slot),
                pyre_object::PY_NULL,
            )
        };
        let v = unsafe {
            pyre_object::listobject::w_list_getitem(
                pyre_object::gc_roots::shadow_stack_get(src_memo_slot),
                i,
            )
        };
        if let Some(v) = v
            && !v.is_null()
        {
            let _r = pyre_object::gc_roots::push_roots();
            let _ = pyre_object::gc_roots::pin_root(v);
            let v_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            unsafe {
                pyre_object::listobject::w_list_setitem(
                    pyre_object::gc_roots::shadow_stack_get(dst_slot),
                    i,
                    pyre_object::gc_roots::shadow_stack_get(v_slot),
                )
            };
        }
    }
    pyre_object::gc_roots::shadow_stack_get(dst_slot)
}

/// `Unpickler_set_memo` dict branch: `_Unpickler_NewMemo(PyDict_GET_SIZE)`,
/// `_Unpickler_MemoPut` each pair against the current memo, then
/// `_Unpickler_MemoCleanup` and install the empty array.
fn memo_assign_from_dict(self_slot: usize, w_mapping: PyObjectRef) -> Result<(), PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let _ = pyre_object::gc_roots::pin_root(w_mapping);
    let mapping_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let n = unsafe {
        pyre_object::dictmultiobject::w_dict_len(pyre_object::gc_roots::shadow_stack_get(
            mapping_slot,
        ))
    };
    let new_memo = pyre_object::listobject::w_list_new_empty();
    let _ = pyre_object::gc_roots::pin_root(new_memo);
    let new_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    for _ in 0..n {
        unsafe {
            pyre_object::listobject::w_list_append(
                pyre_object::gc_roots::shadow_stack_get(new_slot),
                pyre_object::PY_NULL,
            )
        };
    }
    let pairs = unsafe {
        pyre_object::dictmultiobject::w_dict_items(pyre_object::gc_roots::shadow_stack_get(
            mapping_slot,
        ))
    };
    let mut flat = Vec::with_capacity(pairs.len() * 2);
    for (k, v) in pairs {
        flat.push(k);
        flat.push(v);
    }
    let n_flat = flat.len();
    let base = pyre_object::gc_roots::pin_roots(&flat);
    for i in (0..n_flat).step_by(2) {
        let w_key = pyre_object::gc_roots::shadow_stack_get(base + i);
        if unsafe { !pyre_object::is_int_or_long(w_key) } {
            return Err(PyError::type_error("memo key must be integers"));
        }
        let idx = pyre_interpreter::baseobjspace::int_w(w_key)?;
        if idx < 0 {
            return Err(PyError::value_error("memo key must be positive integers."));
        }
        memo_put(
            self_slot,
            idx,
            pyre_object::gc_roots::shadow_stack_get(base + i + 1),
        );
    }
    memo_replace_dense(self_slot, pyre_object::gc_roots::shadow_stack_get(new_slot));
    Ok(())
}

// ── reading ──────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum ShortRead {
    /// The top-level dispatch could not read its next opcode.  CPython
    /// `_pickle.c:6978-6981` translates that one short read to EOFError.
    Opcode,
    /// An opcode was present but its payload was incomplete.
    Payload,
}

fn eof_error(message: &str) -> PyError {
    PyError::new(pyre_interpreter::PyErrorKind::EOFError, message)
}

/// Read the dispatch loop's next opcode byte.  A stream ending between
/// complete pickle objects is EOFError, while [`read1`] remains the payload
/// reader whose short result is UnpicklingError.
fn read_opcode(slot: usize) -> Result<u8, PyError> {
    let v = read_exact(slot, 1, ShortRead::Opcode)?;
    Ok(v[0])
}

/// Read one payload byte (from the active frame, else the file).
fn read1(slot: usize) -> Result<u8, PyError> {
    let v = read(slot, 1)?;
    Ok(v[0])
}

/// Read `n` payload bytes (from the active frame, else the file). Returns an
/// owned copy so the result survives later allocations.
fn read(slot: usize, n: usize) -> Result<Vec<u8>, PyError> {
    read_exact(slot, n, ShortRead::Payload)
}

fn read_exact(slot: usize, n: usize, short_read: ShortRead) -> Result<Vec<u8>, PyError> {
    let me = cur(slot);
    if !unsafe { pyre_object::is_none(me.w_frame) } {
        let frame = unsafe { pyre_object::bytesobject::w_bytes_data(me.w_frame) };
        let idx = me.frame_index as usize;
        if idx + n <= frame.len() {
            let out = frame[idx..idx + n].to_vec();
            me.frame_index += n as i64;
            return Ok(out);
        }
        // Frame exhausted — fall through to the file.
        me.w_frame = pyre_object::w_none();
        me.frame_index = 0;
    }
    let w_n = pyre_object::w_int_new(n as i64);
    let read_fn = cur(slot).w_file_read;
    let w_res = call_fn(read_fn, &[w_n])?;
    let data = unsafe { pyre_object::bytesobject::w_bytes_data(w_res) };
    if data.len() < n {
        return Err(match short_read {
            ShortRead::Opcode => eof_error("Ran out of input"),
            ShortRead::Payload => unpickling_error("pickle data was truncated"),
        });
    }
    Ok(data[..n].to_vec())
}

fn dispatch(slot: usize, opcode: u8) -> Result<(), PyError> {
    match opcode {
        x if x == op::PROTO => {
            let p = read1(slot)? as i64;
            if !(0..=HIGHEST_PROTOCOL).contains(&p) {
                return Err(PyError::value_error("unsupported pickle protocol"));
            }
            cur(slot).proto = p;
        }
        x if x == op::FRAME => {
            let sz = read(slot, 8)?;
            let frame_size = read_int_le(&sz) as usize;
            // Load the frame body from the file.
            let w_n = pyre_object::w_int_new(frame_size as i64);
            let read_fn = cur(slot).w_file_read;
            let w_res = call_fn(read_fn, &[w_n])?;
            let data = unsafe { pyre_object::bytesobject::w_bytes_data(w_res) };
            if data.len() < frame_size {
                return Err(unpickling_error("pickle data was truncated"));
            }
            let w_frame = pyre_object::w_bytes_from_bytes(&data[..frame_size]);
            let me = cur(slot);
            me.w_frame = w_frame;
            me.frame_index = 0;
            unpickler_write_barrier(me as *mut W_Unpickler as PyObjectRef);
        }
        x if x == op::NONE => push(slot, pyre_object::w_none()),
        x if x == op::NEWTRUE => push(slot, pyre_object::w_bool_from(true)),
        x if x == op::NEWFALSE => push(slot, pyre_object::w_bool_from(false)),
        x if x == op::BININT => {
            let d = read(slot, 4)?;
            let v = i32::from_le_bytes([d[0], d[1], d[2], d[3]]) as i64;
            push(slot, pyre_object::w_int_new(v));
        }
        x if x == op::BININT1 => {
            let d = read(slot, 1)?;
            push(slot, pyre_object::w_int_new(d[0] as i64));
        }
        x if x == op::BININT2 => {
            let d = read(slot, 2)?;
            push(
                slot,
                pyre_object::w_int_new(u16::from_le_bytes([d[0], d[1]]) as i64),
            );
        }
        x if x == op::LONG1 => {
            let n = read(slot, 1)?[0] as usize;
            let d = read(slot, n)?;
            push(slot, decode_long(&d));
        }
        x if x == op::LONG4 => {
            let nb = read(slot, 4)?;
            let n = i32::from_le_bytes([nb[0], nb[1], nb[2], nb[3]]);
            if n < 0 {
                // Corrupt or hostile pickle -- we never write one like this.
                return Err(unpickling_error("LONG pickle has negative byte count"));
            }
            let d = read(slot, n as usize)?;
            push(slot, decode_long(&d));
        }
        x if x == op::BINFLOAT => {
            let d = read(slot, 8)?;
            let f = f64::from_be_bytes([d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]]);
            push(slot, pyre_object::w_float_new(f));
        }
        x if x == op::INT => {
            let s = read_line(slot)?;
            let w = match s.as_str() {
                "00" => pyre_object::w_bool_from(false),
                "01" => pyre_object::w_bool_from(true),
                _ => parse_int_text(&s)?,
            };
            push(slot, w);
        }
        x if x == op::LONG => {
            let mut s = read_line(slot)?;
            // strip the Python 2 'L' suffix, if present.
            if s.ends_with('L') {
                s.pop();
            }
            push(slot, parse_int_text(&s)?);
        }
        x if x == op::FLOAT => {
            let s = read_line(slot)?;
            let f = s
                .trim()
                .parse::<f64>()
                .map_err(|_| PyError::value_error("could not convert string to float"))?;
            push(slot, pyre_object::w_float_new(f));
        }
        x if x == op::UNICODE => {
            // raw-unicode-escape over the line's raw bytes.
            let data = read_line_bytes(slot)?;
            let w_bytes = pyre_object::w_bytes_from_bytes(&data);
            let w_uni = call_meth(
                w_bytes,
                "decode",
                &[pyre_object::w_str_new("raw-unicode-escape")],
            )?;
            push(slot, w_uni);
        }
        x if x == op::SHORT_BINUNICODE => {
            let n = read(slot, 1)?[0] as usize;
            let d = read(slot, n)?;
            push(slot, str_from_utf8(&d)?);
        }
        x if x == op::BINUNICODE => {
            let nb = read(slot, 4)?;
            let n = u32::from_le_bytes([nb[0], nb[1], nb[2], nb[3]]) as usize;
            let d = read(slot, n)?;
            push(slot, str_from_utf8(&d)?);
        }
        x if x == op::BINUNICODE8 => {
            let nb = read(slot, 8)?;
            let n = read_int_le(&nb) as usize;
            let d = read(slot, n)?;
            push(slot, str_from_utf8(&d)?);
        }
        x if x == op::SHORT_BINBYTES => {
            let n = read(slot, 1)?[0] as usize;
            let d = read(slot, n)?;
            push(slot, pyre_object::w_bytes_from_bytes(&d));
        }
        x if x == op::BINBYTES => {
            let nb = read(slot, 4)?;
            let n = u32::from_le_bytes([nb[0], nb[1], nb[2], nb[3]]) as usize;
            let d = read(slot, n)?;
            push(slot, pyre_object::w_bytes_from_bytes(&d));
        }
        x if x == op::BINBYTES8 => {
            let nb = read(slot, 8)?;
            let n = read_int_le(&nb) as usize;
            let d = read(slot, n)?;
            push(slot, pyre_object::w_bytes_from_bytes(&d));
        }
        // ── legacy protocol-0/1 str ───────────────────────────────────
        x if x == op::STRING => {
            // A protocol-0 quoted py2 str: strip the matching outer quotes,
            // decode the bytes-literal escapes, then apply `encoding`/`errors`.
            let line = read_line_bytes(slot)?;
            let data = strip_string_quotes(&line)?;
            let raw = escape_decode(&data)?;
            let w = decode_string(slot, &raw)?;
            push(slot, w);
        }
        x if x == op::BINSTRING => {
            // Deprecated BINSTRING uses a signed 32-bit length.
            let nb = read(slot, 4)?;
            let n = i32::from_le_bytes([nb[0], nb[1], nb[2], nb[3]]);
            if n < 0 {
                return Err(unpickling_error("BINSTRING pickle has negative byte count"));
            }
            let d = read(slot, n as usize)?;
            let w = decode_string(slot, &d)?;
            push(slot, w);
        }
        x if x == op::SHORT_BINSTRING => {
            let n = read(slot, 1)?[0] as usize;
            let d = read(slot, n)?;
            let w = decode_string(slot, &d)?;
            push(slot, w);
        }
        // ── stack ────────────────────────────────────────────────────
        x if x == op::MARK => mark(slot),
        x if x == op::POP => {
            // Pop a stack item, or discard the topmost MARK group.
            let me = cur(slot);
            let n = unsafe { pyre_object::listobject::w_list_len(me.w_stack) };
            if n > 0 {
                pop(slot)?;
            } else {
                pop_mark(slot)?;
            }
        }
        x if x == op::POP_MARK => {
            pop_mark(slot)?;
        }
        x if x == op::DUP => {
            let v = top(slot, "DUP")?;
            push(slot, v);
        }
        // ── tuple ─────────────────────────────────────────────────────
        x if x == op::EMPTY_TUPLE => push(slot, pyre_object::tupleobject::w_tuple_new(Vec::new())),
        x if x == op::TUPLE => {
            let items = pop_mark(slot)?;
            push(slot, list_to_tuple(items));
        }
        x if x == op::TUPLE1 => {
            let a = pop(slot)?;
            push(slot, pyre_object::tupleobject::w_tuple_new(vec![a]));
        }
        x if x == op::TUPLE2 => {
            let b = pop(slot)?;
            let a = pop(slot)?;
            push(slot, pyre_object::tupleobject::w_tuple_new(vec![a, b]));
        }
        x if x == op::TUPLE3 => {
            let c = pop(slot)?;
            let b = pop(slot)?;
            let a = pop(slot)?;
            push(slot, pyre_object::tupleobject::w_tuple_new(vec![a, b, c]));
        }
        // ── list ──────────────────────────────────────────────────────
        x if x == op::EMPTY_LIST => push(slot, pyre_object::listobject::w_list_new(Vec::new())),
        x if x == op::LIST => {
            let items = pop_mark(slot)?;
            push(slot, list_copy(items));
        }
        x if x == op::APPEND => {
            let mut value = pop(slot)?;
            let w_list = pyre_object::with_roots!(value => top(slot, "APPEND"))?;
            call_meth(w_list, "append", &[value])?;
        }
        x if x == op::APPENDS => {
            let items = pop_mark(slot)?;
            // Pin `items` so the `extend`/`append` lookups (which may allocate
            // and collect) do not strand it; re-read the list top per call.
            let _roots = pyre_object::gc_roots::push_roots();
            let _ = pyre_object::gc_roots::pin_root(items);
            let items_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let w_list = top(slot, "APPENDS")?;
            match pyre_interpreter::baseobjspace::findattr_result(w_list, "extend")? {
                Some(extend) if !unsafe { pyre_object::is_none(extend) } => {
                    call_fn(
                        extend,
                        &[pyre_object::gc_roots::shadow_stack_get(items_slot)],
                    )?;
                }
                _ => {
                    // PEP 307 requires extend(); fall back to append() for
                    // objects lacking it (backward compatibility).
                    let n = unsafe {
                        pyre_object::listobject::w_list_len(
                            pyre_object::gc_roots::shadow_stack_get(items_slot),
                        )
                    };
                    for i in 0..n {
                        let w_list = top(slot, "APPENDS")?;
                        let item = unsafe {
                            pyre_object::listobject::w_list_getitem(
                                pyre_object::gc_roots::shadow_stack_get(items_slot),
                                i as i64,
                            )
                            .unwrap()
                        };
                        call_meth(w_list, "append", &[item])?;
                    }
                }
            }
        }
        // ── dict ──────────────────────────────────────────────────────
        x if x == op::EMPTY_DICT => push(slot, pyre_object::dictmultiobject::w_dict_new()),
        x if x == op::DICT => {
            let items = pop_mark(slot)?;
            let w_dict = pyre_object::dictmultiobject::w_dict_new();
            let w_dict = dict_update_from_pairs(w_dict, items)?;
            push(slot, w_dict);
        }
        x if x == op::SETITEM => {
            let mut value = pop(slot)?;
            let mut key = pop(slot)?;
            let w_dict = pyre_object::with_roots!(key, value => top(slot, "SETITEM"))?;
            pyre_interpreter::baseobjspace::setitem(w_dict, key, value)?;
        }
        x if x == op::SETITEMS => {
            let mut items = pop_mark(slot)?;
            let w_dict = pyre_object::with_roots!(items => top(slot, "SETITEMS"))?;
            dict_update_from_pairs(w_dict, items)?;
        }
        // ── set / frozenset ───────────────────────────────────────────
        x if x == op::EMPTY_SET => push(slot, pyre_object::setobject::w_set_new()),
        x if x == op::FROZENSET => {
            let items = pop_mark(slot)?;
            push(slot, list_to_frozenset(items));
        }
        x if x == op::ADDITEMS => {
            let items = pop_mark(slot)?;
            let _roots = pyre_object::gc_roots::push_roots();
            let _ = pyre_object::gc_roots::pin_root(items);
            let items_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let w_set = top(slot, "ADDITEMS")?;
            let w_set_type =
                pyre_interpreter::typedef::gettypeobject(&pyre_object::setobject::SET_TYPE);
            if unsafe { pyre_interpreter::baseobjspace::isinstance_w(w_set, w_set_type) } {
                // `PySet_Check`: `set` and its subclasses (not `frozenset`)
                // take the `update` path; `set.update(items)` dispatches
                // through the (possibly overridden) method.
                call_meth(
                    w_set,
                    "update",
                    &[pyre_object::gc_roots::shadow_stack_get(items_slot)],
                )?;
            } else {
                let n = unsafe {
                    pyre_object::listobject::w_list_len(pyre_object::gc_roots::shadow_stack_get(
                        items_slot,
                    ))
                };
                for i in 0..n {
                    let w_set = top(slot, "ADDITEMS")?;
                    let item = unsafe {
                        pyre_object::listobject::w_list_getitem(
                            pyre_object::gc_roots::shadow_stack_get(items_slot),
                            i as i64,
                        )
                        .unwrap()
                    };
                    call_meth(w_set, "add", &[item])?;
                }
            }
        }
        // ── bytearray ─────────────────────────────────────────────────
        x if x == op::BYTEARRAY8 => {
            let nb = read(slot, 8)?;
            let n = read_int_le(&nb) as usize;
            let d = read(slot, n)?;
            push(
                slot,
                pyre_object::bytearrayobject::w_bytearray_from_bytes(&d),
            );
        }
        // ── proto-5 out-of-band buffers ───────────────────────────────
        x if x == op::NEXT_BUFFER => load_next_buffer(slot)?,
        x if x == op::READONLY_BUFFER => load_readonly_buffer(slot)?,
        // ── global / reduce / build ───────────────────────────────────
        x if x == op::GLOBAL => {
            let module = read_line(slot)?;
            let name = read_line(slot)?;
            // `_pickle.c load_global`: both newline-terminated operands must
            // contain at least one byte. Letting an empty module reach
            // `__import__` leaks its incidental `ValueError("Empty module
            // name")`; the opcode contract reports truncated pickle data.
            if module.is_empty() || name.is_empty() {
                return Err(unpickling_error("pickle data was truncated"));
            }
            // interp_pickle.py:2539-2546 — PyPy's compact homogeneous-list
            // globals are internal sentinels, not importable Python callables.
            if module == "pypy._builtin" {
                let kind = match name.as_str() {
                    "_ascii_list_unpickle" => Some(PackedListKind::Ascii),
                    "_bytes_list_unpickle" => Some(PackedListKind::Bytes),
                    _ => None,
                };
                if let Some(kind) = kind {
                    push(slot, packed_list_sentinel(kind));
                    return Ok(());
                }
            }
            let v = call_find_class_names(slot, &module, &name)?;
            push(slot, v);
        }
        x if x == op::STACK_GLOBAL => {
            let w_name = pop(slot)?;
            let w_module = pop(slot)?;
            if !unsafe { pyre_object::is_str(w_name) } || !unsafe { pyre_object::is_str(w_module) }
            {
                return Err(unpickling_error("STACK_GLOBAL requires str"));
            }
            let v = call_find_class(slot, w_module, w_name)?;
            push(slot, v);
        }
        x if x == op::EXT1 => {
            let code = read(slot, 1)?[0] as i64;
            get_extension(slot, code)?;
        }
        x if x == op::EXT2 => {
            let d = read(slot, 2)?;
            let code = u16::from_le_bytes([d[0], d[1]]) as i64;
            get_extension(slot, code)?;
        }
        x if x == op::EXT4 => {
            let d = read(slot, 4)?;
            let code = i32::from_le_bytes([d[0], d[1], d[2], d[3]]) as i64;
            get_extension(slot, code)?;
        }
        x if x == op::REDUCE => {
            let mut w_args = pop(slot)?;
            let w_func = pyre_object::with_roots!(w_args => top(slot, "REDUCE"))?;
            let _roots = pyre_object::gc_roots::push_roots();
            let _ = pyre_object::gc_roots::pin_root(w_args);
            let args_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let _ = pyre_object::gc_roots::pin_root(w_func);
            let func_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            // `_pickle.c:load_reduce` rejects a non-tuple argument list before
            // the callable ever runs, so an iterable operand is a stream error
            // rather than something argument expansion reports.
            if !unsafe { pyre_object::is_tuple(pyre_object::gc_roots::shadow_stack_get(args_slot)) }
            {
                return Err(PyError::type_error("argument list must be a tuple"));
            }
            let packed_kind = if pyre_interpreter::baseobjspace::is_w(
                pyre_object::gc_roots::shadow_stack_get(func_slot),
                packed_list_sentinel(PackedListKind::Ascii),
            ) {
                Some(PackedListKind::Ascii)
            } else if pyre_interpreter::baseobjspace::is_w(
                pyre_object::gc_roots::shadow_stack_get(func_slot),
                packed_list_sentinel(PackedListKind::Bytes),
            ) {
                Some(PackedListKind::Bytes)
            } else {
                None
            };
            if let Some(kind) = packed_kind {
                // interp_pickle.py:2628-2633 — bypass the generic Python call
                // and rebuild the specialized-list payload directly.
                let w_obj = packed_list_from_args(
                    pyre_object::gc_roots::shadow_stack_get(args_slot),
                    kind,
                )?;
                let _ = pyre_object::gc_roots::pin_root(w_obj);
                let result_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
                let me = cur(slot);
                unsafe {
                    pyre_object::listobject::w_list_setitem(
                        me.w_stack,
                        pyre_object::listobject::w_list_len(me.w_stack) as i64 - 1,
                        pyre_object::gc_roots::shadow_stack_get(result_slot),
                    );
                }
                return Ok(());
            }
            // PyPy `Arguments(space, [], w_stararg=w_args)` accepts every
            // iterable and delegates non-iterable failure to argument
            // expansion; REDUCE does not impose a tuple-only shortcut.
            let mut args = Vec::new();
            pyre_interpreter::argument::combine_starargs_wrapped(
                &mut args,
                pyre_object::gc_roots::shadow_stack_get(args_slot),
                pyre_object::gc_roots::shadow_stack_get(func_slot),
            )?;
            let arg_base = pyre_object::gc_roots::shadow_stack_len();
            for arg in args {
                let _ = pyre_object::gc_roots::pin_root(arg);
            }
            let rooted_args: Vec<PyObjectRef> = (arg_base
                ..pyre_object::gc_roots::shadow_stack_len())
                .map(pyre_object::gc_roots::shadow_stack_get)
                .collect();
            let w_obj = call_fn(
                pyre_object::gc_roots::shadow_stack_get(func_slot),
                &rooted_args,
            )?;
            let _ = pyre_object::gc_roots::pin_root(w_obj);
            let result_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            // The callable remains in place at stack[-1], matching PyPy's
            // `_stack_top`; replace it with the reduction result.
            let me = cur(slot);
            unsafe {
                pyre_object::listobject::w_list_setitem(
                    me.w_stack,
                    pyre_object::listobject::w_list_len(me.w_stack) as i64 - 1,
                    pyre_object::gc_roots::shadow_stack_get(result_slot),
                );
            }
        }
        x if x == op::NEWOBJ => {
            let w_args = pop(slot)?;
            let w_cls = pop(slot)?;
            let w_obj = new_instance_star(w_cls, w_args, None)?;
            push(slot, w_obj);
        }
        x if x == op::NEWOBJ_EX => {
            let w_kwargs = pop(slot)?;
            let w_args = pop(slot)?;
            let w_cls = pop(slot)?;
            let w_obj = new_instance_star(w_cls, w_args, Some(w_kwargs))?;
            push(slot, w_obj);
        }
        x if x == op::BUILD => {
            let mut w_state = pop(slot)?;
            let w_inst = pyre_object::with_roots!(w_state => top(slot, "BUILD"))?;
            build_instance(w_inst, w_state)?;
        }
        // ── memo ──────────────────────────────────────────────────────
        x if x == op::MEMOIZE => {
            let v = top(slot, "MEMOIZE")?;
            memo_append(slot, v);
        }
        x if x == op::PUT => {
            let i = read_line_int(slot)?;
            if i < 0 {
                return Err(PyError::value_error("negative PUT argument"));
            }
            let v = top(slot, "PUT")?;
            memo_put(slot, i, v);
        }
        x if x == op::BINPUT => {
            let i = read(slot, 1)?[0] as i64;
            let v = top(slot, "BINPUT")?;
            memo_put(slot, i, v);
        }
        x if x == op::LONG_BINPUT => {
            let d = read(slot, 4)?;
            let i = u32::from_le_bytes([d[0], d[1], d[2], d[3]]) as i64;
            let v = top(slot, "LONG_BINPUT")?;
            memo_put(slot, i, v);
        }
        x if x == op::GET => {
            let i = read_line_int(slot)?;
            let v = memo_get(slot, i)?;
            push(slot, v);
        }
        x if x == op::BINGET => {
            let i = read(slot, 1)?[0] as i64;
            let v = memo_get(slot, i)?;
            push(slot, v);
        }
        x if x == op::LONG_BINGET => {
            let d = read(slot, 4)?;
            let i = u32::from_le_bytes([d[0], d[1], d[2], d[3]]) as i64;
            let v = memo_get(slot, i)?;
            push(slot, v);
        }
        x if x == op::PERSID => {
            let pid = read_line_bytes(slot)?;
            if !pid.is_ascii() {
                return Err(unpickling_error(
                    "persistent IDs in protocol 0 must be ASCII strings",
                ));
            }
            let w_pid = str_from_utf8(&pid)?;
            let v = persistent_load(slot, w_pid)?;
            push(slot, v);
        }
        x if x == op::BINPERSID => {
            let w_pid = pop(slot)?;
            let v = persistent_load(slot, w_pid)?;
            push(slot, v);
        }
        x if x == op::INST => {
            let module = read_line(slot)?;
            let name = read_line(slot)?;
            let w_cls = call_find_class_names(slot, &module, &name)?;
            let w_args = pop_mark(slot)?;
            let v = instantiate(w_cls, w_args)?;
            push(slot, v);
        }
        x if x == op::OBJ => {
            let args = pop_mark(slot)?;
            let n = unsafe { pyre_object::listobject::w_list_len(args) };
            if n == 0 {
                return Err(unpickling_error("OBJ opcode with empty stack"));
            }
            let mut w_cls = unsafe { pyre_object::listobject::w_list_getitem(args, 0).unwrap() };
            let rest: Vec<PyObjectRef> = (1..n)
                .map(|i| unsafe {
                    pyre_object::listobject::w_list_getitem(args, i as i64).unwrap()
                })
                .collect();
            let w_rest =
                pyre_object::with_roots!(w_cls => pyre_object::listobject::w_list_new(rest));
            let v = instantiate(w_cls, w_rest)?;
            push(slot, v);
        }
        _ => {
            return Err(unpickling_error("unsupported opcode in this build"));
        }
    }
    Ok(())
}

/// Build a tuple from the items of a (popped) stack list.
fn list_to_tuple(items: PyObjectRef) -> PyObjectRef {
    let n = unsafe { pyre_object::listobject::w_list_len(items) };
    // `w_list_getitem` boxes on the Range/Integer/Float strategies, so each
    // item is pinned as it arrives rather than gathered into a plain `Vec`.
    // The popped list has no stack owner left, so retain and reload it too.
    let list_roots = pyre_object::gc_roots::push_roots();
    let list_slot = list_roots.publish(&[items]);
    list_roots.normalize(list_slot, 1);
    let mut rooted = pyre_object::gc_roots::RootedItems::new();
    for i in 0..n {
        rooted.push(unsafe {
            pyre_object::listobject::w_list_getitem(list_roots.get(list_slot), i as i64).unwrap()
        });
    }
    let v = rooted.take();
    pyre_object::tupleobject::w_tuple_new(v)
}

/// Build a frozenset from the items of a (popped) stack list.
fn list_to_frozenset(items: PyObjectRef) -> PyObjectRef {
    let n = unsafe { pyre_object::listobject::w_list_len(items) };
    // `w_list_getitem` boxes on the Range/Integer/Float strategies, so each
    // item is pinned as it arrives rather than gathered into a plain `Vec`.
    let list_roots = pyre_object::gc_roots::push_roots();
    let list_slot = list_roots.publish(&[items]);
    list_roots.normalize(list_slot, 1);
    let mut rooted = pyre_object::gc_roots::RootedItems::new();
    for i in 0..n {
        rooted.push(unsafe {
            pyre_object::listobject::w_list_getitem(list_roots.get(list_slot), i as i64).unwrap()
        });
    }
    let v = rooted.take();
    pyre_object::setobject::w_frozenset_from_items(&v)
}

/// Copy a (popped) stack list into a fresh list.
fn list_copy(items: PyObjectRef) -> PyObjectRef {
    let n = unsafe { pyre_object::listobject::w_list_len(items) };
    // `w_list_getitem` boxes on the Range/Integer/Float strategies, so each
    // item is pinned as it arrives rather than gathered into a plain `Vec`.
    let list_roots = pyre_object::gc_roots::push_roots();
    let list_slot = list_roots.publish(&[items]);
    list_roots.normalize(list_slot, 1);
    let mut rooted = pyre_object::gc_roots::RootedItems::new();
    for i in 0..n {
        rooted.push(unsafe {
            pyre_object::listobject::w_list_getitem(list_roots.get(list_slot), i as i64).unwrap()
        });
    }
    let v = rooted.take();
    pyre_object::listobject::w_list_new(v)
}

/// PyPy `_ascii_list_from_packed` / `_bytes_list_from_packed`.
///
/// Packed format: four-byte little-endian item count, followed by one-byte
/// length-prefixed payloads.  pyre currently has integer/float/object list
/// strategies but no AsciiList/BytesList strategy, so reconstruction uses the
/// semantically equivalent object-list representation.
fn packed_list_from_args(
    w_args: PyObjectRef,
    kind: PackedListKind,
) -> Result<PyObjectRef, PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_args);
    let w_zero = pyre_object::w_int_new(0);
    let _ = pyre_object::gc_roots::pin_root(w_zero);
    let w_packed = pyre_interpreter::baseobjspace::getitem(
        pyre_object::gc_roots::shadow_stack_get(base),
        pyre_object::gc_roots::shadow_stack_get(base + 1),
    )?;
    let _ = pyre_object::gc_roots::pin_root(w_packed);
    if unsafe {
        !pyre_interpreter::baseobjspace::isinstance_bytes_w(
            pyre_object::gc_roots::shadow_stack_get(base + 2),
        )
    } {
        return Err(PyError::type_error("expected bytes"));
    }
    let data = unsafe {
        pyre_object::bytesobject::bytes_like_data(pyre_object::gc_roots::shadow_stack_get(base + 2))
    }
    .to_vec();
    let error = match kind {
        PackedListKind::Ascii => "truncated ascii list pickle data",
        PackedListKind::Bytes => "truncated bytes list pickle data",
    };
    if data.len() < 4 {
        return Err(PyError::value_error(error));
    }
    let count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    let item_base = pyre_object::gc_roots::shadow_stack_len();
    let mut offset = 4usize;
    for _ in 0..count {
        let Some(&length) = data.get(offset) else {
            return Err(PyError::value_error(error));
        };
        offset += 1;
        let end = offset
            .checked_add(length as usize)
            .filter(|&end| end <= data.len())
            .ok_or_else(|| PyError::value_error(error))?;
        let item = match kind {
            PackedListKind::Ascii => {
                let text = std::str::from_utf8(&data[offset..end])
                    .map_err(|_| PyError::value_error(error))?;
                if !text.is_ascii() {
                    return Err(PyError::value_error(error));
                }
                pyre_object::w_str_new_managed(text)
            }
            PackedListKind::Bytes => {
                pyre_object::bytesobject::w_bytes_from_bytes(&data[offset..end])
            }
        };
        let _ = pyre_object::gc_roots::pin_root(item);
        offset = end;
    }
    let items = (item_base..pyre_object::gc_roots::shadow_stack_len())
        .map(pyre_object::gc_roots::shadow_stack_get)
        .collect();
    Ok(pyre_object::w_list_new(items))
}

/// Set `dict[items[2k]] = items[2k+1]` for each pair in a (popped) stack list.
fn dict_update_from_pairs(w_dict: PyObjectRef, items: PyObjectRef) -> Result<PyObjectRef, PyError> {
    // `pop_mark()` detached `items` from the unpickler's traced stack and the
    // DICT opcode passes a freshly allocated dictionary.  Keep both containers
    // alive while key hashing/equality can call back into Python and collect.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_dict);
    let _ = pyre_object::gc_roots::pin_root(items);
    let n = unsafe {
        pyre_object::listobject::w_list_len(pyre_object::gc_roots::shadow_stack_get(base + 1))
    };
    if n % 2 != 0 {
        return Err(unpickling_error("odd number of items for DICT"));
    }
    let mut i = 0;
    while i < n {
        let current_items = pyre_object::gc_roots::shadow_stack_get(base + 1);
        let k =
            unsafe { pyre_object::listobject::w_list_getitem(current_items, i as i64).unwrap() };
        let v = unsafe {
            pyre_object::listobject::w_list_getitem(current_items, (i + 1) as i64).unwrap()
        };
        pyre_interpreter::baseobjspace::setitem(
            pyre_object::gc_roots::shadow_stack_get(base),
            k,
            v,
        )?;
        i += 2;
    }
    Ok(pyre_object::gc_roots::shadow_stack_get(base))
}

/// Read a newline-terminated line (without the trailing newline).
fn read_line_bytes(slot: usize) -> Result<Vec<u8>, PyError> {
    let mut bytes: Vec<u8> = Vec::new();
    loop {
        let b = read1(slot)?;
        if b == b'\n' {
            break;
        }
        bytes.push(b);
    }
    Ok(bytes)
}

fn read_line(slot: usize) -> Result<String, PyError> {
    let bytes = read_line_bytes(slot)?;
    String::from_utf8(bytes)
        .map_err(|err| pyre_interpreter::typedef::utf8_decode_error(err.as_bytes()))
}

/// Read a newline-terminated decimal integer argument (GET / PUT in the
/// text protocols).
fn read_line_int(slot: usize) -> Result<i64, PyError> {
    let s = read_line(slot)?;
    s.trim()
        .parse::<i64>()
        .map_err(|_| PyError::value_error("invalid int literal"))
}

/// Dispatch to `self.find_class(module, name)` through the instance, so a
/// Python subclass override (the standard security hook) is honoured.
/// `call_find_class` for two fresh module/name strings: allocate and pin the
/// module string before allocating the name string, so the second constructor
/// cannot reclaim the first before both are rooted.
fn call_find_class_names(slot: usize, module: &str, name: &str) -> Result<PyObjectRef, PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let w_module = pyre_object::w_str_new_managed(module);
    let _ = pyre_object::gc_roots::pin_root(w_module);
    let module_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let w_name = pyre_object::w_str_new_managed(name);
    call_find_class(
        slot,
        pyre_object::gc_roots::shadow_stack_get(module_slot),
        w_name,
    )
}

fn call_find_class(
    slot: usize,
    w_module: PyObjectRef,
    w_name: PyObjectRef,
) -> Result<PyObjectRef, PyError> {
    // Resolving the bound `find_class` allocates, which can move the nursery;
    // pin the arguments (and re-read `self`) across the lookup.
    let _roots = pyre_object::gc_roots::push_roots();
    let _ = pyre_object::gc_roots::pin_root(w_module);
    let module_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let _ = pyre_object::gc_roots::pin_root(w_name);
    let name_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let method = pyre_interpreter::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(slot),
        "find_class",
    )?;
    call_fn(
        method,
        &[
            pyre_object::gc_roots::shadow_stack_get(module_slot),
            pyre_object::gc_roots::shadow_stack_get(name_slot),
        ],
    )
}

/// Emit the `pickle.find_class` audit event. `interp_pickle.py` calls
/// `space.audit(...)` and lets a blocking audit hook's error propagate.
fn audit_find_class(module: &str, name: &str) -> Result<(), PyError> {
    if let Ok(sys) = import_module("sys")
        && let Ok(audit) = pyre_interpreter::baseobjspace::getattr_str(sys, "audit")
    {
        let _roots = pyre_object::gc_roots::push_roots();
        let audit = _roots.pin_root(audit);
        let w_event = _roots.pin_root(pyre_object::w_str_new("pickle.find_class"));
        let w_module = _roots.pin_root(pyre_object::w_str_new_managed(module));
        let w_name = _roots.pin_root(pyre_object::w_str_new_managed(name));
        call_fn(audit, &[w_event, w_module, w_name])?;
    }
    Ok(())
}

fn audit_find_class_objects(w_module: PyObjectRef, w_name: PyObjectRef) -> Result<(), PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&[w_module, w_name]);
    let module_slot = base;
    let name_slot = base + 1;
    if let Ok(sys) = import_module("sys")
        && let Ok(audit) = pyre_interpreter::baseobjspace::getattr_str(sys, "audit")
    {
        call_fn(
            audit,
            &[
                pyre_object::w_str_new("pickle.find_class"),
                pyre_object::gc_roots::shadow_stack_get(module_slot),
                pyre_object::gc_roots::shadow_stack_get(name_slot),
            ],
        )?;
    }
    Ok(())
}

/// `get_extension` (EXT1 / EXT2 / EXT4) — resolve a `copyreg` extension code
/// to its registered global via `_extension_cache` / `_inverted_registry`.
fn get_extension(slot: usize, code: i64) -> Result<(), PyError> {
    let mut copyreg = import_module("copyreg")?;
    let cache = pyre_object::with_roots!(copyreg => pyre_interpreter::baseobjspace::getattr_str(copyreg, "_extension_cache"))?;
    if let Some(obj) = unsafe { pyre_object::w_dict_lookup(cache, pyre_object::w_int_new(code)) } {
        push(slot, obj);
        return Ok(());
    }
    let inverted = pyre_interpreter::baseobjspace::getattr_str(copyreg, "_inverted_registry")?;
    let key = match unsafe { pyre_object::w_dict_lookup(inverted, pyre_object::w_int_new(code)) } {
        Some(k) if !unsafe { pyre_object::is_none(k) } => k,
        _ => {
            if code <= 0 {
                // Corrupt or hostile pickle (0 is forbidden by add_extension).
                return Err(unpickling_error("EXT specifies code <= 0"));
            }
            return Err(PyError::value_error(format!(
                "unregistered extension code {code}"
            )));
        }
    };
    // `key` is the `(module, name)` tuple registered for this code.
    let w_module = unsafe { pyre_object::tupleobject::w_tuple_getitem(key, 0).unwrap() };
    let w_name = unsafe { pyre_object::tupleobject::w_tuple_getitem(key, 1).unwrap() };
    let obj = call_find_class(slot, w_module, w_name)?;
    // `_extension_cache[code] = obj`; pin `obj` across the dict lookups.
    let _roots = pyre_object::gc_roots::push_roots();
    let _ = pyre_object::gc_roots::pin_root(obj);
    let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let copyreg = import_module("copyreg")?;
    let cache = pyre_interpreter::baseobjspace::getattr_str(copyreg, "_extension_cache")?;
    pyre_interpreter::baseobjspace::setitem(
        cache,
        pyre_object::w_int_new(code),
        pyre_object::gc_roots::shadow_stack_get(obj_slot),
    )?;
    push(slot, pyre_object::gc_roots::shadow_stack_get(obj_slot));
    Ok(())
}

/// `_decode_string` — turn the raw bytes of a STRING / BINSTRING /
/// SHORT_BINSTRING into the object the unpickler pushes: the raw `bytes` when
/// `encoding == "bytes"`, otherwise `bytes.decode(encoding, errors)`.
fn decode_string(slot: usize, data: &[u8]) -> Result<PyObjectRef, PyError> {
    let (encoding, errors, as_bytes) = {
        let me = cur(slot);
        (
            me.encoding.clone(),
            me.errors.clone(),
            me.encoding == "bytes",
        )
    };
    if as_bytes {
        return Ok(pyre_object::w_bytes_from_bytes(data));
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let w_bytes = pyre_object::w_bytes_from_bytes(data);
    let _ = pyre_object::gc_roots::pin_root(w_bytes);
    let b = pyre_object::gc_roots::shadow_stack_len() - 1;
    let w_encoding = pyre_object::w_str_new_managed(&encoding);
    let _ = pyre_object::gc_roots::pin_root(w_encoding);
    let e = pyre_object::gc_roots::shadow_stack_len() - 1;
    let w_errors = pyre_object::w_str_new_managed(&errors);
    let _ = pyre_object::gc_roots::pin_root(w_errors);
    let err_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    call_meth(
        pyre_object::gc_roots::shadow_stack_get(b),
        "decode",
        &[
            pyre_object::gc_roots::shadow_stack_get(e),
            pyre_object::gc_roots::shadow_stack_get(err_slot),
        ],
    )
}

/// Strip the matching outer quotes from a protocol-0 STRING argument.
fn strip_string_quotes(line: &[u8]) -> Result<Vec<u8>, PyError> {
    if line.len() >= 2 && line[0] == line[line.len() - 1] && (line[0] == b'"' || line[0] == b'\'') {
        Ok(line[1..line.len() - 1].to_vec())
    } else {
        Err(unpickling_error(
            "the STRING opcode argument must be quoted",
        ))
    }
}

/// `codecs.escape_decode` over a byte string — decode the Python bytes-literal
/// escapes (`PyBytes_DecodeEscape` semantics, `strict`). Unrecognised escapes
/// keep the backslash verbatim.
fn escape_decode(data: &[u8]) -> Result<Vec<u8>, PyError> {
    let mut out = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        let c = data[i];
        if c != b'\\' {
            out.push(c);
            i += 1;
            continue;
        }
        i += 1;
        if i >= data.len() {
            return Err(PyError::value_error("Trailing \\ in string"));
        }
        let e = data[i];
        i += 1;
        match e {
            b'\n' => {} // line continuation
            b'\\' => out.push(b'\\'),
            b'\'' => out.push(b'\''),
            b'"' => out.push(b'"'),
            b'a' => out.push(0x07),
            b'b' => out.push(0x08),
            b'f' => out.push(0x0c),
            b'n' => out.push(b'\n'),
            b'r' => out.push(b'\r'),
            b't' => out.push(b'\t'),
            b'v' => out.push(0x0b),
            b'0'..=b'7' => {
                // up to three octal digits (the first already consumed).
                let mut val = (e - b'0') as u32;
                let mut k = 0;
                while k < 2 && i < data.len() && (b'0'..=b'7').contains(&data[i]) {
                    val = val * 8 + (data[i] - b'0') as u32;
                    i += 1;
                    k += 1;
                }
                out.push(val as u8);
            }
            b'x' => {
                let hi = data.get(i).and_then(|&b| (b as char).to_digit(16));
                let lo = data.get(i + 1).and_then(|&b| (b as char).to_digit(16));
                match (hi, lo) {
                    (Some(h), Some(l)) => {
                        out.push((h * 16 + l) as u8);
                        i += 2;
                    }
                    _ => return Err(PyError::value_error("invalid \\x escape in STRING")),
                }
            }
            other => {
                // Unrecognised escape: keep the backslash and the character.
                out.push(b'\\');
                out.push(other);
            }
        }
    }
    Ok(out)
}

/// Resolve and invoke `self.persistent_load(pid)` (PERSID / BINPERSID).
fn persistent_load(slot: usize, w_pid: PyObjectRef) -> Result<PyObjectRef, PyError> {
    // `w_pid` was just popped from the stack / freshly built for the opcode and
    // is not in a GC-walked container; resolving the bound `persistent_load`
    // allocates and can move the nursery, so pin it across the lookup.
    let _roots = pyre_object::gc_roots::push_roots();
    let _ = pyre_object::gc_roots::pin_root(w_pid);
    let pid_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let self_obj = pyre_object::gc_roots::shadow_stack_get(slot);
    // `findattr_result` propagates a descriptor's own error instead of panicking;
    // an explicit `persistent_load = None` is kept as the hook so `call_fn(None,
    // pid)` raises `TypeError: 'NoneType' object is not callable` (only an absent
    // attribute disables the hook).
    match pyre_interpreter::baseobjspace::findattr_result(self_obj, "persistent_load")? {
        Some(f) => call_fn(f, &[pyre_object::gc_roots::shadow_stack_get(pid_slot)]),
        None => Err(unpickling_error(
            "A load persistent id instruction was encountered, but no persistent_load function was specified.",
        )),
    }
}

/// `_instantiate` — build an old-style INST / OBJ instance. With args, or a
/// non-type class, or a `__getinitargs__`, call the class; otherwise build
/// via `__new__` without invoking `__init__`.
fn instantiate(mut w_cls: PyObjectRef, mut w_args: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let n = unsafe { pyre_object::listobject::w_list_len(w_args) };
    let has_getinitargs = pyre_object::with_roots!(w_args, w_cls => pyre_interpreter::baseobjspace::findattr_result(w_cls, "__getinitargs__"))?.is_some();
    let is_type = unsafe { pyre_object::typeobject::is_type(w_cls) };
    if n > 0 || !is_type || has_getinitargs {
        let args: Vec<PyObjectRef> = (0..n)
            .map(|i| unsafe { pyre_object::listobject::w_list_getitem(w_args, i as i64).unwrap() })
            .collect();
        call_fn(w_cls, &args)
    } else {
        new_instance(w_cls, &[])
    }
}

/// `cls.__new__(cls, *args)`.
fn new_instance(w_cls: PyObjectRef, args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    if !unsafe { pyre_object::typeobject::is_type(w_cls) } {
        let message = format!(
            "NEWOBJ class argument must be a type, not {}",
            pyre_interpreter::type_methods::arg_type_name(w_cls),
        );
        return Err(unpickling_error(&message));
    }
    let nargs = args.len();
    let mut live = Vec::with_capacity(nargs + 1);
    live.push(w_cls);
    live.extend_from_slice(args);
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.pin_roots(&live);
    let w_new =
        pyre_interpreter::baseobjspace::getattr_str_impl(roots.get(base), "__new__", true, false)
            .map_err(|mut err| {
            err.enrich_attribute_error_str(roots.get(base), "__new__");
            err
        })?;
    let w_cls = roots.get(base);
    let mut reloaded_args = vec![pyre_object::PY_NULL; nargs];
    pyre_object::gc_roots::shadow_stack_copy_range(base + 1, &mut reloaded_args);
    drop(roots);
    let mut call_args = vec![w_cls];
    call_args.extend_from_slice(&reloaded_args);
    call_fn(w_new, &call_args)
}

/// `load_newobj` / `load_newobj_ex`: validate the three wire operands, resolve
/// `cls.__new__`, then build `Arguments([cls], w_stararg=args,
/// w_starstararg=kwargs)`.  The operand shapes are checked here because they
/// are stream errors — a malformed pickle must not reach the callable and
/// surface argument expansion's own TypeError instead.
fn new_instance_star(
    w_cls: PyObjectRef,
    w_args: PyObjectRef,
    w_kwargs: Option<PyObjectRef>,
) -> Result<PyObjectRef, PyError> {
    let opname = if w_kwargs.is_some() {
        "NEWOBJ_EX"
    } else {
        "NEWOBJ"
    };
    if !unsafe { pyre_object::typeobject::is_type(w_cls) } {
        return Err(unpickling_error(&format!(
            "{opname} class argument must be a type, not {}",
            pyre_interpreter::type_methods::arg_type_name(w_cls),
        )));
    }
    if !unsafe { pyre_object::is_tuple(w_args) } {
        return Err(unpickling_error(&format!(
            "{opname} args argument must be a tuple, not {}",
            pyre_interpreter::type_methods::arg_type_name(w_args),
        )));
    }
    if let Some(w_kwargs) = w_kwargs
        && !unsafe { pyre_object::is_dict(w_kwargs) }
    {
        return Err(unpickling_error(&format!(
            "{opname} kwargs argument must be a dict, not {}",
            pyre_interpreter::type_methods::arg_type_name(w_kwargs),
        )));
    }
    // One bracket for the live operands. `getattr_str` opens its own, so
    // pin here and call `getattr_str_impl` into this stack — `load_newobj`
    // then `space.getattr` / `space.call_args` with Arguments livevars.
    let _roots = pyre_object::gc_roots::push_roots();
    let _ = pyre_object::gc_roots::pin_root(w_cls);
    let cls_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let _ = pyre_object::gc_roots::pin_root(w_args);
    let star_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let kwargs_slot = w_kwargs.map(|kwargs| {
        let _ = pyre_object::gc_roots::pin_root(kwargs);
        pyre_object::gc_roots::shadow_stack_len() - 1
    });
    let w_new = pyre_interpreter::baseobjspace::getattr_str_impl(
        pyre_object::gc_roots::shadow_stack_get(cls_slot),
        "__new__",
        true,
        false,
    )
    .map_err(|mut err| {
        err.enrich_attribute_error_str(
            pyre_object::gc_roots::shadow_stack_get(cls_slot),
            "__new__",
        );
        err
    })?;
    let _ = pyre_object::gc_roots::pin_root(w_new);
    let new_slot = pyre_object::gc_roots::shadow_stack_len() - 1;

    // interp_pickle.py `load_newobj`: an empty star-argument
    // container is not iterated.  This is observable for a custom object whose
    // `__len__` returns zero while `__iter__` has effects or yields values.
    if kwargs_slot.is_none()
        && pyre_interpreter::baseobjspace::len_w(pyre_object::gc_roots::shadow_stack_get(
            star_slot,
        ))? == 0
    {
        let w_new = pyre_object::gc_roots::shadow_stack_get(new_slot);
        let w_cls = pyre_object::gc_roots::shadow_stack_get(cls_slot);
        drop(_roots);
        return call_fn(w_new, &[w_cls]);
    }

    let mut unpacked = Vec::new();
    pyre_interpreter::argument::combine_starargs_wrapped(
        &mut unpacked,
        pyre_object::gc_roots::shadow_stack_get(star_slot),
        pyre_object::gc_roots::shadow_stack_get(new_slot),
    )?;
    let positional_base = pyre_object::gc_roots::shadow_stack_len();
    for value in unpacked {
        let _ = pyre_object::gc_roots::pin_root(value);
    }

    let mut keyword_names = Vec::new();
    let mut keyword_values = Vec::new();
    if let Some(kwargs_slot) = kwargs_slot {
        pyre_interpreter::argument::combine_starstarargs_wrapped(
            &mut keyword_names,
            &mut keyword_values,
            pyre_object::gc_roots::shadow_stack_get(kwargs_slot),
            pyre_object::gc_roots::shadow_stack_get(new_slot),
        )?;
    }
    let keyword_name_base = pyre_object::gc_roots::shadow_stack_len();
    for name in keyword_names {
        let _ = pyre_object::gc_roots::pin_root(name);
    }
    let keyword_value_base = pyre_object::gc_roots::shadow_stack_len();
    for value in keyword_values {
        let _ = pyre_object::gc_roots::pin_root(value);
    }

    let mut call_args = Vec::with_capacity(1 + keyword_name_base.saturating_sub(positional_base));
    call_args.push(pyre_object::gc_roots::shadow_stack_get(cls_slot));
    call_args
        .extend((positional_base..keyword_name_base).map(pyre_object::gc_roots::shadow_stack_get));
    let w_new = pyre_object::gc_roots::shadow_stack_get(new_slot);
    if keyword_name_base == keyword_value_base {
        drop(_roots);
        return call_fn(w_new, &call_args);
    }
    let keyword_count = keyword_value_base - keyword_name_base;
    let kwargs: Vec<_> = (0..keyword_count)
        .map(|index| {
            let name = pyre_object::gc_roots::shadow_stack_get(keyword_name_base + index);
            let value = pyre_object::gc_roots::shadow_stack_get(keyword_value_base + index);
            (
                unsafe { pyre_object::unicodeobject::w_str_get_wtf8(name) }.to_owned(),
                value,
            )
        })
        .collect();
    drop(_roots);
    let ec = pyre_interpreter::call::getexecutioncontext();
    if ec.is_null() {
        return Err(unpickling_error("no execution context for NEWOBJ_EX"));
    }
    let frame = unsafe { (*ec).gettopframe() };
    if frame.is_null() {
        return Err(unpickling_error("no frame for NEWOBJ_EX with kwargs"));
    }
    pyre_interpreter::call::call_with_kwargs(unsafe { &mut *frame }, w_new, &call_args, &kwargs)
}

/// `load_build` — apply pickled state to a freshly created instance.
fn build_instance(w_inst: PyObjectRef, w_state: PyObjectRef) -> Result<(), PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let _ = pyre_object::gc_roots::pin_root(w_inst);
    let inst_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let _ = pyre_object::gc_roots::pin_root(w_state);
    let state_slot = pyre_object::gc_roots::shadow_stack_len() - 1;

    // __setstate__ takes precedence.
    if let Some(setstate) = pyre_interpreter::baseobjspace::findattr_result(
        pyre_object::gc_roots::shadow_stack_get(inst_slot),
        "__setstate__",
    )? {
        // `_pickle.c:load_build` only distinguishes present from absent, so an
        // attribute explicitly set to `None` is still called — and reports the
        // ordinary "'NoneType' object is not callable".  Only a missing hook
        // falls through to state restoration.
        let _ = pyre_object::gc_roots::pin_root(setstate);
        let setstate_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        call_fn(
            pyre_object::gc_roots::shadow_stack_get(setstate_slot),
            &[pyre_object::gc_roots::shadow_stack_get(state_slot)],
        )?;
        return Ok(());
    }

    // state may be a (dict-state, slot-state) pair.
    let current_state = pyre_object::gc_roots::shadow_stack_get(state_slot);
    let (w_dict_state, w_slot_state) = if unsafe {
        pyre_object::is_tuple(current_state)
            && pyre_object::tupleobject::w_tuple_len(current_state) == 2
    } {
        (
            unsafe { pyre_object::tupleobject::w_tuple_getitem(current_state, 0).unwrap() },
            unsafe { pyre_object::tupleobject::w_tuple_getitem(current_state, 1).unwrap() },
        )
    } else {
        (current_state, pyre_object::w_none())
    };
    let _ = pyre_object::gc_roots::pin_root(w_dict_state);
    let dict_state_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let _ = pyre_object::gc_roots::pin_root(w_slot_state);
    let slot_state_slot = pyre_object::gc_roots::shadow_stack_len() - 1;

    if !unsafe { pyre_object::is_none(pyre_object::gc_roots::shadow_stack_get(dict_state_slot)) } {
        // `_pickle.c:load_build` requires a real dict and writes through the
        // instance `__dict__` mapping.
        if !unsafe {
            pyre_object::is_dict(pyre_object::gc_roots::shadow_stack_get(dict_state_slot))
        } {
            return Err(unpickling_error("state is not a dictionary"));
        }
        let w_inst_dict = pyre_interpreter::baseobjspace::getattr_str(
            pyre_object::gc_roots::shadow_stack_get(inst_slot),
            "__dict__",
        )?;
        let _ = pyre_object::gc_roots::pin_root(w_inst_dict);
        let inst_dict_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let items = unsafe {
            pyre_object::dictmultiobject::w_dict_items(pyre_object::gc_roots::shadow_stack_get(
                dict_state_slot,
            ))
        };
        for (key, value) in items {
            let _item_roots = pyre_object::gc_roots::push_roots();
            let _ = pyre_object::gc_roots::pin_root(key);
            let key_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let _ = pyre_object::gc_roots::pin_root(value);
            let value_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            // CPython 3.14 Modules/_pickle.c:6837-6843: exact `str`
            // instance-attribute keys pass through the interpreter intern
            // table before PyObject_SetItem.
            let key = pyre_object::gc_roots::shadow_stack_get(key_slot);
            let key = if unsafe { pyre_object::is_exact_type(key, &pyre_object::STR_TYPE) } {
                unsafe { pyre_object::unicodeobject::intern_exact_str(key) }
            } else {
                key
            };
            pyre_interpreter::baseobjspace::setitem(
                pyre_object::gc_roots::shadow_stack_get(inst_dict_slot),
                key,
                pyre_object::gc_roots::shadow_stack_get(value_slot),
            )?;
        }
    }
    if !unsafe { pyre_object::is_none(pyre_object::gc_roots::shadow_stack_get(slot_state_slot)) } {
        if !unsafe {
            pyre_object::is_dict(pyre_object::gc_roots::shadow_stack_get(slot_state_slot))
        } {
            return Err(unpickling_error("slot state is not a dictionary"));
        }
        let items = unsafe {
            pyre_object::dictmultiobject::w_dict_items(pyre_object::gc_roots::shadow_stack_get(
                slot_state_slot,
            ))
        };
        for (key, value) in items {
            let _item_roots = pyre_object::gc_roots::push_roots();
            let _ = pyre_object::gc_roots::pin_root(key);
            let key_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let _ = pyre_object::gc_roots::pin_root(value);
            let value_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            if !unsafe { pyre_object::is_str(pyre_object::gc_roots::shadow_stack_get(key_slot)) } {
                return Err(PyError::type_error(format!(
                    "attribute name must be string, not '{}'",
                    pyre_interpreter::type_methods::arg_type_name(
                        pyre_object::gc_roots::shadow_stack_get(key_slot)
                    ),
                )));
            }
            pyre_interpreter::baseobjspace::setattr(
                pyre_object::gc_roots::shadow_stack_get(inst_slot),
                pyre_object::gc_roots::shadow_stack_get(key_slot),
                pyre_object::gc_roots::shadow_stack_get(value_slot),
            )?;
        }
    }
    Ok(())
}
