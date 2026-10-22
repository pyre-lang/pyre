//! `pypy/module/gc/app_referents.py`: the app-level `GcStats` class,
//! `get_stats` and `dump_rpy_heap`, written at interp-level.

use pyre_object::*;

use super::referents::{dump_rpy_heap_fd, stats};

fn format_gc_stat(value: i64) -> String {
    if value < 1_000_000 {
        format!("{:.1}kB", value as f64 / 1024.0)
    } else {
        format!("{:.1}MB", value as f64 / 1024.0 / 1024.0)
    }
}

fn gc_stats_public_type() -> PyObjectRef {
    static TYPE: pyre_object::gc_roots::RootedOnceRef = pyre_object::gc_roots::RootedOnceRef::new();
    TYPE.get_or_init(|| {
        // PyPy `app_referents.GcStats` is an ordinary app-level class, not a
        // second interpreter TypeDef beside `referents.W_GcStats`.  Build it
        // through `type.__new__`, so it inherits object's allocator and owns
        // the normal mapdict layout an app-level class receives.
        let roots = pyre_object::gc_roots::push_roots();
        let ns_slot = roots.base();
        let _ = roots.pin_root(pyre_object::w_dict_new());
        let store = |name: &str, value: PyObjectRef| unsafe {
            pyre_object::w_dict_setitem_str_no_proxy(roots.get(ns_slot), name, value);
        };
        store("__module__", w_str_new("gc"));
        store(
            "__init__",
            pyre_interpreter::make_builtin_function_with_arity("__init__", gc_stats_public_init, 2),
        );
        store(
            "__repr__",
            pyre_interpreter::make_builtin_function_with_arity("__repr__", gc_stats_repr, 1),
        );
        let bases_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(pyre_object::w_tuple_new(vec![
            pyre_interpreter::typedef::w_object(),
        ]));
        let args = [
            pyre_interpreter::typedef::w_type(),
            w_str_new("GcStats"),
            pyre_object::gc_roots::shadow_stack_get(bases_slot),
            roots.get(ns_slot),
        ];
        pyre_interpreter::builtins::type_descr_new(&args).expect("construct app_referents.GcStats")
    })
}

fn gc_stats_public_init(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    populate_public_gc_stats(args[0], args[1])?;
    Ok(w_none())
}

fn gc_stats_attr_string(
    self_slot: usize,
    name: &'static str,
) -> Result<String, pyre_interpreter::PyError> {
    let value = pyre_interpreter::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(self_slot),
        name,
    )?;
    Ok(unsafe { pyre_interpreter::display::py_str_wtf8(value)? }
        .to_string_lossy()
        .into_owned())
}

fn gc_stats_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let self_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(args[0]);
    let raw = pyre_interpreter::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(self_slot),
        "_s",
    )?;
    let _ = pyre_object::gc_roots::pin_root(raw);
    let raw_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let raw = stats::W_GcStats::from_obj(pyre_object::gc_roots::shadow_stack_get(raw_slot))
        .ok_or_else(|| {
            pyre_interpreter::PyError::type_error("GcStats._s is not a native GcStats")
        })?;
    let total_memory_pressure = raw.total_memory_pressure;
    let total_arena_allocated = format_gc_stat(
        raw.total_allocated_memory - raw.total_rawmalloced_memory - raw.nursery_size,
    );

    // app_referents.py:76-120. Read the public attributes afresh: this class
    // has a normal instance dict upstream, and user mutation affects repr.
    let total_gc_memory = gc_stats_attr_string(self_slot, "total_gc_memory")?;
    let peak_memory = gc_stats_attr_string(self_slot, "peak_memory")?;
    let total_arena_memory = gc_stats_attr_string(self_slot, "total_arena_memory")?;
    let peak_arena_memory = gc_stats_attr_string(self_slot, "peak_arena_memory")?;
    let total_rawmalloced_memory = gc_stats_attr_string(self_slot, "total_rawmalloced_memory")?;
    let peak_rawmalloced_memory = gc_stats_attr_string(self_slot, "peak_rawmalloced_memory")?;
    let nursery_size = gc_stats_attr_string(self_slot, "nursery_size")?;
    let jit_backend_used = gc_stats_attr_string(self_slot, "jit_backend_used")?;
    let total_memory_pressure_text = gc_stats_attr_string(self_slot, "total_memory_pressure")?;
    let memory_used_sum = gc_stats_attr_string(self_slot, "memory_used_sum")?;
    let total_allocated_memory = gc_stats_attr_string(self_slot, "total_allocated_memory")?;
    let peak_allocated_memory = gc_stats_attr_string(self_slot, "peak_allocated_memory")?;
    let jit_backend_allocated = gc_stats_attr_string(self_slot, "jit_backend_allocated")?;
    let memory_allocated_sum = gc_stats_attr_string(self_slot, "memory_allocated_sum")?;
    let total_gc_time_obj = pyre_interpreter::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(self_slot),
        "total_gc_time",
    )?;
    let _ = pyre_object::gc_roots::pin_root(total_gc_time_obj);
    let total_gc_time_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let total_gc_time = pyre_interpreter::baseobjspace::float_w(
        pyre_object::gc_roots::shadow_stack_get(total_gc_time_slot),
    )? / 1000.0;
    let extra = if total_memory_pressure != -1 {
        format!("\n    memory pressure:         {total_memory_pressure_text}")
    } else {
        String::new()
    };
    Ok(w_str_new_managed(&format!(
        concat!(
            "Total memory consumed:\n",
            "    GC used:                 {total_gc_memory} (peak: {peak_memory})\n",
            "       in arenas:            {total_arena_memory} (peak: {peak_arena_memory})\n",
            "       rawmalloced:          {total_rawmalloced_memory} (peak: {peak_rawmalloced_memory})\n",
            "       nursery:              {nursery_size}\n",
            "    raw assembler used:      {jit_backend_used}{extra}\n",
            "    -----------------------------\n",
            "    Total:                   {memory_used_sum}\n\n",
            "    Total memory allocated (includes freelists):\n",
            "    GC allocated:            {total_allocated_memory} (peak: {peak_allocated_memory})\n",
            "       in arenas:            {total_arena_allocated}\n",
            "       rawmalloced:          {total_rawmalloced_memory}\n",
            "       nursery:              {nursery_size}\n",
            "    raw assembler allocated: {jit_backend_allocated}{extra}\n",
            "    -----------------------------\n",
            "    Total:                   {memory_allocated_sum}\n\n",
            "    Total time spent in GC:  {total_gc_time}\n    "
        ),
        total_gc_memory = total_gc_memory,
        peak_memory = peak_memory,
        total_arena_memory = total_arena_memory,
        peak_arena_memory = peak_arena_memory,
        total_rawmalloced_memory = total_rawmalloced_memory,
        peak_rawmalloced_memory = peak_rawmalloced_memory,
        nursery_size = nursery_size,
        jit_backend_used = jit_backend_used,
        extra = extra,
        memory_used_sum = memory_used_sum,
        total_allocated_memory = total_allocated_memory,
        peak_allocated_memory = peak_allocated_memory,
        total_arena_allocated = total_arena_allocated,
        jit_backend_allocated = jit_backend_allocated,
        memory_allocated_sum = memory_allocated_sum,
        total_gc_time = total_gc_time,
    )))
}

fn populate_public_gc_stats(
    obj: PyObjectRef,
    raw: PyObjectRef,
) -> Result<(), pyre_interpreter::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(obj);
    let raw_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(raw);
    let raw = stats::W_GcStats::from_obj(pyre_object::gc_roots::shadow_stack_get(raw_slot))
        .ok_or_else(|| {
            pyre_interpreter::PyError::type_error("GcStats() requires a native GcStats")
        })?;
    let memory_pressure_value = if raw.total_memory_pressure == -1 {
        0
    } else {
        raw.total_memory_pressure
    };
    let formatted = [
        ("total_gc_memory", raw.total_gc_memory),
        ("jit_backend_used", raw.jit_backend_used),
        ("total_memory_pressure", raw.total_memory_pressure),
        ("total_allocated_memory", raw.total_allocated_memory),
        ("jit_backend_allocated", raw.jit_backend_allocated),
        ("peak_memory", raw.peak_memory),
        ("peak_allocated_memory", raw.peak_allocated_memory),
        ("total_arena_memory", raw.total_arena_memory),
        ("total_rawmalloced_memory", raw.total_rawmalloced_memory),
        ("nursery_size", raw.nursery_size),
        ("peak_arena_memory", raw.peak_arena_memory),
        ("peak_rawmalloced_memory", raw.peak_rawmalloced_memory),
        (
            "memory_used_sum",
            raw.total_gc_memory + memory_pressure_value + raw.jit_backend_used,
        ),
        (
            "memory_allocated_sum",
            raw.total_allocated_memory + memory_pressure_value + raw.jit_backend_allocated,
        ),
    ];
    let total_gc_time = raw.total_gc_time;
    pyre_interpreter::baseobjspace::setattr_str(
        pyre_object::gc_roots::shadow_stack_get(obj_slot),
        "_s",
        pyre_object::gc_roots::shadow_stack_get(raw_slot),
    )?;
    for (name, value) in formatted {
        let text = w_str_new_managed(&format_gc_stat(value));
        let _ = pyre_object::gc_roots::pin_root(text);
        let text_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        pyre_interpreter::baseobjspace::setattr_str(
            pyre_object::gc_roots::shadow_stack_get(obj_slot),
            name,
            pyre_object::gc_roots::shadow_stack_get(text_slot),
        )?;
    }
    // Build and pin the value before reading `obj_slot`: Rust evaluates the
    // receiver first, so an inline `w_int_new` here would allocate — and
    // possibly collect — after the argument already held a raw address.
    let time_value = w_int_new(total_gc_time);
    let _ = pyre_object::gc_roots::pin_root(time_value);
    let time_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    pyre_interpreter::baseobjspace::setattr_str(
        pyre_object::gc_roots::shadow_stack_get(obj_slot),
        "total_gc_time",
        pyre_object::gc_roots::shadow_stack_get(time_slot),
    )?;
    Ok(())
}

pub(super) fn new_public_gc_stats(
    memory_pressure: bool,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let raw = stats::new(memory_pressure);
    let raw_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(raw);
    let public_type = gc_stats_public_type();
    let obj = w_instance_new(public_type);
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(obj);
    populate_public_gc_stats(
        pyre_object::gc_roots::shadow_stack_get(obj_slot),
        pyre_object::gc_roots::shadow_stack_get(raw_slot),
    )?;
    Ok(pyre_object::gc_roots::shadow_stack_get(obj_slot))
}

fn gc_call_method(
    obj: PyObjectRef,
    name: &str,
    args: &[PyObjectRef],
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let result = pyre_interpreter::baseobjspace::call_method(obj, name, args);
    if result.is_null() {
        Err(pyre_interpreter::call::take_call_error()
            .unwrap_or_else(|| pyre_interpreter::PyError::runtime_error("method call failed")))
    } else {
        Ok(result)
    }
}

/// The spelling each typeid sidecar wants. `app_referents.py:34` opens
/// `typeids.txt` binary and writes the decompressed bytes; `:40` opens
/// `typeids.lst` text and writes a str, so the object handed to `write`
/// differs even though the surrounding steps do not.
enum TypeidsPayload<'a> {
    Binary(&'a [u8]),
    Text(&'a str),
}

/// `app_referents.py:32,38` `os.path.exists`. Under sandbox the probe is a
/// controller round trip, the way `importing.rs`'s `SeamSourceProvider` does
/// it; `Path::exists` stats the real filesystem, which is exactly what the
/// jail is there to prevent. It escapes the `disallowed-methods` fence only
/// because that list names `std::fs::metadata` rather than the `Path` method
/// wrapping it.
fn typeids_sidecar_exists(path: &std::path::Path) -> bool {
    #[cfg(feature = "sandbox")]
    {
        use std::os::unix::ffi::OsStrExt;
        majit_rlib::rposix_stat::stat(path.as_os_str().as_bytes()).is_ok()
    }
    #[cfg(not(feature = "sandbox"))]
    {
        path.exists()
    }
}

/// `app_referents.py:33-36,39-42`: open the sidecar, write it once, close it.
///
/// The write goes through `builtin_open` and the file object's own `write` /
/// `close`, which is both what upstream writes and what a sandbox build routes
/// to the controller — `std::fs::write` would reach the real filesystem from
/// inside the jail. Errors are left to propagate from those calls, as they do
/// upstream, so the raised `OSError` carries the sidecar's own name rather
/// than the dump's.
fn write_typeids_sidecar(
    path: &std::path::Path,
    payload: TypeidsPayload<'_>,
) -> Result<(), pyre_interpreter::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let name_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(pyre_interpreter::gateway::fsdecode_os_str(
        path.as_os_str(),
    ));
    let mode_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_str_new_managed(match payload {
        TypeidsPayload::Binary(_) => "wb",
        TypeidsPayload::Text(_) => "w",
    }));
    let opened = pyre_interpreter::builtins::builtin_open(&[
        pyre_object::gc_roots::shadow_stack_get(name_slot),
        pyre_object::gc_roots::shadow_stack_get(mode_slot),
    ])?;
    let opened_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(opened);
    // Materialize the payload only now: `builtin_open` allocates, so a value
    // boxed before it would have to be rooted across the open for nothing.
    let data_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(match payload {
        TypeidsPayload::Binary(bytes) => w_bytes_from_bytes(bytes),
        TypeidsPayload::Text(text) => w_str_new_managed(text),
    });
    match gc_call_method(
        pyre_object::gc_roots::shadow_stack_get(opened_slot),
        "write",
        &[pyre_object::gc_roots::shadow_stack_get(data_slot)],
    ) {
        Ok(value) => {
            let _value = pyre_object::gc_roots::pin_root(value);
            gc_call_method(
                pyre_object::gc_roots::shadow_stack_get(opened_slot),
                "close",
                &[],
            )?;
        }
        Err(error) => {
            let mut error = error;
            let error_slot = error.pin(&_roots);
            let _ = gc_call_method(
                pyre_object::gc_roots::shadow_stack_get(opened_slot),
                "close",
                &[],
            );
            error.reload(&_roots, error_slot);
            return Err(error);
        }
    }
    Ok(())
}

pub(super) fn dump_rpy_heap_public(
    file: PyObjectRef,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let file_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(file);
    // Read the slot at every use instead of caching it in a local: the
    // collector forwards the shadow-stack entry, not the Rust binding, and
    // `w_str_new`, `builtin_open`, `getattr_str`, and `gc_call_method` below
    // all allocate between the uses.
    let file = || pyre_object::gc_roots::shadow_stack_get(file_slot);

    if unsafe { is_str(file()) } {
        // app_referents.py:22-40: filename arm opens/truncates the binary
        // dump, closes it, then materializes typeids.txt/.lst if absent.
        let path = pyre_interpreter::gateway::fspath_buf(file())?;
        let mode_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(w_str_new_managed("wb"));
        let opened = pyre_interpreter::builtins::builtin_open(&[
            file(),
            pyre_object::gc_roots::shadow_stack_get(mode_slot),
        ])?;
        let _ = pyre_object::gc_roots::pin_root(opened);
        let opened_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let fileno = gc_call_method(
            pyre_object::gc_roots::shadow_stack_get(opened_slot),
            "fileno",
            &[],
        )?;
        let fd = pyre_interpreter::baseobjspace::int_w(fileno)? as i32;
        match dump_rpy_heap_fd(fd) {
            Ok(()) => {
                gc_call_method(
                    pyre_object::gc_roots::shadow_stack_get(opened_slot),
                    "close",
                    &[],
                )?;
            }
            Err(error) => {
                let mut error = error;
                let error_slot = error.pin(&_roots);
                let _ = gc_call_method(
                    pyre_object::gc_roots::shadow_stack_get(opened_slot),
                    "close",
                    &[],
                );
                error.reload(&_roots, error_slot);
                return Err(error);
            }
        }

        let directory = path.parent().unwrap_or_else(|| std::path::Path::new(""));
        let typeids_txt = directory.join("typeids.txt");
        if !typeids_sidecar_exists(&typeids_txt) {
            let text = majit_gc::get_typeids_text().ok_or_else(|| {
                pyre_interpreter::PyError::not_implemented("operation not implemented by this GC")
            })?;
            write_typeids_sidecar(&typeids_txt, TypeidsPayload::Binary(&text))?;
        }
        let typeids_lst = directory.join("typeids.lst");
        if !typeids_sidecar_exists(&typeids_lst) {
            let list = majit_gc::get_typeids_list().ok_or_else(|| {
                pyre_interpreter::PyError::not_implemented("operation not implemented by this GC")
            })?;
            let data: String = list.into_iter().map(|value| format!("{value}\n")).collect();
            write_typeids_sidecar(&typeids_lst, TypeidsPayload::Text(&data))?;
        }
        return Ok(w_none());
    }

    let fd = if unsafe { is_int(file()) } {
        pyre_interpreter::baseobjspace::int_w(file())? as i32
    } else {
        // app_referents.py:44-49: flush only when the attribute exists, then
        // ask for fileno. AttributeError is the only absence case upstream;
        // a present flush method's exception propagates.
        match pyre_interpreter::baseobjspace::getattr_str(file(), "flush") {
            Ok(flush) => {
                pyre_interpreter::call::call_function_impl_result(flush, &[])?;
            }
            Err(error) if error.kind == pyre_interpreter::PyErrorKind::AttributeError => {}
            Err(error) => return Err(error),
        }
        let fileno = gc_call_method(file(), "fileno", &[])?;
        pyre_interpreter::baseobjspace::int_w(fileno)? as i32
    };
    dump_rpy_heap_fd(fd)?;
    Ok(w_none())
}

/// `app_referents.py get_stats`.
#[pyre_interpreter::pyre_function]
pub(super) fn get_stats(
    #[default(false)] memory_pressure: bool,
) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    new_public_gc_stats(memory_pressure)
}

/// `app_referents.py dump_rpy_heap`.
#[pyre_interpreter::pyre_function]
pub(super) fn dump_rpy_heap(file: PyObjectRef) -> Result<PyObjectRef, pyre_interpreter::PyError> {
    dump_rpy_heap_public(file)
}
