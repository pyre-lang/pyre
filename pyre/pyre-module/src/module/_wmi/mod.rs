//! `_wmi` — CPython's private Windows Management Instrumentation query door.
//!
//! PyPy has no `_wmi` module.  This explicitly requested scope extension is a
//! structural port of CPython 3.14's `PC/_wmimodule.cpp` (`_query_thread`,
//! `wait_event`, and `_wmi_exec_query_impl`): each call owns one pipe, two
//! staged events and one apartment-threaded COM worker.  In particular, the
//! worker is not replaced with PowerShell or a process-global COM connection;
//! those would change the timeout, isolation and concurrent-call semantics.

use pyre_object::{PY_NULL, PyObjectRef};

fn execute_query(query: Vec<u16>) -> Result<Vec<u16>, pyre_interpreter::PyError> {
    use rustpython_host_env::wmi::{BUFFER_SIZE, ExecQueryError, exec_query};
    let query = String::from_utf16(&query)
        .map_err(|_| pyre_interpreter::PyError::value_error("query is not valid UTF-16"))?;
    let result = {
        let _blocked = pyre_interpreter::module::thread::before_external_block();
        exec_query(&query)
    };
    result
        .map(|text| text.encode_utf16().collect())
        .map_err(|error| match error {
            ExecQueryError::MoreData => pyre_interpreter::PyError::os_error(format!(
                "Query returns more than {BUFFER_SIZE} characters"
            )),
            ExecQueryError::Code(code) => {
                pyre_interpreter::PyError::os_error_win32_syscall2(code as i32, PY_NULL, PY_NULL)
            }
        })
}

fn is_select_query(query: &[u16]) -> bool {
    const SELECT: &[u8; 7] = b"select ";
    query.len() >= SELECT.len()
        && query[..SELECT.len()]
            .iter()
            .zip(SELECT)
            .all(|(&unit, &expected)| {
                unit == expected as u16
                    || (expected.is_ascii_lowercase()
                        && unit == expected.to_ascii_uppercase() as u16)
            })
}

/// `_wmi.exec_query($module, /, query)`.
fn exec_query(args: &[PyObjectRef]) -> pyre_interpreter::PyResult {
    let (positional, kwargs) = pyre_interpreter::builtins::split_builtin_kwargs(args);
    pyre_interpreter::builtins::clinic_arity(
        "exec_query",
        positional.len(),
        pyre_interpreter::builtins::real_kwarg_count(kwargs),
        1,
        1,
        0,
    )?;
    let query = positional
        .first()
        .copied()
        .or_else(|| pyre_interpreter::builtins::kwarg_get(kwargs, "query"))
        .ok_or_else(|| {
            pyre_interpreter::PyError::type_error(
                "exec_query() missing required argument 'query' (pos 1)",
            )
        })?;
    if !unsafe { pyre_object::is_str(query) } {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "exec_query() argument 'query' must be str, not {}",
            pyre_interpreter::gateway::short_type_name(query)
        )));
    }

    // The clinic unicode converter has accepted the object, but the wide-char
    // copy and SELECT validation live after the audit in CPython's impl.
    let roots = pyre_object::gc_roots::push_roots();
    let query_slot = roots.base();
    let _ = roots.pin_root(query);
    pyre_interpreter::module::sys::vm::audit("_wmi.exec_query", &[roots.get(query_slot)])?;
    let query = roots.get(query_slot);
    let mut wide: Vec<u16> = unsafe { pyre_object::w_str_get_wtf8(query) }
        .encode_wide()
        .collect();
    if wide.contains(&0) {
        return Err(pyre_interpreter::PyError::value_error(
            "embedded null character",
        ));
    }
    if !is_select_query(&wide) {
        return Err(pyre_interpreter::PyError::value_error(
            "only SELECT queries are supported",
        ));
    }

    let result = execute_query(std::mem::take(&mut wide))?;
    Ok(pyre_object::w_str_from_wtf8_managed(
        rustpython_wtf8::Wtf8Buf::from_wide(&result),
    ))
}

const EXEC_QUERY_DOC: &str = "Runs a WMI query against the local machine.\n\nThis returns a single string with 'name=value' pairs in a flat array separated\nby null characters.";

pyre_interpreter::py_module! {
    "_wmi",
    extra_init: |ns| {
        pyre_interpreter::__pyre_store!(ns, "exec_query", pyre_interpreter::gateway::with_module(
                "_wmi",
                pyre_interpreter::make_module_builtin_function_with_doc(
                    "exec_query",
                    exec_query,
                    EXEC_QUERY_DOC,
                ),
            ));
    },
}
