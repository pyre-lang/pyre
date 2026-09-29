//! A raise that crosses two frames records each frame's name on the
//! traceback when the JIT is off.
//!
//! `PYRE_NO_JIT=1` still dispatches through `eval_loop_jit`. That loop must
//! record the application traceback before it searches the exception table,
//! the same order as `pyopcode.py handle_operation_error`.

use std::rc::Rc;

use pyre_interpreter::call::{register_build_class, set_last_exec_ctx};
use pyre_interpreter::importing;
use pyre_interpreter::pyframe::PyFrame;
use pyre_interpreter::{Mode, PyExecutionContext, compile_source_with_filename};
use pyre_jit::eval::{eval_with_jit, init_jit_hooks, reset_gc_fresh_for_test};

static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn run_harness(program: &str, name: &str) -> Result<(), String> {
    // The off switch is read once, on the first portal entry.
    unsafe { std::env::set_var("PYRE_NO_JIT", "1") };
    pyre_interpreter::stack_check::set_recursion_limit(5000)
        .map_err(|_| "set_recursion_limit failed".to_string())?;
    pyre_module::register();
    init_jit_hooks();
    reset_gc_fresh_for_test();

    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    importing::init_sys_path(&cwd, cwd.as_os_str());
    importing::add_sys_path_0();
    importing::set_sys_argv(&[std::ffi::OsString::from(name)]);

    let code = compile_source_with_filename(program, Mode::Exec, name)
        .map_err(|e| format!("compile error: {e}"))?;

    register_build_class();

    let execution_context = Rc::new(PyExecutionContext::default());
    set_last_exec_ctx(Rc::as_ptr(&execution_context));

    let mut frame = PyFrame::new_with_context(code, execution_context)
        .map_err(|e| format!("frame setup error: {}", e.message_text()))?;

    let canonical = frame.get_w_globals();
    let main_module = pyre_object::w_module_new_aliasing_dict("__main__", canonical);
    importing::set_sys_module("__main__", main_module);

    eval_with_jit(&mut frame, None)
        .map_err(|e| format!("execution error: {}", e.message_text()))?;
    Ok(())
}

fn run_on_worker(program: &'static str, name: &'static str) {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let handle = std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(move || run_harness(program, name))
        .expect("spawn worker thread");
    handle.join().expect("worker thread panicked").expect(name);
}

/// `leaf` raises, `mid` forwards, `outer` catches. The traceback head is the
/// catching frame and `tb_next` walks toward the raise.
#[test]
fn raise_through_two_frames_records_names_with_jit_disabled() {
    run_on_worker(
        r#"
def names(exc):
    out = []
    tb = exc.__traceback__
    while tb is not None:
        out.append(tb.tb_frame.f_code.co_name)
        tb = tb.tb_next
    return tuple(out)

def leaf():
    raise ValueError("boom")

def mid():
    leaf()

def outer():
    try:
        mid()
    except ValueError as e:
        got = names(e)
        assert got == ("outer", "mid", "leaf"), got

outer()
"#,
        "traceback_frame_names.py",
    );
}
