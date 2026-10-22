//! `pyopcode.py` `dispatch_bytecode` services a fired action ticker once,
//! through `decrement_ticker`, before the opcode runs. A ticker left at
//! `-1` (`fire_action_ticker`) must still reach `action_dispatcher` from
//! `eval_loop_jit` and come back non-negative.

use std::rc::Rc;

use pyre_interpreter::executioncontext::ActionFlagOps;
use pyre_interpreter::pyframe::PyFrame;
use pyre_interpreter::{Mode, PyExecutionContext, compile_source_with_filename};
use pyre_jit::eval::{eval_with_jit, init_jit_hooks};

#[test]
fn fired_ticker_is_serviced_once_before_the_opcode() {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(|| {
            pyre_module::register();
            init_jit_hooks();
            let ec = Rc::new(PyExecutionContext::default());
            let ec_ptr = Rc::as_ptr(&ec) as *mut PyExecutionContext;
            pyre_interpreter::call::set_last_exec_ctx(ec_ptr);
            unsafe { (*ec_ptr).actionflag.reset_ticker(-1) };
            let code = compile_source_with_filename("pass\n", Mode::Exec, "ticker.py").unwrap();
            let mut frame = PyFrame::new_with_context(code, ec.clone()).unwrap();
            eval_with_jit(&mut frame, None).unwrap();
            let ticker = unsafe { (*ec_ptr).actionflag.get_ticker() };
            assert!(
                ticker >= 0,
                "action_dispatcher resets a fired ticker, left {ticker}"
            );
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn simple_counted_loop_runs_through_the_portal() {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(|| {
            pyre_module::register();
            init_jit_hooks();
            let ec = Rc::new(PyExecutionContext::default());
            let ec_ptr = Rc::as_ptr(&ec) as *mut PyExecutionContext;
            pyre_interpreter::call::set_last_exec_ctx(ec_ptr);
            let code = compile_source_with_filename(
                "i = 0\nwhile i < 2000:\n    i += 1\nassert i == 2000\n",
                Mode::Exec,
                "t1_loop.py",
            )
            .unwrap();
            let mut frame = PyFrame::new_with_context(code, ec.clone()).unwrap();
            eval_with_jit(&mut frame, None).unwrap();
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn for_range_loop_runs_through_the_portal() {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(|| {
            pyre_module::register();
            init_jit_hooks();
            let ec = Rc::new(PyExecutionContext::default());
            let ec_ptr = Rc::as_ptr(&ec) as *mut PyExecutionContext;
            pyre_interpreter::call::set_last_exec_ctx(ec_ptr);
            let code = compile_source_with_filename(
                "s = 0\nfor i in range(2000):\n    s += i\nassert s == 1999000\n",
                Mode::Exec,
                "t1_for.py",
            )
            .unwrap();
            let mut frame = PyFrame::new_with_context(code, ec.clone()).unwrap();
            eval_with_jit(&mut frame, None).unwrap();
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn list_subscript_loop_runs_through_the_portal() {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(|| {
            pyre_module::register();
            init_jit_hooks();
            let ec = Rc::new(PyExecutionContext::default());
            let ec_ptr = Rc::as_ptr(&ec) as *mut PyExecutionContext;
            pyre_interpreter::call::set_last_exec_ctx(ec_ptr);
            let code = compile_source_with_filename(
                concat!(
                    "a = [0] * 1423\n",
                    "for i in range(1423):\n",
                    "    assert a[i] == 0\n",
                ),
                Mode::Exec,
                "t1_subscr.py",
            )
            .unwrap();
            let mut frame = PyFrame::new_with_context(code, ec.clone()).unwrap();
            eval_with_jit(&mut frame, None).unwrap();
        })
        .unwrap()
        .join()
        .unwrap();
}
