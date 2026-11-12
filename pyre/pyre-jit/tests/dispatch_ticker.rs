//! `pyopcode.py` `dispatch_bytecode` services a fired action ticker once,
//! through `decrement_ticker`, before the opcode runs. A ticker left at
//! `-1` (`fire_action_ticker`) must still reach `action_dispatcher` from
//! `eval_loop_jit` and come back non-negative.

use pyre_interpreter::executioncontext::ActionFlagOps;
use pyre_interpreter::pyframe::PyFrame;
use pyre_interpreter::{Mode, alloc_execution_context, compile_source_with_filename};
use pyre_jit::eval::{eval_with_jit, init_jit_hooks};

#[test]
fn fired_ticker_is_serviced_once_before_the_opcode() {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(|| {
            let _lock =
                pyre_interpreter::module::signal::signalstate::test_support::signal_eval_breaker_test_guard();
            pyre_module::register();
            init_jit_hooks();
            let ec = alloc_execution_context();
            let ec_ptr = ec.as_mut_ptr();
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
            unsafe { (*ec_ptr).actionflag.reset_ticker(0) };
        })
        .unwrap()
        .join()
        .unwrap();
}
