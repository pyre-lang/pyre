//! Error-path contracts independent of any concrete backend or runtime.

use std::sync::Arc;

use majit_backend::eager::CompiledIr;
use majit_backend::{AsmInfo, Backend, BackendError, DeadFrame, ExitRecoveryLayout, JitCellToken};
use majit_ir::{Const, ConstMap, DescrRef, FailDescr, GcRef, InputArgRc, Op, OpCode, OpRc, Value};

#[derive(Default)]
struct RejectingBackend {
    constants: ConstMap<Const>,
    compilations: usize,
}

impl Backend for RejectingBackend {
    fn backend_name(&self) -> &'static str {
        "rejecting-test-backend"
    }
    fn set_constants_pool(&mut self, constants: ConstMap<Const>) {
        self.constants = constants;
    }
    fn setup_once(&mut self) {
        panic!("eager compilation must not redo runtime setup");
    }
    fn finish_once(&mut self) {
        panic!("eager compilation must not tear down the runtime");
    }
    fn set_done_with_this_frame_descr_void(&mut self, _: DescrRef) {
        panic!("eager compilation must preserve CPU descriptors");
    }
    fn set_done_with_this_frame_descr_int(&mut self, _: DescrRef) {
        panic!("eager compilation must preserve CPU descriptors");
    }
    fn set_done_with_this_frame_descr_ref(&mut self, _: DescrRef) {
        panic!("eager compilation must preserve CPU descriptors");
    }
    fn set_done_with_this_frame_descr_float(&mut self, _: DescrRef) {
        panic!("eager compilation must preserve CPU descriptors");
    }
    fn set_exit_frame_with_exception_descr_ref(&mut self, _: DescrRef) {
        panic!("eager compilation must preserve CPU descriptors");
    }
    fn set_propagate_exception_descr(&mut self, _: DescrRef) {
        panic!("eager compilation must preserve CPU descriptors");
    }
    fn compile_loop(
        &mut self,
        _: &[InputArgRc],
        _: &[OpRc],
        _: &JitCellToken,
    ) -> Result<AsmInfo, BackendError> {
        self.compilations += 1;
        assert_eq!(
            self.constants.len(),
            1,
            "submission's pool must reach the backend"
        );
        Err(BackendError::Unsupported(
            "deliberate compile decline".into(),
        ))
    }
    fn compile_bridge(
        &mut self,
        _: &dyn FailDescr,
        _: &[InputArgRc],
        _: &[OpRc],
        _: &JitCellToken,
        _: Option<&ExitRecoveryLayout>,
    ) -> Result<AsmInfo, BackendError> {
        unreachable!()
    }
    fn execute_token(&self, _: &JitCellToken, _: &[Value]) -> DeadFrame {
        panic!("compile failure must not run code or a fallback");
    }
    fn get_latest_descr<'a>(&'a self, _: &'a DeadFrame) -> &'a dyn FailDescr {
        unreachable!()
    }
    fn get_latest_descr_arc(&self, _: &DeadFrame) -> DescrRef {
        unreachable!()
    }
    fn get_int_value(&self, _: &DeadFrame, _: usize) -> i64 {
        unreachable!()
    }
    fn get_value_direct(&self, _: &DeadFrame, _: usize) -> i64 {
        unreachable!()
    }
    fn get_float_value(&self, _: &DeadFrame, _: usize) -> f64 {
        unreachable!()
    }
    fn get_ref_value(&self, _: &DeadFrame, _: usize) -> GcRef {
        unreachable!()
    }
    fn invalidate_loop(&self, _: &JitCellToken) {
        unreachable!()
    }
}

#[test]
fn backend_decline_propagates_and_pending_constants_are_cleared() {
    let mut backend = RejectingBackend::default();
    let finish = OpRc::new(Op::with_descr(
        OpCode::Finish,
        &[],
        majit_ir::descr::make_finish_descr(0, vec![]),
    ));
    let mut constants = ConstMap::default();
    constants.insert(0, Const::Int(42));
    let result = unsafe {
        CompiledIr::compile(
            &mut backend,
            Arc::new(JitCellToken::new(1)),
            &[],
            &[finish],
            constants,
        )
    };
    assert!(
        matches!(result, Err(BackendError::Unsupported(ref message)) if message == "deliberate compile decline")
    );
    drop(result);
    assert_eq!(backend.compilations, 1);
    assert!(backend.constants.is_empty());
}
