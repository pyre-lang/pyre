//! Regression test: pyre's `JitState` impl must declare multi-frame restore
//! support so the metainterp dispatch arms / helper guard-fail path that
//! recover a chain of caller frames are reached at run time
//! (`blackhole.py` parity).
//!
//! Removing the override on `PyreJitState` would silently fall back to the
//! trait default (`false`) and break multi-frame resume without any test
//! signaling the regression — this test pins the property.

use majit_metainterp::JitState;
use pyre_jit_trace::state::PyreJitState;

#[test]
fn pyre_jit_state_supports_multi_frame_restore() {
    let state = PyreJitState::new(0, 0);
    assert!(
        state.supports_multi_frame_restore(),
        "PyreJitState must override JitState::supports_multi_frame_restore \
         to return true (blackhole.py:1800 parity); regression here would \
         silently break dispatch arm / helper guard-fail multi-frame resume",
    );
}
