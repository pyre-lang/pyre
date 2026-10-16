//! `gc_sync::store_singleton` is process-global and never cleared, so this
//! file is its own test binary.

use majit_ir::OpRef;
use majit_metainterp::{JitDriver, JitState};

struct Idle;

impl JitState for Idle {
    type Meta = ();
    type Sym = ();
    type Env = ();

    fn build_meta(&self, _header_pc: usize, _env: &Self::Env) -> Self::Meta {}

    fn extract_live(&mut self, _meta: &Self::Meta) -> Vec<i64> {
        Vec::new()
    }

    fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {}

    fn is_compatible(&self, _meta: &Self::Meta) -> bool {
        true
    }

    fn restore(&mut self, _meta: &Self::Meta, _values: &[i64]) {}

    fn collect_jump_args(_sym: &Self::Sym) -> Vec<OpRef> {
        Vec::new()
    }

    fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        true
    }
}

#[test]
fn install_jitframe_gc_does_not_shadow_process_collector() {
    assert!(
        !majit_gc::gc_sync::is_initialized(),
        "this process must start without a collector"
    );
    assert!(!majit_gc::gc_box_installed());

    // `examples/regex` `gc::install` stores the singleton before `Matcher::new`.
    majit_gc::gc_sync::store_singleton(Box::new(majit_gc::collector::MiniMarkGC::new()));
    assert!(majit_gc::gc_sync::is_initialized());

    let mut driver = JitDriver::<Idle>::new(1);
    majit_metainterp::install_jitframe_gc(&mut driver);
    majit_metainterp::install_jitframe_gc(&mut driver);

    assert!(
        !majit_gc::gc_box_installed(),
        "install_jitframe_gc installed a per-thread box over the process collector"
    );
}
