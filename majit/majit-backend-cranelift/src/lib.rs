mod asm_memory;

/// Cranelift-based JIT code generation backend for majit.
///
/// This crate implements the `majit_backend::Backend` trait using Cranelift
/// to translate majit IR traces into native machine code.
#[expect(
    clippy::too_many_arguments,
    reason = "Cranelift lowering helpers explicitly thread builder, frame, GC-root, descriptor, and ABI state in the same phase boundaries as the RPython backend; grouping them would hide ownership and parity relationships"
)]
pub mod compiler;
pub mod guard;
mod stub;

pub use compiler::{
    CallAssemblerDescr, CraneliftBackend, JitFrameLayoutInfo, clear_gc_allocator,
    force_token_to_dead_frame, get_float_from_deadframe, get_int_from_deadframe,
    get_latest_descr_from_deadframe, get_ref_from_deadframe, get_savedata_ref_from_deadframe,
    install_gc_standalone, jit_exc_class_raw, jit_exc_clear, jit_exc_is_pending, jit_exc_raise,
    jit_exc_value_forward, jit_exc_value_peek, jit_exc_value_raw,
    register_call_assembler_blackhole, register_call_assembler_bridge,
    register_call_assembler_force, register_call_assembler_unbox_int, register_jitframe_layout,
    register_materialize_str_call, register_materialize_str_plain, register_prologue_probe_addr,
    register_resumedata_deopt, register_stack_check_addresses, set_gil_hooks,
    set_savedata_ref_on_deadframe,
};
