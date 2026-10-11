// `opimpl_*` handlers and `OPCODE_IMPLEMENTATIONS`.
// Included into `dispatch.rs` so the handlers share `JitCodeMachine`'s
// private fields. `pyjitpl.py` `setup_insns` / `_get_opimpl_method` /
// `opcode_implementations`.

impl<'mi, S, R> JitCodeMachine<'mi, S, R>
where
    S: JitCodeSym,
    R: JitCodeRuntime,
{
    // Auto-split from `execute_one_instruction`'s match: one `opimpl_*`
    // per arm, dispatched through `OPCODE_IMPLEMENTATIONS`.
    // `pyjitpl.py` `setup_insns` / `_get_opimpl_method` / `opcode_implementations`.

    /// `pyjitpl.py` `MetaInterpStaticData.opcode_implementations`, filled by
    /// `setup_insns` from `_get_opimpl_method`. Indexed by opcode byte.
    const OPCODE_IMPLEMENTATIONS: [OpImplFn<'mi, S, R>; 256] = {
        let mut table: [OpImplFn<'mi, S, R>; 256] = [Self::opimpl_unknown; 256];
        table[jitcode::insns::BC_LIVE as usize] = Self::opimpl_live;
        table[jitcode::insns::BC_UNREACHABLE as usize] = Self::opimpl_unreachable;
        table[jitcode::insns::BC_LOAD_STATE_FIELD as usize] = Self::opimpl_load_state_field;
        table[jitcode::insns::BC_STORE_STATE_FIELD as usize] = Self::opimpl_store_state_field;
        table[jitcode::insns::BC_LOAD_STATE_FIELD_REF as usize] = Self::opimpl_load_state_field_ref;
        table[jitcode::insns::BC_STORE_STATE_FIELD_REF as usize] =
            Self::opimpl_store_state_field_ref;
        table[jitcode::insns::BC_LOAD_STATE_FIELD_FLOAT as usize] =
            Self::opimpl_load_state_field_float;
        table[jitcode::insns::BC_STORE_STATE_FIELD_FLOAT as usize] =
            Self::opimpl_store_state_field_float;
        table[jitcode::insns::BC_LOAD_STATE_ARRAY as usize] = Self::opimpl_load_state_array;
        table[jitcode::insns::BC_STORE_STATE_ARRAY as usize] = Self::opimpl_store_state_array;
        table[jitcode::insns::BC_GETFIELD_VABLE_I as usize] = Self::opimpl_getfield_vable_i;
        table[jitcode::insns::BC_GETFIELD_VABLE_R as usize] = Self::opimpl_getfield_vable_r;
        table[jitcode::insns::BC_GETFIELD_VABLE_F as usize] = Self::opimpl_getfield_vable_f;
        table[jitcode::insns::BC_NEW as usize] = Self::opimpl_new;
        table[jitcode::insns::BC_NEW_WITH_VTABLE as usize] = Self::opimpl_new;
        table[jitcode::insns::BC_SETFIELD_GC_I as usize] = Self::opimpl_setfield_gc_i;
        table[jitcode::insns::BC_SETFIELD_GC_I_C as usize] = Self::opimpl_setfield_gc_i;
        table[jitcode::insns::BC_SETFIELD_GC_R as usize] = Self::opimpl_setfield_gc_i;
        table[jitcode::insns::BC_SETFIELD_GC_F as usize] = Self::opimpl_setfield_gc_i;
        table[jitcode::insns::BC_SETFIELD_RAW_I as usize] = Self::opimpl_setfield_raw_i;
        table[jitcode::insns::BC_SETFIELD_RAW_F as usize] = Self::opimpl_setfield_raw_i;
        table[jitcode::insns::BC_RAW_STORE_I as usize] = Self::opimpl_raw_store_i;
        table[jitcode::insns::BC_RAW_LOAD_I as usize] = Self::opimpl_raw_load_i;
        table[jitcode::insns::BC_RAW_LOAD_F as usize] = Self::opimpl_raw_load_f;
        table[jitcode::insns::BC_GETFIELD_GC_I as usize] = Self::opimpl_getfield_gc_i;
        table[jitcode::insns::BC_GETFIELD_GC_R as usize] = Self::opimpl_getfield_gc_i;
        table[jitcode::insns::BC_GETFIELD_GC_I_PURE as usize] = Self::opimpl_getfield_gc_i;
        table[jitcode::insns::BC_GETFIELD_GC_R_PURE as usize] = Self::opimpl_getfield_gc_i;
        table[jitcode::insns::BC_GETFIELD_GC_F as usize] = Self::opimpl_getfield_gc_f;
        table[jitcode::insns::BC_GETFIELD_GC_F_PURE as usize] = Self::opimpl_getfield_gc_f;
        table[jitcode::insns::BC_GETFIELD_RAW_I as usize] = Self::opimpl_getfield_raw_i;
        table[jitcode::insns::BC_GETFIELD_RAW_F as usize] = Self::opimpl_getfield_raw_f;
        table[jitcode::insns::BC_SETFIELD_VABLE_I_IMM as usize] = Self::opimpl_setfield_vable_i_imm;
        table[jitcode::insns::BC_SETFIELD_VABLE_I as usize] = Self::opimpl_setfield_vable_i;
        table[jitcode::insns::BC_SETFIELD_VABLE_R as usize] = Self::opimpl_setfield_vable_r;
        table[jitcode::insns::BC_SETFIELD_VABLE_F as usize] = Self::opimpl_setfield_vable_f;
        table[jitcode::insns::BC_ARRAYLEN_GC as usize] = Self::opimpl_arraylen_gc;
        table[jitcode::insns::BC_GETARRAYITEM_GC_I as usize] = Self::opimpl_getarrayitem_gc_i;
        table[jitcode::insns::BC_GETARRAYITEM_GC_I_PURE as usize] = Self::opimpl_getarrayitem_gc_i;
        table[jitcode::insns::BC_GETARRAYITEM_GC_F as usize] = Self::opimpl_getarrayitem_gc_f;
        table[jitcode::insns::BC_GETARRAYITEM_GC_F_PURE as usize] = Self::opimpl_getarrayitem_gc_f;
        table[jitcode::insns::BC_GETARRAYITEM_GC_R_RID as usize] =
            Self::opimpl_getarrayitem_gc_r_rid;
        table[jitcode::insns::BC_GETARRAYITEM_GC_R_PURE as usize] =
            Self::opimpl_getarrayitem_gc_r_rid;
        table[jitcode::insns::BC_SETARRAYITEM_GC_I as usize] = Self::opimpl_setarrayitem_gc_i;
        table[jitcode::insns::BC_SETARRAYITEM_GC_R as usize] = Self::opimpl_setarrayitem_gc_i;
        table[jitcode::insns::BC_SETARRAYITEM_GC_F as usize] = Self::opimpl_setarrayitem_gc_i;
        table[jitcode::insns::BC_GETARRAYITEM_VABLE_I as usize] = Self::opimpl_getarrayitem_vable_i;
        table[jitcode::insns::BC_GETARRAYITEM_VABLE_R as usize] = Self::opimpl_getarrayitem_vable_r;
        table[jitcode::insns::BC_GETARRAYITEM_VABLE_F as usize] = Self::opimpl_getarrayitem_vable_f;
        table[jitcode::insns::BC_SETARRAYITEM_VABLE_I as usize] = Self::opimpl_setarrayitem_vable_i;
        table[jitcode::insns::BC_SETARRAYITEM_VABLE_R as usize] = Self::opimpl_setarrayitem_vable_r;
        table[jitcode::insns::BC_SETARRAYITEM_VABLE_F as usize] = Self::opimpl_setarrayitem_vable_f;
        table[jitcode::insns::BC_ARRAYLEN_VABLE as usize] = Self::opimpl_arraylen_vable;
        table[jitcode::insns::BC_ARRAYBASE_VABLE as usize] = Self::opimpl_arraybase_vable;
        table[jitcode::insns::BC_HINT_FORCE_VIRTUALIZABLE as usize] =
            Self::opimpl_hint_force_virtualizable;
        table[jitcode::insns::BC_INT_ADD as usize] = Self::opimpl_int_add;
        table[jitcode::insns::BC_INT_SUB as usize] = Self::opimpl_int_sub;
        table[jitcode::insns::BC_INT_MUL as usize] = Self::opimpl_int_mul;
        table[jitcode::insns::BC_INT_ADD_JUMP_IF_OVF as usize] = Self::opimpl_int_add_jump_if_ovf;
        table[jitcode::insns::BC_INT_SUB_JUMP_IF_OVF as usize] = Self::opimpl_int_sub_jump_if_ovf;
        table[jitcode::insns::BC_INT_MUL_JUMP_IF_OVF as usize] = Self::opimpl_int_mul_jump_if_ovf;
        table[jitcode::insns::BC_INT_AND as usize] = Self::opimpl_int_and;
        table[jitcode::insns::BC_INT_SIGNEXT as usize] = Self::opimpl_int_signext;
        table[jitcode::insns::BC_INT_OR as usize] = Self::opimpl_int_or;
        table[jitcode::insns::BC_INT_XOR as usize] = Self::opimpl_int_xor;
        table[jitcode::insns::BC_INT_LSHIFT as usize] = Self::opimpl_int_lshift;
        table[jitcode::insns::BC_INT_RSHIFT as usize] = Self::opimpl_int_rshift;
        table[jitcode::insns::BC_INT_EQ as usize] = Self::opimpl_int_eq;
        table[jitcode::insns::BC_INT_NE as usize] = Self::opimpl_int_ne;
        table[jitcode::insns::BC_INT_LT as usize] = Self::opimpl_int_lt;
        table[jitcode::insns::BC_INT_LE as usize] = Self::opimpl_int_le;
        table[jitcode::insns::BC_INT_GT as usize] = Self::opimpl_int_gt;
        table[jitcode::insns::BC_INT_GE as usize] = Self::opimpl_int_ge;
        table[jitcode::insns::BC_UINT_RSHIFT as usize] = Self::opimpl_uint_rshift;
        table[jitcode::insns::BC_UINT_MUL_HIGH as usize] = Self::opimpl_uint_mul_high;
        table[jitcode::insns::BC_UINT_LT as usize] = Self::opimpl_uint_lt;
        table[jitcode::insns::BC_UINT_LE as usize] = Self::opimpl_uint_le;
        table[jitcode::insns::BC_UINT_GT as usize] = Self::opimpl_uint_gt;
        table[jitcode::insns::BC_UINT_GE as usize] = Self::opimpl_uint_ge;
        table[jitcode::insns::BC_INT_BETWEEN as usize] = Self::opimpl_int_between;
        table[jitcode::insns::BC_INT_NEG as usize] = Self::opimpl_int_neg;
        table[jitcode::insns::BC_INT_INVERT as usize] = Self::opimpl_int_invert;
        table[jitcode::insns::BC_INT_IS_TRUE as usize] = Self::opimpl_int_is_true;
        table[jitcode::insns::BC_INT_IS_ZERO as usize] = Self::opimpl_int_is_zero;
        table[jitcode::insns::BC_PTR_EQ as usize] = Self::opimpl_ptr_eq;
        table[jitcode::insns::BC_PTR_NE as usize] = Self::opimpl_ptr_ne;
        table[jitcode::insns::BC_INSTANCE_PTR_EQ as usize] = Self::opimpl_instance_ptr_eq;
        table[jitcode::insns::BC_INSTANCE_PTR_NE as usize] = Self::opimpl_instance_ptr_ne;
        table[jitcode::insns::BC_PTR_ISZERO as usize] = Self::opimpl_ptr_iszero;
        table[jitcode::insns::BC_PTR_NONZERO as usize] = Self::opimpl_ptr_nonzero;
        table[jitcode::insns::BC_GOTO_IF_NOT as usize] = Self::opimpl_goto_if_not;
        table[jitcode::insns::BC_GOTO_IF_NOT_INT_IS_TRUE as usize] =
            Self::opimpl_goto_if_not_int_is_true;
        table[jitcode::insns::BC_GOTO_IF_NOT_INT_IS_ZERO as usize] =
            Self::opimpl_goto_if_not_int_is_zero;
        table[jitcode::insns::BC_GOTO_IF_NOT_INT_LT as usize] = Self::opimpl_goto_if_not_int_lt;
        table[jitcode::insns::BC_GOTO_IF_NOT_INT_LE as usize] = Self::opimpl_goto_if_not_int_lt;
        table[jitcode::insns::BC_GOTO_IF_NOT_INT_EQ as usize] = Self::opimpl_goto_if_not_int_lt;
        table[jitcode::insns::BC_GOTO_IF_NOT_INT_NE as usize] = Self::opimpl_goto_if_not_int_lt;
        table[jitcode::insns::BC_GOTO_IF_NOT_INT_GT as usize] = Self::opimpl_goto_if_not_int_lt;
        table[jitcode::insns::BC_GOTO_IF_NOT_INT_GE as usize] = Self::opimpl_goto_if_not_int_lt;
        table[jitcode::insns::BC_GOTO_IF_NOT_FLOAT_LT as usize] = Self::opimpl_goto_if_not_float_lt;
        table[jitcode::insns::BC_GOTO_IF_NOT_FLOAT_LE as usize] = Self::opimpl_goto_if_not_float_lt;
        table[jitcode::insns::BC_GOTO_IF_NOT_FLOAT_EQ as usize] = Self::opimpl_goto_if_not_float_lt;
        table[jitcode::insns::BC_GOTO_IF_NOT_FLOAT_NE as usize] = Self::opimpl_goto_if_not_float_lt;
        table[jitcode::insns::BC_GOTO_IF_NOT_FLOAT_GT as usize] = Self::opimpl_goto_if_not_float_lt;
        table[jitcode::insns::BC_GOTO_IF_NOT_FLOAT_GE as usize] = Self::opimpl_goto_if_not_float_lt;
        table[jitcode::insns::BC_GOTO_IF_NOT_PTR_EQ as usize] = Self::opimpl_goto_if_not_ptr_eq;
        table[jitcode::insns::BC_GOTO_IF_NOT_PTR_NE as usize] = Self::opimpl_goto_if_not_ptr_eq;
        table[jitcode::insns::BC_SWITCH as usize] = Self::opimpl_switch;
        table[jitcode::insns::BC_GOTO_IF_NOT_PTR_ISZERO as usize] =
            Self::opimpl_goto_if_not_ptr_iszero;
        table[jitcode::insns::BC_GOTO_IF_NOT_PTR_NONZERO as usize] =
            Self::opimpl_goto_if_not_ptr_iszero;
        table[jitcode::insns::BC_CATCH_EXCEPTION as usize] = Self::opimpl_catch_exception;
        table[jitcode::insns::BC_LAST_EXCEPTION as usize] = Self::opimpl_last_exception;
        table[jitcode::insns::BC_LAST_EXC_VALUE as usize] = Self::opimpl_last_exc_value;
        table[jitcode::insns::BC_GOTO_IF_EXCEPTION_MISMATCH as usize] =
            Self::opimpl_goto_if_exception_mismatch;
        table[jitcode::insns::BC_RVMPROF_CODE as usize] = Self::opimpl_rvmprof_code;
        table[jitcode::insns::BC_JIT_MERGE_POINT as usize] = Self::opimpl_jit_merge_point;
        table[jitcode::insns::BC_JIT_MERGE_POINT_C as usize] = Self::opimpl_jit_merge_point;
        table[jitcode::insns::BC_LOOP_HEADER as usize] = Self::opimpl_loop_header;
        table[jitcode::insns::BC_JUMP as usize] = Self::opimpl_jump;
        table[jitcode::insns::BC_INLINE_CALL as usize] = Self::opimpl_inline_call;
        table[jitcode::insns::BC_INLINE_CALL_R_I as usize] = Self::opimpl_inline_call_r_i;
        table[jitcode::insns::BC_INLINE_CALL_R_R as usize] = Self::opimpl_inline_call_r_i;
        table[jitcode::insns::BC_INLINE_CALL_R_V as usize] = Self::opimpl_inline_call_r_i;
        table[jitcode::insns::BC_INLINE_CALL_IR_I as usize] = Self::opimpl_inline_call_r_i;
        table[jitcode::insns::BC_INLINE_CALL_IR_R as usize] = Self::opimpl_inline_call_r_i;
        table[jitcode::insns::BC_INLINE_CALL_IR_V as usize] = Self::opimpl_inline_call_r_i;
        table[jitcode::insns::BC_INLINE_CALL_IRF_F as usize] = Self::opimpl_inline_call_r_i;
        table[jitcode::insns::BC_INLINE_CALL_IRF_R as usize] = Self::opimpl_inline_call_r_i;
        table[jitcode::insns::BC_INLINE_CALL_IRF_I as usize] = Self::opimpl_inline_call_r_i;
        table[jitcode::insns::BC_INLINE_CALL_IRF_V as usize] = Self::opimpl_inline_call_r_i;
        table[jitcode::insns::BC_RECURSIVE_CALL_INT as usize] = Self::opimpl_recursive_call_int;
        table[jitcode::insns::BC_RECURSIVE_CALL_REF as usize] = Self::opimpl_recursive_call_int;
        table[jitcode::insns::BC_RECURSIVE_CALL_FLOAT as usize] = Self::opimpl_recursive_call_int;
        table[jitcode::insns::BC_RECURSIVE_CALL_VOID as usize] = Self::opimpl_recursive_call_int;
        table[jitcode::insns::BC_INT_RETURN as usize] = Self::opimpl_int_return;
        table[jitcode::insns::BC_INT_RETURN_C as usize] = Self::opimpl_int_return_c;
        table[jitcode::insns::BC_REF_RETURN as usize] = Self::opimpl_ref_return;
        table[jitcode::insns::BC_FLOAT_RETURN as usize] = Self::opimpl_float_return;
        table[jitcode::insns::BC_VOID_RETURN as usize] = Self::opimpl_void_return;
        table[jitcode::insns::BC_RESIDUAL_CALL_R_V as usize] = Self::opimpl_residual_call_r_v;
        table[jitcode::insns::BC_RESIDUAL_CALL_IR_V as usize] = Self::opimpl_residual_call_r_v;
        table[jitcode::insns::BC_RESIDUAL_CALL_IRF_V as usize] = Self::opimpl_residual_call_r_v;
        table[jitcode::insns::BC_RESIDUAL_CALL_R_I as usize] = Self::opimpl_residual_call_r_i;
        table[jitcode::insns::BC_RESIDUAL_CALL_IR_I as usize] = Self::opimpl_residual_call_r_i;
        table[jitcode::insns::BC_RESIDUAL_CALL_IRF_I as usize] = Self::opimpl_residual_call_r_i;
        table[jitcode::insns::BC_RESIDUAL_CALL_R_R as usize] = Self::opimpl_residual_call_r_r;
        table[jitcode::insns::BC_RESIDUAL_CALL_IR_R as usize] = Self::opimpl_residual_call_r_r;
        table[jitcode::insns::BC_RESIDUAL_CALL_IRF_R as usize] = Self::opimpl_residual_call_r_r;
        table[jitcode::insns::BC_RESIDUAL_CALL_IRF_F as usize] = Self::opimpl_residual_call_irf_f;
        table[jitcode::insns::BC_CALL_ASSEMBLER_VOID as usize] = Self::opimpl_call_assembler_void;
        table[jitcode::insns::BC_CONDITIONAL_CALL_IR_V as usize] =
            Self::opimpl_conditional_call_ir_v;
        table[jitcode::insns::BC_CONDITIONAL_CALL_VALUE_IR_I as usize] =
            Self::opimpl_conditional_call_ir_v;
        table[jitcode::insns::BC_CONDITIONAL_CALL_VALUE_IR_R as usize] =
            Self::opimpl_conditional_call_ir_v;
        table[jitcode::insns::BC_RECORD_KNOWN_RESULT_I_IR_V as usize] =
            Self::opimpl_conditional_call_ir_v;
        table[jitcode::insns::BC_RECORD_KNOWN_RESULT_R_IR_V as usize] =
            Self::opimpl_conditional_call_ir_v;
        table[jitcode::insns::BC_COND_CALL_VOID as usize] = Self::opimpl_cond_call_void;
        table[jitcode::insns::BC_COND_CALL_VALUE_INT as usize] = Self::opimpl_cond_call_void;
        table[jitcode::insns::BC_COND_CALL_VALUE_REF as usize] = Self::opimpl_cond_call_void;
        table[jitcode::insns::BC_RECORD_KNOWN_RESULT_INT as usize] = Self::opimpl_cond_call_void;
        table[jitcode::insns::BC_RECORD_KNOWN_RESULT_REF as usize] = Self::opimpl_cond_call_void;
        table[jitcode::insns::BC_MOVE_I as usize] = Self::opimpl_move_i;
        table[jitcode::insns::BC_MOVE_I_C as usize] = Self::opimpl_move_i_c;
        table[jitcode::insns::BC_CALL_ASSEMBLER_INT as usize] = Self::opimpl_call_assembler_int;
        table[jitcode::insns::BC_MOVE_R as usize] = Self::opimpl_move_r;
        table[jitcode::insns::BC_CALL_ASSEMBLER_REF as usize] = Self::opimpl_call_assembler_ref;
        table[jitcode::insns::BC_MOVE_F as usize] = Self::opimpl_move_f;
        table[jitcode::insns::BC_CALL_ASSEMBLER_FLOAT as usize] = Self::opimpl_call_assembler_float;
        table[jitcode::insns::BC_FLOAT_ADD as usize] = Self::opimpl_float_add;
        table[jitcode::insns::BC_FLOAT_SUB as usize] = Self::opimpl_float_sub;
        table[jitcode::insns::BC_FLOAT_MUL as usize] = Self::opimpl_float_mul;
        table[jitcode::insns::BC_FLOAT_TRUEDIV as usize] = Self::opimpl_float_truediv;
        table[jitcode::insns::BC_FLOAT_NEG as usize] = Self::opimpl_float_neg;
        table[jitcode::insns::BC_FLOAT_ABS as usize] = Self::opimpl_float_abs;
        table[jitcode::insns::BC_FLOAT_LT as usize] = Self::opimpl_float_lt;
        table[jitcode::insns::BC_FLOAT_LE as usize] = Self::opimpl_float_le;
        table[jitcode::insns::BC_FLOAT_EQ as usize] = Self::opimpl_float_eq;
        table[jitcode::insns::BC_FLOAT_NE as usize] = Self::opimpl_float_ne;
        table[jitcode::insns::BC_FLOAT_GT as usize] = Self::opimpl_float_gt;
        table[jitcode::insns::BC_FLOAT_GE as usize] = Self::opimpl_float_ge;
        table[jitcode::insns::BC_CAST_INT_TO_FLOAT as usize] = Self::opimpl_cast_int_to_float;
        table[jitcode::insns::BC_CAST_FLOAT_TO_INT as usize] = Self::opimpl_cast_float_to_int;
        table[jitcode::insns::BC_CAST_PTR_TO_INT as usize] = Self::opimpl_cast_ptr_to_int;
        table[jitcode::insns::BC_CAST_INT_TO_PTR as usize] = Self::opimpl_cast_int_to_ptr;
        table[jitcode::insns::BC_CONVERT_FLOAT_BYTES_TO_LONGLONG as usize] =
            Self::opimpl_convert_float_bytes_to_longlong;
        table[jitcode::insns::BC_CONVERT_LONGLONG_BYTES_TO_FLOAT as usize] =
            Self::opimpl_convert_longlong_bytes_to_float;
        table[jitcode::insns::BC_INT_GUARD_VALUE as usize] = Self::opimpl_int_guard_value;
        table[jitcode::insns::BC_ASSERT_NOT_NONE as usize] = Self::opimpl_assert_not_none;
        table[jitcode::insns::BC_RECORD_EXACT_CLASS as usize] = Self::opimpl_record_exact_class;
        table[jitcode::insns::BC_REF_GUARD_VALUE as usize] = Self::opimpl_ref_guard_value;
        table[jitcode::insns::BC_FLOAT_GUARD_VALUE as usize] = Self::opimpl_float_guard_value;
        table[jitcode::insns::BC_GUARD_CLASS as usize] = Self::opimpl_guard_class;
        table[jitcode::insns::BC_GUARD_CLASS_R as usize] = Self::opimpl_guard_class;
        table[jitcode::insns::BC_RAISE as usize] = Self::opimpl_raise;
        table[jitcode::insns::BC_RERAISE as usize] = Self::opimpl_reraise;
        table[jitcode::insns::BC_ABORT as usize] = Self::opimpl_abort;
        table[jitcode::insns::BC_ABORT_RESULT_R as usize] = Self::opimpl_abort;
        table[jitcode::insns::BC_ABORT_PERMANENT as usize] = Self::opimpl_abort_permanent;
        table[jitcode::insns::BC_NEW_ARRAY as usize] = Self::opimpl_new_array;
        table[jitcode::insns::BC_NEW_ARRAY_CLEAR as usize] = Self::opimpl_new_array;
        table[jitcode::insns::BC_NEWLIST_CLEAR as usize] = Self::opimpl_newlist_clear;
        table[jitcode::insns::BC_INT_ISCONSTANT as usize] = Self::opimpl_int_isconstant;
        table[jitcode::insns::BC_REF_ISCONSTANT as usize] = Self::opimpl_ref_isconstant;
        table[jitcode::insns::BC_REF_ISVIRTUAL as usize] = Self::opimpl_ref_isvirtual;
        table[jitcode::insns::BC_STRLEN as usize] = Self::opimpl_strlen;
        table[jitcode::insns::BC_STRGETITEM as usize] = Self::opimpl_strgetitem;
        table
    };

    // RPython `blackhole.py bhimpl_live` — no-op marker
    // emitted by the codewriter ahead of every guard-bearing
    // instruction.  The two operand bytes are the offset into
    // `MetaInterpStaticData.liveness_info`; consumed by
    // `MIFrame::get_list_of_active_boxes` at guard time, not
    // here.  (See also the same skip in
    // `unwind_to_exception_handler` above.)
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_live(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let _liveness_offset = self.frames.current_mut().next_u16();
        TraceAction::Continue
    }

    // pyjitpl.py opimpl_unreachable: raise AssertionError("unreachable").
    // A landing here is a wrong-path generation/dispatch defect; abort
    // the attempt so the interpreter can resume instead of panicking
    // mid-opcode (which left a stack underflow on the Python frame).
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_unreachable(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        return TraceAction::Abort;
    }

    // -- State field access (register/tape machines) --
    // Argcodes: `d` = u16 descr (`assembler.py write_insn`),
    // `i` = u8 register index (`assembler.py write_insn`).
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_load_state_field(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let field_idx = self.frames.current_mut().next_u16() as usize;
        let dest = self.frames.current_mut().next_reg() as usize;
        // `blackhole.rs handler_load_state_field_di`:
        // `registers_i[dest] = registers_i[slot(field_idx)]`.
        let slot = sym.int_identity_slots_base() + field_idx;
        let (opref, value) = self.read_int_identity_slot(ctx, slot);
        self.set_int_reg(ctx, dest, Some(opref), Some(value));
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_store_state_field(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let field_idx = self.frames.current_mut().next_u16() as usize;
        let src = self.frames.current_mut().next_reg() as usize;
        let (opref, value) = self.read_int_reg(ctx, src);
        // `blackhole.rs handler_store_state_field_di`:
        // `registers_i[slot(field_idx)] = registers_i[src]`.
        let slot = sym.int_identity_slots_base() + field_idx;
        self.set_int_identity_slot(ctx, slot, Some(opref), Some(value));
        TraceAction::Continue
    }

    // Ref-typed scalar state field: same shape as the int load/store
    // but the value lives in the ref register bank, so its OpRef
    // carries Type::Ref (input_arg_ref) and feeds getfield_gc as a
    // real ref base. Argcodes: `d` = u16 field index, `r` = ref reg.
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_load_state_field_ref(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let field_idx = self.frames.current_mut().next_u16() as usize;
        let dest = self.frames.current_mut().next_reg() as usize;
        // `blackhole.rs handler_load_state_field_ref_dr`:
        // `registers_r[dest] = registers_r[ref_slot(field_idx)]`.
        let slot = sym
            .ref_scalar_slot(field_idx)
            .expect("ref state field has no identity slot");
        let (opref, value) = self.read_ref_identity_slot(ctx, slot);
        self.set_ref_reg(ctx, dest, Some(opref), Some(value));
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_store_state_field_ref(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let field_idx = self.frames.current_mut().next_u16() as usize;
        let src = self.frames.current_mut().next_reg() as usize;
        let (opref, value) = self.read_ref_reg(ctx, src);
        // `blackhole.rs handler_store_state_field_ref_dr`:
        // `registers_r[ref_slot(field_idx)] = registers_r[src]`.
        let slot = sym
            .ref_scalar_slot(field_idx)
            .expect("ref state field has no identity slot");
        self.set_ref_identity_slot(ctx, slot, Some(opref), Some(value));
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_load_state_field_float(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let field_idx = self.frames.current_mut().next_u16() as usize;
        let dest = self.frames.current_mut().next_u8() as usize;
        // `blackhole.rs handler_load_state_field_float_df`:
        // `registers_f[dest] = registers_f[float_slot(field_idx)]`.
        let slot = sym
            .float_scalar_slot(field_idx)
            .expect("float state field has no identity slot");
        let (opref, value) = self.read_float_identity_slot(ctx, slot);
        self.set_float_reg(ctx, dest, Some(opref), Some(value));
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_store_state_field_float(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let field_idx = self.frames.current_mut().next_u16() as usize;
        let src = self.frames.current_mut().next_u8() as usize;
        let (opref, value) = self.read_float_reg(ctx, src);
        // `blackhole.rs handler_store_state_field_float_df`:
        // `registers_f[float_slot(field_idx)] = registers_f[src]`.
        let slot = sym
            .float_scalar_slot(field_idx)
            .expect("float state field has no identity slot");
        self.set_float_identity_slot(ctx, slot, Some(opref), Some(value));
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_load_state_array(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let array_idx = self.frames.current_mut().next_u16() as usize;
        let index_reg = self.frames.current_mut().next_reg() as usize;
        let dest = self.frames.current_mut().next_reg() as usize;
        let (_, index_concrete) = self.read_int_reg(ctx, index_reg);
        let elem_idx = index_concrete as usize;
        let Some(slot) = sym.array_elem_slot(array_idx, elem_idx) else {
            return TraceAction::Abort;
        };
        let (opref, value) = self.read_int_identity_slot(ctx, slot);
        self.set_int_reg(ctx, dest, Some(opref), Some(value));
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_store_state_array(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let array_idx = self.frames.current_mut().next_u16() as usize;
        let index_reg = self.frames.current_mut().next_reg() as usize;
        let src = self.frames.current_mut().next_reg() as usize;
        let (_, index_concrete) = self.read_int_reg(ctx, index_reg);
        let elem_idx = index_concrete as usize;
        let (opref, value) = self.read_int_reg(ctx, src);
        // `handler_store_state_array_dii` writes
        // `registers_i[StateFieldLayout::array_elem_slot]`.
        let Some(slot) = sym.array_elem_slot(array_idx, elem_idx) else {
            return TraceAction::Abort;
        };
        self.set_int_identity_slot(ctx, slot, Some(opref), Some(value));
        TraceAction::Continue
    }

    // -- First-class virtualizable access (getfield_vable_*) --
    // `_opimpl_getarrayitem_vable` (and the getfield/setfield siblings)
    // returns `virtualizable_boxes[index]`, a Box carrying both the traced
    // reference AND its concrete value (`getint` / `getref_base` /
    // `getfloatstorage`). Seeded at `initialize_virtualizable` and updated
    // on every `vable_setfield` / `vable_setarrayitem_indexed`. Do NOT peek
    // the live frame here.
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_getfield_vable_i(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        // R7 parity: pyjitpl.py opimpl_getfield_vable_i
        // takes (box, fielddescr, pc); pc threads to
        // _nonstandard_virtualizable.  Capture opcode_pc
        // before read_vable_getfield advances code_cursor.
        let (opcode_pc, vable_reg, field_idx, dest) = {
            let frame = self.frames.current_mut();
            let opcode_pc = frame.code_cursor - 1;
            let (vable_reg, field_idx, dest) = frame.read_vable_getfield();
            (opcode_pc, vable_reg, field_idx, dest)
        };
        let Some((vable_opref, fielddescr)) = self.vable_field_descr(ctx, vable_reg, field_idx)
        else {
            return TraceAction::Abort;
        };
        // Concrete struct pointer for pyjitpl.py MIFrame._nonstandard_virtualizable
        // cache-hit sanity check (plumbing;
        // wires the check itself).
        let vable_struct_ptr = self.read_ref_reg(ctx, vable_reg).1;
        let nonstandard =
            self.nonstandard_virtualizable(ctx, sym, opcode_pc, vable_opref, &fielddescr);
        let (opref, value) = ctx.vable_getfield_int_checked(
            nonstandard,
            self.cpu.as_ref(),
            vable_opref,
            vable_struct_ptr,
            fielddescr,
        );
        self.set_int_reg(ctx, dest, Some(opref), value.map(value_as_int_bits));
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_getfield_vable_r(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let (opcode_pc, vable_reg, field_idx, dest) = {
            let frame = self.frames.current_mut();
            let opcode_pc = frame.code_cursor - 1;
            let (vable_reg, field_idx, dest) = frame.read_vable_getfield();
            (opcode_pc, vable_reg, field_idx, dest)
        };
        let Some((vable_opref, fielddescr)) = self.vable_field_descr(ctx, vable_reg, field_idx)
        else {
            return TraceAction::Abort;
        };
        let vable_struct_ptr = self.read_ref_reg(ctx, vable_reg).1;
        let nonstandard =
            self.nonstandard_virtualizable(ctx, sym, opcode_pc, vable_opref, &fielddescr);
        let (opref, value) = ctx.vable_getfield_ref_checked(
            nonstandard,
            self.cpu.as_ref(),
            vable_opref,
            vable_struct_ptr,
            fielddescr,
        );
        self.set_ref_reg(ctx, dest, Some(opref), value.map(value_as_ref_bits));
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_getfield_vable_f(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let (opcode_pc, vable_reg, field_idx, dest) = {
            let frame = self.frames.current_mut();
            let opcode_pc = frame.code_cursor - 1;
            let (vable_reg, field_idx, dest) = frame.read_vable_getfield();
            (opcode_pc, vable_reg, field_idx, dest)
        };
        let Some((vable_opref, fielddescr)) = self.vable_field_descr(ctx, vable_reg, field_idx)
        else {
            return TraceAction::Abort;
        };
        let vable_struct_ptr = self.read_ref_reg(ctx, vable_reg).1;
        let nonstandard =
            self.nonstandard_virtualizable(ctx, sym, opcode_pc, vable_opref, &fielddescr);
        let (opref, value) = ctx.vable_getfield_float_checked(
            nonstandard,
            self.cpu.as_ref(),
            vable_opref,
            vable_struct_ptr,
            fielddescr,
        );
        self.set_float_reg(ctx, dest, Some(opref), value.map(value_as_float_bits));
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_new(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        // blackhole.py bhimpl_new / bhimpl_new_with_vtable.
        // The tracer both *executes* the allocation (so subsequent
        // setfield/getfield steps in this trace read live memory) and
        // *records* New / NewWithVtable so the optimizer can virtualize
        // the struct away when it does not escape.
        let with_vtable = bytecode == jitcode::insns::BC_NEW_WITH_VTABLE;
        let (size, vtable, type_id, headerless, descr, dest) = {
            let frame = self.frames.current_mut();
            let (descr_idx, dest) = frame.read_new();
            let bh = frame
                .runtime_bh_descr(descr_idx)
                .unwrap_or_else(|| panic!("BC_NEW: descrs[{descr_idx}] is not a BhDescr entry"));
            (
                bh.as_size(),
                bh.get_vtable(),
                bh.resolve_gc_tid(),
                bh.is_headerless(),
                size_descr_ref_from_bh(bh),
                dest,
            )
        };
        // A `headerless` descr means the interpreter owns this struct in
        // its own collected pool (`headerless_structs`), which is what
        // compiled code allocates it from, through
        // `call_malloc_nursery_headerless`. Putting it on the host heap
        // instead hands the interpreter an object its collector cannot
        // see: a moving collector range-checks its own pool, so it
        // neither traces through the object nor forwards the references
        // hanging off it, and the reachable graph below it is lost on
        // the next collection.
        //
        // A headered GC-managed descr (real `type_id`) is the same
        // problem one field deeper: a host-heap block carries no type
        // word at `ref - 8`, so the collector never traces the struct
        // and whatever its ref fields point at dies while the following
        // `getfield` steps of this same trace still read them. It goes
        // to the non-moving old generation, matching `runner.rs`
        // bh_new / bh_new_with_vtable.
        //
        // The allocation must not collect. This runs mid-jitcode with
        // raw object pointers live in the machine's own register bank —
        // the `getfield` result feeding the `setfield` that follows this
        // `new` — and that bank belongs to no root set, so a moving
        // collection here would strand them. Both GC paths are
        // no-collect, and old-gen is mark-sweep, so the pointer handed
        // back to the register bank also survives later collections.
        //
        // A non-GC descr (`type_id == 0`, raw buffer) and an allocation
        // the GC declines keep the host heap; the vtable word at offset
        // 0 (the OBJECTPTR typeptr slot) is written either way so a
        // trace-time GuardClass reads the right class.
        let size = size.max(1);
        let gc_ptr = if headerless {
            majit_gc::alloc_nursery_headerless_no_collect(size).0
        } else if type_id != 0 {
            majit_gc::alloc_oldgen_typed(type_id, size).0
        } else {
            0
        };
        let ptr = if gc_ptr != 0 {
            gc_ptr as i64
        } else if let Some(ptr) = host_malloc_fixedsize(size) {
            // `GcLLDescr_boehm.malloc_fixedsize`, same arm as
            // `llmodel_alloc` / `bh_new_with_vtable`. A raw
            // `alloc_zeroed` block is not in the heap that hook owns.
            ptr as i64
        } else {
            let layout = std::alloc::Layout::from_size_align(size, 8)
                .expect("BC_NEW: invalid struct layout");
            unsafe { std::alloc::alloc_zeroed(layout) as i64 }
        };
        // A null from the hook or from `alloc_zeroed` is the same
        // failure `BC_NEW_ARRAY` aborts on. The vtable store must not
        // run against that pointer.
        if ptr == 0 {
            return TraceAction::Abort;
        }
        if with_vtable && vtable != 0 {
            unsafe { *(ptr as *mut usize) = vtable };
        }
        let kind = if with_vtable {
            OpCode::NewWithVtable
        } else {
            OpCode::New
        };
        // A site that executes the operation itself still owes the
        // funnel its two counts: `execute_and_record` counts what it
        // executes and `_record_helper` counts what it appends.
        ctx.profiler().count_ops(kind, crate::counters::OPS);
        ctx.profiler()
            .count_ops(kind, crate::counters::RECORDED_OPS);
        // `execute_and_record` → `history.record(..., resvalue)`:
        // `History._make_op` builds the RefFrontendOp with the pointer.
        let op = ctx.record_op_with_descr_value(
            kind,
            &[],
            descr,
            Some(Value::Ref(majit_ir::GcRef(ptr as usize))),
        );
        // `execute_new` stamps `heapcache.new(resbox)`;
        // `execute_new_with_vtable` stamps `class_now_known` on top of
        // it. The vtable written at offset 0 just above is the word
        // `cls_of_box` reads back, so the class is known here by
        // construction — a zero one is the "unavailable" spelling and
        // stays unrecorded.
        ctx.heap_cache_mut().new_object(op);
        if with_vtable && vtable != 0 {
            ctx.heap_cache_mut().class_now_known(op);
        }
        self.set_ref_reg(ctx, dest, Some(op), Some(ptr));
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_setfield_gc_i(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        // blackhole.py bhimpl_getfield_raw_f bhimpl_setfield_gc_{i,r,f}: record
        // SetfieldGc (a single op-kind whose descr carries the field
        // type) and write the field through the live struct ptr. The
        // value word is read from the int/ref/float bank by type; the
        // `/rcd` c-form (USE_C_FORM, assembler.py) inlines a signed
        // byte in place of the int-register slot.
        let (struct_reg, value_reg, descr_idx) = {
            let frame = self.frames.current_mut();
            frame.read_setfield_gc()
        };
        let (offset, field_size, fielddescr) = {
            let frame = self.frames.current_mut();
            let bh = frame.runtime_bh_descr(descr_idx).unwrap_or_else(|| {
                panic!("BC_SETFIELD_GC: descrs[{descr_idx}] is not a BhDescr entry")
            });
            let field_size = match bh {
                crate::blackhole::BhDescr::Field { field_size, .. } => *field_size,
                _ => 8,
            };
            let offset = field_offset_from_bh(bh, "BC_SETFIELD_GC");
            let fielddescr = frame
                .runtime_optimizer_descr(descr_idx)
                .unwrap_or_else(|| field_descr_ref_from_bh(bh).1);
            (offset, field_size, fielddescr)
        };
        let (struct_opref, struct_ptr) = self.read_ref_reg(ctx, struct_reg);
        let (value_opref, concrete) = match bytecode {
            jitcode::insns::BC_SETFIELD_GC_R => self.read_ref_reg(ctx, value_reg),
            jitcode::insns::BC_SETFIELD_GC_F => self.read_float_reg(ctx, value_reg),
            jitcode::insns::BC_SETFIELD_GC_I_C => {
                let v = value_reg as u8 as i8 as i64;
                (OpRef::ConstInt(v), v)
            }
            _ => self.read_int_reg(ctx, value_reg),
        };
        let field_key = heapcache_field_key(&fielddescr);
        // `_record_helper` runs `heapcache.invalidate_caches` before it
        // appends. `clear_caches_not_necessary` lists SETFIELD_GC, so
        // only the mark-escaped half runs: a ref written into an
        // already-escaped struct escapes with it. Same shape as
        // BC_SETARRAYITEM_GC below.
        ctx.heapcache_invalidate_caches_varargs(
            OpCode::SetfieldGc,
            None,
            &[struct_opref, value_opref],
        );
        ctx.profiler()
            .count_ops(OpCode::SetfieldGc, crate::counters::OPS);
        ctx.profiler()
            .count_ops(OpCode::SetfieldGc, crate::counters::RECORDED_OPS);
        ctx.record_op_with_descr(OpCode::SetfieldGc, &[struct_opref, value_opref], fielddescr);
        // `execute_setfield_gc`'s trailing `heapcache.setfield`, which
        // `_opimpl_setfield_gc_any` spells `upd.setfield(valuebox)`.
        // The cache stores the Box identity, not the value word.
        if let Some(field_key) = field_key {
            ctx.heapcache_setfield_cached(struct_opref, field_key, value_opref);
        }
        if struct_ptr != 0 {
            // blackhole.py bhimpl_getfield_raw_f stores through the fielddescr,
            // which carries the field's byte width, and the getfield
            // twin above already reads at that width. A sub-word field
            // written as a full word writes over whatever follows it —
            // for a `u32` with a live `u32` sibling behind it, the
            // store of one silently zeroes the other — and a field the
            // walk widened would then disagree with the same store as
            // the backend emits it.
            let addr = (struct_ptr as usize).wrapping_add(offset);
            unsafe {
                match field_size {
                    1 => core::ptr::write_unaligned(addr as *mut u8, concrete as u8),
                    2 => core::ptr::write_unaligned(addr as *mut u16, concrete as u16),
                    4 => core::ptr::write_unaligned(addr as *mut u32, concrete as u32),
                    // A non-{1,2,4,8}-byte field has no fixed-width
                    // primitive store; fall back to the word-sized one.
                    _ => core::ptr::write_unaligned(addr as *mut i64, concrete),
                }
            }
            // A ref store adds a heap edge struct→value; notify the GC
            // on the container so a young value survives a minor
            // collection triggered later in the walk (mirrors
            // `bh_setfield_gc_r` and the setarrayitem case below).
            if bytecode == jitcode::insns::BC_SETFIELD_GC_R
                && majit_gc::gc_owns_object(struct_ptr as usize)
            {
                majit_gc::gc_write_barrier(majit_ir::GcRef(struct_ptr as usize));
            }
        }
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_setfield_raw_i(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        // pyjitpl.py `_opimpl_setfield_raw_any`:
        // `execute_with_descr(rop.SETFIELD_RAW, fielddescr, box, valuebox)`.
        // `blackhole.py bhimpl_setfield_raw_{i,f}` (`iid` / `ifd`).
        // The struct address lives in the int bank (`history.getkind` Signed
        // for a raw pointer), not the ref bank.
        let is_float = bytecode == jitcode::insns::BC_SETFIELD_RAW_F;
        let (struct_reg, value_reg, descr_idx) = {
            let frame = self.frames.current_mut();
            frame.read_setfield_gc()
        };
        let (offset, field_size, fielddescr) = {
            let frame = self.frames.current_mut();
            let bh = frame.runtime_bh_descr(descr_idx).unwrap_or_else(|| {
                panic!("BC_SETFIELD_RAW: descrs[{descr_idx}] is not a BhDescr entry")
            });
            let field_size = match bh {
                crate::blackhole::BhDescr::Field { field_size, .. } => *field_size,
                _ => 8,
            };
            let offset = field_offset_from_bh(bh, "BC_SETFIELD_RAW");
            let fielddescr = frame
                .runtime_optimizer_descr(descr_idx)
                .unwrap_or_else(|| field_descr_ref_from_bh(bh).1);
            (offset, field_size, fielddescr)
        };
        let (struct_opref, struct_ptr) = self.read_int_reg(ctx, struct_reg);
        let (value_opref, concrete) = if is_float {
            self.read_float_reg(ctx, value_reg)
        } else {
            self.read_int_reg(ctx, value_reg)
        };
        ctx.heapcache_invalidate_caches_varargs(
            OpCode::SetfieldRaw,
            None,
            &[struct_opref, value_opref],
        );
        ctx.profiler()
            .count_ops(OpCode::SetfieldRaw, crate::counters::OPS);
        ctx.profiler()
            .count_ops(OpCode::SetfieldRaw, crate::counters::RECORDED_OPS);
        ctx.record_op_with_descr(
            OpCode::SetfieldRaw,
            &[struct_opref, value_opref],
            fielddescr,
        );
        if struct_ptr != 0 {
            let addr = (struct_ptr as usize).wrapping_add(offset);
            unsafe {
                match field_size {
                    1 => core::ptr::write_unaligned(addr as *mut u8, concrete as u8),
                    2 => core::ptr::write_unaligned(addr as *mut u16, concrete as u16),
                    4 => core::ptr::write_unaligned(addr as *mut u32, concrete as u32),
                    _ => core::ptr::write_unaligned(addr as *mut i64, concrete),
                }
            }
        }
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_raw_store_i(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        // blackhole.py bhimpl_raw_store_i: store an int
        // into raw native memory at `base + ea` (byte offset).
        // jtransform.py rewrite_op_raw_store lowers a
        // `raw_storage_setitem` to this op with an
        // `arraydescrof(rffi.CArray(T))` descr. Record RawStore and
        // perform the concrete store through the live raw address
        // while tracing (metainterp executes as it records — the
        // store side effect is the actual write for this iteration,
        // mirroring BC_GETARRAYITEM_GC_I's concrete read).
        //
        // Encoding (`assembler.rs raw_store_i`):
        //   [BC_RAW_STORE_I][base_reg u8][ea_reg u8][value_reg u8]
        //   [descr_idx u16]
        let (base_reg, ea_reg, value_reg, descr_idx) = {
            let frame = self.frames.current_mut();
            let base_reg = frame.next_reg() as usize;
            let ea_reg = frame.next_reg() as usize;
            let value_reg = frame.next_reg() as usize;
            let descr_idx = frame.next_u16() as usize;
            (base_reg, ea_reg, value_reg, descr_idx)
        };
        let Some(descr) = self.dispatch_array_descr_ref(ctx, descr_idx) else {
            return TraceAction::Abort;
        };
        let Some((_base_size, itemsize, _is_signed)) = self.dispatch_array_geometry(descr_idx)
        else {
            return TraceAction::Abort;
        };
        let (base_opref, base_addr) = self.read_int_reg(ctx, base_reg);
        let (ea_opref, ea_value) = self.read_int_reg(ctx, ea_reg);
        let (value_opref, value) = self.read_int_reg(ctx, value_reg);
        // pyjitpl.py `_record_helper` invalidates the
        // heapcache before recording a side-effecting op so a later
        // `raw_load` at the same `(base, ea)` re-reads instead of
        // folding a stale cached value.
        ctx.heapcache_invalidate_caches_varargs(
            OpCode::RawStore,
            None,
            &[base_opref, ea_opref, value_opref],
        );
        ctx.profiler()
            .count_ops(OpCode::RawStore, crate::counters::OPS);
        ctx.profiler()
            .count_ops(OpCode::RawStore, crate::counters::RECORDED_OPS);
        ctx.record_op_with_descr(
            OpCode::RawStore,
            &[base_opref, ea_opref, value_opref],
            descr,
        );
        // Concrete eval: descriptor-sized store at `base + ea`
        // (`ea` is already a byte offset — `emit_dynamic_offset_addr`
        // in the backend adds it to `base` unscaled).
        //
        // SAFETY: the kernel clamps `ea` to an in-bounds byte offset
        // (0 when the access would trap), so `base_addr + ea_value`
        // is within the outer interpreter's linear-memory allocation.
        let item_addr = (base_addr as usize).wrapping_add(ea_value as usize);
        unsafe {
            match itemsize {
                1 => core::ptr::write_unaligned(item_addr as *mut u8, value as u8),
                2 => core::ptr::write_unaligned(item_addr as *mut u16, value as u16),
                4 => core::ptr::write_unaligned(item_addr as *mut u32, value as u32),
                8 => core::ptr::write_unaligned(item_addr as *mut i64, value),
                other => {
                    panic!("BC_RAW_STORE_I: unsupported itemsize {}", other)
                }
            }
        }
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_raw_load_i(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        // blackhole.py bhimpl_raw_load_i: read an int from
        // raw native memory at `base + ea` (byte offset). jtransform.py:
        // 1165-1171 rewrite_op_raw_load lowers a `raw_storage_getitem`
        // to this op with an `arraydescrof(rffi.CArray(T))` descr.
        // Record RawLoadI and perform the concrete read while tracing.
        //
        // Encoding (`assembler.rs raw_load_i`):
        //   [BC_RAW_LOAD_I][base_reg u8][ea_reg u8][descr_idx u16][dst u8]
        let (base_reg, ea_reg, descr_idx, dst) = {
            let frame = self.frames.current_mut();
            let base_reg = frame.next_reg() as usize;
            let ea_reg = frame.next_reg() as usize;
            let descr_idx = frame.next_u16() as usize;
            let dst = frame.next_reg() as usize;
            (base_reg, ea_reg, descr_idx, dst)
        };
        let Some(descr) = self.dispatch_array_descr_ref(ctx, descr_idx) else {
            return TraceAction::Abort;
        };
        let Some((_base_size, itemsize, is_signed)) = self.dispatch_array_geometry(descr_idx)
        else {
            return TraceAction::Abort;
        };
        let (base_opref, base_addr) = self.read_int_reg(ctx, base_reg);
        let (ea_opref, ea_value) = self.read_int_reg(ctx, ea_reg);
        // Concrete eval: descriptor-sized read at `base + ea`. It runs
        // before the record because `execute_and_record` takes the
        // value rather than computing it.
        //
        // SAFETY: the kernel clamps `ea` to an in-bounds byte offset
        // (0 when the access would trap), so `base_addr + ea_value`
        // is within the linear-memory allocation.
        let item_addr = (base_addr as usize).wrapping_add(ea_value as usize);
        let concrete = unsafe {
            match (itemsize, is_signed) {
                (1, true) => core::ptr::read_unaligned(item_addr as *const i8) as i64,
                (1, false) => core::ptr::read_unaligned(item_addr as *const u8) as i64,
                (2, true) => core::ptr::read_unaligned(item_addr as *const i16) as i64,
                (2, false) => core::ptr::read_unaligned(item_addr as *const u16) as i64,
                (4, true) => core::ptr::read_unaligned(item_addr as *const i32) as i64,
                (4, false) => core::ptr::read_unaligned(item_addr as *const u32) as i64,
                (8, _) => core::ptr::read_unaligned(item_addr as *const i64),
                other => {
                    panic!(
                        "BC_RAW_LOAD_I: unsupported (itemsize, signed) = {:?}",
                        other
                    )
                }
            }
        };
        // `RawLoadI` is outside `is_pure_with_descr` at every descr —
        // `test_raw_load_i_stays_non_pure_for_eval_breaker_poll` pins
        // it there, because a pure raw load lets `optimize_guard_false`
        // delete the back-edge eval-breaker poll — so the funnel always
        // records. What it adds over a bare `record_op_with_descr` is
        // the op counters and the concrete stamp every sibling read
        // already carries.
        let opref = ctx.execute_and_record(
            Some(self.cpu.as_ref()),
            OpCode::RawLoadI,
            Some(descr),
            &[base_opref, ea_opref],
            Some(Value::Int(concrete)),
            self.last_exception_value,
        );
        self.set_int_reg(ctx, dst, Some(opref), Some(concrete));
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_raw_load_f(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        // f64 sibling of BC_RAW_LOAD_I: read an f64 from raw native
        // memory at `base + ea` (byte offset) into the FLOAT register
        // bank, preserving the exact bit pattern. Record RawLoadF and
        // perform the concrete read while tracing.
        //
        // Encoding (`assembler.rs raw_load_f`), identical to raw_load_i:
        //   [BC_RAW_LOAD_F][base_reg u8][ea_reg u8][descr_idx u16][dst u8]
        let (base_reg, ea_reg, descr_idx, dst) = {
            let frame = self.frames.current_mut();
            let base_reg = frame.next_u8() as usize;
            let ea_reg = frame.next_u8() as usize;
            let descr_idx = frame.next_u16() as usize;
            let dst = frame.next_u8() as usize;
            (base_reg, ea_reg, descr_idx, dst)
        };
        let Some(descr) = self.dispatch_array_descr_ref(ctx, descr_idx) else {
            return TraceAction::Abort;
        };
        let Some((_base_size, itemsize, _is_signed)) = self.dispatch_array_geometry(descr_idx)
        else {
            return TraceAction::Abort;
        };
        let (base_opref, base_addr) = self.read_int_reg(ctx, base_reg);
        let (ea_opref, ea_value) = self.read_int_reg(ctx, ea_reg);
        // Concrete eval: an 8-byte f64 read at `base + ea`, carried as
        // raw bits in the float bank (set_float_reg takes i64 bits).
        //
        // SAFETY: the kernel clamps `ea` to an in-bounds byte offset,
        // so `base_addr + ea_value` is within the allocation.
        let item_addr = (base_addr as usize).wrapping_add(ea_value as usize);
        let concrete_bits = match itemsize {
            8 => unsafe { core::ptr::read_unaligned(item_addr as *const i64) },
            other => panic!("BC_RAW_LOAD_F: unsupported itemsize = {other}"),
        };
        // Never folds, for the reason given in the `BC_RAW_LOAD_I` arm.
        let opref = ctx.execute_and_record(
            Some(self.cpu.as_ref()),
            OpCode::RawLoadF,
            Some(descr),
            &[base_opref, ea_opref],
            Some(Value::Float(f64::from_bits(concrete_bits as u64))),
            self.last_exception_value,
        );
        self.set_float_reg(ctx, dst, Some(opref), Some(concrete_bits));
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_getfield_gc_i(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        // blackhole.py bhimpl_getfield_gc_i bhimpl_getfield_gc_{i,r}: load the
        // field through the live struct ptr and record GetfieldGc.
        // The _pure aliases (blackhole.py bhimpl_getfield_gc_i_pure) read identically;
        // majit has no separate pure op-kind, so the recorded
        // GetfieldGc{I,R} carries the (immutable) field descr and the
        // pure pass folds it from there.
        let is_ref = matches!(
            bytecode,
            jitcode::insns::BC_GETFIELD_GC_R | jitcode::insns::BC_GETFIELD_GC_R_PURE
        );
        let (struct_reg, descr_idx, dest) = {
            let frame = self.frames.current_mut();
            frame.read_getfield_gc()
        };
        let (offset, field_size, is_field_signed, fielddescr) = {
            let frame = self.frames.current_mut();
            let bh = frame.runtime_bh_descr(descr_idx).unwrap_or_else(|| {
                panic!("BC_GETFIELD_GC: descrs[{descr_idx}] is not a BhDescr entry")
            });
            let (field_size, is_field_signed) = match bh {
                crate::blackhole::BhDescr::Field {
                    field_size,
                    is_field_signed,
                    ..
                } => (*field_size, *is_field_signed),
                _ => (8, false),
            };
            let offset = field_offset_from_bh(bh, "BC_GETFIELD_GC");
            let fielddescr = frame
                .runtime_optimizer_descr(descr_idx)
                .unwrap_or_else(|| field_descr_ref_from_bh(bh).1);
            (offset, field_size, is_field_signed, fielddescr)
        };
        let (struct_opref, struct_ptr) = self.read_ref_reg(ctx, struct_reg);
        // blackhole.py bhimpl_getfield_gc_i reads the field through the fielddescr,
        // which carries the field's byte width; a sub-word integer field
        // (`Char`/`Bool`/`INT` narrower than a word) must be read at that
        // width, not as a full word — otherwise adjacent bytes leak into
        // the value. Ref fields are always word-sized pointers.
        let loaded = if struct_ptr == 0 {
            0
        } else if is_ref {
            // llmodel.py bh_getfield_gc_r / read_ref_at_mem use GCREF,
            // whose width is the target pointer width, not FLOATSTORAGE.
            self.cpu
                .bh_getfield_gc_r(
                    struct_ptr as usize,
                    fielddescr
                        .as_field_descr()
                        .expect("GC ref field descriptor"),
                )
                .0 as i64
        } else {
            let addr = (struct_ptr as usize).wrapping_add(offset);
            unsafe {
                match (field_size, is_field_signed) {
                    (1, true) => core::ptr::read_unaligned(addr as *const i8) as i64,
                    (1, false) => core::ptr::read_unaligned(addr as *const u8) as i64,
                    (2, true) => core::ptr::read_unaligned(addr as *const i16) as i64,
                    (2, false) => core::ptr::read_unaligned(addr as *const u16) as i64,
                    (4, true) => core::ptr::read_unaligned(addr as *const i32) as i64,
                    (4, false) => core::ptr::read_unaligned(addr as *const u32) as i64,
                    // A non-{1,2,4,8}-byte integer field has no
                    // fixed-width primitive read; fall back to the
                    // word-sized read (the pre-sized-read behavior).
                    _ => core::ptr::read_unaligned(addr as *const i64),
                }
            }
        };
        if !is_ref && ctx.is_bridge_trace && crate::heapdbg_enabled() {
            use std::sync::atomic::{AtomicU64, Ordering};
            static N: AtomicU64 = AtomicU64::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            if n < 240 {
                eprintln!(
                    "@@@HEAP getfield_i n={} struct_ptr={:#x} off={} loaded={}",
                    n, struct_ptr, offset, loaded
                );
            }
        }
        let kind = if is_ref {
            OpCode::GetfieldGcR
        } else {
            OpCode::GetfieldGcI
        };
        let value = if is_ref {
            Value::Ref(majit_ir::GcRef(loaded as usize))
        } else {
            Value::Int(loaded)
        };
        // `is_pure_with_descr` admits GETFIELD_GC_{I,R} only for a
        // descr that answers `is_always_pure`, so only an immutable
        // field off a constant struct folds. The null case hands the
        // funnel no concrete: `loaded` is a fabricated 0 rather than a
        // real load, and `llmodel.py protect_speculative_field` rejects
        // a null gcptr before any fold — the executor row would
        // dereference it.
        // `_opimpl_getfield_gc_any_pureornot` opens on
        // `upd = heapcache.get_field_updater(box, fielddescr)` and
        // returns `upd.currfieldbox` without recording when the cache
        // answers; only the miss path reaches `execute_with_descr`,
        // and it stores the result back with
        // `upd.getfield_now_known(resbox)`.  The sibling
        // `BC_GETARRAYITEM_GC_*` arms carry both halves already, and
        // `BC_SETFIELD_GC` fills this very cache — without the two
        // halves here that cache is written and never read.
        //
        // `opimpl_getfield_gc_{i,r,f}` runs one test ahead of the
        // updater: a constant struct's always-pure field bypasses the
        // heapcache completely, executing and returning a Const
        // without reading, writing, or recording anything.  Spelling
        // it as "this load has no cache key" reaches the miss path
        // below, whose `execute_and_record` performs that very fold —
        // `is_pure_with_descr` admits GETFIELD_GC only through
        // `descr.is_always_pure()`, the same predicate — and the key
        // being `None` is what then skips both cache halves.
        let bypasses_heapcache = fielddescr.is_always_pure() && struct_opref.is_constant();
        let field_key = if bypasses_heapcache {
            None
        } else {
            heapcache_field_key(&fielddescr)
        };
        let cached = field_key.and_then(|key| ctx.heapcache_getfield_cached(struct_opref, key));
        let cached_payload = cached.and_then(|cached| match ctx.box_value(cached) {
            Some(Value::Int(n)) => Some(n),
            Some(Value::Ref(r)) => Some(r.0 as i64),
            _ => None,
        });
        // Parentless placeholder descrs can share `field_key`, so a
        // Position word and a frame `base` collide. A hit whose
        // payload is not the live load is that collision: take the
        // miss path and record a fresh op.
        let cached = cached.filter(|_| {
            struct_ptr == 0 || cached_payload.is_none() || cached_payload == Some(loaded)
        });
        let (op, reg_concrete) = if let Some(cached) = cached {
            // `profiler.count_ops(rop.GETFIELD_GC_I,
            // Counters.HEAPCACHED_OPS)` — folded-away op
            // accounting on the cache hit.  The opnum in
            // `_opimpl_getfield_gc_any_pureornot` is the
            // `GETFIELD_GC_I` literal whatever the field's type,
            // so the ref arm shares the int bucket rather than
            // reporting its own `kind`.
            ctx.profiler().count_ops(
                OpCode::GetfieldGcI,
                crate::pyjitpl::counters::HEAPCACHED_OPS,
            );
            // The sanity check compares the freshly executed load
            // against the cached box's own payload
            // (`currfieldbox.getint()` / `.getref_base()`), read
            // here through `box_value` — the const pool,
            // standard-virtualizable shadow and frontend value
            // slot composed into one answer.  A `None` payload is
            // an entry seeded without a live concrete and skips
            // the check.  A null struct fabricated `loaded` rather
            // than reading, so it has nothing to compare either.
            let expected = match ctx.box_value(cached) {
                Some(Value::Int(n)) => Some(n),
                Some(Value::Ref(r)) => Some(r.0 as i64),
                _ => None,
            };
            assert!(
                struct_ptr == 0 || !matches!(expected, Some(exp) if exp != loaded),
                "_opimpl_getfield_gc_any_pureornot sanity check ({}): \
                     loaded {loaded} != cached {expected:?} \
                     (field_key={field_key:?}, struct_ptr={struct_ptr:#x})",
                if is_ref { "ref" } else { "int" },
            );
            // The cached box is returned even on a mismatch, so
            // the register takes its payload, not the fresh load.
            (cached, expected.unwrap_or(loaded))
        } else {
            let op = ctx.execute_and_record(
                Some(self.cpu.as_ref()),
                kind,
                Some(fielddescr),
                &[struct_opref],
                (struct_ptr != 0).then_some(value),
                self.last_exception_value,
            );
            if let Some(field_key) = field_key {
                ctx.heapcache_getfield_now_known(struct_opref, field_key, op);
            }
            (op, loaded)
        };
        if is_ref {
            self.set_ref_reg(ctx, dest, Some(op), Some(reg_concrete));
        } else {
            self.set_int_reg(ctx, dest, Some(op), Some(reg_concrete));
        }
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_getfield_raw_i(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        // pyjitpl.py `opimpl_getfield_raw_i`:
        // `execute_with_descr(rop.GETFIELD_RAW_I, fielddescr, box)`.
        // `blackhole.py bhimpl_getfield_raw_i` (`id>i`).
        let (struct_reg, descr_idx, dest) = {
            let frame = self.frames.current_mut();
            frame.read_getfield_gc()
        };
        let (offset, field_size, is_field_signed, fielddescr) = {
            let frame = self.frames.current_mut();
            let bh = frame.runtime_bh_descr(descr_idx).unwrap_or_else(|| {
                panic!("BC_GETFIELD_RAW_I: descrs[{descr_idx}] is not a BhDescr entry")
            });
            let (field_size, is_field_signed) = match bh {
                crate::blackhole::BhDescr::Field {
                    field_size,
                    is_field_signed,
                    ..
                } => (*field_size, *is_field_signed),
                _ => (8, false),
            };
            let offset = field_offset_from_bh(bh, "BC_GETFIELD_RAW_I");
            let fielddescr = frame
                .runtime_optimizer_descr(descr_idx)
                .unwrap_or_else(|| field_descr_ref_from_bh(bh).1);
            (offset, field_size, is_field_signed, fielddescr)
        };
        let (struct_opref, struct_ptr) = self.read_int_reg(ctx, struct_reg);
        let loaded = if struct_ptr == 0 {
            0
        } else {
            let addr = (struct_ptr as usize).wrapping_add(offset);
            unsafe {
                match (field_size, is_field_signed) {
                    (1, true) => core::ptr::read_unaligned(addr as *const i8) as i64,
                    (1, false) => core::ptr::read_unaligned(addr as *const u8) as i64,
                    (2, true) => core::ptr::read_unaligned(addr as *const i16) as i64,
                    (2, false) => core::ptr::read_unaligned(addr as *const u16) as i64,
                    (4, true) => core::ptr::read_unaligned(addr as *const i32) as i64,
                    (4, false) => core::ptr::read_unaligned(addr as *const u32) as i64,
                    _ => core::ptr::read_unaligned(addr as *const i64),
                }
            }
        };
        ctx.profiler()
            .count_ops(OpCode::GetfieldRawI, crate::counters::OPS);
        ctx.profiler()
            .count_ops(OpCode::GetfieldRawI, crate::counters::RECORDED_OPS);
        let op = ctx.record_op_with_descr(OpCode::GetfieldRawI, &[struct_opref], fielddescr);
        ctx.set_opref_concrete(op, Value::Int(loaded));
        self.set_int_reg(ctx, dest, Some(op), Some(loaded));
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_getfield_gc_f(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        // blackhole.py bhimpl_getfield_gc_f (+ the _pure
        // alias at :1441-1443): load the f64-bit field through the
        // live struct ptr and record GetfieldGcF.
        let (struct_reg, descr_idx, dest) = {
            let frame = self.frames.current_mut();
            frame.read_getfield_gc()
        };
        let (offset, fielddescr) = {
            let frame = self.frames.current_mut();
            let bh = frame.runtime_bh_descr(descr_idx).unwrap_or_else(|| {
                panic!("BC_GETFIELD_GC_F: descrs[{descr_idx}] is not a BhDescr entry")
            });
            let offset = field_offset_from_bh(bh, "BC_GETFIELD_GC_F");
            (
                offset,
                frame
                    .runtime_optimizer_descr(descr_idx)
                    .unwrap_or_else(|| field_descr_ref_from_bh(bh).1),
            )
        };
        let (struct_opref, struct_ptr) = self.read_ref_reg(ctx, struct_reg);
        let loaded = if struct_ptr != 0 {
            unsafe { *((struct_ptr as *const u8).add(offset) as *const i64) }
        } else {
            0
        };
        // See the `BC_GETFIELD_GC_I` arm on the descr gate, the null
        // case, and the two heapcache halves
        // `_opimpl_getfield_gc_any_pureornot` brackets the load with.
        // The float sanity check is the one upstream spells
        // `ConstFloat(resvalue).same_constant(upd.currfieldbox
        // .constbox())` rather than `==`, so that two NaNs compare
        // equal; comparing the raw bit patterns is that same
        // predicate, and `loaded` is already the bits.
        // `opimpl_getfield_gc_{i,r,f}` runs one test ahead of the
        // updater: a constant struct's always-pure field bypasses the
        // heapcache completely, executing and returning a Const
        // without reading, writing, or recording anything.  Spelling
        // it as "this load has no cache key" reaches the miss path
        // below, whose `execute_and_record` performs that very fold —
        // `is_pure_with_descr` admits GETFIELD_GC only through
        // `descr.is_always_pure()`, the same predicate — and the key
        // being `None` is what then skips both cache halves.
        let bypasses_heapcache = fielddescr.is_always_pure() && struct_opref.is_constant();
        let field_key = if bypasses_heapcache {
            None
        } else {
            heapcache_field_key(&fielddescr)
        };
        let cached = field_key.and_then(|key| ctx.heapcache_getfield_cached(struct_opref, key));
        let (op, reg_concrete) = if let Some(cached) = cached {
            // The same `GETFIELD_GC_I` literal — `box_trace.rs`
            // wires its float port to the int bucket for this
            // reason.
            ctx.profiler().count_ops(
                OpCode::GetfieldGcI,
                crate::pyjitpl::counters::HEAPCACHED_OPS,
            );
            let expected = match ctx.box_value(cached) {
                Some(Value::Float(f)) => Some(f.to_bits() as i64),
                _ => None,
            };
            assert!(
                struct_ptr == 0 || !matches!(expected, Some(exp) if exp != loaded),
                "_opimpl_getfield_gc_any_pureornot sanity check (float): \
                     loaded {loaded:#x} != cached {expected:?} \
                     (field_key={field_key:?}, struct_ptr={struct_ptr:#x})",
            );
            (cached, expected.unwrap_or(loaded))
        } else {
            let op = ctx.execute_and_record(
                Some(self.cpu.as_ref()),
                OpCode::GetfieldGcF,
                Some(fielddescr),
                &[struct_opref],
                (struct_ptr != 0).then_some(Value::Float(f64::from_bits(loaded as u64))),
                self.last_exception_value,
            );
            if let Some(field_key) = field_key {
                ctx.heapcache_getfield_now_known(struct_opref, field_key, op);
            }
            (op, loaded)
        };
        self.set_float_reg(ctx, dest, Some(op), Some(reg_concrete));
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_getfield_raw_f(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        // pyjitpl.py `opimpl_getfield_raw_f`:
        // `execute_with_descr(rop.GETFIELD_RAW_F, fielddescr, box)`.
        // `blackhole.py bhimpl_getfield_raw_f` (`id>f`).
        let (struct_reg, descr_idx, dest) = {
            let frame = self.frames.current_mut();
            frame.read_getfield_gc()
        };
        let (offset, fielddescr) = {
            let frame = self.frames.current_mut();
            let bh = frame.runtime_bh_descr(descr_idx).unwrap_or_else(|| {
                panic!("BC_GETFIELD_RAW_F: descrs[{descr_idx}] is not a BhDescr entry")
            });
            let offset = field_offset_from_bh(bh, "BC_GETFIELD_RAW_F");
            let fielddescr = frame
                .runtime_optimizer_descr(descr_idx)
                .unwrap_or_else(|| field_descr_ref_from_bh(bh).1);
            (offset, fielddescr)
        };
        let (struct_opref, struct_ptr) = self.read_int_reg(ctx, struct_reg);
        let loaded_bits = if struct_ptr == 0 {
            0
        } else {
            let addr = (struct_ptr as usize).wrapping_add(offset);
            unsafe { core::ptr::read_unaligned(addr as *const i64) }
        };
        ctx.profiler()
            .count_ops(OpCode::GetfieldRawF, crate::counters::OPS);
        ctx.profiler()
            .count_ops(OpCode::GetfieldRawF, crate::counters::RECORDED_OPS);
        let op = ctx.record_op_with_descr(OpCode::GetfieldRawF, &[struct_opref], fielddescr);
        ctx.set_opref_concrete(op, Value::Float(f64::from_bits(loaded_bits as u64)));
        self.set_float_reg(ctx, dest, Some(op), Some(loaded_bits));
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_setfield_vable_i_imm(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let (opcode_pc, vable_reg, field_idx, imm) = {
            let frame = self.frames.current_mut();
            let opcode_pc = frame.code_cursor - 1;
            let (vable_reg, field_idx, imm) = frame.read_vable_setfield_imm();
            (opcode_pc, vable_reg, field_idx, imm)
        };
        let Some((vable_opref, fielddescr)) = self.vable_field_descr(ctx, vable_reg, field_idx)
        else {
            return TraceAction::Abort;
        };
        let nonstandard =
            self.nonstandard_virtualizable(ctx, sym, opcode_pc, vable_opref, &fielddescr);
        let imm_box = ctx.const_int(imm);
        let _write = ctx.vable_setfield_checked(
            nonstandard,
            vable_opref,
            fielddescr,
            imm_box,
            Some(Value::Int(imm)),
        );
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_setfield_vable_i(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let (opcode_pc, vable_reg, field_idx, src) = {
            let frame = self.frames.current_mut();
            let opcode_pc = frame.code_cursor - 1;
            let (vable_reg, field_idx, src) = frame.read_vable_setfield();
            (opcode_pc, vable_reg, field_idx, src)
        };
        let Some((vable_opref, fielddescr)) = self.vable_field_descr(ctx, vable_reg, field_idx)
        else {
            return TraceAction::Abort;
        };
        let (value, concrete) = self.read_int_reg(ctx, src);
        let nonstandard =
            self.nonstandard_virtualizable(ctx, sym, opcode_pc, vable_opref, &fielddescr);
        let _write = ctx.vable_setfield_checked(
            nonstandard,
            vable_opref,
            fielddescr,
            value,
            Some(Value::Int(concrete)),
        );
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_setfield_vable_r(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let (opcode_pc, vable_reg, field_idx, src) = {
            let frame = self.frames.current_mut();
            let opcode_pc = frame.code_cursor - 1;
            let (vable_reg, field_idx, src) = frame.read_vable_setfield();
            (opcode_pc, vable_reg, field_idx, src)
        };
        let Some((vable_opref, fielddescr)) = self.vable_field_descr(ctx, vable_reg, field_idx)
        else {
            return TraceAction::Abort;
        };
        let (value, concrete) = self.read_ref_reg(ctx, src);
        let nonstandard =
            self.nonstandard_virtualizable(ctx, sym, opcode_pc, vable_opref, &fielddescr);
        let _write = ctx.vable_setfield_checked(
            nonstandard,
            vable_opref,
            fielddescr,
            value,
            Some(Value::Ref(majit_ir::GcRef(concrete as usize))),
        );
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_setfield_vable_f(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let (opcode_pc, vable_reg, field_idx, src) = {
            let frame = self.frames.current_mut();
            let opcode_pc = frame.code_cursor - 1;
            let (vable_reg, field_idx, src) = frame.read_vable_setfield();
            (opcode_pc, vable_reg, field_idx, src)
        };
        let Some((vable_opref, fielddescr)) = self.vable_field_descr(ctx, vable_reg, field_idx)
        else {
            return TraceAction::Abort;
        };
        let (value, concrete) = self.read_float_reg(ctx, src);
        let nonstandard =
            self.nonstandard_virtualizable(ctx, sym, opcode_pc, vable_opref, &fielddescr);
        let _write = ctx.vable_setfield_checked(
            nonstandard,
            vable_opref,
            fielddescr,
            value,
            Some(Value::Float(f64::from_bits(concrete as u64))),
        );
        TraceAction::Continue
    }

    // ── BC_ARRAYLEN_GC ──
    //
    // RPython parity: pyjitpl.py `opimpl_arraylen_gc`
    // (`execute_with_descr(rop.ARRAYLEN_GC, arraydescr, arraybox)`) and
    // blackhole.py `bhimpl_arraylen_gc(cpu, r, d) -> i`.
    //
    // Encoding (`arraylen_gc/rd>i`): [array_reg u8][descr_idx u16][dst u8].
    // Reads the GC array's length word at the descr's lendescr offset
    // (`bh_arraylen_gc`: `sizeof(usize)` signed load), records
    // `OpCode::ArraylenGc` so the optimizer can narrow the lenbound /
    // virtualize, and stamps the result register with the concrete
    // length.  Aborts (fail-loud) if the descr does not resolve to an
    // array descr in the canonical pool.
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_arraylen_gc(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let (array_reg, descr_idx, dst) = {
            let frame = self.frames.current_mut();
            let array_reg = frame.next_reg() as usize;
            let descr_idx = frame.next_u16() as usize;
            let dst = frame.next_reg() as usize;
            (array_reg, descr_idx, dst)
        };
        let Some(descr) = self.dispatch_array_descr_ref(ctx, descr_idx) else {
            return TraceAction::Abort;
        };
        let (array_opref, array_addr) = self.read_ref_reg(ctx, array_reg);
        // Concrete length via the lendescr (`bh_arraylen_gc`); `None`
        // when the cpu is unwired or the descr lacks a lendescr, in
        // which case the recorded op is left unstamped.
        let concrete = ctx.arraylen_sanity_load(array_addr, &descr);
        let reg_concrete = match &concrete {
            Some(majit_ir::Value::Int(n)) => Some(*n),
            _ => None,
        };
        let opref = ctx.opimpl_arraylen_gc(
            self.cpu.as_ref(),
            array_opref,
            descr,
            concrete,
            self.last_exception_value,
        );
        self.set_int_reg(ctx, dst, Some(opref), reg_concrete);
        TraceAction::Continue
    }

    // ── BC_GETARRAYITEM_GC_I ──
    //
    // RPython parity: pyjitpl.py MIFrame._do_getarrayitem_gc_any:
    //
    //     return self.execute_with_descr(rop.GETARRAYITEM_GC_I,
    //                                    arraydescr, arraybox, indexbox)
    //
    // Encoding (`jitcode/assembler.rs`'s `getarrayitem_gc_i`):
    //   [BC_GETARRAYITEM_GC_I][array_reg u8][index_reg u8]
    //   [descr_idx u16][dst u8]
    //
    // The dispatch JitCode body emits this op for `program[pc]`
    // opcode-fetch lowering (`jitcode_lower::lower_dispatch_body`).
    // The Ref register holds the slice data pointer
    // (`codegen_trace.rs`'s `generate_trace_fn` emits
    // `*const #env_type as *const () as usize`); the descr pool entry
    // is a `CanonicalBhDescr::Array { itemsize=1, base_size=0,
    // is_item_signed=false, ... }` (`jitcode/assembler.rs`'s
    // `add_gc_byte_array_descr`).  Concrete eval reads byte at
    // `array_addr + index` and zero-extends to i64 (matching
    // CPython `ord()` 0..=255 semantics).
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_getarrayitem_gc_i(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let (array_reg, index_reg, descr_idx, dst) = {
            let frame = self.frames.current_mut();
            let array_reg = frame.next_reg() as usize;
            let index_reg = frame.next_reg() as usize;
            let descr_idx = frame.next_u16() as usize;
            let dst = frame.next_reg() as usize;
            (array_reg, index_reg, descr_idx, dst)
        };
        let Some(descr) = self.dispatch_array_descr_ref(ctx, descr_idx) else {
            return TraceAction::Abort;
        };
        let Some((base_size, itemsize, is_signed)) = self.dispatch_array_geometry(descr_idx) else {
            return TraceAction::Abort;
        };
        let (array_opref, array_addr) = self.read_ref_reg(ctx, array_reg);
        let (index_opref, index_value) = self.read_int_reg(ctx, index_reg);
        // `getarrayitem_gc_i_pure` shares this body: the load, the
        // heapcache handling and the register write are identical, and
        // only the recorded opcode differs — the same reason
        // `blackhole.py` aliases `bhimpl_getarrayitem_gc_i_pure` onto
        // the plain impl.  The distinction that does matter is which
        // opcode reaches the optimizer: `GetarrayitemGcPureI` is inside
        // the always-pure range, so `OpHelpers.is_pure_with_descr`
        // admits it, while the plain read is admitted by no descr.
        // The codewriter has already made that choice —
        // `OpKind::ArrayRead { pure }` picks the spelling from whether
        // the list is ever mutated (`ll_getitem_foldable_nonneg`).
        let opcode = if bytecode == jitcode::insns::BC_GETARRAYITEM_GC_I_PURE {
            OpCode::GetarrayitemGcPureI
        } else {
            OpCode::GetarrayitemGcI
        };
        let descr_index = descr.index();
        // pyjitpl.py `_do_getarrayitem_gc_any`: check
        // `heapcache.getarrayitem(arraybox, indexbox, arraydescr)`
        // before recording.  Cache hit short-circuits the record,
        // counts `HEAPCACHED_OPS`, and returns the cached box;
        // miss falls through to `execute_with_descr` + the
        // `getarrayitem_now_known` cache store at line 671.
        let cached = ctx.heapcache_getarrayitem(array_opref, index_opref, descr_index);
        // Concrete eval: descriptor-aware sized load with sign
        // extension chosen by `is_item_signed`.  Mirrors PyPy
        // `llmodel.py unpack_arraydescr_size + read_int_at_mem(
        // gcref, ofs + index * size, size, sign)` and the
        // dynasm-side `bh_getarrayitem_gc_i` impl
        // (`runner.rs`); previously this site hard-coded a
        // u8 zero-extend which only happened to work because the
        // sole production caller (dispatch JitCode opcode-fetch)
        // uses `add_gc_byte_array_descr` (`jitcode/assembler.rs`,
        // itemsize=1, is_item_signed=false).  Generalising
        // matches the descriptor-driven contract that
        // BC_GETARRAYITEM_GC_I's name promises.
        //
        // SAFETY: `array_addr` is a GC-managed array pointer
        // threaded through `Ref` register reads; `index_value`
        // is bounded by the outer interpreter's array-length
        // precondition (`codegen_trace.rs`'s `generate_trace_fn`
        // narrows the fat
        // slice pointer to its data ptr; only emitted by
        // `lower_dispatch_body` for slice-typed envs that the
        // outer loop already bounds-checks).
        let item_addr = (array_addr as usize)
            .wrapping_add(base_size)
            .wrapping_add((index_value as usize).wrapping_mul(itemsize));
        let concrete = unsafe {
            match (itemsize, is_signed) {
                (1, true) => *(item_addr as *const i8) as i64,
                (1, false) => *(item_addr as *const u8) as i64,
                (2, true) => *(item_addr as *const i16) as i64,
                (2, false) => *(item_addr as *const u16) as i64,
                (4, true) => *(item_addr as *const i32) as i64,
                (4, false) => *(item_addr as *const u32) as i64,
                (8, _) => *(item_addr as *const i64),
                other => panic!(
                    "getarrayitem_gc_i: unsupported (itemsize, signed) = {:?}",
                    other,
                ),
            }
        };
        // `execute_varargs(pure=True)` → `record_result_of_call_pure`:
        // an all-constant read of an array the opcode itself declares
        // immutable folds to a ConstInt and is not recorded — the same
        // record-time fold `strgetitem(green_str, green_pc)` gets, and
        // for the same reason, that the pure spelling is what licenses
        // it. `is_pure_with_descr` admits `GetarrayitemGcPureI` and no
        // descr admits the plain read, so the plain read is recorded
        // even with two constant arguments; whether an array qualifies
        // is the emitter's call (`OpKind::ArrayRead { pure }`,
        // `jitcode/assembler.rs`'s `getarrayitem_gc_i_pure`), not this
        // site's. Done at record time the read hits the live array
        // directly, so the optimizer's `protect_speculative_array`
        // typeid check — which a raw `&[u8]` data pointer, having no GC
        // type header, would fail — never applies.
        let foldable = opcode == OpCode::GetarrayitemGcPureI
            && array_opref.is_constant()
            && index_opref.is_constant();
        let (opref, reg_concrete) = if foldable {
            // `execute_and_record` counts the operation *before* it
            // decides to fold, so a hand-built constant that skips the
            // funnel under-reports OPS on green pure-array reads.
            //
            // The fold itself stays here rather than routing: the
            // funnel would re-read the item through
            // `Cpu::bh_getarrayitem_gc_i`, a second reader of an array
            // this site has already read directly with the descr's own
            // geometry — and of a raw data pointer carrying no GC type
            // header, which is why the record-time fold is licensed at
            // all (see above).
            ctx.profiler().count_ops(opcode, crate::counters::OPS);
            (ctx.const_int(concrete), concrete)
        } else if let Some(cached) = cached {
            // pyjitpl.py MIFrame._do_getarrayitem_gc_any `count_ops(rop.GETARRAYITEM_GC_I,
            // Counters.HEAPCACHED_OPS)` — folded-away op accounting.
            ctx.profiler()
                .count_ops(opcode, crate::pyjitpl::counters::HEAPCACHED_OPS);
            // pyjitpl.py MIFrame._do_getarrayitem_gc_any sanity check: compare the
            // freshly executed load (`resvalue`) against the
            // cached box's `tobox.getint()`.  On mismatch
            // `_record_helper` records a fallback op whose
            // return value is discarded; `assert 0` fires in
            // debug mode; the function still returns the
            // (stale) cached box.  `_record_helper` routes
            // through `heapcache.invalidate_caches`, but that
            // call short-circuits on GETARRAYITEM_GC_I
            // (`mark_escaped` does not escape the read,
            // `clear_caches_not_necessary` returns True), so
            // the heapcache state is intentionally left
            // untouched.  The cached Box's intrinsic value is
            // the upstream `tobox.getint()` payload — fetched
            // through `box_value(cached)` which composes the
            // const pool, standard-virtualizable shadow, and
            // the frontend object's `value` field (RPython
            // `currfieldbox.getint()` dispatch parity).
            // `None` payload (entry seeded without a live
            // concrete) skips the check.
            let expected = match ctx.box_value(cached) {
                Some(majit_ir::Value::Int(n)) => Some(n),
                _ => None,
            };
            // Cache hit propagates the stale `tobox.getint()`
            // into the destination on mismatch — pyjitpl.py MIFrame._do_getarrayitem_gc_any
            // returns `tobox` so the caller sees the cached
            // box's int, not `resvalue`.  Match that by
            // selecting `expected` (stale) when the assertion
            // fires, else the freshly executed int (which
            // equals expected in the no-mismatch arm).
            let stale = matches!(expected, Some(exp) if exp != concrete);
            if stale {
                // pyjitpl.py `_record_helper` invalidates
                // before recording.  `clear_caches_not_necessary`
                // short-circuits for GETARRAYITEM_GC_I (no-side-
                // effect read), so the only remaining side effect
                // is `mark_escaped` escaping `array_opref` and
                // `index_opref` — match that structure here.
                ctx.heapcache_invalidate_caches_varargs(opcode, None, &[array_opref, index_opref]);
                // The mismatch fallback records without executing: `_record_helper` alone.
                ctx.profiler()
                    .count_ops(opcode, crate::counters::RECORDED_OPS);
                let _ = ctx.record_op_with_descr(opcode, &[array_opref, index_opref], descr);
                debug_assert!(
                    false,
                    "{:?} sanity check failed: \
                     cached={:?} concrete={}",
                    opcode, expected, concrete,
                );
            }
            let reg_concrete = if stale {
                expected.expect("stale only set when expected is Some")
            } else {
                concrete
            };
            (cached, reg_concrete)
        } else {
            ctx.profiler().count_ops(opcode, crate::counters::OPS);
            ctx.profiler()
                .count_ops(opcode, crate::counters::RECORDED_OPS);
            let opref = ctx.record_op_with_descr(opcode, &[array_opref, index_opref], descr);
            // pyjitpl.py MIFrame._do_getarrayitem_gc_any `heapcache.getarrayitem_now_known`.
            // Pair the recorded opref with the live `concrete`
            // payload — mirrors RPython's `resbox` Box carrying
            // both identity and value from `executor.execute`.
            // `Box.value` parity: stamp the result OpRef's
            // frontend value slot so `lookup_opref_concrete(opref)`
            // returns the runtime concrete (RPython
            // `IntFrontendOp(pos, intval)` construction-time
            // field assignment).
            ctx.set_opref_concrete(opref, majit_ir::Value::Int(concrete));
            ctx.heapcache_getarrayitem_now_known(array_opref, index_opref, descr_index, opref);
            (opref, concrete)
        };
        self.set_int_reg(ctx, dst, Some(opref), Some(reg_concrete));
        TraceAction::Continue
    }

    // ── BC_GETARRAYITEM_GC_F ──
    //
    // `opimpl_getarrayitem_gc_f` records
    // `_do_getarrayitem_gc_any(rop.GETARRAYITEM_GC_F, ..., 'f')`.
    // `opimpl_getarrayitem_gc_f_pure` folds a const array and a const
    // index through `executor.wrap_constant` and otherwise records
    // `_do_getarrayitem_gc_any(rop.GETARRAYITEM_GC_PURE_F, ..., 'f')`.
    // `blackhole.py` aliases `bhimpl_getarrayitem_gc_f_pure` onto
    // `bhimpl_getarrayitem_gc_f`, so the load, the heapcache, and the
    // register write are one body; only the recorded opcode differs.
    //
    // Encoding (`jitcode/assembler.rs` `getarrayitem_gc_f`):
    //   [opcode][array_reg u8][index_reg u8][descr_idx u16][dst u8]
    // The element is one f64. The float register stores that f64's
    // raw bits.
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_getarrayitem_gc_f(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let (array_reg, index_reg, descr_idx, dst) = {
            let frame = self.frames.current_mut();
            let array_reg = frame.next_reg() as usize;
            let index_reg = frame.next_reg() as usize;
            let descr_idx = frame.next_u16() as usize;
            let dst = frame.next_reg() as usize;
            (array_reg, index_reg, descr_idx, dst)
        };
        let Some(descr) = self.dispatch_array_descr_ref(ctx, descr_idx) else {
            return TraceAction::Abort;
        };
        let Some((base_size, itemsize, _is_signed)) = self.dispatch_array_geometry(descr_idx)
        else {
            return TraceAction::Abort;
        };
        let (array_opref, array_addr) = self.read_ref_reg(ctx, array_reg);
        let (index_opref, index_value) = self.read_int_reg(ctx, index_reg);
        let opcode = if bytecode == jitcode::insns::BC_GETARRAYITEM_GC_F_PURE {
            OpCode::GetarrayitemGcPureF
        } else {
            OpCode::GetarrayitemGcF
        };
        let descr_index = descr.index();
        // `_do_getarrayitem_gc_any`: `heapcache.getarrayitem` before
        // recording. A hit returns the cached box; a miss records and
        // then `getarrayitem_now_known`.
        let cached = ctx.heapcache_getarrayitem(array_opref, index_opref, descr_index);
        let item_addr = (array_addr as usize)
            .wrapping_add(base_size)
            .wrapping_add((index_value as usize).wrapping_mul(itemsize));
        if itemsize != 8 {
            panic!("getarrayitem_gc_f: unsupported itemsize {itemsize}");
        }
        // SAFETY: `array_addr` is the array data pointer held in the
        // ref register and `index_value` selects an element. itemsize
        // is 8, so this reads one f64 as raw bits. `read_unaligned`
        // matches `bh_getarrayitem_gc_f` / `read_float_at_mem`, which
        // load `FLOATSTORAGE` without assuming natural alignment.
        let concrete = unsafe { core::ptr::read_unaligned(item_addr as *const i64) };
        // `opimpl_getarrayitem_gc_f_pure`: a const array and a const
        // index bypass the heapcache and fold to `wrap_constant`.
        // The plain read stays recorded; `GetarrayitemGcPureF` is what
        // licenses the fold, same as `GetarrayitemGcPureI`.
        let foldable = opcode == OpCode::GetarrayitemGcPureF
            && array_opref.is_constant()
            && index_opref.is_constant();
        let (opref, reg_concrete) = if foldable {
            ctx.profiler().count_ops(opcode, crate::counters::OPS);
            (ctx.const_float(concrete), concrete)
        } else if let Some(cached) = cached {
            ctx.profiler()
                .count_ops(opcode, crate::pyjitpl::counters::HEAPCACHED_OPS);
            // `_do_getarrayitem_gc_any` `typ == 'f'`:
            // `ConstFloat(resvalue).same_constant(tobox.constbox())`,
            // which compares `longlong.extract_bits` so NaN payloads
            // match and `0.0` stays distinct from `-0.0`. A mismatch
            // records a fallback op, asserts in debug, and still
            // answers with the cached box. `clear_caches_not_necessary`
            // short-circuits a getarrayitem, so the heapcache entry
            // stays; `mark_escaped` still runs on the two inputs.
            // `None` (no stamped concrete) skips the check.
            let expected = match ctx.box_value(cached) {
                Some(majit_ir::Value::Float(f)) => Some(f.to_bits() as i64),
                _ => None,
            };
            let stale = matches!(expected, Some(exp) if exp != concrete);
            if stale {
                ctx.heapcache_invalidate_caches_varargs(opcode, None, &[array_opref, index_opref]);
                ctx.profiler()
                    .count_ops(opcode, crate::counters::RECORDED_OPS);
                let _ = ctx.record_op_with_descr(opcode, &[array_opref, index_opref], descr);
                debug_assert!(
                    false,
                    "{:?} sanity check failed: \
                     cached={:?} concrete={}",
                    opcode, expected, concrete,
                );
            }
            let reg_concrete = if stale {
                expected.expect("stale only set when expected is Some")
            } else {
                concrete
            };
            (cached, reg_concrete)
        } else {
            ctx.profiler().count_ops(opcode, crate::counters::OPS);
            ctx.profiler()
                .count_ops(opcode, crate::counters::RECORDED_OPS);
            let opref = ctx.record_op_with_descr(opcode, &[array_opref, index_opref], descr);
            ctx.set_opref_concrete(
                opref,
                majit_ir::Value::Float(f64::from_bits(concrete as u64)),
            );
            ctx.heapcache_getarrayitem_now_known(array_opref, index_opref, descr_index, opref);
            (opref, concrete)
        };
        self.set_float_reg(ctx, dst, Some(opref), Some(reg_concrete));
        TraceAction::Continue
    }

    // ── BC_GETARRAYITEM_GC_R ──
    //
    // Ref-result element read for a raw-pointer array (a
    // `pools[selected]`-shaped read).  Mirrors BC_GETARRAYITEM_GC_I
    // but loads a target-sized GC pointer and writes the ref bank. Unlike
    // the int arm there is NO all-constant fold: the array base is a
    // live state pointer and the result must stay a `GetarrayitemGcR`
    // op so the short preamble re-produces it each loop entry (the
    // whole point of replacing the residual `jit_sel_get_ref` call).
    // The `pools` array is immutable (`_immutable_fields_`), so a
    // re-read during the observer concrete replay returns the same
    // pointer — no `record_observed_*` queue is needed (none exists
    // for getarrayitem).
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_getarrayitem_gc_r_rid(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let (array_reg, index_reg, descr_idx, dst) = {
            let frame = self.frames.current_mut();
            let array_reg = frame.next_reg() as usize;
            let index_reg = frame.next_reg() as usize;
            let descr_idx = frame.next_u16() as usize;
            let dst = frame.next_reg() as usize;
            (array_reg, index_reg, descr_idx, dst)
        };
        let Some(descr) = self.dispatch_array_descr_ref(ctx, descr_idx) else {
            return TraceAction::Abort;
        };
        let (array_opref, array_addr) = self.read_ref_reg(ctx, array_reg);
        let (index_opref, index_value) = self.read_int_reg(ctx, index_reg);
        let descr_index = descr.index();
        let cached = ctx.heapcache_getarrayitem(array_opref, index_opref, descr_index);
        // blackhole.py bhimpl_getarrayitem_gc_r reads GCREF through
        // the CPU; both the array stride and pointer width are typed.
        let concrete = self
            .cpu
            .bh_getarrayitem_gc_r(
                majit_ir::GcRef(array_addr as usize),
                index_value,
                descr.as_array_descr().expect("GC ref array descriptor"),
            )
            .0 as i64;
        // `getarrayitem_gc_r_pure` records `GetarrayitemGcPureR`
        // (`rewrite_op_getarrayitem` `_pure`). An all-constant read
        // folds here, the same way `BC_GETARRAYITEM_GC_I_PURE` does:
        // the pure opcode is what licenses it, and the live pointer
        // is already in hand so `protect_speculative_array` is not
        // asked to classify a block that has no GC type header.
        let pure = bytecode == jitcode::insns::BC_GETARRAYITEM_GC_R_PURE;
        let opcode = if pure {
            OpCode::GetarrayitemGcPureR
        } else {
            OpCode::GetarrayitemGcR
        };
        if pure && array_opref.is_constant() && index_opref.is_constant() {
            ctx.profiler().count_ops(opcode, crate::counters::OPS);
            let opref = ctx.const_ref(concrete);
            self.set_ref_reg(ctx, dst, Some(opref), Some(concrete));
        } else {
            let (opref, reg_concrete) = if let Some(cached) = cached {
                ctx.profiler()
                    .count_ops(opcode, crate::pyjitpl::counters::HEAPCACHED_OPS);
                // `_do_getarrayitem_gc_any`'s `typ == 'r'` arm compares the
                // freshly executed load against the cached box's
                // `tobox.getref_base()`. Same structure as the int/float
                // arm above: a mismatch records a fallback op whose result
                // is discarded, asserts in debug, and still answers with
                // the stale cached box.
                let expected = match ctx.box_value(cached) {
                    Some(Value::Ref(majit_ir::GcRef(p))) => Some(p as i64),
                    _ => None,
                };
                let stale = matches!(expected, Some(exp) if exp != concrete);
                if stale {
                    ctx.heapcache_invalidate_caches_varargs(
                        opcode,
                        None,
                        &[array_opref, index_opref],
                    );
                    ctx.profiler()
                        .count_ops(opcode, crate::counters::RECORDED_OPS);
                    let _ = ctx.record_op_with_descr(opcode, &[array_opref, index_opref], descr);
                    debug_assert!(
                        false,
                        "GetarrayitemGcR sanity check failed: \
                     cached={expected:?} concrete={concrete}",
                    );
                }
                let reg_concrete = if stale {
                    expected.expect("stale only set when expected is Some")
                } else {
                    concrete
                };
                (cached, reg_concrete)
            } else {
                ctx.profiler().count_ops(opcode, crate::counters::OPS);
                ctx.profiler()
                    .count_ops(opcode, crate::counters::RECORDED_OPS);
                let opref = ctx.record_op_with_descr(opcode, &[array_opref, index_opref], descr);
                ctx.set_opref_concrete(opref, Value::Ref(majit_ir::GcRef(concrete as usize)));
                ctx.heapcache_getarrayitem_now_known(array_opref, index_opref, descr_index, opref);
                (opref, concrete)
            };
            self.set_ref_reg(ctx, dst, Some(opref), Some(reg_concrete));
        }
        TraceAction::Continue
    }

    // blackhole.py bhimpl_setarrayitem_gc_i (and `_r` / `_f`): record
    // SetarrayitemGc (a single op-kind whose descr carries the item
    // type) and write the element through the live array data ptr —
    // the store side effect is the actual write for this iteration
    // (mirrors BC_SETFIELD_GC / BC_RAW_STORE_I). Encoding
    // `riid`/`rird`/`rifd`: [array:u8][index:u8][value:u8][descr:u16].
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_setarrayitem_gc_i(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let (array_reg, index_reg, value_reg, descr_idx) = {
            let frame = self.frames.current_mut();
            let array_reg = frame.next_reg() as usize;
            let index_reg = frame.next_reg() as usize;
            let value_reg = frame.next_reg() as usize;
            let descr_idx = frame.next_u16() as usize;
            (array_reg, index_reg, value_reg, descr_idx)
        };
        let Some(descr) = self.dispatch_array_descr_ref(ctx, descr_idx) else {
            return TraceAction::Abort;
        };
        let Some((base_size, itemsize, _is_signed)) = self.dispatch_array_geometry(descr_idx)
        else {
            return TraceAction::Abort;
        };
        let (array_opref, array_addr) = self.read_ref_reg(ctx, array_reg);
        let (index_opref, index_value) = self.read_int_reg(ctx, index_reg);
        let (value_opref, value_concrete) = match bytecode {
            jitcode::insns::BC_SETARRAYITEM_GC_R => self.read_ref_reg(ctx, value_reg),
            jitcode::insns::BC_SETARRAYITEM_GC_F => self.read_float_reg(ctx, value_reg),
            _ => self.read_int_reg(ctx, value_reg),
        };
        let descr_index = descr.index();
        // `execute_setarrayitem_gc` (pyjitpl.py) records through
        // `execute_and_record` → `_record_helper` (pyjitpl.py),
        // which runs `heapcache.invalidate_caches` → `mark_escaped` →
        // `_escape_from_write` (heapcache.py) *before* the op is
        // appended: a ref written into an already-escaped array escapes
        // too, or `invalidate_unescaped` keeps its cached fields alive
        // across the next residual call while the callee can reach and
        // mutate it through the array.  `clear_caches_not_necessary`
        // (heapcache.py) lists SETARRAYITEM_GC, so only the
        // mark-escaped half runs.  Same shape as BC_RAW_STORE_I above.
        ctx.heapcache_invalidate_caches_varargs(
            OpCode::SetarrayitemGc,
            None,
            &[array_opref, index_opref, value_opref],
        );
        ctx.profiler()
            .count_ops(OpCode::SetarrayitemGc, crate::counters::OPS);
        ctx.profiler()
            .count_ops(OpCode::SetarrayitemGc, crate::counters::RECORDED_OPS);
        ctx.record_op_with_descr(
            OpCode::SetarrayitemGc,
            &[array_opref, index_opref, value_opref],
            descr,
        );
        // execute_setarrayitem_gc (pyjitpl.py): update the trace
        // heap cache after the store so a later getarrayitem of the same
        // (array, const index) reads the stored value, not a stale
        // cached element.  Key on descr.index() (the canonical resolved
        // descr index the getarrayitem read path uses), not the raw
        // bytecode operand descr_idx.
        ctx.heapcache_setarrayitem(array_opref, index_opref, descr_index, value_opref);
        if array_addr != 0 {
            let item_addr = (array_addr as usize)
                .wrapping_add(base_size)
                .wrapping_add((index_value as usize).wrapping_mul(itemsize));
            unsafe {
                match itemsize {
                    1 => *(item_addr as *mut u8) = value_concrete as u8,
                    2 => *(item_addr as *mut u16) = value_concrete as u16,
                    4 => *(item_addr as *mut u32) = value_concrete as u32,
                    8 => *(item_addr as *mut i64) = value_concrete,
                    other => {
                        panic!("BC_SETARRAYITEM_GC: unsupported itemsize {other}")
                    }
                }
            }
            // A ref store adds a heap edge array→value; notify the GC on
            // the container so a young value survives a minor collection
            // triggered later in the walk (mirrors bh_setarrayitem_gc_r).
            if bytecode == jitcode::insns::BC_SETARRAYITEM_GC_R
                && majit_gc::gc_owns_object(array_addr as usize)
            {
                majit_gc::gc_write_barrier(majit_ir::GcRef(array_addr as usize));
            }
        }
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_getarrayitem_vable_i(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let (opcode_pc, vable_reg, array_idx, index_reg, dest) = {
            let frame = self.frames.current_mut();
            let opcode_pc = frame.code_cursor - 1;
            let (vable_reg, array_idx, index_reg, dest) = frame.read_vable_getarrayitem();
            (opcode_pc, vable_reg, array_idx, index_reg, dest)
        };
        let Some((vable_opref, fdescr, adescr)) =
            self.vable_array_descrs(ctx, vable_reg, array_idx)
        else {
            return TraceAction::Abort;
        };
        let (index, index_value) = self.read_int_reg(ctx, index_reg);
        // pyjitpl.py `_opimpl_getarrayitem_vable` /
        // `_opimpl_setarrayitem_vable` reach the index through
        // `implement_guard_value` on an `MIFrame`, which owns the
        // framestack the resume snapshot is built from. Promoting here
        // rather than inside `TraceCtx::get_arrayitem_vable_index`
        // keeps that ownership: the guard gets a full-framestack,
        // vable-carrying snapshot instead of the minimal one
        // `TraceCtx::promote_int` can build without an `MIFrameStack`.
        //
        // The hoist moves the promote to the caller, NOT ahead of the
        // branch that selects it. `_get_arrayitem_vable_index`
        // (pyjitpl.py) opens with the promote and is entered only
        // from the standard leg (:1229, :1244); the non-standard leg
        // reaches its `getfield_gc_r` + `get|setarrayitem_gc_*` with
        // the index box as it stands. So the decision is taken first
        // and handed to the `*_checked` leg, and a non-standard access
        // with a non-constant index no longer mints a GUARD_VALUE that
        // over-specializes an ordinary heap read.
        let nonstandard = self.nonstandard_virtualizable(ctx, sym, opcode_pc, vable_opref, &fdescr);
        let index = if nonstandard {
            index
        } else {
            self.implement_guard_value(ctx, sym, index, index_value, opcode_pc)
        };
        let (opref, value) = ctx.vable_getarrayitem_int_checked(
            nonstandard,
            opcode_pc,
            vable_opref,
            index,
            index_value,
            fdescr,
            adescr,
        );
        self.set_int_reg(ctx, dest, Some(opref), value.map(value_as_int_bits));
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_getarrayitem_vable_r(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let (opcode_pc, vable_reg, array_idx, index_reg, dest) = {
            let frame = self.frames.current_mut();
            let opcode_pc = frame.code_cursor - 1;
            let (vable_reg, array_idx, index_reg, dest) = frame.read_vable_getarrayitem();
            (opcode_pc, vable_reg, array_idx, index_reg, dest)
        };
        if crate::vable_read_probe_enabled() {
            let depth = self.frames.len();
            let frame = self.frames.current_mut();
            eprintln!(
                "[vable-read-probe] reg jitcode={} depth={} vable_reg={} \
                 reg_opref={:?} reg_value={:?}",
                frame.jitcode.name,
                depth,
                vable_reg,
                frame.ref_regs[vable_reg],
                frame
                    .ref_regs
                    .get(vable_reg)
                    .copied()
                    .flatten()
                    .and_then(|op| ctx.box_bits(op))
                    .map(|v| format!("0x{v:x}")),
            );
        }
        let Some((vable_opref, fdescr, adescr)) =
            self.vable_array_descrs(ctx, vable_reg, array_idx)
        else {
            return TraceAction::Abort;
        };
        let (index, index_value) = self.read_int_reg(ctx, index_reg);
        // pyjitpl.py `_opimpl_getarrayitem_vable` /
        // `_opimpl_setarrayitem_vable` reach the index through
        // `implement_guard_value` on an `MIFrame`, which owns the
        // framestack the resume snapshot is built from. Promoting here
        // rather than inside `TraceCtx::get_arrayitem_vable_index`
        // keeps that ownership: the guard gets a full-framestack,
        // vable-carrying snapshot instead of the minimal one
        // `TraceCtx::promote_int` can build without an `MIFrameStack`.
        //
        // The hoist moves the promote to the caller, NOT ahead of the
        // branch that selects it. `_get_arrayitem_vable_index`
        // (pyjitpl.py) opens with the promote and is entered only
        // from the standard leg (:1229, :1244); the non-standard leg
        // reaches its `getfield_gc_r` + `get|setarrayitem_gc_*` with
        // the index box as it stands. So the decision is taken first
        // and handed to the `*_checked` leg, and a non-standard access
        // with a non-constant index no longer mints a GUARD_VALUE that
        // over-specializes an ordinary heap read.
        let nonstandard = self.nonstandard_virtualizable(ctx, sym, opcode_pc, vable_opref, &fdescr);
        let index = if nonstandard {
            index
        } else {
            self.implement_guard_value(ctx, sym, index, index_value, opcode_pc)
        };
        let (opref, value) = ctx.vable_getarrayitem_ref_checked(
            nonstandard,
            opcode_pc,
            vable_opref,
            index,
            index_value,
            fdescr,
            adescr,
        );
        self.set_ref_reg(ctx, dest, Some(opref), value.map(value_as_ref_bits));
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_getarrayitem_vable_f(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let (opcode_pc, vable_reg, array_idx, index_reg, dest) = {
            let frame = self.frames.current_mut();
            let opcode_pc = frame.code_cursor - 1;
            let (vable_reg, array_idx, index_reg, dest) = frame.read_vable_getarrayitem();
            (opcode_pc, vable_reg, array_idx, index_reg, dest)
        };
        let Some((vable_opref, fdescr, adescr)) =
            self.vable_array_descrs(ctx, vable_reg, array_idx)
        else {
            return TraceAction::Abort;
        };
        let (index, index_value) = self.read_int_reg(ctx, index_reg);
        // pyjitpl.py `_opimpl_getarrayitem_vable` /
        // `_opimpl_setarrayitem_vable` reach the index through
        // `implement_guard_value` on an `MIFrame`, which owns the
        // framestack the resume snapshot is built from. Promoting here
        // rather than inside `TraceCtx::get_arrayitem_vable_index`
        // keeps that ownership: the guard gets a full-framestack,
        // vable-carrying snapshot instead of the minimal one
        // `TraceCtx::promote_int` can build without an `MIFrameStack`.
        //
        // The hoist moves the promote to the caller, NOT ahead of the
        // branch that selects it. `_get_arrayitem_vable_index`
        // (pyjitpl.py) opens with the promote and is entered only
        // from the standard leg (:1229, :1244); the non-standard leg
        // reaches its `getfield_gc_r` + `get|setarrayitem_gc_*` with
        // the index box as it stands. So the decision is taken first
        // and handed to the `*_checked` leg, and a non-standard access
        // with a non-constant index no longer mints a GUARD_VALUE that
        // over-specializes an ordinary heap read.
        let nonstandard = self.nonstandard_virtualizable(ctx, sym, opcode_pc, vable_opref, &fdescr);
        let index = if nonstandard {
            index
        } else {
            self.implement_guard_value(ctx, sym, index, index_value, opcode_pc)
        };
        let (opref, value) = ctx.vable_getarrayitem_float_checked(
            nonstandard,
            opcode_pc,
            vable_opref,
            index,
            index_value,
            fdescr,
            adescr,
        );
        self.set_float_reg(ctx, dest, Some(opref), value.map(value_as_float_bits));
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_setarrayitem_vable_i(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let (opcode_pc, vable_reg, array_idx, index_reg, src) = {
            let frame = self.frames.current_mut();
            let opcode_pc = frame.code_cursor - 1;
            let (vable_reg, array_idx, index_reg, src) = frame.read_vable_setarrayitem();
            (opcode_pc, vable_reg, array_idx, index_reg, src)
        };
        let Some((vable_opref, fdescr, adescr)) =
            self.vable_array_descrs(ctx, vable_reg, array_idx)
        else {
            return TraceAction::Abort;
        };
        let (index, index_value) = self.read_int_reg(ctx, index_reg);
        // pyjitpl.py `_opimpl_getarrayitem_vable` /
        // `_opimpl_setarrayitem_vable` reach the index through
        // `implement_guard_value` on an `MIFrame`, which owns the
        // framestack the resume snapshot is built from. Promoting here
        // rather than inside `TraceCtx::get_arrayitem_vable_index`
        // keeps that ownership: the guard gets a full-framestack,
        // vable-carrying snapshot instead of the minimal one
        // `TraceCtx::promote_int` can build without an `MIFrameStack`.
        //
        // The hoist moves the promote to the caller, NOT ahead of the
        // branch that selects it. `_get_arrayitem_vable_index`
        // (pyjitpl.py) opens with the promote and is entered only
        // from the standard leg (:1229, :1244); the non-standard leg
        // reaches its `getfield_gc_r` + `get|setarrayitem_gc_*` with
        // the index box as it stands. So the decision is taken first
        // and handed to the `*_checked` leg, and a non-standard access
        // with a non-constant index no longer mints a GUARD_VALUE that
        // over-specializes an ordinary heap read.
        let nonstandard = self.nonstandard_virtualizable(ctx, sym, opcode_pc, vable_opref, &fdescr);
        let index = if nonstandard {
            index
        } else {
            self.implement_guard_value(ctx, sym, index, index_value, opcode_pc)
        };
        let (value, concrete) = self.read_int_reg(ctx, src);
        match ctx.vable_setarrayitem_checked(
            nonstandard,
            opcode_pc,
            vable_opref,
            index,
            index_value,
            fdescr,
            adescr,
            value,
            Value::Int(concrete),
            false,
        ) {
            // Promoted index falls outside the standard virtualizable
            // array (e.g. a transient out-of-bounds state-field index);
            // this slot cannot be virtualized, so abort the trace.
            VableArrayStore::OutOfVable => return TraceAction::Abort,
            VableArrayStore::Stored(_) => {}
        }
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_setarrayitem_vable_r(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let (opcode_pc, vable_reg, array_idx, index_reg, src) = {
            let frame = self.frames.current_mut();
            let opcode_pc = frame.code_cursor - 1;
            let (vable_reg, array_idx, index_reg, src) = frame.read_vable_setarrayitem();
            (opcode_pc, vable_reg, array_idx, index_reg, src)
        };
        let Some((vable_opref, fdescr, adescr)) =
            self.vable_array_descrs(ctx, vable_reg, array_idx)
        else {
            return TraceAction::Abort;
        };
        let (index, index_value) = self.read_int_reg(ctx, index_reg);
        // pyjitpl.py `_opimpl_getarrayitem_vable` /
        // `_opimpl_setarrayitem_vable` reach the index through
        // `implement_guard_value` on an `MIFrame`, which owns the
        // framestack the resume snapshot is built from. Promoting here
        // rather than inside `TraceCtx::get_arrayitem_vable_index`
        // keeps that ownership: the guard gets a full-framestack,
        // vable-carrying snapshot instead of the minimal one
        // `TraceCtx::promote_int` can build without an `MIFrameStack`.
        //
        // The hoist moves the promote to the caller, NOT ahead of the
        // branch that selects it. `_get_arrayitem_vable_index`
        // (pyjitpl.py) opens with the promote and is entered only
        // from the standard leg (:1229, :1244); the non-standard leg
        // reaches its `getfield_gc_r` + `get|setarrayitem_gc_*` with
        // the index box as it stands. So the decision is taken first
        // and handed to the `*_checked` leg, and a non-standard access
        // with a non-constant index no longer mints a GUARD_VALUE that
        // over-specializes an ordinary heap read.
        let nonstandard = self.nonstandard_virtualizable(ctx, sym, opcode_pc, vable_opref, &fdescr);
        let index = if nonstandard {
            index
        } else {
            self.implement_guard_value(ctx, sym, index, index_value, opcode_pc)
        };
        let (value, concrete) = self.read_ref_reg(ctx, src);
        match ctx.vable_setarrayitem_checked(
            nonstandard,
            opcode_pc,
            vable_opref,
            index,
            index_value,
            fdescr,
            adescr,
            value,
            Value::Ref(majit_ir::GcRef(concrete as usize)),
            false,
        ) {
            VableArrayStore::OutOfVable => return TraceAction::Abort,
            VableArrayStore::Stored(_) => {}
        }
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_setarrayitem_vable_f(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let (opcode_pc, vable_reg, array_idx, index_reg, src) = {
            let frame = self.frames.current_mut();
            let opcode_pc = frame.code_cursor - 1;
            let (vable_reg, array_idx, index_reg, src) = frame.read_vable_setarrayitem();
            (opcode_pc, vable_reg, array_idx, index_reg, src)
        };
        let Some((vable_opref, fdescr, adescr)) =
            self.vable_array_descrs(ctx, vable_reg, array_idx)
        else {
            return TraceAction::Abort;
        };
        let (index, index_value) = self.read_int_reg(ctx, index_reg);
        // pyjitpl.py `_opimpl_getarrayitem_vable` /
        // `_opimpl_setarrayitem_vable` reach the index through
        // `implement_guard_value` on an `MIFrame`, which owns the
        // framestack the resume snapshot is built from. Promoting here
        // rather than inside `TraceCtx::get_arrayitem_vable_index`
        // keeps that ownership: the guard gets a full-framestack,
        // vable-carrying snapshot instead of the minimal one
        // `TraceCtx::promote_int` can build without an `MIFrameStack`.
        //
        // The hoist moves the promote to the caller, NOT ahead of the
        // branch that selects it. `_get_arrayitem_vable_index`
        // (pyjitpl.py) opens with the promote and is entered only
        // from the standard leg (:1229, :1244); the non-standard leg
        // reaches its `getfield_gc_r` + `get|setarrayitem_gc_*` with
        // the index box as it stands. So the decision is taken first
        // and handed to the `*_checked` leg, and a non-standard access
        // with a non-constant index no longer mints a GUARD_VALUE that
        // over-specializes an ordinary heap read.
        let nonstandard = self.nonstandard_virtualizable(ctx, sym, opcode_pc, vable_opref, &fdescr);
        let index = if nonstandard {
            index
        } else {
            self.implement_guard_value(ctx, sym, index, index_value, opcode_pc)
        };
        let (value, concrete) = self.read_float_reg(ctx, src);
        match ctx.vable_setarrayitem_checked(
            nonstandard,
            opcode_pc,
            vable_opref,
            index,
            index_value,
            fdescr,
            adescr,
            value,
            Value::Float(f64::from_bits(concrete as u64)),
            false,
        ) {
            VableArrayStore::OutOfVable => return TraceAction::Abort,
            VableArrayStore::Stored(_) => {}
        }
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_arraylen_vable(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let (opcode_pc, vable_reg, array_idx, dest) = {
            let frame = self.frames.current_mut();
            let opcode_pc = frame.code_cursor - 1;
            let (vable_reg, array_idx, dest) = frame.read_vable_arraylen();
            (opcode_pc, vable_reg, array_idx, dest)
        };
        let Some((vable_opref, fdescr, adescr)) =
            self.vable_array_descrs(ctx, vable_reg, array_idx)
        else {
            return TraceAction::Abort;
        };
        let vable_struct_ptr = self.read_ref_reg(ctx, vable_reg).1;
        let nonstandard = self.nonstandard_virtualizable(ctx, sym, opcode_pc, vable_opref, &fdescr);
        let result = ctx.vable_arraylen_vable_checked(
            nonstandard,
            self.cpu.as_ref(),
            vable_opref,
            vable_struct_ptr,
            fdescr,
            adescr,
        );
        // pyjitpl.py MIFrame.opimpl_arraylen_vable `result =
        // vinfo.get_array_length(virtualizable, arrayindex);
        // return ConstInt(result)`.  RPython reads from the live
        // struct; pyre's trace-side shadow is
        // `virtualizable_array_lengths`, populated by
        // `init_virtualizable_boxes` (resume.py:471-486 parity)  allow-line-citation
        // before the trace runs, so it carries the same length
        // RPython would dereference.
        let len = ctx
            .virtualizable_array_lengths()
            .and_then(|lengths| lengths.get(array_idx).copied())
            .unwrap_or(0);
        self.set_int_reg(ctx, dest, Some(result), Some(len as i64));
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_arraybase_vable(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        // Same `rdd>i` operand triple as `arraylen_vable` above, hence
        // the shared decoder.
        let (vable_reg, array_idx, dest) = {
            let frame = self.frames.current_mut();
            frame.read_vable_arraylen()
        };
        let Some((_vable_opref, fdescr, _adescr)) =
            self.vable_array_descrs(ctx, vable_reg, array_idx)
        else {
            return TraceAction::Abort;
        };
        let vable_struct_ptr = self.read_ref_reg(ctx, vable_reg).1;
        // An unresolvable base aborts rather than defaulting: the walk
        // really executes the residual call this address feeds, so a
        // placeholder would be handed to a live callee.
        let Some((result, addr)) = ctx.vable_arraybase_vable(vable_struct_ptr, fdescr) else {
            return TraceAction::Abort;
        };
        self.set_int_reg(ctx, dest, Some(result), Some(addr));
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_hint_force_virtualizable(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let vable_reg = self.frames.current_mut().next_reg() as usize;
        let vable_opref = self.resolve_vable_box(vable_reg);
        ctx.gen_store_back_in_vable(vable_opref);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_int_add(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_binop_i(ctx, OpCode::IntAdd);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_int_sub(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_binop_i(ctx, OpCode::IntSub);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_int_mul(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_binop_i(ctx, OpCode::IntMul);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_int_add_jump_if_ovf(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_int_binop_jump_if_ovf(ctx, sym, OpCode::IntAddOvf);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_int_sub_jump_if_ovf(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_int_binop_jump_if_ovf(ctx, sym, OpCode::IntSubOvf);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_int_mul_jump_if_ovf(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_int_binop_jump_if_ovf(ctx, sym, OpCode::IntMulOvf);
        TraceAction::Continue
    }

    // `int_floordiv` / `int_mod` have no bytecode opcode:
    // `jtransform.py Transformer._do_builtin_call` rewrites both
    // (`rewrite_op_int_floordiv` / `rewrite_op_int_mod`) to
    // `direct_call(ll_int_py_div)` / `direct_call(ll_int_py_mod)`
    // before jitcode emission.
    // Pyre's `specialize.rs::walker_emit_int_py_div_or_mod` emits
    // the same residual call as a `CallI` op directly — no
    // `BC_INT_FLOORDIV` / `BC_INT_MOD` opcode is allocated, so
    // no dispatch arm exists.
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_int_and(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_binop_i(ctx, OpCode::IntAnd);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_int_signext(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_binop_i(ctx, OpCode::IntSignext);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_int_or(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_binop_i(ctx, OpCode::IntOr);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_int_xor(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_binop_i(ctx, OpCode::IntXor);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_int_lshift(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_binop_i(ctx, OpCode::IntLshift);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_int_rshift(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_binop_i(ctx, OpCode::IntRshift);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_int_eq(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_binop_i(ctx, OpCode::IntEq);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_int_ne(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_binop_i(ctx, OpCode::IntNe);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_int_lt(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_binop_i(ctx, OpCode::IntLt);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_int_le(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_binop_i(ctx, OpCode::IntLe);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_int_gt(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_binop_i(ctx, OpCode::IntGt);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_int_ge(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_binop_i(ctx, OpCode::IntGe);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_uint_rshift(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_binop_i(ctx, OpCode::UintRshift);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_uint_mul_high(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_binop_i(ctx, OpCode::UintMulHigh);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_uint_lt(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_binop_i(ctx, OpCode::UintLt);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_uint_le(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_binop_i(ctx, OpCode::UintLe);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_uint_gt(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_binop_i(ctx, OpCode::UintGt);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_uint_ge(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_binop_i(ctx, OpCode::UintGe);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_int_between(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_int_between(ctx);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_int_neg(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_unary_i(ctx, OpCode::IntNeg);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_int_invert(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_unary_i(ctx, OpCode::IntInvert);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_int_is_true(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_unary_i(ctx, OpCode::IntIsTrue);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_int_is_zero(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_unary_i(ctx, OpCode::IntIsZero);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_ptr_eq(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_binop_r_to_i(ctx, OpCode::PtrEq);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_ptr_ne(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_binop_r_to_i(ctx, OpCode::PtrNe);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_instance_ptr_eq(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_binop_r_to_i(ctx, OpCode::InstancePtrEq);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_instance_ptr_ne(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_binop_r_to_i(ctx, OpCode::InstancePtrNe);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_ptr_iszero(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_ptr_nullity(ctx, false);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_ptr_nonzero(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_ptr_nullity(ctx, true);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_goto_if_not(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        // Canonical `iL` encoding (`assembler.py write_insn`):
        // [cond:u8][target:u16].
        let (opcode_pc, cond_idx, target) = {
            let frame = self.frames.current_mut();
            // RPython `pyjitpl.py orgpc = position` parity: the
            // dispatcher's `next_u8()` already stepped past the
            // opcode byte, so `code_cursor - 1` is the byte position
            // of the guard op itself — what `generate_guard(...,
            // resumepc=orgpc)` records. The `live/<offset>` marker
            // sits at `opcode_pc - SIZE_LIVE_OP`, satisfying
            // BlackholeInterpreter::get_current_position_info's
            // `code[pc - SIZE_LIVE_OP] == op_live` check.
            let opcode_pc = frame.code_cursor - 1;
            (
                opcode_pc,
                frame.next_reg() as usize,
                frame.next_u16() as usize,
            )
        };
        let (cond, cond_value) = self.read_int_reg(ctx, cond_idx);
        self.pcseq_branch(
            "goto_if_not",
            opcode_pc,
            cond_value,
            (cond_value == 0).then_some(target),
        );
        self.goto_if_not(ctx, sym, opcode_pc, cond, cond_value, target, true);
        TraceAction::Continue
    }

    // pyjitpl.py opimpl_goto_if_not_int_is_true(box, target):
    //   condbox = self.execute(rop.INT_IS_TRUE, box)
    //   self.opimpl_goto_if_not(condbox, target, ..., replace=False)
    //
    // `jtransform.py optimize_goto_if_not` admits `int_is_true` into
    // the folded-exitswitch set and `flatten.py` then emits
    // `goto_if_not_int_is_true`, so this is a distinct opname with its
    // own byte — only `blackhole.py bhimpl_goto_if_not_int_is_true` aliases the two, and only on
    // the blackhole side, where there is no operation to re-record.
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_goto_if_not_int_is_true(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        // Canonical `iL` encoding: [src:u8][target:u16].
        let (opcode_pc, src_idx, target) = {
            let frame = self.frames.current_mut();
            let opcode_pc = frame.code_cursor - 1;
            (
                opcode_pc,
                frame.next_reg() as usize,
                frame.next_u16() as usize,
            )
        };
        let (src, src_value) = self.read_int_reg(ctx, src_idx);
        let cond_value = (src_value != 0) as i64;
        let cond = ctx.execute_and_record(
            Some(self.cpu.as_ref()),
            OpCode::IntIsTrue,
            None,
            &[src],
            Some(majit_ir::Value::Int(cond_value)),
            self.last_exception_value,
        );
        self.goto_if_not(ctx, sym, opcode_pc, cond, cond_value, target, false);
        TraceAction::Continue
    }

    // pyjitpl.py opimpl_goto_if_not_int_is_zero(box, target):
    //   condbox = execute(rop.INT_IS_ZERO, box)
    //   self.opimpl_goto_if_not(condbox, target, ..., replace=False)
    // i.e. record int_is_zero on the operand, then branch as if the
    // result were a plain bool exitswitch.
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_goto_if_not_int_is_zero(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        // Canonical `iL` encoding: [src:u8][target:u16].
        let (opcode_pc, src_idx, target) = {
            let frame = self.frames.current_mut();
            let opcode_pc = frame.code_cursor - 1;
            (
                opcode_pc,
                frame.next_reg() as usize,
                frame.next_u16() as usize,
            )
        };
        let (src, src_value) = self.read_int_reg(ctx, src_idx);
        let cond_value = if src_value == 0 { 1 } else { 0 };
        let cond = ctx.execute_and_record(
            Some(self.cpu.as_ref()),
            OpCode::IntIsZero,
            None,
            &[src],
            Some(majit_ir::Value::Int(cond_value)),
            self.last_exception_value,
        );
        let guard = if cond_value == 0 {
            OpCode::GuardFalse
        } else {
            OpCode::GuardTrue
        };
        self.record_state_guard(ctx, sym, guard, &[cond], opcode_pc, false);
        if cond_value == 0 {
            self.frames.current_mut().code_cursor = target;
        }
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_goto_if_not_int_lt(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        // Canonical `iiL` encoding: [a:u8][b:u8][target:u16].
        let (opcode_pc, lhs_idx, rhs_idx, target) = {
            let frame = self.frames.current_mut();
            let opcode_pc = frame.code_cursor - 1;
            (
                opcode_pc,
                frame.next_reg() as usize,
                frame.next_reg() as usize,
                frame.next_u16() as usize,
            )
        };
        let (lhs, lhs_value) = self.read_int_reg(ctx, lhs_idx);
        let (rhs, rhs_value) = self.read_int_reg(ctx, rhs_idx);
        let opcode = match bytecode {
            jitcode::insns::BC_GOTO_IF_NOT_INT_LT => OpCode::IntLt,
            jitcode::insns::BC_GOTO_IF_NOT_INT_LE => OpCode::IntLe,
            jitcode::insns::BC_GOTO_IF_NOT_INT_EQ => OpCode::IntEq,
            jitcode::insns::BC_GOTO_IF_NOT_INT_NE => OpCode::IntNe,
            jitcode::insns::BC_GOTO_IF_NOT_INT_GT => OpCode::IntGt,
            jitcode::insns::BC_GOTO_IF_NOT_INT_GE => OpCode::IntGe,
            _ => unreachable!(),
        };
        let cond_value = eval_binop_i(opcode, lhs_value, rhs_value);
        self.record_or_fold_fused_guard(
            ctx,
            sym,
            opcode,
            lhs,
            rhs,
            cond_value != 0,
            opcode_pc,
            target,
        );
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_goto_if_not_float_lt(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        // Canonical `ffL` encoding: [a:u8][b:u8][target:u16].
        let (opcode_pc, lhs_idx, rhs_idx, target) = {
            let frame = self.frames.current_mut();
            let opcode_pc = frame.code_cursor - 1;
            (
                opcode_pc,
                frame.next_reg() as usize,
                frame.next_reg() as usize,
                frame.next_u16() as usize,
            )
        };
        let (lhs, lhs_value) = self.read_float_reg(ctx, lhs_idx);
        let (rhs, rhs_value) = self.read_float_reg(ctx, rhs_idx);
        let a = f64::from_bits(lhs_value as u64);
        let b = f64::from_bits(rhs_value as u64);
        let (opcode, taken) = match bytecode {
            jitcode::insns::BC_GOTO_IF_NOT_FLOAT_LT => (OpCode::FloatLt, a < b),
            jitcode::insns::BC_GOTO_IF_NOT_FLOAT_LE => (OpCode::FloatLe, a <= b),
            jitcode::insns::BC_GOTO_IF_NOT_FLOAT_EQ => (OpCode::FloatEq, a == b),
            jitcode::insns::BC_GOTO_IF_NOT_FLOAT_NE => (OpCode::FloatNe, a != b),
            jitcode::insns::BC_GOTO_IF_NOT_FLOAT_GT => (OpCode::FloatGt, a > b),
            jitcode::insns::BC_GOTO_IF_NOT_FLOAT_GE => (OpCode::FloatGe, a >= b),
            _ => unreachable!(),
        };
        self.record_or_fold_fused_guard(ctx, sym, opcode, lhs, rhs, taken, opcode_pc, target);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_goto_if_not_ptr_eq(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        // Canonical `rrL` encoding: [a:u8][b:u8][target:u16].
        let (opcode_pc, lhs_idx, rhs_idx, target) = {
            let frame = self.frames.current_mut();
            let opcode_pc = frame.code_cursor - 1;
            (
                opcode_pc,
                frame.next_reg() as usize,
                frame.next_reg() as usize,
                frame.next_u16() as usize,
            )
        };
        let (lhs, lhs_value) = self.read_ref_reg(ctx, lhs_idx);
        let (rhs, rhs_value) = self.read_ref_reg(ctx, rhs_idx);
        let (opcode, taken) = match bytecode {
            jitcode::insns::BC_GOTO_IF_NOT_PTR_EQ => (OpCode::PtrEq, lhs_value == rhs_value),
            jitcode::insns::BC_GOTO_IF_NOT_PTR_NE => (OpCode::PtrNe, lhs_value != rhs_value),
            _ => unreachable!(),
        };
        self.record_or_fold_fused_guard(ctx, sym, opcode, lhs, rhs, taken, opcode_pc, target);
        TraceAction::Continue
    }

    // RPython `pyjitpl.py opimpl_switch`: a hit promotes the
    // switched value with GUARD_VALUE and jumps to the case target;
    // a miss records INT_EQ + GUARD_FALSE for every ordered key and
    // falls through to the default path after the switch.
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_switch(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        // Canonical `id` encoding (`assembler.py write_insn`):
        // [value:u8][descr:u16].
        let (opcode_pc, value_idx, descr_idx) = {
            let frame = self.frames.current_mut();
            let opcode_pc = frame.code_cursor - 1;
            (
                opcode_pc,
                frame.next_reg() as usize,
                frame.next_u16() as usize,
            )
        };
        let descr = self
            .frames
            .current_mut()
            .jitcode
            .descr_at(descr_idx)
            .and_then(crate::jitcode::RuntimeBhDescr::as_bh_descr)
            .unwrap_or_else(|| panic!("BC_SWITCH descrs[{descr_idx}] is not a BhDescr"))
            .clone();
        let (value_box, concrete_value) = self.read_int_reg(ctx, value_idx);
        let hit = descr.switch_lookup(concrete_value);
        self.pcseq_branch("switch", opcode_pc, concrete_value, hit);
        if let Some(target) = hit {
            let const_ref = ctx.const_int(concrete_value);
            self.record_state_guard(
                ctx,
                sym,
                OpCode::GuardValue,
                &[value_box, const_ref],
                opcode_pc,
                false,
            );
            self.set_int_reg(ctx, value_idx, Some(const_ref), Some(concrete_value));
            self.frames.current_mut().code_cursor = target;
        } else {
            for &key in descr.switch_const_keys_in_order() {
                let key_ref = ctx.const_int(key);
                // `SwitchDictDescr.attach` builds `const_keys_in_order`
                // as `sorted(dict.keys())`, so a `switch_lookup` miss
                // means no key equals the switched value. Evaluate the
                // comparison rather than assume that: the funnel folds
                // this value into the trace when `value_box` is
                // constant, and a constant that disagrees with the
                // guard beside it is a miscompile with no diagnostic.
                let cond_value = (concrete_value == key) as i64;
                debug_assert_eq!(
                    cond_value, 0,
                    "BC_SWITCH miss chain: key {key} equals the switched value",
                );
                let cond = ctx.execute_and_record(
                    Some(self.cpu.as_ref()),
                    OpCode::IntEq,
                    None,
                    &[value_box, key_ref],
                    Some(majit_ir::Value::Int(cond_value)),
                    self.last_exception_value,
                );
                self.record_state_guard(ctx, sym, OpCode::GuardFalse, &[cond], opcode_pc, false);
            }
        }
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_goto_if_not_ptr_iszero(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        // Canonical `rL` encoding: [src:u8][target:u16].
        let (opcode_pc, src_idx, target) = {
            let frame = self.frames.current_mut();
            let opcode_pc = frame.code_cursor - 1;
            (
                opcode_pc,
                frame.next_reg() as usize,
                frame.next_u16() as usize,
            )
        };
        let (src, src_value) = self.read_ref_reg(ctx, src_idx);
        let nonnull = self.establish_nullity(ctx, sym, src, src_value, opcode_pc);
        // pyjitpl.py:
        //   opimpl_goto_if_not_ptr_nonzero: if not nonnull: self.pc = target
        //   opimpl_goto_if_not_ptr_iszero:  if     nonnull: self.pc = target
        let branch_taken = match bytecode {
            jitcode::insns::BC_GOTO_IF_NOT_PTR_ISZERO => nonnull,
            jitcode::insns::BC_GOTO_IF_NOT_PTR_NONZERO => !nonnull,
            _ => unreachable!(),
        };
        if branch_taken {
            self.frames.current_mut().code_cursor = target;
        }
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_catch_exception(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let _target = self.frames.current_mut().next_u16();
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_last_exception(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let dst = self.frames.current_mut().next_reg() as usize;
        let exc_value = self.last_exception_value;
        // pyjitpl.py opimpl_last_exception:
        //     exc_value = self.metainterp.last_exc_value
        //     assert exc_value
        //     assert self.metainterp.class_of_last_exc_is_const
        //     exc_cls = rclass.ll_cast_to_object(exc_value).typeptr
        //     return ConstInt(ptr2int(exc_cls))
        assert!(exc_value != 0, "last_exception without active exception");
        assert!(
            self.class_of_last_exc_is_const,
            "last_exception requires class_of_last_exc_is_const",
        );
        // `cls_of_box` (model.py) supplies the typeptr
        // resolution wired through `MetaInterp::cls_of_box`; the
        // standalone fallback returns the raw value for tests
        // that pre-date typed exception dispatch.
        let typeptr = self.read_typeptr_from_exception(exc_value);
        let opref = ctx.const_int(typeptr);
        self.set_int_reg(ctx, dst, Some(opref), Some(typeptr));
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_last_exc_value(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let dst = self.frames.current_mut().next_reg() as usize;
        // pyjitpl.py opimpl_last_exc_value:
        //     exc_value = self.metainterp.last_exc_value
        //     assert exc_value
        //     return self.metainterp.last_exc_box
        //
        // The value-null check gates the box read because
        // `clear_exception` (Parity #10) clears only
        // `last_exception_value`; `last_exception_box` may
        // remain populated with stale data after a successful
        // residual call.  PyPy's `assert exc_value` makes the
        // stale-box read fail-fast.
        let value = self.last_exception_value;
        assert!(value != 0, "last_exc_value without active exception");
        let opref = self
            .last_exception_box
            .expect("last_exc_value without exception box");
        self.set_ref_reg(ctx, dst, Some(opref), Some(value));
        TraceAction::Continue
    }

    // pyjitpl.py opimpl_goto_if_exception_mismatch:
    //     last_exc_value = metainterp.last_exc_value
    //     assert last_exc_value
    //     assert metainterp.class_of_last_exc_is_const
    //     cls = ... vtablebox.getaddr() ...
    //     real_instance = rclass.ll_cast_to_object(last_exc_value)
    //     if not rclass.ll_isinstance(real_instance, cls):
    //         self.pc = next_exc_target
    //
    // `class_of_last_exc_is_const` is asserted, so the typeptr is
    // constant for the trace — no guard recorded; the branch is
    // a trace-time decision (the vtable register box is a Const).
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_goto_if_exception_mismatch(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        // Canonical `iL` encoding: [vtable:u8][target:u16].
        let (vtable_idx, target) = {
            let frame = self.frames.current_mut();
            (frame.next_reg() as usize, frame.next_u16() as usize)
        };
        let exc_value = self.last_exception_value;
        assert!(
            exc_value != 0,
            "goto_if_exception_mismatch without active exception",
        );
        assert!(
            self.class_of_last_exc_is_const,
            "goto_if_exception_mismatch requires class_of_last_exc_is_const",
        );
        let (_, bounding_vtable) = self.read_int_reg(ctx, vtable_idx);
        // pyjitpl.py MIFrame.opimpl_goto_if_exception_mismatch:
        //     real_instance = rclass.ll_cast_to_object(last_exc_value)
        //     if not rclass.ll_isinstance(real_instance, cls):
        //         self.pc = next_exc_target
        //
        // `cls_of_box` (model.py) reads the runtime
        // typeptr; `issubclass_of` mirrors the blackhole-side
        // resolution (`handler_goto_if_exception_mismatch` in
        // `blackhole.rs` calls `cpu.bh_issubclass`) over RPython
        // subclass ranges.
        let exc_typeptr = self.read_typeptr_from_exception(exc_value);
        if !self.issubclass_of(exc_typeptr, bounding_vtable) {
            self.frames.current_mut().code_cursor = target;
        }
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_rvmprof_code(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let (leaving_idx, unique_id_idx) = {
            let frame = self.frames.current_mut();
            (frame.next_reg() as usize, frame.next_reg() as usize)
        };
        let leaving = self
            .frames
            .current_mut()
            .getint(ctx, leaving_idx)
            .unwrap_or(0);
        let unique_id = self
            .frames
            .current_mut()
            .getint(ctx, unique_id_idx)
            .unwrap_or(0);
        majit_rlib::rvmprof::cintf::jit_rvmprof_code(leaving, unique_id);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_jit_merge_point(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        // `pyjitpl.py reached_loop_header` builds `live_arg_boxes` from the
        // live framestack. Snapshot portal reds here so a walk that then
        // drains its frames still has jump args and writeback values.
        // Drop any earlier header's JUMP list so CloseLoop cannot consume
        // a stale `close_jump_boxes` from a previous visit.
        ctx.close_jump_boxes = None;
        self.stash_portal_reds(ctx, sym);
        // blackhole.py bhimpl_jit_merge_point parity.
        // Portal merge point: close the loop if at the traced header.
        //
        // Payload shape mirrors upstream `@arguments("self", "i",
        // "I", "R", "F", "I", "R", "F")` (blackhole.py) and
        // pyre's own `JitCodeBuilder::jit_merge_point`
        // (`majit-metainterp/src/jitcode/assembler.rs`)
        // — 1-byte jdindex (either a registers_i pool slot for the
        // `i` form or a raw signed byte for the `c` form) + six
        // typed register lists (`[len:u8][reg:u8 * N]`).
        let opcode = bytecode;
        let frame = self.frames.current_mut();
        // `jtransform.py` `promote_greens` emits a `-live-` (op3) immediately
        // BEFORE the `jit_merge_point` op; the GUARD_FUTURE_CONDITION
        // recorded at loop close resumes through it
        // (`JitCodeBuilder::live_placeholder` in
        // `jitcode/assembler.rs`).  Capture the
        // merge-point op position now, before `next_u8` advances the
        // cursor, so `record_state_guard`'s `frame.pc = resume_pc`
        // swap finds that `-live-` at
        // `mp_opcode_pc - SIZE_LIVE_OP`.
        let mp_opcode_pc = frame.code_cursor - 1;
        let jdindex_byte = frame.next_reg();
        // RPython `blackhole.py BlackholeInterpBuilder._get_method` argcode discrimination:
        //
        //     if argcode == 'i':
        //         value = self.registers_i[ord(code[position])]
        //     elif argcode == 'c':
        //         value = signedord(code[position])
        //
        // BC_JIT_MERGE_POINT is the `i` form: the byte indexes
        // `registers_i` (which carries the constants suffix at
        // `[num_regs_i, num_regs_i + constants_i.len())`, populated
        // by `MIFrame::setup_call`).  BC_JIT_MERGE_POINT_C
        // is the `c` form: the byte IS the signed jdindex.
        let jdindex: usize = if opcode == jitcode::insns::BC_JIT_MERGE_POINT_C {
            // signedord(byte) — byte interpreted as i8 then sign-extended.
            (jdindex_byte as i8) as i64 as usize
        } else {
            let slot = jdindex_byte as usize;
            let resolved = frame.getint(ctx, slot).expect(
                "BC_JIT_MERGE_POINT (i form): jdindex register slot \
                 must hold a populated int constant — assembler.py:312-346 \
                 emits an `i` argcode pointing at the post-regs constants \
                 suffix, finalize_constants seeds those slots at setup_call",
            );
            resolved as usize
        };
        // pyjitpl.py MIFrame.opimpl_jit_merge_point — `staticdata.jitdrivers_sd[jdindex]`
        // selects the JitDriver this merge point belongs to.
        // `codegen_state.rs`'s `generate_state_fields_jit_state`
        // stamps `driver.index()` into
        // this byte at codegen time, so the value must always
        // resolve to a registered slot — anything else
        // indicates a `register_jitdriver_sd` lifecycle bug
        // (warmspot.py WarmRunnerDesc.make_args_specification translation-time
        // `make_args_specification` invariant parity).
        // Production-active assert; replaces an earlier
        // single-driver `== 0` over-restriction that would
        // wrongly fire when more than one JitDriver registers.
        let registered_drivers = ctx.metainterp_sd().jitdrivers_sd.len();
        assert!(
            jdindex < registered_drivers,
            "BC_JIT_MERGE_POINT: jdindex {jdindex} out of range \
             (registered drivers: {registered_drivers}) — \
             pyjitpl.py:1540 staticdata.jitdrivers_sd[jdindex] parity",
        );
        // `pyjitpl.py MIFrame.opimpl_jit_merge_point`:
        // `jitdriver_sd = staticdata.jitdrivers_sd[jdindex]`.
        let no_loop_header_jd = ctx.metainterp_sd().jitdrivers_sd[jdindex].no_loop_header;
        // 5 — register-byte bounds.  Each of the six
        // register lists encodes `[len:u8][reg:u8 * N]` with
        // greens/reds split by kind in `(I, R, F, I, R, F)`
        // order (mirrors `bhimpl_jit_merge_point`'s
        // `@arguments("self", "i", "I", "R", "F", "I", "R",
        // "F")`, blackhole.py).  Each register byte must
        // fall within the JitCode's per-kind register bank.
        let max_regs = [
            frame.jitcode.num_regs_i(),
            frame.jitcode.num_regs_r(),
            frame.jitcode.num_regs_f(),
            frame.jitcode.num_regs_i(),
            frame.jitcode.num_regs_r(),
            frame.jitcode.num_regs_f(),
        ];
        // Slice (audit Issue #4) — verify_green_args
        // (pyjitpl.py).  Slots 0..3 hold the green
        // register bytes; each green register MUST hold a
        // Const at trace time (the `emit_promote_greens` /
        // `<kind>_guard_value` chain at `jtransform.py` `promote_greens`
        // promotes each green to a constant before the
        // `BC_JIT_MERGE_POINT`).  A non-constant green here
        // indicates a macro emission gap.  RPython
        // `assert` ↔ Rust `debug_assert!` parity.
        // pyjitpl.py reached_loop_header / same_greenkey: the
        // merge point's pc green identifies the loop header.  Capture it
        // (the first int green — greens are declared pc-first) so the
        // close gate can decline a merge point whose pc differs from the
        // trace-start `header_pc`.  The promoted greens are constants at
        // trace time (verify_green_args, asserted below).
        let mut mp_green_pc: Option<i64> = None;
        // MAJIT_PCSEQ diagnostic: all int-green constants at this merge
        // point (pc plus any scalar greens the consumer declares).
        let mut mp_green_ints: CallI64s = SmallVec::new();
        // pyjitpl.py same_greenkey compares EVERY green, not just
        // the int slot.  Capture the ref (slot 1) and float (slot 2)
        // green constants too so the header-close gate compares the full
        // green tuple against the captured header greens (`header_greens`).
        let mut mp_green_refs: CallI64s = SmallVec::new();
        let mut mp_green_floats: CallI64s = SmallVec::new();
        // Single-pass tracing: the walk closes back to an interpreter
        // program pc; capture it (below) so the merge-point hook can
        // resume the native loop there. The walk is the sole executor,
        // so the close always transfers walk-final state.
        let capture_walk_reds = true;
        let inner_close = true;
        // pyjitpl.py reached_loop_header `current_merge_points.append`:
        // the live arg boxes (greens slots 0..3 + reds slots 3..6, in
        // operand order) at this merge point. A NESTED inner-loop revisit
        // records these as the inner loop's inputargs so the cross-loop
        // cut can peel the outer prefix as preamble. Built during the
        // tracing walk only — off the compiled hot path.
        let mut live_arg_boxes: SmallVec<[crate::trace_ctx::GreenBox; CALL_INLINE]> =
            SmallVec::new();
        let mut put_back_reds = false;
        // pyjitpl.py opimpl_jit_merge_point `redboxes`.
        let mut redboxes: SmallVec<[(OpRef, majit_ir::Type); CALL_INLINE]> = SmallVec::new();
        // Single-pass: accumulate the walk-final concrete RED values from
        // the live value-bank shadow (slots 3-5 = reds I/R/F in operand
        // order) so the merge-point hook can `restore_values` them into
        // native state — completing the transfer that storage-only
        // `recover` cannot (loop-carried reds never written to the heap).
        let mut walk_reds: CallValues = SmallVec::new();
        let mut green_i_regs: Vec<u8> = Vec::new();
        let mut green_r_regs: Vec<u8> = Vec::new();
        let mut green_f_regs: Vec<u8> = Vec::new();
        let mut red_i_regs: Vec<u8> = Vec::new();
        let mut red_r_regs: Vec<u8> = Vec::new();
        let mut red_f_regs: Vec<u8> = Vec::new();
        for (slot, &max) in max_regs.iter().enumerate().take(6) {
            let count = frame.next_u8() as usize;
            let is_green_slot = slot < 3;
            for _ in 0..count {
                let reg = frame.next_reg();
                let reg_idx = reg as usize;
                if is_green_slot {
                    match slot {
                        0 => green_i_regs.push(reg),
                        1 => green_r_regs.push(reg),
                        2 => green_f_regs.push(reg),
                        _ => {}
                    }
                } else {
                    match slot {
                        3 => red_i_regs.push(reg),
                        4 => red_r_regs.push(reg),
                        5 => red_f_regs.push(reg),
                        _ => {}
                    }
                }
                if capture_walk_reds && slot >= 3 {
                    match slot {
                        3 => {
                            if let Some(v) = frame.getint(ctx, reg_idx) {
                                walk_reds.push(Value::Int(v));
                            }
                        }
                        4 => {
                            if let Some(r) = frame.getref_base(ctx, reg_idx) {
                                walk_reds.push(Value::Ref(majit_ir::GcRef(r as usize)));
                            }
                        }
                        _ => {
                            if let Some(b) = frame.getfloat_storage(ctx, reg_idx) {
                                walk_reds.push(Value::Float(f64::from_bits(b as u64)));
                            }
                        }
                    }
                }
                if inner_close && !is_green_slot {
                    // `prepare_list_of_boxes`: every listed red register is
                    // copied. Greens stay in the green-constant capture
                    // below (`verify_green_args`); they are not redboxes.
                    // A listed red that is empty is a producer bug
                    // (`handle_jit_marker__jit_merge_point` listed a live
                    // Variable).
                    let (opref_opt, ty) = match slot {
                        3 => (
                            frame.int_regs.get(reg_idx).copied().flatten(),
                            majit_ir::Type::Int,
                        ),
                        4 => (
                            frame.ref_regs.get(reg_idx).copied().flatten(),
                            majit_ir::Type::Ref,
                        ),
                        _ => (
                            frame.float_regs.get(reg_idx).copied().flatten(),
                            majit_ir::Type::Float,
                        ),
                    };
                    let opref = opref_opt.unwrap_or_else(|| {
                        panic!(
                            "jit_merge_point listed red register {reg} is empty \
                             (`handle_jit_marker__jit_merge_point` / \
                             `prepare_list_of_boxes` copy every declared red)"
                        )
                    });
                    redboxes.push((opref, ty));
                }
                if slot == 0
                    && let Some(majit_ir::OpRef::ConstInt(v)) =
                        frame.int_regs.get(reg_idx).copied().flatten()
                {
                    if mp_green_pc.is_none() {
                        mp_green_pc = Some(v);
                    }
                    mp_green_ints.push(v);
                }
                // pyjitpl.py same_greenkey: ref (slot 1) and float
                // (slot 2) green constants for the full-green header
                // compare.  `equal_whatever(Float, ..)` compares f64 bits,
                // so store the float green's `to_bits`.
                if slot == 1
                    && let Some(majit_ir::OpRef::ConstPtr(v)) =
                        frame.ref_regs.get(reg_idx).copied().flatten()
                {
                    mp_green_refs.push(v.0 as i64);
                }
                if slot == 2
                    && let Some(majit_ir::OpRef::ConstFloat(v)) =
                        frame.float_regs.get(reg_idx).copied().flatten()
                {
                    mp_green_floats.push(v.to_bits() as i64);
                }
                debug_assert!(
                    reg_idx < max,
                    "BC_JIT_MERGE_POINT: register byte {reg} \
                     out of range for slot {slot} \
                     (kind bank size {max})",
                );
                if is_green_slot && cfg!(debug_assertions) {
                    // Look up the OpRef in the matching
                    // typed register bank.  A None slot
                    // means the register has not been
                    // populated yet — also a macro emission
                    // bug.
                    let opref_opt = match slot {
                        0 => frame.int_regs.get(reg_idx).copied().flatten(),
                        1 => frame.ref_regs.get(reg_idx).copied().flatten(),
                        2 => frame.float_regs.get(reg_idx).copied().flatten(),
                        _ => unreachable!(),
                    };
                    let Some(opref) = opref_opt else {
                        panic!(
                            "BC_JIT_MERGE_POINT: green register \
                             {reg} (slot {slot}) is unset at \
                             trace time (pyjitpl.py:1530 \
                             verify_green_args)",
                        );
                    };
                    assert!(
                        opref.is_constant(),
                        "BC_JIT_MERGE_POINT: green register \
                         {reg} (slot {slot}) holds non-Const \
                         OpRef {opref:?} — emit_promote_greens \
                         (`jtransform.py` `promote_greens`) must run before \
                         the merge point so all greens are \
                         constants (pyjitpl.py:1530-1535 \
                         verify_green_args)",
                    );
                }
            }
        }
        // pyjitpl.py MIFrame.opimpl_jit_merge_point: `redboxes` reach
        // `reached_loop_header` / `put_back_list_of_boxes3` only under
        // `if not self.metainterp.portal_call_depth`. This walker uses
        // `inline_depth()` for that root-portal test (same predicate as
        // the recursive-cut else-branch and `last_mp_green_pc` below).
        // Snapshot helpers re-read these register numbers from
        // `frames.first()` (the root portal frame).
        if ctx.inline_depth() == 0 {
            self.last_mp_green_i = green_i_regs;
            self.last_mp_green_r = green_r_regs;
            self.last_mp_green_f = green_f_regs;
            self.last_mp_red_i.clone_from(&red_i_regs);
            self.last_mp_red_r.clone_from(&red_r_regs);
            self.last_mp_red_f.clone_from(&red_f_regs);
            ctx.portal_green_regs_i.clone_from(&self.last_mp_green_i);
            ctx.portal_green_regs_r.clone_from(&self.last_mp_green_r);
            ctx.portal_green_regs_f.clone_from(&self.last_mp_green_f);
            ctx.portal_red_regs_i.clone_from(&self.last_mp_red_i);
            ctx.portal_red_regs_r.clone_from(&self.last_mp_red_r);
            ctx.portal_red_regs_f.clone_from(&self.last_mp_red_f);
            ctx.live_portal_greens = Some((
                mp_green_ints.to_vec(),
                mp_green_refs.to_vec(),
                mp_green_floats.to_vec(),
            ));
        }
        // pyjitpl.py MIFrame.opimpl_jit_merge_point — a jit_merge_point reached INSIDE an
        // inline recursive-portal callee, while no loop_header has been
        // seen yet (`seen_loop_header_for_jdindex < 0`), is a pure
        // no-op: the `if not jitdriver_sd.no_loop_header: if
        // self.metainterp.portal_call_depth: return` early-out skips the
        // auto loop-header stamp so the callee's merge point (it shares
        // the caller's dispatch jitcode, entered at offset 0) does not
        // corrupt the outer header.  The payload cursor was already
        // advanced above, so the callee continues to its next opcode.
        // A seen>=0 (or `no_loop_header` auto-stamped) depth>0 merge
        // point falls through into the close protocol at the else-branch
        // cut below (pyjitpl.py MIFrame.opimpl_jit_merge_point).
        // pyjitpl.py `debug_merge_point`, the tail of the
        // method every `jit_merge_point` runs through:
        //
        //     if (metainterp.force_finish_trace and
        //             (metainterp.history.length() >
        //              warmrunnerstate.trace_limit * 0.8)):
        //         self._create_segmented_trace_and_blackhole()
        //
        // A green key that already overflowed once carries
        // JC_FORCE_FINISH (set by `prepare_trace_segmenting`,
        // pyjitpl.py).  Its next attempt must not overflow
        // again: closing the trace as a segment here — strictly
        // before `_interpret`'s 1.0x `blackhole_if_trace_too_long`
        // (pyjitpl.py) can be reached — is what stops the
        // key from retracing forever.  The check belongs at a merge
        // point and nowhere else: the guard this records resumes
        // through the `-live-` marker that precedes every
        // `jit_merge_point` op, which an arbitrary mid-walk position
        // has no counterpart for.
        // Record the interpreter pc this merge point names, for the
        // abort-resume correction in
        // `trace_jitcode_with_args_and_runtime`.  It is written before
        // every early return below, because a visit that goes on to
        // return `Continue` still passed through a real opcode
        // boundary, and that boundary is what the correction needs.
        //
        // Restricted to `inline_depth() == 0`: the position it competes
        // with is the ROOT frame's i0, so an inlined callee's own pc
        // would name a position in the wrong code.
        if ctx.inline_depth() == 0 {
            if let Some(u) = mp_green_pc.and_then(|v| usize::try_from(v).ok()) {
                ctx.last_mp_green_pc = Some(u);
            }
        }
        if ctx.force_finish_trace() && ctx.num_ops() > ctx.trace_limit() * 4 / 5 {
            // The loop-vs-bridge split lives inside
            // `create_segmented_trace`, where upstream keeps it
            // (pyjitpl.py MIFrame._create_segmented_trace_and_blackhole) — the check reached here segments
            // whatever trace it is in, exactly as
            // `_create_segmented_trace_and_blackhole` does.
            return self.create_segmented_trace(ctx, sym, mp_opcode_pc, mp_green_pc);
        }
        // `opimpl_jit_merge_point` takes its driver from
        // `staticdata.jitdrivers_sd[jdindex]`.
        let no_loop_header = no_loop_header_jd;
        // pyjitpl.py `_handle_guard_failure` pre-arms the
        // flag when the source guard is a `ResumeAtPositionDescr` (the
        // descr `inline_short_preamble` stamps onto the guards it
        // replays, unroll.py OptUnroll._jump_to_existing_trace / inline_short_preamble). Those guards sit at the
        // target loop's entry, so the bridge grown from one closes at
        // its very first merge point instead of recording another
        // iteration; the pre-arm is what skips the ladder below.
        if self.seen_loop_header_for_jdindex < 0
            && std::mem::take(&mut ctx.bridge_resume_at_position)
        {
            self.seen_loop_header_for_jdindex = jdindex as i32;
        }
        if ctx.inline_depth() > 0 && self.seen_loop_header_for_jdindex < 0 && !no_loop_header {
            return TraceAction::Continue;
        }
        // pyjitpl.py: `reached_loop_header` is called with
        // `self.pc = orgpc`, and when it returns WITHOUT raising the
        // frame restores `self.pc = saved_pc` — the position just AFTER
        // the jit_merge_point — and goes on executing. The merge point
        // is therefore consulted once per visit; the loop body runs
        // before it is consulted again.
        //
        // Pyre fuses the merge point onto the guest instruction that
        // hosts it, so a walk re-entered at the same guest pc (the
        // `current_merge_points.append` path, jitdriver.rs `merge_point`)
        // arrives back at this op with nothing recorded in between and
        // would close again on the spot. This latch is `saved_pc`: skip
        // the ladder exactly once, so the resumed walk executes the
        // instruction's own arm and only re-consults the merge point a
        // full iteration later.
        if ctx.take_merge_point_resumed() {
            return TraceAction::Continue;
        }
        // MAJIT_PCSEQ (W4/D2 diagnostic): log the interpreter green pc
        // captured at EVERY merge-point re-entry (not gated on
        // seen_loop_header like MAJIT_MPTRACE). Confirms the walk holds a
        // concrete per-opcode next-pc = mp_green_pc, the walker-drives-pc
        // data source for the per-opcode single-executor.
        if crate::pcseq_enabled() {
            let sf: Vec<Option<i64>> = (0..3).map(|i| sym.state_field_value(i)).collect();
            eprintln!(
                "@@@PCSEQ mp pc={mp_green_pc:?} greens={mp_green_ints:?} refs={mp_green_refs:?} floats={mp_green_floats:?} hdr={:?} sf={sf:?} num_ops={} seen_lh={}",
                ctx.header_greens,
                ctx.num_ops(),
                self.seen_loop_header_for_jdindex,
            );
        }
        // pyjitpl.py opimpl_jit_merge_point auto
        // loop-header.  When `seen_loop_header_for_jdindex < 0`
        // (no explicit `BC_LOOP_HEADER` has stamped the flag yet),
        // RPython auto-stamps the merge point's jdindex when:
        //
        //     if not any_operation:
        //         return
        //     if not jitdriver_sd.no_loop_header:
        //         if self.metainterp.portal_call_depth:
        //             return
        //         ptoken = self.metainterp.get_procedure_token(greenboxes)
        //         if not has_compiled_targets(ptoken):
        //             return
        //     # automatically add a loop_header if there is none
        //     self.metainterp.seen_loop_header_for_jdindex = jdindex
        //
        // Pyre installs the gate inputs at trace start
        // (`pyjitpl::MetaInterp::setup_tracing` /
        // `force_start_tracing` / `jitdriver::start_bridge_tracing`):
        //   * `portal_call_depth_fn`: live
        //     `MetaInterp.portal_call_depth` sample.
        //   * `compiled_key_for_greens_fn`: live
        //     `get_procedure_token(greenboxes)` +
        //     `has_compiled_targets` for THIS merge point's greens.
        if self.seen_loop_header_for_jdindex < 0 && ctx.num_ops() > 0 {
            // `no_loop_header` hoisted above (EDIT A) and reused here.
            let should_auto_stamp = if no_loop_header {
                // pyjitpl.py MIFrame.opimpl_jit_merge_point path through (skip the
                // `if not jitdriver_sd.no_loop_header:` guard).
                true
            } else {
                // pyjitpl.py MIFrame.opimpl_jit_merge_point: portal_call_depth == 0 AND
                // has_compiled_targets(ptoken).  Both fns are
                // installed at every trace-start path; missing
                // installs would be a structural bug, so default
                // to "don't stamp" rather than over-stamping.
                let depth_zero = ctx
                    .portal_call_depth_fn
                    .as_ref()
                    .map(|f| f() == 0)
                    .unwrap_or(false);
                // pyjitpl.py MIFrame.opimpl_jit_merge_point keys `ptoken` on `greenboxes`
                // — the greens of the merge point being visited RIGHT
                // NOW, not the trace's own header. Keyed on the fixed
                // `ctx.green_key` instead, the stamp re-arms at every
                // merge point the trace reaches once the START key has
                // compiled targets, so `reached_loop_header` runs on
                // body merge points that upstream returns from at 1554
                // without recording anything. Every consequence of that
                // follows: a GUARD_FUTURE_CONDITION per body merge
                // point (upstream emits it only inside
                // reached_loop_header, 2993), a close attempt per body
                // merge point, and — once `retrace_needed` has armed
                // `partial_trace` — a close storm that leaves no room
                // for the one extra iteration a retrace has to trace.
                //
                // The key must be the one the INTERPRETER enters by:
                // `get_procedure_token` is `jit_cell_at_key(greenkey)`,
                // and `compile_loop` attaches the token to that same
                // cell, so upstream cannot own a compiled loop the
                // interpreter cannot reach and this predicate is always
                // false for a key being traced for the first time —
                // which is what makes :1554-1555 `return` the guard
                // against closing a trace on its own first merge point.
                // A lookup keyed on anything else (e.g. scanning a side
                // table for a loop whose header greens happen to match)
                // can answer yes for a loop stored under a key nothing
                // enters, auto-stamp here, and close with nothing but
                // the green-promotion ops recorded.
                let has_targets = mp_green_pc
                    .and_then(|pc| {
                        ctx.merge_point_green_key_hash(
                            pc,
                            &mp_green_ints,
                            &mp_green_refs,
                            &mp_green_floats,
                        )
                    })
                    .zip(ctx.has_compiled_targets_fn.as_ref())
                    .is_some_and(|(key, f)| f(key));
                depth_zero && has_targets
            };
            if should_auto_stamp {
                self.seen_loop_header_for_jdindex = jdindex as i32;
            }
        }
        // pyjitpl.py MetaInterp.reached_loop_header `current_merge_points.append(...)`: the
        // FIRST merge-point visit of a primary trace is the loop header;
        // snapshot its concrete green constants (grouped by IR slot) as
        // the `same_greenkey` reference for every later visit.  A bridge's
        // trace-start header is the guard, so it does not capture
        // this; its `same_greenkey` reads `compiled_key_for_greens`
        // against the source loop instead.
        if !ctx.is_bridge_trace && ctx.header_greens.is_none() {
            ctx.header_greens = Some((
                mp_green_ints.to_vec(),
                mp_green_refs.to_vec(),
                mp_green_floats.to_vec(),
            ));
        }
        // pyjitpl.py opimpl_jit_merge_point close-loop
        // protocol — read the per-driver flag stamped by the
        // previous iteration's `BC_LOOP_HEADER` or by the
        // first-iteration auto-set above:
        //
        //     assert seen_loop_header_for_jdindex == jdindex
        //     seen_loop_header_for_jdindex = -1
        //     reached_loop_header(...)
        if self.seen_loop_header_for_jdindex >= 0 {
            assert_eq!(
                self.seen_loop_header_for_jdindex as usize, jdindex,
                "BC_JIT_MERGE_POINT: seen_loop_header_for_jdindex \
                 {} disagrees with merge-point jdindex {jdindex} — \
                 pyjitpl.py:1559 found a loop_header for a JitDriver \
                 that does not match the following jit_merge_point",
                self.seen_loop_header_for_jdindex,
            );
            self.seen_loop_header_for_jdindex = -1;
            if ctx.inline_depth() > 0 {
                // pyjitpl.py MIFrame.opimpl_jit_merge_point else-branch: a recursive-portal
                // merge point reached at portal_call_depth > 0 is NOT
                // the traced loop's own header. Instead of a close it
                // returns from the inlined callee frame
                // (`finishframe(leave_portal_frame=False)`), then
                // `do_recursive_call(assembler_call=True)` on the caller
                // frame so the recursion follows a CALL_ASSEMBLER into
                // the callee's own compiled loop, then
                // `leave_portal_frame`, then `raise ChangeFrame` to
                // resume the caller.
                //
                // (1) capture old_frame (the inlined portal callee, top
                //     of stack) before the pop (pyjitpl.py) so its
                //     return-destination slot and jitdriver index are
                //     read from the callee, not the caller.
                let old_frame = self.frames.current_mut();
                let jd_no = old_frame.jitcode.jitdriver_sd().unwrap_or(jdindex);
                let (result_kind, result_dst) = if let Some(d) = old_frame.return_i {
                    (Some(JitArgKind::Int), Some(d))
                } else if let Some(d) = old_frame.return_r {
                    (Some(JitArgKind::Ref), Some(d))
                } else if let Some(d) = old_frame.return_f {
                    (Some(JitArgKind::Float), Some(d))
                } else {
                    (None, None)
                };
                // (2) finishframe(leave_portal_frame=False) analog
                //     (pyjitpl.py): pop the inline callee
                //     frame, mirror the ctx inline depth, and restore the
                //     caller sym scalar / vable state the push saved —
                //     but DO NOT wire the return (do_recursive_call's job)
                //     and DO NOT record LEAVE_PORTAL_FRAME yet
                //     (leave_portal_frame=False).
                let mut popped = self.frames.pop().expect("recursive-cut: framestack < 2");
                if popped.inline_frame {
                    ctx.pop_inline_frame();
                }
                // popframe still appends the log close when greenkey
                // is set, even with leave_portal_frame=False.
                if popped.portal_trace_logged {
                    ctx.push_portal_trace_event(popped.portal_jd, None, ctx.get_trace_position());
                }
                if let Some(snapshot) = popped.portal_scalar_state.take() {
                    sym.restore_inline_scalar_state(snapshot);
                }
                self.frames.recycle_frame(popped);
                // (3) do_recursive_call(assembler_call=True) on the caller
                //     (now current): reuse the existing 8-step
                //     CALL_ASSEMBLER recorder. `set_int_reg(result_dst)`
                //     inside binds the result on the CALLER frame
                //     (make_result_of_lastop parity, pyjitpl.py).
                let green_values: Vec<i64> = mp_green_ints
                    .iter()
                    .chain(mp_green_refs.iter())
                    .chain(mp_green_floats.iter())
                    .copied()
                    .collect();
                match self.exec_recursive_call_assembler(
                    ctx,
                    sym,
                    _runtime,
                    result_kind,
                    jdindex,
                    result_dst,
                    &green_values,
                ) {
                    TraceAction::Continue => {}
                    // Abort propagates (missing fresh-reds / target).
                    other => return other,
                }
                // (4) deferred LEAVE_PORTAL_FRAME (pyjitpl.py MIFrame.opimpl_jit_merge_point),
                //     recorded AFTER the CALL_ASSEMBLER so the trace order
                //     is CALL_ASSEMBLER … then LEAVE.
                let jd_box = ctx.const_int(jd_no as i64);
                ctx.record_op(OpCode::LeavePortalFrame, &[jd_box]);
                // (5) raise ChangeFrame (pyjitpl.py): resume the
                //     caller frame in the walker dispatch loop.
                return TraceAction::Continue;
            }
            // pyjitpl.py reached_loop_header, its FIRST statement:
            //
            //     def reached_loop_header(self, greenboxes, redboxes):
            //         self.heapcache.reset()
            //
            // A merge point is where another trace may be cut in
            // (compile.py compile_loop `trace.cut_trace_from`) and where a
            // compiled loop may be entered from the interpreter.  So
            // nothing recorded past this point may depend on a heapcache
            // fact established before it: the guard that proved the fact
            // can end up on the far side of a cut, or simply never run on
            // an entry that starts here.  Resetting forces the tracer to
            // re-emit those guards, which is what keeps the ops after the
            // merge point self-sufficient.
            //
            // Placed after the `inline_depth() > 0` recursive-cut branch
            // above because that branch is upstream's `else` at
            // pyjitpl.py — it returns instead of calling
            // `reached_loop_header`, so the reset must not run on it.
            // Upstream reaches the GUARD_FUTURE_CONDITION at :2993 with
            // only `remove_consts_and_duplicates` and the virtualizable
            // box handling in between, neither of which reads the
            // heapcache, so reset-then-guard is the faithful order.
            // `reached_loop_header`: one `duplicates` dict; reds first,
            // then `virtualizable_boxes[:-1]`. `live_arg_boxes =
            // greenboxes + redboxes` then `+= virtualizable_boxes; pop()`.
            // The jit_interp recorder LABEL is the setup_call InputArgs
            // (reds), so the JUMP/registration list is reds + vable
            // elements — greens stay in `greenboxes` for same_greenkey /
            // procedure tokens.
            let boxes = ctx.reached_loop_header_live_arg_boxes(&mut redboxes, None);
            sym.set_redboxes(&redboxes);
            live_arg_boxes.clear();
            live_arg_boxes.extend(
                boxes
                    .iter()
                    .map(|&(op, ty)| crate::trace_ctx::GreenBox::new(op, ty)),
            );
            put_back_reds = true;
            // pyjitpl.py reached_loop_header: generate a dummy
            // GUARD_FUTURE_CONDITION just before the implicit JUMP so
            // unroll's `jump_to_existing_trace` has a `patchguardop`
            // whose `rd_resume_position` it copies onto every extra
            // virtual-state guard (unroll.py OptUnroll._jump_to_existing_trace, resume.py ResumeDataVirtualAdder.finish).
            // The source-level tracer emits this in `close_loop_args_at`
            // (trace_opcode.rs); the state-field dispatch model
            // reaches the loop header here instead.  Emitted
            // unconditionally at the top of the reached_loop_header
            // equivalent, BEFORE the header-match/close/append branching,
            // so it fires for EVERY outcome — including the inner-loop
            // first-visit append-and-continue path (pyjitpl.py)
            // — matching upstream's top-of-function `generate_guard`
            // emission.  `record_state_guard` captures the matching resume
            // snapshot at `mp_opcode_pc`, mirroring `generate_guard`'s
            // `capture_resumedata` (pyjitpl.py).
            self.record_state_guard(
                ctx,
                sym,
                OpCode::GuardFutureCondition,
                &[],
                mp_opcode_pc,
                false,
            );
            // pyjitpl.py reached_loop_header: close the loop
            // ONLY when the current merge point's green key matches the
            // trace-start (loop-header) key — `same_greenkey`
            // (pyjitpl.py / 3912-3920).  The seen_loop_header
            // flag alone is necessary but not sufficient: the auto-stamp
            // (above) keys on the FIXED trace-start `ctx.green_key`, so it
            // fires at whatever merge point the trace reaches once the
            // start key has compiled targets — not necessarily the loop
            // header.  The pc green is the loop-header discriminator; a
            // merge point whose pc differs from `header_pc` is a
            // different green key, which RPython appends to
            // current_merge_points and keeps tracing past.  Closing there
            // emits a JUMP from a non-header pc back to the header
            // inputargs, manufacturing a degenerate loop whose now-
            // redundant exit guards const-fold away (infinite loop).  A
            // jitdriver with no int pc green keeps the flag-only close.
            // pyjitpl.py reached_loop_header: a bridge has no own
            // loop header to loop back to — it closes by JUMPing into
            // its parent loop (`has_compiled_targets(greenboxes)`), which
            // lives at `bridge_target_header_pc`. Closing on a transient
            // revisit of the bridge's own `resume_pc` (`header_pc`) bakes
            // a degenerate empty bridge that jumps back with no forward
            // progress. A primary trace still self-closes at `header_pc`.
            let close_target_pc = if ctx.is_bridge_trace {
                ctx.bridge_target_header_pc.unwrap_or(ctx.header_pc)
            } else {
                ctx.header_pc
            };
            let pc_matches = mp_green_pc.is_none_or(|pc| pc == close_target_pc as i64);
            // pyjitpl.py/3912 same_greenkey: beyond the pc, close
            // only when EVERY green — the scalars and the ref the
            // consumer declares — equals the trace-start header's.  Compare
            // element-wise against the header greens captured on the first
            // visit (`header_greens`) — the SAME merge-point green
            // vocabulary — which is `Box.same_constant` per Const type
            // (ConstInt/ConstFloat bitwise, ConstPtr pointer identity),
            // exactly what plain slot-grouped `Vec` equality performs.
            // The reference is NOT `green_key_values` (the
            // back-edge/can_enter_jit key carries a different arity) nor
            // `current_merge_points[0].green_boxes` (InputArg placeholders,
            // no Const → an always-empty filter that would decline every
            // close and hang).
            let same_greenkey = if ctx.is_bridge_trace {
                // pyjitpl.py `same_greenkey` over the greens of the merge
                // point just reached. A bridge's trace-start header is
                // the guard, not the parent loop, so the reference is
                // the compiled loop those greens name
                // (`compiled_key_for_greens` / `loop_header_greens`).
                // It has to be this bridge's source loop: the same pc
                // in another code object is a different green key.
                // `None` means that loop was compiled without stored
                // greens; the pc check above stays the only
                // discriminator, which is what this arm used to be.
                let greens = (
                    mp_green_ints.to_vec(),
                    mp_green_refs.to_vec(),
                    mp_green_floats.to_vec(),
                );
                match ctx
                    .compiled_key_for_greens_fn
                    .as_ref()
                    .and_then(|lookup| lookup(&greens))
                {
                    Some(key) => key == ctx.green_key,
                    None => true,
                }
            } else if let Some((h_ints, h_refs, h_floats)) = ctx.header_greens.as_ref() {
                mp_green_ints.as_slice() == h_ints.as_slice()
                    && mp_green_refs.as_slice() == h_refs.as_slice()
                    && mp_green_floats.as_slice() == h_floats.as_slice()
            } else {
                // Header greens not captured (no prior visit): the (pc,
                // code) hash is the loop identity — fall back to pc-only.
                true
            };
            let header_matches = pc_matches && same_greenkey;
            if crate::mptrace_enabled() {
                eprintln!(
                    "@@@MPTRACE visit pc={mp_green_pc:?} header_pc={} close_target={close_target_pc} matches={header_matches} num_ops={}",
                    ctx.header_pc,
                    ctx.num_ops(),
                );
            }
            if header_matches {
                if crate::jitdriver::spdiag_enabled() {
                    eprintln!(
                        "@@@SPDIAG HEADER-CLOSE close_target_pc={close_target_pc} mp_green_pc={mp_green_pc:?} walk_reds={walk_reds:?}"
                    );
                }
                // pyjitpl.py `get_procedure_token(greenboxes)` —
                // the greens of the merge point just reached.
                let close_greens = (
                    mp_green_ints.to_vec(),
                    mp_green_refs.to_vec(),
                    mp_green_floats.to_vec(),
                );
                ctx.close_greens = Some(close_greens.clone());
                ctx.close_green_pc = mp_green_pc;
                if ctx.is_bridge_trace {
                    // pyjitpl.py MetaInterp.reached_loop_header: a guard-origin bridge
                    // first consults the procedure token for the
                    // merge point just reached.  If none has compiled
                    // targets, it does NOT close on the first visit;
                    // it falls through to the current_merge_points
                    // scan, appends first visits, and only closes on a
                    // repeated same-greenkey merge point.
                    let already_compiled_here = ctx
                        .close_green_key_hash()
                        .zip(ctx.has_compiled_targets_fn.as_ref())
                        .is_some_and(|(key, f)| f(key));
                    // Take the structured key and derive the hash from
                    // it, rather than taking the hash and leaving the
                    // key behind: the merge point this registers is
                    // later read by consumers that install cell flags,
                    // and a hash alone reaches a cell only by bucket.
                    let (close_key, close_key_typed) = match ctx.close_green_key() {
                        Some(k) => (k.get_uhash(), Some(k)),
                        None => (ctx.green_key, ctx.green_key_values().cloned()),
                    };
                    if !already_compiled_here
                        && ctx
                            .find_merge_point_same_greenkey(close_key, close_key_typed.as_ref())
                            .is_none()
                    {
                        let original_boxes = live_arg_boxes.to_vec();
                        if crate::mptrace_enabled() {
                            eprintln!(
                                "@@@MPTRACE bridge-add-mp key={close_key} header_pc={} num_ops={}",
                                ctx.header_pc,
                                ctx.num_ops(),
                            );
                        }
                        // `MergePoint::header_pc` is this visit's guest pc
                        // (`same_greenkey`'s pc green), not the
                        // trace-start `ctx.header_pc`.
                        // A present negative green is not this visit's
                        // header. Falling back to the trace-start pc
                        // would file it on a different loop.
                        let recorded_pc = match mp_green_pc {
                            Some(pc) => Self::guest_pc_position(pc),
                            None => ctx.header_pc,
                        };
                        ctx.add_merge_point_with_key(
                            close_key,
                            close_key_typed,
                            original_boxes,
                            recorded_pc,
                        );
                        self.put_back_list_of_boxes3(
                            &red_i_regs,
                            &red_r_regs,
                            &red_f_regs,
                            &redboxes,
                        );
                        return TraceAction::Continue;
                    }
                }
                if capture_walk_reds {
                    // Single-pass: stash the resume-aligned close pc (the
                    // interpreter green pc, NOT the JitCode op cursor) so
                    // the merge-point hook can resume the native loop
                    // there in lieu of the observer replay. The loop-carried
                    // red values are transferred into native state by the
                    // hook (`restore_values`); storage caches re-derive via
                    // `recover`.
                    ctx.walk_final_pc = mp_green_pc.map(Self::guest_pc_position);
                    ctx.walk_final_reds = std::mem::take(&mut walk_reds).into_vec();
                }
                // GUARD_FUTURE_CONDITION already emitted unconditionally at
                // the reached_loop_header entry above (pyjitpl.py).
                return TraceAction::CloseLoop;
            }
            // No same_greenkey match — fall through and keep tracing
            // (the merge point op is otherwise a no-op while recording).
            //
            // pyjitpl.py reached_loop_header: a merge point
            // whose pc differs from the trace-start header is a different
            // green key. RPython scans current_merge_points for a prior
            // same_greenkey visit; if found it closes the loop THERE
            // (cutting the outer prefix as preamble); otherwise it appends
            // and keeps tracing. Record the inner merge point under its
            // own green key and its own guest pc. The scan is
            // `find_merge_point_same_greenkey`, not `(key, trace header_pc)`.
            //
            // The S0 census that established this — append-and-observe
            // with NO close, confirming the inner key is stable and
            // detected on revisit before the cut close was enabled — ran
            // behind a `MAJIT_INNERMP` gate that no longer exists. Nothing
            // reads that name today; re-running the census means adding
            // the gate back, not setting a variable.
            if inner_close && let Some(pc) = mp_green_pc {
                // pyjitpl.py MetaInterp.reached_loop_header, which runs BEFORE the
                // `current_merge_points` scan:
                //
                //     ptoken = self.get_procedure_token(greenboxes)
                //     if has_compiled_targets(ptoken):
                //         self.compile_trace(live_arg_boxes, ptoken)
                //
                // `greenboxes` is the merge point just reached, so a
                // loop that ALREADY has compiled code is jumped into,
                // never re-derived by cutting this trace at it. The
                // cut (compile.py compile_loop) is for the other case: an
                // inner loop nobody has compiled yet, which
                // `compile_loop` then attaches to
                // `original_boxes[:num_green_args]` — the INNER
                // greenkey (pyjitpl.py).
                //
                // The JUMP is what the `already_compiled_here` arm
                // below performs: it publishes the token key and
                // returns `CloseLoop`, and the driver runs
                // `close_bridge` (guard origin) or
                // `compile_trace_from_interp` (interp origin).
                //
                // It is sound only because the key this arm derives
                // is the one the interpreter ENTERS by.  While the
                // key was `green_key_from_code_ptr(green_key_raw.0,
                // pc)` — `JitState::code_ptr()` defaulting to 0, not
                // the driver's `GreenKey::hash_u64` — a compiled loop
                // could sit under a key nothing enters, and jumping
                // into it was measured as a logo miscompile (992635
                // against 996310) and a SIGSEGV.  The four
                // procedure-token consults now share
                // `merge_point_green_key_hash`, so a loop is stored
                // under the key it is reached by and the jump lands
                // in code the interpreter can also enter.
                //
                // The earlier measurement against this lever —
                // cel's `nested_list_loop_varying_trip_count` keeping
                // its results and losing its 4 aborts while `spread
                // 0..32` deopts went 959 → 1763 over 4000 rows and
                // 1275 → 2601 over 16000 — was taken BEFORE that key
                // unification, i.e. against jumps into loops filed
                // under keys nothing enters.  It does not carry over
                // and must be re-measured before being cited again.
                // Same for the note that routing the closing JUMP's
                // target tokens off the token it enters rather than
                // off the bridge origin (`unroll.py UnrollOptimizer.optimize_bridge`
                // `cell_token = jump_op.getdescr()` — pyre's
                // `compile_bridge` hands `optimize_bridge` the ORIGIN
                // loop's `target_tokens`) recovered only 7%.
                //
                // A declined `compile_trace` does not append here.
                // The walker has no `MetaInterp` to scan
                // `current_merge_points`. `JitDriver::keep_tracing_after_declined_jump`
                // is that scan (`reached_loop_header`): no prior
                // same-greenkey entry appends and the walk continues;
                // a prior entry falls through to `compile_loop`.
                //
                // The token lookup below is unconditional, where
                // upstream guards it with `if not self.partial_trace:`
                // (:3002).  That gate must not be spelled
                // `is_bridge_trace`: `partial_trace` is set only by
                // `retrace_needed` (pyjitpl.py), so it means
                // "this is a RETRACE", and a bridge from a guard
                // failure runs the consult upstream just like a
                // primary entry.  Its live reading is
                // `MetaInterp::partial_trace()`, which this loop
                // cannot reach while it holds the TraceCtx borrow —
                // harmless while the lookup only decides a log line,
                // and part of what the JUMP half has to carry.
                // The structured key first, hash second: this merge
                // point is registered below and later read by the
                // segmenting consumers, which install cell flags and
                // so need a key a chain walk can match, not a bucket.
                let Some(inner_key_typed) =
                    ctx.merge_point_green_key(pc, &mp_green_ints, &mp_green_refs, &mp_green_floats)
                else {
                    return TraceAction::Continue;
                };
                let inner_key = inner_key_typed.get_uhash();
                let already_compiled_here = ctx
                    .has_compiled_targets_fn
                    .as_ref()
                    .is_some_and(|f| f(inner_key));
                // Producer side of slots 50/67, which count only what
                // happens once a close has been published. Counted here,
                // before the branch, so a zero downstream separates "the
                // walk never reached this decision" from "it reached it
                // and the target was not compiled" — the two render
                // identically in those two slots. `is_some_and` also
                // answers false when the callback is absent, so the
                // reached-count is what makes an uninstalled
                // `has_compiled_targets_fn` visible rather than
                // indistinguishable from a real "no".
                //
                // The sibling `already_compiled_here` on the
                // `is_bridge_trace` path above is a DIFFERENT decision —
                // whether to append a first-visit merge point — and never
                // publishes `close_jump_into_key`, so it deliberately
                // carries no slot.
                crate::mc_diag_bump(68); // xloop_close_decision_reached
                if already_compiled_here {
                    crate::mc_diag_bump(69); // xloop_close_target_compiled
                    // pyjitpl.py MetaInterp.reached_loop_header — the merge point just reached already owns a
                    // procedure token, so upstream JUMPs into it rather than deriving a second
                    // copy of that loop by cutting this trace.  `compile_trace` raises on
                    // success (`raise_if_successful`, pyjitpl.py), which is why the
                    // `current_merge_points` scan below is never reached in that case.
                    //
                    // The dispatcher holds no `&mut MetaInterp`, so the attempt is published to
                    // the driver: `close_jump_into_key` names the token, `close_greens` /
                    // `close_green_pc` name the greens it is keyed by (pyjitpl.py
                    // `get_procedure_token(greenboxes)` reads the greens of the merge point just
                    // reached, not the trace-start header's).
                    //
                    // pyjitpl.py compile_trace is retried on every
                    // header visit (`if not self.partial_trace`).
                    crate::mc_diag_bump(70); // xloop_close_published
                    ctx.close_greens = Some((
                        mp_green_ints.to_vec(),
                        mp_green_refs.to_vec(),
                        mp_green_floats.to_vec(),
                    ));
                    ctx.close_green_pc = Some(pc);
                    ctx.close_jump_into_key = Some(inner_key);
                    if capture_walk_reds {
                        ctx.walk_final_pc = Some(Self::guest_pc_position(pc));
                        ctx.walk_final_reds = std::mem::take(&mut walk_reds).into_vec();
                    }
                    if crate::majit_log_enabled() {
                        eprintln!(
                            "[jit] merge point pc={pc} has compiled loop key={inner_key} \
                                     — compile_trace JUMP (pyjitpl.py compile_trace)"
                        );
                    }
                    // GUARD_FUTURE_CONDITION was already emitted unconditionally at the
                    // reached_loop_header entry above (pyjitpl.py).
                    return TraceAction::CloseLoop;
                } else if ctx
                    .find_merge_point_same_greenkey(inner_key, Some(&inner_key_typed))
                    .is_some()
                {
                    if crate::jitdriver::spdiag_enabled() {
                        eprintln!(
                            "@@@SPDIAG INNER-CUT-CLOSE pc={pc} inner_key={inner_key} walk_reds={walk_reds:?}"
                        );
                    }
                    if crate::closedbg_enabled() {
                        let portal = &self.frames.frames[0];
                        for (i, slot) in portal.int_regs.iter().enumerate() {
                            if let Some(o) = slot {
                                eprintln!("@@@RED int[{i}]={o:?}");
                            }
                        }
                        for (j, slot) in portal.ref_regs.iter().enumerate() {
                            if let Some(o) = slot {
                                eprintln!("@@@RED ref[{j}]={o:?}");
                            }
                        }
                    }
                    // same_greenkey revisit of a nested inner loop →
                    // close HERE and cut the outer prefix as preamble.
                    // Setting cut_inner_green_key routes compile_loop
                    // through cross_loop_cut (compile.py compile_loop).
                    ctx.cut_inner_green_key = Some(inner_key);
                    // pyjitpl.py `get_procedure_token(greenboxes)`
                    // reads the greens of the merge point just
                    // reached — here the INNER loop's, not the
                    // trace-start header's.
                    ctx.close_greens = Some((
                        mp_green_ints.to_vec(),
                        mp_green_refs.to_vec(),
                        mp_green_floats.to_vec(),
                    ));
                    ctx.close_green_pc = Some(pc);
                    if capture_walk_reds {
                        // Single-pass: resume at the inner loop's
                        // interpreter green pc (the loop variable the
                        // hook assigns to `pc`). The loop-carried red
                        // values captured above are transferred into
                        // native state by the merge-point hook
                        // (`restore_values`); storage caches are then
                        // re-derived by `recover`.
                        ctx.walk_final_pc = Some(Self::guest_pc_position(pc));
                        ctx.walk_final_reds = std::mem::take(&mut walk_reds).into_vec();
                    }
                    // GUARD_FUTURE_CONDITION already emitted
                    // unconditionally at the reached_loop_header entry
                    // above (pyjitpl.py).
                    return TraceAction::CloseLoop;
                } else {
                    // first visit → append and keep tracing
                    // (pyjitpl.py MetaInterp.reached_loop_header). For the state-field dispatch
                    // model the merge point's loop-carried values are the
                    // RED state fields (the closing JUMP = collect_jump_args),
                    // NOT the green operands captured in `live_arg_boxes`
                    // (those are promoted constants folded inline in the cut
                    // body). Register the SAME construction the close uses
                    // — `JitState::collect_jump_args_with_boxes` reached
                    // through `JitCodeSym::loop_carried_boxes` — so the cut
                    // label's inputarg arity matches the JUMP's
                    // (compile.py compile_loop jump.numargs()==label.numargs()), the
                    // way RPython's single `live_arg_boxes` list does by
                    // construction (pyjitpl.py MetaInterp.remove_consts_and_duplicates). Falls back to the
                    // operand-captured boxes for interpreters with no state
                    // fields at all.
                    //
                    // Building this from the scalar state fields alone (as
                    // this site used to) silently omits the virtualizable's
                    // element boxes, so an interpreter whose state is purely
                    // `[int; virt]` / `[float; virt]` arrays registered just
                    // the greens plus one unexpanded vable ref while its
                    // close expanded to one box per element — the arity
                    // mismatch that made every nested loop decline.
                    let original_boxes = live_arg_boxes.to_vec();
                    if crate::mptrace_enabled() {
                        eprintln!(
                            "@@@MPTRACE add-mp pc={pc} inner_key={inner_key} num_ops={}",
                            ctx.num_ops()
                        );
                    }
                    ctx.add_merge_point_with_key(
                        inner_key,
                        Some(inner_key_typed),
                        original_boxes,
                        Self::guest_pc_position(pc),
                    );
                }
            }
        }
        if put_back_reds {
            // `put_back_list_of_boxes3`: reached_loop_header returned
            // without closing; write the (possibly SAME_AS-rewritten)
            // reds back to the operand registers.
            self.put_back_list_of_boxes3(&red_i_regs, &red_r_regs, &red_f_regs, &redboxes);
        }
        TraceAction::Continue
    }

    /// `pyjitpl.py put_back_list_of_boxes3`: write `redboxes` back to the
    /// operand registers listed on the marker.
    fn put_back_list_of_boxes3(
        &mut self,
        red_i_regs: &[u8],
        red_r_regs: &[u8],
        red_f_regs: &[u8],
        redboxes: &[(OpRef, majit_ir::Type)],
    ) {
        assert_eq!(
            redboxes.len(),
            red_i_regs.len() + red_r_regs.len() + red_f_regs.len(),
            "put_back_list_of_boxes3: redboxes length must equal the three operand lists"
        );
        let frame = self.frames.current_mut();
        let mut i = 0;
        for &reg in red_i_regs {
            frame.int_regs[reg as usize] = Some(redboxes[i].0);
            i += 1;
        }
        for &reg in red_r_regs {
            frame.ref_regs[reg as usize] = Some(redboxes[i].0);
            i += 1;
        }
        for &reg in red_f_regs {
            frame.float_regs[reg as usize] = Some(redboxes[i].0);
            i += 1;
        }
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_loop_header(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        // pyjitpl.py opimpl_loop_header parity:
        //
        //     @arguments("int", "orgpc")
        //     def opimpl_loop_header(self, jdindex, orgpc):
        //         self.metainterp.seen_loop_header_for_jdindex = jdindex
        //
        // The op only sets the per-driver `seen_loop_header_for_jdindex`
        // flag; the actual close happens later in
        // `opimpl_jit_merge_point` (pyjitpl.py — assert flag
        // matches, reset, then `reached_loop_header`).
        //
        // RPython `assembler.py` USE_C_FORM does NOT include
        // `loop_header`, so the only valid argcode is `i` (constants-
        // pool slot — `jitcode/assembler.rs`'s `loop_header` patches the
        // byte at finish() to `num_regs_i + const_idx`).  Decode the
        // byte through the int register box to recover the actual jdindex
        // rather than reading the slot byte as the index directly,
        // mirroring `blackhole.py self.registers_i[ord(code[pos])]`.
        let frame = self.frames.current_mut();
        let jdindex_byte = frame.next_reg();
        let slot = jdindex_byte as usize;
        let jdindex = frame.getint(ctx, slot).expect(
            "BC_LOOP_HEADER (i form): jdindex register slot \
                 must hold a populated int constant — \
                 assembler.rs loop_header emits an `i` argcode \
                 pointing into the post-regs constants suffix",
        );
        let registered_drivers = ctx.metainterp_sd().jitdrivers_sd.len();
        assert!(
            (jdindex as usize) < registered_drivers,
            "BC_LOOP_HEADER: jdindex {jdindex} out of range \
             (registered drivers: {registered_drivers})",
        );
        // Stamp the per-driver flag so the next BC_JIT_MERGE_POINT
        // recognises that this trace passed through a matching
        // loop_header op (pyjitpl.py opimpl_loop_header).  No close trigger
        // here — RPython's `opimpl_loop_header` is a pure flag
        // setter; the close happens in BC_JIT_MERGE_POINT after
        // the assert/reset on the next iteration.
        self.seen_loop_header_for_jdindex = jdindex as i32;
        if crate::pcseq_enabled() {
            eprintln!(
                "@@@PCSEQ loop_header jdindex={jdindex} num_ops={} depth={}",
                ctx.num_ops(),
                self.frames.len(),
            );
        }
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_jump(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let target = self.frames.current_mut().next_u16() as usize;
        self.frames.current_mut().code_cursor = target;
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_inline_call(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        // pyjitpl.py newframe decoder: mint the callee, then
        // fill_registers from the caller bytecode. Do not collect
        // the arg triples into a Vec (9 usizes = 72 B).
        let (sub_idx, num_args) = {
            let frame = self.frames.current_mut();
            (frame.next_u16() as usize, frame.next_u16() as usize)
        };
        // RPython blackhole.py — `j` argcode resolves via
        // `self.descrs[idx]` asserted to be a JitCode.
        // `as_jitcode_owned`, so a recursive helper's back edge resolves
        // to the same callee an owning edge would.  By the time anything
        // executes this operand the callee is published, so the `Weak`
        // upgrades; the `None` window is confined to the helper's own
        // assembly, which never runs code.
        let sub_jitcode = self
            .frames
            .current_mut()
            .jitcode
            .descr_at(sub_idx)
            .and_then(crate::jitcode::RuntimeBhDescr::as_jitcode_owned)
            .unwrap_or_else(|| panic!("BC_INLINE_CALL: descrs[{sub_idx}] is not a JitCode entry"));
        let mut sub_frame = self.frames.take_frame(sub_jitcode, 0, None, Some(ctx));
        sub_frame.inline_frame = true;
        let (return_i, return_r, return_f) = {
            let caller = self.frames.current_mut();
            for _ in 0..num_args {
                let kind = JitArgKind::decode(caller.next_u8());
                let caller_src = caller.next_reg() as usize;
                let callee_dst = caller.next_reg() as usize;
                match kind {
                    JitArgKind::Int => {
                        #[cfg(feature = "jit-audits")]
                        majit_ir::reg_write_audit::note_int_write(
                            sub_frame.int_regs.as_ptr() as usize,
                            callee_dst,
                            caller.int_regs[caller_src],
                        );
                        sub_frame.int_regs[callee_dst] = caller.int_regs[caller_src];
                    }
                    JitArgKind::Ref => {
                        sub_frame.ref_regs[callee_dst] = caller.ref_regs[caller_src];
                    }
                    JitArgKind::Float => {
                        sub_frame.float_regs[callee_dst] = caller.float_regs[caller_src];
                    }
                }
            }
            let dest = {
                let dst = caller.next_reg() as usize;
                if dst == crate::jitcode::NO_RETURN_REG as usize {
                    None
                } else {
                    Some(dst)
                }
            };
            caller.pc = caller.code_cursor;
            let resulttype = caller
                .jitcode
                .core()
                .body()
                .resulttypes
                .as_ref()
                .and_then(|types| types.get(&caller.pc).copied());
            let (return_i, return_r, return_f, result_argcode) = match resulttype {
                Some('i') => (dest, None, None, b'i'),
                Some('r') => (None, dest, None, b'r'),
                Some('f') => (None, None, dest, b'f'),
                _ => (None, None, None, b'v'),
            };
            caller._result_argcode = result_argcode;
            caller.result_arg_index = dest;
            ctx.push_inline_frame((sub_idx, caller.pc), u32::MAX);
            (return_i, return_r, return_f)
        };
        sub_frame.return_i = return_i;
        sub_frame.return_r = return_r;
        sub_frame.return_f = return_f;
        self.frames.push(sub_frame);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_inline_call_r_i(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        match self.exec_typed_inline_call(ctx, sym, bytecode) {
            TraceAction::Continue => {}
            action => return action,
        }
        TraceAction::Continue
    }

    // Recursive portal call (self-recursion).  Unlike
    // BC_INLINE_CALL — which resolves its callee from the parent
    // frame's `descrs` pool — a recursive portal call targets the
    // portal jitcode itself, which is in no descrs slot at emit
    // time.  The opcode therefore carries the jitdriver index, and
    // `exec_recursive_call` resolves both the depth-gated inline
    // decision and the portal jitcode through `JitCodeRuntime`.
    // The result bank is selected by the opcode (`INT`/`REF`/
    // `FLOAT` → typed result; `VOID` → no result).
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_recursive_call_int(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let result_kind = match bytecode {
            jitcode::insns::BC_RECURSIVE_CALL_INT => Some(JitArgKind::Int),
            jitcode::insns::BC_RECURSIVE_CALL_REF => Some(JitArgKind::Ref),
            jitcode::insns::BC_RECURSIVE_CALL_FLOAT => Some(JitArgKind::Float),
            _ => None,
        };
        match self.exec_recursive_call(ctx, sym, _runtime, result_kind) {
            TraceAction::Continue => {}
            action => return action,
        }
        TraceAction::Continue
    }

    // ── Typed return arms ──
    //
    // RPython parity: pyjitpl.py MIFrame._opimpl_any_return
    // (`opimpl_int_return` / `opimpl_ref_return` / `opimpl_float_return`)
    // and `opimpl_void_return` → MetaInterp.finishframe.
    //
    // The dispatch JitCode body emits these as either:
    //   * sub-JitCode body terminator (e.g. a `RETURN` arm with
    //     `return state.regs[r]` lowered by `lower_dispatch_chain`'s
    //     Lowerable arm path; the sub-frame was pushed by the
    //     preceding BC_INLINE_CALL — `inline_frame=true`, no
    //     jitdriver_sd, return_i/r/f filled by the caller's
    //     destination slot).  On return: pop sub-frame, write
    //     result into caller's slot via make_result_of_lastop
    //     (pyjitpl.py).
    //   * dispatch body trailing terminator (lower_dispatch_body
    //     :5466-5499, "default arm typed return"); when this fires,
    //     the framestack drains to empty and we emit
    //     TraceAction::Finish so the outer `finish_and_compile`
    //     (jitdriver.rs::merge_point) drives the compile path —
    //     same precedent as the exception unwind at :935-942.
    //
    // last_exc_value clearing mirrors pyjitpl.py finishframe
    // (Pyre `clear_exception` is the JitCodeMachine equivalent of
    // RPython `self.last_exc_value = lltype.nullptr(...)`).
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_int_return(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.clear_exception();
        let src = self.frames.current_mut().next_reg() as usize;
        let (opref, concrete) = self.read_int_reg(ctx, src);
        let target = self.frames.current_mut().return_i;
        if target.is_none() {
            self.capture_single_pass_finish(ctx, Some(Value::Int(concrete)));
        }
        if self.frames.frames.len() == 1 {
            self.stash_portal_reds(ctx, sym);
        }
        if let Some(snapshot) = self.pop_exception_frame(ctx) {
            sym.restore_inline_scalar_state(snapshot);
        }
        if let Some(target_idx) = target {
            debug_assert!(
                !self.frames.is_empty(),
                "BC_INT_RETURN with return_i=Some but framestack drained",
            );
            self.frames.current_mut().make_result_of_lastop(
                JitArgKind::Int,
                target_idx,
                opref,
                concrete,
            );
        } else if self.frames.is_empty() {
            return TraceAction::Finish {
                finish_args: vec![opref],
                finish_arg_types: vec![majit_ir::Type::Int],
                exit_with_exception: false,
                exc_value: 0,
            };
        } else {
            typed_return_without_caller_destination("BC_INT_RETURN", 'i');
        }
        TraceAction::Continue
    }

    // `int_return/c` — USE_C_FORM short source (`assembler.py`):
    // the return value is one inline signed byte (`signedord`,
    // `blackhole.py`), not a `registers_i` slot. Otherwise
    // identical teardown to `BC_INT_RETURN`.
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_int_return_c(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.clear_exception();
        let value = self.frames.current_mut().next_u8() as i8 as i64;
        let opref = OpRef::ConstInt(value);
        let target = self.frames.current_mut().return_i;
        if target.is_none() {
            self.capture_single_pass_finish(ctx, Some(Value::Int(value)));
        }
        if self.frames.frames.len() == 1 {
            self.stash_portal_reds(ctx, sym);
        }
        if let Some(snapshot) = self.pop_exception_frame(ctx) {
            sym.restore_inline_scalar_state(snapshot);
        }
        if let Some(target_idx) = target {
            debug_assert!(
                !self.frames.is_empty(),
                "BC_INT_RETURN_C with return_i=Some but framestack drained",
            );
            self.frames.current_mut().make_result_of_lastop(
                JitArgKind::Int,
                target_idx,
                opref,
                value,
            );
        } else if self.frames.is_empty() {
            return TraceAction::Finish {
                finish_args: vec![opref],
                finish_arg_types: vec![majit_ir::Type::Int],
                exit_with_exception: false,
                exc_value: 0,
            };
        } else {
            typed_return_without_caller_destination("BC_INT_RETURN_C", 'i');
        }
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_ref_return(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.clear_exception();
        let src = self.frames.current_mut().next_reg() as usize;
        let (opref, concrete) = self.read_ref_reg(ctx, src);
        let target = self.frames.current_mut().return_r;
        if target.is_none() {
            self.capture_single_pass_finish(
                ctx,
                Some(Value::Ref(majit_ir::GcRef(concrete as usize))),
            );
        }
        if self.frames.frames.len() == 1 {
            self.stash_portal_reds(ctx, sym);
        }
        if let Some(snapshot) = self.pop_exception_frame(ctx) {
            sym.restore_inline_scalar_state(snapshot);
        }
        if let Some(target_idx) = target {
            debug_assert!(
                !self.frames.is_empty(),
                "BC_REF_RETURN with return_r=Some but framestack drained",
            );
            self.frames.current_mut().make_result_of_lastop(
                JitArgKind::Ref,
                target_idx,
                opref,
                concrete,
            );
        } else if self.frames.is_empty() {
            return TraceAction::Finish {
                finish_args: vec![opref],
                finish_arg_types: vec![majit_ir::Type::Ref],
                exit_with_exception: false,
                exc_value: 0,
            };
        } else {
            typed_return_without_caller_destination("BC_REF_RETURN", 'r');
        }
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_float_return(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.clear_exception();
        let src = self.frames.current_mut().next_reg() as usize;
        let (opref, concrete) = self.read_float_reg(ctx, src);
        let target = self.frames.current_mut().return_f;
        if target.is_none() {
            self.capture_single_pass_finish(
                ctx,
                Some(Value::Float(f64::from_bits(concrete as u64))),
            );
        }
        if self.frames.frames.len() == 1 {
            self.stash_portal_reds(ctx, sym);
        }
        if let Some(snapshot) = self.pop_exception_frame(ctx) {
            sym.restore_inline_scalar_state(snapshot);
        }
        if let Some(target_idx) = target {
            debug_assert!(
                !self.frames.is_empty(),
                "BC_FLOAT_RETURN with return_f=Some but framestack drained",
            );
            self.frames.current_mut().make_result_of_lastop(
                JitArgKind::Float,
                target_idx,
                opref,
                concrete,
            );
        } else if self.frames.is_empty() {
            return TraceAction::Finish {
                finish_args: vec![opref],
                finish_arg_types: vec![majit_ir::Type::Float],
                exit_with_exception: false,
                exc_value: 0,
            };
        } else {
            typed_return_without_caller_destination("BC_FLOAT_RETURN", 'f');
        }
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_void_return(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.clear_exception();
        self.capture_single_pass_finish(ctx, None);
        if self.frames.frames.len() == 1 {
            self.stash_portal_reds(ctx, sym);
        }
        if let Some(snapshot) = self.pop_exception_frame(ctx) {
            sym.restore_inline_scalar_state(snapshot);
        }
        if self.frames.is_empty() {
            // pyjitpl.py compile_done_with_this_frame exits=[];
            // `done_with_this_frame_descr_from_types` (`pyjitpl.rs`)
            // maps empty finish_arg_types to Type::Void.
            return TraceAction::Finish {
                finish_args: vec![],
                finish_arg_types: vec![],
                exit_with_exception: false,
                exc_value: 0,
            };
        }
        // Sub-frame void return: caller resumes; nothing to write.
        TraceAction::Continue
    }

    // ── canonical *_v call family (Slices 1-2 of
    // pyre-call-family-canonical-migration.md) ──
    //
    // Byte layout matches `blackhole.rs`'s
    // `handler_residual_call_{r,ir,irf}_v`:
    //   funcptr_reg:u8 + (countI:u8 + regI×N) + (countR:u8 +
    //   regR×M) + (countF:u8 + regF×K, IRF only) + descr:u16.
    //
    // `funcptr_reg` is the post-regs constants-pool slot the
    // emitter projected concrete_ptr into (RPython
    // `assembler.py emit_const`). The `d` operand carries
    // the `BhCallDescr`; its `arg_classes` restores source
    // argument order from the grouped I/R/F lists, matching
    // `pyjitpl.py:_build_allboxes`.
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_residual_call_r_v(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let has_int = matches!(
            bytecode,
            jitcode::insns::BC_RESIDUAL_CALL_IR_V | jitcode::insns::BC_RESIDUAL_CALL_IRF_V
        );
        let has_float = bytecode == jitcode::insns::BC_RESIDUAL_CALL_IRF_V;

        let call_jitcode = self.frames.current_mut().jitcode.clone();
        let (target, args_i, args_r, args_f, calldescr, trace_descr) = {
            let frame = self.frames.current_mut();
            let funcptr_reg = frame.next_reg() as u16;
            let mut args_i: CallArgs = SmallVec::new();
            if has_int {
                let count = frame.next_u8() as usize;
                for _ in 0..count {
                    args_i.push(JitCallArg::int(frame.next_reg() as u16));
                }
            }
            let mut args_r: CallArgs = SmallVec::new();
            let count_r = frame.next_u8() as usize;
            for _ in 0..count_r {
                args_r.push(JitCallArg::reference(frame.next_reg() as u16));
            }
            let mut args_f: CallArgs = SmallVec::new();
            if has_float {
                let count = frame.next_u8() as usize;
                for _ in 0..count {
                    args_f.push(JitCallArg::float(frame.next_reg() as u16));
                }
            }
            let calldescr_idx = frame.next_u16();
            let calldescr = call_jitcode
                .descr_at(calldescr_idx as usize)
                .and_then(crate::jitcode::RuntimeBhDescr::as_bh_descr)
                .expect("BC_RESIDUAL_CALL_*_V descr is not BhDescr")
                .as_calldescr();
            let trace_descr = frame
                .jitcode
                .descr_at(calldescr_idx as usize)
                .and_then(crate::jitcode::RuntimeBhDescr::as_optimizer_descr)
                .cloned()
                .unwrap_or_else(|| crate::call_descr::call_descr_from_bh(&calldescr));
            let target = frame
                .jitcode
                .exec
                .call_descr_to_call_target
                .get(&calldescr_idx)
                .copied()
                .unwrap_or_else(|| {
                    let func = frame
                        .int_regs
                        .get(funcptr_reg as usize)
                        .copied()
                        .flatten()
                        .and_then(|op| ctx.box_bits(op))
                        .unwrap_or_else(|| {
                            panic!(
                                "BC_RESIDUAL_CALL_*_V: funcptr slot \
                             {funcptr_reg} is uninitialized"
                            )
                        });
                    JitCallTarget::from_fnaddr(func)
                });
            (target, args_i, args_r, args_f, calldescr, trace_descr)
        };

        let (args, concrete_args, arg_types, raw_i, raw_r, raw_f) =
            self.read_canonical_call_args(ctx, &calldescr.arg_classes, &args_i, &args_r, &args_f);

        let trace_ptr = if target.trace_ptr.is_null() {
            target.concrete_ptr
        } else {
            target.trace_ptr
        };
        let concrete_ptr = if target.concrete_ptr.is_null() {
            trace_ptr
        } else {
            target.concrete_ptr
        };
        let fnaddr_word = target.fnaddr_for_symbolic_check(concrete_ptr);
        if let Some(action) = refuse_walk_local_ref_args(
            ctx,
            concrete_ptr as usize,
            &raw_i,
            &raw_r,
            &args,
            &calldescr.arg_classes,
        ) {
            return action;
        }

        let effect_descr = trace_descr.clone();
        let effectinfo = effect_descr
            .as_call_descr()
            .expect("resolved call descriptor")
            .get_extra_info();

        // pyjitpl.py do_not_in_trace_call parity:
        // `@not_in_trace`-decorated callees execute but are not
        // recorded in the trace IR. For void result this means
        // dispatch the C function and skip the
        // `ctx.call_*_void_typed` recording + `may_force` vable
        // bookkeeping (`forces` is incompatible with
        // `NotInTrace`). If the call raises, abort the trace —
        // PyPy raises `SwitchToBlackhole(ABORT_ESCAPE,
        // raising_exception=True)` which `TraceAction::Abort`
        // mirrors.
        //
        // `MetaInterp::do_not_in_trace_call` (`pyjitpl.rs`)
        // is the same logic on the `MetaInterp` side; the
        // `JitCodeMachine` walker does not currently hold a
        // `MetaInterp` reference, so the clear / dispatch /
        // exception-check sequence is replicated inline using
        // the shared `BH_LAST_EXC_VALUE` TLS already used by
        // other dispatch sites in `blackhole.rs`.
        if effectinfo.oopspecindex == majit_ir::descr::OopSpecIndex::NotInTrace {
            self.clear_exception();
            if majit_jitcode::codewriter::call::is_symbolic_fnaddr(fnaddr_word) {
                return report_symbolic_residual_call_target(
                    ctx,
                    fnaddr_word,
                    Some(&calldescr.arg_classes),
                );
            }
            if !concrete_ptr.is_null() {
                unsafe {
                    majit_backend::call_stub::bh_call_v_by_classes(
                        concrete_ptr as usize,
                        &calldescr.arg_classes,
                        Some(&raw_i),
                        Some(&raw_r),
                        Some(&raw_f),
                    );
                }
            }
            if let Some(action) =
                host_requested_walk_abort(ctx, concrete_ptr as usize, &calldescr.arg_classes)
            {
                return action;
            }
            // `pyjitpl.py do_not_in_trace_call`:
            //     if self.last_exc_value:
            //         raise SwitchToBlackhole(Counters.ABORT_ESCAPE,
            //                                  raising_exception=True)
            // The exception value stays on `BH_LAST_EXC_VALUE`
            // for the blackhole replay; do not clear it here.
            // Mirror `finalize_standard_virtualizable_may_force`
            // (in this file) by stashing
            // `SwitchToBlackhole::abort_escape()` on TraceCtx so
            // the jitdriver-side `TraceAction::Abort` consumer
            // fires `aborted_tracing(ABORT_ESCAPE)` instead of
            // the generic too-long fallback.
            let exc = crate::blackhole::BH_LAST_EXC_VALUE.with(|c| c.get());
            if exc != 0 {
                return TraceAction::SwitchToBlackhole(
                    crate::pyjitpl::SwitchToBlackhole::abort_escape(),
                );
            }
        } else {
            // `pyjitpl.py do_residual_call`'s `OS_LIBFFI_CALL` hook answers
            // `None  # cannot be handled by direct_libffi_call()` on this
            // layer: rebuilding the call out of its `CIF_DESCRIPTION`
            // needs a `MetaInterp`, which this jitcode machine does not
            // hold.  That is upstream's own fallthrough — the release-gil
            // / may-force selection below is what it falls through to.
            // The specialization lives in
            // `MetaInterp::direct_libffi_call` and in the pyre-jit-trace
            // walker's residual-call dispatchers.
            let is_release_gil = effectinfo.is_call_release_gil();
            let is_forces = effectinfo.check_forces_virtual_or_virtualizable();
            let is_loopinvariant =
                effectinfo.extraeffect == majit_ir::descr::ExtraEffect::LoopInvariant;

            // pyjitpl.py execute_varargs parity (plain
            // CALL_N / LOOPINVARIANT_N branch) and pyjitpl.py
            // (MAY_FORCE_N branch).  Both branches share the same
            // first step: `clear_exception()` BEFORE
            // `vable_and_vrefs_before_residual_call`.  The full
            // RPython sequence (executes_varargs helper) is
            //     clear_exception
            //     execute_and_record_varargs        # execute → record
            //     handle_possible_exception / assert_no_exception
            // (`execute_and_record_varargs` runs `executor.execute_varargs`
            // first, then `history.record`). Concrete execute is
            // therefore observed BEFORE the trace IR is written and
            // the post-call exception check decides between
            // GUARD_NO_EXCEPTION and the catch path.
            //
            // 1. clear_exception (`pyjitpl.py` /
            //    `JitCodeMachine::clear_exception` in this file).
            //    PyPy's `clear_exception()`
            //    nulls `self.last_exc_value`.  Pyre's parity
            //    (Parity #10) clears `last_exception_value` and the
            //    `BH_LAST_EXC_VALUE` TLS shim used by
            //    `bh_call_*_dispatch` (`call_stub::bh_call_*`) — the
            //    TLS shim is pyre's structural adapter for surfacing
            //    a callee's exception across the C boundary.
            //    `last_exception_box` is intentionally left untouched
            //    matching upstream: `handle_possible_exception`
            //    overwrites it whenever `last_exc_value` becomes
            //    non-NULL again, and every reader gates on
            //    `last_exc_value` first.
            self.clear_exception();
            // A target we will not call must not stamp force tokens.
            // `aborted_tracing` leaves `TOKEN_TRACING_RESCALL` set.
            if majit_jitcode::codewriter::call::is_symbolic_fnaddr(fnaddr_word) {
                return report_symbolic_residual_call_target(
                    ctx,
                    fnaddr_word,
                    Some(&calldescr.arg_classes),
                );
            }
            if concrete_ptr.is_null() {
                return refuse_null_residual_call_target(ctx, &calldescr.arg_classes);
            }
            // pyjitpl.py `vable_and_vrefs_before_residual_call`
            // walks vrefs FIRST (stamps TOKEN_TRACING_RESCALL), then the
            // virtualizable.  Pyre splits the call into
            // `ctx.vrefs_before_residual_call()` + the vinfo branch in
            // `prepare_standard_virtualizable_before_residual_call`.
            // Without the vrefs stamp, `vrefs_after_residual_call`
            // misreads a fresh vref's `TOKEN_NONE` as "forced" and
            // wrongly emits `VIRTUAL_REF_FINISH` + `CONST_NULL`.
            let active_vable = if is_forces {
                ctx.vrefs_before_residual_call();
                self.prepare_standard_virtualizable_before_residual_call(ctx)
            } else {
                None
            };
            // 2. concrete execute (RPython `executor.execute_varargs`
            //    → `cpu.bh_call_v`).  llmodel.py bh_call_v: a
            //    genuinely void C callee returns nothing, so route
            //    through the void-typed dispatcher instead of
            //    `bh_call_i_dispatch` (which transmutes to
            //    `extern "C" fn(...) -> i64` and reads garbage from
            //    rax/x0). The null target already returned.
            unsafe {
                majit_backend::call_stub::bh_call_v_by_classes(
                    concrete_ptr as usize,
                    &calldescr.arg_classes,
                    Some(&raw_i),
                    Some(&raw_r),
                    Some(&raw_f),
                );
            }
            if let Some(action) =
                host_requested_walk_abort(ctx, concrete_ptr as usize, &calldescr.arg_classes)
            {
                return action;
            }
            // pyjitpl.py MIFrame.do_residual_call — after the residual call,
            // walk the vrefs.  If any were forced by the call
            // then VIRTUAL_REF_FINISH is recorded BEFORE any
            // CALL op is recorded.  RPython's `MetaInterp`
            // owns `virtualref_boxes`; pyre's per-trace
            // counterpart is on `TraceCtx` so the state-field
            // dispatch can reach the same data through `ctx`.
            //
            // Gated on `is_forces` because pyjitpl.py
            // runs `vable_and_vrefs_before_residual_call` +
            // `vrefs_after_residual_call` only inside the
            // `assembler_call or check_forces_virtual_or_virtualizable()`
            // branch.  The plain / release_gil / loopinvariant
            // path (pyjitpl.py) goes through
            // `execute_varargs` which never invokes either
            // hook — calling the after-hook there would see
            // tokens still at TOKEN_NONE (because the before-hook
            // never stamped TOKEN_TRACING_RESCALL) and
            // incorrectly record VIRTUAL_REF_FINISH for live
            // vrefs.
            if is_forces {
                ctx.vrefs_after_residual_call();
            }
            // 3. record IR (`history.record` →
            //    `_record_helper_varargs`). pyjitpl.py
            //    do_residual_call threads the original calldescr's
            //    `EffectInfo` (oopspec, read/write descr sets,
            //    can_invalidate, can_collect,
            //    call_release_gil_target) into the trace IR
            //    instead of re-deriving the default for the opcode.
            //    The OS_LIBFFI_CALL pre-hook in
            //    `pyjitpl.py do_residual_call` declines at the top of
            //    this branch, so the calldescr reaching here is the
            //    original one.
            if is_release_gil {
                ctx.call_release_gil_void_typed_with_effect(
                    trace_ptr,
                    &args,
                    &arg_types,
                    effectinfo.clone(),
                );
            } else if is_forces {
                ctx.call_may_force_void_typed_with_effect(
                    trace_ptr,
                    &args,
                    &arg_types,
                    effectinfo.clone(),
                );
            } else if is_loopinvariant {
                // pyjitpl.py MIFrame.do_residual_call with tp == 'v':
                // _record_helper_varargs returns None for void,
                // so the loop-invariant cache always misses for
                // void calls — concrete C dispatch always runs.
                ctx.call_loopinvariant_void_typed_with_effect(
                    trace_ptr,
                    &args,
                    &arg_types,
                    effectinfo.clone(),
                );
            } else {
                ctx.record_call_with_descr(majit_ir::OpCode::CallN, trace_ptr, &args, trace_descr);
            }
            // 4. for forces: `vable_after_residual_call` +
            //    `generate_guard(GUARD_NOT_FORCED)` (`pyjitpl.py`).
            //    Pyre rolls both into
            //    `finalize_standard_virtualizable_may_force` which
            //    emits `GuardNotForced` before
            //    `handle_possible_exception` runs below — the
            //    upstream order is GUARD_NOT_FORCED first, then
            //    GUARD_NO_EXCEPTION.
            if is_forces {
                let action = self.finalize_standard_virtualizable_may_force(ctx, sym, active_vable);
                if !matches!(action, TraceAction::Continue) {
                    return action;
                }
            }
            match self.finish_residual_call_exception_path(ctx, sym, effectinfo) {
                TraceAction::Continue => {}
                action => return action,
            }
        }
        TraceAction::Continue
    }

    // ── 0: canonical typed (i/r/f) recording arms ──
    //
    // Mirror of the void arm above for the int / ref / float
    // result kinds.  RPython's `pyjitpl.py do_residual_call
    // do_residual_call` is one function dispatching by `tp ==
    // 'i'/'r'/'f'/'v'` inline; pyre splits that across match
    // arms (necessary Rust adaptation: match-arm vs Python
    // `tp` branching).  Each arm replicates the void body line
    // for line, with only these per-kind differences:
    //   * dispatcher: `bh_call_v_dispatch` →
    //     `bh_call_i_dispatch` (int / ref) or
    //     `bh_call_f_dispatch` (float)
    //   * register write-back: `set_int_reg` / `set_ref_reg` /
    //     `set_float_reg` after both NotInTrace-execute and the
    //     normal-execute-and-record paths
    //   * record API: `call_*_typed_with_effect` returning an
    //     `OpRef` (paired with the concrete result for the
    //     register write-back)
    //   * float layout: `BC_RESIDUAL_CALL_IRF_F` is the only
    //     float-result opcode per `resoperation.py call_release_gil_for_descr`
    //     ("no such thing" `R_F` / `IR_F`), so the float arm
    //     reads all three (count, regs) pairs unconditionally —
    //     matching `emit_canonical_call_typed_irf_f`
    //     (jitcode/assembler.rs) which always emits them.
    //
    // These arms are dormant until the producer migration in
    // `pyre/pyre-jit/src/jit/assembler.rs::dispatch_residual_call`
    // typed branch routes through `*_canonical_via_target_with_effect_info`.
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_residual_call_r_i(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let has_int = matches!(
            bytecode,
            jitcode::insns::BC_RESIDUAL_CALL_IR_I | jitcode::insns::BC_RESIDUAL_CALL_IRF_I
        );
        let has_float = bytecode == jitcode::insns::BC_RESIDUAL_CALL_IRF_I;

        let call_jitcode = self.frames.current_mut().jitcode.clone();
        let (target, args_i, args_r, args_f, calldescr, trace_descr, dst) = {
            let frame = self.frames.current_mut();
            let funcptr_reg = frame.next_reg() as u16;
            let mut args_i: CallArgs = SmallVec::new();
            if has_int {
                let count = frame.next_u8() as usize;
                for _ in 0..count {
                    args_i.push(JitCallArg::int(frame.next_reg() as u16));
                }
            }
            let mut args_r: CallArgs = SmallVec::new();
            let count_r = frame.next_u8() as usize;
            for _ in 0..count_r {
                args_r.push(JitCallArg::reference(frame.next_reg() as u16));
            }
            let mut args_f: CallArgs = SmallVec::new();
            if has_float {
                let count = frame.next_u8() as usize;
                for _ in 0..count {
                    args_f.push(JitCallArg::float(frame.next_reg() as u16));
                }
            }
            let calldescr_idx = frame.next_u16();
            let dst = frame.next_reg() as usize;
            let calldescr = call_jitcode
                .descr_at(calldescr_idx as usize)
                .and_then(crate::jitcode::RuntimeBhDescr::as_bh_descr)
                .expect("BC_RESIDUAL_CALL_*_I descr is not BhDescr")
                .as_calldescr();
            let trace_descr = frame
                .jitcode
                .descr_at(calldescr_idx as usize)
                .and_then(crate::jitcode::RuntimeBhDescr::as_optimizer_descr)
                .cloned()
                .unwrap_or_else(|| crate::call_descr::call_descr_from_bh(&calldescr));
            let target = frame
                .jitcode
                .exec
                .call_descr_to_call_target
                .get(&calldescr_idx)
                .copied()
                .unwrap_or_else(|| {
                    let func = frame
                        .int_regs
                        .get(funcptr_reg as usize)
                        .copied()
                        .flatten()
                        .and_then(|op| ctx.box_bits(op))
                        .unwrap_or_else(|| {
                            panic!(
                                "BC_RESIDUAL_CALL_*_I: funcptr slot \
                                 {funcptr_reg} is uninitialized"
                            )
                        });
                    JitCallTarget::from_fnaddr(func)
                });
            (target, args_i, args_r, args_f, calldescr, trace_descr, dst)
        };

        let (args, concrete_args, arg_types, raw_i, raw_r, raw_f) =
            self.read_canonical_call_args(ctx, &calldescr.arg_classes, &args_i, &args_r, &args_f);

        let trace_ptr = if target.trace_ptr.is_null() {
            target.concrete_ptr
        } else {
            target.trace_ptr
        };
        let concrete_ptr = if target.concrete_ptr.is_null() {
            trace_ptr
        } else {
            target.concrete_ptr
        };
        let fnaddr_word = target.fnaddr_for_symbolic_check(concrete_ptr);
        if let Some(action) = refuse_walk_local_ref_args(
            ctx,
            concrete_ptr as usize,
            &raw_i,
            &raw_r,
            &args,
            &calldescr.arg_classes,
        ) {
            return action;
        }

        let effect_descr = trace_descr.clone();
        let effectinfo = effect_descr
            .as_call_descr()
            .expect("resolved call descriptor")
            .get_extra_info();

        if effectinfo.oopspecindex == majit_ir::descr::OopSpecIndex::NotInTrace {
            // pyjitpl.py do_not_in_trace_call:
            //     self.clear_exception()
            //     executor.execute_varargs(self.cpu, self,
            //                              rop.CALL_N, allboxes, descr)
            //     if self.last_exc_value:
            //         raise SwitchToBlackhole(ABORT_ESCAPE,
            //                                 raising_exception=True)
            //     return None
            //
            // RPython forces the dispatch through `CALL_N` (void)
            // regardless of the surface result type and discards
            // the result.  Mirror that here: route the C call
            // through `bh_call_v_dispatch`, do not write back the
            // int destination register, and abort on exception.
            self.clear_exception();
            if majit_jitcode::codewriter::call::is_symbolic_fnaddr(fnaddr_word) {
                return report_symbolic_residual_call_target(
                    ctx,
                    fnaddr_word,
                    Some(&calldescr.arg_classes),
                );
            }
            if !concrete_ptr.is_null() {
                unsafe {
                    majit_backend::call_stub::bh_call_v_by_classes(
                        concrete_ptr as usize,
                        &calldescr.arg_classes,
                        Some(&raw_i),
                        Some(&raw_r),
                        Some(&raw_f),
                    );
                }
            }
            if let Some(action) =
                host_requested_walk_abort(ctx, concrete_ptr as usize, &calldescr.arg_classes)
            {
                return action;
            }
            // `pyjitpl.py do_not_in_trace_call`:
            //     if self.last_exc_value: raise SwitchToBlackhole(
            //         Counters.ABORT_ESCAPE, raising_exception=True)
            // Same stash pattern as the void OS_NOT_IN_TRACE arm
            // above.
            let exc = crate::blackhole::BH_LAST_EXC_VALUE.with(|c| c.get());
            if exc != 0 {
                return TraceAction::SwitchToBlackhole(
                    crate::pyjitpl::SwitchToBlackhole::abort_escape(),
                );
            }
            let _ = dst;
        } else {
            // `pyjitpl.py do_residual_call`'s `OS_LIBFFI_CALL` hook answers
            // `None  # cannot be handled by direct_libffi_call()` on this
            // layer: rebuilding the call out of its `CIF_DESCRIPTION`
            // needs a `MetaInterp`, which this jitcode machine does not
            // hold.  That is upstream's own fallthrough — the release-gil
            // / may-force selection below is what it falls through to.
            // The specialization lives in
            // `MetaInterp::direct_libffi_call` and in the pyre-jit-trace
            // walker's residual-call dispatchers.
            let is_release_gil = effectinfo.is_call_release_gil();
            let is_forces = effectinfo.check_forces_virtual_or_virtualizable();
            let is_loopinvariant =
                effectinfo.extraeffect == majit_ir::descr::ExtraEffect::LoopInvariant;

            // pyjitpl.py do_residual_call:
            //     res = self.metainterp.heapcache
            //         .call_loopinvariant_known_result(allboxes, descr)
            //     if res is not None:
            //         return res
            // Hit on the loop-invariant cache returns the cached
            // result WITHOUT executing the C call or recording a
            // trace op.  Pyre's helper
            // (`TraceCtx::call_loopinvariant_lookup_with_effect` in
            // `history.rs`) does
            // the lookup internally on the record-side, but the
            // concrete `bh_call_i_dispatch` below still ran first
            // — splitting the lookup out matches upstream order.
            if is_loopinvariant
                && let Some((cached_traced, cached_concrete)) = ctx
                    .call_loopinvariant_lookup_with_effect(
                        trace_ptr,
                        &arg_types,
                        majit_ir::Type::Int,
                        effectinfo,
                    )
            {
                self.set_int_reg(ctx, dst, Some(cached_traced), Some(cached_concrete));
                return TraceAction::Continue;
            }

            // pyjitpl.py MIFrame.do_residual_call MAY_FORCE_I branch parity:
            //     clear_exception  ← FIRST
            //     vable_and_vrefs_before_residual_call
            // (vrefs walk + vinfo stamp; see void arm for full citation).
            // Decline a target we will not call before that stamp.
            self.clear_exception();
            if majit_jitcode::codewriter::call::is_symbolic_fnaddr(fnaddr_word) {
                return report_symbolic_residual_call_target(
                    ctx,
                    fnaddr_word,
                    Some(&calldescr.arg_classes),
                );
            }
            if concrete_ptr.is_null() {
                return refuse_null_residual_call_target(ctx, &calldescr.arg_classes);
            }
            let active_vable = if is_forces {
                ctx.vrefs_before_residual_call();
                self.prepare_standard_virtualizable_before_residual_call(ctx)
            } else {
                None
            };
            // Concrete execute via `bh_call_i_dispatch` (i64
            // return) — RPython `executor.execute_varargs` →
            // `cpu.bh_call_i`. The null target already returned.
            let concrete = unsafe {
                majit_backend::call_stub::bh_call_i_by_classes(
                    concrete_ptr as usize,
                    &calldescr.arg_classes,
                    Some(&raw_i),
                    Some(&raw_r),
                    Some(&raw_f),
                )
            };
            if let Some(action) =
                host_requested_walk_abort(ctx, concrete_ptr as usize, &calldescr.arg_classes)
            {
                return action;
            }
            // pyjitpl.py — vrefs_after_residual_call
            // (see void arm for the explanation; gated on
            // `is_forces` because the before-hook only stamps
            // TOKEN_TRACING_RESCALL in that branch).
            if is_forces {
                ctx.vrefs_after_residual_call();
            }
            // pyjitpl.py do_residual_call plain branch:
            //     pure = effectinfo.check_is_elidable()
            //     return self.execute_varargs(rop.CALL_I,
            //                                 allboxes, descr,
            //                                 exc, pure)
            // Pure calls fold after recording via record_result_of_call_pure.
            // (pyjitpl.py).  Only the plain branch carries
            // pure: forces/release_gil/loopinvariant don't combine
            // with elidable in upstream call.py getcalldescr.
            let plain_branch = !is_release_gil && !is_forces && !is_loopinvariant;
            let pure = plain_branch && effectinfo.check_is_elidable();
            let patch_pos = if pure {
                Some(ctx.get_trace_position())
            } else {
                None
            };
            let traced = if is_release_gil {
                ctx.call_release_gil_int_typed_with_effect(
                    trace_ptr,
                    &args,
                    &arg_types,
                    effectinfo.clone(),
                )
            } else if is_forces {
                ctx.call_may_force_int_typed_with_effect(
                    trace_ptr,
                    &args,
                    &arg_types,
                    effectinfo.clone(),
                )
            } else if is_loopinvariant {
                ctx.call_loopinvariant_int_typed_with_effect(
                    trace_ptr,
                    &args,
                    &arg_types,
                    effectinfo.clone(),
                    concrete,
                )
            } else {
                ctx.record_call_with_descr(
                    majit_ir::OpCode::CallI,
                    trace_ptr,
                    &args,
                    trace_descr.clone(),
                )
            };
            // pyjitpl.py execute_varargs:
            //     if pure and not self.metainterp.last_exc_value and op:
            //         op = self.metainterp.record_result_of_call_pure(...)
            // The post-record CALL_I → CALL_PURE_I cut + const fold
            // (`pyjitpl.py`) only fires when the concrete
            // callee did NOT raise.  A raising-pure leaves the
            // recorded CALL_I uncut, and finish_residual_call_exception_path
            // below emits GUARD_EXCEPTION + unwinds.
            let last_exc_value = crate::blackhole::BH_LAST_EXC_VALUE.with(|c| c.get());
            let traced = match patch_pos {
                Some(patch_pos) if last_exc_value == 0 => {
                    let func_ref = ctx.const_int(trace_ptr as usize as i64);
                    let mut call_args: CallOpRefs = SmallVec::new();
                    call_args.push(func_ref);
                    call_args.extend_from_slice(&args);
                    let concrete_values =
                        build_concrete_values(trace_ptr, &concrete_args, &arg_types);
                    ctx.record_result_of_call_pure(
                        traced,
                        &call_args,
                        &concrete_values,
                        trace_descr,
                        patch_pos,
                        majit_ir::OpCode::CallI,
                        majit_ir::Value::Int(concrete),
                    )
                }
                _ => traced,
            };
            // RPython pyjitpl.py opimpl_residual_call_*_may_force_*
            // writes the call result into the frame *before*
            // vable_after_residual_call fires GUARD_NOT_FORCED
            // (see legacy BC_CALL_MAY_FORCE_INT arm for rationale).
            // `pyjitpl.py execute_and_record_varargs` runs the call
            // through `executor.execute_varargs` and hands the result
            // to `history.record_nospec`, so the recorded op carries
            // the executed value on its own frontend slot -- every
            // later `getvalue()` of that box answers it.  Writing the
            // value into the destination register alone leaves
            // `concrete_of_opref` answering `None` for the box, and
            // the two readers then disagree: `_nonstandard_virtualizable`
            // asks the box, so a residual that returns the standard
            // virtualizable (the portal's `reload_top_root`) loses its
            // PTR_EQ against `virtualizable_boxes[-1]` and every later
            // vable access on that register takes the nonstandard leg.
            // The full-body walker already stamps its own residual
            // results this way (`jitcode_dispatch/residual_call.rs`).
            ctx.set_opref_concrete(traced, majit_ir::Value::Int(concrete));
            self.set_int_reg(ctx, dst, Some(traced), Some(concrete));
            if is_forces {
                if crate::majit_log_enabled() {
                    let frame = self.frames.current_mut();
                    eprintln!(
                        "[interpret] residual may_force jitcode={} last_op={} cursor={} \
                         extraeffect={:?} can_raise={} next={:?}",
                        frame.jitcode.name(),
                        frame.last_opcode_position,
                        frame.code_cursor,
                        effectinfo.extraeffect,
                        effectinfo.check_can_raise(false),
                        frame.jitcode.code.get(frame.code_cursor),
                    );
                }
                let action = self.finalize_standard_virtualizable_may_force(ctx, sym, active_vable);
                if !matches!(action, TraceAction::Continue) {
                    return action;
                }
            }
            // pyjitpl.py `exc = exc and not isinstance(op, Const)`:
            // a pure call that const-folded clears `exc`, so
            // `assert_no_exception` runs (no GUARD_NO_EXCEPTION emit).
            // finish_residual_call_exception_path's `assert!(exc == 0)`
            // covers that.  When pure didn't fold, the fresh CALL_PURE
            // op still needs GUARD_NO_EXCEPTION/EXCEPTION based on
            // effectinfo.check_can_raise() — same as non-pure.
            if !(pure && traced.is_constant()) {
                match self.finish_residual_call_exception_path(ctx, sym, effectinfo) {
                    TraceAction::Continue => {}
                    action => return action,
                }
            }
        }
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_residual_call_r_r(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let has_int = matches!(
            bytecode,
            jitcode::insns::BC_RESIDUAL_CALL_IR_R | jitcode::insns::BC_RESIDUAL_CALL_IRF_R
        );
        let has_float = bytecode == jitcode::insns::BC_RESIDUAL_CALL_IRF_R;

        let call_jitcode = self.frames.current_mut().jitcode.clone();
        let (target, args_i, args_r, args_f, calldescr, trace_descr, dst) = {
            let frame = self.frames.current_mut();
            let funcptr_reg = frame.next_reg() as u16;
            let mut args_i: CallArgs = SmallVec::new();
            if has_int {
                let count = frame.next_u8() as usize;
                for _ in 0..count {
                    args_i.push(JitCallArg::int(frame.next_reg() as u16));
                }
            }
            let mut args_r: CallArgs = SmallVec::new();
            let count_r = frame.next_u8() as usize;
            for _ in 0..count_r {
                args_r.push(JitCallArg::reference(frame.next_reg() as u16));
            }
            let mut args_f: CallArgs = SmallVec::new();
            if has_float {
                let count = frame.next_u8() as usize;
                for _ in 0..count {
                    args_f.push(JitCallArg::float(frame.next_reg() as u16));
                }
            }
            let calldescr_idx = frame.next_u16();
            let dst = frame.next_reg() as usize;
            let calldescr = call_jitcode
                .descr_at(calldescr_idx as usize)
                .and_then(crate::jitcode::RuntimeBhDescr::as_bh_descr)
                .expect("BC_RESIDUAL_CALL_*_R descr is not BhDescr")
                .as_calldescr();
            let trace_descr = frame
                .jitcode
                .descr_at(calldescr_idx as usize)
                .and_then(crate::jitcode::RuntimeBhDescr::as_optimizer_descr)
                .cloned()
                .unwrap_or_else(|| crate::call_descr::call_descr_from_bh(&calldescr));
            let target = frame
                .jitcode
                .exec
                .call_descr_to_call_target
                .get(&calldescr_idx)
                .copied()
                .unwrap_or_else(|| {
                    let func = frame
                        .int_regs
                        .get(funcptr_reg as usize)
                        .copied()
                        .flatten()
                        .and_then(|op| ctx.box_bits(op))
                        .unwrap_or_else(|| {
                            panic!(
                                "BC_RESIDUAL_CALL_*_R: funcptr slot \
                                 {funcptr_reg} is uninitialized"
                            )
                        });
                    JitCallTarget::from_fnaddr(func)
                });
            (target, args_i, args_r, args_f, calldescr, trace_descr, dst)
        };

        let (args, concrete_args, arg_types, raw_i, raw_r, raw_f) =
            self.read_canonical_call_args(ctx, &calldescr.arg_classes, &args_i, &args_r, &args_f);

        let trace_ptr = if target.trace_ptr.is_null() {
            target.concrete_ptr
        } else {
            target.trace_ptr
        };
        let concrete_ptr = if target.concrete_ptr.is_null() {
            trace_ptr
        } else {
            target.concrete_ptr
        };
        let fnaddr_word = target.fnaddr_for_symbolic_check(concrete_ptr);
        if let Some(action) = refuse_walk_local_ref_args(
            ctx,
            concrete_ptr as usize,
            &raw_i,
            &raw_r,
            &args,
            &calldescr.arg_classes,
        ) {
            return action;
        }

        let effect_descr = trace_descr.clone();
        let effectinfo = effect_descr
            .as_call_descr()
            .expect("resolved call descriptor")
            .get_extra_info();

        if effectinfo.oopspecindex == majit_ir::descr::OopSpecIndex::NotInTrace {
            // pyjitpl.py do_not_in_trace_call: route the
            // C call through `CALL_N` (void) and discard the
            // result regardless of the surface result type.  See
            // the int sibling at the corresponding NotInTrace
            // branch for the full citation.
            self.clear_exception();
            if majit_jitcode::codewriter::call::is_symbolic_fnaddr(fnaddr_word) {
                return report_symbolic_residual_call_target(
                    ctx,
                    fnaddr_word,
                    Some(&calldescr.arg_classes),
                );
            }
            if !concrete_ptr.is_null() {
                unsafe {
                    majit_backend::call_stub::bh_call_v_by_classes(
                        concrete_ptr as usize,
                        &calldescr.arg_classes,
                        Some(&raw_i),
                        Some(&raw_r),
                        Some(&raw_f),
                    );
                }
            }
            if let Some(action) =
                host_requested_walk_abort(ctx, concrete_ptr as usize, &calldescr.arg_classes)
            {
                return action;
            }
            // `pyjitpl.py do_not_in_trace_call`:
            //     if self.last_exc_value: raise SwitchToBlackhole(
            //         Counters.ABORT_ESCAPE, raising_exception=True)
            // Same stash pattern as the void OS_NOT_IN_TRACE arm
            // above.
            let exc = crate::blackhole::BH_LAST_EXC_VALUE.with(|c| c.get());
            if exc != 0 {
                return TraceAction::SwitchToBlackhole(
                    crate::pyjitpl::SwitchToBlackhole::abort_escape(),
                );
            }
            let _ = dst;
        } else {
            // ResKind::Ref intentionally rejects ReleaseGil per
            // `resoperation.py rop.call_release_gil_for_descr # no such thing`. The
            // producer rejects this combination at
            // `pyre-jit/src/jit/assembler.rs`'s
            // `dispatch_residual_call` so the
            // recorder treats it as an unreachable invariant.
            if effectinfo.is_call_release_gil() {
                panic!(
                    "BC_RESIDUAL_CALL_*_R: ReleaseGil + Ref has no upstream counterpart \
                     (resoperation.py:1243-1244 `# no such thing`)"
                );
            }
            // `pyjitpl.py do_residual_call`'s `OS_LIBFFI_CALL` hook answers
            // `None  # cannot be handled by direct_libffi_call()` on this
            // layer: rebuilding the call out of its `CIF_DESCRIPTION`
            // needs a `MetaInterp`, which this jitcode machine does not
            // hold.  That is upstream's own fallthrough — the release-gil
            // / may-force selection below is what it falls through to.
            // The specialization lives in
            // `MetaInterp::direct_libffi_call` and in the pyre-jit-trace
            // walker's residual-call dispatchers.
            let is_forces = effectinfo.check_forces_virtual_or_virtualizable();
            let is_loopinvariant =
                effectinfo.extraeffect == majit_ir::descr::ExtraEffect::LoopInvariant;

            // pyjitpl.py MIFrame.do_residual_call: heapcache lookup-first for
            // loop-invariant calls (see int sibling for full cite).
            if is_loopinvariant
                && let Some((cached_traced, cached_concrete)) = ctx
                    .call_loopinvariant_lookup_with_effect(
                        trace_ptr,
                        &arg_types,
                        majit_ir::Type::Ref,
                        effectinfo,
                    )
            {
                self.set_ref_reg(ctx, dst, Some(cached_traced), Some(cached_concrete));
                return TraceAction::Continue;
            }

            // pyjitpl.py MIFrame.do_residual_call MAY_FORCE_R branch parity:
            // clear_exception precedes vable_and_vrefs_before_residual_call
            // (vrefs walk + vinfo stamp; see void arm for full citation).
            // Decline a target we will not call before that stamp.
            self.clear_exception();
            if majit_jitcode::codewriter::call::is_symbolic_fnaddr(fnaddr_word) {
                return report_symbolic_residual_call_target(
                    ctx,
                    fnaddr_word,
                    Some(&calldescr.arg_classes),
                );
            }
            if concrete_ptr.is_null() {
                return refuse_null_residual_call_target(ctx, &calldescr.arg_classes);
            }
            let active_vable = if is_forces {
                ctx.vrefs_before_residual_call();
                self.prepare_standard_virtualizable_before_residual_call(ctx)
            } else {
                None
            };
            let concrete = unsafe {
                majit_backend::call_stub::bh_call_i_by_classes(
                    concrete_ptr as usize,
                    &calldescr.arg_classes,
                    Some(&raw_i),
                    Some(&raw_r),
                    Some(&raw_f),
                )
            };
            if let Some(action) =
                host_requested_walk_abort(ctx, concrete_ptr as usize, &calldescr.arg_classes)
            {
                return action;
            }
            // pyjitpl.py — vrefs_after_residual_call
            // (see void arm for the explanation; gated on
            // `is_forces` because the before-hook only stamps
            // TOKEN_TRACING_RESCALL in that branch).
            if is_forces {
                ctx.vrefs_after_residual_call();
            }
            // pyjitpl.py do_residual_call plain branch —
            // see the BC_RESIDUAL_CALL_*_I sibling for the full cite.
            let plain_branch = !is_forces && !is_loopinvariant;
            let pure = plain_branch && effectinfo.check_is_elidable();
            let patch_pos = if pure {
                Some(ctx.get_trace_position())
            } else {
                None
            };
            let traced = if is_forces {
                ctx.call_may_force_ref_typed_with_effect(
                    trace_ptr,
                    &args,
                    &arg_types,
                    effectinfo.clone(),
                )
            } else if is_loopinvariant {
                ctx.call_loopinvariant_ref_typed_with_effect(
                    trace_ptr,
                    &args,
                    &arg_types,
                    effectinfo.clone(),
                    concrete,
                )
            } else {
                ctx.record_call_with_descr(
                    majit_ir::OpCode::CallR,
                    trace_ptr,
                    &args,
                    trace_descr.clone(),
                )
            };
            // pyjitpl.py MIFrame.execute_varargs gate (see int sibling for full cite).
            let last_exc_value = crate::blackhole::BH_LAST_EXC_VALUE.with(|c| c.get());
            let traced = match patch_pos {
                Some(patch_pos) if last_exc_value == 0 => {
                    let func_ref = ctx.const_int(trace_ptr as usize as i64);
                    let mut call_args: CallOpRefs = SmallVec::new();
                    call_args.push(func_ref);
                    call_args.extend_from_slice(&args);
                    let concrete_values =
                        build_concrete_values(trace_ptr, &concrete_args, &arg_types);
                    ctx.record_result_of_call_pure(
                        traced,
                        &call_args,
                        &concrete_values,
                        trace_descr,
                        patch_pos,
                        majit_ir::OpCode::CallR,
                        majit_ir::Value::Ref(majit_ir::GcRef(concrete as usize)),
                    )
                }
                _ => traced,
            };
            // `pyjitpl.py execute_and_record_varargs` runs the call
            // through `executor.execute_varargs` and hands the result
            // to `history.record_nospec`, so the recorded op carries
            // the executed value on its own frontend slot -- every
            // later `getvalue()` of that box answers it.  Writing the
            // value into the destination register alone leaves
            // `concrete_of_opref` answering `None` for the box, and
            // the two readers then disagree: `_nonstandard_virtualizable`
            // asks the box, so a residual that returns the standard
            // virtualizable (the portal's `reload_top_root`) loses its
            // PTR_EQ against `virtualizable_boxes[-1]` and every later
            // vable access on that register takes the nonstandard leg.
            // The full-body walker already stamps its own residual
            // results this way (`jitcode_dispatch/residual_call.rs`).
            ctx.set_opref_concrete(
                traced,
                majit_ir::Value::Ref(majit_ir::GcRef(concrete as usize)),
            );
            self.set_ref_reg(ctx, dst, Some(traced), Some(concrete));
            if is_forces {
                let action = self.finalize_standard_virtualizable_may_force(ctx, sym, active_vable);
                if !matches!(action, TraceAction::Continue) {
                    return action;
                }
            }
            if !(pure && traced.is_constant()) {
                match self.finish_residual_call_exception_path(ctx, sym, effectinfo) {
                    TraceAction::Continue => {}
                    action => return action,
                }
            }
        }
        TraceAction::Continue
    }

    // `BC_RESIDUAL_CALL_IRF_F` is the only float-result opcode
    // (`resoperation.py call_release_gil_for_descr` # no such thing for `R_F` /
    // `IR_F`). The producer (`emit_canonical_call_typed_irf_f`
    // in `jitcode/assembler.rs`) always emits all three
    // (count, regs) pairs even when one is empty, so the
    // recorder reads them unconditionally.
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_residual_call_irf_f(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let call_jitcode = self.frames.current_mut().jitcode.clone();
        let (target, args_i, args_r, args_f, calldescr, trace_descr, dst) = {
            let frame = self.frames.current_mut();
            let funcptr_reg = frame.next_reg() as u16;
            let mut args_i: CallArgs = SmallVec::new();
            let count_i = frame.next_u8() as usize;
            for _ in 0..count_i {
                args_i.push(JitCallArg::int(frame.next_reg() as u16));
            }
            let mut args_r: CallArgs = SmallVec::new();
            let count_r = frame.next_u8() as usize;
            for _ in 0..count_r {
                args_r.push(JitCallArg::reference(frame.next_reg() as u16));
            }
            let mut args_f: CallArgs = SmallVec::new();
            let count_f = frame.next_u8() as usize;
            for _ in 0..count_f {
                args_f.push(JitCallArg::float(frame.next_reg() as u16));
            }
            let calldescr_idx = frame.next_u16();
            let dst = frame.next_reg() as usize;
            let calldescr = call_jitcode
                .descr_at(calldescr_idx as usize)
                .and_then(crate::jitcode::RuntimeBhDescr::as_bh_descr)
                .expect("BC_RESIDUAL_CALL_IRF_F descr is not BhDescr")
                .as_calldescr();
            let trace_descr = frame
                .jitcode
                .descr_at(calldescr_idx as usize)
                .and_then(crate::jitcode::RuntimeBhDescr::as_optimizer_descr)
                .cloned()
                .unwrap_or_else(|| crate::call_descr::call_descr_from_bh(&calldescr));
            let target = frame
                .jitcode
                .exec
                .call_descr_to_call_target
                .get(&calldescr_idx)
                .copied()
                .unwrap_or_else(|| {
                    let func = frame
                        .int_regs
                        .get(funcptr_reg as usize)
                        .copied()
                        .flatten()
                        .and_then(|op| ctx.box_bits(op))
                        .unwrap_or_else(|| {
                            panic!(
                                "BC_RESIDUAL_CALL_IRF_F: funcptr slot \
                                 {funcptr_reg} is uninitialized"
                            )
                        });
                    JitCallTarget::from_fnaddr(func)
                });
            (target, args_i, args_r, args_f, calldescr, trace_descr, dst)
        };

        let (args, concrete_args, arg_types, raw_i, raw_r, raw_f) =
            self.read_canonical_call_args(ctx, &calldescr.arg_classes, &args_i, &args_r, &args_f);

        let trace_ptr = if target.trace_ptr.is_null() {
            target.concrete_ptr
        } else {
            target.trace_ptr
        };
        let concrete_ptr = if target.concrete_ptr.is_null() {
            trace_ptr
        } else {
            target.concrete_ptr
        };
        let fnaddr_word = target.fnaddr_for_symbolic_check(concrete_ptr);
        if let Some(action) = refuse_walk_local_ref_args(
            ctx,
            concrete_ptr as usize,
            &raw_i,
            &raw_r,
            &args,
            &calldescr.arg_classes,
        ) {
            return action;
        }

        let effect_descr = trace_descr.clone();
        let effectinfo = effect_descr
            .as_call_descr()
            .expect("resolved call descriptor")
            .get_extra_info();

        if effectinfo.oopspecindex == majit_ir::descr::OopSpecIndex::NotInTrace {
            // pyjitpl.py do_not_in_trace_call: route the
            // C call through `CALL_N` (void) and discard the
            // result regardless of the surface result type.  See
            // the int sibling at the corresponding NotInTrace
            // branch for the full citation.
            self.clear_exception();
            if majit_jitcode::codewriter::call::is_symbolic_fnaddr(fnaddr_word) {
                return report_symbolic_residual_call_target(
                    ctx,
                    fnaddr_word,
                    Some(&calldescr.arg_classes),
                );
            }
            if !concrete_ptr.is_null() {
                unsafe {
                    majit_backend::call_stub::bh_call_v_by_classes(
                        concrete_ptr as usize,
                        &calldescr.arg_classes,
                        Some(&raw_i),
                        Some(&raw_r),
                        Some(&raw_f),
                    );
                }
            }
            if let Some(action) =
                host_requested_walk_abort(ctx, concrete_ptr as usize, &calldescr.arg_classes)
            {
                return action;
            }
            // `pyjitpl.py do_not_in_trace_call`:
            //     if self.last_exc_value: raise SwitchToBlackhole(
            //         Counters.ABORT_ESCAPE, raising_exception=True)
            // Same stash pattern as the void OS_NOT_IN_TRACE arm
            // above.
            let exc = crate::blackhole::BH_LAST_EXC_VALUE.with(|c| c.get());
            if exc != 0 {
                return TraceAction::SwitchToBlackhole(
                    crate::pyjitpl::SwitchToBlackhole::abort_escape(),
                );
            }
            let _ = dst;
        } else {
            // `pyjitpl.py do_residual_call`'s `OS_LIBFFI_CALL` hook answers
            // `None  # cannot be handled by direct_libffi_call()` on this
            // layer: rebuilding the call out of its `CIF_DESCRIPTION`
            // needs a `MetaInterp`, which this jitcode machine does not
            // hold.  That is upstream's own fallthrough — the release-gil
            // / may-force selection below is what it falls through to.
            // The specialization lives in
            // `MetaInterp::direct_libffi_call` and in the pyre-jit-trace
            // walker's residual-call dispatchers.
            let is_release_gil = effectinfo.is_call_release_gil();
            let is_forces = effectinfo.check_forces_virtual_or_virtualizable();
            let is_loopinvariant =
                effectinfo.extraeffect == majit_ir::descr::ExtraEffect::LoopInvariant;

            // pyjitpl.py MIFrame.do_residual_call: heapcache lookup-first for
            // loop-invariant calls (see int sibling for full cite).
            if is_loopinvariant
                && let Some((cached_traced, cached_concrete_bits)) = ctx
                    .call_loopinvariant_lookup_with_effect(
                        trace_ptr,
                        &arg_types,
                        majit_ir::Type::Float,
                        effectinfo,
                    )
            {
                self.set_float_reg(ctx, dst, Some(cached_traced), Some(cached_concrete_bits));
                return TraceAction::Continue;
            }

            // pyjitpl.py MIFrame.do_residual_call MAY_FORCE_F branch parity:
            // clear_exception precedes vable_and_vrefs_before_residual_call
            // (vrefs walk + vinfo stamp; see void arm for full citation).
            // Decline a target we will not call before that stamp.
            self.clear_exception();
            if majit_jitcode::codewriter::call::is_symbolic_fnaddr(fnaddr_word) {
                return report_symbolic_residual_call_target(
                    ctx,
                    fnaddr_word,
                    Some(&calldescr.arg_classes),
                );
            }
            if concrete_ptr.is_null() {
                return refuse_null_residual_call_target(ctx, &calldescr.arg_classes);
            }
            let active_vable = if is_forces {
                ctx.vrefs_before_residual_call();
                self.prepare_standard_virtualizable_before_residual_call(ctx)
            } else {
                None
            };
            let concrete = unsafe {
                majit_backend::call_stub::bh_call_f_by_classes(
                    concrete_ptr as usize,
                    &calldescr.arg_classes,
                    Some(&raw_i),
                    Some(&raw_r),
                    Some(&raw_f),
                )
            };
            // pyjitpl.py — vrefs_after_residual_call
            // (see void arm for the explanation; gated on
            // `is_forces` because the before-hook only stamps
            // TOKEN_TRACING_RESCALL in that branch).
            if is_forces {
                ctx.vrefs_after_residual_call();
            }
            // pyjitpl.py do_residual_call plain branch —
            // see the BC_RESIDUAL_CALL_*_I sibling for the full cite.
            let plain_branch = !is_release_gil && !is_forces && !is_loopinvariant;
            let pure = plain_branch && effectinfo.check_is_elidable();
            let patch_pos = if pure {
                Some(ctx.get_trace_position())
            } else {
                None
            };
            let traced = if is_release_gil {
                ctx.call_release_gil_float_typed_with_effect(
                    trace_ptr,
                    &args,
                    &arg_types,
                    effectinfo.clone(),
                )
            } else if is_forces {
                ctx.call_may_force_float_typed_with_effect(
                    trace_ptr,
                    &args,
                    &arg_types,
                    effectinfo.clone(),
                )
            } else if is_loopinvariant {
                ctx.call_loopinvariant_float_typed_with_effect(
                    trace_ptr,
                    &args,
                    &arg_types,
                    effectinfo.clone(),
                    concrete.to_bits() as i64,
                )
            } else {
                ctx.record_call_with_descr(
                    majit_ir::OpCode::CallF,
                    trace_ptr,
                    &args,
                    trace_descr.clone(),
                )
            };
            // pyjitpl.py MIFrame.execute_varargs gate (see int sibling for full cite).
            let last_exc_value = crate::blackhole::BH_LAST_EXC_VALUE.with(|c| c.get());
            let traced = match patch_pos {
                Some(patch_pos) if last_exc_value == 0 => {
                    let func_ref = ctx.const_int(trace_ptr as usize as i64);
                    let mut call_args: CallOpRefs = SmallVec::new();
                    call_args.push(func_ref);
                    call_args.extend_from_slice(&args);
                    let concrete_values =
                        build_concrete_values(trace_ptr, &concrete_args, &arg_types);
                    ctx.record_result_of_call_pure(
                        traced,
                        &call_args,
                        &concrete_values,
                        trace_descr,
                        patch_pos,
                        majit_ir::OpCode::CallF,
                        majit_ir::Value::Float(concrete),
                    )
                }
                _ => traced,
            };
            // `pyjitpl.py execute_and_record_varargs` runs the call
            // through `executor.execute_varargs` and hands the result
            // to `history.record_nospec`, so the recorded op carries
            // the executed value on its own frontend slot -- every
            // later `getvalue()` of that box answers it.  Writing the
            // value into the destination register alone leaves
            // `concrete_of_opref` answering `None` for the box, and
            // the two readers then disagree: `_nonstandard_virtualizable`
            // asks the box, so a residual that returns the standard
            // virtualizable (the portal's `reload_top_root`) loses its
            // PTR_EQ against `virtualizable_boxes[-1]` and every later
            // vable access on that register takes the nonstandard leg.
            // The full-body walker already stamps its own residual
            // results this way (`jitcode_dispatch/residual_call.rs`).
            ctx.set_opref_concrete(traced, majit_ir::Value::Float(concrete));
            self.set_float_reg(ctx, dst, Some(traced), Some(concrete.to_bits() as i64));
            if is_forces {
                let action = self.finalize_standard_virtualizable_may_force(ctx, sym, active_vable);
                if !matches!(action, TraceAction::Continue) {
                    return action;
                }
            }
            if !(pure && traced.is_constant()) {
                match self.finish_residual_call_exception_path(ctx, sym, effectinfo) {
                    TraceAction::Continue => {}
                    action => return action,
                }
            }
        }
        TraceAction::Continue
    }

    // BC_CALL_ASSEMBLER_VOID uses `(fn_ptr_idx:u16,
    // num_args:u16, [(kind:u8, reg:u8)]...)` — the
    // assembler-token path is not in the canonical *_v family
    // and of pyre-call-family-canonical-migration.md
    // owns its migration.
    // pyjitpl.py _opimpl_recursive_call →
    // do_recursive_call → do_residual_call(assembler_call=True).
    // RPython's assembler_call path (py:2007-2083) unconditionally
    // enters the vable/guard sequence:
    //   1. clear_exception
    //   2. vable_and_vrefs_before_residual_call
    //   3. execute (CALL_MAY_FORCE_N via executor.execute_varargs)
    //   4. vrefs_after_residual_call
    //   5. record (CALL_ASSEMBLER_N via direct_assembler_call)
    //   6. vable_after_residual_call + GUARD_NOT_FORCED
    //   7. KEEPALIVE on vablebox (`pyjitpl.py`)
    //   8. handle_possible_exception (GUARD_NO_EXCEPTION / unwind)
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_call_assembler_void(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let (fn_ptr_idx, arg_regs) = {
            let frame = self.frames.current_mut();
            let fn_ptr_idx = frame.next_u16() as usize;
            let num_args = frame.next_u16() as usize;
            let mut arg_regs = Vec::with_capacity(num_args);
            for _ in 0..num_args {
                let kind = JitArgKind::decode(frame.next_u8());
                let reg = frame.next_reg();
                arg_regs.push(JitCallArg {
                    kind,
                    reg: reg as u16,
                });
            }
            (fn_ptr_idx, arg_regs)
        };
        let mut args = Vec::with_capacity(arg_regs.len());
        let mut concrete_args = Vec::with_capacity(arg_regs.len());
        let mut arg_types = Vec::with_capacity(arg_regs.len());
        let mut raw_i = Vec::new();
        let mut raw_r = Vec::new();
        let mut raw_f = Vec::new();
        let mut arg_classes = String::new();
        for arg_spec in &arg_regs {
            let (arg, concrete, arg_type) = self.read_call_arg(ctx, *arg_spec);
            args.push(arg);
            concrete_args.push(concrete);
            arg_types.push(arg_type);
            match arg_spec.kind {
                JitArgKind::Int => {
                    raw_i.push(concrete);
                    arg_classes.push('i');
                }
                JitArgKind::Ref => {
                    raw_r.push(concrete);
                    arg_classes.push('r');
                }
                JitArgKind::Float => {
                    raw_f.push(concrete);
                    arg_classes.push('f');
                }
            }
        }
        let (token_number, concrete_ptr) = self
            .frames
            .current_mut()
            .jitcode
            .call_assembler_target(fn_ptr_idx);
        // 1. `clear_exception`.
        self.clear_exception();
        // Refusals with no `do_residual_call` counterpart, before
        // `vable_and_vrefs_before_residual_call`. `aborted_tracing`
        // leaves `TOKEN_TRACING_RESCALL` in place.
        if majit_jitcode::codewriter::call::is_symbolic_fnaddr(concrete_ptr as i64) {
            return report_symbolic_residual_call_target(
                ctx,
                concrete_ptr as i64,
                Some(&arg_classes),
            );
        }
        if concrete_ptr.is_null() {
            return refuse_null_residual_call_target(ctx, &arg_classes);
        }
        if let Some(action) = refuse_walk_local_ref_args(
            ctx,
            concrete_ptr as usize,
            &raw_i,
            &raw_r,
            &args,
            &arg_classes,
        ) {
            return action;
        }
        // `direct_assembler_call` reads the token before it records. A
        // missing token aborts here, before the stamp and the concrete
        // call, so a replay of this opcode does not run the call twice.
        let Some(arc) = _runtime.jitcell_token_arc_for_number(token_number) else {
            return TraceAction::Abort;
        };
        // 2. `vable_and_vrefs_before_residual_call`: vrefs first, then
        //    the virtualizable.
        ctx.vrefs_before_residual_call();
        let active_vable = self.prepare_standard_virtualizable_before_residual_call(ctx);
        // 3. execute, tp == 'v'. The null target already returned.
        unsafe {
            majit_backend::call_stub::bh_call_v_by_classes(
                concrete_ptr as usize,
                &arg_classes,
                Some(&raw_i),
                Some(&raw_r),
                Some(&raw_f),
            );
        }
        if let Some(action) = host_requested_walk_abort(ctx, concrete_ptr as usize, &arg_classes) {
            ctx.vrefs_after_residual_call();
            let _ = Self::escaped_standard_virtualizable(ctx, active_vable);
            return action;
        }
        // 4. `vrefs_after_residual_call` before the CALL_ASSEMBLER record.
        ctx.vrefs_after_residual_call();
        // 5. record `CALL_ASSEMBLER_N`.
        ctx.call_assembler_void_arc_typed(arc, &args, &arg_types);
        // 6. vable_after_residual_call + GUARD_NOT_FORCED
        //    (pyjitpl.py)
        let vable_opref = active_vable.as_ref().map(|a| a.vable_opref);
        let action = self.finalize_standard_virtualizable_may_force(ctx, sym, active_vable);
        if !matches!(action, TraceAction::Continue) {
            return action;
        }
        // 7. `pyjitpl.py MIFrame.do_residual_call`:
        //        if vablebox is not None:
        //            self.metainterp.history.record1(rop.KEEPALIVE,
        //                                            vablebox, None)
        //    Assembler-call branch threads the active vable
        //    box through KEEPALIVE so the optimizer does not
        //    DCE the box across the asm-side call boundary.
        if let Some(vbox) = vable_opref {
            ctx.record_op(majit_ir::OpCode::Keepalive, &[vbox]);
        }
        // 8. handle_possible_exception (pyjitpl.py)
        match self.finish_call_assembler_exception_path(ctx, sym) {
            TraceAction::Continue => {}
            action => return action,
        }
        TraceAction::Continue
    }

    // ── canonical conditional_call / record_known_result ──
    // `rewrite_call(..., force_ir=True)`: lead + funcptr + I + R + d
    // (`blackhole.rs` `handler_conditional_call_*` / `handler_record_known_result_*`).
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_conditional_call_ir_v(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let (first_reg, target, args_i, args_r, calldescr, dst) = {
            let frame = self.frames.current_mut();
            let first_reg = frame.next_reg() as u16;
            let funcptr_reg = frame.next_reg() as u16;
            let count_i = frame.next_u8() as usize;
            let mut args_i = CallArgs::with_capacity(count_i);
            for _ in 0..count_i {
                args_i.push(JitCallArg::int(frame.next_reg() as u16));
            }
            let count_r = frame.next_u8() as usize;
            let mut args_r = CallArgs::with_capacity(count_r);
            for _ in 0..count_r {
                args_r.push(JitCallArg::reference(frame.next_reg() as u16));
            }
            let calldescr_idx = frame.next_u16();
            let calldescr = frame
                .jitcode
                .descr_at(calldescr_idx as usize)
                .and_then(crate::jitcode::RuntimeBhDescr::as_bh_descr)
                .expect("canonical cond/record descr is not BhDescr")
                .as_calldescr()
                .clone();
            let target = frame
                .jitcode
                .exec
                .call_descr_to_call_target
                .get(&calldescr_idx)
                .copied()
                .unwrap_or_else(|| {
                    let func = frame
                        .int_regs
                        .get(funcptr_reg as usize)
                        .copied()
                        .flatten()
                        .and_then(|op| ctx.box_bits(op))
                        .unwrap_or_else(|| {
                            panic!(
                                "canonical cond/record: funcptr slot {funcptr_reg} \
                             is uninitialized"
                            )
                        });
                    JitCallTarget::from_fnaddr(func)
                });
            let dst = if matches!(
                bytecode,
                jitcode::insns::BC_CONDITIONAL_CALL_VALUE_IR_I
                    | jitcode::insns::BC_CONDITIONAL_CALL_VALUE_IR_R
            ) {
                Some(frame.next_reg() as u16)
            } else {
                None
            };
            (first_reg, target, args_i, args_r, calldescr, dst)
        };
        let (args, concrete_args, arg_types, raw_i, raw_r, raw_f) =
            self.read_canonical_call_args(ctx, &calldescr.arg_classes, &args_i, &args_r, &[]);
        let trace_ptr = if target.trace_ptr.is_null() {
            target.concrete_ptr
        } else {
            target.trace_ptr
        };
        let concrete_ptr = if target.concrete_ptr.is_null() {
            trace_ptr
        } else {
            target.concrete_ptr
        };
        let fnaddr_word = target.fnaddr_for_symbolic_check(concrete_ptr);
        let slot = target.effect_info_slot;
        match bytecode {
            jitcode::insns::BC_CONDITIONAL_CALL_IR_V => {
                // `_opimpl_conditional_call_*`: skip only when the
                // register's own SSA box is already a Const. Inventing
                // ConstInt from this iteration's concrete value makes
                // `is_constant()` true for a live register and bakes
                // the snapshot into the trace.
                let (first_box, first_val) = self.read_int_reg(ctx, first_reg as usize);
                // `opimpl_conditional_call_ir_v`: ConstInt(0) records
                // nothing so the heapcache can keep args virtual.
                if first_box.is_constant() && first_val == 0 {
                    // skip
                } else {
                    ctx.cond_call_void_typed(first_box, trace_ptr, &args, &arg_types, slot);
                    let mut allboxes: CallOpRefs = SmallVec::new();
                    allboxes.push(first_box);
                    allboxes.push(ctx.const_int(trace_ptr as usize as i64));
                    allboxes.extend_from_slice(&args);
                    // `_record_helper_varargs` invalidates before it
                    // appends. `cond_call_void_typed` only records.
                    ctx.heapcache_invalidate_caches_varargs(
                        OpCode::CondCallN,
                        Some(&calldescr.extra_info),
                        &allboxes,
                    );
                    self.clear_exception();
                    if first_val != 0 {
                        if let Some(action) = refuse_walk_local_ref_args(
                            ctx,
                            concrete_ptr as usize,
                            &raw_i,
                            &raw_r,
                            &args,
                            &calldescr.arg_classes,
                        ) {
                            return action;
                        }
                        if majit_jitcode::codewriter::call::is_symbolic_fnaddr(fnaddr_word) {
                            return report_symbolic_residual_call_target(
                                ctx,
                                fnaddr_word,
                                Some(&calldescr.arg_classes),
                            );
                        }
                        // `do_conditional_call` → `execute_varargs` → `cpu.bh_call_v`.
                        if !concrete_ptr.is_null() {
                            unsafe {
                                majit_backend::call_stub::bh_call_v_by_classes(
                                    concrete_ptr as usize,
                                    &calldescr.arg_classes,
                                    Some(&raw_i),
                                    Some(&raw_r),
                                    Some(&raw_f),
                                );
                            }
                        }
                        if let Some(action) = host_requested_walk_abort(
                            ctx,
                            concrete_ptr as usize,
                            &calldescr.arg_classes,
                        ) {
                            return action;
                        }
                    }
                    match self.finish_residual_call_exception_path(ctx, sym, &calldescr.extra_info)
                    {
                        TraceAction::Continue => {}
                        action => return action,
                    }
                }
            }
            jitcode::insns::BC_CONDITIONAL_CALL_VALUE_IR_I => {
                let (first_box, first_val) = self.read_int_reg(ctx, first_reg as usize);
                // `_opimpl_conditional_call_value`: Const nonnull
                // returns the value box without recording.
                if first_box.is_constant() && first_val != 0 {
                    if let Some(dst) = dst {
                        self.set_int_reg(ctx, dst as usize, Some(first_box), Some(first_val));
                    }
                } else {
                    let patch_pos = ctx.get_trace_position();
                    let traced = ctx
                        .cond_call_value_int_typed(first_box, trace_ptr, &args, &arg_types, slot);
                    let mut allboxes: CallOpRefs = SmallVec::new();
                    allboxes.push(first_box);
                    allboxes.push(ctx.const_int(trace_ptr as usize as i64));
                    allboxes.extend_from_slice(&args);
                    ctx.heapcache_invalidate_caches_varargs(
                        OpCode::CondCallValueI,
                        Some(&calldescr.extra_info),
                        &allboxes,
                    );
                    self.clear_exception();
                    let concrete_result = if first_val == 0 {
                        if let Some(action) = refuse_walk_local_ref_args(
                            ctx,
                            concrete_ptr as usize,
                            &raw_i,
                            &raw_r,
                            &args,
                            &calldescr.arg_classes,
                        ) {
                            return action;
                        }
                        if majit_jitcode::codewriter::call::is_symbolic_fnaddr(fnaddr_word) {
                            return report_symbolic_residual_call_target(
                                ctx,
                                fnaddr_word,
                                Some(&calldescr.arg_classes),
                            );
                        }
                        // `do_conditional_call(is_value=True)` → `cpu.bh_call_i`.
                        let n = if concrete_ptr.is_null() {
                            0
                        } else {
                            unsafe {
                                majit_backend::call_stub::bh_call_i_by_classes(
                                    concrete_ptr as usize,
                                    &calldescr.arg_classes,
                                    Some(&raw_i),
                                    Some(&raw_r),
                                    Some(&raw_f),
                                )
                            }
                        };
                        if let Some(action) = host_requested_walk_abort(
                            ctx,
                            concrete_ptr as usize,
                            &calldescr.arg_classes,
                        ) {
                            return action;
                        }
                        n
                    } else {
                        first_val
                    };
                    // `do_conditional_call(is_value=True)` →
                    // `execute_varargs(..., pure=True)`.
                    // Skip the fold when the helper raised.
                    let last_exc = crate::blackhole::BH_LAST_EXC_VALUE.with(|c| c.get());
                    let traced = if last_exc == 0 {
                        let mut call_args: CallOpRefs = SmallVec::new();
                        call_args.push(first_box);
                        call_args.push(ctx.const_int(trace_ptr as usize as i64));
                        call_args.extend_from_slice(&args);
                        let mut concrete_values: CallValues = SmallVec::new();
                        concrete_values.push(majit_ir::Value::Int(first_val));
                        concrete_values.extend(build_concrete_values(
                            trace_ptr,
                            &concrete_args,
                            &arg_types,
                        ));
                        ctx.record_result_of_call_pure(
                            traced,
                            &call_args,
                            &concrete_values,
                            crate::call_descr::make_call_descr_with_effect(
                                &arg_types,
                                majit_ir::Type::Int,
                                calldescr.extra_info.clone(),
                            ),
                            patch_pos,
                            majit_ir::OpCode::CondCallValueI,
                            majit_ir::Value::Int(concrete_result),
                        )
                    } else {
                        traced
                    };
                    if last_exc == 0
                        && let Some(dst) = dst
                    {
                        self.set_int_reg(ctx, dst as usize, Some(traced), Some(concrete_result));
                    }
                    if !(last_exc == 0 && traced.is_constant()) {
                        match self.finish_residual_call_exception_path(
                            ctx,
                            sym,
                            &calldescr.extra_info,
                        ) {
                            TraceAction::Continue => {}
                            action => return action,
                        }
                    }
                }
            }
            jitcode::insns::BC_CONDITIONAL_CALL_VALUE_IR_R => {
                let (first_box, first_val) = self.read_ref_reg(ctx, first_reg as usize);
                if first_box.is_constant() && first_val != 0 {
                    if let Some(dst) = dst {
                        self.set_ref_reg(ctx, dst as usize, Some(first_box), Some(first_val));
                    }
                } else {
                    let patch_pos = ctx.get_trace_position();
                    let traced = ctx
                        .cond_call_value_ref_typed(first_box, trace_ptr, &args, &arg_types, slot);
                    let mut allboxes: CallOpRefs = SmallVec::new();
                    allboxes.push(first_box);
                    allboxes.push(ctx.const_int(trace_ptr as usize as i64));
                    allboxes.extend_from_slice(&args);
                    ctx.heapcache_invalidate_caches_varargs(
                        OpCode::CondCallValueR,
                        Some(&calldescr.extra_info),
                        &allboxes,
                    );
                    self.clear_exception();
                    let concrete_result = if first_val == 0 {
                        if let Some(action) = refuse_walk_local_ref_args(
                            ctx,
                            concrete_ptr as usize,
                            &raw_i,
                            &raw_r,
                            &args,
                            &calldescr.arg_classes,
                        ) {
                            return action;
                        }
                        if majit_jitcode::codewriter::call::is_symbolic_fnaddr(fnaddr_word) {
                            return report_symbolic_residual_call_target(
                                ctx,
                                fnaddr_word,
                                Some(&calldescr.arg_classes),
                            );
                        }
                        // `do_conditional_call(is_value=True)` → `cpu.bh_call_r`.
                        let p = if concrete_ptr.is_null() {
                            0
                        } else {
                            unsafe {
                                majit_backend::call_stub::bh_call_i_by_classes(
                                    concrete_ptr as usize,
                                    &calldescr.arg_classes,
                                    Some(&raw_i),
                                    Some(&raw_r),
                                    Some(&raw_f),
                                )
                            }
                        };
                        if let Some(action) = host_requested_walk_abort(
                            ctx,
                            concrete_ptr as usize,
                            &calldescr.arg_classes,
                        ) {
                            return action;
                        }
                        p
                    } else {
                        first_val
                    };
                    let last_exc = crate::blackhole::BH_LAST_EXC_VALUE.with(|c| c.get());
                    let traced = if last_exc == 0 {
                        let mut call_args: CallOpRefs = SmallVec::new();
                        call_args.push(first_box);
                        call_args.push(ctx.const_int(trace_ptr as usize as i64));
                        call_args.extend_from_slice(&args);
                        let mut concrete_values: CallValues = SmallVec::new();
                        concrete_values
                            .push(majit_ir::Value::Ref(majit_ir::GcRef(first_val as usize)));
                        concrete_values.extend(build_concrete_values(
                            trace_ptr,
                            &concrete_args,
                            &arg_types,
                        ));
                        ctx.record_result_of_call_pure(
                            traced,
                            &call_args,
                            &concrete_values,
                            crate::call_descr::make_call_descr_with_effect(
                                &arg_types,
                                majit_ir::Type::Ref,
                                calldescr.extra_info.clone(),
                            ),
                            patch_pos,
                            majit_ir::OpCode::CondCallValueR,
                            majit_ir::Value::Ref(majit_ir::GcRef(concrete_result as usize)),
                        )
                    } else {
                        traced
                    };
                    if last_exc == 0
                        && let Some(dst) = dst
                    {
                        self.set_ref_reg(ctx, dst as usize, Some(traced), Some(concrete_result));
                    }
                    if !(last_exc == 0 && traced.is_constant()) {
                        match self.finish_residual_call_exception_path(
                            ctx,
                            sym,
                            &calldescr.extra_info,
                        ) {
                            TraceAction::Continue => {}
                            action => return action,
                        }
                    }
                }
            }
            jitcode::insns::BC_RECORD_KNOWN_RESULT_I_IR_V => {
                let (first_box, _) = self.read_int_reg(ctx, first_reg as usize);
                ctx.profiler()
                    .count_ops(OpCode::RecordKnownResult, crate::counters::RECORDED_OPS);
                ctx.record_known_result_typed(
                    first_box,
                    trace_ptr,
                    &args,
                    &arg_types,
                    majit_ir::Type::Int,
                    calldescr.extra_info.clone(),
                );
                let mut allboxes: CallOpRefs = SmallVec::new();
                allboxes.push(first_box);
                allboxes.push(ctx.const_int(trace_ptr as usize as i64));
                allboxes.extend_from_slice(&args);
                ctx.heapcache_invalidate_caches_varargs(
                    OpCode::RecordKnownResult,
                    Some(&calldescr.extra_info),
                    &allboxes,
                );
            }
            jitcode::insns::BC_RECORD_KNOWN_RESULT_R_IR_V => {
                let (first_box, _) = self.read_ref_reg(ctx, first_reg as usize);
                ctx.profiler()
                    .count_ops(OpCode::RecordKnownResult, crate::counters::RECORDED_OPS);
                ctx.record_known_result_typed(
                    first_box,
                    trace_ptr,
                    &args,
                    &arg_types,
                    majit_ir::Type::Ref,
                    calldescr.extra_info.clone(),
                );
                let mut allboxes: CallOpRefs = SmallVec::new();
                allboxes.push(first_box);
                allboxes.push(ctx.const_int(trace_ptr as usize as i64));
                allboxes.extend_from_slice(&args);
                ctx.heapcache_invalidate_caches_varargs(
                    OpCode::RecordKnownResult,
                    Some(&calldescr.extra_info),
                    &allboxes,
                );
            }
            _ => unreachable!(),
        }
        TraceAction::Continue
    }

    // ── helper-side ext payload (no remaining emit; decode kept) ──
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_cond_call_void(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let (first_reg, fn_ptr_idx, arg_regs, dst) = {
            let frame = self.frames.current_mut();
            let first_reg = frame.next_reg() as u16;
            let fn_ptr_idx = frame.next_u16() as usize;
            let num_args = frame.next_u8() as usize;
            let mut arg_regs = Vec::with_capacity(num_args);
            for _ in 0..num_args {
                let kind = JitArgKind::decode(frame.next_u8());
                let reg = frame.next_reg();
                arg_regs.push(JitCallArg {
                    kind,
                    reg: reg as u16,
                });
            }
            let dst = if matches!(
                bytecode,
                jitcode::insns::BC_COND_CALL_VALUE_INT | jitcode::insns::BC_COND_CALL_VALUE_REF
            ) {
                Some(frame.next_reg() as u16)
            } else {
                None
            };
            (first_reg, fn_ptr_idx, arg_regs, dst)
        };
        let mut args = Vec::with_capacity(arg_regs.len());
        let mut concrete_args = Vec::with_capacity(arg_regs.len());
        let mut arg_types = Vec::with_capacity(arg_regs.len());
        for arg_spec in &arg_regs {
            let (arg, concrete, arg_type) = self.read_call_arg(ctx, *arg_spec);
            args.push(arg);
            concrete_args.push(concrete);
            arg_types.push(arg_type);
        }
        let target = *self.frames.current_mut().jitcode.call_target(fn_ptr_idx);
        let trace_ptr = if target.trace_ptr.is_null() {
            target.concrete_ptr
        } else {
            target.trace_ptr
        };
        let concrete_ptr = if target.concrete_ptr.is_null() {
            trace_ptr
        } else {
            target.concrete_ptr
        };
        let fnaddr_word = target.fnaddr_for_symbolic_check(concrete_ptr);
        let slot = target.effect_info_slot;
        let extra_info = crate::call_descr::effect_info_for_slot(slot);
        let mut raw_i = Vec::new();
        let mut raw_r = Vec::new();
        let mut raw_f = Vec::new();
        let mut arg_classes = String::new();
        for (spec, &concrete) in arg_regs.iter().zip(concrete_args.iter()) {
            match spec.kind {
                JitArgKind::Int => {
                    raw_i.push(concrete);
                    arg_classes.push('i');
                }
                JitArgKind::Ref => {
                    raw_r.push(concrete);
                    arg_classes.push('r');
                }
                JitArgKind::Float => {
                    raw_f.push(concrete);
                    arg_classes.push('f');
                }
            }
        }
        match bytecode {
            jitcode::insns::BC_COND_CALL_VOID => {
                // RPython pyjitpl.py opimpl_conditional_call_ir_v:
                //   if condition != 0: call func(args)
                let (first_box, first_val) = self.read_int_reg(ctx, first_reg as usize);
                if first_box.is_constant() && first_val == 0 {
                    // skip
                } else {
                    ctx.cond_call_void_typed(first_box, trace_ptr, &args, &arg_types, slot);
                    let mut allboxes: CallOpRefs = SmallVec::new();
                    allboxes.push(first_box);
                    allboxes.push(ctx.const_int(trace_ptr as usize as i64));
                    allboxes.extend_from_slice(&args);
                    ctx.heapcache_invalidate_caches_varargs(
                        OpCode::CondCallN,
                        Some(&extra_info),
                        &allboxes,
                    );
                    self.clear_exception();
                    if first_val != 0 {
                        if let Some(action) = refuse_walk_local_ref_args(
                            ctx,
                            concrete_ptr as usize,
                            &raw_i,
                            &raw_r,
                            &args,
                            &arg_classes,
                        ) {
                            return action;
                        }
                        if majit_jitcode::codewriter::call::is_symbolic_fnaddr(fnaddr_word) {
                            return report_symbolic_residual_call_target(
                                ctx,
                                fnaddr_word,
                                Some(&arg_classes),
                            );
                        }
                        // leftover cond_call_void_ext → `cpu.bh_call_v`.
                        if !concrete_ptr.is_null() {
                            unsafe {
                                majit_backend::call_stub::bh_call_v_by_classes(
                                    concrete_ptr as usize,
                                    &arg_classes,
                                    Some(&raw_i),
                                    Some(&raw_r),
                                    Some(&raw_f),
                                );
                            }
                        }
                        if let Some(action) =
                            host_requested_walk_abort(ctx, concrete_ptr as usize, &arg_classes)
                        {
                            return action;
                        }
                    }
                    match self.finish_residual_call_exception_path(ctx, sym, &extra_info) {
                        TraceAction::Continue => {}
                        action => return action,
                    }
                }
            }
            jitcode::insns::BC_COND_CALL_VALUE_INT => {
                // RPython pyjitpl.py opimpl_conditional_call_value_ir_i
                let (first_box, first_val) = self.read_int_reg(ctx, first_reg as usize);
                if first_box.is_constant() && first_val != 0 {
                    if let Some(dst) = dst {
                        self.set_int_reg(ctx, dst as usize, Some(first_box), Some(first_val));
                    }
                } else {
                    let patch_pos = ctx.get_trace_position();
                    let traced = ctx
                        .cond_call_value_int_typed(first_box, trace_ptr, &args, &arg_types, slot);
                    let mut allboxes: CallOpRefs = SmallVec::new();
                    allboxes.push(first_box);
                    allboxes.push(ctx.const_int(trace_ptr as usize as i64));
                    allboxes.extend_from_slice(&args);
                    ctx.heapcache_invalidate_caches_varargs(
                        OpCode::CondCallValueI,
                        Some(&extra_info),
                        &allboxes,
                    );
                    self.clear_exception();
                    let concrete_result = if first_val == 0 {
                        if let Some(action) = refuse_walk_local_ref_args(
                            ctx,
                            concrete_ptr as usize,
                            &raw_i,
                            &raw_r,
                            &args,
                            &arg_classes,
                        ) {
                            return action;
                        }
                        if majit_jitcode::codewriter::call::is_symbolic_fnaddr(fnaddr_word) {
                            return report_symbolic_residual_call_target(
                                ctx,
                                fnaddr_word,
                                Some(&arg_classes),
                            );
                        }
                        // leftover cond_call_value_int_ext → `cpu.bh_call_i`.
                        let n = if concrete_ptr.is_null() {
                            0
                        } else {
                            unsafe {
                                majit_backend::call_stub::bh_call_i_by_classes(
                                    concrete_ptr as usize,
                                    &arg_classes,
                                    Some(&raw_i),
                                    Some(&raw_r),
                                    Some(&raw_f),
                                )
                            }
                        };
                        if let Some(action) =
                            host_requested_walk_abort(ctx, concrete_ptr as usize, &arg_classes)
                        {
                            return action;
                        }
                        n
                    } else {
                        first_val
                    };
                    let last_exc = crate::blackhole::BH_LAST_EXC_VALUE.with(|c| c.get());
                    let traced = if last_exc == 0 {
                        let mut call_args: CallOpRefs = SmallVec::new();
                        call_args.push(first_box);
                        call_args.push(ctx.const_int(trace_ptr as usize as i64));
                        call_args.extend_from_slice(&args);
                        let mut concrete_values: CallValues = SmallVec::new();
                        concrete_values.push(majit_ir::Value::Int(first_val));
                        concrete_values.extend(build_concrete_values(
                            trace_ptr,
                            &concrete_args,
                            &arg_types,
                        ));
                        ctx.record_result_of_call_pure(
                            traced,
                            &call_args,
                            &concrete_values,
                            crate::call_descr::make_call_descr_with_effect(
                                &arg_types,
                                majit_ir::Type::Int,
                                extra_info.clone(),
                            ),
                            patch_pos,
                            majit_ir::OpCode::CondCallValueI,
                            majit_ir::Value::Int(concrete_result),
                        )
                    } else {
                        traced
                    };
                    if last_exc == 0
                        && let Some(dst) = dst
                    {
                        self.set_int_reg(ctx, dst as usize, Some(traced), Some(concrete_result));
                    }
                    if !(last_exc == 0 && traced.is_constant()) {
                        match self.finish_residual_call_exception_path(ctx, sym, &extra_info) {
                            TraceAction::Continue => {}
                            action => return action,
                        }
                    }
                }
            }
            jitcode::insns::BC_COND_CALL_VALUE_REF => {
                // RPython pyjitpl.py opimpl_conditional_call_value_ir_r:
                // value is a ref — read from ref register bank.
                let (first_box, first_val) = self.read_ref_reg(ctx, first_reg as usize);
                if first_box.is_constant() && first_val != 0 {
                    if let Some(dst) = dst {
                        self.set_ref_reg(ctx, dst as usize, Some(first_box), Some(first_val));
                    }
                } else {
                    let patch_pos = ctx.get_trace_position();
                    let traced = ctx
                        .cond_call_value_ref_typed(first_box, trace_ptr, &args, &arg_types, slot);
                    let mut allboxes: CallOpRefs = SmallVec::new();
                    allboxes.push(first_box);
                    allboxes.push(ctx.const_int(trace_ptr as usize as i64));
                    allboxes.extend_from_slice(&args);
                    ctx.heapcache_invalidate_caches_varargs(
                        OpCode::CondCallValueR,
                        Some(&extra_info),
                        &allboxes,
                    );
                    self.clear_exception();
                    let concrete_result = if first_val == 0 {
                        if let Some(action) = refuse_walk_local_ref_args(
                            ctx,
                            concrete_ptr as usize,
                            &raw_i,
                            &raw_r,
                            &args,
                            &arg_classes,
                        ) {
                            return action;
                        }
                        if majit_jitcode::codewriter::call::is_symbolic_fnaddr(fnaddr_word) {
                            return report_symbolic_residual_call_target(
                                ctx,
                                fnaddr_word,
                                Some(&arg_classes),
                            );
                        }
                        // leftover cond_call_value_ref_ext → `cpu.bh_call_r`.
                        let p = if concrete_ptr.is_null() {
                            0
                        } else {
                            unsafe {
                                majit_backend::call_stub::bh_call_i_by_classes(
                                    concrete_ptr as usize,
                                    &arg_classes,
                                    Some(&raw_i),
                                    Some(&raw_r),
                                    Some(&raw_f),
                                )
                            }
                        };
                        if let Some(action) =
                            host_requested_walk_abort(ctx, concrete_ptr as usize, &arg_classes)
                        {
                            return action;
                        }
                        p
                    } else {
                        first_val
                    };
                    let last_exc = crate::blackhole::BH_LAST_EXC_VALUE.with(|c| c.get());
                    let traced = if last_exc == 0 {
                        let mut call_args: CallOpRefs = SmallVec::new();
                        call_args.push(first_box);
                        call_args.push(ctx.const_int(trace_ptr as usize as i64));
                        call_args.extend_from_slice(&args);
                        let mut concrete_values: CallValues = SmallVec::new();
                        concrete_values
                            .push(majit_ir::Value::Ref(majit_ir::GcRef(first_val as usize)));
                        concrete_values.extend(build_concrete_values(
                            trace_ptr,
                            &concrete_args,
                            &arg_types,
                        ));
                        ctx.record_result_of_call_pure(
                            traced,
                            &call_args,
                            &concrete_values,
                            crate::call_descr::make_call_descr_with_effect(
                                &arg_types,
                                majit_ir::Type::Ref,
                                extra_info.clone(),
                            ),
                            patch_pos,
                            majit_ir::OpCode::CondCallValueR,
                            majit_ir::Value::Ref(majit_ir::GcRef(concrete_result as usize)),
                        )
                    } else {
                        traced
                    };
                    if last_exc == 0
                        && let Some(dst) = dst
                    {
                        self.set_ref_reg(ctx, dst as usize, Some(traced), Some(concrete_result));
                    }
                    if !(last_exc == 0 && traced.is_constant()) {
                        match self.finish_residual_call_exception_path(ctx, sym, &extra_info) {
                            TraceAction::Continue => {}
                            action => return action,
                        }
                    }
                }
            }
            jitcode::insns::BC_RECORD_KNOWN_RESULT_INT => {
                // RPython pyjitpl.py opimpl_record_known_result_i.
                // `jtransform.py Transformer.rewrite_op_jit_record_known_result` uses op.args[0] (the
                // known-result var) as the fake result var for
                // `getcalldescr`; here that maps to `Type::Int`
                // because the bytecode is `_i_ir_v`.
                let (first_box, _) = self.read_int_reg(ctx, first_reg as usize);
                // `opimpl_record_known_result_i_ir_v` records without executing.
                ctx.profiler()
                    .count_ops(OpCode::RecordKnownResult, crate::counters::RECORDED_OPS);
                ctx.record_known_result_typed(
                    first_box,
                    trace_ptr,
                    &args,
                    &arg_types,
                    majit_ir::Type::Int,
                    extra_info.clone(),
                );
                let mut allboxes: CallOpRefs = SmallVec::new();
                allboxes.push(first_box);
                allboxes.push(ctx.const_int(trace_ptr as usize as i64));
                allboxes.extend_from_slice(&args);
                ctx.heapcache_invalidate_caches_varargs(
                    OpCode::RecordKnownResult,
                    Some(&extra_info),
                    &allboxes,
                );
            }
            jitcode::insns::BC_RECORD_KNOWN_RESULT_REF => {
                // RPython pyjitpl.py opimpl_record_known_result_r —
                // `_r_ir_v` opname, calldescr result type is
                // `Type::Ref`.
                let (first_box, _) = self.read_ref_reg(ctx, first_reg as usize);
                // `opimpl_record_known_result_r_ir_v` records without executing.
                ctx.profiler()
                    .count_ops(OpCode::RecordKnownResult, crate::counters::RECORDED_OPS);
                ctx.record_known_result_typed(
                    first_box,
                    trace_ptr,
                    &args,
                    &arg_types,
                    majit_ir::Type::Ref,
                    extra_info.clone(),
                );
                let mut allboxes: CallOpRefs = SmallVec::new();
                allboxes.push(first_box);
                allboxes.push(ctx.const_int(trace_ptr as usize as i64));
                allboxes.extend_from_slice(&args);
                ctx.heapcache_invalidate_caches_varargs(
                    OpCode::RecordKnownResult,
                    Some(&extra_info),
                    &allboxes,
                );
            }
            _ => unreachable!(),
        }
        TraceAction::Continue
    }

    // RPython `blackhole.py` `bhimpl_int_copy`. Operand
    // order is `[src][dst]` per argcode `i>i`
    // (`assembler.py write_insn`).
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_move_i(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let (src, dst) = {
            let frame = self.frames.current_mut();
            (frame.next_reg() as usize, frame.next_reg() as usize)
        };
        let (value, concrete) = self.read_int_reg(ctx, src);
        self.set_int_reg(ctx, dst, Some(value), Some(concrete));
        TraceAction::Continue
    }

    // `int_copy/c>i` — USE_C_FORM short source (`assembler.py`):
    // the small ConstInt is one inline signed byte (`signedord`,
    // `blackhole.py`), not a `registers_i` slot. Operand order
    // `[const][dst]` per argcode `c>i`.
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_move_i_c(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let (value, dst) = {
            let frame = self.frames.current_mut();
            (frame.next_u8() as i8 as i64, frame.next_reg() as usize)
        };
        self.set_int_reg(ctx, dst, Some(OpRef::ConstInt(value)), Some(value));
        TraceAction::Continue
    }

    // The Pure half of this arm is retired — every Pure call site
    // emits canonical BC_RESIDUAL_CALL_*_I; the canonical walker reads the
    // calldescr's `check_is_elidable()` and routes through
    // `record_result_of_call_pure`.  Only BC_CALL_ASSEMBLER_INT
    // survives here.
    // pyjitpl.py do_residual_call(assembler_call=True)
    // with tp == 'i'. See BC_CALL_ASSEMBLER_VOID for the full
    // RPython sequence citation.
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_call_assembler_int(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let (fn_ptr_idx, dst, arg_regs) = {
            let frame = self.frames.current_mut();
            let fn_ptr_idx = frame.next_u16() as usize;
            let dst = frame.next_reg() as usize;
            let num_args = frame.next_u16() as usize;
            let mut arg_regs = Vec::with_capacity(num_args);
            for _ in 0..num_args {
                let kind = JitArgKind::decode(frame.next_u8());
                let reg = frame.next_reg();
                arg_regs.push(JitCallArg {
                    kind,
                    reg: reg as u16,
                });
            }
            (fn_ptr_idx, dst, arg_regs)
        };
        let mut args = Vec::with_capacity(arg_regs.len());
        let mut concrete_args = Vec::with_capacity(arg_regs.len());
        let mut arg_types = Vec::with_capacity(arg_regs.len());
        let mut raw_i = Vec::new();
        let mut raw_r = Vec::new();
        let mut raw_f = Vec::new();
        let mut arg_classes = String::new();
        for arg_spec in &arg_regs {
            let (arg, concrete, arg_type) = self.read_call_arg(ctx, *arg_spec);
            args.push(arg);
            concrete_args.push(concrete);
            arg_types.push(arg_type);
            match arg_spec.kind {
                JitArgKind::Int => {
                    raw_i.push(concrete);
                    arg_classes.push('i');
                }
                JitArgKind::Ref => {
                    raw_r.push(concrete);
                    arg_classes.push('r');
                }
                JitArgKind::Float => {
                    raw_f.push(concrete);
                    arg_classes.push('f');
                }
            }
        }
        let (token_number, concrete_ptr) = self
            .frames
            .current_mut()
            .jitcode
            .call_assembler_target(fn_ptr_idx);
        self.clear_exception();
        if majit_jitcode::codewriter::call::is_symbolic_fnaddr(concrete_ptr as i64) {
            return report_symbolic_residual_call_target(
                ctx,
                concrete_ptr as i64,
                Some(&arg_classes),
            );
        }
        if concrete_ptr.is_null() {
            return refuse_null_residual_call_target(ctx, &arg_classes);
        }
        if let Some(action) = refuse_walk_local_ref_args(
            ctx,
            concrete_ptr as usize,
            &raw_i,
            &raw_r,
            &args,
            &arg_classes,
        ) {
            return action;
        }
        let Some(arc) = _runtime.jitcell_token_arc_for_number(token_number) else {
            return TraceAction::Abort;
        };
        ctx.vrefs_before_residual_call();
        let active_vable = self.prepare_standard_virtualizable_before_residual_call(ctx);
        let concrete = unsafe {
            majit_backend::call_stub::bh_call_i_by_classes(
                concrete_ptr as usize,
                &arg_classes,
                Some(&raw_i),
                Some(&raw_r),
                Some(&raw_f),
            )
        };
        if let Some(action) = host_requested_walk_abort(ctx, concrete_ptr as usize, &arg_classes) {
            ctx.vrefs_after_residual_call();
            let _ = Self::escaped_standard_virtualizable(ctx, active_vable);
            return action;
        }
        ctx.vrefs_after_residual_call();
        let traced = ctx.call_assembler_int_arc_typed(arc, &args, &arg_types);
        self.set_int_reg(ctx, dst, Some(traced), Some(concrete));
        let vable_opref = active_vable.as_ref().map(|a| a.vable_opref);
        let action = self.finalize_standard_virtualizable_may_force(ctx, sym, active_vable);
        if !matches!(action, TraceAction::Continue) {
            return action;
        }
        // `pyjitpl.py MIFrame.do_residual_call` KEEPALIVE on the vable box.
        if let Some(vbox) = vable_opref {
            ctx.record_op(majit_ir::OpCode::Keepalive, &[vbox]);
        }
        match self.finish_call_assembler_exception_path(ctx, sym) {
            TraceAction::Continue => {}
            action => return action,
        }
        TraceAction::Continue
    }

    // -- Ref-typed bytecodes
    // RPython `blackhole.py` `bhimpl_ref_copy`. `[src][dst]` per `r>r`.
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_move_r(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let (src, dst) = {
            let frame = self.frames.current_mut();
            (frame.next_reg() as usize, frame.next_reg() as usize)
        };
        let (value, concrete) = self.read_ref_reg(ctx, src);
        self.set_ref_reg(ctx, dst, Some(value), Some(concrete));
        TraceAction::Continue
    }

    // The Pure half is retired — every Pure call site emits canonical
    // `BC_RESIDUAL_CALL_*`, and the canonical walker reads the
    // calldescr's `check_is_elidable()` and routes through
    // `record_result_of_call_pure`.
    // pyjitpl.py do_residual_call(assembler_call=True)
    // with tp == 'r'. See BC_CALL_ASSEMBLER_VOID for citation.
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_call_assembler_ref(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let (fn_ptr_idx, dst, arg_regs) = {
            let frame = self.frames.current_mut();
            let fn_ptr_idx = frame.next_u16() as usize;
            let dst = frame.next_reg() as usize;
            let num_args = frame.next_u16() as usize;
            let mut arg_regs = Vec::with_capacity(num_args);
            for _ in 0..num_args {
                let kind = JitArgKind::decode(frame.next_u8());
                let reg = frame.next_reg();
                arg_regs.push(JitCallArg {
                    kind,
                    reg: reg as u16,
                });
            }
            (fn_ptr_idx, dst, arg_regs)
        };
        let mut args = Vec::with_capacity(arg_regs.len());
        let mut concrete_args = Vec::with_capacity(arg_regs.len());
        let mut arg_types = Vec::with_capacity(arg_regs.len());
        let mut raw_i = Vec::new();
        let mut raw_r = Vec::new();
        let mut raw_f = Vec::new();
        let mut arg_classes = String::new();
        for arg_spec in &arg_regs {
            let (arg, concrete, arg_type) = self.read_call_arg(ctx, *arg_spec);
            args.push(arg);
            concrete_args.push(concrete);
            arg_types.push(arg_type);
            match arg_spec.kind {
                JitArgKind::Int => {
                    raw_i.push(concrete);
                    arg_classes.push('i');
                }
                JitArgKind::Ref => {
                    raw_r.push(concrete);
                    arg_classes.push('r');
                }
                JitArgKind::Float => {
                    raw_f.push(concrete);
                    arg_classes.push('f');
                }
            }
        }
        let (token_number, concrete_ptr) = self
            .frames
            .current_mut()
            .jitcode
            .call_assembler_target(fn_ptr_idx);
        self.clear_exception();
        if majit_jitcode::codewriter::call::is_symbolic_fnaddr(concrete_ptr as i64) {
            return report_symbolic_residual_call_target(
                ctx,
                concrete_ptr as i64,
                Some(&arg_classes),
            );
        }
        if concrete_ptr.is_null() {
            return refuse_null_residual_call_target(ctx, &arg_classes);
        }
        if let Some(action) = refuse_walk_local_ref_args(
            ctx,
            concrete_ptr as usize,
            &raw_i,
            &raw_r,
            &args,
            &arg_classes,
        ) {
            return action;
        }
        let Some(arc) = _runtime.jitcell_token_arc_for_number(token_number) else {
            return TraceAction::Abort;
        };
        ctx.vrefs_before_residual_call();
        let active_vable = self.prepare_standard_virtualizable_before_residual_call(ctx);
        // tp == 'r' — leftover wrappers use `bh_call_i_by_classes`.
        let concrete = unsafe {
            majit_backend::call_stub::bh_call_i_by_classes(
                concrete_ptr as usize,
                &arg_classes,
                Some(&raw_i),
                Some(&raw_r),
                Some(&raw_f),
            )
        };
        if let Some(action) = host_requested_walk_abort(ctx, concrete_ptr as usize, &arg_classes) {
            ctx.vrefs_after_residual_call();
            let _ = Self::escaped_standard_virtualizable(ctx, active_vable);
            return action;
        }
        ctx.vrefs_after_residual_call();
        let traced = ctx.call_assembler_ref_arc_typed(arc, &args, &arg_types);
        self.set_ref_reg(ctx, dst, Some(traced), Some(concrete));
        let vable_opref = active_vable.as_ref().map(|a| a.vable_opref);
        let action = self.finalize_standard_virtualizable_may_force(ctx, sym, active_vable);
        if !matches!(action, TraceAction::Continue) {
            return action;
        }
        // `pyjitpl.py MIFrame.do_residual_call` KEEPALIVE on the vable box.
        if let Some(vbox) = vable_opref {
            ctx.record_op(majit_ir::OpCode::Keepalive, &[vbox]);
        }
        match self.finish_call_assembler_exception_path(ctx, sym) {
            TraceAction::Continue => {}
            action => return action,
        }
        TraceAction::Continue
    }

    // -- Float-typed bytecodes
    // RPython `blackhole.py` `bhimpl_float_copy`. `[src][dst]` per `f>f`.
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_move_f(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let (src, dst) = {
            let frame = self.frames.current_mut();
            (frame.next_reg() as usize, frame.next_reg() as usize)
        };
        let (value, concrete) = self.read_float_reg(ctx, src);
        self.set_float_reg(ctx, dst, Some(value), Some(concrete));
        TraceAction::Continue
    }

    // The Pure half is retired — every Pure call site emits canonical
    // `BC_RESIDUAL_CALL_*`, and the canonical walker reads the
    // calldescr's `check_is_elidable()` and routes through
    // `record_result_of_call_pure`.
    // pyjitpl.py do_residual_call(assembler_call=True)
    // with tp == 'f'. See BC_CALL_ASSEMBLER_VOID for citation.
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_call_assembler_float(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let (fn_ptr_idx, dst, arg_regs) = {
            let frame = self.frames.current_mut();
            let fn_ptr_idx = frame.next_u16() as usize;
            let dst = frame.next_reg() as usize;
            let num_args = frame.next_u16() as usize;
            let mut arg_regs = Vec::with_capacity(num_args);
            for _ in 0..num_args {
                let kind = JitArgKind::decode(frame.next_u8());
                let reg = frame.next_reg();
                arg_regs.push(JitCallArg {
                    kind,
                    reg: reg as u16,
                });
            }
            (fn_ptr_idx, dst, arg_regs)
        };
        let mut args = Vec::with_capacity(arg_regs.len());
        let mut concrete_args = Vec::with_capacity(arg_regs.len());
        let mut arg_types = Vec::with_capacity(arg_regs.len());
        let mut raw_i = Vec::new();
        let mut raw_r = Vec::new();
        let mut raw_f = Vec::new();
        let mut arg_classes = String::new();
        for arg_spec in &arg_regs {
            let (arg, concrete, arg_type) = self.read_call_arg(ctx, *arg_spec);
            args.push(arg);
            concrete_args.push(concrete);
            arg_types.push(arg_type);
            match arg_spec.kind {
                JitArgKind::Int => {
                    raw_i.push(concrete);
                    arg_classes.push('i');
                }
                JitArgKind::Ref => {
                    raw_r.push(concrete);
                    arg_classes.push('r');
                }
                JitArgKind::Float => {
                    raw_f.push(concrete);
                    arg_classes.push('f');
                }
            }
        }
        let (token_number, concrete_ptr) = self
            .frames
            .current_mut()
            .jitcode
            .call_assembler_target(fn_ptr_idx);
        self.clear_exception();
        if majit_jitcode::codewriter::call::is_symbolic_fnaddr(concrete_ptr as i64) {
            return report_symbolic_residual_call_target(
                ctx,
                concrete_ptr as i64,
                Some(&arg_classes),
            );
        }
        if concrete_ptr.is_null() {
            return refuse_null_residual_call_target(ctx, &arg_classes);
        }
        if let Some(action) = refuse_walk_local_ref_args(
            ctx,
            concrete_ptr as usize,
            &raw_i,
            &raw_r,
            &args,
            &arg_classes,
        ) {
            return action;
        }
        let Some(arc) = _runtime.jitcell_token_arc_for_number(token_number) else {
            return TraceAction::Abort;
        };
        ctx.vrefs_before_residual_call();
        let active_vable = self.prepare_standard_virtualizable_before_residual_call(ctx);
        // tp == 'f' — leftover wrappers return packed bits via `bh_call_i`.
        let concrete = unsafe {
            majit_backend::call_stub::bh_call_i_by_classes(
                concrete_ptr as usize,
                &arg_classes,
                Some(&raw_i),
                Some(&raw_r),
                Some(&raw_f),
            )
        };
        if let Some(action) = host_requested_walk_abort(ctx, concrete_ptr as usize, &arg_classes) {
            ctx.vrefs_after_residual_call();
            let _ = Self::escaped_standard_virtualizable(ctx, active_vable);
            return action;
        }
        ctx.vrefs_after_residual_call();
        let traced = ctx.call_assembler_float_arc_typed(arc, &args, &arg_types);
        self.set_float_reg(ctx, dst, Some(traced), Some(concrete));
        let vable_opref = active_vable.as_ref().map(|a| a.vable_opref);
        let action = self.finalize_standard_virtualizable_may_force(ctx, sym, active_vable);
        if !matches!(action, TraceAction::Continue) {
            return action;
        }
        // `pyjitpl.py MIFrame.do_residual_call` KEEPALIVE on the vable box.
        if let Some(vbox) = vable_opref {
            ctx.record_op(majit_ir::OpCode::Keepalive, &[vbox]);
        }
        match self.finish_call_assembler_exception_path(ctx, sym) {
            TraceAction::Continue => {}
            action => return action,
        }
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_float_add(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_binop_f(ctx, OpCode::FloatAdd);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_float_sub(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_binop_f(ctx, OpCode::FloatSub);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_float_mul(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_binop_f(ctx, OpCode::FloatMul);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_float_truediv(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_binop_f(ctx, OpCode::FloatTrueDiv);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_float_neg(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_unary_f(ctx, OpCode::FloatNeg);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_float_abs(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_unary_f(ctx, OpCode::FloatAbs);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_float_lt(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_compare_f(ctx, OpCode::FloatLt);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_float_le(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_compare_f(ctx, OpCode::FloatLe);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_float_eq(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_compare_f(ctx, OpCode::FloatEq);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_float_ne(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_compare_f(ctx, OpCode::FloatNe);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_float_gt(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_compare_f(ctx, OpCode::FloatGt);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_float_ge(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_compare_f(ctx, OpCode::FloatGe);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_cast_int_to_float(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_cast_int_to_float(ctx);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_cast_float_to_int(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_cast_float_to_int(ctx);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_cast_ptr_to_int(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_cast_ptr_to_int(ctx);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_cast_int_to_ptr(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_cast_int_to_ptr(ctx);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_convert_float_bytes_to_longlong(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_convert_float_bytes_to_longlong(ctx);
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_convert_longlong_bytes_to_float(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.trace_convert_longlong_bytes_to_float(ctx);
        TraceAction::Continue
    }

    // pyjitpl.py opimpl_int_guard_value → implement_guard_value
    // Blackhole: no-op.  Tracing: emit GUARD_VALUE to promote.
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_int_guard_value(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let (src, opcode_pc) = {
            let frame = self.frames.current_mut();
            let opcode_pc = frame.code_cursor - 1;
            (frame.next_reg() as usize, opcode_pc)
        };
        let (opref, concrete) = self.read_int_reg(ctx, src);
        let const_ref = ctx.const_int(concrete);
        self.record_state_guard(
            ctx,
            sym,
            OpCode::GuardValue,
            &[opref, const_ref],
            opcode_pc,
            false,
        );
        // `implement_guard_value`'s `if isinstance(box, Const): return box`.
        if !opref.is_constant() {
            self.replace_box(ctx, opref, const_ref, Type::Int);
        }
        TraceAction::Continue
    }

    // pyjitpl.py opimpl_assert_not_none.  Blackhole:
    // asserts the concrete ref is non-null and advances past
    // the 1-byte ref operand.  Tracing: route through
    // `TraceCtx::trace_assert_not_none` which gates on
    // `heap_cache.is_nullity_known` + bumps `HEAPCACHED_OPS`
    // on cache hit per pyjitpl.py.
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_assert_not_none(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let src = self.frames.current_mut().next_reg() as usize;
        let (opref, concrete) = self.read_ref_reg(ctx, src);
        ctx.trace_assert_not_none(opref, concrete);
        TraceAction::Continue
    }

    // pyjitpl.py opimpl_record_exact_class.  Blackhole:
    // no-op (handler_record_exact_class advances past the 2-byte
    // (ref, int) operand).  Tracing: route through
    // `TraceCtx::trace_record_exact_class` which gates on
    // `heap_cache.is_class_known` + bumps `HEAPCACHED_OPS` on
    // cache hit per pyjitpl.py.  The class operand follows
    // blackhole.py `@arguments("r", "i")` and remains the
    // ConstInt vtable address that RPython passes as `clsbox`.
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_record_exact_class(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let src = self.frames.current_mut().next_reg() as usize;
        let cls = self.frames.current_mut().next_reg() as usize;
        let (box_opref, _) = self.read_ref_reg(ctx, src);
        let (cls_opref, _) = self.read_int_reg(ctx, cls);
        ctx.trace_record_exact_class(box_opref, cls_opref);
        TraceAction::Continue
    }

    // pyjitpl.py opimpl_ref_guard_value → implement_guard_value
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_ref_guard_value(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let (src, opcode_pc) = {
            let frame = self.frames.current_mut();
            let opcode_pc = frame.code_cursor - 1;
            (frame.next_reg() as usize, opcode_pc)
        };
        let (opref, concrete) = self.read_ref_reg(ctx, src);
        let const_ref = ctx.const_ref(concrete);
        self.record_state_guard(
            ctx,
            sym,
            OpCode::GuardValue,
            &[opref, const_ref],
            opcode_pc,
            false,
        );
        // `implement_guard_value`'s `if isinstance(box, Const): return box`.
        if !opref.is_constant() {
            self.replace_box(ctx, opref, const_ref, Type::Ref);
        }
        TraceAction::Continue
    }

    // pyjitpl.py opimpl_float_guard_value = _opimpl_guard_value
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_float_guard_value(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let (src, opcode_pc) = {
            let frame = self.frames.current_mut();
            let opcode_pc = frame.code_cursor - 1;
            (frame.next_reg() as usize, opcode_pc)
        };
        let (opref, concrete) = self.read_float_reg(ctx, src);
        let const_ref = ctx.const_float(concrete);
        self.record_state_guard(
            ctx,
            sym,
            OpCode::GuardValue,
            &[opref, const_ref],
            opcode_pc,
            false,
        );
        // `implement_guard_value`'s `if isinstance(box, Const): return box`.
        if !opref.is_constant() {
            self.replace_box(ctx, opref, const_ref, Type::Float);
        }
        TraceAction::Continue
    }

    // pyjitpl.py opimpl_guard_class:
    //     clsbox = self.cls_of_box(box)
    //     if not self.metainterp.heapcache.is_class_known(box):
    //         self.metainterp.generate_guard(rop.GUARD_CLASS, box, clsbox,
    //                                        resumepc=orgpc)
    //         self.metainterp.heapcache.class_now_known(box)
    //     return clsbox
    // `jtransform.py handle_getfield_typeptr` emits this for every
    // read of the header's class word; the result lands in the bank
    // the read was allocated to (`BC_GUARD_CLASS` int,
    // `BC_GUARD_CLASS_R` ref).
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_guard_class(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let byte = bytecode;
        let (opcode_pc, src, dst) = {
            let frame = self.frames.current_mut();
            let opcode_pc = frame.code_cursor - 1;
            let src = frame.next_reg() as usize;
            let dst = frame.next_reg() as usize;
            (opcode_pc, src, dst)
        };
        let (opref, concrete) = self.read_ref_reg(ctx, src);
        if concrete == 0 {
            return TraceAction::Abort;
        }
        let typeptr = self.read_typeptr_from_exception(concrete);
        let cls_const = ctx.const_int(typeptr);
        if !ctx.heap_cache().is_class_known(opref) {
            self.record_state_guard(
                ctx,
                sym,
                majit_ir::OpCode::GuardClass,
                &[opref, cls_const],
                opcode_pc,
                /* after_residual_call */ false,
            );
            ctx.heap_cache_mut().class_now_known(opref);
        }
        if byte == jitcode::insns::BC_GUARD_CLASS {
            self.set_int_reg(ctx, dst, Some(cls_const), Some(typeptr));
        } else {
            let cls_ref = ctx.const_ref(typeptr);
            self.set_ref_reg(ctx, dst, Some(cls_ref), Some(typeptr));
        }
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_raise(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        // pyjitpl.py opimpl_raise:
        //     if not self.metainterp.heapcache.is_class_known(exc_value_box):
        //         clsbox = self.cls_of_box(exc_value_box)
        //         self.metainterp.generate_guard(rop.GUARD_CLASS,
        //                                        exc_value_box, clsbox,
        //                                        resumepc=orgpc)
        //     self.metainterp.class_of_last_exc_is_const = True
        //     self.metainterp.last_exc_value = ...
        //     self.metainterp.last_exc_box = ...
        //     self.metainterp.popframe()
        //     self.metainterp.finishframe_exception()
        //
        // RPython `pyjitpl.py orgpc = position` parity: the
        // dispatcher's `next_u8()` already stepped past the
        // BC_RAISE byte, so `code_cursor - 1` is the byte position
        // of opimpl_raise itself — what `generate_guard(...,
        // resumepc=orgpc)` records.
        let opcode_pc = self.frames.current_mut().code_cursor - 1;
        let src = self.frames.current_mut().next_reg() as usize;
        let (opref, concrete) = self.read_ref_reg(ctx, src);
        if concrete == 0 {
            return TraceAction::Abort;
        }
        // pyjitpl.py MIFrame.opimpl_raise: record GUARD_CLASS unless heapcache
        // already knows the exception's class (heapcache.py is_class_known
        // is_class_known).  `cls_of_box` (model.py) reads
        // the typeptr at offset 0; `default_cls_of_box`
        // (`pyjitpl.rs`) implements the standalone fallback.
        // Recording the guard promotes the runtime class to a Const
        // — that is what justifies the unconditional
        // `class_of_last_exc_is_const = true` below.
        if !ctx.heap_cache().is_class_known(opref) {
            let typeptr = self.read_typeptr_from_exception(concrete);
            let cls_const = ctx.const_int(typeptr);
            // pyjitpl.py generate_guard(..., resumepc=orgpc):
            // GUARD_CLASS belongs to the regular-opimpl family, so
            // `after_residual_call=False` — the snapshot reads
            // liveness at `pc - SIZE_LIVE_OP` and the temporary
            // `frame.pc = orgpc` swap inside `record_state_guard`
            // pins it to the BC_RAISE byte.
            self.record_state_guard(
                ctx,
                sym,
                majit_ir::OpCode::GuardClass,
                &[opref, cls_const],
                opcode_pc,
                /* after_residual_call */ false,
            );
        }
        self.last_exception_box = Some(opref);
        self.last_exception_value = concrete;
        self.class_of_last_exc_is_const = true;
        {
            let frame = self.frames.current_mut();
            // pyopcode.py raise_varargs: RAISE_VARARGS with an
            // explicit value records at the raising instruction.
            super::record_application_traceback(
                concrete,
                ctx.virtualizable_heap_ptr().unwrap_or(std::ptr::null()),
                frame,
            );
        }
        self.pop_exception_frame(ctx);
        return self.unwind_to_exception_handler(ctx);
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_reraise(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        if self.last_exception_value == 0 {
            return TraceAction::Abort;
        }
        // RaiseWithExplicitTraceback is the bare-reraise path and
        // deliberately skips record_application_traceback.
        self.pop_exception_frame(ctx);
        return self.unwind_to_exception_handler(ctx);
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_abort(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        debug_assert!(trace_abort_bytecode(bytecode));
        // `abort/>r` is the same bailout as `abort/` (`handler_abort_result_marker_r`).
        // The destination byte is not consumed: the frame is left
        // immediately, matching the blackhole handler.
        self.log_bytecode_abort(if bytecode == jitcode::insns::BC_ABORT {
            "BC_ABORT"
        } else {
            "BC_ABORT_RESULT_R"
        });
        // A helper's `BC_ABORT` is a compile-time "this path cannot
        // be traced". The reconstructed registers that reached it
        // will reach it again; retrying rebuilds the same abort.
        if crate::is_bridge_walking() || ctx.is_bridge_trace {
            ctx.deterministic_bridge_abort = true;
        }
        return TraceAction::Abort;
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_abort_permanent(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        self.log_bytecode_abort("BC_ABORT_PERMANENT");
        return TraceAction::AbortPermanent;
    }

    // `jtransform.py` `rewrite_op_malloc_varsize` → `new_array` /
    // `new_array_clear`. A `#[jit_inline]` varsize literal emits the
    // byte; record `OpCode::NewArray{,Clear}` so `optimize_NEW_ARRAY`
    // can keep the block virtual, and allocate the live payload the
    // rest of this trace reads (`bhimpl_new_array{,_clear}`).
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_new_array(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let clear = bytecode == jitcode::insns::BC_NEW_ARRAY_CLEAR;
        let (length_reg, array_descr_idx, dest) = {
            let frame = self.frames.current_mut();
            frame.read_new_array()
        };
        let (array_base_size, array_itemsize, array_len_offset, array_type_id) = {
            let frame = self.frames.current_mut();
            let bh = frame.runtime_bh_descr(array_descr_idx).unwrap_or_else(|| {
                panic!("BC_NEW_ARRAY: descrs[{array_descr_idx}] is not a BhDescr entry")
            });
            let (base_size, itemsize, _signed) = bh.unpack_arraydescr_size();
            (
                base_size,
                itemsize,
                bh.array_len_offset(),
                bh.resolve_gc_tid(),
            )
        };
        let Some(array_descr) = self.dispatch_array_descr_ref(ctx, array_descr_idx) else {
            return TraceAction::Abort;
        };
        let (length_opref, length_val) = self.read_int_reg(ctx, length_reg);
        let length_count =
            usize::try_from(length_val).expect("BC_NEW_ARRAY: negative array length");
        let array_payload = array_itemsize
            .checked_mul(length_count)
            .and_then(|var| array_base_size.checked_add(var))
            .expect("BC_NEW_ARRAY: array size overflow");
        let array_payload = array_payload.max(1);
        // Same no-collect rule as `BC_NEW`: the register bank is not
        // a root set. Old-gen is non-moving; an untyped descr falls
        // through to the host allocator, which still zeroes when
        // `clear` is set and writes the length word.
        let array_gc_ptr = if array_type_id != 0 {
            majit_gc::alloc_oldgen_typed(array_type_id, array_payload).0
        } else {
            0
        };
        // Under an installed collector a typed descr
        // (`array_type_id != 0`) has a GC header and a tracing layout;
        // `alloc_oldgen_typed` returning 0 is a failed allocation, and
        // host storage would drop both. Abort the trace. With no
        // collector nothing traces the block: a published
        // `malloc_fixedsize` (`GcLLDescr_boehm.malloc_fn_ptr`) or the
        // host allocator owns it.
        let array_ptr = if array_gc_ptr != 0 {
            array_gc_ptr as i64
        } else if array_type_id != 0 && majit_gc::collector_installed() {
            return TraceAction::Abort;
        } else if let Some(ptr) = host_malloc_fixedsize(array_payload) {
            ptr as i64
        } else {
            let layout = std::alloc::Layout::from_size_align(array_payload, 8)
                .expect("BC_NEW_ARRAY: invalid array layout");
            unsafe { std::alloc::alloc_zeroed(layout) as i64 }
        };
        if array_ptr != 0 {
            if clear {
                unsafe {
                    std::ptr::write_bytes(array_ptr as *mut u8, 0, array_payload);
                }
            }
            if let Some(len_ofs) = array_len_offset {
                unsafe {
                    *((array_ptr as *mut u8).add(len_ofs) as *mut usize) = length_val as usize;
                }
            }
        }
        let kind = if clear {
            OpCode::NewArrayClear
        } else {
            OpCode::NewArray
        };
        ctx.profiler().count_ops(kind, crate::counters::OPS);
        ctx.profiler()
            .count_ops(kind, crate::counters::RECORDED_OPS);
        let abox_op = if clear {
            ctx.record_new_array_clear(length_opref, array_descr)
        } else {
            ctx.record_new_array(length_opref, array_descr)
        };
        ctx.set_opref_concrete(abox_op, Value::Ref(majit_ir::GcRef(array_ptr as usize)));
        ctx.heap_cache_mut()
            .new_array(abox_op, length_opref, length_opref.is_constant());
        self.set_ref_reg(ctx, dest, Some(abox_op), Some(array_ptr));
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_newlist_clear(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        // opimpl_newlist_clear (pyjitpl.py): decompose ONE
        // opcode into New + SetfieldGc(length) + NewArrayClear +
        // SetfieldGc(items).  The tracer both *records* all four
        // resops (so the optimizer can virtualize the list header +
        // its items block when they do not escape) and *executes* the
        // two live allocations plus the two field stores (so the
        // subsequent steps of this same trace read live memory), the
        // same dual discipline BC_NEW / BC_SETFIELD_GC follow.
        let (
            length_reg,
            struct_descr_idx,
            length_descr_idx,
            items_descr_idx,
            array_descr_idx,
            dest,
        ) = self.frames.current_mut().read_newlist_clear();
        // structdescr (`list` header SizeDescr): size + gc identity +
        // the DescrRef the recorded New carries.
        let (struct_size, struct_type_id, struct_headerless, struct_descr) = {
            let frame = self.frames.current_mut();
            let bh = frame.runtime_bh_descr(struct_descr_idx).unwrap_or_else(|| {
                panic!("BC_NEWLIST_CLEAR: descrs[{struct_descr_idx}] is not a BhDescr entry")
            });
            (
                bh.as_size(),
                bh.resolve_gc_tid(),
                bh.is_headerless(),
                size_descr_ref_from_bh(bh),
            )
        };
        // lengthdescr / itemsdescr: plain FieldDescrs — byte offset +
        // the DescrRef the two recorded SetfieldGc ops carry.
        let (length_offset, length_fielddescr) = {
            let frame = self.frames.current_mut();
            let bh = frame.runtime_bh_descr(length_descr_idx).unwrap_or_else(|| {
                panic!("BC_NEWLIST_CLEAR: descrs[{length_descr_idx}] is not a BhDescr entry")
            });
            let offset = field_offset_from_bh(bh, "BC_NEWLIST_CLEAR length");
            let descr = frame
                .runtime_optimizer_descr(length_descr_idx)
                .unwrap_or_else(|| field_descr_ref_from_bh(bh).1);
            (offset, descr)
        };
        let (items_offset, items_fielddescr) = {
            let frame = self.frames.current_mut();
            let bh = frame.runtime_bh_descr(items_descr_idx).unwrap_or_else(|| {
                panic!("BC_NEWLIST_CLEAR: descrs[{items_descr_idx}] is not a BhDescr entry")
            });
            let offset = field_offset_from_bh(bh, "BC_NEWLIST_CLEAR items");
            let descr = frame
                .runtime_optimizer_descr(items_descr_idx)
                .unwrap_or_else(|| field_descr_ref_from_bh(bh).1);
            (offset, descr)
        };
        // arraydescr: geometry for the live items-block allocation
        // (`base_size + length*itemsize` bytes, cleared, length word
        // at `len_offset`), matching `bh_new_array`
        // (runner.rs `dynasm_alloc_oldgen_varsize_typed_and_set_len`).
        let (array_base_size, array_itemsize, array_len_offset, array_type_id) = {
            let frame = self.frames.current_mut();
            let bh = frame.runtime_bh_descr(array_descr_idx).unwrap_or_else(|| {
                panic!("BC_NEWLIST_CLEAR: descrs[{array_descr_idx}] is not a BhDescr entry")
            });
            let (base_size, itemsize, _signed) = bh.unpack_arraydescr_size();
            (
                base_size,
                itemsize,
                bh.array_len_offset(),
                bh.resolve_gc_tid(),
            )
        };
        let Some(array_descr) = self.dispatch_array_descr_ref(ctx, array_descr_idx) else {
            return TraceAction::Abort;
        };
        // The length operand feeds both the length-field store and the
        // items-block element count (opimpl_newlist_clear passes the
        // same `sizebox` to `_opimpl_setfield_gc_any` and
        // `opimpl_new_array_clear`).
        let (length_opref, length_val) = self.read_int_reg(ctx, length_reg);

        // ── 1. sbox: live-alloc the `list` header (BC_NEW no-collect
        // discipline — the struct ptr is live in the register bank,
        // which is no root set, so the array allocation that follows
        // must not move it; old-gen is mark-sweep non-moving and both
        // GC paths are no-collect). ──
        let struct_size = struct_size.max(1);
        let struct_gc_ptr = if struct_headerless {
            majit_gc::alloc_nursery_headerless_no_collect(struct_size).0
        } else if struct_type_id != 0 {
            majit_gc::alloc_oldgen_typed(struct_type_id, struct_size).0
        } else {
            0
        };
        let struct_ptr = if struct_gc_ptr != 0 {
            struct_gc_ptr as i64
        } else if let Some(ptr) = host_malloc_fixedsize(struct_size) {
            ptr as i64
        } else {
            let layout = std::alloc::Layout::from_size_align(struct_size, 8)
                .expect("BC_NEWLIST_CLEAR: invalid list-header layout");
            unsafe { std::alloc::alloc_zeroed(layout) as i64 }
        };
        ctx.profiler().count_ops(OpCode::New, crate::counters::OPS);
        ctx.profiler()
            .count_ops(OpCode::New, crate::counters::RECORDED_OPS);
        let sbox_op = ctx.record_op_with_descr(OpCode::New, &[], struct_descr);
        ctx.set_opref_concrete(sbox_op, Value::Ref(majit_ir::GcRef(struct_ptr as usize)));
        // `opimpl_newlist_clear` composes `opimpl_new`,
        // `_opimpl_setfield_gc_any`, `opimpl_new_array_clear` and a
        // second `_opimpl_setfield_gc_any`; each carries the heapcache
        // effect stamped alongside it here.
        ctx.heap_cache_mut().new_object(sbox_op);

        // ── 2. store the length into the header's `length` field. ──
        ctx.profiler()
            .count_ops(OpCode::SetfieldGc, crate::counters::OPS);
        ctx.profiler()
            .count_ops(OpCode::SetfieldGc, crate::counters::RECORDED_OPS);
        let length_field_key = heapcache_field_key(&length_fielddescr);
        ctx.heapcache_invalidate_caches_varargs(OpCode::SetfieldGc, None, &[sbox_op, length_opref]);
        ctx.record_op_with_descr(
            OpCode::SetfieldGc,
            &[sbox_op, length_opref],
            length_fielddescr,
        );
        if let Some(field_key) = length_field_key {
            ctx.heapcache_setfield_cached(sbox_op, field_key, length_opref);
        }
        if struct_ptr != 0 {
            unsafe { *((struct_ptr as *mut u8).add(length_offset) as *mut i64) = length_val };
        }

        // ── 3. abox: live-alloc the cleared items block.  Payload is
        // `base_size + length*itemsize`, zero-filled by the old-gen
        // allocator (`finish_alloc_in_oldgen` write_bytes 0 — the
        // CLEAR), no-collect; the length word is written at
        // `len_offset` for a length-prefixed array. ──
        let length_count =
            usize::try_from(length_val).expect("BC_NEWLIST_CLEAR: negative list length");
        let array_payload = array_itemsize
            .checked_mul(length_count)
            .and_then(|var| array_base_size.checked_add(var))
            .expect("BC_NEWLIST_CLEAR: items-block size overflow");
        let array_payload = array_payload.max(1);
        let array_gc_ptr = if array_type_id != 0 {
            majit_gc::alloc_oldgen_typed(array_type_id, array_payload).0
        } else {
            0
        };
        let array_ptr = if array_gc_ptr != 0 {
            array_gc_ptr as i64
        } else if let Some(ptr) = host_malloc_fixedsize(array_payload) {
            ptr as i64
        } else {
            let layout = std::alloc::Layout::from_size_align(array_payload, 8)
                .expect("BC_NEWLIST_CLEAR: invalid items-block layout");
            unsafe { std::alloc::alloc_zeroed(layout) as i64 }
        };
        if array_ptr != 0
            && let Some(len_ofs) = array_len_offset
        {
            unsafe { *((array_ptr as *mut u8).add(len_ofs) as *mut i64) = length_val };
        }
        ctx.profiler()
            .count_ops(OpCode::NewArrayClear, crate::counters::OPS);
        ctx.profiler()
            .count_ops(OpCode::NewArrayClear, crate::counters::RECORDED_OPS);
        let abox_op = ctx.record_new_array_clear(length_opref, array_descr);
        ctx.set_opref_concrete(abox_op, Value::Ref(majit_ir::GcRef(array_ptr as usize)));
        // `execute_new_array_clear`'s `heapcache.new_array`. Only a
        // constant length makes the array a virtual candidate, which is
        // the `isinstance(lengthbox, Const)` the flag stands for.
        ctx.heap_cache_mut()
            .new_array(abox_op, length_opref, length_opref.is_constant());

        // ── 4. store the items block into the header's `items` field.
        // A ref store adds a heap edge header→items, so notify the GC
        // on the container (mirrors BC_SETFIELD_GC_R / bh_setfield_gc_r).
        ctx.profiler()
            .count_ops(OpCode::SetfieldGc, crate::counters::OPS);
        ctx.profiler()
            .count_ops(OpCode::SetfieldGc, crate::counters::RECORDED_OPS);
        let items_field_key = heapcache_field_key(&items_fielddescr);
        ctx.heapcache_invalidate_caches_varargs(OpCode::SetfieldGc, None, &[sbox_op, abox_op]);
        ctx.record_op_with_descr(OpCode::SetfieldGc, &[sbox_op, abox_op], items_fielddescr);
        if let Some(field_key) = items_field_key {
            ctx.heapcache_setfield_cached(sbox_op, field_key, abox_op);
        }
        if struct_ptr != 0 {
            unsafe { *((struct_ptr as *mut u8).add(items_offset) as *mut i64) = array_ptr };
            if majit_gc::gc_owns_object(struct_ptr as usize) {
                majit_gc::gc_write_barrier(majit_ir::GcRef(struct_ptr as usize));
            }
        }

        // ── 5. bind the list header to the destination ref register. ──
        self.set_ref_reg(ctx, dest, Some(sbox_op), Some(struct_ptr));
        TraceAction::Continue
    }

    // pyjitpl.py `_opimpl_isconstant` / `opimpl_int_isconstant`:
    // `return ConstInt(isinstance(box, Const))`.  No IR op.
    // Byte layout `[src][dst]` per `bhhandler_i_i!`.
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_int_isconstant(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let (src, dest) = {
            let frame = self.frames.current_mut();
            let src = frame.next_reg() as usize;
            let dest = frame.next_reg() as usize;
            (src, dest)
        };
        let (opref, _) = self.read_int_reg(ctx, src);
        let value = opref.is_constant() as i64;
        let dest_box = ctx.const_int(value);
        self.set_int_reg(ctx, dest, Some(dest_box), Some(value));
        TraceAction::Continue
    }

    // pyjitpl.py `_opimpl_isconstant` / `opimpl_ref_isconstant`:
    // `return ConstInt(isinstance(box, Const))`.  No IR op.
    // Byte layout `[src][dst]` per `bhhandler_r_i!`.
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_ref_isconstant(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let (src, dest) = {
            let frame = self.frames.current_mut();
            let src = frame.next_reg() as usize;
            let dest = frame.next_reg() as usize;
            (src, dest)
        };
        let (opref, _) = self.read_ref_reg(ctx, src);
        let value = opref.is_constant() as i64;
        let dest_box = ctx.const_int(value);
        self.set_int_reg(ctx, dest, Some(dest_box), Some(value));
        TraceAction::Continue
    }

    // pyjitpl.py `_opimpl_isvirtual` / `opimpl_ref_isvirtual`:
    // `return ConstInt(heapcache.is_likely_virtual(box))`.  No IR op.
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_ref_isvirtual(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let (src, dest) = {
            let frame = self.frames.current_mut();
            let src = frame.next_reg() as usize;
            let dest = frame.next_reg() as usize;
            (src, dest)
        };
        let (opref, _) = self.read_ref_reg(ctx, src);
        let value = ctx.is_likely_virtual(opref) as i64;
        let dest_box = ctx.const_int(value);
        self.set_int_reg(ctx, dest, Some(dest_box), Some(value));
        TraceAction::Continue
    }

    // `blackhole.py` `bhimpl_strlen` / `pyjitpl.py` strlen recording.
    // Encoding `strlen/r>i`: [string_reg u8][dst u8]. Concrete length
    // is `Backend::bh_strlen` (the `rstr.STR` length word). `Cpu::bh_strlen`
    // returns `None` unless `str_descr` is registered, and the example
    // cpu does not register one; the compiled load still uses
    // `inject_builtin_string_descrs`.
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_strlen(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let (src, dst) = {
            let frame = self.frames.current_mut();
            let src = frame.next_reg() as usize;
            let dst = frame.next_reg() as usize;
            (src, dst)
        };
        let (string, addr) = self.read_ref_reg(ctx, src);
        assert_ne!(addr, 0, "strlen: null string");
        let value = unsafe {
            ((addr as usize).wrapping_add(std::mem::size_of::<usize>()) as *const usize)
                .read_unaligned() as i64
        };
        let opref = ctx.execute_and_record(
            Some(self.cpu.as_ref()),
            OpCode::Strlen,
            None,
            &[string],
            Some(majit_ir::Value::Int(value)),
            self.last_exception_value,
        );
        self.set_int_reg(ctx, dst, Some(opref), Some(value));
        TraceAction::Continue
    }

    // `blackhole.py` `bhimpl_strgetitem`. Encoding `strgetitem/ri>i`:
    // [string_reg u8][index_reg u8][dst u8].
    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_strgetitem(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        let (src, index_reg, dst) = {
            let frame = self.frames.current_mut();
            let src = frame.next_reg() as usize;
            let index_reg = frame.next_reg() as usize;
            let dst = frame.next_reg() as usize;
            (src, index_reg, dst)
        };
        let (string, addr) = self.read_ref_reg(ctx, src);
        let (index, index_value) = self.read_int_reg(ctx, index_reg);
        assert_ne!(addr, 0, "strgetitem: null string");
        let len = unsafe {
            ((addr as usize).wrapping_add(std::mem::size_of::<usize>()) as *const usize)
                .read_unaligned() as i64
        };
        assert!(
            index_value >= 0 && index_value < len,
            "strgetitem: index {index_value} outside 0..{len}"
        );
        let chars = 2 * std::mem::size_of::<usize>();
        let value = unsafe {
            ((addr as usize)
                .wrapping_add(chars)
                .wrapping_add(index_value as usize) as *const u8)
                .read_unaligned() as i64
        };
        let opref = ctx.execute_and_record(
            Some(self.cpu.as_ref()),
            OpCode::Strgetitem,
            None,
            &[string, index],
            Some(majit_ir::Value::Int(value)),
            self.last_exception_value,
        );
        self.set_int_reg(ctx, dst, Some(opref), Some(value));
        TraceAction::Continue
    }

    #[inline(never)]
    #[allow(unused_variables)]
    fn opimpl_unknown(
        &mut self,
        ctx: &mut TraceCtx,
        sym: &mut S,
        _runtime: &R,
        bytecode: u8,
    ) -> TraceAction {
        panic!("unknown jitcode bytecode {bytecode}");
    }
}
