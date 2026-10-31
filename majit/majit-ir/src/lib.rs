#[macro_use]
pub mod icf;
pub use icf::{icf_identity_token, icf_path_hash};
pub mod bitstring;
pub mod const_ptr_table;
pub mod debug;
pub mod descr;
pub mod descr_registry;
pub mod effectinfo;
pub mod environ;
pub mod eval_breaker_word;
pub mod field_entry;
pub mod forwarding;
pub mod helper_fnaddr;
pub mod indexmap_ext;
pub mod intbound;
pub mod jit;
pub mod llop;
pub mod op_descr;
pub mod op_info;
pub mod op_type_index;
pub mod operand;
#[cfg(any(test, feature = "jit-audits"))]
pub mod opref_audit;
pub mod optimize;
pub mod ptr_info;
pub mod rawbuffer;
#[cfg(any(test, feature = "jit-audits"))]
pub mod reg_write_audit;
pub mod resoperation;
pub mod resumecode;
pub mod resumedata;
pub mod rvec;
#[cfg(feature = "test-support")]
pub mod test_support;
pub mod value;

// Proc-macro expansions name `::majit_ir::linkme` so consumers do not
// need a direct `linkme` dependency.
#[doc(hidden)]
pub use linkme;

// Re-export key types at crate root for convenience.
pub use descr::{
    AccumInfo, ArrayDescr, ArrayFlag, CallDescr, DebugMergePointDescr, DebugMergePointInfo, Descr,
    DescrRef, FailDescr, FailDescrCell, FailDescrStore, FieldDescr, GC_FLAG_SHIFT,
    GC_HEADER_WORD_SIZE, GC_IMMORTAL_LEAF_HEADER, GC_TYPE_ID_BITS, GCFLAG_NO_HEAP_PTRS,
    GCFLAG_TRACK_YOUNG_PTRS, GcCache, HandleFailClass, InteriorFieldDescr, JitCodeDescr, LLType,
    LoopTargetDescr, LoopTokenDescr, QuasiImmutDescr, QuasiImmutHandle, QuasiImmutLoopToken,
    RdLocs, SimpleCallDescr, SimpleFailDescr, SimpleFieldDescr, SizeDescr, SwitchDescr,
    TargetArgLoc, UnpackAtExitInfo, descr_identity, descr_instance_ptr, make_array_descr,
    make_array_descr_signed, make_call_descr, make_field_descr, make_field_descr_full,
    make_loop_target_descr, make_malloc_array_calldescr, make_malloc_array_nonstandard_calldescr,
    make_malloc_big_fixedsize_calldescr, make_malloc_str_calldescr, make_malloc_unicode_calldescr,
    make_memcpy_calldescr, make_size_descr_full, make_size_descr_with_vtable, make_tid_field_descr,
    make_vtable_field_descr, memcpy_fn_addr, recover_fail_descr_cell, unpack_fielddescr,
};
pub use effectinfo::{
    CallInfoCollection, EffectInfo, ExtraEffect, OopSpecIndex, RuntimeHelperKind,
    UnsupportedFieldExc, consider_array, consider_struct, frozenset_or_none,
};
pub use indexmap_ext::{ConstMap, IndexMapExt};
pub use op_type_index::{OpTypeIndex, PosIndex};
pub use resoperation::{
    AbstractValue, ArrayDescrInfo, BoxEnv, FieldDescrInfo, GuardPendingFieldEntry, OPCODE_COUNT,
    Op, OpCode, OpRc, OpRef, RdVirtualInfo, VectorizationInfo, VirtualFieldsInfo, format_trace,
};
pub use resumecode::NumberingRef;
pub use value::{
    CallArgWord, CallResultWord, Const, FAILARGS_LIMIT, GREEN_INLINE, GREEN_UHASH_MULT,
    GREEN_UHASH_SEED, GcRef, GreenAsI64, GreenKey, GreenSlot, GreenSlotFloat, GreenSlotInt,
    GreenSlotRef, GreenSlotStr, GreenSlotUnicode, GreenSpec, GreenType, InputArg, InputArgRc,
    JitDriverVar, PYPYJIT_GREEN_TYPES, RefCurrentFn, RefReleaseFn, RefRetainFn, RetainedGreens,
    SharedConstPool, StrEqFn, StrHashFn, Type, Value, VarKind, equal_whatever,
    gc_current_object_address, gc_id_or_identityhash, green_slot_of, green_type_to_ir,
    green_uhash_step, make_str_slot, pypyjit_greenargs, pypyjit_greenkey, pypyjit_greenkey_uhash,
    set_gc_current_object_address, set_gc_id_or_identityhash, set_ref_resolver, set_str_resolver,
    set_unicode_resolver,
};
