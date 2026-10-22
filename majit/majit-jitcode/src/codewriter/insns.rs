//! Canonical opcode-name table for jitcode bytecode dispatch.
//!
//! RPython parity: `rpython/jit/codewriter/assembler.py __init__`
//! `Assembler.insns` lives in the same module as `Assembler`. Pyre's
//! mechanical mirror of `rpython/jit/codewriter/` is split between
//! `majit/majit-translate/src/codewriter/` (the graph-reading half) and
//! this directory (the runtime half, `Assembler`'s state included); the
//! canonical home for `wellknown_bh_insns` + `BC_*` constants +
//! `insn_byte` is therefore here, not in `majit-metainterp`.
//!
//! See `epic_e_task86_canonical_home_design_2026_05_04.md` for the
//! full design rationale + 5-slice mechanical-move plan. Slice #86c
//! moved the `BC_*` constants from `majit-metainterp::jitcode` to
//! this module; subsequent slices migrate `wellknown_bh_insns` /
//! `insn_byte` (#86d), import paths (#86e), classification doc
//! comments (#86f), and remove the back-compat re-exports (#88).
//!
//! TODO: pyre's runtime jitcode builder uses
//! fixed `BC_*` byte values (compile-time stable) instead of
//! RPython's dynamic `setdefault(key, len(self.insns))` allocation.
//! Compile-time stability is required because pyre serialises
//! `jitcodes.bin` at build time and the runtime decoder
//! reads those exact bytes. RPython runs the assembler at startup
//! and never serialises across process boundaries, so dynamic
//! allocation is fine there. Codex review Correction 1
//! (`epic_e_codex_review_2026_05_04.md`) flags that this byte-
//! stability adaptation should be documented at the table site —
//! see slice #86f.

use indexmap::IndexMap;

// Canonical RPython keys filling free low bytes (`blackhole.py bhimpl_check_neg_index`).
// One byte per `(opname, argcodes)` per `assembler.py:221
// setdefault(key, len(self.insns))` — alias-shared bytes would violate the
// 1:1 invariant that `pyjitpl.py setup_insns` enforces.
pub const BC_CAST_FLOAT_TO_INT: u8 = 0;
pub const BC_ARRAYLEN_GC: u8 = 1;
/// RPython `blackhole.py` `bhimpl_int_is_true(a) -> !!a` — the
/// boolean-coercion of an int, produced by a `Bool` exitswitch source
/// that is read before the branch rather than fused into
/// `goto_if_not_int_is_true`. Takes slot 2 (was free; `BC_ARRAYLEN_VABLE`
/// is pinned at byte 74 in the vable group).
pub const BC_INT_IS_TRUE: u8 = 2;
/// Inline-immediate `setfield_vable_i`. `setfield_vable_i` is not in
/// `USE_C_FORM` (`assembler.py`), so a constant value takes a pool slot
/// and `check_result` rejects `num_regs_i + len(constants_i) > 256`.
/// One constant per Python instruction (`dispatch_bytecode`'s
/// `last_instr = intmask(next_instr)`) overflows that cap. Payload is
/// `r` + u16 imm + `d` (the same `VableField` descr as
/// `setfield_vable_i/rid`). The value is a bytecode offset or
/// `valuestackdepth`; both already fit in u16 because jump targets
/// (`assembler.py` `fix_labels`) are u16.
pub const BC_SETFIELD_VABLE_I_IMM: u8 = 32;
pub const BC_GETINTERIORFIELD_GC_I: u8 = 3;
pub const BC_GETINTERIORFIELD_GC_R: u8 = 4;
pub const BC_GETINTERIORFIELD_GC_F: u8 = 5;
pub const BC_GETLISTITEM_GC_I: u8 = 6;
pub const BC_GETLISTITEM_GC_R: u8 = 7;
pub const BC_GETLISTITEM_GC_F: u8 = 8;
pub const BC_NEWLIST: u8 = 9;
pub const BC_NEWLIST_CLEAR: u8 = 10;
pub const BC_NEWLIST_HINT: u8 = 11;

// Single-object / array allocation — `blackhole.py:1301-1327`
// `bhimpl_new` / `bhimpl_new_with_vtable` / `bhimpl_new_array`.
// Bytes above the prior high-water so they never disturb existing
// serialised byte assignments; 213 belongs to `BC_INT_BETWEEN`, 214 to
// `BC_NEW_ARRAY_CLEAR` (both defined below with the BUILD_TUPLE notes),
// and 215/216 to the c-argcode array forms `BC_SETARRAYITEM_GC_R_C` /
// `BC_NEW_ARRAY_CLEAR_C`, so `new` / `new_array` / `new_with_vtable`
// take the next free bytes 217/218/219. The blackhole handler bodies
// already exist (`handler_new` etc.) but were unwired until these keys
// joined `wellknown_bh_insns()`.
pub const BC_NEW: u8 = 217;
pub const BC_NEW_ARRAY: u8 = 218;
pub const BC_NEW_WITH_VTABLE: u8 = 219;

// Recursive-portal call (`#[jit_interp]` mainloop calling itself, e.g. tl.py:177
// `interp(code, pc+offset)`).  Siblings of the `BC_CALL_ASSEMBLER_*` family
// (bytes 50-53): the depth-exceeded branch of a recursive call records one of
// these as an assembler call into the same green-key loop, while the
// depth-bounded branch inline-traces the callee (`should_inline_core`).  Bytes
// continue the above-high-water sequence (223-226, above the `setfield_gc_i/rcd`
// C-form byte 222) so existing serialised byte assignments are undisturbed.
// These consts only DECLARE the intended byte
// numbers; a byte is not actually reserved (`is_reserved_opcode_byte`) until its
// key joins `extension_insns()`, and the blackhole handler is bound there.
// That registration + the macro lowering + the `run_one_step` dispatch arm are
// wired together (the registration may shift dynamic byte assignments, so it is
// verified against the serialised JitCode artifacts at that point, not here).  The
// `recursive_call_bytes_reserved_without_collision` test guards these chosen
// numbers against the currently-reserved tables.
pub const BC_RECURSIVE_CALL_INT: u8 = 223;
pub const BC_RECURSIVE_CALL_REF: u8 = 224;
pub const BC_RECURSIVE_CALL_FLOAT: u8 = 225;
pub const BC_RECURSIVE_CALL_VOID: u8 = 226;

/// `blackhole.py` `bhimpl_strlen` / `bhimpl_strgetitem`
/// (`@arguments("cpu", "r", returns="i")` /
/// `@arguments("cpu", "r", "i", returns="i")`).  High-water bytes so
/// existing serialised assignments stay put.  `pyre_cpu` overrides the
/// default inline-chars layout to follow `W_UnicodeObject.value`.
pub const BC_STRLEN: u8 = 242;
pub const BC_STRGETITEM: u8 = 243;
/// `strgetitem` with a `USE_C_FORM` index (`assembler.py`).
pub const BC_STRGETITEM_C: u8 = 244;

/// `setinteriorfield_gc_{i,r,f}` — `blackhole.py`
/// `bhimpl_setinteriorfield_gc_{i,r,f}`
/// (`@arguments("cpu", "r", "i", "i"|"r"|"f", "d")`).
/// The load siblings occupy bytes 3–5; these stores take the next free
/// bytes above [`BC_STRGETITEM_C`] so existing serialised assignments stay put.
pub const BC_SETINTERIORFIELD_GC_I: u8 = 245;
pub const BC_SETINTERIORFIELD_GC_R: u8 = 246;
pub const BC_SETINTERIORFIELD_GC_F: u8 = 247;
/// `blackhole.py` `bhimpl_int_signext`. Lowest free byte at or above 248.
pub const BC_INT_SIGNEXT: u8 = 248;

/// `blackhole.py` `bhimpl_setfield_raw_i` / `bhimpl_setfield_raw_f` /
/// `bhimpl_getfield_raw_f`. Int-bank siblings of the GC field stores, and
/// the float twin of `getfield_raw_i`. High-water bytes above
/// [`BC_INT_SIGNEXT`]; 249 is [`BC_ABORT_RESULT_I`].
pub const BC_SETFIELD_RAW_I: u8 = 252;
pub const BC_SETFIELD_RAW_F: u8 = 250;
pub const BC_GETFIELD_RAW_F: u8 = 251;
/// `blackhole.py` `bhimpl_int_force_ge_zero(i)`: `0` if `i < 0` else `i`.
/// `pyjitpl.py` generates `opimpl_int_force_ge_zero` beside `int_neg`.
/// `rlist.py ll_alloc_and_set` clamps `count` with this unary.
/// Next byte above [`BC_SETFIELD_RAW_I`].
pub const BC_INT_FORCE_GE_ZERO: u8 = 253;
/// `jtransform.py` `jit.isvirtual` on an int-kind operand (`%s_isvirtual`
/// % kind). `pyjitpl.py` only aliases `opimpl_ref_isvirtual`; a raw
/// `Vec` header is an address word, so the same `_opimpl_isvirtual`
/// box test runs on an int register.
/// Next free high-water byte after [`BC_INT_FORCE_GE_ZERO`].
/// (`assembler.py` `setdefault(key, len(self.insns))` never reuses a byte).
pub const BC_INT_ISVIRTUAL: u8 = 254;

/// "This cached control opcode is absent" sentinel for the
/// `blackhole.py:72-74` fields (`op_live`, `op_catch_exception`,
/// `op_rvmprof_code`).
///
/// Upstream spells the fallback `insns.get('live/', -1)` and then tests
/// `opcode == self.op_live` against `ord(code[position])`.  `-1` lies
/// outside that `0..=255` range, so an absent entry can never match a real
/// instruction.  A `u8` has no value outside the range, so the same
/// guarantee is bought by reserving the byte instead: no opname may be
/// assigned `BC_ABSENT`, and [`is_reserved_opcode_byte`] keeps both dynamic
/// allocators (`Assembler::next_dynamic_opnum` here and pyre's runtime
/// `record_insn_key`) off it.
pub const BC_ABSENT: u8 = u8::MAX;

pub const BC_LOOP_HEADER: u8 = 12;
pub const BC_ABORT: u8 = 13;
pub const BC_ABORT_PERMANENT: u8 = 14;
/// RPython `blackhole.py` `bhimpl_unreachable()` raises
/// `AssertionError("unreachable")`. Distinct from `BC_ABORT_PERMANENT`
/// which permits the interpreter to take over via
/// `DispatchError::RaiseException`.
pub const BC_UNREACHABLE: u8 = 19;
/// RPython `blackhole.py:913` `bhimpl_goto_if_not_int_is_true =
/// bhimpl_goto_if_not` aliases the handler.  `assembler.py:221
/// setdefault()` still allocates a distinct byte per `(opname,
/// argcodes)` key — `goto_if_not_int_is_true/iL` gets its own byte
/// here (alias); the canonical `goto_if_not/iL` lives at
/// [`BC_GOTO_IF_NOT`] = 18.  Both bytes route to the same handler.
pub const BC_GOTO_IF_NOT_INT_IS_TRUE: u8 = 15;
pub const BC_JUMP: u8 = 16;
pub const BC_INLINE_CALL: u8 = 17;
/// RPython `flatten.py` `opname = 'goto_if_not'` — the canonical
/// boolean-exitswitch branch.  `blackhole.py bhimpl_goto_if_not`
/// is the handler shared with the
/// [`BC_GOTO_IF_NOT_INT_IS_TRUE`] alias per `blackhole.py
/// bhimpl_goto_if_not_int_is_true = bhimpl_goto_if_not`.  Slot 18
/// (formerly `BC_RESIDUAL_CALL_VOID`, freed by the call-family
/// canonical migration) now houses this canonical key so each
/// `(opname, argcodes)` per `assembler.py:221 setdefault()` keeps
/// its own byte.
pub const BC_GOTO_IF_NOT: u8 = 18;
pub const BC_MOVE_I: u8 = 21;
// Canonical RPython keys taking previously-freed slots 22-26 and 28-31
// (`blackhole.py bhimpl_check_neg_index` + `blackhole.py` conditional/known
// result + `blackhole.py bhimpl_record_known_result_i_ir_v` record_known_result + raw/indexed
// loads + bounds check).  Each `(opname, argcodes)` gets a distinct
// byte per `assembler.py:221 setdefault()` semantics.
pub const BC_RAW_LOAD_I: u8 = 20;
pub const BC_RAW_LOAD_F: u8 = 22;
pub const BC_GC_LOAD_INDEXED_I: u8 = 23;
pub const BC_GC_LOAD_INDEXED_F: u8 = 24;
pub const BC_CHECK_NEG_INDEX: u8 = 25;
pub const BC_CONDITIONAL_CALL_IR_V: u8 = 26;
pub const BC_CONDITIONAL_CALL_VALUE_IR_I: u8 = 28;
pub const BC_CONDITIONAL_CALL_VALUE_IR_R: u8 = 29;
pub const BC_RECORD_KNOWN_RESULT_I_IR_V: u8 = 30;
pub const BC_RECORD_KNOWN_RESULT_R_IR_V: u8 = 31;
// Ref-typed bytecodes
pub const BC_MOVE_R: u8 = 27;
// Float-typed bytecodes
pub const BC_MOVE_F: u8 = 33;
/// `blackhole.py` `bhimpl_setarrayitem_raw_i`
/// (`@arguments("cpu", "i", "i", "i", "d")`). Slot 34 was the former
/// `BC_CALL_FLOAT` byte; `assembler.py` `setdefault(key, len(self.insns))`
/// assigned this canonical key the lowest free byte, which is the byte
/// `pipeline.insns` already serialises for `setarrayitem_raw_i/iiid`.
pub const BC_SETARRAYITEM_RAW_I: u8 = 34;
/// `blackhole.py` `bhimpl_getarrayitem_raw_i`
/// (`@arguments("cpu", "i", "i", "d", returns="i")`). Slot 35 was the
/// former `BC_CALL_PURE_FLOAT` byte; same `setdefault` assignment as
/// [`BC_SETARRAYITEM_RAW_I`].
pub const BC_GETARRAYITEM_RAW_I: u8 = 35;
/// `blackhole.py` `bhimpl_getarrayitem_raw_f`
/// (`@arguments("cpu", "i", "i", "d", returns="f")`).
pub const BC_GETARRAYITEM_RAW_F: u8 = 36;
/// `blackhole.py` `bhimpl_setarrayitem_raw_f`
/// (`@arguments("cpu", "i", "i", "f", "d")`).
pub const BC_SETARRAYITEM_RAW_F: u8 = 37;
/// `int_copy/d>i` — u16 immediate source. `int_copy` is in `USE_C_FORM`
/// (`assembler.py`), so a signed-byte value is already `int_copy/c>i`.
/// A bytecode offset past 127 would otherwise take a `constants_i` slot
/// (`emit_const` `assert 0 <= val < 256`). Jump targets (`fix_labels`)
/// are u16, so a PC green fits this form and stays a `ConstInt` at the
/// merge without consuming the shared one-byte pool.
pub const BC_MOVE_I_D: u8 = 38;
// slots 39..=40 (formerly BC_CALL_MAY_FORCE_{REF,FLOAT}) freed —
// the may_force policy now rides on
// `EffectInfo.extraeffect = EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE` carried
// by the canonical `BC_RESIDUAL_CALL_{R,IR,IRF}_{I,R,F}` family
// (`effectinfo.py`). Slot 38 (formerly BC_CALL_MAY_FORCE_INT) is
// `BC_MOVE_I_D`.
// slot 41 (formerly BC_CALL_MAY_FORCE_VOID) freed — same
// EffectInfo-carries-policy rationale as above; see slots 39..=40.
// slot 42 (formerly BC_CALL_RELEASE_GIL_INT) freed
// — release-GIL policy rides on
// `EffectInfo.call_release_gil_target` (`effectinfo.py
// is_call_release_gil`).
// BC_CALL_RELEASE_GIL_REF (slot 43) intentionally absent:
// resoperation.py (`# no such thing`) excludes
// CALL_RELEASE_GIL_R from the upstream opcode table.
// slot 44 (formerly BC_CALL_RELEASE_GIL_FLOAT) freed — see slot 42
// above.
// slot 45 (formerly BC_CALL_RELEASE_GIL_VOID) freed — see slot 42
// above.
// slots 46..=48 (formerly BC_CALL_LOOPINVARIANT_{INT,REF,FLOAT}) freed
// — loop-invariant policy rides on
// `EffectInfo.extraeffect = EF_LOOPINVARIANT` (`effectinfo.py`).
// slot 49 (formerly BC_CALL_LOOPINVARIANT_VOID) freed — see slots
// 46..=48 above.
pub const BC_CALL_ASSEMBLER_INT: u8 = 50;
pub const BC_CALL_ASSEMBLER_REF: u8 = 51;
pub const BC_CALL_ASSEMBLER_FLOAT: u8 = 52;
pub const BC_CALL_ASSEMBLER_VOID: u8 = 53;
// Ref-typed scalar state field: the field lives in the ref register bank
// (registers_r) so getfield_gc/setfield_gc on it heap-cache and the
// optimizer can reason about its identity. `d` = u16 field index, `r` =
// ref register. (slots 54/55 were free between the call_assembler family
// and the int state-field family.)
pub const BC_LOAD_STATE_FIELD_REF: u8 = 54;
pub const BC_STORE_STATE_FIELD_REF: u8 = 55;
pub const BC_LOAD_STATE_FIELD: u8 = 56;
pub const BC_STORE_STATE_FIELD: u8 = 57;
// slots 58-59 (formerly load_state_array / store_state_array) freed:
// a plain `[int]` is ptr then len; cells are `getarrayitem_raw_i`.
pub const BC_LOAD_STATE_FIELD_FLOAT: u8 = 60;
pub const BC_STORE_STATE_FIELD_FLOAT: u8 = 61;
pub const BC_GETFIELD_VABLE_I: u8 = 62;
pub const BC_GETFIELD_VABLE_R: u8 = 63;
pub const BC_GETFIELD_VABLE_F: u8 = 64;
pub const BC_SETFIELD_VABLE_I: u8 = 65;
pub const BC_SETFIELD_VABLE_R: u8 = 66;
pub const BC_SETFIELD_VABLE_F: u8 = 67;
pub const BC_GETARRAYITEM_VABLE_I: u8 = 68;
pub const BC_GETARRAYITEM_VABLE_R: u8 = 69;
pub const BC_GETARRAYITEM_VABLE_F: u8 = 70;
pub const BC_SETARRAYITEM_VABLE_I: u8 = 71;
pub const BC_SETARRAYITEM_VABLE_R: u8 = 72;
pub const BC_SETARRAYITEM_VABLE_F: u8 = 73;
pub const BC_ARRAYLEN_VABLE: u8 = 74;
pub const BC_HINT_FORCE_VIRTUALIZABLE: u8 = 75;
/// RPython bhimpl_ref_return: callee returns a ref value.
pub const BC_REF_RETURN: u8 = 76;
/// blackhole.py bhimpl_raise: raise an exception from a ref register.
pub const BC_RAISE: u8 = 77;
/// blackhole.py bhimpl_reraise: re-raise exception_last_value.
pub const BC_RERAISE: u8 = 78;
// RPython jtransform.py rewrite_op_jit_conditional_call — conditional_call_ir_v
pub const BC_COND_CALL_VOID: u8 = 79;
// RPython jtransform.py rewrite_op_jit_conditional_call_value — conditional_call_value_ir_i / conditional_call_value_ir_r
pub const BC_COND_CALL_VALUE_INT: u8 = 80;
pub const BC_COND_CALL_VALUE_REF: u8 = 81;
// RPython jtransform.py rewrite_op_jit_record_known_result — record_known_result_i_ir_v / record_known_result_r_ir_v
pub const BC_RECORD_KNOWN_RESULT_INT: u8 = 82;
pub const BC_RECORD_KNOWN_RESULT_REF: u8 = 83;
/// pyjitpl.py opimpl_int_guard_value: promote int to constant via GUARD_VALUE.
pub const BC_INT_GUARD_VALUE: u8 = 84;
/// pyjitpl.py opimpl_ref_guard_value: promote ref to constant via GUARD_VALUE.
pub const BC_REF_GUARD_VALUE: u8 = 85;
/// pyjitpl.py opimpl_float_guard_value: promote float to constant via GUARD_VALUE.
pub const BC_FLOAT_GUARD_VALUE: u8 = 86;
/// blackhole.py bhimpl_jit_merge_point: portal merge point marker.
/// `iIRFIRF` form: jdindex byte is a `registers_i` pool slot index
/// (assembler.py emit_const default path when `allow_short=False`
/// or value > 127).
pub const BC_JIT_MERGE_POINT: u8 = 87;
/// `cIRFIRF` form of `bhimpl_jit_merge_point` selected by
/// assembler.py `USE_C_FORM` membership when jdindex fits in
/// `signed i8` (`-128..=127`). The jdindex byte is the raw signed
/// value (assembler.py emit_const short branch +
/// blackhole.py `argcode == 'c'` handler reads `signedord`).
pub const BC_JIT_MERGE_POINT_C: u8 = 131;
pub const BC_LIVE: u8 = 88;
pub const BC_CATCH_EXCEPTION: u8 = 89;
pub const BC_LAST_EXC_VALUE: u8 = 90;
/// RPython blackhole.py:987 `last_exception/>i`.
pub const BC_LAST_EXCEPTION: u8 = 129;
/// RPython blackhole.py:976-985 `goto_if_exception_mismatch/iL`.
pub const BC_GOTO_IF_EXCEPTION_MISMATCH: u8 = 130;
/// blackhole.py bhimpl_rvmprof_code: rvmprof enter/leave marker.
pub const BC_RVMPROF_CODE: u8 = 91;

// RPython jtransform.py `optimize_goto_if_not` fuses
// `v = int_lt(x, y); exitswitch = v` into
// `exitswitch = ('int_lt', x, y)`, emitted by flatten.py:247-250 as
// the jitcode op `goto_if_not_int_lt`. blackhole.py consumes
// the fused form with dedicated bhimpls.
//
// majit currently reserves one `BC_GOTO_IF_NOT_*` per RPython opname
// variant; the 'c' short-const argcode (assembler.py `USE_C_FORM`)
// is not yet supported in the pyre JitCodeBuilder so only the canonical
// `iiL` / `ffL` / `rrL` forms get a BC_* allocation here.
pub const BC_GOTO_IF_NOT_INT_LT: u8 = 92;
pub const BC_GOTO_IF_NOT_INT_LE: u8 = 93;
pub const BC_GOTO_IF_NOT_INT_EQ: u8 = 94;
pub const BC_GOTO_IF_NOT_INT_NE: u8 = 95;
pub const BC_GOTO_IF_NOT_INT_GT: u8 = 96;
pub const BC_GOTO_IF_NOT_INT_GE: u8 = 97;
pub const BC_GOTO_IF_NOT_FLOAT_LT: u8 = 98;
pub const BC_GOTO_IF_NOT_FLOAT_LE: u8 = 99;
pub const BC_GOTO_IF_NOT_FLOAT_EQ: u8 = 100;
pub const BC_GOTO_IF_NOT_FLOAT_NE: u8 = 101;
pub const BC_GOTO_IF_NOT_FLOAT_GT: u8 = 102;
pub const BC_GOTO_IF_NOT_FLOAT_GE: u8 = 103;
pub const BC_GOTO_IF_NOT_PTR_EQ: u8 = 104;
pub const BC_GOTO_IF_NOT_PTR_NE: u8 = 105;
// blackhole.py `bhimpl_goto_if_not_int_is_zero(a, target, pc)`:
// take target iff `a != 0`. jtransform.py `_rewrite_equality`
// folds `int_eq(x, 0)` into `int_is_zero(x)`; flatten.py:247 then
// specialises the bool exitswitch into `goto_if_not_int_is_zero/iL`.
pub const BC_GOTO_IF_NOT_INT_IS_ZERO: u8 = 106;

// blackhole.py bhimpl_int_push / bhimpl_ref_push /
// bhimpl_float_push and matching pops — one-slot scratch for the
// cycle-break path emitted by flatten.py `insert_renamings`.
pub const BC_INT_PUSH: u8 = 107;
pub const BC_REF_PUSH: u8 = 108;
pub const BC_FLOAT_PUSH: u8 = 109;
pub const BC_INT_POP: u8 = 110;
pub const BC_REF_POP: u8 = 111;
pub const BC_FLOAT_POP: u8 = 112;

pub const BC_INT_ADD: u8 = 113;
pub const BC_INT_SUB: u8 = 114;
pub const BC_INT_MUL: u8 = 115;
// 116, 117 — reserved (no opcode for `int_floordiv` / `int_mod`).
// `jtransform.py` rewrites both via `_do_builtin_call` to
// `direct_call(ll_int_py_div)` / `direct_call(ll_int_py_mod)` before
// jitcode emission; the bytecode dispatch path therefore has no
// `BC_INT_FLOORDIV` / `BC_INT_MOD` opcode.  Pyre's residual call at
// `pyre-jit-trace/src/jitcode_dispatch/specialize.rs::walker_emit_int_py_div_or_mod`
// emits `CallI(ll_int_py_div, ...)` / `CallI(ll_int_py_mod, ...)`
// for the same effect.  Slot numbers stay reserved so opcode
// renumbering stays stable across this gap.
pub const BC_INT_AND: u8 = 118;
pub const BC_INT_OR: u8 = 119;
pub const BC_INT_XOR: u8 = 120;
pub const BC_INT_LSHIFT: u8 = 121;
pub const BC_INT_RSHIFT: u8 = 122;
pub const BC_INT_EQ: u8 = 123;
pub const BC_INT_NE: u8 = 124;
pub const BC_INT_LT: u8 = 125;
pub const BC_INT_LE: u8 = 126;
pub const BC_INT_GT: u8 = 127;
pub const BC_INT_GE: u8 = 128;
pub const BC_INT_NEG: u8 = 132;
pub const BC_FLOAT_ADD: u8 = 133;
pub const BC_FLOAT_SUB: u8 = 134;
pub const BC_FLOAT_MUL: u8 = 135;
pub const BC_FLOAT_TRUEDIV: u8 = 136;
pub const BC_FLOAT_NEG: u8 = 139;
pub const BC_FLOAT_ABS: u8 = 140;
pub const BC_INT_INVERT: u8 = 141;
/// `blackhole.py` `bhimpl_int_is_zero(a) -> !a` — the sibling of
/// [`BC_INT_IS_TRUE`], produced by `jtransform.py` `_rewrite_equality`
/// when an `int_eq` has the zero constant as one operand and the result
/// is read rather than fused into `goto_if_not_int_is_zero`.  The
/// handler was already wired in `wire_bhimpl_handlers`, but with no
/// byte here `insn_byte` panicked on the key instead.  Takes the next
/// byte above `BC_RAW_STORE_F` (235).
pub const BC_INT_IS_ZERO: u8 = 236;
// `blackhole.py bhimpl_int_isconstant` / `bhimpl_ref_isconstant` /
// `bhimpl_ref_isvirtual`.  `look_inside_iff` folds `jit.isconstant` /
// `jit.isvirtual` to these ops (`jtransform.py rewrite_op_direct_call`
// `jit.isconstant` / `jit.isvirtual`).
pub const BC_INT_ISCONSTANT: u8 = 239;
pub const BC_REF_ISCONSTANT: u8 = 237;
pub const BC_REF_ISVIRTUAL: u8 = 238;
pub const BC_UINT_RSHIFT: u8 = 142;
pub const BC_UINT_MUL_HIGH: u8 = 143;
pub const BC_UINT_LT: u8 = 144;
pub const BC_UINT_LE: u8 = 145;
pub const BC_UINT_GT: u8 = 146;
pub const BC_UINT_GE: u8 = 147;
// Ref/nullity primitives — RPython `blackhole.py:584-610`
// `bhimpl_{ptr_eq,ptr_ne,ptr_iszero,ptr_nonzero,instance_ptr_eq,instance_ptr_ne}`.
pub const BC_PTR_EQ: u8 = 151;
pub const BC_PTR_NE: u8 = 152;
pub const BC_INSTANCE_PTR_EQ: u8 = 153;
pub const BC_INSTANCE_PTR_NE: u8 = 154;
pub const BC_PTR_ISZERO: u8 = 155;
pub const BC_PTR_NONZERO: u8 = 156;
// Unary ptr nullity exitswitch specialisations — `blackhole.py bhimpl_goto_if_not_ptr_iszero`
// `bhimpl_goto_if_not_ptr_{iszero,nonzero}`.
pub const BC_GOTO_IF_NOT_PTR_ISZERO: u8 = 157;
pub const BC_GOTO_IF_NOT_PTR_NONZERO: u8 = 158;
// canonical residual_call_*_v opcodes — RPython `blackhole.py:1240-1255`
// `bhimpl_residual_call_{r,ir,irf}_v`. Distinct opcodes per argcode shape so
// `setup_insns` (in `blackhole.rs`) keeps its 1:1 opcode→key invariant.
// These slots are reserved in the 159-255 free range from the
// call-family canonical migration.
pub const BC_RESIDUAL_CALL_R_V: u8 = 159;
pub const BC_RESIDUAL_CALL_IR_V: u8 = 160;
pub const BC_RESIDUAL_CALL_IRF_V: u8 = 161;
// canonical residual_call_*_i / *_r / *_f opcodes — RPython
// `blackhole.py:1208-1239 bhimpl_residual_call_{r,ir,irf}_{i,r,f}`.
// One opcode per (arg-shape, return-kind) pair so `setup_insns`
// keeps its 1:1 invariant.
pub const BC_RESIDUAL_CALL_R_I: u8 = 162;
pub const BC_RESIDUAL_CALL_IR_I: u8 = 163;
pub const BC_RESIDUAL_CALL_IRF_I: u8 = 164;
pub const BC_RESIDUAL_CALL_R_R: u8 = 165;
pub const BC_RESIDUAL_CALL_IR_R: u8 = 166;
pub const BC_RESIDUAL_CALL_IRF_R: u8 = 167;
pub const BC_RESIDUAL_CALL_IRF_F: u8 = 168;
// Typed return opcodes — RPython `blackhole.py:841-862`
// `bhimpl_int_return`, `bhimpl_float_return`, `bhimpl_void_return`.
// pyre's portal return is REF (see BC_REF_RETURN) but the insns map
// still needs every upstream return flavour so
// `pyjitpl.py` `setup_insns` fields do not fall back to
// `u8::MAX` sentinels.
pub const BC_INT_RETURN: u8 = 148;
pub const BC_FLOAT_RETURN: u8 = 149;
pub const BC_VOID_RETURN: u8 = 150;

// `getfield_gc_*` / `setfield_gc_*` — RPython `blackhole.py bhimpl_getfield_gc_i`
// `bhimpl_getfield_gc_{i,r,f}` + `bhimpl_setfield_gc_{i,r,f}` —
// descr-keyed instance field load/store on the GC heap.  Pyre emits
// these at build time from the assembler so the bytes are pinned
// canonical to keep the byte → opname map stable across build/runtime.
pub const BC_GETFIELD_GC_I: u8 = 169;
pub const BC_GETFIELD_GC_R: u8 = 170;
pub const BC_GETFIELD_GC_F: u8 = 171;
pub const BC_SETFIELD_GC_I: u8 = 172;
pub const BC_SETFIELD_GC_R: u8 = 173;

// `getarrayitem_gc_*` / `setarrayitem_gc_*` — RPython
// `blackhole.py bhimpl_getarrayitem_gc_i`.  `bhimpl_getarrayitem_gc_{i,r,f}` defines
// the `rid` (ref + int_idx + descr) argcode shape; array indices are
// always int-classified, so the index operand always lands in the `i`
// register bank.
pub const BC_GETARRAYITEM_GC_R_RID: u8 = 174;
// `getfield_raw_i/id>i` — `blackhole.py bhimpl_getfield_raw_i`
// (`@arguments("cpu", "i", "d", returns="i")`).  Int-bank sibling of
// `bhimpl_getfield_gc_i`.  Byte 175 is the gap between the GC array
// load and the store family.
pub const BC_GETFIELD_RAW_I: u8 = 175;
pub const BC_SETARRAYITEM_GC_I: u8 = 176;
pub const BC_SETARRAYITEM_GC_R: u8 = 177;
pub const BC_SETARRAYITEM_GC_F: u8 = 178;
pub const BC_GETARRAYITEM_GC_I: u8 = 197;
pub const BC_GETARRAYITEM_GC_F: u8 = 198;

// `float_*` comparison ops — RPython `blackhole.py bhimpl_float_lt`
// `bhimpl_float_{lt,le,eq,ne,gt,ge}` — float comparisons returning int.
pub const BC_FLOAT_EQ: u8 = 179;
pub const BC_FLOAT_NE: u8 = 180;
pub const BC_FLOAT_LT: u8 = 181;
pub const BC_FLOAT_GT: u8 = 182;

// `cast_*_to_*` conversion ops — RPython `bhimpl_cast_ptr_to_int`
// (`blackhole.py`), `bhimpl_cast_int_to_ptr`
// (`blackhole.py`), `bhimpl_cast_int_to_float`
// (`blackhole.py`).
pub const BC_CAST_INT_TO_FLOAT: u8 = 183;
pub const BC_CAST_INT_TO_PTR: u8 = 184;
pub const BC_CAST_PTR_TO_INT: u8 = 185;

// Float<->longlong bit reinterpret — `bhimpl_convert_float_bytes_to_longlong`
// / `bhimpl_convert_longlong_bytes_to_float` (`blackhole.py`): a bitcast
// of the 64-bit pattern, NOT a value cast (unlike `cast_float_to_int`).
pub const BC_CONVERT_FLOAT_BYTES_TO_LONGLONG: u8 = 229;
pub const BC_CONVERT_LONGLONG_BYTES_TO_FLOAT: u8 = 230;

// Overflow-checked arithmetic fused with its branch — `bhimpl_int_{add,sub,mul}
// _jump_if_ovf` (`blackhole.py:478-497`).  Argcode `Lii>i`: a 2-byte overflow
// label + two int reads + one int write (written only on the no-overflow path).
// The metatracer records these as `IntAddOvf`/`IntSubOvf`/`IntMulOvf` + a
// box-less `GuardNoOverflow`/`GuardOverflow` (`pyjitpl/dispatch.rs`); a guard
// failure resumes at the fused op and the blackhole performs the branch itself.
pub const BC_INT_ADD_JUMP_IF_OVF: u8 = 231;
pub const BC_INT_SUB_JUMP_IF_OVF: u8 = 232;
pub const BC_INT_MUL_JUMP_IF_OVF: u8 = 233;

// pyre-only `arraybase_vable/rdd>i` — the address of item 0 of a
// virtualizable array field, as an integer.  Argcode and operand layout are
// the same `rdd>i` triple as its sibling `arraylen_vable` (BC_ARRAYLEN_VABLE =
// 74), so `read_vable_arraylen` decodes both.
//
// Nothing emits this opcode. It is defined, decoded, walked, and wired into the
// blackhole, but it is not a working feature.
// The only intended emitter is a `#[jit_interp]` macro lowering for
// `state.<varr>.as_mut_ptr()`, which is deliberately held back: with it
// applied, tl's `jit_residual_not_double_executed` runs the ROLL residual 7
// times where the program contains 40, and returns 17 where the interpreter
// returns 0. The remaining resume-position defect must be fixed before adding
// the emitter.
//
// RPython has no counterpart, and the reason is the interesting part: upstream
// never needs the base as a *value* because every `*_vable` op resolves it
// internally from `(fdescr, adescr)` off the live virtualizable.  The op exists
// only to hand that address to a residual call, which is an escape — the callee
// can permute items that are the trace's own SSA values.  Executing it
// therefore also marks the current `TraceCtx` as escaped, and the enclosing
// CALL_MAY_FORCE aborts with ABORT_ESCAPE before the trace reaches a backend.
pub const BC_ARRAYBASE_VABLE: u8 = 234;

// `switch/id` — RPython `blackhole.py` `bhimpl_switch` —
// table-of-cases dispatch keyed by an int register + a descr selecting
// the case table.
pub const BC_SWITCH: u8 = 186;

// `inline_call_{r,ir,irf}_{i,r,v,f}` — RPython `blackhole.py bhimpl_inline_call_r_i`
// `bhimpl_inline_call_*` — inlined helper-jitcode dispatch.  The argcode
// suffix encodes which kind families the helper takes (R/IR/IRF) and
// what kind it returns (i/r/v/f).
pub const BC_INLINE_CALL_R_I: u8 = 187;
pub const BC_INLINE_CALL_R_R: u8 = 188;
pub const BC_INLINE_CALL_R_V: u8 = 189;
pub const BC_INLINE_CALL_IR_I: u8 = 190;
pub const BC_INLINE_CALL_IR_R: u8 = 191;
pub const BC_INLINE_CALL_IR_V: u8 = 192;
pub const BC_INLINE_CALL_IRF_F: u8 = 193;
pub const BC_INLINE_CALL_IRF_R: u8 = 194;
pub const BC_INLINE_CALL_IRF_I: u8 = 202;
pub const BC_INLINE_CALL_IRF_V: u8 = 203;

// Float `<=` / `>=` comparisons — RPython `blackhole.py bhimpl_float_le`
// `bhimpl_float_{le,ge}` (le @726, ge @746).  `BC_FLOAT_LE` /
// `BC_FLOAT_GE` join the existing `BC_FLOAT_{EQ,NE,LT,GT}` quartet at
// 179-182.
pub const BC_FLOAT_LE: u8 = 204;
pub const BC_FLOAT_GE: u8 = 205;

// `setfield_gc_f/rfd` — RPython `blackhole.py bhimpl_setfield_gc_f`
// `bhimpl_setfield_gc_f`.  Float-result sibling of
// `bhimpl_setfield_gc_{i,r}` which already pin `BC_SETFIELD_GC_{I,R}`
// at 172-173; deferred until now because pyre's float-field stores
// only landed alongside the float comparison fix.
pub const BC_SETFIELD_GC_F: u8 = 206;

// `getfield_gc_*_pure/rd>X` and `getarrayitem_gc_*_pure/rid>X` —
// RPython `blackhole.py:1339-1341, 1441-1443` aliases
// `bhimpl_get{field,arrayitem}_gc_{i,r,f}_pure =
// bhimpl_get{field,arrayitem}_gc_{i,r,f}`.  The Python aliases share
// a function body but each `(opname, argcodes)` key still gets its
// own opcode byte through `assembler.py:220
// self.insns.setdefault(key, len(self.insns))`.  The walker dispatch
// at `pyre/pyre-jit-trace/src/jitcode_dispatch.rs` routes the `_pure`
// keys to the same handler as the non-pure form, but the byte must
// be distinct or `pyjitpl.py setup_insns`'s
// `assert opcode_implementations[value] is None` and pyre's
// `jitcode_runtime.rs`'s `overlay_insns` duplicate-byte assert would fire.
pub const BC_GETFIELD_GC_I_PURE: u8 = 207;
pub const BC_GETFIELD_GC_R_PURE: u8 = 208;
pub const BC_GETFIELD_GC_F_PURE: u8 = 209;
pub const BC_GETARRAYITEM_GC_I_PURE: u8 = 210;
pub const BC_GETARRAYITEM_GC_R_PURE: u8 = 211;
pub const BC_GETARRAYITEM_GC_F_PURE: u8 = 212;

// `blackhole.py bhimpl_int_between(a, b, c) -> i` —
// `@arguments("i", "i", "i", returns="i")` gives canonical key
// `int_between/iii>i`.  Pyre emits this from
// `make_ll_isinstance`'s `has_subclasses` body
// (`translator/rtyper/rtyper.rs::build_ll_isinstance_const_nonnull_graph`),
// matching `rclass.py ll_isinstance`'s range-check arm.
pub const BC_INT_BETWEEN: u8 = 213;

// `blackhole.py bhimpl_new_array_clear` —
// `@arguments("cpu", "i", "d", returns="r")` gives canonical key
// `new_array_clear/id>r`.  Pyre emits this from the BUILD_TUPLE
// `popvalues` unroll (`pyre-jit/src/jit/codewriter.rs` — the
// `pyframe.py` fixed-size items-array allocation).
pub const BC_NEW_ARRAY_CLEAR: u8 = 214;

// `c`-argcode forms (`assembler.py emit_const(allow_short=
// True)`): `new_array_clear` and `setarrayitem_gc_r` are both in
// `USE_C_FORM` (`assembler.py`), so a small ConstInt operand
// (-128..127) is written inline as one signed byte instead of a
// constant-pool `i` slot, giving keys `new_array_clear/cd>r` and
// `setarrayitem_gc_r/rcrd`.
pub const BC_SETARRAYITEM_GC_R_C: u8 = 215;
pub const BC_NEW_ARRAY_CLEAR_C: u8 = 216;

// `int_copy` is in `USE_C_FORM` (`assembler.py`): a small ConstInt
// source (-128..127) is written inline as one signed byte instead of a
// constant-pool `i` slot, giving the key `int_copy/c>i`. pyre materialises
// every int/bool constant through `int_copy` (`OpKind::ConstInt`/`ConstBool`
// → `bhimpl_int_copy`), so the short form is the common small-constant path.
// Byte 220 (not 217, which is `BC_NEW`): every `(opname, argcodes)` key
// needs a distinct byte (`assembler.py:220`), and the byte→opname inverse
// must stay 1:1 (`insns_byte_to_opname` panics otherwise).
pub const BC_MOVE_I_C: u8 = 220;

// `int_return` is in `USE_C_FORM` (`assembler.py`): a small const-int
// return source (`FlatOp::IntReturn(Const)`) is written inline as one
// signed byte, giving the key `int_return/c` (sibling of `int_return/i`,
// BC_INT_RETURN = 148).  Byte 221 (not 218, which is `BC_NEW_ARRAY`).
pub const BC_INT_RETURN_C: u8 = 221;

// `setfield_gc` is in `USE_C_FORM` (`assembler.py`): a small ConstInt
// store value (-128..127) is written inline as one signed byte, giving
// the key `setfield_gc_i/rcd` (sibling of `setfield_gc_i/rid`,
// BC_SETFIELD_GC_I = 172).  A wider constant or a register value keeps
// the `i` pool-slot form.  Byte 222 (not 219, which is
// `BC_NEW_WITH_VTABLE`).
pub const BC_SETFIELD_GC_I_C: u8 = 222;

// `setarrayitem_gc_i` is in `USE_C_FORM` (`assembler.py`): a small
// ConstInt store value (-128..127) is written inline as one signed byte,
// giving the const-VALUE key `setarrayitem_gc_i/ricd` (a sibling of the
// const-INDEX `setarrayitem_gc_r/rcrd`, BC_SETARRAYITEM_GC_R_C = 215, and
// of the register-value `setarrayitem_gc_i/riid`, BC_SETARRAYITEM_GC_I =
// 176).  A wider constant or a register value keeps the `i` pool-slot
// form.
pub const BC_SETARRAYITEM_GC_I_C: u8 = 227;

// `raw_store_i/iiid` — `blackhole.py bhimpl_raw_store_i`
// (`@arguments("cpu", "i", "i", "i", "d")`): raw address + byte offset +
// int value + `arraydescrof(rffi.CArray(T))` (`jtransform.py rewrite_op_raw_store
// rewrite_op_raw_store`).  Sibling of the read-side `BC_RAW_LOAD_I` (20);
// takes the next free byte.
pub const BC_RAW_STORE_I: u8 = 228;

// `raw_store_f/iifd` — `blackhole.py bhimpl_raw_store_f`
// (`@arguments("cpu", "i", "i", "f", "d")`): the float-valued sibling of
// `BC_RAW_STORE_I`.  `rewrite_op_raw_store` picks the value's kind, so a
// `double` written through a raw address (the libffi exchange buffer's
// float result among them) needs its own byte.
pub const BC_RAW_STORE_F: u8 = 235;

// pyre-only `abort/>r` — Ref-result variant of `abort/` (BC_ABORT = 13)
// emitted by `Assembler::encode_op`'s default branch when an `OpKind::
// Abort { result_kind: Ref }` reaches the assembler.  Lives in
// `extension_insns()` alongside `abort/` / `abort_permanent/`.
pub const BC_ABORT_RESULT_R: u8 = 195;

// pyre-only `abort/>i` — the Int-result counterpart of `abort/>r`, emitted
// when the unsupported operation owns an int SSA result (a GC interior
// address refused by `rewrite_op_getsubstruct`).
pub const BC_ABORT_RESULT_I: u8 = 249;

// pyre-only `vtable_method_ptr/rd>i` — emitted by `OpKind::
// VtableMethodPtr` (`assembler.rs`'s `encode_op`).  RPython has no counterpart
// because Python dispatch goes through `cpu.bh_call_*` resolved at
// runtime; pyre's Rust port hits this only when a `dyn Trait` method
// pointer must be reified into the bytecode stream (backend-epic
// adaptation, see `blackhole.rs`'s `handler_vtable_method_ptr_bail`, which
// aborts the frame rather than manufacturing a pointer).
pub const BC_VTABLE_METHOD_PTR: u8 = 196;

// `record_quasiimmut_field/rdd` — RPython `blackhole.py bhimpl_record_quasiimmut_field`
// `bhimpl_record_quasiimmut_field`.  Records that a quasi-immutable
// struct field has been read so the optimizer can install a
// `QUASIIMMUT_FIELD` guard before the corresponding `getfield_gc_*`.
// Argcodes: `rdd` (struct ref + fielddescr + mutatefielddescr).
pub const BC_RECORD_QUASIIMMUT_FIELD: u8 = 199;

// `jit_force_quasi_immutable/rd` — RPython `blackhole.py
// bhimpl_jit_force_quasi_immutable`.  Argcodes `rd`: struct ref +
// mutate fielddescr.  Byte 39 is the first unused opcode number
// (`wellknown_bh_insns` ∪ `extension_insns` fill 199-254).
pub const BC_JIT_FORCE_QUASI_IMMUTABLE: u8 = 39;

// `assert_not_none/r` — RPython `pyjitpl.py opimpl_assert_not_none`
// + `blackhole.py bhimpl_assert_not_none`.  Records that `box` is
// known non-null so subsequent nullity-aware sites
// (`_establish_nullity`, KnownClass guards) can skip their own checks.
// Argcodes: `r` (struct ref).
pub const BC_ASSERT_NOT_NONE: u8 = 200;

// `record_exact_class/ri` — RPython `pyjitpl.py:393-410
// opimpl_record_exact_class` + `blackhole.py bhimpl_record_exact_class`.
// Records that `box`'s class is exactly `cls` (vtable ref constant) so
// subsequent class-aware sites can skip the GuardClass.  Argcodes:
// `ri` (struct ref + class pointer as int), matching
// `blackhole.py @arguments("r", "i")`.
pub const BC_RECORD_EXACT_CLASS: u8 = 201;

// `guard_class/r>i` and `guard_class/r>r` — RPython `blackhole.py`
// `bhimpl_guard_class(cpu, struct): return cpu.bh_classof(struct)`, emitted
// by `jtransform.py handle_getfield_typeptr` for every read of the
// object header's class word.  Upstream's result is `Ptr(OBJECT_VTABLE)`;
// pyre's `PyObject.ob_type` is read as a raw word in some graphs and as a
// GC ref in others, and the guard keeps the bank of the read it replaced.
// Distinct from `BC_REF_ISCONSTANT` (237) / `BC_REF_ISVIRTUAL` (238)
// / `BC_INT_ISCONSTANT` (239).  `Assembler.insns` is 1:1.
pub const BC_GUARD_CLASS: u8 = 240;
pub const BC_GUARD_CLASS_R: u8 = 241;

pub const MAX_HOST_CALL_ARITY: usize = 16;

/// Lookup a bytecode opcode by its `opname/argcodes` key.
///
/// RPython `assembler.py:220-222`:
/// ```text
/// key = opname + '/' + ''.join(argcodes)
/// num = self.insns.setdefault(key, len(self.insns))
/// self.code[startposition] = chr(num)
/// ```
///
/// majit currently pre-populates the dict from `wellknown_bh_insns` so
/// numbers match the hardcoded `BC_*` constants consumed by the
/// blackhole dispatch. Once dispatch becomes table-driven the
/// pre-population will drop and numbers will be allocated in emission
/// order exactly like RPython.
///
/// Panics if `key` is not registered — mirrors the `assert 0 <= num <=
/// 0xFF` behaviour RPython relies on at the assembler layer.
pub fn insn_byte(key: &str) -> u8 {
    insn_byte_opt(key).unwrap_or_else(|| panic!("insn_byte: unregistered insns key {key:?}"))
}

/// Non-panicking lookup against the merged
/// `wellknown_bh_insns()` + `extension_insns()` table.  Returns
/// `Some(byte)` for canonical/extension keys (build/runtime-stable
/// `BC_*`) and `None` for translator-only keys.  `Assembler::get_opnum`
/// uses the `None` return to fall through to the
/// `assembler.py setdefault(key, len(self.insns))` dynamic-byte
/// allocator, adjusted for pyre's fixed-byte table by scanning from
/// zero and skipping reserved bytes.  A `None` here is not a
/// fail-loud condition — it just means the key flows through the
/// dynamic allocator instead of a reserved byte.
///
/// Build-time `JitCodeBuilder::write_insn` callers use the panicking
/// [`insn_byte`] variant (every key the BH dispatcher knows about must
/// be registered here); the translator-pipeline assembler consults
/// this `_opt` variant so unregistered translator-only keys can take
/// the dynamic-allocation path without tripping the runtime decoder.
pub fn insn_byte_opt(key: &str) -> Option<u8> {
    use std::sync::OnceLock;
    static TABLE: OnceLock<IndexMap<&'static str, u8>> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
        // Build-time consumers (`JitCodeBuilder::write_insn`) lookup any
        // canonical opname (RPython parity) AND any pyre-only Rust-
        // adaptation opname (`abort/`, `abort_permanent/`). Merging both
        // tables keeps every legal key resolvable from a single entry
        // point while the source of truth remains split — canonical
        // opnames live in `wellknown_bh_insns()`, pyre-only quarantine
        // lives in `extension_insns()`.
        let mut merged = wellknown_bh_insns();
        for (k, v) in extension_insns() {
            assert!(
                !merged.contains_key(k),
                "insn_byte: pyre extension key {k:?} collides with \
                 wellknown_bh_insns; the two tables must be disjoint",
            );
            merged.insert(k, v);
        }
        merged
    });
    table.get(key).copied()
}

/// Highest byte value reserved across `wellknown_bh_insns()` ∪
/// `extension_insns()`.  This is a diagnostic for pyre's fixed
/// `BC_*` range; translator-only opcode allocation must not use this
/// as a start offset because gaps below the high-water remain legal
/// dynamic bytes.
pub fn canonical_byte_high_water() -> u8 {
    use std::sync::OnceLock;
    static HIGH_WATER: OnceLock<u8> = OnceLock::new();
    *HIGH_WATER.get_or_init(|| {
        let mut max = 0u8;
        for (_, v) in wellknown_bh_insns() {
            if v > max {
                max = v;
            }
        }
        for (_, v) in extension_insns() {
            if v > max {
                max = v;
            }
        }
        max
    })
}

/// True when `byte` is pinned by either canonical RPython-mirror keys
/// (`wellknown_bh_insns`) or pyre-only extension keys
/// (`extension_insns`).  Multiple keys may intentionally share a
/// byte because some RPython blackhole handlers are aliases; this helper
/// only answers whether the byte is reserved, not whether the table is
/// one-to-one.
pub fn is_reserved_opcode_byte(byte: u8) -> bool {
    use std::sync::OnceLock;
    static RESERVED: OnceLock<[bool; 256]> = OnceLock::new();
    let reserved = RESERVED.get_or_init(|| {
        let mut reserved = [false; 256];
        for (_, value) in wellknown_bh_insns() {
            reserved[value as usize] = true;
        }
        for (_, value) in extension_insns() {
            reserved[value as usize] = true;
        }
        // Keep the absent-control-opcode sentinel out of every allocator's
        // pool so `opcode == op_live` cannot match a real instruction.
        reserved[BC_ABSENT as usize] = true;
        reserved
    });
    reserved[byte as usize]
}

/// Fixed majit blackhole opcode-name table.
///
/// RPython's `Assembler.insns` is a dense dict grown by
/// `Assembler.write_insn()` in emission order. majit's current runtime
/// `JitCodeBuilder` still emits fixed `BC_*` numbers, so this helper is an
/// adapter table rather than a line-by-line port of `assembler.py`.
/// Downstream consumers use it only for `insns.get('...', -1)`-style opcode
/// cache fields and for wiring handlers against majit's fixed bytecodes.
///
/// The `argcodes` alphabet follows `assembler.py:162-196`:
///   `i` int reg, `r` ref reg, `f` float reg, `c` short-const int,
///   `I/R/F` constant-pool int/ref/float, `L` label, `d` descr,
///   `N` `ListOfKind` (mixed-kind literal list).
pub fn wellknown_bh_insns() -> IndexMap<&'static str, u8> {
    let mut m = IndexMap::new();

    // pyjitpl.py — fields `setup_insns` probes explicitly.
    m.insert("live/", BC_LIVE);
    m.insert("catch_exception/L", BC_CATCH_EXCEPTION);
    m.insert("rvmprof_code/ii", BC_RVMPROF_CODE);
    // pyjitpl.py:2240-2243 typed return accessors:
    //   op_int_return / op_ref_return / op_float_return / op_void_return
    // pyre's portal result type is REF so `ref_return/r` is the only
    // one produced by the current emitter, but the other three must be
    // registered so `setup_insns` does not fall back to `u8::MAX` for
    // them.
    m.insert("int_return/i", BC_INT_RETURN);
    // `int_return/c` — USE_C_FORM short source (`assembler.py`): a small
    // const-int return value inline as one signed byte.
    m.insert("int_return/c", BC_INT_RETURN_C);
    m.insert("ref_return/r", BC_REF_RETURN);
    m.insert("float_return/f", BC_FLOAT_RETURN);
    m.insert("void_return/", BC_VOID_RETURN);

    // `abort/` and `abort_permanent/` are pyre-only Rust adaptations and
    // live in `extension_insns()` — their byte values
    // are still `BC_ABORT` / `BC_ABORT_PERMANENT`, but they are kept out
    // of the canonical-mirror table to keep `wellknown_bh_insns()` a
    // strict subset of RPython's `Assembler.insns`. See the comment on
    // `extension_insns` for the borrow-checker rationale.

    // RPython blackhole.py `bhimpl_unreachable()` raises
    // `AssertionError("unreachable")`. Distinct opcode from
    // `abort_permanent/` so the interpreter fallback path cannot be taken.
    m.insert("unreachable/", BC_UNREACHABLE);

    // The 6 `*_state_*` keys were quarantined into
    // `extension_insns()`. They model the
    // proc-macro-generated JIT-machine-state addressing scheme and have
    // no RPython counterpart — see the doc-comment on
    // `extension_insns` for the proc-macro/runtime-bridge
    // rationale.

    // Virtualizable operations — RPython canonical argcode shapes from
    // blackhole.py bhimpl_getarrayitem_vable_i and :1446-1495.  The legacy helper-side
    // `JitCodeBuilder::vable_*` methods still emit their old compact
    // payloads for the state-field macro path; Pyre's SSA assembler uses
    // the `*_with_base` methods that match these keys byte-for-byte.
    m.insert("getfield_vable_i/rd>i", BC_GETFIELD_VABLE_I);
    m.insert("getfield_vable_r/rd>r", BC_GETFIELD_VABLE_R);
    m.insert("getfield_vable_f/rd>f", BC_GETFIELD_VABLE_F);
    m.insert("setfield_vable_i/rid", BC_SETFIELD_VABLE_I);
    m.insert("setfield_vable_r/rrd", BC_SETFIELD_VABLE_R);
    m.insert("setfield_vable_f/rfd", BC_SETFIELD_VABLE_F);
    m.insert("getarrayitem_vable_i/ridd>i", BC_GETARRAYITEM_VABLE_I);
    m.insert("getarrayitem_vable_r/ridd>r", BC_GETARRAYITEM_VABLE_R);
    m.insert("getarrayitem_vable_f/ridd>f", BC_GETARRAYITEM_VABLE_F);
    m.insert("setarrayitem_vable_i/riidd", BC_SETARRAYITEM_VABLE_I);
    m.insert("setarrayitem_vable_r/rirdd", BC_SETARRAYITEM_VABLE_R);
    m.insert("setarrayitem_vable_f/rifdd", BC_SETARRAYITEM_VABLE_F);
    m.insert("arraylen_vable/rdd>i", BC_ARRAYLEN_VABLE);
    m.insert("hint_force_virtualizable/r", BC_HINT_FORCE_VIRTUALIZABLE);

    // Control flow / structural markers that actually emit.
    // pyjitpl.py `op_goto = insns.get('goto/L', -1)` and
    // blackhole.py `bhimpl_goto(target): return target` — the
    // canonical key is `goto/L`.
    m.insert("goto/L", BC_JUMP);
    // loop_header takes a single int constant operand (the jitdriver index).
    // RPython jtransform.py handle_jit_marker__loop_header emits
    // SpaceOperation('loop_header', [c_index], None); blackhole.py bhimpl_loop_header
    // bhimpl_loop_header(jdindex) is @arguments("i").
    m.insert("loop_header/i", BC_LOOP_HEADER);
    m.insert("raise/r", BC_RAISE);
    m.insert("reraise/", BC_RERAISE);
    // blackhole.py `@arguments("self", returns="i") bhimpl_last_exception`
    // yields canonical key `last_exception/>i`.
    m.insert("last_exception/>i", BC_LAST_EXCEPTION);
    m.insert(
        "goto_if_exception_mismatch/iL",
        BC_GOTO_IF_EXCEPTION_MISMATCH,
    );
    // flatten.py:347 emits `last_exc_value, '->', reg`, so
    // assembler.py grows the canonical key `last_exc_value/>r`.
    m.insert("last_exc_value/>r", BC_LAST_EXC_VALUE);
    // assembler.py:163,181-220 builds the canonical key from the full
    // 7-arg shape (`jdindex + I/R/F + I/R/F`). assembler.py:312 places
    // `jit_merge_point` in `USE_C_FORM`, so the jdindex argcode is `c`
    // when the value fits in a signed byte and `i` (constants-pool
    // slot index) otherwise. Both forms reach the same
    // `bhimpl_jit_merge_point` (blackhole.py) because the
    // `@arguments("i", ...)` decoder dispatches on the runtime argcode
    // (blackhole.py `argtype == 'i'` branch).
    m.insert("jit_merge_point/cIRFIRF", BC_JIT_MERGE_POINT_C);
    m.insert("jit_merge_point/iIRFIRF", BC_JIT_MERGE_POINT);
    // RPython `blackhole.py:1240-1255` `bhimpl_residual_call_{r,ir,irf}_v`.
    // The call-family canonical migration retired the
    // legacy `BC_RESIDUAL_CALL_VOID` (=18) byte layout in favour of the
    // canonical `iRd / iIRd / iIRFd` argcode triple; byte 18 now houses
    // the canonical `goto_if_not/iL` key as [`BC_GOTO_IF_NOT`].
    m.insert("residual_call_r_v/iRd", BC_RESIDUAL_CALL_R_V);
    m.insert("residual_call_ir_v/iIRd", BC_RESIDUAL_CALL_IR_V);
    m.insert("residual_call_irf_v/iIRFd", BC_RESIDUAL_CALL_IRF_V);
    m.insert("residual_call_r_i/iRd>i", BC_RESIDUAL_CALL_R_I);
    m.insert("residual_call_ir_i/iIRd>i", BC_RESIDUAL_CALL_IR_I);
    m.insert("residual_call_irf_i/iIRFd>i", BC_RESIDUAL_CALL_IRF_I);
    m.insert("residual_call_r_r/iRd>r", BC_RESIDUAL_CALL_R_R);
    m.insert("residual_call_ir_r/iIRd>r", BC_RESIDUAL_CALL_IR_R);
    m.insert("residual_call_irf_r/iIRFd>r", BC_RESIDUAL_CALL_IRF_R);
    m.insert("residual_call_irf_f/iIRFd>f", BC_RESIDUAL_CALL_IRF_F);
    // jtransform.py rewrite_op_jit_record_known_result / 1672-1688 conditional/known-result family
    // intentionally omitted. The helper-side `BC_COND_CALL_*` /
    // `BC_RECORD_KNOWN_RESULT_*` adapters encode argc + per-arg kind tags
    // in a flat payload, which is not line-by-line compatible with the
    // canonical `iiIRd` / `riIRd>r` argcode layout. The translator-owned
    // codewriter pipeline emits the real canonical keys when it actually
    // assembles those operations.
    // blackhole.py:1278-1319 inline-call family intentionally omitted.
    // The helper-side `BC_INLINE_CALL` adapter in majit-metainterp uses a
    // typed arg + caller-destination payload that is not line-by-line compatible
    // with canonical `inline_call_*` argcodes. The real RPython-shape
    // `inline_call_*` keys come from the translator/codewriter pipeline
    // when they are actually emitted; pre-registering them here would make
    // `wellknown_bh_insns()` claim a bytecode contract this runtime does
    // not truthfully expose.  The pyre-only nested-bytecode shape is
    // registered separately as `inline_call_nested_ext/P` in
    // `extension_insns()`.

    // jtransform.py optimize_goto_if_not / flatten.py:247 — fused `goto_if_not_<op>_<type>`.  allow-line-citation
    // Argcodes follow assembler.py: two registers + label.
    m.insert("goto_if_not_int_lt/iiL", BC_GOTO_IF_NOT_INT_LT);
    m.insert("goto_if_not_int_le/iiL", BC_GOTO_IF_NOT_INT_LE);
    m.insert("goto_if_not_int_eq/iiL", BC_GOTO_IF_NOT_INT_EQ);
    m.insert("goto_if_not_int_ne/iiL", BC_GOTO_IF_NOT_INT_NE);
    m.insert("goto_if_not_int_gt/iiL", BC_GOTO_IF_NOT_INT_GT);
    m.insert("goto_if_not_int_ge/iiL", BC_GOTO_IF_NOT_INT_GE);
    m.insert("goto_if_not_float_lt/ffL", BC_GOTO_IF_NOT_FLOAT_LT);
    m.insert("goto_if_not_float_le/ffL", BC_GOTO_IF_NOT_FLOAT_LE);
    m.insert("goto_if_not_float_eq/ffL", BC_GOTO_IF_NOT_FLOAT_EQ);
    m.insert("goto_if_not_float_ne/ffL", BC_GOTO_IF_NOT_FLOAT_NE);
    m.insert("goto_if_not_float_gt/ffL", BC_GOTO_IF_NOT_FLOAT_GT);
    m.insert("goto_if_not_float_ge/ffL", BC_GOTO_IF_NOT_FLOAT_GE);
    m.insert("goto_if_not_ptr_eq/rrL", BC_GOTO_IF_NOT_PTR_EQ);
    m.insert("goto_if_not_ptr_ne/rrL", BC_GOTO_IF_NOT_PTR_NE);
    m.insert("goto_if_not_ptr_iszero/rL", BC_GOTO_IF_NOT_PTR_ISZERO);
    m.insert("goto_if_not_ptr_nonzero/rL", BC_GOTO_IF_NOT_PTR_NONZERO);
    m.insert("goto_if_not_int_is_zero/iL", BC_GOTO_IF_NOT_INT_IS_ZERO);

    // flatten.py `insert_renamings` cycle-break push/pop pairs.
    // Argcodes follow assembler.py:162-196 / blackhole.py:661-679:
    // push takes one register source (`i`/`r`/`f`), pop writes one
    // register destination (`>i`/`>r`/`>f`).
    m.insert("int_push/i", BC_INT_PUSH);
    m.insert("ref_push/r", BC_REF_PUSH);
    m.insert("float_push/f", BC_FLOAT_PUSH);
    m.insert("int_pop/>i", BC_INT_POP);
    m.insert("ref_pop/>r", BC_REF_POP);
    m.insert("float_pop/>f", BC_FLOAT_POP);

    m.insert("int_add/ii>i", BC_INT_ADD);
    m.insert("int_sub/ii>i", BC_INT_SUB);
    m.insert("int_mul/ii>i", BC_INT_MUL);
    // blackhole.py:478-497 overflow-checked arithmetic — `Lii>i`: 2-byte label +
    // 2 int reads + 1 int write (only on the no-overflow path).
    m.insert("int_add_jump_if_ovf/Lii>i", BC_INT_ADD_JUMP_IF_OVF);
    m.insert("int_sub_jump_if_ovf/Lii>i", BC_INT_SUB_JUMP_IF_OVF);
    m.insert("int_mul_jump_if_ovf/Lii>i", BC_INT_MUL_JUMP_IF_OVF);
    // `int_floordiv/ii>i` / `int_mod/ii>i` intentionally absent —
    // `jtransform.py` rewrites via `_do_builtin_call`, so the
    // SSA-name → bytecode table never matches these.  See the
    // `BC_INT_AND` constants block above for the parity rationale.
    m.insert("int_and/ii>i", BC_INT_AND);
    m.insert("int_signext/ii>i", BC_INT_SIGNEXT);
    m.insert("int_or/ii>i", BC_INT_OR);
    m.insert("int_xor/ii>i", BC_INT_XOR);
    m.insert("int_lshift/ii>i", BC_INT_LSHIFT);
    m.insert("int_rshift/ii>i", BC_INT_RSHIFT);
    m.insert("int_eq/ii>i", BC_INT_EQ);
    m.insert("int_ne/ii>i", BC_INT_NE);
    m.insert("int_lt/ii>i", BC_INT_LT);
    m.insert("int_le/ii>i", BC_INT_LE);
    m.insert("int_gt/ii>i", BC_INT_GT);
    m.insert("int_ge/ii>i", BC_INT_GE);
    m.insert("int_neg/i>i", BC_INT_NEG);
    m.insert("int_invert/i>i", BC_INT_INVERT);
    m.insert("int_is_true/i>i", BC_INT_IS_TRUE);
    m.insert("int_is_zero/i>i", BC_INT_IS_ZERO);
    m.insert("int_force_ge_zero/i>i", BC_INT_FORCE_GE_ZERO);
    m.insert("uint_rshift/ii>i", BC_UINT_RSHIFT);
    m.insert("uint_mul_high/ii>i", BC_UINT_MUL_HIGH);
    m.insert("uint_lt/ii>i", BC_UINT_LT);
    m.insert("uint_le/ii>i", BC_UINT_LE);
    m.insert("uint_gt/ii>i", BC_UINT_GT);
    m.insert("uint_ge/ii>i", BC_UINT_GE);
    // `blackhole.py bhimpl_int_between(a, b, c) -> i` —
    // `a <= b < c` range check.  Emitted by
    // `make_ll_isinstance`'s `has_subclasses` body
    // (`rclass.py ll_isinstance` range arm).
    m.insert("int_between/iii>i", BC_INT_BETWEEN);
    // Ref/nullity primitives — `blackhole.py:584-610`.
    m.insert("ptr_eq/rr>i", BC_PTR_EQ);
    m.insert("ptr_ne/rr>i", BC_PTR_NE);
    m.insert("instance_ptr_eq/rr>i", BC_INSTANCE_PTR_EQ);
    m.insert("instance_ptr_ne/rr>i", BC_INSTANCE_PTR_NE);
    m.insert("ptr_iszero/r>i", BC_PTR_ISZERO);
    m.insert("ptr_nonzero/r>i", BC_PTR_NONZERO);
    // Per-opname float primitives — `blackhole.py bhimpl_float_add`
    // `bhimpl_float_{add,sub,mul,truediv,neg,abs}`.
    m.insert("float_add/ff>f", BC_FLOAT_ADD);
    m.insert("float_sub/ff>f", BC_FLOAT_SUB);
    m.insert("float_mul/ff>f", BC_FLOAT_MUL);
    m.insert("float_truediv/ff>f", BC_FLOAT_TRUEDIV);
    m.insert("float_neg/f>f", BC_FLOAT_NEG);
    m.insert("float_abs/f>f", BC_FLOAT_ABS);

    // Typed register copy — `blackhole.py:638-646`
    // `bhimpl_{int,ref,float}_copy`. `@arguments("i"|"r"|"f",
    // returns="i"|"r"|"f")` yields canonical keys
    // `{int,ref,float}_copy/X>X`. pyre's `move_{i,r,f}` emitters route
    // through these bytes; flatten.py `insert_renamings` is the
    // main RPython producer of `int_copy` ops (cycle-break renamings),
    // which pyre's super-inst expansion also re-uses.
    m.insert("int_copy/i>i", BC_MOVE_I);
    // `int_copy/c>i` — USE_C_FORM short source (`assembler.py`): a small
    // ConstInt source takes the inline `c` byte instead of a pool `i` slot.
    // pyre materialises every small int/bool constant through `int_copy`
    // (`OpKind::ConstInt`/`ConstBool`), so the short form is the common path
    // and must share the fixed byte with the runtime blackhole dispatch
    // (`blackhole.rs` BC_MOVE_I_C wiring) rather than draw a dynamic byte.
    m.insert("int_copy/c>i", BC_MOVE_I_C);
    m.insert("int_copy/d>i", BC_MOVE_I_D);
    m.insert("ref_copy/r>r", BC_MOVE_R);
    m.insert("float_copy/f>f", BC_MOVE_F);

    // Guard-value promotions — `blackhole.py:648-656`
    // `bhimpl_{int,ref,float}_guard_value`. Body is a no-op on the
    // blackhole side; `pyjitpl.py:1512-1515`
    // `opimpl_{int,ref,float}_guard_value` = `_opimpl_guard_value`
    // emits GUARD_VALUE during tracing to promote the operand.
    m.insert("int_guard_value/i", BC_INT_GUARD_VALUE);
    m.insert("ref_guard_value/r", BC_REF_GUARD_VALUE);
    m.insert("float_guard_value/f", BC_FLOAT_GUARD_VALUE);

    // Truthy-exitswitch branch — `flatten.py:245` emits the canonical
    // `goto_if_not/iL`; `blackhole.py:913
    // bhimpl_goto_if_not_int_is_true = bhimpl_goto_if_not` aliases the
    // handler on the alias key `goto_if_not_int_is_true/iL`.
    // `assembler.py:221 setdefault()` allocates one byte per
    // `(opname, argcodes)`: the canonical key gets [`BC_GOTO_IF_NOT`]
    // and the alias key gets [`BC_GOTO_IF_NOT_INT_IS_TRUE`].  Both
    // bytes route to the same handler at runtime.
    m.insert("goto_if_not/iL", BC_GOTO_IF_NOT);
    m.insert("goto_if_not_int_is_true/iL", BC_GOTO_IF_NOT_INT_IS_TRUE);

    // GC heap field load/store — `blackhole.py bhimpl_getfield_gc_i` `bhimpl_
    // {get,set}field_gc_{i,r,f}`.  Argcodes: `rd` (struct ref + descr)
    // for gets, `r{i,r,f}d` (struct ref + value reg + descr) for sets.
    m.insert("getfield_gc_i/rd>i", BC_GETFIELD_GC_I);
    m.insert("getfield_gc_r/rd>r", BC_GETFIELD_GC_R);
    m.insert("getfield_gc_f/rd>f", BC_GETFIELD_GC_F);
    m.insert("setfield_gc_i/rid", BC_SETFIELD_GC_I);
    // USE_C_FORM short value (`assembler.py`): a small
    // ConstInt store value takes the inline `c` byte in place of the
    // `i` value register.
    m.insert("setfield_gc_i/rcd", BC_SETFIELD_GC_I_C);
    m.insert("setfield_gc_r/rrd", BC_SETFIELD_GC_R);
    m.insert("setfield_gc_f/rfd", BC_SETFIELD_GC_F);
    // RPython `blackhole.py:1441-1443` aliases `bhimpl_getfield_gc_{i,r,f}_pure
    // = bhimpl_getfield_gc_{i,r,f}` — pure-getter shape on quasi-immutable
    // descrs.  Each `(opname, argcodes)` key gets its own byte per
    // `assembler.py setdefault(key, len(self.insns))`; `pyjitpl.py
    // setup_insns` and pyre's `jitcode_runtime.rs` `overlay_insns` both assert no
    // duplicate bytes.  Walker dispatch routes `_pure` to the same handler.
    m.insert("getfield_gc_i_pure/rd>i", BC_GETFIELD_GC_I_PURE);
    m.insert("getfield_gc_r_pure/rd>r", BC_GETFIELD_GC_R_PURE);
    m.insert("getfield_gc_f_pure/rd>f", BC_GETFIELD_GC_F_PURE);
    // `blackhole.py bhimpl_getfield_raw_i`: raw struct field, int base.
    m.insert("getfield_raw_i/id>i", BC_GETFIELD_RAW_I);
    m.insert("getfield_raw_f/id>f", BC_GETFIELD_RAW_F);
    // `blackhole.py bhimpl_setfield_raw_{i,f}`: int-bank store, not in
    // USE_C_FORM (`assembler.py`).
    m.insert("setfield_raw_i/iid", BC_SETFIELD_RAW_I);
    m.insert("setfield_raw_f/ifd", BC_SETFIELD_RAW_F);

    // GC heap array element load/store — `blackhole.py bhimpl_getarrayitem_gc_i`.
    // `bhimpl_getarrayitem_gc_{i,r,f}` register only the canonical `rid`
    // (ref + int_idx + descr) shape; array indices are always int-classified,
    // and the assembler enforces the int-index argcode at emit time
    // (`assembler.rs` `OpKind::ArrayRead`/`ArrayWrite`).
    // Setters are 4-byte payloads `ri{i,r,f}d` per
    // `bhimpl_setarrayitem_gc_{i,r,f}(cpu, array, index, newvalue,
    // arraydescr)` (`blackhole.py bhimpl_setarrayitem_gc_i`): array (Ref), index (Int),
    // newvalue ({Int,Ref,Float}), arraydescr.
    m.insert("getarrayitem_gc_r/rid>r", BC_GETARRAYITEM_GC_R_RID);
    m.insert("setarrayitem_gc_i/riid", BC_SETARRAYITEM_GC_I);
    m.insert("setarrayitem_gc_r/rird", BC_SETARRAYITEM_GC_R);
    m.insert("setarrayitem_gc_f/rifd", BC_SETARRAYITEM_GC_F);
    // `c`-argcode forms (USE_C_FORM, `assembler.py/312`): small
    // ConstInt index/length written inline as one signed byte.
    // `new_array_clear/id>r` itself is registered with the other
    // allocation opcodes (`new` / `new_with_vtable` / `new_array`) below.
    m.insert("setarrayitem_gc_r/rcrd", BC_SETARRAYITEM_GC_R_C);
    // const-VALUE form: a small ConstInt store value inline as one signed
    // byte (`setarrayitem_gc_i` ∈ USE_C_FORM, `assembler.py`), keyed
    // off the int value bank like `setfield_gc_i/rcd`.
    m.insert("setarrayitem_gc_i/ricd", BC_SETARRAYITEM_GC_I_C);
    m.insert("new_array_clear/cd>r", BC_NEW_ARRAY_CLEAR_C);
    m.insert("getarrayitem_gc_i/rid>i", BC_GETARRAYITEM_GC_I);
    m.insert("getarrayitem_gc_f/rid>f", BC_GETARRAYITEM_GC_F);
    // Raw-array element load/store — `blackhole.py`
    // `bhimpl_getarrayitem_raw_{i,f}` / `bhimpl_setarrayitem_raw_{i,f}`.
    // The array address lives in the int bank (`iid>i` / `iiid`); there
    // is no `_r` form (`jtransform.py` `rewrite_op_getarrayitem` /
    // `rewrite_op_setarrayitem` refuse a GC pointer in a raw array).
    m.insert("getarrayitem_raw_i/iid>i", BC_GETARRAYITEM_RAW_I);
    m.insert("getarrayitem_raw_f/iid>f", BC_GETARRAYITEM_RAW_F);
    m.insert("setarrayitem_raw_i/iiid", BC_SETARRAYITEM_RAW_I);
    m.insert("setarrayitem_raw_f/iifd", BC_SETARRAYITEM_RAW_F);
    // RPython `blackhole.py:1339-1341` aliases
    // `bhimpl_getarrayitem_gc_{i,r,f}_pure = bhimpl_getarrayitem_gc_{i,r,f}`
    // — pure-getter shape on immutable-array descrs.  Each
    // `(opname, argcodes)` key gets its own byte per `assembler.py:220
    // setdefault(key, len(self.insns))`; walker dispatch routes
    // `_pure` to the same handler as the non-pure form.
    m.insert("getarrayitem_gc_i_pure/rid>i", BC_GETARRAYITEM_GC_I_PURE);
    m.insert("getarrayitem_gc_r_pure/rid>r", BC_GETARRAYITEM_GC_R_PURE);
    m.insert("getarrayitem_gc_f_pure/rid>f", BC_GETARRAYITEM_GC_F_PURE);

    // Array length — `blackhole.py` `bhimpl_arraylen_gc`
    // (`@arguments("cpu", "r", "d", returns="i")`).
    // `bhimpl_arraylen_vable` (`blackhole.py`,
    // `@arguments("cpu", "r", "d", "d", returns="i")`) is pinned with
    // its vable siblings above at [`BC_ARRAYLEN_VABLE`] = 74.
    m.insert("arraylen_gc/rd>i", BC_ARRAYLEN_GC);

    // String length / item — `blackhole.py` `bhimpl_strlen` /
    // `bhimpl_strgetitem`.  The Skip-spine rewrite of
    // `__strlen` / `__string_byte_getitem` emits these; `pyre_cpu`
    // follows `W_UnicodeObject.value`.
    m.insert("strlen/r>i", BC_STRLEN);
    m.insert("strgetitem/ri>i", BC_STRGETITEM);
    m.insert("strgetitem/rc>i", BC_STRGETITEM_C);

    // Interior-field load — `blackhole.py bhimpl_getinteriorfield_gc_i`
    // `bhimpl_getinteriorfield_gc_{i,r,f}`
    // (`@arguments("cpu", "r", "i", "d", returns="i"|"r"|"f")`).
    m.insert("getinteriorfield_gc_i/rid>i", BC_GETINTERIORFIELD_GC_I);
    m.insert("getinteriorfield_gc_r/rid>r", BC_GETINTERIORFIELD_GC_R);
    m.insert("getinteriorfield_gc_f/rid>f", BC_GETINTERIORFIELD_GC_F);
    // `blackhole.py` `bhimpl_setinteriorfield_gc_{i,r,f}`
    // (`@arguments("cpu", "r", "i", "i"|"r"|"f", "d")`).
    m.insert("setinteriorfield_gc_i/riid", BC_SETINTERIORFIELD_GC_I);
    m.insert("setinteriorfield_gc_r/rird", BC_SETINTERIORFIELD_GC_R);
    m.insert("setinteriorfield_gc_f/rifd", BC_SETINTERIORFIELD_GC_F);

    // List item load — `blackhole.py bhimpl_getlistitem_gc_i`
    // `bhimpl_getlistitem_gc_{i,r,f}`
    // (`@arguments("cpu", "r", "i", "d", "d", returns="i"|"r"|"f")`).
    m.insert("getlistitem_gc_i/ridd>i", BC_GETLISTITEM_GC_I);
    m.insert("getlistitem_gc_r/ridd>r", BC_GETLISTITEM_GC_R);
    m.insert("getlistitem_gc_f/ridd>f", BC_GETLISTITEM_GC_F);

    // Quasi-immutable record — `blackhole.py bhimpl_record_quasiimmut_field`
    // `bhimpl_record_quasiimmut_field`.  Argcodes `rdd`: struct ref +
    // field descr + mutate descr (no result).
    m.insert("record_quasiimmut_field/rdd", BC_RECORD_QUASIIMMUT_FIELD);
    m.insert("jit_force_quasi_immutable/rd", BC_JIT_FORCE_QUASI_IMMUTABLE);

    // Heapcache hint markers — `pyjitpl.py:385-410`.
    // `assert_not_none/r` records that `box` is non-null;
    // `record_exact_class/ri` records that `box`'s class equals the
    // `cls` constant (vtable pointer carried through the int bank).
    // `record_exact_class` is a no-op at
    // blackhole resume; `assert_not_none` checks the concrete ref just
    // like `blackhole.py bhimpl_assert_not_none`.  The trace-time dispatcher routes them through
    // `TraceCtx::trace_assert_not_none` /
    // `TraceCtx::trace_record_exact_class`, which gate on
    // `heap_cache.is_nullity_known` / `is_class_known` and bump
    // `HEAPCACHED_OPS` on cache hit per `pyjitpl.py:387-388 / 396-397`.
    // The `/ri` suffix matches `blackhole.py:616`; tracing reboxes a
    // constant int class pointer as a ConstPtr before recording.
    m.insert("assert_not_none/r", BC_ASSERT_NOT_NONE);
    m.insert("int_isconstant/i>i", BC_INT_ISCONSTANT);
    m.insert("ref_isconstant/r>i", BC_REF_ISCONSTANT);
    m.insert("ref_isvirtual/r>i", BC_REF_ISVIRTUAL);
    m.insert("int_isvirtual/i>i", BC_INT_ISVIRTUAL);
    m.insert("record_exact_class/ri", BC_RECORD_EXACT_CLASS);
    // `guard_class` — `blackhole.py bhimpl_guard_class`; see `BC_GUARD_CLASS`.
    m.insert("guard_class/r>i", BC_GUARD_CLASS);
    m.insert("guard_class/r>r", BC_GUARD_CLASS_R);

    // Float comparisons — `blackhole.py bhimpl_float_lt`
    // `bhimpl_float_{lt,le,eq,ne,gt,ge}` — float pair → int (0/1).
    m.insert("float_eq/ff>i", BC_FLOAT_EQ);
    m.insert("float_ne/ff>i", BC_FLOAT_NE);
    m.insert("float_lt/ff>i", BC_FLOAT_LT);
    m.insert("float_le/ff>i", BC_FLOAT_LE);
    m.insert("float_gt/ff>i", BC_FLOAT_GT);
    m.insert("float_ge/ff>i", BC_FLOAT_GE);

    // Cross-kind casts — `bhimpl_cast_ptr_to_int` (`blackhole.py`),
    // `bhimpl_cast_int_to_ptr` (`blackhole.py`),
    // `bhimpl_cast_int_to_float` (`blackhole.py`),
    // `bhimpl_cast_float_to_int` (`blackhole.py`).
    m.insert("cast_int_to_float/i>f", BC_CAST_INT_TO_FLOAT);
    m.insert("cast_int_to_ptr/i>r", BC_CAST_INT_TO_PTR);
    m.insert("cast_ptr_to_int/r>i", BC_CAST_PTR_TO_INT);
    m.insert("cast_float_to_int/f>i", BC_CAST_FLOAT_TO_INT);
    m.insert(
        "convert_float_bytes_to_longlong/f>i",
        BC_CONVERT_FLOAT_BYTES_TO_LONGLONG,
    );
    m.insert(
        "convert_longlong_bytes_to_float/i>f",
        BC_CONVERT_LONGLONG_BYTES_TO_FLOAT,
    );

    // Switch dispatch — `blackhole.py` `bhimpl_switch`.
    // Argcodes `id`: int discriminator + descr selecting the case table.
    m.insert("switch/id", BC_SWITCH);

    // Inlined helper-jitcode dispatch — `blackhole.py bhimpl_inline_call_r_i`
    // `bhimpl_inline_call_*`.  Argcodes encode (a) which kind families
    // the helper takes after the descr — `R` (ref-only), `IR`
    // (int+ref), `IRF` (int+ref+float) — and (b) the helper's return
    // kind — `>i`/`>r`/`>f` for typed returns, no suffix for void.  The
    // descr `d` slot resolves to the helper's `JitCode` at runtime.
    m.insert("inline_call_r_i/dR>i", BC_INLINE_CALL_R_I);
    m.insert("inline_call_r_r/dR>r", BC_INLINE_CALL_R_R);
    m.insert("inline_call_r_v/dR", BC_INLINE_CALL_R_V);
    m.insert("inline_call_ir_i/dIR>i", BC_INLINE_CALL_IR_I);
    m.insert("inline_call_ir_r/dIR>r", BC_INLINE_CALL_IR_R);
    m.insert("inline_call_ir_v/dIR", BC_INLINE_CALL_IR_V);
    m.insert("inline_call_irf_i/dIRF>i", BC_INLINE_CALL_IRF_I);
    m.insert("inline_call_irf_r/dIRF>r", BC_INLINE_CALL_IRF_R);
    m.insert("inline_call_irf_f/dIRF>f", BC_INLINE_CALL_IRF_F);
    m.insert("inline_call_irf_v/dIRF", BC_INLINE_CALL_IRF_V);

    // List construction — `blackhole.py bhimpl_newlist`
    // `bhimpl_newlist{,_clear,_hint}`
    // (`@arguments("cpu", "i", "d", "d", "d", "d", returns="r")`):
    // length + 4 descrs (struct + length + items + array) → ref result.
    m.insert("newlist/idddd>r", BC_NEWLIST);
    m.insert("newlist_clear/idddd>r", BC_NEWLIST_CLEAR);
    m.insert("newlist_hint/idddd>r", BC_NEWLIST_HINT);

    // Single-object / array allocation — `blackhole.py:1301-1327`
    // `bhimpl_new` (`@arguments("cpu", "d", returns="r")`),
    // `bhimpl_new_with_vtable` (same shape; descr carries the class),
    // `bhimpl_new_array{,_clear}` (`@arguments("cpu", "i", "d",
    // returns="r")`): length + arraydescr → ref result.
    m.insert("new/d>r", BC_NEW);
    m.insert("new_with_vtable/d>r", BC_NEW_WITH_VTABLE);
    m.insert("new_array/id>r", BC_NEW_ARRAY);
    m.insert("new_array_clear/id>r", BC_NEW_ARRAY_CLEAR);

    // Raw memory load — `blackhole.py bhimpl_raw_load_i`
    // `bhimpl_raw_load_{i,f}` (`@arguments("cpu", "i", "i", "d",
    // returns="i"|"f")`): addr + offset + arraydescr.
    m.insert("raw_load_i/iid>i", BC_RAW_LOAD_I);
    m.insert("raw_load_f/iid>f", BC_RAW_LOAD_F);

    // Raw memory store — `blackhole.py:1504-1509`
    // `bhimpl_raw_store_{i,f}` (`@arguments("cpu", "i", "i", "i"|"f",
    // "d")`): raw address + byte offset + value + arraydescr.  The
    // handlers are wired in `blackhole.rs` (`handler_raw_store_i` /
    // `handler_raw_store_f`).
    m.insert("raw_store_i/iiid", BC_RAW_STORE_I);
    m.insert("raw_store_f/iifd", BC_RAW_STORE_F);

    // GC indexed load — `blackhole.py bhimpl_gc_load_indexed_i`
    // `bhimpl_gc_load_indexed_{i,f}` (`@arguments("cpu", "r", "i", "i",
    // "i", "i", returns="i"|"f")`): base ref + index + scale + base_ofs +
    // bytes.
    m.insert("gc_load_indexed_i/riiii>i", BC_GC_LOAD_INDEXED_I);
    m.insert("gc_load_indexed_f/riiii>f", BC_GC_LOAD_INDEXED_F);

    // Negative-index normalisation — `blackhole.py bhimpl_check_neg_index`
    // `bhimpl_check_neg_index` (`@arguments("cpu", "r", "i", "d",
    // returns="i")`): array ref + index + arraydescr → wrapped index.
    m.insert("check_neg_index/rid>i", BC_CHECK_NEG_INDEX);

    // Conditional residual call — `blackhole.py bhimpl_conditional_call_ir_v`
    // `bhimpl_conditional_call_ir_v` (`@arguments("cpu", "i", "i", "I",
    // "R", "d")`) is fired iff the int `condition` is non-zero;
    // `bhimpl_conditional_call_value_ir_{i,r}` returns the `value`
    // operand when the helper isn't fired so the result is always
    // typed.
    m.insert("conditional_call_ir_v/iiIRd", BC_CONDITIONAL_CALL_IR_V);
    m.insert(
        "conditional_call_value_ir_i/iiIRd>i",
        BC_CONDITIONAL_CALL_VALUE_IR_I,
    );
    m.insert(
        "conditional_call_value_ir_r/riIRd>r",
        BC_CONDITIONAL_CALL_VALUE_IR_R,
    );

    // Record-known-result — `blackhole.py bhimpl_record_known_result_i_ir_v`
    // `bhimpl_record_known_result_{i,r}_ir_v`
    // (`@arguments("cpu", "i"|"r", "i", "I", "R", "d")`): tells the
    // tracer that the helper's result equals the pre-supplied operand,
    // so the call is elided on the trace.
    m.insert(
        "record_known_result_i_ir_v/iiIRd",
        BC_RECORD_KNOWN_RESULT_I_IR_V,
    );
    m.insert(
        "record_known_result_r_ir_v/riIRd",
        BC_RECORD_KNOWN_RESULT_R_IR_V,
    );

    m
}

/// Pyre-only opcodes that have NO RPython counterpart and arise from
/// Rust language adaptations permitted by `CLAUDE.md` ("Rust language
/// adaptations are permitted ONLY when unavoidable AND minimal" / "the
/// proc-macro/runtime bridge").
///
/// Two adaptation sources land here:
///
/// 1. **Borrow-checker tracing-abort signal** (`abort/`,
///    `abort_permanent/`).  RPython's tracing dispatch (`pyjitpl.py`)
///    bails out via Python exceptions — `SwitchToBlackhole`,
///    `ContinueRunningNormally`, `AssertionError("unreachable")` — and
///    the GC/exception machinery rebuilds the dispatch state on the
///    bailout side. Pyre cannot unwind through the trace loop because
///    the recording state (`PyreSym`, `TraceCtx`, per-kind register
///    banks, symbolic stack) is held by `&mut` references that must
///    remain sound on the unwind path; even `panic::catch_unwind` cannot
///    safely resume because the borrow checker sees the references as
///    still-live. Pyre's `JitCodeMachine` therefore emits explicit
///    `abort/` / `abort_permanent/` opcodes that the dispatch loop
///    converts into `DispatchError::AbortTrace` /
///    `DispatchError::AbortPermanentTrace` `Result` returns.
///
/// 2. **Proc-macro JIT-machine state addressing** (the 4 `*_state_field*`
///    keys).  Pyre uses Rust proc-macros (`#[jit_interp]`,
///    `#[jit_driver]`, `state_fields = { ... }`) to generate the entire
///    JIT machine from a single function definition. RPython's
///    metaprogramming is runtime / annotator-driven and has no
///    proc-macro counterpart. The generated machine carries a
///    `JitCodeSym` whose scalar state fields use flat slot indices (`d` argcode)
///    without an explicit virtualizable-pointer register. The canonical
///    `setfield_vable_*` shape requires an `r` vable pointer plus a `d`
///    `FieldDescr`, so it cannot address these proc-macro-owned slots. Keeping
///    the scalar state opcodes in `pyre_local_bh_insns()` isolates that adaptation
///    while `wellknown_bh_insns()` remains canonical. A plain `[int]` array is
///    ptr then len and uses `getarrayitem_raw_i`.
///
/// All extension keys retain their fixed `BC_*` byte values in the same
/// number-space as the canonical opcodes; only the catalogue is split
/// from `wellknown_bh_insns()`.  `insn_byte` merges this table with
/// `wellknown_bh_insns()` at OnceLock init time, so build-time
/// `write_insn(...)` callers resolve transparently. Runtime dispatch
/// reads `BC_*` constants directly and is unaffected.
pub fn extension_insns() -> IndexMap<&'static str, u8> {
    let mut m = IndexMap::new();
    // Borrow-checker abort signals.
    m.insert("abort/", BC_ABORT);
    m.insert("abort_permanent/", BC_ABORT_PERMANENT);
    // Proc-macro JIT-machine state addressing — emit sites are in
    // `majit-macros/src/jit_interp/jitcode_lower.rs`. Argcodes:
    //   `d` = state-slot index (u16); `i` = int register (u16).
    m.insert("load_state_field_ref/dr", BC_LOAD_STATE_FIELD_REF);
    m.insert("store_state_field_ref/dr", BC_STORE_STATE_FIELD_REF);
    m.insert("load_state_field/di", BC_LOAD_STATE_FIELD);
    m.insert("store_state_field/di", BC_STORE_STATE_FIELD);
    m.insert("load_state_field_float/df", BC_LOAD_STATE_FIELD_FLOAT);
    m.insert("store_state_field_float/df", BC_STORE_STATE_FIELD_FLOAT);
    // TODO: pyre nested-bytecode `inline_call`.
    //
    // RPython's canonical `inline_call_*` keys (`/dIRF>i`, `/dIR>r`, …)
    // dispatch through a real C-ABI `fnaddr` stored on `BhDescr::JitCode`.
    // Pyre does not compile inlined helpers into separate native
    // functions — guard-failure resume must re-interpret the helper as
    // nested bytecode.  Byte 17 (`BC_INLINE_CALL`) is reused for this
    // pyre-only handler (`handler_inline_call_nested_ext`).  The `P`
    // pseudo-argcode is opaque from the canonical RPython argcodes
    // alphabet: payload is `sub_idx u16 + num_args u16 + num_args ×
    // (kind u8, src u16, dst u16) + 3 × (return slot u16; u16::MAX = None)`.
    // Generic walkers must consult `decode_op_at` (which knows `P`) for
    // length, not the canonical argcodes table.
    m.insert("inline_call_nested_ext/P", BC_INLINE_CALL);
    // Byte reservations for `BC_CALL_ASSEMBLER_{INT,REF,FLOAT,VOID}`
    // (50-53). Tracing `opimpl_call_assembler_*` still indexes `opimpl`
    // on these constants; `is_reserved_opcode_byte` keeps dynamic
    // allocators off them. Jitcode no longer emits `call_assembler_*`;
    // recursive portal calls emit `BC_RECURSIVE_CALL_*`. `blackhole.py`
    // has no `bhimpl_call_assembler`; the production builder leaves
    // these bytes unwired.
    m.insert("call_assembler_int_ext/P", BC_CALL_ASSEMBLER_INT);
    m.insert("call_assembler_ref_ext/P", BC_CALL_ASSEMBLER_REF);
    m.insert("call_assembler_float_ext/P", BC_CALL_ASSEMBLER_FLOAT);
    m.insert("call_assembler_void_ext/P", BC_CALL_ASSEMBLER_VOID);
    // Byte reservations for `BC_COND_CALL_{VOID,VALUE_INT,VALUE_REF}`
    // and `BC_RECORD_KNOWN_RESULT_{INT,REF}` (79-83). Tracing
    // `opimpl_cond_call_void` still indexes `opimpl` on these constants;
    // `is_reserved_opcode_byte` keeps dynamic allocators off them.
    // Jitcode emits the canonical `conditional_call_ir_v` /
    // `conditional_call_value_ir_{i,r}` / `record_known_result_{i,r}_ir_v`
    // keys. `blackhole.py` `bhimpl_conditional_call_ir_v` /
    // `bhimpl_conditional_call_value_ir_{i,r}` / pass-bodied
    // `bhimpl_record_known_result_{i,r}_ir_v` handle those; the
    // production builder leaves these leftover bytes unwired.
    m.insert("cond_call_void_ext/P", BC_COND_CALL_VOID);
    m.insert("cond_call_value_int_ext/P", BC_COND_CALL_VALUE_INT);
    m.insert("cond_call_value_ref_ext/P", BC_COND_CALL_VALUE_REF);
    m.insert("record_known_result_int_ext/P", BC_RECORD_KNOWN_RESULT_INT);
    m.insert("record_known_result_ref_ext/P", BC_RECORD_KNOWN_RESULT_REF);
    // pyre-only `abort/>r` — Ref-result variant of `abort/`.  Emitted by
    // `Assembler::encode_op`'s default branch when an `OpKind::Abort` with
    // `result_kind: Ref` reaches the assembler.  The blackhole-side
    // handler `handler_abort_result_marker_r` (in `blackhole.rs`) aborts
    // exactly like `abort/` (`BC_ABORT = 13`); the destination byte exists
    // only because the unsupported operation still owns an SSA result.
    m.insert("abort/>r", BC_ABORT_RESULT_R);
    // pyre-only `abort/>i` — Int-result variant, handled by
    // `handler_abort_result_marker_i` exactly like `abort/`.
    m.insert("abort/>i", BC_ABORT_RESULT_I);
    // pyre-only `vtable_method_ptr/rd>i` — emitted by
    // `OpKind::VtableMethodPtr` (`assembler.rs`'s `encode_op`).  RPython's dispatch
    // resolves dyn-method addresses through `cpu.bh_call_*` at runtime
    // and never needs to reify a method pointer into the bytecode stream;
    // pyre's Rust port hits this only on `dyn Trait` calls, where the
    // backend epic must look up the vtable slot itself.
    m.insert("vtable_method_ptr/rd>i", BC_VTABLE_METHOD_PTR);
    // pyre-only `arraybase_vable/rdd>i` — address of item 0 of a
    // virtualizable array field.  Registered here rather than in
    // `wellknown_bh_insns` beside `arraylen_vable` because RPython has no
    // such op: upstream resolves the base inside each `*_vable` op and never
    // materialises it as a value.  See [`BC_ARRAYBASE_VABLE`] for why
    // producing it is by definition an escape.
    m.insert("arraybase_vable/rdd>i", BC_ARRAYBASE_VABLE);
    // Inline u16 immediate for `setfield_vable_i`. See `BC_SETFIELD_VABLE_I_IMM`.
    m.insert("setfield_vable_i_imm/rdd", BC_SETFIELD_VABLE_I_IMM);
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every distinct `(opname, argcodes)` key must own a UNIQUE opcode
    /// byte, and every `BC_*` constant must own a UNIQUE value.
    /// RPython `assembler.py` `Assembler.write_insn` `setdefault(key,
    /// len(self.insns))` assigns a fresh int per key, so the byte→opname
    /// inverse is 1:1 by construction.  pyre hand-assigns the `BC_*`
    /// constants, so this test is the structural replacement for that
    /// guarantee: both the walker (`jitcode_dispatch`) and the fixed-byte
    /// `JitCodeMachine` (`pyjitpl::dispatch`) route by raw byte, and a byte
    /// shared by two keys silently mis-dispatches one of them (a `match`
    /// arm shadows the later const, `names[byte] = key` is last-wins).  The
    /// USE_C_FORM bytes once collided here (`int_copy/c>i`↔`new/d>r` @217,
    /// `int_return/c`↔`new_array/...` @218, `setfield_gc_i/rcd`↔
    /// `new_with_vtable/d>r` @219); `int_isvirtual/i>i` later collided
    /// with `setfield_raw_f/ifd` at 250, then with `int_force_ge_zero/i>i`
    /// at 253.
    #[test]
    fn opcode_bytes_are_one_to_one() {
        let src = include_str!("insns.rs");
        let mut const_owner: [Option<&str>; 256] = [None; 256];
        let mut n_consts = 0usize;
        for line in src.lines() {
            let line = line.trim();
            let Some(rest) = line.strip_prefix("pub const ") else {
                continue;
            };
            let Some((name, after_name)) = rest.split_once(':') else {
                continue;
            };
            if !name.starts_with("BC_") {
                continue;
            }
            let Some((_, rhs)) = after_name.split_once('=') else {
                panic!("BC_* constant {name} has no '='");
            };
            let rhs = rhs.trim().trim_end_matches(';').trim();
            let byte = if rhs == "u8::MAX" {
                u8::MAX
            } else {
                rhs.parse::<u8>()
                    .unwrap_or_else(|e| panic!("BC_* constant {name} value {rhs:?}: {e}"))
            };
            if let Some(prev) = const_owner[byte as usize] {
                panic!(
                    "opcode byte {byte} is assigned to both {prev} and {name}; \
                     assembler.py setdefault(key, len(self.insns)) never reuses a byte"
                );
            }
            const_owner[byte as usize] = Some(name);
            n_consts += 1;
        }
        assert!(
            n_consts > 200,
            "parsed only {n_consts} BC_* constants from insns.rs; parser likely broken"
        );

        let mut byte_to_key: [Option<&str>; 256] = [None; 256];
        for table in [wellknown_bh_insns(), extension_insns()] {
            for (key, byte) in table {
                if let Some(prev) = byte_to_key[byte as usize] {
                    assert_eq!(
                        prev, key,
                        "opcode byte {byte} maps to two distinct keys \
                         {prev:?} and {key:?}; every key needs its own byte \
                         (assembler.py setdefault(key, len(self.insns)))",
                    );
                }
                byte_to_key[byte as usize] = Some(key);
            }
        }
    }
}

#[cfg(test)]
mod recursive_call_byte_tests {
    use super::*;

    /// The reserved `BC_RECURSIVE_CALL_*` bytes (223-226) must be mutually
    /// distinct, sit above the prior high-water (`BC_NEW_WITH_VTABLE` = 219),
    /// and not collide with any byte already registered in `wellknown_bh_insns`
    /// or `extension_insns`.  Guards against the #199-class byte-collision
    /// (BC_INT_BETWEEN vs BC_NEW) at the point the bytes are reserved, before
    /// the recursive-call lowering/handlers are wired.
    #[test]
    fn recursive_call_bytes_reserved_without_collision() {
        let reserved = [
            BC_RECURSIVE_CALL_INT,
            BC_RECURSIVE_CALL_REF,
            BC_RECURSIVE_CALL_FLOAT,
            BC_RECURSIVE_CALL_VOID,
        ];

        let mut sorted: Vec<u8> = reserved.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            reserved.len(),
            "recursive_call bytes must be mutually distinct"
        );

        for b in reserved {
            assert!(
                b > BC_NEW_WITH_VTABLE,
                "recursive_call byte {b} must be above the prior high-water {}",
                BC_NEW_WITH_VTABLE
            );
        }

        let registered: Vec<u8> = wellknown_bh_insns()
            .values()
            .chain(extension_insns().values())
            .copied()
            .collect();
        for b in reserved {
            assert!(
                !registered.contains(&b),
                "recursive_call byte {b} collides with an already-registered insn"
            );
        }
    }

    /// `blackhole.py:72-74` reads the cached control opcodes as
    /// `insns.get(key, -1)` and compares them against `ord(code[position])`,
    /// so an absent key can never match a real instruction.  `BC_ABSENT`
    /// carries that guarantee into `u8` by staying out of every allocator's
    /// pool; if some opname ever claimed it, `handle_exception_in_frame`
    /// would read that instruction as `live/` or `catch_exception/L`.
    #[test]
    fn absent_control_opcode_sentinel_is_never_a_real_instruction() {
        assert!(
            is_reserved_opcode_byte(BC_ABSENT),
            "BC_ABSENT must be reserved so no dynamic allocator hands it out"
        );

        let registered: Vec<u8> = wellknown_bh_insns()
            .values()
            .chain(extension_insns().values())
            .copied()
            .collect();
        assert!(
            !registered.contains(&BC_ABSENT),
            "BC_ABSENT must not name an instruction"
        );
    }
}
