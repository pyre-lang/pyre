//! Pins the build-time descr pool's "one ArrayDescr per ARRAY" property.
//!
//! `cpu.arraydescrof(ARRAY)` is one descr per ARRAY lltype
//! (`descr.py get_array_descr` / `gccache._cache_array[ARRAY_OR_STRUCT]`).
//! Two ops that name the same ARRAY but land on different pool entries make
//! the heap cache treat them as different arrays: a pre-store read stays live
//! across the store, and a wrapper's length/item reads miss the descr the
//! walker builds that array with. The frozen pool and the jitcodes that name
//! it are the authority; this file reads them. It lives under `tests/` so it
//! is not in the pyre-jit LLBC read set — a `src/` edit would report
//! `LLBC STALE` for a check that does not change translated bodies.

use majit_ir::Type;
use majit_translate::codewriter::jtransform::{LIST_FLOAT_ITEMS_ARRAY, LIST_INT_ITEMS_ARRAY};
use majit_translate::front::mir::OBJECT_REF_GCARRAY_TYPE_ID;
use majit_translate::jitcode::BhDescr;
use pyre_jit_trace::jitcode_runtime::{
    DecodedOp, all_jitcodes, decoded_ops, descr_ref_at, descr_table,
};
use pyre_jit_trace::state::pyobject_gcarray_descr;

/// The prepass links `pyre-module` and records its `#[pyre_class]` statics
/// in `PYRE_CLASS_DESCRIPTORS`. This integration-test binary keeps that
/// slice only when it names the crate, the same way
/// `random_core_residuals_use_registered_genrand32_address` does.
fn link_module_pytypes() {
    pyre_module::register();
}

/// Pool indices of every `d`/`j` operand of `op`.
///
/// The walk matches `decode_op_at`'s argcode sizes (`i|c|r|f` one byte, `L`
/// three, `d|j` two little-endian, `I|R|F` length-prefixed, `>` plus its result
/// char one byte) but records only descr indices. An unknown char — `P` is
/// the live case — stops rather than guessing a pyre-only payload: a descr
/// after that point is not one this pin can name, and inventing a skip would
/// silently drop a later `d`.
fn descr_operands(code: &[u8], op: &DecodedOp) -> Vec<usize> {
    let mut out = Vec::new();
    let mut cursor = op.pc + 1;
    let mut chars = op.argcodes.chars();
    while let Some(c) = chars.next() {
        match c {
            'i' | 'c' | 'r' | 'f' => cursor += 1,
            'L' => cursor += majit_jitcode::jitcode::LABEL_OPERAND_SIZE,
            'd' | 'j' => {
                let Some(&lo) = code.get(cursor) else { break };
                let Some(&hi) = code.get(cursor + 1) else {
                    break;
                };
                out.push(u16::from_le_bytes([lo, hi]) as usize);
                cursor += 2;
            }
            'I' | 'R' | 'F' => {
                let Some(&n) = code.get(cursor) else { break };
                cursor += 1 + n as usize;
            }
            '>' => {
                if chars.next().is_none() {
                    break;
                }
                cursor += 1;
            }
            _ => break,
        }
    }
    out
}

/// A `&[*mut PyObject]` / `&[*const PyObject]` is the length-prefixed
/// `GcArray(Ptr(PyObject))` every other access to that block names
/// (`array_projection_metadata` remaps both spellings to
/// `OBJECT_REF_GCARRAY_TYPE_ID`). A pool entry that still carries the
/// headerless spelling is a second ARRAY for the same block: slice reads
/// and list/tuple stores no longer share a descr, and a cached element
/// survives the store.
#[test]
fn object_pointer_slices_have_no_headerless_descr() {
    let table = descr_table();
    for i in 0..table.len() {
        let Some(BhDescr::Array { array_type_id, .. }) = table.get(i) else {
            continue;
        };
        assert!(
            array_type_id.as_deref() != Some("[*mut PyObject]")
                && array_type_id.as_deref() != Some("[*const PyObject]"),
            "pool ArrayDescr {i} still carries the headerless object-pointer \
             slice identity {array_type_id:?}; that spelling is the thin \
             pointer, not the length-prefixed GcArray the slice is",
        );
    }
}

/// `arraylen_gc` is `rewrite_op_getarraysize` → `cpu.arraydescrof(ARRAY)`.
/// A pool entry with `len_offset is None` is the `nolength=True` shape;
/// blackhole `bh_arraylen_gc` then panics (`llmodel.py`). The
/// assembler must not emit that opcode against such a descr.
#[test]
fn arraylen_gc_descrs_carry_lendescr() {
    link_module_pytypes();
    let table = descr_table();
    for jc in all_jitcodes() {
        for op in decoded_ops(&jc.code) {
            if op.key != "arraylen_gc/rd>i" {
                continue;
            }
            for idx in descr_operands(&jc.code, &op) {
                match table.get(idx) {
                    Some(BhDescr::Array {
                        len_offset,
                        array_type_id,
                        ..
                    }) => {
                        assert!(
                            len_offset.is_some(),
                            "{} arraylen_gc pool {idx} identity {array_type_id:?} \
                             has no lendescr; bh_arraylen_gc panics on that descr",
                            jc.name,
                        );
                    }
                    other => panic!(
                        "{} arraylen_gc descr operand {idx} is {other:?}, \
                         expected Array with a length header",
                        jc.name,
                    ),
                }
            }
        }
    }
}

/// `get_array_descr` keys `_cache_array` on the ARRAY identity. A
/// length-prefixed int or float block (`len_offset == Some(0)`) with no
/// `array_type_id` is minted identity-less, so two sites that describe the
/// same `GcArray<i64>` / `GcArray<f64>` do not join and the heap cache
/// treats them as different arrays. The message names the pool index
/// because the frozen table is positional: a silent `is_some()` miss
/// would not say which mint lost its name.
#[test]
fn length_prefixed_int_and_float_gc_arrays_are_named() {
    let table = descr_table();
    for i in 0..table.len() {
        let Some(BhDescr::Array {
            len_offset,
            item_type,
            array_type_id,
            ..
        }) = table.get(i)
        else {
            continue;
        };
        if *len_offset != Some(0) {
            continue;
        }
        if !matches!(*item_type, Type::Int | Type::Float) {
            continue;
        }
        assert!(
            array_type_id.is_some(),
            "pool ArrayDescr {i} is a length-prefixed {item_type:?} gc array \
             with no array_type_id; identity-less mints of the same layout \
             do not share a `_cache_array` slot",
        );
    }
}

const LIST_INNER_BODIES: [&str; 4] = [
    "w_list_getitem_inner",
    "w_list_setitem_inner",
    "w_list_append_inner",
    "w_list_pop_end_inner",
];

/// The charon list-inner bodies are the graphs the walker descends for
/// getitem/setitem/append/pop. Their `getarrayitem_gc_{i,f}` /
/// `setarrayitem_gc_{i,f}` ops must name the same ARRAY the runtime
/// `int_gcarray_descr` / `float_gcarray_descr` already published
/// (`LIST_INT_ITEMS_ARRAY` / `LIST_FLOAT_ITEMS_ARRAY`), or a build-time
/// store and a walker-side read are two heap-cache keys and the pre-store
/// read stays live. Finding each of the four jitcodes is required: a
/// rename that dropped a body would otherwise pass by matching nothing.
#[test]
fn list_inner_bodies_use_one_descr_per_item_kind() {
    link_module_pytypes();
    let table = descr_table();
    let mut found = [false; LIST_INNER_BODIES.len()];
    for jc in all_jitcodes() {
        let Some(slot) = LIST_INNER_BODIES.iter().position(|&n| n == jc.name) else {
            continue;
        };
        found[slot] = true;
        for op in decoded_ops(&jc.code) {
            let expected = if op.key.starts_with("getarrayitem_gc_i")
                || op.key.starts_with("setarrayitem_gc_i")
            {
                LIST_INT_ITEMS_ARRAY
            } else if op.key.starts_with("getarrayitem_gc_f")
                || op.key.starts_with("setarrayitem_gc_f")
            {
                LIST_FLOAT_ITEMS_ARRAY
            } else {
                continue;
            };
            for idx in descr_operands(&jc.code, &op) {
                match table.get(idx) {
                    Some(BhDescr::Array { array_type_id, .. }) => {
                        assert_eq!(
                            array_type_id.as_deref(),
                            Some(expected),
                            "{} {} descr operand {idx} has identity \
                             {array_type_id:?}, expected {expected}",
                            jc.name,
                            op.key,
                        );
                    }
                    other => panic!(
                        "{} {} descr operand {idx} is {other:?}, expected \
                         Array with identity {expected}",
                        jc.name, op.key,
                    ),
                }
            }
        }
    }
    for (slot, name) in LIST_INNER_BODIES.iter().enumerate() {
        assert!(
            found[slot],
            "jitcode {name} was not found in all_jitcodes(); without it this \
             pin matches no get/setarrayitem and would pass vacuously",
        );
    }
}

/// Helper-innermost resume of `list.pop()` (`pickle_terminal_raise_resume`
/// `load_stop` → `self.stack.pop()`) blackhole-interprets this body. The
/// first switch's Bytes (5) and Ascii (6) arms must not land on `abort`:
/// fused `bytes_items.pop()` / `ascii_items.pop()` residualized as
/// `abort/>r` and the blackhole then panics (`canonical blackhole aborted`).
#[test]
fn w_list_pop_end_inner_bytes_and_ascii_arms_do_not_abort() {
    link_module_pytypes();
    let jc = all_jitcodes()
        .iter()
        .find(|jc| jc.name == "w_list_pop_end_inner")
        .expect("w_list_pop_end_inner in all_jitcodes");
    let table = descr_table();
    let mut switch_dict = None;
    for op in decoded_ops(&jc.code) {
        if !op.opname.starts_with("switch") {
            continue;
        }
        for idx in descr_operands(&jc.code, &op) {
            if let Some(BhDescr::Switch { dict, .. }) = table.get(idx) {
                switch_dict = Some(dict.clone());
                break;
            }
        }
        if switch_dict.is_some() {
            break;
        }
    }
    let dict = switch_dict.expect("w_list_pop_end_inner has a strategy switch");
    let ops: Vec<_> = decoded_ops(&jc.code).collect();
    for key in [5_i64, 6] {
        let target = *dict
            .get(&key)
            .unwrap_or_else(|| panic!("strategy switch missing key {key}: {dict:?}"));
        let first_non_live = ops
            .iter()
            .find(|op| op.pc >= target && op.opname != "live")
            .unwrap_or_else(|| panic!("no op at/after pc {target} for key {key}"));
        assert!(
            !first_non_live.opname.starts_with("abort"),
            "w_list_pop_end_inner strategy {key} target {target} lands on {} \
             (pickle BH abort path)",
            first_non_live.key
        );
    }
}

/// `__majit_wrap_random` reads its `&[PyObjectRef]` argument with
/// `arraylen_gc` / `getarrayitem_gc_r`. Those ops name
/// `OBJECT_REF_GCARRAY_TYPE_ID` in the frozen pool; rehydrating that
/// entry must yield the runtime descr the walker builds the same array
/// with (`pyobject_gcarray_descr`), not a second mint of the same
/// identity. A mismatched `index()` is a heap-cache miss on every
/// wrapper argument read. Both op kinds must appear: a wrapper that
/// kept the length read and dropped the item read (or the reverse)
/// would otherwise pass on a single match.
#[test]
fn builtin_wrapper_args_reads_bridge_to_the_runtime_object_descr() {
    link_module_pytypes();
    let table = descr_table();
    let expected = pyobject_gcarray_descr().index();
    let mut saw_wrapper = false;
    let mut saw_arraylen = false;
    let mut saw_getitem = false;
    for jc in all_jitcodes() {
        if jc.name != "__majit_wrap_random" {
            continue;
        }
        saw_wrapper = true;
        for op in decoded_ops(&jc.code) {
            let is_arraylen = op.key == "arraylen_gc/rd>i";
            let is_getitem = op.key == "getarrayitem_gc_r/rid>r";
            if !is_arraylen && !is_getitem {
                continue;
            }
            for idx in descr_operands(&jc.code, &op) {
                let Some(BhDescr::Array { array_type_id, .. }) = table.get(idx) else {
                    continue;
                };
                if array_type_id.as_deref() != Some(OBJECT_REF_GCARRAY_TYPE_ID) {
                    continue;
                }
                assert_eq!(
                    descr_ref_at(idx).unwrap().index(),
                    expected,
                    "{} {} pool {idx} identity is {OBJECT_REF_GCARRAY_TYPE_ID} \
                     but rehydrates to a different runtime descr than \
                     pyobject_gcarray_descr()",
                    jc.name,
                    op.key,
                );
                if is_arraylen {
                    saw_arraylen = true;
                }
                if is_getitem {
                    saw_getitem = true;
                }
            }
        }
    }
    assert!(
        saw_wrapper,
        "jitcode __majit_wrap_random was not found in all_jitcodes()",
    );
    assert!(
        saw_arraylen,
        "__majit_wrap_random has no arraylen_gc/rd>i whose pool identity is \
         {OBJECT_REF_GCARRAY_TYPE_ID}",
    );
    assert!(
        saw_getitem,
        "__majit_wrap_random has no getarrayitem_gc_r/rid>r whose pool \
         identity is {OBJECT_REF_GCARRAY_TYPE_ID}",
    );
}

/// `W_BytesIO.read_slice` `p + count` used to be `jit_ll_strconcat`
/// because Int FieldRead dests stayed Unknown→`'r'`. The shipped
/// jitcode must keep int add / listslice `rii`, not strconcat.
#[test]
fn read_slice_jitcode_has_no_strconcat() {
    use majit_jitcode::codewriter::jitcode::ConstIRelocKind;
    link_module_pytypes();
    let jc = all_jitcodes()
        .iter()
        .find(|jc| {
            matches!(
                &jc.fnaddr_reloc,
                Some(ConstIRelocKind::FnAddr { path, .. })
                    if path.contains("interp_bytesio") && path.contains("read_slice")
            )
        })
        .expect("interp_bytesio read_slice jitcode");
    let ops: Vec<_> = decoded_ops(&jc.code)
        .map(|op| format!("{}@{}", op.key, op.pc))
        .collect();
    eprintln!(
        "read_slice name={} code_len={} nops={}\n{ops:#?}",
        jc.name,
        jc.code.len(),
        ops.len()
    );
    assert!(
        !ops.iter().any(|s| s.contains("strconcat")),
        "read_slice must not residual ll_strconcat; ops={ops:?}"
    );
}

/// Every GC ArrayDescr the producer minted carries a non-zero ARRAY identity
/// (`effectinfo.py compute_bitstrings` has no sentinel). Virtualizable
/// frame arrays keep `type_id` 0 by design (`vable_arraydescrof`) so a
/// `GUARD_GC_TYPE` is not false for the other allocator arm. A raw
/// `rffi.CArray` descr (`raw_carray_descrof`) is not GC-managed and
/// keeps `type_id` 0: there is no collector type identity for a
/// headerless native array.
#[test]
fn array_descrs_carry_nonzero_identity() {
    let table = descr_table();
    for i in 0..table.len() {
        let Some(BhDescr::Array {
            type_id,
            array_type_id,
            ei_index,
            is_gc_managed,
            ..
        }) = table.get(i)
        else {
            continue;
        };
        if !*is_gc_managed {
            continue;
        }
        if *ei_index == u32::MAX && array_type_id.is_none() {
            continue;
        }
        assert_ne!(
            *type_id, 0,
            "pool ArrayDescr {i} has ARRAY identity 0 \
             array_type_id={array_type_id:?} ei_index={ei_index}"
        );
    }
}
