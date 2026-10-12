//! Every `unroll_safe` in the shipped interpreter LLBC is one that was
//! reviewed.
//!
//! `unroll_safe` is not a loop annotation.  `codewriter/policy.rs`
//! `look_inside_graph` cancels `contains_loop` for a hinted graph, so the
//! attribute changes *what the walker descends into*: the hinted graph and
//! its whole callee closure enter the candidate set.  The tree's history
//! with that is why this file exists — the same three attributes were added,
//! reverted for a SIGBUS traced to an sret ABI mismatch in a callee the hint
//! newly reached, and only re-landed once that callee was published
//! correctly.
//!
//! So an addition here is a descent-scope change that needs its own
//! evidence, and this test makes adding one without saying so fail.  It is
//! deliberately a *subset* check rather than an equality check: a developer's
//! `build/llbc` is routinely older than the source (a `pyre-interpreter`
//! edit is invisible until re-extraction), and a stale artefact must not
//! produce a false red.  A stale artefact can only under-report, which
//! passes; a new hint can only over-report, which fails.

use majit_charon_reader::Llbc;
use majit_translate::front::llbc_hints::harvest_hints_from_llbcs;

const INTERPRETER_LLBC: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../build/llbc/pyre-interpreter.ullbc"
);

/// The leaf name of every function reviewed as `unroll_safe`, with the
/// upstream decorator it mirrors.
///
/// Matched on the leaf rather than the full path because the harvester
/// spells a method with its impl block (`pyframe::<Impl>::fast2locals`) and
/// a free function without one (`builtins::leading_non_null_count`); the
/// leaves are unambiguous across the interpreter.
const REVIEWED_UNROLL_SAFE: &[(&str, &str)] = &[
    // `abstractinst.py` carries `@jit.unroll_safe` on all three.
    (
        "isinstance",
        "abstractinst.py p_recursive_isinstance_w caller",
    ),
    ("issubclass", "abstractinst.py abstract_issubclass_w caller"),
    (
        "p_abstract_issubclass_w",
        "abstractinst.py _abstract_issubclass_w",
    ),
    // `pyframe.py` `fast2locals`.
    ("fast2locals", "pyframe.py fast2locals"),
    // `pyframe.py` `popvalues` is `@jit.unroll_safe`; the CALL arm pops its
    // `nargs` explicit arguments through it before `call_valuestack`, and the
    // count is fixed by the call site's bytecode.
    ("call_explicit_args", "pyframe.py popvalues"),
    // The shared-opcode twin of the same `popvalues`: BUILD_* arms pop an
    // operand count fixed by the opcode's oparg.
    ("pop_n", "pyframe.py popvalues"),
    // `descroperation.py get_and_call_function(space, w_descr, w_obj, *args_w)`:
    // `*args_w` has a length fixed per call site, so the copies into the
    // `funccall` / `Arguments` argument list unroll.
    (
        "get_and_call_function",
        "descroperation.py get_and_call_function",
    ),
    // `baseobjspace.py call_function(self, w_func, *args_w)`: the same
    // per-call-site `*args_w`, copied by `list(args_w)`.
    ("call_function_impl_result", "baseobjspace.py call_function"),
    // `argument.py _match_signature` is `@jit.unroll_safe`: the positional
    // fill of a user function, and the argument roots of the builtin
    // `BuiltinCode.funcrun` path that reaches it.
    ("fill_user_function_args", "argument.py _match_signature"),
    (
        "call_builtin_code_positional",
        "argument.py _match_signature",
    ),
    // `pyopcode.py BUILD_SET` is `@jit.unroll_safe`: the per-item `add` loop
    // runs the opcode's constant item count.
    ("builtin_set_add_items_impl", "pyopcode.py BUILD_SET"),
    // `pyframe.py popvalues` / `peekvalues` are `@jit.unroll_safe`: the count
    // is an opcode argument.
    ("popvalues", "pyframe.py popvalues"),
    ("peekvalues", "pyframe.py peekvalues"),
    // `pyopcode.py UNPACK_SEQUENCE` pushes through `pyframe.py
    // pushrevvalues`, `@jit.unroll_safe`.
    ("opcode_unpack_sequence", "pyframe.py pushrevvalues"),
    // The frame builders `function.py _flat_pycall` fills: the argument copy
    // and cell setup of `pyframe.py initialize_frame_scopes`, both
    // `@jit.unroll_safe`, over a code object's constant counts.
    (
        "try_new_for_call_with_closure_and_globals_obj",
        "function.py _flat_pycall",
    ),
    (
        "finish_for_call_with_globals_obj",
        "pyframe.py initialize_frame_scopes",
    ),
    // `typeobject.py` `lookup_starting_at`, the MRO-suffix walk `super`
    // attribute lookup runs.  Without the hint `look_inside_graph` rejects the
    // graph for its loop and the whole lookup is one opaque residual.
    //
    // This is the descent-scope history in the file header repeating.  The hint
    // let the walk reach `wtf8_key_is_utf8`, published through
    // `jit_fnaddr.rs`'s ABI-unsound hatch because `&Wtf8` is two machine words
    // against the residual ABI's one, and executing it during the sub-walk
    // faulted.  Unlike the sret case, the callee cannot be published correctly
    // — the residual ABI has no fat-pointer argument at all — so the descent
    // declines at it instead (`is_abi_unsound_argument_residual`).
    //
    // Evidence for the descent-scope change: with that decline in place
    // `zero_arg_super_attr` matches its committed dynasm `.jitstats` baseline
    // on every counter, and the corpus gates unchanged against its baselines
    // with `fbw_rolled_back_with_effects` zero — a `JITSTATS_BADNESS_FIELDS`
    // member in `check.py`, gated corpus-wide, so a rise anywhere is a red.
    (
        "super_getattribute_wtf8",
        "typeobject.py lookup_starting_at",
    ),
    // No upstream counterpart by name; the loop is a bounded scan of a
    // fixed-size argument slice.
    ("leading_non_null_count", "flat builtin-keyword ABI scan"),
    // The positional-prefix loop of that scan lives here. `_match_signature`
    // is `@jit.unroll_safe`; the hint belongs on the graph that owns the loop.
    (
        "builtin_positional_count",
        "argument.py _match_signature positional prefix",
    ),
    // `argument.py` carries `@jit.unroll_safe` on `_match_signature`, the
    // keyword-binding loop this one mirrors; both are bounded by a signature
    // fixed at the callee rather than by the call's arguments.
    ("bind_builtin_kwargs", "argument.py _match_signature"),
    // `executioncontext.py` marks both `f_back`-chain walks `@jit.unroll_safe`.
    //
    // Evidence for the descent-scope change: neither graph is reachable from
    // the portal BFS today, so admitting them and their callee closures moved
    // no counter.  The whole corpus gates unchanged against its committed
    // `.jitstats` baselines on all three backends, and
    // `fbw_rolled_back_with_effects` stayed zero — it is a
    // `JITSTATS_BADNESS_FIELDS` member in `check.py`, so it is gated
    // corpus-wide rather than per fixture and a rise anywhere would have been
    // a red rather than a statistic.
    (
        "gettopframe_nohidden",
        "executioncontext.py gettopframe_nohidden",
    ),
    (
        "getnextframe_nohidden",
        "executioncontext.py getnextframe_nohidden",
    ),
    // No upstream counterpart: this materializes a 3.14 `FrameLocalsProxy`.
    // Its slot loop scans the same fixed-size locals-plus array `fast2locals`
    // does and takes the hint for the same reason — without it the function is
    // one residual call, and the `f_locals` read behind it forces the
    // virtualizable for that call's whole length.
    //
    // NOT the same evidence as the two above, and this line used to claim it
    // was: the body is REACHED.  `frame_locals_proxy_snapshot` holds a jitcode
    // in the metadata artefact, where `gettopframe_nohidden` and
    // `getnextframe_nohidden` do not.  The abort population and the baselines
    // were measured unchanged; unreachability is not why.
    (
        "frame_locals_proxy_snapshot",
        "fast2locals' scan of the same locals-plus array",
    ),
    // No upstream counterpart either: this is the mapping read behind a
    // `fr.f_locals["x"]`, and its scan is bounded by the same
    // `locals_plus_names` array — varnames plus cellvars plus freevars — that
    // `fast2locals` walks.  It cannot share `fast_local_index`, because a
    // running comprehension's iteration variable is readable through the proxy
    // even though assigning to that name goes to the extras dict.  What the
    // body does per candidate is not what the hint is about: `hash_w_strict`
    // and `eq_w` can run application code, but they are calls, not the loop
    // bound.
    //
    // Same class of evidence as `frame_locals_proxy_snapshot`, and for the
    // same reason it is NOT the `*_nohidden` argument: the body is REACHED.
    // `locals_plus_value` holds a jitcode in the generated metadata artefact,
    // so admitting it and its callee closure is a real descent-scope change
    // and unreachability proves nothing here.
    //
    // Measured in PR #1599's CI, on the tree that carries this attribute:
    // `pyre/check.py` passed in four jobs across all three runner OSes —
    // dynasm and cranelift on ubuntu-24.04, dynasm on windows-latest, and
    // dynasm+cranelift on macos-latest — with every committed per-fixture
    // jitstats baseline unchanged.  That covers `fbw_rolled_back_with_effects`
    // corpus-wide rather than per fixture, because it is a
    // `JITSTATS_BADNESS_FIELDS` member in `check.py`, so a rise anywhere would
    // have been a red rather than a statistic.
    (
        "locals_plus_value",
        "fast2locals' scan of the same locals-plus array, read through the proxy",
    ),
    // The undeduplicated `keys` / `values` / `items` scan behind the same
    // proxy, and the count `__len__` answers without materializing anything.
    // Both are bounded by the very `locals_plus_names` array `fast2locals`
    // walks.  The `f_extra_locals` half is bounded by a dict length instead:
    // `entry_count` reads that length, and the collectors below append that
    // half through `dont_look_inside` helpers (`append_extra_locals`,
    // `append_extra_locals_items`) that stay residual calls.
    //
    // `entry_count` is named for what it counts rather than `length` because
    // this table matches by leaf and `length` is already the leaf of
    // `bufferview::BufferView::length`.
    //
    // Evidence for the descent-scope change: this is the `*_nohidden`
    // argument, not the `frame_locals_proxy_snapshot` one.  Neither graph
    // holds a jitcode in the generated metadata artefact -- `jit_metadata.json`
    // lists no `pin_entries`, and its one `length` jitcode resolves through
    // `symbolic_fnaddr_paths` to `bufferview::BufferView::length` -- so
    // admitting them and their callee closures reaches nothing the portal BFS
    // already walks.  Measured on the tree carrying both attributes:
    // `pyre/check.py` ran the full corpus on dynasm and cranelift, every
    // committed per-fixture jitstats baseline held except the seven this
    // Windows workstation also moves at `origin/main` with identical deltas,
    // and `fbw_rolled_back_with_effects` stayed zero -- it is a
    // `JITSTATS_BADNESS_FIELDS` member in `check.py`, so it is gated
    // corpus-wide rather than per fixture and a rise anywhere would have been
    // a red rather than a statistic.
    (
        "pin_entries",
        "fast2locals' scan of the same locals-plus array, undeduplicated",
    ),
    (
        "entry_count",
        "fast2locals' scan of the same locals-plus array, counted only",
    ),
    // `collect_entries` (behind `keys` / `values`) and `items`.
    // `collect_entries` no longer walks `pin_entries`: the slot loops live
    // in `append_bound_entries`, the same positional `0..n` shape as
    // `fast2locals`. `items` still collects the pairs `pin_entries` pinned.
    // The `f_extra_locals` half is not in these loops; it is appended by
    // the residual `append_extra_locals{,_items}`.
    //
    // `list(FrameLocalsProxy)` and `tuple(FrameLocalsProxy)` enter
    // `append_bound_entries` through `extend_bound_entries` /
    // `extend_from_frame_locals_proxy` and through `bound_part_list` /
    // `tuple_from_frame_locals_proxy`. Each stays hinted so a loop inlined
    // into that graph does not make `contains_loop` decline `list.__init__`
    // / `tuple.__new__`.
    (
        "collect_entries",
        "fast2locals' scan collected as the proxy keys or values list",
    ),
    (
        "append_bound_entries",
        "pyframe.py fast2locals positional slot walk, appending proxy entries",
    ),
    (
        "extend_bound_entries",
        "pyframe.py fast2locals positional slot walk from list.extend",
    ),
    (
        "extend_from_frame_locals_proxy",
        "pyframe.py fast2locals positional slot walk from list.extend",
    ),
    (
        "bound_part_list",
        "pyframe.py fast2locals positional slot walk from tuple.__new__",
    ),
    (
        "tuple_from_frame_locals_proxy",
        "pyframe.py fast2locals positional slot walk from tuple.__new__",
    ),
    (
        "items",
        "fast2locals' scan collected as the proxy items list",
    ),
    // `pyopcode.py` `dispatch_bytecode` is `@jit.unroll_safe`; its
    // EXTENDED_ARG loop is split here into the interpreter decoder and the
    // two scalar projections consumed by `eval_loop_jit`.  The projections
    // exist because the generated JIT's residual ABI cannot publish the
    // decoder's Result<(usize, Instruction, OpArg)> as one value.  Each loop
    // is bounded by the bytecode format's at-most-three EXTENDED_ARG units.
    //
    // Evidence for all three descent-scope changes: the complete cranelift
    // `pyre/check.py` corpus passed in PR #1580's CI with every committed
    // per-fixture jitstats baseline unchanged, including the corpus-wide
    // `fbw_rolled_back_with_effects == 0` gate.  The public decoder has no
    // jitcode in the generated metadata; both scalar projections do, so the
    // evidence covers the reached pair as well as the unreachable graph.
    (
        "decode_instruction_forward",
        "pyopcode.py dispatch_bytecode EXTENDED_ARG loop",
    ),
    (
        "decode_instruction_forward_pc",
        "pyopcode.py dispatch_bytecode EXTENDED_ARG loop (PC projection)",
    ),
    (
        "decode_instruction_forward_packed",
        "pyopcode.py dispatch_bytecode EXTENDED_ARG loop (opcode/oparg projection)",
    ),
    // `ctypefunc.py W_CTypeFunc._call` carries `@jit.unroll_safe`, and the two
    // halves it is split into here both take it: the body that converts the
    // arguments into the libffi exchange buffer, and the `finally` that
    // releases what those conversions allocated.  Both loops are bounded by
    // `len(self.fargs)`, which is a trace constant once the function type is
    // promoted off a constant cdata.
    //
    // Same class of evidence as `locals_plus_value`, and NOT the `*_nohidden`
    // argument: both bodies are REACHED — each holds a jitcode in the
    // generated metadata artefact — so unreachability proves nothing and the
    // corpus is what has to answer.  Measured on the tree that carries these
    // attributes, `pyre/check.py --backend dynasm` ran the whole synthetic and
    // macro corpus with every committed per-fixture jitstats baseline
    // unchanged.  That covers `fbw_rolled_back_with_effects` corpus-wide
    // rather than per fixture, because it is a `JITSTATS_BADNESS_FIELDS`
    // member in `check.py`, so a rise anywhere would have been a red rather
    // than a statistic.  The run's one failure was
    // `synth/load_name_builtin_cell_fold`'s pypy ratio against the
    // `max-pypy-ratio=1` that fixture carried at the time; `bench: lengthen
    // sub-50ms synthetic fixtures` raised it to 2.5 on main, and the measured
    // 1.6x passes the replacement.
    ("do_call", "ctypefunc.py W_CTypeFunc._call"),
    (
        "release_arguments",
        "ctypefunc.py W_CTypeFunc._call's finally block",
    ),
    // `look_inside_iff` does `func = unroll_safe(func)` (`rlib/jit.py`
    // `look_inside_iff.inner`).  The harvested name is the inlined original
    // (`_orig_*`), not the dispatch wrapper.
    //
    // Evidence: `pypy/module/sys/vm.py getframe` is
    // `@jit.look_inside_iff(jit.isconstant(depth))`.  Windows
    // `pyre/check.py --backend dynasm` on this SHA passed, and a local
    // remesure of the getframe/list/join fixtures did not raise
    // `fbw_rolled_back_with_effects`.
    (
        "_orig_getframe",
        "rlib/jit.py look_inside_iff unroll_safe(getframe)",
    ),
    // `functional.py min_max_multiple_args` is
    // `@jit.look_inside_iff(jit.loop_unrolling_heuristic(args_w, len(args_w), 3))`.
    // `look_inside_iff.inner` does `func = unroll_safe(func)`, so the
    // harvested name is `_orig_min_max_multiple_args`.
    //
    // Evidence: the cranelift re-record after this hint
    // (`foriter_load_special_with`, `global_store_plain_dict_globals`)
    // kept `fbw_rolled_back_with_effects=0`.
    (
        "_orig_min_max_multiple_args",
        "rlib/jit.py look_inside_iff unroll_safe(min_max_multiple_args)",
    ),
    // Same `func = unroll_safe(func)` as above, for
    // `stringmethods.py _str_join_many_items`.
    (
        "_orig_str_join_many_items",
        "rlib/jit.py look_inside_iff unroll_safe(_str_join_many_items)",
    ),
    // Same `func = unroll_safe(func)`, for `listobject.py
    // ListStrategy._extend_from_tuple`, which is `@jit.look_inside_iff(...
    // loop_unrolling_heuristic(tup_w, len(tup_w), UNROLL_CUTOFF))`.  With the
    // descent, no synth fixture's recorded `fbw_rolled_back_with_effects`
    // moved.
    (
        "_orig_extend_from_tuple",
        "rlib/jit.py look_inside_iff unroll_safe(_extend_from_tuple)",
    ),
    // Same `func = unroll_safe(func)`, for `intobject.py _pow_nomod`, which is
    // `@jit.look_inside_iff(lambda iv, iw: jit.isconstant(iw))`.
    (
        "_orig_int_pow_nomod",
        "rlib/jit.py look_inside_iff unroll_safe(_pow_nomod)",
    ),
    // `error.py OperationError.normalize_exception` is `@jit.unroll_safe`.
    // Both pyre methods that share this leaf (the throw-surface table and
    // the `PyError` materialiser) are loop-free, so `look_inside_graph`
    // already admits them; the hint is the upstream decorator, not a
    // descent-scope change.
    (
        "normalize_exception",
        "error.py OperationError.normalize_exception",
    ),
    // `pyframe.py dropvaluesuntil` is `@jit.unroll_safe` and promotes
    // `finaldepth`. The exception-table handler calls it after
    // `lookup_exceptiontable`; without the hint the stack-clear loop is
    // one residual. Measured on this branch after the hint landed:
    // `exception_reraise_tb_depth_jitstress` at N=200 stayed under its
    // 4x pypy gate on dynasm/cranelift/wasm, and the three-backend
    // snapshot was re-recorded.
    ("dropvaluesuntil", "pyframe.py dropvaluesuntil"),
    // `executioncontext.py run_trace_func` is `@jit.unroll_safe`.
    // Loop-free, so `look_inside_graph` already admits it; the hint is
    // the upstream decorator, not a descent-scope change.
    ("run_trace_func", "executioncontext.py run_trace_func"),
    // `pyframe.py` stack helpers and scope init are `@jit.unroll_safe`.
    // Each has a loop over n / cells; without the hint `look_inside_graph`
    // residualizes the whole helper. Descent-scope change matching upstream.
    // popvalues / peekvalues return Vec (two residual words). The hint
    // stays off until that ABI exists; otherwise look_inside residualizes
    // a one-slot pointer and the length is garbage (Linux test_uuid
    // type_id OOB).
    ("dropvalues", "pyframe.py dropvalues"),
    ("pushrevvalues", "pyframe.py pushrevvalues"),
    ("dupvalues", "pyframe.py dupvalues"),
    (
        "initialize_frame_scopes",
        "pyframe.py initialize_frame_scopes",
    ),
    ("locals2fast", "pyframe.py locals2fast"),
    // `pyopcode.py` opcode handlers that unroll a short n-loop.
    (
        "validate_check_exc_match_class",
        "pyopcode.py cmp_exc_match",
    ),
    ("swap_values", "pyopcode.py SWAP"),
    ("make_function", "pyopcode.py MAKE_FUNCTION"),
    ("build_map", "pyopcode.py BUILD_MAP"),
    ("build_set", "pyopcode.py BUILD_SET"),
    ("match_keys", "pyopcode.py MATCH_KEYS"),
    ("match_class", "pyopcode.py _match_class"),
    ("unpack_ex", "pyopcode.py UNPACK_EX"),
    ("build_string", "pyopcode.py BUILD_STRING"),
    ("call_kw", "pyopcode.py CALL_FUNCTION_KW"),
    ("build_map_from_refs", "pyopcode.py BUILD_MAP pair store"),
    ("match_keys_value", "pyopcode.py MATCH_KEYS"),
    ("match_class_value", "pyopcode.py _match_class"),
    (
        "validate_check_eg_match_class",
        "pyopcode.py check_except_star_type_valid",
    ),
    ("exception_match", "baseobjspace.py exception_match"),
    // `tupleobject.py` short-tuple arms. `_unroll_condition` is
    // `loop_unrolling_heuristic(..., UNROLL_CUTOFF=10)`; contains/hash
    // fork to these `@jit.unroll_safe` bodies, and `_compare_tuples`
    // is `@jit.look_inside_iff(_unroll_condition_cmp)`.
    (
        "_descr_contains_unroll_safe",
        "tupleobject.py _descr_contains_unroll_safe",
    ),
    ("_descr_hash_unroll", "tupleobject.py _descr_hash_unroll"),
    (
        "_orig_compare_tuples",
        "rlib/jit.py look_inside_iff unroll_safe(_compare_tuples)",
    ),
    // `intobject.py _pow_mod` / `_pow_nomod` are
    // `@jit.look_inside_iff(... jit.isconstant(iw) ...)`. With the descent,
    // no synth fixture's recorded `fbw_rolled_back_with_effects` moved.
    (
        "_orig__pow_mod",
        "rlib/jit.py look_inside_iff unroll_safe(_pow_mod)",
    ),
    (
        "_orig__pow_nomod",
        "rlib/jit.py look_inside_iff unroll_safe(_pow_nomod)",
    ),
    ("_flat_pycall", "function.py _flat_pycall"),
    ("_flat_pycall_defaults", "function.py _flat_pycall_defaults"),
    // Source now marks `action_dispatcher` `dont_look_inside`. The leaf
    // stays: this test reads `build/llbc/pyre-interpreter.ullbc`, which
    // still harvests `_jit_unroll_safe_action_dispatcher`, and the check
    // is harvested ⊆ reviewed.
    ("action_dispatcher", "executioncontext.py action_dispatcher"),
    ("match_signature", "argument.py _match_signature"),
    // `argument.py unpack` is `@jit.look_inside_iff(lambda self:
    // self._jit_few_keywords)`. `look_inside_iff.inner` does
    // `func = unroll_safe(func)`, so the harvested name is `_orig_unpack`.
    //
    // Evidence: `exc_new_wrapper!` and `os_error_family_new` now rebuild
    // `Arguments` from the flat rest slice and call `unpack` the way
    // `descr_new_base_exception` / `W_OSError.descr_new` do. No-keyword
    // construction is `positional_only`, so `_jit_few_keywords` is true
    // and the body is admitted. `allocate_exception` remains
    // `dont_look_inside`, so the walk still residualizes at allocation
    // rather than expanding a new fold row.
    ("_orig_unpack", "argument.py unpack"),
    (
        "_orig__dict_merge_loop",
        "pyopcode.py look_inside_iff unroll_safe(_dict_merge_loop)",
    ),
    (
        "_unpackiterable_known_length_jitlook",
        "baseobjspace.py _unpackiterable_known_length_jitlook",
    ),
    ("save_float", "interp_pickle.py _Framer.write_binfloat"),
    // `rlib/jit.py look_inside_iff.inner` does `func = unroll_safe(func)`.
    // Harvested names are the inlined originals (`_orig_*`), not the
    // dispatch wrappers.
    //
    // `rstruct/formatiterator.py FormatIterator.interpret` is
    // `@jit.look_inside_iff(lambda self, fmt: jit.isconstant(fmt))`.
    // `do_pack` / `do_unpack` take the same predicate so a constant
    // format reaches `interpret` instead of residualizing at the
    // pack/unpack entry. `PackFormatIterator.operate` and
    // `UnpackFormatIterator.operate` (`pypy/module/struct/formatiterator.py`)
    // are `@jit.unroll_safe`; `pack_values` / `unpack_units` are those
    // unit walks.
    //
    // Evidence: `struct_pack_unpack` jitstats on dynasm/cranelift/wasm
    // keep `fbw_rolled_back_with_effects=0` (a `JITSTATS_BADNESS_FIELDS`
    // member, gated corpus-wide).
    (
        "_orig_parse_format",
        "rlib/jit.py look_inside_iff unroll_safe(FormatIterator.interpret)",
    ),
    (
        "_orig_do_pack",
        "rlib/jit.py look_inside_iff unroll_safe(do_pack)",
    ),
    (
        "_orig_do_unpack",
        "rlib/jit.py look_inside_iff unroll_safe(do_unpack)",
    ),
    (
        "pack_values",
        "formatiterator.py PackFormatIterator.operate",
    ),
    (
        "unpack_units",
        "formatiterator.py UnpackFormatIterator.operate",
    ),
];

/// `builtins::leading_non_null_count` has carried its own `unroll_safe`
/// since it was introduced and appears in every recorded cache snapshot, so
/// its absence means the artefact is too old to say anything — skip loudly
/// rather than pass on a read that proves nothing.
const CONTROL: &str = "leading_non_null_count";

fn harvested_unroll_safe() -> Option<Vec<String>> {
    if !std::path::Path::new(INTERPRETER_LLBC).is_file() {
        eprintln!(
            "skipping: {INTERPRETER_LLBC} is missing; run \
             `python3 scripts/extract-llbc.py pyre-interpreter`"
        );
        return None;
    }
    let llbc = Llbc::load(INTERPRETER_LLBC).expect("load pyre-interpreter.ullbc");
    let hints = harvest_hints_from_llbcs(std::slice::from_ref(&llbc));
    let mut paths: Vec<String> = hints
        .iter()
        .filter(|(_, values)| values.iter().any(|h| h == "unroll_safe"))
        .map(|(path, _)| path.clone())
        .collect();
    paths.sort();
    if !paths.iter().any(|p| leaf(p) == CONTROL) {
        eprintln!(
            "skipping: {INTERPRETER_LLBC} carries no `unroll_safe` on {CONTROL}, \
             so it predates the hint inventory entirely; re-extract to exercise \
             this test (harvested: {paths:?})"
        );
        return None;
    }

    // `REVIEWED_UNROLL_SAFE` and the subset check below both match on the
    // leaf, on the stated assumption that leaves are unambiguous across the
    // interpreter.  Nothing else verifies that.  If an unreviewed function
    // elsewhere later takes an already-reviewed leaf name and is hinted, the
    // subset check would pass on the strength of the other function's review
    // — the one outcome this file exists to prevent.
    let mut leaves: std::collections::HashMap<&str, &str> = std::collections::HashMap::new();
    for path in &paths {
        if let Some(previous) = leaves.insert(leaf(path), path.as_str()) {
            panic!(
                "leaf `{}` is ambiguous between {previous} and {path}. \
                 REVIEWED_UNROLL_SAFE matches by leaf, so it cannot tell them \
                 apart and one would ride on the other's review; key the \
                 inventory by full path before adding either.",
                leaf(path),
            );
        }
    }
    Some(paths)
}

fn leaf(path: &str) -> &str {
    path.rsplit("::").next().unwrap_or(path)
}

#[test]
fn every_unroll_safe_in_the_shipped_llbc_is_a_reviewed_one() {
    let Some(paths) = harvested_unroll_safe() else {
        return;
    };
    for path in &paths {
        assert!(
            REVIEWED_UNROLL_SAFE
                .iter()
                .any(|(name, _)| *name == leaf(path)),
            "{path} carries `unroll_safe` but is not in REVIEWED_UNROLL_SAFE. \
             The attribute admits this graph and its callee closure into the \
             candidate set, so it needs its own evidence — measure \
             `fbw_rolled_back_with_effects` (a rise is a correctness verdict, \
             not a statistic) and the per-fixture jitstats before adding it, \
             then list it here. Harvested: {paths:?}",
        );
    }
}

/// `_unpackiterable_known_length` is `@jit.dont_look_inside`; the
/// `@jit.unroll_safe` body is only `_unpackiterable_known_length_jitlook`,
/// reached from `unpackiterable_unroll`.  The shim must stay unhinted.
#[test]
fn the_known_length_unpack_shim_stays_unhinted() {
    let Some(paths) = harvested_unroll_safe() else {
        return;
    };
    assert!(
        !paths
            .iter()
            .any(|p| leaf(p) == "_unpackiterable_known_length"),
        "`unroll_safe` on _unpackiterable_known_length would open the path \
         upstream fences with `@jit.dont_look_inside`. Harvested: {paths:?}",
    );
}
