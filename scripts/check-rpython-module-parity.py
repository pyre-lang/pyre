#!/usr/bin/env python3
"""Report RPython/PyPy module-name parity gaps in the Rust port.

This is an audit helper for actionable module-name gaps.  It normalizes package
entry points (`__init__.py` in Python, `mod.rs`/`lib.rs` in Rust) so the report
focuses on real module names rather than language-specific filesystem
conventions.  Pyre-local Rust boundaries and permanently-unused PyPy layers
are reported separately as ignored entries, with reasons, so they do not drive
blind ports of code pyre will not use.

With `--symbols`, the helper also compares top-level Python class names with
top-level Rust public type names, and top-level Python function names with
top-level Rust public function names, for already-matched modules.  Thin Rust
reexport wrappers are classified separately so shared implementation crates
such as `majit_ir` and `majit_trace` do not turn into false positives.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModulePair:
    label: str
    python_dir: Path
    rust_dir: Path
    # Further directories holding modules of the same upstream package, when
    # the port splits it across crates. `rust_dir` is searched first.
    extra_rust_dirs: tuple[Path, ...] = ()


@dataclass(frozen=True)
class StringSetPair:
    label: str
    python_path: Path
    python_symbol: str
    rust_path: Path
    rust_function: str


DEFAULT_PAIRS = [
    ModulePair(
        "rpython/annotator",
        Path("rpython/annotator"),
        Path("majit/majit-translate/src/annotator"),
    ),
    ModulePair(
        "rpython/config",
        Path("rpython/config"),
        Path("majit/majit-config/src"),
    ),
    ModulePair(
        "rpython/flowspace",
        Path("rpython/flowspace"),
        Path("majit/majit-translate/src/flowspace"),
    ),
    ModulePair(
        "rpython/jit/codewriter",
        Path("rpython/jit/codewriter"),
        Path("majit/majit-translate/src/codewriter"),
        (Path("majit/majit-jitcode/src/codewriter"),),
    ),
    ModulePair(
        "rpython/jit/metainterp",
        Path("rpython/jit/metainterp"),
        Path("majit/majit-metainterp/src"),
    ),
    ModulePair(
        "rpython/jit/metainterp/ruleopt",
        Path("rpython/jit/metainterp/ruleopt"),
        Path("majit/majit-metainterp/src/ruleopt"),
    ),
    ModulePair(
        "rpython/jit/metainterp/optimizeopt",
        Path("rpython/jit/metainterp/optimizeopt"),
        Path("majit/majit-metainterp/src/optimizeopt"),
    ),
    ModulePair(
        "rpython/rtyper",
        Path("rpython/rtyper"),
        Path("majit/majit-translate/src/translator/rtyper"),
    ),
    ModulePair(
        "rpython/rtyper/lltypesystem",
        Path("rpython/rtyper/lltypesystem"),
        Path("majit/majit-translate/src/translator/rtyper/lltypesystem"),
    ),
    ModulePair(
        "rpython/rtyper/lltypesystem/module",
        Path("rpython/rtyper/lltypesystem/module"),
        Path("majit/majit-translate/src/translator/rtyper/lltypesystem/module"),
    ),
    ModulePair(
        "rpython/rtyper/tool",
        Path("rpython/rtyper/tool"),
        Path("majit/majit-translate/src/translator/rtyper/tool"),
    ),
    ModulePair(
        "rpython/tool/algo",
        Path("rpython/tool/algo"),
        Path("majit/majit-translate/src/tool/algo"),
        (Path("majit/majit-jitcode/src/tool/algo"),),
    ),
    ModulePair(
        "rpython/translator",
        Path("rpython/translator"),
        Path("majit/majit-translate/src/translator"),
    ),
]

DEFAULT_STRING_SET_PAIRS = [
    StringSetPair(
        "codewriter USE_C_FORM",
        Path("rpython/jit/codewriter/assembler.py"),
        "USE_C_FORM",
        Path("majit/majit-translate/src/codewriter/assembler.rs"),
        "use_c_form",
    ),
    StringSetPair(
        "runtime USE_C_FORM",
        Path("rpython/jit/codewriter/assembler.py"),
        "USE_C_FORM",
        Path("pyre/pyre-jit/src/jit/assembler.rs"),
        "use_c_form",
    ),
]

DEFAULT_EXCLUDES = {"test", "__pycache__"}
PACKAGE_ENTRY = "mod"

INTENTIONAL_MISSING: dict[str, dict[str, str]] = {
    "rpython/rtyper/lltypesystem": {
        "ll2ctypes": "permanently unused: pyre never simulates lltype programs through ctypes",
        "llarena": "permanently unused: pyre does not port RPython moving-GC arena simulation",
    },
    "rpython/rtyper/tool": {
        "rffi_platform": "permanently unused: pyre uses Rust/Charon layouts instead of C probing",
    },
    "rpython/translator": {
        "c": "permanently unused: pyre must not grow a local translator/c backend tree",
        "exceptiontransform": "represented in Rust Result/? lowering, not a standalone module",
    },
}

INTENTIONAL_EXTRA: dict[str, dict[str, str]] = {
    "rpython/jit/codewriter": {
        "annotation_state": "local Rust boundary for temporary ValueType/SomeValue projection",
        "insns": "local stable byte table derived from assembler.py's dynamic insns table",
        "jtransform_opname": "local transducer for rtyped helper graphs into jtransform shape",
        "jtransform_shadow": "env-gated diagnostic, never production path",
        "transform_profile": "env-gated drain profiler with no upstream runtime effect",
        "type_state": "local concretetype projection boundary during rtyper cutover",
    },
    "rpython/jit/metainterp": {
        "box_trace": "boxed primitive trace helper shared by pyre-jit and pyre-jit-trace",
        "call_descr": "runtime call-descr boundary for codewriter/backend descriptor surfaces",
        "io_buffer": "compiled-loop stdout buffer; RPython interpreter writes directly",
        "jit": "runtime half of rpython/rlib/jit.py; translator half lives under rlib",
        "jit_state": "Rust trait abstraction for interpreter state",
        "jitcode": "runtime ABI boundary around canonical translate-side jitcode.py port",
        "parity": "test-only trace comparison utilities",
        "recorder": "runtime Trace boundary around opencoder/history recording roles",
        "trace_ctx": "Rust tracing context split across history/compile roles",
    },
    "rpython/rtyper": {
        "cutover": "transitional bridge between legacy and orthodox graph paths",
        "flowspace_adapter": "transitional bridge from pyre graph model to flowspace graph model",
        "legacy_annotator": "temporary legacy graph adapter for cutover",
        "legacy_resolve": "temporary legacy call resolution adapter for cutover",
        "pairtype": "Rust carrier for rtyper-side __extend__(pairtype(...)) blocks",
        "pyre_call_registry": "symbolic FunctionPath registration in place of CPython callable identity",
        "unit_variant_fold": "Rust unit-variant PBC pre-folding before jtransform",
    },
    "rpython/translator": {
        "backend": "intentional non-c module for minimal CBuilder-shaped driver shells",
        "rtyper": "crate-local nesting; upstream rtyper remains compared separately",
        "targetspec": "typed carrier for driver.py from_targetspec's open Python dict",
    },
}

INTENTIONAL_SYMBOL_EXTRA: dict[tuple[str, str], dict[str, dict[str, str]]] = {
    ("rpython/config", "config"): {
        "types": {
            "Child": "Rust enum for OptionDescription._children entries",
            "ConfigValue": "Rust carrier for dynamic __getattr__ return values",
            "DependencyEdge": "Rust carrier for upstream requires/suggests tuple pairs",
            "OptionValue": "Rust carrier for upstream Any-typed option values",
            "Owner": "Rust enum for upstream value-owner strings",
        },
    },
    ("rpython/config", "support"): {
        "functions": {
            "detect_number_of_processors_with_path": "test fixture injection for upstream's filename_or_file parameter",
            "detect_pax_with_path": "test fixture injection for upstream's /proc/self/status read",
        },
    },
    ("rpython/annotator", "description"): {
        "types": {
            "CallTableRow": "Rust alias for upstream's Desc-identity-keyed calltable row dict",
            "DescEntry": "Rust discriminated carrier for upstream Desc subclass instances",
            "DescKey": "Rust identity handle for upstream's Desc object keys",
            "FuncDescEntry": "Rust carrier preserving FunctionDesc/MemoDesc identity under one Desc entry",
            "GraphBuilder": "Rust closure carrier for upstream cachedgraph builder callables",
            "GraphCacheKey": "Rust structured carrier for upstream specialization cache keys",
            "SpecializeResult": "Rust typed carrier for upstream specializers returning graph-or-annotation",
        },
    },
    ("rpython/annotator", "bookkeeper"): {
        "types": {
            "PositionKey": "Rust identity carrier for upstream's position_key tuple",
        },
    },
    ("rpython/annotator", "builtin"): {
        "types": {
            "BuiltinAnalyzer": "Rust function-pointer carrier for upstream analyzer_for registry entries",
        },
    },
    ("rpython/annotator", "model"): {
        "types": {
            "AnnotatorException": "Rust enum carrier for upstream AnnotatorError/UnionError/HarmlesslyBlocked exception variants",
            "DescKind": "Rust enum for upstream Desc subclass identity returned by SomePBC.getKind()",
            "ExitCaseKey": "Rust map key for upstream knowntypedata exit-case tuples",
            "KnownType": "Rust enum carrier for upstream live Python type objects stored in knowntype",
            "KnownTypeData": "Rust alias for upstream knowntypedata nested dict shape",
            "SandboxingPayload": "Rust typed payload carried by SomeBuiltin in place of a dynamic analyser attribute",
            "SomeObjectTrait": "Rust trait surface for methods inherited from upstream SomeObject through Python MRO",
            "SomeValue": "Rust closed enum for the upstream SomeObject subclass lattice",
            "SomeValueTag": "Rust discriminant helper for the upstream SomeObject subclass lattice",
        },
    },
    ("rpython/annotator", "policy"): {
        "types": {
            "PolicyError": "Rust error carrier for upstream get_specializer AttributeError/Exception paths",
            "PolicyHandle": "Rust trait-object handle for upstream policy instances and subclass dispatch",
            "PolicyOps": "Rust trait carrier for upstream policy instance methods overridden by subclasses",
            "Specializer": "Rust enum carrier for upstream specialize.py function objects returned by get_specializer",
        },
    },
    ("rpython/annotator", "signature"): {
        "types": {
            "AnnotationSpec": "Rust enum carrier for upstream annotation(t)'s polymorphic Python input value",
            "ParamType": "Rust enum carrier for upstream enforce_signature_args paramtype shapes",
            "SigArgType": "Rust enum carrier for upstream Sig.argtypes dynamic callable/None/Void/NOT_CONSTANT/type cases",
            "TypeMarker": "Rust enum carrier for upstream rlib.types SelfTypeMarker/AnyTypeMarker classes",
        },
    },
    ("rpython/flowspace", "argument"): {
        "types": {
            "CallShape": "Rust carrier for upstream CallSpec._rawshape()'s anonymous (shape_cnt, shape_keys, shape_star) tuple",
        },
    },
    ("rpython/flowspace", "framestate"): {
        "types": {
            "MergeCell": "Rust carrier for upstream FrameState.mergeable cells, which may be Variable, Constant, or None",
            "StackElem": "Rust carrier for upstream FrameState.stack cells, which may be Variable, Constant, or FlowSignal",
        },
    },
    ("rpython/flowspace", "flowcontext"): {
        "types": {
            "FlowContextError": "Rust error carrier for upstream FlowingError/StopFlowing/FlowSignal/BytecodeCorruption exception unwinds",
            "FlowSignalTag": "Rust discriminant for upstream FlowSignal subclass identity used by rebuild_with_args",
            "FrameBlockKind": "Rust discriminant for upstream FrameBlock subclass identity stored on FrameBlock",
            "PendingBlock": "Rust carrier for upstream pendingblocks list containing SpamBlock or EggBlock instances",
        },
    },
    ("rpython/flowspace", "operation"): {
        "types": {
            "BuiltinException": "Rust carrier for upstream operation.py exception-class objects stored in canraise/can_only_throw tables",
            "CanOnlyThrow": "Rust carrier for upstream dynamic `can_only_throw` attribute values consumed by annotator.model.read_can_only_throw",
            "OpKind": "Rust enum replacing upstream's op namespace plus HLOperationMeta-generated per-op classes",
        },
    },
    ("rpython/flowspace", "model"): {
        "functions": {
            "c_last_exception": "Rust accessor for upstream's `c_last_exception = Constant(last_exception)` module-global sentinel",
        },
        "types": {
            "BlockKey": "Rust identity key for upstream dicts keyed by Block object identity, e.g. mkentrymap/copygraph blockmap",
            "BlockRef": "Rust shared mutable reference carrier for upstream Block object references held by FunctionGraph and Link.target",
            "BlockRefExt": "Rust trait surface for Block methods that must be callable on shared BlockRef handles",
            "ConcretetypePlaceholder": "Rust alias for upstream Variable/Constant.concretetype LowLevelType values assigned by the rtyper",
            "ConstValue": "Rust explicit carrier for upstream Constant.value's arbitrary Python object stored via Hashable",
            "GraphFunc": "Rust stand-in for the live Python function object attached to upstream FunctionGraph.func",
            "GraphKey": "Rust identity key for upstream dicts/sets keyed by FunctionGraph object identity",
            "GraphRef": "Rust shared mutable reference carrier for upstream FunctionGraph objects passed by identity",
            "Hlvalue": "Rust enum for upstream mixed Variable-or-Constant cells in args/results/inputargs",
            "HostCall": "Rust callable carrier for upstream memo-specialized Python function invocation",
            "HostCallError": "Rust explicit error channel for upstream host Python callable invocation exceptions",
            "HostCallableFn": "Rust native-closure type for host-level Python callables executed during annotation",
            "HostEnv": "Rust host namespace carrier for upstream live __builtin__/module lookup during flowspace execution",
            "HostGetAttrError": "Rust explicit error channel for upstream Python getattr/descriptor lookup during flowspace execution",
            "HostObject": "Rust identity carrier for upstream arbitrary Python objects stored in Constant.value",
            "LinkArg": "Rust Option wrapper for upstream Link.args cells that may be None during transient frame-state merges",
            "LinkKey": "Rust identity key for upstream sets/dicts keyed by Link object identity",
            "LinkRef": "Rust shared mutable reference carrier for upstream Link object references held by Blocks",
            "VariableInner": "Rust heap object backing Variable identity and mutable slots that upstream stores directly on the Python object",
        },
    },
    ("rpython/jit/codewriter", "heaptracker"): {
        "types": {
            "GcStructVTableCache": "Rust carrier for upstream's dynamic gccache._cache_gcstruct2vtable attribute plus testing_gcstruct2vtable module dict",
        },
    },
    ("rpython/jit/codewriter", "flatten"): {
        "types": {
            "FlatOp": "Rust enum carrier for upstream SSARepr instruction tuples emitted by GraphFlattener.emitline",
            "IntOvfOp": "Rust discriminant for upstream int_{add,sub,mul}_jump_if_ovf opname strings",
            "RegKind": "Rust enum replacing upstream Register.kind string literals ('int', 'ref', 'float')",
            "RegOrConst": "Rust enum for upstream getcolor's Variable-to-Register or Constant passthrough union",
        },
    },
    ("rpython/jit/codewriter", "jtransform"): {
        "types": {
            "CallEffectKind": "Rust enum carrier for upstream EffectInfo extraeffect classifications during call rewrite",
            "CallEffectOverride": "Rust typed carrier for upstream callcontrol/cpu calldescr effect overrides",
            "GraphTransformConfig": "Rust explicit configuration carrier for upstream Transformer constructor/context fields",
            "GraphTransformNote": "Rust diagnostic note emitted by the graph transform pass; upstream logs through policy.log",
            "GraphTransformResult": "Rust result carrier for upstream in-place Transformer.transform side effects",
            "VableFlag": "Rust enum carrier for upstream Transformer.vable_flags marker values",
            "VirtualizableFieldDescriptor": "Rust typed descriptor for upstream virtualizable field and array field metadata",
        },
    },
    ("rpython/jit/codewriter", "call"): {
        "functions": {
            "effectinfo_from_writeanalyze": "implementation lives beside CallControl's WriteAnalysis in Rust and is re-exported from effectinfo for the upstream public module surface",
        },
        "types": {
            "AnalysisCache": "Rust carrier for upstream per-analyzer DependencyTracker fields seen_rw/seen_gc and raise/effect caches",
            "CallDescriptor": "Rust typed calldescr carrier for upstream cpu.calldescrof(..., EffectInfo) results",
            "CallKind": "Rust enum for upstream guess_call_kind string results regular/residual/builtin/recursive",
            "CanRaise": "Rust enum for upstream RaiseAnalyzer result values True/False/'mem'",
            "DefaultVirtualRefInfoHandle": "Rust zero-sized handle for upstream CallControl.virtualref_info default state",
            "DescrIndexRegistry": "Rust descriptor-index cache replacing upstream cpu descriptor identity side effects",
            "GreenFieldInfoHandle": "Rust typed handle for upstream green-field info stored on JitDriverStaticData",
            "JitDriverStaticData": "Rust typed carrier for upstream jitdrivers_sd entries threaded through CallControl",
            "StaticGreenFieldInfoHandle": "Rust typed handle for upstream static green-field metadata",
            "StructFieldLayout": "Rust carrier for upstream lltype field layout queried through cpu.fielddescrof",
            "StructLayout": "Rust carrier for upstream lltype struct layout queried by descriptor construction",
            "VirtualRefInfoHandle": "Rust typed handle for upstream virtualref_info object",
            "VirtualizableInfoHandle": "Rust typed handle for upstream VirtualizableInfo referenced by callcontrol",
            "WriteAnalysis": "Rust typed carrier for upstream ReadWriteAnalyzer tuple-set output",
        },
    },
    ("rpython/annotator", "specialize"): {
        "types": {
            "MemoFamily": "Rust carrier for upstream Bookkeeper.all_specializations UnionFind plus host-call error latch",
        },
    },
    ("rpython/rtyper", "extfunc"): {
        "types": {
            "ExternalAnnotation": "Rust carrier for upstream annotator.signature.annotation(...) inputs passed to register_external",
        },
    },
    ("rpython/rtyper", "rbuiltin"): {
        "functions": {
            "dispatch_rtyper_makerepr": "Rust dispatcher for upstream SomeBuiltin/SomeBuiltinMethod extension-method routing",
            "pair_builtin_method_convert_from_to": "Rust public helper for upstream `pairtype(BuiltinMethodRepr, BuiltinMethodRepr).convert_from_to`",
            "reset_swap_fallback_hits": "temporary diagnostic counter reset for the legacy cast_ptr_to_int InstanceRepr-to-PtrRepr fallback",
            "rtype_bigint_from": "pyre builtin hook for bigint construction; no upstream RPython builtin object with this exact host name",
            "rtype_malloc_raw": "pyre host-name split for raw malloc lowering; upstream routes through malloc policy helpers",
            "rtype_cast_instance_intrinsic": "pyre-internal front-end pointer-downcast helper with no upstream public builtin",
            "rtype_same_as": "pyre host-name split for same_as lowering; upstream routes through low-level operation helpers",
            "somebuiltin_rtyper_makerepr": "Rust free-function carrier for upstream SomeBuiltin.rtyper_makerepr extension method",
            "somebuiltinmethod_rtyper_makerepr": "Rust free-function carrier for upstream SomeBuiltinMethod.rtyper_makerepr extension method",
            "swap_fallback_hits": "temporary diagnostic counter accessor for the legacy cast_ptr_to_int InstanceRepr-to-PtrRepr fallback",
        },
        "types": {
            "BuiltinTyperFn": "Rust function-pointer carrier for upstream rtype_builtin_* callables stored by typer_for",
        },
    },
    ("rpython/rtyper/lltypesystem", "opimpl"): {
        "types": {
            "FoldFn": "Rust function-pointer carrier for upstream opimpl fold callables returned by get_op_impl",
            "r_longlonglong_arg": "Rust type alias for upstream module-global r_longlonglong_arg",
            "r_longlonglong_result": "Rust type alias for upstream module-global r_longlonglong_result",
        },
        "functions": {
            "_normalize": "Rust public surface for upstream private _normalize debug helper",
            "op_char_eq": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_char_ge": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_char_gt": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_char_le": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_char_lt": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_char_ne": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_float_abs": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_float_add": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_float_eq": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_float_ge": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_float_gt": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_float_is_true": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_float_le": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_float_lt": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_float_mul": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_float_ne": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_float_neg": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_float_sub": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_float_truediv": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_int_abs": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_int_gt": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_int_invert": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_int_is_true": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_int_le": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_int_lshift": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_int_ne": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_int_neg": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_llong_abs": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_llong_add": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_llong_and": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_llong_eq": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_llong_ge": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_llong_gt": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_llong_invert": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_llong_is_true": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_llong_le": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_llong_lt": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_llong_mul": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_llong_ne": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_llong_neg": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_llong_or": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_llong_sub": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_llong_xor": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_uint_add": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_uint_and": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_uint_eq": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_uint_floordiv": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_uint_ge": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_uint_gt": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_uint_invert": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_uint_is_true": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_uint_le": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_uint_lt": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_uint_mod": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_uint_mul": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_uint_ne": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_uint_or": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_uint_sub": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_uint_xor": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_ullong_add": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_ullong_and": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_ullong_eq": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_ullong_floordiv": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_ullong_ge": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_ullong_gt": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_ullong_invert": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_ullong_is_true": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_ullong_le": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_ullong_lt": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_ullong_mod": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_ullong_mul": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_ullong_ne": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_ullong_or": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_ullong_sub": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
            "op_ullong_xor": "Rust explicit function for upstream primitive op generated by get_primitive_op_src",
        },
    },
    ("rpython/rtyper/lltypesystem", "rffi"): {
        "types": {
            "KeepaliveKeeper": "Rust public carrier for upstream private KeepaliveKeeper class produced inside _keeper_for_type",
            "_IsLLPtrEntry": "Rust public carrier for upstream private _IsLLPtrEntry ExtRegistryEntry",
            "_StrFinalizerQueue": "Rust public carrier for upstream private _StrFinalizerQueue finalizer helper",
        },
        "functions": {
            "CStruct_with_hints": "Rust typed helper splitting upstream CStruct keyword-argument hints from the base CStruct constructor",
            "_deprecated_get_nonmovingbuffer": "Rust public placeholder for upstream private _deprecated_get_nonmovingbuffer helper",
            "_get_raw_address_buf_from_string": "Rust public placeholder for upstream private _get_raw_address_buf_from_string helper",
            "_get_structcopy_fn": "Rust public placeholder for upstream private _get_structcopy_fn helper",
            "_isfunctype": "Rust public helper for upstream private _isfunctype memo-specialized predicate",
            "_isllptr": "Rust public helper for upstream private _isllptr predicate",
            "_keeper_for_type": "Rust public helper for upstream private _keeper_for_type cache lookup",
            "_make_wrapper_for": "Rust public placeholder for upstream private _make_wrapper_for helper",
            "c_memcpy": "upstream public binding is assigned by llexternal('memcpy', ...), not parsed as a def",
            "c_memset": "upstream public binding is assigned by llexternal('memset', ...), not parsed as a def",
            "cast": "upstream public binding is assigned as ll2ctypes.force_cast, not parsed as a def",
            "free_nonmovingbuffer": "Rust public placeholder for upstream inner free_nonmovingbuffer_ll helper exposed by make_string_mappings",
            "get_nonmovingbuffer": "Rust public placeholder for upstream inner get_nonmovingbuffer_ll helper exposed by make_string_mappings",
            "get_nonmovingbuffer_final_null": "Rust public placeholder for upstream inner get_nonmovingbuffer_ll_final_null helper exposed by make_string_mappings",
            "ll_liststr2charpp": "Rust public placeholder for upstream private ll_liststr2charpp helper",
            "ptradd": "upstream public binding is assigned as ll2ctypes.force_ptradd, not parsed as a def",
        },
    },
    ("rpython/jit/metainterp/optimizeopt", "vstring"): {
        "functions": {
            "_int_add": "Rust public helper for upstream's private `_int_add` used by vstring copy paths",
            "string_copy_parts": "Rust public helper for upstream's per-class `string_copy_parts` method dispatch",
        },
    },
    ("rpython/jit/metainterp/ruleopt", "generate"): {
        "functions": {
            "def_file": "Rust accessor for upstream's `def_file = os.path.join(here, 'real.rules')` local path",
            "generate_mixin": "Rust public helper for the codegen half inside upstream `main(argv)`",
            "generate_real_rules": "Rust public helper for generating from the vendored upstream `real.rules` source",
            "generate_source": "Rust public helper for parse-plus-codegen over caller-provided rule source",
            "out_file": "Rust accessor for upstream's `out_file = os.path.join(..., 'autogenintrules.py')` local path",
        },
    },
    ("rpython/jit/metainterp/ruleopt", "codegen"): {
        "types": {
            "Path": "Rust structured carrier for upstream create_matcher name_paths tuple/list entries",
            "PathElement": "Rust enum for upstream name_paths tuple elements: op/index pairs and constant markers",
        },
    },
    ("rpython/jit/metainterp/ruleopt", "proof"): {
        "types": {
            "Z3Formula": "pyre intentionally introduces this Rust carrier for a Z3 expression plus its validity condition",
        },
    },
    ("rpython/jit/metainterp/ruleopt", "parse"): {
        "types": {
            "Element": "Rust enum carrier for upstream rule element lists containing Compute or Check instances",
            "ParseError": "Rust error carrier for upstream rply LexingError/ParsingError paths",
            "RuleParseError": "Rust parse-plus-typecheck error carrier for caller-facing parse() failures",
            "RuleType": "Rust enum carrier for upstream expression typ values int/bool/IntBound",
            "SourcePos": "Rust source-position carrier for upstream rply token/sourcepos objects",
        },
    },
    ("rpython/jit/metainterp", "jitprof"): {
        "types": {
            "JitProfiler": "Rust canonical profiler implementation name; upstream public Profiler is exposed as a type alias",
            "JitProfilerSnapshot": "Rust POD snapshot for upstream Profiler._print_stats counter/time readback",
            "ProfilerEventGuard": "Rust RAII carrier for upstream try/finally paired profiler/debug start-stop scopes",
        },
    },
    ("rpython/jit/metainterp", "gc"): {
        "types": {
            "GcDescriptionError": "Rust error carrier for get_description ConfigError, type mismatch, and NotImplementedError paths",
        },
    },
    ("rpython/jit/metainterp", "jitdriver"): {
        "types": {
            "DeclarativeJitDriver": "Rust runtime trait for declaring driver schemas; upstream rlib.jit.JitDriver metadata is lowered into JitDriverStaticData",
            "EntryPoint": "Rust runtime carrier for multiple entry points sharing one driver; upstream stores this through warmspot/jitdriver_sd wiring",
            "JitDriver": "Rust runtime orchestration object; upstream metainterp.jitdriver.py exposes only JitDriverStaticData",
            "TraceContinuationSuspendGuard": "Rust RAII guard for re-entrant trace-continuation suspension; upstream uses interpreter control flow rather than a public class",
        },
    },
    ("rpython/jit/metainterp", "virtualizable"): {
        "functions": {
            "item_size_for_type": "Rust cross-crate layout helper used by majit-macros and compile.rs; upstream resolves this through symbolic.py/llmemory descriptor APIs",
        },
        "types": {
            "VableArrayInfo": "Rust typed carrier for upstream VirtualizableInfo.array_fields/array_descrs metadata",
            "VableArrayStorage": "Rust storage-strategy enum for virtualizable array fields; upstream encodes this in descriptor/layout APIs",
            "VableFieldInfo": "Rust typed carrier for upstream VirtualizableInfo.static_fields/static_field_descrs metadata",
            "VableToken": "Rust enum for upstream vable_token raw integer states",
        },
    },
    ("rpython/jit/metainterp", "virtualref"): {
        "types": {
            "JitVirtualRef": "Rust concrete layout for upstream lltype.GcStruct('JitVirtualRef') allocated in VirtualRefInfo.__init__",
            "ObjectHeader": "Rust concrete layout for upstream rclass.OBJECT super field embedded in JitVirtualRef",
        },
        "functions": {
            "set_vref_gc_type_id": "Rust startup hook for pyre-jit's GC type registration; upstream stores the vref type identity on the lltype/GC object model",
        },
    },
    ("rpython/jit/metainterp", "support"): {
        "types": {
            "Address": "Rust alias for upstream llmemory.Address values crossing the metainterp support.py helper boundary",
            "AddressAsInt": "Rust alias for upstream llmemory.AddressAsInt results returned by support.adr2int/ptr2int",
        },
    },
    ("rpython/rtyper/lltypesystem", "lltype"): {
        "types": {
            "ArrayContainer": "Rust enum carrier for upstream Array versus FixedSizeArray container storage",
            "ArrayCore": "Rust shared value carrier for upstream _array storage and parent links",
            "AttributeError": "Rust error carrier for upstream lltype AttributeError raises",
            "GcKind": "Rust enum field replacing upstream raw/gc/prebuilt sibling subclasses",
            "InteriorOffset": "Rust enum carrier for upstream _interior_ptr offset path entries",
            "LowLevelAdtMember": "Rust typed carrier for upstream ADT method dictionary entries",
            "LowLevelPointerType": "Rust enum carrier for upstream Ptr/InteriorPtr low-level pointer types",
            "LowLevelValue": "Rust closed enum for upstream low-level runtime values",
            "MallocFlavor": "Rust enum carrier for upstream malloc flavor strings",
            "OpaqueCore": "Rust shared value carrier for upstream _opaque storage and parent links",
            "ParentIndex": "Rust enum carrier for upstream's field-name-or-item-index parent tuple element",
            "Parentable": "Rust trait carrier for upstream _parentable container behavior",
            "PtrObj": "Rust enum carrier for upstream _ptr._obj0 delayed/null/concrete cases",
            "PtrTarget": "Rust enum carrier for upstream Ptr.TO container-type variants",
            "StructCore": "Rust shared value carrier for upstream _struct storage and parent links",
            "WeakContainer": "Rust reference carrier replacing upstream weakref-backed parent links",
            "_address": "Rust carrier for upstream llmemory.fakeaddress values exposed through lltype Address slots",
            "_array": "Rust public carrier for upstream private _array low-level container",
            "_arraylenref": "Rust public carrier for upstream private _arraylenref helper object",
            "_endmarker": "Rust public carrier for upstream private _endmarker sentinel",
            "_func": "Rust public carrier for upstream private _func function-pointer container",
            "_interior_ptr": "Rust public carrier for upstream private _interior_ptr pointer object",
            "_opaque": "Rust public carrier for upstream private _opaque low-level container",
            "_ptr": "Rust public carrier for upstream private _ptr pointer object",
            "_ptrEntry": "Rust public carrier for upstream private _ptrEntry ExtRegistryEntry",
            "_ptr_obj": "Rust enum carrier for upstream _ptr._obj0 storage variants",
            "_struct": "Rust public carrier for upstream private _struct low-level container",
            "_subarray": "Rust public carrier for upstream private _subarray helper object",
            "_uninitialized": "Rust public carrier for upstream private _uninitialized sentinel",
            "_wref": "Rust public carrier for upstream private _wref weakref helper",
        },
        "functions": {
            "_cast_whatever": "Rust public placeholder for upstream private _cast_whatever helper",
            "_castdepth": "Rust public placeholder for upstream private _castdepth helper",
            "_get_empty_instance_of_struct_variety": "Rust public placeholder for upstream private _get_empty_instance_of_struct_variety helper",
            "_getconcretetype": "Rust public helper matching upstream private _getconcretetype default argument to getfunctionptr",
            "_make_scoped_allocator": "Rust public placeholder for upstream private _make_scoped_allocator helper",
            "_struct_variety": "Rust public placeholder for upstream private _struct_variety helper",
            "attachRuntimeTypeInfo_with_ptrs": "Rust typed helper for upstream attachRuntimeTypeInfo's optional funcptr/destrptr arguments",
            "fixup_solid": "Rust public helper for upstream _ptr solid-normalization behavior",
            "functionptr_with_external_name": "Rust typed helper for upstream functionptr external-name attribute construction",
            "typeOf_value": "Rust split of upstream typeOf for callers already holding LowLevelValue",
        },
    },
    ("rpython/rtyper/lltypesystem", "llheap"): {
        "functions": {
            "_is_pinned": "Rust public testable surface for upstream's private `_is_pinned` helper",
            "free": "Rust function surface for upstream's `from lltype import free` alias",
            "setfield": "Rust function surface for upstream's `setfield = setattr` alias",
        },
    },
    ("rpython/rtyper/lltypesystem", "llmemory"): {
        "types": {
            "OffsetLayout": "Rust trait carrier for byte-size queries that upstream resolves through runtime/fake memory layout objects",
            "_address_fakeaccessor": "Rust public carrier for upstream's private `_address_fakeaccessor` address property helper",
            "_char_fakeaccessor": "Rust public carrier for upstream's private `_char_fakeaccessor` address property helper",
            "_fakeaccessor": "Rust public carrier for upstream's private `_fakeaccessor` base helper",
            "_float_fakeaccessor": "Rust public carrier for upstream's private `_float_fakeaccessor` address property helper",
            "_signed_fakeaccessor": "Rust public carrier for upstream's private `_signed_fakeaccessor` address property helper",
            "_unsigned_fakeaccessor": "Rust public carrier for upstream's private `_unsigned_fakeaccessor` address property helper",
        },
        "functions": {
            "dead_wref": "Rust accessor for upstream's `dead_wref = _wref(None)._as_ptr()` module-global singleton",
        },
    },
    ("rpython/rtyper/lltypesystem", "llgroup"): {
        "types": {
            "CombinedAnd": "Rust typed return carrier for CombinedSymbolic.__and__, which dynamically returns either an int rest value or a CombinedSymbolic",
            "GroupMember": "Rust identity carrier for the raw lltype struct object that upstream stores directly in group.members",
            "GroupPtr": "Rust pointer-identity carrier for upstream grp._as_ptr() values",
            "HALFWORD": "Rust type alias for upstream platform-selected `HALFWORD = rffi.USHORT/UINT` module assignment",
            "r_halfword": "Rust type alias for upstream platform-selected `r_halfword = rffi.r_ushort/r_uint` module assignment",
        },
    },
    ("rpython/rtyper/lltypesystem", "lloperation"): {
        "types": {
            "_LLOP": "Rust public carrier for upstream's private `_LLOP` class backing the `llop` singleton",
        },
        "functions": {
            "ll_operations": "Rust accessor for upstream's `LL_OPERATIONS` module-global operation table",
        },
    },
    ("rpython/rtyper/lltypesystem", "rstr"): {
        "functions": {
            "do_stringformat": "upstream public binding is assigned as `do_stringformat = LLHelpers.do_stringformat`, not parsed as a `def`",
            "ll_join": "upstream public binding is assigned as `ll_join = LLHelpers.ll_join`, not parsed as a `def`",
        },
    },
    ("rpython/rtyper/lltypesystem", "rdict"): {
        "functions": {
            "_ll_dict_del": "Rust public placeholder for upstream's private `_ll_dict_del` helper",
            "_ll_dict_resize_to": "Rust public placeholder for upstream's private `_ll_dict_resize_to` helper",
            "_ll_dict_setitem_lookup_done": "Rust public placeholder for upstream's private `_ll_dict_setitem_lookup_done` helper",
            "_ll_dictnext": "Rust public placeholder for upstream's private `_ll_dictnext` helper",
            "_ll_free_entries": "Rust public placeholder for upstream's private `_ll_free_entries` helper",
            "_ll_getnextitem": "Rust public placeholder for upstream's private `_ll_getnextitem` helper",
            "_ll_malloc_dict": "Rust public placeholder for upstream's private `_ll_malloc_dict` helper",
            "_ll_malloc_entries": "Rust public placeholder for upstream's private `_ll_malloc_entries` helper",
            "_make_ll_keys_values_items": "Rust public placeholder for upstream's private `_make_ll_keys_values_items` helper",
            "ll_dict_items": "Rust public placeholder for upstream's `ll_dict_items = _make_ll_keys_values_items('items')` module-global callable",
            "ll_dict_keys": "Rust public placeholder for upstream's `ll_dict_keys = _make_ll_keys_values_items('keys')` module-global callable",
            "ll_dict_values": "Rust public placeholder for upstream's `ll_dict_values = _make_ll_keys_values_items('values')` module-global callable",
        },
    },
    ("rpython/rtyper/lltypesystem", "rordereddict"): {
        "functions": {
            "_ll_dict_del": "Rust public placeholder for upstream's private `_ll_dict_del` helper",
            "_ll_dict_del_entry": "Rust public placeholder for upstream's private `_ll_dict_del_entry` helper",
            "_ll_dict_entries_size_too_big": "Rust public placeholder for upstream's private `_ll_dict_entries_size_too_big` helper",
            "_ll_dict_insert_no_index": "Rust public placeholder for upstream's private `_ll_dict_insert_no_index` helper",
            "_ll_dict_move_to_first_shift_items": "Rust public placeholder for upstream's private `_ll_dict_move_to_first_shift_items` helper",
            "_ll_dict_rescue": "Rust public placeholder for upstream's private `_ll_dict_rescue` helper",
            "_ll_dict_resize_to": "Rust public placeholder for upstream's private `_ll_dict_resize_to` helper",
            "_ll_dict_setitem_lookup_done": "Rust public placeholder for upstream's private `_ll_dict_setitem_lookup_done` helper",
            "_ll_dictnext_reversed": "Rust public placeholder for upstream's private `_ll_dictnext_reversed` helper",
            "_ll_empty_array": "Rust public placeholder for upstream's private `_ll_empty_array` helper",
            "_ll_free_entries": "Rust public placeholder for upstream's private `_ll_free_entries` helper",
            "_ll_getnextitem": "Rust public placeholder for upstream's private `_ll_getnextitem` helper",
            "_ll_len_of_d_indexes": "Rust public placeholder for upstream's private `_ll_len_of_d_indexes` helper",
            "_ll_malloc_dict": "Rust public placeholder for upstream's private `_ll_malloc_dict` helper",
            "_ll_malloc_entries": "Rust public placeholder for upstream's private `_ll_malloc_entries` helper",
            "_ll_ptr_to_array_of": "Rust public helper for upstream's private `_ll_ptr_to_array_of` pointer-alias factory",
            "_make_ll_keys_values_items": "Rust public placeholder for upstream's private `_make_ll_keys_values_items` helper",
            "_overallocate_entries_len": "Rust public helper for upstream's private `_overallocate_entries_len` sizing formula",
            "build_ll_dict_lookup_helper_graph": "Rust graph-builder helper for upstream `ll_dict_lookup` direct-call synthesis",
            "build_ll_dictnext_helper_graph": "Rust graph-builder helper for upstream `ll_dictnext` direct-call synthesis",
            "build_ll_write_indexes_helper_graph": "Rust graph-builder helper for upstream write-indexes helper synthesis",
            "ll_dict_items": "Rust public placeholder for upstream's `ll_dict_items = _make_ll_keys_values_items('items')` module-global callable",
            "ll_dict_keys": "Rust public placeholder for upstream's `ll_dict_keys = _make_ll_keys_values_items('keys')` module-global callable",
            "ll_dict_values": "Rust public placeholder for upstream's `ll_dict_values = _make_ll_keys_values_items('values')` module-global callable",
        },
    },
    ("rpython/rtyper/lltypesystem", "rlist"): {
        "functions": {
            "_ll_list_resize": "Rust public placeholder for upstream's private `_ll_list_resize` helper",
            "_ll_list_resize_ge": "Rust public placeholder for upstream's private `_ll_list_resize_ge` helper",
            "_ll_list_resize_hint": "Rust public placeholder for upstream's private `_ll_list_resize_hint` helper",
            "_ll_list_resize_hint_really": "Rust public placeholder for upstream's private `_ll_list_resize_hint_really` helper",
            "_ll_list_resize_le": "Rust public placeholder for upstream's private `_ll_list_resize_le` helper",
            "_ll_list_resize_really": "Rust public placeholder for upstream's private `_ll_list_resize_really` helper",
            "_ll_new_empty_item_array": "Rust public placeholder for upstream's private `_ll_new_empty_item_array` helper",
            "_ll_prebuilt_empty_array": "Rust public placeholder for upstream's private `_ll_prebuilt_empty_array` helper",
        },
    },
    ("rpython/rtyper/lltypesystem", "rbuilder"): {
        "functions": {
            "_ll_append": "Rust public placeholder for upstream's private `_ll_append` helper",
            "_ll_append_multiple_char": "Rust public placeholder for upstream's private `_ll_append_multiple_char` helper",
            "stringbuilder_repr": "Rust accessor for upstream's `stringbuilder_repr = StringBuilderRepr()` singleton",
            "unicodebuilder_repr": "Rust accessor for upstream's `unicodebuilder_repr = UnicodeBuilderRepr()` singleton",
        },
    },
    ("rpython/rtyper/lltypesystem", "rbytearray"): {
        "functions": {
            "_empty_bytearray": "Rust public surface for upstream's private `_empty_bytearray` helper",
            "bytearray_repr": "Rust accessor for upstream's `bytearray_repr = ByteArrayRepr()` singleton",
            "empty": "Rust accessor for upstream's `empty = lltype.malloc(BYTEARRAY, 0, immortal=True)` singleton",
        },
    },
    ("rpython/rtyper", "controllerentry"): {
        "types": {
            "ControlledBox": "Rust value carrier for upstream's controlled_instance_* functions that are XXX sentinels special-cased by ExtRegistryEntry",
        },
    },
    ("rpython/rtyper", "callparse"): {
        "types": {
            "RResult": "Rust carrier for upstream getrresult's dynamic return of either a Repr instance or lltype.Void",
        },
    },
    ("rpython/rtyper", "error"): {
        "types": {
            "TyperWhere": "Rust structured carrier for upstream's dynamic TyperError.where tuple",
        },
    },
    ("rpython/rtyper", "extregistry"): {
        "types": {
            "ExtRegistryEntryKey": "Rust hash/equality carrier for upstream ExtRegistryEntry structural keys",
            "RegisteredAnnotation": "Rust payload for upstream ExtRegistryEntry subclasses returning fixed annotations",
        },
    },
    ("rpython/rtyper", "llinterp"): {
        "types": {
            "WrappedGraph": "Rust concrete carrier for wrap_graph's returned Python closure plus graph/self_arg attributes",
            "_address_of_local_var": "Rust public carrier for upstream's private _address_of_local_var helper class",
            "_address_of_local_var_accessor": "Rust public carrier for upstream's private _address_of_local_var_accessor helper class",
            "_address_of_thread_local": "Rust public carrier for upstream's private _address_of_thread_local helper class",
        },
    },
    ("rpython/rtyper", "rtyper"): {
        "types": {
            "LowLevelFunction": "Rust graph-backed carrier for upstream LowLevelOpList.gendirectcall's live helper-function object",
        },
    },
    ("rpython/rtyper", "rmodel"): {
        "functions": {
            "address_repr": "Rust accessor for upstream raddress.py address_repr singleton while module parity keeps raddress folded into rmodel",
            "can_be_null_rtype_bool": "Rust helper for the upstream CanBeNull.rtype_bool mixin default",
            "impossible_repr": "Rust accessor for upstream `impossible_repr = VoidRepr()` singleton",
            "inputconst_from_lltype": "Rust split of inputconst's low-level type branch for typed callers",
            "rtyper_makekey": "Rust dispatcher for upstream SomeValue.rtyper_makekey extension methods",
            "rtyper_makerepr": "Rust dispatcher for upstream SomeValue.rtyper_makerepr extension methods",
        },
        "types": {
            "AddressRepr": "Rust home for upstream raddress.py AddressRepr while module parity keeps raddress folded into rmodel",
            "BuiltinConstKey": "Rust key carrier for upstream rtyper_makekey builtin const identity cases",
            "DescOrConst": "Rust enum carrier for upstream convert_desc_or_const's Desc-or-Constant dynamic union",
            "InteriorPtrRepr": "Rust home for upstream rptr.py InteriorPtrRepr while module parity keeps rptr folded into rmodel",
            "LLADTMethRepr": "Rust home for upstream rptr.py LLADTMethRepr while module parity keeps rptr folded into rmodel",
            "PtrRepr": "Rust home for upstream rptr.py PtrRepr while module parity keeps rptr folded into rmodel",
            "RTypeResult": "Rust result alias for upstream rtype_* methods returning value-or-None or raising TyperError",
            "ReprKey": "Rust structured key for upstream SomeValue.rtyper_makekey dynamic tuple/list keys",
            "ReprState": "Rust carrier for upstream Repr._initialized plus setup owner state",
            "TypedAddressAccessRepr": "Rust home for upstream raddress.py TypedAddressAccessRepr while module parity keeps raddress folded into rmodel",
        },
    },
    ("rpython/rtyper", "rpbc"): {
        "functions": {
            "assert_no_indirect_call_targets": "pyre verification helper for indirect-call lowering; upstream relies on Python object graph inspection in tests",
            "lower_indirect_calls": "pyre pre-rtyper lowering pass for Rust FunctionPath registries in place of live Python callable identity",
            "pair_function_repr_base_rtype_is_": "Rust public helper for upstream FunctionReprBase pairtype rtype_is_ dispatch",
            "pair_mu_mu_rtype_is_": "Rust public helper for upstream MultipleUnrelatedFrozenPBCRepr pairtype rtype_is_ dispatch",
            "somepbc_rtyper_makerepr": "Rust free-function carrier for upstream SomePBC.rtyper_makerepr extension method",
        },
    },
    ("rpython/rtyper", "rrange"): {
        "types": {
            "RangeIteratorRepr": "Rust implementation of upstream lltypesystem.rrange.RangeIteratorRepr, re-exported from the lltypesystem module",
            "RangeIter": "Rust carrier for upstream lltypesystem.rrange RANGEITER/RANGESTITER low-level structs used by ll_rangenext_* helpers",
        },
    },
    ("rpython/rtyper", "annlowlevel"): {
        "types": {
            "ADTSigArg": "Rust typed carrier for upstream ADTInterface sigtemplate argument entries, which are tuple/list values in Python",
            "ADTSigTemplate": "Rust typed carrier for upstream ADTInterface sigtemplate `(args, result)` tuple values",
            "DelayedConst": "Rust carrier for upstream MixLevelHelperAnnotator.delayedconsts tuple entries",
            "DelayedFunc": "Rust carrier for upstream MixLevelHelperAnnotator.delayedfuncs tuple entries",
            "HLStrEntry": "Rust public carrier for upstream generated hlstr ExtRegistryEntry helper",
            "KeyCompValue": "Rust enum for upstream KeyComp.val's LowLevelType-or-constant dynamic union",
            "LLStrEntry": "Rust public carrier for upstream generated llstr ExtRegistryEntry helper",
            "PendingHelper": "Rust carrier for upstream MixLevelHelperAnnotator.pending tuple entries",
            "StringEntryDirection": "Rust enum for upstream make_string_entries hl/ll helper direction",
            "StringEntryHelper": "Rust callable carrier for upstream make_string_entries generated helpers",
            "StringEntryType": "Rust enum for upstream make_string_entries str/unicode helper family",
        },
        "functions": {
            "cast_base_ptr_to_nongc_instance": "Rust function form of upstream `cast_base_ptr_to_nongc_instance = cast_base_ptr_to_instance` module alias",
        },
    },
    ("rpython/rtyper", "rbool"): {
        "functions": {
            "bool_repr": "Rust accessor for upstream's `bool_repr = BoolRepr()` singleton",
            "pair_bool_float_convert_from_to": "Rust public helper for upstream's `pairtype(BoolRepr, FloatRepr).convert_from_to`",
            "pair_bool_integer_convert_from_to": "Rust public helper for upstream's `pairtype(BoolRepr, IntegerRepr).convert_from_to`",
            "pair_float_bool_convert_from_to": "Rust public helper for upstream's `pairtype(FloatRepr, BoolRepr).convert_from_to`",
            "pair_integer_bool_convert_from_to": "Rust public helper for upstream's `pairtype(IntegerRepr, BoolRepr).convert_from_to`",
        },
    },
    ("rpython/rtyper", "rfloat"): {
        "functions": {
            "float_repr": "Rust accessor for upstream's `float_repr = FloatRepr()` singleton",
            "rtype_compare_template": "Rust public helper for upstream's private `_rtype_compare_template` used by pairtype dispatch",
            "rtype_template": "Rust public helper for upstream's private `_rtype_template` used by pairtype dispatch",
        },
    },
    ("rpython/rtyper", "rtuple"): {
        "functions": {
            "ll_equal": "Rust public helper for upstream's private `_ll_equal` fallback comparator",
            "pair_tuple_int_rtype_getitem": "Rust public dispatcher for upstream `__extend__(pairtype(TupleRepr, IntegerRepr)).rtype_getitem`",
            "pair_tuple_repr_rtype_contains": "Rust public dispatcher for upstream `__extend__(pairtype(TupleRepr, Repr)).rtype_contains`",
            "pair_tuple_tuple_convert_from_to": "Rust public dispatcher for upstream `__extend__(pairtype(TupleRepr, TupleRepr)).convert_from_to`",
            "pair_tuple_tuple_rtype_add": "Rust public dispatcher for upstream `__extend__(pairtype(TupleRepr, TupleRepr)).rtype_add`",
            "pair_tuple_tuple_rtype_eq": "Rust public dispatcher for upstream `__extend__(pairtype(TupleRepr, TupleRepr)).rtype_eq`",
            "pair_tuple_tuple_rtype_is_": "Rust public dispatcher for upstream `__extend__(pairtype(TupleRepr, TupleRepr)).rtype_is_`",
            "pair_tuple_tuple_rtype_ne": "Rust public dispatcher for upstream `__extend__(pairtype(TupleRepr, TupleRepr)).rtype_ne`",
        },
    },
    ("rpython/rtyper", "rint"): {
        "functions": {
            "pair_float_integer_convert_from_to": "Rust public helper for upstream's `pairtype(FloatRepr, IntegerRepr).convert_from_to`",
            "pair_integer_float_convert_from_to": "Rust public helper for upstream's `pairtype(IntegerRepr, FloatRepr).convert_from_to`",
            "pair_integer_integer_convert_from_to": "Rust public helper for upstream's `pairtype(IntegerRepr, IntegerRepr).convert_from_to`",
            "rtype_add_ovf": "Rust public helper for upstream's `pairtype(IntegerRepr, IntegerRepr).rtype_add_ovf` method",
            "rtype_call_helper": "Rust public helper for upstream's private `_rtype_call_helper` used by pairtype dispatch",
            "rtype_compare_template": "Rust public helper for upstream's private `_rtype_compare_template` used by pairtype dispatch",
            "rtype_template": "Rust public helper for upstream's private `_rtype_template` used by pairtype dispatch",
            "signed_repr": "Rust accessor for upstream's `signed_repr = getintegerrepr(...)` singleton",
            "signedlonglong_repr": "Rust accessor for upstream's `signedlonglong_repr = getintegerrepr(...)` singleton",
            "signedlonglonglong_repr": "Rust accessor for upstream's `signedlonglonglong_repr = getintegerrepr(...)` singleton",
            "unsigned_repr": "Rust accessor for upstream's `unsigned_repr = getintegerrepr(...)` singleton",
            "unsignedlonglong_repr": "Rust accessor for upstream's `unsignedlonglong_repr = getintegerrepr(...)` singleton",
            "unsignedlonglonglong_repr": "Rust accessor for upstream's `unsignedlonglonglong_repr = getintegerrepr(...)` singleton",
        },
    },
    ("rpython/rtyper", "rnone"): {
        "functions": {
            "none_repr": "Rust accessor for upstream's `none_repr = NoneRepr()` singleton",
            "pair_any_none_convert_from_to": "Rust public helper for upstream's `pairtype(Repr, NoneRepr).convert_from_to`",
            "pair_any_none_rtype_is_": "Rust public helper for upstream's `pairtype(Repr, NoneRepr).rtype_is_`",
            "pair_none_any_convert_from_to": "Rust public helper for upstream's `pairtype(NoneRepr, Repr).convert_from_to`",
            "pair_none_any_rtype_is_": "Rust public helper for upstream's `pairtype(NoneRepr, Repr).rtype_is_`",
        },
    },
    ("rpython/rtyper", "raddress"): {
        "types": {
            "Address": "Rust type alias exposing upstream's imported llmemory.Address surface",
            "fakeaddress": "Rust type alias exposing upstream's imported llmemory.fakeaddress surface",
        },
    },
    ("rpython/rtyper/tool", "mkrffi"): {
        "types": {
            "CType": "Rust explicit carrier for upstream live ctypes type objects passed to RffiSource.proc_tp",
            "FunctionDecl": "Rust explicit carrier for upstream ctypes._CFuncPtr entries consumed by proc_func/proc_namespace",
            "MkrffiError": "Rust error carrier for upstream NotImplementedError paths in ctypes-to-rffi conversion",
            "SimpleCType": "Rust enum for upstream SIMPLE_TYPE_MAPPING ctypes keys",
            "StructDecl": "Rust explicit carrier for upstream ctypes.Structure classes and _fields_",
        },
    },
    ("rpython/rtyper/tool", "rfficache"): {
        "functions": {
            "ask_gcc_source": "Rust deterministic source-construction half of upstream ask_gcc without invoking a C compiler",
            "default_includes": "Rust helper for upstream ask_gcc's platform-dependent include list",
            "parse_signof_c_type": "Rust deterministic parser half of upstream signof_c_type after ask_gcc output exists",
            "parse_sizeof_c_type": "Rust deterministic parser half of upstream sizeof_c_type after ask_gcc output exists",
            "parse_sizeof_c_types": "Rust deterministic parser half of upstream sizeof_c_types after ask_gcc output exists",
            "signof_question": "Rust deterministic question-construction half of upstream signof_c_type",
            "sizeof_question": "Rust deterministic question-construction half of upstream sizeof_c_types",
        },
        "types": {
            "IntTypeDecl": "Rust explicit carrier for upstream populate_inttypes (name, c_name, signed) tuple rows",
            "NumberType": "Rust carrier for upstream lltype.build_number plus rarithmetic.build_int registry entries",
            "RffiCacheError": "Rust error carrier for upstream assert/ValueError failures while parsing compiler answers",
        },
    },
    ("rpython/rtyper/lltypesystem/module", "ll_math"): {
        "functions": {
            "ll_math_acos": "upstream public binding is generated by globals()['ll_math_' + name] = new_unary_math_function(...), not parsed as a def",
            "ll_math_acosh": "upstream public binding is generated by globals()['ll_math_' + name] = new_unary_math_function(...), not parsed as a def",
            "ll_math_asin": "upstream public binding is generated by globals()['ll_math_' + name] = new_unary_math_function(...), not parsed as a def",
            "ll_math_asinh": "upstream public binding is generated by globals()['ll_math_' + name] = new_unary_math_function(...), not parsed as a def",
            "ll_math_atan": "upstream public binding is generated by globals()['ll_math_' + name] = new_unary_math_function(...), not parsed as a def",
            "ll_math_atanh": "upstream public binding is generated by globals()['ll_math_' + name] = new_unary_math_function(...), not parsed as a def",
            "ll_math_ceil": "upstream public binding is generated by globals()['ll_math_' + name] = new_unary_math_function(...), not parsed as a def",
            "ll_math_copysign": "upstream public binding is assigned as `ll_math_copysign = math_copysign`, not parsed as a def",
            "ll_math_cosh": "upstream public binding is generated by globals()['ll_math_' + name] = new_unary_math_function(...), not parsed as a def",
            "ll_math_exp": "upstream public binding is generated by globals()['ll_math_' + name] = new_unary_math_function(...), not parsed as a def",
            "ll_math_expm1": "upstream public binding is generated by globals()['ll_math_' + name] = new_unary_math_function(...), not parsed as a def",
            "ll_math_fabs": "upstream public binding is generated by globals()['ll_math_' + name] = new_unary_math_function(...), not parsed as a def",
            "ll_math_floor": "upstream public binding is assigned as `ll_math_floor = math_floor`, not parsed as a def",
            "ll_math_sinh": "upstream public binding is generated by globals()['ll_math_' + name] = new_unary_math_function(...), not parsed as a def",
            "ll_math_tan": "upstream public binding is generated by globals()['ll_math_' + name] = new_unary_math_function(...), not parsed as a def",
            "ll_math_tanh": "upstream public binding is generated by globals()['ll_math_' + name] = new_unary_math_function(...), not parsed as a def",
        },
        "types": {
            "MathError": "Rust error carrier for upstream ValueError/OverflowError raises in ll_math helpers",
            "MathExternal": "Rust carrier for upstream rffi.llexternal function-pointer placeholders",
            "UnaryMathFunction": "Rust typed carrier for upstream new_unary_math_function's generated helper function object",
        },
    },
    ("rpython/translator", "timing"): {
        "types": {
            "SystemClock": "Rust default clock implementing upstream Timer(timer=time.time) injection",
            "TimeSource": "Rust trait carrier for upstream Timer(timer=callable) clock injection",
        },
    },
    ("rpython/translator", "platform"): {
        "functions": {
            "execute": "Rust free-function helper for the upstream Platform.execute method slice used without a platform object",
        },
        "types": {
            "EnvMapping": "Rust alias for upstream env dict copied by Platform.execute",
        },
    },
    ("rpython/translator", "simplify"): {
        "types": {
            "GraphKeyForSeen": "Rust graph-identity carrier for upstream's seen dict keys",
        },
    },
    ("rpython/tool/algo", "graphlib"): {
        "types": {
            "DfsEvent": "Rust enum carrier for upstream depth_first_search ('start'/'stop') event strings",
            "EdgeDict": "Rust alias for upstream's official edges dict shape",
            "VertexSet": "Rust trait carrier for upstream's set-or-dict vertices protocol",
        },
    },
    ("rpython/tool/algo", "sparsemat"): {
        "types": {
            "SparseMatError": "Rust error carrier for upstream sparse matrix ValueError/IndexError paths",
        },
    },
    ("rpython/tool/algo", "unionfind"): {
        "types": {
            "UnionFindInfo": "Rust trait carrier for upstream dynamic info.absorb(other_info) root payloads",
        },
    },
    ("rpython/jit/codewriter", "codewriter"): {
        "types": {
            "AllJitCodes": "Rust carrier pairing upstream CallControl.jitcodes and CallControl.all_jitcodes for generated::with_all_jitcodes test-fixture access",
        },
    },
    ("rpython/jit/codewriter", "jitcode"): {
        "functions": {
            "bh_field_specs_from_size_descr": "Rust cross-crate helper for pyre-jit-trace to serialize SizeDescr.all_fielddescrs into BhFieldSpec values",
        },
        "types": {
            "BhCallDescr": "Rust serializable mirror of upstream backend CallDescr for blackhole/runtime descriptor tables",
            "BhDescr": "Rust enum carrier for upstream heterogeneous AbstractDescr subclasses in assembler descr lists",
            "BhFieldSpec": "Rust serializable mirror of upstream FieldDescr metadata consumed by blackhole/runtime code",
            "BhInteriorFieldSpec": "Rust serializable mirror of upstream InteriorFieldDescr metadata",
            "BhSizeSpec": "Rust serializable mirror of upstream SizeDescr metadata including all_fielddescrs",
            "CallResultErasedKey": "Rust enum for upstream descr.py RESULT_ERASED cache-key component",
            "JitCodeBody": "Rust OnceLock payload for upstream JitCode.setup(...) fields after Arc shell creation",
            "JitCodeHandle": "Rust identity wrapper for Arc<JitCode>, matching upstream object-identity keyed JitCode sets/lists",
            "StrConstDescriptor": "Rust load-time descriptor for prebuilt string constants that upstream materializes through lltype constants",
        },
    },
    ("rpython/jit/codewriter", "support"): {
        "types": {
            "BuiltinFuncSpec": "Rust typed carrier for upstream builtin_func_for_spec's dynamic (c_func, LIST_OR_DICT) tuple plus fnaddr metadata",
            "BuiltinFuncSpecCacheKey": "Rust structured cache key for upstream rtyper._builtin_func_for_spec_cache tuple keys",
            "NeedResultType": "Rust enum carrier for upstream helper.need_result_type attribute values",
            "NormalizeSlot": "Rust enum carrier for upstream parse_oopspec argtuple entries including Index and constants",
            "NormalizedArg": "Rust enum carrier for upstream normalize_opargs outputs, distinguishing passthrough Variables from materialized Constants",
        },
    },
    ("rpython/jit/codewriter", "longlong"): {
        "functions": {
            "extract_bits": "upstream public binding is assigned as `extract_bits = longlong2float.float2longlong`/`lambda x: x`, not parsed as a `def`",
            "getfloatstorage": "upstream public binding is assigned as `getfloatstorage = lambda x: x`/`longlong2float.float2longlong`, not parsed as a `def`",
            "gethash": "upstream public binding is assigned as `gethash = compute_hash`/`lambda xll: ...`, not parsed as a `def`",
            "gethash_fast": "upstream public binding is assigned as `gethash_fast = longlong2float.float2longlong`/`gethash`, not parsed as a `def`",
            "getrealfloat": "upstream public binding is assigned as `getrealfloat = lambda x: x`/`longlong2float.longlong2float`, not parsed as a `def`",
            "is_longlong": "upstream public binding is assigned as `is_longlong = lambda TYPE: ...`, not parsed as a `def`",
        },
        "types": {
            "r_float_storage": "Rust type alias for upstream's platform-selected `r_float_storage = float/r_longlong` binding",
        },
    },
    ("rpython/jit/codewriter", "policy"): {
        "types": {
            "DefaultJitPolicy": "Rust concrete default implementor for upstream's instantiable `JitPolicy` base class",
            "JitPolicyState": "Rust shared state carrier for fields stored directly on upstream `JitPolicy` instances",
        },
    },
    ("rpython/translator", "driver"): {
        "functions": {
            "annotated_jit_entrypoints_get": "Rust accessor for upstream rlib.entrypoint.annotated_jit_entrypoints module-global list",
            "annotated_jit_entrypoints_register": "Rust mutator for upstream annotated_jit_entrypoints.append((func, argtypes))",
            "secondary_entrypoints_get": "Rust accessor for upstream rlib.entrypoint.secondary_entrypoints dict",
            "secondary_entrypoints_keys": "Rust helper for upstream secondary_entrypoints.keys() error reporting",
            "secondary_entrypoints_register": "Rust mutator for upstream secondary_entrypoints.setdefault(key, []).append((func, argtypes))",
        },
        "types": {
            "EntryPointSpec": "Rust typed carrier for upstream `(func, argtypes)` tuples stored in rlib.entrypoint globals",
            "LibDef": "Rust typed carrier for upstream setup_library's duck-typed `libdef.functions` object",
            "ProceedGoals": "Rust enum carrier for TranslationDriver.proceed's dynamic None/string/list goal argument",
        },
    },
    ("rpython/translator", "translator"): {
        "types": {
            "CallGraphEdge": "Rust value carrier for upstream callgraph dict values `(caller_graph, callee_graph)`",
            "CallGraphKey": "Rust identity key for upstream callgraph dict key `(caller_graph, callee_graph, position_tag)`",
            "FlowingFlags": "Rust carrier for upstream TranslationContext.__init__(**flowing_flags)",
            "Platform": "Rust minimal carrier for upstream get_platform(config) result stored on TranslationContext",
            "TranslationConfig": "Rust typed carrier for upstream config object fields consumed by TranslationContext",
            "TranslationOptions": "Rust typed carrier for upstream config.translation option fields consumed by TranslationContext",
        },
    },
}

INTENTIONAL_SYMBOL_MISSING: dict[tuple[str, str], dict[str, dict[str, str]]] = {
    ("rpython/annotator", "argument"): {
        "types": {
            "ArgErrCount": "represented by ArgErr::Count enum variant",
            "ArgErrMultipleValues": "represented by ArgErr::MultipleValues enum variant",
            "ArgErrUnknownKwds": "represented by ArgErr::UnknownKwds enum variant",
        },
    },
    ("rpython/annotator", "classdesc"): {
        "types": {
            "Sample": "CPython member-descriptor probe for MemberDescriptorTypes; pyre uses typed HostObject/classdict entries instead",
        },
    },
    ("rpython/config", "config"): {
        "types": {
            "BoolConfigUpdate": "deferred with optparse integration until CLI driver code lands",
            "ConfigUpdate": "deferred with optparse integration until CLI driver code lands",
            "ConflictConfigError": "represented by ConfigError::Conflict instead of a separate Rust exception type",
            "OptHelpFormatter": "deferred with optparse integration until CLI driver code lands",
        },
        "functions": {
            "make_dict": "deferred with optparse/config dump integration until a consumer lands",
            "to_optparse": "deferred with optparse integration until CLI driver code lands",
        },
    },
    ("rpython/config", "translationoption"): {
        "functions": {
            "get_platform": "deferred with translator.platform pick_platform until platform compile integration is ported",
            "set_platform": "deferred with translator.platform set_platform until platform compile integration is ported",
        },
    },
    ("rpython/annotator", "specialize"): {
        "types": {
            "AccessDirect": "represented by GraphCacheKey::AccessDirect instead of a standalone marker class",
        },
    },
    ("rpython/flowspace", "flowcontext"): {
        "types": {
            "Break": "represented by FlowSignal::Break rather than a standalone subclass",
            "Continue": "represented by FlowSignal::Continue rather than a standalone subclass",
            "ExceptBlock": "represented by FrameBlockKind::Except on FrameBlock rather than a standalone subclass",
            "FinallyBlock": "represented by FrameBlockKind::Finally on FrameBlock rather than a standalone subclass",
            "IterBlock": "represented by FrameBlockKind::Iter on FrameBlock rather than a standalone subclass",
            "LoopBlock": "represented by FrameBlockKind::Loop on FrameBlock rather than a standalone subclass",
            "Raise": "represented by FlowSignal::Raise rather than a standalone subclass",
            "RaiseImplicit": "represented by FlowSignal::RaiseImplicit rather than a standalone subclass",
            "Return": "represented by FlowSignal::Return rather than a standalone subclass",
            "WithBlock": "represented by FrameBlockKind::With on FrameBlock rather than a standalone subclass",
        },
        "functions": {
            "binaryoperation": "Python opcode-method factory is represented inline by FlowContext::handle_bytecode dispatch",
            "unaryoperation": "Python opcode-method factory is represented inline by FlowContext::handle_bytecode dispatch",
            "unsupportedoperation": "Python opcode-method factory is represented inline by FlowContext::handle_bytecode dispatch",
        },
    },
    ("rpython/flowspace", "model"): {
        "types": {
            "ConstException": "represented by FSException carrying Constant-wrapped type/value rather than Python multiple inheritance",
            "UnwrapException": "Rust typed Hlvalue APIs do not unwrap Variables through Python exceptions",
            "WrapException": "Rust constant wrapping returns typed Result/Option fallbacks at the call sites instead of raising a marker exception",
        },
        "functions": {
            "flattenobj": "Python dynamic recursive tuple/list flattener is unnecessary because Rust graph walkers traverse typed fields directly",
        },
    },
    ("rpython/jit/metainterp/ruleopt", "parse"): {
        "functions": {
            "addkeyword": "upstream rply lexer table builder; Rust lex() encodes the fixed keyword table directly",
            "addtok": "upstream rply lexer table builder; Rust lex() encodes the fixed token table directly",
            "args": "upstream rply production callback; Rust ExprParser::parse_args implements the grammar privately",
            "attr_or_method": "upstream rply production callback; Rust ExprParser parses dotted attribute/method calls privately",
            "check": "upstream rply production callback; Rust parse_rule_element builds Element::Check privately",
            "compute_element": "upstream rply production callback; Rust parse_rule_element builds Element::Compute privately",
            "elements": "upstream rply production callback; Rust parse() accumulates rule elements line-by-line",
            "expression_binop": "upstream rply production callback; Rust ExprParser::parse_expression implements precedence privately",
            "expression_name": "upstream rply production callback; Rust ExprParser::parse_prefix builds Name privately",
            "expression_number": "upstream rply production callback; Rust ExprParser::parse_prefix builds Number privately",
            "expression_parens": "upstream rply production callback; Rust ExprParser::parse_prefix handles parenthesized expressions privately",
            "expression_unary": "upstream rply production callback; Rust ExprParser::parse_prefix builds Invert privately",
            "file": "upstream rply production callback; Rust parse() builds File directly",
            "funccall": "upstream rply production callback; Rust ExprParser::parse_prefix builds FuncCall privately",
            "maybesorry": "upstream rply production callback; Rust parse_rule_element handles SORRY_Z3 directly",
            "methodcall": "upstream rply production callback; Rust ExprParser parses method calls privately",
            "newlines": "upstream rply production callback for line separators; Rust parse() consumes source line structure directly",
            "pattern_const": "upstream rply production callback; Rust PatternParser::parse_pattern builds PatternConst privately",
            "pattern_op": "upstream rply production callback; Rust PatternParser::parse_pattern builds PatternOp privately",
            "pattern_var": "upstream rply production callback; Rust PatternParser::parse_pattern builds PatternVar privately",
            "patternargs": "upstream rply production callback; Rust PatternParser::parse_patternargs implements it privately",
            "print_conflicts": "upstream rply parser-conflict debug hook; Rust hand parser has no generated LR table to inspect",
            "production": "upstream decorator wrapping rply production callbacks with source positions; Rust parsers attach SourcePos directly",
            "rule": "upstream rply production callback; Rust parse_rule_header plus PartialRule::finish builds Rule privately",
        },
    },
    ("rpython/rtyper", "error"): {
        "types": {
            "MissingRTypeOperation": "represented by TyperError::MissingRTypeOperation enum variant",
        },
    },
    ("rpython/rtyper", "rmodel"): {
        "functions": {
            "make_missing_op": "Python dynamically setattr()s missing rtype_* methods; Rust encodes the same defaults statically on the Repr trait",
        },
        "types": {
            "BrokenReprTyperError": "represented by TyperError::BrokenRepr enum variant",
        },
    },
    ("rpython/rtyper", "debug"): {
        "types": {
            "Entry": "two upstream ExtRegistryEntry subclasses are represented by debug_assert/debug_assert_not_none llops and rtyper lowering, not standalone Rust public classes",
        },
    },
    ("rpython/rtyper", "callparse"): {
        "types": {
            "ConstHolder": "represented by the Holder::Const enum variant instead of a standalone subclass",
            "ItemHolder": "represented by the Holder::Item enum variant instead of a standalone subclass",
            "NewTupleHolder": "represented by the Holder::NewTuple enum variant instead of a standalone subclass",
            "VarHolder": "represented by the Holder::Var enum variant instead of a standalone subclass",
        },
    },
    ("rpython/jit/metainterp/optimizeopt", "heap"): {
        "types": {
            "AbstractCachedEntry": "implemented as private Rust helper methods/free helpers shared by private CachedField and ArrayCachedItem structs",
            "ArrayCacheSubMap": "implemented as a private Rust cache struct; not exported from the optimizer module API",
            "ArrayCachedItem": "implemented as a private Rust cache struct; not exported from the optimizer module API",
            "CachedField": "implemented as a private Rust cache struct; not exported from the optimizer module API",
        },
    },
    ("rpython/rtyper", "extregistry"): {
        "types": {
            "AutoRegisteringType": "Python metaclass registration side-effect is replaced by explicit Rust ExtRegistryEntry variants and registration matches",
        },
    },
    ("rpython/rtyper", "llinterp"): {
        "types": {
            "Tracer": "deferred llinterp HTML trace/debug facility; Rust keeps the upstream tracer slot opaque until eval tracing hooks land",
        },
        "functions": {
            "type_name": "deferred with LLException.__str__ parity; Rust LLException display stays opaque until exception type-name containers are ported",
        },
    },
    ("rpython/rtyper", "normalizecalls"): {
        "functions": {
            "create_class_constructors": "deferred class-PBC constructor-call support; instantiate helpers and class-PBC getattr merging are ported separately",
        },
        "types": {
            "TooLateForNewSubclass": "Rust assign_inheritance_ids uses eager recomputation/append-only fallback instead of upstream's lazy symbolic exception",
            "TotalOrderSymbolic": "Rust stores materialized subclass-range ids after sorting the same reversed-MRO witnesses, not lazy ComputedIntSymbolic objects",
        },
    },
    ("rpython/rtyper", "annlowlevel"): {
        "types": {
            "cachedtype": "Python metaclass cache is represented by explicit Rust Lazy/LazyLock/cache maps at each call site",
        },
    },
    ("rpython/rtyper/tool", "rfficache"): {
        "functions": {
            "signof_c_type": "permanently unused C probing wrapper around ask_gcc; pyre uses parse_signof_c_type once external answers are supplied",
        },
    },
    ("rpython/translator", "platform"): {
        "functions": {
            "is_host_build": "deferred with platform factory/global platform selection; current Rust port only preserves Platform.execute",
            "pick_platform": "deferred with platform factory/global platform selection and C backend compile integration",
            "set_platform": "deferred with platform factory/global platform selection and C backend compile integration",
        },
    },
    ("rpython/translator", "transform"): {
        "functions": {
            "insert_ll_stackcheck": "deferred rtyper-phase stack-check insertion; transform.rs currently ports the annotator-phase graph transforms only",
        },
    },
    ("rpython/translator", "driver"): {
        "functions": {
            "taskdef": "Python decorator attaching task metadata is represented by private Rust TaskDef values passed to SimpleTaskEngine::register_task",
        },
    },
    ("rpython/translator", "unsimplify"): {
        "functions": {
            "call_final_function": "deferred until MixLevelHelperAnnotator-backed finalizer graph injection is ported",
            "call_initial_function": "deferred until MixLevelHelperAnnotator-backed startup graph injection is ported",
        },
    },
    ("rpython/rtyper/lltypesystem", "llmemory"): {
        "types": {
            "AddressAsInt": "symbolic address-int carrier is intentionally absent; pyre cast_adr_to_int supports emulated/forced numeric folds only",
            "GCHeaderAntiOffset": "GC-transform-only offset; pyre does not run the RPython GC transformer",
            "GCHeaderOffset": "GC-transform-only offset; pyre does not run the RPython GC transformer",
        },
    },
    ("rpython/rtyper/lltypesystem", "opimpl"): {
        "functions": {
            "checkadr": "Rust constfold helpers enforce address carriers through ConstValue pattern matches",
            "checkptr": "Rust constfold helpers enforce pointer carriers through ConstValue pattern matches",
            "get_primitive_op_src": "upstream dynamic primitive-op factory is represented by explicit Rust op_* functions and registry entries",
            "op_cast_pointer": "deferred until ConstValue carries the lltype pointer structure needed by lltype.cast_pointer",
            "op_cast_primitive": "deferred until RESTYPE-threaded primitive casts cover the full lltype primitive lattice",
            "op_combine_ushort": "deferred until llgroup CombinedSymbolic carriers are represented in ConstValue",
            "op_debug_fatalerror": "side-effecting/debug operation intentionally omitted from the constfold registry",
            "op_debug_flush": "side-effecting/debug operation intentionally omitted from the constfold registry",
            "op_debug_flush_log": "side-effecting/debug operation intentionally omitted from the constfold registry",
            "op_debug_nonnull_pointer": "side-effecting/debug assertion operation intentionally omitted from the constfold registry",
            "op_debug_offset": "side-effecting/debug operation intentionally omitted from the constfold registry",
            "op_direct_arrayitems": "deferred until lltype interior-pointer carriers are represented in ConstValue",
            "op_direct_fieldptr": "deferred until lltype interior-pointer carriers are represented in ConstValue",
            "op_direct_ptradd": "deferred until raw pointer arithmetic carriers include bounds/offset metadata",
            "op_extract_ushort": "deferred until llgroup CombinedSymbolic carriers are represented in ConstValue",
            "op_gc_bit": "side-effecting/randomized GC helper intentionally omitted from the constfold registry",
            "op_gc_gettypeptr_group": "deferred until llgroup/vtable carriers are represented in ConstValue",
            "op_gc_ignore_finalizer": "side-effecting GC helper intentionally omitted from the constfold registry",
            "op_gc_increase_root_stack_depth": "side-effecting GC helper intentionally omitted from the constfold registry",
            "op_gc_move_out_of_nursery": "side-effecting GC helper intentionally omitted from the constfold registry",
            "op_gc_stack_bottom": "side-effecting GC helper intentionally omitted from the constfold registry",
            "op_gc_store": "side-effecting memory store intentionally omitted from the constfold registry",
            "op_gc_store_indexed": "side-effecting memory store intentionally omitted from the constfold registry",
            "op_gc_writebarrier": "side-effecting GC write-barrier helper intentionally omitted from the constfold registry",
            "op_gc_writebarrier_before_copy": "side-effecting GC write-barrier helper intentionally omitted from the constfold registry",
            "op_get_group_member": "deferred until llgroup GroupMemberOffset carriers are represented in ConstValue",
            "op_get_member_index": "upstream placeholder raises NotImplementedError; Rust keeps it absent from the fold registry",
            "op_get_next_group_member": "deferred until llgroup GroupMemberOffset carriers are represented in ConstValue",
            "op_getarrayitem": "deferred until lltype array/pointer carriers support immutable element reads in ConstValue",
            "op_getarraysize": "deferred until lltype array/pointer carriers are represented in ConstValue",
            "op_getarraysubstruct": "deferred until lltype array/pointer carriers are represented in ConstValue",
            "op_getfield": "deferred until lltype struct/pointer carriers support immutable field reads in ConstValue",
            "op_getinteriorarraysize": "deferred until lltype interior-pointer carriers are represented in ConstValue",
            "op_getinteriorfield": "deferred until lltype interior-pointer carriers are represented in ConstValue",
            "op_getsubstruct": "deferred until lltype struct/pointer carriers are represented in ConstValue",
            "op_have_debug_prints": "side-effecting/debug operation intentionally omitted from the constfold registry",
            "op_is_group_member_nonzero": "deferred until llgroup GroupMemberOffset carriers are represented in ConstValue",
            "op_jit_ffi_save_result": "JIT bookkeeping operation intentionally omitted from the constfold registry",
            "op_jit_force_quasi_immutable": "JIT bookkeeping operation intentionally omitted from the constfold registry",
            "op_jit_force_virtual": "JIT bookkeeping operation intentionally omitted from the constfold registry",
            "op_jit_force_virtualizable": "JIT bookkeeping operation intentionally omitted from the constfold registry",
            "op_jit_is_virtual": "JIT bookkeeping operation intentionally omitted from the constfold registry",
            "op_jit_record_exact_class": "JIT bookkeeping operation intentionally omitted from the constfold registry",
            "op_ll_get_timestamp_unit": "runtime timestamp query intentionally omitted from the constfold registry",
            "op_ll_read_timestamp": "runtime timestamp query intentionally omitted from the constfold registry",
            "op_lllong_floordiv": "deferred until ConstValue carries r_longlonglong 128-bit integer values",
            "op_lllong_lshift": "deferred until ConstValue carries r_longlonglong 128-bit integer values",
            "op_lllong_mod": "deferred until ConstValue carries r_longlonglong 128-bit integer values",
            "op_lllong_rshift": "deferred until ConstValue carries r_longlonglong 128-bit integer values",
            "op_raw_store": "side-effecting memory store intentionally omitted from the constfold registry",
            "op_revdb_do_next_call": "reverse-debugger hook intentionally omitted from the constfold registry",
            "op_shrink_array": "side-effecting GC helper intentionally omitted from the constfold registry",
            "op_ulllong_lshift": "deferred until ConstValue carries r_ulonglonglong 128-bit integer values",
            "op_ulllong_rshift": "deferred until ConstValue carries r_ulonglonglong 128-bit integer values",
        },
    },
    ("rpython/rtyper/lltypesystem", "lltype"): {
        "types": {
            "ContainerType": "Rust folds the LowLevelType/ContainerType class hierarchy into LowLevelType variants plus container structs",
            "FuncForwardReference": "Rust folds ForwardReference/GcForwardReference/FuncForwardReference into ForwardReference._gckind",
            "GcArray": "Rust folds Array/GcArray into Array._gckind",
            "GcForwardReference": "Rust folds ForwardReference/GcForwardReference/FuncForwardReference into ForwardReference._gckind",
            "GcOpaqueType": "Rust folds OpaqueType/GcOpaqueType into OpaqueType._gckind",
            "GcStruct": "Rust folds Struct/RttiStruct/GcStruct into Struct._gckind plus optional runtime type info",
            "InvalidCast": "Rust surfaces invalid casts as typed Result errors instead of a standalone exception class",
            "Number": "Rust folds Primitive/Number singleton classes into LowLevelType primitive variants",
            "Primitive": "Rust folds Primitive/Number singleton classes into LowLevelType primitive variants",
            "RttiStruct": "Rust folds Struct/RttiStruct/GcStruct into Struct._gckind plus optional runtime type info",
            "WeakValueDictionary": "Rust container caches use explicit strong/weak carrier fields instead of upstream weakref subclass hooks",
        },
        "functions": {
            "saferecursive": "Python decorator helper is represented by Rust recursion guards at the equality/hash call sites",
        },
    },
    ("rpython/jit/metainterp/optimizeopt", "rewrite"): {
        "types": {
            "CallLoopinvariantOptimizationResult": "callback object is represented inline by the CALL_LOOPINVARIANT optimization arm updating loop_invariant_results/producers",
        },
    },
    ("rpython/jit/metainterp/optimizeopt", "pure"): {
        "types": {
            "CallPureOptimizationResult": "callback object is represented inline by OptPure's CALL_PURE demotion/emission path recording call_pure_positions",
            "DefaultOptimizationResult": "callback object is represented inline by OptPure optimize_default and OptimizationResult emission handling",
        },
    },
    ("rpython/jit/metainterp", "heapcache"): {
        "functions": {
            "add_flags": "requires HeapCache-owned heapc_flags storage in pyre; represented by HeapCache::_set_flag",
            "maybe_replace_with_const": "requires HeapCache-owned const replacement storage in pyre; represented by HeapCache::maybe_replace_with_const",
            "remove_flags": "requires HeapCache-owned heapc_flags storage in pyre; represented by HeapCache::_remove_flag",
            "test_flags": "requires HeapCache-owned heapc_flags storage in pyre; represented by HeapCache::_check_flag",
        },
    },
    ("rpython/flowspace", "operation"): {
        "types": {
            "CallArgs": "represented by OpKind::CallArgs and HLOperation dispatch rather than a standalone Rust subclass",
            "CallOp": "represented by OpKind call-family canraise logic instead of a standalone base class",
            "Contains": "represented by OpKind::Contains and unaryop contains registrations rather than a standalone Rust subclass",
            "DoubleDispatchMixin": "represented by OpKind::dispatch plus the crate-local _REGISTRY_DOUBLE table",
            "GetAttr": "represented by OpKind::GetAttr plus HLOperation::constfold_getattr",
            "HLOperationMeta": "represented by the static OpKind enum and registry tables instead of a runtime metaclass",
            "Iter": "represented by OpKind::Iter and HLOperation constfold/eval dispatch",
            "NewDict": "represented by OpKind::NewDict and HLOperation::consider",
            "NewList": "represented by OpKind::NewList and HLOperation::consider",
            "NewSlice": "represented by OpKind::NewSlice and HLOperation::consider",
            "NewTuple": "represented by OpKind::NewTuple, pyfunc tuple folding, and HLOperation::consider",
            "Next": "represented by OpKind::Next and flowcontext next handling",
            "OverflowingOperation": "represented by OpKind::can_overflow and OpKind::ovf_variant",
            "Pow": "represented by OpKind::Pow and pyfunc pow folding",
            "PureOperation": "represented by OpKind::pure and HLOperation::constfold",
            "PureOperation1": "represented by OpKind::arity plus HLOperation::constfold",
            "PureOperation2": "represented by OpKind::arity plus HLOperation::constfold",
            "SimpleCall": "represented by OpKind::SimpleCall and specialcase lookup in flowcontext",
            "SingleDispatchMixin": "represented by OpKind::dispatch plus the crate-local _REGISTRY_SINGLE table",
        },
        "functions": {
            "add_operator": "upstream dynamic class factory is represented by the static OpKind table",
            "delete": "represented by OpKind::Delete pyfunc/canraise metadata",
            "do_delslice": "represented by OpKind::DelSlice pyfunc/canraise metadata",
            "do_float": "represented by OpKind::Float pyfunc folding",
            "do_getslice": "represented by OpKind::GetSlice pyfunc/canraise metadata",
            "do_index": "represented by OpKind::Index pyfunc folding",
            "do_int": "represented by OpKind::Int pyfunc folding",
            "do_long": "represented by OpKind::Long pyfunc folding",
            "do_setslice": "represented by OpKind::SetSlice pyfunc/canraise metadata",
            "get": "represented by OpKind::Get pyfunc/canraise metadata",
            "inplace_add": "represented by OpKind::InplaceAdd metadata",
            "inplace_and": "represented by OpKind::InplaceAnd metadata",
            "inplace_div": "represented by OpKind::InplaceDiv metadata",
            "inplace_floordiv": "represented by OpKind::InplaceFloorDiv metadata",
            "inplace_lshift": "represented by OpKind::InplaceLShift metadata",
            "inplace_mod": "represented by OpKind::InplaceMod metadata",
            "inplace_mul": "represented by OpKind::InplaceMul metadata",
            "inplace_or": "represented by OpKind::InplaceOr metadata",
            "inplace_pow": "represented by OpKind::InplacePow metadata",
            "inplace_rshift": "represented by OpKind::InplaceRShift metadata",
            "inplace_sub": "represented by OpKind::InplaceSub metadata",
            "inplace_truediv": "represented by OpKind::InplaceTrueDiv metadata",
            "inplace_xor": "represented by OpKind::InplaceXor metadata",
            "new_style_type": "represented by OpKind::Type pyfunc folding",
            "next": "represented by OpKind::Next and flowcontext next handling",
            "set": "represented by OpKind::Set pyfunc/canraise metadata",
            "unsupported": "represented by OpKind::Format/Trunc/Buffer fallback metadata",
            "userdel": "represented by OpKind::UserDel metadata",
        },
    },
    ("rpython/flowspace", "specialcase"): {
        "types": {
            "StdOutBuffer": "Pyre records rpython_print_* as HostObject call targets and does not execute the print buffer in flowspace",
        },
        "functions": {
            "redirect_function": "Python import-time registry mutator is represented by the static SPECIAL_CASES LazyLock table",
            "register_flow_sc": "Python decorator registry mutator is represented by the static SPECIAL_CASES LazyLock table",
            "rpython_print_end": "represented as a HOST_ENV builtin call target emitted by FlowContext::handle_print_function",
            "rpython_print_item": "represented as a HOST_ENV builtin call target emitted by FlowContext::handle_print_function",
            "rpython_print_newline": "represented as a HOST_ENV builtin call target emitted by FlowContext::handle_print_function",
        },
    },
    ("rpython/jit/codewriter", "format"): {
        "functions": {
            "unformat_assembler": "reverse text-to-SSA test parser is deferred until Rust SSARepr stores parseable FlatOp operands instead of typed pipeline-only SpaceOperations",
        },
    },
    ("rpython/jit/codewriter", "jtransform"): {
        "types": {
            "NotSupported": "represented by Rust typed RewriteResult fallthroughs and explicit panic/error paths instead of a Python control-flow exception",
            "UnsupportedMallocFlags": "represented by Rust typed malloc lowering branches instead of a standalone Python exception class",
            "VirtualizableArrayField": "represented by Transformer.vable_array_vars metadata plus assertion diagnostics rather than a standalone exception payload",
        },
        "functions": {
            "constant_fold_ll_issubclass": "ported as constant_fold_ll_issubclass + constant_fold_ll_issubclass_flowgraph; identity is LowLevelFunction graph_key, not the helper name. Matcher is threaded from exceptiondata.fn_exception_match",
            "is_test_calldescr": "Rust tests use typed CallDescriptor values, not upstream's string/_for_tests_only calldescr sentinel",
        },
    },
    ("rpython/jit/codewriter", "support"): {
        "types": {
            "Entry": "RPython ExtRegistryEntry for maybe_on_top_of_llinterp; pyre does not specialize Python ExtRegistry entries",
            "Index": "represented by NormalizeSlot::Index rather than a standalone placeholder class",
            "LLtypeHelpers": "upstream helper-method namespace is flattened into CallControl fnaddr/oopspec registries in Rust",
        },
        "functions": {
            "annotate": "RPython test helper that builds annotator/rtyper graphs for a Python function; pyre translates Rust source through front/flowspace/annotator/rtyper instead",
            "autodetect_jit_markers_redvars": "upstream reds='auto' graph mutation over jit_marker ops; pyre carries JitDriver red/green metadata explicitly",
            "decode_hp_hint_args": "upstream decodes jit_marker SpaceOperation args; pyre markers are typed call targets lowered by jtransform",
            "get_call_oopspec_opargs": "folded into decode_builtin_call via CallControl oopspec registries and parse_oopspec/normalize_opargs",
            "get_gcid_oopspec": "gc_id op variant is not emitted by pyre's current typed graph pipeline",
            "get_identityhash_oopspec": "gc_identityhash op variant is not emitted by pyre's current typed graph pipeline",
            "get_send_oopspec": "OO/ADT send oopspec name synthesis has no pyre consumer; Rust call targets carry the resolved oopspec registry entry",
            "getargtypes": "RPython annotate() helper for Python value samples; pyre uses typed Rust source and lltype lowering instead",
            "getgraph": "RPython test helper built on annotate(); pyre graph construction is source-translation based",
            "maybe_on_top_of_llinterp": "RPython untranslated-test wrapper around LLInterpreter; pyre has no llinterp execution layer for generated graphs",
            "sort_vars": "upstream Variable sort by getkind for jit_marker arg lists; pyre keeps red/green groups typed by RegKind during lowering",
            "split_before_jit_merge_point": "upstream mutates flow blocks around jit_merge_point; pyre marker lowering does not split Python flowspace blocks this way",
            "u_to_longlong": "longlong helper is represented by Rust integer casts at concrete helper sites rather than a public support.py function",
        },
    },
    ("rpython/tool/algo", "graphlib"): {
        "functions": {
            "break_cycles": "upstream immediately skips this obsolete edge-cutting helper; pyre keeps break_cycles_v only",
            "show_graph": "GraphPage/DotGen GUI debug helper with no pyre translation-time consumer",
        },
    },
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def python_modules(path: Path, excludes: set[str]) -> set[str]:
    modules = set()
    for child in path.iterdir():
        if child.name in excludes:
            continue
        if child.is_file() and child.suffix == ".py":
            modules.add("mod" if child.stem == "__init__" else child.stem)
        elif child.is_dir() and (child / "__init__.py").is_file():
            modules.add(child.name)
    return modules


def rust_file_module_name(child: Path) -> str:
    if child.stem == "lib":
        return "mod"
    return child.stem


def rust_modules(path: Path, excludes: set[str]) -> set[str]:
    modules = set()
    for child in path.iterdir():
        if child.name in excludes:
            continue
        if child.is_file() and child.suffix == ".rs":
            modules.add(rust_file_module_name(child))
        elif child.is_dir() and (child / "mod.rs").is_file():
            modules.add(child.name)
    return modules


def python_module_path(path: Path, module: str) -> Path:
    if module == PACKAGE_ENTRY:
        return path / "__init__.py"
    file_path = path / f"{module}.py"
    if file_path.is_file():
        return file_path
    return path / module / "__init__.py"


def rust_module_path(path: Path, module: str) -> Path:
    if module == PACKAGE_ENTRY:
        lib_path = path / "lib.rs"
        if lib_path.is_file():
            return lib_path
        return path / "mod.rs"
    file_path = path / f"{module}.rs"
    if file_path.is_file():
        return file_path
    return path / module / "mod.rs"


PYTHON_TOP_LEVEL_SYMBOL = re.compile(r"^(class|def)\s+([A-Za-z_][A-Za-z0-9_]*)\b")
PYTHON_BLOCK_START = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\b")
PYTHON_MODULE_CONTROL_BLOCKS = {
    "else",
    "elif",
    "except",
    "finally",
    "for",
    "if",
    "try",
    "while",
    "with",
}


def python_top_level_symbols(path: Path) -> dict[str, set[str]]:
    symbols = {"types": set(), "functions": set()}
    block_stack: list[tuple[int, str]] = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()

        while block_stack and indent <= block_stack[-1][0]:
            block_stack.pop()

        in_class_or_def = any(kind in {"class", "def"} for _, kind in block_stack)
        symbol_match = PYTHON_TOP_LEVEL_SYMBOL.match(stripped)
        if symbol_match:
            kind, name = symbol_match.groups()
            if not in_class_or_def and not name.startswith("_"):
                if kind == "class":
                    symbols["types"].add(name)
                else:
                    symbols["functions"].add(name)
            # A `def`/`class` block can span multiple physical lines, e.g.
            # `def to_optparse(...,\n                extra_usage=None):`.
            # Treat it as a block immediately so nested helpers in the body do
            # not get misclassified as module-level symbols.
            block_stack.append((indent, kind))
            continue

        block_match = PYTHON_BLOCK_START.match(stripped)
        if (
            block_match
            and stripped.endswith(":")
            and block_match.group(1) in PYTHON_MODULE_CONTROL_BLOCKS
        ):
            block_stack.append((indent, "control"))
    return symbols


RUST_PUB_ITEM = re.compile(
    r"^pub\s+(?:unsafe\s+)?(?:extern\s+(?:\"[^\"]+\"\s+)?)?"
    r"(struct|enum|trait|type|fn)\s+(?:r#)?([A-Za-z_][A-Za-z0-9_]*)\b"
)
RUST_TOP_LEVEL_ITEM = re.compile(
    r"^(?:pub(?:\([^)]*\))?\s+)?(?:unsafe\s+)?(?:extern\s+(?:\"[^\"]+\"\s+)?)?"
    r"(struct|enum|trait|type|fn)\s+(?:r#)?([A-Za-z_][A-Za-z0-9_]*)\b"
)
RUST_PUB_REEXPORT = re.compile(r"^pub\s+use\s+")
RUST_ITEM_START = re.compile(
    r"^(?:pub(?:\([^)]*\))?\s+)?(?:unsafe\s+)?(?:extern\s+(?:\"[^\"]+\"\s+)?)?"
    r"(struct|enum|trait|type|fn|const|static|impl|mod)\b"
)
RUST_TYPE_MACRO_INVOCATION = re.compile(
    r"^(?P<macro>[A-Za-z_][A-Za-z0-9_]*)!\s*\(\s*(?:r#)?"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?:,|\))"
)
RUST_TYPE_MACRO_NAMES = {"binop_struct", "gc_class"}


def _strip_rust_line(line: str) -> str:
    line = line.strip()
    if line.startswith("//"):
        return ""
    if line.startswith("#["):
        return ""
    return line


def _split_top_level_commas(text: str) -> list[str]:
    parts = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if char == "{":
            depth += 1
        elif char == "}":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _extract_rust_reexport_names(statement: str) -> set[str]:
    statement = statement.strip().removesuffix(";").strip()
    if not statement.startswith("pub use "):
        return set()
    path = statement[len("pub use ") :].strip()
    if "*" in path:
        return set()
    if "{" not in path:
        leaf = path.rsplit("::", 1)[-1].strip()
        if " as " in leaf:
            leaf = leaf.rsplit(" as ", 1)[-1].strip()
        return {leaf} if leaf and leaf not in {"crate", "self", "super"} else set()

    start = path.find("{")
    end = path.rfind("}")
    if end < start:
        return set()
    names = set()
    for item in _split_top_level_commas(path[start + 1 : end]):
        if not item:
            continue
        if "{" in item:
            names.update(_extract_rust_reexport_names(f"pub use {item};"))
            continue
        if " as " in item:
            item = item.rsplit(" as ", 1)[-1].strip()
        elif "::" in item:
            item = item.rsplit("::", 1)[-1].strip()
        if item and item not in {"crate", "self", "super"}:
            names.add(item)
    return names


def rust_top_level_symbols(
    path: Path,
) -> tuple[dict[str, set[str]], dict[str, set[str]], set[str], bool]:
    symbols = {"types": set(), "functions": set()}
    nonpub_symbols = {"types": set(), "functions": set()}
    reexports: set[str] = set()
    has_pub_reexport = False
    has_direct_item = False
    depth = 0
    in_block_comment = False
    reexport_lines: list[str] | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line
        if in_block_comment:
            if "*/" in line:
                line = line.split("*/", 1)[1]
                in_block_comment = False
            else:
                continue
        while "/*" in line:
            before, after = line.split("/*", 1)
            if "*/" in after:
                after = after.split("*/", 1)[1]
                line = before + after
            else:
                line = before
                in_block_comment = True
                break

        candidate = _strip_rust_line(line)
        if reexport_lines is not None:
            if candidate:
                reexport_lines.append(candidate)
            if ";" in candidate:
                statement = " ".join(reexport_lines)
                reexports.update(_extract_rust_reexport_names(statement))
                reexport_lines = None
            continue

        if depth == 0 and candidate:
            pub_match = RUST_PUB_ITEM.match(candidate)
            if pub_match:
                kind = pub_match.group(1)
                bucket = "functions" if kind == "fn" else "types"
                symbols[bucket].add(pub_match.group(2))
                has_direct_item = True
            elif item_match := RUST_TOP_LEVEL_ITEM.match(candidate):
                kind = item_match.group(1)
                bucket = "functions" if kind == "fn" else "types"
                nonpub_symbols[bucket].add(item_match.group(2))
                has_direct_item = True
            elif RUST_PUB_REEXPORT.match(candidate):
                has_pub_reexport = True
                if ";" in candidate:
                    reexports.update(_extract_rust_reexport_names(candidate))
                else:
                    reexport_lines = [candidate]
                continue
            elif macro_match := RUST_TYPE_MACRO_INVOCATION.match(candidate):
                if macro_match.group("macro") in RUST_TYPE_MACRO_NAMES:
                    symbols["types"].add(macro_match.group("name"))
                    has_direct_item = True
            elif RUST_ITEM_START.match(candidate):
                if not re.match(r"mod\s+tests\b", candidate):
                    has_direct_item = True

        depth += line.count("{") - line.count("}")
        if depth < 0:
            depth = 0

    return symbols, nonpub_symbols, reexports, has_pub_reexport and not has_direct_item and not reexports


def _strings_from_ast_collection(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "set":
        if len(node.args) != 1 or node.keywords:
            raise ValueError("expected set([...]) with one positional argument")
        node = node.args[0]
    if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        raise ValueError("expected a list, tuple, or set literal")

    values = set()
    for item in node.elts:
        if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
            raise ValueError("expected string-only collection literal")
        values.add(item.value)
    return values


def python_string_set(path: Path, symbol: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == symbol:
                    return _strings_from_ast_collection(node.value)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == symbol
            and node.value is not None
        ):
            return _strings_from_ast_collection(node.value)
    raise ValueError(f"{symbol} not found in {path}")


def _rust_function_body(text: str, function: str) -> str:
    match = re.search(rf"\bfn\s+{re.escape(function)}\s*\([^)]*\)\s*->\s*bool\s*\{{", text)
    if not match:
        raise ValueError(f"{function} function not found")
    start = match.end() - 1
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index]
    raise ValueError(f"{function} body is not closed")


def rust_string_literals_in_bool_function(path: Path, function: str) -> set[str]:
    body = _rust_function_body(path.read_text(encoding="utf-8"), function)
    strings = set()
    for match in re.finditer(r'"(?:\\.|[^"\\])*"', body):
        value = ast.literal_eval(match.group(0))
        if isinstance(value, str):
            strings.add(value)
    return strings


def compare_string_sets(root: Path, pairs: list[StringSetPair]) -> list[dict[str, object]]:
    results = []
    for pair in pairs:
        py_path = root / pair.python_path
        rs_path = root / pair.rust_path
        py_values = python_string_set(py_path, pair.python_symbol)
        rs_values = rust_string_literals_in_bool_function(rs_path, pair.rust_function)
        results.append(
            {
                "label": pair.label,
                "python_path": pair.python_path.as_posix(),
                "python_symbol": pair.python_symbol,
                "rust_path": pair.rust_path.as_posix(),
                "rust_function": pair.rust_function,
                "matched": sorted(py_values & rs_values),
                "missing_in_rust": sorted(py_values - rs_values),
                "extra_in_rust": sorted(rs_values - py_values),
            }
        )
    return results


def compare_symbols_for_pair(
    root: Path, pair: ModulePair, matched: list[str]
) -> list[dict[str, object]]:
    python_dir = root / pair.python_dir
    rust_dirs = [root / pair.rust_dir, *(root / extra for extra in pair.extra_rust_dirs)]
    results = []

    for module in matched:
        if module == PACKAGE_ENTRY:
            continue
        py_path = python_module_path(python_dir, module)
        # A module split across crates keeps a file of the same name in each;
        # its symbols are the union, reported under the first one found.
        rs_paths = [
            path
            for path in (rust_module_path(rust_dir, module) for rust_dir in rust_dirs)
            if path.is_file()
        ]
        if not py_path.is_file() or not rs_paths:
            continue
        rs_path = rs_paths[0]

        py_symbols = python_top_level_symbols(py_path)
        rs_symbols = {"types": set(), "functions": set()}
        rs_nonpub_symbols = {"types": set(), "functions": set()}
        rs_reexports: set[str] = set()
        is_reexport = True
        for path in rs_paths:
            symbols, nonpub_symbols, reexports, path_is_reexport = rust_top_level_symbols(path)
            for kind in rs_symbols:
                rs_symbols[kind] |= symbols[kind]
                rs_nonpub_symbols[kind] |= nonpub_symbols[kind]
            rs_reexports |= reexports
            is_reexport = is_reexport and path_is_reexport
        rs_type_names = rs_symbols["types"] | rs_reexports
        rs_function_names = rs_symbols["functions"] | rs_reexports
        rs_implemented_function_names = rs_function_names | rs_nonpub_symbols["functions"]
        raw_missing_types = py_symbols["types"] - rs_type_names
        raw_missing_functions = py_symbols["functions"] - rs_implemented_function_names
        raw_extra_types = rs_symbols["types"] - py_symbols["types"]
        raw_extra_functions = rs_symbols["functions"] - py_symbols["functions"]
        implemented_private_functions = (
            py_symbols["functions"] & rs_nonpub_symbols["functions"] - rs_function_names
        )
        intentional_missing = INTENTIONAL_SYMBOL_MISSING.get((pair.label, module), {})
        intentional_extra = INTENTIONAL_SYMBOL_EXTRA.get((pair.label, module), {})
        ignored_missing_types = {
            name: reason
            for name, reason in intentional_missing.get("types", {}).items()
            if name in raw_missing_types
        }
        ignored_missing_functions = {
            name: reason
            for name, reason in intentional_missing.get("functions", {}).items()
            if name in raw_missing_functions
        }
        ignored_extra_types = {
            name: reason
            for name, reason in intentional_extra.get("types", {}).items()
            if name in raw_extra_types
        }
        ignored_extra_functions = {
            name: reason
            for name, reason in intentional_extra.get("functions", {}).items()
            if name in raw_extra_functions
        }
        result = {
            "module": module,
            "python_path": py_path.relative_to(root).as_posix(),
            "rust_path": rs_path.relative_to(root).as_posix(),
            "types": {
                "matched": sorted(py_symbols["types"] & rs_type_names),
                "missing": sorted(raw_missing_types - ignored_missing_types.keys()),
                "ignored_missing": dict(sorted(ignored_missing_types.items())),
                "extra": sorted(raw_extra_types - ignored_extra_types.keys()),
                "ignored_extra": dict(sorted(ignored_extra_types.items())),
            },
            "functions": {
                "matched": sorted(py_symbols["functions"] & rs_function_names),
                "implemented_private": sorted(implemented_private_functions),
                "missing": sorted(raw_missing_functions - ignored_missing_functions.keys()),
                "ignored_missing": dict(sorted(ignored_missing_functions.items())),
                "extra": sorted(raw_extra_functions - ignored_extra_functions.keys()),
                "ignored_extra": dict(sorted(ignored_extra_functions.items())),
            },
            "skipped_reexport": is_reexport,
        }
        results.append(result)
    return results


def compare_pair(root: Path, pair: ModulePair, excludes: set[str]) -> dict[str, object]:
    python_dir = root / pair.python_dir
    rust_dir = root / pair.rust_dir
    if not python_dir.is_dir():
        raise SystemExit(f"missing Python directory: {pair.python_dir}")
    if not rust_dir.is_dir():
        raise SystemExit(f"missing Rust directory: {pair.rust_dir}")

    py_modules = python_modules(python_dir, excludes)
    rs_modules = rust_modules(rust_dir, excludes)
    for extra in pair.extra_rust_dirs:
        if not (root / extra).is_dir():
            raise SystemExit(f"missing Rust directory: {extra}")
        rs_modules |= rust_modules(root / extra, excludes)
    raw_missing = py_modules - rs_modules
    raw_extra = rs_modules - py_modules
    ignored_missing = {
        name: reason
        for name, reason in INTENTIONAL_MISSING.get(pair.label, {}).items()
        if name in raw_missing
    }
    ignored_extra = {
        name: reason
        for name, reason in INTENTIONAL_EXTRA.get(pair.label, {}).items()
        if name in raw_extra
    }
    missing = sorted(raw_missing - ignored_missing.keys())
    extra = sorted(raw_extra - ignored_extra.keys())
    matched = sorted(py_modules & rs_modules)
    return {
        "label": pair.label,
        "python_dir": pair.python_dir.as_posix(),
        "rust_dir": pair.rust_dir.as_posix(),
        "matched": matched,
        "missing": missing,
        "extra": extra,
        "ignored_missing": dict(sorted(ignored_missing.items())),
        "ignored_extra": dict(sorted(ignored_extra.items())),
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare immediate RPython/PyPy module names with their Rust "
            "port directories."
        )
    )
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="include Python test packages in module comparison",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON instead of text",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when any missing or extra module is found",
    )
    parser.add_argument(
        "--symbols",
        action="store_true",
        help="also compare top-level class/function names with Rust pub item names",
    )
    parser.add_argument(
        "--strict-symbols",
        action="store_true",
        help="exit non-zero when --symbols finds any non-reexport symbol gap",
    )
    parser.add_argument(
        "--jit-strings",
        action="store_true",
        help="also compare selected JIT/codewriter string-name tables",
    )
    parser.add_argument(
        "--strict-jit-strings",
        action="store_true",
        help="exit non-zero when --jit-strings finds a string-name gap",
    )
    return parser.parse_args(argv)


def print_text(results: list[dict[str, object]], show_symbols: bool) -> None:
    for result in results:
        print(f"## {result['label']} -> {result['rust_dir']}")
        missing = result["missing"]
        extra = result["extra"]
        if missing:
            print("missing: " + ", ".join(missing))
        else:
            print("missing: <none>")
        if extra:
            print("extra: " + ", ".join(extra))
        else:
            print("extra: <none>")
        ignored_missing = result["ignored_missing"]
        ignored_extra = result["ignored_extra"]
        if ignored_missing:
            print(
                "ignored missing: "
                + "; ".join(f"{name} ({reason})" for name, reason in ignored_missing.items())
            )
        if ignored_extra:
            print(
                "ignored extra: "
                + "; ".join(f"{name} ({reason})" for name, reason in ignored_extra.items())
            )
        if show_symbols:
            symbol_results = result["symbols"]
            symbol_gaps = [
                item
                for item in symbol_results
                if item["types"]["missing"]
                or item["types"]["extra"]
                or item["types"]["ignored_missing"]
                or item["types"]["ignored_extra"]
                or item["functions"]["missing"]
                or item["functions"]["extra"]
                or item["functions"]["implemented_private"]
                or item["functions"]["ignored_missing"]
                or item["functions"]["ignored_extra"]
                or item["skipped_reexport"]
            ]
            if not symbol_gaps:
                print("symbols: <none>")
            else:
                print("symbols:")
                for item in symbol_gaps:
                    if item["skipped_reexport"]:
                        print(
                            f"  {item['module']}: skipped reexport wrapper "
                            f"({item['rust_path']})"
                        )
                    else:
                        details = []
                        if item["types"]["missing"]:
                            details.append(
                                "missing types " + ", ".join(item["types"]["missing"])
                            )
                        if item["types"]["ignored_missing"]:
                            details.append(
                                "ignored missing types "
                                + "; ".join(
                                    f"{name} ({reason})"
                                    for name, reason in item["types"][
                                        "ignored_missing"
                                    ].items()
                                )
                            )
                        if item["types"]["extra"]:
                            details.append(
                                "extra types " + ", ".join(item["types"]["extra"])
                            )
                        if item["types"]["ignored_extra"]:
                            details.append(
                                "ignored extra types "
                                + "; ".join(
                                    f"{name} ({reason})"
                                    for name, reason in item["types"][
                                        "ignored_extra"
                                    ].items()
                                )
                            )
                        if item["functions"]["missing"]:
                            details.append(
                                "missing functions "
                                + ", ".join(item["functions"]["missing"])
                            )
                        if item["functions"]["implemented_private"]:
                            details.append(
                                "implemented private functions "
                                + ", ".join(item["functions"]["implemented_private"])
                            )
                        if item["functions"]["ignored_missing"]:
                            details.append(
                                "ignored missing functions "
                                + "; ".join(
                                    f"{name} ({reason})"
                                    for name, reason in item["functions"][
                                        "ignored_missing"
                                    ].items()
                                )
                            )
                        if item["functions"]["extra"]:
                            details.append(
                                "extra functions "
                                + ", ".join(item["functions"]["extra"])
                            )
                        if item["functions"]["ignored_extra"]:
                            details.append(
                                "ignored extra functions "
                                + "; ".join(
                                    f"{name} ({reason})"
                                    for name, reason in item["functions"][
                                        "ignored_extra"
                                    ].items()
                                )
                            )
                        print(f"  {item['module']}: " + "; ".join(details))
        print()


def print_string_set_text(results: list[dict[str, object]]) -> None:
    print("## JIT string parity")
    for result in results:
        details = []
        if result["missing_in_rust"]:
            details.append("missing in Rust " + ", ".join(result["missing_in_rust"]))
        if result["extra_in_rust"]:
            details.append("extra in Rust " + ", ".join(result["extra_in_rust"]))
        if not details:
            details.append("<none>")
        print(f"{result['label']}: " + "; ".join(details))
    print()


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    root = repo_root()
    excludes = set(DEFAULT_EXCLUDES)
    if args.include_tests:
        excludes.discard("test")

    results = [compare_pair(root, pair, excludes) for pair in DEFAULT_PAIRS]
    show_symbols = args.symbols or args.strict_symbols
    show_jit_strings = args.jit_strings or args.strict_jit_strings
    if show_symbols:
        for pair, result in zip(DEFAULT_PAIRS, results):
            result["symbols"] = compare_symbols_for_pair(root, pair, result["matched"])
    string_set_results = (
        compare_string_sets(root, DEFAULT_STRING_SET_PAIRS) if show_jit_strings else []
    )
    if args.json:
        if show_jit_strings:
            payload = {"modules": results, "jit_strings": string_set_results}
        else:
            payload = results
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print_text(results, show_symbols)
        if show_jit_strings:
            print_string_set_text(string_set_results)

    has_gap = any(result["missing"] or result["extra"] for result in results)
    has_symbol_gap = False
    if show_symbols:
        has_symbol_gap = any(
            (
                item["types"]["missing"]
                or item["types"]["extra"]
                or item["functions"]["missing"]
                or item["functions"]["extra"]
            )
            and not item["skipped_reexport"]
            for result in results
            for item in result["symbols"]
        )
    has_string_set_gap = any(
        result["missing_in_rust"] or result["extra_in_rust"]
        for result in string_set_results
    )
    if args.strict and has_gap:
        return 1
    if args.strict_symbols and has_symbol_gap:
        return 1
    if args.strict_jit_strings and has_string_set_gap:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
