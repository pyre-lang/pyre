//! Narrow semantic graph scaffold for the future graph-based translator.
//!
//! This is intentionally much smaller than a full Rust compiler IR.  It exists
//! to model only the semantics needed by majit's translation/codewriter layer.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;

use crate::flowspace::model::ConstValue;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockId(pub usize);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ValueType {
    Int,
    /// `lltype.Unsigned` — register class is `'int'` per
    /// `getkind(Unsigned) == 'int'`, distinct from `Signed` only at
    /// the rtyper dispatch level (`rbool.py uint_is_true`,
    /// `rint.py` cast family).  Downstream consumers that do not
    /// distinguish signedness (regalloc, codewriter,
    /// valuetype_to_someshell) treat `Unsigned` identically to `Int`
    /// via `Int | Unsigned` arms.
    ///
    /// Produced by the MIR front-end (`front::mir`) for Rust
    /// `u8`/`u16`/`u32`/`u64`/`usize` typed args.  Cast routing
    /// through `simple_call(__builtin__.float/bool, v_uint)`,
    /// `simple_call(rarithmetic.intmask, v_uint)`, and
    /// `simple_call(rarithmetic.r_uint, v)` lives in `front::mir`'s
    /// `Cast` lowering per
    /// `rbuiltin.py:178-189` / `rbuiltin.py:220-225` /
    /// `rarithmetic.py`.
    Unsigned,
    /// RPython `lltype.SingleFloat` — `rffi.r_singlefloat`, a Rust `f32`.
    ///
    /// Distinct from both [`ValueType::Int`] and [`ValueType::Float`].
    /// `history.getkind` banks it as `'int'` ("singlefloats are stored in
    /// an int") but only when the CPU supports singlefloats; the base
    /// backend does not (`backend/model.py:20`), and RPython gives it no
    /// arithmetic at all — `rffi` converts through `cast_singlefloat_to_float`
    /// before computing and back afterwards, so the rtyper never emits a
    /// float-kind operation over a SingleFloat operand.
    ///
    /// It is spelled here so the codewriter policy can refuse a graph that
    /// carries one, rather than collapsing into `Int` and letting Rust's
    /// native `f32` arithmetic lower to an integer operation over the
    /// float's bit pattern.
    SingleFloat,
    /// RPython `r_longlonglong` / `lltype.SignedLongLongLong`.
    ///
    /// This is a translation-level 128-bit primitive used by
    /// `rbigint` when `SHIFT == 63`. It deliberately remains distinct
    /// from the word-sized `Int`: PyPy's source translator and rtyper
    /// preserve it, while `history.getkind` rejects it at the JIT
    /// codewriter boundary because it occupies 16 bytes.
    Int128,
    /// RPython `r_ulonglonglong` / `lltype.UnsignedLongLongLong`.
    UInt128,
    /// RPython `SomeBool` (`annotator/model.py`): a Python `bool`
    /// at the annotator level.  Distinct from `Int` (RPython `SomeInteger`,
    /// `:200-264`) because PyPy's flowspace `UNARY_NOT` (`flowcontext.py`,
    /// `op.bool` then `guessbool` fork) and `UNARY_INVERT` (`:188-191`,
    /// `op.invert`) dispatch on which one is on the stack:
    /// `not` for booleans, `~` for integers.
    ///
    /// Every comparison opname (`lt`/`le`/`eq`/`ne`/`gt`/`ge`,
    /// `operation.py add_operator(.., dispatch=2, pure=True)`),
    /// `is`/`is not`, the unary `bool` op, and Rust `bool`-typed locals
    /// produce `Bool`.
    ///
    /// Bool literals lower in `front::mir` to the dedicated
    /// `OpKind::ConstBool(bool)` variant; the rtyper
    /// adapter lifts it to `Constant(True/False, lltype.Bool)` so the
    /// annotator selects `SomeBool` and the rtyper picks `BoolRepr`.
    /// The codewriter collapses storage to the int kind via
    /// `getkind(Bool) == 'int'`
    /// (`rpython/jit/codewriter/flatten.py:getkind`); the assembler
    /// emits `int_copy/i>i` for both `ConstInt` and `ConstBool`.
    ///
    /// Pyre's annotator-side `valuetype_to_someshell`
    /// (`annotation_state.rs::valuetype_to_someshell`) lifts `Bool`
    /// to `SomeBool(unspecified)`.  The rtyper has `BoolRepr`
    /// available (`rmodel.rs::BoolRepr`); the high-level dispatch in
    /// `RPythonTyper::translate_op` continues to share rint paths for
    /// arithmetic operations because RPython's `BoolRepr.lowleveltype
    /// = Bool` is integer-compatible at the LL level (everything
    /// except logical-vs-bitwise distinguishes).
    Bool,
    /// `lltype.Ptr(<inner>)` — pointer / GC reference family.  The
    /// optional `String` carries the Rust type-root identifier
    /// (the leading path segment with generic arguments stripped) for
    /// the pointee when the producer knows it; `None` is the
    /// un-narrowed / opaque case
    /// (mirrors `lltype.Ptr(<unspecified>)` upstream where the rtyper
    /// resolves through `getinstancerepr(rtyper, None, Gc)` to the
    /// abstract `object`-root `InstanceRepr`).
    ///
    /// `valuetype_to_someshell::Ref` intentionally projects every
    /// `Ref(_)` fallback to `SomeInstance(classdef=None)`.  Typed
    /// pointer precision must be attached before that fallback by a
    /// producer that has the real lltype object in hand, e.g. the
    /// Rust-source walker writing
    /// `SomeValue::Ptr(SomePtr::new(<ll_ptrtype>))` directly into
    /// `Variable.annotation`.  Reconstructing `SomePtr` later from this
    /// root string would lose RPython's module/global context and
    /// object-identity-based lltype semantics.
    Ref(Option<String>),
    Float,
    /// RPython `SomeString` — a Rust `str`/`String`/`Wtf8`/`Wtf8Buf`
    /// value (`tyref_is_string_value`).  Distinct from `Ref` so a
    /// string-typed struct field seeds a `SomeString` shell rather than
    /// the classdef-less `SomeInstance(None)` the `Ref` fallback yields.
    Str,
    /// RPython `SomeStringBuilder` (`rlib/rstring.py`) — the annotator
    /// shell for a `StringBuilder()` value.  Distinct from `Ref` so the
    /// annotator seeds a `SomeValue::StringBuilder` shell that the rtyper
    /// binds to `StringBuilderRepr` (`lltypesystem/rbuilder.rs`), lowering
    /// to `Ptr(STRINGBUILDER)`.  `getkind(Ptr(Gc)) == 'ref'` places it in
    /// the GC-ref register bank alongside every `Ref` / `Str`.
    StringBuilder,
    Void,
    State,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnknownKind {
    /// `syn::Stmt::Macro` — any macro invocation at statement position
    /// (`panic!(...)`, `debug_assert!(...)`, `println!(...)`, etc).
    /// Always skipped in `lower_stmt` (no flow-graph analogue — RPython
    /// source has no macros).
    MacroStmt,
    /// `syn::Expr::Lit` with a kind pyre cannot yet model.  The `variant`
    /// tag names the specific syn literal kind (`Str`, `Float`,
    /// `ByteStr`, `Verbatim`) so downstream diagnostics show the exact
    /// failure category without re-walking the syn AST. (An earlier
    /// `MAJIT_UNKNOWN_DUMP` dump consumed this too; nothing reads that
    /// gate now.)
    UnsupportedLiteral { variant: UnsupportedLiteralKind },
    /// `syn::Expr::*` variants pyre cannot yet lower.  Tag names the
    /// specific expression kind so diagnostics and the `Unknown`
    /// markers left in SSA graphs identify the remaining port gap.
    UnsupportedExpr { variant: UnsupportedExprKind },
}

/// Reason an `syn::Lit::*` could not be lowered.  Parity with RPython
/// `flowspace/flowcontext.py` — unsupported literals raise
/// `FlowingError` with a kind-specific message; pyre's analogue
/// records the kind on the Unknown marker op so the abort path is
/// traceable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnsupportedLiteralKind {
    Str,
    Float,
    ByteStr,
    Verbatim,
    Other,
}

/// Reason an `syn::Expr::*` could not be lowered.  Same role as
/// `UnsupportedLiteralKind` — tags the specific `FlowingError`-
/// equivalent abort cause.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnsupportedExprKind {
    Array,
    Async,
    Await,
    Const,
    ForLoop,
    Group,
    Infer,
    Let,
    Macro,
    Range,
    RawAddr,
    Repeat,
    Struct,
    Tuple,
    TryBlock,
    Verbatim,
    Yield,
    /// Rust `!x` whose operand has no statically-known bool/int type.
    /// RPython distinguishes UNARY_NOT (`flowcontext.py`) from
    /// UNARY_INVERT (`flowcontext.py`) at the bytecode token,
    /// so an Unknown operand cannot pick the parity-correct flowspace
    /// op without guessing.
    UnaryNotUnknownOperand,
    /// Multi-segment `Expr::Path` (e.g. `Into::into`, `Foo::bar`) used
    /// as a value rather than as a call target.  RPython has no analogue
    /// — `flowcontext.py:LOAD_GLOBAL` lifts a name lookup to a constant
    /// (registered `FuncDesc` / class) or raises `FlowingError`; pyre
    /// has no `Const(PBCRef)` IR variant for a bare path-as-value, so
    /// classifying these explicitly as `Abort` keeps the producer site
    /// distinct from the "cross-block local was not threaded" category
    /// (the prior fall-through emitted naked `OpKind::Input { name:
    /// "Foo::bar" }` and tripped `adapter cross-block body Input`).
    PathConstantRef,
    /// Rust `|args| body` closure expression used as a value (closure
    /// capture / function-pointer argument).  RPython has no closure
    /// model in flowspace — `flowcontext.py MAKE_FUNCTION` raises
    /// `FlowingError` for nested-fn / lambda surfaces (only top-level
    /// `FunctionGraph`s round-trip through the bookkeeper).  Pyre's
    /// `syn::Expr::Closure` arm produces an `Unknown` marker for the
    /// closure value rather than walking the body — inlining it would
    /// be a deviation (treats the closure as a synchronous block).
    /// Classified separately from `OtherExpr` so the producer site is
    /// distinguishable from the catch-all in dual-gate diagnostics.
    Closure,
    /// Not a `syn::Expr` variant: the embedding host explicitly declined to
    /// name a helper required to lower an otherwise recognized operation.
    /// It is carried in this family because `OpKind::Abort` accepts an
    /// `UnknownKind`, whose only structured reason payload is
    /// `UnsupportedExpr`.
    HostHelperRefused,
    OtherExpr,
}

pub use majit_jitcode::rclass::ImmutableRank;

/// Payload banks of one `Result::branch` call.
///
/// `ok` / `err` are the Result variant fields. `continue_ty` / `break_ty`
/// are the ControlFlow variant fields. `None` is a void payload and is
/// not read or written.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ResultBranchPayloads {
    pub ok: Option<ValueType>,
    pub err: Option<ValueType>,
    pub continue_ty: Option<ValueType>,
    pub break_ty: Option<ValueType>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CallTarget {
    Method {
        name: String,
        receiver_root: Option<String>,
        /// Resolved call path stamped at codewriter time by
        /// `stamp_classdef_hints_on_graph`. This is a synthesized
        /// [`crate::parse::CallPath`] (re-derived from the receiver's
        /// `SomeInstance.classdef` → `MethodDesc.selfclassdef` → impl
        /// type), NOT a graph-object pointer. It is pyre's surrogate for
        /// `call.py:29`'s graph-object dict key: the consumer does
        /// `function_graphs.get(&path)` directly, the `CallPath`-keyed
        /// stand-in for `getfunctionptr(graph)` (call.py). Transient
        /// (`#[serde(skip)]`) — never reaches codegen / serde.
        #[serde(skip, default)]
        resolved_path: Option<crate::parse::CallPath>,
        /// Caller's LLBC-local `FunDecl.def_id`. Unique only inside that
        /// artefact; `graphs_from` / `target_to_path` resolve by
        /// [`resolved_path`] / the FunctionPath segments the decl's
        /// `name_path` maps to, never by this number. Transient
        /// (`#[serde(skip)]`).
        #[serde(skip, default)]
        fun_decl_id: Option<u64>,
        /// Set only on `Result::branch`. Resolved from the two type decls.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        branch_payloads: Option<ResultBranchPayloads>,
    },
    FunctionPath {
        segments: Vec<String>,
        /// Caller's LLBC-local `FunDecl.def_id`. See [`Self::Method`].
        #[serde(skip, default)]
        fun_decl_id: Option<u64>,
    },
    /// Rust frontend adaptation for constructors that RPython's rtyper erases
    /// before jtransform. This variant must only be produced after frontend
    /// resolution proves the call is not a user-defined function.
    ///
    /// `name` is the variant/ctor leaf (`"Continue"` for
    /// `StepResult::Continue`).  `owner_path` is the qualifying segments
    /// preceding the leaf (`["StepResult"]`), empty for ctors that have
    /// no owner (`Ok`/`Err`/`Some`).  Owner path matters for downstream
    /// identity: `HostObject::new_class(qualname)` keys on the joined
    /// path, so `StepResult::Continue` and `JitAction::Continue` produce
    /// distinct ClassDescs and don't collide on the bare leaf — matching
    /// RPython `bookkeeper.py getdesc(pyobj)` which dedupes by
    /// object identity, not leaf string.
    SyntheticTransparentCtor {
        name: String,
        #[serde(default)]
        owner_path: Vec<String>,
        /// `true` when the constructed ADT is a `struct`, whose value is
        /// described entirely by the field writes that follow it.
        ///
        /// A sum-type variant constructor sets `false`.  Charon does not spell
        /// the enum tag as a MIR operand, so a rewrite that turns the
        /// constructor into a bare allocation would drop it, and only `Result`
        /// and `Option` — whose variant order the language fixes — can have
        /// one stamped back on ([`sum_variant_ctor`]).  `false` is therefore
        /// the safe answer and the one every constructor that cannot prove
        /// otherwise records: a consumer that reads this decides whether to
        /// allocate, and dropping a tag is silent.
        ///
        /// [`sum_variant_ctor`]: crate::codewriter::jtransform
        #[serde(default)]
        is_struct: bool,
        /// The variant's declaration index (Charon `AggregateKind::Adt`
        /// `variant_idx`), recorded where the frontend resolved a
        /// `TypeDeclKind::Enum` variant.  A payload-less variant folded
        /// to a prebuilt singleton carries its discriminant from here to
        /// runtime materialization
        /// (`JitCodeBody::unit_variant_consts`).  `None` for struct
        /// ctors, positional aggregates, and hand-built targets.
        #[serde(default)]
        variant_tag: Option<i64>,
    },
    /// RPython: `indirect_call` opname. Receiver's static type is a
    /// `dyn Trait` (Rust fat pointer); at JIT time the actual callee
    /// is resolved via vtable.  `trait_root` + `method_name` together
    /// key the candidate family in `CallControl.trait_method_impls`.
    Indirect {
        trait_root: String,
        method_name: String,
    },
    UnsupportedExpr,
}

pub const FN_CONST_HEAD: &str = "__fn_const";

pub fn fn_const_segments(target: &CallTarget) -> Option<&[String]> {
    match target {
        CallTarget::FunctionPath { segments, .. }
            if segments.first().map(String::as_str) == Some(FN_CONST_HEAD)
                && segments.len() > 1 =>
        {
            Some(&segments[1..])
        }
        _ => None,
    }
}

impl CallTarget {
    pub fn method(name: impl Into<String>, receiver_root: Option<String>) -> Self {
        Self::Method {
            name: name.into(),
            receiver_root,
            resolved_path: None,
            fun_decl_id: None,
            branch_payloads: None,
        }
    }

    pub fn with_branch_payloads(mut self, payloads: ResultBranchPayloads) -> Self {
        if let CallTarget::Method {
            branch_payloads, ..
        } = &mut self
        {
            *branch_payloads = Some(payloads);
        }
        self
    }

    pub fn resolved_path(&self) -> Option<&crate::parse::CallPath> {
        match self {
            CallTarget::Method { resolved_path, .. } => resolved_path.as_ref(),
            _ => None,
        }
    }

    /// Charon `FunDecl.def_id` stamped on a direct callee, if any.
    pub fn fun_decl_id(&self) -> Option<u64> {
        match self {
            CallTarget::Method { fun_decl_id, .. }
            | CallTarget::FunctionPath { fun_decl_id, .. } => *fun_decl_id,
            _ => None,
        }
    }

    /// Stamp the Charon `FunDecl.def_id` that names this callee.
    pub fn with_fun_decl_id(mut self, id: u64) -> Self {
        match &mut self {
            CallTarget::Method { fun_decl_id, .. }
            | CallTarget::FunctionPath { fun_decl_id, .. } => {
                *fun_decl_id = Some(id);
            }
            _ => {}
        }
        self
    }

    /// Stamp the registered graph path for a method callee.
    pub fn with_resolved_path(mut self, path: crate::parse::CallPath) -> Self {
        if let CallTarget::Method { resolved_path, .. } = &mut self {
            *resolved_path = Some(path);
        }
        self
    }

    pub fn function_path<I, S>(segments: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::FunctionPath {
            segments: segments.into_iter().map(Into::into).collect(),
            fun_decl_id: None,
        }
    }

    pub fn indirect(trait_root: impl Into<String>, method_name: impl Into<String>) -> Self {
        Self::Indirect {
            trait_root: trait_root.into(),
            method_name: method_name.into(),
        }
    }

    pub fn synthetic_transparent_ctor(name: impl Into<String>) -> Self {
        Self::SyntheticTransparentCtor {
            name: name.into(),
            owner_path: Vec::new(),
            is_struct: false,
            variant_tag: None,
        }
    }

    /// Owner-qualified variant — preserves the full `Owner::Variant`
    /// path so downstream `HostObject` identity does not collide on
    /// the bare leaf (e.g. `StepResult::Continue` vs
    /// `JitAction::Continue`).
    pub fn synthetic_transparent_ctor_with_owner(
        owner_path: Vec<String>,
        name: impl Into<String>,
    ) -> Self {
        Self::SyntheticTransparentCtor {
            name: name.into(),
            owner_path,
            is_struct: false,
            variant_tag: None,
        }
    }

    /// [`synthetic_transparent_ctor_with_owner`] for a resolved enum
    /// variant, carrying the variant's declaration index so a
    /// payload-less variant folded to a prebuilt singleton keeps its
    /// discriminant value.
    ///
    /// [`synthetic_transparent_ctor_with_owner`]: Self::synthetic_transparent_ctor_with_owner
    pub fn synthetic_transparent_enum_variant_ctor(
        owner_path: Vec<String>,
        name: impl Into<String>,
        variant_tag: i64,
    ) -> Self {
        Self::SyntheticTransparentCtor {
            name: name.into(),
            owner_path,
            is_struct: false,
            variant_tag: Some(variant_tag),
        }
    }

    /// [`synthetic_transparent_ctor_with_owner`] for a constructor the caller
    /// has resolved to a `struct` declaration rather than an enum variant.
    ///
    /// Separate from the general constructor so that "this is a struct" is
    /// something a caller states from a resolved `TypeDeclKind`, never a
    /// default a new call site inherits by omission.
    ///
    /// [`synthetic_transparent_ctor_with_owner`]: Self::synthetic_transparent_ctor_with_owner
    pub fn synthetic_transparent_struct_ctor(
        owner_path: Vec<String>,
        name: impl Into<String>,
    ) -> Self {
        Self::SyntheticTransparentCtor {
            name: name.into(),
            owner_path,
            is_struct: true,
            variant_tag: None,
        }
    }

    pub fn receiver_root(&self) -> Option<&str> {
        match self {
            CallTarget::Method { receiver_root, .. } => receiver_root.as_deref(),
            _ => None,
        }
    }

    pub fn path_segments(&self) -> Option<Vec<&str>> {
        match self {
            CallTarget::Method { name, .. } => Some(vec![name.as_str()]),
            CallTarget::FunctionPath { segments, .. } => {
                Some(segments.iter().map(String::as_str).collect())
            }
            // `(owner_path, name)` is the lookup identity per the
            // variant docs; collapsing onto the leaf alone would
            // make `Instruction::LoadFast` and `OtherEnum::LoadFast`
            // share segments and break any downstream caller that
            // uses `path_segments()` for qualified identity.
            CallTarget::SyntheticTransparentCtor {
                name, owner_path, ..
            } => {
                let mut segs: Vec<&str> = owner_path.iter().map(String::as_str).collect();
                segs.push(name.as_str());
                Some(segs)
            }
            CallTarget::Indirect { method_name, .. } => Some(vec![method_name.as_str()]),
            CallTarget::UnsupportedExpr => None,
        }
    }
}

/// Split a Rust-qualified type path at top-level `::` separators while
/// preserving separators inside generic arguments as part of the owning
/// segment.  Charon's named owner is already a structural path; reconstructing
/// a synthetic ctor from a rendered instantiation such as
/// `core::option::Option<Result<*mut m::Object, e::Error>>` must therefore
/// yield `core`, `option`, `Option<Result<*mut m::Object, e::Error>>`, not
/// split the payload's module paths into false owner segments.
pub(crate) fn split_qualified_path(path: &str) -> Vec<String> {
    let bytes = path.as_bytes();
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut depth = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'<' => depth += 1,
            b'>' => depth = depth.saturating_sub(1),
            b':' if depth == 0 && bytes.get(i + 1) == Some(&b':') => {
                if start < i {
                    out.push(path[start..i].to_string());
                }
                i += 2;
                start = i;
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    if start < path.len() {
        out.push(path[start..].to_string());
    }
    out
}

impl fmt::Display for CallTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CallTarget::Method {
                name,
                receiver_root: Some(receiver_root),
                ..
            } => write!(f, "{receiver_root}.{name}"),
            CallTarget::Method {
                name,
                receiver_root: None,
                ..
            } => f.write_str(name),
            CallTarget::FunctionPath { segments, .. } => f.write_str(&segments.join("::")),
            CallTarget::SyntheticTransparentCtor {
                name, owner_path, ..
            } => {
                if owner_path.is_empty() {
                    write!(f, "<synthetic-transparent-ctor {name}>")
                } else {
                    write!(
                        f,
                        "<synthetic-transparent-ctor {}::{name}>",
                        owner_path.join("::")
                    )
                }
            }
            CallTarget::Indirect {
                trait_root,
                method_name,
            } => write!(f, "<dyn {trait_root}>::{method_name}"),
            CallTarget::UnsupportedExpr => f.write_str("<unsupported-call-expr>"),
        }
    }
}

/// RPython call ops always carry `op.args[0]` as the funcptr operand.
/// Pyre keeps the same semantic slot but needs two Rust-level shapes:
/// a symbolic direct-call target or the runtime `Variable` holding the
/// funcptr value for indirect calls.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallFuncPtr {
    Target(CallTarget),
    Value(crate::flowspace::model::Variable),
}

/// RPython `flatten.py IndirectCallTargets`:
///
/// ```python
/// class IndirectCallTargets(object):
///     def __init__(self, lst):
///         self.lst = lst       # list of JitCodes
/// ```
///
/// Sidecar attached to `OpKind::CallResidual` when the residual call is the
/// tail of a regular-indirect lowering (`jtransform.py`).  The assembler
/// merges the JitCode handles into `Assembler.indirectcalltargets` so the
/// metainterp can later look up jitcodes by function-pointer address during
/// runtime dispatch (`pyjitpl.py`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndirectCallTargets {
    /// `Arc<JitCode>` shells (identity-keyed via `JitCodeHandle`) for
    /// every candidate impl in the `(trait, method)` family.  Matches
    /// RPython `flatten.py:55` `lst # list of JitCodes` shape.
    pub lst: Vec<crate::jitcode::JitCodeHandle>,
}

/// Signedness of one scalar field word. Both integer widths use the
/// `int` register class; `Float` is an `f64` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScalarFieldWidth {
    Signed,
    Unsigned,
    Float,
}

/// Byte offset of a scalar field whose owner has a Charon layout and no
/// field list. `fielddescrof` uses this word when the name lookup misses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ScalarFieldWord {
    pub offset: usize,
    pub size: usize,
    pub width: ScalarFieldWidth,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDescriptor {
    pub name: String,
    pub owner_root: Option<String>,
    /// Object-identity token for the owning struct / enum-variant type,
    /// minted from the full Charon `name_path()` at the field-resolution
    /// source (`resolve_adt_field`) where the qualified path is still in
    /// hand.  The layout layer keys offsets / sizes on this token instead
    /// of the bare-leaf `owner_root` string, so two distinct type
    /// definitions sharing a leaf name resolve to distinct offsets — the
    /// analog of RPython holding the live `lltype.Struct` object.  `None`
    /// when the descriptor was built outside the typed-ADT path (synthetic
    /// tuples, tests, builder construction).
    ///
    /// Excluded from `PartialEq` / `Hash` (see the manual impls below): it
    /// is functionally determined by `owner_root` at the typed sites, and
    /// keeping it out of identity means a descriptor built with the token
    /// stays equal to / hashes with one built without it, so op dedup /
    /// CSE over `FieldDescriptor` is unaffected.  That last clause is
    /// **defensive, not responsive**: there is no such pass.  Nothing keys
    /// a map on `OpKind` / `SpaceOperation` or compares them, and this
    /// type's manual `PartialEq` / `Hash` exist only to satisfy `OpKind`'s
    /// `derive` — the `HashMap<OpKind, …>` registries elsewhere are
    /// `flowspace::operation::OpKind`, a different type.  Do not reason
    /// from this exclusion as though a consumer were relying on it.
    pub owner_id: Option<majit_ir::descr::StructId>,
    /// Whether the access this descriptor was built for reached its
    /// container through a pointer: `(*p).f` is `Some(true)`, `local.f`
    /// is `Some(false)`, and `None` means the producer did not record it
    /// (every synthetic front-end helper, tests, builder construction).
    ///
    /// A field read on a *non-dereferenced local aggregate* cannot be an
    /// access to a live virtualizable — the container is a value in this
    /// frame, not an object the JIT is tracking — which is what
    /// `jtransform.py:990-993 fresh_virtualizable` suppresses upstream.
    /// Upstream obtains that by annotating `Frame.__init__`
    /// (`pypy/interpreter/pyframe.py`); the equivalent annotation is
    /// unreachable here because the constructor is shredded into ~65
    /// blocks by its call terminators while `vable_flags` is per-block,
    /// so the fact is recovered structurally instead.  Only `Some(false)`
    /// suppresses: `None` keeps the pre-existing behaviour, so a producer
    /// that does not know cannot silently disable the protocol.
    ///
    /// Sitting on `FieldDescriptor` is a compromise and worth naming: this
    /// describes the *access*, not the *field*, so a purist would put it on
    /// `OpKind::FieldRead` / `FieldWrite`.  Those are enum variants with no
    /// field defaults, so that costs an edit at all 282 construction sites
    /// to say "unknown" 280 times; the sites that can actually answer
    /// the question (place projections and the aggregate constructor's
    /// field stores) all build the descriptor.  It is only sound here
    /// because the flag is excluded from equality — see below.
    ///
    /// Excluded from `PartialEq` / `Hash` for the same reason as
    /// `owner_id`.  Unlike `owner_id` it is *not* functionally determined
    /// by the other fields: `Rvalue::Ref` aliases `&x` to `x`'s own
    /// Variable, so `let r = &q; (*r).f` and `q.f` reach one base Variable
    /// with opposite deref-ness.  Excluding it is still sound, because two
    /// descriptors that compare equal name one base Variable and therefore
    /// one container object, so they always want the same answer — and in
    /// that aliased case the container is the local aggregate, making
    /// `Some(false)` the correct reading and `Some(true)` merely
    /// conservative.
    pub base_is_deref: Option<bool>,
    /// The projection this descriptor was built for had its **address**
    /// taken (`&p.f`, `&raw mut (*p).f`) rather than its value read.
    ///
    /// The front models a reference as an alias of its referent
    /// (`Rvalue::Ref` / `RawPtr` both resolve to the referent's Variable,
    /// see the module header), so `&raw mut (*p).f` reaches the codewriter
    /// as an ordinary `FieldRead` of `f`.  For most consumers that
    /// imprecision is the long-standing model.  For a **virtualizable**
    /// field it is not survivable: the read is not kept as a value at all
    /// — an array field read is recorded in `vable_array_vars` and the op
    /// is dropped, so the operand the address was wanted for is left
    /// undefined.  `pyframe::FrameLocalsRoot::new` is the witnessed case;
    /// it registers `addr_of_mut!((*frame).locals_cells_stack_w)` as a GC
    /// root, and the only consumer is `try_gc_add_root`.
    ///
    /// So this suppresses the virtualizable lowering, the same way
    /// `base_is_local_aggregate` does.  `false` is "not recorded, or not
    /// an address-of" and keeps the existing lowering, so the conservative
    /// default is the one a producer gets by saying nothing — the opposite
    /// polarity to `base_is_deref`, where the *suppressing* value is
    /// `Some(false)` and an unknowing producer must say `None`.
    ///
    /// This does **not** repair the aliasing itself.  An address-of still
    /// yields the field's value, here as everywhere else; what it stops is
    /// the virtualizable path turning that into a dropped operand.  A real
    /// fix needs a distinct place-address op and its consumers.
    ///
    /// Excluded from `PartialEq` / `Hash` for the same reason as
    /// `owner_id` and `base_is_deref`.
    pub taken_by_address: bool,
    /// The field's layout spelling is an inline `Vec<T>` (three words), not
    /// a pointer to one. Set on the field read the front emits for the
    /// projection; `vec_part` then selects the buffer pointer or the length.
    /// Excluded from equality: it describes the access, and the part below
    /// is what makes two reads of one field distinct.
    pub inline_vec: bool,
    /// Which word of an inline aggregate this read loads. `None` is the
    /// field's own value. `Buf` / `Len` add `vec_layout::probe`'s offset;
    /// `FatData` / `FatLen` / `FatVtable` add `fat_ptr_layout::probe`'s offset.
    /// The add sits on top of the field's own offset.
    pub vec_part: Option<VecFieldPart>,
    /// Declared `Struct._gckind` of the owning struct or enum, recorded
    /// where the container type is still in hand. `Some(false)` is Raw:
    /// a Signed base stays an address (`getfield_raw`). `None` means the
    /// producer did not record it and the codewriter falls back to the
    /// layout registry. Excluded from `PartialEq` / `Hash` like `owner_id`.
    pub owner_declared_gc: Option<bool>,
    /// `(variant, field)` in the owner's host `variant_field_offsets`.
    /// A struct is variant 0. `fielddescrof` reads that byte offset when
    /// the base is a raw address; the explicit sum shell's `8 + i * 8`
    /// payload slots stay the gc representation. Excluded from
    /// `PartialEq` / `Hash` like `owner_declared_gc`.
    #[serde(default)]
    pub host_index: Option<(u32, u32)>,
    /// Layout word of a scalar field on an opaque dependency struct.
    /// `None` when the type registry can name the field. Part of equality:
    /// the offset is the access.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scalar_word: Option<ScalarFieldWord>,
}

/// One word of an inline aggregate the field read does not load whole.
///
/// `Buf` / `Len` are `alloc::vec::Vec<T>` (`vec_layout`). `FatData` /
/// `FatLen` are the two words of `Box<[T]>` (`fat_ptr_layout`).
/// `FatVtable` is that same metadata word typed as a pointer: `&dyn Trait`
/// stores the data pointer, then the vtable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VecFieldPart {
    Buf,
    Len,
    FatData,
    FatLen,
    FatVtable,
}

impl PartialEq for FieldDescriptor {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.owner_root == other.owner_root
            && self.vec_part == other.vec_part
            && self.scalar_word == other.scalar_word
    }
}

impl Eq for FieldDescriptor {}

impl std::hash::Hash for FieldDescriptor {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.owner_root.hash(state);
        self.vec_part.hash(state);
        self.scalar_word.hash(state);
    }
}

impl FieldDescriptor {
    pub fn new(name: impl Into<String>, owner_root: Option<String>) -> Self {
        Self {
            name: name.into(),
            owner_root,
            owner_id: None,
            base_is_deref: None,
            taken_by_address: false,
            inline_vec: false,
            vec_part: None,
            owner_declared_gc: None,
            host_index: None,
            scalar_word: None,
        }
    }

    /// Builder-style setter for the owning-type identity token.
    pub fn with_owner_id(mut self, owner_id: Option<majit_ir::descr::StructId>) -> Self {
        self.owner_id = owner_id;
        self
    }

    /// `Some(true)` when the owning struct or enum is a declared Gc type.
    pub fn with_owner_declared_gc(mut self, declared_gc: Option<bool>) -> Self {
        self.owner_declared_gc = declared_gc;
        self
    }

    /// `(variant, field)` into the owner's host layout. A struct is variant 0.
    pub fn with_host_index(mut self, host_index: Option<(u32, u32)>) -> Self {
        self.host_index = host_index;
        self
    }

    /// Builder-style setter for the access's container shape.
    /// `true` for `(*p).f`, `false` for `local.f`.
    pub fn with_base_is_deref(mut self, base_is_deref: bool) -> Self {
        self.base_is_deref = Some(base_is_deref);
        self
    }

    /// The container is known to be a non-dereferenced local aggregate,
    /// so this access cannot be reaching a live virtualizable.
    pub fn base_is_local_aggregate(&self) -> bool {
        self.base_is_deref == Some(false)
    }

    /// Builder-style setter for [`Self::taken_by_address`].
    pub fn with_taken_by_address(mut self, taken_by_address: bool) -> Self {
        self.taken_by_address = taken_by_address;
        self
    }

    pub fn with_inline_vec(mut self, inline_vec: bool) -> Self {
        self.inline_vec = inline_vec;
        self
    }

    /// Load one measured word of this `Vec` field. Clears the address mark:
    /// the result is the word's value, not the address of the three-word
    /// aggregate.
    pub fn with_vec_part(mut self, part: VecFieldPart) -> Self {
        self.vec_part = Some(part);
        self.taken_by_address = false;
        self
    }

    /// Record the layout word of an opaque scalar field.
    pub fn with_scalar_word(mut self, word: ScalarFieldWord) -> Self {
        self.scalar_word = Some(word);
        self
    }

    /// This access must not lower through the virtualizable protocol:
    /// either the container is a local aggregate rather than a live
    /// virtualizable, or what was taken is the field's address rather
    /// than its value.
    pub fn suppresses_virtualizable(&self) -> bool {
        self.base_is_local_aggregate() || self.taken_by_address
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpKind {
    Input {
        name: String,
        ty: ValueType,
        /// Optional class identity for `Ref`-typed parameters carried
        /// from the front-end's `syn::Type` projection.  Mirrors how
        /// PyPy's annotator routes typed `&Foo` through
        /// `bookkeeper.getuniqueclassdef` (`description.py pycall
        /// FunctionDesc.pycall`) so the rtyper's `find_attribute`
        /// (`rclass.py:556`) lands on the actual `ClassDef`.  The
        /// `front::mir` param lowering currently leaves this `None` for
        /// every parameter; typed pointer precision, when carried, comes
        /// from the leaf segment of the param's `ValueType::Ref(_)` root
        /// when that leaf matches a known struct in
        /// `program.struct_fields`.  Non-`Ref` params (Int, Float, Bool,
        /// Void) always have `None`.
        class_root: Option<String>,
    },
    ConstInt(i64),
    /// Function-pointer integer constant (`jtransform.py
    /// direct_funcptr_value` / `handle_residual_call`). Same `'i'` bank
    /// as [`ConstInt`]; `path` is `assembler.py emit_const`'s symbolic
    /// object (`llmemory.AddressAsInt` / `lltype` function pointer) so
    /// the assembler records provenance instead of pooling a bare int.
    ConstFnAddr {
        value: i64,
        path: String,
        /// True when no build address existed and `value` is a
        /// `symbolic_fnaddr_for_path` hash. Set at the
        /// `fnaddr_binding_for_target` lookup, never inferred from bits.
        symbolic: bool,
    },
    /// Word-sized unsigned integer constant. RPython stores the same Python
    /// integer value as `ConstInt`, but its `Constant.concretetype` is
    /// `lltype.Unsigned`; keeping the source tag prevents `r_uint ∪ int`
    /// merges when MIR joins an unsigned literal with another `usize` value.
    ConstUInt(u64),
    /// Translation-time `r_longlonglong` constant. This carrier may
    /// appear in flow/rtyping graphs but must never be coerced into a
    /// JIT register-kind `ConstInt`.
    ConstInt128(i128),
    /// Translation-time `r_ulonglonglong` constant.
    ConstUInt128(u128),
    /// RPython `model.py` `Constant(bool_value)` — a bool constant
    /// whose `concretetype` is `lltype.Bool`. The codewriter folds Bool
    /// into kind `'int'` (`rpython/jit/codewriter/flatten.py:getkind`),
    /// so backend lowering shares the integer materialization path with
    /// `ConstInt`. Distinct from `ConstInt(0/1)` at the annotator /
    /// rtyper layer: lifts to `SomeBool` (`annotator/model.py`)
    /// instead of `SomeInteger`, and selects `BoolRepr`
    /// (`rpython/rtyper/rbool.py`) instead of `IntegerRepr`.
    ConstBool(bool),
    /// RPython `model.py` `Constant(rfloat)` — a float constant whose
    /// `concretetype` is `lltype.Float`.  Stored as the f64 bit pattern
    /// (`history.py ConstFloat.getfloatstorage`) so PartialEq/Hash
    /// stay derivable.  The assembler materialises this through the
    /// existing `constants_f` pool with a `float_copy` op, mirroring
    /// the `ConstInt` → `int_copy` lowering.
    ConstFloat(u64),
    /// RPython `model.py` `Constant(value)` whose `concretetype` is
    /// `lltype.SingleFloat` — a Rust `f32` literal.  Stored as the f32
    /// bit pattern, which is what distinguishes it from [`OpKind::ConstFloat`]:
    /// a Charon float constant records its width (`{"Float": {"value":
    /// "...", "ty": "F32"}}`), and parsing both widths into an f64 bit
    /// pattern is what erased it.
    ///
    /// It carries no [`ValueType`] field, for the same reason
    /// [`OpKind::ConstInt128`] carries none: the width is in the variant
    /// name.  The codewriter policy reaches it the way `policy.py:96-98`
    /// reaches upstream's `Constant(value, SingleFloat)` through
    /// `op.args`, and refuses the graph.
    ConstSingleFloat(u32),
    /// RPython `model.py` `Constant(str_value)` after
    /// `StringRepr.convert_const` resolves the host string to a
    /// prebuilt `Ptr(STR)`. The pre-jtransform string-constant fold
    /// carries bytes here; the assembler mints the lltype pointer and
    /// materialises it through the ref constant pool.
    ConstStr(Vec<u8>),
    /// Interned Python str constant — `box_str_constant` over a proven
    /// string literal. Distinct from [`ConstStr`]: that is rstr `Ptr(STR)`
    /// (`StringRepr.convert_const`); this is the wrapped immortal
    /// `W_UnicodeObject` a residual `w_name` argument carries as one Ref.
    ConstInternedStr(Vec<u8>),
    /// RPython `model.py` `Constant(host_object)` resolved by the
    /// rtyper to a singleton instance pointer
    /// (`rtyper/rpbc.py::SingleFrozenPBCRepr`).  Stored as a thin
    /// `HostObject` handle so the assembler can stash
    /// `obj.identity_id()` in the ref-kind constant pool
    /// (`assembler.rs::emit_const_r`) and emit `ref_copy/r>r` —
    /// mirroring the `ConstInt` → `int_copy` lowering for the
    /// ref bank.  Producers: the pre-jtransform unit-variant
    /// rewrite pass (`translator/rtyper/unit_variant_fold.rs`)
    /// folds `OpKind::Call { target: SyntheticTransparentCtor,
    /// args: [] }` matching `is_synthetic_unit_variant_path`
    /// into this variant.
    ConstRef(crate::flowspace::model::HostObject),
    /// RPython `lltype.nullptr(T)` / null GC ref constant. Pyre spells
    /// the common sentinel as `PY_NULL`; it materialises in the ref bank
    /// as address 0, not as an integer register.
    ConstRefNull,
    /// The RPython annotator constant `None` — a `Constant(NoneType)` whose
    /// `concretetype` is `lltype.Void`.  Distinct from [`ConstRefNull`]: the
    /// latter is a null GCREF (`SomePtr`), whereas this folds to
    /// `SomeValue::None_` (`SomeNone`) so the rtyper's `none_repr` /
    /// `decompose_slice_args` `stop_is_none` selection see a true `None`.
    /// The sole producer is the gated slice-range front recognizer, which
    /// plants it as the `stop` operand of a `[start:]` (`StartOnly`)
    /// `getslice`; it mirrors how `LOAD_CONST None` pushes `Constant(None)`
    /// upstream.  Const-inlined by `legacy_const_define_hlvalue`, so no
    /// SpaceOp reaches the assembler.
    ConstNone,
    /// Process-wide singleton pointer that must be materialised in the
    /// ref bank (e.g. PyPy `space.fromcache(DictStrategy)` singletons
    /// represented by pyre's Rust `pub static *_DICT_STRATEGY` values).
    ConstRefAddr(i64),
    /// An identity-bearing symbolic constant — pyre's model-graph
    /// carrier for RPython's `CDefinedIntSymbolic` singletons
    /// (`rpython/rlib/jit.py _we_are_jitted`).  `tag` is the
    /// process-unique `flowspace::model::ConstValue::SpecTag` id the
    /// symbolic is keyed by (e.g.
    /// [`crate::translator::backendopt::constfold::WE_ARE_JITTED_TAG_ID`]);
    /// `ty` is its `concretetype` (`Bool` for pyre's
    /// `we_are_jitted() -> bool`).
    ///
    /// RPython's rtyper `specialize_call` rewrites the `direct_call`
    /// in-place to `inputconst(Signed, _we_are_jitted)`
    /// (`rpython/rlib/jit.py`), so the symbolic `Constant`
    /// flows through the single graph that both the backend and
    /// `codewriter/jtransform.py` consume.  Pyre's rtyper is a
    /// type oracle on an ephemeral graph that never rewrites the
    /// surviving model graph, so the symbolic is injected here by
    /// `front::mir` (the model-graph builder) instead.  Downstream:
    /// `flowspace_adapter` re-materialises it as `Constant(SpecTag,
    /// <ty>)` for the rtyper / `constfold::replace_we_are_jitted`
    /// (→ `false`) genc path, and `jtransform` folds it to
    /// `ConstBool(true)` keyed on the `SpecTag` identity (parity with
    /// `jtransform.py Transformer.rewrite_op_int_is_true value is _we_are_jitted`).  No
    /// `ConstSymbolic` survives `jtransform`, so the backend never
    /// materialises one.
    ConstSymbolic {
        tag: u64,
        ty: ValueType,
    },
    FieldRead {
        base: crate::flowspace::model::Variable,
        field: FieldDescriptor,
        ty: ValueType,
        /// RPython `jtransform.py:867-903` may rewrite immutable /
        /// quasi-immutable reads to `getfield_*_pure`.  Carries the
        /// chosen opcode flavour through flatten/assembly so the
        /// runtime sees the `_pure` bytecode variant instead of having
        /// to rediscover purity from the descriptor later.
        pure: bool,
    },
    FieldWrite {
        base: crate::flowspace::model::Variable,
        field: FieldDescriptor,
        /// RPython `setfield_gc(p, x, descr)` where `x` is an
        /// `AbstractValue` — either a `Variable` (register operand) or a
        /// `Constant` (inline literal).  `LinkArg::Value`/`Const` mirrors
        /// that union so a small-int constant can stay inline as the
        /// stored value (`setfield_gc_i/rcd`) instead of being hoisted
        /// into a separate `ConstInt -> register` materialisation.  Until
        /// the MIR front-end stops materialising (the c-form activation),
        /// every producer mints `LinkArg::Value`.
        value: LinkArg,
        ty: ValueType,
    },
    /// RPython `malloc(GcStruct, flavor='gc')` lowered by
    /// `jtransform.py rewrite_op_malloc` to `new(descr)`.
    /// Synthetic Rust aggregate constructors reach pyre before the missing
    /// rtyper malloc lowering, so jtransform restores that exact operation
    /// using the aggregate owner to resolve its size descriptor.
    New {
        owner: String,
    },
    /// `lltype.malloc(T, flavor='raw')` for a fixed-size struct.
    ///
    /// `jtransform.py _rewrite_raw_malloc` turns this into a residual
    /// direct call of `raw_malloc_fixedsize` (the `_zero` helper when
    /// `zero` is set). The result is a Signed address. The MIR front
    /// end emits it when a call passes `&` / `&mut` of a primitive into
    /// a raw scalar-pointer parameter: the borrow aliases the word, and
    /// `history.py` `getkind` of that parameter is the address.
    RawMalloc {
        owner: String,
        zero: bool,
    },
    /// `lltype.free(p, flavor='raw')`.
    ///
    /// `jtransform.py rewrite_op_free` rewrites this to a residual of
    /// `raw_free` carrying `OopSpecIndex::RawFree`. The call cannot raise.
    RawFree {
        ptr: crate::flowspace::model::Variable,
    },
    /// RPython `malloc(STRUCT, flavor='gc')` for a fixed-size GcStruct: the
    /// heap allocation of a boxed object (`runtime_object::lltype::malloc_typed`).
    /// Lowered to the `new_with_vtable` jitcode op (executor
    /// `OpCode::NewWithVtable`). `owner` is the crate-stripped nominal struct
    /// path (e.g. `"floatobject::W_FloatObject"`); the assembler resolves the
    /// size descriptor from it via `bh_size_spec_from_callcontrol`, whose
    /// `path_hash(owner)` keys the runtime `gc_cache._cache_size` Arc carrying
    /// the struct size + gc type-id.  Synthetic fixtures without a registered
    /// path retain their exact bare owner.
    ///
    /// `vtable` is the type-pointer (the `&FLOAT_TYPE` / `&INT_TYPE` /
    /// `&COMPLEX_TYPE` static address) the runtime stamps into the fresh
    /// object's `ob_type` (and, via `get_instantiate`, its `w_class`).  The GC
    /// allocator reads it from the size descriptor's `vtable` field — NOT from
    /// `type_id`, which resolves the struct *size* only (`gc_cache._cache_size`
    /// carries no type pointer).  `fuse_boxing_alloc` captures it from the
    /// boxed constructor's dropped `ob_header.ob_type` store; a `0` vtable is
    /// an unresolved/non-boxing placeholder the assembler rejects.  The result
    /// register is always a fresh `Ref` ('r').
    NewWithVtable {
        owner: String,
        vtable: i64,
    },
    /// RPython `new_array(v_length, arraydescr)` — the uninitialised
    /// fixed-size array allocation `do_fixed_newlist` emits when
    /// `ARRAY.OF` is neither a GC `Ptr` nor a `Struct`.  Operand shape
    /// matches `NewArrayClear`; only the opcode differs
    /// (`new_array/id>r`, `bhimpl_new_array`).
    NewArray {
        length: crate::flowspace::model::Variable,
        item_ty: ValueType,
        /// ARRAY identity for `cpu.arraydescrof(ARRAY)`, same role as
        /// `ArrayRead::array_type_id`.
        array_type_id: Option<String>,
    },
    /// RPython `new_array_clear(v_length, arraydescr)` — the cleared
    /// fixed-size array allocation `do_fixed_newlist_clear` emits
    /// (`corresponds to rtyper.rlist.ll_alloc_and_clear: needs to clear
    /// the items`).  Also the allocation half `do_fixed_newlist` picks
    /// when `ARRAY.OF` is a GC `Ptr` or `Struct` — the exact case that
    /// distinguishes `new_array_clear` from the uninitialised `new_array`.
    ///
    /// `length` is the item count (`_get_initial_newlist_length`,
    /// `jtransform.py`).  `item_ty` / `array_type_id` resolve the
    /// `arraydescr` at assembly exactly like `ArrayRead`/`ArrayWrite`; a
    /// cleared list holds GC pointers (`Ptr(GcArray(OBJECTPTR))`), so
    /// `item_ty` is `Ref`.  The allocator zero-fills every slot
    /// (`bhimpl_new_array_clear`, `blackhole.py`), so the result
    /// register is always a fresh `Ref` ('r') and the emitted key is
    /// `new_array_clear/id>r`.
    NewArrayClear {
        length: crate::flowspace::model::Variable,
        item_ty: ValueType,
        /// ARRAY identity for `cpu.arraydescrof(ARRAY)`, same role as
        /// `ArrayRead::array_type_id`.
        array_type_id: Option<String>,
    },
    /// The compound resizable-list allocation `opimpl_newlist_clear`
    /// records (`pyjitpl.py`) — the resized-layout counterpart to
    /// `do_resizable_newlist_clear` (`jtransform.py`).  Allocates
    /// the resizable-list `GcStruct` header (the struct named `"list"`),
    /// zero-clears a fresh items array of `length` slots, and wires the two
    /// header fields `length`/`items` so the result is a fully-formed empty
    /// resizable list.  `bhimpl_newlist_clear` (`blackhole.py`,
    /// `@arguments("cpu", "i", "d", "d", returns="r")`) runs the same four
    /// steps at the blackhole boundary: `new(structdescr)` +
    /// `setfield(length)` + `new_array_clear(items_arraydescr)` +
    /// `setfield(items)`.
    ///
    /// The owner struct is always `"list"` and its fields are always
    /// `length`/`items`, so neither is carried on the variant — only the
    /// `new_array_clear` sub-operation's operands survive: `length` (item
    /// count), `item_ty` (the items array element type, `Ref` for a cleared
    /// list of GC pointers), and `array_type_id` (the items ARRAY identity
    /// for `cpu.arraydescrof(ARRAY)`).  Field shape is identical to
    /// `NewArrayClear` above.
    NewListClear {
        length: crate::flowspace::model::Variable,
        item_ty: ValueType,
        /// ARRAY identity for `cpu.arraydescrof(ARRAY)`, same role as
        /// `NewArrayClear::array_type_id`.
        array_type_id: Option<String>,
    },
    ArrayRead {
        base: crate::flowspace::model::Variable,
        index: crate::flowspace::model::Variable,
        item_ty: ValueType,
        /// RPython: ARRAY identity for `cpu.arraydescrof(ARRAY)`.
        /// Distinguishes arrays with the same item_ty but different
        /// container types (e.g. `Vec<Point>` vs `Vec<Line>`).
        array_type_id: Option<String>,
        /// RPython: `ARRAY_INSIDE._hints.get('nolength', False)`
        /// (descr.py:359). `true` when the operand addresses a raw
        /// items region with no length header; `false` (the common
        /// length-prefixed shape) lays the length word at offset 0
        /// and items past it.
        nolength: bool,
        /// RPython: the foldable/immutable element load
        /// `getarrayitem_gc_{i,r,f}_pure`.  `ll_getitem_foldable_nonneg`
        /// (rlist.py:721-724, `oopspec = 'list.getitem_foldable(l,
        /// index)'`) is selected by `rtype_getitem` when `not
        /// listdef.listitem.mutated` (rlist.py:256-258), and the
        /// `FixedSizeListRepr` iterator next is `ll_listnext_foldable`
        /// when `not r_list.listitem.mutated`
        /// (lltypesystem/rlist.py).  `true` only for such a
        /// foldable/unmutated selection or an immutable container —
        /// NEVER for a mutable list element.
        pure: bool,
    },
    /// `arraylen_gc(array, arraydescr)` — read a GcArray's length
    /// header (`len(l.items)`, rlist.py:251).  Carries only the array
    /// base; the result is always the length `Int`.
    ArrayLen {
        base: crate::flowspace::model::Variable,
        /// ARRAY identity for `cpu.arraydescrof(ARRAY)`.
        array_type_id: Option<String>,
        /// `ARRAY_INSIDE._hints.get('nolength', False)` (descr.py).
        /// `false` (length-prefixed) lays the length word at offset 0.
        nolength: bool,
    },
    ArrayWrite {
        base: crate::flowspace::model::Variable,
        index: crate::flowspace::model::Variable,
        /// RPython `setarrayitem_gc(p, i, x, descr)` where `x` is an
        /// `AbstractValue` — `op.args[2]` may be a `Variable` or a
        /// `Constant` (jtransform.py:803).  `LinkArg::Value`/`Const`
        /// mirrors that union so a small-int constant value can stay
        /// inline (`setarrayitem_gc_i/ricd`) instead of being hoisted
        /// into a `ConstInt -> register` materialisation, matching the
        /// FieldWrite c-form activation.
        value: LinkArg,
        item_ty: ValueType,
        /// RPython: ARRAY identity for `cpu.arraydescrof(ARRAY)`.
        array_type_id: Option<String>,
        /// RPython: `ARRAY_INSIDE._hints.get('nolength', False)`
        /// (descr.py:359). See `ArrayRead::nolength`.
        nolength: bool,
    },
    /// `raw_load_{i,f}` — one element read out of raw (non-GC) memory at a
    /// byte offset (`jtransform.py rewrite_op_raw_load`).  The
    /// descr is `arraydescrof(rffi.CArray(T))`: no length header, item
    /// width and signedness from `T` — carried here explicitly because
    /// [`ValueType`] collapses widths.
    RawLoad {
        base: crate::flowspace::model::Variable,
        offset: crate::flowspace::model::Variable,
        item_ty: ValueType,
        itemsize: usize,
        is_item_signed: bool,
    },
    /// `raw_store_{i,f}` (`jtransform.py rewrite_op_raw_store`).
    /// See [`OpKind::RawLoad`] for the descr shape.
    RawStore {
        base: crate::flowspace::model::Variable,
        offset: crate::flowspace::model::Variable,
        value: crate::flowspace::model::Variable,
        item_ty: ValueType,
        itemsize: usize,
        is_item_signed: bool,
    },
    /// RPython: getinteriorfield_gc_i/r/f — read a field of an array-of-structs element.
    /// effectinfo.py add_interiorfield: generates "readinteriorfield" effect.
    /// effectinfo.py:327-340: also implicitly generates "readarray" effect.
    InteriorFieldRead {
        base: crate::flowspace::model::Variable,
        index: crate::flowspace::model::Variable,
        field: FieldDescriptor,
        item_ty: ValueType,
        array_type_id: Option<String>,
    },
    /// RPython: setinteriorfield_gc — write a field of an array-of-structs element.
    /// effectinfo.py:349-350: generates "interiorfield" effect.
    /// effectinfo.py:327-340: also implicitly generates "array" effect.
    InteriorFieldWrite {
        base: crate::flowspace::model::Variable,
        index: crate::flowspace::model::Variable,
        field: FieldDescriptor,
        value: crate::flowspace::model::Variable,
        item_ty: ValueType,
        array_type_id: Option<String>,
    },
    Call {
        target: CallTarget,
        /// Mixed Variable | Constant, matching
        /// `flowspace/model.py SpaceOperation.args`.
        args: Vec<LinkArg>,
        result_ty: ValueType,
    },
    GuardTrue {
        cond: crate::flowspace::model::Variable,
    },
    GuardFalse {
        cond: crate::flowspace::model::Variable,
    },

    /// Guard that a value equals a compile-time constant.
    ///
    /// RPython upstream emits three opnames in the guard_value family
    /// via `codewriter/jtransform.py rewrite_op_hint`:
    /// `int_guard_value` / `ref_guard_value` / `float_guard_value`,
    /// each a 1-input/0-output pointer-or-value compare per the arg's
    /// `getkind()`.
    ///
    /// `str_guard_value` (`jit.py` for `promote_string`, `:647`
    /// for `promote_unicode`) is NOT modeled as a distinct op: it is the
    /// value-equality promotion of an `rstr.STR` / `rstr.UNICODE` low-level
    /// string, comparing the inline char array
    /// (`support.py _ll_2_str_eq_nonnull` indexes `s.chars[i]`).
    /// pyre interpreter strings are `W_UnicodeObject` GC refs, never
    /// `Ptr(rstr.STR)`, so there is no char array to value-compare; the
    /// `PromoteString` / `PromoteUnicode` rewrite arms lower their ref
    /// operand through the ref-kind member of this family — `r_guard_value`
    /// (`kind_char == 'r'`), an identity guard on the string pointer.
    GuardValue {
        value: crate::flowspace::model::Variable,
        /// `'i'` int, `'r'` ref, `'f'` float — matching the
        /// `<kind>_guard_value` family naming at
        /// `codewriter/jtransform.py:611`.
        kind_char: char,
    },
    /// `jtransform.py handle_getfield_typeptr` — a read of the
    /// object header's class word is not a load but a `guard_class`: the
    /// tracer pins the receiver's class and the op's result is that class
    /// as a constant, which is what lets every `isinstance`-shaped test
    /// downstream fold instead of recording `getfield` + `ptr_eq` +
    /// `guard_true`.  The blackhole reads the word (`bhimpl_guard_class`,
    /// `cpu.bh_classof`).
    ///
    /// pyre's class word is `PyObject.ob_type`, embedded first in every
    /// object as `ob_header`; `heaptracker::is_header_word` knows it by
    /// that name.  The result register keeps the kind the read had —
    /// `int` when the pointer was typed raw, `ref` when it was typed
    /// GC — so the consumers of the original read are untouched; the
    /// class-pointer hint ops (`record_exact_class/ri`) already carry the
    /// class through the int bank.
    GuardClass {
        base: crate::flowspace::model::Variable,
    },
    /// Project a callee function pointer out of a `dyn Trait` receiver's
    /// vtable for the named method slot.  Result is integer-typed so it
    /// can be fed to `int_guard_value` (RPython `jtransform.py`).
    ///
    /// TODO: pyre adaptation of `rclass.py getclsfield()`. RPython
    /// emits a `cast_pointer + getfield(vtable_struct, method_name)` chain
    /// because `ClassRepr` models the vtable as an explicit `Struct`. Rust
    /// `dyn Trait` vtable layout is compiler-internal (unstable ABI), so
    /// pyre cannot model the vtable as an IR struct — this single op stands
    /// in for the chain and must be emitted by the rtyper-equivalent layer
    /// (`translator/rtyper/rclass.rs`), never by `jtransform`.
    VtableMethodPtr {
        receiver: crate::flowspace::model::Variable,
        trait_root: String,
        method_name: String,
    },
    /// Indirect call — `funcptr` is the `Variable` carrying the callable
    /// produced by the rtyper layer (e.g. from `VtableMethodPtr` for
    /// `dyn Trait` dispatch). `args` are the full call arguments,
    /// including the receiver. `graphs` mirrors the trailing `c_graphs`
    /// constant from `rpbc.py`: `Some(full_family)` when known,
    /// `None` otherwise.
    ///
    /// RPython: `rpython/rtyper/rpbc.py`
    /// ```python
    /// vlist.append(hop.inputconst(Void, row_of_graphs.values()))
    /// v = hop.genop('indirect_call', vlist, resulttype=rresult)
    /// ```
    /// Lowered downstream by `jtransform.py rewrite_op_indirect_call`.
    IndirectCall {
        funcptr: crate::flowspace::model::Variable,
        args: Vec<crate::flowspace::model::Variable>,
        graphs: Option<Vec<crate::parse::CallPath>>,
        /// Pre-rtyping identity of a Rust dyn-trait vtable slot.  The MIR
        /// frontend preserves the actual `Field[Adt(vtable), slot]` as
        /// `funcptr`; `rpbc::lower_indirect_calls` consumes this key to fill
        /// the same `c_graphs` family as `FunctionReprBase.call`, then clears
        /// it.  Ordinary stored function pointers have no such key.
        family_key: Option<(String, String)>,
        result_ty: ValueType,
    },
    /// Virtualizable field read → reads from boxes, no heap op.
    /// RPython: `getfield_vable_i/r/f`
    VableFieldRead {
        base: crate::flowspace::model::Variable,
        field_index: usize,
        ty: ValueType,
    },
    /// Virtualizable field write → writes to boxes, no heap op.
    /// RPython: `setfield_vable_i/r/f`
    ///
    /// `value` is a [`LinkArg`] (register or inline constant) like
    /// [`OpKind::FieldWrite`]: `rewrite_op_setfield` passes the setfield
    /// `v_value` straight to `setfield_vable_%s`
    /// (`jtransform.py`), which may be a `Constant`
    /// (`flatten.py:360-371`).  `setfield_vable_i` is not in `USE_C_FORM`
    /// (`assembler.py:312-345`), so a constant value always takes the
    /// pool `i` slot (`setfield_vable_i/rid`), never the short `c` byte.
    VableFieldWrite {
        base: crate::flowspace::model::Variable,
        field_index: usize,
        value: LinkArg,
        ty: ValueType,
    },
    /// Virtualizable array read → reads from boxes.
    /// RPython: `getarrayitem_vable_i/r/f`
    VableArrayRead {
        base: crate::flowspace::model::Variable,
        array_index: usize,
        elem_index: crate::flowspace::model::Variable,
        item_ty: ValueType,
        /// RPython: arraydescr.itemsize from VirtualizableInfo.array_descrs.
        array_itemsize: usize,
        /// RPython: arraydescr.is_item_signed() from VirtualizableInfo.array_descrs.
        array_is_signed: bool,
    },
    /// Virtualizable array write → writes to boxes.
    /// RPython: `setarrayitem_vable_i/r/f`. `rewrite_op_setarrayitem` passes
    /// the value through; a `Constant` stays a `LinkArg::Const` and the
    /// assembler encodes it (`jtransform.py` `rewrite_op_setarrayitem`).
    VableArrayWrite {
        base: crate::flowspace::model::Variable,
        array_index: usize,
        elem_index: crate::flowspace::model::Variable,
        value: LinkArg,
        item_ty: ValueType,
        /// RPython: arraydescr.itemsize from VirtualizableInfo.array_descrs.
        array_itemsize: usize,
        /// RPython: arraydescr.is_item_signed() from VirtualizableInfo.array_descrs.
        array_is_signed: bool,
    },
    /// Virtualizable array length → reads the length off the boxes.
    /// RPython: `arraylen_vable`.
    ///
    /// Emitted by `jtransform.py rewrite_op_getarraysize`, the third
    /// consumer of `vable_array_vars` alongside [`OpKind::VableArrayRead`]
    /// and [`OpKind::VableArrayWrite`].  Without it a `len()` over a
    /// virtualizable array is the one route that still reads the raw array
    /// pointer, so the field read producing that pointer cannot be dropped
    /// the way `rewrite_op_getfield`'s `except VirtualizableArrayField:`
    /// handler drops it.
    ///
    /// Carries the two descrs its siblings do even though the length comes
    /// off the vable-array one alone: the bytecode key is
    /// `arraylen_vable/rdd>i` (`insns.rs`), and `vable_array_index_pair_at`
    /// rejects a pair that is not `(VableArray, Array)`.
    VableArrayLen {
        base: crate::flowspace::model::Variable,
        array_index: usize,
        item_ty: ValueType,
        /// RPython: arraydescr.itemsize from VirtualizableInfo.array_descrs.
        array_itemsize: usize,
        /// RPython: arraydescr.is_item_signed() from VirtualizableInfo.array_descrs.
        array_is_signed: bool,
    },
    /// Binary arithmetic/comparison operation.
    /// RPython: `int_add`, `int_lt`, etc.
    BinOp {
        op: String,
        lhs: crate::flowspace::model::Variable,
        rhs: crate::flowspace::model::Variable,
        result_ty: ValueType,
    },
    /// Unary operation.
    /// RPython: `int_neg`, `bool_not`, etc.
    UnaryOp {
        op: String,
        operand: crate::flowspace::model::Variable,
        result_ty: ValueType,
    },

    /// Force virtualizable: flush boxes to heap.
    /// RPython: `hint_force_virtualizable(vable)`
    VableForce {
        base: crate::flowspace::model::Variable,
    },

    /// JIT hint operation — `hint(x, **kwds)`.
    ///
    /// RPython models a single `hint` operator
    /// (`rpython/flowspace/operation.py add_operator('hint', None,
    /// dispatch=1)`) whose kwarg dict selects the behaviour; `rlib/jit.py
    /// promote(x)` and `#[elidable_promote]`'s per-arg wrapper both lower to
    /// it.  Pyre carries the kwarg key as the structured [`kind`] field
    /// instead of a `Void`-constant dict (which `OpKind::Call` cannot hold),
    /// so the typing oracle (`flowspace_adapter`) and the JIT codewriter
    /// (`jtransform::rewrite_op_hint`) dispatch on the `kind` rather than a
    /// synthesised marker name.  Outside the JIT the op is an identity on
    /// `value` (the flowspace oracle lowers it to `same_as`); the JIT
    /// codewriter rewrites it to the `<kind>_guard_value` family
    /// (`codewriter/jtransform.py`).
    Hint {
        value: crate::flowspace::model::Variable,
        kind: crate::hints::HintKind,
    },

    // ── Call effect classification (generated by jtransform) ────
    //
    // RPython jtransform.py `rewrite_call()`: args are split by kind
    // into three ListOfKind lists. The opname encodes the kind signature:
    //   residual_call_ir_i  = int+ref args, int result
    //   call_elidable_r_v   = ref args, void result
    //
    /// Elidable (pure) call — no side effects, result depends only on args.
    /// RPython: `call_elidable_{kinds}_{reskind}(funcptr, calldescr, [i], [r], [f])`
    /// `funcptr` mirrors `op.args[0]` in RPython. Direct calls keep a
    /// symbolic target; indirect calls carry the runtime `Variable`
    /// produced by rtype.
    CallElidable {
        funcptr: CallFuncPtr,
        descriptor: crate::call::CallDescriptor,
        args_i: Vec<crate::flowspace::model::Variable>,
        args_r: Vec<crate::flowspace::model::Variable>,
        args_f: Vec<crate::flowspace::model::Variable>,
        result_kind: char,
    },
    /// Residual call — has side effects, must be preserved.
    /// RPython: `residual_call_{kinds}_{reskind}(funcptr, calldescr, [i], [r], [f])`.
    /// See `CallElidable` for `funcptr` semantics.
    /// `indirect_targets` mirrors the `IndirectCallTargets(lst)` sidecar
    /// that RPython appends to the extraargs for an indirect_call family
    /// (`jtransform.py:547`).  `None` for direct-call lowering.
    CallResidual {
        funcptr: CallFuncPtr,
        descriptor: crate::call::CallDescriptor,
        args_i: Vec<crate::flowspace::model::Variable>,
        args_r: Vec<crate::flowspace::model::Variable>,
        args_f: Vec<crate::flowspace::model::Variable>,
        result_kind: char,
        indirect_targets: Option<IndirectCallTargets>,
    },
    /// May-force call — can trigger GC or force virtualizables.
    /// RPython: `call_may_force_{kinds}_{reskind}(funcptr, calldescr, [i], [r], [f])`.
    /// See `CallElidable` for `funcptr` semantics.
    CallMayForce {
        funcptr: CallFuncPtr,
        descriptor: crate::call::CallDescriptor,
        args_i: Vec<crate::flowspace::model::Variable>,
        args_r: Vec<crate::flowspace::model::Variable>,
        args_f: Vec<crate::flowspace::model::Variable>,
        result_kind: char,
    },

    // ── Call kind classification (generated by jtransform via CallControl) ──
    //
    // RPython jtransform.py `rewrite_call()`: args are split by kind
    // into three lists (int, ref, float). The opname encodes the kind signature:
    //   inline_call_ir_i  = int+ref args, int result
    //   residual_call_r_v = ref args, void result
    //
    // `result_kind`: 'i', 'r', 'f', or 'v' (RPython `getkind(result.concretetype)`)
    /// Inline call — callee is a regular candidate graph.
    /// RPython: `inline_call_{kinds}_{reskind}(jitcode, [i_args], [r_args], [f_args])`
    /// RPython jtransform.py.
    InlineCall {
        /// RPython: `callcontrol.get_jitcode(targetgraph)` returns the
        /// callee JitCode object itself, not its final `index`.
        /// pyre carries the same identity-bearing handle until the
        /// assembler snapshots the final descriptor table.
        jitcode: crate::jitcode::JitCodeHandle,
        /// Integer arguments (RPython: ListOfKind('int', ...))
        args_i: Vec<crate::flowspace::model::Variable>,
        /// Reference arguments (RPython: ListOfKind('ref', ...))
        args_r: Vec<crate::flowspace::model::Variable>,
        /// Float arguments (RPython: ListOfKind('float', ...))
        args_f: Vec<crate::flowspace::model::Variable>,
        /// Result kind: 'i', 'r', 'f', or 'v'
        result_kind: char,
        /// Source-order kind chars of `args_i`/`args_r`/`args_f` (`i`/`r`/`f`).
        /// `rebucket_kind_lists` replays this so a promoted int does not jump
        /// ahead of a ref that preceded it.
        arg_classes: String,
    },
    /// Recursive call — back to the portal entry point.
    /// RPython: `recursive_call_{reskind}(jd_index, [green_i], [green_r], [green_f], [red_i], [red_r], [red_f])`
    /// RPython jtransform.py handle_recursive_call.
    RecursiveCall {
        /// RPython: `jitdriver_sd.index`
        jd_index: usize,
        /// Green args (loop-invariant) split by kind
        greens_i: Vec<crate::flowspace::model::Variable>,
        greens_r: Vec<crate::flowspace::model::Variable>,
        greens_f: Vec<crate::flowspace::model::Variable>,
        /// Red args (loop-variant) split by kind
        reds_i: Vec<crate::flowspace::model::Variable>,
        reds_r: Vec<crate::flowspace::model::Variable>,
        reds_f: Vec<crate::flowspace::model::Variable>,
        /// Result kind
        result_kind: char,
        /// Source-order kind chars of the green lists, then the red lists.
        green_classes: String,
        red_classes: String,
    },

    //
    // These correspond to RPython's `_handle_jit_call()` in jtransform.py.
    // The codewriter converts calls to `jit.*` oopspec functions into
    // dedicated opcodes instead of residual calls.
    /// jtransform.py — `jit_debug(string, arg1, arg2, arg3, arg4)`.
    /// Emits debug info into the trace (like debug_merge_point).
    JitDebug {
        args: Vec<crate::flowspace::model::Variable>,
    },
    /// jtransform.py:1733 — `{kind}_assert_green(value)`.
    /// Asserts the value is compile-time constant during tracing.
    AssertGreen {
        value: crate::flowspace::model::Variable,
        kind_char: char,
    },
    /// jtransform.py:1736 — `current_trace_length()`.
    /// Returns the current length of the trace being compiled.
    CurrentTraceLength,
    /// jtransform.py:1738 — `{kind}_isconstant(value)`.
    /// Returns whether the value is currently known to be constant.
    IsConstant {
        value: crate::flowspace::model::Variable,
        kind_char: char,
    },
    /// jtransform.py:1741 — `{kind}_isvirtual(value)`.
    /// Returns whether the value is currently virtualized.
    IsVirtual {
        value: crate::flowspace::model::Variable,
        kind_char: char,
    },
    /// RPython `flowspace` `isinstance(obj, cls)` (annotator/unaryop.py +
    /// `rtyper.rs`'s `translate_operation` dispatch — `Repr.rtype_isinstance`). The
    /// front-end emits this op at `match` cascade sites where the
    /// variant pattern carries a payload (`TupleStruct`) and a
    /// ptr_eq-against-singleton check would be insufficient.
    /// `class_carrier` is typically a `ConstRef`-wrapped vtable
    /// Constant resolved by the rtyper to a CLASSTYPE pointer; the
    /// rtyper then dispatches to
    /// [`InstanceRepr::rtype_isinstance`](crate::translator::rtyper::rclass::InstanceRepr::rtype_isinstance)
    /// which mints either `ll_isinstance_const_{,nonnull}_<flavor>_<class_identity>`
    /// (Constant `class_carrier`) or the generic `ll_isinstance`
    /// (Variable `class_carrier`).
    IsInstance {
        obj: crate::flowspace::model::Variable,
        class_carrier: crate::flowspace::model::Variable,
        result_ty: ValueType,
    },

    // ── Conditional call ops (jtransform.py _rewrite_op_cond_call) ──────
    //
    // RPython: `jit_conditional_call` / `jit_conditional_call_value` llops
    // are rewritten to `conditional_call_{kinds}_{reskind}`.
    /// jtransform.py rewrite_op_jit_conditional_call — `conditional_call_{ir}_{v}`.
    /// If condition is true, call the function. Always produces void.
    /// RPython: `COND_CALL(condition, funcptr, calldescr, args...)`
    ConditionalCall {
        condition: crate::flowspace::model::Variable,
        funcptr: CallTarget,
        descriptor: crate::call::CallDescriptor,
        args_i: Vec<crate::flowspace::model::Variable>,
        args_r: Vec<crate::flowspace::model::Variable>,
        args_f: Vec<crate::flowspace::model::Variable>,
    },
    /// jtransform.py rewrite_op_jit_conditional_call_value — `conditional_call_value_{ir}_{reskind}`.
    /// If value is falsy (0/NULL/None), call the function and return its result.
    /// RPython: `COND_CALL_VALUE(value, funcptr, calldescr, args...)`
    ConditionalCallValue {
        value: crate::flowspace::model::Variable,
        funcptr: CallTarget,
        descriptor: crate::call::CallDescriptor,
        args_i: Vec<crate::flowspace::model::Variable>,
        args_r: Vec<crate::flowspace::model::Variable>,
        args_f: Vec<crate::flowspace::model::Variable>,
        result_kind: char,
    },

    /// jtransform.py rewrite_op_jit_record_known_result — `record_known_result_{i|r}_ir_v`.
    /// Produced by `rewrite_op_jit_record_known_result`; pairs an elidable call
    /// with its known result for constant folding by OptPure.
    /// RPython layout: `record_known_result_{reskind}(result, funcptr, calldescr, [i], [r])`
    RecordKnownResult {
        /// The known result value (arg 0 of the jit_record_known_result llop).
        result_value: crate::flowspace::model::Variable,
        funcptr: CallTarget,
        descriptor: crate::call::CallDescriptor,
        args_i: Vec<crate::flowspace::model::Variable>,
        args_r: Vec<crate::flowspace::model::Variable>,
        args_f: Vec<crate::flowspace::model::Variable>,
        /// 'i' or 'r' — kind of the known result (no float support).
        result_kind: char,
    },

    /// RPython `record_quasiimmut_field(v_inst, fielddescr, mutatefielddescr)`
    /// — `jtransform.py`.  Emitted by `rewrite_op_getfield` when the
    /// field is quasi-immutable; paired with a subsequent pure-read.  The
    /// metainterp/blackhole counterpart (`blackhole.py:1537-1539
    /// bhimpl_record_quasiimmut_field`) is a no-op during blackhole execution
    /// but the descriptors drive guard/invalidation accounting in the
    /// optimizer (`quasiimmut.py`).
    ///
    /// TODO: RPython derives `mutate_field` via
    /// `quasiimmut.get_mutate_field_name(name)` which expects the lltype
    /// `inst_` prefix (`quasiimmut.py`).  Rust structs have no such
    /// prefix, so we use the literal `mutate_<fieldname>` convention.
    RecordQuasiImmutField {
        base: crate::flowspace::model::Variable,
        field: FieldDescriptor,
        mutate_field: FieldDescriptor,
    },

    /// Liveness marker — RPython `-live-` operation.
    /// Inserted by jtransform after calls that may need guard resumption.
    /// Expanded by compute_liveness() to include all values alive at
    /// this point. RPython: jtransform.py:469,481,533.
    Live,

    /// JitDriver merge-point marker — RPython `jit_merge_point` opname.
    /// Emitted by `handle_jit_marker__jit_merge_point` (jtransform.py)
    /// with the portal jitdriver's index and green/red arguments split by
    /// kind (`make_three_lists`). In upstream the args are a flat
    /// `SpaceOperation.args` vec `[index_const, greens_i, greens_r,
    /// greens_f, reds_i, reds_r, reds_f]`; pyre stores them as structured
    /// fields to avoid re-splitting on every consumer.
    JitMergePoint {
        jitdriver_index: usize,
        greens_i: Vec<crate::flowspace::model::Variable>,
        greens_r: Vec<crate::flowspace::model::Variable>,
        greens_f: Vec<crate::flowspace::model::Variable>,
        reds_i: Vec<crate::flowspace::model::Variable>,
        reds_r: Vec<crate::flowspace::model::Variable>,
        reds_f: Vec<crate::flowspace::model::Variable>,
        /// Source-order kind chars of the green lists, then the red lists.
        green_classes: String,
        red_classes: String,
    },

    /// JitDriver loop-header marker — RPython `loop_header` opname.
    /// Emitted by `handle_jit_marker__loop_header` (jtransform.py)
    /// with the jitdriver's index as its single Constant arg.
    /// `can_enter_jit` markers alias to this (jtransform.py).
    LoopHeader {
        jitdriver_index: usize,
    },

    /// Translation marker emitted when a graph operation cannot be lowered
    /// to a canonical opname.
    /// Reaching the op at runtime means tracing or blackhole resume
    /// crossed an untranslatable graph slice; downstream handlers
    /// advance past it (see `blackhole.rs::handler_abort_marker`).
    /// Distinct from RPython's `SwitchToBlackhole` exception path —
    /// RPython aborts before lowering so no equivalent opname exists.
    /// Kept under `kind: UnknownKind` because the same diagnostic enum
    /// is reused by `front::mir`'s `LowerError::Unsupported`.
    Abort {
        kind: UnknownKind,
    },

    /// RPython `BUILD_TUPLE` (`flowspace/flowcontext.py`) /
    /// `newtuple` operation (`operation.py`).  Constructs a
    /// new tuple from N element values; the result is a fresh tuple
    /// object distinct from any individual element.
    NewTuple {
        args: Vec<crate::flowspace::model::Variable>,
    },

    /// RPython `BUILD_LIST` (`flowspace/flowcontext.py`) / `newlist`
    /// operation (`operation.py`).  Constructs a fresh resizable list
    /// from N element values; the result is a new list object distinct
    /// from any individual element.
    NewList {
        args: Vec<crate::flowspace::model::Variable>,
    },

    /// RPython `getslice` operation (`operation.py`) — `l[start:stop]`.
    /// The three operands are `(list, start, stop)`; the annotator's
    /// `getslice` handler (`unaryop.py`) builds a fresh
    /// `listdef.offspring` (a copy, not a view), and the rtyper lowers it
    /// through `AbstractBaseListRepr.rtype_getslice`
    /// (`rlist.py:409-414`) to a `gendirectcall` of the per-kind
    /// `ll_listslice_{startonly,startstop,minusone}` helper — so the op
    /// never survives to the assembler as a bare primitive.  Emitted only
    /// by the gated slice-range front recognizer; a graph that drops to
    /// the legacy walker keeps the residual `slice::index` call instead.
    GetSlice {
        args: Vec<crate::flowspace::model::Variable>,
    },

    /// A pre-lowered, register-shaped blackhole opcode emitted by the
    /// opname-dispatch convergence spine (`codewriter::jtransform_opname`).
    ///
    /// The rtyper lowers certain helper graphs (the `ll_str*` family) to
    /// upstream-shaped low-level `SpaceOperation`s whose opnames map 1:1 onto
    /// a register-only blackhole insn with NO descriptor operand — e.g.
    /// `strlen` (`r>i`), `strgetitem` (`ri>i`), `strsetitem` (`rii`),
    /// `newstr` (`i>r`).  Rather than mint a distinct rich `OpKind` per such
    /// opname (each forcing an arm in every exhaustive `OpKind` match), the
    /// transducer carries the blackhole `opname` and its operand `Variable`s
    /// directly here.  The assembler's default `encode_op` arm emits
    /// `{opname}/{argcodes}`, with operand kinds inferred from each operand's
    /// register bank and the result kind (if any) appended from `op.result`
    /// — matching the `bhimpl_<opname>` argcode shape the runtime handler is
    /// keyed on (`blackhole.rs`).
    ///
    /// Only for opnames whose lowering is a single register-shaped insn with
    /// no descriptor operand; effect classification remains opname-driven
    /// (e.g. `newstr`/`newunicode` are `MemoryErrorOnly` in
    /// `codewriter::call::op_can_raise`).  Descriptor-bearing ops
    /// (getfield/getarrayitem/…) keep their dedicated rich `OpKind`s.
    LoweredBlackholeOp {
        opname: String,
        args: Vec<crate::flowspace::model::Variable>,
    },

    /// A single-segment path resolving to a crate-local `static`
    /// declaration (typically a SHOUTY_CASE constant like
    /// `GC_WEAKREF_BOX_TYPE`, `INT_TYPE`, `MODULE_DICT_TYPE`).
    ///
    /// RPython parity: `flowspace/flowcontext.py:LOAD_GLOBAL` lifts a
    /// module-scope name lookup to a `Constant(value)` whose payload
    /// is the resolved object (`flowspace/flowcontext.py`); the
    /// annotator binds the result via `unionof(s_Constant)` without
    /// emitting a SpaceOperation (the bound `Variable` *is* the
    /// graph-level definition).  Pyre's adapter emits a placeholder
    /// `same_as` SpaceOperation whose first arg is the static's
    /// declared `Hlvalue::Constant` so checkgraph's defining-var
    /// set includes the producer (this differs from PyPy where
    /// `LOAD_GLOBAL` returns a Constant Hlvalue directly and no
    /// SpaceOperation is needed — pyre's frontend always emits an
    /// op so cross-block reads have a defined producer).
    ///
    /// PRE-EXISTING-ADAPTATION: RPython has no concept of a
    /// "static at a crate path" — it has module-global Python names
    /// looked up by `LOAD_GLOBAL`.  The `segments` field carries the
    /// fully-qualified path (`["module_path", "STATIC_NAME"]`)
    /// matching the upstream lookup key shape.
    ///
    /// `value` carries the literal-evaluated initializer
    /// when `extract_static_decls` could fold the RHS to a
    /// `ConstValue` (bool/int/float/string literals, including the
    /// `-LIT` unary-neg shape and the `thread_local!` `const { LIT }`
    /// wrapper).  `Some(v)` reaches the flowspace adapter as the
    /// concrete `Constant(v)` operand of the synthetic `same_as` op,
    /// matching PyPy `LOAD_GLOBAL` (`flowcontext.py`) pushing the
    /// resolved object.  `None` is rejected before JitCode assembly:
    /// RPython has no blackhole `same_as/*` opcode, so unresolved
    /// statics must be folded to constants or lowered through a real
    /// host-evaluator path first.
    LoadStatic {
        segments: Vec<String>,
        ty: ValueType,
        value: Option<crate::flowspace::model::ConstValue>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpaceOperation {
    /// RPython `flowspace/model.py:439 self.result = result      #
    /// either Variable or Constant instance`.  In the
    /// rtyper-orthodox path the canonical writer is
    /// `flowspace/operation.py:75 self.result = Variable()`,
    /// matching the `Option<Variable>` carrier — `None` reserved for
    /// the `direct_call`-shape test fixtures whose result slot is
    /// unbound (`rpython/translator/backendopt/test/test_graphanalyze.py:76`).
    pub result: Option<crate::flowspace::model::Variable>,
    pub kind: OpKind,
}

/// RPython `Block.exitswitch`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExitSwitch {
    Value(crate::flowspace::model::Variable),
    LastException,
    /// `jtransform.py optimize_goto_if_not` fuses a comparison
    /// op into the exitswitch: `block.exitswitch = (opname,) +
    /// tuple(op.args) + ('-live-before',)`.  Carries the RPython
    /// comparison opname (e.g. `int_lt`) and the operands; the trailing
    /// `-live-before` marker is implicit (always present for a fused
    /// guard) and re-applied at emit time (`flatten.py:248-253`).
    Fused {
        opname: String,
        args: Vec<crate::flowspace::model::Variable>,
    },
}

/// RPython `Link.exitcase`.
///
/// Upstream stores the concrete switch value itself here: `False` /
/// `True` for boolean branches, the Python `Exception` class object for
/// catch-all exception links, or a specific exception class object for
/// typed handlers (`flowspace/model.py`,
/// `flowspace/flowcontext.py:127-143`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExitCase {
    Bool(bool),
    Const(ConstValue),
    /// The interpreter's error-carrier class (`crate::ErrorCarrierSpec`),
    /// the program's `OperationError`: a handler written `except
    /// OperationError as e` catches with this exitcase, and `e` is the
    /// carrier.  The class object is minted by the annotator's bookkeeper
    /// (`Bookkeeper::set_exception_carrier`), which the front does not
    /// reach, so the link names it symbolically; the flowspace adapter
    /// resolves it to that class and the codewriter's
    /// `error_carrier_edges` rewrites it to the runtime domain.
    ErrorCarrier,
}

/// RPython `flowspace/model.py` `Link`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Link {
    pub args: Vec<LinkArg>,
    pub target: BlockId,
    pub exitcase: Option<ExitCase>,
    /// RPython `Link.prevblock` — the block this Link exits from.
    pub prevblock: Option<BlockId>,
    /// RPython `Link.llexitcase` — the low-level value matched by
    /// `goto_if_exception_mismatch` (`flatten.py`).  For
    /// typed exception links this is the rtyper-produced class
    /// identity constant; pyre carries it as a full `ConstValue` so
    /// non-Int llexitcase shapes (`lltype.Ptr`, host class objects)
    /// round-trip to the backend intact.
    pub llexitcase: Option<ConstValue>,
    pub last_exception: Option<LinkArg>,
    pub last_exc_value: Option<LinkArg>,
}

impl Link {
    /// `Variable`-typed constructor — direct counterpart to RPython
    /// `Link(args=[Hlvalue], target=Block, exitcase=...)` where each
    /// `Hlvalue` is the upstream `Variable` instance pulled from an
    /// `op.result` / `Block.inputargs` slot.  Mirrors the upstream
    /// `flowspace/model.py` arity assert via `graph` lookup
    /// for `target.inputargs`.
    pub fn from_variables(
        graph: &FunctionGraph,
        args: Vec<crate::flowspace::model::Variable>,
        target: BlockId,
        exitcase: Option<ExitCase>,
    ) -> Self {
        assert_eq!(
            args.len(),
            graph.block(target).inputargs.len(),
            "output args mismatch"
        );
        Self::new_mixed(
            args.into_iter().map(LinkArg::Value).collect(),
            target,
            exitcase,
        )
    }

    pub fn new_mixed(args: Vec<LinkArg>, target: BlockId, exitcase: Option<ExitCase>) -> Self {
        Self {
            args,
            target,
            exitcase,
            prevblock: None,
            llexitcase: None,
            last_exception: None,
            last_exc_value: None,
        }
    }

    pub fn with_prevblock(mut self, prevblock: BlockId) -> Self {
        self.prevblock = Some(prevblock);
        self
    }

    pub fn with_llexitcase(mut self, llexitcase: ConstValue) -> Self {
        self.llexitcase = Some(llexitcase);
        self
    }

    /// Structural counterpart of RPython rtyper's
    /// `convert_link()` (`rtyper.py`): copy the flow-level
    /// `exitcase` into the low-level `llexitcase` slot for primitive
    /// branch values.  The `"default"` switch sentinel is not a
    /// low-level case and stays `None`.
    pub fn with_llexitcase_from_exitcase(mut self) -> Self {
        self.llexitcase = match &self.exitcase {
            Some(ExitCase::Bool(value)) => Some(ConstValue::Bool(*value)),
            Some(ExitCase::Const(value)) if value.string_eq("default") => None,
            Some(ExitCase::Const(value)) => Some(value.clone()),
            // A class exitcase is converted by the rtyper
            // (`get_type_repr(...).convert_const`), not copied.
            Some(ExitCase::ErrorCarrier) => None,
            None => None,
        };
        self
    }

    pub fn extravars(
        mut self,
        last_exception: Option<LinkArg>,
        last_exc_value: Option<LinkArg>,
    ) -> Self {
        self.last_exception = last_exception;
        self.last_exc_value = last_exc_value;
        self
    }

    /// RPython `flatten.py` `if link.exitcase is Exception`.
    pub fn catches_all_exceptions(&self) -> bool {
        self.exitcase == Some(exception_exitcase())
    }
}

pub fn exception_exitcase() -> ExitCase {
    ExitCase::Const(ConstValue::builtin("Exception"))
}

/// `except OperationError` — see [`ExitCase::ErrorCarrier`].
pub fn error_carrier_exitcase() -> ExitCase {
    ExitCase::ErrorCarrier
}

/// RPython `Link.args` items are Variables or Constants —
/// `Link.args: List[Hlvalue]` (`flowspace/model.py`) where
/// `Hlvalue = Variable | Constant`.  Both arms carry the
/// upstream-orthodox flowspace handles directly: `Value` wraps a
/// [`crate::flowspace::model::Variable`] (concretetype on the inline
/// cell) and `Const` wraps a [`crate::flowspace::model::Constant`]
/// (concretetype on the struct field) so type-sensitive renaming /
/// Void filtering at link sites can read kinds off either arm without
/// projecting through a side table.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LinkArg {
    Value(crate::flowspace::model::Variable),
    Const(crate::flowspace::model::Constant),
}

impl LinkArg {
    /// Read the backing [`crate::flowspace::model::Variable`] for a
    /// `LinkArg::Value`; `None` for constants.  Direct over the
    /// upstream-orthodox storage — no graph projection needed.
    ///
    /// RPython parity: `Link.args` upstream is `List[Hlvalue]` where
    /// each `Hlvalue::Variable` carries the operand identity inline
    /// (`flowspace/model.py`).
    pub fn as_variable(&self) -> Option<&crate::flowspace::model::Variable> {
        match self {
            Self::Value(var) => Some(var),
            Self::Const(_) => None,
        }
    }

    /// The Variable arm. Constants panic — call sites that have not
    /// yet grown a Constant case use this while `Call.args` migrates
    /// to the mixed `SpaceOperation.args` shape.
    pub fn expect_variable(&self) -> &crate::flowspace::model::Variable {
        self.as_variable()
            .expect("Call.args entry is a Constant; handle LinkArg::Const")
    }

    /// Owned Variable arm.
    pub fn into_variable(self) -> crate::flowspace::model::Variable {
        match self {
            Self::Value(var) => var,
            Self::Const(_) => panic!("Call.args entry is a Constant; handle LinkArg::Const"),
        }
    }

    /// Remap the backing `Variable` of a `LinkArg::Value` through
    /// `remap`, leaving a `LinkArg::Const` literal untouched.  Mirrors
    /// the `remap_link_arg` closure in
    /// [`remap_control_flow_metadata_var`] and RPython
    /// `Hlvalue.replace(mapping)` where a `Constant` returns itself.
    pub fn map_value(
        &self,
        remap: impl FnOnce(&crate::flowspace::model::Variable) -> crate::flowspace::model::Variable,
    ) -> Self {
        match self {
            Self::Value(var) => Self::Value(remap(var)),
            Self::Const(value) => Self::Const(value.clone()),
        }
    }
}

impl From<ConstValue> for LinkArg {
    /// Wrap a raw [`ConstValue`] in a [`crate::flowspace::model::Constant`]
    /// with no concretetype attached.  Mirrors RPython
    /// `Constant(value)` (`flowspace/model.py __init__`) where
    /// `concretetype` defaults to `None` until the rtyper sets it.
    fn from(value: ConstValue) -> Self {
        Self::Const(crate::flowspace::model::Constant::new(value))
    }
}

impl From<crate::flowspace::model::Constant> for LinkArg {
    fn from(c: crate::flowspace::model::Constant) -> Self {
        Self::Const(c)
    }
}

impl std::ops::Deref for LinkArg {
    type Target = crate::flowspace::model::Variable;
    fn deref(&self) -> &Self::Target {
        self.expect_variable()
    }
}

impl PartialEq<crate::flowspace::model::Variable> for LinkArg {
    fn eq(&self, other: &crate::flowspace::model::Variable) -> bool {
        self.as_variable() == Some(other)
    }
}

impl PartialEq<LinkArg> for crate::flowspace::model::Variable {
    fn eq(&self, other: &LinkArg) -> bool {
        other == self
    }
}

impl From<crate::flowspace::model::Variable> for LinkArg {
    /// RPython `SpaceOperation.args` is a mixed list of Variable /
    /// Constant (`flowspace/model.py SpaceOperation.__init__`).
    fn from(v: crate::flowspace::model::Variable) -> Self {
        Self::Value(v)
    }
}

/// Collect call operands into the mixed `SpaceOperation.args` shape.
pub fn call_args(
    vars: impl IntoIterator<Item = crate::flowspace::model::Variable>,
) -> Vec<LinkArg> {
    vars.into_iter().map(LinkArg::from).collect()
}

/// Project Variable operands out of mixed call args.
/// Constants are skipped — RPython `isinstance(a, Variable)` filter.
pub fn call_arg_vars(args: &[LinkArg]) -> Vec<crate::flowspace::model::Variable> {
    args.iter()
        .filter_map(LinkArg::as_variable)
        .cloned()
        .collect()
}

/// `simple_call(__cast_instance_intrinsic, operand, const(root))`.
///
/// The root is a trailing `ByteStr` Constant —
/// `annotator/builtin.rs cast_instance_intrinsic` reads `args[1]`.
pub fn cast_instance_call(
    root: impl Into<String>,
    operand: crate::flowspace::model::Variable,
) -> OpKind {
    let root = root.into();
    cast_instance_call_result(root.clone(), operand, ValueType::Ref(Some(root)))
}

/// Same as [`cast_instance_call`] with an explicit result type.
/// Niche-null string payloads use `ValueType::Str` rather than `Ref`.
pub fn cast_instance_call_result(
    root: impl Into<String>,
    operand: crate::flowspace::model::Variable,
    result_ty: ValueType,
) -> OpKind {
    OpKind::Call {
        target: CallTarget::function_path([crate::runtime_names::shims::CAST_INSTANCE]),
        args: vec![
            LinkArg::from(operand),
            LinkArg::from(ConstValue::byte_str(root.into())),
        ],
        result_ty,
    }
}

/// The target-struct root of a [`cast_instance_call`], if `kind` is one.
pub fn cast_instance_root(kind: &OpKind) -> Option<&str> {
    let OpKind::Call {
        target: CallTarget::FunctionPath { segments, .. },
        args,
        ..
    } = kind
    else {
        return None;
    };
    if segments.as_slice() != [crate::runtime_names::shims::CAST_INSTANCE] {
        return None;
    }
    match args.get(1) {
        Some(LinkArg::Const(c)) => c.value.as_pystr(),
        _ => None,
    }
}

/// `cast_pointer(PTRTYPE, ptr)` marker (`lltype.py cast_pointer`).
/// The target class is a trailing `ByteStr` Constant; the adapter
/// interns it through the bookkeeper.
pub fn cast_pointer_call(
    root: impl Into<String>,
    operand: crate::flowspace::model::Variable,
) -> OpKind {
    let root = root.into();
    OpKind::Call {
        target: CallTarget::function_path(["__cast_pointer"]),
        args: vec![
            LinkArg::from(operand),
            LinkArg::from(ConstValue::byte_str(&root)),
        ],
        result_ty: ValueType::Ref(Some(root)),
    }
}

/// The target-struct root of a [`cast_pointer_call`], if `kind` is one.
pub fn cast_pointer_root(kind: &OpKind) -> Option<&str> {
    let OpKind::Call {
        target: CallTarget::FunctionPath { segments, .. },
        args,
        ..
    } = kind
    else {
        return None;
    };
    if segments.as_slice() != ["__cast_pointer"] {
        return None;
    }
    match args.get(1) {
        Some(LinkArg::Const(c)) => c.value.as_pystr(),
        _ => None,
    }
}

/// [`cast_instance_root`] when `args[0]` is `operand`.
pub fn cast_instance_of<'a>(
    kind: &'a OpKind,
    operand: &crate::flowspace::model::Variable,
) -> Option<&'a str> {
    let root = cast_instance_root(kind)?;
    let OpKind::Call { args, .. } = kind else {
        return None;
    };
    (args.first().and_then(LinkArg::as_variable) == Some(operand)).then_some(root)
}

/// Struct owner a [`cast_instance_call`] names for layout lookup.
///
/// The constant is the class key. When the front knows the declaration
/// path it records that path as `Ref(Some(path))` on the result — the
/// spelling [`crate_relative_struct_owner`] gives a constructor — and
/// this returns that path. A result that is not a named reference keeps
/// the constant, which is what [`cast_instance_call`] paints into both.
pub(crate) fn cast_instance_layout_owner(kind: &OpKind) -> Option<&str> {
    let root = cast_instance_root(kind)?;
    let OpKind::Call { result_ty, .. } = kind else {
        return Some(root);
    };
    match result_ty {
        ValueType::Ref(Some(owner)) if !owner.is_empty() => Some(owner.as_str()),
        _ => Some(root),
    }
}

/// A basic block in the control flow graph.
///
/// RPython equivalent: `flowspace/model.py Block` — slots
/// `inputargs operations exitswitch exits`.  Upstream has no separate
/// terminator surface: fall-through is `exitswitch=None` with a single
/// `Link`, bool branches are `exitswitch=Variable` with two Links
/// carrying `Bool(false)`/`Bool(true)` exitcases, can-raise is
/// `exitswitch=c_last_exception`, and final blocks have `exits=()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    pub id: BlockId,
    /// Phi-node inputs: values provided by incoming Links.
    ///
    /// RPython parity: `Block.inputargs: List[Variable]`
    /// (`flowspace/model.py Block([Variable("etype"),
    /// Variable("evalue")])`) — each predecessor Link carries
    /// values that map 1:1 to these inputargs.  Pyre stores the
    /// upstream-orthodox `Variable` directly; consumers match on the
    /// `Variable` object by identity.
    pub inputargs: Vec<crate::flowspace::model::Variable>,
    pub operations: Vec<SpaceOperation>,
    /// RPython `Block.exitswitch`.
    pub exitswitch: Option<ExitSwitch>,
    /// RPython `Block.exits`.
    pub exits: Vec<Link>,
    /// RPython `flowspace/flowcontext.py mergeblock`: when a
    /// `SpamBlock` is generalised by a later `union`, the prior
    /// candidate is marked `block.dead = True` and `block.operations
    /// = ()` and its exits are recloseblock'd to forward to the new
    /// SpamBlock.  `simplify.eliminate_empty_blocks` then collapses
    /// the dead-block forwarding chain into a single multi-incoming
    /// merge block.  Pyre's tree-recursive `Expr::Match` /
    /// `Expr::If` lowering produces the post-collapse CFG directly
    /// today (no chain to collapse), so the field is always `false`
    /// in the AST graph; it becomes load-bearing once the Z4
    /// flowcontext-walker rewrite materialises intermediate
    /// SpamBlocks per fold step.
    pub dead: bool,
    /// Framestate snapshot captured at the moment this block was closed
    /// (its `set_goto` / `set_branch` / `set_return`).  Read by the
    /// frontend's lazy cross-block local installer to recover, after the
    /// fact, what `(name → Variable)` mapping was visible to a
    /// predecessor when its outgoing link fired.  Build-time only —
    /// downstream rtyper/jtransform passes ignore this field; it is
    /// neither serialised nor required to remain populated past the
    /// front end.  RPython parity: `flowspace/flowcontext.py SpamBlock
    /// SpamBlock.__init__ self.framestate = framestate` attaches a
    /// framestate to the block representing the locals state visible at
    /// that block boundary; pyre stores the analogous "exit-time"
    /// snapshot here so cross-block reads in successors can derive
    /// `Link.args` retroactively.
    pub framestate: Option<FrameState>,
}

/// Locals snapshot captured at a block boundary — see `Block.framestate`
/// for usage semantics.  Entries are stored **densely** at graph-wide
/// first-bind slot positions: slot `i` corresponds to the name at
/// `GraphBuildContext::local_first_bind_order[i]`, and `None` indicates
/// the slot is unbound at this snapshot point (RPython's "undefined
/// local" sentinel).  Names are appended to the graph-wide order on
/// first bind anywhere in the function and never moved or removed —
/// `LocalBindingSnapshot::restore` does NOT roll the order back, so the
/// slot index of a name is truly invariant across the entire lowering.
/// This mirrors RPython `co_varnames` slot order — every predecessor's
/// exit snapshot walks the same slot positions, so two predecessors of
/// the same merge point line up positionally even when one of them
/// rolled back its bindings via `LocalBindingSnapshot::restore`.
///
/// RPython parity: `flowspace/framestate.py FrameState` — a tuple of
/// `(locals_w, stack, last_exception, blocklist, next_offset)`.  Pyre's
/// graphs are lowered from Charon ULLBC rather than Python bytecode, so
/// the `stack` / `last_exception` / `blocklist` / `next_offset`
/// projections are vestigially empty.
///
/// All five fields are present in the struct now so
/// shape parity is locked in at the model layer; downstream merging
/// passes already thread them via `union`.  The flowcontext walker
/// (`flowspace::flowcontext::FlowContext`) and `flowspace::framestate::
/// FrameState` are the upstream-orthodox surfaces this struct is
/// converging toward — eventually flowspace graphs are produced directly
/// and this slot-indexed `FrameState` retires.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FrameState {
    /// Slot `i` ↔ graph-wide first-bind name at index `i`; `None` if
    /// unbound at this snapshot point.  RPython parity:
    /// `framestate.py __init__ self.locals_w` — list of `Variable | Constant
    /// | None` indexed by `co_varnames` slot.  Carries upstream
    /// `Variable` identity directly; the parallel `locals_w`
    /// `Vec<Option<Hlvalue>>` carrier (next field) still distinguishes
    /// `Hlvalue::Constant` cells which this Variable-only carrier
    /// collapses to `None`.
    pub entries: Vec<Option<crate::flowspace::model::Variable>>,
    /// Parallel `Hlvalue` carrier matching upstream
    /// `framestate.py __init__ self.locals_w` shape — list of
    /// `Variable | Constant | None` indexed by `co_varnames` slot.
    /// Populated by `getstate` / `union` alongside `entries`; the
    /// long-term plan is to promote this to the single source of truth
    /// and retire the per-entry `Variable` carrier in `entries`.  Production read
    /// sites still consume `entries`; future substeps swap them over
    /// slot-by-slot before flipping the construction direction
    /// (fixtures populate `locals_w` directly, `entries` becomes
    /// the derived projection).
    ///
    /// Fixtures that build a `FrameState` by hand keep this empty;
    /// only `FrameState::union` populates it today.  Read sites that
    /// rely on it must therefore route through a unioned state, not
    /// a hand-built fixture.
    pub locals_w: Vec<Option<crate::flowspace::model::Hlvalue>>,
    /// `framestate.py:21 self.stack` — value-stack content at the
    /// snapshot point.  Empty for AST-frontend snapshots until the
    /// flowcontext-style walker introduces stack push/pop on Expr
    /// nodes; the `union` invariant requires both predecessors agree
    /// on stack content (upstream `framestate.py _union(self.stack,
    /// other.stack)` zips positionally — equal-length always for any
    /// program reachable to a join).
    pub stack: Vec<crate::flowspace::framestate::StackElem>,
    /// `framestate.py:22 self.last_exception` — pending FSException at
    /// the snapshot point.  None for AST-frontend snapshots until
    /// the flowcontext-style walker introduces exception handling.
    pub last_exception: Option<crate::flowspace::model::FSException>,
    /// `framestate.py:23 self.blocklist` — block-stack snapshot
    /// (`SETUP_*` / `POP_BLOCK` depth at the snapshot point).  Empty
    /// for AST-frontend snapshots until the flowcontext-style walker
    /// introduces frame-block management.  `framestate.py matches` asserts
    /// blocklist equality across merge candidates as a precondition.
    pub blocklist: Vec<crate::flowspace::flowcontext::FrameBlock>,
    /// `framestate.py:24 self.next_offset` — bytecode offset resumed
    /// at after the snapshot.  `0` for AST-frontend snapshots until
    /// the flowcontext-style walker uses an AST-node index (the equivalent of a
    /// virtual-bytecode tape position).  `framestate.py matches`
    /// asserts next_offset equality across merge candidates as a
    /// precondition.
    pub next_offset: i64,
}

impl FrameState {
    /// Read the backing `Variable` bound at `entries[slot]`.  `Some`
    /// when the slot is bound; `None` for unbound (None-killed) slots
    /// or out-of-range indices.  `entries` holds `Variable`s directly,
    /// so this is a plain positional read.
    pub fn entry_var(
        &self,
        slot: usize,
        _graph: &FunctionGraph,
    ) -> Option<crate::flowspace::model::Variable> {
        self.entries.get(slot).cloned().flatten()
    }

    /// Authoritative locals view — upstream `framestate.py __init__ self.locals_w`
    /// IS the locals source-of-truth.  Pyre's `union` / `getstate`
    /// constructors populate `self.locals_w` in lockstep with
    /// `self.entries` so production callers consult the `Hlvalue` carrier
    /// directly via this view.  Hand-built fixtures that pre-date the
    /// Z4.A.6 carrier swap may leave `self.locals_w` empty; in that case
    /// derive the view from `self.entries` here so a unioned and a
    /// fixture-built FrameState produce the same `mergeable` projection
    /// without each read site re-implementing the fallback.  The
    /// derivation mirrors what `FrameState::union` and
    /// `GraphBuildContext::getstate` already do, so the result is
    /// bit-identical to the carrier they would have produced.
    pub(crate) fn locals_w_view<'a>(
        &'a self,
        graph: &FunctionGraph,
    ) -> std::borrow::Cow<'a, [Option<crate::flowspace::model::Hlvalue>]> {
        if self.locals_w.len() == self.entries.len() {
            std::borrow::Cow::Borrowed(&self.locals_w)
        } else {
            std::borrow::Cow::Owned(
                (0..self.entries.len())
                    .map(|i| {
                        self.entry_var(i, graph)
                            .map(crate::flowspace::model::Hlvalue::Variable)
                    })
                    .collect(),
            )
        }
    }

    /// Compute a state at least as general as both `self` and `other`
    /// via positional zip over slots — direct port of RPython
    /// `framestate.py _union` (`return [union(v1, v2) for v1, v2
    /// in zip(seq1, seq2)]`) and `framestate.py FrameState.union`.
    /// When one side has fewer slots than the other (a name appended
    /// after the shorter side's snapshot was taken), the missing
    /// slots are treated as `None` so the positional zip extends to
    /// `max(len(self), len(other))`.
    ///
    /// Per-slot semantics (RPython `framestate.py union`):
    ///   - `(None, _) | (_, None)` → None-kill (`framestate.py:
    ///     110-111`); merged slot is `None`.
    ///   - `(Some(s), Some(o))` with matching Variable identity →
    ///     carry through that Variable (`framestate.py if w1 == w2:
    ///     return w1`).
    ///   - `(Some(s), Some(o))` with disagreeing Variables →
    ///     `graph.alloc_value_var()` for a fresh slot at this
    ///     position (`framestate.py return Variable()`
    ///     analogue).
    ///
    /// Type unification is NOT performed here — upstream's per-slot
    /// `union(w1, w2)` (`framestate.py`) compares Hlvalue
    /// identity only, with type-side reconciliation deferred to the
    /// annotator (`annrpython.py`).  Pyre follows the same convention:
    /// callers query types via `FunctionGraph::concretetype_of(&var)`
    /// at the point of use; the prior per-slot `value_type` field on
    /// `FrameStateEntry` was a deviation that has been retired.
    ///
    /// Returns `Some(merged_state)` when the union succeeds; `None`
    /// when any per-projection union raises `UnionError`
    /// (`framestate.py try: ... except UnionError: return None`).
    /// The `None` return propagates upstream's "merge candidates
    /// disagree, fall back to a fresh SpamBlock" path at
    /// `flowcontext.py mergeblock`.  Agreement slots in the
    /// locals projection carry the predecessor's Variable, disagreement
    /// slots get a freshly-allocated Variable (the upstream `Variable()`
    /// analogue).  Callers detect "this slot is a fresh phi" by
    /// comparing the merged Variable identity against the predecessor's
    /// for the same slot.  The install is then driven via
    /// `lazy_install_local_at_current_block(.., Some(merged_var))` so
    /// the Input op carries the same Variable the merged state already
    /// refers to.
    pub fn union(&self, other: &FrameState, graph: &mut FunctionGraph) -> Option<FrameState> {
        // Line-by-line port of `framestate.py FrameState.union`:
        //
        //     def union(self, other):
        //         try:
        //             locals = _union(self.locals_w, other.locals_w)
        //             stack = _union(self.stack, other.stack)
        //             if self.last_exception is None and other.last_exception is None:
        //                 exc = None
        //             else:
        //                 args1 = self._exc_args()
        //                 args2 = other._exc_args()
        //                 exc = FSException(union(args1[0], args2[0]),
        //                         union(args1[1], args2[1]))
        //         except UnionError:
        //             return None
        //         return FrameState(locals, stack, exc, self.blocklist, self.next_offset)
        //
        // Upstream `union` does NOT compare `self.blocklist` /
        // `self.next_offset` against `other.*` — those equality
        // assertions live in `framestate.py matches`, which is a
        // separate predicate the caller invokes AFTER union.  Pyre's
        // frontend keeps both projections at trivial defaults
        // today (empty Vec, next_offset 0); even when they diverge,
        // upstream still produces a merged FrameState that just
        // carries `self.{blocklist,next_offset}` and lets `matches`
        // catch the mismatch downstream.
        //
        // Body order (`framestate.py`): locals → stack →
        // exception, all inside the try/except envelope.  Pyre
        // **reorders** to (stack → exception → locals): the locals
        // fold is total (Variable domain has no UnionError analogue)
        // but invokes `graph.alloc_value_var()` which advances a
        // global allocator counter.  Stack and exception unions CAN
        // return `None`/`Err` once the Z4 walker activates real
        // cells; doing them first keeps `union` atomic — no
        // `alloc_value_var` writes unless the whole merge succeeds.
        // Upstream's `Variable()`
        // construction is side-effect-free (per-instance object), so
        // the reorder is a Rust-side atomic-safety adaptation with
        // no observable upstream-parity divergence.

        // `framestate.py stack = _union(self.stack, other.stack)`.
        // `flowspace::framestate::union_stack` returns `Err(UnionError)`
        // on stack length disagreement, SpecTag mismatch, or
        // FlowSignal-type mismatch — propagate as `None` per upstream's
        // try/except envelope (`framestate.py`).
        let stack = crate::flowspace::framestate::union_stack(&self.stack, &other.stack).ok()?;
        // `framestate.py:81-87`: both `last_exception` None → None;
        // otherwise FSException carries `union(args1[i], args2[i])` for
        // `(w_type, w_value)`.  Per-cell `union` returns `Err(UnionError)`
        // on SpecTag mismatch — propagate as `None`.  When `union()`
        // returns `Ok(None)` for an exception slot (one side undefined-
        // local), we propagate as `None` too — exception slot `None`
        // is not a valid merge result because both sides carry
        // `Constant(None)` sentinels through `_exc_args`, so an
        // undefined-local result here would only arise from a sentinel-
        // vs-non-sentinel mix, which signals the same kind of merge
        // refusal upstream returns None for.
        let last_exception = if self.last_exception.is_none() && other.last_exception.is_none() {
            None
        } else {
            let a = exc_args(&self.last_exception);
            let b = exc_args(&other.last_exception);
            let w_type = crate::flowspace::framestate::union(Some(&a[0]), Some(&b[0])).ok()??;
            let w_value = crate::flowspace::framestate::union(Some(&a[1]), Some(&b[1])).ok()??;
            Some(crate::flowspace::model::FSException::new(w_type, w_value))
        };
        // `framestate.py locals = _union(self.locals_w, other.locals_w)`.
        // Run LAST so the locals projection is computed only after the
        // failure-prone stack / exception projections above have
        // succeeded (each `.ok()?` aborts the whole merge on a per-cell
        // `UnionError`).
        //
        // Direct line-by-line port: walk `self.locals_w` and
        // `other.locals_w` positionally and dispatch each cell pair
        // through `flowspace::framestate::union` (Hlvalue-domain
        // per-cell union — `framestate.py`).  Upstream
        // `_union(seq1, seq2)` zips equal-length sequences; pyre's
        // frontend can grow the locals projection across the
        // sub-block boundary when a name is first bound on one side,
        // so we extend to `max(len, len)` and treat the trailing
        // unpadded slot as `None` (per-cell `union` returns
        // `Ok(None)` when either side is `None`, matching upstream
        // `framestate.py` undefined-local kill).
        let self_view = self.locals_w_view(graph);
        let other_view = other.locals_w_view(graph);
        let len = std::cmp::max(self_view.len(), other_view.len());
        // Per `framestate.py`, a `UnionError` raised by ANY
        // per-cell `union(w1, w2)` is caught at the FrameState level
        // and the whole `FrameState.union` returns `None`.  Collect
        // into `Result<Vec<_>, UnionError>` and propagate failure via
        // `.ok()?` so a SpecTag/FlowSignal disagreement in the locals
        // projection aborts the merge rather than silently None-
        // killing the offending slot.
        let locals_w: Vec<Option<crate::flowspace::model::Hlvalue>> = (0..len)
            .map(|i| {
                let w1 = self_view.get(i).and_then(|c| c.as_ref());
                let w2 = other_view.get(i).and_then(|c| c.as_ref());
                crate::flowspace::framestate::union(w1, w2)
            })
            .collect::<Result<Vec<_>, _>>()
            .ok()?;
        drop(self_view);
        drop(other_view);
        // Derive `entries` from `locals_w` — `locals_w` is now the
        // primary carrier matching `framestate.py __init__ self.locals_w`,
        // and `entries` is a backward-compatibility view for callers
        // that have not yet migrated.  `Hlvalue::Variable(v)` is
        // cloned directly into the entry, carrying the merged value's
        // identity (`framestate.py/113-114 union` returns either a
        // carry-through Variable or a fresh phi `Variable()`).
        // `Hlvalue::Constant(_)` cells carry no Variable identity
        // (Constants are not Variable-bound in the pyre IR), so they
        // collapse to `None` in the `entries` view — matching the
        // prior Variable-only locals fold's lossy projection on the
        // few sites where a Constant could land in locals via
        // `setstate`.
        let entries: Vec<Option<crate::flowspace::model::Variable>> = locals_w
            .iter()
            .map(|slot| {
                slot.as_ref().and_then(|hv| match hv {
                    crate::flowspace::model::Hlvalue::Variable(v) => Some(v.clone()),
                    crate::flowspace::model::Hlvalue::Constant(_) => None,
                })
            })
            .collect();
        Some(FrameState {
            entries,
            locals_w,
            stack,
            last_exception,
            blocklist: self.blocklist.clone(),
            next_offset: self.next_offset,
        })
    }

    /// Output arguments to thread `self` (a predecessor's exit state)
    /// into the merge block whose entry framestate is `target`.  RPython
    /// parity: `framestate.py FrameState.getoutputargs` walks
    /// `targetstate.mergeable` positionally, picking `self.mergeable[i]`
    /// at every Variable position.  `target.entries` is dense at the
    /// graph-wide first-bind slot positions, with `None` at slots that
    /// were None-killed by the union; surviving (`Some`) slot `i` lines
    /// up with `self.entries[i]` because both predecessors share the
    /// same graph-wide first-bind order.
    ///
    /// Panics if `target` contains a `Some` slot at an index where
    /// `self.entries[i]` is `None` or out of range — that would mean
    /// `target` was not produced by `self.union(_)`, which violates the
    /// merge invariant.
    pub fn getoutputargs(&self, target: &FrameState, graph: &FunctionGraph) -> Vec<LinkArg> {
        self.try_getoutputargs(target, graph).expect(
            "getoutputargs: target Variable slot unbound in self / stack-length mismatch \
             — union invariant violated",
        )
    }

    /// Fallible [`Self::getoutputargs`].  Returns `None` instead of
    /// panicking when a target `Variable` slot has no corresponding
    /// `Some` cell in `self` (locals) or no positional cell in `self`'s
    /// flattened stack.  For union-derived target states this never
    /// happens (union None-kills a slot unless every predecessor binds
    /// it), so [`Self::getoutputargs`] keeps its panicking contract; the
    /// cyclic framestate path pre-seeds loop-header entries with live-in
    /// phis that bypass the union, so it threads links through this
    /// checked variant and declines a phantom-slot mismatch to the
    /// monotonic fallback.
    pub fn try_getoutputargs(
        &self,
        target: &FrameState,
        graph: &FunctionGraph,
    ) -> Option<Vec<LinkArg>> {
        // Line-by-line port of `framestate.py getoutputargs`:
        //
        //     def getoutputargs(self, targetstate):
        //         result = []
        //         mergeable = self.mergeable
        //         for i, w_target in enumerate(targetstate.mergeable):
        //             if isinstance(w_target, Variable):
        //                 result.append(mergeable[i])
        //         return result
        //
        // Upstream returns `List[Hlvalue]` (Variable | Constant cells).
        // Pyre's matching IR carrier is `LinkArg`
        // (`Value(Variable) | Const(Constant)`) — a closed sum that
        // aligns 1:1 with the upstream cell domain.  Hlvalue→LinkArg
        // routing per cell is centralised in `hlvalue_to_linkarg`:
        //   - `Hlvalue::Variable(v)` → `LinkArg::Value(v.clone())` —
        //     the Variable identity carries through inline, so a
        //     single upstream `Variable` reaches every Link threading
        //     it without a separate slot allocation.
        //   - `Hlvalue::Constant(c)` → `LinkArg::Const(c.clone())` —
        //     direct carry, no synthetic op or slot required.
        //
        // Upstream's `mergeable` is `locals_w + recursively_flatten(
        // stack) + [exc_type, exc_value]`.  Pyre walks the same three
        // projections in the same order so the positional mapping
        // between target and self is preserved across the locals→
        // stack→exception boundary.
        //
        // Locals projection consults `locals_w` (the `Hlvalue` carrier
        // matching `framestate.py self.locals_w`) for both the
        // target's Variable predicate and self's cell contribution —
        // same shape as the stack and exception projections below.
        // `locals_w_view` returns the populated `Hlvalue` slice for
        // union/getstate-derived states and derives it from `entries`
        // for hand-built fixtures that pre-date the Z4.A.6 swap.
        let mut result: Vec<LinkArg> = Vec::new();
        // (1) locals projection — `framestate.mergeable` head.
        let target_locals_view = target.locals_w_view(graph);
        let self_locals_view = self.locals_w_view(graph);
        for (i, w_target) in target_locals_view.iter().enumerate() {
            if matches!(
                w_target,
                Some(crate::flowspace::model::Hlvalue::Variable(_))
            ) {
                let w_self = self_locals_view.get(i).and_then(|c| c.as_ref())?;
                result.push(hlvalue_to_linkarg(w_self));
            }
        }
        // (2) stack projection — `recursively_flatten(stack)` middle
        // segment.  Walk both flattened stacks in step so position-`i`
        // lines up between the target and self mergeable views; push
        // `self_flat_stack[i]` (routed through `hlvalue_to_linkarg`)
        // at every position where the target cell is a `Variable`.
        let target_flat_stack = crate::flowspace::framestate::recursively_flatten(&target.stack);
        let self_flat_stack = crate::flowspace::framestate::recursively_flatten(&self.stack);
        for (i, w_target) in target_flat_stack.iter().enumerate() {
            if matches!(w_target, crate::flowspace::model::Hlvalue::Variable(_)) {
                let w_self = self_flat_stack.get(i)?;
                result.push(hlvalue_to_linkarg(w_self));
            }
        }
        // (3) exception args projection — `[exc_type, exc_value]`
        // tail per `framestate.py mergeable`.  `exc_args`
        // substitutes `Constant(None)` sentinels when no exception is
        // pending, so the Variable predicate skips empty-exception
        // states; non-empty exception cells get the same Hlvalue→
        // LinkArg routing as the stack projection.
        let target_exc = exc_args(&target.last_exception);
        let self_exc = exc_args(&self.last_exception);
        for (w_target, w_self) in target_exc.iter().zip(self_exc.iter()) {
            if matches!(w_target, crate::flowspace::model::Hlvalue::Variable(_)) {
                result.push(hlvalue_to_linkarg(w_self));
            }
        }
        Some(result)
    }

    /// Enumerate every `Variable` cell across the full mergeable
    /// projection — RPython `framestate.py getvariables` parity:
    ///
    /// ```python
    /// def getvariables(self):
    ///     return [w for w in self.mergeable if isinstance(w, Variable)]
    /// ```
    ///
    /// Walks `locals + recursively_flatten(stack) + [exc_type, exc_value]`
    /// in that order — the same `mergeable` shape `getoutputargs` traverses
    /// — and returns every Variable cell in positional order.  Used at
    /// merge-block construction (`SpamBlock(framestate)` upstream) so the
    /// block's `inputargs` line up 1:1 with `getoutputargs(target=self)`
    /// from any predecessor.
    ///
    /// Locals projection consults `self.locals_w` per upstream
    /// `framestate.py getvariables`'s `mergeable` walk —
    /// `mergeable`'s locals head IS `locals_w`.  `locals_w_view`
    /// returns the populated `Hlvalue` slice for union/getstate-derived
    /// states and derives it from `entries` for hand-built fixtures
    /// that pre-date the Z4.A.6 swap.  Stack and exception
    /// projections carry `Hlvalue` directly and route through the
    /// same Variable filter as upstream.
    pub fn getvariables(&self, graph: &FunctionGraph) -> Vec<crate::flowspace::model::Variable> {
        use crate::flowspace::model::Hlvalue;
        let mut out: Vec<crate::flowspace::model::Variable> = Vec::new();
        // (1) Locals — `framestate.mergeable` head.
        for slot in self.locals_w_view(graph).iter() {
            if let Some(Hlvalue::Variable(v)) = slot {
                out.push(v.clone());
            }
        }
        // (2) Stack — `recursively_flatten(self.stack)` middle segment.
        for h in crate::flowspace::framestate::recursively_flatten(&self.stack) {
            if let Hlvalue::Variable(v) = h {
                out.push(v);
            }
        }
        // (3) Exception args — `[exc_type, exc_value]` tail.
        for h in exc_args(&self.last_exception).iter() {
            if let Hlvalue::Variable(v) = h {
                out.push(v.clone());
            }
        }
        out
    }

    /// `framestate.py FrameState.copy` — "Make a copy of this
    /// state in which all Variables are fresh."
    ///
    /// ```python
    /// def copy(self):
    ///     exc = self.last_exception
    ///     if exc is not None:
    ///         exc = FSException(_copy(exc.w_type), _copy(exc.w_value))
    ///     return FrameState(map(_copy, self.locals_w),
    ///                       map(_copy, self.stack),
    ///                       exc, self.blocklist, self.next_offset)
    /// ```
    ///
    /// `_copy(v)` (framestate.py) independently creates a fresh
    /// `Variable(v)` per occurrence — NO shared mapping across cells.
    /// Constants and None slots pass through unchanged.
    pub fn copy(&self, graph: &mut FunctionGraph) -> FrameState {
        use crate::flowspace::model::{Hlvalue, Variable};
        // `_copy(v)` — each call independently mints a fresh Variable.
        // `Variable(v)` (model.py) copies the source's name
        // prefix via `rename(v)`, which reads `v._name`; carry the
        // source slot's name onto the fresh Variable so the freshened
        // state's phi inputs keep their human-readable names.
        let copy_hlvalue = |h: &Hlvalue, graph: &mut FunctionGraph| -> Hlvalue {
            match h {
                Hlvalue::Variable(src) => {
                    // `_copy(v) -> Variable(v)` (framestate.py): fresh
                    // Variable carrying the source's `_name` prefix
                    // (annotation stays None).
                    let mut nv = graph.alloc_value_var();
                    nv.rename_from(src);
                    Hlvalue::Variable(nv)
                }
                other => other.clone(),
            }
        };
        let copy_var = |v: &Variable, graph: &mut FunctionGraph| -> Variable {
            let mut nv = graph.alloc_value_var();
            nv.rename_from(v);
            nv
        };
        // `locals_w` is the authoritative carrier (framestate.py __init__).
        // When `locals_w` is empty (pre-Z4.A.6 fixtures), fall back
        // to `entries` as the seed.
        let locals_w: Vec<Option<Hlvalue>> = if self.locals_w.len() == self.entries.len() {
            self.locals_w
                .iter()
                .map(|slot| slot.as_ref().map(|h| copy_hlvalue(h, graph)))
                .collect()
        } else {
            self.entries
                .iter()
                .map(|e| e.as_ref().map(|v| Hlvalue::Variable(copy_var(v, graph))))
                .collect()
        };
        let entries: Vec<Option<Variable>> = locals_w
            .iter()
            .map(|slot| {
                slot.as_ref().and_then(|hv| match hv {
                    Hlvalue::Variable(v) => Some(v.clone()),
                    Hlvalue::Constant(_) => None,
                })
            })
            .collect();
        // `map(_copy, self.stack)` — each stack element independently copied.
        // framestate.py: FlowSignal → rebuild with recursively
        // copied args.
        let stack: Vec<crate::flowspace::framestate::StackElem> =
            self.stack
                .iter()
                .map(|elem| match elem {
                    crate::flowspace::framestate::StackElem::Value(h) => {
                        crate::flowspace::framestate::StackElem::Value(copy_hlvalue(h, graph))
                    }
                    crate::flowspace::framestate::StackElem::Signal(sig) => {
                        use crate::flowspace::flowcontext::FlowSignal;
                        let copied_args: Vec<Hlvalue> =
                            sig.args().iter().map(|a| copy_hlvalue(a, graph)).collect();
                        crate::flowspace::framestate::StackElem::Signal(
                            FlowSignal::rebuild_with_args(sig.tag(), copied_args),
                        )
                    }
                })
                .collect();
        // `exc = FSException(_copy(exc.w_type), _copy(exc.w_value))`
        let last_exception = self.last_exception.as_ref().map(|exc| {
            crate::flowspace::model::FSException::new(
                copy_hlvalue(&exc.w_type, graph),
                copy_hlvalue(&exc.w_value, graph),
            )
        });
        FrameState {
            entries,
            locals_w,
            stack,
            last_exception,
            blocklist: self.blocklist.clone(),
            next_offset: self.next_offset,
        }
    }

    /// `framestate.py FrameState.matches` — "Two states match if
    /// they only differ by using different Variables at the same place."
    ///
    /// ```python
    /// def matches(self, other):
    ///     assert self.blocklist == other.blocklist
    ///     assert self.next_offset == other.next_offset
    ///     for w1, w2 in zip(self.mergeable, other.mergeable):
    ///         if not (w1 == w2 or (isinstance(w1, Variable) and
    ///                              isinstance(w2, Variable))):
    ///             return False
    ///     return True
    /// ```
    ///
    /// blocklist and next_offset must agree (asserted) — `matches` is a
    /// post-`union` comparison invoked from `flowcontext.py` and the
    /// candidate-list precondition is that all candidates share the
    /// same join-point coordinates.
    ///
    /// The mergeable projection walks `locals + recursively_flatten(
    /// stack) + [exc_type, exc_value]` (same shape as `getvariables` /
    /// `getoutputargs`).  Two cells match iff they are equal or both
    /// are `Variable` cells (regardless of identity — phi slot equality
    /// is by structure, not Variable identity).  `(None, None)` matches; any other
    /// mix involving `None` does not.
    pub fn matches(&self, other: &FrameState, graph: &FunctionGraph) -> bool {
        use crate::flowspace::model::Hlvalue;
        assert_eq!(
            self.blocklist, other.blocklist,
            "matches: blocklist mismatch"
        );
        assert_eq!(
            self.next_offset, other.next_offset,
            "matches: next_offset mismatch"
        );
        // (1) Locals — `framestate.mergeable` head.
        let self_locals = self.locals_w_view(graph);
        let other_locals = other.locals_w_view(graph);
        if self_locals.len() != other_locals.len() {
            return false;
        }
        let opt_cell_match = |w1: &Option<Hlvalue>, w2: &Option<Hlvalue>| match (w1, w2) {
            (None, None) => true,
            (Some(Hlvalue::Variable(_)), Some(Hlvalue::Variable(_))) => true,
            (Some(x), Some(y)) => x == y,
            _ => false,
        };
        if !self_locals
            .iter()
            .zip(other_locals.iter())
            .all(|(w1, w2)| opt_cell_match(w1, w2))
        {
            return false;
        }
        drop(self_locals);
        drop(other_locals);
        // (2) Stack — `recursively_flatten(self.stack)` middle segment.
        let self_stack = crate::flowspace::framestate::recursively_flatten(&self.stack);
        let other_stack = crate::flowspace::framestate::recursively_flatten(&other.stack);
        if self_stack.len() != other_stack.len() {
            return false;
        }
        let cell_match = |w1: &Hlvalue, w2: &Hlvalue| match (w1, w2) {
            (Hlvalue::Variable(_), Hlvalue::Variable(_)) => true,
            (x, y) => x == y,
        };
        if !self_stack
            .iter()
            .zip(other_stack.iter())
            .all(|(w1, w2)| cell_match(w1, w2))
        {
            return false;
        }
        // (3) Exception args — `[exc_type, exc_value]` tail.
        let self_exc = exc_args(&self.last_exception);
        let other_exc = exc_args(&other.last_exception);
        self_exc
            .iter()
            .zip(other_exc.iter())
            .all(|(w1, w2)| cell_match(w1, w2))
    }
}

/// Hlvalue→LinkArg routing per cell, matching upstream
/// `framestate.py getoutputargs` which appends the polymorphic
/// `mergeable[i]` cell directly into `Link.args`.  Pyre's `LinkArg` is
/// the matching closed sum: `Hlvalue::Variable(v)` → `LinkArg::Value(v)`
/// (the Variable carries its identity inline so downstream readers
/// match on the Variable object directly);
/// `Hlvalue::Constant(c)` → `LinkArg::Const(c)` (direct carry, no
/// synthetic op or slot allocation required for the Constant domain).
fn hlvalue_to_linkarg(w: &crate::flowspace::model::Hlvalue) -> LinkArg {
    use crate::flowspace::model::Hlvalue;
    match w {
        Hlvalue::Variable(v) => LinkArg::Value(v.clone()),
        Hlvalue::Constant(c) => LinkArg::Const(c.clone()),
    }
}

/// RPython `framestate.py _exc_args` — return `[w_type, w_value]`,
/// substituting `Constant(None)` sentinels when no exception is
/// pending.
fn exc_args(
    last_exception: &Option<crate::flowspace::model::FSException>,
) -> [crate::flowspace::model::Hlvalue; 2] {
    use crate::flowspace::model::{Constant, Hlvalue};
    match last_exception {
        None => [
            Hlvalue::Constant(Constant::new(ConstValue::None)),
            Hlvalue::Constant(Constant::new(ConstValue::None)),
        ],
        Some(exc) => [exc.w_type.clone(), exc.w_value.clone()],
    }
}

/// RPython `eliminate_empty_blocks(graph)`
/// (`rpython/translator/simplify.py`).
///
/// ```python
/// def eliminate_empty_blocks(graph):
///     for link in list(graph.iterlinks()):
///         while not link.target.operations:
///             block1 = link.target
///             if block1.exitswitch is not None:
///                 break
///             if not block1.exits:
///                 break
///             exit = block1.exits[0]
///             assert block1 is not exit.target, \
///                 "the graph contains an empty infinite loop"
///             subst = dict(zip(block1.inputargs, link.args))
///             link.args = [v.replace(subst) for v in exit.args]
///             link.target = exit.target
/// ```
///
/// Pyre's tree-recursive `Expr::Match` / `Expr::If` lowering produces
/// the post-collapse CFG directly (no chain to collapse), so this
/// pass is a no-op on production graphs today.  It becomes
/// load-bearing once Z4's flowcontext-walker rewrite materialises
/// intermediate `SpamBlock`s per fold step (the upstream
/// `flowcontext.py mergeblock` chain pattern).
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
pub fn eliminate_empty_blocks(graph: &mut FunctionGraph) {
    use std::collections::HashMap;
    // upstream: `for link in list(graph.iterlinks()):` — walk every
    // (block_id, exit_idx) pair.
    let block_count = graph.blocks.len();
    for block_id_idx in 0..block_count {
        let block_id = BlockId(block_id_idx);
        let exit_count = graph.block(block_id).exits.len();
        for exit_idx in 0..exit_count {
            // upstream: `while not link.target.operations:` — walk
            // through the dead-block chain.
            loop {
                let target = graph.block(block_id).exits[exit_idx].target;
                let target_block = graph.block(target);
                if !target_block.operations.is_empty() {
                    break;
                }
                if target_block.exitswitch.is_some() {
                    break;
                }
                if target_block.exits.is_empty() {
                    break;
                }
                // upstream: `exit = block1.exits[0]`.
                let target_inputargs: Vec<crate::flowspace::model::Variable> =
                    target_block.inputargs.clone();
                let target_exit = target_block.exits[0].clone();
                // upstream: `assert block1 is not exit.target, "the
                // graph contains an empty infinite loop"` (`simplify.py:64`).
                assert!(
                    target_exit.target != target,
                    "the graph contains an empty infinite loop: {:?}",
                    target
                );
                // upstream: `subst = dict(zip(block1.inputargs,
                // link.args))`.
                let link_args = graph.block(block_id).exits[exit_idx].args.clone();
                let subst: HashMap<crate::flowspace::model::Variable, LinkArg> =
                    target_inputargs.into_iter().zip(link_args).collect();
                // upstream: `link.args = [v.replace(subst) for v in
                // exit.args]`.
                let new_args: Vec<LinkArg> = target_exit
                    .args
                    .iter()
                    .map(|arg| match arg.as_variable() {
                        Some(v) => subst.get(v).cloned().unwrap_or_else(|| arg.clone()),
                        None => arg.clone(),
                    })
                    .collect();
                // upstream: `link.target = exit.target`.
                let link = &mut graph.block_mut(block_id).exits[exit_idx];
                link.args = new_args;
                link.target = target_exit.target;
            }
        }
    }
}

/// Retarget each edge into a block that exists only to raise the implicit
/// `AssertionError` so it names `exceptblock` directly, giving
/// [`remove_assertion_errors`] the `exit.target is graph.exceptblock` shape
/// its upstream counterpart is written against.
///
/// Upstream's flow space raises `assert` as a link straight to `exceptblock`
/// (`flowcontext.py` `_implicit_`), so `simplify.remove_assertion_errors`
/// never has a block in between and its predicate can be literal.
/// `set_raise` / `set_raise_implicit` always install one, and the MIR front
/// fills it with the operations that build the exception, so the same raise
/// arrives here one block further out.
///
/// A block qualifies when it has a single exit, that exit is the
/// `AssertionError` raise, and every operation it holds satisfies
/// `can_remove_op` — `CanRemove` (`simplify.py`), the predicate the
/// dead-op pass would apply to those operations once the raise is gone.  An
/// operation outside it — anything that can raise or write — leaves the edge
/// pointing at its block.
///
/// The bypassed block keeps its own exits and falls out as unreachable, the
/// same way `eliminate_empty_blocks` (`simplify.py`) leaves the blocks
/// it rewires past.
///
/// Returns the number of retargeted edges.
pub fn retarget_assert_raise_blocks(graph: &mut FunctionGraph) -> usize {
    use crate::flowspace::model::{HOST_ENV, HostObject};

    /// The raise args of a block that exists only to raise the implicit
    /// `AssertionError`; `None` for anything else.
    fn bare_assert_raise_args<'a>(
        block: &'a Block,
        exceptblock: BlockId,
        assert_err_class: &HostObject,
    ) -> Option<&'a Vec<LinkArg>> {
        let [exit] = block.exits.as_slice() else {
            return None;
        };
        if exit.target != exceptblock {
            return None;
        }
        if !matches!(
            exit.args.first(),
            Some(LinkArg::Const(c))
                if matches!(
                    &c.value,
                    ConstValue::HostObject(h) if h == assert_err_class
                )
        ) {
            return None;
        }
        block
            .operations
            .iter()
            .all(|op| crate::inline::can_remove_op(&op.kind))
            .then_some(&exit.args)
    }

    let assert_err_class = HOST_ENV
        .lookup_builtin("AssertionError")
        .expect("HOST_ENV missing AssertionError");
    let exceptblock = graph.exceptblock;
    let mut retargeted = 0usize;
    // Each edge asks its own target, so there is no side table to keep in
    // step with the mutation — upstream carries no such map either, and a
    // qualifying block is cheap to recognise: one exit, one constant, and
    // operations the dead-op pass would reclaim.
    for block_idx in 0..graph.blocks.len() {
        for exit_idx in 0..graph.blocks[block_idx].exits.len() {
            let target = graph.blocks[block_idx].exits[exit_idx].target;
            // An edge already naming `exceptblock` is the shape the pass
            // matches; it is also every qualifying block's own exit, which is
            // what keeps this from rewriting the raise onto itself.
            if target == exceptblock {
                continue;
            }
            let Some(args) =
                bare_assert_raise_args(graph.block(target), exceptblock, &assert_err_class)
                    .cloned()
            else {
                continue;
            };
            let exit = &mut graph.blocks[block_idx].exits[exit_idx];
            exit.target = exceptblock;
            exit.args = args;
            retargeted += 1;
        }
    }
    retargeted
}

/// RPython `remove_assertion_errors(graph)` (simplify.py) over
/// the front model graph.
///
/// ```python
/// def remove_assertion_errors(graph):
///     """Remove branches that go directly to raising an AssertionError,
///     assuming that AssertionError shouldn't occur at run-time.  Note that
///     this is how implicit exceptions are removed (see _implicit_ in
///     flowcontext.py).
///     """
///     for block in list(graph.iterblocks()):
///         for i in range(len(block.exits)-1, -1, -1):
///             exit = block.exits[i]
///             if not (exit.target is graph.exceptblock and
///                     exit.args[0] == Constant(AssertionError)):
///                 continue
///             if len(block.exits) < 2:
///                 break
///             if block.canraise:
///                 if exit.exitcase is None:
///                     break
///                 if len(block.exits) == 2:
///                     block.exitswitch = None
///                     exit.exitcase = None
///             lst = list(block.exits)
///             del lst[i]
///             block.recloseblock(*lst)
/// ```
///
/// One extension beyond the upstream body: when the removal leaves a
/// single exit on a *non-canraise value switch*, the survivor is
/// promoted to an unconditional link (`block.exitswitch = None;
/// exits[0].exitcase = None` — `backendopt/removeassert.py:84-89
/// kill_assertion_link`).  Upstream flow graphs only carry Bool or
/// last-exception exitswitches into this pass, and `flatten.py
/// insert_exits` tolerates a leftover single Bool case (`assert
/// link.exitcase in (None, False, True)`); Rust enum-match lowering
/// produces Int discriminant switches whose leftover `Int` case that
/// assert rejects, so the `kill_assertion_link` normalisation applies
/// here.  For the canraise two-exit case the survivor (`exits[0]`)
/// already has `exitcase = None`, so the one code path covers both.
///
/// Returns the number of removed exits so the caller can gate the
/// follow-up dead-condition sweep (`removeassert.py` — "now melt
/// away the (hopefully) dead operation that compute the condition").
///
/// The predicate here is upstream's literal `exit.target is
/// graph.exceptblock`.  Graphs whose raise sits one block further out are
/// brought into that shape by [`retarget_assert_raise_blocks`], which the
/// caller runs first.
pub fn remove_assertion_errors(graph: &mut FunctionGraph) -> usize {
    use crate::flowspace::model::{HOST_ENV, HostObject};

    fn targets_assertion_error(
        exit: &Link,
        exceptblock: BlockId,
        assert_err_class: &HostObject,
    ) -> bool {
        exit.target == exceptblock
            && matches!(
                exit.args.first(),
                Some(LinkArg::Const(c))
                    if matches!(
                        &c.value,
                        ConstValue::HostObject(h) if h == assert_err_class
                    )
            )
    }

    let assert_err_class = HOST_ENV
        .lookup_builtin("AssertionError")
        .expect("HOST_ENV missing AssertionError");
    let exceptblock = graph.exceptblock;
    let mut removed = 0usize;
    // upstream: `for block in list(graph.iterblocks())`.
    for block_idx in 0..graph.blocks.len() {
        let mut i = graph.blocks[block_idx].exits.len();
        while i > 0 {
            i -= 1;
            let block = &graph.blocks[block_idx];
            let Some(exit) = block.exits.get(i) else {
                break;
            };
            // upstream: `if not (exit.target is graph.exceptblock and
            // exit.args[0] == Constant(AssertionError)): continue`.
            if !targets_assertion_error(exit, exceptblock, &assert_err_class) {
                continue;
            }
            // upstream: `if len(block.exits) < 2: break`.
            if block.exits.len() < 2 {
                break;
            }
            // upstream: `if block.canraise: if exit.exitcase is None:
            // break`.
            if block.canraise() && exit.exitcase.is_none() {
                break;
            }
            let exits_len = block.exits.len();
            let block = &mut graph.blocks[block_idx];
            // upstream: `lst = list(block.exits); del lst[i];
            // block.recloseblock(*lst)`.
            block.exits.remove(i);
            if exits_len == 2 {
                // Promote the survivor to an unconditional link —
                // upstream's canraise arm (`simplify.py:333-335`) plus
                // the `kill_assertion_link` normalisation for value
                // switches (`removeassert.py`, see above).  Clear
                // the low-level case too so no branch metadata lingers on
                // a now-unconditional edge, matching `fold_constant_exitswitch`.
                block.exitswitch = None;
                block.exits[0].exitcase = None;
                block.exits[0].llexitcase = None;
            }
            removed += 1;
        }
    }
    removed
}

/// Fold a constant exitswitch to its taken link — the model-layer
/// slice of `constant_fold_graph`'s link folding
/// (`rpython/translator/backendopt/constfold.py`): when a block's
/// `exitswitch` Variable is bound in the same block by a constant
/// (`ConstBool` / `ConstInt`, optionally through the `bool` UnaryOp
/// wrap [`FunctionGraph::set_branch`] appends), keep only the
/// matching exit and promote it to an unconditional link.  The dead
/// condition ops melt away in the caller's [`prune_dead_phis`]
/// sweep; the disconnected arm is emptied by
/// [`clear_unreachable_blocks`].  Returns the number of folded
/// switches.
pub fn fold_constant_exitswitch(graph: &mut FunctionGraph) -> usize {
    use crate::flowspace::model::Variable;
    use std::collections::HashMap;

    // target block id → predecessor block ids.  A block with a single
    // predecessor lets a constant defined upstream propagate into it
    // through the link args, exactly as `constant_fold_graph`'s link
    // folding does (`backendopt/constfold.py`).
    let mut preds: HashMap<BlockId, Vec<BlockId>> = HashMap::new();
    for b in &graph.blocks {
        for e in &b.exits {
            preds.entry(e.target).or_default().push(b.id);
        }
    }

    // Resolve `v` to its defining `OpKind` — first in `block`, then, when
    // `v` is one of `block`'s inputargs, through a chain of
    // single-predecessor links (positional inputarg ↔ link-arg).  A
    // single predecessor dominates the block, so a `LinkArg::Value` whose
    // backing Variable is defined in the predecessor propagates soundly.
    fn resolve_def<'g>(
        graph: &'g FunctionGraph,
        preds: &HashMap<BlockId, Vec<BlockId>>,
        mut block: BlockId,
        mut v: Variable,
    ) -> Option<&'g OpKind> {
        let mut seen: std::collections::HashSet<BlockId> = std::collections::HashSet::new();
        loop {
            let b = &graph.blocks[block.0];
            if let Some(op) = b
                .operations
                .iter()
                .find(|op| op.result.as_ref() == Some(&v))
            {
                return Some(&op.kind);
            }
            // `v` is not defined by an op here; if it is an inputarg and the
            // block has exactly one predecessor, hop to the link arg feeding
            // that inputarg slot.
            let slot = b.inputargs.iter().position(|a| a == &v)?;
            if !seen.insert(block) {
                return None;
            }
            let [pred] = preds.get(&block).map(Vec::as_slice).unwrap_or(&[]) else {
                return None;
            };
            let pred_block = &graph.blocks[pred.0];
            let link = pred_block.exits.iter().find(|e| e.target == block)?;
            // Only a `LinkArg::Value` threads a variable identity onward; an
            // inline `LinkArg::Const` literal is not a `ConstBool`/`ConstInt`
            // op def (the front folds `we_are_jitted()` to a `ConstBool` op
            // whose result is threaded as a `Value`, so the value chain
            // covers it) — decline the rarer inline-const shape here.
            let LinkArg::Value(pv) = link.args.get(slot)? else {
                return None;
            };
            v = pv.clone();
            block = *pred;
        }
    }

    let mut folded = 0usize;
    for block_idx in 0..graph.blocks.len() {
        let block = &graph.blocks[block_idx];
        let Some(ExitSwitch::Value(sw)) = block.exitswitch.clone() else {
            continue;
        };
        let block_id = block.id;
        let mut kind = resolve_def(graph, &preds, block_id, sw.clone());
        if let Some(OpKind::UnaryOp { op, operand, .. }) = kind
            && op == "bool"
        {
            kind = resolve_def(graph, &preds, block_id, operand.clone());
        }
        // The matched exitcase spellings: an `If` branch carries
        // `ExitCase::Bool` (`set_branch`), a MIR `SwitchInt` carries
        // `ExitCase::Const(Int)` plus the `"default"` catch-all arm
        // (`front::mir` terminator lowering).
        let (bool_case, int_case) = match kind {
            Some(OpKind::ConstBool(b)) => (Some(*b), Some(i64::from(*b))),
            Some(OpKind::ConstInt(n)) => ((*n == 0 || *n == 1).then_some(*n != 0), Some(*n)),
            Some(OpKind::ConstUInt(n)) => {
                ((*n == 0 || *n == 1).then_some(*n != 0), Some(*n as i64))
            }
            _ => continue,
        };
        let block = &graph.blocks[block_idx];
        let matches_case = |link: &Link| match &link.exitcase {
            Some(ExitCase::Bool(b)) => Some(*b) == bool_case,
            Some(ExitCase::Const(ConstValue::Int(n))) => Some(*n) == int_case,
            _ => false,
        };
        let is_default = |link: &Link| {
            matches!(
                &link.exitcase,
                Some(ExitCase::Const(ConstValue::UniStr(s))) if s == "default"
            )
        };
        // `replace_exitswitch_by_constant` (simplify.py) collects
        // *every* arm whose exitcase equals the constant and asserts exactly
        // one survivor (`assert len(newexits) == 1`), falling back to the
        // `"default"` catch-all only when none match: a constant matching
        // two arms is a malformed switch that checkgraph's exitcase
        // uniqueness invariant (`flowspace/model.py`, `checkgraph`:
        // `assert len(allexitcases) == len(block.exits)`) forbids.
        let matching: Vec<usize> = block
            .exits
            .iter()
            .enumerate()
            .filter(|(_, link)| matches_case(link))
            .map(|(idx, _)| idx)
            .collect();
        let chosen = match matching.as_slice() {
            [only] => *only,
            [] => {
                let Some(default_idx) = block.exits.iter().position(is_default) else {
                    continue;
                };
                default_idx
            }
            _ => panic!(
                "const-fold: constant switch matched {} arms (duplicate exitcase) in graph {:?}",
                matching.len(),
                graph.name
            ),
        };
        let block = &mut graph.blocks[block_idx];
        let mut taken = block.exits.swap_remove(chosen);
        taken.exitcase = None;
        taken.llexitcase = None;
        block.exits = vec![taken];
        block.exitswitch = None;
        folded += 1;
    }
    folded
}

/// Non-canonical calling-convention entries emitted into one rich graph.
///
/// `simplify.transform_dead_op_vars_in_blocks` accepts multiple graphs and
/// treats every graph start as a root.  Pyre's generated closure and
/// specialization bodies can instead share one `FunctionGraph` allocation,
/// with a secondary start represented by a no-predecessor block whose every
/// inputarg is defined by its matching `OpKind::Input` parameter operation.
/// A disconnected phi/branch block lacks that shape and is not a root.
fn secondary_entry_roots(graph: &FunctionGraph) -> Vec<BlockId> {
    let with_predecessor: HashSet<BlockId> = graph
        .blocks
        .iter()
        .flat_map(|block| block.exits.iter().map(|link| link.target))
        .collect();
    graph
        .blocks
        .iter()
        .filter(|block| {
            block.id != graph.startblock
                && block.id != graph.returnblock
                && block.id != graph.exceptblock
                && !block.inputargs.is_empty()
                && !with_predecessor.contains(&block.id)
                && block.inputargs.iter().all(|inputarg| {
                    block.operations.iter().any(|op| {
                        matches!(op.kind, OpKind::Input { .. })
                            && op.result.as_ref() == Some(inputarg)
                    })
                })
        })
        .map(|block| block.id)
        .collect()
}

/// Empty every block unreachable from a calling-convention entry in place —
/// operations, exits, inputargs cleared, the `Vec` slot kept because
/// `BlockId` doubles as the index.  Needed after
/// [`fold_constant_exitswitch`] disconnects an arm: the registry
/// lift (`translate_op`) and [`prune_dead_phis`] both walk blocks by
/// index.  The canonical `startblock` and each generated secondary entry are
/// roots; disconnected branch/phi arms are cleared.
pub fn clear_unreachable_blocks(graph: &mut FunctionGraph) {
    let mut reachable = vec![false; graph.blocks.len()];
    let mut worklist = secondary_entry_roots(graph);
    worklist.push(graph.startblock);
    while let Some(b) = worklist.pop() {
        if std::mem::replace(&mut reachable[b.0], true) {
            continue;
        }
        for link in &graph.blocks[b.0].exits {
            worklist.push(link.target);
        }
    }
    // The return / except sinks stay even when no live link reaches
    // them — graph plumbing addresses them unconditionally.
    reachable[graph.returnblock.0] = true;
    reachable[graph.exceptblock.0] = true;
    for (idx, block) in graph.blocks.iter_mut().enumerate() {
        if !reachable[idx] {
            block.operations.clear();
            block.exits.clear();
            block.inputargs.clear();
            block.exitswitch = None;
            // Drop the locals snapshot too: a retained framestate keeps
            // variable references alive into later analysis passes, which
            // defeats the unreachable-block drain.
            block.framestate = None;
        }
    }
}

/// Remove dead aggregate constructions — a `SyntheticTransparentCtor`
/// (or a no-arg `Box::new_uninit`) call whose result escapes *only* into
/// its own `FieldWrite` field stores (never read, returned, or passed) —
/// along with those stores.
///
/// This is the `remove_simple_mallocs` shape from
/// `rpython/translator/backendopt/malloc.py`: a `malloc` whose result
/// flows only into `setfield` stores and is never read is dead, so the
/// malloc and its stores are dropped.  `prune_dead_phis`
/// (`transform_dead_op_vars`) cannot do this on its own because a
/// `FieldWrite` is side-effecting (not a pure op), so it pins its
/// `base` operand in `read_vars` and the aggregate survives.
///
/// The construction only reaches the front-end as a malloc + store
/// chain (`OpKind::Call { SyntheticTransparentCtor }` +
/// `OpKind::FieldWrite`) for the heterogeneous tuple/struct path; the
/// pure `OpKind::NewTuple` form is already swept by `prune_dead_phis`.
/// A discarded `(a, b)` pair (e.g. `let _ = (truth, expect_true);`)
/// lowers to this chain, and its dead field stores otherwise force a
/// primitive→object store whose `(BoolRepr|IntegerRepr, InstanceRepr)`
/// convert the rtyper rejects.
///
/// Conservative by construction: a constructor result is treated as
/// dead only when its *every* use is as a `FieldWrite.base` (the one
/// read site this pass exempts) and every such store has no result of
/// its own.  Any other reference — a `FieldRead`/`InteriorFieldWrite`
/// base, a `FieldWrite` *value*, a call argument, an exitswitch, a
/// `Link.arg`, a terminal-block inputarg — pins the result and the
/// chain is kept.  Runs to a fixpoint so a store of one dead aggregate
/// into another exposes the inner one once the outer store is gone.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn function_path_ends_with(segments: &[String], tail: &[&str]) -> bool {
    segments.len() >= tail.len()
        && segments[segments.len() - tail.len()..]
            .iter()
            .map(String::as_str)
            .eq(tail.iter().copied())
}

/// True when `segments`/`args` spell the no-arg `boxed::Box::new_uninit`
/// allocation.  The crate root may be present or already stripped; the tail
/// pins the owner so an unrelated `new_uninit` leaf is never classified as
/// this allocation.  Arity is pinned so a multi-arg path sharing the leaf
/// is never swept or fused as a fresh box.
fn is_box_new_uninit_path(segments: &[String], args: &[LinkArg]) -> bool {
    args.is_empty() && function_path_ends_with(segments, &["boxed", "Box", "new_uninit"])
}

/// True when `segments` is exactly `core::ptr::write` of two arguments — the
/// store that initializes an uninit box, and the raw-write arm
/// `sink_fused_boxing_aggregates_at_raw_writes` rematerializes.
fn is_core_ptr_write_path(segments: &[String], args: &[LinkArg]) -> bool {
    args.len() == 2
        && segments
            .iter()
            .map(String::as_str)
            .eq(["core", "ptr", "write"])
}

/// True when `segments`/`args` spell `Box::assume_init` or
/// `MaybeUninit::assume_init` of one argument — the type-level transmute
/// that follows `ptr::write` into an uninit box.
fn is_assume_init_path(segments: &[String], args: &[LinkArg]) -> bool {
    args.len() == 1
        && (function_path_ends_with(segments, &["boxed", "Box", "assume_init"])
            || function_path_ends_with(segments, &["MaybeUninit", "assume_init"]))
}

pub fn remove_dead_aggregates(graph: &mut FunctionGraph) -> usize {
    use crate::flowspace::model::Variable;

    let is_removable_ctor = |kind: &OpKind| match kind {
        OpKind::Call {
            target: CallTarget::SyntheticTransparentCtor { .. },
            ..
        } => true,
        // `Box::new_uninit()` — a no-arg heap-box allocation. The `vec![…]`
        // construction (`box_assume_init_into_vec_unsafe(box [..])`) whose
        // consumer `front::mir` rewrote to `newlist` leaves the box dead:
        // its result flows only into the `FieldWrite` that stored the now-
        // unused `Array` aggregate. Treat it as a removable ctor so the box,
        // its store, and (once un-pinned) the `Array` cascade out together.
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } => is_box_new_uninit_path(segments, args),
        _ => false,
    };

    let mut total_removed = 0usize;
    loop {
        // Pass 1: gather every "real" read.  A `FieldWrite.base` is the
        // store *target*, not a value read, so it is the single ref this
        // pass does not count; `FieldWrite.value` (if a Variable) is a
        // genuine read and every other op contributes its full
        // `op_variable_refs` set.
        let mut real_read: HashSet<Variable> = HashSet::new();
        for block in &graph.blocks {
            for op in &block.operations {
                if let OpKind::FieldWrite { value, .. } = &op.kind {
                    if let Some(var) = value.as_variable() {
                        real_read.insert(var.clone());
                    }
                } else {
                    for var in crate::inline::op_variable_refs(&op.kind) {
                        real_read.insert(var);
                    }
                }
            }
            match &block.exitswitch {
                Some(ExitSwitch::Value(var)) => {
                    real_read.insert(var.clone());
                }
                // A fused comparison reads its operands at the branch.
                Some(ExitSwitch::Fused { args, .. }) => {
                    for arg in args {
                        real_read.insert(arg.clone());
                    }
                }
                Some(ExitSwitch::LastException) | None => {}
            }
            for link in &block.exits {
                for arg in &link.args {
                    if let Some(var) = arg.as_variable() {
                        real_read.insert(var.clone());
                    }
                }
            }
            // A terminal block (no exits) implicitly uses every inputarg
            // — the same return-value pin `prune_dead_phis` Step 1 makes.
            if block.exits.is_empty() {
                for iarg in &block.inputargs {
                    real_read.insert(iarg.clone());
                }
            }
        }

        // Pass 2: a constructor result is dead when it is not read and
        // every store into it is result-less (so dropping the store
        // leaves no dangling definition).
        let mut dead: HashSet<Variable> = HashSet::new();
        for block in &graph.blocks {
            for op in &block.operations {
                if !is_removable_ctor(&op.kind) {
                    continue;
                }
                let Some(result) = &op.result else { continue };
                if real_read.contains(result) {
                    continue;
                }
                let stores_clean =
                    graph
                        .blocks
                        .iter()
                        .flat_map(|b| &b.operations)
                        .all(|o| match &o.kind {
                            OpKind::FieldWrite { base, .. } if base == result => o.result.is_none(),
                            _ => true,
                        });
                if stores_clean {
                    dead.insert(result.clone());
                }
            }
        }
        if dead.is_empty() {
            break;
        }

        // Pass 3: drop the dead constructors and their field stores.
        let mut removed = 0usize;
        for block in &mut graph.blocks {
            block.operations.retain(|op| {
                let drop = if is_removable_ctor(&op.kind) {
                    op.result.as_ref().is_some_and(|r| dead.contains(r))
                } else if let OpKind::FieldWrite { base, .. } = &op.kind {
                    dead.contains(base)
                } else {
                    false
                };
                if drop {
                    removed += 1;
                }
                !drop
            });
        }
        total_removed += removed;
        if removed == 0 {
            break;
        }
    }
    total_removed
}

/// Resolve one registered struct layout without discarding nominal identity.
///
/// RPython indexes layouts by the live `lltype.Struct` object.  Pyre's
/// corresponding identity is `StructId`; a bare leaf match is not admissible
/// because two modules may define the same leaf.  Exact string hits remain
/// valid for synthetic/test layouts that have no StructId registration.
pub(crate) fn registered_struct_layout<'a>(
    owner: &str,
    struct_field_attrs: &'a std::collections::HashMap<String, Vec<(String, ValueType)>>,
) -> Option<&'a Vec<(String, ValueType)>> {
    if let Some(exact) = struct_field_attrs.get(owner) {
        return Some(exact);
    }
    let owner_id = majit_ir::descr::struct_id_for_name(owner)?;
    let mut matches = struct_field_attrs.iter().filter_map(|(key, rows)| {
        (majit_ir::descr::struct_id_for_name(key) == Some(owner_id)).then_some(rows)
    });
    let first = matches.next()?;
    matches.all(|rows| rows == first).then_some(first)
}

/// Crate-relative struct path: Charon's declaration path with this
/// workspace's crate root removed, every remaining module segment kept.
///
/// [`synthetic_transparent_ctor_owner`] reduces a constructor the same way.
/// The layout map is keyed by this spelling.
pub(crate) fn crate_relative_struct_owner(name_path: &str, source_crate: Option<&str>) -> String {
    let mut segments: Vec<&str> = name_path.split("::").collect();
    let name = segments.pop().unwrap_or(name_path);
    relative_owner_from_segments(&segments, name, source_crate)
}

fn relative_owner_from_segments(
    owner_path: &[&str],
    name: &str,
    source_crate: Option<&str>,
) -> String {
    let mut segments = owner_path;
    if segments.first().is_some_and(|root| {
        source_crate == Some(*root) || crate::local_crates::is_local_crate_root(root)
    }) {
        segments = &segments[1..];
    }
    if segments.is_empty() {
        name.to_string()
    } else {
        format!("{}::{name}", segments.join("::"))
    }
}

/// The nominal struct identity carried by a resolved transparent constructor.
///
/// `front::mir` records Charon's full declaration path on `owner_path`.  The
/// runtime descriptor namespace strips only this workspace's crate root, so
/// preserve every remaining module segment instead of collapsing back to the
/// leaf.  This is the Rust-source counterpart of RPython carrying the same
/// `lltype.Struct` object from `malloc(STRUCT)` into every layout/descr lookup.
fn synthetic_transparent_ctor_owner(
    target: &CallTarget,
    source_crate: Option<&str>,
) -> Option<String> {
    let CallTarget::SyntheticTransparentCtor {
        name, owner_path, ..
    } = target
    else {
        return None;
    };
    let segments: Vec<&str> = owner_path.iter().map(String::as_str).collect();
    Some(relative_owner_from_segments(&segments, name, source_crate))
}

/// `jtransform.py rewrite_op_malloc`: a GcStruct whose first-field
/// chain reaches a typeptr header is `new_with_vtable`; otherwise `new`.
/// The vtable word on the op is 0: `descr.get_vtable()` is the runtime
/// size descr (`heaptracker.get_vtable_for_gcstruct`).
pub fn malloc_opkind(owner: String, has_vtable: bool) -> OpKind {
    if has_vtable {
        OpKind::NewWithVtable { owner, vtable: 0 }
    } else {
        OpKind::New { owner }
    }
}

/// Integer 0 stored into a GCREF field (`vable_token: 0`). Upstream
/// records `ConstPtr(NULL)`.
pub fn link_arg_is_int_zero(graph: &FunctionGraph, value: &LinkArg) -> bool {
    match value {
        LinkArg::Const(c) => matches!(
            c.value,
            crate::flowspace::model::ConstValue::Int(0)
                | crate::flowspace::model::ConstValue::Int128(0)
                | crate::flowspace::model::ConstValue::UInt128(0)
        ),
        LinkArg::Value(var) => graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .find(|op| op.result.as_ref() == Some(var))
            .is_some_and(|op| {
                matches!(
                    op.kind,
                    OpKind::ConstInt(0)
                        | OpKind::ConstUInt(0)
                        | OpKind::ConstInt128(0)
                        | OpKind::ConstUInt128(0)
                )
            }),
    }
}

/// Immediate dominators for [`lower_struct_ptr_writes`].
///
/// Same answers as the iterative dataflow that starts every non-entry
/// block at the full block set: `dom(start) = {start}`, and for every
/// other block `dom(b) = {b} ∪ ⋂ dom(p)` over its predecessors. An empty
/// predecessor set intersects to nothing, so a non-start block with no
/// predecessors is dominated only by itself. A block that never meets
/// `start` or one of those predecessor-less roots keeps the top set —
/// every block dominates it.
///
/// The tree hangs those roots off one virtual root (index `n`) so a walk
/// up [`Self::dominates`] is the membership test. Blocks the walk never
/// reaches stay `DOM_UNDEF` and answer as the top set.
const DOM_UNDEF: u32 = u32::MAX;

struct BlockDominators {
    /// Immediate dominator of each block, plus the virtual root at `n`.
    idom: Vec<u32>,
    n: usize,
}

impl BlockDominators {
    fn compute(graph: &FunctionGraph) -> Self {
        let n = graph.blocks.len();
        let start = graph.startblock.0;
        let vr = n as u32;
        let mut preds: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut succs: Vec<Vec<u32>> = vec![Vec::new(); n + 1];
        for (src, block) in graph.blocks.iter().enumerate() {
            for link in &block.exits {
                let dst = link.target.0;
                preds[dst].push(src as u32);
                succs[src].push(dst as u32);
            }
        }
        for pred in &mut preds {
            pred.sort_unstable();
            pred.dedup();
        }
        for block in 0..n {
            // `start` keeps `{start}` even when a back-edge targets it.
            // Every other block with no predecessor is its own root.
            if block == start || preds[block].is_empty() {
                succs[n].push(block as u32);
                preds[block] = vec![vr];
            }
        }

        let mut seen = vec![false; n + 1];
        let mut post = Vec::with_capacity(n + 1);
        let mut stack = vec![(vr, 0usize)];
        seen[n] = true;
        while let Some(&(node, next_i)) = stack.last() {
            if next_i < succs[node as usize].len() {
                let child = succs[node as usize][next_i];
                stack.last_mut().expect("dominator dfs frame").1 = next_i + 1;
                if !seen[child as usize] {
                    seen[child as usize] = true;
                    stack.push((child, 0));
                }
            } else {
                post.push(node);
                stack.pop();
            }
        }
        post.reverse();
        let mut rpo_of = vec![DOM_UNDEF; n + 1];
        for (index, &node) in post.iter().enumerate() {
            rpo_of[node as usize] = index as u32;
        }

        let mut idom = vec![DOM_UNDEF; n + 1];
        idom[n] = vr;
        let mut changed = true;
        let mut passes = 0usize;
        while changed {
            passes += 1;
            if passes > n.saturating_mul(n).saturating_add(1) {
                panic!("block dominators did not converge");
            }
            changed = false;
            for &block in post.iter().skip(1) {
                let mut chosen: Option<u32> = None;
                for &pred in &preds[block as usize] {
                    if idom[pred as usize] == DOM_UNDEF {
                        continue;
                    }
                    chosen = Some(match chosen {
                        None => pred,
                        Some(current) => intersect_idoms(current, pred, &idom, &rpo_of),
                    });
                }
                let Some(new_idom) = chosen else {
                    continue;
                };
                if idom[block as usize] != new_idom {
                    idom[block as usize] = new_idom;
                    changed = true;
                }
            }
        }
        Self { idom, n }
    }

    /// `true` when `dominator` is in `dom(block)`, including the top-set
    /// answer for a block unreachable from every root.
    fn dominates(&self, dominator: usize, block: usize) -> bool {
        if dominator == block {
            return true;
        }
        if self.idom.get(block).copied().unwrap_or(DOM_UNDEF) == DOM_UNDEF {
            return true;
        }
        let vr = self.n as u32;
        let target = dominator as u32;
        let mut finger = block as u32;
        for _ in 0..=self.n {
            if finger == target {
                return true;
            }
            if finger == vr {
                return false;
            }
            let next = self.idom[finger as usize];
            if next == DOM_UNDEF || next == finger {
                return false;
            }
            finger = next;
        }
        false
    }
}

fn intersect_idoms(mut finger1: u32, mut finger2: u32, idom: &[u32], rpo_of: &[u32]) -> u32 {
    let guard = idom.len();
    let mut steps = 0usize;
    while finger1 != finger2 {
        steps += 1;
        if steps > guard {
            panic!("block dominator intersect did not converge");
        }
        while rpo_of[finger1 as usize] > rpo_of[finger2 as usize] {
            finger1 = idom[finger1 as usize];
        }
        while rpo_of[finger2 as usize] > rpo_of[finger1 as usize] {
            finger2 = idom[finger2 as usize];
        }
    }
    finger1
}

/// Op-result and phi-input index for [`lower_struct_ptr_writes`]'s
/// producer walk. One scan replaces the per-step rescan of every block.
#[expect(
    clippy::mutable_key_type,
    reason = "Variable equality and hashing use immutable identity IDs while annotations are excluded"
)]
struct ProducerIndex<'g> {
    graph: &'g FunctionGraph,
    defined: rustc_hash::FxHashSet<crate::flowspace::model::Variable>,
    input_slot: rustc_hash::FxHashMap<crate::flowspace::model::Variable, (BlockId, usize)>,
    incoming: rustc_hash::FxHashMap<BlockId, Vec<(usize, usize)>>,
    memo: rustc_hash::FxHashMap<
        (crate::flowspace::model::Variable, u32),
        Option<crate::flowspace::model::Variable>,
    >,
}

impl<'g> ProducerIndex<'g> {
    fn build(graph: &'g FunctionGraph) -> Self {
        let mut defined = rustc_hash::FxHashSet::default();
        let mut input_slot = rustc_hash::FxHashMap::default();
        let mut incoming: rustc_hash::FxHashMap<BlockId, Vec<(usize, usize)>> =
            rustc_hash::FxHashMap::default();
        for (block_idx, block) in graph.blocks.iter().enumerate() {
            for op in &block.operations {
                if let Some(result) = op.result.as_ref() {
                    defined.insert(result.clone());
                }
            }
            for (slot, input) in block.inputargs.iter().enumerate() {
                input_slot.entry(input.clone()).or_insert((block.id, slot));
            }
            for (exit_idx, link) in block.exits.iter().enumerate() {
                incoming
                    .entry(link.target)
                    .or_default()
                    .push((block_idx, exit_idx));
            }
        }
        Self {
            graph,
            defined,
            input_slot,
            incoming,
            memo: rustc_hash::FxHashMap::default(),
        }
    }

    fn root(
        &mut self,
        var: &crate::flowspace::model::Variable,
        depth: u32,
    ) -> Option<crate::flowspace::model::Variable> {
        if let Some(cached) = self.memo.get(&(var.clone(), depth)) {
            return cached.clone();
        }
        let found = self.root_uncached(var, depth);
        self.memo.insert((var.clone(), depth), found.clone());
        found
    }

    fn root_uncached(
        &mut self,
        var: &crate::flowspace::model::Variable,
        depth: u32,
    ) -> Option<crate::flowspace::model::Variable> {
        if depth == 0 {
            return None;
        }
        if self.defined.contains(var) {
            return Some(var.clone());
        }
        let (target, slot) = self.input_slot.get(var).copied()?;
        let mut incoming_vars = Vec::new();
        if let Some(links) = self.incoming.get(&target) {
            incoming_vars.reserve(links.len());
            for &(block_idx, exit_idx) in links {
                let arg = self.graph.blocks[block_idx].exits[exit_idx]
                    .args
                    .get(slot)?;
                incoming_vars.push(arg.as_variable()?.clone());
            }
        }
        if incoming_vars.is_empty() {
            return None;
        }
        let mut root = None;
        for incoming in incoming_vars {
            let incoming_root = self.root(&incoming, depth - 1)?;
            match &root {
                None => root = Some(incoming_root),
                Some(seen) if seen == &incoming_root => {}
                Some(_) => return None,
            }
        }
        root
    }
}

/// Lower `core::ptr::write(raw as *mut T, T { fields... })` to field stores.
///
/// This is the Rust-source spelling of RPython's ordinary alloc-then-init
/// graph shape: after `malloc(T)`, each member is written with its own
/// `setfield`.  Charon extracts `ptr::write` as one generic FunDecl plus a
/// concrete `generics.types[0]` at every call site.  Routing all those calls
/// through one `FunctionPath(["core", "ptr", "write"])` therefore collapses
/// the monomorphisations onto one `FunctionDesc`; the whole-program annotator
/// then tries to union unrelated aggregate arguments (observed first as
/// `W_LongObject ∪ Cell`).  RPython has no corresponding callable or union:
/// the stores are operations in each caller graph.
///
/// The MIR front has already made the concrete type explicit on both sides:
/// the destination is a `__cast_instance_intrinsic(T)` result and the value is a
/// `SyntheticTransparentCtor(T)` followed by ordered `FieldWrite`s.  The cast
/// constant stays the class leaf; its result type carries the declaration
/// path when that path is longer than the leaf, the spelling
/// [`synthetic_transparent_ctor_owner`] reduces to.  Calls in the
/// source often split the aggregate from the stable allocation with
/// residual GC hooks; resolve only phis whose every incoming path reaches the
/// same constructor, then re-emit the registered complete field layout in
/// declaration order at the destination.  A value that is already an instance
/// of that struct — a by-value parameter whose fields were rewritten, not a
/// constructor — is copied with one field read of the source and one field
/// store into the destination per registered field, the same layout list the
/// constructor route stores (embedded header fields included).  The
/// constructor route stays first: a ctor-rooted aggregate is never copied
/// this way.  A field whose declared type is a by-value nested struct is
/// copied per leaf through that inner struct's own descriptors
/// (`heaptracker.py all_fielddescrs` recursion; `jtransform.py
/// rewrite_op_getsubstruct`), not as one `Ref` word.  The call's unit
/// result, when still carried by MIR bookkeeping, becomes a `ConstNone`
/// with the same `Void` representation.  Any indirect, disagreeing-phi,
/// mismatched-owner, or incomplete spelling is left untouched to fail
/// loud rather than guessing at a memory layout.
pub fn lower_struct_ptr_writes(
    graph: &mut FunctionGraph,
    struct_field_attrs: &std::collections::HashMap<String, Vec<(String, ValueType)>>,
) -> usize {
    lower_struct_ptr_writes_with(graph, struct_field_attrs, &|_, _| None)
}

pub(crate) fn lower_struct_ptr_writes_with(
    graph: &mut FunctionGraph,
    struct_field_attrs: &std::collections::HashMap<String, Vec<(String, ValueType)>>,
    expand_inline_struct: &dyn Fn(&str, &str) -> Option<Vec<(FieldDescriptor, ValueType)>>,
) -> usize {
    use crate::flowspace::model::Variable;

    // Most lowered graphs never call `core::ptr::write`. Dominators and the
    // producer index are both quadratic in the size of the ones that do, so
    // they wait until a call of that shape is actually present.
    let has_ptr_write = graph.blocks.iter().any(|block| {
        block.operations.iter().any(|op| {
            let OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                result_ty: ValueType::Void,
            } = &op.kind
            else {
                return false;
            };
            is_core_ptr_write_path(segments, args)
        })
    });
    if !has_ptr_write {
        return 0;
    }
    let mut producers = ProducerIndex::build(graph);
    // Filled on the first cross-block store. Same-block stores compare
    // operation order and never ask.
    let mut dominators: Option<BlockDominators> = None;

    #[derive(Clone)]
    enum FieldInit {
        /// Constructor route: each value is already the stored operand.
        Stores(Vec<(FieldDescriptor, LinkArg, ValueType)>),
        /// Existing aggregate: read `source.<field>` then store it.
        Copy {
            source: Variable,
            fields: Vec<(FieldDescriptor, ValueType)>,
        },
    }

    #[derive(Clone)]
    struct Rewrite {
        block: usize,
        op: usize,
        destination: Variable,
        owner: String,
        init: FieldInit,
        result: Option<Variable>,
    }

    let mut rewrites = Vec::new();
    for (bi, block) in graph.blocks.iter().enumerate() {
        for (oi, op) in block.operations.iter().enumerate() {
            let OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                result_ty: ValueType::Void,
            } = &op.kind
            else {
                continue;
            };
            if args.len() != 2
                || !segments
                    .iter()
                    .map(String::as_str)
                    .eq(["core", "ptr", "write"])
            {
                continue;
            }
            let destination = &args[0];
            let aggregate = producers.root(&args[1], 16);
            // Class key and layout owner are different strings on one cast:
            // the constant is the leaf, the result's `Ref` name is the
            // declaration path. [`cast_instance_layout_owner`] reads the path.
            let destination_owner = block.operations[..oi].iter().find_map(|candidate| {
                match candidate.result.as_ref() {
                    Some(result) if result == destination => {
                        cast_instance_layout_owner(&candidate.kind)
                    }
                    _ => None,
                }
            });
            let aggregate_ctor = aggregate.as_ref().and_then(|aggregate| {
                graph
                    .blocks
                    .iter()
                    .flat_map(|block| &block.operations)
                    .find_map(|candidate| match (&candidate.result, &candidate.kind) {
                        (Some(result), OpKind::Call { target, .. }) if result == aggregate => {
                            let owner = synthetic_transparent_ctor_owner(
                                target,
                                graph.name.split("::").next(),
                            )?;
                            let leaf = match target {
                                CallTarget::SyntheticTransparentCtor { name, .. } => {
                                    Some(name.clone())
                                }
                                _ => None,
                            };
                            Some((owner, leaf))
                        }
                        _ => None,
                    })
            });
            // A ctor-rooted aggregate keeps the constructor route, including
            // every decline that leaves the call residual.  The copy below
            // runs only when there is no constructor producer, or that
            // producer has no constructor owner.
            if aggregate.is_none() || aggregate_ctor.is_none() {
                let Some(owner) = destination_owner else {
                    continue;
                };
                let Some(layout) = registered_struct_layout(owner, struct_field_attrs) else {
                    continue;
                };
                let Some(source) = args[1].as_variable().cloned() else {
                    continue;
                };
                let Some(destination) = destination.as_variable().cloned() else {
                    continue;
                };
                let owner_id = majit_ir::descr::struct_id_for_name(owner);
                // The copy is the aggregate's stored words, including a
                // virtualizable array field.  A live virtualizable access
                // would drop that read and refuse the following store
                // (`jtransform.py` `_check_no_vable_array`).  The source is
                // the by-value instance, not a live virtualizable, so the
                // same `base_is_deref == Some(false)` the front records for
                // `local.field` keeps both the read and the store ordinary.
                let fields = layout
                    .iter()
                    .map(|(name, ty)| {
                        (
                            FieldDescriptor::new(name.clone(), Some(owner.to_string()))
                                .with_owner_id(owner_id)
                                .with_base_is_deref(false),
                            crate::virtualizable_decl::overlay_virtualizable_llfield_value_type(
                                owner,
                                name,
                                ty.clone(),
                            ),
                        )
                    })
                    .collect();
                rewrites.push(Rewrite {
                    block: bi,
                    op: oi,
                    destination,
                    owner: owner.to_string(),
                    init: FieldInit::Copy { source, fields },
                    result: op.result.clone(),
                });
                continue;
            }
            let aggregate = aggregate.expect("ctor producer checked");
            let (aggregate_owner, aggregate_leaf) = aggregate_ctor.expect("ctor owner checked");
            // The destination names its pointee by class root, the
            // declaration's bare leaf unless that leaf was withdrawn as
            // ambiguous. The constructor carries that leaf beside its module
            // path. The name → `StructId` table is published only after the
            // whole program is lowered, so the leaf is compared directly.
            let same_owner = destination_owner.is_some_and(|destination| {
                destination == aggregate_owner
                    || aggregate_leaf.as_deref() == Some(destination)
                    || majit_ir::descr::struct_id_for_name(destination)
                        .zip(majit_ir::descr::struct_id_for_name(&aggregate_owner))
                        .is_some_and(|(destination_id, aggregate_id)| {
                            destination_id == aggregate_id
                        })
            });
            if !same_owner {
                continue;
            }
            // A stored value may be a Variable of the store's block rather than
            // of this one. `checkgraph` (`flowspace/model.py`) does not accept
            // that as it stands, but the store's block dominates this one, so
            // the value is available on every path here and
            // `thread_undefined_op_operands` (the `SSA_to_SSI` step,
            // `backendopt/ssa.py`) adds the inputargs and link arguments that
            // carry it once the simplification sweeps are done.
            let stores: Vec<_> = graph
                .blocks
                .iter()
                .enumerate()
                .flat_map(|(store_bi, store_block)| {
                    store_block
                        .operations
                        .iter()
                        .enumerate()
                        .map(move |(store_oi, candidate)| (store_bi, store_oi, candidate))
                })
                .filter_map(|(store_bi, store_oi, candidate)| match &candidate.kind {
                    OpKind::FieldWrite {
                        base,
                        field,
                        value,
                        ty,
                    } if producers.root(base, 16).as_ref() == Some(&aggregate) => {
                        Some((store_bi, store_oi, field.clone(), value.clone(), ty.clone()))
                    }
                    _ => None,
                })
                .collect();
            // A store dominates the call when it is earlier in the same block,
            // or its block dominates the call block. Cross-block dominators are
            // built on the first such store — graphs with only same-block
            // stores never build them.
            if stores.iter().any(|(store_bi, store_oi, _, _, _)| {
                let store_dominates_call = if *store_bi == bi {
                    *store_oi < oi
                } else {
                    dominators
                        .get_or_insert_with(|| BlockDominators::compute(graph))
                        .dominates(*store_bi, bi)
                };
                !store_dominates_call
            }) {
                continue;
            }
            let Some(layout) = registered_struct_layout(&aggregate_owner, struct_field_attrs)
            else {
                continue;
            };
            if stores.len() != layout.len() {
                continue;
            }
            let mut ordered_stores = Vec::with_capacity(layout.len());
            let mut complete = true;
            for (name, _) in layout {
                let mut matches = stores
                    .iter()
                    .filter(|(_, _, field, _, _)| &field.name == name);
                let Some(store) = matches.next() else {
                    complete = false;
                    break;
                };
                if matches.next().is_some() {
                    complete = false;
                    break;
                }
                ordered_stores.push((store.2.clone(), store.3.clone(), store.4.clone()));
            }
            if !complete {
                continue;
            }
            rewrites.push(Rewrite {
                block: bi,
                op: oi,
                destination: destination.clone().into_variable(),
                owner: aggregate_owner.clone(),
                init: FieldInit::Stores(ordered_stores),
                result: op.result.clone(),
            });
        }
    }

    let rewritten = rewrites.len();
    for rewrite in rewrites.into_iter().rev() {
        let mut replacement = Vec::new();
        match rewrite.init {
            FieldInit::Stores(stores) => {
                for (field, value, ty) in stores {
                    match expand_inline_struct(&rewrite.owner, &field.name)
                        .zip(value.as_variable().cloned())
                    {
                        Some((leaves, inner)) => {
                            for (leaf, leaf_ty) in leaves {
                                let read = graph.alloc_value_var();
                                replacement.push(SpaceOperation {
                                    result: Some(read.clone()),
                                    kind: OpKind::FieldRead {
                                        base: inner.clone(),
                                        field: leaf.clone(),
                                        ty: leaf_ty.clone(),
                                        pure: false,
                                    },
                                });
                                replacement.push(SpaceOperation {
                                    result: None,
                                    kind: OpKind::FieldWrite {
                                        base: rewrite.destination.clone(),
                                        field: leaf,
                                        value: LinkArg::Value(read),
                                        ty: leaf_ty,
                                    },
                                });
                            }
                        }
                        None => replacement.push(SpaceOperation {
                            result: None,
                            kind: OpKind::FieldWrite {
                                base: rewrite.destination.clone(),
                                field,
                                value,
                                ty,
                            },
                        }),
                    }
                }
            }
            FieldInit::Copy { source, fields } => {
                for (field, ty) in fields {
                    let leaves = expand_inline_struct(&rewrite.owner, &field.name)
                        .unwrap_or_else(|| vec![(field, ty)]);
                    for (leaf, leaf_ty) in leaves {
                        let read = graph.alloc_value_var();
                        replacement.push(SpaceOperation {
                            result: Some(read.clone()),
                            kind: OpKind::FieldRead {
                                base: source.clone(),
                                field: leaf.clone(),
                                ty: leaf_ty.clone(),
                                pure: false,
                            },
                        });
                        replacement.push(SpaceOperation {
                            result: None,
                            kind: OpKind::FieldWrite {
                                base: rewrite.destination.clone(),
                                field: leaf,
                                value: LinkArg::Value(read),
                                ty: leaf_ty,
                            },
                        });
                    }
                }
            }
        }
        if let Some(result) = rewrite.result {
            replacement.push(SpaceOperation {
                result: Some(result),
                kind: OpKind::ConstNone,
            });
        }
        graph.blocks[rewrite.block]
            .operations
            .splice(rewrite.op..=rewrite.op, replacement);
    }
    rewritten
}

/// Replace a by-value `FieldWrite` of an inlined struct with one store per
/// inner leaf.
///
/// `heaptracker.py all_fielddescrs` recurses into `isinstance(FIELD,
/// lltype.Struct)` and never numbers the nested field itself, so the
/// rtyper emits `getsubstruct` plus leaf `setfield`s. A `FieldWrite`
/// whose field type is that nested struct is the same copy: one
/// `FieldRead` of each inner leaf from the source and one `FieldWrite`
/// of that leaf onto the destination, using the inner struct's own
/// descriptors (`rewrite_op_getsubstruct`). The typeptr leaf stays;
/// `jtransform.py rewrite_op_setfield` / `is_typeptr_getset` drops it
/// on a vtable object.
pub(crate) fn typed_malloc_arg_vars(graph: &FunctionGraph) -> std::collections::HashSet<u64> {
    graph
        .blocks
        .iter()
        .flat_map(|block| &block.operations)
        .filter_map(|op| {
            let OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                ..
            } = &op.kind
            else {
                return None;
            };
            if segments.len() < 2 || segments[segments.len() - 2] != "lltype" {
                return None;
            }
            let flavor =
                crate::front::clause_spec::unspecialized_leaf(&segments[segments.len() - 1]);
            if !matches!(
                flavor,
                "malloc" | "malloc_typed" | "malloc_typed_managed" | "malloc_typed_stable"
            ) {
                return None;
            }
            Some(args.iter().filter_map(LinkArg::as_variable))
        })
        .flatten()
        .map(|var| var.id())
        .collect()
}

pub(crate) fn lower_inlined_struct_field_writes(
    graph: &mut FunctionGraph,
    expand_inline_struct: &dyn Fn(&str, &str) -> Option<Vec<(FieldDescriptor, ValueType)>>,
) -> Result<usize, String> {
    struct Rewrite {
        block: usize,
        op: usize,
        destination: crate::flowspace::model::Variable,
        source: crate::flowspace::model::Variable,
        leaves: Vec<(FieldDescriptor, ValueType)>,
    }
    let malloc_args = typed_malloc_arg_vars(graph);
    let mut rewrites = Vec::new();
    for (bi, block) in graph.blocks.iter().enumerate() {
        for (oi, op) in block.operations.iter().enumerate() {
            let OpKind::FieldWrite {
                base, field, value, ..
            } = &op.kind
            else {
                continue;
            };
            if malloc_args.contains(&base.id()) {
                continue;
            }
            let Some(owner) = field.owner_root.as_deref() else {
                continue;
            };
            let Some(leaves) = expand_inline_struct(owner, &field.name) else {
                continue;
            };
            let Some(source) = value.as_variable().cloned() else {
                return Err(format!(
                    "by-value store of inlined struct field {owner}.{} is not a variable",
                    field.name
                ));
            };
            rewrites.push(Rewrite {
                block: bi,
                op: oi,
                destination: base.clone(),
                source,
                leaves,
            });
        }
    }
    let rewritten = rewrites.len();
    for rewrite in rewrites.into_iter().rev() {
        let mut replacement = Vec::new();
        for (leaf, leaf_ty) in rewrite.leaves {
            let read = graph.alloc_value_var();
            replacement.push(SpaceOperation {
                result: Some(read.clone()),
                kind: OpKind::FieldRead {
                    base: rewrite.source.clone(),
                    field: leaf.clone(),
                    ty: leaf_ty.clone(),
                    pure: false,
                },
            });
            replacement.push(SpaceOperation {
                result: None,
                kind: OpKind::FieldWrite {
                    base: rewrite.destination.clone(),
                    field: leaf,
                    value: LinkArg::Value(read),
                    ty: leaf_ty,
                },
            });
        }
        graph.blocks[rewrite.block]
            .operations
            .splice(rewrite.op..=rewrite.op, replacement);
    }
    Ok(rewritten)
}

/// Fuse the boxing-constructor idiom into a native GC allocation.
///
/// pyre's boxing constructors (`floatobject::w_float_new` etc.) are written
/// in Rust as `malloc_typed(W_FloatObject { ob_header: …, floatval: v })` —
/// construct the whole struct on the stack, then heap-copy.  `front::mir`
/// lowers that to a `SyntheticTransparentCtor` aggregate (`%agg`) + per-field
/// `FieldWrite`s + a residual `Call(runtime_object::lltype::malloc_typed, [%agg])`.
/// RPython's orthodox form is alloc-then-init (`p = malloc(S); p.f = v`); the
/// rtyper lowers `malloc` to the GC allocation op.  pyre's rtyper is an
/// ephemeral type oracle that never rewrites the surviving model graph, so the
/// lowering is produced here instead: rewrite the cluster to an allocation
/// `-> %ret` + one payload `FieldWrite(%ret, …)` per field the allocation does
/// not carry, dropping the `ob_header` (PyObject base) subtree.
///
/// Which allocation is `rewrite_op_malloc`'s fork, and it turns on what the
/// STRUCT carries rather than on what the cluster stores.  A constant type
/// pointer is captured into `NewWithVtable.vtable` (the runtime stamps the new
/// object's `ob_type` / `w_class` from it — `type_id` resolves struct size
/// only), matching the runtime tracer oracle (`box_trace.rs trace_box_float`:
/// one `NewWithVtable` carrying a real type pointer + one `SetfieldGc` for the
/// payload).  A cluster with no type word at all takes the plain `New` upstream
/// emits for a struct it has no vtable to take.
///
/// There is deliberately no third arm for a type word that is present but not
/// constant.  Upstream cannot reach that shape — its typeptr is a property of
/// the STRUCT, not of the instance — and pyre cannot lower it: the type word
/// travels only on the allocation, since `optimize_setfield_gc` removes a
/// `setfield_gc` whose descr `is_typeptr()` and keeps only a constant value,
/// as `known_class` (`heaptracker.all_fielddescrs` drops `typeptr` from every
/// field list). Lowering it anyway ships an object whose type word is never
/// written.
///
/// The payload store is inserted *after* the `NewWithVtable` (which reuses the
/// malloc result Variable `%ret`), since the original aggregate field stores
/// precede the malloc call and would be use-before-def if retargeted in place.
/// The orphaned aggregate ctor + header `FieldWrite`s become dead and are swept
/// by the `remove_dead_aggregates` + `prune_dead_phis` passes that follow in
/// `simplify_lowered_graph`.
///
/// Every refusal below is a bare `continue` that leaves `malloc_typed`
/// residual, and none of them says so anywhere.  They are counted through
/// [`crate::decline`] under the two gate names declared next; see that
/// module for why the shape filters at the top of the site loop are
/// deliberately not among them.
pub fn fuse_boxing_alloc(
    graph: &mut FunctionGraph,
    struct_field_attrs: &std::collections::HashMap<String, Vec<(String, ValueType)>>,
) -> usize {
    fuse_boxing_alloc_with(graph, struct_field_attrs, &|_, _| None)
}

pub(crate) fn fuse_boxing_alloc_with(
    graph: &mut FunctionGraph,
    struct_field_attrs: &std::collections::HashMap<String, Vec<(String, ValueType)>>,
    expand_inline_struct: &dyn Fn(&str, &str) -> Option<Vec<(FieldDescriptor, ValueType)>>,
) -> usize {
    use crate::flowspace::model::Variable;
    // Gate names come from `crate::decline::gate` rather than being
    // spelled here: a name defined at its call site can be referenced
    // while the recorder module is absent, which is how a half-applied
    // edit left this function naming two gates the tree could not build.
    use crate::decline::gate::{
        FUSE_BOXING_ALLOC as FUSE_GATE, RESOLVE_HEADER_PLAN as VTABLE_GATE,
    };
    // Derive a boxing struct's scalar payload fields from its registered field
    // layout, in struct-declaration order, skipping the `ob_header` (PyObject
    // base): the header's type pointer is captured separately into
    // `NewWithVtable.vtable` (see `resolve_header_plan`) and the runtime stamps
    // `ob_type` / `w_class` from it, so only the non-header setfield(s) are
    // re-emitted (oracle: `box_trace.rs trace_box_float` / `trace_box_int`).
    // The one header field that survives as its own store is a `w_class` the
    // vtable cannot stand for, which `resolve_header_plan` hands back as a
    // payload.
    // This mirrors the type-generic malloc lowering in the front end
    // (`rpython/rtyper/rbuiltin.py` `rtype_malloc`; `rclass.py` reads
    // the exact lltype's field layout); `struct_field_attrs` is the
    // already-computed layout map the front end owns, so this pass performs no
    // front-end (Llbc) reads.
    //
    // An unknown struct yields `None` and is left unfused — the same fail-safe
    // the old hardcoded set applied to any struct outside the numeric four.
    //
    // The `PyObject` base field carries the type pointer, which
    // `resolve_vtable_addr` lifts into `NewWithVtable.vtable`, so it is skipped
    // here (the runtime stamps `ob_type`/`w_class` from the descriptor).  Two
    // spellings reach this pass: the hand-written boxing structs
    // (`W_FloatObject` etc.) name it `ob_header`, while `#[pyre_class]` injects
    // it as `ob` (pyre-macros `expand_pyre_class`). Both name the same base,
    // so both are recognised as the header and neither is re-emitted as a
    // payload setfield.
    fn is_header_field(name: &str) -> bool {
        name == "ob_header" || name == "ob"
    }
    fn payload_fields(
        owner: &str,
        struct_field_attrs: &std::collections::HashMap<String, Vec<(String, ValueType)>>,
    ) -> Option<Vec<(String, ValueType)>> {
        Some(
            registered_struct_layout(owner, struct_field_attrs)?
                .iter()
                .filter(|(name, _)| !is_header_field(name))
                .cloned()
                .collect(),
        )
    }

    // `owner`'s registered field layout, in struct-declaration order.
    // The allocator spelling decides more than whether this is a boxing
    // cluster.  `rewrite_op_malloc` forks on the malloc's flavor before it
    // forks on the struct, and refuses outright the flavors its allocation op
    // cannot express: `raw` diverts to `_rewrite_raw_malloc`, and a
    // `nonmovable` hint raises `UnsupportedMallocFlags`, because the op it
    // would emit IS a nursery bump and nothing in the op can say otherwise.
    fn gc_malloc_flavor(target: &CallTarget) -> Option<&str> {
        let CallTarget::FunctionPath { segments, .. } = target else {
            return None;
        };
        if segments.len() < 2 || segments[segments.len() - 2] != "lltype" {
            return None;
        }
        // A specialized copy is the same allocator.
        match crate::front::clause_spec::unspecialized_leaf(&segments[segments.len() - 1]) {
            flavor @ ("malloc"
            | "malloc_typed"
            | "malloc_typed_managed"
            | "malloc_typed_stable") => Some(flavor),
            _ => None,
        }
    }
    // Whether the cluster's allocator hands back an object the collector may
    // move.  `malloc_typed_stable` is `try_gc_alloc_stable_raw`, whose contract
    // is an old-gen pointer stable across every collection, and `malloc_typed`
    // (`alloc_with_gc_header`) leaks outside the collector entirely; both are
    // the structural twin of the `nonmovable` hint `rewrite_op_malloc` refuses.
    //
    // The refusal applies only to the arm this fork newly reaches.  A cluster
    // whose type word is a constant has lowered to `NewWithVtable` since long
    // before the `New` arm existed, and its structs carry the flavor on their
    // published `SizeDescr` instead; re-deciding that here would regress
    // clusters this fork is not about.
    fn allocates_movable(flavor: &str) -> bool {
        matches!(flavor, "malloc" | "malloc_typed_managed")
    }

    // Resolve the type-pointer the dropped `ob_header.ob_type` store carries:
    // `%agg.ob_header = %h; %h.ob_type = __cast_instance_intrinsic(ConstRefAddr(t))`.
    // The runtime stamps the new object's `ob_type`/`w_class` from this address
    // (read out of the `NewWithVtable` size descriptor), so it must travel with
    // the op rather than being dropped.  Returns `0` when the cluster carries no
    // resolvable constant type-pointer (e.g. a synthetic test fixture).
    //
    // `jtransform.py rewrite_op_malloc` gets the vtable from malloc's
    // `STRUCT`. `malloc_typed(value)` instead recovers it from
    // `value.ob_header.ob_type`; it must match `w_class`, since values may carry
    // a subclass there.
    fn store_value(
        graph: &FunctionGraph,
        base: &Variable,
        field_name: &str,
        site: (usize, usize),
    ) -> Option<Variable> {
        let store = unique_store(graph, base, field_name)?;
        store
            .locations
            .iter()
            .any(|&(block, op)| store_dominates_site(graph, base, (block, op), site))
            .then(|| store.value.as_variable().cloned())
            .flatten()
    }
    /// Whether the graph writes `base.field_name` at all, however many times.
    /// [`unique_store`] collapses "never written" and "written inconsistently"
    /// into one `None`; the malloc fork needs them apart, because a struct that
    /// never writes a type word is upstream's plain `new`, while one that
    /// writes several is a cluster this pass cannot read an answer out of.
    fn any_store(graph: &FunctionGraph, base: &Variable, field_name: &str) -> bool {
        graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
            matches!(&op.kind, OpKind::FieldWrite { base: b, field, .. }
                if b == base && field.name.as_str() == field_name)
        })
    }
    /// The unique graph-wide value stored to `base.field_name`, or `None` if
    /// absent or conflicting. Without reaching-definition analysis, choosing
    /// among disagreeing stores would depend on block order, so the pass
    /// declines. This matches `BaseMallocRemover.compute_lifetimes`, which
    /// disables malloc optimization for multiple creation points ("aliasing
    /// problems"). Identical redundant writes retain all their locations so
    /// the dominance check can accept whichever one reaches the allocation.
    struct UniqueStore<'g> {
        field: &'g FieldDescriptor,
        value: &'g LinkArg,
        ty: &'g ValueType,
        locations: Vec<(usize, usize)>,
    }
    fn unique_store<'g>(
        graph: &'g FunctionGraph,
        base: &Variable,
        field_name: &str,
    ) -> Option<UniqueStore<'g>> {
        let mut found: Option<UniqueStore<'g>> = None;
        for (block_idx, block) in graph.blocks.iter().enumerate() {
            for (op_idx, op) in block.operations.iter().enumerate() {
                let OpKind::FieldWrite {
                    base: b,
                    field,
                    value,
                    ty,
                } = &op.kind
                else {
                    continue;
                };
                if b != base || field.name.as_str() != field_name {
                    continue;
                }
                match &mut found {
                    None => {
                        found = Some(UniqueStore {
                            field,
                            value,
                            ty,
                            locations: vec![(block_idx, op_idx)],
                        })
                    }
                    Some(store) if store.value == value => {
                        store.locations.push((block_idx, op_idx));
                    }
                    Some(_) => return None,
                }
            }
        }
        found
    }
    /// Whether this write must execute between `base`'s construction and the
    /// allocation site.  A graph-wide unique write is not enough: when the
    /// same aggregate crosses both arms of a branch, a write on only one arm
    /// does not initialize the allocation reached at the join.
    ///
    /// Compute dominance relative to the producer of `base`, rather than the
    /// function entry.  That preserves the valid sibling shape where each arm
    /// constructs and initializes its own header before the two header values
    /// merge: the other arm is not reachable from that producer and therefore
    /// is not a path along which that particular header can arrive uninitialized.
    fn store_dominates_site(
        graph: &FunctionGraph,
        base: &Variable,
        store: (usize, usize),
        site: (usize, usize),
    ) -> bool {
        let Some(definition) = graph.blocks.iter().enumerate().find_map(|(bi, block)| {
            block
                .operations
                .iter()
                .position(|op| op.result.as_ref() == Some(base))
                .map(|oi| (bi, oi))
        }) else {
            return false;
        };
        if store.0 == definition.0 && store.1 <= definition.1 {
            return false;
        }

        let reaches = |from: usize, to: usize, avoid: Option<usize>| {
            let mut seen = vec![false; graph.blocks.len()];
            let mut pending = vec![from];
            while let Some(block) = pending.pop() {
                if seen[block] || (avoid == Some(block) && block != from) {
                    continue;
                }
                if block == to {
                    return true;
                }
                seen[block] = true;
                pending.extend(graph.blocks[block].exits.iter().map(|link| link.target.0));
            }
            false
        };

        if store.0 == site.0 {
            return store.1 < site.1 && reaches(definition.0, store.0, None);
        }
        if !reaches(definition.0, store.0, None) || !reaches(store.0, site.0, None) {
            return false;
        }
        store.0 == definition.0 || !reaches(definition.0, site.0, Some(store.0))
    }
    /// Resolve `var` to producer variables through `Block.inputargs` phis.
    /// Stores name producers rather than phis, and every incoming root must
    /// later agree. `depth` terminates loop-carried phis. Upstream records the
    /// same relation by unioning link args with inputargs in
    /// `rpython/translator/backendopt/malloc.py:169`; this local pass recovers
    /// those endpoints on demand.
    fn store_roots(
        graph: &FunctionGraph,
        var: &Variable,
        depth: u32,
        out: &mut Vec<Variable>,
    ) -> bool {
        if depth == 0 {
            return false;
        }
        if graph
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .any(|o| o.result.as_ref() == Some(var))
        {
            out.push(var.clone());
            return true;
        }
        let Some((target_id, slot)) = graph.blocks.iter().find_map(|b| {
            b.inputargs
                .iter()
                .position(|arg| arg == var)
                .map(|slot| (b.id, slot))
        }) else {
            return false;
        };
        for block in &graph.blocks {
            for link in &block.exits {
                if link.target != target_id {
                    continue;
                }
                // A link whose arity does not cover the slot is a malformed
                // edge; refuse rather than read past it.
                let Some(arg) = link.args.get(slot).and_then(LinkArg::as_variable) else {
                    return false;
                };
                if !store_roots(graph, arg, depth - 1, out) {
                    return false;
                }
            }
        }
        !out.is_empty()
    }
    fn is_cast_instance_of(
        graph: &FunctionGraph,
        var: &Variable,
        of: &Variable,
        depth: u32,
    ) -> bool {
        if depth == 0 {
            return false;
        }
        graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .any(|op| {
                op.result.as_ref() == Some(var)
                    && matches!(
                        &op.kind,
                        OpKind::Call {
                            target: CallTarget::FunctionPath { segments, .. },
                            args,
                            ..
                        } if segments.first().map(String::as_str)
                            == Some(crate::runtime_names::shims::CAST_INSTANCE)
                            && args.first().and_then(LinkArg::as_variable).is_some_and(|inner| {
                                inner == of || is_cast_instance_of(graph, inner, of, depth - 1)
                            })
                    )
            })
    }
    fn refers_to_alloc(
        graph: &FunctionGraph,
        var: &Variable,
        alloc: &Variable,
        depth: u32,
    ) -> bool {
        if var == alloc {
            return true;
        }
        let mut roots = Vec::new();
        store_roots(graph, var, depth, &mut roots)
            && !roots.is_empty()
            && roots
                .iter()
                .all(|root| root == alloc || is_cast_instance_of(graph, root, alloc, depth))
    }
    /// The unique `core::ptr::write` that stores a value into `alloc`, and
    /// that value.  Several writes, or a write of a non-variable, decline.
    fn ptr_write_into(graph: &FunctionGraph, alloc: &Variable) -> Option<(usize, usize, Variable)> {
        let mut found: Option<(usize, usize, Variable)> = None;
        for (bi, block) in graph.blocks.iter().enumerate() {
            for (oi, op) in block.operations.iter().enumerate() {
                let OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    args,
                    ..
                } = &op.kind
                else {
                    continue;
                };
                if !is_core_ptr_write_path(segments, args) {
                    continue;
                }
                let Some(dest) = args[0].as_variable() else {
                    continue;
                };
                if !refers_to_alloc(graph, dest, alloc, 8) {
                    continue;
                }
                let Some(value) = args[1].as_variable() else {
                    continue;
                };
                let value = value.clone();
                if found.is_some() {
                    return None;
                }
                found = Some((bi, oi, value));
            }
        }
        found
    }
    fn assume_init_of(graph: &FunctionGraph, alloc: &Variable) -> Option<(usize, usize, Variable)> {
        let mut found: Option<(usize, usize, Variable)> = None;
        for (bi, block) in graph.blocks.iter().enumerate() {
            for (oi, op) in block.operations.iter().enumerate() {
                let OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    args,
                    ..
                } = &op.kind
                else {
                    continue;
                };
                if !is_assume_init_path(segments, args) {
                    continue;
                }
                let Some(operand) = args[0].as_variable() else {
                    continue;
                };
                if !refers_to_alloc(graph, operand, alloc, 8) {
                    continue;
                }
                let Some(result) = op.result.clone() else {
                    continue;
                };
                if found.is_some() {
                    return None;
                }
                found = Some((bi, oi, result));
            }
        }
        found
    }
    /// Resolve `var` to the constant address `terminal` reads off its producer.
    ///
    /// The walk itself is shared by every header spelling: it steps through
    /// `__cast_instance_intrinsic[<root>]` pointer reinterprets, and through the
    /// links when `var` is a `Block.inputargs` phi rather than an op result.
    /// A boxing cluster whose header stores sit before a call is exactly that
    /// shape — each call ends a block, so the header value crosses the
    /// boundary as a link arg while its producer stays in the predecessor.
    /// Every predecessor must agree: a phi merging two different pointers is
    /// not a constant and must keep the cluster unfused.  `depth` bounds the
    /// walk, which is what keeps a loop-carried phi (whose own link arg is
    /// itself) from recursing forever.
    fn resolve_addr<F>(
        graph: &FunctionGraph,
        var: &Variable,
        depth: u32,
        terminal: &F,
    ) -> Option<i64>
    where
        F: Fn(&FunctionGraph, &OpKind, u32) -> Option<i64>,
    {
        if depth == 0 {
            return None;
        }
        let producer = graph
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .find(|o| o.result.as_ref() == Some(var));
        let Some(producer) = producer else {
            let (target_id, slot) = graph.blocks.iter().find_map(|b| {
                b.inputargs
                    .iter()
                    .position(|arg| arg == var)
                    .map(|slot| (b.id, slot))
            })?;
            let mut resolved: Option<i64> = None;
            for block in &graph.blocks {
                for link in &block.exits {
                    if link.target != target_id {
                        continue;
                    }
                    // A link whose arity does not cover the slot is a malformed
                    // edge; treat it as a disagreement rather than reading past it.
                    let addr = link
                        .args
                        .get(slot)
                        .and_then(LinkArg::as_variable)
                        .and_then(|arg| resolve_addr(graph, arg, depth - 1, terminal))?;
                    match resolved {
                        None => resolved = Some(addr),
                        Some(seen) if seen == addr => {}
                        Some(_) => return None,
                    }
                }
            }
            return resolved;
        };
        if let OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } = &producer.kind
            && segments.first().map(String::as_str)
                == Some(crate::runtime_names::shims::CAST_INSTANCE)
            && let Some(operand) = args.first().and_then(LinkArg::as_variable)
        {
            return resolve_addr(graph, operand, depth - 1, terminal);
        }
        terminal(graph, &producer.kind, depth)
    }
    fn const_ref_addr(graph: &FunctionGraph, var: &Variable, depth: u32) -> Option<i64> {
        resolve_addr(graph, var, depth, &|_, kind, _| match kind {
            OpKind::ConstRefAddr(addr) => Some(*addr),
            _ => None,
        })
    }
    /// The `&T` a runtime `get_instantiate(&T)` call was handed.
    ///
    /// The helper is accepted only under the external runtime owner or the
    /// generic corpus owner. The caller separately proves that its sole
    /// argument is the exact class address stored in the header.
    fn get_instantiate_arg_addr(graph: &FunctionGraph, var: &Variable, depth: u32) -> Option<i64> {
        resolve_addr(graph, var, depth, &|graph, kind, depth| match kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                ..
            } if args.len() == 1
                && matches!(
                    segments.as_slice(),
                    [.., owner, leaf]
                        if matches!(owner.as_str(), "pyobject" | "object_model")
                            && leaf == "get_instantiate"
                ) =>
            {
                const_ref_addr(graph, &args[0], depth - 1)
            }
            _ => None,
        })
    }
    /// Whether the header named by `root` declares `field_name` in its layout.
    ///
    /// `root` is a `SyntheticTransparentCtor` result, so the header struct is
    /// the ctor's own name. An unresolvable owner or an unregistered layout
    /// answers `None`: neither is evidence that a field is missing. Callers
    /// separately reject stores that contradict a registered layout.
    fn header_declares_field(
        graph: &FunctionGraph,
        root: &Variable,
        field_name: &str,
        struct_field_attrs: &std::collections::HashMap<String, Vec<(String, ValueType)>>,
    ) -> Option<bool> {
        let owner = graph
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .find_map(|op| match (&op.result, &op.kind) {
                (Some(r), OpKind::Call { target, .. }) if r == root => {
                    synthetic_transparent_ctor_owner(target, graph.name.split("::").next())
                }
                _ => None,
            });
        let owner = owner?;
        registered_struct_layout(&owner, struct_field_attrs)
            .map(|rows| rows.iter().any(|(name, _)| name == field_name))
    }
    struct Payload {
        field: FieldDescriptor,
        value: LinkArg,
        ty: ValueType,
    }
    /// What the fusion keeps of the cluster's `ob_header`.
    ///
    /// `vtable` is the `ob_type` type-pointer the `NewWithVtable` carries; the
    /// runtime stamps both header slots from it.
    ///
    /// `w_class` is the class store to re-emit as a `FieldWrite` of its own,
    /// present exactly when the kept vtable does not stand for it — the
    /// cluster's `w_class` is not `get_instantiate(&T)` for the very `&T` its
    /// `ob_type` names.  Dropping such a store and re-synthesising the class
    /// from the vtable answers the wrong type: `w_long_from_raw`
    /// (longobject.rs) pairs `&LONG_TYPE` with `get_instantiate(&INT_TYPE)` so
    /// that `type(1 << 100) is int`, and `w_bytes_subclass_from_bytes`
    /// (bytesobject.rs) pins the caller's class on the shadow stack and reads
    /// it back for `type(B(b'x'))`.  Keeping it as a separate store is what
    /// upstream does with every field an allocation does not carry:
    /// `jtransform.py rewrite_op_malloc` emits `new_with_vtable`
    /// carrying only the sizedescr, and each remaining field arrives as its own
    /// `setfield`.
    struct HeaderPlan {
        /// `Some` lowers to `NewWithVtable` carrying the type word; `None` is
        /// upstream's plain `new` — a struct with no type word to stamp.
        vtable: Option<i64>,
        w_class: Option<Payload>,
    }
    let resolve_header_plan = |graph: &FunctionGraph, agg: &Variable, site: (usize, usize)| {
        let Some(header) = store_value(graph, agg, "ob_header", site)
            .or_else(|| store_value(graph, agg, "ob", site))
        else {
            crate::decline::record(
                VTABLE_GATE,
                "no-unique-ob_header-store",
                format_args!("{}", graph.name),
            );
            return None;
        };
        let mut roots = Vec::new();
        if !store_roots(graph, &header, 8, &mut roots) {
            crate::decline::record(
                VTABLE_GATE,
                "header-roots-unresolvable",
                format_args!("{}", graph.name),
            );
            return None;
        }
        // `None` until the first root is read; the inner `Option` is the
        // vtable when that root's type word is a constant and `None` when the
        // root writes no type word at all, so roots disagreeing about *which
        // allocation this is* are caught by the same equality that catches
        // roots naming different types.
        let mut resolved: Option<Option<i64>> = None;
        // `None` until the first root is read; `Some(None)` once a root's class
        // folds into the vtable, `Some(Some(_))` once one does not.
        let mut w_class: Option<Option<(FieldDescriptor, LinkArg, ValueType)>> = None;
        for root in &roots {
            // `rewrite_op_malloc` reads the vtable off the STRUCT, statically
            // (`heaptracker.get_vtable_for_gcstruct`), and forks on whether the
            // struct has one at all: `new_with_vtable` when it does, plain
            // `new` when it does not.  Neither arm ever writes the type word as
            // a field — `heaptracker.all_fielddescrs` drops `typeptr` from
            // every field list ("dealt otherwise"), so no setfield for it is
            // ever built.
            //
            // pyre recovers that same static answer from the registered header
            // layout, its rtyper carrying no `_hints`. Data flow then proves
            // that a declared type word has a dominating constant store; only
            // a layout that omits the word is the plain `new` case.
            let Some(declares_type_word) =
                header_declares_field(graph, root, "ob_type", struct_field_attrs)
            else {
                crate::decline::record(
                    VTABLE_GATE,
                    "header-layout-unregistered-or-ambiguous",
                    format_args!("{}", graph.name),
                );
                return None;
            };
            let type_store = unique_store(graph, root, "ob_type");
            let root_vtable = match type_store {
                Some(store)
                    if declares_type_word
                        && store.locations.iter().any(|&(block, op)| {
                            store_dominates_site(graph, root, (block, op), site)
                        }) =>
                {
                    let Some(vtable) = store
                        .value
                        .as_variable()
                        .and_then(|v| const_ref_addr(graph, v, 8))
                    else {
                        // A type word this pass cannot fold is NOT the `new`
                        // case, and re-emitting it as its own store is not an
                        // option: `optimize_setfield_gc` removes a `setfield_gc`
                        // whose descr `is_typeptr()`, keeping only a CONSTANT
                        // value and only as `known_class` — the same
                        // `heaptracker.all_fielddescrs` rule that keeps the
                        // type word out of the field lists. A plain `new` here would
                        // therefore ship an object whose type word is never
                        // written at all.  RPython cannot reach this shape,
                        // its typeptr being a property of the STRUCT rather
                        // than of the instance, so there is no arm to port for
                        // it: refuse, as upstream refuses the flavors its op
                        // cannot express.
                        //
                        // In a test fixture the commonest cause is simply that
                        // the `PyType` singleton addresses were not supplied,
                        // so the `&T` read stayed a residual call rather than a
                        // `ConstRefAddr`.
                        crate::decline::record(
                            VTABLE_GATE,
                            "ob_type-not-a-constant-address",
                            format_args!("{}", graph.name),
                        );
                        return None;
                    };
                    Some(vtable)
                }
                Some(_) if !declares_type_word => {
                    crate::decline::record(
                        VTABLE_GATE,
                        "ob_type-store-contradicts-layout",
                        format_args!("{}", graph.name),
                    );
                    return None;
                }
                // A declared type word must have one reaching constant store.
                // Absence, conflicting writes, or a write that does not
                // dominate the allocation are all incomplete initialization,
                // never evidence for upstream's plain `new` arm.
                _ if declares_type_word => {
                    crate::decline::record(
                        VTABLE_GATE,
                        if any_store(graph, root, "ob_type") {
                            "ob_type-store-conflicting-or-nondominating"
                        } else {
                            "declared-ob_type-store-missing"
                        },
                        format_args!("{}", graph.name),
                    );
                    return None;
                }
                None => None,
                Some(_) => unreachable!("layout contradiction handled above"),
            };
            let Some(declares_class_word) =
                header_declares_field(graph, root, "w_class", struct_field_attrs)
            else {
                return None;
            };
            let store = match unique_store(graph, root, "w_class") {
                // A store to a field absent from the registered layout makes
                // the layout evidence self-contradictory, so keep the original
                // allocation rather than guessing which source is correct.
                Some(_) if !declares_class_word => {
                    crate::decline::record(
                        VTABLE_GATE,
                        "w_class-store-contradicts-layout",
                        format_args!("{}", graph.name),
                    );
                    return None;
                }
                Some(store)
                    if store.locations.iter().any(|&(block, op)| {
                        store_dominates_site(graph, root, (block, op), site)
                    }) =>
                {
                    // With no vtable on the allocation there is nothing the
                    // class store could fold into, so it is always kept.
                    let folds = root_vtable.is_some_and(|root_vtable| {
                        store
                            .value
                            .as_variable()
                            .and_then(|value| get_instantiate_arg_addr(graph, value, 8))
                            == Some(root_vtable)
                    });
                    (!folds).then(|| (store.field.clone(), store.value.clone(), store.ty.clone()))
                }
                // RPython's root OBJECT declares only `typeptr`. With no
                // per-instance class word, there is nothing that can disagree
                // with the vtable carried by `NewWithVtable`, and nothing
                // downstream can read one either: every `w_class` consumer
                // keys on a field descriptor named `w_class`
                // (`descr.rs FieldDescr::is_w_class`), which a struct that
                // never declares the field cannot produce.
                None if !declares_class_word => None,
                // A layout that declares `w_class` still needs a unique store
                // proving that the vtable stands for that per-instance class.
                Some(_) | None => {
                    crate::decline::record(
                        VTABLE_GATE,
                        "no-unique-w_class-store",
                        format_args!("{}", graph.name),
                    );
                    return None;
                }
            };
            match &w_class {
                None => w_class = Some(store),
                // Predecessors disagreeing about the class — one folding into
                // the vtable and another not, or two naming different classes
                // — merge into one malloc with no single store standing for
                // the cluster.
                Some(seen) => {
                    let agrees = match (seen, &store) {
                        (None, None) => true,
                        (Some((seen_field, seen_value, _)), Some((field, value, _))) => {
                            seen_field == field && seen_value == value
                        }
                        _ => false,
                    };
                    if !agrees {
                        crate::decline::record(
                            VTABLE_GATE,
                            "header-roots-disagree-about-w_class",
                            format_args!("{}", graph.name),
                        );
                        return None;
                    }
                }
            }
            match resolved {
                None => resolved = Some(root_vtable),
                Some(seen) if seen == root_vtable => {}
                // Predecessors building headers for different types merge into
                // one malloc: no single vtable stands for the whole cluster.
                // A root that carries a type word meeting one that does not
                // lands here too: the allocation cannot both stamp the word and
                // have none to stamp.
                Some(_) => {
                    crate::decline::record(
                        VTABLE_GATE,
                        "header-roots-name-different-vtables",
                        format_args!("{}", graph.name),
                    );
                    return None;
                }
            }
        }
        // `store_roots` only answers `true` for a non-empty `roots`, and every
        // iteration either sets `resolved` or returns, so the `?` below is
        // unreachable rather than a decline path — it is not recorded, and a
        // `None` from there would be a bug in that invariant, not a refusal.
        Some(HeaderPlan {
            vtable: resolved?,
            w_class: w_class
                .flatten()
                .map(|(field, value, ty)| Payload { field, value, ty }),
        })
    };

    struct Site {
        block: usize,
        op: usize,
        aggregate: crate::flowspace::model::Variable,
        result: crate::flowspace::model::Variable,
        owner: String,
        /// `Some` lowers to `NewWithVtable` carrying the type word; `None`
        /// lowers to the plain `New` upstream emits for a struct with none.
        vtable: Option<i64>,
        /// The header's `w_class` store when the vtable does not stand for it,
        /// re-emitted ahead of the payload stores so the whole rewrite lands in
        /// struct order (`ob_header` is field zero of every boxing struct).
        w_class: Option<Payload>,
        payloads: Vec<Payload>,
        /// Cluster ops the New rewrite replaces: the `ptr::write` that
        /// initialized an uninit box, and the `new_uninit` call when New is
        /// emitted at `assume_init`.
        dead_ops: Vec<(usize, usize)>,
    }

    let mut sites: Vec<Site> = Vec::new();
    for (bi, block) in graph.blocks.iter().enumerate() {
        for (oi, op) in block.operations.iter().enumerate() {
            // The two shape tests below are the population filter, not a
            // decline: every operation in the graph is offered to them and
            // almost all fail.  Recording here would make the instrument
            // part of the population it measures — the fuse's count would
            // be dominated by ops that were never boxing clusters.
            // Recording starts once the op IS a `malloc_typed` call, i.e.
            // once the pass has committed to lowering it.
            let OpKind::Call { target, args, .. } = &op.kind else {
                continue;
            };
            // Two allocation spellings: `lltype::malloc*` of the aggregate,
            // and the uninit-box cluster `Box::new_uninit` + `ptr::write` +
            // `assume_init`.  Both lower to the same `New` / `NewWithVtable`
            // plus field stores.  Missing the shape is the population filter.
            let malloc_flavor = gc_malloc_flavor(target).filter(|_| args.len() == 1);
            let is_uninit = matches!(
                target,
                CallTarget::FunctionPath { segments, .. } if is_box_new_uninit_path(segments, args)
            );
            let (flavor, agg_operand, result, rewrite_bi, rewrite_oi, dead_ops) =
                if let Some(flavor) = malloc_flavor {
                    let Some(result) = op.result.clone() else {
                        crate::decline::record(
                            FUSE_GATE,
                            "malloc-call-has-no-result-var",
                            format_args!("{}", graph.name),
                        );
                        continue;
                    };
                    let Some(agg_operand) = args[0].as_variable().cloned() else {
                        crate::decline::record(
                            FUSE_GATE,
                            "aggregate-roots-unresolvable",
                            format_args!("{}", graph.name),
                        );
                        continue;
                    };
                    (flavor, agg_operand, result, bi, oi, Vec::new())
                } else if is_uninit {
                    let Some(uninit) = op.result.clone() else {
                        continue;
                    };
                    let Some((write_bi, write_oi, written)) = ptr_write_into(graph, &uninit) else {
                        continue;
                    };
                    let mut dead_ops = vec![(write_bi, write_oi)];
                    let (rewrite_bi, rewrite_oi, result) =
                        if let Some((abi, aoi, ares)) = assume_init_of(graph, &uninit) {
                            dead_ops.push((bi, oi));
                            (abi, aoi, ares)
                        } else {
                            (bi, oi, uninit)
                        };
                    ("malloc", written, result, rewrite_bi, rewrite_oi, dead_ops)
                } else {
                    continue;
                };
            // The aggregate itself can reach the malloc as a `Block.inputargs`
            // phi, not only its header: a constructor that builds the struct up
            // front and then branches — `w_float_new` builds the `W_FloatObject`
            // before `gc_interp::enabled()` and `try_gc_alloc_stable_raw`, each
            // of which ends a block — leaves the ctor in a dominator.  Every
            // lookup below keys on `%agg` by exact variable, so resolve the phi
            // to the ctor it stands for before asking any of them.  A merge of
            // two different aggregates has no single owner or payload set, and
            // declines.  Using the root also keeps `Site.aggregate` naming the
            // variable a `core::ptr::write` arm stores, which is what
            // `sink_fused_boxing_aggregates_at_raw_writes` matches on.
            let mut agg_roots = Vec::new();
            if !store_roots(graph, &agg_operand, 8, &mut agg_roots) {
                crate::decline::record(
                    FUSE_GATE,
                    "aggregate-roots-unresolvable",
                    format_args!("{}", graph.name),
                );
                continue;
            }
            let Some((agg, other_roots)) = agg_roots.split_first() else {
                crate::decline::record(
                    FUSE_GATE,
                    "aggregate-has-no-roots",
                    format_args!("{}", graph.name),
                );
                continue;
            };
            if other_roots.iter().any(|root| root != agg) {
                crate::decline::record(
                    FUSE_GATE,
                    "aggregate-roots-disagree",
                    format_args!("{}", graph.name),
                );
                continue;
            }
            // `%agg` must be a `SyntheticTransparentCtor` for a known boxing
            // struct.  Search graph-wide: the ctor and the malloc call land in
            // the same block, but the field stores feeding the aggregate may sit
            // in earlier blocks (each preceding call ends a block).
            let owner = graph
                .blocks
                .iter()
                .flat_map(|b| &b.operations)
                .find_map(|o| match (&o.result, &o.kind) {
                    (Some(r), OpKind::Call { target, .. }) if r == agg => {
                        synthetic_transparent_ctor_owner(target, graph.name.split("::").next())
                    }
                    _ => None,
                });
            let Some(owner) = owner else {
                crate::decline::record(
                    FUSE_GATE,
                    "aggregate-is-not-a-synthetic-ctor",
                    format_args!("{}", graph.name),
                );
                continue;
            };
            let Some(fields) = payload_fields(&owner, struct_field_attrs) else {
                // Either the struct has no registered field layout at all,
                // or its leaf name is ambiguous across the layout map.  Both
                // leave `malloc_typed` residual; `registered_layout` is where
                // they part, and neither is visible in the fused count.
                crate::decline::record(
                    FUSE_GATE,
                    "struct-layout-unregistered-or-ambiguous",
                    format_args!("{owner} in {}", graph.name),
                );
                continue;
            };
            // Resolve every payload field's store: `FieldWrite { base: %agg,
            // field.name == payload }`.  A malformed cluster missing any payload
            // store is left untouched so the annotate wall still flags it.
            // Conflicting stores also decline: no reaching write is known.
            let mut payloads = Vec::with_capacity(fields.len());
            let mut complete = true;
            for (field_name, payload_ty) in &fields {
                // Inlined nested structs (`W_ListObject.int_items: IntArray`)
                // are not payload setfields of the boxing cluster.
                // `NewWithVtable` zero-fills them (`clear_gc_fields`); the
                // owner then stores dotted leaves (`int_items.block`)
                // through `list.int_set_items` / `heaptracker.py
                // all_fielddescrs`. A by-value store of the nested struct
                // would apply the inner offsets to the outer object and
                // overlay `ob_header`.
                if expand_inline_struct(&owner, field_name).is_some() {
                    continue;
                }
                let found = unique_store(graph, agg, field_name.as_str()).and_then(|store| {
                    store
                        .locations
                        .iter()
                        .any(|&(block, op)| {
                            store_dominates_site(graph, agg, (block, op), (rewrite_bi, rewrite_oi))
                        })
                        .then(|| (store.field.clone(), store.value.clone()))
                });
                match found {
                    Some((field, value)) => payloads.push(Payload {
                        field,
                        value,
                        ty: payload_ty.clone(),
                    }),
                    None => {
                        complete = false;
                        break;
                    }
                }
            }
            if !complete {
                crate::decline::record(
                    FUSE_GATE,
                    "payload-store-missing-or-conflicting",
                    format_args!("{owner} in {}", graph.name),
                );
                continue;
            }
            // Leave the cluster unfused when the `ob_header.ob_type` store
            // carries no resolvable constant type-pointer.  A `NewWithVtable`
            // requires a non-zero type pointer — a zero vtable would stamp a
            // null `ob_type`, and the assembler fail-closes on it
            // (`assembler.rs` `new_with_vtable`).  Untouched, the
            // `lltype::malloc_typed` stays a residual call, the same graceful
            // outcome an incomplete payload store takes above.  The type
            // pointer may be genuinely absent — the `PyType` singleton
            // addresses unavailable (`HostStaticAddrs.pytypes` empty, as in the
            // pure-translate registry fixture), so the `&FLOAT_TYPE` read
            // lowers to a residual `FunctionPath` call rather than a
            // `ConstRefAddr`; the production driver supplies those addresses,
            // so there the pointer resolves.  Predecessors that disagree about
            // the type or the class also decline — see `HeaderPlan`.
            //
            // Which of those it was is not knowable from the `None`; the
            // `model::resolve_header_plan` rows are where that is recorded,
            // and this row is the count of clusters the fuse gave up on for
            // any header reason at all.
            let Some(header) = resolve_header_plan(graph, agg, (rewrite_bi, rewrite_oi)) else {
                crate::decline::record(
                    FUSE_GATE,
                    "vtable-unresolved",
                    format_args!("{owner} in {}", graph.name),
                );
                continue;
            };
            // `rewrite_op_malloc` refuses a malloc its allocation op cannot
            // express rather than lowering it approximately: a `nonmovable`
            // hint raises `UnsupportedMallocFlags`.  The `New` arm carries no
            // type word to stamp and no long-published `SizeDescr` flavor
            // behind it, so lowering a non-movable cluster into one would
            // silently re-flavor the object into a collected nursery block.
            // `descr.rs`'s instance group states what follows: a minor
            // collection inside a residual moves the fresh object, and the
            // caller finishes writing into the dead pre-move copy.
            if header.vtable.is_none() && !allocates_movable(flavor) {
                crate::decline::record(
                    FUSE_GATE,
                    "unsupported-malloc-flags-nonmovable",
                    format_args!("{owner} in {}", graph.name),
                );
                continue;
            }
            sites.push(Site {
                block: rewrite_bi,
                op: rewrite_oi,
                aggregate: agg.clone(),
                result: result.clone(),
                owner,
                vtable: header.vtable,
                w_class: header.w_class,
                payloads,
                dead_ops,
            });
        }
    }

    let fused = sites.len();
    let fused_aggregates: Vec<_> = sites.iter().map(|site| site.aggregate.clone()).collect();
    // Rewrite highest `(block, op)` first so an insert does not shift
    // not-yet-processed rewrite indices. Removals of earlier `new_uninit` /
    // `ptr::write` still shift remaining sites; update those indices after
    // each mutation. Uninit-box rewrite locations are `assume_init`, which
    // can invert allocation order inside one block.
    sites.sort_by_key(|site| (site.block, site.op));
    while let Some(site) = sites.pop() {
        let inserted = usize::from(site.w_class.is_some()) + site.payloads.len();
        {
            let block = &mut graph.blocks[site.block];
            block.operations[site.op] = SpaceOperation {
                result: Some(site.result.clone()),
                // `rewrite_op_malloc` picks the opcode from what the struct
                // carries: with a static vtable it is `new_with_vtable`, without
                // one it is `new`.  Both take the size descriptor and nothing else.
                kind: match site.vtable {
                    Some(vtable) => OpKind::NewWithVtable {
                        owner: site.owner,
                        vtable,
                    },
                    None => OpKind::New { owner: site.owner },
                },
            };
            // The kept stores follow the allocation, in struct order: the header's
            // `w_class` where the vtable does not stand for it, then the payloads.
            // Each is a plain `FieldWrite` the assembler lowers to its own
            // `setfield_gc`, which is the shape
            // `Transformer.rewrite_op_malloc` leaves every field the allocation
            // itself does not carry.
            for (k, payload) in site.w_class.into_iter().chain(site.payloads).enumerate() {
                block.operations.insert(
                    site.op + 1 + k,
                    SpaceOperation {
                        result: None,
                        kind: OpKind::FieldWrite {
                            base: site.result.clone(),
                            field: payload.field,
                            value: payload.value,
                            ty: payload.ty,
                        },
                    },
                );
            }
        }
        for remaining in &mut sites {
            if remaining.block == site.block && remaining.op > site.op {
                remaining.op += inserted;
            }
            for (sbi, soi) in &mut remaining.dead_ops {
                if *sbi == site.block && *soi > site.op {
                    *soi += inserted;
                }
            }
        }
        let mut dead = site.dead_ops;
        for (dbi, doi) in &mut dead {
            if *dbi == site.block && *doi > site.op {
                *doi += inserted;
            }
        }
        dead.sort_unstable();
        for (dbi, doi) in dead.into_iter().rev() {
            if dbi == site.block && doi == site.op {
                continue;
            }
            graph.blocks[dbi].operations.remove(doi);
            for remaining in &mut sites {
                if remaining.block == dbi && remaining.op > doi {
                    remaining.op -= 1;
                }
                for (sbi, soi) in &mut remaining.dead_ops {
                    if *sbi == dbi && *soi > doi {
                        *soi -= 1;
                    }
                }
            }
        }
    }
    sink_fused_boxing_aggregates_at_raw_writes(graph, &fused_aggregates);
    fused
}

/// Rematerialize a fused boxing aggregate only at a raw `ptr::write` use.
///
/// `gc_interp`'s stable-allocation arm writes the original by-value Rust
/// aggregate into memory, while the ordinary arm passes that same aggregate
/// to `lltype::malloc_typed`.  After the latter is lowered to
/// `NewWithVtable`, leaving the shared aggregate in the dominator would make
/// its synthetic constructors execute on the ordinary JIT path even though
/// only the GC-interpreter arm still reads it.  RPython's
/// `remove_simple_mallocs` performs the equivalent escape-sensitive motion:
/// keep the materialization at the escaping store, and let the dead-variable
/// sweep remove it from paths on which it does not escape.
///
/// A rename map is the direct equivalent of RPython's per-clone variable
/// mapping: it preserves the nested header/outer-object shape without a
/// side-table that survives this local graph rewrite.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
fn sink_fused_boxing_aggregates_at_raw_writes(
    graph: &mut FunctionGraph,
    aggregates: &[crate::flowspace::model::Variable],
) -> usize {
    use crate::flowspace::model::Variable;
    use std::collections::{HashMap, HashSet};

    let is_ctor = |op: &SpaceOperation| {
        matches!(
            op.kind,
            OpKind::Call {
                target: CallTarget::SyntheticTransparentCtor { .. },
                ..
            }
        )
    };
    let is_raw_write = |kind: &OpKind| {
        matches!(kind, OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, args, .. }
            if is_core_ptr_write_path(segments, args))
    };

    let mut sinks = Vec::new();
    for aggregate in aggregates {
        for (bi, block) in graph.blocks.iter().enumerate() {
            for (oi, op) in block.operations.iter().enumerate() {
                if is_raw_write(&op.kind)
                    && matches!(&op.kind, OpKind::Call { args, .. } if &args[1] == aggregate)
                {
                    sinks.push((bi, oi, aggregate.clone()));
                }
            }
        }
    }
    sinks.sort_by_key(|(bi, oi, _)| (*bi, *oi));

    let mut moved = 0;
    for (bi, oi, aggregate) in sinks.into_iter().rev() {
        // Include the outer aggregate plus nested aggregate values stored into
        // it (notably `PyObject ob_header`).  Scalar payloads and type-pointer
        // casts remain ordinary operands and are threaded to the sink by the
        // existing SSA-to-SSI repair pass.
        let mut allocs = HashSet::from([aggregate.clone()]);
        loop {
            let mut added = false;
            for op in graph.blocks.iter().flat_map(|block| &block.operations) {
                let OpKind::FieldWrite { base, value, .. } = &op.kind else {
                    continue;
                };
                if !allocs.contains(base) {
                    continue;
                }
                let Some(value) = value.as_variable() else {
                    continue;
                };
                let produced_by_ctor =
                    graph
                        .blocks
                        .iter()
                        .flat_map(|block| &block.operations)
                        .any(|candidate| {
                            candidate.result.as_ref() == Some(value) && is_ctor(candidate)
                        });
                if produced_by_ctor && allocs.insert(value.clone()) {
                    added = true;
                }
            }
            if !added {
                break;
            }
        }

        let template: Vec<_> = graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .filter(|op| {
                op.result
                    .as_ref()
                    .is_some_and(|result| allocs.contains(result) && is_ctor(op))
                    || matches!(&op.kind, OpKind::FieldWrite { base, .. } if allocs.contains(base))
            })
            .cloned()
            .collect();
        if template.is_empty() {
            continue;
        }

        let mut renaming = HashMap::new();
        for alloc in &allocs {
            let mut fresh = Variable::new();
            fresh.rename_from(alloc);
            renaming.insert(alloc.clone(), fresh);
        }
        let remap = |var: &Variable| renaming.get(var).cloned().unwrap_or_else(|| var.clone());
        let cloned: Vec<_> = template
            .iter()
            .map(|op| SpaceOperation {
                result: op.result.as_ref().map(&remap),
                kind: crate::inline::remap_op_kind(&op.kind, &remap),
            })
            .collect();

        let block = &mut graph.blocks[bi];
        if let OpKind::Call { args, .. } = &mut block.operations[oi].kind {
            args[1] = remap(&aggregate).into();
        }
        moved += cloned.len();
        block.operations.splice(oi..oi, cloned);
    }
    moved
}

/// Re-thread op operands the boxing lowering left referenced across a block
/// boundary without a matching inputarg.  [`fuse_boxing_alloc`] relocates the
/// payload `FieldWrite` to the `malloc_typed` site and the dead-var sweeps
/// (`prune_dead_boxing_remnants` / `prune_dead_phis`) then strip the inputargs
/// that carried the boxing cluster's cross-block values — the scalar payload
/// (a parameter defined before the `get_instantiate` call that ends the entry
/// block) and the `__cast_instance_intrinsic` chain that turns the `NewWithVtable`
/// result into the returned `PyObjectRef`.  This pass runs *after* those
/// sweeps and re-threads every such operand through the predecessor chain,
/// restoring the per-block operand invariant the adapter
/// (`flowspace_adapter::function_graph_to_flowspace`) requires — every
/// referenced operand defined as a block inputarg or op result.
///
/// An operand is threaded only when it is available on every predecessor path
/// into the use block.  This is the same condition `SSA_to_SSI` enforces while
/// adding one inputarg to every block and one argument to every incoming link
/// (`rpython/translator/backendopt/ssa.py`).  Diamond joins are valid
/// when the definition dominates the split; a value defined in only one arm,
/// or one with no upstream definition, is left untouched so the adapter can
/// Skip the malformed graph.  Back edges provisionally satisfy the recursive
/// check, but a separate ancestor-definition check prevents a source-less SCC
/// from manufacturing a definition.  Graphs with no cross-block-undefined
/// operand collect no work and stay byte-identical.
pub fn thread_undefined_op_operands(graph: &mut FunctionGraph) {
    use crate::flowspace::model::Variable;
    use std::collections::{HashMap, HashSet};

    // target block id → source block ids feeding it.
    let mut preds: HashMap<BlockId, Vec<BlockId>> = HashMap::new();
    for b in &graph.blocks {
        for e in &b.exits {
            preds.entry(e.target).or_default().push(b.id);
        }
    }

    fn ancestor_has_definition(
        graph: &FunctionGraph,
        preds: &HashMap<BlockId, Vec<BlockId>>,
        block: BlockId,
        var: &Variable,
        seen: &mut HashSet<BlockId>,
    ) -> bool {
        if !seen.insert(block) {
            return false;
        }
        graph.variable_defined_in_block(block, var)
            || preds.get(&block).is_some_and(|ps| {
                ps.iter()
                    .any(|&pred| ancestor_has_definition(graph, preds, pred, var, seen))
            })
    }

    fn available_on_every_predecessor_path(
        graph: &FunctionGraph,
        preds: &HashMap<BlockId, Vec<BlockId>>,
        block: BlockId,
        var: &Variable,
        visiting: &mut HashSet<BlockId>,
        memo: &mut HashMap<BlockId, bool>,
    ) -> bool {
        if graph.variable_defined_in_block(block, var) {
            return true;
        }
        if let Some(&known) = memo.get(&block) {
            return known;
        }
        // `ensure_variable_at_block` installs the target inputarg before
        // following a back edge.  Treat the same in-progress cycle as
        // available here; an acyclic entry path still has to reach a real
        // definition, and `ancestor_has_definition` rejects a source-less SCC.
        if !visiting.insert(block) {
            return true;
        }
        let available = preds.get(&block).is_some_and(|ps| {
            !ps.is_empty()
                && ps.iter().all(|&pred| {
                    available_on_every_predecessor_path(graph, preds, pred, var, visiting, memo)
                })
        });
        visiting.remove(&block);
        memo.insert(block, available);
        available
    }

    let mut work: Vec<(BlockId, Variable)> = Vec::new();
    for block in &graph.blocks {
        if block.dead {
            continue;
        }
        for op in &block.operations {
            for var in crate::inline::op_variable_refs(&op.kind) {
                if graph.variable_defined_in_block(block.id, &var) {
                    continue;
                }
                let has_definition =
                    ancestor_has_definition(graph, &preds, block.id, &var, &mut HashSet::new());
                let available = available_on_every_predecessor_path(
                    graph,
                    &preds,
                    block.id,
                    &var,
                    &mut HashSet::new(),
                    &mut HashMap::new(),
                );
                if has_definition && available {
                    work.push((block.id, var));
                }
            }
        }
    }
    // `ensure_variable_at_block` is idempotent, so a duplicate (block, var)
    // pair from two operands referencing the same Variable is a harmless no-op
    // on the second call.
    for (block_id, var) in work {
        graph.ensure_variable_at_block(block_id, &var);
    }
}

/// Sweep the construct-on-stack header remnants that [`fuse_boxing_alloc`]
/// orphans.  Once `malloc_typed(struct)` becomes a `NewWithVtable` (which
/// carries the type pointer / `w_class` through its vtable descriptor) the
/// original aggregate ctor, the inner `PyObject` header ctor, their
/// `ob_header` / `ob_type` / `w_class` field stores, and the
/// `__cast_instance_intrinsic` casts feeding `ob_type` / `w_class` are all dead.
/// The exception is a `w_class` the vtable cannot stand for: `fuse_boxing_alloc`
/// re-emits that store on the new object, so its value stays live and the
/// liveness gate below keeps the producer.
///
/// `front::mir` threads each of those dead values across the block boundary
/// the preceding `Call` opens (`get_instantiate` / `malloc_typed` each end a
/// block) by reusing the *same* `Variable` as both the producing block's op
/// result and the successor block's inputarg.  Neither generic sweep reclaims
/// the cluster in that shape:
///
///   * [`remove_dead_aggregates`] (and the old form of this sweep) counts
///     every `Link.arg` as an unconditional read, so the threaded copy keeps
///     the producer alive even though nothing on the far side reads it.
///   * [`prune_dead_phis`] keeps the producer because a `FieldWrite` is
///     side-effecting and pins its `base`.
///
/// So this sweep runs `transform_dead_op_vars`' dependency-flow liveness
/// (`simplify.py:425-479`) — a `Link.arg` is live iff the target inputarg it
/// feeds is live, not unconditionally — combined with the malloc-removal
/// exemption (`malloc.py remove_simple_mallocs`): a store into a fresh
/// aggregate nothing reads is itself dead, so a `SyntheticTransparentCtor`
/// `FieldWrite.base` is *not* a liveness root.  The exemption is scoped to
/// ctor bases — a store *through* an aliasing or loaded base (a cast or a
/// load) is a real heap side effect and roots its base like any other op.
/// The header producers eligible for removal are all side-effect-free:
/// a `SyntheticTransparentCtor` stack construct, a `__cast_instance_intrinsic`
/// pointer reinterpret (`exception_cannot_occur` → `cast_pointer`), and the
/// `runtime_object::object_model::get_instantiate` read of a type's `instantiate`
/// slot feeding the dropped `w_class`.  `get_instantiate` is an `Acquire`
/// atomic load of an init-once slot (the `set_instantiate` `Release` mutator
/// writes it during `init_typeobjects`); it is removable not because the slot
/// is immutable but because the load is observation-free — dropping a load
/// whose result no reader reaches only relaxes ordering nothing depends on.
/// Their operands route through `dependencies[result]` like a pure op rather than
/// pinning — so a dead cast drops its constant feed and a dead `get_instantiate`
/// drops the cast feeding it.  Liveness still gates removal, so a producer the
/// flow reaches (e.g. a `get_instantiate` whose result is genuinely used) is
/// never removed.  One global liveness pass reaches the whole cross-block
/// cluster at once; the inputargs / link args the removed producers leave
/// dangling (and the now-dead address constants) are reclaimed by the
/// [`prune_dead_phis`] pass the lowering runs next.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
pub(crate) fn prune_dead_boxing_remnants(graph: &mut FunctionGraph) -> usize {
    use crate::flowspace::model::Variable;
    use std::collections::HashMap;

    let is_removable_producer = |kind: &OpKind| match kind {
        OpKind::Call {
            target: CallTarget::SyntheticTransparentCtor { .. },
            ..
        } => true,
        OpKind::Call {
            target: CallTarget::FunctionPath { segments, .. },
            args,
            ..
        } => {
            // `__cast_instance_intrinsic` — the front-end pointer-downcast
            // narrow (`cast_instance_call`: operand + const(root)).
            // Pin the arity so an unrelated multi-arg path that happens to
            // share the synthetic marker leaf is never swept as a cast.
            let is_cast = segments.first().map(String::as_str)
                == Some(crate::runtime_names::shims::CAST_INSTANCE)
                && args.len() == 2;
            // The same single-argument runtime helper recognized by
            // `get_instantiate_arg_addr`; its result feeds only the class word
            // removed with the dead header. Restrict the owner so an unrelated
            // function with the same leaf is never classified as removable.
            let is_get_instantiate = args.len() == 1
                && matches!(
                    segments.as_slice(),
                    [.., owner, leaf]
                        if matches!(owner.as_str(), "pyobject" | "object_model")
                            && leaf == "get_instantiate"
                );
            // `boxed::Box::new_uninit()` — the no-arg heap-box allocation the
            // `vec![…]` lowering (`box_assume_init_into_vec_unsafe(box [..])`)
            // opens.  Once a consumer rewrite turns the `vec!` into a
            // `newlist`, the box result flows only into the `FieldWrite` that
            // stored the now-unused `Array` aggregate, so the box is a fresh
            // alloc nothing reads.  Pin the arity so an unrelated multi-arg
            // path sharing the leaf is never swept.
            is_cast || is_get_instantiate || is_box_new_uninit_path(segments, args)
        }
        _ => false,
    };

    let block_index: HashMap<BlockId, usize> = graph
        .blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.id, i))
        .collect();

    // Fresh heap allocations are the only bases whose field stores may be
    // dropped: a store into an allocation nothing reads is itself dead
    // (`malloc.py remove_simple_mallocs`).  Two producer shapes qualify — a
    // `SyntheticTransparentCtor` stack construct and a no-arg
    // `boxed::Box::new_uninit()` heap box (the `vec![…]` lowering's fresh
    // allocation, which nothing aliases legitimately).  A store *through* an
    // aliasing or loaded base (`__cast_instance_intrinsic` / `get_instantiate` / a
    // parameter / …) is a real heap side effect, so the exemption is scoped to
    // these fresh-allocation results — every other store roots its base.
    let fresh_alloc_results: HashSet<Variable> = graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .filter(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::SyntheticTransparentCtor { .. },
                ..
            } => true,
            OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                args,
                ..
            } => is_box_new_uninit_path(segments, args),
            _ => false,
        })
        .filter_map(|op| op.result.clone())
        .collect();

    // Liveness roots + dependency edges (`simplify.py:425-462`) with the two
    // boxing exemptions above.
    let mut read_vars: HashSet<Variable> = HashSet::new();
    let mut dependencies: HashMap<Variable, Vec<Variable>> = HashMap::new();
    for block in &graph.blocks {
        for op in &block.operations {
            match &op.kind {
                // A store into a fresh allocation is live iff the allocation
                // is live; the `base` is deliberately not rooted, so an unread
                // allocation's stores die with it (malloc-removal exemption).
                OpKind::FieldWrite { base, value, .. } if fresh_alloc_results.contains(base) => {
                    if let Some(var) = value.as_variable() {
                        dependencies
                            .entry(base.clone())
                            .or_default()
                            .push(var.clone());
                    }
                }
                // A store through any other base is a real heap side effect:
                // root both the base and the stored value.
                OpKind::FieldWrite { base, value, .. } => {
                    read_vars.insert(base.clone());
                    if let Some(var) = value.as_variable() {
                        read_vars.insert(var.clone());
                    }
                }
                // Side-effect-free header producers route operands like a pure
                // op, so a dead producer drops its feed instead of pinning it.
                kind if is_removable_producer(kind) => match &op.result {
                    Some(result) => dependencies
                        .entry(result.clone())
                        .or_default()
                        .extend(crate::inline::op_variable_refs(kind)),
                    None => read_vars.extend(crate::inline::op_variable_refs(kind)),
                },
                // Every other op is side-effecting: its operands are real reads.
                kind => read_vars.extend(crate::inline::op_variable_refs(kind)),
            }
        }
        match &block.exitswitch {
            Some(ExitSwitch::Value(var)) => {
                read_vars.insert(var.clone());
            }
            // A fused comparison reads its operands at the branch.
            Some(ExitSwitch::Fused { args, .. }) => {
                for arg in args {
                    read_vars.insert(arg.clone());
                }
            }
            Some(ExitSwitch::LastException) | None => {}
        }
        // `simplify.transform_dead_op_vars` treats every terminal-block
        // input argument as read.
        if block.exits.is_empty() {
            read_vars.extend(block.inputargs.iter().cloned());
        }
        // Cross-block: a link arg is live iff the target inputarg it feeds is.
        for link in &block.exits {
            let Some(&ti) = block_index.get(&link.target) else {
                continue;
            };
            let target_iargs = &graph.blocks[ti].inputargs;
            for (arg, target_iarg) in link.args.iter().zip(target_iargs.iter()) {
                if let Some(arg_var) = arg.as_variable() {
                    dependencies
                        .entry(target_iarg.clone())
                        .or_default()
                        .push(arg_var.clone());
                }
            }
        }
    }
    // Real parameters are always live (`simplify.py:431-433` start inputargs).
    if let Some(&i) = block_index.get(&graph.startblock) {
        read_vars.extend(graph.blocks[i].inputargs.iter().cloned());
    }
    // Backward flow (`simplify.py:471-479`).
    let mut pending: Vec<Variable> = read_vars.iter().cloned().collect();
    while let Some(var) = pending.pop() {
        if let Some(deps) = dependencies.get(&var).cloned() {
            for dep in deps {
                if read_vars.insert(dep.clone()) {
                    pending.push(dep);
                }
            }
        }
    }

    // A removable producer whose result the flow leaves unread is dead.  A
    // ctor is eligible only when its every store is result-less (dropping it
    // leaves no dangling definition), matching `remove_dead_aggregates`.
    let dead: HashSet<Variable> = graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .filter_map(|op| {
            let result = op.result.as_ref()?;
            if read_vars.contains(result) || !is_removable_producer(&op.kind) {
                return None;
            }
            let stores_clean =
                graph
                    .blocks
                    .iter()
                    .flat_map(|b| &b.operations)
                    .all(|o| match &o.kind {
                        OpKind::FieldWrite { base, .. } if base == result => o.result.is_none(),
                        _ => true,
                    });
            stores_clean.then(|| result.clone())
        })
        .collect();
    if dead.is_empty() {
        return 0;
    }

    // Drop the dead producers and every field store targeting one.
    let mut removed = 0usize;
    for block in &mut graph.blocks {
        block.operations.retain(|op| {
            let drop = op.result.as_ref().is_some_and(|r| dead.contains(r))
                || matches!(&op.kind, OpKind::FieldWrite { base, .. } if dead.contains(base));
            if drop {
                removed += 1;
            }
            !drop
        });
    }
    removed
}

/// Remove dead operations and input arguments using the backward dataflow from
/// `simplify.transform_dead_op_vars_in_blocks`.
///
/// Removable operations contribute operand dependencies from their result;
/// side-effecting and potentially raising operations contribute immediate
/// reads. Exit switches and terminal-block inputs are reads, while each link
/// argument depends on the target input it supplies. Start-block parameters are
/// pinned as a calling-convention contract, then liveness flows backward to a
/// fixpoint before unread removable producers and their link arguments are
/// dropped.
///
/// TODO: `start_blocks` is `{graph.startblock} ∪
/// {blocks with no incoming link}` rather than the strict single-graph
/// `{graphs[0].startblock}` of `simplify.py`.  The `generated::*`
/// pipeline emits closures / specialisations as secondary entries
/// (function parameters at a non-zero block id with no predecessors)
/// that are calling-convention contracts but are not registered with
/// the simplification pass as separate graphs.  Pinning them as
/// additional starts matches the *intent* of `simplify.py:431-433`'s
/// multi-graph branch (which pins one start per graph in the input
/// set) without requiring a `translator.annotator` to enumerate the
/// secondary entries.  Convergence: surface the secondary entries via
/// an explicit `start_blocks: HashSet<BlockId>` parameter or migrate
/// `generated::*` to register secondary entries with the simplifier
/// directly.
///
///   1. Walk every reachable block.  For each operation, evaluate
///      `canremove(op, block)` per `simplify.py`:
///      `op.opname in CanRemove and op is not block.raising_op`,
///      where `block.raising_op` is `block.operations[-1]` whenever
///      `block.canraise` (`flowspace/model.py`).  Pyre's
///      `Block::canraise()` mirrors the upstream
///      `block.exitswitch is c_last_exception` check; the raising
///      op is the last entry in `block.operations`.
///        - If `can_remove_op(kind)` AND the op is not the raising_op
///          AND `op.result.is_some()`: route operands via
///          `dependencies[op.result] += operands`
///          (`simplify.py:444-445 dependencies[op.result].
///          update(op.args)` for `canremove`-classified ops).
///          Operands become live only if `op.result` becomes live.
///        - Otherwise: operands join `read_vars` immediately
///          (`simplify.py read_vars.update(op.args)`
///          for non-`canremove` ops).  Raising-op operands therefore
///          stay live unconditionally — the raise side-effect is
///          observable.
///          The `exitswitch` (if a Variable) joins `read_vars`.  For
///          each block with no exits (`returnblock` / `exceptblock` /
///          otherwise terminal), the inputargs are also added — return
///          and except blocks implicitly use their inputs
///          (`simplify.transform_dead_op_vars`).
///   2. For every Link, record `dependencies[targetarg].add(linkarg)`
///      mapping (`simplify.py:457-458`).  This is the key
///      difference from a naïve forward pass — link args are NOT
///      direct reads, they are dependencies whose liveness flows
///      backward from a live target inputarg.  A self-referential
///      phi (back-edge `Link.args[i] == phi_i`) is therefore kept
///      alive only if `phi_i` itself is reached by some external
///      reader; without one, the dependency cycle dies.
///   3. The `startblock`'s inputargs are pinned to `read_vars` as a
///      calling-convention contract (`simplify.py:463-467`,
///      single-graph case `simplify.py:428-429 start_blocks =
///      {graphs[0].startblock}`).  `returnblock` / `exceptblock`
///      contracts are already covered by the empty-exits rule
///      above.
///   4. Backward-flow `read_vars` over `dependencies` to a fixpoint
///      (`simplify.py flow_read_var_backward`).
///   5. Drop every pure op whose result is not in `read_vars`
///      (`simplify.py:484-488` removable-op drop), gated by the
///      same `canremove(op, block)` predicate from Step 1 — the
///      raising op of a `canraise` block is preserved here even if
///      its result is dead.  Inputarg-shaped `OpKind::Input` ops
///      are protected by Step 1+3+dependency-routing pinning their
///      result `Variable` in `read_vars`; the dead ones are swept here
///      alongside Step 7's inputarg trim (the matching
///      `block.inputargs[i]` removal becomes a no-op when the
///      Input op is already gone).  Naked `OpKind::Input` ops
///      (legacy frontend fallback for cross-block reads not yet
///      routed through the lazy installer) are likewise removed
///      when their result Variable is dead — `simplify.py` itself has
///      no standalone Input shape, but the line-by-line port
///      sweeps them under the same canremove rule.
///
///      TODO: translator-gated arms not ported.
///      Upstream's Step 5 has two further `elif` arms after the
///      canremove drop:
///        - `simplify.py elif op.opname == 'simple_call'`
///          removes `simple_call(builtin, ...)` whose first arg is
///          a `Constant` whose value is in `CanRemoveBuiltins`
///          (`simplify.py:418-420 = {hasattr: True}`).  The
///          `FunctionGraph` IR pyre's prune_dead_phis operates on
///          carries call shapes as `OpKind::Call` /
///          `OpKind::IndirectCall` etc., not as `simple_call`-named
///          ops, and pyre's frontend never emits a Call to `hasattr`
///          — the arm is structurally inapplicable.
///        - `simplify.py elif op.opname == 'direct_call'`
///          removes the call when `has_no_side_effects(translator,
///          graph) and op is not block.raising_op`.  Gated on
///          `translator is not None`; pyre's caller passes
///          `translator=None`, so the arm is unreachable.  The
///          `op is not block.raising_op` clause is moot for the
///          same reason.
///          Convergence path: surface a translator-equivalent (callee
///          effect classifier) and route `OpKind::Call`-with-elidable-EI
///          through this Step's drop path.
///   6. For each reachable block, walk its exits and drop
///      `Link.args[i]` for every `i` where `target.inputargs[i]`
///      is not in `read_vars` — inputarg-side trim happens in a
///      second pass to preserve the `len(link.args) ==
///      len(link.target.inputargs)` invariant (`simplify.py:
///      512-516`, the explicit assertion at `:513`).
///   7. Walk each reachable non-terminal block once more and drop
///      `block.inputargs[i]` (and its matching `OpKind::Input` op
///      if still present in `block.operations`) for every dead
///      Variable (`simplify.py:520-524`).
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
)]
pub fn prune_dead_phis(graph: &mut FunctionGraph) {
    use crate::inline::can_remove_op;
    use std::collections::HashMap;
    let start = graph.startblock;
    let return_block = graph.returnblock;
    let except_block = graph.exceptblock;
    let block_index: HashMap<BlockId, usize> = graph
        .blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.id, i))
        .collect();

    // `start_blocks` mirrors `simplify.py:431-433`'s multi-graph branch:
    // `graph.startblock` plus every block with no incoming link
    // (calling-convention entries the `generated::*` pipeline emits
    // for closures / specialisations).  Each is a BFS root, so `blocks`
    // (the union of `iterblocks()` over each entry) covers every block
    // reachable from any entry.
    let start_blocks: HashSet<BlockId> = std::iter::once(start)
        .chain(secondary_entry_roots(graph))
        .collect();

    // BFS reachability mirrors `flowspace/model.py iterblocks()`,
    // unioned across `start_blocks`:
    //
    //     def iterblocks(self):
    //         block = self.startblock
    //         yield block
    //         seen = {block: True}
    //         stack = list(block.exits[::-1])
    //         while stack:
    //             block = stack.pop().target
    //             if block not in seen:
    //                 yield block
    //                 seen[block] = True
    //                 stack += block.exits[::-1]
    let reachable: HashSet<BlockId> = {
        let mut seen: HashSet<BlockId> = HashSet::new();
        let mut stack: Vec<BlockId> = Vec::new();
        for &sb in &start_blocks {
            if seen.insert(sb)
                && let Some(&i) = block_index.get(&sb)
            {
                stack.extend(graph.blocks[i].exits.iter().rev().map(|e| e.target));
            }
        }
        while let Some(target) = stack.pop() {
            if seen.insert(target)
                && let Some(&i) = block_index.get(&target)
            {
                stack.extend(graph.blocks[i].exits.iter().rev().map(|e| e.target));
            }
        }
        seen
    };

    // Step 1 + 2: gather initial `read_vars` (op operands of non-pure
    // ops + exitswitches + terminal-block inputargs) and
    // `dependencies` (target inputarg ← link arg, plus pure-op
    // operands routed via `op.result`).
    //
    // `simplify.py canremove`:
    //
    //     def canremove(op, block):
    //         return op.opname in CanRemove and op is not block.raising_op
    //
    // `block.raising_op` (`flowspace/model.py`) is
    // `block.operations[-1]` whenever `block.canraise` (i.e.
    // `block.exitswitch is c_last_exception`).  Pure-classified
    // ops parked there as overflow / zero-divide checks
    // (`int_add_ovf`, `int_floordiv_zer`, ...) MUST NOT be DCE'd
    // even if their result Variable is unread — the raise side-effect
    // is observable.
    // RPython `simplify.py:425-426`:
    //     read_vars = set()  # set of variables really used
    //     dependencies = defaultdict(set) # map {Var: list-of-Vars-it-depends-on}
    //
    // Both sidetables are Variable-keyed (PyPy uses Python object
    // identity for Variable's __hash__ / __eq__); pyre matches via
    // Rc<i64>-backed Variable._nr.
    let mut read_vars: HashSet<crate::flowspace::model::Variable> = HashSet::new();
    let mut dependencies: HashMap<
        crate::flowspace::model::Variable,
        Vec<crate::flowspace::model::Variable>,
    > = HashMap::new();
    for block in &graph.blocks {
        if !reachable.contains(&block.id) {
            continue;
        }
        let raising_op_idx = if block.canraise() && !block.operations.is_empty() {
            Some(block.operations.len() - 1)
        } else {
            None
        };
        for (i, op) in block.operations.iter().enumerate() {
            let operands: Vec<crate::flowspace::model::Variable> =
                crate::inline::op_variable_refs(&op.kind);
            // `simplify.py:441-445`:
            //   if not canremove(op, block):    read_vars.update(args)
            //   else:                           dependencies[result] += args
            let removable = can_remove_op(&op.kind) && Some(i) != raising_op_idx;
            if let Some(result_var) = op.result.clone()
                && removable
            {
                dependencies.entry(result_var).or_default().extend(operands);
            } else {
                for var in operands {
                    read_vars.insert(var);
                }
            }
        }
        match &block.exitswitch {
            Some(ExitSwitch::Value(var)) => {
                read_vars.insert(var.clone());
            }
            // A fused comparison reads its operands at the branch.
            Some(ExitSwitch::Fused { args, .. }) => {
                for arg in args {
                    read_vars.insert(arg.clone());
                }
            }
            Some(ExitSwitch::LastException) | None => {}
        }
        // `simplify.py:459-462`: terminal blocks (no exits)
        // implicitly use every inputarg.
        if block.exits.is_empty() {
            for iarg in &block.inputargs {
                read_vars.insert(iarg.clone());
            }
        }
        for link in &block.exits {
            // `simplify.py:512-513` len-equality is asserted at the
            // exits-walk; the dependency-routing pass exercises the
            // same invariant by zipping `link.args` with
            // `link.target.inputargs`.
            let &target_idx = block_index
                .get(&link.target)
                .expect("simplify.py:512 — link.target must be a graph block");
            let target_block = &graph.blocks[target_idx];
            assert_eq!(
                link.args.len(),
                target_block.inputargs.len(),
                "simplify.py:513 — len(link.args) == len(link.target.inputargs)",
            );
            for (arg, target_iarg) in link.args.iter().zip(target_block.inputargs.iter()) {
                if let Some(arg_var) = arg.as_variable() {
                    dependencies
                        .entry(target_iarg.clone())
                        .or_default()
                        .push(arg_var.clone());
                    // A void operand is omitted from `Call.args`
                    // (`NON_VOID_ARGS`) but the definition still crosses
                    // this edge. Keeping the target inputarg read retains
                    // that `ConstNone` and the link that carries it.
                    if FunctionGraph::concretetype_of(arg_var) == ConcreteType::Void {
                        read_vars.insert(target_iarg.clone());
                    }
                }
            }
        }
    }
    // Step 3: pin every `start_blocks` inputarg (multi-graph branch
    // `simplify.py:431-433`, with `graph.startblock` and orphan-entry
    // blocks treated as additional starts per the function-level doc
    // comment).
    for &sb in &start_blocks {
        if let Some(&i) = block_index.get(&sb) {
            for iarg in &graph.blocks[i].inputargs {
                read_vars.insert(iarg.clone());
            }
        }
    }

    // Step 4: backward flow.
    // `simplify.py flow_read_var_backward`.
    let mut pending: Vec<crate::flowspace::model::Variable> = read_vars.iter().cloned().collect();
    while let Some(var) = pending.pop() {
        if let Some(deps) = dependencies.get(&var).cloned() {
            for dep in deps {
                if read_vars.insert(dep.clone()) {
                    pending.push(dep);
                }
            }
        }
    }

    // Step 5: drop every pure op whose result is dead.
    // `simplify.py:484-488 if op.result not in read_vars: if
    // canremove(op, block): del block.operations[i]`.
    // Inputarg-shaped Input ops keep their result Variable in `read_vars`
    // via Step 1+3+dependency-routing, so the live ones survive
    // unconditionally; dead ones get removed alongside their
    // matching inputarg trim in Step 7.  Naked Input ops (any Input
    // op whose result is not pinned by a matching inputarg) do
    // NOT have inputarg pinning, so the same `result not in
    // read_vars` predicate retires them — `simplify.py` itself has
    // no standalone Input op shape, but the line-by-line port
    // sweeps them under the same canremove rule.
    //
    // The reverse `for i in range(len-1, -1, -1)` walk
    // (`simplify.py:484`) is preserved: we resolve `raising_op_idx`
    // before the loop and check `Some(i) != raising_op_idx` so a
    // pure op that happens to be the block's raising_op
    // (`canremove`'s `op is not block.raising_op` clause,
    // `simplify.py:436`) survives even when its result is unread.
    // Pre-compute the (block_id, op_index) pairs of dead pure ops in
    // an immutable pass; the mutable removal pass below operates on
    // the precomputed list to avoid an overlapping borrow.
    //
    // RPython `simplify.py:484-488`:
    //     if op.result not in read_vars:
    //         if canremove(op, block):
    //             del block.operations[i]
    let mut dead_op_positions: Vec<(BlockId, usize)> = Vec::new();
    for block in &graph.blocks {
        if !reachable.contains(&block.id) {
            continue;
        }
        let raising_op_idx = if block.canraise() && !block.operations.is_empty() {
            Some(block.operations.len() - 1)
        } else {
            None
        };
        for (i, op) in block.operations.iter().enumerate() {
            let dead = match op.result.as_ref() {
                Some(r) => !read_vars.contains(r),
                None => false,
            };
            if dead && can_remove_op(&op.kind) && Some(i) != raising_op_idx {
                dead_op_positions.push((block.id, i));
            }
        }
    }
    for block in &mut graph.blocks {
        let mut dead_indices: Vec<usize> = dead_op_positions
            .iter()
            .filter_map(|(bid, idx)| (*bid == block.id).then_some(*idx))
            .collect();
        dead_indices.sort_unstable();
        for i in dead_indices.into_iter().rev() {
            block.operations.remove(i);
        }
    }

    // Step 6: trim Link.args at indices whose target inputarg is dead.
    // `simplify.py:512-516`.  Walk in reverse so removals don't shift
    // surviving indices.
    //
    // The scope is the *target*'s reachability, not the source's.  Step 7
    // trims the inputargs of every reachable block, and a block the
    // reachability walk excludes — an orphan `eliminate_empty_blocks`
    // bypassed, or one of the merge blocks jtransform leaves whose
    // inputargs are phi targets rather than parameters — can still name a
    // reachable block as its link target.  Skipping that link because its
    // own block is unreachable breaks `len(link.args) ==
    // len(link.target.inputargs)` with no diagnostic:
    // `remove_duplicate_inputargs` reads each column by index across every
    // incoming link, so the untrimmed one contributes the value one slot
    // over, and the union-find merges two variables that are not the same
    // value — a rename applied to the whole graph.  Reachability is closed
    // under exits, so this only ever adds links, never drops one.
    //
    // Upstream cannot reach this: `graph.iterblocks()` *is* its block list,
    // so an unreachable block is not in `blocks` and has no link to skip.
    for block_idx in 0..graph.blocks.len() {
        let exits_len = graph.blocks[block_idx].exits.len();
        for exit_idx in 0..exits_len {
            let target = graph.blocks[block_idx].exits[exit_idx].target;
            if !reachable.contains(&target) {
                continue;
            }
            let target_iargs: Vec<crate::flowspace::model::Variable> = {
                let &i = block_index
                    .get(&target)
                    .expect("simplify.py:512 — link.target must be a graph block");
                graph.blocks[i].inputargs.clone()
            };
            assert_eq!(
                graph.blocks[block_idx].exits[exit_idx].args.len(),
                target_iargs.len(),
                "simplify.py:513 — len(link.args) == len(link.target.inputargs)",
            );
            for i in (0..target_iargs.len()).rev() {
                let target_iarg = &target_iargs[i];
                if !read_vars.contains(target_iarg) {
                    graph.blocks[block_idx].exits[exit_idx].args.remove(i);
                }
            }
        }
    }

    // Step 7: drop dead inputargs + their matching `OpKind::Input` ops.
    // `simplify.py`.  Walk every reachable block; startblock is
    // already pinned in Step 3, return / except blocks in Step 1.
    for block_idx in 0..graph.blocks.len() {
        let block_id = graph.blocks[block_idx].id;
        if !reachable.contains(&block_id) || block_id == return_block || block_id == except_block {
            continue;
        }
        let inputargs: Vec<crate::flowspace::model::Variable> =
            graph.blocks[block_idx].inputargs.clone();
        for i in (0..inputargs.len()).rev() {
            let iarg = &inputargs[i];
            if read_vars.contains(iarg) {
                continue;
            }
            graph.blocks[block_idx].inputargs.remove(i);
            if let Some(op_idx) = graph.blocks[block_idx].operations.iter().position(|op| {
                matches!(op.kind, OpKind::Input { .. }) && op.result.as_ref() == Some(iarg)
            }) {
                graph.blocks[block_idx].operations.remove(op_idx);
            }
        }
    }

    remove_duplicate_inputargs(graph);
}

/// `true` when [`remove_duplicate_inputargs`] can change link arity or merge
/// a phi column. Equal raw columns are what the first fixpoint round merges
/// while every representative is still itself; a length mismatch is the
/// truncation (or the missing-arg panic) of the rewrite.
fn phi_columns_need_merge(
    graph: &FunctionGraph,
    entries: &std::collections::HashMap<BlockId, Vec<(usize, usize)>>,
) -> bool {
    use std::hash::{Hash, Hasher};

    let mut buckets: rustc_hash::FxHashMap<u64, Vec<usize>> = rustc_hash::FxHashMap::default();
    for (&block_id, links) in entries {
        let arity = graph.block(block_id).inputargs.len();
        for &(pred_idx, link_idx) in links {
            if graph.blocks[pred_idx].exits[link_idx].args.len() != arity {
                return true;
            }
        }
        if arity < 2 {
            continue;
        }
        buckets.clear();
        for col in 0..arity {
            let mut hasher = rustc_hash::FxHasher::default();
            for &(pred_idx, link_idx) in links {
                graph.blocks[pred_idx].exits[link_idx].args[col].hash(&mut hasher);
            }
            let fingerprint = hasher.finish();
            let mut duplicated = false;
            if let Some(previous) = buckets.get(&fingerprint) {
                for &earlier in previous {
                    let same = links.iter().all(|&(pred_idx, link_idx)| {
                        let args = &graph.blocks[pred_idx].exits[link_idx].args;
                        args[earlier] == args[col]
                    });
                    if same {
                        duplicated = true;
                        break;
                    }
                }
            }
            if duplicated {
                return true;
            }
            buckets.entry(fingerprint).or_default().push(col);
        }
    }
    false
}

/// Crate-local [`FunctionGraph`] port of RPython
/// `translator/simplify.py remove_identical_vars_SSA`.
///
/// The earlier subset only removed literally duplicated `inputargs`, which
/// missed upstream's phi-tuple equivalence pass and left codewriter graphs
/// with self-conflicting register-allocation inputs.  Keep the same shape as
/// upstream: collect each block's incoming phi columns, union equivalent
/// variables, rebuild input/link arg lists, then rename the remaining block
/// body references.
pub fn remove_duplicate_inputargs(graph: &mut FunctionGraph) {
    use crate::tool::algo::unionfind::{UnionFind, UnionFindInfo};
    use std::collections::HashMap;

    let start = graph.startblock;
    let return_block = graph.returnblock;
    let except_block = graph.exceptblock;

    #[derive(Clone)]
    struct Representative {
        rep: LinkArg,
    }

    impl UnionFindInfo for Representative {
        fn absorb(&mut self, _other: Self) {}
    }

    let mut entries: HashMap<BlockId, Vec<(usize, usize)>> = HashMap::new();
    for (pred_idx, block) in graph.blocks.iter().enumerate() {
        for (link_idx, link) in block.exits.iter().enumerate() {
            if link.target == start || link.target == return_block || link.target == except_block {
                continue;
            }
            entries
                .entry(link.target)
                .or_default()
                .push((pred_idx, link_idx));
        }
    }

    // No duplicate raw column and every incoming link already matches the
    // block's input arity: the fixpoint's first round would insert each
    // link arg as its own representative and remove nothing, and the
    // rewrite would copy those args back. Leave the graph untouched.
    if !phi_columns_need_merge(graph, &entries) {
        return;
    }

    let mut uf: UnionFind<LinkArg, Representative> =
        UnionFind::new(|arg: &LinkArg| Representative { rep: arg.clone() });

    struct PhiCols {
        original_len: usize,
        cols: Vec<(usize, crate::flowspace::model::Variable)>,
    }

    let mut inputs: HashMap<BlockId, PhiCols> = HashMap::new();
    for (&block_id, links) in &entries {
        let inputargs = graph.block(block_id).inputargs.clone();
        for (arg_i, _) in inputargs.iter().enumerate() {
            for &(pred_idx, link_idx) in links {
                if graph.blocks[pred_idx].exits[link_idx]
                    .args
                    .get(arg_i)
                    .is_none()
                {
                    panic!(
                        "remove_identical_vars_SSA: link.args[{arg_i}] missing \
                         (graph {}, target {:?})",
                        graph.name, block_id,
                    );
                }
            }
        }
        let original_len = inputargs.len();
        inputs.insert(
            block_id,
            PhiCols {
                original_len,
                cols: inputargs.into_iter().enumerate().collect(),
            },
        );
    }

    fn resolve_phi_arg(uf: &mut UnionFind<LinkArg, Representative>, arg: &LinkArg) -> LinkArg {
        // Constants are never unioned. A variable's representative changes
        // only after `union`, which inserts it; until then it is its own root.
        let LinkArg::Value(var) = arg else {
            return arg.clone();
        };
        let key = LinkArg::Value(var.clone());
        if uf.contains(&key) {
            uf.find_rep(key)
        } else {
            key
        }
    }

    fn simplify_phis(
        uf: &mut UnionFind<LinkArg, Representative>,
        graph: &FunctionGraph,
        links: &[(usize, usize)],
        cols: &mut Vec<(usize, crate::flowspace::model::Variable)>,
    ) -> bool {
        use std::hash::{Hash, Hasher};

        let mut to_remove: Vec<usize> = Vec::new();
        // Resolved column keys share one buffer. `kept` records the first
        // column of each distinct key; a later equal key unions into it.
        let mut arena: Vec<LinkArg> = Vec::new();
        struct Kept {
            start: usize,
            input: crate::flowspace::model::Variable,
        }
        let mut kept: Vec<Kept> = Vec::new();
        let mut buckets: rustc_hash::FxHashMap<u64, Vec<usize>> = rustc_hash::FxHashMap::default();
        let width = links.len();
        for (index, &(orig_col, ref input)) in cols.iter().enumerate() {
            let start = arena.len();
            let mut hasher = rustc_hash::FxHasher::default();
            (width as u64).hash(&mut hasher);
            for &(pred_idx, link_idx) in links {
                let resolved =
                    resolve_phi_arg(uf, &graph.blocks[pred_idx].exits[link_idx].args[orig_col]);
                resolved.hash(&mut hasher);
                arena.push(resolved);
            }
            let fingerprint = hasher.finish();
            let mut existing = None;
            if let Some(candidates) = buckets.get(&fingerprint) {
                for &kept_index in candidates {
                    let prior = &arena[kept[kept_index].start..kept[kept_index].start + width];
                    if prior == &arena[start..start + width] {
                        existing = Some(kept[kept_index].input.clone());
                        break;
                    }
                }
            }
            // PRE-EXISTING-ADAPTATION: the all-equal phi collapse of
            // `simplify.py` (`if all_equal(new_args):
            // uf.union(new_args[0], input)`) is omitted here.  Upstream that
            // collapse deliberately produces cross-block variable references
            // — it removes a merge block's inputarg and leaves the body
            // reading a value defined in a predecessor — and relies on the
            // next `all_passes` entry, `SSA_to_SSI` (backendopt/ssa.py),
            // to repair them by re-threading every used-but-undefined variable
            // as a fresh inputarg through all incoming links.  pyre runs this
            // pass on the `crate::model` front-end graph, which has no
            // `SSA_to_SSI` (the faithful `ssa_to_ssi` port at
            // translator/backendopt/ssa.rs operates on `crate::flowspace::model`,
            // a distinct IR).  Without the repair the collapse strands the body
            // reference and the flowspace adapter rejects it as an undefined
            // operand.  `prune_dead_phis` (transform_dead_op_vars) has already
            // dropped every dead inputarg, so each surviving column is used and
            // `SSA_to_SSI` would re-thread it regardless — the collapse is a
            // no-op net of the repair, and skipping it yields the same graph
            // with the column left threaded.  The codewriter's phi-tuple
            // equivalence (`simplify.py:565-568`) is the duplicate-column merge
            // handled below, which is SSI-safe because both columns are
            // inputargs of the same block.  Convergence path: port
            // `SSA_to_SSI` to `crate::model` (or unify the two `FunctionGraph`
            // IRs and run the standard `all_passes` including `ssa_to_ssi`).
            if let Some(existing) = existing {
                uf.union(LinkArg::Value(existing), LinkArg::Value(input.clone()));
                to_remove.push(index);
                arena.truncate(start);
            } else {
                buckets.entry(fingerprint).or_default().push(kept.len());
                kept.push(Kept {
                    start,
                    input: input.clone(),
                });
            }
        }
        for index in to_remove.iter().rev() {
            cols.remove(*index);
        }
        !to_remove.is_empty()
    }

    let block_ids: Vec<BlockId> = inputs.keys().copied().collect();
    // A block's resolved columns change only after some `union`. Re-walking
    // a block that already observed the current union-find removes nothing.
    // The epoch stored is the one the block observed, before its own unions
    // advance it: those unions can make the block's earlier columns equal,
    // and `remove_identical_vars_SSA` walks every block again while
    // `progress` is set.
    let mut epoch: u64 = 0;
    let mut seen_epoch: HashMap<BlockId, u64> = HashMap::new();
    loop {
        let mut progress = false;
        for block_id in &block_ids {
            let observed = epoch;
            if seen_epoch.get(block_id).copied() == Some(observed) {
                continue;
            }
            let changed = {
                let links = entries.get(block_id).expect("entry list for input block");
                let state = inputs.get_mut(block_id).expect("inputs block");
                simplify_phis(&mut uf, graph, links, &mut state.cols)
            };
            if changed {
                epoch += 1;
                progress = true;
            }
            seen_epoch.insert(*block_id, observed);
        }
        if !progress {
            break;
        }
    }

    let keys: Vec<LinkArg> = uf.keys().cloned().collect();
    #[expect(
        clippy::mutable_key_type,
        reason = "Variable equality and hashing use immutable identity IDs while annotations are excluded; this exactly models RPython object-identity dictionary keys without a divergent side table"
    )]
    let mut renaming: HashMap<
        crate::flowspace::model::Variable,
        crate::flowspace::model::Variable,
    > = HashMap::new();
    for key in keys {
        let LinkArg::Value(src) = key.clone() else {
            continue;
        };
        if let Some(info) = uf.get(&key)
            && let LinkArg::Value(dst) = &info.rep
            && src != *dst
        {
            renaming.insert(src, dst.clone());
        }
    }

    for (&block_id, state) in &inputs {
        let links = entries
            .get(&block_id)
            .expect("entry list for input block")
            .clone();
        let removed = state.cols.len() != state.original_len;
        let truncate = links.iter().any(|&(pred_idx, link_idx)| {
            graph.blocks[pred_idx].exits[link_idx].args.len() != state.original_len
        });
        if !removed && !truncate {
            continue;
        }
        let surviving: Vec<crate::flowspace::model::Variable> =
            state.cols.iter().map(|(_, input)| input.clone()).collect();
        {
            // simplify.py drops the phi slot from `block.inputargs` and the
            // predecessor `link.args`.  Pyre phi blocks also carry a matching
            // `OpKind::Input` op per slot, so remove the Input ops whose result
            // is a dropped slot — otherwise the later rename leaves a spurious
            // in-block definition for a value no predecessor supplies.
            #[expect(
                clippy::mutable_key_type,
                reason = "Variable equality and hashing use immutable identity IDs while annotations are excluded; this exactly models RPython object-identity dictionary keys without a divergent side table"
            )]
            let surviving_set: std::collections::HashSet<
                &crate::flowspace::model::Variable,
            > = surviving.iter().collect();
            let block = graph.block_mut(block_id);
            let dropped: Vec<crate::flowspace::model::Variable> = block
                .inputargs
                .iter()
                .filter(|v| !surviving_set.contains(*v))
                .cloned()
                .collect();
            block.inputargs = surviving;
            for slot in &dropped {
                if let Some(op_idx) = block.operations.iter().position(|op| {
                    matches!(op.kind, OpKind::Input { .. }) && op.result.as_ref() == Some(slot)
                }) {
                    block.operations.remove(op_idx);
                }
            }
        }
        for (pred_idx, link_idx) in links {
            let new_args: Vec<LinkArg> = state
                .cols
                .iter()
                .map(|(col, _)| graph.blocks[pred_idx].exits[link_idx].args[*col].clone())
                .collect();
            graph.blocks[pred_idx].exits[link_idx].args = new_args;
        }
    }

    if renaming.is_empty() {
        return;
    }

    let remap_var = |var: &crate::flowspace::model::Variable| {
        renaming.get(var).cloned().unwrap_or_else(|| var.clone())
    };
    for block in &mut graph.blocks {
        for inputarg in &mut block.inputargs {
            *inputarg = remap_var(inputarg);
        }
        for op in &mut block.operations {
            op.result = op.result.as_ref().map(&remap_var);
            op.kind = crate::inline::remap_op_kind(&op.kind, &remap_var);
        }
        let (exitswitch, exits) = remap_control_flow_metadata_var(
            &block.exitswitch,
            &block.exits,
            remap_var,
            |block_id| block_id,
        );
        block.exitswitch = exitswitch;
        block.exits = exits;
    }

    // A second sweep removes any same-Variable inputargs created by the final
    // rename.  This is the same fixed-point effect as upstream's loop.
    for block_idx in 0..graph.blocks.len() {
        let block_id = graph.blocks[block_idx].id;
        if block_id == start || block_id == return_block || block_id == except_block {
            continue;
        }
        #[expect(
            clippy::mutable_key_type,
            reason = "Variable equality and hashing use immutable identity IDs while annotations are excluded; this exactly models RPython object-identity dictionary keys without a divergent side table"
        )]
        let mut first_seen: HashMap<crate::flowspace::model::Variable, usize> = HashMap::new();
        let mut duplicate_slots: Vec<(usize, usize)> = Vec::new();
        for (i, inputarg) in graph.blocks[block_idx].inputargs.iter().enumerate() {
            if let Some(first_i) = first_seen.get(inputarg).copied() {
                duplicate_slots.push((i, first_i));
            } else {
                first_seen.insert(inputarg.clone(), i);
            }
        }
        if duplicate_slots.is_empty() {
            continue;
        }
        let mut removable_slots: Vec<(usize, usize)> = Vec::new();
        for (dup_i, first_i) in duplicate_slots {
            let mut removable = true;
            for pred_idx in 0..graph.blocks.len() {
                for link in &graph.blocks[pred_idx].exits {
                    if link.target != block_id {
                        continue;
                    }
                    if link.args.get(dup_i) != link.args.get(first_i) {
                        removable = false;
                        break;
                    }
                }
                if !removable {
                    break;
                }
            }
            if removable {
                removable_slots.push((dup_i, first_i));
            }
        }
        for (dup_i, _) in removable_slots.into_iter().rev() {
            // Drop the `OpKind::Input` op paired with the DUPLICATE slot by its
            // slot position — `del block.inputargs[i]` by index
            // (simplify.py).  After the rename several Input ops can
            // share the survivor's result, so a result match would risk
            // dropping the survivor (first-seen) slot's own `name`/`ty`/
            // `class_root` metadata instead of the duplicate's.  The k-th
            // `OpKind::Input` op mirrors the k-th inputarg (paired in order at
            // construction; earlier passes preserve that order).
            let mut seen_inputs = 0usize;
            let mut dup_op_idx = None;
            for (op_idx, op) in graph.blocks[block_idx].operations.iter().enumerate() {
                if matches!(op.kind, OpKind::Input { .. }) {
                    if seen_inputs == dup_i {
                        dup_op_idx = Some(op_idx);
                        break;
                    }
                    seen_inputs += 1;
                }
            }
            graph.blocks[block_idx].inputargs.remove(dup_i);
            if let Some(op_idx) = dup_op_idx {
                graph.blocks[block_idx].operations.remove(op_idx);
            }
            for pred_idx in 0..graph.blocks.len() {
                for link in &mut graph.blocks[pred_idx].exits {
                    if link.target == block_id {
                        link.args.remove(dup_i);
                    }
                }
            }
        }
    }
}

impl Block {
    pub fn canraise(&self) -> bool {
        matches!(self.exitswitch, Some(ExitSwitch::LastException))
    }

    /// Variable-identity iter over [`Self::inputargs`] — direct
    /// over the upstream-orthodox `Vec<Variable>` storage, matching
    /// `Block.inputargs: List[Variable]` (`flowspace/model.py`).
    pub fn input_variables(&self) -> impl Iterator<Item = &crate::flowspace::model::Variable> + '_ {
        self.inputargs.iter()
    }

    /// RPython `flowspace/model.py closeblock` / `:250 recloseblock`
    /// mark a block's exits tuple as populated.  Pyre mirrors the
    /// "has this block been closed?" predicate by checking that either
    /// `exits` has at least one `Link` or `exitswitch` is set.
    /// During graph construction, an unclosed block has
    /// `exits=[]`, `exitswitch=None` — the upstream equivalent of
    /// `type(block.exits) is list` pre-`closeblock`.
    pub fn is_closed(&self) -> bool {
        !self.exits.is_empty() || self.exitswitch.is_some()
    }

    /// Complement of `is_closed` — true if the front-end has not yet
    /// stamped a terminator / exits onto the block.  Used to gate
    /// fall-through code that adds the block's own exit.
    pub fn is_open(&self) -> bool {
        !self.is_closed()
    }
}

/// RPython `flowspace/model.py renamevariables` applies a
/// value renaming to `inputargs` / `operations` / `exitswitch` /
/// `link.args` — Variable-direct.  Takes a `Fn(&Variable) -> Variable`
/// renamer so callers whose alias map is keyed on Variable identity
/// (e.g. jtransform's `Transformer.aliases: HashMap<Variable,
/// Variable>` or inline's per-block value_map) write the rename
/// directly without round-tripping through a slot index.  Mirrors
/// upstream's Variable-identity rewrite path.
pub fn remap_control_flow_metadata_var<FValue, FBlock>(
    exitswitch: &Option<ExitSwitch>,
    exits: &[Link],
    remap_value: FValue,
    remap_block: FBlock,
) -> (Option<ExitSwitch>, Vec<Link>)
where
    FValue: Fn(&crate::flowspace::model::Variable) -> crate::flowspace::model::Variable,
    FBlock: Fn(BlockId) -> BlockId,
{
    let remap_link_arg = |arg: &LinkArg| -> LinkArg {
        match arg {
            LinkArg::Value(var) => LinkArg::Value(remap_value(var)),
            LinkArg::Const(value) => LinkArg::Const(value.clone()),
        }
    };
    let exitswitch = exitswitch.as_ref().map(|switch| match switch {
        ExitSwitch::Value(var) => ExitSwitch::Value(remap_value(var)),
        ExitSwitch::LastException => ExitSwitch::LastException,
        ExitSwitch::Fused { opname, args } => ExitSwitch::Fused {
            opname: opname.clone(),
            args: args.iter().map(&remap_value).collect(),
        },
    });
    let exits = exits
        .iter()
        .map(|link| Link {
            args: link.args.iter().map(&remap_link_arg).collect(),
            target: remap_block(link.target),
            exitcase: link.exitcase.clone(),
            prevblock: link.prevblock.map(&remap_block),
            llexitcase: link.llexitcase.clone(),
            last_exception: link.last_exception.as_ref().map(&remap_link_arg),
            last_exc_value: link.last_exc_value.as_ref().map(&remap_link_arg),
        })
        .collect();
    (exitswitch, exits)
}

/// `Variable.concretetype` analogue collapsed to the four kinds the
/// codewriter needs.
///
/// RPython stores `.concretetype` inline on each `Variable` after
/// `RPythonTyper.specialize()` rewrites the graph; pyre reads the same
/// attribute off the `Variable` directly, so
/// [`FunctionGraph::concretetype_of`] reads
/// `getkind(var.concretetype.borrow())` line-for-line with upstream.
/// `Unknown` is the pre-rtyper sentinel — `Variable` analogue before
/// the rtyper ran.
///
/// The variants line up 1:1 with [`getkind`]'s output strings
/// (`rpython/jit/metainterp/history.py`):
///
/// - [`Self::Signed`] = `"int"`
/// - [`Self::GcRef`]  = `"ref"`
/// - [`Self::Float`]  = `"float"`
/// - [`Self::Void`]   = `"void"`
///
/// [`Self::Unknown`] is a pyre-only sentinel for the
/// pre-`setconcretetype` window where reading `var.concretetype`
/// upstream would `AttributeError`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConcreteType {
    /// Signed integer (RPython `Signed` / i64).
    Signed,
    /// GC reference (RPython `Ptr(GcStruct)`).
    GcRef,
    /// Float (RPython `Float` / f64).
    Float,
    /// Void (RPython `Void`).
    Void,
    /// Unknown / unresolved.
    Unknown,
}

/// `rpython/jit/metainterp/history.py getkind(TYPE, ...)`.
///
/// Direct line-by-line port — the canonical
/// `LowLevelType → 'int' / 'ref' / 'float' / 'void'` projection that
/// every JIT codewriter / metainterp / regalloc consumer reads kind
/// from.  Pyre returns the [`ConcreteType`] enum form (same kind
/// space).  The three `supports_*` parameters default to `true`
/// just like the JIT uses — `metainterp_sd` flips them off only
/// for backends without longlong / singlefloat / float support.
///
/// ```py
/// def getkind(TYPE, supports_floats=True,
///                   supports_longlong=True,
///                   supports_singlefloats=True):
///     if TYPE is lltype.Void:
///         return "void"
///     elif isinstance(TYPE, lltype.Primitive):
///         if TYPE is lltype.Float and supports_floats:
///             return 'float'
///         if TYPE is lltype.SingleFloat and supports_singlefloats:
///             return 'int'     # singlefloats are stored in an int
///         if TYPE in (lltype.Float, lltype.SingleFloat):
///             raise NotImplementedError("type %s not supported" % TYPE)
///         if (TYPE != llmemory.Address and
///             rffi.sizeof(TYPE) > rffi.sizeof(lltype.Signed)):
///             if supports_longlong and TYPE is not lltype.LongFloat:
///                 assert rffi.sizeof(TYPE) == 8
///                 return 'float'
///             raise NotImplementedError("type %s is too large" % TYPE)
///         return "int"
///     elif isinstance(TYPE, lltype.Ptr):
///         if TYPE.TO._gckind == 'raw':
///             return "int"
///         else:
///             return "ref"
///     else:
///         raise NotImplementedError("type %s not supported" % TYPE)
/// ```
///
/// [`getkind`] is this function with every `supports_*` flag true, and
/// it panics with the strings the codewriter already matches. Policy
/// calls [`try_getkind`] so a refused graph stays residual.
pub fn try_getkind(
    ty: &crate::translator::rtyper::lltypesystem::lltype::LowLevelType,
    supports_floats: bool,
    supports_longlong: bool,
    supports_singlefloats: bool,
) -> Result<ConcreteType, GetkindError> {
    use crate::translator::rtyper::lltypesystem::lltype::{GcKind, LowLevelType};
    match ty {
        // `if TYPE is lltype.Void: return "void"`
        LowLevelType::Void => Ok(ConcreteType::Void),
        // `if TYPE is lltype.Float and supports_floats: return 'float'`
        // `if TYPE in (lltype.Float, lltype.SingleFloat): raise NotImplementedError`
        LowLevelType::Float => {
            if supports_floats {
                Ok(ConcreteType::Float)
            } else {
                Err(GetkindError::NotSupported)
            }
        }
        // `if TYPE is lltype.SingleFloat and supports_singlefloats:
        //      return 'int'  # singlefloats are stored in an int`
        LowLevelType::SingleFloat => {
            if supports_singlefloats {
                Ok(ConcreteType::Signed)
            } else {
                Err(GetkindError::NotSupported)
            }
        }
        // `if rffi.sizeof(TYPE) > rffi.sizeof(lltype.Signed):
        //      if supports_longlong and TYPE is not lltype.LongFloat:
        //          assert rffi.sizeof(TYPE) == 8
        //          return 'float'
        //      raise NotImplementedError("type %s is too large" % TYPE)`
        // On 64-bit `Signed` is 8 bytes, so SignedLongLong does not
        // exceed it and falls through to `"int"` even when
        // `supports_longlong` is false. On 32-bit the 8-byte longlong
        // takes the `'float'` slot only when the flag is set.
        LowLevelType::SignedLongLong | LowLevelType::UnsignedLongLong => {
            if std::mem::size_of::<i64>() > std::mem::size_of::<isize>() {
                if supports_longlong {
                    Ok(ConcreteType::Float)
                } else {
                    Err(GetkindError::TooLarge)
                }
            } else {
                Ok(ConcreteType::Signed)
            }
        }
        // Other primitives whose width is at most `Signed`.
        LowLevelType::Signed
        | LowLevelType::Unsigned
        | LowLevelType::Bool
        | LowLevelType::Char
        | LowLevelType::UniChar
        | LowLevelType::Address => Ok(ConcreteType::Signed),
        // 16-byte primitives, and `LongFloat`, which the longlong arm
        // excludes. `supports_longlong` only rescues a width of 8.
        LowLevelType::SignedLongLongLong
        | LowLevelType::UnsignedLongLongLong
        | LowLevelType::LongFloat => Err(GetkindError::TooLarge),
        // `elif isinstance(TYPE, lltype.Ptr):
        //      if TYPE.TO._gckind == 'raw': return "int"
        //      else: return "ref"`
        LowLevelType::Ptr(ptr) => match ptr_gckind(&ptr.TO) {
            GcKind::Raw => Ok(ConcreteType::Signed),
            GcKind::Gc | GcKind::Prebuilt => Ok(ConcreteType::GcRef),
        },
        // Not a valid `concretetype`. `history.py` raises
        // `NotImplementedError("type %s not supported" % TYPE)`.
        LowLevelType::InteriorPtr(_)
        | LowLevelType::Func(_)
        | LowLevelType::Struct(_)
        | LowLevelType::Array(_)
        | LowLevelType::FixedSizeArray(_)
        | LowLevelType::Opaque(_)
        | LowLevelType::ForwardReference(_) => Err(GetkindError::NotSupported),
    }
}

/// Why [`try_getkind`] refused a low-level type. Policy turns either
/// variant into a residual graph; [`getkind`] panics instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GetkindError {
    /// `NotImplementedError("type %s not supported" % TYPE)`.
    NotSupported,
    /// `NotImplementedError("type %s is too large" % TYPE)`.
    TooLarge,
}

/// [`try_getkind`] with `supports_floats`, `supports_longlong` and
/// `supports_singlefloats` all true. A refusal panics with the string
/// the codewriter's concretetype adapter already matches.
pub fn getkind(ty: &crate::translator::rtyper::lltypesystem::lltype::LowLevelType) -> ConcreteType {
    use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
    match try_getkind(ty, true, true, true) {
        Ok(kind) => kind,
        Err(_) => match ty {
            LowLevelType::SignedLongLongLong
            | LowLevelType::UnsignedLongLongLong
            | LowLevelType::LongFloat => {
                panic!("getkind: type {ty:?} not supported (history.py:62 NotImplementedError)")
            }
            LowLevelType::InteriorPtr(_)
            | LowLevelType::Func(_)
            | LowLevelType::Struct(_)
            | LowLevelType::Array(_)
            | LowLevelType::FixedSizeArray(_)
            | LowLevelType::Opaque(_)
            | LowLevelType::ForwardReference(_) => {
                panic!("getkind: type {ty:?} not supported as concretetype (history.py:70)")
            }
            _ => {
                panic!("getkind: type {ty:?} not supported (history.py:62 NotImplementedError)")
            }
        },
    }
}

/// Project a [`ConcreteType`] back to a canonical
/// `LowLevelType` representative — the inverse direction of
/// [`getkind`] for the kind-only adapter callers (regalloc canonical
/// exceptblock, type_state synth merge, codewriter test
/// scaffolding).  Each branch chooses a representative that
/// round-trips through `getkind`:
///
/// - `Signed` → `LowLevelType::Signed`.
/// - `Float`  → `LowLevelType::Float`.
/// - `Void`   → `LowLevelType::Void`.
/// - `GcRef`  → [`crate::translator::rtyper::rclass::OBJECTPTR`]
///   (`Ptr(OBJECT)`, the canonical GC pointer the rtyper itself
///   stamps onto exception-block `evalue` Variables).
/// - `Unknown` → `None` (Variable.concretetype stays unset;
///   [`FunctionGraph::concretetype`] reads `Unknown` for the slot).
///
/// Lossy for richer Ptr-of-specific-Struct values, so
/// [`FunctionGraph::set_concretetype`] only writes through when the
/// existing Variable type's `getkind` does NOT already match the
/// requested kind (i.e. the rtyper's authoritative type wins).
pub(crate) fn concrete_to_canonical_lltype(
    ty: ConcreteType,
) -> Option<crate::translator::rtyper::lltypesystem::lltype::LowLevelType> {
    use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
    match ty {
        ConcreteType::Signed => Some(LowLevelType::Signed),
        ConcreteType::Float => Some(LowLevelType::Float),
        ConcreteType::Void => Some(LowLevelType::Void),
        ConcreteType::GcRef => Some(crate::translator::rtyper::rclass::OBJECTPTR.clone()),
        ConcreteType::Unknown => None,
    }
}

/// Helper for [`getkind`] — extract `_gckind` from a `Ptr.TO`
/// payload.  RPython exposes `_gckind` as an attribute on every
/// pointee (`Struct._gckind`, `Array._gckind`, etc.) so the
/// `TYPE.TO._gckind == 'raw'` test in `getkind` works directly;
/// pyre's `PtrTarget` enum surface needs a small dispatch.
fn ptr_gckind(
    target: &crate::translator::rtyper::lltypesystem::lltype::PtrTarget,
) -> crate::translator::rtyper::lltypesystem::lltype::GcKind {
    use crate::translator::rtyper::lltypesystem::lltype::PtrTarget;
    match target {
        PtrTarget::Struct(s) => s._gckind,
        PtrTarget::Array(a) => a._gckind,
        PtrTarget::FixedSizeArray(_) => {
            crate::translator::rtyper::lltypesystem::lltype::GcKind::Raw
        }
        PtrTarget::Opaque(o) => o._gckind,
        PtrTarget::ForwardReference(f) => f._gckind,
        PtrTarget::Func(_) => crate::translator::rtyper::lltypesystem::lltype::GcKind::Raw,
    }
}

/// Per-function effect attributes — the metadata `call.py` /
/// `collectanalyze.py` read off `graph.func` (or, for a graph-less
/// external call, off `funcobj`). Carried directly on the
/// [`FunctionGraph`] (`graph.func`) for functions that have a graph, and
/// in `CallControl::external_funcobjs` keyed by `CallPath` for graph-less
/// externals (the `jit.*` intrinsics, `Vec::len`, etc.).
///
/// The raw source-attribute tokens (`_elidable_function_`,
/// `_jit_loop_invariant_`, the open `_jit_*_` policy hints like
/// `look_inside` / `unroll_safe` / `aroundstate`) also live on
/// [`FunctionGraph::hints`]. [`FunctionGraph::push_hint`] projects the
/// policy tokens onto the fields [`crate::codewriter::policy`] reads
/// (`elidable`, `jit_look_inside`, `unroll_safe`). This struct is the
/// typed carrier of `getattr(func, <attr>)`, for both the policy and the
/// CallControl effect analyzers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FuncEffects {
    /// `func.oopspec` (rlib/jit.py `@oopspec(spec)`). Its presence is
    /// the builtin signal (call.py:135 `if hasattr(targetgraph.func,
    /// 'oopspec'): return 'builtin'`).
    pub oopspec: Option<String>,
    /// `argnames = ll_func.__code__.co_varnames[:nb_args]`
    /// (support.py:705) — declaration-order parameter names used by
    /// `parse_oopspec` to resolve identifier slots in the spec's `(...)`
    /// pattern. Empty when no `(...)`-bearing spec was registered.
    pub oopspec_argnames: Vec<String>,
    /// `func._gctransformer_hint_cannot_collect_` (collectanalyze.py:15)
    /// — `analyze_can_collect` returns False immediately.
    pub cannot_collect: bool,
    /// `funcobj.random_effects_on_gcobjs` (collectanalyze.py). Only
    /// meaningful for an external (graph-less) funcobj; a graph-bearing
    /// function derives can-collect by walking the graph instead.
    pub random_effects_on_gcobjs: bool,
    /// `funcobj.canraise` (`canraise.py analyze_external_call`:
    /// `getattr(fnobj, 'canraise', True)`). Defaults to true. `rffi.py
    /// llexternal` stamps false on the C function it creates.
    pub canraise: bool,
    /// `LL_OPERATIONS[op.opname].canmallocgc` (collectanalyze.py) for a
    /// callee that has no graph.
    ///
    /// Upstream reaches the collector through an *operation* — `malloc` /
    /// `malloc_varsize` with `flavor='gc'`, or any LLOp declared
    /// `canmallocgc=True` — and answers the two analyzers differently for it:
    /// `CollectAnalyzer.analyze_simple_operation` returns True while
    /// `RandomEffectsAnalyzer.analyze_simple_operation` returns False
    /// (effectinfo.py). A crate that is not lowered to LLBC spells
    /// that same operation as a call to a graph-less callee, so it needs the
    /// same split: this flag answers can-collect only, leaving random-effects
    /// alone. `random_effects_on_gcobjs` cannot stand in — it answers both,
    /// and an elidable caller of a random-effects callee is a contradiction
    /// upstream rejects outright (`assert not (elidable_function and
    /// random_effects_on_gcobjs)`, rffi.py:160).
    pub canmallocgc: bool,
    /// pyre `#[majit_macros::elidable_cannot_raise]` or
    /// `#[majit_macros::dont_look_inside_cannot_raise]` assertion that the
    /// callee never raises.
    pub cannot_raise_assertion: bool,
    /// pyre `#[majit_macros::elidable_or_memerror]` user assertion that
    /// the callee raises only MemoryError.
    pub memerror_only_assertion: bool,
    /// `func._elidable_function_` (call.py:239) — the typed carrier the
    /// CallControl analyzers read (`is_elidable`, indirect-family check).
    /// The same `"elidable"` token also stays in [`FunctionGraph::hints`].
    /// [`FunctionGraph::push_hint`] projects it onto this field, which
    /// `look_inside_graph` reads.
    pub elidable: bool,
    /// `func._jit_loop_invariant_` (call.py:240).
    pub loop_invariant: bool,
    /// `func._gctransformer_hint_close_stack_` (call.py `guess_call_kind`
    /// / `get_jitcode`) — a close_stack callee must never produce a
    /// JitCode and is classified `Residual` by `guess_call_kind`.
    pub close_stack: bool,
    /// `func._call_aroundstate_target_` (`call.py` `CallControl.getcalldescr`).
    /// `(funcptr identity, save_err)`. The identity is the funcptr's path,
    /// and when Charon recorded `link_name` a tab plus that symbol. The
    /// machine address is not in the LLBC; `getcalldescr` resolves it from
    /// `function_fnaddrs` (`register_macro_helper_trace_fnaddr`). A miss
    /// leaves `symbolic_fnaddr_for_path` of the funcptr, which
    /// `rewrite_call_release_gil_target` maps through `jit_trace_fnaddrs`.
    pub call_aroundstate_target: Option<(String, i64)>,
    /// `func._dont_inline_`, set by
    /// `WarmRunnerDesc.split_graph_and_record_jitdriver` on the copied
    /// portal.  The backend inliner reads this off the function object; it is
    /// separate from JIT policy hints such as `_jit_unroll_safe_`.
    pub dont_inline: bool,
    /// `func._jit_look_inside_`. `None` is the missing attribute
    /// (`hasattr` is false) and `look_inside_graph` falls through to
    /// `look_inside_function`. `Some` overrides that guess.
    pub jit_look_inside: Option<bool>,
    /// `func._jit_unroll_safe_`. The missing attribute is false, so a
    /// backedge still makes `contains_loop` true.
    pub unroll_safe: bool,
    /// `func.__module__`, the defining module's path as Rust's
    /// `module_path!()` spells it (`pyre_module::module::unicodedata`).
    /// `JitPolicy.look_inside_function` subclasses read it
    /// (`pypy/module/pypyjit/policy.py` `PyPyJitPolicy`); `None` stands
    /// for the absent attribute that `func.__module__ or '?'` covers.
    pub module: Option<String>,
}

impl Default for FuncEffects {
    fn default() -> Self {
        Self {
            oopspec: None,
            oopspec_argnames: Vec::new(),
            cannot_collect: false,
            random_effects_on_gcobjs: false,
            // `canraise.py analyze_external_call` defaults missing to True.
            canraise: true,
            canmallocgc: false,
            cannot_raise_assertion: false,
            memerror_only_assertion: false,
            elidable: false,
            loop_invariant: false,
            close_stack: false,
            call_aroundstate_target: None,
            dont_inline: false,
            jit_look_inside: None,
            unroll_safe: false,
            module: None,
        }
    }
}

impl FuncEffects {
    /// Project one policy token onto the attribute `look_inside_graph`
    /// reads, and an `oopspec:` token onto `func.oopspec`.
    /// `guess_call_kind` treats a set `oopspec` as the builtin signal.
    /// `"elidable"` and `"unroll_safe"` stick. `_jit_look_inside_`
    /// is first-wins: a later `dont_look_inside` or `jit_look_inside`
    /// does not replace a value already written, matching the hint scan
    /// that returns the first override.
    pub fn apply_policy_hint(&mut self, hint: &str) {
        if let Some(spec) = hint.strip_prefix("oopspec:") {
            if !spec.is_empty() {
                self.oopspec = Some(spec.to_string());
            }
            return;
        }
        match hint {
            "elidable" => self.elidable = true,
            "unroll_safe" => self.unroll_safe = true,
            "dont_look_inside" => {
                if self.jit_look_inside.is_none() {
                    self.jit_look_inside = Some(false);
                }
            }
            _ => {
                if let Some(rest) = hint.strip_prefix("jit_look_inside") {
                    let flag = match rest.trim_start_matches('=').trim() {
                        "" | "true" | "True" => true,
                        "false" | "False" => false,
                        _ => true,
                    };
                    if self.jit_look_inside.is_none() {
                        self.jit_look_inside = Some(flag);
                    }
                }
            }
        }
    }

    /// Fold `other`'s set attributes into `self`. Used when a graph is
    /// registered *after* a `mark_*` already recorded effects on the
    /// graph-less external funcobj record for the same path, so the typed
    /// carrier ends up registration-order-insensitive (every present
    /// value / set flag in `other` wins; cleared flags never unset `self`).
    /// `_jit_look_inside_` keeps the value already written on `self`.
    pub fn merge_from(&mut self, other: &FuncEffects) {
        if other.oopspec.is_some() {
            self.oopspec = other.oopspec.clone();
        }
        if !other.oopspec_argnames.is_empty() {
            self.oopspec_argnames = other.oopspec_argnames.clone();
        }
        self.cannot_collect |= other.cannot_collect;
        self.random_effects_on_gcobjs |= other.random_effects_on_gcobjs;
        // Default is true. `&=` keeps an `llexternal` false stamp when the
        // other record is the default.
        self.canraise &= other.canraise;
        self.canmallocgc |= other.canmallocgc;
        self.cannot_raise_assertion |= other.cannot_raise_assertion;
        self.memerror_only_assertion |= other.memerror_only_assertion;
        self.elidable |= other.elidable;
        self.loop_invariant |= other.loop_invariant;
        self.close_stack |= other.close_stack;
        if other.call_aroundstate_target.is_some() {
            self.call_aroundstate_target = other.call_aroundstate_target.clone();
        }
        self.dont_inline |= other.dont_inline;
        if self.jit_look_inside.is_none() {
            self.jit_look_inside = other.jit_look_inside;
        }
        self.unroll_safe |= other.unroll_safe;
        if other.module.is_some() {
            self.module = other.module.clone();
        }
    }
}

/// Identity of a source funcobj: `(source_identity or owner_root, name)`.
///
/// `name` is the graph's `name_path()` — Charon's fully-qualified path for
/// a free function (unique per source), `owner_root` the impl type for a
/// method (so two impls' same-named methods, e.g. `PyFrame::push_value` vs
/// `MIFrame::push_value`, stay distinct).  Every alias spelling of one
/// source funcobj resolves to the same `GraphKey`.
pub type GraphKey = (Option<String>, String);

/// A funcobj's flow graph, built on first demand
/// (`description.py FunctionDesc.cachedgraph`). Clones share the one graph.
#[derive(Clone)]
pub struct LazyGraph(std::rc::Rc<LazyGraphCell>);

struct LazyGraphCell {
    graph: std::cell::OnceCell<Option<std::rc::Rc<FunctionGraph>>>,
    build: std::cell::Cell<Option<Box<dyn FnOnce() -> Option<FunctionGraph>>>>,
    /// What the funcobj declares before its body is built — the
    /// `pygraph.py PyGraph.__init__` part, taken from the code object: the
    /// startblock with the parameters, and the header fields. `None` for a
    /// graph that was built from the start.
    header: Option<std::rc::Rc<FunctionGraph>>,
}

impl LazyGraph {
    /// A graph that is already built.
    pub fn built(graph: impl Into<std::rc::Rc<FunctionGraph>>) -> Self {
        Self(std::rc::Rc::new(LazyGraphCell {
            graph: std::cell::OnceCell::from(Some(graph.into())),
            build: std::cell::Cell::new(None),
            header: None,
        }))
    }

    /// A graph `build` produces on first demand; `None` from `build` means
    /// the funcobj has no graph. `header` is what the funcobj declares: the
    /// built graph starts from it.
    pub fn deferred(
        header: impl Into<std::rc::Rc<FunctionGraph>>,
        build: impl FnOnce() -> Option<FunctionGraph> + 'static,
    ) -> Self {
        Self(std::rc::Rc::new(LazyGraphCell {
            graph: std::cell::OnceCell::new(),
            build: std::cell::Cell::new(Some(Box::new(build))),
            header: Some(header.into()),
        }))
    }

    /// A funcobj declared with no graph: an external function object
    /// (`rffi.llexternal`), whose `header` is what it declares.
    pub fn external(header: impl Into<std::rc::Rc<FunctionGraph>>) -> Self {
        Self(std::rc::Rc::new(LazyGraphCell {
            graph: std::cell::OnceCell::from(None),
            build: std::cell::Cell::new(None),
            header: Some(header.into()),
        }))
    }

    /// Whether the funcobj is known to have no graph, read without building
    /// one.
    pub fn is_graphless(&self) -> bool {
        matches!(self.0.graph.get(), Some(None))
    }

    /// The funcobj identity, read without building the graph.
    pub fn graph_key(&self) -> GraphKey {
        self.header().graph_key()
    }

    /// What the funcobj declares, read without building the graph: the
    /// declared header, or the graph of one built from the start.
    pub fn header(&self) -> &FunctionGraph {
        match (&self.0.header, self.0.graph.get()) {
            (Some(header), _) => header,
            (None, Some(Some(graph))) => graph,
            (None, _) => unreachable!("a LazyGraph without a header is built"),
        }
    }

    /// The graph, built on the first call.
    pub fn get(&self) -> Option<&std::rc::Rc<FunctionGraph>> {
        let cell = &*self.0;
        cell.graph
            .get_or_init(|| cell.build.take()?().map(std::rc::Rc::new))
            .as_ref()
    }

    /// Whether the graph has been built (or found absent) already.
    pub fn is_built(&self) -> bool {
        self.0.graph.get().is_some()
    }

    /// The graph for writing. A graph another holder shares is copied
    /// first, so the write is seen through this handle only.
    pub fn get_mut(&mut self) -> Option<&mut FunctionGraph> {
        if std::rc::Rc::get_mut(&mut self.0).is_none() {
            *self = match self.get() {
                Some(graph) => Self::built(graph.clone()),
                None => return None,
            };
        }
        let cell = std::rc::Rc::get_mut(&mut self.0).expect("unshared handle");
        if cell.graph.get().is_none() {
            let built = cell
                .build
                .take()
                .and_then(|build| build())
                .map(std::rc::Rc::new);
            let _ = cell.graph.set(built);
        }
        cell.graph.get_mut()?.as_mut().map(std::rc::Rc::make_mut)
    }

    /// Whether both handles share one graph.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        std::rc::Rc::ptr_eq(&self.0, &other.0)
    }
}

impl fmt::Debug for LazyGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0.graph.get() {
            Some(Some(graph)) => graph.fmt(f),
            Some(None) => f.write_str("LazyGraph(<no graph>)"),
            None => f.write_str("LazyGraph(<not built>)"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionGraph {
    pub name: String,
    /// Stable identity of the source function object.
    ///
    /// RPython keys graph metadata by the Python function object's identity,
    /// not by `FunctionGraph.name`.  Charon can emit many distinct generated
    /// gateway functions with the same leaf name in different modules, so the
    /// qualified declaration path is carried here to keep those funcobjs
    /// distinct while aliases of one declaration continue to share a graph.
    /// Synthetic/test graphs leave this as `None` and retain the historical
    /// `(owner_root, name)` identity fallback.
    pub source_identity: Option<String>,
    /// Caller's LLBC-local `FunDecl.def_id` of the declaration this graph
    /// was lowered from. Unique within that artefact only; graph lookup
    /// uses the registered path, never this number. `None` for synthetic
    /// and test graphs.
    pub fun_decl_id: Option<u64>,
    /// Impl-block self-type root for graphs produced from `impl <T> { fn m(&self, ...) }`.
    /// Mirrors PyPy's `graph.func.im_class` access (the bound-method's class
    /// reference): RPython lifts `self` as `SomeInstance(getuniqueclassdef(im_class))`
    /// at `description.py FunctionDesc.pycall`, then the rtyper resolves
    /// `self.<field>` against that ClassDef.  Pyre's per-graph
    /// `derive_subject_inputcells` (`flowspace_adapter.rs`) uses this field
    /// to project the receiver inputarg as `SomeInstance(Some(classdef))` instead
    /// of the abstract `SomeInstance(None)` shell that `valuetype_to_someshell(Ref)`
    /// yields when no class hint is available.  `None` for free functions and
    /// synthetic test-fixture graphs.
    pub owner_root: Option<String>,
    pub startblock: BlockId,
    /// RPython `flowspace/model.py FunctionGraph.returnblock` —
    /// `Block([return_var])` with `operations=()` and `exits=()`.
    /// Blocks returning a value route to it via
    /// `Link([value], returnblock)` held in `Block.exits`.
    pub returnblock: BlockId,
    /// RPython `FunctionGraph.exceptblock` — `Block([etype, evalue])`.
    pub exceptblock: BlockId,
    pub blocks: Vec<Block>,
    pub notes: Vec<String>,
    /// The source-level return type string (Rust `syn::ReturnType`
    /// rendered through `qualified_full_type_string`). RPython
    /// equivalent: `funcptr._obj.TO.RESULT` on the
    /// `lltype.FuncType(ARGS, RESULT)` — every graph carries its result
    /// type intrinsically as `op.result.concretetype` of the
    /// returnblock's input variable.  Pyre stores the source string and
    /// projects to `Type` via `return_type_string_to_value_type` so the
    /// JIT codewriter's signature validator reads `FUNC.RESULT` directly
    /// off the callee graph.
    ///
    /// `None` for synthetic test-fixture graphs constructed via
    /// `FunctionGraph::new("name")`; production paths populate via
    /// `with_return_type(rt)` after construction (parse.rs + lib.rs
    /// free-function, trait-method, and inherent-method registration).
    pub return_type: Option<String>,
    /// Class key of the declared return value when it is a by-value ADT
    /// (`core::option::Option<PyError>`, a named struct) — the
    /// `SomeInstance` result a `_signature_` would declare.  `return_type`
    /// only carries the residual-call register kind (`ref`), so a
    /// `dont_look_inside` stub reads the class from here.  `None` for
    /// scalar, pointer and unit returns.
    pub return_class_root: Option<String>,
    /// The declared return is a pointer to the low-level `STR` storage
    /// (`*mut BytesBlock`) — the `SomeString` result a `_signature_`
    /// (`returns=types.str()`) would declare.  `return_type` carries only its
    /// register kind (`ref`), so a `dont_look_inside` stub reads the string
    /// result from here.
    pub return_is_str: bool,
    /// Field-layout spelling of a raw pointer onto a Rust container
    /// (`*mut Vec<T>`, `*mut VecDeque<T>`, `*mut [T]`). A `dont_look_inside`
    /// stub projects it to the same `SomeList` a field of that type carries.
    /// `return_type` is only `ref`, and that shell is a classdef-less
    /// instance, which cannot union with the `GcArray` list.
    pub return_container_root: Option<String>,
    /// Per-graph JIT hints — the `_jit_*_` / `_elidable_function_`
    /// attributes RPython `policy.py look_inside_graph` reads off
    /// `graph.func`. Pyre carries them on the graph itself so
    /// [`crate::policy::JitPolicy::look_inside_graph`] reads the canonical
    /// carrier directly. Empty for graphs with no hints (the common case
    /// and all `FunctionGraph::new` fixtures).
    pub hints: Vec<String>,
    /// RPython `graph.access_directly`. `specialize.py default_specialize`
    /// writes it on the `(AccessDirect, key)` graph. Pyre keeps one
    /// `FunctionGraph` for both specializations, so `find_all_graphs` stamps
    /// this field when its BFS reaches the AccessDirect specialization, right
    /// before `look_inside_graph` reads it. An AccessDirect graph the final
    /// call graph never reaches is never stamped.
    pub access_directly: bool,
    /// Formals of the AccessDirect specialization whose binding still carries
    /// the flag. `None` means that specialization was never bound
    /// (`default_specialize` never built the `(AccessDirect, key)` graph /
    /// `bindinputargs` never ran on its startblock). `Some(ids)` is the
    /// startblock binding after the annotator fixpoint (`bindinputargs` then
    /// `mergeinputargs` / `unionof`); ids only ever shrink.
    pub access_directly_inputs: Option<Vec<u64>>,
    /// Per-function effect attributes RPython reads off `graph.func`
    /// (`func.oopspec`, `_gctransformer_hint_cannot_collect_`, …). Default
    /// (all unset) for `FunctionGraph::new` fixtures; production
    /// registration stamps these through `CallControl::mark_*`.
    pub func: FuncEffects,
}

/// `flowspace/model.py copygraph`, for the non-shallow fresh-Variable case.
///
/// Block ids are positional in the rich model, so the copied graph retains
/// the complete `blocks` order and every `BlockId` while replacing every
/// Variable reachable from any block-owned field with a fresh identity.
#[expect(
    clippy::mutable_key_type,
    reason = "Eq and Hash use immutable Variable identity; concretetype/name cells do not participate"
)]
pub fn copygraph(graph: &FunctionGraph) -> FunctionGraph {
    use crate::flowspace::model::Variable;
    use std::collections::HashMap;

    let mut varmap: HashMap<Variable, Variable> = HashMap::new();
    let mut intern = |var: &Variable| {
        varmap.entry(var.clone()).or_insert_with(|| var.copy());
    };
    for block in &graph.blocks {
        for var in &block.inputargs {
            intern(var);
        }
        for op in &block.operations {
            for var in crate::inline::op_variable_refs(&op.kind) {
                intern(&var);
            }
            if let Some(result) = &op.result {
                intern(result);
            }
        }
        for link in &block.exits {
            for arg in link
                .args
                .iter()
                .chain(link.last_exception.iter())
                .chain(link.last_exc_value.iter())
            {
                if let LinkArg::Value(var) = arg {
                    intern(var);
                }
            }
        }
        match &block.exitswitch {
            Some(ExitSwitch::Value(var)) => intern(var),
            Some(ExitSwitch::Fused { args, .. }) => {
                for var in args {
                    intern(var);
                }
            }
            Some(ExitSwitch::LastException) | None => {}
        }
    }

    let remap_var = |var: &Variable| {
        varmap
            .get(var)
            .cloned()
            .expect("copygraph: every block-owned Variable must be interned")
    };
    let blocks = graph
        .blocks
        .iter()
        .map(|block| {
            let operations = block
                .operations
                .iter()
                .map(|op| SpaceOperation {
                    result: op.result.as_ref().map(&remap_var),
                    kind: crate::inline::remap_op_kind(&op.kind, &remap_var),
                })
                .collect();
            let (exitswitch, exits) = remap_control_flow_metadata_var(
                &block.exitswitch,
                &block.exits,
                &remap_var,
                |block| block,
            );
            Block {
                id: block.id,
                inputargs: block.inputargs.iter().map(&remap_var).collect(),
                operations,
                exitswitch,
                exits,
                dead: block.dead,
                // FrameState Variables are frontend build-time snapshots only;
                // downstream graph copies retain that inert attribution verbatim.
                framestate: block.framestate.clone(),
            }
        })
        .collect();

    FunctionGraph {
        name: graph.name.clone(),
        source_identity: graph.source_identity.clone(),
        fun_decl_id: graph.fun_decl_id,
        owner_root: graph.owner_root.clone(),
        startblock: graph.startblock,
        returnblock: graph.returnblock,
        exceptblock: graph.exceptblock,
        blocks,
        notes: graph.notes.clone(),
        return_type: graph.return_type.clone(),
        return_class_root: graph.return_class_root.clone(),
        return_is_str: graph.return_is_str,
        return_container_root: graph.return_container_root.clone(),
        hints: graph.hints.clone(),
        access_directly: graph.access_directly,
        access_directly_inputs: graph.access_directly_inputs.as_ref().map(|ids| {
            ids.iter()
                .filter_map(|id| {
                    varmap
                        .iter()
                        .find(|(var, _)| var.id() == *id)
                        .map(|(_, copied)| copied.id())
                })
                .collect()
        }),
        func: graph.func.clone(),
    }
}

impl FunctionGraph {
    pub fn new(name: impl Into<String>) -> Self {
        let entry = BlockId(0);
        let returnblock = BlockId(1);
        let exceptblock = BlockId(2);
        // Canonical inputargs (returnvar / etype / evalue) get named
        // Variables matching `flowspace/model.py:21-25`
        // (`Variable("returnvar")`, `Variable("etype")`,
        // `Variable("evalue")`).  Each one's `concretetype` cell
        // stays `None` until the rtyper resolves it; until then
        // `concretetype(v)` returns `ConcreteType::Unknown` (the
        // sentinel for the pre-`setconcretetype` window), which
        // matches the pre-rtyper Variable-with-no-`concretetype`
        // shape upstream.
        let var_returnvar = crate::flowspace::model::Variable::named("returnvar");
        let var_etype = crate::flowspace::model::Variable::named("etype");
        let var_evalue = crate::flowspace::model::Variable::named("evalue");
        Self {
            name: name.into(),
            startblock: entry,
            returnblock,
            exceptblock,
            // RPython `flowspace/model.py FunctionGraph.__init__`:
            //   startblock created empty; returnblock = Block([return_var]);
            //   exceptblock = Block([Variable('etype'), Variable('evalue')]).
            // Final blocks have `operations=()` and `exits=()`; fall-through
            // for the startblock is likewise `exits=[]` until the front-end
            // closes it.
            blocks: vec![
                Block {
                    id: entry,
                    inputargs: Vec::new(),
                    operations: Vec::new(),
                    exitswitch: None,
                    exits: Vec::new(),
                    framestate: None,
                    dead: false,
                },
                Block {
                    id: returnblock,
                    inputargs: vec![var_returnvar],
                    operations: Vec::new(),
                    exitswitch: None,
                    exits: Vec::new(),
                    framestate: None,
                    dead: false,
                },
                Block {
                    id: exceptblock,
                    inputargs: vec![var_etype, var_evalue],
                    operations: Vec::new(),
                    exitswitch: None,
                    exits: Vec::new(),
                    framestate: None,
                    dead: false,
                },
            ],
            notes: Vec::new(),
            return_type: None,
            return_class_root: None,
            return_is_str: false,
            return_container_root: None,
            owner_root: None,
            source_identity: None,
            fun_decl_id: None,
            hints: Vec::new(),
            access_directly: false,
            access_directly_inputs: None,
            func: FuncEffects::default(),
        }
    }

    /// Builder-style setter for `return_type`. Used by production
    /// registration paths (parse.rs / lib.rs) where the source-level
    /// type is available; test fixtures that construct via
    /// `FunctionGraph::new("name")` may skip and leave `None`. The JIT
    /// codewriter signature validator leaves the result un-validated for
    /// `None`-carrying graphs.
    pub fn with_return_type(mut self, rt: impl Into<String>) -> Self {
        self.return_type = Some(rt.into());
        self
    }

    /// The source funcobj's [`GraphKey`].
    pub fn graph_key(&self) -> GraphKey {
        (
            self.source_identity
                .clone()
                .or_else(|| self.owner_root.clone()),
            self.name.clone(),
        )
    }

    /// Append one hint token and project a policy token onto `func`.
    ///
    /// `"elidable"` and `"unroll_safe"` stick. `_jit_look_inside_` is
    /// first-wins, including when the token is already in [`Self::hints`]:
    /// a direct assignment of the vec still projects the first time this
    /// runs. `specialize:memo` and the other non-policy tokens stay in
    /// the vec only.
    pub fn push_hint(&mut self, hint: impl Into<String>) {
        let hint = hint.into();
        self.func.apply_policy_hint(&hint);
        if !self.hints.iter().any(|h| h == &hint) {
            self.hints.push(hint);
        }
    }

    /// Project every token already in [`Self::hints`] onto `func`.
    ///
    /// First insert stores the graph as it was built. A `hints` vec
    /// assigned before that insert has not passed through [`Self::push_hint`].
    pub fn project_policy_hints(&mut self) {
        for hint in self.hints.clone() {
            self.func.apply_policy_hint(&hint);
        }
    }

    /// Builder-style setter for `hints`. Production registration paths
    /// (lib.rs free-function, trait-method, and inherent-method loops)
    /// stamp the parsed `_jit_*_` / `_elidable_function_` hints onto the
    /// graph; test fixtures may skip and leave the list empty. Each token
    /// is applied in order through [`Self::push_hint`].
    pub fn with_hints(mut self, hints: Vec<String>) -> Self {
        for hint in hints {
            self.push_hint(hint);
        }
        self
    }

    /// Builder-style setter for `owner_root` — the impl-block self-type
    /// root for graphs produced from `impl <T> { fn m(&self, ...) }`.
    /// `front::mir` sets this from the impl-method owner it records as
    /// `SemanticFunction.self_ty_root`;
    /// `derive_subject_inputcells` consults it to project the receiver
    /// inputarg as `SomeInstance(Some(getuniqueclassdef(im_class)))`.
    pub fn with_owner_root(mut self, owner: impl Into<String>) -> Self {
        self.owner_root = Some(owner.into());
        self
    }

    pub fn with_source_identity(mut self, identity: impl Into<String>) -> Self {
        self.source_identity = Some(identity.into());
        self
    }

    pub fn with_fun_decl_id(mut self, id: u64) -> Self {
        self.fun_decl_id = Some(id);
        self
    }

    /// Return the canonical exception block and its `(etype, evalue)`
    /// inputarg Variables.
    ///
    /// RPython parity: `flowspace/model.py:21-25` `exceptblock` has
    /// two inputargs, `(etype, evalue)`, and exists eagerly on every
    /// graph.
    pub fn exceptblock_arg_vars(
        &self,
    ) -> (
        BlockId,
        crate::flowspace::model::Variable,
        crate::flowspace::model::Variable,
    ) {
        let inputargs = &self.block(self.exceptblock).inputargs;
        (self.exceptblock, inputargs[0].clone(), inputargs[1].clone())
    }

    pub fn create_block(&mut self) -> BlockId {
        let id = BlockId(self.blocks.len());
        self.blocks.push(Block {
            id,
            inputargs: Vec::new(),
            operations: Vec::new(),
            exitswitch: None,
            exits: Vec::new(),
            framestate: None,
            dead: false,
        });
        id
    }

    /// Create a merge block whose `inputargs` come from
    /// `framestate.getvariables()` — RPython `flowcontext.py
    /// SpamBlock(framestate)` parity:
    ///
    /// ```python
    /// class SpamBlock(Block):
    ///     def __init__(self, framestate):
    ///         Block.__init__(self, framestate.getvariables())
    ///         self.framestate = framestate
    /// ```
    ///
    /// The block's `inputargs` collect Variables from
    /// `locals + flatten(stack) + [exc_type, exc_value]` in positional
    /// order — the same `mergeable` shape `FrameState::getoutputargs`
    /// walks on the predecessor side.  Pairing the two halves of the
    /// recloseblock at this entry point keeps `link.args.len() ==
    /// target.inputargs.len()` (`simplify.py:513`) invariant for every
    /// predecessor of the new block.
    pub fn create_block_from_framestate(&mut self, fs: &FrameState) -> BlockId {
        let inputargs = fs.getvariables(self);
        let id = BlockId(self.blocks.len());
        self.blocks.push(Block {
            id,
            inputargs,
            operations: Vec::new(),
            exitswitch: None,
            exits: Vec::new(),
            framestate: Some(fs.clone()),
            dead: false,
        });
        id
    }

    /// Create a block with `num_args` fresh inputarg `Variable`s
    /// (Phi nodes), each minted via `alloc_value_var_with_type` with
    /// `ConcreteType::Unknown` and returned to the caller for direct
    /// use.  RPython parity: `model.py` `Block`
    /// where each inputarg is a fresh Variable.
    pub fn create_block_with_arg_vars(
        &mut self,
        num_args: usize,
    ) -> (BlockId, Vec<crate::flowspace::model::Variable>) {
        let id = BlockId(self.blocks.len());
        let inputargs: Vec<crate::flowspace::model::Variable> = (0..num_args)
            .map(|_| self.alloc_value_var_with_type(ConcreteType::Unknown))
            .collect();
        self.blocks.push(Block {
            id,
            inputargs: inputargs.clone(),
            operations: Vec::new(),
            exitswitch: None,
            exits: Vec::new(),
            framestate: None,
            dead: false,
        });
        (id, inputargs)
    }

    /// Mint a fresh value [`crate::flowspace::model::Variable`] with
    /// `ConcreteType::Unknown`.
    pub fn alloc_value_var(&mut self) -> crate::flowspace::model::Variable {
        self.alloc_value_var_with_type(ConcreteType::Unknown)
    }

    /// Emit a `core::ptr::null_mut()` call into `block` and return its result
    /// `Variable` — the repr-adaptive null pointer used for a niche
    /// `Option<NonNull<T>>` `None` (the discriminant null-test rhs and the
    /// `Some`/`None` combinator folds).  `null_mut()` types as a classdef-less
    /// nullable `SomeInstance` (annotate) and recovers the null via
    /// `convert_const(None)` on the result repr (rtype), so it unions and
    /// converts against the pointer receiver's own `InstanceRepr`; a bare
    /// `ConstRefNull` is a fixed GCREF `_ptr` that does neither.
    pub fn push_null_mut_ptr(&mut self, block: BlockId) -> crate::flowspace::model::Variable {
        let res = self.alloc_value_var();
        self.block_mut(block).operations.push(SpaceOperation {
            result: Some(res.clone()),
            kind: OpKind::Call {
                target: CallTarget::function_path(["core", "ptr", "null_mut"]),
                args: crate::model::call_args(vec![]),
                result_ty: ValueType::Ref(None),
            },
        });
        res
    }

    /// One repr-adaptive source for every niche-Option null.
    ///
    /// Emits `core::ptr::null_mut()`, then, when `cast` is set, narrows that
    /// null in the same block onto the payload repr and returns the narrowed
    /// value.
    pub fn push_niche_null(
        &mut self,
        block: BlockId,
        cast: Option<&(String, ValueType)>,
    ) -> crate::flowspace::model::Variable {
        let null = self.push_null_mut_ptr(block);
        let Some((root, ty)) = cast else {
            return null;
        };
        let narrowed = self.alloc_value_var_with_type(ConcreteType::Unknown);
        self.block_mut(block).operations.push(SpaceOperation {
            result: Some(narrowed.clone()),
            kind: cast_instance_call_result(root.clone(), null, ty.clone()),
        });
        narrowed
    }

    /// The discriminant of the niche `Option` `opt` into `result` (`Some` = 1):
    /// `not (opt is None)`, spelled `eq(is_(opt, None), False)`.  A `ne` would
    /// be value inequality on a `StringRepr` / `ListRepr` payload
    /// (`rstr.py` / `rlist.py` `rtype_ne`), not a null test.  `fn_ptr` selects
    /// `null_fn` (`ll_ptrtype`); every other niche uses [`Self::push_niche_null`].
    pub fn push_niche_is_some(
        &mut self,
        block: BlockId,
        opt: crate::flowspace::model::Variable,
        fn_ptr: bool,
        cast: Option<&(String, ValueType)>,
        result: crate::flowspace::model::Variable,
    ) {
        let nullc = if fn_ptr {
            self.push_null_fn_ptr_with_spelling(block, None)
        } else {
            self.push_niche_null(block, cast)
        };
        let is_none = self.alloc_value_var();
        let false_var = self.alloc_value_var();
        let ops = &mut self.block_mut(block).operations;
        ops.push(SpaceOperation {
            result: Some(is_none.clone()),
            kind: OpKind::BinOp {
                op: "is_".to_string(),
                lhs: opt,
                rhs: nullc,
                result_ty: ValueType::Int,
            },
        });
        ops.push(SpaceOperation {
            result: Some(false_var.clone()),
            kind: OpKind::ConstBool(false),
        });
        ops.push(SpaceOperation {
            result: Some(result),
            kind: OpKind::BinOp {
                op: "eq".to_string(),
                lhs: is_none,
                rhs: false_var,
                result_ty: ValueType::Int,
            },
        });
    }

    /// Null function pointer for `Option<fn>`'s `None` arm. Annotates as
    /// `SomePtr(FuncType)` (`fn_null_constant`), the same `ll_ptrtype` as a
    /// `fn` field read, so the two arms union. A `null_mut()` null is a
    /// classdef-less `SomeInstance` and does not.
    pub fn push_null_fn_ptr(&mut self, block: BlockId) -> crate::flowspace::model::Variable {
        self.push_null_fn_ptr_with_spelling(block, None)
    }

    /// [`Self::push_null_fn_ptr`] with the FUNC type `lltype.nullptr` takes.
    /// `spelling` is the `fn(inputs) -> output` (or `Option<fn(...)>`)
    /// signature so the null is the same `FuncType` as a `Some` of that
    /// type. `None` keeps the coarse empty `FuncType` for a comparison
    /// against an untyped `fn` / `Option<fn>` address.
    pub fn push_null_fn_ptr_with_spelling(
        &mut self,
        block: BlockId,
        spelling: Option<&str>,
    ) -> crate::flowspace::model::Variable {
        let res = self.alloc_value_var();
        let args = match spelling {
            Some(s) if !s.is_empty() => vec![LinkArg::from(ConstValue::byte_str(s))],
            _ => crate::model::call_args(vec![]),
        };
        self.block_mut(block).operations.push(SpaceOperation {
            result: Some(res.clone()),
            kind: OpKind::Call {
                target: CallTarget::function_path(["core", "ptr", "null_fn"]),
                args,
                result_ty: ValueType::Int,
            },
        });
        res
    }

    /// Mint a fresh value [`crate::flowspace::model::Variable`] with its
    /// [`ConcreteType`] stamped at construction — pyre's analogue of
    /// upstream `Variable(concretetype=...)` (RPython
    /// `flowspace/model.py:Variable.__init__`).  The Variable's
    /// `concretetype` cell receives the canonical `LowLevelType`
    /// representative for `ty` (see [`concrete_to_canonical_lltype`]);
    /// the rtyper's `setconcretetype` later overrides it by writing the
    /// authoritative type onto this same cell.
    pub fn alloc_value_var_with_type(
        &mut self,
        ty: ConcreteType,
    ) -> crate::flowspace::model::Variable {
        let var = crate::flowspace::model::Variable::new();
        var.set_concretetype(concrete_to_canonical_lltype(ty));
        var
    }

    /// `Variable.concretetype` getter — reads the `Variable.concretetype`
    /// cell directly.  RPython's resolver iterates `Variable` instances
    /// directly (`rtyper.py setconcretetype v.concretetype = ...`); use this when
    /// the caller already holds a `&Variable` (e.g.
    /// `OpKind::BinOp.lhs / .rhs`, `block.inputargs[i]`).
    pub fn concretetype_of(var: &crate::flowspace::model::Variable) -> ConcreteType {
        match var.concretetype.borrow().as_ref() {
            Some(lltype) => getkind(lltype),
            None => ConcreteType::Unknown,
        }
    }

    /// Variable-direct kind-publication helper — writes through the
    /// `Variable.concretetype` cell directly, preserving a richer
    /// rtyper-written `LowLevelType` when its `getkind` already
    /// matches `ty`.  Used by passes that already hold a `&Variable`
    /// and want to skip the slot → Variable lookup.
    pub fn set_concretetype_of_inline(var: &crate::flowspace::model::Variable, ty: ConcreteType) {
        let preserve_existing = matches!(
            (ty, var.concretetype.borrow().as_ref()),
            (kind, Some(existing)) if getkind(existing) == kind
        );
        if !preserve_existing {
            let synthetic = concrete_to_canonical_lltype(ty);
            var.set_concretetype(synthetic);
        }
    }

    /// Walk the startblock-reachable block closure (`iterblocks()`
    /// parity, `rpython/flowspace/model.py`) and collect every distinct
    /// [`crate::flowspace::model::Variable`] it references: block
    /// `inputargs`, operation operands
    /// ([`crate::inline::op_variable_refs`]) and results, link `args` /
    /// exception extravars, and the `exitswitch` condition.  Distinct by
    /// identity, in first-appearance DFS order.
    ///
    /// The canonical value enumeration for passes that walk the graph's
    /// values (regalloc coloring, `Variable.annotation` reset /
    /// commit).  Reads the value set straight off the IR.  Values
    /// minted but never referenced are naturally absent — every
    /// consumer filters such values out anyway (`getcolor` returns
    /// `None`, `annotation` is empty).
    #[expect(
        clippy::mutable_key_type,
        reason = "Eq and Hash use immutable identity/value data; interior mutation is excluded, matching RPython identity-keyed dict semantics"
    )]
    pub fn iter_variables(&self) -> Vec<crate::flowspace::model::Variable> {
        use crate::flowspace::model::Variable;
        let mut seen: std::collections::HashSet<Variable> = std::collections::HashSet::new();
        let mut out: Vec<Variable> = Vec::new();
        let push = |v: Variable,
                    seen: &mut std::collections::HashSet<Variable>,
                    out: &mut Vec<Variable>| {
            if seen.insert(v.clone()) {
                out.push(v);
            }
        };
        // `iterblocks()` parity (`rpython/flowspace/model.py`): walk only
        // the startblock-reachable closure over `Block.exits`, id-keyed
        // because block ids need not be index-aligned
        // (`flowspace_adapter::reachable_block_ids`).  Unreachable lowered
        // blocks (orphan `on_unwind` cleanup, pruned `SwitchInt` arms) carry
        // no annotated value — `remove_dead_blocks` drops them before rtyping.
        let by_id: std::collections::HashMap<BlockId, &Block> =
            self.blocks.iter().map(|b| (b.id, b)).collect();
        let mut block_seen: std::collections::HashSet<BlockId> = std::collections::HashSet::new();
        let mut stack = vec![self.startblock];
        while let Some(bid) = stack.pop() {
            if !block_seen.insert(bid) {
                continue;
            }
            let Some(block) = by_id.get(&bid) else {
                continue;
            };
            for v in block.input_variables() {
                push(v.clone(), &mut seen, &mut out);
            }
            for op in &block.operations {
                for v in crate::inline::op_variable_refs(&op.kind) {
                    push(v, &mut seen, &mut out);
                }
                if let Some(result) = &op.result {
                    push(result.clone(), &mut seen, &mut out);
                }
            }
            for link in &block.exits {
                for arg in &link.args {
                    if let Some(v) = arg.as_variable() {
                        push(v.clone(), &mut seen, &mut out);
                    }
                }
                if let Some(v) = link.last_exception.as_ref().and_then(LinkArg::as_variable) {
                    push(v.clone(), &mut seen, &mut out);
                }
                if let Some(v) = link.last_exc_value.as_ref().and_then(LinkArg::as_variable) {
                    push(v.clone(), &mut seen, &mut out);
                }
            }
            if let Some(ExitSwitch::Value(cond)) = &block.exitswitch {
                push(cond.clone(), &mut seen, &mut out);
            }
            stack.extend(block.exits.iter().rev().map(|e| e.target));
        }
        out
    }

    /// Walk the startblock-reachable block closure in `iterblocks()`
    /// order (`rpython/flowspace/model.py`) and return the visited
    /// [`BlockId`]s.  The startblock is yielded first, then each block's
    /// exits are pushed reversed so the first exit is visited first —
    /// the canonical DFS order RPython passes consume.  Block ids need
    /// not be index-aligned with `self.blocks` storage order, so this
    /// is keyed by id, not by Vec position.
    pub fn iterblocks_order(&self) -> Vec<BlockId> {
        let by_id: std::collections::HashMap<BlockId, &Block> =
            self.blocks.iter().map(|b| (b.id, b)).collect();
        let mut seen: std::collections::HashSet<BlockId> = std::collections::HashSet::new();
        let mut order: Vec<BlockId> = Vec::new();
        let mut stack = vec![self.startblock];
        while let Some(bid) = stack.pop() {
            if !seen.insert(bid) {
                continue;
            }
            order.push(bid);
            if let Some(block) = by_id.get(&bid) {
                stack.extend(block.exits.iter().rev().map(|e| e.target));
            }
        }
        order
    }

    /// `FunctionGraph.iterblocks()` (`rpython/flowspace/model.py`): the
    /// startblock-reachable blocks, in [`Self::iterblocks_order`].  A
    /// block left in storage that no link reaches is not visited.
    pub fn iterblocks(&self) -> Vec<&Block> {
        let by_id: std::collections::HashMap<BlockId, &Block> =
            self.blocks.iter().map(|b| (b.id, b)).collect();
        self.iterblocks_order()
            .into_iter()
            .filter_map(|bid| by_id.get(&bid).copied())
            .collect()
    }

    /// Push an op whose fresh result `Variable` is minted in place
    /// when `has_result` is true; callers receive that `Variable`
    /// directly.
    pub fn push_op_var(
        &mut self,
        block: BlockId,
        kind: OpKind,
        has_result: bool,
    ) -> Option<crate::flowspace::model::Variable> {
        let result_var = has_result.then(|| self.alloc_value_var());
        self.blocks[block.0].operations.push(SpaceOperation {
            result: result_var.clone(),
            kind,
        });
        result_var
    }

    /// Push an op with a caller-supplied `result_var`.  Used when the
    /// `Variable` was allocated upstream (e.g. `FrameState::union`
    /// pre-allocates phi variables) and the op now needs to be emitted
    /// with the same `Variable` identity.
    pub fn push_op_with_result_var(
        &mut self,
        block: BlockId,
        kind: OpKind,
        result_var: crate::flowspace::model::Variable,
    ) {
        self.blocks[block.0].operations.push(SpaceOperation {
            result: Some(result_var),
            kind,
        });
    }

    /// RPython `flowspace/model.py recloseblock(*exits)` — stamp
    /// `link.prevblock` on each exit and install them as the block's
    /// exits, overwriting any previous contents.  The `exitswitch`
    /// field is passed alongside so callers who set a bool branch or
    /// can-raise shape keep both halves of the canonical CFG surface
    /// updated in a single call (pyre-side ergonomics; upstream writes
    /// `block.exitswitch = ...` before `closeblock`).
    pub fn set_control_flow_metadata(
        &mut self,
        block: BlockId,
        exitswitch: Option<ExitSwitch>,
        exits: Vec<Link>,
    ) {
        let block_ref = &mut self.blocks[block.0];
        block_ref.exitswitch = exitswitch;
        block_ref.exits = exits
            .into_iter()
            .map(|link| link.with_prevblock(block))
            .collect();
    }

    /// RPython `flowspace/model.py recloseblock(*exits)` — stamp
    /// `link.prevblock` on each exit and install them as the block's
    /// exits, overwriting any previous contents.  Like upstream, this
    /// does not touch `exitswitch`; callers that want to change the
    /// branch/raise discriminator must set it separately before
    /// `closeblock`/`recloseblock`.
    pub fn recloseblock(&mut self, block: BlockId, exits: Vec<Link>) {
        self.blocks[block.0].exits = exits
            .into_iter()
            .map(|link| link.with_prevblock(block))
            .collect();
    }

    /// RPython `flowspace/model.py closeblock(*exits)` —
    /// `assert self.exits == [], "block already closed"` before
    /// delegating to `recloseblock`.  Keep the invariant as a regular
    /// assert, not `debug_assert!`, so release builds match upstream's
    /// fail-fast behavior.
    pub fn closeblock(&mut self, block: BlockId, exits: Vec<Link>) {
        assert!(
            self.blocks[block.0].exits.is_empty(),
            "block {:?} already closed",
            block
        );
        self.recloseblock(block, exits);
    }

    /// Shorthand for the single-exit fall-through shape — one Link to
    /// `target` carrying `args`, `exitswitch = None`.  Upstream
    /// equivalent: `block.closeblock(Link(args, target))`
    /// (`flowspace/model.py`).
    pub fn set_goto(
        &mut self,
        block: BlockId,
        target: BlockId,
        args: Vec<crate::flowspace::model::Variable>,
    ) {
        // `flowspace/model.py Link.__init__` asserts
        // `len(args) == len(target.inputargs)` at construction time.
        // `set_goto` is a production-only wrapper whose `target` is
        // a fully-formed block (inputargs already populated by the
        // caller path); enforce the assert here so a mismatch
        // surfaces at the recloseblock site rather than as an
        // unbalanced subst dict in `eliminate_empty_blocks`
        // (`simplify.py:513`).  The inliner uses
        // `set_control_flow_metadata` directly when threading
        // remapped exits whose targets may transiently lack
        // inputargs; that lower-level path intentionally bypasses
        // this assert.
        let target_inputarg_count = self.block(target).inputargs.len();
        assert_eq!(
            args.len(),
            target_inputarg_count,
            "set_goto: args.len() ({}) != target.inputargs.len() ({}) — \
             block {:?} → target {:?} on graph {:?}",
            args.len(),
            target_inputarg_count,
            block,
            target,
            self.name,
        );
        let link = Link::from_variables(self, args, target, None);
        self.set_control_flow_metadata(block, None, vec![link]);
    }

    /// Close `block` with a single Goto whose `Link.args` keep mixed
    /// Variable|Constant operands — `flowspace/model.py Link`.
    pub fn set_goto_mixed(&mut self, block: BlockId, target: BlockId, args: Vec<LinkArg>) {
        let target_inputarg_count = self.block(target).inputargs.len();
        assert_eq!(
            args.len(),
            target_inputarg_count,
            "set_goto_mixed: args.len() ({}) != target.inputargs.len() ({}) — \
             block {:?} → target {:?} on graph {:?}",
            args.len(),
            target_inputarg_count,
            block,
            target,
            self.name,
        );
        let link = Link::new_mixed(args, target, None);
        self.set_control_flow_metadata(block, None, vec![link]);
    }

    /// Close `block` with a single-exit Link into `target_block` whose
    /// `args` are derived from `pred_state.getoutputargs(target_state,
    /// self)`.  Direct port of `flowcontext.py`:
    ///
    /// ```python
    /// outputargs = currentstate.getoutputargs(newstate)
    /// block.recloseblock(Link(outputargs, newblock))
    /// ```
    ///
    /// `pred_state` is the predecessor's exit framestate (upstream
    /// `currentstate`); `target_state` is the merge block's entry
    /// framestate (upstream `newstate`).  Variable cells route into
    /// `LinkArg::Value(Variable)` (locals via `LinkArg::value(graph,
    /// vid)` lookup, stack / exception via `hlvalue_to_linkarg`);
    /// Constant cells in the stack / exception projections flow
    /// through `LinkArg::Const` directly.  Mixed-shape outputargs are
    /// threaded into a single Link via `Link::new_mixed`, mirroring
    /// upstream's `Link(outputargs, newblock)` constructor where
    /// `outputargs: List[Hlvalue]` carries Variable + Constant cells
    /// side by side.  Use this in place of `set_goto(.., args)` at
    /// every merge-close site that has both predecessor and target
    /// framestates in hand.
    pub fn set_goto_from_framestate(
        &mut self,
        block: BlockId,
        target_block: BlockId,
        pred_state: &FrameState,
        target_state: &FrameState,
    ) {
        let outputargs = pred_state.getoutputargs(target_state, self);
        // Backfill ancestor-defined Variable args into `block`'s
        // inputargs + predecessor link args via the eager-threading
        // helper.  When `target_state.mergeable` carries a carry-through
        // Variable (e.g. the function parameter `cond` that flows
        // unchanged through both arms of an `if`), `getoutputargs`
        // returns it as `LinkArg::Value(cond)` — but `block` may not
        // define `cond` locally, so the Link.args would reference an
        // undefined operand (`flowspace_adapter.rs`).  Walking
        // back through the predecessor chain to add `cond` as a
        // `block.inputargs` entry restores the
        // `flowspace/flowcontext.py setstate(block.framestate)`
        // shape that the AST walker doesn't produce eagerly.
        for arg in &outputargs {
            if let LinkArg::Value(var) = arg {
                self.ensure_variable_at_block(block, var);
            }
        }
        // `flowspace/model.py Link.__init__` asserts
        // `len(args) == len(target.inputargs)` at construction time.
        // Run the same check here so a framestate-driven recloseblock
        // catches the mismatch at the merge-close site rather than
        // surfacing it downstream as an unbalanced subst dict in
        // `eliminate_empty_blocks` (`simplify.py`).
        let target_inputarg_count = self.block(target_block).inputargs.len();
        assert_eq!(
            outputargs.len(),
            target_inputarg_count,
            "set_goto_from_framestate: outputargs.len() ({}) != target.inputargs.len() ({}) — \
             block {:?} → target {:?} on graph {:?}",
            outputargs.len(),
            target_inputarg_count,
            block,
            target_block,
            self.name,
        );
        let link = Link::new_mixed(outputargs, target_block, None);
        self.set_control_flow_metadata(block, None, vec![link]);
    }

    /// `flowcontext.py currentblock.closeblock(Link(outputargs, block))`
    /// — compute link args via `getoutputargs` and install the Link
    /// WITHOUT calling [`Self::ensure_variable_at_block`].
    ///
    /// RPython's `closeblock` (`model.py`) asserts the block is open
    /// (exits empty) before installing exits.  Use [`Self::recloseblock`]
    /// or [`Self::recloseblock_link`] when re-closing an already-closed
    /// block.
    pub fn closeblock_link(
        &mut self,
        block: BlockId,
        target_block: BlockId,
        pred_state: &FrameState,
        target_state: &FrameState,
    ) {
        let outputargs = pred_state.getoutputargs(target_state, self);
        let target_inputarg_count = self.block(target_block).inputargs.len();
        assert_eq!(
            outputargs.len(),
            target_inputarg_count,
            "closeblock_link: outputargs.len() ({}) != target.inputargs.len() ({}) — \
             block {:?} → target {:?} on graph {:?}",
            outputargs.len(),
            target_inputarg_count,
            block,
            target_block,
            self.name,
        );
        let link = Link::new_mixed(outputargs, target_block, None);
        self.closeblock(block, vec![link]);
    }

    /// Like [`Self::closeblock_link`] but for blocks that may already be
    /// closed — delegates to [`Self::recloseblock`] instead of
    /// [`Self::closeblock`].  `model.py recloseblock`.
    pub fn recloseblock_link(
        &mut self,
        block: BlockId,
        target_block: BlockId,
        pred_state: &FrameState,
        target_state: &FrameState,
    ) {
        let outputargs = pred_state.getoutputargs(target_state, self);
        let target_inputarg_count = self.block(target_block).inputargs.len();
        assert_eq!(
            outputargs.len(),
            target_inputarg_count,
            "recloseblock_link: outputargs.len() ({}) != target.inputargs.len() ({}) — \
             block {:?} → target {:?} on graph {:?}",
            outputargs.len(),
            target_inputarg_count,
            block,
            target_block,
            self.name,
        );
        let link = Link::new_mixed(outputargs, target_block, None);
        self.recloseblock(block, vec![link]);
    }

    /// Backfill `var` as a definition reachable at `block`.
    ///
    /// Cycle-safe top-down: adds `var` to `block.inputargs` BEFORE
    /// recursing into predecessors so a back-edge that reaches `block`
    /// again finds the in-progress slot via
    /// [`Self::variable_defined_in_block`] and short-circuits.  For
    /// each predecessor edge `(pred_block, exit_idx)`, recursively
    /// ensures `pred_block` defines `var`, then appends
    /// `LinkArg::Value(var)` to `pred_block.exits[exit_idx].args` so
    /// the arity invariant `len(link.args) == len(target.inputargs)`
    /// is preserved.
    ///
    /// Returns `true` when `var` is now reachable at `block` (was
    /// already defined, or newly threaded).  Returns `false` only when
    /// `block` has no predecessors AND `var` is not defined locally
    /// — a malformed graph state callers should treat as a bug.
    ///
    /// `flowspace/flowcontext.py setstate(block.framestate)`
    /// populates `block.inputargs` with all carry-through Variables at
    /// block entry by walking top-down.  Pyre's AST walker doesn't do
    /// this eagerly, so when a framestate-driven merge
    /// (`create_block_from_framestate`) surfaces an ancestor-defined
    /// Variable in `block.inputargs`, this helper backfills the
    /// predecessor chain bottom-up to the same final graph shape.
    pub fn ensure_variable_at_block(
        &mut self,
        block: BlockId,
        var: &crate::flowspace::model::Variable,
    ) -> bool {
        if self.variable_defined_in_block(block, var) {
            return true;
        }
        let pred_edges: Vec<(BlockId, usize)> = self
            .blocks
            .iter()
            .flat_map(|b| {
                let bid = b.id;
                b.exits.iter().enumerate().filter_map(move |(i, exit)| {
                    if exit.target == block {
                        Some((bid, i))
                    } else {
                        None
                    }
                })
            })
            .collect();
        if pred_edges.is_empty() {
            return false;
        }
        // If `block` is a phi-block (has any `OpKind::Input` op), every
        // inputarg must be paired with an `OpKind::Input { name, .. }` op.
        // When this helper adds a fresh inputarg via the recursive backfill
        // we must also emit the paired op so the invariant holds; otherwise
        // `closeblock_link` / `getoutputargs` panics on arity mismatch.
        let is_phi_block = self
            .block(block)
            .operations
            .iter()
            .any(|op| matches!(op.kind, OpKind::Input { .. }));
        self.block_mut(block).inputargs.push(var.clone());
        if is_phi_block {
            let name = self.value_name_for(var).unwrap_or_default();
            self.push_op_with_result_var(
                block,
                OpKind::Input {
                    name,
                    ty: ValueType::Unknown,
                    class_root: None,
                },
                var.clone(),
            );
        }
        for (pred_block, exit_idx) in pred_edges {
            let ok = self.ensure_variable_at_block(pred_block, var);
            if !ok {
                // The Variable's source name is read identity-first off
                // its `OpKind::Input` op via `value_name_for`; the panic
                // identifies the variable by `id` + source name.
                let name = self.value_name_for(var);
                // Dump the blocks whose framestate references `var` and the
                // blocks that actually define it.  Surfaces stale carry-
                // through bindings (e.g. a sibling control-flow path's
                // local leaking into a slot whose backing Variable is no
                // longer reachable via the predecessor chain) without
                // forcing the engineer to rebuild a custom dump.
                let mut framestate_blocks: Vec<BlockId> = Vec::new();
                for b in &self.blocks {
                    let Some(fs) = b.framestate.as_ref() else {
                        continue;
                    };
                    let in_locals = fs.entries.iter().any(|e| e.as_ref() == Some(var));
                    let in_stack = crate::flowspace::framestate::recursively_flatten(&fs.stack)
                        .iter()
                        .any(|h| {
                            matches!(h, crate::flowspace::model::Hlvalue::Variable(v) if v == var)
                        });
                    if in_locals || in_stack {
                        framestate_blocks.push(b.id);
                    }
                }
                let mut def_blocks: Vec<BlockId> = Vec::new();
                for b in &self.blocks {
                    if b.inputargs.contains(var)
                        || b.operations
                            .iter()
                            .any(|op| op.result.as_ref() == Some(var))
                    {
                        def_blocks.push(b.id);
                    }
                }
                panic!(
                    "ensure_variable_at_block: predecessor {:?} cannot define \
                     Variable id={} (name={:?}) when threading to block {:?} on graph {:?} \
                     — graph corruption: no transitive predecessor chain leads \
                     to a definition site. \
                     Variable appears in framestate of blocks {:?} and in inputargs/operations of blocks {:?}.",
                    pred_block,
                    var.id(),
                    name,
                    block,
                    self.name,
                    framestate_blocks,
                    def_blocks,
                );
            }
            self.block_mut(pred_block).exits[exit_idx]
                .args
                .push(LinkArg::Value(var.clone()));
        }
        true
    }

    /// `true` iff `var` is defined inside `block` — either as a
    /// `Block.inputargs` entry or as the result of one of its
    /// `operations`.  `flowspace/model.py:checkgraph` —
    /// every operand referenced from a block must be defined in that
    /// block.
    pub fn variable_defined_in_block(
        &self,
        block: BlockId,
        var: &crate::flowspace::model::Variable,
    ) -> bool {
        let b = self.block(block);
        if b.inputargs.contains(var) {
            return true;
        }
        b.operations
            .iter()
            .any(|op| op.result.as_ref() == Some(var))
    }

    /// Shorthand for the boolean-branch shape — two Links with
    /// `Bool(false)` / `Bool(true)` exitcases, `exitswitch =
    /// ExitSwitch::Value(cond)`.  Upstream equivalent:
    /// `block.exitswitch = cond;
    ///  block.closeblock(Link(false_args, if_false, False),
    ///                   Link(true_args,  if_true,  True))`
    /// (`flowspace/model.py` + `:304`).
    ///
    /// RPython `flowcontext.py JUMP_IF_FALSE` unconditionally evaluates
    /// `op.bool(w_value).eval(self)` before `guessbool`, so every
    /// `block.exitswitch` Variable is the result of a `bool` HighLevelOp.
    /// The rtyper then specialises `bool` per repr (`rmodel.py CanBeNull
    /// CanBeNull → ptr_nonzero`; `rint.py IntegerRepr → identity`;
    /// `rstr.py → str_nonzero`; etc.).  Pyre's flatten consumer asserts
    /// `block.exitswitch.concretetype == lltype.Bool`
    /// (`flatten.py:248`) — without the unconditional `bool` wrap, a
    /// composite-pattern `match` / `if let` scrutinee of `Ref` kind
    /// reaches `FlatOp::GotoIfNot` (`assembler.rs`), the
    /// hard-coded `goto_if_not/iL` opname forces the walker into the
    /// wrong register bank, and every LoadAttr/StoreAttr arm aborts with
    /// `RegisterOutOfRange` (issue #115).  Routing the cond through a
    /// `bool` hop here re-establishes the upstream invariant in one
    /// place; the rtyper handles the per-repr specialisation downstream.
    pub fn set_branch(
        &mut self,
        block: BlockId,
        cond: crate::flowspace::model::Variable,
        if_true: BlockId,
        true_args: Vec<crate::flowspace::model::Variable>,
        if_false: BlockId,
        false_args: Vec<crate::flowspace::model::Variable>,
    ) {
        // upstream: `op.bool(w_value).eval(self)` — append `bool` hop to
        // `block.operations` before installing it as exitswitch.  The
        // wrap is idempotent (a `bool(bool(_))` chain folds to identity
        // through `IntegerRepr::rtype_bool` / `BoolRepr::rtype_bool`),
        // mirroring upstream which also wraps unconditionally.
        let cond = self
            .push_op_var(
                block,
                OpKind::UnaryOp {
                    op: "bool".into(),
                    operand: cond,
                    result_ty: ValueType::Bool,
                },
                true,
            )
            .expect("UnaryOp { op: \"bool\", .. } produces a value");
        // `flowspace/model.py Link.__init__` arity assert per
        // arm — same rationale as `set_goto`.
        let true_inputarg_count = self.block(if_true).inputargs.len();
        assert_eq!(
            true_args.len(),
            true_inputarg_count,
            "set_branch: true_args.len() ({}) != if_true.inputargs.len() ({}) — \
             block {:?} → if_true {:?} on graph {:?}",
            true_args.len(),
            true_inputarg_count,
            block,
            if_true,
            self.name,
        );
        let false_inputarg_count = self.block(if_false).inputargs.len();
        assert_eq!(
            false_args.len(),
            false_inputarg_count,
            "set_branch: false_args.len() ({}) != if_false.inputargs.len() ({}) — \
             block {:?} → if_false {:?} on graph {:?}",
            false_args.len(),
            false_inputarg_count,
            block,
            if_false,
            self.name,
        );
        let false_link =
            Link::from_variables(self, false_args, if_false, Some(ExitCase::Bool(false)))
                .with_llexitcase_from_exitcase();
        let true_link = Link::from_variables(self, true_args, if_true, Some(ExitCase::Bool(true)))
            .with_llexitcase_from_exitcase();
        self.set_control_flow_metadata(
            block,
            Some(ExitSwitch::Value(cond)),
            vec![false_link, true_link],
        );
    }

    /// Route a return through the graph's canonical `returnblock`.
    ///
    /// RPython `flowcontext.py RETURN_VALUE` pops `w_returnvalue`
    /// from the stack and wraps it in `Return(w_returnvalue)`; in
    /// `flowcontext.py Return.nomoreblocks`, the link is built
    /// as `Link([w_result], ctx.graph.returnblock)`.  For `def foo():
    /// pass`, the bytecode is `LOAD_CONST None; RETURN_VALUE`, so
    /// `w_result` is `Constant(None)` and `Link.args = [Constant(None)]`
    /// — an `Hlvalue::Constant`, not a freshly allocated Variable.
    ///
    /// `value == None` wires `LinkArg::Const(ConstValue::None)` with
    /// `concretetype = Void` into `Link.args[0]` directly, matching
    /// upstream's `Constant(None)`-on-the-stack shape with the
    /// `NoneRepr.lowleveltype = Void` (`rpython/rtyper/rnone.py`)
    /// kind stamped at construction.  Pyre's downstream consumers
    /// (`flatten::constant_kind`, assembler `emit_const`) honour the
    /// explicit concretetype so the constant flows through
    /// `FlatOp::VoidReturn` instead of the `constvalue_kind(None) → 'r'`
    /// fallback that would otherwise route a Ref-kind void constant
    /// into the unsupported `emit_const_r(None)` pool.
    pub fn set_return(&mut self, block: BlockId, value: Option<crate::flowspace::model::Variable>) {
        let returnblock = self.returnblock;
        let arg = match value {
            Some(var) => LinkArg::Value(var),
            None => LinkArg::Const(crate::flowspace::model::Constant::with_concretetype(
                ConstValue::None,
                crate::translator::rtyper::lltypesystem::lltype::VOID,
            )),
        };
        let link = Link::new_mixed(vec![arg], returnblock, None);
        self.set_control_flow_metadata(block, None, vec![link]);
    }

    /// Route `block` to the graph's canonical `exceptblock` — the
    /// upstream-shaped exit for an unrecoverable exception.
    ///
    /// RPython `flowspace/model.py:21-25` declares
    /// `exceptblock = Block([Variable('etype'), Variable('evalue')])` as
    /// the single raise destination per graph.  Predecessor blocks route
    /// to it via `Link(args=[etype, evalue], target=exceptblock)` held in
    /// `Block.exits` — there is no upstream terminator variant that
    /// flags "this block raises".
    ///
    /// Emits an unconditional Link to `graph.exceptblock` carrying the
    /// `RaiseImplicit` `(w_type, w_value)` pair: the `AssertionError`
    /// *class* Constant and a separate `AssertionError(msg)` *instance*
    /// Constant (`flowcontext.py`).  The two slots are never
    /// the same value — `w_value` is an exception instance, not the
    /// class.  A bare raise with no concrete payload — a Rust
    /// `UnwindResume` / `Abort` — has no source exception class, so
    /// `exc_cls` is the upstream `else: Exception` default and the
    /// instance message is the fixed `"implicit Exception shouldn't
    /// occur"`.  Constants are always resolvable by the cutover adapter
    /// (`flowspace_adapter.rs`'s `LinkArg::Const`), unlike the fresh
    /// `alloc_value_var()` placeholders this previously emitted: those
    /// had no producing operation and tripped the adapter's "undefined
    /// operand as Link.args" invariant on every reachable raise block.
    /// `last_exception` / `last_exc_value` extravars are not usable here:
    /// `checkgraph` (`flowspace/model.rs`) permits them only
    /// on the exception exits of a `canraise` block, not on this
    /// unconditional single exit.
    ///
    /// The `_reason` string is retained for optional GraphTransformNote
    /// annotations (see `jtransform.rs::transform_graph`'s abort note);
    /// pass `""` when not applicable.
    ///
    /// Used only where no concrete exception payload is available at
    /// the raise site — the evaluated arguments path should call
    /// `set_raise_values` instead.
    pub fn set_raise(&mut self, block: BlockId, _reason: &str) {
        use crate::flowspace::model::{HOST_ENV, HostObject};
        let exceptblock = self.exceptblock;
        // w_type = Constant(AssertionError), w_value = Constant(AssertionError(msg)).
        let etype = LinkArg::from(ConstValue::builtin("AssertionError"));
        let cls = HOST_ENV
            .lookup_builtin("AssertionError")
            .unwrap_or_else(|| panic!("HOST_ENV missing builtin AssertionError"));
        let message = ConstValue::byte_str("implicit Exception shouldn't occur");
        let instance = HostObject::new_instance(cls, vec![message]);
        let evalue = LinkArg::from(ConstValue::HostObject(instance));
        let link = Link::new_mixed(vec![etype, evalue], exceptblock, None);
        self.set_control_flow_metadata(block, None, vec![link]);
    }

    /// Terminate `block` with the implicit-exception raise shape of
    /// RPython `RaiseImplicit.nomoreblocks`
    /// (`flowspace/flowcontext.py`):
    ///
    /// ```python
    /// msg = "implicit %s shouldn't occur" % exc_cls.__name__
    /// w_type = Constant(AssertionError)
    /// w_value = Constant(AssertionError(msg))
    /// link = Link([w_type, w_value], ctx.graph.exceptblock)
    /// ctx.recorder.crnt_block.closeblock(link)
    /// ```
    ///
    /// Both link args are Constants with `args[0]` carrying the
    /// `AssertionError` class itself — the shape
    /// [`remove_assertion_errors`] matches to prune the branch under
    /// the "AssertionError shouldn't occur at run-time" assumption
    /// (`simplify.py:321-346`).
    pub fn set_raise_implicit(&mut self, block: BlockId, exc_name: &str) {
        use crate::flowspace::model::{Constant, HOST_ENV, HostObject};
        let assert_err = HOST_ENV
            .lookup_builtin("AssertionError")
            .expect("HOST_ENV missing AssertionError");
        let msg = format!("implicit {exc_name} shouldn't occur");
        let w_type = LinkArg::Const(Constant::new(ConstValue::HostObject(assert_err.clone())));
        let w_value = LinkArg::Const(Constant::new(ConstValue::HostObject(
            HostObject::new_instance(assert_err, vec![ConstValue::UniStr(msg)]),
        )));
        let link = Link::new_mixed(vec![w_type, w_value], self.exceptblock, None);
        self.set_control_flow_metadata(block, None, vec![link]);
    }

    /// Terminate `block` with a Link to `exceptblock` carrying the
    /// caller-provided `(etype, evalue)` pair.
    ///
    /// RPython `flowspace/flowcontext.py Raise.nomoreblocks`:
    /// ```python
    /// raise FSException(self.w_type, self.w_value)
    /// ```
    /// The two `W_*` values flow from the preceding `RAISE_VARARGS`
    /// at `flowspace/flowcontext.py`, where `popvalue()`s
    /// provide the exception type / value already on the stack.
    /// This API lets pyre's macro lowering thread the evaluated
    /// Variables into the exceptblock Link so the exception payload
    /// is preserved, not discarded in favor of fresh `alloc_value_var()`
    /// placeholders as `set_raise` does.
    pub fn set_raise_values(
        &mut self,
        block: BlockId,
        etype: crate::flowspace::model::Variable,
        evalue: crate::flowspace::model::Variable,
    ) {
        self.set_goto(block, self.exceptblock, vec![etype, evalue]);
    }

    pub fn block(&self, block: BlockId) -> &Block {
        &self.blocks[block.0]
    }

    /// `Variable.rename(name)` (model.py) — name the Variable
    /// object's `_name`. Identity-shared `_name` (`Rc<RefCell<..>>`)
    /// makes the rename visible through every clone, including the
    /// `Block.inputargs` entry and the registry clone, so the cleaned
    /// name reaches `backendopt/ssa`. `rename` is first-wins ("don't
    /// rename several times"), so re-naming an already-named Variable
    /// is a no-op.
    pub fn name_value_var(
        &mut self,
        var: &crate::flowspace::model::Variable,
        name: impl Into<String>,
    ) {
        let mut v = var.clone();
        v.rename(&name.into());
    }

    /// Source name of `var` if it entered SSA through an
    /// `OpKind::Input` op, resolved by Variable identity directly — no
    /// graph-slot round trip. This is the raw, uncleaned name (the
    /// equivalent of `co_varnames`, read by `signature_for` and the
    /// flowspace adapter's same-name dedup); it is distinct from the
    /// cleaned `Variable._name` carried on the object via `rename`
    /// (read by `backendopt/ssa`). Values that do not enter through an
    /// `Input` op (e.g. a `let` binding to an arithmetic result) have
    /// no source name.
    ///
    /// The `Input` op is the *only* carrier of the raw name: `rename`
    /// stores `clean_name(name)` on `Variable._name` (which appends a
    /// trailing `_` and maps non-identifier chars to `_`), so the
    /// object name cannot reconstruct `co_varnames`. For startblock
    /// formal parameters the recovery is exact 1:1 — the param-binding
    /// loop (`lower_expr_into_graph_with_signature` / `build_function_
    /// graph`) emits a named `Input` op for every inputarg — so the
    /// `arg{N}` fallback in `GraphStore::signature_from_graph` is
    /// unreachable for well-formed graphs. `pygraph.py:16` instead names
    /// the initial-block locals straight from `code.co_varnames` and
    /// stores `code.signature` on the `PyGraph` wrapper; pyre's lifted
    /// callee graphs carry no such wrapper, so the raw name is recovered
    /// from the `Input` op once at registration and kept on the
    /// `GraphStore` slot thereafter. Orthodox reader for callers that
    /// already hold the `Variable` (e.g. `GraphStore::signature_from_graph`
    /// walking `startblock.inputargs`); RPython reads the source name off
    /// the Variable / `co_varnames`, never via an integer value index.
    pub fn value_name_for(&self, var: &crate::flowspace::model::Variable) -> Option<String> {
        // Unnamed values (arithmetic temporaries, constants) carry no
        // source name; `Variable.renamed` is the O(1) gate that keeps
        // the `Input`-op lookup off the hot path for the common case.
        if !var.renamed() {
            return None;
        }
        let var_id = var.id();
        for block in &self.blocks {
            for op in &block.operations {
                if let OpKind::Input { name, .. } = &op.kind
                    && op.result.as_ref().map(|r| r.id()) == Some(var_id)
                {
                    return Some(name.clone());
                }
            }
        }
        None
    }

    /// `Variable.rename(name)` — alias of [`Self::name_value_var`].
    /// Both honour the first-wins idempotency of `rename`; used by
    /// `FrameState::copy` (name-prefix carry-over) and the `mergeblock`
    /// generalize arm (`flowcontext.py`).
    pub fn rename_value_var(
        &mut self,
        var: &crate::flowspace::model::Variable,
        name: impl Into<String>,
    ) {
        self.name_value_var(var, name);
    }

    pub fn block_mut(&mut self, block: BlockId) -> &mut Block {
        &mut self.blocks[block.0]
    }

    /// Append `var` to `block.inputargs` — callers that synthesise
    /// inputargs at front-end / merge time pass the freshly minted
    /// Variable directly onto the upstream-orthodox `Vec<Variable>`
    /// storage.
    pub fn push_inputarg_var(&mut self, block: BlockId, var: crate::flowspace::model::Variable) {
        self.block_mut(block).inputargs.push(var);
    }

    /// Iterate all blocks. RPython: `graph.iterblocks()`.
    pub fn iter_blocks(&self) -> impl Iterator<Item = &Block> {
        self.blocks.iter()
    }

    /// Iterate all (block, op) pairs. RPython: `graph.iterblockops()`.
    pub fn iter_block_ops(&self) -> impl Iterator<Item = (&Block, &SpaceOperation)> {
        self.blocks
            .iter()
            .flat_map(|b| b.operations.iter().map(move |op| (b, op)))
    }

    /// Get successor block IDs for a block.
    ///
    /// RPython `flowspace/model.py FunctionGraph.iterblocks`:
    /// successor set is derived from `Block.exits` only.  Final blocks
    /// (`exits == ()`) — returnblock / exceptblock — have no successors.
    pub fn successors(&self, block: BlockId) -> Vec<BlockId> {
        self.block(block)
            .exits
            .iter()
            .map(|link| link.target)
            .collect()
    }

    /// Get predecessor block IDs for a block.
    pub fn predecessors(&self, target: BlockId) -> Vec<BlockId> {
        self.blocks
            .iter()
            .filter(|b| self.successors(b.id).contains(&target))
            .map(|b| b.id)
            .collect()
    }

    /// Count total operations across all blocks.
    pub fn num_ops(&self) -> usize {
        self.blocks.iter().map(|b| b.operations.len()).sum()
    }

    /// Check if a block is a loop header (has a back-edge predecessor).
    pub fn is_loop_header(&self, block: BlockId) -> bool {
        self.predecessors(block)
            .iter()
            .any(|&pred| pred.0 >= block.0)
    }

    /// Pretty-print the graph (RPython `graph.show()`).
    pub fn dump(&self) -> String {
        let mut out = format!(
            "=== {} ({} blocks, {} ops) ===\n",
            self.name,
            self.blocks.len(),
            self.num_ops()
        );
        for block in &self.blocks {
            let args: Vec<String> = block.inputargs.iter().map(|v| v.name()).collect();
            if args.is_empty() {
                out.push_str(&format!("  Block {}:\n", block.id.0));
            } else {
                out.push_str(&format!("  Block {}({}):\n", block.id.0, args.join(", ")));
            }
            for op in &block.operations {
                let result = op
                    .result
                    .as_ref()
                    .map(|v| format!("{} = ", v.name()))
                    .unwrap_or_default();
                out.push_str(&format!("    {}{:?}\n", result, op.kind));
            }
            // Upstream `flowspace/model.py __repr__` prints the block
            // shape as "block@N with K exits[(exitswitch)]".  Mirror the
            // same summary from pyre's canonical exitswitch/exits pair.
            match &block.exitswitch {
                Some(switch) => out.push_str(&format!(
                    "    → {} exits ({:?})\n",
                    block.exits.len(),
                    switch
                )),
                None => out.push_str(&format!("    → {} exits\n", block.exits.len())),
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {

    /// `merge_from`'s contract is that every set flag in `other` wins and that
    /// registration order does not matter. Each boolean is folded by hand, so
    /// a field added to `FuncEffects` without a matching `|=` is silently
    /// dropped — which is how `canmallocgc` was first written.
    ///
    /// The two directions are the two callers: `insert_function_graph_indexed`
    /// folds a pending `external_funcobjs` record into a graph registered
    /// later, and `GraphStore::insert` folds two aliases of one graph. A flag
    /// that survives only one direction still loses metadata in the other.
    #[test]
    fn merge_from_folds_every_effect_flag_in_both_directions() {
        let set = FuncEffects {
            cannot_collect: true,
            random_effects_on_gcobjs: true,
            canraise: false,
            canmallocgc: true,
            cannot_raise_assertion: true,
            memerror_only_assertion: true,
            elidable: true,
            loop_invariant: true,
            close_stack: true,
            dont_inline: true,
            jit_look_inside: Some(false),
            unroll_safe: true,
            module: Some("m".to_string()),
            ..FuncEffects::default()
        };

        let mut into_default = FuncEffects::default();
        into_default.merge_from(&set);
        assert_eq!(
            into_default, set,
            "a set flag in `other` must win over a cleared `self`"
        );

        let mut from_default = set.clone();
        from_default.merge_from(&FuncEffects::default());
        assert_eq!(
            from_default, set,
            "a cleared flag in `other` must never unset `self`"
        );

        // `_jit_look_inside_` is first-wins. A later alias must not flip it.
        // `_jit_unroll_safe_` ORs.
        let mut first = FuncEffects {
            jit_look_inside: Some(false),
            ..FuncEffects::default()
        };
        let later = FuncEffects {
            jit_look_inside: Some(true),
            unroll_safe: true,
            ..FuncEffects::default()
        };
        first.merge_from(&later);
        assert_eq!(first.jit_look_inside, Some(false));
        assert!(first.unroll_safe);
    }
    use super::*;

    #[test]
    fn copygraph_preserves_structure_with_fresh_variable_identities() {
        use std::collections::HashSet;

        fn variable_ids(graph: &FunctionGraph) -> HashSet<u64> {
            let mut ids = HashSet::new();
            for block in &graph.blocks {
                ids.extend(block.inputargs.iter().map(|var| var.id()));
                for op in &block.operations {
                    ids.extend(
                        crate::inline::op_variable_refs(&op.kind)
                            .iter()
                            .map(|var| var.id()),
                    );
                    ids.extend(op.result.iter().map(|var| var.id()));
                }
                for link in &block.exits {
                    for arg in link
                        .args
                        .iter()
                        .chain(link.last_exception.iter())
                        .chain(link.last_exc_value.iter())
                    {
                        if let LinkArg::Value(var) = arg {
                            ids.insert(var.id());
                        }
                    }
                }
                match &block.exitswitch {
                    Some(ExitSwitch::Value(var)) => {
                        ids.insert(var.id());
                    }
                    Some(ExitSwitch::Fused { args, .. }) => {
                        ids.extend(args.iter().map(|var| var.id()));
                    }
                    Some(ExitSwitch::LastException) | None => {}
                }
            }
            ids
        }

        let mut graph = FunctionGraph::new("copy_fixture");
        let start = graph.startblock;
        let lhs = graph.alloc_value_var_with_type(ConcreteType::Signed);
        let rhs = graph.alloc_value_var_with_type(ConcreteType::Signed);
        let result = graph.alloc_value_var_with_type(ConcreteType::Signed);
        graph.block_mut(start).inputargs = vec![lhs.clone(), rhs.clone()];
        graph.block_mut(start).operations.push(SpaceOperation {
            result: Some(result.clone()),
            kind: OpKind::BinOp {
                op: "int_add".into(),
                lhs: lhs.clone(),
                rhs: rhs.clone(),
                result_ty: ValueType::Int,
            },
        });
        let returnblock = graph.returnblock;
        graph.block_mut(returnblock).inputargs = vec![result.copy()];
        graph.recloseblock(
            start,
            vec![Link::from_variables(
                &graph,
                vec![result.clone()],
                returnblock,
                None,
            )],
        );

        let copied = copygraph(&graph);

        assert_eq!(copied.blocks.len(), graph.blocks.len());
        assert_eq!(copied.startblock, graph.startblock);
        assert_eq!(copied.returnblock, graph.returnblock);
        assert_eq!(copied.exceptblock, graph.exceptblock);
        let original_op = &graph.block(start).operations[0];
        let copied_op = &copied.block(start).operations[0];
        let (
            OpKind::BinOp {
                op: original_name,
                result_ty: original_ty,
                ..
            },
            OpKind::BinOp {
                op: copied_name,
                result_ty: copied_ty,
                ..
            },
        ) = (&original_op.kind, &copied_op.kind)
        else {
            panic!("copygraph must preserve the BinOp shape")
        };
        assert_eq!(original_name, copied_name);
        assert_eq!(original_ty, copied_ty);
        assert_eq!(original_op.result.is_some(), copied_op.result.is_some());

        let original_ids = variable_ids(&graph);
        let copied_ids = variable_ids(&copied);
        assert!(
            original_ids.is_disjoint(&copied_ids),
            "copygraph must not share any operative Variable identity"
        );
    }

    #[test]
    fn transparent_ctor_owner_preserves_nominal_path_and_strips_local_crate() {
        let local = CallTarget::synthetic_transparent_struct_ctor(
            vec!["pyre_object".into(), "intobject".into()],
            "W_IntObject",
        );
        assert_eq!(
            synthetic_transparent_ctor_owner(&local, Some("pyre_object")).as_deref(),
            Some("intobject::W_IntObject")
        );

        let foreign = CallTarget::synthetic_transparent_struct_ctor(
            vec!["foreign_crate".into(), "shadow".into()],
            "W_IntObject",
        );
        assert_eq!(
            synthetic_transparent_ctor_owner(&foreign, Some("pyre_object")).as_deref(),
            Some("foreign_crate::shadow::W_IntObject")
        );
    }

    /// `CallTarget::SyntheticTransparentCtor::path_segments()` must
    /// return the full `(owner_path..., name)` join — `front/mir.rs`
    /// `Aggregate` / `ShallowInitBox` lowerings rely on the qualified
    /// identity to distinguish same-leaf ctors across owner enums
    /// (`StepResult::Continue` vs `JitAction::Continue`).  Collapsing
    /// to the bare leaf would collide their `HostObject` /
    /// `getdesc(pyobj)` keys (`bookkeeper.py`).
    #[test]
    fn synthetic_transparent_ctor_path_segments_preserve_owner_path() {
        let bare = CallTarget::synthetic_transparent_ctor("Continue");
        assert_eq!(bare.path_segments(), Some(vec!["Continue"]));

        let owned = CallTarget::synthetic_transparent_ctor_with_owner(
            vec!["StepResult".to_string()],
            "Continue",
        );
        assert_eq!(owned.path_segments(), Some(vec!["StepResult", "Continue"]));

        let nested = CallTarget::synthetic_transparent_ctor_with_owner(
            vec!["outer".to_string(), "Inner".to_string()],
            "Leaf",
        );
        assert_eq!(nested.path_segments(), Some(vec!["outer", "Inner", "Leaf"]));
    }

    #[test]
    fn qualified_path_split_preserves_paths_inside_generic_arguments() {
        assert_eq!(
            split_qualified_path(
                "core::option::Option<Result<*mut pyobject::PyObject,error::PyError>>"
            ),
            vec![
                "core",
                "option",
                "Option<Result<*mut pyobject::PyObject,error::PyError>>",
            ]
        );
    }

    #[test]
    fn graph_allocates_values_and_blocks() {
        let mut graph = FunctionGraph::new("demo");
        let entry = graph.startblock;
        let cond_var = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "x".into(),
                    ty: ValueType::Int,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let next = graph.create_block();
        graph.set_branch(entry, cond_var, next, vec![], next, vec![]);
        assert_eq!(graph.blocks.len(), 4);
        // `set_branch` mirrors RPython `flowcontext.py JUMP_IF_FALSE`
        // `op.bool(w_value).eval(self)` — every exitswitch is the result
        // of an appended `bool` HighLevelOp, so a branching block carries
        // the original Input op plus the bool wrap (2 ops total).
        assert_eq!(graph.block(entry).operations.len(), 2);
        assert_eq!(graph.block(graph.returnblock).inputargs.len(), 1);
        assert_eq!(graph.block(graph.exceptblock).inputargs.len(), 2);
    }

    /// `fold_constant_exitswitch` folds a switch whose discriminant is a
    /// constant defined not in the switch block itself but in a single
    /// predecessor and threaded across the link arg — the shape the front
    /// `we_are_jitted()` → `ConstBool(true)` fold produces (the `Call`
    /// terminator splits the constant into a separate block from the
    /// `!we_are_jitted()` branch).  Mirrors `constant_fold_graph`'s link
    /// propagation (`backendopt/constfold.py`).
    #[test]
    fn fold_constant_exitswitch_resolves_single_pred_link_arg() {
        let mut graph = FunctionGraph::new("cross_block_fold");
        let entry = graph.startblock;
        // entry: `c = ConstBool(true)`, then goto B(c).
        let c = graph
            .push_op_var(entry, OpKind::ConstBool(true), true)
            .unwrap();
        // B: one inputarg (the threaded constant); branch on it — false arm
        // to `dead`, true arm to `live`.
        let (b, b_args) = graph.create_block_with_arg_vars(1);
        let dead = graph.create_block();
        let live = graph.create_block();
        graph.set_goto(entry, b, vec![c]);
        graph.set_branch(b, b_args[0].clone(), live, vec![], dead, vec![]);
        // Give the arms a benign terminator so the reachability sweep has a
        // well-formed graph to walk.  Zero-arg return block so the arms need
        // no threaded values.
        graph.block_mut(graph.returnblock).inputargs.clear();
        graph.set_goto(live, graph.returnblock, vec![]);
        graph.set_goto(dead, graph.returnblock, vec![]);

        let folded = fold_constant_exitswitch(&mut graph);
        assert_eq!(folded, 1, "the cross-block constant switch must fold");
        // B keeps exactly the taken (true) arm, unconditional.
        let b_block = graph.block(b);
        assert!(b_block.exitswitch.is_none(), "switch removed after fold");
        assert_eq!(b_block.exits.len(), 1, "only the taken arm survives");
        assert_eq!(b_block.exits[0].target, live, "the true arm is taken");
        // After `clear_unreachable_blocks` the false arm is emptied.
        clear_unreachable_blocks(&mut graph);
        assert!(
            graph.block(dead).operations.is_empty() && graph.block(dead).exits.is_empty(),
            "the dead (false) arm is emptied by the reachability sweep"
        );
    }

    #[test]
    fn set_control_flow_metadata_stamps_prevblock() {
        let mut graph = FunctionGraph::new("demo");
        let entry = graph.startblock;
        let next = graph.create_block();
        let link = Link::from_variables(&graph, vec![], next, None);
        graph.set_control_flow_metadata(entry, None, vec![link]);
        assert_eq!(graph.block(entry).exits[0].prevblock, Some(entry));
    }

    #[test]
    #[should_panic(expected = "output args mismatch")]
    fn link_new_panics_on_output_arg_mismatch() {
        let mut graph = FunctionGraph::new("demo");
        let (target, _) = graph.create_block_with_arg_vars(1);
        let _ = Link::from_variables(&graph, vec![], target, None);
    }

    #[test]
    fn set_return_routes_non_void_returns_via_returnblock() {
        let mut graph = FunctionGraph::new("demo");
        let entry = graph.startblock;
        let value_var = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "x".into(),
                    ty: ValueType::Int,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(value_var.clone()));
        // Upstream `flowspace/model.py Block` identifies the routed
        // return by Block.exits carrying a single Link(value, returnblock)
        // with exitswitch=None.
        let entry_block = graph.block(entry);
        assert!(entry_block.exitswitch.is_none());
        assert_eq!(entry_block.exits.len(), 1);
        assert_eq!(entry_block.exits[0].prevblock, Some(entry));
        assert_eq!(entry_block.exits[0].target, graph.returnblock);
        assert_eq!(entry_block.exits[0].args, vec![LinkArg::Value(value_var)]);
    }

    fn assertion_error_except_link(graph: &FunctionGraph, exitcase: Option<ExitCase>) -> Link {
        Link::new_mixed(
            vec![
                LinkArg::from(ConstValue::builtin("AssertionError")),
                LinkArg::from(ConstValue::None),
            ],
            graph.exceptblock,
            exitcase,
        )
        .with_llexitcase_from_exitcase()
    }

    #[test]
    fn remove_assertion_errors_prunes_direct_exceptblock_exit() {
        let mut graph = FunctionGraph::new("direct_assertion_error");
        let entry = graph.startblock;
        let live = graph.create_block();
        let cond = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "cond".into(),
                    ty: ValueType::Bool,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let survivor = Link::from_variables(&graph, vec![], live, Some(ExitCase::Bool(false)))
            .with_llexitcase_from_exitcase();
        let assertion = assertion_error_except_link(&graph, Some(ExitCase::Bool(true)));
        graph.set_control_flow_metadata(
            entry,
            Some(ExitSwitch::Value(cond)),
            vec![survivor, assertion],
        );

        assert_eq!(remove_assertion_errors(&mut graph), 1);
        let entry_block = graph.block(entry);
        assert_eq!(entry_block.exits.len(), 1);
        assert_eq!(entry_block.exits[0].target, live);
        assert!(entry_block.exitswitch.is_none());
        assert!(entry_block.exits[0].exitcase.is_none());
        assert!(entry_block.exits[0].llexitcase.is_none());
    }

    /// The MIR front raises through a block of its own, so the edge only
    /// reaches the shape `remove_assertion_errors` matches after the
    /// retarget.  A block holding operations the dead-op pass would reclaim
    /// qualifies; one holding a `getslice` — outside `CanRemove` — does not.
    #[test]
    fn retarget_assert_raise_blocks_judges_the_block_by_can_remove() {
        let build = |unremovable: bool| {
            let mut graph = FunctionGraph::new("indirect_assertion_error");
            let entry = graph.startblock;
            let live = graph.create_block();
            let raise = graph.create_block();
            let cond = graph
                .push_op_var(
                    entry,
                    OpKind::Input {
                        name: "cond".into(),
                        ty: ValueType::Bool,
                        class_root: None,
                    },
                    true,
                )
                .unwrap();
            graph.set_branch(entry, cond, raise, vec![], live, vec![]);
            let base = graph
                .push_op_var(raise, OpKind::ConstInt(42), true)
                .unwrap();
            if unremovable {
                graph
                    .push_op_var(raise, OpKind::GetSlice { args: vec![base] }, true)
                    .unwrap();
            }
            graph.set_raise_implicit(raise, "ValueError");
            (graph, entry, live, raise)
        };

        let (mut removable, entry, live, _) = build(false);
        assert_eq!(retarget_assert_raise_blocks(&mut removable), 1);
        assert_eq!(remove_assertion_errors(&mut removable), 1);
        let entry_block = removable.block(entry);
        assert_eq!(entry_block.exits.len(), 1);
        assert_eq!(entry_block.exits[0].target, live);
        assert!(entry_block.exitswitch.is_none());
        assert!(entry_block.exits[0].exitcase.is_none());

        let (mut kept, entry, _, raise) = build(true);
        assert_eq!(retarget_assert_raise_blocks(&mut kept), 0);
        assert_eq!(remove_assertion_errors(&mut kept), 0);
        assert!(
            kept.block(entry)
                .exits
                .iter()
                .any(|exit| exit.target == raise)
        );
    }

    #[test]
    fn remove_assertion_errors_keeps_whole_graph_single_exit_raise() {
        let mut graph = FunctionGraph::new("whole_graph_assertion_error");
        let entry = graph.startblock;
        graph.set_raise_implicit(entry, "ValueError");

        assert_eq!(remove_assertion_errors(&mut graph), 0);
        assert_eq!(graph.block(entry).exits.len(), 1);
        assert_eq!(graph.block(entry).exits[0].target, graph.exceptblock);
    }

    #[test]
    fn set_raise_routes_assertionerror_class_and_instance_to_exceptblock() {
        use crate::flowspace::model::ConstValue;
        let mut graph = FunctionGraph::new("demo");
        let entry = graph.startblock;
        graph.set_raise(entry, "");
        // RaiseImplicit.nomoreblocks (flowcontext.py): the
        // exceptblock Link carries the AssertionError *class* as w_type
        // and a distinct AssertionError *instance* as w_value — never the
        // class in both slots.
        let entry_block = graph.block(entry);
        assert_eq!(entry_block.exits.len(), 1);
        let link = &entry_block.exits[0];
        assert_eq!(link.target, graph.exceptblock);
        assert_eq!(link.args.len(), 2);
        let host_of = |arg: &LinkArg| match arg {
            LinkArg::Const(c) => match &c.value {
                ConstValue::HostObject(h) => h.clone(),
                other => panic!("raise payload must be a HostObject Constant, got {other:?}"),
            },
            other => panic!("raise payload must be a Const link arg, got {other:?}"),
        };
        let w_type = host_of(&link.args[0]);
        assert!(w_type.is_class(), "w_type must be the exception class");
        assert!(w_type.qualname().contains("AssertionError"));
        let w_value = host_of(&link.args[1]);
        assert!(
            w_value.is_instance(),
            "w_value must be an exception instance, not the class"
        );
        assert_ne!(
            link.args[0], link.args[1],
            "exceptblock must not carry the same value in both slots"
        );
    }

    #[test]
    fn recloseblock_preserves_existing_exitswitch() {
        let mut graph = FunctionGraph::new("demo");
        let entry = graph.startblock;
        let cond_var = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "cond".into(),
                    ty: ValueType::Int,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let target = graph.create_block();
        graph.block_mut(entry).exitswitch = Some(ExitSwitch::Value(cond_var.clone()));

        let link = Link::from_variables(&graph, vec![], target, None);
        graph.recloseblock(entry, vec![link]);

        assert_eq!(
            graph.block(entry).exitswitch,
            Some(ExitSwitch::Value(cond_var))
        );
        assert_eq!(graph.block(entry).exits[0].prevblock, Some(entry));
    }

    #[test]
    #[should_panic(expected = "already closed")]
    fn closeblock_panics_when_called_twice() {
        let mut graph = FunctionGraph::new("demo");
        let entry = graph.startblock;
        let first = graph.create_block();
        let second = graph.create_block();

        let first_link = Link::from_variables(&graph, vec![], first, None);
        let second_link = Link::from_variables(&graph, vec![], second, None);
        graph.closeblock(entry, vec![first_link]);
        graph.closeblock(entry, vec![second_link]);
    }

    #[test]
    fn framestate_union_kills_one_sided_local() {
        // RPython parity: `framestate.py:110-111` "if w1 is None or w2
        // is None: return None" — a slot present in one predecessor
        // but missing in the other is killed (dropped) from the merged
        // state.  Pyre realises this via dense positional entries:
        // position 0 (`x`) is bound on both sides but with distinct
        // Variable identities, position 1 (`self_only`) bound only on
        // self, position 2 (`other_only`) bound only on other, so both
        // one-sided slots collapse to None-kill at union.
        let mut graph = FunctionGraph::new("test");
        let x_self = graph.alloc_value_var();
        let self_only = graph.alloc_value_var();
        let x_other = graph.alloc_value_var();
        let other_only = graph.alloc_value_var();
        let self_state = FrameState {
            entries: vec![Some(x_self.clone()), Some(self_only), None],
            ..Default::default()
        };
        let other_state = FrameState {
            entries: vec![Some(x_other.clone()), None, Some(other_only)],
            ..Default::default()
        };
        let merged = self_state.union(&other_state, &mut graph).expect(
            "test invariant: AST frontend union is total — entries domain has no \
             UnionError, vestigial stack/exc/blocklist/next_offset (framestate.py:78)",
        );
        assert_eq!(
            merged.entries.len(),
            3,
            "merged state preserves positional length = max(len)"
        );
        let survivor = merged.entries[0]
            .as_ref()
            .expect("slot 0 (x) is bound on both sides and must survive");
        // Slot 0's predecessors disagreed on Variable identity, so
        // `union` minted a fresh phi Variable distinct from both.
        assert!(
            *survivor != x_self && *survivor != x_other,
            "disagreement must mint a fresh phi Variable distinct from both predecessors"
        );
        // Slots 1 (`self_only`) and 2 (`other_only`) are bound on
        // exactly one side each, so both collapse to None-kill —
        // upstream `framestate.py:110-111` "if w1 or w2 is None: …
        // return None" semantics.
        assert!(
            merged.entries[1].is_none() && merged.entries[2].is_none(),
            "one-sided slots must be None-killed positionally"
        );
    }

    /// Convenience helper: install an `OpKind::Input` op AND register its
    /// backing Variable as a phi inputarg on the same block.  Mirrors what
    /// the eager-phi installer at union sites does.
    fn install_phi(
        graph: &mut FunctionGraph,
        block: BlockId,
        name: &str,
    ) -> crate::flowspace::model::Variable {
        let phi_var = graph
            .push_op_var(
                block,
                OpKind::Input {
                    name: name.to_string(),
                    ty: ValueType::Int,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.push_inputarg_var(block, phi_var.clone());
        phi_var
    }

    #[test]
    fn prune_dead_phis_drops_orphan_phi_and_link_arg() {
        // entry → merge (phi 'x' never read) → returnblock(void)
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let const_v_var = graph
            .push_op_var(entry, OpKind::ConstInt(42), true)
            .unwrap();
        let merge = graph.create_block();
        install_phi(&mut graph, merge, "x");
        graph.set_goto(entry, merge, vec![const_v_var.clone()]);
        graph.set_return(merge, None);

        prune_dead_phis(&mut graph);

        assert!(
            graph.block(merge).inputargs.is_empty(),
            "orphan phi inputarg must be removed"
        );
        let entry_exit = &graph.block(entry).exits[0];
        assert!(
            entry_exit.args.is_empty(),
            "predecessor link arg matching the pruned phi must be removed"
        );
        let has_phi_op = graph
            .block(merge)
            .operations
            .iter()
            .any(|op| matches!(&op.kind, OpKind::Input { name, .. } if name == "x"));
        assert!(!has_phi_op, "orphan phi `OpKind::Input` must be dropped");
    }

    #[test]
    fn prune_dead_phis_trims_an_unreachable_predecessors_link_with_the_target() {
        //   entry  ─┐
        //           ├→ merge(inputargs [dead, live]) → returnblock(live)
        //   orphan ─┘
        //
        // `orphan` has no predecessor and is not a calling-convention entry —
        // its inputarg is a phi target with no backing `OpKind::Input`, the
        // shape `jtransform` leaves behind — so the reachability walk excludes
        // it.  Steps 6 and 7 both walk `reachable`, so `merge`'s dead inputarg
        // goes while `orphan`'s link keeps both args, and every later consumer
        // that zips a link against its target's inputargs is then off by one.
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let e_dead = graph.push_op_var(entry, OpKind::ConstInt(1), true).unwrap();
        let e_live = graph.push_op_var(entry, OpKind::ConstInt(2), true).unwrap();

        let merge = graph.create_block();
        install_phi(&mut graph, merge, "dead");
        let live_phi = install_phi(&mut graph, merge, "live");
        graph.set_goto(entry, merge, vec![e_dead, e_live.clone()]);
        graph.set_return(merge, Some(live_phi));

        let orphan = graph.create_block();
        let stranded = graph.alloc_value_var();
        graph.push_inputarg_var(orphan, stranded);
        let o_dead = graph
            .push_op_var(orphan, OpKind::ConstInt(3), true)
            .unwrap();
        let o_live = graph
            .push_op_var(orphan, OpKind::ConstInt(4), true)
            .unwrap();
        graph.set_goto(orphan, merge, vec![o_dead, o_live.clone()]);

        prune_dead_phis(&mut graph);

        assert_eq!(
            graph.block(merge).inputargs.len(),
            1,
            "the unread phi column is dropped"
        );
        assert_eq!(
            graph.block(entry).exits[0].args,
            vec![LinkArg::Value(e_live)],
            "the reachable predecessor keeps the live column"
        );
        assert_eq!(
            graph.block(orphan).exits[0].args,
            vec![LinkArg::Value(o_live)],
            "an unreachable predecessor's link must be trimmed with its target, \
             not left naming the column that was dropped"
        );
    }

    #[test]
    fn prune_dead_phis_retains_live_single_source_phi_pending_ssa_to_ssi() {
        // entry -> merge(phi 'x' read by a BinOp whose result is the
        // function return value) -> returnblock(reads return value).
        // RPython `remove_identical_vars_SSA` (simplify.py) would
        // collapse a phi whose incoming args are all the same value and rename
        // downstream readers to that representative, even a live reader.
        // `remove_duplicate_inputargs` omits that all-equal collapse
        // (PRE-EXISTING-ADAPTATION): on `crate::model` there is no `SSA_to_SSI`
        // repair pass, so the collapse would strand the reader on a value
        // defined in the predecessor and the flowspace adapter would reject it
        // as an undefined operand.  A *live* column is left threaded instead —
        // `SSA_to_SSI` would re-thread it regardless, so the collapse is a net
        // no-op.  This test pins the retained-phi shape; flip it back to the
        // collapse assertions once `SSA_to_SSI` is ported to `crate::model`.
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let const_v_var = graph.push_op_var(entry, OpKind::ConstInt(7), true).unwrap();
        let merge = graph.create_block();
        let phi_x_var = install_phi(&mut graph, merge, "x");
        graph.set_goto(entry, merge, vec![const_v_var.clone()]);
        // BinOp whose result IS read (by `set_return`).  Backward
        // dataflow needs a live consumer of the BinOp result for the
        // pure-op-args→dependencies routing to keep phi_x alive.
        let doubled_var = graph
            .push_op_var(
                merge,
                OpKind::BinOp {
                    op: "int_add".into(),
                    lhs: phi_x_var.clone(),
                    rhs: phi_x_var.clone(),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(merge, Some(doubled_var.clone()));

        prune_dead_phis(&mut graph);

        assert_eq!(
            graph.block(merge).inputargs,
            vec![phi_x_var.clone()],
            "live single-source phi is retained (all-equal collapse deferred pending SSA_to_SSI)"
        );
        let entry_exit = &graph.block(entry).exits[0];
        assert_eq!(
            entry_exit.args.len(),
            1,
            "the predecessor link arg feeding the retained phi is kept"
        );
        let binop = graph
            .block(merge)
            .operations
            .iter()
            .find(|op| op.result.as_ref() == Some(&doubled_var))
            .expect("live BinOp remains");
        assert!(
            matches!(
                &binop.kind,
                OpKind::BinOp { lhs, rhs, .. } if lhs == &phi_x_var && rhs == &phi_x_var
            ),
            "live reader keeps reading the retained phi (not renamed)"
        );
    }

    #[test]
    fn prune_dead_phis_collapses_dead_phi_feeding_dead_pure_op() {
        // A phi whose only "reader" is a pure op whose result is itself
        // dead.  Both must collapse together — `simplify.py:441-445`'s
        // `dependencies[op.result] += op.args` for `canremove` ops
        // means the phi vid never enters `read_vars` (because the
        // pure op's result never enters either), and Step 5's blanket
        // pure-op DCE removes the orphan op so the phi can be safely
        // trimmed at Step 7.
        //
        // entry → merge(phi 'x', dead-result BinOp reading phi_x) →
        // returnblock(void).
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let const_v_var = graph.push_op_var(entry, OpKind::ConstInt(7), true).unwrap();
        let merge = graph.create_block();
        let phi_x_var = install_phi(&mut graph, merge, "x");
        graph.set_goto(entry, merge, vec![const_v_var]);
        let doubled_var = graph
            .push_op_var(
                merge,
                OpKind::BinOp {
                    op: "int_add".into(),
                    lhs: phi_x_var.clone(),
                    rhs: phi_x_var,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        // Void return — `doubled_var` has no live consumer.
        graph.set_return(merge, None);

        prune_dead_phis(&mut graph);

        assert!(
            graph.block(merge).inputargs.is_empty(),
            "phi feeding only a dead pure op must collapse"
        );
        let has_binop = graph.block(merge).operations.iter().any(|op| {
            op.result.as_ref() == Some(&doubled_var) && matches!(op.kind, OpKind::BinOp { .. })
        });
        assert!(!has_binop, "dead pure op must be removed alongside the phi");
        let entry_exit = &graph.block(entry).exits[0];
        assert!(
            entry_exit.args.is_empty(),
            "predecessor link arg matching the pruned phi must be removed"
        );
    }

    #[test]
    fn prune_dead_phis_keeps_dead_raising_getslice() {
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let list = graph
            .push_op_var(entry, OpKind::ConstRefNull, true)
            .unwrap();
        let start = graph.push_op_var(entry, OpKind::ConstInt(0), true).unwrap();
        let stop = graph.push_op_var(entry, OpKind::ConstInt(1), true).unwrap();
        let slice = graph
            .push_op_var(
                entry,
                OpKind::GetSlice {
                    args: vec![list, start, stop],
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, None);

        prune_dead_phis(&mut graph);

        assert!(
            graph
                .block(entry)
                .operations
                .iter()
                .any(|op| op.result.as_ref() == Some(&slice)),
            "dead getslice must remain because `simplify.CanRemove` excludes it"
        );
    }

    #[test]
    fn remove_dead_aggregates_drops_discarded_ctor_and_its_field_stores() {
        // entry: tmp = SyntheticTransparentCtor("Tuple");
        //        setfield(tmp.__pos_0 = b0); setfield(tmp.__pos_1 = b1);
        //        return void.  `tmp` is never read, so the ctor and both
        // field stores are dead (`remove_simple_mallocs`).
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let b0 = graph
            .push_op_var(entry, OpKind::ConstBool(true), true)
            .unwrap();
        let b1 = graph
            .push_op_var(entry, OpKind::ConstBool(false), true)
            .unwrap();
        let tmp = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor("Tuple"),
                    args: crate::model::call_args(vec![]),
                    result_ty: ValueType::Ref(Some("Tuple".into())),
                },
                true,
            )
            .unwrap();
        for (i, v) in [b0, b1].into_iter().enumerate() {
            graph.push_op_var(
                entry,
                OpKind::FieldWrite {
                    base: tmp.clone(),
                    field: FieldDescriptor {
                        name: format!("__pos_{i}"),
                        owner_root: Some("Tuple".into()),
                        owner_id: None,
                        base_is_deref: None,
                        taken_by_address: false,
                        inline_vec: false,
                        vec_part: None,
                        owner_declared_gc: None,
                        host_index: None,
                        scalar_word: None,
                    },
                    value: LinkArg::Value(v),
                    ty: ValueType::Ref(None),
                },
                false,
            );
        }
        graph.set_return(entry, None);

        let removed = remove_dead_aggregates(&mut graph);

        assert_eq!(removed, 3, "ctor + two field stores must be removed");
        let has_ctor_or_store = graph.block(entry).operations.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::SyntheticTransparentCtor { .. },
                    ..
                } | OpKind::FieldWrite { .. }
            )
        });
        assert!(
            !has_ctor_or_store,
            "dead aggregate ctor + stores must be gone"
        );
    }

    #[test]
    fn remove_dead_aggregates_keeps_returned_ctor() {
        // The same construction, but `tmp` is the return value — it is
        // read, so neither the ctor nor its stores may be removed.
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let b0 = graph
            .push_op_var(entry, OpKind::ConstBool(true), true)
            .unwrap();
        let tmp = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor("Tuple"),
                    args: crate::model::call_args(vec![]),
                    result_ty: ValueType::Ref(Some("Tuple".into())),
                },
                true,
            )
            .unwrap();
        graph.push_op_var(
            entry,
            OpKind::FieldWrite {
                base: tmp.clone(),
                field: FieldDescriptor {
                    name: "__pos_0".into(),
                    owner_root: Some("Tuple".into()),
                    owner_id: None,
                    base_is_deref: None,
                    taken_by_address: false,
                    inline_vec: false,
                    vec_part: None,
                    owner_declared_gc: None,
                    host_index: None,
                    scalar_word: None,
                },
                value: LinkArg::Value(b0),
                ty: ValueType::Ref(None),
            },
            false,
        );
        graph.set_return(entry, Some(tmp.clone()));

        let removed = remove_dead_aggregates(&mut graph);

        assert_eq!(removed, 0, "a returned aggregate must be kept");
        assert!(
            graph
                .block(entry)
                .operations
                .iter()
                .any(|op| op.result.as_ref() == Some(&tmp)),
            "the live ctor op must survive"
        );
    }

    #[test]
    fn lower_struct_ptr_writes_threads_a_predecessor_local_store_value() {
        // The aggregate is constructed in `entry`, but only the aggregate
        // itself is threaded into `write_block`.  The re-emitted field store
        // reads `entry`'s `value`, which `entry` dominates, so
        // `thread_undefined_op_operands` carries it over the link and the
        // store reads `write_block`'s own inputarg.
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let value = graph.push_op_var(entry, OpKind::ConstInt(7), true).unwrap();
        let aggregate = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor("Payload"),
                    args: crate::model::call_args(vec![]),
                    result_ty: ValueType::Ref(Some("Payload".into())),
                },
                true,
            )
            .unwrap();
        graph.push_op_var(
            entry,
            OpKind::FieldWrite {
                base: aggregate.clone(),
                field: FieldDescriptor {
                    name: "item".into(),
                    owner_root: Some("Payload".into()),
                    owner_id: None,
                    base_is_deref: None,
                    taken_by_address: false,
                    inline_vec: false,
                    vec_part: None,
                    owner_declared_gc: None,
                    host_index: None,
                    scalar_word: None,
                },
                value: LinkArg::Value(value.clone()),
                ty: ValueType::Int,
            },
            false,
        );
        let (write_block, carried) = graph.create_block_with_arg_vars(1);
        graph.set_goto(entry, write_block, vec![aggregate]);
        let raw = graph
            .push_op_var(write_block, OpKind::ConstRefNull, true)
            .unwrap();
        let destination = graph
            .push_op_var(
                write_block,
                crate::model::cast_instance_call("Payload", raw),
                true,
            )
            .unwrap();
        graph.push_op_var(
            write_block,
            OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec!["core".into(), "ptr".into(), "write".into()],
                    fun_decl_id: None,
                },
                args: crate::model::call_args(vec![destination.clone(), carried[0].clone()]),
                result_ty: ValueType::Void,
            },
            false,
        );
        graph.set_return(write_block, None);

        let attrs = std::collections::HashMap::from([(
            "Payload".to_string(),
            vec![("item".to_string(), ValueType::Int)],
        )]);
        assert_eq!(lower_struct_ptr_writes(&mut graph, &attrs), 1);
        thread_undefined_op_operands(&mut graph);
        let block = graph.block(write_block);
        assert!(!block.operations.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.iter().map(String::as_str).eq(["core", "ptr", "write"])
            )
        }));
        let stored = block
            .operations
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::FieldWrite {
                    base,
                    field,
                    value: LinkArg::Value(stored),
                    ..
                } if base == &destination && field.name == "item" => Some(stored.clone()),
                _ => None,
            })
            .expect("the store is re-emitted at the write");
        assert!(block.inputargs.contains(&stored));
        let link = &graph.block(entry).exits[0];
        assert_eq!(link.target, write_block);
        assert!(link.args.contains(&LinkArg::Value(value)));
    }

    #[test]
    fn lower_struct_ptr_writes_matches_a_leaf_destination_to_a_qualified_ctor() {
        // The destination names the pointee by its bare class root while the
        // constructor carries the declaration's module path; the name →
        // `StructId` table is empty while the front lowers.
        let mut graph = FunctionGraph::new("pyre_object::m::alloc");
        let entry = graph.startblock;
        let aggregate = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor_with_owner(
                        vec!["pyre_object".into(), "m".into()],
                        "Payload",
                    ),
                    args: crate::model::call_args(vec![]),
                    result_ty: ValueType::Ref(Some("Payload".into())),
                },
                true,
            )
            .unwrap();
        graph.push_op_var(
            entry,
            OpKind::FieldWrite {
                base: aggregate.clone(),
                field: FieldDescriptor {
                    name: "item".into(),
                    owner_root: Some("Payload".into()),
                    owner_id: None,
                    base_is_deref: None,
                    taken_by_address: false,
                    inline_vec: false,
                    vec_part: None,
                    owner_declared_gc: None,
                    host_index: None,
                    scalar_word: None,
                },
                value: LinkArg::from(ConstValue::Int(7)),
                ty: ValueType::Int,
            },
            false,
        );
        let raw = graph
            .push_op_var(entry, OpKind::ConstRefNull, true)
            .unwrap();
        let destination = graph
            .push_op_var(
                entry,
                crate::model::cast_instance_call("Payload", raw),
                true,
            )
            .unwrap();
        graph.push_op_var(
            entry,
            OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec!["core".into(), "ptr".into(), "write".into()],
                    fun_decl_id: None,
                },
                args: crate::model::call_args(vec![destination.clone(), aggregate]),
                result_ty: ValueType::Void,
            },
            false,
        );
        graph.set_return(entry, None);

        let attrs = std::collections::HashMap::from([(
            "m::Payload".to_string(),
            vec![("item".to_string(), ValueType::Int)],
        )]);
        assert_eq!(lower_struct_ptr_writes(&mut graph, &attrs), 1);
        assert!(graph.block(entry).operations.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::FieldWrite { base, field, .. } if base == &destination && field.name == "item"
            )
        }));
    }

    #[test]
    fn lower_struct_ptr_writes_declines_nondominating_constant_store() {
        // A literal value is valid in every block, but the store itself is
        // conditional. Hoisting it to the merge would initialize objects that
        // took the other arm, so dominance is required independently of SSI
        // value scope.
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let aggregate = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor("Payload"),
                    args: crate::model::call_args(vec![]),
                    result_ty: ValueType::Ref(Some("Payload".into())),
                },
                true,
            )
            .unwrap();
        let cond = graph
            .push_op_var(entry, OpKind::ConstBool(true), true)
            .unwrap();
        let (left, left_args) = graph.create_block_with_arg_vars(1);
        let (right, right_args) = graph.create_block_with_arg_vars(1);
        let (join, joined) = graph.create_block_with_arg_vars(1);
        graph.block_mut(entry).exitswitch = Some(ExitSwitch::Value(cond));
        graph.closeblock(
            entry,
            vec![
                Link::from_variables(
                    &graph,
                    vec![aggregate.clone()],
                    left,
                    Some(ExitCase::Bool(true)),
                ),
                Link::from_variables(&graph, vec![aggregate], right, Some(ExitCase::Bool(false))),
            ],
        );
        graph.push_op_var(
            left,
            OpKind::FieldWrite {
                base: left_args[0].clone(),
                field: FieldDescriptor {
                    name: "item".into(),
                    owner_root: Some("Payload".into()),
                    owner_id: None,
                    base_is_deref: None,
                    taken_by_address: false,
                    inline_vec: false,
                    vec_part: None,
                    owner_declared_gc: None,
                    host_index: None,
                    scalar_word: None,
                },
                value: LinkArg::from(ConstValue::Int(7)),
                ty: ValueType::Int,
            },
            false,
        );
        graph.set_goto(left, join, vec![left_args[0].clone()]);
        graph.set_goto(right, join, vec![right_args[0].clone()]);
        let raw = graph.push_op_var(join, OpKind::ConstRefNull, true).unwrap();
        let destination = graph
            .push_op_var(join, crate::model::cast_instance_call("Payload", raw), true)
            .unwrap();
        graph.push_op_var(
            join,
            OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec!["core".into(), "ptr".into(), "write".into()],
                    fun_decl_id: None,
                },
                args: crate::model::call_args(vec![destination, joined[0].clone()]),
                result_ty: ValueType::Void,
            },
            false,
        );
        graph.set_return(join, None);

        let attrs = std::collections::HashMap::from([(
            "Payload".to_string(),
            vec![("item".to_string(), ValueType::Int)],
        )]);
        assert_eq!(lower_struct_ptr_writes(&mut graph, &attrs), 0);
    }

    #[test]
    fn lower_struct_ptr_writes_copies_existing_aggregate_fields() {
        // `ptr::write(dst, src)` where `src` is an inputarg of the
        // destination's struct, not a constructor.  Each registered field,
        // header names included, becomes a read of `src` then a store into
        // `dst`, in declaration order.
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let source = graph.alloc_value_var();
        graph.block_mut(entry).inputargs.push(source.clone());
        let raw = graph
            .push_op_var(entry, OpKind::ConstRefNull, true)
            .unwrap();
        let destination = graph
            .push_op_var(
                entry,
                crate::model::cast_instance_call("PyFrame", raw),
                true,
            )
            .unwrap();
        let call_result = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec!["core".into(), "ptr".into(), "write".into()],
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![destination.clone(), source.clone()]),
                    result_ty: ValueType::Void,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, None);

        let attrs = std::collections::HashMap::from([(
            "PyFrame".to_string(),
            vec![
                ("ob_header.w_class".to_string(), ValueType::Ref(None)),
                ("pycode".to_string(), ValueType::Ref(Some("PyCode".into()))),
            ],
        )]);
        assert_eq!(lower_struct_ptr_writes(&mut graph, &attrs), 1);

        let lowered = &graph.block(entry).operations;
        assert!(
            !lowered.iter().any(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } if segments.iter().map(String::as_str).eq(["core", "ptr", "write"])
                )
            }),
            "core::ptr::write must be replaced"
        );
        // raw null, cast, then read/write/read/write, then the unit result.
        assert_eq!(lowered.len(), 7);
        let field = |name: &str| {
            FieldDescriptor::new(name, Some("PyFrame".to_string()))
                .with_owner_id(majit_ir::descr::struct_id_for_name("PyFrame"))
        };
        let read_then_write = |read_at: usize, write_at: usize, name: &str, ty: ValueType| {
            match &lowered[read_at].kind {
                OpKind::FieldRead {
                    base,
                    field: got,
                    ty: got_ty,
                    pure: false,
                } => {
                    assert_eq!(base, &source);
                    assert_eq!(got, &field(name));
                    assert!(got.base_is_local_aggregate());
                    assert_eq!(got_ty, &ty);
                }
                other => panic!("expected read of {name}, got {other:?}"),
            }
            match &lowered[write_at].kind {
                OpKind::FieldWrite {
                    base,
                    field: got,
                    value: LinkArg::Value(value),
                    ty: got_ty,
                } => {
                    assert_eq!(base, &destination);
                    assert_eq!(got, &field(name));
                    assert!(got.base_is_local_aggregate());
                    assert_eq!(got_ty, &ty);
                    assert_eq!(lowered[read_at].result.as_ref(), Some(value));
                }
                other => panic!("expected write of {name}, got {other:?}"),
            }
        };
        read_then_write(2, 3, "ob_header.w_class", ValueType::Ref(None));
        read_then_write(4, 5, "pycode", ValueType::Ref(Some("PyCode".into())));
        assert!(matches!(lowered[6].kind, OpKind::ConstNone));
        assert_eq!(lowered[6].result.as_ref(), Some(&call_result));
    }

    #[test]
    fn lower_struct_ptr_writes_copies_inlined_struct_leaves() {
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let source = graph.alloc_value_var();
        graph.block_mut(entry).inputargs.push(source.clone());
        let raw = graph
            .push_op_var(entry, OpKind::ConstRefNull, true)
            .unwrap();
        let destination = graph
            .push_op_var(entry, crate::model::cast_instance_call("Outer", raw), true)
            .unwrap();
        graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec!["core".into(), "ptr".into(), "write".into()],
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![destination.clone(), source.clone()]),
                    result_ty: ValueType::Void,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, None);

        let attrs = std::collections::HashMap::from([(
            "Outer".to_string(),
            vec![
                ("inner".to_string(), ValueType::Ref(None)),
                ("tail".to_string(), ValueType::Int),
            ],
        )]);
        let inner_leaf = |name: &str| {
            FieldDescriptor::new(name, Some("Inner".to_string())).with_base_is_deref(false)
        };
        let expand = |owner: &str, field: &str| {
            (owner == "Outer" && field == "inner").then(|| {
                vec![
                    (inner_leaf("head"), ValueType::Ref(None)),
                    (inner_leaf("meta"), ValueType::Ref(None)),
                ]
            })
        };
        assert_eq!(lower_struct_ptr_writes_with(&mut graph, &attrs, &expand), 1);

        let lowered = &graph.block(entry).operations;
        assert!(
            !lowered.iter().any(|op| {
                matches!(
                    &op.kind,
                    OpKind::FieldWrite { field, .. } if field.name == "inner"
                )
            }),
            "the inlined field itself must not remain as a store"
        );
        let writes: Vec<_> = lowered
            .iter()
            .filter_map(|op| match &op.kind {
                OpKind::FieldWrite { field, ty, .. } => Some((field.name.as_str(), ty.clone())),
                _ => None,
            })
            .collect();
        assert_eq!(
            writes,
            vec![
                ("head", ValueType::Ref(None)),
                ("meta", ValueType::Ref(None)),
                ("tail", ValueType::Int),
            ]
        );
        for op in lowered {
            if let OpKind::FieldWrite { field, base, .. } = &op.kind {
                assert_eq!(base, &destination);
                if field.name == "head" || field.name == "meta" {
                    assert_eq!(field.owner_root.as_deref(), Some("Inner"));
                }
            }
            if let OpKind::FieldRead { field, base, .. } = &op.kind {
                assert_eq!(base, &source);
                if field.name == "head" || field.name == "meta" {
                    assert_eq!(field.owner_root.as_deref(), Some("Inner"));
                }
            }
        }
    }

    #[test]
    fn lower_inlined_struct_field_writes_copies_leaves() {
        let mut graph = FunctionGraph::new("copy_header");
        let entry = graph.startblock;
        let dst = graph.alloc_value_var();
        let src = graph.alloc_value_var();
        graph.block_mut(entry).inputargs = vec![dst.clone(), src.clone()];
        graph.push_op_var(
            entry,
            OpKind::FieldWrite {
                base: dst.clone(),
                field: FieldDescriptor::new("inner", Some("Outer".into())),
                value: LinkArg::Value(src.clone()),
                ty: ValueType::Ref(None),
            },
            false,
        );
        graph.set_return(entry, None);

        let inner_leaf = |name: &str| {
            FieldDescriptor::new(name, Some("Inner".to_string())).with_base_is_deref(false)
        };
        let expand = |owner: &str, field: &str| {
            (owner == "Outer" && field == "inner").then(|| {
                vec![
                    (inner_leaf("head"), ValueType::Ref(None)),
                    (inner_leaf("meta"), ValueType::Ref(None)),
                ]
            })
        };
        assert_eq!(
            lower_inlined_struct_field_writes(&mut graph, &expand).expect("expand"),
            1
        );

        let lowered = &graph.block(entry).operations;
        assert!(
            !lowered.iter().any(|op| {
                matches!(
                    &op.kind,
                    OpKind::FieldWrite { field, .. } if field.name == "inner"
                )
            }),
            "the inlined field itself must not remain as a store"
        );
        let writes: Vec<_> = lowered
            .iter()
            .filter_map(|op| match &op.kind {
                OpKind::FieldWrite {
                    field, ty, base, ..
                } => {
                    assert_eq!(base, &dst);
                    assert_eq!(field.owner_root.as_deref(), Some("Inner"));
                    Some((field.name.as_str(), ty.clone()))
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            writes,
            vec![
                ("head", ValueType::Ref(None)),
                ("meta", ValueType::Ref(None)),
            ]
        );
        for op in lowered {
            if let OpKind::FieldRead { field, base, .. } = &op.kind {
                assert_eq!(base, &src);
                assert_eq!(field.owner_root.as_deref(), Some("Inner"));
            }
        }
    }

    /// The cast constant stays the class leaf. The result type carries the
    /// declaration path, and that path is the owner both routes compare.
    /// With the name table empty, a leaf constant does not select the
    /// qualified layout and does not match the constructor owner.
    #[test]
    fn lower_struct_ptr_writes_uses_the_cast_result_decl_path() {
        use crate::test_support::register_struct_ids_serialized;

        let _registry = register_struct_ids_serialized(std::collections::HashMap::new());
        let owner = "pyframe::PyFrame";
        let attrs = std::collections::HashMap::from([(
            owner.to_string(),
            vec![("pycode".to_string(), ValueType::Ref(None))],
        )]);

        let copy_of = |qualified: bool| {
            let mut graph = FunctionGraph::new("pyre_interpreter::pyframe::FrameBox::new");
            let entry = graph.startblock;
            let source = graph.alloc_value_var();
            graph.block_mut(entry).inputargs.push(source.clone());
            let raw = graph
                .push_op_var(entry, OpKind::ConstRefNull, true)
                .unwrap();
            let cast = if qualified {
                crate::model::cast_instance_call_result(
                    "PyFrame",
                    raw,
                    ValueType::Ref(Some(owner.to_string())),
                )
            } else {
                crate::model::cast_instance_call("PyFrame", raw)
            };
            assert_eq!(crate::model::cast_instance_root(&cast), Some("PyFrame"));
            let destination = graph.push_op_var(entry, cast, true).unwrap();
            graph.push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec!["core".into(), "ptr".into(), "write".into()],
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![destination, source]),
                    result_ty: ValueType::Void,
                },
                false,
            );
            graph.set_return(entry, None);
            (graph, entry)
        };

        let (mut qualified_copy, entry) = copy_of(true);
        assert_eq!(lower_struct_ptr_writes(&mut qualified_copy, &attrs), 1);
        let copied = qualified_copy
            .block(entry)
            .operations
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::FieldWrite { field, .. } if field.name == "pycode" => Some(field),
                _ => None,
            })
            .expect("qualified cast lowers the copy");
        assert_eq!(copied.owner_root.as_deref(), Some(owner));
        assert!(copied.base_is_local_aggregate());
        let (mut leaf_copy, _) = copy_of(false);
        assert_eq!(lower_struct_ptr_writes(&mut leaf_copy, &attrs), 0);

        let ctor_of = |qualified: bool| {
            let mut graph = FunctionGraph::new("pyre_interpreter::pyframe::FrameBox::new");
            let entry = graph.startblock;
            let aggregate = graph
                .push_op_var(
                    entry,
                    OpKind::Call {
                        target: CallTarget::synthetic_transparent_struct_ctor(
                            vec!["pyre_interpreter".into(), "pyframe".into()],
                            "PyFrame",
                        ),
                        args: Vec::new(),
                        result_ty: ValueType::Ref(Some(
                            "pyre_interpreter::pyframe::PyFrame".into(),
                        )),
                    },
                    true,
                )
                .unwrap();
            let pycode = graph
                .push_op_var(entry, OpKind::ConstRefNull, true)
                .unwrap();
            graph.push_op_var(
                entry,
                OpKind::FieldWrite {
                    base: aggregate.clone(),
                    field: FieldDescriptor::new("pycode", Some("PyFrame".to_string())),
                    value: LinkArg::Value(pycode),
                    ty: ValueType::Ref(None),
                },
                false,
            );
            let raw = graph
                .push_op_var(entry, OpKind::ConstRefNull, true)
                .unwrap();
            let cast = if qualified {
                crate::model::cast_instance_call_result(
                    "PyFrame",
                    raw,
                    ValueType::Ref(Some(owner.to_string())),
                )
            } else {
                crate::model::cast_instance_call("PyFrame", raw)
            };
            let destination = graph.push_op_var(entry, cast, true).unwrap();
            graph.push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec!["core".into(), "ptr".into(), "write".into()],
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![destination.clone(), aggregate]),
                    result_ty: ValueType::Void,
                },
                false,
            );
            graph.set_return(entry, None);
            (graph, destination)
        };

        let (mut qualified_ctor, destination) = ctor_of(true);
        assert_eq!(lower_struct_ptr_writes(&mut qualified_ctor, &attrs), 1);
        assert!(qualified_ctor.blocks.iter().any(|block| {
            block.operations.iter().any(|op| {
                matches!(
                    &op.kind,
                    OpKind::FieldWrite { base, field, .. }
                        if base == &destination && field.name == "pycode"
                )
            })
        }));
        let (mut leaf_ctor, _) = ctor_of(false);
        // `cast_instance_call` records the class root on the result. That
        // root is the constructor's leaf, so the ctor route lowers too.
        assert_eq!(lower_struct_ptr_writes(&mut leaf_ctor, &attrs), 1);
    }

    #[test]
    fn block_dominators_match_iterative_dataflow() {
        use std::collections::HashSet;

        fn reference(graph: &FunctionGraph) -> Vec<HashSet<usize>> {
            let block_count = graph.blocks.len();
            let mut predecessors = vec![HashSet::new(); block_count];
            for (source, block) in graph.blocks.iter().enumerate() {
                for link in &block.exits {
                    predecessors[link.target.0].insert(source);
                }
            }
            let all_blocks: HashSet<usize> = (0..block_count).collect();
            let start = graph.startblock.0;
            let mut dominators = vec![all_blocks.clone(); block_count];
            dominators[start] = HashSet::from([start]);
            let mut changed = true;
            while changed {
                changed = false;
                for block in 0..block_count {
                    if block == start {
                        continue;
                    }
                    let mut predecessor_iter = predecessors[block].iter().copied();
                    let mut next = if let Some(first) = predecessor_iter.next() {
                        let mut intersection = dominators[first].clone();
                        for predecessor in predecessor_iter {
                            intersection
                                .retain(|candidate| dominators[predecessor].contains(candidate));
                        }
                        intersection
                    } else {
                        HashSet::new()
                    };
                    next.insert(block);
                    if next != dominators[block] {
                        dominators[block] = next;
                        changed = true;
                    }
                }
            }
            dominators
        }

        fn assert_same(graph: &FunctionGraph) {
            let expected = reference(graph);
            let got = super::BlockDominators::compute(graph);
            for block in 0..graph.blocks.len() {
                for dominator in 0..graph.blocks.len() {
                    assert_eq!(
                        got.dominates(dominator, block),
                        expected[block].contains(&dominator),
                        "start {} block {block} dominator {dominator} edges {:?}",
                        graph.startblock.0,
                        graph
                            .blocks
                            .iter()
                            .enumerate()
                            .flat_map(|(src, block)| {
                                block.exits.iter().map(move |link| (src, link.target.0))
                            })
                            .collect::<Vec<_>>()
                    );
                }
            }
        }

        // Predecessor-less blocks are their own dominator set. An edge from
        // one of them into a block also reached from start wipes every
        // proper dominator. A cycle with no root stays dominated by all.
        let mut graph = FunctionGraph::new("dom-edges");
        let isolated = graph.create_block();
        let join = graph.create_block();
        let cycle_a = graph.create_block();
        let cycle_b = graph.create_block();
        let entry = graph.startblock;
        graph.blocks[entry.0]
            .exits
            .push(Link::new_mixed(vec![], join, None));
        graph.blocks[isolated.0]
            .exits
            .push(Link::new_mixed(vec![], join, None));
        graph.blocks[cycle_a.0]
            .exits
            .push(Link::new_mixed(vec![], cycle_b, None));
        graph.blocks[cycle_b.0]
            .exits
            .push(Link::new_mixed(vec![], cycle_a, None));
        assert_same(&graph);

        let mut state = 0xC0FFEE_u64;
        for _trial in 0..80 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let n = 3 + (state as usize % 12);
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let mut graph = FunctionGraph::new("dom-random");
            while graph.blocks.len() < n {
                graph.create_block();
            }
            let start = (state as usize) % n;
            graph.startblock = BlockId(start);
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let edge_count = (state as usize) % (n * 2 + 1);
            for _ in 0..edge_count {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let src = (state as usize) % n;
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let dst = (state as usize) % n;
                graph.blocks[src]
                    .exits
                    .push(Link::new_mixed(vec![], BlockId(dst), None));
            }
            assert_same(&graph);
        }
    }

    #[test]
    fn remove_duplicate_inputargs_keeps_distinct_phi_columns() {
        let mut graph = FunctionGraph::new("distinct-phi");
        let entry = graph.startblock;
        let v1 = graph.push_op_var(entry, OpKind::ConstInt(1), true).unwrap();
        let v2 = graph.push_op_var(entry, OpKind::ConstInt(2), true).unwrap();
        let merge = graph.create_block();
        let p = install_phi(&mut graph, merge, "p");
        let q = install_phi(&mut graph, merge, "q");
        graph.set_goto(entry, merge, vec![v1.clone(), v2.clone()]);

        remove_duplicate_inputargs(&mut graph);

        assert_eq!(graph.block(merge).inputargs, vec![p, q]);
        assert_eq!(
            graph.block(entry).exits[0].args,
            vec![LinkArg::Value(v1), LinkArg::Value(v2)]
        );
    }

    #[test]
    fn remove_duplicate_inputargs_merges_duplicate_columns_across_blocks() {
        let mut graph = FunctionGraph::new("dup-phi");
        let entry = graph.startblock;
        let v1 = graph.push_op_var(entry, OpKind::ConstInt(1), true).unwrap();
        let merge = graph.create_block();
        let p = install_phi(&mut graph, merge, "p");
        let q = install_phi(&mut graph, merge, "q");
        graph.set_goto(entry, merge, vec![v1.clone(), v1.clone()]);
        let q_use = graph
            .push_op_var(
                merge,
                OpKind::BinOp {
                    op: "int_add".into(),
                    lhs: q.clone(),
                    rhs: q.clone(),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let next = graph.create_block();
        let a = install_phi(&mut graph, next, "a");
        let b = install_phi(&mut graph, next, "b");
        graph.set_goto(merge, next, vec![p.clone(), q.clone()]);
        let b_use = graph
            .push_op_var(
                next,
                OpKind::BinOp {
                    op: "int_add".into(),
                    lhs: b.clone(),
                    rhs: b.clone(),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();

        remove_duplicate_inputargs(&mut graph);

        assert_eq!(graph.block(merge).inputargs, vec![p.clone()]);
        assert_eq!(graph.block(next).inputargs, vec![a.clone()]);
        assert_eq!(graph.block(entry).exits[0].args, vec![LinkArg::Value(v1)]);
        assert_eq!(
            graph.block(merge).exits[0].args,
            vec![LinkArg::Value(p.clone())]
        );
        let q_op = graph
            .block(merge)
            .operations
            .iter()
            .find(|op| op.result.as_ref() == Some(&q_use))
            .expect("q's reader survives");
        assert!(
            matches!(
                &q_op.kind,
                OpKind::BinOp { lhs, rhs, .. } if lhs == &p && rhs == &p
            ),
            "the merged phi is renamed to the surviving column"
        );
        let b_op = graph
            .block(next)
            .operations
            .iter()
            .find(|op| op.result.as_ref() == Some(&b_use))
            .expect("b's reader survives");
        assert!(
            matches!(
                &b_op.kind,
                OpKind::BinOp { lhs, rhs, .. } if lhs == &a && rhs == &a
            ),
            "a downstream phi fed by the merged pair collapses too"
        );
    }

    /// One predecessor feeds `[v2, v0, v2]` into `[v0, v1, v2]`. The first
    /// walk unions `v2` into `v0`; that union makes the `v0` column equal
    /// to `v1`, so the same block has to be walked again.
    #[test]
    fn remove_duplicate_inputargs_rewalks_block_after_its_own_unions() {
        let mut graph = FunctionGraph::new("self-union-phi");
        let entry = graph.startblock;
        let merge = graph.create_block();
        let v0 = install_phi(&mut graph, merge, "v0");
        let v1 = install_phi(&mut graph, merge, "v1");
        let v2 = install_phi(&mut graph, merge, "v2");
        graph.set_goto(entry, merge, vec![v2.clone(), v0.clone(), v2.clone()]);
        let v1_use = graph
            .push_op_var(
                merge,
                OpKind::BinOp {
                    op: "int_add".into(),
                    lhs: v1.clone(),
                    rhs: v1.clone(),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let v2_use = graph
            .push_op_var(
                merge,
                OpKind::BinOp {
                    op: "int_add".into(),
                    lhs: v2.clone(),
                    rhs: v2.clone(),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();

        remove_duplicate_inputargs(&mut graph);

        assert_eq!(graph.block(merge).inputargs, vec![v0.clone()]);
        assert_eq!(
            graph.block(entry).exits[0].args,
            vec![LinkArg::Value(v0.clone())]
        );
        for use_var in [v1_use, v2_use] {
            let op = graph
                .block(merge)
                .operations
                .iter()
                .find(|op| op.result.as_ref() == Some(&use_var))
                .expect("renamed reader survives");
            assert!(
                matches!(
                    &op.kind,
                    OpKind::BinOp { lhs, rhs, .. } if lhs == &v0 && rhs == &v0
                ),
                "v1 and v2 are renamed to the surviving column"
            );
        }
    }

    /// Entry `(x, x, y, y)` and back edge `(c, d, e, e)` into `(a, b, c, d)`.
    /// The first walk drops `d` into `c`; the next walk, seeing that union,
    /// drops `b` into `a`. Two columns remain.
    #[test]
    fn remove_duplicate_inputargs_rewalks_back_edge_phi() {
        let mut graph = FunctionGraph::new("back-edge-phi");
        let entry = graph.startblock;
        let x = graph.push_op_var(entry, OpKind::ConstInt(1), true).unwrap();
        let y = graph.push_op_var(entry, OpKind::ConstInt(2), true).unwrap();
        let e = graph.push_op_var(entry, OpKind::ConstInt(3), true).unwrap();
        let loop_block = graph.create_block();
        let a = install_phi(&mut graph, loop_block, "a");
        let b = install_phi(&mut graph, loop_block, "b");
        let c = install_phi(&mut graph, loop_block, "c");
        let d = install_phi(&mut graph, loop_block, "d");
        graph.set_goto(
            entry,
            loop_block,
            vec![x.clone(), x.clone(), y.clone(), y.clone()],
        );
        graph.set_goto(
            loop_block,
            loop_block,
            vec![c.clone(), d.clone(), e.clone(), e.clone()],
        );
        let b_use = graph
            .push_op_var(
                loop_block,
                OpKind::BinOp {
                    op: "int_add".into(),
                    lhs: b.clone(),
                    rhs: b.clone(),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let d_use = graph
            .push_op_var(
                loop_block,
                OpKind::BinOp {
                    op: "int_add".into(),
                    lhs: d.clone(),
                    rhs: d.clone(),
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();

        remove_duplicate_inputargs(&mut graph);

        assert_eq!(
            graph.block(loop_block).inputargs,
            vec![a.clone(), c.clone()]
        );
        assert_eq!(
            graph.block(entry).exits[0].args,
            vec![LinkArg::Value(x), LinkArg::Value(y)]
        );
        assert_eq!(
            graph.block(loop_block).exits[0].args,
            vec![LinkArg::Value(c.clone()), LinkArg::Value(e)]
        );
        let b_op = graph
            .block(loop_block)
            .operations
            .iter()
            .find(|op| op.result.as_ref() == Some(&b_use))
            .expect("b's reader survives");
        assert!(matches!(
            &b_op.kind,
            OpKind::BinOp { lhs, rhs, .. } if lhs == &a && rhs == &a
        ));
        let d_op = graph
            .block(loop_block)
            .operations
            .iter()
            .find(|op| op.result.as_ref() == Some(&d_use))
            .expect("d's reader survives");
        assert!(matches!(
            &d_op.kind,
            OpKind::BinOp { lhs, rhs, .. } if lhs == &c && rhs == &c
        ));
    }

    /// `registered_struct_layout` resolves a spelling that is not a key by
    /// asking the StructId registry — the arm every synthetic fixture in this
    /// file skips by publishing the exact ctor spelling. It may only answer
    /// when every key sharing the id carries the same rows, so a registry that
    /// maps two spellings with different layouts onto one id declines instead
    /// of returning whichever the iteration reached first.
    #[test]
    fn registered_struct_layout_resolves_and_declines_across_owner_spellings() {
        use crate::test_support::register_struct_ids_serialized;

        let full = "model_registered_layout_test::Owner";
        let relative = "registered_layout_test::Owner";
        let alias = "model_registered_layout_test::OwnerAlias";
        let other = "model_registered_layout_test::Other";
        let owner_id = majit_ir::descr::StructId::from_canonical(full);
        let other_id = majit_ir::descr::StructId::from_canonical(other);
        // One registration for the whole test: the guard is not reentrant, and
        // the differing-rows case varies the `attrs` argument, not the table.
        let _registry = register_struct_ids_serialized(std::collections::HashMap::from([
            (full.to_string(), Some(owner_id)),
            (relative.to_string(), Some(owner_id)),
            (alias.to_string(), Some(owner_id)),
            (other.to_string(), Some(other_id)),
        ]));

        let rows = || {
            vec![
                ("ob_header".to_string(), ValueType::Ref(None)),
                ("intval".to_string(), ValueType::Int),
            ]
        };

        // Only the crate-relative spelling is a key; the full one reaches the
        // same rows through the id.
        let attrs = std::collections::HashMap::from([(relative.to_string(), rows())]);
        assert_eq!(registered_struct_layout(full, &attrs), Some(&rows()));
        assert_eq!(registered_struct_layout(relative, &attrs), Some(&rows()));

        // An owner the registry knows, but whose id no key shares.
        assert_eq!(registered_struct_layout(other, &attrs), None);

        // An owner no registry entry names at all.
        assert_eq!(
            registered_struct_layout("unregistered::Owner", &attrs),
            None
        );

        // Two keys, one id, disagreeing rows: the answer is not a coin flip.
        let ambiguous = std::collections::HashMap::from([
            (relative.to_string(), rows()),
            (
                alias.to_string(),
                vec![("ob_header".to_string(), ValueType::Ref(None))],
            ),
        ]);
        assert_eq!(registered_struct_layout(full, &ambiguous), None);
    }

    /// The four numeric boxing structs' complete field layouts. Synthetic
    /// fixtures have no StructId registry, so they publish the exact owner
    /// spelling carried by their constructor; production qualified aliases
    /// resolve through `registered_struct_layout`'s StructId lookup.
    fn numeric_boxing_attrs() -> std::collections::HashMap<String, Vec<(String, ValueType)>> {
        std::collections::HashMap::from([
            (
                "PyObject".to_string(),
                vec![
                    ("ob_type".to_string(), ValueType::Ref(None)),
                    ("w_class".to_string(), ValueType::Ref(None)),
                ],
            ),
            (
                "W_FloatObject".to_string(),
                vec![
                    ("ob_header".to_string(), ValueType::Ref(None)),
                    ("floatval".to_string(), ValueType::Float),
                ],
            ),
            (
                "W_IntObject".to_string(),
                vec![
                    ("ob_header".to_string(), ValueType::Ref(None)),
                    ("intval".to_string(), ValueType::Int),
                ],
            ),
            (
                "W_ComplexObject".to_string(),
                vec![
                    ("ob_header".to_string(), ValueType::Ref(None)),
                    ("real".to_string(), ValueType::Float),
                    ("imag".to_string(), ValueType::Float),
                ],
            ),
            (
                "W_LongObject".to_string(),
                vec![
                    ("ob_header".to_string(), ValueType::Ref(None)),
                    ("value".to_string(), ValueType::Ref(None)),
                ],
            ),
        ])
    }

    /// Push the header a base-type constructor builds: a `PyObject` synthetic
    /// ctor storing `ob_type: &T` and `w_class: get_instantiate(&T)`, both fed
    /// from the same `&T` constant.  `fuse_boxing_alloc` keeps the vtable only
    /// where those two agree, so a cluster modelling `ob_type` alone stays
    /// unfused and its `malloc_typed` residual.
    fn push_boxing_header(
        graph: &mut FunctionGraph,
        blk: BlockId,
        ty_addr: i64,
    ) -> crate::flowspace::model::Variable {
        type Var = crate::flowspace::model::Variable;
        let header_field = |base: &Var, name: &str, value: &Var| OpKind::FieldWrite {
            base: base.clone(),
            field: FieldDescriptor {
                name: name.into(),
                owner_root: Some("PyObject".into()),
                owner_id: None,
                base_is_deref: None,
                taken_by_address: false,
                inline_vec: false,
                vec_part: None,
                owner_declared_gc: None,
                host_index: None,
                scalar_word: None,
            },
            value: LinkArg::Value(value.clone()),
            ty: ValueType::Ref(None),
        };
        let ob_type = graph
            .push_op_var(blk, OpKind::ConstRefAddr(ty_addr), true)
            .unwrap();
        let instantiate_arg = graph
            .push_op_var(blk, OpKind::ConstRefAddr(ty_addr), true)
            .unwrap();
        let w_class = graph
            .push_op_var(
                blk,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            "runtime_object".into(),
                            "object_model".into(),
                            "get_instantiate".into(),
                        ],
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![instantiate_arg]),
                    result_ty: ValueType::Ref(Some("object".into())),
                },
                true,
            )
            .unwrap();
        let header = graph
            .push_op_var(
                blk,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor("PyObject"),
                    args: crate::model::call_args(vec![]),
                    result_ty: ValueType::Ref(Some("PyObject".into())),
                },
                true,
            )
            .unwrap();
        graph.push_op_var(blk, header_field(&header, "ob_type", &ob_type), false);
        graph.push_op_var(blk, header_field(&header, "w_class", &w_class), false);
        header
    }

    #[test]
    fn fuse_boxing_alloc_lowers_float_box_cluster() {
        // entry: %v       = const (the boxed payload);
        //        %header  = const (the ob_header value);
        //        %agg     = SyntheticTransparentCtor("W_FloatObject");
        //        FieldWrite(%agg.ob_header = %header);   // header store
        //        FieldWrite(%agg.floatval = %v);         // payload store
        //        %ret     = runtime_object.lltype.malloc_typed(%agg);
        //        return %ret.
        // fuse_boxing_alloc replaces the malloc with
        //        %ret     = NewWithVtable("W_FloatObject");
        //        FieldWrite(%ret.floatval = %v, ty=Float);
        // so the header rides the vtable descriptor (oracle: one NewWithVtable
        // + one floatval setfield, matching `codegen.rs trace_box_float`).
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let v = graph
            .push_op_var(entry, OpKind::ConstFloat(0.0f64.to_bits()), true)
            .unwrap();
        // The `ob_header` carries the resolvable `ob_type` / `w_class` stores
        // `w_float_new` writes, so `fuse_boxing_alloc` captures a non-zero
        // vtable.  A cluster whose `ob_type` store resolves to no constant is
        // left unfused (residual `malloc_typed`), so the header must model the
        // real store chain for the fusion to fire.
        let header = push_boxing_header(&mut graph, entry, 4357049520);
        let agg = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor("W_FloatObject"),
                    args: crate::model::call_args(vec![]),
                    result_ty: ValueType::Ref(Some("W_FloatObject".into())),
                },
                true,
            )
            .unwrap();
        graph.push_op_var(
            entry,
            OpKind::FieldWrite {
                base: agg.clone(),
                field: FieldDescriptor {
                    name: "ob_header".into(),
                    owner_root: Some("W_FloatObject".into()),
                    owner_id: None,
                    base_is_deref: None,
                    taken_by_address: false,
                    inline_vec: false,
                    vec_part: None,
                    owner_declared_gc: None,
                    host_index: None,
                    scalar_word: None,
                },
                value: LinkArg::Value(header),
                ty: ValueType::Ref(None),
            },
            false,
        );
        graph.push_op_var(
            entry,
            OpKind::FieldWrite {
                base: agg.clone(),
                field: FieldDescriptor {
                    name: "floatval".into(),
                    owner_root: Some("W_FloatObject".into()),
                    owner_id: None,
                    base_is_deref: None,
                    taken_by_address: false,
                    inline_vec: false,
                    vec_part: None,
                    owner_declared_gc: None,
                    host_index: None,
                    scalar_word: None,
                },
                value: LinkArg::Value(v.clone()),
                ty: ValueType::Ref(None),
            },
            false,
        );
        let ret = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            crate::runtime_names::crates::OBJECT.into(),
                            "lltype".into(),
                            "malloc_typed".into(),
                        ],
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![agg.clone()]),
                    result_ty: ValueType::Ref(Some("W_FloatObject".into())),
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(ret.clone()));

        let mut ambiguous_graph = graph.clone();
        let mut ambiguous_attrs = numeric_boxing_attrs();
        let float_rows = ambiguous_attrs
            .remove("W_FloatObject")
            .expect("fixture float layout");
        ambiguous_attrs.insert("floatobject::W_FloatObject".to_string(), float_rows);
        ambiguous_attrs.insert(
            "shadow::W_FloatObject".to_string(),
            vec![
                ("ob_header".to_string(), ValueType::Ref(None)),
                ("wrong_payload".to_string(), ValueType::Int),
            ],
        );
        assert_eq!(
            fuse_boxing_alloc(&mut ambiguous_graph, &ambiguous_attrs),
            0,
            "two qualified owners with the same leaf must decline fusion"
        );
        assert!(
            ambiguous_graph.block(entry).operations.iter().any(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::FunctionPath { segments, .. },
                        ..
                    } if segments.last().map(String::as_str) == Some("malloc_typed")
                )
            }),
            "ambiguous leaf layout keeps malloc_typed residual"
        );

        for allocator in [
            "malloc_typed",
            "malloc_typed_managed",
            "malloc_typed_stable",
        ] {
            let mut candidate = graph.clone();
            let allocation = candidate
                .block_mut(entry)
                .operations
                .iter_mut()
                .find(|op| op.result.as_ref() == Some(&ret))
                .expect("malloc operation");
            let OpKind::Call {
                target: CallTarget::FunctionPath { segments, .. },
                ..
            } = &mut allocation.kind
            else {
                panic!("allocation must be a direct call")
            };
            *segments.last_mut().expect("allocator leaf") = allocator.to_string();

            let fused = fuse_boxing_alloc(&mut candidate, &numeric_boxing_attrs());
            assert_eq!(
                fused, 1,
                "{allocator}: exactly one boxing cluster must fuse"
            );

            let ops = &candidate.block(entry).operations;
            let nwv_pos = ops
                .iter()
                .position(|op| {
                    matches!(&op.kind, OpKind::NewWithVtable { owner, .. } if owner == "W_FloatObject")
                })
                .unwrap_or_else(|| panic!("{allocator}: NewWithVtable must be emitted"));
            assert_eq!(
                ops[nwv_pos].result.as_ref(),
                Some(&ret),
                "{allocator}: NewWithVtable must reuse the malloc result register"
            );
            match &ops[nwv_pos + 1].kind {
                OpKind::FieldWrite {
                    base, field, ty, ..
                } => {
                    assert_eq!(
                        base, &ret,
                        "{allocator}: payload store must target the allocation result"
                    );
                    assert_eq!(field.name, "floatval", "payload field must be floatval");
                    assert_eq!(
                        *ty,
                        ValueType::Float,
                        "payload store must be retyped to Float"
                    );
                }
                other => panic!(
                    "{allocator}: expected payload FieldWrite after NewWithVtable, got {other:?}"
                ),
            }
            assert!(
                !ops.iter().any(|op| matches!(
                    &op.kind,
                    OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                        if segments.last().map(String::as_str) == Some(allocator)
                )),
                "{allocator}: no allocation call may survive the fusion"
            );
        }
    }

    #[test]
    fn fuse_boxing_alloc_lowers_box_new_uninit_write_assume_init_cluster() {
        // `Box::new_uninit(); ptr::write(box, agg); assume_init(box)` is the
        // uninit spelling of the same malloc-then-init cluster: NewWithVtable
        // plus the payload store, with the uninit call, the write, and
        // assume_init removed.
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let v = graph
            .push_op_var(entry, OpKind::ConstFloat(0.0f64.to_bits()), true)
            .unwrap();
        let header = push_boxing_header(&mut graph, entry, 4357049520);
        let agg = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor("W_FloatObject"),
                    args: crate::model::call_args(vec![]),
                    result_ty: ValueType::Ref(Some("W_FloatObject".into())),
                },
                true,
            )
            .unwrap();
        graph.push_op_var(
            entry,
            OpKind::FieldWrite {
                base: agg.clone(),
                field: FieldDescriptor {
                    name: "ob_header".into(),
                    owner_root: Some("W_FloatObject".into()),
                    owner_id: None,
                    base_is_deref: None,
                    taken_by_address: false,
                    inline_vec: false,
                    vec_part: None,
                    owner_declared_gc: None,
                    host_index: None,
                    scalar_word: None,
                },
                value: LinkArg::Value(header),
                ty: ValueType::Ref(None),
            },
            false,
        );
        graph.push_op_var(
            entry,
            OpKind::FieldWrite {
                base: agg.clone(),
                field: FieldDescriptor {
                    name: "floatval".into(),
                    owner_root: Some("W_FloatObject".into()),
                    owner_id: None,
                    base_is_deref: None,
                    taken_by_address: false,
                    inline_vec: false,
                    vec_part: None,
                    owner_declared_gc: None,
                    host_index: None,
                    scalar_word: None,
                },
                value: LinkArg::Value(v.clone()),
                ty: ValueType::Ref(None),
            },
            false,
        );
        let uninit = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            "alloc".into(),
                            "boxed".into(),
                            "Box".into(),
                            "new_uninit".into(),
                        ],
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![]),
                    result_ty: ValueType::Ref(Some("W_FloatObject".into())),
                },
                true,
            )
            .unwrap();
        graph.push_op_var(
            entry,
            OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: vec!["core".into(), "ptr".into(), "write".into()],
                    fun_decl_id: None,
                },
                args: crate::model::call_args(vec![uninit.clone(), agg.clone()]),
                result_ty: ValueType::Void,
            },
            false,
        );
        let ret = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec!["boxed".into(), "Box".into(), "assume_init".into()],
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![uninit.clone()]),
                    result_ty: ValueType::Ref(Some("W_FloatObject".into())),
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(ret.clone()));

        let fused = fuse_boxing_alloc(&mut graph, &numeric_boxing_attrs());
        assert_eq!(fused, 1, "the uninit-box cluster must fuse");

        let ops = &graph.block(entry).operations;
        let nwv_pos = ops
            .iter()
            .position(|op| {
                matches!(&op.kind, OpKind::NewWithVtable { owner, vtable }
                    if owner == "W_FloatObject" && *vtable == 4357049520)
            })
            .expect("NewWithVtable must replace assume_init");
        assert_eq!(
            ops[nwv_pos].result.as_ref(),
            Some(&ret),
            "NewWithVtable must reuse the assume_init result register"
        );
        match &ops[nwv_pos + 1].kind {
            OpKind::FieldWrite {
                base, field, ty, ..
            } => {
                assert_eq!(base, &ret);
                assert_eq!(field.name, "floatval");
                assert_eq!(*ty, ValueType::Float);
            }
            other => panic!("expected payload FieldWrite after NewWithVtable, got {other:?}"),
        }
        assert!(
            !ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, args, .. }
                    if is_box_new_uninit_path(segments, args)
                        || is_core_ptr_write_path(segments, args)
                        || is_assume_init_path(segments, args)
            )),
            "new_uninit / ptr::write / assume_init must not survive the fusion: {ops:#?}"
        );
    }

    #[test]
    fn fuse_boxing_alloc_two_same_block_uninit_clusters_with_reversed_assume_init_order() {
        // Allocate A then B, but `assume_init` B before A. Rewrite locations
        // are the `assume_init` ops, so discovery order and rewrite order
        // disagree; removing A's earlier `new_uninit` must not leave B's
        // rewrite index stale.
        fn push_uninit_cluster(
            graph: &mut FunctionGraph,
            entry: crate::model::BlockId,
            bits: u64,
            ty_addr: i64,
        ) -> crate::flowspace::model::Variable {
            let v = graph
                .push_op_var(entry, OpKind::ConstFloat(bits), true)
                .unwrap();
            let header = push_boxing_header(graph, entry, ty_addr);
            let agg = graph
                .push_op_var(
                    entry,
                    OpKind::Call {
                        target: CallTarget::synthetic_transparent_ctor("W_FloatObject"),
                        args: crate::model::call_args(vec![]),
                        result_ty: ValueType::Ref(Some("W_FloatObject".into())),
                    },
                    true,
                )
                .unwrap();
            graph.push_op_var(
                entry,
                OpKind::FieldWrite {
                    base: agg.clone(),
                    field: FieldDescriptor {
                        name: "ob_header".into(),
                        owner_root: Some("W_FloatObject".into()),
                        owner_id: None,
                        base_is_deref: None,
                        taken_by_address: false,
                        inline_vec: false,
                        vec_part: None,
                        owner_declared_gc: None,
                        host_index: None,
                        scalar_word: None,
                    },
                    value: LinkArg::Value(header),
                    ty: ValueType::Ref(None),
                },
                false,
            );
            graph.push_op_var(
                entry,
                OpKind::FieldWrite {
                    base: agg.clone(),
                    field: FieldDescriptor {
                        name: "floatval".into(),
                        owner_root: Some("W_FloatObject".into()),
                        owner_id: None,
                        base_is_deref: None,
                        taken_by_address: false,
                        inline_vec: false,
                        vec_part: None,
                        owner_declared_gc: None,
                        host_index: None,
                        scalar_word: None,
                    },
                    value: LinkArg::Value(v),
                    ty: ValueType::Ref(None),
                },
                false,
            );
            let uninit = graph
                .push_op_var(
                    entry,
                    OpKind::Call {
                        target: CallTarget::function_path(["alloc", "boxed", "Box", "new_uninit"]),
                        args: crate::model::call_args(vec![]),
                        result_ty: ValueType::Ref(Some("W_FloatObject".into())),
                    },
                    true,
                )
                .unwrap();
            graph.push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::function_path(["core", "ptr", "write"]),
                    args: crate::model::call_args(vec![uninit.clone(), agg]),
                    result_ty: ValueType::Void,
                },
                false,
            );
            uninit
        }

        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let uninit_a = push_uninit_cluster(&mut graph, entry, 0.0f64.to_bits(), 4357049520);
        let uninit_b = push_uninit_cluster(&mut graph, entry, 1.0f64.to_bits(), 4357049521);
        let ret_b = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::function_path(["boxed", "Box", "assume_init"]),
                    args: crate::model::call_args(vec![uninit_b]),
                    result_ty: ValueType::Ref(Some("W_FloatObject".into())),
                },
                true,
            )
            .unwrap();
        let ret_a = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::function_path(["boxed", "Box", "assume_init"]),
                    args: crate::model::call_args(vec![uninit_a]),
                    result_ty: ValueType::Ref(Some("W_FloatObject".into())),
                },
                true,
            )
            .unwrap();
        graph.push_op_var(
            entry,
            OpKind::Call {
                target: CallTarget::function_path(["keep_both"]),
                args: crate::model::call_args(vec![ret_a, ret_b]),
                result_ty: ValueType::Void,
            },
            false,
        );
        graph.set_return(entry, None);

        let fused = fuse_boxing_alloc(&mut graph, &numeric_boxing_attrs());
        assert_eq!(fused, 2, "both reversed-order uninit clusters must fuse");
        let nwv = graph
            .block(entry)
            .operations
            .iter()
            .filter(|op| matches!(op.kind, OpKind::NewWithVtable { .. }))
            .count();
        assert_eq!(nwv, 2, "each cluster must emit NewWithVtable");
        assert!(
            !graph.block(entry).operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, args, .. }
                    if is_box_new_uninit_path(segments, args)
                        || is_core_ptr_write_path(segments, args)
                        || is_assume_init_path(segments, args)
            )),
            "new_uninit / ptr::write / assume_init must not survive reversed-order fusion"
        );
    }

    #[test]
    fn fuse_boxing_alloc_ignores_box_new_lookalike() {
        // `boxed::Box::new(agg)` shares the owner but is not the no-arg
        // uninit allocation; without an `lltype::malloc*` flavor it stays
        // residual.
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let v = graph
            .push_op_var(entry, OpKind::ConstFloat(0.0f64.to_bits()), true)
            .unwrap();
        let header = push_boxing_header(&mut graph, entry, 4357049520);
        let agg = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor("W_FloatObject"),
                    args: crate::model::call_args(vec![]),
                    result_ty: ValueType::Ref(Some("W_FloatObject".into())),
                },
                true,
            )
            .unwrap();
        graph.push_op_var(
            entry,
            OpKind::FieldWrite {
                base: agg.clone(),
                field: FieldDescriptor {
                    name: "ob_header".into(),
                    owner_root: Some("W_FloatObject".into()),
                    owner_id: None,
                    base_is_deref: None,
                    taken_by_address: false,
                    inline_vec: false,
                    vec_part: None,
                    owner_declared_gc: None,
                    host_index: None,
                    scalar_word: None,
                },
                value: LinkArg::Value(header),
                ty: ValueType::Ref(None),
            },
            false,
        );
        graph.push_op_var(
            entry,
            OpKind::FieldWrite {
                base: agg.clone(),
                field: FieldDescriptor {
                    name: "floatval".into(),
                    owner_root: Some("W_FloatObject".into()),
                    owner_id: None,
                    base_is_deref: None,
                    taken_by_address: false,
                    inline_vec: false,
                    vec_part: None,
                    owner_declared_gc: None,
                    host_index: None,
                    scalar_word: None,
                },
                value: LinkArg::Value(v),
                ty: ValueType::Ref(None),
            },
            false,
        );
        let ret = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec!["boxed".into(), "Box".into(), "new".into()],
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![agg.clone()]),
                    result_ty: ValueType::Ref(Some("W_FloatObject".into())),
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(ret));

        assert_eq!(
            fuse_boxing_alloc(&mut graph, &numeric_boxing_attrs()),
            0,
            "Box::new is not the uninit allocation"
        );
        assert!(
            graph.block(entry).operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().map(String::as_str) == Some("new")
            )),
            "the look-alike Box::new call must remain residual"
        );
    }

    #[test]
    fn fuse_boxing_alloc_lowers_complex_box_two_payloads() {
        // A two-payload boxing struct (W_ComplexObject: real, imag) fuses to a
        // single NewWithVtable followed by two payload setfields in struct
        // order, both retyped to Float.
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let re = graph
            .push_op_var(entry, OpKind::ConstFloat(0.0f64.to_bits()), true)
            .unwrap();
        let im = graph
            .push_op_var(entry, OpKind::ConstFloat(0.0f64.to_bits()), true)
            .unwrap();
        // Resolvable `ob_header` header stores so `fuse_boxing_alloc` captures
        // a non-zero vtable; a cluster whose `ob_type` store resolves to no
        // constant is left unfused (residual `malloc_typed`).
        let header = push_boxing_header(&mut graph, entry, 4357049520);
        let agg = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor("W_ComplexObject"),
                    args: crate::model::call_args(vec![]),
                    result_ty: ValueType::Ref(Some("W_ComplexObject".into())),
                },
                true,
            )
            .unwrap();
        graph.push_op_var(
            entry,
            OpKind::FieldWrite {
                base: agg.clone(),
                field: FieldDescriptor {
                    name: "ob_header".into(),
                    owner_root: Some("W_ComplexObject".into()),
                    owner_id: None,
                    base_is_deref: None,
                    taken_by_address: false,
                    inline_vec: false,
                    vec_part: None,
                    owner_declared_gc: None,
                    host_index: None,
                    scalar_word: None,
                },
                value: LinkArg::Value(header),
                ty: ValueType::Ref(None),
            },
            false,
        );
        for (name, v) in [("real", re.clone()), ("imag", im.clone())] {
            graph.push_op_var(
                entry,
                OpKind::FieldWrite {
                    base: agg.clone(),
                    field: FieldDescriptor {
                        name: name.into(),
                        owner_root: Some("W_ComplexObject".into()),
                        owner_id: None,
                        base_is_deref: None,
                        taken_by_address: false,
                        inline_vec: false,
                        vec_part: None,
                        owner_declared_gc: None,
                        host_index: None,
                        scalar_word: None,
                    },
                    value: LinkArg::Value(v),
                    ty: ValueType::Ref(None),
                },
                false,
            );
        }
        let ret = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            crate::runtime_names::crates::OBJECT.into(),
                            "lltype".into(),
                            "malloc_typed".into(),
                        ],
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![agg.clone()]),
                    result_ty: ValueType::Ref(Some("W_ComplexObject".into())),
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(ret.clone()));

        let fused = fuse_boxing_alloc(&mut graph, &numeric_boxing_attrs());
        assert_eq!(fused, 1, "the complex boxing cluster must fuse");

        let ops = &graph.block(entry).operations;
        let nwv_pos = ops
            .iter()
            .position(|op| {
                matches!(&op.kind, OpKind::NewWithVtable { owner, .. } if owner == "W_ComplexObject")
            })
            .expect("NewWithVtable must be emitted");
        // Two payload stores follow, real then imag, both targeting %ret/Float.
        let mut names = Vec::new();
        for off in 1..=2 {
            match &ops[nwv_pos + off].kind {
                OpKind::FieldWrite {
                    base, field, ty, ..
                } => {
                    assert_eq!(base, &ret, "payload store must target the alloc result");
                    assert_eq!(*ty, ValueType::Float, "complex payloads are Float");
                    names.push(field.name.clone());
                }
                other => panic!("expected payload FieldWrite, got {other:?}"),
            }
        }
        assert_eq!(
            names,
            vec!["real".to_string(), "imag".to_string()],
            "payload stores must follow struct order"
        );
    }

    #[test]
    fn fuse_boxing_alloc_ignores_unknown_owner() {
        // A malloc_typed of a non-boxing aggregate (no `payload_for` entry)
        // is left untouched — the fusion is opt-in per recognised struct.
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let agg = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor("SomeOtherStruct"),
                    args: crate::model::call_args(vec![]),
                    result_ty: ValueType::Ref(Some("SomeOtherStruct".into())),
                },
                true,
            )
            .unwrap();
        let ret = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            crate::runtime_names::crates::OBJECT.into(),
                            "lltype".into(),
                            "malloc_typed".into(),
                        ],
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![agg.clone()]),
                    result_ty: ValueType::Ref(Some("SomeOtherStruct".into())),
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(ret));

        let fused = fuse_boxing_alloc(&mut graph, &numeric_boxing_attrs());
        assert_eq!(fused, 0, "unknown boxing owner must not fuse");
        assert!(
            !graph
                .block(entry)
                .operations
                .iter()
                .any(|op| matches!(&op.kind, OpKind::NewWithVtable { .. })),
            "no NewWithVtable may be emitted for an unrecognised owner"
        );
    }

    #[test]
    fn fuse_boxing_alloc_keeps_a_w_class_the_vtable_cannot_stand_for() {
        // Fusion lets the runtime stamp both `ob_type` and `w_class` off the
        // kept vtable, which answers for the cluster's own `w_class` only where
        // that is `get_instantiate(&T)` for the very `&T` its `ob_type` names.
        // Any other spelling is a class the vtable cannot re-synthesise:
        // `w_long_from_raw` (longobject.rs) pairs `&LONG_TYPE` with
        // `get_instantiate(&INT_TYPE)`, and `w_bytes_subclass_from_bytes`
        // (bytesobject.rs) stores the base `ob_type` beside a `w_class` read
        // back off the shadow stack, where re-synthesising would answer the
        // base type for `type(B(b'x'))`.  Both still lower — the store is kept
        // as its own `FieldWrite`, which is what `Transformer.rewrite_op_malloc` leaves
        // every field the allocation does not carry.
        type Var = crate::flowspace::model::Variable;
        const FLOAT_TYPE_ADDR: i64 = 4357049520;
        const OTHER_TYPE_ADDR: i64 = 4357049600;

        fn call(graph: &mut FunctionGraph, blk: BlockId, path: &[&str], args: Vec<Var>) -> Var {
            graph
                .push_op_var(
                    blk,
                    OpKind::Call {
                        target: CallTarget::FunctionPath {
                            segments: path.iter().map(|s| (*s).to_string()).collect(),
                            fun_decl_id: None,
                        },
                        args: crate::model::call_args(args),
                        result_ty: ValueType::Ref(Some("object".into())),
                    },
                    true,
                )
                .unwrap()
        }
        fn instantiate_of(graph: &mut FunctionGraph, blk: BlockId, addr: i64) -> Var {
            let arg = graph
                .push_op_var(blk, OpKind::ConstRefAddr(addr), true)
                .unwrap();
            call(
                graph,
                blk,
                &["runtime_object", "object_model", "get_instantiate"],
                vec![arg],
            )
        }
        /// A one-payload `W_FloatObject` cluster storing `ob_type:
        /// &FLOAT_TYPE` beside the `w_class` `feed` builds.
        fn cluster(feed: &dyn Fn(&mut FunctionGraph, BlockId) -> Var) -> FunctionGraph {
            let field = |base: &Var, name: &str, owner: &str, value: &Var| OpKind::FieldWrite {
                base: base.clone(),
                field: FieldDescriptor {
                    name: name.into(),
                    owner_root: Some(owner.into()),
                    owner_id: None,
                    base_is_deref: None,
                    taken_by_address: false,
                    inline_vec: false,
                    vec_part: None,
                    owner_declared_gc: None,
                    host_index: None,
                    scalar_word: None,
                },
                value: LinkArg::Value(value.clone()),
                ty: ValueType::Ref(None),
            };
            let mut graph = FunctionGraph::new("test");
            let entry = graph.startblock;
            let payload = graph
                .push_op_var(entry, OpKind::ConstFloat(0.0f64.to_bits()), true)
                .unwrap();
            let ob_type = graph
                .push_op_var(entry, OpKind::ConstRefAddr(FLOAT_TYPE_ADDR), true)
                .unwrap();
            let w_class = feed(&mut graph, entry);
            let header = graph
                .push_op_var(
                    entry,
                    OpKind::Call {
                        target: CallTarget::synthetic_transparent_ctor("PyObject"),
                        args: crate::model::call_args(vec![]),
                        result_ty: ValueType::Ref(Some("PyObject".into())),
                    },
                    true,
                )
                .unwrap();
            graph.push_op_var(
                entry,
                field(&header, "ob_type", "PyObject", &ob_type),
                false,
            );
            graph.push_op_var(
                entry,
                field(&header, "w_class", "PyObject", &w_class),
                false,
            );
            let agg = graph
                .push_op_var(
                    entry,
                    OpKind::Call {
                        target: CallTarget::synthetic_transparent_ctor("W_FloatObject"),
                        args: crate::model::call_args(vec![]),
                        result_ty: ValueType::Ref(Some("W_FloatObject".into())),
                    },
                    true,
                )
                .unwrap();
            graph.push_op_var(
                entry,
                field(&agg, "ob_header", "W_FloatObject", &header),
                false,
            );
            graph.push_op_var(
                entry,
                field(&agg, "floatval", "W_FloatObject", &payload),
                false,
            );
            let ret = call(
                &mut graph,
                entry,
                &[
                    crate::runtime_names::crates::OBJECT,
                    "lltype",
                    "malloc_typed",
                ],
                vec![agg],
            );
            graph.set_return(entry, Some(ret));
            graph
        }

        // The first row is the control: the same cluster, differing only in the
        // `w_class` feed, fuses with the store folded away — so a kept store
        // below is attributable to the feed and not to some other gate.
        let same = |graph: &mut FunctionGraph, blk| instantiate_of(graph, blk, FLOAT_TYPE_ADDR);
        let other = |graph: &mut FunctionGraph, blk| instantiate_of(graph, blk, OTHER_TYPE_ADDR);
        let shadow_stack = |graph: &mut FunctionGraph, blk| {
            let base = graph.push_op_var(blk, OpKind::ConstInt(0), true).unwrap();
            call(
                graph,
                blk,
                &[
                    crate::runtime_names::crates::OBJECT,
                    "gc_roots",
                    "shadow_stack_get",
                ],
                vec![base],
            )
        };
        // Every row fuses; they differ in whether the `w_class` store survives
        // the rewrite as a `FieldWrite` of its own.
        let rows: [(&str, &dyn Fn(&mut FunctionGraph, BlockId) -> Var, bool); 3] = [
            ("get_instantiate of the type ob_type names", &same, false),
            ("get_instantiate of another type", &other, true),
            ("a w_class read off the shadow stack", &shadow_stack, true),
        ];
        for (feed, build, keeps_store) in rows {
            let mut graph = cluster(build);
            let entry = graph.startblock;
            assert_eq!(
                fuse_boxing_alloc(&mut graph, &numeric_boxing_attrs()),
                1,
                "{feed}: wrong number of fused clusters"
            );
            let residual = graph.block(entry).operations.iter().any(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                        if segments.last().map(String::as_str) == Some("malloc_typed")
                )
            });
            assert!(!residual, "{feed}: the fused cluster leaves no residual");
            // The kept store must land on the NEW object, not on the dropped
            // header aggregate, and carry the very value the cluster stored.
            let result = match &graph.block(entry).operations.iter().find_map(|op| {
                matches!(op.kind, OpKind::NewWithVtable { .. }).then(|| op.result.clone())
            }) {
                Some(Some(result)) => result.clone(),
                _ => panic!("{feed}: no NewWithVtable was emitted"),
            };
            let kept: Vec<_> = graph
                .block(entry)
                .operations
                .iter()
                .filter_map(|op| match &op.kind {
                    OpKind::FieldWrite {
                        base, field, value, ..
                    } if base == &result && field.name == "w_class" => Some(value.clone()),
                    _ => None,
                })
                .collect();
            assert_eq!(
                kept.len(),
                usize::from(keeps_store),
                "{feed}: wrong number of w_class stores on the new object"
            );
            if keeps_store {
                // Read the expectation off the cluster's own store, which is
                // still in the block — `prune_dead_boxing_remnants` is what
                // drops it, and that runs separately.  Excluding the new
                // object's base keeps this from matching the kept store and
                // comparing it against itself.
                let expected = graph
                    .block(entry)
                    .operations
                    .iter()
                    .find_map(|op| match &op.kind {
                        OpKind::FieldWrite {
                            base, field, value, ..
                        } if base != &result && field.name == "w_class" => Some(value.clone()),
                        _ => None,
                    })
                    .expect("the cluster stored a w_class");
                assert_eq!(kept[0], expected, "{feed}: the kept store lost its value");
            }
        }
    }

    #[test]
    fn fuse_boxing_alloc_resolves_an_aggregate_that_crosses_a_link() {
        // The *aggregate* crosses the boundary, not only its header.
        // `w_float_new` builds the whole `W_FloatObject` up front and then
        // branches: `gc_interp::enabled()` and `try_gc_alloc_stable_raw` each
        // end a block, so the ctor stays in a dominator while the malloc sees
        // a `Block.inputargs` phi. Owner, payload stores and `ob_header` are
        // all looked up by exact `%agg`, so without resolving the phi first
        // the cluster declines for want of a ctor -- which is why the
        // canonical float constructor never fused in production.
        //
        // The other arm stores the same aggregate through `core::ptr::write`.
        // Fusing the malloc must not damage it: the write has to keep reading
        // a fully constructed object. (Sinking that materialization to the
        // write is an optimisation `sink_fused_boxing_aggregates_at_raw_writes`
        // performs when it can match the variable; it is not required for
        // correctness, so this asserts the object, not the sinking.)
        type Var = crate::flowspace::model::Variable;
        const FLOAT_TYPE_ADDR: i64 = 4357049520;

        fn call(graph: &mut FunctionGraph, blk: BlockId, path: &[&str], args: Vec<Var>) -> Var {
            graph
                .push_op_var(
                    blk,
                    OpKind::Call {
                        target: CallTarget::FunctionPath {
                            segments: path.iter().map(|s| (*s).to_string()).collect(),
                            fun_decl_id: None,
                        },
                        args: crate::model::call_args(args),
                        result_ty: ValueType::Ref(Some("object".into())),
                    },
                    true,
                )
                .unwrap()
        }
        fn field(base: &Var, name: &str, owner: &str, value: &Var) -> OpKind {
            OpKind::FieldWrite {
                base: base.clone(),
                field: FieldDescriptor {
                    name: name.into(),
                    owner_root: Some(owner.into()),
                    owner_id: None,
                    base_is_deref: None,
                    taken_by_address: false,
                    inline_vec: false,
                    vec_part: None,
                    owner_declared_gc: None,
                    host_index: None,
                    scalar_word: None,
                },
                value: LinkArg::Value(value.clone()),
                ty: ValueType::Ref(None),
            }
        }
        fn ctor(graph: &mut FunctionGraph, blk: BlockId, name: &str) -> Var {
            graph
                .push_op_var(
                    blk,
                    OpKind::Call {
                        target: CallTarget::synthetic_transparent_ctor(name),
                        args: crate::model::call_args(vec![]),
                        result_ty: ValueType::Ref(Some(name.into())),
                    },
                    true,
                )
                .unwrap()
        }

        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;

        // The whole cluster, built in the dominator.
        let cast = |graph: &mut FunctionGraph| {
            let ty = graph
                .push_op_var(entry, OpKind::ConstRefAddr(FLOAT_TYPE_ADDR), true)
                .unwrap();
            graph
                .push_op_var(entry, cast_instance_call("PyType", ty), true)
                .unwrap()
        };
        let ob_type = cast(&mut graph);
        let w_class_cast = cast(&mut graph);
        let w_class = call(
            &mut graph,
            entry,
            &["runtime_object", "object_model", "get_instantiate"],
            vec![w_class_cast],
        );
        let header = ctor(&mut graph, entry, "PyObject");
        graph.push_op_var(
            entry,
            field(&header, "ob_type", "PyObject", &ob_type),
            false,
        );
        graph.push_op_var(
            entry,
            field(&header, "w_class", "PyObject", &w_class),
            false,
        );
        let payload = graph
            .push_op_var(entry, OpKind::ConstFloat(0.0f64.to_bits()), true)
            .unwrap();
        let agg = ctor(&mut graph, entry, "W_FloatObject");
        graph.push_op_var(
            entry,
            field(&agg, "ob_header", "W_FloatObject", &header),
            false,
        );
        graph.push_op_var(
            entry,
            field(&agg, "floatval", "W_FloatObject", &payload),
            false,
        );

        // `gc_interp::enabled()` ends the dominator; the aggregate crosses.
        let (probe, probe_args) = graph.create_block_with_arg_vars(1);
        graph.set_goto(entry, probe, vec![agg.clone()]);

        // `try_gc_alloc_stable_raw` ends the next block, then the branch.
        let raw = call(
            &mut graph,
            probe,
            &[
                crate::runtime_names::crates::OBJECT,
                "gc_hook",
                "try_gc_alloc_stable_raw",
            ],
            vec![],
        );
        let (gc_arm, gc_args) = graph.create_block_with_arg_vars(2);
        let (plain_arm, plain_args) = graph.create_block_with_arg_vars(1);
        graph.block_mut(probe).exitswitch = Some(ExitSwitch::Value(raw.clone()));
        graph.closeblock(
            probe,
            vec![
                Link::from_variables(
                    &graph,
                    vec![probe_args[0].clone(), raw.clone()],
                    gc_arm,
                    Some(ExitCase::Bool(true)),
                ),
                Link::from_variables(
                    &graph,
                    vec![probe_args[0].clone()],
                    plain_arm,
                    Some(ExitCase::Bool(false)),
                ),
            ],
        );

        // GC arm: write the aggregate into the raw allocation.
        graph.push_op_var(
            gc_arm,
            OpKind::Call {
                target: CallTarget::FunctionPath {
                    segments: ["core", "ptr", "write"]
                        .iter()
                        .map(|s| s.to_string())
                        .collect(),
                    fun_decl_id: None,
                },
                args: crate::model::call_args(vec![gc_args[1].clone(), gc_args[0].clone()]),
                result_ty: ValueType::Void,
            },
            false,
        );
        graph.set_return(gc_arm, Some(gc_args[1].clone()));

        // Ordinary arm: the `malloc_typed` the fusion rewrites.
        let ret = call(
            &mut graph,
            plain_arm,
            &[
                crate::runtime_names::crates::OBJECT,
                "lltype",
                "malloc_typed",
            ],
            vec![plain_args[0].clone()],
        );
        graph.set_return(plain_arm, Some(ret));

        assert_eq!(
            fuse_boxing_alloc(&mut graph, &numeric_boxing_attrs()),
            1,
            "the aggregate's phi must resolve to its ctor"
        );

        let vtables: Vec<i64> = graph
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .filter_map(|op| match &op.kind {
                OpKind::NewWithVtable { vtable, .. } => Some(*vtable),
                _ => None,
            })
            .collect();
        assert_eq!(vtables, vec![FLOAT_TYPE_ADDR], "wrong vtable stamped");
        assert!(
            !graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                        if segments.last().map(String::as_str) == Some("malloc_typed")
                )
            }),
            "the fused cluster must leave no residual malloc_typed"
        );

        // The GC arm must still store a fully constructed object: whatever the
        // write reads has to reach a `W_FloatObject` ctor carrying its header.
        let written = graph
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .find_map(|op| match &op.kind {
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments, .. },
                    args,
                    ..
                } if segments
                    .iter()
                    .map(String::as_str)
                    .eq(["core", "ptr", "write"]) =>
                {
                    Some(args[1].clone().into_variable())
                }
                _ => None,
            })
            .expect("the raw write must survive the rewrite");
        // Follow the links the way the pass does, then confirm the ctor and
        // its header store are both still present for that root.
        let mut roots: Vec<Var> = Vec::new();
        let mut frontier = vec![written];
        while let Some(var) = frontier.pop() {
            if graph
                .blocks
                .iter()
                .flat_map(|b| &b.operations)
                .any(|o| o.result.as_ref() == Some(&var))
            {
                roots.push(var);
                continue;
            }
            let Some((target, slot)) = graph.blocks.iter().find_map(|b| {
                b.inputargs
                    .iter()
                    .position(|arg| arg == &var)
                    .map(|slot| (b.id, slot))
            }) else {
                panic!("the written value is neither an op result nor an inputarg");
            };
            for b in &graph.blocks {
                for link in &b.exits {
                    if link.target == target {
                        frontier
                            .extend(link.args.get(slot).and_then(LinkArg::as_variable).cloned());
                    }
                }
            }
        }
        assert!(!roots.is_empty(), "the written value resolves to nothing");
        for root in &roots {
            assert!(
                graph.blocks.iter().flat_map(|b| &b.operations).any(|o| {
                    o.result.as_ref() == Some(root)
                        && matches!(
                            &o.kind,
                            OpKind::Call { target: CallTarget::SyntheticTransparentCtor { name, .. }, .. }
                                if name == "W_FloatObject"
                        )
                }),
                "the raw write must still read a W_FloatObject ctor"
            );
            assert!(
                graph.blocks.iter().flat_map(|b| &b.operations).any(|o| {
                    matches!(&o.kind, OpKind::FieldWrite { base, field, .. }
                        if base == root && field.name.as_str() == "ob_header")
                }),
                "the written aggregate must keep its ob_header store"
            );
        }
    }

    #[test]
    fn fuse_boxing_alloc_declines_a_field_written_twice_over() {
        // A cluster built before a branch may be rewritten on one arm. Without
        // reaching definitions, conflicting payload or header stores must
        // decline; redundant stores of the same value may still fuse.
        type Var = crate::flowspace::model::Variable;
        const FLOAT_TYPE_ADDR: i64 = 4357049520;
        const OTHER_TYPE_ADDR: i64 = 4357049600;

        fn call(graph: &mut FunctionGraph, blk: BlockId, path: &[&str], args: Vec<Var>) -> Var {
            graph
                .push_op_var(
                    blk,
                    OpKind::Call {
                        target: CallTarget::FunctionPath {
                            segments: path.iter().map(|s| (*s).to_string()).collect(),
                            fun_decl_id: None,
                        },
                        args: crate::model::call_args(args),
                        result_ty: ValueType::Ref(Some("object".into())),
                    },
                    true,
                )
                .unwrap()
        }
        fn field(base: &Var, name: &str, owner: &str, value: &Var) -> OpKind {
            OpKind::FieldWrite {
                base: base.clone(),
                field: FieldDescriptor {
                    name: name.into(),
                    owner_root: Some(owner.into()),
                    owner_id: None,
                    base_is_deref: None,
                    taken_by_address: false,
                    inline_vec: false,
                    vec_part: None,
                    owner_declared_gc: None,
                    host_index: None,
                    scalar_word: None,
                },
                value: LinkArg::Value(value.clone()),
                ty: ValueType::Ref(None),
            }
        }
        fn ctor(graph: &mut FunctionGraph, blk: BlockId, name: &str) -> Var {
            graph
                .push_op_var(
                    blk,
                    OpKind::Call {
                        target: CallTarget::synthetic_transparent_ctor(name),
                        args: crate::model::call_args(vec![]),
                        result_ty: ValueType::Ref(Some(name.into())),
                    },
                    true,
                )
                .unwrap()
        }

        /// A `W_FloatObject` cluster built in the dominator, malloc'd behind a
        /// branch, with `rewrite` free to add a second store on one arm.
        fn cluster(
            rewrite: &dyn Fn(&mut FunctionGraph, BlockId, &Var, &Var),
        ) -> (FunctionGraph, BlockId) {
            let mut graph = FunctionGraph::new("test");
            let entry = graph.startblock;
            let cast = |graph: &mut FunctionGraph, addr: i64| {
                let ty = graph
                    .push_op_var(entry, OpKind::ConstRefAddr(addr), true)
                    .unwrap();
                graph
                    .push_op_var(entry, cast_instance_call("PyType", ty), true)
                    .unwrap()
            };
            let ob_type = cast(&mut graph, FLOAT_TYPE_ADDR);
            let w_class_cast = cast(&mut graph, FLOAT_TYPE_ADDR);
            let w_class = call(
                &mut graph,
                entry,
                &["runtime_object", "object_model", "get_instantiate"],
                vec![w_class_cast],
            );
            let header = ctor(&mut graph, entry, "PyObject");
            graph.push_op_var(
                entry,
                field(&header, "ob_type", "PyObject", &ob_type),
                false,
            );
            graph.push_op_var(
                entry,
                field(&header, "w_class", "PyObject", &w_class),
                false,
            );
            let payload = graph
                .push_op_var(entry, OpKind::ConstFloat(0.0f64.to_bits()), true)
                .unwrap();
            let agg = ctor(&mut graph, entry, "W_FloatObject");
            graph.push_op_var(
                entry,
                field(&agg, "ob_header", "W_FloatObject", &header),
                false,
            );
            graph.push_op_var(
                entry,
                field(&agg, "floatval", "W_FloatObject", &payload),
                false,
            );

            // Both arms carry one aggregate to the join through a phi.
            let cond = call(
                &mut graph,
                entry,
                &[crate::runtime_names::crates::OBJECT, "gc_interp", "enabled"],
                vec![],
            );
            let (arm, arm_args) = graph.create_block_with_arg_vars(1);
            let (skip, skip_args) = graph.create_block_with_arg_vars(1);
            graph.block_mut(entry).exitswitch = Some(ExitSwitch::Value(cond.clone()));
            graph.closeblock(
                entry,
                vec![
                    Link::from_variables(
                        &graph,
                        vec![agg.clone()],
                        arm,
                        Some(ExitCase::Bool(true)),
                    ),
                    Link::from_variables(
                        &graph,
                        vec![agg.clone()],
                        skip,
                        Some(ExitCase::Bool(false)),
                    ),
                ],
            );
            rewrite(&mut graph, arm, &agg, &header);

            let (join, join_args) = graph.create_block_with_arg_vars(1);
            graph.set_goto(arm, join, vec![arm_args[0].clone()]);
            graph.set_goto(skip, join, vec![skip_args[0].clone()]);
            let ret = call(
                &mut graph,
                join,
                &[
                    crate::runtime_names::crates::OBJECT,
                    "lltype",
                    "malloc_typed",
                ],
                vec![join_args[0].clone()],
            );
            graph.set_return(join, Some(ret));
            (graph, join)
        }

        let untouched = |_: &mut FunctionGraph, _: BlockId, _: &Var, _: &Var| {};
        let payload_again = |graph: &mut FunctionGraph, blk: BlockId, agg: &Var, _: &Var| {
            let other = graph
                .push_op_var(blk, OpKind::ConstFloat(1.0f64.to_bits()), true)
                .unwrap();
            graph.push_op_var(blk, field(agg, "floatval", "W_FloatObject", &other), false);
        };
        let payload_same = |graph: &mut FunctionGraph, blk: BlockId, agg: &Var, _: &Var| {
            // Repeating the same `LinkArg` is redundant, not ambiguous.
            let same = graph
                .blocks
                .iter()
                .flat_map(|b| &b.operations)
                .find_map(|o| match &o.kind {
                    OpKind::FieldWrite {
                        base, field, value, ..
                    } if base == agg && field.name.as_str() == "floatval" => {
                        value.as_variable().cloned()
                    }
                    _ => None,
                })
                .expect("the dominator stores floatval");
            graph.push_op_var(blk, field(agg, "floatval", "W_FloatObject", &same), false);
        };
        let header_again = |graph: &mut FunctionGraph, blk: BlockId, agg: &Var, _: &Var| {
            let other_header = ctor(graph, blk, "PyObject");
            graph.push_op_var(
                blk,
                field(agg, "ob_header", "W_FloatObject", &other_header),
                false,
            );
        };
        let obtype_again = |graph: &mut FunctionGraph, blk: BlockId, _: &Var, header: &Var| {
            let other = graph
                .push_op_var(blk, OpKind::ConstRefAddr(OTHER_TYPE_ADDR), true)
                .unwrap();
            let cast = graph
                .push_op_var(blk, cast_instance_call("PyType", other), true)
                .unwrap();
            graph.push_op_var(blk, field(header, "ob_type", "PyObject", &cast), false);
        };

        let rows: [(
            &str,
            &dyn Fn(&mut FunctionGraph, BlockId, &Var, &Var),
            usize,
        ); 5] = [
            ("no second store", &untouched, 1),
            ("floatval rewritten on one arm", &payload_again, 0),
            ("floatval restored to the same value", &payload_same, 1),
            ("ob_header rewritten on one arm", &header_again, 0),
            ("ob_type rewritten on one arm", &obtype_again, 0),
        ];
        for (row, rewrite, expected) in rows {
            let (mut graph, join) = cluster(rewrite);
            assert_eq!(
                fuse_boxing_alloc(&mut graph, &numeric_boxing_attrs()),
                expected,
                "{row}: wrong number of fused clusters"
            );
            let residual = graph.block(join).operations.iter().any(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                        if segments.last().map(String::as_str) == Some("malloc_typed")
                )
            });
            assert_eq!(
                residual,
                expected == 0,
                "{row}: malloc_typed residual must survive exactly when the cluster declines"
            );
        }

        // A single graph-wide store is still insufficient when it lives on
        // only one branch. Move the otherwise-valid type-word write out of the
        // dominator and onto the true arm; the false arm reaches the same
        // malloc with the shared header uninitialized.
        let (mut graph, join) = cluster(&untouched);
        let entry = graph.startblock;
        let store_pos = graph
            .block(entry)
            .operations
            .iter()
            .position(|op| {
                matches!(&op.kind, OpKind::FieldWrite { field, .. }
                    if field.name.as_str() == "ob_type")
            })
            .expect("fixture has a dominating ob_type store");
        let branch_only_store = graph.block_mut(entry).operations.remove(store_pos);
        let true_arm = graph
            .block(entry)
            .exits
            .iter()
            .find(|link| link.exitcase == Some(ExitCase::Bool(true)))
            .expect("fixture has a true arm")
            .target;
        graph.push_op_var(true_arm, branch_only_store.kind, false);

        assert_eq!(
            fuse_boxing_alloc(&mut graph, &numeric_boxing_attrs()),
            0,
            "an ob_type store on only one arm must not initialize the joined malloc"
        );
        assert!(graph.block(join).operations.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().map(String::as_str) == Some("malloc_typed")
            )
        }));
    }

    #[test]
    fn fuse_boxing_alloc_resolves_a_header_that_crosses_a_link() {
        // The header reaches malloc as a phi while its stores remain on the
        // producer variable. Resolve that producer first, and require all
        // merged roots to name one vtable.
        type Var = crate::flowspace::model::Variable;
        const FLOAT_TYPE_ADDR: i64 = 4357049520;
        const OTHER_TYPE_ADDR: i64 = 4357049600;

        fn call(graph: &mut FunctionGraph, blk: BlockId, path: &[&str], args: Vec<Var>) -> Var {
            graph
                .push_op_var(
                    blk,
                    OpKind::Call {
                        target: CallTarget::FunctionPath {
                            segments: path.iter().map(|s| (*s).to_string()).collect(),
                            fun_decl_id: None,
                        },
                        args: crate::model::call_args(args),
                        result_ty: ValueType::Ref(Some("object".into())),
                    },
                    true,
                )
                .unwrap()
        }
        fn field(base: &Var, name: &str, owner: &str, value: &Var) -> OpKind {
            OpKind::FieldWrite {
                base: base.clone(),
                field: FieldDescriptor {
                    name: name.into(),
                    owner_root: Some(owner.into()),
                    owner_id: None,
                    base_is_deref: None,
                    taken_by_address: false,
                    inline_vec: false,
                    vec_part: None,
                    owner_declared_gc: None,
                    host_index: None,
                    scalar_word: None,
                },
                value: LinkArg::Value(value.clone()),
                ty: ValueType::Ref(None),
            }
        }
        fn ctor(graph: &mut FunctionGraph, blk: BlockId, name: &str) -> Var {
            graph
                .push_op_var(
                    blk,
                    OpKind::Call {
                        target: CallTarget::synthetic_transparent_ctor(name),
                        args: crate::model::call_args(vec![]),
                        result_ty: ValueType::Ref(Some(name.into())),
                    },
                    true,
                )
                .unwrap()
        }
        /// The nested `PyObject` header, built entirely in `blk`.
        fn header_in(graph: &mut FunctionGraph, blk: BlockId, addr: i64) -> Var {
            let cast = |graph: &mut FunctionGraph| {
                let ty = graph
                    .push_op_var(blk, OpKind::ConstRefAddr(addr), true)
                    .unwrap();
                graph
                    .push_op_var(blk, cast_instance_call("PyType", ty), true)
                    .unwrap()
            };
            let ob_type = cast(graph);
            let w_class_cast = cast(graph);
            let w_class = call(
                graph,
                blk,
                &["runtime_object", "object_model", "get_instantiate"],
                vec![w_class_cast],
            );
            let header = ctor(graph, blk, "PyObject");
            graph.push_op_var(blk, field(&header, "ob_type", "PyObject", &ob_type), false);
            graph.push_op_var(blk, field(&header, "w_class", "PyObject", &w_class), false);
            header
        }
        /// The outer object and its `malloc_typed`, taking `header` from
        /// wherever it was built.
        fn outer_in(graph: &mut FunctionGraph, blk: BlockId, header: &Var) {
            let payload = graph
                .push_op_var(blk, OpKind::ConstFloat(0.0f64.to_bits()), true)
                .unwrap();
            let agg = ctor(graph, blk, "W_FloatObject");
            graph.push_op_var(
                blk,
                field(&agg, "ob_header", "W_FloatObject", header),
                false,
            );
            graph.push_op_var(
                blk,
                field(&agg, "floatval", "W_FloatObject", &payload),
                false,
            );
            let ret = call(
                graph,
                blk,
                &[
                    crate::runtime_names::crates::OBJECT,
                    "lltype",
                    "malloc_typed",
                ],
                vec![agg],
            );
            graph.set_return(blk, Some(ret));
        }

        /// The header built once, then relayed across `crossings` blocks
        /// before the object that stores it — a run of calls before the
        /// allocation leaves exactly this.
        fn chain(crossings: usize) -> FunctionGraph {
            let mut graph = FunctionGraph::new("test");
            let entry = graph.startblock;
            let mut carried = vec![header_in(&mut graph, entry, FLOAT_TYPE_ADDR)];
            let mut from = entry;
            for _ in 0..crossings {
                let (next, args) = graph.create_block_with_arg_vars(1);
                graph.set_goto(from, next, carried);
                carried = args;
                from = next;
            }
            outer_in(&mut graph, from, &carried[0]);
            graph
        }
        /// Two predecessors each building their own header, and the object
        /// storing whichever one the merge carried.
        fn merge(left_addr: i64, right_addr: i64) -> FunctionGraph {
            let mut graph = FunctionGraph::new("test");
            let entry = graph.startblock;
            let cond = graph.push_op_var(entry, OpKind::ConstInt(0), true).unwrap();
            let (join, carried) = graph.create_block_with_arg_vars(1);
            let arms: Vec<Link> = [(true, left_addr), (false, right_addr)]
                .into_iter()
                .map(|(case, addr)| {
                    let arm = graph.create_block();
                    let header = header_in(&mut graph, arm, addr);
                    graph.set_goto(arm, join, vec![header]);
                    Link::from_variables(&graph, vec![], arm, Some(ExitCase::Bool(case)))
                })
                .collect();
            graph.block_mut(entry).exitswitch = Some(ExitSwitch::Value(cond));
            graph.closeblock(entry, arms);
            outer_in(&mut graph, join, &carried[0]);
            graph
        }

        let rows: [(&str, &dyn Fn() -> FunctionGraph, usize); 4] = [
            ("header crosses one link", &|| chain(1), 1),
            ("header crosses two links", &|| chain(2), 1),
            (
                "merged headers naming one type",
                &|| merge(FLOAT_TYPE_ADDR, FLOAT_TYPE_ADDR),
                1,
            ),
            (
                "merged headers naming two types",
                &|| merge(FLOAT_TYPE_ADDR, OTHER_TYPE_ADDR),
                0,
            ),
        ];
        for (shape, build, expected) in rows {
            let mut graph = build();
            assert_eq!(
                fuse_boxing_alloc(&mut graph, &numeric_boxing_attrs()),
                expected,
                "{shape}: wrong number of fused clusters"
            );
            let vtables: Vec<i64> = graph
                .blocks
                .iter()
                .flat_map(|b| &b.operations)
                .filter_map(|op| match &op.kind {
                    OpKind::NewWithVtable { vtable, .. } => Some(*vtable),
                    _ => None,
                })
                .collect();
            // Naming the address separates a walk that reached the header's
            // own roots from one that read some other constant in the graph.
            let expected_vtables = if expected == 0 {
                Vec::new()
            } else {
                vec![FLOAT_TYPE_ADDR]
            };
            assert_eq!(vtables, expected_vtables, "{shape}: wrong vtable stamped");
            let residual = graph.blocks.iter().flat_map(|b| &b.operations).any(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                        if segments.last().map(String::as_str) == Some("malloc_typed")
                )
            });
            assert_eq!(
                residual,
                expected == 0,
                "{shape}: malloc_typed residual must survive exactly when the cluster declines"
            );
        }
    }

    #[test]
    fn fuse_boxing_alloc_sweeps_nested_header_chain() {
        // Faithful `w_float_new` shape: the boxing struct's header is a nested
        // `PyObject` ctor whose `ob_type` is a `__cast_instance_intrinsic` pointer
        // reinterpret of the `&FLOAT_TYPE` constant and whose `w_class` is a
        // `get_instantiate` read of that same constant.  After fusion drops
        // the `ob_header` (it rides the `NewWithVtable` descriptor), the
        // entire header sub-tree — the inner `PyObject` ctor, the outer
        // `W_FloatObject` ctor, the two `__cast_instance_intrinsic` casts and the
        // `get_instantiate` — is dead and must be swept, leaving only the
        // `NewWithVtable`, its `floatval` payload store, and the return cast.
        // (The two `ConstRefAddr` constants legitimately survive as dead
        // constants; the dual-gate seeds them from the constant table, so they
        // do not diverge.)
        type Var = crate::flowspace::model::Variable;
        let cast_instance = |to: &str, arg: &Var| crate::model::cast_instance_call(to, arg.clone());
        let field = |base: &Var, name: &str, owner: &str, value: &Var| OpKind::FieldWrite {
            base: base.clone(),
            field: FieldDescriptor {
                name: name.into(),
                owner_root: Some(owner.into()),
                owner_id: None,
                base_is_deref: None,
                taken_by_address: false,
                inline_vec: false,
                vec_part: None,
                owner_declared_gc: None,
                host_index: None,
                scalar_word: None,
            },
            value: LinkArg::Value(value.clone()),
            ty: ValueType::Ref(None),
        };
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let value = graph
            .push_op_var(entry, OpKind::ConstFloat(0.0f64.to_bits()), true)
            .unwrap();
        let ty_addr1 = graph
            .push_op_var(entry, OpKind::ConstRefAddr(4357049520), true)
            .unwrap();
        let ob_type = graph
            .push_op_var(entry, cast_instance("PyType", &ty_addr1), true)
            .unwrap();
        let ty_addr2 = graph
            .push_op_var(entry, OpKind::ConstRefAddr(4357049520), true)
            .unwrap();
        // `w_class` is the type's `instantiate` slot rather than the type
        // pointer itself, so it reaches the header through a `get_instantiate`
        // read of the same `&FLOAT_TYPE` constant `ob_type` carries.
        let w_class_cast = graph
            .push_op_var(entry, cast_instance("PyType", &ty_addr2), true)
            .unwrap();
        let w_class = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            "runtime_object".into(),
                            "object_model".into(),
                            "get_instantiate".into(),
                        ],
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![w_class_cast]),
                    result_ty: ValueType::Ref(Some("object".into())),
                },
                true,
            )
            .unwrap();
        // Inner `PyObject` header ctor + its field stores.
        let header = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor("PyObject"),
                    args: crate::model::call_args(vec![]),
                    result_ty: ValueType::Ref(Some("PyObject".into())),
                },
                true,
            )
            .unwrap();
        graph.push_op_var(
            entry,
            field(&header, "ob_type", "PyObject", &ob_type),
            false,
        );
        graph.push_op_var(
            entry,
            field(&header, "w_class", "PyObject", &w_class),
            false,
        );
        // Outer `W_FloatObject` ctor + its `ob_header` / `floatval` stores.
        let agg = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor("W_FloatObject"),
                    args: crate::model::call_args(vec![]),
                    result_ty: ValueType::Ref(Some("W_FloatObject".into())),
                },
                true,
            )
            .unwrap();
        graph.push_op_var(
            entry,
            field(&agg, "ob_header", "W_FloatObject", &header),
            false,
        );
        graph.push_op_var(
            entry,
            field(&agg, "floatval", "W_FloatObject", &value),
            false,
        );
        let raw = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            crate::runtime_names::crates::OBJECT.into(),
                            "lltype".into(),
                            "malloc_typed".into(),
                        ],
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![agg.clone()]),
                    result_ty: ValueType::Ref(Some("W_FloatObject".into())),
                },
                true,
            )
            .unwrap();
        let ret = graph
            .push_op_var(entry, cast_instance("PyObject", &raw), true)
            .unwrap();
        graph.set_return(entry, Some(ret.clone()));

        let fused = fuse_boxing_alloc(&mut graph, &numeric_boxing_attrs());
        assert_eq!(fused, 1, "the nested-header boxing cluster must fuse");
        // Fusion orphans the header sub-tree but no longer sweeps it itself;
        // the remnant sweep is a separate pass (run right after fusion in
        // `simplify_lowered_graph`).
        prune_dead_boxing_remnants(&mut graph);

        let ops = &graph.block(entry).operations;
        // The dead header sub-tree is gone: no SyntheticTransparentCtor and no
        // `__cast_instance_intrinsic["PyType"]` cast survives.
        assert!(
            !ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::SyntheticTransparentCtor { .. },
                    ..
                }
            )),
            "every dead aggregate ctor must be swept: {:#?}",
            ops.iter().map(|o| &o.kind).collect::<Vec<_>>()
        );
        assert!(
            !ops.iter()
                .any(|op| cast_instance_root(&op.kind) == Some("PyType")),
            "the dead ob_type/w_class header casts must be swept"
        );
        assert!(
            !ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().map(String::as_str) == Some("get_instantiate")
            )),
            "the dead w_class instantiate read must be swept"
        );
        // The live spine survives: NewWithVtable + floatval store + return cast.
        // The vtable (type pointer) is captured from the dropped
        // `ob_header.ob_type = cast(ConstRefAddr(4357049520))` store so the
        // runtime can stamp `ob_type` / `w_class`.
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::NewWithVtable { owner, vtable }
                    if owner == "W_FloatObject" && *vtable == 4357049520
            )),
            "NewWithVtable must survive carrying the captured type pointer"
        );
        assert!(
            ops.iter()
                .any(|op| cast_instance_root(&op.kind) == Some("PyObject")),
            "the live return cast must survive"
        );
        let _ = ret;
    }

    #[test]
    fn fuse_boxing_alloc_lowers_non_numeric_struct_generically() {
        // A non-numeric boxing struct (`W_SetObject`: ob_header + items + len +
        // hash) fuses through the generic field-layout derivation, not the old
        // hardcoded numeric set.  The three non-header fields become payload
        // setfields in struct order, each re-typed from the registered layout;
        // `ob_header` is dropped (rides the `NewWithVtable` type pointer).
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let items = graph
            .push_op_var(entry, OpKind::ConstRefAddr(0), true)
            .unwrap();
        let len = graph.push_op_var(entry, OpKind::ConstInt(3), true).unwrap();
        let hash = graph
            .push_op_var(entry, OpKind::ConstInt(-1), true)
            .unwrap();
        // Resolvable `ob_header` header stores so the fusion fires.
        let header = push_boxing_header(&mut graph, entry, 4357049520);
        let agg = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor("W_SetObject"),
                    args: crate::model::call_args(vec![]),
                    result_ty: ValueType::Ref(Some("W_SetObject".into())),
                },
                true,
            )
            .unwrap();
        let field =
            |base: &crate::flowspace::model::Variable, name: &str, value, ty| OpKind::FieldWrite {
                base: base.clone(),
                field: FieldDescriptor {
                    name: name.into(),
                    owner_root: Some("W_SetObject".into()),
                    owner_id: None,
                    base_is_deref: None,
                    taken_by_address: false,
                    inline_vec: false,
                    vec_part: None,
                    owner_declared_gc: None,
                    host_index: None,
                    scalar_word: None,
                },
                value,
                ty,
            };
        graph.push_op_var(
            entry,
            field(
                &agg,
                "ob_header",
                LinkArg::Value(header),
                ValueType::Ref(None),
            ),
            false,
        );
        graph.push_op_var(
            entry,
            field(&agg, "items", LinkArg::Value(items), ValueType::Ref(None)),
            false,
        );
        graph.push_op_var(
            entry,
            field(&agg, "len", LinkArg::Value(len), ValueType::Ref(None)),
            false,
        );
        graph.push_op_var(
            entry,
            field(&agg, "hash", LinkArg::Value(hash), ValueType::Ref(None)),
            false,
        );
        let ret = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            crate::runtime_names::crates::OBJECT.into(),
                            "lltype".into(),
                            "malloc".into(),
                        ],
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![agg.clone()]),
                    result_ty: ValueType::Ref(Some("W_SetObject".into())),
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(ret.clone()));

        // Synthetic fixtures have no StructId registry, so use the ctor's
        // exact owner spelling. Production qualified aliases resolve by id.
        let attrs = std::collections::HashMap::from([
            (
                "PyObject".to_string(),
                vec![
                    ("ob_type".to_string(), ValueType::Ref(None)),
                    ("w_class".to_string(), ValueType::Ref(None)),
                ],
            ),
            (
                "W_SetObject".to_string(),
                vec![
                    ("ob_header".to_string(), ValueType::Ref(None)),
                    ("items".to_string(), ValueType::Ref(None)),
                    ("len".to_string(), ValueType::Int),
                    ("hash".to_string(), ValueType::Int),
                ],
            ),
        ]);
        let fused = fuse_boxing_alloc(&mut graph, &attrs);
        assert_eq!(fused, 1, "the non-numeric boxing cluster must fuse");

        let ops = &graph.block(entry).operations;
        let nwv_pos = ops
            .iter()
            .position(|op| {
                matches!(&op.kind, OpKind::NewWithVtable { owner, vtable }
                    if owner == "W_SetObject" && *vtable == 4357049520)
            })
            .expect("NewWithVtable must be emitted with the captured type pointer");
        assert_eq!(
            ops[nwv_pos].result.as_ref(),
            Some(&ret),
            "NewWithVtable must reuse the malloc result register"
        );
        // Three payload stores follow in struct order (ob_header dropped),
        // each re-typed from the registered layout.
        let mut got: Vec<(String, ValueType)> = Vec::new();
        for off in 1..=3 {
            match &ops[nwv_pos + off].kind {
                OpKind::FieldWrite {
                    base, field, ty, ..
                } => {
                    assert_eq!(base, &ret, "payload store must target the alloc result");
                    got.push((field.name.clone(), ty.clone()));
                }
                other => panic!("expected payload FieldWrite, got {other:?}"),
            }
        }
        assert_eq!(
            got,
            vec![
                ("items".to_string(), ValueType::Ref(None)),
                ("len".to_string(), ValueType::Int),
                ("hash".to_string(), ValueType::Int),
            ],
            "payloads must be the non-header fields in struct order, re-typed \
             from the registered layout"
        );
        assert!(
            !ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().map(String::as_str) == Some("malloc_typed")
            )),
            "no malloc_typed call may survive the fusion"
        );
    }

    #[test]
    fn fuse_boxing_alloc_skips_inlined_nested_struct_payload() {
        // `w_list_adopt_int_items` builds `W_ListObject` with a by-value
        // `int_items: IntArray` store. Copying that store onto the fused
        // `NewWithVtable` result would apply `IntArray.block`'s offset 0
        // to the list and overlay `ob_header`. Production
        // `simplify_lowered_graph_expanding` skips those fields; this
        // fixture is that skip.
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let allocated = graph.push_op_var(entry, OpKind::ConstInt(7), true).unwrap();
        let items = graph
            .push_op_var(entry, OpKind::ConstRefAddr(0), true)
            .unwrap();
        let strategy = graph.push_op_var(entry, OpKind::ConstInt(1), true).unwrap();
        let nested = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor("IntArray"),
                    args: crate::model::call_args(vec![]),
                    result_ty: ValueType::Ref(Some("IntArray".into())),
                },
                true,
            )
            .unwrap();
        let header = push_boxing_header(&mut graph, entry, 4357049520);
        let agg = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor("W_ListObject"),
                    args: crate::model::call_args(vec![]),
                    result_ty: ValueType::Ref(Some("W_ListObject".into())),
                },
                true,
            )
            .unwrap();
        let field =
            |base: &crate::flowspace::model::Variable, name: &str, value, ty| OpKind::FieldWrite {
                base: base.clone(),
                field: FieldDescriptor {
                    name: name.into(),
                    owner_root: Some("W_ListObject".into()),
                    owner_id: None,
                    base_is_deref: None,
                    taken_by_address: false,
                    inline_vec: false,
                    vec_part: None,
                    owner_declared_gc: None,
                    host_index: None,
                    scalar_word: None,
                },
                value,
                ty,
            };
        graph.push_op_var(
            entry,
            field(
                &agg,
                "ob_header",
                LinkArg::Value(header),
                ValueType::Ref(None),
            ),
            false,
        );
        graph.push_op_var(
            entry,
            field(&agg, "allocated", LinkArg::Value(allocated), ValueType::Int),
            false,
        );
        graph.push_op_var(
            entry,
            field(&agg, "items", LinkArg::Value(items), ValueType::Ref(None)),
            false,
        );
        graph.push_op_var(
            entry,
            field(&agg, "strategy", LinkArg::Value(strategy), ValueType::Int),
            false,
        );
        graph.push_op_var(
            entry,
            field(
                &agg,
                "int_items",
                LinkArg::Value(nested),
                ValueType::Ref(Some("IntArray".into())),
            ),
            false,
        );
        let ret = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            crate::runtime_names::crates::OBJECT.into(),
                            "lltype".into(),
                            "malloc_typed_managed".into(),
                        ],
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![agg.clone()]),
                    result_ty: ValueType::Ref(Some("W_ListObject".into())),
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(ret.clone()));

        let attrs = std::collections::HashMap::from([
            (
                "PyObject".to_string(),
                vec![
                    ("ob_type".to_string(), ValueType::Ref(None)),
                    ("w_class".to_string(), ValueType::Ref(None)),
                ],
            ),
            (
                "W_ListObject".to_string(),
                vec![
                    ("ob_header".to_string(), ValueType::Ref(None)),
                    ("allocated".to_string(), ValueType::Int),
                    ("items".to_string(), ValueType::Ref(None)),
                    ("strategy".to_string(), ValueType::Int),
                    (
                        "int_items".to_string(),
                        ValueType::Ref(Some("IntArray".into())),
                    ),
                ],
            ),
        ]);
        let expand = |owner: &str, field: &str| {
            (owner == "W_ListObject" && field == "int_items").then_some(vec![(
                FieldDescriptor {
                    name: "int_items.block".into(),
                    owner_root: Some("W_ListObject".into()),
                    owner_id: None,
                    base_is_deref: None,
                    taken_by_address: false,
                    inline_vec: false,
                    vec_part: None,
                    owner_declared_gc: None,
                    host_index: None,
                    scalar_word: None,
                },
                ValueType::Ref(None),
            )])
        };
        let fused = fuse_boxing_alloc_with(&mut graph, &attrs, &expand);
        assert_eq!(fused, 1, "the list boxing cluster must fuse");

        let ops = &graph.block(entry).operations;
        let nwv_pos = ops
            .iter()
            .position(|op| {
                matches!(&op.kind, OpKind::NewWithVtable { owner, vtable }
                    if owner == "W_ListObject" && *vtable == 4357049520)
            })
            .expect("NewWithVtable must be emitted with the captured type pointer");
        let mut got: Vec<String> = Vec::new();
        for op in ops.iter().skip(nwv_pos + 1) {
            match &op.kind {
                OpKind::FieldWrite { base, field, .. } if base == &ret => {
                    got.push(field.name.clone());
                }
                _ => {}
            }
        }
        assert_eq!(
            got,
            vec![
                "allocated".to_string(),
                "items".to_string(),
                "strategy".to_string(),
            ],
            "inlined nested int_items must not become a payload store on the fused result"
        );
    }

    #[test]
    fn fuse_boxing_alloc_lowers_young_alloc_dict_object() {
        // `w_dict_new_kwargs` builds a `W_DictObject` aggregate and calls
        // `malloc_typed_managed(agg)`.  The flavor match is the same
        // `gc_malloc_flavor` path as `w_float_new`; this pass rewrites it
        // to `NewWithVtable` plus the non-header setfields so the
        // constructor's jitcode records (`allocate_and_init_instance`).
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let dstorage = graph
            .push_op_var(entry, OpKind::ConstRefAddr(0), true)
            .unwrap();
        let dstrategy = graph
            .push_op_var(entry, OpKind::ConstRefAddr(0x1000), true)
            .unwrap();
        let keys_version = graph.push_op_var(entry, OpKind::ConstInt(0), true).unwrap();
        let clear_gen = graph.push_op_var(entry, OpKind::ConstInt(0), true).unwrap();
        let header = push_boxing_header(&mut graph, entry, 4357049520);
        let agg = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor("W_DictObject"),
                    args: crate::model::call_args(vec![]),
                    result_ty: ValueType::Ref(Some("W_DictObject".into())),
                },
                true,
            )
            .unwrap();
        let field =
            |base: &crate::flowspace::model::Variable, name: &str, value, ty| OpKind::FieldWrite {
                base: base.clone(),
                field: FieldDescriptor {
                    name: name.into(),
                    owner_root: Some("W_DictObject".into()),
                    owner_id: None,
                    base_is_deref: None,
                    taken_by_address: false,
                    inline_vec: false,
                    vec_part: None,
                    owner_declared_gc: None,
                    host_index: None,
                    scalar_word: None,
                },
                value,
                ty,
            };
        graph.push_op_var(
            entry,
            field(
                &agg,
                "ob_header",
                LinkArg::Value(header),
                ValueType::Ref(None),
            ),
            false,
        );
        graph.push_op_var(
            entry,
            field(
                &agg,
                "dstorage",
                LinkArg::Value(dstorage),
                ValueType::Ref(None),
            ),
            false,
        );
        graph.push_op_var(
            entry,
            field(
                &agg,
                "dstrategy",
                LinkArg::Value(dstrategy),
                ValueType::Ref(None),
            ),
            false,
        );
        graph.push_op_var(
            entry,
            field(
                &agg,
                "keys_version",
                LinkArg::Value(keys_version),
                ValueType::Int,
            ),
            false,
        );
        graph.push_op_var(
            entry,
            field(&agg, "clear_gen", LinkArg::Value(clear_gen), ValueType::Int),
            false,
        );
        let ret = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::FunctionPath {
                        segments: vec![
                            crate::runtime_names::crates::OBJECT.into(),
                            "lltype".into(),
                            "malloc_typed_managed".into(),
                        ],
                        fun_decl_id: None,
                    },
                    args: crate::model::call_args(vec![agg.clone()]),
                    result_ty: ValueType::Ref(Some("W_DictObject".into())),
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(ret.clone()));

        let attrs = std::collections::HashMap::from([
            (
                "PyObject".to_string(),
                vec![
                    ("ob_type".to_string(), ValueType::Ref(None)),
                    ("w_class".to_string(), ValueType::Ref(None)),
                ],
            ),
            (
                "W_DictObject".to_string(),
                vec![
                    ("ob_header".to_string(), ValueType::Ref(None)),
                    ("dstorage".to_string(), ValueType::Ref(None)),
                    ("dstrategy".to_string(), ValueType::Ref(None)),
                    ("keys_version".to_string(), ValueType::Int),
                    ("clear_gen".to_string(), ValueType::Int),
                ],
            ),
        ]);
        let fused = fuse_boxing_alloc(&mut graph, &attrs);
        assert_eq!(
            fused, 1,
            "young malloc_typed_managed W_DictObject must fuse"
        );

        let ops = &graph.block(entry).operations;
        let nwv_pos = ops
            .iter()
            .position(|op| {
                matches!(&op.kind, OpKind::NewWithVtable { owner, vtable }
                    if owner == "W_DictObject" && *vtable == 4357049520)
            })
            .expect("NewWithVtable must replace young malloc_typed_managed");
        assert_eq!(ops[nwv_pos].result.as_ref(), Some(&ret));
        let mut got: Vec<(String, ValueType)> = Vec::new();
        for off in 1..=4 {
            match &ops[nwv_pos + off].kind {
                OpKind::FieldWrite {
                    base, field, ty, ..
                } => {
                    assert_eq!(base, &ret);
                    got.push((field.name.clone(), ty.clone()));
                }
                other => panic!("expected payload FieldWrite, got {other:?}"),
            }
        }
        assert_eq!(
            got,
            vec![
                ("dstorage".to_string(), ValueType::Ref(None)),
                ("dstrategy".to_string(), ValueType::Ref(None)),
                ("keys_version".to_string(), ValueType::Int),
                ("clear_gen".to_string(), ValueType::Int),
            ],
        );
        assert!(
            !ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().map(String::as_str) == Some("malloc_typed_managed")
            )),
            "young malloc_typed_managed must not survive the fusion"
        );
    }

    #[test]
    fn fuse_boxing_alloc_lowers_a_header_that_declares_no_class_word() {
        // Fusion normally proves that the per-instance `w_class` agrees with
        // `ob_type`. A layout without that field has no class word to compare,
        // so its type pointer is sufficient. The table also verifies that a
        // declared-but-unstored class word and a store contradicting the layout
        // both fail closed.
        type Var = crate::flowspace::model::Variable;
        const CLASS_ADDR: i64 = 4357049520;
        const OTHER_CLASS_ADDR: i64 = 4357049600;

        /// What the cluster's header stores into `w_class`, if anything.
        enum WClass {
            /// No `w_class` store at all — the one-word header shape.
            Unstored,
            /// `get_instantiate(&T)` for the type at this address.
            Instantiate(i64),
            /// A class read back off the shadow stack: a store whose value the
            /// vtable cannot be compared against, the spelling
            /// `w_bytes_subclass_from_bytes` (bytesobject.rs) uses.
            ShadowStack,
        }

        fn call(graph: &mut FunctionGraph, blk: BlockId, path: &[&str], args: Vec<Var>) -> Var {
            graph
                .push_op_var(
                    blk,
                    OpKind::Call {
                        target: CallTarget::FunctionPath {
                            segments: path.iter().map(|s| (*s).to_string()).collect(),
                            fun_decl_id: None,
                        },
                        args: crate::model::call_args(args),
                        result_ty: ValueType::Ref(Some("object".into())),
                    },
                    true,
                )
                .unwrap()
        }
        /// A one-payload `W_IntObject` cluster whose header is a `header_owner`
        /// ctor storing `ob_type: &CLASS_ADDR` beside the `w_class` shape.
        fn cluster(header_owner: &str, w_class: &WClass) -> FunctionGraph {
            let field = |base: &Var, name: &str, owner: &str, value: &Var| OpKind::FieldWrite {
                base: base.clone(),
                field: FieldDescriptor {
                    name: name.into(),
                    owner_root: Some(owner.into()),
                    owner_id: None,
                    base_is_deref: None,
                    taken_by_address: false,
                    inline_vec: false,
                    vec_part: None,
                    owner_declared_gc: None,
                    host_index: None,
                    scalar_word: None,
                },
                value: LinkArg::Value(value.clone()),
                ty: ValueType::Ref(None),
            };
            let mut graph = FunctionGraph::new("test");
            let entry = graph.startblock;
            let payload = graph.push_op_var(entry, OpKind::ConstInt(7), true).unwrap();
            let ob_type = graph
                .push_op_var(entry, OpKind::ConstRefAddr(CLASS_ADDR), true)
                .unwrap();
            let header = graph
                .push_op_var(
                    entry,
                    OpKind::Call {
                        target: CallTarget::synthetic_transparent_ctor(header_owner),
                        args: crate::model::call_args(vec![]),
                        result_ty: ValueType::Ref(Some(header_owner.into())),
                    },
                    true,
                )
                .unwrap();
            graph.push_op_var(
                entry,
                field(&header, "ob_type", header_owner, &ob_type),
                false,
            );
            let stored = match w_class {
                WClass::Unstored => None,
                WClass::Instantiate(addr) => {
                    let arg = graph
                        .push_op_var(entry, OpKind::ConstRefAddr(*addr), true)
                        .unwrap();
                    Some(call(
                        &mut graph,
                        entry,
                        &["runtime_object", "object_model", "get_instantiate"],
                        vec![arg],
                    ))
                }
                WClass::ShadowStack => {
                    let base = graph.push_op_var(entry, OpKind::ConstInt(0), true).unwrap();
                    Some(call(
                        &mut graph,
                        entry,
                        &[
                            crate::runtime_names::crates::OBJECT,
                            "gc_roots",
                            "shadow_stack_get",
                        ],
                        vec![base],
                    ))
                }
            };
            if let Some(value) = stored {
                graph.push_op_var(
                    entry,
                    field(&header, "w_class", header_owner, &value),
                    false,
                );
            }
            let agg = graph
                .push_op_var(
                    entry,
                    OpKind::Call {
                        target: CallTarget::synthetic_transparent_ctor("W_IntObject"),
                        args: crate::model::call_args(vec![]),
                        result_ty: ValueType::Ref(Some("W_IntObject".into())),
                    },
                    true,
                )
                .unwrap();
            graph.push_op_var(
                entry,
                field(&agg, "ob_header", "W_IntObject", &header),
                false,
            );
            graph.push_op_var(entry, field(&agg, "intval", "W_IntObject", &payload), false);
            let ret = call(
                &mut graph,
                entry,
                &[
                    crate::runtime_names::crates::OBJECT,
                    "lltype",
                    "malloc_typed",
                ],
                vec![agg],
            );
            graph.set_return(entry, Some(ret));
            graph
        }

        // Synthetic fixtures have no StructId registry, so publish the exact
        // owner spelling the ctor carries. Production aliases resolve through
        // `registered_struct_layout`'s StructId lookup, never by leaf name.
        let one_word: &[&str] = &["ob_type"];
        let two_word: &[&str] = &["ob_type", "w_class"];
        let rows: [(&str, &str, Option<&[&str]>, WClass, usize); 6] = [
            (
                "a one-word header declaring no class word",
                "TypeOnlyHeader",
                Some(one_word),
                WClass::Unstored,
                1,
            ),
            (
                "a two-word header storing its class word",
                "PyObject",
                Some(two_word),
                WClass::Instantiate(CLASS_ADDR),
                1,
            ),
            (
                "a two-word header leaving its class word unstored",
                "PyObject",
                Some(two_word),
                WClass::Unstored,
                0,
            ),
            (
                "a header whose layout is not registered at all",
                "TypeOnlyHeader",
                None,
                WClass::Unstored,
                0,
            ),
            (
                "a one-word layout contradicted by a class-word store",
                "TypeOnlyHeader",
                Some(one_word),
                WClass::ShadowStack,
                0,
            ),
            (
                "a one-word layout storing another type's class word",
                "TypeOnlyHeader",
                Some(one_word),
                WClass::Instantiate(OTHER_CLASS_ADDR),
                0,
            ),
        ];
        for (label, header_owner, layout, w_class, expected) in rows {
            let mut graph = cluster(header_owner, &w_class);
            let entry = graph.startblock;
            let mut attrs = std::collections::HashMap::from([(
                "W_IntObject".to_string(),
                vec![
                    ("ob_header".to_string(), ValueType::Ref(None)),
                    ("intval".to_string(), ValueType::Int),
                ],
            )]);
            if let Some(names) = layout {
                let rows = names
                    .iter()
                    .map(|name| ((*name).to_string(), ValueType::Ref(None)))
                    .collect();
                attrs.insert(header_owner.to_string(), rows);
            }
            assert_eq!(
                fuse_boxing_alloc(&mut graph, &attrs),
                expected,
                "{label}: wrong number of fused clusters"
            );
            let ops = &graph.block(entry).operations;
            let residual = ops.iter().any(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                        if segments.last().map(String::as_str) == Some("malloc_typed")
                )
            });
            assert_eq!(
                residual,
                expected == 0,
                "{label}: malloc_typed residual must survive exactly when the cluster declines"
            );
            if expected == 1 {
                assert!(
                    ops.iter().any(|op| matches!(
                        &op.kind,
                        OpKind::NewWithVtable { owner, vtable }
                            if owner == "W_IntObject" && *vtable == CLASS_ADDR
                    )),
                    "{label}: the fused allocation must carry the resolved ob_type address"
                );
            }
        }
    }

    #[test]
    fn fuse_boxing_alloc_forks_new_and_new_with_vtable_on_the_type_word() {
        // `jtransform.py rewrite_op_malloc`'s fork, and its two refusals.
        //
        // The opcode is decided by what the STRUCT carries, never by what the
        // cluster happens to store: a constant type word is taken onto the
        // allocation as `new_with_vtable`, a struct with no type word at all
        // gets the plain `new`, and the flavors the allocation op cannot
        // express are refused instead of approximated.
        //
        // The last row is the one that must never lower.  A type word that is
        // present but not constant cannot travel on the allocation, and it
        // cannot follow as its own store either: `optimize_setfield_gc` removes
        // a `setfield_gc` whose descr `is_typeptr()` and keeps only a constant
        // value, as `known_class` (`heaptracker.all_fielddescrs`). Emitting `new` for
        // it ships an object whose type word is never written.
        type Var = crate::flowspace::model::Variable;
        const CLASS_ADDR: i64 = 4357049520;

        /// What the cluster's header stores into `ob_type`.
        enum ObType {
            /// No store at all — upstream's struct with no vtable to take.
            Unstored,
            /// The layout declares a type word, but this path never writes it.
            DeclaredUnstored,
            /// A constant type pointer.
            Constant,
            /// A class read back off the shadow stack: present, but no
            /// constant this pass can fold onto the allocation.
            ShadowStack,
        }

        fn cluster(ob_type: &ObType, flavor: &str) -> FunctionGraph {
            let field = |base: &Var, name: &str, owner: &str, value: &Var| OpKind::FieldWrite {
                base: base.clone(),
                field: FieldDescriptor {
                    name: name.into(),
                    owner_root: Some(owner.into()),
                    owner_id: None,
                    base_is_deref: None,
                    taken_by_address: false,
                    inline_vec: false,
                    vec_part: None,
                    owner_declared_gc: None,
                    host_index: None,
                    scalar_word: None,
                },
                value: LinkArg::Value(value.clone()),
                ty: ValueType::Ref(None),
            };
            let mut graph = FunctionGraph::new("test");
            let entry = graph.startblock;
            let payload = graph.push_op_var(entry, OpKind::ConstInt(7), true).unwrap();
            let header = graph
                .push_op_var(
                    entry,
                    OpKind::Call {
                        target: CallTarget::synthetic_transparent_ctor("TypeOnlyHeader"),
                        args: crate::model::call_args(vec![]),
                        result_ty: ValueType::Ref(Some("TypeOnlyHeader".into())),
                    },
                    true,
                )
                .unwrap();
            let stored = match ob_type {
                ObType::Unstored | ObType::DeclaredUnstored => None,
                ObType::Constant => Some(
                    graph
                        .push_op_var(entry, OpKind::ConstRefAddr(CLASS_ADDR), true)
                        .unwrap(),
                ),
                ObType::ShadowStack => {
                    let base = graph.push_op_var(entry, OpKind::ConstInt(0), true).unwrap();
                    Some(
                        graph
                            .push_op_var(
                                entry,
                                OpKind::Call {
                                    target: CallTarget::FunctionPath {
                                        segments: [
                                            crate::runtime_names::crates::OBJECT,
                                            "gc_roots",
                                            "shadow_stack_get",
                                        ]
                                        .iter()
                                        .map(|s| (*s).to_string())
                                        .collect(),
                                        fun_decl_id: None,
                                    },
                                    args: crate::model::call_args(vec![base]),
                                    result_ty: ValueType::Ref(Some("object".into())),
                                },
                                true,
                            )
                            .unwrap(),
                    )
                }
            };
            if let Some(value) = stored {
                graph.push_op_var(
                    entry,
                    field(&header, "ob_type", "TypeOnlyHeader", &value),
                    false,
                );
            }
            let agg = graph
                .push_op_var(
                    entry,
                    OpKind::Call {
                        target: CallTarget::synthetic_transparent_ctor("W_IntObject"),
                        args: crate::model::call_args(vec![]),
                        result_ty: ValueType::Ref(Some("W_IntObject".into())),
                    },
                    true,
                )
                .unwrap();
            graph.push_op_var(
                entry,
                field(&agg, "ob_header", "W_IntObject", &header),
                false,
            );
            graph.push_op_var(entry, field(&agg, "intval", "W_IntObject", &payload), false);
            let ret = graph
                .push_op_var(
                    entry,
                    OpKind::Call {
                        target: CallTarget::FunctionPath {
                            segments: [crate::runtime_names::crates::OBJECT, "lltype", flavor]
                                .iter()
                                .map(|s| (*s).to_string())
                                .collect(),
                            fun_decl_id: None,
                        },
                        args: crate::model::call_args(vec![agg]),
                        result_ty: ValueType::Ref(Some("W_IntObject".into())),
                    },
                    true,
                )
                .unwrap();
            graph.set_return(entry, Some(ret));
            graph
        }

        /// `Some(None)` expects a plain `New`; `Some(Some(addr))` expects a
        /// `NewWithVtable` carrying that address; `None` expects no fusion.
        type Expected = Option<Option<i64>>;
        let rows: [(&str, ObType, &str, Expected); 7] = [
            (
                "a constant type word rides on the allocation",
                ObType::Constant,
                "malloc_typed",
                Some(Some(CLASS_ADDR)),
            ),
            (
                "a struct with no type word takes the plain new",
                ObType::Unstored,
                "malloc",
                Some(None),
            ),
            (
                "the managed movable allocator takes it too",
                ObType::Unstored,
                "malloc_typed_managed",
                Some(None),
            ),
            (
                "a non-moving allocator is refused, not re-flavored",
                ObType::Unstored,
                "malloc_typed_stable",
                None,
            ),
            (
                "a collector-external allocator is refused as well",
                ObType::Unstored,
                "malloc_typed",
                None,
            ),
            (
                "a type word that is present but not constant never lowers",
                ObType::ShadowStack,
                "malloc",
                None,
            ),
            (
                "a declared type word that is not stored never becomes plain new",
                ObType::DeclaredUnstored,
                "malloc",
                None,
            ),
        ];

        for (label, ob_type, flavor, expected) in rows {
            let mut graph = cluster(&ob_type, flavor);
            let entry = graph.startblock;
            let header_layout = match ob_type {
                ObType::Unstored => vec![],
                ObType::DeclaredUnstored | ObType::Constant | ObType::ShadowStack => {
                    vec![("ob_type".to_string(), ValueType::Ref(None))]
                }
            };
            let attrs = std::collections::HashMap::from([
                (
                    "W_IntObject".to_string(),
                    vec![
                        ("ob_header".to_string(), ValueType::Ref(None)),
                        ("intval".to_string(), ValueType::Int),
                    ],
                ),
                ("TypeOnlyHeader".to_string(), header_layout),
            ]);
            assert_eq!(
                fuse_boxing_alloc(&mut graph, &attrs),
                usize::from(expected.is_some()),
                "{label}: wrong number of fused clusters"
            );
            let ops = &graph.block(entry).operations;
            let residual = ops.iter().any(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                        if segments.last().map(String::as_str) == Some(flavor)
                )
            });
            assert_eq!(
                residual,
                expected.is_none(),
                "{label}: the residual malloc must survive exactly when the cluster declines"
            );
            let allocation = ops.iter().find(|op| match &op.kind {
                OpKind::NewWithVtable { owner, .. } | OpKind::New { owner } => {
                    owner == "W_IntObject"
                }
                _ => false,
            });
            let emitted = allocation.map(|op| match &op.kind {
                OpKind::NewWithVtable { vtable, .. } => Some(*vtable),
                _ => None,
            });
            assert_eq!(emitted, expected, "{label}: wrong allocation opcode");
            // Whichever arm ran, the type word is never re-emitted as a store
            // onto the allocation: `heaptracker.all_fielddescrs` keeps `typeptr` out of
            // every field list, and a `setfield_gc` for it would be removed
            // downstream anyway.  (The dropped `ob_header` subtree keeps its
            // own store until the later remnant sweep, so this asks about the
            // allocation's result specifically.)
            if let Some(result) = allocation.and_then(|op| op.result.as_ref()) {
                assert!(
                    !ops.iter().any(|op| matches!(
                        &op.kind,
                        OpKind::FieldWrite { base, field, .. }
                            if base == result && field.name.as_str() == "ob_type"
                    )),
                    "{label}: the fused allocation must not carry an ob_type store"
                );
            }
        }
    }

    #[test]
    fn prune_dead_boxing_remnants_keeps_cast_used_as_live_store_base() {
        // The malloc-removal exemption (a store's base is not rooted) must be
        // scoped to fresh `SyntheticTransparentCtor` aggregates.  A store
        // *through* a `__cast_instance_intrinsic` alias is a real heap side effect:
        // even when the cast's result is read nowhere but the store base, the
        // cast and the store must survive.  The same graph carries a genuinely
        // dead `SyntheticTransparentCtor` whose store IS swept, so the pass is
        // proven to run rather than trivially early-returning.
        type Var = crate::flowspace::model::Variable;
        let field = |base: &Var, name: &str, value: &Var| OpKind::FieldWrite {
            base: base.clone(),
            field: FieldDescriptor {
                name: name.into(),
                owner_root: None,
                owner_id: None,
                base_is_deref: None,
                taken_by_address: false,
                inline_vec: false,
                vec_part: None,
                owner_declared_gc: None,
                host_index: None,
                scalar_word: None,
            },
            value: LinkArg::Value(value.clone()),
            ty: ValueType::Ref(None),
        };
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let obj = graph.alloc_value_var();
        graph.push_inputarg_var(entry, obj.clone());
        let payload = graph
            .push_op_var(entry, OpKind::ConstFloat(0.0f64.to_bits()), true)
            .unwrap();
        // A real mutation through a pointer reinterpret: `p` is read only as
        // the store base, yet the store must persist.
        let p = graph
            .push_op_var(
                entry,
                crate::model::cast_instance_call("W_FloatObject", obj.clone()),
                true,
            )
            .unwrap();
        graph.push_op_var(entry, field(&p, "floatval", &payload), false);
        // A genuinely dead fresh aggregate whose store is removable.
        let dead_agg = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor("PyObject"),
                    args: crate::model::call_args(vec![]),
                    result_ty: ValueType::Ref(Some("PyObject".into())),
                },
                true,
            )
            .unwrap();
        graph.push_op_var(entry, field(&dead_agg, "ob_type", &payload), false);
        graph.set_return(entry, Some(obj.clone()));

        prune_dead_boxing_remnants(&mut graph);

        let ops = &graph.block(entry).operations;
        assert!(
            ops.iter().any(|op| op.result.as_ref() == Some(&p)),
            "the cast aliasing a live store base must survive"
        );
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::FieldWrite { base, .. } if base == &p
            )),
            "the real store through the cast must survive"
        );
        assert!(
            !ops.iter().any(|op| op.result.as_ref() == Some(&dead_agg)),
            "the genuinely dead aggregate ctor must still be swept"
        );
    }

    #[test]
    fn prune_dead_boxing_remnants_sweeps_cross_block_threaded_cluster() {
        // The lowered `w_float_new` shape after fusion: the dead header cast
        // is produced in the entry block and threaded — by *reusing the same
        // `Variable`* as both the entry op result and the successor block's
        // inputarg (`set_goto_from_framestate` / `ensure_variable_at_block`) —
        // into the block where the dead `PyObject` header ctor reads it.  The
        // old `Link.arg`-counts-as-read sweep kept the whole cluster (the
        // threaded copy pinned the cast; the `FieldWrite` pinned the ctor);
        // the dependency-flow sweep reaches it because the threaded copy's
        // target inputarg is itself unread, then `prune_dead_phis` reclaims
        // the dangling inputarg / link arg / address constant.
        type Var = crate::flowspace::model::Variable;
        let cast_pytype = |arg: &Var| crate::model::cast_instance_call("PyType", arg.clone());
        let field = |base: &Var, name: &str, value: &Var| OpKind::FieldWrite {
            base: base.clone(),
            field: FieldDescriptor {
                name: name.into(),
                owner_root: Some("PyObject".into()),
                owner_id: None,
                base_is_deref: None,
                taken_by_address: false,
                inline_vec: false,
                vec_part: None,
                owner_declared_gc: None,
                host_index: None,
                scalar_word: None,
            },
            value: LinkArg::Value(value.clone()),
            ty: ValueType::Ref(None),
        };
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let value = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "value".into(),
                    ty: ValueType::Float,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.push_inputarg_var(entry, value.clone());
        let get_instantiate = |arg: &Var| OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: vec![
                    "runtime_object".into(),
                    "object_model".into(),
                    "get_instantiate".into(),
                ],
                fun_decl_id: None,
            },
            args: crate::model::call_args(vec![arg.clone()]),
            result_ty: ValueType::Ref(Some("object".into())),
        };
        let ty_addr1 = graph
            .push_op_var(entry, OpKind::ConstRefAddr(4357049520), true)
            .unwrap();
        // DEAD: the header `ob_type` cast, defined in the entry block.
        let cast = graph
            .push_op_var(entry, cast_pytype(&ty_addr1), true)
            .unwrap();
        let ty_addr2 = graph
            .push_op_var(entry, OpKind::ConstRefAddr(4357049520), true)
            .unwrap();
        // DEAD: a second cast feeding `get_instantiate`, whose result is the
        // header `w_class`.  `get_instantiate` is an observation-free
        // `instantiate`-slot load kept alive only by the threaded w_class
        // store, so it (and the cast feeding it) must be swept once the
        // header is dropped.
        let cast2 = graph
            .push_op_var(entry, cast_pytype(&ty_addr2), true)
            .unwrap();
        let inst = graph
            .push_op_var(entry, get_instantiate(&cast2), true)
            .unwrap();
        // LIVE: the fused boxing allocation + its payload store.
        let boxed = graph
            .push_op_var(
                entry,
                OpKind::NewWithVtable {
                    owner: "W_FloatObject".into(),
                    vtable: 0x5000_0000,
                },
                true,
            )
            .unwrap();
        graph.push_op_var(entry, field(&boxed, "floatval", &value), false);

        // Successor block reached after the boundary `get_instantiate` /
        // `malloc_typed` call would have opened; its inputargs reuse the SAME
        // `Variable`s the entry threads (id-reuse, not fresh phis).
        let blk = graph.create_block();
        graph.push_inputarg_var(blk, value.clone());
        graph.push_inputarg_var(blk, cast.clone());
        graph.push_inputarg_var(blk, inst.clone());
        graph.push_inputarg_var(blk, boxed.clone());
        graph.set_goto(
            entry,
            blk,
            vec![value.clone(), cast.clone(), inst.clone(), boxed.clone()],
        );

        // DEAD: the inner `PyObject` header ctor + its stores reading the cast
        // and the `get_instantiate` result.
        let header = graph
            .push_op_var(
                blk,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor("PyObject"),
                    args: crate::model::call_args(vec![]),
                    result_ty: ValueType::Ref(Some("PyObject".into())),
                },
                true,
            )
            .unwrap();
        graph.push_op_var(blk, field(&header, "ob_type", &cast), false);
        graph.push_op_var(blk, field(&header, "w_class", &inst), false);
        // LIVE: the return cast reads the boxing allocation.
        let ret = graph
            .push_op_var(
                blk,
                crate::model::cast_instance_call("PyObject", boxed.clone()),
                true,
            )
            .unwrap();
        graph.set_return(blk, Some(ret.clone()));

        prune_dead_boxing_remnants(&mut graph);
        prune_dead_phis(&mut graph);

        let kinds: Vec<&OpKind> = graph
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .map(|o| &o.kind)
            .collect();
        assert!(
            !kinds.iter().any(|k| matches!(
                k,
                OpKind::Call {
                    target: CallTarget::SyntheticTransparentCtor { .. },
                    ..
                }
            )),
            "the cross-block dead header ctor must be swept: {kinds:#?}"
        );
        assert!(
            !kinds
                .iter()
                .any(|k| cast_instance_root(k) == Some("PyType")),
            "the dead PyType cast threaded across the block boundary must be swept"
        );
        assert!(
            kinds.iter().any(
                |k| matches!(k, OpKind::NewWithVtable { owner, .. } if owner == "W_FloatObject")
            ),
            "the live NewWithVtable must survive"
        );
        assert!(
            kinds
                .iter()
                .any(|k| cast_instance_root(k) == Some("PyObject")),
            "the live return cast must survive"
        );
        assert!(
            !kinds.iter().any(|k| matches!(
                k,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, .. }
                    if segments.last().map(String::as_str) == Some("get_instantiate")
            )),
            "the dead get_instantiate (w_class feed) must be swept once the header is dropped"
        );
        assert!(
            !graph.block(blk).inputargs.contains(&cast)
                && !graph.block(blk).inputargs.contains(&inst),
            "the dangling threaded inputargs must be reclaimed by prune_dead_phis"
        );
        let _ = (ret, header, cast2, ty_addr1, ty_addr2);
    }

    #[test]
    fn prune_dead_boxing_remnants_sweeps_cross_block_threaded_vec_box() {
        // The `vec![e0]` lowering after a consumer rewrite to `newlist`: the
        // `box_assume_init_into_vec_unsafe(box [e0])` idiom leaves a no-arg
        // `boxed::Box::new_uninit()` allocation storing a fresh `Array`
        // aggregate.  `front::mir`'s framestate threading splits the box
        // allocation from the block that consumed it, reusing the SAME
        // `Variable` as both the entry op result and the successor block's
        // inputarg, and threads the box pointer through the `Link`.  The
        // successor no longer reads the box (its consumer became a `newlist`
        // reading the elements directly), so the whole box → Array → element
        // store cluster is dead.  `remove_dead_aggregates` cannot prove it —
        // it counts the threaded `Link.arg` as an unconditional read — but the
        // dependency-flow sweep reaches it: the box's target inputarg is unread
        // and the box is a fresh alloc exempt from store-base rooting.
        type Var = crate::flowspace::model::Variable;
        let field = |base: &Var, name: &str, value: &Var| OpKind::FieldWrite {
            base: base.clone(),
            field: FieldDescriptor {
                name: name.into(),
                owner_root: None,
                owner_id: None,
                base_is_deref: None,
                taken_by_address: false,
                inline_vec: false,
                vec_part: None,
                owner_declared_gc: None,
                host_index: None,
                scalar_word: None,
            },
            value: LinkArg::Value(value.clone()),
            ty: ValueType::Ref(None),
        };
        let box_new_uninit = || OpKind::Call {
            target: CallTarget::FunctionPath {
                segments: vec!["boxed".into(), "Box".into(), "new_uninit".into()],
                fun_decl_id: None,
            },
            args: crate::model::call_args(vec![]),
            result_ty: ValueType::Ref(Some("Box".into())),
        };
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        // A live `vec!` element (a genuine parameter).
        let elem = graph.alloc_value_var();
        graph.push_inputarg_var(entry, elem.clone());
        // DEAD: the `Box::new_uninit()` heap box the `vec!` lowering opens.
        let boxed = graph.push_op_var(entry, box_new_uninit(), true).unwrap();
        // DEAD: the fresh `Array` aggregate the box stores, plus its `__pos_0`
        // element write.
        let array = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor("Array"),
                    args: crate::model::call_args(vec![]),
                    result_ty: ValueType::Ref(Some("Array".into())),
                },
                true,
            )
            .unwrap();
        graph.push_op_var(entry, field(&array, "__pos_0", &elem), false);
        // DEAD: the store threading the `Array` into the box.
        graph.push_op_var(entry, field(&boxed, "ptr", &array), false);

        // Successor block reached after the boundary the framestate threading
        // opens; its inputarg reuses the SAME `boxed` Variable (id-reuse), and
        // nothing in it reads the box.
        let blk = graph.create_block();
        graph.push_inputarg_var(blk, boxed.clone());
        graph.set_goto(entry, blk, vec![boxed.clone()]);
        graph.set_return(blk, None);

        let removed = prune_dead_boxing_remnants(&mut graph);
        assert!(
            removed >= 3,
            "the box, the Array ctor, and both field stores must be swept, got {removed}"
        );

        let kinds: Vec<&OpKind> = graph
            .blocks
            .iter()
            .flat_map(|b| &b.operations)
            .map(|o| &o.kind)
            .collect();
        assert!(
            !kinds.iter().any(|k| matches!(
                k,
                OpKind::Call { target: CallTarget::FunctionPath { segments, .. }, args, .. }
                    if args.is_empty()
                        && segments.last().map(String::as_str) == Some("new_uninit")
            )),
            "the dead cross-block `Box::new_uninit` box must be swept: {kinds:#?}"
        );
        assert!(
            !kinds.iter().any(|k| matches!(
                k,
                OpKind::Call {
                    target: CallTarget::SyntheticTransparentCtor { .. },
                    ..
                }
            )),
            "the dead `Array` aggregate ctor must be swept"
        );
        assert!(
            !kinds.iter().any(|k| matches!(k, OpKind::FieldWrite { .. })),
            "both dead field stores (the `Array` element write and the box store) must be swept"
        );

        // `prune_dead_phis` reclaims the now-dangling threaded inputarg / link
        // arg the sweep leaves behind.
        prune_dead_phis(&mut graph);
        assert!(
            !graph.block(blk).inputargs.contains(&boxed),
            "the dangling threaded box inputarg must be reclaimed by prune_dead_phis"
        );
    }

    #[test]
    fn prune_dead_phis_collapses_chained_dead_phis() {
        // entry → merge1(phi 'x') → merge2(phi 'y' fed by phi 'x') → returnblock(void)
        // Neither 'x' nor 'y' is read.  Backward dataflow correctly
        // identifies both as dead in a single pass: 'y' is unread, so
        // its dependency `phi_x` (link arg from merge1) doesn't get
        // promoted, and 'x' itself is unread either.
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let const_v_var = graph.push_op_var(entry, OpKind::ConstInt(1), true).unwrap();
        let merge1 = graph.create_block();
        let phi_x_var = install_phi(&mut graph, merge1, "x");
        graph.set_goto(entry, merge1, vec![const_v_var]);
        let merge2 = graph.create_block();
        install_phi(&mut graph, merge2, "y");
        graph.set_goto(merge1, merge2, vec![phi_x_var]);
        graph.set_return(merge2, None);

        prune_dead_phis(&mut graph);

        assert!(
            graph.block(merge1).inputargs.is_empty(),
            "chained dead phi 'x' must collapse after 'y' goes"
        );
        assert!(
            graph.block(merge2).inputargs.is_empty(),
            "leaf dead phi 'y' must be pruned"
        );
        assert!(
            graph.block(entry).exits[0].args.is_empty(),
            "entry → merge1 link arg dropped after 'x' pruned"
        );
        assert!(
            graph.block(merge1).exits[0].args.is_empty(),
            "merge1 → merge2 link arg dropped after 'y' pruned"
        );
    }

    #[test]
    fn prune_dead_phis_preserves_special_blocks() {
        // Function-entry inputargs (startblock), return-value
        // (returnblock), and exception escape (exceptblock) inputargs
        // are part of the calling convention / runtime contract and
        // must NEVER be pruned even if locally unread.
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        // Add an Input op at startblock that never gets read — emulates
        // an unused function parameter.
        let unused_param_var = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "param".into(),
                    ty: ValueType::Int,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.push_inputarg_var(entry, unused_param_var.clone());
        graph.set_return(entry, None);

        let pre_returnblock_args = graph.block(graph.returnblock).inputargs.clone();
        let pre_exceptblock_args = graph.block(graph.exceptblock).inputargs.clone();

        prune_dead_phis(&mut graph);

        assert_eq!(
            graph.block(entry).inputargs,
            vec![unused_param_var.clone()],
            "startblock inputargs are calling-convention; never pruned"
        );
        assert_eq!(
            graph.block(graph.returnblock).inputargs.clone(),
            pre_returnblock_args,
            "returnblock inputargs are runtime contract; never pruned"
        );
        assert_eq!(
            graph.block(graph.exceptblock).inputargs.clone(),
            pre_exceptblock_args,
            "exceptblock inputargs are runtime contract; never pruned"
        );
    }

    #[test]
    fn prune_dead_phis_preserves_pure_raising_op() {
        // `simplify.py canremove`'s `op is not block.raising_op`
        // clause: a pure-classified op (e.g. `int_add` modelling
        // `int_add_ovf`) parked as the LAST op of a `canraise` block
        // must NOT be DCE'd even when its result is unread.  The raise
        // side-effect is observable, so removing the op would be a
        // semantic regression.
        //
        // Shape: entry — int_add(unread result) — exitswitch=LastException;
        //   normal exit → returnblock; except exit → exceptblock.
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let lhs_var = graph.push_op_var(entry, OpKind::ConstInt(1), true).unwrap();
        let rhs_var = graph.push_op_var(entry, OpKind::ConstInt(2), true).unwrap();
        let raising_var = graph
            .push_op_var(
                entry,
                OpKind::BinOp {
                    op: "int_add".into(),
                    lhs: lhs_var,
                    rhs: rhs_var,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        // Mark the entry block as canraise: exitswitch=LastException,
        // exits=[normal → returnblock, except → exceptblock].
        let returnblock = graph.returnblock;
        let exceptblock = graph.exceptblock;
        // Synthesize the (etype, evalue) pair that exceptblock expects
        // — values flow as Constants from the front-end raise site,
        // but for the test we need only the link arity to match.
        let etype_var = graph.push_op_var(entry, OpKind::ConstInt(0), true).unwrap();
        let evalue_var = graph.push_op_var(entry, OpKind::ConstInt(0), true).unwrap();
        let normal_arg = LinkArg::Value(raising_var.clone());
        let exc_etype_arg = LinkArg::Value(etype_var);
        let exc_evalue_arg = LinkArg::Value(evalue_var);
        {
            let block = graph.block_mut(entry);
            block.exitswitch = Some(ExitSwitch::LastException);
            block.exits = vec![
                Link::new_mixed(vec![normal_arg], returnblock, None),
                Link::new_mixed(vec![exc_etype_arg, exc_evalue_arg], exceptblock, None),
            ];
        }

        prune_dead_phis(&mut graph);

        let still_present = graph
            .block(entry)
            .operations
            .iter()
            .any(|op| op.result.as_ref() == Some(&raising_var));
        assert!(
            still_present,
            "pure raising_op (last op of canraise block) must survive DCE per \
             simplify.py:436 `op is not block.raising_op`",
        );
    }

    #[test]
    fn prune_dead_phis_skips_non_canonical_entry_blocks() {
        // Closure / generic-specialisation lowering can put function
        // parameters at a non-zero block id with no incoming link
        // (the `generated::*` pipeline does this for `eval_loop_jit`'s
        // body block).  Even if the parameter isn't read graph-side,
        // pruning must NOT remove it — the calling convention assumes
        // every inputarg is supplied.  The block is identified as
        // "entry-like" by having no link targeting it.
        let mut graph = FunctionGraph::new("test");
        // Build: startblock → returnblock(void).  Add an extra
        // orphan-entry block with one Input op + one inputarg, no
        // predecessors at all (not even from startblock).  The block
        // is unreachable but represents the "non-canonical entry"
        // shape pyre emits in some generated pipelines.
        graph.set_return(graph.startblock, None);
        let orphan_entry = graph.create_block();
        let unused_param_var = graph
            .push_op_var(
                orphan_entry,
                OpKind::Input {
                    name: "captured".into(),
                    ty: ValueType::Ref(None),
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.push_inputarg_var(orphan_entry, unused_param_var.clone());

        clear_unreachable_blocks(&mut graph);
        assert_eq!(
            graph.block(orphan_entry).inputargs,
            vec![unused_param_var.clone()],
            "reachability cleanup must preserve a secondary calling-convention entry"
        );

        prune_dead_phis(&mut graph);

        assert_eq!(
            graph.block(orphan_entry).inputargs,
            vec![unused_param_var.clone()],
            "non-canonical entry block (no incoming link) must keep its inputargs"
        );
    }

    /// `FrameState::union` walks the merged stack.  The disagreement at
    /// stack slot 0 (Variable vs Variable with distinct identities)
    /// yields a fresh phi Variable per `framestate.py return
    /// Variable()`; that fresh identity is distinct from both
    /// predecessors.
    #[test]
    fn union_registers_fresh_stack_phi_variable_with_valueid() {
        use crate::flowspace::framestate::StackElem;
        use crate::flowspace::model::{Hlvalue, Variable};

        let mut graph = FunctionGraph::new("union_stack_phi");
        let v_a = Variable::new();
        let v_b = Variable::new();

        let pred_a = FrameState {
            entries: Vec::new(),
            stack: vec![StackElem::Value(Hlvalue::Variable(v_a.clone()))],
            ..Default::default()
        };
        let pred_b = FrameState {
            entries: Vec::new(),
            stack: vec![StackElem::Value(Hlvalue::Variable(v_b.clone()))],
            ..Default::default()
        };

        let merged = pred_a
            .union(&pred_b, &mut graph)
            .expect("disagreeing-Variable union must succeed");
        let StackElem::Value(Hlvalue::Variable(phi)) = &merged.stack[0] else {
            panic!("disagreement must mint a fresh phi Variable");
        };
        assert!(
            *phi != v_a && *phi != v_b,
            "fresh phi Variable must be distinct from both predecessors after union"
        );
    }

    /// Carry-through stack Variable (identical Hlvalue identity on both
    /// predecessors) carries through to the same Variable handle without
    /// minting a fresh phi — `framestate.py:108 if w1 == w2: return w1`
    /// parity.
    #[test]
    fn union_stack_carry_through_variable_keeps_existing_valueid() {
        use crate::flowspace::framestate::StackElem;
        use crate::flowspace::model::{Hlvalue, Variable};

        let mut graph = FunctionGraph::new("union_stack_carry");
        let shared = Variable::new();

        let pred_a = FrameState {
            entries: Vec::new(),
            stack: vec![StackElem::Value(Hlvalue::Variable(shared.clone()))],
            ..Default::default()
        };
        let pred_b = FrameState {
            entries: Vec::new(),
            stack: vec![StackElem::Value(Hlvalue::Variable(shared.clone()))],
            ..Default::default()
        };

        let merged = pred_a
            .union(&pred_b, &mut graph)
            .expect("carry-through union must succeed");
        let StackElem::Value(Hlvalue::Variable(carried)) = &merged.stack[0] else {
            panic!("Variable carry-through expected");
        };
        assert_eq!(
            carried, &shared,
            "carry-through must reuse the shared Variable identity"
        );
    }

    /// `FSException` projections route through the same registration
    /// walk: a fresh phi Variable minted by the per-cell exception
    /// `union` lands in `merged.last_exception` with a paired slot.
    #[test]
    fn union_registers_fresh_exception_phi_variable_with_valueid() {
        use crate::flowspace::model::{FSException, Hlvalue, Variable};

        let mut graph = FunctionGraph::new("union_exc_phi");
        let v_t1 = Variable::new();
        let v_t2 = Variable::new();
        let v_v1 = Variable::new();
        let v_v2 = Variable::new();
        let pred_a = FrameState {
            entries: Vec::new(),
            stack: Vec::new(),
            last_exception: Some(FSException::new(
                Hlvalue::Variable(v_t1.clone()),
                Hlvalue::Variable(v_v1.clone()),
            )),
            ..Default::default()
        };
        let pred_b = FrameState {
            entries: Vec::new(),
            stack: Vec::new(),
            last_exception: Some(FSException::new(
                Hlvalue::Variable(v_t2.clone()),
                Hlvalue::Variable(v_v2.clone()),
            )),
            ..Default::default()
        };

        let merged = pred_a
            .union(&pred_b, &mut graph)
            .expect("disagreeing-exception union must succeed");
        let exc = merged
            .last_exception
            .as_ref()
            .expect("merged must carry FSException");
        let Hlvalue::Variable(v_t_phi) = &exc.w_type else {
            panic!("w_type phi expected");
        };
        let Hlvalue::Variable(v_v_phi) = &exc.w_value else {
            panic!("w_value phi expected");
        };
        assert!(
            *v_t_phi != v_t1 && *v_t_phi != v_t2,
            "w_type disagreement must mint a fresh phi Variable"
        );
        assert!(
            *v_v_phi != v_v1 && *v_v_phi != v_v2,
            "w_value disagreement must mint a fresh phi Variable"
        );
    }

    /// `FrameState::union` derives the parallel
    /// `locals_w` (`Hlvalue` carrier matching `framestate.py
    /// self.locals_w`) from the unioned `entries` (Variable carrier).
    /// Each defined slot's Variable carries through directly via
    /// clone; None-killed slots stay None.
    #[test]
    fn union_derives_locals_w_hlvalue_carrier_from_entries() {
        use crate::flowspace::model::{Hlvalue, Variable};
        let mut graph = FunctionGraph::new("locals_w_derive");
        let v_shared = Variable::new();
        let v_a_only = Variable::new();
        let v_b_only = Variable::new();

        let pred_a = FrameState {
            entries: vec![Some(v_shared.clone()), Some(v_a_only.clone()), None],
            ..Default::default()
        };
        let pred_b = FrameState {
            entries: vec![Some(v_shared.clone()), None, Some(v_b_only.clone())],
            ..Default::default()
        };
        let merged = pred_a
            .union(&pred_b, &mut graph)
            .expect("union must succeed");
        assert_eq!(
            merged.entries.len(),
            3,
            "entries carrier preserves max-extension length",
        );
        assert_eq!(
            merged.locals_w.len(),
            merged.entries.len(),
            "locals_w must be in lockstep with entries",
        );
        // Slot 0: agreement at shared_vid → carry-through, locals_w[0]
        // is the same Variable identity.
        assert!(
            matches!(&merged.locals_w[0], Some(Hlvalue::Variable(v)) if v.id() == v_shared.id()),
            "carry-through slot 0 must mirror shared Variable identity",
        );
        // Slots 1, 2: None-killed at union, locals_w mirrors None.
        assert!(
            merged.locals_w[1].is_none() && merged.locals_w[2].is_none(),
            "None-killed slots stay None in locals_w",
        );
    }

    /// `set_goto_from_framestate` threads `getoutputargs`'s
    /// mixed-shape result through `Link::new_mixed` per
    /// `flowcontext.py:438 block.recloseblock(Link(outputargs,
    /// newblock))`.  A predecessor with one locals slot + one stack
    /// Constant + one stack Variable yields a Link whose `args`
    /// carry both `LinkArg::Value` (locals + stack Variable) and
    /// `LinkArg::Const` (stack Constant) side by side.
    #[test]
    fn set_goto_from_framestate_emits_mixed_link_args_through_new_mixed() {
        use crate::flowspace::framestate::StackElem;
        use crate::flowspace::model::{Constant, Hlvalue, Variable};

        let mut graph = FunctionGraph::new("set_goto_mixed");
        let pred = graph.create_block();

        let v_local = Variable::new();
        let v_stack = Variable::new();

        let pred_state = FrameState {
            entries: vec![Some(v_local.clone())],
            stack: vec![
                StackElem::Value(Hlvalue::Constant(Constant::new(ConstValue::Int(42)))),
                StackElem::Value(Hlvalue::Variable(v_stack)),
            ],
            ..Default::default()
        };
        // Construct a target FrameState whose mergeable projection
        // demands all three cells via `Variable` placeholders (locals
        // slot is Some, stack cells are Hlvalue::Variable).
        let v_target_stack_0 = Variable::new();
        let v_target_stack_1 = Variable::new();
        let target_state = FrameState {
            entries: vec![Some(v_local.clone())],
            stack: vec![
                StackElem::Value(Hlvalue::Variable(v_target_stack_0)),
                StackElem::Value(Hlvalue::Variable(v_target_stack_1)),
            ],
            ..Default::default()
        };
        // SpamBlock(framestate) parity — inputargs derive from
        // target_state.getvariables() so the `simplify.py:513` invariant
        // `len(link.args) == len(target.inputargs)` is satisfied by
        // construction.
        let merge = graph.create_block_from_framestate(&target_state);

        graph.set_goto_from_framestate(pred, merge, &pred_state, &target_state);
        let pred_meta = graph.block(pred);
        assert_eq!(pred_meta.exits.len(), 1, "single-exit Link expected");
        let link = &pred_meta.exits[0];
        assert_eq!(link.target, merge);
        assert_eq!(link.args.len(), 3, "locals + stack const + stack variable");
        assert!(
            matches!(&link.args[0], LinkArg::Value(v) if v.id() == v_local.id()),
            "locals projection must carry the local Variable identity",
        );
        assert!(
            matches!(&link.args[1], LinkArg::Const(c) if matches!(c.value, ConstValue::Int(42))),
            "stack Constant must route to LinkArg::Const directly",
        );
        assert!(
            matches!(&link.args[2], LinkArg::Value(_)),
            "stack Variable must route to LinkArg::Value",
        );
    }

    /// `flowspace/model.py Link.__init__` asserts
    /// `len(args) == len(target.inputargs)` at link construction.  The
    /// pyre port enforces the same predicate inside
    /// `set_goto_from_framestate` so a framestate-driven recloseblock
    /// catches a mergeable-shape mismatch before the link is wired.
    #[test]
    #[should_panic(expected = "outputargs.len() (1) != target.inputargs.len() (0)")]
    fn set_goto_from_framestate_panics_on_inputarg_length_mismatch() {
        use crate::flowspace::model::Variable;
        let mut graph = FunctionGraph::new("inputarg_len_assert");
        let pred = graph.create_block();
        // Target block created without inputargs — empty inputargs vec.
        let target = graph.create_block();
        let v_local = Variable::new();
        let pred_state = FrameState {
            entries: vec![Some(v_local.clone())],
            ..Default::default()
        };
        // pred_state.mergeable Variable cells: 1 locals.
        // target_state.mergeable Variable cells: 1 locals.
        // getoutputargs returns 1 LinkArg, but target.inputargs is empty.
        let target_state = FrameState {
            entries: vec![Some(v_local.clone())],
            ..Default::default()
        };
        graph.set_goto_from_framestate(pred, target, &pred_state, &target_state);
    }

    /// `ensure_variable_at_block` short-circuits when the Variable is
    /// already in `block.inputargs` — no graph mutation, returns true.
    #[test]
    fn ensure_variable_at_block_idempotent_on_existing_inputarg() {
        use crate::flowspace::model::Variable;
        let mut graph = FunctionGraph::new("ensure_var_idempotent_input");
        let block = graph.create_block();
        let v = Variable::new();
        graph.block_mut(block).inputargs.push(v.clone());

        let inputargs_before = graph.block(block).inputargs.len();
        let ok = graph.ensure_variable_at_block(block, &v);
        assert!(ok, "Variable already in inputargs reports success");
        assert_eq!(
            graph.block(block).inputargs.len(),
            inputargs_before,
            "no duplicate inputarg insertion",
        );
    }

    /// `ensure_variable_at_block` short-circuits when the Variable is
    /// already the result of an op in `block` — definition-by-op-result.
    #[test]
    fn ensure_variable_at_block_idempotent_on_existing_op_result() {
        let mut graph = FunctionGraph::new("ensure_var_idempotent_op");
        let block = graph.create_block();
        let _v = graph.push_op_var(
            block,
            OpKind::Input {
                name: "x".to_string(),
                ty: ValueType::Unknown,
                class_root: None,
            },
            true,
        );
        let var = graph.block(block).operations[0]
            .result
            .as_ref()
            .unwrap()
            .clone();

        let inputargs_before = graph.block(block).inputargs.len();
        let ok = graph.ensure_variable_at_block(block, &var);
        assert!(ok, "op-result definition reports success");
        assert_eq!(
            graph.block(block).inputargs.len(),
            inputargs_before,
            "no inputarg insertion when op result already defines var",
        );
    }

    /// `ensure_variable_at_block` returns false when `block` has no
    /// predecessors AND `var` is not defined locally — the malformed
    /// graph case callers should treat as a bug.
    #[test]
    fn ensure_variable_at_block_returns_false_on_orphan_block_without_definition() {
        use crate::flowspace::model::Variable;
        let mut graph = FunctionGraph::new("ensure_var_orphan");
        let orphan = graph.create_block();
        let v = Variable::new();
        let ok = graph.ensure_variable_at_block(orphan, &v);
        assert!(!ok, "no predecessors + not defined => fails (no threading)");
        assert!(
            graph.block(orphan).inputargs.is_empty(),
            "failure path performs no mutation",
        );
    }

    /// `ensure_variable_at_block` threads a Variable defined in a
    /// predecessor through one intermediate block: appends to the
    /// predecessor's exit args + adds to the block's inputargs.
    /// Mirrors the carry-through case in `flowspace/flowcontext.py
    /// setstate(block.framestate)` at block entry.
    #[test]
    fn ensure_variable_at_block_threads_one_level_through_predecessor() {
        use crate::flowspace::model::Variable;
        let mut graph = FunctionGraph::new("ensure_var_one_level");
        let pred = graph.create_block();
        let target = graph.create_block();
        let v = Variable::new();
        // Define v in pred via an Input op result.
        let _ = graph.push_op_var(
            pred,
            OpKind::Input {
                name: "p".to_string(),
                ty: ValueType::Unknown,
                class_root: None,
            },
            true,
        );
        // Override the op result Variable to `v` so the definition
        // predicate matches.
        graph.block_mut(pred).operations[0].result = Some(v.clone());
        graph.set_goto(pred, target, vec![]);

        let ok = graph.ensure_variable_at_block(target, &v);
        assert!(ok, "predecessor defines v => threading succeeds");
        assert_eq!(
            graph.block(target).inputargs.len(),
            1,
            "target gains v as inputarg",
        );
        assert!(graph.block(target).inputargs.contains(&v));
        assert_eq!(
            graph.block(pred).exits[0].args.len(),
            1,
            "pred exit args grow by one entry",
        );
        assert!(
            matches!(&graph.block(pred).exits[0].args[0], LinkArg::Value(var) if var.id() == v.id()),
            "pred exit args carry the same Variable identity",
        );
    }

    /// `ensure_variable_at_block` recurses through two levels: the
    /// intermediate block also gains the inputarg and the deepest
    /// predecessor's exit args grow.
    #[test]
    fn ensure_variable_at_block_threads_two_levels_through_predecessor_chain() {
        use crate::flowspace::model::Variable;
        let mut graph = FunctionGraph::new("ensure_var_two_level");
        let root = graph.create_block();
        let mid = graph.create_block();
        let leaf = graph.create_block();
        let v = Variable::new();
        let _ = graph.push_op_var(
            root,
            OpKind::Input {
                name: "p".to_string(),
                ty: ValueType::Unknown,
                class_root: None,
            },
            true,
        );
        graph.block_mut(root).operations[0].result = Some(v.clone());
        graph.set_goto(root, mid, vec![]);
        graph.set_goto(mid, leaf, vec![]);

        let ok = graph.ensure_variable_at_block(leaf, &v);
        assert!(ok, "two-level chain threading succeeds");
        assert!(
            graph.block(leaf).inputargs.contains(&v),
            "leaf gains v inputarg",
        );
        assert!(
            graph.block(mid).inputargs.contains(&v),
            "mid gains v inputarg",
        );
        assert_eq!(
            graph.block(root).exits[0].args.len(),
            1,
            "root exit args grow",
        );
        assert_eq!(
            graph.block(mid).exits[0].args.len(),
            1,
            "mid exit args grow",
        );
    }

    /// Cycle-safe: a back-edge predecessor finds the in-progress
    /// inputarg slot via `variable_defined_in_block` and short-circuits.
    /// Loop shape: `header → body_tail → header`.
    #[test]
    fn ensure_variable_at_block_handles_back_edge_without_infinite_recursion() {
        use crate::flowspace::model::Variable;
        let mut graph = FunctionGraph::new("ensure_var_back_edge");
        let entry = graph.create_block();
        let header = graph.create_block();
        let body_tail = graph.create_block();
        let v = Variable::new();
        let _ = graph.push_op_var(
            entry,
            OpKind::Input {
                name: "p".to_string(),
                ty: ValueType::Unknown,
                class_root: None,
            },
            true,
        );
        graph.block_mut(entry).operations[0].result = Some(v.clone());
        graph.set_goto(entry, header, vec![]);
        graph.set_goto(header, body_tail, vec![]);
        graph.set_goto(body_tail, header, vec![]);

        let ok = graph.ensure_variable_at_block(header, &v);
        assert!(ok, "back-edge cycle is handled");
        assert!(
            graph.block(header).inputargs.contains(&v),
            "header inputarg seeded",
        );
        assert!(
            graph.block(body_tail).inputargs.contains(&v),
            "body_tail inputarg also seeded — back-edge predecessor demanded v",
        );
        assert_eq!(
            graph.block(entry).exits[0].args.len(),
            1,
            "entry → header link args extended",
        );
        assert_eq!(
            graph.block(body_tail).exits[0].args.len(),
            1,
            "body_tail → header link args extended",
        );
        assert_eq!(
            graph.block(header).exits[0].args.len(),
            1,
            "header → body_tail link args extended (body_tail demands v)",
        );
    }

    #[test]
    fn thread_undefined_op_operand_across_diamond_when_definition_dominates_split() {
        let mut graph = FunctionGraph::new("thread_operand_diamond");
        let entry = graph.startblock;
        let left = graph.create_block();
        let right = graph.create_block();
        let join = graph.create_block();
        let value = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "value".into(),
                    ty: ValueType::Float,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        let cond = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "cond".into(),
                    ty: ValueType::Bool,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.set_branch(entry, cond, left, vec![], right, vec![]);
        graph.set_goto(left, join, vec![]);
        graph.set_goto(right, join, vec![]);
        let base = graph
            .push_op_var(
                join,
                OpKind::Call {
                    target: CallTarget::synthetic_transparent_ctor("Box"),
                    args: crate::model::call_args(vec![]),
                    result_ty: ValueType::Ref(Some("Box".into())),
                },
                true,
            )
            .unwrap();
        graph.block_mut(join).operations.push(SpaceOperation {
            result: None,
            kind: OpKind::FieldWrite {
                base,
                field: FieldDescriptor {
                    name: "payload".into(),
                    owner_root: Some("Box".into()),
                    owner_id: None,
                    base_is_deref: None,
                    taken_by_address: false,
                    inline_vec: false,
                    vec_part: None,
                    owner_declared_gc: None,
                    host_index: None,
                    scalar_word: None,
                },
                value: LinkArg::Value(value.clone()),
                ty: ValueType::Float,
            },
        });

        thread_undefined_op_operands(&mut graph);

        for block in [left, right, join] {
            assert!(
                graph.block(block).inputargs.contains(&value),
                "dominating value is threaded through {block:?}",
            );
        }
        assert!(graph.block(entry).exits.iter().all(|link| {
            link.args
                .iter()
                .any(|arg| matches!(arg, LinkArg::Value(v) if v == &value))
        }));
    }

    #[test]
    fn thread_undefined_op_operand_declines_value_defined_in_one_diamond_arm() {
        let mut graph = FunctionGraph::new("thread_operand_partial_diamond");
        let entry = graph.startblock;
        let left = graph.create_block();
        let right = graph.create_block();
        let join = graph.create_block();
        let cond = graph
            .push_op_var(
                entry,
                OpKind::Input {
                    name: "cond".into(),
                    ty: ValueType::Bool,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.set_branch(entry, cond, left, vec![], right, vec![]);
        let value = graph
            .push_op_var(
                left,
                OpKind::Input {
                    name: "left_only".into(),
                    ty: ValueType::Float,
                    class_root: None,
                },
                true,
            )
            .unwrap();
        graph.set_goto(left, join, vec![]);
        graph.set_goto(right, join, vec![]);
        let result = graph.alloc_value_var();
        graph.block_mut(join).operations.push(SpaceOperation {
            result: Some(result),
            kind: OpKind::UnaryOp {
                op: "same_as".into(),
                operand: value.clone(),
                result_ty: ValueType::Float,
            },
        });

        thread_undefined_op_operands(&mut graph);

        assert!(
            !graph.block(join).inputargs.contains(&value),
            "one-arm definition must not be invented at the join",
        );
        assert!(graph.block(right).inputargs.is_empty());
    }

    /// `framestate.py getvariables` walks `locals + flatten(stack) +
    /// [exc_type, exc_value]` in order and filters Variable cells.  The
    /// Pyre port emits Variables from each projection in the same
    /// positional order so `block.inputargs` line up with
    /// `getoutputargs` cells.
    #[test]
    fn getvariables_walks_mergeable_in_locals_stack_exc_order() {
        use crate::flowspace::framestate::StackElem;
        use crate::flowspace::model::{Constant, FSException, Hlvalue, Variable};

        let graph = FunctionGraph::new("getvariables_shape");
        let v_local = Variable::new();
        let v_stack = Variable::new();
        let v_exc_type = Variable::new();
        let v_exc_value = Variable::new();

        let fs = FrameState {
            entries: vec![Some(v_local.clone()), None],
            stack: vec![
                StackElem::Value(Hlvalue::Constant(Constant::new(ConstValue::Int(7)))),
                StackElem::Value(Hlvalue::Variable(v_stack.clone())),
            ],
            last_exception: Some(FSException::new(
                Hlvalue::Variable(v_exc_type.clone()),
                Hlvalue::Variable(v_exc_value.clone()),
            )),
            ..Default::default()
        };
        let vars = fs.getvariables(&graph);
        assert_eq!(
            vars.len(),
            4,
            "1 locals + 1 stack Variable + 2 exc Variables"
        );
        assert_eq!(vars[0].id(), v_local.id(), "locals slot first");
        assert_eq!(
            vars[1].id(),
            v_stack.id(),
            "stack Variable second (Constant filtered)"
        );
        assert_eq!(vars[2].id(), v_exc_type.id(), "exc_type third");
        assert_eq!(vars[3].id(), v_exc_value.id(), "exc_value fourth");
    }

    /// `create_block_from_framestate` materialises the merge block
    /// shape that `flowcontext.py SpamBlock(framestate)` builds —
    /// `block.inputargs = framestate.getvariables()`, attached
    /// `block.framestate = framestate`.  Inputarg Variables carry the
    /// framestate's Variable identities directly.
    #[test]
    fn create_block_from_framestate_sets_inputargs_and_attaches_state() {
        use crate::flowspace::framestate::StackElem;
        use crate::flowspace::model::{Hlvalue, Variable};

        let mut graph = FunctionGraph::new("spamblock_shape");
        let v_local = Variable::new();
        let v_stack = Variable::new();

        let fs = FrameState {
            entries: vec![Some(v_local.clone())],
            stack: vec![StackElem::Value(Hlvalue::Variable(v_stack.clone()))],
            ..Default::default()
        };
        let merge = graph.create_block_from_framestate(&fs);
        let block = graph.block(merge);
        assert_eq!(
            block.inputargs.len(),
            2,
            "1 locals + 1 stack Variable (no exception)"
        );
        assert_eq!(block.inputargs[0].id(), v_local.id());
        assert_eq!(block.inputargs[1].id(), v_stack.id());
        assert!(
            block.framestate.is_some(),
            "block.framestate stamped (SpamBlock parity)"
        );
        assert_eq!(&block.inputargs[0], &v_local);
        assert_eq!(&block.inputargs[1], &v_stack);
    }

    /// The critical round-trip invariant: predecessor's
    /// `getoutputargs(target=merged_state)` length matches
    /// `create_block_from_framestate(merged_state).inputargs.len()` —
    /// the simplify.py:513 invariant `len(link.args) ==
    /// len(target.inputargs)`.  Demonstrated against a non-trivial
    /// mergeable shape (locals + stack Variable + stack Constant +
    /// exception cells) so the locals/stack/exc walk all participate.
    #[test]
    fn outputargs_and_inputargs_round_trip_for_full_mergeable_shape() {
        use crate::flowspace::framestate::StackElem;
        use crate::flowspace::model::{Constant, FSException, Hlvalue, Variable};

        let mut graph = FunctionGraph::new("mergeable_round_trip");
        let v_a = Variable::new();
        let v_b = Variable::new();

        // Two predecessors disagreeing on stack slot 0 (Variable phi)
        // and on exception (Variable phi); locals agree on v_a.
        let v_t1 = Variable::new();
        let v_t2 = Variable::new();
        let v_v1 = Variable::new();
        let v_v2 = Variable::new();
        let pred_a = FrameState {
            entries: vec![Some(v_a.clone())],
            stack: vec![
                StackElem::Value(Hlvalue::Variable(v_b.clone())),
                StackElem::Value(Hlvalue::Constant(Constant::new(ConstValue::Int(1)))),
            ],
            last_exception: Some(FSException::new(
                Hlvalue::Variable(v_t1),
                Hlvalue::Variable(v_v1),
            )),
            ..Default::default()
        };
        let pred_b = FrameState {
            entries: vec![Some(v_a.clone())],
            stack: vec![
                StackElem::Value(Hlvalue::Variable(v_b.clone())),
                StackElem::Value(Hlvalue::Constant(Constant::new(ConstValue::Int(1)))),
            ],
            last_exception: Some(FSException::new(
                Hlvalue::Variable(v_t2),
                Hlvalue::Variable(v_v2),
            )),
            ..Default::default()
        };
        let merged = pred_a
            .union(&pred_b, &mut graph)
            .expect("union must succeed");
        // Construct the merge block from the merged framestate; then
        // ask each predecessor for its outputargs against the same
        // target.  The shapes must line up positionally — the
        // simplify.py:513 invariant.
        let merge_block_id = graph.create_block_from_framestate(&merged);
        let merge_block = graph.block(merge_block_id);
        let outputargs_a = pred_a.getoutputargs(&merged, &graph);
        let outputargs_b = pred_b.getoutputargs(&merged, &graph);
        assert_eq!(
            outputargs_a.len(),
            merge_block.inputargs.len(),
            "pred_a outputargs length matches merge block inputargs",
        );
        assert_eq!(
            outputargs_b.len(),
            merge_block.inputargs.len(),
            "pred_b outputargs length matches merge block inputargs",
        );
    }

    /// `try_getoutputargs` returns `None` (rather than panicking like
    /// `getoutputargs`) when the target binds a `Variable` at a locals
    /// slot the predecessor leaves `None` — the phantom-slot mismatch the
    /// cyclic framestate path's loop-header pre-seed can produce when a
    /// live-in phi is scrubbed undefined on one edge.  The all-bound case
    /// still returns `Some`.
    #[test]
    fn try_getoutputargs_declines_phantom_slot_returns_some_when_bound() {
        use crate::flowspace::model::Variable;

        let graph = FunctionGraph::new("try_getoutputargs_phantom");
        // Target binds a Variable phi at slot 0 (a loop-header live-in).
        let target = FrameState {
            entries: vec![Some(Variable::new())],
            ..Default::default()
        };
        // Predecessor whose slot 0 was scrubbed to `None` (undefined on
        // this edge): the target Variable slot has no self cell to thread.
        let pred_undefined = FrameState {
            entries: vec![None],
            ..Default::default()
        };
        assert!(
            pred_undefined.try_getoutputargs(&target, &graph).is_none(),
            "phantom slot 0 undefined in predecessor declines to None",
        );
        // Predecessor that does bind slot 0 threads it through.
        let v_self = Variable::new();
        let pred_bound = FrameState {
            entries: vec![Some(v_self.clone())],
            ..Default::default()
        };
        let out = pred_bound
            .try_getoutputargs(&target, &graph)
            .expect("bound slot threads through");
        assert_eq!(out, vec![LinkArg::Value(v_self)]);
    }

    // ── annotator-monomorphization — CallTarget::Method ──
    // transient ClassDefKey transport invariants.

    #[test]
    fn call_target_method_resolved_path_is_skipped_by_serde() {
        use crate::parse::CallPath;
        let path = CallPath::for_impl_method("PyFrame", "foo");
        let t = CallTarget::Method {
            name: "foo".into(),
            receiver_root: Some("Bar".into()),
            resolved_path: Some(path),
            fun_decl_id: None,
            branch_payloads: None,
        };
        let json = serde_json::to_string(&t).expect("encode");
        assert!(
            !json.contains("resolved_path"),
            "serialized form leaks resolved_path: {json}"
        );
        let round_trip: CallTarget = serde_json::from_str(&json).expect("decode");
        assert_eq!(round_trip.resolved_path(), None);
    }

    #[test]
    fn push_niche_null_without_cast_emits_only_null_mut() {
        let mut graph = FunctionGraph::new("push_niche_null");
        let block = graph.startblock;
        let var = graph.push_niche_null(block, None);
        let ops = &graph.block(block).operations;
        assert_eq!(ops.len(), 1, "a cast-less niche null is only null_mut");
        match &ops[0] {
            SpaceOperation {
                result: Some(result),
                kind:
                    OpKind::Call {
                        target,
                        args,
                        result_ty,
                    },
            } => {
                assert_eq!(result, &var);
                assert_eq!(target.to_string(), "core::ptr::null_mut");
                assert!(args.is_empty());
                assert_eq!(result_ty, &ValueType::Ref(None));
            }
            other => panic!("expected null_mut call, got {other:?}"),
        }
    }
}
